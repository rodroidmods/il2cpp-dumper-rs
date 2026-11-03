//! Live-Resolution token resolver + per-image recovery application.
//!
//! Bridge between the analysis-side recovery pipeline (`crate::recovery`)
//! and dotnetdll's Resolution-bound token model:
//!
//! - same-image call/field targets → Definition tokens (MethodIndex)
//! - cross-image targets → MemberRefs against an external type reference
//! - `ManagedType` → `MethodType` conversion pushes external refs as needed

use crate::recovery::emit::TokenResolver;
use crate::recovery::types::{FieldRef, ManagedType, MethodRef, NamedType, PrimKind};
use dotnetdll::prelude::*;
use dotnetdll::resolved::signature::{
    CallingConvention, ManagedMethod, Parameter, ParameterType, ReturnType,
};
use dotnetdll::resolved::types::{
    BaseType, ExternalTypeReference, ResolutionScope, TypeSource, ValueKind,
};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

pub struct ResolutionBackedResolver<'r, 'res, 'a> {
    pub resolution: &'r RefCell<&'res mut Resolution<'a>>,
    pub img_idx: usize,
    /// global type-def index → dotnet type index in this image's resolution
    pub type_map: HashMap<usize, usize>,
    pub assembly_refs: RefCell<HashMap<String, AssemblyRefIndex>>,
    pub type_ref_cache: RefCell<HashMap<(String, String, String), TypeRefIndex>>,
    pub console_write_line_cache: RefCell<Option<MethodRefIndex>>,
}

impl<'r, 'res, 'a> ResolutionBackedResolver<'r, 'res, 'a> {
    fn ensure_assembly(&self, name: &str) -> AssemblyRefIndex {
        if let Some(existing) = self.assembly_refs.borrow().get(name) {
            return *existing;
        }
        let idx =
            self.resolution
                .borrow_mut()
                .push_assembly_reference(ExternalAssemblyReference::new(Cow::<str>::Owned(
                    name.to_string(),
                )));
        self.assembly_refs
            .borrow_mut()
            .insert(name.to_string(), idx);
        idx
    }

    fn push_type_ref(
        &self,
        ns: Option<String>,
        name: String,
        scope_assembly: &str,
    ) -> TypeRefIndex {
        let scope = ResolutionScope::Assembly(self.ensure_assembly(scope_assembly));
        let etr = ExternalTypeReference::new(ns.map(Cow::Owned), name, scope);
        self.resolution.borrow_mut().push_type_reference(etr)
    }

    fn type_ref_for(&self, nt: &NamedType) -> Option<TypeRefIndex> {
        let key = (
            nt.namespace.clone(),
            nt.name.clone(),
            nt.assembly.clone().unwrap_or_else(|| "mscorlib".into()),
        );
        if let Some(idx) = self.type_ref_cache.borrow().get(&key) {
            return Some(*idx);
        }
        let asm = key.2.clone();
        let idx = self.push_type_ref(
            if nt.namespace.is_empty() {
                None
            } else {
                Some(nt.namespace.clone())
            },
            nt.name.clone(),
            &asm,
        );
        self.type_ref_cache.borrow_mut().insert(key, idx);
        Some(idx)
    }

    fn class_type_of_named(&self, nt: &NamedType) -> Option<MethodType> {
        let tref = self.type_ref_for(nt)?;
        Some(MethodType::Base(Box::new(BaseType::Type {
            value_kind: Some(ValueKind::Class),
            source: TypeSource::User(UserType::Reference(tref)),
        })))
    }

    /// Convert analysis ManagedType → dotnetdll MethodType with real externals.
    pub fn managed_to_method_type(&self, t: &ManagedType) -> Option<MethodType> {
        let base = match t {
            ManagedType::Primitive(p) => match p {
                PrimKind::Void => return None,
                PrimKind::Boolean => BaseType::Boolean,
                PrimKind::Char => BaseType::Char,
                PrimKind::I1 => BaseType::Int8,
                PrimKind::U1 => BaseType::UInt8,
                PrimKind::I2 => BaseType::Int16,
                PrimKind::U2 => BaseType::UInt16,
                PrimKind::I4 => BaseType::Int32,
                PrimKind::U4 => BaseType::UInt32,
                PrimKind::I8 => BaseType::Int64,
                PrimKind::U8 => BaseType::UInt64,
                PrimKind::R4 => BaseType::Float32,
                PrimKind::R8 => BaseType::Float64,
                PrimKind::I | PrimKind::U => BaseType::IntPtr,
            },
            ManagedType::Object | ManagedType::Opaque(_) => BaseType::Object,
            ManagedType::String => BaseType::String,
            ManagedType::IntPtr => BaseType::IntPtr,
            ManagedType::ByRef(_) | ManagedType::Pointer(_) => BaseType::IntPtr,
            ManagedType::SzArray(elem) => {
                let inner = self.managed_to_method_type(elem)?;
                return Some(MethodType::Base(Box::new(BaseType::vector(inner))));
            }
            ManagedType::Class(nt) => return self.class_type_of_named(nt),
            ManagedType::Unknown => return None,
        };
        Some(MethodType::Base(Box::new(base)))
    }

    fn same_image_method_index(&self, m: &MethodRef) -> Option<MethodIndex> {
        if m.image_index != self.img_idx || m.type_def_index < 0 {
            return None;
        }
        let dotnet_ty = *self.type_map.get(&(m.type_def_index as usize))?;
        let ti = (**self.resolution.borrow()).type_definition_index(dotnet_ty)?;
        self.resolution
            .borrow()
            .method_index(ti, m.method_index as usize)
    }

    fn signature_of(&self, m: &MethodRef) -> ManagedMethod<MethodType> {
        let params: Vec<Parameter<MethodType>> = m
            .sig
            .params
            .iter()
            .map(|p| {
                Parameter(
                    vec![],
                    ParameterType::Value(self.managed_to_method_type(p).unwrap_or_else(object_mt)),
                )
            })
            .collect();
        let ret = if matches!(m.sig.ret, ManagedType::Primitive(PrimKind::Void)) {
            ReturnType::VOID
        } else {
            ReturnType(
                vec![],
                Some(ParameterType::Value(
                    self.managed_to_method_type(&m.sig.ret)
                        .unwrap_or_else(object_mt),
                )),
            )
        };
        ManagedMethod::<MethodType> {
            instance: !m.sig.is_static,
            explicit_this: false,
            calling_convention: CallingConvention::Default,
            parameters: params,
            return_type: ret,
            varargs: None,
        }
    }

    fn external_method_source(&self, m: &MethodRef) -> Option<MethodSource> {
        let parent_ty = self.class_type_of_named(&NamedType {
            namespace: m.declaring_namespace.clone(),
            name: m.declaring_type_name.clone(),
            assembly: Some(m.assembly.clone()),
            type_def_index: m.type_def_index,
        })?;
        let emr = ExternalMethodReference::new(
            MethodReferenceParent::Type(parent_ty),
            m.name.clone(),
            self.signature_of(m),
        );
        let idx = self.resolution.borrow_mut().push_method_reference(emr);
        Some(MethodSource::User(UserMethod::Reference(idx)))
    }

    /// Apply recovered bodies to every method slot in this image.
    /// Returns (recovered, failed).
    pub fn apply_recovery(
        &mut self,
        ctx: &crate::recovery::RecoveryContext,
        image_name: &str,
        metadata: &crate::il2cpp::metadata::Metadata,
        il2cpp: &crate::il2cpp::base::Il2Cpp,
        method_slots: &HashMap<usize, (usize, usize, u64)>,
        method_owner_tdi: &HashMap<usize, usize>,
    ) -> (usize, usize) {
        if !ctx.assembly_allowed(image_name) {
            return (0, 0);
        }
        let sorted = il2cpp.build_sorted_method_addresses();
        let mut recovered = 0usize;
        let mut failed_no_rva = 0usize;
        let mut failed_size = 0usize;
        let mut failed_bytes = 0usize;
        let mut failed_analysis = 0usize;

        for (mi_global, &(dotnet_ty, method_pos, rva)) in method_slots.iter() {
            if rva == 0 {
                failed_no_rva += 1;
                continue;
            }
            let Some(method_def) = metadata.method_defs.get(*mi_global) else {
                continue;
            };
            let is_static = (method_def.flags & 0x0010) != 0;
            let ret = il2cpp
                .types
                .get(method_def.return_type as usize)
                .map(|t| crate::recovery::managed_type_of(t, metadata, il2cpp, 0))
                .unwrap_or(ManagedType::Unknown);
            let params: Vec<ManagedType> = (0..method_def.parameter_count as usize)
                .filter_map(|j| {
                    let pidx = method_def.parameter_start as usize + j;
                    let pd = metadata.parameter_defs.get(pidx)?;
                    il2cpp
                        .types
                        .get(pd.type_index as usize)
                        .map(|t| crate::recovery::managed_type_of(t, metadata, il2cpp, 0))
                })
                .collect();
            let declaring = metadata
                .type_defs
                .get(*mi_global)
                .and_then(|_| {
                    // method_slots keyed by global mi; find owning tdi from map
                    method_slots.get(mi_global).map(|_| ())
                })
                .and_then(|()| {
                    let tdi = *method_owner_tdi.get(mi_global)?;
                    let td = metadata.type_defs.get(tdi)?;
                    Some(std::sync::Arc::new(crate::recovery::types::NamedType {
                        namespace: metadata
                            .get_string_from_index(td.namespace_index)
                            .unwrap_or_default(),
                        name: metadata
                            .get_string_from_index(td.name_index)
                            .unwrap_or_default(),
                        assembly: None,
                        type_def_index: tdi as i32,
                    }))
                });
            // HFA float aggregates (Vector3, Quaternion, ...) travel in
            // consecutive FP argument registers; the comp names let locals
            // bind those registers and render `rootPosition.x` not `v26`.
            let param_comps: Vec<Vec<String>> = params
                .iter()
                .map(|pt| match pt {
                    crate::recovery::types::ManagedType::Class(nt)
                        if nt.type_def_index >= 0
                            && nt.type_def_index < metadata.type_defs.len() as i32 =>
                    {
                        crate::recovery::hfa_float_components(nt.type_def_index, metadata, il2cpp)
                    }
                    _ => Vec::new(),
                })
                .collect();
            let ret_hidden_buffer = ctx.is_arm32
                && crate::recovery::arm32_ret_uses_hidden_buffer(&ret, metadata, il2cpp);
            // Extract parameter display names from metadata so the local table
            // can render named parameters instead of positional arg0/arg1.
            let param_names: Vec<String> = (0..method_def.parameter_count as usize)
                .filter_map(|j| {
                    let pidx = method_def.parameter_start as usize + j;
                    let pd = metadata.parameter_defs.get(pidx)?;
                    Some(
                        metadata
                            .get_string_from_index(pd.name_index)
                            .unwrap_or_else(|_| format!("arg{j}")),
                    )
                })
                .collect();
            let seed = crate::recovery::locals::SeedInfo {
                param_types: params.clone(),
                ret_type: ret,
                is_static,
                declaring,
                param_names,
                param_comp_names: param_comps,
                arm32_param_words: params
                    .iter()
                    .map(|ty| match ty {
                        crate::recovery::types::ManagedType::Class(nt) => {
                            crate::recovery::arm32_value_type_size(
                                nt.type_def_index,
                                metadata,
                                il2cpp,
                            )
                            .map(|size| size.saturating_add(3) / 4)
                            .unwrap_or(1)
                        }
                        _ => 1,
                    })
                    .collect(),
                is_arm32: ctx.is_arm32,
                ret_hidden_buffer,
            };

            let size = il2cpp.get_method_body_size(rva, &sorted);
            if size == 0 || size > ctx.max_method_bytes {
                failed_size += 1;
                continue;
            }
            let Some(bytes) = il2cpp.read_bytes_at_rva(rva, size) else {
                failed_bytes += 1;
                continue;
            };

            crate::recovery::verify::set_tv_source("dll");
            match ctx.recover_method(&bytes, rva, &seed, self, metadata, il2cpp) {
                Some(out) => {
                    let Some(ti) = (**self.resolution.borrow()).type_definition_index(dotnet_ty)
                    else {
                        failed_analysis += 1;
                        continue;
                    };
                    let res = &mut **self.resolution.borrow_mut();
                    if method_pos >= res[ti].methods.len() {
                        failed_analysis += 1;
                        continue;
                    }
                    res[ti].methods[method_pos].body = Some(dotnetdll::resolved::body::Method {
                        header: dotnetdll::resolved::body::Header {
                            initialize_locals: true,
                            maximum_stack_size: out.max_stack,
                            local_variables: out.locals,
                        },
                        instructions: out.instructions,
                        data_sections: vec![],
                    });
                    recovered += 1;
                }
                None => {
                    failed_analysis += 1;
                }
            }
        }
        // breakdown available via debug logging when needed
        let _ = (failed_no_rva, failed_size, failed_bytes);
        failed_analysis += failed_size + failed_bytes;
        if std::env::var("ILR_TIMING").is_ok() {
            eprintln!("{}", crate::recovery::timing::report());
        }
        (recovered, failed_analysis)
    }
}

impl<'r, 'res, 'a> TokenResolver for ResolutionBackedResolver<'r, 'res, 'a> {
    fn user_method(&self, m: &MethodRef) -> Option<UserMethod> {
        if let Some(mi) = self.same_image_method_index(m) {
            return Some(UserMethod::Definition(mi));
        }
        match self.external_method_source(m) {
            Some(MethodSource::User(um)) => Some(um),
            _ => None,
        }
    }

    fn field_source(&self, f: &FieldRef) -> Option<FieldSource> {
        // external refs uniformly — valid IL either way; same-image definition
        // lookup is unreliable post-hoc because field positions include
        // static/const ordering differences.
        let parent_ty = MethodType::Base(Box::new(BaseType::Type {
            value_kind: Some(ValueKind::Class),
            source: TypeSource::User(UserType::Reference(self.type_ref_for(&NamedType {
                namespace: String::new(),
                name: f.declaring_type_name.clone(),
                assembly: None,
                type_def_index: f.type_def_index,
            })?)),
        }));
        let fty = self
            .managed_to_method_type(&f.field_type)
            .unwrap_or_else(object_mt);
        let efr = ExternalFieldReference::new(
            FieldReferenceParent::Type(parent_ty),
            member_from_method(fty),
            Cow::Owned(f.name.clone()),
        );
        let idx = self.resolution.borrow_mut().push_field_reference(efr);
        Some(FieldSource::Reference(idx))
    }

    fn method_type(&self, t: &ManagedType) -> Option<MethodType> {
        self.managed_to_method_type(t)
    }

    fn user_method_for_ctor(&self, t: &NamedType) -> Option<UserMethod> {
        if t.type_def_index >= 0 {
            if let Some(&dotnet_ty) = self.type_map.get(&(t.type_def_index as usize)) {
                if let Some(ti) = (**self.resolution.borrow()).type_definition_index(dotnet_ty) {
                    let found = {
                        let res = self.resolution.borrow();
                        res.enumerate_methods(ti)
                            .find(|(_, m)| {
                                m.name.as_ref() == ".ctor" && m.signature.parameters.is_empty()
                            })
                            .map(|(mi, _)| mi)
                    };
                    if let Some(mi) = found {
                        return Some(UserMethod::Definition(mi));
                    }
                }
            }
        }
        None
    }

    fn console_write_line(&self) -> Option<MethodSource> {
        if let Some(cached) = *self.console_write_line_cache.borrow() {
            return Some(MethodSource::User(UserMethod::Reference(cached)));
        }
        let console = self.class_type_of_named(&NamedType {
            namespace: "System".into(),
            name: "Console".into(),
            assembly: Some("mscorlib".into()),
            type_def_index: -1,
        })?;
        let sig = ManagedMethod::<MethodType> {
            instance: false,
            explicit_this: false,
            calling_convention: CallingConvention::Default,
            parameters: vec![Parameter(
                vec![],
                ParameterType::Value(MethodType::Base(Box::new(BaseType::String))),
            )],
            return_type: ReturnType::VOID,
            varargs: None,
        };
        let emr =
            ExternalMethodReference::new(MethodReferenceParent::Type(console), "WriteLine", sig);
        let idx = self.resolution.borrow_mut().push_method_reference(emr);
        *self.console_write_line_cache.borrow_mut() = Some(idx);
        Some(MethodSource::User(UserMethod::Reference(idx)))
    }
}

// helpers ----------------------------------------------------------------------

fn object_mt() -> MethodType {
    MethodType::Base(Box::new(BaseType::Object))
}

fn member_from_method(mt: MethodType) -> MemberType {
    match mt {
        MethodType::Base(b) => MemberType::Base(Box::new(b.map(member_from_method))),
        MethodType::TypeGeneric(i) => MemberType::TypeGeneric(i),
        MethodType::MethodGeneric(_) => MemberType::Base(Box::new(BaseType::Object)),
    }
}
