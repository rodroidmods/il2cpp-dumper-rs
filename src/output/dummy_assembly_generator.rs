use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use dotnetdll::resolved::body;
use dotnetdll::resolved::il::Instruction;
use dotnetdll::resolved::members::{self, Field, Method, Property, Event, ParameterMetadata};
use dotnetdll::resolved::module::Module;
use dotnetdll::resolved::signature::{
    MethodSignature, Parameter, ReturnType,
};
use dotnetdll::resolved::types::{
    BaseType, ExternalTypeReference, MemberType, MethodType,
    TypeDefinition as DotNetTypeDef, TypeFlags, TypeSource,
    ResolutionScope, UserType, ValueKind, Accessibility as TypeAccessibility,
};
use dotnetdll::resolved::assembly::ExternalAssemblyReference;
use dotnetdll::resolved::generic;
use dotnetdll::resolved::attribute::{Attribute, CustomAttributeData, NamedArg, FixedArg};
use dotnetdll::resolution::{Resolution, MethodRefIndex};

use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::base::Il2Cpp;
use crate::il2cpp::enums::Il2CppTypeEnum;
use crate::il2cpp::structures::*;
use crate::executor::Il2CppExecutor;
use crate::config::Config;
use crate::error::Result;

#[allow(dead_code)]
struct AttrCtors {
    address_ctor: MethodRefIndex,
    field_offset_ctor: MethodRefIndex,
    token_ctor: MethodRefIndex,
    attribute_ctor: MethodRefIndex,
    metadata_offset_ctor: MethodRefIndex,
    thread_static_ctor: MethodRefIndex,
    field_rva_ctor: MethodRefIndex,
}

fn make_named_string_attr<'a>(ctor: MethodRefIndex, fields: Vec<(&str, String)>) -> Attribute<'a> {
    let named_args = fields.into_iter().map(|(name, value)| {
        NamedArg::Field(
            Cow::Owned(name.to_string()),
            FixedArg::String(Some(Cow::Owned(value))),
        )
    }).collect();
    Attribute::new(
        members::UserMethod::Reference(ctor),
        CustomAttributeData {
            constructor_args: vec![],
            named_args,
        },
    )
}

fn push_attr_ctor(resolution: &mut Resolution<'_>, asm_ref: dotnetdll::resolution::AssemblyRefIndex, name: String) -> MethodRefIndex {
    let type_ref = resolution.push_type_reference(
        ExternalTypeReference::new(
            None,
            Cow::Owned(name),
            ResolutionScope::Assembly(asm_ref),
        ),
    );
    let parent_type = MethodType::Base(Box::new(BaseType::Type {
        value_kind: Some(ValueKind::Class),
        source: TypeSource::User(UserType::Reference(type_ref)),
    }));
    resolution.push_method_reference(
        members::ExternalMethodReference::new(
            members::MethodReferenceParent::Type(parent_type),
            ".ctor",
            MethodSignature::new(true, ReturnType::VOID, vec![]),
        ),
    )
}

fn setup_il2cpp_dummy_dll_refs(resolution: &mut Resolution<'_>) -> AttrCtors {
    let dummy_asm_ref = resolution.push_assembly_reference(
        ExternalAssemblyReference::new("Il2CppDummyDll"),
    );

    AttrCtors {
        address_ctor: push_attr_ctor(resolution, dummy_asm_ref, "AddressAttribute".to_string()),
        field_offset_ctor: push_attr_ctor(resolution, dummy_asm_ref, "FieldOffsetAttribute".to_string()),
        token_ctor: push_attr_ctor(resolution, dummy_asm_ref, "TokenAttribute".to_string()),
        attribute_ctor: push_attr_ctor(resolution, dummy_asm_ref, "AttributeAttribute".to_string()),
        metadata_offset_ctor: push_attr_ctor(resolution, dummy_asm_ref, "MetadataOffsetAttribute".to_string()),
        thread_static_ctor: push_attr_ctor(resolution, dummy_asm_ref, "ThreadStaticAttribute".to_string()),
        field_rva_ctor: push_attr_ctor(resolution, dummy_asm_ref, "FieldRvaAttribute".to_string()),
    }
}

#[allow(dead_code)]
struct DummyDllContext {
    type_map: HashMap<usize, usize>,
    mscorlib_ref: dotnetdll::resolution::AssemblyRefIndex,
    void_type_ref: dotnetdll::resolution::TypeRefIndex,
    gen_param_total: usize,
}

fn type_accessibility_from_flags(flags: u32) -> TypeAccessibility {
    use dotnetdll::resolved::Accessibility;
    match flags & 0x7 {
        0x0 => TypeAccessibility::NotPublic,
        0x1 => TypeAccessibility::Public,
        0x2 => TypeAccessibility::Nested(Accessibility::Public),
        0x3 => TypeAccessibility::Nested(Accessibility::Private),
        0x4 => TypeAccessibility::Nested(Accessibility::Family),
        0x5 => TypeAccessibility::Nested(Accessibility::Assembly),
        0x6 => TypeAccessibility::Nested(Accessibility::FamilyANDAssembly),
        0x7 => TypeAccessibility::Nested(Accessibility::FamilyORAssembly),
        _ => TypeAccessibility::NotPublic,
    }
}

fn member_accessibility_from_flags(flags: u16) -> dotnetdll::resolved::Accessibility {
    match flags & 0x7 {
        0x1 => dotnetdll::resolved::Accessibility::Private,
        0x2 => dotnetdll::resolved::Accessibility::FamilyANDAssembly,
        0x3 => dotnetdll::resolved::Accessibility::Assembly,
        0x4 => dotnetdll::resolved::Accessibility::Family,
        0x5 => dotnetdll::resolved::Accessibility::FamilyORAssembly,
        0x6 => dotnetdll::resolved::Accessibility::Public,
        _ => dotnetdll::resolved::Accessibility::Private,
    }
}

fn field_accessibility_from_attrs(attrs: u32) -> dotnetdll::resolved::Accessibility {
    match attrs & 0x7 {
        0x1 => dotnetdll::resolved::Accessibility::Private,
        0x2 => dotnetdll::resolved::Accessibility::FamilyANDAssembly,
        0x3 => dotnetdll::resolved::Accessibility::Assembly,
        0x4 => dotnetdll::resolved::Accessibility::Family,
        0x5 => dotnetdll::resolved::Accessibility::FamilyORAssembly,
        0x6 => dotnetdll::resolved::Accessibility::Public,
        _ => dotnetdll::resolved::Accessibility::Private,
    }
}

pub fn generate_dummy_dlls(
    executor: &Il2CppExecutor,
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
    config: &Config,
    output_dir: &str,
) -> Result<()> {
    use rayon::prelude::*;

    let dummy_dir = Path::new(output_dir).join("DummyDll");
    fs::create_dir_all(&dummy_dir)
        .map_err(|e| crate::error::Error::Other(format!("Failed to create DummyDll dir: {e}")))?;

    let image_defs = metadata.image_defs.clone();
    let assembly_defs = metadata.assembly_defs.clone();
    let type_defs_all = metadata.type_defs.clone();
    let field_defs_all = metadata.field_defs.clone();
    let method_defs_all = metadata.method_defs.clone();
    let property_defs_all = metadata.property_defs.clone();
    let event_defs_all = metadata.event_defs.clone();
    let parameter_defs_all = metadata.parameter_defs.clone();
    let nested_type_indices = metadata.nested_type_indices.clone();
    let interface_indices = metadata.interface_indices.clone();
    let generic_containers = metadata.generic_containers.clone();
    let generic_parameters = metadata.generic_parameters.clone();
    let constraint_indices = metadata.constraint_indices.clone();
    let types_all = il2cpp.types.clone();
    let _version = metadata.version;

    let mut global_type_map: HashMap<usize, (usize, usize)> = HashMap::new();

    for (img_idx, image_def) in image_defs.iter().enumerate() {
        let type_start = image_def.type_start as usize;
        let type_end = type_start + image_def.type_count as usize;
        for index in type_start..type_end {
            global_type_map.insert(index, (img_idx, index));
        }
    }

    let dll_outputs: Vec<(String, Vec<u8>)> = image_defs.par_iter().enumerate().filter_map(|(img_idx, image_def)| {
        let image_name = metadata.get_string_from_index(image_def.name_index)
            .unwrap_or_else(|_| format!("Assembly-{img_idx}.dll"));

        let assembly_name = assembly_defs.get(image_def.assembly_index as usize)
            .and_then(|ad| metadata.get_string_from_index(ad.aname.name_index).ok())
            .unwrap_or_else(|| image_name.replace(".dll", ""));

        let mut resolution = Resolution::new(Module::new(&image_name));
        resolution.assembly = Some(dotnetdll::resolved::assembly::Assembly::new(&assembly_name));

        if let Some(ad) = assembly_defs.get(image_def.assembly_index as usize) {
            if let Some(ref mut asm) = resolution.assembly {
                asm.version = dotnetdll::resolved::assembly::Version {
                    major: ad.aname.major as u16,
                    minor: ad.aname.minor as u16,
                    build: if ad.aname.build >= 0 { ad.aname.build as u16 } else { 0 },
                    revision: if ad.aname.revision >= 0 { ad.aname.revision as u16 } else { 0 },
                };
            }
        }

        resolution.type_definitions.clear();

        let mscorlib_ref = resolution.push_assembly_reference(
            ExternalAssemblyReference::new("mscorlib"),
        );

        let void_type_ref = resolution.push_type_reference(
            ExternalTypeReference::new(
                Some(Cow::Borrowed("System")),
                "Void",
                ResolutionScope::Assembly(mscorlib_ref),
            ),
        );

        let mut type_map: HashMap<usize, usize> = HashMap::new();
        let type_start = image_def.type_start as usize;
        let type_end = type_start + image_def.type_count as usize;

        for index in type_start..type_end {
            if let Some(type_def) = type_defs_all.get(index) {
                let ns = metadata.get_string_from_index(type_def.namespace_index).unwrap_or_default();
                let name = metadata.get_string_from_index(type_def.name_index)
                    .unwrap_or_else(|_| format!("Type_{index}"));

                let ns_opt = if ns.is_empty() { None } else { Some(Cow::Owned(ns)) };
                let mut td = DotNetTypeDef::new(ns_opt, Cow::Owned(name));

                td.flags = TypeFlags::default();
                td.flags.accessibility = type_accessibility_from_flags(type_def.flags);
                td.flags.abstract_type = (type_def.flags & 0x80) != 0;
                td.flags.sealed = (type_def.flags & 0x100) != 0;
                if (type_def.flags & 0x20) != 0 {
                    td.flags.kind = dotnetdll::resolved::types::Kind::Interface;
                }
                td.flags.special_name = (type_def.flags & 0x400) != 0;
                td.flags.serializable = (type_def.flags & 0x2000) != 0;
                td.flags.before_field_init = (type_def.flags & 0x0010_0000) != 0;
                td.flags.runtime_special_name = (type_def.flags & 0x800) != 0;

                let dotnet_idx = resolution.type_definitions.len();
                resolution.type_definitions.push(td);
                type_map.insert(index, dotnet_idx);
            }
        }

        let attr_ctors = setup_il2cpp_dummy_dll_refs(&mut resolution);

        let ctx = DummyDllContext {
            type_map: type_map.clone(),
            mscorlib_ref,
            void_type_ref,
            gen_param_total: generic_parameters.len(),
        };

        for index in type_start..type_end {
            if let Some(type_def) = type_defs_all.get(index) {
                if let Some(&dotnet_idx) = type_map.get(&index) {
                    for i in 0..type_def.nested_type_count as usize {
                        if let Some(&nested_il2cpp) = nested_type_indices.get(type_def.nested_types_start as usize + i) {
                            if let Some(&nested_dotnet) = type_map.get(&(nested_il2cpp as usize)) {
                                if let Some(type_idx) = resolution.type_definition_index(dotnet_idx) {
                                    resolution.type_definitions[nested_dotnet].encloser = Some(type_idx);
                                }
                            }
                        }
                    }
                }
            }
        }

        for index in type_start..type_end {
            if let Some(type_def) = type_defs_all.get(index).cloned() {
                if let Some(&dotnet_idx) = type_map.get(&index) {
                    if type_def.generic_container_index >= 0 {
                        if let Some(gc) = generic_containers.get(type_def.generic_container_index as usize) {
                            for i in 0..gc.type_argc as usize {
                                let gp_idx = gc.generic_parameter_start as usize + i;
                                if let Some(gp) = generic_parameters.get(gp_idx) {
                                    let gp_name = metadata.get_string_from_index(gp.name_index as i32)
                                        .unwrap_or_else(|_| format!("T{i}"));
                                    let mut gen_param: generic::Type<'_> =
                                        generic::Generic::new(Cow::Owned(gp_name));

                                    gen_param.special_constraint.reference_type = (gp.flags & 0x04) != 0;
                                    gen_param.special_constraint.value_type = (gp.flags & 0x08) != 0;
                                    gen_param.special_constraint.has_default_constructor = (gp.flags & 0x10) != 0;

                                    match gp.flags & 0x03 {
                                        0x01 => gen_param.variance = generic::Variance::Covariant,
                                        0x02 => gen_param.variance = generic::Variance::Contravariant,
                                        _ => {}
                                    }

                                    for ci in 0..gp.constraints_count as usize {
                                        let constraint_type_idx = constraint_indices
                                            .get(gp.constraints_start as usize + ci)
                                            .copied()
                                            .unwrap_or(-1);
                                        if constraint_type_idx >= 0 {
                                            if let Some(ct) = types_all.get(constraint_type_idx as usize) {
                                                let ctype = il2cpp_type_to_member(
                                                    ct, &types_all, &type_map, &mut resolution, &ctx,
                                                );
                                                gen_param.type_constraints.push(generic::Constraint {
                                                    attributes: vec![],
                                                    custom_modifiers: vec![],
                                                    constraint_type: ctype,
                                                });
                                            }
                                        }
                                    }

                                    resolution.type_definitions[dotnet_idx].generic_parameters.push(gen_param);
                                }
                            }
                        }
                    }

                    if type_def.parent_index >= 0 {
                        if let Some(parent_type) = types_all.get(type_def.parent_index as usize) {
                            let parent_source = il2cpp_type_to_type_source(
                                parent_type, &types_all, &type_map, &mut resolution, &ctx,
                            );
                            if let Some(src) = parent_source {
                                resolution.type_definitions[dotnet_idx].extends = Some(src);
                            }
                        }
                    }

                    for i in 0..type_def.interfaces_count as usize {
                        if let Some(&iface_type_idx) = interface_indices.get(type_def.interfaces_start as usize + i) {
                            if let Some(iface_type) = types_all.get(iface_type_idx as usize) {
                                let iface_source = il2cpp_type_to_type_source(
                                    iface_type, &types_all, &type_map, &mut resolution, &ctx,
                                );
                                if let Some(src) = iface_source {
                                    resolution.type_definitions[dotnet_idx].implements.push((vec![], src));
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut multicast_delegate_set: HashMap<usize, bool> = HashMap::new();
        for index in type_start..type_end {
            if let Some(type_def) = type_defs_all.get(index) {
                let is_mcd = if type_def.parent_index >= 0 {
                    if let Some(parent_type) = types_all.get(type_def.parent_index as usize) {
                        if let Some(td) = get_type_def_from_il2cpp_type(parent_type, &type_defs_all) {
                            let ns = metadata.get_string_from_index(td.namespace_index).unwrap_or_default();
                            let name = metadata.get_string_from_index(td.name_index).unwrap_or_default();
                            ns == "System" && name == "MulticastDelegate"
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                multicast_delegate_set.insert(index, is_mcd);
            }
        }

        for index in type_start..type_end {
            if let Some(type_def) = type_defs_all.get(index).cloned() {
                if let Some(&dotnet_idx) = type_map.get(&index) {
                    let skip_body = *multicast_delegate_set.get(&index).unwrap_or(&false);

                    let type_token = if metadata.variant == crate::il2cpp::metadata::MetadataVariant::Codm {
                        0
                    } else {
                        type_def.token
                    };
                    let token_attr = make_named_string_attr(
                        attr_ctors.token_ctor,
                        vec![("Token", format!("0x{:X}", type_token))],
                    );
                    resolution.type_definitions[dotnet_idx].attributes.push(token_attr);

                    let field_end = type_def.field_start + type_def.field_count as i32;
                    for fi in type_def.field_start..field_end {
                        if let Some(field_def) = field_defs_all.get(fi as usize) {
                            let fname = metadata.get_string_from_index(field_def.name_index)
                                .unwrap_or_else(|_| format!("field_{fi}"));

                            let ft = types_all.get(field_def.type_index as usize);
                            let member_type = ft
                                .map(|t| il2cpp_type_to_member(t, &types_all, &type_map, &mut resolution, &ctx))
                                .unwrap_or_else(member_object);

                            let field_attrs = ft.map(|t| t.attrs).unwrap_or(0x6);
                            let access = field_accessibility_from_attrs(field_attrs);
                            let is_static = (field_attrs & 0x10) != 0;
                            let is_literal = (field_attrs & 0x40) != 0;
                            let is_init_only = (field_attrs & 0x20) != 0;

                            let mut field = Field::new(is_static, access, Cow::Owned(fname), member_type);
                            field.literal = is_literal;
                            field.init_only = is_init_only;

                            if let Some(fdv) = metadata.get_field_default_value(fi) {
                                if fdv.data_index != -1 {
                                    match executor.try_get_default_value(
                                        fdv.type_index, fdv.data_index, metadata, il2cpp,
                                    ) {
                                        Ok(dv) => {
                                            field.default = Some(default_value_to_constant(&dv));
                                        }
                                        Err(_) => {}
                                    }
                                }
                            }

                            let field_token = if metadata.variant == crate::il2cpp::metadata::MetadataVariant::Codm {
                                0
                            } else {
                                field_def.token
                            };
                            field.attributes.push(make_named_string_attr(
                                attr_ctors.token_ctor,
                                vec![("Token", format!("0x{:X}", field_token))],
                            ));

                            if !is_literal {
                                let ft_enum = ft.map(|t| Il2CppTypeEnum::from_u8(t.type_enum));
                                let is_vt = matches!(ft_enum, Some(Some(Il2CppTypeEnum::ValueType)));
                                let field_offset = il2cpp.get_field_offset_from_index(
                                    index, (fi - type_def.field_start) as usize, fi as usize,
                                    is_vt, is_static,
                                );
                                if field_offset >= 0 {
                                    field.attributes.push(make_named_string_attr(
                                        attr_ctors.field_offset_ctor,
                                        vec![("Offset", format!("0x{:X}", field_offset))],
                                    ));
                                }

                                if config.dump_static_field_metadata {
                                    if let Some(ft_ref) = ft {
                                        let layout = crate::il2cpp::field_layout::analyze_field_layout(
                                            metadata,
                                            il2cpp,
                                            &type_def,
                                            index,
                                            (fi - type_def.field_start) as usize,
                                            fi as usize,
                                            field_def,
                                            ft_ref,
                                            config.max_field_rva_dump_bytes,
                                        );
                                        if layout.is_thread_static && is_static {
                                            field.attributes.push(make_named_string_attr(
                                                attr_ctors.thread_static_ctor,
                                                vec![("Offset", format!("0x{:X}", layout.effective_offset))],
                                            ));
                                        }
                                        let layout_attrs = crate::output::static_field_exporter::dummy_dll_layout_attrs(
                                            &layout, config,
                                        );
                                        for (name, value) in layout_attrs {
                                            if name == "ThreadStatic" {
                                                continue;
                                            }
                                            let ctor = if name.starts_with("FieldRva") || name == "MetadataOffset" {
                                                attr_ctors.field_rva_ctor
                                            } else {
                                                attr_ctors.metadata_offset_ctor
                                            };
                                            field.attributes.push(make_named_string_attr(
                                                ctor,
                                                vec![(name, value)],
                                            ));
                                        }
                                    }
                                }
                            }

                            resolution.type_definitions[dotnet_idx].fields.push(field);
                        }
                    }

                    let method_end = type_def.method_start + type_def.method_count as i32;
                    for mi in type_def.method_start..method_end {
                        if let Some(method_def) = method_defs_all.get(mi as usize).cloned() {
                            let mname = metadata.get_string_from_index(method_def.name_index as i32)
                                .unwrap_or_else(|_| format!("method_{mi}"));

                            let ret_il2cpp = types_all.get(method_def.return_type as usize);
                            let ret_type = ret_il2cpp
                                .map(|t| il2cpp_type_to_return(t, &types_all, &type_map, &mut resolution, &ctx))
                                .unwrap_or(ReturnType::VOID);

                            let mut params = Vec::new();
                            let mut param_metadata = Vec::new();
                            for j in 0..method_def.parameter_count as usize {
                                let pidx = method_def.parameter_start as usize + j;
                                if let Some(pdef) = parameter_defs_all.get(pidx) {
                                    let pt = types_all.get(pdef.type_index as usize);
                                    let ptype = pt
                                        .map(|t| il2cpp_type_to_method_type(t, &types_all, &type_map, &mut resolution, &ctx))
                                        .unwrap_or_else(method_object);

                                    let is_byref = pt.map(|t| t.byref == 1).unwrap_or(false);
                                    if is_byref {
                                        params.push(Parameter::reference(ptype));
                                    } else {
                                        params.push(Parameter::value(ptype));
                                    }

                                    let pname = metadata.get_string_from_index(pdef.name_index)
                                        .unwrap_or_else(|_| format!("param{j}"));
                                    let mut pmeta = ParameterMetadata::name(Cow::Owned(pname));
                                    pmeta.is_in = (pt.map(|t| t.attrs).unwrap_or(0) & 0x0001) != 0;
                                    pmeta.is_out = (pt.map(|t| t.attrs).unwrap_or(0) & 0x0002) != 0;
                                    pmeta.optional = (pt.map(|t| t.attrs).unwrap_or(0) & 0x0010) != 0;

                                    if let Some(pdv) = metadata.get_parameter_default_value(pidx as i32) {
                                        if pdv.data_index != -1 {
                                            match executor.try_get_default_value(
                                                pdv.type_index, pdv.data_index, metadata, il2cpp,
                                            ) {
                                                Ok(dv) => {
                                                    pmeta.default = Some(default_value_to_constant(&dv));
                                                }
                                                Err(_) => {}
                                            }
                                        }
                                    }

                                    param_metadata.push(Some(pmeta));
                                }
                            }

                            let is_static = (method_def.flags & 0x10) != 0;
                            let is_abstract = (method_def.flags & 0x0400) != 0;
                            let is_virtual = (method_def.flags & 0x0040) != 0;
                            let is_final = (method_def.flags & 0x0020) != 0;
                            let is_hide_by_sig = (method_def.flags & 0x0080) != 0;
                            let is_special_name = (method_def.flags & 0x0800) != 0;
                            let is_rt_special_name = (method_def.flags & 0x1000) != 0;
                            let has_new_slot = (method_def.flags & 0x0100) != 0;

                            let sig = if is_static {
                                MethodSignature::static_member(ret_type.clone(), params)
                            } else {
                                MethodSignature::new(true, ret_type.clone(), params)
                            };

                            let has_body = !is_abstract && !skip_body;

                            let method_body = if has_body {
                                let is_void = ret_il2cpp
                                    .map(|t| Il2CppTypeEnum::from_u8(t.type_enum) == Some(Il2CppTypeEnum::Void))
                                    .unwrap_or(true);

                                if is_void {
                                    Some(body::Method::new(vec![Instruction::Return]))
                                } else {
                                    let is_value = ret_il2cpp
                                        .map(|t| {
                                            let te = Il2CppTypeEnum::from_u8(t.type_enum);
                                            matches!(te,
                                                Some(Il2CppTypeEnum::ValueType)
                                                | Some(Il2CppTypeEnum::Boolean)
                                                | Some(Il2CppTypeEnum::Char)
                                                | Some(Il2CppTypeEnum::I1)
                                                | Some(Il2CppTypeEnum::U1)
                                                | Some(Il2CppTypeEnum::I2)
                                                | Some(Il2CppTypeEnum::U2)
                                                | Some(Il2CppTypeEnum::I4)
                                                | Some(Il2CppTypeEnum::U4)
                                                | Some(Il2CppTypeEnum::I8)
                                                | Some(Il2CppTypeEnum::U8)
                                                | Some(Il2CppTypeEnum::R4)
                                                | Some(Il2CppTypeEnum::R8)
                                                | Some(Il2CppTypeEnum::I)
                                                | Some(Il2CppTypeEnum::U)
                                            )
                                        })
                                        .unwrap_or(false);

                                    if is_value {
                                        Some(body::Method::new(vec![
                                            Instruction::LoadConstantInt32(0),
                                            Instruction::Return,
                                        ]))
                                    } else {
                                        Some(body::Method::new(vec![
                                            Instruction::LoadNull,
                                            Instruction::Return,
                                        ]))
                                    }
                                }
                            } else {
                                None
                            };

                            let access = member_accessibility_from_flags(method_def.flags);
                            let mut method = Method::new(access, sig, Cow::Owned(mname), method_body);

                            method.abstract_member = is_abstract;
                            method.virtual_member = is_virtual;
                            method.sealed = is_final;
                            method.hide_by_sig = is_hide_by_sig;
                            method.special_name = is_special_name;
                            method.runtime_special_name = is_rt_special_name;
                            method.parameter_metadata = param_metadata;

                            if has_new_slot {
                                method.vtable_layout = members::VtableLayout::NewSlot;
                            }

                            method.attributes.push(make_named_string_attr(
                                attr_ctors.token_ctor,
                                vec![("Token", format!("0x{:X}", method_def.token))],
                            ));

                            if !is_abstract {
                                let method_pointer = il2cpp.get_method_pointer(&image_name, &method_def);
                                if method_pointer > 0 {
                                    let rva = il2cpp.get_rva(method_pointer);
                                    let offset = il2cpp.map_vatr(method_pointer).unwrap_or(0);
                                    let mut addr_fields = vec![
                                        ("RVA", format!("0x{:X}", rva)),
                                        ("Offset", format!("0x{:X}", offset)),
                                        ("VA", format!("0x{:X}", method_pointer)),
                                    ];
                                    if method_def.slot != 0xFFFF {
                                        addr_fields.push(("Slot", method_def.slot.to_string()));
                                    }
                                    method.attributes.push(make_named_string_attr(
                                        attr_ctors.address_ctor,
                                        addr_fields,
                                    ));
                                }
                            }

                            if method_def.generic_container_index >= 0 {
                                if let Some(gc) = generic_containers.get(method_def.generic_container_index as usize) {
                                    for gi in 0..gc.type_argc as usize {
                                        let gp_idx = gc.generic_parameter_start as usize + gi;
                                        if let Some(gp) = generic_parameters.get(gp_idx) {
                                            let gp_name = metadata.get_string_from_index(gp.name_index as i32)
                                                .unwrap_or_else(|_| format!("M{gi}"));
                                            let mut gen_param: generic::Method<'_> =
                                                generic::Generic::new(Cow::Owned(gp_name));

                                            gen_param.special_constraint.reference_type = (gp.flags & 0x04) != 0;
                                            gen_param.special_constraint.value_type = (gp.flags & 0x08) != 0;
                                            gen_param.special_constraint.has_default_constructor = (gp.flags & 0x10) != 0;

                                            match gp.flags & 0x03 {
                                                0x01 => gen_param.variance = generic::Variance::Covariant,
                                                0x02 => gen_param.variance = generic::Variance::Contravariant,
                                                _ => {}
                                            }

                                            method.generic_parameters.push(gen_param);
                                        }
                                    }
                                }
                            }

                            resolution.type_definitions[dotnet_idx].methods.push(method);
                        }
                    }

                    let prop_end = type_def.property_start + type_def.property_count as i32;
                    for pi in type_def.property_start..prop_end {
                        if let Some(prop_def) = property_defs_all.get(pi as usize) {
                            let pname = metadata.get_string_from_index(prop_def.name_index)
                                .unwrap_or_else(|_| format!("property_{pi}"));

                            let prop_type: MemberType = if prop_def.get >= 0 {
                                let gm_idx = (type_def.method_start + prop_def.get) as usize;
                                method_defs_all.get(gm_idx)
                                    .and_then(|gm| types_all.get(gm.return_type as usize))
                                    .map(|t| il2cpp_type_to_member(t, &types_all, &type_map, &mut resolution, &ctx))
                                    .unwrap_or_else(member_object)
                            } else if prop_def.set >= 0 {
                                let sm_idx = (type_def.method_start + prop_def.set) as usize;
                                method_defs_all.get(sm_idx)
                                    .and_then(|sm| parameter_defs_all.get(sm.parameter_start as usize))
                                    .and_then(|p| types_all.get(p.type_index as usize))
                                    .map(|t| il2cpp_type_to_member(t, &types_all, &type_map, &mut resolution, &ctx))
                                    .unwrap_or_else(member_object)
                            } else {
                                member_object()
                            };

                            let is_static_prop = if prop_def.get >= 0 {
                                let gm_idx = (type_def.method_start + prop_def.get) as usize;
                                method_defs_all.get(gm_idx)
                                    .map(|m| (m.flags & 0x10) != 0)
                                    .unwrap_or(false)
                            } else if prop_def.set >= 0 {
                                let sm_idx = (type_def.method_start + prop_def.set) as usize;
                                method_defs_all.get(sm_idx)
                                    .map(|m| (m.flags & 0x10) != 0)
                                    .unwrap_or(false)
                            } else {
                                false
                            };

                            let mut property = Property::new(
                                is_static_prop,
                                Cow::Owned(pname),
                                Parameter::value(prop_type),
                            );
                            property.special_name = (prop_def.attrs & 0x0200) != 0;
                            property.runtime_special_name = (prop_def.attrs & 0x0400) != 0;

                            resolution.type_definitions[dotnet_idx].properties.push(property);
                        }
                    }

                    let event_end = type_def.event_start + type_def.event_count as i32;
                    for ei in type_def.event_start..event_end {
                        if let Some(event_def) = event_defs_all.get(ei as usize) {
                            let ename = metadata.get_string_from_index(event_def.name_index)
                                .unwrap_or_else(|_| format!("event_{ei}"));

                            let etype = types_all.get(event_def.type_index as usize)
                                .map(|t| il2cpp_type_to_member(t, &types_all, &type_map, &mut resolution, &ctx))
                                .unwrap_or_else(member_object);

                            let add_method = if event_def.add >= 0 {
                                let add_mi = (type_def.method_start + event_def.add) as usize;
                                if let Some(md) = method_defs_all.get(add_mi) {
                                    let access = member_accessibility_from_flags(md.flags);
                                    let is_static_event = (md.flags & 0x10) != 0;
                                    let sig = if is_static_event {
                                        MethodSignature::static_member(
                                            ReturnType::VOID,
                                            vec![Parameter::value(method_type_from_member(&etype))],
                                        )
                                    } else {
                                        MethodSignature::new(
                                            true,
                                            ReturnType::VOID,
                                            vec![Parameter::value(method_type_from_member(&etype))],
                                        )
                                    };
                                    let add_name = metadata.get_string_from_index(md.name_index as i32)
                                        .unwrap_or_else(|_| format!("add_{ename}"));
                                    Method::new(
                                        access,
                                        sig,
                                        Cow::Owned(add_name),
                                        if !skip_body { Some(body::Method::new(vec![Instruction::Return])) } else { None },
                                    )
                                } else {
                                    make_stub_event_method(&format!("add_{ename}"), &etype, skip_body)
                                }
                            } else {
                                make_stub_event_method(&format!("add_{ename}"), &etype, skip_body)
                            };

                            let remove_method = if event_def.remove >= 0 {
                                let rm_mi = (type_def.method_start + event_def.remove) as usize;
                                if let Some(md) = method_defs_all.get(rm_mi) {
                                    let access = member_accessibility_from_flags(md.flags);
                                    let is_static_event = (md.flags & 0x10) != 0;
                                    let sig = if is_static_event {
                                        MethodSignature::static_member(
                                            ReturnType::VOID,
                                            vec![Parameter::value(method_type_from_member(&etype))],
                                        )
                                    } else {
                                        MethodSignature::new(
                                            true,
                                            ReturnType::VOID,
                                            vec![Parameter::value(method_type_from_member(&etype))],
                                        )
                                    };
                                    let rm_name = metadata.get_string_from_index(md.name_index as i32)
                                        .unwrap_or_else(|_| format!("remove_{ename}"));
                                    Method::new(
                                        access,
                                        sig,
                                        Cow::Owned(rm_name),
                                        if !skip_body { Some(body::Method::new(vec![Instruction::Return])) } else { None },
                                    )
                                } else {
                                    make_stub_event_method(&format!("remove_{ename}"), &etype, skip_body)
                                }
                            } else {
                                make_stub_event_method(&format!("remove_{ename}"), &etype, skip_body)
                            };

                            let mut event = Event::new(
                                Cow::Owned(ename),
                                etype,
                                add_method,
                                remove_method,
                            );
                            event.special_name = (event_def.type_index as u32 & 0x0200) != 0;

                            resolution.type_definitions[dotnet_idx].events.push(event);
                        }
                    }
                }
            }
        }

        if metadata.version > 20.0 {
            for index in type_start..type_end {
                if let Some(type_def) = type_defs_all.get(index).cloned() {
                    if let Some(&dotnet_idx) = type_map.get(&index) {
                        add_attribute_attributes(
                            metadata, il2cpp, executor, img_idx,
                            type_def.custom_attribute_index, type_def.token,
                            &type_defs_all, &attr_ctors,
                            &mut resolution.type_definitions[dotnet_idx].attributes,
                        );

                        let field_end = type_def.field_start + type_def.field_count as i32;
                        for fi in type_def.field_start..field_end {
                            if let Some(field_def) = field_defs_all.get(fi as usize) {
                                let fi_in_type = (fi - type_def.field_start) as usize;
                                if fi_in_type < resolution.type_definitions[dotnet_idx].fields.len() {
                                    add_attribute_attributes(
                                        metadata, il2cpp, executor, img_idx,
                                        field_def.custom_attribute_index, field_def.token,
                                        &type_defs_all, &attr_ctors,
                                        &mut resolution.type_definitions[dotnet_idx].fields[fi_in_type].attributes,
                                    );
                                }
                            }
                        }

                        let method_end = type_def.method_start + type_def.method_count as i32;
                        for mi in type_def.method_start..method_end {
                            if let Some(method_def) = method_defs_all.get(mi as usize) {
                                let mi_in_type = (mi - type_def.method_start) as usize;
                                if mi_in_type < resolution.type_definitions[dotnet_idx].methods.len() {
                                    add_attribute_attributes(
                                        metadata, il2cpp, executor, img_idx,
                                        method_def.custom_attribute_index, method_def.token,
                                        &type_defs_all, &attr_ctors,
                                        &mut resolution.type_definitions[dotnet_idx].methods[mi_in_type].attributes,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        let dll_name = if image_name.ends_with(".dll") {
            image_name.clone()
        } else {
            format!("{image_name}.dll")
        };

        match resolution.write(Default::default()) {
            Ok(bytes) => Some((dll_name, bytes)),
            Err(e) => {
                eprintln!("WARNING: Failed to serialize {dll_name}: {e:?}");
                None
            }
        }
    }).collect();

    dll_outputs.par_iter().for_each(|(name, bytes)| {
        let dll_path = dummy_dir.join(name);
        if let Err(e) = fs::write(&dll_path, bytes) {
            eprintln!("WARNING: Failed to write {name}: {e}");
        }
    });

    generate_il2cpp_dummy_dll(&dummy_dir)?;

    Ok(())
}

fn make_stub_event_method<'a>(name: &str, etype: &MemberType, skip_body: bool) -> Method<'a> {
    let sig = MethodSignature::new(
        true,
        ReturnType::VOID,
        vec![Parameter::value(method_type_from_member(etype))],
    );
    Method::new(
        dotnetdll::resolved::Accessibility::Public,
        sig,
        Cow::Owned(name.to_string()),
        if !skip_body { Some(body::Method::new(vec![Instruction::Return])) } else { None },
    )
}

fn add_attribute_attributes<'a>(
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
    executor: &Il2CppExecutor,
    image_index: usize,
    custom_attribute_index: i32,
    token: u32,
    type_defs_all: &[Il2CppTypeDefinition],
    attr_ctors: &AttrCtors,
    target_attrs: &mut Vec<Attribute<'a>>,
) {
    let attr_idx = match metadata.get_custom_attribute_index(image_index, custom_attribute_index, token) {
        Some(idx) => idx,
        None => return,
    };

    if metadata.version < 29.0 {
        if attr_idx >= metadata.attribute_type_ranges.len() {
            return;
        }
        let range = &metadata.attribute_type_ranges[attr_idx];
        let range_start = range.start as usize;
        let range_count = range.count as usize;
        for i in 0..range_count {
            let type_idx = match metadata.attribute_types.get(range_start + i) {
                Some(&idx) => idx,
                None => continue,
            };
            if type_idx < 0 || (type_idx as usize) >= il2cpp.types.len() {
                continue;
            }
            let attr_type = &il2cpp.types[type_idx as usize];
            let type_name = get_type_name_from_il2cpp_type(attr_type, metadata, type_defs_all);

            let method_pointer = if attr_idx < executor.custom_attribute_generators.len() {
                executor.custom_attribute_generators[attr_idx]
            } else {
                0
            };

            let mut fields = vec![("Name", type_name)];
            if method_pointer > 0 {
                let rva = il2cpp.get_rva(method_pointer);
                let offset = il2cpp.map_vatr(method_pointer).unwrap_or(0);
                fields.push(("RVA", format!("0x{:X}", rva)));
                fields.push(("Offset", format!("0x{:X}", offset)));
            }

            target_attrs.push(make_named_string_attr(attr_ctors.attribute_ctor, fields));
        }
    } else {
        let start_range = match metadata.attribute_data_ranges.get(attr_idx) {
            Some(r) => r.clone(),
            None => return,
        };
        let end_range = match metadata.attribute_data_ranges.get(attr_idx + 1) {
            Some(r) => r.clone(),
            None => return,
        };

        if end_range.start_offset <= start_range.start_offset {
            return;
        }

        let data_offset = metadata.header.attribute_data_offset as u64 + start_range.start_offset as u64;
        let data_size = (end_range.start_offset - start_range.start_offset) as usize;
        if data_size == 0 || data_size > 1024 * 1024 {
            return;
        }

        let data = match metadata.peek_bytes_at(data_offset, data_size) {
            Ok(d) => d.to_vec(),
            Err(_) => return,
        };

        let mut reader = match crate::executor::custom_attribute_reader::CustomAttributeDataReader::new(data) {
            Ok(r) => r,
            Err(_) => return,
        };

        if reader.count == 0 {
            return;
        }

        for _ in 0..reader.count {
            match reader.get_ctor_type_name(metadata) {
                Ok(type_name) => {
                    let fields = vec![("Name", type_name)];
                    target_attrs.push(make_named_string_attr(attr_ctors.attribute_ctor, fields));
                }
                Err(_) => break,
            }
        }
    }
}

fn get_type_name_from_il2cpp_type(
    il2cpp_type: &Il2CppType,
    metadata: &Metadata,
    type_defs: &[Il2CppTypeDefinition],
) -> String {
    let te = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
    match te {
        Some(Il2CppTypeEnum::Class) | Some(Il2CppTypeEnum::ValueType) => {
            let td_idx = il2cpp_type.datapoint as usize;
            if let Some(td) = type_defs.get(td_idx) {
                let ns = metadata.get_string_from_index(td.namespace_index).unwrap_or_default();
                let name = metadata.get_string_from_index(td.name_index).unwrap_or_default();
                if ns.is_empty() { name } else { format!("{ns}.{name}") }
            } else {
                format!("Type_{td_idx}")
            }
        }
        _ => format!("il2cpp_type_{}", il2cpp_type.type_enum),
    }
}

fn generate_il2cpp_dummy_dll(dummy_dir: &Path) -> Result<()> {
    let mut resolution = Resolution::new(Module::new("Il2CppDummyDll.dll"));
    resolution.assembly = Some(dotnetdll::resolved::assembly::Assembly::new("Il2CppDummyDll"));
    resolution.type_definitions.clear();

    let mscorlib_ref = resolution.push_assembly_reference(
        ExternalAssemblyReference::new("mscorlib"),
    );

    let attribute_base_ref = resolution.push_type_reference(
        ExternalTypeReference::new(
            Some(Cow::Borrowed("System")),
            "Attribute",
            ResolutionScope::Assembly(mscorlib_ref),
        ),
    );
    let attribute_base = TypeSource::User(UserType::Reference(attribute_base_ref));

    let string_type = MemberType::Base(Box::new(BaseType::String));

    let attr_types = &[
        ("AddressAttribute", vec!["RVA", "Offset", "VA", "Slot"]),
        ("FieldOffsetAttribute", vec!["Offset"]),
        ("TokenAttribute", vec!["Token"]),
        ("AttributeAttribute", vec!["Name", "RVA", "Offset"]),
        ("MetadataOffsetAttribute", vec!["Offset"]),
    ];

    for (name, fields) in attr_types {
        let mut td = DotNetTypeDef::new(None, Cow::Owned(name.to_string()));
        td.extends = Some(attribute_base.clone());
        td.flags.accessibility = TypeAccessibility::Public;
        td.flags.before_field_init = true;

        for field_name in fields {
            let field = Field::new(
                false,
                dotnetdll::resolved::Accessibility::Public,
                Cow::Owned(field_name.to_string()),
                string_type.clone(),
            );
            td.fields.push(field);
        }

        let ctor = Method::constructor(
            dotnetdll::resolved::Accessibility::Public,
            vec![],
            Some(body::Method::new(vec![Instruction::Return])),
        );
        td.methods.push(ctor);

        resolution.type_definitions.push(td);
    }

    let dll_path = dummy_dir.join("Il2CppDummyDll.dll");
    match resolution.write(Default::default()) {
        Ok(bytes) => {
            if let Err(e) = fs::write(&dll_path, &bytes) {
                eprintln!("WARNING: Failed to write Il2CppDummyDll.dll: {e}");
            }
        }
        Err(e) => {
            eprintln!("WARNING: Failed to serialize Il2CppDummyDll.dll: {e:?}");
        }
    }

    Ok(())
}

fn default_value_to_constant(dv: &crate::executor::il2cpp_executor::DefaultValue) -> members::Constant {
    use crate::executor::il2cpp_executor::DefaultValue;
    match dv {
        DefaultValue::Bool(v) => members::Constant::Boolean(*v),
        DefaultValue::U8(v) => members::Constant::UInt8(*v),
        DefaultValue::I8(v) => members::Constant::Int8(*v),
        DefaultValue::Char(v) => members::Constant::Char(*v as u16),
        DefaultValue::U16(v) => members::Constant::UInt16(*v),
        DefaultValue::I16(v) => members::Constant::Int16(*v),
        DefaultValue::U32(v) => members::Constant::UInt32(*v),
        DefaultValue::I32(v) => members::Constant::Int32(*v),
        DefaultValue::U64(v) => members::Constant::UInt64(*v),
        DefaultValue::I64(v) => members::Constant::Int64(*v),
        DefaultValue::F32(v) => members::Constant::Float32(*v),
        DefaultValue::F64(v) => members::Constant::Float64(*v),
        DefaultValue::String(v) => members::Constant::String(v.encode_utf16().collect()),
        DefaultValue::Null => members::Constant::Null,
    }
}

fn member_object() -> MemberType {
    MemberType::Base(Box::new(BaseType::Object))
}

fn method_object() -> MethodType {
    MethodType::Base(Box::new(BaseType::Object))
}

fn method_type_from_member(m: &MemberType) -> MethodType {
    match m {
        MemberType::Base(b) => MethodType::Base(Box::new(base_member_to_method(b))),
        MemberType::TypeGeneric(i) => MethodType::TypeGeneric(*i),
    }
}

fn base_member_to_method(b: &BaseType<MemberType>) -> BaseType<MethodType> {
    match b {
        BaseType::Type { value_kind, source } => BaseType::Type {
            value_kind: *value_kind,
            source: match source {
                TypeSource::User(u) => TypeSource::User(*u),
                TypeSource::Generic { base, parameters } => TypeSource::Generic {
                    base: *base,
                    parameters: parameters.iter().map(|p| method_type_from_member(p)).collect(),
                },
            },
        },
        BaseType::Boolean => BaseType::Boolean,
        BaseType::Char => BaseType::Char,
        BaseType::Int8 => BaseType::Int8,
        BaseType::UInt8 => BaseType::UInt8,
        BaseType::Int16 => BaseType::Int16,
        BaseType::UInt16 => BaseType::UInt16,
        BaseType::Int32 => BaseType::Int32,
        BaseType::UInt32 => BaseType::UInt32,
        BaseType::Int64 => BaseType::Int64,
        BaseType::UInt64 => BaseType::UInt64,
        BaseType::Float32 => BaseType::Float32,
        BaseType::Float64 => BaseType::Float64,
        BaseType::IntPtr => BaseType::IntPtr,
        BaseType::UIntPtr => BaseType::UIntPtr,
        BaseType::Object => BaseType::Object,
        BaseType::String => BaseType::String,
        BaseType::Vector(mods, inner) => {
            BaseType::Vector(mods.clone(), method_type_from_member(inner))
        }
        BaseType::ValuePointer(mods, opt) => {
            BaseType::ValuePointer(mods.clone(), opt.as_ref().map(method_type_from_member))
        }
        _ => BaseType::Object,
    }
}

fn get_type_def_from_il2cpp_type<'a>(
    il2cpp_type: &Il2CppType,
    type_defs: &'a [Il2CppTypeDefinition],
) -> Option<&'a Il2CppTypeDefinition> {
    let type_enum = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
    match type_enum {
        Some(Il2CppTypeEnum::Class) | Some(Il2CppTypeEnum::ValueType) => {
            type_defs.get(il2cpp_type.datapoint as usize)
        }
        _ => None,
    }
}

fn resolve_type_source_for_typedef(
    td_index: usize,
    type_map: &HashMap<usize, usize>,
    resolution: &mut Resolution<'_>,
    ctx: &DummyDllContext,
    _is_valuetype: bool,
) -> TypeSource<MemberType> {
    if let Some(&dotnet_idx) = type_map.get(&td_index) {
        if let Some(type_idx) = resolution.type_definition_index(dotnet_idx) {
            return TypeSource::User(UserType::Definition(type_idx));
        }
    }

    let type_ref = resolution.push_type_reference(
        ExternalTypeReference::new(
            Some(Cow::Owned(String::new())),
            format!("__External_{td_index}"),
            ResolutionScope::Assembly(ctx.mscorlib_ref),
        ),
    );
    TypeSource::User(UserType::Reference(type_ref))
}

fn il2cpp_type_to_type_source(
    il2cpp_type: &Il2CppType,
    types: &[Il2CppType],
    type_map: &HashMap<usize, usize>,
    resolution: &mut Resolution<'_>,
    ctx: &DummyDllContext,
) -> Option<TypeSource<MemberType>> {
    let type_enum = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum)?;
    match type_enum {
        Il2CppTypeEnum::Class | Il2CppTypeEnum::ValueType => {
            let td_index = il2cpp_type.datapoint as usize;
            Some(resolve_type_source_for_typedef(
                td_index, type_map, resolution, ctx,
                type_enum == Il2CppTypeEnum::ValueType,
            ))
        }
        _ => {
            let mt = il2cpp_type_to_member(il2cpp_type, types, type_map, resolution, ctx);
            match mt {
                MemberType::Base(b) => match *b {
                    BaseType::Type { value_kind: _, source } => Some(source),
                    _ => None,
                },
                _ => None,
            }
        }
    }
}

fn il2cpp_type_to_base_member(
    il2cpp_type: &Il2CppType,
    _types: &[Il2CppType],
    type_map: &HashMap<usize, usize>,
    resolution: &mut Resolution<'_>,
    ctx: &DummyDllContext,
) -> BaseType<MemberType> {
    let type_enum = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
    match type_enum {
        Some(Il2CppTypeEnum::Boolean) => BaseType::Boolean,
        Some(Il2CppTypeEnum::Char) => BaseType::Char,
        Some(Il2CppTypeEnum::I1) => BaseType::Int8,
        Some(Il2CppTypeEnum::U1) => BaseType::UInt8,
        Some(Il2CppTypeEnum::I2) => BaseType::Int16,
        Some(Il2CppTypeEnum::U2) => BaseType::UInt16,
        Some(Il2CppTypeEnum::I4) => BaseType::Int32,
        Some(Il2CppTypeEnum::U4) => BaseType::UInt32,
        Some(Il2CppTypeEnum::I8) => BaseType::Int64,
        Some(Il2CppTypeEnum::U8) => BaseType::UInt64,
        Some(Il2CppTypeEnum::R4) => BaseType::Float32,
        Some(Il2CppTypeEnum::R8) => BaseType::Float64,
        Some(Il2CppTypeEnum::String) => BaseType::String,
        Some(Il2CppTypeEnum::Object) => BaseType::Object,
        Some(Il2CppTypeEnum::I) => BaseType::IntPtr,
        Some(Il2CppTypeEnum::U) => BaseType::UIntPtr,
        Some(Il2CppTypeEnum::Void) => BaseType::Object,
        Some(Il2CppTypeEnum::Class) | Some(Il2CppTypeEnum::ValueType) => {
            let td_index = il2cpp_type.datapoint as usize;
            let vk = if type_enum == Some(Il2CppTypeEnum::ValueType) {
                Some(ValueKind::ValueType)
            } else {
                Some(ValueKind::Class)
            };
            let source = resolve_type_source_for_typedef(td_index, type_map, resolution, ctx, vk == Some(ValueKind::ValueType));
            BaseType::Type { value_kind: vk, source }
        }
        Some(Il2CppTypeEnum::SzArray) => {
            BaseType::vector(member_object())
        }
        Some(Il2CppTypeEnum::Array) => {
            BaseType::vector(member_object())
        }
        Some(Il2CppTypeEnum::Ptr) => {
            BaseType::ValuePointer(vec![], None)
        }
        Some(Il2CppTypeEnum::GenericInst) => {
            BaseType::Object
        }
        _ => BaseType::Object,
    }
}

fn il2cpp_type_to_member(
    il2cpp_type: &Il2CppType,
    types: &[Il2CppType],
    type_map: &HashMap<usize, usize>,
    resolution: &mut Resolution<'_>,
    ctx: &DummyDllContext,
) -> MemberType {
    let type_enum = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
    if let Some(Il2CppTypeEnum::Var) = type_enum {
        let dp = il2cpp_type.datapoint as usize;
        let idx = if ctx.gen_param_total > 0 && dp < ctx.gen_param_total { dp } else { 0 };
        return MemberType::TypeGeneric(idx);
    }
    MemberType::Base(Box::new(il2cpp_type_to_base_member(il2cpp_type, types, type_map, resolution, ctx)))
}

fn il2cpp_type_to_base_method(
    il2cpp_type: &Il2CppType,
    _types: &[Il2CppType],
    type_map: &HashMap<usize, usize>,
    resolution: &mut Resolution<'_>,
    ctx: &DummyDllContext,
) -> BaseType<MethodType> {
    let type_enum = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
    match type_enum {
        Some(Il2CppTypeEnum::Boolean) => BaseType::Boolean,
        Some(Il2CppTypeEnum::Char) => BaseType::Char,
        Some(Il2CppTypeEnum::I1) => BaseType::Int8,
        Some(Il2CppTypeEnum::U1) => BaseType::UInt8,
        Some(Il2CppTypeEnum::I2) => BaseType::Int16,
        Some(Il2CppTypeEnum::U2) => BaseType::UInt16,
        Some(Il2CppTypeEnum::I4) => BaseType::Int32,
        Some(Il2CppTypeEnum::U4) => BaseType::UInt32,
        Some(Il2CppTypeEnum::I8) => BaseType::Int64,
        Some(Il2CppTypeEnum::U8) => BaseType::UInt64,
        Some(Il2CppTypeEnum::R4) => BaseType::Float32,
        Some(Il2CppTypeEnum::R8) => BaseType::Float64,
        Some(Il2CppTypeEnum::String) => BaseType::String,
        Some(Il2CppTypeEnum::Object) => BaseType::Object,
        Some(Il2CppTypeEnum::I) => BaseType::IntPtr,
        Some(Il2CppTypeEnum::U) => BaseType::UIntPtr,
        Some(Il2CppTypeEnum::Void) => BaseType::Object,
        Some(Il2CppTypeEnum::Class) | Some(Il2CppTypeEnum::ValueType) => {
            let td_index = il2cpp_type.datapoint as usize;
            let vk = if type_enum == Some(Il2CppTypeEnum::ValueType) {
                Some(ValueKind::ValueType)
            } else {
                Some(ValueKind::Class)
            };
            let source = resolve_type_source_for_typedef(td_index, type_map, resolution, ctx, vk == Some(ValueKind::ValueType));
            let method_source = match source {
                TypeSource::User(u) => TypeSource::User(u),
                TypeSource::Generic { base, parameters } => TypeSource::Generic {
                    base,
                    parameters: parameters.into_iter().map(|p| method_type_from_member(&p)).collect(),
                },
            };
            BaseType::Type { value_kind: vk, source: method_source }
        }
        Some(Il2CppTypeEnum::SzArray) => {
            BaseType::vector(method_object())
        }
        Some(Il2CppTypeEnum::Array) => {
            BaseType::vector(method_object())
        }
        Some(Il2CppTypeEnum::Ptr) => {
            BaseType::ValuePointer(vec![], None)
        }
        _ => BaseType::Object,
    }
}

fn il2cpp_type_to_method_type(
    il2cpp_type: &Il2CppType,
    types: &[Il2CppType],
    type_map: &HashMap<usize, usize>,
    resolution: &mut Resolution<'_>,
    ctx: &DummyDllContext,
) -> MethodType {
    let type_enum = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
    if let Some(Il2CppTypeEnum::Var) = type_enum {
        let dp = il2cpp_type.datapoint as usize;
        let idx = if ctx.gen_param_total > 0 && dp < ctx.gen_param_total { dp } else { 0 };
        return MethodType::TypeGeneric(idx);
    }
    if let Some(Il2CppTypeEnum::MVar) = type_enum {
        let dp = il2cpp_type.datapoint as usize;
        let idx = if ctx.gen_param_total > 0 && dp < ctx.gen_param_total { dp } else { 0 };
        return MethodType::MethodGeneric(idx);
    }
    MethodType::Base(Box::new(il2cpp_type_to_base_method(il2cpp_type, types, type_map, resolution, ctx)))
}

fn il2cpp_type_to_return(
    il2cpp_type: &Il2CppType,
    types: &[Il2CppType],
    type_map: &HashMap<usize, usize>,
    resolution: &mut Resolution<'_>,
    ctx: &DummyDllContext,
) -> ReturnType<MethodType> {
    let type_enum = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
    if let Some(Il2CppTypeEnum::Void) = type_enum {
        return ReturnType::VOID;
    }
    let mt = il2cpp_type_to_method_type(il2cpp_type, types, type_map, resolution, ctx);
    if il2cpp_type.byref == 1 {
        ReturnType::reference(mt)
    } else {
        ReturnType::value(mt)
    }
}
