use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;
use crate::error::Result;
use crate::il2cpp::base::*;
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::enums::*;
use crate::il2cpp::structures::*;
use crate::executor::Il2CppExecutor;
use super::script_json::*;
use super::header_constants;
use super::cpp_scaffolding::CppScaffolding;
use super::name_mangler::MangledNameBuilder;
use super::header_manager::UnityHeaders;

use crate::utils::{sanitize_cpp_identifier, NameSanitizerOptions};

static C_PRIMITIVE_TYPES: &[&str] = &[
    "void", "bool", "char", "int8_t", "uint8_t", "int16_t", "uint16_t",
    "int32_t", "uint32_t", "int64_t", "uint64_t", "float", "double",
    "intptr_t", "uintptr_t", "Il2CppChar",
];

fn needs_struct_prefix(type_name: &str) -> bool {
    let base = type_name.trim_end_matches('*');
    !C_PRIMITIVE_TYPES.contains(&base)
}

struct StructFieldInfo {
    field_type_name: String,
    field_name: String,
    is_value_type: bool,
    is_custom_type: bool,
}

struct StructVTableMethodInfo {
    method_name: String,
}

struct StructRGCTXInfo {
    rgctx_type: i64,
    type_name: Option<String>,
    class_name: Option<String>,
    method_name: Option<String>,
}

struct StructInfo {
    type_name: String,
    is_value_type: bool,
    parent: Option<String>,
    fields: Vec<StructFieldInfo>,
    static_fields: Vec<StructFieldInfo>,
    vtable_methods: Vec<Option<StructVTableMethodInfo>>,
    rgctxs: Vec<StructRGCTXInfo>,
}

pub struct StructGenerator;

/// Mutable context for header generation (il2cpp.h).
/// Holds state that grows during generic class discovery.
struct HeaderGenCtx {
    /// Maps generic_class pointer → specialized struct name (e.g., "List_1_System_Int32")
    generic_class_struct_name_dic: HashMap<u64, String>,
    /// HashSet for dedup of struct names
    struct_name_hash_set: HashSet<String>,
    /// Newly discovered generic class pointers found during field parsing
    newly_discovered: Vec<u64>,
}

impl StructGenerator {
    pub fn write_all(
        executor: &mut Il2CppExecutor,
        metadata: &mut Metadata,
        il2cpp: &mut Il2Cpp,
        config: &crate::config::Config,
        output_dir: &str,
        static_catalog: Option<&crate::output::static_field_exporter::StaticFieldCatalog>,
    ) -> Result<()> {
        let output_path = Path::new(output_dir);

        let script_json = Self::build_script_json(executor, metadata, il2cpp, config, static_catalog)?;
        let string_literal_json = Self::build_string_literal_json(metadata)?;
        let header = Self::build_header(executor, metadata, il2cpp, config)?;
        let mut functions_header = String::new();
        if config.generate_cpp_scaffold {
            functions_header = CppScaffolding::build(executor, metadata, il2cpp).unwrap_or_default();
        }

        let mut unity_type_header = String::new();
        let mut unity_api_header = String::new();
        if config.generate_unity_headers {
            let api_exports: HashSet<String> = il2cpp
                .exported_symbols
                .iter()
                .filter(|n| n.starts_with("il2cpp_") || n.starts_with("mono_"))
                .cloned()
                .collect();
            let metadata_version = il2cpp.version;
            let field_offsets_are_pointers = false; // Could be improved by checking MetadataRegistration
            let mut headers = UnityHeaders::guess_headers_for_binary(
                metadata_version as f32,
                il2cpp.is_32bit,
                &api_exports,
                field_offsets_are_pointers
            );
            
            if let Some(h) = headers.pop() { // Use the best guessed headers
                unity_type_header = h.get_type_header_text(il2cpp.is_32bit);
                unity_api_header = h.get_api_header_text();
            }
        }

        use rayon::prelude::*;
        let mut writes: Vec<(&str, &[u8])> = vec![
            ("script.json", script_json.as_bytes()),
            ("stringliteral.json", string_literal_json.as_bytes()),
            ("il2cpp.h", header.as_bytes()),
        ];
        if config.generate_cpp_scaffold {
            writes.push(("il2cpp-functions.h", functions_header.as_bytes()));
        }
        if config.generate_unity_headers {
            if !unity_type_header.is_empty() {
                writes.push(("il2cpp-types.h", unity_type_header.as_bytes()));
            }
            if !unity_api_header.is_empty() {
                writes.push(("il2cpp-api.h", unity_api_header.as_bytes()));
            }
        }
        writes.par_iter().for_each(|(name, data)| {
            let path = output_path.join(name);
            if let Err(e) = fs::write(&path, data) {
                eprintln!("WARNING: Failed to write {name}: {e}");
            }
        });

        if config.generate_cpp_scaffold && config.generate_unity_headers && !unity_type_header.is_empty() {
            let project_root = output_path.join("cpp_project");
            if let Err(e) = CppScaffolding::write_project(
                executor, metadata, il2cpp,
                &unity_type_header, &unity_api_header, &project_root,
            ) {
                eprintln!("WARNING: Failed to write cpp_project scaffolding: {e}");
            }
        }

        Ok(())
    }

    fn build_script_json(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        config: &crate::config::Config,
        static_catalog: Option<&crate::output::static_field_exporter::StaticFieldCatalog>,
    ) -> Result<String> {
        use rayon::prelude::*;

        let mut script = ScriptJson::new();
        let mut addresses_set: HashSet<u64> = HashSet::new();

        let struct_name_dic = Self::build_struct_name_dic(executor, metadata, il2cpp);
        let type_def_image_names = Self::build_type_def_image_names(metadata);

        let type_jobs: Vec<(usize, String)> = metadata
            .image_defs
            .iter()
            .flat_map(|image_def| {
                let image_name = metadata
                    .get_string_from_index(image_def.name_index)
                    .unwrap_or_default();
                let type_end = image_def.type_start as usize + image_def.type_count as usize;
                (image_def.type_start as usize..type_end)
                    .map(|type_def_index| (type_def_index, image_name.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();

        let chunks: Vec<(Vec<ScriptMethod>, HashSet<u64>)> = type_jobs
            .par_iter()
            .map(|(type_def_index, image_name)| {
                let mut local_exec = Il2CppExecutor::new_for_worker(executor);
                let mut methods = Vec::new();
                let mut local_addrs = HashSet::new();

                let type_def = metadata.type_defs[*type_def_index].clone();
                let type_name = local_exec.get_type_def_name(
                    &type_def,
                    *type_def_index,
                    metadata,
                    il2cpp,
                    true,
                    true,
                );

                let method_end = type_def.method_start as usize + type_def.method_count as usize;
                for method_index in type_def.method_start as usize..method_end {
                    let method_def = metadata.method_defs[method_index].clone();
                    let method_name_raw = metadata
                        .get_string_from_index(method_def.name_index as i32)
                        .unwrap_or_default();
                    let method_pointer = il2cpp.get_method_pointer(image_name, &method_def);

                    if method_pointer > 0 {
                        let rva = il2cpp.get_rva(method_pointer);
                        local_addrs.insert(rva);
                        let method_full_name = format!("{}$${}", type_name, method_name_raw);
                        let mangled_name = if config.mangle_names {
                            MangledNameBuilder::mangle_method(
                                &mut local_exec,
                                metadata,
                                il2cpp,
                                &type_def,
                                *type_def_index,
                                &method_def,
                            )
                        } else {
                            method_full_name.clone()
                        };

                        let (dotnet_sig, group) = if config.enhanced_ida_metadata {
                            (
                                Some(format!("{}::{}()", type_name, method_name_raw)),
                                Some(format!(
                                    "{}/{}",
                                    image_name.trim_end_matches(".dll"),
                                    type_name.replace('.', "/")
                                )),
                            )
                        } else {
                            (None, None)
                        };
                        let (signature, type_sig) = Self::build_method_signature(
                            &mut local_exec,
                            metadata,
                            il2cpp,
                            &method_def,
                            &type_def,
                            &method_full_name,
                            &struct_name_dic,
                            None,
                        );
                        methods.push(ScriptMethod {
                            address: rva,
                            name: mangled_name,
                            signature,
                            type_signature: type_sig,
                            dotnet_signature: dotnet_sig,
                            group,
                        });
                    }

                    if let Some(spec_indices) =
                        il2cpp.method_definition_method_specs.get(&method_index)
                    {
                        for spec_idx in spec_indices {
                            let spec_ptr = il2cpp
                                .method_spec_generic_method_pointers
                                .get(spec_idx)
                                .copied()
                                .unwrap_or(0);
                            if spec_ptr == 0 {
                                continue;
                            }
                            let spec_rva = il2cpp.get_rva(spec_ptr);
                            local_addrs.insert(spec_rva);

                            let (spec_type_name, spec_method_name) = local_exec
                                .get_method_spec_name(*spec_idx, metadata, il2cpp, true);
                            let method_full_name =
                                format!("{}$${}", spec_type_name, spec_method_name);

                            let (class_inst, method_inst) =
                                local_exec.get_method_spec_generic_context(*spec_idx, il2cpp);
                            let generic_context =
                                Il2CppGenericContext { class_inst, method_inst };

                            let mangled_name = if config.mangle_names {
                                MangledNameBuilder::mangle_method_spec(
                                    &mut local_exec,
                                    metadata,
                                    il2cpp,
                                    &type_def,
                                    *type_def_index,
                                    &method_def,
                                    &generic_context,
                                )
                            } else {
                                method_full_name.clone()
                            };

                            let (dotnet_sig, group) = if config.enhanced_ida_metadata {
                                (
                                    Some(format!("{}::{}()", spec_type_name, spec_method_name)),
                                    Some(format!(
                                        "{}/{}",
                                        image_name.trim_end_matches(".dll"),
                                        spec_type_name.replace('.', "/")
                                    )),
                                )
                            } else {
                                (None, None)
                            };

                            let (signature, type_sig) = Self::build_method_signature(
                                &mut local_exec,
                                metadata,
                                il2cpp,
                                &method_def,
                                &type_def,
                                &method_full_name,
                                &struct_name_dic,
                                Some(&generic_context),
                            );

                            methods.push(ScriptMethod {
                                address: spec_rva,
                                name: mangled_name,
                                signature,
                                type_signature: type_sig,
                                dotnet_signature: dotnet_sig,
                                group,
                            });
                        }
                    }
                }

                (methods, local_addrs)
            })
            .collect();

        for (methods, addrs) in chunks {
            script.script_methods.extend(methods);
            addresses_set.extend(addrs);
        }

        Self::collect_all_addresses(&mut addresses_set, executor, il2cpp);

        if il2cpp.version >= 27.0 {
            Self::scan_v27_metadata_usages(
                &mut script,
                executor,
                metadata,
                il2cpp,
                &struct_name_dic,
                &type_def_image_names,
                config,
            );
        } else if il2cpp.version > 16.0 {
            Self::add_metadata_usages(
                &mut script,
                executor,
                metadata,
                il2cpp,
                &struct_name_dic,
                &type_def_image_names,
                config,
            );
        }

        if let Some(catalog) = static_catalog {
            catalog.enrich_script_json(&mut script, il2cpp);
        }

        let mut sorted_addresses: Vec<u64> = addresses_set.into_iter().filter(|a| *a > 0).collect();
        sorted_addresses.sort_unstable();
        script.addresses = sorted_addresses;

        let json = script.to_json().map_err(|e| crate::error::Error::Other(e.to_string()))?;
        Ok(json)
    }

    fn collect_all_addresses(
        addresses_set: &mut HashSet<u64>,
        executor: &Il2CppExecutor,
        il2cpp: &Il2Cpp,
    ) {
        if il2cpp.version >= 24.2 {
            for pointers in il2cpp.code_gen_module_method_pointers.values() {
                for ptr in pointers {
                    if *ptr > 0 { addresses_set.insert(il2cpp.get_rva(*ptr)); }
                }
            }
        } else {
            for ptr in &il2cpp.method_pointers {
                if *ptr > 0 { addresses_set.insert(il2cpp.get_rva(*ptr)); }
            }
        }

        for ptr in &il2cpp.generic_method_pointers {
            if *ptr > 0 { addresses_set.insert(il2cpp.get_rva(*ptr)); }
        }
        for ptr in &il2cpp.invoker_pointers {
            if *ptr > 0 { addresses_set.insert(il2cpp.get_rva(*ptr)); }
        }

        if il2cpp.version < 29.0 {
            for ptr in &executor.custom_attribute_generators {
                if *ptr > 0 { addresses_set.insert(il2cpp.get_rva(*ptr)); }
            }
        }

        if il2cpp.version >= 22.0 {
            for ptr in &il2cpp.reverse_pinvoke_wrappers {
                if *ptr > 0 { addresses_set.insert(il2cpp.get_rva(*ptr)); }
            }
            for ptr in &il2cpp.unresolved_virtual_call_pointers {
                if *ptr > 0 { addresses_set.insert(il2cpp.get_rva(*ptr)); }
            }
        }
    }

    fn build_method_signature(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        method_def: &Il2CppMethodDefinition,
        type_def: &Il2CppTypeDefinition,
        method_full_name: &str,
        struct_name_dic: &HashMap<usize, String>,
        generic_context: Option<&Il2CppGenericContext>,
    ) -> (String, String) {
        let mut type_signature_parts: Vec<Il2CppTypeEnum> = Vec::new();

        let method_return_type = il2cpp.types[method_def.return_type as usize].clone();
        let return_type_c = Self::parse_type(&method_return_type, struct_name_dic, executor, metadata, il2cpp, generic_context, None);
        let return_c = if method_return_type.byref == 1 {
            type_signature_parts.push(Il2CppTypeEnum::Ptr);
            format!("{}*", return_type_c)
        } else {
            let te = Il2CppTypeEnum::from_u8(method_return_type.type_enum).unwrap_or(Il2CppTypeEnum::Void);
            type_signature_parts.push(if method_return_type.byref == 1 { Il2CppTypeEnum::Ptr } else { te });
            return_type_c
        };

        let mut param_strs = Vec::new();

        let is_static = (method_def.flags as u32 & method_attributes::STATIC) != 0;
        if !is_static {
            let byval_type = il2cpp.types[type_def.byval_type_index as usize].clone();
            let te = Il2CppTypeEnum::from_u8(byval_type.type_enum)
                .unwrap_or(Il2CppTypeEnum::Object);
            type_signature_parts.push(te);
            if type_def.is_value_type() {
                let klass_idx = byval_type.klass_index() as usize;
                let base_name = struct_name_dic.get(&klass_idx)
                    .map(|s| s.as_str())
                    .unwrap_or("Il2CppObject");
                param_strs.push(format!("{}* __this", base_name));
            } else {
                let this_type = Self::parse_type(
                    &byval_type,
                    struct_name_dic, executor, metadata, il2cpp, generic_context, None,
                );
                param_strs.push(format!("{} __this", this_type));
            }
        } else if il2cpp.version <= 24.0 {
            type_signature_parts.push(Il2CppTypeEnum::Ptr);
            param_strs.push("Il2CppObject* __this".to_string());
        }

        for j in 0..method_def.parameter_count as usize {
            let param_def = metadata.parameter_defs[method_def.parameter_start as usize + j].clone();
            let param_name = metadata.get_string_from_index(param_def.name_index)
                .unwrap_or_else(|_| "param".to_string());
            let param_type = il2cpp.types[param_def.type_index as usize].clone();
            let param_c_type = Self::parse_type(&param_type, struct_name_dic, executor, metadata, il2cpp, generic_context, None);
            let (param_c, sig_type) = if param_type.byref == 1 {
                (format!("{}*", param_c_type), Il2CppTypeEnum::Ptr)
            } else {
                let te = Il2CppTypeEnum::from_u8(param_type.type_enum).unwrap_or(Il2CppTypeEnum::Object);
                (param_c_type, te)
            };
            type_signature_parts.push(sig_type);
            param_strs.push(format!("{} {}", param_c, fix_name(&param_name)));
        }

        type_signature_parts.push(Il2CppTypeEnum::Ptr);
        param_strs.push("const MethodInfo* method".to_string());

        let signature = format!("{} {} ({});",
            return_c, fix_name(method_full_name), param_strs.join(", "));
        let type_sig = get_method_type_signature(&type_signature_parts);

        (signature, type_sig)
    }

    fn resolve_generic_type_var(
        il2cpp: &Il2Cpp,
        inst_addr: u64,
        param_num: u32,
    ) -> Option<Il2CppType> {
        if inst_addr == 0 { return None; }
        let generic_inst = il2cpp.read_generic_inst(inst_addr).ok()?;
        let pointers = il2cpp.read_ptr_array(generic_inst.type_argv, generic_inst.type_argc).ok()?;
        let pointer = *pointers.get(param_num as usize)?;
        il2cpp.get_il2cpp_type(pointer).cloned()
    }

    pub fn resolve_generic_type_var_pub(
        il2cpp: &Il2Cpp,
        inst_addr: u64,
        param_num: u32,
    ) -> Option<Il2CppType> {
        Self::resolve_generic_type_var(il2cpp, inst_addr, param_num)
    }

    fn parse_type(
        il2cpp_type: &Il2CppType,
        struct_name_dic: &HashMap<usize, String>,
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        context: Option<&Il2CppGenericContext>,
        mut hdr_ctx: Option<&mut HeaderGenCtx>,
    ) -> String {
        let te = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
        match te {
            Some(Il2CppTypeEnum::Void) => "void".to_string(),
            Some(Il2CppTypeEnum::Boolean) => "bool".to_string(),
            Some(Il2CppTypeEnum::Char) => "uint16_t".to_string(),
            Some(Il2CppTypeEnum::I1) => "int8_t".to_string(),
            Some(Il2CppTypeEnum::U1) => "uint8_t".to_string(),
            Some(Il2CppTypeEnum::I2) => "int16_t".to_string(),
            Some(Il2CppTypeEnum::U2) => "uint16_t".to_string(),
            Some(Il2CppTypeEnum::I4) => "int32_t".to_string(),
            Some(Il2CppTypeEnum::U4) => "uint32_t".to_string(),
            Some(Il2CppTypeEnum::I8) => "int64_t".to_string(),
            Some(Il2CppTypeEnum::U8) => "uint64_t".to_string(),
            Some(Il2CppTypeEnum::R4) => "float".to_string(),
            Some(Il2CppTypeEnum::R8) => "double".to_string(),
            Some(Il2CppTypeEnum::String) => "System_String_o*".to_string(),
            Some(Il2CppTypeEnum::I) => "intptr_t".to_string(),
            Some(Il2CppTypeEnum::U) => "uintptr_t".to_string(),
            Some(Il2CppTypeEnum::Object) | Some(Il2CppTypeEnum::TypedByRef) => "Il2CppObject*".to_string(),
            Some(Il2CppTypeEnum::ValueType) => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                if let Some(td) = metadata.type_defs.get(klass_idx) {
                    if td.is_enum() {
                        if let Some(elem_type) = il2cpp.types.get(td.element_type_index as usize).cloned() {
                            return Self::parse_type(&elem_type, struct_name_dic, executor, metadata, il2cpp, context, hdr_ctx);
                        }
                    }
                    if let Some(sn) = struct_name_dic.get(&klass_idx) {
                        return format!("{}_o", sn);
                    }
                }
                "Il2CppObject*".to_string()
            }
            Some(Il2CppTypeEnum::Class) => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                if let Some(sn) = struct_name_dic.get(&klass_idx) {
                    format!("{}_o*", sn)
                } else {
                    "Il2CppObject*".to_string()
                }
            }
            Some(Il2CppTypeEnum::SzArray) | Some(Il2CppTypeEnum::Array) => {
                if il2cpp_type.datapoint != 0 {
                    if let Some(element_type) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        let elem_struct_name = Self::get_il2cpp_struct_name(&element_type, struct_name_dic, il2cpp, context);
                        return format!("{}_array*", elem_struct_name);
                    }
                }
                "Il2CppArray*".to_string()
            }
            Some(Il2CppTypeEnum::GenericInst) => {
                let generic_class_ptr = il2cpp_type.generic_class();
                if generic_class_ptr != 0 {
                    if let Ok(generic_class) = il2cpp.read_generic_class(generic_class_ptr) {
                        if let Some((type_def, td_idx)) = executor.get_generic_class_type_definition(&generic_class, metadata, il2cpp) {
                            // Use specialized name from hdr_ctx if available, fallback to base name
                            let type_struct_name = if let Some(ctx) = hdr_ctx.as_deref_mut() {
                                if let Some(name) = ctx.generic_class_struct_name_dic.get(&generic_class_ptr) {
                                    let name = name.clone();
                                    // Add to newly_discovered if this is a new unique name
                                    if ctx.struct_name_hash_set.insert(name.clone()) {
                                        ctx.newly_discovered.push(generic_class_ptr);
                                    }
                                    Some(name)
                                } else {
                                    struct_name_dic.get(&td_idx).cloned()
                                }
                            } else {
                                struct_name_dic.get(&td_idx).cloned()
                            };
                            if let Some(sn) = type_struct_name {
                                if type_def.is_value_type() {
                                    if type_def.is_enum() {
                                        if let Some(elem) = il2cpp.types.get(type_def.element_type_index as usize).cloned() {
                                            return Self::parse_type(&elem, struct_name_dic, executor, metadata, il2cpp, context, None);
                                        }
                                    }
                                    return format!("{}_o", sn);
                                } else {
                                    return format!("{}_o*", sn);
                                }
                            }
                        }
                    }
                }
                "Il2CppObject*".to_string()
            }
            Some(Il2CppTypeEnum::Var) => {
                if let Some(ctx) = context {
                    let generic_param = executor.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp);
                    if let Some(gp) = generic_param {
                        if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.class_inst, gp.num as u32) {
                            return Self::parse_type(&resolved, struct_name_dic, executor, metadata, il2cpp, None, None);
                        }
                    }
                }
                "Il2CppObject*".to_string()
            }
            Some(Il2CppTypeEnum::MVar) => {
                if let Some(ctx) = context {
                    let generic_param = executor.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp);
                    if let Some(gp) = generic_param {
                        // C# issue #687: if method_inst == 0 && class_inst != 0, fall back to VAR
                        if ctx.method_inst == 0 && ctx.class_inst != 0 {
                            if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.class_inst, gp.num as u32) {
                                return Self::parse_type(&resolved, struct_name_dic, executor, metadata, il2cpp, None, None);
                            }
                        } else {
                            if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.method_inst, gp.num as u32) {
                                return Self::parse_type(&resolved, struct_name_dic, executor, metadata, il2cpp, None, None);
                            }
                        }
                    }
                }
                "Il2CppObject*".to_string()
            }
            Some(Il2CppTypeEnum::Ptr) => {
                if il2cpp_type.datapoint != 0 {
                    if let Some(ori_type) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        let inner = Self::parse_type(&ori_type, struct_name_dic, executor, metadata, il2cpp, context, hdr_ctx);
                        return format!("{}*", inner);
                    }
                }
                "void*".to_string()
            }
            _ => "Il2CppObject*".to_string(),
        }
    }

    fn build_struct_name_dic(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> HashMap<usize, String> {
        let mut dic = HashMap::new();
        let mut name_set: HashSet<String> = HashSet::new();

        let image_defs = metadata.image_defs.clone();
        for image_def in &image_defs {
            let type_end = image_def.type_start as usize + image_def.type_count as usize;
            for type_index in image_def.type_start as usize..type_end {
                let type_def = metadata.type_defs[type_index].clone();
                let type_name = executor.get_type_def_name(&type_def, type_index, metadata, il2cpp, true, true);
                let struct_name = fix_name(&type_name);
                let unique = get_unique_name(&struct_name, &mut name_set);
                dic.insert(type_index, unique);
            }
        }
        dic
    }

    pub fn build_struct_name_dic_pub(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> HashMap<usize, String> {
        Self::build_struct_name_dic(executor, metadata, il2cpp)
    }

    fn build_type_def_image_names(metadata: &Metadata) -> HashMap<usize, String> {
        let mut dic = HashMap::new();
        let image_defs = metadata.image_defs.clone();
        for image_def in &image_defs {
            let image_name = metadata.get_string_from_index(image_def.name_index).unwrap_or_default();
            let type_end = image_def.type_start as usize + image_def.type_count as usize;
            for type_index in image_def.type_start as usize..type_end {
                dic.insert(type_index, image_name.clone());
            }
        }
        dic
    }

    pub fn build_type_def_image_names_pub(metadata: &Metadata) -> HashMap<usize, String> {
        Self::build_type_def_image_names(metadata)
    }

    fn classify_types_for_header(
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        struct_name_dic: &HashMap<usize, String>,
        generic_class_struct_name_dic: &HashMap<u64, String>,
    ) -> crate::output::cpp_type_model::CppTypeGroupRegistry {
        use crate::output::cpp_type_model::{CppTypeGroup, CppTypeGroupRegistry};
        let mut registry = CppTypeGroupRegistry::new();

        let resolve_type_to_name = |type_index: usize| -> Option<String> {
            let t = il2cpp.types.get(type_index)?;
            Self::resolve_il2cpp_type_name(t, il2cpp, struct_name_dic, generic_class_struct_name_dic)
        };

        let generic_method_defs: HashSet<usize> = il2cpp
            .method_definition_method_specs
            .keys()
            .copied()
            .collect();

        for (method_index, method_def) in metadata.method_defs.iter().enumerate() {
            let is_generic = method_def.generic_container_index >= 0
                || generic_method_defs.contains(&method_index);
            let group = if is_generic {
                CppTypeGroup::TypesFromGenericMethods
            } else {
                CppTypeGroup::TypesFromMethods
            };

            if let Some(name) = resolve_type_to_name(method_def.return_type as usize) {
                registry.assign(&name, group);
            }
            for j in 0..method_def.parameter_count as usize {
                let p_idx = method_def.parameter_start as usize + j;
                if let Some(param_def) = metadata.parameter_defs.get(p_idx) {
                    if let Some(name) = resolve_type_to_name(param_def.type_index as usize) {
                        if registry.group_of(&name) != Some(CppTypeGroup::TypesFromMethods) {
                            registry.assign(&name, group);
                        }
                    }
                }
            }
        }

        for (_, dic) in metadata.metadata_usage_dic.iter() {
            for (_, encoded) in dic.iter() {
                let shift = crate::il2cpp::enums::Il2CppMetadataUsage::encoded_index_shift(il2cpp.version);
                let mask = (1u32 << shift) - 1;
                let usage_tag = encoded & mask;
                let idx = (encoded >> shift) as usize;
                match usage_tag {
                    1 => {
                        if let Some(name) = struct_name_dic.get(&idx) {
                            if registry.group_of(name).is_none() {
                                registry.assign(name, CppTypeGroup::TypesFromUsages);
                            }
                        }
                    }
                    2 => {
                        if let Some(name) = resolve_type_to_name(idx) {
                            if registry.group_of(&name).is_none() {
                                registry.assign(&name, CppTypeGroup::TypesFromUsages);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for name in struct_name_dic.values() {
            if registry.group_of(name).is_none() {
                registry.assign(name, CppTypeGroup::UnusedConcreteTypes);
            }
        }
        for name in generic_class_struct_name_dic.values() {
            if registry.group_of(name).is_none() {
                registry.assign(name, CppTypeGroup::UnusedConcreteTypes);
            }
        }

        registry
    }

    fn resolve_il2cpp_type_name(
        il2cpp_type: &Il2CppType,
        il2cpp: &Il2Cpp,
        struct_name_dic: &HashMap<usize, String>,
        generic_class_struct_name_dic: &HashMap<u64, String>,
    ) -> Option<String> {
        let te = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum)?;
        match te {
            Il2CppTypeEnum::Class | Il2CppTypeEnum::ValueType => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                struct_name_dic.get(&klass_idx).cloned()
            }
            Il2CppTypeEnum::GenericInst => {
                let gc_ptr = il2cpp_type.generic_class();
                if gc_ptr != 0 {
                    if let Some(n) = generic_class_struct_name_dic.get(&gc_ptr) {
                        return Some(n.clone());
                    }
                }
                let klass_idx = il2cpp_type.klass_index() as usize;
                struct_name_dic.get(&klass_idx).cloned()
            }
            Il2CppTypeEnum::Ptr | Il2CppTypeEnum::SzArray | Il2CppTypeEnum::Array => {
                if il2cpp_type.datapoint != 0 {
                    let inner = il2cpp.types.get(il2cpp_type.datapoint as usize)?;
                    Self::resolve_il2cpp_type_name(inner, il2cpp, struct_name_dic, generic_class_struct_name_dic)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn add_metadata_usages(
        script: &mut ScriptJson,
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        struct_name_dic: &HashMap<usize, String>,
        _type_def_image_names: &HashMap<usize, String>,
        config: &crate::config::Config,
    ) {
        if metadata.metadata_usage_dic.is_empty() { return; }
        let usage_dic = metadata.metadata_usage_dic.clone();

        for (usage_type, entries) in &usage_dic {
            for (dest_index, source_index) in entries {
                let dest = *dest_index as usize;
                if dest >= il2cpp.metadata_usages.len() { continue; }
                let address = il2cpp.metadata_usages[dest];
                if address == 0 { continue; }
                let rva = il2cpp.get_rva(address);
                let src = *source_index as usize;

                match *usage_type {
                    1 => {
                        if src < il2cpp.types.len() {
                            let type_ref = il2cpp.types[src].clone();
                            let type_name = executor.get_type_name(&type_ref, metadata, il2cpp, true, false);
                            let sig = if let Some(sn) = Self::get_struct_name_for_type(&type_ref, struct_name_dic, metadata) {
                                format!("{}_c*", fix_name(&sn))
                            } else {
                                "Il2CppClass*".to_string()
                            };
                            script.script_metadata.push(ScriptMetadata {
                                address: rva,
                                name: format!("{}_TypeInfo", type_name),
                                signature: Some(sig.clone()),
                            });
                            if config.enhanced_ida_metadata {
                                script.type_info_pointers.push(ScriptTypeInfo {
                                    address: rva,
                                    name: format!("{}_TypeInfo", fix_name(&type_name)),
                                    type_str: sig,
                                    dotnet_type: Some(type_name.clone()),
                                });
                            }
                        }
                    }
                    2 => {
                        if src < il2cpp.types.len() {
                            let type_ref = il2cpp.types[src].clone();
                            let type_name = executor.get_type_name(&type_ref, metadata, il2cpp, true, false);
                            script.script_metadata.push(ScriptMetadata {
                                address: rva,
                                name: format!("{}_var", type_name),
                                signature: Some("Il2CppType*".to_string()),
                            });
                            if config.enhanced_ida_metadata {
                                script.type_ref_pointers.push(ScriptTypeInfo {
                                    address: rva,
                                    name: format!("{}_TypeRef", fix_name(&type_name)),
                                    type_str: "Il2CppType*".to_string(),
                                    dotnet_type: Some(type_name.clone()),
                                });
                            }
                        }
                    }
                    3 => {
                        if let Some(method_def) = metadata.method_defs.get(src).cloned() {
                            if let Some(type_def) = metadata.type_defs.get(method_def.declaring_type as usize).cloned() {
                                let td_idx = method_def.declaring_type as usize;
                                let type_name = executor.get_type_def_name(&type_def, td_idx, metadata, il2cpp, true, true);
                                let method_name = metadata.get_string_from_index(method_def.name_index as i32)
                                    .unwrap_or_else(|_| "?".to_string());
                                let image_name = _type_def_image_names.get(&td_idx).cloned().unwrap_or_default();
                                let method_pointer = il2cpp.get_method_pointer(&image_name, &method_def);
                                let method_address = if method_pointer > 0 { il2cpp.get_rva(method_pointer) } else { 0 };
                                script.script_metadata_methods.push(ScriptMetadataMethod {
                                    address: rva,
                                    name: format!("Method${}.{}()", type_name, method_name),
                                    method_address,
                                });
                            }
                        }
                    }
                    4 => {
                        if src < metadata.field_refs.len() {
                            let field_ref = metadata.field_refs[src].clone();
                            let il2cpp_type = il2cpp.types[field_ref.type_index as usize].clone();
                            let type_name = executor.get_type_name(&il2cpp_type, metadata, il2cpp, true, false);
                            let klass_idx = il2cpp_type.klass_index() as usize;
                            if let Some(td) = metadata.type_defs.get(klass_idx) {
                                let field_idx = td.field_start as usize + field_ref.field_index as usize;
                                if let Some(fd) = metadata.field_defs.get(field_idx) {
                                    let field_name = metadata.get_string_from_index(fd.name_index).unwrap_or_default();
                                    let label = format!("Field${}.{}", type_name, field_name);
                                    script.script_metadata.push(ScriptMetadata {
                                        address: rva,
                                        name: label.clone(),
                                        signature: None,
                                    });
                                    if config.enhanced_ida_metadata {
                                        script.field_infos.push(ScriptFieldInfo {
                                            address: rva,
                                            name: fix_name(&format!("{}.{}_Field", type_name, field_name)),
                                            value: format!("{}.{}", type_name, field_name),
                                        });
                                    }
                                }
                            }
                        }
                    }
                    5 => {
                        if let Ok(string_literal) = metadata.get_string_literal_from_index(src) {
                            let safe_id: String = string_literal.chars()
                                .take(32)
                                .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
                                .collect();
                            script.script_strings.push(ScriptString {
                                address: rva,
                                value: string_literal,
                                name: Some(format!("StringLiteral_{}", safe_id)),
                            });
                        }
                    }
                    6 => {
                        if src < il2cpp.method_specs.len() {
                            let _method_spec = il2cpp.method_specs[src].clone();
                            let (spec_type_name, spec_method_name) = executor.get_method_spec_name(src, metadata, il2cpp, true);
                            let method_address = il2cpp.method_spec_generic_method_pointers
                                .get(&src).copied()
                                .filter(|p| *p > 0)
                                .map(|p| il2cpp.get_rva(p))
                                .unwrap_or(0);
                            script.script_metadata_methods.push(ScriptMetadataMethod {
                                address: rva,
                                name: format!("Method${}.{}()", spec_type_name, spec_method_name),
                                method_address,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn scan_v27_metadata_usages(
        script: &mut ScriptJson,
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        struct_name_dic: &HashMap<usize, String>,
        type_def_image_names: &HashMap<usize, String>,
        config: &crate::config::Config,
    ) {
        let pointer_size = if il2cpp.is_32bit { 4u64 } else { 8u64 };
        let data_sections = il2cpp.data_sections.clone();

        for sec in &data_sections {
            let sec_end = std::cmp::min(sec.offset_end, il2cpp.stream.len() as u64).saturating_sub(pointer_size);
            let mut pos = sec.offset;
            while pos < sec_end {
                let metadata_value = if il2cpp.is_32bit {
                    il2cpp.stream.peek_u32_at(pos).unwrap_or(0) as u64
                } else {
                    il2cpp.stream.peek_u64_at(pos).unwrap_or(0)
                };
                pos += pointer_size;

                if metadata_value >= u32::MAX as u64 { continue; }
                let encoded_token = metadata_value as u32;
                let usage = (encoded_token & 0xE0000000) >> 29;
                if usage == 0 || usage > 7 { continue; }
                let decoded_index = (encoded_token & 0x1FFFFFFE) >> 1;
                let expected = ((usage << 29) | (decoded_index << 1)) + 1;
                if metadata_value != expected as u64 { continue; }

                let addr = pos - pointer_size;
                let va = il2cpp.map_rtva(addr);
                if va == 0 { continue; }
                let rva = il2cpp.get_rva(va);

                match usage {
                    1 => {
                        if (decoded_index as usize) < il2cpp.types.len() {
                            let type_ref = il2cpp.types[decoded_index as usize].clone();
                            let type_name = executor.get_type_name(&type_ref, metadata, il2cpp, true, false);
                            let sig = if let Some(sn) = Self::get_struct_name_for_type(&type_ref, struct_name_dic, metadata) {
                                if sn.ends_with("_array") {
                                    "Il2CppClass*".to_string()
                                } else {
                                    format!("{}_c*", fix_name(&sn))
                                }
                            } else {
                                "Il2CppClass*".to_string()
                            };
                            script.script_metadata.push(ScriptMetadata {
                                address: rva,
                                name: format!("{}_TypeInfo", type_name),
                                signature: Some(sig.clone()),
                            });
                            if config.enhanced_ida_metadata {
                                script.type_info_pointers.push(ScriptTypeInfo {
                                    address: rva,
                                    name: format!("{}_TypeInfo", fix_name(&type_name)),
                                    type_str: sig,
                                    dotnet_type: Some(type_name.clone()),
                                });
                            }
                        }
                    }
                    2 => {
                        if (decoded_index as usize) < il2cpp.types.len() {
                            let type_ref = il2cpp.types[decoded_index as usize].clone();
                            let type_name = executor.get_type_name(&type_ref, metadata, il2cpp, true, false);
                            script.script_metadata.push(ScriptMetadata {
                                address: rva,
                                name: format!("{}_var", type_name),
                                signature: Some("Il2CppType*".to_string()),
                            });
                            if config.enhanced_ida_metadata {
                                script.type_ref_pointers.push(ScriptTypeInfo {
                                    address: rva,
                                    name: format!("{}_TypeRef", fix_name(&type_name)),
                                    type_str: "Il2CppType*".to_string(),
                                    dotnet_type: Some(type_name.clone()),
                                });
                            }
                        }
                    }
                    3 => {
                        if let Some(method_def) = metadata.method_defs.get(decoded_index as usize).cloned() {
                            if let Some(type_def) = metadata.type_defs.get(method_def.declaring_type as usize).cloned() {
                                let td_idx = method_def.declaring_type as usize;
                                let type_name = executor.get_type_def_name(&type_def, td_idx, metadata, il2cpp, true, true);
                                let method_name = metadata.get_string_from_index(method_def.name_index as i32)
                                    .unwrap_or_else(|_| "?".to_string());
                                let image_name = type_def_image_names.get(&td_idx).cloned().unwrap_or_default();
                                let method_pointer = il2cpp.get_method_pointer(&image_name, &method_def);
                                let method_address = if method_pointer > 0 { il2cpp.get_rva(method_pointer) } else { 0 };
                                script.script_metadata_methods.push(ScriptMetadataMethod {
                                    address: rva,
                                    name: format!("Method${}.{}()", type_name, method_name),
                                    method_address,
                                });
                            }
                        }
                    }
                    4 => {
                        if (decoded_index as usize) < metadata.field_refs.len() {
                            let field_ref = metadata.field_refs[decoded_index as usize].clone();
                            if (field_ref.type_index as usize) >= il2cpp.types.len() { continue; }
                            let il2cpp_type = il2cpp.types[field_ref.type_index as usize].clone();
                            let type_name = executor.get_type_name(&il2cpp_type, metadata, il2cpp, true, false);
                            let klass_idx = il2cpp_type.klass_index() as usize;
                            if let Some(td) = metadata.type_defs.get(klass_idx) {
                                let field_idx = td.field_start as usize + field_ref.field_index as usize;
                                if let Some(fd) = metadata.field_defs.get(field_idx) {
                                    let field_name = metadata.get_string_from_index(fd.name_index).unwrap_or_default();
                                    script.script_metadata.push(ScriptMetadata {
                                        address: rva,
                                        name: format!("Field${}.{}", type_name, field_name),
                                        signature: None,
                                    });
                                    if config.enhanced_ida_metadata {
                                        script.field_infos.push(ScriptFieldInfo {
                                            address: rva,
                                            name: fix_name(&format!("{}.{}_Field", type_name, field_name)),
                                            value: format!("{}.{}", type_name, field_name),
                                        });
                                    }
                                }
                            }
                        }
                    }
                    5 => {
                        if let Ok(string_literal) = metadata.get_string_literal_from_index(decoded_index as usize) {
                            let safe_id: String = string_literal.chars()
                                .take(32)
                                .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
                                .collect();
                            script.script_strings.push(ScriptString {
                                address: rva,
                                value: string_literal,
                                name: Some(format!("StringLiteral_{}", safe_id)),
                            });
                        }
                    }
                    6 => {
                        if (decoded_index as usize) < il2cpp.method_specs.len() {
                            let (spec_type_name, spec_method_name) = executor.get_method_spec_name(decoded_index as usize, metadata, il2cpp, true);
                            let method_address = il2cpp.method_spec_generic_method_pointers
                                .get(&(decoded_index as usize)).copied()
                                .filter(|p| *p > 0)
                                .map(|p| il2cpp.get_rva(p))
                                .unwrap_or(0);
                            script.script_metadata_methods.push(ScriptMetadataMethod {
                                address: rva,
                                name: format!("Method${}.{}()", spec_type_name, spec_method_name),
                                method_address,
                            });
                        }
                    }
                    7 => {
                        if (decoded_index as usize) < metadata.field_refs.len() {
                            let field_ref = metadata.field_refs[decoded_index as usize].clone();
                            if (field_ref.type_index as usize) >= il2cpp.types.len() { continue; }
                            let il2cpp_type = il2cpp.types[field_ref.type_index as usize].clone();
                            let type_name = executor.get_type_name(&il2cpp_type, metadata, il2cpp, true, false);
                            let klass_idx = il2cpp_type.klass_index() as usize;
                            if let Some(td) = metadata.type_defs.get(klass_idx) {
                                let field_idx = td.field_start as usize + field_ref.field_index as usize;
                                if let Some(fd) = metadata.field_defs.get(field_idx) {
                                    let field_name = metadata.get_string_from_index(fd.name_index).unwrap_or_default();
                                    let field_path = format!("{}.{}", type_name, field_name);
                                    script.script_metadata.push(ScriptMetadata {
                                        address: rva,
                                        name: format!("FieldRva${}", field_path),
                                        signature: None,
                                    });
                                    let mut value = field_path.clone();
                                    if config.dump_field_rva_data {
                                        if let Some(fdv) = metadata.get_field_default_value(field_idx as i32) {
                                            if fdv.data_index >= 0 {
                                                let meta_off = metadata.get_default_value_offset(fdv.data_index);
                                                if let Some(bytes) = crate::il2cpp::field_layout::read_metadata_bytes(
                                                    metadata, meta_off, config.max_field_rva_dump_bytes.min(256),
                                                ) {
                                                    value = format!(
                                                        "{} hex={}",
                                                        field_path,
                                                        crate::output::static_field_exporter::bytes_to_hex_preview(&bytes, 128),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                    script.field_rvas.push(ScriptFieldInfo {
                                        address: rva,
                                        name: format!("{}_FieldRva", field_path.replace('.', "_")),
                                        value,
                                    });
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn get_struct_name_for_type(
        il2cpp_type: &Il2CppType,
        struct_name_dic: &HashMap<usize, String>,
        _metadata: &Metadata,
    ) -> Option<String> {
        let te = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum)?;
        match te {
            Il2CppTypeEnum::Class | Il2CppTypeEnum::ValueType => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                struct_name_dic.get(&klass_idx).cloned()
            }
            _ => None,
        }
    }

    fn get_il2cpp_struct_name(
        il2cpp_type: &Il2CppType,
        struct_name_dic: &HashMap<usize, String>,
        il2cpp: &Il2Cpp,
        context: Option<&Il2CppGenericContext>,
    ) -> String {
        let te = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
        match te {
            Some(Il2CppTypeEnum::Void) | Some(Il2CppTypeEnum::Boolean) | Some(Il2CppTypeEnum::Char) |
            Some(Il2CppTypeEnum::I1) | Some(Il2CppTypeEnum::U1) | Some(Il2CppTypeEnum::I2) |
            Some(Il2CppTypeEnum::U2) | Some(Il2CppTypeEnum::I4) | Some(Il2CppTypeEnum::U4) |
            Some(Il2CppTypeEnum::I8) | Some(Il2CppTypeEnum::U8) | Some(Il2CppTypeEnum::R4) |
            Some(Il2CppTypeEnum::R8) | Some(Il2CppTypeEnum::String) | Some(Il2CppTypeEnum::TypedByRef) |
            Some(Il2CppTypeEnum::I) | Some(Il2CppTypeEnum::U) | Some(Il2CppTypeEnum::Object) |
            Some(Il2CppTypeEnum::ValueType) | Some(Il2CppTypeEnum::Class) => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                struct_name_dic.get(&klass_idx).cloned().unwrap_or_else(|| "System_Object".to_string())
            }
            Some(Il2CppTypeEnum::Ptr) => {
                if il2cpp_type.datapoint != 0 {
                    if let Some(ori_type) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        return Self::get_il2cpp_struct_name(&ori_type, struct_name_dic, il2cpp, context);
                    }
                }
                "System_Object".to_string()
            }
            Some(Il2CppTypeEnum::SzArray) => {
                if il2cpp_type.datapoint != 0 {
                    if let Some(element_type) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        let elem_name = Self::get_il2cpp_struct_name(&element_type, struct_name_dic, il2cpp, context);
                        return format!("{}_array", elem_name);
                    }
                }
                "System_Object".to_string()
            }
            Some(Il2CppTypeEnum::Array) => {
                if il2cpp_type.datapoint != 0 {
                    if let Some(element_type) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        let elem_name = Self::get_il2cpp_struct_name(&element_type, struct_name_dic, il2cpp, context);
                        return format!("{}_array", elem_name);
                    }
                }
                "System_Object".to_string()
            }
            Some(Il2CppTypeEnum::GenericInst) => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                struct_name_dic.get(&klass_idx).cloned().unwrap_or_else(|| "System_Object".to_string())
            }
            Some(Il2CppTypeEnum::Var) => {
                if let Some(ctx) = context {
                    if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.class_inst, il2cpp_type.datapoint as u32) {
                        return Self::get_il2cpp_struct_name(&resolved, struct_name_dic, il2cpp, None);
                    }
                }
                "System_Object".to_string()
            }
            Some(Il2CppTypeEnum::MVar) => {
                if let Some(ctx) = context {
                    if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.method_inst, il2cpp_type.datapoint as u32) {
                        return Self::get_il2cpp_struct_name(&resolved, struct_name_dic, il2cpp, None);
                    }
                }
                "System_Object".to_string()
            }
            _ => "System_Object".to_string(),
        }
    }

    fn parse_array_class_struct(
        buf: &mut String,
        element_type: &Il2CppType,
        struct_name_dic: &HashMap<usize, String>,
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        context: Option<&Il2CppGenericContext>,
    ) {
        let struct_name = Self::get_il2cpp_struct_name(element_type, struct_name_dic, il2cpp, context);
        let element_c_type = Self::parse_type(element_type, struct_name_dic, executor, metadata, il2cpp, context, None);
        writeln!(buf, "struct {}_array {{", struct_name).ok();
        writeln!(buf, "\tIl2CppObject obj;").ok();
        writeln!(buf, "\tIl2CppArrayBounds *bounds;").ok();
        writeln!(buf, "\til2cpp_array_size_t max_length;").ok();
        writeln!(buf, "\t{} m_Items[65535];", element_c_type).ok();
        writeln!(buf, "}};").ok();
    }

    fn build_string_literal_json(metadata: &Metadata) -> Result<String> {
        use rayon::prelude::*;
        let entries: Vec<StringLiteralEntry> = (0..metadata.string_literals.len())
            .into_par_iter()
            .filter_map(|i| {
                metadata
                    .get_string_literal_from_index(i)
                    .ok()
                    .map(|value| StringLiteralEntry { index: i, value })
            })
            .collect();
        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        Ok(json)
    }

    fn build_header(
        executor: &mut Il2CppExecutor,
        metadata: &mut Metadata,
        il2cpp: &mut Il2Cpp,
        config: &crate::config::Config,
    ) -> Result<String> {
        let version = il2cpp.version;
        let version_header = match header_constants::get_version_header(version) {
            Some(h) => h,
            None => {
                eprintln!("WARNING: IL2CPP version [{version}] does not support generating .h files");
                return Ok(String::new());
            }
        };

        use rayon::prelude::*;

        let struct_name_dic = Self::build_struct_name_dic(executor, metadata, il2cpp);
        let type_def_image_names = Self::build_type_def_image_names(metadata);

        let mut array_classes: HashMap<String, String> = HashMap::new();

        // Build genericClassStructNameDic (C# lines 58-73)
        let mut generic_class_struct_name_dic: HashMap<u64, String> = HashMap::new();
        let mut name_generic_class_dic: HashMap<String, Il2CppType> = HashMap::new();
        let mut generic_class_list: Vec<u64> = Vec::new();
        let struct_name_hash_set: HashSet<String> = struct_name_dic.values().cloned().collect();
        {
            let types_clone = il2cpp.types.clone();
            for il2cpp_type in &types_clone {
                let te = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
                if te != Some(Il2CppTypeEnum::GenericInst) { continue; }
                let generic_class_ptr = il2cpp_type.generic_class();
                if generic_class_ptr == 0 { continue; }
                let generic_class = match il2cpp.read_generic_class(generic_class_ptr) {
                    Ok(gc) => gc,
                    Err(_) => continue,
                };
                let type_def_result = executor.get_generic_class_type_definition(&generic_class, metadata, il2cpp);
                let (type_def, td_idx) = match type_def_result {
                    Some(td) => td,
                    None => continue,
                };
                let type_base_name = match struct_name_dic.get(&td_idx) {
                    Some(n) => n.clone(),
                    None => continue,
                };
                let type_to_replace_name = fix_name(&executor.get_type_def_name(&type_def, td_idx, metadata, il2cpp, true, true));
                let type_replace_name = fix_name(&executor.get_type_name(il2cpp_type, metadata, il2cpp, true, false));
                let type_struct_name = type_base_name.replace(&type_to_replace_name, &type_replace_name);
                name_generic_class_dic.insert(type_struct_name.clone(), il2cpp_type.clone());
                generic_class_struct_name_dic.insert(generic_class_ptr, type_struct_name);
            }
        }

        // Parallel StructInfo for non-generic type definitions (map-reduce array class fragments).
        let type_defs = metadata.type_defs.clone();
        let base_results: Vec<(Option<StructInfo>, HashMap<String, String>)> = type_defs
            .par_iter()
            .enumerate()
            .map(|(type_index, type_def)| {
                let type_name = match struct_name_dic.get(&type_index) {
                    Some(n) => n.clone(),
                    None => return (None, HashMap::new()),
                };

                let mut local_exec = Il2CppExecutor::new_for_worker(executor);
                let mut info = StructInfo {
                    type_name,
                    is_value_type: type_def.is_value_type(),
                    parent: None,
                    fields: Vec::new(),
                    static_fields: Vec::new(),
                    vtable_methods: Vec::new(),
                    rgctxs: Vec::new(),
                };
                let mut local_arrays = HashMap::new();

                Self::add_parent(il2cpp, type_def, &struct_name_dic, metadata, &mut info);
                Self::add_fields(
                    &mut local_exec,
                    metadata,
                    il2cpp,
                    type_def,
                    &struct_name_dic,
                    &mut info,
                    &mut local_arrays,
                    None,
                    None,
                );
                Self::add_vtable_methods(metadata, il2cpp, type_def, &mut info);
                Self::add_rgctx(
                    &mut local_exec,
                    metadata,
                    il2cpp,
                    type_def,
                    type_index,
                    &type_def_image_names,
                    &mut info,
                );

                (Some(info), local_arrays)
            })
            .collect();

        let mut struct_info_list: Vec<StructInfo> = Vec::with_capacity(base_results.len());
        for (info_opt, local_arrays) in base_results {
            if let Some(info) = info_opt {
                struct_info_list.push(info);
            }
            for (name, frag) in local_arrays {
                array_classes.entry(name).or_insert(frag);
            }
        }

        // Process generic class instances using fixpoint loop
        // C# uses a self-expanding for loop: for(int i=0; i<genericClassList.Count; i++)
        // where ParseType can add new entries during iteration.
        let mut hdr_ctx = HeaderGenCtx {
            generic_class_struct_name_dic: generic_class_struct_name_dic.clone(),
            struct_name_hash_set: struct_name_hash_set,
            newly_discovered: Vec::new(),
        };

        // Build initial list from all GENERICINST types in il2cpp.types
        for il2cpp_type in il2cpp.types.clone().iter() {
            let te = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);
            if te != Some(Il2CppTypeEnum::GenericInst) { continue; }
            let generic_class_ptr = il2cpp_type.generic_class();
            if generic_class_ptr == 0 { continue; }
            if !hdr_ctx.generic_class_struct_name_dic.contains_key(&generic_class_ptr) { continue; }
            let type_struct_name = hdr_ctx.generic_class_struct_name_dic[&generic_class_ptr].clone();
            if !hdr_ctx.struct_name_hash_set.insert(type_struct_name) { continue; }
            generic_class_list.push(generic_class_ptr);
        }

        // Fixpoint loop: process generic classes, discovering new ones as field types are parsed
        let mut processed = 0;
        loop {
            let current_len = generic_class_list.len();
            if processed >= current_len { break; }

            for idx in processed..current_len {
                let pointer = generic_class_list[idx];
                let generic_class = match il2cpp.read_generic_class(pointer) {
                    Ok(gc) => gc,
                    Err(_) => continue,
                };
                let type_def_result = executor.get_generic_class_type_definition(&generic_class, metadata, il2cpp);
                let (type_def, _td_idx) = match type_def_result {
                    Some(td) => td,
                    None => continue,
                };
                let type_struct_name = match hdr_ctx.generic_class_struct_name_dic.get(&pointer) {
                    Some(n) => n.clone(),
                    None => continue,
                };
                let mut info = StructInfo {
                    type_name: type_struct_name,
                    is_value_type: type_def.is_value_type(),
                    parent: None,
                    fields: Vec::new(),
                    static_fields: Vec::new(),
                    vtable_methods: Vec::new(),
                    rgctxs: Vec::new(),
                };
                let context = Il2CppGenericContext {
                    class_inst: generic_class.context.class_inst,
                    method_inst: generic_class.context.method_inst,
                };
                Self::add_parent(il2cpp, &type_def, &struct_name_dic, metadata, &mut info);
                Self::add_fields(
                    executor,
                    metadata,
                    il2cpp,
                    &type_def,
                    &struct_name_dic,
                    &mut info,
                    &mut array_classes,
                    Some(&context),
                    Some(&mut hdr_ctx),
                );
                Self::add_vtable_methods(metadata, il2cpp, &type_def, &mut info);
                struct_info_list.push(info);
            }

            processed = current_len;

            // Drain newly discovered generic classes into the main list
            let new_ptrs: Vec<u64> = hdr_ctx.newly_discovered.drain(..).collect();
            generic_class_list.extend(new_ptrs);
        }

        let type_group_registry = Self::classify_types_for_header(
            metadata,
            il2cpp,
            &struct_name_dic,
            &generic_class_struct_name_dic,
        );

        let header_struct = if config.use_topological_sort {
            let type_decls: Vec<crate::output::cpp_ast::CppTypeDecl> = struct_info_list.iter().map(|info| {
                crate::output::cpp_ast::CppTypeDecl {
                    name: info.type_name.clone(),
                    is_value_type: info.is_value_type,
                    parent: info.parent.clone(),
                    instance_fields: info.fields.iter().map(|f| crate::output::cpp_ast::CppField {
                        type_name: f.field_type_name.clone(),
                        field_name: f.field_name.clone(),
                        is_value_type: f.is_value_type,
                        is_custom_type: f.is_custom_type,
                    }).collect(),
                    static_fields: info.static_fields.iter().map(|f| crate::output::cpp_ast::CppField {
                        type_name: f.field_type_name.clone(),
                        field_name: f.field_name.clone(),
                        is_value_type: f.is_value_type,
                        is_custom_type: f.is_custom_type,
                    }).collect(),
                    vtable: info.vtable_methods.iter().map(|v| crate::output::cpp_ast::CppVTableEntry {
                        method_name: v.as_ref().map(|m| m.method_name.clone()),
                    }).collect(),
                    rgctxs: info.rgctxs.iter().map(|r| crate::output::cpp_ast::CppRGCTXEntry {
                        rgctx_type: r.rgctx_type,
                        type_name: r.type_name.clone(),
                        class_name: r.class_name.clone(),
                        method_name: r.method_name.clone(),
                    }).collect(),
                }
            }).collect();

            let layout = crate::output::cpp_type_dependency_graph::CppCompilerLayout::from_str(&config.compiler_layout);
            let mut emitter = crate::output::cpp_ast::CppHeaderEmitter::new(layout, il2cpp.is_32bit, il2cpp.is_pe);
            match emitter.emit_all_with_groups(&type_decls, Some(&type_group_registry)) {
                Ok(out) => out,
                Err(e) => {
                    eprintln!("WARNING: {e}. Falling back to recursive emission order.");
                    let struct_info_by_name: HashMap<String, usize> = struct_info_list.iter().enumerate()
                        .map(|(i, info)| (format!("{}_o", info.type_name), i))
                        .collect();
                    let mut forward_declared: HashSet<String> = HashSet::new();
                    let mut struct_cache: HashSet<usize> = HashSet::new();
                    let mut legacy_buf = String::with_capacity(1 << 18);
                    for i in 0..struct_info_list.len() {
                        Self::recursion_struct_info(i, &struct_info_list, &struct_info_by_name, &mut struct_cache, &mut forward_declared, &mut legacy_buf, il2cpp.is_32bit, il2cpp.is_pe, &config.compiler_layout);
                    }
                    legacy_buf
                }
            }
        } else {
            let struct_info_by_name: HashMap<String, usize> = struct_info_list.iter().enumerate()
                .map(|(i, info)| (format!("{}_o", info.type_name), i))
                .collect();

            let mut forward_declared: HashSet<String> = HashSet::new();
            let mut struct_cache: HashSet<usize> = HashSet::new();
            let mut legacy_buf = String::with_capacity(1 << 18);
            for i in 0..struct_info_list.len() {
                Self::recursion_struct_info(i, &struct_info_list, &struct_info_by_name, &mut struct_cache, &mut forward_declared, &mut legacy_buf, il2cpp.is_32bit, il2cpp.is_pe, &config.compiler_layout);
            }
            legacy_buf
        };

        // Parallel MethodInfo header fragments (dedupe by generic method pointer, first wins).
        let method_info_jobs: Vec<(usize, String)> = metadata
            .image_defs
            .iter()
            .flat_map(|image_def| {
                let image_name = metadata
                    .get_string_from_index(image_def.name_index)
                    .unwrap_or_default();
                let type_end = image_def.type_start as usize + image_def.type_count as usize;
                (image_def.type_start as usize..type_end)
                    .map(|type_def_index| (type_def_index, image_name.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();

        let method_info_chunks: Vec<Vec<(u64, String)>> = method_info_jobs
            .par_iter()
            .map(|(type_def_index, image_name)| {
                let struct_type_name = match struct_name_dic.get(type_def_index) {
                    Some(n) => n.clone(),
                    None => return Vec::new(),
                };
                let type_def = &metadata.type_defs[*type_def_index];
                let mut local_exec = Il2CppExecutor::new_for_worker(executor);
                let mut local = Vec::new();
                let method_end = type_def.method_start as usize + type_def.method_count as usize;
                for method_index in type_def.method_start as usize..method_end {
                    let method_def = metadata.method_defs[method_index].clone();
                    let Some(spec_indices) = il2cpp.method_definition_method_specs.get(&method_index) else {
                        continue;
                    };
                    for spec_idx in spec_indices {
                        if *spec_idx >= il2cpp.method_specs.len() {
                            continue;
                        }
                        // Note: do NOT filter on method_index_index < 0 here.
                        // C# generates MethodInfo for ALL method specs with genericMethodPointer > 0,
                        // including those with only class-level generics (method_index_index == -1).
                        let generic_method_pointer = il2cpp
                            .method_spec_generic_method_pointers
                            .get(spec_idx)
                            .copied()
                            .unwrap_or(0);
                        if generic_method_pointer == 0 {
                            continue;
                        }
                        let method_info_rva = il2cpp.get_rva(generic_method_pointer);
                        let method_info_name = format!("MethodInfo_{:X}", method_info_rva);
                        let method_rgctxs = Self::collect_rgctx_info_for_method(
                            &mut local_exec,
                            metadata,
                            il2cpp,
                            image_name,
                            &method_def,
                        );
                        let mut frag = String::new();
                        Self::generate_method_info(
                            &mut frag,
                            &method_info_name,
                            &struct_type_name,
                            &method_rgctxs,
                            il2cpp.version,
                        );
                        local.push((generic_method_pointer, frag));
                    }
                }
                local
            })
            .collect();

        let mut method_info_header = String::with_capacity(1 << 16);
        let mut method_info_cache: HashSet<u64> = HashSet::new();
        for chunk in method_info_chunks {
            for (ptr, frag) in chunk {
                if method_info_cache.insert(ptr) {
                    method_info_header.push_str(&frag);
                }
            }
        }

        let mut array_class_header = String::with_capacity(1 << 14);
        for frag in array_classes.values() {
            array_class_header.push_str(frag);
        }

        let mut buf = String::with_capacity(1 << 20);
        write!(buf, "#include <stdint.h>\n#include <stdbool.h>\n\n").ok();
        buf.push_str(header_constants::generic_header());
        buf.push_str(version_header);
        buf.push_str(&header_struct);
        buf.push_str(&array_class_header);
        buf.push_str(&method_info_header);

        Ok(buf)
    }

    fn add_parent(
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        struct_name_dic: &HashMap<usize, String>,
        _metadata: &Metadata,
        info: &mut StructInfo,
    ) {
        if type_def.is_value_type() || type_def.is_enum() { return; }
        if type_def.parent_index < 0 { return; }
        if let Some(parent) = il2cpp.types.get(type_def.parent_index as usize) {
            let te = Il2CppTypeEnum::from_u8(parent.type_enum);
            if te == Some(Il2CppTypeEnum::Object) { return; }
            let klass_idx = parent.klass_index() as usize;
            if let Some(sn) = struct_name_dic.get(&klass_idx) {
                info.parent = Some(sn.clone());
            }
        }
    }

    fn add_fields(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        struct_name_dic: &HashMap<usize, String>,
        info: &mut StructInfo,
        array_classes: &mut HashMap<String, String>,
        context: Option<&Il2CppGenericContext>,
        mut hdr_ctx: Option<&mut HeaderGenCtx>,
    ) {
        if type_def.field_count == 0 { return; }
        let field_end = type_def.field_start as usize + type_def.field_count as usize;
        let mut instance_field_name_cache: HashSet<String> = HashSet::new();
        let mut static_field_name_cache: HashSet<String> = HashSet::new();

        for i in type_def.field_start as usize..field_end {
            let field_def = metadata.field_defs[i].clone();
            let field_type = il2cpp.types[field_def.type_index as usize].clone();

            if (field_type.attrs & field_attributes::LITERAL) != 0 { continue; }

            let te = Il2CppTypeEnum::from_u8(field_type.type_enum);
            if te == Some(Il2CppTypeEnum::SzArray) || te == Some(Il2CppTypeEnum::Array) {
                if field_type.datapoint != 0 {
                    if let Some(element_type) = il2cpp.types.get(field_type.datapoint as usize).cloned() {
                        let elem_struct_name = Self::get_il2cpp_struct_name(&element_type, struct_name_dic, il2cpp, context);
                        let array_struct_name = format!("{}_array", elem_struct_name);
                        if !array_classes.contains_key(&array_struct_name) {
                            let mut frag = String::new();
                            Self::parse_array_class_struct(
                                &mut frag,
                                &element_type,
                                struct_name_dic,
                                executor,
                                metadata,
                                il2cpp,
                                context,
                            );
                            array_classes.insert(array_struct_name, frag);
                        }
                    }
                }
            }

            let field_type_name = Self::parse_type(&field_type, struct_name_dic, executor, metadata, il2cpp, context, hdr_ctx.as_deref_mut());
            let mut field_name = fix_name(&metadata.get_string_from_index(field_def.name_index).unwrap_or_else(|_| "field".to_string()));

            let is_static = (field_type.attrs & field_attributes::STATIC) != 0;
            let name_cache = if is_static { &mut static_field_name_cache } else { &mut instance_field_name_cache };
            if !name_cache.insert(field_name.clone()) {
                let mut suffix = 1u32;
                let base = field_name.clone();
                loop {
                    field_name = format!("{}_{}", base, suffix);
                    if name_cache.insert(field_name.clone()) { break; }
                    suffix += 1;
                }
            }

            let is_vt = Self::is_value_type_check(&field_type, metadata, il2cpp, executor, context);
            let is_ct = Self::is_custom_type_check(&field_type, metadata, il2cpp, executor, context);

            let field_info = StructFieldInfo {
                field_type_name,
                field_name,
                is_value_type: is_vt,
                is_custom_type: is_ct,
            };

            if is_static {
                info.static_fields.push(field_info);
            } else {
                info.fields.push(field_info);
            }
        }
    }

    fn is_value_type_check(il2cpp_type: &Il2CppType, metadata: &Metadata, il2cpp: &Il2Cpp, executor: &Il2CppExecutor, context: Option<&Il2CppGenericContext>) -> bool {
        match Il2CppTypeEnum::from_u8(il2cpp_type.type_enum) {
            Some(Il2CppTypeEnum::ValueType) => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                if let Some(td) = metadata.type_defs.get(klass_idx) {
                    return !td.is_enum();
                }
                false
            }
            Some(Il2CppTypeEnum::GenericInst) => {
                let generic_class_ptr = il2cpp_type.generic_class();
                if generic_class_ptr != 0 {
                    if let Ok(generic_class) = il2cpp.read_generic_class(generic_class_ptr) {
                        if let Some((td, _)) = executor.get_generic_class_type_definition(&generic_class, metadata, il2cpp) {
                            return td.is_value_type() && !td.is_enum();
                        }
                    }
                }
                false
            }
            Some(Il2CppTypeEnum::Var) => {
                if let Some(ctx) = context {
                    let generic_param = executor.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp);
                    if let Some(gp) = generic_param {
                        if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.class_inst, gp.num as u32) {
                            return Self::is_value_type_check(&resolved, metadata, il2cpp, executor, None);
                        }
                    }
                }
                false
            }
            Some(Il2CppTypeEnum::MVar) => {
                if let Some(ctx) = context {
                    let generic_param = executor.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp);
                    if let Some(gp) = generic_param {
                        if ctx.method_inst == 0 && ctx.class_inst != 0 {
                            if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.class_inst, gp.num as u32) {
                                return Self::is_value_type_check(&resolved, metadata, il2cpp, executor, None);
                            }
                        } else {
                            if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.method_inst, gp.num as u32) {
                                return Self::is_value_type_check(&resolved, metadata, il2cpp, executor, None);
                            }
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn is_custom_type_check(il2cpp_type: &Il2CppType, metadata: &Metadata, il2cpp: &Il2Cpp, executor: &Il2CppExecutor, context: Option<&Il2CppGenericContext>) -> bool {
        match Il2CppTypeEnum::from_u8(il2cpp_type.type_enum) {
            Some(Il2CppTypeEnum::Ptr) => {
                if il2cpp_type.datapoint != 0 {
                    if let Some(ori) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        return Self::is_custom_type_check(&ori, metadata, il2cpp, executor, context);
                    }
                }
                false
            }
            Some(Il2CppTypeEnum::String) | Some(Il2CppTypeEnum::Class)
            | Some(Il2CppTypeEnum::Array) | Some(Il2CppTypeEnum::SzArray) => true,
            Some(Il2CppTypeEnum::ValueType) => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                if let Some(td) = metadata.type_defs.get(klass_idx) {
                    if td.is_enum() {
                        if let Some(elem) = il2cpp.types.get(td.element_type_index as usize).cloned() {
                            return Self::is_custom_type_check(&elem, metadata, il2cpp, executor, context);
                        }
                    }
                    return true;
                }
                false
            }
            Some(Il2CppTypeEnum::GenericInst) => {
                let generic_class_ptr = il2cpp_type.generic_class();
                if generic_class_ptr != 0 {
                    if let Ok(generic_class) = il2cpp.read_generic_class(generic_class_ptr) {
                        if let Some((td, _)) = executor.get_generic_class_type_definition(&generic_class, metadata, il2cpp) {
                            if td.is_enum() {
                                if let Some(elem) = il2cpp.types.get(td.element_type_index as usize).cloned() {
                                    return Self::is_custom_type_check(&elem, metadata, il2cpp, executor, context);
                                }
                            }
                            return true;
                        }
                    }
                }
                true
            }
            Some(Il2CppTypeEnum::Var) => {
                if let Some(ctx) = context {
                    let generic_param = executor.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp);
                    if let Some(gp) = generic_param {
                        if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.class_inst, gp.num as u32) {
                            return Self::is_custom_type_check(&resolved, metadata, il2cpp, executor, None);
                        }
                    }
                }
                false
            }
            Some(Il2CppTypeEnum::MVar) => {
                if let Some(ctx) = context {
                    let generic_param = executor.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp);
                    if let Some(gp) = generic_param {
                        if ctx.method_inst == 0 && ctx.class_inst != 0 {
                            if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.class_inst, gp.num as u32) {
                                return Self::is_custom_type_check(&resolved, metadata, il2cpp, executor, None);
                            }
                        } else {
                            if let Some(resolved) = Self::resolve_generic_type_var(il2cpp, ctx.method_inst, gp.num as u32) {
                                return Self::is_custom_type_check(&resolved, metadata, il2cpp, executor, None);
                            }
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn add_vtable_methods(
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        info: &mut StructInfo,
    ) {
        let mut dic: BTreeMap<u16, String> = BTreeMap::new();

        for i in 0..type_def.vtable_count as usize {
            let vtable_index = type_def.vtable_start as usize + i;
            if vtable_index >= metadata.vtable_methods.len() { continue; }

            let encoded = metadata.vtable_methods[vtable_index];
            let usage = (encoded & 0xE0000000) >> 29;
            // v27+ uses different index encoding: (encoded & 0x1FFFFFFEU) >> 1
            let index = if metadata.version >= 27.0 {
                (encoded & 0x1FFFFFFE) >> 1
            } else {
                encoded & 0x1FFFFFFF
            };

            let method_def = if usage == 6 {
                if (index as usize) < il2cpp.method_specs.len() {
                    let spec = &il2cpp.method_specs[index as usize];
                    metadata.method_defs.get(spec.method_definition_index as usize).cloned()
                } else {
                    None
                }
            } else {
                metadata.method_defs.get(index as usize).cloned()
            };

            if let Some(md) = method_def {
                if md.slot != 0xFFFF {
                    let name = metadata.get_string_from_index(md.name_index as i32).unwrap_or_else(|_| "unknown".to_string());
                    dic.insert(md.slot, fix_name(&name));
                }
            }
        }

        if !dic.is_empty() {
            let max_slot = *dic.keys().last().unwrap() as usize;
            let mut vtable_vec: Vec<Option<StructVTableMethodInfo>> = Vec::with_capacity(max_slot + 1);
            for i in 0..=max_slot {
                if let Some(name) = dic.get(&(i as u16)) {
                    vtable_vec.push(Some(StructVTableMethodInfo { method_name: name.clone() }));
                } else {
                    vtable_vec.push(None);
                }
            }
            info.vtable_methods = vtable_vec;
        }
    }

    fn add_rgctx(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        _type_index: usize,
        type_def_image_names: &HashMap<usize, String>,
        info: &mut StructInfo,
    ) {
        let type_def_idx = _type_index;
        let image_name = type_def_image_names.get(&type_def_idx).cloned().unwrap_or_default();
        let collection = executor.get_rgctx_definition_for_type(&image_name, type_def, metadata, il2cpp);
        if let Some(definitions) = collection {
            for def in &definitions {
                let rgctx_type_val = def.rgctx_type;
                let mut rgctx_info = StructRGCTXInfo {
                    rgctx_type: rgctx_type_val,
                    type_name: None,
                    class_name: None,
                    method_name: None,
                };
                let rgctx_data_type_index: Option<i32> = if def.def_data.is_some() {
                    def.def_data.as_ref().map(|d| d.rgctx_data_dummy)
                } else if def.constrained_data.is_some() {
                    def.constrained_data.as_ref().map(|d| d.type_index)
                } else if def.data_va != 0 {
                    if let Ok(offset) = il2cpp.map_vatr(def.data_va) {
                        il2cpp.stream.peek_i32_at(offset).ok()
                    } else { None }
                } else { None };
                if let Some(data_index) = rgctx_data_type_index {
                    match rgctx_type_val as i32 {
                        1 => {
                            let type_idx = data_index as usize;
                            if type_idx < il2cpp.types.len() {
                                let il2cpp_type = il2cpp.types[type_idx].clone();
                                let name = executor.get_type_name(&il2cpp_type, metadata, il2cpp, true, false);
                                rgctx_info.type_name = Some(fix_name(&name));
                            }
                        }
                        2 => {
                            let type_idx = data_index as usize;
                            if type_idx < il2cpp.types.len() {
                                let il2cpp_type = il2cpp.types[type_idx].clone();
                                let name = executor.get_type_name(&il2cpp_type, metadata, il2cpp, true, false);
                                rgctx_info.class_name = Some(fix_name(&name));
                            }
                        }
                        3 => {
                            let method_idx = data_index as usize;
                            if method_idx < il2cpp.method_specs.len() {
                                let (type_name, method_name) = executor.get_method_spec_name(method_idx, metadata, il2cpp, true);
                                rgctx_info.method_name = Some(fix_name(&format!("{}.{}", type_name, method_name)));
                            }
                        }
                        _ => {}
                    }
                }
                info.rgctxs.push(rgctx_info);
            }
        }
    }

    fn write_gcc_parent_fields(
        idx: usize,
        list: &[StructInfo],
        name_map: &HashMap<String, usize>,
        buf: &mut String,
    ) {
        let info = &list[idx];
        if let Some(parent_name) = &info.parent {
            let parent_key = format!("{}_o", parent_name);
            if let Some(&parent_idx) = name_map.get(&parent_key) {
                // Recursively write fields from the topmost base class down
                Self::write_gcc_parent_fields(parent_idx, list, name_map, buf);
                
                let p_info = &list[parent_idx];
                for field in &p_info.fields {
                    if field.is_custom_type && needs_struct_prefix(&field.field_type_name) {
                        writeln!(buf, "\tstruct {} {};", field.field_type_name, field.field_name).ok();
                    } else {
                        writeln!(buf, "\t{} {};", field.field_type_name, field.field_name).ok();
                    }
                }
            }
        }
    }

    fn ensure_value_type_defined(
        field_type_name: &str,
        list: &[StructInfo],
        name_map: &HashMap<String, usize>,
        cache: &mut HashSet<usize>,
        forward_declared: &mut HashSet<String>,
        buf: &mut String,
        is_32bit: bool,
        is_pe: bool,
        compiler_layout: &str,
    ) {
        if let Some(&field_idx) = name_map.get(field_type_name) {
            Self::recursion_struct_info(field_idx, list, name_map, cache, forward_declared, buf, is_32bit, is_pe, compiler_layout);
        } else if forward_declared.insert(field_type_name.to_string()) {
            writeln!(buf, "struct {} {{", field_type_name).ok();
            writeln!(buf, "\tuint8_t _stub;").ok();
            writeln!(buf, "}};").ok();
        }
    }
    fn recursion_struct_info(
        idx: usize,
        list: &[StructInfo],
        name_map: &HashMap<String, usize>,
        cache: &mut HashSet<usize>,
        forward_declared: &mut HashSet<String>,
        buf: &mut String,
        is_32bit: bool,
        is_pe: bool,
        compiler_layout: &str,
    ) {
        if !cache.insert(idx) { return; }
        let info = &list[idx];

        if let Some(parent_name) = &info.parent {
            let parent_key = format!("{}_o", parent_name);
            if let Some(&parent_idx) = name_map.get(&parent_key) {
                Self::recursion_struct_info(parent_idx, list, name_map, cache, forward_declared, buf, is_32bit, is_pe, compiler_layout);
            }
        }

        for field in &info.fields {
            if field.is_value_type { // ONLY hard-value dependencies, fixes topological cycle loops
                Self::ensure_value_type_defined(&field.field_type_name, list, name_map, cache, forward_declared, buf, is_32bit, is_pe, compiler_layout);
            }
        }
        for field in &info.static_fields {
            if field.is_value_type {
                Self::ensure_value_type_defined(&field.field_type_name, list, name_map, cache, forward_declared, buf, is_32bit, is_pe, compiler_layout);
            }
        }

        if compiler_layout == "GCC" {
            if is_pe && !info.is_value_type {
                if is_32bit { writeln!(buf, "struct __declspec(align(4)) {}_Fields {{", info.type_name).ok(); } 
                else { writeln!(buf, "struct __declspec(align(8)) {}_Fields {{", info.type_name).ok(); }
            } else {
                writeln!(buf, "struct {}_Fields {{", info.type_name).ok();
            }
            // Flatten fields from parents first
            Self::write_gcc_parent_fields(idx, list, name_map, buf);
        } else {
            if let Some(parent_name) = &info.parent {
                writeln!(buf, "struct {}_Fields : {}_Fields {{", info.type_name, parent_name).ok();
            } else {
                if is_pe && !info.is_value_type {
                    if is_32bit { writeln!(buf, "struct __declspec(align(4)) {}_Fields {{", info.type_name).ok(); } 
                    else { writeln!(buf, "struct __declspec(align(8)) {}_Fields {{", info.type_name).ok(); }
                } else {
                    writeln!(buf, "struct {}_Fields {{", info.type_name).ok();
                }
            }
        }

        for field in &info.fields {
            if field.is_custom_type && needs_struct_prefix(&field.field_type_name) {
                writeln!(buf, "\tstruct {} {};", field.field_type_name, field.field_name).ok();
            } else {
                writeln!(buf, "\t{} {};", field.field_type_name, field.field_name).ok();
            }
        }
        writeln!(buf, "}};").ok();

        if !info.rgctxs.is_empty() {
            writeln!(buf, "struct {}_RGCTXs {{", info.type_name).ok();
            for (i, rgctx) in info.rgctxs.iter().enumerate() {
                match rgctx.rgctx_type as i32 {
                    1 => {
                        let tn = rgctx.type_name.as_deref().unwrap_or("unknown");
                        writeln!(buf, "\tIl2CppType* _{}_{};", i, tn).ok();
                    }
                    2 => {
                        let cn = rgctx.class_name.as_deref().unwrap_or("unknown");
                        writeln!(buf, "\tIl2CppClass* _{}_{};", i, cn).ok();
                    }
                    3 => {
                        let mn = rgctx.method_name.as_deref().unwrap_or("unknown");
                        writeln!(buf, "\tMethodInfo* _{}_{};", i, mn).ok();
                    }
                    _ => {}
                }
            }
            writeln!(buf, "}};").ok();
        }

        if !info.vtable_methods.is_empty() {
            writeln!(buf, "struct {}_VTable {{", info.type_name).ok();
            for (i, method) in info.vtable_methods.iter().enumerate() {
                write!(buf, "\tVirtualInvokeData _{}_", i).ok();
                if let Some(m) = method {
                    write!(buf, "{}", m.method_name).ok();
                } else {
                    write!(buf, "unknown").ok();
                }
                writeln!(buf, ";").ok();
            }
            writeln!(buf, "}};").ok();
        }


        writeln!(buf, "struct {}_c {{", info.type_name).ok();
        writeln!(buf, "\tIl2CppClass_1 _1;").ok();
        if !info.static_fields.is_empty() {
            writeln!(buf, "\tstruct {}_StaticFields* static_fields;", info.type_name).ok();
        } else {
            writeln!(buf, "\tvoid* static_fields;").ok();
        }
        if !info.rgctxs.is_empty() {
            writeln!(buf, "\t{}_RGCTXs* rgctx_data;", info.type_name).ok();
        } else {
            writeln!(buf, "\tIl2CppRGCTXData* rgctx_data;").ok();
        }
        writeln!(buf, "\tIl2CppClass_2 _2;").ok();
        if !info.vtable_methods.is_empty() {
            writeln!(buf, "\t{}_VTable vtable;", info.type_name).ok();
        } else {
            writeln!(buf, "\tVirtualInvokeData vtable[32];").ok();
        }
        writeln!(buf, "}};").ok();

        writeln!(buf, "struct {}_o {{", info.type_name).ok();
        if !info.is_value_type {
            writeln!(buf, "\t{}_c *klass;", info.type_name).ok();
            writeln!(buf, "\tvoid *monitor;").ok();
        }
        writeln!(buf, "\t{}_Fields fields;", info.type_name).ok();
        writeln!(buf, "}};").ok();

        if !info.static_fields.is_empty() {
            writeln!(buf, "struct {}_StaticFields {{", info.type_name).ok();
            for field in &info.static_fields {
                if field.is_custom_type && needs_struct_prefix(&field.field_type_name) {
                    writeln!(buf, "\tstruct {} {};", field.field_type_name, field.field_name).ok();
                } else {
                    writeln!(buf, "\t{} {};", field.field_type_name, field.field_name).ok();
                }
            }
            writeln!(buf, "}};").ok();
        }
    }

    fn collect_rgctx_info_for_method(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        image_name: &str,
        method_def: &Il2CppMethodDefinition,
    ) -> Vec<StructRGCTXInfo> {
        let mut rgctxs = Vec::new();
        let collection = executor.get_rgctx_definition_for_method(image_name, method_def, metadata, il2cpp);
        if let Some(definitions) = collection {
            for def in &definitions {
                let rgctx_type_val = def.rgctx_type;
                let mut rgctx_info = StructRGCTXInfo {
                    rgctx_type: rgctx_type_val,
                    type_name: None,
                    class_name: None,
                    method_name: None,
                };
                let type_idx_val = def.type_index();
                let data_index: Option<i32> = if type_idx_val == -1 { None } else { Some(type_idx_val) };
                if let Some(data_val) = data_index {
                    match rgctx_type_val as i32 {
                        1 => {
                            let type_idx = data_val as usize;
                            if type_idx < il2cpp.types.len() {
                                let il2cpp_type = il2cpp.types[type_idx].clone();
                                let name = executor.get_type_name(&il2cpp_type, metadata, il2cpp, true, false);
                                rgctx_info.type_name = Some(fix_name(&name));
                            }
                        }
                        2 => {
                            let type_idx = data_val as usize;
                            if type_idx < il2cpp.types.len() {
                                let il2cpp_type = il2cpp.types[type_idx].clone();
                                let name = executor.get_type_name(&il2cpp_type, metadata, il2cpp, true, false);
                                rgctx_info.class_name = Some(fix_name(&name));
                            }
                        }
                        3 => {
                            let method_idx = data_val as usize;
                            if method_idx < il2cpp.method_specs.len() {
                                let (type_name, method_name) = executor.get_method_spec_name(method_idx, metadata, il2cpp, true);
                                rgctx_info.method_name = Some(fix_name(&format!("{}.{}", type_name, method_name)));
                            }
                        }
                        _ => {}
                    }
                }
                rgctxs.push(rgctx_info);
            }
        }
        rgctxs
    }

    fn generate_method_info(
        buf: &mut String,
        method_info_name: &str,
        struct_type_name: &str,
        rgctxs: &[StructRGCTXInfo],
        version: f64,
    ) {
        if !rgctxs.is_empty() {
            writeln!(buf, "struct {}_RGCTXs {{", method_info_name).ok();
            for (i, rgctx) in rgctxs.iter().enumerate() {
                match rgctx.rgctx_type as i32 {
                    1 => {
                        let tn = rgctx.type_name.as_deref().unwrap_or("unknown");
                        writeln!(buf, "\tIl2CppType* _{}_{};", i, tn).ok();
                    }
                    2 => {
                        let cn = rgctx.class_name.as_deref().unwrap_or("unknown");
                        writeln!(buf, "\tIl2CppClass* _{}_{};", i, cn).ok();
                    }
                    3 => {
                        let mn = rgctx.method_name.as_deref().unwrap_or("unknown");
                        writeln!(buf, "\tMethodInfo* _{}_{};", i, mn).ok();
                    }
                    _ => {}
                }
            }
            writeln!(buf, "}};").ok();
        }

        writeln!(buf, "struct {} {{", method_info_name).ok();
        writeln!(buf, "\tIl2CppMethodPointer methodPointer;").ok();
        if version >= 29.0 {
            writeln!(buf, "\tIl2CppMethodPointer virtualMethodPointer;").ok();
            writeln!(buf, "\tInvokerMethod invoker_method;").ok();
        } else {
            writeln!(buf, "\tvoid* invoker_method;").ok();
        }
        writeln!(buf, "\tconst char* name;").ok();
        if version <= 24.0 {
            writeln!(buf, "\t{}_c *declaring_type;", struct_type_name).ok();
        } else {
            writeln!(buf, "\t{}_c *klass;", struct_type_name).ok();
        }
        writeln!(buf, "\tconst Il2CppType *return_type;").ok();
        if version >= 29.0 {
            writeln!(buf, "\tconst Il2CppType** parameters;").ok();
        } else {
            writeln!(buf, "\tconst void* parameters;").ok();
        }
        if !rgctxs.is_empty() {
            writeln!(buf, "\tconst {}_RGCTXs* rgctx_data;", method_info_name).ok();
        } else {
            writeln!(buf, "\tconst Il2CppRGCTXData* rgctx_data;").ok();
        }
        writeln!(buf, "\tunion").ok();
        writeln!(buf, "\t{{").ok();
        writeln!(buf, "\t\tconst void* genericMethod;").ok();
        if version >= 27.0 {
            writeln!(buf, "\t\tconst void* genericContainerHandle;").ok();
        } else {
            writeln!(buf, "\t\tconst void* genericContainer;").ok();
        }
        writeln!(buf, "\t}};").ok();
        if version <= 24.0 {
            writeln!(buf, "\tint32_t customAttributeIndex;").ok();
        }
        writeln!(buf, "\tuint32_t token;").ok();
        writeln!(buf, "\tuint16_t flags;").ok();
        writeln!(buf, "\tuint16_t iflags;").ok();
        writeln!(buf, "\tuint16_t slot;").ok();
        writeln!(buf, "\tuint8_t parameters_count;").ok();
        writeln!(buf, "\tuint8_t bitflags;").ok();
        writeln!(buf, "}};").ok();
    }
}

fn fix_name(name: &str) -> String {
    sanitize_cpp_identifier(name, NameSanitizerOptions {
        allow_dollar: false,
        avoid_double_underscore_prefix: true,
    })
}

fn get_unique_name(name: &str, set: &mut HashSet<String>) -> String {
    let mut fix = name.to_string();
    let mut i = 1;
    while !set.insert(fix.clone()) {
        fix = format!("{}_{}", name, i);
        i += 1;
    }
    fix
}

fn get_method_type_signature(types: &[Il2CppTypeEnum]) -> String {
    let mut sig = String::with_capacity(types.len());
    for te in types {
        sig.push(match te {
            Il2CppTypeEnum::Void => 'v',
            Il2CppTypeEnum::Boolean | Il2CppTypeEnum::Char
            | Il2CppTypeEnum::I1 | Il2CppTypeEnum::U1
            | Il2CppTypeEnum::I2 | Il2CppTypeEnum::U2
            | Il2CppTypeEnum::I4 | Il2CppTypeEnum::U4 => 'i',
            Il2CppTypeEnum::I8 | Il2CppTypeEnum::U8 => 'j',
            Il2CppTypeEnum::R4 => 'f',
            Il2CppTypeEnum::R8 => 'd',
            _ => 'i',
        });
    }
    sig
}
