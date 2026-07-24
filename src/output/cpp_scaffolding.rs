use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::fs;
use std::path::Path;
use rayon::prelude::*;
use crate::error::Result;
use crate::il2cpp::base::*;
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::enums::*;
use crate::il2cpp::structures::*;
use crate::executor::Il2CppExecutor;
use super::struct_generator::StructGenerator;

pub struct CppScaffolding;

struct FuncEntry {
    rva: u64,
    return_type: String,
    func_name: String,
    params: String,
    method_info_rva: Option<u64>,
}

impl CppScaffolding {
    pub fn build(
        executor: &mut Il2CppExecutor,
        metadata: &mut Metadata,
        il2cpp: &mut Il2Cpp,
    ) -> Result<String> {
        let struct_name_dic = StructGenerator::build_struct_name_dic_pub(executor, metadata, il2cpp);
        let type_def_image_names = StructGenerator::build_type_def_image_names_pub(metadata);

        let mut method_info_map: HashMap<(usize, usize), u64> = HashMap::new();

        Self::collect_method_info_addresses(
            &mut method_info_map, executor, metadata, il2cpp, &type_def_image_names,
        );

        // Parallel per-type entry collection; sequential dedupe by RVA (first wins, type order).
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

        let per_type_entries: Vec<Vec<FuncEntry>> = type_jobs
            .par_iter()
            .map(|(type_def_index, image_name)| {
                let mut local_exec = Il2CppExecutor::new_for_worker(executor);
                let type_def = metadata.type_defs[*type_def_index].clone();
                let type_name = local_exec.get_type_def_name(
                    &type_def,
                    *type_def_index,
                    metadata,
                    il2cpp,
                    true,
                    true,
                );
                let mut entries = Vec::new();
                let method_end = type_def.method_start as usize + type_def.method_count as usize;

                for method_index in type_def.method_start as usize..method_end {
                    let method_def = metadata.method_defs[method_index].clone();
                    let method_name_raw = metadata
                        .get_string_from_index(method_def.name_index as i32)
                        .unwrap_or_default();
                    let method_pointer = il2cpp.get_method_pointer(image_name, &method_def);
                    let mi_rva = method_info_map.get(&(*type_def_index, method_index)).copied();

                    if method_pointer > 0 {
                        let rva = il2cpp.get_rva(method_pointer);
                        let func_name = format!(
                            "{}$${}",
                            fix_name_scaffold(&type_name),
                            fix_name_scaffold(&method_name_raw)
                        );
                        let (return_type, params) = Self::build_func_signature(
                            &mut local_exec,
                            metadata,
                            il2cpp,
                            &method_def,
                            &type_def,
                            *type_def_index,
                            &struct_name_dic,
                            None,
                        );
                        entries.push(FuncEntry {
                            rva,
                            return_type,
                            func_name,
                            params,
                            method_info_rva: mi_rva,
                        });
                    } else if mi_rva.is_some() {
                        let func_name = format!(
                            "{}$${}",
                            fix_name_scaffold(&type_name),
                            fix_name_scaffold(&method_name_raw)
                        );
                        let (return_type, params) = Self::build_func_signature(
                            &mut local_exec,
                            metadata,
                            il2cpp,
                            &method_def,
                            &type_def,
                            *type_def_index,
                            &struct_name_dic,
                            None,
                        );
                        entries.push(FuncEntry {
                            rva: 0,
                            return_type,
                            func_name,
                            params,
                            method_info_rva: mi_rva,
                        });
                    }

                    if let Some(spec_indices) = il2cpp.method_definition_method_specs.get(&method_index) {
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
                            let (spec_type_name, spec_method_name) =
                                local_exec.get_method_spec_name(*spec_idx, metadata, il2cpp, true);
                            let func_name = format!(
                                "{}$${}",
                                fix_name_scaffold(&spec_type_name),
                                fix_name_scaffold(&spec_method_name)
                            );
                            let (class_inst, method_inst) =
                                local_exec.get_method_spec_generic_context(*spec_idx, il2cpp);
                            let generic_context = Il2CppGenericContext {
                                class_inst,
                                method_inst,
                            };
                            let (return_type, params) = Self::build_func_signature(
                                &mut local_exec,
                                metadata,
                                il2cpp,
                                &method_def,
                                &type_def,
                                *type_def_index,
                                &struct_name_dic,
                                Some(&generic_context),
                            );
                            entries.push(FuncEntry {
                                rva: spec_rva,
                                return_type,
                                func_name,
                                params,
                                method_info_rva: None,
                            });
                        }
                    }
                }
                entries
            })
            .collect();

        let mut entries: Vec<FuncEntry> = Vec::new();
        let mut seen_rvas: HashSet<u64> = HashSet::new();
        for local in per_type_entries {
            for entry in local {
                if entry.rva > 0 && !seen_rvas.insert(entry.rva) {
                    continue;
                }
                entries.push(entry);
            }
        }

        let mut header = String::with_capacity(entries.len() * 120 + 2048);

        writeln!(header, "// IL2CPP Function Pointers").ok();
        writeln!(header, "// Generated by Il2CppDumper - Rust Edition").ok();
        writeln!(header, "//").ok();
        writeln!(header, "// Usage: #include this file after setting il2cpp_base to the base address").ok();
        writeln!(header, "// of libil2cpp.so in memory.").ok();
        writeln!(header, "").ok();
        writeln!(header, "#pragma once").ok();
        writeln!(header, "").ok();
        writeln!(header, "#ifndef DO_APP_FUNC").ok();
        writeln!(header, "#define DO_APP_FUNC(offset, rettype, name, args) \\").ok();
        writeln!(header, "    typedef rettype (*name##_t) args; \\").ok();
        writeln!(header, "    static name##_t name = (name##_t)(il2cpp_base + offset);").ok();
        writeln!(header, "#endif").ok();
        writeln!(header, "").ok();
        writeln!(header, "#ifndef DO_APP_FUNC_METHODINFO").ok();
        writeln!(header, "#define DO_APP_FUNC_METHODINFO(offset, name) \\").ok();
        writeln!(header, "    static struct MethodInfo** name = (struct MethodInfo**)(il2cpp_base + offset);").ok();
        writeln!(header, "#endif").ok();
        writeln!(header, "").ok();
        writeln!(header, "// ******************************************************************************").ok();
        writeln!(header, "// * Application method definitions").ok();
        writeln!(header, "// ******************************************************************************").ok();
        writeln!(header, "").ok();

        for entry in &entries {
            if entry.rva > 0 {
                writeln!(header, "DO_APP_FUNC(0x{:X}, {}, {}, ({}));",
                    entry.rva,
                    entry.return_type,
                    entry.func_name,
                    entry.params,
                ).ok();
            }

            if let Some(mi_rva) = entry.method_info_rva {
                writeln!(header, "DO_APP_FUNC_METHODINFO(0x{:X}, {}__MethodInfo);",
                    mi_rva,
                    entry.func_name,
                ).ok();
            }
        }

        Ok(header)
    }

    fn collect_method_info_addresses(
        map: &mut HashMap<(usize, usize), u64>,
        executor: &mut Il2CppExecutor,
        metadata: &mut Metadata,
        il2cpp: &mut Il2Cpp,
        type_def_image_names: &HashMap<usize, String>,
    ) {
        if il2cpp.version >= 27.0 {
            Self::collect_method_info_v27(map, executor, metadata, il2cpp, type_def_image_names);
        } else if il2cpp.version > 16.0 {
            Self::collect_method_info_pre_v27(map, metadata, il2cpp);
        }
    }

    fn collect_method_info_pre_v27(
        map: &mut HashMap<(usize, usize), u64>,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) {
        if metadata.metadata_usage_dic.is_empty() { return; }
        let usage_dic = metadata.metadata_usage_dic.clone();

        for (usage_type, entries) in &usage_dic {
            if *usage_type != 3 { continue; }
            for (dest_index, source_index) in entries {
                let dest = *dest_index as usize;
                if dest >= il2cpp.metadata_usages.len() { continue; }
                let address = il2cpp.metadata_usages[dest];
                if address == 0 { continue; }
                let rva = il2cpp.get_rva(address);
                let src = *source_index as usize;

                if let Some(method_def) = metadata.method_defs.get(src) {
                    let td_idx = method_def.declaring_type as usize;
                    map.insert((td_idx, src), rva);
                }
            }
        }
    }

    fn collect_method_info_v27(
        map: &mut HashMap<(usize, usize), u64>,
        _executor: &mut Il2CppExecutor,
        metadata: &mut Metadata,
        il2cpp: &mut Il2Cpp,
        _type_def_image_names: &HashMap<usize, String>,
    ) {
        let pointer_size = if il2cpp.is_32bit { 4u64 } else { 8u64 };
        let data_sections = il2cpp.data_sections.clone();

        for sec in &data_sections {
            let sec_end = std::cmp::min(sec.offset_end, il2cpp.stream.len() as u64).saturating_sub(pointer_size);
            let mut pos = sec.offset;
            while pos < sec_end {
                il2cpp.stream.set_position(pos);
                let metadata_value = if il2cpp.is_32bit {
                    il2cpp.stream.read_u32().unwrap_or(0) as u64
                } else {
                    il2cpp.stream.read_u64().unwrap_or(0)
                };
                let saved_pos = il2cpp.stream.position();
                pos = saved_pos;

                if metadata_value >= u32::MAX as u64 { continue; }
                let encoded_token = metadata_value as u32;
                let usage = (encoded_token & 0xE0000000) >> 29;
                if usage != 3 { continue; }
                let decoded_index = (encoded_token & 0x1FFFFFFE) >> 1;
                let expected = ((usage << 29) | (decoded_index << 1)) + 1;
                if metadata_value != expected as u64 { continue; }

                let addr = pos - pointer_size;
                let va = il2cpp.map_rtva(addr);
                let rva = il2cpp.get_rva(va);

                if let Some(method_def) = metadata.method_defs.get(decoded_index as usize) {
                    let td_idx = method_def.declaring_type as usize;
                    map.insert((td_idx, decoded_index as usize), rva);
                }

                if il2cpp.stream.position() != saved_pos {
                    il2cpp.stream.set_position(saved_pos);
                }
            }
        }
    }

    fn build_func_signature(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        method_def: &Il2CppMethodDefinition,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
        struct_name_dic: &HashMap<usize, String>,
        generic_context: Option<&Il2CppGenericContext>,
    ) -> (String, String) {
        let method_return_type = il2cpp.types[method_def.return_type as usize].clone();
        let return_type = Self::parse_scaffolding_type(
            &method_return_type, struct_name_dic, executor, metadata, il2cpp, generic_context,
        );

        let return_c = if method_return_type.byref == 1 {
            format!("{}*", return_type)
        } else {
            return_type
        };

        let mut param_strs = Vec::new();

        let is_static = (method_def.flags as u32 & 0x0010) != 0;
        if !is_static {
            if type_def.is_value_type() {
                let base_name = struct_name_dic.get(&type_def_index)
                    .map(|s| s.as_str())
                    .unwrap_or("Il2CppObject");
                param_strs.push(format!("{}* __this", base_name));
            } else {
                let byval_type = il2cpp.types[type_def.byval_type_index as usize].clone();
                let this_type = Self::parse_scaffolding_type(
                    &byval_type, struct_name_dic, executor, metadata, il2cpp, generic_context,
                );
                if this_type.ends_with('*') {
                    param_strs.push(format!("{} __this", this_type));
                } else {
                    param_strs.push(format!("{}* __this", this_type));
                }
            }
        } else if il2cpp.version <= 24.0 {
            param_strs.push("Il2CppObject* __this".to_string());
        }

        for j in 0..method_def.parameter_count as usize {
            let param_def = metadata.parameter_defs[method_def.parameter_start as usize + j].clone();
            let param_name = metadata.get_string_from_index(param_def.name_index)
                .unwrap_or_else(|_| "param".to_string());
            let param_type = il2cpp.types[param_def.type_index as usize].clone();
            let param_c = Self::parse_scaffolding_type(
                &param_type, struct_name_dic, executor, metadata, il2cpp, generic_context,
            );
            let param_c_final = if param_type.byref == 1 {
                format!("{}*", param_c)
            } else {
                param_c
            };
            param_strs.push(format!("{} {}", param_c_final, fix_name_scaffold(&param_name)));
        }

        param_strs.push("const MethodInfo* method".to_string());

        (return_c, param_strs.join(", "))
    }

    fn parse_scaffolding_type(
        il2cpp_type: &Il2CppType,
        struct_name_dic: &HashMap<usize, String>,
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        context: Option<&Il2CppGenericContext>,
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
                            return Self::parse_scaffolding_type(&elem_type, struct_name_dic, executor, metadata, il2cpp, context);
                        }
                    }
                    if let Some(sn) = struct_name_dic.get(&klass_idx) {
                        return format!("{}", sn);
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
                "Il2CppArray*".to_string()
            }
            Some(Il2CppTypeEnum::GenericInst) => {
                let generic_class_ptr = il2cpp_type.generic_class();
                if generic_class_ptr != 0 {
                    if let Ok(generic_class) = il2cpp.read_generic_class(generic_class_ptr) {
                        if let Some((td, td_idx)) = executor.get_generic_class_type_definition(&generic_class, metadata, il2cpp) {
                            if let Some(sn) = struct_name_dic.get(&td_idx) {
                                if td.is_value_type() {
                                    if td.is_enum() {
                                        if let Some(elem) = il2cpp.types.get(td.element_type_index as usize).cloned() {
                                            return Self::parse_scaffolding_type(&elem, struct_name_dic, executor, metadata, il2cpp, context);
                                        }
                                    }
                                    return sn.clone();
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
                    let gp = executor.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp);
                    if let Some(gp) = gp {
                        if let Some(resolved) = StructGenerator::resolve_generic_type_var_pub(il2cpp, ctx.class_inst, gp.num as u32) {
                            return Self::parse_scaffolding_type(&resolved, struct_name_dic, executor, metadata, il2cpp, None);
                        }
                    }
                }
                "Il2CppObject*".to_string()
            }
            Some(Il2CppTypeEnum::MVar) => {
                if let Some(ctx) = context {
                    let gp = executor.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp);
                    if let Some(gp) = gp {
                        if ctx.method_inst == 0 && ctx.class_inst != 0 {
                            if let Some(resolved) = StructGenerator::resolve_generic_type_var_pub(il2cpp, ctx.class_inst, gp.num as u32) {
                                return Self::parse_scaffolding_type(&resolved, struct_name_dic, executor, metadata, il2cpp, None);
                            }
                        } else {
                            if let Some(resolved) = StructGenerator::resolve_generic_type_var_pub(il2cpp, ctx.method_inst, gp.num as u32) {
                                return Self::parse_scaffolding_type(&resolved, struct_name_dic, executor, metadata, il2cpp, None);
                            }
                        }
                    }
                }
                "Il2CppObject*".to_string()
            }
            Some(Il2CppTypeEnum::Ptr) => {
                if il2cpp_type.datapoint != 0 {
                    if let Some(ori_type) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        let inner = Self::parse_scaffolding_type(&ori_type, struct_name_dic, executor, metadata, il2cpp, context);
                        return format!("{}*", inner);
                    }
                }
                "void*".to_string()
            }
            _ => "Il2CppObject*".to_string(),
        }
    }
}

impl CppScaffolding {
    pub fn write_project(
        executor: &mut Il2CppExecutor,
        metadata: &mut Metadata,
        il2cpp: &mut Il2Cpp,
        type_header_text: &str,
        api_header_text: &str,
        project_root: &Path,
    ) -> Result<()> {
        let appdata_dir = project_root.join("appdata");
        let framework_dir = project_root.join("framework");
        let user_dir = project_root.join("user");
        fs::create_dir_all(&appdata_dir).ok();
        fs::create_dir_all(&framework_dir).ok();
        fs::create_dir_all(&user_dir).ok();

        let functions_header = Self::build(executor, metadata, il2cpp).unwrap_or_default();
        fs::write(appdata_dir.join("il2cpp-functions.h"), Self::wrap_header("IL2CPP application method pointers", &functions_header)).ok();

        let types_header_content = Self::wrap_types_header(il2cpp, type_header_text);
        fs::write(appdata_dir.join("il2cpp-types.h"), types_header_content).ok();

        let api_filtered = Self::filter_api_header(api_header_text, &il2cpp.api_export_rvas);
        fs::write(appdata_dir.join("il2cpp-api-functions.h"), Self::wrap_header("IL2CPP API function declarations", &api_filtered)).ok();

        let api_ptr_header = Self::build_api_function_ptr_header(&il2cpp.api_export_rvas);
        fs::write(appdata_dir.join("il2cpp-api-functions-ptr.h"), Self::wrap_header("IL2CPP API function pointers", &api_ptr_header)).ok();

        let types_ptr_header = Self::build_types_ptr_header(executor, metadata, il2cpp);
        fs::write(appdata_dir.join("il2cpp-types-ptr.h"), Self::wrap_header("IL2CPP application type definition addresses", &types_ptr_header)).ok();

        let md_ver = il2cpp.version;
        let md_int = (md_ver * 10.0).round() as i32;
        let version_header = format!("#pragma once\n\n#define __IL2CPP_METADATA_VERSION {md_int}\n");
        fs::write(appdata_dir.join("il2cpp-metadata-version.h"), version_header).ok();

        fs::write(framework_dir.join("dllmain.cpp"), TEMPLATE_DLLMAIN_CPP).ok();
        fs::write(framework_dir.join("helpers.cpp"), TEMPLATE_HELPERS_CPP).ok();
        fs::write(framework_dir.join("helpers.h"), TEMPLATE_HELPERS_H).ok();
        fs::write(framework_dir.join("il2cpp-appdata.h"), TEMPLATE_IL2CPP_APPDATA_H).ok();
        fs::write(framework_dir.join("il2cpp-init.cpp"), TEMPLATE_IL2CPP_INIT_CPP).ok();
        fs::write(framework_dir.join("il2cpp-init.h"), TEMPLATE_IL2CPP_INIT_H).ok();
        fs::write(framework_dir.join("pch-il2cpp.cpp"), TEMPLATE_PCH_IL2CPP_CPP).ok();
        fs::write(framework_dir.join("pch-il2cpp.h"), TEMPLATE_PCH_IL2CPP_H).ok();

        Self::write_if_not_exists(&user_dir.join("main.cpp"), TEMPLATE_MAIN_CPP);
        Self::write_if_not_exists(&user_dir.join("main.h"), TEMPLATE_MAIN_H);

        let project_name = "IL2CppDLL";
        let project_guid = Self::make_guid(0);
        let filter_guid1 = Self::make_guid(1);
        let filter_guid2 = Self::make_guid(2);
        let filter_guid3 = Self::make_guid(3);
        let solution_guid = Self::make_guid(4);

        let project_file = format!("{project_name}.vcxproj");
        let filters_file = format!("{project_file}.filters");
        let solution_file = format!("{project_name}.sln");

        let vcxproj = TEMPLATE_VCXPROJ.replace("%PROJECTGUID%", &project_guid);
        Self::write_if_not_exists(&project_root.join(&project_file), &vcxproj);

        let filters = TEMPLATE_VCXPROJ_FILTERS
            .replace("%GUID1%", &filter_guid1)
            .replace("%GUID2%", &filter_guid2)
            .replace("%GUID3%", &filter_guid3);
        Self::write_if_not_exists(&project_root.join(&filters_file), &filters);

        let sln = TEMPLATE_SLN
            .replace("%PROJECTGUID%", &project_guid)
            .replace("%PROJECTNAME%", project_name)
            .replace("%PROJECTFILE%", &project_file)
            .replace("%SOLUTIONGUID%", &solution_guid);
        Self::write_if_not_exists(&project_root.join(&solution_file), &sln);

        Ok(())
    }

    fn write_if_not_exists(path: &Path, contents: &str) {
        if !path.exists() {
            let _ = fs::write(path, contents);
        }
    }

    fn make_guid(seed: u64) -> String {
        let h = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDEADBEEFCAFEBABE;
        let a = (h & 0xFFFFFFFF) as u32;
        let b = ((h >> 32) & 0xFFFF) as u16;
        let c = ((h >> 48) & 0xFFFF) as u16;
        let d = ((seed.wrapping_add(1)).wrapping_mul(0xBF58476D1CE4E5B9)) ^ 0xC0FFEE00C0FFEE00;
        format!(
            "{a:08X}-{b:04X}-{c:04X}-{:04X}-{:012X}",
            ((d >> 48) & 0xFFFF) as u16,
            d & 0x0000_FFFF_FFFF_FFFF
        )
    }

    fn wrap_header(section: &str, body: &str) -> String {
        let mut out = String::with_capacity(body.len() + 256);
        writeln!(out, "// Generated C++ file by Il2CppDumper - Rust Edition").ok();
        writeln!(out, "// * {section}").ok();
        writeln!(out).ok();
        out.push_str(body);
        if !out.ends_with('\n') { out.push('\n'); }
        out
    }

    fn wrap_types_header(il2cpp: &Il2Cpp, type_header_text: &str) -> String {
        let bits = if il2cpp.is_32bit { 32 } else { 64 };
        let mut out = String::with_capacity(type_header_text.len() + 4096);
        writeln!(out, "// Generated C++ file by Il2CppDumper - Rust Edition").ok();
        writeln!(out).ok();
        writeln!(out, "#pragma once").ok();
        writeln!(out).ok();
        writeln!(out, "#if defined(_IDACLANG_) || defined(_BINARYNINJA_)").ok();
        writeln!(out, "#define IS_LIBCLANG_DECOMPILER").ok();
        writeln!(out, "#endif").ok();
        writeln!(out).ok();
        writeln!(out, "#if defined(_GHIDRA_) || defined(_IDA_) || defined(IS_LIBCLANG_DECOMPILER)").ok();
        writeln!(out, "#define IS_DECOMPILER").ok();
        writeln!(out, "#endif").ok();
        writeln!(out).ok();
        writeln!(out, "#if defined(_GHIDRA_) || defined(_IDA_)").ok();
        writeln!(out, "typedef unsigned __int8 uint8_t;").ok();
        writeln!(out, "typedef unsigned __int16 uint16_t;").ok();
        writeln!(out, "typedef unsigned __int32 uint32_t;").ok();
        writeln!(out, "typedef unsigned __int64 uint64_t;").ok();
        writeln!(out, "typedef __int8 int8_t;").ok();
        writeln!(out, "typedef __int16 int16_t;").ok();
        writeln!(out, "typedef __int32 int32_t;").ok();
        writeln!(out, "typedef __int64 int64_t;").ok();
        writeln!(out, "#endif").ok();
        writeln!(out).ok();
        writeln!(out, "#if defined(IS_LIBCLANG_DECOMPILER)").ok();
        writeln!(out, "typedef unsigned char uint8_t;").ok();
        writeln!(out, "typedef unsigned short uint16_t;").ok();
        writeln!(out, "typedef unsigned int uint32_t;").ok();
        writeln!(out, "typedef unsigned long uint64_t;").ok();
        writeln!(out, "typedef char int8_t;").ok();
        writeln!(out, "typedef short int16_t;").ok();
        writeln!(out, "typedef int int32_t;").ok();
        writeln!(out, "typedef long int64_t;").ok();
        writeln!(out).ok();
        writeln!(out, "#ifdef linux").ok();
        writeln!(out, "#undef linux").ok();
        writeln!(out, "#endif").ok();
        writeln!(out, "#endif").ok();
        writeln!(out).ok();
        writeln!(out, "#if defined(_GHIDRA_) || defined(IS_LIBCLANG_DECOMPILER)").ok();
        writeln!(out, "typedef int{bits}_t intptr_t;").ok();
        writeln!(out, "typedef uint{bits}_t uintptr_t;").ok();
        writeln!(out, "typedef uint{bits}_t size_t;").ok();
        writeln!(out, "#endif").ok();
        writeln!(out).ok();
        writeln!(out, "#ifndef IS_DECOMPILER").ok();
        writeln!(out, "#define _CPLUSPLUS_").ok();
        writeln!(out, "#endif").ok();
        writeln!(out).ok();
        writeln!(out, "// ******************************************************************************").ok();
        writeln!(out, "// * IL2CPP internal types").ok();
        writeln!(out, "// ******************************************************************************").ok();
        writeln!(out).ok();
        out.push_str(type_header_text);
        if !out.ends_with('\n') { out.push('\n'); }
        writeln!(out).ok();
        writeln!(out, "#ifndef IS_DECOMPILER").ok();
        writeln!(out, "namespace app {{").ok();
        writeln!(out, "#endif").ok();
        writeln!(out).ok();
        writeln!(out, "// Application types are emitted through il2cpp.h (#include \"../../il2cpp.h\") separately.").ok();
        writeln!(out).ok();
        writeln!(out, "#ifndef IS_DECOMPILER").ok();
        writeln!(out, "}}").ok();
        writeln!(out, "#endif").ok();
        out
    }

    fn filter_api_header(text: &str, available: &HashMap<String, u64>) -> String {
        if available.is_empty() {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len());
        for line in text.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("DO_API") || trimmed.starts_with("DO_API_NO_RETURN") {
                if let Some(fn_name) = Self::extract_api_name(trimmed) {
                    if available.contains_key(&fn_name) {
                        out.push_str(line);
                        out.push('\n');
                    }
                } else {
                    out.push_str(line);
                    out.push('\n');
                }
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }

    fn extract_api_name(line: &str) -> Option<String> {
        let paren = line.find('(')?;
        let tail = &line[paren + 1..];
        let mut depth = 1;
        let mut end = 0usize;
        for (i, c) in tail.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 { end = i; break; }
                }
                _ => {}
            }
        }
        let args = &tail[..end];
        let parts: Vec<&str> = args.split(',').collect();
        if parts.len() < 2 { return None; }
        Some(parts[1].trim().to_string())
    }

    fn build_api_function_ptr_header(exports: &HashMap<String, u64>) -> String {
        let mut names: Vec<(&String, &u64)> = exports.iter().collect();
        names.sort_by(|a, b| a.0.cmp(b.0));
        let mut out = String::with_capacity(exports.len() * 48);
        out.push_str("#pragma once\n\n");
        for (name, rva) in names {
            writeln!(out, "#define {name}_ptr 0x{rva:08X}").ok();
        }
        out
    }

    fn build_types_ptr_header(
        executor: &mut Il2CppExecutor,
        metadata: &mut Metadata,
        il2cpp: &mut Il2Cpp,
    ) -> String {
        let struct_name_dic = StructGenerator::build_struct_name_dic_pub(executor, metadata, il2cpp);
        let mut out = String::with_capacity(metadata.type_defs.len() * 64);
        out.push_str("#pragma once\n\n");
        let type_defs_len = metadata.type_defs.len();
        for td_idx in 0..type_defs_len {
            let name = match struct_name_dic.get(&td_idx) {
                Some(n) => n.clone(),
                None => continue,
            };
            if name.is_empty() { continue; }
            writeln!(out, "DO_TYPEDEF(0x00000000, {name});").ok();
        }
        out
    }
}

const TEMPLATE_DLLMAIN_CPP: &str = r#"// Generated C++ file by Il2CppDumper - Rust Edition
// DLL entry point

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "il2cpp-init.h"
#include "main.h"

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        init_il2cpp();
        CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) Run, NULL, 0, NULL);
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
"#;

const TEMPLATE_HELPERS_CPP: &str = r#"// Generated C++ file by Il2CppDumper - Rust Edition
// Helper functions

#include "pch-il2cpp.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#include <codecvt>
#include "helpers.h"

extern const LPCWSTR LOG_FILE;

uintptr_t il2cppi_get_base_address() {
    return (uintptr_t) GetModuleHandleW(L"GameAssembly.dll");
}

void il2cppi_log_write(std::string text) {
    HANDLE hfile = CreateFileW(LOG_FILE, FILE_APPEND_DATA, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hfile == INVALID_HANDLE_VALUE)
        MessageBoxW(0, L"Could not open log file", 0, 0);
    DWORD written;
    WriteFile(hfile, text.c_str(), (DWORD) text.length(), &written, NULL);
    WriteFile(hfile, "\r\n", 2, &written, NULL);
    CloseHandle(hfile);
}

void il2cppi_new_console() {
    AllocConsole();
    freopen_s((FILE**) stdout, "CONOUT$", "w", stdout);
}

#if _MSC_VER >= 1920
std::string il2cppi_to_string(Il2CppString* str) {
    std::u16string u16(reinterpret_cast<const char16_t*>(str->chars));
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>{}.to_bytes(u16);
}

std::string il2cppi_to_string(app::String* str) {
    return il2cppi_to_string(reinterpret_cast<Il2CppString*>(str));
}
#endif
"#;

const TEMPLATE_HELPERS_H: &str = r#"// Generated C++ file by Il2CppDumper - Rust Edition
// Helper functions

#pragma once

#include <string>
#include <sstream>
#include <iomanip>

#include "il2cpp-metadata-version.h"

uintptr_t il2cppi_get_base_address();
void il2cppi_log_write(std::string text);
void il2cppi_new_console();

#if _MSC_VER >= 1920
std::string il2cppi_to_string(Il2CppString* str);
std::string il2cppi_to_string(app::String* str);
#endif

template<typename T> bool il2cppi_is_initialized(T* metadataItem) {
#if __IL2CPP_METADATA_VERSION < 270
    return *metadataItem != 0;
#else
    return !((uintptr_t) *metadataItem & 1);
#endif
}

template<typename T> std::string to_hex_string(T i) {
    std::stringstream stream;
    stream << "0x" << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex << i;
    return stream.str();
}
"#;

const TEMPLATE_IL2CPP_APPDATA_H: &str = r#"// Generated C++ file by Il2CppDumper - Rust Edition
// IL2CPP application data

#pragma once

#include <cstdint>

#include "il2cpp-types.h"
#include "il2cpp-api-functions-ptr.h"

#define DO_API(r, n, p) extern r (*n) p
#include "il2cpp-api-functions.h"
#undef DO_API

#define DO_APP_FUNC(a, r, n, p) extern r (*n) p
#define DO_APP_FUNC_METHODINFO(a, n) extern struct MethodInfo ** n
namespace app {
    #include "il2cpp-functions.h"
}
#undef DO_APP_FUNC
#undef DO_APP_FUNC_METHODINFO

#define DO_TYPEDEF(a, n) extern n ## __Class** n ## __TypeInfo
namespace app {
    #include "il2cpp-types-ptr.h"
}
#undef DO_TYPEDEF
"#;

const TEMPLATE_IL2CPP_INIT_CPP: &str = r#"// Generated C++ file by Il2CppDumper - Rust Edition
// IL2CPP application initializer

#include "pch-il2cpp.h"

#include "il2cpp-appdata.h"
#include "il2cpp-init.h"
#include "helpers.h"

#define DO_API(r, n, p) r (*n) p
#include "il2cpp-api-functions.h"
#undef DO_API

#define DO_APP_FUNC(a, r, n, p) r (*n) p
#define DO_APP_FUNC_METHODINFO(a, n) struct MethodInfo ** n
namespace app {
#include "il2cpp-functions.h"
}
#undef DO_APP_FUNC
#undef DO_APP_FUNC_METHODINFO

#define DO_TYPEDEF(a, n) n ## __Class** n ## __TypeInfo
namespace app {
#include "il2cpp-types-ptr.h"
}
#undef DO_TYPEDEF

void init_il2cpp()
{
    uintptr_t baseAddress = il2cppi_get_base_address();

    using namespace app;

    #define DO_API(r, n, p) n = (r (*) p)(baseAddress + n ## _ptr)
    #include "il2cpp-api-functions.h"
    #undef DO_API

    #define DO_APP_FUNC(a, r, n, p) n = (r (*) p)(baseAddress + a)
    #define DO_APP_FUNC_METHODINFO(a, n) n = (struct MethodInfo **)(baseAddress + a)
    #include "il2cpp-functions.h"
    #undef DO_APP_FUNC
    #undef DO_APP_FUNC_METHODINFO

    #define DO_TYPEDEF(a, n) n ## __TypeInfo = (n ## __Class**) (baseAddress + a);
    #include "il2cpp-types-ptr.h"
    #undef DO_TYPEDEF
}
"#;

const TEMPLATE_IL2CPP_INIT_H: &str = r#"// Generated C++ file by Il2CppDumper - Rust Edition
// IL2CPP application initializer

#pragma once

void init_il2cpp();
"#;

const TEMPLATE_PCH_IL2CPP_CPP: &str = r#"// pch.cpp: source file corresponding to the pre-compiled header

#include "pch-il2cpp.h"
"#;

const TEMPLATE_PCH_IL2CPP_H: &str = r#"// pch.h: This is a precompiled header file.

#ifndef PCH_IL2CPP_H
#define PCH_IL2CPP_H

#endif
"#;

const TEMPLATE_MAIN_CPP: &str = r#"// Generated C++ file by Il2CppDumper - Rust Edition
// Custom injected code entry point

#include "pch-il2cpp.h"

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <iostream>
#include "il2cpp-appdata.h"
#include "helpers.h"

using namespace app;

extern const LPCWSTR LOG_FILE = L"il2cpp-log.txt";

void Run()
{
    il2cpp_thread_attach(il2cpp_domain_get());
}
"#;

const TEMPLATE_MAIN_H: &str = r#"// Generated C++ file by Il2CppDumper - Rust Edition
// Custom injected code entry point

#pragma once

void Run();
"#;

const TEMPLATE_VCXPROJ: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <VCProjectVersion>16.0</VCProjectVersion>
    <ProjectGuid>{%PROJECTGUID%}</ProjectGuid>
    <RootNamespace>IL2CppDLL</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>DynamicLibrary</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>DynamicLibrary</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ItemDefinitionGroup>
    <ClCompile>
      <PrecompiledHeader>Use</PrecompiledHeader>
      <PrecompiledHeaderFile>pch-il2cpp.h</PrecompiledHeaderFile>
      <LanguageStandard>stdcpp17</LanguageStandard>
      <AdditionalIncludeDirectories>framework;appdata;user;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
    </ClCompile>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClInclude Include="appdata\il2cpp-types.h" />
    <ClInclude Include="appdata\il2cpp-types-ptr.h" />
    <ClInclude Include="appdata\il2cpp-functions.h" />
    <ClInclude Include="appdata\il2cpp-api-functions.h" />
    <ClInclude Include="appdata\il2cpp-api-functions-ptr.h" />
    <ClInclude Include="appdata\il2cpp-metadata-version.h" />
    <ClInclude Include="framework\helpers.h" />
    <ClInclude Include="framework\il2cpp-appdata.h" />
    <ClInclude Include="framework\il2cpp-init.h" />
    <ClInclude Include="framework\pch-il2cpp.h" />
    <ClInclude Include="user\main.h" />
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="framework\dllmain.cpp" />
    <ClCompile Include="framework\helpers.cpp" />
    <ClCompile Include="framework\il2cpp-init.cpp" />
    <ClCompile Include="framework\pch-il2cpp.cpp">
      <PrecompiledHeader>Create</PrecompiledHeader>
    </ClCompile>
    <ClCompile Include="user\main.cpp" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
</Project>
"#;

const TEMPLATE_VCXPROJ_FILTERS: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <Filter Include="appdata">
      <UniqueIdentifier>{%GUID1%}</UniqueIdentifier>
    </Filter>
    <Filter Include="framework">
      <UniqueIdentifier>{%GUID2%}</UniqueIdentifier>
    </Filter>
    <Filter Include="user">
      <UniqueIdentifier>{%GUID3%}</UniqueIdentifier>
    </Filter>
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="appdata\il2cpp-types.h"><Filter>appdata</Filter></ClInclude>
    <ClInclude Include="appdata\il2cpp-types-ptr.h"><Filter>appdata</Filter></ClInclude>
    <ClInclude Include="appdata\il2cpp-functions.h"><Filter>appdata</Filter></ClInclude>
    <ClInclude Include="appdata\il2cpp-api-functions.h"><Filter>appdata</Filter></ClInclude>
    <ClInclude Include="appdata\il2cpp-api-functions-ptr.h"><Filter>appdata</Filter></ClInclude>
    <ClInclude Include="appdata\il2cpp-metadata-version.h"><Filter>appdata</Filter></ClInclude>
    <ClInclude Include="framework\helpers.h"><Filter>framework</Filter></ClInclude>
    <ClInclude Include="framework\il2cpp-appdata.h"><Filter>framework</Filter></ClInclude>
    <ClInclude Include="framework\il2cpp-init.h"><Filter>framework</Filter></ClInclude>
    <ClInclude Include="framework\pch-il2cpp.h"><Filter>framework</Filter></ClInclude>
    <ClInclude Include="user\main.h"><Filter>user</Filter></ClInclude>
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="framework\dllmain.cpp"><Filter>framework</Filter></ClCompile>
    <ClCompile Include="framework\helpers.cpp"><Filter>framework</Filter></ClCompile>
    <ClCompile Include="framework\il2cpp-init.cpp"><Filter>framework</Filter></ClCompile>
    <ClCompile Include="framework\pch-il2cpp.cpp"><Filter>framework</Filter></ClCompile>
    <ClCompile Include="user\main.cpp"><Filter>user</Filter></ClCompile>
  </ItemGroup>
</Project>
"#;

const TEMPLATE_SLN: &str = r#"Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "%PROJECTNAME%", "%PROJECTFILE%", "{%PROJECTGUID%}"
EndProject
Global
    GlobalSection(SolutionConfigurationPlatforms) = preSolution
        Debug|x64 = Debug|x64
        Release|x64 = Release|x64
    EndGlobalSection
    GlobalSection(ProjectConfigurationPlatforms) = postSolution
        {%PROJECTGUID%}.Debug|x64.ActiveCfg = Debug|x64
        {%PROJECTGUID%}.Debug|x64.Build.0 = Debug|x64
        {%PROJECTGUID%}.Release|x64.ActiveCfg = Release|x64
        {%PROJECTGUID%}.Release|x64.Build.0 = Release|x64
    EndGlobalSection
    GlobalSection(SolutionProperties) = preSolution
        HideSolutionNode = FALSE
    EndGlobalSection
    GlobalSection(ExtensibilityGlobals) = postSolution
        SolutionGuid = {%SOLUTIONGUID%}
    EndGlobalSection
EndGlobal
"#;

fn fix_name_scaffold(name: &str) -> String {
    crate::utils::sanitize_cpp_identifier(
        name,
        crate::utils::NameSanitizerOptions {
            allow_dollar: true,
            avoid_double_underscore_prefix: true,
        },
    )
}
