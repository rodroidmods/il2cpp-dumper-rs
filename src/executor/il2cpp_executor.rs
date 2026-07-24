use std::collections::HashMap;
use crate::error::Result;
use crate::il2cpp::base::Il2Cpp;
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::enums::Il2CppTypeEnum;
use crate::il2cpp::structures::*;
use crate::utils::escape_string;

#[derive(Debug)]
pub enum DefaultValue {
    Bool(bool),
    U8(u8),
    I8(i8),
    Char(char),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    Null,
}

impl std::fmt::Display for DefaultValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefaultValue::Bool(v) => write!(f, "{}", if *v { "true" } else { "false" }),
            DefaultValue::U8(v) => write!(f, "{v}"),
            DefaultValue::I8(v) => write!(f, "{v}"),
            DefaultValue::Char(v) => {
                let escaped = match *v {
                    '\\' => "'\\\\'" .to_string(),
                    '\'' => "'\\''" .to_string(),
                    '\n' => "'\\n'" .to_string(),
                    '\r' => "'\\r'" .to_string(),
                    '\t' => "'\\t'" .to_string(),
                    '\0' => "'\\0'" .to_string(),
                    c if c.is_control() => format!("'\\x{:04X}'", c as u32),
                    c => format!("'{c}'"),
                };
                write!(f, "{escaped}")
            }
            DefaultValue::U16(v) => write!(f, "{v}"),
            DefaultValue::I16(v) => write!(f, "{v}"),
            DefaultValue::U32(v) => write!(f, "{v}"),
            DefaultValue::I32(v) => write!(f, "{v}"),
            DefaultValue::U64(v) => write!(f, "{v}"),
            DefaultValue::I64(v) => write!(f, "{v}"),
            DefaultValue::F32(v) => write!(f, "{v}"),
            DefaultValue::F64(v) => write!(f, "{v}"),
            DefaultValue::String(v) => write!(f, "\"{}\"", escape_string(v)),
            DefaultValue::Null => write!(f, "null"),
        }
    }
}


pub struct Il2CppExecutor {
    pub custom_attribute_generators: Vec<u64>,
    type_name_cache: HashMap<(u64, u32, bool, bool), String>,
    type_def_name_cache: HashMap<(usize, bool, bool), String>,
    generic_class_cache: HashMap<u64, Il2CppGenericClass>,
    generic_inst_cache: HashMap<u64, Il2CppGenericInst>,
    generic_inst_params_cache: HashMap<(u64, u64), String>,
    generic_container_params_cache: HashMap<(i32, i32), String>,
    modifier_cache: HashMap<u32, String>,
}

impl Il2CppExecutor {
    pub fn new(metadata: &Metadata, il2cpp: &mut Il2Cpp) -> Result<Self> {
        let mut custom_attribute_generators = Vec::new();

        if il2cpp.version >= 27.0 && il2cpp.version < 29.0 {
            let total_count: usize = metadata.image_defs.iter()
                .map(|img| img.custom_attribute_count as usize)
                .sum();
            custom_attribute_generators.resize(total_count, 0u64);
        } else if il2cpp.version < 27.0 {
            custom_attribute_generators = il2cpp.custom_attribute_generators.clone();
        }

        Ok(Self {
            custom_attribute_generators,
            type_name_cache: HashMap::new(),
            type_def_name_cache: HashMap::new(),
            generic_class_cache: HashMap::new(),
            generic_inst_cache: HashMap::new(),
            generic_inst_params_cache: HashMap::new(),
            generic_container_params_cache: HashMap::new(),
            modifier_cache: HashMap::new(),
        })
    }

    pub fn get_type_name(
        &mut self,
        il2cpp_type: &Il2CppType,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        add_namespace: bool,
        is_nested: bool,
    ) -> String {
        let cache_key = (il2cpp_type.datapoint, il2cpp_type.bits, add_namespace, is_nested);
        if let Some(cached) = self.type_name_cache.get(&cache_key) {
            return cached.clone();
        }

        let result = self.get_type_name_impl(il2cpp_type, metadata, il2cpp, add_namespace, is_nested);
        self.type_name_cache.insert(cache_key, result.clone());
        result
    }

    fn get_type_name_impl(
        &mut self,
        il2cpp_type: &Il2CppType,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        add_namespace: bool,
        is_nested: bool,
    ) -> String {
        let type_enum = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);

        match type_enum {
            Some(Il2CppTypeEnum::SzArray) => {
                let element_ptr = il2cpp_type.type_ptr();
                if let Some(element_type) = il2cpp.get_il2cpp_type(element_ptr).cloned() {
                    let element_name = self.get_type_name(&element_type, metadata, il2cpp, add_namespace, false);
                    return format!("{element_name}[]");
                }
                "object[]".to_string()
            }
            Some(Il2CppTypeEnum::Array) => {
                let arr_ptr = il2cpp_type.array();
                if let Some(element_type) = il2cpp.get_il2cpp_type(arr_ptr).cloned() {
                    let element_name = self.get_type_name(&element_type, metadata, il2cpp, add_namespace, false);
                    return format!("{element_name}[,]");
                }
                "object[]".to_string()
            }
            Some(Il2CppTypeEnum::Ptr) => {
                if let Some(ori_type) = il2cpp.get_il2cpp_type(il2cpp_type.type_ptr()).cloned() {
                    let name = self.get_type_name(&ori_type, metadata, il2cpp, add_namespace, false);
                    return format!("{name}*");
                }
                "void*".to_string()
            }
            Some(Il2CppTypeEnum::Var) | Some(Il2CppTypeEnum::MVar) => {
                if let Some(param) = self.get_generic_parameter_from_type(il2cpp_type, metadata, il2cpp) {
                    return metadata.get_string_from_index(param.name_index as i32).unwrap_or_else(|_| "T".to_string());
                }
                "T".to_string()
            }
            Some(Il2CppTypeEnum::Class) | Some(Il2CppTypeEnum::ValueType) | Some(Il2CppTypeEnum::GenericInst) => {
                let mut result = String::new();

                let (type_def_opt, generic_class_opt) = if type_enum == Some(Il2CppTypeEnum::GenericInst) {
                    let gc_addr = il2cpp_type.generic_class();
                    let gc = if let Some(cached) = self.generic_class_cache.get(&gc_addr) {
                        cached.clone()
                    } else {
                        match il2cpp.peek_generic_class(gc_addr) {
                            Ok(gc) => {
                                self.generic_class_cache.insert(gc_addr, gc.clone());
                                gc
                            }
                            Err(_) => return "object".to_string(),
                        }
                    };

                    let td = self.get_generic_class_type_definition(&gc, metadata, il2cpp).map(|(td, _)| td);
                    (td, Some(gc))
                } else {
                    let td = self.get_type_definition_from_type(il2cpp_type, metadata, il2cpp);
                    (td, None)
                };

                let type_def = match type_def_opt {
                    Some(td) => td,
                    None => return "object".to_string(),
                };

                if type_def.declaring_type_index != -1 {
                    if let Some(declaring_type) = il2cpp.types.get(type_def.declaring_type_index as usize).cloned() {
                        let declaring_name = self.get_type_name(&declaring_type, metadata, il2cpp, add_namespace, true);
                        result.push_str(&declaring_name);
                        result.push('.');
                    }
                } else if add_namespace {
                    if let Ok(ns) = metadata.get_string_from_index(type_def.namespace_index) {
                        if !ns.is_empty() {
                            result.push_str(&ns);
                            result.push('.');
                        }
                    }
                }

                let mut type_name = metadata.get_string_from_index(type_def.name_index)
                    .unwrap_or_else(|_| "?".to_string());

                if let Some(backtick_idx) = type_name.find('`') {
                    type_name.truncate(backtick_idx);
                }

                result.push_str(&type_name);

                if is_nested {
                    return result;
                }

                if let Some(gc) = generic_class_opt {
                    let gi_addr = gc.context.class_inst;
                    let gi = if let Some(cached) = self.generic_inst_cache.get(&gi_addr) {
                        cached.clone()
                    } else {
                        match il2cpp.peek_generic_inst(gi_addr) {
                            Ok(gi) => {
                                self.generic_inst_cache.insert(gi_addr, gi.clone());
                                gi
                            }
                            Err(_) => return result,
                        }
                    };
                    let params = self.get_generic_inst_params(&gi, metadata, il2cpp);
                    result.push_str(&params);
                } else if type_def.generic_container_index >= 0 {
                    if let Some(gc) = metadata.generic_containers.get(type_def.generic_container_index as usize) {
                        let gc = gc.clone();
                        let params = self.get_generic_container_params(&gc, metadata);
                        result.push_str(&params);
                    }
                }

                result
            }
            _ => {
                if let Some(te) = type_enum {
                    if let Some(name) = te.type_name() {
                        return name.to_string();
                    }
                }
                let masked = il2cpp_type.type_enum & 0x7F;
                if masked != il2cpp_type.type_enum {
                    if let Some(te) = Il2CppTypeEnum::from_u8(masked) {
                        if let Some(name) = te.type_name() {
                            return name.to_string();
                        }
                    }
                }
                "object".to_string()
            }
        }
    }

    pub fn get_type_def_name(
        &mut self,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        add_namespace: bool,
        generic_parameter: bool,
    ) -> String {
        let cache_key = (type_def_index, add_namespace, generic_parameter);
        if let Some(cached) = self.type_def_name_cache.get(&cache_key) {
            return cached.clone();
        }

        let mut prefix = String::new();

        if type_def.declaring_type_index != -1 {
            if let Some(declaring_type) = il2cpp.types.get(type_def.declaring_type_index as usize).cloned() {
                let declaring_name = self.get_type_name(&declaring_type, metadata, il2cpp, add_namespace, true);
                prefix = format!("{declaring_name}.");
            }
        } else if add_namespace {
            if let Ok(ns) = metadata.get_string_from_index(type_def.namespace_index) {
                if !ns.is_empty() {
                    prefix = format!("{ns}.");
                }
            }
        }

        let mut type_name = metadata.get_string_from_index(type_def.name_index)
            .unwrap_or_else(|_| "?".to_string());

        if type_def.generic_container_index >= 0 {
            if let Some(backtick_idx) = type_name.find('`') {
                type_name.truncate(backtick_idx);
            }
            if generic_parameter {
                if let Some(gc) = metadata.generic_containers.get(type_def.generic_container_index as usize) {
                    let gc = gc.clone();
                    let params = self.get_generic_container_params(&gc, metadata);
                    type_name.push_str(&params);
                }
            }
        }

        let result = format!("{prefix}{type_name}");
        self.type_def_name_cache.insert(cache_key, result.clone());
        result
    }

    pub fn get_generic_inst_params(
        &mut self,
        generic_inst: &Il2CppGenericInst,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> String {
        let cache_key = (generic_inst.type_argv, generic_inst.type_argc);
        if let Some(cached) = self.generic_inst_params_cache.get(&cache_key) {
            return cached.clone();
        }

        let mut param_names = Vec::new();

        let pointers = match il2cpp.peek_ptr_array(generic_inst.type_argv, generic_inst.type_argc) {
            Ok(p) => p,
            Err(_) => {
                let result = "<>".to_string();
                self.generic_inst_params_cache.insert(cache_key, result.clone());
                return result;
            }
        };

        for ptr in pointers {
            if let Some(t) = il2cpp.get_il2cpp_type(ptr).cloned() {
                param_names.push(self.get_type_name(&t, metadata, il2cpp, false, false));
            } else {
                param_names.push("?".to_string());
            }
        }

        let result = format!("<{}>", param_names.join(", "));
        self.generic_inst_params_cache.insert(cache_key, result.clone());
        result
    }

    pub fn get_generic_container_params(
        &mut self,
        generic_container: &Il2CppGenericContainer,
        metadata: &Metadata,
    ) -> String {
        let cache_key = (generic_container.generic_parameter_start, generic_container.type_argc);
        if let Some(cached) = self.generic_container_params_cache.get(&cache_key) {
            return cached.clone();
        }

        let mut param_names = Vec::new();
        for i in 0..generic_container.type_argc {
            let param_index = generic_container.generic_parameter_start + i;
            if let Some(param) = metadata.generic_parameters.get(param_index as usize) {
                let name = metadata.get_string_from_index(param.name_index as i32)
                    .unwrap_or_else(|_| "T".to_string());
                param_names.push(name);
            }
        }

        let result = format!("<{}>", param_names.join(", "));
        self.generic_container_params_cache.insert(cache_key, result.clone());
        result
    }

    pub fn get_method_spec_name(
        &mut self,
        method_spec_index: usize,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        add_namespace: bool,
    ) -> (String, String) {
        let method_spec = il2cpp.method_specs[method_spec_index].clone();

        let method_def_index = method_spec.method_definition_index;
        if method_def_index < 0 || method_def_index as usize >= metadata.method_defs.len() {
            return (
                format!("UnknownType_MethodSpec{method_spec_index}"),
                "UnknownMethod".to_string(),
            );
        }
        let method_def = metadata.method_defs[method_def_index as usize].clone();

        let declaring_type_index = method_def.declaring_type;
        if declaring_type_index < 0 || declaring_type_index as usize >= metadata.type_defs.len() {
            return (
                format!("UnknownType_MethodSpec{method_spec_index}"),
                "UnknownMethod".to_string(),
            );
        }
        let type_def = metadata.type_defs[declaring_type_index as usize].clone();
        let type_def_index = declaring_type_index as usize;

        let mut type_name = self.get_type_def_name(&type_def, type_def_index, metadata, il2cpp, add_namespace, false);

        if method_spec.class_index_index != -1 {
            if let Some(class_inst) = il2cpp.generic_insts.get(method_spec.class_index_index as usize).cloned() {
                let params = self.get_generic_inst_params(&class_inst, metadata, il2cpp);
                type_name.push_str(&params);
            }
        }

        let mut method_name = metadata.get_string_from_index(method_def.name_index as i32)
            .unwrap_or_else(|_| "?".to_string());

        if method_spec.method_index_index != -1 {
            if let Some(method_inst) = il2cpp.generic_insts.get(method_spec.method_index_index as usize).cloned() {
                let params = self.get_generic_inst_params(&method_inst, metadata, il2cpp);
                method_name.push_str(&params);
            }
        }

        (type_name, method_name)
    }

    pub fn try_get_default_value(
        &self,
        type_index: i32,
        data_index: i32,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> std::result::Result<DefaultValue, u64> {
        let pointer = metadata.get_default_value_offset(data_index);
        let default_value_type = match il2cpp.types.get(type_index as usize) {
            Some(t) => t,
            None => return Err(pointer),
        };

        let mut r = metadata.stream.slice_reader_at(pointer);
        let type_enum = Il2CppTypeEnum::from_u8(default_value_type.type_enum);

        match type_enum {
            Some(Il2CppTypeEnum::Boolean) => {
                r.read_bool().map(DefaultValue::Bool).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::U1) => {
                r.read_u8().map(DefaultValue::U8).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::I1) => {
                r.read_i8().map(DefaultValue::I8).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::Char) => {
                r.read_u16()
                    .map(|v| DefaultValue::Char(char::from_u32(v as u32).unwrap_or('\0')))
                    .map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::U2) => {
                r.read_u16().map(DefaultValue::U16).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::I2) => {
                r.read_i16().map(DefaultValue::I16).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::U4) => {
                if il2cpp.version >= 29.0 {
                    r.read_compressed_u32().map(DefaultValue::U32).map_err(|_| pointer)
                } else {
                    r.read_u32().map(DefaultValue::U32).map_err(|_| pointer)
                }
            }
            Some(Il2CppTypeEnum::I4) => {
                if il2cpp.version >= 29.0 {
                    r.read_compressed_i32().map(DefaultValue::I32).map_err(|_| pointer)
                } else {
                    r.read_i32().map(DefaultValue::I32).map_err(|_| pointer)
                }
            }
            Some(Il2CppTypeEnum::U8) => {
                r.read_u64().map(DefaultValue::U64).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::I8) => {
                r.read_i64().map(DefaultValue::I64).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::R4) => {
                r.read_f32().map(DefaultValue::F32).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::R8) => {
                r.read_f64().map(DefaultValue::F64).map_err(|_| pointer)
            }
            Some(Il2CppTypeEnum::String) => {
                if il2cpp.version >= 29.0 {
                    let length = r.read_compressed_i32().map_err(|_| pointer)?;
                    if length == -1 {
                        return Ok(DefaultValue::Null);
                    }
                    let bytes = r.read_bytes(length as usize).map_err(|_| pointer)?;
                    Ok(DefaultValue::String(String::from_utf8_lossy(bytes).to_string()))
                } else {
                    let length = r.read_i32().map_err(|_| pointer)?;
                    let s = r.read_string(length as usize).map_err(|_| pointer)?;
                    Ok(DefaultValue::String(s))
                }
            }
            _ => Err(pointer),
        }
    }

    pub fn new_for_worker(base: &Il2CppExecutor) -> Self {
        Self {
            custom_attribute_generators: base.custom_attribute_generators.clone(),
            type_name_cache: HashMap::new(),
            type_def_name_cache: HashMap::new(),
            generic_class_cache: HashMap::new(),
            generic_inst_cache: HashMap::new(),
            generic_inst_params_cache: HashMap::new(),
            generic_container_params_cache: HashMap::new(),
            modifier_cache: HashMap::new(),
        }
    }

    pub fn get_modifiers(&mut self, flags: u32) -> &str {
        if !self.modifier_cache.contains_key(&flags) {
            let mut result = String::new();
            let access = flags & 0x0007;
            match access {
                0x0001 => result.push_str("private "),
                0x0006 => result.push_str("public "),
                0x0004 => result.push_str("protected "),
                0x0003 | 0x0002 => result.push_str("internal "),
                0x0005 => result.push_str("protected internal "),
                _ => {}
            }
            if (flags & 0x0010) != 0 { result.push_str("static "); }
            if (flags & 0x0400) != 0 {
                result.push_str("abstract ");
                if (flags & 0x0100) == 0x0000 { result.push_str("override "); }
            } else if (flags & 0x0020) != 0 {
                if (flags & 0x0100) == 0x0000 { result.push_str("sealed override "); }
            } else if (flags & 0x0040) != 0 {
                if (flags & 0x0100) == 0x0100 { result.push_str("virtual "); }
                else { result.push_str("override "); }
            }
            if (flags & 0x2000) != 0 { result.push_str("extern "); }
            self.modifier_cache.insert(flags, result);
        }
        self.modifier_cache.get(&flags).unwrap()
    }

    pub fn get_generic_class_type_definition(
        &self,
        generic_class: &Il2CppGenericClass,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> Option<(Il2CppTypeDefinition, usize)> {
        if il2cpp.version >= 27.0 {
            let il2cpp_type = il2cpp.get_il2cpp_type(generic_class.type_ptr)?;
            let klass_idx = il2cpp_type.klass_index() as usize;
            metadata.type_defs.get(klass_idx).map(|td| (td.clone(), klass_idx))
        } else {
            let idx = generic_class.type_definition_index;
            if idx == u32::MAX as u64 || idx == u64::MAX || idx == 0 {
                return None;
            }
            let idx = idx as usize;
            metadata.type_defs.get(idx).map(|td| (td.clone(), idx))
        }
    }

    fn get_type_definition_from_type(
        &self,
        il2cpp_type: &Il2CppType,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> Option<Il2CppTypeDefinition> {
        self.get_type_definition_from_il2cpp_type(il2cpp_type, metadata, il2cpp)
    }

    fn get_type_definition_from_il2cpp_type(
        &self,
        il2cpp_type: &Il2CppType,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> Option<Il2CppTypeDefinition> {
        if il2cpp.version >= 27.0 && il2cpp.is_dumped {
            let handle = il2cpp_type.type_handle();
            let raw_offset = handle
                .wrapping_sub(il2cpp.image_base)
                .wrapping_sub(metadata.header.type_definitions_offset as u64);

            if !metadata.type_def_offset_to_index.is_empty() {
                let index = metadata.type_def_offset_to_index.get(&raw_offset)?;
                return metadata.type_defs.get(*index).cloned();
            }

            let td_size = Il2CppTypeDefinition::byte_size(metadata.version) as u64;
            if td_size == 0 { return None; }
            let index = (raw_offset / td_size) as usize;
            return metadata.type_defs.get(index).cloned();
        }
        let klass_index = il2cpp_type.klass_index() as i64;
        if klass_index >= 0 && (klass_index as usize) < metadata.type_defs.len() {
            return metadata.type_defs.get(klass_index as usize).cloned();
        }
        None
    }

    pub fn get_generic_parameter_from_type(
        &self,
        il2cpp_type: &Il2CppType,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> Option<Il2CppGenericParameter> {
        if il2cpp.version >= 27.0 && il2cpp.is_dumped {
            let handle = il2cpp_type.generic_parameter_handle();
            let raw_offset = handle
                .wrapping_sub(il2cpp.image_base)
                .wrapping_sub(metadata.header.generic_parameters_offset as u64);

            if !metadata.generic_param_offset_to_index.is_empty() {
                let index = metadata.generic_param_offset_to_index.get(&raw_offset)?;
                return metadata.generic_parameters.get(*index).cloned();
            }

            let param_size = Il2CppGenericParameter::byte_size(metadata.version) as u64;
            if param_size == 0 { return None; }
            let index = (raw_offset / param_size) as usize;
            return metadata.generic_parameters.get(index).cloned();
        }
        let param_index = il2cpp_type.generic_parameter_index() as i64;
        if param_index >= 0 && (param_index as usize) < metadata.generic_parameters.len() {
            return metadata.generic_parameters.get(param_index as usize).cloned();
        }
        None
    }

    pub fn get_rgctx_definition_for_type(
        &self,
        image_name: &str,
        type_def: &Il2CppTypeDefinition,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> Option<Vec<Il2CppRGCTXDefinition>> {
        if il2cpp.version >= 24.2 {
            if let Some(module_dic) = il2cpp.rgctxs_dictionary.get(image_name) {
                return module_dic.get(&type_def.token).cloned();
            }
            None
        } else {
            if type_def.rgctx_count > 0 && type_def.rgctx_start_index >= 0 {
                let start = type_def.rgctx_start_index as usize;
                let count = type_def.rgctx_count as usize;
                if start + count <= metadata.rgctx_entries.len() {
                    return Some(metadata.rgctx_entries[start..start + count].to_vec());
                }
            }
            None
        }
    }

    pub fn get_rgctx_definition_for_method(
        &self,
        image_name: &str,
        method_def: &Il2CppMethodDefinition,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> Option<Vec<Il2CppRGCTXDefinition>> {
        if il2cpp.version >= 24.2 {
            if let Some(module_dic) = il2cpp.rgctxs_dictionary.get(image_name) {
                return module_dic.get(&method_def.token).cloned();
            }
            None
        } else {
            if method_def.rgctx_count > 0 && method_def.rgctx_start_index >= 0 {
                let start = method_def.rgctx_start_index as usize;
                let count = method_def.rgctx_count as usize;
                if start + count <= metadata.rgctx_entries.len() {
                    return Some(metadata.rgctx_entries[start..start + count].to_vec());
                }
            }
            None
        }
    }

    pub fn get_method_spec_generic_context(
        &self,
        method_spec_index: usize,
        il2cpp: &Il2Cpp,
    ) -> (u64, u64) {
        let method_spec = &il2cpp.method_specs[method_spec_index];
        let class_inst_pointer = if method_spec.class_index_index >= 0 {
            il2cpp.generic_inst_pointers.get(method_spec.class_index_index as usize).copied().unwrap_or(0)
        } else { 0 };
        let method_inst_pointer = if method_spec.method_index_index >= 0 {
            il2cpp.generic_inst_pointers.get(method_spec.method_index_index as usize).copied().unwrap_or(0)
        } else { 0 };
        (class_inst_pointer, method_inst_pointer)
    }
}
