use std::collections::HashMap;
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::base::Il2Cpp;
use crate::il2cpp::enums::Il2CppTypeEnum;
use crate::il2cpp::structures::*;
use crate::executor::Il2CppExecutor;

pub struct MangledNameBuilder {
    buf: String,
    substitution_map: HashMap<String, usize>,
    current_sub_index: usize,
}

impl MangledNameBuilder {
    fn new() -> Self {
        Self {
            buf: String::from("_Z"),
            substitution_map: HashMap::new(),
            current_sub_index: 0,
        }
    }

    fn begin_name(&mut self) { self.buf.push('N'); }
    fn begin_generics(&mut self) { self.buf.push('I'); }
    fn write_end(&mut self) { self.buf.push('E'); }

    fn write_identifier(&mut self, id: &str) {
        let sanitized = crate::utils::sanitize_mangled_identifier_chars(id);
        self.buf.push_str(&sanitized.len().to_string());
        self.buf.push_str(&sanitized);
    }

    fn skip_sub_index(&mut self) {
        self.current_sub_index += 1;
    }

    fn try_write_substitution(&mut self, key: &str) -> bool {
        if let Some(&index) = self.substitution_map.get(key) {
            self.buf.push('S');
            if index > 0 {
                let mut idx = index - 1;
                const BASE36: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
                if idx == 0 {
                    self.buf.push(BASE36[0] as char);
                } else {
                    let mut chars: Vec<char> = Vec::new();
                    while idx > 0 {
                        chars.push(BASE36[idx % 36] as char);
                        idx /= 36;
                    }
                    chars.reverse();
                    for c in chars { self.buf.push(c); }
                }
            }
            self.buf.push('_');
            return true;
        }
        false
    }

    fn register_substitution(&mut self, key: String) {
        self.substitution_map.insert(key, self.current_sub_index);
        self.current_sub_index += 1;
    }

    fn finish(self) -> String { self.buf }

    fn base_clean(name: &str) -> &str {
        if let Some(pos) = name.find('`') { &name[..pos] } else { name }
    }

    fn type_def_key(td: &Il2CppTypeDefinition, td_index_guess: Option<usize>) -> String {
        match td_index_guess {
            Some(i) => format!("td#{i}::{}::{}", td.namespace_index, td.name_index),
            None => format!("td::{}::{}", td.namespace_index, td.name_index),
        }
    }

    fn write_type_name_from_td(
        &mut self,
        td: &Il2CppTypeDefinition,
        _td_index: usize,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        executor: &mut Il2CppExecutor,
        generic_inst: Option<&Il2CppGenericInst>,
    ) {
        let ns = metadata.get_string_from_index(td.namespace_index).unwrap_or_default();
        for part in ns.split('.') {
            if !part.is_empty() {
                self.write_identifier(part);
                self.skip_sub_index();
            }
        }

        if td.declaring_type_index != -1 {
            if let Some(declaring_type) = il2cpp.types.get(td.declaring_type_index as usize).cloned() {
                let decl_td_idx_opt = match Il2CppTypeEnum::from_u8(declaring_type.type_enum) {
                    Some(Il2CppTypeEnum::ValueType) | Some(Il2CppTypeEnum::Class) => {
                        Some(declaring_type.klass_index() as usize)
                    }
                    _ => None,
                };
                if let Some(dt_idx) = decl_td_idx_opt {
                    if let Some(decl_td) = metadata.type_defs.get(dt_idx).cloned() {
                        let decl_name_raw = metadata.get_string_from_index(decl_td.name_index).unwrap_or_default();
                        let decl_clean = Self::base_clean(&decl_name_raw);
                        self.write_identifier(decl_clean);
                        self.skip_sub_index();
                    } else {
                        let decl_name = executor.get_type_name(&declaring_type, metadata, il2cpp, false, false);
                        self.write_identifier(&decl_name);
                        self.skip_sub_index();
                    }
                } else {
                    let decl_name = executor.get_type_name(&declaring_type, metadata, il2cpp, false, false);
                    self.write_identifier(&decl_name);
                    self.skip_sub_index();
                }
            }
        }

        let name_raw = metadata.get_string_from_index(td.name_index).unwrap_or_default();
        let name_clean = Self::base_clean(&name_raw).to_string();
        self.write_identifier(&name_clean);

        if let Some(inst) = generic_inst {
            self.skip_sub_index();
            self.write_generic_inst(inst, il2cpp, metadata, executor);
        }
    }

    fn write_complex_type_from_td(
        &mut self,
        td: &Il2CppTypeDefinition,
        td_index: usize,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        executor: &mut Il2CppExecutor,
        generic_inst: Option<&Il2CppGenericInst>,
    ) {
        let key_base = Self::type_def_key(td, Some(td_index));
        let key = match generic_inst {
            Some(inst) => format!("{key_base}::gi@{:x}", inst as *const _ as usize),
            None => key_base,
        };
        if self.try_write_substitution(&key) {
            return;
        }
        self.begin_name();
        self.write_type_name_from_td(td, td_index, metadata, il2cpp, executor, generic_inst);
        self.write_end();
        self.register_substitution(key);
    }

    fn write_nested_system_type(
        &mut self,
        namespace: &str,
        type_name: &str,
    ) {
        let key = format!("sys::{namespace}::{type_name}");
        if self.try_write_substitution(&key) {
            return;
        }
        self.begin_name();
        for part in namespace.split('.') {
            if !part.is_empty() {
                self.write_identifier(part);
                self.skip_sub_index();
            }
        }
        self.write_identifier(type_name);
        self.write_end();
        self.register_substitution(key);
    }

    fn write_generic_inst(
        &mut self,
        inst: &Il2CppGenericInst,
        il2cpp: &Il2Cpp,
        metadata: &Metadata,
        executor: &mut Il2CppExecutor,
    ) {
        self.begin_generics();
        if let Ok(pointers) = il2cpp.read_ptr_array(inst.type_argv, inst.type_argc) {
            for ptr in &pointers {
                if let Some(t) = il2cpp.get_il2cpp_type(*ptr).cloned() {
                    self.write_type(&t, metadata, il2cpp, executor);
                } else {
                    self.buf.push('v');
                }
            }
        }
        self.write_end();
    }

    fn write_type(
        &mut self,
        il2cpp_type: &Il2CppType,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        executor: &mut Il2CppExecutor,
    ) {
        let te = Il2CppTypeEnum::from_u8(il2cpp_type.type_enum);

        if let Some(Il2CppTypeEnum::Void) = te {
            self.buf.push('v');
            return;
        }

        let mut is_nested_type_element = false;

        if il2cpp_type.byref == 1 {
            self.buf.push('R');
            is_nested_type_element = true;
        }

        match te {
            Some(Il2CppTypeEnum::Ptr) => {
                self.buf.push('P');
                is_nested_type_element = true;
                if il2cpp_type.datapoint != 0 {
                    if let Some(inner) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        self.write_type(&inner, metadata, il2cpp, executor);
                    } else {
                        self.buf.push('v');
                    }
                } else {
                    self.buf.push('v');
                }
                if is_nested_type_element { self.skip_sub_index(); }
                return;
            }
            Some(Il2CppTypeEnum::SzArray) | Some(Il2CppTypeEnum::Array) => {
                self.buf.push_str("A_");
                is_nested_type_element = true;
                if il2cpp_type.datapoint != 0 {
                    if let Some(element) = il2cpp.types.get(il2cpp_type.datapoint as usize).cloned() {
                        self.write_type(&element, metadata, il2cpp, executor);
                    } else {
                        self.buf.push('v');
                    }
                } else {
                    self.buf.push('v');
                }
                if is_nested_type_element { self.skip_sub_index(); }
                return;
            }
            _ => {}
        }

        match te {
            Some(Il2CppTypeEnum::Boolean) => self.buf.push('b'),
            Some(Il2CppTypeEnum::Char) => self.buf.push('w'),
            Some(Il2CppTypeEnum::I1) => self.buf.push('a'),
            Some(Il2CppTypeEnum::U1) => self.buf.push('h'),
            Some(Il2CppTypeEnum::I2) => self.buf.push('s'),
            Some(Il2CppTypeEnum::U2) => self.buf.push('t'),
            Some(Il2CppTypeEnum::I4) => self.buf.push('i'),
            Some(Il2CppTypeEnum::U4) => self.buf.push('j'),
            Some(Il2CppTypeEnum::I8) => self.buf.push('l'),
            Some(Il2CppTypeEnum::U8) => self.buf.push('m'),
            Some(Il2CppTypeEnum::R4) => self.buf.push('f'),
            Some(Il2CppTypeEnum::R8) => self.buf.push('d'),
            Some(Il2CppTypeEnum::I) | Some(Il2CppTypeEnum::U) => {
                self.buf.push_str("Pv");
            }
            Some(Il2CppTypeEnum::String) => {
                self.write_nested_system_type("System", "String");
            }
            Some(Il2CppTypeEnum::Object) => {
                self.write_nested_system_type("System", "Object");
            }
            Some(Il2CppTypeEnum::TypedByRef) => {
                self.write_nested_system_type("System", "TypedReference");
            }
            Some(Il2CppTypeEnum::ValueType) | Some(Il2CppTypeEnum::Class) => {
                let klass_idx = il2cpp_type.klass_index() as usize;
                if let Some(td) = metadata.type_defs.get(klass_idx).cloned() {
                    self.write_complex_type_from_td(&td, klass_idx, metadata, il2cpp, executor, None);
                } else {
                    self.buf.push('v');
                }
            }
            Some(Il2CppTypeEnum::GenericInst) => {
                let generic_class_ptr = il2cpp_type.generic_class();
                let mut wrote = false;
                if generic_class_ptr != 0 {
                    if let Ok(generic_class) = il2cpp.read_generic_class(generic_class_ptr) {
                        if let Some((td, td_idx)) = executor.get_generic_class_type_definition(&generic_class, metadata, il2cpp) {
                            let key = format!(
                                "td#{td_idx}::gi_ctx@0x{:x}",
                                generic_class.context.class_inst
                            );
                            if self.try_write_substitution(&key) {
                                wrote = true;
                            } else if let Ok(inst) = il2cpp.read_generic_inst(generic_class.context.class_inst) {
                                self.begin_name();
                                self.write_type_name_from_td(&td, td_idx, metadata, il2cpp, executor, Some(&inst));
                                self.write_end();
                                self.register_substitution(key);
                                wrote = true;
                            }
                        }
                    }
                }
                if !wrote { self.buf.push('v'); }
            }
            Some(Il2CppTypeEnum::Var) | Some(Il2CppTypeEnum::MVar) => {
                self.buf.push('v');
            }
            _ => {
                self.buf.push('v');
            }
        }

        if is_nested_type_element {
            self.skip_sub_index();
        }
    }

    pub fn mangle_method(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
        method_def: &Il2CppMethodDefinition,
    ) -> String {
        Self::mangle_method_inner(
            executor, metadata, il2cpp, type_def, type_def_index, method_def, None, "",
        )
    }

    pub fn mangle_method_info(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
        method_def: &Il2CppMethodDefinition,
    ) -> String {
        Self::mangle_method_inner(
            executor, metadata, il2cpp, type_def, type_def_index, method_def, None, "MethodInfo",
        )
    }

    pub fn mangle_method_spec(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
        method_def: &Il2CppMethodDefinition,
        generic_context: &Il2CppGenericContext,
    ) -> String {
        Self::mangle_method_inner(
            executor, metadata, il2cpp, type_def, type_def_index, method_def,
            Some(generic_context), "",
        )
    }

    pub fn mangle_type_info(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
    ) -> String {
        Self::mangle_data(executor, metadata, il2cpp, type_def, type_def_index, "TypeInfo")
    }

    pub fn mangle_type_ref(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
    ) -> String {
        Self::mangle_data(executor, metadata, il2cpp, type_def, type_def_index, "TypeRef")
    }

    fn mangle_data(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
        prefix: &str,
    ) -> String {
        let mut b = MangledNameBuilder::new();
        b.begin_name();

        if !prefix.is_empty() {
            b.write_identifier(prefix);
            b.skip_sub_index();
        }

        b.write_type_name_from_td(type_def, type_def_index, metadata, il2cpp, executor, None);
        let key = Self::type_def_key(type_def, Some(type_def_index));
        b.register_substitution(key);
        b.write_end();

        b.finish()
    }

    fn method_has_generic_params(method_def: &Il2CppMethodDefinition) -> bool {
        method_def.generic_container_index >= 0
    }

    fn mangle_method_inner(
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        type_def: &Il2CppTypeDefinition,
        type_def_index: usize,
        method_def: &Il2CppMethodDefinition,
        generic_context: Option<&Il2CppGenericContext>,
        prefix: &str,
    ) -> String {
        let mut b = MangledNameBuilder::new();
        b.begin_name();

        if !prefix.is_empty() {
            b.write_identifier(prefix);
            b.skip_sub_index();
        }

        let class_inst_opt = generic_context.and_then(|ctx| {
            if ctx.class_inst != 0 {
                il2cpp.read_generic_inst(ctx.class_inst).ok()
            } else {
                None
            }
        });

        b.write_type_name_from_td(type_def, type_def_index, metadata, il2cpp, executor, class_inst_opt.as_ref());
        let td_key_base = Self::type_def_key(type_def, Some(type_def_index));
        let td_key = match &class_inst_opt {
            Some(_) => match generic_context {
                Some(ctx) => format!("{td_key_base}::gi_ctx@0x{:x}", ctx.class_inst),
                None => td_key_base,
            },
            None => td_key_base,
        };
        b.register_substitution(td_key);

        let method_name = metadata.get_string_from_index(method_def.name_index as i32).unwrap_or_default();
        match method_name.as_str() {
            ".ctor" => b.buf.push_str("C1"),
            ".cctor" => b.write_identifier("cctor"),
            _ => b.write_identifier(&method_name),
        }

        let method_inst_opt = generic_context.and_then(|ctx| {
            if ctx.method_inst != 0 {
                il2cpp.read_generic_inst(ctx.method_inst).ok()
            } else {
                None
            }
        });

        let is_generic_method = method_inst_opt.is_some() || Self::method_has_generic_params(method_def);

        if let Some(inst) = &method_inst_opt {
            b.skip_sub_index();
            b.write_generic_inst(inst, il2cpp, metadata, executor);
        }

        b.write_end();

        let has_return_first = is_generic_method && prefix.is_empty();

        if has_return_first {
            if let Some(ret_type) = il2cpp.types.get(method_def.return_type as usize).cloned() {
                b.write_type(&ret_type, metadata, il2cpp, executor);
            } else {
                b.buf.push('v');
            }
        }

        if prefix == "MethodInfo" {
            b.buf.push('v');
        } else if method_def.parameter_count == 0 {
            if !has_return_first {
                b.buf.push('v');
            } else {
            }
        } else {
            for j in 0..method_def.parameter_count as usize {
                let param_def = metadata.parameter_defs[method_def.parameter_start as usize + j].clone();
                let param_type = il2cpp.types[param_def.type_index as usize].clone();
                b.write_type(&param_type, metadata, il2cpp, executor);
            }
        }

        b.skip_sub_index();

        b.finish()
    }
}
