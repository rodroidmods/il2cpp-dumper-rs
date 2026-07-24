use std::collections::{BTreeSet, HashMap};
use crate::io::{checked_array_len, BinaryStream};
use crate::error::{Result, Error};
use crate::search::SearchSection;
use crate::disassembler::Architecture;
use super::structures::*;

pub fn auto_plus_count_limit(is_wasm: bool) -> u64 {
    if is_wasm { 0x35000 } else { 0x50000 }
}

pub fn apply_auto_plus_heuristics(
    stream: &mut BinaryStream,
    mut code_registration: u64,
    cr: &Il2CppCodeRegistration,
    count_limit: u64,
) -> u64 {
    if code_registration == 0 || stream.version < 24.2 {
        return code_registration;
    }
    let ptr_size = stream.pointer_size() as u64;

    if stream.version == 31.0 {
        if cr.generic_method_pointers_count > count_limit {
            code_registration = code_registration.saturating_sub(ptr_size * 2);
        } else {
            stream.version = 29.0;
            eprintln!("Info: Change il2cpp version to: 29");
        }
    }
    if stream.version == 29.0 {
        if cr.generic_method_pointers_count > count_limit {
            stream.version = 29.1;
            code_registration = code_registration.saturating_sub(ptr_size * 2);
            eprintln!("Info: Change il2cpp version to: 29.1");
        }
    }
    if stream.version == 27.0 {
        if cr.reverse_pinvoke_wrapper_count > count_limit {
            stream.version = 27.1;
            code_registration = code_registration.saturating_sub(ptr_size);
            eprintln!("Info: Change il2cpp version to: 27.1");
        }
    }
    if stream.version == 24.4 {
        code_registration = code_registration.saturating_sub(ptr_size * 2);
        if cr.reverse_pinvoke_wrapper_count > count_limit {
            stream.version = 24.5;
            code_registration = code_registration.saturating_sub(ptr_size);
            eprintln!("Info: Change il2cpp version to: 24.5");
        }
    }
    if stream.version == 24.2 && cr.interop_data_count == 0 {
        stream.version = 24.3;
        code_registration = code_registration.saturating_sub(ptr_size * 2);
        eprintln!("Info: Change il2cpp version to: 24.3");
    }
    code_registration
}

pub fn refine_code_registration(
    stream: &mut BinaryStream,
    code_registration: u64,
    map_vatr: &dyn Fn(u64) -> Result<u64>,
    count_limit: u64,
) -> Result<u64> {
    if code_registration == 0 || stream.version < 24.2 {
        return Ok(code_registration);
    }
    let cr_offset = map_vatr(code_registration)?;
    stream.set_position(cr_offset);
    let version = stream.version;
    let cr = Il2CppCodeRegistration::read(stream, version)?;
    Ok(apply_auto_plus_heuristics(stream, code_registration, &cr, count_limit))
}

#[derive(Debug, Clone)]
pub struct VaSegment {
    pub vaddr: u64,
    pub memsz: u64,
    pub offset: u64,
}

pub struct Il2Cpp {
    pub stream: BinaryStream,
    pub version: f64,
    pub is_32bit: bool,

    pub types: Vec<Il2CppType>,
    pub method_pointers: Vec<u64>,
    pub generic_method_pointers: Vec<u64>,
    pub invoker_pointers: Vec<u64>,
    pub field_offsets: Vec<Vec<i32>>,
    pub metadata_usages: Vec<u64>,
    pub custom_attribute_generators: Vec<u64>,

    pub generic_insts: Vec<Il2CppGenericInst>,
    pub generic_inst_pointers: Vec<u64>,
    pub method_specs: Vec<Il2CppMethodSpec>,
    pub generic_method_table: Vec<Il2CppGenericMethodFunctionsDefinitions>,

    pub code_gen_modules: HashMap<String, Il2CppCodeGenModule>,
    pub code_gen_module_method_pointers: HashMap<String, Vec<u64>>,

    pub method_definition_method_specs: HashMap<usize, Vec<usize>>,
    pub method_spec_generic_method_pointers: HashMap<usize, u64>,

    pub code_registration: u64,
    pub metadata_registration: u64,
    pub is_dumped: bool,
    pub image_base: u64,

    field_offset_pointers: Vec<u64>,
    field_offsets_are_pointers: bool,
    type_dic: HashMap<u64, usize>,
    pub va_segments: Vec<VaSegment>,
    pub data_sections: Vec<SearchSection>,
    pub rgctxs_dictionary: HashMap<String, HashMap<u32, Vec<Il2CppRGCTXDefinition>>>,
    pub is_pe: bool,
    pub reverse_pinvoke_wrappers: Vec<u64>,
    pub unresolved_virtual_call_pointers: Vec<u64>,
    pub arch: Option<Architecture>,
    pub e_machine: u16,
    pub exported_symbols: Vec<String>,
    pub api_export_rvas: HashMap<String, u64>,
    pub codm: bool,
    pub type_definition_sizes: Vec<Il2CppTypeDefinitionSizes>,
}

impl Il2Cpp {
    pub fn new(mut stream: BinaryStream, version: f64, is_32bit: bool) -> Self {
        stream.version = version;
        stream.is_32bit = is_32bit;
        Self {
            stream,
            version,
            is_32bit,
            types: Vec::new(),
            method_pointers: Vec::new(),
            generic_method_pointers: Vec::new(),
            invoker_pointers: Vec::new(),
            field_offsets: Vec::new(),
            metadata_usages: Vec::new(),
            custom_attribute_generators: Vec::new(),
            generic_insts: Vec::new(),
            generic_inst_pointers: Vec::new(),
            method_specs: Vec::new(),
            generic_method_table: Vec::new(),
            code_gen_modules: HashMap::new(),
            code_gen_module_method_pointers: HashMap::new(),
            method_definition_method_specs: HashMap::new(),
            method_spec_generic_method_pointers: HashMap::new(),
            code_registration: 0,
            metadata_registration: 0,
            is_dumped: false,
            image_base: 0,
            field_offset_pointers: Vec::new(),
            field_offsets_are_pointers: false,
            type_dic: HashMap::new(),
            va_segments: Vec::new(),
            data_sections: Vec::new(),
            rgctxs_dictionary: HashMap::new(),
            is_pe: false,
            reverse_pinvoke_wrappers: Vec::new(),
            unresolved_virtual_call_pointers: Vec::new(),
            arch: None,
            e_machine: 0,
            exported_symbols: Vec::new(),
            api_export_rvas: HashMap::new(),
            codm: false,
            type_definition_sizes: Vec::new(),
        }
    }

    pub fn from_elf(elf: &crate::formats::elf::Elf) -> Self {
        let method_definition_method_specs: HashMap<usize, Vec<usize>> = elf
            .method_definition_method_specs
            .iter()
            .map(|(k, v)| (*k as usize, v.clone()))
            .collect();
        Self {
            stream: elf.stream.clone(),
            version: elf.stream.version,
            is_32bit: elf.is_32bit,
            types: elf.types.clone(),
            method_pointers: elf.method_pointers.clone(),
            generic_method_pointers: elf.generic_method_pointers.clone(),
            invoker_pointers: elf.invoker_pointers.clone(),
            field_offsets: Vec::new(),
            metadata_usages: elf.metadata_usages.clone(),
            custom_attribute_generators: elf.custom_attribute_generators.clone(),
            generic_insts: elf.generic_insts.clone(),
            generic_inst_pointers: elf.generic_inst_pointers.clone(),
            method_specs: elf.method_specs.clone(),
            generic_method_table: elf.generic_method_table.clone(),
            code_gen_modules: elf.code_gen_modules.clone(),
            code_gen_module_method_pointers: elf.code_gen_module_method_pointers.clone(),
            method_definition_method_specs,
            method_spec_generic_method_pointers: elf.method_spec_generic_method_pointers.clone(),
            code_registration: 0,
            metadata_registration: 0,
            is_dumped: elf.is_dumped,
            image_base: elf.stream.image_base,
            field_offset_pointers: elf.field_offsets.clone(),
            field_offsets_are_pointers: elf.field_offsets_are_pointers,
            type_dic: elf.type_dic.clone(),
            va_segments: elf.segments.iter().map(|s| VaSegment {
                vaddr: s.p_vaddr,
                memsz: s.p_memsz,
                offset: s.p_offset,
            }).collect(),
            data_sections: elf.segments.iter().filter(|s| {
                s.p_memsz != 0 && matches!(s.p_flags, 2 | 4 | 6)
            }).map(|s| SearchSection::new(
                s.p_offset,
                s.p_offset + s.p_filesz,
                s.p_vaddr,
                s.p_vaddr + s.p_memsz,
            )).collect(),
            rgctxs_dictionary: elf.rgctxs_dictionary.clone(),
            is_pe: false,
            reverse_pinvoke_wrappers: Vec::new(),
            unresolved_virtual_call_pointers: Vec::new(),
            arch: Architecture::from_elf_machine(elf.header.e_machine),
            e_machine: elf.header.e_machine,
            exported_symbols: Vec::new(),
            api_export_rvas: HashMap::new(),
            codm: elf.codm_diag,
            type_definition_sizes: elf.type_definition_sizes.clone(),
        }
    }

    pub fn init_with_auto_plus(
        &mut self,
        code_registration: u64,
        metadata_registration: u64,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
        is_wasm: bool,
    ) -> Result<()> {
        let orig_version = self.stream.version;
        let limit = auto_plus_count_limit(is_wasm);
        let cr = match refine_code_registration(
            &mut self.stream,
            code_registration,
            map_vatr,
            limit,
        ) {
            Ok(c) => c,
            Err(_) => code_registration,
        };
        self.version = self.stream.version;

        match self.init(cr, metadata_registration, map_vatr) {
            Ok(()) => Ok(()),
            Err(e) if orig_version == 31.0 && self.stream.version != 29.0 => {
                self.stream.version = 29.0;
                self.version = 29.0;
                eprintln!("Info: Retry init with il2cpp version 29 ({e})");
                self.init(cr, metadata_registration, map_vatr)
            }
            Err(e) => Err(e),
        }
    }

    pub fn init(
        &mut self,
        code_registration: u64,
        metadata_registration: u64,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Result<()> {
        self.code_registration = code_registration;
        self.metadata_registration = metadata_registration;
        self.stream.is_32bit = self.is_32bit;
        self.version = self.stream.version;


        let mr_offset = map_vatr(metadata_registration)?;
        let mr = {
            self.stream.set_position(mr_offset);
            Il2CppMetadataRegistration::read(&mut self.stream, self.version)?
        };



        let types_offset = map_vatr(mr.types)?;
        self.stream.set_position(types_offset);
        let type_ptrs = self.stream.read_ptr_array_inline(mr.types_count as usize)?;


        self.types.clear();
        self.type_dic.clear();
        let tolerant = self.codm && self.is_dumped;
        let mut decoded = 0usize;
        let mut skipped = 0usize;
        for (idx, ptr) in type_ptrs.iter().enumerate() {
            if tolerant {
                let t_offset = match map_vatr(*ptr) {
                    Ok(o) => o,
                    Err(_) => {
                        self.types.push(Il2CppType::default());
                        skipped += 1;
                        continue;
                    }
                };
                self.stream.set_position(t_offset);
                let mut t = match Il2CppType::read(&mut self.stream) {
                    Ok(t) => t,
                    Err(_) => {
                        self.types.push(Il2CppType::default());
                        skipped += 1;
                        continue;
                    }
                };
                let pre = t.type_enum;
                t.init_codm(self.version);
                if pre != t.type_enum {
                    decoded += 1;
                }
                self.types.push(t);
                self.type_dic.insert(*ptr, idx);
            } else {
                let t_offset = map_vatr(*ptr).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("type[{}] ptr=0x{:x}: {}", idx, ptr, e)))?;
                self.stream.set_position(t_offset);
                let mut t = Il2CppType::read(&mut self.stream)?;
                if self.codm {
                    t.init_codm(self.version);
                } else {
                    t.init(self.version);
                }
                self.types.push(t);
                self.type_dic.insert(*ptr, idx);
            }
        }
        if tolerant {
            eprintln!("[CODM] init_codm decoded {} of {} Il2CppType entries (skipped {})",
                decoded, type_ptrs.len(), skipped);
        }


        if mr.generic_insts_count > 0 {

            let gi_offset = map_vatr(mr.generic_insts)?;
            self.generic_inst_pointers = self.stream.read_ptr_array(gi_offset, mr.generic_insts_count as usize)?;

            self.generic_insts.clear();
            for ptr in &self.generic_inst_pointers.clone() {
                let offset = map_vatr(*ptr)?;
                self.stream.set_position(offset);
                self.generic_insts.push(Il2CppGenericInst::read(&mut self.stream)?);
            }

        }

        if mr.method_specs_count > 0 && mr.method_specs > 0 {

            let ms_offset = map_vatr(mr.method_specs)?;
            self.stream.set_position(ms_offset);
            self.method_specs.clear();
            for _ in 0..mr.method_specs_count {
                self.method_specs.push(Il2CppMethodSpec::read(&mut self.stream)?);
            }

        }

        if mr.generic_method_table > 0 && mr.generic_method_table_count > 0 {

            let gmt_offset = map_vatr(mr.generic_method_table)?;
            self.stream.set_position(gmt_offset);
            self.generic_method_table.clear();
            for _ in 0..mr.generic_method_table_count {
                self.generic_method_table.push(
                    Il2CppGenericMethodFunctionsDefinitions::read(&mut self.stream, self.version)?
                );
            }

        }


        let cr_offset = map_vatr(code_registration)?;
        self.stream.set_position(cr_offset);
        let cr = Il2CppCodeRegistration::read(&mut self.stream, self.version)?;


        if self.codm {
            if cr.method_pointers > 0 && cr.method_pointers_count > 0 {
                match map_vatr(cr.method_pointers) {
                    Ok(mp_offset) => match self.stream.read_ptr_array(mp_offset, cr.method_pointers_count as usize) {
                        Ok(v) => self.method_pointers = v,
                        Err(e) => eprintln!("[WARN] Failed to read method_pointers: {e}"),
                    },
                    Err(e) => eprintln!("[WARN] Failed to map method_pointers: {e}"),
                }
            }

            if cr.generic_method_pointers > 0 && cr.generic_method_pointers_count > 0 {
                match map_vatr(cr.generic_method_pointers) {
                    Ok(gmp_offset) => match self.stream.read_ptr_array(gmp_offset, cr.generic_method_pointers_count as usize) {
                        Ok(v) => self.generic_method_pointers = v,
                        Err(e) => eprintln!("[WARN] Failed to read generic_method_pointers: {e}"),
                    },
                    Err(e) => eprintln!("[WARN] Failed to map generic_method_pointers: {e}"),
                }
            }

            if cr.invoker_pointers > 0 && cr.invoker_pointers_count > 0 {
                match map_vatr(cr.invoker_pointers) {
                    Ok(ip_offset) => match self.stream.read_ptr_array(ip_offset, cr.invoker_pointers_count as usize) {
                        Ok(v) => self.invoker_pointers = v,
                        Err(e) => eprintln!("[WARN] Failed to read invoker_pointers: {e}"),
                    },
                    Err(e) => eprintln!("[WARN] Failed to map invoker_pointers: {e}"),
                }
            }

            if self.version < 27.0 && cr.custom_attribute_generators > 0 && cr.custom_attribute_count > 0 {
                match map_vatr(cr.custom_attribute_generators) {
                    Ok(ca_offset) => match self.stream.read_ptr_array(ca_offset, cr.custom_attribute_count as usize) {
                        Ok(v) => self.custom_attribute_generators = v,
                        Err(e) => eprintln!("[WARN] Failed to read custom_attribute_generators: {e}"),
                    },
                    Err(e) => eprintln!("[WARN] Failed to map custom_attribute_generators: {e}"),
                }
            }

            if mr.metadata_usages > 0 && mr.metadata_usages_count > 0 && self.version < 27.0 {
                match map_vatr(mr.metadata_usages) {
                    Ok(mu_offset) => match self.stream.read_ptr_array(mu_offset, mr.metadata_usages_count as usize) {
                        Ok(v) => self.metadata_usages = v,
                        Err(e) => eprintln!("[WARN] Failed to read metadata_usages: {e}"),
                    },
                    Err(e) => eprintln!("[WARN] Failed to map metadata_usages: {e}"),
                }
            }

            self.field_offsets_are_pointers = self.version > 21.0;
            if mr.field_offsets > 0 && mr.field_offsets_count > 0 {
                match map_vatr(mr.field_offsets) {
                    Ok(fo_offset) => {
                        if self.field_offsets_are_pointers {
                            match self.stream.read_ptr_array(fo_offset, mr.field_offsets_count as usize) {
                                Ok(v) => self.field_offset_pointers = v,
                                Err(e) => eprintln!("[WARN] Failed to read field_offset_pointers: {e}"),
                            }
                        } else if let Ok(n) = checked_array_len(
                            mr.field_offsets_count,
                            4,
                            Some(self.stream.remaining_from(fo_offset)),
                            "field_offsets(raw)",
                        ) {
                            self.stream.set_position(fo_offset);
                            let mut raw = Vec::with_capacity(n);
                            for _ in 0..n {
                                match self.stream.read_i32() {
                                    Ok(v) => raw.push(v),
                                    Err(_) => break,
                                }
                            }
                            self.field_offsets = vec![raw];
                        } else {
                            eprintln!("[WARN] field_offsets_count invalid: {}", mr.field_offsets_count);
                        }
                    },
                    Err(e) => eprintln!("[WARN] Failed to map field_offsets: {e}"),
                }
            }

            if mr.type_definitions_sizes > 0 && mr.type_definitions_sizes_count > 0 {
                if let Ok(sizes_offset) = map_vatr(mr.type_definitions_sizes) {
                    if let Ok(n) = checked_array_len(
                        mr.type_definitions_sizes_count,
                        std::mem::size_of::<Il2CppTypeDefinitionSizes>().max(1),
                        Some(self.stream.remaining_from(sizes_offset)),
                        "type_definition_sizes",
                    ) {
                        self.stream.set_position(sizes_offset);
                        self.type_definition_sizes.clear();
                        self.type_definition_sizes.reserve(n);
                        for _ in 0..n {
                            match Il2CppTypeDefinitionSizes::read(&mut self.stream) {
                                Ok(s) => self.type_definition_sizes.push(s),
                                Err(_) => break,
                            }
                        }
                    }
                }
            }
        } else {
            if cr.method_pointers > 0 && cr.method_pointers_count > 0 {
                let mp_offset = map_vatr(cr.method_pointers)?;
                self.method_pointers = self.stream.read_ptr_array(mp_offset, cr.method_pointers_count as usize)?;
            }

            if cr.generic_method_pointers > 0 && cr.generic_method_pointers_count > 0 {
                let gmp_offset = map_vatr(cr.generic_method_pointers)?;
                self.generic_method_pointers = self.stream.read_ptr_array(gmp_offset, cr.generic_method_pointers_count as usize)?;
            }

            if cr.invoker_pointers > 0 && cr.invoker_pointers_count > 0 {
                let ip_offset = map_vatr(cr.invoker_pointers)?;
                self.invoker_pointers = self.stream.read_ptr_array(ip_offset, cr.invoker_pointers_count as usize)?;
            }

            if self.version < 27.0 && cr.custom_attribute_generators > 0 && cr.custom_attribute_count > 0 {
                let ca_offset = map_vatr(cr.custom_attribute_generators)?;
                self.custom_attribute_generators = self.stream.read_ptr_array(ca_offset, cr.custom_attribute_count as usize)?;
            }

            if mr.metadata_usages > 0 && mr.metadata_usages_count > 0 && self.version < 27.0 {
                let mu_offset = map_vatr(mr.metadata_usages)?;
                self.metadata_usages = self.stream.read_ptr_array(mu_offset, mr.metadata_usages_count as usize)?;
            }

            self.field_offsets_are_pointers = self.version > 21.0;
            if mr.field_offsets > 0 && mr.field_offsets_count > 0 {
                let fo_offset = map_vatr(mr.field_offsets)?;
                if self.field_offsets_are_pointers {
                    self.field_offset_pointers = self.stream.read_ptr_array(fo_offset, mr.field_offsets_count as usize)?;
                } else {
                    let n = checked_array_len(
                        mr.field_offsets_count,
                        4,
                        Some(self.stream.remaining_from(fo_offset)),
                        "field_offsets(raw)",
                    )?;
                    self.stream.set_position(fo_offset);
                    let mut raw = Vec::with_capacity(n);
                    for _ in 0..n {
                        raw.push(self.stream.read_i32()?);
                    }
                    self.field_offsets = vec![raw];
                }
            }

            if mr.type_definitions_sizes > 0 && mr.type_definitions_sizes_count > 0 {
                let sizes_offset = map_vatr(mr.type_definitions_sizes)?;
                let n = checked_array_len(
                    mr.type_definitions_sizes_count,
                    std::mem::size_of::<Il2CppTypeDefinitionSizes>().max(1),
                    Some(self.stream.remaining_from(sizes_offset)),
                    "type_definition_sizes",
                )?;
                self.stream.set_position(sizes_offset);
                self.type_definition_sizes.clear();
                self.type_definition_sizes.reserve(n);
                for _ in 0..n {
                    self.type_definition_sizes.push(Il2CppTypeDefinitionSizes::read(&mut self.stream)?);
                }
            }
        }

        self.build_method_spec_lookup();

        if self.version >= 24.2 {

            self.load_code_gen_modules(&cr, map_vatr)?;

        }


        Ok(())
    }

    fn build_method_spec_lookup(&mut self) {
        self.method_definition_method_specs.clear();
        self.method_spec_generic_method_pointers.clear();

        for (table_idx, table) in self.generic_method_table.iter().enumerate() {
            if (table.generic_method_index as usize) < self.method_specs.len() {
                let ms = &self.method_specs[table.generic_method_index as usize];
                let method_def_idx = ms.method_definition_index as usize;

                self.method_definition_method_specs
                    .entry(method_def_idx)
                    .or_default()
                    .push(table.generic_method_index as usize);

                let method_idx = table.indices.method_index as usize;
                if method_idx < self.generic_method_pointers.len() {
                    self.method_spec_generic_method_pointers
                        .insert(table.generic_method_index as usize, self.generic_method_pointers[method_idx]);
                }
            }
            let _ = table_idx;
        }
    }

    fn load_code_gen_modules(
        &mut self,
        cr: &Il2CppCodeRegistration,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Result<()> {
        if cr.code_gen_modules == 0 || cr.code_gen_modules_count == 0 {
            return Ok(());
        }

        let modules_offset = map_vatr(cr.code_gen_modules)?;
        let module_ptrs = self.stream.read_ptr_array(modules_offset, cr.code_gen_modules_count as usize)?;

        for ptr in module_ptrs {
            let mod_offset = map_vatr(ptr)?;
            self.stream.set_position(mod_offset);
            let module = Il2CppCodeGenModule::read(&mut self.stream, self.version)?;

            let name_offset = map_vatr(module.module_name)?;
            let module_name = self.stream.read_string_to_null_at(name_offset)?;

            let method_ptrs = if module.method_pointers > 0 && module.method_pointer_count > 0 {
                let mp_offset = map_vatr(module.method_pointers)?;
                match self.stream.read_ptr_array(mp_offset, module.method_pointer_count as usize) {
                    Ok(v) => v,
                    Err(_) => {
                        let n = module.method_pointer_count as u64;
                        let max = self.stream.len() / self.stream.pointer_size() as u64;
                        if n > 0 && n <= max && n <= crate::io::MAX_BINARY_ARRAY_ELEMS as u64 {
                            vec![0u64; n as usize]
                        } else {
                            Vec::new()
                        }
                    }
                }
            } else {
                Vec::new()
            };

            self.code_gen_module_method_pointers.insert(module_name.clone(), method_ptrs);
            self.code_gen_modules.insert(module_name, module);
        }

        Ok(())
    }

    pub fn get_method_pointer(&self, image_name: &str, method_def: &Il2CppMethodDefinition) -> u64 {
        if self.version >= 24.2 {
            if let Some(ptrs) = self.code_gen_module_method_pointers.get(image_name) {
                let method_pointer_index = (method_def.token & 0x00FFFFFF) as usize;
                if method_pointer_index > 0 && method_pointer_index <= ptrs.len() {
                    return ptrs[method_pointer_index - 1];
                }
            }
            0
        } else {
            let idx = method_def.method_index as usize;
            if idx < self.method_pointers.len() {
                self.method_pointers[idx]
            } else {
                0
            }
        }
    }

    pub fn get_raw_field_offset(
        &self,
        type_def_index: usize,
        field_index_in_type: usize,
        field_index: usize,
    ) -> i32 {
        if self.field_offsets_are_pointers {
            if type_def_index >= self.field_offset_pointers.len() {
                return -1;
            }
            let ptr = self.field_offset_pointers[type_def_index];
            if ptr == 0 {
                return -1;
            }
            let target_va = ptr + (field_index_in_type as u64 * 4);
            let read_pos = match self.map_vatr(target_va) {
                Ok(p) => p,
                Err(_) => return -1,
            };
            self.stream.peek_i32_at(read_pos).unwrap_or(-1)
        } else if !self.field_offsets.is_empty() {
            let flat = &self.field_offsets[0];
            if field_index < flat.len() {
                flat[field_index]
            } else {
                -1
            }
        } else {
            -1
        }
    }

    pub fn get_field_offset_from_index(
        &self,
        type_def_index: usize,
        field_index_in_type: usize,
        field_index: usize,
        is_value_type: bool,
        is_static: bool,
    ) -> i32 {
        let raw = self.get_raw_field_offset(type_def_index, field_index_in_type, field_index);
        let decoded = super::field_layout::decode_field_offset(raw);
        if raw < 0 && !decoded.is_thread_static {
            return raw;
        }
        let offset = decoded.effective;

        if offset > 0 && is_value_type && !is_static {
            let header = if self.is_32bit { 8 } else { 16 };
            return offset - header;
        }

        offset
    }

    pub fn peek_generic_class(&self, addr: u64) -> Result<Il2CppGenericClass> {
        let offset = self.map_vatr(addr)?;
        let mut r = self.stream.slice_reader_at(offset);
        let type_definition_index;
        let type_ptr;
        if self.version >= 27.0 {
            type_definition_index = 0;
            type_ptr = r.read_ptr()?;
        } else {
            type_definition_index = r.read_ptr()?;
            type_ptr = 0;
        }
        let class_inst = r.read_ptr()?;
        let method_inst = r.read_ptr()?;
        let cached_class = r.read_ptr()?;
        Ok(Il2CppGenericClass {
            type_definition_index,
            type_ptr,
            context: Il2CppGenericContext { class_inst, method_inst },
            cached_class,
        })
    }

    pub fn peek_generic_inst(&self, addr: u64) -> Result<Il2CppGenericInst> {
        let offset = self.map_vatr(addr)?;
        let mut r = self.stream.slice_reader_at(offset);
        Ok(Il2CppGenericInst {
            type_argc: r.read_ptr()?,
            type_argv: r.read_ptr()?,
        })
    }

    pub fn peek_ptr_array(&self, addr: u64, count: u64) -> Result<Vec<u64>> {
        let offset = self.map_vatr(addr)?;
        let mut r = self.stream.slice_reader_at(offset);
        let n = crate::io::checked_array_len(
            count,
            self.stream.pointer_size(),
            Some(self.stream.remaining_from(offset)),
            "peek_ptr_array",
        )?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(r.read_ptr()?);
        }
        Ok(out)
    }

    pub fn get_rva(&self, pointer: u64) -> u64 {
        if self.image_base > 0 {
            pointer.wrapping_sub(self.image_base)
        } else {
            pointer
        }
    }

    pub fn get_il2cpp_type(&self, pointer: u64) -> Option<&Il2CppType> {
        self.type_dic.get(&pointer).and_then(|idx| self.types.get(*idx))
    }

    pub fn map_vatr(&self, addr: u64) -> Result<u64> {
        for seg in &self.va_segments {
            if addr >= seg.vaddr && addr <= seg.vaddr + seg.memsz {
                return Ok(addr - seg.vaddr + seg.offset);
            }
        }
        Err(Error::AddressNotMapped(addr))
    }

    pub fn map_rtva(&self, offset: u64) -> u64 {
        for seg in &self.va_segments {
            if offset >= seg.offset && offset <= seg.offset + seg.memsz {
                return offset - seg.offset + seg.vaddr;
            }
        }
        0
    }

    pub fn read_generic_class(&self, addr: u64) -> Result<Il2CppGenericClass> {
        self.peek_generic_class(addr)
    }

    pub fn read_generic_inst(&self, addr: u64) -> Result<Il2CppGenericInst> {
        self.peek_generic_inst(addr)
    }

    pub fn read_ptr_array(&self, addr: u64, count: u64) -> Result<Vec<u64>> {
        self.peek_ptr_array(addr, count)
    }

    pub fn detect_architecture(&self) -> Architecture {
        if let Some(arch) = self.arch {
            return arch;
        }
        Architecture::from_bitness(self.is_32bit, self.is_pe)
    }

    pub fn build_sorted_method_addresses(&self) -> Vec<u64> {
        let mut addrs = BTreeSet::new();

        for &ptr in &self.method_pointers {
            if ptr > 0 {
                addrs.insert(self.get_rva(ptr));
            }
        }

        for &ptr in &self.generic_method_pointers {
            if ptr > 0 {
                addrs.insert(self.get_rva(ptr));
            }
        }

        for ptrs in self.code_gen_module_method_pointers.values() {
            for &ptr in ptrs {
                if ptr > 0 {
                    addrs.insert(self.get_rva(ptr));
                }
            }
        }

        for (&_spec_idx, &ptr) in &self.method_spec_generic_method_pointers {
            if ptr > 0 {
                addrs.insert(self.get_rva(ptr));
            }
        }

        addrs.into_iter().collect()
    }

    pub fn get_method_body_size(&self, rva: u64, sorted_addrs: &[u64]) -> usize {
        const MAX_BODY: usize = 0x4000;
        const MIN_BODY: usize = 4;

        match sorted_addrs.binary_search(&rva) {
            Ok(idx) => {
                if idx + 1 < sorted_addrs.len() {
                    let next = sorted_addrs[idx + 1];
                    let diff = (next - rva) as usize;
                    diff.min(MAX_BODY).max(MIN_BODY)
                } else {
                    MAX_BODY
                }
            }
            Err(_) => MAX_BODY,
        }
    }

    pub fn read_bytes_at_rva(&self, rva: u64, size: usize) -> Option<Vec<u8>> {
        let va = if self.image_base > 0 {
            rva.wrapping_add(self.image_base)
        } else {
            rva
        };

        let file_offset = match self.map_vatr(va) {
            Ok(o) => o,
            Err(_) => {
                if rva < self.stream.len() {
                    rva
                } else {
                    return None;
                }
            }
        };

        let data = self.stream.data();
        let start = file_offset as usize;
        let end = (start + size).min(data.len());

        if start >= data.len() {
            return None;
        }

        Some(data[start..end].to_vec())
    }
}
