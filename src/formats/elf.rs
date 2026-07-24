use std::collections::{HashMap, HashSet};
use crate::io::BinaryStream;
use crate::search::{SectionHelper, SearchSection};
use crate::error::{Error, Result};
use crate::il2cpp::structures::*;

fn read_sleb128(stream: &mut BinaryStream) -> Result<i64> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = stream.read_u8()?;
        let low = (byte & 0x7F) as i64;
        result |= low << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 64 && (byte & 0x40) != 0 {
                result |= -(1i64 << shift);
            }
            return Ok(result);
        }
        if shift >= 64 {
            return Err(Error::InvalidFormat("sleb128 overflow".into()));
        }
    }
}

pub const PT_NULL: u32 = 0;
pub const PT_LOAD: u32 = 1;
pub const PT_DYNAMIC: u32 = 2;

pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

pub const DT_NULL: i64 = 0;
pub const DT_PLTGOT: i64 = 3;
pub const DT_HASH: i64 = 4;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_STRSZ: i64 = 10;
pub const DT_RELA: i64 = 7;
pub const DT_RELASZ: i64 = 8;
pub const DT_INIT: i64 = 12;
pub const DT_REL: i64 = 17;
pub const DT_RELSZ: i64 = 18;
pub const DT_GNU_HASH: i64 = 0x6FFFFEF5_u32 as i64;

pub const EM_386: u16 = 3;
pub const EM_ARM: u16 = 40;
pub const EM_X86_64: u16 = 62;
pub const EM_AARCH64: u16 = 183;

pub const R_386_32: u32 = 1;
pub const R_ARM_ABS32: u32 = 2;
pub const R_AARCH64_ABS64: u32 = 257;
pub const R_AARCH64_RELATIVE: u32 = 1027;
pub const R_X86_64_64: u32 = 1;
pub const R_X86_64_RELATIVE: u32 = 8;

pub const SHT_SYMTAB: u32 = 2;
pub const SHT_STRTAB: u32 = 3;
pub const SHT_DYNSYM: u32 = 11;
pub const SHN_UNDEF: u16 = 0;
pub const SHT_LOUSER: u32 = 0x80000000;

#[derive(Debug, Clone, Default)]
pub struct ElfHeader {
    pub e_ident: Vec<u8>,
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

#[derive(Debug, Clone, Default)]
pub struct ElfPhdr {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ElfDyn {
    pub d_tag: i64,
    pub d_un: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ElfSym {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ElfShdr {
    pub sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
    pub sh_link: u32,
    pub sh_info: u32,
    pub sh_addralign: u64,
    pub sh_entsize: u64,
}

pub struct Elf {
    pub stream: BinaryStream,
    pub is_32bit: bool,
    pub is_dumped: bool,
    pub codm_diag: bool,
    pub header: ElfHeader,
    pub segments: Vec<ElfPhdr>,
    dynamic: Vec<ElfDyn>,
    symbols: Vec<ElfSym>,
    sections: Vec<ElfShdr>,
    pub code_registration: Option<Il2CppCodeRegistration>,
    pub metadata_registration: Option<Il2CppMetadataRegistration>,
    pub method_pointers: Vec<u64>,
    pub generic_method_pointers: Vec<u64>,
    pub invoker_pointers: Vec<u64>,
    pub custom_attribute_generators: Vec<u64>,
    pub reverse_pinvoke_wrappers: Vec<u64>,
    pub unresolved_virtual_call_pointers: Vec<u64>,
    pub types: Vec<Il2CppType>,
    pub type_dic: HashMap<u64, usize>,
    pub metadata_usages: Vec<u64>,
    pub field_offsets: Vec<u64>,
    pub field_offsets_are_pointers: bool,
    pub type_definition_sizes: Vec<Il2CppTypeDefinitionSizes>,
    pub generic_inst_pointers: Vec<u64>,
    pub generic_insts: Vec<Il2CppGenericInst>,
    pub generic_method_table: Vec<Il2CppGenericMethodFunctionsDefinitions>,
    pub method_specs: Vec<Il2CppMethodSpec>,
    pub method_definition_method_specs: HashMap<i32, Vec<usize>>,
    pub method_spec_generic_method_pointers: HashMap<usize, u64>,
    pub code_gen_modules: HashMap<String, Il2CppCodeGenModule>,
    pub code_gen_module_method_pointers: HashMap<String, Vec<u64>>,
    pub rgctxs_dictionary: HashMap<String, HashMap<u32, Vec<Il2CppRGCTXDefinition>>>,
    metadata_usages_count: u64,
}

impl Elf {
    pub fn new(data: Vec<u8>, is_32bit: bool) -> Result<Self> {
        let mut elf = Self {
            stream: BinaryStream::new(data),
            is_32bit,
            is_dumped: false,
            codm_diag: false,
            header: ElfHeader::default(),
            segments: Vec::new(),
            dynamic: Vec::new(),
            symbols: Vec::new(),
            sections: Vec::new(),
            code_registration: None,
            metadata_registration: None,
            method_pointers: Vec::new(),
            generic_method_pointers: Vec::new(),
            invoker_pointers: Vec::new(),
            custom_attribute_generators: Vec::new(),
            reverse_pinvoke_wrappers: Vec::new(),
            unresolved_virtual_call_pointers: Vec::new(),
            types: Vec::new(),
            type_dic: HashMap::new(),
            metadata_usages: Vec::new(),
            field_offsets: Vec::new(),
            field_offsets_are_pointers: false,
            type_definition_sizes: Vec::new(),
            generic_inst_pointers: Vec::new(),
            generic_insts: Vec::new(),
            generic_method_table: Vec::new(),
            method_specs: Vec::new(),
            method_definition_method_specs: HashMap::new(),
            method_spec_generic_method_pointers: HashMap::new(),
            code_gen_modules: HashMap::new(),
            code_gen_module_method_pointers: HashMap::new(),
            rgctxs_dictionary: HashMap::new(),
            metadata_usages_count: 0,
        };
        elf.stream.is_32bit = is_32bit;
        let _ = elf.load();
        Ok(elf)
    }

    pub fn new_with_codm_diag(data: Vec<u8>, is_32bit: bool, codm_diag: bool) -> Result<Self> {
        let mut elf = Self::new_unloaded(data, is_32bit)?;
        elf.codm_diag = codm_diag;
        let _ = elf.load();
        Ok(elf)
    }

    fn new_unloaded(data: Vec<u8>, is_32bit: bool) -> Result<Self> {
        let mut elf = Self {
            stream: BinaryStream::new(data),
            is_32bit,
            is_dumped: false,
            codm_diag: false,
            header: ElfHeader::default(),
            segments: Vec::new(),
            dynamic: Vec::new(),
            symbols: Vec::new(),
            sections: Vec::new(),
            code_registration: None,
            metadata_registration: None,
            method_pointers: Vec::new(),
            generic_method_pointers: Vec::new(),
            invoker_pointers: Vec::new(),
            custom_attribute_generators: Vec::new(),
            reverse_pinvoke_wrappers: Vec::new(),
            unresolved_virtual_call_pointers: Vec::new(),
            types: Vec::new(),
            type_dic: HashMap::new(),
            metadata_usages: Vec::new(),
            field_offsets: Vec::new(),
            field_offsets_are_pointers: false,
            type_definition_sizes: Vec::new(),
            generic_inst_pointers: Vec::new(),
            generic_insts: Vec::new(),
            generic_method_table: Vec::new(),
            method_specs: Vec::new(),
            method_definition_method_specs: HashMap::new(),
            method_spec_generic_method_pointers: HashMap::new(),
            code_gen_modules: HashMap::new(),
            code_gen_module_method_pointers: HashMap::new(),
            rgctxs_dictionary: HashMap::new(),
            metadata_usages_count: 0,
        };
        elf.stream.is_32bit = is_32bit;
        Ok(elf)
    }

    pub fn load(&mut self) -> Result<()> {
        self.read_header()?;
        self.read_program_headers()?;

        if self.is_dumped {
            self.fix_program_segments();
        }

        let pt_dynamic_idx = self.segments.iter().position(|s| s.p_type == PT_DYNAMIC);
        if pt_dynamic_idx.is_none() {
            return Err(Error::InvalidFormat("No PT_DYNAMIC segment found".into()));
        }

        self.read_dynamic(pt_dynamic_idx.unwrap())?;

        if self.is_dumped {
            self.fix_dynamic_section();
        }

        self.read_symbols()?;

        if !self.is_dumped {
            self.process_relocations()?;
        }

        Ok(())
    }

    fn read_header(&mut self) -> Result<()> {
        self.stream.set_position(0);
        let e_ident = self.stream.read_bytes(16)?;
        self.header.e_ident = e_ident;
        self.header.e_type = self.stream.read_u16()?;
        self.header.e_machine = self.stream.read_u16()?;
        self.header.e_version = self.stream.read_u32()?;
        if self.is_32bit {
            self.header.e_entry = self.stream.read_u32()? as u64;
            self.header.e_phoff = self.stream.read_u32()? as u64;
            self.header.e_shoff = self.stream.read_u32()? as u64;
        } else {
            self.header.e_entry = self.stream.read_u64()?;
            self.header.e_phoff = self.stream.read_u64()?;
            self.header.e_shoff = self.stream.read_u64()?;
        }
        self.header.e_flags = self.stream.read_u32()?;
        self.header.e_ehsize = self.stream.read_u16()?;
        self.header.e_phentsize = self.stream.read_u16()?;
        self.header.e_phnum = self.stream.read_u16()?;
        self.header.e_shentsize = self.stream.read_u16()?;
        self.header.e_shnum = self.stream.read_u16()?;
        self.header.e_shstrndx = self.stream.read_u16()?;
        Ok(())
    }

    fn read_program_headers(&mut self) -> Result<()> {
        self.stream.set_position(self.header.e_phoff);
        self.segments.clear();
        for _ in 0..self.header.e_phnum {
            let mut phdr = ElfPhdr::default();
            if self.is_32bit {
                phdr.p_type = self.stream.read_u32()?;
                phdr.p_offset = self.stream.read_u32()? as u64;
                phdr.p_vaddr = self.stream.read_u32()? as u64;
                phdr.p_paddr = self.stream.read_u32()? as u64;
                phdr.p_filesz = self.stream.read_u32()? as u64;
                phdr.p_memsz = self.stream.read_u32()? as u64;
                phdr.p_flags = self.stream.read_u32()?;
                phdr.p_align = self.stream.read_u32()? as u64;
            } else {
                phdr.p_type = self.stream.read_u32()?;
                phdr.p_flags = self.stream.read_u32()?;
                phdr.p_offset = self.stream.read_u64()?;
                phdr.p_vaddr = self.stream.read_u64()?;
                phdr.p_paddr = self.stream.read_u64()?;
                phdr.p_filesz = self.stream.read_u64()?;
                phdr.p_memsz = self.stream.read_u64()?;
                phdr.p_align = self.stream.read_u64()?;
            }
            self.segments.push(phdr);
        }
        Ok(())
    }

    fn read_dynamic(&mut self, seg_idx: usize) -> Result<()> {
        let offset = self.segments[seg_idx].p_offset;
        let filesz = self.segments[seg_idx].p_filesz;
        self.stream.set_position(offset);

        let entry_size = if self.is_32bit { 8u64 } else { 16 };
        let count = filesz / entry_size;

        self.dynamic.clear();
        for _ in 0..count {
            let mut dyn_entry = ElfDyn::default();
            if self.is_32bit {
                dyn_entry.d_tag = self.stream.read_i32()? as i64;
                dyn_entry.d_un = self.stream.read_u32()? as u64;
            } else {
                dyn_entry.d_tag = self.stream.read_i64()?;
                dyn_entry.d_un = self.stream.read_u64()?;
            }
            let is_null = dyn_entry.d_tag == DT_NULL;
            self.dynamic.push(dyn_entry);
            if is_null {
                break;
            }
        }
        Ok(())
    }

    fn read_symbols(&mut self) -> Result<()> {
        let symbol_count = self.get_symbol_count()?;
        let symtab = match self.find_dynamic_entry(DT_SYMTAB) {
            Some(e) => e.d_un,
            None => return Ok(()),
        };

        if symbol_count == 0 {
            return Ok(());
        }

        let dynsym_offset = self.map_vatr(symtab)?;
        self.stream.set_position(dynsym_offset);
        self.symbols.clear();

        for _ in 0..symbol_count {
            let mut sym = ElfSym::default();
            if self.is_32bit {
                sym.st_name = self.stream.read_u32()?;
                sym.st_value = self.stream.read_u32()? as u64;
                sym.st_size = self.stream.read_u32()? as u64;
                sym.st_info = self.stream.read_u8()?;
                sym.st_other = self.stream.read_u8()?;
                sym.st_shndx = self.stream.read_u16()?;
            } else {
                sym.st_name = self.stream.read_u32()?;
                sym.st_info = self.stream.read_u8()?;
                sym.st_other = self.stream.read_u8()?;
                sym.st_shndx = self.stream.read_u16()?;
                sym.st_value = self.stream.read_u64()?;
                sym.st_size = self.stream.read_u64()?;
            }
            self.symbols.push(sym);
        }
        Ok(())
    }

    fn get_symbol_count(&mut self) -> Result<usize> {
        if let Some(hash_entry) = self.find_dynamic_entry(DT_HASH) {
            let addr = self.map_vatr(hash_entry.d_un)?;
            self.stream.set_position(addr);
            let _nbucket = self.stream.read_u32()?;
            let nchain = self.stream.read_u32()?;
            return Ok(nchain as usize);
        }

        if let Some(gnu_hash_entry) = self.find_dynamic_entry(DT_GNU_HASH) {
            let addr = self.map_vatr(gnu_hash_entry.d_un)?;
            self.stream.set_position(addr);
            let nbuckets = self.stream.read_u32()?;
            let symoffset = self.stream.read_u32()?;
            let bloom_size = self.stream.read_u32()?;
            let _bloom_shift = self.stream.read_u32()?;

            let bloom_word_size = if self.is_32bit { 4u64 } else { 8 };
            let buckets_address = addr + 16 + (bloom_word_size * bloom_size as u64);

            let max_buckets = self.stream.remaining_from(buckets_address) / 4;
            if nbuckets as u64 > max_buckets || nbuckets as u64 > crate::io::MAX_BINARY_ARRAY_ELEMS as u64 {
                return Err(Error::InvalidArraySize {
                    count: nbuckets as u64,
                    elem_size: 4,
                    context: format!(
                        "DT_GNU_HASH nbuckets invalid (max from remaining={max_buckets})"
                    ),
                });
            }

            self.stream.set_position(buckets_address);
            let mut buckets = Vec::with_capacity(nbuckets as usize);
            for _ in 0..nbuckets {
                buckets.push(self.stream.read_u32()?);
            }

            let last_symbol = buckets.iter().copied().max().unwrap_or(0);
            if last_symbol < symoffset {
                return Ok(symoffset as usize);
            }

            let chains_base = buckets_address + 4 * nbuckets as u64;
            self.stream.set_position(chains_base + (last_symbol - symoffset) as u64 * 4);
            let mut count = last_symbol;
            loop {
                let chain_entry = self.stream.read_u32()?;
                count += 1;
                if (chain_entry & 1) != 0 {
                    break;
                }
            }
            return Ok(count as usize);
        }

        if let Some(symtab_entry) = self.find_dynamic_entry(DT_SYMTAB) {
            let symtab_addr = symtab_entry.d_un;
            let sym_entry_size = if self.is_32bit { 16u64 } else { 24 };
            if let Some(next_addr) = self.dynamic.iter()
                .filter(|e| e.d_un > symtab_addr && e.d_tag != DT_NULL)
                .map(|e| e.d_un)
                .min()
            {
                let table_size = next_addr.saturating_sub(symtab_addr);
                let count = table_size / sym_entry_size;
                if count > 0 {
                    return Ok(count as usize);
                }
            }
        }

        Ok(0)
    }

    fn find_dynamic_entry(&self, tag: i64) -> Option<&ElfDyn> {
        self.dynamic.iter().find(|e| e.d_tag == tag)
    }

    fn process_relocations(&mut self) -> Result<()> {
        if self.is_32bit {
            self.process_rel_relocations()?;
        } else {
            self.process_rela_relocations()?;
        }
        if self.codm_diag {
            let _ = self.process_android_packed_relocations();
        }
        Ok(())
    }

    fn process_android_packed_relocations(&mut self) -> Result<()> {
        const DT_ANDROID_REL: i64 = 0x6000000F;
        const DT_ANDROID_RELSZ: i64 = 0x60000010;
        const DT_ANDROID_RELA: i64 = 0x60000011;
        const DT_ANDROID_RELASZ: i64 = 0x60000012;

        let (rel_un, rel_sz, has_addend) = if let Some(e) = self.find_dynamic_entry(DT_ANDROID_RELA) {
            let sz = self.find_dynamic_entry(DT_ANDROID_RELASZ).map(|s| s.d_un).unwrap_or(0);
            (e.d_un, sz, true)
        } else if let Some(e) = self.find_dynamic_entry(DT_ANDROID_REL) {
            let sz = self.find_dynamic_entry(DT_ANDROID_RELSZ).map(|s| s.d_un).unwrap_or(0);
            (e.d_un, sz, false)
        } else {
            return Ok(());
        };
        if rel_sz == 0 {
            return Ok(());
        }

        let table_offset = self.map_vatr(rel_un)?;
        let table_end = table_offset + rel_sz;
        if table_end as usize > self.stream.data().len() {
            return Ok(());
        }

        if rel_sz < 4
            || self.stream.data()[table_offset as usize] != b'A'
            || self.stream.data()[table_offset as usize + 1] != b'P'
            || self.stream.data()[table_offset as usize + 2] != b'S'
            || self.stream.data()[table_offset as usize + 3] != b'2'
        {
            return Ok(());
        }

        self.stream.set_position(table_offset + 4);

        let group_count = read_sleb128(&mut self.stream)?;
        if group_count <= 0 {
            return Ok(());
        }
        let mut offset = read_sleb128(&mut self.stream)? as i64;
        let mut addend: i64 = 0;
        let symbols = self.symbols.clone();
        let is_aarch64 = self.header.e_machine == EM_AARCH64;
        let is_x86_64 = self.header.e_machine == EM_X86_64;

        let mut applied = 0usize;
        let mut unrecognized = 0usize;
        let mut map_failed = 0usize;
        let mut total = 0usize;

        const RELOCATION_GROUPED_BY_INFO_FLAG: i64 = 1;
        const RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG: i64 = 2;
        const RELOCATION_GROUPED_BY_ADDEND_FLAG: i64 = 4;
        const RELOCATION_GROUP_HAS_ADDEND_FLAG: i64 = 8;

        for _ in 0..group_count {
            let group_size = read_sleb128(&mut self.stream)?;
            if group_size <= 0 {
                break;
            }
            let group_flags = read_sleb128(&mut self.stream)?;
            let group_offset_delta = if group_flags & RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG != 0 {
                read_sleb128(&mut self.stream)?
            } else {
                0
            };
            let mut group_r_info = if group_flags & RELOCATION_GROUPED_BY_INFO_FLAG != 0 {
                read_sleb128(&mut self.stream)?
            } else {
                0
            };
            if group_flags & RELOCATION_GROUPED_BY_ADDEND_FLAG != 0
                && group_flags & RELOCATION_GROUP_HAS_ADDEND_FLAG != 0
            {
                addend = addend.wrapping_add(read_sleb128(&mut self.stream)?);
            } else if group_flags & RELOCATION_GROUP_HAS_ADDEND_FLAG == 0 {
                addend = 0;
            }

            for _ in 0..group_size {
                offset = offset.wrapping_add(if group_flags & RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG != 0 {
                    group_offset_delta
                } else {
                    read_sleb128(&mut self.stream)?
                });

                let r_info = if group_flags & RELOCATION_GROUPED_BY_INFO_FLAG == 0 {
                    let v = read_sleb128(&mut self.stream)?;
                    group_r_info = v;
                    v
                } else {
                    group_r_info
                };

                if has_addend
                    && group_flags & RELOCATION_GROUPED_BY_ADDEND_FLAG == 0
                    && group_flags & RELOCATION_GROUP_HAS_ADDEND_FLAG != 0
                {
                    addend = addend.wrapping_add(read_sleb128(&mut self.stream)?);
                } else if !has_addend {
                    addend = 0;
                }

                total += 1;
                let rel_type = (r_info & 0xFFFFFFFF) as u32;
                let sym_idx = ((r_info as u64) >> 32) as usize;
                let r_offset = offset as u64;
                let r_addend = addend;

                let value: Option<u64> = if is_aarch64 {
                    if rel_type == R_AARCH64_ABS64 && sym_idx < symbols.len() {
                        Some((symbols[sym_idx].st_value as i64 + r_addend) as u64)
                    } else if rel_type == R_AARCH64_RELATIVE {
                        Some(r_addend as u64)
                    } else {
                        None
                    }
                } else if is_x86_64 {
                    if rel_type == R_X86_64_64 && sym_idx < symbols.len() {
                        Some((symbols[sym_idx].st_value as i64 + r_addend) as u64)
                    } else if rel_type == R_X86_64_RELATIVE {
                        Some(r_addend as u64)
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(val) = value {
                    let saved = self.stream.position();
                    match self.map_vatr(r_offset) {
                        Ok(write_offset) => {
                            self.stream.set_position(write_offset);
                            self.stream.write_u64(val)?;
                            self.stream.set_position(saved);
                            applied += 1;
                        }
                        Err(_) => {
                            map_failed += 1;
                        }
                    }
                } else {
                    unrecognized += 1;
                }
            }
        }

        let _ = (total, applied, unrecognized, map_failed);
        Ok(())
    }

    fn process_rel_relocations(&mut self) -> Result<()> {
        let rel_un = match self.find_dynamic_entry(DT_REL) {
            Some(e) => e.d_un,
            None => return Ok(()),
        };
        let relsz = match self.find_dynamic_entry(DT_RELSZ) {
            Some(e) => e.d_un,
            None => return Ok(()),
        };

        let rel_offset = self.map_vatr(rel_un)?;
        let count = relsz / 8;
        self.stream.set_position(rel_offset);
        let is_x86 = self.header.e_machine == EM_386;

        let symbols = self.symbols.clone();
        for _ in 0..count {
            let r_offset = self.stream.read_u32()?;
            let r_info = self.stream.read_u32()?;
            let rel_type = r_info & 0xFF;
            let sym_idx = (r_info >> 8) as usize;

            if (rel_type == R_386_32 && is_x86) || (rel_type == R_ARM_ABS32 && !is_x86) {
                if sym_idx < symbols.len() {
                    let value = symbols[sym_idx].st_value as u32;
                    let saved = self.stream.position();
                    let write_offset = self.map_vatr(r_offset as u64)?;
                    self.stream.set_position(write_offset);
                    self.stream.write_u32(value)?;
                    self.stream.set_position(saved);
                }
            }
        }
        Ok(())
    }

    fn process_rela_relocations(&mut self) -> Result<()> {
        let rela_un = match self.find_dynamic_entry(DT_RELA) {
            Some(e) => e.d_un,
            None => return Ok(()),
        };
        let relasz = match self.find_dynamic_entry(DT_RELASZ) {
            Some(e) => e.d_un,
            None => return Ok(()),
        };

        let rela_offset = self.map_vatr(rela_un)?;
        let count = relasz / 24;
        self.stream.set_position(rela_offset);

        let is_aarch64 = self.header.e_machine == EM_AARCH64;
        let is_x86_64 = self.header.e_machine == EM_X86_64;
        let symbols = self.symbols.clone();

        let mut applied = 0usize;
        let mut unrecognized = 0usize;
        let mut map_failed = 0usize;

        for _ in 0..count {
            let r_offset = self.stream.read_u64()?;
            let r_info = self.stream.read_u64()?;
            let r_addend = self.stream.read_i64()?;

            let rel_type = (r_info & 0xFFFFFFFF) as u32;
            let sym_idx = (r_info >> 32) as usize;

            let value: Option<u64> = if is_aarch64 {
                if rel_type == R_AARCH64_ABS64 && sym_idx < symbols.len() {
                    Some((symbols[sym_idx].st_value as i64 + r_addend) as u64)
                } else if rel_type == R_AARCH64_RELATIVE {
                    Some(r_addend as u64)
                } else {
                    None
                }
            } else if is_x86_64 {
                if rel_type == R_X86_64_64 && sym_idx < symbols.len() {
                    Some((symbols[sym_idx].st_value as i64 + r_addend) as u64)
                } else if rel_type == R_X86_64_RELATIVE {
                    Some(r_addend as u64)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(val) = value {
                let saved = self.stream.position();
                match self.map_vatr(r_offset) {
                    Ok(write_offset) => {
                        self.stream.set_position(write_offset);
                        self.stream.write_u64(val)?;
                        self.stream.set_position(saved);
                        applied += 1;
                    }
                    Err(_) => {
                        map_failed += 1;
                    }
                }
            } else {
                unrecognized += 1;
            }
        }
        let _ = (applied, unrecognized, map_failed);
        Ok(())
    }

    fn fix_program_segments(&mut self) {
        for phdr in &mut self.segments {
            phdr.p_offset = phdr.p_vaddr;
            phdr.p_vaddr += self.stream.image_base;
            phdr.p_filesz = phdr.p_memsz;
        }
    }

    fn fix_dynamic_section(&mut self) {
        let fix_tags: &[i64] = &[DT_PLTGOT, DT_HASH, DT_STRTAB, DT_SYMTAB, 7, DT_INIT, 13, DT_REL, 23, 25, 26];
        for dyn_entry in &mut self.dynamic {
            if fix_tags.contains(&dyn_entry.d_tag) {
                dyn_entry.d_un += self.stream.image_base;
            } else if self.codm_diag && dyn_entry.d_tag == DT_GNU_HASH {
                dyn_entry.d_un += self.stream.image_base;
            }
        }
    }

    pub fn map_vatr(&self, addr: u64) -> Result<u64> {
        for phdr in &self.segments {
            if addr >= phdr.p_vaddr && addr <= phdr.p_vaddr + phdr.p_memsz {
                return Ok(addr - phdr.p_vaddr + phdr.p_offset);
            }
        }
        Err(Error::AddressNotMapped(addr))
    }

    pub fn map_rtva(&self, offset: u64) -> u64 {
        for phdr in &self.segments {
            if offset >= phdr.p_offset && offset <= phdr.p_offset + phdr.p_filesz {
                return offset - phdr.p_offset + phdr.p_vaddr;
            }
        }
        0
    }

    pub fn map_vatr_array(&mut self, addr: u64, count: u64) -> Result<Vec<u64>> {
        if addr == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let elem = self.stream.pointer_size() as u64;
        let max_by_file = self.stream.len() / elem.max(1);
        if count > max_by_file || count > crate::io::MAX_BINARY_ARRAY_ELEMS as u64 {
            return Err(Error::InvalidArraySize {
                count,
                elem_size: elem as usize,
                context: format!(
                    "map_vatr_array: count looks invalid (addr=0x{addr:x}, max_by_file={max_by_file})"
                ),
            });
        }
        let offset = self.map_vatr(addr)?;
        self.stream.read_ptr_array(offset, count as usize)
    }

    pub fn map_vatr_u32_array(&mut self, addr: u64, count: u64) -> Result<Vec<u32>> {
        if addr == 0 || count == 0 {
            return Ok(Vec::new());
        }
        let max_by_file = self.stream.len() / 4;
        if count > max_by_file || count > crate::io::MAX_BINARY_ARRAY_ELEMS as u64 {
            return Err(Error::InvalidArraySize {
                count,
                elem_size: 4,
                context: format!(
                    "map_vatr_u32_array: count looks invalid (addr=0x{addr:x}, max_by_file={max_by_file})"
                ),
            });
        }
        let offset = self.map_vatr(addr)?;
        self.stream.read_u32_array(offset, count as usize)
    }

    pub fn list_exported_symbols(&mut self) -> Result<Vec<(String, u64)>> {
        let mut exports = Vec::new();
        let mut seen = HashSet::new();

        for sh_type in [SHT_DYNSYM, SHT_SYMTAB] {
            if let Some(idx) = self.sections.iter().position(|s| s.sh_type == sh_type) {
                let sec = self.sections[idx].clone();
                if sec.sh_entsize == 0 || sec.sh_size == 0 { continue; }
                let count = sec.sh_size / sec.sh_entsize;
                let strtab_idx = sec.sh_link as usize;
                if strtab_idx >= self.sections.len() { continue; }
                let strtab_offset = self.sections[strtab_idx].sh_offset;

                self.stream.set_position(sec.sh_offset);
                for _ in 0..count {
                    let sym = if self.is_32bit {
                        let st_name = self.stream.read_u32()?;
                        let st_value = self.stream.read_u32()? as u64;
                        let _st_size = self.stream.read_u32()?;
                        let _st_info = self.stream.read_u8()?;
                        let _st_other = self.stream.read_u8()?;
                        let st_shndx = self.stream.read_u16()?;
                        (st_name, st_value, st_shndx)
                    } else {
                        let st_name = self.stream.read_u32()?;
                        let _st_info = self.stream.read_u8()?;
                        let _st_other = self.stream.read_u8()?;
                        let st_shndx = self.stream.read_u16()?;
                        let st_value = self.stream.read_u64()?;
                        let _st_size = self.stream.read_u64()?;
                        (st_name, st_value, st_shndx)
                    };
                    if sym.1 == 0 || sym.2 == SHN_UNDEF { continue; }
                    let name = match self.stream.read_string_to_null_at(strtab_offset + sym.0 as u64) {
                        Ok(n) => n,
                        Err(_) => continue,
                    };
                    if name.is_empty() || seen.contains(&name) { continue; }
                    seen.insert(name.clone());
                    exports.push((name, sym.1));
                }
            }
        }

        if exports.is_empty() {
            let strtab_un = match self.find_dynamic_entry(DT_STRTAB) {
                Some(e) => e.d_un,
                None => return Ok(exports),
            };
            let dynstr_offset = match self.map_vatr(strtab_un) {
                Ok(o) => o,
                Err(_) => return Ok(exports),
            };
            let syms = self.symbols.clone();
            for sym in &syms {
                if sym.st_value == 0 || sym.st_shndx == SHN_UNDEF { continue; }
                let name = match self.stream.read_string_to_null_at(dynstr_offset + sym.st_name as u64) {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                if name.is_empty() || seen.contains(&name) { continue; }
                seen.insert(name.clone());
                exports.push((name, sym.st_value));
            }
        }

        Ok(exports)
    }

    pub fn symbol_search(&mut self) -> Result<Option<(u64, u64)>> {
        let strtab_un = match self.find_dynamic_entry(DT_STRTAB) {
            Some(e) => e.d_un,
            None => return Ok(None),
        };

        let dynstr_offset = self.map_vatr(strtab_un)?;
        let mut code_reg = 0u64;
        let mut metadata_reg = 0u64;

        for sym in &self.symbols {
            let name = self.stream.read_string_to_null_at(dynstr_offset + sym.st_name as u64)?;
            if name == "g_CodeRegistration" {
                code_reg = sym.st_value;
            } else if name == "g_MetadataRegistration" {
                metadata_reg = sym.st_value;
            }
        }

        if code_reg > 0 && metadata_reg > 0 {
            Ok(Some((code_reg, metadata_reg)))
        } else {
            Ok(None)
        }
    }

    pub fn get_section_helper(&self, method_count: usize, type_definitions_count: usize, image_count: usize) -> SectionHelper<'_> {
        let mut data_list = Vec::new();
        let mut exec_list = Vec::new();
        let mut all_sections = Vec::new();

        for phdr in &self.segments {
            if phdr.p_memsz != 0 {
                let section = SearchSection::new(
                    phdr.p_offset,
                    phdr.p_offset + phdr.p_filesz,
                    phdr.p_vaddr,
                    phdr.p_vaddr + phdr.p_memsz,
                );

                all_sections.push(section.clone());

                match phdr.p_flags {
                    1 | 3 | 5 | 7 => exec_list.push(section),
                    2 | 4 | 6 => data_list.push(section),
                    _ => {}
                }
            }
        }

        let bss = data_list.clone();

        SectionHelper::new(
            self.stream.data(),
            self.is_32bit,
            self.stream.version,
            true,
            all_sections,
            data_list,
            exec_list,
            bss,
            method_count,
            type_definitions_count,
            self.metadata_usages_count as usize,
            image_count,
        )
    }

    pub fn check_dump(&mut self) -> bool {
        if self.header.e_shnum == 0 || self.header.e_shoff == 0 {
            return true;
        }

        self.stream.set_position(self.header.e_shoff);
        self.sections.clear();

        for _ in 0..self.header.e_shnum {
            let mut shdr = ElfShdr::default();
            if self.is_32bit {
                shdr.sh_name = self.stream.read_u32().unwrap_or(0);
                shdr.sh_type = self.stream.read_u32().unwrap_or(0);
                shdr.sh_flags = self.stream.read_u32().unwrap_or(0) as u64;
                shdr.sh_addr = self.stream.read_u32().unwrap_or(0) as u64;
                shdr.sh_offset = self.stream.read_u32().unwrap_or(0) as u64;
                shdr.sh_size = self.stream.read_u32().unwrap_or(0) as u64;
                shdr.sh_link = self.stream.read_u32().unwrap_or(0);
                shdr.sh_info = self.stream.read_u32().unwrap_or(0);
                shdr.sh_addralign = self.stream.read_u32().unwrap_or(0) as u64;
                shdr.sh_entsize = self.stream.read_u32().unwrap_or(0) as u64;
            } else {
                shdr.sh_name = self.stream.read_u32().unwrap_or(0);
                shdr.sh_type = self.stream.read_u32().unwrap_or(0);
                shdr.sh_flags = self.stream.read_u64().unwrap_or(0);
                shdr.sh_addr = self.stream.read_u64().unwrap_or(0);
                shdr.sh_offset = self.stream.read_u64().unwrap_or(0);
                shdr.sh_size = self.stream.read_u64().unwrap_or(0);
                shdr.sh_link = self.stream.read_u32().unwrap_or(0);
                shdr.sh_info = self.stream.read_u32().unwrap_or(0);
                shdr.sh_addralign = self.stream.read_u64().unwrap_or(0);
                shdr.sh_entsize = self.stream.read_u64().unwrap_or(0);
            }
            self.sections.push(shdr);
        }

        if (self.header.e_shstrndx as usize) < self.sections.len() {
            let shstrndx = self.sections[self.header.e_shstrndx as usize].sh_offset;
            for section in &self.sections {
                if let Ok(name) = self.stream.read_string_to_null_at(shstrndx + section.sh_name as u64) {
                    if name == ".text" {
                        return false;
                    }
                }
            }
        }

        true
    }

    pub fn check_protection(&mut self) -> bool {
        if self.find_dynamic_entry(DT_INIT).is_some() {
            println!("WARNING: find .init_proc");
            return true;
        }

        if let Some(strtab) = self.find_dynamic_entry(DT_STRTAB) {
            if let Ok(dynstr_offset) = self.map_vatr(strtab.d_un) {
                for sym in &self.symbols {
                    if let Ok(name) = self.stream.read_string_to_null_at(dynstr_offset + sym.st_name as u64) {
                        if name == "JNI_OnLoad" {
                            println!("WARNING: find JNI_OnLoad");
                            return true;
                        }
                    }
                }
            }
        }

        if self.sections.iter().any(|s| s.sh_type == SHT_LOUSER) {
            println!("WARNING: find SHT_LOUSER section");
            return true;
        }

        false
    }

    pub fn get_rva(&self, pointer: u64) -> u64 {
        if self.is_dumped {
            pointer - self.stream.image_base
        } else {
            pointer
        }
    }

    pub fn set_properties(&mut self, version: f64, metadata_usages_count: u64) {
        self.stream.version = version;
        self.metadata_usages_count = metadata_usages_count;
    }

    pub fn init(&mut self, code_registration_addr: u64, metadata_registration_addr: u64) -> Result<()> {
        let version = self.stream.version;

        let cr_offset = self.map_vatr(code_registration_addr)?;
        self.stream.set_position(cr_offset);
        let cr = Il2CppCodeRegistration::read(&mut self.stream, version)?;

        let mr_offset = self.map_vatr(metadata_registration_addr)?;
        self.stream.set_position(mr_offset);
        let mr = Il2CppMetadataRegistration::read(&mut self.stream, version)?;

        self.load_pointers(&cr, &mr)?;
        self.load_types(&mr)?;
        self.load_generics(&mr)?;

        if version >= 24.2 {
            self.load_code_gen_modules(&cr)?;
        }

        self.code_registration = Some(cr);
        self.metadata_registration = Some(mr);

        Ok(())
    }

    fn load_pointers(&mut self, cr: &Il2CppCodeRegistration, mr: &Il2CppMetadataRegistration) -> Result<()> {
        let version = self.stream.version;

        if self.codm_diag {
            if version <= 24.1 && cr.method_pointers > 0 && cr.method_pointers_count > 0 {
                match self.map_vatr_array(cr.method_pointers, cr.method_pointers_count) {
                    Ok(v) => self.method_pointers = v,
                    Err(e) => eprintln!("[WARN] Failed to load method_pointers: {e}"),
                }
            }

            if cr.generic_method_pointers_count > 0 {
                match self.map_vatr_array(cr.generic_method_pointers, cr.generic_method_pointers_count) {
                    Ok(v) => self.generic_method_pointers = v,
                    Err(e) => eprintln!("[WARN] Failed to load generic_method_pointers: {e}"),
                }
            }

            if cr.invoker_pointers_count > 0 {
                match self.map_vatr_array(cr.invoker_pointers, cr.invoker_pointers_count) {
                    Ok(v) => self.invoker_pointers = v,
                    Err(e) => eprintln!("[WARN] Failed to load invoker_pointers: {e}"),
                }
            }

            if version < 27.0 && cr.custom_attribute_count > 0 {
                match self.map_vatr_array(cr.custom_attribute_generators, cr.custom_attribute_count) {
                    Ok(v) => self.custom_attribute_generators = v,
                    Err(e) => eprintln!("[WARN] Failed to load custom_attribute_generators: {e}"),
                }
            }

            if version > 16.0 && version < 27.0 && self.metadata_usages_count > 0 {
                match self.map_vatr_array(mr.metadata_usages, self.metadata_usages_count) {
                    Ok(v) => self.metadata_usages = v,
                    Err(e) => eprintln!("[WARN] Failed to load metadata_usages: {e}"),
                }
            }

            if version >= 22.0 && cr.reverse_pinvoke_wrapper_count > 0 {
                match self.map_vatr_array(cr.reverse_pinvoke_wrappers, cr.reverse_pinvoke_wrapper_count) {
                    Ok(v) => self.reverse_pinvoke_wrappers = v,
                    Err(e) => eprintln!("[WARN] Failed to load reverse_pinvoke_wrappers: {e}"),
                }
            }

            if version >= 22.0 && cr.unresolved_virtual_call_count > 0 {
                match self.map_vatr_array(cr.unresolved_virtual_call_pointers, cr.unresolved_virtual_call_count) {
                    Ok(v) => self.unresolved_virtual_call_pointers = v,
                    Err(e) => eprintln!("[WARN] Failed to load unresolved_virtual_call_pointers: {e}"),
                }
            }
        } else {
            if version <= 24.1 && cr.method_pointers > 0 && cr.method_pointers_count > 0 {
                self.method_pointers = self.map_vatr_array(cr.method_pointers, cr.method_pointers_count)?;
            }

            if cr.generic_method_pointers_count > 0 {
                self.generic_method_pointers = self.map_vatr_array(cr.generic_method_pointers, cr.generic_method_pointers_count)?;
            }

            if cr.invoker_pointers_count > 0 {
                self.invoker_pointers = self.map_vatr_array(cr.invoker_pointers, cr.invoker_pointers_count)?;
            }

            if version < 27.0 && cr.custom_attribute_count > 0 {
                self.custom_attribute_generators = self.map_vatr_array(cr.custom_attribute_generators, cr.custom_attribute_count)?;
            }

            if version > 16.0 && version < 27.0 && self.metadata_usages_count > 0 {
                self.metadata_usages = self.map_vatr_array(mr.metadata_usages, self.metadata_usages_count)?;
            }

            if version >= 22.0 && cr.reverse_pinvoke_wrapper_count > 0 {
                self.reverse_pinvoke_wrappers = self.map_vatr_array(cr.reverse_pinvoke_wrappers, cr.reverse_pinvoke_wrapper_count)?;
            }

            if version >= 22.0 && cr.unresolved_virtual_call_count > 0 {
                self.unresolved_virtual_call_pointers = self.map_vatr_array(cr.unresolved_virtual_call_pointers, cr.unresolved_virtual_call_count)?;
            }
        }

        Ok(())
    }

    fn load_types(&mut self, mr: &Il2CppMetadataRegistration) -> Result<()> {
        let version = self.stream.version;
        let type_pointers = self.map_vatr_array(mr.types, mr.types_count)?;

        self.types.clear();
        self.type_dic.clear();

        let mut decoded = 0usize;
        let mut skipped = 0usize;
        for (idx, ptr) in type_pointers.iter().enumerate() {
            if self.codm_diag && self.is_dumped {
                let offset = match self.map_vatr(*ptr) {
                    Ok(o) => o,
                    Err(_) => {
                        self.types.push(Il2CppType::default());
                        skipped += 1;
                        continue;
                    }
                };
                self.stream.set_position(offset);
                let mut il2cpp_type = match Il2CppType::read(&mut self.stream) {
                    Ok(t) => t,
                    Err(_) => {
                        self.types.push(Il2CppType::default());
                        skipped += 1;
                        continue;
                    }
                };
                let pre = il2cpp_type.type_enum;
                il2cpp_type.init_codm(version);
                if pre != il2cpp_type.type_enum {
                    decoded += 1;
                }
                self.types.push(il2cpp_type);
                self.type_dic.insert(*ptr, idx);
            } else if self.codm_diag {
                let offset = match self.map_vatr(*ptr) {
                    Ok(o) => o,
                    Err(_) => {
                        self.types.push(Il2CppType::default());
                        skipped += 1;
                        continue;
                    }
                };
                self.stream.set_position(offset);
                let mut il2cpp_type = match Il2CppType::read(&mut self.stream) {
                    Ok(t) => t,
                    Err(_) => {
                        self.types.push(Il2CppType::default());
                        skipped += 1;
                        continue;
                    }
                };
                let pre = il2cpp_type.type_enum;
                il2cpp_type.init_codm(version);
                if pre != il2cpp_type.type_enum {
                    decoded += 1;
                }
                self.types.push(il2cpp_type);
                self.type_dic.insert(*ptr, idx);
            } else {
                let offset = self.map_vatr(*ptr)?;
                self.stream.set_position(offset);
                let mut il2cpp_type = Il2CppType::read(&mut self.stream)?;
                il2cpp_type.init(version);
                self.types.push(il2cpp_type);
                self.type_dic.insert(*ptr, idx);
            }
        }
        if self.codm_diag && skipped > 0 {
            eprintln!("[CODM] init_codm decoded {} of {} Il2CppType entries (skipped {})",
                decoded, type_pointers.len(), skipped);
        }

        self.field_offsets_are_pointers = version > 21.0;
        if version == 21.0 && mr.field_offsets_count >= 6 {
            if let Ok(test) = self.map_vatr_array(mr.field_offsets, std::cmp::min(6, mr.field_offsets_count)) {
                if test.len() >= 6 {
                    self.field_offsets_are_pointers = test[0] == 0 && test[1] == 0 && test[2] == 0 &&
                        test[3] == 0 && test[4] == 0 && test[5] > 0;
                }
            }
        }

        if self.codm_diag {
            match self.map_vatr_array(mr.field_offsets, mr.field_offsets_count) {
                Ok(v) => self.field_offsets = v,
                Err(e) => eprintln!("[WARN] Failed to load field_offsets: {e}"),
            }
        } else {
            self.field_offsets = self.map_vatr_array(mr.field_offsets, mr.field_offsets_count)?;
        }

        if mr.type_definitions_sizes > 0 && mr.type_definitions_sizes_count > 0 {
            if self.codm_diag {
                if let Ok(sizes_offset) = self.map_vatr(mr.type_definitions_sizes) {
                    if let Ok(n) = crate::io::checked_array_len(
                        mr.type_definitions_sizes_count,
                        std::mem::size_of::<Il2CppTypeDefinitionSizes>().max(1),
                        Some(self.stream.remaining_from(sizes_offset)),
                        "type_definition_sizes(codm)",
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
            } else {
                let sizes_offset = self.map_vatr(mr.type_definitions_sizes)?;
                let n = crate::io::checked_array_len(
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

        Ok(())
    }

    fn load_generics(&mut self, mr: &Il2CppMetadataRegistration) -> Result<()> {
        let version = self.stream.version;

        if self.codm_diag {
            match self.map_vatr_array(mr.generic_insts, mr.generic_insts_count) {
                Ok(v) => self.generic_inst_pointers = v,
                Err(e) => {
                    eprintln!("[WARN] Failed to load generic_inst_pointers: {e}");
                    return Ok(());
                }
            }

            self.generic_insts.clear();
            for ptr in &self.generic_inst_pointers {
                match self.map_vatr(*ptr) {
                    Ok(offset) => {
                        self.stream.set_position(offset);
                        match Il2CppGenericInst::read(&mut self.stream) {
                            Ok(gi) => self.generic_insts.push(gi),
                            Err(_) => self.generic_insts.push(Il2CppGenericInst::default()),
                        }
                    }
                    Err(_) => self.generic_insts.push(Il2CppGenericInst::default()),
                }
            }

            if mr.generic_method_table_count > 0 {
                if let Ok(offset) = self.map_vatr(mr.generic_method_table) {
                    self.stream.set_position(offset);
                    self.generic_method_table.clear();
                    for _ in 0..mr.generic_method_table_count {
                        match Il2CppGenericMethodFunctionsDefinitions::read(&mut self.stream, version) {
                            Ok(t) => self.generic_method_table.push(t),
                            Err(_) => break,
                        }
                    }
                }
            }

            if mr.method_specs_count > 0 {
                if let Ok(offset) = self.map_vatr(mr.method_specs) {
                    self.stream.set_position(offset);
                    self.method_specs.clear();
                    for _ in 0..mr.method_specs_count {
                        match Il2CppMethodSpec::read(&mut self.stream) {
                            Ok(ms) => self.method_specs.push(ms),
                            Err(_) => break,
                        }
                    }
                }
            }
        } else {
            self.generic_inst_pointers = self.map_vatr_array(mr.generic_insts, mr.generic_insts_count)?;

            self.generic_insts.clear();
            for ptr in &self.generic_inst_pointers {
                let offset = self.map_vatr(*ptr)?;
                self.stream.set_position(offset);
                self.generic_insts.push(Il2CppGenericInst::read(&mut self.stream)?);
            }

            if mr.generic_method_table_count > 0 {
                let offset = self.map_vatr(mr.generic_method_table)?;
                self.stream.set_position(offset);
                self.generic_method_table.clear();
                for _ in 0..mr.generic_method_table_count {
                    self.generic_method_table.push(Il2CppGenericMethodFunctionsDefinitions::read(&mut self.stream, version)?);
                }
            }

            if mr.method_specs_count > 0 {
                let offset = self.map_vatr(mr.method_specs)?;
                self.stream.set_position(offset);
                self.method_specs.clear();
                for _ in 0..mr.method_specs_count {
                    self.method_specs.push(Il2CppMethodSpec::read(&mut self.stream)?);
                }
            }
        }

        self.method_definition_method_specs.clear();
        self.method_spec_generic_method_pointers.clear();

        for (_table_idx, table) in self.generic_method_table.iter().enumerate() {
            if (table.generic_method_index as usize) < self.method_specs.len() {
                let ms = &self.method_specs[table.generic_method_index as usize];
                let method_def_idx = ms.method_definition_index;

                self.method_definition_method_specs
                    .entry(method_def_idx)
                    .or_default()
                    .push(table.generic_method_index as usize);

                if table.indices.method_index >= 0 &&
                    (table.indices.method_index as usize) < self.generic_method_pointers.len() {
                    self.method_spec_generic_method_pointers.insert(
                        table.generic_method_index as usize,
                        self.generic_method_pointers[table.indices.method_index as usize],
                    );
                }
            }
        }

        Ok(())
    }

    fn load_code_gen_modules(&mut self, cr: &Il2CppCodeRegistration) -> Result<()> {
        let version = self.stream.version;
        let module_pointers = self.map_vatr_array(cr.code_gen_modules, cr.code_gen_modules_count)?;

        for ptr in module_pointers {
            let offset = self.map_vatr(ptr)?;
            self.stream.set_position(offset);
            let module = Il2CppCodeGenModule::read(&mut self.stream, version)?;
            let name_offset = self.map_vatr(module.module_name)?;
            let module_name = self.stream.read_string_to_null_at(name_offset)?;

            let method_ptrs = if module.method_pointer_count > 0 && module.method_pointers > 0 {
                match self.map_vatr_array(module.method_pointers, module.method_pointer_count as u64) {
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

            let mut rgctx_def_dic: HashMap<u32, Vec<Il2CppRGCTXDefinition>> = HashMap::new();

            if module.rgctxs_count > 0 {
                let rgctxs_offset = self.map_vatr(module.rgctxs)?;
                self.stream.set_position(rgctxs_offset);
                let mut rgctxs = Vec::new();
                for _ in 0..module.rgctxs_count {
                    rgctxs.push(Il2CppRGCTXDefinition::read(&mut self.stream, version)?);
                }

                let ranges_offset = self.map_vatr(module.rgctx_ranges)?;
                self.stream.set_position(ranges_offset);
                let mut rgctx_ranges = Vec::new();
                for _ in 0..module.rgctx_ranges_count {
                    rgctx_ranges.push(Il2CppTokenRangePair::read(&mut self.stream)?);
                }

                for range_pair in &rgctx_ranges {
                    let start = range_pair.range.start as usize;
                    let length = range_pair.range.length as usize;
                    if start + length <= rgctxs.len() {
                        rgctx_def_dic.insert(range_pair.token, rgctxs[start..start + length].to_vec());
                    }
                }
            }

            self.rgctxs_dictionary.insert(module_name.clone(), rgctx_def_dic);
            self.code_gen_modules.insert(module_name, module);
        }

        Ok(())
    }

    pub fn get_field_offset_from_index(&mut self, type_index: usize, field_index_in_type: usize, field_index: usize, is_value_type: bool, is_static: bool) -> i32 {
        let result = if self.field_offsets_are_pointers {
            if type_index >= self.field_offsets.len() {
                return -1;
            }
            let ptr = self.field_offsets[type_index];
            if ptr == 0 {
                return -1;
            }
            match self.map_vatr(ptr) {
                Ok(base) => {
                    self.stream.set_position(base + (field_index_in_type as u64) * 4);
                    self.stream.read_i32().unwrap_or(-1)
                }
                Err(_) => return -1,
            }
        } else {
            if field_index >= self.field_offsets.len() {
                return -1;
            }
            self.field_offsets[field_index] as i32
        };

        if result > 0 && is_value_type && !is_static {
            let adjust = if self.is_32bit { 8 } else { 16 };
            result - adjust
        } else {
            result
        }
    }

    pub fn get_method_pointer(&self, image_name: &str, method_token: u32, method_index: i32) -> u64 {
        let version = self.stream.version;
        if version >= 24.2 {
            if let Some(ptrs) = self.code_gen_module_method_pointers.get(image_name) {
                let idx = (method_token & 0x00FFFFFF) as usize;
                if idx > 0 && idx <= ptrs.len() {
                    return ptrs[idx - 1];
                }
            }
        } else {
            let idx = method_index as usize;
            if method_index >= 0 && idx < self.method_pointers.len() {
                return self.method_pointers[idx];
            }
        }
        0
    }

    pub fn auto_plus_init(&mut self, code_reg: Option<u64>, metadata_reg: Option<u64>) -> Result<bool> {
        let code_registration = code_reg.unwrap_or(0);
        let metadata_registration = metadata_reg.unwrap_or(0);
        if code_registration == 0 || metadata_registration == 0 {
            return Ok(false);
        }
        match self.init_with_auto_plus(code_registration, metadata_registration) {
            Ok(()) => Ok(true),
            Err(e) => {
                eprintln!(
                    "[WARN] Registration init failed at CR=0x{code_registration:x} MR=0x{metadata_registration:x}: {e}"
                );
                Ok(false)
            }
        }
    }

    pub fn init_with_auto_plus(
        &mut self,
        code_registration: u64,
        metadata_registration: u64,
    ) -> Result<()> {
        let orig_version = self.stream.version;
        let limit = crate::il2cpp::auto_plus_count_limit(false);
        let cr = if code_registration != 0 && self.stream.version >= 24.2 {
            match self.map_vatr(code_registration) {
                Ok(cr_offset) => {
                    self.stream.set_position(cr_offset);
                    match Il2CppCodeRegistration::read(&mut self.stream, orig_version) {
                        Ok(cr_struct) => crate::il2cpp::apply_auto_plus_heuristics(
                            &mut self.stream,
                            code_registration,
                            &cr_struct,
                            limit,
                        ),
                        Err(_) => code_registration,
                    }
                }
                Err(_) => code_registration,
            }
        } else {
            code_registration
        };

        match self.init(cr, metadata_registration) {
            Ok(()) => Ok(()),
            Err(e) if orig_version == 31.0 && self.stream.version != 29.0 => {
                self.stream.version = 29.0;
                eprintln!("Info: Retry init with il2cpp version 29 ({e})");
                self.init(cr, metadata_registration)
            }
            Err(e) => Err(e),
        }
    }

    /// ARM32 pattern search from __mod_init_func / executable segments.
    /// Matches the C# Elf.Search() method for 32-bit ARM ELFs.
    pub fn search_arm32(&mut self, version: f64) -> Option<(u64, u64)> {
        if self.header.e_machine != EM_ARM || !self.is_32bit {
            return None;
        }

        let got = self.find_dynamic_entry(DT_PLTGOT)?.d_un;

        let exec_segments: Vec<(u64, u64)> = self.segments.iter()
            .filter(|p| p.p_type == PT_LOAD && (p.p_flags & PF_X) != 0)
            .map(|p| (p.p_offset, p.p_filesz))
            .collect();

        // ARM feature bytes pattern:
        // ? 0x10 ? 0xE7  (LDR R1, [X])
        // ? 0x00 ? 0xE0  (ADD R0, X, X)
        // ? 0x20 ? 0xE0  (ADD R2, X, X)
        let pattern: [(usize, u8); 6] = [
            (1, 0x10), (3, 0xE7),
            (5, 0x00), (7, 0xE0),
            (9, 0x20), (11, 0xE0),
        ];

        let mut results: Vec<u64> = Vec::new();

        for (seg_offset, seg_size) in &exec_segments {
            self.stream.set_position(*seg_offset);
            let buff = match self.stream.read_bytes(*seg_size as usize) {
                Ok(b) => b,
                Err(_) => continue,
            };

            for i in 0..buff.len().saturating_sub(12) {
                let mut matched = true;
                for &(off, val) in &pattern {
                    if i + off >= buff.len() || buff[i + off] != val {
                        matched = false;
                        break;
                    }
                }
                if matched {
                    // Check LDR bit (byte[2] bit 4 must be 1)
                    let hex_char = buff[i + 2];
                    let bit3 = (hex_char >> 4) & 1;
                    if bit3 == 1 {
                        results.push(i as u64);
                    }
                }
            }
        }

        if results.len() != 1 {
            return None;
        }

        let result = results[0] as u32;
        let image_base = self.stream.image_base as u32;

        if version < 24.0 {
            self.stream.set_position(result as u64 + 0x14);
            let code_registration = self.stream.read_u32().ok()? as u64 + got;
            self.stream.set_position(result as u64 + 0x18);
            let ptr = self.stream.read_u32().ok()? as u64 + got;
            let ptr_offset = self.map_vatr(ptr).ok()?;
            self.stream.set_position(ptr_offset);
            let metadata_registration = self.stream.read_u32().ok()? as u64;
            Some((code_registration, metadata_registration))
        } else {
            // version >= 24
            self.stream.set_position(result as u64 + 0x14);
            let code_registration = self.stream.read_u32().ok()? as u64
                + result as u64 + 0xC + image_base as u64;
            self.stream.set_position(result as u64 + 0x10);
            let ptr = self.stream.read_u32().ok()? as u64 + result as u64 + 0x8;
            let ptr_offset = self.map_vatr(ptr + image_base as u64).ok()?;
            self.stream.set_position(ptr_offset);
            let metadata_registration = self.stream.read_u32().ok()? as u64;
            Some((code_registration, metadata_registration))
        }
    }
}
