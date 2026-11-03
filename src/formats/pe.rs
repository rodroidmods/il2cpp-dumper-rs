use crate::error::{Error, Result};
use crate::io::BinaryStream;
use crate::search::{SearchSection, SectionHelper};
use std::collections::HashMap;

pub const IMAGE_DOS_SIGNATURE: u16 = 0x5A4D;
pub const IMAGE_NT_SIGNATURE: u32 = 0x00004550;

pub const IMAGE_SCN_CNT_CODE: u32 = 0x00000020;
pub const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x00000040;
pub const IMAGE_SCN_MEM_EXECUTE: u32 = 0x20000000;

pub const IMAGE_DIRECTORY_ENTRY_EXPORT: usize = 0;
pub const IMAGE_DIRECTORY_ENTRY_IMPORT: usize = 1;
pub const IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT: usize = 13;

#[derive(Debug, Clone, Default)]
pub struct PeSectionHeader {
    pub name: String,
    pub virtual_size: u32,
    pub virtual_address: u32,
    pub size_of_raw_data: u32,
    pub pointer_to_raw_data: u32,
    pub pointer_to_relocations: u32,
    pub pointer_to_linenumbers: u32,
    pub number_of_relocations: u16,
    pub number_of_linenumbers: u16,
    pub characteristics: u32,
}

#[derive(Debug, Clone, Default)]
pub struct DataDirectory {
    pub virtual_address: u32,
    pub size: u32,
}

pub struct Pe {
    pub stream: BinaryStream,
    pub is_32bit: bool,
    image_base: u64,
    pub sections: Vec<PeSectionHeader>,
    data_directories: Vec<DataDirectory>,
    /// IAT slot VA → `dll!symbol`. The PE analogue of Mach-O bind symbols /
    /// ELF JUMP_SLOT relocs: names every import stub so codegen helpers
    /// calling through the IAT get real names.
    pub bind_symbols: HashMap<u64, String>,
}

impl Pe {
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let mut pe = Self {
            stream: BinaryStream::new(data),
            is_32bit: true,
            image_base: 0,
            sections: Vec::new(),
            data_directories: Vec::new(),
            bind_symbols: HashMap::new(),
        };
        pe.load()?;
        Ok(pe)
    }

    fn load(&mut self) -> Result<()> {
        self.stream.set_position(0);
        let e_magic = self.stream.read_u16()?;
        if e_magic != IMAGE_DOS_SIGNATURE {
            return Err(Error::InvalidFormat("Invalid DOS signature".into()));
        }

        self.stream.set_position(0x3C);
        let e_lfanew = self.stream.read_u32()?;

        self.stream.set_position(e_lfanew as u64);
        let nt_sig = self.stream.read_u32()?;
        if nt_sig != IMAGE_NT_SIGNATURE {
            return Err(Error::InvalidFormat("Invalid NT signature".into()));
        }

        let _machine = self.stream.read_u16()?;
        let number_of_sections = self.stream.read_u16()?;
        let _time_date_stamp = self.stream.read_u32()?;
        let _pointer_to_symbol_table = self.stream.read_u32()?;
        let _number_of_symbols = self.stream.read_u32()?;
        let size_of_optional_header = self.stream.read_u16()?;
        let _characteristics = self.stream.read_u16()?;

        let optional_header_start = self.stream.position();
        let optional_magic = self.stream.read_u16()?;

        if optional_magic == 0x20B {
            self.is_32bit = false;
            self.stream.is_32bit = false;
            self.stream.set_position(optional_header_start + 24);
            self.image_base = self.stream.read_u64()?;

            self.stream.set_position(optional_header_start + 108);
            let num_rva_and_sizes = self.stream.read_u32()?;

            self.data_directories.clear();
            for _ in 0..std::cmp::min(num_rva_and_sizes, 16) {
                self.data_directories.push(DataDirectory {
                    virtual_address: self.stream.read_u32()?,
                    size: self.stream.read_u32()?,
                });
            }
        } else {
            self.is_32bit = true;
            self.stream.is_32bit = true;
            self.stream.set_position(optional_header_start + 28);
            self.image_base = self.stream.read_u32()? as u64;

            self.stream.set_position(optional_header_start + 92);
            let num_rva_and_sizes = self.stream.read_u32()?;

            self.data_directories.clear();
            for _ in 0..std::cmp::min(num_rva_and_sizes, 16) {
                self.data_directories.push(DataDirectory {
                    virtual_address: self.stream.read_u32()?,
                    size: self.stream.read_u32()?,
                });
            }
        }

        self.stream.image_base = self.image_base;

        self.stream
            .set_position(optional_header_start + size_of_optional_header as u64);
        self.sections.clear();
        for _ in 0..number_of_sections {
            let name_bytes = self.stream.read_bytes(8)?;
            let name = String::from_utf8_lossy(
                &name_bytes
                    .iter()
                    .copied()
                    .take_while(|&b| b != 0)
                    .collect::<Vec<u8>>(),
            )
            .to_string();

            self.sections.push(PeSectionHeader {
                name,
                virtual_size: self.stream.read_u32()?,
                virtual_address: self.stream.read_u32()?,
                size_of_raw_data: self.stream.read_u32()?,
                pointer_to_raw_data: self.stream.read_u32()?,
                pointer_to_relocations: self.stream.read_u32()?,
                pointer_to_linenumbers: self.stream.read_u32()?,
                number_of_relocations: self.stream.read_u16()?,
                number_of_linenumbers: self.stream.read_u16()?,
                characteristics: self.stream.read_u32()?,
            });
        }

        Ok(())
    }

    pub fn map_vatr(&self, mut addr: u64) -> Result<u64> {
        if addr >= self.image_base {
            addr -= self.image_base;
        }
        for section in &self.sections {
            let start = section.virtual_address as u64;
            let end = start + section.virtual_size as u64;
            if addr >= start && addr < end {
                return Ok(
                    addr - section.virtual_address as u64 + section.pointer_to_raw_data as u64
                );
            }
        }
        Err(Error::AddressNotMapped(addr))
    }

    pub fn map_rtva(&self, addr: u64) -> u64 {
        for section in &self.sections {
            let start = section.pointer_to_raw_data as u64;
            let end = start + section.size_of_raw_data as u64;
            if addr >= start && addr < end {
                return addr - section.pointer_to_raw_data as u64
                    + section.virtual_address as u64
                    + self.image_base;
            }
        }
        0
    }

    pub fn list_exported_symbols(&mut self) -> Result<Vec<(String, u64)>> {
        let mut exports = Vec::new();
        if self.data_directories.len() <= IMAGE_DIRECTORY_ENTRY_EXPORT {
            return Ok(exports);
        }
        let export_dir = &self.data_directories[IMAGE_DIRECTORY_ENTRY_EXPORT];
        if export_dir.virtual_address == 0 {
            return Ok(exports);
        }
        let export_offset = self.map_vatr(export_dir.virtual_address as u64)?;
        self.stream.set_position(export_offset);
        let _characteristics = self.stream.read_u32()?;
        let _time_date_stamp = self.stream.read_u32()?;
        let _major_version = self.stream.read_u16()?;
        let _minor_version = self.stream.read_u16()?;
        let _name_rva = self.stream.read_u32()?;
        let _base = self.stream.read_u32()?;
        let _number_of_functions = self.stream.read_u32()?;
        let number_of_names = self.stream.read_u32()?;
        let address_of_functions = self.stream.read_u32()?;
        let address_of_names = self.stream.read_u32()?;
        let address_of_name_ordinals = self.stream.read_u32()?;
        let names_offset = self.map_vatr(address_of_names as u64)?;
        let ordinals_offset = self.map_vatr(address_of_name_ordinals as u64)?;
        let functions_offset = self.map_vatr(address_of_functions as u64)?;
        for i in 0..number_of_names {
            self.stream.set_position(names_offset + i as u64 * 4);
            let name_rva = self.stream.read_u32()?;
            let name_offset = match self.map_vatr(name_rva as u64) {
                Ok(o) => o,
                Err(_) => continue,
            };
            let name = match self.stream.read_string_to_null_at(name_offset) {
                Ok(n) => n,
                Err(_) => continue,
            };
            self.stream.set_position(ordinals_offset + i as u64 * 2);
            let ordinal = self.stream.read_u16()?;
            self.stream
                .set_position(functions_offset + ordinal as u64 * 4);
            let func_rva = self.stream.read_u32()?;
            exports.push((name, func_rva as u64));
        }
        Ok(exports)
    }

    /// Walk the Import Directory Table (and delay-load imports) and record
    /// every First-Thunk / bound IAT slot as `dll!symbol` in `bind_symbols`.
    /// This is the PE counterpart of Mach-O's dyld bind opcodes and ELF's
    /// JUMP_SLOT relocations: it names every import stub authoritatively so
    /// the recovery pass can classify helpers that call through the IAT.
    pub fn collect_bind_symbols(&mut self) -> Result<()> {
        self.collect_import_directory(IMAGE_DIRECTORY_ENTRY_IMPORT)?;
        // Delay-load imports are common in Unity player builds.
        let _ = self.collect_delay_import_directory();
        Ok(())
    }

    fn collect_import_directory(&mut self, dir_index: usize) -> Result<()> {
        if self.data_directories.len() <= dir_index {
            return Ok(());
        }
        let dir = self.data_directories[dir_index].clone();
        if dir.virtual_address == 0 || dir.size == 0 {
            return Ok(());
        }

        let mut desc_rva = dir.virtual_address as u64;
        // IDT is an array of IMAGE_IMPORT_DESCRIPTOR (20 bytes) ending in all-zero.
        for _ in 0..(dir.size as usize / 20 + 1) {
            let desc_off = match self.map_vatr(desc_rva) {
                Ok(o) => o,
                Err(_) => break,
            };
            self.stream.set_position(desc_off);
            let original_first_thunk = self.stream.read_u32()?;
            let _time_stamp = self.stream.read_u32()?;
            let _forwarder_chain = self.stream.read_u32()?;
            let name_rva = self.stream.read_u32()?;
            let first_thunk = self.stream.read_u32()?;
            if name_rva == 0 || (original_first_thunk == 0 && first_thunk == 0) {
                break; // terminator
            }

            let dll_name = self
                .stream
                .read_string_to_null_at(self.map_vatr(name_rva as u64).unwrap_or(0))
                .unwrap_or_default();
            let dll_name = dll_name.trim().to_string();

            // Walk both the OriginalFirstThunk (INT, names never overwritten)
            // and FirstThunk (IAT, the slots code actually calls through).
            // Bound/loaded dumps have IAT entries patched to addresses; the
            // INT still holds the RVAs, so pair them index-by-index.
            let int_rva = if original_first_thunk != 0 {
                original_first_thunk
            } else {
                first_thunk
            };
            let mut idx: u64 = 0;
            loop {
                let int_entry = self.read_thunk_entry(int_rva as u64 + idx * 4)?;
                let iat_va = self.image_base + first_thunk as u64 + idx * 4;
                if int_entry == 0 {
                    break; // NULL terminator for this DLL's thunk array
                }
                // Ordinal flag differs between PE32 (high bit) and PE32+ (high bit).
                let ordinal_flag: u64 = if self.is_32bit {
                    0x8000_0000
                } else {
                    0x8000_0000_0000_0000
                };
                let name = if int_entry & ordinal_flag != 0 {
                    let ordinal = int_entry & 0xFFFF;
                    format!("{dll_name}!ordinal_{ordinal}")
                } else {
                    let hint_rva = int_entry & 0x7FFF_FFFF;
                    // IMAGE_IMPORT_BY_NAME: 2-byte hint then the symbol name.
                    match self
                        .map_vatr(hint_rva)
                        .ok()
                        .and_then(|off| self.stream.read_string_to_null_at(off + 2).ok())
                    {
                        Some(n) if !n.is_empty() => format!("{dll_name}!{n}"),
                        _ => format!("{dll_name}!hint_{hint_rva:#x}"),
                    }
                };
                if let Ok(slot_off) = self.map_vatr(iat_va - self.image_base) {
                    let _ = slot_off;
                    self.bind_symbols.insert(iat_va, name);
                }
                idx += 1;
                if idx > 100_000 {
                    break; // safety: malformed table
                }
            }
            desc_rva += 20;
        }
        Ok(())
    }

    fn read_thunk_entry(&mut self, rva: u64) -> Result<u64> {
        let off = self.map_vatr(rva)?;
        self.stream.set_position(off);
        if self.is_32bit {
            Ok(self.stream.read_u32()? as u64)
        } else {
            self.stream.read_u64()
        }
    }

    fn collect_delay_import_directory(&mut self) -> Result<()> {
        if self.data_directories.len() <= IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT {
            return Ok(());
        }
        let dir = self.data_directories[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT].clone();
        if dir.virtual_address == 0 || dir.size == 0 {
            return Ok(());
        }
        // ImgDelayDescr: 64-bit RVAs (even in PE32 for newer MSVC links).
        let mut desc_rva = dir.virtual_address as u64;
        for _ in 0..(dir.size as usize / 32 + 1) {
            let desc_off = match self.map_vatr(desc_rva) {
                Ok(o) => o,
                Err(_) => break,
            };
            self.stream.set_position(desc_off);
            let _gr_attr = self.stream.read_u32()?;
            let _cb_all = self.stream.read_u32()?;
            let _sz_hint = self.stream.read_u32()?;
            let name_rva = self.stream.read_u32()?;
            let _phmod = self.stream.read_u32()?;
            let int_rva = self.stream.read_u32()?;
            let iat_rva = self.stream.read_u32()?;
            if name_rva == 0 || int_rva == 0 {
                break;
            }
            let dll_name = self
                .stream
                .read_string_to_null_at(self.map_vatr(name_rva as u64).unwrap_or(0))
                .unwrap_or_default();
            let dll_name = dll_name.trim().to_string();
            let mut idx: u64 = 0;
            loop {
                let int_entry = self.read_thunk_entry(int_rva as u64 + idx * 4)?;
                if int_entry == 0 {
                    break;
                }
                let iat_va = self.image_base + iat_rva as u64 + idx * 4;
                let ordinal_flag: u64 = if self.is_32bit {
                    0x8000_0000
                } else {
                    0x8000_0000_0000_0000
                };
                let name = if int_entry & ordinal_flag != 0 {
                    format!("{dll_name}!ordinal_{}", int_entry & 0xFFFF)
                } else {
                    let hint_rva = int_entry & 0x7FFF_FFFF;
                    match self
                        .map_vatr(hint_rva)
                        .ok()
                        .and_then(|off| self.stream.read_string_to_null_at(off + 2).ok())
                    {
                        Some(n) if !n.is_empty() => format!("{dll_name}!{n}"),
                        _ => format!("{dll_name}!hint_{hint_rva:#x}"),
                    }
                };
                self.bind_symbols.insert(iat_va, name);
                idx += 1;
                if idx > 100_000 {
                    break;
                }
            }
            desc_rva += 32;
        }
        Ok(())
    }

    pub fn symbol_search(&mut self) -> Result<Option<(u64, u64)>> {
        if self.data_directories.len() <= IMAGE_DIRECTORY_ENTRY_EXPORT {
            return Ok(None);
        }

        let export_dir = &self.data_directories[IMAGE_DIRECTORY_ENTRY_EXPORT];
        if export_dir.virtual_address == 0 {
            return Ok(None);
        }

        let export_offset = self.map_vatr(export_dir.virtual_address as u64)?;
        self.stream.set_position(export_offset);

        let _characteristics = self.stream.read_u32()?;
        let _time_date_stamp = self.stream.read_u32()?;
        let _major_version = self.stream.read_u16()?;
        let _minor_version = self.stream.read_u16()?;
        let _name_rva = self.stream.read_u32()?;
        let _base = self.stream.read_u32()?;
        let _number_of_functions = self.stream.read_u32()?;
        let number_of_names = self.stream.read_u32()?;
        let address_of_functions = self.stream.read_u32()?;
        let address_of_names = self.stream.read_u32()?;
        let address_of_name_ordinals = self.stream.read_u32()?;

        let names_offset = self.map_vatr(address_of_names as u64)?;
        let ordinals_offset = self.map_vatr(address_of_name_ordinals as u64)?;
        let functions_offset = self.map_vatr(address_of_functions as u64)?;

        let mut code_reg = 0u64;
        let mut metadata_reg = 0u64;

        for i in 0..number_of_names {
            self.stream.set_position(names_offset + i as u64 * 4);
            let name_rva = self.stream.read_u32()?;
            let name_offset = self.map_vatr(name_rva as u64)?;
            let name = self.stream.read_string_to_null_at(name_offset)?;

            self.stream.set_position(ordinals_offset + i as u64 * 2);
            let ordinal = self.stream.read_u16()?;

            self.stream
                .set_position(functions_offset + ordinal as u64 * 4);
            let func_rva = self.stream.read_u32()?;

            if name == "g_CodeRegistration" {
                code_reg = func_rva as u64 + self.image_base;
            } else if name == "g_MetadataRegistration" {
                metadata_reg = func_rva as u64 + self.image_base;
            }
        }

        if code_reg > 0 && metadata_reg > 0 {
            Ok(Some((code_reg, metadata_reg)))
        } else {
            Ok(None)
        }
    }

    /// Raw bytes + RVA of a named section (e.g. ".pdata", ".xdata").
    /// Used by exception-table recovery.
    pub fn section_bytes_by_name(&self, name: &str) -> Option<(u64, Vec<u8>)> {
        for sect in &self.sections {
            if sect.name == name {
                let start = sect.pointer_to_raw_data as usize;
                let end = (start + sect.size_of_raw_data as usize).min(self.stream.len() as usize);
                if start >= end {
                    return Some((sect.virtual_address as u64, Vec::new()));
                }
                return Some((
                    sect.virtual_address as u64,
                    self.stream.data()[start..end].to_vec(),
                ));
            }
        }
        None
    }

    /// Writable/readable data sections used for v27+ metadata-usage scans (FieldRva, etc.).
    pub fn data_search_sections(&self) -> Vec<SearchSection> {
        let mut data_list = Vec::new();
        for section in &self.sections {
            if section.virtual_size == 0 {
                continue;
            }
            let search_section = SearchSection::new(
                section.pointer_to_raw_data as u64,
                (section.pointer_to_raw_data + section.size_of_raw_data) as u64,
                section.virtual_address as u64 + self.image_base,
                (section.virtual_address + section.virtual_size) as u64 + self.image_base,
            );
            let chars = section.characteristics;
            let is_code = (chars & IMAGE_SCN_CNT_CODE) != 0 || (chars & IMAGE_SCN_MEM_EXECUTE) != 0;
            let is_data = (chars & IMAGE_SCN_CNT_INITIALIZED_DATA) != 0;
            if is_data && !is_code {
                data_list.push(search_section);
            }
        }
        data_list
    }

    pub fn get_section_helper(
        &self,
        method_count: usize,
        type_definitions_count: usize,
        metadata_usages_count: usize,
        image_count: usize,
        version: f64,
    ) -> SectionHelper<'_> {
        let data_list = self.data_search_sections();
        let mut exec_list = Vec::new();
        let mut all_sections = Vec::new();

        for section in &self.sections {
            if section.virtual_size == 0 {
                continue;
            }

            let search_section = SearchSection::new(
                section.pointer_to_raw_data as u64,
                (section.pointer_to_raw_data + section.size_of_raw_data) as u64,
                section.virtual_address as u64 + self.image_base,
                (section.virtual_address + section.virtual_size) as u64 + self.image_base,
            );

            all_sections.push(search_section.clone());

            let chars = section.characteristics;
            let is_code = (chars & IMAGE_SCN_CNT_CODE) != 0 || (chars & IMAGE_SCN_MEM_EXECUTE) != 0;
            let is_data = (chars & IMAGE_SCN_CNT_INITIALIZED_DATA) != 0;

            if is_code && !is_data {
                exec_list.push(search_section);
            } else if is_data && !is_code {
            } else {
            }
        }

        let bss = data_list.clone();

        SectionHelper::new(
            self.stream.data(),
            self.is_32bit,
            version,
            false,
            all_sections,
            data_list,
            exec_list,
            bss,
            method_count,
            type_definitions_count,
            metadata_usages_count,
            image_count,
        )
    }

    pub fn check_dump(&self) -> bool {
        if self.is_32bit {
            self.image_base != 0x10000000
        } else {
            self.image_base != 0x180000000
        }
    }

    pub fn load_from_memory(&mut self, addr: u64) {
        self.image_base = addr;
        self.stream.image_base = addr;
        for section in &mut self.sections {
            section.pointer_to_raw_data = section.virtual_address;
            section.size_of_raw_data = section.virtual_size;
        }
    }

    pub fn get_rva(&self, pointer: u64) -> u64 {
        pointer - self.image_base
    }

    pub fn image_base(&self) -> u64 {
        self.image_base
    }
}
