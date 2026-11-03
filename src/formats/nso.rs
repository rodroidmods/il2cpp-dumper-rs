use crate::error::{Error, Result};
use crate::formats::elf::{
    DT_GNU_HASH, DT_HASH, DT_RELA, DT_RELASZ, DT_STRTAB, DT_SYMTAB, ElfDyn, ElfSym,
    R_AARCH64_ABS64, R_AARCH64_RELATIVE, SHN_UNDEF,
};
use crate::io::BinaryStream;
use crate::search::{SearchSection, SectionHelper};

pub const NSO_MAGIC: u32 = 0x304F534E;

#[derive(Clone)]
struct NsoSegment {
    file_offset: u64,
    memory_offset: u64,
    decompressed_size: u64,
}

pub struct Nso {
    pub stream: BinaryStream,
    pub is_32bit: bool,
    segments: Vec<NsoSegment>,
    bss_segment: Option<NsoSegment>,
    dynamic: Vec<ElfDyn>,
    symbols: Vec<ElfSym>,
}

impl Nso {
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let header = Self::parse_header(&data)?;
        let is_compressed = (header.flags & 7) != 0;
        let decompressed = Self::decompress_if_needed(&data, &header)?;

        let text_seg = NsoSegment {
            file_offset: header.text_file_offset as u64,
            memory_offset: header.text_memory_offset as u64,
            decompressed_size: header.text_decompressed_size as u64,
        };
        let rodata_seg = NsoSegment {
            file_offset: header.rodata_file_offset as u64,
            memory_offset: header.rodata_memory_offset as u64,
            decompressed_size: header.rodata_decompressed_size as u64,
        };
        let data_seg = NsoSegment {
            file_offset: header.data_file_offset as u64,
            memory_offset: header.data_memory_offset as u64,
            decompressed_size: header.data_decompressed_size as u64,
        };

        let segments = vec![text_seg.clone(), rodata_seg.clone(), data_seg.clone()];

        let mut nso = Self {
            stream: BinaryStream::new(decompressed),
            is_32bit: false,
            segments,
            bss_segment: None,
            dynamic: Vec::new(),
            symbols: Vec::new(),
        };

        if !is_compressed {
            nso.read_mod0_and_dynamic(&header)?;
        }

        Ok(nso)
    }

    fn read_mod0_and_dynamic(&mut self, header: &NsoHeader) -> Result<()> {
        self.stream.set_position(header.text_file_offset as u64 + 4);
        let mod_offset = self.stream.read_u32()? as u64;

        self.stream
            .set_position(header.text_file_offset as u64 + mod_offset + 4);
        let dynamic_offset = self.stream.read_u32()? as u64 + mod_offset;
        let bss_start = self.stream.read_u32()? as u64;
        let bss_end = self.stream.read_u32()? as u64;

        self.bss_segment = Some(NsoSegment {
            file_offset: bss_start,
            memory_offset: bss_start,
            decompressed_size: bss_end - bss_start,
        });

        let data_end = header.data_memory_offset as u64 + header.data_decompressed_size as u64;
        let max_entries = ((data_end - dynamic_offset) / 16) as usize;

        let dyn_file_offset = self.map_vatr(dynamic_offset)?;
        self.stream.set_position(dyn_file_offset);

        let mut dynamic_section: Vec<ElfDyn> = Vec::new();
        for _ in 0..max_entries {
            let d_tag = self.stream.read_i64()?;
            let d_un = self.stream.read_u64()?;
            if d_tag == 0 {
                break;
            }
            dynamic_section.push(ElfDyn { d_tag, d_un });
        }

        let symbol_table = self.read_symbols(&dynamic_section)?;
        self.process_relocations(&dynamic_section, &symbol_table)?;
        self.dynamic = dynamic_section;
        self.symbols = symbol_table;

        Ok(())
    }

    fn read_symbols(&mut self, dynamic_section: &[ElfDyn]) -> Result<Vec<ElfSym>> {
        let symbol_count = self.get_symbol_count(dynamic_section)?;
        let dynsym_entry = dynamic_section.iter().find(|d| d.d_tag == DT_SYMTAB);
        let dynsym_addr = match dynsym_entry {
            Some(e) => e.d_un,
            None => return Ok(Vec::new()),
        };

        let dynsym_offset = self.map_vatr(dynsym_addr)?;
        self.stream.set_position(dynsym_offset);

        let mut symbols = Vec::with_capacity(symbol_count);
        for _ in 0..symbol_count {
            let st_name = self.stream.read_u32()?;
            let st_info = self.stream.read_u8()?;
            let st_other = self.stream.read_u8()?;
            let st_shndx = self.stream.read_u16()?;
            let st_value = self.stream.read_u64()?;
            let st_size = self.stream.read_u64()?;
            symbols.push(ElfSym {
                st_name,
                st_info,
                st_other,
                st_shndx,
                st_value,
                st_size,
            });
        }

        Ok(symbols)
    }

    fn get_symbol_count(&mut self, dynamic_section: &[ElfDyn]) -> Result<usize> {
        if let Some(hash_entry) = dynamic_section.iter().find(|d| d.d_tag == DT_HASH) {
            let addr = self.map_vatr(hash_entry.d_un)?;
            self.stream.set_position(addr);
            let _nbucket = self.stream.read_u32()?;
            let nchain = self.stream.read_u32()?;
            return Ok(nchain as usize);
        }

        if let Some(gnu_hash) = dynamic_section.iter().find(|d| d.d_tag == DT_GNU_HASH) {
            let addr = self.map_vatr(gnu_hash.d_un)?;
            self.stream.set_position(addr);
            let nbuckets = self.stream.read_u32()?;
            let symoffset = self.stream.read_u32()?;
            let bloom_size = self.stream.read_u32()?;
            let _bloom_shift = self.stream.read_u32()?;
            let buckets_address = addr + 16 + (8 * bloom_size as u64);

            self.stream.set_position(buckets_address);
            let mut last_symbol = 0u32;
            for _ in 0..nbuckets {
                let bucket = self.stream.read_u32()?;
                if bucket > last_symbol {
                    last_symbol = bucket;
                }
            }

            if last_symbol < symoffset {
                return Ok(symoffset as usize);
            }

            let chains_base = buckets_address + 4 * nbuckets as u64;
            self.stream
                .set_position(chains_base + (last_symbol - symoffset) as u64 * 4);
            loop {
                let chain_entry = self.stream.read_u32()?;
                last_symbol += 1;
                if (chain_entry & 1) != 0 {
                    break;
                }
            }
            return Ok(last_symbol as usize);
        }

        Ok(0)
    }

    fn process_relocations(
        &mut self,
        dynamic_section: &[ElfDyn],
        symbol_table: &[ElfSym],
    ) -> Result<()> {
        let rela_entry = match dynamic_section.iter().find(|d| d.d_tag == DT_RELA) {
            Some(e) => e,
            None => return Ok(()),
        };
        let relasz_entry = match dynamic_section.iter().find(|d| d.d_tag == DT_RELASZ) {
            Some(e) => e,
            None => return Ok(()),
        };

        let rela_offset = self.map_vatr(rela_entry.d_un)?;
        let rela_count = relasz_entry.d_un / 24;

        println!("Applying relocations...");

        self.stream.set_position(rela_offset);
        let mut rela_entries = Vec::with_capacity(rela_count as usize);
        for _ in 0..rela_count {
            let r_offset = self.stream.read_u64()?;
            let r_info = self.stream.read_u64()?;
            let r_addend = self.stream.read_i64()?;
            rela_entries.push((r_offset, r_info, r_addend));
        }

        for (r_offset, r_info, r_addend) in &rela_entries {
            let rel_type = (*r_info & 0xFFFFFFFF) as u32;
            let sym_idx = (*r_info >> 32) as usize;

            match rel_type {
                R_AARCH64_ABS64 => {
                    if sym_idx < symbol_table.len() {
                        let symbol = &symbol_table[sym_idx];
                        let write_offset = self.map_vatr(*r_offset)?;
                        let value = symbol.st_value.wrapping_add(*r_addend as u64);
                        self.stream.set_position(write_offset);
                        self.stream.write_u64(value)?;
                    }
                }
                R_AARCH64_RELATIVE => {
                    let write_offset = self.map_vatr(*r_offset)?;
                    self.stream.set_position(write_offset);
                    self.stream.write_u64(*r_addend as u64)?;
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn parse_header(data: &[u8]) -> Result<NsoHeader> {
        if data.len() < 0x100 {
            return Err(Error::InvalidFormat("NSO header too small".into()));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != NSO_MAGIC {
            return Err(Error::InvalidFormat("Invalid NSO magic".into()));
        }

        let r = |off: usize| {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };

        Ok(NsoHeader {
            flags: r(12),
            text_file_offset: r(16),
            text_memory_offset: r(20),
            text_decompressed_size: r(24),
            rodata_file_offset: r(32),
            rodata_memory_offset: r(36),
            rodata_decompressed_size: r(40),
            data_file_offset: r(48),
            data_memory_offset: r(52),
            data_decompressed_size: r(56),
            bss_size: r(60),
            text_compressed_size: r(0x60),
            rodata_compressed_size: r(0x64),
            data_compressed_size: r(0x68),
        })
    }

    fn decompress_if_needed(data: &[u8], header: &NsoHeader) -> Result<Vec<u8>> {
        let text_compressed = (header.flags & 1) != 0;
        let rodata_compressed = (header.flags & 2) != 0;
        let data_compressed = (header.flags & 4) != 0;

        if !text_compressed && !rodata_compressed && !data_compressed {
            return Ok(data.to_vec());
        }

        let total_size = header.data_memory_offset as usize
            + header.data_decompressed_size as usize
            + header.bss_size as usize;

        let mut result = vec![0u8; total_size.max(0x100)];
        result[..0x100.min(data.len())].copy_from_slice(&data[..0x100.min(data.len())]);

        let new_ro_offset = header.text_file_offset + header.text_decompressed_size;
        let new_data_offset = new_ro_offset + header.rodata_decompressed_size;

        result[16..20].copy_from_slice(&header.text_file_offset.to_le_bytes());
        result[32..36].copy_from_slice(&new_ro_offset.to_le_bytes());
        result[48..52].copy_from_slice(&new_data_offset.to_le_bytes());
        result[12..16].copy_from_slice(&0u32.to_le_bytes());

        Self::decompress_segment(
            data,
            &mut result,
            header.text_file_offset as usize,
            header.text_file_offset as usize,
            header.text_decompressed_size as usize,
            header.text_compressed_size as usize,
            text_compressed,
        )?;

        Self::decompress_segment(
            data,
            &mut result,
            header.rodata_file_offset as usize,
            new_ro_offset as usize,
            header.rodata_decompressed_size as usize,
            header.rodata_compressed_size as usize,
            rodata_compressed,
        )?;

        Self::decompress_segment(
            data,
            &mut result,
            header.data_file_offset as usize,
            new_data_offset as usize,
            header.data_decompressed_size as usize,
            header.data_compressed_size as usize,
            data_compressed,
        )?;

        Ok(result)
    }

    fn decompress_segment(
        src: &[u8],
        dst: &mut [u8],
        file_offset: usize,
        dest_offset: usize,
        decompressed_size: usize,
        compressed_size: usize,
        is_compressed: bool,
    ) -> Result<()> {
        if is_compressed {
            let end = file_offset + compressed_size;
            if end > src.len() {
                return Err(Error::InvalidFormat(
                    "Compressed segment out of bounds".into(),
                ));
            }
            let compressed = &src[file_offset..end];
            let decompressed = lz4_flex::decompress(compressed, decompressed_size)
                .map_err(|e| Error::InvalidFormat(format!("LZ4 decompression failed: {}", e)))?;
            let copy_len = std::cmp::min(decompressed.len(), dst.len().saturating_sub(dest_offset));
            dst[dest_offset..dest_offset + copy_len].copy_from_slice(&decompressed[..copy_len]);
        } else {
            let end = file_offset + decompressed_size;
            if end > src.len() {
                return Err(Error::InvalidFormat(
                    "Uncompressed segment out of bounds".into(),
                ));
            }
            let copy_len = std::cmp::min(decompressed_size, dst.len().saturating_sub(dest_offset));
            dst[dest_offset..dest_offset + copy_len]
                .copy_from_slice(&src[file_offset..file_offset + copy_len]);
        }
        Ok(())
    }

    pub fn map_vatr(&self, addr: u64) -> Result<u64> {
        for seg in &self.segments {
            if addr >= seg.memory_offset && addr <= seg.memory_offset + seg.decompressed_size {
                return Ok(addr - seg.memory_offset + seg.file_offset);
            }
        }
        Err(Error::AddressNotMapped(addr))
    }

    pub fn map_rtva(&self, offset: u64) -> u64 {
        for seg in &self.segments {
            if offset >= seg.file_offset && offset <= seg.file_offset + seg.decompressed_size {
                return offset - seg.file_offset + seg.memory_offset;
            }
        }
        0
    }

    pub fn data_search_sections(&self) -> Vec<SearchSection> {
        let mut data_list = Vec::new();
        if self.segments.len() >= 3 {
            let rodata = &self.segments[1];
            data_list.push(SearchSection::new(
                rodata.file_offset,
                rodata.file_offset + rodata.decompressed_size,
                rodata.memory_offset,
                rodata.memory_offset + rodata.decompressed_size,
            ));
            let data_seg = &self.segments[2];
            data_list.push(SearchSection::new(
                data_seg.file_offset,
                data_seg.file_offset + data_seg.decompressed_size,
                data_seg.memory_offset,
                data_seg.memory_offset + data_seg.decompressed_size,
            ));
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
        let mut exec_list = Vec::new();
        let data_list = self.data_search_sections();
        let mut bss_list = Vec::new();
        let mut all = Vec::new();

        if self.segments.len() >= 3 {
            let text = &self.segments[0];
            let s = SearchSection::new(
                text.file_offset,
                text.file_offset + text.decompressed_size,
                text.memory_offset,
                text.memory_offset + text.decompressed_size,
            );
            all.push(s.clone());
            exec_list.push(s);

            let rodata = &self.segments[1];
            let s = SearchSection::new(
                rodata.file_offset,
                rodata.file_offset + rodata.decompressed_size,
                rodata.memory_offset,
                rodata.memory_offset + rodata.decompressed_size,
            );
            all.push(s.clone());

            let data_seg = &self.segments[2];
            let s = SearchSection::new(
                data_seg.file_offset,
                data_seg.file_offset + data_seg.decompressed_size,
                data_seg.memory_offset,
                data_seg.memory_offset + data_seg.decompressed_size,
            );
            all.push(s.clone());
        }

        if let Some(bss) = &self.bss_segment {
            let s = SearchSection::new(
                bss.file_offset,
                bss.file_offset + bss.decompressed_size,
                bss.memory_offset,
                bss.memory_offset + bss.decompressed_size,
            );
            all.push(s.clone());
            bss_list.push(s);
        }

        let bss = if bss_list.is_empty() {
            data_list.clone()
        } else {
            bss_list
        };

        SectionHelper::new(
            self.stream.data(),
            self.is_32bit,
            version,
            true,
            all,
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
        false
    }

    pub fn list_exported_symbols(&mut self) -> Result<Vec<(String, u64)>> {
        let mut exports = Vec::new();
        let strtab_entry = self.dynamic.iter().find(|d| d.d_tag == DT_STRTAB);
        let strtab_addr = match strtab_entry {
            Some(e) => e.d_un,
            None => return Ok(exports),
        };
        let strtab_offset = match self.map_vatr(strtab_addr) {
            Ok(o) => o,
            Err(_) => return Ok(exports),
        };
        for sym in self.symbols.clone() {
            if sym.st_value == 0 || sym.st_shndx == SHN_UNDEF {
                continue;
            }
            let name = match self
                .stream
                .read_string_to_null_at(strtab_offset + sym.st_name as u64)
            {
                Ok(n) => n,
                Err(_) => continue,
            };
            if name.is_empty() {
                continue;
            }
            exports.push((name, sym.st_value));
        }
        Ok(exports)
    }

    pub fn get_rva(&self, pointer: u64) -> u64 {
        pointer
    }
}

struct NsoHeader {
    flags: u32,
    text_file_offset: u32,
    text_memory_offset: u32,
    text_decompressed_size: u32,
    rodata_file_offset: u32,
    rodata_memory_offset: u32,
    rodata_decompressed_size: u32,
    data_file_offset: u32,
    data_memory_offset: u32,
    data_decompressed_size: u32,
    bss_size: u32,
    text_compressed_size: u32,
    rodata_compressed_size: u32,
    data_compressed_size: u32,
}
