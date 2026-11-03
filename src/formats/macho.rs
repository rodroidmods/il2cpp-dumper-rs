use crate::error::{Error, Result};
use crate::io::BinaryStream;
use crate::search::{SearchSection, SectionHelper};
use std::collections::HashMap;

pub const MH_MAGIC: u32 = 0xFEEDFACE;
pub const MH_MAGIC_64: u32 = 0xFEEDFACF;
pub const FAT_MAGIC: u32 = 0xCAFEBABE;
pub const FAT_CIGAM: u32 = 0xBEBAFECA;

pub const LC_SEGMENT: u32 = 0x01;
pub const LC_SYMTAB: u32 = 0x02;
pub const LC_SEGMENT_64: u32 = 0x19;
pub const LC_ENCRYPTION_INFO: u32 = 0x21;
pub const LC_ENCRYPTION_INFO_64: u32 = 0x2C;
pub const LC_DYLD_INFO: u32 = 0x22;
pub const LC_DYLD_INFO_ONLY: u32 = 0x80000022;
pub const LC_DYLD_CHAINED_FIXUPS: u32 = 0x80000034;

pub const S_ATTR_PURE_INSTRUCTIONS: u32 = 0x80000000;
pub const S_ATTR_SOME_INSTRUCTIONS: u32 = 0x00000400;

#[derive(Debug, Clone)]
pub struct FatArch {
    pub cputype: u32,
    pub cpusubtype: u32,
    pub offset: u32,
    pub size: u32,
    pub align: u32,
    pub magic: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Segment {
    pub segname: String,
    pub vmaddr: u64,
    pub vmsize: u64,
    pub fileoff: u64,
    pub filesize: u64,
    pub nsects: u32,
}

#[derive(Debug, Clone, Default)]
struct Section {
    pub sectname: String,
    pub addr: u64,
    pub size: u64,
    pub offset: u32,
    pub flags: u32,
}

#[derive(Debug, Clone, Default)]
struct SymtabCmd {
    pub symoff: u32,
    pub nsyms: u32,
    pub stroff: u32,
    pub strsize: u32,
}

#[derive(Debug, Clone, Default)]
struct NlistEntry {
    pub n_strx: u32,
    pub n_type: u8,
    pub n_sect: u8,
    pub n_desc: i16,
    pub n_value: u64,
}

pub fn parse_fat(data: &[u8]) -> Result<Vec<FatArch>> {
    if data.len() < 8 {
        return Err(Error::InvalidFormat("FAT header too small".into()));
    }
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != FAT_MAGIC && magic != FAT_CIGAM {
        return Err(Error::InvalidFormat("Not a FAT Mach-O".into()));
    }

    let nfat = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    let mut arches = Vec::new();
    let mut offset = 8usize;

    for _ in 0..nfat {
        if offset + 20 > data.len() {
            break;
        }
        let cputype = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let cpusubtype = u32::from_be_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let arch_offset = u32::from_be_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let size = u32::from_be_bytes([
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
            data[offset + 15],
        ]);
        let align = u32::from_be_bytes([
            data[offset + 16],
            data[offset + 17],
            data[offset + 18],
            data[offset + 19],
        ]);

        let slice_magic = if (arch_offset as usize) + 4 <= data.len() {
            u32::from_le_bytes([
                data[arch_offset as usize],
                data[arch_offset as usize + 1],
                data[arch_offset as usize + 2],
                data[arch_offset as usize + 3],
            ])
        } else {
            0
        };

        arches.push(FatArch {
            cputype,
            cpusubtype,
            offset: arch_offset,
            size,
            align,
            magic: slice_magic,
        });
        offset += 20;
    }

    Ok(arches)
}

pub fn extract_fat_slice(data: &[u8], arch: &FatArch) -> Result<Vec<u8>> {
    let start = arch.offset as usize;
    let end = start + arch.size as usize;
    if end > data.len() {
        return Err(Error::InvalidFormat("Fat slice extends beyond file".into()));
    }
    Ok(data[start..end].to_vec())
}

#[derive(Debug, Clone, Default)]
struct DyldInfoCmd {
    rebase_off: u32,
    rebase_size: u32,
    /// File offsets/sizes of the (non-lazy) bind and lazy-bind opcode tables.
    /// Bind opcodes map GOT/lazy-stub slot VAs → imported symbol names; that
    /// is how codegen helpers calling through `__got` stubs get real names.
    bind_off: u32,
    bind_size: u32,
    lazy_bind_off: u32,
    lazy_bind_size: u32,
    weak_bind_off: u32,
    weak_bind_size: u32,
}

/// Which LC_DYLD_INFO bind opcode stream to walk.
///
/// All three encode the same opcode language into different VA ranges;
/// `Weak` was declared but never parsed, which left every weak-bound slot
/// (weak symbols imported by the binary) unnamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindStream {
    /// Non-lazy binds: the `__got` slots.
    Regular,
    /// Lazy binds: the `__la_symbol_ptr` stub slots, DONE-separated.
    Lazy,
    /// Weak binds: same opcodes, weak-symbol slots.
    Weak,
}

#[derive(Debug, Clone, Default)]
struct ChainedFixupsCmd {
    dataoff: u32,
    datasize: u32,
}

pub struct MachO {
    pub stream: BinaryStream,
    pub is_32bit: bool,
    pub codm_apply_fixups: bool,
    pub is_arm64e: bool,
    pub segments: Vec<Segment>,
    sections: Vec<Section>,
    symbols: Vec<NlistEntry>,
    string_table: Vec<u8>,
    vmaddr: u64,
    dyld_info: Option<DyldInfoCmd>,
    chained_fixups: Option<ChainedFixupsCmd>,
    /// GOT/lazy-stub slot VA → imported symbol name, from LC_DYLD_INFO bind
    /// opcodes. Authoritative names for codegen helpers that call through
    /// `__got` stubs (`ADRP x16; LDR x16,[x16,#off]; BR x16`).
    pub bind_symbols: HashMap<u64, String>,
}

impl MachO {
    pub fn new(data: Vec<u8>, is_32bit: bool) -> Result<Self> {
        Self::new_internal(data, is_32bit, false)
    }

    pub fn new_with_codm_fixups(data: Vec<u8>, is_32bit: bool, apply_fixups: bool) -> Result<Self> {
        Self::new_internal(data, is_32bit, apply_fixups)
    }

    fn new_internal(data: Vec<u8>, is_32bit: bool, codm_apply_fixups: bool) -> Result<Self> {
        let mut macho = Self {
            stream: BinaryStream::new(data),
            is_32bit,
            codm_apply_fixups,
            is_arm64e: false,
            segments: Vec::new(),
            sections: Vec::new(),
            symbols: Vec::new(),
            string_table: Vec::new(),
            vmaddr: 0,
            dyld_info: None,
            chained_fixups: None,
            bind_symbols: HashMap::new(),
        };
        macho.stream.is_32bit = is_32bit;
        macho.load()?;
        // Apply the linker's rebase pass offline whenever the binary still
        // carries the info needed to do it. Without this, on-disk pointers
        // in __mod_init_func / __const / __data stay as RVAs (e.g. raw
        // 0x04f6aea0 instead of the runtime VA 0x104f6aea0), and downstream
        // map_vatr -> read_exact blows up with "failed to fill whole buffer"
        // because the truncated VA maps to a bogus file offset past EOF.
        //
        // - chained_fixups: modern format used on arm64e and recent arm64
        //   App Store builds (LC_DYLD_CHAINED_FIXUPS)
        // - dyld_info:      legacy format still used by lots of Unity iOS
        //   builds (LC_DYLD_INFO / LC_DYLD_INFO_ONLY rebase opcodes)
        //
        // Only the rebase parts run -- bind / lazy_bind aren't touched.
        if !macho.codm_apply_fixups
            && (macho.chained_fixups.is_some() || macho.dyld_info.is_some() || macho.is_arm64e)
        {
            macho.codm_apply_fixups = true;
        }
        if macho.codm_apply_fixups {
            if let Err(e) = macho.process_apple_fixups() {
                if std::env::var("ILR_BIND_DEBUG").is_ok() {
                    eprintln!("[bind] process_apple_fixups error: {e:?}");
                }
            }
        }
        Ok(macho)
    }

    fn load(&mut self) -> Result<()> {
        self.stream.set_position(0);
        let magic = self.stream.read_u32()?;

        let expected = if self.is_32bit { MH_MAGIC } else { MH_MAGIC_64 };
        if magic != expected {
            return Err(Error::InvalidFormat("Invalid Mach-O magic".into()));
        }

        let cputype = self.stream.read_i32()? as u32;
        let cpusubtype = self.stream.read_i32()? as u32;
        // CPU_TYPE_ARM64 = 0x0100000C, CPU_SUBTYPE_ARM64E = 2 (low 24 bits)
        self.is_arm64e = cputype == 0x0100000C && (cpusubtype & 0x00FFFFFF) == 2;
        let _filetype = self.stream.read_u32()?;
        let ncmds = self.stream.read_u32()?;
        let _sizeofcmds = self.stream.read_u32()?;
        let _flags = self.stream.read_u32()?;

        if !self.is_32bit {
            let _reserved = self.stream.read_u32()?;
        }

        let mut symtab: Option<SymtabCmd> = None;
        let mut cryptid = 0u32;

        let lc_segment_cmd = if self.is_32bit {
            LC_SEGMENT
        } else {
            LC_SEGMENT_64
        };
        let lc_enc_cmd = if self.is_32bit {
            LC_ENCRYPTION_INFO
        } else {
            LC_ENCRYPTION_INFO_64
        };

        for _ in 0..ncmds {
            let cmd_pos = self.stream.position();
            let cmd = self.stream.read_u32()?;
            let cmdsize = self.stream.read_u32()?;

            if cmd == lc_segment_cmd {
                self.stream.set_position(cmd_pos);
                let seg = self.read_segment()?;
                let nsects = seg.nsects;

                if seg.segname == "__TEXT" {
                    self.vmaddr = seg.vmaddr;
                }

                self.segments.push(seg);

                for _ in 0..nsects {
                    let section = self.read_section()?;
                    self.sections.push(section);
                }
            } else if cmd == LC_SYMTAB {
                self.stream.set_position(cmd_pos + 8);
                symtab = Some(SymtabCmd {
                    symoff: self.stream.read_u32()?,
                    nsyms: self.stream.read_u32()?,
                    stroff: self.stream.read_u32()?,
                    strsize: self.stream.read_u32()?,
                });
            } else if cmd == lc_enc_cmd {
                self.stream.set_position(cmd_pos + 8);
                let _cryptoff = self.stream.read_u32()?;
                let _cryptsize = self.stream.read_u32()?;
                cryptid = self.stream.read_u32()?;
            } else if cmd == LC_DYLD_INFO || cmd == LC_DYLD_INFO_ONLY {
                self.stream.set_position(cmd_pos + 8);
                let rebase_off = self.stream.read_u32()?;
                let rebase_size = self.stream.read_u32()?;
                let bind_off = self.stream.read_u32()?;
                let bind_size = self.stream.read_u32()?;
                let weak_bind_off = self.stream.read_u32()?;
                let weak_bind_size = self.stream.read_u32()?;
                let lazy_bind_off = self.stream.read_u32()?;
                let lazy_bind_size = self.stream.read_u32()?;
                self.dyld_info = Some(DyldInfoCmd {
                    rebase_off,
                    rebase_size,
                    bind_off,
                    bind_size,
                    lazy_bind_off,
                    lazy_bind_size,
                    weak_bind_off,
                    weak_bind_size,
                });
            } else if cmd == LC_DYLD_CHAINED_FIXUPS {
                self.stream.set_position(cmd_pos + 8);
                let dataoff = self.stream.read_u32()?;
                let datasize = self.stream.read_u32()?;
                self.chained_fixups = Some(ChainedFixupsCmd { dataoff, datasize });
            }

            self.stream.set_position(cmd_pos + cmdsize as u64);
        }

        if let Some(st) = symtab {
            self.load_symbols(&st)?;
        }

        if cryptid != 0 {
            eprintln!("ERROR: This Mach-O executable is encrypted and cannot be processed.");
        }

        Ok(())
    }

    fn read_segment(&mut self) -> Result<Segment> {
        let _cmd = self.stream.read_u32()?;
        let _cmdsize = self.stream.read_u32()?;
        let segname_bytes = self.stream.read_bytes(16)?;
        let segname = String::from_utf8_lossy(&segname_bytes)
            .trim_end_matches('\0')
            .to_string();

        let mut seg = Segment::default();
        seg.segname = segname;
        if self.is_32bit {
            seg.vmaddr = self.stream.read_u32()? as u64;
            seg.vmsize = self.stream.read_u32()? as u64;
            seg.fileoff = self.stream.read_u32()? as u64;
            seg.filesize = self.stream.read_u32()? as u64;
        } else {
            seg.vmaddr = self.stream.read_u64()?;
            seg.vmsize = self.stream.read_u64()?;
            seg.fileoff = self.stream.read_u64()?;
            seg.filesize = self.stream.read_u64()?;
        }
        let _maxprot = self.stream.read_i32()?;
        let _initprot = self.stream.read_i32()?;
        seg.nsects = self.stream.read_u32()?;
        let _flags = self.stream.read_u32()?;

        Ok(seg)
    }

    fn read_section(&mut self) -> Result<Section> {
        let sectname_bytes = self.stream.read_bytes(16)?;
        let sectname = String::from_utf8_lossy(&sectname_bytes)
            .trim_end_matches('\0')
            .to_string();
        let _segname = self.stream.read_bytes(16)?;

        let mut sect = Section::default();
        sect.sectname = sectname;
        if self.is_32bit {
            sect.addr = self.stream.read_u32()? as u64;
            sect.size = self.stream.read_u32()? as u64;
        } else {
            sect.addr = self.stream.read_u64()?;
            sect.size = self.stream.read_u64()?;
        }
        sect.offset = self.stream.read_u32()?;
        let _align = self.stream.read_u32()?;
        let _reloff = self.stream.read_u32()?;
        let _nreloc = self.stream.read_u32()?;
        sect.flags = self.stream.read_u32()?;
        let _reserved1 = self.stream.read_u32()?;
        let _reserved2 = self.stream.read_u32()?;
        if !self.is_32bit {
            let _reserved3 = self.stream.read_u32()?;
        }

        Ok(sect)
    }

    fn load_symbols(&mut self, symtab: &SymtabCmd) -> Result<()> {
        self.stream.set_position(symtab.stroff as u64);
        self.string_table = self.stream.read_bytes(symtab.strsize as usize)?;

        self.stream.set_position(symtab.symoff as u64);
        self.symbols.clear();

        for _ in 0..symtab.nsyms {
            let mut entry = NlistEntry::default();
            entry.n_strx = self.stream.read_u32()?;
            entry.n_type = self.stream.read_u8()?;
            entry.n_sect = self.stream.read_u8()?;
            if self.is_32bit {
                entry.n_desc = self.stream.read_i16()?;
                entry.n_value = self.stream.read_u32()? as u64;
            } else {
                let n_desc_u16 = self.stream.read_u16()?;
                entry.n_desc = n_desc_u16 as i16;
                entry.n_value = self.stream.read_u64()?;
            }
            self.symbols.push(entry);
        }

        Ok(())
    }

    fn get_symbol_name(&self, sym: &NlistEntry) -> String {
        let start = sym.n_strx as usize;
        if start >= self.string_table.len() {
            return String::new();
        }
        let end = self.string_table[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| start + p)
            .unwrap_or(self.string_table.len());
        String::from_utf8_lossy(&self.string_table[start..end]).to_string()
    }

    pub fn map_vatr(&self, addr: u64) -> Result<u64> {
        let try_addr = |a: u64| -> Option<u64> {
            for sect in &self.sections {
                if a >= sect.addr && a <= sect.addr + sect.size {
                    if sect.sectname == "__bss" {
                        continue;
                    }
                    return Some(a - sect.addr + sect.offset as u64);
                }
            }
            for seg in &self.segments {
                if a >= seg.vmaddr && a < seg.vmaddr + seg.vmsize {
                    return Some(a - seg.vmaddr + seg.fileoff);
                }
            }
            None
        };
        if let Some(off) = try_addr(addr) {
            return Ok(off);
        }
        if (self.codm_apply_fixups || self.is_arm64e) && (addr >> 52) != 0 {
            let target_low = addr & 0x0000_000F_FFFF_FFFF;
            let fixed = target_low.wrapping_add(self.vmaddr);
            if let Some(off) = try_addr(fixed) {
                return Ok(off);
            }
        }
        Err(Error::AddressNotMapped(addr))
    }

    pub fn map_rtva(&self, offset: u64) -> u64 {
        for sect in &self.sections {
            if offset >= sect.offset as u64 && offset <= sect.offset as u64 + sect.size {
                if sect.sectname == "__bss" {
                    return 0;
                }
                return offset - sect.offset as u64 + sect.addr;
            }
        }
        for seg in &self.segments {
            if offset >= seg.fileoff && offset < seg.fileoff + seg.filesize {
                return offset - seg.fileoff + seg.vmaddr;
            }
        }
        0
    }

    pub fn read_uint_ptr(&mut self) -> Result<u64> {
        if self.is_32bit {
            return Ok(self.stream.read_u32()? as u64);
        }
        let pointer = self.stream.read_u64()?;
        if pointer > self.vmaddr + 0xFFFFFFFF {
            let addr = self.stream.position();
            for sect in &self.sections {
                if addr >= sect.offset as u64 && addr <= sect.offset as u64 + sect.size {
                    if sect.sectname == "__const" || sect.sectname == "__data" {
                        let rva = pointer - self.vmaddr;
                        let masked = rva & 0xFFFFFFFF;
                        return Ok(masked + self.vmaddr);
                    }
                    break;
                }
            }
        }
        Ok(pointer)
    }

    pub fn symbol_search(&self) -> Option<(u64, u64)> {
        let mut code_reg = 0u64;
        let mut metadata_reg = 0u64;

        for sym in &self.symbols {
            let name = self.get_symbol_name(sym);
            if name == "_g_CodeRegistration" {
                code_reg = sym.n_value;
            } else if name == "_g_MetadataRegistration" {
                metadata_reg = sym.n_value;
            }
        }

        if code_reg > 0 && metadata_reg > 0 {
            Some((code_reg, metadata_reg))
        } else {
            None
        }
    }

    pub fn list_exported_symbols(&self) -> Vec<(String, u64)> {
        let mut exports = Vec::new();
        for sym in &self.symbols {
            if sym.n_value == 0 || sym.n_sect == 0 {
                continue;
            }
            let name = self.get_symbol_name(sym);
            if name.is_empty() {
                continue;
            }
            let name = if name.starts_with('_') {
                name[1..].to_string()
            } else {
                name
            };
            exports.push((name, sym.n_value));
        }
        exports
    }

    pub fn search_mod_init_func(&mut self, version: f64) -> Option<(u64, u64)> {
        let mod_init = self
            .sections
            .iter()
            .find(|s| s.sectname == "__mod_init_func")?
            .clone();

        if self.is_32bit {
            self.search_32bit(&mod_init, version)
        } else {
            self.search_64bit(&mod_init, version)
        }
    }

    fn search_32bit(&mut self, mod_init: &Section, version: f64) -> Option<(u64, u64)> {
        let feature_bytes_1: [u8; 2] = [0x0, 0x22]; // MOVS R2, #0
        let feature_bytes_2: [u8; 4] = [0x78, 0x44, 0x79, 0x44]; // ADD R0, PC; ADD R1, PC

        let count = (mod_init.size / 4) as usize;
        self.stream.set_position(mod_init.offset as u64);
        let mut addrs = Vec::with_capacity(count);
        for _ in 0..count {
            addrs.push(self.stream.read_u32().unwrap_or(0) as u64);
        }

        for a in &addrs {
            if *a == 0 {
                continue;
            }
            let i = *a - 1; // ARM Thumb bit
            if let Ok(mapped) = self.map_vatr(i) {
                self.stream.set_position(mapped + 4);
                let buff = self.stream.read_bytes(2).unwrap_or_default();
                if buff == feature_bytes_1 {
                    self.stream.set_position(mapped + 18);
                    let buff2 = self.stream.read_bytes(4).unwrap_or_default();
                    if buff2 == feature_bytes_2 {
                        self.stream.set_position(mapped + 10);
                        let mov_bytes = self.stream.read_bytes(8).unwrap_or_default();
                        let subaddr = decode_mov_arm32(&mov_bytes).wrapping_add(i + 24 - 1);
                        if let Ok(rsubaddr) = self.map_vatr(subaddr) {
                            self.stream.set_position(rsubaddr);
                            let mov_bytes2 = self.stream.read_bytes(8).unwrap_or_default();
                            let ptr = decode_mov_arm32(&mov_bytes2).wrapping_add(subaddr + 16);
                            if let Ok(ptr_offset) = self.map_vatr(ptr) {
                                self.stream.set_position(ptr_offset);
                                let metadata_registration =
                                    self.stream.read_u32().unwrap_or(0) as u64;

                                self.stream.set_position(rsubaddr + 8);
                                let buff3 = self.stream.read_bytes(4).unwrap_or_default();
                                self.stream.set_position(rsubaddr + 14);
                                let buff4 = self.stream.read_bytes(4).unwrap_or_default();
                                let combined: Vec<u8> =
                                    buff3.iter().chain(buff4.iter()).cloned().collect();

                                let code_extra = if version < 21.0 { 22u64 } else { 26u64 };
                                let code_registration =
                                    decode_mov_arm32(&combined).wrapping_add(subaddr + code_extra);

                                return Some((code_registration, metadata_registration));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn search_64bit(&mut self, mod_init: &Section, version: f64) -> Option<(u64, u64)> {
        let feature_bytes_1: [u8; 4] = [0x2, 0x0, 0x80, 0xD2]; // MOV X2, #0
        let feature_bytes_2: [u8; 4] = [0x3, 0x0, 0x80, 0x52]; // MOV W3, #0

        let count = (mod_init.size / 8) as usize;
        self.stream.set_position(mod_init.offset as u64);
        let mut addrs = Vec::with_capacity(count);
        for _ in 0..count {
            addrs.push(self.stream.read_u64().unwrap_or(0));
        }

        let code_registration = 0u64;
        let metadata_registration = 0u64;

        for i in &addrs {
            if *i == 0 {
                continue;
            }
            let mapped = match self.map_vatr(*i) {
                Ok(m) => m,
                Err(_) => continue,
            };

            if version < 23.0 {
                // v<23: FeatureBytes1 then FeatureBytes2, OR vice versa
                self.stream.set_position(mapped);
                let buff = self.stream.read_bytes(4).unwrap_or_default();
                if buff == feature_bytes_1 {
                    let buff2 = self.stream.read_bytes(4).unwrap_or_default();
                    if buff2 == feature_bytes_2 {
                        self.stream.set_position(mapped + 16);
                        let inst = self.stream.read_bytes(4).unwrap_or_default();
                        if is_adr(&inst) {
                            let subaddr = decode_adr(*i + 16, &inst);
                            if let Some(result) = self.resolve_adrp_pair(*i, subaddr) {
                                return Some(result);
                            }
                        }
                    }
                } else {
                    self.stream.set_position(mapped + 0x10);
                    let buff2 = self.stream.read_bytes(4).unwrap_or_default();
                    if buff2 == feature_bytes_2 {
                        let buff3 = self.stream.read_bytes(4).unwrap_or_default();
                        if buff3 == feature_bytes_1 {
                            self.stream.set_position(mapped + 8);
                            let inst = self.stream.read_bytes(4).unwrap_or_default();
                            if is_adr(&inst) {
                                let subaddr = decode_adr(*i + 8, &inst);
                                if let Some(result) = self.resolve_adrp_pair(*i, subaddr) {
                                    return Some(result);
                                }
                            }
                        }
                    }
                }
            }

            if version >= 23.0 && version < 24.0 {
                // v==23: offset+16 has FeatureBytes1 then FeatureBytes2
                self.stream.set_position(mapped + 16);
                let buff = self.stream.read_bytes(4).unwrap_or_default();
                if buff == feature_bytes_1 {
                    let buff2 = self.stream.read_bytes(4).unwrap_or_default();
                    if buff2 == feature_bytes_2 {
                        self.stream.set_position(mapped + 8);
                        let inst = self.stream.read_bytes(4).unwrap_or_default();
                        let subaddr = decode_adr(*i + 8, &inst);
                        if let Some(result) = self.resolve_adrp_pair(*i, subaddr) {
                            return Some(result);
                        }
                    }
                }
            }

            if version >= 24.0 {
                // v>=24: offset+16 has FeatureBytes2 then FeatureBytes1 (swapped)
                self.stream.set_position(mapped + 16);
                let buff = self.stream.read_bytes(4).unwrap_or_default();
                if buff == feature_bytes_2 {
                    let buff2 = self.stream.read_bytes(4).unwrap_or_default();
                    if buff2 == feature_bytes_1 {
                        self.stream.set_position(mapped + 8);
                        let inst = self.stream.read_bytes(4).unwrap_or_default();
                        let subaddr = decode_adr(*i + 8, &inst);
                        if let Some(result) = self.resolve_adrp_pair(*i, subaddr) {
                            return Some(result);
                        }
                    }
                }
            }
        }

        if code_registration != 0 && metadata_registration != 0 {
            Some((code_registration, metadata_registration))
        } else {
            None
        }
    }

    fn resolve_adrp_pair(&mut self, _base_addr: u64, subaddr: u64) -> Option<(u64, u64)> {
        let rsubaddr = self.map_vatr(subaddr).ok()?;
        self.stream.set_position(rsubaddr);
        let adrp_bytes = self.stream.read_bytes(4).unwrap_or_default();
        let add_bytes = self.stream.read_bytes(4).unwrap_or_default();
        let code_registration = decode_adrp(subaddr, &adrp_bytes) + decode_add(&add_bytes);

        self.stream.set_position(rsubaddr + 8);
        let adrp_bytes2 = self.stream.read_bytes(4).unwrap_or_default();
        let add_bytes2 = self.stream.read_bytes(4).unwrap_or_default();
        let metadata_registration =
            decode_adrp(subaddr + 8, &adrp_bytes2) + decode_add(&add_bytes2);

        if code_registration != 0 && metadata_registration != 0 {
            Some((code_registration, metadata_registration))
        } else {
            None
        }
    }

    /// Data sections scanned for v27+ metadata usages (FieldRva pointers, etc.).
    pub fn data_search_sections(&self) -> Vec<SearchSection> {
        let mut data_list = Vec::new();
        for sect in &self.sections {
            let search_section = SearchSection::new(
                sect.offset as u64,
                sect.offset as u64 + sect.size,
                sect.addr,
                sect.addr + sect.size,
            );
            if self.is_32bit {
                if sect.sectname == "__const" {
                    data_list.push(search_section);
                }
            } else if sect.sectname == "__const"
                || sect.sectname == "__cstring"
                || sect.sectname == "__data"
            {
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
        let mut bss_list = Vec::new();
        let mut all_sections = Vec::new();

        for sect in &self.sections {
            let search_section = SearchSection::new(
                sect.offset as u64,
                sect.offset as u64 + sect.size,
                sect.addr,
                sect.addr + sect.size,
            );

            all_sections.push(search_section.clone());

            if sect.flags == (S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS) {
                exec_list.push(search_section);
            } else if sect.flags == 1 {
                bss_list.push(search_section);
            }
        }

        SectionHelper::new(
            self.stream.data(),
            self.is_32bit,
            version,
            false,
            all_sections,
            data_list,
            exec_list,
            bss_list,
            method_count,
            type_definitions_count,
            metadata_usages_count,
            image_count,
        )
    }

    /// Raw bytes + image-relative start of a named `__TEXT`/`__DATA` section
    /// ("segname,sectname" or just "sectname"). Used by exception-table
    /// recovery to reach `__unwind_info` / `__gcc_except_tab`.
    pub fn section_bytes_by_name(&self, name: &str) -> Option<(u64, Vec<u8>)> {
        let wanted = name.rsplit(',').next()?.trim();
        for sect in &self.sections {
            if sect.sectname == wanted {
                let start = sect.offset as u64;
                let end = (start + sect.size).min(self.stream.len());
                if start >= end {
                    return None;
                }
                let bytes = self.stream.data()[start as usize..end as usize].to_vec();
                return Some((sect.addr, bytes));
            }
        }
        None
    }

    pub fn image_base(&self) -> u64 {
        self.vmaddr
    }

    fn process_apple_fixups(&mut self) -> Result<()> {
        if self.dyld_info.is_some() {
            self.process_dyld_bind_opcodes(BindStream::Regular)?;
            self.process_dyld_bind_opcodes(BindStream::Lazy)?;
            self.process_dyld_bind_opcodes(BindStream::Weak)?;
            self.process_dyld_rebase_opcodes()?;
        }
        if self.chained_fixups.is_some() {
            self.process_chained_fixups()?;
        }
        Ok(())
    }

    /// Parse an LC_DYLD_INFO bind opcode stream (regular / lazy / weak) and
    /// record every bound slot's VA → symbol name in `bind_symbols`.
    fn process_dyld_bind_opcodes(&mut self, stream: BindStream) -> Result<()> {
        let Some(cmd) = self.dyld_info.clone() else {
            return Ok(());
        };
        let (table_off, table_size) = match stream {
            BindStream::Regular => (cmd.bind_off, cmd.bind_size),
            BindStream::Lazy => (cmd.lazy_bind_off, cmd.lazy_bind_size),
            BindStream::Weak => (cmd.weak_bind_off, cmd.weak_bind_size),
        };
        if table_off == 0 || table_size == 0 {
            return Ok(());
        }
        let base = table_off as u64;
        let end = base + table_size as u64;
        if end > self.stream.len() as u64 {
            if std::env::var("ILR_BIND_DEBUG").is_ok() {
                eprintln!(
                    "[bind] table out of bounds: {base:#x}+{table_size:#x} > {:#x}",
                    self.stream.len()
                );
            }
            return Ok(());
        }
        if std::env::var("ILR_BIND_DEBUG").is_ok() {
            eprintln!(
                "[bind] parsing table {stream:?} base={base:#x} end={end:#x} segments={} is32={}",
                self.segments.len(),
                self.is_32bit
            );
        }

        // Bind opcodes per dyld `MachOBindingOpcodes` (dyld source,
        // mach-o/bind.h). Verified against a real UnityFramework table:
        // `0x11 0x40 "__ZNKSt…"` = SET_DYLIB_ORDINAL_IMM(1) +
        // SET_SYMBOL_TRAILING.
        const BIND_OPCODE_MASK: u8 = 0xF0;
        const OP_DONE: u8 = 0x00;
        const OP_SET_DYLIB_ORD_IMM: u8 = 0x10;
        const OP_SET_DYLIB_ORD_ULEB: u8 = 0x20;
        const OP_SET_DYLIB_SPECIAL_IMM: u8 = 0x30;
        const OP_SET_SYMBOL_STRING: u8 = 0x40;
        const OP_SET_TYPE_IMM: u8 = 0x50;
        const OP_SET_ADDEND_SLEB: u8 = 0x60;
        const OP_SET_SEG_OFF_ULEB: u8 = 0x70;
        const OP_ADD_ADDR_ULEB: u8 = 0x80;
        // NOTE: there is deliberately NO `ADD_ADDR_IMM_SCALED` here.
        //
        // dyld's bind opcode set (MachOFile.h / LLVM MachOObjectFile.cpp) is:
        //   0x00 DONE            0x10 DYLIB_ORDINAL_IMM   0x20 DYLIB_ORDINAL_ULEB
        //   0x30 DYLIB_SPECIAL   0x40 SYMBOL_TRAILING     0x50 TYPE_IMM
        //   0x60 ADDEND_SLEB     0x70 SEGMENT_AND_OFFSET  0x80 ADD_ADDR_ULEB
        //   0x90 DO_BIND         0xA0 DO_BIND_ADD_ADDR_ULEB
        //   0xB0 DO_BIND_ADD_ADDR_IMM_SCALED           0xC0 DO_BIND_ULEB_TIMES_SKIPPING_ULEB
        //   0xD0 THREADED
        // A legacy `ADD_ADDR_IMM_SCALED` was `0x40 | 0x80 == 0xC0`, which now
        // belongs to DO_BIND_ULEB_TIMES_SKIPPING_ULEB. Keeping an arm for it
        // shadowed the real opcode AND failed to consume that opcode's two
        // ULEB operands, desynchronising the rest of the stream: on
        // UnityFramework this yielded 29 bound slots instead of 56,298.
        const OP_DO_BIND: u8 = 0x90;
        const OP_DO_BIND_ADD_ADDR_ULEB: u8 = 0xA0;
        const OP_DO_BIND_ADD_ADDR_IMM_SCALED: u8 = 0xB0;
        const OP_DO_BIND_TIMES_SKIPPING: u8 = 0xC0;
        const OP_THREADED: u8 = 0xD0;

        let ptr_size: u64 = if self.is_32bit { 4 } else { 8 };
        let segments = self.segments.clone();
        let mut seg_index: usize = 0;
        let mut seg_offset: u64 = 0;
        let mut sym_index: Option<u32> = None;
        let mut sym_inline: Option<String> = None;

        let mut slot_name: Vec<(u64, String)> = Vec::new();
        let mut op_hist = [0u32; 16];
        // hard cap: malformed tables must not explode memory
        let max_slots = 1_000_000usize;

        let bind_one = |seg_index: usize,
                        seg_offset: u64,
                        sym_index: Option<u32>,
                        sym_inline: &Option<String>,
                        out: &mut Vec<(u64, String)>| {
            if out.len() >= max_slots {
                return;
            }
            if let Some(seg) = segments.get(seg_index) {
                let va = seg.vmaddr.wrapping_add(seg_offset);
                let name = if let Some(i) = sym_index {
                    // resolved later — inline closure can't borrow self;
                    // symbol index lookups happen via pre-scanned strings
                    i.to_string()
                } else {
                    sym_inline.clone().unwrap_or_default()
                };
                if !name.is_empty() {
                    out.push((va, name));
                }
            }
        };

        self.stream.set_position(base);
        if std::env::var("ILR_BIND_DEBUG").is_ok() {
            let first = self.stream.read_u8().ok();
            eprintln!("[bind] first byte at {base:#x}: {:?}", first);
            self.stream.set_position(base);
        }
        while self.stream.position() < end {
            let byte = self.stream.read_u8()?;
            let op = byte & BIND_OPCODE_MASK;
            let imm = byte & 0x0F;
            if std::env::var("ILR_BIND_DEBUG").is_ok() {
                op_hist[op as usize >> 4] += 1;
            }
            match op {
                OP_DONE => {
                    // DONE terminates a *sequence*, not necessarily the stream.
                    // The legacy lazy table is a series of DONE-separated
                    // entries, but a unified (dyld3/dyld4) bind stream can also
                    // carry DONE between sequences, so stopping at the first
                    // one silently discards every later binding. The `end`
                    // bound (from bind_size / lazy_bind_size) is the real
                    // terminator, so reset the per-sequence state and keep
                    // walking.
                    sym_index = None;
                    sym_inline = None;
                    seg_offset = 0;
                }
                OP_SET_TYPE_IMM | OP_SET_DYLIB_ORD_IMM | OP_SET_DYLIB_SPECIAL_IMM => {}
                OP_SET_SEG_OFF_ULEB => {
                    seg_index = imm as usize;
                    seg_offset = read_uleb128(&mut self.stream)?;
                }
                OP_ADD_ADDR_ULEB => {
                    seg_offset = seg_offset.wrapping_add(read_uleb128(&mut self.stream)?);
                }
                OP_SET_DYLIB_ORD_ULEB => {
                    let _ = read_uleb128(&mut self.stream)?;
                }
                OP_SET_SYMBOL_STRING => {
                    let mut name = Vec::new();
                    while self.stream.position() < end {
                        let b = self.stream.read_u8()?;
                        if b == 0 {
                            break;
                        }
                        name.push(b);
                    }
                    sym_inline = Some(String::from_utf8_lossy(&name).to_string());
                }
                OP_SET_ADDEND_SLEB => {
                    let _ = read_uleb128(&mut self.stream)?;
                }
                op if op == OP_DO_BIND => {
                    if std::env::var("ILR_BIND_DEBUG").is_ok() && slot_name.len() < 3 {
                        eprintln!(
                            "[bind] DO_BIND seg={seg_index} off={seg_offset:#x} sym_idx={sym_index:?} inline={:?}",
                            sym_inline
                        );
                    }
                    bind_one(
                        seg_index,
                        seg_offset,
                        sym_index,
                        &sym_inline,
                        &mut slot_name,
                    );
                    seg_offset = seg_offset.wrapping_add(ptr_size);
                    // The symbol is intentionally NOT cleared here. dyld keeps
                    // the current symbol until SET_SYMBOL_TRAILING_FLAGS_IMM
                    // changes it (or a DONE breaks the sequence), which is how
                    // one symbol can bind several consecutive slots. Clearing
                    // it per bind named only 18,691 of 56,298 slots on
                    // UnityFramework — one third of all symbolication.
                }
                OP_DO_BIND_ADD_ADDR_ULEB => {
                    bind_one(
                        seg_index,
                        seg_offset,
                        sym_index,
                        &sym_inline,
                        &mut slot_name,
                    );
                    seg_offset = seg_offset
                        .wrapping_add(ptr_size)
                        .wrapping_add(read_uleb128(&mut self.stream)?);
                }
                OP_DO_BIND_ADD_ADDR_IMM_SCALED => {
                    bind_one(
                        seg_index,
                        seg_offset,
                        sym_index,
                        &sym_inline,
                        &mut slot_name,
                    );
                    seg_offset = seg_offset.wrapping_add(ptr_size * (1 + imm as u64));
                }
                OP_DO_BIND_TIMES_SKIPPING => {
                    let count = read_uleb128(&mut self.stream)?;
                    let skip = read_uleb128(&mut self.stream)?;
                    if count < 100_000 {
                        for _ in 0..count {
                            bind_one(
                                seg_index,
                                seg_offset,
                                sym_index,
                                &sym_inline,
                                &mut slot_name,
                            );
                            seg_offset = seg_offset.wrapping_add(ptr_size).wrapping_add(skip);
                        }
                    }
                }
                OP_THREADED => {
                    // 0xD0/0xE0 threaded bind sub-opcodes (arm64e chained
                    // binds); the sub-opcode is in `imm`. Sub-opcode 0
                    // (SET_BIND_ORDINAL_TABLE_SIZE_ULEB) carries a ULEB that
                    // MUST be consumed — ignoring it desynchronises the rest
                    // of the stream exactly like the old 0xC0 arm did.
                    // Ordinals of a threaded chain are rebound through the
                    // chained-fixups path, which is where the slot names for
                    // those bindings come from.
                    if imm == 0 {
                        let _ = read_uleb128(&mut self.stream)?;
                    }
                }
                _ => {}
            }
        }

        if std::env::var("ILR_BIND_DEBUG").is_ok() {
            eprintln!(
                "[bind] {stream:?} slots collected: {} (first: {:?}) hist={:?}",
                slot_name.len(),
                slot_name.first(),
                op_hist
            );
        }
        self.bind_symbols.extend(slot_name);
        Ok(())
    }

    fn process_chained_fixups(&mut self) -> Result<()> {
        let cmd = match &self.chained_fixups {
            Some(c) => c.clone(),
            None => return Ok(()),
        };
        if cmd.datasize < 32 {
            return Ok(());
        }
        let base = cmd.dataoff as u64;

        self.stream.set_position(base);
        let _fixups_version = self.stream.read_u32()?;
        let starts_offset = self.stream.read_u32()?;
        let _imports_offset = self.stream.read_u32()?;
        let _symbols_offset = self.stream.read_u32()?;
        let _imports_count = self.stream.read_u32()?;
        let _imports_format = self.stream.read_u32()?;
        let _symbols_format = self.stream.read_u32()?;

        let starts_pos = base + starts_offset as u64;
        self.stream.set_position(starts_pos);
        let seg_count = self.stream.read_u32()?;

        for i in 0..seg_count {
            self.stream.set_position(starts_pos + 4 + (i as u64) * 4);
            let seg_info_off = self.stream.read_u32()?;
            if seg_info_off == 0 {
                continue;
            }
            let seg_info_pos = starts_pos + seg_info_off as u64;
            self.process_chained_segment(seg_info_pos)?;
        }
        Ok(())
    }

    fn process_chained_segment(&mut self, seg_info_pos: u64) -> Result<()> {
        self.stream.set_position(seg_info_pos);
        let _size = self.stream.read_u32()?;
        let page_size = self.stream.read_u16()?;
        let pointer_format = self.stream.read_u16()?;
        let segment_offset = self.stream.read_u64()?;
        let _max_valid_pointer = self.stream.read_u32()?;
        let page_count = self.stream.read_u16()?;
        let _padding = self.stream.read_u16()?;

        if page_size == 0 {
            return Ok(());
        }

        let mut page_starts = Vec::with_capacity(page_count as usize);
        for _ in 0..page_count {
            page_starts.push(self.stream.read_u16()?);
        }

        let seg_file_off = match self
            .segments
            .iter()
            .find(|s| s.vmaddr == segment_offset || s.fileoff == segment_offset)
        {
            Some(s) => s.fileoff,
            None => self
                .segments
                .iter()
                .min_by_key(|s| {
                    if s.vmaddr >= segment_offset {
                        s.vmaddr - segment_offset
                    } else {
                        u64::MAX
                    }
                })
                .map(|s| s.fileoff)
                .unwrap_or(0),
        };
        let seg_vmaddr = self
            .segments
            .iter()
            .find(|s| s.fileoff == seg_file_off)
            .map(|s| s.vmaddr)
            .unwrap_or(self.vmaddr);
        let _ = seg_vmaddr;

        const DYLD_CHAINED_PTR_START_NONE: u16 = 0xFFFF;

        for (page_index, &start) in page_starts.iter().enumerate() {
            if start == DYLD_CHAINED_PTR_START_NONE {
                continue;
            }
            let page_file_base = seg_file_off + (page_index as u64) * (page_size as u64);
            let mut chain_off = page_file_base + start as u64;

            loop {
                if chain_off >= self.stream.data().len() as u64 {
                    break;
                }
                self.stream.set_position(chain_off);
                let raw = self.stream.read_u64()?;

                let (target, next, stride) = decode_chained_pointer(raw, pointer_format);
                if let Some(mut rebased_va) = target {
                    if is_offset_format(pointer_format) {
                        rebased_va = rebased_va.wrapping_add(self.vmaddr);
                    }
                    self.stream.set_position(chain_off);
                    self.stream.write_u64(rebased_va)?;
                }

                if next == 0 {
                    break;
                }
                chain_off = chain_off.wrapping_add((next as u64).wrapping_mul(stride as u64));
            }
        }

        Ok(())
    }

    fn process_dyld_rebase_opcodes(&mut self) -> Result<()> {
        let cmd = match &self.dyld_info {
            Some(c) => c.clone(),
            None => return Ok(()),
        };
        if cmd.rebase_size == 0 {
            return Ok(());
        }
        let base = cmd.rebase_off as u64;
        let end = base + cmd.rebase_size as u64;
        let ptr_size = if self.is_32bit { 4u64 } else { 8u64 };

        const REBASE_OPCODE_MASK: u8 = 0xF0;
        const REBASE_IMMEDIATE_MASK: u8 = 0x0F;
        const REBASE_OPCODE_DONE: u8 = 0x00;
        const REBASE_OPCODE_SET_TYPE_IMM: u8 = 0x10;
        const REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB: u8 = 0x20;
        const REBASE_OPCODE_ADD_ADDR_ULEB: u8 = 0x30;
        const REBASE_OPCODE_ADD_ADDR_IMM_SCALED: u8 = 0x40;
        const REBASE_OPCODE_DO_REBASE_IMM_TIMES: u8 = 0x50;
        const REBASE_OPCODE_DO_REBASE_ULEB_TIMES: u8 = 0x60;
        const REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB: u8 = 0x70;
        const REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB: u8 = 0x80;
        const REBASE_TYPE_POINTER: u8 = 1;

        let mut seg_index: usize = 0;
        let mut seg_offset: u64 = 0;
        let mut rebase_type: u8 = 0;

        self.stream.set_position(base);
        while self.stream.position() < end {
            let byte = self.stream.read_u8()?;
            let op = byte & REBASE_OPCODE_MASK;
            let imm = byte & REBASE_IMMEDIATE_MASK;
            match op {
                REBASE_OPCODE_DONE => break,
                REBASE_OPCODE_SET_TYPE_IMM => {
                    rebase_type = imm;
                }
                REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB => {
                    seg_index = imm as usize;
                    seg_offset = read_uleb128(&mut self.stream)?;
                }
                REBASE_OPCODE_ADD_ADDR_ULEB => {
                    seg_offset = seg_offset.wrapping_add(read_uleb128(&mut self.stream)?);
                }
                REBASE_OPCODE_ADD_ADDR_IMM_SCALED => {
                    seg_offset = seg_offset.wrapping_add((imm as u64).wrapping_mul(ptr_size));
                }
                REBASE_OPCODE_DO_REBASE_IMM_TIMES => {
                    for _ in 0..imm {
                        self.apply_rebase_at(seg_index, seg_offset, rebase_type, ptr_size)?;
                        seg_offset = seg_offset.wrapping_add(ptr_size);
                    }
                }
                REBASE_OPCODE_DO_REBASE_ULEB_TIMES => {
                    let count = read_uleb128(&mut self.stream)?;
                    for _ in 0..count {
                        self.apply_rebase_at(seg_index, seg_offset, rebase_type, ptr_size)?;
                        seg_offset = seg_offset.wrapping_add(ptr_size);
                    }
                }
                REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB => {
                    self.apply_rebase_at(seg_index, seg_offset, rebase_type, ptr_size)?;
                    seg_offset = seg_offset
                        .wrapping_add(ptr_size)
                        .wrapping_add(read_uleb128(&mut self.stream)?);
                }
                REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB => {
                    let count = read_uleb128(&mut self.stream)?;
                    let skip = read_uleb128(&mut self.stream)?;
                    for _ in 0..count {
                        self.apply_rebase_at(seg_index, seg_offset, rebase_type, ptr_size)?;
                        seg_offset = seg_offset.wrapping_add(ptr_size).wrapping_add(skip);
                    }
                }
                _ => break,
            }
        }
        let _ = rebase_type;
        let _ = REBASE_TYPE_POINTER;
        Ok(())
    }

    fn apply_rebase_at(
        &mut self,
        seg_index: usize,
        seg_offset: u64,
        _rebase_type: u8,
        ptr_size: u64,
    ) -> Result<()> {
        if seg_index >= self.segments.len() {
            return Ok(());
        }
        let seg = &self.segments[seg_index];
        let target_va = seg.vmaddr + seg_offset;
        let file_off = match self.map_vatr(target_va) {
            Ok(o) => o,
            Err(_) => return Ok(()),
        };
        let saved = self.stream.position();
        self.stream.set_position(file_off);
        if ptr_size == 8 {
            let raw = self.stream.read_u64()?;
            if raw != 0 && raw < self.vmaddr {
                self.stream.set_position(file_off);
                self.stream.write_u64(raw.wrapping_add(self.vmaddr))?;
            }
        } else {
            let raw = self.stream.read_u32()? as u64;
            if raw != 0 && raw < self.vmaddr {
                self.stream.set_position(file_off);
                self.stream
                    .write_u32(raw.wrapping_add(self.vmaddr) as u32)?;
            }
        }
        self.stream.set_position(saved);
        Ok(())
    }

    pub fn check_dump(&self) -> bool {
        // Normal/decrypted IPA: __TEXT fileoff=0, vmaddr=0x100000000 (structure intact)
        // Raw memory dump:      __TEXT fileoff=vmaddr (file IS the memory image)
        for seg in &self.segments {
            if seg.segname == "__TEXT" {
                return seg.fileoff > 0 && seg.fileoff == seg.vmaddr;
            }
        }
        false
    }

    pub fn get_rva(&self, pointer: u64) -> u64 {
        pointer - self.vmaddr
    }
}

fn is_adr(inst: &[u8]) -> bool {
    if inst.len() < 4 {
        return false;
    }
    let value = u32::from_le_bytes([inst[0], inst[1], inst[2], inst[3]]);
    (value & 0x9F000000) == 0x10000000
}

fn decode_adr(pc: u64, inst: &[u8]) -> u64 {
    if inst.len() < 4 {
        return 0;
    }
    let value = u32::from_le_bytes([inst[0], inst[1], inst[2], inst[3]]);
    let immhi = ((value >> 5) & 0x7FFFF) as i64;
    let immlo = ((value >> 29) & 0x3) as i64;
    let imm = (immhi << 2) | immlo;
    let sign_extended = if imm & (1 << 20) != 0 {
        imm | !0xFFFFF // sign extend 21-bit
    } else {
        imm
    };
    (pc as i64 + sign_extended) as u64
}

fn decode_adrp(pc: u64, inst: &[u8]) -> u64 {
    if inst.len() < 4 {
        return 0;
    }
    let value = u32::from_le_bytes([inst[0], inst[1], inst[2], inst[3]]);
    let immhi = ((value >> 5) & 0x7FFFF) as i64;
    let immlo = ((value >> 29) & 0x3) as i64;
    let imm = ((immhi << 2) | immlo) << 12;
    let sign_extended = if imm & (1i64 << 32) != 0 {
        imm | !0xFFFFFFFF
    } else {
        imm
    };
    ((pc as i64 & !0xFFF) + sign_extended) as u64
}

fn decode_add(inst: &[u8]) -> u64 {
    if inst.len() < 4 {
        return 0;
    }
    let value = u32::from_le_bytes([inst[0], inst[1], inst[2], inst[3]]);
    let imm12 = (value >> 10) & 0xFFF;
    let shift = (value >> 22) & 0x3;
    if shift == 1 {
        (imm12 << 12) as u64
    } else {
        imm12 as u64
    }
}

fn read_uleb128(stream: &mut BinaryStream) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = stream.read_u8()?;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            break;
        }
    }
    Ok(result)
}

fn is_offset_format(pointer_format: u16) -> bool {
    const DYLD_CHAINED_PTR_64_OFFSET: u16 = 6;
    const DYLD_CHAINED_PTR_ARM64E_USERLAND: u16 = 9;
    const DYLD_CHAINED_PTR_ARM64E_USERLAND24: u16 = 12;
    matches!(
        pointer_format,
        DYLD_CHAINED_PTR_64_OFFSET
            | DYLD_CHAINED_PTR_ARM64E_USERLAND
            | DYLD_CHAINED_PTR_ARM64E_USERLAND24
    )
}

fn decode_chained_pointer(raw: u64, pointer_format: u16) -> (Option<u64>, u32, u32) {
    const DYLD_CHAINED_PTR_ARM64E: u16 = 1;
    const DYLD_CHAINED_PTR_64: u16 = 2;
    const DYLD_CHAINED_PTR_64_OFFSET: u16 = 6;
    const DYLD_CHAINED_PTR_ARM64E_USERLAND: u16 = 9;
    const DYLD_CHAINED_PTR_ARM64E_FIRMWARE: u16 = 10;
    const DYLD_CHAINED_PTR_ARM64E_USERLAND24: u16 = 12;

    match pointer_format {
        DYLD_CHAINED_PTR_64 | DYLD_CHAINED_PTR_64_OFFSET => {
            let bind = (raw >> 63) & 0x1;
            let next = ((raw >> 51) & 0xFFF) as u32;
            if bind != 0 {
                (None, next, 4)
            } else {
                let target = raw & 0x0000_0007_FFFF_FFFF;
                (Some(target), next, 4)
            }
        }
        DYLD_CHAINED_PTR_ARM64E
        | DYLD_CHAINED_PTR_ARM64E_USERLAND
        | DYLD_CHAINED_PTR_ARM64E_FIRMWARE
        | DYLD_CHAINED_PTR_ARM64E_USERLAND24 => {
            let auth = (raw >> 63) & 0x1;
            let bind = (raw >> 62) & 0x1;
            let next = ((raw >> 51) & 0x7FF) as u32;
            if bind != 0 {
                (None, next, 8)
            } else if auth != 0 {
                (Some(raw & 0xFFFF_FFFF), next, 8)
            } else {
                let target = raw & 0x0000_000F_FFFF_FFFF;
                (Some(target), next, 8)
            }
        }
        _ => (None, 0, 4),
    }
}

fn decode_mov_arm32(data: &[u8]) -> u64 {
    if data.len() < 4 {
        return 0;
    }
    let low = u16::from_le_bytes([data[0], data[1]]);
    let high = if data.len() >= 4 {
        u16::from_le_bytes([data[2], data[3]])
    } else {
        0
    };
    let imm8 = (low & 0xFF) as u32;
    let imm3 = ((low >> 12) & 0x7) as u32;
    let i = ((high >> 10) & 0x1) as u32;
    let imm4 = (high & 0xF) as u32;
    let result = (imm4 << 12) | (i << 11) | (imm3 << 8) | imm8;
    result as u64
}
