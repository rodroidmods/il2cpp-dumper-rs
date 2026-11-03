//! WebAssembly (Unity WebGL) handling.
//!
//! A Wasm build has no loader and no ELF-style address mapping: the *linear
//! memory* is the address space, and the data section is a list of
//! (memory offset, bytes) segments — Unity emits one object per segment, so
//! their file offsets and their memory addresses are unrelated and the linker
//! leaves file-level padding between them. Measured on a real Unity WebGL
//! build: 23,164 data segments, 17,267 of them starting at a file offset that
//! is not 4-aligned, 187 KB of padding holes between them.
//!
//! Treating the file as if it were the image fails in two ways, and both were
//! observed here:
//!
//!   * a pointer stored at an aligned *address* commonly sits at an unaligned
//!     file offset, so an alignment check on the file offset silently skips
//!     three quarters of the pointers; and
//!   * a struct can straddle a gap, so reading it as a consecutive run of file
//!     bytes reads padding instead of the struct — which is how a count came
//!     out as `0x10000000` (`Invalid array size: count=268435456`).
//!
//! So the data segments are *materialized* into a linear-memory image, exactly
//! as a Wasm engine does with them, and everything downstream works on that
//! image, where address == file offset and a hole reads as the zeros the
//! runtime would see.

use crate::error::{Error, Result};
use crate::io::BinaryStream;
use crate::search::{SearchSection, SectionHelper};

pub const WASM_MAGIC: u32 = 0x6D736100;
pub const WASM_VERSION: u32 = 1;

/// Upper bound on the materialized image. A Unity WebGL memory is declared in
/// pages (often 32 MB); this only guards against a corrupt or hostile memory
/// section asking for gigabytes.
const MAX_IMAGE_BYTES: u64 = 512 * 1024 * 1024;

const WASM_PAGE_SIZE: u64 = 65536;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmSectionId {
    Custom = 0,
    Type = 1,
    Import = 2,
    Function = 3,
    Table = 4,
    Memory = 5,
    Global = 6,
    Export = 7,
    Start = 8,
    Element = 9,
    Code = 10,
    Data = 11,
    DataCount = 12,
    Unknown = 255,
}

impl From<u8> for WasmSectionId {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Custom,
            1 => Self::Type,
            2 => Self::Import,
            3 => Self::Function,
            4 => Self::Table,
            5 => Self::Memory,
            6 => Self::Global,
            7 => Self::Export,
            8 => Self::Start,
            9 => Self::Element,
            10 => Self::Code,
            11 => Self::Data,
            12 => Self::DataCount,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct WasmDataSegment {
    /// Linear-memory address the segment initializes.
    offset: i64,
    size: usize,
    /// File offset of the segment's bytes.
    data_offset: u64,
}

pub struct Wasm {
    /// The linear-memory image (address == offset), not the `.wasm` file.
    pub stream: BinaryStream,
    pub is_32bit: bool,
    data_segments: Vec<WasmDataSegment>,
    /// Bytes copied into the image, for the report.
    pub initialized_bytes: u64,
    /// Holes between segments (padding the runtime sees as zeros).
    pub hole_bytes: u64,
}

impl Wasm {
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let mut wasm = Self {
            stream: BinaryStream::new(data),
            is_32bit: true,
            data_segments: Vec::new(),
            initialized_bytes: 0,
            hole_bytes: 0,
        };
        wasm.stream.is_32bit = true;
        wasm.load()?;
        Ok(wasm)
    }

    fn load(&mut self) -> Result<()> {
        self.stream.set_position(0);
        let magic = self.stream.read_u32()?;
        if magic != WASM_MAGIC {
            return Err(Error::InvalidFormat("Invalid WebAssembly magic".into()));
        }
        let version = self.stream.read_u32()?;
        if version != WASM_VERSION {
            return Err(Error::InvalidFormat(format!(
                "Unsupported WASM version: {}",
                version
            )));
        }

        let mut declared_memory: Option<u64> = None;
        let data_len = self.stream.len();
        while self.stream.position() < data_len {
            let section_id_byte = self.stream.read_u8()?;
            let section_id = WasmSectionId::from(section_id_byte);
            let section_size = self.read_leb128_unsigned()?;
            let section_offset = self.stream.position();

            match section_id {
                WasmSectionId::Memory => {
                    declared_memory = self.parse_memory_section(section_offset).ok().flatten();
                }
                WasmSectionId::Data => {
                    self.parse_data_section(section_offset, section_size)?;
                }
                WasmSectionId::Custom => {
                    let name_len = self.read_leb128_unsigned()?;
                    let _name = self.stream.read_bytes(name_len)?;
                }
                _ => {}
            }

            self.stream
                .set_position(section_offset + section_size as u64);
        }

        self.materialize_memory(declared_memory)
    }

    /// `memory <count> (<limits>)` — the initial size, in pages.
    fn parse_memory_section(&mut self, offset: u64) -> Result<Option<u64>> {
        self.stream.set_position(offset);
        let count = self.read_leb128_unsigned()?;
        let mut pages = None;
        for _ in 0..count {
            let flags = self.read_leb128_unsigned()?;
            let min = self.read_leb128_unsigned()? as u64;
            if pages.is_none() {
                pages = Some(min.saturating_mul(WASM_PAGE_SIZE));
            }
            if flags & 0x01 != 0 {
                let _max = self.read_leb128_unsigned()?;
            }
        }
        Ok(pages)
    }

    fn parse_data_section(&mut self, offset: u64, _size: usize) -> Result<()> {
        self.stream.set_position(offset);
        let num_segments = self.read_leb128_unsigned()?;

        for _ in 0..num_segments {
            let flags = self.read_leb128_unsigned()?;
            let mut segment = WasmDataSegment::default();

            match flags {
                0 => {
                    segment.offset = self.read_offset_expression()?;
                }
                1 => {
                    // Passive segment: initializes nothing until `memory.init`.
                    segment.offset = -1;
                }
                2 => {
                    let _memory_index = self.read_leb128_unsigned()?;
                    segment.offset = self.read_offset_expression()?;
                }
                _ => {}
            }

            segment.size = self.read_leb128_unsigned()?;
            segment.data_offset = self.stream.position();
            self.stream
                .set_position(self.stream.position() + segment.size as u64);
            self.data_segments.push(segment);
        }

        Ok(())
    }

    /// An `expr` operand for a data segment offset: the `i32.const` form Unity
    /// emits (`i32.const N end`), or a `global.get` for PIC/dylink builds.
    fn read_offset_expression(&mut self) -> Result<i64> {
        let opcode = self.stream.read_u8()?;
        let value = match opcode {
            0x41 => self.read_leb128_signed()?, // i32.const
            0x23 => {
                // `global.get <idx>`: the base only exists at instantiation
                // time, so the segment address is unknown here. Record it as
                // unplaced rather than inventing an address; `i32.const`
                // builds (every statically linked Unity build) are unaffected.
                let _index = self.read_leb128_unsigned()?;
                if self.stream.read_u8()? != 0x0B {
                    return Ok(-1);
                }
                return Ok(-1);
            }
            0x42 => self.read_leb128_signed()?, // i64.const (memory64)
            _ => 0,
        };
        let _end = self.stream.read_u8()?;
        Ok(value)
    }

    /// Build the linear-memory image the runtime would see.
    fn materialize_memory(&mut self, declared: Option<u64>) -> Result<()> {
        let mut highest_end = 0u64;
        for segment in &self.data_segments {
            if segment.offset < 0 {
                continue;
            }
            highest_end = highest_end.max(segment.offset as u64 + segment.size as u64);
        }

        // The declared memory covers the runtime's own heap/stack beyond the
        // data; the image only needs to reach the last initialized byte, but
        // taking the declared size too keeps `memory.size`-derived arithmetic
        // (and any read inside BSS) meaningful.
        let mut size = match declared {
            Some(declared) if declared <= MAX_IMAGE_BYTES => declared.max(highest_end),
            _ => highest_end,
        };
        size = size.saturating_add(WASM_PAGE_SIZE);
        size = size.div_ceil(WASM_PAGE_SIZE) * WASM_PAGE_SIZE;
        if size > MAX_IMAGE_BYTES {
            return Err(Error::InvalidFormat(format!(
                "Wasm linear memory of {size} bytes exceeds the {MAX_IMAGE_BYTES}-byte limit"
            )));
        }
        if size == 0 {
            return Err(Error::InvalidFormat("Wasm image is empty".into()));
        }

        let file = self.stream.data().to_vec();
        let mut image = vec![0u8; size as usize];
        let mut copied = 0u64;
        for segment in &self.data_segments {
            if segment.offset < 0 {
                continue;
            }
            let start = segment.offset as usize;
            let end = start.saturating_add(segment.size);
            let src = segment.data_offset as usize;
            if src + segment.size > file.len() || end > image.len() {
                continue;
            }
            image[start..end].copy_from_slice(&file[src..src + segment.size]);
            copied += segment.size as u64;
        }

        self.initialized_bytes = copied;
        self.hole_bytes = highest_end.saturating_sub(copied);
        self.stream = BinaryStream::new(image);
        self.stream.is_32bit = true;
        Ok(())
    }

    fn read_leb128_unsigned(&mut self) -> Result<usize> {
        let mut result = 0usize;
        let mut shift = 0;
        loop {
            let byte = self.stream.read_u8()?;
            result |= ((byte & 0x7F) as usize) << shift;
            if (byte & 0x80) == 0 {
                break;
            }
            shift += 7;
        }
        Ok(result)
    }

    fn read_leb128_signed(&mut self) -> Result<i64> {
        let mut result = 0i64;
        let mut shift = 0;
        let mut byte;
        loop {
            byte = self.stream.read_u8()?;
            result |= ((byte & 0x7F) as i64) << shift;
            shift += 7;
            if (byte & 0x80) == 0 {
                break;
            }
        }
        if shift < 64 && (byte & 0x40) != 0 {
            result |= !0i64 << shift;
        }
        Ok(result)
    }

    /// Address == offset in the materialized image.
    pub fn map_vatr(&self, addr: u64) -> Result<u64> {
        if addr < self.stream.len() {
            Ok(addr)
        } else {
            Err(Error::AddressNotMapped(addr))
        }
    }

    pub fn map_rtva(&self, offset: u64) -> u64 {
        offset
    }

    /// The whole linear memory is the data area the search may consider.
    pub fn data_search_sections(&self) -> Vec<SearchSection> {
        let len = self.stream.len();
        vec![SearchSection::new(0, len, 0, len)]
    }

    pub fn get_section_helper(
        &self,
        method_count: usize,
        type_definitions_count: usize,
        metadata_usages_count: usize,
        image_count: usize,
        version: f64,
    ) -> SectionHelper<'_> {
        // No executable sections: Wasm code lives in the code section as
        // bytecode and is never present in the linear memory, so a code
        // pointer search has nothing to look at (and must not be pointed at
        // the image, where code offsets mean nothing).
        let exec_list: Vec<SearchSection> = Vec::new();
        let data_list = self.data_search_sections();
        let all = data_list.clone();
        let bss = data_list.clone();

        SectionHelper::new(
            self.stream.data(),
            self.is_32bit,
            version,
            false,
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

    pub fn get_rva(&self, pointer: u64) -> u64 {
        pointer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal `.wasm` whose data section holds `(memory address, bytes)`
    /// segments, with the file-level padding real Unity builds have.
    fn wasm_with_segments(memory_pages: u32, segments: &[(u32, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&WASM_MAGIC.to_le_bytes());
        out.extend_from_slice(&WASM_VERSION.to_le_bytes());

        // memory section
        let mut mem = Vec::new();
        mem.push(1u8); // one memory
        mem.push(0x01); // has max
        leb(&mut mem, memory_pages as u64);
        leb(&mut mem, memory_pages as u64);
        out.push(5);
        leb(&mut out, mem.len() as u64);
        out.extend_from_slice(&mem);

        // data section
        let mut data = Vec::new();
        leb(&mut data, segments.len() as u64);
        for (addr, bytes) in segments {
            data.push(0u8); // active, memory 0
            data.push(0x41); // i32.const
            leb_signed(&mut data, *addr as i64);
            data.push(0x0B); // end
            leb(&mut data, bytes.len() as u64);
            data.extend_from_slice(bytes);
        }
        out.push(11);
        leb(&mut out, data.len() as u64);
        out.extend_from_slice(&data);
        out
    }

    /// Signed LEB128, as an `i32.const` operand is encoded.
    fn leb_signed(out: &mut Vec<u8>, value: i64) {
        let mut value = value;
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
            out.push(if done { byte } else { byte | 0x80 });
            if done {
                break;
            }
        }
    }

    fn leb(out: &mut Vec<u8>, mut value: u64) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    /// A pointer array split across two segments with a gap between them is
    /// read from the *memory image*, so a struct spanning the gap sees the
    /// zeros the runtime sees instead of file padding.
    #[test]
    fn segments_are_materialized_into_linear_memory() {
        let mut first = vec![0u8; 16];
        first[0..4].copy_from_slice(&0x1000u32.to_le_bytes());
        // second segment starts 8 bytes after the first ends: a gap in memory
        let second = 0xDEADBEEFu32.to_le_bytes().to_vec();
        let file = wasm_with_segments(1, &[(0x100, first), (0x110 + 8, second.clone())]);
        let wasm = Wasm::new(file).unwrap();

        // address == offset, and the gap reads as zeros
        assert_eq!(wasm.map_vatr(0x100).unwrap(), 0x100);
        assert_eq!(
            wasm.stream.peek_bytes_at(0x100, 4).unwrap(),
            0x1000u32.to_le_bytes()
        );
        assert_eq!(wasm.stream.peek_bytes_at(0x110, 8).unwrap(), [0u8; 8]);
        assert_eq!(
            wasm.stream.peek_bytes_at(0x118, 4).unwrap(),
            second.as_slice(),
            "the second segment lands at its memory address, not its file offset"
        );
    }

    /// The search must see the whole linear memory as one region.
    #[test]
    fn search_region_is_the_whole_memory() {
        let file = wasm_with_segments(1, &[(0x40, b"mscorlib.dll\0".to_vec())]);
        let wasm = Wasm::new(file).unwrap();
        let sections = wasm.data_search_sections();
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].address, 0);
        assert_eq!(sections[0].offset_end, wasm.stream.len() as u64);
        // the string is visible to a scan of the image
        assert_eq!(wasm.initialized_bytes, 13);
        assert!(
            wasm.stream
                .peek_bytes_at(0x40, 13)
                .unwrap()
                .starts_with(b"mscorlib.dll")
        );
    }
}
