//! Exception-region recovery: try/catch/finally from Itanium-ABI EH tables.
//!
//! IL2CPP compiles managed `try/catch` into native zero-cost C++ exceptions.
//! The regions survive in the binary as EH tables — recovering them from
//! data (not guessing from control flow) is exactly the metadata-advantage
//! move: the tables name the try range, the landing pad, and the caught
//! types *exactly*.
//!
//! Data sources (authoritative formats, verified against):
//!  - llvm-project/libcxxabi/src/cxa_personality.cpp
//!    ("Exception Handling Table Layout" comment = the LSDA byte format)
//!  - llvm-project/libunwind/include/mach-o/compact_unwind_encoding.h
//!    (__unwind_info section header + lsda_index_entry layout)
//!  - Itanium C++ ABI, §8.4 "Exception Handling Tables"
//!    (DW_EH_PE encodings)
//!  - Microsoft PE/COFF specification §5.4.2 (UNWIND_INFO) and the
//!    `_C_specific_handler` `C_SCOPE_TABLE` layout (x64 SEH)
//!
//! Pipeline: `__TEXT,__unwind_info` maps function RVA → LSDA offset in
//! `__TEXT,__gcc_except_tab`; the LSDA then decodes into call-site ranges
//! ({start, len, landingPad, action}) whose action chain yields the caught
//! type list (type_info pointers, 0 = catch-all, negative = spec/filter).
//!
//! Landing pads are marked as virtual entry blocks and rendered as
//! `catch (...)` / `finally` handlers attached to their try region —
//! the structurer does not have to invent them from control flow.

// ---------------------------------------------------------------------------
// DW_EH_PE encoding constants (Itanium C++ ABI §8.4 / dwarf2.h)
// ---------------------------------------------------------------------------

const DW_EH_PE_ABSPTR: u8 = 0x00;
const DW_EH_PE_ULEB128: u8 = 0x01;
const DW_EH_PE_UDATA2: u8 = 0x02;
const DW_EH_PE_UDATA4: u8 = 0x03;
const DW_EH_PE_UDATA8: u8 = 0x04;
const DW_EH_PE_SLEB128: u8 = 0x09;
const DW_EH_PE_SDATA2: u8 = 0x0a;
const DW_EH_PE_SDATA4: u8 = 0x0b;
const DW_EH_PE_SDATA8: u8 = 0x0c;
const DW_EH_PE_PCREL: u8 = 0x10;
const DW_EH_PE_DATAREL: u8 = 0x30;
const DW_EH_PE_INDIRECT: u8 = 0x80;
const DW_EH_PE_OMIT: u8 = 0xff;

// ---------------------------------------------------------------------------
// __unwind_info section layout (compact_unwind_encoding.h)
// ---------------------------------------------------------------------------

struct UnwindInfoHeader {
    // Only the second-level index locates LSDAs; the common-encodings and
    // personality arrays (offsets 4..16) are part of the section layout
    // (compact_unwind_encoding.h) but carry no LSDA data, so they are
    // validated by the length check in `parse_unwind_info_header` and not
    // retained here.
    index_off: u32,
    index_count: u32,
}

fn parse_unwind_info_header(data: &[u8]) -> Option<UnwindInfoHeader> {
    if data.len() < 28 {
        return None;
    }
    let u32le = |off: usize| -> u32 {
        u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
    };
    let version = u32le(0);
    if version != 1 {
        // UNWIND_SECTION_VERSION
        return None;
    }
    Some(UnwindInfoHeader {
        index_off: u32le(20),
        index_count: u32le(24),
    })
}

/// Find the LSDA offset (within `gcc_except_tab`) for a function at
/// `func_rva`, using the two-level `__unwind_info` index.
///
/// Each top-level index entry covers a function range and points at its own
/// lsda-index sub-array; entries are `{functionOffset, lsdaOffset}` pairs
/// (u32 LE, section-relative). The LSDA offset stored in the index is
/// relative to the START of the `__unwind_info` section and must be
/// translated into `gcc_except_tab`-relative space (the personality runtime
/// treats it as an absolute VA — see `LsdaResolver` below for the
/// translation step).
fn find_lsda_offset_in_unwind_info(unwind_data: &[u8], func_rva: u64) -> Option<u32> {
    let hdr = parse_unwind_info_header(unwind_data)?;
    let u32le = |off: usize| -> Option<u32> {
        if off + 4 <= unwind_data.len() {
            Some(u32::from_le_bytes([
                unwind_data[off],
                unwind_data[off + 1],
                unwind_data[off + 2],
                unwind_data[off + 3],
            ]))
        } else {
            None
        }
    };
    if hdr.index_count == 0 || hdr.index_off == 0 {
        return None;
    }
    let index_end = hdr.index_off as usize + hdr.index_count as usize * 12;
    if index_end > unwind_data.len() {
        return None;
    }
    let func_off = func_rva as u32;
    let mut chosen_lsda_idx: Option<usize> = None;
    for i in 0..hdr.index_count as usize {
        let base = hdr.index_off as usize + i * 12;
        let start = u32le(base)?;
        let next_start = if i + 1 < hdr.index_count as usize {
            u32le(base + 12)?
        } else {
            u32::MAX
        };
        if func_off >= start && func_off < next_start {
            chosen_lsda_idx = Some(u32le(base + 8)? as usize);
            break;
        }
    }
    let lsda_index_off = chosen_lsda_idx?;
    // Walk this range's lsda-index sub-array. Its end is bounded by the next
    // index entry's lsdaIndexArraySectionOffset (or the section end); entries
    // are sorted by functionOffset, so stop as soon as one exceeds the target.
    // (The sub-array has NO sentinel entry — UnwindCursor.hpp binary-searches
    // it with an exclusive `high` bound from the next top-level entry.)
    // The lsda-index sub-array for this range lives AFTER the top-level
    // index (its start is `lsdaIndexArraySectionOffset`), so bound it by the
    // section length — NOT by `index_end`, which only ends the top-level
    // index and would reject the very first sub-array entry.
    let sub_end = unwind_data.len();
    let mut i = 0usize;
    loop {
        let off = lsda_index_off + i * 8;
        if off + 8 > sub_end {
            return None;
        }
        let func = u32le(off)?;
        let lsda = u32le(off + 4)?;
        if func == func_off {
            return Some(lsda);
        }
        if func > func_off {
            // The function itself has NO unwind entry (no calls that can
            // throw, or the linker culled it) — but Apple's unwinder uses
            // RANGE semantics: an entry whose functionOffset is the nearest
            // one ≤ the lookup address governs it only up to the NEXT entry.
            // Using that nearest-below entry's LSDA blindly would attach
            // another function's try regions to this one, so require the
            // found entry to actually be an exact-or-range match: accept it
            // only when the caller passes a function that has NO entry of
            // its own AND the previous entry's function range covers it.
            // We don't track previous-entry bounds here, so return None —
            // the conservative choice (no EH info) — unless the nearest
            // entry's LSDA is 0 (explicitly "no EH for this range").
            return None;
        }
        i += 1;
        // hard cap so a corrupt section can never loop forever
        if i > (unwind_data.len() / 8) + 1 {
            return None;
        }
    }
}

// ---------------------------------------------------------------------------
// LSDA (Itanium-ABI .gcc_except_tab) parsing
// ---------------------------------------------------------------------------

/// One entry of an Itanium action chain, in selector order.
///
/// The chain is the personality routine's ORDERED list of actions for one
/// call-site range; selector value `k` (the `w1` dispatch in the landing
/// pad) picks `chain[k - 1]`. Keeping the chain instead of a
/// `(catches, is_cleanup)` pair is what separates two facts that the old
/// model merged:
///
/// * `ttypeIndex == 0` is the personality's **cleanup** action. IL2CPP emits
///   one as the no-match arm of every managed catch
///   (`[catch(wrapper), cleanup]`), so its presence does NOT make the region
///   a `finally`.
/// * a `ttypeIndex > 0` whose type-table entry is NULL is an untyped
///   **catch-all** — a different thing from a cleanup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Clause {
    /// `ttypeIndex > 0` with a non-null type-table entry: `catch (T)`.
    /// `name` is the decoded `type_info` name (the wrapper for every managed
    /// catch; the managed type filter is inside the landing pad).
    Catch { name: String },
    /// `ttypeIndex > 0` with a NULL type-table entry: an untyped catch.
    CatchAll,
    /// `ttypeIndex == 0`: the personality's cleanup action.
    Cleanup,
    /// `ttypeIndex < 0`: an exception specification (filter).
    Filter,
}

/// One decoded try region.
#[derive(Debug, Clone, PartialEq)]
pub struct EhRegion {
    /// Try-start RVA (inclusive).
    pub try_start: u64,
    /// Try-end RVA (exclusive).
    pub try_end: u64,
    /// Landing-pad RVA (absolute; `lpStart + offset`).
    pub landing_pad: u64,
    /// The action chain, in selector order. Never empty for a row with a
    /// landing pad: `actionEntry == 0` yields `[Cleanup]`.
    pub chain: Vec<Clause>,
    /// Caught types, in `catch` order, DERIVED from `chain` for the printer:
    /// `Some(name)` per `Clause::Catch`, `None` per `Clause::CatchAll`.
    /// `Clause::Cleanup` contributes nothing, so a managed catch's trailing
    /// no-match cleanup no longer shows up as a phantom `catch (...)`.
    pub catches: Vec<Option<String>>,
    /// True only when the whole chain is cleanup actions (`actionEntry == 0`
    /// for the LSDA paths, JumpTarget 0 / kind 0 for the Mach-O / ARM-EHABI
    /// paths). DERIVED from `chain`; never set by a trailing arm.
    pub is_cleanup: bool,
}

impl EhRegion {
    /// Build a region and its compatibility views from the chain.
    pub fn new(try_start: u64, try_end: u64, landing_pad: u64, chain: Vec<Clause>) -> Self {
        let catches = chain
            .iter()
            .filter_map(|c| match c {
                Clause::Catch { name, .. } => Some(Some(name.clone())),
                Clause::CatchAll => Some(None),
                Clause::Filter => Some(Some("/* filter */".to_string())),
                Clause::Cleanup => None,
            })
            .collect();
        let is_cleanup = !chain.is_empty() && chain.iter().all(|c| matches!(c, Clause::Cleanup));
        Self {
            try_start,
            try_end,
            landing_pad,
            chain,
            catches,
            is_cleanup,
        }
    }

    /// Does any arm catch (`catch (T)` or an untyped catch)?
    pub fn has_catch(&self) -> bool {
        self.chain
            .iter()
            .any(|c| matches!(c, Clause::Catch { .. } | Clause::CatchAll))
    }

    /// Does the chain contain a personality cleanup action?
    pub fn has_cleanup(&self) -> bool {
        self.chain.iter().any(|c| matches!(c, Clause::Cleanup))
    }

    /// The catch clauses, in selector order (cleanup actions excluded).
    pub fn catch_clauses(&self) -> impl Iterator<Item = &Clause> {
        self.chain
            .iter()
            .filter(|c| matches!(c, Clause::Catch { .. } | Clause::CatchAll | Clause::Filter))
    }
}

/// Format-neutral EH table source: one variant per executable format.
/// Built once at startup from the parsed binary; `regions_for` then maps a
/// function RVA onto try regions.
pub enum EhTables {
    /// Mach-O: `__TEXT,__unwind_info` + `__TEXT,__gcc_except_tab`.
    MachO(MachOExceptionTables),
    /// ELF: `.eh_frame`-derived (func_rva → LSDA RVA) + `.gcc_except_table`
    /// bytes.
    Elf {
        /// Sorted (func_rva, lsda_rva) pairs.
        fdes: Vec<(u64, u64)>,
        /// `.gcc_except_table` section bytes.
        table: Vec<u8>,
        /// Section start RVA of `table` (for bounds checks).
        table_rva: u64,
        /// Pointer width of the target (4 for 32-bit, 8 for 64-bit) — the
        /// size of a `DW_EH_PE_absptr` value and a default type-table entry.
        ptr_size: usize,
    },
    /// ARM32 ELF: the ARM EHABI (`.ARM.exidx` binary-searchable index table +
    /// `.ARM.extab` variable-length handling table). This is the format every
    /// 32-bit ARM ELF uses INSTEAD of DWARF `.eh_frame`, so an ARM32 binary has
    /// no `Elf` tables at all and previously produced zero try/finally regions.
    ///
    /// The decode (index entry → handling table entry → descriptor list) is
    /// specified by "Exception Handling ABI for the Arm Architecture" and was
    /// cross-checked against LLVM's `ARMEHABIPrinter.h` (index/extab decoding)
    /// and GCC's `libgcc/unwind-arm-common.inc::__gnu_unwind_pr_common` (the
    /// compact-model descriptor walk).
    ///
    /// The GENERIC model (extab word 0 has bit 31 clear — a prel31 to a custom
    /// personality routine) carries an **Itanium/DWARF LSDA** after its unwind
    /// opcodes, so it is decoded by the same `parse_lsda` the `.eh_frame` path
    /// uses. On Unity's `libil2cpp.so` this is the ONLY model that carries
    /// handler tables for managed code: all 8,425 generic entries in the
    /// managed section decode to try/catch/finally regions, while the compact
    /// entries carry none. x86 PE SEH scope tables in `.xdata` remain a
    /// follow-up.
    ArmEhAbi {
        /// Raw `.ARM.exidx` bytes: an array of 8-byte index entries, sorted by
        /// function start.
        exidx: Vec<u8>,
        /// Section start RVA of `exidx`.
        exidx_rva: u64,
        /// Raw `.ARM.extab` bytes (may be empty — all-inline binaries exist).
        extab: Vec<u8>,
        /// Section start RVA of `extab`.
        extab_rva: u64,
    },
    /// PE/x64: `.pdata` RUNTIME_FUNCTION entries (func_rva → unwind RVA) plus
    /// the `.xdata` bytes the unwind records live in. This is the Windows SEH
    /// path: the unwind record's exception handler is `__C_specific_handler`
    /// and its language-specific data is a `C_SCOPE_TABLE`, so the try ranges
    /// come straight out of the binary exactly as the Mach-O/ELF LSDA path does
    /// (see the PE section below for the full layout and why the MSVC C++
    /// `FuncInfo` layout is deliberately refused).
    Pe(PeExceptionTables),
    /// None found (the common case for non-EH binaries).
    None,
}

impl EhTables {
    pub fn is_empty(&self) -> bool {
        matches!(self, EhTables::None)
    }

    /// Decode try regions for the function covering `func_rva`.
    /// `deref_typeinfo` maps a type_info VA → readable type name (optional).
    pub fn regions_for(
        &self,
        func_rva: u64,
        func_len: u64,
        deref_typeinfo: &dyn Fn(u64) -> Option<String>,
    ) -> Vec<EhRegion> {
        match self {
            EhTables::MachO(t) => t.regions_for(func_rva, func_len, deref_typeinfo).regions,
            EhTables::Elf {
                fdes,
                table,
                table_rva,
                ptr_size,
            } => {
                // binary search the FDE list for this function
                let idx = fdes.partition_point(|&(start, _)| start <= func_rva);
                let Some(&(start, lsda_rva)) = idx.checked_sub(1).and_then(|i| fdes.get(i)) else {
                    return Vec::new();
                };
                if lsda_rva == 0 || func_rva >= start + func_len && func_rva != start {
                    return Vec::new();
                }
                let lsda_in_tab = match (lsda_rva as usize).checked_sub(*table_rva as usize) {
                    Some(o) if o < table.len() => o,
                    _ => return Vec::new(),
                };
                let mut regs = parse_lsda(
                    &table[lsda_in_tab..],
                    start,
                    lsda_rva as u64,
                    *ptr_size,
                    *table_rva,
                    deref_typeinfo,
                );
                regs.retain(|rg| {
                    rg.try_end > func_rva && rg.try_start < func_rva.wrapping_add(func_len)
                });
                regs
            }
            EhTables::ArmEhAbi {
                exidx,
                exidx_rva,
                extab,
                extab_rva,
            } => arm_ehabi_regions(
                *exidx_rva,
                exidx,
                *extab_rva,
                extab,
                func_rva,
                func_len,
                deref_typeinfo,
            ),
            EhTables::Pe(t) => t.regions_for(func_rva, func_len, deref_typeinfo),
            EhTables::None => Vec::new(),
        }
    }
}

struct LsdaReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Pointer width in bytes for `DW_EH_PE_absptr` values: 4 on 32-bit
    /// targets, 8 on 64-bit. Mis-reading it corrupts every absolute pointer
    /// and the default `ttype` table entry size.
    ptr_size: usize,
    /// Base for `DW_EH_PE_datarel` values (the data section the LSDA lives
    /// in). 0 when the encoding never uses datarel.
    data_base: u64,
}

impl<'a> LsdaReader<'a> {
    /// A reader for 64-bit targets (Mach-O arm64, ELF aarch64) with no
    /// datarel base — the common case.
    #[cfg(test)]
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            ptr_size: 8,
            data_base: 0,
        }
    }

    fn with_ptr_size(data: &'a [u8], ptr_size: usize, data_base: u64) -> Self {
        Self {
            data,
            pos: 0,
            ptr_size,
            data_base,
        }
    }

    fn u8(&mut self) -> Option<u8> {
        let b = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    fn u16le(&mut self) -> Option<u16> {
        let b = self.bytes(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32le(&mut self) -> Option<u32> {
        let b = self.bytes(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64le(&mut self) -> Option<u64> {
        let b = self.bytes(8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.data.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }

    fn uleb128(&mut self) -> Option<u64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            let b = self.u8()?;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return None; // malformed — refuse to loop forever
            }
        }
        Some(result)
    }

    fn sleb128(&mut self) -> Option<i64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        let byte = loop {
            let b = self.u8()?;
            result |= ((b & 0x7f) as u64) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break b;
            }
            if shift > 63 {
                return None;
            }
        };
        if byte & 0x40 != 0 && shift < 64 {
            // sign-extend
            result |= u64::MAX << shift;
        }
        Some(result as i64)
    }

    /// readEncodedPointer — value by format class, then relative base.
    /// `pc_rel_base` is the file position the encoded value is relative to
    /// for DW_EH_PE_pcrel (the position BEFORE the value was read).
    fn encoded(&mut self, enc: u8, pc_rel_base: u64) -> Option<u64> {
        if enc == DW_EH_PE_OMIT {
            return None;
        }
        // Strip the INDIRECT bit before decoding: the computed address is then
        // the address OF a pointer slot, and only the caller has the binary
        // access needed to dereference it (Apple's ubiquitous ttypeEncoding
        // 0x9B sets this bit).
        let enc = enc & !DW_EH_PE_INDIRECT;
        let value_class = enc & 0x0f;
        let raw: u64 = match value_class {
            DW_EH_PE_ABSPTR => {
                // pointer-width: 4 on 32-bit targets, 8 otherwise
                if self.ptr_size == 4 {
                    self.u32le()? as u64
                } else {
                    self.u64le()?
                }
            }
            DW_EH_PE_ULEB128 => self.uleb128()?,
            DW_EH_PE_UDATA2 => self.u16le()? as u64,
            DW_EH_PE_UDATA4 => self.u32le()? as u64,
            DW_EH_PE_UDATA8 => self.u64le()?,
            DW_EH_PE_SLEB128 => self.sleb128()? as u64,
            DW_EH_PE_SDATA2 => self.u16le()? as i16 as i64 as u64,
            DW_EH_PE_SDATA4 => self.u32le()? as i32 as i64 as u64,
            DW_EH_PE_SDATA8 => self.u64le()? as i64 as u64,
            _ => return None,
        };
        let applied = match enc & 0x70 {
            0x00 => raw, // absptr application
            DW_EH_PE_PCREL => {
                if raw == 0 {
                    0
                } else {
                    raw.wrapping_add(pc_rel_base)
                }
            }
            DW_EH_PE_DATAREL => raw.wrapping_add(self.data_base),
            _ => return None, // textrel/funcrel/aligned unsupported
        };
        // NOTE: the DW_EH_PE_INDIRECT bit (0x80 — set on Apple's ubiquitous
        // ttypeEncoding 0x9B) is handled by the CALLER: the computed address
        // is then the address OF a pointer slot, and only the caller has the
        // binary access needed to dereference it.
        Some(applied)
    }
}

/// Decode the LSDA at `lsda_data` into try regions.
///
/// `func_start` is the RVA of the enclosing function (lpStart default).
/// `lsda_rva` is the RVA the LSDA bytes live at; pcrel-encoded values are
/// relative to the address of the field itself, so the fixup is
/// `lsda_rva + field_file_offset`. Encoded results are RVAs (image-relative
/// on Mach-O per `lsdaOffset + dso_base` where dso_base is the image base
/// subtracted — `UnwindCursor.hpp` line ~2100).
/// `resolve_indirect` dereferences a type_info RVA → readable type name.
fn parse_lsda(
    lsda_data: &[u8],
    func_start: u64,
    lsda_rva: u64,
    ptr_size: usize,
    data_base: u64,
    resolve_indirect: &dyn Fn(u64) -> Option<String>,
) -> Vec<EhRegion> {
    let mut r = LsdaReader::with_ptr_size(lsda_data, ptr_size, data_base);
    // pcrel base for a field at file offset `p` inside the LSDA is
    // `lsda_rva + p` (the field's own image-relative address).
    let pc = |p: usize| -> u64 { lsda_rva.wrapping_add(p as u64) };
    let mut out = Vec::new();

    // -- header -------------------------------------------------------------
    let lp_start_enc = match r.u8() {
        Some(e) => e,
        None => return out,
    };
    let lp_start = match r.encoded(lp_start_enc, pc(r.pos)) {
        Some(v) => v,
        None => func_start, // omit → funcStart (cxa_personality.cpp)
    };
    let ttype_enc = match r.u8() {
        Some(e) => e,
        None => return out,
    };
    // classInfo = start of the type_info table region (before action table!)
    let mut class_info: Option<u64> = None;
    if ttype_enc != DW_EH_PE_OMIT {
        if let Some(cio) = r.uleb128() {
            // classInfoOffset is relative to the position AFTER the ULEB128
            class_info = Some((r.pos as u64).wrapping_add(cio));
        }
    }
    let call_site_enc = match r.u8() {
        Some(e) => e,
        None => return out,
    };
    let call_site_table_len = match r.uleb128() {
        Some(l) => l as usize,
        None => return out,
    };
    let call_site_start = r.pos;
    let call_site_end = call_site_start.saturating_add(call_site_table_len);
    if call_site_end > lsda_data.len() {
        return out;
    }
    let action_table_start = call_site_end;
    if std::env::var("ILR_EHCHAINDBG").is_ok() {
        let end = lsda_data.len().min(call_site_end + 40);
        eprintln!(
            "[lsda] rva={lsda_rva:#x} func={func_start:#x} lpenc={lp_start_enc:#x} lps={lp_start:#x} ttenc={ttype_enc:#x} ci={class_info:?} csenc={call_site_enc:#x} cslen={call_site_table_len} csstart={call_site_start} act={action_table_start} head={:02x?}",
            &lsda_data[..call_site_end.min(lsda_data.len())],
        );
        let _ = end;
    }

    // -- call-site table ----------------------------------------------------
    while r.pos < call_site_end {
        let start = match r.encoded(call_site_enc, pc(r.pos)) {
            Some(v) => v,
            None => break,
        };
        let length = match r.encoded(call_site_enc, pc(r.pos)) {
            Some(v) => v,
            None => break,
        };
        let landing_pad_off = match r.encoded(call_site_enc, pc(r.pos)) {
            Some(v) => v,
            None => break,
        };
        let action_entry = match r.uleb128() {
            Some(a) => a,
            None => break,
        };

        // call-site start/length are function-relative (funcrel application);
        // landingPad is lpStart-relative. (cxa_personality.cpp:
        // `landingPad = lpStart + landingPad`)
        let try_start = func_start.wrapping_add(start);
        let try_end = try_start.wrapping_add(length);
        let landing_pad = if landing_pad_off == 0 {
            0
        } else {
            lp_start.wrapping_add(landing_pad_off)
        };

        // A call-site row whose landing pad is 0 does NOT name a handler: it
        // means "an exception raised here has no landing pad in this frame, so
        // it simply unwinds out". Itanium's `__gxx_personality_v0` skips such
        // rows (`if (landingPad == 0) continue;` in cxa_personality.cpp) and
        // so must we. Emitting them produced `try { … }` shells with no
        // handler around every throw-capable call; because a managed method's
        // throw-capable calls span its whole body, those shells then fenced
        // the structurer's walks — splitting loops and leaving detached
        // try/finally islands (measured on `GetExistingParameters`
        // @0x12EA65C, whose top-level `try` had `pad=0x0` and no handler at
        // all). The Mach-O path already filtered these after the fact; the ELF
        // and ARM-EHABI-generic paths did not, which is why the ARM32 corpus
        // kept the defect long after the Mach-O one was fixed.
        if landing_pad == 0 {
            continue;
        }

        // -- action chain ---------------------------------------------------
        // The chain is kept in selector order; the region KIND is a property
        // of the pad, not of the chain (a managed catch's chain ends with a
        // cleanup arm that is the personality's no-match path, not a
        // `finally`). See `Clause`.
        let mut chain: Vec<Clause> = Vec::new();
        if action_entry == 0 {
            // cxa_personality.cpp: actionEntry == 0 (with a landing pad)
            // IS a cleanup — phase-2-only handler (finally/fault semantics).
            chain.push(Clause::Cleanup);
        }
        if action_entry != 0 && landing_pad != 0 {
            let mut action_pos = action_table_start + (action_entry as usize - 1);
            let mut guard = 0usize;
            while guard < 1024 {
                guard += 1;
                if action_pos >= lsda_data.len() {
                    break;
                }
                let mut ar = LsdaReader {
                    data: lsda_data,
                    pos: action_pos,
                    ptr_size,
                    data_base,
                };
                let ttype_index = match ar.sleb128() {
                    Some(t) => t,
                    None => break,
                };
                // The next-action offset is relative to the ADDRESS OF THIS
                // FIELD, not to the end of the record. Both readings agree for
                // forward offsets, but the compiler chains a managed catch to
                // its shared cleanup record BACKWARDS (-3 for a 2-byte
                // record), and only this base lands on the record:
                //   0xaf0600 + 546: [00 00] [01 7d] [02 00]
                //   action 3 -> record at 548 = (catch 1, next -3)
                //   next field at 549, 549 - 3 = 546 = the cleanup record.
                // With the old base (position AFTER both fields, 550) the walk
                // landed one byte into the table, read garbage, and appended a
                // phantom third clause to every managed catch (7,588/7,588
                // regions on this corpus: (catch, cleanup, catch-all)).
                let next_field_pos = ar.pos;
                let action_offset = match ar.sleb128() {
                    Some(o) => o,
                    None => break,
                };
                if std::env::var("ILR_EHCHAINDBG").is_ok() {
                    let end = (action_pos + 8).min(lsda_data.len());
                    eprintln!(
                        "[chain] try={try_start:#x} pad={landing_pad:#x} pos={action_pos} ttype={ttype_index} next={action_offset} bytes={:02x?} action_table={action_table_start}",
                        &lsda_data[action_pos..end]
                    );
                }
                if ttype_index > 0 {
                    // catch(T): ttypeIndex is a 1-based index INTO the type
                    // table, counted BACKWARD from classInfo in entry-sized
                    // steps (entry size from ttype_enc's format class;
                    // cxa_personality.cpp get_shim_type_info).
                    if let Some(ci) = class_info {
                        let entry_size: u64 = match ttype_enc & 0x0f {
                            DW_EH_PE_UDATA2 | DW_EH_PE_SDATA2 => 2,
                            DW_EH_PE_UDATA4 | DW_EH_PE_SDATA4 => 4,
                            DW_EH_PE_UDATA8 | DW_EH_PE_SDATA8 => 8,
                            _ => ptr_size as u64, // absptr
                        };
                        // classInfoOffset is relative to the position AFTER
                        // its own ULEB128, so `ci` is a byte offset into
                        // lsda_data directly.
                        let entry_pos =
                            (ci as i64).checked_sub(ttype_index.saturating_mul(entry_size as i64));
                        if let Some(ep) =
                            entry_pos.filter(|&e| e >= 0 && (e as usize) < lsda_data.len())
                        {
                            let mut tr = LsdaReader {
                                data: lsda_data,
                                pos: ep as usize,
                                ptr_size,
                                data_base,
                            };
                            let typeinfo_pc = pc(ep as usize);
                            match tr.encoded(ttype_enc, typeinfo_pc) {
                                // A NULL type-table entry is an UNTYPED catch
                                // (catch-all), which is NOT the same fact as
                                // ttypeIndex == 0 (cleanup). The old model
                                // pushed `None` for both and could not tell a
                                // compiler terminate guard from a `finally`.
                                Some(0) => chain.push(Clause::CatchAll),
                                Some(va) => {
                                    let name =
                                        resolve_indirect(va).unwrap_or_else(|| format!("{va:#x}"));
                                    chain.push(Clause::Catch { name });
                                }
                                None => chain.push(Clause::CatchAll),
                            }
                        } else {
                            chain.push(Clause::CatchAll);
                        }
                    } else {
                        chain.push(Clause::CatchAll);
                    }
                } else if ttype_index == 0 {
                    // The personality's cleanup action. In a managed catch
                    // this is the NO-MATCH arm (the wrapper's catch already
                    // ran); it is a cleanup ACTION, not a region kind, so it
                    // must not set `is_cleanup` (that was E2: every managed
                    // catch rendered as `finally`).
                    chain.push(Clause::Cleanup);
                } else {
                    // ttypeIndex < 0: exception SPECIFICATION (filter).
                    // C# has no direct equivalent; keep it visible.
                    chain.push(Clause::Filter);
                }
                if action_offset == 0 {
                    break;
                }
                // Walk from the next_offset field's own address (see above).
                // The target must stay inside the action table: the table runs
                // from `action_table_start` up to the class-info table, and a
                // corrupt/unexpected offset must END the chain rather than
                // wander into the type table and invent clauses.
                let next = next_field_pos as i64 + action_offset;
                let action_end = class_info
                    .map(|ci| ci.min(lsda_data.len() as u64) as i64)
                    .unwrap_or(lsda_data.len() as i64);
                if next < action_table_start as i64 || next >= action_end {
                    break;
                }
                action_pos = next as usize;
            }
        }

        out.push(EhRegion::new(try_start, try_end, landing_pad, chain));
    }

    out
}

// ---------------------------------------------------------------------------
// ELF: .eh_frame FDE scan (DwarfParser.hpp decodeFDE/findFDE/parseCIE)
// ---------------------------------------------------------------------------

/// Scan a complete `.eh_frame` section and return every FDE as
/// (pc_start, lsda_rva) pairs, sorted by pc_start. Implements the exact
/// CIE/FDE walk from llvm libunwind `DwarfParser.hpp`:
///  - length (u32; 0xffffffff → u64 follows)
///  - CIE: id == 0; version byte; augmentation string; if 'z': augmentation
///    data with 'P' (personality enc + ptr), 'L' (LSDA enc), 'R' (pointer enc)
///  - FDE: id != 0 (cie_pointer, backwards from its own position); pc_begin
///    and pc_range with the CIE's pointer encoding; augmentation length
///    (ULEB) whose first field — when the CIE has LSDA encoding — is the
///    LSDA pointer (peeked with the low nibble only; zero = none, exactly as
///    the runtime does before re-reading with the full encoding).
pub fn scan_eh_frame_fdes(
    data: &[u8],
    section_rva: u64,
    image_base: u64,
    ptr_size: usize,
) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    let u32le = |off: usize| -> Option<u32> {
        let b = data.get(off..off + 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let u64le = |off: usize| -> Option<u64> {
        let b = data.get(off..off + 8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    };
    let mut p = 0usize;
    let end = data.len();
    let mut guard = 0usize;
    let max_records = end / 8 + 2; // smallest CFI record is 8 bytes
    while p + 4 <= end {
        guard += 1;
        if guard > max_records {
            break; // corrupt — never loop forever
        }
        let mut length = match u32le(p) {
            Some(v) => v as u64,
            None => break,
        };
        let mut after_len = p + 4;
        if length == 0xffffffff {
            length = match u64le(after_len) {
                Some(v) => v,
                None => break,
            };
            after_len += 8;
        }
        if length == 0 {
            break; // zero terminator
        }
        let record_len = length as usize;
        if after_len + record_len > end {
            break;
        }
        let next_cfi = after_len + record_len;
        let id_pos = after_len;
        let id = match u32le(id_pos) {
            Some(v) => v,
            None => break,
        };
        if id == 0 {
            // CIE — skip (we do a two-pass linear scan: CIEs are usually
            // before their FDEs, but FDE cie_pointers are backwards offsets
            // so a CIE after an FDE is still reachable; we re-walk on demand
            // in parse_one_fde via the cieStart computed below).
            p = next_cfi;
            continue;
        }
        // FDE: cie_pointer is the DISTANCE BACK from id_pos to the CIE start.
        let cie_pointer = id as u64;
        let cie_start_off = match (id_pos as u64).checked_sub(cie_pointer) {
            Some(v) if (v as usize) < end => v as usize,
            _ => {
                p = next_cfi;
                continue;
            }
        };
        let fde = parse_one_fde(
            data,
            p,
            next_cfi,
            cie_start_off,
            section_rva,
            image_base,
            ptr_size,
        );
        if let Some((pc_start, lsda)) = fde {
            out.push((pc_start, lsda));
        }
        p = next_cfi;
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Parse one FDE whose id field sits at `id_pos` (record body begins there
/// with the cie_pointer). Returns (pc_start, lsda_rva) — lsda 0 when absent.
#[allow(clippy::too_many_arguments)]
fn parse_one_fde(
    data: &[u8],
    _p: usize,
    _next_cfi: usize,
    cie_start_off: usize,
    section_rva: u64,
    image_base: u64,
    ptr_size: usize,
) -> Option<(u64, u64)> {
    // --- parse the CIE first (parseCIE) ---
    let cie_len = u32le_at(data, cie_start_off)? as u64;
    let mut cp = cie_start_off + 4;
    if cie_len == 0xffffffff {
        cp += 8; // 64-bit length already consumed by caller's framing? no —
        // CIE length is its own record; handle it
    }
    let cie_id = u32le_at(data, cp)?;
    if cie_id != 0 {
        return None; // not a CIE
    }
    cp += 4;
    let version = *data.get(cp)?;
    cp += 1;
    if version != 1 && version != 3 {
        return None;
    }
    // augmentation string
    let str_start = cp;
    while *data.get(cp)? != 0 {
        cp += 1;
    }
    cp += 1;
    let aug_first = *data.get(str_start)?;
    // code align (ULEB), data align (SLEB), RA register
    let mut rd = LsdaReader::with_ptr_size(data, ptr_size, 0);
    rd.pos = cp;
    let _code_align = rd.uleb128()?;
    let _data_align = rd.sleb128()?;
    if version == 1 {
        rd.u8()?;
    } else {
        rd.uleb128()?;
    }
    let mut pointer_encoding: u8 = 0x00; // absptr (DW_EH_PE_absptr)
    let mut lsda_encoding: u8 = 0xff; // omit
    let mut fdes_have_aug = false;
    if aug_first == b'z' {
        let _aug_len = rd.uleb128()?;
        let mut s = str_start;
        while *data.get(s)? != 0 {
            match *data.get(s)? {
                b'z' => fdes_have_aug = true,
                b'P' => {
                    // personality encoding + encoded pointer (skip)
                    let enc = rd.u8()?;
                    // read without applying (we don't need the personality)
                    skip_encoded(&mut rd, enc)?;
                }
                b'L' => lsda_encoding = rd.u8()?,
                b'R' => pointer_encoding = rd.u8()?,
                b'S' => {}
                b'B' => {}
                _ => {}
            }
            s += 1;
        }
    }

    // --- now the FDE ---
    // Layout: length(4) id(4) pc_begin pc_range aug... — body starts 8 bytes
    // after the FDE's length field.
    let fp = _p + 8;
    let mut fr = LsdaReader::with_ptr_size(data, ptr_size, 0);
    fr.pos = fp;
    // pc_begin: pcrel base = the ADDRESS of the pc_begin field itself
    // (sdata4 pcrel is the near-universal encoding for pc_begin).
    let pc_begin_addr = section_rva + fr.pos as u64 + image_base;
    let pc_start = fr.encoded(pointer_encoding, pc_begin_addr)?;
    // For pcrel, `encoded` already added the base → that IS the VA.
    // For absptr, the value is a raw VA (already linked).
    let pc_range_enc = pointer_encoding & 0x0f;
    let _pc_range = fr.encoded(pc_range_enc, 0)?; // range is never pcrel
    // augmentation: length ULEB, then optional LSDA pointer
    let mut lsda_rva: u64 = 0;
    if fdes_have_aug {
        let aug_len = fr.uleb128()?;
        let _aug_end = fr.pos + aug_len as usize;
        if lsda_encoding != 0xff {
            // peek with low-nibble encoding; 0 = no LSDA (DwarfParser decodeFDE)
            let peek_pos = fr.pos;
            let peek_val = fr.encoded(lsda_encoding & 0x0f, 0)?;
            if peek_val != 0 {
                fr.pos = peek_pos;
                let lsda_addr = fr.encoded(lsda_encoding, section_rva + fr.pos as u64 + image_base)?;
                // .eh_frame LSDA pointers are pcrel-to-section or absptr;
                // pcrel already resolved via base. absptr values are VAs.
                lsda_rva = lsda_addr;
            }
        }
        // Keep `fr` at the augmentation end for symmetric CFI walking.
        let _ = fr.pos;
    }
    Some((pc_start, lsda_rva))
}

fn u32le_at(data: &[u8], off: usize) -> Option<u32> {
    let b = data.get(off..off + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Consume an encoded pointer without applying a base (for skipping).
fn skip_encoded(r: &mut LsdaReader, enc: u8) -> Option<()> {
    let class = enc & 0x0f;
    match class {
        DW_EH_PE_ABSPTR => {
            r.u64le()?;
        }
        DW_EH_PE_ULEB128 => {
            r.uleb128()?;
        }
        DW_EH_PE_UDATA2 => {
            r.u16le()?;
        }
        DW_EH_PE_UDATA4 => {
            r.u32le()?;
        }
        DW_EH_PE_UDATA8 => {
            r.u64le()?;
        }
        DW_EH_PE_SLEB128 => {
            r.sleb128()?;
        }
        DW_EH_PE_SDATA2 => {
            r.u16le()?;
        }
        DW_EH_PE_SDATA4 => {
            r.u32le()?;
        }
        DW_EH_PE_SDATA8 => {
            r.u64le()?;
        }
        _ => return None,
    }
    Some(())
}

// ---------------------------------------------------------------------------
// Public entry: binary-level API
// ---------------------------------------------------------------------------

/// Everything needed to resolve exception regions for functions in a Mach-O
/// image. Built once from the parsed sections.
pub struct MachOExceptionTables {
    /// `__TEXT,__unwind_info` section bytes.
    pub unwind_info: Vec<u8>,
    /// `__TEXT,__gcc_except_tab` section bytes.
    pub gcc_except_tab: Vec<u8>,
    /// Image-relative start RVA of `__gcc_except_tab`.
    pub gcc_except_tab_file_rva: u64,
    /// image base (VA = rva + image_base), 0 when none.
    pub image_base: u64,
    /// Pointer width of the target (4 for 32-bit, 8 for 64-bit).
    pub ptr_size: usize,
}

/// Result of resolving a single function's EH info.
pub struct FunctionEh {
    /// Decoded try regions (RVA-space). Empty when the function has no
    /// exception regions (the overwhelmingly common case).
    pub regions: Vec<EhRegion>,
}

impl MachOExceptionTables {
    /// Resolve EH regions for the function at `func_rva`.
    /// `deref_typeinfo` maps a type_info VA → readable type name (optional).
    pub fn regions_for(
        &self,
        func_rva: u64,
        func_len: u64,
        deref_typeinfo: &dyn Fn(u64) -> Option<String>,
    ) -> FunctionEh {
        let mut regions = Vec::new();
        // __unwind_info functionOffsets and lsdaOffsets are IMAGE-relative
        // (RVA). UnwindCursor.hpp: `lsda = lsdaIndex.lsdaOffset(mid) +
        // sects.dso_base` where dso_base is the Mach-O header VA — so the
        // LSDA bytes live at RVA == lsdaOffset directly.
        let Some(lsda_rva) = find_lsda_offset_in_unwind_info(&self.unwind_info, func_rva) else {
            return FunctionEh { regions };
        };
        // Translate the image-relative LSDA RVA into an offset within the
        // `__gcc_except_tab` section bytes we hold.
        let Some(lsda_in_tab) = (lsda_rva as u64).checked_sub(self.gcc_except_tab_file_rva) else {
            return FunctionEh { regions };
        };
        let lsda_in_tab = lsda_in_tab as usize;
        if lsda_in_tab >= self.gcc_except_tab.len() {
            return FunctionEh { regions };
        }
        let data = &self.gcc_except_tab[lsda_in_tab..];
        let mut regs = parse_lsda(
            data,
            func_rva,
            lsda_rva as u64,
            self.ptr_size,
            self.gcc_except_tab_file_rva,
            &|rva| deref_typeinfo(rva),
        );
        // call sites with landingPad == 0 mean "no handler here" — drop them
        regs.retain(|rg| rg.landing_pad != 0);
        // sanity: keep only regions that actually intersect this function
        regs.retain(|rg| rg.try_end > func_rva && rg.try_start < func_rva.wrapping_add(func_len));
        regions = regs;
        FunctionEh { regions }
    }
}

// ---------------------------------------------------------------------------
// PE/COFF x64 SEH (`.pdata` / `.xdata`)
// ---------------------------------------------------------------------------
//
// On x64 Windows every non-leaf function owns a `RUNTIME_FUNCTION` entry in
// `.pdata` = { BeginAddress; EndAddress; UnwindInfoAddress; } (three ULONGs,
// all RVAs). The third field points into `.xdata` at an `UNWIND_INFO`:
//
//   struct UNWIND_INFO {                       // PE/COFF spec §5.4.2
//       UBYTE Version       : 3;               // 1 for x64
//       UBYTE Flags         : 5;               // UNW_FLAG_{EHANDLER,UHANDLER,CHAININFO}
//       UBYTE SizeOfProlog;                    // bytes
//       UBYTE CountOfCodes;                    // UNWIND_CODE slots (2 bytes each)
//       UBYTE FrameRegister : 4;               // 0 = none
//       UBYTE FrameOffset   : 4;
//       UNWIND_CODE UnwindCode[CountOfCodes];
//       // then, 4-byte aligned:
//       //   UNW_FLAG_CHAININFO -> a nested RUNTIME_FUNCTION (12 bytes) whose
//       //                         UnwindInfoAddress is the parent's unwind info
//       //   otherwise, when a handler flag is set -> EXCEPTION_DIRECTORY:
//       //       ULONG ExceptionHandler;        // RVA of the personality routine
//       //       <language-specific handler data, inline>
//   }
//
// IL2CPP's Windows codegen uses the **SEH** personality `__C_specific_handler`
// rather than the MSVC C++ `__CxxFrameHandler3` `FuncInfo` (a scan of the
// reference `Simple_2019_4_34/GameAssembly.dll` finds no `0x1993052x` magic
// anywhere in the image — see `docs/IL_RECOVERY.md`, session 58), so the
// language-specific data is a `C_SCOPE_TABLE`:
//
//   struct C_SCOPE_TABLE {
//       ULONG Count;
//       struct { ULONG BeginAddress; ULONG EndAddress; ULONG HandlerAddress;
//                ULONG JumpTarget; } ScopeRecord[Count];   // all RVAs
//   }
//
// `HandlerAddress` names the filter/funclet and `JumpTarget` the handler body
// the filter transfers to (0 for a `__finally`, whose funclet is
// `HandlerAddress`). Each scope record is therefore one managed `try` whose
// native range the table states exactly — the same currency as the other
// formats' `EhRegion`, so it folds into `try { } catch { } finally { }` through
// the existing structurer with no special case.
//
// Every read is bounds-checked against the section bytes and every record is
// validated to lie inside the function's own body before it becomes a region.
// That validation is the safety property that matters: a mis-parse (a GS cookie
// in front of the table, or a table belonging to a chained callee) yields
// records outside the function, which are dropped — rather than bogus try
// ranges that fence the structurer's walks. A failed decode is a no-op, never
// an invented region.

/// `UNWIND_INFO.Flags` bit for "an exception handler is present".
const UNW_FLAG_EHANDLER: u8 = 0x01;
/// `UNWIND_INFO.Flags` bit for "a termination handler is present".
const UNW_FLAG_UHANDLER: u8 = 0x02;
/// `UNWIND_INFO.Flags` bit for "the unwind record chains to a parent's".
const UNW_FLAG_CHAININFO: u8 = 0x04;
/// The `__CxxFrameHandler3` `FuncInfo` magic (`0x1993052x`). Sniffed ONLY so a
/// PE that really does use MSVC C++ EH is refused deliberately instead of being
/// mis-read as a scope table — IL2CPP does not emit it, but a hand-written or
/// mingw-built image might, and silently fabricating regions for it would be
/// worse than recovering none.
const MSVC_FUNCINFO_MAGIC: u32 = 0x1993_0520;
/// Mask dropping the low 4 bits of the magic (the `FuncInfo` version nibble).
const MSVC_FUNCINFO_MAGIC_MASK: u32 = 0xffff_fff0;
/// Upper bound on `C_SCOPE_TABLE.Count`; a larger value is a mis-parse, not a
/// real table (the largest scope table in the corpus is well under this).
const MAX_SCOPE_RECORDS: u32 = 4096;
/// The catch-clause marker used wherever the native tables name a managed catch
/// but the PE path cannot resolve its type name (the filter funclet performs
/// the real managed type check, so the type is not in the table). Mirrors the
/// `class_name_of_slot` marker the caller already understands.
const MANAGED_CATCH: &str = "/* managed catch */";

/// Everything needed to resolve SEH try regions in a PE image. Built once from
/// the parsed `.pdata`/`.xdata` sections (see `main.rs`).
pub struct PeExceptionTables {
    /// Sorted (func_rva, unwind_info_rva) pairs from `.pdata`.
    pub funcs: Vec<(u64, u64)>,
    /// Raw `.xdata` bytes: the `UNWIND_INFO` records and their language-specific
    /// handler data. Looked up by RVA through `xdata_rva`.
    pub xdata: Vec<u8>,
    /// Image-relative start RVA of `xdata` (for the rva → offset translation).
    pub xdata_rva: u64,
    /// Pointer width of the target: 8 for PE32+, 4 for PE32. PE32 uses a
    /// frame-based SEH chain with no `.pdata`, so the decoder returns no regions
    /// for it — stated, never invented.
    pub ptr_size: usize,
}

impl PeExceptionTables {
    /// Byte offset of `rva` within the held `.xdata` bytes, or `None` when the
    /// RVA is outside the section (every caller treats that as "no data").
    fn off(&self, rva: u64) -> Option<usize> {
        let o = rva.checked_sub(self.xdata_rva)?;
        let o = usize::try_from(o).ok()?;
        (o < self.xdata.len()).then_some(o)
    }

    /// Decode the SEH try regions for the function covering `func_rva`.
    /// `func_len` is the method body size, used to reject any scope record that
    /// is not inside this function (see the module section above).
    pub fn regions_for(
        &self,
        func_rva: u64,
        func_len: u64,
        _deref_typeinfo: &dyn Fn(u64) -> Option<String>,
    ) -> Vec<EhRegion> {
        // PE32 carries no .pdata — its SEH is a frame-based chain — so there is
        // no scope table to read. Refuse explicitly rather than guess.
        if self.ptr_size != 8 {
            return Vec::new();
        }
        // `.pdata` is sorted by function start: take the last entry at or before
        // the query and require it to actually cover the function.
        let idx = self.funcs.partition_point(|&(s, _)| s <= func_rva);
        let Some(&(start, unwind_rva)) = idx.checked_sub(1).and_then(|i| self.funcs.get(i)) else {
            return Vec::new();
        };
        if unwind_rva == 0 {
            return Vec::new(); // leaf function — nothing can throw
        }
        if func_rva < start
            || (func_rva != start && func_rva >= start.wrapping_add(func_len.max(1)))
        {
            return Vec::new();
        }
        let mut regs = self.decode_unwind(start, func_len, unwind_rva, 0);
        regs.retain(|rg| {
            rg.try_end > func_rva && rg.try_start < func_rva.wrapping_add(func_len.max(1))
        });
        // Same diagnostic convention as the ARM EHABI path: one line per
        // function that actually carries SEH regions, so a real PE dump can be
        // checked with `ILR_EHDBG=1` without a debugger.
        if !regs.is_empty() && std::env::var("ILR_EHDBG").is_ok() {
            eprintln!(
                "[eh] PE {func_rva:#x} unwind={unwind_rva:#x} func_len={func_len:#x} regs={}",
                regs.len()
            );
        }
        regs
    }

    /// Follow one `UNWIND_INFO` (chasing `UNW_FLAG_CHAININFO`) and decode the
    /// scope table it ultimately carries. Returns empty on any malformed or
    /// absent structure; `depth` bounds the chain so a cycle cannot loop.
    fn decode_unwind(
        &self,
        func_base: u64,
        func_len: u64,
        unwind_rva: u64,
        depth: usize,
    ) -> Vec<EhRegion> {
        if depth > 8 {
            return Vec::new();
        }
        let Some(base) = self.off(unwind_rva) else {
            return Vec::new();
        };
        let d = &self.xdata[base..];
        if d.len() < 4 {
            return Vec::new();
        }
        let version = d[0] & 0x07;
        let flags = d[0] >> 3;
        if version != 1 && version != 2 {
            return Vec::new();
        }
        let count_codes = d[2] as usize;
        let codes_end = 4 + 2 * count_codes;
        if codes_end > d.len() {
            return Vec::new();
        }
        if flags & UNW_FLAG_CHAININFO != 0 {
            // A chained record ends with a nested RUNTIME_FUNCTION (12 bytes)
            // whose third ULONG is the parent's UNWIND_INFO. The handler (and
            // therefore the try table) lives in the parent, so follow it.
            if codes_end + 12 > d.len() {
                return Vec::new();
            }
            let chained = u32::from_le_bytes([
                d[codes_end + 8],
                d[codes_end + 9],
                d[codes_end + 10],
                d[codes_end + 11],
            ]);
            if chained == 0 {
                return Vec::new();
            }
            return self.decode_unwind(func_base, func_len, chained as u64, depth + 1);
        }
        if flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) == 0 {
            return Vec::new();
        }
        // The EXCEPTION_DIRECTORY starts on the next 4-byte boundary after the
        // unwind codes: 4 header bytes + 2 per code slot, rounded up.
        let dir_off = 4 + 4 * ((count_codes + 1) / 2);
        if dir_off + 4 > d.len() {
            return Vec::new();
        }
        let handler_rva =
            u32::from_le_bytes([d[dir_off], d[dir_off + 1], d[dir_off + 2], d[dir_off + 3]]);
        if handler_rva == 0 {
            return Vec::new();
        }
        let lang_off = dir_off + 4;
        let lang = &d[lang_off..];
        // Refuse the MSVC C++ layout explicitly: IL2CPP does not emit it, and
        // reading a FuncInfo as a scope table would invent regions.
        if lang.len() >= 4 {
            let first = u32::from_le_bytes([lang[0], lang[1], lang[2], lang[3]]);
            if first & MSVC_FUNCINFO_MAGIC_MASK == MSVC_FUNCINFO_MAGIC {
                return Vec::new();
            }
        }
        // Primary shape: an inline `C_SCOPE_TABLE` right after the handler RVA
        // (`__C_specific_handler`). `__GSHandlerCheck_EH` prepends a GS cookie,
        // so the table can start one dword later; both offsets are tried, and a
        // candidate is accepted only when its records validate inside this
        // function — which is what stops a cookie read as a `Count` from
        // fabricating a region.
        for skip in [0usize, 4] {
            let count_at = lang_off + skip;
            if count_at + 4 > d.len() {
                continue;
            }
            let count = u32::from_le_bytes([
                d[count_at],
                d[count_at + 1],
                d[count_at + 2],
                d[count_at + 3],
            ]);
            if count == 0 || count > MAX_SCOPE_RECORDS {
                continue;
            }
            if let Some(regs) =
                self.decode_scope_records(&d[count_at + 4..], count as usize, func_base, func_len)
            {
                if !regs.is_empty() {
                    return regs;
                }
            }
        }
        // Secondary shape: the language data is a POINTER (RVA) to the scope
        // table, which some toolchains place elsewhere in `.xdata`.
        if lang.len() >= 4 {
            let table_rva = u32::from_le_bytes([lang[0], lang[1], lang[2], lang[3]]);
            if let Some(tb) = self.off(table_rva as u64) {
                let t = &self.xdata[tb..];
                if t.len() >= 4 {
                    let count = u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
                    if count > 0 && count <= MAX_SCOPE_RECORDS {
                        if let Some(regs) =
                            self.decode_scope_records(&t[4..], count as usize, func_base, func_len)
                        {
                            if !regs.is_empty() {
                                return regs;
                            }
                        }
                    }
                }
            }
        }
        Vec::new()
    }

    /// Decode `count` 16-byte `C_SCOPE_TABLE` records into regions.
    ///
    /// Returns `None` when the bytes are not all present (a truncated count is a
    /// mis-parse, not a partial table) and otherwise the records that pass the
    /// range validation. A record's `BeginAddress`/`EndAddress` MUST lie inside
    /// `[func_base, func_base+func_len)`; an `EndAddress` of 0 means "to the end
    /// of the function" (MSVC's spelling).
    fn decode_scope_records(
        &self,
        recs: &[u8],
        count: usize,
        func_base: u64,
        func_len: u64,
    ) -> Option<Vec<EhRegion>> {
        let need = count.checked_mul(16)?;
        if recs.len() < need {
            return None;
        }
        let lo = func_base;
        let hi = func_base.wrapping_add(func_len.max(1));
        let mut out = Vec::new();
        for i in 0..count {
            let o = i * 16;
            let rd = |k: usize| -> u64 {
                u32::from_le_bytes([
                    recs[o + k],
                    recs[o + k + 1],
                    recs[o + k + 2],
                    recs[o + k + 3],
                ]) as u64
            };
            let begin = rd(0);
            let mut end = rd(4);
            let handler = rd(8);
            let jump = rd(12);
            if begin == 0 || handler == 0 {
                continue;
            }
            if end == 0 {
                end = hi;
            }
            if end <= begin {
                continue;
            }
            // Must be a range inside THIS function's body. Anything else means
            // the table belongs to another (chained) function or we
            // mis-located it — refuse rather than fence the structurer.
            if begin < lo || begin >= hi || end > hi {
                continue;
            }
            // JumpTarget 0 = no handler body to transfer to (a `__finally`),
            // whose funclet is `HandlerAddress`; otherwise the handler body.
            let is_cleanup = jump == 0;
            let landing_pad = if is_cleanup { handler } else { jump };
            let chain = if is_cleanup {
                vec![Clause::Cleanup]
            } else {
                vec![Clause::Catch {
                    name: MANAGED_CATCH.to_string(),
                }]
            };
            out.push(EhRegion::new(begin, end, landing_pad, chain));
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Tests (self-contained; no binary fixtures needed)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ARM EHABI (`.ARM.exidx` / `.ARM.extab`)
// ---------------------------------------------------------------------------
//
// Every 32-bit ARM ELF that supports exception handling carries two sections
// in place of DWARF `.eh_frame`:
//
//   `.ARM.exidx` — a fixed-stride (8-byte), function-sorted index table. Each
//     entry is { prel31(function_start); content } where `content` is either a
//     prel31 to a handling-table entry in `.ARM.extab`, an INLINE compact entry
//     (bit 31 set), or EXIDX_CANTUNWIND (0x1).
//
//   `.ARM.extab` — variable-length handling-table entries. An entry whose first
//     word has bit 31 set is the ARM-defined compact model: bits 24-27 select
//     personality routine 0/1/2, bits 16-23 hold the count of additional
//     opcode words (0/1/2 only), and the remaining bytes are unwind opcodes
//     followed by the descriptor list. An entry whose first word has bit 31
//     clear is the generic model (prel31 to a custom personality routine plus
//     private data) and is deliberately NOT decoded here.
//
// Descriptors are walked exactly as GCC's `__gnu_unwind_pr_common` does: a
// scope word that encodes { length, offset } relative to the function start,
// with the low bit of each half selecting the descriptor kind —
//   (0,0) cleanup : one prel31 landing pad
//   (1,0) catch   : a landing-pad word + a type_info word
//   (0,1) throws  : an N-entry exception-specification list
// terminated by a zero word. The kind test is `(offset&1)<<1 | (length&1)`, so
// kind 1 is the CATCH form and kind 2 the exception specification.

/// Read a little-endian `u32` at byte offset `off`.
fn rd_le_u32(bytes: &[u8], off: usize) -> Option<u32> {
    bytes
        .get(off..off + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Decode an R_ARM_PREL31 word at byte offset `off`: bits 0-30 are a signed
/// offset from the address of the word itself, bit 31 is data. `at` is the
/// address of that word in the same RVA space the result should be in.
fn arm_prel31(bytes: &[u8], at: u64, off: usize) -> Option<u64> {
    let w = rd_le_u32(bytes, off)?;
    let v = (w & 0x7fff_ffff) as i64;
    let signed = if v & 0x4000_0000 != 0 {
        v - 0x8000_0000
    } else {
        v
    };
    Some((at as i64 + signed) as u64 & 0xffff_ffff)
}

/// Decode the descriptor list of a compact-model handling-table entry.
fn parse_arm_descriptors(
    extab: &[u8],
    extab_rva: u64,
    start: usize,
    func: u64,
    scope32: bool,
) -> Vec<EhRegion> {
    let mut out = Vec::new();
    let mut o = start;
    // The list is terminated by a zero word; the cap is only a guard against a
    // malformed stream, never the real terminator.
    for _ in 0..4096 {
        let (length, offset, kind);
        if scope32 {
            let (Some(l), Some(off)) = (rd_le_u32(extab, o), rd_le_u32(extab, o + 4)) else {
                break;
            };
            if l == 0 && off == 0 {
                break;
            }
            kind = ((off & 1) << 1) | (l & 1);
            length = (l & !1) as u64;
            offset = (off & !1) as u64;
            o += 8;
        } else {
            let Some(w) = rd_le_u32(extab, o) else { break };
            if w == 0 {
                break;
            }
            let l = w & 0xffff;
            let off = (w >> 16) & 0xffff;
            kind = ((off & 1) << 1) | (l & 1);
            length = (l & !1) as u64;
            offset = (off & !1) as u64;
            o += 4;
        }
        let try_start = func.wrapping_add(offset);
        let try_end = try_start.wrapping_add(length);
        match kind {
            0 => {
                // Cleanup: one prel31 landing pad.
                let lp = arm_prel31(extab, extab_rva + o as u64, o).unwrap_or(0);
                out.push(EhRegion::new(try_start, try_end, lp, vec![Clause::Cleanup]));
                o += 4;
            }
            1 => {
                // Catch: a landing-pad word then a type_info word. The type is a
                // VA that cannot be named without typing metadata, so it renders
                // as `catch (...)`. 0xfffffffe is the "no-throw" idiom and is
                // not a managed catch.
                let lp = arm_prel31(extab, extab_rva + o as u64, o).unwrap_or(0);
                let ty = rd_le_u32(extab, o + 4).unwrap_or(0);
                if ty != 0xffff_fffe {
                    out.push(EhRegion::new(try_start, try_end, lp, vec![Clause::CatchAll]));
                }
                o += 8;
            }
            _ => {
                // Exception specification: `N` type_info words; a leading
                // high bit adds a trailing no-match landing pad.
                let Some(nw) = rd_le_u32(extab, o) else { break };
                let count = (nw & 0x7fff_ffff) as usize;
                if count > 4096 {
                    break;
                }
                o += 4 + 4 * count;
                if nw & 0x8000_0000 != 0 {
                    o += 4;
                }
            }
        }
    }
    out
}

/// Decode the EH regions for one function from a `.ARM.exidx`/`.ARM.extab`
/// pair. `func_rva`/`func_len` bound the function whose regions are wanted.
fn arm_ehabi_regions(
    exidx_rva: u64,
    exidx: &[u8],
    extab_rva: u64,
    extab: &[u8],
    func_rva: u64,
    func_len: u64,
    deref_typeinfo: &dyn Fn(u64) -> Option<String>,
) -> Vec<EhRegion> {
    let n = exidx.len() / 8;
    if n == 0 {
        return Vec::new();
    }
    // Index entries are sorted by function start, so binary-search the last
    // entry whose function begins at or before `func_rva`.
    let func_of = |i: usize| arm_prel31(exidx, exidx_rva + (i as u64) * 8, i * 8);
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        match func_of(mid) {
            Some(f) if f <= func_rva => lo = mid + 1,
            _ => hi = mid,
        }
    }
    let Some(i) = lo.checked_sub(1) else {
        return Vec::new();
    };
    let Some(func) = func_of(i) else {
        return Vec::new();
    };
    // The entry must actually be THIS function's: the found function has to
    // start at or before the queried RVA and still cover it.
    if func_rva < func || (func_rva != func && func_rva >= func.wrapping_add(func_len.max(1))) {
        return Vec::new();
    }
    let Some(content) = rd_le_u32(exidx, i * 8 + 4) else {
        return Vec::new();
    };
    if content == 1 {
        return Vec::new(); // EXIDX_CANTUNWIND
    }
    // Inline compact entry: the handling table is the whole 31-bit word, so it
    // carries unwind opcodes only — by construction it has no descriptor list.
    if content & 0x8000_0000 != 0 {
        return Vec::new();
    }
    if extab.is_empty() {
        return Vec::new();
    }
    let Some(entry) = arm_prel31(exidx, exidx_rva + (i as u64) * 8 + 4, i * 8 + 4) else {
        return Vec::new();
    };
    let Some(ent_off) = (entry.checked_sub(extab_rva)).map(|o| o as usize) else {
        return Vec::new();
    };
    let Some(w0) = rd_le_u32(extab, ent_off) else {
        return Vec::new();
    };
    if w0 & 0x8000_0000 == 0 {
        // GENERIC model: the payload is an Itanium/DWARF LSDA, not an EHABI
        // descriptor list. The unwind opcodes come first — LLVM's
        // `decode_eht_entry` fixes their length at `((word1>>24)+1)` words — so
        // the LSDA starts at `entry + 8 + 4*lenbyte`. The existing `.eh_frame`
        // LSDA parser decodes it unchanged (LPStart omitted ⟹ the function
        // start, which is how these tables are emitted).
        //
        // This is the payload that carries the managed `try`/`catch`/`finally`
        // tables on Android: libil2cpp has 8,425 of these entries in its
        // managed section, and the compact entries carry no descriptors at all.
        let Some(w1) = rd_le_u32(extab, ent_off + 4) else {
            return Vec::new();
        };
        let lenbyte = ((w1 >> 24) & 0xff) as usize;
        let lsda_off = ent_off + 8 + 4 * lenbyte;
        if lsda_off >= extab.len() {
            return Vec::new();
        }
        let lsda_rva = extab_rva + lsda_off as u64;
        let mut regs = parse_lsda(
            &extab[lsda_off..],
            func,
            lsda_rva,
            4,
            extab_rva,
            deref_typeinfo,
        );
        regs.retain(|rg| {
            rg.try_end > func_rva && rg.try_start < func_rva.wrapping_add(func_len.max(1))
        });
        if std::env::var("ILR_EHDBG").is_ok() {
            eprintln!(
                "[eh] generic {func_rva:#x} ent={ent_off:#x} lenbyte={lenbyte} regs={}",
                regs.len()
            );
        }
        return regs;
    }
    let personality = (w0 >> 24) & 0xf;
    let (extra_words, scope32) = match personality {
        0 => (0u32, false),
        1 => ((w0 >> 16) & 0xff, false),
        2 => ((w0 >> 16) & 0xff, true),
        _ => return Vec::new(), // index 3-15 reserved
    };
    let desc_off = ent_off + 4 + 4 * extra_words as usize;
    let mut regs = parse_arm_descriptors(extab, extab_rva, desc_off, func, scope32);
    regs.retain(|rg| {
        rg.try_end > func_rva && rg.try_start < func_rva.wrapping_add(func_len.max(1))
    });
    regs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsda_roundtrip_basic_catch() {
        // Byte layout (offsets annotated) — computed in steps so the offsets
        // can't drift: the type table lives BEFORE `classInfo` (which points
        // at the table's END; entries index backward) and the type_info
        // pointer is pcrel-encoded, so resolved RVA =
        // lsda_rva + field_offset + stored_delta.
        let mut lsda: Vec<u8> = Vec::new();
        lsda.push(0x00); //                              [0]  lpStartEncoding absptr
        lsda.extend_from_slice(&0x1000u64.to_le_bytes()); // [1..9] lpStart
        lsda.push(0x9b); //                              [9]  ttypeEncoding pcrel|sdata4
        let cio_off = lsda.len() as u8; //               [10] classInfoOffset byte
        lsda.push(0x00); // placeholder, patched below
        lsda.push(0x01); //                              [11] callSiteEncoding uleb128
        lsda.push(0x04); //                              [12] callSiteTableLength = 4
        let cs_start = 13usize; // call-site table starts here (r.pos after header)
        assert_eq!(cs_start, lsda.len());
        lsda.push(0x00); //                              [13] cs: start = 0
        lsda.push(0x40); //                              [14] cs: len = 0x40
        lsda.push(0x10); //                              [15] cs: landingPad = 0x10
        lsda.push(0x01); //                              [16] cs: action = 1
        let action_at = cs_start + 4; //                 [17] action table
        assert_eq!(action_at, lsda.len());
        lsda.push(0x01); //                              [17] ttypeIndex = 1
        lsda.push(0x00); //                              [18] actionOffset = 0
        let type_entry_at = lsda.len() as u64; //        [19]
        lsda.extend_from_slice(&(0x2000u32.to_le_bytes())); // [19..23] type ptr (pcrel)
        let class_info = lsda.len() as u64; //           23 = one-past the entry
        lsda.extend_from_slice(&[0u8; 2]); // pad tail
        // classInfoOffset is relative to the position AFTER its own byte:
        lsda[cio_off as usize] = (class_info - (cio_off as u64 + 1)) as u8;

        let lsda_rva = 0x3000u64;
        let expect_ti = lsda_rva + type_entry_at + 0x2000; // pcrel fixup
        let deref = |rva: u64| -> Option<String> {
            if rva == expect_ti {
                Some("System.Exception".to_string())
            } else {
                None
            }
        };
        let regions = parse_lsda(&lsda, 0x1000, lsda_rva, 8, 0, &deref);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.try_start, 0x1000);
        assert_eq!(r.try_end, 0x1040);
        assert_eq!(r.landing_pad, 0x1010); // lpStart(0x1000) + 0x10
        assert_eq!(r.catches.len(), 1, "action chain must yield one catch");
        assert_eq!(
            r.catches[0],
            Some("System.Exception".to_string()),
            "type_info pcrel resolution should produce the exact catch type"
        );
    }

    /// LSDA with the corpus's real action-table shape (see
    /// `managed_catch_chain_walks_the_shared_cleanup_record`), parameterised
    /// by the call-site row's action entry.
    fn corpus_style_chain_lsda(action_entry: u8) -> Vec<u8> {
        let mut lsda: Vec<u8> = Vec::new();
        lsda.push(0xff); // [0] lpStartEncoding omit
        lsda.push(0x9b); // [1] ttypeEncoding pcrel|indirect|sdata4
        let cio_off = lsda.len(); // [2] classInfoOffset, patched below
        lsda.push(0x00);
        lsda.push(0x01); // [3] callSiteEncoding uleb128
        lsda.push(0x04); // [4] callSiteTableLength = 4
        lsda.push(0x00); // [5] start = 0
        lsda.push(0x40); // [6] len = 0x40
        lsda.push(0x10); // [7] landingPad = 0x10
        lsda.push(action_entry); // [8] action entry
        assert_eq!(lsda.len(), 9);
        lsda.extend_from_slice(&[0x00, 0x00]); // [9]  action 1: cleanup, next 0
        lsda.extend_from_slice(&[0x01, 0x7d]); // [11] action 3: catch 1, next -3
        lsda.extend_from_slice(&[0x02, 0x00]); // [13] action 5: catch-all, next 0
        lsda.extend_from_slice(&0u32.to_le_bytes()); // [15] type entry 2 = NULL
        lsda.extend_from_slice(&0x2000u32.to_le_bytes()); // [19] type entry 1
        let class_info = lsda.len() as u64; // 23
        lsda[cio_off] = (class_info - (cio_off as u64 + 1)) as u8;
        lsda
    }

    fn corpus_deref(rva: u64) -> Option<String> {
        // type entry 1 at offset 19, pcrel from lsda_rva 0x3000
        if rva == 0x3000 + 19 + 0x2000 {
            Some("System.Exception".to_string())
        } else {
            None
        }
    }
    /// A managed catch compiles to the Itanium chain
    /// [catch(wrapper) -> cleanup]. The trailing arm is the personality's
    /// NO-MATCH path (ttypeIndex 0 = a cleanup ACTION), not a destructor run
    /// and not a second clause, so the region is a CATCH, not a cleanup.
    /// Reading the arm as the region kind made every managed catch render as
    /// C# `finally` (E2: 4,970/4,970 typed rows).
    #[test]
    fn a_chain_ending_in_a_catch_all_is_a_catch_not_a_cleanup() {
        let mut lsda: Vec<u8> = Vec::new();
        lsda.push(0x00); //                              [0]  lpStartEncoding absptr
        lsda.extend_from_slice(&0x1000u64.to_le_bytes()); // [1..9] lpStart
        lsda.push(0x9b); //                              [9]  ttypeEncoding pcrel|indirect|sdata4
        let cio_off = lsda.len() as u8; //               [10] classInfoOffset byte
        lsda.push(0x00); // placeholder, patched below
        lsda.push(0x01); //                              [11] callSiteEncoding uleb128
        lsda.push(0x04); //                              [12] callSiteTableLength = 4
        let cs_start = 13usize;
        assert_eq!(cs_start, lsda.len());
        lsda.push(0x00); //                              [13] cs: start = 0
        lsda.push(0x40); //                              [14] cs: len = 0x40
        lsda.push(0x10); //                              [15] cs: landingPad = 0x10
        lsda.push(0x01); //                              [16] cs: action = 1
        assert_eq!(cs_start + 4, lsda.len());
        lsda.push(0x01); //                              [17] arm 1: ttypeIndex = 1
        lsda.push(0x02); //                              [18] arm 1: actionOffset -> [21]
        lsda.push(0x00); //                              [19] filler
        lsda.push(0x00); //                              [20] filler
        lsda.push(0x00); //                              [21] arm 2: ttypeIndex = 0 (fallback)
        lsda.push(0x00); //                              [22] arm 2: actionOffset = 0 (end)
        let type_entry_at = lsda.len() as u64; //        [23]
        lsda.extend_from_slice(&0x2000u32.to_le_bytes()); // [23..27] type ptr (pcrel)
        let class_info = lsda.len() as u64; //           27 = one-past the entry
        lsda.extend_from_slice(&[0u8; 2]);
        lsda[cio_off as usize] = (class_info - (cio_off as u64 + 1)) as u8;

        let lsda_rva = 0x3000u64;
        let expect_ti = lsda_rva + type_entry_at + 0x2000;
        let deref = |rva: u64| -> Option<String> {
            if rva == expect_ti {
                Some("System.Exception".to_string())
            } else {
                None
            }
        };
        let regions = parse_lsda(&lsda, 0x1000, lsda_rva, 8, 0, &deref);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert!(
            !r.is_cleanup,
            "a catch chain must not be classified as a cleanup: {r:?}"
        );
        assert_eq!(
            r.chain,
            vec![
                Clause::Catch {
                    name: "System.Exception".to_string()
                },
                Clause::Cleanup
            ],
            "one typed arm, then the personality's no-match cleanup: {r:?}"
        );
        assert_eq!(
            r.catches,
            vec![Some("System.Exception".to_string())],
            "the cleanup arm is not a catch clause and must not appear"
        );
        assert!(r.has_catch() && r.has_cleanup());
    }

    /// The exact action table this corpus's compiler emits for one managed
    /// catch, taken from libil2cpp.so LSDA @0xaf0600 (function 0x15a07dc):
    ///
    ///   546: 00 00     action 1: (ttype 0 = cleanup, next 0)
    ///   548: 01 7d     action 3: (ttype 1 = catch, next -3)
    ///   550: 02 00     action 5: (ttype 2 = NULL entry = catch-all, next 0)
    ///   552..: zeros, then the type table (index 1 at 560, ci = 568)
    ///
    /// Action 3's -3 is a BACKWARD offset: 549 - 3 = 546, the shared cleanup
    /// record. Applying it from the end of the record (550 - 3 = 547) lands
    /// one byte into the table, reads [00, 01] as (cleanup, next +1) and
    /// appends a phantom third clause - which is what every one of the 7,588
    /// managed-catch rows in this binary used to decode to.
    #[test]
    fn managed_catch_chain_walks_the_shared_cleanup_record() {
        let lsda = corpus_style_chain_lsda(3);
        let regions = parse_lsda(&lsda, 0x1000, 0x3000, 8, 0, &corpus_deref);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(
            r.chain,
            vec![
                Clause::Catch {
                    name: "System.Exception".to_string()
                },
                Clause::Cleanup
            ],
            "catch arm 3 must chain BACKWARDS to the shared cleanup record: {r:?}"
        );
        assert_eq!(r.catches, vec![Some("System.Exception".to_string())]);
        assert!(!r.is_cleanup);
    }

    /// ttypeIndex > 0 with a NULL type-table entry is an untyped catch, and
    /// must not be confused with ttypeIndex == 0 (a cleanup action). The
    /// compiler emits this shape for the terminate guard it puts around calls
    /// inside a cleanup: 1,645 of the 9,233 handler rows in this binary.
    #[test]
    fn null_type_entry_is_a_catch_all_not_a_cleanup() {
        let lsda = corpus_style_chain_lsda(5);
        let regions = parse_lsda(&lsda, 0x1000, 0x3000, 8, 0, &corpus_deref);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(
            r.chain,
            vec![Clause::CatchAll],
            "a null type entry is a catch-all clause: {r:?}"
        );
        assert!(r.has_catch());
        assert!(!r.has_cleanup());
        assert!(!r.is_cleanup, "an untyped catch is not a cleanup");
        assert_eq!(r.catches, vec![None]);
    }

    #[test]
    fn lsda_cleanup_only() {
        // landingPad offset 0 = "no handler at this call site" (the runtime
        // checks landingPad == 0 BEFORE adding lpStart — cxa_personality.cpp).
        // A cleanup with an actual landing pad uses actionEntry = 0.
        let mut lsda: Vec<u8> = Vec::new();
        lsda.push(0xff); // lpStartEncoding omit → lpStart = funcStart
        lsda.push(0xff); // ttypeEncoding omit → no classInfo
        lsda.push(0x01); // callSiteEncoding uleb128
        lsda.push(0x04); // callSiteTableLength = 4 (one 4-field site)
        lsda.push(0x00); // start 0
        lsda.push(0x20); // len 0x20
        lsda.push(0x10); // landingPad offset 0x10 → lpStart + 0x10
        lsda.push(0x00); // actionEntry 0 → cleanup
        let regions = parse_lsda(&lsda, 0x8000, 0x9000, 8, 0, &|_| None);
        assert_eq!(regions.len(), 1);
        assert!(regions[0].catches.is_empty());
        assert!(regions[0].is_cleanup);
        assert_eq!(regions[0].landing_pad, 0x8010); // funcStart + 0x10
    }

    #[test]
    fn unwind_info_two_level_lookup() {
        // Minimal __unwind_info: header(28) | 1 index entry (12) | 1 lsda
        // entry (8). The top-level index ends at 40, so the lsda-index sub-
        // array starts there. The lookup must NOT require a sentinel entry.
        let mut ui: Vec<u8> = Vec::new();
        ui.extend_from_slice(&1u32.to_le_bytes()); // version
        ui.extend_from_slice(&0u32.to_le_bytes()); // common enc off
        ui.extend_from_slice(&0u32.to_le_bytes()); // common enc count
        ui.extend_from_slice(&0u32.to_le_bytes()); // personality off
        ui.extend_from_slice(&0u32.to_le_bytes()); // personality count
        let index_off = 28u32;
        ui.extend_from_slice(&index_off.to_le_bytes()); // index off
        ui.extend_from_slice(&1u32.to_le_bytes()); // index count = 1
        // index entry @28: functionOffset=0x1000, secondLevelPage=0,
        // lsdaIndexArraySectionOffset=40
        ui.extend_from_slice(&0x1000u32.to_le_bytes());
        ui.extend_from_slice(&0u32.to_le_bytes());
        ui.extend_from_slice(&40u32.to_le_bytes());
        // lsda index entry @40: functionOffset=0x1000, lsdaOffset=0x1234
        ui.extend_from_slice(&0x1000u32.to_le_bytes());
        ui.extend_from_slice(&0x1234u32.to_le_bytes());

        let got = find_lsda_offset_in_unwind_info(&ui, 0x1000);
        assert_eq!(got, Some(0x1234));
        // miss on a function before the covered range
        assert_eq!(find_lsda_offset_in_unwind_info(&ui, 0x9000), None);
    }

    #[test]
    fn uleb_sleb_roundtrip() {
        let mut r = LsdaReader::new(&[0xE5, 0x8E, 0x26]);
        assert_eq!(r.uleb128(), Some(624_485));
        let mut r = LsdaReader::new(&[0x7F]);
        assert_eq!(r.sleb128(), Some(-1));
        let mut r = LsdaReader::new(&[0x80, 0x7F]);
        assert_eq!(r.sleb128(), Some(-128));
    }

    #[test]
    fn encoded_absptr_is_pointer_width() {
        // A 32-bit absptr value must consume 4 bytes, a 64-bit one 8 — reading
        // the wrong width mis-parses every field that follows it.
        let mut r = LsdaReader::with_ptr_size(&[0x10, 0, 0, 0], 4, 0);
        assert_eq!(r.encoded(DW_EH_PE_ABSPTR, 0), Some(0x10));
        assert_eq!(r.pos, 4, "32-bit absptr consumes exactly four bytes");
        let mut r = LsdaReader::with_ptr_size(&[0x10, 0, 0, 0, 0, 0, 0, 0], 8, 0);
        assert_eq!(r.encoded(DW_EH_PE_ABSPTR, 0), Some(0x10));
        assert_eq!(r.pos, 8, "64-bit absptr consumes exactly eight bytes");
    }

    #[test]
    fn encoded_datarel_adds_the_data_base() {
        // DW_EH_PE_datarel | DW_EH_PE_udata4 (0x13): the stored value is an
        // offset from the data section, so the base must be added. Returning
        // the raw offset (the old behaviour) produced an address in whatever
        // section happened to start at file offset 0.
        let mut r = LsdaReader::with_ptr_size(&[0x10, 0, 0, 0], 8, 0x1000);
        assert_eq!(
            r.encoded(DW_EH_PE_DATAREL | DW_EH_PE_UDATA4, 0),
            Some(0x1010)
        );
    }

    #[test]
    fn lsda_absptr_lpstart_honours_32bit_pointer_width() {
        // Whole-LSDA regression: on a 32-bit target the lpStart field is four
        // bytes, so the call-site table starts four bytes earlier. Parsing it
        // as 64-bit shifts every following field and yields garbage (or no
        // regions at all).
        let mut lsda: Vec<u8> = Vec::new();
        lsda.push(0x00); // lpStartEncoding = absptr
        lsda.extend_from_slice(&0x1000u32.to_le_bytes()); // 4-byte lpStart
        lsda.push(0xff); // ttypeEncoding = omit
        lsda.push(0x01); // callSiteEncoding = uleb128
        lsda.push(0x04); // callSiteTableLength = 4
        lsda.push(0x00); // call-site start = 0
        lsda.push(0x20); // call-site length = 0x20
        lsda.push(0x10); // landingPad offset = 0x10
        lsda.push(0x00); // actionEntry = 0 (cleanup)
        let regions = parse_lsda(&lsda, 0x5000, 0x6000, 4, 0, &|_| None);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].landing_pad, 0x1010); // lpStart(0x1000) + 0x10
        assert_eq!(regions[0].try_start, 0x5000);
        assert_eq!(regions[0].try_end, 0x5020);
    }

    // -- ARM EHABI ---------------------------------------------------------

    /// An 8-byte index entry whose first word is a prel31 to `func` from `at`.
    fn exidx_entry(at: u64, func: u64, content: u32) -> [u8; 8] {
        let d = (func as i64 - at as i64) & 0x7fff_ffff;
        let w0 = (d as u32) & 0x7fff_ffff;
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&w0.to_le_bytes());
        out[4..].copy_from_slice(&content.to_le_bytes());
        out
    }

    /// A prel31 word at `at` pointing at `target` (bit 31 clear).
    fn prel31_word(at: u64, target: u64) -> u32 {
        ((target as i64 - at as i64) & 0x7fff_ffff) as u32
    }

    fn arm_tables(exidx: Vec<u8>, extab: Vec<u8>) -> EhTables {
        EhTables::ArmEhAbi {
            exidx,
            exidx_rva: 0x8000,
            extab,
            extab_rva: 0x9000,
        }
    }

    /// A compact PR1 entry whose descriptor list is a single cleanup.
    ///
    /// This is the shape the decoder exists for: a `try/finally` body whose
    /// cleanup landing pad is named exactly by the table instead of guessed from
    /// the control flow.
    #[test]
    fn arm_ehabi_cleanup_descriptor_yields_a_finally_region() {
        let exidx = exidx_entry(0x8000, 0x1000, prel31_word(0x8004, 0x9000)).to_vec();
        // PI1, zero extra opcode words, two finish opcodes.
        let mut extab = Vec::new();
        extab.extend_from_slice(&0x8100_B0B0u32.to_le_bytes()); // handling header
        extab.extend_from_slice(&0x0010_0020u32.to_le_bytes()); // scope len 0x20, off 0x10
        extab.extend_from_slice(&prel31_word(0x9008, 0x1030).to_le_bytes());
        extab.extend_from_slice(&0u32.to_le_bytes()); // terminator
        let regs = arm_tables(exidx, extab).regions_for(0x1000, 0x100, &|_| None);
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].try_start, 0x1010);
        assert_eq!(regs[0].try_end, 0x1030);
        assert_eq!(regs[0].landing_pad, 0x1030);
        assert!(regs[0].is_cleanup);
        assert!(regs[0].catches.is_empty());
    }

    /// A compact PR1 entry with a catch descriptor.
    #[test]
    fn arm_ehabi_catch_descriptor_yields_a_catch_region() {
        let exidx = exidx_entry(0x8000, 0x1000, prel31_word(0x8004, 0x9000)).to_vec();
        let mut extab = Vec::new();
        extab.extend_from_slice(&0x8100_B0B0u32.to_le_bytes());
        // length 0x21 (odd -> kind bit), offset 0x10 (even) -> kind 1 = catch.
        extab.extend_from_slice(&0x0010_0021u32.to_le_bytes());
        extab.extend_from_slice(&prel31_word(0x9008, 0x1080).to_le_bytes()); // landing pad
        extab.extend_from_slice(&0xdead_beefu32.to_le_bytes()); // type_info VA
        extab.extend_from_slice(&0u32.to_le_bytes());
        let regs = arm_tables(exidx, extab).regions_for(0x1000, 0x100, &|_| None);
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].landing_pad, 0x1080);
        assert!(!regs[0].is_cleanup);
        assert_eq!(regs[0].catches, vec![None]);
    }

    /// An inline compact entry is self-contained: no descriptors, no regions.
    #[test]
    fn arm_ehabi_inline_entry_has_no_descriptors() {
        let exidx = exidx_entry(0x8000, 0x1000, 0x8000_B0B0).to_vec();
        let regs = arm_tables(exidx, Vec::new()).regions_for(0x1000, 0x100, &|_| None);
        assert!(regs.is_empty());
    }

    /// EXIDX_CANTUNWIND, and a truncated generic entry, yield no regions.
    #[test]
    fn arm_ehabi_cantunwind_and_truncated_generic_yield_nothing() {
        let cant = exidx_entry(0x8000, 0x1000, 1).to_vec();
        assert!(
            arm_tables(cant, Vec::new())
                .regions_for(0x1000, 0x100, &|_| None)
                .is_empty()
        );
        // Generic (extab word 0 has bit 31 clear), but the entry is only the
        // personality-routine word — there is no LSDA to decode.
        let exidx = exidx_entry(0x8000, 0x1000, prel31_word(0x8004, 0x9000)).to_vec();
        let extab = prel31_word(0x9000, 0x5000).to_le_bytes().to_vec();
        assert!(
            arm_tables(exidx, extab)
                .regions_for(0x1000, 0x100, &|_| None)
                .is_empty()
        );
    }

    /// A GENERIC entry whose payload is an Itanium LSDA yields its regions.
    ///
    /// This is the model that carries managed `try`/`finally` on Android: the
    /// payload is a DWARF LSDA after the unwind opcodes (offset
    /// `8 + 4*lenbyte`, lenbyte 0 here), decoded by the same `parse_lsda` the
    /// `.eh_frame` path uses. The landing pad is LPStart-relative and LPStart
    /// is omitted, so it is function-relative.
    #[test]
    fn arm_ehabi_generic_entry_decodes_an_itanium_lsda() {
        let exidx = exidx_entry(0x8000, 0x1000, prel31_word(0x8004, 0x9000)).to_vec();
        let mut extab = Vec::new();
        extab.extend_from_slice(&prel31_word(0x9000, 0x9000).to_le_bytes()); // word0: PR, bit31 clear
        extab.extend_from_slice(&0u32.to_le_bytes()); // word1: lenbyte 0 -> LSDA at +8
        extab.push(0xff); // lpStartEncoding = omit
        extab.push(0xff); // ttypeEncoding = omit
        extab.push(0x03); // callSiteEncoding = udata4
        extab.push(0x0d); // callSiteTableLength = 13
        extab.extend_from_slice(&0x20u32.to_le_bytes()); // start = 0x20
        extab.extend_from_slice(&0x10u32.to_le_bytes()); // length = 0x10
        extab.extend_from_slice(&0x40u32.to_le_bytes()); // landingPad = 0x40
        extab.push(0x00); // actionEntry = 0 -> cleanup
        let regs = arm_tables(exidx, extab).regions_for(0x1000, 0x100, &|_| None);
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].try_start, 0x1020);
        assert_eq!(regs[0].try_end, 0x1030);
        assert_eq!(regs[0].landing_pad, 0x1040);
        assert!(regs[0].is_cleanup);
    }

    /// An index entry for a DIFFERENT function is never borrowed.
    #[test]
    fn arm_ehabi_entry_only_covers_its_own_function() {
        let exidx = exidx_entry(0x8000, 0x1000, prel31_word(0x8004, 0x9000)).to_vec();
        let mut extab = Vec::new();
        extab.extend_from_slice(&0x8100_B0B0u32.to_le_bytes());
        extab.extend_from_slice(&0x0010_0020u32.to_le_bytes());
        extab.extend_from_slice(&prel31_word(0x9008, 0x1030).to_le_bytes());
        extab.extend_from_slice(&0u32.to_le_bytes());
        let tables = arm_tables(exidx, extab);
        // 0x2000 is past the indexed function's range.
        assert!(tables.regions_for(0x2000, 0x100, &|_| None).is_empty());
    }

    // -- PE/COFF x64 SEH ---------------------------------------------------

    /// A `.xdata` image with one `UNWIND_INFO` at RVA 0x6000 (== byte 0) whose
    /// `EXCEPTION_DIRECTORY` names `handler_rva` and is followed by the given
    /// inline language-specific payload.
    fn pe_xdata_with_handler(handler_rva: u32, lang_payload: &[u8]) -> Vec<u8> {
        let mut d = Vec::new();
        d.push(0x01 | (UNW_FLAG_EHANDLER << 3)); // Version 1, UNW_FLAG_EHANDLER
        d.push(0x10); // SizeOfProlog
        d.push(0x00); // CountOfCodes = 0 (no unwind codes, so dir is at offset 4)
        d.push(0x00); // FrameRegister/FrameOffset
        d.extend_from_slice(&handler_rva.to_le_bytes()); // ExceptionHandler RVA
        d.extend_from_slice(lang_payload); // language-specific data
        d
    }

    fn pe_tables(xdata: Vec<u8>, func_rva: u64, unwind_rva: u64) -> EhTables {
        EhTables::Pe(PeExceptionTables {
            funcs: vec![(func_rva, unwind_rva)],
            xdata,
            xdata_rva: 0x6000,
            ptr_size: 8,
        })
    }

    /// Encode a `C_SCOPE_TABLE` body: `Count` then the 16-byte records.
    fn scope_table(records: &[(u32, u32, u32, u32)]) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(&(records.len() as u32).to_le_bytes());
        for (b, e, h, j) in records {
            d.extend_from_slice(&b.to_le_bytes());
            d.extend_from_slice(&e.to_le_bytes());
            d.extend_from_slice(&h.to_le_bytes());
            d.extend_from_slice(&j.to_le_bytes());
        }
        d
    }

    /// A `__except` scope (a real `JumpTarget`) becomes a catch region.
    #[test]
    fn pe_scope_table_catch_yields_a_try_region() {
        // Function at 0x1000, 0x200 bytes: try 0x1020..0x1040, filter at 0x1080,
        // handler body (JumpTarget) at 0x1090.
        let tbl = scope_table(&[(0x1020, 0x1040, 0x1080, 0x1090)]);
        let xdata = pe_xdata_with_handler(0x1234, &tbl);
        let regs = pe_tables(xdata, 0x1000, 0x6000).regions_for(0x1000, 0x200, &|_| None);
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].try_start, 0x1020);
        assert_eq!(regs[0].try_end, 0x1040);
        assert_eq!(
            regs[0].landing_pad, 0x1090,
            "JumpTarget names the handler body"
        );
        assert!(!regs[0].is_cleanup);
        assert_eq!(
            regs[0].catches,
            vec![Some("/* managed catch */".to_string())]
        );
    }

    /// A `__finally` scope (`JumpTarget` 0) becomes a cleanup region.
    #[test]
    fn pe_scope_table_finally_is_a_cleanup() {
        let tbl = scope_table(&[(0x1000, 0x1050, 0x1100, 0)]);
        let xdata = pe_xdata_with_handler(0x1234, &tbl);
        let regs = pe_tables(xdata, 0x1000, 0x6000).regions_for(0x1000, 0x200, &|_| None);
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].landing_pad, 0x1100, "the funclet is HandlerAddress");
        assert!(regs[0].is_cleanup);
        assert!(regs[0].catches.is_empty());
    }

    /// A record whose range is outside this function is never adopted — this is
    /// the property that keeps a mis-parse from fencing the structurer.
    #[test]
    fn pe_scope_record_outside_the_function_is_dropped() {
        let tbl = scope_table(&[(0x3000, 0x3020, 0x3080, 0x3090)]);
        let xdata = pe_xdata_with_handler(0x1234, &tbl);
        assert!(
            pe_tables(xdata, 0x1000, 0x6000)
                .regions_for(0x1000, 0x200, &|_| None)
                .is_empty()
        );
    }

    /// `__GSHandlerCheck_EH` prepends a GS cookie before the scope table; the
    /// decoder must skip it and still find the table — and must NOT read the
    /// cookie as a `Count` and fabricate a region from the bytes after it.
    #[test]
    fn pe_gs_cookie_prefix_is_skipped() {
        let mut lang = Vec::new();
        lang.extend_from_slice(&0x40u32.to_le_bytes()); // GS cookie (a plausible Count too)
        lang.extend_from_slice(&scope_table(&[(0x1000, 0x1040, 0x1080, 0x1090)]));
        let xdata = pe_xdata_with_handler(0x1234, &lang);
        let regs = pe_tables(xdata, 0x1000, 0x6000).regions_for(0x1000, 0x200, &|_| None);
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].try_start, 0x1000);
        assert_eq!(regs[0].try_end, 0x1040);
    }

    /// `UNW_FLAG_CHAININFO` records carry no handler of their own; the scope
    /// table lives in the parent they chain to.
    #[test]
    fn pe_chained_unwind_info_is_followed() {
        let mut xdata = Vec::new();
        // Child: Version 1, UNW_FLAG_CHAININFO, no unwind codes, then a nested
        // RUNTIME_FUNCTION whose third ULONG is the parent's UNWIND_INFO RVA.
        xdata.push(0x01 | (UNW_FLAG_CHAININFO << 3));
        xdata.push(0x00);
        xdata.push(0x00);
        xdata.push(0x00);
        xdata.extend_from_slice(&0x1000u32.to_le_bytes()); // BeginAddress
        xdata.extend_from_slice(&0x1200u32.to_le_bytes()); // EndAddress
        xdata.extend_from_slice(&0x6100u32.to_le_bytes()); // parent UNWIND_INFO RVA
        while xdata.len() < 0x100 {
            xdata.push(0); // pad so the parent lands at RVA 0x6100 (byte 0x100)
        }
        let parent =
            pe_xdata_with_handler(0x1234, &scope_table(&[(0x1010, 0x1030, 0x1080, 0x1090)]));
        xdata.extend_from_slice(&parent);
        let regs = pe_tables(xdata, 0x1000, 0x6000).regions_for(0x1000, 0x200, &|_| None);
        assert_eq!(regs.len(), 1, "the handler lives in the chained parent");
        assert_eq!(regs[0].try_start, 0x1010);
        assert_eq!(regs[0].try_end, 0x1030);
    }

    /// A count larger than the bytes present is a mis-parse: refuse it entirely
    /// rather than decode a partial table.
    #[test]
    fn pe_malformed_scope_table_yields_nothing() {
        let mut lang = 64u32.to_le_bytes().to_vec(); // claims 64 records, none present
        lang.extend_from_slice(&[0u8; 8]);
        let xdata = pe_xdata_with_handler(0x1234, &lang);
        assert!(
            pe_tables(xdata, 0x1000, 0x6000)
                .regions_for(0x1000, 0x200, &|_| None)
                .is_empty()
        );
    }

    /// PE32 has no `.pdata` scope tables — the decoder states that, never
    /// invents a region from a 32-bit image.
    #[test]
    fn pe32_has_no_pdata_scope_tables() {
        let tbl = scope_table(&[(0x1020, 0x1040, 0x1080, 0x1090)]);
        let xdata = pe_xdata_with_handler(0x1234, &tbl);
        let tables = EhTables::Pe(PeExceptionTables {
            funcs: vec![(0x1000, 0x6000)],
            xdata,
            xdata_rva: 0x6000,
            ptr_size: 4,
        });
        assert!(tables.regions_for(0x1000, 0x200, &|_| None).is_empty());
    }

    /// The MSVC C++ `FuncInfo` layout (`0x1993052x`) is refused explicitly: it
    /// cannot be read as a scope table, and fabricating regions would be worse
    /// than recovering none.
    #[test]
    fn pe_msvc_cxx_funcinfo_is_refused() {
        let mut lang = MSVC_FUNCINFO_MAGIC.to_le_bytes().to_vec();
        lang.extend_from_slice(&[0u8; 32]);
        let xdata = pe_xdata_with_handler(0x1234, &lang);
        assert!(
            pe_tables(xdata, 0x1000, 0x6000)
                .regions_for(0x1000, 0x200, &|_| None)
                .is_empty()
        );
    }
}
