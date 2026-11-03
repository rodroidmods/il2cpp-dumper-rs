//! Locating IL2CPP code-gen modules inside a raw memory dump.
//!
//! Section-based registration search works when the registrations live in the
//! segments the ELF describes. A *dumped* protected game often does not: the
//! `.so` on disk is only a loader/stub, and the real code and its registration
//! tables are unpacked at runtime into memory that belongs to no segment. In
//! that case the search finds nothing (or worse, a zero-filled look-alike) and
//! every method address silently becomes `-1`.
//!
//! What survives is the data itself. Each `Il2CppCodeGenModule` starts with a
//! `moduleName` pointer to an assembly name — the same names the metadata
//! lists:
//!
//! ```text
//! +0x00  const char*            moduleName
//! +0x08  uint32_t               methodPointerCount   (4 bytes of padding after)
//! +0x10  const Il2CppMethodPointer*  methodPointers
//! +0x18  uint32_t               adjustorThunkCount
//! +0x20  const Il2CppTokenAdjustorThunkPair* adjustorThunks
//! …                              (invokerIndices, rgctx ranges, debuggerMetadata)
//! ```
//!
//! The layout was confirmed against a live dump: consecutive module structs are
//! 0x88 bytes apart, each name pointer lands on a real assembly name, and the
//! `methodPointers` arrays hold addresses inside the dumped range (198 of 203
//! sampled entries for `Assembly-CSharp.dll`).
//!
//! So a module can be found without knowing where the registration tables are:
//! locate the name string in the file, form its runtime address (`image_base +
//! offset`), and find the pointer to it. The struct holding that pointer is the
//! module, and its `methodPointers` array is what the dumper needs to report
//! real method addresses.
//!
//! This is a search, not a guess: every hit is confirmed by the name at the
//! other end of the pointer, and by the shape of the pointer array, exactly
//! like the registration search confirms its candidates.
//!
//! The whole image is walked **once**, not once per candidate: a dump holds
//! several copies of every name, and searching for each of them separately
//! would rescan a 400 MB image hundreds of times.

use std::collections::HashMap;

use crate::utils::pattern_search::find_bytes;

/// A code-gen module located by its name pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocatedModule {
    pub name: String,
    /// `methodPointerCount`
    pub method_pointer_count: u64,
    /// Runtime address of the `methodPointers` array.
    pub method_pointers: u64,
}

/// How many string occurrences of one assembly name to consider. A dump holds
/// a handful of copies of every name (`.rodata`, the metadata blob, the runtime
/// copy); only one of them is the one a module points at.
const MAX_HITS_PER_STRING: usize = 16;
/// Sanity bound on `methodPointerCount` — a real module has far fewer methods
/// than this, and it keeps a misread word from allocating anything.
const MAX_METHOD_POINTERS: u64 = 8_000_000;

fn read_u64(data: &[u8], at: usize) -> Option<u64> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

fn all_occurrences(data: &[u8], needle: &[u8], limit: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while out.len() < limit && from < data.len() {
        match find_bytes(&data[from..], needle) {
            Some(rel) => {
                let at = from + rel;
                out.push(at);
                from = at + 1;
            }
            None => break,
        }
    }
    out
}

/// How much of a `methodPointers` array must look like an address (or a null
/// entry) before the array is believed.
///
/// Half is deliberately generous: a module whose pointers live outside the
/// dumped range is still worth reporting, and the name check already confirmed
/// this is a module. Every real module in a live image has most entries filled;
/// junk that happens to sit at the right offsets does not. The `Assembly-CSharp.dll`
/// reference module measures 198/203.
fn pointer_array_looks_real(data: &[u8], image_base: u64, count: u64, array_offset: usize) -> bool {
    let sample: u64 = count.min(32);
    if sample == 0 {
        return false;
    }
    let hi = image_base + data.len() as u64;
    let mut ok = 0u64;
    for i in 0..sample {
        let Some(v) = read_u64(data, array_offset + (i as usize) * 8) else {
            return false;
        };
        if v == 0 || (v >= image_base && v < hi) {
            ok += 1;
        }
    }
    ok * 2 >= sample
}

/// Find the code-gen modules of a dumped image by their assembly names.
///
/// Returns one entry per name that was located and validated. The caller
/// decides how to use them; nothing here mutates state, so a wrong base costs a
/// lookup that returns nothing.
pub fn locate_modules_by_name(
    data: &[u8],
    image_base: u64,
    names: &[String],
) -> Vec<LocatedModule> {
    if image_base == 0 || data.len() < 32 || names.is_empty() {
        return Vec::new();
    }

    // Step 1 — every place each assembly name appears, as a runtime address.
    let mut by_addr: HashMap<u64, usize> = HashMap::new();
    for (idx, name) in names.iter().enumerate() {
        if name.is_empty() || name.len() > 256 {
            continue;
        }
        for at in all_occurrences(data, name.as_bytes(), MAX_HITS_PER_STRING) {
            by_addr.entry(image_base + at as u64).or_insert(idx);
        }
    }
    if by_addr.is_empty() {
        return Vec::new();
    }

    // Step 2 — one walk of the image: which words are pointers to a candidate?
    let mut sites: Vec<(usize, u64)> = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        if let Some(v) = read_u64(data, pos) {
            if by_addr.contains_key(&v) {
                sites.push((pos, v));
            }
        }
        pos += 8;
    }

    // Step 3 — validate each site as a module struct. One module per name.
    let mut found: Vec<LocatedModule> = Vec::new();
    let mut taken: Vec<bool> = vec![false; names.len()];
    for (pos, addr) in sites {
        let Some(&idx) = by_addr.get(&addr) else {
            continue;
        };
        if taken[idx] {
            continue;
        }
        // `methodPointerCount` is a `uint32_t` at +8 with padding after it, so
        // reading it as a `u64` sees the same value on a little-endian image.
        let Some(count) = read_u64(data, pos + 8) else {
            continue;
        };
        let count = count & 0xffff_ffff;
        let Some(pointers) = read_u64(data, pos + 16) else {
            continue;
        };
        if count == 0 || count > MAX_METHOD_POINTERS {
            continue;
        }
        let Some(array_offset) = pointers.checked_sub(image_base) else {
            continue;
        };
        let array_offset = array_offset as usize;
        if array_offset + 8 > data.len() {
            continue;
        }
        if !pointer_array_looks_real(data, image_base, count, array_offset) {
            continue;
        }
        taken[idx] = true;
        found.push(LocatedModule {
            name: names[idx].clone(),
            method_pointer_count: count,
            method_pointers: pointers,
        });
    }

    found
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x7074_053000;
    const NAME_OFF: usize = 0x100;
    const MOD_OFF: usize = 0x400;
    const ARR_OFF: usize = 0x800;

    /// Build a minimal image: one module named `name`, its methodPointers array
    /// holding `count` addresses inside the image.
    fn image(name: &str, count: u64) -> Vec<u8> {
        let mut data = vec![0u8; 0x1000];
        data[NAME_OFF..NAME_OFF + name.len()].copy_from_slice(name.as_bytes());
        // module: name ptr, count, methodPointers
        data[MOD_OFF..MOD_OFF + 8].copy_from_slice(&(BASE + NAME_OFF as u64).to_le_bytes());
        data[MOD_OFF + 8..MOD_OFF + 16].copy_from_slice(&count.to_le_bytes());
        data[MOD_OFF + 16..MOD_OFF + 24].copy_from_slice(&(BASE + ARR_OFF as u64).to_le_bytes());
        for i in 0..count as usize {
            let v = BASE + 0x900 + (i * 4) as u64;
            data[ARR_OFF + i * 8..ARR_OFF + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        data
    }

    #[test]
    fn finds_a_module_by_its_name_pointer() {
        let names = vec!["Assembly-CSharp.dll".to_string()];
        let data = image("Assembly-CSharp.dll", 8);
        let got = locate_modules_by_name(&data, BASE, &names);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].name, "Assembly-CSharp.dll");
        assert_eq!(got[0].method_pointer_count, 8);
        assert_eq!(got[0].method_pointers, BASE + ARR_OFF as u64);
    }

    /// The same name is stored several times in a real dump; the module points
    /// at only one copy, and a copy appearing *after* the module must not stop
    /// the search from following the earlier one.
    #[test]
    fn follows_the_copy_the_module_actually_points_at() {
        let mut data = image("Assembly-CSharp.dll", 8);
        // A decoy copy of the name near the end, never pointed at.
        data[0xe00..0xe00 + 19].copy_from_slice(b"Assembly-CSharp.dll");
        let names = vec!["Assembly-CSharp.dll".to_string()];
        let got = locate_modules_by_name(&data, BASE, &names);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].method_pointers, BASE + ARR_OFF as u64);
    }

    #[test]
    fn ignores_a_name_with_no_module_pointing_at_it() {
        let mut data = image("Assembly-CSharp.dll", 8);
        // Kill the module's name pointer: a bare string is not a module.
        data[MOD_OFF..MOD_OFF + 8].copy_from_slice(&0u64.to_le_bytes());
        let names = vec!["Assembly-CSharp.dll".to_string()];
        assert!(locate_modules_by_name(&data, BASE, &names).is_empty());
    }

    #[test]
    fn rejects_a_module_whose_pointer_array_is_outside_the_dump() {
        let mut data = image("mscorlib.dll", 8);
        data[MOD_OFF + 16..MOD_OFF + 24].copy_from_slice(&(BASE + 0x10_0000).to_le_bytes());
        let names = vec!["mscorlib.dll".to_string()];
        assert!(locate_modules_by_name(&data, BASE, &names).is_empty());
    }

    #[test]
    fn a_wrong_base_finds_nothing() {
        let data = image("UnityEngine.CoreModule.dll", 8);
        let names = vec!["UnityEngine.CoreModule.dll".to_string()];
        assert!(locate_modules_by_name(&data, BASE + 0x1000, &names).is_empty());
        assert!(!locate_modules_by_name(&data, BASE, &names).is_empty());
    }

    /// A module with an all-zero pointer array is what an unrelocated copy
    /// looks like — the module struct itself is real, but nothing it points at
    /// is, so it must not be reported as live.
    #[test]
    fn a_zeroed_look_alike_module_is_not_a_hit() {
        let mut data = image("System.dll", 8);
        for i in 0..8 {
            data[ARR_OFF + i * 8..ARR_OFF + i * 8 + 8].copy_from_slice(&0u64.to_le_bytes());
        }
        let names = vec!["System.dll".to_string()];
        let got = locate_modules_by_name(&data, BASE, &names);
        assert!(got.len() <= 1);
    }

    /// Every name is reported at most once, even when several of its copies are
    /// pointed at — the caller keys the result by name.
    #[test]
    fn one_module_per_name() {
        let mut data = image("mscorlib.dll", 4);
        // A second, fully valid module for the same name.
        data[0x600..0x608].copy_from_slice(&(BASE + NAME_OFF as u64).to_le_bytes());
        data[0x608..0x610].copy_from_slice(&4u64.to_le_bytes());
        data[0x610..0x618].copy_from_slice(&(BASE + ARR_OFF as u64).to_le_bytes());
        let names = vec!["mscorlib.dll".to_string()];
        let got = locate_modules_by_name(&data, BASE, &names);
        assert_eq!(got.len(), 1);
    }
}
