//! TEMPORARY probes for suspected bugs — do not merge as-is.

use super::*;

#[test]
fn probe_arm64_address_parallelism() {
    // cmp + b.ne => 2 IR insns but only ONE address (suspected desync)
    let words: Vec<u32> = vec![
        0xD2800000, // movz x0, #0
        0xF100001F, // subs xzr, x0, #0
        0x54000041, // b.ne
        0xD2800021, // movz x1, #1
        0xD65F03C0, // ret
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).unwrap();
    println!(
        "PROBE arm64: insns={} addresses={} (equal? {})",
        lifted.insns.len(),
        lifted.addresses.len(),
        lifted.insns.len() == lifted.addresses.len()
    );
    for (i, insn) in lifted.insns.iter().enumerate() {
        println!("  {i}: {insn}");
    }
    assert_eq!(
        lifted.insns.len(),
        lifted.addresses.len(),
        "addresses must be parallel to insns"
    );
}

#[test]
fn probe_arm32_address_parallelism() {
    // cmp + bne (conditional) — suspected desync AND unconditional-jump mislift
    let words: Vec<u32> = vec![
        0xE3A00000, // mov r0, #0
        0xE3500000, // cmp r0, #0
        0x1A000001, // bne +1
        0xE3A01001, // mov r1, #1
        0xE12FFF1E, // bx lr
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1000, 64).unwrap();
    println!(
        "PROBE arm32: insns={} addresses={}",
        lifted.insns.len(),
        lifted.addresses.len()
    );
    for (i, insn) in lifted.insns.iter().enumerate() {
        println!("  {i}: {insn}");
    }
    assert_eq!(
        lifted.insns.len(),
        lifted.addresses.len(),
        "addresses must be parallel to insns"
    );
}

/// Read the iOS `UnityFramework` probe fixture, or `None` when this checkout
/// does not have it.
///
/// The fixture lives under the REPO root's `libs/`, one level above this crate,
/// and is not shipped. A probe must SKIP rather than fail without it — these
/// are manual diagnostics, not gates. `ILR_PROBE_BIN` overrides the path.
fn probe_binary() -> Option<Vec<u8>> {
    let candidates: [Option<String>; 3] = [
        std::env::var("ILR_PROBE_BIN").ok(),
        Some("libs/ios/UnityFramework".to_string()),
        Some("../libs/ios/UnityFramework".to_string()),
    ];
    let path = candidates
        .into_iter()
        .flatten()
        .find(|p| std::path::Path::new(p).is_file())?;
    std::fs::read(&path).ok()
}

/// Decode a raw window of the shipped iOS binary so native mnemonics can be
/// read directly (no dump, no pipeline). Set ILR_PROBE_RVA=0x… to move it.
#[test]
fn probe_decode_window() {
    let rva: u64 = std::env::var("ILR_PROBE_RVA")
        .ok()
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x266d988);
    let len: usize = std::env::var("ILR_PROBE_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x140);
    let Some(data) = probe_binary() else {
        eprintln!("probe_decode_window: no iOS UnityFramework fixture found — skipping");
        return;
    };
    let start = rva as usize;
    let body = &data[start..start + len];
    let (decoded, fails) = super::verify::decode_all_arm64(body, rva);
    println!(
        "PROBE decode @{rva:#x} len={len:#x} insns={} fails={fails}",
        decoded.len()
    );
    let decoder = yaxpeax_arm::armv8::a64::InstDecoder::default();
    use yaxpeax_arch::{Decoder, U8Reader};
    let mut reader = U8Reader::new(body);
    for d in &decoded {
        let raw = match decoder.decode(&mut reader) {
            Ok(i) => format!("opcode={:?} operands={:?}", i.opcode, i.operands),
            Err(e) => format!("DECODE-ERR {e:?}"),
        };
        println!(
            "  {:#010x}  {:<28} {:?}\n      {raw}",
            d.addr, d.mnemonic, d.class
        );
    }
}

#[test]
fn probe_vec_shapes() {
    let words: Vec<u32> = vec![
        0x0c230000, 0x0c670b01, 0x0d409123, 0x0d40c901, 0x0d009061, 0x2e25dc42, 0x2e25dce5,
        0x2e22dc21, 0x2e22dca2, 0x2e21dc21, 0x2e23dc42, 0x0e0404c6, 0x0e040610, 0x0e040652,
        0x0e040610,
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let decoder = yaxpeax_arm::armv8::a64::InstDecoder::default();
    let mut reader = yaxpeax_arch::U8Reader::new(&bytes);
    for (wi, w) in words.iter().enumerate() {
        use yaxpeax_arch::Decoder;
        match decoder.decode(&mut reader) {
            Ok(i) => println!(
                "PROBE-V {:08x}: {}  | opcode={:?} operands={:?}",
                w, i, i.opcode, i.operands
            ),
            Err(e) => println!("PROBE-V {:08x}: DECODE-ERR {:?}", w, e),
        }
        let _ = wi;
    }
}

/// Diagnostics for the `cond-branch without real compare` class.
///
/// For each address in `ILR_CONDLIST` (one `0x…` per line, from the verifier),
/// decode the native window ending at that branch and report the opcode
/// immediately BEFORE it — i.e. what sits between the flag-setting compare
/// and the branch that consumes it. The tally says exactly which opcodes the
/// lifter must NOT invalidate `pending_cmp` on.
#[test]
fn probe_cond_gap() {
    let Ok(list) = std::env::var("ILR_CONDLIST") else {
        return;
    };
    let Ok(text) = std::fs::read_to_string(&list) else {
        return;
    };
    let Some(data) = probe_binary() else { return };
    let decoder = yaxpeax_arm::armv8::a64::InstDecoder::default();
    use yaxpeax_arch::{Decoder, U8Reader};
    let mut tally: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut with_cmp: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut n = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(addr) = u64::from_str_radix(line.trim_start_matches("0x"), 16) else {
            continue;
        };
        // window: 16 instructions (64 bytes) ending AT the branch
        let start = addr.saturating_sub(64);
        let body = &data[start as usize..addr as usize + 4];
        let mut reader = U8Reader::new(body);
        let mut prev: Option<String> = None;
        let mut last_cmp_gap: Option<usize> = None;
        let mut idx = 0usize;
        let mut branch_at: Option<usize> = None;
        while let Ok(i) = decoder.decode(&mut reader) {
            let this_addr = start + (idx as u64) * 4;
            if this_addr == addr {
                branch_at = Some(idx);
                break;
            }
            if matches!(
                i.opcode,
                yaxpeax_arm::armv8::a64::Opcode::SUBS
                    | yaxpeax_arm::armv8::a64::Opcode::ADDS
                    | yaxpeax_arm::armv8::a64::Opcode::ANDS
                    | yaxpeax_arm::armv8::a64::Opcode::BICS
            ) {
                last_cmp_gap = Some(idx);
            }
            prev = Some(format!("{:?}", i.opcode));
            idx += 1;
        }
        if branch_at.is_none() {
            continue;
        }
        n += 1;
        if let Some(k) = prev {
            *tally.entry(k).or_default() += 1;
        }
        let gap = last_cmp_gap
            .map(|c| branch_at.unwrap() - c - 1)
            .unwrap_or(usize::MAX);
        *with_cmp.entry(gap.min(99)).or_default() += 1;
    }
    println!("PROBE-COND total={n}");
    println!("--- opcode immediately before the branch ---");
    for (k, v) in tally.iter().rev().take(40) {
        println!("  {v:>6}  {k}");
    }
    println!("--- distance back to nearest CMP/CMN/SUBS/ADDS (99 = none) ---");
    for (k, v) in with_cmp.iter() {
        println!("  {v:>6}  gap={k}");
    }
}

/// Decode hex words from ILR_PROBE_WORDS (comma separated) and print the
/// opcode + operands + the lifted IR, so candidate encodings for new arms can
/// be grounded against the real decoder instead of guessed at.
#[test]
fn probe_words() {
    let raw = std::env::var("ILR_PROBE_WORDS").unwrap_or_default();
    if raw.trim().is_empty() {
        return;
    }
    let words: Vec<u32> = raw
        .split(',')
        .filter_map(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .collect();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let decoder = yaxpeax_arm::armv8::a64::InstDecoder::default();
    use yaxpeax_arch::{Decoder, U8Reader};
    let mut reader = U8Reader::new(&bytes);
    for w in &words {
        match decoder.decode(&mut reader) {
            Ok(i) => println!(
                "WORD {w:#010x}  {i}  opcode={:?}  operands={:?}",
                i.opcode, i.operands
            ),
            Err(e) => println!("WORD {w:#010x}  DECODE-ERR {e:?}"),
        }
    }
    let lifted = lift::lift_arm64(&bytes, 0x1000, bytes.len(), None).unwrap();
    for (i, insn) in lifted.insns.iter().enumerate() {
        println!("  IR {i}: {insn}");
    }
}

/// Reproduce `rewrite_pinned_copies` on the real `FindMatchingElement`
/// prologue with a controlled signature, so the ABI pin rewriting can be
/// bisected without a 5-minute full dump.
#[test]
fn probe_pinned_copies() {
    use ir::Operand as Op2;
    let words: [u32; 16] = [
        0xD10283FF, 0xA9046FFC, 0xA90567FA, 0xA9065FF8, 0xA90757F6, 0xA9084FF4, 0xA9097BFD,
        0x910243FD, 0xAA0703F7, 0xAA0603F5, 0xAA0403F9, 0xAA0303F4, 0xAA0203F6, 0xAA0103FB,
        0xF9000FE0, 0xB001C348,
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    for nparams in [3usize, 6, 7] {
        for is_static in [false, true] {
            let lifted = lift::lift_arm64(&bytes, 0x1f4f660, 256, None).unwrap();
            let mut insns = lifted.insns;
            let seed = locals::SeedInfo {
                param_types: (0..nparams)
                    .map(|_| ManagedType::Primitive(PrimKind::I4))
                    .collect(),
                ret_type: ManagedType::Primitive(PrimKind::I4),
                is_static,
                declaring: None,
                param_names: Vec::new(),
                param_comp_names: Vec::new(),
                arm32_param_words: Vec::new(),
                is_arm32: false,
                ret_hidden_buffer: false,
            };
            locals::rewrite_pinned_copies(&mut insns, &seed);
            let head: Vec<String> = insns
                .iter()
                .take(20)
                .map(|i| match i.operands.get(1) {
                    Some(Op2::Register(r)) => format!(
                        "{}<={r}",
                        match i.operands.first() {
                            Some(Op2::Register(d)) => d.clone(),
                            _ => "?".into(),
                        }
                    ),
                    _ => String::new(),
                })
                .filter(|s| !s.is_empty())
                .collect();
            println!("PROBE nparams={nparams} static={is_static} -> {head:?}");
        }
    }
}
