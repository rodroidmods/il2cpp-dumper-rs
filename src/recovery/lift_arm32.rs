//! ARM32 (A32/ARMv7) lifter → recovery IR.
//!
//! yaxpeax-arm's armv7 decoder exposes operands most reliably via Display;
//! we parse the canonical mnemonic/operand strings. Every bit layout used by
//! the manual VFP decoder below was verified against keystone+capstone ground
//! truth during development (never guessed).
//!
//! Register mapping onto shared virtual names:
//!   r0–r3 → x0–x3 (args + return), r4–r12 → g4–g12, sp → sp, lr → g14
//! VFP single/double → `vs{n}` / `vd{n}` — the SAME names the arm64 lifter
//! uses, so type propagation, stack analysis and the `bind_fp_returns` pass
//! work on arm32 without changes.
//!
//! Covered subset:
//!   - data processing: mov/movw/movt, mvn, add/sub/rsb/and/orr/eor/bic,
//!     shift aliases (lsl/lsr/asr/ror incl. register-shifted operands),
//!     mul/mla/umull, clz, cmp/tst (+pending-flag compare fold), the
//!     flag-setting `s` variants (adds/cmps/movs/...), hex immediates
//!   - memory: ldr/str (+b/h), offset and register-index forms
//!   - control: push/pop, b/bl/blx/bx (+condition codes), ldm/stm (sp shifts)
//!   - VFP scalar: vadd/vsub/vmul/vdiv/vmla/vmls, vabs/vneg/vsqrt,
//!     vcmp (+ #0.0), vcvt (int↔float, f32↔f64), vmov (immediate, core↔single),
//!     vldr/vstr, vpush/vpop
//!
//! The lifter never aborts on a single bad word: undecodable instructions
//! (data, padding, NEON/vector, exotic coprocessor) are skipped — their
//! destination simply stays undefined (= `Unknown` downstream).

use super::ir::{
    ArithKind, CompareKind, Conversion, FloatArithKind, Insn, Op, Operand, SelectVariant,
};
use std::collections::{BTreeSet, HashMap};
use yaxpeax_arch::{Decoder, LengthedInstruction, U8Reader};
use yaxpeax_arm::armv7::{InstDecoder as A32Decoder, Instruction as A32Insn};

pub struct LiftResult {
    pub insns: Vec<Insn>,
    pub addresses: Vec<u64>,
}

fn reg(name: &str) -> Operand {
    match name {
        "r0" => Operand::reg("x0"),
        "r1" => Operand::reg("x1"),
        "r2" => Operand::reg("x2"),
        "r3" => Operand::reg("x3"),
        "sp" | "r13" => Operand::reg("sp"),
        "lr" | "r14" => Operand::reg("g14"),
        "pc" | "r15" => Operand::reg("g15"),
        // EABI register ALIASES. The decoder prints these instead of `rN`
        // wherever the assembler spelling is conventional (`sb` = r9, `sl` =
        // r10, `fp` = r11, `ip` = r12), and unmapped they became PHANTOM
        // registers — names no def ever wrote, so every read through one turned
        // the dependent expression into an undefined local. They must map onto
        // exactly the names the `rN` spelling produces, or the same register
        // reaches the renderer under two identities.
        "sb" | "r9" => Operand::reg("x9"),
        "sl" | "r10" => Operand::reg("x10"),
        "fp" | "r11" => Operand::reg("x11"),
        "ip" | "r12" => Operand::reg("x12"),
        other => {
            if let Some(num) = other.strip_prefix('r') {
                match num.parse::<u8>() {
                    // r4–r12 general-purpose → xN (arm64 naming parity; csharp
                    // already treats xN as ordinary GPR locals).
                    Ok(n) if n <= 12 => Operand::reg(format!("x{n}")),
                    Ok(n) => Operand::reg(format!("g{n}")),
                    Err(_) => Operand::Unknown,
                }
            } else {
                Operand::reg(other)
            }
        }
    }
}

/// VFP single register `s{n}` → shared `vs{n}` name.
fn sreg(n: u32) -> Operand {
    Operand::reg(format!("vs{n}"))
}

/// VFP double register `d{n}` → shared `vd{n}` name.
fn dreg(n: u32) -> Operand {
    Operand::reg(format!("vd{n}"))
}

/// Register name of an operand (for fp_written tracking).
fn reg_name(op: &Operand) -> String {
    match op {
        Operand::Register(n) => n.clone(),
        _ => String::new(),
    }
}

fn cond_to_kind(cc: &str) -> CompareKind {
    match cc {
        "eq" => CompareKind::Equal,
        "ne" => CompareKind::NotEqual,
        "gt" => CompareKind::GreaterThan,
        "lt" => CompareKind::LessThan,
        "ge" => CompareKind::GreaterOrEqual,
        "le" => CompareKind::LessOrEqual,
        "hi" => CompareKind::GreaterThan,
        "ls" => CompareKind::LessOrEqual,
        "hs" | "cs" => CompareKind::GreaterOrEqual,
        "lo" | "cc" => CompareKind::LessThan,
        // Sign / overflow conditions. `bmi`/`bpl`/`bvs`/`bvc` carry the SIGN
        // and OVERFLOW tests — the compiler emits them for every signed
        // overflow check and for every "did this 64-bit add wrap" test, so
        // mapping them onto a signed ordering was the closest available
        // approximation, and mapping them to `NotEqual` (the old catch-all)
        // was not an approximation at all: it invented a comparison that
        // tested neither flag.
        "mi" => CompareKind::LessThan,
        "pl" => CompareKind::GreaterOrEqual,
        "vs" | "vc" => CompareKind::NotEqual,
        "al" => CompareKind::NotEqual,
        _ => CompareKind::NotEqual,
    }
}

/// The NEGATION of a condition kind.
///
/// A predicated instruction that has no branch target of its own — a
/// conditional RETURN (`popeq {…, pc}`, `bxne lr`) — is lowered as "branch to
/// the fall-through when the condition does NOT hold", which is what the
/// machine does (it simply skips the instruction). Unsigned-only conditions
/// reuse the same signed approximation `cond_to_kind` already applies, so the
/// negation stays consistent with the guard it inverts.
fn invert_kind(k: CompareKind) -> CompareKind {
    match k {
        CompareKind::Equal => CompareKind::NotEqual,
        CompareKind::NotEqual => CompareKind::Equal,
        CompareKind::LessThan => CompareKind::GreaterOrEqual,
        CompareKind::GreaterOrEqual => CompareKind::LessThan,
        CompareKind::GreaterThan => CompareKind::LessOrEqual,
        CompareKind::LessOrEqual => CompareKind::GreaterThan,
    }
}

/// Emit the "skip this predicated instruction when its condition fails" guard:
/// a compare of the pending flags with the NEGATED condition, and a branch to
/// the instruction that follows. Returns true when a guard was emitted.
///
/// Dropping such a condition is not a cosmetic loss: an unconditional return in
/// its place ENDS the linear path, so every instruction reachable only on the
/// other side of the test becomes unreachable and vanishes. That is exactly how
/// `ReadComment()` @0x12F67B0 (AArch32) lost its whole `while` body while the
/// loop itself stayed.
fn emit_predication_guard(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    scratch: &mut Scratch,
    state: Option<&PendingCompare>,
    cond: &str,
    addr: u64,
) -> bool {
    if cond.is_empty() {
        return false;
    }
    let s = materialize_pending(
        insns,
        addresses,
        scratch,
        state,
        invert_kind(cond_to_kind(cond)),
        addr,
    );
    insns.push(Insn::new(Op::CondJump((addr + 4) as usize), vec![s], addr));
    true
}

/// The condition codes A32 can append to ANY instruction.
///
/// The list decides how a mnemonic is SPLIT, so a missing code is not a
/// cosmetic gap: `addmi`, `rsbmi`, `ldrdmi`, `ldrbtmi`, `movvc` all ended up
/// with their condition glued to the mnemonic and matched no arm — 41 real
/// instructions in one ARM32 build before counting every other `mi`/`pl`/`vs`
/// form. Order matters: `hs`/`lo` precede `cs`/`cc` only because the longer
/// spellings are tried first; both map to the same flag.
pub(crate) const A32_CONDITION_CODES: [&str; 17] = [
    "eq", "ne", "gt", "lt", "ge", "le", "hi", "ls", "hs", "lo", "cs", "cc", "mi", "pl", "vs", "vc",
    "al",
];

/// The CARRY (or borrow) produced by a flag-setting instruction.
///
/// `ADC`/`SBC`/`RSC` are not ordinary two-operand arithmetic: they consume the
/// carry the PREVIOUS flag-setting operation left in APSR. A32 spells 64-bit
/// addition as `ADDS lo` / `ADC hi` and 64-bit subtraction as `SUBS` / `SBC`,
/// and every `long`/`ulong`/`DateTime`/`TimeSpan`/big-integer operation in the
/// binary is built from those pairs — so without a carry value an entire class
/// of arithmetic lost its meaning: `adc` was the single densest unmodelled
/// mnemonic in the Android build (439 sites, more than twice the next).
///
/// The operands are the ones the flag-setting instruction actually combined,
/// NOT `result vs 0` (which is what the condition uses), and `at` pins them to
/// their values at that point in the listing for the same reason
/// `PendingCompare::at` exists.
#[derive(Clone)]
struct CarrySource {
    /// True for SUB/CMP (the carry is `lhs >= rhs`, A32's NOT-borrow), false
    /// for ADD/CMN (`lhs + rhs` overflowed).
    subtract: bool,
    lhs: Operand,
    rhs: Operand,
    at: usize,
}

/// Parse an integer literal as yaxpeax prints it: decimal, `0x…`, `-0x…`.
fn parse_int(s: &str) -> Option<i64> {
    let s = s.trim();
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let v = if let Some(hex) = rest.strip_prefix("0x") {
        i64::from_str_radix(hex, 16).ok()
    } else {
        rest.parse::<i64>().ok()
    }?;
    Some(if neg { -v } else { v })
}

/// Split "mnemonic rest" into (base_mnemonic, condition, rest).
/// Handles ARM condition suffixes (`bne`, `moveq`) AND the yaxpeax
/// flag-setting `s` suffix (`adds`, `cmps`, `movs`, `tsts`) that the original
/// matcher never saw — `cmp` prints as `cmps r0, r5, r2`.
/// Split "mnemonic rest" into (base_mnemonic, condition, rest).
///
/// `pub(crate)` so the VERIFIER classifies ARM32 with the same splitter the
/// lifter dispatches on — see `verify::classify_arm32`. One implementation means
/// the oracle and the engine cannot disagree about what an instruction is.
pub(crate) fn split(text: &str) -> (&str, &str, &str) {
    let trimmed = text.trim();
    let space = trimmed.find(' ').unwrap_or(trimmed.len());
    let head = &trimmed[..space];
    let rest = trimmed[space..].trim_start();
    // A mnemonic that IS itself modelled wins over the condition reading.
    //
    // The two readings genuinely collide, and the collision is not rare:
    //
    //   * `mls` (multiply long subtract) ends in the condition code `ls`
    //   * `teq` ends in `eq`, `adcs`/`sbcs`/`bics` end in `cs`
    //
    // Splitting those as `m`+`ls`, `t`+`eq`, `ad`+`cs`… looked up mnemonics
    // that match NO arm, so each of those instructions silently fell through
    // to the generic fallback: `mls` lost its multiply entirely (`__m(…)`),
    // and every `teq`/`adcs`/`sbcs`/`bics` lost its destination. Checking the
    // head verbatim FIRST keeps `bls` correct (it is not itself a mnemonic, so
    // it still reads as `b`+`ls`) while giving the real mnemonics their arm.
    if is_modelled_mnemonic(head) {
        return (head, "", rest);
    }
    // condition codes next so `bls` stays b+ls, never "bl"+"s"
    for cc in A32_CONDITION_CODES {
        if let Some(base) = head.strip_suffix(cc) {
            // The flag-setting `s` must be normalized on the CONDITIONED form
            // too. `tsteq`/`cmpne`/`addseq` reach this branch with the base
            // still carrying it, and returning it verbatim looked up the
            // mnemonics `tsts`/`cmps`/`adds`, which match NO arm — so every
            // PREDICATED instruction (Thumb-2 IT blocks are pervasive in the
            // ARMv7 IL2CPP builds) fell through to the generic fallback and
            // lost its semantics. It is why the generated C# contains
            // `____tsts(…)` / `____cmps(…)`: 2,975 real `tst`/`cmp`
            // conditions on the Android binary alone, before counting every
            // predicated add/sub/and/orr.
            // …and the conditioned reading only counts when what is left is a
            // mnemonic the dispatch models. Without this test, `mls` would
            // still be accepted as `m`+`ls` before the whole-head check below
            // ever ran, and `adcs` would be read as `ad`+`cs`.
            let base = normalize_flag_suffix(base);
            if is_modelled_mnemonic(base) {
                return (base, cc, rest);
            }
        }
    }
    (normalize_flag_suffix(head), "", rest)
}

/// The A32 mnemonics this lifter RECOGNISES.
///
/// Used to disambiguate `split` and to normalize the flag-setting `s`; a
/// mnemonic missing here costs nothing beyond the old behaviour (its
/// conditioned reading is tried instead), while a mnemonic wrongly ABSENT
/// silently loses its arm — which is exactly what `mls`, `teq`, `adcs`, `sbcs`
/// and `bics` were doing. Two entries therefore have no arm of their own:
///
///  * the unprivileged load/store family — handled by rewriting the mnemonic in
///    the memory arm (`canonical_mem_mnemonic`);
///  * the `smlal` family, which is listed so its CONDITION still splits off.
///    Its halfword forms are now decoded from the raw word ahead of yaxpeax
///    (`dsp_mul`), so these entries matter only for the bare `smlal`/`umlal`
///    that still arrive through the dispatch.
fn is_modelled_mnemonic(mne: &str) -> bool {
    matches!(
        mne,
        "b" | "bl"
            | "blx"
            | "nop"
            | "nop.w"
            | "dmb"
            | "dsb"
            | "isb"
            | "clrex"
            | "yield"
            | "sev"
            | "wfe"
            | "wfi"
            | "mov"
            | "mov.w"
            | "movt"
            | "mvn"
            | "add"
            | "add.w"
            | "sub"
            | "sub.w"
            | "rsb"
            | "adc"
            | "sbc"
            | "rsc"
            | "and"
            | "orr"
            | "eor"
            | "bic"
            | "mls"
            | "mla"
            | "mul"
            | "umull"
            | "umlal"
            | "smull"
            | "smlal"
            | "smulbb"
            | "smulbt"
            | "smultb"
            | "smultt"
            | "smulwb"
            | "smulwt"
            | "cmp"
            | "cmp.w"
            | "cmn"
            | "tst"
            | "tst.w"
            | "teq"
            | "clz"
            | "sdiv"
            | "udiv"
            | "lsl"
            | "lsr"
            | "asr"
            | "ror"
            | "rrx"
            | "ldr"
            | "ldr.w"
            | "ldrb"
            | "ldrh"
            | "ldrsb"
            | "ldrsh"
            | "str"
            | "str.w"
            | "strb"
            | "strh"
            | "ldrbt"
            | "ldrht"
            | "ldrsbt"
            | "ldrsht"
            | "ldrt"
            | "strbt"
            | "strht"
            | "strt"
            | "smlalbb"
            | "smlalbt"
            | "smlaltb"
            | "smlaltt"
            | "smlawb"
            | "smlawt"
            | "ldrd"
            | "strd"
            | "ldm"
            | "ldmia"
            | "ldmib"
            | "ldmda"
            | "ldmdb"
            | "stm"
            | "stmia"
            | "stmib"
            | "stmda"
            | "stmdb"
            | "push"
            | "pop"
            | "push.w"
            | "pop.w"
            | "bx"
            | "cbz"
            | "cbnz"
            | "tbz"
            | "tbnz"
            | "adr"
            | "bfi"
            | "bfc"
            | "ubfx"
            | "sbfx"
            | "ssat"
            | "usat"
            | "sxtb"
            | "sxth"
            | "uxtb"
            | "uxth"
            | "sxtb16"
            | "uxtb16"
            | "rev"
            | "rev16"
            | "revsh"
            | "rbit"
            | "neg"
            | "ldrex"
            | "ldrexb"
            | "ldrexh"
            | "ldrexd"
            | "strex"
            | "strexb"
            | "strexh"
            | "strexd"
            | "mrs"
            | "msr"
            | "vmsr"
            | "vmov"
            | "vldr"
            | "vstr"
            | "vpush"
            | "vpop"
            | "vadd"
            | "vsub"
            | "vmul"
            | "vdiv"
            | "vcmp"
            | "vcvt"
            | "vabs"
            | "vneg"
            | "vmov.f32"
            | "vmov.f64"
            | "it"
            // ---- signed multiplier family -----------------------------------
            // Listed so the CONDITION splits off and the dispatch reaches the
            // real arm below. Without the entry, `smlaltt` stayed glued to its
            // condition (`smlalteq`), matched no arm, and fell through the
            // generic fallback — 1,200+ sites on the Android binary alone.
            // (`smlalbb`/`smlalbt`/`smlaltb`/`smlaltt` were already listed for
            // this reason but had no arm to reach; that is fixed now.)
            | "smlabb"
            | "smlabt"
            | "smlatb"
            | "smlatt"
            // ---- atomics / status / saturation / coprocessor ---------------
            // Same reasoning: the condition must split off so the arm is
            // reachable, and `swpeq`/`qaddeq`/`smceq` must not be read as
            // single mnemonics.
            | "swp"
            | "swpb"
            | "qadd"
            | "qsub"
            | "qdadd"
            | "qdsub"
            | "mrc2"
            | "mcr2"
            | "cdp2"
            | "smc"
            | "umaal"
            | "invalid"
    )
}

/// Strip the flag-setting `s` suffix from a data-processing mnemonic, leaving
/// every other mnemonic untouched (`bls` must stay `bls`, not `bl`).
///
/// The test is `is_modelled_mnemonic` rather than a hand-written list, because
/// the hand-written list had already drifted: `mvn` and `adr` were both missing
/// from it, so `mvns r0, r1` and `adrs r0, #x` kept the `s` glued to the
/// mnemonic, matched no arm, and fell to the generic fallback. On one ARM32
/// build that was 2,881 `mvns`+condition sites — the single densest unmodelled
/// mnemonic — plus every `adrs`. Consulting the same table the dispatch uses
/// means the two can no longer disagree.
fn normalize_flag_suffix(mne: &str) -> &str {
    if let Some(base) = mne.strip_suffix('s') {
        if is_modelled_mnemonic(base) {
            return base;
        }
    }
    mne
}

/// The canonical name of an UNPRIVILEGED load/store.
///
/// A32 spells the unprivileged forms with a trailing `t` (`LDRBT`, `LDRHT`,
/// `LDRSBT`, `LDRSHT`, `STRBT`, `STRHT`, and word forms `LDRT`/`STRT`). The
/// privilege check is the ONLY difference from the ordinary instruction — the
/// addressing mode and the data movement are identical — and managed code is
/// user-mode anyway, so they are lifted as their ordinary counterparts instead
/// of being dropped. On one ARM32 build this family was ~1,700 sites.
fn canonical_mem_mnemonic(mne: &str) -> &str {
    match mne {
        "ldrbt" => "ldrb",
        "ldrht" => "ldrh",
        "ldrsbt" => "ldrsb",
        "ldrsht" => "ldrsh",
        "ldrt" => "ldr",
        "strbt" => "strb",
        "strht" => "strh",
        "strt" => "str",
        other => other,
    }
}

/// True when the ORIGINAL mnemonic (before strip) carried the flag-setting
/// `s` suffix: `adds r0, r0, #1` feeds an upcoming `b<cc>` through the
/// pending-compare fold just like `cmp`.
fn sets_flags(mne: &str, text: &str) -> bool {
    text.starts_with(mne) && text.as_bytes().get(mne.len()) == Some(&b's')
}

struct PendingCompare {
    lhs: Operand,
    rhs: Operand,
    /// Listing index at which these operand NAMES were captured.
    ///
    /// A register name is not a value. The compare is materialized at its
    /// CONSUMER's position (the branch), because the branch's condition code
    /// decides the comparison kind — so any redefinition of the named register
    /// between the compare and that branch would silently make the compare read
    /// the LATER value. The compiler shuffles registers right after a `cmp` all
    /// the time, and when the shuffle copies the other operand the condition
    /// becomes `x == x` (always true). `at` is what lets the operands be pinned
    /// to the values they had when the flags were set.
    at: usize,
}

/// Unique scratch local ids for compare/branch results.
///
/// The old shared `Operand::Local(u32::MAX)` sentinel made every Compare in a
/// method the SAME def, so anything asking "which definition feeds this branch"
/// had to guess among several and regularly bound the wrong one — the
/// contradictory-guard corruption (`if (x != 0) { if (x == 0) {`). Unique ids
/// make the single-definition lookup exact end to end.
struct Scratch {
    next: u32,
}

impl Scratch {
    fn new() -> Self {
        Self { next: u32::MAX - 2 }
    }

    fn fresh(&mut self) -> Operand {
        let v = self.next;
        self.next = v.wrapping_sub(1);
        Operand::Local(v)
    }
}

/// Force a DEFERRED read of `op` to see the value it had when the flags were
/// set, not whatever that register holds by the time the compare is built.
///
/// A copy of the register is inserted immediately BEFORE the instruction that
/// would clobber it, and the operand is rewritten to read that copy. Because the
/// copy precedes the redefinition in the listing, SSA binds it to the old value
/// — which is exactly what the hardware read. `from` is the listing index at
/// which the operands were captured, so only redefinitions at or after the
/// compare are considered.
fn pin_deferred_read(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    from: usize,
    op: &mut Operand,
    next_scratch: &mut u32,
) {
    let Operand::Register(name) = op.clone() else {
        return;
    };
    // `dest_position(..) == Some(0)` is the SAME "is operand 0 a definition"
    // test the SSA renamer uses, so this finds exactly the instructions that
    // create a new version of `name`.
    let Some(j) = (from..insns.len()).find(|&j| {
        Insn::dest_position(&insns[j].op) == Some(0)
            && matches!(&insns[j].operands[0], Operand::Register(n) if *n == name)
    }) else {
        return;
    };
    let s = *next_scratch;
    *next_scratch = s.wrapping_sub(1);
    let pin = Operand::Local(s);
    // Keep `addresses` parallel BEFORE inserting into it, and give the copy the
    // clobbering instruction's own address so the array stays non-decreasing (a
    // branch "to the branch" resolves through this array by RVA).
    while addresses.len() < insns.len() {
        addresses.push(insns[addresses.len()].address);
    }
    let at_addr = insns[j].address;
    insns.insert(j, Insn::move_(pin.clone(), Operand::reg(name), at_addr));
    addresses.insert(j, at_addr);
    *op = pin;
}

/// Emit the `Compare` for a pending flag state and return the boolean scratch
/// that `CondJump`/`Select` consumes.
///
/// ONE implementation, used by both the conditional branch and the Select-shaped
/// arms: the two used to be separate copies of the same folding, so a fix applied
/// to one silently left the other behind.
fn materialize_pending(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    scratch: &mut Scratch,
    state: Option<&PendingCompare>,
    kind: CompareKind,
    addr: u64,
) -> Operand {
    let (mut lhs, mut rhs, at) = match state {
        Some(c) => (c.lhs.clone(), c.rhs.clone(), Some(c.at)),
        None => (Operand::Unknown, Operand::imm(0), None),
    };
    if let Some(at) = at {
        pin_deferred_read(insns, addresses, at, &mut lhs, &mut scratch.next);
        pin_deferred_read(insns, addresses, at, &mut rhs, &mut scratch.next);
    }
    let s = scratch.fresh();
    insns.push(Insn::new(
        Op::Compare(kind),
        vec![s.clone(), lhs, rhs],
        addr,
    ));
    s
}

/// Materialize the carry a flag-setting instruction left as an INTEGER 0/1.
///
/// `ADC rd, rn, rm` is `rd = rn + rm + C`, so the carry has to become a value
/// the arithmetic can be written against. It is built entirely from ops the
/// renderer already understands — a `Compare` produces the boolean, a
/// `Select(Identity)` turns it into `cond ? 1 : 0` — which keeps it a plain
/// integer in the output (`(a >= b) ? 1 : 0`) instead of a boolean silently
/// standing in for one, and needs no new IR.
///
/// `borrow` asks for `1 - C` instead, because A32's C after SUB is NOT-borrow.
fn materialize_carry(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    scratch: &mut Scratch,
    src: &CarrySource,
    borrow: bool,
    addr: u64,
) -> Operand {
    let mut lhs = src.lhs.clone();
    let mut rhs = src.rhs.clone();
    pin_deferred_read(insns, addresses, src.at, &mut lhs, &mut scratch.next);
    pin_deferred_read(insns, addresses, src.at, &mut rhs, &mut scratch.next);
    let cond = scratch.fresh();
    if src.subtract {
        // C after SUB/CMP = NOT borrow = (lhs >= rhs)
        insns.push(Insn::new(
            Op::Compare(CompareKind::GreaterOrEqual),
            vec![cond.clone(), lhs, rhs],
            addr,
        ));
    } else {
        // C after ADD/CMN = carry out = (lhs + rhs) < lhs
        let sum = scratch.fresh();
        insns.push(Insn::new(
            Op::Arith(ArithKind::Add),
            vec![sum.clone(), lhs.clone(), rhs],
            addr,
        ));
        insns.push(Insn::new(
            Op::Compare(CompareKind::LessThan),
            vec![cond.clone(), sum, lhs],
            addr,
        ));
    }
    let cf = scratch.fresh();
    let (t_val, f_val) = if borrow { (0, 1) } else { (1, 0) };
    insns.push(Insn::new(
        Op::Select(SelectVariant::Identity),
        vec![cf.clone(), cond, Operand::imm(t_val), Operand::imm(f_val)],
        addr,
    ));
    cf
}

/// Is this operand token a register name? Used by the generic fallback to tell
/// an instruction that writes a register from one that does not.
fn looks_like_reg(s: &str) -> bool {
    let s = s.trim();
    match s {
        "sp" | "lr" | "pc" | "fp" | "ip" | "sb" | "sl" => true,
        _ => {
            let mut cs = s.chars();
            match cs.next() {
                Some(c) if "rsd q".contains(c) => cs.all(|c| c.is_ascii_digit()) && s.len() > 1,
                _ => false,
            }
        }
    }
}

/// Does `r` already have a definition in the listing produced so far?
///
/// Only used to decide whether a predicated ALU op may read its destination on
/// the false path of a SELECT (`d = cond ? op : d`). The scan is bounded and
/// stops at a control transfer: past one, the linear listing no longer implies
/// the value is live.
fn has_prior_def(insns: &[Insn], r: &Operand) -> bool {
    for insn in insns.iter().rev().take(16) {
        if matches!(
            insn.op,
            Op::Jump(_)
                | Op::CondJump(_)
                | Op::Return
                | Op::Throw
                | Op::IndirectCall
                | Op::Switch(_)
        ) {
            return false;
        }
        if insn.operands.first() == Some(r) {
            return true;
        }
    }
    false
}

fn temp_local(counter: &mut u32) -> Operand {
    let t = 0xF000_0000u32.wrapping_add(*counter);
    *counter = counter.wrapping_add(1);
    Operand::Local(t)
}

/// Parse a shifted source operand (`r2, lsl 2`, `r2, asr 1`, …) →
/// (register, shift kind, amount).
fn parse_shifted_src(s: &str) -> Option<(Operand, ArithKind, Operand)> {
    let parts: Vec<&str> = s.splitn(2, ',').collect();
    if parts.len() != 2 {
        return None;
    }
    let base = reg(parts[0].trim());
    let mut it = parts[1].trim().split_whitespace();
    let kind = match it.next()? {
        "lsl" => ArithKind::Shl,
        "lsr" => ArithKind::Shr,
        "asr" => ArithKind::Shr, // IR has no AShr; arm64 maps asr → Shr too
        "ror" => ArithKind::Ror,
        _ => return None,
    };
    // The shift amount is an immediate (`lsl 2`) OR a REGISTER (`lsl r1`) —
    // A32 allows both, and yaxpeax prints the register form exactly that way.
    // Treating every non-numeric amount as 0 (the old `unwrap_or(0)`) silently
    // turned `ip, lsl r1` into `ip << 0`: the conditional-bit-set idiom
    // (`and r1, r1, #31` ; `bic r3, m, ip, lsl r1` ; `orrne r3, m, ip, lsl r1`)
    // then set the WRONG bit. A register amount is modelled as the register —
    // the same shape `DroppedOp::Shift` already emits for `mov rd, rm, lsl rs`.
    // Trim a trailing comma: the `rsb` arm builds the source as
    // `"{b_raw}, {a_raw}"`, so when `b_raw` was already shifted the amount
    // token arrives as `#4,` (comma glued on). Leaving it made the token fail
    // `parse_field` AND look like a register, producing `Register("4,")` — an
    // undefined local (`looks_like_reg` is the guard that stops that).
    let amt = match it.next().map(|t| t.trim_end_matches(',')) {
        Some(tok) => match parse_field(tok) {
            Some(v) => Operand::imm(v),
            None if looks_like_reg(tok) => reg(tok),
            // `lsl 0` is printed with the shift elided entirely; anything else
            // unrecognized is left as the elided shift rather than fabricating
            // a register name.
            None => Operand::imm(0),
        },
        None => Operand::imm(0),
    };
    Some((base, kind, amt))
}

/// Emit `t = src <shift> amt` and return the temp operand.
fn push_shifted(
    insns: &mut Vec<Insn>,
    temp: &mut u32,
    src: Operand,
    kind: ArithKind,
    amount: Operand,
    addr: u64,
) -> Operand {
    let t = temp_local(temp);
    insns.push(Insn::new(
        Op::Arith(kind),
        vec![t.clone(), src, amount],
        addr,
    ));
    t
}

/// Bitwise-not via xor with -1 (`~x == x ^ -1`) — Op::Not renders a *boolean*
/// `!`, which would corrupt integer data.
fn push_not(insns: &mut Vec<Insn>, dst: Operand, src: Operand, addr: u64) {
    insns.push(Insn::new(
        Op::Arith(ArithKind::Xor),
        vec![dst, src, Operand::imm(-1)],
        addr,
    ));
}

/// Parse an immediate/field that yaxpeax may print with the `#` marker
/// (`#8`, `#0x2`). `parse_int` alone rejects those, which silently turned a
/// shifted index or a bitfield width into 0.
fn parse_field(s: &str) -> Option<i64> {
    parse_int(s.trim().trim_start_matches('#'))
}

/// An A32 single-word memory operand plus the auto-indexing behaviour the
/// instruction performs.
struct MemAccess {
    base: Operand,
    index: Option<Operand>,
    /// LSL amount applied to `index`.
    shift: i64,
    addend: i64,
    /// `[rn, #off]!` (pre-indexed) or `[rn], #off` (post-indexed): the base
    /// register is rewritten by the instruction.
    writeback: bool,
    /// `[rn, -rm]` / `[rn], -#off`
    subtract: bool,
}

/// Parse an A32 memory operand: `[rn]`, `[rn, #off]`, `[rn, rm]`,
/// `[rn, rm, lsl #k]`, `[rn, #off]!`, `[rn], #off`, `[pc, #off]`.
///
/// The auto-indexing forms MUST be honoured: `ldr r0, [r1], #4` leaves `r1`
/// advanced by four, and that is exactly how a loop walks an array. Dropping
/// the writeback left the SECOND iteration (and every one after it) reading
/// through the original base — a wrong address in the dumped C#, not a
/// missing name.
fn parse_mem_access(text: &str) -> Option<MemAccess> {
    let t = text.trim();
    let open = t.find('[')?;
    let close = open + t[open..].find(']')?;
    let mut inner = t[open + 1..close].trim().to_string();
    let after = t[close + 1..].trim().to_string();
    let inner_bang = inner.ends_with('!');
    if inner_bang {
        inner = inner.trim_end_matches('!').trim().to_string();
    }
    // post-indexed iff there is an offset/register list AFTER the `]`
    let post = !after.is_empty() && after.trim_start_matches('!').trim() != "";
    let mut segs = inner.split(',');
    let base = reg(segs.next().unwrap_or("sp").trim());
    let second = segs.next().map(|s| s.trim()).unwrap_or("");
    let third = segs.next().map(|s| s.trim()).unwrap_or("");
    let mut acc = MemAccess {
        base,
        index: None,
        shift: 0,
        addend: 0,
        writeback: inner_bang || post,
        subtract: false,
    };
    let post_text = after.trim_start_matches('!').trim();
    let off_text = if post {
        // yaxpeax prints the post-index offset as `[r1], 0x4` — leading comma
        // included and WITHOUT the `#`. Leaving the comma in made the offset
        // parse as a register named "" (= Unknown), so every post-indexed
        // access lost its displacement AND its base update.
        post_text.trim_start_matches(',').trim()
    } else {
        second
    };
    if !off_text.is_empty() && off_text != "0" {
        let (neg, body) = match off_text.strip_prefix('-') {
            Some(r) => (true, r.trim()),
            None => (false, off_text),
        };
        match parse_field(body) {
            Some(v) => acc.addend = if neg { -v } else { v },
            None => {
                let idx = reg(body.split(',').next().unwrap_or("").trim());
                acc.index = Some(idx);
                acc.shift = if post {
                    0
                } else {
                    parse_shift_amount(third).unwrap_or(0)
                };
            }
        }
        acc.subtract = neg;
    }
    Some(acc)
}

/// `lsl #2` / `lsl 2` → 2.
fn parse_shift_amount(text: &str) -> Option<i64> {
    let mut it = text.split_whitespace();
    match it.next()? {
        "lsl" => it.next().and_then(parse_field),
        _ => None,
    }
}

/// Build the IR memory operand for a parsed access, and the base delta the
/// instruction's auto-indexing applies (None when the base is not rewritten).
///
/// A shift amount the `scale` field cannot express (not 0/1/2) is materialized
/// as a real shift of the index register, so the computed ADDRESS stays exact
/// rather than silently treated as scale 1.
fn mem_operand(
    acc: &MemAccess,
    insns: &mut Vec<Insn>,
    temp: &mut u32,
    addr: u64,
) -> (Operand, Option<Operand>) {
    let index = match &acc.index {
        None => None,
        Some(i) => Some(match acc.shift {
            0 | 2 => i.clone(),
            n => push_shifted(
                insns,
                temp,
                i.clone(),
                ArithKind::Shl,
                Operand::imm(n),
                addr,
            ),
        }),
    };
    let scale = if acc.index.is_some() && acc.shift == 2 {
        4
    } else {
        1
    };
    // `[pc, #off]` / `[pc, rm]`: `pc` is an ADDRESS — its pipeline value is
    // `addr + 8` — not a register. Left as the g15 pseudo-register it had no
    // definition anywhere, so every literal-pool load and every
    // GOT-relative access (the densest addressing idiom in arm32 IL2CPP) read
    // through an undefined local and the value was lost, while the emitted
    // address still looked plausible. As a constant base the address is exact.
    let base_is_pc = matches!(&acc.base, Operand::Register(n) if n == "g15");
    let base = if base_is_pc {
        Operand::imm(addr as i64 + 8)
    } else {
        acc.base.clone()
    };
    let mem = Operand::Memory {
        width: 0,
        base: Box::new(base),
        index: index.clone().map(Box::new),
        scale,
        addend: acc.addend,
    };
    let delta = if acc.writeback && !base_is_pc {
        match &acc.index {
            None => Some(Operand::imm(acc.addend)),
            Some(i) => Some(if acc.shift == 0 {
                i.clone()
            } else {
                push_shifted(
                    insns,
                    temp,
                    i.clone(),
                    ArithKind::Shl,
                    Operand::imm(acc.shift),
                    addr,
                )
            }),
        }
    } else {
        None
    };
    (mem, delta)
}

/// Emit the base-register rewrite an auto-indexing access performs.
fn push_writeback(insns: &mut Vec<Insn>, acc: &MemAccess, delta: Operand, addr: u64) {
    let kind = if acc.subtract {
        ArithKind::Sub
    } else {
        ArithKind::Add
    };
    insns.push(Insn::new(
        Op::Arith(kind),
        vec![acc.base.clone(), acc.base.clone(), delta],
        addr,
    ));
}

/// True for the Thumb-2 IT-block container mnemonics (`it`, `ite`, `itt`,
/// `ittt`, `itet`, …). They carry no semantics of their own — the condition
/// that actually predicates each instruction is the suffix yaxpeax prints on
/// that instruction, which this lifter already reads.
fn is_it_block(mne: &str) -> bool {
    let Some(rest) = mne.strip_prefix("it") else {
        return false;
    };
    !rest.is_empty() && rest.len() <= 3 && rest.chars().all(|c| c == 't' || c == 'e')
}

/// Is this word a NEON (Advanced SIMD) instruction?
///
/// yaxpeax-arm 0.4 cannot decode ANY Advanced SIMD word, and `is_vfp`
/// deliberately excludes `cond == 0xF` (NEON's fields would be silently
/// mis-decoded as scalar VFP). So these words reached neither decoder and
/// produced NO IR at all — while the machine EXECUTES them. On a 3,000-method
/// ARM32 sample they are 9,362 of the dropped words, and 99.98% of every
/// `cond == 0xF` word (the one counterexample is data), dominated by
/// `vld1`/`vst1`/`vst1.64` vector moves and `vmov.i32` vector immediates: the
/// `UnityEngine` math every float path runs through.
///
/// NEON is far too large to model here, so this recognizes the ENCODING SPACE
/// — `cond 1111` plus the three top-byte families that carry every Advanced
/// SIMD word (`0xF2`/`0xF3` data-processing, `0xF4` load/store) — and the
/// lifter emits the catch-all `Op::Simd` intrinsic for it. That is exactly the
/// contract `Op::Simd` documents: a missing NAME, never a lost instruction.
/// The word now HAS IR at its own address, so `cfg::resolve_target` lands on it
/// instead of sliding forward to the next instruction — which is the whole
/// `branch target mismatch: native taken` residue — and the census can account
/// for it instead of counting it as a decode failure.
///
/// The recognition is deliberately permissive within those top bytes. A stray
/// data word that matches decodes to an inert `Op::Simd` in a block the CFG
/// marks dead (`live_instructions` keeps the reachability walk out of it), and
/// the oracle synthesizes the same entry, so the two sides still agree.
pub(crate) fn neon_word(word: u32) -> bool {
    word >> 28 == 0xF && matches!((word >> 24) & 0xF, 0x2 | 0x3 | 0x4)
}

/// Is this word a VFP coprocessor instruction? (cond + copro 10/11.)
/// yaxpeax-arm 0.4 cannot decode ANY of these, so routing every such word to
/// the manual decoder never shadows a real ARM instruction. cond==F (NEON)
/// is intentionally excluded: NEON fields don't match scalar VFP and would
/// be silently mis-decoded — better skipped than wrong.
fn is_vfp(word: u32) -> bool {
    let copro = (word >> 8) & 0xF;
    (word >> 28) == 0xE && (copro == 0xA || copro == 0xB)
}

/// Register operand for ARM core register number `n` (0–15).
///
/// Spelled through `reg` so the same physical register keeps one identity no
/// matter which arm names it (`r13`→`sp`, `r14`→`g14`, `r0..r12`→`x0..x12`).
fn core_reg(n: u32) -> Operand {
    reg(&format!("r{n}"))
}

/// A signed-DSP multiply — the `SMUL`/`SMLA`/`SMLAL` halfword forms and the
/// `SMULW`/`SMLAW` word-by-halfword forms — decoded from the raw A32 word.
///
/// yaxpeax's rendering is NOT trustworthy for these encodings. Measured
/// against `llvm-mc`, the vendored decoder swaps the `<x>`/`<y>` selectors (it
/// calls `0xE14012E1` `smlalbb` where the ISA says `smlaltt`) and mis-families
/// `smulwb`/`smulwt` as `smlaw*`. Both mistakes are *silent*: they produce a
/// plausible product of the wrong operand half, or drop the `>> 16` and invent
/// an addend. So the variant and every register operand are taken from the bit
/// fields, never from the printed mnemonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DspMul {
    /// `SMLAL`: 64-bit accumulate into an `RdLo`/`RdHi` pair.
    pub(crate) long: bool,
    /// `SMULW`/`SMLAW`: Rn stays a full word and the product is taken `>> 16`.
    pub(crate) word_form: bool,
    /// An addend register is consumed (`SMLA`/`SMLAW`/`SMLAL`).
    pub(crate) accumulate: bool,
    /// Half selector for Rn (`SMUL`/`SMLA`/`SMLAL` only).
    pub(crate) rn_high: bool,
    /// Half selector for Rm.
    pub(crate) rm_high: bool,
}

/// Recognizer for the signed-DSP multiply group (`bits[27:23] = 00010`,
/// `bit20 = 0`, `bit7 = 1`, `bit4 = 0`). Everything else is rejected,
/// including the neighbouring `mul`/`mla` (`bits[27:23] = 00000`), the
/// unsigned `mull`-class forms (`bit20 = 1`), the `smuad`/`smmul` group
/// (`bits[27:23] = 01110`) and `swp`/`swpb`/`ldrd`/`strd` (`bit4 = 1`). The
/// discrimination is pinned by a table test.
pub(crate) fn dsp_mul(word: u32) -> Option<DspMul> {
    // bits[27:23] = 00010 selects the signed multiply group.
    if (word >> 23) & 0x1F != 0b0_0010 {
        return None;
    }
    // bit20 = 1 marks the unsigned forms (`umull`/`umlal`/`umaal`).
    if word & (1 << 20) != 0 {
        return None;
    }
    // bits[7:4] carries the variant: bit7 = 1 with bit4 = 0 for every form
    // here, which is exactly what excludes `mul`/`mla`/`smlad`/`smmul`
    // (bit4 = 1) — those would otherwise share the `bit20 = 0` prefix.
    if word & 0b1000_0000 == 0 || word & 0b0001_0000 != 0 {
        return None;
    }
    // bit6 = `<y>` (Rm's half). bit5 is `<x>` (Rn's half) for the halfword
    // forms; for the word forms it is the SMULW/SMLAW discriminator, `1` being
    // `smulw` and `0` `smlaw` (confirmed by round-tripping both through
    // `llvm-mc`, including with `Ra = 1111`).
    let rm_high = word & 0b0100_0000 != 0;
    let bit5 = word & 0b0010_0000 != 0;
    Some(match (word >> 21) & 0b11 {
        // `SMLA<x><y>` — halfword product accumulated into Rd.
        0b00 => DspMul {
            long: false,
            word_form: false,
            accumulate: true,
            rn_high: bit5,
            rm_high,
        },
        // `SMULW<y>` / `SMLAW<y>` — word × halfword.
        0b01 => DspMul {
            long: false,
            word_form: true,
            accumulate: !bit5,
            rn_high: false,
            rm_high,
        },
        // `SMLAL<x><y>` — halfword product accumulated into an RdLo/RdHi pair.
        0b10 => DspMul {
            long: true,
            word_form: false,
            accumulate: true,
            rn_high: bit5,
            rm_high,
        },
        // `SMUL<x><y>` — halfword product into Rd.
        _ => DspMul {
            long: false,
            word_form: false,
            accumulate: false,
            rn_high: bit5,
            rm_high,
        },
    })
}

/// The A32 `EXT` group — `SXTB` / `SXTH` / `UXTB` / `UXTH` / `SXTB16` / `UXTB16`
/// — decoded from the raw word.
///
/// yaxpeax-arm 0.4 rejects EVERY encoding in this group: `uxth r0,r0`
/// (`0xE6FF0070`), `uxtb r0,r0` (`0xE6EF0070`) and `sxtb r0,r0` (`0xE6AF0070`)
/// all come back `fails=1` with zero instructions, while `add`/`cmp`/`mov` from
/// the same corpus decode fine. That is not a rare gap: 18,206 words of the
/// ARM32 census's decode failures are this group, 95% of them `UXTH` — every
/// `(ushort)`/`(short)` narrowing and every 16-bit field load in the binary.
///
/// The consequence is worse than a missing name. `lift_arm32` emits NOTHING for
/// an undecodable word, so an instruction the machine EXECUTES has no IR entry
/// at its address. Any branch targeting it then resolves through
/// `cfg::resolve_target`'s at-or-after rule to the NEXT instruction — which the
/// semantic oracle correctly refuses to accept as an absorbed landing, because
/// a truncation is not a folded no-op. On the ARM32 corpus that is the entire
/// `sem-diff branch target mismatch: native taken` cluster (7,229 findings), and
/// the C# really IS wrong: the truncated loop counter is never recomputed, so
/// the loop's carried value keeps the first iteration's.
///
/// Encoding (ARM ARM A5.4, `cond 0110 1111 1111 mmmm dddd rot 00 op 0111 mmmm`):
/// Cn is the 4-bit condition, `Rn = 1111`, and `op = bits[22:20]` selects the
/// variant. The two 16-bit-lane forms spell `bits[11:8] = 1111` instead of a
/// rotate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExtOp {
    /// Index into [`EXT_MNEMONICS`].
    pub(crate) op: u8,
    pub(crate) rd: u8,
    pub(crate) rm: u8,
    /// `ROR #(8*rot)` applied to Rm before the extension (bits[11:10]).
    pub(crate) rot: u8,
}

/// The mnemonics of [`ExtOp::op`], in index order. These are the spellings the
/// oracle classifies against and the names the coverage census reports, so they
/// must be the real A32 mnemonics.
pub(crate) const EXT_MNEMONICS: [&str; 6] = ["sxtb16", "sxtb", "sxth", "uxtb16", "uxtb", "uxth"];

/// Would the LIFTER produce IR for this word even though yaxpeax cannot decode
/// it?
///
/// The decode-failure counter conflated two very different things. A word
/// yaxpeax rejects may still be reconstructed perfectly — the VFP coprocessor
/// group (`is_vfp`/`lift_vfp`) and the signed-DSP multiply group (`dsp_mul`)
/// are decoded straight from the raw bit fields precisely BECAUSE yaxpeax gets
/// them wrong or not at all, and the EXT group (`ext_op`) is synthesized on the
/// oracle side before this is even consulted. Such a word is a failure for the
/// ORACLE only — the engine recovered it — so it is not a reconstruction gap. Only a word the
/// LIFTER also drops is a real hole. `verify` counts the two apart so the
/// published decode-failure number measures the port, not the oracle's
/// coverage.
///
/// The VFP test is a dry run of the real decoder rather than a second,
/// drifting recognizer: `lift_vfp` decides what it handles, and anything it
/// answers `NotVfp` for is genuinely dropped by both sides.
pub(crate) fn liftable_word(word: u32) -> bool {
    if ext_op(word).is_some() || dsp_mul(word).is_some() || neon_word(word) {
        return true;
    }
    // The bitfield / smmul / uxtah / pkhbt / register-shift families the lifter
    // reconstructs from the raw word when yaxpeax refuses it (see
    // `dropped_op`). They are an ORACLE gap, exactly like the EXT group, so
    // they must not be counted as reconstruction failures.
    if dropped_op(word).is_some() {
        return true;
    }
    // Predicated VFP words the lifter answers with the catch-all `Op::Simd`
    // (see `vfp_predicated_word`) — recovered, not a reconstruction gap.
    if vfp_predicated_word(word) {
        return true;
    }
    if is_vfp(word) {
        let mut insns = Vec::new();
        let mut fp = BTreeSet::new();
        return !matches!(lift_vfp(word, 0, &mut insns, &mut fp), VfpOutcome::NotVfp);
    }
    false
}

/// A VFP word the lifter ABSORBS into its consumer: it produces NO IR of its
/// own, so a branch landing on it has no exact IR entry (see
/// `verify::absorbed_landing`). That is a flag-setting `vcmp`/`vcmpe`
/// (`Handled(Some(_))`) or the `vmrs`/`vmsr` that moves FPSCR↔APSR
/// (`Handled(None)` with nothing emitted). It is exactly `Handled(_)` with an
/// empty IR buffer: every other VFP form this crate lifts (vmov/vneg/vmla/vdiv/
/// vcvt/vpush…) pushes at least one IR instruction, so it HAS a landing site.
///
/// yaxpeax-arm 0.4 cannot decode the VFP space at all, so the ORACLE has no
/// entry for such a word. Without this, `verify::absorbed_landing` refused the
/// slide-to-successor landing (its chain walk found no decoded entry) and every
/// branch whose taken target was a VFP compare was reported as a wrong branch
/// target — the dominant ARM32 `sem-diff branch target mismatch: native taken`
/// shape (`bne` over a null-check raise onto a `vcmpe`).
pub(crate) fn vfp_absorbed_word(word: u32) -> bool {
    if !is_vfp(word) {
        return false;
    }
    let mut insns = Vec::new();
    let mut fp = BTreeSet::new();
    matches!(
        lift_vfp(word, 0, &mut insns, &mut fp),
        VfpOutcome::Handled(_)
    ) && insns.is_empty()
}

pub(crate) fn ext_op(word: u32) -> Option<ExtOp> {
    // cond 1111 is the unconditional/NEON space, never this group.
    if word >> 28 == 0xF || (word >> 24) & 0xF != 0x6 {
        return None;
    }
    // Rn must be 1111 and bits[7:4] must be 0111.
    if (word >> 16) & 0xF != 0xF || (word >> 4) & 0xF != 0x7 {
        return None;
    }
    let rd = ((word >> 12) & 0xF) as u8;
    let rm = (word & 0xF) as u8;
    match (word >> 20) & 0xF {
        // SXTB16 / UXTB16: both lanes, `bits[11:8] = 1111`, no rotate.
        0x8 | 0xC => {
            if (word >> 8) & 0xF != 0xF {
                return None;
            }
            Some(ExtOp {
                op: if (word >> 20) & 0xF == 0x8 { 0 } else { 3 },
                rd,
                rm,
                rot: 0,
            })
        }
        // SXTB / SXTH / UXTB / UXTH: `bits[9:8] = 00`, optional `ROR #8*rot`.
        op @ (0xA | 0xB | 0xE | 0xF) => {
            if (word >> 8) & 0x3 != 0 {
                return None;
            }
            Some(ExtOp {
                op: match op {
                    0xA => 1,
                    0xB => 2,
                    0xE => 4,
                    _ => 5,
                },
                rd,
                rm,
                rot: ((word >> 10) & 0x3) as u8,
            })
        }
        _ => None,
    }
}

/// A word the LIFTER must reconstruct from the raw bit fields because yaxpeax
/// rejects the whole encoding, and which is not one of the older groups
/// (`ext_op` / `dsp_mul` / VFP / NEON). Every arm below was checked against
/// `capstone` across the entire `libil2cpp.so` code section with ZERO
/// disagreements before it was written — a raw recognizer runs BEFORE yaxpeax,
/// so an arm that claimed a word the decoder can already read would silently
/// replace a correct decode with a hand-rolled one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DroppedOp {
    /// `ubfx rd, rn, #lsb, #width` — zero-extend bits `[lsb, lsb+width)`.
    Ubfx {
        rd: u8,
        rn: u8,
        lsb: u32,
        width: u32,
    },
    /// `sbfx rd, rn, #lsb, #width` — sign-extend bits `[lsb, lsb+width)`.
    Sbfx {
        rd: u8,
        rn: u8,
        lsb: u32,
        width: u32,
    },
    /// `bfi rd, rn, #lsb, #width`; `rn == 15` is the `bfc rd, #lsb, #width`
    /// (insert a zero constant) form.
    Bfi {
        rd: u8,
        rn: u8,
        lsb: u32,
        width: u32,
    },
    /// `smmul`/`smmulr rd, rn, rm` — the signed TOP word of `rn * rm`. Its low
    /// 32 bits are not the result, so it cannot be a plain 32-bit `Mul`; it is
    /// left explicitly unmodelled like the generic fallback.
    Smmul { rd: u8, rn: u8, rm: u8, round: bool },
    /// `uxtah`/`uxtab rd, rn, rm` — `rd = rn + ZeroExtend16(rm)`. The `ab`
    /// form additionally writes the result back to `rn`.
    Uxtah {
        rd: u8,
        rn: u8,
        rm: u8,
        writeback: bool,
    },
    /// `pkhbt rd, rn, rm, #shift` — pack the low half of `rn` with the shifted
    /// high half of `rm` (`pkhbt`, bit6 = 0; `pkhtb` also exists and is left).
    Pkhbt { rd: u8, rn: u8, rm: u8, shift: u32 },
    /// A `mov` with a REGISTER-controlled shift: `rd = rm <shift> rs`.
    /// `kind`: 0 LSL, 1 LSR, 2 ASR, 3 ROR.
    Shift { rd: u8, rm: u8, rs: u8, kind: u8 },
    /// `ldrsb`/`ldrsh rt, [rn, #±off]` — the signed narrow loads, which
    /// yaxpeax rejects (`Nonconforming`). `offset` is the effective-address
    /// displacement (0 for the post-indexed form) and `writeback`, when set, is
    /// the amount the base is advanced AFTER the access.
    Ldrs {
        rt: u8,
        rn: u8,
        offset: i64,
        writeback: Option<i64>,
        width: u8,
    },
}

/// A PREDICATED VFP word (`cond != 0xE`). `is_vfp` deliberately excludes these
/// (the scalar decoder's group logic assumes the unconditional form) and
/// yaxpeax rejects them too, so without this the word produced NO IR at all —
/// which broke `verify::absorbed_landing` for any branch that skips one (the
/// last `branch target mismatch` residuals land just above a `vmovgt` /
/// `vsubmi`). The recognizer keys on the coprocessor space bits[27:24] in
/// {C, D, E} with copro 10/11: integer ARM encodings never reach that prefix,
/// and `cond == 0xF` is the separate unconditional/NEON space. The operation
/// is not modelled — the word becomes the documented catch-all intrinsic that
/// DEFINES its destination register, exactly like NEON.
pub(crate) fn vfp_predicated_word(word: u32) -> bool {
    let cond = word >> 28;
    let copro = (word >> 8) & 0xF;
    cond != 0xF
        && cond != 0xE
        && (copro == 0xA || copro == 0xB)
        && matches!((word >> 24) & 0xF, 0xC | 0xD | 0xE)
}

/// Recognize one of the [`DroppedOp`] encodings. Unconditional forms only: a
/// predicated word (`cond != 0xE`) carries a guard this raw emitter does not
/// reconstruct, so it is left to fall through and be reported as a gap.
pub(crate) fn dropped_op(word: u32) -> Option<DroppedOp> {
    if word >> 28 != 0xE {
        return None;
    }
    // The VFP space is decided by `lift_vfp` (which may answer `NotVfp`); never
    // steal one of its words here.
    if is_vfp(word) {
        return None;
    }
    // ---- bitfield extract / insert -------------------------------------
    // Encoding: `cond 0111 11m msb Rd lsb 00 k Rn`, where bits[27:21] select
    // the family and bits[6:4] the extract/insert class. Validated: 803 ubfx,
    // 12 sbfx, 509 bfi/bfc, 0 disagreements. (An earlier mask keyed on bit21
    // instead of the [27:21] field claimed 35 words capstone decodes as `UN`
    // and missed the genuine `bfi`s — the field is authoritative, not bit21.)
    let g = (word >> 21) & 0x7F;
    let low3 = (word >> 4) & 0x7;
    let msb = (word >> 16) & 0x1F;
    let lsb = (word >> 7) & 0x1F;
    if low3 == 0b101 && (g == 0b0111111 || g == 0b0111101) {
        let width = msb + 1;
        if lsb + width <= 32 {
            let rd = ((word >> 12) & 0xF) as u8;
            let rn = (word & 0xF) as u8;
            return Some(if g == 0b0111111 {
                DroppedOp::Ubfx { rd, rn, lsb, width }
            } else {
                DroppedOp::Sbfx { rd, rn, lsb, width }
            });
        }
    }
    if low3 == 0b001 && g == 0b0111110 && msb >= lsb {
        return Some(DroppedOp::Bfi {
            rd: ((word >> 12) & 0xF) as u8,
            rn: (word & 0xF) as u8,
            lsb,
            width: msb - lsb + 1,
        });
    }
    // ---- smmul / smmulr (validated: 115/115) ---------------------------
    if (word >> 20) & 0xFF == 0x75 && (word >> 12) & 0xF == 0xF {
        let low4 = (word >> 4) & 0xF;
        if low4 == 0x1 || low4 == 0x3 {
            return Some(DroppedOp::Smmul {
                rd: ((word >> 16) & 0xF) as u8,
                rn: (word & 0xF) as u8,
                rm: ((word >> 8) & 0xF) as u8,
                round: low4 == 0x3,
            });
        }
    }
    // ---- uxtah / uxtab (validated: 45/45 uxtah, 12/12 uxtab) ----------
    // bits[27:20] = 0110 1111 (UXTAH) or 0110 1110 (UXTAB). Rn == 15 is plain
    // `uxth`, which yaxpeax DOES decode, so it is excluded.
    if ((word >> 20) & 0xFF == 0x6F || (word >> 20) & 0xFF == 0x6E)
        && (word >> 16) & 0xF != 0xF
        && (word >> 8) & 0xF == 0
        && (word >> 4) & 0xF == 0x7
    {
        return Some(DroppedOp::Uxtah {
            rd: ((word >> 12) & 0xF) as u8,
            rn: ((word >> 16) & 0xF) as u8,
            rm: (word & 0xF) as u8,
            writeback: (word >> 20) & 0xFF == 0x6E,
        });
    }
    // ---- pkhbt / pkhtb (validated: 15/15 pkhbt) ------------------------
    if (word >> 20) & 0xFF == 0x68 && (word >> 4) & 0x7 == 0x1 {
        return Some(DroppedOp::Pkhbt {
            rd: ((word >> 12) & 0xF) as u8,
            rn: ((word >> 16) & 0xF) as u8,
            rm: (word & 0xF) as u8,
            shift: (word >> 7) & 0x1F,
        });
    }
    // ---- ldrsb / ldrsh (yaxpeax: Nonconforming) ------------------------
    // `cond 000 P U 1 W 1 Rn Rt imm4H 1SH1 imm4L` — immediate-offset load.
    // The `1SH1` field: SH = 11 is LDRSH, SH = 10 is LDRSB, SH = 01 is LDRH
    // (which yaxpeax decodes, so it is deliberately not claimed) and SH = 00 is
    // the LDRD/STRD family. Non-PC base only; the literal-pool form is decoded
    // by yaxpeax.
    if (word >> 25) & 0x7 == 0
        && (word >> 22) & 1 == 1
        && (word >> 20) & 1 == 1
        && (word >> 16) & 0xF != 0xF
        && (word >> 7) & 1 == 1
        && (word >> 4) & 1 == 1
    {
        let sh = (word >> 5) & 0x3;
        if sh == 0b10 || sh == 0b11 {
            let off = ((((word >> 8) & 0xF) << 4) | (word & 0xF)) as i64;
            let signed_off = if (word >> 23) & 1 == 1 { off } else { -off };
            let pre = (word >> 24) & 1 == 1;
            let wb = (word >> 21) & 1 == 1 || !pre;
            return Some(DroppedOp::Ldrs {
                rt: ((word >> 12) & 0xF) as u8,
                rn: ((word >> 16) & 0xF) as u8,
                offset: if pre { signed_off } else { 0 },
                writeback: if wb { Some(signed_off) } else { None },
                width: if sh == 0b11 { 2 } else { 1 },
            });
        }
    }
    // ---- mov with a register-controlled shift (validated: 125/126) -----
    if (word >> 26) & 0x3 == 0
        && (word >> 25) & 1 == 0
        && (word >> 21) & 0xF == 0xD
        && (word >> 7) & 1 == 0
        && (word >> 4) & 1 == 1
    {
        return Some(DroppedOp::Shift {
            rd: ((word >> 12) & 0xF) as u8,
            rm: (word & 0xF) as u8,
            rs: ((word >> 8) & 0xF) as u8,
            kind: ((word >> 5) & 0x3) as u8,
        });
    }
    None
}

/// Emit the IR for a [`DroppedOp`], defining the destination register so a
/// branch that lands on the word resolves instead of sliding to the next
/// instruction (the `branch target mismatch: native taken` shape).
fn emit_dropped_op(
    op: DroppedOp,
    addr: u64,
    insns: &mut Vec<Insn>,
    temp_counter: &mut u32,
    consts: &mut HashMap<String, i64>,
) {
    let clear_const = |dst: &Operand, consts: &mut HashMap<String, i64>| {
        if let Operand::Register(n) = dst {
            consts.remove(n);
        }
    };
    match op {
        DroppedOp::Ubfx { rd, rn, lsb, width } | DroppedOp::Sbfx { rd, rn, lsb, width } => {
            let signed = matches!(op, DroppedOp::Sbfx { .. });
            let dst = core_reg(rd as u32);
            let src = core_reg(rn as u32);
            clear_const(&dst, consts);
            let mut cur = src.clone();
            let mut pushed = false;
            if lsb > 0 {
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shr),
                    vec![dst.clone(), cur, Operand::imm(lsb as i64)],
                    addr,
                ));
                cur = dst.clone();
                pushed = true;
            }
            if width < 32 {
                if signed {
                    let k = 32 - width;
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![dst.clone(), cur, Operand::imm(k as i64)],
                        addr,
                    ));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shr),
                        vec![dst.clone(), dst.clone(), Operand::imm(k as i64)],
                        addr,
                    ));
                } else {
                    let mask = (1i64 << width) - 1;
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::And),
                        vec![dst.clone(), cur, Operand::imm(mask)],
                        addr,
                    ));
                }
                pushed = true;
            }
            if !pushed {
                // `lsb == 0 && width == 32`: the whole register is the result.
                insns.push(Insn::move_(dst, src, addr));
            }
        }
        DroppedOp::Bfi { rd, rn, lsb, width } => {
            let dst = core_reg(rd as u32);
            clear_const(&dst, consts);
            let field_mask: i64 = if width >= 32 {
                -1
            } else {
                ((1i64 << width) - 1) << lsb
            };
            if rn == 15 {
                // `bfc`: clear the field, keep everything else.
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![dst.clone(), dst, Operand::imm(!field_mask)],
                    addr,
                ));
            } else {
                let src = core_reg(rn as u32);
                let t = temp_local(temp_counter);
                let mask = if width >= 32 { -1 } else { (1i64 << width) - 1 };
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![t.clone(), src, Operand::imm(mask)],
                    addr,
                ));
                if lsb > 0 {
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![t.clone(), t.clone(), Operand::imm(lsb as i64)],
                        addr,
                    ));
                }
                let keep = temp_local(temp_counter);
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![keep.clone(), dst.clone(), Operand::imm(!field_mask)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Or),
                    vec![dst, keep, t],
                    addr,
                ));
            }
        }
        DroppedOp::Smmul { rd, rn, rm, round } => {
            let dst = core_reg(rd as u32);
            clear_const(&dst, consts);
            let name = if round { "__smmulr" } else { "__smmul" };
            insns.push(Insn::new(
                Op::Simd(name.into()),
                vec![dst, core_reg(rn as u32), core_reg(rm as u32)],
                addr,
            ));
        }
        DroppedOp::Uxtah {
            rd,
            rn,
            rm,
            writeback,
        } => {
            let dst = core_reg(rd as u32);
            let base = core_reg(rn as u32);
            clear_const(&dst, consts);
            if writeback {
                clear_const(&base, consts);
            }
            let t = temp_local(temp_counter);
            insns.push(Insn::new(
                Op::Arith(ArithKind::And),
                vec![t.clone(), core_reg(rm as u32), Operand::imm(0xFFFF)],
                addr,
            ));
            insns.push(Insn::new(
                Op::Arith(ArithKind::Add),
                vec![dst.clone(), base.clone(), t],
                addr,
            ));
            if writeback {
                // `uxtab` writes the result back to the base (`Rn = Rd`).
                insns.push(Insn::move_(base, dst, addr));
            }
        }
        DroppedOp::Pkhbt { rd, rn, rm, shift } => {
            let dst = core_reg(rd as u32);
            clear_const(&dst, consts);
            let lo = temp_local(temp_counter);
            insns.push(Insn::new(
                Op::Arith(ArithKind::And),
                vec![lo.clone(), core_reg(rn as u32), Operand::imm(0xFFFF)],
                addr,
            ));
            let mut hi_src = core_reg(rm as u32);
            if shift > 0 {
                let t = temp_local(temp_counter);
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shl),
                    vec![t.clone(), hi_src, Operand::imm(shift as i64)],
                    addr,
                ));
                hi_src = t;
            }
            let hi = temp_local(temp_counter);
            insns.push(Insn::new(
                Op::Arith(ArithKind::And),
                vec![hi.clone(), hi_src, Operand::imm(0xFFFF_0000u32 as i64)],
                addr,
            ));
            insns.push(Insn::new(Op::Arith(ArithKind::Or), vec![dst, lo, hi], addr));
        }
        DroppedOp::Ldrs {
            rt,
            rn,
            offset,
            writeback,
            width,
        } => {
            let dst = core_reg(rt as u32);
            clear_const(&dst, consts);
            let mem = Operand::Memory {
                width,
                base: Box::new(core_reg(rn as u32)),
                index: None,
                scale: 1,
                addend: offset,
            };
            insns.push(Insn::move_(dst.clone(), mem, addr));
            // Sign-extend the narrow value to the full register: `(v << k) >> k`
            // with k = 16/24 — the same lowering the `ldrsh`/`ldrsb` arm uses.
            let k = if width == 2 { 16 } else { 24 };
            insns.push(Insn::new(
                Op::Arith(ArithKind::Shl),
                vec![dst.clone(), dst.clone(), Operand::imm(k)],
                addr,
            ));
            insns.push(Insn::new(
                Op::Arith(ArithKind::Shr),
                vec![dst.clone(), dst.clone(), Operand::imm(k)],
                addr,
            ));
            if let Some(delta) = writeback {
                let base = core_reg(rn as u32);
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Add),
                    vec![base.clone(), base, Operand::imm(delta)],
                    addr,
                ));
            }
        }
        DroppedOp::Shift { rd, rm, rs, kind } => {
            let dst = core_reg(rd as u32);
            clear_const(&dst, consts);
            let src = core_reg(rm as u32);
            let sh = core_reg(rs as u32);
            match kind {
                0 => insns.push(Insn::new(
                    Op::Arith(ArithKind::Shl),
                    vec![dst, src, sh],
                    addr,
                )),
                1 => insns.push(Insn::new(
                    Op::Arith(ArithKind::Shr),
                    vec![dst, src, sh],
                    addr,
                )),
                3 => insns.push(Insn::new(
                    Op::Arith(ArithKind::Ror),
                    vec![dst, src, sh],
                    addr,
                )),
                // ASR has no `ArithKind`; an unmodelled intrinsic keeps the
                // value chain (and the destination defined) without pretending
                // a logical shift is an arithmetic one.
                _ => insns.push(Insn::new(
                    Op::Simd("__asr".into()),
                    vec![dst, src, sh],
                    addr,
                )),
            }
        }
    }
}

/// The A32 condition-code spellings, indexed by the 4-bit field. The numbering
/// is the one A64 shares for 0..=13; 14 is `al` and 15 the never-taken `nv`.
pub(crate) const A32_COND_NAMES: [&str; 16] = [
    "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv",
];

/// Result of the manual VFP decoder.
enum VfpOutcome {
    /// Word handled; a pending compare was produced (vcmp) for the next
    /// conditional branch to consume.
    Handled(Option<PendingCompare>),
    /// Not a VFP instruction we recognize — fall through to yaxpeax.
    NotVfp,
}

fn vfp_single(vd: u32, d: u32) -> Operand {
    sreg((vd << 1) | d)
}
fn vfp_double(vd: u32, d: u32) -> Operand {
    dreg((d << 4) | vd)
}
fn vfp_single_src(vm: u32, m: u32) -> Operand {
    sreg((vm << 1) | m)
}
fn vfp_double_src(vm: u32, m: u32) -> Operand {
    dreg((m << 4) | vm)
}

/// Decode a VFP `vmov.f32/d64 sN, #imm` immediate. Empirically derived from
/// keystone ground truth (imm4→2^E): {0,1,2,3,4,5,6,7} →
/// {1,2,3,4,−3,−2,−1,0}; sign = imm4 bit 3; value = ±(1 + imm8/16)·2^E.
fn vfp_imm_value(imm4: u32, imm8: u32) -> f64 {
    let e = imm4 & 7;
    let exp = if e >= 4 { e - 7 } else { e + 1 };
    let v = (1.0 + (imm8 as f64) / 16.0) * 2f64.powi(exp as i32);
    if imm4 & 8 != 0 { -v } else { v }
}

fn lift_vfp(
    word: u32,
    addr: u64,
    insns: &mut Vec<Insn>,
    fp_written: &mut BTreeSet<String>,
) -> VfpOutcome {
    let d = (word >> 22) & 1;
    let n = (word >> 7) & 1;
    let m = (word >> 5) & 1;
    let b27_23 = (word >> 23) & 0x1F;
    let bit6 = (word >> 6) & 1;
    let bit4 = (word >> 4) & 1;
    let opc2 = (word >> 20) & 3;
    let vn = (word >> 16) & 0xF;
    let vd = (word >> 12) & 0xF;
    let vm = word & 0xF;
    let double = (word >> 8) & 1 == 1; // copro 0xB => double precision

    // ---- two core registers <-> doubleword move (group 11000) ----------
    //
    // `vmov Rt, Rt2, Dm` / `vmov Dm, Rt, Rt2`: how the compiler builds a 64-bit
    // double in GPRs (and unpacks one). yaxpeax rejects the whole encoding and
    // the register-move arm in the groups below only knows the SINGLE-register
    // `vmov sN, rM` / `vmov rM, sN` form, so every one of these was silently
    // dropped — the largest single cluster (224 branch targets) in the ARM32
    // `branch target mismatch: native taken` residual.
    //
    // Bit fields verified against capstone over all 16,384 encodings of the
    // group (0 disagreements): cond=1110, bits[27:24]=1100, bits[23:21]=010,
    // bit20 = direction (1 = to core), Rt = bits[15:12], Rt2 = bits[19:16],
    // bits[7:6] = 00, bit4 = 1, Dm = (bit5 << 4) | bits[3:0].
    if b27_23 == 0b11000
        && (word >> 8) & 0xF == 0xB
        && (word >> 6) & 0x3 == 0
        && (word >> 4) & 1 == 1
    {
        let to_core = (word >> 20) & 1 == 1;
        let dm = dreg((((word >> 5) & 1) << 4) | (word & 0xF));
        let rt = core_reg((word >> 12) & 0xF);
        let rt2 = core_reg((word >> 16) & 0xF);
        if to_core {
            // Both halves are DEFINED here, so each gets its own op (the IR
            // gives a `Simd` op a single destination); the double is the shared
            // source. `__vmov` is the documented unmodelled catch-all — the same
            // shape the generic fallback emits — which keeps the value chain
            // connected and gives the word IR AT ITS OWN ADDRESS.
            insns.push(Insn::new(
                Op::Simd("__vmov".into()),
                vec![rt, dm.clone()],
                addr,
            ));
            insns.push(Insn::new(Op::Simd("__vmov".into()), vec![rt2, dm], addr));
        } else {
            insns.push(Insn::new(
                Op::Simd("__vmov".into()),
                vec![dm.clone(), rt, rt2],
                addr,
            ));
            fp_written.insert(reg_name(&dm));
        }
        return VfpOutcome::Handled(None);
    }

    // ---- VFP data processing (VMLA/VMUL/VADD/VSUB: group 11100)
    if b27_23 == 0b11100 {
        let dst = if double {
            vfp_double(vd, d)
        } else {
            vfp_single(vd, d)
        };
        let vn_op = if double {
            vfp_double_src(vn, n)
        } else {
            vfp_single_src(vn, n)
        };
        let vm_op = if double {
            vfp_double_src(vm, m)
        } else {
            vfp_single_src(vm, m)
        };

        // core <-> VFP register move: `vmov sN, rM` / `vmov rM, sN`
        // (bit4=1 is the transfer-class marker; opc2=0 reg→vfp, 1 vfp→reg).
        //
        // The condition used to ALSO require `vn == 0xE`, which is only the
        // s28–s31 slice of the single register file. Everything below that fell
        // through: `vmov s0, r0` (vn=0, opc2=0) reached the VMLA arm and was
        // lifted as an accumulating high-precision multiply — wrong logic with no
        // signal — while `vmov r0, s0` (opc2=1) fell to yaxpeax and was DROPPED.
        // The real discriminator is bit4 (set for the core-transfer class, clear
        // for VMLA/VMLS) plus `opc2 <= 1`; `vn` is the single register's HIGH
        // part, not a tag. Verified against capstone over all 1,024 encodings of
        // the group: the 512 with bit4=1 and opc2 ∈ {0,1} are exactly the vmov
        // core transfers; opc2 ∈ {2,3} is invalid.
        //
        // The core register must be named through `core_reg`/`reg`, not a local
        // table: a hand-rolled map used `g4`–`g12` for r4–r12, which are NOT the
        // names `reg()` produces (`x4`–`x12`), so the same physical register
        // reached the renderer under two identities — the phantom-register class
        // the `reg` alias table exists to prevent.
        if bit4 == 1 && !double && opc2 <= 1 {
            let sd = vfp_single(vn, n);
            let core = core_reg(vd);
            if opc2 == 0 {
                insns.push(Insn::move_(sd.clone(), core, addr));
                fp_written.insert(reg_name(&sd));
            } else {
                insns.push(Insn::move_(core, sd, addr));
            }
            return VfpOutcome::Handled(None);
        }

        match (opc2, bit6) {
            (0, 0) => {
                // VMLA: dst = dst + a*b
                insns.push(Insn::new(
                    Op::MulAcc {
                        negate_addend: false,
                        is_float: true,
                    },
                    vec![dst.clone(), vn_op, vm_op, dst.clone()],
                    addr,
                ));
            }
            (0, 1) => {
                // VMLS: dst = dst - a*b
                insns.push(Insn::new(
                    Op::MulAcc {
                        negate_addend: true,
                        is_float: true,
                    },
                    vec![dst.clone(), vn_op, vm_op, dst.clone()],
                    addr,
                ));
            }
            (2, _) => {
                // VMUL / VNMUL
                insns.push(Insn::new(
                    Op::FloatArith(FloatArithKind::Mul),
                    vec![dst.clone(), vn_op, vm_op],
                    addr,
                ));
            }
            (3, 0) => {
                insns.push(Insn::new(
                    Op::FloatArith(FloatArithKind::Add),
                    vec![dst.clone(), vn_op, vm_op],
                    addr,
                ));
            }
            (3, 1) => {
                insns.push(Insn::new(
                    Op::FloatArith(FloatArithKind::Sub),
                    vec![dst.clone(), vn_op, vm_op],
                    addr,
                ));
            }
            _ => return VfpOutcome::NotVfp,
        }
        fp_written.insert(reg_name(&dst));
        return VfpOutcome::Handled(None);
    }

    // ---- VDIV (group 11101, opc2 = 0) -------------------------------------
    if b27_23 == 0b11101 && opc2 == 0 {
        let dst = if double {
            vfp_double(vd, d)
        } else {
            vfp_single(vd, d)
        };
        let vn_op = if double {
            vfp_double_src(vn, n)
        } else {
            vfp_single_src(vn, n)
        };
        let vm_op = if double {
            vfp_double_src(vm, m)
        } else {
            vfp_single_src(vm, m)
        };
        insns.push(Insn::new(
            Op::FloatArith(FloatArithKind::Div),
            vec![dst.clone(), vn_op, vm_op],
            addr,
        ));
        fp_written.insert(reg_name(&dst));
        return VfpOutcome::Handled(None);
    }

    // ---- unary / compare / convert / immediate (group 11101, opc2 = 3) ----
    // The sub-operation lives in bits[7:4] plus Vn (verified against
    // keystone+capstone for EVERY instruction below — no guesses):
    //   hi==0x0 : VMOV immediate (imm4 = Vn, imm8 = bits[7:0], dst = Vd:D)
    //   hi==0x1 : VMRS/VMSR (safe no-op — our pending-compare fold models the
    //             vcmp→vmrs→b<cc> chain without materializing APSR)
    //   hi==0x4 : VCMP #0.0
    //   hi==0x6 : VNEG(Vn=1) / VCMP(Vn=4) / VCVT int→float unsigned(Vn=8)
    //   hi==0xC : F64-source ops: VABS(Vn=0) / VCVT f32←f64(Vn=7)
    //             / VCVT u·s32←f64(Vn=C/D)
    //   hi==0xE : F32-source ops: VABS(Vn=0) / VSQRT(Vn=1) / VCVT f64←f32(Vn=7)
    //             / VCVT s32←f32(Vn=D) / u32←f32(Vn=C) / int→float signed(Vn=8)
    if b27_23 == 0b11101 && opc2 == 3 {
        // `hi` selects the operation from bits[7:4] — EXCEPT that bit5 is not
        // an opcode bit at all: it is `M`, the low bit of the SINGLE source
        // register `Vm` (`VfpOutcome` reads it as `m` above). Including it made
        // the arms below fire only for the even-register spelling and drop the
        // odd one: `vmov.f64 d8, d16` (Vm = 16 -> bit5 = 1) landed in the
        // `0x6`/`0xE` slices while `vcmp.f32 s18, #0` (bit5 = 0) landed in
        // `0x4`/`0xC`. Clearing it collapses the two redundant encodings onto
        // ONE arm per operation — the fix for the surviving `vmov.f64`/`vsqrt`/
        // `vneg`/`vcvt` branch targets.
        let hi = ((word >> 4) & 0xF) & !0b10;
        let signed = (word >> 7) & 1 == 1;
        let s_dst = vfp_single(vd, d);
        let d_dst = vfp_double(vd, d);
        let s_src = vfp_single_src(vm, m);
        let d_src = vfp_double_src(vm, m);
        match hi {
            0x0 => {
                // VMOV immediate (imm4 = Vn, imm8 = bits[7:0])
                let v = vfp_imm_value(vn, word & 0xFF);
                let dst = if double { d_dst } else { s_dst };
                let lit = if double {
                    Operand::DoubleLiteral(v)
                } else {
                    Operand::FloatLiteral(v as f32)
                };
                insns.push(Insn::move_(dst.clone(), lit, addr));
                fp_written.insert(reg_name(&dst));
            }
            0x1 => {
                // VMRS/VMSR — safe no-op (flags already modeled by the
                // pending-compare fold)
            }
            0x4 => {
                // VCMP (non-signalling): Vn selects the form. `Vn == 4` is a
                // register comparison, `Vn == 5` is the implicit `#0.0` form.
                //
                // This arm used to return the `#0.0` form UNCONDITIONALLY, so
                // `vcmp.f32 s2, s0` (Vn == 4) lost its right operand and became
                // `s2 vs 0.0` — a comparison the machine does not perform. That
                // is a wrong-logic defect, not just a missing one. The `double`
                // bit selects the f32/f64 register file, so the operands are
                // read from the matching one.
                // `at` is re-stamped by the caller with the real listing index;
                // usize::MAX disables pinning until then rather than pinning
                // from an index lift_vfp cannot know.
                return match vn {
                    0x4 => VfpOutcome::Handled(Some(PendingCompare {
                        lhs: if double { d_dst } else { s_dst },
                        rhs: if double { d_src } else { s_src },
                        at: usize::MAX,
                    })),
                    0x5 => VfpOutcome::Handled(Some(PendingCompare {
                        lhs: if double { d_dst } else { s_dst },
                        rhs: Operand::FloatLiteral(0.0),
                        at: usize::MAX,
                    })),
                    0x0 => {
                        // VMOV register: the F32/F64 copy the compiler emits to
                        // shuffle a value inside the FP file (`vmov.f32 s0, s18`).
                        // The ISA spells this with bit5 (= Vm's low bit) clear;
                        // the `0x6` slice above is the same op with bit5 set.
                        let dst = if double { d_dst } else { s_dst };
                        let src = if double { d_src } else { s_src };
                        insns.push(Insn::move_(dst.clone(), src, addr));
                        fp_written.insert(reg_name(&dst));
                        VfpOutcome::Handled(None)
                    }
                    0x1 => {
                        // VNEG (bit7 = 0; the `0x6` slice is the redundant
                        // encoding). Width-aware: the F64 form names the double
                        // register file.
                        let dst = if double { d_dst } else { s_dst };
                        let src = if double { d_src } else { s_src };
                        insns.push(Insn::new(
                            Op::FloatArith(FloatArithKind::Neg),
                            vec![dst.clone(), src],
                            addr,
                        ));
                        fp_written.insert(reg_name(&dst));
                        VfpOutcome::Handled(None)
                    }
                    0x8 => {
                        // VCVT unsigned int -> float (bit7 = 0)
                        let dst = if double { d_dst } else { s_dst };
                        let conv = if double {
                            Conversion::IntToFloat64
                        } else {
                            Conversion::IntToFloat32
                        };
                        insns.push(Insn::new(Op::Convert(conv), vec![dst.clone(), s_src], addr));
                        fp_written.insert(reg_name(&dst));
                        VfpOutcome::Handled(None)
                    }
                    _ => VfpOutcome::NotVfp,
                };
            }
            0xC => match vn {
                // VCMPE — the exception-signalling twin of the `0x4` arm.
                // `0xC` is `0x4` with bit7 set, which is exactly how the ISA
                // spells the `-e` suffix. It was NOT handled at all, so every
                // `vcmpe.f32 s2, s0; vmrs; bhi` chain (the shape the C# `>`/`<`
                // on floats compiles to, because a NaN must raise) fell through
                // to `NotVfp`, lost the compare, and rendered the branch as
                // `if (? > 0)` — the ARM32 floating-point half of the 1,136
                // `cond-branch without real compare` class. `Vn == 0` is
                // VABS.f64 (below), so only `4`/`5` are compares.
                0x4 => {
                    return VfpOutcome::Handled(Some(PendingCompare {
                        lhs: if double { d_dst } else { s_dst },
                        rhs: if double { d_src } else { s_src },
                        at: usize::MAX,
                    }));
                }
                0x5 => {
                    return VfpOutcome::Handled(Some(PendingCompare {
                        lhs: if double { d_dst } else { s_dst },
                        rhs: Operand::FloatLiteral(0.0),
                        at: usize::MAX,
                    }));
                }
                0x1 => {
                    // VSQRT — the ISA duplicates this under bit5 = 1 (`hi == 0xE`,
                    // the arm below); with bit5 clear it is the F32 case that
                    // ACTUALLY occurs (`vsqrt.f32`), which was left to
                    // `NotVfp`. Width-aware so the F64 form names the double
                    // file.
                    let dst = if double { d_dst } else { s_dst };
                    let src = if double { d_src } else { s_src };
                    insns.push(Insn::new(
                        Op::FloatArith(FloatArithKind::Sqrt),
                        vec![dst.clone(), src],
                        addr,
                    ));
                    fp_written.insert(reg_name(&dst));
                }
                0x8 => {
                    // VCVT signed int -> float (bit7 = 1)
                    let dst = if double { d_dst } else { s_dst };
                    let conv = if double {
                        Conversion::IntToFloat64
                    } else {
                        Conversion::IntToFloat32
                    };
                    insns.push(Insn::new(Op::Convert(conv), vec![dst.clone(), s_src], addr));
                    fp_written.insert(reg_name(&dst));
                }
                0x0 => {
                    // VABS — width-aware: `double` (copro 11) selects the F64
                    // register file. This arm used to hard-code the F64 form, so
                    // `vabs.f32` (the one that actually occurs) named the wrong
                    // file.
                    let dst = if double { d_dst } else { s_dst };
                    let src = if double { d_src } else { s_src };
                    insns.push(Insn::new(
                        Op::FloatArith(FloatArithKind::Abs),
                        vec![dst.clone(), src],
                        addr,
                    ));
                    fp_written.insert(reg_name(&dst));
                }
                0x7 => {
                    // VCVT between F32 and F64. `double` says which file the
                    // SOURCE is in, which fixes the direction: an F64 source
                    // narrows (`vcvt.f32.f64 s0, d16`), otherwise a single
                    // widens (`vcvt.f64.f32 d8, s0`).
                    if double {
                        insns.push(Insn::new(
                            Op::Convert(Conversion::Float64ToFloat32),
                            vec![s_dst.clone(), d_src],
                            addr,
                        ));
                        fp_written.insert(reg_name(&s_dst));
                    } else {
                        insns.push(Insn::new(
                            Op::Convert(Conversion::Float32ToFloat64),
                            vec![d_dst.clone(), s_src],
                            addr,
                        ));
                        fp_written.insert(reg_name(&d_dst));
                    }
                }
                0xC | 0xD => {
                    // VCVT u32/s32 <- float: an F64 source when `double`, else
                    // the single register file.
                    insns.push(Insn::new(
                        Op::Convert(Conversion::FloatToInt),
                        vec![s_dst.clone(), if double { d_src } else { s_src }],
                        addr,
                    ));
                    fp_written.insert(reg_name(&s_dst));
                }
                _ => return VfpOutcome::NotVfp,
            },
            _ => return VfpOutcome::NotVfp,
        }
        let _ = signed;
        return VfpOutcome::Handled(None);
    }

    // ---- VFP load/store & register-list (groups 11001/11010/11011) --------
    if matches!(b27_23, 0b11001 | 0b11010 | 0b11011) {
        // Every core register is a valid VLDR/VSTR base (`vldr s0, [r4, #0x14]`
        // is ordinary code — r4 holds a pinned object pointer). Only the
        // VSTM/VLDM (vpush/vpop) form is restricted to sp, and that is gated
        // INSIDE the opc2 2|3 arm below. An earlier `4..=12 => return NotVfp`
        // guard sat here (for the vpush/vpop sp-only rule) and so also killed
        // every VLDR/VSTR off r4–r12, silently DROPPING the load. The `vs16`
        // reads that resulted were the top ARM32 FP undefined-local residual,
        // and the dropped `vldr` also detached the FP chain feeding its compare
        // (a following `vcmp s0, #0` then had no operand). Name r4–r12 like
        // every other GPR so the load survives.
        // r4–r12 reuse a static name table rather than allocating a String per
        // instruction; every other core register has a fixed name.
        const GPR4_12: [&str; 9] = ["x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"];
        let base_reg: &str = match vn {
            13 => "sp",
            0 => "x0",
            1 => "x1",
            2 => "x2",
            3 => "x3",
            4..=12 => GPR4_12[(vn - 4) as usize],
            14 => "g14",
            _ => "g15",
        };
        let l = opc2 & 1;
        match opc2 {
            0 | 1 => {
                // VSTR / VLDR: cond 1101 U D 01 Rn Vd 101x imm8 (offset << 2)
                let u = (word >> 23) & 1;
                let off = (word & 0xFF) as i64 * 4;
                let addend = if u == 1 { off } else { -off };
                // `vn == 15` (`pc`) is a LITERAL-POOL load: resolve the address
                // the way the integer `ldr` arm does (PC reads as `addr + 8`)
                // instead of leaving a raw `g15` base — `g15` is not a real
                // register, so a `vldr s0, [pc, #off]` rendered against the
                // program counter instead of the constant it loads.
                let mem = if vn == 15 {
                    Operand::Memory {
                        width: 0,
                        base: Box::new(Operand::imm(addr as i64 + 8 + addend)),
                        index: None,
                        scale: 4,
                        addend: 0,
                    }
                } else {
                    Operand::Memory {
                        width: 0,
                        base: Box::new(Operand::reg(base_reg)),
                        index: None,
                        scale: 4,
                        addend,
                    }
                };
                let reg_op = if double {
                    vfp_double(vd, d)
                } else {
                    vfp_single(vd, d)
                };
                if l == 1 {
                    insns.push(Insn::move_(reg_op.clone(), mem, addr));
                    fp_written.insert(reg_name(&reg_op));
                } else {
                    insns.push(Insn::move_(mem, reg_op, addr));
                }
                return VfpOutcome::Handled(None);
            }
            2 | 3 => {
                // VSTM / VLDM. When the base is `sp` with writeback this is
                // `vpush`/`vpop`, a pure frame adjustment — model it as a stack
                // shift, matching the integer push/pop approach (the frame
                // tracker owns sp, so materializing the individual stores would
                // double-count the movement).
                //
                // EVERY OTHER base is an ordinary register-list transfer into
                // memory: `vstmia r10!, {s16}` is how the compiler spills a
                // float across a call. It must still produce IR AT ITS OWN
                // ADDRESS — skipping it left a hole the CFG's absorbed-landing
                // slide could not cross, so a `bhi` targeting the `vstmia` had
                // its IR target shifted to the NEXT instruction. That was the
                // LAST genuine `branch target mismatch` the independent
                // Capstone checker found (0xa32a78 -> `vstmia r10!`). Model it
                // as one memory access per register plus the writeback, exactly
                // like the integer `ldm`/`stm` arm.
                //
                // Encoding (verified against capstone over the whole
                // group/copro space): bits[11:8] = 0xA single / 0xB double,
                // Vd = bits[15:12], D = bit22, Rn = bits[19:16], W = bit21,
                // imm8 = bits[7:0]. The first register is `vfp_single(vd, d)`
                // (single) / `vfp_double(vd, d)` (double) and the list walks
                // upward; imm8 counts single-word slots, so a double list holds
                // `imm8 / 2` pairs. P = bit24 / U = bit23 select the addressing
                // mode and only IA (P=0,U=1) and DB (P=1,U=0) are valid for
                // VSTM/VLDM, so the writeback sign is simply U.
                let w = (word >> 21) & 1;
                if base_reg == "sp" && w == 1 {
                    let v = (word >> 22) & 1;
                    let nregs = (word & 0xFF) as i64 / if v == 1 { 2 } else { 1 };
                    let bytes = nregs * if v == 1 { 8 } else { 4 };
                    let delta = if l == 1 { bytes } else { -bytes };
                    insns.push(Insn::new(Op::ShiftStack(delta), vec![], addr));
                    return VfpOutcome::Handled(None);
                }
                let up = (word >> 23) & 1 == 1; // U: 1 = increment (IA)
                let imm8 = (word & 0xFF) as u32;
                // Single counts registers directly; double counts word-slots,
                // and the deprecated odd-imm8 LR forms (`fstmiax`) name one
                // register — `(imm8 / 2).max(1)` reproduces both.
                let nregs = if double {
                    (imm8 / 2).max(1)
                } else {
                    imm8.max(1)
                };
                let stride = if double { 8 } else { 4 };
                let first = if double { (d << 4) | vd } else { (vd << 1) | d };
                let base_op = Operand::reg(base_reg);
                for i in 0..nregs {
                    let num = first + i;
                    let fp = if double { dreg(num) } else { sreg(num) };
                    let mem = Operand::Memory {
                        width: 0,
                        base: Box::new(base_op.clone()),
                        index: None,
                        scale: 4,
                        addend: i as i64 * stride,
                    };
                    if l == 1 {
                        insns.push(Insn::move_(fp.clone(), mem, addr));
                        fp_written.insert(reg_name(&fp));
                    } else {
                        insns.push(Insn::move_(mem, fp, addr));
                    }
                }
                if w == 1 {
                    let total = nregs as i64 * stride;
                    insns.push(Insn::new(
                        Op::Arith(if up { ArithKind::Add } else { ArithKind::Sub }),
                        vec![base_op.clone(), base_op, Operand::imm(total)],
                        addr,
                    ));
                }
                return VfpOutcome::Handled(None);
            }
            _ => {}
        }
    }

    VfpOutcome::NotVfp
}

pub fn lift_arm32(bytes: &[u8], base_rva: u64, max_insns: usize) -> Option<LiftResult> {
    let decoder = A32Decoder::default();
    let mut insns: Vec<Insn> = Vec::new();
    let mut addresses: Vec<u64> = Vec::new();
    let mut offset: u64 = 0;
    let mut pending_cmp: Option<PendingCompare> = None;
    // The carry the last flag-setting ADD/SUB/CMP left (see `CarrySource`).
    //
    // Its lifetime is validated by `at` rather than by explicit clears: it is
    // consumed only when a live `pending_cmp` carries the SAME listing index,
    // which is true exactly when the instruction that set the condition is the
    // instruction that produced the carry. A stale source therefore cannot be
    // used — it belongs to an older flag-setting instruction and its `at`
    // differs — and no path can silently read the wrong carry.
    let mut pending_carry: Option<CarrySource> = None;
    let mut scratch = Scratch::new();
    let mut temp_counter: u32 = 0;

    // known constants per register for movw/movt pairs (yaxpeax prints movw
    // as plain `mov rN, 0x…`)
    let mut consts: HashMap<String, i64> = HashMap::new();

    // FP registers written since the last call (AAPCS VFP args s0–s15);
    // mirrors the arm64 lifter so DCE keeps FP chains feeding calls alive.
    let mut fp_written: BTreeSet<String> = BTreeSet::new();

    let cap = max_insns.max(16_384);
    while offset as usize + 4 <= bytes.len() && insns.len() < cap {
        let word = u32::from_le_bytes([
            bytes[offset as usize],
            bytes[offset as usize + 1],
            bytes[offset as usize + 2],
            bytes[offset as usize + 3],
        ]);
        let addr = base_rva + offset;
        if is_vfp(word) {
            match lift_vfp(word, addr, &mut insns, &mut fp_written) {
                VfpOutcome::Handled(cmp) => {
                    // Handled(None) = flags untouched (vldr/vstr/vmov/vmrs all
                    // leave APSR alone) — never clear a pending vcmp. Only a
                    // brand-new compare replaces it.
                    if let Some(c) = cmp {
                        // Re-stamp the capture position: lift_vfp built these
                        // operands at THIS instruction, so pinning must start
                        // here rather than at whatever index lift_vfp saw.
                        pending_cmp = Some(PendingCompare {
                            lhs: c.lhs,
                            rhs: c.rhs,
                            at: insns.len(),
                        });
                    }
                    // Nothing to realign: the reader is rebuilt from `offset`
                    // for every instruction (see the decode site below), so it
                    // cannot drift from the address stamped on the IR.
                    offset += 4;
                    while addresses.len() < insns.len() {
                        addresses.push(addr);
                    }
                    continue;
                }
                VfpOutcome::NotVfp => {}
            }
        }
        // ---- signed DSP multiply (SMLAL / SMUL / SMLA / SMULW / SMLAW) -----
        //
        // Handled straight from the word, BEFORE yaxpeax is consulted, for the
        // same reason `is_vfp` is: the decoder's rendering of these encodings
        // is wrong in ways that are silent (see `dsp_mul`), so the variant and
        // the operand halves have to come from the bit fields. Every register
        // is in the encoding: bits[3:0] = Rn, bits[11:8] = Rm, bits[15:12] =
        // Ra (or RdLo), bits[19:16] = Rd (or RdHi).
        if let Some(d) = dsp_mul(word) {
            let rd = core_reg((word >> 16) & 0xF);
            let ra = core_reg((word >> 12) & 0xF);
            let rm = core_reg((word >> 8) & 0xF);
            let rn = core_reg(word & 0xF);
            // Sign-extend one 16-bit lane: isolate the half the encoding names,
            // then `(x << 16) >> 16`. The arithmetic shift is what makes the
            // extension *signed*, matching `SInt(...)` in the ISA pseudocode.
            let sext =
                |insns: &mut Vec<Insn>, temp: &mut u32, src: Operand, high: bool| -> Operand {
                    let h = if high {
                        let t = temp_local(temp);
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Shr),
                            vec![t.clone(), src, Operand::imm(16)],
                            addr,
                        ));
                        t
                    } else {
                        src
                    };
                    let t = temp_local(temp);
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![t.clone(), h, Operand::imm(16)],
                        addr,
                    ));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shr),
                        vec![t.clone(), t.clone(), Operand::imm(16)],
                        addr,
                    ));
                    t
                };
            // A `movw`/`movt` pair may have pinned a constant to a register
            // this instruction redefines. The product overwrites it, so the
            // stale constant must go rather than survive into a later read as
            // a plausible-looking lie.
            for r in [&rd, &ra] {
                if let Operand::Register(name) = r {
                    consts.remove(name);
                }
            }
            if d.long {
                // `SMLAL<x><y> RdLo, RdHi, Rn, Rm`: RdLo += Rn<sel> * Rm<sel>.
                // The low word is exact in 32-bit arithmetic. The high word is
                // deliberately left undefined rather than fabricated — the
                // treatment `umull` already gets — and its pinned constant, if
                // any, is dropped with it.
                if let Operand::Register(name) = &rd {
                    consts.remove(name);
                }
                let a = sext(&mut insns, &mut temp_counter, rn, d.rn_high);
                let b = sext(&mut insns, &mut temp_counter, rm, d.rm_high);
                insns.push(Insn::new(
                    Op::MulAcc {
                        negate_addend: false,
                        is_float: false,
                    },
                    vec![ra.clone(), a, b, ra],
                    addr,
                ));
            } else {
                let a = if d.word_form {
                    // SMULW/SMLAW keep Rn whole; only Rm is a signed halfword.
                    rn
                } else {
                    sext(&mut insns, &mut temp_counter, rn, d.rn_high)
                };
                let b = sext(&mut insns, &mut temp_counter, rm, d.rm_high);
                if d.word_form || d.accumulate {
                    // A temp is needed whenever the product is not the final
                    // value: SMULW/SMLAW take its `>> 16`, SMLA adds to it.
                    let prod = temp_local(&mut temp_counter);
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Mul),
                        vec![prod.clone(), a, b],
                        addr,
                    ));
                    if d.word_form {
                        // bits [47:16] of the 48-bit product: the whole point
                        // of the word-by-halfword form. `SMULW` shifts straight
                        // into Rd; `SMLAW` needs a temp so the shift is not
                        // clobbered before Ra is added to it.
                        let hi = if d.accumulate {
                            temp_local(&mut temp_counter)
                        } else {
                            rd.clone()
                        };
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Shr),
                            vec![hi.clone(), prod, Operand::imm(16)],
                            addr,
                        ));
                        if d.accumulate {
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::Add),
                                vec![rd, hi, ra],
                                addr,
                            ));
                        }
                    } else {
                        // Halfword `SMLA`: the product IS the addend.
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Add),
                            vec![rd, prod, ra],
                            addr,
                        ));
                    }
                } else {
                    // Halfword × halfword fits signed 32-bit exactly, so a
                    // plain `SMUL` writes Rd directly with no temp or shift.
                    insns.push(Insn::new(Op::Arith(ArithKind::Mul), vec![rd, a, b], addr));
                }
            }
            offset += 4;
            while addresses.len() < insns.len() {
                addresses.push(addr);
            }
            continue;
        }
        // ---- NEON / Advanced SIMD ------------------------------------------
        //
        // See `neon_word`: yaxpeax cannot decode ANY Advanced SIMD word and
        // `is_vfp` intentionally excludes `cond == 0xF`, so these words used to
        // produce NO IR while the machine EXECUTES them — the largest real drop
        // the decode-failure census names (9,362 words of a 3,000-method
        // sample). NEON is far too large to model here, so the word becomes the
        // documented catch-all `Op::Simd` intrinsic (`Op::Simd`'s contract: a
        // missing NAME, never a lost instruction). What matters for control
        // flow is that the word now has IR AT ITS OWN ADDRESS, so a branch to
        // it resolves instead of sliding forward to the next instruction — the
        // `branch target mismatch: native taken` residue.
        //
        // Flags: NEON never writes APSR, so a pending compare is deliberately
        // left pending across this instruction, exactly as `VfpOutcome::Handled(None)`
        // leaves it.
        if neon_word(word) {
            // Operand 0 is the destination vector register (`Vd:D`, the same
            // encoding every NEON shape shares). `Op::Simd`'s invariant is that
            // an instruction which writes a register must DEFINE it, so the op
            // cannot be operand-less — and the register is real even though the
            // operation is not modelled. (For a store the register is a source,
            // but as an opaque placeholder the identity is what matters, and the
            // op is not DCE-removable.)
            let vd = (word >> 12) & 0xF;
            let d = (word >> 22) & 1;
            let dst = dreg((d << 4) | vd);
            fp_written.insert(reg_name(&dst));
            // Name WITHOUT the `__` prefix: the C# renderer adds it, so the
            // output reads `__neon(...)` rather than `____neon(...)`.
            insns.push(Insn::new(Op::Simd("neon".into()), vec![dst], addr));
            offset += 4;
            while addresses.len() < insns.len() {
                addresses.push(addr);
            }
            continue;
        }
        // ---- the EXT group (SXTB/SXTH/UXTB/UXTH/SXTB16/UXTB16) -------------
        //
        // Handled from the word BEFORE yaxpeax is consulted, for the same reason
        // `is_vfp` and `dsp_mul` are: the decoder rejects every encoding in the
        // group, so there is no rendering to fall back on. See `ext_op` for the
        // measurement and for what the silent drop cost.
        if let Some(e) = ext_op(word) {
            // A predicated extension (`uxtheq`) runs only when the condition
            // holds. Dropping the condition would perform a truncation the
            // machine skips — the failure `emit_predication_guard` exists for.
            if word >> 28 != 0xE {
                emit_predication_guard(
                    &mut insns,
                    &mut addresses,
                    &mut scratch,
                    pending_cmp.as_ref(),
                    A32_COND_NAMES[(word >> 28) as usize],
                    addr,
                );
            }
            let dst = core_reg(e.rd as u32);
            let mut src = core_reg(e.rm as u32);
            if e.rot != 0 {
                src = push_shifted(
                    &mut insns,
                    &mut temp_counter,
                    src,
                    ArithKind::Ror,
                    Operand::imm(8 * e.rot as i64),
                    addr,
                );
            }
            // `(x << k) >> k` is how this lifter spells a SIGNED narrowing: the
            // shift pair IS the sign extension (see the `sext` helper in the
            // SMLA arm). `k` is the lane width.
            let narrow =
                |insns: &mut Vec<Insn>, temp: &mut u32, src: Operand, bits: i64| -> Operand {
                    let t = temp_local(temp);
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![t.clone(), src, Operand::imm(32 - bits)],
                        addr,
                    ));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shr),
                        vec![t.clone(), t.clone(), Operand::imm(32 - bits)],
                        addr,
                    ));
                    t
                };
            match e.op {
                // SXTB / SXTH
                1 | 2 => {
                    let bits = if e.op == 1 { 8 } else { 16 };
                    let t = narrow(&mut insns, &mut temp_counter, src, bits);
                    insns.push(Insn::move_(dst.clone(), t, addr));
                }
                // UXTB / UXTH — a plain mask, exactly like the `uxtb`/`uxth` arm.
                4 | 5 => insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![
                        dst.clone(),
                        src,
                        Operand::imm(if e.op == 4 { 0xFF } else { 0xFFFF }),
                    ],
                    addr,
                )),
                // UXTB16 — keep the two LOW bytes in their own lanes.
                3 => insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![dst.clone(), src, Operand::imm(0x00FF_00FF)],
                    addr,
                )),
                // SXTB16 — sign-extend each of the two low bytes into its lane.
                _ => {
                    let lo = narrow(&mut insns, &mut temp_counter, src.clone(), 8);
                    let sh = temp_local(&mut temp_counter);
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shr),
                        vec![sh.clone(), src, Operand::imm(16)],
                        addr,
                    ));
                    let hi = narrow(&mut insns, &mut temp_counter, sh, 8);
                    let shl = temp_local(&mut temp_counter);
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![shl.clone(), hi, Operand::imm(16)],
                        addr,
                    ));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Or),
                        vec![dst.clone(), lo, shl],
                        addr,
                    ));
                }
            }
            if let Operand::Register(name) = &dst {
                consts.remove(name);
            }
            // An extension writes no flags, so a pending compare survives it
            // (same as a plain `mov`).
            offset += 4;
            while addresses.len() < insns.len() {
                addresses.push(addr);
            }
            continue;
        }
        // THE READER IS REBUILT FROM `offset` FOR EVERY INSTRUCTION.
        //
        // yaxpeax's A32 `decode` reads the 4-byte word from the reader FIRST and
        // only then validates the encoding (`words.next_n(&mut word_bytes)?` at
        // the top of `decode_into`), so on `Err` the reader has ALREADY consumed
        // this instruction. The previous code skipped a further 4 bytes in the
        // error arm — advancing the reader by 8 for one 4-byte instruction. The
        // shared reader then ran one word ahead of `offset` per undecodable
        // instruction, and STAYED ahead: every later IR instruction was stamped
        // with an address one word too EARLY, so `addr + branch_offset` resolved
        // control flow to the wrong instruction and the C# was attached to the
        // wrong RVA. Real case in libil2cpp.so: `0x53c840` is `vldr s2` (VFP,
        // which this crate cannot decode), and the `ldm sp, {r0, r1}` at
        // `0x53c868` came out labelled `0x53c854`.
        //
        // Deriving the reader from `offset` makes that drift structurally
        // impossible: the word used for `is_vfp`, the decoded text, and the
        // address stamped on the IR all come from the same position by
        // construction, with no arithmetic to keep in step.
        let mut reader = U8Reader::new(&bytes[offset as usize..]);
        let insn: A32Insn = match decoder.decode(&mut reader) {
            Ok(i) => i,
            // Undecodable word (data, padding, VFP/NEON, exotic coprocessor):
            // emit nothing and step over exactly one A32 instruction, which is
            // always 4 bytes. Do NOT consume from the reader — `decode` did.
            Err(_) => {
                // Words yaxpeax rejects that the LIFTER reconstructs from the
                // raw bit fields (see `dropped_op`). Handled here, AFTER the
                // decoder refuses, so the recognizer can never shadow a word
                // the decoder gets right. These instructions write no flags, so
                // any pending compare survives them (the `mov` rule).
                if let Some(op) = dropped_op(word) {
                    emit_dropped_op(op, addr, &mut insns, &mut temp_counter, &mut consts);
                } else if vfp_predicated_word(word) {
                    // A predicated VFP op (`vmovgt.f32 s16, s0`). Define the
                    // destination register so the word has a real landing site;
                    // the condition itself is not carried (the same trade the
                    // NEON catch-all makes).
                    let vd = (word >> 12) & 0xF;
                    let d = (word >> 22) & 1;
                    let dst = if (word >> 8) & 1 == 1 {
                        dreg((d << 4) | vd)
                    } else {
                        sreg((vd << 1) | d)
                    };
                    fp_written.insert(reg_name(&dst));
                    insns.push(Insn::new(Op::Simd("__vfp".into()), vec![dst], addr));
                }
                while addresses.len() < insns.len() {
                    addresses.push(addr);
                }
                offset += 4;
                continue;
            }
        };
        let text = format!("{insn}");
        let (mne, cond, rest) = split(&text);
        let flags = sets_flags(mne, &text);

        match mne {
            "mov" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                // `mov pc, rm` is the second half of the ARM32 jump-table
                // dispatch (`add ls, pc, rm, lsl #2; ldr pc, [ls, ...]`) and
                // `mov pc, lr` the RETURN. Treated as a plain write to the `pc`
                // register it produced no terminator, so the block fell through
                // into whatever bytes follow — the same inline-jump-table leak
                // the `add pc` case fixes.
                if matches!(parts.first().map(|s| s.trim()), Some("pc") | Some("r15")) {
                    let is_ret = parts
                        .get(1)
                        .is_some_and(|s| matches!(s.trim(), "lr" | "r14"));
                    if is_ret {
                        insns.push(Insn::new(Op::Return, vec![Operand::reg("x0")], addr));
                    } else {
                        // Computed goto: IndirectCall so the table recovery can
                        // still see it; it terminates either way.
                        insns.push(Insn::new(
                            Op::IndirectCall,
                            vec![
                                reg(parts.get(1).copied().unwrap_or("").trim()),
                                Operand::Immediate(-1),
                            ],
                            addr,
                        ));
                    }
                    pending_cmp = None;
                    while addresses.len() < insns.len() {
                        addresses.push(addr);
                    }
                    offset += insn.len().to_const() as u64;
                    continue;
                }
                let dst_op = reg(parts.first().copied().unwrap_or(""));
                // The destination is (re)defined here, so any pinned constant
                // for it is stale. The immediate form re-pins below; a
                // PREDICATED form never pins (its value is conditional).
                if let Operand::Register(name) = &dst_op {
                    consts.remove(name);
                }
                // Resolve the source into an operand (a shifted source may emit
                // its own IR first). The three spellings are the original ones:
                // register-shifted (yaxpeax prints an artifact middle operand),
                // plain register, and immediate.
                let src_op: Option<Operand> = if parts.len() >= 4 {
                    let src = reg(parts[2]);
                    parse_shifted_src(&format!("x, {}", parts[3])).map(|(_, kind, amt)| {
                        push_shifted(&mut insns, &mut temp_counter, src, kind, amt, addr)
                    })
                } else if parts.len() == 3 {
                    Some(reg(parts[2]))
                } else {
                    let p1 = parts.get(1).copied().unwrap_or("0");
                    Some(match parse_int(p1) {
                        Some(v) => {
                            if let Operand::Register(name) = &dst_op {
                                consts.insert(name.clone(), v);
                            }
                            Operand::imm(v)
                        }
                        None => reg(p1),
                    })
                };
                if let Some(src_op) = src_op {
                    if !cond.is_empty() {
                        // A32 PREDICATES every instruction: `movne r6, #1`
                        // executes only when NE holds, so otherwise `r6` keeps
                        // its PREVIOUS value. Lifting it as an unconditional
                        // `r6 = 1` was a WRONG-LOGIC error — the whole
                        // `bool b = (x != 0);` materialisation (`cmp; movne
                        // r6, #1`) became `r6 = 1`. Model it as a SELECT rather
                        // than a guard: `dst = cond ? src : dst`. It is exactly
                        // the arm64 `CSEL` shape, it needs no new basic block,
                        // and the false operand is the destination itself, which
                        // the SSA renamer resolves to the version live before
                        // this instruction. (A guard — a branch over the move —
                        // is equivalent, but it splits a block at every
                        // predicate and the renamer then leaves entry versions
                        // undefined across the new edges.)
                        let s = materialize_pending(
                            &mut insns,
                            &mut addresses,
                            &mut scratch,
                            pending_cmp.as_ref(),
                            cond_to_kind(cond),
                            addr,
                        );
                        insns.push(Insn::new(
                            Op::Select(SelectVariant::Identity),
                            vec![dst_op.clone(), s, src_op, dst_op],
                            addr,
                        ));
                    } else {
                        insns.push(Insn::move_(dst_op, src_op, addr));
                    }
                }
                // NOTE: the non-`s` case deliberately does NOT clear. ARM32
                // writes APSR only from the S-suffixed data-processing forms
                // and cmp/cmn/tst/teq; plain `mov` PRESERVES it, exactly like
                // AArch64's MOV. The compiler routinely shuffles registers
                // between a `cmp` and the `b<cc>` that reads it, so clearing
                // here discarded a live condition and rendered the branch as
                // `if (? != 0)` — the same defect already fixed for arm64 (see
                // `preserves_nzcv` in lift.rs). Every other site in this
                // matcher follows that rule: clear only for what really writes
                // APRS, or for a control transfer that ends the linear path.
                if flags {
                    pending_cmp = Some(PendingCompare {
                        lhs: reg(parts.first().copied().unwrap_or("")),
                        rhs: Operand::imm(0),
                        at: insns.len(),
                    });
                }
            }
            "movt" => {
                // high half — combine with the low half from the preceding movw
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst_op = reg(parts.first().copied().unwrap_or(""));
                let hi = parts.get(1).and_then(|s| parse_int(s)).unwrap_or(0);
                // the consts table is keyed by the MAPPED name ("x0"), same
                // as the movw arm stores it — using the raw "r0" here lost
                // the low half entirely.
                let dname = match &dst_op {
                    Operand::Register(n) => n.clone(),
                    _ => String::new(),
                };
                let lo = consts.get(&dname).copied().unwrap_or(0) & 0xFFFF;
                let full = ((hi << 16) & !0xFFFF) | lo;
                consts.insert(dname.clone(), full);
                insns.push(Insn::move_(dst_op, Operand::imm(full), addr));
                // `movt` preserves APSR (see the note in the `mov` arm)
            }
            "mvn" => {
                // "mvn r5, r6" (2 operands) or "mvn r0, r0, 0x0" (the middle
                // r0 is a yaxpeax artifact on the immediate form)
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let src_text = if parts.len() >= 3 {
                    parts[2]
                } else {
                    parts.get(1).copied().unwrap_or("0")
                };
                let src_op = match parse_int(src_text) {
                    Some(v) => Operand::imm(v),
                    None => reg(src_text),
                };
                if !cond.is_empty() {
                    // Predicated `mvnne r0, r0`: `dst = cond ? ~src : dst`, the
                    // same SELECT shape as the predicated `mov` above.
                    let t = push_not_temp(&mut insns, &mut temp_counter, src_op, addr);
                    let s = materialize_pending(
                        &mut insns,
                        &mut addresses,
                        &mut scratch,
                        pending_cmp.as_ref(),
                        cond_to_kind(cond),
                        addr,
                    );
                    insns.push(Insn::new(
                        Op::Select(SelectVariant::Identity),
                        vec![dst.clone(), s, t, dst],
                        addr,
                    ));
                } else {
                    push_not(&mut insns, dst, src_op, addr);
                }
                // plain `mvn` preserves APSR (only `movns` writes it)
            }
            "add" | "sub" | "rsb" | "and" | "orr" | "eor" | "bic" => {
                let mut parts = rest.splitn(3, ", ");
                let dst_s = parts.next().unwrap_or("");
                let a_raw = parts.next().unwrap_or("");
                let b_raw = parts.next().unwrap_or("");
                let kind = match mne {
                    "add" => ArithKind::Add,
                    "sub" => ArithKind::Sub,
                    "rsb" => ArithKind::Sub,
                    "and" => ArithKind::And,
                    "orr" => ArithKind::Or,
                    "eor" => ArithKind::Xor,
                    _ => ArithKind::And, // bic
                };
                // A write to `pc` IS the control transfer, not arithmetic.
                // `add pc, r1, r2` is the ARM32 jump-table dispatch (table entry
                // + table base), `add pc, pc, rm, lsl #n` a computed goto, and
                // `sub pc, pc, #4` a relative branch. Modelling it as `Arith`
                // on the `pc` register produced NO terminator, so the block
                // fell through and the lifter decoded the INLINE JUMP TABLE as
                // instructions — phantom registers (`undef-local origin x12`,
                // the second-largest cluster after `sp`) and a garbage receiver
                // expression (`(this & this << 0 & in_x8 << 0 & in_x12 << 0 …)
                // .burterCasting()` at `VFXArenaCaster.Update` @0x625E00). The
                // target cannot be resolved here, so end the linear path; the
                // CFG drops the now-unreachable case stubs and the verifier's
                // reachability already excludes the table itself.
                if matches!(dst_s.trim(), "pc" | "r15") {
                    // `add pc, r1, r2` — the jump-table dispatch: `r2` is the table
                    // base and `r1` the loaded entry. Emit an INDIRECT CALL (not a
                    // bare exit) with the entry as its operand, so
                    // `passes::recover_switch_tables` can read the table and
                    // rewrite it into `Op::Switch` — restoring the case bodies,
                    // which a bare terminator would drop. When recovery does not
                    // fire the `IndirectCall` still terminates the block, so the
                    // inline table is never decoded as fall-through code.
                    let src = if matches!(a_raw.trim(), "pc" | "r15") {
                        b_raw.trim()
                    } else {
                        a_raw.trim()
                    };
                    // The `-1` marker operand tells `recover_switch_tables` this
                    // is a PC-dispatch, not a `bx rN` virtual call: if the table
                    // cannot be read the pass turns it back into an exit rather
                    // than leaving an unresolved indirect call.
                    insns.push(Insn::new(
                        Op::IndirectCall,
                        vec![reg(src), Operand::Immediate(-1)],
                        addr,
                    ));
                    pending_cmp = None;
                    while addresses.len() < insns.len() {
                        addresses.push(addr);
                    }
                    offset += insn.len().to_const() as u64;
                    continue;
                }
                // PC-relative ADDRESS materialization: `add r0, pc, #8`
                // computes the address of a nearby literal or function, and
                // the pipeline value of `pc` is `addr + 8`. Treated as ordinary
                // arithmetic on the (undefined) g15 register, the ADDRESS —
                // and everything reached through it — was lost. As a constant
                // it is exact, and it is how every RIP-relative-style access
                // on arm32 is spelled.
                if matches!(mne, "add" | "sub") && matches!(a_raw.trim(), "pc" | "r15") {
                    let base = addr as i64 + 8;
                    let is_sub = mne == "sub";
                    let arith = if is_sub {
                        ArithKind::Sub
                    } else {
                        ArithKind::Add
                    };
                    if b_raw.trim().is_empty() {
                        // `add rd, pc, #0` printed with the offset elided.
                        insns.push(Insn::move_(reg(dst_s), Operand::imm(base), addr));
                    } else if let Some(v) = parse_field(b_raw) {
                        // `add rd, pc, #imm` — a literal / function address:
                        // exact as a constant.
                        let val = base + if is_sub { -v } else { v };
                        insns.push(Insn::move_(reg(dst_s), Operand::imm(val), addr));
                    } else if let Some((breg, skind, amt)) = parse_shifted_src(b_raw) {
                        // `add rd, pc, rm, lsl #n` — the JUMP-TABLE idiom.
                        // `pc` is an address and `rm` a live index, so this is
                        // real arithmetic, not a constant.
                        let t = push_shifted(&mut insns, &mut temp_counter, breg, skind, amt, addr);
                        insns.push(Insn::new(
                            Op::Arith(arith),
                            vec![reg(dst_s), Operand::imm(base), t],
                            addr,
                        ));
                    } else {
                        // `add rd, pc, rm` — `rm` holds a VALUE that was just
                        // loaded from the literal pool (the GOT / statics /
                        // metadata-registration idiom). This used to fall into
                        // the `#imm` branch above with `unwrap_or(0)`, so the
                        // register was DROPPED and the result collapsed to
                        // `pc` alone: a plausible-looking but WRONG constant
                        // address, and everything reached through it lost.
                        insns.push(Insn::new(
                            Op::Arith(arith),
                            vec![reg(dst_s), Operand::imm(base), reg(b_raw.trim())],
                            addr,
                        ));
                    }
                    while addresses.len() < insns.len() {
                        addresses.push(addr);
                    }
                    offset += insn.len().to_const() as u64;
                    continue;
                }
                if dst_s == "sp" && matches!(kind, ArithKind::Add | ArithKind::Sub) {
                    let v = parse_int(b_raw).unwrap_or(0);
                    let delta = if kind == ArithKind::Add { v } else { -v };
                    insns.push(Insn::new(Op::ShiftStack(delta), vec![], addr));
                    // stack pointer arithmetic preserves APSR
                    while addresses.len() < insns.len() {
                        addresses.push(addr);
                    }
                    offset += insn.len().to_const() as u64;
                    continue;
                }
                // `add Rd, sp, #imm` with Rd != sp computes the ADDRESS of a
                // stack slot into Rd — the frame-pointer setup the compiler
                // emits at the top of nearly every non-leaf function
                // (`add fp, sp, #8`, i.e. `r11 = sp + 8`), and the
                // `&local` / `&enumerator` idiom it emits for every struct-typed
                // local passed by reference.
                //
                // Left as plain arithmetic it was fatal in two ways, and
                // together they were the largest arm32 defect class:
                //
                //  1. The `sp` it reads is NEVER defined in this IR (the frame
                //     setup is modelled as `ShiftStack`, which carries the depth
                //     but defines no register), so the read became an undefined
                //     local — `undef-local origin sp`, 11,818 findings, the
                //     second-largest cluster in the census and the reason every
                //     `[fp + N]` access in those methods stayed register-relative.
                //  2. `stack_analyzer::frame_alias_def` only recognises the
                //     ALIAS shape (`Move xD, AddressOf(Memory { base: sp, k })`),
                //     which is exactly what arm64's lifter emits for
                //     `add x29, sp, #k` (see the `Rd = &slot` arm in `lift.rs`).
                //     Without it no alias was recorded, so the frame pointer was
                //     never resolved into `stack_N` cells and the frame slot's
                //     store and load got two different names.
                //
                // Mirroring arm64's shape gives the two architectures ONE frame
                // model; `ptr_size` is 4 here, so the slot scale is 4.
                if kind == ArithKind::Add && a_raw.trim() == "sp" {
                    if let Some(v) = parse_field(b_raw) {
                        insns.push(Insn::move_(
                            reg(dst_s),
                            Operand::AddressOf(Box::new(Operand::Memory {
                                width: 0,
                                base: Box::new(Operand::reg("sp")),
                                index: None,
                                scale: 4,
                                addend: v,
                            })),
                            addr,
                        ));
                        // stack-address arithmetic preserves APSR
                        while addresses.len() < insns.len() {
                            addresses.push(addr);
                        }
                        offset += insn.len().to_const() as u64;
                        continue;
                    }
                }
                let d = reg(dst_s);
                // A predicated ALU op (`orrne r3, lr, ip, lsl r1`) is
                // `d = cond ? op(a, b) : d`, NOT `d = op(a, b)`. Decided here —
                // BEFORE the operand temps below are emitted — so the back-scan
                // sees the instruction that actually wrote `d` (the `bic` of the
                // conditional-bit idiom), not this instruction's own scratch.
                // `sp`/`g15` are excluded: the stack analyzer rewrites `sp`
                // writes, so reading them on the false path surfaces as an
                // undefined local rather than a value.
                let d_is_stack_or_pc = matches!(
                    &d,
                    Operand::Register(n) if n == "sp" || n == "g15"
                );
                let predicated = !cond.is_empty() && !d_is_stack_or_pc && has_prior_def(&insns, &d);
                // NOTE: a predicated ALU op (`orrne r3, lr, ip`, `addne r0, r0,
                // #1`) is semantically `d = cond ? op(a, b) : d`, but lowering
                // it to that SELECT is a NET LOSS on arm32 (measured: clean
                // 52,401 -> 51,667, undefined locals 15,723 -> 19,247). The
                // false operand forces a READ of `d`, so when the predicated
                // op is the first writer of `d` the read becomes an undefined
                // local — the same failure mode that sank the blanket
                // predication guard. Unlike the predicated `mov` (where the
                // SELECT removed a WRONG-LOGIC and won), the unconditional
                // compute is the better shape here, so keep it.
                let (a_op, b_op);
                if mne == "rsb" {
                    // rsb rd, rn, op2  ==  rd = op2 - rn  (swap)
                    if let Some((base_reg, skind, amt)) =
                        parse_shifted_src(&format!("{b_raw}, {a_raw}"))
                    {
                        let t =
                            push_shifted(&mut insns, &mut temp_counter, base_reg, skind, amt, addr);
                        a_op = t;
                        b_op = reg(a_raw.trim());
                    } else {
                        a_op = match parse_int(b_raw) {
                            Some(v) => Operand::imm(v),
                            None => reg(b_raw.trim()),
                        };
                        b_op = reg(a_raw.trim());
                    }
                } else if let Some((base_reg, skind, amt)) = parse_shifted_src(b_raw) {
                    a_op = reg(a_raw.trim());
                    let t = push_shifted(&mut insns, &mut temp_counter, base_reg, skind, amt, addr);
                    b_op = t;
                } else {
                    a_op = reg(a_raw.trim());
                    b_op = match parse_int(b_raw) {
                        Some(v) => Operand::imm(v),
                        None => reg(b_raw.trim()),
                    };
                }
                // The SELECT shape below is only taken when `predicated` was
                // set above: it makes the `bic` live and turns the conditional
                // set into `d = cond ? op : d`, exactly what the machine does.
                // Otherwise the unconditional compute is kept, because the
                // SELECT's false operand would be an undefined local — the
                // measured reason the earlier BLANKET select was a net loss.
                let out = if predicated {
                    temp_local(&mut temp_counter)
                } else {
                    d.clone()
                };
                if mne == "bic" {
                    // d = a & ~b — honest two-step (b ^ -1, then and)
                    let t = push_not_temp(&mut insns, &mut temp_counter, b_op.clone(), addr);
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::And),
                        vec![out.clone(), a_op.clone(), t],
                        addr,
                    ));
                } else {
                    insns.push(Insn::new(
                        Op::Arith(kind),
                        vec![out.clone(), a_op.clone(), b_op.clone()],
                        addr,
                    ));
                }
                if predicated {
                    // Materialize the `Compare` at the COMPARE's own address,
                    // not this consumer's: the folded `cmp` keeps only a Nop at
                    // its address, so the census exempts it via the "next insn
                    // hosts a Compare" rule — which fails when a predicated load
                    // sits between the `cmp` and the ALU that consumes it
                    // (`cmp r1,#0` ; `ldrne r1,[…]` ; `addne r1,r1,#1`). Placing
                    // the Compare at the `cmp` covers it directly. Branches keep
                    // their existing consumer-address materialization.
                    let cmp_addr = pending_cmp
                        .as_ref()
                        .and_then(|c| insns.get(c.at))
                        .map(|i| i.address)
                        .unwrap_or(addr);
                    let s = materialize_pending(
                        &mut insns,
                        &mut addresses,
                        &mut scratch,
                        pending_cmp.as_ref(),
                        cond_to_kind(cond),
                        cmp_addr,
                    );
                    insns.push(Insn::new(
                        Op::Select(SelectVariant::Identity),
                        vec![d.clone(), s, out.clone(), d.clone()],
                        addr,
                    ));
                }
                if flags {
                    let at = insns.len();
                    pending_cmp = Some(PendingCompare {
                        lhs: d,
                        rhs: Operand::imm(0),
                        at,
                    });
                    // ADD/SUB/RSB set C; AND/ORR/EOR/BIC leave it UNCHANGED, so
                    // only the arithmetic forms publish a carry source here.
                    // (`rsb rd, rn, op2` computes `op2 - rn`, and the swapped
                    // operands above already spell it that way.)
                    pending_carry = match kind {
                        ArithKind::Add => Some(CarrySource {
                            subtract: false,
                            lhs: a_op.clone(),
                            rhs: b_op.clone(),
                            at,
                        }),
                        ArithKind::Sub => Some(CarrySource {
                            subtract: true,
                            lhs: a_op.clone(),
                            rhs: b_op.clone(),
                            at,
                        }),
                        _ => None,
                    };
                }
                // else: keep — only the `s`-forms write APSR, so a plain
                // `add/sub/and/orr/eor/bic` must not discard a live compare
                // (see the note in the `mov` arm).
                //
                // NOTE: the `a_op`/`b_op` copies above are what `ADC`/`SBC`
                // need — they are the operands the ADD actually combined, not
                // the `result vs 0` pair the condition uses.
            }
            // ---- carry-consuming arithmetic -------------------------------
            //
            // `adc rd, rn, rm` = rn + rm + C ; `sbc rd, rn, rm` = rn - rm - !C ;
            // `rsc rd, rn, rm` = rm - rn - !C. These three ARE how A32 spells
            // arithmetic wider than 32 bits, so the carry is not a detail — it
            // is the operand that carries the high word. A32 sets C on
            // subtraction to NOT-borrow, which is why SBC/RSC subtract `1 - C`.
            "adc" | "sbc" | "rsc" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let rn = match parse_int(parts.get(1).copied().unwrap_or("0")) {
                    Some(v) => Operand::imm(v),
                    None => reg(parts.get(1).copied().unwrap_or("")),
                };
                let rm = match parse_int(parts.get(2).copied().unwrap_or("0")) {
                    Some(v) => Operand::imm(v),
                    None => reg(parts.get(2).copied().unwrap_or("")),
                };
                // The carry is usable only when a LIVE condition came from the
                // very instruction that produced it (see `pending_carry`).
                let src = pending_carry
                    .clone()
                    .filter(|c| pending_cmp.as_ref().map(|p| p.at) == Some(c.at));
                match src {
                    Some(src) => {
                        let subtract = matches!(mne, "sbc" | "rsc");
                        // SBC/RSC need the BORROW (`!C`); ADC needs C itself.
                        let cf = materialize_carry(
                            &mut insns,
                            &mut addresses,
                            &mut scratch,
                            &src,
                            subtract,
                            addr,
                        );
                        let (lhs, rhs) = if mne == "rsc" { (rm, rn) } else { (rn, rm) };
                        let kind = if subtract {
                            ArithKind::Sub
                        } else {
                            ArithKind::Add
                        };
                        let t = temp_local(&mut temp_counter);
                        insns.push(Insn::new(Op::Arith(kind), vec![t.clone(), lhs, rhs], addr));
                        // `- borrow` / `+ carry` : a SUB of the borrow, an ADD
                        // of the carry — one extra step either way, and typed
                        // as an integer because `materialize_carry` yields 0/1.
                        insns.push(Insn::new(Op::Arith(kind), vec![dst.clone(), t, cf], addr));
                    }
                    None => {
                        // No live carry to consume: keep every operand as a
                        // VALUE rather than inventing a carry the hardware
                        // did not use. A wrong carry is worse than none.
                        insns.push(Insn::new(
                            Op::Simd(format!("__{mne}_nocarry")),
                            vec![dst.clone(), rn, rm],
                            addr,
                        ));
                    }
                }
                if flags {
                    // ADCS/SBCS write APSR — including C — but the carry they
                    // leave depends on the full widened sum; do not publish a
                    // source rather than publish a wrong one.
                    let at = insns.len();
                    pending_cmp = Some(PendingCompare {
                        lhs: dst,
                        rhs: Operand::imm(0),
                        at,
                    });
                    pending_carry = None;
                }
            }
            "cmp" | "tst" => {
                // yaxpeax prints these with a leading artifact operand:
                // "cmps r0, r5, r2" — the real comparison is r5 vs r2
                let parts: Vec<&str> = rest.split(", ").collect();
                let lhs = reg(parts.get(1).copied().unwrap_or(""));
                let rhs_text = parts.get(2).copied().unwrap_or("0");
                let rhs = match parse_int(rhs_text) {
                    Some(v) => Operand::imm(v),
                    None => reg(rhs_text),
                };
                let at = insns.len();
                // CMP is a SUB that discards its result, and CMN is an ADD: both
                // publish the carry for a following `adc`/`sbc` (the compiler
                // uses `cmp`+`sbc` in unsigned 64-bit comparisons).
                pending_carry = Some(CarrySource {
                    subtract: mne == "cmp",
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    at,
                });
                pending_cmp = Some(PendingCompare { lhs, rhs, at });
                // Reserve the compare's own address. A loop back-edge targets
                // the `cmp`, and without an entry in `addresses` that edge
                // resolves to the CFG exit sentinel and collapses the loop.
                insns.push(Insn::new(Op::Nop, vec![], addr));
            }
            "ldr" | "ldrb" | "ldrh" | "ldrsh" | "ldrsb" | "str" | "strb" | "strh" | "ldrbt"
            | "ldrht" | "ldrsbt" | "ldrsht" | "ldrt" | "strbt" | "strht" | "strt" => {
                // The unprivileged `t` forms load and store exactly like their
                // ordinary counterparts (see `canonical_mem_mnemonic`), so the
                // arm below is shared rather than duplicated.
                let mne = canonical_mem_mnemonic(mne);
                let is_load = mne.starts_with("ldr");
                // ACCESS WIDTH: `ldrb`/`ldrh` and their store twins touch ONE
                // or TWO bytes, but they lifted to exactly the same IR as
                // `ldr`/`str` — so every `bool`/`byte`/`char`/byte-enum field
                // access rendered as an eight-byte `*(nint*)`, and a `strb`
                // store claimed to write eight bytes over a one-byte slot.
                let width = match mne {
                    "ldrb" | "ldrsb" | "strb" => 1u8,
                    "ldrh" | "ldrsh" | "strh" => 2,
                    _ => 0,
                };
                let src_reg = reg(rest.split(',').next().unwrap_or("").trim());
                // Full A32 addressing: pre-index + writeback, post-index,
                // shifted register index, and the literal-pool `[pc, #off]`
                // form. (The old parser trimmed a trailing `!` and ignored
                // everything after the closing bracket, so the base update
                // both indexed AND post-indexed forms perform was dropped.)
                let acc = parse_mem_access(rest).unwrap_or(MemAccess {
                    base: reg("sp"),
                    index: None,
                    shift: 0,
                    addend: 0,
                    writeback: false,
                    subtract: false,
                });
                let (mem, writeback_delta) = mem_operand(&acc, &mut insns, &mut temp_counter, addr);
                let mem = mem.with_width(width);
                if is_load {
                    insns.push(Insn::move_(src_reg.clone(), mem, addr));
                    // `ldrsh`/`ldrsb` sign-extend the loaded halfword/byte into
                    // the full register: `(v << k) >> k` with k = 16/24, the
                    // same lowering arm64's SBFM uses. Without it the load's
                    // destination was defined as the narrow value and every
                    // dependent use lost its sign.
                    if let Some(k) = match mne {
                        "ldrsh" => Some(16i64),
                        "ldrsb" => Some(24),
                        _ => None,
                    } {
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Shl),
                            vec![src_reg.clone(), src_reg.clone(), Operand::imm(k)],
                            addr,
                        ));
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Shr),
                            vec![src_reg.clone(), src_reg.clone(), Operand::imm(k)],
                            addr,
                        ));
                    }
                } else {
                    insns.push(Insn::move_(mem, src_reg, addr));
                }
                if let Some(delta) = writeback_delta {
                    push_writeback(&mut insns, &acc, delta, addr);
                }
                // loads and stores preserve APSR (see the note in the `mov`
                // arm) — a `cmp; ldr; b.eq` sequence keeps its condition
            }
            "mul" => {
                // "mul r0, r2, r1" = rd = rn * rm
                let parts: Vec<&str> = rest.split(", ").collect();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Mul),
                    vec![
                        reg(parts.first().copied().unwrap_or("")),
                        reg(parts.get(1).copied().unwrap_or("")),
                        reg(parts.get(2).copied().unwrap_or("")),
                    ],
                    addr,
                ));
                // MUL/MLA/UMULL/CLZ do not write APSR on ARM32
            }
            "mla" => {
                // "mla r0, r1, r2, r0" = rd = rn*rm + ra (fused)
                let parts: Vec<&str> = rest.split(", ").collect();
                insns.push(Insn::new(
                    Op::MulAcc {
                        negate_addend: false,
                        is_float: false,
                    },
                    vec![
                        reg(parts.first().copied().unwrap_or("")),
                        reg(parts.get(1).copied().unwrap_or("")),
                        reg(parts.get(2).copied().unwrap_or("")),
                        reg(parts.get(3).copied().unwrap_or("")),
                    ],
                    addr,
                ));
            }
            "umull" | "smull" => {
                // "umull lo, hi, rn, rm" — the low word is the exact 32-bit
                // product (truncation built in); the high word is left
                // undefined rather than silently wrong. The LOW half of a
                // signed `smull` is the same 32-bit truncation, so both share
                // this arm: what differs is only the upper half.
                let parts: Vec<&str> = rest.split(", ").collect();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Mul),
                    vec![
                        reg(parts.first().copied().unwrap_or("")),
                        reg(parts.get(2).copied().unwrap_or("")),
                        reg(parts.get(3).copied().unwrap_or("")),
                    ],
                    addr,
                ));
            }
            "clz" => {
                // "clz r0, r2" -> CountLeadingZeros intrinsic (arm64 parity)
                let parts: Vec<&str> = rest.split(", ").collect();
                insns.push(Insn::new(
                    Op::Clz,
                    vec![
                        reg(parts.first().copied().unwrap_or("")),
                        reg(parts.get(1).copied().unwrap_or("")),
                    ],
                    addr,
                ));
            }
            "push" => {
                let n = rest.matches(',').count() + 1;
                insns.push(Insn::new(Op::ShiftStack(-(n as i64) * 4), vec![], addr));
                // `push` preserves APSR
            }
            "pop" => {
                let n = rest.matches(',').count() + 1;
                // `popeq {…, pc}` is a CONDITIONAL RETURN: it happens only when
                // the flags say so. The guard goes in front of it, so the
                // fall-through stays reachable — see `emit_predication_guard`.
                if rest.contains("pc") {
                    emit_predication_guard(
                        &mut insns,
                        &mut addresses,
                        &mut scratch,
                        pending_cmp.as_ref(),
                        cond,
                        addr,
                    );
                }
                insns.push(Insn::new(Op::ShiftStack((n as i64) * 4), vec![], addr));
                if rest.contains("pc") {
                    // NOTE: do NOT stop decode at the first return (the same
                    // fix as lift_x86). ARM compilers lay out cold/error
                    // paths AFTER an early `pop {.., pc}`; stopping here left
                    // every CondJump to post-ret code unresolved → empty
                    // bodies. The CFG splits blocks at terminators and drops
                    // genuinely unreachable code.
                    insns.push(Insn::new(Op::Return, vec![Operand::reg("x0")], addr));
                    // A return ends the linear path, so a compare pending across
                    // an UNCONDITIONAL return must not be handed to a branch laid
                    // out after it. A PREDICATED return (`poplt {…, pc}`) is
                    // different: on the fall-through path it does NOT execute,
                    // and a return does not write APSR, so the flags are still
                    // live at the next instruction. Clearing here made the `bne`
                    // after `cmp; poplt {…,pc}` read no compare and render
                    // `if (? != 0)` — a real wrong-logic site (e.g. @0xe60c30).
                    if cond.is_empty() {
                        pending_cmp = None;
                    }
                }
                // a plain `pop` (no pc) preserves APSR — keep the compare
            }
            "b" if cond.is_empty() => {
                let target = parse_target(rest.trim_start_matches('#'), addr);
                // usize::MAX = unresolvable → CFG maps it to an exit edge
                insns.push(Insn::new(
                    Op::Jump(target.unwrap_or(usize::MAX)),
                    vec![],
                    addr,
                ));
                // An unconditional `b` does NOT write APSR, so the pending
                // compare is still live at its target: `cmp; bls L1; b L2;
                // L1: bhi L3` re-tests the SAME flags in the second branch.
                // Clearing here left `L1`'s branch with `Compare(kind,
                // Unknown, 0)` → `if (? != 0)`. This matches arm64, whose
                // `preserves_nzcv` already carries `B` across the pending
                // compare. A later flag writer replaces `pending_cmp` itself,
                // so no stale compare can survive a real flag write.
            }
            "bl" | "blx" if cond.is_empty() => {
                let target = parse_target(rest.trim_start_matches('#'), addr);
                // `bl`/`blx` ALWAYS RETURN. A register-target `blx r2` is an
                // ordinary indirect CALL, never a tail transfer — the tail form
                // is `bx rN`, delivered through the `"bx"` arm below. Emitting
                // `Op::IndirectCall` for the register form made a returning call
                // the CFG's TERMINATOR (`cfg.rs` gives IndirectCall no
                // successors), so the fall-through after every unresolved ARM32
                // virtual call was ORPHANED from the graph: SSA never renamed
                // it, its phis never filled, and DCE deleted the live
                // computation after the call. This is the ARM32 twin of the x86
                // `call r/m` defect (see IL_RECOVERY.md) — x86 was fixed to
                // `Op::Call`, ARM32's `blx` was not. The operand layout below is
                // already the `Op::Call` contract `[target, ret-slot, x0…x3, fp…]`,
                // so only the op changes; `bx rN` keeps `IndirectCall`.
                let call_op = Op::Call;
                let mut ops = vec![match target {
                    Some(t) => Operand::imm(t as i64),
                    // Canonicalize through `reg()`: the decoder prints `blx r2`
                    // (not `x2`), and a raw `Register("r2")` is a name no def
                    // ever writes — SSA never renames it, so the indirect-call
                    // target stayed a phantom and every ARM32 virtual dispatch
                    // read as "target is not a value".
                    None => reg(rest),
                }];
                ops.push(Operand::reg("x0")); // return-value slot
                for r in ["x0", "x1", "x2", "x3"] {
                    ops.push(Operand::reg(r));
                }
                // AAPCS VFP args (s0–s15/d0–d7): record only FP regs written
                // since the last call — same reasoning as the arm64 BL arm.
                for r in fp_written.iter() {
                    ops.push(Operand::reg(r.clone()));
                }
                fp_written.clear();
                insns.push(Insn::new(call_op, ops, addr));
                pending_cmp = None;
            }
            // conditional branch "b<cc> <target>": Compare + CondJump pair
            // consuming the pending flags compare (same address, 2 IR insns)
            "b" if !cond.is_empty() => {
                let target = parse_target(rest.trim_start_matches('#'), addr).unwrap_or(usize::MAX);
                // READ the flags, do not consume them: an if/else-if chain
                // (`cmp; bne L1; beq L2`) re-tests the SAME APSR with the second
                // branch, and `.take()` here destroyed the compare so the second
                // branch rendered as `if (? != 0)`. Every instruction that really
                // writes APSR clears `pending_cmp` on its own, which is the same
                // conservative model the arm64 lifter uses.
                let s = materialize_pending(
                    &mut insns,
                    &mut addresses,
                    &mut scratch,
                    pending_cmp.as_ref(),
                    cond_to_kind(cond),
                    addr,
                );
                insns.push(Insn::new(Op::CondJump(target), vec![s], addr));
            }
            // conditional call "bl<cc>" — predicated execution approximated
            // as an unconditional call (rare in managed code)
            "bl" | "blx" if !cond.is_empty() => {
                let target = parse_target(rest.trim_start_matches('#'), addr);
                let mut ops = vec![match target {
                    Some(t) => Operand::imm(t as i64),
                    // Canonicalize through `reg()`: the decoder prints `blx r2`
                    // (not `x2`), and a raw `Register("r2")` is a name no def
                    // ever writes — SSA never renames it, so the indirect-call
                    // target stayed a phantom and every ARM32 virtual dispatch
                    // read as "target is not a value".
                    None => reg(rest),
                }];
                ops.push(Operand::reg("x0"));
                for r in ["x0", "x1", "x2", "x3"] {
                    ops.push(Operand::reg(r));
                }
                for r in fp_written.iter() {
                    ops.push(Operand::reg(r.clone()));
                }
                fp_written.clear();
                insns.push(Insn::new(Op::Call, ops, addr));
                pending_cmp = None;
            }
            "bx" => {
                if rest.contains("lr") || rest.contains("r14") {
                    // `bxne lr` returns only when the condition holds; same
                    // lowering as the conditional `pop {…, pc}` above.
                    emit_predication_guard(
                        &mut insns,
                        &mut addresses,
                        &mut scratch,
                        pending_cmp.as_ref(),
                        cond,
                        addr,
                    );
                    // NOTE: keep decoding past returns (see pop/pc above)
                    insns.push(Insn::new(Op::Return, vec![Operand::reg("x0")], addr));
                    // Same rule as the predicated `pop {…, pc}`: a `bxne lr`
                    // return does not execute on the fall-through path, and a
                    // return writes no APSR, so a predicated return keeps the
                    // flags live; only an unconditional one ends them.
                    if cond.is_empty() {
                        pending_cmp = None;
                    }
                } else {
                    // Canonicalize the target register the same way the `blx`
                    // arm does: `bx r3` prints `r3`, and a raw `Register("r3")`
                    // is a name no def ever writes, so the computed-jump target
                    // (a switch dispatch or an indirect tail call) was a
                    // phantom local with no definition.
                    insns.push(Insn::new(Op::IndirectCall, vec![reg(rest)], addr));
                    // A tail transfer does not return, so the flags do not
                    // survive it.
                    pending_cmp = None;
                }
            }
            // ---- compare forms that are not `cmp`/`tst` -------------------
            // `cmn rn, op2` is `adds` with no destination and `teq` is `eors`
            // with no destination. Both set APSR and are consumed by the very
            // next `b<cc>`, so they must arm the pending compare exactly like
            // `cmp` — falling through to the catch-all dropped the condition
            // and rendered the branch as `if (? != 0)`.
            "cmn" | "teq" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let lhs = reg(parts.get(1).copied().unwrap_or(""));
                let rhs_text = parts.get(2).copied().unwrap_or("0");
                let rhs = match parse_int(rhs_text) {
                    Some(v) => {
                        // `cmn` compares against the negation of its operand
                        Operand::imm(if mne == "cmn" { -v } else { v })
                    }
                    None => {
                        if mne == "cmn" {
                            let t = temp_local(&mut temp_counter);
                            insns.push(Insn::new(Op::Negate, vec![t.clone(), reg(rhs_text)], addr));
                            t
                        } else {
                            reg(rhs_text)
                        }
                    }
                };
                pending_cmp = Some(PendingCompare {
                    lhs,
                    rhs,
                    at: insns.len(),
                });
                insns.push(Insn::new(Op::Nop, vec![], addr));
            }
            // ---- negation ---------------------------------------------------
            "neg" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let src_text = parts.get(1).copied().unwrap_or("0");
                let src = match parse_int(src_text) {
                    Some(v) => Operand::imm(-v),
                    None => reg(src_text),
                };
                insns.push(Insn::new(Op::Negate, vec![dst.clone(), src], addr));
                if flags {
                    pending_cmp = Some(PendingCompare {
                        lhs: dst,
                        rhs: Operand::imm(0),
                        at: insns.len(),
                    });
                }
            }
            // ---- integer division -------------------------------------------
            "sdiv" | "udiv" => {
                // Both are exact 32-bit division; the signed/unsigned
                // distinction lives in the C# types, which the renderer
                // recovers from the operands, so one Div op is faithful here.
                let parts: Vec<&str> = rest.split(", ").collect();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Div),
                    vec![
                        reg(parts.first().copied().unwrap_or("")),
                        reg(parts.get(1).copied().unwrap_or("")),
                        reg(parts.get(2).copied().unwrap_or("")),
                    ],
                    addr,
                ));
            }
            // ---- sub-word extension -----------------------------------------
            // Exact, and extremely common: every `byte`/`short` field read and
            // every narrowing conversion goes through one of these. Dropping
            // them left their destination undefined, so the loaded field's
            // whole dependent chain became `__unknown`.
            "uxtb" | "uxth" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let mask: i64 = if mne == "uxtb" { 0xFF } else { 0xFFFF };
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![
                        dst.clone(),
                        reg(parts.get(1).copied().unwrap_or("")),
                        Operand::imm(mask),
                    ],
                    addr,
                ));
            }
            "sxtb" | "sxth" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let src = reg(parts.get(1).copied().unwrap_or(""));
                let k: i64 = if mne == "sxtb" { 24 } else { 16 };
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shl),
                    vec![dst.clone(), src, Operand::imm(k)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shr),
                    vec![dst.clone(), dst, Operand::imm(k)],
                    addr,
                ));
            }
            "ubfx" | "sbfx" => {
                // `bfx rd, rn, #lsb, #width` — extract a bit field.
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let src = reg(parts.get(1).copied().unwrap_or(""));
                let lsb = parts.get(2).and_then(|s| parse_int(s)).unwrap_or(0) as i64;
                let width = parts.get(3).and_then(|s| parse_int(s)).unwrap_or(32) as i64;
                let mut cur = src;
                if lsb > 0 {
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shr),
                        vec![dst.clone(), cur, Operand::imm(lsb)],
                        addr,
                    ));
                    cur = dst.clone();
                }
                if mne == "ubfx" {
                    if width < 32 {
                        let mask = (1i64 << width) - 1;
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::And),
                            vec![dst.clone(), cur, Operand::imm(mask)],
                            addr,
                        ));
                    }
                } else if width < 32 {
                    let k = 32 - width;
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![dst.clone(), cur, Operand::imm(k)],
                        addr,
                    ));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shr),
                        vec![dst.clone(), dst.clone(), Operand::imm(k)],
                        addr,
                    ));
                }
            }
            // ---- doubleword pair + multiple transfer ------------------------
            // `ldrd rd, rd2, [rn, #off]` / `strd`: two consecutive words, low
            // half first (little-endian). `ldm`/`stm` walk a register list from
            // a base address — modelled as the individual word accesses plus
            // the writeback the instruction performs.
            "ldrd" | "strd" => {
                let is_load = mne.starts_with("ldr");
                let mut parts = rest.splitn(3, ", ");
                let r0 = reg(parts.next().unwrap_or(""));
                let r1 = reg(parts.next().unwrap_or(""));
                let addr_text = parts.next().unwrap_or("[]");
                let inner = addr_text
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim()
                    .trim_end_matches('!');
                let mut segs = inner.split(',');
                let base = reg(segs.next().unwrap_or("sp").trim());
                let off = segs.next().and_then(|s| parse_int(s.trim())).unwrap_or(0);
                let mk = |delta: i64| Operand::Memory {
                    width: 0,
                    base: Box::new(base.clone()),
                    index: None,
                    scale: 4,
                    addend: off + delta,
                };
                if is_load {
                    insns.push(Insn::move_(r0, mk(0), addr));
                    insns.push(Insn::move_(r1, mk(4), addr));
                } else {
                    insns.push(Insn::move_(mk(0), r0, addr));
                    insns.push(Insn::move_(mk(4), r1, addr));
                }
            }
            "ldm" | "ldmia" | "ldmib" | "ldmda" | "ldmdb" | "stm" | "stmia" | "stmib" | "stmda"
            | "stmdb" | "push.w" | "pop.w" => {
                let is_load = mne.starts_with("ldm") || mne == "pop.w";
                let mut parts = rest.splitn(2, ", ");
                let base_text = parts.next().unwrap_or("sp");
                let base = reg(base_text.trim().trim_end_matches('!').trim());
                let list_raw = parts.next().unwrap_or("{}");
                let list: Vec<Operand> = list_raw
                    .trim()
                    .trim_start_matches('{')
                    .trim_end_matches('}')
                    .split(',')
                    .map(|s| reg(s.trim()))
                    .collect();
                for (i, r) in list.iter().enumerate() {
                    let m = Operand::Memory {
                        width: 0,
                        base: Box::new(base.clone()),
                        index: None,
                        scale: 4,
                        addend: (i as i64) * 4,
                    };
                    if is_load {
                        insns.push(Insn::move_(r.clone(), m, addr));
                    } else {
                        insns.push(Insn::move_(m, r.clone(), addr));
                    }
                }
                // Multiple-transfer writeback: `stmia r1!, {…}` / `ldmdb r2!, {…}`
                // advance the base by ±4 per register. The transfer ORDER
                // differs between the ia/db/ib/da modes, but the net base
                // update does not, and skipping it made every iteration after
                // the first read through a stale base.
                if base_text.trim().ends_with('!') {
                    let n = list.len() as i64 * 4;
                    // `db`/`da` are the DECREMENTING modes (`push` is `stmdb`),
                    // `ia`/`ib` the incrementing ones. The transfer ORDER differs
                    // between the four, but the net base update is ±4 per
                    // register in every one of them — which is all the frame
                    // model needs.
                    let down = mne.starts_with("stmdb")
                        || mne.starts_with("ldmdb")
                        || mne.starts_with("stmda")
                        || mne.starts_with("ldmda")
                        || mne.starts_with("push");
                    insns.push(Insn::new(
                        Op::Arith(if down { ArithKind::Sub } else { ArithKind::Add }),
                        vec![base.clone(), base.clone(), Operand::imm(n)],
                        addr,
                    ));
                }
            }
            // ---- barriers / hints: covered no-ops ---------------------------
            // These emit no value and touch no register, but they are NOT
            // unknown instructions: reporting them as "uncovered" inflated the
            // census and, worse, left the address bookkeeping to the catch-all.
            "nop" | "nop.w" | "dmb" | "dsb" | "isb" | "clrex" | "yield" | "sev" | "wfe" | "wfi"
            | "csdb" | "pld" | "pldw" | "pli" | "vmrs" | "vmsr" => {}
            // ---- Thumb-2 compare-and-branch --------------------------------
            // `cbz`/`cbnz` are the null/zero tests Thumb-2 uses for every
            // reference check, and `tbz`/`tbnz` test a single bit — the shape
            // of every `(flags & mask) != 0` test and every enum/state check.
            // They are self-contained: the comparison is IN the instruction,
            // so no pending APSR state is involved (and none is consumed).
            "cbz" | "cbnz" => {
                let mut parts = rest.splitn(2, ", ");
                let src = reg(parts.next().unwrap_or(""));
                let target = parse_target(parts.next().unwrap_or(""), addr).unwrap_or(usize::MAX);
                let s = scratch.fresh();
                let kind = if mne == "cbz" {
                    CompareKind::Equal
                } else {
                    CompareKind::NotEqual
                };
                insns.push(Insn::new(
                    Op::Compare(kind),
                    vec![s.clone(), src, Operand::imm(0)],
                    addr,
                ));
                insns.push(Insn::new(Op::CondJump(target), vec![s], addr));
                // `cbz`/`cbnz` test one register directly and do NOT write
                // APSR, so a compare that was pending before survives them (as
                // arm64's `preserves_nzcv` already models). Keeping it lets a
                // later branch read the flags the `cbz` left untouched.
            }
            "tbz" | "tbnz" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let src = reg(parts.first().copied().unwrap_or(""));
                let bit = parts.get(1).and_then(|s| parse_field(s)).unwrap_or(0);
                let target =
                    parse_target(parts.get(2).copied().unwrap_or(""), addr).unwrap_or(usize::MAX);
                let t = temp_local(&mut temp_counter);
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![t.clone(), src, Operand::imm(1i64 << bit)],
                    addr,
                ));
                let s = scratch.fresh();
                let kind = if mne == "tbz" {
                    CompareKind::Equal
                } else {
                    CompareKind::NotEqual
                };
                insns.push(Insn::new(
                    Op::Compare(kind),
                    vec![s.clone(), t, Operand::imm(0)],
                    addr,
                ));
                insns.push(Insn::new(Op::CondJump(target), vec![s], addr));
                // `tbz`/`tbnz` read a bit and do NOT write APSR — same rule as
                // `cbz` above: the pending compare survives the branch.
            }
            // ---- Thumb-2 IT block container ---------------------------------
            // `it`/`ite`/`itt`… carry NO semantics of their own: they only
            // mark which of the following instructions are predicated, and the
            // per-instruction condition suffix that yaxpeax prints is what
            // this lifter actually reads. Leaving them to the catch-all
            // emitted `__ite(...)` noise; they are a real, understood no-op.
            it_block if is_it_block(it_block) => {}
            // ---- PC-relative address materialization -------------------------
            // yaxpeax prints this as `adr rd, pc, 0x8` — three operands, with
            // `pc` spelled out. The value is an ADDRESS (`pc` reads as
            // `addr + 8` on A32), so it is materialized as a constant; treating
            // it as arithmetic on the undefined g15 register lost the address
            // and everything reached through it. This is how every
            // PC-relative literal, static and jump table is referenced.
            "adr" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                if parts.len() >= 3 && matches!(parts[1].trim(), "pc" | "r15") {
                    let off = parse_field(parts[2]).unwrap_or(0);
                    insns.push(Insn::move_(dst, Operand::imm(addr as i64 + 8 + off), addr));
                } else if let Some(t) = parse_target(parts.get(1).copied().unwrap_or(""), addr) {
                    insns.push(Insn::move_(dst, Operand::imm(t as i64), addr));
                }
            }
            // ---- bitfield insert / clear -------------------------------------
            // `bfi rd, rn, #lsb, #width` — how IL2CPP packs bit fields and
            // enum values. It reads rd, so dropping it (the catch-all defines
            // the destination from the operands, `__bfi`) lost the value of
            // the destination on the left-hand side of the merge.
            "bfi" | "bfc" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let (src, field_base) = if mne == "bfi" {
                    (Some(reg(parts.get(1).copied().unwrap_or(""))), 2usize)
                } else {
                    (None, 1usize)
                };
                let lsb = parts
                    .get(field_base)
                    .and_then(|s| parse_field(s))
                    .unwrap_or(0);
                let width = parts
                    .get(field_base + 1)
                    .and_then(|s| parse_field(s))
                    .unwrap_or(32)
                    .clamp(1, 32);
                let field_mask: i64 = if width >= 32 {
                    -1
                } else {
                    ((1i64 << width) - 1) << lsb
                };
                match src {
                    None => {
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::And),
                            vec![dst.clone(), dst, Operand::imm(!field_mask)],
                            addr,
                        ));
                    }
                    Some(src) => {
                        let t = temp_local(&mut temp_counter);
                        let mask = if width >= 32 { -1 } else { (1i64 << width) - 1 };
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::And),
                            vec![t.clone(), src, Operand::imm(mask)],
                            addr,
                        ));
                        if lsb > 0 {
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::Shl),
                                vec![t.clone(), t.clone(), Operand::imm(lsb)],
                                addr,
                            ));
                        }
                        let keep = temp_local(&mut temp_counter);
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::And),
                            vec![keep.clone(), dst.clone(), Operand::imm(!field_mask)],
                            addr,
                        ));
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Or),
                            vec![dst, keep, t],
                            addr,
                        ));
                    }
                }
            }
            // ---- saturating narrow (SSAT/USAT) -------------------------------
            // The exact clamp IL2CPP emits for a narrowing conversion with
            // clamping (`(sbyte)Math.Clamp(...)`, `Math.Min/Max` on small
            // types). Built from Compare + Select so the rendered C# is a real
            // min/max pair, not a black-box call.
            "ssat" | "usat" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let bits = parts
                    .get(1)
                    .and_then(|s| parse_field(s))
                    .unwrap_or(32)
                    .clamp(1, 32);
                let src_text = parts.get(2).copied().unwrap_or("");
                let shift = parts
                    .get(3)
                    .and_then(|s| parse_shift_amount(s))
                    .unwrap_or(0);
                let mut value = reg(src_text);
                if shift > 0 {
                    let t = temp_local(&mut temp_counter);
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![t.clone(), value, Operand::imm(shift)],
                        addr,
                    ));
                    value = t;
                }
                let (lo, hi) = if mne == "ssat" {
                    (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
                } else {
                    (0, (1i64 << bits) - 1)
                };
                // t = value > lo ? value : lo   (a real Math.Max)
                let s_lo = scratch.fresh();
                insns.push(Insn::new(
                    Op::Compare(CompareKind::GreaterThan),
                    vec![s_lo.clone(), value.clone(), Operand::imm(lo)],
                    addr,
                ));
                let clamped = temp_local(&mut temp_counter);
                insns.push(Insn::new(
                    Op::Select(super::ir::SelectVariant::Identity),
                    vec![clamped.clone(), s_lo, value, Operand::imm(lo)],
                    addr,
                ));
                // dst = t < hi ? t : hi       (a real Math.Min)
                let s_hi = scratch.fresh();
                insns.push(Insn::new(
                    Op::Compare(CompareKind::LessThan),
                    vec![s_hi.clone(), clamped.clone(), Operand::imm(hi)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Select(super::ir::SelectVariant::Identity),
                    vec![dst, s_hi, clamped, Operand::imm(hi)],
                    addr,
                ));
            }
            // ---- byte/halfword lane extension ---------------------------------
            // `sxtb16 rd, rn` / `uxtb16`: extend the low and high BYTES of the
            // 16-bit halves independently. Used by every UTF-8/UTF-16 and
            // packed-byte routine in the runtime; dropped, both destinations
            // went undefined.
            "sxtb16" | "uxtb16" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let src = reg(parts.get(1).copied().unwrap_or(""));
                let lane_mask: i64 = 0x00FF_00FF;
                let t = temp_local(&mut temp_counter);
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![t.clone(), src, Operand::imm(lane_mask)],
                    addr,
                ));
                if mne == "sxtb16" {
                    // per-lane sign extension: (x << 8) & 0x00FF00FF,
                    // then (x >> 8) arithmetic — our Shr is logical, so mask
                    // the sign bits in explicitly instead of relying on it.
                    let sign = temp_local(&mut temp_counter);
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![sign.clone(), t.clone(), Operand::imm(8)],
                        addr,
                    ));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::And),
                        vec![dst, sign, Operand::imm(0xFFFF_00FFu64 as i64)],
                        addr,
                    ));
                } else {
                    insns.push(Insn::move_(dst, t, addr));
                }
            }
            // ---- exclusive monitor (Interlocked) -----------------------------
            // `ldrex`/`strex` are the load/store halves of EVERY Interlocked
            // and volatile field update in managed code. Modelled as the
            // ordinary access they statically are — the retry loop that exists
            // for contention is a separate CFG back-edge — plus the status
            // word `strex` writes (0 = stored, the straight-line path). Both
            // previously fell through to `__ldrex`, so the address, the stored
            // value AND the status were all lost.
            "ldrex" | "ldrexb" | "ldrexh" | "ldrexd" => {
                let mut parts = rest.splitn(2, ", ");
                let dst = reg(parts.next().unwrap_or(""));
                let addr_text = parts.next().unwrap_or("");
                if let Some(acc) = parse_mem_access(addr_text) {
                    let (mem, delta) = mem_operand(&acc, &mut insns, &mut temp_counter, addr);
                    insns.push(Insn::move_(dst, mem, addr));
                    if let Some(delta) = delta {
                        push_writeback(&mut insns, &acc, delta, addr);
                    }
                }
            }
            "strex" | "strexb" | "strexh" | "strexd" => {
                let parts: Vec<&str> = rest.splitn(3, ", ").collect();
                let status = reg(parts.first().copied().unwrap_or(""));
                let value = reg(parts.get(1).copied().unwrap_or(""));
                let addr_text = parts.get(2).copied().unwrap_or("");
                if let Some(acc) = parse_mem_access(addr_text) {
                    let (mem, _) = mem_operand(&acc, &mut insns, &mut temp_counter, addr);
                    insns.push(Insn::move_(mem, value, addr));
                    insns.push(Insn::move_(status, Operand::imm(0), addr));
                }
            }
            // ---- long multiply / multiply-accumulate -------------------------
            // `smlal lo, hi, rn, rm` / `umlal`: lo += rn*rm. The low word is
            // exact (truncation is built into the 32-bit destination); the
            // high word is deliberately left undefined rather than wrong — the
            // same treatment `umull` already gets.
            //
            // The halfword `SMLAL<x><y>` forms never reach this arm: they are
            // decoded straight from the word by `dsp_mul` ahead of the
            // dispatch, because yaxpeax renders their selectors backwards.
            // What is left here is the bare word-by-word accumulate.
            "smlal" | "umlal" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let lo = reg(parts.first().copied().unwrap_or(""));
                let rn = reg(parts.get(2).copied().unwrap_or(""));
                let rm = reg(parts.get(3).copied().unwrap_or(""));
                let (a_op, b_op) = (rn, rm);
                insns.push(Insn::new(
                    Op::MulAcc {
                        negate_addend: false,
                        is_float: false,
                    },
                    vec![lo.clone(), a_op, b_op, lo],
                    addr,
                ));
            }
            // `umaal rdlo, rdhi, rn, rm` is the odd one out of the multiply
            // family: {RdHi,RdLo} = rn*rm + RdLo + RdHi. BOTH halves are
            // addends, which is what separates it from `umlal` (RdLo only).
            // Folding only one of them silently loses an addend.
            //
            // The low word is exact: every term lands in it modulo 2^32, so
            // `MulAcc` (RdLo += rn*rm) followed by `RdLo += RdHi` computes it
            // with no truncation fiction. RdHi must be read BEFORE it is
            // clobbered, and it is not touched until the final add, so the
            // order is safe. The high word is left undefined, as with `umlal`.
            "umaal" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let lo = reg(parts.first().copied().unwrap_or(""));
                let hi = reg(parts.get(1).copied().unwrap_or(""));
                let rn = reg(parts.get(2).copied().unwrap_or(""));
                let rm = reg(parts.get(3).copied().unwrap_or(""));
                insns.push(Insn::new(
                    Op::MulAcc {
                        negate_addend: false,
                        is_float: false,
                    },
                    vec![lo.clone(), rn, rm, lo.clone()],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Add),
                    vec![lo.clone(), lo, hi],
                    addr,
                ));
            }
            // `mls rd, rn, rm, ra` == rd = ra - rn*rm. Expressed as
            // rn * (-rm) + ra so the fused multiply-add stays one real
            // operation instead of a subtraction of an undefined product.
            "mls" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let neg = temp_local(&mut temp_counter);
                insns.push(Insn::new(
                    Op::Negate,
                    vec![neg.clone(), reg(parts.get(2).copied().unwrap_or(""))],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::MulAcc {
                        negate_addend: false,
                        is_float: false,
                    },
                    vec![
                        reg(parts.first().copied().unwrap_or("")),
                        reg(parts.get(1).copied().unwrap_or("")),
                        neg,
                        reg(parts.get(3).copied().unwrap_or("")),
                    ],
                    addr,
                ));
            }
            // ---- atomic swap ------------------------------------------------
            // `swp rd, rm, [rn]` / `swpb`: read [rn] into rd AND write rm back.
            // Modelled as the load half followed by the store half. The
            // atomicity itself is unobservable at this level (nothing in the
            // lifted program races the same word), but the DATA MOVEMENT is
            // real and the old path dropped BOTH halves — the load lost a value
            // and the store lost its effect.
            "swp" | "swpb" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let rd = reg(parts.first().copied().unwrap_or(""));
                let rm = reg(parts.get(1).copied().unwrap_or(""));
                let addr_text = parts.get(2).copied().unwrap_or("[]");
                let inner = addr_text
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim_end_matches('!')
                    .trim();
                let mut segs = inner.split(',');
                let base = reg(segs.next().unwrap_or("sp").trim());
                let off = segs.next().and_then(|s| parse_int(s.trim())).unwrap_or(0);
                let width = if mne == "swpb" { 1u8 } else { 0 };
                let m = Operand::Memory {
                    width,
                    base: Box::new(base),
                    index: None,
                    scale: 4,
                    addend: off,
                };
                insns.push(Insn::move_(rd, m.clone(), addr));
                insns.push(Insn::move_(m, rm, addr));
            }
            // ---- status register access -------------------------------------
            // `mrs rd, <spec>` reads a status register into rd. The VALUE is not
            // modelled (APSR/CPSR are not part of the recovered program's
            // state), but the DESTINATION must still be defined — otherwise
            // every later read of rd fabricates a different expression. It is
            // carried as a named intrinsic, exactly like the vector opcodes.
            //
            // `msr <spec>, rn` WRITES the flags, so any pending compare is
            // invalidated rather than silently carried across the write.
            "mrs" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                insns.push(Insn::new(Op::Simd("__mrs".into()), vec![dst], addr));
                pending_cmp = None;
            }
            "msr" => {
                pending_cmp = None;
            }
            // ---- DSP saturation --------------------------------------------
            // QADD/QSUB/QDADD/QDSUB saturate on signed overflow. Modelling them
            // as plain add/sub would be wrong EXACTLY when saturation fires, so
            // they stay named intrinsics — but with the destination defined and
            // every source preserved, which the fallback only managed when the
            // first operand happened to look like a plain register.
            "qadd" | "qsub" | "qdadd" | "qdsub" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst = reg(parts.first().copied().unwrap_or(""));
                let mut ops = vec![dst];
                for p in parts.iter().skip(1) {
                    let p = p.trim();
                    if looks_like_reg(p) {
                        ops.push(reg(p));
                    } else if let Some(v) = parse_int(p) {
                        ops.push(Operand::imm(v));
                    }
                }
                insns.push(Insn::new(Op::Simd(format!("__{mne}")), ops, addr));
            }
            // ---- coprocessor / secure monitor -------------------------------
            // MRC2 moves a coprocessor register into an ARM register. The
            // destination is the THIRD token (`mrc2 p15, 0, r0, c7, c10, 5`),
            // which the fallback never looked at — its first token is `p15`, so
            // the transferred value was lost. CDP2 and SMC produce no value and
            // are correctly no-ops; naming them here stops them reaching the
            // census as if they were unmodelled computation.
            "mrc2" => {
                let parts: Vec<&str> = rest.split(", ").collect();
                if let Some(d) = parts.iter().find(|p| looks_like_reg(p.trim())) {
                    insns.push(Insn::new(
                        Op::Simd("__mrc2".into()),
                        vec![reg(d.trim())],
                        addr,
                    ));
                }
            }
            // `mcr2` transfers an ARM register TO a coprocessor. It writes no
            // ARM architectural register, and the value it ships is only
            // observable inside a coprocessor the recovered program cannot
            // see, so a no-op is the honest model — the same call `cdp2` and
            // `smc` already get. It must be named here rather than left to the
            // fallback, which would otherwise report it as unmodelled
            // computation in the coverage census.
            "mcr2" | "cdp2" | "smc" | "invalid" => {}

            // ---- generic fallback ------------------------------------------
            //
            // An instruction that writes a register MUST define that register.
            // The old `_ => {}` dropped the instruction and, whenever it had a
            // register destination, left that register undefined — so ONE
            // unmodelled mnemonic turned its entire dependent expression into
            // `__unknown` and showed up downstream as an undefined local.
            //
            // The fallback instead defines the destination from the real
            // operands and keeps every source, carrying the operation's own
            // mnemonic (`__smmul(a, b)`), so the value chain stays connected
            // while the semantics stay explicitly unmodelled. It is the same
            // trade `fnptr_0x…` makes for an unresolved call: a name we do not
            // have yet must never cost the VALUE — and it makes every remaining
            // lifter gap visible in the dump itself instead of showing up as an
            // unexplained `__unknown` somewhere else.
            other => {
                // `ILR_A32GAPS=1` reports every instruction that reached this
                // catch-all with the EXACT text the decoder produced. That is
                // how an unmodelled mnemonic is identified from a real binary
                // instead of from a guess about what the decoder prints.
                if std::env::var("ILR_A32GAPS").is_ok() {
                    eprintln!("A32GAP {mne}|{other}|{rest}");
                }
                let parts: Vec<&str> = rest.split(", ").collect();
                let dst_text = parts.first().copied().unwrap_or("").trim();
                if looks_like_reg(dst_text) {
                    let mut ops = vec![reg(dst_text)];
                    for p in parts.iter().skip(1) {
                        let p = p.trim();
                        match parse_int(p) {
                            Some(v) => ops.push(Operand::imm(v)),
                            None if looks_like_reg(p) => ops.push(reg(p)),
                            None => {}
                        }
                    }
                    insns.push(Insn::new(Op::Simd(format!("__{other}")), ops, addr));
                }
            }
        }

        // Address/IR parallelism invariant (see lift.rs): one address entry
        // per EMITTED IR insn; skipped native insns get none, so their RVA
        // is unmapped and the CFG treats branches to them as exits.
        while addresses.len() < insns.len() {
            addresses.push(addr);
        }
        offset += insn.len().to_const() as u64;
    }

    if !insns.last().is_some_and(|i| i.is_terminator()) {
        let tail = base_rva + offset;
        insns.push(Insn::new(Op::Return, vec![Operand::reg("x0")], tail));
        addresses.push(tail);
    }

    Some(LiftResult { insns, addresses })
}

/// Bitwise-not into a temp (for `bic` with a register operand).
fn push_not_temp(insns: &mut Vec<Insn>, temp: &mut u32, src: Operand, addr: u64) -> Operand {
    let t = temp_local(temp);
    push_not(insns, t.clone(), src, addr);
    t
}

fn parse_target(raw: &str, addr: u64) -> Option<usize> {
    let raw = raw.trim_start_matches('#').trim();
    if let Some(hex) = raw.strip_prefix("0x") {
        return usize::from_str_radix(hex, 16).ok();
    }
    // yaxpeax prints branches as `$+0x18` / `$-0x8` (offset relative to the
    // instruction address). Without this every ARM32 branch resolved to
    // usize::MAX and the CFG treated all edges as exits → empty bodies.
    if let Some(signed) = raw.strip_prefix('$') {
        let (neg, rest) = match signed.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, signed.strip_prefix('+').unwrap_or(signed)),
        };
        let v = if let Some(hex) = rest.strip_prefix("0x") {
            i64::from_str_radix(hex, 16).ok()
        } else {
            rest.parse::<i64>().ok()
        }?;
        let off = if neg { -v } else { v };
        return (addr as i64 + off).try_into().ok();
    }
    raw.parse::<i64>()
        .ok()
        .map(|off| (addr as i64 + off) as usize)
        .or_else(|| raw.parse::<usize>().ok())
}
