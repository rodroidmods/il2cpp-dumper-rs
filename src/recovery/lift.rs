//! ARM64 lifter: native A64 instructions → recovery IR.
//!
//! Design notes:
//! - registers canonicalized to 64-bit names (`w3` → `x3`);
//!   `Register(_, 31)` is ZERO (xzr), `RegisterOrSP(_, 31)` is SP
//! - flag-producing ops (SUBS/ADDS/ANDS into xzr = compiler's cmp/cmn/tst)
//!   fold into a single Compare consumed by the next B.cond in-block
//! - scalar FP registers canonicalized to "vN" slots; FMOV-immediate and
//!   literal-pool loads materialize Float/Double literals directly
//!   (`pool_reader` callback supplies binary bytes at RVA)
//! - unresolved patterns degrade to Operand::Unknown (emits diagnostics)

use super::ir::{ArithKind, CompareKind, FloatArithKind, Insn, Op, Operand, SelectVariant};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use yaxpeax_arch::{Decoder, LengthedInstruction, U8Reader};
use yaxpeax_arm::armv8::a64::{
    InstDecoder as A64Decoder, Instruction as A64Insn, Opcode as A64Opcode, Operand as A64Op,
    SIMDSizeCode, ShiftStyle,
};

pub struct LiftResult {
    pub insns: Vec<Insn>,
    pub addresses: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Lifter coverage census
// ---------------------------------------------------------------------------
//
// `lift_arm64` ends in a `_ =>` catch-all. Before this census existed that arm
// was SILENT: any AArch64 opcode without an explicit arm became `dst = Unknown`
// (or, when it had no register destination, vanished entirely) with no count,
// no opcode name and no address recorded anywhere. That is why "which
// instructions does this lifter not understand?" was unanswerable from the
// dump — every one of them instead surfaced much later as an unexplained
// `__unknown` buried inside a body, looking like a resolver or type-propagation
// bug rather than a missing lifter arm.
//
// The census turns that fall-through into an enumerable worklist: opcode name,
// how often, how often it cost a real value, and sample addresses to decode.
// It is what the lifter arms get written FROM — data, not guesses about which
// opcodes an optimizing C++ compiler emits.

/// One row of the unhandled-opcode census.
#[derive(Clone, Default)]
pub struct CensusRow {
    /// Times this opcode fell through to the `_ =>` arm.
    pub count: u64,
    /// Times it fell through WITH a register destination, i.e. a value that is
    /// now a bare `Unknown` in the generated C#. These are the rows that cost
    /// fidelity; rows with `effectful == 0` are correctly ignorable
    /// (NOP/DSB/ISB/MSR/MRS/PRFM) and are kept only for completeness.
    pub effectful: u64,
    /// Up to 8 `(rva, raw instruction word)` samples, enough to see which
    /// encoding variant is in use when writing the arm.
    pub samples: Vec<(u64, u32)>,
}

/// Census state, keyed by opcode name.
static LIFT_CENSUS: LazyLock<Mutex<BTreeMap<String, CensusRow>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Census collection is opt-in: it allocates a name per fall-through, which is
/// not something a normal dump should pay for.
fn census_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ILR_LIFT_CENSUS").is_ok() || std::env::var("ILR_LIFT_CENSUS_FILE").is_ok()
    })
}

fn census_unhandled(op: A64Opcode, addr: u64, word: u32, effectful: bool) {
    if !census_enabled() {
        return;
    }
    let Ok(mut m) = LIFT_CENSUS.lock() else {
        return;
    };
    let row = m.entry(format!("{op:?}")).or_default();
    row.count += 1;
    if effectful {
        row.effectful += 1;
    }
    if row.samples.len() < 8 {
        row.samples.push((addr, word));
    }
}

/// Opcodes the generic value-producing arm below turned into a named
/// `__<mnemonic>` intrinsic.
///
/// These are NOT coverage gaps any more — the destination is defined and every
/// source is preserved, so no value is lost — but they ARE the worklist for
/// writing real semantics, which is why they are counted separately from the
/// fall-through census instead of vanishing into it.
static INTRINSIC_CENSUS: LazyLock<Mutex<BTreeMap<String, u64>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn census_intrinsic(op: A64Opcode) {
    if !census_enabled() {
        return;
    }
    let Ok(mut m) = INTRINSIC_CENSUS.lock() else {
        return;
    };
    *m.entry(format!("{op:?}")).or_default() += 1;
}

/// `(distinct opcodes, occurrences)` lowered to a named intrinsic.
pub fn lift_intrinsic_totals() -> (usize, u64) {
    let Ok(m) = INTRINSIC_CENSUS.lock() else {
        return (0, 0);
    };
    (m.len(), m.values().sum())
}

/// The intrinsic worklist as a TSV section, appended to the census report.
fn intrinsic_report_section() -> String {
    let Ok(m) = INTRINSIC_CENSUS.lock() else {
        return String::new();
    };
    let mut rows: Vec<(&String, &u64)> = m.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let mut out = String::new();
    out.push_str("\n# Named-intrinsic worklist — opcodes lowered to `__<mnemonic>(…)`.\n");
    out.push_str("#\n# These define their destination and keep every operand, so no value is\n");
    out.push_str("# lost; they are listed because their SEMANTICS are not modelled yet.\n");
    out.push_str("# Writing a real arm for a row removes it from here.\n#\n#\tcount\topcode\n");
    for (name, n) in rows {
        out.push_str(&format!("{n}\t{name}\n"));
    }
    out
}

/// Totals across every opcode: `(distinct opcodes, fall-throughs, effectful)`.
pub fn lift_census_totals() -> (usize, u64, u64) {
    let Ok(m) = LIFT_CENSUS.lock() else {
        return (0, 0, 0);
    };
    (
        m.len(),
        m.values().map(|r| r.count).sum(),
        m.values().map(|r| r.effectful).sum(),
    )
}

/// One-line summary for stderr / the dump log.
pub fn lift_census_summary() -> String {
    let (n, total, eff) = lift_census_totals();
    let (in_, ic) = lift_intrinsic_totals();
    format!(
        "[lift] unhandled opcodes: {n} distinct, {total} fall-throughs, {eff} of them lost a value; \
         named intrinsics: {in_} distinct, {ic} instructions"
    )
}

/// Full census as a self-describing TSV, sorted by how much fidelity each
/// opcode costs (descending `effectful`, then `count`).
///
/// `ILR_LIFT_CENSUS_FILE=<path>` writes this; `ILR_LIFT_CENSUS=1` prints the
/// summary to stderr without a file.
pub fn lift_census_report() -> String {
    let Ok(m) = LIFT_CENSUS.lock() else {
        return String::new();
    };
    let mut rows: Vec<(&String, &CensusRow)> = m.iter().collect();
    rows.sort_by(|a, b| {
        b.1.effectful
            .cmp(&a.1.effectful)
            .then(b.1.count.cmp(&a.1.count))
            .then(a.0.cmp(b.0))
    });
    let (n, total, eff) = (
        rows.len(),
        rows.iter().map(|(_, r)| r.count).sum::<u64>(),
        rows.iter().map(|(_, r)| r.effectful).sum::<u64>(),
    );
    let mut out = String::with_capacity(rows.len() * 96 + 512);
    out.push_str("# Lifter coverage census — AArch64 opcodes `lift_arm64` does not implement.\n");
    out.push_str("#\n");
    out.push_str(
        "# Each row fell through the `_ =>` catch-all. `effectful` counts the times the\n",
    );
    out.push_str("# instruction had a register destination, i.e. the generated C# shows a bare\n");
    out.push_str("# `Unknown` where a real value belongs. Rows with effectful=0 (NOP/DSB/ISB/\n");
    out.push_str("# MSR/MRS/PRFM) are correctly ignorable and are listed only so the list is\n");
    out.push_str("# complete. Writing an arm for a row REMOVES it from here, so this file is\n");
    out.push_str("# also the progress measure for the lifter port.\n");
    out.push_str("#\n");
    out.push_str(&format!(
        "# totals: {n} distinct opcodes, {total} fall-throughs, {eff} lost a value\n"
    ));
    out.push_str("#\n# effectful\tcount\topcode\tsamples(rva:word)\n");
    for (name, r) in rows {
        let samples: Vec<String> = r
            .samples
            .iter()
            .map(|(a, w)| format!("{a:#x}:{w:08x}"))
            .collect();
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            r.effectful,
            r.count,
            name,
            samples.join(" ")
        ));
    }
    out.push_str(&intrinsic_report_section());
    out
}

/// Binary accessor for literal-pool reads (f32/f64 materialization).
pub type PoolReader<'a> = dyn Fn(u64) -> Option<Vec<u8>> + 'a;

fn gpr(num: u16, sp: bool) -> String {
    if sp && num == 31 {
        "sp".into()
    } else if !sp && num == 31 {
        "xzr".into()
    } else {
        format!("x{num}")
    }
}

/// Scalar FP/SIMD register name: sN/dN/qN unified per width; we keep dN for
/// 64-bit and sN for 32-bit views distinct (aliasing handled by typing).
fn fp_reg(size: SIMDSizeCode, num: u16) -> String {
    match size {
        SIMDSizeCode::B => format!("vb{num}"),
        SIMDSizeCode::H => format!("vh{num}"),
        // Keep scalar S-register values in the same namespace as vector S
        // lanes: `sN` is lane 0 of `vN`, while `vN.s[k]` is lane k. The
        // flattened index is `4*N + k` (see `s_lane_name`). Using `vsN` here
        // aliased unrelated registers, e.g. `s8` with `v2.s[0]`.
        SIMDSizeCode::S => scalar_s_name(num),
        SIMDSizeCode::D => format!("vd{num}"),
        SIMDSizeCode::Q => format!("vq{num}"),
    }
}

/// Byte width of an FP register NAME produced by [`fp_reg`].
///
/// `None` for a GPR name. Needed because the STP/LDP pair arms pick the second
/// element's address by adding the element size to the first's, and the only
/// thing they have to go on is the register name.
fn fp_reg_bytes(name: &str) -> Option<u8> {
    let rest = name.strip_prefix('v')?;
    match rest.as_bytes().first()? {
        b'd' => Some(8),
        // A `q` view is the FULL 128-bit register: 16 bytes. Sizing it at 8 (it
        // used to share the `d` arm) placed the second element of
        // `stp q0, q1, [sp, #32]` at `[sp+40]` — inside the first register —
        // aliasing the two halves of the pair.
        b'q' => Some(16),
        b'b' => Some(1),
        b'h' => Some(2),
        // `s` — and anything else, kept at the widest thing the 32-bit view can
        // be, so an unforeseen name can never over-advance into a neighbour.
        _ => Some(4),
    }
}

/// ARM64 condition-code (Bcc encoding, `cond&0xF`) → comparison kind.
/// Shared with the verifier's semantic diff so both sides apply ONE table.
pub(crate) fn cond_to_kind(code: u8) -> CompareKind {
    match code {
        0 => CompareKind::Equal,
        1 => CompareKind::NotEqual,
        2 | 10 => CompareKind::GreaterOrEqual,
        3 | 11 => CompareKind::LessThan,
        4 => CompareKind::LessThan,
        5 => CompareKind::GreaterOrEqual,
        8 => CompareKind::GreaterThan,
        9 => CompareKind::LessOrEqual,
        12 => CompareKind::GreaterThan,
        13 => CompareKind::LessOrEqual,
        _ => CompareKind::NotEqual,
    }
}

/// Symbolic NZCV produced by the last flag-setting instruction, consumed by
/// the next `Bcc`.
///
/// `Cmp` is the plain case (`cmp`/`cmn`/`tst`/`fcmp`). `Merge` models
/// `cmp …; ccmp …` — ARM64's conditional compare, which selects between a
/// second comparison and a CONSTANT flag nibble:
///
/// ```text
/// result_flags = P ? cmp(lhs, rhs) : nzcv      where P = eval(ccmp_cond, prev_flags)
/// ```
///
/// Compilers emit this shape for compound conditions (`if (a == null ||
/// b == null) throw …`), and dropping it left every such branch with
/// `Compare(kind, Unknown, 0)` — the mass "conditional branch without real
/// compare" wrong-logic class.
///
/// `at` is the index in the lifted listing at which these operands were
/// CAPTURED, and it is what keeps a deferred read honest. The operands are
/// register NAMES, but a name is not a value: if anything between the compare
/// and the branch that consumes it redefines that register, the name resolves
/// to the NEW value by the time the `Compare` is emitted, which sits at the
/// BRANCH's position in the listing. The compare then compares a value with
/// itself and the branch reads as unconditionally true.
///
/// Real bytes, `0x4fba14`:
///
/// ```text
/// 0x4fba10  bl     …
/// 0x4fba14  cmp    x0, x21      ; the call result against the OLD x21
/// 0x4fba18  mov    x21, x0      ; x21 := the call result
/// 0x4fba1c  b.eq   +0x3c        ; taken iff the result equals the old x21
/// ```
///
/// Emitted as `Compare(Equal)[s, Register("x0"), Register("x21")]` at
/// `0x4fba1c` — where `x21` is `x0`, so the condition is `x0 == x0`. Every
/// `x == x`-style nonsense condition in the dump (`(this & 1) == 0`,
/// `x.GetType() == x.GetType()`) is this one mechanism.
///
/// `ir::Insn::dest_position` is the same "is operand 0 a def" test the SSA
/// renamer uses, and the branch's condition is resolved through it twice
/// (once by renderer `branch_condition`, once post-fix), so pinning the
/// MATERIALIZED read is enough — no SSA change is needed.
enum FlagState {
    /// Flags from a plain compare.
    Cmp {
        lhs: Operand,
        rhs: Operand,
        /// Listing index where these operands were captured.
        at: usize,
    },
    /// Flags from `cmp; ccmp` (or `cmn; ccmn`).
    Merge {
        /// Previous compare, evaluated with `ccmp_cond` to get `P`.
        prev_lhs: Operand,
        prev_rhs: Operand,
        /// Listing index where the PREVIOUS compare's operands were captured.
        prev_at: usize,
        ccmp_cond: u8,
        /// The conditional compare's own operands.
        lhs: Operand,
        rhs: Operand,
        /// Listing index where THIS compare's operands were captured.
        at: usize,
        /// The `#nzcv` immediate taken when `P` is false.
        nzcv: u8,
    },
}

/// Force a DEFERRED read of `op` to see the value it had when the flags were
/// set, not whatever that register holds by the time the compare is built.
///
/// A compare is materialized at its CONSUMER's position in the listing (the
/// branch, or the select), because the consumer's condition code is what
/// decides the comparison kind. Its operands, though, are register names read
/// out of the flag-setting instruction — and SSA renames a register use by the
/// listing position it appears at. Any redefinition between the two therefore
/// silently rewrites the compare's operand to the LATER value. When the
/// redefinition happens to be a copy of the other operand, the compare becomes
/// `x == x`, which renders as a plausible-looking but ALWAYS-TRUE condition.
///
/// The fix is to pin the read at the redefinition: a copy of the register is
/// inserted immediately BEFORE the instruction that would clobber it, and the
/// compare is rewritten to read that copy. Because the copy precedes the
/// redefinition in the listing, SSA binds it to the old value — which is
/// exactly what the hardware read.
///
/// `from` is the listing index at which the operands were captured, so only
/// redefinitions at or after the compare are considered; a definition before it
/// is already the value the compare reads.
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
    // jump "to the branch" resolves through this array by RVA).
    while addresses.len() < insns.len() {
        addresses.push(insns[addresses.len()].address);
    }
    let at_addr = insns[j].address;
    insns.insert(j, Insn::move_(pin.clone(), Operand::reg(name), at_addr));
    addresses.insert(j, at_addr);
    *op = pin;
}

/// ARM64 `ConditionHolds(cond, NZCV)` — does condition code `cond` hold for the
/// flag nibble `nzcv` (bit3=N, bit2=Z, bit1=C, bit0=V)?
pub(crate) fn eval_cond_code(code: u8, nzcv: u8) -> bool {
    let n = nzcv & 8 != 0;
    let z = nzcv & 4 != 0;
    let c = nzcv & 2 != 0;
    let v = nzcv & 1 != 0;
    match code & 0xF {
        0x0 => z,            // eq
        0x1 => !z,           // ne
        0x2 => c,            // cs / hs
        0x3 => !c,           // cc / lo
        0x4 => n,            // mi
        0x5 => !n,           // pl
        0x6 => v,            // vs
        0x7 => !v,           // vc
        0x8 => c && !z,      // hi
        0x9 => !c || z,      // ls
        0xA => n == v,       // ge
        0xB => n != v,       // lt
        0xC => !z && n == v, // gt
        0xD => z || n != v,  // le
        _ => true,           // al / nv
    }
}

#[derive(Default)]
struct ConstTracker {
    values: HashMap<String, i64>,
}

impl ConstTracker {
    fn apply(&mut self, op: &A64Opcode, dst: &str, imm: i64, shift: u8) -> Option<i64> {
        match op {
            A64Opcode::MOVZ => {
                let v = imm << shift;
                self.values.insert(dst.to_string(), v);
                Some(v)
            }
            A64Opcode::MOVN => {
                let v = !(imm << shift);
                self.values.insert(dst.to_string(), v);
                Some(v)
            }
            A64Opcode::MOVK => {
                let base = self.values.get_mut(dst)?;
                let keep_mask = !((0xFFFFi64) << shift);
                *base = (*base & keep_mask) | (imm << shift);
                Some(*base)
            }
            _ => None,
        }
    }

    fn kill(&mut self, reg: &str) {
        self.values.remove(reg);
    }

    fn get(&self, reg: &str) -> Option<i64> {
        self.values.get(reg).copied()
    }
}

/// Index of the `x8` slot on an `Op::Call` operand list
/// `[target, dest, x0 … x7, x8]`.
const SRET_SLOT: usize = 10;

/// RECORD THE AAPCS64 INDIRECT-RESULT POINTER (`x8`) AS A CALL ARGUMENT.
///
/// A function returning a struct does not return it in x0: the CALLER passes
/// the address of the destination buffer in x8 and the callee writes through
/// it. The `BL` arm builds its operand list from x0–x7 only, so that pointer
/// was never an operand of the call — the argument vanished and, with it, the
/// only visible evidence of where a struct result lands (`GetTimeSlotStatus`
/// @0x470204 is called with `&result` in x8 and the method reloads it from
/// `[sp,#0xc]` afterwards; the dump showed a 2-argument call, the native has
/// three).
///
/// Recorded ONLY for the sret idiom — an `x8 ← &[sp + imm]` materialization in
/// the same straight-line run as the call — so an x8 left over from earlier
/// code can never become a phantom third argument.
fn attach_sret_register(insns: &mut [Insn]) -> usize {
    let mut n = 0;
    for i in 0..insns.len() {
        // The `BL` arm's list is `[target, dest, x0 … x7]` — SRET_SLOT operands,
        // and inserting AT SRET_SLOT appends the x8 slot.
        if !matches!(insns[i].op, Op::Call) || insns[i].operands.len() < SRET_SLOT {
            continue;
        }
        let mut j = i;
        let mut ok = false;
        while j > 0 {
            j -= 1;
            let prev = &insns[j];
            if prev.is_terminator() || matches!(prev.op, Op::Call | Op::Jump(_) | Op::CondJump(_)) {
                break;
            }
            if let Some(Operand::Register(r)) = prev.dest() {
                if r == "x8" {
                    ok = matches!(prev.operands.get(1), Some(Operand::AddressOf(_)));
                    break;
                }
            }
        }
        if ok {
            insns[i].operands.insert(SRET_SLOT, Operand::reg("x8"));
            n += 1;
        }
    }
    n
}

pub fn lift_arm64(
    bytes: &[u8],
    base_rva: u64,
    max_insns: usize,
    pool_reader: Option<&PoolReader<'_>>,
) -> Option<LiftResult> {
    let decoder = A64Decoder::default();
    let mut reader = U8Reader::new(bytes);
    let mut insns: Vec<Insn> = Vec::new();
    let mut addresses: Vec<u64> = Vec::new();
    let mut offset: u64 = 0;
    let mut pending_cmp: Option<FlagState> = None;
    let mut consts = ConstTracker::default();

    let cap = max_insns.max(16_384);

    // FP registers written since the last call site (ascending register
    // number at use time). AAPCS64 FP-arg tracking for BL — see the call
    // arm below for why this exists.
    let mut fp_written: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    // Serial for synthetic shift-expansion temps (see shifted_reg_src).
    let mut virt_serial: u32 = 0;

    // Scratch local ids for branch tests (Compare/And results that only feed
    // a CondJump). Each def gets a UNIQUE id counting down from `u32::MAX - 2`:
    // the old shared sentinels made every compare multi-def, so renderers had
    // to GUESS which def feeds a branch — and the guess regularly bound the
    // wrong expression to a condition (the `if (x != 0) { if (x == 0) {`
    // contradictory-guard corruption class). Unique ids make single-def
    // lookups exact end-to-end.
    let mut next_scratch: u32 = u32::MAX - 2;
    macro_rules! fresh_scratch {
        () => {{
            let v = next_scratch;
            next_scratch = v.wrapping_sub(1);
            Operand::Local(v)
        }};
    }

    // Lift the WHOLE body. The body length is exact (metadata-provided) and
    // A64 is fixed-width, so decoding past the first RET is safe — and
    // required: methods with early returns (guard clauses) previously lost
    // every instruction after the first return. Unreachable trailing
    // fragments (literal pools etc.) form unreachable blocks that the CFG
    // consumers never visit.
    while offset as usize + 4 <= bytes.len() && insns.len() < cap {
        // A word we cannot decode must NOT delete the method. `ok()?` here
        // returned `None` from `lift_arm64`, and the caller turns that into
        // `fail!("lift failed")` — i.e. the ENTIRE body became a stub. One
        // VFP/NEON/SVE word (or a literal-pool word the body/size estimate
        // overlaps) was enough to erase a method that is otherwise perfectly
        // liftable, which is a whole-method failure produced by a single-word
        // gap. yaxpeax's A64 `decode_into` does `words.next_n(&mut word_bytes)?`
        // BEFORE any decoding decision (its source, `armv8/a64.rs`), so the
        // reader has already consumed exactly this 4-byte instruction when it
        // errors — the same convention the ARM32 lifter documents. Emit no IR
        // for it, step one word, and keep lifting. The word stays visible to
        // the oracle, which walks the bytes itself and counts it as uncovered
        // rather than as covered-but-wrong.
        // A word we cannot decode must NOT delete the method. `ok()?` here
        // returned `None` from `lift_arm64`, and the caller turns that into
        // `fail!("lift failed")` — i.e. the ENTIRE body became a stub. One
        // VFP/NEON/SVE word (or a literal-pool word the body/size estimate
        // overlaps) was enough to erase a method that is otherwise perfectly
        // liftable, which is a whole-method failure produced by a single-word
        // gap. yaxpeax's A64 `decode_into` does `words.next_n(&mut word_bytes)?`
        // BEFORE any decoding decision (its source, `armv8/a64.rs`), so the
        // reader has already consumed exactly this 4-byte instruction when it
        // errors — the same convention the ARM32 lifter documents. Emit no IR
        // for it, step one word, and keep lifting. The word stays visible to
        // the oracle, which walks the bytes itself and counts it as uncovered
        // rather than as covered-but-wrong.
        let insn: A64Insn = match decoder.decode(&mut reader) {
            Ok(i) => i,
            Err(_) => {
                offset += 4;
                continue;
            }
        };
        let addr = base_rva + offset;

        // Macro shared by the arithmetic/logic arms: resolve source operand
        // `idx`, which may be a shifted register — emit a synthetic
        // `reg <op> amt` into a fresh temp when needed (the plain
        // `operand_at` path drops shifted sources entirely).
        macro_rules! resolve_src {
            ($idx:expr) => {{
                let plain = operand_at(&insn, $idx);
                match shifted_reg_src(&insn, $idx) {
                    None => plain,
                    Some((reg, _kind, 0)) => Some(Operand::reg(reg)),
                    Some((reg, kind, amt)) => {
                        let t = Operand::reg(format!("xt{}", virt_serial));
                        virt_serial += 1;
                        insns.push(Insn::new(
                            Op::Arith(kind),
                            vec![t.clone(), Operand::reg(reg), Operand::imm(amt as i64)],
                            addr,
                        ));
                        Some(t)
                    }
                }
            }};
        }

        // Address/IR parallelism invariant: `addresses` must stay parallel to
        // `insns` (one entry per EMITTED IR insn, none for decoded-but-skipped
        // native insns). Cfg::build maps branch-target RVAs through this array
        // to derive insn indices — any drift misroutes every later branch.
        // Multi-insn groups (Compare+CondJump, LDP pairs, TBZ triples) share
        // the same address; a jump "to the branch" lands on the first insn of
        // the group, which is exactly the right semantic target.
        // Declared before `end_iteration!` so the macro can see it: a
        // macro_rules body resolves outer locals at its DEFINITION site, so a
        // binding introduced later in the loop body is not in scope there.
        let carried_cmp: Option<FlagState>;

        macro_rules! end_iteration {
            () => {{
                while addresses.len() < insns.len() {
                    addresses.push(addr);
                }
                offset += insn.len().to_const() as u64;
                // const-tracking kill for non-mov-family writers
                if let Some(d) = insn.operands.first().and_then(gpr_of) {
                    match insn.opcode {
                        A64Opcode::MOVZ | A64Opcode::MOVK | A64Opcode::MOVN => {}
                        _ => consts.kill(&d),
                    }
                }
                // Flag carry-across (see `preserves_nzcv`): this instruction
                // left NZCV untouched, so the compare that was pending before
                // it is still the live condition. Put it back — arms clear
                // `pending_cmp` unconditionally, which threw away the real
                // compare in the compiler's ubiquitous `cmp; mov; b.cond`
                // sequence and left the branch with `Compare(kind, Unknown,
                // 0)` → `if (? != 0)`.
                if carried_cmp.is_some() {
                    pending_cmp = carried_cmp;
                }
            }};
        }

        // AArch64 writes NZCV from only a handful of instructions; a plain
        // `MOV` between the `cmp` and the `b.cond` that reads it is the norm,
        // not an exception. Capture the pending compare across any opcode
        // that provably does not write the flags, and restore it in
        // `end_iteration!` after the arm has run.
        carried_cmp = if preserves_nzcv(insn.opcode) {
            pending_cmp.take()
        } else {
            None
        };

        match insn.opcode {
            // ---- moves / constants -------------------------------------
            A64Opcode::MOVZ | A64Opcode::MOVK | A64Opcode::MOVN => {
                let dst = gpr_of(&insn.operands[0])?;
                let (imm, shift): (i64, u8) = match insn.operands.get(1) {
                    Some(A64Op::ImmShift(v, sh)) => (*v as i64, *sh),
                    Some(A64Op::Imm16(v)) => (*v as i64, 0),
                    Some(A64Op::Immediate(v)) => (*v as i64, 0),
                    _ => (0, 0),
                };
                if let Some(full) = consts.apply(&insn.opcode, &dst, imm, shift) {
                    insns.push(Insn::move_(
                        Operand::reg(dst.clone()),
                        Operand::imm(full),
                        addr,
                    ));
                } else {
                    consts.kill(&dst);
                    insns.push(Insn::move_(Operand::reg(dst), Operand::Unknown, addr));
                }
                pending_cmp = None;
                end_iteration!();
                continue;
            }

            // ---- scalar FP moves / literals ------------------------------
            A64Opcode::FMOV if is_fp_operand(&insn.operands[0]) => {
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let wide = dst.starts_with("vd");
                // Reinterpreting a bit pattern: the WIDTH of the destination
                // decides whether those bits are an f32 or an f64.
                let lit_of = |bits: i64| {
                    if wide {
                        Operand::DoubleLiteral(f64::from_bits(bits as u64))
                    } else {
                        Operand::FloatLiteral(f32::from_bits(bits as u32))
                    }
                };
                let zero = if wide {
                    Operand::DoubleLiteral(0.0)
                } else {
                    Operand::FloatLiteral(0.0)
                };
                match &insn.operands[1] {
                    // FMOV Sd/Dd, #imm8
                    A64Op::ImmediateDouble(d) => {
                        let lit = if dst.starts_with("vs") {
                            Operand::FloatLiteral(*d as f32)
                        } else {
                            Operand::DoubleLiteral(*d)
                        };
                        insns.push(Insn::move_(Operand::reg(dst.clone()), lit, addr));
                    }
                    // FMOV Sd/Dd, Wn/Xn — move a GPR's BIT PATTERN into an FP
                    // register. IL2CPP materializes FP constants in a GPR
                    // (MOV/MOVK chain) and then reinterprets them, so a tracked
                    // constant folds into a real literal (`* 1.2f`); anything
                    // else is a genuine bit reinterpretation of a live GPR.
                    //
                    // The old code asked `fp_of` for a *GPR* operand — always
                    // `None` — and `unwrap_or_default()` on an `Option<String>`
                    // yields the EMPTY string, so `fmov s1, wzr` produced
                    // `vs1 = Register("")`: a register that does not exist.
                    src_gpr => {
                        let src = match gpr_of(src_gpr) {
                            // the zero register is FP +0.0, not a bit move
                            Some(r) if r == "wzr" || r == "xzr" => Some(zero),
                            Some(r) => Some(
                                consts
                                    .get(&r)
                                    .map(lit_of)
                                    .unwrap_or_else(|| Operand::reg(r)),
                            ),
                            None => fp_of(src_gpr).map(Operand::reg),
                        };
                        insns.push(Insn::move_(
                            Operand::reg(dst.clone()),
                            src.unwrap_or(Operand::Unknown),
                            addr,
                        ));
                    }
                }
                fp_written.insert(dst);
                pending_cmp = None;
                end_iteration!();
                continue;
            }
            // `fmov <Xd|Wd>, <Sn|Dn>` — the mirror direction: reinterpret an FP
            // register's bits into a GPR. The arm above is guarded on an FP
            // DESTINATION, so all 259 of these fell through the catch-all and
            // defined their GPR destination as `Unknown`. They are how
            // `BitConverter.SingleToInt32Bits`, float `GetHashCode` and every
            // `CompareTo`-style helper get their input, so the loss silently
            // deleted the value the whole method was about.
            A64Opcode::FMOV
                if !is_fp_operand(&insn.operands[0]) && is_fp_operand(&insn.operands[1]) =>
            {
                let dst = gpr_of(&insn.operands[0]).unwrap_or_default();
                if !dst.is_empty() && dst != "wzr" && dst != "xzr" {
                    insns.push(Insn::move_(Operand::reg(dst), fp_value_at(&insn, 1), addr));
                }
                pending_cmp = None;
            }

            // SIMD `mov vD.<arr>, vN.<arr>` is an alias of
            // `orr vD.<arr>, vN.<arr>, vN.<arr>`. Keep it out of the GPR
            // arm below: `gpr_of` intentionally has no answer for a SIMD
            // destination, and using `?` there used to abort the *entire*
            // method lift. This is common in IL2CPP prologues that preserve
            // Vector3/Vector4 arguments in callee-saved v8-v15 registers.
            // A true copy is represented as Move (so ordinary copy
            // propagation can see it); other vector ORRs retain all inputs
            // as an explicit SIMD operation until the renderer can lower
            // their element semantics.
            A64Opcode::ORR if simd_name(&insn.operands[0]).is_some() => {
                let dst = simd_name(&insn.operands[0]).unwrap();
                let lhs = any_operand_at(&insn, 1);
                let rhs = any_operand_at(&insn, 2);
                if lhs == rhs {
                    insns.push(Insn::move_(Operand::reg(dst), lhs, addr));
                    // A whole-register copy also defines every narrower view
                    // used by scalar FP instructions. Keeping only `vq13 =
                    // vq10` left a later `s13` read looking like an unrelated,
                    // undefined register (`vs52`), even though MOV Vd,Vn had
                    // copied those exact bits. Project the copied low S/D
                    // lanes into their scalar SSA names as well; the full Q/D
                    // move remains above for vector consumers.
                    if let (Some((dst_reg, dst_bytes)), Some((src_reg, src_bytes))) = (
                        simd_copy_container(&insn.operands[0]),
                        simd_copy_container(&insn.operands[2]),
                    ) {
                        if dst_bytes == src_bytes {
                            for lane in 0..(dst_bytes / 4) {
                                insns.push(Insn::move_(
                                    Operand::reg(s_lane_name(dst_reg, lane)),
                                    Operand::reg(s_lane_name(src_reg, lane)),
                                    addr,
                                ));
                            }
                            for lane in 0..(dst_bytes / 8) {
                                let dst = if lane == 0 {
                                    format!("vd{dst_reg}")
                                } else {
                                    format!("vd{dst_reg}[{lane}]")
                                };
                                let src = if lane == 0 {
                                    format!("vd{src_reg}")
                                } else {
                                    format!("vd{src_reg}[{lane}]")
                                };
                                insns.push(Insn::move_(Operand::reg(dst), Operand::reg(src), addr));
                            }
                        }
                    }
                } else {
                    insns.push(Insn::new(
                        Op::Simd("ORR".into()),
                        vec![Operand::reg(dst), lhs, rhs],
                        addr,
                    ));
                }
                pending_cmp = None;
            }
            A64Opcode::ORR => {
                let dst = gpr_of(&insn.operands[0])?;
                if matches!(&insn.operands[1], A64Op::Register(_, 31)) {
                    // `MOV xd, xm` decodes as ORR xd, xzr, xm<LSL 0>. operand_at
                    // only maps plain registers, so pull the shifted-register
                    // form (shift amount 0) explicitly — missing it made every
                    // `MOV x20, x0` receiver pin emit Unknown and broke the
                    // ABI-copy seeding downstream.
                    let src = match insn.operands.get(2) {
                        Some(A64Op::RegShift(_, 0, _, r)) if *r == 31 => Operand::imm(0),
                        Some(A64Op::RegShift(_, 0, _, r)) => Operand::reg(gpr(*r, false)),
                        _ => operand_at(&insn, 2).unwrap_or(Operand::Unknown),
                    };
                    insns.push(Insn::move_(Operand::reg(dst), src, addr));
                    pending_cmp = None;
                    end_iteration!();
                    continue;
                }
                let a = operand_at(&insn, 1);
                let b = resolve_src!(2);
                push_binary(&mut insns, Op::Arith(ArithKind::Or), dst, a, b, addr);
                pending_cmp = None;
            }

            A64Opcode::ADRP => {
                let dst = gpr_of(&insn.operands[0])?;
                if let A64Op::PCOffset(off) = insn.operands[1] {
                    // yaxpeax reports the raw immhi:immlo << 12; the ADRP
                    // base is the PAGE-ALIGNED PC, so mask off the low bits
                    // of the current address before adding the offset.
                    let page = ((addr as i64) & !0xFFFi64).wrapping_add(off);
                    consts.values.insert(dst.clone(), page);
                    insns.push(Insn::move_(Operand::reg(dst), Operand::imm(page), addr));
                }
                pending_cmp = None;
            }
            A64Opcode::ADR => {
                let dst = gpr_of(&insn.operands[0])?;
                if let A64Op::PCOffset(off) = insn.operands[1] {
                    insns.push(Insn::move_(
                        Operand::reg(dst),
                        Operand::imm(addr as i64 + off),
                        addr,
                    ));
                }
                pending_cmp = None;
            }

            // ---- arithmetic ----------------------------------------------
            // A SIMD destination is the vector encoding of ADD/SUB, which
            // this scalar arm does not model. Decline it (fall through to
            // the generic intrinsic arm) instead of `?`-aborting the entire
            // method: one `add v0.2s` used to erase every method that
            // contained one.
            A64Opcode::ADD | A64Opcode::SUB if gpr_of(&insn.operands[0]).is_some() => {
                let dst = gpr_of(&insn.operands[0])?;
                let kind = if insn.opcode == A64Opcode::ADD { ArithKind::Add } else { ArithKind::Sub };
                let imm = match insn.operands.get(2).cloned() {
                    Some(A64Op::ImmShift(v, _)) => Some(v as i64),
                    Some(A64Op::Immediate(v)) => Some(v as i64),
                    _ => None,
                };
                // `ADD/SUB sp, sp, #imm` moves the STACK POINTER: pure frame
                // bookkeeping, modelled as `ShiftStack` for the depth pass.
                //
                // `ADD Rd, sp, #imm` with Rd != sp is NOT that. It computes the
                // ADDRESS of a stack slot into Rd — the `&local` / `&enumerator`
                // idiom the compiler emits for every struct-typed local whose
                // method takes it by reference (e.g. `add x0, sp, #0x18` right
                // before the `IEnumerator.MoveNext(&enum)` call). The old guard
                // tested only the SOURCE, so those instructions were lifted as
                // `ShiftStack` too, with NO destination operand. Two defects
                // followed, and together they are the largest single source of
                // corruption in the dump:
                //
                //  1. The destination register was never defined anywhere, so
                //     every later read of it became an undefined local. That is
                //     the `__abi`/`__unknown` CALL-RECEIVER class: the ABI scrub
                //     pass marks undefined argument registers as `MissingAbiArg`,
                //     which is why 21,878 call sites render `__abi.Method(…)`
                //     whose real receiver is a live local.
                //  2. The stack-depth pass added the offset to the frame model
                //     even though sp does not move here, so a slot stored before
                //     the instruction and loaded after it got two different
                //     `stack_N` names and the spill round-trip stopped
                //     resolving — an undefined local per spill slot per method.
                if dst == "sp" {
                    let v = imm.unwrap_or(0);
                    let delta = if kind == ArithKind::Add { v } else { -v };
                    insns.push(Insn::new(Op::ShiftStack(delta), vec![], addr));
                } else if operand_at(&insn, 1) == Some(Operand::reg("sp")) && imm.is_some() {
                    // `Rd = &slot`. `Operand::Memory` on its own is a LOAD
                    // everywhere else in this IR, so the address is wrapped.
                    let v = imm.unwrap_or(0);
                    let addend = if kind == ArithKind::Add { v } else { -v };
                    insns.push(Insn::move_(
                        Operand::reg(dst),
                        Operand::AddressOf(Box::new(Operand::Memory { width: 0,
                            base: Box::new(Operand::reg("sp")),
                            index: None,
                            scale: 8,
                            addend,
                        })),
                        addr,
                    ));
                } else {
                    let a = operand_at(&insn, 1);
                    let b = resolve_src!(2);
                    push_binary(&mut insns, Op::Arith(kind), dst, a, b, addr);
                }
                pending_cmp = None;
            }
            // Same contract as ADD/SUB above: a SIMD destination is the
            // vector form (`and v1.8b, …`), not this GPR arm. Decline to the
            // generic intrinsic instead of aborting the whole method.
            A64Opcode::AND | A64Opcode::EOR | A64Opcode::BIC | A64Opcode::ORN
            | A64Opcode::EON | A64Opcode::LSLV | A64Opcode::LSRV
                if gpr_of(&insn.operands[0]).is_some() => {
                // `BIC`/`ORN`/`EON` are the encoder's "inverted second operand"
                // forms of `AND`/`ORR`/`EOR`: rather than materialize a `NOT`
                // the assembler folds the complement into the opcode. They had
                // NO arm, so the entire instruction fell to the catch-all and
                // its destination became `Unknown` — 547 `ORN` + 415 `BIC` +
                // 150 `BICS` + `EON`, the third-largest hole the census found.
                // The inversion is materialized as an explicit `Not` so the
                // operand count still matches the encoder's exactly.
                let invert_rhs = matches!(
                    insn.opcode,
                    A64Opcode::BIC | A64Opcode::ORN | A64Opcode::EON
                );
                // `and`/`eor`/`bic`/`orn`/`eon` with the ZERO register as
                // destination is a genuinely dead instruction: the result is
                // discarded and — unlike `ands`/`bics`/`tst` — none of these
                // five writes NZCV. So it must NOT become a flag test (that
                // would invent a condition out of nothing) and it must not
                // clear `pending_cmp` either, or a live compare sitting
                // between this insn and its branch would be destroyed. Reserve
                // the address with a `Nop` and leave the flags alone.
                if matches!(&insn.operands[0], A64Op::Register(_, 31)) {
                    insns.push(Insn::new(Op::Nop, vec![], addr));
                    end_iteration!();
                    continue;
                }
                let dst = gpr_of(&insn.operands[0])?;
                let kind = match insn.opcode {
                    A64Opcode::AND | A64Opcode::BIC => ArithKind::And,
                    A64Opcode::EOR | A64Opcode::EON => ArithKind::Xor,
                    A64Opcode::ORN => ArithKind::Or,
                    A64Opcode::LSLV => ArithKind::Shl,
                    _ => ArithKind::Shr,
                };
                let a = operand_at(&insn, 1);
                let mut b = resolve_src!(2);
                if invert_rhs {
                    if let Some(bv) = b {
                        let t = Operand::reg(format!("xt{}", virt_serial));
                        virt_serial += 1;
                        insns.push(Insn::new(Op::Not, vec![t.clone(), bv], addr));
                        b = Some(t);
                    }
                }
                push_binary(&mut insns, Op::Arith(kind), dst, a, b, addr);
                pending_cmp = None;
            }
            A64Opcode::MUL | A64Opcode::SDIV | A64Opcode::UDIV => {
                let dst = gpr_of(&insn.operands[0])?;
                let kind = match insn.opcode {
                    A64Opcode::MUL => ArithKind::Mul,
                    _ => ArithKind::Div,
                };
                let a = operand_at(&insn, 1);
                let b = operand_at(&insn, 2);
                push_binary(&mut insns, Op::Arith(kind), dst, a, b, addr);
                pending_cmp = None;
            }
            A64Opcode::NEG => {
                // [dest, src] — renderers read operand[1] as the negated value
                let dst = gpr_of(&insn.operands[0])?;
                let d = Operand::reg(dst);
                insns.push(Insn::new(Op::Negate, vec![d.clone(), d], addr));
                pending_cmp = None;
            }

            // ---- scalar FP arithmetic -------------------------------------
            A64Opcode::FADD | A64Opcode::FSUB | A64Opcode::FMUL | A64Opcode::FDIV
                if is_fp_operand(&insn.operands[0]) =>
            {
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let a = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                let bsrc = fp_of(&insn.operands.get(2).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                let kind = match insn.opcode {
                    A64Opcode::FADD => FloatArithKind::Add,
                    A64Opcode::FSUB => FloatArithKind::Sub,
                    A64Opcode::FMUL => FloatArithKind::Mul,
                    _ => FloatArithKind::Div,
                };
                insns.push(Insn::new(
                    Op::FloatArith(kind),
                    vec![Operand::reg(dst.clone()), Operand::reg(a), Operand::reg(bsrc)],
                    addr,
                ));
                fp_written.insert(dst);
                pending_cmp = None;
            }
            A64Opcode::FNEG if is_fp_operand(&insn.operands[0]) => {
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let a = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                insns.push(Insn::new(
                    Op::FloatArith(FloatArithKind::Neg),
                    vec![Operand::reg(dst.clone()), Operand::reg(a)],
                    addr,
                ));
                fp_written.insert(dst);
                pending_cmp = None;
            }
            // unary FP ops: FSQRT/FABS define their operand-0 register from
            // operand 1. Without them the FSQRT in jump-math helpers
            // (`sqrt(-2*h/a)`) defined its dest as Unknown and rendered as
            // `__unknown` in otherwise fully-recovered formulas.
            A64Opcode::FSQRT | A64Opcode::FABS
                if is_fp_operand(&insn.operands[0]) =>
            {
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let a = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                let kind = if matches!(insn.opcode, A64Opcode::FSQRT) {
                    FloatArithKind::Sqrt
                } else {
                    FloatArithKind::Abs
                };
                insns.push(Insn::new(
                    Op::FloatArith(kind),
                    vec![Operand::reg(dst.clone()), Operand::reg(a)],
                    addr,
                ));
                fp_written.insert(dst);
                pending_cmp = None;
            }

            // ---- vector FP arithmetic (2S/4S lane decomposition) ----------
            // IL2CPP compiles whole-vector float math (Vector3/Quaternion/
            // Color helpers, matrix rows, ...) to SIMD FMUL.2S/.4S, FADD,
            // ... forms. Lower each lane to the scalar register it aliases
            // (Vn lane k with 32-bit elements == S(4n+k), the same vs name
            // the scalar pipeline uses) so the existing FloatArith / type
            // propagation / renderer recovers the math instead of Unknown.
            A64Opcode::FADD | A64Opcode::FSUB | A64Opcode::FMUL | A64Opcode::FDIV
            | A64Opcode::FNEG | A64Opcode::FABS | A64Opcode::FSQRT
                if vec_fp2s(&insn.operands[0]).is_some() =>
            {
                let (vn, lanes) = vec_fp2s(&insn.operands[0]).unwrap();
                let unary =
                    matches!(insn.opcode, A64Opcode::FNEG | A64Opcode::FABS | A64Opcode::FSQRT);
                let kind = match insn.opcode {
                    A64Opcode::FADD => FloatArithKind::Add,
                    A64Opcode::FSUB => FloatArithKind::Sub,
                    A64Opcode::FMUL => FloatArithKind::Mul,
                    A64Opcode::FDIV => FloatArithKind::Div,
                    A64Opcode::FNEG => FloatArithKind::Neg,
                    A64Opcode::FABS => FloatArithKind::Abs,
                    _ => FloatArithKind::Sqrt,
                };
                let a_reg = |k: u8| vec_src_lane(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing), k);
                let b_reg = |k: u8| {
                    if unary {
                        String::new()
                    } else {
                        vec_src_lane(
                            &insn.operands.get(2).copied().unwrap_or(A64Op::Nothing),
                            k,
                        )
                    }
                };
                let mut ok = true;
                for k in 0..lanes {
                    let dst = s_lane_name(vn, k);
                    let a = a_reg(k);
                    let b = b_reg(k);
                    if a.is_empty() || (!unary && b.is_empty()) {
                        ok = false;
                        break;
                    }
                    let mut ops = vec![Operand::reg(dst.clone()), Operand::reg(a)];
                    if !unary {
                        ops.push(Operand::reg(b));
                    }
                    insns.push(Insn::new(Op::FloatArith(kind), ops, addr));
                    fp_written.insert(dst);
                }
                if !ok {
                    // unsupported element/shape — degrade like the default arm
                    insns.push(Insn::move_(
                        Operand::reg(s_lane_name(vn, 0)),
                        Operand::Unknown,
                        addr,
                    ));
                }
                pending_cmp = None;
            }
            // DUP splat: `mov vD.2s, vN.s[l]` — every lane copies the source
            // element's scalar register. (yaxpeax reports DUP as MOV.)
            A64Opcode::DUP
                if vec_fp2s(&insn.operands[0]).is_some()
                    && matches!(insn.operands.get(1), Some(A64Op::SIMDRegisterElementsLane(_, _, SIMDSizeCode::S, _))) =>
            {
                let (vn, lanes) = vec_fp2s(&insn.operands[0]).unwrap();
                let (svn, l) = match &insn.operands[1] {
                    A64Op::SIMDRegisterElementsLane(_, n, SIMDSizeCode::S, lane) => (*n, *lane),
                    _ => unreachable!(),
                };
                let src = s_lane_name(svn, l);
                for k in 0..lanes {
                    let dst = s_lane_name(vn, k);
                    insns.push(Insn::move_(
                        Operand::reg(dst.clone()),
                        Operand::reg(src.clone()),
                        addr,
                    ));
                    fp_written.insert(dst);
                }
                pending_cmp = None;
            }

            // ---- loads/stores ---------------------------------------------
            A64Opcode::LDR | A64Opcode::LDUR => {
                let dst_op = insn.operands[0].clone();
                if matches!(dst_op, A64Op::SIMDRegister(_, _)) {
                    // FP/SIMD load — possibly from literal pool.
                    //
                    // `fp_of` answers only for the scalar `S`/`D` views, so a
                    // `ldr q0, [x2, #0x30]` (the 128-bit VECTOR reload) named
                    // its destination the EMPTY string: the load emitted
                    // `Register("") = *(...)` and the value was lost. Every
                    // later reader of that register — the byte-shuffle `ext`,
                    // the field `str q0` — then read an undefined `vq0`
                    // (290 findings, the top arm64 corruption cluster).
                    // `simd_name` names every SIMD shape, is identical to
                    // `fp_of` for `S`/`D`, and adds the `Q` view.
                    let dst = simd_name(&dst_op).unwrap_or_default();
                    let dst2 = dst.clone();
                    // The register VIEW states the access width: `ldr s0` reads
                    // four bytes, `ldr d0` eight, `ldr q0` sixteen. Leaving it
                    // at 0 made every FP load render as `*(nint*)` — an 8-byte
                    // access to a float field, which is also why a `ref float`
                    // parameter's read looked like a pointer-sized integer.
                    let access_width = fp_reg_bytes(&dst).unwrap_or(0);
                    let loaded = try_literal_pool_load(&insn, addr, bytes, base_rva, offset, pool_reader);
                    match loaded {
                        Some(val) => {
                            insns.push(Insn::move_(Operand::reg(dst), val, addr));
                        }
                        None => {
                            let mem = memory_operand(insn.operands.get(1))
                                .map(|m| m.with_width(access_width));
                            insns.push(Insn::move_(
                                Operand::reg(dst),
                                mem.unwrap_or(Operand::Unknown),
                                addr,
                            ));
                        }
                    }
                    push_writeback(&mut insns, memory_writeback(insn.operands.get(1)), addr);
                    fp_written.insert(dst2);
                    pending_cmp = None;
                    end_iteration!();
                    continue;
                }
                let dst = gpr_of(&dst_op)?;
                let mem = memory_operand(insn.operands.get(1));
                insns.push(Insn::move_(
                    Operand::reg(dst),
                    mem.unwrap_or(Operand::Unknown),
                    addr,
                ));
                push_writeback(&mut insns, memory_writeback(insn.operands.get(1)), addr);
                pending_cmp = None;
            }
            A64Opcode::LDRB | A64Opcode::LDRH | A64Opcode::LDRSB | A64Opcode::LDRSH
            | A64Opcode::LDRSW | A64Opcode::LDTR | A64Opcode::LDTRB | A64Opcode::LDTRH
            | A64Opcode::LDTRSB | A64Opcode::LDTRSH | A64Opcode::LDTRSW | A64Opcode::LDURB
            | A64Opcode::LDURH | A64Opcode::LDURSB | A64Opcode::LDURSH | A64Opcode::LDURSW
            // Acquire/release + exclusive-access loads. These are ordinary
            // loads for value purposes: `ldar` is `[Rt, Rn]`, `ldaxr` is
            // `[Rt, RegPreIndex(Rn, 0)]` — the same two shapes the arms around
            // them already accept (`memory_operand` handles both). They were
            // missing, so the census put `LDAXR` at 326, `LDARB` at 118,
            // `LDAR` at 70, `LDXR`/`LDAXRB` below that: ~550 live values
            // reduced to `Unknown`, which then poisoned every branch, field
            // access and call argument downstream of the atomic.
            | A64Opcode::LDAR | A64Opcode::LDARB | A64Opcode::LDARH | A64Opcode::LDAPR
            | A64Opcode::LDAPRB | A64Opcode::LDAPRH | A64Opcode::LDXR | A64Opcode::LDXRB
            | A64Opcode::LDXRH | A64Opcode::LDAXR | A64Opcode::LDAXRB | A64Opcode::LDAXRH => {
                let dst = gpr_of(&insn.operands[0])?;
                // ACCESS WIDTH: this arm holds the NARROW loads (`ldrb`/`ldrh`
                // and the signed, unscaled, acquire/release and exclusive
                // twins). All of them used to lift exactly like `ldr`, so a
                // one-byte field read rendered as an eight-byte
                // `*(nint*)(...)`. `ldtr`/`ldur`/`ldar`/`ldxr` with a `W`
                // destination are plain words and stay at 0 (the register
                // operand already carries that width).
                let width = match insn.opcode {
                    A64Opcode::LDRB
                    | A64Opcode::LDTRB
                    | A64Opcode::LDURB
                    | A64Opcode::LDARB
                    | A64Opcode::LDAPRB
                    | A64Opcode::LDXRB
                    | A64Opcode::LDAXRB
                    | A64Opcode::LDRSB
                    | A64Opcode::LDTRSB
                    | A64Opcode::LDURSB => 1u8,
                    A64Opcode::LDRH
                    | A64Opcode::LDTRH
                    | A64Opcode::LDURH
                    | A64Opcode::LDARH
                    | A64Opcode::LDAPRH
                    | A64Opcode::LDXRH
                    | A64Opcode::LDAXRH
                    | A64Opcode::LDRSH
                    | A64Opcode::LDTRSH
                    | A64Opcode::LDURSH => 2,
                    A64Opcode::LDRSW | A64Opcode::LDTRSW | A64Opcode::LDURSW => 4,
                    _ => 0,
                };
                let mem = memory_operand(insn.operands.get(1))
                    .map(|m| m.with_width(width))
                    .unwrap_or(Operand::Unknown);
                insns.push(Insn::move_(Operand::reg(dst), mem, addr));
                push_writeback(&mut insns, memory_writeback(insn.operands.get(1)), addr);
                pending_cmp = None;
            }
            // `ldxp`/`ldaxp`: two-register exclusive load, `[Rt, Rt2, [Rn]]`.
            // Dropping it severed BOTH halves of an atomic pair read.
            // `ldpsw` is the same two-register shape and lands on the same
            // element-size rule: both destinations are `W` registers, so the
            // pair occupies 4-byte slots.
            A64Opcode::LDXP | A64Opcode::LDAXP | A64Opcode::LDPSW => {
                let base = memory_operand(insn.operands.get(2));
                if let (Some(a), Some(b), Some(m0)) = (
                    gpr_of(&insn.operands[0]),
                    gpr_of(&insn.operands[1]),
                    base,
                ) {
                    let esz = if gpr_bits(&insn.operands[0]) == 64 { 8 } else { 4 };
                    insns.push(Insn::move_(Operand::reg(a), m0.clone(), addr));
                    insns.push(Insn::move_(Operand::reg(b), mem_plus(&m0, esz as i64), addr));
                }
                pending_cmp = None;
            }
            A64Opcode::STR | A64Opcode::STUR | A64Opcode::STRB | A64Opcode::STRH
            | A64Opcode::STTR | A64Opcode::STTRB | A64Opcode::STTRH
            // Scaled and unscaled narrow stores (`sturb`/`sturh`) — 373 + 148
            // occurrences. Single-threaded memory semantics are identical to
            // `strb`/`strh`; only the addressing mode differs, and that is
            // already carried by the memory operand.
            | A64Opcode::STURB | A64Opcode::STURH
            // Release stores (`stlr`/`stlrb`/`stlrh`) — the store half of the
            // acquire/release fences around a field write. `[Rt, Rn]`.
            | A64Opcode::STLR | A64Opcode::STLRB | A64Opcode::STLRH => {
                // The stored VALUE comes from EITHER register file.
                //
                // `operand_at` names GPRs only, so `str d0, [x19, #0x10]` — the
                // store half of a double field write — returned `None` and the
                // instruction was dropped with NO IR at all. The load side has
                // always handled SIMD (`LDR` checks `SIMDRegister` explicitly),
                // so the pipeline could read a double it could never write: the
                // field store vanished from the output while the read that
                // produced its value stayed, which is the single largest
                // instruction-coverage hole in the whole lifter (2,359 `str`/
                // `stur` on the reference binary) and a genuine semantic loss —
                // a side effect the native code performs and the C# does not.
                let val = value_operand_at(&insn, 0);
                // ACCESS WIDTH: `strb`/`strh` write ONE or TWO bytes. Emitting
                // them as `*(nint*)` claimed an eight-byte store across a
                // one-byte slot — a write that would clobber the seven
                // following fields. The `stxr` family is handled in its own arm
                // below.
                let width = match insn.opcode {
                    A64Opcode::STRB
                    | A64Opcode::STTRB
                    | A64Opcode::STURB
                    | A64Opcode::STLRB => 1u8,
                    A64Opcode::STRH
                    | A64Opcode::STTRH
                    | A64Opcode::STURH
                    | A64Opcode::STLRH => 2,
                    _ => {
                        if let Some(name) = simd_name(&insn.operands[0]) {
                            // `str s0, [x]` writes FOUR bytes, not eight — see
                            // the FP load note. The structurer renders a
                            // width-4 store whose value is a float register as
                            // `*(float*)`, so the store keeps its real type.
                            fp_reg_bytes(&name).unwrap_or(0)
                        } else if insn.operands.first().and_then(gpr_of).is_some() {
                            (gpr_bits(&insn.operands[0]) / 8) as u8
                        } else {
                            0
                        }
                    }
                };
                let mem = memory_operand(insn.operands.get(1)).map(|m| m.with_width(width));
                if let (true, Some(m)) = (!matches!(val, Operand::Unknown), mem) {
                    insns.push(Insn::move_(m, val, addr));
                }
                push_writeback(&mut insns, memory_writeback(insn.operands.get(1)), addr);
                pending_cmp = None;
            }
            // `stxr`/`stlxr`/`stxp`/`stlxp` — store-exclusive. Operands are
            // `[Rs(status), Rt(value), [Rn]]` (plus `Rt2` for the pair form).
            //
            // The STORE is emitted, because it is the observable effect (and
            // its absence is why event accessors rendered as an empty retry
            // shell: the write-back was simply gone). The status register is
            // deliberately NOT defined here: it is the outcome of the atomic
            // attempt, not a value the program computes, and defining it as
            // `0` would fold the `cbnz Ws, retry` exit test to a constant and
            // turn the CAS retry loop into straight-line code — a bigger
            // distortion than leaving the outcome symbolic.
            A64Opcode::STXR | A64Opcode::STXRB | A64Opcode::STXRH | A64Opcode::STLXR
            | A64Opcode::STLXRB | A64Opcode::STLXRH => {
                // Either register file — see the plain-store arm.
                let val = value_operand_at(&insn, 1);
                // Narrow exclusive stores keep their width too (`stxrb` = 1,
                // `stxrh` = 2); `stxr`/`stlxr` are word-sized.
                let width = match insn.opcode {
                    A64Opcode::STXRB | A64Opcode::STLXRB => 1u8,
                    A64Opcode::STXRH | A64Opcode::STLXRH => 2,
                    _ => 0,
                };
                let mem = memory_operand(insn.operands.get(2)).map(|m| m.with_width(width));
                if let (true, Some(m)) = (!matches!(val, Operand::Unknown), mem) {
                    insns.push(Insn::move_(m, val, addr));
                }
                push_writeback(&mut insns, memory_writeback(insn.operands.get(2)), addr);
                pending_cmp = None;
            }
            A64Opcode::STXP | A64Opcode::STLXP => {
                let base = memory_operand(insn.operands.get(3));
                if let (Some(a), Some(b), Some(m0)) = (
                    operand_at(&insn, 1),
                    operand_at(&insn, 2),
                    base,
                ) {
                    let esz = if gpr_bits(&insn.operands[1]) == 64 { 8 } else { 4 };
                    insns.push(Insn::move_(m0.clone(), a, addr));
                    insns.push(Insn::move_(mem_plus(&m0, esz as i64), b, addr));
                }
                pending_cmp = None;
            }

            // ---- vector loads/stores (LD1/ST1, 32-bit float elements) ----
            // Whole-vector struct math loads/stores its operands with
            // `ld1 {v.2s/4s}, [x]` / lane forms. Lower to per-lane scalar
            // 4-byte accesses so the stack analyzer + field oracle see the
            // individual floats (and resolve struct fields by offset).
            A64Opcode::LD1 | A64Opcode::LD1R => {
                let base_mem = memory_operand(insn.operands.get(1));
                let is_rep = matches!(insn.opcode, A64Opcode::LD1R);
                // full-register: ld1 {v.2s/4s}, [x] / ld1r {v.2s/4s}, [x]
                if let Some((reg, lanes)) = match &insn.operands[0] {
                    A64Op::SIMDRegisterGroup(SIMDSizeCode::D, n, SIMDSizeCode::S, c) => {
                        if *c >= 1 { Some((*n, 2u8)) } else { None }
                    }
                    A64Op::SIMDRegisterGroup(SIMDSizeCode::Q, n, SIMDSizeCode::S, c) => {
                        if *c >= 1 { Some((*n, 4u8)) } else { None }
                    }
                    _ => None,
                } {
                    if is_rep {
                        // ld1r: load the element once, broadcast to all lanes
                        if let Some(m0) = base_mem.clone() {
                            let loaded = Operand::reg(s_lane_name(reg, 0));
                            insns.push(Insn::move_(loaded, m0, addr));
                            fp_written.insert(s_lane_name(reg, 0));
                            for k in 1..lanes {
                                let dst = s_lane_name(reg, k);
                                insns.push(Insn::move_(
                                    Operand::reg(dst.clone()),
                                    Operand::reg(s_lane_name(reg, 0)),
                                    addr,
                                ));
                                fp_written.insert(dst);
                            }
                        }
                    } else if let Some(m0) = base_mem.clone() {
                        for k in 0..lanes {
                            let dst = s_lane_name(reg, k);
                            insns.push(Insn::move_(
                                Operand::reg(dst.clone()),
                                mem_plus(&m0, k as i64 * 4),
                                addr,
                            ));
                            fp_written.insert(dst);
                        }
                    }
                    pending_cmp = None;
                    end_iteration!();
                    continue;
                }
                // single-element lane: ld1 {v3.s}[1], [x] → s(4*3+1)
                if let Some((reg, lane)) = match &insn.operands[0] {
                    A64Op::SIMDRegisterGroupLane(n, SIMDSizeCode::S, lane, _) => Some((*n, *lane)),
                    _ => None,
                } {
                    if let Some(m) = base_mem {
                        let dst = s_lane_name(reg, lane);
                        insns.push(Insn::move_(Operand::reg(dst.clone()), m, addr));
                        fp_written.insert(dst);
                    }
                    pending_cmp = None;
                    end_iteration!();
                    continue;
                }
            }
            A64Opcode::ST1 => {
                // full-register: st1 {v.2s/4s}, [x]
                let base_mem = memory_operand(insn.operands.get(1));
                if let Some((reg, lanes)) = match &insn.operands[0] {
                    A64Op::SIMDRegisterGroup(SIMDSizeCode::D, n, SIMDSizeCode::S, c) => {
                        if *c >= 1 { Some((*n, 2u8)) } else { None }
                    }
                    A64Op::SIMDRegisterGroup(SIMDSizeCode::Q, n, SIMDSizeCode::S, c) => {
                        if *c >= 1 { Some((*n, 4u8)) } else { None }
                    }
                    _ => None,
                } {
                    if let Some(m0) = base_mem {
                        for k in 0..lanes {
                            let src = s_lane_name(reg, k);
                            insns.push(Insn::move_(
                                mem_plus(&m0, k as i64 * 4),
                                Operand::reg(src),
                                addr,
                            ));
                        }
                    }
                    pending_cmp = None;
                    end_iteration!();
                    continue;
                }
                // single-element lane: st1 {v1.s}[1], [x]
                if let Some((reg, lane)) = match &insn.operands[0] {
                    A64Op::SIMDRegisterGroupLane(n, SIMDSizeCode::S, lane, _) => Some((*n, *lane)),
                    _ => None,
                } {
                    if let Some(m) = base_mem {
                        let src = s_lane_name(reg, lane);
                        insns.push(Insn::move_(m, Operand::reg(src), addr));
                    }
                    pending_cmp = None;
                    end_iteration!();
                    continue;
                }
            }
            A64Opcode::LDP => {
                // Byte-accurate offset — see STP note below. Post-index form
                // (`ldp x29,x30,[sp],#16`, the canonical epilogue) accesses
                // [base+0]/[base+size] and adds the increment AFTER; without
                // this arm the whole pair load vanished from the IR.
                let (base, off, post_inc): (Option<String>, i64, i64) = match insn.operands.get(2) {
                    Some(A64Op::RegisterOrSP(_, r)) => (Some(gpr(*r, true)), 0i64, 0),
                    Some(A64Op::RegPreIndex(r, o, _)) => (Some(gpr(*r, true)), *o as i64, 0),
                    Some(A64Op::RegPostIndex(r, o)) => (Some(gpr(*r, true)), 0i64, *o as i64),
                    _ => (None, 0, 0),
                };
                match (gpr_of(&insn.operands[0]), gpr_of(&insn.operands[1]), base.clone()) {
                    (Some(a), Some(b), Some(bs)) => {
                        let elem_width = register_width(&insn.operands[0]);
                        let elem_width = if elem_width == 0 { 8 } else { elem_width };
                        insns.push(Insn::move_(
                            Operand::reg(a),
                            Operand::Memory { width: elem_width, base: Box::new(Operand::reg(bs.clone())), index: None, scale: elem_width, addend: off },
                            addr,
                        ));
                        insns.push(Insn::move_(
                            Operand::reg(b),
                            Operand::Memory { width: elem_width, base: Box::new(Operand::reg(bs)), index: None, scale: elem_width, addend: off + elem_width as i64 },
                            addr,
                        ));
                    }
                    // SIMD pair reload (`LDP s8, s9, [sp]`): the compiler
                    // spills float/vector registers through the frame right
                    // before tail-merged calls — dropping these severs the
                    // def→use chain for every argument of that call.
                    _ => {
                        if let (Some((a, b, esz)), Some(bs)) = (fp_pair_of(&insn), base) {
                            let mem1 = Operand::Memory { width: esz, base: Box::new(Operand::reg(bs.clone())), index: None, scale: esz, addend: off };
                            let mem2 = Operand::Memory { width: esz, base: Box::new(Operand::reg(bs)), index: None, scale: esz, addend: off + esz as i64 };
                            let ra = a.clone();
                            let rb = b.clone();
                            insns.push(Insn::move_(Operand::reg(ra.clone()), mem1, addr));
                            insns.push(Insn::move_(Operand::reg(rb.clone()), mem2, addr));
                            fp_written.insert(a);
                            fp_written.insert(b);
                        }
                    }
                }
                if post_inc != 0 {
                    insns.push(Insn::new(Op::ShiftStack(post_inc), vec![], addr));
                }
                pending_cmp = None;
            }
            A64Opcode::STP => {
                // RegPreIndex(base, offset, _wb): the offset is a BYTE count
                // already scaled by the decoder — the old code treated every
                // STP/LDP as storing to [sp+0]/[sp+8] (addends 0 and 8),
                // clobbering real callee-saved spills like
                // `STP x22,x21,[sp,#0x40]`. Spill slots then aliased each
                // other and the SSA renaming propagated the wrong values.
                // RegPostIndex (`stp x0,x1,[sp],#16` arg-restore epilogues):
                // access at [base+0]/[base+size], increment afterwards.
                let (base, off, post_inc): (Option<String>, i64, i64) = match insn.operands.get(2) {
                    Some(A64Op::RegisterOrSP(_, r)) => (Some(gpr(*r, true)), 0i64, 0),
                    Some(A64Op::RegPreIndex(r, o, _)) => (Some(gpr(*r, true)), *o as i64, 0),
                    Some(A64Op::RegPostIndex(r, o)) => (Some(gpr(*r, true)), 0i64, *o as i64),
                    _ => (None, 0, 0),
                };
                match (
                    value_operand_at_opt(&insn, 0),
                    value_operand_at_opt(&insn, 1),
                    base.clone(),
                ) {
                    (Some(v1), Some(v2), Some(bs)) => {
                        // The second element sits `st_scale` bytes after the
                        // first. `starts_with("v")` was standing in for "is a
                        // vector register", and every FP name starts with `v`
                        // — including `vd` (64-bit), which it then advanced by
                        // 4 instead of 8, aliasing the second double onto the
                        // low half of the first. Ask the name for its width.
                        let st_scale = register_width(&insn.operands[0]).max(1);
                        insns.push(Insn::move_(
                            Operand::Memory { width: st_scale, base: Box::new(Operand::reg(bs.clone())), index: None, scale: st_scale, addend: off },
                            v1,
                            addr,
                        ));
                        insns.push(Insn::move_(
                            Operand::Memory { width: st_scale, base: Box::new(Operand::reg(bs)), index: None, scale: st_scale, addend: off + st_scale as i64 },
                            v2,
                            addr,
                        ));
                    }
                    // SIMD pair spill (`STP s0, s1, [sp]`).
                    _ => {
                        if let (Some((v1, v2, esz)), Some(bs)) = (fp_pair_of(&insn), base) {
                            let mem1 = Operand::Memory { width: esz, base: Box::new(Operand::reg(bs.clone())), index: None, scale: esz, addend: off };
                            let mem2 = Operand::Memory { width: esz, base: Box::new(Operand::reg(bs)), index: None, scale: esz, addend: off + esz as i64 };
                            insns.push(Insn::move_(mem1, Operand::reg(v1), addr));
                            insns.push(Insn::move_(mem2, Operand::reg(v2), addr));
                        }
                    }
                }
                if post_inc != 0 {
                    insns.push(Insn::new(Op::ShiftStack(post_inc), vec![], addr));
                }
                pending_cmp = None;
            }

            // ---- compares ---------------------------------------------------
            A64Opcode::SUBS | A64Opcode::ADDS | A64Opcode::ANDS | A64Opcode::BICS => {
                if matches!(&insn.operands[0], A64Op::Register(_, 31)) {
                    // The `test` family: `tst a, b` is `ands xzr, a, b, #0`
                    // (and `bics xzr, a, b` the complement form). Both test
                    // `a AND ±b` against ZERO. Recording the operand pair as a
                    // plain `Cmp` made every mask test render as `a == b`.
                    if matches!(insn.opcode, A64Opcode::ANDS | A64Opcode::BICS) {
                        let lhs = resolve_src!(1).unwrap_or(Operand::Unknown);
                        let rhs = resolve_src!(2);
                        let inverted = matches!(insn.opcode, A64Opcode::BICS);
                        pending_cmp = Some(test_flags(
                            &mut insns,
                            &mut virt_serial,
                            lhs,
                            rhs,
                            inverted,
                            addr,
                        ));
                        insns.push(Insn::new(Op::Nop, vec![], addr));
                        end_iteration!();
                        continue;
                    }
                    // `cmp xN, xM` is encoded `subs xzr, xN, xM`, and yaxpeax
                    // hands back xM as a `RegShift(LSL, 0, …)` operand — NOT a
                    // plain Register. `operand_at` returns None for RegShift,
                    // so the RHS silently fell back to `0`: every
                    // register-register compare was lifted as `a == 0` /
                    // `a >= 0`, corrupting the condition of every guard whose
                    // second operand is a live value instead of an immediate.
                    // Resolve through `resolve_src!` (which materializes a
                    // shift temp when the amount is non-zero) so the RHS
                    // survives. Same for a shifted LHS.
                    // Resolve the operands BEFORE reading `insns.len()`:
                    // `resolve_src!` can emit a shift temp, and `at` must be
                    // an index at or after everything that produced this
                    // compare.
                    let lhs = resolve_src!(1).unwrap_or(Operand::Unknown);
                    let rhs = resolve_src!(2).unwrap_or(Operand::imm(0));
                    pending_cmp = Some(FlagState::Cmp { lhs, rhs, at: insns.len() });
                    // Folded into the following B.cond — BUT still reserve an
                    // address entry (Nop) so a branch targeting THIS compare
                    // (loop back-edges, merged test-and-retest idioms) resolves
                    // instead of falling off the listing to the exit sentinel.
                    insns.push(Insn::new(Op::Nop, vec![], addr));
                    end_iteration!();
                    continue;
                }
                let dst = gpr_of(&insn.operands[0])?;
                let kind = match insn.opcode {
                    A64Opcode::SUBS => ArithKind::Sub,
                    A64Opcode::ADDS => ArithKind::Add,
                    _ => ArithKind::And,
                };
                let a = operand_at(&insn, 1);
                let mut b = operand_at(&insn, 2);
                // `bics xd, xn, xm` writes `xn & ~xm` AND sets the flags; the
                // value form needs the inverted operand too.
                if matches!(insn.opcode, A64Opcode::BICS) {
                    if let Some(bv) = b {
                        let t = Operand::reg(format!("xt{}", virt_serial));
                        virt_serial += 1;
                        insns.push(Insn::new(Op::Not, vec![t.clone(), bv], addr));
                        b = Some(t);
                    }
                }
                let flag_dst = dst.clone();
                push_binary(&mut insns, Op::Arith(kind), dst, a, b, addr);
                // SUBS/ADDS/ANDS/BICS ALSO write NZCV, so the following B.cond
                // tests the RESULT against zero. Clearing the compare here left
                // every `subs w1, w8, #1 ; b.lt` with `Compare(kind, Unknown,
                // 0)` — `if (? < 0)` — the top arm64 wrong-logic cluster
                // (`cond-branch without real compare`, 176 findings). The `xzr`
                // form above folds its operand pair directly; the value form
                // compares the computed destination, which is exactly what the
                // flag-setting operation left in the register. `at` points just
                // past the arith, whose own address already anchors the group.
                pending_cmp = Some(FlagState::Cmp {
                    lhs: Operand::reg(flag_dst),
                    rhs: Operand::imm(0),
                    at: insns.len(),
                });
            }

            // ---- FP compares ---------------------------------------------------
            // `fcmp s0, s1` / `fcmp d0, #0.0` set NZCV exactly like SUBS does;
            // the following B.cond consumes the pending compare. Without this
            // arm the catch-all defined the FP register as Unknown and cleared
            // pending_cmp, so every float decision rendered as
            // `Compare(kind, Unknown, 0)` → `if (__unknown > 0)` — the mass
            // "logic vanished" signature in float-heavy methods.
            A64Opcode::FCMP | A64Opcode::FCMPE => {
                let lhs_name = fp_of(&insn.operands[0]).unwrap_or_default();
                let lhs = Operand::reg(lhs_name.clone());
                let rhs = match insn.operands.get(1) {
                    // `fcmp <S|D>n, #0.0` — immediate zero of the operand width
                    Some(A64Op::ImmediateDouble(_)) => {
                        if lhs_name.starts_with("vd") {
                            Operand::DoubleLiteral(0.0)
                        } else {
                            Operand::FloatLiteral(0.0)
                        }
                    }
                    Some(A64Op::SIMDRegister(_, _)) => fp_of(&insn.operands[1])
                        .map(Operand::reg)
                        .unwrap_or(Operand::Unknown),
                    _ => Operand::Unknown,
                };
                pending_cmp = Some(FlagState::Cmp { lhs, rhs, at: insns.len() });
                // Folded into the following B.cond, but reserve the address
                // with a Nop: the SECOND compare of an if/elseif chain is a
                // common branch target (`b.pl 0x…` → the next `fcmp`), and
                // without an address entry that edge resolves to the exit
                // sentinel and collapses the whole structure.
                insns.push(Insn::new(Op::Nop, vec![], addr));
                end_iteration!();
                continue;
            }
            // `fccmp s0, s1, #nzcv, cond` — the floating-point twin of `ccmp`,
            // and the reason float compound conditions (`if (x < a || x > b)`)
            // lost their shape: it is how the compiler joins two float tests
            // into one branch. Same `Merge` semantics as `CCMP`, with FP
            // operands in the first two slots.
            A64Opcode::FCCMP | A64Opcode::FCCMPE => {
                let lhs = fp_of(&insn.operands[0]).map(Operand::reg).unwrap_or(Operand::Unknown);
                let rhs = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .map(Operand::reg)
                    .unwrap_or(Operand::Unknown);
                let nzcv = match insn.operands.get(2) {
                    Some(A64Op::Immediate(v)) => *v as u8,
                    _ => 0,
                };
                let ccmp_cond = match insn.operands.get(3) {
                    Some(A64Op::ConditionCode(c)) => *c,
                    _ => 0,
                };
                pending_cmp = match pending_cmp.take() {
                    Some(FlagState::Cmp { lhs: pl, rhs: pr, at: prev_at }) => {
                        Some(FlagState::Merge {
                            prev_lhs: pl,
                            prev_rhs: pr,
                            prev_at,
                            ccmp_cond,
                            lhs,
                            rhs,
                            at: insns.len(),
                            nzcv,
                        })
                    }
                    // No incoming compare to guard on — keep this compare's
                    // operands so the branch still tests real values.
                    _ => Some(FlagState::Cmp { lhs, rhs, at: insns.len() }),
                };
                insns.push(Insn::new(Op::Nop, vec![], addr));
                end_iteration!();
                continue;
            }

            // ---- conditional compares (ccmp / ccmn) --------------------------
            // `ccmp rn, rm|#imm, #nzcv, cond` sets the flags from a comparison
            // of its operands when `cond` holds on the INCOMING flags, and from
            // the `#nzcv` literal otherwise. It is the compiler's
            // compound-condition builder — `if (a == null || b == null)` — so
            // it must WRAP the pending compare rather than clobber it. The old
            // catch-all defined the (unused) destination and cleared
            // `pending_cmp`, so every guarded branch downstream rendered
            // `Compare(kind, Unknown, 0)` — the mass "conditional branch
            // without real compare" wrong-logic class.
            A64Opcode::CCMP | A64Opcode::CCMN => {
                let lhs = operand_at(&insn, 0).unwrap_or(Operand::Unknown);
                let rhs = match insn.operands.get(1) {
                    Some(A64Op::Immediate(v)) => Operand::imm(*v as i64),
                    Some(A64Op::ImmShift(v, _)) => Operand::imm(*v as i64),
                    _ => operand_at(&insn, 1).unwrap_or(Operand::imm(0)),
                };
                let nzcv = match insn.operands.get(2) {
                    Some(A64Op::Immediate(v)) => *v as u8,
                    _ => 0,
                };
                let ccmp_cond = match insn.operands.get(3) {
                    Some(A64Op::ConditionCode(c)) => *c,
                    _ => 0,
                };
                pending_cmp = match pending_cmp.take() {
                    Some(FlagState::Cmp { lhs: pl, rhs: pr, at: prev_at }) => {
                        Some(FlagState::Merge {
                            prev_lhs: pl,
                            prev_rhs: pr,
                            prev_at,
                            ccmp_cond,
                            lhs,
                            rhs,
                            at: insns.len(),
                            nzcv,
                        })
                    }
                    // No incoming compare to guard on (the flags came from a
                    // writer this lifter does not model). Keep the compare
                    // operand so the branch still tests a real value instead
                    // of `__unknown`; the lost `cond` guard is recorded.
                    _ => {
                        super::verify::ledger_note(
                            addr,
                            super::verify::LedgerKind::FoldedBranch,
                        );
                        Some(FlagState::Cmp { lhs, rhs, at: insns.len() })
                    }
                };
                // Reserve the address: a later `cmp`/`b.cond` in the chain can
                // target the ccmp site directly.
                insns.push(Insn::new(Op::Nop, vec![], addr));
                end_iteration!();
                continue;
            }

            // ---- branches ----------------------------------------------------
            A64Opcode::Bcc(code) => {
                let target = branch_target_rva(&insn, addr);
                // NZCV persists until a flag-setting insn rewrites it: an
                // if/else-if chain (`cmp; b.ne L1; b.eq L2`) re-tests the
                // SAME flags with the second branch. `.take()` here used to
                // destroy the pending compare, so the second branch fell
                // back to `Compare(kind, Unknown, 0)` — the mass
                // "wrong logic" class. Read (clone) instead of consume;
                // the next non-branch arm clears it, which is a safe
                // (conservative) approximation of flag clobbering.
                let bcode = code & 0xF;
                // ONE implementation, shared with the select arm below. These
                // used to be two separate copies of the same `Cmp`/`Merge`
                // folding, which meant every fix had to be applied twice and a
                // missed copy silently diverged (the `Merge` reduction here had
                // drifted from the one in `materialize_cond`).
                let s = materialize_cond(
                    &mut insns,
                    &mut addresses,
                    &mut next_scratch,
                    pending_cmp.as_ref(),
                    bcode,
                    addr,
                );
                insns.push(Insn::new(Op::CondJump(target.unwrap_or(0)), vec![s], addr));
            }
            // ---- conditional select ------------------------------------------
            // `dst = cond ? a : transform(b)` — the branchless select the
            // compiler emits for `Math.Min`/`Math.Max`, `?:` against constants,
            // clamps, and EVERY managed `bool` materialisation (`CSET` is
            // `CSINC` with both sources `xzr` → `cond ? 1 : 0`). There was no
            // arm for any of them: all 17,281 occurrences in the reference
            // binary fell through the catch-all and became `dst = Unknown`, so
            // the operation disappeared AND its destination was overwritten
            // with a placeholder.
            A64Opcode::CSEL | A64Opcode::CSINC | A64Opcode::CSINV | A64Opcode::CSNEG
            | A64Opcode::FCSEL => {
                let is_fcsel = insn.opcode == A64Opcode::FCSEL;
                let dst = if is_fcsel {
                    // FCSEL writes a SIMD/FP register, not a GPR.
                    fp_of(&insn.operands[0]).unwrap_or_default()
                } else {
                    gpr_of(&insn.operands[0])?
                };
                // Operand 3 is the condition code the select tests.
                let cond_code = match insn.operands.get(3) {
                    Some(A64Op::ConditionCode(c)) => *c,
                    _ => 0,
                };
                let mut variant = match insn.opcode {
                    A64Opcode::CSINC => SelectVariant::Increment,
                    A64Opcode::CSINV => SelectVariant::Invert,
                    A64Opcode::CSNEG => SelectVariant::Negate,
                    // CSEL and FCSEL are both the plain select.
                    _ => SelectVariant::Identity,
                };
                // Sources come from EITHER register file: `CSEL` selects
                // between GPRs, `FCSEL` between S/D registers. Reading them
                // with the GPR-only reader made both arms of every `FCSEL`
                // `Unknown` (see `value_operand_at`).
                let mut a = value_operand_at(&insn, 1);
                let mut b = value_operand_at(&insn, 2);
                // NOTE: the compare is read from `carried_cmp`, not
                // `pending_cmp`. `CSEL` READS the flags without writing them,
                // so it is listed in `preserves_nzcv`, and the carry-across
                // logic ABOVE the match has therefore already moved the live
                // compare out of `pending_cmp` into `carried_cmp`. Reading
                // `pending_cmp` here would always see `None` and degrade every
                // select back to `cond ? a : b` on an unknown condition.
                let live = carried_cmp.as_ref().or(pending_cmp.as_ref());
                let cond = materialize_cond(
                    &mut insns,
                    &mut addresses,
                    &mut next_scratch,
                    live,
                    cond_code,
                    addr,
                );
                // `CSET` / `CSETM`: both sources are the architectural zero
                // register, so the transform applies to a CONSTANT. Folding the
                // alias here is what makes `CSET` come out as the boolean it
                // actually is instead of `xzr + 1` / `~xzr`.
                //
                // The test is `Immediate(0)`, not `Register("xzr")`: a
                // zero-register READ is the constant zero, and `operand_at`
                // already canonicalises it that way on purpose (see its doc —
                // leaving it a register made every read mint an SSA version of
                // an "xzr" slot that rendered as an undefined local).
                //
                // THE TRUE ARM IS THE ENCODED CONDITION'S FALL-THROUGH, NOT
                // ITS TAKEN SIDE. `CSINC` is `cond ? Xn : Xm + 1`, and both
                // sources are zero here, so this is `cond ? 0 : 1` — never
                // `cond ? 1 : 0`. The subtler half is WHICH condition `cond`
                // is: `cset Wd, <cond>` is the alias of `CSINC Wd, WZR, WZR,
                // invert(<cond>)` (Arm ARM: “CSET … is equivalent to CSINC
                // <Wd>, WZR, WZR, invert(<cond>)”), and the decoder reports the
                // ENCODED condition, not the mnemonic a listing prints. So
                // `cset w3, ne` decodes as `csinc w3, wzr, wzr, eq` and must
                // lift to `eq ? 0 : 1`. Folding it as `cond ? 1 : 0` therefore
                // inverted EVERY managed `bool` the compiler materialises: a
                // recovered `b = x == y;` rendered as `b = x != y;`. `CSETM`
                // (`csinv Wd, WZR, WZR, invert(<cond>)`) has the same shape
                // with `~0 == -1` on the false arm.
                let is_zero = |o: &Operand| matches!(o, Operand::Immediate(0));
                if is_zero(&a) && is_zero(&b) {
                    match variant {
                        // CSET  → cond ? 0 : 1
                        SelectVariant::Increment => {
                            a = Operand::imm(0);
                            b = Operand::imm(1);
                            variant = SelectVariant::Identity;
                        }
                        // CSETM → cond ? 0 : -1
                        SelectVariant::Invert => {
                            a = Operand::imm(0);
                            b = Operand::imm(-1);
                            variant = SelectVariant::Identity;
                        }
                        _ => {}
                    }
                }
                insns.push(Insn::new(
                    Op::Select(variant),
                    vec![Operand::reg(dst), cond, a, b],
                    addr,
                ));
                pending_cmp = None;
            }
            A64Opcode::CBZ | A64Opcode::CBNZ => {
                let target = branch_target_rva(&insn, addr);
                let reg = operand_at(&insn, 0).unwrap_or(Operand::Unknown);
                let kind = if insn.opcode == A64Opcode::CBZ {
                    CompareKind::Equal
                } else {
                    CompareKind::NotEqual
                };
                let s = fresh_scratch!();
                insns.push(Insn::new(Op::Compare(kind), vec![s.clone(), reg, Operand::imm(0)], addr));
                insns.push(Insn::new(Op::CondJump(target.unwrap_or(0)), vec![s], addr));
                pending_cmp = None;
            }
            A64Opcode::TBZ | A64Opcode::TBNZ => {
                let target = branch_target_rva(&insn, addr);
                let reg = operand_at(&insn, 0).unwrap_or(Operand::Unknown);
                // yaxpeax encodes the TBZ/TBNZ bit index as `Imm16`, not
                // `Immediate`/`ImmShift` (armv8/a64.rs: `Operand::Imm16(b)` for
                // the 32-bit form, `Imm16(b | 0x20)` for the 64-bit form). It
                // was missing from this match, so `bit` was ALWAYS 0 and every
                // test-branch was lifted as a bit-0 test:
                //
                //   0x0189dc64  tbnz w24, #0x1f, $+0xbc
                //     ir Arith(And) [v, x24, #1]        <- must be #0x80000000
                //
                // That is why `Initialize` @0x189D9D4 emitted guards such as
                // `if (!index) { if (index >= 1) ... }`: the sign test on the
                // token count became a test of bit 0. Mask off the 0x20 flag
                // that distinguishes the 64-bit form.
                let bit: i64 = match insn.operands.get(1) {
                    Some(A64Op::Imm16(v)) => (*v as i64) & 0x1f,
                    Some(A64Op::Immediate(v)) => *v as i64,
                    Some(A64Op::ImmShift(v, _)) => *v as i64,
                    _ => 0,
                };
                let mask = 1i64 << bit.clamp(0, 63);
                let kind = if insn.opcode == A64Opcode::TBZ {
                    CompareKind::Equal
                } else {
                    CompareKind::NotEqual
                };
                let m = fresh_scratch!();
                let s = fresh_scratch!();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![m.clone(), reg, Operand::imm(mask)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Compare(kind),
                    vec![s.clone(), m, Operand::imm(0)],
                    addr,
                ));
                insns.push(Insn::new(Op::CondJump(target.unwrap_or(0)), vec![s], addr));
                pending_cmp = None;
            }
            A64Opcode::B => {
                let target = branch_target_rva(&insn, addr);
                insns.push(Insn::new(Op::Jump(target.unwrap_or(0)), vec![], addr));
                pending_cmp = None;
            }
            A64Opcode::BR => {
                let reg = gpr_of(&insn.operands[0]).unwrap_or_default();
                insns.push(Insn::new(Op::IndirectCall, vec![Operand::reg(reg)], addr));
                pending_cmp = None;
            }
            A64Opcode::RETAA | A64Opcode::RETAB => {
                // pointer-authenticated returns are returns, not calls
                insns.push(Insn::new(Op::Return, vec![Operand::reg("x0")], addr));
                pending_cmp = None;
            }
            A64Opcode::BL => {
                let target = branch_target_rva(&insn, addr);
                let call_op = if target.is_some() { Op::Call } else { Op::IndirectCall };
                let mut ops = vec![match target {
                    Some(t) => Operand::imm(t as i64),
                    None => gpr_of(&insn.operands[0]).map(|r| Operand::reg(r)).unwrap_or(Operand::Unknown),
                }];
                ops.push(Operand::reg("x0")); // return-value slot
                for r in ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"] {
                    ops.push(Operand::reg(r));
                }
                // AAPCS64 passes FP args in s0–s7/d0–d7. Without recording
                // them as call uses, every FloatArith feeding the call looks
                // dead and DCE deletes the entire computation (the missing
                // Vector3 math). Record only FP regs written since the last
                // call — those are the ones this call actually consumes;
                // unconditional s0–s7 entries would phantom-arg every call.
                // ABI ORDER, not lexicographic order. The set is
                // `BTreeSet<String>`, so `vs40` (s10) sorted BEFORE `vs8`
                // (s2) and the second float argument was bound to the wrong
                // register (or to none). Sort by the flattened index instead.
                let mut fps: Vec<&String> = fp_written.iter().collect();
                fps.sort_by_key(|n| {
                    n.strip_prefix("vs")
                        .and_then(|d| d.parse::<u32>().ok())
                        .unwrap_or(u32::MAX)
                });
                for r in fps {
                    ops.push(Operand::reg(r.clone()));
                }
                fp_written.clear();
                insns.push(Insn::new(call_op, ops, addr));
                pending_cmp = None;
            }
            A64Opcode::BLR => {
                // Ordinary indirect call: virtual dispatch, interface dispatch,
                // delegate / function-pointer invocation.
                //
                // THIS ARM WAS MISSING. `BLR` fell through to the catch-all,
                // which reads operand 0 as a DESTINATION — and for `BLR Xn`
                // operand 0 is the call TARGET register. So it emitted
                // `Xn = Unknown`:
                //   1. the call itself disappeared from the IR entirely, and
                //   2. the register holding the callee was overwritten with
                //      `Unknown`, poisoning every later use of it.
                //
                // Worse, it silently disabled a whole subsystem.
                // `resolve.rs::rewrite_virtual_calls` — its `vtable_slots` map,
                // its `BLR [klass+vtable+slot*psize]` matcher, its
                // `vtable_dispatch_resolves_through_seeded_receiver` test —
                // keys on `Op::Call` (`resolve.rs:215`, `:340`, `:553`). With
                // no `BLR` arm, `Op::Call` was never produced for an indirect
                // call, so that resolver's input never arrived and it could
                // never fire. That is where the resolved virtual calls went,
                // and why the dump was dominated by unresolved `fn_…` thunks.
                //
                // The census put this at the top of the lifter's gaps: 37,655
                // occurrences, 55% of all fall-throughs, the single largest
                // hole in the port.
                //
                // Operand layout matches `BL`: `[target, x0 return slot,
                // x0..x7 args, fp args]` — `locals.rs` reads call args from
                // operand 2, and `dest_position(Op::Call) == Some(1)`.
                let target = gpr_of(&insn.operands[0]).unwrap_or_default();
                let mut ops = vec![Operand::reg(target)];
                ops.push(Operand::reg("x0")); // return-value slot
                for r in ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"] {
                    ops.push(Operand::reg(r));
                }
                // Same AAPCS64 FP-argument rule as `BL`: only the FP registers
                // written since the previous call are arguments of THIS call;
                // unconditional s0–s7 entries would phantom-arg every call.
                // ABI ORDER, not lexicographic order. The set is
                // `BTreeSet<String>`, so `vs40` (s10) sorted BEFORE `vs8`
                // (s2) and the second float argument was bound to the wrong
                // register (or to none). Sort by the flattened index instead.
                let mut fps: Vec<&String> = fp_written.iter().collect();
                fps.sort_by_key(|n| {
                    n.strip_prefix("vs")
                        .and_then(|d| d.parse::<u32>().ok())
                        .unwrap_or(u32::MAX)
                });
                for r in fps {
                    ops.push(Operand::reg(r.clone()));
                }
                fp_written.clear();
                insns.push(Insn::new(Op::Call, ops, addr));
                pending_cmp = None;
            }
            A64Opcode::BLRAA | A64Opcode::BLRAAZ | A64Opcode::BLRAB | A64Opcode::BLRABZ => {
                // Pointer-authenticated indirect call. This RETURNS, so it must
                // be `Op::Call` — modelling a returning call as
                // `Op::IndirectCall` was wrong because `IndirectCall` is a CFG
                // TERMINATOR (`cfg.rs` gives it no successors), which ends the
                // block at the call and makes everything after it unreachable.
                // Same operand layout as `BLR`.
                let target = gpr_of(&insn.operands[0]).unwrap_or_default();
                let mut ops = vec![Operand::reg(target)];
                ops.push(Operand::reg("x0"));
                for r in ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"] {
                    ops.push(Operand::reg(r));
                }
                // ABI ORDER, not lexicographic order. The set is
                // `BTreeSet<String>`, so `vs40` (s10) sorted BEFORE `vs8`
                // (s2) and the second float argument was bound to the wrong
                // register (or to none). Sort by the flattened index instead.
                let mut fps: Vec<&String> = fp_written.iter().collect();
                fps.sort_by_key(|n| {
                    n.strip_prefix("vs")
                        .and_then(|d| d.parse::<u32>().ok())
                        .unwrap_or(u32::MAX)
                });
                for r in fps {
                    ops.push(Operand::reg(r.clone()));
                }
                fp_written.clear();
                insns.push(Insn::new(Op::Call, ops, addr));
                pending_cmp = None;
            }

            // ---- fused multiply-add / wide multiply -----------------------
            A64Opcode::MADD | A64Opcode::MSUB | A64Opcode::SMADDL | A64Opcode::UMADDL
            | A64Opcode::SMSUBL | A64Opcode::UMSUBL => {
                let dst = gpr_of(&insn.operands[0])?;
                let a = operand_at(&insn, 1).unwrap_or(Operand::Unknown);
                let b = operand_at(&insn, 2).unwrap_or(Operand::Unknown);
                let c = operand_at(&insn, 3).unwrap_or(Operand::imm(0));
                let negate = matches!(
                    insn.opcode,
                    A64Opcode::MSUB | A64Opcode::SMSUBL | A64Opcode::UMSUBL
                );
                insns.push(Insn::new(
                    Op::MulAcc { negate_addend: negate, is_float: false },
                    vec![Operand::reg(dst), a, b, c],
                    addr,
                ));
                pending_cmp = None;
            }
            A64Opcode::SMULL | A64Opcode::UMULL => {
                let dst = gpr_of(&insn.operands[0])?;
                let a = operand_at(&insn, 1).unwrap_or(Operand::Unknown);
                let b = operand_at(&insn, 2).unwrap_or(Operand::Unknown);
                push_binary(&mut insns, Op::Arith(ArithKind::Mul), dst, Some(a), Some(b), addr);
                pending_cmp = None;
            }

            // ---- bitfield/shift aliases (UBFM/SBFM/BFM) ---------------------
            // lsr xd, xn, #n  ==  ubfm xn, n, width-1 ; asr == sbfm same shape;
            // lsl == ubfm with imms+1 == immr; ubfx/sbfx are the general forms.
            //
            // ALL of them are exactly `ROR(src, immr) & wmask`, so instead of
            // pattern-matching the two aliases (which left every `ubfx`/`sbfx`
            // as `Unknown` — silently, because an arm existed and the census
            // therefore stayed quiet) DecodeBitMasks supplies the mask and the
            // rotate is materialized. `BFM` is the same with the destination's
            // bits outside the field preserved.
            A64Opcode::UBFM | A64Opcode::SBFM | A64Opcode::BFM => {
                let dst = gpr_of(&insn.operands[0])?;
                let src = operand_at(&insn, 1).unwrap_or(Operand::Unknown);
                let immr = match insn.operands.get(2).cloned() {
                    Some(A64Op::Immediate(v)) => v as u32,
                    _ => 0,
                };
                let imms = match insn.operands.get(3).cloned() {
                    Some(A64Op::Immediate(v)) => v as u32,
                    _ => 63,
                };
                let bits = gpr_bits(&insn.operands[0]);
                let wmask = decode_bit_masks(bits, imms, immr);
                let all_ones = if bits == 64 { u64::MAX } else { 0xffff_ffff };
                let rot = ror_const(&mut insns, &mut virt_serial, src, immr, bits, addr);
                // The extracted field, before any sign extension.
                let field = if wmask == all_ones {
                    rot
                } else {
                    let t = Operand::reg(format!("xt{}", virt_serial));
                    virt_serial += 1;
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::And),
                        vec![t.clone(), rot, Operand::imm(wmask as i64)],
                        addr,
                    ));
                    t
                };
                if matches!(insn.opcode, A64Opcode::BFM) {
                    // (dst & ~wmask) | (ROR(src, immr) & wmask). `~wmask` is a
                    // constant, so it folds to an immediate rather than needing
                    // an IR `Not` on a value.
                    let keep_mask = if bits == 64 {
                        !wmask
                    } else {
                        0xffff_ffffu64 ^ wmask
                    };
                    let keep = Operand::reg(format!("xt{}", virt_serial));
                    virt_serial += 1;
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::And),
                        vec![
                            keep.clone(),
                            Operand::reg(dst.clone()),
                            Operand::imm(keep_mask as i64),
                        ],
                        addr,
                    ));
                    push_binary(
                        &mut insns,
                        Op::Arith(ArithKind::Or),
                        dst,
                        Some(keep),
                        Some(field),
                        addr,
                    );
                } else if matches!(insn.opcode, A64Opcode::SBFM) {
                    // Sign-extend from the top of the field: `(v << k) >> k`
                    // with an arithmetic shift. `Shr` on the signed result is
                    // the arithmetic form; when the field already ends at the
                    // operand top (`imms == bits - 1`) the value is unchanged,
                    // which is the plain `asr` alias.
                    if imms >= bits - 1 {
                        insns.push(Insn::move_(Operand::reg(dst), field, addr));
                    } else {
                        let k = (bits - 1 - imms) as i64;
                        let sh = Operand::reg(format!("xt{}", virt_serial));
                        virt_serial += 1;
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Shl),
                            vec![sh.clone(), field, Operand::imm(k)],
                            addr,
                        ));
                        push_binary(
                            &mut insns,
                            Op::Arith(ArithKind::Shr),
                            dst,
                            Some(sh),
                            Some(Operand::imm(k)),
                            addr,
                        );
                    }
                } else {
                    insns.push(Insn::move_(Operand::reg(dst), field, addr));
                }
                pending_cmp = None;
            }
            A64Opcode::EXTR => {
                // `extr Rd, Rn, Rm, #lsb` == `(Rn:Rm) >> lsb`: the high half
                // comes from Rn, the low half from Rm. Roslyn's IL2CPP emits it
                // for the wide shifts it cannot express as a single LSR/LSL
                // pair (64-bit rotate emulation, `Math.BigMul`-style splitting,
                // hash mixing) — 1020 occurrences, the largest remaining hole.
                let dst = gpr_of(&insn.operands[0])?;
                let hi = operand_at(&insn, 1).unwrap_or(Operand::Unknown);
                let lo = operand_at(&insn, 2).unwrap_or(Operand::Unknown);
                let lsb = match insn.operands.get(3) {
                    Some(A64Op::Immediate(v)) => *v as u32,
                    _ => 0,
                };
                let bits = gpr_bits(&insn.operands[0]);
                if lsb == 0 {
                    insns.push(Insn::move_(Operand::reg(dst), hi, addr));
                } else if lsb >= bits {
                    insns.push(Insn::move_(Operand::reg(dst), lo, addr));
                } else {
                    let a = Operand::reg(format!("xt{}", virt_serial));
                    virt_serial += 1;
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shr),
                        vec![a.clone(), hi, Operand::imm(lsb as i64)],
                        addr,
                    ));
                    let b = Operand::reg(format!("xt{}", virt_serial));
                    virt_serial += 1;
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Shl),
                        vec![b.clone(), lo, Operand::imm((bits - lsb) as i64)],
                        addr,
                    ));
                    push_binary(
                        &mut insns,
                        Op::Arith(ArithKind::Or),
                        dst,
                        Some(a),
                        Some(b),
                        addr,
                    );
                }
                pending_cmp = None;
            }
            A64Opcode::ASRV => {
                let dst = gpr_of(&insn.operands[0])?;
                let a = operand_at(&insn, 1).unwrap_or(Operand::Unknown);
                let b = operand_at(&insn, 2).unwrap_or(Operand::Unknown);
                // `asrv` is the arithmetic (sign-propagating) right shift. The
                // IR has one `Shr`; on a signed operand it *is* the arithmetic
                // shift, which is how the type propagation types these.
                push_binary(&mut insns, Op::Arith(ArithKind::Shr), dst, Some(a), Some(b), addr);
                pending_cmp = None;
            }
            A64Opcode::RORV => {
                let dst = gpr_of(&insn.operands[0])?;
                let a = operand_at(&insn, 1).unwrap_or(Operand::Unknown);
                let b = operand_at(&insn, 2).unwrap_or(Operand::Unknown);
                push_binary(&mut insns, Op::Arith(ArithKind::Ror), dst, Some(a), Some(b), addr);
                pending_cmp = None;
            }
            A64Opcode::CLZ => {
                // [dest, src] — CountLeadingZeros reads operand[1]
                let dst = gpr_of(&insn.operands[0])?;
                let src = operand_at(&insn, 1).unwrap_or(Operand::Unknown);
                insns.push(Insn::new(Op::Clz, vec![Operand::reg(dst), src], addr));
                pending_cmp = None;
            }
            A64Opcode::SCVTF | A64Opcode::UCVTF => {
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                // The source is an INTEGER value that may live in EITHER
                // register file. `operand_at` names GPRs only, so
                // `scvtf s0, s0` (reinterpreting a 4-byte load as an int and
                // converting it to float — `(float)token.Count`) produced
                // Convert[.., Unknown] and the whole numerator vanished.
                let src = value_operand_at(&insn, 1);
                let conv = if dst.starts_with("vd") {
                    super::ir::Conversion::IntToFloat64
                } else {
                    super::ir::Conversion::IntToFloat32
                };
                insns.push(Insn::new(
                    Op::Convert(conv),
                    vec![Operand::reg(dst), src],
                    addr,
                ));
                pending_cmp = None;
            }
            A64Opcode::FCVTZS | A64Opcode::FCVTZU => {
                let dst = gpr_of(&insn.operands[0])?;
                let src = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .map(Operand::reg)
                    .unwrap_or(Operand::Unknown);
                insns.push(Insn::new(
                    Op::Convert(super::ir::Conversion::FloatToInt),
                    vec![Operand::reg(dst), src],
                    addr,
                ));
                pending_cmp = None;
            }
            A64Opcode::FCVT => {
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let src = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .map(Operand::reg)
                    .unwrap_or(Operand::Unknown);
                let conv = if dst.starts_with("vd") {
                    super::ir::Conversion::Float32ToFloat64
                } else {
                    super::ir::Conversion::Float64ToFloat32
                };
                insns.push(Insn::new(
                    Op::Convert(conv),
                    vec![Operand::reg(dst), src],
                    addr,
                ));
                pending_cmp = None;
            }
            A64Opcode::FMAX | A64Opcode::FMIN | A64Opcode::FMAXNM | A64Opcode::FMINNM
                if is_fp_operand(&insn.operands[0]) =>
            {
                // A SELECT, not an assignment. This arm used to build the
                // compare and then unconditionally assign the FIRST operand,
                // dropping the comparison on the floor — every floating
                // max/min came out as `x = a`, which is why recovered
                // float-heavy maths looked like it had lost its logic. With
                // `Op::Select` in the IR the condition survives into the
                // rendered ternary.
                //
                // `fmaxnm`/`fminnm` are the IEEE-754 `maxNum`/`minNum` forms;
                // they differ from `fmax`/`fmin` only in NaN propagation,
                // which pseudo-C# has no spelling for — the select is the
                // recovered shape either way.
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let a = fp_value_at(&insn, 1);
                let b = fp_value_at(&insn, 2);
                let kind = match insn.opcode {
                    A64Opcode::FMAX | A64Opcode::FMAXNM => CompareKind::GreaterThan,
                    _ => CompareKind::LessThan,
                };
                let s = fresh_scratch!();
                insns.push(Insn::new(
                    Op::Compare(kind),
                    vec![s.clone(), a.clone(), b.clone()],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Select(SelectVariant::Identity),
                    vec![Operand::reg(dst.clone()), s, a, b],
                    addr,
                ));
                fp_written.insert(dst);
                pending_cmp = None;
            }
            // ---- FP rounding / difference --------------------------------
            // `frint*` is how .NET's Math.Round / Math.Floor / Math.Ceiling /
            // Math.Truncate reach the CPU. Without an arm each one fell to the
            // catch-all and the rounded value became `Unknown` — 260 `FRINTM`
            // + 228 `FRINTP` in the census. Rendered as named intrinsics, the
            // same convention `Sqrt`/`Abs`/`CountLeadingZeros` already use.
            //
            // NOTE: FRINTA (ties away from zero — what `Math.Round` does) and
            // FRINTN (ties to even — the default FP pipeline) both map to
            // `Round`; which tie rule the encoder chose is not recoverable
            // from the operand.
            A64Opcode::FRINTN | A64Opcode::FRINTP | A64Opcode::FRINTM | A64Opcode::FRINTZ
            | A64Opcode::FRINTA
                if is_fp_operand(&insn.operands[0]) =>
            {
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let src = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                let kind = match insn.opcode {
                    A64Opcode::FRINTM => FloatArithKind::Floor,
                    A64Opcode::FRINTP => FloatArithKind::Ceiling,
                    A64Opcode::FRINTZ => FloatArithKind::Truncate,
                    _ => FloatArithKind::Round,
                };
                insns.push(Insn::new(
                    Op::FloatArith(kind),
                    vec![Operand::reg(dst.clone()), Operand::reg(src)],
                    addr,
                ));
                fp_written.insert(dst);
                pending_cmp = None;
            }
            A64Opcode::FABD if is_fp_operand(&insn.operands[0]) => {
                // |a - b|. Computed through the destination register itself
                // (`d0 = d1 - d2; d0 = Abs(d0)`) so no extra scratch slot is
                // needed; SSA renaming keeps the two writes distinct.
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let a = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                let b = fp_of(&insn.operands.get(2).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                insns.push(Insn::new(
                    Op::FloatArith(FloatArithKind::Sub),
                    vec![Operand::reg(dst.clone()), Operand::reg(a), Operand::reg(b)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::FloatArith(FloatArithKind::Abs),
                    vec![Operand::reg(dst.clone()), Operand::reg(dst.clone())],
                    addr,
                ));
                fp_written.insert(dst);
                pending_cmp = None;
            }
            A64Opcode::FNMUL if is_fp_operand(&insn.operands[0]) => {
                // `-(a * b)` — the negated-multiply form, again through the
                // destination register.
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let a = fp_of(&insn.operands.get(1).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                let b = fp_of(&insn.operands.get(2).copied().unwrap_or(A64Op::Nothing))
                    .unwrap_or_default();
                insns.push(Insn::new(
                    Op::FloatArith(FloatArithKind::Mul),
                    vec![Operand::reg(dst.clone()), Operand::reg(a), Operand::reg(b)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::FloatArith(FloatArithKind::Neg),
                    vec![Operand::reg(dst.clone()), Operand::reg(dst.clone())],
                    addr,
                ));
                fp_written.insert(dst);
                pending_cmp = None;
            }
            // ---- float→int with an explicit rounding direction ------------
            // `fcvtms/fcvtps/fcvtns/fcvtas` (and the unsigned `…u` forms) are
            // the conversions `(int)Math.Floor(x)` / `Math.Ceiling` /
            // `Math.Round` compile to. Only the truncating `fcvtzs`/`fcvtzu`
            // had an arm, so every rounded conversion produced `Unknown` —
            // 43 `FCVTMS` + 41 `FCVTPS` and the rest of the family.
            A64Opcode::FCVTMS | A64Opcode::FCVTMU | A64Opcode::FCVTPS | A64Opcode::FCVTPU
            | A64Opcode::FCVTNS | A64Opcode::FCVTNU | A64Opcode::FCVTAS | A64Opcode::FCVTAU
                // The vector forms (`fcvtms v0.2d, v1.2d`) have no GPR
                // destination; without this guard `gpr_of` below returns
                // `None` and the `?` aborts the ENTIRE method lift, turning a
                // single missing arm into a missing function.
                if matches!(&insn.operands[0], A64Op::Register(_, _)) =>
            {
                let dst = gpr_of(&insn.operands[0])?;
                let src = fp_value_at(&insn, 1);
                let round = match insn.opcode {
                    A64Opcode::FCVTMS | A64Opcode::FCVTMU => FloatArithKind::Floor,
                    A64Opcode::FCVTPS | A64Opcode::FCVTPU => FloatArithKind::Ceiling,
                    _ => FloatArithKind::Round,
                };
                let t = Operand::reg(format!("vt{}", virt_serial));
                virt_serial += 1;
                insns.push(Insn::new(Op::FloatArith(round), vec![t.clone(), src], addr));
                insns.push(Insn::new(
                    Op::Convert(super::ir::Conversion::FloatToInt),
                    vec![Operand::reg(dst), t],
                    addr,
                ));
                pending_cmp = None;
            }
            // Scalar-only: the vector forms (`fmla v0.4s, …`) are a different
            // shape and must not abort the whole method lift through the `?`
            // (or, worse, define `Register("")` as four lanes' worth of
            // bogus destinations — the un-guarded arms below did exactly that).
            A64Opcode::FMADD | A64Opcode::FMSUB | A64Opcode::FNMADD | A64Opcode::FNMSUB
                if is_fp_operand(&insn.operands[0]) =>
            {
                let dst = fp_of(&insn.operands[0]).unwrap_or_default();
                let a = fp_value_at(&insn, 1);
                let b = fp_value_at(&insn, 2);
                let c = fp_value_at(&insn, 3);
                let negate = matches!(insn.opcode, A64Opcode::FMSUB | A64Opcode::FNMSUB);
                insns.push(Insn::new(
                    Op::MulAcc { negate_addend: negate, is_float: true },
                    vec![Operand::reg(dst), a, b, c],
                    addr,
                ));
                pending_cmp = None;
            }


            // ---- returns -------------------------------------------------------
            A64Opcode::RET => {
                // RET's operand names the jump target (LR); managed ABI returns x0
                insns.push(Insn::new(Op::Return, vec![Operand::reg("x0")], addr));
                pending_cmp = None;
            }

            // ---- generic value-producing fallback ----------------------------
            //
            // Every opcode with a hand-written arm above is consumed before
            // this point. What is left is the long tail of vector/SIMD and
            // rare scalar ops (`MOVI`, `DUP`, `UMOV`, the `REV`/`RBIT` family,
            // `UMINV`/`ADDV`/`UMAXV` reductions, `TBL`/`EXT`/`ZIP`/`UZP`,
            // `LD2`/`LD3`/`LD4`, widening/narrowing pairs, `UMULH`/`SMULH`,
            // `ADC`, `MRS`, ...).
            //
            // The rule this arm encodes: **an instruction that produces a
            // value must define its destination.** Reaching here with a
            // register destination means the semantics are not modelled yet,
            // but the destination and every source are KNOWN — so emit the
            // opcode's own mnemonic as a named pseudo-intrinsic with all
            // operands preserved. `v12 = __movi(...)` is a name we do not have
            // yet; `v12 = Unknown` (the old behaviour, and for a vector
            // destination, no statement at all) is a value that is gone, and
            // every later read of that register silently fabricated something
            // else. That is the same contract `fnptr_0x…` uses for an
            // unresolved call, and the same reason it mattered there.
            //
            // Instructions with NO register destination (DSB/DMB/ISB/CLREX/
            // PRFM/SB/HINT/BRK — 4,500+ in this binary, all flag-and-barrier
            // lore with no produced value) still fall through to the census
            // arm below and are correctly ignored.
            _ if generic_intrinsic_dst(&insn).is_some() => {
                let name = format!("{:?}", insn.opcode);
                census_intrinsic(insn.opcode);
                let dst = generic_intrinsic_dst(&insn).unwrap();
                let mut ops = vec![Operand::reg(dst)];
                // For MOVI, preserve the element size as a third operand so the
                // renderer can expand the immediate to the full 64-bit value.
                if insn.opcode == A64Opcode::MOVI {
                    if let Some(A64Op::SIMDRegisterElements(_, _, esz)) = insn.operands.first() {
                        let lane_bytes: i64 = match esz {
                            SIMDSizeCode::B => 1,
                            SIMDSizeCode::H => 2,
                            SIMDSizeCode::S => 4,
                            SIMDSizeCode::D => 8,
                            _ => 0,
                        };
                        if lane_bytes > 0 {
                            ops.push(Operand::imm(lane_bytes));
                        }
                    }
                }
                for k in 1..insn.operands.len() {
                    if matches!(insn.operands[k], A64Op::Nothing) {
                        break;
                    }
                    ops.push(any_operand_at(&insn, k));
                }
                insns.push(Insn::new(Op::Simd(name), ops, addr));
                pending_cmp = None;
            }

            // ---- unhandled opcodes ------------------------------------------------
            //
            // The fall-through. Every AArch64 opcode without an explicit arm
            // above lands here and is reduced to `dst = Unknown` (or, with no
            // register destination, is dropped entirely). That is a fidelity
            // loss, not a neutral no-op, so record it in the census: without
            // it the only trace of a missing arm is an unexplained `Unknown`
            // thousands of statements later, which is indistinguishable from a
            // resolver bug. `effectful` separates a genuine lost value from an
            // instruction that is legitimately ignorable (NOP/DSB/ISB/MSR).
            _ => {
                let word = bytes
                    .get((addr - base_rva) as usize..)
                    .and_then(|s| s.get(..4))
                    .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
                    .unwrap_or(0);
                let gpr_dst = insn.operands.first().and_then(gpr_of);
                let fp_dst = insn.operands.first().and_then(|o| match o {
                    A64Op::SIMDRegister(sz, n) => Some(fp_reg(*sz, *n)),
                    _ => None,
                });
                census_unhandled(insn.opcode, addr, word, gpr_dst.is_some() || fp_dst.is_some());
                if let Some(d) = gpr_dst {
                    insns.push(Insn::move_(Operand::reg(d), Operand::Unknown, addr));
                } else if let Some(d) = fp_dst {
                    insns.push(Insn::move_(Operand::reg(d), Operand::Unknown, addr));
                }
                pending_cmp = None;
            }
        }

        end_iteration!();
    }

    // The sret pointer is a CALL argument the `BL` arm cannot see (it lives in
    // x8, outside the x0–x7 dump); attach it before any consumer reads the
    // operand list.
    attach_sret_register(&mut insns);

    if !insns.last().is_some_and(|i| i.is_terminator()) {
        let tail_addr = base_rva + offset;
        insns.push(Insn::new(Op::Return, vec![Operand::reg("x0")], tail_addr));
        addresses.push(tail_addr);
    }

    Some(LiftResult { insns, addresses })
}

// helpers ---------------------------------------------------------------------

/// Smallest legal lifter scratch local id. Scratch ids are allocated DOWNWARD
/// from `u32::MAX - 2`; every id >= this floor is lifter-internal and can
/// never collide with a real method local (metadata version ids are small).
pub const SCRATCH_FLOOR: u32 = 0xF000_0000;

/// True for the lifter's internal scratch local ids (see [`SCRATCH_FLOOR`]),
/// including the two legacy shared sentinels (`u32::MAX - 1` / `u32::MAX`).
pub fn is_scratch(v: u32) -> bool {
    v >= SCRATCH_FLOOR
}

fn is_fp_operand(op: &A64Op) -> bool {
    matches!(
        op,
        A64Op::SIMDRegister(SIMDSizeCode::S | SIMDSizeCode::D, _)
    )
}

fn fp_of(op: &A64Op) -> Option<String> {
    match op {
        A64Op::SIMDRegister(sz @ (SIMDSizeCode::S | SIMDSizeCode::D), n) => Some(fp_reg(*sz, *n)),
        _ => None,
    }
}

/// Vector SIMD register of 32-bit float elements: `Vn.2S` (64-bit container,
/// 2 lanes) or `Vn.4S` (128-bit, 4 lanes) → (register number, lane count).
fn vec_fp2s(op: &A64Op) -> Option<(u16, u8)> {
    match op {
        A64Op::SIMDRegisterElements(SIMDSizeCode::D, n, SIMDSizeCode::S) => Some((*n, 2)),
        A64Op::SIMDRegisterElements(SIMDSizeCode::Q, n, SIMDSizeCode::S) => Some((*n, 4)),
        _ => None,
    }
}

/// Name of a scalar S-register: direct mapping `sN → vsN`.
fn scalar_s_name(num: u16) -> String {
    // The scalar `sN` view IS lane 0 of `vN`, and vector lanes are named with
    // the flattened identity `vs{4*reg + lane}` (`s_lane_name`, and
    // `highvar`'s ABI-supplied check). Returning `vs{num}` here gave the SAME
    // architectural register two names: `fsub s8` defined/used `vs8`, while
    // `mov v8.16b, v0.16b` defined `vs32` (lane 0 of v8). The copy was then
    // dead, the loop's roll lost its definition, and the emitted code read an
    // ABI entry slot (`in_vs8`) instead of the Range() result (0x18A53C4).
    // It also aliased `s8` with `v2.s[0]` (`vs8`), two unrelated values.
    format!("vs{}", num as u32 * 4)
}

/// Name of a 32-bit SIMD lane. Scalar `sN` is represented as lane 0 of `vN`,
/// so scalar and vector operations share a name only when they access the same
/// architectural bits.
fn s_lane_name(vreg: u16, lane: u8) -> String {
    format!("vs{}", vreg as u32 * 4 + lane as u32)
}

/// Source operand of one lane of a vector FP op: a same-shaped vector
/// contributes its own lane k; an element-lane operand (vector-by-scalar
/// forms like `fmul v.2s, v.2s, v.s[1]`) broadcasts its single element.
fn vec_src_lane(op: &A64Op, k: u8) -> String {
    match op {
        A64Op::SIMDRegisterElements(_, n, SIMDSizeCode::S) => s_lane_name(*n, k),
        A64Op::SIMDRegisterElementsLane(_, n, SIMDSizeCode::S, lane) => s_lane_name(*n, *lane),
        _ => String::new(),
    }
}

/// LDR Xt/Xd, <label> literal-pool form: PCOffset operand + readable bytes.
fn try_literal_pool_load(
    insn: &A64Insn,
    addr: u64,
    _bytes: &[u8],
    _base_rva: u64,
    _offset: u64,
    pool_reader: Option<&PoolReader<'_>>,
) -> Option<Operand> {
    let pc_off = insn.operands.iter().find_map(|op| match op {
        A64Op::PCOffset(off) => Some(*off),
        _ => None,
    })?;
    let pool_addr = (addr as i64 + pc_off) as u64;
    let reader = pool_reader?;
    let data = reader(pool_addr)?;
    let dst_reg = fp_of(&insn.operands[0])?;
    if dst_reg.starts_with("vs") && data.len() >= 4 {
        let f = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        Some(Operand::FloatLiteral(f))
    } else if data.len() >= 8 {
        let d = f64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        Some(Operand::DoubleLiteral(d))
    } else {
        None
    }
}

fn gpr_of(op: &A64Op) -> Option<String> {
    match op {
        A64Op::Register(_, r) => Some(gpr(*r, false)),
        A64Op::RegisterOrSP(_, r) => Some(gpr(*r, true)),
        _ => None,
    }
}

/// SIMD pair operand of `STP/STNP d1, d2` / `LDP` (S-, D- or Q-width):
/// `(reg1, reg2, element_size_bytes)`. Pair spills of float registers are how
/// the compiler passes computed floats into tail-merged call blocks.
///
/// Q pairs are the 128-bit VECTOR spill form (`stp q0, q1, [sp, #32]`), and
/// they were missing here. `fp_of` answers only for the scalar `S`/`D` views,
/// so a Q pair supplied no operand at all and the STP/LDP arms emitted NOTHING
/// — the instruction vanished from the IR with no diagnostic and the coverage
/// oracle reported it uncovered. Measured on the arm64 corpus: `stp` 5,922 +
/// `ldp` 2,792 uncovered instructions, the largest arm64 coverage class by an
/// order of magnitude, and every one is a real spill/reload whose loss severs
/// the def→use chain for whatever the vector held.
fn fp_pair_of(insn: &A64Insn) -> Option<(String, String, u8)> {
    let (n1, esz) = match insn.operands.first()? {
        A64Op::SIMDRegister(SIMDSizeCode::S, n) => (fp_reg(SIMDSizeCode::S, *n), 4u8),
        A64Op::SIMDRegister(SIMDSizeCode::D, n) => (fp_reg(SIMDSizeCode::D, *n), 8u8),
        A64Op::SIMDRegister(SIMDSizeCode::Q, n) => (fp_reg(SIMDSizeCode::Q, *n), 16u8),
        _ => return None,
    };
    let n2 = match insn.operands.get(1)? {
        A64Op::SIMDRegister(SIMDSizeCode::S, n) => fp_reg(SIMDSizeCode::S, *n),
        A64Op::SIMDRegister(SIMDSizeCode::D, n) => fp_reg(SIMDSizeCode::D, *n),
        A64Op::SIMDRegister(SIMDSizeCode::Q, n) => fp_reg(SIMDSizeCode::Q, *n),
        _ => return None,
    };
    Some((n1, n2, esz))
}

/// A register NAME for any SIMD/SIMD-vector operand shape.
///
/// `fp_of` deliberately answers only for the *scalar* `S`/`D` forms the scalar
/// FP pipeline works in. Vector shapes have no name at all under it, which is
/// why every opcode whose operands are vectors looked like it had no operands:
/// `fcsel` (a scalar FP select with SIMD sources) and the whole `MOVI`/`DUP`/
/// `TBL`/reduction family. Naming the container makes those values exist.
///
/// A 32-bit LANE is named by its scalar `s` alias (`v10.s[3]` IS `s13`) — the
/// same identity the scalar vector-FP decomposition already uses — so a lane
/// write and a scalar read are one value, not two.
fn simd_name(op: &A64Op) -> Option<String> {
    match op {
        A64Op::SIMDRegister(c, n) => Some(fp_reg(*c, *n)),
        A64Op::SIMDRegisterElements(c, n, _) => Some(fp_reg(*c, *n)),
        A64Op::SIMDRegisterElementsLane(_c, n, SIMDSizeCode::S, lane) => {
            Some(s_lane_name(*n, *lane))
        }
        A64Op::SIMDRegisterElementsLane(c, n, _, lane) => {
            Some(format!("{}[{lane}]", fp_reg(*c, *n)))
        }
        A64Op::SIMDRegisterElementsMultipleLane(c, n, esz, lane, _) => {
            if *esz == SIMDSizeCode::S {
                Some(s_lane_name(*n, *lane))
            } else {
                Some(format!("{}[{lane}]", fp_reg(*c, *n)))
            }
        }
        A64Op::SIMDRegisterGroup(c, n, _, _) => Some(fp_reg(*c, *n)),
        A64Op::SIMDRegisterGroupLane(n, c, _, _) => Some(fp_reg(*c, *n)),
        _ => None,
    }
}

/// `(physical V register, byte width)` for a full-width SIMD value. Used when
/// a bitwise MOV alias copies a Q/D register and scalar instructions later read
/// one of its S/D views.
fn simd_copy_container(op: &A64Op) -> Option<(u16, u8)> {
    match op {
        A64Op::SIMDRegister(SIMDSizeCode::D, n)
        | A64Op::SIMDRegisterElements(SIMDSizeCode::D, n, _) => Some((*n, 8)),
        A64Op::SIMDRegister(SIMDSizeCode::Q, n)
        | A64Op::SIMDRegisterElements(SIMDSizeCode::Q, n, _) => Some((*n, 16)),
        _ => None,
    }
}

/// Destination register NAME of an instruction the generic intrinsic arm must
/// define: any GPR, or any SIMD shape. `None` means the instruction produces
/// no architectural value and may be dropped.
fn generic_intrinsic_dst(insn: &A64Insn) -> Option<String> {
    let first = insn.operands.first()?;
    if let Some(g) = gpr_of(first) {
        // a destination of `xzr`/`wzr` discards the result — there is no value
        if g == "xzr" || g == "wzr" {
            return None;
        }
        return Some(g);
    }
    simd_name(first)
}

/// Read operand `i` as a VALUE from either register file, plus the immediate
/// forms the SIMD encodings use.
fn any_operand_at(insn: &A64Insn, i: usize) -> Operand {
    if let Some(v) = operand_at(insn, i) {
        return v;
    }
    match insn.operands.get(i) {
        Some(A64Op::SIMDRegister(_, _)) => fp_value_at(insn, i),
        Some(op) => simd_name(op).map(Operand::reg).unwrap_or_else(|| {
            if let Some(A64Op::ImmediateDouble(d)) = insn.operands.get(i) {
                Operand::DoubleLiteral(*d)
            } else {
                Operand::Unknown
            }
        }),
        None => Operand::Unknown,
    }
}

/// SIMD/FP register operand `i` as a VALUE. Unlike `fp_of(…).unwrap_or_default()`
/// — which yields the EMPTY string and therefore `Register("")`, a register that
/// does not exist and that the renderer can only report as an unsupported
/// operand — a non-SIMD operand becomes `Unknown`, the honest answer.
fn fp_value_at(insn: &A64Insn, i: usize) -> Operand {
    simd_name(&insn.operands.get(i).cloned().unwrap_or(A64Op::Nothing))
        .map(|r| Operand::reg(r))
        .unwrap_or(Operand::Unknown)
}

/// [`value_operand_at`] but distinguishing "no such operand" from "the operand
/// is a SIMD register we could not name" — callers that must not silently emit
/// a store of `Unknown` need the difference.
fn value_operand_at_opt(insn: &A64Insn, i: usize) -> Option<Operand> {
    if let Some(v) = operand_at(insn, i) {
        return Some(v);
    }
    if matches!(insn.operands.get(i), Some(A64Op::SIMDRegister(_, _))) {
        return fp_of(&insn.operands[i]).map(Operand::reg);
    }
    simd_name(&insn.operands.get(i).cloned().unwrap_or(A64Op::Nothing)).map(Operand::reg)
}

/// Read operand `i` as a value from EITHER register file.
///
/// `operand_at` understands only GPR operands. `FCSEL` — the floating-point
/// member of the conditional-select family — takes its two value sources from
/// the FP register file, so reading them through the GPR-only path returned
/// `None` and BOTH arms of the select became `Unknown`. On real bytes:
///
/// ```text
/// 0x00014534  fcsel s0, s12, s2, gt
/// ```
/// lifted to `Select [vs0, Local(scratch), Unknown, Unknown]` and rendered
/// `t4294967293 ? __unknown : __unknown` — the select's entire payload,
/// including which of the two floats is chosen, deleted (943 sites).
fn value_operand_at(insn: &A64Insn, i: usize) -> Operand {
    if let Some(v) = operand_at(insn, i) {
        return v;
    }
    fp_value_at(insn, i)
}

fn operand_at(insn: &A64Insn, i: usize) -> Option<Operand> {
    match insn.operands.get(i)?.clone() {
        op @ (A64Op::Register(_, _) | A64Op::RegisterOrSP(_, _)) => {
            // XZR reads are the architectural ZERO — an immediate, not a
            // register. Leaving `Register("xzr")` made every read create an
            // SSA entry version of the "xzr" slot that renders as an
            // undefined `v16` local (`v16 = *(v15 + 8)` garbage).
            if gpr_of(&op).as_deref() == Some("xzr") {
                return Some(Operand::imm(0));
            }
            gpr_of(&op).map(Operand::reg)
        }
        A64Op::Immediate(v) => Some(Operand::imm(v as i64)),
        A64Op::ImmShift(v, _) => Some(Operand::imm(v as i64)),
        A64Op::ImmShiftMSL(v, _) => Some(Operand::imm(v as i64)),
        A64Op::Imm64(v) => Some(Operand::imm(v as i64)),
        A64Op::Imm16(v) => Some(Operand::imm(v as i64)),
        _ => None,
    }
}

/// Decode an A64 shifted/extended-register SOURCE operand (`add x0,x1,x2,lsl#3`,
/// `orr x9,x9,x8,lsr#2`, `add x0,x1,w2,uxtw#2`). Returns (base register name,
/// scaling kind, shift amount). Shift-by-zero and the pure extend styles (which
/// only widen a register — already 64-bit in our model) return amount 0 so
/// callers can treat them as a plain register read.
///
/// Previously `RegShift` was only special-cased inside the `MOV` alias, and
/// `operand_at` returned None for every shifted source — so the operand
/// silently VANISHED from the binary op (`add x10,x10,w11,lsl#2` degraded to
/// `x10 = x10 + x10`). That broke scaled pointer arithmetic AND LLVM jump-
/// table dispatch, where the case index is `(loaded halfword) << 2`.
fn shifted_reg_src(insn: &A64Insn, idx: usize) -> Option<(String, ArithKind, u8)> {
    let A64Op::RegShift(style, amt, _, reg) = insn.operands.get(idx)?.clone() else {
        return None;
    };
    if amt == 0 {
        return Some((gpr(reg, false), ArithKind::Shl, 0));
    }
    let kind = match style {
        ShiftStyle::LSL => ArithKind::Shl,
        ShiftStyle::LSR => ArithKind::Shr,
        ShiftStyle::ASR => ArithKind::Shr,
        ShiftStyle::ROR => ArithKind::Ror,
        // extend styles widen 32/16/8-bit regs; offset arithmetic uses
        // non-negative values, so identity + trailing shift is exact there.
        ShiftStyle::UXTB
        | ShiftStyle::UXTH
        | ShiftStyle::UXTW
        | ShiftStyle::UXTX
        | ShiftStyle::SXTB
        | ShiftStyle::SXTH
        | ShiftStyle::SXTW
        | ShiftStyle::SXTX => ArithKind::Shl,
    };
    Some((gpr(reg, false), kind, amt))
}

fn memory_operand(op: Option<&A64Op>) -> Option<Operand> {
    match op? {
        op @ (A64Op::Register(_, _) | A64Op::RegisterOrSP(_, _)) => {
            gpr_of(op).map(|b| Operand::Memory {
                width: 0,
                base: Box::new(Operand::reg(b)),
                index: None,
                scale: 8,
                addend: 0,
            })
        }
        A64Op::RegPreIndex(base, off, _) => Some(Operand::Memory {
            width: 0,
            base: Box::new(Operand::reg(gpr(*base, true))),
            index: None,
            scale: 8,
            addend: *off as i64,
        }),
        // Post-index (`ldr x0, [x1], #8` / `[x1], x9`): the ACCESS is at the
        // UNCHANGED base — the offset is only the increment, which
        // `memory_writeback` deliberately does not model yet (see the note
        // there). Reading it as an access offset made every element copy loop
        // load eight bytes past the element it meant.
        A64Op::RegPostIndex(base, _) | A64Op::RegPostIndexReg(base, _) => Some(Operand::Memory {
            width: 0,
            base: Box::new(Operand::reg(gpr(*base, true))),
            index: None,
            scale: 8,
            addend: 0,
        }),
        A64Op::RegRegOffset(base, index, _, _, shift) => Some(Operand::Memory {
            width: 0,
            base: Box::new(Operand::reg(gpr(*base, true))),
            index: Some(Box::new(Operand::reg(gpr(*index, false)))),
            scale: 1u8 << (*shift).min(3),
            addend: 0,
        }),
        _ => None,
    }
}

/// The write-back a PRE-INDEX addressing form performs on its base register,
/// as `(base register, optional index register, byte addend)`.
///
/// ARM64 folds an address update into the access itself: `ldr x0, [x1, #8]!`
/// (`RegPreIndex(.., true)`) writes `x1 + 8` back into `x1`. The lifter
/// modelled only the access, so the base kept its pre-update value and every
/// later read of it used the wrong pointer.
///
/// That is what turned a nested-struct receiver into the enclosing object:
/// in `XmlTextReaderImpl.ParseDoctypeDecl` @0x1F3F0F8
///     0x1f3f1d0  mov x20, x19            ; x20 = this
///     0x1f3f1d4  ldr x22, [x20, #0x28]!  ; x20 = &this->ps;  x22 = ps.chars
///     0x1f3f274  mov x0, x20             ; receiver := &this->ps
///     0x1f3f27c  bl  ParsingState.get_LinePos
/// Without the write-back x20 still held `this`, so the receiver collapsed
/// onto the x0 entry value and `ps.LinePos` rendered as `this.LinePos` — a
/// member `XmlTextReaderImpl` does not even declare.
///
/// POST-index (`ldr x0, [x1], #8`) updates the base AFTER the access: `[x1]`
/// is read at the UNCHANGED base, then `x1 += 8`. `memory_operand` reads the
/// element at addend 0, so without the update the base kept its pre-increment
/// value and every element-copy/iterate loop re-read the SAME element without
/// advancing — a real wrong-logic defect, and the second-largest arm64
/// wrong-logic class (981 `memory offset mismatch: native base19 +8, IR [0]`
/// sem-diffs on the iOS corpus, all of them post-index `ldr`/`str` pairs).
///
/// This was previously deferred because the extra definition makes the base
/// multi-def; see the structurer note where the write-back is emitted.
fn memory_writeback(op: Option<&A64Op>) -> Option<(String, Option<String>, i64)> {
    match op? {
        A64Op::RegPreIndex(base, off, true) => Some((gpr(*base, true), None, *off as i64)),
        A64Op::RegPostIndex(base, off) => Some((gpr(*base, true), None, *off as i64)),
        A64Op::RegPostIndexReg(base, idx) => Some((gpr(*base, true), Some(gpr(*idx, false)), 0)),
        _ => None,
    }
}

/// Emit the `base = base + addend/index` half of an addressing form.
///
/// The access itself is emitted first so its base operand still names the
/// OLD register value; this move then defines the new one. The callers emit
/// it whether or not the access produced IR (an unknown stored value drops
/// the store but the native write-back still happens).
fn push_writeback(insns: &mut Vec<Insn>, wb: Option<(String, Option<String>, i64)>, addr: u64) {
    let Some((base, index, addend)) = wb else {
        return;
    };
    let (kind, rhs) = match index {
        Some(idx) => (ArithKind::Add, Operand::reg(idx)),
        None if addend > 0 => (ArithKind::Add, Operand::imm(addend)),
        None if addend < 0 => (ArithKind::Sub, Operand::imm(-addend)),
        None => return,
    };
    push_binary(
        insns,
        Op::Arith(kind),
        base.clone(),
        Some(Operand::reg(base)),
        Some(rhs),
        addr,
    );
}

/// Same memory operand offset by `byte_add` bytes (lane k of a vector
/// load/store: [x] → [x + 4k]). Non-memory operands degrade to Unknown.
fn mem_plus(op: &Operand, byte_add: i64) -> Operand {
    match op {
        Operand::Memory {
            width: 0,
            base,
            index,
            scale,
            addend,
        } if byte_add == 0 => op.clone(),
        Operand::Memory {
            width: 0,
            base,
            index,
            scale,
            addend,
        } => Operand::Memory {
            width: 0,
            base: base.clone(),
            index: index.clone(),
            scale: *scale,
            addend: addend + byte_add,
        },
        _ => Operand::Unknown,
    }
}

/// Does `op` leave the condition flags (NZCV) untouched?
///
/// AArch64 writes NZCV from a small, closed set: `ADDS`/`SUBS`/`ANDS`/`BICS`
/// (the `CMP`/`CMN`/`TST` aliases decode to these), `ADCS`/`SBCS`,
/// `CCMP`/`CCMN`, `FCMP`/`FCMPE`/`FCCMP`, and an explicit `MSR NZCV, x`.
/// Everything else preserves them — which is why the compiler freely
/// shuffles registers between a `cmp` and the `b.cond` that consumes it, and
/// why clearing `pending_cmp` on every instruction produced the mass
/// `Compare(kind, Unknown, 0)` → `if (? != 0)` wrong-logic class.
///
/// Deliberately conservative: only opcodes listed here count as preserving,
/// so an opcode this table does not know still clears the pending compare
/// exactly as before. A wrong "preserves" answer is worse than a miss — it
/// would bind a stale condition to a branch — so the list holds only
/// instructions with an unambiguously absent flag write. Calls
/// (`BL`/`BLR`/`BR`) and returns are absent on purpose: NZCV is caller-saved,
/// so a compare never survives a call.
/// Materialize the pending flag state as a BOOLEAN VALUE in the IR and return
/// the operand holding it.
///
/// The branch arm has always done this translation inline; `CSEL` needs
/// exactly the same one, and duplicating it is how the two drift apart, so it
/// lives here and both callers share it.
///
/// The `Merge` case (`cmp; ccmp`) collapses exactly as it does for a branch:
/// the result is `P ? cmp(lhs, rhs) : <#nzcv constant>`, and because that
/// fallback arm is a CONSTANT it reduces to `P && Q` (when the constant reads
/// true) or `!P || Q` (when it reads false). That reduction is equally valid
/// feeding a select, since a constant arm is a constant in either reading.
///
/// Scratch ids are allocated downward from `u32::MAX - 2`, matching the
/// `fresh_scratch!` macro in the lifter loop.
///
/// Before the compare is emitted, every REGISTER operand is pinned to the value
/// it held when the flags were set (`pin_deferred_read`). Without that, a
/// redefinition between the compare and this point makes the name resolve to
/// the new value and the compare degenerates to `x == x` — always true.
fn materialize_cond(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    next_scratch: &mut u32,
    state: Option<&FlagState>,
    cond_code: u8,
    addr: u64,
) -> Operand {
    // Takes the counter as an ARGUMENT rather than capturing it: the pin helper
    // below also allocates from the same counter, and a `FnMut` closure holding
    // a unique borrow of it would make those calls borrow-check failures.
    let take = |ns: &mut u32| {
        let v = *ns;
        *ns = v.wrapping_sub(1);
        Operand::Local(v)
    };
    let code = cond_code & 0xF;
    let kind = cond_to_kind(code);
    match state {
        Some(FlagState::Merge {
            prev_lhs,
            prev_rhs,
            prev_at,
            ccmp_cond,
            lhs,
            rhs,
            at,
            nzcv,
        }) => {
            let k_true = eval_cond_code(code, *nzcv);
            let p_kind = if k_true {
                cond_to_kind(*ccmp_cond).invert()
            } else {
                cond_to_kind(*ccmp_cond)
            };
            // Pin BOTH halves separately: the previous compare's operands were
            // captured at `prev_at` and the ccmp's at `at`, so a redefinition
            // between them belongs to only one of the two halves.
            let mut pl = prev_lhs.clone();
            let mut pr = prev_rhs.clone();
            pin_deferred_read(insns, addresses, *prev_at, &mut pl, next_scratch);
            pin_deferred_read(insns, addresses, *prev_at, &mut pr, next_scratch);
            let sp = take(next_scratch);
            insns.push(Insn::new(
                Op::Compare(p_kind),
                vec![sp.clone(), pl, pr],
                addr,
            ));
            let mut cl = lhs.clone();
            let mut cr = rhs.clone();
            pin_deferred_read(insns, addresses, *at, &mut cl, next_scratch);
            pin_deferred_read(insns, addresses, *at, &mut cr, next_scratch);
            let sq = take(next_scratch);
            insns.push(Insn::new(Op::Compare(kind), vec![sq.clone(), cl, cr], addr));
            let sm = take(next_scratch);
            insns.push(Insn::new(
                Op::Arith(if k_true {
                    ArithKind::Or
                } else {
                    ArithKind::And
                }),
                vec![sm.clone(), sp, sq],
                addr,
            ));
            sm
        }
        Some(FlagState::Cmp { lhs, rhs, at }) => {
            let mut l = lhs.clone();
            let mut r = rhs.clone();
            pin_deferred_read(insns, addresses, *at, &mut l, next_scratch);
            pin_deferred_read(insns, addresses, *at, &mut r, next_scratch);
            let s = take(next_scratch);
            insns.push(Insn::new(Op::Compare(kind), vec![s.clone(), l, r], addr));
            s
        }
        None => {
            let s = take(next_scratch);
            insns.push(Insn::new(
                Op::Compare(kind),
                vec![s.clone(), Operand::Unknown, Operand::imm(0)],
                addr,
            ));
            s
        }
    }
}

fn preserves_nzcv(op: A64Opcode) -> bool {
    use A64Opcode::*;
    matches!(
        op,
        // moves, constants, logic and arithmetic WITHOUT the S suffix
        MOVN | MOVK | MOVZ | ADD | SUB | ADC | SBC | AND | ORR | ORN | EOR | EON | BIC
            | BFM | UBFM | SBFM | EXTR | ADR | ADRP | MOVI
        // multiply / divide / variable shifts: no flag write
            | MADD | MSUB | SMADDL | SMSUBL | SMULH | UMADDL | UMSUBL | UMULH
            | UDIV | SDIV | LSLV | LSRV | ASRV | RORV
        // bit / byte manipulation
            | RBIT | REV | REV16 | REV32 | CLZ | CLS
            | CRC32B | CRC32H | CRC32W | CRC32X
            | CRC32CB | CRC32CH | CRC32CW | CRC32CX
        // loads / stores (incl. atomics and pairs): never touch NZCV
            | LDLAR | LDLARB | LDLARH | LDAXR | LDAXRB | LDAXRH | LDAXP
            | LDAR | LDARB | LDARH | LDNP | LDP | LDPSW | LDR | LDRB | LDRH
            | LDRSB | LDRSH | LDRSW | LDTR | LDTRB | LDTRH | LDTRSB | LDTRSH
            | LDTRSW | LDUR | LDURB | LDURH | LDURSB | LDURSH | LDURSW
            | LDXR | LDXRB | LDXRH | LDXP
            | STLLR | STLLRB | STLLRH | STLR | STLRB | STLRH | STLXP | STLXR
            | STLXRB | STLXRH | STNP | STP | STR | STRB | STRH | STTR | STTRB
            | STTRH | STUR | STURB | STURH | STXR | STXRB | STXRH | STXP
            | LD1 | LD1R | LD2 | LD2R | LD3 | LD3R | LD4 | LD4R
            | ST1 | ST2 | ST3 | ST4
        // Conditional branches do not write NZCV and their FALL-THROUGH is
        // a real successor, so a compare may legitimately carry across them.
        // An unconditional B is deliberately NOT here: it writes nothing
        // either, but the instruction that follows it in address order is
        // usually not the branch target, so carrying the compare there bound
        // the wrong condition. `passes::resolve_cross_block_conditions`
        // supplies those by dominance instead.
            | CBZ | CBNZ | TBZ | TBNZ
        // conditional select: READS the flags, does not write them
            | CSEL | CSINC | CSINV | CSNEG
        // FP / SIMD arithmetic and conversions: no NZCV
            | FMOV | FABS | FNEG | FSQRT | FMUL | FDIV | FADD | FSUB | FMAX
            | FMIN | FMAXNM | FMINNM | FCSEL
            | FMADD | FMSUB | FNMADD | FNMSUB
            | SCVTF | UCVTF | FCVT | FCVTZS | FCVTZU | BFCVT
            | FRINTN | FRINTP | FRINTM | FRINTZ | FRINTA | FRINTX | FRINTI
    )
}

/// `ILR_DISASM=<rva>` — print one method's NATIVE instruction stream beside the
/// IR the lifter produced from it, address by address, then list the addresses
/// only one side has.
///
/// This is the direct answer to "does the generated C# correspond to the
/// disassembly?". The left column is what the decoder for the TARGET
/// architecture says the machine code is (pass `arch`);
/// the `ir` lines are what the lifter made of it. The two mismatch lists ARE the
/// coverage gap, and they are the per-method view of what the corpus-wide
/// lifter census measures:
///
///   * a native instruction with **no IR** at its address = a missing lifter
///     arm — the instruction was dropped or folded into a placeholder;
///   * IR at an address with **no native** instruction = invented. Only these
///     are expected there: `Nop` (an address reserved for a compare that was
///     folded into its branch), `ShiftStack` (the stack analyzer's bookkeeping),
///     the synthetic tail `Return`, and the extra operands of a multi-insn group
///     (`Compare`+`CondJump`, `LDP` pairs), which share the first instruction's
///     address by design. Anything else there is a real defect.
///
/// Prefer this over eyeballing: it enumerates, per method, exactly which
/// instruction has no representation instead of leaving it to be noticed.
/// Which decoder the "native" column of `disasm_align_report` must use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TargetArch {
    Arm64,
    Arm32,
    X64,
}

pub fn disasm_align_report(bytes: &[u8], rva: u64, insns: &[Insn], arch: TargetArch) -> String {
    // THE NATIVE COLUMN MUST BE DECODED FOR THE BINARY'S OWN ARCHITECTURE.
    //
    // This function used `A64Decoder` unconditionally, so on an ARM32 or x86
    // target it "decoded" the method's bytes as AArch64 and printed fiction: the
    // first method tried on the Android binary came out as `invalid x16, x19,
    // [sp, #-0x26]`, 26 "instructions" from a body the lifter read 240 IR insns
    // out of. Any check of the generated C# against "the disassembly" on a
    // non-ARM64 target was therefore meaningless — the left column was not the
    // disassembly. Same class as the earlier `verify.rs` defect that looked for a
    // branch's compare at the branch's own address: an oracle that misreads its
    // own reference silently reports whatever it likes.
    let mut natives: Vec<(u64, String)> = Vec::new();
    match arch {
        TargetArch::Arm64 => {
            let decoder = A64Decoder::default();
            let mut off = 0usize;
            while off + 4 <= bytes.len() {
                let mut reader = U8Reader::new(&bytes[off..]);
                match decoder.decode(&mut reader) {
                    Ok(inst) => {
                        let len = inst.len().to_const() as usize;
                        natives.push((rva + off as u64, format!("{inst}")));
                        off += len.max(4);
                    }
                    Err(_) => off += 4,
                }
            }
        }
        TargetArch::Arm32 => {
            use yaxpeax_arch::LengthedInstruction as _;
            let decoder = yaxpeax_arm::armv7::InstDecoder::default();
            let mut off = 0usize;
            while off + 2 <= bytes.len() {
                let mut reader = U8Reader::new(&bytes[off..]);
                match decoder.decode(&mut reader) {
                    Ok(inst) => {
                        // A32 lengths are 2 or 4 bytes (Thumb-2 exists here).
                        let len = inst.len().to_const() as usize;
                        natives.push((rva + off as u64, format!("{inst}")));
                        off += len.max(2);
                    }
                    Err(_) => off += 4,
                }
            }
        }
        TargetArch::X64 => {
            use iced_x86::{Decoder as X86De, DecoderOptions, FastFormatter};
            let mut dec = X86De::with_ip(64, bytes, rva, DecoderOptions::NONE);
            let mut fmt = FastFormatter::new();
            let mut text = String::new();
            while dec.can_decode() {
                let insn = dec.decode();
                if insn.code() == iced_x86::Code::INVALID {
                    break;
                }
                text.clear();
                fmt.format(&insn, &mut text);
                natives.push((insn.ip(), text.clone()));
            }
        }
    }

    let mut out = String::new();
    out.push_str(&format!(
        "== {rva:#x}: native {} insns, lifted IR {} insns ==\n",
        natives.len(),
        insns.len()
    ));
    let mut covered = 0usize;
    for (addr, text) in &natives {
        out.push_str(&format!("  {addr:#010x}  {text}\n"));
        let at: Vec<&Insn> = insns.iter().filter(|i| i.address == *addr).collect();
        if at.is_empty() {
            out.push_str("              !! NO IR for this instruction\n");
        } else {
            covered += 1;
            for i in at {
                out.push_str(&format!(
                    "              ir {:<16} {:?}\n",
                    format!("{:?}", i.op),
                    i.operands
                ));
            }
        }
    }

    let mut orphan: Vec<&Insn> = insns
        .iter()
        .filter(|i| !natives.iter().any(|(a, _)| *a == i.address))
        .collect();
    if !orphan.is_empty() {
        orphan.sort_by_key(|i| i.address);
        out.push_str(&format!(
            "  -- {} IR insns at no native address (Nop / ShiftStack / tail Return /\n     multi-insn group operands are expected here):\n",
            orphan.len()
        ));
        for i in orphan.iter().take(20) {
            out.push_str(&format!(
                "  {:#010x}  ir {:?} {:?}\n",
                i.address, i.op, i.operands
            ));
        }
        if orphan.len() > 20 {
            out.push_str("  ...\n");
        }
    }
    out.push_str(&format!(
        "== coverage: {covered}/{} native instructions have IR; {} missing ==\n",
        natives.len(),
        natives.len().saturating_sub(covered)
    ));
    out
}

fn branch_target_rva(insn: &A64Insn, addr: u64) -> Option<usize> {
    for op in &insn.operands {
        if let A64Op::PCOffset(off) = op {
            return Some((addr as i64 + off) as usize);
        }
    }
    None
}

fn push_binary(
    insns: &mut Vec<Insn>,
    op: Op,
    dest: String,
    a: Option<Operand>,
    b: Option<Operand>,
    addr: u64,
) {
    let mut operands = vec![Operand::reg(dest), a.unwrap_or(Operand::Unknown)];
    if let Some(b) = b {
        operands.push(b);
    }
    insns.push(Insn::new(op, operands, addr));
}

/// Bit width of an A64 general-purpose operand (`Wn` = 32, `Xn` = 64).
///
/// The bitfield arms need it because the SAME `immr`/`imms` immediates mean
/// different things at the two widths, and because `ROR` of a constant has to
/// wrap inside the operand size rather than inside 64 bits.
fn gpr_bits(op: &A64Op) -> u32 {
    match op {
        A64Op::Register(yaxpeax_arm::armv8::a64::SizeCode::X, _)
        | A64Op::RegisterOrSP(yaxpeax_arm::armv8::a64::SizeCode::X, _) => 64,
        _ => 32,
    }
}

/// Architectural byte width for a scalar register operand used by a load or
/// store. The `STR` opcode family covers both W/X and S/D/Q registers; its
/// mnemonic alone does not carry enough width information for the IR.
fn register_width(op: &A64Op) -> u8 {
    if let Some(name) = simd_name(op) {
        return fp_reg_bytes(&name).unwrap_or(0);
    }
    if gpr_of(op).is_some() {
        return (gpr_bits(op) / 8) as u8;
    }
    0
}

/// ARM ARM `DecodeBitMasks(N, imms, immr, FALSE)` — returns `wmask`.
///
/// `BFM`/`UBFM`/`SBFM` are all defined as `ROR(src, immr) & wmask`, so the
/// mask is the whole semantics of the instruction. Without it `ubfx`/`sbfx`
/// (the general field-extract forms) had to be guessed at; the previous arm
/// only recognised the two shift aliases (`imms == 63` → `lsr`,
/// `imms + 1 == immr` → `lsl`) and emitted `Unknown` for everything else —
/// silently, because an arm existed and the census therefore stayed quiet.
fn decode_bit_masks(bits: u32, imms: u32, immr: u32) -> u64 {
    // `N:NOT(imms)` — N is bit 6 for the 64-bit forms (SF == 1) and 0 for the
    // 32-bit ones; the leading-set-bit position of that concatenation is
    // `len`, i.e. log2 of the replicated element size.
    let n = u32::from(bits == 64);
    let concat = (n << 6) | ((!imms) & 0x3f);
    if concat == 0 {
        // Architecturally UNDEFINED (len < 1). No mask is the honest answer.
        return 0;
    }
    let len = 31 - concat.leading_zeros();
    let levels = (1u32 << len) - 1;
    let s = imms & levels;
    let r = immr & levels;
    let d = s.wrapping_sub(r) & levels;
    let esize = 1u32 << len;
    let ones = |n: u32| -> u64 { if n >= 64 { u64::MAX } else { (1u64 << n) - 1 } };
    let welem = ones(s + 1);
    let telem = ones(d + 1);
    let _ = telem; // SBFM's sign bit; BFM/UBFM need only `wmask`
    let mut wmask = 0u64;
    let mut k = 0u32;
    while k < bits {
        wmask |= welem << k;
        k += esize;
    }
    if bits == 64 {
        wmask.rotate_right(r)
    } else {
        (wmask as u32).rotate_right(r) as u64
    }
}

/// `ROR` of a register by a constant, materialized with the shift/or pair the
/// ISA has (there is no constant-rotate encoding; the compiler emits `EXTR`,
/// which lands here through the same helper).
fn ror_const(
    insns: &mut Vec<Insn>,
    virt_serial: &mut u32,
    src: Operand,
    amount: u32,
    bits: u32,
    addr: u64,
) -> Operand {
    if amount == 0 {
        return src;
    }
    let a = amount % bits;
    if a == 0 {
        return src;
    }
    let mut tmp = |op: Op| -> Operand {
        let t = Operand::reg(format!("xt{}", *virt_serial));
        *virt_serial += 1;
        insns.push(Insn::new(
            op,
            vec![t.clone(), src.clone(), Operand::imm(a as i64)],
            addr,
        ));
        t
    };
    let hi = tmp(Op::Arith(ArithKind::Shr));
    let lo = tmp(Op::Arith(ArithKind::Shl));
    let r = Operand::reg(format!("xt{}", *virt_serial));
    *virt_serial += 1;
    insns.push(Insn::new(
        Op::Arith(ArithKind::Or),
        vec![r.clone(), hi, lo],
        addr,
    ));
    r
}

/// Flag state produced by the `test` family (`ands`/`bics` with `xzr`).
///
/// `tst a, b` is `ands xzr, a, b` and tests `a AND b` against ZERO — it is not
/// a comparison of `a` with `b`. Recording the raw operand pair as a plain
/// compare rendered every mask test as `a == b`: plausible text, wrong logic.
/// The AND is materialized so the recorded condition is `(a & b) == 0`.
fn test_flags(
    insns: &mut Vec<Insn>,
    virt_serial: &mut u32,
    lhs: Operand,
    rhs: Option<Operand>,
    invert_rhs: bool,
    addr: u64,
) -> FlagState {
    let mut rhs = rhs;
    if invert_rhs {
        if let Some(bv) = rhs {
            let t = Operand::reg(format!("xt{}", *virt_serial));
            *virt_serial += 1;
            insns.push(Insn::new(Op::Not, vec![t.clone(), bv], addr));
            rhs = Some(t);
        }
    }
    let acc = Operand::reg(format!("xt{}", *virt_serial));
    *virt_serial += 1;
    insns.push(Insn::new(
        Op::Arith(ArithKind::And),
        vec![acc.clone(), lhs, rhs.unwrap_or(Operand::Unknown)],
        addr,
    ));
    // `at` is read AFTER the pushes above: the AND writes a scratch register
    // name (`xt{n}`), and anchoring the pin at the index of an instruction that
    // DEFINES one of the compare's own operands would make the pin read an
    // undefined local.
    FlagState::Cmp {
        lhs: acc,
        rhs: Operand::imm(0),
        at: insns.len(),
    }
}
