//! x86/x64 lifter (SysV + MSVC ABIs) → recovery IR, using iced-x86.
//!
//! Register normalization maps ABI registers onto the shared virtual names
//! (`x0..x5`) so locals seeding / stack analysis / C# rendering work for both
//! architectures. Flag-producing ops (CMP/TEST) fold with the next Jcc.

use super::ir::{
    ArithKind, CompareKind, Conversion, FloatArithKind, Insn, LanePick, Op, Operand, RET_REG,
    SelectVariant, SimdOp,
};
use iced_x86::{
    ConditionCode, Decoder, DecoderOptions, FlowControl, Instruction as X86Insn, MemorySize,
    Mnemonic, OpKind, Register,
};
use std::collections::BTreeSet;

fn sse_op(insn: &X86Insn, n: usize, abi: X86Abi) -> Option<Operand> {
    if (n as u32) >= insn.op_count() {
        return None;
    }
    match insn.op_kind(n as u32) {
        OpKind::Register => Some(Operand::reg(gpr(insn.op_register(n as u32), abi))),
        OpKind::Memory => Some(memory_operand(insn, abi)),
        _ => None,
    }
}

pub struct LiftResult {
    pub insns: Vec<Insn>,
    pub addresses: Vec<u64>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum X86Abi {
    SysV,
    Msvc,
}

/// Ordinal 0–15 of a general-purpose register at ANY width, or `None` when the
/// register is not one of the sixteen GPRs.
///
/// x86 spells ONE physical register several ways — `al`, `ah`, `ax`, `eax` and
/// `rax` are all register 0 — and real code mixes the widths constantly:
/// `setcc al` feeds a later `movzx eax, al`; `shl eax, cl` shifts by the byte
/// register; a byte load is compared as a word. Naming the widths separately
/// invents a register no definition ever writes, so the byte-sized write is
/// INVISIBLE and every read through another spelling becomes an undefined
/// local — which is how a single `setcc al` corrupts a whole method.
///
/// iced's enum lays each width out as a contiguous block in the same order, so
/// the ordinal is a plain range subtraction. The high-byte registers (`ah`…
/// `bh`) are the SAME register as their low-byte twin and take the same
/// ordinal, matching every other lifter in this crate.
fn gpr_ordinal_of(reg: Register) -> Option<u32> {
    let n = reg as u32;
    Some(match n {
        53..=68 => n - 53, // rax .. r15
        37..=52 => n - 37, // eax .. r15d
        21..=36 => n - 21, // ax  .. r15w
        1..=4 => n - 1,    // al, cl, dl, bl
        5..=20 => n - 5,   // ah..bh, spl..dil, r8l..r15l
        _ => return None,
    })
}

/// The canonical name for GPR ordinal `ord` under `abi`.
///
/// The ABI ordinals are deliberate, not a rename of the register file: argument
/// register *i* takes the name `x{i}` whatever physical register the ABI puts it
/// in, so every downstream pass (locals seeding, argument alignment, C#
/// rendering) speaks one positional vocabulary across ARM64, ARM32, x64 SysV and
/// x64 MSVC.
///
/// The one place that mapping must NOT hold is ordinal 0 on x86-64: there `rax`
/// is the result register and argument 0 is a DIFFERENT register (`rdi` under
/// SysV, `rcx` under MSVC). Naming both `x0` — which is correct on ARM64, where
/// `x0` really is both — is what let a call result and the receiver share one
/// name, so `locals::rewrite_pinned_copies` retired the receiver whenever a
/// result was written and an `x0` argument slot could read a discarded return
/// value. Ordinal 0 is therefore [`RET_REG`], never `x0`.
fn gpr_name_for_ordinal(ord: u32, abi: X86Abi) -> String {
    match (ord, abi) {
        (0, _) => RET_REG.into(),
        (1, X86Abi::SysV) => "x3".into(),
        (1, X86Abi::Msvc) => "x0".into(),
        (2, X86Abi::SysV) => "x2".into(),
        (2, X86Abi::Msvc) => "x1".into(),
        (3, _) => "g3".into(),
        (4, _) => "sp".into(),
        (5, _) => "fp".into(),
        (6, X86Abi::SysV) => "x1".into(),
        (6, X86Abi::Msvc) => "g6".into(),
        (7, X86Abi::SysV) => "x0".into(),
        (7, X86Abi::Msvc) => "g7".into(),
        (8, X86Abi::SysV) => "x4".into(),
        (8, X86Abi::Msvc) => "x2".into(),
        (9, X86Abi::SysV) => "x5".into(),
        (9, X86Abi::Msvc) => "x3".into(),
        (other, _) => format!("g{other}"),
    }
}

/// The argument registers of `abi`, in positional order — exactly the slots a
/// call's operand dump must carry.
///
/// This list is ABI-SPECIFIC and was hardcoded to the SysV six on BOTH call
/// arms. MSVC passes only FOUR arguments in registers (`rcx, rdx, r8, r9`) and
/// everything beyond that on the stack, so under that convention `x4`/`x5` named
/// registers that do not exist:
///
///   * they manufactured two locals per method (`slot=x4`, `slot=x5`) that no
///     instruction ever writes — `locals::create` builds a `LocalInfo` from
///     whatever slot names appear as operands;
///   * they produced two `__abi` placeholders in every call whose callee is not
///     named, which is where a large share of the remaining `__abi` came from;
///   * and they inflated every call's operand count by two, which the argument
///     alignment pass then has to reason about.
///
/// The names are POSITIONS, so the MSVC list is `x0..x3` and position 4 onward is
/// simply not a register — the stack slots are modelled separately as
/// `stack_N` locals.
fn arg_reg_names(abi: X86Abi) -> &'static [&'static str] {
    match abi {
        X86Abi::SysV => &["x0", "x1", "x2", "x3", "x4", "x5"],
        X86Abi::Msvc => &["x0", "x1", "x2", "x3"],
    }
}

fn gpr(reg: Register, abi: X86Abi) -> String {
    // SSE/AVX registers must use the SAME FP names (`vf0`…) that `sse_op`
    // emits and that every downstream FP helper (`fp_of`, float detection,
    // SSA/stack analysis) already speaks. The catch-all below used to render
    // them as `r164`-style raw ordinals, so one XMM reached through `op_at`
    // became a DIFFERENT local from the same XMM reached through `sse_op` —
    // defs and uses stopped matching. This is the x86 twin of the arm64
    // FP-store naming bug (`operand_at` naming only GPRs).
    let n = reg as u32;
    let xmm = Register::XMM0 as u32;
    if (xmm..xmm + 32).contains(&n) {
        return format!("vf{}", n - xmm);
    }
    // Every vector file, by LANE INDEX: `xmm3`, `ymm3` and `zmm3` are one
    // physical register at three widths, and SSE with AVX continues the same
    // value through them. Naming the widths apart invents a register no def
    // ever writes. (The three blocks are contiguous in iced's enum.)
    for base in [Register::XMM0, Register::YMM0, Register::ZMM0] {
        let b = base as u32;
        if (b..b + 32).contains(&n) {
            return format!("vf{}", n - b);
        }
    }
    // Scalar registers collapse to their 64-bit ordinal first, so every width
    // of one physical register lands on one name.
    match gpr_ordinal_of(reg) {
        Some(ord) => gpr_name_for_ordinal(ord, abi),
        None => format!("r{}", n),
    }
}

/// Build the memory operand for an x86 instruction, resolving the TWO special
/// bases correctly.
///
/// * **RIP-relative** (`is_ip_rel_memory_operand`) is the only way x64 reaches
///   a static, a global, or the IL2CPP metadata usage tables. It used to be
///   named through the generic `gpr` fallback, whose catch-all produced
///   `r{ordinal}` — a register that is NEVER defined anywhere in the method. So
///   every static/global access read through a phantom register and the whole
///   dependent chain became `__unknown`. The address is fully determined by the
///   instruction, so it is materialized exactly: `abs` plus the real absolute
///   address (the established spelling for an absolute access).
/// * **no base, no index** (`mov eax, [0x1234]`) is already absolute; `abs` is
///   the right base and the displacement is the address.
/// The ACCESS WIDTH in bytes of an instruction's memory operand: 1/2/4/8 for the
/// scalar widths this IR can spell, 0 for anything else.
///
/// iced already knows the size (`Instruction::memory_size`), so leaving it at
/// the pointer-width default inside `memory_operand` was pure loss: a
/// `mov byte ptr [x], 1` rendered as `*(nint*)(x) = 1` — an EIGHT-byte store
/// that clobbers the seven following fields, which through a `ref` parameter or
/// an array element is a wrong program, not a wrong spelling. The read side is
/// the same bug mirrored: `movzx eax, byte ptr [x]` read eight bytes where the
/// source reads one.
///
/// Only the scalar classes map here. A packed SSE access (16/32/64 bytes), a
/// broadcast, or `MemorySize::Unknown` — which is also what `lea` reports, and a
/// `lea` operand is an ADDRESS rather than an access — stays 0, the renderer's
/// existing pointer-width spelling. So an unhandled class can only lose
/// precision; it can never invent a width.
fn mem_access_width(insn: &X86Insn) -> u8 {
    match insn.memory_size() {
        MemorySize::UInt8 | MemorySize::Int8 => 1,
        MemorySize::UInt16 | MemorySize::Int16 => 2,
        MemorySize::UInt32 | MemorySize::Int32 => 4,
        MemorySize::UInt64 | MemorySize::Int64 => 8,
        _ => 0,
    }
}

fn memory_operand(insn: &X86Insn, abi: X86Abi) -> Operand {
    let base = insn.memory_base();
    // iced folds `IP + instruction length + displacement` INTO the operand's
    // displacement for an IP-relative access, so for that form
    // `memory_displacement64()` already IS the absolute address. Adding
    // `ip + len` on top of it (the old `rip_addend`) doubled the term —
    // `mov rax,[rip+0x10]` at 0x1000 lifted to `abs + 0x201E` instead of
    // `abs + 0x1017`, so every static/global/metadata access on x64 pointed at
    // an address in the middle of nowhere.
    let (b, addend) = if insn.is_ip_rel_memory_operand() {
        (Operand::reg("abs"), insn.memory_displacement64() as i64)
    } else if base == Register::RIP {
        // A RIP base that iced did not flag as IP-relative: the displacement is
        // still raw here, so the `pc`-relative bias has to be applied by hand.
        (
            Operand::reg("abs"),
            insn.memory_displacement64() as i64 + insn.ip() as i64 + insn.len() as i64,
        )
    } else if base == Register::None {
        // no base, no index (`mov eax, [0x1234]`): the displacement IS the address
        (Operand::reg("abs"), insn.memory_displacement64() as i64)
    } else {
        (
            Operand::reg(gpr(base, abi)),
            insn.memory_displacement64() as i64,
        )
    };
    let index = if insn.memory_index() == Register::None {
        None
    } else {
        Some(Box::new(Operand::reg(gpr(insn.memory_index(), abi))))
    };
    Operand::Memory {
        width: mem_access_width(insn),
        base: Box::new(b),
        index,
        scale: (insn.memory_index_scale() as u8).max(1),
        addend,
    }
}

/// Is this register the 64-bit form of a general-purpose register?
fn reg_is_64(r: Register) -> bool {
    use Register::*;
    matches!(
        r,
        RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15
    )
}

/// Emit a READ-MODIFY-WRITE through a MEMORY destination.
///
/// `add [rax], ecx` must become a real load, operation and store. Handing the
/// renderer an `Arith` whose destination is a memory operand gives it nothing to
/// attach a store to — the same category error as inlining a slot's value inside
/// `&` — and the old fallback refused memory destinations outright, so every
/// store to a static field, an array element or a `ref` parameter performed by a
/// RMW mnemonic vanished from the output while the value chain nearby stayed
/// intact. Measured on a real x64 IL2CPP binary this is one of the two densest
/// remaining gaps.
fn emit_rmw_arith(
    insns: &mut Vec<Insn>,
    scratch: &mut Scratch,
    mem: Operand,
    kind: ArithKind,
    src: Operand,
    addr: u64,
) {
    let t = scratch.fresh();
    insns.push(Insn::move_(t.clone(), mem.clone(), addr));
    insns.push(Insn::new(
        Op::Arith(kind),
        vec![t.clone(), t.clone(), src],
        addr,
    ));
    insns.push(Insn::move_(mem, t, addr));
}

/// `neg [mem]` / `not [mem]`: same load-op-store shape with the unary op.
fn emit_rmw_unary(insns: &mut Vec<Insn>, scratch: &mut Scratch, mem: Operand, addr: u64, op: Op) {
    let t = scratch.fresh();
    insns.push(Insn::move_(t.clone(), mem.clone(), addr));
    insns.push(Insn::new(op, vec![t.clone(), t.clone()], addr));
    insns.push(Insn::move_(mem, t, addr));
}

/// Turn a pending `bt`-style bit test into the single-bit condition a
/// carry-testing consumer reads.
///
/// `bt` writes **CF only**. Reading it as an ordering (`cond_kind` maps `jb` to
/// `LessThan`) produced `if (v < 0)`-shaped nonsense for what is really
/// `(v & (1 << bit)) != 0`. Returns None for every condition that does not test
/// CF, so a `bt` followed by `je` still degrades to a missing condition instead
/// of a wrong one.
fn bit_scratch_if_carry(
    insns: &mut Vec<Insn>,
    scratch: &mut Scratch,
    pending_bit: Option<(Operand, BitIndex)>,
    cc: ConditionCode,
    addr: u64,
) -> Option<Operand> {
    let (value, bit) = pending_bit?;
    let bit_is_set = match cc {
        ConditionCode::b => true,
        ConditionCode::ae => false,
        _ => return None,
    };
    let shifted = scratch.fresh();
    match bit {
        BitIndex::Constant(0) => insns.push(Insn::move_(shifted.clone(), value, addr)),
        BitIndex::Constant(bit) => insns.push(Insn::new(
            Op::Arith(ArithKind::Shr),
            vec![shifted.clone(), value, Operand::imm(bit)],
            addr,
        )),
        BitIndex::Variable { value: idx, mask } => {
            // `bt r/m, r` masks the register count to the operand width before
            // shifting (`& 31` for a 32-bit operand, `& 63` for 64-bit), so the
            // mask is part of the semantics, not a convenience.
            let n = scratch.fresh();
            insns.push(Insn::new(
                Op::Arith(ArithKind::And),
                vec![n.clone(), idx, Operand::imm(mask)],
                addr,
            ));
            insns.push(Insn::new(
                Op::Arith(ArithKind::Shr),
                vec![shifted.clone(), value, n],
                addr,
            ));
        }
    }
    let masked = scratch.fresh();
    insns.push(Insn::new(
        Op::Arith(ArithKind::And),
        vec![masked.clone(), shifted, Operand::imm(1)],
        addr,
    ));
    let cond = scratch.fresh();
    insns.push(Insn::new(
        Op::Compare(if bit_is_set {
            CompareKind::NotEqual
        } else {
            CompareKind::Equal
        }),
        vec![cond.clone(), masked, Operand::imm(0)],
        addr,
    ));
    Some(cond)
}

/// Materialize the x86 carry as an INTEGER 0/1, built from ops the renderer
/// already understands (a `Compare` for the flag, a `Select` for the 0/1).
///
/// x86's CF after a SUBTRACT is the BORROW (`lhs < rhs`), unlike A32/A64 where
/// the same flag means NOT-borrow — which is why this lives here instead of
/// being shared with the arm32 lifter.
fn materialize_x86_carry(
    insns: &mut Vec<Insn>,
    scratch: &mut Scratch,
    subtract: bool,
    lhs: Operand,
    rhs: Operand,
    addr: u64,
) -> Operand {
    let cond = scratch.fresh();
    if subtract {
        insns.push(Insn::new(
            Op::Compare(CompareKind::LessThan),
            vec![cond.clone(), lhs, rhs],
            addr,
        ));
    } else {
        // carry out of ADD = (lhs + rhs) < lhs
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
    insns.push(Insn::new(
        Op::Select(SelectVariant::Identity),
        vec![cf.clone(), cond, Operand::imm(1), Operand::imm(0)],
        addr,
    ));
    cf
}

fn cond_kind(code: ConditionCode) -> CompareKind {
    use ConditionCode as C;
    match code {
        C::e => CompareKind::Equal,
        C::ne => CompareKind::NotEqual,
        C::b | C::be | C::l | C::le => CompareKind::LessThan,
        C::ae | C::a | C::ge | C::g => CompareKind::GreaterOrEqual,
        C::s => CompareKind::LessThan,
        C::ns => CompareKind::GreaterOrEqual,
        // Parity. It is only ever meaningful here as the FP "unordered" result
        // of the `ucomiss`/`comisd` immediately before (`PF` is set exactly
        // when a NaN made the comparison unordered), so `p` reads as
        // "not equal" and `np` as "equal" — which is what the NaN test that
        // compiles to `ucomiss; jp` actually computes.
        C::p => CompareKind::NotEqual,
        C::np => CompareKind::Equal,
        _ => CompareKind::NotEqual,
    }
}

/// The bit position a pending `bt`/`bts`/`btr`/`btc` selected.
///
/// Kept as an enum rather than a plain constant because x86 has two encodings:
/// `bt r/m, imm8` (constant, exact shift) and `bt r/m, r` (a live index that is
/// masked to the operand width). Collapsing the second into "no bit known" made
/// every register-index bit test silently lose its condition.
#[derive(Clone, Debug)]
enum BitIndex {
    Constant(i64),
    Variable { value: Operand, mask: i64 },
}

struct PendingCmp {
    lhs: Operand,
    rhs: Operand,
    /// Listing index at which these operand NAMES were captured.
    ///
    /// The compare is materialized at its CONSUMER (the Jcc), so a redefinition
    /// of a named register in between would make the compare read the LATER
    /// value. `at` is what lets the operands be pinned to the value they had
    /// when the flags were set.
    at: usize,
}

/// Unique scratch local ids for compare/branch results.
///
/// The old shared `Operand::Local(u32::MAX)` sentinel made every Compare in a
/// method the SAME def, so "which definition feeds this branch" had to be
/// guessed among several and regularly bound the wrong one — the
/// contradictory-guard corruption (`if (x != 0) { if (x == 0) {`).
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

/// Force a DEFERRED read to see the value it had when the flags were set. A
/// copy of the register is inserted immediately before the instruction that
/// would clobber it and the operand is rewritten to read that copy, which SSA
/// then binds to the old value — exactly what the hardware read.
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
    let Some(j) = (from..insns.len()).find(|&j| {
        Insn::dest_position(&insns[j].op) == Some(0)
            && matches!(&insns[j].operands[0], Operand::Register(n) if *n == name)
    }) else {
        return;
    };
    let s = *next_scratch;
    *next_scratch = s.wrapping_sub(1);
    let pin = Operand::Local(s);
    while addresses.len() < insns.len() {
        addresses.push(insns[addresses.len()].address);
    }
    let at_addr = insns[j].address;
    insns.insert(j, Insn::move_(pin.clone(), Operand::reg(name), at_addr));
    addresses.insert(j, at_addr);
    *op = pin;
}

/// Emit the `Compare` for a pending flag state and return the boolean scratch
/// the `CondJump`/`Select` consumes. One implementation for every consumer.
/// `Compare(kind, Unknown, 0)` materializations — the wrong-logic class.
///
/// Incremented once per flag CONSUMER (`jcc`/`setcc`/`cmovcc`) that reached
/// [`materialize_pending`] with no live compare. This is the one defect class
/// where the emitted text looks C#-shaped and is WRONG, so it is counted at the
/// point of creation rather than inferred from the output: `concord` reports it
/// beside its method-level `lost-condition` count so the two cannot disagree
/// about whether a condition was lost.
static LOST_CONDITION_MATERIALIZED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Total [`materialize_pending`] calls that had no live compare (see above).
pub fn lost_condition_materialized() -> u64 {
    LOST_CONDITION_MATERIALIZED.load(std::sync::atomic::Ordering::Relaxed)
}

fn materialize_pending(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    scratch: &mut Scratch,
    state: Option<&PendingCmp>,
    kind: CompareKind,
    addr: u64,
) -> Operand {
    let (mut lhs, mut rhs, at) = match state {
        Some(c) => (c.lhs.clone(), c.rhs.clone(), Some(c.at)),
        None => {
            // No live compare reached this consumer. The BRANCH is real (it was
            // decoded here) and its test is not, which is exactly the
            // wrong-logic class — so the operand is the explicit unknown, never
            // a plausible-looking literal, and the event is counted.
            LOST_CONDITION_MATERIALIZED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (Operand::Unknown, Operand::imm(0), None)
        }
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

pub fn lift_x86_64(
    bytes: &[u8],
    base_rva: u64,
    max_insns: usize,
    abi: X86Abi,
) -> Option<LiftResult> {
    // IP = the method's RVA, NOT 0: near_branch_target() (and insn.ip())
    // are then real RVAs. With IP=0 every BACKWARD call/branch target
    // underflows to a huge wrapped address and renders as
    // `fn_ffffffffffffff..` — the compiler calls earlier helper methods all
    // the time. (The disassembler/x86.rs path already uses with_ip.)
    let mut decoder = Decoder::with_ip(64, bytes, base_rva, DecoderOptions::NONE);
    let mut insns: Vec<Insn> = Vec::new();
    let mut addresses: Vec<u64> = Vec::new();
    let mut pending: Option<PendingCmp> = None;
    // `ucomiss`/`comisd` also write EFLAGS, so they need their own slot: the
    // next Jcc must read the FP comparison, not a stale integer one (and vice
    // versa). Without this, every `ucomisd xmm0, xmm1; ja` rendered as
    // `if (? != 0)` — all FP ordering tests in the binary were lost.
    let mut pending_fp: Option<PendingCmp> = None;
    // `bt`/`bts`/`btr`/`btc r/m, #bit` leave their result in **CF only**. A
    // consumer that tests CF (`jb`/`jae`, and `setb`/`setae`) reads exactly
    // that bit, so the test is kept here as (value, bit index) and turned into
    // a real single-bit condition at the consumer. The index is an Operand, not
    // a constant: `bt eax, ecx` takes its bit from a register, and the old
    // constant-only form dropped those entirely.
    let mut pending_bit: Option<(Operand, BitIndex)> = None;
    // The carry `ADC`/`SBB` consume, as (subtract, lhs, rhs) of the previous
    // flag-writing ADD/SUB/CMP/INC/DEC.
    //
    // x86 spells 64-bit addition as `add`/`adc` and 64-bit subtraction as
    // `sub`/`sbb`, and every `long`/`ulong`/`DateTime`/decimal/big-integer
    // operation in a managed binary is built from those pairs — so without the
    // carry, `adc` and `sbb` lose the high word of the whole operation. Note
    // x86's CF after a SUBTRACT is the BORROW (`lhs < rhs`), the opposite of
    // A32/A64's NOT-borrow, so the two lifters must materialize it differently.
    let mut pending_carry: Option<(bool, Operand, Operand)> = None;
    // XMM registers written since the previous call (ABI float arguments).
    // The arm64 and arm32 lifters both honour this contract; x86 did not, so
    // FP arguments were absent from every call's operand list and the ABI
    // scrub pass then deleted the FP computation feeding them.
    let mut fp_written: BTreeSet<String> = BTreeSet::new();
    let mut scratch = Scratch::new();

    let cap = max_insns.max(16_384);
    while decoder.can_decode() && insns.len() < cap {
        let insn = decoder.decode();
        if insn.code() == iced_x86::Code::INVALID {
            break;
        }
        let addr = insn.ip();
        // NOTE: addresses must stay parallel to insns — pushed after the
        // match below, once per EMITTED IR insn (multi-insn groups like
        // Compare+CondJump share the address; skipped native insns get none).

        let mem = || -> Option<Operand> {
            if insn.memory_base() == Register::None
                && insn.memory_index() == Register::None
                && !insn.is_ip_rel_memory_operand()
            {
                return None;
            }
            Some(memory_operand(&insn, abi))
        };

        // Operand accessor. INDEXED, not binary: the previous
        // `if n == 0 { op0 } else { op1 }` made `op_at(2)` read OPERAND 1, so
        // every three-operand form that names an immediate last multiplied by
        // the wrong thing — `imul eax, ecx, 0x10` lifted to `eax = ecx * ecx`,
        // a silently wrong product from operands that were both real. The
        // guard keeps `op_kind`/`op_register` in range (they debug-assert
        // otherwise), and every register goes through `gpr` so a width-specific
        // spelling (`al`, `ymm3`) cannot name a register nothing defines.
        let op_at = |n: usize| -> Option<Operand> {
            if (n as u32) >= insn.op_count() {
                return None;
            }
            let i = n as u32;
            match insn.op_kind(i) {
                OpKind::Register => Some(Operand::reg(gpr(insn.op_register(i), abi))),
                OpKind::Immediate8 => Some(Operand::imm(insn.immediate8() as i8 as i64)),
                OpKind::Immediate8to16 => Some(Operand::imm(insn.immediate8to16() as i64)),
                OpKind::Immediate8to32 => Some(Operand::imm(insn.immediate8to32() as i64)),
                OpKind::Immediate8to64 => Some(Operand::imm(insn.immediate8to64())),
                OpKind::Immediate16 => Some(Operand::imm(insn.immediate16() as i16 as i64)),
                OpKind::Immediate32 => Some(Operand::imm(insn.immediate32() as i32 as i64)),
                OpKind::Immediate32to64 => Some(Operand::imm(insn.immediate32to64())),
                OpKind::Immediate64 => Some(Operand::imm(insn.immediate64() as i64)),
                OpKind::Memory => mem(),
                _ => None,
            }
        };

        // x86 has NO instruction that re-establishes a pending compare for
        // you — a Jcc reads whatever the LAST flag writer left. Keeping a
        // compare pending across a later flag-writing instruction therefore
        // invented conditions that are not in the binary (`cmp a,b; add eax,1;
        // jne` rendered the jne as `a == b`). This is the mirror image of the
        // arm64 rule in lift.rs: arm64 must PIN a compare across
        // flag-preserving instructions, x86 must DROP it across flag-writing
        // ones, because here nothing re-arms it. A compare survives only
        // instructions PROVEN not to touch EFLAGS (see `preserves_flags`), so
        // a miss degrades to a missing condition and never to a wrong one.
        // Branches/calls/returns read the flags without writing them, so they
        // are excluded by flow control rather than by mnemonic.
        let control = matches!(
            insn.flow_control(),
            FlowControl::ConditionalBranch
                | FlowControl::UnconditionalBranch
                | FlowControl::IndirectBranch
                | FlowControl::Call
                | FlowControl::IndirectCall
                | FlowControl::Return
        );
        // READ the carry that the PREVIOUS flag-writing instruction left
        // before any of that state is cleared: the clear below is keyed on the
        // CURRENT mnemonic, and `adc`/`sbb` are themselves flag writers, so
        // consulting the tracker afterwards would always find it empty.
        let carry_in = pending_carry.clone();
        if !control && !preserves_flags(insn.mnemonic()) {
            pending = None;
            // the FP comparison is consumed from EFLAGS just like the integer
            // one, so a flag writer kills it too
            pending_fp = None;
            // and so does a bit test: its CF is not the previous compare's
            pending_bit = None;
            pending_carry = None;
        }

        match insn.mnemonic() {
            Mnemonic::Mov | Mnemonic::Lea | Mnemonic::Movzx | Mnemonic::Movsx
            | Mnemonic::Movsxd => {
                if let (Some(d), Some(s)) = (op_at(0), op_at(1)) {
                    insns.push(Insn::move_(d, s, addr));
                }
            }
            // The bitwise SIMD forms are exact element-wise operations, so the
            // shared integer ops model them (no lane reinterpretation).
            // `xorps xmm0, xmm0` — the standard way to zero one — hits the
            // reg,reg→0 idiom below. These are dense in FP-heavy code and all
            // of them previously fell through to the catch-all.
            Mnemonic::Add | Mnemonic::Sub | Mnemonic::And | Mnemonic::Or | Mnemonic::Xor
            | Mnemonic::Inc | Mnemonic::Dec
            | Mnemonic::Xorps | Mnemonic::Xorpd | Mnemonic::Pxor
            | Mnemonic::Andps | Mnemonic::Andpd | Mnemonic::Pand
            | Mnemonic::Orps | Mnemonic::Orpd | Mnemonic::Por => {
                let d = op_at(0);
                // `x xor x` → constant zero for EVERY xor form. The integer
                // `xor eax, eax` was handled here; the SIMD forms
                // (`xorps`/`xorpd`/`pxor` — how a compiler zeroes an XMM
                // register) were NOT, despite this comment previously claiming
                // they were, so they fell through to a real
                // `Arith(Xor) vf0, vf0, vf0`: SSA has no version to give either
                // read of `vf0` and the register renders undefined (`vf0` was
                // 513 of the residual undefined locals). The AVX three-operand
                // twins are folded by the same test.
                let two_op = insn.op_count() == 2
                    && matches!(
                        insn.mnemonic(),
                        Mnemonic::Xor | Mnemonic::Xorps | Mnemonic::Xorpd | Mnemonic::Pxor
                    )
                    && insn.op0_kind() == OpKind::Register
                    && insn.op1_kind() == OpKind::Register
                    && insn.op0_register() == insn.op1_register();
                let three_op = insn.op_count() == 3
                    && matches!(
                        insn.mnemonic(),
                        Mnemonic::Vxorps | Mnemonic::Vxorpd | Mnemonic::Vpxor
                    )
                    && insn.op0_kind() == OpKind::Register
                    && insn.op1_kind() == OpKind::Register
                    && insn.op2_kind() == OpKind::Register
                    && insn.op0_register() == insn.op1_register()
                    && insn.op1_register() == insn.op2_register();
                if two_op || three_op {
                    let zp = d.unwrap_or(Operand::Unknown);
                    insns.push(Insn::move_(zp.clone(), Operand::imm(0), addr));
                    // The zero this just wrote sets ZF, so a following `je`
                    // reads THIS, not an older compare. `xorps xmm0,xmm0; jne`
                    // is not a real sequence in this compiler's output, but the
                    // rule costs nothing and keeps the state machine uniform.
                    if insn.mnemonic() == Mnemonic::Xor {
                        pending = Some(PendingCmp {
                            lhs: zp,
                            rhs: Operand::imm(0),
                            at: insns.len(),
                        });
                    }
                    // NOTE: every `continue` here must fill the address
                    // array first — the fill below is skipped otherwise and
                    // the NEXT native insn's address back-fills this slot.
                    // Branch targets pointing at this (xor-idiom) insn then
                    // fail to resolve → end-sentinel → collapsed bodies.
                    while addresses.len() < insns.len() {
                        addresses.push(addr);
                    }
                    continue;
                }
                // stack adjustments
                if matches!(&d, Some(Operand::Register(n)) if n == "sp") {
                    // The immediate is read through `op_at`, NOT through
                    // `op1_kind` with a two-arm match. `add rsp, 0x28` and
                    // `sub rsp, 0x28` are encoded with an IMM8 that
                    // SIGN-EXTENDS to 64 bits, so iced reports the operand kind
                    // as `Immediate8to64` — which matched neither the
                    // `Immediate8` nor the `Immediate32` arm, fell to `_ => 0`,
                    // and made EVERY x64 frame adjustment `ShiftStack(0)`.
                    //
                    // A frame delta of zero is worse than a missing one: the
                    // stack analyzer then models no frame at all (so `sp`-slot
                    // accesses get no offsets) and `recover_tail_calls` never
                    // sees `torn_down`, so every x64 tail call stayed an
                    // "indirect jump (dispatch not recovered)".
                    let v = match op_at(1) {
                        Some(Operand::Immediate(v)) => v,
                        _ => 0,
                    };
                    let is_add = mnemonic_is_add(&insn);
                    insns.push(Insn::new(
                        Op::ShiftStack(if is_add { v } else { -v }),
                        vec![],
                        addr,
                    ));
                } else {
                    let kind = match insn.mnemonic() {
                        Mnemonic::Add | Mnemonic::Inc => ArithKind::Add,
                        Mnemonic::Sub | Mnemonic::Dec => ArithKind::Sub,
                        Mnemonic::And | Mnemonic::Andps | Mnemonic::Andpd | Mnemonic::Pand => {
                            ArithKind::And
                        }
                        Mnemonic::Or | Mnemonic::Orps | Mnemonic::Orpd | Mnemonic::Por => {
                            ArithKind::Or
                        }
                        _ => ArithKind::Xor,
                    };
                    let a = d.unwrap_or(Operand::Unknown);
                    // inc/dec are the single-operand forms: `dec ecx` is
                    // `ecx -= 1`
                    let bb = if matches!(insn.mnemonic(), Mnemonic::Inc | Mnemonic::Dec) {
                        Operand::imm(1)
                    } else {
                        op_at(1).unwrap_or(Operand::Unknown)
                    };
                    if matches!(a, Operand::Memory { .. }) {
                        emit_rmw_arith(&mut insns, &mut scratch, a, kind, bb, addr);
                        // the loaded temp is not addressable here, so no carry
                        // is published for the memory form rather than a wrong
                        // one (`(mem + src) < mem` would name the memory twice)
                        pending_carry = None;
                    } else {
                        // THREE operands: `Op::Arith` is rendered as
                        // `operands[1] OP operands[2]` with the destination in
                        // slot 0 (see `def_expr_inner`), so the in-place form
                        // has to name the destination as its own first source.
                        // A two-operand `vec![a, b]` renders `a = b OP <?>` —
                        // the real source is consumed and a fabricated
                        // `Unknown` takes its place, which is how `add eax, ecx`
                        // came out as `eax = ecx + <?>`.
                        insns.push(Insn::new(
                            Op::Arith(kind),
                            vec![a.clone(), a.clone(), bb.clone()],
                            addr,
                        ));
                        // CF out of ADD = `(a + b) < a`; CF out of SUB = borrow
                        // = `a < b`. Both are exactly what ADC/SBB read.
                        pending_carry = match kind {
                            ArithKind::Add => Some((false, a.clone(), bb.clone())),
                            ArithKind::Sub => Some((true, a.clone(), bb.clone())),
                            _ => None,
                        };
                        // An integer arithmetic instruction sets the flags from
                        // its RESULT as well: `sub $1,%ecx; je L` really is
                        // `if (ecx - 1 == 0)`, and ZF comes from the subtraction,
                        // not from any earlier `cmp`.
                        //
                        // Only `Cmp`/`Test` published a pending compare, so the
                        // arithmetic form left the Jcc with nothing to read and
                        // it rendered `if (? != 0)` — the `while (__unknown !=
                        // 0)` loops and their unrecovered bodies. The comparison
                        // is against ZERO because that is what ZF means, and
                        // `at` pins the register AFTER the arith so a
                        // redefinition before the Jcc still reads the result.
                        //
                        // The memory form (`sub dword [m], 1`) is deliberately
                        // left alone: pinning a `Memory` operand cannot see an
                        // intervening store (a `mov` preserves flags, so the
                        // store would not clear the pending compare), and a
                        // WRONG condition is worse than a missing one.
                        if matches!(
                            insn.mnemonic(),
                            Mnemonic::Add
                                | Mnemonic::Sub
                                | Mnemonic::And
                                | Mnemonic::Or
                                | Mnemonic::Xor
                                | Mnemonic::Inc
                                | Mnemonic::Dec
                        ) {
                            pending = Some(PendingCmp {
                                lhs: a,
                                rhs: Operand::imm(0),
                                at: insns.len(),
                            });
                        }
                    }
                }
            }
            // ---- carry-consuming arithmetic ---------------------------------
            // `adc dst, src` = dst + src + CF ; `sbb dst, src` = dst - src - CF.
            // The carry is taken from the previous flag-writing ADD/SUB/CMP
            // (see `carry_in`) and turned into a real integer 0/1 value with a
            // Compare + Select, so the widened sum stays exact instead of the
            // whole instruction degrading to an `__Adc(…)` stub.
            Mnemonic::Adc | Mnemonic::Sbb => {
                let dst = op_at(0).unwrap_or(Operand::Unknown);
                let src = op_at(1).unwrap_or(Operand::Unknown);
                let subtract = insn.mnemonic() == Mnemonic::Sbb;
                match carry_in {
                    Some((src_sub, lhs, rhs)) => {
                        let cf = materialize_x86_carry(
                            &mut insns,
                            &mut scratch,
                            src_sub,
                            lhs,
                            rhs,
                            addr,
                        );
                        let kind = if subtract { ArithKind::Sub } else { ArithKind::Add };
                        if matches!(dst, Operand::Memory { .. }) {
                            emit_rmw_arith(&mut insns, &mut scratch, dst.clone(), kind, src, addr);
                            emit_rmw_arith(&mut insns, &mut scratch, dst, kind, cf, addr);
                        } else {
                            insns.push(Insn::new(
                                Op::Arith(kind),
                                vec![dst.clone(), dst.clone(), src],
                                addr,
                            ));
                            insns.push(Insn::new(
                                Op::Arith(kind),
                                vec![dst.clone(), dst, cf],
                                addr,
                            ));
                        }
                    }
                    None => {
                        // No live carry: keep every operand as a VALUE rather
                        // than inventing the carry the hardware did not use.
                        let mut ops = vec![dst];
                        if let Some(o) = op_at(1) {
                            ops.push(o);
                        }
                        insns.push(Insn::new(
                            Op::Simd(format!("__{:?}_nocarry", insn.mnemonic())),
                            ops,
                            addr,
                        ));
                    }
                }
            }
            Mnemonic::Imul if insn.op_count() == 3 => {
                let a = op_at(1).unwrap_or(Operand::Unknown);
                let b = op_at(2).unwrap_or(Operand::Unknown);
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Mul),
                    vec![op_at(0).unwrap_or(Operand::Unknown), a, b],
                    addr,
                ));
            }
            // `sal` is the SAME operation as `shl` (arithmetic and logical left
            // shift are identical — the sign is only a mnemonic distinction);
            // iced keeps them as separate `Mnemonic` variants, so the `/4`
            // opcode reached the catch-all as an `__Sal(...)` stub whenever the
            // compiler emitted the encoding that decodes to `sal` rather than
            // `shl`. Mapping it here restores the shift instead of naming it.
            Mnemonic::Shl | Mnemonic::Sal | Mnemonic::Shr | Mnemonic::Sar => {
                let kind = match insn.mnemonic() {
                    Mnemonic::Shl | Mnemonic::Sal => ArithKind::Shl,
                    Mnemonic::Shr | Mnemonic::Sar => ArithKind::Shr,
                    _ => unreachable!(),
                };
                let a = op_at(0).unwrap_or(Operand::Unknown);
                let n = op_at(1).unwrap_or(Operand::imm(1));
                if matches!(a, Operand::Memory { .. }) {
                    emit_rmw_arith(&mut insns, &mut scratch, a, kind, n, addr);
                } else {
                    insns.push(Insn::new(Op::Arith(kind), vec![a.clone(), a, n], addr));
                }
                // SAR sign-fills and SHR zero-fills. `Shr` renders as C# `>>`,
                // which is arithmetic on a signed operand — the same
                // convention the arm64 lifter already uses for both LSR and
                // ASR, so the two share one shift kind rather than inventing a
                // second one here.
            }
            Mnemonic::Rol | Mnemonic::Ror => {
                let kind = if insn.mnemonic() == Mnemonic::Rol {
                    ArithKind::Rol
                } else {
                    ArithKind::Ror
                };
                let a = op_at(0).unwrap_or(Operand::Unknown);
                let n = op_at(1).unwrap_or(Operand::imm(1));
                if matches!(a, Operand::Memory { .. }) {
                    emit_rmw_arith(&mut insns, &mut scratch, a, kind, n, addr);
                } else {
                    insns.push(Insn::new(Op::Arith(kind), vec![a.clone(), a, n], addr));
                }
            }
            // ---- BMI1 / BMI2 ------------------------------------------------
            //
            // Every one of these is an EXACT integer bit operation, so it is
            // modelled rather than left as a `__Mnemonic(…)` stub. They are what
            // a -march=native build emits for `BitOperations.TrailingZeroCount`,
            // hash mixing, bitmask iteration and byte-swap helpers.
            //
            // `shlx`/`shrx`/`sarx`/`rorx` are the FLAG-FREE forms: same result
            // as `shl`/`shr`/`sar`/`ror`, but the count is an explicit operand 2
            // and EFLAGS is untouched. `Shr` renders as C# `>>`, which is
            // arithmetic on a signed operand — the same convention the arm64
            // lifter already uses for both LSR and ASR (see lift.rs) — so
            // `shrx`/`sarx` share it rather than inventing a second shift kind.
            Mnemonic::Shlx | Mnemonic::Shrx | Mnemonic::Sarx | Mnemonic::Rorx => {
                let kind = match insn.mnemonic() {
                    Mnemonic::Shlx => ArithKind::Shl,
                    Mnemonic::Rorx => ArithKind::Ror,
                    _ => ArithKind::Shr,
                };
                let d = op_at(0).unwrap_or(Operand::Unknown);
                let s = op_at(1).unwrap_or(Operand::Unknown);
                let n = op_at(2).unwrap_or(Operand::Unknown);
                insns.push(Insn::new(Op::Arith(kind), vec![d, s, n], addr));
            }
            // `andn dst, src1, src2` = ~src1 & src2.
            Mnemonic::Andn => {
                let d = op_at(0).unwrap_or(Operand::Unknown);
                let a = op_at(1).unwrap_or(Operand::Unknown);
                let b = op_at(2).unwrap_or(Operand::Unknown);
                let neg = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Xor),
                    vec![neg.clone(), a, Operand::imm(-1)],
                    addr,
                ));
                insns.push(Insn::new(Op::Arith(ArithKind::And), vec![d, neg, b], addr));
            }
            // `blsi` = src & -src (isolate the lowest set bit);
            // `blsr` = src & (src - 1) (clear it);
            // `blsmsk` = src ^ (src - 1) (mask up to and including it).
            // All three are exact, and the trio is how a compiler lowers
            // `x & -x`, `x & (x - 1)` and popcount-style scanning.
            Mnemonic::Blsi | Mnemonic::Blsr | Mnemonic::Blsmsk => {
                let d = op_at(0).unwrap_or(Operand::Unknown);
                let src = op_at(1).unwrap_or(Operand::Unknown);
                if insn.mnemonic() == Mnemonic::Blsi {
                    // A real negation, so it stays exact at the most negative
                    // value (`-INT_MIN` wraps to itself, which is what the
                    // hardware produces).
                    let neg = scratch.fresh();
                    insns.push(Insn::new(Op::Negate, vec![neg.clone(), src.clone()], addr));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::And),
                        vec![d, src, neg],
                        addr,
                    ));
                } else {
                    let dec = scratch.fresh();
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Sub),
                        vec![dec.clone(), src.clone(), Operand::imm(1)],
                        addr,
                    ));
                    let kind = if insn.mnemonic() == Mnemonic::Blsr {
                        ArithKind::And
                    } else {
                        ArithKind::Xor
                    };
                    insns.push(Insn::new(Op::Arith(kind), vec![d, src, dec], addr));
                }
            }
            // `bextr dst, src, control`: control[7:0] is the start bit and
            // control[15:8] the length, so the result is
            // `(src >> start) & ((1 << length) - 1)`.
            Mnemonic::Bextr => {
                let d = op_at(0).unwrap_or(Operand::Unknown);
                let src = op_at(1).unwrap_or(Operand::Unknown);
                let ctl = op_at(2).unwrap_or(Operand::Unknown);
                let start = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![start.clone(), ctl.clone(), Operand::imm(0xFF)],
                    addr,
                ));
                let len0 = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shr),
                    vec![len0.clone(), ctl, Operand::imm(8)],
                    addr,
                ));
                let len = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![len.clone(), len0, Operand::imm(0xFF)],
                    addr,
                ));
                let shifted = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shr),
                    vec![shifted.clone(), src, start],
                    addr,
                ));
                let span = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shl),
                    vec![span.clone(), Operand::imm(1), len],
                    addr,
                ));
                let mask = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Sub),
                    vec![mask.clone(), span, Operand::imm(1)],
                    addr,
                ));
                insns.push(Insn::new(Op::Arith(ArithKind::And), vec![d, shifted, mask], addr));
            }
            // `bzhi dst, src, index`: `src` with every bit at or above `index`
            // cleared — exactly `src & ((1 << index) - 1)`.
            Mnemonic::Bzhi => {
                let d = op_at(0).unwrap_or(Operand::Unknown);
                let src = op_at(1).unwrap_or(Operand::Unknown);
                let idx = op_at(2).unwrap_or(Operand::Unknown);
                let span = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shl),
                    vec![span.clone(), Operand::imm(1), idx],
                    addr,
                ));
                let mask = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Sub),
                    vec![mask.clone(), span, Operand::imm(1)],
                    addr,
                ));
                insns.push(Insn::new(Op::Arith(ArithKind::And), vec![d, src, mask], addr));
            }
            // `mulx dst_hi, dst_lo, src`: `rdx * src` into a pair.
            //
            // The LOW half is exact in 32-bit arithmetic. The HIGH half is the
            // widening product, which this model has no op for, so it is
            // published as a named intrinsic that consumes the same operands —
            // undefined and VISIBLE rather than left holding a stale value
            // (which is how a dropped widening multiply turns into a wrong
            // answer instead of a missing one).
            Mnemonic::Mulx => {
                let hi = op_at(0).unwrap_or(Operand::Unknown);
                let lo = op_at(1).unwrap_or(Operand::Unknown);
                let src = op_at(2).unwrap_or(Operand::Unknown);
                let acc = Operand::reg(gpr(Register::RDX, abi));
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Mul),
                    vec![lo, acc.clone(), src.clone()],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Simd("__mulx_hi".into()),
                    vec![hi, acc, src],
                    addr,
                ));
            }
            Mnemonic::Not => {
                // NOT = bitwise complement: x ^ -1. (Op::Not is a *boolean*
                // not and renders as `!`, which would corrupt integer data —
                // the arm64 path uses this xor idiom too.)
                let d = op_at(0).unwrap_or(Operand::Unknown);
                if matches!(d, Operand::Memory { .. }) {
                    emit_rmw_arith(&mut insns, &mut scratch, d, ArithKind::Xor, Operand::imm(-1), addr);
                } else {
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Xor),
                        vec![d.clone(), d.clone(), Operand::imm(-1)],
                        addr,
                    ));
                }
            }
            Mnemonic::Neg => {
                let d = op_at(0).unwrap_or(Operand::Unknown);
                if matches!(d, Operand::Memory { .. }) {
                    emit_rmw_unary(&mut insns, &mut scratch, d, addr, Op::Negate);
                } else {
                    insns.push(Insn::new(
                        Op::Negate,
                        vec![d.clone(), d.clone()],
                        addr,
                    ));
                }
            }
            Mnemonic::Imul if insn.op_count() == 2 => {
                // two-op form: dest *= src
                let a = op_at(0).unwrap_or(Operand::Unknown);
                let b = op_at(1).unwrap_or(Operand::Unknown);
                if matches!(a, Operand::Memory { .. }) {
                    emit_rmw_arith(&mut insns, &mut scratch, a, ArithKind::Mul, b, addr);
                } else {
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Mul),
                        vec![a.clone(), a, b],
                        addr,
                    ));
                }
            }
            Mnemonic::Mul | Mnemonic::Imul | Mnemonic::Idiv | Mnemonic::Div => {
                // one-op forms act on rdx:rax — model result into x0.
                //
                // The ONE-operand `imul r/m` (implicit `rdx:rax` operand)
                // lands here: the 2- and 3-operand forms above have arms
                // guarded on `op_count`, so a 1-operand `imul` fell through
                // them and was never modelled — the product's low half was
                // left undefined. Signed and unsigned agree on the low 64
                // bits, so `Mul`'s kind is exact for it.
                let kind = match insn.mnemonic() {
                    Mnemonic::Mul | Mnemonic::Imul => ArithKind::Mul,
                    Mnemonic::Idiv | Mnemonic::Div => ArithKind::Div,
                    _ => ArithKind::Rem,
                };
                let a = Operand::reg(RET_REG);
                let b = op_at(0).unwrap_or(Operand::Unknown);
                insns.push(Insn::new(
                    Op::Arith(kind),
                    vec![Operand::reg(RET_REG), a.clone(), b.clone()],
                    addr,
                ));
                // `div`/`idiv` leave the REMAINDER in the high half as well,
                // and that is the whole of x86's `%`: the compiler emits
                // `cqo; idiv` and then reads `rdx`. Modelling only the
                // quotient left every modulo reading whatever the high half
                // happened to hold — a wrong value, not a missing one.
                //
                // `mul` needs no second result here: its high half is the
                // widening product, which the `umull`/`mulx` treatment leaves
                // undefined rather than fabricating from 32-bit ops.
                if matches!(insn.mnemonic(), Mnemonic::Idiv | Mnemonic::Div) {
                    let hi = Operand::reg(gpr(Register::RDX, abi));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Rem),
                        vec![hi, a, b],
                        addr,
                    ));
                }
            }
            // Every `setcc`, not just the ten ordered ones. `setp`/`setnp`
            // (and `sets`/`setns`) were missing, and a one-operand instruction
            // with no arm fell through to the catch-all, which emitted NOTHING
            // — so the byte kept whatever it held before and the `(a != b)`
            // that followed read a stale value. `setp`/`setnp` are how a NaN
            // test compiles after `ucomiss`/`comisd`, so they are not exotic.
            // `bswap r32/r64` reverses the byte order of the WHOLE register.
            // It is exact and it WRITES, and it used to reach the catch-all,
            // whose one-operand case emitted nothing — so the register kept its
            // UN-swapped value and every subsequent read of it was silently
            // wrong. The width comes from the operand, because `bswap r64`
            // reverses eight bytes and not four.
            Mnemonic::Bswap => {
                let d = op_at(0).unwrap_or(Operand::Unknown);
                if matches!(d, Operand::Register(_)) {
                    let bytes = if insn.op0_register().size() >= 8 { 8 } else { 4 };
                    let v = d.clone();
                    // OR the bytes back together, each moved to its mirrored
                    // position: byte `i` becomes byte `bytes - 1 - i`.
                    let mut acc: Option<Operand> = None;
                    for i in 0..bytes {
                        let mut piece = v.clone();
                        if i > 0 {
                            let s = scratch.fresh();
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::Shr),
                                vec![s.clone(), piece, Operand::imm((i * 8) as i64)],
                                addr,
                            ));
                            piece = s;
                        }
                        let m = scratch.fresh();
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::And),
                            vec![m.clone(), piece, Operand::imm(0xFF)],
                            addr,
                        ));
                        if i + 1 != bytes {
                            let s = scratch.fresh();
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::Shl),
                                vec![s.clone(), m, Operand::imm(((bytes - 1 - i) * 8) as i64)],
                                addr,
                            ));
                            piece = s;
                        } else {
                            piece = m;
                        }
                        acc = Some(match acc {
                            None => piece,
                            Some(a) => {
                                let t = scratch.fresh();
                                insns.push(Insn::new(
                                    Op::Arith(ArithKind::Or),
                                    vec![t.clone(), a, piece],
                                    addr,
                                ));
                                t
                            }
                        });
                    }
                    if let Some(a) = acc {
                        insns.push(Insn::move_(d, a, addr));
                    }
                }
            }
            Mnemonic::Sete
            | Mnemonic::Setne
            | Mnemonic::Setl
            | Mnemonic::Setle
            | Mnemonic::Setg
            | Mnemonic::Setge
            | Mnemonic::Setb
            | Mnemonic::Setbe
            | Mnemonic::Seta
            | Mnemonic::Setae
            | Mnemonic::Setp
            | Mnemonic::Setnp
            | Mnemonic::Sets
            | Mnemonic::Setns
                if insn.op_count() >= 1 =>
            {
                // materialize flag condition into a byte register
                let cond = cond_kind(
                    if insn.condition_code() == ConditionCode::None {
                        match insn.mnemonic() {
                            Mnemonic::Sete => ConditionCode::e,
                            Mnemonic::Setne => ConditionCode::ne,
                            Mnemonic::Setl => ConditionCode::l,
                            Mnemonic::Setle => ConditionCode::le,
                            Mnemonic::Setg => ConditionCode::g,
                            Mnemonic::Setge => ConditionCode::ge,
                            Mnemonic::Setb => ConditionCode::b,
                            Mnemonic::Setbe => ConditionCode::be,
                            Mnemonic::Seta => ConditionCode::a,
                            Mnemonic::Setp => ConditionCode::p,
                            Mnemonic::Setnp => ConditionCode::np,
                            Mnemonic::Sets => ConditionCode::s,
                            Mnemonic::Setns => ConditionCode::ns,
                            _ => ConditionCode::ae,
                        }
                    } else {
                        insn.condition_code()
                    },
                );
                // FP comparison wins when it was the last one set. The flags are
                // READ, not consumed — several setcc instructions can follow one
                // compare — and any flag writer clears the pending state itself.
                let s = match bit_scratch_if_carry(
                    &mut insns,
                    &mut scratch,
                    pending_bit.clone(),
                    insn.condition_code(),
                    addr,
                ) {
                    Some(s) => s,
                    None => materialize_pending(
                        &mut insns,
                        &mut addresses,
                        &mut scratch,
                        pending_fp.as_ref().or(pending.as_ref()),
                        cond,
                        addr,
                    ),
                };
                // store scratch as 0/1 into the destination slot
                if let Some(d) = op_at(0) {
                    insns.push(Insn::move_(d, s, addr));
                }
            }
            Mnemonic::Push => {
                insns.push(Insn::new(Op::ShiftStack(-8), vec![], addr));
            }
            Mnemonic::Pop => {
                insns.push(Insn::new(Op::ShiftStack(8), vec![], addr));
            }
            Mnemonic::Movss
            | Mnemonic::Movsd
            | Mnemonic::Movaps
            | Mnemonic::Movapd
            | Mnemonic::Movups
            | Mnemonic::Movupd
            // integer SIMD moves + the GPR↔XMM transfers. `movd`/`movq` are
            // how a float crosses between the two register files; dropping
            // them lost the value on one side of every such conversion.
            | Mnemonic::Movdqu
            | Mnemonic::Movdqa
            | Mnemonic::Movntdq
            | Mnemonic::Lddqu
            | Mnemonic::Movd
            | Mnemonic::Movq
                // `movsd` names BOTH the scalar-double SSE move and the string
                // primitive, so the string form has to be excluded here or it
                // consumes this arm, finds no explicit operand (its operands are
                // the implicit `[rdi]`/`[rsi]`) and emits nothing at all.
                if !is_string_op(&insn) =>
            {
                // SSE moves between xmm registers / memory
                let d = sse_op(&insn, 0, abi);
                let src = sse_op(&insn, 1, abi);
                if let (Some(d), Some(s)) = (d, src) {
                    insns.push(Insn::move_(d, s, addr));
                }
            }
            // ---- string primitives ------------------------------------------
            //
            // `movs`/`stos`/`lods`/`scas`/`cmps` address their operands
            // IMPLICITLY (rdi/rsi and, for some, the accumulator). iced reports
            // those as `OpKind::MemoryESRDI`/`MemorySegRSI`, which `op_at` does
            // not map, so an inline `rep movsb`/`rep stosq` — how a struct copy
            // or an array clear compiles — fell through the catch-all and was
            // DROPPED, losing its memory write, both pointer updates and its
            // counter.
            //
            // The one-element data movement is emitted exactly as the ISA
            // defines it, plus the EXACT end state a `rep` prefix leaves on the
            // fall-through path: the pointers advance by `count * element` and
            // the counter reaches zero. DF is assumed CLEAR (forward), which is
            // what the ABI requires of every compiler-emitted sequence.
            //
            // `scas`/`cmps` also SET FLAGS from their comparison, and with
            // `repe`/`repne` they stop EARLY — so neither their pointer end
            // state nor their condition is expressible without a loop. They are
            // emitted as named intrinsics that consume the same operands, with
            // the flags left clobbered (they are not in `preserves_flags`), so a
            // following branch degrades to a MISSING condition rather than an
            // invented one.
            Mnemonic::Movsb | Mnemonic::Movsw | Mnemonic::Movsd | Mnemonic::Movsq
            | Mnemonic::Stosb | Mnemonic::Stosw | Mnemonic::Stosd | Mnemonic::Stosq
            | Mnemonic::Lodsb | Mnemonic::Lodsw | Mnemonic::Lodsd | Mnemonic::Lodsq
            | Mnemonic::Scasb | Mnemonic::Scasw | Mnemonic::Scasd | Mnemonic::Scasq
            | Mnemonic::Cmpsb | Mnemonic::Cmpsw | Mnemonic::Cmpsd | Mnemonic::Cmpsq => {
                let di = Operand::reg(gpr(Register::RDI, abi));
                let si = Operand::reg(gpr(Register::RSI, abi));
                let cx = Operand::reg(gpr(Register::RCX, abi));
                let acc = Operand::reg(RET_REG);
                let elem: i64 = match insn.mnemonic() {
                    Mnemonic::Movsb | Mnemonic::Stosb | Mnemonic::Lodsb | Mnemonic::Scasb
                    | Mnemonic::Cmpsb => 1,
                    Mnemonic::Movsw | Mnemonic::Stosw | Mnemonic::Lodsw | Mnemonic::Scasw
                    | Mnemonic::Cmpsw => 2,
                    Mnemonic::Movsd | Mnemonic::Stosd | Mnemonic::Lodsd | Mnemonic::Scasd
                    | Mnemonic::Cmpsd => 4,
                    _ => 8,
                };
                let is_mov = matches!(
                    insn.mnemonic(),
                    Mnemonic::Movsb | Mnemonic::Movsw | Mnemonic::Movsd | Mnemonic::Movsq
                );
                let is_stos = matches!(
                    insn.mnemonic(),
                    Mnemonic::Stosb | Mnemonic::Stosw | Mnemonic::Stosd | Mnemonic::Stosq
                );
                let is_lods = matches!(
                    insn.mnemonic(),
                    Mnemonic::Lodsb | Mnemonic::Lodsw | Mnemonic::Lodsd | Mnemonic::Lodsq
                );
                // The access WIDTH is carried, not left at the pointer-width
                // default: a `movsb` is a one-byte access and must not render
                // as an eight-byte one.
                let slot = |base: Operand| Operand::Memory {
                    base: Box::new(base),
                    index: None,
                    scale: 1,
                    addend: 0,
                    width: elem as u8,
                };
                if is_mov || is_stos || is_lods {
                    if is_mov {
                        insns.push(Insn::move_(slot(di.clone()), slot(si.clone()), addr));
                    } else if is_stos {
                        insns.push(Insn::move_(slot(di.clone()), acc.clone(), addr));
                    } else {
                        insns.push(Insn::move_(acc.clone(), slot(si.clone()), addr));
                    }
                    if insn.has_rep_prefix() {
                        // advance = count * element
                        let step = scratch.fresh();
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Mul),
                            vec![step.clone(), cx.clone(), Operand::imm(elem)],
                            addr,
                        ));
                        if is_mov || is_stos {
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::Add),
                                vec![di.clone(), di.clone(), step.clone()],
                                addr,
                            ));
                        }
                        if is_mov || is_lods {
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::Add),
                                vec![si.clone(), si.clone(), step],
                                addr,
                            ));
                        }
                        insns.push(Insn::move_(cx, Operand::imm(0), addr));
                    }
                } else {
                    let is_scas = matches!(
                        insn.mnemonic(),
                        Mnemonic::Scasb | Mnemonic::Scasw | Mnemonic::Scasd | Mnemonic::Scasq
                    );
                    let mut ops = vec![slot(di.clone())];
                    if is_scas {
                        ops.push(acc);
                    } else {
                        ops.push(slot(si.clone()));
                    }
                    if insn.has_rep_prefix() {
                        ops.push(cx);
                    }
                    insns.push(Insn::new(
                        Op::Simd(if is_scas { "__scas".into() } else { "__cmps".into() }),
                        ops,
                        addr,
                    ));
                }
            }
            Mnemonic::Addsd | Mnemonic::Addss | Mnemonic::Subsd | Mnemonic::Subss
            | Mnemonic::Mulsd | Mnemonic::Mulss | Mnemonic::Divsd | Mnemonic::Divss => {
                // TWO forms reach here:
                //   * SSE   `subss xmm6, xmm7`      — iced reports op_count 2
                //     (dest, src); the DESTINATION is also the first source.
                //   * AVX   `vsubss xmm6, xmm1, xmm7` — op_count 3.
                // Requiring operand 2 unconditionally dropped EVERY 2-operand
                // SSE scalar op with no diagnostic: the instruction vanished
                // from the IR, and when it happened to be a branch target the
                // branch landing slid to the next instruction and the semantic oracle reported a spurious `branch target mismatch`.
                let d = sse_op(&insn, 0, abi);
                let (a, b) = if insn.op_count() >= 3 {
                    (sse_op(&insn, 1, abi), sse_op(&insn, 2, abi))
                } else {
                    // 2-operand: `dst op= src`; the first source is the dest.
                    (d.clone(), sse_op(&insn, 1, abi))
                };
                if let (Some(dd), Some(aa), Some(bb)) = (d, a, b) {
                    let kind = match insn.mnemonic() {
                        Mnemonic::Addsd | Mnemonic::Addss => FloatArithKind::Add,
                        Mnemonic::Subsd | Mnemonic::Subss => FloatArithKind::Sub,
                        Mnemonic::Mulsd | Mnemonic::Mulss => FloatArithKind::Mul,
                        _ => FloatArithKind::Div,
                    };
                    insns.push(Insn::new(
                        Op::FloatArith(kind),
                        vec![dd, aa, bb],
                        addr,
                    ));
                }
            }
            Mnemonic::Sqrtss | Mnemonic::Sqrtsd => {
                // unary FP sqrt: dst = sqrt(src) — same Sqrt intrinsic the
                // arm64 FSQRT path emits (renders as `Sqrt(x)`)
                let d = sse_op(&insn, 0, abi);
                let a = sse_op(&insn, 1, abi);
                if let (Some(dd), Some(aa)) = (d, a) {
                    insns.push(Insn::new(
                        Op::FloatArith(FloatArithKind::Sqrt),
                        vec![dd, aa],
                        addr,
                    ));
                }
            }
            Mnemonic::Cvtsi2ss | Mnemonic::Cvtsi2sd => {
                // int -> float: dst = (float)src. The int may be in a GPR
                // (sse_op falls back to gpr) or memory.
                let d = sse_op(&insn, 0, abi);
                let a = sse_op(&insn, 1, abi);
                if let (Some(dd), Some(aa)) = (d, a) {
                    let conv = if insn.mnemonic() == Mnemonic::Cvtsi2sd {
                        super::ir::Conversion::IntToFloat64
                    } else {
                        super::ir::Conversion::IntToFloat32
                    };
                    insns.push(Insn::new(Op::Convert(conv), vec![dd, aa], addr));
                }
            }
            Mnemonic::Cvttss2si | Mnemonic::Cvttsd2si => {
                // float -> int truncation: dst = (int)src (GPR dest)
                let d = op_at(0);
                let a = sse_op(&insn, 1, abi);
                if let (Some(dd), Some(aa)) = (d, a) {
                    insns.push(Insn::new(
                        Op::Convert(super::ir::Conversion::FloatToInt),
                        vec![dd, aa],
                        addr,
                    ));
                }
            }
            Mnemonic::Lzcnt => {
                // count leading zeros — same CountLeadingZeros intrinsic
                let d = op_at(0);
                let a = op_at(1);
                if let (Some(dd), Some(aa)) = (d, a) {
                    insns.push(Insn::new(Op::Clz, vec![dd, aa], addr));
                }
            }
            Mnemonic::Minsd | Mnemonic::Minss => {
                // min/max map to the same conditional-compare+dst pattern the
                // arm64 FMAX/FMIN path uses (compare scratch, keep first op)
                let d = sse_op(&insn, 0, abi);
                let a = sse_op(&insn, 1, abi);
                if let (Some(dd), Some(aa)) = (d.clone(), a) {
                    insns.push(Insn::new(
                        Op::Compare(CompareKind::LessThan),
                        vec![scratch.fresh(), aa.clone(), dd.clone()],
                        addr,
                    ));
                    insns.push(Insn::move_(dd, aa, addr));
                }
            }
            Mnemonic::Maxsd | Mnemonic::Maxss => {
                let d = sse_op(&insn, 0, abi);
                let a = sse_op(&insn, 1, abi);
                if let (Some(dd), Some(aa)) = (d.clone(), a) {
                    insns.push(Insn::new(
                        Op::Compare(CompareKind::GreaterThan),
                        vec![scratch.fresh(), aa.clone(), dd.clone()],
                        addr,
                    ));
                    insns.push(Insn::move_(dd, aa, addr));
                }
            }
            Mnemonic::Vfmadd132sd | Mnemonic::Vfmadd132ss
            | Mnemonic::Vfmadd213sd | Mnemonic::Vfmadd213ss
            | Mnemonic::Vfmadd231sd | Mnemonic::Vfmadd231ss
            | Mnemonic::Vfmadd132pd | Mnemonic::Vfmadd132ps
            | Mnemonic::Vfmadd213pd | Mnemonic::Vfmadd213ps
            | Mnemonic::Vfmadd231pd | Mnemonic::Vfmadd231ps => {
                // FMA: dst = a*b + c (or permuted); fused multiply-add has no
                // C# operator — MulAcc renders as the same arithmetic.
                let d = sse_op(&insn, 0, abi);
                let a = sse_op(&insn, 1, abi);
                let c = sse_op(&insn, 2, abi);
                if let (Some(dd), Some(aa), Some(cc)) = (d, a, c) {
                    let (mul_a, mul_b, add) = match insn.mnemonic() {
                        Mnemonic::Vfmadd132sd | Mnemonic::Vfmadd132ss
                        | Mnemonic::Vfmadd132pd | Mnemonic::Vfmadd132ps => (aa.clone(), cc.clone(), cc.clone()),
                        Mnemonic::Vfmadd213sd | Mnemonic::Vfmadd213ss
                        | Mnemonic::Vfmadd213pd | Mnemonic::Vfmadd213ps => (aa.clone(), cc.clone(), aa.clone()),
                        _ => (cc.clone(), aa.clone(), aa.clone()),
                    };
                    insns.push(Insn::new(
                        Op::MulAcc { negate_addend: false, is_float: true },
                        vec![dd, mul_a, mul_b, add],
                        addr,
                    ));
                }
            }
            // ---- bit test / modify ------------------------------------------
            // `bt` sets CF from one bit; the modify forms also write that bit
            // back. The mask is a real constant, so `bts`/`btr`/`btc` are
            // exact, and the test is handed to whatever consumes CF.
            Mnemonic::Bt | Mnemonic::Bts | Mnemonic::Btr | Mnemonic::Btc => {
                let src = op_at(0).unwrap_or(Operand::Unknown);
                // The bit position is masked to the OPERAND width by the
                // hardware: 5 bits for a 32-bit destination, 6 for 64-bit.
                let mask: i64 = if reg_is_64(insn.op0_register()) { 63 } else { 31 };
                let bit = match insn.op1_kind() {
                    OpKind::Immediate8 => Some(BitIndex::Constant(
                        (insn.immediate8() as i64) & mask,
                    )),
                    OpKind::Immediate8to16 | OpKind::Immediate8to32 | OpKind::Immediate8to64 => {
                        Some(BitIndex::Constant(insn.immediate8to64() & mask))
                    }
                    // `bt r/m, r` — a live bit index. Only the low byte of the
                    // source register participates, and the value chain is kept
                    // so the shift is exact at the consumer.
                    OpKind::Register => Some(BitIndex::Variable {
                        value: op_at(1).unwrap_or(Operand::Unknown),
                        mask,
                    }),
                    _ => None,
                };
                match (insn.mnemonic(), bit) {
                    (Mnemonic::Bt, Some(b)) => pending_bit = Some((src, b)),
                    (Mnemonic::Bts, Some(BitIndex::Constant(b))) => {
                        let mask = Operand::imm(1i64 << b);
                        if matches!(src, Operand::Memory { .. }) {
                            emit_rmw_arith(&mut insns, &mut scratch, src, ArithKind::Or, mask, addr);
                        } else {
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::Or),
                                vec![src.clone(), src, mask],
                                addr,
                            ));
                        }
                    }
                    (Mnemonic::Btr, Some(BitIndex::Constant(b))) => {
                        let mask = Operand::imm(!(1i64 << b));
                        if matches!(src, Operand::Memory { .. }) {
                            emit_rmw_arith(&mut insns, &mut scratch, src, ArithKind::And, mask, addr);
                        } else {
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::And),
                                vec![src.clone(), src, mask],
                                addr,
                            ));
                        }
                    }
                    (Mnemonic::Btc, Some(BitIndex::Constant(b))) => {
                        let mask = Operand::imm(1i64 << b);
                        if matches!(src, Operand::Memory { .. }) {
                            emit_rmw_arith(&mut insns, &mut scratch, src, ArithKind::Xor, mask, addr);
                        } else {
                            insns.push(Insn::new(
                                Op::Arith(ArithKind::Xor),
                                vec![src.clone(), src, mask],
                                addr,
                            ));
                        }
                    }
                    // A register bit index on a MODIFY form (`bts eax, ecx`): the
                    // mask is dynamic, so the constant-mask arms above cannot
                    // express it. Emit the real shift/mask value pair so the bit
                    // is still written back correctly instead of vanishing.
                    (m, Some(BitIndex::Variable { value: idx, mask })) => {
                        let n = scratch.fresh();
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::And),
                            vec![n.clone(), idx, Operand::imm(mask)],
                            addr,
                        ));
                        let one = scratch.fresh();
                        insns.push(Insn::new(
                            Op::Arith(ArithKind::Shl),
                            vec![one.clone(), Operand::imm(1), n],
                            addr,
                        ));
                        let kind = match m {
                            Mnemonic::Bts => ArithKind::Or,
                            Mnemonic::Btr => ArithKind::And,
                            _ => ArithKind::Xor,
                        };
                        let flipped = if m == Mnemonic::Btr {
                            let inv = scratch.fresh();
                            insns.push(Insn::new(Op::Negate, vec![inv.clone(), one], addr));
                            inv
                        } else {
                            one
                        };
                        if matches!(src, Operand::Memory { .. }) {
                            emit_rmw_arith(&mut insns, &mut scratch, src, kind, flipped, addr);
                        } else {
                            insns.push(Insn::new(
                                Op::Arith(kind),
                                vec![src.clone(), src, flipped],
                                addr,
                            ));
                        }
                    }
                    // a bit index we could not read at all: keep the value chain
                    (_, _) => {
                        let mut ops = vec![src];
                        if let Some(o) = op_at(1) {
                            ops.push(o);
                        }
                        insns.push(Insn::new(
                            Op::Simd(format!("__{:?}", insn.mnemonic())),
                            ops,
                            addr,
                        ));
                    }
                }
            }
            // ---- double-width shifts ----------------------------------------
            // `shld dst, src, #n` = `(dst << n) | (src >> (w - n))` — how the
            // compiler shifts a value wider than a register (128-bit hashes,
            // big-integer code, and every 64-bit multiply expansion on x86).
            // Exact when the count is an immediate; the `cl` form keeps its
            // value through the fallback instead of guessing.
            //
            // The COUNT is the LAST operand: `shld dst, src, imm8` puts the
            // immediate at index 2, not 1. Guarding on `op1_kind` meant no
            // `shld`/`shrd` ever matched this arm and every one of them fell to
            // the generic fallback (`__Shld(a, b)` — a value with a name and no
            // semantics), which is how a 128-bit shift silently became a
            // no-op in the output while still looking plausible.
            Mnemonic::Shld | Mnemonic::Shrd
                if matches!(insn.op2_kind(), OpKind::Immediate8) =>
            {
                let d = op_at(0).unwrap_or(Operand::Unknown);
                let s = op_at(1).unwrap_or(Operand::Unknown);
                let n = insn.immediate8() as i64;
                let width: i64 = if reg_is_64(insn.op0_register()) {
                    64
                } else {
                    32
                };
                let left = insn.mnemonic() == Mnemonic::Shld;
                let (inner_kind, outer_kind) = if left {
                    (ArithKind::Shr, ArithKind::Shl)
                } else {
                    (ArithKind::Shl, ArithKind::Shr)
                };
                if n == 0 {
                    // the destination is unchanged (only the flags are set)
                    insns.push(Insn::move_(d.clone(), d, addr));
                } else {
                    let part_a = scratch.fresh();
                    insns.push(Insn::new(
                        Op::Arith(outer_kind),
                        vec![part_a.clone(), d.clone(), Operand::imm(n)],
                        addr,
                    ));
                    let part_b = scratch.fresh();
                    insns.push(Insn::new(
                        Op::Arith(inner_kind),
                        vec![part_b.clone(), s, Operand::imm(width - n)],
                        addr,
                    ));
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::Or),
                        vec![d, part_a, part_b],
                        addr,
                    ));
                }
            }
            // `shld dst, src, cl` — the variable-count form. x86 masks the
            // count to the operand width (5 bits for 32-bit, 6 for 64-bit) and
            // shifts BOTH halves by the same amount, the second by `width - n`.
            // The count is a real value, so the pair stays exact.
            Mnemonic::Shld | Mnemonic::Shrd
                if matches!(insn.op2_kind(), OpKind::Register)
                    && insn.op2_register() == Register::CL =>
            {
                let d = op_at(0).unwrap_or(Operand::Unknown);
                let s = op_at(1).unwrap_or(Operand::Unknown);
                let width: i64 = if reg_is_64(insn.op0_register()) {
                    64
                } else {
                    32
                };
                let mask = width - 1;
                let left = insn.mnemonic() == Mnemonic::Shld;
                let (inner_kind, outer_kind) = if left {
                    (ArithKind::Shr, ArithKind::Shl)
                } else {
                    (ArithKind::Shl, ArithKind::Shr)
                };
                let n = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::And),
                    vec![n.clone(), Operand::reg(gpr(Register::CL, abi)), Operand::imm(mask)],
                    addr,
                ));
                let m = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Sub),
                    vec![m.clone(), Operand::imm(width), n.clone()],
                    addr,
                ));
                let m = {
                    let masked = scratch.fresh();
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::And),
                        vec![masked.clone(), m, Operand::imm(mask)],
                        addr,
                    ));
                    masked
                };
                let part_a = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(outer_kind),
                    vec![part_a.clone(), d.clone(), n],
                    addr,
                ));
                let part_b = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(inner_kind),
                    vec![part_b.clone(), s, m],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Or),
                    vec![d, part_a, part_b],
                    addr,
                ));
            }
            // ---- accumulator sign extension ---------------------------------
            // `cdq`/`cqo` sign-extend the accumulator into its high half; the
            // following `idiv` reads both. Without this the high half was
            // whatever the previous operation left there, so the division's
            // operands were wrong — and every `long`/`ulong` division in the
            // binary goes through this pair.
            Mnemonic::Cdq | Mnemonic::Cqo | Mnemonic::Cwd => {
                let hi = Operand::reg(gpr(Register::RDX, abi));
                let cond = scratch.fresh();
                insns.push(Insn::new(
                    Op::Compare(CompareKind::LessThan),
                    vec![cond.clone(), Operand::reg(RET_REG), Operand::imm(0)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Select(SelectVariant::Identity),
                    vec![hi, cond, Operand::imm(-1), Operand::imm(0)],
                    addr,
                ));
            }
            Mnemonic::Cwde | Mnemonic::Cbw | Mnemonic::Cdqe => {
                // widen the accumulator with sign extension: (x << k) >> k.
                //
                // `cdqe` (32 → 64) belongs here and had no arm at all, so it
                // reached the catch-all — which emits NOTHING for a
                // zero-operand instruction. `rax` therefore kept its stale
                // high half, and every 64-bit pointer or index derived from a
                // 32-bit `int` expression was silently wrong. It was never
                // reported as an undefined local either, because `eax` and
                // `rax` collapse to one slot (`gpr_ordinal_of`): the read
                // found a definition, just the wrong one.
                let k = match insn.mnemonic() {
                    Mnemonic::Cwde => 16,
                    Mnemonic::Cdqe => 32,
                    _ => 8,
                };
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shl),
                    vec![Operand::reg(RET_REG), Operand::reg(RET_REG), Operand::imm(k)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Shr),
                    vec![Operand::reg(RET_REG), Operand::reg(RET_REG), Operand::imm(k)],
                    addr,
                ));
            }
            // ---- exchange / atomic read-modify-write -------------------------
            // `xchg` swaps both operands with no flags; `cmpxchg` is the whole
            // of `Interlocked.CompareExchange`; `lock xadd` is
            // `Interlocked.Add`/`Increment` with the OLD value returned. All
            // three were unmodelled, so the compare-exchange loops in the
            // runtime and in every `Interlocked` user lost their dataflow.
            Mnemonic::Xchg if insn.op_count() >= 2 => {
                let a = op_at(0).unwrap_or(Operand::Unknown);
                let b = op_at(1).unwrap_or(Operand::Unknown);
                let t = scratch.fresh();
                insns.push(Insn::move_(t.clone(), a.clone(), addr));
                insns.push(Insn::move_(a, b.clone(), addr));
                insns.push(Insn::move_(b, t, addr));
            }
            Mnemonic::Xadd if insn.op_count() >= 2 => {
                let dst = op_at(0).unwrap_or(Operand::Unknown);
                let src = op_at(1).unwrap_or(Operand::Unknown);
                let old = scratch.fresh();
                insns.push(Insn::move_(old.clone(), dst.clone(), addr));
                let sum = scratch.fresh();
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Add),
                    vec![sum.clone(), old.clone(), src.clone()],
                    addr,
                ));
                insns.push(Insn::move_(dst, sum, addr));
                insns.push(Insn::move_(src, old, addr));
            }
            Mnemonic::Cmpxchg if insn.op_count() >= 2 => {
                let dst = op_at(0).unwrap_or(Operand::Unknown);
                let src = op_at(1).unwrap_or(Operand::Unknown);
                // old = *dst    (the value the comparison is made against)
                let old = scratch.fresh();
                insns.push(Insn::move_(old.clone(), dst.clone(), addr));
                // rax is read pre-modification, so keep an explicit copy of it
                // for the flag state the branch consumes.
                let acc = Operand::reg(RET_REG);
                let pre = scratch.fresh();
                insns.push(Insn::move_(pre.clone(), acc.clone(), addr));
                let cond = scratch.fresh();
                insns.push(Insn::new(
                    Op::Compare(CompareKind::Equal),
                    vec![cond.clone(), pre.clone(), old.clone()],
                    addr,
                ));
                // dst = (rax == old) ? src : old
                let chosen = scratch.fresh();
                insns.push(Insn::new(
                    Op::Select(SelectVariant::Identity),
                    vec![chosen.clone(), cond.clone(), src, old.clone()],
                    addr,
                ));
                insns.push(Insn::move_(dst, chosen, addr));
                // rax keeps its value when equal, otherwise takes old
                insns.push(Insn::new(
                    Op::Select(SelectVariant::Identity),
                    vec![acc, cond, Operand::reg(RET_REG), old.clone()],
                    addr,
                ));
                // ZF reflects the comparison, so the following Jcc reads the
                // very same operands.
                pending = Some(PendingCmp {
                    lhs: pre,
                    rhs: old,
                    at: insns.len(),
                });
            }
            // ---- rounding and rounding conversions ---------------------------
            // SSE4.1 `round*`: imm8[1:0] selects the mode — 00 nearest-even,
            // 01 floor, 10 ceiling, 11 truncate — which is exactly the set
            // `Math.Round`, `Math.Floor`, `Math.Ceiling` and `Math.Truncate`
            // compile to. Mapped onto the same FloatArith rounding intrinsics
            // the arm64 FRINT* path uses.
            Mnemonic::Roundss | Mnemonic::Roundsd | Mnemonic::Roundps | Mnemonic::Roundpd => {
                let kind = match insn.immediate8() & 3 {
                    1 => FloatArithKind::Floor,
                    2 => FloatArithKind::Ceiling,
                    3 => FloatArithKind::Truncate,
                    _ => FloatArithKind::Round,
                };
                let d = sse_op(&insn, 0, abi);
                let a = sse_op(&insn, 1, abi);
                if let (Some(dd), Some(aa)) = (d, a) {
                    insns.push(Insn::new(Op::FloatArith(kind), vec![dd, aa], addr));
                }
            }
            // `cvtss2si`/`cvtsd2si`: the ROUNDING float→int conversion (its
            // truncating twin is already handled) — a real int/float boundary in
            // the dataflow, not just a name.
            Mnemonic::Cvtss2si | Mnemonic::Cvtsd2si => {
                let d = op_at(0);
                let a = sse_op(&insn, 1, abi);
                if let (Some(dd), Some(aa)) = (d, a) {
                    insns.push(Insn::new(
                        Op::Convert(super::ir::Conversion::FloatToInt),
                        vec![dd, aa],
                        addr,
                    ));
                }
            }
            Mnemonic::Cmp => {
                let lhs = op_at(0).unwrap_or(Operand::Unknown);
                let rhs = op_at(1).unwrap_or(Operand::imm(0));
                pending = Some(PendingCmp {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    at: insns.len(),
                });
                // CF after CMP is the BORROW (`lhs < rhs`, unsigned), which is
                // what a following `sbb` consumes — the compiler compares two
                // 64-bit values in halves, exactly this way.
                pending_carry = Some((true, lhs, rhs));
            }
            Mnemonic::Test => {
                // `test a, b` computes `a AND b` and tests THAT against ZERO.
                // It is not a comparison of `a` with `b`, and recording the raw
                // operand pair as one is the largest single source of nonsense
                // conditions in an x64 dump:
                //
                //   * `test rax, rax` — the canonical null check, and the shape
                //     every lazy static / `ldsfld` guard is compiled to — came
                //     out as `rax == rax` / `rax != rax`, which is constant and
                //     says nothing about the code that is there;
                //   * a mask test `test eax, 1; jz` came out as `eax == 1`,
                //     plausible-looking text with the wrong logic.
                //
                // Measured on the v24.5 PE dump: 23,073 self-comparisons
                // (`sf0 != sf0`, `this != this`, `in_x3 != in_x3`, …) from the
                // same-register form alone. arm64 already lifts its `tst`
                // family this way (`test_flags`); x86 did not.
                //
                // Same operand → the AND IS the operand, so the condition is a
                // plain null check `a == 0` and no scratch register is needed.
                // Different operands → materialize the AND so the recorded
                // condition is `(a & b) == 0`. `test` clears CF and OF, so the
                // unsigned conditions do not survive it either way, and SF is
                // the sign of that same AND, which keeps `js`/`jl` exact.
                let a = op_at(0).unwrap_or(Operand::Unknown);
                let b = op_at(1).unwrap_or(Operand::Unknown);
                pending_carry = None;
                let lhs = if a == b {
                    a
                } else {
                    let acc = scratch.fresh();
                    insns.push(Insn::new(
                        Op::Arith(ArithKind::And),
                        vec![acc.clone(), a, b],
                        addr,
                    ));
                    acc
                };
                // `at` is read AFTER any push above: the AND defines a scratch
                // local, and pinning it at the index of an instruction that
                // DEFINES one of the compare's own operands would make the pin
                // read an undefined local.
                pending = Some(PendingCmp {
                    lhs,
                    rhs: Operand::imm(0),
                    at: insns.len(),
                });
            }
            Mnemonic::Jmp => {
                if insn.flow_control() == FlowControl::IndirectBranch {
                    // A tail transfer does not return, so no compare survives it.
                    pending = None;
                    // `jmp r/m` — a tail call, or an unrecovered jump table. The
                    // TARGET is kept as operand 0, exactly as arm64's `br` does
                    // (`lift.rs`): `passes::recover_tail_calls` reads it to turn
                    // a frame-torn-down indirect branch into
                    // `return fnptr_xN(…)`, and the switch recognizer reads the
                    // table setup around it. The old `vec![]` threw the register
                    // away, so neither could fire and every indirect branch —
                    // including every x64 tail call — reached the structurer as
                    // "indirect jump (dispatch not recovered)".
                    insns.push(Insn::new(
                        Op::IndirectCall,
                        vec![op_at(0).unwrap_or(Operand::Unknown)],
                        addr,
                    ));
                } else {
                    // A direct `jmp` does NOT write EFLAGS, so a compare that
                    // was pending survives it — `cmp; ja L1; jmp L2; L1: jb L3`
                    // re-tests the SAME flags. The generic clear above already
                    // exempts control transfers for exactly this reason; the
                    // explicit `pending = None` that used to sit here
                    // contradicted it and cost those branches their condition.
                    // iced IP values already include our 0-based base → RVA
                    insns.push(Insn::new(Op::Jump(insn.near_branch_target() as usize), vec![], addr));
                }
            }
            Mnemonic::Call => {
                // A CALL CLOBBERS EFLAGS. This arm used to leave every pending
                // flag state intact, and the gate above does not clear it
                // either (it exempts control transfers, which is right for a
                // direct `jmp` and for the branch that READS the flags). The
                // `ret` arm a few lines below already does this
                // ("control transfer: no compare may cross it"), so `call` was
                // the one hole — and it is the dangerous DIRECTION: `cmp; call
                // helper; jne L` kept the `cmp`'s operand pair across the call
                // and rendered `if (a != b)` where the machine actually tests
                // flags the callee set. Plausible text, wrong logic, invisible
                // to every counter because it does not look like a defect.
                //
                // IL2CPP calls managed helpers constantly, so this pattern is
                // dense; clearing here can only turn a fabricated condition
                // into the honest `__unknown`, never lose a real one, because
                // the ABI guarantees nothing about EFLAGS across a call.
                pending = None;
                pending_fp = None;
                pending_bit = None;
                pending_carry = None;
                if insn.flow_control() == FlowControl::IndirectCall {
                    // `call r/m` — a COMPUTED callee: virtual/interface
                    // dispatch, a delegate `Invoke`, a function pointer, or a
                    // shared generic body reached through the `MethodInfo` the
                    // call site loads.
                    //
                    // THIS ARM DID NOT EXIST. `call reg` emitted
                    // `Op::IndirectCall` with NO OPERANDS, which cost both the
                    // callee and the control flow:
                    //   1. the target was thrown away, so nothing downstream
                    //      could name it — `rewrite_virtual_calls`,
                    //      `resolve_methodinfo_usages` and the rgctx resolver
                    //      all read the callee from operand 0;
                    //   2. `Op::IndirectCall` is a CFG TERMINATOR (`cfg.rs`
                    //      gives it no successors), so the block ENDED at a
                    //      call the callee RETURNS from. Everything after it
                    //      became unreachable and the structurer rendered it as
                    //      an "indirect jump (dispatch not recovered)" comment
                    //      plus a forced `goto` — 6,131 sites on the v24.5 PE
                    //      dump.
                    //
                    // A returning call is `Op::Call` with the ordinary layout
                    // `[target, ret-slot, x0 … x5 args, xmm args]`, which is
                    // exactly the arm64 `BLR` contract (`lift.rs`). The target
                    // is the memory access itself for the `call [k+vtable]`
                    // form; `resolve_indirect_target` matches both shapes.
                    let mut ops = vec![op_at(0).unwrap_or(Operand::Unknown), Operand::reg(RET_REG)];
                    for &r in arg_reg_names(abi) {
                        ops.push(Operand::reg(r));
                    }
                    for r in fp_written.iter() {
                        ops.push(Operand::reg(r.clone()));
                    }
                    fp_written.clear();
                    insns.push(Insn::new(Op::Call, ops, addr));
                } else {
                    let t = insn.near_branch_target() as usize;
                    let mut ops = vec![Operand::imm(t as i64), Operand::reg(RET_REG)];
                    for &r in arg_reg_names(abi) {
                        ops.push(Operand::reg(r));
                    }
                    // XMM0..N carry the float arguments — same contract as the
                    // arm64/arm32 lifters, and what keeps the FP computation
                    // feeding the call alive through DCE.
                    for r in fp_written.iter() {
                        ops.push(Operand::reg(r.clone()));
                    }
                    fp_written.clear();
                    insns.push(Insn::new(Op::Call, ops, addr));
                }
            }
            Mnemonic::Ret => {
                pending = None; // control transfer: no compare may cross it
                insns.push(Insn::new(Op::Return, vec![Operand::reg(RET_REG)], addr));
                while addresses.len() < insns.len() {
                    addresses.push(addr);
                }
                // NOTE: do NOT stop the linear decode at the first RET. x86
                // compilers lay out cold/error paths AFTER an early return
                // (ret; then more code that earlier conditional branches
                // target). Breaking here left every CondJump to post-ret code
                // unresolved → end-sentinel → the whole body emitted as empty
                // ("body not recoverable"). Keep decoding the slice; the CFG
                // splits blocks at terminators and reachability analysis
                // drops anything genuinely unreachable.
            }
            Mnemonic::Nop
            | Mnemonic::Int3
            | Mnemonic::Endbr64
            | Mnemonic::Pause
            | Mnemonic::Lfence
            | Mnemonic::Mfence
            | Mnemonic::Sfence
            | Mnemonic::Clflush => {}
            // ---- FP compare + float width conversion -----------------------
            // `ucomiss`/`comisd` set ZF/PF/CF from the FP ordering; the next
            // Jcc tests them. Modelled as a pending compare whose operands are
            // the XMM registers themselves, which is what a C# `a < b` on
            // floats looks like once lifted.
            Mnemonic::Ucomiss
            | Mnemonic::Ucomisd
            | Mnemonic::Comiss
            | Mnemonic::Comisd
            // VEX-encoded twins (`vcomiss`/`vucomiss`/…). These are the exact
            // same EFLAGS-setting operation — AVX just moved them into the VEX
            // space — and they were MISSING, so every AVX float decision
            // (`vucomiss xmm2, xmm1; jbe`) lost its compare and rendered
            // `if (? != 0)`. This is the x86 counterpart of the ARM32 `vcmpe`
            // gap fixed in the same change: both architectures handled the base
            // spelling and dropped the second encoding of the identical
            // instruction. The 2-operand VEX form (`vucomiss xmm1, xmm2`) keeps
            // the operands in slots 0/1, same as the legacy form, so `sse_op`
            // reads them unchanged.
            | Mnemonic::Vcomiss
            | Mnemonic::Vcomisd
            | Mnemonic::Vucomiss
            | Mnemonic::Vucomisd => {
                pending_fp = Some(PendingCmp {
                    lhs: sse_op(&insn, 0, abi).unwrap_or(Operand::Unknown),
                    rhs: sse_op(&insn, 1, abi).unwrap_or(Operand::Unknown),
                    at: insns.len(),
                });
            }
            Mnemonic::Cvtss2sd | Mnemonic::Cvtsd2ss => {
                let d = sse_op(&insn, 0, abi);
                let a = sse_op(&insn, 1, abi);
                if let (Some(dd), Some(aa)) = (d, a) {
                    let conv = if insn.mnemonic() == Mnemonic::Cvtss2sd {
                        Conversion::Float32ToFloat64
                    } else {
                        Conversion::Float64ToFloat32
                    };
                    insns.push(Insn::new(Op::Convert(conv), vec![dd, aa], addr));
                }
            }
            // `cmov<cc> r32, r/m32` — a conditional move, which is the same
            // ternary the arm64 CSEL arm emits. The condition comes from the
            // pending compare (integer or FP), so this consumes it exactly as
            // a Jcc would.
            Mnemonic::Cmovo
            | Mnemonic::Cmovno
            | Mnemonic::Cmovb
            | Mnemonic::Cmovae
            | Mnemonic::Cmove
            | Mnemonic::Cmovne
            | Mnemonic::Cmovbe
            | Mnemonic::Cmova
            | Mnemonic::Cmovs
            | Mnemonic::Cmovns
            | Mnemonic::Cmovp
            | Mnemonic::Cmovnp
            | Mnemonic::Cmovl
            | Mnemonic::Cmovge
            | Mnemonic::Cmovle
            | Mnemonic::Cmovg => {
                let dst = op_at(0).unwrap_or(Operand::Unknown);
                let src = op_at(1).unwrap_or(Operand::Unknown);
                let s = materialize_pending(
                    &mut insns,
                    &mut addresses,
                    &mut scratch,
                    pending_fp.as_ref().or(pending.as_ref()),
                    cond_kind(insn.condition_code()),
                    addr,
                );
                insns.push(Insn::new(
                    Op::Select(SelectVariant::Identity),
                    vec![dst.clone(), s, src, dst],
                    addr,
                ));
            }
            Mnemonic::Leave => {
                // `leave` = `mov rsp, rbp` + `pop rbp`
                insns.push(Insn::move_(
                    Operand::reg("sp"),
                    Operand::reg("fp"),
                    addr,
                ));
                insns.push(Insn::move_(
                    Operand::reg("fp"),
                    Operand::Memory { width: 0,
                        base: Box::new(Operand::reg("sp")),
                        index: None,
                        scale: 1,
                        addend: 0,
                    },
                    addr,
                ));
            }
            // ---- count-register branches ------------------------------------
            // `loop`/`jrcxz` do NOT test EFLAGS: they test RCX. Left to the
            // conditional-branch arm below, they consumed the pending compare
            // and rendered as `if (? != 0)` — a condition the binary does not
            // have, attached to the right edge. Modelled from the register they
            // actually read: `jrcxz` is `rcx == 0`, `loop` is `--rcx != 0`.
            Mnemonic::Jrcxz | Mnemonic::Jecxz | Mnemonic::Jcxz => {
                let cx = Operand::reg(gpr(Register::RCX, abi));
                let cond = scratch.fresh();
                insns.push(Insn::new(
                    Op::Compare(CompareKind::Equal),
                    vec![cond.clone(), cx, Operand::imm(0)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::CondJump(insn.near_branch_target() as usize),
                    vec![cond],
                    addr,
                ));
            }
            Mnemonic::Loop => {
                let cx = Operand::reg(gpr(Register::RCX, abi));
                insns.push(Insn::new(
                    Op::Arith(ArithKind::Sub),
                    vec![cx.clone(), cx.clone(), Operand::imm(1)],
                    addr,
                ));
                let cond = scratch.fresh();
                insns.push(Insn::new(
                    Op::Compare(CompareKind::NotEqual),
                    vec![cond.clone(), cx, Operand::imm(0)],
                    addr,
                ));
                insns.push(Insn::new(
                    Op::CondJump(insn.near_branch_target() as usize),
                    vec![cond],
                    addr,
                ));
            }
            _ if insn.flow_control() == FlowControl::ConditionalBranch => {
                let target = insn.near_branch_target() as usize;
                // READ the flags, do not consume them: `cmp; jne L1; je L2`
                // re-tests the SAME EFLAGS with the second branch, and
                // `.take()` here destroyed the compare so the second branch
                // rendered as `if (? != 0)`. A flag writer clears the pending
                // state on its own, so keeping it is safe.
                let cc = insn.condition_code();
                let s = match bit_scratch_if_carry(
                    &mut insns,
                    &mut scratch,
                    pending_bit.clone(),
                    cc,
                    addr,
                ) {
                    Some(s) => s,
                    None => materialize_pending(
                        &mut insns,
                        &mut addresses,
                        &mut scratch,
                        pending_fp.as_ref().or(pending.as_ref()),
                        cond_kind(cc),
                        addr,
                    ),
                };
                insns.push(Insn::new(Op::CondJump(target), vec![s], addr));
            }
            // ---- generic fallback ------------------------------------------
            //
            // An instruction that writes a register MUST define that register.
            // The old `_ => {}` dropped every unhandled mnemonic, so one missing
            // arm left its destination undefined and the whole dependent
            // expression became `__unknown` (counted downstream as an undefined
            // local). The fallback defines the destination from the real operands
            // and keeps every source, carrying the operation's own mnemonic
            // (`__Paddd(a, b)`), so the value chain stays connected while the
            // semantics stay explicitly unmodelled.
            //
            // Deliberately NOT covered: a flag-writing instruction with a MEMORY
            // destination. There is no correct generic approximation for
            // `add [rax], ecx` (a `Move` would silently drop the read-modify-
            // write), and inventing one would trade a visible gap for wrong
            // logic — the exact trade this work is trying to remove.
            // ---- packed / vector SSE + AVX --------------------------------
            // A mnemonic that has an EXACT whole-register equivalent in this
            // scalar-per-register model is modelled by `lift_packed_simd`;
            // everything else keeps the value-preserving `__<Mnemonic>(…)`
            // stub the catch-all below would have produced. Placed after every
            // concrete arm so the existing scalar `addss`/`movss`/bitwise arms
            // keep precedence.
            // LANE-ADDRESSED vector ops: a different KIND of operation from the
            // whole-register arithmetic above, because its meaning IS the lane
            // arrangement (`shufps`, `unpcklps`, `psrldq`, `pshufd`, `paddd`).
            // Tried first so the more specific form wins; it only emits when it
            // returns `true`.
            m if lift_lane_simd(&insn, m, abi, &mut insns, &mut scratch, addr) => {}
            m if lift_packed_simd(&insn, m, abi, &mut insns, &mut scratch, addr) => {}
            other => {
                // `ILR_X86GAPS=1` reports every mnemonic that reaches this
                // catch-all, so an unmodelled instruction can be identified
                // from a REAL binary instead of from a guess about what the
                // decoder emits. This is the same discipline `ILR_A32GAPS`
                // applies to the arm32 lifter, and it is the only honest way to
                // find what is still missing on a target with no corpus here.
                if std::env::var("ILR_X86GAPS").is_ok() {
                    let kinds = if insn.op_count() == 0 {
                        String::from("-")
                    } else {
                        (0..insn.op_count())
                            .map(|n| format!("{:?}", insn.op_kind(n)))
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    eprintln!("X86GAP {other:?}|ops={kinds}");
                }
                let dst = op_at(0);
                if matches!(dst, Some(Operand::Register(_))) {
                    let mut ops = vec![dst.clone().unwrap_or(Operand::Unknown)];
                    for n in 1..insn.op_count() as usize {
                        if let Some(o) = op_at(n) {
                            ops.push(o);
                        }
                    }
                    // A ONE-operand form must still DEFINE its register. The
                    // old `ops.len() > 1` guard silently dropped them, so e.g.
                    // `bswap eax` left `eax` holding its previous value and
                    // every later read of it was wrong rather than missing.
                    // Naming the destination again as the source keeps the
                    // shape honest: `eax = __Bswap(eax)`.
                    if ops.len() == 1 {
                        ops.push(ops[0].clone());
                    }
                    insns.push(Insn::new(Op::Simd(format!("__{other:?}")), ops, addr));
                }
            }
        }

        // Track FP destinations centrally rather than in every FP arm: any
        // emitted IR insn whose destination is a `vfN` register is a float
        // being materialized, and a call must list those as arguments.
        if let Some(Operand::Register(r)) = insns.last().and_then(|i| i.operands.first()) {
            if r.starts_with("vf") {
                fp_written.insert(r.clone());
            }
        }

        // one address entry per emitted IR insn (see note at loop head)
        while addresses.len() < insns.len() {
            addresses.push(addr);
        }
    }

    if std::env::var("ILR_X86RAW").is_ok() {
        eprintln!("[x86raw] base={:#x} n={}", base_rva, insns.len());
        for (i, insn) in insns.iter().enumerate() {
            eprintln!(
                "[x86raw] i{} addr={:#x} op={:?} ops={:?}",
                i, insn.address, insn.op, insn.operands
            );
        }
    }

    if !insns.last().is_some_and(|i| i.is_terminator()) {
        let tail = base_rva;
        insns.push(Insn::new(Op::Return, vec![Operand::reg(RET_REG)], tail));
        addresses.push(tail);
    }
    Some(LiftResult { insns, addresses })
}

/// Is this one of the string primitives, whose operands are IMPLICIT?
///
/// `movsd` and `cmpsd` are also SSE mnemonics (a scalar double move and a
/// scalar double compare), so the mnemonic alone is ambiguous. iced reports the
/// string form with `[rdi]`/`[rsi]` as its operands, which is what this tests.
fn is_string_op(insn: &X86Insn) -> bool {
    (0..insn.op_count().min(2)).any(|n| {
        matches!(
            insn.op_kind(n),
            OpKind::MemoryESRDI
                | OpKind::MemoryESEDI
                | OpKind::MemoryESDI
                | OpKind::MemorySegRSI
                | OpKind::MemorySegESI
                | OpKind::MemorySegSI
        )
    })
}

fn mnemonic_is_add(insn: &X86Insn) -> bool {
    matches!(insn.mnemonic(), Mnemonic::Add | Mnemonic::Inc)
}

/// The operand triple for a packed/vector BINARY op, honouring the two
/// encodings that reach the same destination:
///
///   * SSE  `mulps xmm0, xmm1`        — op_count 2; the destination is ALSO
///     the first source (`xmm0 = xmm0 * xmm1`).
///   * AVX  `vmulps xmm0, xmm1, xmm2` — op_count 3; both sources explicit.
///
/// `None` when any operand could not be read, so the caller keeps the
/// value-preserving `__<Mnemonic>(…)` stub instead of emitting a wrong one.
fn simd_binary_ops(insn: &X86Insn, abi: X86Abi) -> Option<(Operand, Operand, Operand)> {
    let d = sse_op(insn, 0, abi)?;
    if insn.op_count() >= 3 {
        let a = sse_op(insn, 1, abi)?;
        let b = sse_op(insn, 2, abi)?;
        Some((d, a, b))
    } else {
        let b = sse_op(insn, 1, abi)?;
        Some((d.clone(), d, b))
    }
}

/// The `(destination, source)` pair of a UNARY vector op / conversion. The
/// destination is NOT a source here: `cvtdq2ps xmm0, xmm1` reads `xmm1` and
/// writes `xmm0`, and treating the destination as source 0 (as the binary
/// helper does) would make it read a value the instruction discards.
fn simd_unary_ops(insn: &X86Insn, abi: X86Abi) -> Option<(Operand, Operand)> {
    let d = sse_op(insn, 0, abi)?;
    let s = sse_op(insn, 1, abi)?;
    Some((d, s))
}

/// Model a packed/vector SSE/AVX instruction that has an EXACT whole-register
/// equivalent in this IR's per-register scalar model. Returns `true` when it
/// emitted the modelled form; `false` leaves the instruction to the catch-all,
/// which keeps the honest value-preserving `__<Mnemonic>(…)`.
///
/// The IR names ONE value per vector register (`vf0`…`vf31`, see `gpr`), so a
/// 128-bit lane cannot be addressed individually. That still buys real
/// semantics for everything that is not lane-arrangement:
///
///   * **conversions** — `cvtdq2ps`, `cvtps2pd`, `cvtpd2ps`, `cvtdq2pd`,
///     `cvtps2dq`, `cvttps2dq`, `cvttpd2dq`, `cvtpd2dq` and their AVX twins —
///     become real `Op::Convert`s. On the real Unity-6 x64 PE corpus this is
///     the single largest unmodelled family (`cvtdq2ps` 2,433 sites,
///     `cvtps2pd` 550, `cvtpd2ps` 165, `cvtdq2pd` 121);
///   * **packed FP arithmetic** (`addps`/`subps`/`mulps`/`divps`, the `pd`
///     and AVX `v…ss`/`v…sd` forms) becomes the same elementwise
///     `FloatArith` the scalar `addss` family already emits;
///   * **vector moves** (`movaps`/`movups`/`movdqa`/`movdqu`/`movd`/`movq`
///     and their AVX twins) become real transfers — `move_` handles a memory
///     side on either end, so a spill is a load and a fill a store;
///   * **packed FP min/max** becomes compare + `Select`;
///   * **`pandn`** becomes the `(~a) & b` it is.
///
/// Deliberately LEFT as a named stub (same rule the rewrite catalog uses: a
/// failed match leaves the lower-level sequence untouched): lane permutations
/// and byte shifts (`shufps`, `unpcklps`, `psrldq`, `pshufd`, `punpck*`) and
/// horizontal reductions — whose entire meaning is the lane arrangement this
/// model cannot name — plus packed INTEGER arithmetic (`paddd`/`psubd`/
/// `psllw`), whose per-lane carries a single scalar register would get wrong.
#[allow(clippy::too_many_lines)]
fn lift_packed_simd(
    insn: &X86Insn,
    m: Mnemonic,
    abi: X86Abi,
    insns: &mut Vec<Insn>,
    scratch: &mut Scratch,
    addr: u64,
) -> bool {
    // ---- vector moves (exact whole-register transfer) ---------------------
    if matches!(
        m,
        Mnemonic::Movaps
            | Mnemonic::Movapd
            | Mnemonic::Movups
            | Mnemonic::Movupd
            | Mnemonic::Movdqa
            | Mnemonic::Movdqu
            | Mnemonic::Movd
            | Mnemonic::Movq
            | Mnemonic::Vmovaps
            | Mnemonic::Vmovapd
            | Mnemonic::Vmovups
            | Mnemonic::Vmovupd
            | Mnemonic::Vmovdqa
            | Mnemonic::Vmovdqu
            | Mnemonic::Vmovd
            | Mnemonic::Vmovq
    ) {
        if let (Some(d), Some(s)) = (sse_op(insn, 0, abi), sse_op(insn, 1, abi)) {
            insns.push(Insn::move_(d, s, addr));
            return true;
        }
        return false;
    }

    // ---- conversions ------------------------------------------------------
    let conv = match m {
        Mnemonic::Cvtdq2ps | Mnemonic::Vcvtdq2ps => Some(Conversion::IntToFloat32),
        Mnemonic::Cvtdq2pd | Mnemonic::Vcvtdq2pd => Some(Conversion::IntToFloat64),
        Mnemonic::Cvtps2dq
        | Mnemonic::Vcvtps2dq
        | Mnemonic::Cvttps2dq
        | Mnemonic::Vcvttps2dq
        | Mnemonic::Cvtpd2dq
        | Mnemonic::Vcvtpd2dq
        | Mnemonic::Cvttpd2dq
        | Mnemonic::Vcvttpd2dq => Some(Conversion::FloatToInt),
        Mnemonic::Cvtps2pd | Mnemonic::Vcvtps2pd => Some(Conversion::Float32ToFloat64),
        Mnemonic::Cvtpd2ps | Mnemonic::Vcvtpd2ps => Some(Conversion::Float64ToFloat32),
        _ => None,
    };
    if let Some(c) = conv {
        if let Some((d, s)) = simd_unary_ops(insn, abi) {
            insns.push(Insn::new(Op::Convert(c), vec![d, s], addr));
            return true;
        }
        return false;
    }

    // `vcvtsi2ss xmm0, xmm1, r32` — three operands, and the INT is operand 2
    // (operand 1 is the merge source). The plain `cvtsi2ss` arm above covers
    // only the two-operand SSE form, so every AVX build's int->float stayed a
    // stub.
    if matches!(m, Mnemonic::Vcvtsi2ss | Mnemonic::Vcvtsi2sd) {
        let d = sse_op(insn, 0, abi);
        let s = if insn.op_count() >= 3 {
            sse_op(insn, 2, abi)
        } else {
            sse_op(insn, 1, abi)
        };
        if let (Some(d), Some(s)) = (d, s) {
            let c = if m == Mnemonic::Vcvtsi2sd {
                Conversion::IntToFloat64
            } else {
                Conversion::IntToFloat32
            };
            insns.push(Insn::new(Op::Convert(c), vec![d, s], addr));
            return true;
        }
        return false;
    }

    // ---- packed/scalar-AVX FP arithmetic (elementwise) --------------------
    let fk = match m {
        Mnemonic::Addps
        | Mnemonic::Addpd
        | Mnemonic::Vaddps
        | Mnemonic::Vaddpd
        | Mnemonic::Vaddss
        | Mnemonic::Vaddsd => Some(FloatArithKind::Add),
        Mnemonic::Subps
        | Mnemonic::Subpd
        | Mnemonic::Vsubps
        | Mnemonic::Vsubpd
        | Mnemonic::Vsubss
        | Mnemonic::Vsubsd => Some(FloatArithKind::Sub),
        Mnemonic::Mulps
        | Mnemonic::Mulpd
        | Mnemonic::Vmulps
        | Mnemonic::Vmulpd
        | Mnemonic::Vmulss
        | Mnemonic::Vmulsd => Some(FloatArithKind::Mul),
        Mnemonic::Divps
        | Mnemonic::Divpd
        | Mnemonic::Vdivps
        | Mnemonic::Vdivpd
        | Mnemonic::Vdivss
        | Mnemonic::Vdivsd => Some(FloatArithKind::Div),
        _ => None,
    };
    if let Some(k) = fk {
        if let Some((d, a, b)) = simd_binary_ops(insn, abi) {
            insns.push(Insn::new(Op::FloatArith(k), vec![d, a, b], addr));
            return true;
        }
        return false;
    }

    // ---- unary packed FP square root --------------------------------------
    if matches!(
        m,
        Mnemonic::Sqrtps
            | Mnemonic::Sqrtpd
            | Mnemonic::Vsqrtps
            | Mnemonic::Vsqrtpd
            | Mnemonic::Vsqrtss
            | Mnemonic::Vsqrtsd
    ) {
        if let Some((d, s)) = simd_unary_ops(insn, abi) {
            insns.push(Insn::new(
                Op::FloatArith(FloatArithKind::Sqrt),
                vec![d, s],
                addr,
            ));
            return true;
        }
        return false;
    }

    // ---- packed/scalar-AVX FP min / max (compare + select) ----------------
    let is_min = matches!(
        m,
        Mnemonic::Minps
            | Mnemonic::Minpd
            | Mnemonic::Vminps
            | Mnemonic::Vminpd
            | Mnemonic::Vminss
            | Mnemonic::Vminsd
    );
    let is_max = matches!(
        m,
        Mnemonic::Maxps
            | Mnemonic::Maxpd
            | Mnemonic::Vmaxps
            | Mnemonic::Vmaxpd
            | Mnemonic::Vmaxss
            | Mnemonic::Vmaxsd
    );
    if is_min || is_max {
        if let Some((d, a, b)) = simd_binary_ops(insn, abi) {
            let cond = scratch.fresh();
            let kind = if is_min {
                CompareKind::LessThan
            } else {
                CompareKind::GreaterOrEqual
            };
            insns.push(Insn::new(
                Op::Compare(kind),
                vec![cond.clone(), a.clone(), b.clone()],
                addr,
            ));
            insns.push(Insn::new(
                Op::Select(SelectVariant::Identity),
                vec![d, cond, a, b],
                addr,
            ));
            return true;
        }
        return false;
    }

    // ---- packed bitwise-not-and (`pandn dst, src` = ~dst & src) ------------
    // `andnps`/`vandnps` are the SSE spelling of the same "not-and": the
    // bitwise forms operate on the whole register, so the packed and the
    // integer `pandn` lowerings are identical.
    if matches!(
        m,
        Mnemonic::Pandn | Mnemonic::Vpandn | Mnemonic::Andnps | Mnemonic::Vandnps
    ) {
        if let Some((d, a, b)) = simd_binary_ops(insn, abi) {
            let inv = scratch.fresh();
            insns.push(Insn::new(
                Op::Arith(ArithKind::Xor),
                vec![inv.clone(), a, Operand::imm(-1)],
                addr,
            ));
            insns.push(Insn::new(Op::Arith(ArithKind::And), vec![d, inv, b], addr));
            return true;
        }
        return false;
    }

    false
}

// ---------------------------------------------------------------------------
// LANE-ADDRESSED vector operations
// ---------------------------------------------------------------------------
//
// `lift_packed_simd` above models everything whose meaning survives in a
// ONE-value-per-register model. The family below is the `shufps`/`unpcklps`/
// `psrldq`/`pshufd`/`paddd` family whose ENTIRE meaning is the lane
// arrangement, which that model cannot name — so each one used to leave the
// lifter as `Op::Simd("__Shufps")`, i.e. a name for an operation nobody could
// read. On the x64 reference corpus that was 4,666 of the 4,985 remaining
// unmodelled stubs (`shufps` 2,385, `psrldq` 1,255, `unpcklps` 1,026, …).
//
// Ghidra's answer is the subpiece model (see `space.cc`'s `SpacebaseSpace` and
// the `SUBPIECE`/`PIECE` semantics in `opbehavior.cc`): a register is addressed
// as `(offset, size)` pieces, so a permutation is a list of piece NUMBERS and
// the operation carries no magic at all. The equivalent here is a LANE index —
// the same information at the granularity a register-value IR can express — and
// it is carried as DATA (`SimdOp::Arrange`), not as a string, so the operation
// can be folded, range-checked and inverted instead of merely printed.

/// True when operand `n` names a 256-bit (`ymm`) or wider vector register.
///
/// A VEX form with `L = 1` operates on the register's UPPER lanes as well, and
/// this IR names one physical vector register at every width (`gpr` maps
/// `xmm3`/`ymm3`/`zmm3` to the same value), so a 128-bit lane table would
/// describe only half of the result. Those forms are therefore REFUSED and keep
/// the value-preserving stub — a partial description is a wrong value, which is
/// worse than a missing name.
fn is_wide_vector(insn: &X86Insn, n: u32) -> bool {
    if n >= insn.op_count() || insn.op_kind(n) != OpKind::Register {
        return false;
    }
    let r = insn.op_register(n) as u32;
    let ymm = Register::YMM0 as u32;
    let zmm = Register::ZMM0 as u32;
    (ymm..ymm + 32).contains(&r) || (zmm..zmm + 32).contains(&r)
}

/// The `imm8` operand at position `n`, when the instruction has one there.
///
/// The vector immediate forms are all plain `Immediate8` (`ib`); the sign- or
/// zero-extending kinds belong to `add`/`push`, so refusing anything else keeps
/// a memory-operand form from being read as a lane selector.
fn imm8_at(insn: &X86Insn, n: u32) -> Option<u8> {
    if n >= insn.op_count() || insn.op_kind(n) != OpKind::Immediate8 {
        return None;
    }
    Some(insn.immediate8())
}

/// `(dst, source A, source B, imm8)` for the two-source, imm8-carrying vector
/// forms (`shufps`/`shufpd`, `pinsrb/w/d/q`).
///
/// SSE spells the destination as its own first source (`shufps xmm0, xmm1, ib`
/// reads `xmm0`), AVX makes both sources explicit (`vshufps xmm0, xmm1, xmm2,
/// ib`). The lane math is identical; only operand 1's identity changes, and
/// resolving that HERE is what keeps every caller's lane table a constant.
fn two_source_lanes(insn: &X86Insn, abi: X86Abi) -> Option<(Operand, Operand, Operand, u8)> {
    let d = sse_op(insn, 0, abi)?;
    let imm = imm8_at(insn, insn.op_count().saturating_sub(1))?;
    if insn.op_count() >= 4 {
        let a = sse_op(insn, 1, abi)?;
        let b = sse_op(insn, 2, abi)?;
        Some((d, a, b, imm))
    } else {
        let b = sse_op(insn, 1, abi)?;
        Some((d.clone(), d, b, imm))
    }
}

/// `(dst, source, imm8)` for the ONE-source imm8 forms (`pshufd`, `pshufhw`,
/// `psrldq`, `pextrw`). `psrldq xmm0, ib` has only two operands, and there the
/// destination IS the source.
fn one_source_lanes(insn: &X86Insn, abi: X86Abi) -> Option<(Operand, Operand, u8)> {
    let d = sse_op(insn, 0, abi)?;
    if insn.op_count() >= 3 {
        let s = sse_op(insn, 1, abi)?;
        Some((d, s, imm8_at(insn, 2)?))
    } else {
        Some((d.clone(), d, imm8_at(insn, 1)?))
    }
}

/// Emit `dst = rearrange(a[, b])` and report that the instruction was modelled.
fn push_arrange(
    insns: &mut Vec<Insn>,
    addr: u64,
    dst: Operand,
    a: Operand,
    b: Option<Operand>,
    elem_bytes: u8,
    lanes: Vec<LanePick>,
) -> bool {
    let mut ops = vec![dst, a];
    if let Some(b) = b {
        ops.push(b);
    }
    insns.push(Insn::new(
        Op::SimdLane(SimdOp::Arrange { elem_bytes, lanes }),
        ops,
        addr,
    ));
    true
}

/// The fixed two-source interleave/unpack family: both sources explicit after
/// `simd_binary_ops`, lane list a constant.
fn push_two_source(
    insn: &X86Insn,
    abi: X86Abi,
    insns: &mut Vec<Insn>,
    addr: u64,
    elem_bytes: u8,
    lanes: Vec<LanePick>,
) -> bool {
    match simd_binary_ops(insn, abi) {
        Some((d, a, b)) => push_arrange(insns, addr, d, a, Some(b), elem_bytes, lanes),
        None => false,
    }
}

/// `pinsrb/w/d/q dst, src, imm8`: lane `imm % n` of the destination becomes the
/// low lane of the scalar source, and every other lane is kept.
///
/// The scalar is a GPR, and a GPR value in this IR is a one-lane value, so
/// `LanePick::B(0)` names exactly the bits the instruction inserts — no
/// widening or truncation has to be invented.
fn push_insert(
    insn: &X86Insn,
    abi: X86Abi,
    insns: &mut Vec<Insn>,
    addr: u64,
    elem_bytes: u8,
) -> bool {
    let Some((d, a, b, imm)) = two_source_lanes(insn, abi) else {
        return false;
    };
    // 16 bytes per register, so this is the register's lane count at this
    // width: 16 byte lanes, 8 word lanes, 4 dword lanes, 2 qword lanes.
    let n: u8 = 16 / elem_bytes.max(1);
    let k = imm % n;
    let lanes: Vec<LanePick> = (0..n)
        .map(|i| {
            if i == k {
                LanePick::B(0)
            } else {
                LanePick::A(i)
            }
        })
        .collect();
    push_arrange(insns, addr, d, a, Some(b), elem_bytes, lanes)
}

/// Model a LANE-ADDRESSED vector instruction exactly. Returns `true` when it
/// emitted the operation; `false` leaves the instruction to `lift_packed_simd`
/// and then to the catch-all, which keeps the value-preserving
/// `__<Mnemonic>(…)` stub.
///
/// Every lane table below is the instruction's definition, not a guess:
///
///   * `shufps d, s, ib` — out[0] = d[ib&3], out[1] = d[(ib>>2)&3],
///     out[2] = s[(ib>>4)&3], out[3] = s[(ib>>6)&3] (DWORD lanes);
///   * `shufpd d, s, ib` — out[0] = d[ib&1], out[1] = s[(ib>>1)&1] (QWORD);
///   * `pshufd d, s, ib` — all four DWORD lanes from `s`, in `ib` field order;
///   * `pshuflw`/`pshufhw` — four of the eight WORD lanes are permuted and the
///     other four pass through unchanged (which of the two halves differs);
///   * `unpcklps`/`unpckhps` — interleave the LOW/HIGH two DWORD lanes of each
///     source; `unpcklpd`/`unpckhpd`/`punpck*` are the same for their lane
///     width, which is exactly why one table per width covers the family;
///   * `psrldq`/`pslldq` — a BYTE shift of the whole register, zero filled.
///     This is `16 - imm` lane moves, not a shift of a 128-bit integer, and
///     the fill is a real `Zero` lane;
///   * `pextrb/w/d/q` — one lane, zero-extended into a GPR;
///   * `paddb/w/d/q`, `psubb/w/d/q` — per-lane modular arithmetic, no carry
///     across a lane boundary (which a register-wide `Arith(Add)` would get
///     WRONG, and is why these cannot go through `Op::Arith`);
///   * `pcmpeq* reg, reg` — every lane compares equal to itself, so the whole
///     register is all-ones.
#[allow(clippy::too_many_lines)]
fn lift_lane_simd(
    insn: &X86Insn,
    m: Mnemonic,
    abi: X86Abi,
    insns: &mut Vec<Insn>,
    // No scratch slot is needed: every lowering below writes a VALUE derived
    // from its real operands, so none of them has to materialise a temporary.
    _scratch: &mut Scratch,
    addr: u64,
) -> bool {
    use LanePick::{A, B, Zero};

    // The destination first: a 256-bit form's lane table would describe only
    // its low half, so it must keep the stub instead.
    if is_wide_vector(insn, 0) {
        return false;
    }

    // ---- A. imm8-selected two-source arrangements -------------------------
    if matches!(
        m,
        Mnemonic::Shufps | Mnemonic::Vshufps | Mnemonic::Shufpd | Mnemonic::Vshufpd
    ) {
        let Some((d, a, b, imm)) = two_source_lanes(insn, abi) else {
            return false;
        };
        let (elem_bytes, lanes) = if matches!(m, Mnemonic::Shufps | Mnemonic::Vshufps) {
            (
                4u8,
                vec![
                    A(imm & 3),
                    A((imm >> 2) & 3),
                    B((imm >> 4) & 3),
                    B((imm >> 6) & 3),
                ],
            )
        } else {
            (8u8, vec![A(imm & 1), B((imm >> 1) & 1)])
        };
        return push_arrange(insns, addr, d, a, Some(b), elem_bytes, lanes);
    }

    // ---- B. imm8-selected one-source arrangements -------------------------
    if matches!(m, Mnemonic::Pshufd | Mnemonic::Vpshufd) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        return push_arrange(
            insns,
            addr,
            d,
            s,
            None,
            4,
            vec![
                A(imm & 3),
                A((imm >> 2) & 3),
                A((imm >> 4) & 3),
                A((imm >> 6) & 3),
            ],
        );
    }
    if matches!(m, Mnemonic::Pshuflw | Mnemonic::Vpshuflw) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        // The LOW four WORD lanes are permuted; lanes 4-7 pass through.
        return push_arrange(
            insns,
            addr,
            d,
            s,
            None,
            2,
            vec![
                A(imm & 3),
                A((imm >> 2) & 3),
                A((imm >> 4) & 3),
                A((imm >> 6) & 3),
                A(4),
                A(5),
                A(6),
                A(7),
            ],
        );
    }
    if matches!(m, Mnemonic::Pshufhw | Mnemonic::Vpshufhw) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        // Mirror image: lanes 0-3 pass through, the HIGH four are permuted.
        return push_arrange(
            insns,
            addr,
            d,
            s,
            None,
            2,
            vec![
                A(0),
                A(1),
                A(2),
                A(3),
                A(4 + (imm & 3)),
                A(4 + ((imm >> 2) & 3)),
                A(4 + ((imm >> 4) & 3)),
                A(4 + ((imm >> 6) & 3)),
            ],
        );
    }
    // A right byte shift: output byte k is input byte k + imm, zero past the
    // top. 16 one-byte lanes, so the index arithmetic is done in u16 to keep
    // `imm` = 255 from wrapping a u8 and naming lane 0 again.
    if matches!(m, Mnemonic::Psrldq | Mnemonic::Vpsrldq) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        let lanes: Vec<LanePick> = (0..16u16)
            .map(|k| {
                let src = k + imm as u16;
                if src < 16 { A(src as u8) } else { Zero }
            })
            .collect();
        return push_arrange(insns, addr, d, s, None, 1, lanes);
    }
    if matches!(m, Mnemonic::Pslldq | Mnemonic::Vpslldq) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        let lanes: Vec<LanePick> = (0..16u16)
            .map(|k| {
                if k >= imm as u16 {
                    A((k - imm as u16) as u8)
                } else {
                    Zero
                }
            })
            .collect();
        return push_arrange(insns, addr, d, s, None, 1, lanes);
    }

    // ---- B2. extract one lane into a GPR ---------------------------------
    if matches!(m, Mnemonic::Pextrb | Mnemonic::Vpextrb) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        return push_arrange(insns, addr, d, s, None, 1, vec![A(imm & 15)]);
    }
    if matches!(m, Mnemonic::Pextrw | Mnemonic::Vpextrw) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        return push_arrange(insns, addr, d, s, None, 2, vec![A(imm & 7)]);
    }
    if matches!(m, Mnemonic::Pextrd | Mnemonic::Vpextrd) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        return push_arrange(insns, addr, d, s, None, 4, vec![A(imm & 3)]);
    }
    if matches!(m, Mnemonic::Pextrq | Mnemonic::Vpextrq) {
        let Some((d, s, imm)) = one_source_lanes(insn, abi) else {
            return false;
        };
        return push_arrange(insns, addr, d, s, None, 8, vec![A(imm & 1)]);
    }

    // ---- C. insert one GPR lane into a vector ----------------------------
    if matches!(m, Mnemonic::Pinsrb | Mnemonic::Vpinsrb) {
        return push_insert(insn, abi, insns, addr, 1);
    }
    if matches!(m, Mnemonic::Pinsrw | Mnemonic::Vpinsrw) {
        return push_insert(insn, abi, insns, addr, 2);
    }
    if matches!(m, Mnemonic::Pinsrd | Mnemonic::Vpinsrd) {
        return push_insert(insn, abi, insns, addr, 4);
    }
    if matches!(m, Mnemonic::Pinsrq | Mnemonic::Vpinsrq) {
        return push_insert(insn, abi, insns, addr, 8);
    }

    // ---- D. fixed-arrangement interleave (`unpck*`, `punpck*`) -----------
    if matches!(m, Mnemonic::Unpcklps | Mnemonic::Vunpcklps) {
        return push_two_source(insn, abi, insns, addr, 4, vec![A(0), B(0), A(1), B(1)]);
    }
    if matches!(m, Mnemonic::Unpckhps | Mnemonic::Vunpckhps) {
        return push_two_source(insn, abi, insns, addr, 4, vec![A(2), B(2), A(3), B(3)]);
    }
    if matches!(m, Mnemonic::Unpcklpd | Mnemonic::Vunpcklpd) {
        return push_two_source(insn, abi, insns, addr, 8, vec![A(0), B(0)]);
    }
    if matches!(m, Mnemonic::Unpckhpd | Mnemonic::Vunpckhpd) {
        return push_two_source(insn, abi, insns, addr, 8, vec![A(1), B(1)]);
    }
    if matches!(m, Mnemonic::Punpcklqdq | Mnemonic::Vpunpcklqdq) {
        return push_two_source(insn, abi, insns, addr, 8, vec![A(0), B(0)]);
    }
    if matches!(m, Mnemonic::Punpckhqdq | Mnemonic::Vpunpckhqdq) {
        return push_two_source(insn, abi, insns, addr, 8, vec![A(1), B(1)]);
    }
    if matches!(m, Mnemonic::Punpckldq | Mnemonic::Vpunpckldq) {
        return push_two_source(insn, abi, insns, addr, 4, vec![A(0), B(0), A(1), B(1)]);
    }
    if matches!(m, Mnemonic::Punpckhdq | Mnemonic::Vpunpckhdq) {
        return push_two_source(insn, abi, insns, addr, 4, vec![A(2), B(2), A(3), B(3)]);
    }
    if matches!(m, Mnemonic::Punpcklwd | Mnemonic::Vpunpcklwd) {
        let lanes = (0..4u8).flat_map(|i| [A(i), B(i)]).collect();
        return push_two_source(insn, abi, insns, addr, 2, lanes);
    }
    if matches!(m, Mnemonic::Punpckhwd | Mnemonic::Vpunpckhwd) {
        let lanes = (4..8u8).flat_map(|i| [A(i), B(i)]).collect();
        return push_two_source(insn, abi, insns, addr, 2, lanes);
    }
    if matches!(m, Mnemonic::Punpcklbw | Mnemonic::Vpunpcklbw) {
        let lanes = (0..8u8).flat_map(|i| [A(i), B(i)]).collect();
        return push_two_source(insn, abi, insns, addr, 1, lanes);
    }
    if matches!(m, Mnemonic::Punpckhbw | Mnemonic::Vpunpckhbw) {
        let lanes = (8..16u8).flat_map(|i| [A(i), B(i)]).collect();
        return push_two_source(insn, abi, insns, addr, 1, lanes);
    }

    // ---- E. packed INTEGER arithmetic (per lane, no cross-lane carry) ----
    let packed: Option<(u8, ArithKind)> = match m {
        Mnemonic::Paddb | Mnemonic::Vpaddb => Some((1, ArithKind::Add)),
        Mnemonic::Paddw | Mnemonic::Vpaddw => Some((2, ArithKind::Add)),
        Mnemonic::Paddd | Mnemonic::Vpaddd => Some((4, ArithKind::Add)),
        Mnemonic::Paddq | Mnemonic::Vpaddq => Some((8, ArithKind::Add)),
        Mnemonic::Psubb | Mnemonic::Vpsubb => Some((1, ArithKind::Sub)),
        Mnemonic::Psubw | Mnemonic::Vpsubw => Some((2, ArithKind::Sub)),
        Mnemonic::Psubd | Mnemonic::Vpsubd => Some((4, ArithKind::Sub)),
        Mnemonic::Psubq | Mnemonic::Vpsubq => Some((8, ArithKind::Sub)),
        _ => None,
    };
    if let Some((elem_bytes, kind)) = packed {
        return match simd_binary_ops(insn, abi) {
            Some((d, a, b)) => {
                insns.push(Insn::new(
                    Op::SimdLane(SimdOp::PackedArith { elem_bytes, kind }),
                    vec![d, a, b],
                    addr,
                ));
                true
            }
            None => false,
        };
    }

    // ---- F. packed compare of a value with ITSELF -> all-ones -------------
    if matches!(
        m,
        Mnemonic::Pcmpeqb
            | Mnemonic::Vpcmpeqb
            | Mnemonic::Pcmpeqw
            | Mnemonic::Vpcmpeqw
            | Mnemonic::Pcmpeqd
            | Mnemonic::Vpcmpeqd
            | Mnemonic::Pcmpeqq
            | Mnemonic::Vpcmpeqq
    ) {
        // SSE2 two-operand: `pcmpeqd xmm0, xmm0`. AVX: the two SOURCES are
        // operands 1 and 2 (`vpcmpeqd xmm0, xmm1, xmm1`). Every lane then
        // compares equal to itself, so the result is all-ones in every lane.
        let same = match insn.op_count() {
            2 => {
                insn.op0_kind() == OpKind::Register
                    && insn.op1_kind() == OpKind::Register
                    && insn.op0_register() == insn.op1_register()
            }
            n if n >= 3 => {
                insn.op1_kind() == OpKind::Register
                    && insn.op2_kind() == OpKind::Register
                    && insn.op1_register() == insn.op2_register()
            }
            _ => false,
        };
        if same {
            if let Some(d) = sse_op(insn, 0, abi) {
                // All bits set. The register-granular model renders integer
                // values, and `-1` is all-ones at every width, so this is the
                // exact mask the instruction produces.
                insns.push(Insn::move_(d, Operand::imm(-1), addr));
                return true;
            }
        }
        // Different operands: a per-lane COMPARE into a lane mask, which this
        // model has no way to name. Refuse and let the catch-all keep the
        // value-preserving stub rather than invent a whole-register compare.
        return false;
    }

    false
}

/// Instructions PROVEN not to write EFLAGS.
///
/// The list only needs to cover what actually sits between a compare and the
/// branch consuming it — data movement, SSE/AVX arithmetic (which sets MXCSR,
/// not EFLAGS), stack traffic, and the flag *readers* (setcc/cmovcc). Anything
/// unlisted is treated as a flag writer and clears the pending compare, so a
/// missed entry can only cost a condition (`if (? != 0)`), never invent a wrong
/// one.
fn preserves_flags(m: Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Mov
            | Mnemonic::Lea
            | Mnemonic::Movzx
            | Mnemonic::Movsx
            | Mnemonic::Movsxd
            | Mnemonic::Movd
            | Mnemonic::Movq
            | Mnemonic::Movss
            | Mnemonic::Movsd
            // `movs`/`stos`/`lods` do not write EFLAGS either (only
            // `scas`/`cmps` do), so a pending compare survives a `rep movsb`
            // exactly as it survives a plain move.
            | Mnemonic::Movsb
            | Mnemonic::Movsw
            | Mnemonic::Movsq
            | Mnemonic::Stosb
            | Mnemonic::Stosw
            | Mnemonic::Stosd
            | Mnemonic::Stosq
            | Mnemonic::Lodsb
            | Mnemonic::Lodsw
            | Mnemonic::Lodsd
            | Mnemonic::Lodsq
            | Mnemonic::Movaps
            | Mnemonic::Movapd
            | Mnemonic::Movups
            | Mnemonic::Movupd
            | Mnemonic::Addss
            | Mnemonic::Addsd
            | Mnemonic::Subss
            | Mnemonic::Subsd
            | Mnemonic::Mulss
            | Mnemonic::Mulsd
            | Mnemonic::Divss
            | Mnemonic::Divsd
            | Mnemonic::Sqrtss
            | Mnemonic::Sqrtsd
            | Mnemonic::Minss
            | Mnemonic::Minsd
            | Mnemonic::Maxss
            | Mnemonic::Maxsd
            // no-flag writers that DO produce values
            | Mnemonic::Xchg
            | Mnemonic::Movbe
            | Mnemonic::Bswap
            | Mnemonic::Roundss
            | Mnemonic::Roundsd
            | Mnemonic::Roundps
            | Mnemonic::Roundpd
            | Mnemonic::Cdq
            | Mnemonic::Cqo
            | Mnemonic::Cwd
            | Mnemonic::Cwde
            | Mnemonic::Cbw
            | Mnemonic::Cdqe
            | Mnemonic::Cvtss2si
            | Mnemonic::Cvtsd2si
            | Mnemonic::Cvtsi2ss
            | Mnemonic::Cvtsi2sd
            | Mnemonic::Cvttss2si
            | Mnemonic::Cvttsd2si
            | Mnemonic::Push
            | Mnemonic::Pop
            | Mnemonic::Nop
            | Mnemonic::Int3
            | Mnemonic::Endbr64
            // the compare arms re-arm `pending` themselves
            | Mnemonic::Cmp
            | Mnemonic::Test
            | Mnemonic::Sete
            | Mnemonic::Setne
            | Mnemonic::Setl
            | Mnemonic::Setle
            | Mnemonic::Setg
            | Mnemonic::Setge
            | Mnemonic::Setb
            | Mnemonic::Setbe
            | Mnemonic::Seta
            | Mnemonic::Setae
            // The parity and sign setcc forms READ the flags like the ordered
            // ones. They were absent, so the pending FP compare was cleared
            // before the `setnp`/`setp` arm could read it and the NaN test that
            // compiles to `ucomiss; setnp` lost its condition entirely.
            | Mnemonic::Setp
            | Mnemonic::Setnp
            | Mnemonic::Sets
            | Mnemonic::Setns
            // BMI1/BMI2: every one of these is defined to leave EFLAGS alone
            // (that is why the `shlx`/`shrx`/`sarx`/`rorx` forms exist at all).
            | Mnemonic::Shlx
            | Mnemonic::Shrx
            | Mnemonic::Sarx
            | Mnemonic::Rorx
            | Mnemonic::Andn
            | Mnemonic::Blsi
            | Mnemonic::Blsr
            | Mnemonic::Blsmsk
            | Mnemonic::Bextr
            | Mnemonic::Bzhi
            | Mnemonic::Mulx
            // FMA: sets MXCSR, never EFLAGS
            | Mnemonic::Vfmadd132ss
            | Mnemonic::Vfmadd132sd
            | Mnemonic::Vfmadd213ss
            | Mnemonic::Vfmadd213sd
            | Mnemonic::Vfmadd231ss
            | Mnemonic::Vfmadd231sd
            | Mnemonic::Vfmadd132ps
            | Mnemonic::Vfmadd132pd
            | Mnemonic::Vfmadd213ps
            | Mnemonic::Vfmadd213pd
            | Mnemonic::Vfmadd231ps
            | Mnemonic::Vfmadd231pd
            // ---- packed / vector SSE + AVX ---------------------------------
            // None of these write EFLAGS, so a pending integer compare must
            // SURVIVE them (`mulps` between a `cmp` and its `jne` used to
            // clear the condition and lose the branch). Modelled by
            // `lift_packed_simd` above.
            | Mnemonic::Addps
            | Mnemonic::Addpd
            | Mnemonic::Vaddps
            | Mnemonic::Vaddpd
            | Mnemonic::Vaddss
            | Mnemonic::Vaddsd
            | Mnemonic::Subps
            | Mnemonic::Subpd
            | Mnemonic::Vsubps
            | Mnemonic::Vsubpd
            | Mnemonic::Vsubss
            | Mnemonic::Vsubsd
            | Mnemonic::Mulps
            | Mnemonic::Mulpd
            | Mnemonic::Vmulps
            | Mnemonic::Vmulpd
            | Mnemonic::Vmulss
            | Mnemonic::Vmulsd
            | Mnemonic::Divps
            | Mnemonic::Divpd
            | Mnemonic::Vdivps
            | Mnemonic::Vdivpd
            | Mnemonic::Vdivss
            | Mnemonic::Vdivsd
            | Mnemonic::Sqrtps
            | Mnemonic::Sqrtpd
            | Mnemonic::Vsqrtps
            | Mnemonic::Vsqrtpd
            | Mnemonic::Vsqrtss
            | Mnemonic::Vsqrtsd
            | Mnemonic::Minps
            | Mnemonic::Minpd
            | Mnemonic::Vminps
            | Mnemonic::Vminpd
            | Mnemonic::Vminss
            | Mnemonic::Vminsd
            | Mnemonic::Maxps
            | Mnemonic::Maxpd
            | Mnemonic::Vmaxps
            | Mnemonic::Vmaxpd
            | Mnemonic::Vmaxss
            | Mnemonic::Vmaxsd
            | Mnemonic::Cvtdq2ps
            | Mnemonic::Vcvtdq2ps
            | Mnemonic::Cvtdq2pd
            | Mnemonic::Vcvtdq2pd
            | Mnemonic::Cvtps2dq
            | Mnemonic::Vcvtps2dq
            | Mnemonic::Cvttps2dq
            | Mnemonic::Vcvttps2dq
            | Mnemonic::Cvtpd2dq
            | Mnemonic::Vcvtpd2dq
            | Mnemonic::Cvttpd2dq
            | Mnemonic::Vcvttpd2dq
            | Mnemonic::Cvtps2pd
            | Mnemonic::Vcvtps2pd
            | Mnemonic::Cvtpd2ps
            | Mnemonic::Vcvtpd2ps
            | Mnemonic::Vcvtsi2ss
            | Mnemonic::Vcvtsi2sd
            // (Movaps/Movapd/Movups/Movupd/Movd/Movq are already handled by
            // the scalar SIMD-move arm above — listing them here was an
            // unreachable pattern.)
            | Mnemonic::Movdqa
            | Mnemonic::Movdqu
            | Mnemonic::Vmovaps
            | Mnemonic::Vmovapd
            | Mnemonic::Vmovups
            | Mnemonic::Vmovupd
            | Mnemonic::Vmovdqa
            | Mnemonic::Vmovdqu
            | Mnemonic::Vmovd
            | Mnemonic::Vmovq
            | Mnemonic::Pandn
            | Mnemonic::Vpandn
            // The SSE `not-and` twin, now lowered by the same arm.
            | Mnemonic::Andnps
            | Mnemonic::Vandnps
            // ---- lane-arrangement family (see `lift_lane_simd`) -------------
            // None of these writes EFLAGS: they are data movement inside the
            // vector file, so a pending integer compare must SURVIVE one
            // (`cmp eax, ecx; shufps xmm0, xmm1, 0x1b; jne L` is a real
            // sequence and used to lose the branch condition).
            | Mnemonic::Shufps
            | Mnemonic::Vshufps
            | Mnemonic::Shufpd
            | Mnemonic::Vshufpd
            | Mnemonic::Pshufd
            | Mnemonic::Vpshufd
            | Mnemonic::Pshuflw
            | Mnemonic::Vpshuflw
            | Mnemonic::Pshufhw
            | Mnemonic::Vpshufhw
            | Mnemonic::Psrldq
            | Mnemonic::Vpsrldq
            | Mnemonic::Pslldq
            | Mnemonic::Vpslldq
            | Mnemonic::Pextrb
            | Mnemonic::Vpextrb
            | Mnemonic::Pextrw
            | Mnemonic::Vpextrw
            | Mnemonic::Pextrd
            | Mnemonic::Vpextrd
            | Mnemonic::Pextrq
            | Mnemonic::Vpextrq
            | Mnemonic::Pinsrb
            | Mnemonic::Vpinsrb
            | Mnemonic::Pinsrw
            | Mnemonic::Vpinsrw
            | Mnemonic::Pinsrd
            | Mnemonic::Vpinsrd
            | Mnemonic::Pinsrq
            | Mnemonic::Vpinsrq
            | Mnemonic::Unpcklps
            | Mnemonic::Vunpcklps
            | Mnemonic::Unpckhps
            | Mnemonic::Vunpckhps
            | Mnemonic::Unpcklpd
            | Mnemonic::Vunpcklpd
            | Mnemonic::Unpckhpd
            | Mnemonic::Vunpckhpd
            | Mnemonic::Punpcklqdq
            | Mnemonic::Vpunpcklqdq
            | Mnemonic::Punpckhqdq
            | Mnemonic::Vpunpckhqdq
            | Mnemonic::Punpckldq
            | Mnemonic::Vpunpckldq
            | Mnemonic::Punpckhdq
            | Mnemonic::Vpunpckhdq
            | Mnemonic::Punpcklwd
            | Mnemonic::Vpunpcklwd
            | Mnemonic::Punpckhwd
            | Mnemonic::Vpunpckhwd
            | Mnemonic::Punpcklbw
            | Mnemonic::Vpunpcklbw
            | Mnemonic::Punpckhbw
            | Mnemonic::Vpunpckhbw
            | Mnemonic::Paddb
            | Mnemonic::Vpaddb
            | Mnemonic::Paddw
            | Mnemonic::Vpaddw
            | Mnemonic::Paddd
            | Mnemonic::Vpaddd
            | Mnemonic::Paddq
            | Mnemonic::Vpaddq
            | Mnemonic::Psubb
            | Mnemonic::Vpsubb
            | Mnemonic::Psubw
            | Mnemonic::Vpsubw
            | Mnemonic::Psubd
            | Mnemonic::Vpsubd
            | Mnemonic::Psubq
            | Mnemonic::Vpsubq
            | Mnemonic::Pcmpeqb
            | Mnemonic::Vpcmpeqb
            | Mnemonic::Pcmpeqw
            | Mnemonic::Vpcmpeqw
            | Mnemonic::Pcmpeqd
            | Mnemonic::Vpcmpeqd
            | Mnemonic::Pcmpeqq
            | Mnemonic::Vpcmpeqq
            // ---- conditional moves: the flag READERS that were MISSING ------
            // A `cmovcc` READS the flags and writes none, exactly like the
            // `setcc` forms above — and this function's own doc comment always
            // claimed it was covered. It was not, and the consequence was not a
            // lost condition here or there: the gate above clears `pending`
            // BEFORE the `match` reaches the `Cmovcc` arm, so that arm's
            // `materialize_pending(pending_fp.or(pending))` always found `None`
            // and EVERY `cmp; cmovcc` in the binary materialized
            // `Compare(kind, Unknown, 0)` — the whole wrong-logic class, from
            // one missing list entry.
            //
            // Listing them also keeps the compare live PAST the move, which is
            // correct: `cmp; cmovne; jne L` re-tests the same flags on the
            // hardware, so a later `jcc` must still be able to read them.
            | Mnemonic::Cmovo
            | Mnemonic::Cmovno
            | Mnemonic::Cmovb
            | Mnemonic::Cmovae
            | Mnemonic::Cmove
            | Mnemonic::Cmovne
            | Mnemonic::Cmovbe
            | Mnemonic::Cmova
            | Mnemonic::Cmovs
            | Mnemonic::Cmovns
            | Mnemonic::Cmovp
            | Mnemonic::Cmovnp
            | Mnemonic::Cmovl
            | Mnemonic::Cmovge
            | Mnemonic::Cmovle
            | Mnemonic::Cmovg
            // the overflow setcc pair (the ordered ones are listed above)
            | Mnemonic::Seto
            | Mnemonic::Setno
            // ---- pack/scalar vector moves (neither EFLAGS nor MXCSR) --------
            | Mnemonic::Movhps
            | Mnemonic::Movlps
            | Mnemonic::Movhpd
            | Mnemonic::Movlpd
            | Mnemonic::Movlhps
            | Mnemonic::Movhlps
            | Mnemonic::Movshdup
            | Mnemonic::Movsldup
            | Mnemonic::Movddup
            | Mnemonic::Movntps
            | Mnemonic::Movntpd
            | Mnemonic::Movntdq
            | Mnemonic::Movntdqa
            | Mnemonic::Movnti
            | Mnemonic::Movntq
            | Mnemonic::Vmovhps
            | Mnemonic::Vmovlps
            | Mnemonic::Vmovhpd
            | Mnemonic::Vmovlpd
            | Mnemonic::Vmovlhps
            | Mnemonic::Vmovhlps
            | Mnemonic::Vmovshdup
            | Mnemonic::Vmovsldup
            | Mnemonic::Vmovddup
            | Mnemonic::Vmovntps
            | Mnemonic::Vmovntpd
            | Mnemonic::Vmovntdq
            | Mnemonic::Vmovntdqa
            // ---- packed integer logical / shift / mask family ---------------
            // `pandn` was the ONLY logical op listed, so a pending integer
            // compare was destroyed by any `pand`/`por`/`pxor` sitting between
            // it and its branch — the same defect the FP family above already
            // documents for `mulps`. None of these writes EFLAGS.
            | Mnemonic::Pand
            | Mnemonic::Vpand
            | Mnemonic::Por
            | Mnemonic::Vpor
            | Mnemonic::Pxor
            | Mnemonic::Vpxor
            | Mnemonic::Psllw
            | Mnemonic::Vpsllw
            | Mnemonic::Pslld
            | Mnemonic::Vpslld
            | Mnemonic::Psllq
            | Mnemonic::Vpsllq
            | Mnemonic::Psrlw
            | Mnemonic::Vpsrlw
            | Mnemonic::Psrld
            | Mnemonic::Vpsrld
            | Mnemonic::Psrlq
            | Mnemonic::Vpsrlq
            | Mnemonic::Psraw
            | Mnemonic::Vpsraw
            | Mnemonic::Psrad
            | Mnemonic::Vpsrad
            | Mnemonic::Pmovmskb
            | Mnemonic::Vpmovmskb
            | Mnemonic::Movmskps
            | Mnemonic::Movmskpd
            | Mnemonic::Pmuludq
            | Mnemonic::Vpmuludq
            | Mnemonic::Pmaddwd
            | Mnemonic::Vpmaddwd
            | Mnemonic::Pavgb
            | Mnemonic::Vpavgb
            | Mnemonic::Pavgw
            | Mnemonic::Vpavgw
            | Mnemonic::Pabsb
            | Mnemonic::Vpabsb
            | Mnemonic::Pabsw
            | Mnemonic::Vpabsw
            | Mnemonic::Pabsd
            | Mnemonic::Vpabsd
            | Mnemonic::Pminsb
            | Mnemonic::Vpminsb
            | Mnemonic::Pminsw
            | Mnemonic::Vpminsw
            | Mnemonic::Pminsd
            | Mnemonic::Vpminsd
            | Mnemonic::Pminub
            | Mnemonic::Vpminub
            | Mnemonic::Pminuw
            | Mnemonic::Vpminuw
            | Mnemonic::Pminud
            | Mnemonic::Vpminud
            | Mnemonic::Pmaxsb
            | Mnemonic::Vpmaxsb
            | Mnemonic::Pmaxsw
            | Mnemonic::Vpmaxsw
            | Mnemonic::Pmaxsd
            | Mnemonic::Vpmaxsd
            | Mnemonic::Pmaxub
            | Mnemonic::Vpmaxub
            | Mnemonic::Pmaxuw
            | Mnemonic::Vpmaxuw
            | Mnemonic::Pmaxud
            | Mnemonic::Vpmaxud
            | Mnemonic::Pshufb
            | Mnemonic::Vpshufb
            | Mnemonic::Palignr
            | Mnemonic::Vpalignr
            | Mnemonic::Pblendw
            | Mnemonic::Vpblendw
            // ---- fences, prefetch, cache, and x87 state ---------------------
            // Each of these is already handled by a value-preserving arm; it
            // was simply absent here, so each one silently cleared a live
            // compare. `Rdtsc`/`Rdtscp` write EDX:EAX, not EFLAGS.
            | Mnemonic::Lfence
            | Mnemonic::Mfence
            | Mnemonic::Sfence
            | Mnemonic::Pause
            | Mnemonic::Clflush
            | Mnemonic::Clflushopt
            | Mnemonic::Prefetchnta
            | Mnemonic::Prefetcht0
            | Mnemonic::Prefetcht1
            | Mnemonic::Prefetcht2
            | Mnemonic::Prefetchw
            | Mnemonic::Vzeroupper
            | Mnemonic::Vzeroall
            | Mnemonic::Emms
            | Mnemonic::Rdtsc
            | Mnemonic::Rdtscp
            // x87 data movement and arithmetic set the FPU status word, never
            // EFLAGS. (`fcomi`/`fucomi` are deliberately NOT here — those DO
            // write EFLAGS, and listing one would fabricate a condition.)
            | Mnemonic::Fld
            | Mnemonic::Fst
            | Mnemonic::Fstp
            | Mnemonic::Fild
            | Mnemonic::Fistp
            | Mnemonic::Fldz
            | Mnemonic::Fld1
            | Mnemonic::Fxch
            | Mnemonic::Fadd
            | Mnemonic::Fsub
            | Mnemonic::Fsubr
            | Mnemonic::Fmul
            | Mnemonic::Fdiv
            | Mnemonic::Fdivr
            | Mnemonic::Fabs
            | Mnemonic::Fchs
            | Mnemonic::Fsqrt
            | Mnemonic::Fnstsw
            | Mnemonic::Fstsw
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lift a bare instruction stream at a known RVA under the SysV ABI.
    fn lift(bytes: &[u8]) -> Vec<Insn> {
        lift_x86_64(bytes, 0x1000, 1024, X86Abi::SysV)
            .expect("x86 lift must not abort")
            .insns
    }

    fn dst_reg(is: &[Insn]) -> Option<&str> {
        match is.first().and_then(|i| i.operands.first()) {
            Some(Operand::Register(n)) => Some(n.as_str()),
            _ => None,
        }
    }

    #[test]
    fn packed_fp_arithmetic_is_modelled() {
        // 0F 58 C1  addps xmm0, xmm1
        let v = lift(&[0x0F, 0x58, 0xC1]);
        assert!(
            matches!(&v[0].op, Op::FloatArith(FloatArithKind::Add)),
            "addps must be a real add, got {:?}",
            v[0]
        );
        assert_eq!(dst_reg(&v), Some("vf0"));
        // 0F 5C C1 subps / 0F 59 C1 mulps / 0F 5E C1 divps
        assert!(matches!(
            &lift(&[0x0F, 0x5C, 0xC1])[0].op,
            Op::FloatArith(FloatArithKind::Sub)
        ));
        assert!(matches!(
            &lift(&[0x0F, 0x59, 0xC1])[0].op,
            Op::FloatArith(FloatArithKind::Mul)
        ));
        assert!(matches!(
            &lift(&[0x0F, 0x5E, 0xC1])[0].op,
            Op::FloatArith(FloatArithKind::Div)
        ));
        // 66 0F 58 C1  addpd xmm0, xmm1
        assert!(matches!(
            &lift(&[0x66, 0x0F, 0x58, 0xC1])[0].op,
            Op::FloatArith(FloatArithKind::Add)
        ));
        // 0F 51 C1  sqrtps xmm0, xmm1
        assert!(matches!(
            &lift(&[0x0F, 0x51, 0xC1])[0].op,
            Op::FloatArith(FloatArithKind::Sqrt)
        ));
    }

    #[test]
    fn packed_conversions_are_modelled() {
        // 0F 5B C1  cvtdq2ps xmm0, xmm1 — the largest unmodelled x64 family
        let v = lift(&[0x0F, 0x5B, 0xC1]);
        assert!(
            matches!(&v[0].op, Op::Convert(Conversion::IntToFloat32)),
            "cvtdq2ps must convert, got {:?}",
            v[0]
        );
        // 0F 5A C1  cvtps2pd
        assert!(matches!(
            &lift(&[0x0F, 0x5A, 0xC1])[0].op,
            Op::Convert(Conversion::Float32ToFloat64)
        ));
        // 66 0F 5A C1  cvtpd2ps
        assert!(matches!(
            &lift(&[0x66, 0x0F, 0x5A, 0xC1])[0].op,
            Op::Convert(Conversion::Float64ToFloat32)
        ));
        // F3 0F E6 C1  cvtdq2pd
        assert!(matches!(
            &lift(&[0xF3, 0x0F, 0xE6, 0xC1])[0].op,
            Op::Convert(Conversion::IntToFloat64)
        ));
        // 66 0F 5B C1  cvtps2dq
        assert!(matches!(
            &lift(&[0x66, 0x0F, 0x5B, 0xC1])[0].op,
            Op::Convert(Conversion::FloatToInt)
        ));
    }

    #[test]
    fn vector_moves_are_real_transfers() {
        // 0F 28 C1  movaps xmm0, xmm1
        let v = lift(&[0x0F, 0x28, 0xC1]);
        assert!(
            matches!(&v[0].op, Op::Move),
            "movaps must be a Move, got {:?}",
            v[0]
        );
        // 0F 10 C1  movups xmm0, xmm1
        assert!(matches!(&lift(&[0x0F, 0x10, 0xC1])[0].op, Op::Move));
    }

    #[test]
    fn packed_min_max_is_a_select() {
        // 0F 5D C1  minps xmm0, xmm1
        let v = lift(&[0x0F, 0x5D, 0xC1]);
        assert!(
            v.iter()
                .any(|i| matches!(&i.op, Op::Select(SelectVariant::Identity))),
            "minps must be compare+select, got {:?}",
            v
        );
        // 0F 5F C1  maxps xmm0, xmm1
        let v = lift(&[0x0F, 0x5F, 0xC1]);
        assert!(
            v.iter()
                .any(|i| matches!(&i.op, Op::Select(SelectVariant::Identity)))
        );
    }

    #[test]
    fn pandn_is_modelled_as_not_and() {
        // 66 0F DF C1  pandn xmm0, xmm1
        let v = lift(&[0x66, 0x0F, 0xDF, 0xC1]);
        assert!(
            v.iter().any(|i| matches!(&i.op, Op::Arith(ArithKind::And))),
            "pandn must produce an And, got {:?}",
            v
        );
    }

    // -----------------------------------------------------------------------
    // Lane-addressed vector ops: the exact mapping of every modelled mnemonic
    // -----------------------------------------------------------------------
    //
    // Each test names the instruction's OWN lane table and nothing else. These
    // are the facts that used to be a string (`Op::Simd("__Shufps")`) and are
    // now data a consumer can fold — so a wrong table here is a wrong VALUE,
    // which is exactly what the tests are for.

    /// The single lane-arrange op a lift produced, as `(elem_bytes, lanes)`.
    fn arrange_of(v: &[Insn]) -> (u8, Vec<LanePick>) {
        match &v[0].op {
            Op::SimdLane(SimdOp::Arrange { elem_bytes, lanes }) => (*elem_bytes, lanes.clone()),
            other => panic!("expected a lane arrange, got {other:?} — insn {v:?}"),
        }
    }

    /// Every modelled lowering must stay INSIDE its source register: a pick of
    /// lane 5 from a 4-lane register would be a wrong value, not a stub.
    fn assert_in_range(v: &[Insn]) {
        match &v[0].op {
            Op::SimdLane(op) => assert!(
                op.source_lanes_in_range(),
                "a lowering selected a lane that does not exist: {op:?}"
            ),
            other => panic!("expected a lane op, got {other:?}"),
        }
    }

    /// `shufps xmm0, xmm1, 0x1b` — 0x1b = 0b00_01_10_11: out[0] = d[3],
    /// out[1] = d[2], out[2] = s[1], out[3] = s[0].
    #[test]
    fn shufps_selects_its_documented_lanes() {
        // 0F C6 C1 1B
        let v = lift(&[0x0F, 0xC6, 0xC1, 0x1B]);
        assert_eq!(
            arrange_of(&v),
            (
                4,
                vec![
                    LanePick::A(3),
                    LanePick::A(2),
                    LanePick::B(1),
                    LanePick::B(0)
                ]
            )
        );
        assert_eq!(
            v[0].operands.len(),
            3,
            "dst, source A, source B: {:?}",
            v[0]
        );
        assert_in_range(&v);
    }

    /// `shufps xmm0, xmm1, 0x00` broadcasts lane 0 of each half.
    #[test]
    fn shufps_zero_broadcasts_lane_zero() {
        let v = lift(&[0x0F, 0xC6, 0xC1, 0x00]);
        assert_eq!(
            arrange_of(&v),
            (
                4,
                vec![
                    LanePick::A(0),
                    LanePick::A(0),
                    LanePick::B(0),
                    LanePick::B(0)
                ]
            )
        );
    }

    /// `shufps xmm0, xmm0, 0xe4` is the identity permutation, and it must BE
    /// the identity lane list — the case a folding consumer depends on.
    #[test]
    fn shufps_identity_permutation_is_the_identity_lane_list() {
        // 0F C6 C0 E4
        let v = lift(&[0x0F, 0xC6, 0xC0, 0xE4]);
        assert_eq!(
            arrange_of(&v),
            (
                4,
                vec![
                    LanePick::A(0),
                    LanePick::A(1),
                    LanePick::B(2),
                    LanePick::B(3)
                ]
            )
        );
    }

    /// `shufpd xmm0, xmm1, 0x1` takes d's HIGH quarter into lane 0 and s's LOW
    /// quarter into lane 1.
    #[test]
    fn shufpd_selects_its_two_quarters() {
        // 66 0F C6 C1 01
        let v = lift(&[0x66, 0x0F, 0xC6, 0xC1, 0x01]);
        assert_eq!(arrange_of(&v), (8, vec![LanePick::A(1), LanePick::B(0)]));
        assert_in_range(&v);
    }

    /// `pshufd xmm0, xmm1, 0x1b` permutes all four DWORDs of its single source.
    #[test]
    fn pshufd_permutes_four_dwords() {
        // 66 0F 70 C1 1B
        let v = lift(&[0x66, 0x0F, 0x70, 0xC1, 0x1B]);
        assert_eq!(
            arrange_of(&v),
            (
                4,
                vec![
                    LanePick::A(3),
                    LanePick::A(2),
                    LanePick::A(1),
                    LanePick::A(0)
                ]
            )
        );
        assert_eq!(v[0].operands.len(), 2, "one source only: {:?}", v[0]);
        assert_in_range(&v);
    }

    /// `pshuflw` permutes the LOW four words and passes the high four through —
    /// the pass-through is part of the semantics, not an omission.
    #[test]
    fn pshuflw_permutes_the_low_words_and_keeps_the_high_ones() {
        // F2 0F 70 C1 1B
        let v = lift(&[0xF2, 0x0F, 0x70, 0xC1, 0x1B]);
        assert_eq!(
            arrange_of(&v),
            (
                2,
                vec![
                    LanePick::A(3),
                    LanePick::A(2),
                    LanePick::A(1),
                    LanePick::A(0),
                    LanePick::A(4),
                    LanePick::A(5),
                    LanePick::A(6),
                    LanePick::A(7),
                ]
            )
        );
        assert_in_range(&v);
    }

    /// `pshufhw` is the mirror image.
    #[test]
    fn pshufhw_permutes_the_high_words_and_keeps_the_low_ones() {
        // F3 0F 70 C1 1B
        let v = lift(&[0xF3, 0x0F, 0x70, 0xC1, 0x1B]);
        assert_eq!(
            arrange_of(&v),
            (
                2,
                vec![
                    LanePick::A(0),
                    LanePick::A(1),
                    LanePick::A(2),
                    LanePick::A(3),
                    LanePick::A(7),
                    LanePick::A(6),
                    LanePick::A(5),
                    LanePick::A(4),
                ]
            )
        );
        assert_in_range(&v);
    }

    /// `psrldq xmm0, 4` moves byte 4 into byte 0 and zero-fills the top four.
    #[test]
    fn psrldq_shifts_by_bytes_and_zero_fills() {
        // 66 0F 73 D8 04
        let v = lift(&[0x66, 0x0F, 0x73, 0xD8, 0x04]);
        let (elem, lanes) = arrange_of(&v);
        assert_eq!(elem, 1);
        assert_eq!(lanes.len(), 16, "a byte shift addresses sixteen byte lanes");
        for (i, l) in lanes.iter().take(12).enumerate() {
            assert_eq!(*l, LanePick::A((i + 4) as u8), "output byte {i}");
        }
        assert!(
            lanes[12..].iter().all(|l| *l == LanePick::Zero),
            "top four zero-fill"
        );
        assert_in_range(&v);
    }

    /// The mirror form, and the saturation case: a shift of the whole register
    /// is all zeroes, never a wrapped lane 0.
    #[test]
    fn pslldq_shifts_left_and_a_full_shift_is_zero() {
        // 66 0F 73 F8 04
        let v = lift(&[0x66, 0x0F, 0x73, 0xF8, 0x04]);
        let (elem, lanes) = arrange_of(&v);
        assert_eq!(elem, 1);
        assert!(lanes[..4].iter().all(|l| *l == LanePick::Zero));
        for (i, l) in lanes[4..].iter().enumerate() {
            assert_eq!(*l, LanePick::A(i as u8), "output byte {}", i + 4);
        }
        // 66 0F 73 D8 10 — a shift by 16 discards every byte.
        let v = lift(&[0x66, 0x0F, 0x73, 0xD8, 0x10]);
        let (_, lanes) = arrange_of(&v);
        assert!(
            lanes.iter().all(|l| *l == LanePick::Zero),
            "all zero-fill: {lanes:?}"
        );
        assert_in_range(&v);
    }

    /// `pextrw eax, xmm0, 3` extracts WORD lane 3 into a GPR.
    #[test]
    fn pextrw_extracts_a_word_lane_into_a_gpr() {
        // 66 0F C5 C0 03
        let v = lift(&[0x66, 0x0F, 0xC5, 0xC0, 0x03]);
        assert_eq!(arrange_of(&v), (2, vec![LanePick::A(3)]));
        assert!(
            matches!(&v[0].operands[0], Operand::Register(n) if n == RET_REG),
            "the extract writes a GPR, not a vector register: {:?}",
            v[0]
        );
        assert_in_range(&v);
    }

    /// `pinsrw xmm0, eax, 3` replaces WORD lane 3 and keeps every other lane.
    #[test]
    fn pinsrw_inserts_into_one_word_lane() {
        // 66 0F C4 C0 03
        let v = lift(&[0x66, 0x0F, 0xC4, 0xC0, 0x03]);
        assert_eq!(
            arrange_of(&v),
            (
                2,
                vec![
                    LanePick::A(0),
                    LanePick::A(1),
                    LanePick::A(2),
                    LanePick::B(0),
                    LanePick::A(4),
                    LanePick::A(5),
                    LanePick::A(6),
                    LanePick::A(7),
                ]
            )
        );
        assert_in_range(&v);
    }

    /// The interleave family, one table per lane width.
    #[test]
    fn the_interleave_family_has_the_documented_tables() {
        // 0F 14 C1  unpcklps
        assert_eq!(
            arrange_of(&lift(&[0x0F, 0x14, 0xC1])),
            (
                4,
                vec![
                    LanePick::A(0),
                    LanePick::B(0),
                    LanePick::A(1),
                    LanePick::B(1)
                ]
            )
        );
        // 0F 15 C1  unpckhps
        assert_eq!(
            arrange_of(&lift(&[0x0F, 0x15, 0xC1])),
            (
                4,
                vec![
                    LanePick::A(2),
                    LanePick::B(2),
                    LanePick::A(3),
                    LanePick::B(3)
                ]
            )
        );
        // 66 0F 14 C1  unpcklpd
        assert_eq!(
            arrange_of(&lift(&[0x66, 0x0F, 0x14, 0xC1])),
            (8, vec![LanePick::A(0), LanePick::B(0)])
        );
        // 66 0F 15 C1  unpckhpd
        assert_eq!(
            arrange_of(&lift(&[0x66, 0x0F, 0x15, 0xC1])),
            (8, vec![LanePick::A(1), LanePick::B(1)])
        );
        // 66 0F 6C C1  punpcklqdq
        assert_eq!(
            arrange_of(&lift(&[0x66, 0x0F, 0x6C, 0xC1])),
            (8, vec![LanePick::A(0), LanePick::B(0)])
        );
        // 66 0F 6D C1  punpckhqdq
        assert_eq!(
            arrange_of(&lift(&[0x66, 0x0F, 0x6D, 0xC1])),
            (8, vec![LanePick::A(1), LanePick::B(1)])
        );
        // 66 0F 60 C1  punpcklbw — sixteen byte lanes, interleaved low halves
        let v = lift(&[0x66, 0x0F, 0x60, 0xC1]);
        let (elem, lanes) = arrange_of(&v);
        assert_eq!(elem, 1);
        assert_eq!(lanes.len(), 16);
        for i in 0..8u8 {
            assert_eq!(lanes[i as usize * 2], LanePick::A(i));
            assert_eq!(lanes[i as usize * 2 + 1], LanePick::B(i));
        }
        assert_in_range(&v);
    }

    /// Packed INTEGER arithmetic is per-lane: a register-wide add would carry
    /// between the lanes and compute a different value.
    #[test]
    fn packed_integer_arithmetic_is_per_lane() {
        // 66 0F FE C1  paddd
        match &lift(&[0x66, 0x0F, 0xFE, 0xC1])[0].op {
            Op::SimdLane(SimdOp::PackedArith { elem_bytes, kind }) => {
                assert_eq!(*elem_bytes, 4);
                assert_eq!(*kind, ArithKind::Add);
            }
            other => panic!("paddd must be a packed add, got {other:?}"),
        }
        // 66 0F FB C1  psubq
        match &lift(&[0x66, 0x0F, 0xFB, 0xC1])[0].op {
            Op::SimdLane(SimdOp::PackedArith { elem_bytes, kind }) => {
                assert_eq!(*elem_bytes, 8);
                assert_eq!(*kind, ArithKind::Sub);
            }
            other => panic!("psubq must be a packed sub, got {other:?}"),
        }
    }

    /// `pcmpeqd xmm0, xmm0` is all-ones in every lane; a compare of two
    /// DIFFERENT values is a lane mask this model cannot name, so it must stay
    /// a value-preserving stub rather than become a whole-register compare.
    #[test]
    fn a_self_compare_is_all_ones_and_a_real_compare_is_untouched() {
        // 66 0F 76 C0  pcmpeqd xmm0, xmm0
        let v = lift(&[0x66, 0x0F, 0x76, 0xC0]);
        assert!(
            matches!(&v[0].op, Op::Move)
                && matches!(v[0].operands.get(1), Some(Operand::Immediate(-1))),
            "a self-compare is all-ones, got {:?}",
            v[0]
        );
        // 66 0F 76 C1  pcmpeqd xmm0, xmm1
        let v = lift(&[0x66, 0x0F, 0x76, 0xC1]);
        assert!(
            matches!(&v[0].op, Op::Simd(_)),
            "a mask-producing compare keeps its stub, got {:?}",
            v[0]
        );
    }

    /// An AVX three-operand shuffle resolves BOTH sources as operands 1 and 2,
    /// while the SSE form keeps the destination as source A.
    #[test]
    fn an_avx_shuffle_resolves_both_sources_explicitly() {
        // C5 F0 C6 C2 1B  vshufps xmm0, xmm1, xmm2, 0x1b
        let v = lift(&[0xC5, 0xF0, 0xC6, 0xC2, 0x1B]);
        assert_eq!(
            arrange_of(&v),
            (
                4,
                vec![
                    LanePick::A(3),
                    LanePick::A(2),
                    LanePick::B(1),
                    LanePick::B(0)
                ]
            )
        );
        assert_eq!(v[0].operands.len(), 3);
        assert_eq!(v[0].operands[0], Operand::Register("vf0".into()));
        assert_eq!(v[0].operands[1], Operand::Register("vf1".into()));
        assert_eq!(v[0].operands[2], Operand::Register("vf2".into()));
    }

    /// A lane rearrangement does not write EFLAGS, so the compare behind a
    /// branch must SURVIVE it. Getting this wrong silently loses the branch
    /// condition (`if (? != 0)`), which is a wrong-logic defect, not a stub.
    #[test]
    fn a_lane_op_preserves_a_pending_compare() {
        // 39 C8        cmp eax, ecx
        // 0F C6 C1 1B  shufps xmm0, xmm1, 0x1b
        // 75 00        jne +0
        let v = lift(&[0x39, 0xC8, 0x0F, 0xC6, 0xC1, 0x1B, 0x75, 0x00]);
        assert!(
            v.iter().any(|i| matches!(i.op, Op::Compare(_))),
            "the compare must survive the shuffle: {v:?}"
        );
        assert!(
            v.iter().any(|i| matches!(i.op, Op::SimdLane(_))),
            "and the shuffle must be modelled: {v:?}"
        );
    }

    /// `sal` is `shl`: the same shift, and it must not be a stub.
    #[test]
    fn sal_is_the_same_left_shift_as_shl() {
        // D1 E0  sal eax, 1
        let v = lift(&[0xD1, 0xE0]);
        assert!(
            v.iter().any(|i| matches!(i.op, Op::Arith(ArithKind::Shl))),
            "sal must shift, got {v:?}"
        );
        assert!(
            !v.iter()
                .any(|i| matches!(&i.op, Op::Simd(n) if n.contains("Sal"))),
            "sal must not reach the catch-all: {v:?}"
        );
    }

    /// `andnps` is the SSE `pandn`: a whole-register not-and, modelled exactly.
    #[test]
    fn andnps_is_modelled_as_not_and() {
        // 0F 55 C1  andnps xmm0, xmm1
        let v = lift(&[0x0F, 0x55, 0xC1]);
        assert!(
            v.iter().any(|i| matches!(&i.op, Op::Arith(ArithKind::And))),
            "andnps must produce an And, got {v:?}"
        );
        assert!(
            !v.iter().any(|i| matches!(&i.op, Op::Simd(_))),
            "andnps must not reach the catch-all: {v:?}"
        );
    }

    /// The lane tables survive the round trip through `Op`'s own text form,
    /// which is what the run report and the verifier read.
    #[test]
    fn a_lane_op_renders_its_mapping() {
        let v = lift(&[0x0F, 0xC6, 0xC1, 0x1B]);
        let text = v[0].op.to_string();
        assert_eq!(text, "arrange32[a3,a2,b1,b0]");
        assert!(
            !text.contains("__"),
            "a modelled op must not read like an unmodelled intrinsic: {text}"
        );
    }

    // -----------------------------------------------------------------------
    // Condition survivors: the flag-preserving gate
    // -----------------------------------------------------------------------
    // These pin the wrong-logic class. A missing entry in `preserves_flags`
    // clears a live compare, and an over-broad one keeps a compare the machine
    // has already destroyed — the second is the dangerous direction, because
    // the emitted text still looks perfectly plausible.

    /// `cmp eax, ecx; cmovne edx, esi; jne L` — a `cmovcc` READS the flags, so
    /// it must render with the real condition, and the branch after it must
    /// still see the same compare. Before `Cmovcc` was listed in
    /// `preserves_flags`, the gate cleared `pending` before the `Cmovcc` arm
    /// ran, so BOTH consumers materialized `Compare(kind, Unknown, 0)` — one
    /// missing list entry lost every conditional move's condition in the
    /// binary.
    #[test]
    fn a_conditional_move_keeps_the_condition_for_itself_and_the_next_branch() {
        // 39 C8        cmp eax, ecx
        // 0F 45 D6     cmovne edx, esi
        // 75 00        jne .+2
        let v = lift(&[0x39, 0xC8, 0x0F, 0x45, 0xD6, 0x75, 0x00]);
        let compares: Vec<&Insn> = v
            .iter()
            .filter(|i| matches!(i.op, Op::Compare(_)))
            .collect();
        assert_eq!(
            compares.len(),
            2,
            "one materialized compare per flag consumer (cmov + branch): {v:?}"
        );
        for (n, c) in compares.iter().enumerate() {
            assert!(
                !matches!(c.operands.get(1), Some(Operand::Unknown)),
                "condition {n} lost its compare; `cmovcc` must be in \
                 `preserves_flags`: {c:?}"
            );
        }
    }

    /// `cmp; call helper; jne L` — a CALL clobbers EFLAGS, so the branch must
    /// NOT be bound to the pre-call compare. Its condition is honestly unknown
    /// (the callee sets the flags we cannot see).
    ///
    /// This is the fabrication direction: while the `Call` arm left `pending`
    /// intact, the branch rendered `if (a != b)` using operands the callee had
    /// already overwritten — wrong logic that no counter could see, because the
    /// text looks exactly like a correct null/comparison test.
    #[test]
    fn a_compare_must_not_survive_a_call() {
        // 39 C8              cmp eax, ecx
        // E8 F9 0F 00 00     call 0x2000
        // 75 00              jne .+2
        // C3                 ret
        let v = lift(&[0x39, 0xC8, 0xE8, 0xF9, 0x0F, 0x00, 0x00, 0x75, 0x00, 0xC3]);
        let compares: Vec<&Insn> = v
            .iter()
            .filter(|i| matches!(i.op, Op::Compare(_)))
            .collect();
        assert_eq!(
            compares.len(),
            1,
            "the `cmp` alone materializes nothing: {v:?}"
        );
        assert!(
            matches!(compares[0].operands.get(1), Some(Operand::Unknown)),
            "the branch's compare must not be the one the call destroyed: {:?}",
            compares[0]
        );
    }

    /// The `preserves_flags` table, checked in BOTH directions.
    ///
    /// Missing a flag-neutral mnemonic costs a condition (safe); listing a flag
    /// WRITER keeps a compare the hardware has destroyed (unsafe). The cmovcc
    /// gap showed this table cannot be trusted without a test, so the readers,
    /// the vector families and the writers are all named here.
    #[test]
    fn the_flag_preserving_table_covers_the_readers_and_rejects_the_writers() {
        for m in [
            // the readers: `cmovcc` was missing entirely
            Mnemonic::Cmovne,
            Mnemonic::Cmove,
            Mnemonic::Cmovg,
            Mnemonic::Cmovae,
            Mnemonic::Seto,
            Mnemonic::Setno,
            // packed integer logical / mask / shift (only `pandn` was listed)
            Mnemonic::Pand,
            Mnemonic::Por,
            Mnemonic::Pxor,
            Mnemonic::Psrld,
            Mnemonic::Pmovmskb,
            Mnemonic::Pshufb,
            Mnemonic::Palignr,
            Mnemonic::Vpand,
            Mnemonic::Vpor,
            Mnemonic::Vpxor,
            // moves, fences, prefetch, x87 data movement
            Mnemonic::Movntps,
            Mnemonic::Movhps,
            Mnemonic::Lfence,
            Mnemonic::Pause,
            Mnemonic::Vzeroupper,
            Mnemonic::Prefetchnta,
            Mnemonic::Fld,
        ] {
            assert!(
                preserves_flags(m),
                "{m:?} writes no EFLAGS and must be listed, or a live compare is \
                 cleared and the branch loses its condition"
            );
        }
        for m in [
            // flag WRITERS: listing any of these would fabricate a condition
            Mnemonic::Fcomi,
            Mnemonic::Fucomi,
            Mnemonic::Bt,
            Mnemonic::Bsr,
            Mnemonic::Popcnt,
            Mnemonic::Sahf,
            Mnemonic::Popf,
            Mnemonic::Adc,
            Mnemonic::Sbb,
        ] {
            assert!(
                !preserves_flags(m),
                "{m:?} writes EFLAGS and must NOT be listed: keeping its compare \
                 would bind a branch to a test the hardware destroyed"
            );
        }
    }
}
