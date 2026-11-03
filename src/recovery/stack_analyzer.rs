//! Native stack analysis (port of Cpp2IL `Analysis/StackAnalyzer.cs`).
//!
//! Tracks net push depth per block via worklist fixpoint, then rewrites
//! sp-relative memory operands into position-independent virtual registers
//! named `stack_N` — exactly Cpp2IL's scheme. ARM64 prologues mostly address
//! locals off a frame pointer copied from sp, so we also resolve those aliases.

use super::cfg::Cfg;
use super::ir::{Insn, Op, Operand};
use std::collections::{HashMap, HashSet};

pub struct StackAnalysis {
    /// Net stack delta at entry to each block.
    pub in_depth: Vec<i64>,
}

pub fn analyze(insns: &mut [Insn], cfg: &Cfg) -> StackAnalysis {
    analyze_with_ptr_size(insns, cfg, 8)
}

/// `ptr_size` selects the slot scale (8 for ARM64/x64, 4 for ARM32).
pub fn analyze_with_ptr_size(insns: &mut [Insn], cfg: &Cfg, ptr_size: u64) -> StackAnalysis {
    let mut in_state: Vec<Option<i64>> = vec![None; cfg.blocks.len()];
    let mut out_state: Vec<i64> = vec![0; cfg.blocks.len()];
    let mut insn_depth: HashMap<usize, i64> = HashMap::new();

    // Convergence budget: normal dataflow needs a few visits per block, but
    // a cycle with non-zero net ShiftStack (mislifted prologues, LDP/STP
    // offset loss) diverges forever — the old fixed 500k-per-block cap let
    // that burn CPU for seconds per method. Budget the TOTAL work instead;
    // when it runs out the depths are simply approximate.
    let total_budget: u64 = (cfg.blocks.len() as u64).saturating_mul(64) + 256;
    let mut budget = total_budget;

    // only revisit successors when the incoming depth actually changed
    // (standard dataflow convergence; prevents loop-thrash)
    let mut worklist: Vec<(usize, i64)> = vec![(0, 0)];
    while let Some((bi, incoming)) = worklist.pop() {
        if budget == 0 {
            break;
        }
        budget -= 1;
        if in_state[bi].is_some() {
            if in_state[bi] == Some(incoming) {
                continue;
            }
        }
        in_state[bi] = Some(incoming);

        let b = &cfg.blocks[bi];
        let mut depth = incoming;
        for i in b.start..b.end {
            insn_depth.insert(i, depth);
            if let Op::ShiftStack(d) = insns[i].op {
                depth += d;
            }
        }
        out_state[bi] = depth;

        for &succ in cfg.successors[bi].iter() {
            if in_state[succ] != Some(depth) {
                worklist.push((succ, depth));
            }
        }
    }

    excise_callee_save_round_trips(insns, &insn_depth);
    resolve_frame_aliases(insns, &insn_depth, ptr_size);
    rewrite_sp_operands(insns, &insn_depth, ptr_size);
    coalesce_hfa_stack_lanes(insns, ptr_size);
    coalesce_address_taken_buffers(insns, ptr_size);

    StackAnalysis {
        in_depth: out_state,
    }
}

/// Excise callee-saved register save/restore round-trips: the prologue's
/// `stp d9,d8,[sp,#-96]!` and every matching epilogue `ldp d9,d8,[sp],#96`
/// are pure ABI machinery — the saved register's incoming value is stored
/// and restored untouched, so the pair has NO source-level semantics. Left
/// in, they poison the pipeline two ways: the store READS the incoming
/// register (an SSA entry version never defined in-body → renders as
/// `v41 = v10` with v10 undefined), and the restore defines a version
/// nobody uses. Standard decompilers hide these entirely.
///
/// Pairing is by REGISTER IDENTITY (not stack slot): the epilogue's sp
/// writeback can be lifted before the loads, making depth-relative slot
/// arithmetic unreliable — but the register is exact on both sides.
/// Safety rules:
/// 1. Whitelist only CALLEE-SAVED registers (x19–x30, d/v8–15, rbx/rbp/
///    r12–r15, XMM6–15, …). Caller-saved spills (argument spills like
///    `str x0,[sp]`) carry live values across calls and must stay.
/// 2. A register with TWO prologue stores is skipped (frame-slot reuse).
/// 3. EVERY stack load of that register must be DEAD: following its
///    fall-through, the value is overwritten or the body exits without ever
///    being read, with no in-body branch or call in between. A load whose
///    value is read before any redefinition is a real spill-reload, not ABI
///    machinery. This used to be a 16-insn "terminator envelope" test that
///    accepted ANY `CondJump`/`Jump` as an epilogue, so a mid-body reload
///    followed by an ordinary null check was erased and every use read the
///    clobbered register instead (Initialize 0x189D9D4, `ldr x27,[sp,#0x10]`
///    after `mov x27,x0`).
/// 4. A register with zero stack loads (saved, never restored — early-
///    exit paths) loses just the store. Safe either way: a restore's
///    value is dead at exit, and mid-body uses read the SSA entry
///    version, which survives (only the store's own read disappears).
/// Runs BEFORE sp-rewrite so operands are still raw `Memory{base:sp}`;
/// erased slots never become `stack_N` locals at all.
pub(crate) fn excise_callee_save_round_trips(
    insns: &mut [Insn],
    _insn_depth: &HashMap<usize, i64>,
) -> usize {
    use super::verify::{LedgerKind, ledger_note};

    /// Callee-saved register names across the three supported ISAs, as the
    /// lifters name them. Matching is exact-name; FP views included per view
    /// width (arm64 names SIMD regs "vd8"/"vs8" by width, x86 "vf6").
    const CALLEE_SAVED: &[&str] = &[
        // ARM64 integer: x19–x28 + FP(x29) + LR(x30)
        "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        // ARM64 SIMD callee-saved v8–v15, per width view. `vs` identities
        // flatten lane k of vN to `vs(4*N+k)`, so only lanes 0 and 1 of
        // v8-v15 are callee-saved (`vs32/33`, `vs36/37`, …, `vs60/61`).
        "vd8", "vd9", "vd10", "vd11", "vd12", "vd13", "vd14", "vd15", "vs32", "vs33", "vs36",
        "vs37", "vs40", "vs41", "vs44", "vs45", "vs48", "vs49", "vs52", "vs53", "vs56", "vs57",
        "vs60", "vs61", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
        // x86-64 (SysV + MSVC union): rbx, rbp, r12–r15 (+rdi/rsi on MSVC)
        "rbx", "rbp", "r12", "r13", "r14", "r15", "rdi", "rsi",
        // …and the ABI-NORMALIZED names the x86 lifter actually emits
        // (`gpr_name_for_ordinal`): ordinal 3 = rbx → `g3`, 5 = rbp → `fp`,
        // 12–15 = r12–r15 → `g12`–`g15`, and 6/7 = rsi/rdi → `g6`/`g7` under
        // MSVC. SysV names ordinals 6/7 `x1`/`x0` (argument registers) and they
        // are deliberately NOT listed. Without these twins the raw names above
        // matched nothing x86 produced, so no x64 callee-save pair was ever
        // excised and every saved register stayed undefined (`g3` was the #1
        // residual origin at 603).
        "g3", "fp", "g6", "g7", "g12", "g13", "g14", "g15",
        // XMM6–15 callee-saved on Windows, harmless to include elsewhere
        "vf6", "vf7", "vf8", "vf9", "vf10", "vf11", "vf12", "vf13", "vf14", "vf15",
        // ARM32: r4–r11 + lr
        "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "lr", // x86-32
        "esi", "edi", "ebx", "ebp",
    ];
    let is_callee_saved = |r: &str| CALLEE_SAVED.contains(&r);
    let min_addr = insns.first().map(|i| i.address).unwrap_or(0);
    // Whether `insn` READS register `r`. A plain register destination is a
    // write; a memory/address destination still reads its base operands.
    fn operand_reads(op: &Operand, r: &str) -> bool {
        match op {
            Operand::Register(n) => n == r,
            Operand::Memory { base, index, .. } => {
                operand_reads(base, r)
                    || index.as_ref().is_some_and(|i| operand_reads(i, r))
            }
            Operand::AddressOf(inner) | Operand::ArrayLength(inner) => operand_reads(inner, r),
            Operand::ArrayAccess { array, index } => {
                operand_reads(array, r) || operand_reads(index, r)
            }
            Operand::InstanceField { base, .. } => operand_reads(base, r),
            _ => false,
        }
    }
    fn reads_reg(insn: &Insn, r: &str) -> bool {
        let dest = Insn::dest_position(&insn.op);
        insn.operands.iter().enumerate().any(|(i, op)| {
            if dest == Some(i) {
                !matches!(op, Operand::Register(_)) && operand_reads(op, r)
            } else {
                operand_reads(op, r)
            }
        })
    }
    fn writes_reg(insn: &Insn, r: &str) -> bool {
        Insn::dest_position(&insn.op)
            .and_then(|p| insn.operands.get(p))
            .is_some_and(|op| matches!(op, Operand::Register(n) if n == r))
    }
    // The loaded value is dead when its fall-through reaches a method exit
    // without reading it and without a branch/call that could take another
    // path. A pure redefinition also kills it. Any doubt keeps the load.
    let value_is_dead = |from: usize, r: &str| -> bool {
        for n in &insns[from + 1..] {
            if reads_reg(n, r) {
                return false;
            }
            if writes_reg(n, r) {
                return true;
            }
            match n.op {
                Op::Return | Op::Throw | Op::IndirectCall => return true,
                // Only an out-of-body/exit jump is a provable exit; an
                // in-body jump can reach a read on the other side.
                Op::Jump(t) if t >= insns.len() => return true,
                Op::Jump(_) | Op::CondJump(_) | Op::Switch(_) => return false,
                Op::Call | Op::CallVoid | Op::NewObj(_) => return false,
                _ => {}
            }
        }
        false
    };
    // A STACK-relative access, by frame register. This must accept the FRAME
    // POINTER as well as the stack pointer: x64 saves a callee-saved register
    // as `mov [rbp+0x20], rbx` once `lea rbp,[rsp+0x20]` has established the
    // frame, and arm64 as `str x19, [x29, #…]`. Recognising only `sp` meant the
    // pairs went unseen and the SAVE survived with its read of the incoming
    // (undefined) register — `g3`/`g6`/`g7` were the top three residual
    // undefined-local origins (603/257/238) for exactly this reason. The
    // frame register is only ever a frame base in this IR (`gpr_name_for_ordinal`
    // names RBP `fp`, and arm64 names x29 `x29`), so accepting it is exact.
    // A STACK-relative access, as `(is_frame_based, byte_addend)`. It must
    // accept the FRAME POINTER as well as the stack pointer: x64 saves a
    // callee-saved register as `mov [rbp+0x20], rbx` once `lea rbp,[rsp+0x20]`
    // has established the frame, and arm64 as `str x19, [x29, #…]`. Recognising
    // only `sp` meant the pairs went unseen and the SAVE survived with its read
    // of the incoming (undefined) register — `g3`/`g6`/`g7` were the top three
    // residual undefined-local origins (603/257/238). The frame register is
    // only ever a frame base in this IR (`gpr_name_for_ordinal` names RBP `fp`,
    // and arm64 names x29 `x29`), so accepting it is exact.
    let stack_ref = |o: &Operand| -> Option<(bool, i64)> {
        match o {
            Operand::Memory {
                base,
                index: None,
                addend,
                ..
            } => match base.as_ref() {
                Operand::Register(b) if matches!(b.as_str(), "sp" | "esp" | "rsp") => {
                    Some((false, *addend))
                }
                Operand::Register(b) if matches!(b.as_str(), "fp" | "rbp" | "x29") => {
                    Some((true, *addend))
                }
                _ => None,
            },
            _ => None,
        }
    };

    // Prologue stores: callee-saved register spilled to the stack within the
    // first 64 bytes of the body. One store per register, else skip it.
    let mut store_of: HashMap<String, (usize, bool, i64)> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();
    // Registers already (re)written in-body. A SAVE stores the register's
    // INCOMING value, so it must precede any in-body write; a store of a value
    // the body just computed is an ordinary local store, not a save. Counting
    // it as a second "prologue store" marked the register AMBIGUOUS and blocked
    // the excision entirely — `mov [rbp+0x20],rbx` (save) followed 0x20 bytes
    // later by `mov [rbp+0x28],ebx` (store the just-zeroed value) left the save
    // in place, reading the undefined incoming `rbx` (`g3` was 603 findings).
    let mut written: HashSet<String> = HashSet::new();
    for (i, insn) in insns.iter().enumerate() {
        if let Some(pos) = Insn::dest_position(&insn.op) {
            if let Some(Operand::Register(d)) = insn.operands.get(pos) {
                written.insert(d.clone());
            }
        }
        if insn.op != Op::Move {
            continue;
        }
        let (Some(dest), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            continue;
        };
        let Some((save_frame, save_addend)) = stack_ref(dest) else {
            continue;
        };
        let Operand::Register(r) = src else { continue };
        if !is_callee_saved(r) || insn.address.saturating_sub(min_addr) >= 64 {
            continue;
        }
        if written.contains(r) {
            continue;
        }
        if store_of.contains_key(r) {
            ambiguous.insert(r.clone());
        } else {
            store_of.insert(r.clone(), (i, save_frame, save_addend));
        }
    }
    if store_of.is_empty() {
        return 0;
    }

    // All stack loads per whitelisted register, anywhere in the body.
    let mut loads_of: HashMap<String, Vec<(usize, bool, i64)>> = HashMap::new();
    for (i, insn) in insns.iter().enumerate() {
        if insn.op != Op::Move {
            continue;
        }
        let (Some(dest), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            continue;
        };
        let Operand::Register(r) = dest else { continue };
        if !is_callee_saved(r) {
            continue;
        }
        let Some((load_frame, load_addend)) = stack_ref(src) else {
            continue;
        };
        loads_of
            .entry(r.clone())
            .or_default()
            .push((i, load_frame, load_addend));
    }

    let mut to_erase: HashSet<usize> = HashSet::new();
    for (r, &(si, save_frame, save_addend)) in &store_of {
        if ambiguous.contains(r) {
            continue;
        }
        let Some(idxs) = loads_of.get(r) else {
            // saved but never restored (early-exit paths): drop the store
            to_erase.insert(si);
            continue;
        };
        // A load from a DIFFERENT frame slot is a reload of another spill, not
        // this save's restore — it must not block the excision. Only frame-based
        // accesses are compared: an `sp`-based epilogue goes through a writeback
        // whose addend cannot be matched against the prologue's.
        let considered: Vec<usize> = idxs
            .iter()
            .filter(|(_, load_frame, load_addend)| {
                !(*load_frame && save_frame && *load_addend != save_addend)
            })
            .map(|(i, _, _)| *i)
            .collect();
        let dead: Vec<usize> = considered
            .iter()
            .copied()
            .filter(|&i| value_is_dead(i, r))
            .collect();
        to_erase.extend(dead.iter().copied());
        // The save can go only when no surviving load reads its value.
        if dead.len() == considered.len() {
            to_erase.insert(si);
        }
    }

    let n = to_erase.len();
    for i in to_erase {
        let addr = insns[i].address;
        ledger_note(addr, LedgerKind::CalleeSaveRoundTrip);
        insns[i] = Insn::nop(addr);
    }
    n
}

/// `mov x29, sp` / `add x29, sp, #k` records an alias; later `[x29 + N]`
/// becomes stack slot `(k + N - depth_at_copy) / ptr_size`.
///
/// WHAT COUNTS AS AN ALIAS. Only an instruction that puts a POINTER INTO THE
/// FRAME into a register:
///
///   * `mov xD, sp`              → `Move xD, Register("sp")`
///   * `add xD, sp, #k`          → `Move xD, AddressOf(Memory{base: sp, k})`
///     (and the plain `xD = sp` degenerate form, `AddressOf(sp)`)
///
/// A `Move xD, Memory { base: sp, addend: N }` is **not** that: it is a LOAD
/// of a stack VALUE — `ldr xD, [sp, #N]`, and in particular every `ldp`/
/// `pop` epilogue restore of a callee-saved register. Treating a load as an
/// alias says "this register holds sp", so from that point on EVERY
/// `[xD + N]` access was rebased into a bogus `stack_(N/8)` slot. Registers
/// are reused: the frame slot `ldp x24, x23, [sp]` restores x23 from the
/// frame, but x23 held a heap OBJECT pointer for the whole body, so each
/// `[x23 + 0x4c]` field access (the real one at 0x2530a0 of method 0x252d8c)
/// became `[stack_9 + 4]` — a different address entirely, which the verifier
/// reports as `memory offset mismatch: native +76, IR [4]` and which also
/// invented a never-defined `stack_9` local. The bare `Memory` form is
/// unambiguous in this IR: `lift.rs` wraps the address of a slot in
/// `AddressOf` precisely because `Memory` on its own is a load everywhere
/// else, so an `AddressOf` source is the only non-register shape that
/// yields a frame pointer.
///
/// An alias is dropped as soon as its register is REdefined by anything
/// else (a fresh `ldr x23, [x0]` moves the register off the frame). Without
/// that invalidation a single prologue alias would claim the register for
/// the rest of the body, including blocks where it holds a live value.
fn resolve_frame_aliases(insns: &mut [Insn], insn_depth: &HashMap<usize, i64>, ptr_size: u64) {
    // reg → (bias in bytes, stack depth at the copy)
    let mut aliases: HashMap<String, (i64, i64)> = HashMap::new();
    for i in 0..insns.len() {
        let here = insn_depth.get(&i).copied().unwrap_or(0);
        // 1. Consume: rewrite reads through a LIVE alias (uses in an
        //    instruction refer to the value that instruction reads, so this
        //    runs before the instruction's own definition is applied).
        for op in &mut insns[i].operands {
            rewrite_alias_operand(op, &aliases, here, ptr_size);
        }
        // 2. Update the alias set for this instruction's own destination.
        match frame_alias_def(&insns[i]) {
            Some((dst, bias)) => {
                aliases.insert(dst, (bias, here));
            }
            None => {
                if let Some(pos) = Insn::dest_position(&insns[i].op) {
                    if let Some(Operand::Register(d)) = insns[i].operands.get(pos) {
                        aliases.remove(d);
                    }
                }
            }
        }
    }
}

/// `xD = sp` (bias 0) or `xD = sp + k` (bias `k`) — see
/// [`resolve_frame_aliases`] for why the bare-`Memory` load form is excluded.
/// The register name is returned owned so the caller can mutate `insns`.
fn frame_alias_def(insn: &Insn) -> Option<(String, i64)> {
    if !matches!(insn.op, Op::Move) {
        return None;
    }
    let Operand::Register(dst) = insn.operands.first()? else {
        return None;
    };
    let bias = match insn.operands.get(1)? {
        // `mov xD, sp` — the value of sp itself.
        Operand::Register(b) if b == "sp" => 0,
        // `add xD, sp, #k` — sp plus a frame offset (address of a slot).
        Operand::AddressOf(inner) => match inner.as_ref() {
            Operand::Memory {
                base,
                index: None,
                addend,
                ..
            } => match base.as_ref() {
                Operand::Register(b) if b == "sp" => *addend,
                _ => return None,
            },
            Operand::Register(b) if b == "sp" => 0,
            _ => return None,
        },
        _ => return None,
    };
    Some((dst.clone(), bias))
}

/// A stack operand for an access at byte offset `slot*scale + off`.
///
/// `stack_N` names a memory CELL, and cells are `scale` bytes wide. An access
/// that lands INSIDE a cell — a narrower field of a slot whose address was
/// taken, e.g. `add x8, sp, #8` (the indirect-result buffer) followed by
/// `ldr w19, [sp, #0xc]` (4 bytes INTO that buffer) — must stay a memory
/// access relative to the cell.
///
/// Flooring the offset to the cell name instead made `sp+8` and `sp+12` the
/// SAME operand, so the value read out of the buffer was provably equal to the
/// buffer's own ADDRESS: the address printed as the value, the callee's store
/// through x8 became invisible, and the read that `ldr w19,[sp,#0xc]` performs
/// was eliminated as a redundant copy.
fn slot_operand(slot: i64, off: i64, scale: i64, width: u8) -> Operand {
    if off == 0 {
        return Operand::reg(format!("stack_{slot}"));
    }
    Operand::Memory {
        base: Box::new(Operand::reg(format!("stack_{slot}"))),
        index: None,
        scale: scale as u8,
        addend: off,
        width,
    }
}

fn rewrite_alias_operand(
    op: &mut Operand,
    aliases: &HashMap<String, (i64, i64)>,
    depth_here: i64,
    ptr_size: u64,
) {
    match op {
        Operand::Memory {
            base,
            addend,
            width,
            ..
        } => {
            if let Operand::Register(b) = base.as_ref() {
                if let Some(&(bias, copy_depth)) = aliases.get(b) {
                    let scale = ptr_size as i64;
                    // `[xD + addend]` where `xD == sp + bias` at depth
                    // `copy_depth`, and sp is at `depth_here` now: the access
                    // sits `bias + addend` bytes above the copy's sp, and the
                    // frame has moved `depth_here - copy_depth` bytes since.
                    let num = bias + *addend - (depth_here - copy_depth);
                    *op = slot_operand(num / scale, num.rem_euclid(scale), scale, *width);
                }
            }
        }
        Operand::AddressOf(inner) => rewrite_alias_operand(inner, aliases, depth_here, ptr_size),
        _ => {}
    }
}

/// Remaining `[sp + N]` operands become absolute slots `stack_(N/8 - depth)`;
/// ShiftStack ops are removed (their effect is baked into depths already).
fn rewrite_sp_operands(insns: &mut [Insn], insn_depth: &HashMap<usize, i64>, ptr_size: u64) {
    let scale = ptr_size as i64;
    for (i, insn) in insns.iter_mut().enumerate() {
        if matches!(insn.op, Op::ShiftStack(_)) {
            insn.op = Op::Move;
            insn.operands.clear();
            continue;
        }
        let here = insn_depth.get(&i).copied().unwrap_or(0);
        for op in &mut insn.operands {
            match op {
                Operand::Memory {
                    base,
                    addend,
                    index: None,
                    width,
                    ..
                } => {
                    if let Operand::Register(b) = base.as_ref() {
                        if b == "sp" {
                            let num = (*addend) - here * scale;
                            *op = slot_operand(num / scale, num.rem_euclid(scale), scale, *width);
                        }
                    }
                }
                Operand::AddressOf(inner) => {
                    if let Operand::Memory {
                        base,
                        addend,
                        index: None,
                        width,
                        ..
                    } = inner.as_mut()
                    {
                        if let Operand::Register(b) = base.as_ref() {
                            if b == "sp" {
                                let num = (*addend) - here * scale;
                                **inner =
                                    slot_operand(num / scale, num.rem_euclid(scale), scale, *width);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Keep the last lane of an HFA (for example `Vector3`) in the same addressable
/// stack object as its first lanes.  AAPCS64 lowers an HFA argument to scalar
/// SIMD registers, and compilers commonly spill it as `stp s0,s1,[sp,#N]` plus
/// `str s2,[sp,#N+8]`.  The first instruction becomes `stack_K` plus a four-byte
/// interior store, while the second used to become a *different* `stack_K+1`
/// local.  If `&stack_K` is then passed to a helper, that local is neither
/// initialized in the emitted C# nor guaranteed adjacent to `stack_K`.
///
/// Only fold a neighboring cell when the IR proves the exact consecutive HFA
/// lane pattern: the base cell receives lane n, the interior four-byte store
/// receives lane n+1, and the next cell receives lane n+2.  This avoids
/// treating arbitrary adjacent stack locals as one aggregate.  The final lane
/// is represented as a four-byte memory access from the address-taken base.
fn coalesce_hfa_stack_lanes(insns: &mut [Insn], ptr_size: u64) {
    use std::collections::HashSet;

    // The lane naming used by this pattern is the AArch64 flattened `vsN`
    // namespace; keep the rewrite confined to that ABI and its 8-byte frame
    // slots.
    if ptr_size != 8 {
        return;
    }

    fn stack_slot(op: &Operand) -> Option<i64> {
        let Operand::Register(name) = op else {
            return None;
        };
        name.strip_prefix("stack_")?.parse().ok()
    }
    fn lane(op: &Operand) -> Option<i64> {
        let Operand::Register(name) = op else {
            return None;
        };
        let lane: i64 = name.strip_prefix("vs")?.parse().ok()?;
        (lane % 4 == 0).then_some(lane)
    }

    let addressed: HashSet<i64> = insns
        .iter()
        .flat_map(|insn| &insn.operands)
        .filter_map(|op| match op {
            Operand::AddressOf(inner) => stack_slot(inner),
            _ => None,
        })
        .collect();
    if addressed.is_empty() {
        return;
    }

    let mut folded = HashSet::new();
    for &base_slot in &addressed {
        for i in 0..insns.len().saturating_sub(2) {
            let insn = &insns[i];
            if insn.op != Op::Move
                || stack_slot(insn.operands.first().unwrap_or(&Operand::Unknown)) != Some(base_slot)
            {
                continue;
            }
            let Some(first_lane) = insn.operands.get(1).and_then(lane) else {
                continue;
            };
            let second = &insns[i + 1];
            let third = &insns[i + 2];
            let contiguous_lanes = second.address == insn.address
                && third.address == insn.address.saturating_add(4)
                && second.op == Op::Move
                && matches!(second.operands.first(), Some(Operand::Memory { base, addend: 4, width: 4, .. }) if stack_slot(base) == Some(base_slot))
                && second.operands.get(1).and_then(lane) == Some(first_lane + 4)
                && third.op == Op::Move
                && stack_slot(third.operands.first().unwrap_or(&Operand::Unknown))
                    == Some(base_slot + 1)
                && third.operands.get(1).and_then(lane) == Some(first_lane + 8);
            if !contiguous_lanes {
                continue;
            }
            let next_slot = base_slot + 1;
            if addressed.contains(&next_slot) {
                continue;
            }
            folded.insert((base_slot, next_slot));
        }
    }

    for (base_slot, next_slot) in folded {
        for insn in insns.iter_mut() {
            for operand in &mut insn.operands {
                if stack_slot(operand) == Some(next_slot) {
                    *operand = Operand::Memory {
                        base: Box::new(Operand::reg(format!("stack_{base_slot}"))),
                        index: None,
                        scale: ptr_size as u8,
                        addend: ptr_size as i64,
                        width: 4,
                    };
                }
            }
        }
    }
}

/// Coalesce a zero-initialised, address-taken stack buffer into its base cell.
///
/// IL2CPP returns a value-type struct through a hidden sret pointer:
/// ```text
/// stp xzr,xzr,[sp,#8]     ; zero 16 bytes  → stack_A, stack_A+1
/// str xzr,[sp,#0x18]      ; next 8 bytes   → stack_A+2
/// add x8, sp, #8 ; bl GetEnumerator
/// ldr x8, [sp,#0x18]      ; Current = stack_A+2
/// ```
/// The analyzer models each slot as its own `stack_N`, so one 24-byte
/// enumerator becomes `var enumerator = 0; stack_66 = 0;` and `Current` reads
/// a cell with no declared type. Re-expressing every member slot `k > 0` as a
/// memory access off the base (`&stack_A + 8k`) is memory-identical at the IR
/// level — the addresses are exactly the ones the slots denoted — and leaves
/// ONE address-taken local for the struct, which the field/foreach passes can
/// type and name.
fn coalesce_address_taken_buffers(insns: &mut [Insn], ptr_size: u64) {
    use std::collections::HashSet;
    if ptr_size != 8 {
        return;
    }
    fn slot(op: &Operand) -> Option<i64> {
        match op {
            Operand::Register(n) => n.strip_prefix("stack_")?.parse().ok(),
            _ => None,
        }
    }
    let mut addressed: HashSet<i64> = HashSet::new();
    for insn in insns.iter() {
        for op in &insn.operands {
            if let Operand::AddressOf(inner) = op {
                if let Some(s) = slot(inner) {
                    addressed.insert(s);
                }
            }
        }
    }
    if addressed.is_empty() {
        return;
    }
    let mut bases: Vec<i64> = addressed.into_iter().collect();
    bases.sort_unstable();
    for base in bases {
        let is_object = |m: i64| -> bool {
            insns.iter().any(|i| {
                i.operands.iter().any(|op| match op {
                    Operand::AddressOf(inner) => slot(inner) == Some(m),
                    Operand::Memory { base: b, .. } => slot(b) == Some(m),
                    _ => false,
                })
            })
        };
        // The zero-init run: base, base+1, … at contiguous instruction
        // addresses (STP covers two slots at one address, then STR +4/+8).
        let mut members: Vec<i64> = Vec::new();
        let mut next = base;
        let mut last_addr: Option<u64> = None;
        for insn in insns.iter() {
            if insn.op != Op::Move {
                continue;
            }
            let Some(Operand::Register(n)) = insn.operands.first() else {
                continue;
            };
            let Some(k) = n.strip_prefix("stack_").and_then(|s| s.parse::<i64>().ok()) else {
                continue;
            };
            if k != next || !matches!(insn.operands.get(1), Some(Operand::Immediate(0))) {
                continue;
            }
            if let Some(la) = last_addr {
                if insn.address < la || insn.address > la + 8 {
                    continue;
                }
            }
            members.push(k);
            next += 1;
            last_addr = Some(insn.address);
            if members.len() >= 16 {
                break;
            }
        }
        if members.len() < 2 {
            continue;
        }
        if members.iter().skip(1).any(|&m| is_object(m)) {
            continue;
        }
        let mut unsafe_member = false;
        for insn in insns.iter() {
            let dp = Insn::dest_position(&insn.op);
            for (idx, op) in insn.operands.iter().enumerate() {
                if let Some(k) = slot(op) {
                    if members.contains(&k) && Some(idx) == dp && insn.op != Op::Move {
                        unsafe_member = true;
                    }
                }
            }
        }
        if unsafe_member {
            continue;
        }
        let base_reg = Operand::reg(format!("stack_{base}"));
        for insn in insns.iter_mut() {
            let dp = Insn::dest_position(&insn.op);
            for (idx, op) in insn.operands.iter_mut().enumerate() {
                let Some(k) = slot(op) else { continue };
                let off = k - base;
                if off <= 0 || !members.contains(&k) {
                    continue;
                }
                *op = Operand::Memory {
                    base: Box::new(base_reg.clone()),
                    index: None,
                    scale: 8,
                    addend: off * 8,
                    // A pure-register destination is a store of the whole
                    // slot; every other occurrence is a value read.
                    width: if Some(idx) == dp { 8 } else { 0 },
                };
            }
        }
    }
}

#[cfg(test)]
mod coalesce_tests {
    use super::*;

    #[test]
    fn zero_init_address_taken_buffer_coalesces_to_base_fields() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![Operand::reg("stack_5"), Operand::imm(0)], 0x1000),
            Insn::new(Op::Move, vec![Operand::reg("stack_6"), Operand::imm(0)], 0x1000),
            Insn::new(Op::Move, vec![Operand::reg("stack_7"), Operand::imm(0)], 0x1004),
            Insn::new(
                Op::Move,
                vec![
                    Operand::reg("x0"),
                    Operand::AddressOf(Box::new(Operand::reg("stack_5"))),
                ],
                0x1008,
            ),
            Insn::new(Op::Move, vec![Operand::reg("x8"), Operand::reg("stack_7")], 0x100c),
        ];
        coalesce_address_taken_buffers(&mut insns, 8);
        assert!(
            !insns.iter().any(|i| i.operands.iter().any(|o| {
                matches!(o, Operand::Register(n) if n == "stack_6" || n == "stack_7")
            })),
            "member slots must become base-relative memory: {insns:?}"
        );
        match &insns[4].operands[1] {
            Operand::Memory { addend, .. } => assert_eq!(*addend, 16),
            other => panic!("Current read must be [base+16], got {other:?}"),
        }
        // The base stays a register (the one address-taken local).
        assert!(matches!(&insns[0].operands[0], Operand::Register(n) if n == "stack_5"));
    }

    #[test]
    fn an_addressed_member_is_never_folded() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![Operand::reg("stack_5"), Operand::imm(0)], 0x1000),
            Insn::new(Op::Move, vec![Operand::reg("stack_6"), Operand::imm(0)], 0x1000),
            Insn::new(
                Op::Move,
                vec![
                    Operand::reg("x0"),
                    Operand::AddressOf(Box::new(Operand::reg("stack_6"))),
                ],
                0x1004,
            ),
        ];
        coalesce_address_taken_buffers(&mut insns, 8);
        assert!(insns.iter().any(|i| i.operands.iter().any(|o| {
            matches!(o, Operand::Register(n) if n == "stack_6")
        })));
    }
}
