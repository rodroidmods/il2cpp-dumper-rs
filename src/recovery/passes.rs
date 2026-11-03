//! Optimization/cleanup passes over the SSA listing.
//!
//! Ports of the corresponding Cpp2IL `Analysis/*` files; each documents its
//! origin. Order of application mirrors MethodAnalysisContext.Analyze().

use super::cfg::Cfg;
use super::ir::{ArithKind, CompareKind, Insn, Op, Operand};

/// The value a register SLOT holds at one point of the linear walks in
/// [`bind_fp_returns`] / [`bind_fp_call_returns`], and the evidence needed
/// before the walk may substitute it for a read.
///
/// `def` is the instruction that wrote it; `None` is the ENTRY version — the
/// incoming argument or the pre-body contents, live on entry and therefore
/// reaching every read.
///
/// `from_call` records that the slot's current value was produced by a
/// float/double-returning CALL, which is the only reason
/// [`bind_fp_call_returns`] exists: the lifter's `BL` arm defines the INTEGER
/// return slot and nothing else, so after a float call the FP slot's SSA
/// version is still the pre-call one and every read of it is stale. A slot last
/// written by ordinary arithmetic is NOT stale — SSA renamed it — and treating
/// it as stale is what cost `OutBounce`@0x14D6330 its parameter (see
/// [`slot_write_reaches`]).
#[derive(Clone)]
struct SlotValue {
    /// The value a read should see. A scalar return is `Local(call_dest)`;
    /// an HFA component is `call_dest.<field>` — a DISTINCT value per
    /// register. Binding every component to the same local destroyed the
    /// distinction (`Vector3` distance code became `(z - z)`).
    value: Operand,
    def: Option<usize>,
    from_call: bool,
}

/// Instruction index → CFG block (`usize::MAX` when unplaced), plus the
/// dominator tree over the same blocks.
///
/// Built from the listing as it stands, which is enough for the two FP passes:
/// both only rewrite OPERANDS, never control flow, so the blocks and the
/// dominator tree stay valid for the whole walk.
fn blocks_and_dominators(insns: &[Insn]) -> (Vec<usize>, super::dominator::DominatorInfo) {
    // `DominatorInfo::compute_from` writes `processed[entry]` unconditionally,
    // so an empty listing would index into empty vectors. A body that lifted to
    // nothing has no reads to rebind either.
    if insns.is_empty() {
        return (
            Vec::new(),
            super::dominator::DominatorInfo {
                idom: Vec::new(),
                rpo: Vec::new(),
                frontiers: Vec::new(),
                dom_children: Vec::new(),
            },
        );
    }
    let cfg = super::ssa::rebuild_cfg(insns);
    let mut block_of = vec![usize::MAX; insns.len()];
    for (bi, b) in cfg.blocks.iter().enumerate() {
        for i in b.start..b.end.min(block_of.len()) {
            block_of[i] = bi;
        }
    }
    let dom = super::dominator::DominatorInfo::compute(&cfg);
    (block_of, dom)
}

/// May the slot write at `def` PROVABLY reach the read at `at`?
///
/// The walks below hold "the last write of this slot" as a straight-line
/// approximation. On a method whose arms each end in `ret` that is a write from
/// a block that cannot reach the read at all, and substituting it invents a
/// value: `OutBounce`@0x14D6330's arms reuse one FP register, so insn 16 read
/// insn 14's sum as its base (`v9 = v13 - A` where the native reads `%xmm2`,
/// i.e. the parameter `t`) — a wrong value, and the offset-based semantic diff
/// cannot see it.
///
/// A write reaches a read when it sits in the same block at a lower index, or
/// when its block dominates the read's. The ENTRY version reaches everything: it
/// is live on entry, which is exactly why `return x;` needs it. An instruction
/// with no block refuses — without CFG evidence there is nothing to prove.
fn slot_write_reaches(
    block_of: &[usize],
    dom: &super::dominator::DominatorInfo,
    def: Option<usize>,
    at: usize,
) -> bool {
    let Some(di) = def else {
        return true;
    };
    let (Some(&db), Some(&ub)) = (block_of.get(di), block_of.get(at)) else {
        return false;
    };
    if db == usize::MAX || ub == usize::MAX {
        return false;
    }
    if db == ub {
        return di < at;
    }
    dom.dominates(db, ub)
}

/// The lifter models every `RET` as returning the value in x0 (the integer
/// return register). FP-returning methods put their result in v0 (s0/d0 on
/// arm64, xmm0 — "vf0" — on x86-64), so the Return reads a stale x0 and
/// DCE then deletes the whole float computation as dead, leaving garbage
/// like `return v3;` in pure-math methods (`FallingFromRestY(t, a)` =
/// `0.5f * a * t * t` compiled to four FMULs into s0 + RET). Rewrite each
/// Return to read the FP result register's live SSA version when the method
/// returns a scalar float/double.
///
/// Version numbers are SSA ids (0-first per slot), so the entry version of
/// the FP register is the minimum id mapping to it — exactly what a method
/// returning its parameter untouched (`return x;`) needs, since that has no
/// def at all. Later defs supersede it while walking toward each Return.
pub fn bind_fp_returns(
    insns: &mut [Insn],
    version_slot: &HashMap<u32, String>,
    ret: &super::types::ManagedType,
) {
    use super::types::{ManagedType, PrimKind};
    let is_r4 = matches!(ret, ManagedType::Primitive(PrimKind::R4));
    let is_r8 = matches!(ret, ManagedType::Primitive(PrimKind::R8));
    if !is_r4 && !is_r8 {
        return;
    }
    let arm = version_slot
        .values()
        .any(|s| s.starts_with("vs") || s.starts_with("vd"));
    let target: &str = match (is_r8, arm) {
        (true, true) => "vd0",
        (false, true) => "vs0",
        _ => "vf0", // x86-64 SysV: XMM0
    };
    // entry version of the FP result register
    let entry = version_slot
        .iter()
        .filter(|(_, s)| s.as_str() == target)
        .map(|(v, _)| *v)
        .min();
    let Some(entry) = entry else {
        return; // register never touched by the body — nothing to bind
    };
    let (block_of, dom) = blocks_and_dominators(insns);
    let mut cur: HashMap<String, SlotValue> = HashMap::new();
    cur.insert(
        target.to_string(),
        SlotValue {
            value: Operand::Local(entry),
            def: None,
            from_call: false,
        },
    );
    for (idx, insn) in insns.iter_mut().enumerate() {
        if matches!(insn.op, Op::Return) {
            if let Some(Operand::Local(d)) = insn.operands.first() {
                if let Some(sv) = cur.get(target) {
                    // Only a value that PROVABLY reaches this Return may replace
                    // what it reads; see [`slot_write_reaches`].
                    if sv.value != Operand::Local(*d)
                        && slot_write_reaches(&block_of, &dom, sv.def, idx)
                    {
                        insn.operands[0] = sv.value.clone();
                    }
                }
            }
        }
        if let Some(Operand::Local(dst)) = insn.dest() {
            if let Some(slot) = version_slot.get(dst) {
                cur.insert(
                    slot.clone(),
                    SlotValue {
                        value: Operand::Local(*dst),
                        def: Some(idx),
                        from_call: false,
                    },
                );
            }
        }
    }
}

/// Bind the FP result register of a CALL whose callee returns a float/double.
///
/// The lifter's `BL` arm defines only the INTEGER return slot (`x0`/`rv`) on
/// every call, but a float/double-returning callee leaves its result in v0 —
/// `s0`/`d0` on arm64, `xmm0` ("vf0") on x86-64. Nothing therefore ever
/// defined the register the caller actually reads, so every read of it saw the
/// stale pre-call value instead of the result.
///
/// Measured on `0x424f4` (arm64 iOS): `str s0, [x19, #0xc8]` after
/// `Random.Range(min, max)` rendered as `_rotationRange = _minRotationRange`
/// — the call's first ARGUMENT, because `vs0` still held the `ldr` that fed
/// the call. That is a wrong VALUE, and the offset-based semantic diff cannot
/// see it (the memory offset matches); it is also the top arm64
/// undefined-local cluster after the vector-load fix (`vs0` 147 / `vd0` 58).
///
/// Same walk as [`bind_fp_returns`], one register later: keep the FP result
/// slot's live value, rebind every READ of the slot to it, and set it from a
/// float-returning call's result slot. Runs AFTER `resolve::run_with_fields`,
/// which is what fills a call's `dest_type` in from the callee's `sig.ret`.
///
/// ## Why a CALL result is the only value that may be substituted
///
/// "Rebind every read of the slot to its live value" is only sound when the
/// read really is stale, and the slot's value is only stale after a call: the
/// lifter's `BL` arm defines `x0`/`rv` and nothing else, so the FP slot has NO
/// version carrying the call's result and every read of it still names the
/// pre-call value. Everywhere else SSA has already renamed the read, so its
/// operand is the right version and replacing it can only lose information.
///
/// That is not theoretical. `OutBounce`@0x14D6330 (x64) reuses ONE FP register
/// for the parameter `t` and for every arm's running result, and each arm ends
/// in `ret`, so the straight-line walk held the PREVIOUS arm's sum as "the live
/// value" and rebound the next arm's parameter read to it:
///
/// ```text
///   native (every arm):  (t - A) * 7.5625 * (t - A) + C
///   emitted:             v9 = v13 - A;  v13 = v9 * 7.5625;  v18 = v13 * v9 + C
/// ```
///
/// `v13` there is the fall-through arm's sum, not `t` — a wrong value the
/// offset-based semantic diff cannot see. The same walk also rewrote the last
/// arm's `t` multiplicand to that arm's own partial product
/// (`v25 = v24 * v24` where the native multiplies by `t`). Both are fixed by
/// requiring `from_call` AND [`slot_write_reaches`]; the arm64 repair this pass
/// exists for — a store of `s0` immediately after a float-returning
/// `Random.Range`, @0x424f4 — is untouched, because there the slot's live value
/// genuinely IS the call's result.
///
/// A `Return` DOES need the entry version substituted into it (`return x;`),
/// which is why [`bind_fp_returns`] must not copy this restriction.
pub fn bind_fp_call_returns(
    insns: &mut [Insn],
    version_slot: &HashMap<u32, String>,
    hfa_stride: usize,
    hfa_fields: impl Fn(&super::types::ManagedType) -> Vec<super::types::FieldRef>,
) -> usize {
    use super::types::{ManagedType, PrimKind};
    let arm = version_slot
        .values()
        .any(|s| s.starts_with("vs") || s.starts_with("vd"));
    let s_slot: &str = if arm { "vs0" } else { "vf0" };
    let d_slot: &str = if arm { "vd0" } else { "vf0" };
    // Cheap gate before the CFG/dominator build: `cur` is keyed by the slot of
    // the operands and destinations the walk visits, so a method with no local
    // in an FP result slot can never enter the rebind path at all — the walk
    // would be a no-op, and half the dump would pay for a graph it cannot use.
    let in_fp_slot = |s: &String| {
        s.as_str() == s_slot || s.as_str() == d_slot || s.starts_with("vs") || s.starts_with("vd")
    };
    if !version_slot.values().any(in_fp_slot) {
        return 0;
    }
    let (block_of, dom) = blocks_and_dominators(insns);
    let mut cur: HashMap<String, SlotValue> = HashMap::new();
    let mut changed = 0usize;
    for (idx, insn) in insns.iter_mut().enumerate() {
        // Reads first: an instruction's sources see the value live BEFORE it,
        // so `fadd s0, s0, s1` must keep its old s0 and only the def updates.
        let dp = Insn::dest_position(&insn.op);
        // A `Phi`'s operands are PER-PREDECESSOR values, not reads of a
        // linearly-live register. `cur` is a straight-line approximation: it
        // holds the LAST write to the slot in instruction order, which on a
        // diamond is the one from the arm that happens to be laid out later.
        // Rebinding a phi operand through it collapsed
        // `ViewScreenAdapter.UpdateRootView` @0x6341f0's `phi(vs0: v51, v52)`
        // into `phi(v52, v52)`: the vmov arm's `v51 = 1.0` then looked dead, so
        // `dead_code_elimination` deleted it, and later `ssa::eliminate`
        // re-materialized the edge copy `v14 = v51` from that edge's live-in —
        // a read whose definition was already gone. That is the top ARM32
        // `undef-local origin vs0` cluster. A phi operand is already the right
        // version for its edge; the linear map must never touch it.
        if !matches!(insn.op, Op::Phi) {
            for i in 0..insn.operands.len() {
                if Some(i) == dp {
                    continue;
                }
                if let Operand::Local(u) = &insn.operands[i] {
                    if let Some(slot) = version_slot.get(u) {
                        if let Some(sv) = cur.get(slot) {
                            // All three tests are required; see the doc comment
                            // above — a rebind from a non-call write, or from a
                            // write that cannot reach this read, is a wrong
                            // value.
                            if sv.from_call
                                && sv.value != Operand::Local(*u)
                                && slot_write_reaches(&block_of, &dom, sv.def, idx)
                            {
                                insn.operands[i] = sv.value.clone();
                                changed += 1;
                            }
                        }
                    }
                }
            }
        }
        // A write to the slot becomes its new live value. It is NOT a call
        // result, so it can never drive a rebind: a read whose slot was last
        // written by ordinary arithmetic is already the version SSA chose.
        if let Some(Operand::Local(d)) = insn.dest() {
            if let Some(slot) = version_slot.get(d) {
                // Retire the binding for ANY slot the walk is currently
                // holding, not just the scalar result slot. `cur` is seeded
                // with one entry per HFA component (`vs0`, `vs4`, `vs8`, ...)
                // at the call, so testing only `s_slot`/`d_slot` left
                // `cur["vs4"]` and `cur["vs8"]` pointing at the call forever:
                // every later read of ANY version of those slots was rewritten
                // to `call.y`/`call.z`, including versions defined by fsub/fmul.
                // On 0x1530F2C that produced `Sqrt(t2.y + (t.x-t2.x)*(t.x-t2.x)
                // + t2.y)` where x is right precisely because vs0 was tracked.
                if slot == s_slot || slot == d_slot || cur.contains_key(slot) {
                    cur.insert(
                        slot.clone(),
                        SlotValue {
                            value: Operand::Local(*d),
                            def: Some(idx),
                            from_call: false,
                        },
                    );
                }
            }
        }
        // A float/double-returning call leaves its result in the FP slot. An
        // HFA (`Vector3`/`Quaternion`/`Color`) occupies one s-register per
        // 32-bit field — s0,s1,s2 for a Vector3 — named `vs0,vs1,…`, the same
        // scalar names `fp_pair_of` gives the spill pair that carries them.
        if matches!(insn.op, Op::Call) {
            // (slot, value the read must see)
            let mut binds: Vec<(String, Operand)> = Vec::new();
            match insn.dest_type.clone() {
                Some(ManagedType::Primitive(PrimKind::R4)) => {
                    binds.push((s_slot.to_string(), Operand::Local(0)));
                }
                Some(ManagedType::Primitive(PrimKind::R8)) => {
                    binds.push((d_slot.to_string(), Operand::Local(0)));
                }
                Some(ty @ ManagedType::Class(_)) => {
                    let fields = hfa_fields(&ty);
                    for (k, field) in fields.into_iter().enumerate() {
                        // ARM64: component k is v<k>.s0 → slot `vs{4k}`.
                        // ARM32: components are s0,s1,… → `vs{k}`.
                        binds.push((format!("vs{}", k * hfa_stride), Operand::Local(0)));
                        if let Some((slot, value)) = binds.last_mut() {
                            *value = Operand::InstanceField {
                                base: Box::new(Operand::Local(0)),
                                field,
                            };
                            let _ = slot;
                        }
                    }
                }
                _ => {}
            }
            if let Some(Operand::Local(d)) = insn.operands.get(1).cloned() {
                for (slot, mut value) in binds {
                    // Every component of the returned struct is a FIELD of the
                    // call's result local — distinct per register, and typed
                    // from the field metadata.
                    if let Operand::InstanceField { base, .. } = &mut value {
                        **base = Operand::Local(d);
                    } else if let Operand::Local(v) = &mut value {
                        *v = d;
                    }
                    cur.insert(
                        slot,
                        SlotValue {
                            value,
                            def: Some(idx),
                            from_call: true,
                        },
                    );
                }
            }
        }
    }
    changed
}
use std::collections::{HashMap, HashSet};

/// DeadCodeEliminator: fixpoint nop-ing pure defs with zero uses
/// (calls/stores/branches never removed). Cpp2IL `DeadCodeEliminator.cs`.
pub fn dead_code_elimination(insns: &mut [Insn]) -> usize {
    let mut removed = 0;
    // removal cascades (A feeds B feeds C…) need a handful of rounds; a hard
    // round cap keeps the fixpoint O(rounds·N) instead of O(N²) worst case
    const MAX_DCE_ROUNDS: usize = 32;
    let mut rounds = 0usize;
    loop {
        rounds += 1;
        if rounds > MAX_DCE_ROUNDS {
            return removed;
        }
        #[allow(unused_variables)]
        let mut use_count: HashMap<u32, usize> = HashMap::new();
        for insn in insns.iter() {
            let mut uses = Vec::new();
            for op in &insn.operands {
                collect_locals(op, &mut uses);
            }
            for u in uses {
                *use_count.entry(u).or_default() += 1;
            }
        }
        let mut changed = false;
        for insn in insns.iter_mut() {
            if !insn.is_removable_if_dead() {
                continue;
            }
            if let Some(Operand::Local(d)) = insn.dest().cloned() {
                // self-referential defs still count their own use — subtract it
                let external_uses = use_count.get(&d).copied().unwrap_or(0)
                    - usize::from(insn_operands_reference(insn, d));
                if external_uses == 0 {
                    super::verify::ledger_note(insn.address, super::verify::LedgerKind::Dce);
                    *insn = Insn::nop(insn.address);
                    changed = true;
                    removed += 1;
                }
            }
        }
        if !changed {
            return removed;
        }
    }
}

fn insn_operands_reference(insn: &Insn, local: u32) -> bool {
    let mut v = Vec::new();
    for op in &insn.operands {
        collect_locals(op, &mut v);
    }
    v.contains(&local)
}

fn collect_locals(op: &Operand, out: &mut Vec<u32>) {
    // EXHAUSTIVE ON PURPOSE. This match is the only thing that tells
    // `dead_code_elimination` which locals an instruction reads, and it is
    // private to this pass — nothing else would notice if a new operand
    // variant started carrying a sub-operand and this walker did not descend
    // into it. That is precisely how `InstanceField` was missed: the base was
    // a real use, the walker did not see it, the def that computed the
    // receiver was Nop'd as dead, and the surviving read printed as
    // `in_xN` (an ABI name for a value the method itself defines). Do not
    // replace this with `_ => {}`: a new variant must be a compile error
    // here so that whoever adds it decides, explicitly, whether it reads.
    match op {
        Operand::Local(l) => out.push(*l),
        Operand::Memory { base, index, .. } => {
            collect_locals(base, out);
            if let Some(i) = index {
                collect_locals(i, out);
            }
        }
        Operand::AddressOf(inner) | Operand::ArrayLength(inner) => collect_locals(inner, out),
        Operand::ArrayAccess { array, index } => {
            collect_locals(array, out);
            collect_locals(index, out);
        }
        // The receiver of a field access is READ by it.
        Operand::InstanceField { base, .. } => collect_locals(base, out),
        // Leaves: no nested operand, so no local can hide inside.
        Operand::Register(_)
        | Operand::Immediate(_)
        | Operand::FloatLiteral(_)
        | Operand::DoubleLiteral(_)
        | Operand::StringLiteral { .. }
        | Operand::Method(_)
        | Operand::Field(_)
        | Operand::Type(_)
        | Operand::MissingAbiArg
        | Operand::Unknown => {}
    }
}

/// Liveness-based dead-code elimination over the FINAL (post-elim) listing.
///
/// The use-count DCE above cannot break self-referential dead chains: a
/// compiler keeps a loop counter's decrement alive only through its own
/// copies (`v55 = v31 - 1; v31 = v55`) — each def has one use (the other
/// half of the chain), so plain use-counts see nothing removable even
/// though NO read of the variable can reach a method exit or side effect.
/// That is exactly the `while (*(p-8) != 0) { v30 += 16; v31 -= 1; }`
/// shape: the `v31` counter is written every iteration, read only by its
/// own next write, never read at any exit — dead. Backward liveness
/// (a local is live at a def only if a later read can reach an exit or
/// side effect) breaks the cycle and both halves vanish.
///
/// Runs on phi-free IR (after `ssa::eliminate`): block CFG from
/// `rebuild_cfg`, standard backward dataflow, removables whose dest is dead
/// at the def point become Nops; cascades are re-run to a fixpoint.
pub fn liveness_dce(insns: &mut [Insn]) -> usize {
    let mut removed_total = 0usize;
    // The CFG is INVARIANT across the fixpoint: only `is_removable_if_dead`
    // instructions are Nop'd, and that predicate excludes every terminator
    // (ir.rs), so leaders/successors never change. Rebuild it once instead of
    // once per outer round.
    let cfg = super::ssa::rebuild_cfg(insns);
    let n = cfg.blocks.len();
    if n == 0 {
        return 0;
    }
    loop {
        let mut live_in: Vec<std::collections::HashSet<u32>> =
            vec![std::collections::HashSet::new(); n];
        let mut live_out: Vec<std::collections::HashSet<u32>> =
            vec![std::collections::HashSet::new(); n];
        let mut changed = true;
        let mut iters = 0usize;
        while changed && iters < 128 {
            changed = false;
            iters += 1;
            // live_out = union of successors' live_in
            for bi in 0..n {
                let mut lo: std::collections::HashSet<u32> = std::collections::HashSet::new();
                for &s in &cfg.successors[bi] {
                    if s < n {
                        lo.extend(live_in[s].iter());
                    }
                }
                if lo != live_out[bi] {
                    live_out[bi] = lo;
                    changed = true;
                }
            }
            for bi in (0..n).rev() {
                let b = &cfg.blocks[bi];
                let mut live = live_out[bi].clone();
                for i in (b.start..b.end).rev() {
                    let insn = &insns[i];
                    let removable = insn.is_removable_if_dead();
                    let d = insn.dest().and_then(|o| o.as_local());
                    match d {
                        Some(dv) => {
                            if removable && !live.contains(&dv) {
                                continue; // dead def: contributes nothing
                            }
                            for u in insn.use_locals() {
                                live.insert(u);
                            }
                            // a kept def kills every earlier writer of dv
                            live.remove(&dv);
                        }
                        None => {
                            for u in insn.use_locals() {
                                live.insert(u);
                            }
                        }
                    }
                }
                if live != live_in[bi] {
                    live_in[bi] = live;
                    changed = true;
                }
            }
        }
        // liveness converged: collect dead defs in one more backward walk
        let mut dead: Vec<usize> = Vec::new();
        for bi in (0..n).rev() {
            let b = &cfg.blocks[bi];
            let mut live = live_out[bi].clone();
            for i in (b.start..b.end).rev() {
                let insn = &insns[i];
                let removable = insn.is_removable_if_dead();
                match insn.dest().and_then(|o| o.as_local()) {
                    Some(dv) if removable && !live.contains(&dv) => {
                        dead.push(i);
                        continue;
                    }
                    Some(dv) => {
                        for u in insn.use_locals() {
                            live.insert(u);
                        }
                        live.remove(&dv);
                    }
                    None => {
                        for u in insn.use_locals() {
                            live.insert(u);
                        }
                    }
                }
            }
        }
        if dead.is_empty() {
            break;
        }
        dead.sort_unstable();
        for i in dead {
            if !matches!(insns[i].op, Op::Nop) {
                super::verify::ledger_note(insns[i].address, super::verify::LedgerKind::Dce);
                insns[i] = Insn::nop(insns[i].address);
                removed_total += 1;
            }
        }
    }
    removed_total
}

/// Copy + constant propagation through single-def locals and phi-free moves
/// (post-SSA-elimination form). Cpp2IL `Simplifier.cs` + `SsaSimplifier.cs`.
///
/// Conservative rules preserved from the original:
/// - never propagate into Memory base/index or FieldReference positions
///   unless replacement is itself a local (holds by construction here: this
///   pass only ever replaces a TOP-LEVEL `Operand::Local`, and it never
///   descends into a `Memory`'s base/index, which live inside one operand)
/// - a local with >1 definition is not propagated (join-safety)
///
/// ## The CHAIN step needs a reachability proof the entry check does not give
///
/// The rules above are enforced on the local AT THE USE (`def_count == 1 &&
/// !loop_carried`). `resolve` then walks FURTHER: a use of `v` where `v = w`
/// resolves to `w`'s `value_of` entry, and that entry is the **last textual
/// definition** of `w` — which may sit in an arm this use cannot be reached
/// from at all. That is the same defect class the FP passes had
/// ([`slot_write_reaches`]): a value taken from a write that never reaches the
/// read. `def_count` and the dominator tree now gate EVERY hop, so a hop is
/// taken only when the definition is unique AND provably reaches the use.
///
/// The guard is inert on well-formed IR. If `v = w` is a valid read of `w` at
/// `d_v` and the use of `v` at `u` is dominated by `d_v`, then `w`'s
/// definition dominates `u` by transitivity of dominance — so the added
/// conditions cannot refuse a substitution the old code made on consistent
/// input. What they change is IR that was ALREADY inconsistent (this pipeline
/// reports undefined locals, so that IR exists): there the pass used to
/// materialize a plausible-looking constant from an unrelated arm, and a
/// downstream reader cannot tell an invented value from a real one. It now
/// fails CLOSED, keeping the bare local — a wrong value is the only outcome
/// worse than a visible undefined `vN`.
pub fn copy_and_constant_propagation(insns: &mut [Insn]) -> usize {
    // Locals whose STORAGE is addressed anywhere are mutable through that
    // address. One textual store does not determine a later read: IL2CPP
    // initialises an `out` cell (`v = 0`), passes `&v` to the callee, and the
    // callee writes the real value. Forwarding the initialiser invented a
    // null token at 0x18A3A1C (`TryGetValue(key, &cell)`), which folded the
    // null check to `true` and made the whole update block — a field store
    // and `UpdateProbability` — unreachable. Exclude these locals from the
    // value map, and refuse to propagate a COPY of one (a snapshot becomes a
    // re-read of the mutated cell).
    let mut addressed: HashSet<u32> = HashSet::new();
    for insn in insns.iter() {
        for op in &insn.operands {
            collect_addressed_locals(op, &mut addressed);
        }
    }
    // def counts per local
    let mut def_count: HashMap<u32, usize> = HashMap::new();
    let mut value_of: HashMap<u32, Operand> = HashMap::new();
    // where each forwardable value was defined (for the back-edge guard below)
    let mut def_index: HashMap<u32, usize> = HashMap::new();
    // last definition of every local, forwardable or not — the transitive
    // dependence check walks through operands like a load as well
    let mut def_of: HashMap<u32, usize> = HashMap::new();
    for (i, insn) in insns.iter().enumerate() {
        if let Some(Operand::Local(d)) = insn.dest() {
            def_of.insert(*d, i);
        }
    }

    for (i, insn) in insns.iter().enumerate() {
        if matches!(insn.op, Op::Move) {
            if let (Some(Operand::Local(d)), Some(src)) =
                (insn.dest().cloned(), insn.operands.get(1))
            {
                *def_count.entry(d).or_default() += 1;
                // forwardable values: locals and constants only
                let forwardable = matches!(
                    src,
                    Operand::Local(_)
                        | Operand::Immediate(_)
                        | Operand::StringLiteral { .. }
                        | Operand::FloatLiteral(_)
                        | Operand::DoubleLiteral(_)
                        | Operand::Method(_)
                        | Operand::Field(_)
                );
                // The source must not depend on `d` ITSELF, even through the
                // forward chain: `v0 = v3` with `v3 = [v0 +0x28]` is a
                // MUTATION of `v0` (`do { v0 = *(v0 +40); } while (…)`), and
                // forwarding `v0` to `v3` lets every later use of `v0`
                // re-evaluate the mutation at a point where `v0` has already
                // been advanced — `Exception.GetBaseException` returned the
                // field of the updated pointer instead of the value captured
                // before the update.
                if forwardable
                    && !addressed.contains(&d)
                    && !matches!(src, Operand::Local(w) if addressed.contains(w))
                    && !reads_local_transitively(src, d, &def_of, insns)
                {
                    value_of.insert(d, src.clone());
                    def_index.insert(d, i);
                }
            } else if let Some(Operand::Local(d)) = insn.dest() {
                *def_count.entry(*d).or_default() += 1;
            }
        } else if let Some(Operand::Local(d)) = insn.dest().cloned() {
            *def_count.entry(d).or_default() += 1;
        }
    }

    // A use that its own (single) definition reaches only around a BACK EDGE —
    // the definition sits LATER in the same basic block — reads the value the
    // local held at the END of the previous iteration, not the definition's
    // value. The C# compiler's rotated loops are exactly this shape:
    //
    //     while (inner != null) inner = inner.InnerException;
    //  →  back = inner; inner = [inner, #0x28]; cbnz inner, -8; return back;
    //
    // Forwarding `inner` there collapsed the captured `back` onto the
    // already-advanced register: `Exception.GetBaseException()`@0x1A6FA74
    // returned the field of the updated pointer (i.e. `null`) instead of the
    // base exception. Read such a use by NAME; never forward a definition
    // across the back edge that feeds it.
    //
    // Nothing forwardable -> nothing to prove, and the CFG + dominator build is
    // the most expensive thing in this pass. `value_of` and `def_index` are
    // filled together, so an empty `value_of` cannot hide a hop.
    let (block_of, dom) = if value_of.is_empty() {
        (Vec::new(), None)
    } else {
        let (b, d) = blocks_and_dominators(insns);
        (b, Some(d))
    };
    let loop_carried = |v: u32, use_i: usize| -> bool {
        def_index
            .get(&v)
            .is_some_and(|&d| d > use_i && block_of.get(d).copied() == block_of.get(use_i).copied())
    };
    // May the definition `value_of[v]` was taken from PROVABLY reach the use at
    // `use_i`? Without this a hop can carry a value out of an arm that cannot
    // reach the read; see the pass doc.
    let hop_reaches_use = |v: u32, use_i: usize| -> bool {
        let Some(dom) = dom.as_ref() else {
            return false; // no CFG evidence -> fail closed
        };
        match def_index.get(&v) {
            // `value_of[v]` without a recorded definition cannot happen (they
            // are inserted together); refuse rather than assume entry.
            None => false,
            Some(&d) => slot_write_reaches(&block_of, dom, Some(d), use_i),
        }
    };

    // collapse chains
    let resolve = |start: Operand,
                   value_of: &HashMap<u32, Operand>,
                   depth_limit: usize,
                   use_i: usize|
     -> Operand {
        let mut cur = start.clone();
        for _ in 0..depth_limit {
            let next = match &cur {
                // EVERY hop needs the same evidence as the entry: a unique
                // definition that reaches this use. `loop_carried` alone only
                // covers the same-block rotated-loop back edge.
                Operand::Local(v)
                    if def_count.get(v).copied().unwrap_or(0) == 1
                        && !loop_carried(*v, use_i)
                        && hop_reaches_use(*v, use_i) =>
                {
                    value_of.get(v).cloned()
                }
                _ => None,
            };
            match next {
                Some(n) => cur = n,
                None => break,
            }
        }
        cur
    };

    let mut replaced = 0usize;
    for (i, insn) in insns.iter_mut().enumerate() {
        if matches!(insn.op, Op::Move | Op::Phi | Op::Nop) {
            continue;
        }
        for pos in 0..insn.operands.len() {
            // Destination positions are never rewritten: a def's destination is
            // not a read of the value it assigns. `Insn::dest_position` is the
            // project's single source of truth for that, so use it instead of a
            // local op list — the list here omitted `Convert`, `Select`, `Clz`,
            // `MulAcc`, `Simd` and `SimdLane`, whose operand 0 is also a
            // destination. Reaching one of those needs `value_of[v]` for a local
            // `v` that THIS instruction defines, i.e. a second definition of
            // `v`, which the `def_count == 1` entry check already refused — so
            // this is a trap removed rather than a behaviour change.
            let is_dest_pos = Insn::dest_position(&insn.op) == Some(pos);
            if is_dest_pos {
                continue;
            }
            let candidate = match &insn.operands[pos] {
                Operand::Local(v)
                    if def_count.get(v).copied().unwrap_or(0) == 1 && !loop_carried(*v, i) =>
                {
                    Operand::Local(*v)
                }
                _ => continue,
            };
            let resolved = resolve(candidate, &value_of, 8, i);
            if resolved != insn.operands[pos] {
                // memory base/index only accept local replacements
                let in_memory_position = false; // operands are flattened IR
                let _ = in_memory_position;
                insn.operands[pos] = resolved;
                replaced += 1;
            }
        }
    }
    replaced
}

/// Collect every local whose storage is addressed by `op`, recursively
/// through compound operands. `&local` and `&*(local)` both make the local a
/// mutable cell: a callee can write it.
fn collect_addressed_locals(op: &Operand, out: &mut HashSet<u32>) {
    // Exhaustive for the same reason as `collect_locals`: a catch-all here
    // would silently stop descending into whatever sub-operand a future
    // variant introduces, and escape analysis would then call a live cell
    // unaddressed.
    match op {
        Operand::AddressOf(inner) => {
            // `&local` and `&*(local)` both name a cell directly; any deeper
            // shape is just a value whose address is taken, and the recursion
            // below still walks it.
            let cell = match inner.as_ref() {
                Operand::Memory { base, .. } => base.as_ref(),
                other => other,
            };
            if let Operand::Local(v) = cell {
                out.insert(*v);
            }
            collect_addressed_locals(inner, out);
        }
        Operand::ArrayLength(inner) => collect_addressed_locals(inner, out),
        Operand::Memory { base, index, .. } => {
            collect_addressed_locals(base, out);
            if let Some(i) = index {
                collect_addressed_locals(i, out);
            }
        }
        Operand::ArrayAccess { array, index } => {
            collect_addressed_locals(array, out);
            collect_addressed_locals(index, out);
        }
        Operand::InstanceField { base, .. } => collect_addressed_locals(base, out),
        Operand::Register(_)
        | Operand::Local(_)
        | Operand::Immediate(_)
        | Operand::FloatLiteral(_)
        | Operand::DoubleLiteral(_)
        | Operand::StringLiteral { .. }
        | Operand::Method(_)
        | Operand::Field(_)
        | Operand::Type(_)
        | Operand::MissingAbiArg
        | Operand::Unknown => {}
    }
}

/// Does the VALUE of `op` depend on `local` — directly, or through the def
/// chain of any local it mentions (`def_of`: local → the instruction that
/// defines it)? Bounded, and conservative (`true`) when the budget runs out:
/// refusing a forward can only cost inlining, never a wrong value.
fn reads_local_transitively(
    op: &Operand,
    local: u32,
    def_of: &HashMap<u32, usize>,
    insns: &[Insn],
) -> bool {
    let mut stack = vec![op.clone()];
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut guard = 0usize;
    while let Some(cur) = stack.pop() {
        guard += 1;
        if guard > 64 {
            return true;
        }
        let mut v = Vec::new();
        collect_locals(&cur, &mut v);
        for w in v {
            if w == local {
                return true;
            }
            if !seen.insert(w) {
                continue;
            }
            if let Some(&d) = def_of.get(&w) {
                let dpos = Insn::dest_position(&insns[d].op);
                for (k, o) in insns[d].operands.iter().enumerate() {
                    if Some(k) == dpos {
                        continue;
                    }
                    stack.push(o.clone());
                }
            }
        }
    }
    false
}

/// Constant folding + algebraic identities. Cpp2IL `ConstantFolder.cs`.
/// Iterated with propagation by the orchestrator (≤8 rounds).
pub fn constant_folding(insns: &mut [Insn]) -> usize {
    let mut folded = 0;
    for i in 0..insns.len() {
        let op = insns[i].op.clone();
        let (a, b) = (
            insns[i].operands.get(1).cloned(),
            insns[i].operands.get(2).cloned(),
        );

        let result: Option<Operand> = match (&op, a, b) {
            (
                Op::Arith(ArithKind::Add),
                Some(Operand::Immediate(x)),
                Some(Operand::Immediate(y)),
            ) => Some(Operand::Immediate(x.wrapping_add(y))),
            (
                Op::Arith(ArithKind::Sub),
                Some(Operand::Immediate(x)),
                Some(Operand::Immediate(y)),
            ) => Some(Operand::Immediate(x.wrapping_sub(y))),
            (
                Op::Arith(ArithKind::Mul),
                Some(Operand::Immediate(x)),
                Some(Operand::Immediate(y)),
            ) => Some(Operand::Immediate(x.wrapping_mul(y))),
            (
                Op::Arith(ArithKind::And),
                Some(Operand::Immediate(x)),
                Some(Operand::Immediate(y)),
            ) => Some(Operand::Immediate(x & y)),
            (
                Op::Arith(ArithKind::Or),
                Some(Operand::Immediate(x)),
                Some(Operand::Immediate(y)),
            ) => Some(Operand::Immediate(x | y)),
            (
                Op::Arith(ArithKind::Xor),
                Some(Operand::Immediate(x)),
                Some(Operand::Immediate(y)),
            ) => Some(Operand::Immediate(x ^ y)),
            (
                Op::Arith(ArithKind::Shl),
                Some(Operand::Immediate(x)),
                Some(Operand::Immediate(y)),
            ) => Some(Operand::Immediate(x.wrapping_shl(y.clamp(0, 63) as u32))),
            (
                Op::Arith(ArithKind::Shr),
                Some(Operand::Immediate(x)),
                Some(Operand::Immediate(y)),
            ) => Some(Operand::Immediate(x.wrapping_shr(y.clamp(0, 63) as u32))),
            (Op::Compare(k), Some(Operand::Immediate(x)), Some(Operand::Immediate(y))) => {
                let b = match k {
                    CompareKind::Equal => x == y,
                    CompareKind::NotEqual => x != y,
                    CompareKind::LessThan => x < y,
                    CompareKind::GreaterThan => x > y,
                    CompareKind::LessOrEqual => x <= y,
                    CompareKind::GreaterOrEqual => x >= y,
                };
                Some(Operand::Immediate(b as i64))
            }
            _ => None,
        };

        if let Some(val) = result {
            let dst = insns[i].dest().cloned();
            if let Some(dst) = dst {
                insns[i] = Insn::move_(dst, val, insns[i].address);
                folded += 1;
            }
        }
    }
    folded
}

/// Cross-block flag liveness for the condition-folding lifters (ARM32 today).
///
/// The lifter models the flags as a single `pending_cmp` carried along the
/// LISTING in address order and cleared at every flag-writing instruction.
/// That is right for a compare and its consumer in one straight-line run —
/// and across an unconditional `b`, which writes no APSR — but it cannot see
/// that a flag writer lying between the compare and a LATER branch is not on
/// that branch's path. The canonical shape (ARM32, `libil2cpp.so`):
///
/// ```text
///     cmp  r1, r2
///     strb r0, [r4, #0x15]
///     bne  T              ; flags live on the taken edge
///     ... bl / ldr ...    ; the fall-through clobbers the flags
///     T: mov r0, r4
///        ble  U           ; re-tests the SAME APSR
/// ```
///
/// `T` has ONE predecessor (the `bne`), and no flag writer lies on that edge,
/// so the flags ARE live at `ble`; the linear model nonetheless materialized
/// `Compare(kind, Unknown, 0)` there (its `pending_cmp` was killed by the
/// fall-through's `bl`). This pass runs a forward dataflow over the CFG that
/// carries the reaching comparison across edges where it survives, and
/// rewrites the `Unknown` operands of the feeding `Compare` so the branch
/// renders with its real condition instead of `if (? != 0)`.
///
/// SOUNDNESS. The carried state is the operand pair of a REAL (non-Unknown)
/// `Compare` that a branch or a `Select` consumes — a materialized APSR
/// comparison, never a synthetic one like the carry `materialize_carry`
/// builds. It is dropped at any instruction that can write the flags (a call,
/// an arithmetic/multiply op) and at any redefinition of one of its operand
/// REGISTERS (which would make the operand read the wrong value at the
/// consumer). At a join every predecessor must agree on the same operand pair,
/// else the state is dropped. Operands are left as registers, so SSA renames
/// them at the consumer exactly as it renames the compare at its own consumer;
/// no copy is inserted and no block is split, so the CFG the caller passes in
/// stays valid and instruction indices do not move.
///
/// Returns the number of branch conditions rewritten.
pub fn resolve_cross_block_conditions(insns: &mut [Insn], graph: &Cfg) -> usize {
    use std::collections::{HashMap, HashSet, VecDeque};

    // A condition scratch is "consumed" when a branch or a select reads it;
    // only such a `Compare` materializes the real APSR, so only those arm the
    // carried state (the carry `materialize_carry` builds feeds an `Arith`).
    let mut consumed: HashSet<u32> = HashSet::new();
    for ins in insns.iter() {
        match &ins.op {
            Op::CondJump(_) => {
                if let Some(Operand::Local(v)) = ins.operands.first() {
                    consumed.insert(*v);
                }
            }
            Op::Select(_) => {
                if let Some(Operand::Local(v)) = ins.operands.get(1) {
                    consumed.insert(*v);
                }
            }
            _ => {}
        }
    }
    if consumed.is_empty() {
        return 0;
    }

    let nblocks = graph.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
    for (pb, succ) in graph.successors.iter().enumerate() {
        for &sb in succ {
            if sb < nblocks {
                preds[sb].push(pb);
            }
        }
    }

    type Flags = Option<(Operand, Operand)>;
    let mut out_state: Vec<Flags> = vec![None; nblocks];
    let mut rewrites: HashMap<usize, (Operand, Operand)> = HashMap::new();

    // Transfer one block, returning the state at its end and recording every
    // consumer `Compare` whose Unknown operands the reaching state can fill.
    let transfer =
        |bi: usize, mut state: Flags, rewrites: &mut HashMap<usize, (Operand, Operand)>| -> Flags {
            let b = graph.blocks[bi];
            for i in b.start..b.end {
                let ins = &insns[i];
                match &ins.op {
                    // A call, a multiply or a `clz` can write the flags and is not
                    // a tracked def, so the carried value stops here.
                    //
                    // `Op::Arith` is deliberately NOT here even though an s-form
                    // data-processing op writes APSR: the IR has no s bit, so it
                    // also covers `mvn` (`Arith(Xor) r, #0, #-1`) and every plain
                    // `add`/`and`, which PRESERVE the flags. Killing on all of them
                    // discarded exactly the live compares this pass exists to find
                    // (`mvn r5, #0; ble L` is the 0x64c424 site). The s-forms are
                    // not lost by the omission: an s-form that sets the flags the
                    // target reads materializes a REAL `Compare` there (the linear
                    // `pending_cmp` is not killed between the two), so the target
                    // is not `Unknown` and this pass never touches it.
                    Op::MulAcc { .. }
                    | Op::Clz
                    | Op::Call
                    | Op::CallVoid
                    | Op::IndirectCall
                    | Op::Return
                    | Op::Throw
                    | Op::NewObj(_)
                    | Op::NewArr(_)
                    | Op::Box(_)
                    | Op::ObjcMsgSend { .. } => state = None,
                    Op::Compare(_) => {
                        if let Some(Operand::Local(v)) = ins.operands.first() {
                            if consumed.contains(v) {
                                let lhs = ins.operands.get(1);
                                let rhs = ins.operands.get(2);
                                let unknown = matches!(lhs, Some(Operand::Unknown))
                                    || matches!(lhs, Some(Operand::MissingAbiArg));
                                if !unknown {
                                    if let (Some(l), Some(r)) = (lhs, rhs) {
                                        state = Some((l.clone(), r.clone()));
                                    }
                                } else if let Some((l, r)) = state.clone() {
                                    rewrites.entry(i).or_insert((l, r));
                                }
                            }
                        }
                    }
                    _ => {}
                }
                // A redefinition of an operand register makes the carried pair
                // stale at every later use, so it cannot be forwarded.
                if let Some(Operand::Register(name)) = ins.dest() {
                    if let Some((l, r)) = &state {
                        let hit = matches!(l, Operand::Register(n) if n == name)
                            || matches!(r, Operand::Register(n) if n == name);
                        if hit {
                            state = None;
                        }
                    }
                }
            }
            state
        };

    let mut queued = vec![true; nblocks];
    let mut wl: VecDeque<usize> = (0..nblocks).collect();
    while let Some(bi) = wl.pop_front() {
        queued[bi] = false;
        // Meet over predecessors: every edge in must agree on the operand pair.
        let mut incoming: Flags = None;
        let mut first = true;
        for &pb in &preds[bi] {
            let s = out_state[pb].clone();
            if first {
                incoming = s;
                first = false;
            } else if incoming != s {
                incoming = None;
                break;
            }
        }
        let out = transfer(bi, incoming, &mut rewrites);
        if out != out_state[bi] {
            out_state[bi] = out;
            for &sb in &graph.successors[bi] {
                if sb < nblocks && !queued[sb] {
                    queued[sb] = true;
                    wl.push_back(sb);
                }
            }
        }
    }

    let mut n = 0;
    for (i, (l, r)) in rewrites {
        if let Some(ins) = insns.get_mut(i) {
            // The transfer function above treats `MissingAbiArg` as a hole
            // exactly like `Unknown` (both mean "no operand recovered"), but
            // this apply step accepted only `Unknown` — so a compare whose
            // operand slot was an ABI hole had its recovery computed and then
            // silently discarded. Accept both, the way the arming rule does.
            if matches!(ins.op, Op::Compare(_))
                && matches!(
                    ins.operands.get(1),
                    Some(Operand::Unknown) | Some(Operand::MissingAbiArg)
                )
            {
                ins.operands[1] = l;
                ins.operands[2] = r;
                n += 1;
            }
        }
    }
    n
}

/// BooleanFlagSimplifier + EqualityBranchInverter equivalents:
/// `(x == 0) == 0` chains collapse to direct comparisons.
///
/// AUDITED (linear-definition-search class): the backwards scan below finds
/// the definition of `inner_v`, but it keys on the SSA VERSION (`dest()` ==
/// `inner_v`), not on a register slot, so there is exactly one instruction it
/// can find — and if that definition is not the one reaching this compare the
/// read was already broken before this pass ran. It reads the found
/// instruction (kind + dest_type), it never carries a value out of it, so it
/// cannot invent one. No reachability guard is needed.
pub fn simplify_boolean_flags(insns: &mut [Insn]) -> usize {
    let mut simplified = 0;
    for i in 0..insns.len() {
        let Some(Op::Compare(kind)) = Some(insns[i].op.clone()) else {
            continue;
        };
        let lhs = insns[i].operands.get(1).cloned();
        let rhs = insns[i].operands.get(2).cloned();
        // pattern: cmp.eq(cmpInner_result, 0) where inner was an equality → ne
        if kind == CompareKind::Equal {
            if let Some(Operand::Immediate(0)) = rhs {
                if let Some(Operand::Local(inner_v)) = lhs {
                    for j in (0..i).rev().take(16) {
                        if let Some(Operand::Local(dv)) = insns[j].dest() {
                            if *dv == inner_v {
                                if let Op::Compare(
                                    inner_kind @ (CompareKind::Equal | CompareKind::NotEqual),
                                ) = insns[j].op
                                {
                                    insns[j].op = Op::Compare(match inner_kind {
                                        CompareKind::Equal => CompareKind::NotEqual,
                                        _ => CompareKind::Equal,
                                    });
                                    insns[j].dest_type = insns[i].dest_type.clone();
                                    super::verify::ledger_note(
                                        insns[i].address,
                                        super::verify::LedgerKind::FoldedBranch,
                                    );
                                    insns[i] = Insn::nop(insns[i].address);
                                    simplified += 1;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    simplified
}

/// InjectedCheckRemover: null-check guards (`cmp obj, #0; b.ne ok;
/// <throw block>`) whose throw target only raises are folded away.
/// Structural subset — full version needs throw-block classification.
pub fn remove_null_check_guards(insns: &mut [Insn], cfg: &Cfg) -> usize {
    let mut removed = 0;
    for bi in 0..cfg.blocks.len() {
        let b = cfg.blocks[bi];
        let last = b.end - 1;
        let Some(Op::CondJump(taken)) = Some(insns[last].op.clone()) else {
            continue;
        };
        // condition must be "x != 0" style null check
        let Some(cmp_idx) = last.checked_sub(1) else {
            continue;
        };
        if !matches!(insns[cmp_idx].op, Op::Compare(CompareKind::NotEqual)) {
            continue;
        }
        // taken target must be a THROW-ONLY block: either a bare `Throw` or
        // the compiler's NullReferenceException stub
        // (`NewObj(NullRef); call raise; Throw` + feeder moves). Previously
        // only ≤4-insn bare-throw blocks matched, so the stub's
        // `throw X.GetSystem<object>()` garbage survived into every
        // null-guarded method.
        let is_throw_block = cfg
            .blocks
            .get(taken)
            .map(|tb| {
                let (s, e) = (tb.start, tb.end);
                if e <= s || insns[e - 1].op != Op::Throw || e - s > 8 {
                    return false;
                }
                let mut calls = 0usize;
                insns[s..e].iter().all(|i| match i.op {
                    Op::Throw => true,
                    Op::Call | Op::CallVoid | Op::IndirectCall | Op::NewObj(_) => {
                        calls += 1;
                        calls <= 1
                    }
                    Op::Move | Op::Nop | Op::Arith(_) | Op::Phi => true,
                    _ => false, // any control flow / real logic → not a stub
                })
            })
            .unwrap_or(false);
        if is_throw_block {
            super::verify::ledger_note(
                insns[last].address,
                super::verify::LedgerKind::GuardExcision,
            );
            super::verify::ledger_note(
                insns[cmp_idx].address,
                super::verify::LedgerKind::GuardExcision,
            );
            insns[last] = Insn::nop(insns[last].address);
            insns[cmp_idx] = Insn::nop(insns[cmp_idx].address);
            // Erase the now-unreachable stub block itself — leaving it in
            // the listing keeps rendering `throw <garbage>` text even
            // though it can never execute.
            if let Some(tb) = cfg.blocks.get(taken) {
                for i in insns[tb.start..tb.end].iter() {
                    super::verify::ledger_note(i.address, super::verify::LedgerKind::GuardExcision);
                }
                for i in insns[tb.start..tb.end].iter_mut() {
                    *i = Insn::nop(i.address);
                }
            }
            removed += 1;
        }
    }
    removed
}

/// Resolve a control-flow index forward over things that DO NOTHING — `Nop`s
/// and unconditional `Jump`s — stopping at the first instruction with any
/// effect. Used by [`drop_vacuous_cond_branches`]; bounded so a cycle of empty
/// blocks cannot hang the walk.
fn resolve_past_nops(insns: &[Insn], mut i: usize) -> Option<usize> {
    let n = insns.len();
    let mut visited = vec![false; n];
    loop {
        if i >= n {
            return Some(i);
        }
        if visited[i] {
            // Cycle detected — refuse to resolve rather than return a
            // potentially wrong index that could silently corrupt control flow.
            return None;
        }
        visited[i] = true;
        match &insns[i].op {
            Op::Nop => i += 1,
            Op::Jump(t) => i = *t,
            _ => return Some(i),
        }
    }
}

/// A conditional branch whose two arms go to the same place is not a branch.
///
/// Predicate, provable without the CFG: resolve the TAKEN target and the
/// FALL-THROUGH index ([`resolve_past_nops`]) to the first instruction that
/// actually does something. If both resolutions land on the same instruction,
/// then taking the branch and falling through execute exactly the same code —
/// the condition has no observable effect.
///
/// This is the IL2CPP null-check collapse, and it is why the dump carries
/// `if (cond) { goto L; } L:` — a `goto` to the very next statement — at scale
/// (494 occurrences in one iOS dump). The native shape is
///
/// ```text
///   cbnz w8, +8         ; class != null -> skip the raise
///   bl   <raise_null>   ; the raise itself: an il2cpp_codegen_* helper
///   ...                 ; both arms arrive here
/// ```
///
/// The `bl` is a runtime helper, so it is excised as `DroppedRuntimeCall` —
/// which leaves the skipped slot EMPTY and makes the two arms coincide. The
/// assembled branch is then a no-op that still *renders* as a condition, and
/// that text is what reads as a corrupted method. Nop the condition; the ops
/// that only fed it are collected by the DCE that runs next, and the address is
/// ledgered so the wrong-logic oracle does not report the deliberate erasure as
/// a missing condition.
///
/// Decisions are computed from the ORIGINAL listing before anything is
/// replaced, so the result does not depend on iteration order.
pub fn drop_vacuous_cond_branches(insns: &mut [Insn]) -> usize {
    let mut vacuous: Vec<usize> = Vec::new();
    for i in 0..insns.len() {
        let Op::CondJump(taken) = insns[i].op else {
            continue;
        };
        if let (Some(t), Some(f)) = (
            resolve_past_nops(insns, taken),
            resolve_past_nops(insns, i + 1),
        ) {
            if t == f {
                vacuous.push(i);
            }
        }
    }
    let mut removed = 0;
    for i in vacuous {
        super::verify::ledger_note(insns[i].address, super::verify::LedgerKind::VacuousBranch);
        // The folded compare shares the branch's address (Compare + CondJump are
        // one native instruction); note it too, since the DCE below will very
        // likely erase it once its only consumer is gone.
        if let Some(prev) = i.checked_sub(1).and_then(|p| insns.get(p)) {
            if matches!(prev.op, Op::Compare(_)) {
                super::verify::ledger_note(prev.address, super::verify::LedgerKind::VacuousBranch);
            }
        }
        insns[i] = Insn::nop(insns[i].address);
        removed += 1;
    }
    removed
}

/// Put the preserved indirect-result operand back at [`SRET_OPERAND`].
///
/// The slot index is the ARM64 ABI slot, not a signature position, so it must
/// be RESTORED AT THAT INDEX, not appended: padding with explicit "no live
/// register" operands is what keeps `[target, dest, receiver, params…, x8]`
/// addressable after a signature-based truncation. Appending it at the end of a
/// short list instead (offset by one per missing parameter) is invisible in the
/// output and makes the renderer read the slot as `None` — the out-parameter
/// that `add x8, sp, #8; bl StructReturning` passes is then simply not printed.
pub fn place_sret(ops: &mut Vec<Operand>, sret: Option<Operand>) {
    use super::ir::Operand as Op_;
    let Some(s) = sret else { return };
    // A signature long enough to reach the slot would make it alias a real
    // parameter; dropping it is better than printing a parameter twice.
    while ops.len() < SRET_OPERAND {
        ops.push(Op_::MissingAbiArg);
    }
    if ops.len() == SRET_OPERAND {
        ops.push(s);
    }
}

/// CallArgumentTrimmer: strip operands beyond the signature's expectation
/// (notably the hidden MethodInfo* on shared generics).
/// Cpp2IL `CallArgumentTrimmer.cs`.
pub fn trim_call_arguments(insns: &mut [Insn]) -> usize {
    let mut trimmed = 0;
    for insn in insns.iter_mut() {
        let params: Option<usize> = match insn.operands.first() {
            Some(Operand::Method(m)) => Some(m.sig.params.len()),
            // An UNRESOLVED call target carries no signature to trim against,
            // but a TRAILING `MissingAbiArg` still cannot mean anything. That
            // operand is the position fact "the ABI had no live register for
            // this slot"; with no declared parameter for the slot to
            // correspond to, printing it emits a bare `__abi` in an argument
            // list — 25,668 of them binary-wide, all of them trailing, and all
            // of them the renderer reporting a hole it was handed rather than
            // a hole that exists. Interior ones are KEPT: there they mark a
            // genuine gap BETWEEN two recovered arguments.
            _ => {
                let mut end = insn.operands.len();
                while end > 0 && matches!(insn.operands[end - 1], Operand::MissingAbiArg) {
                    end -= 1;
                }
                if end < insn.operands.len() && end >= 2 {
                    insn.operands.truncate(end);
                    trimmed += 1;
                }
                continue;
            }
        };
        let this_count = usize::from(
            matches!(insn.op, Op::Call | Op::CallVoid | Op::IndirectCall) && params.is_some(),
        );
        // `Op::Call` ALWAYS carries a destination slot at operand 1: every
        // `bl`/`blr` arm of the lifter allocates x0 there, `dest_position`
        // reports `Some(1)` for it, and both `call_parts` and the statement
        // renderer skip that operand before reading the receiver. Counting it
        // only when the callee actually returns a value (the old
        // `!m.sig.is_void_like()`) therefore made `expected` ONE SHORT for
        // every void-returning method, and this pass truncated the call's
        // LAST ARGUMENT — 58k arguments binary-wide that the renderer then had
        // no operand for, and reported as `__unknown`.
        // `Op::CallVoid` is the op that has no destination slot.
        let dest_slot = usize::from(matches!(insn.op, Op::Call));
        let expected = 1 + dest_slot + this_count + params.unwrap_or(0);
        // AAPCS64 indirect-result slot (operand 10). It sits ONE PAST the
        // declared parameters, so the truncation below would delete it — but
        // it is a real argument (the address of the buffer a struct-returning
        // callee writes through), not register-dump filler. Only a recovered
        // value is preserved; the lifter records the slot at all only for the
        // `add x8, sp, #imm; bl` idiom.
        let sret = (matches!(insn.op, Op::Call | Op::CallVoid)
            && insn.operands.len() > SRET_OPERAND)
            .then(|| insn.operands[SRET_OPERAND].clone())
            .filter(|o| {
                !matches!(
                    o,
                    Operand::Unknown | Operand::MissingAbiArg | Operand::Register(_)
                )
            });
        if insn.op == Op::Call && insn.operands.len() > expected.max(3) {
            insn.operands.truncate(expected.max(3));
            place_sret(&mut insn.operands, sret);
            trimmed += 1;
        } else if insn.op == Op::CallVoid && insn.operands.len() > expected.max(2) {
            insn.operands.truncate(expected.max(2));
            place_sret(&mut insn.operands, sret);
            trimmed += 1;
        }
    }
    trimmed
}

/// ARM64: AAPCS64 indirect-result register — operand index of the `x8` slot on
/// an `Op::Call` operand list `[target, dest, x0 … x7, x8]`.
///
/// The index is a property of the ARM64 operand LAYOUT that the lifter builds,
/// the ABI realignment in `locals` preserves, this pass preserves, and the
/// renderer reads — all four must agree on it.
pub const SRET_OPERAND: usize = 10;

/// TURN A TAIL CALL BACK INTO A CALL.
///
/// `b <code outside this method>` is a tail call: it runs another function and
/// returns ITS result to our caller. The lifter, though, can only see a branch
/// whose target RVA is not an instruction of this method, so the CFG records no
/// successor and the renderer takes the `t_blk >= blocks.len()` arm — "exit
/// edge, emit nothing". The call vanished entirely.
///
/// Verified on `CollectWeeklyFacebookGift` @0x47014C (arm64, iOS): after the
/// `SetLastWeekFacebookGiftWasCollected` dispatch the native loads x0/x1/x2 and
/// does `b 0x3f8824` (through the epilogue), so the method's success path calls a
/// 4th function that the dump never mentioned; and on the SAME method the
/// failure path (`bl 0x4a58bac; ret`) is a genuine non-tail call, which the CFG
/// rendered — the two must look different because they ARE different.
///
/// Rewriting to the `bl`/`ret` pair reuses the existing machinery end to end:
/// the register→SSA-local pass fills the arguments from the live x0–x8, the
/// sret slot is preserved when the caller set one up, and the `Op::Return`
/// makes the block's value flow to the method's own return type.
/// `is_method_start` decides whether the branch target is a KNOWN method's code
/// address. That is what separates a tail call from a TAIL-MERGE — a `b` into a
/// shared epilogue that the compiler placed outside this method's metadata
/// body (the splicer below recovers those by lifting the target's code into this
/// body). Both are out-of-body branches, so the address alone cannot tell them
/// apart; only the metadata can, and a shared epilogue is not a method start.
///
/// `ret_reg` names the register the synthesised call's result lands in. It is
/// `"x0"` on the ARM ABIs, where the result register IS argument register 0,
/// and [`crate::recovery::ir::RET_REG`] on x86-64, where it is a different
/// physical register. Passing the destination wrong would make the call's
/// result and its receiver the same local.
pub fn recover_tail_calls(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    is_method_start: &dyn Fn(u64) -> bool,
    ret_reg: &str,
) -> usize {
    if insns.is_empty() {
        return 0;
    }
    let lo = insns.iter().map(|i| i.address).min().unwrap_or(0);
    let hi = insns.iter().map(|i| i.address).max().unwrap_or(0);
    // Is `x8` holding the address of an indirect-result buffer at each index?
    // Same evidence rule as `lift::attach_sret_register`: a write to x8 whose
    // source is an `AddressOf`, with no intervening control transfer or call
    // (after which x8's value is not ours to reason about).
    //
    // AUDITED (linear-value class): this is a LINEAR register liveness over the
    // listing, so it is the same straight-line approximation the FP passes had
    // — but its only consumer is the decision to append a `Register("x8")`
    // operand to a call synthesized below, and that operand is dropped again by
    // `trim_call_arguments` (its sret preservation accepts a recovered VALUE,
    // never a bare register name). A false positive therefore cannot change a
    // value or an argument; a false negative only omits the register. It never
    // substitutes a value from the write it tracked, so no reachability guard
    // is required.
    let mut x8_is_sret: Vec<bool> = Vec::with_capacity(insns.len());
    {
        let mut live = false;
        for insn in insns.iter() {
            // state ENTERING this instruction, so the branch itself is covered
            x8_is_sret.push(live);
            if insn.is_terminator() || matches!(insn.op, Op::Call | Op::Jump(_) | Op::CondJump(_)) {
                live = false;
            } else if let Some(Operand::Register(r)) = insn.dest() {
                if r == "x8" {
                    live = matches!(insn.operands.get(1), Some(Operand::AddressOf(_)));
                }
            }
        }
    }
    let mut out: Vec<Insn> = Vec::with_capacity(insns.len() + 8);
    // `addresses` is index-ALIGNED with `insns` — `Cfg::build` resolves a
    // branch target RVA to an index into it and then uses that index to walk
    // `insns`. Two instructions out of one therefore needs one extra address
    // entry, or every branch target after this point resolves one instruction
    // early (which silently DISCONNECTS the method's out-of-line blocks: the
    // failure tail of `CollectWeeklyFacebookGift` lost its call that way).
    let mut out_addr: Vec<u64> = Vec::with_capacity(addresses.len() + 8);
    let addr_of = |insns: &[Insn]| -> Vec<u64> { insns.iter().map(|i| i.address).collect() };
    let src_addr = addr_of(insns);
    let mut n = 0;
    for (k, insn) in insns.drain(..).enumerate() {
        let a = src_addr.get(k).copied().unwrap_or(insn.address);

        // INDIRECT TAIL CALL: `br Xn` with the frame already torn down.
        //
        // This is the sibling of the direct case below and was the single
        // largest remaining source of unresolved dispatch. On arm64 a virtual
        // (or interface / delegate) tail call compiles to the callee-saved
        // restores, `add sp, sp, #frame`, then `br xN` where xN was loaded from
        // a vtable slot. The `br` is therefore NOT a jump table — but it used to
        // reach the structurer as a bare `IndirectCall`, which is a CFG
        // terminator with no successor, so the structurer emitted
        // "indirect jump (dispatch not recovered)" plus a forced `goto` into
        // the following block. Measured over a 4000-method sample: 541 of the
        // 610 surviving dispatch sites (88.7%) are this shape, and they are
        // real code — `return fnptr_xN(args)` — that the text was reporting as
        // lost dispatch instead.
        //
        // Discriminator: the frame is gone. `stack_analyzer` models
        // `add sp, sp, #N` as `ShiftStack(+N)` and the prologue `sub sp, sp, #N`
        // as `ShiftStack(-N)`, so a POSITIVE delta between here and the last
        // control transfer means this `br` leaves the function rather than
        // jumping within it. A jump-table dispatch is always mid-function, so
        // it can never satisfy this. Switch recovery still runs AFTER this
        // pass on whatever `br`s remain.
        if matches!(insn.op, Op::IndirectCall) {
            // The callee operand, kept AS-IS for the synthesised call's
            // operand 0 so the resolver can still name it from the slot load.
            //
            // arm64/arm32 `br`/`bx` is a register. So is x86 `jmp rax`. But
            // x86 `jmp [klass + vtable_slot]` is a MEMORY operand — and that
            // is how MSVC compiles a VIRTUAL TAIL CALL:
            //
            // ```text
            // mov rax, [rcx]          ; rax = klass
            // mov r8,  [rax+0x838]    ; companion MethodInfo of the slot below
            // add rsp, 0x30           ; frame torn down
            // pop rbx
            // jmp [rax+0x830]         ; slot 112 → never returns here
            // ```
            //
            // Restricting this arm to `Register` left every one of those as a
            // bare `Op::IndirectCall`, which the vtable resolver
            // (`resolve::rewrite_virtual_calls`) then rewrote into a
            // FALLING-THROUGH `Op::Call`: the block gained an edge into the
            // code laid out after the `jmp` (which the `jmp` never reaches)
            // and the method lost its only exit, so `check_cfg` reported
            // `block N cannot reach an exit` for the whole method and the
            // renderer ended it with `goto <earlier label>;` — an infinite
            // loop in place of `return this.Virtual(...)`. Measured on v24.5:
            // all 5 remaining x64 `WRONG-LOGIC` findings were this shape
            // (0x162ca0, 0x168a80, 0x4afba0, 0x2a9640, 0x37cc50).
            //
            // The INDEXED form (`jmp [table + idx*8]`) is a jump-table
            // dispatch, never a tail call: an IL2CPP vtable access is an
            // immediate offset with no index register. Those stay for
            // `recover_switch_tables`.
            let target = match insn.operands.first() {
                Some(t @ Operand::Register(_)) => t.clone(),
                Some(t @ Operand::Memory { index: None, .. }) => t.clone(),
                // A dispatch through anything else (an indexed table, a
                // recovered local, a PC-relative value) is not this shape.
                _ => {
                    out.push(insn);
                    out_addr.push(a);
                    continue;
                }
            };
            let mut torn_down = false;
            for prev in out.iter().rev().take(24) {
                if prev.is_terminator()
                    || matches!(
                        prev.op,
                        Op::Call
                            | Op::CallVoid
                            | Op::Jump(_)
                            | Op::CondJump(_)
                            | Op::Return
                            | Op::Throw
                    )
                {
                    break;
                }
                if let Op::ShiftStack(d) = prev.op {
                    // The NEAREST frame adjustment decides: a prologue `sub`
                    // anywhere earlier is irrelevant once a later `add` moved
                    // sp back up.
                    torn_down = d > 0;
                    break;
                }
            }
            if !torn_down {
                out.push(insn);
                out_addr.push(a);
                continue;
            }
            let mut regs = vec!["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];
            if x8_is_sret.get(k).copied().unwrap_or(false) {
                regs.push("x8");
            }
            // Operand list is `[target, dest, x0 … x7, x8?]` — the same layout
            // the direct tail call below builds, so `raw_call_stmt` renders the
            // indirect target as `fnptr_<site>(…)` and reads the arguments from
            // the same slots. On x86 the target is the slot-access `Memory`
            // instead of a register name, which `resolve_indirect_target`
            // matches directly (`call [klass + vtable]` is the same shape as
            // `jmp [klass + vtable]`).
            let mut ops = vec![target.clone(), Operand::reg(ret_reg)];
            for r in regs {
                ops.push(Operand::reg(r));
            }
            // FP argument registers, parallel to the x0–x7 dump above. A
            // managed `bl` records `fp_written` here; a tail call is recovered
            // from a bare `b`, so nothing was recorded and a float parameter
            // whose value is simply CARRIED INTO the callee (`s0` untouched
            // after entry — the whole point of a tail call) had no operand at
            // all. `align_call_arguments` then found no register of the FP
            // class and emitted `MissingAbiArg`, which every render path drops
            // trailing: `this.target.set_pitch()` for
            // `set_pitch(float)`. Only registers the method actually declares
            // are kept after alignment; the rest are scrubbed as ABI filler.
            // Flattened lane-0 names for s0..s7 (see lift::scalar_s_name):
            for r in ["vs0", "vs4", "vs8", "vs12", "vs16", "vs20", "vs24", "vs28"] {
                ops.push(Operand::reg(r));
            }
            out.push(Insn::new(Op::Call, ops, a));
            out_addr.push(a);
            out.push(Insn::new(Op::Return, vec![Operand::reg(ret_reg)], a));
            out_addr.push(a);
            // ONE native transfer became a CALL **and** a RETURN. The A/A
            // fingerprint therefore sees one more control-transfer event than
            // the raw lift did — legitimately, so the site is ledgered and the
            // comparator excuses exactly that many.
            super::verify::ledger_note(a, super::verify::LedgerKind::TailCall);
            n += 1;
            continue;
        }

        let t = match insn.op {
            Op::Jump(t) => t as u64,
            _ => {
                out.push(insn);
                out_addr.push(a);
                continue;
            }
        };
        // Only a target OUTSIDE this method's own instruction range is a tail
        // call. An in-range branch is an ordinary control-flow edge the CFG
        // resolves (or a jump-table artefact), and must stay a branch.
        if t == 0 || (t >= lo && t <= hi) || !is_method_start(t) {
            out.push(insn);
            out_addr.push(a);
            continue;
        }
        // `x8` is the AAPCS64 INDIRECT-RESULT register, not a ninth argument:
        // a call site records it only when it materialized the pointer
        // (`add x8, sp, #imm`) immediately before the transfer. Dumping it
        // unconditionally handed the renderer whatever scratch value last
        // lived in x8 — in `CollectWeeklyFacebookGift` that was the `w8`
        // class-init guard load, which then printed as a phantom second
        // argument of `SetLastWeekFacebookGiftWasCollected(int)`.
        let mut ops = vec![Operand::imm(t as i64), Operand::reg(ret_reg)];
        let mut regs = vec!["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];
        if x8_is_sret.get(k).copied().unwrap_or(false) {
            regs.push("x8");
        }
        for r in regs {
            ops.push(Operand::reg(r));
        }
        // FP argument registers (see the indirect tail call above): a value
        // carried unchanged into a tail-called callee lives in s0..s7 and a
        // bare `b` records nothing about it.
        // Flattened lane-0 names for s0..s7 (see lift::scalar_s_name):
            for r in ["vs0", "vs4", "vs8", "vs12", "vs16", "vs20", "vs24", "vs28"] {
            ops.push(Operand::reg(r));
        }
        out.push(Insn::new(Op::Call, ops, a));
        out_addr.push(a);
        out.push(Insn::new(Op::Return, vec![Operand::reg("x0")], a));
        out_addr.push(a);
        // See the indirect arm above: the native `b <other method>` IS a call
        // plus a return, so the extra fingerprint event is expected.
        super::verify::ledger_note(a, super::verify::LedgerKind::TailCall);
        n += 1;
    }
    *insns = out;
    *addresses = out_addr;
    n
}

/// Does a lifted out-of-body TAIL branch back INTO the method body it was
/// reached from?
///
/// This is the stub-versus-real-transfer discriminator in the tail-merge
/// splicer. A generic-method dispatch STUB reloads the callee-saved registers,
/// compares a token and branches back INTO the body, where the real payload
/// call still sits — so it must not be spliced. A genuinely tail-called body
/// has no edge back into the method that jumped to it.
///
/// Targets are RAW RVAs at this stage: the CFG has not yet rewritten them to
/// indices (`Cfg::build` runs after the splicer).
pub fn tail_branches_back_into(tail: &[Insn], body_lo: u64, body_hi: u64) -> bool {
    tail.iter().any(|i| {
        let t = match i.op {
            Op::Jump(t) | Op::CondJump(t) => t as u64,
            _ => return false,
        };
        t >= body_lo && t < body_hi
    })
}

/// RemoveUnused locals: mark versions never read so emitter can skip them.
pub fn unused_locals(insns: &[Insn], table: &super::locals::LocalTable) -> HashSet<usize> {
    let mut used: HashSet<u32> = HashSet::new();
    for insn in insns {
        let mut v = Vec::new();
        for op in &insn.operands {
            collect_locals(op, &mut v);
        }
        used.extend(v);
    }
    table
        .index_of
        .iter()
        .filter(|(version, _)| !used.contains(version))
        .map(|(_, idx)| *idx)
        .collect()
}

/// MetadataInitGuardRemover port (structural subset of Cpp2IL's pass).
///
/// Compiler shape for every managed method prologue:
///   t0 = klass_ptr (immediate)          ; Move(local, Immediate)
///   t1 = *t0                            ; Move(local, Memory{base})
///   t2 = t1 & 1                         ; And
///   t3 = (t2 == 0)                      ; Compare(Equal, _, 0)
///   brtrue → slow path (call class-init, jmp merge)
///
/// Initialization is idempotent, so assuming "already initialized" is safe:
/// excise the check, redirect flow to the merge point, and erase the
/// slow-path region (gated: short, exactly one call, no other branches).
///
/// `_cfg` is deliberately untouched: this pass Nops insns in place and never
/// changes block boundaries, and `ssa::build` recorded each phi's predecessor
/// BLOCK INDICES — rebuilding the CFG here would renumber them and make
/// `ssa::eliminate` index out of range. Callers pass the graph only to keep
/// the pass signature uniform with its siblings.
/// The local an instruction DEFINES, for the value-defining ops the guard
/// eraser is allowed to delete. `None` for stores (operand 0 is a `Memory`,
/// not a local) and for control ops.
fn guard_region_def(insn: &Insn) -> Option<u32> {
    match &insn.op {
        // [0] is the call TARGET here, [1] is the result slot.
        Op::Call | Op::CallVoid | Op::IndirectCall | Op::NewObj(_) | Op::ObjcMsgSend { .. } => {
            insn.operands.get(1).and_then(|o| o.as_local())
        }
        Op::Move
        | Op::Arith(_)
        | Op::FloatArith(_)
        | Op::Compare(_)
        | Op::Not
        | Op::Negate
        | Op::Phi
        | Op::Convert(_)
        | Op::Select(_)
        | Op::Box(_)
        | Op::LoadLength
        | Op::Clz
        | Op::MulAcc { .. }
        | Op::NewArr(..) => insn.operands.first().and_then(|o| o.as_local()),
        _ => None,
    }
}

/// Does this operand read local `v`? Recurses into addressing sub-operands —
/// a phi feeding an `arr[i]` index is still a use.
fn guard_operand_reads(op: &Operand, v: u32) -> bool {
    // Exhaustive, like `collect_locals`: a guard whose only mention of a
    // local is as a field base is still a READ of it, and answering `false`
    // here lets `remove_init_guards` delete a live check.
    match op {
        Operand::Local(l) => *l == v,
        Operand::Memory { base, index, .. } => {
            guard_operand_reads(base, v)
                || index.as_deref().is_some_and(|i| guard_operand_reads(i, v))
        }
        Operand::AddressOf(inner) => guard_operand_reads(inner, v),
        Operand::ArrayAccess { array, index } => {
            guard_operand_reads(array, v) || guard_operand_reads(index, v)
        }
        Operand::ArrayLength(inner) => guard_operand_reads(inner, v),
        Operand::InstanceField { base, .. } => guard_operand_reads(base, v),
        Operand::Register(_)
        | Operand::Immediate(_)
        | Operand::FloatLiteral(_)
        | Operand::DoubleLiteral(_)
        | Operand::StringLiteral { .. }
        | Operand::Method(_)
        | Operand::Field(_)
        | Operand::Type(_)
        | Operand::MissingAbiArg
        | Operand::Unknown => false,
    }
}

pub fn remove_init_guards(insns: &mut [Insn], _cfg: &mut Cfg) -> usize {
    let mut removed = 0usize;
    // ILR_GCENSUS=1: per-method rejection-reason tally (aggregated over the
    // whole method, printed once at exit)
    let gc = crate::recovery::env_flags::flag("ILR_GCENSUS");
    let mut g: std::collections::HashMap<&'static str, u64> = std::collections::HashMap::new();
    macro_rules! mark {
        ($r:expr) => {{
            if gc {
                *g.entry($r).or_insert(0) += 1;
            }
        }};
    }

    // locate candidate compares (either polarity: the branch arm that runs
    // the init is decided structurally below, not from the compare kind)
    let mut i = 0usize;
    while i < insns.len() {
        let Some(Op::Compare(CompareKind::Equal | CompareKind::NotEqual)) =
            Some(insns[i].op.clone())
        else {
            i += 1;
            continue;
        };
        // next real insn must be the conditional branch
        let j = i + 1;
        if j >= insns.len() || !matches!(insns[j].op, Op::CondJump(_)) {
            if gc {
                eprintln!("[gcen] cmp_without_condjump");
            }
            i += 1;
            continue;
        }
        if gc {
            eprintln!("[gcen] candidate cmp={i}");
        }
        // lhs must trace back: And(_, 1) ← Memory{base ← Immediate}
        let Some(lhs_local) = insns[i].operands.get(1).and_then(|o| o.as_local()) else {
            mark!("lhs_not_local");
            i += 1;
            continue;
        };
        let Some(and_idx) = trace_single_def(insns, i, lhs_local) else {
            mark!("no_and_def");
            i += 1;
            continue;
        };
        // Either  and t, #1  (bit test) or a direct byte/word load compare.
        let is_and = matches!(insns[and_idx].op, Op::Arith(ArithKind::And))
            && insns[and_idx].operands.get(2) == Some(&Operand::Immediate(1));
        if !is_and {
            // direct-load compare is acceptable — treat the load as the
            // "and" position for the rest of the matcher
        }
        // And's LHS must be a Memory load whose base derives from an immediate
        let Some(mem_src) = insns[and_idx]
            .operands
            .get(1)
            .cloned()
            .and_then(|op| match &op {
                Operand::Local(v) => trace_source(insns, and_idx, *v),
                other => Some(other.clone()),
            })
        else {
            mark!("lhs_not_mem_chain");
            i += 1;
            continue;
        };
        let Operand::Memory { base, .. } = &mem_src else {
            mark!("lhs_mem_but_not_memop");
            i += 1;
            continue;
        };
        let base_ok = match base.as_ref() {
            Operand::Local(b) => addr_of(insns, and_idx, *b).is_some(),
            Operand::Immediate(_) => true,
            _ => false,
        };
        if !base_ok {
            mark!("base_not_imm");
            i += 1;
            continue;
        }

        // ---- structural gate on the slow path (polarity-agnostic) ----
        // The metadata-init arm is the one whose straight-line region
        // CONTAINS the class-init call(s); the other arm is the fast skip.
        // Which physical side that is depends on the compiler's branch
        // polarity (brtrue-on-uninit vs brtrue-on-init), so decide by
        // structure, never by the compare kind.
        const CHAIN_LIMIT: usize = 12;
        let fall = j + 1;
        let Op::CondJump(taken) = insns[j].op.clone() else {
            unreachable!()
        };

        // Walk a side forward: stop at any control op. Returns where the
        // walk stopped, how many call-like ops it saw, and the terminal
        // Jump target when the walk stopped AT a Jump.
        fn chain_walk(insns: &[Insn], mut cur: usize) -> (usize, usize, Option<usize>) {
            let mut calls = 0usize;
            for _ in 0..CHAIN_LIMIT {
                if cur >= insns.len() {
                    return (cur, calls, None);
                }
                match insns[cur].op {
                    Op::Jump(t) => return (cur, calls, Some(t)),
                    // A THROW counts as an effect, not as the end of a pure
                    // skip. Returning on it with the count unchanged made this
                    // walk report "0 calls" for an arm that transfers control
                    // out of the method, and the gate below reads "0 calls" as
                    // "this is the fast skip that does nothing". On
                    // `Game`'s `.cctor` @0x1cb257c (1860 bytes, 465 native
                    // insns) the taken arm was `ldr x0,[x20]; mov w1,#0x57;
                    // bl …` where the `bl` had already been rewritten to
                    // `Op::Throw` by `resolve` — so the walk stopped one insn
                    // early, counted nothing, and the pass erased the real
                    // static-initializer envelope (2 calls + the
                    // `s_initialized = true` store) and rewired the guard past
                    // it. The whole method then rendered as a lone `throw`: 87%
                    // of its semantics gone. A throw is an observable effect;
                    // it can never be the "pure" side of a guard.
                    Op::Throw => return (cur, calls + 1, None),
                    Op::CondJump(_) | Op::Return => return (cur, calls, None),
                    Op::Call
                    | Op::CallVoid
                    | Op::IndirectCall
                    | Op::NewObj(_)
                    | Op::ObjcMsgSend { .. } => calls += 1,
                    _ => {}
                }
                cur += 1;
            }
            (cur, calls, None)
        }
        let (taken_end, taken_calls, taken_jump) = chain_walk(insns, taken);
        let (fall_end, fall_calls, _fall_jump) = chain_walk(insns, fall);

        // side roles: (slow_start, slow_erase_end /*exclusive*/, merge_idx)
        let mut slow_start = usize::MAX;
        let mut slow_erase_end = usize::MAX;
        let mut merge_idx = usize::MAX;
        if taken_calls == 0 && (1..=3).contains(&fall_calls) {
            // brtrue on *initialized*: taken skips ahead to the merge which
            // sits inside/at the end of the fall-through init region
            if taken > fall && taken <= fall_end && taken <= and_idx + 48 {
                slow_start = fall;
                slow_erase_end = taken;
                merge_idx = taken;
            }
        } else if fall_calls == 0 && (1..=3).contains(&taken_calls) {
            // brtrue on *uninitialized*: taken runs the init chain whose
            // terminal Jump leads to the merge
            if let Some(m) = taken_jump {
                if m > taken && m <= taken_end + 2 && m <= and_idx + 48 {
                    slow_start = taken;
                    slow_erase_end = taken_end; // includes the terminal Jump
                    merge_idx = m;
                }
            }
        }
        if merge_idx == usize::MAX && taken_calls >= 1 && taken_jump.is_none() && fall_calls == 0 {
            // taken-slow WITHOUT a terminal Jump: the init chain runs on
            // past the linear walk and falls through into the shared code
            // the fast arm also reaches. Find that join in the CFG: walk
            // the pure straight-line block chain from `taken`; the merge is
            // the first chain block reachable from the fall arm. Linear
            // heuristics can't express this (this is the OnBeforeMovement
            // guard class), but the graph can.
            let cfg2 = super::ssa::rebuild_cfg(insns);
            let block_of = |idx: usize| {
                cfg2.blocks
                    .iter()
                    .position(|b| idx >= b.start && idx < b.end)
            };
            let pure_op = |op: &Op| {
                matches!(
                    op,
                    Op::Move
                        | Op::Arith(_)
                        | Op::FloatArith(_)
                        | Op::Phi
                        | Op::Nop
                        | Op::Call
                        | Op::CallVoid
                        | Op::IndirectCall
                        | Op::NewObj(_)
                        | Op::NewArr(..)
                        | Op::ObjcMsgSend { .. }
                )
            };
            if let (Some(tb), Some(fb)) = (block_of(taken), block_of(fall)) {
                // straight-line pure chain of blocks from the slow arm
                let mut chain: Vec<usize> = vec![tb];
                let mut cur = tb;
                for _ in 0..12 {
                    let blk = &cfg2.blocks[cur];
                    if !(blk.start..blk.end).all(|k| pure_op(&insns[k].op)) {
                        break; // chain ended at real code — possible merge
                    }
                    let last = &insns[blk.end - 1];
                    let nxt = match last.op {
                        Op::Jump(t) => block_of(t),
                        _ if !last.is_terminator() && blk.end < insns.len() => block_of(blk.end),
                        _ => None,
                    };
                    match nxt {
                        Some(nb) if nb > cur => {
                            cur = nb;
                            chain.push(nb);
                        }
                        _ => break,
                    }
                }
                // BFS reachability from the fast (fall) arm, bounded
                let reachable = |want: usize| -> bool {
                    let mut seen = vec![false; cfg2.blocks.len()];
                    let mut stack = vec![fb];
                    let mut steps = 0usize;
                    while let Some(b) = stack.pop() {
                        steps += 1;
                        if steps > 1024 || b >= cfg2.blocks.len() || seen[b] {
                            continue;
                        }
                        if b == want {
                            return true;
                        }
                        seen[b] = true;
                        for &s in cfg2.successors[b].iter() {
                            if s < cfg2.blocks.len() {
                                stack.push(s);
                            }
                        }
                    }
                    false
                };
                // first chain block (after the guard's own) both sides reach
                let mut merge_b = None;
                for (ci, &b) in chain.iter().enumerate() {
                    if ci > 0 && reachable(b) {
                        merge_b = Some(b);
                        break;
                    }
                }
                // or the first block after the pure chain (real code head)
                let merge_b = merge_b.or_else(|| {
                    let head = chain.last().copied()?;
                    let blk = &cfg2.blocks[head];
                    let nxt = match insns[blk.end - 1].op {
                        Op::Jump(t) => block_of(t),
                        _ if !insns[blk.end - 1].is_terminator() && blk.end < insns.len() => {
                            block_of(blk.end)
                        }
                        _ => None,
                    };
                    nxt.filter(|&nb| nb > head && reachable(nb))
                });
                if let Some(mb) = merge_b {
                    let m = cfg2.blocks[mb].start;
                    if m > taken && m <= and_idx + 96 {
                        slow_start = taken;
                        slow_erase_end = m;
                        merge_idx = m;
                    } else if gc {
                        eprintln!(
                            "[gcen] cfg merge mb={mb} m={m} taken={taken} and={and_idx} out-of-range"
                        );
                    }
                } else if gc {
                    eprintln!(
                        "[gcen] cfg no-merge taken_blk={tb} chain={:?} fb={fb} taken={taken} fall={fall}",
                        chain
                    );
                }
            }
        }
        if merge_idx == usize::MAX {
            mark!("merge_not_found");
            i += 1;
            continue;
        }
        if slow_erase_end <= slow_start || slow_erase_end - slow_start < 2 {
            mark!("region_too_short");
            i += 1;
            continue;
        }
        // A CALL IS NOT INIT BOILERPLATE.
        //
        // Every branch above matches on STRUCTURE — arms with calls vs an arm
        // without, joined ahead. That shape is also exactly what a per-usage
        // lazy-init guard looks like, and IL2CPP emits ONE per metadata usage,
        // so one state machine can carry dozens. Erasing the chain deletes the
        // `bl` sitting in it, while the callee still runs natively.
        //
        // Measured on 0x1fe2ca0: the lifter produced 272 `Op::Call` for 272
        // native `bl` (byte-identical count — `ILR_AFTERLIFT`), the renderer
        // saw 160, and this pass alone accounted for the 112 that vanished.
        // "Assume the class is initialized" is a readability convention and it
        // has a fidelity price: pay it only for chains that have NOTHING to
        // run. A chain that calls something is either real logic or a genuine
        // one-time init, and in both cases the disassembly contains a call the
        // recovered text must not silently drop.
        let region_has_call = (slow_start..slow_erase_end).any(|k| {
            matches!(
                insns[k].op,
                Op::Call
                    | Op::CallVoid
                    | Op::IndirectCall
                    | Op::NewObj(_)
                    | Op::NewArr(..)
                    | Op::ObjcMsgSend { .. }
            )
        });
        if region_has_call {
            mark!("region_has_call");
            i += 1;
            continue;
        }

        // NEVER ERASE A VALUE THE JOIN STILL READS.
        //
        // The purity gate below admits `Op::Phi`, and `ssa::materialize`
        // splices each phi at its block's FIRST insn — so when the join block
        // opens with a phi, `slow_erase_end` (= the join's instruction index)
        // cuts the join's OWN phi header into the erased region. The guard then
        // disappears while the value it selected stays referenced, and the
        // join reads a local nothing defines any more: the undefined-local
        // class (`return v11;`).
        //
        // Verified on `RuntimeMarshal.MarshalString` @0x17005f8 (ARM32). Native:
        //   ldrb r0,[flag]; cmp r0,#0; bne join; <init chain: 3 loads, bl,
        //   ldr, mov, strb flag=1>; join: ...
        // The erase region 13..21 spanned the init chain AND the join's phi
        // (which fed the method's `Return`), so the rendered body lost the
        // whole conditional and its `bl` while keeping `return v11;`.
        //
        // Erasing is safe only when nothing the region defines is read outside
        // it. Otherwise the guard is real control flow: leave it alone.
        let defs_read_outside = (slow_start..slow_erase_end).any(|k| {
            guard_region_def(&insns[k]).is_some_and(|d| {
                insns.iter().enumerate().any(|(k2, ins2)| {
                    (k2 < slow_start || k2 >= slow_erase_end)
                        && ins2.operands.iter().any(|o| guard_operand_reads(o, d))
                })
            })
        });
        if defs_read_outside {
            mark!("region_defs_read_after_join");
            i += 1;
            continue;
        }
        // The erased region must be pure init machinery: straight-line data
        // movement, arithmetic, calls and phis — any real control op (or an
        // actual store-less compare) inside means this is real logic, not a
        // guard. This whitelist is what keeps the removal from eating real
        // if/else bodies that merely happen to follow a bit-test.
        let is_pure_env = |op: &Op| {
            matches!(
                op,
                Op::Move
                    | Op::Arith(_)
                    | Op::FloatArith(_)
                    | Op::Phi
                    | Op::Nop
                    | Op::Call
                    | Op::CallVoid
                    | Op::IndirectCall
                    | Op::NewObj(_)
                    | Op::NewArr(..)
                    | Op::ObjcMsgSend { .. }
                    | Op::Box(_)
                    | Op::Convert(_)
                    | Op::Not
                    | Op::Negate
                    | Op::MulAcc { .. }
                    | Op::Clz
                    | Op::LoadLength
                    | Op::ShiftStack(_)
            )
        };
        // Purity gate: the erased region must be straight-line init
        // machinery (moves, arithmetic, calls, phis). Any real control op
        // means this is logic, not a guard envelope — UNLESS it is the
        // tail-merged shared re-test at the region's end (handled below).
        if !(slow_start..slow_erase_end).all(|k| is_pure_env(&insns[k].op)) {
            // TAIL-MERGED ENVELOPES: the compiler can land the fast arm's
            // jump on a SHARED re-test just past the init region (a nested
            // class-init check tail-merged into this one). The region then
            // ends with:  Compare K (v, 0); Phi(dest, fast_val, init_val);
            // CondJump T.  Both arms reaching that re-test have
            // "initialized" true — the fast arm skipped init because the
            // guard byte was already set, and the slow arm just stored a
            // nonzero constant to the SAME slot the guard tests — so with
            // Compare(Equal) the branch never fires on either path. Keeping
            // it renders the contradictory `if (X != 0) { if (X == 0) {`
            // class. Erase the init region AND the re-test's Compare+branch,
            // but KEEP the Phi (its value — fast_val vs init_val — is real
            // per-arm data flow that downstream code consumes) and point
            // the guard's redirect at it, so SSA lowering materializes both
            // edge values exactly as before.
            let mut tm_done = false;
            let t_last = slow_erase_end - 1;
            if matches!(insns[t_last].op, Op::CondJump(_)) {
                // walk back over Phis/Nops to the re-test's Compare
                let mut c = t_last;
                while c > slow_start && matches!(insns[c - 1].op, Op::Phi | Op::Nop) {
                    c -= 1;
                }
                if c > slow_start && matches!(insns[c - 1].op, Op::Compare(_)) {
                    c -= 1;
                    let kind = match &insns[c].op {
                        Op::Compare(k) => *k,
                        _ => unreachable!(),
                    };
                    // head must be exactly Compare + (Phi|Nop)* + CondJump
                    let head_ok = (c + 1..t_last).all(|k| matches!(insns[k].op, Op::Phi | Op::Nop));
                    // store-proof: the init region stores a nonzero constant
                    // (directly, or via a nearby temp) to the same memory
                    // slot this guard tests
                    let guard_addend: Option<i64> = match &mem_src {
                        Operand::Memory { addend, .. } => Some(*addend),
                        _ => None,
                    };
                    let nonzero_store = |k: usize| -> bool {
                        if !matches!(&insns[k].op, Op::Move) {
                            return false;
                        }
                        let Some(Operand::Memory { addend, .. }) = insns[k].operands.first() else {
                            return false;
                        };
                        if *addend != guard_addend.unwrap_or(i64::MIN) {
                            return false;
                        }
                        match insns[k].operands.get(1) {
                            Some(Operand::Immediate(v)) => *v != 0,
                            Some(Operand::Local(l)) => (0..k).rev().take(8).any(|m| {
                                insns[m].dest().and_then(|d| d.as_local()) == Some(*l)
                                    && matches!(
                                        insns[m].operands.get(1),
                                        Some(Operand::Immediate(v)) if *v != 0
                                    )
                            }),
                            _ => false,
                        }
                    };
                    let store_proof =
                        guard_addend.is_some() && (slow_start..c).any(|k| nonzero_store(k));
                    if head_ok
                        && store_proof
                        && matches!(kind, CompareKind::Equal)
                        && c + 1 < insns.len()
                    {
                        // erase the re-test's Compare + branch; keep Phis
                        let mut kill = Vec::new();
                        for k in c..=t_last {
                            if matches!(insns[k].op, Op::Compare(_) | Op::CondJump(_)) {
                                kill.push(k);
                            }
                        }
                        for k in kill {
                            super::verify::ledger_note(
                                insns[k].address,
                                super::verify::LedgerKind::GuardExcision,
                            );
                            insns[k] = Insn::nop(insns[k].address);
                        }
                        slow_erase_end = c;
                        // the fast arm must now flow through the kept Phi
                        merge_idx = c + 1;
                        tm_done = true;
                    }
                }
            }
            if !tm_done {
                mark!("region_impure");
                i += 1;
                continue;
            }
        }

        // ---- rewrite ----------------------------------------------------
        // nop guard insns [and_idx ..= j] plus their feeders
        let mut nops_this_guard = 0usize;
        for k in and_idx..=j {
            if !matches!(insns[k].op, Op::Nop) {
                super::verify::ledger_note(
                    insns[k].address,
                    super::verify::LedgerKind::GuardExcision,
                );
                insns[k] = Insn::nop(insns[k].address);
                nops_this_guard += 1;
            }
        }
        // also nop feeder moves above the And (klass ptr + deref chain):
        // any Move whose destination is used by the And or the deref load
        // chase the whole feeder chain: any Move whose destination is a
        // known feeder OR whose value feeds a known feeder gets erased
        // (covers klass-ptr immediate → deref load → mask chains)
        if let Some(and_lhs) = insns[and_idx].operands.get(1).and_then(|o| o.as_local()) {
            let mut feeders: Vec<u32> = vec![and_lhs];
            for k in (0..and_idx).rev().take(10) {
                if !matches!(insns[k].op, Op::Move) {
                    continue;
                }
                let Some(d) = insns[k].dest().and_then(|d| d.as_local()) else {
                    continue;
                };
                if feeders.contains(&d) {
                    super::verify::ledger_note(
                        insns[k].address,
                        super::verify::LedgerKind::GuardExcision,
                    );
                    insns[k] = Insn::nop(insns[k].address);
                    nops_this_guard += 1;
                    // push THIS move's source local; also descend through
                    // memory bases (deref loads: b = *a)
                    match insns[k].operands.get(1) {
                        Some(Operand::Local(src)) => feeders.push(*src),
                        Some(Operand::Memory { base, .. }) => {
                            if let Some(src) = base.as_local() {
                                feeders.push(src);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        // nop the whole init region (unreachable now)
        for k in slow_start..slow_erase_end {
            if !matches!(insns[k].op, Op::Nop) {
                super::verify::ledger_note(
                    insns[k].address,
                    super::verify::LedgerKind::GuardExcision,
                );
                insns[k] = Insn::nop(insns[k].address);
                nops_this_guard += 1;
            }
        }
        // jump over everything to the merge
        insns[j] = Insn::new(Op::Jump(merge_idx), vec![], insns[j].address);

        removed += nops_this_guard;
        i = merge_idx.max(i + 1);
    }

    if removed > 0 {
        // Do NOT overwrite `cfg` here: phis were already materialized into
        // the listing by `ssa::build`+`materialize`, and their pred records
        // reference block indices of the cfg as it existed at build time. A
        // rebuild after Nop'ing guard branches merges blocks and shifts
        // indices, so `ssa::eliminate` would index out of range. The guard
        // removal only Nops/redirects instructions — the pre-existing block
        // partition stays a valid superset for anything still reading it,
        // and `ssa::eliminate` rebuilds the cfg itself when it finishes.
        dead_code_elimination(insns);
    }
    if gc && !g.is_empty() {
        let mut v: Vec<_> = g.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        eprintln!(
            "[gcen] removed={removed} {}",
            v.iter()
                .map(|(k, c)| format!("{k}={c}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    removed
}

/// Find the single dominating definition of `local` scanning backwards.
fn trace_single_def(insns: &[Insn], at: usize, local: u32) -> Option<usize> {
    for k in (0..at).rev().take(48) {
        if let Some(Operand::Local(d)) = insns[k].dest() {
            if *d == local {
                return Some(k);
            }
        }
    }
    None
}

/// What fed `local`'s definition at `at` (Move sources only).
fn trace_source(insns: &[Insn], at: usize, local: u32) -> Option<Operand> {
    let def = trace_single_def(insns, at, local)?;
    if !matches!(insns[def].op, Op::Move) {
        return None;
    }
    insns[def].operands.get(1).cloned()
}

/// Fold a register to its constant ADDRESS value through the classic
/// `ADRP page; ADD reg, reg, #low` (and double-add) materialization chain.
/// Without this, most metadata-init guards were REJECTED: constant folding
/// doesn't always merge the pair into one Move, so the base register's def
/// is an `Arith(Add)`, not an `Immediate`, and `base_ok` bailed — leaving
/// ~7k guard+retest shells in the dump. Returns the folded constant.
fn addr_of(insns: &[Insn], at: usize, local: u32) -> Option<i64> {
    fn go(insns: &[Insn], at: usize, local: u32, depth: usize) -> Option<i64> {
        if depth > 4 {
            return None;
        }
        let def = trace_single_def(insns, at, local)?;
        match &insns[def].op {
            Op::Move => match insns[def].operands.get(1) {
                Some(Operand::Immediate(v)) => Some(*v),
                _ => None,
            },
            // ADRP+ADD / second ADD: reg = reg + imm
            Op::Arith(ArithKind::Add) => {
                let rhs = match insns[def].operands.get(2) {
                    Some(Operand::Immediate(v)) => *v,
                    _ => return None,
                };
                let lhs = match insns[def].operands.get(1) {
                    Some(Operand::Local(b)) => go(insns, def, *b, depth + 1)?,
                    Some(Operand::Immediate(v)) => *v,
                    _ => return None,
                };
                Some(lhs + rhs)
            }
            _ => None,
        }
    }
    go(insns, at, local, 0)
}

/// SwitchTableRecovery: ARM64 jump-table shape
///   Move(t0, Imm(table_addr))            ; ADRP+ADD materialized
///   Move(t1, Mem{base:t0, index:i, scale:4})
///   IndirectCall [reg == t1's origin]    ; BR
/// reads the table via `read_u32` and rewrites the branch into
/// Op::Switch(case_target_insn_idx…) with case values preserved.
/// Constant-propagate an operand through Move/Add chains (bounded).
fn const_value(insns: &[Insn], at: usize, op: &Operand) -> Option<u64> {
    const DEPTH: usize = 6;
    fn walk(insns: &[Insn], at: usize, op: &Operand, d: usize) -> Option<u64> {
        if d > DEPTH {
            return None;
        }
        match op {
            Operand::Immediate(v) => Some(*v as u64),
            Operand::Register(name) => {
                let def = (0..at).rev().find(
                    |&k| matches!(insns[k].dest(), Some(Operand::Register(n)) if *n == *name),
                )?;
                match insns[def].op {
                    Op::Move => walk(insns, def, insns[def].operands.get(1)?, d + 1),
                    Op::Arith(ArithKind::Add) => {
                        let a = walk(insns, def, insns[def].operands.get(1)?, d + 1)?;
                        let b = walk(insns, def, insns[def].operands.get(2)?, d + 1)?;
                        Some(a.wrapping_add(b))
                    }
                    _ => None,
                }
            }
            Operand::Local(v) => {
                let def = (0..at)
                    .rev()
                    .find(|&k| insns[k].dest().and_then(|x| x.as_local()) == Some(*v))?;
                match insns[def].op {
                    Op::Move => walk(insns, def, insns[def].operands.get(1)?, d + 1),
                    Op::Arith(ArithKind::Add) => {
                        let a = walk(insns, def, insns[def].operands.get(1)?, d + 1)?;
                        let b = walk(insns, def, insns[def].operands.get(2)?, d + 1)?;
                        Some(a.wrapping_add(b))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
    walk(insns, at, op, 0)
}

/// Nearest definition of a register/local operand before `before`. Works both
/// pre-SSA (physical registers, nearest previous write wins) and post-SSA
/// (SSA locals, single def — same result).
///
/// AUDITED (linear-value class) — CLEARED for the post-SSA users, NOT guarded
/// for the pre-SSA ones. Post-SSA the operand is an SSA version, which has one
/// definition, so "nearest earlier write" IS the reaching definition and a
/// dominance check could never refuse a hop. Pre-SSA (the jump-table
/// recognizers, which run before renaming) the operand is a PHYSICAL register
/// and this really is a linear approximation — but nothing is taken on trust:
/// every candidate table is VALIDATED against the binary (`read_u32` at the
/// computed address, at least two entries landing inside the method's own
/// instruction range, and a `cmp idx, #N` bound), so a base read out of a
/// sibling arm has to produce plausible in-span targets to be accepted rather
/// than rejected. Guarding it here would also refuse legitimate
/// cross-block materializations, since `def_at` is shared with
/// `remove_init_guards`; if a wrong switch is ever observed in a dump diff, the
/// place to add `slot_write_reaches` is the recognizer's dispatch-register
/// lookup — with `ILR_SWGAPS` listing what was rejected — not this helper.
fn def_at(insns: &[Insn], op: &Operand, before: usize) -> Option<usize> {
    match op {
        Operand::Local(v) => (0..before)
            .rev()
            .find(|&k| insns[k].dest().and_then(|d| d.as_local()) == Some(*v)),
        Operand::Register(n) => (0..before)
            .rev()
            .find(|&k| matches!(insns[k].dest(), Some(Operand::Register(m)) if m == n)),
        _ => None,
    }
}

/// The value operand of a def/use chain must be a register or an SSA local.
fn as_value(op: &Operand) -> Option<&Operand> {
    match op {
        Operand::Local(_) | Operand::Register(_) => Some(op),
        _ => None,
    }
}

/// Chase a pure register-copy chain backwards (`mov w8, w24`) and return the
/// value the copy was made from.
///
/// LLVM always materialises the switch index into a fresh register before
/// indexing the jump table, so a bounds guard and the table load never name the
/// same physical register:
///
/// ```text
/// 17ecf80: cmp  w24, #0x8        <- guard names w24
/// 17ecf84: b.hi default
/// 17ecf8c: mov  w8, w24          <- index copied to a fresh register
/// 17ecf98: ldrb w11, [x9, x8]    <- load names x8
/// ```
///
/// Correlating guard to load on raw register identity therefore never matched:
/// measured over the corpus, 0 of 219 dispatch sites have the guard on the
/// load's own register and 215 of 219 name a different one. Chasing the copy
/// chain on both sides restores the correlation. Bounded to 4 hops; every
/// result is still validated by the table read itself.
fn canonical_value(insns: &[Insn], op: &Operand, before: usize) -> Operand {
    let mut cur = op.clone();
    for _ in 0..4 {
        let Some(def_i) = def_at(insns, &cur, before) else {
            break;
        };
        let ins = &insns[def_i];
        if !matches!(ins.op, Op::Move) {
            break;
        }
        let Some(src) = ins.operands.get(1).and_then(as_value) else {
            break;
        };
        if *src == cur {
            break;
        }
        cur = src.clone();
    }
    cur
}

/// Chase a switch-value chain back to its indexed memory load.
///
/// LLVM ARM64 jump tables build the dispatch register as
/// `anchor + (loaded_entry << shift)`. The shape is
/// `Move e = Mem[T + idx*s]; [Shl z = e << k;] Add c = a + z; IndirectCall c`
/// (k is the shift found between the load and the add). Returns (load insn
/// index, accumulated shift) when the chain bottoms out in an indexed load;
/// None otherwise.
fn chase_table_load(insns: &[Insn], start: &Operand, at: usize) -> Option<(usize, u32)> {
    let mut cur = as_value(start)?.clone();
    let mut shift: u32 = 0;
    for _ in 0..4 {
        let d = def_at(insns, &cur, at)?;
        match &insns[d].op {
            Op::Arith(ArithKind::Shl) => {
                let amt = match insns[d].operands.get(2)? {
                    Operand::Immediate(v) => *v,
                    _ => return None,
                };
                if amt < 0 || amt > 31 {
                    return None;
                }
                shift = shift.saturating_add(amt as u32);
                cur = as_value(insns[d].operands.get(1)?)?.clone();
            }
            Op::Move => {
                let src = insns[d].operands.get(1)?;
                if let Operand::Memory { index: Some(_), .. } = src {
                    return Some((d, shift));
                }
                cur = as_value(src)?.clone();
            }
            _ => return None,
        }
    }
    None
}

/// What a jump-table recognizer proved about one dispatch.
///
/// `targets` alone is not the whole fact. The guard that bounds the table
/// (`cmp idx, #N; b.hi default`) names the **default target** — the block a C#
/// `switch` reaches when no case matches — and the table load names the
/// **discriminant** (the value being switched on). Both used to be discarded:
/// the switch instruction was rebuilt with an EMPTY operand vector, so the
/// renderer could only emit the case arms and had to guess the discriminant by
/// scanning back for any `Compare`. Measured before the fix: 0 of 222 switches
/// carried a `default:` and 37 printed the literal `switchValue`.
///
/// `default` is the guard's taken-target **RVA** (not an instruction index):
/// `ssa::transform` remaps `Jump`/`CondJump`/`Switch` targets because it
/// inserts phi-copy instructions, and anything stored in `operands` is not
/// remapped. An RVA survives.
struct TableRecovery {
    /// Case target RVAs; index == case value.
    targets: Vec<u64>,
    /// The guard's taken-target RVA, when the bound compare feeds a
    /// conditional branch (the only shape that proves a default exists).
    default: Option<u64>,
    /// The canonical table-index operand — the value the switch dispatches on.
    disc: Option<Operand>,
}

/// LLVM/Clang ARM64 anchored jump table (16- or 32-bit entries):
///
/// ```text
///   adr   x10, .Lanchor            Move a = Imm(anchor)
///   ldrh  w11, [x9, x8, lsl #1]    Move e = Mem[T + idx*2]    (scale 2/4)
///   add   x10, x10, w11, lsl #2    Shl z = e << k;  Add c = a + z
///   br    x10                      IndirectCall c
/// ```
///
/// Entry `case` holds `(target - anchor) / (1 << k)`, so
/// `target = anchor + entry * (1 << k)`. Before the ARM64 lift fix the
/// shifted-register source was dropped (`add x10,x10,w11,lsl#2` degraded to
/// `x10 + x10`), which dead-coded the load and hid every such dispatch.
fn anchor_table_targets(
    insns: &[Insn],
    i: usize,
    lo: u64,
    hi: u64,
    read_u32: &dyn Fn(u64) -> Option<u32>,
) -> Option<TableRecovery> {
    let c = as_value(insns[i].operands.first()?)?.clone();
    let add_i = def_at(insns, &c, i)?;
    if !matches!(insns[add_i].op, Op::Arith(ArithKind::Add)) {
        return None;
    }
    let add = &insns[add_i];
    let a = add.operands.get(1)?.clone();
    let b = add.operands.get(2)?.clone();
    // One add source is the const anchor, the other is the table-value chain.
    let (anchor, load_i, shift) = match (
        const_value(insns, add_i, &a),
        chase_table_load(insns, &b, add_i),
    ) {
        (Some(anchor), Some((li, sh))) => (anchor, li, sh),
        _ => match (
            const_value(insns, add_i, &b),
            chase_table_load(insns, &a, add_i),
        ) {
            (Some(anchor), Some((li, sh))) => (anchor, li, sh),
            _ => return None,
        },
    };
    let Operand::Memory {
        base,
        addend,
        scale,
        index,
        ..
    } = &insns[load_i].operands[1]
    else {
        return None;
    };
    // Entry width. LLVM/Clang choose the narrowest entry that still covers the
    // code span the table indexes: 2-byte entries for small tables, 4-byte for
    // larger, 8-byte for 64-bit offsets — and, whenever the span divided by 4
    // fits in a byte, a BYTE table loaded with `ldrb`. On arm64 the byte form
    // is what clang actually emits for most C# `switch` statements (they are
    // small), so the "common" tbl is `ldrb`.
    //
    // This used to be `if *scale != 2 && *scale != 4 { return None }`, which
    // rejected `scale == 1` — and THAT is why `recover_switch_tables` recovered
    // only 6 tables across a 3,941-method sample while 511 dispatch sites
    // stayed `IndirectCall`. The fallout is not confined to the switch: an
    // unresolved `IndirectCall` is a CFG terminator, so the case bodies are
    // never walked, and every fact inside them (conditions, calls, field
    // offsets, immediates) is reported as lost. Measured on 0x2647100: 21 of 46
    // blocks never entered, 18 of 27 conditions missing — one dispatch failure
    // accounting for the whole deficit.
    let (stride, entry_bits): (u64, u32) = match *scale {
        1 => (1, 8),
        2 => (2, 16),
        4 => (4, 32),
        8 => (8, 64),
        _ => return None,
    };
    // Case-count bound from the dispatch guard right before the table: LLVM
    // emits `cmp idx, #N; b.hi default` then indexes the table 0..=N. Without
    // this cap the scan keeps reading whatever data follows the real table
    // (adjacent jump tables / pointers), whose entries also happen to land
    // inside the method span and invent phantom switch arms.
    let mut max_case: u64 = 1023;
    // Find the `cmp idx, #N` guard that bounds the table, right before the dispatch.
    //
    // NEAREST FIRST. Scanning forward took the EARLIEST `cmp reg, #imm` that
    // mentioned the index register - an unrelated check earlier in the method,
    // usually `cmp x0, #0` - and capped the table there. On 0x1cfbc6c that set
    // `max_case = 0`, so only one entry was read, `out.len() >= 2` failed and the
    // whole 6-arm switch fell back to `IndirectCall`: its case bodies were never
    // walked and all 61 native conditions plus 31 calls inside them were reported
    // as lost. The real bound is the LAST such compare before the dispatch.
    //
    // CORRELATION BY VALUE, NOT BY REGISTER NAME. The guard names the original
    // index register while the table load names a fresh copy of it (see
    // `canonical_value`). Matching on register identity alone never fires, which
    // leaves `max_case` at its 1023 initialiser and makes the scan read whatever
    // follows the real table: NGUIMath.ResizeWidget @0x17ECE8C has a 9-entry
    // table and emitted 479 phantom cases.
    let want = index.as_deref().map(|ix| canonical_value(insns, ix, load_i));
    // Strict shape first (`cmp` feeding a `b.cond`), then a lenient fallback so
    // sites that previously matched on an unusual guard shape keep their bound.
    //
    // The strict shape is ALSO the only evidence of a default arm: `b.hi` is
    // taken exactly when the index is out of range, so its target is the block
    // C# would write under `default:`. It is captured here instead of being
    // dropped on the floor.
    let mut strict: Option<(u64, u64)> = None;
    let mut lenient: Option<u64> = None;
    for k in (i.saturating_sub(40)..i).rev() {
        if !matches!(insns[k].op, Op::Compare(_)) {
            continue;
        }
        let op1 = insns[k].operands.get(1);
        let op2 = insns[k].operands.get(2);
        let imm_of = |o: Option<&Operand>| match o {
            Some(Operand::Immediate(n)) if *n >= 0 && *n < 1024 => Some(*n as u64),
            _ => None,
        };
        let bound_for = |reg: Option<&Operand>, imm: Option<&Operand>| -> Option<u64> {
            let (reg, imm) = (reg?, imm?);
            match want.as_ref() {
                // No index operand to correlate against: keep the old
                // register-identity rule so behaviour is unchanged there.
                None => match (index.as_deref(), reg) {
                    (Some(Operand::Local(ix)), Operand::Local(v)) if ix == v => {}
                    (Some(Operand::Register(ix)), Operand::Register(v)) if ix == v => {}
                    _ => return None,
                },
                Some(w) => {
                    if canonical_value(insns, reg, k) != *w {
                        return None;
                    }
                }
            }
            imm_of(Some(imm))
        };
        if let Some(n) = bound_for(op1, op2).or_else(|| bound_for(op2, op1)) {
            // The default target is the branch this compare guards. At this
            // point (step 1.5, before `Cfg::build`) a `CondJump` still holds
            // an RVA, which is exactly the form the renderer can resolve
            // without depending on instruction numbering.
            match insns.get(k + 1).map(|x| &x.op) {
                Some(Op::CondJump(t)) => {
                    strict = Some((n, *t as u64));
                    break;
                }
                _ => {
                    if lenient.is_none() {
                        lenient = Some(n);
                    }
                }
            }
        }
    }
    // A lenient bound still caps the table, but it proves nothing about a
    // default arm (no branch consumes the compare), so `default` stays None.
    let default = strict.map(|(_, d)| d);
    match (strict, lenient) {
        (Some((n, _)), _) => max_case = n,
        (None, Some(n)) => max_case = n,
        (None, None) => {}
    }

    let tbl = const_value(insns, load_i, base)?.wrapping_add(*addend as u64);
    let mul = 1u64 << shift;
    if crate::recovery::env_flags::flag("ILR_SWDBG") {
        eprintln!(
            "[anchor] @{:#x} anchor={:#x} load_i={} shift={} stride={} entry_bits={} max_case={} tbl={:#x} first_read={:?} lo={:#x} hi={:#x}",
            insns[i].address,
            anchor,
            load_i,
            shift,
            stride,
            entry_bits,
            max_case,
            tbl,
            read_u32(tbl),
            lo,
            hi
        );
    }
    let mut out = Vec::new();
    for case in 0..=max_case {
        // `read_u32` is a little-endian 32-bit window, so the low 8/16 bits of
        // the window ARE the entry for the 1- and 2-byte forms. Walking the
        // stride by `case * stride` keeps the byte form (stride 1) reading
        // consecutive bytes where the real table is.
        let Some(w0) = read_u32(tbl.wrapping_add(case * stride)) else {
            break;
        };
        let raw = match entry_bits {
            8 => (w0 & 0xFF) as u64,
            16 => (w0 & 0xFFFF) as u64,
            32 => w0 as u64,
            _ => {
                let Some(w1) = read_u32(tbl.wrapping_add(case * stride + 4)) else {
                    break;
                };
                (w0 as u64) | ((w1 as u64) << 32)
            }
        };
        let tgt = anchor.wrapping_add(raw.wrapping_mul(mul));
        if tgt < lo || tgt > hi {
            break;
        }
        out.push(tgt);
    }
    (out.len() >= 2).then_some(TableRecovery {
        targets: out,
        default,
        disc: want,
    })
}

/// `ILR_SWGAPS=<path>` — append every dispatch (`br Xn`) that the recognizer
/// could NOT resolve, together with the ops leading into it.
///
/// Without this the remaining failures are unenumerable: a `br` is a CFG
/// terminator, so an unresolved one silently truncates the walk and the only
/// visible symptom is a count of "unwalked blocks" somewhere else entirely.
/// The dump turns "483 dispatch sites still fail" into "N are shape A, M are
/// shape B", which is what makes the next change aimable rather than a guess.
fn swgap_dump(insns: &[Insn], i: usize, why: &str) {
    use std::io::Write;
    use std::sync::OnceLock;
    static PATH: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    let Some(p) = PATH.get_or_init(|| std::env::var("ILR_SWGAPS").ok().map(Into::into)) else {
        return;
    };
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(p)
    else {
        return;
    };
    let _ = writeln!(
        f,
        "GAP @{:#x} why={} dst={:?}",
        insns[i].address,
        why,
        insns[i].operands.first()
    );
    for k in i.saturating_sub(12)..i {
        let _ = writeln!(
            f,
            "    [{k}] @{:#x}: {:?} {:?}",
            insns[k].address, insns[k].op, insns[k].operands
        );
    }
}

/// ARM32 jump-table dispatch:
///
/// ```text
///   add  r2, pc, #8          Move base = Imm(table_addr)   (PC materialisation)
///   lsl  r1, r1, #2          Move idx4 = idx*4     (folds into the load)
///   ldr  r1, [r1, r2]        Move entry = Mem{base: idx4, index: base}
///   add  pc, r1, r2          IndirectCall entry       (the dispatch)
/// ```
///
/// Two things make Shape 1 reject this. The TABLE BASE sits in the load's
/// INDEX operand, not `base` (the compiler prefers `[idx, base]` here), and
/// the entries are 4-byte offsets ADDED to the base — `target = base + entry` —
/// not `anchor + entry * (1 << k)`. A `const_value` probe on the index operand
/// recovers the base regardless of which side was chosen.
fn arm32_table_targets(
    insns: &[Insn],
    i: usize,
    lo: u64,
    hi: u64,
    read_u32: &dyn Fn(u64) -> Option<u32>,
) -> Option<Vec<u64>> {
    let c = as_value(insns[i].operands.first()?)?.clone();
    let load_i = def_at(insns, &c, i)?;
    if !matches!(insns[load_i].op, Op::Move) {
        return None;
    }
    let Operand::Memory {
        base,
        index,
        addend,
        ..
    } = insns[load_i].operands.get(1)?
    else {
        return None;
    };
    let tbl = index
        .as_deref()
        .and_then(|ix| const_value(insns, load_i, ix))
        .or_else(|| const_value(insns, load_i, base))?;
    let tbl = tbl.wrapping_add(*addend as u64);
    let mut out = Vec::new();
    for case in 0..1024u64 {
        let Some(raw) = read_u32(tbl.wrapping_add(case * 4)) else {
            break;
        };
        let off = raw as u64;
        let rel = tbl.wrapping_add(off);
        if off >= 1 && (lo..=hi).contains(&rel) {
            out.push(rel);
        } else if (lo..=hi).contains(&off) {
            // An absolute-RVA table (rarer on ARM32).
            out.push(off);
        } else {
            break;
        }
    }
    (out.len() >= 2).then_some(out)
}

pub fn recover_switch_tables(
    insns: &mut [Insn],
    addresses: &[u64],
    read_u32: &dyn Fn(u64) -> Option<u32>,
) -> usize {
    // FIRST index wins per address (one native insn = several IR insns that
    // share its address; a jump-table case target is an INSTRUCTION boundary,
    // so it must land on the group's first insn — see the same fix in
    // `cfg::Cfg::build`).
    let mut addr_to_idx: HashMap<u64, usize> = HashMap::with_capacity(addresses.len());
    for (i, &a) in addresses.iter().enumerate() {
        addr_to_idx.entry(a).or_insert(i);
    }
    let lo = addresses.first().copied().unwrap_or(0);
    let hi = addresses.last().copied().unwrap_or(u64::MAX);
    let mut recovered = 0usize;

    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::IndirectCall) {
            continue;
        }
        if crate::recovery::env_flags::flag("ILR_SWDBG") {
            eprintln!(
                "[sw] @{:#x} IndirectCall op0={:?}",
                insns[i].address,
                insns[i].operands.first()
            );
            for k in i.saturating_sub(14)..i {
                eprintln!(
                    "   [{k}] @{:#x}: op={:?} operands={:?}",
                    insns[k].address, insns[k].op, insns[k].operands
                );
            }
        }

        // Shape 1: LLVM anchored table (16/32-bit entries + ADR anchor).
        let recovery = anchor_table_targets(insns, i, lo, hi, read_u32);
        // Shape 2 (legacy): scale-4 word table whose base is immediate-
        // derived; entries are absolute RVAs or byte offsets from table base.
        // This shape has no guard correlation, so it proves no default arm.
        let recovery = recovery.or_else(|| {
            let mut load_idx = None;
            for k in i.saturating_sub(12)..i {
                let has_dest = matches!(
                    insns[k].dest(),
                    Some(Operand::Register(_)) | Some(Operand::Local(_))
                );
                if !has_dest || !matches!(insns[k].op, Op::Move) {
                    continue;
                }
                if let Some(Operand::Memory {
                    index: Some(_),
                    scale,
                    ..
                }) = insns[k].operands.get(1)
                {
                    if *scale == 4 {
                        load_idx = Some(k);
                    }
                }
            }
            let k = load_idx?;
            let Operand::Memory { base, addend, .. } = &insns[k].operands[1] else {
                return None;
            };
            let table_addr = match base.as_ref() {
                Operand::Immediate(v) => Some(*v as u64),
                _ => const_value(insns, k, base.as_ref()),
            }?;
            let table_addr = table_addr.wrapping_add(*addend as u64);
            let mut out = Vec::new();
            for case in 0..1024u64 {
                let Some(raw) = read_u32(table_addr.wrapping_add(case * 4)) else {
                    break;
                };
                let candidate = raw as u64;
                if (lo..=hi).contains(&candidate) {
                    out.push(candidate);
                } else if (lo..=hi).contains(&table_addr.wrapping_add(candidate)) {
                    out.push(table_addr.wrapping_add(candidate));
                } else {
                    break;
                }
            }
            (out.len() >= 2).then_some(out).map(|targets| TableRecovery {
                targets,
                default: None,
                disc: None,
            })
        });

        // Shape 3: ARM32 word-offset table (see `arm32_table_targets`).
        let recovery = recovery.or_else(|| {
            arm32_table_targets(insns, i, lo, hi, read_u32).map(|targets| TableRecovery {
                targets,
                default: None,
                disc: None,
            })
        });

        let Some(recovery) = recovery else {
            // A dispatch we could not resolve. The ARM32 lifter marks a
            // PC-dispatch (`add pc, …` / `mov pc, …`) with a trailing
            // `Immediate(-1)` sentinel, so it is distinguishable from a
            // `bx rN` VIRTUAL call (which is a real call and must stay). An
            // unresolved PC-dispatch renders a garbage `fnptr_…()` and flags
            // the whole method `suspect`; an exit is the honest best we can do —
            // the targets are unknown either way.
            let pc_dispatch = matches!(insns[i].operands.last(), Some(Operand::Immediate(-1)));
            if pc_dispatch {
                insns[i] = Insn::new(Op::Jump(usize::MAX), vec![], insns[i].address);
            }
            swgap_dump(insns, i, "no-shape");
            continue;
        };
        // Convert case targets to instruction indices. A target that does not
        // land on an instruction boundary is kept as `usize::MAX` (an exit);
        // the boundary check inside the recognizer already refuses to read past
        // the table, so this only catches an entry the reader accepted.
        let mut targets = Vec::with_capacity(recovery.targets.len());
        for rva in &recovery.targets {
            targets.push(addr_to_idx.get(rva).copied().unwrap_or(usize::MAX));
        }
        // Carry the two facts the case list cannot express. Before this, the
        // switch instruction was rebuilt with an EMPTY operand vector and the
        // renderer had to guess the discriminant (37 of 222 printed the literal
        // `switchValue`) and could not emit a `default:` at all (0 of 222).
        //
        // Layout is fixed so a reader does not have to sniff it:
        //   operands[0] = discriminant (`Unknown` when the table load named no
        //                 index, i.e. shapes 2/3)
        //   operands[1] = default target RVA, or `Immediate(-1)` for "none"
        // `ssa::use_slots` walks every operand, so the discriminant is renamed
        // to its SSA version for free and DCE keeps the value the switch reads.
        let disc = recovery.disc.unwrap_or(Operand::Unknown);
        let default = Operand::Immediate(recovery.default.map(|d| d as i64).unwrap_or(-1));
        insns[i] = Insn::new(Op::Switch(targets), vec![disc, default], insns[i].address);
        recovered += 1;
    }

    if recovered > 0 {
        dead_code_elimination(insns);
    }
    recovered
}

/// Regression tests for the private operand walkers in this module.
///
/// These live here rather than in `tests.rs` because the walkers are private:
/// the point of the test is the walker itself, not a pipeline effect, and
/// widening their visibility to reach them from a sibling module would be a
/// worse trade than a four-line test module.
#[cfg(test)]
mod walker_tests {
    use super::*;
    use crate::recovery::{FieldRef, ManagedType, PrimKind};

    fn ifield(base: Operand) -> Operand {
        Operand::InstanceField {
            base: Box::new(base),
            field: FieldRef {
                type_def_index: 1,
                field_index: 2,
                name: "_setting".into(),
                declaring_type_name: "Setting".into(),
                field_type: ManagedType::Primitive(PrimKind::Boolean),
                is_static: false,
            },
        }
    }

    /// `remove_init_guards` erases an init region only when nothing the region
    /// defines is read outside it, and it asks this predicate. A field base is
    /// a READ: without the `InstanceField` arm the predicate answered `false`
    /// for `v._field`, the region was judged unread and a live value was
    /// erased with it.
    #[test]
    fn a_field_base_reads_its_receiver() {
        assert!(guard_operand_reads(&ifield(Operand::Local(7)), 7));
        assert!(!guard_operand_reads(&ifield(Operand::Local(7)), 8));
        // nested: a field of a field of a memory load off the local
        let deep = ifield(Operand::Memory {
            base: Box::new(Operand::Local(7)),
            index: None,
            scale: 1,
            addend: 0x30,
            width: 0,
        });
        assert!(guard_operand_reads(&deep, 7));
    }

    /// Same contract for the DCE use-count walker, at nesting depth 2.
    #[test]
    fn a_nested_field_base_is_collected_once_per_mention() {
        let mut out = Vec::new();
        collect_locals(&ifield(ifield(Operand::Local(7))), &mut out);
        assert_eq!(out, vec![7]);
    }

    // ----------------------------------------------------------------------
    // S2/S3: the switch instruction must carry the facts the case list cannot.
    // ----------------------------------------------------------------------

    /// Build the S1 jump-table fixture (byte table at `TBL`, anchor `ANCHOR`,
    /// index copied into a fresh register) and return the instruction list.
    ///
    /// `guard` selects the shape in front of the table:
    ///   `Guard::Strict`  — `cmp x8,#3; b.hi 0x1050` (the LLVM bound + default)
    ///   `Guard::Lenient` — `cmp x8,#3` with no branch consuming it
    ///   `Guard::None`    — no compare at all
    fn switch_fixture(guard: Guard) -> Vec<Insn> {
        const TBL: u64 = 0x5000;
        const ANCHOR: u64 = 0x1020;
        let mut insns: Vec<Insn> = Vec::new();
        for k in 0..12u64 {
            insns.push(Insn::new(Op::Nop, vec![], 0x1000 + k * 4));
        }
        match guard {
            Guard::Strict => {
                insns.push(Insn::new(
                    Op::Compare(CompareKind::GreaterThan),
                    vec![Operand::Local(9), Operand::reg("x8"), Operand::imm(3)],
                    0x1030,
                ));
                // The branch a `cmp; b.hi` pair puts here. Its operand is an
                // RVA at this stage (step 1.5, before `Cfg::build`).
                insns.push(Insn::new(Op::CondJump(0x1050), vec![Operand::Local(9)], 0x1034));
            }
            Guard::Lenient => {
                insns.push(Insn::new(
                    Op::Compare(CompareKind::GreaterThan),
                    vec![Operand::Local(9), Operand::reg("x8"), Operand::imm(3)],
                    0x1030,
                ));
                // Not a branch: this compare proves a bound but no default arm.
                insns.push(Insn::new(Op::Nop, vec![], 0x1034));
            }
            Guard::None => {}
        }
        // The table is indexed by x7, a COPY of x8 — the S1 shape.
        insns.push(Insn::move_(Operand::reg("x7"), Operand::reg("x8"), 0x1038));
        insns.push(Insn::move_(Operand::reg("x9"), Operand::imm(TBL as i64), 0x103c));
        insns.push(Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::reg("x9"), Operand::reg("x9"), Operand::imm(0)],
            0x1040,
        ));
        insns.push(Insn::move_(Operand::reg("x10"), Operand::imm(ANCHOR as i64), 0x1044));
        insns.push(Insn::move_(
            Operand::reg("x11"),
            Operand::Memory {
                width: 1,
                base: Box::new(Operand::reg("x9")),
                index: Some(Box::new(Operand::reg("x7"))),
                scale: 1,
                addend: 0,
            },
            0x1048,
        ));
        insns.push(Insn::new(
            Op::Arith(ArithKind::Shl),
            vec![Operand::reg("xt0"), Operand::reg("x11"), Operand::imm(2)],
            0x104c,
        ));
        insns.push(Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::reg("x10"), Operand::reg("x10"), Operand::reg("xt0")],
            0x1050,
        ));
        insns.push(Insn::new(Op::IndirectCall, vec![Operand::reg("x10")], 0x1054));
        insns
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Guard {
        Strict,
        Lenient,
        None,
    }

    /// Run `recover_switch_tables` over the fixture and return the recovered
    /// switch instruction's operands.
    fn switch_operands(guard: Guard) -> (Vec<usize>, Vec<Operand>) {
        const TBL: u64 = 0x5000;
        let mut insns = switch_fixture(guard);
        let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
        // 4 real entries, then garbage that still lands in span: only the
        // guard can tell them apart.
        let bytes: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0xff, 0xff, 0xff];
        let read = move |a: u64| -> Option<u32> {
            if a < TBL || a + 4 > TBL + bytes.len() as u64 {
                return None;
            }
            let i = (a - TBL) as usize;
            Some(u32::from_le_bytes([
                bytes[i],
                bytes[i + 1],
                bytes[i + 2],
                bytes[i + 3],
            ]))
        };
        let n = recover_switch_tables(&mut insns, &addresses, &read);
        assert_eq!(n, 1, "the table must be recovered");
        insns
            .iter()
            .find_map(|i| match &i.op {
                Op::Switch(t) => Some((t.clone(), i.operands.clone())),
                _ => None,
            })
            .expect("switch")
    }

    /// The guard's taken target IS the default arm, and the case list cannot
    /// express it. Before the fix the switch was rebuilt with `vec![]`, so the
    /// renderer had no way to emit `default:` (measured: 0 of 222 switches).
    ///
    /// NEUTRALISATION: restore `vec![]` in `recover_switch_tables` and this
    /// assertion panics on the empty operand list.
    #[test]
    fn switch_default_target_reaches_the_ir() {
        let (targets, operands) = switch_operands(Guard::Strict);
        assert_eq!(targets.len(), 4, "the guard bounds the table at 4 entries");
        assert_eq!(
            operands.get(1),
            Some(&Operand::Immediate(0x1050)),
            "operands[1] must be the guard's taken-target RVA (the default arm)"
        );
    }

    /// The discriminant must be the CANONICAL table index, not the fresh
    /// register the load names: the guard and the load never share a register,
    /// and the renderer used to guess the value by scanning back for any
    /// `Compare` (37 of 222 switches printed the literal `switchValue`).
    ///
    /// NEUTRALISATION: store `index` instead of `want` and this asserts `x7`.
    #[test]
    fn switch_discriminant_is_the_canonical_index() {
        let (_, operands) = switch_operands(Guard::Strict);
        assert_eq!(
            operands.first(),
            Some(&Operand::reg("x8")),
            "operands[0] must be the canonical index (x8), not the load's copy (x7)"
        );
    }

    /// No guard means no default. The recognizer still recovers the table (the
    /// read stops when an entry leaves the method span), but it must not
    /// invent a default arm.
    #[test]
    fn no_guard_means_no_default() {
        let (targets, operands) = switch_operands(Guard::None);
        assert!(targets.len() >= 2, "the table is still recovered");
        assert_eq!(
            operands.get(1),
            Some(&Operand::Immediate(-1)),
            "Immediate(-1) is the \"no default\" sentinel"
        );
    }

    /// A bound compare that no branch consumes caps the table but proves
    /// nothing about a default arm: the lenient path must leave it unset.
    #[test]
    fn lenient_bound_is_not_a_default() {
        let (targets, operands) = switch_operands(Guard::Lenient);
        assert_eq!(targets.len(), 4, "the lenient bound still caps the table");
        assert_eq!(
            operands.get(1),
            Some(&Operand::Immediate(-1)),
            "a compare with no branch behind it is not evidence of a default"
        );
    }
}
