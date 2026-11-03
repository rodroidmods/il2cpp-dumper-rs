//! Basic-block construction over the flat IR listing.
//!
//! Branch targets arrive from the lifter as raw RVAs stored in the
//! `Jump`/`CondJump` index fields; this pass rewrites them into instruction
//! indices and partitions the listing into blocks (Cpp2IL's ISILControlFlowGraph).
//!
//! Successors are N-way (`Vec<Vec<usize>>`) with SEMANTIC ORDER:
//!   - CondJump: [taken, fallthrough]  — slot 0 is the taken edge
//!   - Switch:   every resolvable case target, in target order
//!   - Jump:     [target]
//!   - Return/Throw/IndirectCall: []
//!   - anything else: [fallthrough]
//! Consumers that only care about taken/fallthrough use the `SUCCESSOR_TAKEN`
//! / `SUCCESSOR_FALL` slot constants and `succ_fallthrough()`.

use super::ir::{Insn, Op};
use std::collections::HashMap;

/// Semantic slot of the taken edge of a conditional branch (slot 0).
pub const SUCCESSOR_TAKEN: usize = 0;
/// Semantic slot of the fallthrough edge of a conditional branch (slot 1).
pub const SUCCESSOR_FALL: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub start: usize,
    pub end: usize, // exclusive
}

pub struct Cfg {
    /// Blocks in address order.
    pub blocks: Vec<Block>,
    /// N-way successors per block, in semantic order (see module docs).
    pub successors: Vec<Vec<usize>>,
}

impl Cfg {
    /// Build blocks; `rva_to_index` maps branch-target RVAs → instruction idx.
    pub fn build(insns: &mut [Insn], addresses: &[u64]) -> Option<Cfg> {
        Self::build_with_landing_pads(insns, addresses, &[])
    }

    /// As [`Cfg::build`], plus `landing_pads`: RVA starts of EH landing pads.
    ///
    /// A pad is a handler entry, not ordinary flow. When its only incoming
    /// branch is the unreachable trampoline after a noreturn raise
    /// (`bl raise; b pad`), deleting that branch must not merge the pad into
    /// the preceding block: EH padding is addressed by RVA, and a pad that
    /// does not start a block cannot be a handler. Forcing the leader keeps
    /// the pad a block for the structurer's `region_of_block`/`handler_of`
    /// mapping while the CFG edge stays gone.
    pub fn build_with_landing_pads(
        insns: &mut [Insn],
        addresses: &[u64],
        landing_pads: &[u64],
    ) -> Option<Cfg> {
        // Debug: ILR_CFG_MISS=0xBASE dumps address/index mapping mismatches
        let dbg_miss = match std::env::var("ILR_CFG_MISS").as_deref() {
            Ok("1") => true,
            Ok(v) => insns.first().is_some_and(|i| {
                u64::from_str_radix(v.trim_start_matches("0x"), 16).ok() == Some(i.address)
            }),
            Err(_) => false,
        };
        // FIRST index wins for a duplicated address. One native instruction
        // expands to SEVERAL IR insns that all carry its address (Compare +
        // CondJump, Arith + Compare, …). A branch targets the INSTRUCTION, so
        // it must land on the FIRST of that group. `.collect()` kept the LAST,
        // which split the group: e.g. `TBZ x21,#0` → target resolved to the
        // CBZ's CondJump instead of its Compare, orphaning the Compare in a
        // block with no predecessors — SSA then never renamed its register
        // uses and they rendered as `__unknown` (the whole
        // `if (__unknown != 0)` corruption class), plus the dominance tree
        // lost that block's subtree.
        let mut rva_to_index: HashMap<u64, usize> = HashMap::with_capacity(addresses.len());
        for (i, &a) in addresses.iter().enumerate() {
            rva_to_index.entry(a).or_insert(i);
        }
        if dbg_miss {
            eprintln!("[cfg] insns={} addresses={}", insns.len(), addresses.len());
            for (i, insn) in insns.iter().enumerate() {
                let ai = addresses.get(i).copied();
                if ai != Some(insn.address) {
                    eprintln!(
                        "[cfg-mismatch] i{} insn_addr={:#x} arr_addr={:?} op={:?}",
                        i, insn.address, ai, insn.op
                    );
                }
            }
        }

        // leaders: first insn, branch targets, insns after terminators
        let mut leaders = vec![false; insns.len()];
        if !insns.is_empty() {
            leaders[0] = true;
        }
        for (i, _insn) in insns.iter().enumerate() {
            if i + 1 < insns.len() && insns[i].is_terminator() {
                leaders[i + 1] = true;
            }
        }

        // First pass: rewrite RVA targets to instruction indices. Targets
        // landing past the listing (jump tables, padding, tail-dup exits)
        // resolve to one-past-the-end, which successors treat as an exit —
        // a single unresolvable branch must NOT abort the whole method.
        // In-body addresses that fall ON or INSIDE an absorbed instruction
        // resolve to the first instruction at-or-after them instead; the
        // reasoning and the pairing requirement are on `resolve_target`.
        let end_sentinel = insns.len();
        for (i, insn) in insns.iter_mut().enumerate() {
            let op = std::mem::replace(&mut insn.op, Op::Move);
            insn.op = match op {
                Op::Jump(rva) => Op::Jump(
                    resolve_target(&rva_to_index, &addresses, rva, i).unwrap_or(end_sentinel),
                ),
                Op::CondJump(rva) => Op::CondJump(
                    resolve_target(&rva_to_index, &addresses, rva, i).unwrap_or(end_sentinel),
                ),
                other => other,
            };
        }

        for insn in insns.iter() {
            match &insn.op {
                Op::Jump(t) => {
                    if *t < leaders.len() {
                        leaders[*t] = true;
                    }
                }
                Op::CondJump(t) => {
                    if *t < leaders.len() {
                        leaders[*t] = true;
                    }
                }
                // Switch case targets must start blocks — otherwise the case
                // code is glued to unrelated blocks and successor edges can't
                // express the dispatch (see rebuild_cfg, same fix).
                Op::Switch(targets) => {
                    for &t in targets.iter() {
                        if t < leaders.len() {
                            leaders[t] = true;
                        }
                    }
                }
                _ => {}
            }
        }

        // Handler-entry pads are addressed by RVA, not by an edge. The
        // unwinder edge is not part of this CFG, and once the unreachable
        // trampoline branch after a noreturn raise is neutralised a pad can
        // have no branch target at all. Force the leader at the first
        // instruction at-or-after the pad (the same slide `resolve_target`
        // uses) so the pad remains a block the EH mapping can name.
        for &pad in landing_pads {
            if pad == 0 {
                continue;
            }
            let mut best: Option<usize> = None;
            let mut best_addr = u64::MAX;
            for (i, &a) in addresses.iter().enumerate() {
                if a >= pad && a < best_addr {
                    best_addr = a;
                    best = Some(i);
                }
            }
            if let Some(i) = best {
                leaders[i] = true;
            }
        }

        let mut blocks = Vec::new();
        let mut start = 0;
        for i in 1..=insns.len() {
            if i == insns.len() || leaders[i] {
                blocks.push(Block { start, end: i });
                start = i;
            }
        }

        // block lookup by instruction index
        let block_of: Vec<usize> = {
            let mut v = vec![usize::MAX; insns.len()];
            for (bi, b) in blocks.iter().enumerate() {
                for i in b.start..b.end {
                    v[i] = bi;
                }
            }
            v
        };

        let valid =
            |t: usize| -> Option<usize> { block_of.get(t).copied().filter(|&x| x != usize::MAX) };
        let fall_of = |b: &Block| -> Option<usize> {
            if b.end < insns.len() {
                valid(b.end)
            } else {
                None
            }
        };

        let mut successors = Vec::with_capacity(blocks.len());
        for b in &blocks {
            let last = &insns[b.end - 1];
            let succ: Vec<usize> = match &last.op {
                Op::Jump(t) => {
                    let mut succ: Vec<usize> = valid(*t).into_iter().collect();
                    // Dispatch-stub redirect: an unconditional `B` to an
                    // out-of-body stub (resolved to the end sentinel) whose
                    // payload — the real tail-merged call — sits immediately
                    // after the jump, still in-body. The stub branches back
                    // to it; without this edge the call block is reached only
                    // from the check blocks and the arguments computed before
                    // the jump (float math) never flow into the call — SSA
                    // sees them as dead and DCE deletes them.
                    if succ.is_empty() && *t >= end_sentinel {
                        if let Some(fb) = fall_of(b) {
                            let payload = &insns[blocks[fb].start..blocks[fb].end];
                            if payload
                                .iter()
                                .any(|i| matches!(i.op, Op::Call | Op::IndirectCall))
                            {
                                succ.push(fb);
                            }
                        }
                    }
                    succ
                }
                Op::CondJump(t) => {
                    // semantic order: taken, fallthrough
                    [valid(*t), fall_of(b)].into_iter().flatten().collect()
                }
                Op::Switch(targets) => {
                    // EVERY resolvable case target, in target order. Duplicate
                    // block targets (multiple cases → same block) collapse to
                    // one edge — graph-wise identical, and dominance/phi
                    // insertion treat parallel edges as one anyway.
                    let mut seen = Vec::new();
                    for &t in targets {
                        if let Some(blk) = valid(t) {
                            if !seen.contains(&blk) {
                                seen.push(blk);
                            }
                        }
                    }
                    seen
                }
                Op::Return | Op::Throw | Op::IndirectCall => Vec::new(),
                _ => fall_of(b).into_iter().collect(),
            };
            successors.push(succ);
        }

        Some(Cfg { blocks, successors })
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

fn resolve_target(
    map: &HashMap<u64, usize>,
    addresses: &[u64],
    rva: usize,
    branch_idx: usize,
) -> Option<usize> {
    // The lifter stores raw RVAs in Jump/CondJump. They may exceed the
    // instruction count, so distinguish "already an index" (small) from RVA.
    // Heuristic: values < insns len that exactly match a known instruction
    // address are ambiguous only if the method is > 4 GB — not possible.
    if let Some(&idx) = map.get(&(rva as u64)) {
        return Some(idx);
    }
    // An instruction INDEX, not an RVA. `Cfg::build` rewrites
    // `Jump`/`CondJump` operands from RVAs to indices, so any later pass over
    // the same IR sees indices. A real method's RVA is never below its own
    // instruction count (the smallest body in the iOS corpus starts thousands
    // of bytes into __TEXT), so for a lifted body this is unambiguous — and it
    // is what makes `Cfg::build` IDEMPOTENT.
    //
    // Without this arm the second pass resolved every operand to `None` and
    // rewrote every branch in the method to the exit sentinel: a single re-run
    // silently deleted the method's entire control flow (all 67 conditions of
    // `UpdateControls` @0x117674 became straight-line code). Synthetic IR built
    // directly in tests is identity-mapped (`addresses[i] == i`), which this
    // arm covers as well.
    if rva < addresses.len() {
        return Some(rva);
    }
    // A genuine body address that no IR instruction STARTS at. That case is
    // routine, not corruption: on x86 the flag-setting instruction is absorbed
    // into the branch that consumes its flags (`cmp`/`test` set `PendingCmp`,
    // and the Compare + CondJump are emitted at the JCC's address), so the
    // absorbed instruction's own address never gets an IR entry — and any
    // OTHER branch that targets it has nowhere exact to land. The same is true
    // on ARM64 for any instruction a pass folded away.
    //
    // Resolve to the first instruction AT OR AFTER the address. The absorbed
    // instruction is a PREFIX of the one that replaced it, so the successor
    // that jumps there must rejoin the instruction that now stands for it —
    // that is what `at-or-after` means here. Sending the edge to the exit
    // sentinel instead DELETES a real in-body edge: measured on v24.5, the
    // x86 static-init guard (`cmp [guard], 0` + `jne <skip>`) had its taken
    // edge read as a return, which is also why the oracle reported 8,452
    // branch-target mismatches.
    //
    // This is deliberately the FIRST half of one change. On its own it is
    // fidelity-correct but a readability regression, because the structurer
    // renders the restored diamond as a labelled jump instead of the guard it
    // is: alone it moved `goto L` 52,031 → 61,194 (+17.6%) for a 1,089-finding
    // oracle gain. It ships together with `csharp::fold_branch_over_join`,
    // which collapses `if (c) { goto L; } <body> L:` back into
    // `if (!c) { <body> }` — and THAT pairing is the accepted form (see the
    // measurement in docs/IL_RECOVERY.md). Keep them together: reverting one
    // without the other reproduces the trade.
    //
    // Bounds: only an address strictly inside the body is absorbable. Below the
    // lowest address (a backward edge out of the method) or above the highest
    // (jump tables, padding, tail-dup exits) stays an exit edge — that is the
    // caller's sentinel and must not be redirected to the first/last insn.
    //
    // "Inside the body" must be judged on the ADDRESS RANGE, not on the first
    // and last LISTING elements: the structurer APPENDS an out-of-line cold /
    // exception region, so the listing is not address-ordered and its last
    // element can sit below an earlier one. Using `.last()` as the upper bound
    // silently refused every in-body target above the appended region. Measured
    // on v24.5 `0x14eb20`: the guard is `cmp byte [rip+…], 0` … `jne 0x14eb4a`
    // … `mov byte [rip+…], 1`, and `0x14eb4a` is a real 5-byte
    // `cmp qword [rbx+0x30], 0` that the lifter folds into the `jne` at
    // `0x14eb57`. It is above the appended region's address, so the bound test
    // rejected it and the guard edge went to the exit sentinel — the method
    // rendered as an UNGUARDED init block, which is wrong logic. This bound was
    // the reason the at-or-after repair looked like it recovered only four
    // methods.
    let want = rva as u64;
    let first = addresses.iter().copied().min()?;
    let last = addresses.iter().copied().max()?;
    if want <= first || want > last {
        return None;
    }
    // Order-independent: `addresses` is ascending for a lifted body, but the
    // sentinel-free contract here is "smallest address ≥ target", which does
    // not depend on the listing being sorted.
    let mut best: Option<usize> = None;
    let mut best_addr = u64::MAX;
    for (i, &a) in addresses.iter().enumerate() {
        if a >= want && a < best_addr {
            best_addr = a;
            best = Some(i);
        }
    }
    // A slide that lands on the BRANCH ITSELF is not a rejoin — it is a
    // manufactured SELF-LOOP, and `check_cfg` reads it as an unrecovered trap
    // (`block N cannot reach an exit`) that the binary never had.
    //
    // The at-or-after rule above is justified by an absorbed INSTRUCTION
    // PREFIX: the branch targets a real instruction that the lifter folded
    // into its consumer, so the successor must rejoin that consumer. When the
    // nearest instruction at-or-after the target IS the branch, there is no
    // such prefix — every byte between the target and the branch belongs to no
    // decoded instruction, which is what the padding/tail bytes look like when
    // a measured body runs past the method's real end. Measured on v24.5
    // `0x4afba0`: past the final `ret` the listing holds the null-check throw
    // stub (`call 0x11c930`) and then `jmp 0x4afccd`, whose target sits INSIDE
    // that call's displacement; the slide mapped it to the `jmp` itself and the
    // renderer emitted `do { /* spin wait (native self-loop) */ } while (true);`
    // on the null-check failure path.
    //
    // An EXACT hit is unaffected: a real `L: jne L` self-loop maps through
    // `map`, landing on the first instruction of that address's group — which
    // cannot be the branch when the group has a Compare before it, and when it
    // is, `want == addresses[branch_idx]` means no slide happened.
    if best == Some(branch_idx) {
        return None;
    }
    best
}
