//! Minimal SSA construction + elimination (port of Cpp2IL
//! `Analysis/SsaForm.cs`, Cytron-style renaming).
//!
//! Phis live OUT-OF-BAND ([`PhiRecord`]) while renaming runs so instruction
//! indices never shift mid-analysis. [`materialize`] splices them into the
//! listing (one renumbering pass) so downstream passes see ordinary `Op::Phi`
//! instructions; [`eliminate`] lowers surviving phis into Moves on predecessor
//! edges (second pass).
//!
//! Parallel-copy safety: every phi destination is a *fresh* local version no
//! other instruction references, so sequential per-edge move insertion cannot
//! create read-after-write hazards between sibling phis.

use super::cfg::{Block, Cfg};
use super::dominator::DominatorInfo;
use super::ir::{Insn, Op, Operand};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

type Slot = String;

/// One structural edit pass: insert `insertions` before given indices and drop
/// `removals`, remapping branch targets through the renumbering.
fn transform(
    insns: &[Insn],
    mut insertions: BTreeMap<usize, Vec<Insn>>,
    removals: &BTreeSet<usize>,
) -> Vec<Insn> {
    let extra: usize = insertions.values().map(Vec::len).sum();
    let mut out: Vec<Insn> = Vec::with_capacity(insns.len() + extra);
    let mut map = vec![usize::MAX; insns.len()];
    // for REMOVED indices: where the next surviving instruction landed, so
    // a branch that targeted a removed insn still resolves to real code
    // (phi elimination removes block-start phis that branches point at)
    let mut removed_map = vec![usize::MAX; insns.len()];
    // reverse so `.last()` is the smallest scheduled point
    let mut pending: Vec<(usize, Vec<Insn>)> = insertions
        .iter_mut()
        .map(|(k, v)| (*k, std::mem::take(v)))
        .collect();
    pending.sort_by(|a, b| b.0.cmp(&a.0));

    for (i, insn) in insns.iter().enumerate() {
        while let Some((at, _)) = pending.last() {
            if *at <= i {
                let (_, batch) = pending.pop().unwrap();
                out.extend(batch);
            } else {
                break;
            }
        }
        if removals.contains(&i) {
            removed_map[i] = out.len();
            continue;
        }
        map[i] = out.len();
        out.push(insn.clone());
    }
    while let Some((_, batch)) = pending.pop() {
        out.extend(batch);
    }
    // Resolve an old index → new position. A target at/after the OLD listing
    // end is the EXIT SENTINEL that `Cfg::build` installs for branches that
    // leave the body (tail-merged epilogues, `b` into a shared tail). It MUST
    // be re-anchored to the NEW listing end: returning it unchanged is how a
    // manufactured SELF-LOOP was born. Inserting one insn (a Phi, an edge
    // move) before the branch shifts the branch from i to i+1 while the stale
    // sentinel — which was exactly `old_len` == i+1 — suddenly resolves to
    // the branch's OWN index. The structurer then sees `B → B`, calls it a
    // natural loop, and renders the whole method as
    // `do { /* spin wait (native self-loop) */ } while (true);` — ~8k methods
    // binary-wide, and the source of the surviving "branch target mismatch"
    // findings in the disassembler differential.
    let new_len = out.len();
    let remap = |t: usize| -> usize {
        if t >= insns.len() {
            return new_len;
        }
        if let Some(&n) = map.get(t) {
            if n != usize::MAX {
                return n;
            }
        }
        if let Some(&n) = removed_map.get(t) {
            if n != usize::MAX {
                return n;
            }
        }
        t
    };
    for insn in &mut out {
        match &mut insn.op {
            Op::Jump(t) => {
                *t = remap(*t);
            }
            Op::CondJump(t) => {
                *t = remap(*t);
            }
            // Switch case targets are instruction INDICES exactly like a jump
            // target, and `transform` ins/dels instructions (phi copies in,
            // phis out). Leaving them un-remapped made every recovered `switch`
            // point at pre-SSA positions after this pass, so the case bodies
            // were unreachable to the structurer: on 0x1cfbc6c case 0 resolved
            // to insn 91 instead of 118 and 99 of 125 blocks were never
            // walked, taking every condition and call inside them with them.
            Op::Switch(targets) => {
                for t in targets.iter_mut() {
                    *t = remap(*t);
                }
            }
            _ => {}
        }
    }
    out
}

/// Recompute blocks/successors for an already-index-targeted listing.
pub fn rebuild_cfg(insns: &[Insn]) -> Cfg {
    let n = insns.len();
    let mut leaders = vec![false; n];
    if n > 0 {
        leaders[0] = true;
    }
    for insn in insns.iter() {
        match &insn.op {
            Op::Jump(t) | Op::CondJump(t) => {
                if *t < n {
                    leaders[*t] = true;
                }
            }
            // Switch case targets are branches too: every one of them must
            // start a block, otherwise case targets land INSIDE another block
            // and the block successors (below) can't express the dispatch —
            // which silently broke dominance, SSA placement and loop
            // detection for every switch-containing method.
            Op::Switch(targets) => {
                for &t in targets.iter() {
                    if t < n {
                        leaders[t] = true;
                    }
                }
            }
            _ => {}
        }
    }
    for (i, insn) in insns.iter().enumerate() {
        if i + 1 < n && insn.is_terminator() {
            leaders[i + 1] = true;
        }
    }

    let mut blocks: Vec<Block> = Vec::new();
    let mut start = 0;
    for i in 1..=n {
        if i == n || leaders[i] {
            blocks.push(Block { start, end: i });
            start = i;
        }
    }

    let block_of: Vec<usize> = {
        let mut v = vec![usize::MAX; n];
        for (bi, b) in blocks.iter().enumerate() {
            for i in b.start..b.end {
                v[i] = bi;
            }
        }
        v
    };

    let successors = blocks
        .iter()
        .map(|b| {
            let last = &insns[b.end - 1];
            let tgt = |t: usize| -> Option<usize> {
                block_of.get(t).copied().filter(|&x| x != usize::MAX)
            };
            let fall = || -> Option<usize> {
                if b.end < n {
                    Some(block_of[b.end])
                } else {
                    None
                }
            };
            match &last.op {
                Op::Jump(t) => tgt(*t).into_iter().collect(),
                Op::CondJump(t) => [tgt(*t), fall()].into_iter().flatten().collect(),
                // A switch block's successors are ALL case targets, in target
                // order. Carrying the full fan-out keeps dominance correct
                // (without any edge the switch block appeared disconnected
                // and every dominated block lost its idom) AND gives SSA/phi
                // placement the real graph so each case arm gets its own
                // phi edges.
                Op::Switch(targets) => {
                    let mut seen: Vec<usize> = Vec::new();
                    for &t in targets {
                        if let Some(blk) = tgt(t) {
                            if !seen.contains(&blk) {
                                seen.push(blk);
                            }
                        }
                    }
                    seen
                }
                Op::Return | Op::Throw | Op::IndirectCall => Vec::new(),
                _ => fall().into_iter().collect(),
            }
        })
        .collect();

    Cfg { blocks, successors }
}

pub fn predecessors(cfg: &Cfg) -> Vec<Vec<usize>> {
    let n = cfg.blocks.len();
    let mut preds = vec![Vec::new(); n];
    for i in 0..n {
        for &s in &cfg.successors[i] {
            if !preds[s].contains(&i) {
                preds[s].push(i);
            }
        }
    }
    preds
}

/// Sentinel key reserving the global SSA version counter inside the
/// per-slot `counters` map. Real register names can never collide with it.
const GLOBAL_COUNTER_SLOT: &str = "#global";

fn def_slot(insn: &Insn) -> Option<Slot> {
    match insn.dest()? {
        Operand::Register(name) => Some(name.clone()),
        _ => None,
    }
}

fn use_slots(insn: &Insn) -> Vec<Slot> {
    let mut out = Vec::new();
    fn walk(op: &Operand, out: &mut Vec<Slot>) {
        // Exhaustive: this runs on the pre-typing register IR, where
        // `InstanceField` cannot exist yet, but a catch-all would silently
        // stop descending into whatever sub-operand a future variant adds.
        match op {
            Operand::Register(n) => out.push(n.clone()),
            Operand::Memory { base, index, .. } => {
                walk(base, out);
                if let Some(i) = index {
                    walk(i, out);
                }
            }
            Operand::AddressOf(inner) | Operand::ArrayLength(inner) => walk(inner, out),
            Operand::ArrayAccess { array, index } => {
                walk(array, out);
                walk(index, out);
            }
            Operand::InstanceField { base, .. } => walk(base, out),
            Operand::Local(_)
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
    for (i, op) in insn.operands.iter().enumerate() {
        // ONE source of truth for the write slot (ir.rs dest_position) so
        // Convert/MulAcc/Clz results are never mistaken for reads.
        //
        // Only a *pure register* destination (`mov x0, ...`) is a def. A
        // MEMORY destination (`str x0, [x19, #8]`) is a STORE: its base and
        // index registers are real reads and must be renamed too. Skipping
        // operand#0 wholesale left ~175k raw `Register` operands in the
        // post-SSA listing — the `this`/frame pointer of every store, which
        // the renderer can only print as `__unknown`.
        let is_pure_def =
            Insn::dest_position(&insn.op) == Some(i) && matches!(op, Operand::Register(_));
        if !is_pure_def {
            walk(op, &mut out);
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct PhiRecord {
    pub block: usize,
    pub slot: Slot,
    pub dest_version: u32,
    pub preds: Vec<usize>,
    pub args: Vec<Option<u32>>,
}

pub struct SsaOutput {
    /// renamed listing (Register operands → versioned Locals)
    pub insns: Vec<Insn>,
    pub phis: Vec<PhiRecord>,
    /// Local version id → originating slot name
    pub version_slot: HashMap<u32, Slot>,
    /// Versions defined by a CALL's ABI clobber (no listing instruction).
    pub call_defined: HashSet<u32>,
}

fn op_is_call(op: &Op) -> bool {
    matches!(
        op,
        Op::Call | Op::CallVoid | Op::IndirectCall | Op::ObjcMsgSend { .. }
    )
}

pub fn build(
    insns: &[Insn],
    cfg: &Cfg,
    dom: &DominatorInfo,
    clobber_slots: &[Slot],
) -> SsaOutput {
    build_with_handler_entries(insns, cfg, dom, clobber_slots, &HashSet::new())
}

/// As [`build`], with EH handler-entry blocks treated as SSA KILL POINTS.
///
/// A landing pad is entered by the unwinder, which supplies the exception
/// object in the first argument register and preserves NOTHING else. The CFG
/// cannot express that edge: it only carries synchronous edges, and the
/// compiler leaves an unreachable `b <pad>` after a noreturn raise call,
/// which manufactures a false predecessor. Renaming the pad along that edge
/// bound `x0` to whatever the loop body last wrote there, and the handler
/// rendered the body's call receiver as the exception:
/// `IL2CPP_IsInst((&enumerator).MoveNext(), MethodInfo(...))`.
///
/// At a handler entry every slot therefore starts from its ENTRY version
/// (the value the method was called with), never from the dominator path,
/// and no phi may be placed there. Slots the handler itself defines before
/// use keep their in-block definition.
pub fn build_with_handler_entries(
    insns: &[Insn],
    cfg: &Cfg,
    dom: &DominatorInfo,
    clobber_slots: &[Slot],
    handler_entries: &HashSet<usize>,
) -> SsaOutput {
    let n_blocks = cfg.blocks.len();
    let preds = predecessors(cfg);

    let block_of: Vec<usize> = {
        let mut v = vec![usize::MAX; insns.len()];
        for (bi, b) in cfg.blocks.iter().enumerate() {
            for i in b.start..b.end {
                v[i] = bi;
            }
        }
        v
    };

    let mut def_sites: HashMap<Slot, BTreeSet<usize>> = HashMap::new();
    for (i, insn) in insns.iter().enumerate() {
        if let Some(s) = def_slot(insn) {
            def_sites.entry(s).or_default().insert(i);
        }
        // A call is an ABI definition of the caller-saved FP return
        // registers. Modeling it is a CORRECTNESS requirement: without it a
        // read after the call names the pre-call SSA version, and the merge
        // rules can fold two different returned components to one value —
        // measured as `(z - z)` in real Vector3 distance math.
        if op_is_call(&insn.op) {
            for slot in clobber_slots {
                def_sites.entry(slot.clone()).or_default().insert(i);
            }
        }
    }
    let escaped: HashSet<Slot> = insns
        .iter()
        .flat_map(|i| i.operands.iter())
        .filter_map(|op| match op {
            Operand::AddressOf(inner) => match inner.as_ref() {
                Operand::Register(n) => Some(n.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();

    // ---- PRUNED SSA: liveness-gated iterated dominance frontiers ------------
    // Minimal SSA explodes on compiler-shaped code (every ABI register is
    // multiply-defined); Cpp2IL prunes with liveness, so do the same:
    // place a phi for `slot` at join f only when `slot` is live-in at f.
    let live_in = compute_live_in(insns, cfg);

    let mut phi_sites: HashMap<Slot, BTreeSet<usize>> = HashMap::new();
    let ssa_dbg = crate::recovery::env_flags::flag("ILR_SSA_DBG")
        && insns.first().is_some_and(|i| {
            let a = i.address;
            a >= 0x2db38 && a < 0x2dc80
        });
    let mut candidates: Vec<Slot> = def_sites
        .keys()
        .filter(|s| def_sites[*s].len() > 1 || escaped.contains(*s))
        .cloned()
        .collect();

    // SINGLE-DEF slots whose definition does NOT dominate every block where
    // the slot is live-in are joins too: the merge block reads the value
    // along some path that skips the definition (or an entry value flows in).
    // The old `len() > 1` filter skipped these, so a call in a merge block
    // got the ENTRY version of `vs0–vs7` instead of the version computed in
    // a sibling branch — the whole FP computation then looked dead and DCE
    // deleted it (the "missing Vector3 math").
    {
        let live_in = &live_in;
        let dom = dom;
        for (slot, sites) in def_sites.iter() {
            if sites.len() != 1 || candidates.contains(slot) {
                continue;
            }
            let def_block = block_of[*sites.iter().next().unwrap()];
            let need = live_in
                .iter()
                .enumerate()
                .any(|(b, set)| set.contains(slot) && !dom.dominates(def_block, b));
            if ssa_dbg {
                eprintln!(
                    "[ssa1def] slot={} def_blk={} need={} livein_blocks={:?}",
                    slot,
                    def_block,
                    need,
                    live_in
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| s.contains(slot))
                        .map(|(b, _)| b)
                        .collect::<Vec<_>>()
                );
            }
            if need {
                candidates.push(slot.clone());
                // The frontier walk below may still place NOTHING for this
                // slot: dominance frontiers only mark joins reachable via
                // frontiers of def blocks, and when the def block dominates
                // every CFG node the walk sees (empty frontiers), the join
                // where the value is actually consumed never gets a phi —
                // the consumer then reads the stale ENTRY version. Place
                // phis directly at every live-in join the def does not
                // dominate (the Cytron set, recovered from the liveness
                // side).
                let mut extra: Vec<usize> = live_in
                    .iter()
                    .enumerate()
                    .filter(|(b, set)| {
                        set.contains(slot)
                            && preds[*b].len() >= 2
                            && !dom.dominates(def_block, *b)
                            // A handler entry kills every slot: a phi here
                            // would merge values that never reach the pad.
                            && !handler_entries.contains(b)
                    })
                    .map(|(b, _)| b)
                    .collect();
                if !extra.is_empty() {
                    if ssa_dbg {
                        eprintln!("[ssa1def-join] slot={} extra_phis={:?}", slot, extra);
                    }
                    let entry = phi_sites.entry(slot.clone()).or_default();
                    extra.drain(..).for_each(|b| {
                        entry.insert(b);
                    });
                }
            }
        }
    }

    // per-slot live sets for the worklist below (slot → set of blocks where
    // the slot may be live at entry); recomputed lazily from live_in masks
    for slot in candidates {
        if ssa_dbg {
            eprintln!("[ssa-cand] slot={} sites={:?}", slot, def_sites[&slot]);
        }
        // quick reject: never live anywhere → no phis
        let mut work: VecDeque<usize> = def_sites[&slot].iter().map(|&i| block_of[i]).collect();
        let mut queued: BTreeSet<usize> = work.iter().copied().collect();
        while let Some(b) = work.pop_front() {
            queued.remove(&b);
            if ssa_dbg {
                eprintln!(
                    "[ssa-work] slot={} pop_blk={} frontiers={:?}",
                    slot, b, dom.frontiers[b]
                );
            }
            for &f in &dom.frontiers[b] {
                // prune: skip joins where the slot is not live-in
                if !live_in.get(f).is_some_and(|set| set.contains(&slot)) {
                    continue;
                }
                // handler entries are kill points, never merge points
                if handler_entries.contains(&f) {
                    continue;
                }
                // placing a phi makes f a def site → iterate its frontier too
                if phi_sites.entry(slot.clone()).or_default().insert(f) && queued.insert(f) {
                    work.push_back(f);
                }
            }
        }
    }

    // Belt and braces: the single-def path above can enqueue a handler block
    // outside the frontier walk, so scrub the finished sets too.
    if !handler_entries.is_empty() {
        for set in phi_sites.values_mut() {
            set.retain(|b| !handler_entries.contains(b));
        }
        phi_sites.retain(|_, set| !set.is_empty());
    }

    if ssa_dbg {
        for (slot, blocks) in &phi_sites {
            eprintln!("[ssa-phi] slot={} sites={:?}", slot, blocks);
        }
    }

    // ---- renaming ------------------------------------------------------------
    let mut counters: HashMap<Slot, u32> = HashMap::new();
    let mut versions: HashMap<Slot, Vec<u32>> = HashMap::new();
    let mut version_slot: HashMap<u32, Slot> = HashMap::new();
    let mut call_defined: HashSet<u32> = HashSet::new();

    // GLOBAL version counter: version ids must be unique across ALL slots.
    // The per-slot `counters` map handed the SAME id to different slots (each
    // slot's first version was 0, second 1, …), so `Local(3)` meant x0's v3
    // at one site and x1's v3 at another — locals::create collapsed them into
    // one table entry, reaching-defs linked wrong defs to uses, and receivers
    // rendered as bare `v0` instead of `this`. One monotone global counter
    // keeps every (slot → version) mapping injective.
    fn next_global_version(counters: &mut HashMap<Slot, u32>) -> &mut u32 {
        counters.entry(GLOBAL_COUNTER_SLOT.to_string()).or_insert(0)
    }

    // ENTRY SEEDING: registers read before any def hold the method's incoming
    // ABI values (this pointer, parameters). Without virtual entry versions,
    // such reads rename to Unknown — the source of pervasive `default`
    // receivers/args. Pre-seed one entry version per observed slot; these
    // stay on the stacks for the whole function (never popped).
    {
        let mut all_slots: BTreeSet<Slot> = def_sites.keys().cloned().collect();
        for insn in insns.iter() {
            for slot in use_slots(insn) {
                all_slots.insert(slot);
            }
        }
        for slot in all_slots {
            let c = next_global_version(&mut counters);
            let v = *c;
            *c += 1;
            version_slot.insert(v, slot.clone());
            versions.entry(slot).or_default().push(v);
        }
    }

    let fresh = |counters: &mut HashMap<Slot, u32>,
                 version_slot: &mut HashMap<u32, Slot>,
                 slot: &Slot|
     -> u32 {
        let c = next_global_version(counters);
        let v = *c;
        *c += 1;
        version_slot.insert(v, slot.clone());
        v
    };

    // ---- ROTATED loops: the entry block is its own latch --------------------
    //
    // The C# compiler emits `do { body } while (test)` as `body; test; bne
    // body`, so the method's first block is often the loop header AND its own
    // latch. A phi is the wrong tool there: block 0's only incoming CFG edge
    // is the back edge, so a phi at block 0 has no argument for the CALLER's
    // edge — the loop-carried registers end up reading their entry version on
    // every iteration (`while (this._innerException != 0)` with an empty body).
    //
    // A phi is also unnecessary. Because a block is straight-line, every
    // instruction in it runs on every iteration, so the value a loop-carried
    // slot has at the top of the loop is *exactly* the last version defined in
    // the block. Rotate it: drop the phi (uses before the def then keep the
    // pre-seeded ENTRY version, which already holds the caller's value) and
    // copy the block's last definition of the slot back onto that entry
    // version in front of the terminator. No new block, no index churn, and
    // no instruction that a `do { … } while (…)` body would re-enter by
    // mistake.
    let mut rotated: Vec<(Slot, u32)> = Vec::new();
    if n_blocks > 0 && cfg.successors.first().is_some_and(|s| s.contains(&0)) {
        let pinned: Vec<Slot> = phi_sites
            .iter()
            .filter(|(_, blocks)| blocks.contains(&0))
            .map(|(slot, _)| slot.clone())
            .collect();
        for slot in pinned {
            let Some(ev) = versions.get(&slot).and_then(|v| v.first().copied()) else {
                continue;
            };
            let mut emptied = false;
            if let Some(set) = phi_sites.get_mut(&slot) {
                set.remove(&0);
                emptied = set.is_empty();
            }
            if emptied {
                phi_sites.remove(&slot);
            }
            rotated.push((slot, ev));
        }
    }

    let mut phis: Vec<PhiRecord> = Vec::new();
    for (slot, blocks) in &phi_sites {
        for &b in blocks {
            phis.push(PhiRecord {
                block: b,
                slot: slot.clone(),
                dest_version: u32::MAX,
                preds: preds[b].clone(),
                args: vec![None; preds[b].len()],
            });
        }
    }
    phis.sort_by(|a, b| a.block.cmp(&b.block).then(a.slot.cmp(&b.slot)));
    let phis_by_block: HashMap<usize, Vec<usize>> =
        phis.iter()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (i, p)| {
                acc.entry(p.block).or_default().push(i);
                acc
            });

    let mut use_rewrites: HashMap<usize, Vec<(Slot, Operand)>> = HashMap::new();
    let mut dest_rewrite: HashMap<usize, u32> = HashMap::new();
    let mut call_ret_rewrite: HashMap<usize, u32> = HashMap::new();

    // Pre-assign every phi's destination version BEFORE the renaming DFS.
    // The merge block of a tail-merged call is usually a sibling of the
    // blocks feeding it (idom = method entry), so the DFS visits it AFTER
    // all its predecessors — their edge fills then hit an unassigned dest
    // and silently produce `Unknown` phi args, which DCE reads as "value
    // unused" and deletes the whole computation feeding that edge (the
    // missing FP math). With dests pre-assigned, every edge fills.
    let mut phi_dests: HashMap<usize, Vec<(Slot, u32)>> = HashMap::new();
    for pid in 0..phis.len() {
        let v = fresh(&mut counters, &mut version_slot, &phis[pid].slot);
        phis[pid].dest_version = v;
        phi_dests
            .entry(phis[pid].block)
            .or_default()
            .push((phis[pid].slot.clone(), v));
    }

    struct Frame {
        block: usize,
        child_i: usize,
        pushed: Vec<Slot>,
    }

    let mut visited = vec![false; n_blocks];
    if n_blocks > 0 {
        let mut stack = vec![Frame {
            block: 0,
            child_i: 0,
            pushed: Vec::new(),
        }];

        while let Some(frame) = stack.last_mut() {
            let block = frame.block;

            if !visited[block] {
                visited[block] = true;

                // 1. push this block's (pre-assigned) phi destinations onto
                // the version stacks so body uses after the phis resolve to
                // the merged value.
                if let Some(dests) = phi_dests.get(&block) {
                    for (slot, v) in dests {
                        versions.entry(slot.clone()).or_default().push(*v);
                        frame.pushed.push(slot.clone());
                    }
                }

                // 2. body uses + defs
                let b = &cfg.blocks[block];
                let handler_entry = handler_entries.contains(&block);
                let mut handler_defs: HashMap<Slot, u32> = HashMap::new();
                for i in b.start..b.end {
                    for slot in use_slots(&insns[i]) {
                        let val = if handler_entry {
                            // The pad's entry state is its own: unwinding
                            // preserves no register. Values defined earlier in
                            // THIS block are real; everything else is the
                            // method-entry value, never a dominator-path def.
                            match handler_defs.get(&slot) {
                                Some(&v) => Operand::Local(v),
                                None => versions
                                    .get(&slot)
                                    .and_then(|v| v.first().copied())
                                    .map(Operand::Local)
                                    .unwrap_or(Operand::Unknown),
                            }
                        } else {
                            match versions.get(&slot).and_then(|v| v.last()) {
                                Some(&v) => Operand::Local(v),
                                None => Operand::Unknown,
                            }
                        };
                        use_rewrites.entry(i).or_default().push((slot, val));
                    }
                    if let Some(slot) = def_slot(&insns[i]) {
                        let v = fresh(&mut counters, &mut version_slot, &slot);
                        versions.entry(slot.clone()).or_default().push(v);
                        frame.pushed.push(slot.clone());
                        if handler_entry {
                            handler_defs.insert(slot, v);
                        }
                        if insns[i].op == Op::Call {
                            call_ret_rewrite.insert(i, v);
                        } else {
                            dest_rewrite.insert(i, v);
                        }
                    }
                    if op_is_call(&insns[i].op) {
                        for slot in clobber_slots {
                            let v = fresh(&mut counters, &mut version_slot, slot);
                            versions.entry(slot.clone()).or_default().push(v);
                            frame.pushed.push(slot.clone());
                            call_defined.insert(v);
                        }
                    }
                }
            }

            // 3. fill successor-edge phi args with current exit versions —
            // over the FULL n-way successor set (switch fan-out included):
            // a case arm that lands on the merge block previously contributed
            // no phi arg, so the merge's phi read an unrenamed value.
            for succ in cfg.successors[block].iter().copied() {
                if let Some(ids) = phis_by_block.get(&succ) {
                    for &pid in ids {
                        let dest_set = phis[pid].dest_version != u32::MAX;
                        if !dest_set {
                            continue;
                        }
                        let ver = versions
                            .get(&phis[pid].slot)
                            .and_then(|v| v.last().copied());
                        if let Some(edge) = phis[pid].preds.iter().position(|&p| p == block) {
                            if crate::recovery::env_flags::flag("ILR_SSA_DBG") {
                                eprintln!(
                                    "[ssa-fill] B{} -> B{} phi slot={:?} edge={} dest_set={} ver={:?}",
                                    block, succ, phis[pid].slot, edge, dest_set, ver
                                );
                            }
                            phis[pid].args[edge] = ver;
                        }
                    }
                }
            }

            // 4. recurse / unwind over the dominator tree in ORIGINAL child
            // order. dom_children is a tree, so each child is pushed exactly
            // once — no visited filtering needed. (The old code re-filtered
            // by !visited and indexed the SHRUNKEN list with the stale
            // child_i, skipping entire subtrees: loop bodies were never
            // SSA-renamed whenever a sibling subtree came first.)
            let children: &[usize] = &dom.dom_children[block];
            if frame.child_i < children.len() {
                let child = children[frame.child_i];
                frame.child_i += 1;
                stack.push(Frame {
                    block: child,
                    child_i: 0,
                    pushed: Vec::new(),
                });
            } else {
                for slot in frame.pushed.drain(..) {
                    if let Some(v) = versions.get_mut(&slot) {
                        v.pop();
                    }
                }
                stack.pop();
            }
        }
    }

    // `ILR_SSA_METHOD=0xRVA` narrows the two SSA debug dumps below to ONE
    // method (matched on its entry address). Without it the `ILR_SSA_DBG`
    // trace prints for every method in the binary — hundreds of MB.
    let ssa_method_only = std::env::var("ILR_SSA_METHOD")
        .ok()
        .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok());
    let ssa_method_matches =
        ssa_method_only.is_none_or(|a| insns.first().map(|i| i.address) == Some(a));
    if crate::recovery::env_flags::flag("ILR_SSA_UNVISIT") && ssa_method_matches {
        for (bi, b) in cfg.blocks.iter().enumerate() {
            if bi == 0 || visited.get(bi).copied().unwrap_or(false) {
                continue;
            }
            // The block's own TERMINATOR (its last insn) is what decides its
            // outgoing edges, and the block BEFORE it decides the incoming one.
            // An orphan with an empty predecessor set is the fall-through of a
            // terminator that produced no edge (a `Jump` out of body, or an
            // `IndirectCall`/`Return`); printing the shape says which.
            let lasti = b.end.saturating_sub(1);
            let prev = if b.start > 0 { Some(b.start - 1) } else { None };
            let s = cfg.successors.get(bi);
            let s = match s {
                Some(v) => format!("{v:?}"),
                None => "None".into(),
            };
            eprintln!(
                "[ssa-unvisited] B{bi} range={}..{} insns={} idom={:?} in_rpo={} preds={:?} succs={}",
                b.start,
                b.end,
                b.end - b.start,
                dom.idom.get(bi).copied().flatten(),
                dom.rpo.contains(&bi),
                preds.get(bi),
                s
            );
            if let Some(li) = insns.get(lasti) {
                eprintln!(
                    "    last i{lasti} @{:#x} op={:?} ops={:?}",
                    li.address, li.op, li.operands
                );
            }
            if let Some(pi) = prev {
                if let Some(pi_insn) = insns.get(pi) {
                    eprintln!("    prev i{pi} @{:#x} op={:?}", pi_insn.address, pi_insn.op);
                }
            }
            for i in b.start..b.end {
                eprintln!(
                    "    i{i} @{:#x} op={:?} operands={:?}",
                    insns[i].address, insns[i].op, insns[i].operands
                );
            }
        }
        let claimed: usize = cfg.blocks.iter().map(|b| b.end - b.start).sum();
        eprintln!(
            "[ssa-coverage] insns={} claimed_by_blocks={} blocks={}",
            insns.len(),
            claimed,
            cfg.blocks.len()
        );
    }

    // ---- sweep: blocks the dominator DFS never reached ----------------------
    // A CFG block with NO predecessors is not part of the dominator tree
    // (`idom = None`), so the renaming DFS above never visits it. Leaving its
    // register uses unrenamed is never right: the renderer has no expression
    // for a bare register and prints `__unknown`, and the value such a block
    // reads is exactly the pre-seeded ENTRY version (the entry versions stay
    // on the stacks for the whole function — they are never popped). Sweeping
    // these blocks closes the last hole through which raw `Register(..)`
    // operands reached the final listing (≈300k operands binary-wide).
    if n_blocks > 0 {
        for bi in 0..n_blocks {
            if visited[bi] {
                continue;
            }
            let b = &cfg.blocks[bi];
            let handler_entry = handler_entries.contains(&bi);
            let mut handler_defs: HashMap<Slot, u32> = HashMap::new();
            for i in b.start..b.end {
                for slot in use_slots(&insns[i]) {
                    let val = if handler_entry {
                        match handler_defs.get(&slot) {
                            Some(&v) => Operand::Local(v),
                            None => versions
                                .get(&slot)
                                .and_then(|v| v.first().copied())
                                .map(Operand::Local)
                                .unwrap_or(Operand::Unknown),
                        }
                    } else {
                        match versions.get(&slot).and_then(|v| v.last().copied()) {
                            Some(v) => Operand::Local(v),
                            // Never seen anywhere: bind a fresh version to the
                            // slot so `locals::create` can still map it by slot
                            // name (x0→this, x1→param0, …) instead of printing
                            // nothing.
                            None => {
                                let nv = fresh(&mut counters, &mut version_slot, &slot);
                                versions.entry(slot.clone()).or_default().push(nv);
                                Operand::Local(nv)
                            }
                        }
                    };
                    use_rewrites.entry(i).or_default().push((slot, val));
                }
                if let Some(slot) = def_slot(&insns[i]) {
                    let v = fresh(&mut counters, &mut version_slot, &slot);
                    versions.entry(slot.clone()).or_default().push(v);
                    if handler_entry {
                        handler_defs.insert(slot, v);
                    }
                    if insns[i].op == Op::Call {
                        call_ret_rewrite.insert(i, v);
                    } else {
                        dest_rewrite.insert(i, v);
                    }
                }
            }
        }
    }

    // ---- apply collected rewrites ---------------------------------------------
    let mut out_insns: Vec<Insn> = insns.to_vec();
    for (i, rews) in use_rewrites {
        for (slot, val) in rews {
            replace_use_in_place(&mut out_insns[i], &slot, &val);
        }
    }
    for (&i, &v) in &dest_rewrite {
        if let Some(d) = out_insns[i].dest_mut() {
            *d = Operand::Local(v);
        }
    }
    for (&i, &v) in &call_ret_rewrite {
        if out_insns[i].operands.len() > 1 {
            out_insns[i].operands[1] = Operand::Local(v);
        }
    }

    // Rotated loops: land each loop-carried slot's last in-header value back
    // on its entry version, in front of the header's terminator, so the header
    // re-reads it on the next pass. The terminator is where the block ends, so
    // the copy runs on the fall-through AND on the taken edge — exactly the
    // "this iteration's result is the next iteration's input" the phi would
    // have expressed, but attached to the slot's entry version, which is also
    // what the caller handed in.
    if !rotated.is_empty() {
        let b0 = cfg.blocks[0];
        let at = if b0.end > b0.start && insns[b0.end - 1].is_terminator() {
            b0.end - 1
        } else {
            b0.end
        };
        let addr = insns
            .get(at.min(insns.len().saturating_sub(1)))
            .map(|i| i.address)
            .unwrap_or(0);
        let mut insertions: BTreeMap<usize, Vec<Insn>> = BTreeMap::new();
        for (slot, entry_v) in rotated {
            let mut last: Option<u32> = None;
            for i in b0.start..b0.end {
                if def_slot(&insns[i]).as_deref() == Some(slot.as_str()) {
                    if let Some(v) = dest_rewrite.get(&i).or_else(|| call_ret_rewrite.get(&i)) {
                        last = Some(*v);
                    }
                }
            }
            if let Some(v) = last {
                insertions.entry(at).or_default().push(Insn::move_(
                    Operand::Local(entry_v),
                    Operand::Local(v),
                    addr,
                ));
            }
        }
        out_insns = transform(&out_insns, insertions, &BTreeSet::new());
    }

    if crate::recovery::env_flags::flag("ILR_VERIFY") {
        fn count_regs(op: &Operand, n: &mut usize) {
            match op {
                Operand::Register(_) => *n += 1,
                Operand::Memory { base, index, .. } => {
                    count_regs(base, n);
                    if let Some(i) = index {
                        count_regs(i, n);
                    }
                }
                Operand::AddressOf(x) | Operand::ArrayLength(x) => count_regs(x, n),
                Operand::ArrayAccess { array, index } => {
                    count_regs(array, n);
                    count_regs(index, n);
                }
                _ => {}
            }
        }
        let mut n = 0usize;
        for insn in &out_insns {
            for op in &insn.operands {
                count_regs(op, &mut n);
            }
        }
        for _ in 0..n {
            super::verify::bump_cluster_external("POST-SSA raw register survivor [ssa-bug]");
        }
        for insn in &out_insns {
            let tag = match &insn.op {
                Op::Move => "Move",
                Op::Arith(_) => "Arith",
                Op::Compare(_) => "Compare",
                Op::Call => "Call",
                Op::CallVoid => "CallVoid",
                Op::CondJump(_) => "CondJump",
                Op::Jump(_) => "Jump",
                Op::Return => "Return",
                Op::Nop => "Nop",
                _ => "other",
            };
            for (i, op) in insn.operands.iter().enumerate() {
                let mut c = 0usize;
                count_regs(op, &mut c);
                for _ in 0..c {
                    super::verify::bump_cluster_external(&format!(
                        "POST-SSA survivor {tag} operand#{i} (dest_pos={:?})",
                        Insn::dest_position(&insn.op)
                    ));
                }
            }
        }
    }

    SsaOutput {
        insns: out_insns,
        phis,
        version_slot,
        call_defined,
    }
}

/// Splice phi insns into the listing at block starts so downstream passes see
/// them as ordinary `Op::Phi` instructions.
pub fn materialize(output: &mut SsaOutput, cfg: &mut Cfg) {
    // Block boundaries BEFORE the phis are spliced in. `eliminate` reads
    // `phi.preds` as block indices, and splicing phis in can SPLIT and
    // RENUMBER blocks (see the translation at the end of this function), so the
    // pre-splice ranges are captured here and mapped onto the rebuilt cfg.
    let old_blocks = cfg.blocks.clone();
    let mut insertions: BTreeMap<usize, Vec<Insn>> = BTreeMap::new();
    let mut ordered_ids: Vec<usize> = (0..output.phis.len()).collect();
    ordered_ids.sort_by_key(|&id| {
        let p = &output.phis[id];
        (cfg.blocks[p.block].start, p.dest_version)
    });

    for &pid in &ordered_ids {
        let p = &output.phis[pid];
        let at = cfg.blocks[p.block].start;
        let addr = output.insns.get(at).map(|x| x.address).unwrap_or(0);
        let mut operands = vec![Operand::Local(p.dest_version)];
        for a in &p.args {
            operands.push(match a {
                Some(v) => Operand::Local(*v),
                None => Operand::Unknown,
            });
        }
        insertions.entry(at).or_default().push(Insn {
            op: Op::Phi,
            operands,
            dest_type: None,
            address: addr,
        });
    }

    // How many insns the splice puts at or before each OLD index. `transform`
    // has no removals here, so an old index i lands at `i + prefix[i]`.
    let insert_sizes: Vec<(usize, usize)> =
        insertions.iter().map(|(at, v)| (*at, v.len())).collect();
    let mut prefix = vec![0usize; output.insns.len()];
    {
        let mut acc = 0usize;
        let mut k = 0usize;
        for (i, slot) in prefix.iter_mut().enumerate() {
            while k < insert_sizes.len() && insert_sizes[k].0 <= i {
                acc += insert_sizes[k].1;
                k += 1;
            }
            *slot = acc;
        }
    }

    output.insns = transform(&output.insns, insertions, &BTreeSet::new());
    *cfg = rebuild_cfg(&output.insns);

    // ---- translate phi predecessor BLOCK INDICES to the rebuilt cfg --------
    //
    // A phi is spliced in at its block's START. When that start follows a
    // terminator (a loop header, a join after a conditional — the common case)
    // the phi is itself a leader AND the old block start remains a leader
    // (branch targets were remapped past the phi), so the block is SPLIT and
    // every later block index shifts by one. `phi.preds` were recorded against
    // the PRE-splice numbering, so `eliminate` then read the wrong predecessor
    // block and planted the edge copy at the end of some unrelated block.
    //
    // Measured on `SmallXmlParser.ReadContent` @0x31D1E0: the loop-carried x0
    // phi's back-edge copy landed in a block the definition could not reach, so
    // `liveness_dce` (correctly, for the graph it was given) deleted the live
    // `this.reader` load and the output read an undefined local. The edge copy
    // must go at the end of the block that really ends with the terminator, so
    // each old block maps to the new block containing its LAST instruction.
    let mut new_block_of = vec![usize::MAX; output.insns.len()];
    for (bi, b) in cfg.blocks.iter().enumerate() {
        for p in b.start..b.end.min(output.insns.len()) {
            new_block_of[p] = bi;
        }
    }
    let mut remap = vec![usize::MAX; old_blocks.len()];
    for (b, blk) in old_blocks.iter().enumerate() {
        if blk.end == 0 {
            continue;
        }
        let last = blk.end - 1;
        let new_pos = last + prefix.get(last).copied().unwrap_or(0);
        if let Some(&nb) = new_block_of.get(new_pos) {
            remap[b] = nb;
        }
    }
    for phi in output.phis.iter_mut() {
        for p in phi.preds.iter_mut() {
            if let Some(&nb) = remap.get(*p) {
                if nb != usize::MAX {
                    *p = nb;
                }
            }
        }
    }
}

/// Lower every surviving `Op::Phi` into Moves at predecessor edges and remove
/// the phi instructions themselves.
pub fn eliminate(output: &mut SsaOutput, cfg: &Cfg) {
    // locate surviving phi insns: one pass over the listing matched against
    // a version→phi map (was O(phixinsns) — dominated large methods)
    let dest_to_pid: HashMap<u32, usize> = output
        .phis
        .iter()
        .enumerate()
        .map(|(pid, p)| (p.dest_version, pid))
        .collect();
    let mut phi_at: HashMap<usize, usize> = HashMap::new(); // insn idx → pid
    for (i, insn) in output.insns.iter().enumerate() {
        if insn.op == Op::Phi {
            if let Some(Operand::Local(v)) = insn.operands.first() {
                if let Some(&pid) = dest_to_pid.get(v) {
                    phi_at.insert(i, pid);
                }
            }
        }
    }

    // edge moves keyed at "end of pred block, before its terminator".
    //
    // DETERMINISTIC order: `phi_at` is a HashMap (insn idx → phi id), and
    // iterating it directly pushed the moves that share one insertion point in
    // per-process-random order. Several phis commonly share an insertion
    // point, so the emitted edge-copy sequence — and therefore the final
    // statement order inside the block — changed from run to run on the same
    // binary (a statement like `v33 = …; v34 = attrName;` came out swapped).
    // Sort by the phi's own position in the listing, which is the order the
    // phis were materialized in; the edge loop below stays in edge order.
    let mut phi_order: Vec<(usize, usize)> = phi_at.iter().map(|(&i, &pid)| (i, pid)).collect();
    phi_order.sort_unstable();

    // The PHI INSTRUCTIONS carry the args that were actually materialized into
    // the listing. `output.phis[pid].args` can be stale by now: later passes
    // (`highvar`, `restore_phi_structure`) rewrite the listing, and the record
    // keeps the pre-rewrite values. Building the collapse decision and the
    // edge copies from the record left degenerate x4/x5 join phis (`phi(a,a)`)
    // un-collapsed, then removed the phi insn without planting a definition —
    // tens of thousands of undefined locals in the dump. The listing is the
    // authority.
    let materialized_args: HashMap<usize, Vec<Option<u32>>> = phi_at
        .keys()
        .map(|&idx| {
            let args = output.insns[idx]
                .operands
                .iter()
                .skip(1)
                .map(|o| match o {
                    Operand::Local(v) => Some(*v),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (idx, args)
        })
        .collect();
    let args_for = |insn_idx: usize, pid: usize| -> Vec<Option<u32>> {
        match materialized_args.get(&insn_idx) {
            Some(a) if !a.is_empty() => a.clone(),
            _ => output.phis[pid].args.clone(),
        }
    };

    // ---- degenerate-phi collapse ----------------------------------------
    //
    // When a phi's REAL args (dropping `None` slots from unreachable preds)
    // all name the SAME version, the merge is a rename: every predecessor
    // hands the same SSA version to `dest_version`. Instead of planting a
    // Move at every pred's terminator — the 30% of temp references the agent
    // traced to SSA phi elimination — we substitute every use of
    // `dest_version` with the common arg and skip inserting any Moves for
    // this phi. The phi insn itself is already in `removals`.
    //
    // This is always value-correct because SSA's uniqueness of def means the
    // common arg is the ONLY definition that can reach the phi's block on any
    // live edge; there is no other value that could be observed at a use.
    //
    // The `dest_version` never leaks out: it is fresh (allocated in `build`)
    // and only this phi writes it, so after substitution no reference to it
    // remains.
    //
    // The decision is a FIXPOINT, not one pass. Collapsing `phi(a,a)` turns a
    // consumer `phi(dest,dest)` into `phi(a,a)` only once the first collapse
    // is known, and a long if/else chain ends in exactly such a cascade: the
    // shared tail `phi(x5, x5, x5, …)` names the intermediate join destinations,
    // so a single-pass decision left it un-collapsed and planted edge copies
    // reading versions whose phis had just been deleted (24 undefined locals on
    // `BundleStoreItem.Initialize` @0x166D8F8). Every round re-reads the
    // arguments with all collapses proved so far applied, so the decision sees
    // terminal values.
    let entry_args: Vec<Vec<Option<u32>>> = phi_order
        .iter()
        .map(|&(insn_idx, pid)| args_for(insn_idx, pid))
        .collect();
    let mut phi_args: Vec<Vec<Option<u32>>> = entry_args.clone();
    let mut substitution: HashMap<u32, u32> = HashMap::new();
    // Phis whose collapse would close a cycle stay materialized. SSA cannot
    // construct a phi cycle (dest versions are fresh), but we defend against
    // the pathological and must not strand a read on a removed phi.
    let mut blocked: HashSet<u32> = HashSet::new();
    if !crate::recovery::env_flags::flag("ILR_NOSSA") {
        loop {
            // Re-derive the arguments from the materialized listing with the
            // current substitution — never from the previous round's vector, so
            // a value cannot composite with a stale one.
            for (k, args) in phi_args.iter_mut().enumerate() {
                *args = entry_args[k].clone();
                for a in args.iter_mut() {
                    if let Some(v) = a {
                        if let Some(&t) = substitution.get(v) {
                            *v = t;
                        }
                    }
                }
            }
            let mut round: Vec<(u32, u32)> = Vec::new();
            for (k, &(_insn_idx, pid)) in phi_order.iter().enumerate() {
                let dest = output.phis[pid].dest_version;
                if substitution.contains_key(&dest) || blocked.contains(&dest) {
                    continue;
                }
                let mut iter = phi_args[k].iter().flatten().copied();
                let Some(first) = iter.next() else { continue };
                if iter.all(|a| a == first) {
                    round.push((dest, first));
                }
            }
            if round.is_empty() {
                break;
            }
            let round_map: HashMap<u32, u32> = round.iter().copied().collect();
            let mut accepted: Vec<(u32, u32)> = Vec::new();
            let mut progress = false;
            for &(dest, _arg) in &round {
                // Chase dest → arg → … through this round's collapses and the
                // substitution already proved, to the terminal version. A
                // repeated version means a cycle.
                let mut cur = dest;
                let mut seen: HashSet<u32> = HashSet::new();
                let mut cycled = false;
                while let Some(&nxt) = round_map.get(&cur).or_else(|| substitution.get(&cur)) {
                    if !seen.insert(cur) {
                        cycled = true;
                        break;
                    }
                    cur = nxt;
                }
                if cycled {
                    blocked.insert(dest);
                    progress = true;
                } else if cur != dest {
                    accepted.push((dest, cur));
                    progress = true;
                }
            }
            for (dest, terminal) in accepted {
                substitution.insert(dest, terminal);
            }
            if !progress {
                break;
            }
        }
        // One last refresh so the edge-copy insertion below reads the same
        // terminal arguments the collapse decision settled on.
        for (k, args) in phi_args.iter_mut().enumerate() {
            *args = entry_args[k].clone();
            for a in args.iter_mut() {
                if let Some(v) = a {
                    if let Some(&t) = substitution.get(v) {
                        *v = t;
                    }
                }
            }
        }
    }

    if !substitution.is_empty() {
        // Rewrite every Local operand referring to a collapsed dest, in-place
        // over the listing. The phi insns themselves are already scheduled for
        // removal, so we don't need to skip them.
        for insn in &mut output.insns {
            for op in &mut insn.operands {
                rewrite_dest(op, &substitution);
            }
        }
    }

    let collapsed_set: HashSet<usize> = phi_order
        .iter()
        .filter(|(_i, pid)| substitution.contains_key(&output.phis[*pid].dest_version))
        .map(|(i, _)| *i)
        .collect();

    let mut insertions: BTreeMap<usize, Vec<Insn>> = BTreeMap::new();
    for (k, &(insn_idx, pid)) in phi_order.iter().enumerate() {
        // Skip the collapsed phis — their edge copies are dead now that every
        // use of `dest_version` names the common arg directly.
        if collapsed_set.contains(&insn_idx) {
            continue;
        }
        let phi = &output.phis[pid];
        let args = &phi_args[k];
        let mut placed = 0usize;
        for (edge, &p) in phi.preds.iter().enumerate() {
            let Some(Some(arg)) = args.get(edge).map(|a| *a) else {
                continue;
            };
            // Bounds-guard every index: phi.preds were recorded against the
            // CFG at build time; guard excision / removals can shift or drop
            // blocks (this aborted whole dumps via index-out-of-bounds).
            let Some(blk) = cfg.blocks.get(p) else {
                continue;
            };
            if blk.end == 0 {
                continue;
            }
            let pred_end = blk.end - 1;
            let Some(term_ref) = output.insns.get(pred_end) else {
                continue;
            };
            let at = if term_ref.is_terminator() {
                pred_end
            } else {
                pred_end + 1
            };
            let addr = term_ref.address;
            insertions.entry(at).or_default().push(Insn::move_(
                Operand::Local(phi.dest_version),
                Operand::Local(arg),
                addr,
            ));
            placed += 1;
        }
        // No edge could be materialized (stale predecessor list): define the
        // dest at the block head rather than removing the phi and leaving
        // every use undefined. If the args disagree this is a best-effort
        // value, but it is still a definition on every path.
        if placed == 0 {
            if let Some(arg) = args.iter().flatten().next().copied() {
                let at = cfg
                    .blocks
                    .get(phi.block)
                    .map(|b| b.start)
                    .unwrap_or(0);
                let addr = output.insns.get(at).map(|x| x.address).unwrap_or(0);
                insertions.entry(at).or_default().push(Insn::move_(
                    Operand::Local(phi.dest_version),
                    Operand::Local(arg),
                    addr,
                ));
            }
        }
    }

    let removals: BTreeSet<usize> = phi_at.keys().copied().collect();
    output.insns = transform(&output.insns, insertions, &removals);
    output.phis.clear();
    // The CFG is left stale on purpose: `materialize`/`eliminate` are the last
    // consumers of it, and rebuilding costs a full CFG construction per method.
    // Callers that need a fresh CFG after phi elimination build their own.
}

/// Rewrite every `Operand::Local(v)` referring to a phi dest that collapsed
/// to its common arg. Recurses through compound operands so Memory bases and
/// ArrayAccess indices also see the substitution.
fn rewrite_dest(op: &mut Operand, map: &HashMap<u32, u32>) {
    match op {
        Operand::Local(v) => {
            if let Some(&new_v) = map.get(v) {
                *v = new_v;
            }
        }
        Operand::Memory { base, index, .. } => {
            rewrite_dest(base, map);
            if let Some(i) = index {
                rewrite_dest(i, map);
            }
        }
        Operand::AddressOf(inner) | Operand::ArrayLength(inner) => rewrite_dest(inner, map),
        Operand::ArrayAccess { array, index } => {
            rewrite_dest(array, map);
            rewrite_dest(index, map);
        }
        Operand::InstanceField { base, .. } => rewrite_dest(base, map),
        _ => {}
    }
}

fn replace_use_in_place(insn: &mut Insn, slot: &str, rep: &Operand) {
    fn walk(op: &mut Operand, slot: &str, rep: &Operand) {
        match op {
            Operand::Register(n) if n == slot => *op = rep.clone(),
            Operand::Memory { base, index, .. } => {
                walk(base, slot, rep);
                if let Some(i) = index {
                    walk(i, slot, rep);
                }
            }
            Operand::AddressOf(inner) | Operand::ArrayLength(inner) => walk(inner, slot, rep),
            Operand::ArrayAccess { array, index } => {
                walk(array, slot, rep);
                walk(index, slot, rep);
            }
            Operand::InstanceField { base, .. } => walk(base, slot, rep),
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
    for op in &mut insn.operands {
        walk(op, slot, rep);
    }
}

/// Backward liveness dataflow over slots (register names pre-renaming).
/// Returns per-block live-IN slot sets. Iterates to fixpoint (bounded by
/// block count iterations in practice; guarded anyway).
pub fn compute_live_in(insns: &[Insn], cfg: &Cfg) -> Vec<HashSet<String>> {
    let n_blocks = cfg.blocks.len();
    let mut use_sets: Vec<HashSet<String>> = vec![HashSet::new(); n_blocks];
    let mut def_sets: Vec<HashSet<String>> = vec![HashSet::new(); n_blocks];

    for (bi, b) in cfg.blocks.iter().enumerate() {
        for i in b.start..b.end {
            // upward-exposed uses: use before def within the block
            for slot in use_slots(&insns[i]) {
                if !def_sets[bi].contains(&slot) {
                    use_sets[bi].insert(slot);
                }
            }
            if let Some(slot) = def_slot(&insns[i]) {
                def_sets[bi].insert(slot);
            }
        }
    }

    let mut live_in: Vec<HashSet<String>> = vec![HashSet::new(); n_blocks];
    let mut live_out: Vec<HashSet<String>> = vec![HashSet::new(); n_blocks];
    let mut changed = true;
    let mut guard = 0usize;
    // Buffer reuse: the old shape allocated one fresh OUT set plus a clone of
    // IN for EVERY block on EVERY fixpoint pass. Swapping the new sets into
    // the tables keeps the same values and the same schedule; only the
    // allocations are recycled (values are membership/equality tested only,
    // so set bucket order cannot reach the output).
    let mut scratch_out: HashSet<String> = HashSet::new();
    let mut scratch_in: HashSet<String> = HashSet::new();
    while changed && guard < 10_000 {
        changed = false;
        guard += 1;
        for bi in (0..n_blocks).rev() {
            // out = ∪ in(succ) — full n-way successor set
            scratch_out.clear();
            for succ in cfg.successors[bi].iter().copied() {
                scratch_out.extend(live_in[succ].iter().cloned());
            }
            // in = use ∪ (out - def)
            scratch_in.clone_from(&use_sets[bi]);
            for s in &scratch_out {
                if !def_sets[bi].contains(s) {
                    scratch_in.insert(s.clone());
                }
            }
            if scratch_out != live_out[bi] {
                std::mem::swap(&mut live_out[bi], &mut scratch_out);
                changed = true;
            }
            if scratch_in != live_in[bi] {
                std::mem::swap(&mut live_in[bi], &mut scratch_in);
                changed = true;
            }
        }
    }
    live_in
}
