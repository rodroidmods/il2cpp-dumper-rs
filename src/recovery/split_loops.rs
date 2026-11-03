//! IR-level node splitting for irreducible cycles.
//!
//! A cycle whose edge `t -> h` has `h` reaching `t` but not dominating it does
//! not have a single header, so the structurer can only render it with labels
//! and `goto`s. This pass makes such a region single-entry: it clones every
//! instruction of the cycle's strongly-connected region and sends the entries
//! that bypass the cycle header `h` to the clone. The original copy keeps its
//! header entries, so `h` now dominates its whole body; the clone starts at the
//! cloned bypass entry, so it is single-entry as well. Repeated up to
//! `max_splits` times, one entry group is peeled off per split.
//!
//! Invariants (post-`ssa::eliminate`):
//!   * branch targets are INSTRUCTION INDICES into the flat listing;
//!   * there are no `Op::Phi`s to emit;
//!   * `Operand::Local` is a C# variable slot, not an SSA version, so cloned
//!     code may reuse the same locals: the copies execute on mutually
//!     exclusive paths.
//!
//! Safety over output: every ambiguous case refuses the split. A no-op is
//! acceptable; a wrong edge is not.

use super::cfg::Cfg;
use super::dominator::DominatorInfo;
use super::ir::{Insn, Op};
use super::ssa::rebuild_cfg;
use std::collections::{HashMap, HashSet};

/// Refuse to clone a body larger than this many basic blocks.
const MAX_BODY_BLOCKS: usize = 64;
/// Refuse to clone a body larger than this many instructions.
const MAX_BODY_INSNS: usize = 2000;

/// Split irreducible cycles in `insns`, at most `max_splits` clones.
///
/// Returns the number of successful splits. Branch targets are instruction
/// indices and are remapped for cloned code; the original body stays in the
/// listing (it may become unreachable after later splits, which is harmless).
pub fn split_irreducible(insns: &mut Vec<Insn>, max_splits: usize) -> usize {
    if max_splits == 0 || insns.is_empty() {
        return 0;
    }
    // Switch tables make the clone remap risky: the task's contract is to
    // leave such methods completely untouched. Phis cannot occur after SSA
    // elimination, but cloning one would duplicate a definition, so bail too.
    if insns
        .iter()
        .any(|i| matches!(&i.op, Op::Switch(_) | Op::Phi))
    {
        return 0;
    }
    // Appending clones to the listing turns a trailing fallthrough into an
    // edge into the first clone. Only a block terminator without a fallthrough
    // edge may end the original listing.
    if let Some(last) = insns.last() {
        if has_fallthrough(&last.op) {
            return 0;
        }
    }

    let mut splits = 0usize;
    // Progress guard: cloning the same cycle twice means the clone inherited
    // the same inner irreducibility and the pass would loop, inflating the
    // listing. Cycle identity is keyed on the two block START ADDRESSES, which
    // survive instruction insertion (block indices do not).
    // Cycles already offered to the splitter. A clone that inherits the same
    // inner irreducibility produces the same cycle again; retrying it only
    // inflates the listing. Skip it and split a DIFFERENT cycle instead.
    let mut attempted: std::collections::HashSet<(u64, u64)> =
        std::collections::HashSet::new();
    while splits < max_splits {
        let cfg = rebuild_cfg(insns);
        if cfg.blocks.is_empty() {
            break;
        }
        let dom = DominatorInfo::compute(&cfg);
        let candidates = cycle_candidates(&cfg, &dom);
        if std::env::var("ILR_RELOOPDBG").is_ok() {
            eprintln!("[split-dbg] candidates={:?}", candidates.iter().take(8).collect::<Vec<_>>());
        }
        if candidates.is_empty() {
            break;
        }

        let mut progressed = false;
        for (t, h) in candidates {
            let key = (
                insns[cfg.blocks[t].start].address,
                insns[cfg.blocks[h].start].address,
            );
            if attempted.contains(&key) {
                continue;
            }
            if try_split_fallthrough(insns, &cfg, &dom, t, h)
                || try_split(insns, &cfg, &dom, t, h)
            {
                attempted.insert(key);
                splits += 1;
                progressed = true;
                break;
            }
            attempted.insert(key);
        }
        if !progressed {
            break;
        }
    }
    splits
}


/// Duplicate SHARED TAILS at the IR level until each one has a single
/// predecessor.
///
/// A tree emitter can place a block exactly once, so a block reached from two
/// arms that are not siblings has no position and can only be spelled as a
/// `goto`. The census says that is 58% of all jumps. Duplicating the tail into
/// each predecessor removes the join entirely, and it is source-faithful: the
/// C# compiler merges identical continuation blocks, so the source had a copy
/// per branch.
///
/// Only SINGLE-BLOCK tails are duplicated, and only over explicit edges:
///   * the tail block must end in Return/Throw/Jump/CondJump (no fallthrough
///     into the next block, which appending would silently create);
///   * it must contain no Phi and no Switch (cloning a table would duplicate
///     the dispatch);
///   * it must not be a loop header or a back-edge target;
///   * the predecessor must reach it through an explicit branch (a
///     fallthrough predecessor would need the clone inserted mid-listing);
///   * the predecessor must not be inside the tail itself.
///
/// Appending at the END of the listing is what makes this sound: no existing
/// instruction index moves, so every branch target outside the clone stays
/// valid, and the clone's own targets are already outside the tail.
pub fn split_shared_tails(
    insns: &mut Vec<Insn>,
    max_clones: usize,
    max_tail_insns: usize,
    max_chain: usize,
    safe_only: bool,
) -> usize {
    if max_clones == 0 || insns.is_empty() {
        return 0;
    }
    // Appending is only safe when the listing already ends in a terminator:
    // otherwise the last block would start falling into the clone.
    match insns.last().map(|i| &i.op) {
        Some(Op::Return) | Some(Op::Throw) | Some(Op::Jump(_)) | Some(Op::CondJump(_))
        | Some(Op::Switch(_)) | Some(Op::IndirectCall) => {}
        _ => return 0,
    }
    // ONE pass over ONE CFG snapshot: every candidate is cloned from the same
    // block list, and a clone appended at the end cannot change the shape of
    // the candidates already examined. Rebuilding the CFG per clone made this
    // O(clones x blocks) for no benefit.
    let mut clones = 0usize;
    let mut taken: HashSet<usize> = HashSet::new();
    {
        let cfg = rebuild_cfg(insns);
        let n = cfg.blocks.len();
        if n < 2 {
            return 0;
        }
        let dom = DominatorInfo::compute(&cfg);
        let preds = predecessors(&cfg);
        for j in 0..n {
            if clones >= max_clones {
                break;
            }
            if preds[j].len() < 2 || taken.contains(&j) {
                continue;
            }
            // The shared region is a CHAIN: the shared block plus every
            // following block that is reachable only from inside the chain.
            // A single block was the first version; a chain is what the
            // census actually asks for (58% of the jumps are shared
            // continuations, and most of those tails are 2-6 blocks long).
            // Contiguity in the listing is REQUIRED, not assumed: the clone is
            // one instruction range, so a hole (a block reachable from
            // outside) would drag unrelated code into the copy.
            // `max_chain` is 1 in production. MEASURED: chains of 2..8 blocks
            // make the corpus WORSE, not better (19,343 -> 19,425 gotos; 41
            // methods worse, 13 better, 0x14F5338 went 21 -> 65). The emitter
            // does not get more placeable blocks out of a longer clone — it
            // gets a larger listing with more shared edges left over. The
            // parameter stays so the chain path is tested and can be re-tried
            // when the emitter changes.
            let mut region: Vec<usize> = vec![j];
            loop {
                let last = *region.last().unwrap();
                if region.len() >= max_chain {
                    break;
                }
                let next = last + 1;
                if next >= n {
                    break;
                }
                // The edge into `next` must be a fallthrough from `last`.
                let last_blk = cfg.blocks[last];
                if last_blk.end <= last_blk.start {
                    break;
                }
                if !has_fallthrough(&insns[last_blk.end - 1].op) {
                    break;
                }
                // ... and nothing outside the region may enter it.
                if preds[next].is_empty()
                    || !preds[next]
                        .iter()
                        .all(|p| *p >= j && *p <= next && (region.contains(p) || *p == next))
                {
                    break;
                }
                if taken.contains(&next) {
                    break;
                }
                region.push(next);
            }
            let start = cfg.blocks[j].start;
            let end = cfg.blocks[*region.last().unwrap()].end;
            if end <= start || end - start > max_tail_insns {
                continue;
            }
            // A loop header (or any block a predecessor is dominated by) owns
            // a back edge: duplicating it duplicates the loop.
            if region.iter().any(|&b| preds[b].iter().any(|&p| dom.dominates(b, p))) {
                continue;
            }
            if insns[start..end]
                .iter()
                .any(|i| matches!(&i.op, Op::Phi | Op::Switch(_)))
            {
                continue;
            }
            let last_op = &insns[end - 1].op;
            if !matches!(
                last_op,
                Op::Return | Op::Throw | Op::Jump(_) | Op::CondJump(_)
            ) {
                continue;
            }
            // A clone is appended at the END of the listing. When its tail
            // ends by jumping BACK to a lower address, the address walker has
            // to spell that edge as a `goto`. A tail that returns, throws or
            // jumps forward costs the walker nothing.
            let backward_tail = match last_op {
                Op::Jump(t) | Op::CondJump(t) => *t < start,
                _ => false,
            };
            if safe_only && backward_tail {
                continue;
            }
            // Explicit predecessors only, and never one inside the tail.
            let mut sites: Vec<usize> = Vec::new(); // branch insn index
            let mut skipped = false;
            for &p in preds[j].iter() {
                if p >= n || (p >= j && cfg.blocks[p].start < end) {
                    skipped = true;
                    break;
                }
                let pb = cfg.blocks[p];
                if pb.end <= pb.start {
                    skipped = true;
                    break;
                }
                let pi = pb.end - 1;
                match &insns[pi].op {
                    Op::Jump(t) | Op::CondJump(t) if *t == start => sites.push(pi),
                    _ => {
                        skipped = true;
                        break;
                    }
                }
            }
            if skipped || sites.len() < 2 {
                continue;
            }
            if std::env::var("ILR_RELOOPDBG").is_ok() {
                let kind = match last_op {
                    Op::Return => "ret",
                    Op::Throw => "throw",
                    Op::Jump(_) | Op::CondJump(_) if backward_tail => "back",
                    _ => "fwd",
                };
                eprintln!(
                    "[clone] tail B{j}..B{} insns {start}..{end} kind={kind} preds={}",
                    region.last().unwrap(),
                    sites.len()
                );
            }
            // Keep the first site as the canonical copy; clone for the rest.
            for &branch in sites.iter().skip(1) {
                if clones >= max_clones {
                    break;
                }
                let at = insns.len();
                let mut cloned: Vec<Insn> = insns[start..end].to_vec();
                // Internal branch targets move with the copy. Targets outside
                // the range are untouched: the clone's own successors are
                // already outside the region (that is what makes it a tail).
                for insn in cloned.iter_mut() {
                    match &mut insn.op {
                        Op::Jump(t) | Op::CondJump(t) if *t >= start && *t < end => {
                            *t = at + (*t - start);
                        }
                        _ => {}
                    }
                }
                insns.extend(cloned);
                match &mut insns[branch].op {
                    Op::Jump(t) | Op::CondJump(t) => *t = at,
                    _ => unreachable!("checked above"),
                }
                clones += 1;
            }
            for &b in region.iter() {
                taken.insert(b);
            }
        }
    }
    clones
}

/// Is the edge into the successor of this opcode a fallthrough (rather than an
/// explicit branch target)? `CondJump` has both an explicit and a fallthrough
/// edge, so it counts as having a fallthrough.
fn has_fallthrough(op: &Op) -> bool {
    !matches!(
        op,
        Op::Jump(_) | Op::Return | Op::Throw | Op::IndirectCall | Op::Switch(_)
    )
}

/// Can `from` reach `to` over `succs`? Bounded by the block count.
fn reaches(succs: &[Vec<usize>], from: usize, to: usize) -> bool {
    if from == to {
        return true;
    }
    let n = succs.len();
    if from >= n || to >= n {
        return false;
    }
    let mut seen = vec![false; n];
    let mut stack = vec![from];
    seen[from] = true;
    while let Some(b) = stack.pop() {
        for &s in &succs[b] {
            if s == to {
                return true;
            }
            if s < n && !seen[s] {
                seen[s] = true;
                stack.push(s);
            }
        }
    }
    false
}

/// Blocks reachable from `entry` (the method entry is block 0).
fn reachable_from(cfg: &Cfg, entry: usize) -> Vec<bool> {
    let n = cfg.blocks.len();
    let mut seen = vec![false; n];
    if n == 0 || entry >= n {
        return seen;
    }
    let mut stack = vec![entry];
    seen[entry] = true;
    while let Some(b) = stack.pop() {
        for &s in &cfg.successors[b] {
            if s < n && !seen[s] {
                seen[s] = true;
                stack.push(s);
            }
        }
    }
    seen
}

/// Predecessor lists (deduplicated) over the CFG.
fn predecessors(cfg: &Cfg) -> Vec<Vec<usize>> {
    let n = cfg.blocks.len();
    let mut preds = vec![Vec::new(); n];
    for (p, succs) in cfg.successors.iter().enumerate() {
        for &s in succs {
            if s < n && !preds[s].contains(&p) {
                preds[s].push(p);
            }
        }
    }
    preds
}

/// Every entry-reachable cycle edge `t -> h` where `h` reaches `t` but does
/// not dominate `t`.
///
/// Sorted by target block, then source block: the earliest block of the cycle
/// is the most natural header to keep on the original copy. Candidates whose
/// target is not entered from outside the cycle are filtered out later by
/// `try_split`; they need no split.
fn cycle_candidates(cfg: &Cfg, dom: &DominatorInfo) -> Vec<(usize, usize)> {
    let n = cfg.blocks.len();
    let reachable = reachable_from(cfg, 0);
    let mut out = Vec::new();
    for t in 0..n {
        if !reachable[t] {
            continue;
        }
        for &h in &cfg.successors[t] {
            if h >= n || h == t || !reachable[h] {
                continue;
            }
            // A natural back edge: the header dominates its latch.
            if dom.dominates(h, t) {
                continue;
            }
            // Not dominated, but still part of a cycle only when h gets back
            // to t. Otherwise this is an ordinary forward edge.
            if !reaches(&cfg.successors, h, t) {
                continue;
            }
            out.push((t, h));
        }
    }
    out.sort_by_key(|&(t, h)| (h, t));
    out.dedup();
    out
}

/// The cycle's body: blocks reachable from `h` that can also reach `t`.
///
/// `t -> h` closes the cycle, so this is the strongly-connected component of
/// `h` and `t`. The result is sorted by block index, which is also the order
/// in which its instructions are cloned.
fn loop_body(cfg: &Cfg, h: usize, t: usize) -> Option<Vec<usize>> {
    let n = cfg.blocks.len();
    if h >= n || t >= n {
        return None;
    }
    let preds = predecessors(cfg);
    let from_h = reachable_from(cfg, h);

    let mut to_t = vec![false; n];
    let mut stack = vec![t];
    to_t[t] = true;
    while let Some(b) = stack.pop() {
        for &p in &preds[b] {
            if p < n && !to_t[p] {
                to_t[p] = true;
                stack.push(p);
            }
        }
    }

    let body: Vec<usize> = (0..n).filter(|&b| from_h[b] && to_t[b]).collect();
    if body.is_empty() {
        None
    } else {
        Some(body)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeKind {
    /// Reached through a `Jump`/taken `CondJump` instruction operand.
    Explicit,
    /// Reached by physical fallthrough from the preceding block.
    Fallthrough,
}

/// Try to split the cycle `t -> h` once.
///
/// Returns `true` only after the listing has been cloned and the bypassing
/// entries retargeted. Any validation failure leaves `insns` untouched.
/// Clone-content safety WITHOUT the fallthrough-contiguity requirement: the
/// clone builder appends an explicit `Jump` whenever a cloned block's
/// physical fallthrough would land on the wrong clone block.
fn clone_content_safe(insns: &[Insn], cfg: &Cfg, body: &[usize]) -> bool {
    let n = insns.len();
    for &b in body {
        let blk = cfg.blocks[b];
        if blk.end <= blk.start || blk.end > n {
            return false;
        }
        for insn in &insns[blk.start..blk.end] {
            if matches!(&insn.op, Op::IndirectCall | Op::Phi | Op::Switch(_)) {
                return false;
            }
        }
    }
    true
}

/// Build a clone of `body` to be INSERTED at `insert_at` (so a fallthrough
/// predecessor flows into it). Returns (old→clone index map, cloned insns,
/// clone length). A block whose physical fallthrough does not reach the next
/// cloned block gets an explicit `Jump` appended; the structurer then sees the
/// same successor the original had.
fn build_insert_clone(
    insns: &[Insn],
    cfg: &Cfg,
    body: &[usize],
    insn_block: &[usize],
    insert_at: usize,
) -> Option<(HashMap<usize, usize>, Vec<Insn>, usize)> {
    let n = insns.len();
    let mut extra: Vec<Option<usize>> = Vec::with_capacity(body.len());
    let mut lens: Vec<usize> = Vec::with_capacity(body.len());
    for (pos, &b) in body.iter().enumerate() {
        let blk = cfg.blocks[b];
        let last = insns.get(blk.end.checked_sub(1)?)?;
        let mut jump: Option<usize> = None;
        if has_fallthrough(&last.op) {
            if blk.end >= n {
                // Falls off the original listing: preserve that as an exit.
                jump = Some(usize::MAX);
            } else {
                let t = blk.end;
                let q = *insn_block.get(t)?;
                let next = body.get(pos + 1).copied();
                let aligned = q != usize::MAX
                    && Some(q) == next
                    && cfg.blocks.get(q).map(|x| x.start) == Some(t);
                if !aligned {
                    jump = Some(t);
                }
            }
        }
        lens.push(blk.end - blk.start + usize::from(jump.is_some()));
        extra.push(jump);
    }
    let clone_len: usize = lens.iter().sum();
    if clone_len == 0 {
        return None;
    }
    let old_len = n;
    let new_len = old_len + clone_len;
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut next = insert_at;
    for (pos, &b) in body.iter().enumerate() {
        let blk = cfg.blocks[b];
        for old in blk.start..blk.end {
            map.insert(old, next);
            next += 1;
        }
        if extra[pos].is_some() {
            next += 1;
        }
    }
    let mut cloned: Vec<Insn> = Vec::with_capacity(clone_len);
    for (pos, &b) in body.iter().enumerate() {
        let blk = cfg.blocks[b];
        for old in blk.start..blk.end {
            let mut insn = insns[old].clone();
            insert_remap(&mut insn.op, &map, insert_at, old_len, new_len);
            cloned.push(insn);
        }
        if let Some(t) = extra[pos] {
            let addr = insns[blk.end - 1].address;
            let tgt = if t == usize::MAX {
                new_len
            } else if let Some(&nt) = map.get(&t) {
                nt
            } else if t >= old_len {
                new_len
            } else if t >= insert_at {
                t + clone_len
            } else {
                t
            };
            cloned.push(Insn::new(Op::Jump(tgt), vec![], addr));
        }
    }
    if cloned.len() != clone_len {
        return None;
    }
    Some((map, cloned, clone_len))
}

/// Split a cycle whose extra entry is a PHYSICAL FALLTHROUGH.
///
/// The clone is inserted at the fallthrough predecessor's end, so that edge
/// flows into the clone without changing the predecessor. Every branch target
/// at/after the insertion point is shifted by the clone length; explicit
/// bypass branches are retargeted into the clone. Only ONE fallthrough entry
/// is handled per split; further entries are peeled in later iterations.
fn try_split_fallthrough(
    insns: &mut Vec<Insn>,
    cfg: &Cfg,
    dom: &DominatorInfo,
    t: usize,
    h: usize,
) -> bool {
    let dbg = std::env::var("ILR_RELOOPDBG").is_ok();
    let n_blocks = cfg.blocks.len();
    let n_insns = insns.len();
    if n_insns == 0 || h >= n_blocks || t >= n_blocks || h == t {
        return false;
    }
    let Some(body) = loop_body(cfg, h, t) else {
        return false;
    };
    if body.len() < 2 || body.len() > MAX_BODY_BLOCKS || body.contains(&0) {
        return false;
    }
    let mut body_insns = 0usize;
    for &b in &body {
        let blk = cfg.blocks[b];
        if blk.end <= blk.start || blk.end > n_insns {
            return false;
        }
        body_insns += blk.end - blk.start;
        if body_insns > MAX_BODY_INSNS {
            return false;
        }
    }
    let reachable = reachable_from(cfg, 0);
    let mut insn_block = vec![usize::MAX; n_insns];
    for (bi, blk) in cfg.blocks.iter().enumerate() {
        for i in blk.start.min(n_insns)..blk.end.min(n_insns) {
            insn_block[i] = bi;
        }
    }
    let mut in_body = vec![false; n_blocks];
    for &b in &body {
        in_body[b] = true;
    }
    let mut has_anchor = false;
    // (branch instruction index, old target instruction index)
    let mut explicit: Vec<(usize, usize)> = Vec::new();
    // (pred block, target block) for fallthrough extra entries
    let mut falls: Vec<(usize, usize)> = Vec::new();
    for p in 0..n_blocks {
        if in_body[p] || !reachable[p] {
            continue;
        }
        let blk = cfg.blocks[p];
        if blk.end <= blk.start || blk.end > n_insns {
            return false;
        }
        let last_idx = blk.end - 1;
        let last = &insns[last_idx];
        let mut edges: Vec<(usize, EdgeKind, usize)> = Vec::new();
        match &last.op {
            Op::Jump(dst) => {
                if let Some(&q) = insn_block.get(*dst) {
                    if q != usize::MAX {
                        edges.push((q, EdgeKind::Explicit, *dst));
                    }
                }
            }
            Op::CondJump(dst) => {
                if let Some(&q) = insn_block.get(*dst) {
                    if q != usize::MAX {
                        edges.push((q, EdgeKind::Explicit, *dst));
                    }
                }
                if blk.end < n_insns {
                    let q = insn_block[blk.end];
                    if q != usize::MAX {
                        edges.push((q, EdgeKind::Fallthrough, blk.end));
                    }
                }
            }
            Op::Switch(_) => return false,
            Op::Return | Op::Throw | Op::IndirectCall => {}
            _ => {
                if blk.end < n_insns {
                    let q = insn_block[blk.end];
                    if q != usize::MAX {
                        edges.push((q, EdgeKind::Fallthrough, blk.end));
                    }
                }
            }
        }
        for (q, kind, target_insn) in edges {
            if q >= n_blocks || !in_body[q] {
                continue;
            }
            if q == h {
                has_anchor = true;
                continue;
            }
            if dom.dominates(h, p) || target_insn != cfg.blocks[q].start {
                return false;
            }
            match kind {
                EdgeKind::Explicit => {
                    if !matches!(&last.op, Op::Jump(_) | Op::CondJump(_)) {
                        return false;
                    }
                    explicit.push((last_idx, target_insn));
                }
                EdgeKind::Fallthrough => falls.push((p, q)),
            }
        }
    }
    // Unlike the explicit path, a header anchor is NOT required: when the
    // only entries into the cycle are bypasses, the clone owns the loop and
    // the original copy simply becomes unreachable (dead code is harmless and
    // the later DCE removes it).
    let _ = has_anchor;
    if falls.len() != 1 {
        if dbg { eprintln!("[split-dbg] t={t} h={h}: fallthrough count={}", falls.len()); }
        return false;
    }
    // The fallthrough clone must be entered by physically following the pred,
    // so the target must be the first block of the clone in index order.
    let (fp, fq) = falls[0];
    let first = *body.iter().min().unwrap();
    let insert_at = cfg.blocks[fp].end;
    if fq != first || cfg.blocks[first].start != insert_at {
        if dbg { eprintln!("[split-dbg] t={t} h={h}: first={first} fq={fq} insert_at={insert_at}"); }
        return false;
    }
    if !clone_content_safe(insns, cfg, &body) {
        if dbg {
            eprintln!("[split-dbg] fallthrough t={t} h={h}: clone content unsafe");
        }
        return false;
    }
    let old_len = n_insns;
    let Some((map, cloned, clone_len)) =
        build_insert_clone(insns, cfg, &body, &insn_block, insert_at)
    else {
        return false;
    };
    let new_len = old_len + clone_len;
    // Shift every original target at/after the insertion point.
    for insn in insns.iter_mut() {
        insert_shift(&mut insn.op, insert_at, old_len, new_len);
    }
    // Retarget explicit bypass branches into the clone. Their own index shifts
    // when they sit at/after the insertion point.
    for (branch_idx, old_target) in explicit {
        let bi = if branch_idx >= insert_at {
            branch_idx + clone_len
        } else {
            branch_idx
        };
        let Some(&new_target) = map.get(&old_target) else {
            return false;
        };
        if bi >= insns.len() {
            return false;
        }
        match &mut insns[bi].op {
            Op::Jump(x) | Op::CondJump(x) => *x = new_target,
            _ => return false,
        }
    }
    insns.splice(insert_at..insert_at, cloned);
    if dbg {
        eprintln!(
            "[split-dbg] fallthrough split t={t} h={h} at={insert_at} len={clone_len}"
        );
    }
    true
}

/// Shift targets of an original instruction for a clone inserted at `at`.
fn insert_shift(op: &mut Op, at: usize, old_len: usize, new_len: usize) {
    match op {
        Op::Jump(t) | Op::CondJump(t) => {
            if *t >= old_len {
                *t = new_len;
            } else if *t >= at {
                *t += new_len - old_len;
            }
        }
        Op::Switch(ts) => {
            for t in ts.iter_mut() {
                if *t >= old_len {
                    *t = new_len;
                } else if *t >= at {
                    *t += new_len - old_len;
                }
            }
        }
        _ => {}
    }
}

/// Remap a clone instruction's targets: body targets to the clone, exits to
/// the new end, other targets shifted past the insertion point.
fn insert_remap(
    op: &mut Op,
    map: &HashMap<usize, usize>,
    at: usize,
    old_len: usize,
    new_len: usize,
) {
    fn one(t: &mut usize, map: &HashMap<usize, usize>, at: usize, old_len: usize, new_len: usize) {
        if let Some(&n) = map.get(&*t) {
            *t = n;
        } else if *t >= old_len {
            *t = new_len;
        } else if *t >= at {
            *t += new_len - old_len;
        }
    }
    match op {
        Op::Jump(t) | Op::CondJump(t) => one(t, map, at, old_len, new_len),
        Op::Switch(ts) => {
            for t in ts.iter_mut() {
                one(t, map, at, old_len, new_len);
            }
        }
        _ => {}
    }
}

fn try_split(
    insns: &mut Vec<Insn>,
    cfg: &Cfg,
    dom: &DominatorInfo,
    t: usize,
    h: usize,
) -> bool {
    let dbg = std::env::var("ILR_RELOOPDBG").is_ok();
    let n_blocks = cfg.blocks.len();
    let n_insns = insns.len();
    if n_insns == 0 || h >= n_blocks || t >= n_blocks || h == t {
        return false;
    }

    let body = match loop_body(cfg, h, t) {
        Some(b) => b,
        None => {
            if dbg { eprintln!("[split-dbg] t={t} h={h}: no body"); }
            return false;
        }
    };
    // A body containing the method entry cannot be redirected: that entry is
    // implicit, so the original copy would keep an entry that bypasses `h`.
    if body.len() < 2 || body.len() > MAX_BODY_BLOCKS || body.contains(&0) {
        if dbg { eprintln!("[split-dbg] t={t} h={h}: body len={} contains0={}", body.len(), body.contains(&0)); }
        return false;
    }

    let mut body_insns = 0usize;
    for &b in &body {
        let blk = cfg.blocks[b];
        if blk.end <= blk.start || blk.end > n_insns {
            return false;
        }
        body_insns = body_insns.saturating_add(blk.end - blk.start);
        if body_insns > MAX_BODY_INSNS {
            return false;
        }
    }

    let reachable = reachable_from(cfg, 0);
    let mut insn_block = vec![usize::MAX; n_insns];
    for (bi, blk) in cfg.blocks.iter().enumerate() {
        for i in blk.start.min(n_insns)..blk.end.min(n_insns) {
            insn_block[i] = bi;
        }
    }
    let mut in_body = vec![false; n_blocks];
    for &b in &body {
        in_body[b] = true;
    }

    // Entries into the body from outside it. The cycle header `h` is the
    // anchor: its entries stay on the original copy. Every other entry is the
    // "extra entry" that makes `h` fail to dominate its own cycle, and those
    // are moved to the clone.
    let mut has_anchor = false;
    // (branch instruction, old target instruction, target block)
    let mut retargets: Vec<(usize, usize, usize)> = Vec::new();

    for p in 0..n_blocks {
        if in_body[p] || !reachable[p] {
            continue;
        }
        let blk = cfg.blocks[p];
        if blk.end <= blk.start || blk.end > n_insns {
            return false;
        }
        let last_idx = blk.end - 1;
        let last = &insns[last_idx];

        // Edges this block's terminator actually produces. They must agree
        // with `rebuild_cfg`; disagreement means the listing and the CFG have
        // different ideas about control flow, so no split is safe.
        let mut edges: Vec<(usize, EdgeKind, usize)> = Vec::new();
        match &last.op {
            Op::Jump(dst) => {
                if let Some(&q) = insn_block.get(*dst) {
                    if q != usize::MAX {
                        edges.push((q, EdgeKind::Explicit, *dst));
                    }
                }
            }
            Op::CondJump(dst) => {
                if let Some(&q) = insn_block.get(*dst) {
                    if q != usize::MAX {
                        edges.push((q, EdgeKind::Explicit, *dst));
                    }
                }
                if blk.end < n_insns {
                    let q = insn_block[blk.end];
                    if q != usize::MAX {
                        edges.push((q, EdgeKind::Fallthrough, blk.end));
                    }
                }
            }
            Op::Switch(_) => return false,
            Op::Return | Op::Throw | Op::IndirectCall => {}
            _ => {
                if blk.end < n_insns {
                    let q = insn_block[blk.end];
                    if q != usize::MAX {
                        edges.push((q, EdgeKind::Fallthrough, blk.end));
                    }
                }
            }
        }

        let mut expected: Vec<usize> = cfg.successors[p]
            .iter()
            .copied()
            .filter(|&q| q < n_blocks && in_body[q])
            .collect();
        let mut actual: Vec<usize> = edges
            .iter()
            .filter(|e| e.0 < n_blocks && in_body[e.0])
            .map(|e| e.0)
            .collect();
        expected.sort_unstable();
        actual.sort_unstable();
        if expected != actual {
            if dbg { eprintln!("[split-dbg] t={t} h={h}: edge mismatch at p={p} expected={expected:?} actual={actual:?}"); }
            return false;
        }

        for (q, kind, target_insn) in edges {
            if q >= n_blocks || !in_body[q] {
                continue;
            }
            if q == h {
                // Header entry: it does not bypass the header, so leaving it
                // on the original keeps `h` the single entry of that copy.
                has_anchor = true;
                continue;
            }
            // A bypass entry must not already be dominated by h (then it
            // should have been part of the body), and it must be an explicit
            // branch target we can retarget.
            if dom.dominates(h, p) {
                return false;
            }
            if kind != EdgeKind::Explicit {
                if dbg { eprintln!("[split-dbg] t={t} h={h}: fallthrough bypass entry p={p} -> q={q}"); }
                return false;
            }
            if target_insn != cfg.blocks[q].start {
                return false;
            }
            if !matches!(&last.op, Op::Jump(_) | Op::CondJump(_)) {
                return false;
            }
            retargets.push((last_idx, target_insn, q));
        }
    }

    if !has_anchor || retargets.is_empty() {
        if dbg { eprintln!("[split-dbg] t={t} h={h}: anchor={has_anchor} retargets={}", retargets.len()); }
        return false;
    }

    if !clone_is_safe(insns, cfg, &body, &insn_block) {
        if dbg { eprintln!("[split-dbg] t={t} h={h}: clone_is_safe refused"); }
        return false;
    }

    // Old instruction index -> clone index. Built for the WHOLE body before
    // cloning so forward branches inside the body can be remapped too.
    let old_len = insns.len();
    let mut next = old_len;
    let mut map: HashMap<usize, usize> = HashMap::new();
    for &b in &body {
        let blk = cfg.blocks[b];
        for old in blk.start..blk.end {
            map.insert(old, next);
            next = next.saturating_add(1);
        }
    }
    let new_len = next;
    if new_len <= old_len {
        return false;
    }

    let mut resolved: Vec<(usize, usize, usize)> = Vec::with_capacity(retargets.len());
    for (branch_idx, old_target, target_block) in retargets {
        if branch_idx >= old_len || target_block >= n_blocks {
            return false;
        }
        let Some(&new_target) = map.get(&old_target) else {
            return false;
        };
        resolved.push((branch_idx, old_target, new_target));
    }

    // Re-check the operand shape before any mutation, so the retarget phase
    // below cannot fail halfway through.
    for &(branch_idx, old_target, _) in &resolved {
        let ok = match &insns[branch_idx].op {
            Op::Jump(x) | Op::CondJump(x) => *x == old_target,
            _ => false,
        };
        if !ok {
            return false;
        }
    }

    let mut cloned: Vec<Insn> = Vec::with_capacity(new_len - old_len);
    for &b in &body {
        let blk = cfg.blocks[b];
        for old in blk.start..blk.end {
            let mut insn = insns[old].clone();
            remap_cloned_targets(&mut insn.op, &map, old_len, new_len);
            cloned.push(insn);
        }
    }
    if cloned.len() != new_len - old_len {
        return false;
    }

    // A target at/after the old end is the exit sentinel. Some of those values
    // become valid indices once the clone is appended; move them to the NEW
    // end so they keep denoting an exit. Clone targets are already remapped
    // through `map` and are deliberately not touched.
    for insn in insns.iter_mut() {
        bump_past_end_targets(&mut insn.op, old_len, new_len);
    }
    insns.extend(cloned);

    for (branch_idx, old_target, new_target) in resolved {
        match &mut insns[branch_idx].op {
            Op::Jump(x) | Op::CondJump(x) if *x == old_target => *x = new_target,
            _ => {}
        }
    }
    true
}

/// Every instruction of every body block must clone without changing its
/// fallthrough meaning, and no indirect call may be duplicated.
fn clone_is_safe(insns: &[Insn], cfg: &Cfg, body: &[usize], insn_block: &[usize]) -> bool {
    let n = insns.len();
    for (pos, &b) in body.iter().enumerate() {
        let blk = cfg.blocks[b];
        if blk.end <= blk.start || blk.end > n {
            return false;
        }
        for insn in &insns[blk.start..blk.end] {
            if matches!(&insn.op, Op::IndirectCall | Op::Phi | Op::Switch(_)) {
                return false;
            }
        }
        let last = &insns[blk.end - 1];
        if !has_fallthrough(&last.op) {
            continue;
        }
        if blk.end >= n {
            // Falls off the end of the listing (an exit). That is only
            // preserved if this is the last block of the clone as well.
            if pos + 1 != body.len() {
                return false;
            }
            continue;
        }
        // The physical next block in the clone is `body[pos + 1]`; it must be
        // the same block the original fallthrough reached.
        if pos + 1 >= body.len() || body[pos + 1] != insn_block[blk.end] {
            return false;
        }
    }
    true
}

/// Remap a cloned instruction's branch targets into the clone.
///
/// Targets outside the body keep pointing at the same original instruction.
/// Targets that used to be past the old end are exits; the appended clone
/// would otherwise turn them into in-range indices, so they are moved to the
/// new end.
fn remap_cloned_targets(
    op: &mut Op,
    map: &HashMap<usize, usize>,
    old_len: usize,
    new_len: usize,
) {
    match op {
        Op::Jump(t) | Op::CondJump(t) => remap_one(t, map, old_len, new_len),
        Op::Switch(targets) => {
            for t in targets.iter_mut() {
                remap_one(t, map, old_len, new_len);
            }
        }
        _ => {}
    }
}

fn remap_one(t: &mut usize, map: &HashMap<usize, usize>, old_len: usize, new_len: usize) {
    if let Some(&new_target) = map.get(&*t) {
        *t = new_target;
    } else if *t >= old_len && *t < new_len {
        *t = new_len;
    }
}

/// Move branch targets that were past the old listing end (the exit sentinel)
/// to the new end, so appending the clone cannot make them valid indices.
fn bump_past_end_targets(op: &mut Op, old_len: usize, new_len: usize) {
    match op {
        Op::Jump(t) | Op::CondJump(t) => {
            if *t >= old_len && *t < new_len {
                *t = new_len;
            }
        }
        Op::Switch(targets) => {
            for t in targets.iter_mut() {
                if *t >= old_len && *t < new_len {
                    *t = new_len;
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::ir::Operand;

    fn dummy(dst: u32, address: u64) -> Insn {
        Insn::new(
            Op::Move,
            vec![Operand::Local(dst), Operand::Immediate(1)],
            address,
        )
    }

    fn jump(target: usize, address: u64) -> Insn {
        Insn::new(Op::Jump(target), vec![], address)
    }

    fn cond(target: usize, address: u64) -> Insn {
        Insn::new(Op::CondJump(target), vec![], address)
    }

    fn ret(address: u64) -> Insn {
        Insn::new(Op::Return, vec![], address)
    }




    /// Three blocks branch to one tail. Two clones are appended, the first
    /// branch keeps the original, and the clone's own branch target (outside
    /// the copied range) still points at the original successor.
    #[test]
    fn duplicates_a_single_block_shared_tail() {
        let mut insns = vec![
            cond(2, 0x1000),  // 0
            jump(6, 0x1004),  // 1  -> T
            cond(3, 0x1008),  // 2
            jump(6, 0x100c),  // 3  -> T
            jump(6, 0x1010),  // 4  -> T
            ret(0x1014),      // 5  (unreachable filler, keeps 5 out of T)
            dummy(1, 0x1018), // 6  T
            cond(8, 0x101c),  // 7  T -> U
            dummy(2, 0x1020), // 8  U
            ret(0x1024),      // 9
        ];
        let n = split_shared_tails(&mut insns, 8, 60, 1, false);
        assert_eq!(n, 2);
        assert_eq!(insns.len(), 14);
        assert!(matches!(insns[1].op, Op::Jump(t) if t == 6));
        assert!(matches!(insns[3].op, Op::Jump(t) if t == 10));
        assert!(matches!(insns[4].op, Op::Jump(t) if t == 12));
        assert!(matches!(insns[11].op, Op::CondJump(t) if t == 8));
    }

    /// Same CFG with `max_chain = 2`: the clone carries BOTH blocks and its
    /// internal branch is remapped onto the clone's own copy, not the
    /// original.
    #[test]
    fn duplicates_a_two_block_chain_and_remaps_its_internal_branch() {
        let mut insns = vec![
            cond(2, 0x1000),
            jump(6, 0x1004),
            cond(3, 0x1008),
            jump(6, 0x100c),
            jump(6, 0x1010),
            ret(0x1014),
            dummy(1, 0x1018), // 6  T
            cond(8, 0x101c),  // 7  T -> U
            dummy(2, 0x1020), // 8  U
            ret(0x1024),      // 9
        ];
        let n = split_shared_tails(&mut insns, 8, 60, 2, false);
        assert_eq!(n, 2);
        assert_eq!(insns.len(), 18);
        assert!(matches!(insns[3].op, Op::Jump(t) if t == 10));
        assert!(matches!(insns[4].op, Op::Jump(t) if t == 14));
        // First clone: [T, T -> clone(U), clone(U), ret]
        assert!(matches!(insns[10].op, Op::Move));
        assert!(matches!(insns[11].op, Op::CondJump(t) if t == 12));
        assert!(matches!(insns[12].op, Op::Move));
        assert!(matches!(insns[13].op, Op::Return));
        assert!(matches!(insns[15].op, Op::CondJump(t) if t == 16));
    }

    /// H -> B -> L -> H, with an extra entry E -> X -> B that bypasses H.
    /// The cycle edge L -> H is not natural: a path reaches L without H.
    #[test]
    fn splits_an_irreducible_cycle_for_its_extra_entry() {
        let mut insns = vec![
            dummy(0, 0x1000), // 0  E
            cond(4, 0x1004),  // 1  E -> H (fallthrough to X)
            dummy(1, 0x1008), // 2  X
            jump(6, 0x100c),  // 3  X -> B   (extra entry)
            dummy(2, 0x1010), // 4  H
            jump(6, 0x1014),  // 5  H -> B
            dummy(3, 0x1018), // 6  B
            jump(8, 0x101c),  // 7  B -> L
            dummy(4, 0x1020), // 8  L
            jump(4, 0x1024),  // 9  L -> H   (cycle edge)
        ];
        let old_len = insns.len();

        let splits = split_irreducible(&mut insns, 4);
        assert_eq!(splits, 1);
        assert_eq!(insns.len(), old_len + 6);

        // The bypass entry now targets the clone of B, which starts after the
        // clone of H's two instructions.
        match insns[3].op {
            Op::Jump(target) => {
                assert!(
                    target >= old_len,
                    "extra entry still targets the original body: {target}"
                );
                assert_eq!(target, old_len + 2);
            }
            ref other => panic!("X no longer ends in a jump: {other:?}"),
        }

        let cfg = rebuild_cfg(&insns);
        let dom = DominatorInfo::compute(&cfg);
        // Original blocks: H = 2, B = 3, L = 4. With the bypass moved to the
        // clone, the original L -> H edge is natural: H dominates L.
        assert!(cfg.successors[4].contains(&2));
        assert!(
            dom.dominates(2, 4),
            "the original non-dominated cycle edge is still irreducible"
        );
    }

    /// E -> H -> L -> H is a natural loop; the only non-natural-looking edge
    /// (H -> L) has no entry outside the loop at L, so nothing is split.
    #[test]
    fn leaves_a_natural_loop_alone() {
        let mut insns = vec![
            dummy(0, 0x2000), // 0 E
            jump(2, 0x2004),  // 1 E -> H
            dummy(1, 0x2008), // 2 H
            jump(4, 0x200c),  // 3 H -> L
            dummy(2, 0x2010), // 4 L
            cond(2, 0x2014),  // 5 L -> H (taken), fallthrough to exit
            ret(0x2018),      // 6 exit
        ];
        let before_len = insns.len();

        let splits = split_irreducible(&mut insns, 4);
        assert_eq!(splits, 0);
        assert_eq!(insns.len(), before_len);

        let cfg = rebuild_cfg(&insns);
        let dom = DominatorInfo::compute(&cfg);
        // H is block 1, L is block 2; the loop stays natural.
        assert!(dom.dominates(1, 2));
    }

    /// A method containing a switch must be left completely untouched.
    #[test]
    fn bails_on_switch_methods() {
        let mut insns = vec![
            dummy(0, 0x3000),
            Insn::new(Op::Switch(vec![2, 3]), vec![Operand::Local(0)], 0x3004),
            ret(0x3008),
            jump(2, 0x300c),
        ];
        let before = insns.clone();

        let splits = split_irreducible(&mut insns, 4);
        assert_eq!(splits, 0);
        assert_eq!(insns.len(), before.len());
        for (a, b) in insns.iter().zip(before.iter()) {
            assert_eq!(a.op, b.op);
            assert_eq!(a.operands, b.operands);
            assert_eq!(a.address, b.address);
        }
    }
}
