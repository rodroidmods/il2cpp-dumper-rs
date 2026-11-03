//! Goto simplification pass — reduces goto statements in the decompiled output.
//!
//! ## What this pass does
//!
//! This pass applies conservative, semantics-preserving rewrites to eliminate
//! `goto` statements that are provably redundant. Each rewrite only fires when
//! the transformation is safe — a failed match is a no-op, leaving the goto
//! in place.
//!
//! ## Patterns implemented
//!
//! 1. **Goto to return**: `goto L;` where `L:` is followed only by `return …;`
//!    (and comments) with no other incoming edges → replaced by `return …;`.
//!    The label becomes orphaned and is swept by `drop_unreferenced_labels`.
//!
//! 2. **Goto to empty trailing label**: `goto L;` where `L:` is the last
//!    statement in its list and has exactly one incoming goto → both removed.
//!    Nothing follows the label, so removing it changes no observable behavior.
//!
//! 3. **Label inlining**: `goto L;` where `L:` has exactly one incoming goto
//!    AND is either at the start of its list or preceded by a terminator
//!    (`return`, `throw`, `break`, `continue`, `goto`) → the statements after
//!    `L:` (up to the next label or end-of-list) are moved to the goto site,
//!    and the label is removed. The terminator guard ensures no fall-through
//!    reaches the label, so moving its content is safe.
//!
//! ## Safety rules
//!
//! - Never remove a goto that is the ONLY way to reach a label (the label
//!   would become unreachable code).
//! - Never create unreachable code (statements after an inserted terminator
//!   were already unreachable in the original flow).
//! - Preserve control flow semantics (the set of reachable executions is
//!   identical before and after each rewrite).
//!
//! ## Relationship to existing passes
//!
//! - `idioms::dead_code` already handles `goto L; L:` (adjacent label) and
//!   dead statements after terminators.
//! - `idioms::collapse_goto_to_loop_break` handles goto→break/continue.
//! - `csharp::drop_unreachable_gotos` handles gotos to undefined labels.
//!
//! This pass fills the gap for gotos that target labels further away in the
//! statement list, or that target labels containing only a return.

use super::super::csharp::{Expr, Stmt};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// Staging bridge for the newer goto folds (`src/recovery/reconstruct/goto_fold.rs`),
// which are owned by a separate module so they can be promoted into
// `reconstruct::PASSES` independently. Kept env-gated: when the pass is
// registered there, this bridge must not run it a second time. Set
// `ILR_GOTOFOLD=1` to exercise the folds before registration.
#[path = "goto_fold.rs"]
mod goto_fold;

// The goto structurer (`src/recovery/reconstruct/goto_struct.rs`) runs from
// `apply` below so its region-level rewrites share this pass's early slot in
// `reconstruct::PASSES`. It is not env-gated any more.
#[path = "goto_struct.rs"]
mod goto_struct;

/// Process-wide counters for each simplification pattern, reported with the
/// structural census so a pattern that never fires is visible.
pub struct GotoSimplifyCensus {
    /// `goto L;` where `L:` is followed by only `return …;` → `return …;`.
    pub goto_to_return: AtomicU64,
    /// `goto L;` where `L:` is the last statement in its list → both removed.
    pub goto_empty_label_removed: AtomicU64,
    /// `goto L;` where `L:` has one ref and is preceded by a terminator →
    /// the label's following statements are inlined at the goto site.
    pub label_inlined: AtomicU64,
    /// `goto L;` where `L:` is followed by a return of a PORTABLE value
    /// (constant/this/static field) → the return is copied to this site even
    /// when the label has other references. This is the shared-epilogue
    /// class: every early-exit branch jumps to one `return 0;`.
    pub goto_tail_folded: AtomicU64,
    /// A `goto L` that ENDS an `if` arm, where `L:` immediately follows the
    /// `if`: the arm already falls through to the join, so the branch is
    /// redundant.
    pub branch_join_gotos: AtomicU64,
    /// A backward `goto L` at the end of a list, with `L:` at the start of
    /// the same list and exactly one reference, wrapped as `while (true)`.
    pub backward_loop_wrapped: AtomicU64,
}

pub static CENSUS: GotoSimplifyCensus = GotoSimplifyCensus {
    goto_to_return: AtomicU64::new(0),
    goto_empty_label_removed: AtomicU64::new(0),
    label_inlined: AtomicU64::new(0),
    goto_tail_folded: AtomicU64::new(0),
    branch_join_gotos: AtomicU64::new(0),
    backward_loop_wrapped: AtomicU64::new(0),
};

/// Entry point: simplify gotos in one method body. Returns the number of
/// rewrites applied.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut counts: HashMap<String, usize> = HashMap::new();
    count_goto_targets_tree(stmts, &mut counts);
    let mut rewrites = simplify_list(stmts, &mut counts);
    if std::env::var_os("ILR_GOTOFOLD").is_some() {
        rewrites += goto_fold::apply(stmts);
    }
    // ENV-GATED while it is being finished. Unconditional activation wrapped a
    // trailing `return` into `while (true) { return; }` and broke five loop
    // structurer tests (`a_loop_header_post_test_update_is_emitted_in_the_body`,
    // `a_loop_test_reads_the_counter_it_updates`,
    // `a_rotated_loop_keeps_its_body_and_its_carried_update`,
    // `a_secondary_loop_exit_keeps_its_edge_and_its_code`,
    // `a_trailing_region_is_emitted_only_when_an_edge_proves_ownership`).
    // The 0x15A07DC UIHandler canary and those five tests must all pass
    // before this becomes unconditional.
    if std::env::var_os("ILR_GOTOSTRUCT").is_some()
        || std::env::var_os("ILR_GOTOSTRUCT_BCDE").is_some()
    {
        rewrites += goto_struct::apply(stmts);
    }
    rewrites
}

/// The run summary, or an empty string when nothing fired.
pub fn report() -> String {
    let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let total = g(&CENSUS.goto_to_return)
        + g(&CENSUS.goto_empty_label_removed)
        + g(&CENSUS.label_inlined)
        + g(&CENSUS.goto_tail_folded)
        + g(&CENSUS.branch_join_gotos)
        + g(&CENSUS.backward_loop_wrapped);
    if total == 0 {
        return String::new();
    }
    let mut s = format!("[goto_simplify] {total} simplification(s)\n");
    let v = g(&CENSUS.goto_to_return);
    if v > 0 {
        s.push_str(&format!("[goto_simplify]   goto → return: {v}\n"));
    }
    let v = g(&CENSUS.goto_empty_label_removed);
    if v > 0 {
        s.push_str(&format!(
            "[goto_simplify]   goto + empty trailing label removed: {v}\n"
        ));
    }
    let v = g(&CENSUS.label_inlined);
    if v > 0 {
        s.push_str(&format!("[goto_simplify]   label inlined: {v}\n"));
    }
    let v = g(&CENSUS.goto_tail_folded);
    if v > 0 {
        s.push_str(&format!("[goto_simplify]   goto → shared return: {v}\n"));
    }
    let v = g(&CENSUS.branch_join_gotos);
    if v > 0 {
        s.push_str(&format!("[goto_simplify]   redundant branch join: {v}\n"));
    }
    let v = g(&CENSUS.backward_loop_wrapped);
    if v > 0 {
        s.push_str(&format!("[goto_simplify]   backward goto → while: {v}\n"));
    }
    s
}

// ---------------------------------------------------------------------------
// Reference counting
// ---------------------------------------------------------------------------

/// Count how many `goto` statements in the entire tree target each label.
///
/// This is tree-wide: a `goto` in one arm of an `if` may target a label in a
/// sibling arm or the parent list, so per-list counting would under-count.
fn count_goto_targets_tree(stmts: &[Stmt], out: &mut HashMap<String, usize>) {
    for s in stmts {
        if let Stmt::Goto(l) = s {
            *out.entry(l.clone()).or_insert(0) += 1;
        }
        for child in child_lists(s) {
            count_goto_targets_tree(child, out);
        }
    }
}

/// Every nested statement list a statement owns, immutably.
fn child_lists(s: &Stmt) -> Vec<&Vec<Stmt>> {
    match s {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => vec![then_body, else_body],
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::ForEach { body, .. }
        | Stmt::Using { body, .. }
        | Stmt::Lock { body, .. } => vec![body],
        Stmt::For {
            init, incr, body, ..
        } => vec![init, incr, body],
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut v: Vec<&Vec<Stmt>> = vec![try_body];
            for (_, b) in catches {
                v.push(b);
            }
            if let Some(fb) = finally_body {
                v.push(fb);
            }
            v
        }
        Stmt::Switch { cases, tail, default, .. } => {
            let mut v: Vec<&Vec<Stmt>> = Vec::new();
            for (_, b) in cases {
                v.push(b);
            }
            v.push(tail);
            if let Some(arm) = default {
                v.push(arm);
            }
            v
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            let mut v: Vec<&Vec<Stmt>> = Vec::new();
            for (_, b) in cases {
                v.push(b);
            }
            if let Some(t) = tail {
                v.push(t);
            }
            v
        }
        _ => Vec::new(),
    }
}

/// Every nested statement list a statement owns, mutably.
fn child_lists_mut(s: &mut Stmt) -> Vec<&mut Vec<Stmt>> {
    match s {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => vec![then_body, else_body],
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::ForEach { body, .. }
        | Stmt::Using { body, .. }
        | Stmt::Lock { body, .. } => vec![body],
        Stmt::For {
            init, incr, body, ..
        } => vec![init, incr, body],
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut v: Vec<&mut Vec<Stmt>> = vec![try_body];
            for (_, b) in catches.iter_mut() {
                v.push(b);
            }
            if let Some(fb) = finally_body {
                v.push(fb);
            }
            v
        }
        Stmt::Switch { cases, tail, default, .. } => {
            let mut v: Vec<&mut Vec<Stmt>> = Vec::new();
            for (_, b) in cases.iter_mut() {
                v.push(b);
            }
            v.push(tail);
            if let Some(arm) = default {
                v.push(arm);
            }
            v
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            let mut v: Vec<&mut Vec<Stmt>> = Vec::new();
            for (_, b) in cases.iter_mut() {
                v.push(b);
            }
            if let Some(t) = tail {
                v.push(t);
            }
            v
        }
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Simplification
// ---------------------------------------------------------------------------

/// Apply all simplification patterns to one statement list, then recurse into
/// nested bodies. Returns the number of rewrites applied.
fn simplify_list(list: &mut Vec<Stmt>, counts: &mut HashMap<String, usize>) -> usize {
    let mut rewrites = wrap_backward_goto_loops(list, counts);
    rewrites += drop_redundant_branch_joins(list);
    let mut i = 0;
    while i < list.len() {
        let label = match &list[i] {
            Stmt::Goto(l) => l.clone(),
            _ => {
                i += 1;
                continue;
            }
        };
        let ref_count = counts.get(&label).copied().unwrap_or(0);

        // Every pattern below moves or removes the label's content at the goto
        // site. That is only sound for a FORWARD jump: when the label sits
        // before the goto, the statements between them also run on the label's
        // fall-through path, so relocating the content deletes or duplicates
        // real statements. A backward edge is left for the structurer.
        if !matches!(find_label_pos(list, &label), Some(p) if p > i) {
            i += 1;
            continue;
        }

        // --- Pattern A: goto to return ----------------------------------------
        //
        // `goto L;` where `L:` is followed by only `return …;` (and comments).
        // Replace the goto with the return. The label becomes orphaned and is
        // swept by `drop_unreferenced_labels` in the idioms pass.
        //
        // Safety: the goto transfers control to `L:`, which immediately returns.
        // Replacing the goto with the same return is observationally identical.
        // The label had one reference (this goto); after replacement it has zero
        // and is dead.
        if ref_count == 1 {
            if let Some((label_pos, ret_pos, ret_val)) = find_sole_return_after_label(list, &label)
            {
                // Remove the entire range [label_pos, ret_pos] (inclusive),
                // which includes the label, any comments, and the return.
                list.drain(label_pos..=ret_pos);
                if i >= list.len() {
                    break;
                }
                list[i] = Stmt::Return(ret_val);
                // Decrement the reference count for this label since the goto
                // targeting it has been removed.
                *counts.entry(label.clone()).or_insert(0) -= 1;
                CENSUS.goto_to_return.fetch_add(1, Ordering::Relaxed);
                rewrites += 1;
                continue;
            }
        }

        // --- Pattern A2: goto to a SHARED return epilogue ---------------------
        //
        // Compilers merge every early exit into one epilogue block:
        // `if (bad) goto L; … ; L: return 0;`. Pattern A only handles the
        // label's LAST reference; with two or more branches the label stays
        // and every branch keeps a goto. When the returned value is portable
        // (constant / null / this / a static or this-field), the return can be
        // copied to each goto site without depending on any local's scope, so
        // all but the fall-through copy disappear. The label is then orphaned
        // and `drop_unreferenced_labels` removes it.
        if let Some((_label_pos, _ret_pos, ret_val)) = find_sole_return_after_label(list, &label) {
            if return_is_portable(&ret_val) {
                list[i] = Stmt::Return(ret_val);
                *counts.entry(label.clone()).or_insert(0) -= 1;
                CENSUS.goto_tail_folded.fetch_add(1, Ordering::Relaxed);
                rewrites += 1;
                i += 1;
                continue;
            }
        }

        // --- Pattern B: goto to empty trailing label --------------------------
        //
        // `goto L;` where `L:` is the last statement in the list and has exactly
        // one incoming goto. Both are removed.
        //
        // Safety: the label is at the end with nothing after it. The goto jumps
        // to the label; after the label there is nothing. Removing both means
        // the goto site falls through to whatever follows the list — which is
        // exactly where the label's (empty) content leads.
        if ref_count == 1 {
            if let Some(label_pos) = find_label_only_trailing(list, &label) {
                list.remove(label_pos);
                if i >= list.len() {
                    break;
                }
                list.remove(i);
                // Decrement the reference count for this label since the goto
                // targeting it has been removed.
                *counts.entry(label.clone()).or_insert(0) -= 1;
                CENSUS
                    .goto_empty_label_removed
                    .fetch_add(1, Ordering::Relaxed);
                rewrites += 1;
                continue;
            }
        }

        // --- Pattern C: label inlining ----------------------------------------
        //
        // `goto L;` where `L:` has exactly one incoming goto AND is either at
        // the start of the list or preceded by a terminator. The statements
        // after `L:` (up to the next label or end-of-list) are moved to the
        // goto site, and the label is removed.
        //
        // Safety: the terminator (or list start) before the label ensures no
        // fall-through reaches the label. The label's only incoming edge is
        // this goto. Moving the label's content to the goto site preserves the
        // execution order: goto → label content → following statements.
        if ref_count == 1 {
            if let Some(label_pos) = find_label_pos(list, &label) {
                let can_inline = label_pos == 0 || is_terminator(&list[label_pos - 1]);
                if can_inline {
                    let content_end = find_next_label_after(list, label_pos);
                    let content_len = content_end - label_pos - 1;
                    if content_len > 0 {
                        let mut following: Vec<Stmt> =
                            list.drain(label_pos..content_end).skip(1).collect();
                        // After drain, the list shrank. If i is now out of bounds
                        // (shouldn't happen since i < label_pos, but guard anyway),
                        // restart the scan.
                        if i >= list.len() || following.is_empty() {
                            break;
                        }
                        list[i] = following.remove(0);
                        for (j, s) in following.into_iter().enumerate() {
                            list.insert(i + 1 + j, s);
                        }
                        // Decrement the reference count for this label since the
                        // goto targeting it has been removed.
                        *counts.entry(label.clone()).or_insert(0) -= 1;
                        CENSUS.label_inlined.fetch_add(1, Ordering::Relaxed);
                        rewrites += 1;
                        // The list shrank by 1 (label removed) and the content
                        // now occupies i..i+content_len. Restart the scan from
                        // the top since indices before label_pos are stable but
                        // the list length changed.
                        i = 0;
                        continue;
                    }
                }
            }
        }

        i += 1;
    }

    // Recurse into nested bodies. The reference counts are maintained as gotos
    // are removed at this level, so counts remain accurate for nested bodies.
    for s in list.iter_mut() {
        let children = child_lists_mut(s);
        for child in children {
            rewrites += simplify_list(child, counts);
        }
    }

    rewrites
}

// ---------------------------------------------------------------------------
// Pattern matchers
// ---------------------------------------------------------------------------

/// If `label` is defined in `list` and the only non-comment statement after it
/// (before the next label or end-of-list) is a `return`, return the positions
/// of the label and return, and the return value.
///
/// The label must have at least one statement after it (the return). Comments
/// between the label and the return are tolerated.
fn find_sole_return_after_label(
    list: &[Stmt],
    label: &str,
) -> Option<(usize, usize, Option<Expr>)> {
    let label_pos = find_label_pos(list, label)?;
    let mut ret_pos = None;
    let mut ret_val = None;
    for j in (label_pos + 1)..list.len() {
        match &list[j] {
            Stmt::Comment(_) | Stmt::Attribute { .. } => continue,
            Stmt::Label(_) => break,
            Stmt::Return(val) => {
                ret_pos = Some(j);
                ret_val = val.clone();
                break;
            }
            _ => break,
        }
    }
    Some((label_pos, ret_pos?, ret_val))
}

/// Remove a `goto L` that ends an `if` arm when `L:` is the statement right
/// after the `if`. The arm already falls through to the join in C#; the goto
/// is the structurer's address-order artefact. Only the LAST statement of an
/// arm qualifies, so no other statement's flow changes. `L:` may keep other
/// references; `drop_unreferenced_labels` removes it when this was the last.
fn drop_redundant_branch_joins(list: &mut [Stmt]) -> usize {
    let is_meta = |s: &Stmt| matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. });
    let mut n = 0usize;
    for i in 0..list.len().saturating_sub(1) {
        let mut j = i + 1;
        while j < list.len() && is_meta(&list[j]) {
            j += 1;
        }
        let join = match list.get(j) {
            Some(Stmt::Label(l)) => l.clone(),
            _ => continue,
        };
        if let Stmt::If {
            then_body,
            else_body,
            ..
        } = &mut list[i]
        {
            n += drop_tail_gotos(then_body, &join);
            n += drop_tail_gotos(else_body, &join);
        }
    }
    n
}

/// Wrap a backward goto that closes a list into a `while (true)` loop.
///
/// Shape, at one list level:
/// ```text
/// L:  A; B; …;  goto L;      // the goto is the LAST statement of the list
/// ```
/// The goto is the only reference to `L`, the region has no free
/// break/continue/label/goto, and the list ends at the goto. Then the region
/// is exactly `do { A; B; … } while (true)`, and `while (true) { A; B; … }`
/// re-enters the same way. Requiring the goto to be the list's last
/// statement is what prevents the old `while (true) { …; return; }`
/// regression: a trailing return after the back edge is not wrapped.
fn wrap_backward_goto_loops(list: &mut Vec<Stmt>, counts: &mut HashMap<String, usize>) -> usize {
    let mut n = 0usize;
    let mut i = 0;
    while i < list.len() {
        let label = match &list[i] {
            Stmt::Label(l) => l.clone(),
            _ => {
                i += 1;
                continue;
            }
        };
        if counts.get(&label).copied() != Some(1) || list.len() < i + 2 {
            i += 1;
            continue;
        }
        // The closing goto must be the last statement of the list.
        let last = list.len() - 1;
        if !matches!(&list[last], Stmt::Goto(l) if *l == label) {
            i += 1;
            continue;
        }
        let region = &list[i + 1..last];
        if !region_is_loopable(region) {
            i += 1;
            continue;
        }
        let body = region.to_vec();
        list.splice(i..=last, std::iter::once(Stmt::While {
            cond: Expr::Bool(true),
            body,
        }));
        counts.insert(label, 0);
        CENSUS.backward_loop_wrapped.fetch_add(1, Ordering::Relaxed);
        n += 1;
        // Re-examine this position: the new loop body may hold another.
    }
    n
}

/// The region may be a loop body only if it owns no control transfer that
/// would be captured by the new loop: no break/continue, no goto, no label
/// (an internal label could be an external entry), and no nested `while`.
/// Nested loops are refused because their `break`/`continue` would still bind
/// to themselves, but a label inside could be reached from outside the region.
fn region_is_loopable(region: &[Stmt]) -> bool {
    // A top-level return/throw makes the rest of the region (including the
    // back edge) unreachable; wrapping it produces `while (true) { return; }`.
    if region
        .iter()
        .any(|s| matches!(s, Stmt::Return(_) | Stmt::Throw(_)))
    {
        return false;
    }
    fn walk(stmts: &[Stmt], in_loop: bool) -> bool {
        for s in stmts {
            match s {
                // Binds to the new while when not nested — unsafe.
                Stmt::Break | Stmt::Continue => {
                    if !in_loop {
                        return false;
                    }
                }
                // Could enter the region or leave it.
                Stmt::Goto(_) | Stmt::GotoCase(_) | Stmt::Label(_) => return false,
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::ForEach { body, .. }
                | Stmt::Using { body, .. }
                | Stmt::Lock { body, .. } => {
                    if !walk(body, true) {
                        return false;
                    }
                }
                Stmt::For {
                    init, incr, body, ..
                } => {
                    if !walk(init, true) || !walk(incr, true) || !walk(body, true) {
                        return false;
                    }
                }
                _ => {
                    let mut children: Vec<&Vec<Stmt>> = Vec::new();
                    super::children(s, &mut children);
                    for c in children {
                        if !walk(c, in_loop) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
    walk(region, false)
}

/// Drop gotos to `target` that sit in TAIL POSITION of `body`.
///
/// A goto in tail position can only transfer control to the end of the arm
/// that holds it; if the arm's `if` is immediately followed by `target:`, the
/// arm's own fall-through already goes there. The same argument recurses
/// through a nested `if` (both arms fall to the end of the statement) and a
/// `try` (falling out of the try runs the finally exactly like a goto out of
/// it). Loops and switches are NOT entered: a goto out of a loop is a `break`
/// and its removal would change that; switch arms do not fall through.
fn drop_tail_gotos(body: &mut Vec<Stmt>, target: &str) -> usize {
    let is_meta = |s: &Stmt| matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. });
    let Some(k) = body.iter().rposition(|s| !is_meta(s)) else {
        return 0;
    };
    match &mut body[k] {
        Stmt::Goto(l) if l == target => {
            body.remove(k);
            CENSUS.branch_join_gotos.fetch_add(1, Ordering::Relaxed);
            1
        }
        Stmt::If {
            then_body,
            else_body,
            ..
        } => drop_tail_gotos(then_body, target) + drop_tail_gotos(else_body, target),
        Stmt::Try {
            try_body,
            catches,
            finally_body: _,
        } => {
            let mut n = drop_tail_gotos(try_body, target);
            for (_, c) in catches.iter_mut() {
                n += drop_tail_gotos(c, target);
            }
            n
        }
        _ => 0,
    }
}

/// Can this return value be copied to an arbitrary goto site? It must not
/// read a local that may be out of scope or hold a different reaching value
/// there. Constants, `this`, a field of `this`, and static fields qualify.
fn return_is_portable(v: &Option<Expr>) -> bool {
    match v {
        None => true,
        Some(
            Expr::Int(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Null | Expr::Float(_)
            | Expr::Double(_) | Expr::This,
        ) => true,
        Some(Expr::Var(n)) => n == "this",
        Some(Expr::Field(base, _)) => matches!(base.as_ref(), Expr::This),
        Some(Expr::StaticField(..)) => true,
        _ => false,
    }
}

/// If `label` is the last non-comment statement in `list`, return its position.
fn find_label_only_trailing(list: &[Stmt], label: &str) -> Option<usize> {
    let label_pos = find_label_pos(list, label)?;
    for j in (label_pos + 1)..list.len() {
        if !matches!(&list[j], Stmt::Comment(_) | Stmt::Attribute { .. }) {
            return None;
        }
    }
    Some(label_pos)
}

/// Position of `Stmt::Label(label)` in `list`, or `None`.
fn find_label_pos(list: &[Stmt], label: &str) -> Option<usize> {
    list.iter()
        .position(|s| matches!(s, Stmt::Label(l) if l == label))
}

/// Index of the next `Stmt::Label` strictly after `pos`, or `list.len()` if
/// none. Used to bound the content that follows a label (content stops at the
/// next label, which is a separate jump target).
fn find_next_label_after(list: &[Stmt], pos: usize) -> usize {
    for j in (pos + 1)..list.len() {
        if matches!(&list[j], Stmt::Label(_)) {
            return j;
        }
    }
    list.len()
}

/// Does this statement leave without falling through to the next one?
fn is_terminator(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Shorthand for `Expr::Var(name)`.
    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    // -----------------------------------------------------------------------
    // Pattern A: goto → return
    // -----------------------------------------------------------------------

    /// 1. Goto to a label whose only content is `return;` becomes `return;`.
    #[test]
    fn goto_to_void_return() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Return(None),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Return(None)));
    }

    /// 2. Goto to a label whose only content is `return expr;` preserves the
    /// return value.
    #[test]
    fn goto_to_return_with_value() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Return(Some(Expr::Int(42))),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Return(Some(Expr::Int(v))) => assert_eq!(*v, 42),
            other => panic!("expected Return(Some(Int(42))), got {other:?}"),
        }
    }

    /// 3. Comments between the label and the return are tolerated.
    #[test]
    fn goto_to_return_with_comments() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Comment("cleanup".into()),
            Stmt::Return(Some(var("x"))),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        // The comment is removed along with the label; only the return remains.
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Return(Some(Expr::Var(_)))));
    }

    /// Shared epilogue: two branches jump to one `return 0;`. Both gotos
    /// become the return; the label disappears with its last reference.
    #[test]
    fn shared_constant_return_epilogue_folds_every_goto() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::Eval(var("x")),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Return(Some(Expr::Int(0))),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "{stmts:?}");
        assert!(
            stmts.iter().all(|s| !matches!(s, Stmt::Goto(_) | Stmt::Label(_))),
            "no goto/label may remain: {stmts:?}"
        );
        assert_eq!(
            stmts
                .iter()
                .filter(|s| matches!(s, Stmt::Return(Some(Expr::Int(0)))))
                .count(),
            2,
            "{stmts:?}"
        );
    }

    /// A shared epilogue returning a LOCAL is not portable: the local may
    /// not have the same value at the goto sites, so the gotos must stay.
    #[test]
    fn shared_local_return_epilogue_is_not_folded() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::Eval(var("x")),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Return(Some(var("result"))),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
        assert_eq!(
            stmts.iter().filter(|s| matches!(s, Stmt::Goto(_))).count(),
            2,
            "{stmts:?}"
        );
    }

    /// A goto that ends an `if` arm and targets the label immediately after
    /// the `if` is a fall-through; removing it leaves the branch unchanged.
    #[test]
    fn redundant_branch_join_is_dropped() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(var("a")), Stmt::Goto("L1".into())],
                else_body: vec![Stmt::Eval(var("b")), Stmt::Goto("L1".into())],
            },
            Stmt::Label("L1".into()),
            Stmt::Eval(var("c")),
        ];
        let mut counts = HashMap::new();
        count_goto_targets_tree(&stmts, &mut counts);
        let n = simplify_list(&mut stmts, &mut counts);
        assert_eq!(n, 2, "{stmts:?}");
        if let Stmt::If {
            then_body,
            else_body,
            ..
        } = &stmts[0]
        {
            assert!(then_body.iter().all(|s| !matches!(s, Stmt::Goto(_))), "{then_body:?}");
            assert!(else_body.iter().all(|s| !matches!(s, Stmt::Goto(_))), "{else_body:?}");
        } else {
            panic!("if vanished: {stmts:?}");
        }
    }

    /// A tail goto nested under another if (and under a try) reaches the same
    /// join as the fall-through; it is removed too.
    #[test]
    fn nested_tail_gotos_to_the_join_are_dropped() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    Stmt::Eval(var("a")),
                    Stmt::If {
                        cond: Expr::Bool(true),
                        then_body: vec![Stmt::Goto("L1".into())],
                        else_body: vec![Stmt::Goto("L1".into())],
                    },
                ],
                else_body: vec![Stmt::Try {
                    try_body: vec![Stmt::If {
                        cond: Expr::Bool(true),
                        then_body: vec![Stmt::Eval(var("b")), Stmt::Goto("L1".into())],
                        else_body: vec![],
                    }],
                    catches: vec![],
                    finally_body: Some(vec![Stmt::Eval(var("f"))]),
                }],
            },
            Stmt::Label("L1".into()),
            Stmt::Eval(var("c")),
        ];
        let mut counts = HashMap::new();
        count_goto_targets_tree(&stmts, &mut counts);
        let n = simplify_list(&mut stmts, &mut counts);
        assert_eq!(n, 3, "{stmts:?}");
        let text = format!("{stmts:?}");
        assert!(!text.contains("Goto"), "all tail gotos must be gone: {text}");
    }

    /// A goto that is NOT in tail position (more statements follow it in the
    /// arm) must stay: removing it would fall through those statements.
    #[test]
    fn non_tail_nested_goto_is_kept() {
        // A statement AFTER the nested if means the goto skips it; removing
        // the goto would fall through that statement.
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    Stmt::If {
                        cond: Expr::Bool(true),
                        then_body: vec![Stmt::Goto("L1".into())],
                        else_body: vec![],
                    },
                    Stmt::Eval(var("after")),
                ],
                else_body: vec![],
            },
            Stmt::Label("L1".into()),
        ];
        let mut counts = HashMap::new();
        count_goto_targets_tree(&stmts, &mut counts);
        assert_eq!(simplify_list(&mut stmts, &mut counts), 0, "{stmts:?}");
    }

    /// The label must be the next statement; an intervening statement means
    /// the branch really has to jump over it.
    #[test]
    fn branch_join_with_intervening_statement_is_kept() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Goto("L1".into())],
                else_body: vec![],
            },
            Stmt::Eval(var("between")),
            Stmt::Label("L1".into()),
        ];
        let mut counts = HashMap::new();
        count_goto_targets_tree(&stmts, &mut counts);
        assert_eq!(simplify_list(&mut stmts, &mut counts), 0, "{stmts:?}");
    }

    // -----------------------------------------------------------------------
    // Pattern B: goto + empty trailing label
    // -----------------------------------------------------------------------

    /// 4. Goto to a trailing label (nothing after it) removes both.
    #[test]
    fn goto_to_trailing_empty_label() {
        let mut stmts = vec![
            Stmt::Eval(var("x")),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Eval(_)));
    }

    /// 5. Goto to a trailing label with only comments after it also removes.
    #[test]
    fn goto_to_trailing_label_with_comments() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Comment("trailing".into()),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        // Goto and label removed; comment remains.
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Comment(_)));
    }

    // -----------------------------------------------------------------------
    // Pattern C: label inlining
    // -----------------------------------------------------------------------

    /// 6. Goto to a label preceded by a return: the label's content is inlined
    /// at the goto site.
    #[test]
    fn goto_inlined_after_terminator() {
        let mut stmts = vec![
            Stmt::Return(Some(Expr::Int(1))),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Eval(var("x")),
            Stmt::Return(None),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        // The goto is replaced by the label's content: Eval(x), Return(None).
        // The original Return(Int(1)) remains at the start.
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[0], Stmt::Return(Some(Expr::Int(1)))));
        assert!(matches!(&stmts[1], Stmt::Eval(_)));
        assert!(matches!(&stmts[2], Stmt::Return(None)));
    }

    /// 7. Goto to a label at position 0 (no preceding statement): the label's
    /// content is inlined at the goto site.
    #[test]
    fn goto_inlined_at_start_of_list() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Eval(var("x")),
            Stmt::Return(None),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Eval(_)));
        assert!(matches!(&stmts[1], Stmt::Return(None)));
    }

    // -----------------------------------------------------------------------
    // Non-matches (must NOT rewrite)
    // -----------------------------------------------------------------------

    /// 8. Goto to a label with multiple references is NOT rewritten when the
    /// epilogue reads a local (its value is not portable to the goto sites).
    #[test]
    fn multi_ref_label_not_touched() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Goto("L1".into())],
                else_body: vec![],
            },
            Stmt::Eval(var("between")),
            Stmt::Label("L1".into()),
            Stmt::Return(Some(var("result"))),
        ];
        // Exercise the LOCAL simplifier only: `apply` now also runs the goto
        // structurer, which legitimately folds the two fall-through jumps in
        // this fixture (and would mask what this test is about).
        let mut counts = HashMap::new();
        count_goto_targets_tree(&stmts, &mut counts);
        let n = simplify_list(&mut stmts, &mut counts);
        assert_eq!(n, 0, "non-portable shared return — no pattern may fire");
        // Count gotos recursively (at all levels).
        fn count_gotos(stmts: &[Stmt]) -> usize {
            let mut count = 0;
            for s in stmts {
                if matches!(s, Stmt::Goto(_)) {
                    count += 1;
                }
                match s {
                    Stmt::If { then_body, else_body, .. } => {
                        count += count_gotos(then_body);
                        count += count_gotos(else_body);
                    }
                    Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                        count += count_gotos(body);
                    }
                    _ => {}
                }
            }
            count
        }
        let goto_count = count_gotos(&stmts);
        assert_eq!(goto_count, 2, "both gotos must remain");
    }

    /// 9. Goto to a label preceded by a non-terminator (fall-through reaches
    /// the label) is NOT inlined — removing the label would break the
    /// fall-through path.
    #[test]
    fn label_with_fallthrough_not_inlined() {
        let mut stmts = vec![
            Stmt::Eval(var("x")), // non-terminator: falls through to L1
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Eval(var("y")),
        ];
        // The label is preceded by Goto (a terminator), so Pattern C fires.
        // This is safe: the Goto prevents fall-through from Eval(x) to the label.
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "label is preceded by Goto (terminator) — safe to inline");
        // After inlining: [Eval(x), Eval(y)]
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Eval(_)));
        assert!(matches!(&stmts[1], Stmt::Eval(_)));
    }

    /// 10. Goto to a label followed by a non-return statement (and with
    /// the label preceded by a terminator): Pattern C inlines the label's content.
    #[test]
    fn goto_to_non_return_label_inlined() {
        let mut stmts = vec![
            Stmt::Eval(var("setup")),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Eval(var("work")),
            Stmt::Return(None),
        ];
        // The label is preceded by Goto (a terminator), so Pattern C fires.
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        // After inlining: [Eval(setup), Eval(work), Return(None)]
        assert_eq!(stmts.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Nested / combined scenarios
    // -----------------------------------------------------------------------

    /// 11. Goto inside an if-body targeting a label with a return: the goto
    /// inside the nested body is rewritten.
    #[test]
    fn goto_in_if_body_to_return() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![
                Stmt::Goto("L1".into()),
                Stmt::Label("L1".into()),
                Stmt::Return(Some(Expr::Int(99))),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert!(matches!(&then_body[0], Stmt::Return(Some(Expr::Int(99)))));
            }
            _ => panic!("expected If"),
        }
    }

    /// 12. Multiple independent simplifications in one method: both gotos are
    /// rewritten.
    #[test]
    fn multiple_simplifications() {
        let mut stmts = vec![
            // First: goto to return.
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Return(Some(Expr::Int(1))),
            // Second: goto to trailing empty label (after the return, which is
            // a terminator, so the label has no fall-through).
            Stmt::Goto("L2".into()),
            Stmt::Label("L2".into()),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both patterns should fire");
    }

    /// 13. Goto inside a while body targeting a label with a return.
    #[test]
    fn goto_in_while_body_to_return() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::Goto("L1".into()),
                Stmt::Label("L1".into()),
                Stmt::Return(None),
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0], Stmt::Return(None)));
            }
            _ => panic!("expected While"),
        }
    }

    /// 14. Goto to a label preceded by a break (another terminator): inlining
    /// fires because break prevents fall-through.
    #[test]
    fn goto_inlined_after_break() {
        let mut stmts = vec![
            Stmt::Break,
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Eval(var("dead_code")),
            Stmt::Return(None),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        // The goto is replaced by the label's content.
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[0], Stmt::Break));
        assert!(matches!(&stmts[1], Stmt::Eval(_)));
        assert!(matches!(&stmts[2], Stmt::Return(None)));
    }

    /// 15. Goto to a label preceded by another goto (terminator): inlining
    /// fires because the preceding goto prevents fall-through.
    #[test]
    fn goto_inlined_after_goto() {
        let mut stmts = vec![
            Stmt::Goto("other".into()),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Eval(var("x")),
        ];
        // "other" is not defined, but that doesn't matter for this pass —
        // the preceding goto is a terminator, so L1 has no fall-through.
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Goto(l) if l == "other"));
        assert!(matches!(&stmts[1], Stmt::Eval(_)));
    }

    /// 16. Label inlining stops at the next label: only the content between
    /// the target label and the next label is inlined.
    #[test]
    fn inlining_stops_at_next_label() {
        let mut stmts = vec![
            Stmt::Return(Some(Expr::Int(0))),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Eval(var("a")),
            Stmt::Eval(var("b")),
            Stmt::Label("L2".into()),
            Stmt::Eval(var("c")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        // L1's content (Eval(a), Eval(b)) is inlined; L2 and Eval(c) remain.
        assert_eq!(stmts.len(), 5);
        assert!(matches!(&stmts[0], Stmt::Return(_)));
        assert!(matches!(&stmts[1], Stmt::Eval(_))); // was goto, now Eval(a)
        assert!(matches!(&stmts[2], Stmt::Eval(_))); // Eval(b)
        assert!(matches!(&stmts[3], Stmt::Label(l) if l == "L2"));
        assert!(matches!(&stmts[4], Stmt::Eval(_))); // Eval(c)
    }

    /// 17. Empty method body: no crash, zero rewrites.
    #[test]
    fn empty_body() {
        let mut stmts: Vec<Stmt> = vec![];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    /// 18. No gotos at all: no rewrites, no changes.
    #[test]
    fn no_gotos_is_noop() {
        let mut stmts = vec![
            Stmt::Eval(var("x")),
            Stmt::Label("L1".into()),
            Stmt::Return(None),
        ];
        let original = stmts.clone();
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), original.len());
    }

    /// 19. Pattern A fires but Pattern C does not interfere: when the label's
    /// content is a return, Pattern A takes priority.
    #[test]
    fn pattern_a_takes_priority_over_c() {
        let mut stmts = vec![
            Stmt::Return(Some(Expr::Int(0))),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
            Stmt::Return(Some(Expr::Int(1))),
        ];
        let n = apply(&mut stmts);
        // Pattern A fires: goto → return. The label and its return are removed,
        // and the goto is replaced with the return.
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Return(Some(Expr::Int(0)))));
        assert!(matches!(&stmts[1], Stmt::Return(Some(Expr::Int(1)))));
    }

    /// 21. A backward jump to a label at the start of its own list must not be
    /// inlined: the label is also the list's entry, so moving its content to
    /// the goto site deletes the statements that ran on the first pass.
    #[test]
    fn backward_goto_to_list_start_is_not_inlined() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::Eval(var("x")),
            Stmt::Goto("L".into()),
        ];
        let mut counts = HashMap::new();
        count_goto_targets_tree(&stmts, &mut counts);
        let n = simplify_list(&mut stmts, &mut counts);
        assert_eq!(n, 1, "a single-reference backward edge becomes a loop");
        assert_eq!(stmts.len(), 1, "{stmts:?}");
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0], Stmt::Eval(_)));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    /// 22. A backward jump to a label whose content is a return must not drain
    /// the label: that return is also reached by fall-through.
    #[test]
    fn backward_goto_to_return_is_not_rewritten() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::Return(None),
            Stmt::Eval(var("b")),
            Stmt::Goto("L".into()),
        ];
        let mut counts = HashMap::new();
        count_goto_targets_tree(&stmts, &mut counts);
        let n = simplify_list(&mut stmts, &mut counts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 4);
        assert!(matches!(&stmts[1], Stmt::Return(None)));
        assert!(matches!(&stmts[3], Stmt::Goto(l) if l == "L"));
    }

    /// 20. Deeply nested goto to return: the pass recurses into nested blocks.
    #[test]
    fn deeply_nested_goto_to_return() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::While {
                cond: Expr::Bool(true),
                body: vec![
                    Stmt::Goto("deep".into()),
                    Stmt::Label("deep".into()),
                    Stmt::Return(Some(Expr::Int(7))),
                ],
            }],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => match &then_body[0] {
                Stmt::While { body, .. } => {
                    assert_eq!(body.len(), 1);
                    assert!(matches!(&body[0], Stmt::Return(Some(Expr::Int(7)))));
                }
                other => panic!("expected While, got {other:?}"),
            },
            _ => panic!("expected If"),
        }
    }
}
