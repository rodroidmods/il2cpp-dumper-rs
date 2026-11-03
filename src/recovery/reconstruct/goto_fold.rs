//! Conservative goto folding — raise remaining jumps into structured control
//! flow.
//!
//! The structurer and [`super::goto_simplify`] already remove the easy jumps.
//! What survives into `dump.cs` is mostly jumps whose target is the natural
//! fall-through of the statement that contains them. This pass owns those
//! shapes. Every rewrite fires only when the jump and the fall-through provably
//! land on the same label; a failed match is a no-op.
//!
//! ## Rules
//!
//! 1. **If-shell to the next label**: `if (c) goto L; L:` where the `if` has no
//!    else and its body is exactly the jump, `L` immediately follows the `if`,
//!    and no other `goto` targets `L` → the whole `if` is deleted. The branch
//!    is a no-op: both edges reach `L`. Conditions with observable evaluation
//!    (calls, dereferences, indexed reads, division) are not matched, and any
//!    label nested in an arm keeps the shell in place. Comments in the arms are
//!    preserved in place of the shell.
//!
//! 2. **Loop-tail jump to `break`**: a `goto L` in tail position of a loop body
//!    (directly, or through the arms of nested `if`s / the body of a `using` or
//!    `lock`) where `L` is the label immediately after the loop → `break`. When
//!    the condition is false the original jump fell out of the body and the
//!    loop continued; `break` under the same condition leaves the loop, which
//!    is exactly the jump's effect. `if (c) goto L;` at the end of a body and
//!    else-arm / nested-if tails are covered.
//!
//! 3. **Jump to method end → `return;`**: a `goto L` where `L` is a label at
//!    the very end of the method body (everything after it is only labels,
//!    comments or attributes) and the method has no value returns and no
//!    iterator yields → `return;`. Jumps inside a `finally` body are left
//!    alone (C# forbids leaving a `finally`).
//!
//! 4. **Fall-through jump**: a `goto L` whose next control-flow statement in
//!    the same list is `L:`, or which is in tail position of a transparent
//!    nested list (an `if` arm, `try`/`catch` body, `using`/`lock` body) whose
//!    enclosing statement is immediately followed by `L:` → the jump is
//!    deleted. Tail position of a loop is excluded here (it needs `break`;
//!    rule 2), and so are `switch` arms.
//!
//! 5. **All arms jump to the next label**: an `if`/`else` whose arms end in
//!    `goto L` while `L` immediately follows the `if` → each such jump is
//!    deleted (they all fall through to `L`). This is the "every arm" half of
//!    rule 4, applied per arm.
//!
//! 6. **Chain collapse**: `L_A: goto L_B;` where no fall-through can enter
//!    `L_A` (the preceding executable statement is a terminator) → every
//!    `goto L_A` in the method is retargeted to `L_B`, and `L_A` plus its jump
//!    are deleted. `L_B` must be defined in the same statement list.
//!
//! Rules 1–6 run to a small fixpoint because one rewrite can expose another
//! (a chain collapse turns `goto A` into `goto B`, which is now a fall-through
//! jump). The pass never deletes a label that a remaining `goto` still targets:
//! only rule 6 deletes a label, and it first retargets every reference to it.
//!
//! ## Relationship to the other passes
//!
//! [`super::goto_simplify`] handles non-local `goto → return` / label inlining;
//! `idioms::dead_code` handles the adjacent `goto L; L:` pair at list level;
//! `idioms::collapse_goto_to_loop_break` handles the direct loop-exit shape.
//! This module fills the nesting gaps: `if` shells, else arms, transparent
//! blocks, chained labels and method-end jumps.

use crate::recovery::csharp::{BinOp, Expr, Stmt};
use crate::recovery::reconstruct::{children, children_mut};
use std::collections::{HashMap, HashSet};

/// Bounded fixpoint. Each round strictly removes a jump or a label, so the
/// bound is only a guard against a future rule that rewrites without progress.
const MAX_ROUNDS: usize = 16;

/// Entry point: fold gotos in one method body. Returns the number of rewrites.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut total = 0;
    for _ in 0..MAX_ROUNDS {
        let round =
            chain_collapse(stmts) + rewrite_fallthrough(stmts) + rewrite_goto_to_return(stmts);
        if round == 0 {
            break;
        }
        total += round;
    }
    total
}

// ---------------------------------------------------------------------------
// Rules 1, 2, 4, 5: fall-through and loop-exit rewrites
// ---------------------------------------------------------------------------

/// Rewrite jumps that land on the statement list's natural fall-through.
fn rewrite_fallthrough(stmts: &mut Vec<Stmt>) -> usize {
    let mut counts = HashMap::new();
    count_goto_targets(stmts, &mut counts);
    rewrite_list(stmts, &[], false, false, &counts)
}

/// Count every `goto` targeting each label, tree-wide.
fn count_goto_targets(stmts: &[Stmt], out: &mut HashMap<String, usize>) {
    for s in stmts {
        if let Stmt::Goto(l) = s {
            *out.entry(l.clone()).or_insert(0) += 1;
        }
        let mut kids = Vec::new();
        children(s, &mut kids);
        for k in kids {
            count_goto_targets(k, out);
        }
    }
}

/// Labels reached by falling out of `list[i]`.
///
/// Statements after `i` may include a run of labels (`L1: L2: …`); all of them
/// are reached by fall-through. Comments and attributes do not execute, so they
/// are skipped. When the list ends (only comments/attributes remain), the
/// fall-through continues at the parent's `tail` labels.
fn after_labels(list: &[Stmt], i: usize, tail: &[String]) -> Vec<String> {
    let mut labels = Vec::new();
    let mut j = i + 1;
    loop {
        while j < list.len() && matches!(&list[j], Stmt::Comment(_) | Stmt::Attribute { .. }) {
            j += 1;
        }
        match list.get(j) {
            Some(Stmt::Label(l)) => {
                labels.push(l.clone());
                j += 1;
            }
            _ => break,
        }
    }
    if labels.is_empty() {
        if list[i + 1..].iter().all(is_non_executable) {
            return tail.to_vec();
        }
        return labels;
    }
    // A label run at the very end of the list falls out into the parent's
    // tail, so those labels are fall-through targets too.
    if list[j..].iter().all(is_non_executable) {
        for t in tail {
            if !labels.iter().any(|l| l == t) {
                labels.push(t.clone());
            }
        }
    }
    labels
}

fn is_non_executable(s: &Stmt) -> bool {
    matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. })
}

/// `tail` is the set of labels reached by falling out of the end of `list`.
/// `in_finally` suppresses `return` rewrites inside a `finally` body.
/// `guarded` is set when deleting jumps here could delete the only payload of
/// an enclosing `if` whose condition is not provably pure: the C# emitter's
/// empty-shell sanitizer would then drop the condition (and any call inside
/// it). No jump is deleted in that position.
fn rewrite_list(
    list: &mut Vec<Stmt>,
    tail: &[String],
    in_finally: bool,
    guarded: bool,
    counts: &HashMap<String, usize>,
) -> usize {
    let mut rewrites = 0;
    let mut i = 0;
    while i < list.len() {
        let after = after_labels(list, i, tail);

        // Rule 4 (same-list half): the jump's target is the next label.
        if !guarded {
            if let Stmt::Goto(l) = &list[i] {
                if after.iter().any(|a| a == l) {
                    list.remove(i);
                    rewrites += 1;
                    continue;
                }
            }
        }

        // Rule 1: `if (c) goto L;` shell whose label is the fall-through.
        if !guarded {
            if let Some(kept) = if_shell_remnant(&list[i], &after, counts) {
                list.splice(i..=i, kept);
                rewrites += 1;
                continue;
            }
        }

        match &mut list[i] {
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                // Rules 4 and 5: tail jumps in either arm fall through to the
                // label after the `if`. An emptied arm only endangers an
                // impure condition when that arm consists solely of jumps and
                // the OTHER arm is empty too: then the sanitizer drops the
                // shell and loses the call in the condition. An arm with any
                // real statement keeps the shell alive, so its jumps are free
                // to go even under an impure condition.
                let has_payload = |b: &[Stmt]| {
                    b.iter().any(|s| {
                        !matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. } | Stmt::Label(_))
                    })
                };
                let only_jumps = |b: &[Stmt]| {
                    b.iter().all(|s| {
                        matches!(
                            s,
                            Stmt::Goto(_)
                                | Stmt::Comment(_)
                                | Stmt::Attribute { .. }
                                | Stmt::Label(_)
                        )
                    })
                };
                let cond_impure = !is_pure_condition(cond);
                let guard_then =
                    guarded || (cond_impure && !has_payload(else_body) && only_jumps(then_body));
                let guard_else =
                    guarded || (cond_impure && !has_payload(then_body) && only_jumps(else_body));
                rewrites += rewrite_list(then_body, &after, in_finally, guard_then, counts);
                rewrites += rewrite_list(else_body, &after, in_finally, guard_else, counts);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. } => {
                // Rule 2: a tail jump to the loop's exit label is a `break`.
                // Fall-through from the body goes to the loop header, so the
                // body itself is visited with an empty tail. A loop is never
                // noise, so its body is not guarded.
                rewrites += rewrite_loop_tail(body, &after);
                rewrites += rewrite_body_jumps_to_break(body, &after);
                rewrites += rewrite_list(body, &[], in_finally, false, counts);
            }
            Stmt::For {
                init, incr, body, ..
            } => {
                rewrites += rewrite_loop_tail(body, &after);
                rewrites += rewrite_body_jumps_to_break(body, &after);
                rewrites += rewrite_list(init, &[], in_finally, false, counts);
                rewrites += rewrite_list(incr, &[], in_finally, false, counts);
                rewrites += rewrite_list(body, &[], in_finally, false, counts);
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                rewrites += rewrite_list(body, &after, in_finally, false, counts);
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                rewrites += rewrite_list(try_body, &after, in_finally, false, counts);
                for (_, body) in catches.iter_mut() {
                    rewrites += rewrite_list(body, &after, in_finally, false, counts);
                }
                if let Some(fb) = finally_body {
                    rewrites += rewrite_list(fb, &after, true, false, counts);
                }
            }
            Stmt::Switch { cases, tail: st, default, .. } => {
                // Falling out of a case body runs the next case; `break` is
                // required there, so no tail labels are propagated.
                for (_, body) in cases.iter_mut() {
                    rewrites += rewrite_list(body, &[], in_finally, false, counts);
                }
                rewrites += rewrite_list(st, &after, in_finally, false, counts);
                if let Some(arm) = default {
                    rewrites += rewrite_list(arm, &after, in_finally, false, counts);
                }
            }
            Stmt::StringSwitch { cases, tail: st, .. } => {
                for (_, body) in cases.iter_mut() {
                    rewrites += rewrite_list(body, &[], in_finally, false, counts);
                }
                if let Some(t) = st {
                    rewrites += rewrite_list(t, &after, in_finally, false, counts);
                }
            }
            _ => {}
        }
        i += 1;
    }
    rewrites
}

/// Rule 1: matcher and remnant builder.
///
/// Returns the comments to keep in place of the `if` shell when the shell can
/// be deleted. Every executable statement in either arm except the single jump
/// disqualifies the rewrite, and so does any label or attribute (they must be
/// preserved). `counts` is computed once per round; it can only over-count
/// references because rewrites remove gotos, which is the conservative
/// direction for the "no other goto targets L" guard.
fn if_shell_remnant(
    stmt: &Stmt,
    after: &[String],
    counts: &HashMap<String, usize>,
) -> Option<Vec<Stmt>> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };
    if after.is_empty() || !is_pure_condition(cond) {
        return None;
    }

    let mut target = None;
    let mut kept = Vec::new();
    for s in then_body {
        match s {
            Stmt::Comment(c) => kept.push(Stmt::Comment(c.clone())),
            Stmt::Goto(l) if target.is_none() => target = Some(l.clone()),
            _ => return None,
        }
    }
    let target = target?;
    if !after.iter().any(|a| *a == target) {
        return None;
    }
    if counts.get(&target).copied().unwrap_or(0) != 1 {
        return None;
    }
    // An else arm with control flow would be dropped; comments are harmless.
    for s in else_body {
        match s {
            Stmt::Comment(c) => kept.push(Stmt::Comment(c.clone())),
            _ => return None,
        }
    }
    Some(kept)
}

/// A condition whose evaluation is a plain value computation: no call, no
/// allocation, no dereference/index (both can throw), no division/remainder.
/// Deleting an `if` shell must not delete an observable effect or an exception.
fn is_pure_condition(e: &Expr) -> bool {
    match e {
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::MissingAbiArg
        | Expr::Unknown
        | Expr::StaticField(_, _) => true,
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) => is_pure_condition(a),
        Expr::Field(base, _) => is_pure_condition(base),
        // A deref is a load, but it can FAULT on a null/invalid base, and
        // deleting the only branch that evaluates it would drop that fault.
        // Whole-shell deletion (rule 1) therefore still refuses it. Deleting
        // only a jump INSIDE an arm is allowed when the other arm keeps the
        // shell alive — see the per-arm guard in `rewrite_list`.
        Expr::Bin(op, a, b) => {
            !matches!(op, BinOp::Div | BinOp::Rem)
                && is_pure_condition(a)
                && is_pure_condition(b)
        }
        Expr::Cmp(_, a, b) => is_pure_condition(a) && is_pure_condition(b),
        _ => false,
    }
}

/// Rule 2: rewrite tail jumps in a loop body targeting the loop's exit label.
fn rewrite_loop_tail(body: &mut Vec<Stmt>, exit_labels: &[String]) -> usize {
    if exit_labels.is_empty() {
        return 0;
    }
    rewrite_tail_list(body, exit_labels)
}

/// Rule 2b: ANY jump inside the loop body whose target is the loop's exit
/// label is a `break`, at any nesting depth. `break` leaves exactly this loop
/// and continues at the label that immediately follows it, which is where the
/// jump went. Exclusions:
/// * nested loops — a `break` there would leave the nested loop, not this
///   one (the nested loop's own visit handles its exits);
/// * switch arms — `break` binds to the switch, not the loop.
fn rewrite_body_jumps_to_break(list: &mut Vec<Stmt>, exit_labels: &[String]) -> usize {
    if exit_labels.is_empty() {
        return 0;
    }
    let mut n = 0;
    for s in list.iter_mut() {
        match s {
            Stmt::Goto(l) if exit_labels.iter().any(|e| e == l) => {
                *s = Stmt::Break;
                n += 1;
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                n += rewrite_body_jumps_to_break(then_body, exit_labels);
                n += rewrite_body_jumps_to_break(else_body, exit_labels);
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                n += rewrite_body_jumps_to_break(try_body, exit_labels);
                for (_, b) in catches.iter_mut() {
                    n += rewrite_body_jumps_to_break(b, exit_labels);
                }
                if let Some(f) = finally_body {
                    n += rewrite_body_jumps_to_break(f, exit_labels);
                }
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                n += rewrite_body_jumps_to_break(body, exit_labels);
            }
            // Nested loops and switches own their `break`.
            _ => {}
        }
    }
    n
}

/// Rewrite the last executable statement of `list` (labels, comments and
/// attributes do not execute). Returns the number of jumps converted.
fn rewrite_tail_list(list: &mut Vec<Stmt>, targets: &[String]) -> usize {
    let last = (0..list.len()).rev().find(|&j| {
        !matches!(
            &list[j],
            Stmt::Comment(_) | Stmt::Attribute { .. } | Stmt::Label(_)
        )
    });
    match last {
        Some(j) => rewrite_tail_stmt(&mut list[j], targets),
        None => 0,
    }
}

fn rewrite_tail_stmt(stmt: &mut Stmt, targets: &[String]) -> usize {
    match stmt {
        Stmt::Goto(l) if targets.iter().any(|t| t == l) => {
            *stmt = Stmt::Break;
            1
        }
        // `break` inside an `if` arm leaves the enclosing loop.
        Stmt::If {
            then_body,
            else_body,
            ..
        } => rewrite_tail_list(then_body, targets) + rewrite_tail_list(else_body, targets),
        // `break` inside a using/lock body still leaves the enclosing loop
        // (Dispose / Monitor.Exit run in both the goto and the break).
        Stmt::Using { body, .. } | Stmt::Lock { body, .. } => rewrite_tail_list(body, targets),
        // A switch arm's `break` exits the switch, and a nested loop's `break`
        // exits the inner loop: neither is the jump's destination.
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Rule 3: goto to method end → return;
// ---------------------------------------------------------------------------

/// Replace jumps to a trailing method label with `return;` when the method is
/// void-like. The method has no signature in the AST, so "void-like" means no
/// value return and no iterator yield anywhere in the body.
fn rewrite_goto_to_return(stmts: &mut Vec<Stmt>) -> usize {
    if !is_void_like(stmts) {
        return 0;
    }
    let mut end_labels = HashSet::new();
    for i in 0..stmts.len() {
        if let Stmt::Label(l) = &stmts[i] {
            if stmts[i + 1..].iter().all(|s| {
                matches!(
                    s,
                    Stmt::Label(_) | Stmt::Comment(_) | Stmt::Attribute { .. }
                )
            }) {
                end_labels.insert(l.clone());
            }
        }
    }
    if end_labels.is_empty() {
        return 0;
    }
    goto_return_list(stmts, &end_labels, false)
}

fn is_void_like(stmts: &[Stmt]) -> bool {
    for s in stmts {
        match s {
            Stmt::Return(Some(_)) | Stmt::YieldReturn(_) | Stmt::YieldBreak => return false,
            _ => {}
        }
        let mut kids = Vec::new();
        children(s, &mut kids);
        for k in kids {
            if !is_void_like(k) {
                return false;
            }
        }
    }
    true
}

fn goto_return_list(list: &mut [Stmt], labels: &HashSet<String>, in_finally: bool) -> usize {
    let mut n = 0;
    for s in list.iter_mut() {
        match s {
            Stmt::Goto(l) if !in_finally && labels.contains(l.as_str()) => {
                *s = Stmt::Return(None);
                n += 1;
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                n += goto_return_list(then_body, labels, in_finally);
                n += goto_return_list(else_body, labels, in_finally);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. } => {
                n += goto_return_list(body, labels, in_finally);
            }
            Stmt::For {
                init, incr, body, ..
            } => {
                n += goto_return_list(init, labels, in_finally);
                n += goto_return_list(incr, labels, in_finally);
                n += goto_return_list(body, labels, in_finally);
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                n += goto_return_list(body, labels, in_finally);
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                n += goto_return_list(try_body, labels, in_finally);
                for (_, body) in catches.iter_mut() {
                    n += goto_return_list(body, labels, in_finally);
                }
                if let Some(fb) = finally_body {
                    n += goto_return_list(fb, labels, true);
                }
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, body) in cases.iter_mut() {
                    n += goto_return_list(body, labels, in_finally);
                }
                n += goto_return_list(tail, labels, in_finally);
                if let Some(arm) = default {
                    n += goto_return_list(arm, labels, in_finally);
                }
            }
            Stmt::StringSwitch { cases, tail, .. } => {
                for (_, body) in cases.iter_mut() {
                    n += goto_return_list(body, labels, in_finally);
                }
                if let Some(t) = tail {
                    n += goto_return_list(t, labels, in_finally);
                }
            }
            _ => {}
        }
    }
    n
}

// ---------------------------------------------------------------------------
// Rule 6: chain collapse (`L_A: goto L_B;`)
// ---------------------------------------------------------------------------

/// Collapse every eligible label chain in the tree. Each iteration removes one
/// label, so this terminates.
fn chain_collapse(stmts: &mut Vec<Stmt>) -> usize {
    let mut total = 0;
    loop {
        let mut found = None;
        find_chain(stmts, &mut found);
        let Some((a, b)) = found else { break };
        // Retarget every reference first: the label may be targeted from a
        // nested list (a switch arm, an if arm) that is not a sibling of the
        // label's own list.
        retarget_gotos(stmts, &a, &b);
        if !remove_chain(stmts, &a) {
            // The AST did not contain the shape after retargeting; leave the
            // retargeted edges alone (semantically identical) and stop.
            break;
        }
        total += 1;
    }
    total
}

/// Find the first `L_A: goto L_B;` chain, depth-first.
fn find_chain(list: &[Stmt], out: &mut Option<(String, String)>) {
    if out.is_some() {
        return;
    }
    for p in 0..list.len() {
        let Stmt::Label(a) = &list[p] else { continue };
        if !no_fallthrough_into(list, p) {
            continue;
        }
        let end = next_label_pos(list, p);
        let mut b = None;
        let mut ok = true;
        for s in &list[p + 1..end] {
            match s {
                Stmt::Comment(_) => {}
                Stmt::Goto(target) if b.is_none() => b = Some(target.clone()),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let Some(b) = b else { continue };
        if b == *a {
            continue;
        }
        if find_label_pos(list, &b).is_none() {
            continue;
        }
        *out = Some((a.clone(), b));
        return;
    }
    for s in list {
        let mut kids = Vec::new();
        children(s, &mut kids);
        for k in kids {
            find_chain(k, out);
            if out.is_some() {
                return;
            }
        }
    }
}

/// True when no execution can fall into `list[pos]` from the preceding
/// statement. A preceding terminator blocks fall-through; the start of a list
/// does not (the enclosing construct falls in), and a preceding label does not
/// (it may itself be jumped to).
fn no_fallthrough_into(list: &[Stmt], pos: usize) -> bool {
    let mut j = pos;
    while j > 0 {
        j -= 1;
        match &list[j] {
            Stmt::Comment(_) | Stmt::Attribute { .. } => continue,
            Stmt::Label(_) => return false,
            Stmt::Return(_)
            | Stmt::Throw(_)
            | Stmt::Goto(_)
            | Stmt::Break
            | Stmt::Continue
            | Stmt::GotoCase(_)
            | Stmt::YieldBreak => return true,
            _ => return false,
        }
    }
    false
}

fn retarget_gotos(list: &mut [Stmt], from: &str, to: &str) -> usize {
    let mut n = 0;
    for s in list.iter_mut() {
        if let Stmt::Goto(l) = s {
            if l == from {
                *l = to.to_string();
                n += 1;
            }
            continue;
        }
        for k in children_mut(s) {
            n += retarget_gotos(k, from, to);
        }
    }
    n
}

/// Remove `L_A:` and the sole jump after it, once every `goto L_A` has been
/// retargeted. Returns whether the label was removed.
fn remove_chain(list: &mut Vec<Stmt>, a: &str) -> bool {
    if let Some(p) = find_label_pos(list, a) {
        let end = next_label_pos(list, p);
        let mut goto_pos = None;
        let mut ok = true;
        for j in p + 1..end {
            match &list[j] {
                Stmt::Comment(_) => {}
                Stmt::Goto(_) if goto_pos.is_none() => goto_pos = Some(j),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            if let Some(j) = goto_pos {
                list.remove(j);
                list.remove(p);
                return true;
            }
        }
    }
    for s in list.iter_mut() {
        for k in children_mut(s) {
            if remove_chain(k, a) {
                return true;
            }
        }
    }
    false
}

fn find_label_pos(list: &[Stmt], label: &str) -> Option<usize> {
    list.iter()
        .position(|s| matches!(s, Stmt::Label(l) if l == label))
}

fn next_label_pos(list: &[Stmt], pos: usize) -> usize {
    for j in (pos + 1)..list.len() {
        if matches!(&list[j], Stmt::Label(_)) {
            return j;
        }
    }
    list.len()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::{CmpKind, Stmt};

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    fn eval(name: &str) -> Stmt {
        Stmt::Eval(var(name))
    }

    fn cond() -> Expr {
        Expr::Cmp(CmpKind::Eq, Box::new(var("c")), Box::new(Expr::Int(0)))
    }

    fn if_goto(target: &str) -> Stmt {
        Stmt::If {
            cond: cond(),
            then_body: vec![Stmt::Goto(target.to_string())],
            else_body: vec![],
        }
    }

    fn goto_count(stmts: &[Stmt]) -> usize {
        let mut n = 0;
        for s in stmts {
            if matches!(s, Stmt::Goto(_)) {
                n += 1;
            }
            let mut kids = Vec::new();
            children(s, &mut kids);
            for k in kids {
                n += goto_count(k);
            }
        }
        n
    }

    fn break_count(stmts: &[Stmt]) -> usize {
        let mut n = 0;
        for s in stmts {
            if matches!(s, Stmt::Break) {
                n += 1;
            }
            let mut kids = Vec::new();
            children(s, &mut kids);
            for k in kids {
                n += break_count(k);
            }
        }
        n
    }

    // ------------------------------------------------------------------
    // Rule 1: if-shell to the next label
    // ------------------------------------------------------------------

    #[test]
    fn if_shell_to_next_label_is_deleted() {
        let mut stmts = vec![if_goto("L"), Stmt::Label("L".into()), eval("after")];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Label(l) if l == "L"));
        assert!(matches!(&stmts[1], Stmt::Eval(_)));
    }

    #[test]
    fn if_shell_comments_are_preserved() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Comment("then".into()), Stmt::Goto("L".into())],
                else_body: vec![Stmt::Comment("else".into())],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 3, "both comments kept in place of the shell");
        assert!(matches!(&stmts[0], Stmt::Comment(_)));
        assert!(matches!(&stmts[1], Stmt::Comment(_)));
        assert!(matches!(&stmts[2], Stmt::Label(_)));
    }

    #[test]
    fn if_shell_with_call_condition_is_kept() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Call {
                    recv: None,
                    name: "Check".into(),
                    args: vec![],
                },
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        // Neither the shell nor the jump may go: deleting the jump would leave
        // an empty shell, which the emitter's sanitizer drops together with
        // the call in the condition.
        assert_eq!(n, 0);
        match &stmts[0] {
            Stmt::If { then_body, .. } => assert!(matches!(&then_body[0], Stmt::Goto(_))),
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn if_shell_with_deref_condition_is_kept() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(Expr::RawDeref(Box::new(var("p")), 8, 4)),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        // The dereference can fault, so the shell and its jump stay.
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn if_shell_with_foreign_goto_keeps_label_and_shell() {
        let mut stmts = vec![
            if_goto("L"),
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![],
            },
            eval("between"),
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        // The first shell could only be deleted if L had no other reference;
        // the second shell is not adjacent to L. Neither fires.
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
        assert_eq!(goto_count(&stmts), 2);
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "L")));
    }

    #[test]
    fn if_shell_with_nested_label_is_kept() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Label("X".into()), Stmt::Goto("L".into())],
                else_body: vec![],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        // The shell cannot be deleted (it would drop L_X), but the jump after
        // the label is still a fall-through and is removed; L_X survives.
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Label(l) if l == "X"));
            }
            other => panic!("expected If, got {other:?}"),
        }
        assert!(
            stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "L")),
            "the fall-through label is never deleted"
        );
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn if_shell_with_nonempty_else_is_kept() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![Stmt::Goto("M".into())],
            },
            Stmt::Label("L".into()),
            Stmt::Label("M".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        // Both jumps fall through (L and M are the label run after the if),
        // but the shell itself is kept because both arms carry control flow.
        assert_eq!(n, 2);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
        assert_eq!(goto_count(&stmts), 0);
    }

    // ------------------------------------------------------------------
    // Rule 2: loop tail → break
    // ------------------------------------------------------------------

    #[test]
    fn loop_tail_if_goto_becomes_break() {
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![eval("work"), if_goto("L")],
            },
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 2);
                match &body[1] {
                    Stmt::If { then_body, .. } => {
                        assert!(matches!(then_body.last(), Some(Stmt::Break)));
                    }
                    other => panic!("expected trailing If, got {other:?}"),
                }
            }
            other => panic!("expected While, got {other:?}"),
        }
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "L")));
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn loop_tail_else_arm_goto_becomes_break() {
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![Stmt::If {
                    cond: cond(),
                    then_body: vec![eval("work")],
                    else_body: vec![Stmt::Goto("L".into())],
                }],
            },
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(break_count(&stmts), 1);
    }

    #[test]
    fn loop_tail_nested_if_goto_becomes_break() {
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![eval("work"), if_goto("L")],
            },
            Stmt::Label("L".into()),
        ];
        // The direct tail shape; a doubly nested if is exercised by the
        // else-arm test above. Ensure the `if` survives with a break inside.
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => match body.last() {
                Some(Stmt::If { then_body, .. }) => {
                    assert!(matches!(then_body.last(), Some(Stmt::Break)));
                }
                other => panic!("expected trailing If, got {other:?}"),
            },
            other => panic!("expected While, got {other:?}"),
        }
    }

    #[test]
    fn loop_tail_direct_goto_becomes_break() {
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![eval("work"), Stmt::Goto("L".into())],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(body.last(), Some(Stmt::Break)));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    #[test]
    fn loop_tail_goto_to_other_label_is_kept() {
        // M is before the loop, so it is not the loop's fall-through exit.
        let mut stmts = vec![
            Stmt::Label("M".into()),
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![eval("work"), Stmt::Goto("M".into())],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
        assert_eq!(break_count(&stmts), 0);
    }

    #[test]
    fn nested_loop_goto_is_not_a_break() {
        // The goto exits both loops; `break` would exit only the inner one.
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![
                    Stmt::While {
                        cond: Expr::Bool(true),
                        body: vec![Stmt::Goto("L".into())],
                    },
                    eval("after_inner"),
                ],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
        assert_eq!(break_count(&stmts), 0);
    }

    #[test]
    fn switch_arm_goto_is_not_a_break() {
        // The goto exits the loop, but a `break` in the switch arm exits only
        // the switch. The jump must stay.
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![Stmt::Switch {
                    disc: "v".into(),
                    cases: vec![(0, vec![Stmt::Goto("L".into())])],
                    tail: vec![],
                
                    default: None,}],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
        assert_eq!(break_count(&stmts), 0);
    }

    // ------------------------------------------------------------------
    // Rule 3: goto to method end → return
    // ------------------------------------------------------------------

    #[test]
    fn goto_to_method_end_becomes_return() {
        let mut stmts = vec![eval("work"), Stmt::Goto("L".into()), Stmt::Label("L".into())];
        let n = rewrite_goto_to_return(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[1], Stmt::Return(None)));
        assert!(
            stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "L")),
            "the label is kept"
        );
    }

    #[test]
    fn goto_to_method_end_with_value_return_is_kept() {
        let mut stmts = vec![
            Stmt::Goto("L".into()),
            Stmt::Return(Some(Expr::Int(1))),
            Stmt::Label("L".into()),
        ];
        let n = rewrite_goto_to_return(&mut stmts);
        assert_eq!(n, 0, "non-void methods must not get a bare return");
    }

    #[test]
    fn goto_to_method_end_in_iterator_is_kept() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::YieldReturn(Expr::Int(1))],
                else_body: vec![],
            },
            Stmt::Goto("L".into()),
            Stmt::Label("L".into()),
        ];
        let n = rewrite_goto_to_return(&mut stmts);
        assert_eq!(n, 0, "`return;` is not legal in an iterator");
    }

    #[test]
    fn goto_inside_finally_is_kept() {
        let mut stmts = vec![
            Stmt::Try {
                try_body: vec![eval("work")],
                catches: vec![],
                finally_body: Some(vec![Stmt::Goto("L".into())]),
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_goto_to_return(&mut stmts);
        assert_eq!(n, 0, "C# cannot leave a finally body");
    }

    #[test]
    fn goto_to_non_trailing_label_is_kept() {
        let mut stmts = vec![
            Stmt::Goto("L".into()),
            Stmt::Label("L".into()),
            eval("still runs"),
        ];
        let n = rewrite_goto_to_return(&mut stmts);
        assert_eq!(n, 0, "reaching L runs the following statement");
    }

    #[test]
    fn goto_into_trailing_label_run_becomes_return() {
        let mut stmts = vec![
            Stmt::Goto("L2".into()),
            Stmt::Label("L1".into()),
            Stmt::Label("L2".into()),
            Stmt::Comment("end".into()),
        ];
        let n = rewrite_goto_to_return(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[0], Stmt::Return(None)));
    }

    // ------------------------------------------------------------------
    // Rules 4/5: fall-through gotos in arms
    // ------------------------------------------------------------------

    #[test]
    fn tail_goto_in_if_arm_is_fallthrough() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![eval("work"), Stmt::Goto("L".into())],
                else_body: vec![],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                assert_eq!(then_body.len(), 1);
                assert!(else_body.is_empty());
            }
            other => panic!("expected If, got {other:?}"),
        }
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "L")));
    }

    #[test]
    fn non_tail_arm_goto_is_kept() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Goto("L".into()), eval("after_jump")],
                else_body: vec![],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 0, "deleting the jump would run the intervening statement");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn arm_tail_goto_falls_through_label_run_into_parent_label() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Goto("P".into()), Stmt::Label("L1".into())],
                else_body: vec![],
            },
            Stmt::Label("P".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "P")));
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Label(l) if l == "L1"));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn label_run_without_parent_label_is_not_a_fallthrough() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Goto("P".into()), Stmt::Label("L1".into())],
                else_body: vec![],
            },
            eval("between"),
            Stmt::Label("P".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 0, "falling out of the arm runs `between`, not P");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn same_list_commented_fallthrough_is_deleted() {
        let mut stmts = vec![
            Stmt::Goto("L".into()),
            Stmt::Comment("between".into()),
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "L")));
    }

    #[test]
    fn both_if_arms_tail_gotos_are_deleted() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![eval("a"), Stmt::Goto("L".into())],
                else_body: vec![eval("b"), Stmt::Goto("L".into())],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 2);
        assert_eq!(goto_count(&stmts), 0);
        match &stmts[0] {
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                assert_eq!(then_body.len(), 1);
                assert_eq!(else_body.len(), 1);
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn only_the_matching_arm_goto_is_deleted() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![eval("a"), Stmt::Goto("L".into())],
                else_body: vec![eval("b"), Stmt::Goto("M".into())],
            },
            Stmt::Label("L".into()),
            eval("mid"),
            Stmt::Label("M".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn try_and_using_tails_fall_through_to_the_next_label() {
        let mut stmts = vec![
            Stmt::Try {
                try_body: vec![eval("t"), Stmt::Goto("L".into())],
                catches: vec![("Exception".into(), vec![eval("c"), Stmt::Goto("L".into())])],
                finally_body: None,
            },
            Stmt::Label("L".into()),
        ];
        assert_eq!(rewrite_fallthrough(&mut stmts), 2);
        assert_eq!(goto_count(&stmts), 0);

        let mut using = vec![
            Stmt::Using {
                decl: Box::new(eval("open")),
                body: vec![eval("u"), Stmt::Goto("L".into())],
            },
            Stmt::Label("L".into()),
        ];
        assert_eq!(rewrite_fallthrough(&mut using), 1);
        assert_eq!(goto_count(&using), 0);
    }

    #[test]
    fn nested_if_arm_tail_goto_to_label_after_outer_if_is_deleted() {
        // Every path through the nested arms reaches `L` immediately after the
        // outer `if`; the jump is a fall-through and can go.
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::If {
                    cond: cond(),
                    then_body: vec![eval("work"), Stmt::Goto("L".into())],
                    else_body: vec![],
                }],
                else_body: vec![],
            },
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn nested_arm_goto_with_intervening_statement_is_kept() {
        // Jumping to `L` skips `only_when_false`; deleting the jump would run
        // it on the true path.
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![
                    Stmt::If {
                        cond: cond(),
                        then_body: vec![Stmt::Goto("L".into())],
                        else_body: vec![],
                    },
                    eval("only_when_false"),
                ],
                else_body: vec![],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    // ------------------------------------------------------------------
    // Rule 6: chain collapse
    // ------------------------------------------------------------------

    #[test]
    fn chain_collapse_retargets_and_removes_label() {
        let mut stmts = vec![
            Stmt::Goto("A".into()),
            Stmt::Label("A".into()),
            Stmt::Goto("B".into()),
            Stmt::Label("B".into()),
            eval("body"),
        ];
        let n = chain_collapse(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[0], Stmt::Goto(l) if l == "B"));
        assert!(!stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "A")));
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "B")));
    }

    #[test]
    fn chain_collapse_keeps_label_with_fallthrough_entry() {
        let mut stmts = vec![
            eval("fall in"),
            Stmt::Label("A".into()),
            Stmt::Goto("B".into()),
            Stmt::Label("B".into()),
        ];
        let n = chain_collapse(&mut stmts);
        assert_eq!(n, 0, "deleting L_A would drop the fall-through path");
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "A")));
    }

    #[test]
    fn chain_collapse_keeps_label_with_extra_statement() {
        let mut stmts = vec![
            Stmt::Goto("A".into()),
            Stmt::Label("A".into()),
            Stmt::Goto("B".into()),
            eval("not just a jump"),
            Stmt::Label("B".into()),
        ];
        let n = chain_collapse(&mut stmts);
        assert_eq!(n, 0);
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "A")));
    }

    #[test]
    fn chain_collapse_retargets_nested_goto() {
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Goto("A".into())],
                else_body: vec![],
            },
            Stmt::Goto("A".into()),
            Stmt::Label("A".into()),
            Stmt::Goto("B".into()),
            Stmt::Label("B".into()),
        ];
        let n = chain_collapse(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Goto(l) if l == "B"));
            }
            other => panic!("expected If, got {other:?}"),
        }
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "B")));
    }

    #[test]
    fn chain_collapse_requires_a_defined_target() {
        let mut stmts = vec![
            Stmt::Goto("A".into()),
            Stmt::Label("A".into()),
            Stmt::Goto("B".into()),
        ];
        let n = chain_collapse(&mut stmts);
        assert_eq!(n, 0);
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "A")));
    }

    #[test]
    fn chain_collapse_rejects_self_loop() {
        let mut stmts = vec![
            Stmt::Goto("A".into()),
            Stmt::Label("A".into()),
            Stmt::Goto("A".into()),
        ];
        let n = chain_collapse(&mut stmts);
        assert_eq!(n, 0);
        assert!(stmts.iter().any(|s| matches!(s, Stmt::Label(l) if l == "A")));
    }

    // ------------------------------------------------------------------
    // Whole-pass behaviour
    // ------------------------------------------------------------------

    #[test]
    fn apply_runs_to_fixpoint_across_chain_and_fallthrough() {
        let mut stmts = vec![
            Stmt::Goto("A".into()),
            Stmt::Label("A".into()),
            Stmt::Goto("B".into()),
            Stmt::Label("B".into()),
            Stmt::Return(None),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "chain collapse then fall-through deletion");
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn apply_keeps_foreign_target_labels() {
        let mut stmts = vec![
            Stmt::Goto("A".into()),
            Stmt::Label("A".into()),
            Stmt::Goto("B".into()),
            Stmt::Label("B".into()),
            Stmt::Goto("C".into()),
            Stmt::Label("C".into()),
            Stmt::Return(None),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 2);
        // Every label still targeted by a jump must still exist. After the
        // fixpoint nothing should dangle.
        let mut labels = HashSet::new();
        fn collect(list: &[Stmt], out: &mut HashSet<String>) {
            for s in list {
                if let Stmt::Label(l) = s {
                    out.insert(l.clone());
                }
            }
        }
        collect(&stmts, &mut labels);
        fn check(list: &[Stmt], labels: &HashSet<String>) -> bool {
            for s in list {
                if let Stmt::Goto(l) = s {
                    if !labels.contains(l) {
                        return false;
                    }
                }
                let mut kids = Vec::new();
                children(s, &mut kids);
                for k in kids {
                    if !check(k, labels) {
                        return false;
                    }
                }
            }
            true
        }
        assert!(check(&stmts, &labels), "a goto targets a deleted label");
    }

    #[test]
    fn empty_body_is_noop() {
        let mut stmts = vec![];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn side_effectful_arm_statements_are_never_moved() {
        // The if shell is not eligible (body has effects); nothing moves.
        let mut stmts = vec![
            Stmt::If {
                cond: cond(),
                then_body: vec![eval("effect"), Stmt::Goto("L".into())],
                else_body: vec![Stmt::Goto("L".into())],
            },
            Stmt::Label("L".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        // Both arm-tail jumps fall through to L and are removed; the
        // side-effecting statement in the then arm stays where it was.
        assert_eq!(n, 2);
        match &stmts[0] {
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                assert_eq!(then_body.len(), 1);
                assert!(matches!(&then_body[0], Stmt::Eval(_)));
                assert!(else_body.is_empty());
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn goto_to_exit_through_enclosing_if_becomes_break() {
        // if (c) { while (w) { if (x) { continue; } else { goto L2; } } } L2:
        let mut stmts = vec![
            Stmt::If {
                cond: var("c"),
                then_body: vec![Stmt::While {
                    cond: var("w"),
                    body: vec![Stmt::If {
                        cond: var("x"),
                        then_body: vec![Stmt::Continue],
                        else_body: vec![Stmt::Goto("L2".into())],
                    }],
                }],
                else_body: vec![],
            },
            Stmt::Label("L2".into()),
        ];
        let n = rewrite_fallthrough(&mut stmts);
        assert!(n >= 1, "the exit jump must convert");
        let mut found_break = false;
        fn scan(s: &[Stmt], b: &mut bool) {
            for x in s {
                if matches!(x, Stmt::Break) { *b = true; }
                let mut kids: Vec<&Vec<Stmt>> = Vec::new();
                children(x, &mut kids);
                for k in kids { scan(k, b); }
            }
        }
        scan(&stmts, &mut found_break);
        assert!(found_break, "goto L2 should be a break: {stmts:?}");
    }
}
