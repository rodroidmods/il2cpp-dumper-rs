//! Fold a `goto` that is really a fall-through.
//!
//! Two shapes survive every structural pass because they are produced *by*
//! the passes that invert branches:
//!
//! ```text
//! if (c) { …; goto L; }        // L is the very next statement, so the
//! L:                           // arm already falls through to it
//! ```
//!
//! ```text
//! …                            // a bare `goto L;` immediately before `L:`
//! goto L;
//! L:
//! ```
//!
//! Both are pure spelling: control reaches `L` with or without the jump, and
//! no observable evaluation is skipped (the goto operand is a label, not an
//! expression). Removing them is unconditional — there is no shape that a
//! fall-through `goto` could be *needed* for.
//!
//! Labels left with no remaining reference are swept afterwards with the same
//! `strip_quiet_labels` the cleanup pass uses, so the pair disappears
//! together.

use std::collections::HashSet;

use crate::recovery::csharp::{self, Stmt};

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    visit(stmts, &mut n);
    if n > 0 {
        let mut referenced: HashSet<String> = HashSet::new();
        csharp::collect_gotos(stmts, &mut referenced);
        csharp::strip_quiet_labels(stmts, &referenced);
    }
    n
}


/// Lift a `goto` that jumps OUT of nested `if` arms to a join label.
///
/// The `goto` is the structurer's only spelling for "skip the rest of this
/// nest", and it is correct — but it is not what C# source looks like. The
/// skipped statements run exactly when the innermost test is FALSE, so they
/// belong in its `else` arm. Applying this bottom-up turns a nest of forward
/// `goto`s into nested `if`/`else` with identical control flow and no labels.
///
/// Sound because every crossed construct is an `if` whose `else` is empty (so
/// falling out of it reaches the region), and the region contains no label (a
/// landing site moved into an arm would strand an edge from outside).
fn can_inject(s: &Stmt, label: &str) -> bool {
    let Stmt::If {
        then_body,
        else_body,
        ..
    } = s
    else {
        return false;
    };
    if !else_body.iter().all(is_noise) {
        return false;
    }
    if trailing_goto_index(then_body, label).is_some() {
        return true;
    }
    then_body.iter().rev().any(|c| can_inject(c, label))
}

/// Move `region` into the `else` arm of the `if` that carries the jump to
/// `label`, descending through the arm's tail constructs.
///
/// Statements AFTER the carrying construct in this arm run only when the jump
/// is not taken, so they join the region on the way down:
///
/// ```text
/// if (c) { X; if (d) { Y; goto L } Z }   →   if (c) { X; if (d) { Y }
///                                                              else { Z; … } }
/// ```
fn inject_tail_region(s: &mut Stmt, label: &str, region: Vec<Stmt>) -> bool {
    let Stmt::If {
        then_body,
        else_body,
        ..
    } = s
    else {
        return false;
    };
    if !else_body.iter().all(is_noise) {
        return false;
    }
    if let Some(idx) = trailing_goto_index(then_body, label) {
        then_body.truncate(idx);
        *else_body = region;
        return true;
    }
    let mut j = then_body.len();
    while j > 0 {
        j -= 1;
        if matches!(then_body[j], Stmt::Comment(_)) {
            continue;
        }
        let tail = &then_body[j + 1..];
        if tail.iter().any(|s| matches!(s, Stmt::Label(_))) {
            continue;
        }
        let mut inner = tail.to_vec();
        inner.extend(region.clone());
        if inject_tail_region(&mut then_body[j], label, inner) {
            then_body.truncate(j + 1);
            return true;
        }
    }
    false
}

fn trailing_goto_index(body: &[Stmt], label: &str) -> Option<usize> {
    let mut end = body.len();
    while end > 0 && matches!(body[end - 1], Stmt::Comment(_) | Stmt::Attribute { .. }) {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    match &body[end - 1] {
        Stmt::Goto(x) if x == label => Some(end - 1),
        _ => None,
    }
}

fn is_noise(s: &Stmt) -> bool {
    matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. } | Stmt::Label(_))
}

/// One sweep of the cross-level lift: for every label defined in `list`, take
/// the statements between the preceding construct and the label as the region
/// and push it into that construct's tail `if`.
fn lift_cross_level(list: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    loop {
        let mut done = false;
        for k in 0..list.len() {
            let Stmt::Label(l) = &list[k] else {
                continue;
            };
            let l = l.clone();
            // The construct that carries the jump is somewhere BEFORE the
            // label, not necessarily adjacent: the statements between it and
            // the label are exactly the ones the jump skips. Take the
            // nearest one that can absorb them.
            let mut found = None;
            for i in (0..k).rev() {
                if matches!(list[i], Stmt::Comment(_)) {
                    continue;
                }
                if !can_inject(&list[i], &l) {
                    continue;
                }
                if list[i + 1..k]
                    .iter()
                    .any(|s| matches!(s, Stmt::Label(_)))
                {
                    continue;
                }
                found = Some(i);
                break;
            }
            let Some(i) = found else {
                continue;
            };
            let region: Vec<Stmt> = list[i + 1..k].to_vec();
            if inject_tail_region(&mut list[i], &l, region) {
                list.drain(i + 1..k);
                n += 1;
                done = true;
                break;
            }
        }
        if !done {
            return n;
        }
    }
}
fn visit(list: &mut Vec<Stmt>, n: &mut usize) {
    // Cross-level first: it removes the jump and the statements it skips
    // together, which the same-list rules below cannot see.
    *n += lift_cross_level(list);
    // A trailing `goto L` in a construct whose successor in THIS list is `L:`
    // is that construct's fall-through.
    for i in 0..list.len().saturating_sub(1) {
        let Some(Stmt::Label(l)) = list.get(i + 1) else {
            continue;
        };
        let l = l.clone();
        if let Some(s) = list.get_mut(i) {
            *n += fold_into_successor(s, &l);
        }
    }
    // `goto L;` immediately followed by `L:` in the same list.
    let mut i = 0usize;
    while i + 1 < list.len() {
        let same = matches!(
            (&list[i], &list[i + 1]),
            (Stmt::Goto(a), Stmt::Label(b)) if a == b
        );
        if same {
            list.remove(i);
            *n += 1;
        } else {
            i += 1;
        }
    }
    for s in list.iter_mut() {
        for c in super::children_mut(s) {
            visit(c, n);
        }
    }
}

/// Fold a `goto` that is the last statement of `s` and names the statement
/// right after `s` in its enclosing list.
///
/// * `if`/`lock`/`using`/`try` — control leaves the construct and reaches the
///   successor anyway, so the jump is pure spelling.
/// * loops and `switch` — control would NOT reach the successor (it would
///   iterate / fall to the next arm), so the jump becomes the `break` that
///   means the same thing.
fn fold_into_successor(s: &mut Stmt, label: &str) -> usize {
    match s {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => drop_trailing_goto(then_body, label) + drop_trailing_goto(else_body, label),
        Stmt::Lock { body, .. } | Stmt::Using { body, .. } => drop_trailing_goto(body, label),
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut c = drop_trailing_goto(try_body, label);
            for (_, b) in catches.iter_mut() {
                c += drop_trailing_goto(b, label);
            }
            if let Some(f) = finally_body {
                c += drop_trailing_goto(f, label);
            }
            c
        }
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::For { body, .. }
        | Stmt::ForEach { body, .. } => break_trailing_goto(body, label),
        Stmt::Switch {
            cases,
            tail,
            default,
            ..
        } => {
            let mut c = 0;
            for (_, b) in cases.iter_mut() {
                c += break_trailing_goto(b, label);
            }
            c += break_trailing_goto(tail, label);
            if let Some(d) = default {
                c += break_trailing_goto(d, label);
            }
            c
        }
        _ => 0,
    }
}

fn break_trailing_goto(body: &mut Vec<Stmt>, label: &str) -> usize {
    if let Some(last) = body.last_mut() {
        if matches!(last, Stmt::Goto(x) if x == label) {
            *last = Stmt::Break;
            return 1;
        }
    }
    0
}

fn drop_trailing_goto(body: &mut Vec<Stmt>, label: &str) -> usize {
    if matches!(body.last(), Some(Stmt::Goto(x)) if x == label) {
        body.pop();
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(n: &str) -> Stmt {
        Stmt::Label(n.to_string())
    }

    fn goto(n: &str) -> Stmt {
        Stmt::Goto(n.to_string())
    }

    #[test]
    fn a_bare_goto_before_its_label_is_dropped() {
        let mut stmts = vec![
            Stmt::Eval(crate::recovery::csharp::Expr::Int(1)),
            goto("L0"),
            label("L0"),
            Stmt::Return(None),
        ];
        assert_eq!(apply(&mut stmts), 1);
        // The jump AND the label it named are gone: nothing references `L0`
        // any more, so the quiet-label sweep takes it with the goto.
        assert!(!stmts.iter().any(|s| matches!(s, Stmt::Goto(_))));
        assert!(!stmts.iter().any(|s| matches!(s, Stmt::Label(_))));
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn a_trailing_goto_in_an_if_arm_before_the_label_is_dropped() {
        let mut stmts = vec![
            Stmt::If {
                cond: crate::recovery::csharp::Expr::Bool(true),
                then_body: vec![
                    Stmt::Eval(crate::recovery::csharp::Expr::Int(1)),
                    goto("L0"),
                ],
                else_body: vec![goto("L0")],
            },
            label("L0"),
            Stmt::Return(None),
        ];
        assert_eq!(apply(&mut stmts), 2);
        let Stmt::If {
            then_body,
            else_body,
            ..
        } = &stmts[0]
        else {
            panic!("if")
        };
        assert_eq!(then_body.len(), 1);
        assert!(else_body.is_empty());
    }

    #[test]
    fn a_goto_to_a_different_label_survives() {
        let mut stmts = vec![goto("L1"), label("L0"), Stmt::Return(None), label("L1")];
        assert_eq!(apply(&mut stmts), 0);
    }
}
