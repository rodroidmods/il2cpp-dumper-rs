//! Hoist a merge block out of an `if` arm.
//!
//! ## Shape
//!
//! The structurer emits a branch's common continuation at the tail of the
//! last-visited arm and jumps into it from the other arm:
//!
//! ```text
//! if (c) {
//!     A;
//!     goto L;
//! } else {
//!     B;
//!     return;
//!     L:              // merge block emitted INSIDE the arm
//!     C;
//! }
//! ```
//!
//! Moving `L: C` out of the arm — immediately after the `if` — gives the same
//! control flow (the then-arm still jumps to `L`, the else-arm still returns
//! before it) and turns the jump into an ordinary forward join that
//! [`super::goto_fold`] can delete:
//!
//! ```text
//! if (c) { A; goto L; } else { B; return; }
//! L:
//! C;
//! ```
//!
//! ## Why this is safe
//!
//! * The moved tail is a contiguous suffix of the arm, so every path that
//!   fell into it before still does — now by falling out of the `if`.
//! * The arm's sibling must TERMINATE on every path (return/throw/goto to the
//!   hoisted label/break/continue). Otherwise its fall-through would start
//!   executing the hoisted tail after the move, which it did not before.
//! * The label must be referenced from outside the arm; a purely internal
//!   label is part of the arm's own loop structure and stays.
//! * Moving statements out of an arm is refused when that could change C#
//!   scope: the tail may not declare typed locals (`Assign(_, Some(_), _)`),
//!   and may not read names declared inside the arm before the label
//!   (typed assigns, `foreach` iteration variables).
//!
//! Measured on Temple Run 2: ~3,950 remaining forward gotos target a merge
//! block emitted inside an arm; this is the largest forward-join class.

use super::super::csharp::{Expr, InterpPart, Stmt};
use super::children_mut;
use std::collections::{HashMap, HashSet};

/// One hoist: move `arm[tail_at..]` to `parent[i + 1..]`.
fn apply_list(stmts: &mut Vec<Stmt>, targets: &HashMap<String, usize>) -> usize {
    let mut total = 0;
    // Bottom-up: an inner `if` can hoist its tail into the enclosing arm, and
    // the enclosing `if` may then hoist it again in the same call.
    for s in stmts.iter_mut() {
        for body in children_mut(s) {
            total += apply_list(body, targets);
        }
    }
    let mut i = 0;
    while i < stmts.len() {
        let hoisted = match &stmts[i] {
            Stmt::If {
                then_body,
                else_body,
                ..
            } => hoist_from_else(else_body, then_body, targets)
                .or_else(|| hoist_from_then(then_body, else_body, targets)),
            _ => None,
        };
        if let Some((mut tail, from_else)) = hoisted {
            let hoisted_len = tail.len();
            if from_else {
                if let Stmt::If { else_body, .. } = &mut stmts[i] {
                    else_body.truncate(else_body.len() - hoisted_len);
                }
            } else if let Stmt::If { then_body, .. } = &mut stmts[i] {
                then_body.truncate(then_body.len() - hoisted_len);
            }
            // The tail is inserted right after the `if`; its labels move with
            // it, so every reference still resolves.
            tail.shrink_to_fit();
            for (k, s) in tail.into_iter().enumerate() {
                stmts.insert(i + 1 + k, s);
            }
            total += 1;
            // Re-examine the same `if`: the sibling may have become empty and
            // another label in the remaining arm may now qualify.
            continue;
        }
        i += 1;
    }
    total
}

/// Try to hoist the merge tail out of the ELSE arm. The THEN arm must
/// terminate on every path.
fn hoist_from_else(
    else_body: &[Stmt],
    then_body: &[Stmt],
    targets: &HashMap<String, usize>,
) -> Option<(Vec<Stmt>, bool)> {
    hoist_arm(else_body, then_body, targets).map(|t| (t, true))
}

fn hoist_from_then(
    then_body: &[Stmt],
    else_body: &[Stmt],
    targets: &HashMap<String, usize>,
) -> Option<(Vec<Stmt>, bool)> {
    hoist_arm(then_body, else_body, targets).map(|t| (t, false))
}

fn hoist_arm(
    arm: &[Stmt],
    sibling: &[Stmt],
    targets: &HashMap<String, usize>,
) -> Option<Vec<Stmt>> {
    if arm.is_empty() {
        return None;
    }
    // The sibling must not fall out of the `if` into the hoisted tail. This
    // includes the EMPTY sibling: an empty arm always falls through, and after
    // the hoist it would execute the moved tail, which it did not before.
    if !list_terminates(sibling) {
        return None;
    }
    // Find the earliest label in the arm that something OUTSIDE the arm jumps
    // to. Purely internal labels belong to the arm's own structure.
    let mut internal: HashMap<&str, usize> = HashMap::new();
    count_targets_in(arm, &mut internal);
    let mut start = None;
    for (idx, s) in arm.iter().enumerate() {
        if let Stmt::Label(l) = s {
            let total = targets.get(l.as_str()).copied().unwrap_or(0);
            let here = internal.get(l.as_str()).copied().unwrap_or(0);
            if total > here {
                start = Some(idx);
                break;
            }
        }
    }
    let start = start?;
    let tail = &arm[start..];
    // The tail must carry something executable; a bare label is handled by
    // the goto folds.
    if !tail.iter().any(stmt_is_executable) {
        return None;
    }
    if !scope_safe(&arm[..start], tail) {
        return None;
    }
    Some(tail.to_vec())
}

fn stmt_is_executable(s: &Stmt) -> bool {
    !matches!(s, Stmt::Label(_) | Stmt::Comment(_) | Stmt::Attribute { .. })
}

/// Does every path through this list terminate (no fall-through)?
fn list_terminates(list: &[Stmt]) -> bool {
    match list.last() {
        None => false,
        Some(s) => stmt_terminates(s),
    }
}

fn stmt_terminates(s: &Stmt) -> bool {
    match s {
        Stmt::Return(_)
        | Stmt::Throw(_)
        | Stmt::Goto(_)
        | Stmt::GotoCase(_)
        | Stmt::Break
        | Stmt::Continue
        | Stmt::YieldBreak => true,
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            !then_body.is_empty()
                && !else_body.is_empty()
                && list_terminates(then_body)
                && list_terminates(else_body)
        }
        _ => false,
    }
}

/// Collect every goto target occurring inside `list` (recursively).
fn count_targets_in<'a>(list: &'a [Stmt], out: &mut HashMap<&'a str, usize>) {
    for s in list {
        match s {
            Stmt::Goto(l) => *out.entry(l.as_str()).or_insert(0) += 1,
            _ => {
                let mut kids: Vec<&Vec<Stmt>> = Vec::new();
                super::children(s, &mut kids);
                for k in kids {
                    count_targets_in(k, out);
                }
            }
        }
    }
}

/// Refuse moves that would break C# scoping.
fn scope_safe(before: &[Stmt], tail: &[Stmt]) -> bool {
    // A declaration in the moved tail would move relative to its readers.
    if tail.iter().any(declares_typed_local) {
        return false;
    }
    let mut declared: HashSet<String> = HashSet::new();
    for s in before {
        collect_declared(s, &mut declared);
    }
    if declared.is_empty() {
        return true;
    }
    !tail.iter().any(|s| stmt_reads_any(s, &declared))
}

fn declares_typed_local(s: &Stmt) -> bool {
    matches!(s, Stmt::Assign(_, Some(_), _))
}

fn collect_declared(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Assign(name, Some(_), _) => {
            out.insert(name.clone());
        }
        Stmt::ForEach { var_name, .. } => {
            out.insert(var_name.clone());
        }
        _ => {
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            for k in kids {
                for x in k {
                    collect_declared(x, out);
                }
            }
        }
    }
}

fn stmt_reads_any(s: &Stmt, names: &HashSet<String>) -> bool {
    let mut found = false;
    expr_or_stmt_reads(s, names, &mut found);
    found
}

fn expr_or_stmt_reads(s: &Stmt, names: &HashSet<String>, found: &mut bool) {
    if *found {
        return;
    }
    fn chk(e: &Expr, names: &HashSet<String>, found: &mut bool) {
        if !*found && expr_reads_any(e, names) {
            *found = true;
        }
    }
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => chk(e, names, found),
        Stmt::Return(Some(e)) => chk(e, names, found),
        Stmt::Assign(_, _, e) => chk(e, names, found),
        Stmt::If { cond, then_body, else_body } => {
            chk(cond, names, found);
            for x in then_body.iter().chain(else_body.iter()) {
                expr_or_stmt_reads(x, names, found);
            }
        }
        Stmt::While { cond, body } => {
            chk(cond, names, found);
            for x in body {
                expr_or_stmt_reads(x, names, found);
            }
        }
        Stmt::DoWhile { body, cond } => {
            for x in body {
                expr_or_stmt_reads(x, names, found);
            }
            chk(cond, names, found);
        }
        Stmt::For {
            init,
            cond,
            incr,
            body,
        } => {
            for x in init.iter().chain(incr.iter()) {
                expr_or_stmt_reads(x, names, found);
            }
            if let Some(c) = cond {
                chk(c, names, found);
            }
            for x in body {
                expr_or_stmt_reads(x, names, found);
            }
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            chk(collection, names, found);
            chk(index, names, found);
            chk(value, names, found);
        }
        Stmt::EventAssign { field, value, .. } => {
            chk(field, names, found);
            chk(value, names, found);
        }
        Stmt::Lock { expr, body } => {
            chk(expr, names, found);
            for x in body {
                expr_or_stmt_reads(x, names, found);
            }
        }
        Stmt::ForEach { body, .. } => {
            for x in body {
                expr_or_stmt_reads(x, names, found);
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases {
                for x in b {
                    expr_or_stmt_reads(x, names, found);
                }
            }
            for x in tail {
                expr_or_stmt_reads(x, names, found);
            }
            if let Some(arm) = default {
                for x in arm {
                    expr_or_stmt_reads(x, names, found);
                }
            }
        }
        Stmt::StringSwitch { disc, cases, tail } => {
            chk(disc, names, found);
            for (_, b) in cases {
                for x in b {
                    expr_or_stmt_reads(x, names, found);
                }
            }
            if let Some(t) = tail {
                for x in t {
                    expr_or_stmt_reads(x, names, found);
                }
            }
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            for x in try_body {
                expr_or_stmt_reads(x, names, found);
            }
            for (_, b) in catches {
                for x in b {
                    expr_or_stmt_reads(x, names, found);
                }
            }
            if let Some(f) = finally_body {
                for x in f {
                    expr_or_stmt_reads(x, names, found);
                }
            }
        }
        Stmt::Using { decl, body } => {
            expr_or_stmt_reads(decl, names, found);
            for x in body {
                expr_or_stmt_reads(x, names, found);
            }
        }
        _ => {}
    }
}

fn expr_reads_any(e: &Expr, names: &HashSet<String>) -> bool {
    match e {
        Expr::Var(n) => names.contains(n),
        Expr::Field(b, _)
        | Expr::RawDeref(b, _, _)
        | Expr::RawAddr(b, _)
        | Expr::AddressOf(b)
        | Expr::Cast(_, b)
        | Expr::Not(b)
        | Expr::Neg(b)
        | Expr::Await(b) => expr_reads_any(b, names),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            expr_reads_any(a, names) || expr_reads_any(b, names)
        }
        Expr::Cond(c, a, b) => {
            expr_reads_any(c, names) || expr_reads_any(a, names) || expr_reads_any(b, names)
        }
        Expr::Call { recv, args, .. } => {
            recv.as_deref().is_some_and(|r| expr_reads_any(r, names))
                || args.iter().any(|a| expr_reads_any(a, names))
        }
        Expr::RawCall(_, args, _)
        | Expr::NewArr(_, args)
        | Expr::NewObj(_, args)
        | Expr::ArrayInit { items: args, .. }
        | Expr::CollectionInit {
            ctor_args: args, items: _, ..
        } => args.iter().any(|a| expr_reads_any(a, names)),
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            ctor_args.iter().any(|a| expr_reads_any(a, names))
                || fields.iter().any(|(_, v)| expr_reads_any(v, names))
        }
        Expr::Index { collection, index } => {
            expr_reads_any(collection, names) || expr_reads_any(index, names)
        }
        Expr::NullConditional { receiver, access } => {
            expr_reads_any(receiver, names) || expr_reads_any(access, names)
        }
        Expr::MethodChain { receiver, methods } => {
            expr_reads_any(receiver, names)
                || methods.iter().any(|m| m.args.iter().any(|a| expr_reads_any(a, names)))
        }
        Expr::Lambda { body, .. } => expr_reads_any(body, names),
        Expr::SwitchExpr { discrim, arms, default_arm } => {
            expr_reads_any(discrim, names)
                || arms.iter().any(|(_, v)| expr_reads_any(v, names))
                || default_arm.as_deref().is_some_and(|d| expr_reads_any(d, names))
        }
        Expr::PatternMatch { expr, .. } => expr_reads_any(expr, names),
        Expr::StringInterp { parts } => parts.iter().any(|p| match p {
            InterpPart::Expr(e, _) => expr_reads_any(e, names),
            _ => false,
        }),
        Expr::Tuple { elements, .. } => elements.iter().any(|e| expr_reads_any(e, names)),
        _ => false,
    }
}


/// `if (c) goto L;` with an empty else → the condition and the target.
fn guard_goto(stmt: &Stmt) -> Option<(String, Expr)> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };
    if !else_body.is_empty() || then_body.len() != 1 {
        return None;
    }
    match &then_body[0] {
        Stmt::Goto(l) => Some((l.clone(), cond.clone())),
        _ => None,
    }
}

/// Invert a forward guard jump over a terminating block:
///
/// ```text
/// if (c) goto L;      // skip the block when c is true
/// B;                  // every path through B terminates
/// L:
/// ```
/// becomes
/// ```text
/// if (!c) { B }       // guard clause: run B only when c is false
/// L:
/// ```
///
/// `B` must terminate on every path (otherwise its fall-through would reach
/// `L` and execute the tail, which the jump skipped), contain no labels
/// (a goto cannot enter a C# block), and declare no local read after `L`.
fn invert_guards_in_list(list: &mut Vec<Stmt>) -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < list.len() {
        let Some((label, cond)) = guard_goto(&list[i]) else {
            i += 1;
            continue;
        };
        let Some(t) = list[i + 1..]
            .iter()
            .position(|s| matches!(s, Stmt::Label(l) if *l == label))
            .map(|p| i + 1 + p)
        else {
            i += 1;
            continue;
        };
        let block: Vec<Stmt> = list[i + 1..t].to_vec();
        if block.is_empty() || !list_terminates(&block) {
            i += 1;
            continue;
        }
        if block.iter().any(stmt_has_any_label) {
            i += 1;
            continue;
        }
        if !guard_scope_safe(&block, &list[t + 1..]) {
            i += 1;
            continue;
        }
        let new_if = Stmt::If {
            cond: Expr::Not(Box::new(cond)),
            then_body: block,
            else_body: vec![],
        };
        list.splice(i..t, vec![new_if]);
        total += 1;
        i += 1;
    }
    total
}

fn stmt_has_any_label(s: &Stmt) -> bool {
    if matches!(s, Stmt::Label(_)) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    kids.iter().any(|k| k.iter().any(stmt_has_any_label))
}

/// A local declared in the moved block must not be read after the join.
fn guard_scope_safe(block: &[Stmt], tail: &[Stmt]) -> bool {
    let mut declared: HashSet<String> = HashSet::new();
    for s in block {
        collect_declared(s, &mut declared);
    }
    if declared.is_empty() {
        return true;
    }
    !tail.iter().any(|s| stmt_reads_any(s, &declared))
}

fn invert_guards(stmts: &mut Vec<Stmt>) -> usize {
    let mut total = 0;
    for s in stmts.iter_mut() {
        for body in children_mut(s) {
            total += invert_guards(body);
        }
    }
    total + invert_guards_in_list(stmts)
}

/// Remove labels no goto references (after the rewrites above).
pub(crate) fn drop_unreferenced_labels(stmts: &mut Vec<Stmt>) {
    let mut targets: HashMap<String, usize> = HashMap::new();
    count_all_targets(stmts, &mut targets);
    drop_labels_rec(stmts, &targets);
}

fn drop_labels_rec(stmts: &mut Vec<Stmt>, targets: &HashMap<String, usize>) {
    stmts.retain(|s| !matches!(s, Stmt::Label(l) if !targets.contains_key(l)));
    for s in stmts.iter_mut() {
        for body in children_mut(s) {
            drop_labels_rec(body, targets);
        }
    }
}

/// Run the hoist over one method body. Returns the number of hoists.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut targets: HashMap<String, usize> = HashMap::new();
    count_all_targets(stmts, &mut targets);
    if targets.is_empty() {
        return 0;
    }
    let mut total = 0;
    for _ in 0..8 {
        let n = apply_list(stmts, &targets) + invert_guards(stmts);
        total += n;
        if n == 0 {
            break;
        }
    }
    drop_unreferenced_labels(stmts);
    total
}

fn count_all_targets(stmts: &[Stmt], out: &mut HashMap<String, usize>) {
    for s in stmts {
        match s {
            Stmt::Goto(l) => *out.entry(l.clone()).or_insert(0) += 1,
            _ => {
                let mut kids: Vec<&Vec<Stmt>> = Vec::new();
                super::children(s, &mut kids);
                for k in kids {
                    count_all_targets(k, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn call(n: &str) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: None,
            name: n.to_string(),
            args: vec![],
        })
    }

    fn label(n: &str) -> Stmt {
        Stmt::Label(n.to_string())
    }

    #[test]
    fn hoists_merge_tail_out_of_else_arm() {
        // if (c) { goto L; } else { work; return; L: tail; }
        let mut stmts = vec![
            Stmt::If {
                cond: var("c"),
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![call("work"), Stmt::Return(None), label("L"), call("tail")],
            },
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[0], Stmt::If { else_body, .. } if else_body.len() == 2));
        assert!(matches!(&stmts[1], Stmt::Label(l) if l == "L"));
        assert!(matches!(&stmts[2], Stmt::Eval(_)));
    }

    #[test]
    fn refuses_when_sibling_falls_through() {
        // The then arm does not terminate: hoisting would start executing the
        // moved tail on its fall-through path.
        let mut stmts = vec![Stmt::If {
            cond: var("c"),
            then_body: vec![call("a")],
            else_body: vec![call("b"), label("L"), call("tail")],
        }];
        // add a goto so the label has an outside reference
        stmts.insert(0, Stmt::Goto("L".into()));
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn refuses_internal_label() {
        // No goto outside the arm: the label belongs to the arm's structure.
        let mut stmts = vec![Stmt::If {
            cond: var("c"),
            then_body: vec![Stmt::Return(None)],
            else_body: vec![label("L"), call("tail"), Stmt::Return(None)],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn refuses_typed_decl_in_tail() {
        let mut stmts = vec![
            Stmt::Goto("L".into()),
            Stmt::If {
                cond: var("c"),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![
                    label("L"),
                    Stmt::Assign("v1".into(), Some("Foo".into()), var("x")),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn hoists_from_then_when_else_terminates() {
        let mut stmts = vec![Stmt::If {
            cond: var("c"),
            then_body: vec![call("a"), label("L"), call("tail")],
            else_body: vec![Stmt::Return(None)],
        }];
        // reference from the else side: add a goto before the if
        stmts.insert(0, Stmt::Goto("L".into()));
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[1], Stmt::If { then_body, .. } if then_body.len() == 1));
    }
}
