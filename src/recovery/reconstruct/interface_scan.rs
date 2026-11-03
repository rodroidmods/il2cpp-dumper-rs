//! Remove the dead IL2CPP interface-offset scan around a resolved call.
//!
//! An interface dispatch compiles to an inline scan of `klass->interfaceOffsets`
//! (guarded by `klass->interfaces_count`) whose only product is the
//! `VirtualInvokeData*` the call dereferences:
//!
//! ```text
//! if (*(short*)(obj +302) != 0)
//! {
//!     var ptr = obj.static_fields + 8;
//!     var i = *(short*)(obj +302);
//!     while (*(nint*)(ptr -8) != typeof(I))
//!     {
//!         ptr = ptr + 16;
//!         i = i - 1;
//!         if (i != 0) continue;
//!         fn_12d879c(obj, typeof(I), 1);
//!         break;
//!     }
//!     obj.SetLocalTransform(…);          // the resolved call
//! }
//! ```
//!
//! Once `resolve::rewrite_interface_invoke_calls` has proved the callee, the
//! invoke-data pointer is DEAD: nothing reads the scan's locals after it. The
//! loop is then pure plumbing (loads, comparisons, local increments, and the
//! invoke-data helper) and the whole guard is removable; the source always
//! calls the method, so the `interfaces_count` test goes too.
//!
//! Safety. A candidate is removed only when:
//!  * the loop is the interface-offset scan (`*(…ptr ± 8) != typeof(I)`);
//!  * the loop's PREFIX before the first real statement is local updates,
//!    branches, or unresolved runtime helpers only;
//!  * the TAIL contains at least one real statement and NO control transfer
//!    (`break`/`continue`/`goto`/label) — a label or jump would bind to the
//!    removed loop;
//!  * EVERY read of every local the removed prefix defines happens inside
//!    that prefix (method-wide census, target strings included). The tail is
//!    KEPT, so a tail read of a scan local refuses the site; a reused display
//!    name outside refuses too.

use super::super::csharp::{BinOp, CmpKind, Expr, Stmt};
use std::collections::{HashMap, HashSet};

fn plain_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn target_idents(target: &str, f: impl FnMut(&str)) {
    let mut f = f;
    for tok in target.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if plain_ident(tok) {
            f(tok);
        }
    }
}

/// Count every READ of every name, method-wide (Var occurrences plus
/// identifier tokens in assignment TARGETS, which are strings).
fn count_reads_list(list: &[Stmt], out: &mut HashMap<String, usize>) {
    for s in list {
        crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
            fn walk(e: &Expr, out: &mut HashMap<String, usize>) {
                if let Expr::Var(n) = e {
                    *out.entry(n.clone()).or_default() += 1;
                }
                for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
                    walk(c, out);
                }
            }
            walk(e, out);
        });
        if let Stmt::Assign(t, _, _) = s {
            target_idents(t, |n| *out.entry(n.to_string()).or_default() += 1);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            count_reads_list(k, out);
        }
    }
}

fn collect_loop_reads(s: &Stmt, out: &mut HashSet<String>) {
    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
        fn walk(e: &Expr, out: &mut HashSet<String>) {
            if let Expr::Var(n) = e {
                out.insert(n.clone());
            }
            for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
                walk(c, out);
            }
        }
        walk(e, out);
    });
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            collect_loop_reads(x, out);
        }
    }
}

fn type_of_name(e: &Expr) -> Option<&str> {
    if let Expr::TypeOf(t) = e {
        Some(t)
    } else {
        None
    }
}

/// The pointer variable of a `*(…ptr ± N) == typeof(I)` scan condition.
fn scan_cond_ptr(cond: &Expr) -> Option<String> {
    let Expr::Cmp(kind, a, b) = cond else {
        return None;
    };
    if !matches!(kind, CmpKind::Eq | CmpKind::Ne) {
        return None;
    }
    let other = if type_of_name(b).is_some() {
        a.as_ref()
    } else if type_of_name(a).is_some() {
        b.as_ref()
    } else {
        return None;
    };
    match other {
        // `*(nint*)(ptr -8)`: the fold puts the displacement in the deref's
        // own addend; a separately computed address keeps it in the base.
        Expr::RawDeref(inner, off, _) => match inner.as_ref() {
            Expr::Var(v) if off.abs() <= 64 => Some(v.clone()),
            Expr::Bin(BinOp::Add | BinOp::Sub, x, k) => match (x.as_ref(), k.as_ref()) {
                (Expr::Var(v), Expr::Int(_)) => Some(v.clone()),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

fn is_plumbing_name(name: &str) -> bool {
    name.starts_with("fn_")
        || name.starts_with("IL2CPP_")
        || name.starts_with("il2cpp_")
        || name.starts_with("fnptr_")
}

/// Does the expression call anything that is not an unresolved runtime helper?
fn expr_plumbing_only(e: &Expr) -> bool {
    let mut ok = true;
    fn walk(e: &Expr, ok: &mut bool) {
        match e {
            Expr::Call { name, .. } => {
                if !(name.starts_with("fn_")
                    || name.starts_with("IL2CPP_")
                    || name.starts_with("il2cpp_")
                    || name.starts_with("fnptr_"))
                {
                    *ok = false;
                }
            }
            Expr::RawCall(..) => {}
            Expr::MethodGroup(_) => *ok = false,
            _ => {}
        }
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
            walk(c, ok);
        }
    }
    walk(e, &mut ok);
    ok
}

fn is_plumbing_eval(e: &Expr) -> bool {
    match e {
        Expr::Call { name, .. } => {
            name.starts_with("fn_")
                || name.starts_with("IL2CPP_")
                || name.starts_with("il2cpp_")
                || name.starts_with("fnptr_")
        }
        Expr::RawCall(..) => true,
        _ => false,
    }
}

/// One statement of the scan PREFIX: local bookkeeping, a branch, or runtime
/// plumbing. Anything else starts the TAIL.
fn prefix_stmt_ok(s: &Stmt, assigned: &mut HashSet<String>) -> bool {
    match s {
        Stmt::Assign(name, _, rhs) => {
            if !plain_ident(name) || !expr_plumbing_only(rhs) {
                return false;
            }
            assigned.insert(name.clone());
            true
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            if !expr_plumbing_only(cond) {
                return false;
            }
            for arm in [then_body, else_body] {
                for a in arm {
                    match a {
                        Stmt::Break
                        | Stmt::Continue
                        | Stmt::Goto(_)
                        | Stmt::Label(_)
                        | Stmt::Comment(_) => {}
                        Stmt::Eval(e) if is_plumbing_eval(e) => {}
                        _ => return false,
                    }
                }
            }
            true
        }
        Stmt::Eval(e) => is_plumbing_eval(e),
        Stmt::Break | Stmt::Continue | Stmt::Goto(_) | Stmt::Label(_) | Stmt::Comment(_) => true,
        _ => false,
    }
}

/// Control transfers that would BIND to the removed loop (`break`,
/// `continue`) or to a removed switch. A `goto` to an external label is the
/// continuation the scan's not-found path jumps to and stays valid.
fn stmt_binds_to_loop(s: &Stmt) -> bool {
    match s {
        Stmt::Break | Stmt::Continue | Stmt::GotoCase(_) => true,
        _ => {
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            kids.iter().any(|k| k.iter().any(stmt_binds_to_loop))
        }
    }
}

fn collect_labels(s: &Stmt, out: &mut HashSet<String>) {
    if let Stmt::Label(l) = s {
        out.insert(l.clone());
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            collect_labels(x, out);
        }
    }
}

fn collect_gotos(s: &Stmt, out: &mut HashSet<String>) {
    if let Stmt::Goto(l) = s {
        out.insert(l.clone());
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            collect_gotos(x, out);
        }
    }
}

/// A statement the source plausibly contains (as opposed to runtime
/// plumbing): a call to a managed method or a store to a field/memory.
fn stmt_is_real(s: &Stmt) -> bool {
    match s {
        Stmt::Assign(t, _, rhs) => {
            if !plain_ident(t) {
                return true;
            }
            let mut real = false;
            fn walk(e: &Expr, real: &mut bool) {
                if let Expr::Call { name, .. } = e {
                    if !is_plumbing_name(name) {
                        *real = true;
                    }
                }
                for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
                    walk(c, real);
                }
            }
            walk(rhs, &mut real);
            real
        }
        Stmt::Eval(e) => !is_plumbing_eval(e),
        Stmt::IndexAssign { .. } | Stmt::EventAssign { .. } => true,
        _ => false,
    }
}

fn tail_is_hoistable(tail: &[Stmt]) -> bool {
    if tail.is_empty() || tail.iter().any(stmt_binds_to_loop) {
        return false;
    }
    tail.iter().any(stmt_is_real) || tail.iter().any(|s| matches!(s, Stmt::Goto(_)))
}

/// Split the body into prefix (see `prefix_stmt_ok`) and tail.
fn split_body(body: &[Stmt], assigned: &mut HashSet<String>) -> Option<Vec<Stmt>> {
    let mut i = 0;
    while i < body.len() && prefix_stmt_ok(&body[i], assigned) {
        i += 1;
    }
    let tail: Vec<Stmt> = body[i..].to_vec();
    if !tail.is_empty() && !tail_is_hoistable(&tail) {
        return None;
    }
    Some(tail)
}

/// Locate a removable scan ending at `w` (a `While` at that index).
/// Returns the first index of the setup assignments and the hoisted tail.
fn scan_at(
    then: &[Stmt],
    w: usize,
    reads: &HashMap<String, usize>,
) -> Option<(usize, usize, Vec<Stmt>)> {
    let dbg = |m: &str| {
        if crate::recovery::env_flags::flag("ILR_SCANDBG") {
            eprintln!("[scandbg] w={w}: {m}");
        }
    };
    let Stmt::While { cond, body } = &then[w] else {
        return None;
    };
    let Some(ptr_var) = scan_cond_ptr(cond) else {
        dbg("cond is not a typeof scan");
        return None;
    };
    let mut body_vars: HashSet<String> = HashSet::new();
    let Some(tail) = split_body(body, &mut body_vars) else {
        if crate::recovery::env_flags::flag("ILR_SCANDBG") {
            let d = format!("{body:?}");
            eprintln!("[scandbg] w={w}: body refused: {}", &d[..d.len().min(2000)]);
        }
        return None;
    };
    let prefix = &body[..body.len() - tail.len()];
    // Variables the loop READS decide which preceding assignments are setup.
    // The whole loop (condition AND body) is walked: the index is read in the
    // body, the pointer in the condition.
    let mut loop_reads: HashSet<String> = HashSet::new();
    collect_loop_reads(&then[w], &mut loop_reads);
    let mut start = w;
    while start > 0 {
        let Stmt::Assign(name, _, rhs) = &then[start - 1] else {
            break;
        };
        if !plain_ident(name) || !loop_reads.contains(name) || !expr_plumbing_only(rhs) {
            break;
        }
        start -= 1;
    }
    if start == w {
        dbg("no setup assignment pair");
        return None;
    }
    // The scan's FOUND path often materialises the same invoke-data pointer
    // just after the loop (`ptr2 = klass + (pairOffset << 4) + 312;`). It
    // reads the loop cursor and writes only that pointer, so absorb trailing
    // plumbing assignments into the removed region; stop at a label or real
    // code (the continuation), which is kept.
    let mut end = w;
    while end + 1 < then.len() {
        let Stmt::Assign(name, _, rhs) = &then[end + 1] else {
            break;
        };
        if !plain_ident(name) || !expr_plumbing_only(rhs) {
            break;
        }
        end += 1;
    }
    let mut defined: HashSet<String> = body_vars;
    for s in &then[start..=end] {
        if let Stmt::Assign(name, _, _) = s {
            defined.insert(name.clone());
        }
    }
    if !defined.contains(&ptr_var) {
        dbg("pointer variable not defined by the scan");
        return None;
    }
    if tail.iter().any(|s| matches!(s, Stmt::Goto(_))) {
        let mut removed_labels: HashSet<String> = HashSet::new();
        for s in &then[start..w] {
            collect_labels(s, &mut removed_labels);
        }
        for s in body.iter().take(prefix.len()) {
            collect_labels(s, &mut removed_labels);
        }
        let mut tail_gotos: HashSet<String> = HashSet::new();
        for s in &tail {
            collect_gotos(s, &mut tail_gotos);
        }
        if tail_gotos.iter().any(|g| removed_labels.contains(g)) {
            dbg("tail goto targets a label inside the removed scan");
            return None;
        }
    }
    // Every read of every defined name must lie INSIDE the removed part
    // (setup + loop PREFIX). The hoisted tail is KEPT, so a tail read of a
    // scan local — or a reuse of the name anywhere else in the method —
    // makes the counts differ and refuses the site.
    let mut removed = HashMap::new();
    count_reads_list(&then[start..w], &mut removed);
    let prefix_loop = Stmt::While {
        cond: cond.clone(),
        body: prefix.to_vec(),
    };
    count_reads_list(std::slice::from_ref(&prefix_loop), &mut removed);
    // Absorbed found-path assignments are removed too; the hoisted TAIL is
    // kept and must not read a scan local (the counts would catch it).
    if end > w {
        count_reads_list(&then[w + 1..=end], &mut removed);
    }
    for n in &defined {
        let total = reads.get(n).copied().unwrap_or(0);
        let inside = removed.get(n).copied().unwrap_or(0);
        if total != inside {
            dbg(&format!("local {n} read outside ({total} vs {inside})"));
            return None;
        }
    }
    Some((start, end, tail))
}

/// `*(short*)(recv + off) ==/!= 0` — the `interfaces_count` guard.
fn cond_is_scan_count(cond: &Expr) -> bool {
    let Expr::Cmp(kind, a, b) = cond else {
        return false;
    };
    if !matches!(kind, CmpKind::Eq | CmpKind::Ne) {
        return false;
    }
    for (x, y) in [(a, b), (b, a)] {
        if matches!(y.as_ref(), Expr::Int(0)) {
            if let Expr::RawDeref(_, _, 2) = x.as_ref() {
                return true;
            }
        }
    }
    false
}

fn try_unwrap_if(stmt: &Stmt, reads: &HashMap<String, usize>) -> Option<Vec<Stmt>> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };
    if !else_body.is_empty() || !cond_is_scan_count(cond) {
        return None;
    }
    for w in 0..then_body.len() {
        if let Some((start, end, tail)) = scan_at(then_body, w, reads) {
            let mut keep = Vec::with_capacity(then_body.len() - (end + 1 - start) + tail.len());
            keep.extend(then_body[..start].iter().cloned());
            keep.extend(tail);
            keep.extend(then_body[end + 1..].iter().cloned());
            return Some(keep);
        }
    }
    None
}

fn strip_one_list(
    list: &mut Vec<Stmt>,
    reads: &HashMap<String, usize>,
    changed: &mut bool,
    n: &mut usize,
) {
    if *changed {
        return;
    }
    let mut i = 0;
    while i < list.len() {
        if let Some(keep) = try_unwrap_if(&list[i], reads) {
            let _ = list.remove(i);
            list.splice(i..i, keep);
            *n += 1;
            *changed = true;
            return;
        }
        // A bare scan directly in this list (no count guard around it).
        if matches!(&list[i], Stmt::While { .. }) {
            if let Some((start, end, tail)) = scan_at(list, i, reads) {
                list.splice(start..=end, tail);
                *n += 1;
                *changed = true;
                return;
            }
        }
        let mut kids: Vec<&mut Vec<Stmt>> = super::children_mut(&mut list[i]);
        for k in kids.drain(..) {
            strip_one_list(k, reads, changed, n);
            if *changed {
                return;
            }
        }
        i += 1;
    }
}

/// Returns the number of scans removed.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    if crate::recovery::env_flags::flag("ILR_NOSCAN") {
        return 0;
    }
    let mut n = 0;
    loop {
        let mut reads = HashMap::new();
        count_reads_list(stmts, &mut reads);
        let mut changed = false;
        strip_one_list(stmts, &reads, &mut changed, &mut n);
        if !changed {
            break;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::Expr;

    fn v(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn scan_if(extra_after: Vec<Stmt>, managed_in_body: bool) -> Stmt {
        let mut body = vec![
            Stmt::Assign(
                "ptr".into(),
                None,
                Expr::Bin(
                    BinOp::Add,
                    Box::new(Expr::Field(Box::new(v("obj")), "static_fields".into())),
                    Box::new(Expr::Int(8)),
                ),
            ),
            Stmt::Assign("i".into(), None, Expr::RawDeref(Box::new(v("obj")), 302, 2)),
            Stmt::While {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(Expr::RawDeref(
                        Box::new(Expr::Bin(
                            BinOp::Sub,
                            Box::new(v("ptr")),
                            Box::new(Expr::Int(8)),
                        )),
                        0,
                        0,
                    )),
                    Box::new(Expr::TypeOf("I".into())),
                ),
                body: {
                    let mut b = vec![
                        Stmt::Assign(
                            "ptr".into(),
                            None,
                            Expr::Bin(BinOp::Add, Box::new(v("ptr")), Box::new(Expr::Int(16))),
                        ),
                        Stmt::Assign(
                            "i".into(),
                            None,
                            Expr::Bin(BinOp::Sub, Box::new(v("i")), Box::new(Expr::Int(1))),
                        ),
                        Stmt::If {
                            cond: Expr::Cmp(
                                CmpKind::Ne,
                                Box::new(v("i")),
                                Box::new(Expr::Int(0)),
                            ),
                            then_body: vec![Stmt::Continue],
                            else_body: vec![],
                        },
                        Stmt::Eval(Expr::Call {
                            recv: None,
                            name: "fn_12d879c".into(),
                            args: vec![],
                        }),
                        Stmt::Break,
                    ];
                    if managed_in_body {
                        b.insert(
                            0,
                            Stmt::Eval(Expr::Call {
                                recv: None,
                                name: "DoManaged".into(),
                                args: vec![],
                            }),
                        );
                    }
                    b
                },
            },
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("obj"))),
                name: "SetLocalTransform".into(),
                args: vec![],
            }),
        ];
        body.extend(extra_after);
        Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::RawDeref(Box::new(v("obj")), 302, 2)),
                Box::new(Expr::Int(0)),
            ),
            then_body: body,
            else_body: vec![],
        }
    }

    #[test]
    fn a_dead_scan_and_its_count_guard_are_removed() {
        let mut stmts = vec![scan_if(vec![], false)];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "SetLocalTransform"
        ));
    }

    #[test]
    fn a_scan_that_reaches_a_real_tail_hoists_it() {
        // The not-found path continues with real statements INSIDE the loop
        // (the structurer leaves the call there, 0x148AB2C). They must be
        // hoisted; the scan scaffolding dropped.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::RawDeref(Box::new(v("obj")), 302, 2)),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Assign(
                    "ptr".into(),
                    None,
                    Expr::Field(Box::new(v("obj")), "static_fields".into()),
                ),
                Stmt::Assign("i".into(), None, Expr::RawDeref(Box::new(v("obj")), 302, 2)),
                Stmt::While {
                    cond: Expr::Cmp(
                        CmpKind::Ne,
                        Box::new(Expr::RawDeref(Box::new(v("ptr")), -8, 0)),
                        Box::new(Expr::TypeOf("I".into())),
                    ),
                    body: vec![
                        Stmt::Assign(
                            "ptr".into(),
                            None,
                            Expr::Bin(BinOp::Add, Box::new(v("ptr")), Box::new(Expr::Int(16))),
                        ),
                        Stmt::Assign(
                            "i".into(),
                            None,
                            Expr::Bin(BinOp::Sub, Box::new(v("i")), Box::new(Expr::Int(1))),
                        ),
                        Stmt::If {
                            cond: Expr::Cmp(
                                CmpKind::Ne,
                                Box::new(v("i")),
                                Box::new(Expr::Int(0)),
                            ),
                            then_body: vec![Stmt::Continue],
                            else_body: vec![],
                        },
                        Stmt::Eval(Expr::Call {
                            recv: None,
                            name: "fn_12d879c".into(),
                            args: vec![],
                        }),
                        Stmt::Eval(Expr::Call {
                            recv: Some(Box::new(Expr::Field(
                                Box::new(Expr::This),
                                "_ctrl".into(),
                            ))),
                            name: "SetSelectionMovement".into(),
                            args: vec![],
                        }),
                        Stmt::Return(None),
                    ],
                },
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        assert_eq!(stmts.len(), 2, "{stmts:?}");
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "SetSelectionMovement"
        ));
        assert!(matches!(&stmts[1], Stmt::Return(None)));
    }

    #[test]
    fn a_tail_with_a_break_is_not_hoisted() {
        // A real statement in the loop followed by a break: hoisting would
        // orphan the break (it binds to the removed loop), so refuse.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::RawDeref(Box::new(v("obj")), 302, 2)),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Assign(
                    "ptr".into(),
                    None,
                    Expr::Field(Box::new(v("obj")), "static_fields".into()),
                ),
                Stmt::While {
                    cond: Expr::Cmp(
                        CmpKind::Ne,
                        Box::new(Expr::RawDeref(Box::new(v("ptr")), -8, 0)),
                        Box::new(Expr::TypeOf("I".into())),
                    ),
                    body: vec![
                        Stmt::Eval(Expr::Call {
                            recv: Some(Box::new(Expr::Field(
                                Box::new(Expr::This),
                                "_ctrl".into(),
                            ))),
                            name: "SetSelectionMovement".into(),
                            args: vec![],
                        }),
                        Stmt::Break,
                    ],
                },
            ],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn a_scan_whose_locals_are_read_later_is_kept() {
        let mut stmts = vec![scan_if(vec![Stmt::Eval(v("ptr"))], false)];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    #[test]
    fn a_scan_containing_a_managed_call_is_kept() {
        let mut stmts = vec![scan_if(vec![], true)];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }
}
