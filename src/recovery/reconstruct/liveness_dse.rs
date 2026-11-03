//! Dead-store elimination over the structured tree (late, backward liveness).
//!
//! Coalesced storage and split Object versions leave long runs of pure stores
//! whose value is overwritten before any read:
//!
//! ```text
//! instance3 = *(nint*)(this.RewardName +1376);   // overwritten on every path
//! instance3 = *(nint*)(this.RewardName +1376);
//! ```
//!
//! The name itself IS read somewhere, so the unread-name rule in
//! `dup_update_fold` cannot remove these. Backward liveness can: a pure store
//! is dead when the name is not live after it (no path reaches a read before
//! the next write). The earlier `dead_store` pass only scans one straight-line
//! run inside one list; this one merges both arms of an `if` and the arms of a
//! `switch`, so a store followed by a branch that overwrites on every path is
//! still removed.
//!
//! Scope of the proof:
//!  * `if/else` and `switch` are walked precisely (both arms, shared tails);
//!  * loops, `using`, `lock` and `try` are NOT descended into: every name read
//!    anywhere inside them is treated as used, and their bodies are untouched;
//!  * `break` inside a switch arm resumes at the arm's live-out;
//!  * a method containing `goto`/`label`/`goto case`/`yield` is skipped
//!    entirely: a jump bypasses lexical kill points, so the lexical walk would
//!    not be sound;
//!  * address-taken names, loop variables and any name read inside a lambda
//!    are never removed.
//!
//! Only stores whose right-hand side is pure (a read of fields/constants) are
//! candidates; a call assignment is kept for its side effect.

use crate::recovery::csharp::{Expr, Stmt};
use std::collections::HashSet;

fn is_plain_name(n: &str) -> bool {
    !n.is_empty()
        && n.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !matches!(n, "this" | "base" | "null" | "true" | "false" | "new" | "default")
}

fn is_pure(e: &Expr) -> bool {
    match e {
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::Unknown
        | Expr::MissingAbiArg
        | Expr::MethodGroup(_)
        | Expr::TypeOf(_)
        | Expr::StaticField(_, _) => true,
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => is_pure(r),
        Expr::AddressOf(x) => is_pure(x),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => is_pure(a) && is_pure(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) => is_pure(x),
        _ => false,
    }
}

#[derive(Default)]
struct Global {
    /// Every plain name mentioned anywhere (read or written).
    all: HashSet<String>,
    /// Names that must never be dropped: address-taken, loop variables, and
    /// every name read from inside a lambda.
    bad: HashSet<String>,
    /// The method contains an arbitrary jump; lexical liveness is unsound.
    skip: bool,
}

fn collect_expr(e: &Expr, g: &mut Global, in_lambda: bool) {
    match e {
        Expr::Var(n) => {
            if is_plain_name(n) {
                g.all.insert(n.clone());
                if in_lambda {
                    g.bad.insert(n.clone());
                }
            }
        }
        Expr::AddressOf(x) | Expr::RawAddr(x, _) => {
            if let Expr::Var(n) = x.as_ref() {
                if is_plain_name(n) {
                    g.bad.insert(n.clone());
                }
            }
        }
        Expr::Lambda { params, .. } => {
            for p in params {
                if is_plain_name(p) {
                    g.bad.insert(p.clone());
                }
            }
            for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
                collect_expr(c, g, true);
            }
            return;
        }
        _ => {}
    }
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
        collect_expr(c, g, in_lambda);
    }
}

fn collect_list(stmts: &[Stmt], g: &mut Global) {
    for s in stmts {
        match s {
            Stmt::Goto(_) | Stmt::Label(_) | Stmt::GotoCase(_) | Stmt::YieldReturn(_)
            | Stmt::YieldBreak => g.skip = true,
            Stmt::Assign(t, _, _) => {
                if is_plain_name(t) {
                    g.all.insert(t.clone());
                } else {
                    for tok in t.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                        if is_plain_name(tok) {
                            g.all.insert(tok.to_string());
                            g.bad.insert(tok.to_string());
                        }
                    }
                }
            }
            Stmt::ForEach {
                var_name,
                collection,
                ..
            } => {
                if is_plain_name(var_name) {
                    g.all.insert(var_name.clone());
                    g.bad.insert(var_name.clone());
                }
                if is_plain_name(collection) {
                    g.all.insert(collection.clone());
                }
            }
            Stmt::Switch { disc, .. } => {
                if is_plain_name(disc) {
                    g.all.insert(disc.clone());
                }
            }
            _ => {}
        }
        crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
            collect_expr(e, g, false)
        });
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_list(k, g);
        }
    }
}

fn expr_uses(e: &Expr, live: &mut HashSet<String>) {
    if let Expr::Var(n) = e {
        if is_plain_name(n) {
            live.insert(n.clone());
        }
    }
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
        expr_uses(c, live);
    }
}

/// Every name read by this statement's direct expressions (child lists are
/// handled by the structural walk).
fn add_uses(s: &Stmt, live: &mut HashSet<String>) {
    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| expr_uses(e, live));
    match s {
        Stmt::ForEach { collection, .. } => {
            if is_plain_name(collection) {
                live.insert(collection.clone());
            }
        }
        Stmt::Switch { disc, .. } => {
            if is_plain_name(disc) {
                live.insert(disc.clone());
            }
        }
        _ => {}
    }
}

/// Conservative use-collection for a subtree that the analysis does not
/// descend into (loop / try / using bodies).
fn reads_into(stmts: &[Stmt], live: &mut HashSet<String>) {
    for s in stmts {
        add_uses(s, live);
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            reads_into(k, live);
        }
    }
}

struct Ctx<'a> {
    g: &'a Global,
    removed: usize,
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    apply_protected(stmts, &HashSet::new())
}

/// As [`apply`], with names that must never be dropped even when the body
/// never reads them again (`ref`/`out` parameters: the caller reads them).
pub fn apply_protected(stmts: &mut Vec<Stmt>, protected: &HashSet<String>) -> usize {
    let mut g = Global::default();
    collect_list(stmts, &mut g);
    g.bad.extend(protected.iter().cloned());
    if g.skip {
        return 0;
    }
    let mut cx = Ctx { g: &g, removed: 0 };
    let mut live: HashSet<String> = HashSet::new();
    backward_list(stmts, &mut live, &mut cx, None);
    cx.removed
}

fn backward_list(
    list: &mut Vec<Stmt>,
    live: &mut HashSet<String>,
    cx: &mut Ctx,
    brk: Option<&HashSet<String>>,
) {
    let mut i = list.len();
    while i > 0 {
        i -= 1;
        let dead = match &list[i] {
            Stmt::Assign(t, _, rhs) => {
                is_plain_name(t) && is_pure(rhs) && !live.contains(t) && !cx.g.bad.contains(t)
            }
            _ => false,
        };
        if dead {
            list.remove(i);
            cx.removed += 1;
            continue;
        }
        match &mut list[i] {
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                let mut lt = live.clone();
                backward_list(then_body, &mut lt, cx, brk);
                let mut le = live.clone();
                backward_list(else_body, &mut le, cx, brk);
                lt.extend(le);
                *live = lt;
            }
            Stmt::Switch {
                cases,
                tail,
                default,
                ..
            } => {
                let mut l_tail = live.clone();
                backward_list(tail, &mut l_tail, cx, None);
                // An arm either falls into the shared tail or breaks out of
                // the switch; both continuations are live at the arm.
                let mut arm_out = l_tail.clone();
                arm_out.extend(live.iter().cloned());
                // A switch with NO default can skip every arm, so the
                // after-switch state is live at the switch entry. With a
                // default, every path executes one arm.
                let mut lin = if default.is_none() {
                    live.clone()
                } else {
                    HashSet::new()
                };
                for (_, body) in cases.iter_mut() {
                    let mut lb = arm_out.clone();
                    backward_list(body, &mut lb, cx, Some(&arm_out));
                    lin.extend(lb);
                }
                if let Some(arm) = default.as_mut() {
                    let mut lb = arm_out.clone();
                    backward_list(arm, &mut lb, cx, Some(&arm_out));
                    lin.extend(lb);
                }
                *live = lin;
            }
            Stmt::StringSwitch { cases, tail, .. } => {
                let mut l_default = live.clone();
                if let Some(t) = tail.as_mut() {
                    backward_list(t, &mut l_default, cx, None);
                }
                let mut arm_out = l_default.clone();
                arm_out.extend(live.iter().cloned());
                let mut lin = if tail.is_none() {
                    live.clone()
                } else {
                    HashSet::new()
                };
                lin.extend(l_default.iter().cloned());
                for (_, body) in cases.iter_mut() {
                    let mut lb = arm_out.clone();
                    backward_list(body, &mut lb, cx, brk);
                    lin.extend(lb);
                }
                *live = lin;
            }
            Stmt::Break => match brk {
                Some(l) => live.extend(l.iter().cloned()),
                None => live.extend(cx.g.all.iter().cloned()),
            },
            Stmt::Continue => {
                live.extend(cx.g.all.iter().cloned());
            }
            // Not descended: their bodies keep every store; their reads are
            // uses of the enclosing scope.
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Lock { body, .. } => {
                reads_into(body, live)
            }
            Stmt::For {
                init,
                incr,
                body,
                ..
            } => {
                reads_into(init, live);
                reads_into(incr, live);
                reads_into(body, live);
            }
            Stmt::ForEach { body, .. } => reads_into(body, live),
            Stmt::Using { decl, body } => {
                let d = std::slice::from_ref(decl.as_ref());
                reads_into(d, live);
                reads_into(body, live);
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                reads_into(try_body, live);
                for (_, b) in catches.iter() {
                    reads_into(b, live);
                }
                if let Some(fb) = finally_body {
                    reads_into(fb, live);
                }
            }
            _ => {}
        }
        add_uses(&list[i], live);
        if let Stmt::Assign(t, _, _) = &list[i] {
            if is_plain_name(t) {
                live.remove(t);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::BinOp;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }
    fn assign(n: &str, e: Expr) -> Stmt {
        Stmt::Assign(n.to_string(), None, e)
    }
    fn field(n: &str) -> Expr {
        Expr::Field(Box::new(Expr::This), n.to_string())
    }

    #[test]
    fn overwritten_pure_store_is_removed() {
        let mut stmts = vec![
            assign("x", field("A")),
            assign("x", field("B")),
            Stmt::Eval(var("x")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("\"A\""), "{d}");
        assert!(d.contains("\"B\""), "{d}");
    }

    #[test]
    fn store_read_on_one_path_is_kept() {
        let mut stmts = vec![
            assign("x", field("A")),
            Stmt::If {
                cond: var("c"),
                then_body: vec![Stmt::Eval(var("x"))],
                else_body: vec![],
            },
            assign("x", field("B")),
            Stmt::Eval(var("x")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
    }

    #[test]
    fn store_overwritten_in_both_arms_is_removed() {
        let mut stmts = vec![
            assign("x", field("A")),
            Stmt::If {
                cond: var("c"),
                then_body: vec![assign("x", field("B"))],
                else_body: vec![assign("x", field("C"))],
            },
            Stmt::Eval(var("x")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("\"A\""), "{d}");
    }

    #[test]
    fn store_not_read_before_end_is_removed() {
        let mut stmts = vec![
            assign("x", field("A")),
            assign(
                "y",
                Expr::Call {
                    recv: None,
                    name: "F".into(),
                    args: vec![],
                },
            ),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("\"A\""), "{d}");
        assert!(d.contains("F"), "{d}");
    }

    #[test]
    fn call_store_is_kept_but_earlier_pure_store_goes() {
        let mut stmts = vec![
            assign("x", field("A")),
            assign(
                "x",
                Expr::Call {
                    recv: None,
                    name: "F".into(),
                    args: vec![],
                },
            ),
            Stmt::Eval(var("x")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("\"A\""), "{d}");
        assert!(d.contains("F"), "{d}");
    }

    #[test]
    fn goto_methods_are_skipped() {
        let mut stmts = vec![
            assign("x", field("A")),
            assign("x", field("B")),
            Stmt::Goto("L1".into()),
            Stmt::Label("L1".into()),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
    }

    #[test]
    fn loop_body_is_not_rewritten_but_its_reads_keep_the_store() {
        let mut stmts = vec![
            assign("x", field("A")),
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![
                    assign("x", field("B")),
                    Stmt::Eval(var("x")),
                    Stmt::Eval(Expr::Bin(
                        BinOp::Add,
                        Box::new(var("x")),
                        Box::new(Expr::Int(1)),
                    )),
                ],
            },
        ];
        let n = apply(&mut stmts);
        // The first store is live because the loop body reads `x`.
        assert_eq!(n, 0, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(d.contains("\"A\""), "{d}");
        assert!(d.contains("\"B\""), "{d}");
    }

    #[test]
    fn address_taken_name_is_never_removed() {
        let mut stmts = vec![
            assign("x", field("A")),
            Stmt::Eval(Expr::AddressOf(Box::new(var("x")))),
            assign("x", field("B")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
    }

    #[test]
    fn lambda_capture_is_never_removed() {
        let mut stmts = vec![
            assign("x", field("A")),
            assign(
                "f",
                Expr::Lambda {
                    params: vec![],
                    body: Box::new(var("x")),
                },
            ),
            assign("x", field("B")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
    }

    #[test]
    fn switch_arm_break_keeps_the_read_after_the_switch() {
        let mut stmts = vec![
            assign("x", field("A")),
            Stmt::Switch {
                disc: "s".into(),
                cases: vec![(0, vec![Stmt::Break])],
                tail: vec![],
                default: None,
            },
            Stmt::Eval(var("x")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
    }

    #[test]
    fn switch_arm_overwrite_kills_earlier_store() {
        let mut stmts = vec![
            assign("x", field("A")),
            Stmt::Switch {
                disc: "s".into(),
                cases: vec![(0, vec![assign("x", field("B")), Stmt::Break])],
                tail: vec![],
                default: Some(vec![assign("x", field("C")), Stmt::Break]),
            },
            Stmt::Eval(var("x")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("\"A\""), "{d}");
    }
}
