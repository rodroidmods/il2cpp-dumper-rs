//! Delegate-constructor argument recovery.
//!
//! `new Action<T>(target, method)` — spelled `T..ctor(target, method)` in the
//! real corpus (there is no `new T(` anywhere in the dump) — is how IL2CPP
//! materialises a delegate. The method cell is usually loaded into a temporary
//! first, so the construction site names a LOCAL:
//!
//! ```text
//! v12 = MethodInfo("Foo.Bar");
//! d = System.Action..ctor(this, v12);
//! ```
//!
//! ## What this pass does
//!
//! It substitutes the METHOD GROUP the local provably holds, so the construction
//! reads with the method it names:
//!
//! ```text
//! v12 = MethodInfo("Foo.Bar");
//! d = System.Action..ctor(this, MethodInfo("Foo.Bar"));
//! ```
//!
//! That is a copy-propagation, not a guess: it never invents a name, and it
//! never changes which method is referenced.
//!
//! ## The proof
//!
//!   * the construction must name a DELEGATE type. `Type..ctor(a, b)` carries
//!     the type in the callee (`ctor_type_name`) when the callee is unresolved;
//!     a `recv..ctor(a, b)` spelling carries none, so it is refused — an
//!     arbitrary two-argument constructor is not a delegate;
//!   * the argument must be a plain local whose NEAREST preceding assignment in
//!     the same list is exactly a `MethodGroup`. A local assigned more than once
//!     does not prove which value reaches the call;
//!   * nothing between that assignment and the construction may mention the
//!     local at all (neither a read, nor a write in a nested body). That is what
//!     rules out a conditional re-assignment the neighbour scan cannot see;
//!   * the binding statement itself is left in place. Removing it would be dead
//!     store elimination, which needs whole-body liveness — out of scope for a
//!     single-list rewrite, and the remaining line is accurate.
//!
//! ## Measured status: the prerequisite was not observed (outcome "b")
//!
//! This pass reported **0** on the x64 corpus. Unlike `closures.rs` its input is
//! constructible in principle — `Expr::MethodGroup` IS produced by the renderer
//! (`csharp.rs`, `Operand::Method` → `Expr::MethodGroup`, which renders as
//! `MethodInfo("Foo.Bar")`) — so the zero is a statement about the OBSERVED
//! shape, not a proof of impossibility. Two facts must both hold for a fold,
//! and at least one does not on the measured corpus:
//!
//!   1. the construction must name a delegate TYPE. It is taken from the callee
//!      (`System.Action..ctor`), and a `recv..ctor(a, b)` spelling carries the
//!      receiver instead, i.e. no type at all — which this file refuses BY
//!      DESIGN, because an arbitrary two-argument constructor is not a
//!      delegate; and
//!   2. the method argument must be a plain local whose nearest preceding
//!      assignment in the same list is a `MethodGroup`. A method cell that
//!      arrives as a `MethodInfo` load (`RawDeref`/`Method` operand that the
//!      renderer did not turn into a `MethodGroup`) fails this.
//!
//! Which of the two holds on the real sites is NOT known from the corpus text
//! alone, and this pass stays the provable subset until it is measured: the
//! copy-propagation below never invents a name and never changes which method
//! is referenced. The missing input, stated for whoever measures it next, is a
//! delegate construction whose callee states its type AND whose method argument
//! is a local bound to a method group.
//!
//! ## What this pass deliberately does NOT do
//!
//! It does not synthesise a method name when the cell was lost. The estimated
//! original annotation for that case (`// delegate method cell not recovered`)
//! has been removed: the phi-collapse shape it keyed on
//! (`args[0] == args[1]` structurally) does not occur in the measured corpus,
//! and an annotation that fires nowhere is noise rather than recovery — the
//! `__unknown`/`__abi` markers already say the same thing where the value
//! really is unknown.

use super::{Expr, Stmt, ctor_type_name, is_ctor_name, list_mentions_deep, walk_mut};

/// Is `t` a delegate type name? Uses the simple (unqualified) name so
/// `System.Action`, `UnityAction`-style suffixes and the BCL delegate families
/// all match, without matching arbitrary user types called `Actionable`.
fn is_delegate_type(t: &str) -> bool {
    let base = t.rsplit('.').next().unwrap_or(t);
    let base = base.split('`').next().unwrap_or(base);
    base == "Action"
        || base == "Func"
        || base == "Predicate"
        || base == "Comparison"
        || base.ends_with("EventHandler")
        || base.ends_with("Callback")
        || base.ends_with("Delegate")
}

/// Is this expression a delegate construction? Returns its argument list.
fn delegate_alloc(e: &Expr) -> Option<&Vec<Expr>> {
    match e {
        Expr::NewObj(t, args) if is_delegate_type(t) => Some(args),
        Expr::Call { name, args, .. } if is_ctor_name(name) => {
            let ty = ctor_type_name(name)?;
            if is_delegate_type(ty) {
                Some(args)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The `MethodGroup` a local provably holds at `at`, if any.
fn bound_method_group(list: &[Stmt], at: usize, local: &str) -> Option<Expr> {
    // The nearest preceding assignment to `local` in this list …
    let bind = (0..at)
        .rev()
        .find(|&p| matches!(&list[p], Stmt::Assign(n, _, _) if n == local))?;
    let Stmt::Assign(_, _, Expr::MethodGroup(m)) = &list[bind] else {
        return None;
    };
    // … and nothing in between may mention the local at all.
    if list[bind + 1..at]
        .iter()
        .any(|s| list_mentions_deep(std::slice::from_ref(s), local))
    {
        return None;
    }
    Some(Expr::MethodGroup(m.clone()))
}

/// The local named as a delegate construction's method argument, if any.
fn delegate_arg_local(e: &Expr) -> Option<String> {
    let args = match e {
        Expr::NewObj(_, args) => args,
        Expr::Call { args, .. } => args,
        _ => return None,
    };
    match args.get(1) {
        Some(Expr::Var(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Rewrite delegate constructions inside `e`.
fn rewrite_expr(e: &mut Expr, list: &[Stmt], at: usize) -> usize {
    let mut n = 0usize;
    // Decide immutably first, then substitute — the argument borrow cannot be
    // held across the mutation.
    if delegate_alloc(e).is_some_and(|args| args.len() == 2) {
        if let Some(local) = delegate_arg_local(e) {
            if let Some(group) = bound_method_group(list, at, &local) {
                if let Expr::Call { args, .. } | Expr::NewObj(_, args) = e {
                    args[1] = group;
                    n += 1;
                }
            }
        }
    }
    match e {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            n += rewrite_expr(a, list, at);
            n += rewrite_expr(b, list, at);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            n += rewrite_expr(x, list, at);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            n += rewrite_expr(r, list, at);
        }
        Expr::Cond(c, a, b) => {
            n += rewrite_expr(c, list, at);
            n += rewrite_expr(a, list, at);
            n += rewrite_expr(b, list, at);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                n += rewrite_expr(r, list, at);
            }
            for a in args {
                n += rewrite_expr(a, list, at);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                n += rewrite_expr(a, list, at);
            }
        }
        Expr::Lambda { body, .. } => n += rewrite_expr(body, list, at),
        _ => {}
    }
    n
}

fn fold_list(list: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    // Snapshot the binding evidence before mutating: `rewrite_expr` needs to see
    // the list as it was.
    let snapshot: Vec<Stmt> = list.clone();
    for (i, s) in list.iter_mut().enumerate() {
        let slot: Option<&mut Expr> = match s {
            Stmt::Assign(_, _, e) => Some(e),
            Stmt::Eval(e) => Some(e),
            Stmt::Return(Some(e)) => Some(e),
            Stmt::Throw(e) => Some(e),
            Stmt::YieldReturn(e) => Some(e),
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
                Some(cond)
            }
            Stmt::For { cond: Some(c), .. } => Some(c),
            _ => None,
        };
        if let Some(e) = slot {
            n += rewrite_expr(e, &snapshot, i);
        }
    }
    n
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        n += fold_list(list);
    });
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.into())
    }
    /// `System.Action..ctor(target, method)`.
    fn ctor(method: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "System.Action..ctor".into(),
            args: vec![Expr::This, method],
        }
    }

    /// The real shape: the method cell sits in a local and the construction
    /// names that local.
    #[test]
    fn a_method_cell_local_is_substituted() {
        let mut stmts = vec![
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("Foo.Bar".to_string()),
            ),
            Stmt::Assign("d".to_string(), None, ctor(var("v12"))),
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[1] {
            Stmt::Assign(_, _, Expr::Call { args, .. }) => {
                assert!(
                    matches!(&args[1], Expr::MethodGroup(m) if m == "Foo.Bar"),
                    "the method group replaced the local"
                );
            }
            other => panic!("statement changed shape: {other:?}"),
        }
        // The binding stays: removing it would be dead-store elimination.
        assert!(matches!(&stmts[0], Stmt::Assign(n, _, _) if n == "v12"));
    }

    /// A local assigned twice: the NEAREST preceding assignment is what reaches
    /// the call, so that one is substituted.
    #[test]
    fn the_nearest_assignment_wins() {
        let mut stmts = vec![
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("Foo.Bar".to_string()),
            ),
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("Foo.Baz".to_string()),
            ),
            Stmt::Assign("d".to_string(), None, ctor(var("v12"))),
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[2] {
            Stmt::Assign(_, _, Expr::Call { args, .. }) => {
                assert!(matches!(&args[1], Expr::MethodGroup(m) if m == "Foo.Baz"));
            }
            other => panic!("statement changed shape: {other:?}"),
        }
    }

    /// A local whose value is not a method group proves nothing.
    #[test]
    fn a_non_method_group_binding_is_refused() {
        let mut stmts = vec![
            Stmt::Assign("v12".to_string(), None, var("other")),
            Stmt::Assign("d".to_string(), None, ctor(var("v12"))),
        ];
        let before = format!("{:?}", stmts);
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(before, format!("{:?}", stmts));
    }

    /// A write to the local between the binding and the call invalidates it.
    #[test]
    fn an_intervening_mention_invalidates_the_binding() {
        let mut stmts = vec![
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("Foo.Bar".to_string()),
            ),
            Stmt::If {
                cond: var("c"),
                then_body: vec![Stmt::Assign("v12".to_string(), None, var("x"))],
                else_body: vec![],
            },
            Stmt::Assign("d".to_string(), None, ctor(var("v12"))),
        ];
        assert_eq!(
            apply(&mut stmts),
            0,
            "a conditional re-assignment is invisible to a neighbour scan"
        );
    }

    /// A construction whose callee carries no type is not provably a delegate.
    #[test]
    fn a_recv_ctor_is_not_a_delegate() {
        let mut stmts = vec![
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("Foo.Bar".to_string()),
            ),
            Stmt::Assign(
                "d".to_string(),
                None,
                Expr::Call {
                    recv: Some(Box::new(var("v1"))),
                    name: ".ctor".into(),
                    args: vec![Expr::This, var("v12")],
                },
            ),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A non-delegate `Type..ctor` is untouched.
    #[test]
    fn a_non_delegate_ctor_is_untouched() {
        let mut stmts = vec![
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("Foo.Bar".to_string()),
            ),
            Stmt::Assign(
                "d".to_string(),
                None,
                Expr::Call {
                    recv: None,
                    name: "System.Collections.Generic.List`1..ctor".into(),
                    args: vec![Expr::This, var("v12")],
                },
            ),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A construction already naming its method group needs no rewrite.
    #[test]
    fn an_already_named_method_group_is_untouched() {
        let mut stmts = vec![Stmt::Eval(ctor(Expr::MethodGroup("Foo.Bar".to_string())))];
        assert_eq!(apply(&mut stmts), 0);
    }
}
