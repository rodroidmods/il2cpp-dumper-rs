//! Remove statements that write a local nobody ever reads.
//!
//! Call-argument staging registers (`instance5 = inError;`) and materialised
//! receiver loads (`instance = this.X;`) survive when a later pass rewrites
//! the call that consumed them: the variable keeps its stores and loses its
//! reads. `liveness_dse` cannot clean these up: it refuses every method that
//! contains a `goto`/`label`, and that is exactly where the surviving staging
//! lives. A WRITE-ONLY local needs no control-flow reasoning at all — if the
//! name is never read anywhere in the method (no expression use, no `&name`,
//! no rendered address path), then removing the store cannot change an
//! observable value.
//!
//! An impure store (`x = foo();`) whose destination is write-only is not
//! deleted — the call is the effect — but it loses the dead assignment and
//! becomes a plain `foo();`.
//!
//! Scope: method-wide over the structured tree (all branches, loops, EH,
//! lambdas). A name is considered read when it appears anywhere except as the
//! plain target of an assignment or a `Declare`.

use std::collections::HashSet;

use crate::recovery::csharp::{Expr, Stmt};

#[allow(dead_code)]
pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    apply_protected(stmts, &HashSet::new())
}

/// As [`apply`], with names treated as observed even when the body never
/// reads them (`ref`/`out` parameters: the caller reads them).
pub(crate) fn apply_protected(stmts: &mut Vec<Stmt>, protected: &HashSet<String>) -> usize {
    let mut mentioned: HashSet<String> = HashSet::new();
    collect_list(stmts, &mut mentioned);
    mentioned.extend(protected.iter().cloned());
    let mut n = 0;
    sweep_list(stmts, &mentioned, &mut n);
    n
}

fn is_plain_name(n: &str) -> bool {
    !n.is_empty()
        && n.chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !matches!(
            n,
            "this" | "base" | "null" | "true" | "false" | "new" | "default"
        )
}

/// A value expression with no effects: dropping its store drops nothing else.
fn value_pure(e: &Expr) -> bool {
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
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => value_pure(r),
        Expr::AddressOf(x) => value_pure(x),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => value_pure(a) && value_pure(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) => value_pure(x),
        _ => false,
    }
}

/// Every identifier token of a rendered path (`*(nint*)(&x + 8)`,
/// `list.Capacity`, …). Tokens are members and locals alike: a member token
/// only over-protects a local that shares its spelling, which is harmless.
fn note_tokens(text: &str, out: &mut HashSet<String>) {
    for tok in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if !tok.is_empty() && is_plain_name(tok) {
            out.insert(tok.to_string());
        }
    }
}

fn collect_expr(e: &Expr, out: &mut HashSet<String>) {
    if let Expr::Var(n) = e {
        if is_plain_name(n) {
            out.insert(n.clone());
        }
    }
    for c in super::il2cpp_cleanup::expr_children(e) {
        collect_expr(c, out);
    }
}

fn collect_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Assign(target, _, rhs) => {
            if is_plain_name(target) {
                collect_expr(rhs, out);
            } else {
                note_tokens(target, out);
                collect_expr(rhs, out);
            }
        }
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => collect_expr(e, out),
        Stmt::Return(Some(e)) => collect_expr(e, out),
        Stmt::If { cond, .. }
        | Stmt::While { cond, .. }
        | Stmt::DoWhile { cond, .. }
        | Stmt::Lock { expr: cond, .. } => collect_expr(cond, out),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                collect_expr(c, out);
            }
        }
        Stmt::Switch { disc, .. } => note_tokens(disc, out),
        Stmt::StringSwitch { disc, .. } => collect_expr(disc, out),
        Stmt::ForEach {
            var_name,
            collection,
            ..
        } => {
            out.insert(var_name.clone());
            note_tokens(collection, out);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            collect_expr(collection, out);
            collect_expr(index, out);
            collect_expr(value, out);
        }
        Stmt::EventAssign { field, value, .. } => {
            collect_expr(field, out);
            collect_expr(value, out);
        }
        Stmt::Attribute { args, .. } => {
            for a in args {
                collect_expr(a, out);
            }
        }
        _ => {}
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        collect_list(k, out);
    }
    if let Stmt::Using { decl, .. } = s {
        collect_stmt(decl, out);
    }
}

fn collect_list(list: &[Stmt], out: &mut HashSet<String>) {
    for s in list {
        collect_stmt(s, out);
    }
}

/// Rewrite one statement in place. Returns true when it was removed.
fn sweep_stmt(s: &mut Stmt, mentioned: &HashSet<String>, n: &mut usize) -> bool {
    match s {
        Stmt::Assign(t, _, rhs) if is_plain_name(t) && !mentioned.contains(t) => {
            if value_pure(rhs) {
                *n += 1;
                return true;
            }
            if matches!(rhs, Expr::Call { .. } | Expr::RawCall(..)) {
                let call = rhs.clone();
                *s = Stmt::Eval(call);
                *n += 1;
                return false;
            }
            if matches!(rhs, Expr::NewObj(_, args) if args.is_empty()) {
                // An UNREAD allocation is unobservable. `il2cpp_object_new`
                // runs no managed code, and a real `new T()` in source always
                // keeps its result (stored, passed, or thrown). A ZERO-ARGUMENT
                // allocation whose name is read nowhere is the runtime's
                // metadata token that `key_function_recovery` sampled into the
                // allocation family — `Camera.FireOnPreCull`'s
                // `var camera = new Camera();` inside its class-init guard,
                // `AppsFlyer.OpenSettings`'s entire "body". IR-level liveness
                // cannot see this: every call records `x0..x7` as ABI uses, so
                // the token looks consumed. An allocation WITH arguments is
                // kept — its constructor call may touch global state.
                *n += 1;
                return true;
            }
            false
        }
        Stmt::Declare(t, _) if is_plain_name(t) && !mentioned.contains(t) => {
            *n += 1;
            true
        }
        _ => false,
    }
}

fn sweep_list(list: &mut Vec<Stmt>, mentioned: &HashSet<String>, n: &mut usize) {
    let mut i = 0;
    while i < list.len() {
        if sweep_stmt(&mut list[i], mentioned, n) {
            list.remove(i);
            continue;
        }
        // `Using.decl` is deliberately left alone: `using (var r = …)` reads
        // `r` implicitly (Dispose), so it is never write-only.
        for k in super::children_mut(&mut list[i]) {
            sweep_list(k, mentioned, n);
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assign(name: &str, rhs: Expr) -> Stmt {
        Stmt::Assign(name.into(), None, rhs)
    }

    #[test]
    fn a_write_only_store_is_removed() {
        let mut s = vec![
            assign("x", Expr::Var("inError".into())),
            assign("y", Expr::Int(1)),
            Stmt::Eval(Expr::Var("y".into())),
        ];
        assert_eq!(apply(&mut s), 1);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn a_protected_out_parameter_keeps_its_store() {
        let mut s = vec![assign("outProgess", Expr::Var("this".into()))];
        let protected: HashSet<String> = ["outProgess".to_string()].into_iter().collect();
        assert_eq!(apply_protected(&mut s, &protected), 0);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn a_read_name_keeps_its_store() {
        let mut s = vec![assign("x", Expr::Int(1)), Stmt::Eval(Expr::Var("x".into()))];
        assert_eq!(apply(&mut s), 0);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn an_impure_store_becomes_a_bare_call() {
        let call = Expr::Call {
            recv: None,
            name: "Reset".into(),
            args: vec![],
        };
        let mut s = vec![assign("x", call)];
        assert_eq!(apply(&mut s), 1);
        assert!(matches!(&s[0], Stmt::Eval(Expr::Call { .. })));
    }

    /// The metadata-token shape (`Camera.FireOnPreCull`,
    /// `AppsFlyer.OpenSettings`): a bare allocation nobody reads.
    #[test]
    fn an_unread_bare_allocation_is_removed() {
        let mut s = vec![
            assign("camera", Expr::NewObj("Camera".into(), vec![])),
            Stmt::Eval(Expr::Var("cam".into())),
        ];
        assert_eq!(apply(&mut s), 1);
        assert_eq!(s.len(), 1);
    }

    /// Its constructor may write global state, so an allocation WITH arguments
    /// is not provably dead.
    #[test]
    fn an_unread_allocation_with_arguments_is_kept() {
        let mut s = vec![assign(
            "ex",
            Expr::NewObj("IndexOutOfRangeException".into(), vec![Expr::Int(1)]),
        )];
        assert_eq!(apply(&mut s), 0);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn a_read_allocation_keeps_its_statement() {
        let mut s = vec![
            assign("list", Expr::NewObj("List`1".into(), vec![])),
            Stmt::Eval(Expr::Var("list".into())),
        ];
        assert_eq!(apply(&mut s), 0);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn an_address_taken_name_keeps_its_store() {
        let mut s = vec![
            assign("x", Expr::Int(1)),
            Stmt::Assign("*(nint*)((nint)&x +8)".into(), None, Expr::Int(0)),
        ];
        assert_eq!(apply(&mut s), 0);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn a_read_in_a_nested_body_counts() {
        let mut s = vec![
            assign("x", Expr::Int(1)),
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![Stmt::Eval(Expr::Var("x".into()))],
            },
        ];
        assert_eq!(apply(&mut s), 0);
    }

    #[test]
    fn a_name_read_only_in_a_rendered_path_counts() {
        let mut s = vec![
            assign("x", Expr::Int(1)),
            Stmt::Assign("this.field = x.Capacity".into(), None, Expr::Int(0)),
        ];
        assert_eq!(apply(&mut s), 0);
    }
}
