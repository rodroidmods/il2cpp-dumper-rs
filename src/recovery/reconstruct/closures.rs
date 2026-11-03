//! Lambda / closure reconstruction.
//!
//! IL2CPP lowers a lambda to a display class (`<>c__DisplayClass…`), captured
//! fields, a generated method and a delegate construction. Where a recognizer
//! can PROVE the shape, it emits the source-level lambda instead:
//! `Expr::Lambda { params, body }`.
//!
//! ## The construction spelling on the real corpus
//!
//! `new T(` appears **zero** times in the real dump: every allocation is a
//! `.ctor` call. A delegate construction is therefore
//!
//! ```text
//! d = System.Action..ctor(this, MethodInfo("<>c.<M>b__1_0"));
//! ```
//!
//! i.e. `Expr::Call { name: "System.Action..ctor", args: [target, method] }` —
//! never `Expr::NewObj`. (The previous version of this file keyed on
//! `Expr::NewObj` and on a `Call` path whose "type" was the whole
//! `"System.Action..ctor"` string, which `is_delegate_type` then rejected as
//! `ctor`; between the two it could not fire at all.) Both spellings are
//! accepted now, and the type is taken from the callee with
//! `ctor_type_name`.
//!
//! ## What this pass can and cannot prove
//!
//! A generated lambda's **body is a different method** (`<>c.<M>b__N_M` is its
//! own definition); the construction site does not contain it. So the only
//! shape that can be folded WITHOUT inventing anything is the one where the
//! body is reachable from the same statement list — a value bound to the
//! generated method's name:
//!
//! ```text
//! <>c.<M>b__1_0 = (x) => x.Value;                     // the body is present
//! d = System.Action`1..ctor(this, MethodInfo("<>c.<M>b__1_0"));
//! ```
//!
//! which folds to `d = (x) => x.Value;`. When the body is not present the
//! construction is left EXACTLY as it was and the pass reports 0 for that site —
//! never a fabricated `() => MethodInfo(…)`. Recovering the cross-method case
//! needs the call graph (the callee's recovered body), which this statement-tree
//! pass deliberately does not have.
//!
//! A display-class instance field (`<>4__this`, captured locals) referenced from
//! inside the body is the closure capture; the field access is kept exactly as
//! written (no capture list is invented) — the value is what matters.
//!
//! ## Delegate-typed fields: `Delegate.Combine` / `Delegate.Remove`
//!
//! Roslyn spells `field += handler` for a delegate-typed field as
//! `field = Delegate.Combine(field, handler)` (and `-=` as `Delegate.Remove`).
//! When the right-hand side is PROVABLY a delegate value — a recovered
//! `Expr::Lambda`, or a local whose nearest preceding assignment is one — and
//! the combine's first operand is structurally the assignment target, the pair
//! is exactly C#'s `+=` / `-=` and is rewritten to `Stmt::EventAssign`. Two
//! shapes are accepted:
//!
//! ```text
//! field = Delegate.Combine(field, lambda);          // one statement
//! t = Delegate.Combine(field, lambda);
//! field = t;                                        // adjacent accumulator
//! ```
//!
//! Anything longer is NOT folded: a combine whose second operand is itself a
//! `Delegate.Combine`/`Delegate.Remove` (a multicast chain the statement tree
//! cannot prove the extent of), a first operand structurally different from the
//! store target, a target that is not a field (or a local proven delegate), and
//! an accumulator temporary with any other mention in the list. Without those
//! proofs the flat `Delegate.Combine(…)` call stays — accurate, if verbose.
//!
//! ## Where the input comes from
//!
//! Two producers exist in the pipeline:
//!   1. `lambda_inline::apply_with_maps` runs FIRST and rewrites a delegate
//!      construction to `Expr::Lambda` directly from the global lambda-body
//!      registry; and
//!   2. `csharp::inject_lambda_bodies` prepends a synthetic
//!      `Assign(name, _, Lambda)` for every construction it can see, which this
//!      pass then folds through `Bindings`.
//!
//! When neither producer fires — no registered body, or a method cell whose
//! identity the renderer lost — `bodies` is empty and the construction is left
//! exactly as written. A zero here therefore means "no provable body", not
//! "the shape is absent" (the `lambda_inline` counter reports the producer's
//! side), EXCEPT for the `Delegate.Combine` rule below, which needs no body and
//! folds on its own evidence.
//!
//! ## The arity gate
//!
//! When the delegate type states its arity, the recovered body must have that
//! many parameters. A mismatch means the binding is some other method's body
//! that happens to share a name, so the fold is refused rather than attaching
//! the wrong body.

use super::{Expr, Stmt, ctor_type_name, is_ctor_name, list_mentions_deep, walk_mut};

/// Is `name` a compiler-generated lambda method (`<…>b__N_M`)?
fn is_generated_lambda(name: &str) -> bool {
    let last = name.rsplit('.').next().unwrap_or(name);
    last.starts_with('<') && last.contains("b__")
}

/// Is `ty` plausibly a delegate type name? Deliberately conservative: only the
/// BCL delegate families plus the conventional `*EventHandler`/`*Callback`/
/// `*Delegate` suffixes, so an unrelated construction is never lambdafied.
fn is_delegate_type(ty: &str) -> bool {
    let t = ty.rsplit('.').next().unwrap_or(ty);
    let base = t.split('`').next().unwrap_or(t);
    base.starts_with("Action")
        || base.starts_with("Func")
        || base == "Predicate"
        || base == "Comparison"
        || base.ends_with("EventHandler")
        || base.ends_with("Delegate")
        || base.ends_with("Callback")
}

/// Number of type arguments in the first `<…>` of `s` (top-level commas + 1),
/// or `None` when there is no generic argument list.
fn angle_arity(s: &str) -> Option<usize> {
    let open = s.find('<')?;
    let close = s.rfind('>')?;
    if close <= open {
        return None;
    }
    let inner = &s[open + 1..close];
    if inner.trim().is_empty() {
        return Some(0);
    }
    let mut depth = 0i32;
    let mut n = 1usize;
    for c in inner.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => n += 1,
            _ => {}
        }
    }
    Some(n)
}

/// Parameter count of a delegate type: `Action`/`Action<N>`/`Action<…>` give N
/// parameters; `Func`/`Func<N>`/`Func<…>` give N-1 (the last type argument is
/// the return type). `None` when the name does not state an arity.
fn delegate_arity(ty: &str) -> Option<usize> {
    let t = ty.rsplit('.').next().unwrap_or(ty);
    let base = t.split('`').next().unwrap_or(t);
    let angle = angle_arity(t);
    let backtick = t
        .split('`')
        .nth(1)
        .and_then(|d| d.split('[').next())
        .and_then(|d| d.trim().parse::<usize>().ok());
    let args = angle.or(backtick);
    if base.starts_with("Action") {
        Some(args.unwrap_or(0))
    } else if base.starts_with("Func") {
        args.map(|a| a.saturating_sub(1))
    } else {
        None
    }
}

/// The same-list bindings a fold needs.
#[derive(Default)]
struct Bindings {
    /// generated method name → its body, from `Assign(<name>, _, Lambda{..})`.
    bodies: Vec<(String, Expr)>,
    /// local → generated method name, from `Assign(v, _, MethodGroup(name))`.
    aliases: Vec<(String, String)>,
}

impl Bindings {
    fn collect(list: &[Stmt]) -> Self {
        let mut b = Bindings::default();
        for s in list {
            if let Stmt::Assign(name, _, e) = s {
                match e {
                    Expr::Lambda { .. } if is_generated_lambda(name) => {
                        b.bodies.push((name.clone(), e.clone()));
                    }
                    Expr::MethodGroup(m) => b.aliases.push((name.clone(), m.clone())),
                    _ => {}
                }
            }
        }
        b
    }

    /// The generated method name a second argument names, directly or through a
    /// local bound to a `MethodGroup`.
    ///
    /// A local with more than one `MethodGroup` assignment does not prove which
    /// cell reaches the construction, so it is refused rather than resolved to
    /// the first binding.
    fn method_name<'a>(&'a self, e: &'a Expr) -> Option<&'a str> {
        match e {
            Expr::MethodGroup(m) => Some(m.as_str()),
            Expr::Var(v) => {
                let mut hits = self.aliases.iter().filter(|(n, _)| n == v);
                let (_, first) = hits.next()?;
                if hits.any(|(_, other)| other != first) {
                    return None;
                }
                Some(first.as_str())
            }
            _ => None,
        }
    }
}

/// Fold one statement list.
fn fold_list(list: &mut Vec<Stmt>) -> usize {
    let bindings = Bindings::collect(list);
    let mut n = 0usize;
    if !bindings.bodies.is_empty() {
        for s in list.iter_mut() {
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
                n += rewrite_expr(e, &bindings);
            }
        }
    }
    // Runs after the construction fold so a delegate construction on the right
    // is already an `Expr::Lambda` when this rule sees it.
    n += convert_event_assignments(list);
    n
}

/// `Delegate.Combine`/`Delegate.Remove` → `+=`/`-=`.
///
/// Returns `(add, source, handler)` where `add == true` is `Combine`.
fn combine_parts(e: &Expr) -> Option<(bool, &Expr, &Expr)> {
    let Expr::Call { recv: None, name, args } = e else {
        return None;
    };
    let add = if name == "Delegate.Combine" || name.ends_with(".Delegate.Combine") {
        true
    } else if name == "Delegate.Remove" || name.ends_with(".Delegate.Remove") {
        false
    } else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    // A second combine/remove is a multicast chain; its full extent (and hence
    // whether `+=` re-associates it) is not provable from this list.
    if combine_parts(&args[0]).is_some() || combine_parts(&args[1]).is_some() {
        return None;
    }
    Some((add, &args[0], &args[1]))
}

/// Is `e` a value the recovered tree PROVES to be a delegate?
///
/// A recovered `Expr::Lambda` is a delegate by construction. A local qualifies
/// only when the nearest preceding assignment in this list is a lambda or a
/// delegate-typed construction; a later reassignment to anything else makes the
/// question unprovable and the answer `false`.
fn proven_delegate_value(list: &[Stmt], before: usize, e: &Expr) -> bool {
    match e {
        Expr::Lambda { .. } => true,
        Expr::NewObj(ty, _) => is_delegate_type(ty),
        Expr::Call { name, .. } if is_ctor_name(name) => {
            ctor_type_name(name).is_some_and(is_delegate_type)
        }
        Expr::Var(v) => {
            let assignment = list[..before].iter().rev().find_map(|s| match s {
                Stmt::Assign(n, _, value) if n == v => Some(value),
                _ => None,
            });
            assignment.is_some_and(|value| proven_delegate_value(list, before, value))
        }
        _ => false,
    }
}

/// The rendered left-hand side of a store (`this.Evt`, `Game.OnTick`, `v3`).
fn rendered(e: &Expr) -> String {
    let mut s = String::new();
    e.render(&mut s);
    s
}

/// Is the store target a member path rather than a bare local?
///
/// `Assign` keeps its left side as the RENDERED text, so a field store arrives
/// as `this.Evt` / `Type.Field`; a bare identifier is a local (or a field the
/// renderer did not qualify) and needs separate type evidence. Raw pointer
/// stores (`*(nint*)(v3 +56)`) are refused: `nint += delegate` is not a
/// source-level event update, and the field annotation pass cannot prove the
/// slot is delegate-typed.
fn field_like_lhs(lhs: &str) -> bool {
    lhs.contains('.')
        && !lhs.contains('(')
        && !lhs.contains('[')
        && !lhs.starts_with('*')
        && !lhs.starts_with("&")
}

/// Does the store target agree with the combine's first operand?
fn same_lhs(source: &Expr, lhs: &str) -> bool {
    let text = rendered(source);
    text == lhs || text.replace(' ', "") == lhs.replace(' ', "")
}

/// The store target an `EventAssign` can describe without changing meaning.
///
/// A member path is accepted when the value combined into it is a proven
/// delegate (the RHS proof is done by the caller). A bare local is accepted
/// only when it was already assigned a delegate earlier in the list; otherwise
/// the local's type is unknown and `+=` could be a numeric/string operation.
fn event_target_ok(list: &[Stmt], at: usize, lhs: &str) -> bool {
    if field_like_lhs(lhs) {
        return true;
    }
    list[..at]
        .iter()
        .rev()
        .find_map(|s| match s {
            Stmt::Assign(n, _, value) if n == lhs => Some(value),
            _ => None,
        })
        .is_some_and(|value| proven_delegate_value(list, at, value))
}

/// Does any statement from `from` on mention `name` (read or write, at any
/// depth)? Used before deleting an accumulator temporary's store.
fn mentioned_from(list: &[Stmt], from: usize, name: &str) -> bool {
    list[from.min(list.len())..]
        .iter()
        .any(|s| list_mentions_deep(std::slice::from_ref(s), name))
}

/// One-statement shape: `target = Delegate.Combine(target, handler);`.
fn try_single_event_assign(list: &[Stmt], at: usize) -> Option<(Expr, Expr, bool)> {
    let Stmt::Assign(lhs, _, rhs) = &list[at] else {
        return None;
    };
    let (add, source, handler) = combine_parts(rhs)?;
    if !same_lhs(source, lhs) {
        return None;
    }
    if !event_target_ok(list, at, lhs) {
        return None;
    }
    if !proven_delegate_value(list, at, handler) {
        return None;
    }
    // `Expr::Var` renders the recovered left side verbatim, including the
    // member path the statement already carried.
    Some((Expr::Var(lhs.clone()), handler.clone(), add))
}

/// Adjacent accumulator shape:
///
/// ```text
/// t = Delegate.Combine(target, handler);
/// target = t;
/// ```
///
/// The temporary must be mentioned nowhere after the store, otherwise deleting
/// its binding would lose a value another statement observes.
fn try_pair_event_assign(list: &[Stmt], at: usize) -> Option<(Expr, Expr, bool)> {
    if at + 1 >= list.len() {
        return None;
    }
    let Stmt::Assign(temp, _, rhs) = &list[at] else {
        return None;
    };
    let Stmt::Assign(lhs, _, Expr::Var(stored)) = &list[at + 1] else {
        return None;
    };
    if stored != temp {
        return None;
    }
    let (add, source, handler) = combine_parts(rhs)?;
    if !same_lhs(source, lhs) {
        return None;
    }
    // The target must not be the accumulator itself (then the store is a
    // self-assignment, not an event update) and the temporary must not be
    // observed after the store.
    if lhs == temp || mentioned_from(list, at + 2, temp) {
        return None;
    }
    if !event_target_ok(list, at, lhs) {
        return None;
    }
    if !proven_delegate_value(list, at, handler) {
        return None;
    }
    Some((Expr::Var(lhs.clone()), handler.clone(), add))
}

/// Rewrite the `Delegate.Combine`/`Delegate.Remove` assignment shapes to
/// `Stmt::EventAssign`. Every failed proof is a no-op.
fn convert_event_assignments(list: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    let mut i = 0usize;
    while i < list.len() {
        if let Some((field, value, add)) = try_single_event_assign(list, i) {
            list[i] = Stmt::EventAssign { field, value, add };
            n += 1;
            i += 1;
            continue;
        }
        if let Some((field, value, add)) = try_pair_event_assign(list, i) {
            list[i] = Stmt::EventAssign { field, value, add };
            list.remove(i + 1);
            n += 1;
            // Re-examine the same index: the removed store may have exposed
            // another adjacent pair.
            continue;
        }
        i += 1;
    }
    n
}

/// Rewrite delegate constructions inside `e`, recursing into children.
fn rewrite_expr(e: &mut Expr, b: &Bindings) -> usize {
    if let Some(lambda) = try_fold(e, b) {
        *e = lambda;
        return 1;
    }
    let mut n = 0usize;
    match e {
        Expr::Bin(_, a, c) | Expr::Cmp(_, a, c) => {
            n += rewrite_expr(a, b);
            n += rewrite_expr(c, b);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            n += rewrite_expr(x, b);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            n += rewrite_expr(r, b);
        }
        Expr::Cond(c, a, d) => {
            n += rewrite_expr(c, b);
            n += rewrite_expr(a, b);
            n += rewrite_expr(d, b);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                n += rewrite_expr(r, b);
            }
            for a in args {
                n += rewrite_expr(a, b);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                n += rewrite_expr(a, b);
            }
        }
        Expr::Lambda { body, .. } => n += rewrite_expr(body, b),
        _ => {}
    }
    n
}

/// The delegate-construction → lambda fold, applied to ONE expression.
///
/// Fires only when the two-operand delegate shape is present, the construction
/// names a delegate TYPE (so an unrelated `.ctor` call is never touched), the
/// generated method's body is bound in the same list, and the body's parameter
/// count agrees with the delegate type's arity.
fn try_fold(e: &Expr, b: &Bindings) -> Option<Expr> {
    // The type name of the construction, and its argument list.
    let (ty, args): (Option<&str>, &[Expr]) = match e {
        Expr::NewObj(ty, args) => (Some(ty.as_str()), args.as_slice()),
        Expr::Call { name, args, .. } if is_ctor_name(name) => {
            (ctor_type_name(name), args.as_slice())
        }
        _ => return None,
    };
    if args.len() != 2 {
        return None;
    }
    // A `recv..ctor(target, method)` spelling carries no type in the tree, so
    // nothing proves it is a delegate: refuse rather than guess.
    let ty = ty?;
    let known_delegate = is_delegate_type(ty);
    if !known_delegate && !matches!(ty, "object" | "System.Object" | "") {
        return None;
    }
    let m = b.method_name(&args[1])?;
    if !is_generated_lambda(m) {
        return None;
    }
    let (_, binding) = b.bodies.iter().find(|(n, _)| n == m)?;
    let Expr::Lambda { params, .. } = binding else {
        return None;
    };
    if known_delegate {
        if let Some(arity) = delegate_arity(ty) {
            if params.len() != arity {
                return None;
            }
        }
    }
    Some(binding.clone())
}

/// Run the closure recognizer over one method body.
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

    fn lambda(params: &[&str], body: Expr) -> Expr {
        Expr::Lambda {
            params: params.iter().map(|p| p.to_string()).collect(),
            body: Box::new(body),
        }
    }
    fn var(n: &str) -> Expr {
        Expr::Var(n.into())
    }
    /// `System.Action`1..ctor(this, <method>)` — the real construction spelling.
    fn ctor(method: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "System.Action`1..ctor".into(),
            args: vec![Expr::This, method],
        }
    }

    /// The provable shape in the REAL spelling: the generated body is bound in
    /// the same list, then the construction referencing it folds.
    #[test]
    fn ctor_call_with_a_provable_body_folds() {
        let name = "<>c.<M>b__1_0";
        let body = lambda(&["x"], Expr::Field(Box::new(var("x")), "Value".into()));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body.clone()),
            Stmt::Assign(
                "action".to_string(),
                None,
                ctor(Expr::MethodGroup(name.to_string())),
            ),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "the construction must fold to the lambda");
        match &stmts[1] {
            Stmt::Assign(_, _, Expr::Lambda { params, .. }) => assert_eq!(params.len(), 1),
            other => panic!("expected a folded lambda, got {other:?}"),
        }
    }

    /// The real IL2CPP indirection: the method cell is loaded into a local and
    /// the construction names that local.
    #[test]
    fn a_method_cell_held_in_a_local_is_resolved() {
        let name = "<>c.<M>b__2_0";
        let body = lambda(&["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Assign("v12".to_string(), None, Expr::MethodGroup(name.to_string())),
            Stmt::Assign("d".to_string(), None, ctor(var("v12"))),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[2], Stmt::Assign(_, _, Expr::Lambda { .. })));
    }

    /// Refusal: with no provable body in the list, the construction is left
    /// EXACTLY as it was — never a fabricated lambda.
    #[test]
    fn ctor_call_without_a_provable_body_is_untouched() {
        let mut stmts = vec![Stmt::Eval(ctor(Expr::MethodGroup(
            "<>c.<M>b__1_0".to_string(),
        )))];
        let before = format!("{:?}", stmts);
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(
            before,
            format!("{:?}", stmts),
            "the call must be preserved verbatim"
        );
    }

    /// A mismatched binding (wrong arity for the delegate type) is refused even
    /// though the name matches.
    #[test]
    fn arity_mismatch_is_refused() {
        let name = "<>c.<M>b__2_0";
        let body = lambda(&[], var("x")); // 0 params, Action`1 wants 1
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(ctor(Expr::MethodGroup(name.to_string()))),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A construction whose callee carries no type (`v12..ctor(a, b)`) proves
    /// nothing about being a delegate.
    #[test]
    fn a_recv_ctor_without_a_type_is_refused() {
        let name = "<>c.<M>b__1_0";
        let body = lambda(&["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("v12"))),
                name: ".ctor".into(),
                args: vec![Expr::This, Expr::MethodGroup(name.to_string())],
            }),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A non-delegate construction is never lambdafied.
    #[test]
    fn non_delegate_constructions_are_not_lambdafied() {
        let name = "<>c.<M>b__1_0";
        let body = lambda(&["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "System.Collections.Generic.List`1..ctor".into(),
                args: vec![Expr::This, Expr::MethodGroup(name.to_string())],
            }),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// The legacy `Expr::NewObj` spelling is still accepted.
    #[test]
    fn the_newobj_spelling_is_still_accepted() {
        let name = "<>c.<M>b__1_0";
        let body = lambda(&["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(Expr::NewObj(
                "System.Action`1".into(),
                vec![Expr::This, Expr::MethodGroup(name.to_string())],
            )),
        ];
        assert_eq!(apply(&mut stmts), 1);
    }

    #[test]
    fn delegate_arity_is_read_from_the_name() {
        assert_eq!(delegate_arity("System.Action"), Some(0));
        assert_eq!(delegate_arity("System.Action`1"), Some(1));
        assert_eq!(delegate_arity("System.Action`2"), Some(2));
        assert_eq!(delegate_arity("System.Func`2"), Some(1));
        assert_eq!(delegate_arity("System.Func`3"), Some(2));
        assert_eq!(delegate_arity("Dictionary<string, List<int>>"), None);
        assert_eq!(angle_arity("Action<Foo, Bar>"), Some(2));
    }

    // -----------------------------------------------------------------------
    // Type-lost constructions
    // -----------------------------------------------------------------------

    /// `normalize_ctor_calls` merges the allocation and the `.ctor`, and the
    /// allocation's type argument is lost, so the construction is spelled
    /// `NewObj("object", …)`. Once the second operand names a generated lambda
    /// with a binding in the list, the shape is the delegate construction.
    #[test]
    fn object_typed_construction_with_a_binding_folds() {
        let name = "<M>b__4_0";
        let body = lambda(&[], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Assign(
                "d".to_string(),
                None,
                Expr::NewObj(
                    "object".into(),
                    vec![Expr::This, Expr::MethodGroup(name.to_string())],
                ),
            ),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[1], Stmt::Assign(_, _, Expr::Lambda { .. })));
    }

    /// A non-`object`, non-delegate construction is still refused.
    #[test]
    fn a_non_object_unknown_type_is_refused() {
        let name = "<M>b__4_1";
        let body = lambda(&[], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(Expr::NewObj(
                "SomeRandomType".into(),
                vec![Expr::This, Expr::MethodGroup(name.to_string())],
            )),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    // -----------------------------------------------------------------------
    // Delegate.Combine / Delegate.Remove → += / -=
    // -----------------------------------------------------------------------

    fn combine(source: Expr, handler: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "Delegate.Combine".into(),
            args: vec![source, handler],
        }
    }

    fn remove(source: Expr, handler: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "Delegate.Remove".into(),
            args: vec![source, handler],
        }
    }

    fn event_field(name: &str) -> Expr {
        Expr::Field(Box::new(Expr::This), name.into())
    }

    /// `this.Evt = Delegate.Combine(this.Evt, lambda)` → `this.Evt += lambda`.
    #[test]
    fn combine_assignment_becomes_add_event() {
        let field = event_field("Evt");
        let mut stmts = vec![Stmt::Assign(
            "this.Evt".to_string(),
            None,
            combine(field, lambda(&["x"], var("x"))),
        )];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::EventAssign { field, value, add } => {
                assert_eq!(rendered(field), "this.Evt");
                assert!(matches!(value, Expr::Lambda { .. }));
                assert!(*add);
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    /// A store target that is not the combine's source is some other
    /// assignment, not an event update: refuse.
    #[test]
    fn mismatched_store_target_is_refused() {
        let mut stmts = vec![Stmt::Assign(
            "this.Other".to_string(),
            None,
            combine(event_field("Evt"), lambda(&[], var("x"))),
        )];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::Assign(..)));
    }

    /// `Delegate.Remove(field, lambda)` is the `-=` spelling.
    #[test]
    fn remove_assignment_becomes_remove_event() {
        let mut stmts = vec![Stmt::Assign(
            "this.Evt".to_string(),
            None,
            remove(event_field("Evt"), lambda(&[], var("x"))),
        )];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(
            &stmts[0],
            Stmt::EventAssign { add: false, .. }
        ));
    }

    /// The adjacent accumulator shape folds and drops the temporary binding.
    #[test]
    fn combine_pair_becomes_add_event() {
        let field = event_field("Evt");
        let mut stmts = vec![
            Stmt::Assign(
                "t".to_string(),
                None,
                combine(field, lambda(&["x"], var("x"))),
            ),
            Stmt::Assign("this.Evt".to_string(), None, var("t")),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert_eq!(stmts.len(), 1, "the accumulator store is gone");
        match &stmts[0] {
            Stmt::EventAssign { field, add, .. } => {
                assert_eq!(rendered(field), "this.Evt");
                assert!(*add);
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    /// A static member path carries its own type evidence.
    #[test]
    fn static_field_target_folds() {
        let field = Expr::StaticField("Game".into(), "OnTick".into());
        let mut stmts = vec![Stmt::Assign(
            "Game.OnTick".to_string(),
            None,
            combine(field, lambda(&[], var("x"))),
        )];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::EventAssign { .. }));
    }

    /// A multicast chain (`Combine(Combine(field, a), b)`) is refused: `+=`
    /// would re-associate a chain the list cannot prove.
    #[test]
    fn multicast_chain_is_refused() {
        let field = event_field("Evt");
        let inner = combine(field, lambda(&[], var("a")));
        let mut stmts = vec![Stmt::Assign(
            "this.Evt".to_string(),
            None,
            combine(inner, lambda(&[], var("b"))),
        )];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::Assign(..)));
    }

    /// A combine whose handler is not a proven delegate is refused.
    #[test]
    fn non_delegate_handler_is_refused() {
        let mut stmts = vec![Stmt::Assign(
            "this.Evt".to_string(),
            None,
            combine(
                event_field("Evt"),
                Expr::Call {
                    recv: None,
                    name: "GetHandler".into(),
                    args: vec![],
                },
            ),
        )];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A local target is only accepted when an earlier assignment proves it is
    /// delegate-typed.
    #[test]
    fn local_target_needs_delegate_evidence() {
        let unproven = vec![Stmt::Assign(
            "d".to_string(),
            None,
            combine(var("d"), lambda(&[], var("x"))),
        )];
        let mut stmts = unproven;
        assert_eq!(apply(&mut stmts), 0);

        let mut stmts = vec![
            Stmt::Assign("d".to_string(), None, lambda(&[], var("a"))),
            Stmt::Assign(
                "d".to_string(),
                None,
                combine(var("d"), lambda(&[], var("b"))),
            ),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[1], Stmt::EventAssign { add: true, .. }));
    }

    /// The accumulator temporary must not be observed after its store.
    #[test]
    fn accumulator_temporary_with_a_later_read_is_refused() {
        let mut stmts = vec![
            Stmt::Assign(
                "t".to_string(),
                None,
                combine(event_field("Evt"), lambda(&[], var("x"))),
            ),
            Stmt::Assign("this.Evt".to_string(), None, var("t")),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Observe".into(),
                args: vec![var("t")],
            }),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(stmts.len(), 3);
    }
}
