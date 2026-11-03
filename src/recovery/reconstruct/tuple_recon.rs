//! Tuple/deconstruction pattern reconstruction:
//! `ValueTuple.Create(a, b, c)` -> `(a, b, c)`,
//! `new ValueTuple<T1, T2>(a, b)` -> `(a, b)`,
//! sequences of `Item1`/`Item2`/... field reads -> tuple deconstruction.
//!
//! ## What this pass does
//!
//! IL2CPP renders C# tuple constructions as either:
//!
//! ```text
//! ValueTuple.Create(a, b, c)                       ->  (a, b, c)
//! new ValueTuple<T1, T2, T3>(a, b, c)              ->  (a, b, c)
//! System.ValueTuple`3..ctor(a, b, c)               ->  (a, b, c)
//! ```
//!
//! And tuple deconstruction patterns:
//!
//! ```text
//! x = tuple.Item1;  y = tuple.Item2;              ->  (x, y) = tuple;
//! ```
//!
//! Named tuples are also supported when the call uses `ValueTuple.Create`
//! with named argument metadata (rare in IL2CPP but handled for completeness).
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - `ValueTuple.Create` calls with 2..=8 arguments (its real overloads)
//! - `NewObj`/ctor calls whose type name is `ValueTuple` and whose declared
//!   arity (`ValueTuple`3`, `ValueTuple<T1, T2>`) equals the argument count
//! - Consecutive, untyped `Item1`/`Item2`/... field reads from the same stable
//!   source (`v`, `this.f`), with distinct plain-identifier targets. A run is
//!   refused when the source is a call/index (merging would change how many
//!   times it is evaluated), when a target aliases the source (`t = t.Item1`),
//!   when a target is not a plain identifier, or when a later read is
//!   interleaved with a write to the source.
//! - No element names are invented: named-tuple metadata is not present in the
//!   IL2CPP field reads, so `names` stays empty/`None`.
//! - A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{Expr, Stmt};

/// Run the tuple reconstruction pass over one method body.
///
/// Returns the number of rewrites applied.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    super::walk_mut(stmts, &mut |list| {
        count += rewrite_tuple_creates(list);
        count += rewrite_tuple_deconstructions(list);
    });
    count
}

/// Detect `ValueTuple.Create(a, b, ...)` and `new ValueTuple<T...>(a, b, ...)`
/// calls and rewrite them to `Expr::Tuple`.
fn rewrite_tuple_creates(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    for stmt in stmts.iter_mut() {
        count += rewrite_tuple_in_stmt(stmt);
    }
    count
}

/// Recursively rewrite tuple-creating expressions within a statement.
fn rewrite_tuple_in_stmt(stmt: &mut Stmt) -> usize {
    let mut count = 0;
    match stmt {
        Stmt::Eval(expr) => count += rewrite_tuple_expr(expr),
        Stmt::Assign(_, _, expr) => count += rewrite_tuple_expr(expr),
        Stmt::Return(Some(expr)) => count += rewrite_tuple_expr(expr),
        Stmt::Throw(expr) => count += rewrite_tuple_expr(expr),
        Stmt::YieldReturn(expr) => count += rewrite_tuple_expr(expr),
        Stmt::If { cond, then_body, else_body, .. } => {
            count += rewrite_tuple_expr(cond);
            count += rewrite_tuple_creates(then_body);
            count += rewrite_tuple_creates(else_body);
        }
        Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
            count += rewrite_tuple_expr(cond);
            count += rewrite_tuple_creates(body);
        }
        Stmt::For { init, cond, incr, body } => {
            for s in init {
                count += rewrite_tuple_in_stmt(s);
            }
            if let Some(c) = cond {
                count += rewrite_tuple_expr(c);
            }
            for s in incr {
                count += rewrite_tuple_in_stmt(s);
            }
            count += rewrite_tuple_creates(body);
        }
        Stmt::ForEach { body, .. } => count += rewrite_tuple_creates(body),
        Stmt::Using { decl, body } => {
            count += rewrite_tuple_in_stmt(decl);
            count += rewrite_tuple_creates(body);
        }
        Stmt::Lock { expr, body } => {
            count += rewrite_tuple_expr(expr);
            count += rewrite_tuple_creates(body);
        }
        Stmt::Try { try_body, catches, finally_body } => {
            count += rewrite_tuple_creates(try_body);
            for (_, b) in catches {
                count += rewrite_tuple_creates(b);
            }
            if let Some(fb) = finally_body {
                count += rewrite_tuple_creates(fb);
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases {
                count += rewrite_tuple_creates(b);
            }
            count += rewrite_tuple_creates(tail);
            if let Some(arm) = default {
                count += rewrite_tuple_creates(arm);
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, b) in cases {
                count += rewrite_tuple_creates(b);
            }
            if let Some(t) = tail {
                count += rewrite_tuple_creates(t);
            }
        }
        _ => {}
    }
    count
}

/// Recursively rewrite tuple-creating expressions within an expression tree.
/// Post-order: children first, then check this node.
fn rewrite_tuple_expr(expr: &mut Expr) -> usize {
    let mut count = 0;

    // Recurse into subexpressions first (post-order).
    match expr {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += rewrite_tuple_expr(a);
            count += rewrite_tuple_expr(b);
        }
        Expr::Not(e) | Expr::Neg(e) | Expr::Cast(_, e) | Expr::AddressOf(e) | Expr::Await(e) => {
            count += rewrite_tuple_expr(e);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            count += rewrite_tuple_expr(r);
        }
        Expr::Cond(c, a, b) => {
            count += rewrite_tuple_expr(c);
            count += rewrite_tuple_expr(a);
            count += rewrite_tuple_expr(b);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                count += rewrite_tuple_expr(r);
            }
            for a in args {
                count += rewrite_tuple_expr(a);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                count += rewrite_tuple_expr(a);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for e in items {
                count += rewrite_tuple_expr(e);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for e in ctor_args {
                count += rewrite_tuple_expr(e);
            }
            for e in items {
                count += rewrite_tuple_expr(e);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for e in ctor_args {
                count += rewrite_tuple_expr(e);
            }
            for (_, v) in fields {
                count += rewrite_tuple_expr(v);
            }
        }
        Expr::Index { collection, index } => {
            count += rewrite_tuple_expr(collection);
            count += rewrite_tuple_expr(index);
        }
        Expr::Lambda { body, .. } => {
            count += rewrite_tuple_expr(body);
        }
        Expr::NullConditional { receiver, access } => {
            count += rewrite_tuple_expr(receiver);
            count += rewrite_tuple_expr(access);
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                count += rewrite_tuple_expr(e);
            }
        }
        Expr::MethodChain { receiver, methods } => {
            count += rewrite_tuple_expr(receiver);
            for m in methods {
                for a in &mut m.args {
                    count += rewrite_tuple_expr(a);
                }
            }
        }
        Expr::SwitchExpr { discrim, arms, default_arm } => {
            count += rewrite_tuple_expr(discrim);
            for (p, r) in arms {
                count += rewrite_tuple_expr(p);
                count += rewrite_tuple_expr(r);
            }
            if let Some(d) = default_arm {
                count += rewrite_tuple_expr(d);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                count += rewrite_tuple_expr(f);
            }
            if let Some(t) = to {
                count += rewrite_tuple_expr(t);
            }
        }
        Expr::IndexFromEnd(inner) => {
            count += rewrite_tuple_expr(inner);
        }
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
        | Expr::StaticField(..)
        | Expr::MethodGroup(_)
        | Expr::PatternMatch { .. }
        | Expr::StringInterp { .. } | Expr::TypeOf(_) => {}
    }

    // Now check if THIS expression is a tuple creation we can rewrite.
    if let Some(tuple) = try_rewrite_as_tuple(expr) {
        *expr = tuple;
        count += 1;
    }

    count
}

/// If `expr` is a `ValueTuple.Create(...)` or `new ValueTuple<...>(...)` call,
/// return the equivalent `Expr::Tuple`. Otherwise return `None`.
fn try_rewrite_as_tuple(expr: &Expr) -> Option<Expr> {
    let args = match expr {
        // Pattern 1: ValueTuple.Create(a, b, c) — static call, no receiver.
        // The real overloads take 1..=8 elements; C# tuple syntax needs 2+.
        Expr::Call {
            recv: None,
            name,
            args,
        } if is_value_tuple_create(name) && (2..=8).contains(&args.len()) => args,
        // Pattern 2: new ValueTuple<T1, T2, ...>(a, b, ...) — NewObj with
        // ValueTuple type. The declared arity must equal the argument count.
        Expr::NewObj(type_name, args)
            if is_value_tuple_type(type_name)
                && tuple_type_arity(type_name) == Some(args.len())
                && args.len() >= 2 =>
        {
            args
        }
        // Pattern 3: ValueTuple`N..ctor(a, b, ...) — ctor call on ValueTuple
        // type, same arity proof.
        Expr::Call {
            recv: None,
            name,
            args,
        } if is_value_tuple_ctor(name)
            && tuple_ctor_arity(name) == Some(args.len())
            && args.len() >= 2 =>
        {
            args
        }
        _ => return None,
    };
    let elements = args.clone();
    // Named-tuple names live in attributes, not in the IL2CPP field layout;
    // never synthesise them.
    let names = vec![None; elements.len()];
    Some(Expr::Tuple { names, elements })
}

/// Does this call name look like `ValueTuple.Create`?
fn is_value_tuple_create(name: &str) -> bool {
    name == "ValueTuple.Create"
        || name.ends_with(".ValueTuple.Create")
        || name == "System.ValueTuple.Create"
        || name.ends_with("ValueTuple.Create")
}

/// Does this type name look like a `ValueTuple<...>`?
fn is_value_tuple_type(type_name: &str) -> bool {
    type_name.starts_with("ValueTuple")
        || type_name.contains("ValueTuple`")
        || type_name.contains("ValueTuple<")
        || type_name.starts_with("System.ValueTuple")
}

/// Declared arity of a tuple type name: ``ValueTuple`3`` or
/// `ValueTuple<int, string>` (nested generics counted at depth 0 only).
fn tuple_type_arity(type_name: &str) -> Option<usize> {
    if let Some((_, tail)) = type_name.rsplit_once('`') {
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return digits.parse().ok();
        }
    }
    let open = type_name.find('<')?;
    let close = type_name.rfind('>')?;
    if close <= open + 1 {
        return None;
    }
    let inner = &type_name[open + 1..close];
    let mut depth = 0usize;
    let mut count = 1usize;
    for c in inner.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => count += 1,
            _ => {}
        }
    }
    Some(count)
}

/// Declared arity of a ValueTuple constructor name (`ValueTuple`2..ctor`).
fn tuple_ctor_arity(name: &str) -> Option<usize> {
    let base = name
        .strip_suffix("..ctor")
        .or_else(|| name.strip_suffix(".ctor"))
        .unwrap_or(name);
    tuple_type_arity(base)
}

/// Does this ctor name belong to a ValueTuple constructor?
fn is_value_tuple_ctor(name: &str) -> bool {
    // Match patterns like "ValueTuple`2..ctor", "System.ValueTuple`3..ctor"
    let base = name.strip_suffix("..ctor")
        .or_else(|| name.strip_suffix(".ctor"))
        .unwrap_or(name);
    is_value_tuple_type(base)
}

/// Detect tuple deconstruction patterns: consecutive `Item1`/`Item2`/...
/// field reads from the same source variable, and rewrite them.
///
/// Pattern:
/// ```text
/// x = src.Item1;
/// y = src.Item2;
/// ```
/// becomes:
/// ```text
/// (x, y) = src;  // rendered as Assign("(x, y)", None, Var("src"))
/// ```
///
/// Since the Stmt::Assign target is a String, we encode the deconstruction
/// as a special assignment target that the renderer can handle.
fn rewrite_tuple_deconstructions(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < stmts.len() {
        // Try to find a run of consecutive Item1, Item2, ... reads.
        if let Some((decon, run_len)) = try_match_deconstruction(&stmts[i..]) {
            // Replace the first statement with the deconstruction.
            stmts[i] = decon;
            // Remove the remaining statements in the run.
            for _ in 1..run_len {
                stmts.remove(i + 1);
            }
            count += 1;
            // Don't advance i — check for another run at the same position.
        } else {
            i += 1;
        }
    }
    count
}

/// Try to match a deconstruction pattern starting at `stmts[0]`.
/// Returns the replacement statement and the number of statements consumed.
fn try_match_deconstruction(stmts: &[Stmt]) -> Option<(Stmt, usize)> {
    // We need at least 2 consecutive Item1/Item2 assignments.
    if stmts.len() < 2 {
        return None;
    }

    // First statement must be: Assign(x, _, Field(src, "Item1")).
    let (first_target, src_expr) = match_item_read(&stmts[0], 1)?;

    // The merge turns two evaluations of `src` into one. That is only
    // semantics-preserving for a stable location, and only legal when the
    // first target does not alias it (`t = t.Item1` changes what the second
    // read sees).
    if !is_stable_source(&src_expr) || !is_plain_target(&first_target) {
        return None;
    }
    if expr_reads_name(&src_expr, &first_target) {
        return None;
    }

    // Collect consecutive Item reads.
    let mut targets = vec![first_target];
    let mut run_len = 1;

    for expected_item in 2.. {
        if run_len >= stmts.len() {
            break;
        }
        match match_item_read(&stmts[run_len], expected_item) {
            Some((target, src))
                if exprs_eq_static(&src, &src_expr)
                    && is_plain_target(&target)
                    // Distinct targets: `(x, x) = t` is not valid C#.
                    && !targets.contains(&target)
                    // No write may alias the source mid-run.
                    && !expr_reads_name(&src_expr, &target) =>
            {
                targets.push(target);
                run_len += 1;
            }
            _ => break,
        }
    }

    // Need at least 2 items for a deconstruction.
    if targets.len() < 2 {
        return None;
    }

    // Verify items are sequential: Item1, Item2, ..., ItemN.
    // (Already guaranteed by the loop above.)

    // Build the deconstruction statement.
    // We encode it as Assign("(x, y, z)", None, src_expr) — the renderer
    // will recognize the parenthesized target as a deconstruction.
    let decon_target = format!("({})", targets.join(", "));
    Some((
        Stmt::Assign(decon_target, None, src_expr),
        run_len,
    ))
}

/// If `stmt` is `Assign(target, _, Field(src, "ItemN"))`, return
/// `Some((target, src_expr))` where N == `expected_item`.
///
/// Only untyped assignments match: a typed `int x = t.Item1` is a declaration
/// and cannot be folded into `(x, y) = t`.
fn match_item_read(stmt: &Stmt, expected_item: u32) -> Option<(String, Expr)> {
    match stmt {
        Stmt::Assign(target, None, Expr::Field(base, field)) => {
            let expected_field = format!("Item{expected_item}");
            if field == &expected_field {
                Some((target.clone(), base.as_ref().clone()))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A target that can appear in a tuple deconstruction assignment: a plain
/// identifier or the discard `_`. Field/index targets are rendered as text and
/// would need syntax we have not proven; refuse them.
fn is_plain_target(target: &str) -> bool {
    if target == "_" {
        return true;
    }
    let mut chars = target.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    target != "this"
        && target != "base"
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A source location whose value can be read twice without changing meaning:
/// a local, `this`, a static field, or a field chain rooted in one of those.
/// Calls and indexers are excluded — `(a, b) = Get()` evaluates `Get()` once,
/// while `a = Get().Item1; b = Get().Item2` evaluates it twice.
fn is_stable_source(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) | Expr::This | Expr::StaticField(..) => true,
        Expr::Field(base, _) => is_stable_source(base),
        Expr::Cast(_, inner) => is_stable_source(inner),
        _ => false,
    }
}

/// Does evaluating `expr` read the simple variable `name`?
fn expr_reads_name(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Var(n) => n == name,
        Expr::This => name == "this",
        Expr::Field(base, _) => expr_reads_name(base, name),
        Expr::Cast(_, inner) => expr_reads_name(inner, name),
        // A static-field base is a type name, not a local.
        Expr::StaticField(..) => false,
        _ => false,
    }
}

/// Structural equality check for expressions — uses derived `PartialEq` to
/// compare variant discriminants and sub-expressions directly without rendering.
fn exprs_eq_static(a: &Expr, b: &Expr) -> bool {
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    // -----------------------------------------------------------------------
    // ValueTuple.Create recognition
    // -----------------------------------------------------------------------

    #[test]
    fn value_tuple_create_2_elements() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "ValueTuple.Create".into(),
            args: vec![var("a"), var("b")],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Tuple { elements, names }) => {
                assert_eq!(elements.len(), 2);
                assert!(names.iter().all(|n| n.is_none()));
            }
            other => panic!("expected Eval(Tuple), got {other:?}"),
        }
    }

    #[test]
    fn value_tuple_create_3_elements() {
        let mut stmts = vec![Stmt::Assign(
            "t".into(),
            None,
            Expr::Call {
                recv: None,
                name: "ValueTuple.Create".into(),
                args: vec![var("a"), var("b"), var("c")],
            },
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Tuple { elements, .. }) => {
                assert_eq!(target, "t");
                assert_eq!(elements.len(), 3);
            }
            other => panic!("expected Assign(_, _, Tuple), got {other:?}"),
        }
    }

    #[test]
    fn system_value_tuple_create() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "System.ValueTuple.Create".into(),
            args: vec![var("x"), var("y")],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Tuple { .. })));
    }

    // -----------------------------------------------------------------------
    // new ValueTuple<...>(...) recognition
    // -----------------------------------------------------------------------

    #[test]
    fn new_value_tuple_generic() {
        let mut stmts = vec![Stmt::Assign(
            "t".into(),
            None,
            Expr::NewObj(
                "ValueTuple<int, string>".into(),
                vec![Expr::Int(1), Expr::Str("hello".into())],
            ),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Tuple { elements, .. }) => {
                assert_eq!(elements.len(), 2);
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn new_value_tuple_backtick_notation() {
        let mut stmts = vec![Stmt::Assign(
            "t".into(),
            None,
            Expr::NewObj(
                "ValueTuple`2".into(),
                vec![var("a"), var("b")],
            ),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::Tuple { .. })));
    }

    // -----------------------------------------------------------------------
    // ValueTuple ctor recognition
    // -----------------------------------------------------------------------

    #[test]
    fn value_tuple_ctor() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "ValueTuple`3..ctor".into(),
            args: vec![var("a"), var("b"), var("c")],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Tuple { .. })));
    }

    // -----------------------------------------------------------------------
    // Deconstruction pattern recognition
    // -----------------------------------------------------------------------

    #[test]
    fn deconstruction_2_items() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(var("t")), "Item1".into())),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(var("t")), "Item2".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Var(src)) => {
                assert_eq!(target, "(x, y)");
                assert_eq!(src, "t");
            }
            other => panic!("expected deconstruction Assign, got {other:?}"),
        }
    }

    #[test]
    fn deconstruction_3_items() {
        let mut stmts = vec![
            Stmt::Assign("a".into(), None, Expr::Field(Box::new(var("tuple")), "Item1".into())),
            Stmt::Assign("b".into(), None, Expr::Field(Box::new(var("tuple")), "Item2".into())),
            Stmt::Assign("c".into(), None, Expr::Field(Box::new(var("tuple")), "Item3".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(target, _, _) => {
                assert_eq!(target, "(a, b, c)");
            }
            other => panic!("expected deconstruction, got {other:?}"),
        }
    }

    #[test]
    fn deconstruction_different_sources_not_merged() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(var("t1")), "Item1".into())),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(var("t2")), "Item2".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "different sources should not merge");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn single_item_read_not_deconstruction() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(var("t")), "Item1".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "single Item read is not a deconstruction");
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    #[test]
    fn single_arg_create_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "ValueTuple.Create".into(),
            args: vec![var("a")],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "single-arg ValueTuple.Create should not rewrite");
    }

    #[test]
    fn non_tuple_call_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Console.WriteLine".into(),
            args: vec![var("a"), var("b")],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn instance_create_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("factory"))),
            name: "Create".into(),
            args: vec![var("a"), var("b")],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "instance Create should not rewrite");
    }

    // -----------------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------------

    #[test]
    fn tuple_renders_correctly() {
        let tuple = Expr::Tuple {
            names: vec![None, None],
            elements: vec![var("a"), var("b")],
        };
        let mut s = String::new();
        tuple.render(&mut s);
        assert_eq!(s, "(a, b)");
    }

    #[test]
    fn named_tuple_renders_correctly() {
        let tuple = Expr::Tuple {
            names: vec![Some("x".into()), Some("y".into())],
            elements: vec![Expr::Int(1), Expr::Int(2)],
        };
        let mut s = String::new();
        tuple.render(&mut s);
        assert_eq!(s, "(x: 1, y: 2)");
    }

    #[test]
    fn tuple_in_assignment_renders() {
        let stmt = Stmt::Assign(
            "t".into(),
            None,
            Expr::Tuple {
                names: vec![None, None, None],
                elements: vec![var("a"), var("b"), var("c")],
            },
        );
        // Just verify it doesn't panic and produces reasonable output.
        let mut s = String::new();
        match &stmt {
            Stmt::Assign(_, _, expr) => expr.render(&mut s),
            _ => unreachable!(),
        }
        assert_eq!(s, "(a, b, c)");
    }

    // -----------------------------------------------------------------------
    // Nested tuple in expression
    // -----------------------------------------------------------------------

    #[test]
    fn tuple_inside_nested_call() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("console"))),
            name: "WriteLine".into(),
            args: vec![Expr::Call {
                recv: None,
                name: "ValueTuple.Create".into(),
                args: vec![var("a"), var("b")],
            }],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "nested ValueTuple.Create should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Call { args, .. }) => {
                assert!(matches!(&args[0], Expr::Tuple { .. }));
            }
            other => panic!("expected Call with Tuple arg, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Partial deconstruction followed by other statements
    // -----------------------------------------------------------------------

    #[test]
    fn deconstruction_preserves_trailing_statements() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(var("t")), "Item1".into())),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(var("t")), "Item2".into())),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("console"))),
                name: "WriteLine".into(),
                args: vec![var("x")],
            }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2, "deconstruction + trailing call");
        assert!(matches!(&stmts[0], Stmt::Assign(t, _, _) if t == "(x, y)"));
    }

    // -----------------------------------------------------------------------
    // Arity proofs
    // -----------------------------------------------------------------------

    /// `ValueTuple.Create` has 1..=8 element overloads; C# tuples need 2+.
    #[test]
    fn value_tuple_create_arity_bounds() {
        let mut nine = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "ValueTuple.Create".into(),
            args: (0..9).map(Expr::Int).collect(),
        })];
        assert_eq!(apply(&mut nine), 0, "Create has no 9-element overload");

        let mut one = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "ValueTuple.Create".into(),
            args: vec![var("a")],
        })];
        assert_eq!(apply(&mut one), 0, "one-element tuples do not exist");
    }

    /// `new ValueTuple<T1, T2, T3>(a, b)` is not a valid tuple construction.
    #[test]
    fn new_tuple_arity_mismatch_not_folded() {
        let mut generic = vec![Stmt::Eval(Expr::NewObj(
            "ValueTuple<int, string, bool>".into(),
            vec![var("a"), var("b")],
        ))];
        assert_eq!(apply(&mut generic), 0);

        let mut backtick = vec![Stmt::Eval(Expr::NewObj(
            "ValueTuple`2".into(),
            vec![var("a"), var("b"), var("c")],
        ))];
        assert_eq!(apply(&mut backtick), 0);
    }

    /// Ctor arity must match the declared tuple arity.
    #[test]
    fn ctor_arity_mismatch_not_folded() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "ValueTuple`2..ctor".into(),
            args: vec![var("a"), var("b"), var("c")],
        })];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A non-generic `ValueTuple` has no elements and cannot be folded.
    #[test]
    fn non_generic_value_tuple_not_folded() {
        let mut stmts = vec![Stmt::Eval(Expr::NewObj(
            "ValueTuple".into(),
            vec![var("a"), var("b")],
        ))];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Nested generic arguments do not inflate the arity count.
    #[test]
    fn nested_generic_arity_counts_top_level_only() {
        let mut stmts = vec![Stmt::Eval(Expr::NewObj(
            "ValueTuple<int, Dictionary<int, string>>".into(),
            vec![var("a"), var("b")],
        ))];
        assert_eq!(apply(&mut stmts), 1);
    }

    // -----------------------------------------------------------------------
    // Deconstruction safety
    // -----------------------------------------------------------------------

    /// A call source is evaluated twice in the original two statements; a
    /// deconstruction evaluates it once. Refuse.
    #[test]
    fn deconstruction_effectful_source_not_merged() {
        let get = || {
            Expr::Call {
                recv: Some(Box::new(var("getTuple"))),
                name: "Invoke".into(),
                args: vec![],
            }
        };
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(get()), "Item1".into())),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(get()), "Item2".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(stmts.len(), 2);
    }

    /// `t = t.Item1; y = t.Item2` reads the *new* t for y; `(t, y) = t` reads
    /// the old one for both. Refuse.
    #[test]
    fn deconstruction_alias_target_not_merged() {
        let mut stmts = vec![
            Stmt::Assign("t".into(), None, Expr::Field(Box::new(var("t")), "Item1".into())),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(var("t")), "Item2".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(stmts.len(), 2);
    }

    /// Non-identifier targets would need syntax we have not proven.
    #[test]
    fn deconstruction_non_identifier_target_not_merged() {
        let mut stmts = vec![
            Stmt::Assign("arr[0]".into(), None, Expr::Field(Box::new(var("t")), "Item1".into())),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(var("t")), "Item2".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Duplicate targets are not a valid deconstruction.
    #[test]
    fn deconstruction_duplicate_target_not_merged() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(var("t")), "Item1".into())),
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(var("t")), "Item2".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(stmts.len(), 2);
    }

    /// A typed first assignment is a declaration; deconstruction cannot
    /// redeclare, so leave it alone.
    #[test]
    fn deconstruction_typed_assignment_not_merged() {
        let mut stmts = vec![
            Stmt::Assign(
                "x".into(),
                Some("int".into()),
                Expr::Field(Box::new(var("t")), "Item1".into()),
            ),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(var("t")), "Item2".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A stable field chain is a valid source; the deconstruction keeps it.
    #[test]
    fn deconstruction_field_source_merged() {
        let pair = Expr::Field(Box::new(Expr::This), "pair".into());
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(pair.clone()), "Item1".into())),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(pair.clone()), "Item2".into())),
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::Assign(target, _, src) => {
                assert_eq!(target, "(x, y)");
                assert_eq!(src, &pair);
            }
            other => panic!("expected deconstruction, got {other:?}"),
        }
    }

    /// Idempotence: a folded tuple and deconstruction are stable.
    #[test]
    fn second_run_is_noop() {
        let mut stmts = vec![
            Stmt::Assign(
                "t".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "ValueTuple.Create".into(),
                    args: vec![var("a"), var("b")],
                },
            ),
            Stmt::Assign("x".into(), None, Expr::Field(Box::new(var("p")), "Item1".into())),
            Stmt::Assign("y".into(), None, Expr::Field(Box::new(var("p")), "Item2".into())),
        ];
        let first = apply(&mut stmts);
        assert_eq!(first, 2, "tuple create + deconstruction");
        let snapshot = format!("{:?}", stmts);
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(format!("{:?}", stmts), snapshot);
    }
}
