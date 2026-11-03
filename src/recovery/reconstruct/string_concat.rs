//! String concatenation reconstruction: `String.Concat(a, b, …)` → `a + b + …`.
//!
//! ## What this pass does
//!
//! IL2CPP renders C# string concatenation as static method calls:
//!
//! ```text
//! String.Concat(a, b)        →  a + b
//! String.Concat(a, b, c)     →  a + b + c
//! String.Concat(a, b, c, d)  →  a + b + c + d
//! ```
//!
//! Two call shapes are recognized:
//!
//! 1. **Dotted static form**: `recv: None, name: "String.Concat"` — the
//!    renderer's default for a static call whose qualifier is folded into the
//!    method name.
//! 2. **Receiver form**: `recv: Some(Var("String")), name: "Concat"` — an
//!    instance-style call on the `String` type, which is still a static
//!    dispatch at the source level.
//!
//! The rewrite produces a left-associative chain of `Bin(Add, …)` expressions,
//! matching how a C# compiler would have emitted the original `+` chain.
//!
//! ## Safety
//!
//! Folding `String.Concat` to `+` is only valid when the C# operator resolves
//! to the built-in string concatenation:
//! - Method name must be exactly `"String.Concat"` (static form) or `"Concat"`
//!   with a `String` receiver — `List.Concat` and friends are different methods.
//! - At least 2 arguments are required (a 1-arg Concat is not a concatenation).
//! - A `null` literal anywhere refuses the fold: `String.Concat(a, null)` is a
//!   string call that converts null to "", while `a + null` is only legal when
//!   the compile-time type of `a` is string (and changes overload resolution).
//! - Every operand must be provably string-valued or a scalar literal, and at
//!   least one operand must be string-valued. Otherwise `Concat(1, 2)` would
//!   fold to the arithmetic `1 + 2`. An unknown local (`v42`) is not proof: the
//!   static call may be `Concat(object, object)`, which `+` cannot express.
//! - A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{BinOp, Expr, Stmt};

/// Instance/static methods whose result is unconditionally a `string`.
///
/// Kept deliberately short: names like `Contains` (bool), `Split` (array) and
/// `Remove` (bool on collections) are *not* string-valued, and `Replace` is
/// also a `StringBuilder` member (returns `StringBuilder`), so it is excluded.
const STRING_RETURNING_METHODS: &[&str] = &[
    "ToString",
    "Trim",
    "TrimStart",
    "TrimEnd",
    "Substring",
    "ToUpper",
    "ToLower",
    "ToUpperInvariant",
    "ToLowerInvariant",
    "PadLeft",
    "PadRight",
    "Normalize",
];

/// Static `String` members that produce a `string`.
const STRING_RETURNING_STATICS: &[&str] = &[
    "String.Concat",
    "String.Format",
    "String.Join",
    "String.Copy",
    "String.Intern",
    "String.IsInterned",
];

/// Run the string-concat reconstruction over one method body.
///
/// Returns the number of `String.Concat` calls rewritten to `Bin(Add, …)` chains.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut rewrites = 0;
    for stmt in stmts.iter_mut() {
        rewrites += rewrite_stmt(stmt);
    }
    rewrites
}

/// Rewrite a single statement, recursing into nested bodies.
fn rewrite_stmt(stmt: &mut Stmt) -> usize {
    let mut rewrites = 0;
    match stmt {
        Stmt::Declare(..) => {}
        Stmt::Eval(expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::Assign(_, _, expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::Return(Some(expr)) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::Throw(expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::YieldReturn(expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::YieldBreak => {}
        Stmt::EventAssign { field, value, .. } => {
            rewrites += rewrite_expr(field);
            rewrites += rewrite_expr(value);
        }
        Stmt::IndexAssign { collection, index, value } => {
            rewrites += rewrite_expr(collection);
            rewrites += rewrite_expr(index);
            rewrites += rewrite_expr(value);
        }
        Stmt::GotoCase(_) => {}
        Stmt::StringSwitch { disc, cases, tail } => {
            rewrites += rewrite_expr(disc);
            for (_, b) in cases {
                rewrites += apply(b);
            }
            if let Some(t) = tail {
                rewrites += apply(t);
            }
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            rewrites += rewrite_expr(cond);
            rewrites += apply(then_body);
            rewrites += apply(else_body);
        }
        Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
            rewrites += rewrite_expr(cond);
            rewrites += apply(body);
        }
        Stmt::For {
            init,
            cond,
            incr,
            body,
        } => {
            for s in init {
                rewrites += rewrite_stmt(s);
            }
            if let Some(c) = cond {
                rewrites += rewrite_expr(c);
            }
            for s in incr {
                rewrites += rewrite_stmt(s);
            }
            rewrites += apply(body);
        }
        Stmt::ForEach { body, .. } => {
            rewrites += apply(body);
        }
        Stmt::Using { decl, body } => {
            rewrites += rewrite_stmt(decl);
            rewrites += apply(body);
        }
        Stmt::Lock { expr, body } => {
            rewrites += rewrite_expr(expr);
            rewrites += apply(body);
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            rewrites += apply(try_body);
            for (_, b) in catches {
                rewrites += apply(b);
            }
            if let Some(fb) = finally_body {
                rewrites += apply(fb);
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases {
                rewrites += apply(b);
            }
            rewrites += apply(tail);
            if let Some(arm) = default {
                rewrites += apply(arm);
            }
        }
        // Statements with no expressions to rewrite.
        Stmt::Return(None)
        | Stmt::Goto(_)
        | Stmt::Label(_)
        | Stmt::Comment(_)
        | Stmt::Attribute { .. }
        | Stmt::Break
        | Stmt::Continue => {}
    }
    rewrites
}

/// Returns `true` when `expr` is a `String.Concat` call that should be
/// rewritten to a `+` chain.
fn is_string_concat(expr: &Expr) -> bool {
    match expr {
        // Static dotted form: String.Concat(a, b, …)
        Expr::Call {
            recv: None,
            name,
            args,
        } => name == "String.Concat" && concat_operands_supported(args),
        // Receiver form: String.Concat(a, b, …) with Var("String") receiver
        Expr::Call {
            recv: Some(recv),
            name,
            args,
        } => {
            name == "Concat"
                && matches!(recv.as_ref(), Expr::Var(t) if t == "String")
                && concat_operands_supported(args)
        }
        _ => false,
    }
}

/// Is `ty` the C# `string` type spelling used by the resolver?
fn is_string_type_name(ty: &str) -> bool {
    matches!(ty, "string" | "String" | "System.String")
}

/// Is `expr` provably a `string` value?
///
/// Only shapes the AST proves without a type oracle count. An unknown local or
/// an arbitrary method result is not evidence: the IL2CPP call may have gone
/// through the `Concat(object, object)` overload.
fn is_definitely_string(expr: &Expr) -> bool {
    match expr {
        Expr::Str(_) | Expr::StringInterp { .. } => true,
        Expr::Cast(ty, _) => is_string_type_name(ty),
        // A concatenation is a string when either side is a string (the C#
        // built-in `+` with a string operand yields a string).
        Expr::Bin(BinOp::Add, a, b) => is_definitely_string(a) || is_definitely_string(b),
        Expr::Cond(_, a, b) => is_definitely_string(a) && is_definitely_string(b),
        Expr::NullConditional { access, .. } => is_definitely_string(access),
        Expr::Call { recv, name, .. } => {
            if STRING_RETURNING_STATICS.contains(&name.as_str()) {
                return true;
            }
            let bare = name.rsplit('.').next().unwrap_or(name);
            if STRING_RETURNING_METHODS.contains(&bare) {
                return true;
            }
            // Receiver form of the static `String` members: `String.Concat(…)`.
            matches!(recv.as_deref(), Some(Expr::Var(q)) if q == "String")
                && matches!(bare, "Concat" | "Format" | "Join" | "Copy" | "Intern" | "IsInterned")
        }
        _ => false,
    }
}

/// Scalar literals that C# accepts on either side of a string `+`.
fn is_scalar_literal(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Str(_) | Expr::Int(_) | Expr::Float(_) | Expr::Double(_) | Expr::Bool(_)
    )
}

/// Are all operands acceptable for a `+` fold?
///
/// Requires at least two operands, no top-level null operand, every operand
/// string-valued or a scalar literal, and at least one operand string-valued
/// (so numeric literals alone cannot turn into arithmetic `+`). A null nested
/// inside a string-valued operand (an interpolation hole, `ToString(null)`)
/// does not affect the outer overload and is allowed.
fn concat_operands_supported(args: &[Expr]) -> bool {
    args.len() >= 2
        && !args.iter().any(|a| matches!(a, Expr::Null))
        && args
            .iter()
            .all(|a| is_definitely_string(a) || is_scalar_literal(a))
        && args.iter().any(is_definitely_string)
}

/// Recursively rewrite `String.Concat` calls to `Bin(Add, …)` chains within an
/// expression tree.
///
/// Post-order traversal: children are rewritten first, so nested concats
/// (`String.Concat(String.Concat(a, b), c)`) are handled naturally — the inner
/// call is rewritten to `(a + b)`, then the outer call sees
/// `String.Concat((a + b), c)` and becomes `((a + b) + c)`.
fn rewrite_expr(expr: &mut Expr) -> usize {
    let mut rewrites = 0;

    // First, recurse into subexpressions (post-order).
    match expr {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            rewrites += rewrite_expr(a);
            rewrites += rewrite_expr(b);
        }
        Expr::Not(e) | Expr::Neg(e) | Expr::Cast(_, e) | Expr::AddressOf(e) | Expr::Await(e) => {
            rewrites += rewrite_expr(e);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            rewrites += rewrite_expr(r);
        }
        Expr::Cond(c, a, b) => {
            rewrites += rewrite_expr(c);
            rewrites += rewrite_expr(a);
            rewrites += rewrite_expr(b);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                rewrites += rewrite_expr(r);
            }
            for a in args {
                rewrites += rewrite_expr(a);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                rewrites += rewrite_expr(a);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for e in items {
                rewrites += rewrite_expr(e);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for e in ctor_args {
                rewrites += rewrite_expr(e);
            }
            for e in items {
                rewrites += rewrite_expr(e);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for e in ctor_args {
                rewrites += rewrite_expr(e);
            }
            for (_, v) in fields {
                rewrites += rewrite_expr(v);
            }
        }
        Expr::Index { collection, index } => {
            rewrites += rewrite_expr(collection);
            rewrites += rewrite_expr(index);
        }
        Expr::Lambda { body, .. } => {
            rewrites += rewrite_expr(body);
        }
        Expr::NullConditional { receiver, access } => {
            rewrites += rewrite_expr(receiver);
            rewrites += rewrite_expr(access);
        }
        // Leaf expressions: nothing to recurse into.
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
        | Expr::StringInterp { .. }
        | Expr::Tuple { .. }
        | Expr::MethodChain { .. } | Expr::SwitchExpr { .. }
        | Expr::Range { .. } | Expr::IndexFromEnd(_) | Expr::TypeOf(_) => {}
    }

    // Now check if THIS expression is a String.Concat call we can rewrite.
    if is_string_concat(expr) {
        // Extract the arguments. We know from is_string_concat that this is a
        // Call with >= 2 args.
        if let Expr::Call { args, .. } = expr {
            let mut args = std::mem::take(args);
            // Build left-associative chain: ((a + b) + c) + d
            let mut result = args.remove(0);
            for arg in args {
                result = Expr::Bin(BinOp::Add, Box::new(result), Box::new(arg));
            }
            *expr = result;
            rewrites += 1;
        }
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    /// A string literal (the only operand shape the AST proves is a string).
    fn s(text: &str) -> Expr {
        Expr::Str(text.to_string())
    }

    /// A provably string-valued expression: `name.ToString()`.
    fn to_string(name: &str) -> Expr {
        Expr::Call {
            recv: Some(Box::new(var(name))),
            name: "ToString".into(),
            args: vec![],
        }
    }

    /// Helper: build `String.Concat(args...)` (static dotted form).
    fn string_concat(args: Vec<Expr>) -> Expr {
        Expr::Call {
            recv: None,
            name: "String.Concat".into(),
            args,
        }
    }

    /// Helper: build `String.Concat(args...)` (receiver form).
    fn string_concat_recv(args: Vec<Expr>) -> Expr {
        Expr::Call {
            recv: Some(Box::new(var("String"))),
            name: "Concat".into(),
            args,
        }
    }

    // -----------------------------------------------------------------------
    // Positive rewrites
    // -----------------------------------------------------------------------

    /// 1. Two-arg concat: `String.Concat("a", "b")` → `"a" + "b"`
    #[test]
    fn two_arg_concat() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![s("a"), s("b")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(BinOp::Add, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Str(v) if v == "a"));
                assert!(matches!(rhs.as_ref(), Expr::Str(v) if v == "b"));
            }
            other => panic!("expected Eval(Bin(Add)), got {other:?}"),
        }
    }

    /// 2. Three-arg concat: `String.Concat("a", "b", "c")` → `("a" + "b") + "c"`
    #[test]
    fn three_arg_concat() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![s("a"), s("b"), s("c")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(BinOp::Add, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Bin(BinOp::Add, _, _)));
                assert!(matches!(rhs.as_ref(), Expr::Str(v) if v == "c"));
            }
            other => panic!("expected Eval(Bin(Add)), got {other:?}"),
        }
    }

    /// 3. Four-arg concat: `String.Concat("a", "b", "c", "d")` →
    ///    `(("a" + "b") + "c") + "d"`
    #[test]
    fn four_arg_concat() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![
            s("a"),
            s("b"),
            s("c"),
            s("d"),
        ]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(BinOp::Add, lhs, rhs)) => {
                assert!(matches!(rhs.as_ref(), Expr::Str(v) if v == "d"));
                match lhs.as_ref() {
                    Expr::Bin(BinOp::Add, inner_lhs, inner_rhs) => {
                        assert!(matches!(inner_rhs.as_ref(), Expr::Str(v) if v == "c"));
                        assert!(matches!(inner_lhs.as_ref(), Expr::Bin(BinOp::Add, _, _)));
                    }
                    other => panic!("expected nested Bin(Add), got {other:?}"),
                }
            }
            other => panic!("expected Eval(Bin(Add)), got {other:?}"),
        }
    }

    /// 4. Concat with literals keeps the source order.
    #[test]
    fn concat_with_literals() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![
            s("hello"),
            s(" "),
            s("world"),
        ]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(BinOp::Add, lhs, rhs)) => {
                assert!(matches!(rhs.as_ref(), Expr::Str(v) if v == "world"));
                match lhs.as_ref() {
                    Expr::Bin(BinOp::Add, inner_lhs, inner_rhs) => {
                        assert!(matches!(inner_lhs.as_ref(), Expr::Str(v) if v == "hello"));
                        assert!(matches!(inner_rhs.as_ref(), Expr::Str(v) if v == " "));
                    }
                    other => panic!("expected nested Bin(Add), got {other:?}"),
                }
            }
            other => panic!("expected Eval(Bin(Add)), got {other:?}"),
        }
    }

    /// 5. Concat in assignment: `s = String.Concat("a", "b")` → `s = "a" + "b"`.
    #[test]
    fn concat_in_assignment() {
        let mut stmts = vec![Stmt::Assign(
            "s".into(),
            None,
            string_concat(vec![s("a"), s("b")]),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Bin(BinOp::Add, lhs, rhs)) => {
                assert_eq!(target, "s");
                assert!(matches!(lhs.as_ref(), Expr::Str(v) if v == "a"));
                assert!(matches!(rhs.as_ref(), Expr::Str(v) if v == "b"));
            }
            other => panic!("expected Assign(_, _, Bin(Add)), got {other:?}"),
        }
    }

    /// 6. Concat in method call: `foo(String.Concat("a", "b"))` → `foo("a" + "b")`.
    #[test]
    fn concat_in_method_call() {
        let inner = string_concat(vec![s("a"), s("b")]);
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "foo".into(),
            args: vec![inner],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Call { name, args, .. }) => {
                assert_eq!(name, "foo");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Bin(BinOp::Add, _, _)));
            }
            other => panic!("expected Eval(Call), got {other:?}"),
        }
    }

    /// 7. Nested concat: `String.Concat(String.Concat("a", "b"), "c")`.
    #[test]
    fn nested_concat() {
        let inner = string_concat(vec![s("a"), s("b")]);
        let outer = string_concat(vec![inner, s("c")]);
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both inner and outer concat should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(BinOp::Add, lhs, rhs)) => {
                assert!(matches!(rhs.as_ref(), Expr::Str(v) if v == "c"));
                match lhs.as_ref() {
                    Expr::Bin(BinOp::Add, inner_lhs, inner_rhs) => {
                        assert!(matches!(inner_lhs.as_ref(), Expr::Str(v) if v == "a"));
                        assert!(matches!(inner_rhs.as_ref(), Expr::Str(v) if v == "b"));
                    }
                    other => panic!("expected nested Bin(Add), got {other:?}"),
                }
            }
            other => panic!("expected Eval(Bin(Add)), got {other:?}"),
        }
    }

    /// 7b. Receiver form: `String.Concat("a", "b")` with Var("String") receiver.
    #[test]
    fn receiver_form_concat() {
        let mut stmts = vec![Stmt::Eval(string_concat_recv(vec![s("a"), s("b")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(BinOp::Add, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Str(v) if v == "a"));
                assert!(matches!(rhs.as_ref(), Expr::Str(v) if v == "b"));
            }
            other => panic!("expected Eval(Bin(Add)), got {other:?}"),
        }
    }

    /// 8. A provably-string call operand (`x.ToString()`) may fold with a literal.
    #[test]
    fn string_returning_operand_folds() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![
            to_string("x"),
            s("!"),
        ]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(BinOp::Add, lhs, rhs)) => {
                assert!(matches!(
                    lhs.as_ref(),
                    Expr::Call { name, .. } if name == "ToString"
                ));
                assert!(matches!(rhs.as_ref(), Expr::Str(v) if v == "!"));
            }
            other => panic!("expected Eval(Bin(Add)), got {other:?}"),
        }
    }

    /// 9. A `(string)` cast is proof of string type.
    #[test]
    fn string_cast_operand_folds() {
        let cast = Expr::Cast("string".into(), Box::new(var("x")));
        let mut stmts = vec![Stmt::Eval(string_concat(vec![cast, s("!")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Bin(BinOp::Add, _, _))));
    }

    /// 10. `Concat("x", 5)` is string concatenation: one string operand selects
    ///     the built-in `+`, and an int literal converts to text.
    #[test]
    fn string_plus_numeric_literal_folds() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![s("x"), Expr::Int(5)]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(BinOp::Add, _, rhs)) => {
                assert!(matches!(rhs.as_ref(), Expr::Int(5)));
            }
            other => panic!("expected Eval(Bin(Add)), got {other:?}"),
        }
    }

    /// 11. Unknown locals are not proof of string type.
    #[test]
    fn unknown_operands_not_folded() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![var("a"), var("b")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "untyped locals may use Concat(object, object)");
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "String.Concat"
        ));
    }

    /// 12. A single string literal is not enough to type the other operand:
    ///     `Concat(object, object)` on an unknown local cannot become `+`.
    #[test]
    fn unknown_plus_literal_not_folded() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![var("a"), s("!")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { .. })));
    }

    /// 13. A null literal changes `Concat` semantics — never fold.
    #[test]
    fn null_operand_not_folded() {
        for args in [
            vec![s("a"), Expr::Null],
            vec![Expr::Null, s("b")],
            vec![Expr::Null, Expr::Null],
            vec![to_string("x"), Expr::Null],
        ] {
            let mut stmts = vec![Stmt::Eval(string_concat(args))];
            let n = apply(&mut stmts);
            assert_eq!(n, 0, "null literal must refuse the fold");
            assert!(matches!(
                &stmts[0],
                Stmt::Eval(Expr::Call { name, .. }) if name == "String.Concat"
            ));
        }
    }

    /// 14. Numeric literals alone are arithmetic `+`, not concatenation.
    #[test]
    fn numeric_literals_not_folded() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![Expr::Int(1), Expr::Int(2)]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "Concat(1, 2) is not 1 + 2");
    }

    /// 15. A second application is a no-op (idempotent).
    #[test]
    fn second_run_is_noop() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![s("a"), s("b")]))];
        assert_eq!(apply(&mut stmts), 1);
        let snapshot = format!("{:?}", stmts);
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(format!("{:?}", stmts), snapshot);
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    /// 16. Single argument — must NOT rewrite.
    #[test]
    fn single_arg_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![s("a")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "single-arg Concat must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "String.Concat"));
    }

    /// 17. Instance method call on non-String receiver — must NOT rewrite.
    #[test]
    fn non_string_receiver_not_rewritten() {
        let expr = Expr::Call {
            recv: Some(Box::new(var("MyType"))),
            name: "Concat".into(),
            args: vec![s("a"), s("b")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "non-String receiver must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "Concat"));
    }

    /// 18. Different method name — must NOT rewrite.
    #[test]
    fn different_method_name_not_rewritten() {
        let expr = Expr::Call {
            recv: None,
            name: "String.Join".into(),
            args: vec![s("a"), s("b")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "String.Join must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "String.Join"));
    }

    /// 19. Zero arguments — must NOT rewrite.
    #[test]
    fn zero_args_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(string_concat(vec![]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "zero-arg Concat must not rewrite");
    }

    /// 20. Concat inside a while loop body is rewritten.
    #[test]
    fn concat_inside_while_body() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Eval(string_concat(vec![s("a"), s("b")]))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[0], Stmt::Eval(Expr::Bin(BinOp::Add, _, _))));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }
}
