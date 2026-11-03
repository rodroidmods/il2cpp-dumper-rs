//! Equality/inequality operator reconstruction:
//! `Object.op_Equality(a, b)` -> `a == b`,
//! `Object.op_Inequality(a, b)` -> `a != b`.
//!
//! ## What this pass does
//!
//! IL2CPP renders C# equality comparisons as static method calls:
//!
//! ```text
//! Object.op_Equality(a, b)    ->  a == b
//! Object.op_Inequality(a, b)  ->  a != b
//! ```
//!
//! This is a HIGH-IMPACT pass: 9,127 sites in the test corpus.
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - Static call only (`recv` is `None`)
//! - Method name must be exactly `"Object.op_Equality"` or `"Object.op_Inequality"`
//! - Must have exactly 2 arguments
//! - A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{CmpKind, Expr, Stmt};

/// Run the equality-ops reconstruction over one method body.
///
/// Returns the number of `Object.op_Equality`/`Object.op_Inequality` calls
/// rewritten to `Cmp(Eq, …)`/`Cmp(Ne, …)` expressions.
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
        Stmt::StringSwitch { cases, tail, .. } => {
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

/// Recursively rewrite `Object.op_Equality`/`Object.op_Inequality` calls to
/// `Cmp(Eq, …)`/`Cmp(Ne, …)` within an expression tree.
///
/// Post-order traversal: children are rewritten first, so nested equality calls
/// (`Object.op_Equality(Object.op_Equality(a, b), c)`) are handled naturally —
/// the inner call is rewritten to `(a == b)`, then the outer call sees
/// `Object.op_Equality((a == b), c)` and becomes `((a == b) == c)`.
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
        Expr::Tuple { elements, .. } => {
            for e in elements {
                rewrites += rewrite_expr(e);
            }
        }
        Expr::MethodChain { receiver, methods } => {
            rewrites += rewrite_expr(receiver);
            for m in methods {
                for a in &mut m.args {
                    rewrites += rewrite_expr(a);
                }
            }
        }
        Expr::SwitchExpr { discrim, arms, default_arm } => {
            rewrites += rewrite_expr(discrim);
            for (p, r) in arms {
                rewrites += rewrite_expr(p);
                rewrites += rewrite_expr(r);
            }
            if let Some(d) = default_arm {
                rewrites += rewrite_expr(d);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                rewrites += rewrite_expr(f);
            }
            if let Some(t) = to {
                rewrites += rewrite_expr(t);
            }
        }
        Expr::IndexFromEnd(inner) => {
            rewrites += rewrite_expr(inner);
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
        | Expr::TypeOf(_) => {}
    }

    // Now check if THIS expression is an equality/inequality call we can rewrite.
    if let Expr::Call {
        recv: None,
        name,
        args,
    } = expr
    {
        if args.len() == 2 {
            if name == "Object.op_Equality" {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Cmp(CmpKind::Eq, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name == "Object.op_Inequality" {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Cmp(CmpKind::Ne, Box::new(a), Box::new(b));
                return rewrites + 1;
            }
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

    /// Helper: build `Object.op_Equality(a, b)` (static dotted form).
    fn op_eq(a: Expr, b: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "Object.op_Equality".into(),
            args: vec![a, b],
        }
    }

    /// Helper: build `Object.op_Inequality(a, b)` (static dotted form).
    fn op_ne(a: Expr, b: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "Object.op_Inequality".into(),
            args: vec![a, b],
        }
    }

    // -----------------------------------------------------------------------
    // Positive rewrites
    // -----------------------------------------------------------------------

    /// 1. Simple equality: `Object.op_Equality(a, b)` -> `a == b`
    #[test]
    fn simple_equality() {
        let mut stmts = vec![Stmt::Eval(op_eq(var("a"), var("b")))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Cmp(CmpKind::Eq, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Var(name) if name == "a"));
                assert!(matches!(rhs.as_ref(), Expr::Var(name) if name == "b"));
            }
            other => panic!("expected Eval(Cmp(Eq)), got {other:?}"),
        }
    }

    /// 2. Simple inequality: `Object.op_Inequality(a, b)` -> `a != b`
    #[test]
    fn simple_inequality() {
        let mut stmts = vec![Stmt::Eval(op_ne(var("a"), var("b")))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Cmp(CmpKind::Ne, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Var(name) if name == "a"));
                assert!(matches!(rhs.as_ref(), Expr::Var(name) if name == "b"));
            }
            other => panic!("expected Eval(Cmp(Ne)), got {other:?}"),
        }
    }

    /// 3. Equality with literals: `Object.op_Equality(x, 0)` -> `x == 0`
    #[test]
    fn equality_with_literals() {
        let mut stmts = vec![Stmt::Eval(op_eq(var("x"), Expr::Int(0)))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Cmp(CmpKind::Eq, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Var(name) if name == "x"));
                assert!(matches!(rhs.as_ref(), Expr::Int(0)));
            }
            other => panic!("expected Eval(Cmp(Eq)), got {other:?}"),
        }
    }

    /// 4. Inequality with null: `Object.op_Inequality(obj, null)` -> `obj != null`
    #[test]
    fn inequality_with_null() {
        let mut stmts = vec![Stmt::Eval(op_ne(var("obj"), Expr::Null))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Cmp(CmpKind::Ne, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Var(name) if name == "obj"));
                assert!(matches!(rhs.as_ref(), Expr::Null));
            }
            other => panic!("expected Eval(Cmp(Ne)), got {other:?}"),
        }
    }

    /// 5. Equality in condition: `if (Object.op_Equality(a, b))` -> `if (a == b)`
    #[test]
    fn equality_in_condition() {
        let mut stmts = vec![Stmt::If {
            cond: op_eq(var("a"), var("b")),
            then_body: vec![Stmt::Eval(var("x"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { cond, .. } => {
                assert!(matches!(cond, Expr::Cmp(CmpKind::Eq, _, _)));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// 6. Equality in assignment: `x = Object.op_Equality(a, b)` -> `x = a == b`
    #[test]
    fn equality_in_assignment() {
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            op_eq(var("a"), var("b")),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Cmp(CmpKind::Eq, lhs, rhs)) => {
                assert_eq!(target, "x");
                assert!(matches!(lhs.as_ref(), Expr::Var(name) if name == "a"));
                assert!(matches!(rhs.as_ref(), Expr::Var(name) if name == "b"));
            }
            other => panic!("expected Assign(_, _, Cmp(Eq)), got {other:?}"),
        }
    }

    /// 7. Nested equality: `Object.op_Equality(Object.op_Equality(a, b), true)`
    ///    -> `(a == b) == true`
    #[test]
    fn nested_equality() {
        let inner = op_eq(var("a"), var("b"));
        let outer = op_eq(inner, Expr::Bool(true));
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both inner and outer equality should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Cmp(CmpKind::Eq, lhs, rhs)) => {
                // LHS should be (a == b)
                assert!(matches!(lhs.as_ref(), Expr::Cmp(CmpKind::Eq, _, _)));
                // RHS should be true
                assert!(matches!(rhs.as_ref(), Expr::Bool(true)));
            }
            other => panic!("expected Eval(Cmp(Eq)), got {other:?}"),
        }
    }

    /// 8. Multiple equality ops in one statement:
    ///    `Object.op_Equality(a, b) && Object.op_Inequality(c, d)`
    ///    -> `(a == b) && (c != d)`
    #[test]
    fn multiple_equality_ops_in_one_statement() {
        let eq = op_eq(var("a"), var("b"));
        let ne = op_ne(var("c"), var("d"));
        let combined = Expr::Bin(
            super::super::super::csharp::BinOp::AndAlso,
            Box::new(eq),
            Box::new(ne),
        );
        let mut stmts = vec![Stmt::Eval(combined)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both equality and inequality should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(super::super::super::csharp::BinOp::AndAlso, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Cmp(CmpKind::Eq, _, _)));
                assert!(matches!(rhs.as_ref(), Expr::Cmp(CmpKind::Ne, _, _)));
            }
            other => panic!("expected Eval(Bin(AndAlso)), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    /// 9. Instance method call (receiver is Some) — must NOT rewrite.
    #[test]
    fn instance_call_not_rewritten() {
        let expr = Expr::Call {
            recv: Some(Box::new(var("obj"))),
            name: "op_Equality".into(),
            args: vec![var("a"), var("b")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "instance method call must not rewrite");
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "op_Equality"
        ));
    }

    /// 10. Wrong argument count (1 arg) — must NOT rewrite.
    #[test]
    fn wrong_arg_count_not_rewritten() {
        let expr = Expr::Call {
            recv: None,
            name: "Object.op_Equality".into(),
            args: vec![var("a")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "single-arg equality must not rewrite");
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "Object.op_Equality"
        ));
    }

    /// 11. Wrong argument count (3 args) — must NOT rewrite.
    #[test]
    fn three_args_not_rewritten() {
        let expr = Expr::Call {
            recv: None,
            name: "Object.op_Equality".into(),
            args: vec![var("a"), var("b"), var("c")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "three-arg equality must not rewrite");
    }

    /// 12. Different method name — must NOT rewrite.
    #[test]
    fn different_method_name_not_rewritten() {
        let expr = Expr::Call {
            recv: None,
            name: "Object.Equals".into(),
            args: vec![var("a"), var("b")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "Object.Equals must not rewrite");
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "Object.Equals"
        ));
    }
}
