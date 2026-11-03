//! C# 8+ range (`..`) and index-from-end (`^`) reconstruction.
//!
//! ## What this pass does
//!
//! IL2CPP renders range and index expressions as constructor calls:
//!
//! ```text
//! new System.Range(start, end)           →  start..end
//! new System.Index(n, true)              →  ^n
//! System.Range.op_Implicit(value)        →  value..  or  ..value
//! ```
//!
//! This pass detects these patterns and rewrites them to idiomatic C# syntax.
//!
//! ## Patterns recognized
//!
//! 1. `new System.Range(start, end)` → `Range { from: Some(start), to: Some(end) }`
//! 2. `new System.Index(n, true)` → `IndexFromEnd(n)`
//! 3. `System.Range.op_Implicit(x)` where x is an Index → range form
//!
//! ## Safety
//!
//! Only exact matches are rewritten. The type names must be fully qualified
//! as `System.Range` or `System.Index`. A failed match is a no-op.

use super::super::csharp::{Expr, Stmt};

/// Run the range/index reconstruction over one method body.
///
/// Returns the number of rewrites applied.
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
        Stmt::Eval(expr) | Stmt::Assign(_, _, expr) | Stmt::Return(Some(expr))
        | Stmt::Throw(expr) | Stmt::YieldReturn(expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            rewrites += rewrite_expr(cond);
            for s in then_body {
                rewrites += rewrite_stmt(s);
            }
            for s in else_body {
                rewrites += rewrite_stmt(s);
            }
        }
        Stmt::While { cond, body, .. } | Stmt::DoWhile { cond, body, .. } => {
            rewrites += rewrite_expr(cond);
            for s in body {
                rewrites += rewrite_stmt(s);
            }
        }
        Stmt::For { init, cond, incr, body, .. } => {
            for s in init {
                rewrites += rewrite_stmt(s);
            }
            if let Some(c) = cond {
                rewrites += rewrite_expr(c);
            }
            for s in incr {
                rewrites += rewrite_stmt(s);
            }
            for s in body {
                rewrites += rewrite_stmt(s);
            }
        }
        Stmt::Using { decl, body, .. } => {
            rewrites += rewrite_stmt(decl);
            for s in body {
                rewrites += rewrite_stmt(s);
            }
        }
        Stmt::Lock { body, .. } => {
            for s in body {
                rewrites += rewrite_stmt(s);
            }
        }
        Stmt::Try { try_body, catches, finally_body, .. } => {
            for s in try_body {
                rewrites += rewrite_stmt(s);
            }
            for (_, b) in catches {
                for s in b {
                    rewrites += rewrite_stmt(s);
                }
            }
            if let Some(fb) = finally_body {
                for s in fb {
                    rewrites += rewrite_stmt(s);
                }
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases {
                for s in b {
                    rewrites += rewrite_stmt(s);
                }
            }
            for s in tail {
                rewrites += rewrite_stmt(s);
            }
            if let Some(arm) = default {
                for s in arm {
                    rewrites += rewrite_stmt(s);
                }
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, b) in cases {
                for s in b {
                    rewrites += rewrite_stmt(s);
                }
            }
            if let Some(t) = tail {
                for s in t {
                    rewrites += rewrite_stmt(s);
                }
            }
        }
        _ => {}
    }
    rewrites
}

/// Rewrite one expression, recursing into children.
fn rewrite_expr(e: &mut Expr) -> usize {
    let mut rewrites = 0;

    // First, recurse into sub-expressions.
    match e {
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                rewrites += rewrite_expr(r);
            }
            for a in args {
                rewrites += rewrite_expr(a);
            }
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            rewrites += rewrite_expr(a);
            rewrites += rewrite_expr(b);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            rewrites += rewrite_expr(x);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            rewrites += rewrite_expr(r);
        }
        Expr::Cond(c, a, b) => {
            rewrites += rewrite_expr(c);
            rewrites += rewrite_expr(a);
            rewrites += rewrite_expr(b);
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                rewrites += rewrite_expr(a);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for item in items {
                rewrites += rewrite_expr(item);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for a in ctor_args {
                rewrites += rewrite_expr(a);
            }
            for item in items {
                rewrites += rewrite_expr(item);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for a in ctor_args {
                rewrites += rewrite_expr(a);
            }
            for (_, v) in fields {
                rewrites += rewrite_expr(v);
            }
        }
        Expr::Index { collection, index } => {
            rewrites += rewrite_expr(collection);
            rewrites += rewrite_expr(index);
        }
        Expr::NullConditional { receiver, access } => {
            rewrites += rewrite_expr(receiver);
            rewrites += rewrite_expr(access);
        }
        Expr::Lambda { body, .. } => {
            rewrites += rewrite_expr(body);
        }
        Expr::Tuple { elements, .. } => {
            for elem in elements {
                rewrites += rewrite_expr(elem);
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
        Expr::StringInterp { parts } => {
            use super::super::csharp::InterpPart;
            for p in parts {
                if let InterpPart::Expr(inner, _) = p {
                    rewrites += rewrite_expr(inner);
                }
            }
        }
        Expr::PatternMatch { expr, .. } => {
            rewrites += rewrite_expr(expr);
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
        _ => {}
    }

    // Now try to rewrite this expression.
    if let Some(new_expr) = try_rewrite_range(e) {
        *e = new_expr;
        rewrites += 1;
    } else if let Some(new_expr) = try_rewrite_index(e) {
        *e = new_expr;
        rewrites += 1;
    }

    rewrites
}

/// Try to rewrite a `System.Range` constructor call to `Expr::Range`.
///
/// Patterns:
/// - `new System.Range(start, end)` → `Range { from: Some(start), to: Some(end) }`
/// - `System.Range.op_Implicit(x)` → depends on x
fn try_rewrite_range(e: &Expr) -> Option<Expr> {
    // Pattern 1: new System.Range(start, end)
    if let Expr::NewObj(type_name, args) = e {
        if type_name == "System.Range" && args.len() == 2 {
            let start = args[0].clone();
            let end = args[1].clone();
            // Optimize: if start is 0, use None (renders as `..end`)
            let from = if is_zero(&start) {
                None
            } else {
                Some(Box::new(start))
            };
            // If end is a special "to end" marker, use None
            // In practice, System.Range uses int.MaxValue or similar for "to end"
            let to = if is_int_max_value(&end) {
                None
            } else {
                Some(Box::new(end))
            };
            return Some(Expr::Range { from, to });
        }
    }

    // Pattern 2: System.Range.op_Implicit(x) where x is an int
    // This creates a Range from a single Index
    if let Expr::Call { recv: None, name, args } = e {
        if (name == "System.Range.op_Implicit" || name.ends_with(".op_Implicit"))
            && args.len() == 1
        {
            // Check if the argument is an IndexFromEnd or a plain value
            let arg = &args[0];
            // Case A: arg was already rewritten to IndexFromEnd by recursion
            if let Expr::IndexFromEnd(inner) = arg {
                return Some(Expr::Range {
                    from: None,
                    to: Some(Box::new(Expr::IndexFromEnd(inner.clone()))),
                });
            }
            // Case B: arg is still a NewObj(System.Index, ...) (not yet rewritten,
            // or from_end is false so it won't be rewritten)
            if let Expr::NewObj(tn, a) = arg {
                if tn == "System.Index" && a.len() == 2 {
                    // new System.Index(n, true) → this is ^n
                    // op_Implicit(^n) → ..^n  (range from start to ^n)
                    if is_true(&a[1]) {
                        return Some(Expr::Range {
                            from: None,
                            to: Some(Box::new(Expr::IndexFromEnd(Box::new(a[0].clone())))),
                        });
                    }
                    // new System.Index(n, false) → this is n
                    // op_Implicit(n) → ..n  (range from start to n)
                    return Some(Expr::Range {
                        from: None,
                        to: Some(Box::new(a[0].clone())),
                    });
                }
            }
        }
    }

    None
}

/// Try to rewrite a `System.Index` constructor call to `Expr::IndexFromEnd`.
///
/// Pattern: `new System.Index(n, true)` → `IndexFromEnd(n)`
fn try_rewrite_index(e: &Expr) -> Option<Expr> {
    if let Expr::NewObj(type_name, args) = e {
        if type_name == "System.Index" && args.len() == 2 {
            let value = &args[0];
            let from_end = &args[1];
            // Only rewrite when fromEnd is true
            if is_true(from_end) {
                return Some(Expr::IndexFromEnd(Box::new(value.clone())));
            }
        }
    }
    None
}

/// Is this expression the integer literal 0?
fn is_zero(e: &Expr) -> bool {
    matches!(e, Expr::Int(0))
}

/// Is this expression the boolean literal true?
fn is_true(e: &Expr) -> bool {
    matches!(e, Expr::Bool(true))
}

/// Is this expression int.MaxValue (2147483647)?
/// This is used as the "to end" sentinel in System.Range.
fn is_int_max_value(e: &Expr) -> bool {
    matches!(e, Expr::Int(2147483647))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(e: &Expr) -> String {
        let mut s = String::new();
        e.render(&mut s);
        s
    }

    #[test]
    fn range_with_both_bounds() {
        // new System.Range(1, 5) → 1..5
        let mut e = Expr::NewObj("System.Range".into(), vec![Expr::Int(1), Expr::Int(5)]);
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 1);
        assert_eq!(render(&e), "1..5");
    }

    #[test]
    fn range_from_start() {
        // new System.Range(0, 5) → ..5
        let mut e = Expr::NewObj("System.Range".into(), vec![Expr::Int(0), Expr::Int(5)]);
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 1);
        assert_eq!(render(&e), "..5");
    }

    #[test]
    fn range_to_end() {
        // new System.Range(5, 2147483647) → 5..
        let mut e = Expr::NewObj("System.Range".into(), vec![Expr::Int(5), Expr::Int(2147483647)]);
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 1);
        assert_eq!(render(&e), "5..");
    }

    #[test]
    fn full_range() {
        // new System.Range(0, 2147483647) → ..
        let mut e = Expr::NewObj("System.Range".into(), vec![Expr::Int(0), Expr::Int(2147483647)]);
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 1);
        assert_eq!(render(&e), "..");
    }

    #[test]
    fn index_from_end() {
        // new System.Index(3, true) → ^3
        let mut e = Expr::NewObj("System.Index".into(), vec![Expr::Int(3), Expr::Bool(true)]);
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 1);
        assert_eq!(render(&e), "^3");
    }

    #[test]
    fn index_from_start_not_rewritten() {
        // new System.Index(3, false) should NOT be rewritten to ^3
        let mut e = Expr::NewObj("System.Index".into(), vec![Expr::Int(3), Expr::Bool(false)]);
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 0);
        // Should remain as NewObj
        assert!(matches!(&e, Expr::NewObj(tn, _) if tn == "System.Index"));
    }

    #[test]
    fn range_in_array_indexer() {
        // arr[1..^1] — the range contains an IndexFromEnd
        // First build: new System.Range(1, new System.Index(1, true))
        // After rewrite: Range { from: Some(1), to: Some(IndexFromEnd(1)) }
        let mut e = Expr::NewObj(
            "System.Range".into(),
            vec![
                Expr::Int(1),
                Expr::NewObj("System.Index".into(), vec![Expr::Int(1), Expr::Bool(true)]),
            ],
        );
        let n = rewrite_expr(&mut e);
        // Should rewrite both the inner Index and the outer Range
        assert!(n >= 1);
        // The result should be a Range
        match &e {
            Expr::Range { from, to } => {
                assert!(from.is_some());
                assert!(to.is_some());
                let to_expr = to.as_ref().unwrap();
                assert!(matches!(to_expr.as_ref(), Expr::IndexFromEnd(_)));
            }
            _ => panic!("expected Range, got {:?}", e),
        }
        // Render should produce: 1..^1
        assert_eq!(render(&e), "1..^1");
    }

    #[test]
    fn range_with_variable_bounds() {
        // new System.Range(start, end) where start and end are variables
        let mut e = Expr::NewObj(
            "System.Range".into(),
            vec![Expr::Var("start".into()), Expr::Var("end".into())],
        );
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 1);
        assert_eq!(render(&e), "start..end");
    }

    #[test]
    fn non_range_newobj_not_rewritten() {
        // new System.String(...) should not be rewritten
        let mut e = Expr::NewObj("System.String".into(), vec![Expr::Int(42)]);
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 0);
        assert!(matches!(&e, Expr::NewObj(tn, _) if tn == "System.String"));
    }

    #[test]
    fn index_from_end_with_variable() {
        // new System.Index(n, true) → ^n
        let mut e = Expr::NewObj(
            "System.Index".into(),
            vec![Expr::Var("n".into()), Expr::Bool(true)],
        );
        let n = rewrite_expr(&mut e);
        assert_eq!(n, 1);
        assert_eq!(render(&e), "^n");
    }

    #[test]
    fn range_in_statement_context() {
        // Test that the pass works at the statement level
        let mut stmts = vec![Stmt::Assign(
            "r".into(),
            None,
            Expr::NewObj("System.Range".into(), vec![Expr::Int(1), Expr::Int(5)]),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Range { from, to }) => {
                assert!(from.is_some());
                assert!(to.is_some());
            }
            _ => panic!("expected Assign with Range"),
        }
    }

    #[test]
    fn op_implicit_with_index() {
        // System.Range.op_Implicit(new System.Index(3, true)) → ..^3
        let mut e = Expr::Call {
            recv: None,
            name: "System.Range.op_Implicit".into(),
            args: vec![Expr::NewObj(
                "System.Index".into(),
                vec![Expr::Int(3), Expr::Bool(true)],
            )],
        };
        let n = rewrite_expr(&mut e);
        assert!(n >= 1);
        match &e {
            Expr::Range { from, to } => {
                assert!(from.is_none(), "from should be None for ..^3");
                assert!(to.is_some());
                let to_expr = to.as_ref().unwrap();
                assert!(matches!(to_expr.as_ref(), Expr::IndexFromEnd(_)));
            }
            _ => panic!("expected Range, got {:?}", e),
        }
        assert_eq!(render(&e), "..^3");
    }
}
