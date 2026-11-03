//! Null-coalescing assignment reconstruction: `x ??= value`.
//!
//! ## Pattern
//!
//! The C# compiler emits `if (x == null) x = value;` for the null-coalescing
//! assignment operator `x ??= value;`. IL2CPP preserves this shape faithfully,
//! so the recovered IR contains:
//!
//! ```text
//! if (x == null) {
//!     x = value;
//! }
//! ```
//!
//! ## Detection
//!
//! The recognizer matches:
//! - `Stmt::If` with no `else_body`
//! - Condition: `Expr::Cmp(CmpKind::Eq, Expr::Var(x), Expr::Null)` (or reversed)
//! - Then-body: exactly one `Stmt::Assign(target, None, value)` where `target == x`
//!
//! ## Safety conditions
//!
//! The rewrite is safe when:
//! 1. The condition tests the SAME variable as the assignment target.
//! 2. The comparison is `== null` (not `!= null` — that would be `x ?? value`).
//! 3. The then-body has exactly one statement (the assignment).
//! 4. No else-body exists (an else would change the semantics).
//!
//! ## Rewritten form
//!
//! The `if` is replaced with:
//! ```text
//! Stmt::Assign(x, None, Expr::Bin(BinOp::NullCoalesceAssign, Expr::Var(x), value))
//! ```
//!
//! The renderer has special-case handling for this pattern and emits
//! `x ??= value;` instead of the redundant `x = x ??= value;`.
//!
//! ## Measured reality
//!
//! On the x64 corpus (47,924 methods), this pattern is rare but present.
//! The nullability pass runs before this one and may convert `x == 0` to
//! `x == null` for reference-typed locals, which creates additional sites
//! for this pass to match.

use super::super::csharp::{BinOp, CmpKind};
use super::{Expr, Stmt, walk_mut};

/// Run the null-coalescing assignment recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += rewrite_list(list);
    });
    count
}

/// Scan a statement list for the null-coalesce pattern and rewrite in place.
fn rewrite_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < stmts.len() {
        if let Some(replacement) = try_match(&stmts[i]) {
            stmts[i] = replacement;
            count += 1;
        }
        i += 1;
    }
    count
}

/// Try to match the null-coalesce pattern on a single statement.
///
/// Returns `Some(replacement)` when the statement is
/// `if (x == null) { x = value; }` and `None` otherwise.
fn try_match(stmt: &Stmt) -> Option<Stmt> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };

    // The else-body must be empty — `x ??= value` has no else arm.
    if !else_body.is_empty() {
        return None;
    }

    // The then-body must be exactly one assignment.
    if then_body.len() != 1 {
        return None;
    }

    let Stmt::Assign(target, ty, value) = &then_body[0] else {
        return None;
    };

    // The assignment must not have a type declaration (that would be a
    // declaration, not an assignment — `??=` requires an existing variable).
    if ty.is_some() {
        return None;
    }

    // Extract the variable name from the condition and check it matches the
    // assignment target.
    let var_name = extract_null_check_var(cond)?;

    // The condition variable must be the same as the assignment target.
    if var_name != target {
        return None;
    }

    // Build the replacement: `x ??= value`
    Some(Stmt::Assign(
        target.clone(),
        None,
        Expr::Bin(
            BinOp::NullCoalesceAssign,
            Box::new(Expr::Var(target.clone())),
            Box::new(value.clone()),
        ),
    ))
}

/// If `cond` is `x == null` or `null == x`, return `Some("x")`.
///
/// Both orderings are matched because the compiler may emit either depending
/// on the register allocation. The nullability pass may have converted
/// `x == 0` to `x == null` for reference-typed locals.
fn extract_null_check_var(cond: &Expr) -> Option<&str> {
    let Expr::Cmp(CmpKind::Eq, lhs, rhs) = cond else {
        return None;
    };
    // `x == null`
    if matches!(rhs.as_ref(), Expr::Null) {
        if let Expr::Var(name) = lhs.as_ref() {
            return Some(name.as_str());
        }
    }
    // `null == x`
    if matches!(lhs.as_ref(), Expr::Null) {
        if let Expr::Var(name) = rhs.as_ref() {
            return Some(name.as_str());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    #[test]
    fn basic_null_coalesce_x_eq_null() {
        // if (x == null) { x = value; }  ->  x ??= value;
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("x".into(), None, v("value"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(name, None, Expr::Bin(BinOp::NullCoalesceAssign, lhs, rhs)) => {
                assert_eq!(name, "x");
                assert!(matches!(lhs.as_ref(), Expr::Var(n) if n == "x"));
                assert!(matches!(rhs.as_ref(), Expr::Var(n) if n == "value"));
            }
            other => panic!("expected Assign(NullCoalesceAssign), got {other:?}"),
        }
    }

    #[test]
    fn reversed_null_eq_x() {
        // if (null == x) { x = value; }  ->  x ??= value;
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(Expr::Null), Box::new(v("x"))),
            then_body: vec![Stmt::Assign("x".into(), None, v("fallback"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(
            &stmts[0],
            Stmt::Assign(name, None, Expr::Bin(BinOp::NullCoalesceAssign, _, _)) if name == "x"
        ));
    }

    #[test]
    fn not_equal_null_is_not_matched() {
        // if (x != null) { x = value; }  should NOT be rewritten.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Ne, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("x".into(), None, v("value"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn else_body_present_prevents_rewrite() {
        // if (x == null) { x = value; } else { other(); }  should NOT match.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("x".into(), None, v("value"))],
            else_body: vec![Stmt::Eval(v("other"))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn mismatched_variable_prevents_rewrite() {
        // if (x == null) { y = value; }  — different variables, no match.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("y".into(), None, v("value"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn typed_declaration_prevents_rewrite() {
        // if (x == null) { int x = value; }  — has type decl, no match.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("x".into(), Some("int".into()), v("value"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn multiple_statements_in_then_prevents_rewrite() {
        // if (x == null) { x = value; other(); }  — more than one stmt.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![
                Stmt::Assign("x".into(), None, v("value")),
                Stmt::Eval(v("other")),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn nested_inside_if_is_found() {
        // The pattern nested inside another if should still be found.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("x".into(), None, v("default"))],
                else_body: vec![],
            }],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }

    #[test]
    fn non_null_comparison_prevents_rewrite() {
        // if (x == 0) { x = value; }  — not a null check.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Int(0))),
            then_body: vec![Stmt::Assign("x".into(), None, v("value"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn complex_value_is_preserved() {
        // The value expression can be complex — it must be preserved exactly.
        let complex = Expr::Call {
            recv: Some(Box::new(v("factory"))),
            name: "Create".into(),
            args: vec![Expr::Int(42)],
        };
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("cache")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Assign("cache".into(), None, complex.clone())],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Bin(BinOp::NullCoalesceAssign, _, rhs)) => {
                // The RHS should be the complex call expression.
                assert!(matches!(rhs.as_ref(), Expr::Call { .. }));
            }
            other => panic!("expected NullCoalesceAssign, got {other:?}"),
        }
    }

    #[test]
    fn multiple_patterns_in_one_list() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("a")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("a".into(), None, v("default_a"))],
                else_body: vec![],
            },
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("b")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("b".into(), None, v("default_b"))],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2);
    }

    #[test]
    fn inside_while_body() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("x".into(), None, v("fallback"))],
                else_body: vec![],
            }],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }
}
