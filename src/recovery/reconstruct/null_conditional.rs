//! Null-conditional operator reconstruction: `x?.Method()` / `x?.Field`.
//!
//! ## Pattern
//!
//! The C# compiler emits null-conditional access as an if-statement:
//!
//! ```text
//! if (x != null) x.Method();
//! ```
//!
//! or:
//!
//! ```text
//! if (x != null) { y = x.Property; }
//! ```
//!
//! IL2CPP preserves this shape faithfully, so the recovered IR contains the
//! same if-statement pattern.
//!
//! ## Detection
//!
//! The recognizer matches:
//! - `Stmt::If` with empty `else_body`
//! - Condition: `Expr::Cmp(CmpKind::Ne, Expr::Var(x), Expr::Null)` (or reversed)
//! - Then-body: exactly one statement
//! - The single statement mentions `x` (the null-checked variable)
//! - The statement is a method call, field access, or assignment whose RHS
//!   involves a method call or field access on `x`
//!
//! ## Safety conditions
//!
//! The rewrite is safe when:
//! 1. The condition is `x != null` (not `x == null` — that's the coalesce pattern).
//! 2. The else-body is empty (no else arm).
//! 3. The then-body has exactly one statement.
//! 4. The statement involves the null-checked variable as a receiver of a
//!    member access (method call or field access).
//!
//! ## Rewritten form
//!
//! The `if` is replaced with:
//! - `Stmt::Eval(Expr::NullConditional { receiver: x, access: x.Method() })`
//!   for bare method calls
//! - `Stmt::Assign(y, None, Expr::NullConditional { receiver: x, access: x.Field })`
//!   for assignments from field/property access
//!
//! The renderer emits `x?.Method()` and `y = x?.Field` respectively.

use super::super::csharp::CmpKind;
use super::{expr_mentions, walk_mut, Expr, Stmt};

/// Run the null-conditional recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += rewrite_list(list);
    });
    count
}

/// Scan a statement list for the null-conditional pattern and rewrite in place.
fn rewrite_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    for stmt in stmts.iter_mut() {
        count += rewrite_stmt(stmt);
    }
    count
}

/// Try to rewrite a single statement from an if-null-check to null-conditional.
fn rewrite_stmt(stmt: &mut Stmt) -> usize {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return 0;
    };

    // The else-body must be empty.
    if !else_body.is_empty() {
        return 0;
    }

    // The then-body must be exactly one statement.
    if then_body.len() != 1 {
        return 0;
    }

    // Extract the variable name from the null check condition.
    let var_name = match extract_null_check_ne(cond) {
        Some(name) => name,
        None => return 0,
    };

    // The single statement must mention the null-checked variable.
    if !stmt_mentions_var(&then_body[0], &var_name) {
        return 0;
    }

    // Try to convert the then-body statement to a null-conditional expression.
    match try_convert_to_null_conditional(&var_name, &then_body[0]) {
        Some(replacement) => {
            *stmt = replacement;
            1
        }
        None => 0,
    }
}

/// If `cond` is `x != null` or `null != x`, return `Some("x")`.
///
/// This matches the NOT-equal null check (opposite of null_coalesce which
/// matches `x == null`). Both orderings are matched because the compiler may
/// emit either depending on register allocation.
fn extract_null_check_ne(cond: &Expr) -> Option<String> {
    let Expr::Cmp(CmpKind::Ne, lhs, rhs) = cond else {
        return None;
    };
    // `x != null`
    if matches!(rhs.as_ref(), Expr::Null) {
        if let Expr::Var(name) = lhs.as_ref() {
            return Some(name.clone());
        }
    }
    // `null != x`
    if matches!(lhs.as_ref(), Expr::Null) {
        if let Expr::Var(name) = rhs.as_ref() {
            return Some(name.clone());
        }
    }
    None
}

/// Check if a statement mentions a variable (shallow: top-level expressions only).
fn stmt_mentions_var(stmt: &Stmt, name: &str) -> bool {
    match stmt {
        Stmt::Eval(e) => expr_mentions(e, name),
        Stmt::Assign(_, _, e) => expr_mentions(e, name),
        Stmt::Return(Some(e)) => expr_mentions(e, name),
        Stmt::Throw(e) => expr_mentions(e, name),
        _ => false,
    }
}

/// Try to convert a statement into a null-conditional form.
///
/// Returns the replacement statement, or `None` if the pattern doesn't match.
///
/// Supported patterns:
/// 1. `Eval(Call { recv: Var(x), name, args })` → `Eval(NullConditional { ... })`
/// 2. `Assign(target, ty, Call { recv: Var(x), name, args })` → `Assign(target, ty, NullConditional { ... })`
/// 3. `Assign(target, ty, Field(Var(x), name))` → `Assign(target, ty, NullConditional { ... })`
/// 4. `Eval(Field(Var(x), name))` → `Eval(NullConditional { ... })`
fn try_convert_to_null_conditional(var_name: &str, stmt: &Stmt) -> Option<Stmt> {
    match stmt {
        // Pattern 1: bare method call `x.Method(args)`
        Stmt::Eval(Expr::Call {
            recv: Some(recv),
            name,
            args,
        }) => {
            if is_var(recv, var_name) {
                let null_cond = Expr::NullConditional {
                    receiver: Box::new(recv.as_ref().clone()),
                    access: Box::new(Expr::Call {
                        recv: Some(Box::new(recv.as_ref().clone())),
                        name: name.clone(),
                        args: args.clone(),
                    }),
                };
                Some(Stmt::Eval(null_cond))
            } else {
                None
            }
        }

        // Pattern 2: assignment from method call `y = x.Method(args)`
        Stmt::Assign(target, ty, Expr::Call {
            recv: Some(recv),
            name,
            args,
        }) => {
            if is_var(recv, var_name) {
                let null_cond = Expr::NullConditional {
                    receiver: Box::new(recv.as_ref().clone()),
                    access: Box::new(Expr::Call {
                        recv: Some(Box::new(recv.as_ref().clone())),
                        name: name.clone(),
                        args: args.clone(),
                    }),
                };
                Some(Stmt::Assign(target.clone(), ty.clone(), null_cond))
            } else {
                None
            }
        }

        // Pattern 3: assignment from field access `y = x.Field`
        Stmt::Assign(target, ty, Expr::Field(recv, field_name)) => {
            if is_var(recv, var_name) {
                let null_cond = Expr::NullConditional {
                    receiver: Box::new(recv.as_ref().clone()),
                    access: Box::new(Expr::Field(Box::new(recv.as_ref().clone()), field_name.clone())),
                };
                Some(Stmt::Assign(target.clone(), ty.clone(), null_cond))
            } else {
                None
            }
        }

        // Pattern 4: bare field access eval `x.Field` (unusual but possible)
        Stmt::Eval(Expr::Field(recv, field_name)) => {
            if is_var(recv, var_name) {
                let null_cond = Expr::NullConditional {
                    receiver: Box::new(recv.as_ref().clone()),
                    access: Box::new(Expr::Field(Box::new(recv.as_ref().clone()), field_name.clone())),
                };
                Some(Stmt::Eval(null_cond))
            } else {
                None
            }
        }

        _ => None,
    }
}

/// Check if an expression is `Var(name)`.
fn is_var(e: &Expr, name: &str) -> bool {
    matches!(e, Expr::Var(n) if n == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::CmpKind;

    fn v(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    /// Helper: build `if (var != null) { then_stmt }` with empty else.
    fn if_ne_null(var: &str, then_body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond: Expr::Cmp(CmpKind::Ne, Box::new(v(var)), Box::new(Expr::Null)),
            then_body,
            else_body: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Positive tests (should rewrite)
    // -----------------------------------------------------------------------

    #[test]
    fn simple_method_call() {
        // if (x != null) x.Method()  →  x?.Method()
        let mut stmts = vec![if_ne_null(
            "x",
            vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("x"))),
                name: "Method".into(),
                args: vec![],
            })],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::NullConditional { receiver, access }) => {
                assert!(matches!(receiver.as_ref(), Expr::Var(n) if n == "x"));
                assert!(matches!(access.as_ref(), Expr::Call { name, .. } if name == "Method"));
            }
            other => panic!("expected Eval(NullConditional), got {other:?}"),
        }
    }

    #[test]
    fn field_access_assignment() {
        // if (x != null) { y = x.Field; }  →  y = x?.Field
        let mut stmts = vec![if_ne_null(
            "x",
            vec![Stmt::Assign(
                "y".into(),
                None,
                Expr::Field(Box::new(v("x")), "Field".into()),
            )],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::NullConditional { receiver, access }) => {
                assert_eq!(target, "y");
                assert!(matches!(receiver.as_ref(), Expr::Var(n) if n == "x"));
                assert!(matches!(access.as_ref(), Expr::Field(_, name) if name == "Field"));
            }
            other => panic!("expected Assign(y, NullConditional), got {other:?}"),
        }
    }

    #[test]
    fn property_access_assignment() {
        // if (x != null) { y = x.get_Value(); }  →  y = x?.get_Value()
        let mut stmts = vec![if_ne_null(
            "x",
            vec![Stmt::Assign(
                "y".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(v("x"))),
                    name: "get_Value".into(),
                    args: vec![],
                },
            )],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::NullConditional { .. }) => {
                assert_eq!(target, "y");
            }
            other => panic!("expected Assign(y, NullConditional), got {other:?}"),
        }
    }

    #[test]
    fn null_check_on_right() {
        // if (null != x) x.Method()  →  x?.Method()
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Ne, Box::new(Expr::Null), Box::new(v("x"))),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("x"))),
                name: "Method".into(),
                args: vec![],
            })],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::NullConditional { .. })));
    }

    #[test]
    fn method_call_with_args() {
        // if (x != null) x.DoSomething(42, "hello")  →  x?.DoSomething(42, "hello")
        let mut stmts = vec![if_ne_null(
            "x",
            vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("x"))),
                name: "DoSomething".into(),
                args: vec![Expr::Int(42), Expr::Str("hello".into())],
            })],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::NullConditional { access, .. }) => {
                if let Expr::Call { args, .. } = access.as_ref() {
                    assert_eq!(args.len(), 2);
                } else {
                    panic!("expected Call access");
                }
            }
            other => panic!("expected Eval(NullConditional), got {other:?}"),
        }
    }

    #[test]
    fn nested_inside_while_is_found() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![if_ne_null(
                "x",
                vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("x"))),
                    name: "Tick".into(),
                    args: vec![],
                })],
            )],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }

    // -----------------------------------------------------------------------
    // Negative tests (should NOT rewrite)
    // -----------------------------------------------------------------------

    #[test]
    fn else_body_not_empty() {
        // if (x != null) x.Method(); else { other(); }  — has else, no match.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Ne, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("x"))),
                name: "Method".into(),
                args: vec![],
            })],
            else_body: vec![Stmt::Eval(v("other"))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn then_body_multiple_statements() {
        // if (x != null) { x.Method(); x.Other(); }  — more than one stmt.
        let mut stmts = vec![if_ne_null(
            "x",
            vec![
                Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("x"))),
                    name: "Method".into(),
                    args: vec![],
                }),
                Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("x"))),
                    name: "Other".into(),
                    args: vec![],
                }),
            ],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn condition_is_not_null_check() {
        // if (x != 0) x.Method()  — not a null check.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Ne, Box::new(v("x")), Box::new(Expr::Int(0))),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("x"))),
                name: "Method".into(),
                args: vec![],
            })],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn then_body_does_not_mention_variable() {
        // if (x != null) { y.Method(); }  — then body doesn't use x.
        let mut stmts = vec![if_ne_null(
            "x",
            vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("y"))),
                name: "Method".into(),
                args: vec![],
            })],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn eq_null_is_not_matched() {
        // if (x == null) x.Method()  — this is == null, not != null.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("x"))),
                name: "Method".into(),
                args: vec![],
            })],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn non_member_access_stmt() {
        // if (x != null) { y = x + 1; }  — not a member access on x.
        let mut stmts = vec![if_ne_null(
            "x",
            vec![Stmt::Assign(
                "y".into(),
                None,
                Expr::Bin(
                    crate::recovery::csharp::BinOp::Add,
                    Box::new(v("x")),
                    Box::new(Expr::Int(1)),
                ),
            )],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn rendering_null_conditional_method_call() {
        // Verify the renderer outputs `x?.Method()`.
        let expr = Expr::NullConditional {
            receiver: Box::new(v("x")),
            access: Box::new(Expr::Call {
                recv: Some(Box::new(v("x"))),
                name: "Method".into(),
                args: vec![],
            }),
        };
        let mut s = String::new();
        expr.render(&mut s);
        assert_eq!(s, "x?.Method()");
    }

    #[test]
    fn rendering_null_conditional_field() {
        // Verify the renderer outputs `x?.Field`.
        let expr = Expr::NullConditional {
            receiver: Box::new(v("x")),
            access: Box::new(Expr::Field(Box::new(v("x")), "Field".into())),
        };
        let mut s = String::new();
        expr.render(&mut s);
        assert_eq!(s, "x?.Field");
    }
}
