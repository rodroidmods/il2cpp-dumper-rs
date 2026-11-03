//! Pattern matching recognition: `if (x is T t) { ... t ... }`.
//!
//! ## Pattern
//!
//! C# 7.0 introduced pattern matching with the `is` expression:
//! ```csharp
//! if (x is string s) {
//!     Console.WriteLine(s.Length);
//! }
//! ```
//!
//! The IL2CPP compiler lowers this to:
//! ```text
//! if (x is string) {
//!     string s = (string)x;
//!     ... s ...
//! }
//! ```
//!
//! ## Detection
//!
//! This pass detects the simplified form where:
//! 1. An `if` condition tests `x is T` (a type check via `isinst_T` call)
//! 2. The then-body starts with `T t = (T)x;` (a cast assignment)
//! 3. The variable `t` is used in the rest of the then-body
//!
//! When detected, the pass:
//! - Replaces the condition with `Expr::PatternMatch { expr: x, type_name: T, binding: t }`
//! - Removes the cast assignment from the then-body
//! - The renderer emits `x is T t` for the condition
//!
//! ## Safety conditions
//!
//! The rewrite is safe when:
//! 1. The condition is a type check (`isinst_T(x)` call).
//! 2. The first statement in the then-body is a cast of the same variable.
//! 3. The cast target type matches the type in the condition.
//! 4. The bound variable `t` is used in the remaining then-body (otherwise the
//!    cast is dead code and removing it changes nothing observable).
//!
//! ## Measured reality
//!
//! Pattern matching is rare in the x64 corpus. Most uses are simple null checks
//! (handled by `null_coalesce`) or type checks without variable binding. This
//! pass handles the subset where the cast is the FIRST statement in the
//! then-body and the types match exactly.

use super::{Expr, Stmt, walk_mut};

/// Run the pattern matching reconstructor over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += reconstruct_list(list);
    });
    count
}

/// Scan a statement list for pattern matching and reconstruct them.
fn reconstruct_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < stmts.len() {
        if try_reconstruct(&mut stmts[i]) {
            count += 1;
        }
        i += 1;
    }
    count
}

/// Try to reconstruct a pattern matching construct in a single statement.
///
/// Returns `true` when the pattern was detected and rewritten, `false` otherwise.
fn try_reconstruct(stmt: &mut Stmt) -> bool {
    let Stmt::If {
        cond,
        then_body,
        else_body: _,
    } = stmt
    else {
        return false;
    };

    // We need at least one statement in the then-body for the cast.
    if then_body.is_empty() {
        return false;
    }

    // Detect the type check in the condition.
    let (checked_var, checked_type) = match extract_type_check(cond) {
        Some(v) => v,
        None => return false,
    };

    // The first statement in the then-body should be a cast of the checked var.
    let Stmt::Assign(cast_target, _cast_ty, cast_expr) = &then_body[0] else {
        return false;
    };
    let cast_target = cast_target.clone();

    // The cast expression should be `Expr::Cast(type, Var(checked_var))`.
    let Expr::Cast(cast_type, inner) = cast_expr else {
        return false;
    };

    // The inner expression should be the checked variable.
    let Expr::Var(inner_var) = inner.as_ref() else {
        return false;
    };

    if inner_var != checked_var {
        return false;
    }

    // The cast type should match the checked type.
    if cast_type != checked_type {
        return false;
    }

    // Verify the bound variable is used in the rest of the then-body.
    // If it's not used, the cast is dead code and we shouldn't introduce a
    // pattern variable that nothing references.
    let rest_uses_binding = then_body[1..]
        .iter()
        .any(|s| stmt_mentions(s, &cast_target));

    if !rest_uses_binding {
        return false;
    }

    // All checks passed! Perform the rewrite:
    // 1. Replace the condition with PatternMatch.
    let checked_var_owned = checked_var.to_string();
    let checked_type_owned = checked_type.to_string();
    *cond = Expr::PatternMatch {
        expr: Box::new(Expr::Var(checked_var_owned)),
        type_name: checked_type_owned,
        binding: cast_target.clone(),
    };

    // 2. Remove the cast assignment from the then-body.
    then_body.remove(0);

    // 3. Replace all uses of `cast_target` in the then-body with a plain Var
    //    reference (the binding is now implicit from the pattern match).
    //    Actually, the variable name stays the same — the pattern match binding
    //    declares `cast_target` in scope. No replacement needed.

    true
}

/// Check if a statement mentions a variable name (shallow — top-level expressions only).
fn stmt_mentions(stmt: &Stmt, name: &str) -> bool {
    match stmt {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => expr_mentions(e, name),
        Stmt::Assign(_, _, e) => expr_mentions(e, name),
        Stmt::Return(Some(e)) => expr_mentions(e, name),
        Stmt::If { cond, then_body, else_body } => {
            expr_mentions(cond, name)
                || list_mentions(then_body, name)
                || list_mentions(else_body, name)
        }
        Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
            expr_mentions(cond, name) || list_mentions(body, name)
        }
        Stmt::For { init, cond, incr, body } => {
            list_mentions(init, name)
                || cond.as_ref().is_some_and(|c| expr_mentions(c, name))
                || list_mentions(incr, name)
                || list_mentions(body, name)
        }
        Stmt::ForEach { var_name, collection, body } => {
            (var_name == name || collection == name) || list_mentions(body, name)
        }
        Stmt::Using { decl, body } => stmt_mentions(decl, name) || list_mentions(body, name),
        Stmt::Lock { expr, body } => expr_mentions(expr, name) || list_mentions(body, name),
        Stmt::Try { try_body, catches, finally_body } => {
            list_mentions(try_body, name)
                || catches.iter().any(|(_, b)| list_mentions(b, name))
                || finally_body.as_ref().is_some_and(|fb| list_mentions(fb, name))
        }
        Stmt::Switch { cases, tail, default, .. } => {
            cases.iter().any(|(_, b)| list_mentions(b, name))
                || list_mentions(tail, name)
                || default.as_ref().is_some_and(|arm| list_mentions(arm, name))
        }
        Stmt::StringSwitch { disc, cases, tail } => {
            expr_mentions(disc, name)
                || cases.iter().any(|(_, b)| list_mentions(b, name))
                || tail.as_ref().is_some_and(|t| list_mentions(t, name))
        }
        _ => false,
    }
}

/// Check if any statement in a list mentions a variable name.
fn list_mentions(stmts: &[Stmt], name: &str) -> bool {
    stmts.iter().any(|s| stmt_mentions(s, name))
}

/// Check if an expression mentions a variable name anywhere.
fn expr_mentions(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Var(n) => n == name,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_mentions(a, name) || expr_mentions(b, name),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            expr_mentions(x, name)
        }
        Expr::Field(r, f) => f == name || expr_mentions(r, name),
        Expr::StaticField(_, f) => f == name,
        Expr::Call { recv, args, .. } => {
            recv.as_deref().is_some_and(|r| expr_mentions(r, name))
                || args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => expr_mentions(r, name),
        Expr::Cond(c, a, b) => {
            expr_mentions(c, name) || expr_mentions(a, name) || expr_mentions(b, name)
        }
        Expr::Index { collection, index } => {
            expr_mentions(collection, name) || expr_mentions(index, name)
        }
        Expr::Lambda { params, body } => {
            !params.iter().any(|p| p.as_str() == name) && expr_mentions(body, name)
        }
        Expr::ArrayInit { items, .. } => items.iter().any(|a| expr_mentions(a, name)),
        Expr::CollectionInit { ctor_args, items, .. } => {
            ctor_args.iter().any(|a| expr_mentions(a, name))
                || items.iter().any(|a| expr_mentions(a, name))
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            ctor_args.iter().any(|a| expr_mentions(a, name))
                || fields.iter().any(|(_, v)| expr_mentions(v, name))
        }
        Expr::NullConditional { receiver, access } => {
            expr_mentions(receiver, name) || expr_mentions(access, name)
        }
        Expr::PatternMatch { expr, .. } => expr_mentions(expr, name),
        Expr::StringInterp { parts } => {
            use crate::recovery::csharp::InterpPart;
            parts.iter().any(|p| match p {
                InterpPart::Literal(_) => false,
                InterpPart::Expr(e, _) => expr_mentions(e, name),
            })
        }
        Expr::Tuple { elements, .. } => elements.iter().any(|e| expr_mentions(e, name)),
        Expr::MethodChain { receiver, methods } => {
            expr_mentions(receiver, name)
                || methods.iter().any(|m| m.args.iter().any(|a| expr_mentions(a, name)))
        }
        Expr::SwitchExpr { discrim, arms, default_arm } => {
            expr_mentions(discrim, name)
                || arms.iter().any(|(p, r)| expr_mentions(p, name) || expr_mentions(r, name))
                || default_arm.as_deref().is_some_and(|e| expr_mentions(e, name))
        }
        Expr::Range { from, to } => {
            from.as_ref().is_some_and(|f| expr_mentions(f, name))
                || to.as_ref().is_some_and(|t| expr_mentions(t, name))
        }
        Expr::IndexFromEnd(inner) => expr_mentions(inner, name),
        Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::Unknown
        | Expr::MissingAbiArg
        | Expr::MethodGroup(_) | Expr::TypeOf(_) => false,
    }
}

/// Extract the variable and type from a type check condition.
///
/// Returns `Some((var_name, type_name))` if the condition is a type check,
/// `None` otherwise.
///
/// The IR does not have a dedicated `is` operator, so type checks appear as
/// calls to type-checking helpers. We look for `isinst_T(x)` calls.
fn extract_type_check(cond: &Expr) -> Option<(&str, &str)> {
    // Shape: A call like `isinst_String(x)`.
    if let Expr::Call { recv: None, name, args } = cond {
        if let Some(type_name) = name.strip_prefix("isinst_") {
            if !type_name.is_empty() && args.len() == 1 {
                if let Expr::Var(var_name) = &args[0] {
                    return Some((var_name.as_str(), type_name));
                }
            }
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
    fn no_pattern_in_plain_if() {
        // A plain if with no type check should not be rewritten.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                crate::recovery::csharp::CmpKind::Eq,
                Box::new(v("x")),
                Box::new(Expr::Null),
            ),
            then_body: vec![Stmt::Assign("y".into(), None, v("x"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn no_pattern_when_no_cast() {
        // A type check without a following cast should not be rewritten.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Call {
                recv: None,
                name: "isinst_String".into(),
                args: vec![v("x")],
            },
            then_body: vec![Stmt::Eval(v("other"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn pattern_reconstructed_with_isinst_call() {
        // if (isinst_String(x)) { string s = (string)x; use(s); }
        // => if (x is String s) { use(s); }
        let mut stmts = vec![Stmt::If {
            cond: Expr::Call {
                recv: None,
                name: "isinst_String".into(),
                args: vec![v("x")],
            },
            then_body: vec![
                Stmt::Assign("s".into(), None, Expr::Cast("String".into(), Box::new(v("x")))),
                Stmt::Eval(v("s")),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        // The condition should now be a PatternMatch.
        match &stmts[0] {
            Stmt::If { cond, then_body, .. } => {
                match cond {
                    Expr::PatternMatch { expr, type_name, binding } => {
                        assert!(matches!(expr.as_ref(), Expr::Var(n) if n == "x"));
                        assert_eq!(type_name, "String");
                        assert_eq!(binding, "s");
                    }
                    other => panic!("expected PatternMatch, got {other:?}"),
                }
                // The cast assignment should be removed.
                assert_eq!(then_body.len(), 1, "cast assignment removed");
                assert!(matches!(&then_body[0], Stmt::Eval(Expr::Var(n)) if n == "s"));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn mismatched_type_prevents_reconstruction() {
        // The cast type doesn't match the type check type.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Call {
                recv: None,
                name: "isinst_String".into(),
                args: vec![v("x")],
            },
            then_body: vec![Stmt::Assign(
                "s".into(),
                None,
                Expr::Cast("Int32".into(), Box::new(v("x"))),
            )],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn mismatched_variable_prevents_reconstruction() {
        // The cast is on a different variable than the type check.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Call {
                recv: None,
                name: "isinst_String".into(),
                args: vec![v("x")],
            },
            then_body: vec![Stmt::Assign(
                "s".into(),
                None,
                Expr::Cast("String".into(), Box::new(v("y"))),
            )],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn empty_then_body_prevents_reconstruction() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Call {
                recv: None,
                name: "isinst_String".into(),
                args: vec![v("x")],
            },
            then_body: vec![],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn unused_binding_prevents_reconstruction() {
        // The bound variable `s` is not used in the rest of the then-body.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Call {
                recv: None,
                name: "isinst_String".into(),
                args: vec![v("x")],
            },
            then_body: vec![
                Stmt::Assign("s".into(), None, Expr::Cast("String".into(), Box::new(v("x")))),
                Stmt::Eval(v("other")), // doesn't use `s`
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "unused binding should not be reconstructed");
    }

    #[test]
    fn nested_pattern_is_found() {
        // Pattern nested inside another if should still be detected.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::If {
                cond: Expr::Call {
                    recv: None,
                    name: "isinst_Int32".into(),
                    args: vec![v("obj")],
                },
                then_body: vec![
                    Stmt::Assign(
                        "n".into(),
                        None,
                        Expr::Cast("Int32".into(), Box::new(v("obj"))),
                    ),
                    Stmt::Eval(v("n")),
                ],
                else_body: vec![],
            }],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }

    #[test]
    fn pattern_match_renders_correctly() {
        // Verify the renderer outputs `x is String s`.
        let expr = Expr::PatternMatch {
            expr: Box::new(v("x")),
            type_name: "String".into(),
            binding: "s".into(),
        };
        let mut out = String::new();
        expr.render(&mut out);
        assert_eq!(out, "x is String s");
    }

    #[test]
    fn multiple_patterns_in_sequence() {
        // Two pattern matches in the same statement list.
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Call {
                    recv: None,
                    name: "isinst_String".into(),
                    args: vec![v("a")],
                },
                then_body: vec![
                    Stmt::Assign("s".into(), None, Expr::Cast("String".into(), Box::new(v("a")))),
                    Stmt::Eval(v("s")),
                ],
                else_body: vec![],
            },
            Stmt::If {
                cond: Expr::Call {
                    recv: None,
                    name: "isinst_Int32".into(),
                    args: vec![v("b")],
                },
                then_body: vec![
                    Stmt::Assign("n".into(), None, Expr::Cast("Int32".into(), Box::new(v("b")))),
                    Stmt::Eval(v("n")),
                ],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both patterns should be reconstructed");
    }

    #[test]
    fn pattern_with_receiver_call() {
        // The checked expression is a field access, not a plain variable.
        // This should NOT match because extract_type_check only handles Var args.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Call {
                recv: None,
                name: "isinst_String".into(),
                args: vec![Expr::Field(Box::new(v("obj")), "Value".into())],
            },
            then_body: vec![
                Stmt::Assign("s".into(), None, Expr::Cast("String".into(), Box::new(v("x")))),
                Stmt::Eval(v("s")),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "non-Var argument should not match");
    }
}
