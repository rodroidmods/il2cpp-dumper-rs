//! Remove unnecessary trailing `continue` statements from loop bodies.
//!
//! ## Why this pass exists
//!
//! The decompiler emits ~4,977 `continue` statements across the corpus. A
//! `continue` as the last statement in a loop body is redundant — the loop
//! naturally continues at that point. Removing them makes the output cleaner
//! and more idiomatic.
//!
//! ## The rule
//!
//! A `Stmt::Continue` that is the **last** statement in a loop body
//! (`while`, `do-while`, `for`, `foreach`) is removed, provided the body
//! contains at least one other statement (an empty loop body with only a
//! `continue` is left alone — that shape is unusual and may indicate an
//! unrecoverable control-flow region).
//!
//! ## Safety
//!
//! - Only the **last** statement in the body is considered. A `continue`
//!   inside an `if` block that happens to be the last statement of the loop
//!   is NOT removed (it is not a direct child of the loop body).
//! - The body must have at least 2 statements before removal (the continue
//!   plus at least one other). This prevents reducing a loop to an empty
//!   body.
//! - The pass recurses into nested structures so continues inside loops
//!   within `if` blocks, `try` bodies, etc. are reached.

use crate::recovery::csharp::{Expr, Stmt};
use super::expr_has_call;

/// Entry point: walk the statement tree and remove trailing continues from
/// loop bodies. Returns the number of continues removed.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut rewrites = 0;
    for stmt in stmts.iter_mut() {
        rewrites += rewrite_stmt(stmt);
    }
    rewrites
}

/// `B; if (COND) continue;` at the tail of a loop body is the same as `B;`
/// when COND is the loop condition and evaluating it twice is equivalent.
/// Only reached for conditions that contain no call, so the extra evaluation
/// cannot change the answer.
fn drop_redundant_continue_test(body: &mut Vec<Stmt>, cond: &Expr) -> usize {
    if body.len() < 2 || expr_has_call(cond) {
        return 0;
    }
    let redundant = matches!(
        body.last(),
        Some(Stmt::If { cond: inner, then_body, else_body })
            if else_body.is_empty()
                && matches!(then_body.as_slice(), [Stmt::Continue])
                && inner == cond
    );
    if !redundant {
        return 0;
    }
    body.pop();
    1
}

/// Recursive worker: for each statement, check if it is a loop with a
/// trailing continue, remove it, then recurse into child bodies.
fn rewrite_stmt(stmt: &mut Stmt) -> usize {
    let mut rewrites = 0;

    match stmt {
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            // Remove trailing continue if the body has other statements.
            if body.len() > 1 {
                if matches!(body.last(), Some(Stmt::Continue)) {
                    body.pop();
                    rewrites += 1;
                }
            }
            // `B; if (COND) continue;` at the tail re-tests the very condition
            // the loop is about to test: drop the guard, keep `B`.
            rewrites += drop_redundant_continue_test(body, cond);
            // Recurse into the body for nested loops.
            rewrites += apply(body);
        }
        Stmt::ForEach { body, .. } => {
            // Remove trailing continue if the body has other statements.
            if body.len() > 1 {
                if matches!(body.last(), Some(Stmt::Continue)) {
                    body.pop();
                    rewrites += 1;
                }
            }
            // Recurse into the body for nested loops.
            rewrites += apply(body);
        }
        Stmt::For { init, incr, body, cond, .. } => {
            // Remove trailing continue from the loop body.
            if body.len() > 1 {
                if matches!(body.last(), Some(Stmt::Continue)) {
                    body.pop();
                    rewrites += 1;
                }
            }
            if let Some(cond) = cond {
                rewrites += drop_redundant_continue_test(body, cond);
            }
            // Recurse into init, incr, and body.
            rewrites += apply(init);
            rewrites += apply(incr);
            rewrites += apply(body);
        }
        // Recurse into other nested structures.
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            rewrites += apply(then_body);
            rewrites += apply(else_body);
        }
        Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
            rewrites += apply(body);
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            rewrites += apply(try_body);
            for (_, catch_body) in catches.iter_mut() {
                rewrites += apply(catch_body);
            }
            if let Some(fb) = finally_body {
                rewrites += apply(fb);
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, case_body) in cases.iter_mut() {
                rewrites += apply(case_body);
            }
            rewrites += apply(tail);
            if let Some(arm) = default {
                rewrites += apply(arm);
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, case_body) in cases.iter_mut() {
                rewrites += apply(case_body);
            }
            if let Some(t) = tail {
                rewrites += apply(t);
            }
        }
        _ => {}
    }

    rewrites
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::{Expr, Stmt};

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    fn eval(name: &str) -> Stmt {
        Stmt::Eval(var(name))
    }

    // 1. Simple while with trailing continue — should be removed.
    #[test]
    fn while_with_trailing_continue() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![eval("x"), Stmt::Continue],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0], Stmt::Eval(Expr::Var(v)) if v == "x"));
            }
            _ => panic!("expected While"),
        }
    }

    // 2. For loop with trailing continue — should be removed.
    #[test]
    fn for_loop_with_trailing_continue() {
        let mut stmts = vec![Stmt::For {
            init: vec![Stmt::Assign("i".into(), None, Expr::Int(0))],
            cond: Some(Expr::Bool(true)),
            incr: vec![Stmt::Assign("i".into(), None, Expr::Int(1))],
            body: vec![eval("work"), Stmt::Continue],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::For { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0], Stmt::Eval(Expr::Var(v)) if v == "work"));
            }
            _ => panic!("expected For"),
        }
    }

    // 3. Do-while with trailing continue — should be removed.
    #[test]
    fn do_while_with_trailing_continue() {
        let mut stmts = vec![Stmt::DoWhile {
            body: vec![eval("x"), eval("y"), Stmt::Continue],
            cond: Expr::Bool(true),
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::DoWhile { body, .. } => {
                assert_eq!(body.len(), 2);
                assert!(!matches!(body.last(), Some(Stmt::Continue)));
            }
            _ => panic!("expected DoWhile"),
        }
    }

    // 4. Continue inside if (not last statement of loop) — should NOT be removed.
    #[test]
    fn continue_inside_if_not_removed() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::If {
                    cond: Expr::Bool(true),
                    then_body: vec![Stmt::Continue],
                    else_body: vec![],
                },
                eval("after_if"),
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        // The continue inside the if must still be there.
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 2);
                match &body[0] {
                    Stmt::If { then_body, .. } => {
                        assert_eq!(then_body.len(), 1);
                        assert!(matches!(&then_body[0], Stmt::Continue));
                    }
                    _ => panic!("expected If"),
                }
            }
            _ => panic!("expected While"),
        }
    }

    // 5. Continue as only statement in loop — should NOT be removed.
    #[test]
    fn continue_as_only_statement_not_removed() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Continue],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0], Stmt::Continue));
            }
            _ => panic!("expected While"),
        }
    }

    // 6. Multiple loops with continues — both should be removed.
    #[test]
    fn multiple_loops_with_continues() {
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![eval("a"), Stmt::Continue],
            },
            Stmt::While {
                cond: Expr::Bool(false),
                body: vec![eval("b"), Stmt::Continue],
            },
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2);
        match &stmts[0] {
            Stmt::While { body, .. } => assert_eq!(body.len(), 1),
            _ => panic!("expected While"),
        }
        match &stmts[1] {
            Stmt::While { body, .. } => assert_eq!(body.len(), 1),
            _ => panic!("expected While"),
        }
    }

    // 7. Nested loops with continues — both inner and outer removed.
    #[test]
    fn nested_loops_with_continues() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::While {
                    cond: Expr::Bool(true),
                    body: vec![eval("inner_work"), Stmt::Continue],
                },
                Stmt::Continue,
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both inner and outer continue should be removed");
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 1, "outer body should have only the inner loop");
                match &body[0] {
                    Stmt::While { body: inner_body, .. } => {
                        assert_eq!(inner_body.len(), 1, "inner body should have only the eval");
                    }
                    _ => panic!("expected inner While"),
                }
            }
            _ => panic!("expected outer While"),
        }
    }

    // 8. Continue in nested if inside loop — should NOT be removed.
    #[test]
    fn continue_in_nested_if_inside_loop_not_removed() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::If {
                    cond: Expr::Bool(true),
                    then_body: vec![eval("check"), Stmt::Continue],
                    else_body: vec![],
                },
                eval("end_of_body"),
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "continue is inside if, not trailing in loop body");
    }

    // 9. Multiple continues in one loop — only the last one is removed.
    #[test]
    fn multiple_continues_only_last_removed() {
        // An if with a continue in then_body, plus a trailing continue.
        // Only the trailing continue should be removed.
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::If {
                    cond: Expr::Bool(true),
                    then_body: vec![Stmt::Continue],
                    else_body: vec![],
                },
                eval("work"),
                Stmt::Continue,
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "only the trailing continue is removed");
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 2, "if + work remain");
                // The if's continue must still be there.
                match &body[0] {
                    Stmt::If { then_body, .. } => {
                        assert!(matches!(&then_body[0], Stmt::Continue));
                    }
                    _ => panic!("expected If"),
                }
            }
            _ => panic!("expected While"),
        }
    }

    // 10. Continue NOT as last statement — should NOT be removed.
    #[test]
    fn continue_not_last_not_removed() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![eval("x"), Stmt::Continue, eval("y")],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "continue is not the last statement");
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 3, "nothing should change");
            }
            _ => panic!("expected While"),
        }
    }

    // Bonus: ForEach with trailing continue.
    #[test]
    fn foreach_with_trailing_continue() {
        let mut stmts = vec![Stmt::ForEach {
            var_name: "item".into(),
            collection: "items".into(),
            body: vec![eval("process"), Stmt::Continue],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::ForEach { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0], Stmt::Eval(Expr::Var(v)) if v == "process"));
            }
            _ => panic!("expected ForEach"),
        }
    }

    // Bonus: No loops — no rewrites.
    #[test]
    fn no_loops_no_rewrites() {
        let mut stmts = vec![eval("a"), eval("b"), Stmt::Continue];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 3);
    }
}
