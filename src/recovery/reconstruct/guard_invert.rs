//! Hoist the else arm of a terminated guard.
//!
//! ## Why this pass exists
//!
//! IL2CPP's control flow is a tree of two-armed branches, and `branch_repair`
//! rebuilds each one as `if/else`. When the then-arm is nothing but a
//! terminator, the else-arm is really the continuation:
//!
//! ```csharp
//! if (this._listBank == null)
//! {
//!     return;
//! } else
//! {
//!     ...the rest of the method...
//! }
//! ```
//!
//! R91 had **9,233** of these. Lifting the arm out of the `else` keeps the
//! early exit and flattens the body one level:
//!
//! ```csharp
//! if (this._listBank == null) return;
//! ...the rest of the method...
//! ```
//!
//! ## The rule
//!
//! `if (C) { <terminator> } else { BODY }` becomes `if (C) { <terminator> }`
//! followed by `BODY` in the same list.
//!
//! ## Why that is exactly equivalent
//!
//! The then-arm cannot fall through, so the only way to reach the statement
//! after the `if` is through the else-arm. Executing `BODY` there is therefore
//! the same as executing it inside the else block — same statements, same
//! order, same condition. Nothing is negated, so no condition has to be
//! invertible and no double-evaluation is introduced.
//!
//! ## Safety
//!
//! * The then-arm must be EXACTLY one terminator (`return`, `return expr`,
//!   `throw`, `break`, `continue`, `goto`). With any other statement in it,
//!   hoisting would run `BODY` on the terminated path too.
//! * An else arm that is a single `if` is left alone: the renderer already
//!   spells that `else if`, which ties the two tests together.
//! * Locals are unique per method, so lifting a block's declarations into the
//!   parent list cannot collide with a sibling.

use super::super::csharp::Stmt;
use super::children_mut;

/// Hoist every else arm whose then-arm terminates. Returns the number hoisted.
pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    hoist_list(stmts, &mut count);
    count
}

fn hoist_list(list: &mut Vec<Stmt>, count: &mut usize) {
    for stmt in list.iter_mut() {
        for child in children_mut(stmt) {
            hoist_list(child, count);
        }
    }
    let mut i = 0;
    while i < list.len() {
        if let Some(arm) = take_else_after_terminator(&mut list[i]) {
            list.splice(i + 1..i + 1, arm);
            *count += 1;
        }
        // Lifting an arm can place a guard directly under the SAME guard:
        // the nested `if (C) { … } else { if (C) { … } else { BODY } }`
        // form only looks non-redundant while the renderer keeps it nested.
        // The first guard already leaves when C holds, so the second test can
        // never decide anything — drop it.
        if i + 1 < list.len() && repeats_terminator_guard(&list[i], &list[i + 1]) {
            list.remove(i + 1);
            *count += 1;
            continue;
        }
        i += 1;
    }
}

/// Is `b` a guard on the same condition as `a`, with both arms leaving?
///
/// `a` has already left the method / loop when the condition holds, and when
/// it does not hold `b` tests the same condition and finds it false, so `b`
/// can never take its arm.
fn repeats_terminator_guard(a: &Stmt, b: &Stmt) -> bool {
    let (
        Stmt::If {
            cond: cond_a,
            then_body: then_a,
            else_body: else_a,
        },
        Stmt::If {
            cond: cond_b,
            then_body: then_b,
            else_body: else_b,
        },
    ) = (a, b)
    else {
        return false;
    };
    else_a.is_empty()
        && else_b.is_empty()
        && matches!(then_a.as_slice(), [t] if terminates(t))
        && matches!(then_b.as_slice(), [t] if terminates(t))
        && cond_a == cond_b
}

fn terminates(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Break | Stmt::Continue | Stmt::Goto(_)
    )
}

fn take_else_after_terminator(stmt: &mut Stmt) -> Option<Vec<Stmt>> {
    let Stmt::If {
        then_body,
        else_body,
        ..
    } = stmt
    else {
        return None;
    };
    if else_body.is_empty() || matches!(else_body.as_slice(), [Stmt::If { .. }]) {
        return None;
    }
    let terminates = matches!(
        then_body.as_slice(),
        [Stmt::Return(_)] | [Stmt::Throw(_)] | [Stmt::Break] | [Stmt::Continue] | [Stmt::Goto(_)]
    );
    if !terminates {
        return None;
    }
    Some(std::mem::take(else_body))
}

#[cfg(test)]
mod tests {
    use super::super::super::csharp::{CmpKind, Expr};
    use super::*;

    fn v(name: &str) -> Expr {
        Expr::Var(name.into())
    }

    fn ne_zero(name: &str) -> Expr {
        Expr::Cmp(CmpKind::Ne, Box::new(v(name)), Box::new(Expr::Int(0)))
    }

    fn assign(name: &str) -> Stmt {
        Stmt::Assign(name.into(), None, Expr::Int(1))
    }

    #[test]
    fn hoists_the_else_arm_of_a_return_guard() {
        let mut stmts = vec![
            Stmt::If {
                cond: ne_zero("a"),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![assign("x")],
            },
            assign("after"),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert_eq!(stmts.len(), 3);
        match &stmts[0] {
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                assert!(matches!(cond, Expr::Cmp(CmpKind::Ne, _, _)));
                assert!(matches!(then_body.as_slice(), [Stmt::Return(None)]));
                assert!(else_body.is_empty());
            }
            other => panic!("expected the guard to survive, got {other:?}"),
        }
        assert!(matches!(&stmts[1], Stmt::Assign(n, _, _) if n == "x"));
        assert!(matches!(&stmts[2], Stmt::Assign(n, _, _) if n == "after"));
    }

    #[test]
    fn hoists_a_break_guard_inside_a_loop() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::If {
                cond: ne_zero("a"),
                then_body: vec![Stmt::Break],
                else_body: vec![assign("x")],
            }],
        }];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::While { body, .. } => assert_eq!(body.len(), 2),
            other => panic!("expected a loop, got {other:?}"),
        }
    }

    #[test]
    fn keeps_a_then_arm_with_more_than_a_terminator() {
        let mut stmts = vec![Stmt::If {
            cond: ne_zero("a"),
            then_body: vec![assign("y"), Stmt::Return(None)],
            else_body: vec![assign("x")],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn hoists_a_goto_guard() {
        let mut stmts = vec![Stmt::If {
            cond: ne_zero("a"),
            then_body: vec![Stmt::Goto("L7".into())],
            else_body: vec![assign("x")],
        }];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[1], Stmt::Assign(n, _, _) if n == "x"));
    }

    #[test]
    fn drops_a_guard_that_repeats_the_one_above_it() {
        let mut stmts = vec![
            Stmt::If {
                cond: ne_zero("a"),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![],
            },
            Stmt::If {
                cond: ne_zero("a"),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![],
            },
            assign("x"),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn hoisting_exposes_and_then_drops_a_repeated_guard() {
        let mut stmts = vec![Stmt::If {
            cond: ne_zero("a"),
            then_body: vec![Stmt::Return(None)],
            else_body: vec![Stmt::If {
                cond: ne_zero("a"),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![assign("x")],
            }],
        }];
        apply(&mut stmts);
        assert_eq!(stmts.len(), 2, "{stmts:?}");
    }

    #[test]
    fn keeps_an_else_if_chain() {
        let mut stmts = vec![Stmt::If {
            cond: ne_zero("a"),
            then_body: vec![Stmt::Return(None)],
            else_body: vec![Stmt::If {
                cond: ne_zero("b"),
                then_body: vec![assign("x")],
                else_body: vec![],
            }],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }
}