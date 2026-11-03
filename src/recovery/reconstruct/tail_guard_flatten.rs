//! Flatten the early-exit cascade at a void method's tail.
//!
//! `branch_repair` builds every test as a two-armed branch, so a flat sequence
//! of early exits comes out as a stair of nested guards:
//!
//! ```csharp
//! if (a != null)
//! {
//!     use(a);
//!     if (b != null)
//!     {
//!         use(b);
//!         if (c != null) { use(c); }
//!     }
//! }
//! ```
//!
//! At the TAIL of a void method that is exactly a flat guard list:
//!
//! ```csharp
//! if (a == null) return;
//! use(a);
//! if (b == null) return;
//! use(b);
//! if (c == null) return;
//! use(c);
//! ```
//!
//! When the test fails in the nested form, control reaches the end of the
//! method; when it fails in the flat form, `return;` does. Both forms run the
//! same statements on the same conditions.
//!
//! ## Safety
//!
//! * Only void methods: falling off the end of a value-returning method is
//!   not a `return`, and no value may be invented.
//! * Only TAIL position: the method's top-level list, and the arms of an `if`
//!   that itself sits in tail position. Loops, `switch`, `try`, `using` and
//!   `lock` continue after they finish, so nothing inside them is tail.
//! * The arm is moved, not rewritten: no statement changes order, and the
//!   only added statement is the `return;` that the original fall-off already
//!   performed.

use super::super::csharp::{Expr, Stmt};

pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    flatten_tail(stmts, &mut n);
    n
}

fn negate(e: Expr) -> Expr {
    match e {
        Expr::Cmp(op, a, b) => Expr::Cmp(op.invert(), a, b),
        Expr::Not(x) => *x,
        other => Expr::Not(Box::new(other)),
    }
}

fn flatten_tail(list: &mut Vec<Stmt>, n: &mut usize) {
    loop {
        // A trailing `return;` is the same fall-off the flatten introduces;
        // it is transparent for finding the tail statement.
        let mut tail_at = match list.len().checked_sub(1) {
            Some(i) => i,
            None => return,
        };
        if matches!(&list[tail_at], Stmt::Return(None)) {
            match tail_at.checked_sub(1) {
                Some(i) => tail_at = i,
                None => return,
            }
        }
        let (guard, body) = match &mut list[tail_at] {
            Stmt::If {
                cond,
                then_body,
                else_body,
            } if else_body.is_empty() && !then_body.is_empty() => {
                // `if (A) { B }` at the tail → `if (!A) return; B`.
                (negate(cond.clone()), std::mem::take(then_body))
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } if then_body.is_empty() && !else_body.is_empty() => {
                // `if (A) {} else { B }` at the tail → `if (A) return; B`.
                (cond.clone(), std::mem::take(else_body))
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } if !else_body.is_empty() => {
                // A true if/else at the tail: both arms end at the method end.
                flatten_tail(then_body, n);
                flatten_tail(else_body, n);
                return;
            }
            _ => return,
        };
        list[tail_at] = Stmt::If {
            cond: guard,
            then_body: vec![Stmt::Return(None)],
            else_body: Vec::new(),
        };
        list.splice(tail_at + 1..tail_at + 1, body);
        *n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::csharp::{CmpKind, Expr};
    use super::*;

    fn null_cmp(name: &str, eq: bool) -> Expr {
        Expr::Cmp(
            if eq { CmpKind::Eq } else { CmpKind::Ne },
            Box::new(Expr::Var(name.into())),
            Box::new(Expr::Null),
        )
    }

    fn use_stmt(name: &str) -> Stmt {
        Stmt::Eval(Expr::Var(name.into()))
    }

    #[test]
    fn a_tail_guard_chain_is_flattened() {
        let mut s = vec![Stmt::If {
            cond: null_cmp("a", false),
            then_body: vec![
                use_stmt("a"),
                Stmt::If {
                    cond: null_cmp("b", false),
                    then_body: vec![use_stmt("b")],
                    else_body: vec![],
                },
            ],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut s), 2);
        assert_eq!(s.len(), 4, "{s:#?}");
        assert!(matches!(&s[0], Stmt::If { then_body, .. }
            if matches!(&then_body[0], Stmt::Return(None))));
    }

    #[test]
    fn an_empty_then_spelling_flattens_without_negation() {
        let mut s = vec![Stmt::If {
            cond: null_cmp("a", true),
            then_body: vec![],
            else_body: vec![use_stmt("a")],
        }];
        assert_eq!(apply(&mut s), 1);
        match &s[0] {
            Stmt::If { cond, then_body, .. } => {
                assert!(matches!(cond, Expr::Cmp(CmpKind::Eq, ..)), "{cond:?}");
                assert!(matches!(&then_body[0], Stmt::Return(None)));
            }
            other => panic!("expected a guard, got {other:?}"),
        }
    }

    #[test]
    fn a_non_tail_if_is_left_alone() {
        let mut s = vec![
            Stmt::If {
                cond: null_cmp("a", false),
                then_body: vec![use_stmt("a")],
                else_body: vec![],
            },
            use_stmt("after"),
        ];
        assert_eq!(apply(&mut s), 0);
        assert_eq!(s.len(), 2);
    }
}
