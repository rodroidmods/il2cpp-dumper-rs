//! `E; return;` inside a VALUE-returning method is `return E;`.
//!
//! C# forbids falling off the end of a value-returning method, so a bare
//! expression statement sitting immediately before an operand-less `return`
//! has exactly one meaning: that expression WAS the return value. The pair is
//! what survives when the walker loses the link between a call's result slot
//! and the return register:
//!
//! * `Camera.get_current` loads the resolved `get_currentInternal` pointer and
//!   `blr`s it; the x0 result never reaches the `Return`, so the body printed
//!   `Camera.currentInternal; return;`.
//! * a pass folded a getter call into its property read (`EventBoss.timeToNext;`
//!   `return;`), leaving the read unbound.
//!
//! Void methods are refused by the CALLER (`Foo(); return;` is idiomatic real
//! C# there), and a `return` that already carries an expression is left alone.

use crate::recovery::csharp::Stmt;

pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    walk(stmts, &mut n);
    n
}

fn walk(list: &mut Vec<Stmt>, n: &mut usize) {
    let mut i = 0;
    while i < list.len() {
        let folded = match (&list[i], list.get(i + 1)) {
            (Stmt::Eval(e), Some(Stmt::Return(None))) => Some(e.clone()),
            _ => None,
        };
        if let Some(e) = folded {
            list.splice(i..=i + 1, [Stmt::Return(Some(e))]);
            *n += 1;
            continue;
        }
        for k in super::children_mut(&mut list[i]) {
            walk(k, n);
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::Expr;

    fn call() -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: None,
            name: "Camera.currentInternal".into(),
            args: vec![],
        })
    }

    #[test]
    fn a_bare_expression_before_an_empty_return_becomes_the_value() {
        let mut s = vec![call(), Stmt::Return(None)];
        assert_eq!(apply(&mut s), 1);
        assert_eq!(s.len(), 1);
        match &s[0] {
            Stmt::Return(Some(Expr::Call { name, .. })) => assert_eq!(name, "Camera.currentInternal"),
            other => panic!("expected a valued return, got {other:?}"),
        }
    }

    /// A `return` that already has its expression is not ours to touch, and a
    /// non-final empty return is not the fall-off-the-end case.
    #[test]
    fn a_return_that_already_carries_a_value_is_left_alone() {
        let mut s = vec![call(), Stmt::Return(Some(Expr::Var("v".into())))];
        assert_eq!(apply(&mut s), 0);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn the_fold_reaches_nested_lists() {
        let mut s = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![call(), Stmt::Return(None)],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut s), 1);
        match &s[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert!(matches!(&then_body[0], Stmt::Return(Some(_))));
            }
            other => panic!("expected the if to survive, got {other:?}"),
        }
    }

    /// Two statements before the return: only the immediately preceding one is
    /// a candidate, and the fold must not swallow the earlier statement.
    #[test]
    fn an_earlier_statement_is_not_absorbed() {
        let mut s = vec![
            Stmt::Eval(Expr::Var("first".into())),
            call(),
            Stmt::Return(None),
        ];
        assert_eq!(apply(&mut s), 1);
        assert_eq!(s.len(), 2);
        assert!(matches!(&s[1], Stmt::Return(Some(_))));
    }
}
