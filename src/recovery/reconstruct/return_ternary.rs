//! Fold `if (C) return E; return F;` into `return C ? E : F;`.
//!
//! Two-tail returns are how the binary spells most boolean getters and
//! tie-breaking comparisons. The `if` costs three lines (plus indentation) for
//! one selection; the conditional operator expresses exactly the same
//! evaluation (C once, then one arm) in one statement.
//!
//! The fold is exact: the arm bodies are single `return` statements, the `if`
//! has no `else`, and the fall-through return is the very next statement in
//! the same list. No label, comment or other statement can sit between, so no
//! path can observe a difference — including calls in C/E/F, whose evaluation
//! order and count are unchanged.
//!
//! A chain `if (a) return 1; if (b) return 2; return 3;` folds bottom-up into
//! `return a ? 1 : b ? 2 : 3;` over repeated passes.

use super::super::csharp::{Expr, Stmt};

pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    fold_list(stmts, &mut n);
    n
}

fn fold_list(list: &mut Vec<Stmt>, n: &mut usize) {
    // Children first: an arm that is itself a folded chain is already one
    // statement when the guard around it is considered.
    for s in list.iter_mut() {
        for k in super::children_mut(s) {
            fold_list(k, n);
        }
    }
    loop {
        let mut folded = 0;
        let mut i = 0;
        while i + 1 < list.len() {
            let fold = match (&list[i], &list[i + 1]) {
                (
                    Stmt::If {
                        cond,
                        then_body,
                        else_body,
                    },
                    Stmt::Return(Some(f)),
                ) if else_body.is_empty() && then_body.len() == 1 => match &then_body[0] {
                    Stmt::Return(Some(e)) => {
                        Some(Expr::Cond(Box::new(cond.clone()), Box::new(e.clone()), Box::new(f.clone())))
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some(cond) = fold {
                list[i] = Stmt::Return(Some(cond));
                list.remove(i + 1);
                *n += 1;
                folded += 1;
                // The new Return may now pair with the statement that follows.
                continue;
            }
            i += 1;
        }
        if folded == 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::csharp::{CmpKind, Expr};
    use super::*;

    fn ret(e: Expr) -> Stmt {
        Stmt::Return(Some(e))
    }

    fn cond_then_ret(cond: Expr, then_ret: Stmt) -> Stmt {
        Stmt::If {
            cond,
            then_body: vec![then_ret],
            else_body: vec![],
        }
    }

    #[test]
    fn a_two_tail_return_becomes_a_ternary() {
        let mut s = vec![
            cond_then_ret(Expr::Var("c".into()), ret(Expr::Int(1))),
            ret(Expr::Int(0)),
        ];
        assert_eq!(apply(&mut s), 1);
        assert_eq!(s.len(), 1);
        assert!(matches!(&s[0], Stmt::Return(Some(Expr::Cond(..)))));
    }

    #[test]
    fn a_chain_folds_into_nested_ternaries() {
        let mut s = vec![
            cond_then_ret(Expr::Var("a".into()), ret(Expr::Int(1))),
            cond_then_ret(Expr::Var("b".into()), ret(Expr::Int(2))),
            ret(Expr::Int(3)),
        ];
        assert_eq!(apply(&mut s), 2);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn a_body_with_more_than_the_return_is_kept() {
        let mut s = vec![
            Stmt::If {
                cond: Expr::Var("c".into()),
                then_body: vec![
                    Stmt::Eval(Expr::Var("x".into())),
                    ret(Expr::Int(1)),
                ],
                else_body: vec![],
            },
            ret(Expr::Int(0)),
        ];
        assert_eq!(apply(&mut s), 0);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn an_else_arm_is_kept() {
        let mut s = vec![
            Stmt::If {
                cond: Expr::Var("c".into()),
                then_body: vec![ret(Expr::Int(1))],
                else_body: vec![ret(Expr::Int(2))],
            },
            ret(Expr::Int(0)),
        ];
        assert_eq!(apply(&mut s), 0);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn a_non_return_fallthrough_is_kept() {
        let mut s = vec![
            cond_then_ret(Expr::Var("c".into()), ret(Expr::Int(1))),
            Stmt::Eval(Expr::Var("x".into())),
        ];
        assert_eq!(apply(&mut s), 0);
    }

    #[test]
    fn a_comparison_condition_still_folds() {
        let cond = Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::Var("a".into())),
            Box::new(Expr::Int(0)),
        );
        let mut s = vec![
            cond_then_ret(cond.clone(), ret(Expr::Int(0))),
            ret(Expr::Int(1)),
        ];
        assert_eq!(apply(&mut s), 1);
        match &s[0] {
            Stmt::Return(Some(Expr::Cond(c, _, _))) => assert_eq!(format!("{c:?}"), format!("{cond:?}")),
            other => panic!("expected a ternary return, got {other:?}"),
        }
    }
}
