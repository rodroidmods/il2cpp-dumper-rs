//! Merge a guard whose whole body is another guard.
//!
//! ## Why this pass exists
//!
//! IL2CPP tests every reference it dereferences, and `branch_repair` rebuilds
//! those tests as blocks. When two of them nest, the inner one is the outer
//! block's ONLY statement:
//!
//! ```csharp
//! if (this._listBoxes != null)
//! {
//!     if (this._setting != null)
//!     {
//!         ...
//!     }
//! }
//! ```
//!
//! R90 had **8,952** of these. Each costs a `{`, an `if`, a `}` and two
//! indentation levels, and reads as nesting the source never had.
//!
//! ## The rule
//!
//! `if (A) { if (B) { BODY } }` (neither `if` carrying an `else`) becomes
//! `if (A && B) { BODY }`.
//!
//! ## Why `&&` is exactly equivalent
//!
//! `A && B` evaluates `A`, then evaluates `B` only when `A` was true, and the
//! result decides the same body. That is the nested form's evaluation order
//! statement for statement, so no side effect moves and none is added — no
//! purity precondition is needed. Short-circuiting is what makes this true;
//! `&` would evaluate `B` unconditionally and would NOT be equivalent.
//!
//! ## Safety
//!
//! * The outer body must hold exactly ONE statement, and it must be the inner
//!   `if`. A sibling statement after the inner guard runs unconditionally, so
//!   folding it into the condition would skip it.
//! * Neither `if` may have an `else`: with an `else` the outer condition still
//!   selects an arm, and `A && B` cannot express that.
//! * No label may live anywhere inside the outer `if`. A label is a jump
//!   target, so another edge can enter the inner block without passing `A`;
//!   the fold would then apply a test that was never made on that path.

use super::super::csharp::{BinOp, Expr, Stmt};
use super::children_mut;

/// Merge every single-statement guard chain. Returns the number merged.
pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    // The branch structures arrive in the EMPTY-THEN spelling
    // (`if (X) {} else { … }`), which the renderer inverts only at print
    // time. As a tree it is the complement of a guard, so the merge below
    // cannot see the chain. Normalise bottom-up first; the inversion is
    // exact and moves nothing across a jump target.
    let mut flipped = 0;
    flip_empty_then(stmts, &mut flipped);
    let mut count = 0;
    merge_list(stmts, &mut count);
    count
}

/// `if (A) {} else { B }` → `if (!A) { B }` at every depth (bottom-up).
fn flip_empty_then(list: &mut [Stmt], count: &mut usize) {
    for stmt in list.iter_mut() {
        for child in children_mut(stmt) {
            flip_empty_then(child, count);
        }
        let Stmt::If {
            cond,
            then_body,
            else_body,
        } = stmt
        else {
            continue;
        };
        if !then_body.is_empty() || else_body.is_empty() {
            continue;
        }
        let body = std::mem::take(else_body);
        *then_body = body;
        *cond = negate(std::mem::replace(cond, Expr::Bool(true)));
        *count += 1;
    }
}

fn negate(e: Expr) -> Expr {
    match e {
        Expr::Cmp(op, a, b) => Expr::Cmp(op.invert(), a, b),
        Expr::Not(x) => *x,
        other => Expr::Not(Box::new(other)),
    }
}

fn merge_list(list: &mut Vec<Stmt>, count: &mut usize) {
    // Bottom-up: an inner chain merges before the guard around it, so a
    // three-level nest collapses in one visit.
    for stmt in list.iter_mut() {
        for child in children_mut(stmt) {
            merge_list(child, count);
        }
    }
    let mut i = 0;
    while i < list.len() {
        let Some(merged) = merge_one(&list[i]) else {
            i += 1;
            continue;
        };
        list[i] = merged;
        *count += 1;
        // The merged condition can expose another level (`(a && b) && c`).
    }
}

fn merge_one(stmt: &Stmt) -> Option<Stmt> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };
    if !else_body.is_empty() && !else_body.iter().all(is_trivia) {
        return None;
    }
    // Comments/attributes between the guards carry no control flow; they ride
    // along into the merged body. Any other sibling statement runs
    // unconditionally, so it blocks the merge.
    let mut prefix: Vec<Stmt> = Vec::new();
    let mut inner: Option<&Stmt> = None;
    for s in then_body {
        match s {
            // Comments/attributes at ANY position carry no control flow; they
            // move to the head of the merged body.
            Stmt::Comment(_) | Stmt::Attribute { .. } => prefix.push(s.clone()),
            Stmt::If { .. } if inner.is_none() => inner = Some(s),
            _ => return None,
        }
    }
    let Some(Stmt::If {
        cond: inner_cond,
        then_body: inner_body,
        else_body: inner_else,
    }) = inner
    else {
        return None;
    };
    if !inner_else.is_empty() && !inner_else.iter().all(is_trivia) {
        return None;
    }
    // A label inside the INNER body is safe: a goto into it bypassed both
    // tests before the merge and bypasses the combined test after it. A label
    // among the outer-body siblings is NOT: the goto lands before the inner
    // `if` today and would land outside the merged guard tomorrow.
    if prefix.iter().any(|s| matches!(s, Stmt::Label(_))) || prefix.iter().any(has_label) {
        return None;
    }
    let mut merged_body = prefix;
    merged_body.extend(inner_body.iter().cloned());
    Some(Stmt::If {
        cond: Expr::Bin(BinOp::AndAlso, Box::new(cond.clone()), Box::new(inner_cond.clone())),
        then_body: merged_body,
        else_body: Vec::new(),
    })
}

fn is_trivia(s: &Stmt) -> bool {
    matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. })
}

fn has_label(stmt: &Stmt) -> bool {
    if matches!(stmt, Stmt::Label(_)) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(stmt, &mut kids);
    kids.iter().any(|k| k.iter().any(has_label))
}

#[cfg(test)]
mod tests {
    use super::super::super::csharp::CmpKind;
    use super::*;

    /// The branch spelling the renderer inverts at print time
    /// (`if (a) {} else { if (b) {} else { … } }`) must collapse through the
    /// same merge as the positive spelling.
    #[test]
    fn an_empty_then_chain_merges_into_one_guard() {
        let cmp = |n: i64| {
            Expr::Cmp(
                CmpKind::Ne,
                Box::new(v("x")),
                Box::new(Expr::Int(n)),
            )
        };
        let mut s = vec![Stmt::If {
            cond: cmp(0),
            then_body: vec![],
            else_body: vec![Stmt::If {
                cond: cmp(1),
                then_body: vec![],
                else_body: vec![Stmt::Return(Some(Expr::Int(0)))],
            }],
        }];
        assert_eq!(apply(&mut s), 1);
        match &s[0] {
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                assert!(matches!(cond, Expr::Bin(BinOp::AndAlso, ..)), "{cond:?}");
                assert!(else_body.is_empty());
                assert!(matches!(&then_body[0], Stmt::Return(Some(_))));
            }
            other => panic!("expected one merged guard, got {other:?}"),
        }
    }

    #[test]
    fn comments_between_the_guards_do_not_block_the_merge() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![
                Stmt::Comment("recovery note".into()),
                Stmt::If {
                    cond: Expr::Bool(true),
                    then_body: vec![Stmt::Eval(v("x"))],
                    else_body: vec![],
                },
            ],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Comment(_)));
                assert!(matches!(&then_body[1], Stmt::Eval(_)));
            }
            other => panic!("expected merged if, got {other:?}"),
        }
    }

    #[test]
    fn a_label_inside_the_inner_body_does_not_block_the_merge() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Label("L1".into()), Stmt::Eval(v("x"))],
                else_body: vec![],
            }],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Label(l) if l == "L1"));
            }
            other => panic!("expected merged if, got {other:?}"),
        }
    }

    #[test]
    fn comment_only_else_arms_do_not_block_the_merge() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(v("x"))],
                else_body: vec![Stmt::Comment("inner".into())],
            }],
            else_body: vec![Stmt::Comment("outer".into())],
        }];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::If { then_body, else_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Eval(_)));
                assert!(else_body.is_empty());
            }
            other => panic!("expected merged if, got {other:?}"),
        }
    }

    #[test]
    fn a_label_between_the_guards_blocks_the_merge() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![
                Stmt::Label("L1".into()),
                Stmt::If {
                    cond: Expr::Bool(true),
                    then_body: vec![Stmt::Eval(v("x"))],
                    else_body: vec![],
                },
            ],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    fn v(name: &str) -> Expr {
        Expr::Var(name.into())
    }

    fn ne_zero(name: &str) -> Expr {
        Expr::Cmp(CmpKind::Ne, Box::new(v(name)), Box::new(Expr::Int(0)))
    }

    fn assign(name: &str) -> Stmt {
        Stmt::Assign(name.into(), None, Expr::Int(1))
    }

    fn if_stmt(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond,
            then_body,
            else_body,
        }
    }

    #[test]
    fn merges_a_guard_whose_body_is_one_guard() {
        let mut stmts = vec![if_stmt(
            ne_zero("a"),
            vec![if_stmt(ne_zero("b"), vec![assign("x")], vec![])],
            vec![],
        )];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::If { cond, then_body, .. } => {
                assert!(matches!(cond, Expr::Bin(BinOp::AndAlso, _, _)));
                assert_eq!(then_body.len(), 1);
            }
            other => panic!("expected one if, got {other:?}"),
        }
    }

    #[test]
    fn merges_three_levels_in_one_pass() {
        let inner = if_stmt(ne_zero("c"), vec![assign("x")], vec![]);
        let mid = if_stmt(ne_zero("b"), vec![inner], vec![]);
        let mut stmts = vec![if_stmt(ne_zero("a"), vec![mid], vec![])];
        assert_eq!(apply(&mut stmts), 2);
        match &stmts[0] {
            Stmt::If { cond, then_body, .. } => {
                assert!(matches!(cond, Expr::Bin(BinOp::AndAlso, _, _)));
                assert!(matches!(then_body.as_slice(), [Stmt::Assign(n, _, _)] if n == "x"));
            }
            other => panic!("expected one if, got {other:?}"),
        }
    }

    #[test]
    fn keeps_an_inner_guard_with_an_else() {
        let mut stmts = vec![if_stmt(
            ne_zero("a"),
            vec![if_stmt(ne_zero("b"), vec![assign("x")], vec![assign("y")])],
            vec![],
        )];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn keeps_a_guard_with_a_sibling_statement() {
        let mut stmts = vec![if_stmt(
            ne_zero("a"),
            vec![assign("pre"), if_stmt(ne_zero("b"), vec![assign("x")], vec![])],
            vec![],
        )];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn a_trailing_comment_inside_the_outer_body_does_not_block_the_merge() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![
                Stmt::If {
                    cond: Expr::Bool(true),
                    then_body: vec![Stmt::Eval(v("x"))],
                    else_body: vec![],
                },
                Stmt::Comment("after".into()),
            ],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 1);
    }

    #[test]
    fn a_label_inside_the_inner_body_merges() {
        // A goto into L4 skipped BOTH guards before the merge and skips the
        // combined one after it: the reachable executions are identical.
        let mut stmts = vec![if_stmt(
            ne_zero("a"),
            vec![if_stmt(
                ne_zero("b"),
                vec![Stmt::Label("L4".into()), assign("x")],
                vec![],
            )],
            vec![],
        )];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Label(l) if l == "L4"));
            }
            other => panic!("expected merged if, got {other:?}"),
        }
    }

    #[test]
    fn merges_inside_a_loop_body() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![if_stmt(
                ne_zero("a"),
                vec![if_stmt(ne_zero("b"), vec![assign("x")], vec![])],
                vec![],
            )],
        }];
        assert_eq!(apply(&mut stmts), 1);
    }
}