//! If-else chain → `switch` reconstruction.
//!
//! ## The shape that actually occurs
//!
//! Measured on the x64 corpus: `switch (` appears **1** time in 2,454,023 lines.
//! The structurer emits if-else chains for jump-table dispatches:
//!
//! ```text
//! if (x == 0) { … }
//! else if (x == 1) { … }
//! else if (x == 2) { … }
//! else { … }
//! ```
//!
//! This pass detects those chains and converts them to proper `switch`
//! statements, which the renderer can emit as `switch (x) { case 0: … }`.
//!
//! ## The pattern
//!
//! The chain is encoded as nested `Stmt::If` where each `else_body` contains
//! exactly one `Stmt::If` (the next arm). The final arm's `else_body` is the
//! default case (or empty if there is no default).
//!
//! Each condition must be `Expr::Cmp(CmpKind::Eq, var, literal)` where:
//!   - `var` is the same `Expr::Var(name)` in every arm
//!   - `literal` is `Expr::Int(n)` with `n >= 0` (switch cases are unsigned)
//!   - All literals are distinct (duplicate cases would be a compile error)
//!
//! ## Why this is safe
//!
//! The rewrite is semantics-preserving because:
//!   1. The discriminant is a simple variable read — no side effects
//!   2. Each arm tests the SAME variable against a DISTINCT literal
//!   3. The chain is exhaustive over the tested values (the final `else`
//!      catches everything else, becoming the `default` case)
//!   4. No arm can fall through to the next (each body is independent)
//!
//! A failed match is a no-op: the if-else chain stays as-is.
//!
//! ## Minimum arm count
//!
//! At least 3 arms are required. A 2-arm `if (x == 0) … else …` is a normal
//! conditional, not a switch.

use super::{Expr, Stmt};
use crate::recovery::csharp::CmpKind;
use std::collections::HashSet;

/// Extract `(variable_name, literal_value)` from `var == N`.
///
/// Returns `None` if the condition is not an equality comparison between a
/// simple variable and a non-negative integer literal. The literal must be
/// non-negative because switch case labels are unsigned (`u64`).
pub fn extract_eq_literal(cond: &Expr) -> Option<(String, u64)> {
    let Expr::Cmp(CmpKind::Eq, lhs, rhs) = cond else {
        return None;
    };
    // Try both orderings: `var == lit` and `lit == var`.
    match (lhs.as_ref(), rhs.as_ref()) {
        (Expr::Var(name), Expr::Int(n)) if *n >= 0 => Some((name.clone(), *n as u64)),
        (Expr::Int(n), Expr::Var(name)) if *n >= 0 => Some((name.clone(), *n as u64)),
        _ => None,
    }
}

/// Check whether `stmt` is the head of an if-else chain with at least `min_arms`
/// arms, all testing the same variable against distinct non-negative integer
/// literals.
pub fn is_if_else_chain(stmt: &Stmt, min_arms: usize) -> bool {
    let arms = collect_if_arms(stmt);
    if arms.len() < min_arms {
        return false;
    }
    // All arms must test the same variable.
    let first_var = arms[0].0.clone();
    if !arms.iter().all(|(v, _, _)| *v == first_var) {
        return false;
    }
    // All literals must be distinct.
    let mut seen = HashSet::new();
    for (_, lit, _) in &arms {
        if !seen.insert(*lit) {
            return false;
        }
    }
    true
}

/// Collect all `(variable_name, literal, then_body)` triples from the chain.
///
/// Walks the nested if-else structure, extracting each arm's condition and
/// body. Stops when the else_body is not a single `Stmt::If` — that final
/// else_body is the default case (returned separately).
pub fn collect_if_arms(stmt: &Stmt) -> Vec<(String, u64, Vec<Stmt>)> {
    let mut arms = Vec::new();
    let mut current = stmt;
    loop {
        let Stmt::If {
            cond,
            then_body,
            else_body,
        } = current
        else {
            break;
        };
        let Some((var, lit)) = extract_eq_literal(cond) else {
            break;
        };
        arms.push((var, lit, then_body.clone()));
        // The chain continues if else_body is exactly one If statement.
        if else_body.len() == 1 {
            if let Stmt::If { .. } = &else_body[0] {
                current = &else_body[0];
                continue;
            }
        }
        break;
    }
    arms
}

/// Extract the default-case body from the tail of the chain.
///
/// After walking all the if-arms, the remaining else_body (which is NOT another
/// if) becomes the default case.
fn extract_tail(stmt: &Stmt) -> Vec<Stmt> {
    let mut current = stmt;
    loop {
        let Stmt::If {
            cond,
            else_body,
            ..
        } = current
        else {
            return Vec::new();
        };
        // If this condition is not a valid eq-literal, we've gone past the chain.
        if extract_eq_literal(cond).is_none() {
            return Vec::new();
        }
        // If else_body is a single If, the chain continues — descend.
        if else_body.len() == 1 {
            if let Stmt::If { .. } = &else_body[0] {
                current = &else_body[0];
                continue;
            }
        }
        // else_body is the default case (may be empty).
        return else_body.clone();
    }
}

/// Plan a switch rewrite for the statement at position `idx` in `list`.
///
/// Returns `None` if the statement is not a convertible if-else chain.
fn plan_switch(list: &[Stmt], idx: usize) -> Option<Stmt> {
    let stmt = &list[idx];
    let arms = collect_if_arms(stmt);
    // Require at least 3 arms (a 2-arm if/else is a conditional, not a switch).
    if arms.len() < 3 {
        return None;
    }
    // All arms must test the same variable.
    let disc = arms[0].0.clone();
    if !arms.iter().all(|(v, _, _)| *v == disc) {
        return None;
    }
    // All literals must be distinct non-negative integers.
    let mut seen = HashSet::new();
    let mut cases = Vec::with_capacity(arms.len());
    for (var, lit, body) in &arms {
        if *var != disc {
            return None;
        }
        if !seen.insert(*lit) {
            // Duplicate literal — not a valid switch.
            return None;
        }
        cases.push((*lit, body.clone()));
    }
    // The final else body is the DEFAULT arm, not a shared tail. It used to be
    // pushed into `tail`, which the printer emits after the closing brace as
    // bare labelled code — so the default body was rendered outside the switch
    // where nothing could reach it.
    let tail = extract_tail(stmt);
    Some(Stmt::Switch {
        disc,
        cases,
        tail: Vec::new(),
        default: (!tail.is_empty()).then_some(tail),
    })
}

/// Run the switch reconstruction pass over one method body.
///
/// Uses pre-order traversal (outside-in): converts chains at the current level
/// first, then recurses into the resulting switch bodies. This ensures the
/// outermost chain is converted before any inner chains, which is necessary
/// because converting an inner chain first would break the outer chain's
/// pattern (the outer chain expects nested `If` statements, not `Switch`).
///
/// Returns the number of rewrites applied.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    apply_inner(stmts)
}

/// Pre-order traversal: convert chains at this level, then recurse.
fn apply_inner(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    // Convert all chains at this level.
    for i in 0..stmts.len() {
        if let Some(switch_stmt) = plan_switch(stmts, i) {
            stmts[i] = switch_stmt;
            n += 1;
        }
    }
    // Recurse into every statement's nested bodies (including newly created
    // switches). This is safe: the recursion only descends, never re-visits
    // the current level.
    for s in stmts.iter_mut() {
        for child_list in super::children_mut(s) {
            n += apply_inner(child_list);
        }
    }
    n
}

/// Public entry point matching the pass signature.
pub fn apply_switch_recon(stmts: &mut Vec<Stmt>) -> usize {
    apply(stmts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::CmpKind;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn eq_lit(name: &str, val: i64) -> Expr {
        Expr::Cmp(CmpKind::Eq, Box::new(var(name)), Box::new(Expr::Int(val)))
    }

    fn make_if(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond,
            then_body,
            else_body,
        }
    }

    fn body_stmt() -> Stmt {
        Stmt::Eval(var("do_work"))
    }

    /// Build a 3-arm if-else chain: `if (x==0) … else if (x==1) … else if (x==2) … else …`
    fn three_arm_chain() -> Stmt {
        make_if(
            eq_lit("x", 0),
            vec![body_stmt()],
            vec![make_if(
                eq_lit("x", 1),
                vec![body_stmt()],
                vec![make_if(
                    eq_lit("x", 2),
                    vec![body_stmt()],
                    vec![body_stmt()], // default case
                )],
            )],
        )
    }

    /// A 3-arm chain converts to a switch with 3 cases and a tail.
    #[test]
    fn three_arm_chain_converts_to_switch() {
        let mut stmts = vec![three_arm_chain()];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::Switch {
                disc,
                cases,
                default,
                ..
            } => {
                assert_eq!(disc, "x");
                assert_eq!(cases.len(), 3);
                assert_eq!(cases[0].0, 0);
                assert_eq!(cases[1].0, 1);
                assert_eq!(cases[2].0, 2);
                // The default case body (one Eval statement).
                assert_eq!(default.as_ref().map(|d| d.len()), Some(1));
            }
            other => panic!("expected Switch, got {other:?}"),
        }
    }

    /// A 5-arm chain converts correctly.
    #[test]
    fn five_arm_chain_converts_correctly() {
        // Build: if(x==0) … else if(x==1) … else if(x==2) … else if(x==3) … else if(x==4) … else …
        let mut stmt = make_if(eq_lit("x", 4), vec![body_stmt()], vec![body_stmt()]);
        for i in (0..4).rev() {
            stmt = make_if(eq_lit("x", i), vec![body_stmt()], vec![stmt]);
        }
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::Switch {
                disc,
                cases,
                default,
                ..
            } => {
                assert_eq!(disc, "x");
                assert_eq!(cases.len(), 5);
                for (i, (val, _)) in cases.iter().enumerate() {
                    assert_eq!(*val, i as u64);
                }
                assert_eq!(default.as_ref().map(|d| d.len()), Some(1));
            }
            other => panic!("expected Switch, got {other:?}"),
        }
    }

    /// A 2-arm chain is NOT converted (it's a conditional, not a switch).
    #[test]
    fn two_arm_chain_is_not_converted() {
        let stmt = make_if(
            eq_lit("x", 0),
            vec![body_stmt()],
            vec![make_if(eq_lit("x", 1), vec![body_stmt()], vec![body_stmt()])],
        );
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    /// Mixed variables are NOT converted.
    #[test]
    fn mixed_variables_are_not_converted() {
        let stmt = make_if(
            eq_lit("x", 0),
            vec![body_stmt()],
            vec![make_if(
                eq_lit("y", 1), // different variable!
                vec![body_stmt()],
                vec![make_if(eq_lit("x", 2), vec![body_stmt()], vec![])],
            )],
        );
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    /// Non-literal conditions are NOT converted.
    #[test]
    fn non_literal_conditions_are_not_converted() {
        // x == y (not a literal)
        let cond = Expr::Cmp(CmpKind::Eq, Box::new(var("x")), Box::new(var("y")));
        let stmt = make_if(
            cond,
            vec![body_stmt()],
            vec![make_if(
                eq_lit("x", 1),
                vec![body_stmt()],
                vec![make_if(eq_lit("x", 2), vec![body_stmt()], vec![])],
            )],
        );
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// The tail (default case) is handled correctly — including empty tail.
    #[test]
    fn empty_tail_is_handled() {
        // Chain with no final else body.
        let stmt = make_if(
            eq_lit("x", 0),
            vec![body_stmt()],
            vec![make_if(
                eq_lit("x", 1),
                vec![body_stmt()],
                vec![make_if(eq_lit("x", 2), vec![body_stmt()], vec![])], // no default
            )],
        );
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::Switch { default, .. } => {
                assert!(default.is_none(), "no default case means no default arm");
            }
            other => panic!("expected Switch, got {other:?}"),
        }
    }

    /// Non-negative check: negative literals are rejected.
    #[test]
    fn negative_literals_are_rejected() {
        let stmt = make_if(
            eq_lit("x", -1),
            vec![body_stmt()],
            vec![make_if(
                eq_lit("x", 0),
                vec![body_stmt()],
                vec![make_if(eq_lit("x", 1), vec![body_stmt()], vec![])],
            )],
        );
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Duplicate literals are rejected.
    #[test]
    fn duplicate_literals_are_rejected() {
        let stmt = make_if(
            eq_lit("x", 0),
            vec![body_stmt()],
            vec![make_if(
                eq_lit("x", 0), // duplicate!
                vec![body_stmt()],
                vec![make_if(eq_lit("x", 1), vec![body_stmt()], vec![])],
            )],
        );
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Nested switches: an if-else chain inside another if-else chain's arm.
    #[test]
    fn nested_switches_are_converted() {
        // Outer chain on `x`, with one arm containing an inner chain on `y`.
        let inner_chain = make_if(
            eq_lit("y", 0),
            vec![body_stmt()],
            vec![make_if(
                eq_lit("y", 1),
                vec![body_stmt()],
                vec![make_if(eq_lit("y", 2), vec![body_stmt()], vec![])],
            )],
        );
        let outer = make_if(
            eq_lit("x", 0),
            vec![inner_chain],
            vec![make_if(
                eq_lit("x", 1),
                vec![body_stmt()],
                vec![make_if(eq_lit("x", 2), vec![body_stmt()], vec![])],
            )],
        );
        let mut stmts = vec![outer];
        // Both the outer and inner chains should convert.
        assert_eq!(apply(&mut stmts), 2);
        // The outer should be a Switch.
        match &stmts[0] {
            Stmt::Switch { disc, cases, .. } => {
                assert_eq!(disc, "x");
                assert_eq!(cases.len(), 3);
                // The first case's body should contain the inner Switch.
                match &cases[0].1[0] {
                    Stmt::Switch { disc: inner_disc, cases: inner_cases, .. } => {
                        assert_eq!(inner_disc, "y");
                        assert_eq!(inner_cases.len(), 3);
                    }
                    other => panic!("expected inner Switch, got {other:?}"),
                }
            }
            other => panic!("expected outer Switch, got {other:?}"),
        }
    }

    /// A non-Eq comparison (e.g., `x < 5`) is not converted.
    #[test]
    fn non_eq_comparison_is_not_converted() {
        let cond = Expr::Cmp(CmpKind::Lt, Box::new(var("x")), Box::new(Expr::Int(0)));
        let stmt = make_if(
            cond,
            vec![body_stmt()],
            vec![make_if(
                eq_lit("x", 1),
                vec![body_stmt()],
                vec![make_if(eq_lit("x", 2), vec![body_stmt()], vec![])],
            )],
        );
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 0);
    }
}
