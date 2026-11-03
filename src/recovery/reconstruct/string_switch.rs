//! If-else chain -> string `switch` reconstruction.
//!
//! ## The pattern
//!
//! C# `switch` statements on strings compile to if-else chains with
//! `String.Equals` calls:
//!
//! ```text
//! if (s.Equals("foo")) { ... }
//! else if (s.Equals("bar")) { ... }
//! else if (s.Equals("baz")) { ... }
//! else { ... }
//! ```
//!
//! This pass detects those chains and converts them to `Stmt::StringSwitch`:
//!
//! ```text
//! switch (s) {
//!     case "foo": ... break;
//!     case "bar": ... break;
//!     case "baz": ... break;
//!     default: ... break;
//! }
//! ```
//!
//! ## Conditions
//!
//! Each arm's condition must be one of:
//!   - Instance form: `Call { recv: Some(Var("s")), name: "Equals", args: [Str("literal")] }`
//!   - Static form: `Call { recv: None, name: "String.Equals", args: [Var("s"), Str("literal")] }`
//!
//! All arms must test the SAME receiver expression, and all case values must be
//! string literals. At least 3 arms are required (otherwise it is just an
//! if-else, not a switch).
//!
//! ## Safety
//!
//! The rewrite is semantics-preserving because:
//!   1. The discriminant is a simple variable read (no side effects)
//!   2. Each arm tests the SAME expression against a DISTINCT string literal
//!   3. The final `else` catches everything else (becomes `default`)
//!   4. No arm can fall through to the next (each body is independent)

use super::{Expr, Stmt};
use std::collections::HashSet;

/// Instrumentation counters.
static SITES_EXAMINED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SUCCESSFUL_FOLDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REFUSAL_TOO_FEW_ARMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REFUSAL_DIFFERENT_VARS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REFUSAL_NON_LITERAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REFUSAL_SIDE_EFFECTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Extract `(discriminator_expr, string_literal)` from a String.Equals condition.
///
/// Handles two forms:
///   - Instance: `s.Equals("literal")` → `Call { recv: Some(s), name: "Equals", args: [Str("literal")] }`
///   - Static: `String.Equals(s, "literal")` → `Call { recv: None, name: "String.Equals", args: [s, Str("literal")] }`
///
/// Returns `None` if the condition is not a string equality check.
fn extract_string_equals(cond: &Expr) -> Option<(Expr, String)> {
    let Expr::Call { recv, name, args } = cond else {
        return None;
    };

    // Instance form: recv.Equals("literal")
    if name == "Equals" || name.ends_with(".Equals") {
        if let Some(recv_expr) = recv {
            if args.len() == 1 {
                if let Expr::Str(lit) = &args[0] {
                    return Some((recv_expr.as_ref().clone(), lit.clone()));
                }
            }
        }
    }

    // Static form: String.Equals(recv, "literal") or Equals(recv, "literal")
    if name == "String.Equals" || name == "Equals" || name.ends_with(".String.Equals") {
        if recv.is_none() && args.len() == 2 {
            if let Expr::Str(lit) = &args[1] {
                return Some((args[0].clone(), lit.clone()));
            }
        }
    }

    None
}

/// Check whether an expression is a pure discriminator (no side effects).
///
/// Accepts `Var`, `This`, `Field`, and `StaticField` — simple reads that
/// cannot observe or cause side effects.
fn is_pure_discriminator(e: &Expr) -> bool {
    match e {
        Expr::Var(_) | Expr::This | Expr::StaticField(_, _) => true,
        Expr::Field(base, _) => is_pure_discriminator(base),
        _ => false,
    }
}

/// Check whether two expressions are structurally identical (by rendered text).
fn exprs_match(a: &Expr, b: &Expr) -> bool {
    let mut sa = String::new();
    let mut sb = String::new();
    a.render(&mut sa);
    b.render(&mut sb);
    sa == sb
}

/// Collect all `(discriminator, string_literal, then_body)` triples from the
/// if-else chain. Stops when the else_body is not a single `Stmt::If`.
fn collect_string_arms(stmt: &Stmt) -> Vec<(Expr, String, Vec<Stmt>)> {
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
        let Some((disc, lit)) = extract_string_equals(cond) else {
            break;
        };
        arms.push((disc, lit, then_body.clone()));
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
fn extract_string_tail(stmt: &Stmt) -> Vec<Stmt> {
    let mut current = stmt;
    loop {
        let Stmt::If {
            cond, else_body, ..
        } = current
        else {
            return Vec::new();
        };
        if extract_string_equals(cond).is_none() {
            return Vec::new();
        }
        if else_body.len() == 1 {
            if let Stmt::If { .. } = &else_body[0] {
                current = &else_body[0];
                continue;
            }
        }
        return else_body.clone();
    }
}

/// Try to match a string switch at position `start` in `stmts`.
///
/// Returns `(discriminator, cases, default_body, span)` where `span` is the
/// number of statements consumed (always 1 — the head `If`).
fn try_match_string_switch(
    stmts: &[Stmt],
    start: usize,
) -> Option<(Expr, Vec<(Expr, Vec<Stmt>)>, Option<Vec<Stmt>>, usize)> {
    let stmt = &stmts[start];
    SITES_EXAMINED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Must be an If statement.
    if !matches!(stmt, Stmt::If { .. }) {
        return None;
    }

    let arms = collect_string_arms(stmt);

    // Need at least 3 arms.
    if arms.len() < 3 {
        REFUSAL_TOO_FEW_ARMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }

    // All arms must test the same discriminator.
    let disc = arms[0].0.clone();
    if !is_pure_discriminator(&disc) {
        REFUSAL_SIDE_EFFECTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    for (arm_disc, _, _) in &arms {
        if !exprs_match(arm_disc, &disc) {
            REFUSAL_DIFFERENT_VARS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    }

    // All case values must be distinct string literals.
    let mut seen = HashSet::new();
    let mut cases = Vec::with_capacity(arms.len());
    for (_, lit, body) in &arms {
        if !seen.insert(lit.clone()) {
            // Duplicate literal — not a valid switch.
            REFUSAL_NON_LITERAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        cases.push((Expr::Str(lit.clone()), body.clone()));
    }

    // The tail is the final else body (default case).
    let tail = extract_string_tail(stmt);
    let default_body = if tail.is_empty() {
        None
    } else {
        Some(tail)
    };

    SUCCESSFUL_FOLDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some((disc, cases, default_body, 1))
}

/// Recurse into nested blocks to find string switches at all levels.
fn recurse_into_nested(stmts: &mut Vec<Stmt>) -> usize {
    let mut rewrites = 0;
    for s in stmts.iter_mut() {
        for child_list in super::children_mut(s) {
            rewrites += apply(child_list);
        }
    }
    rewrites
}

/// Run the string switch reconstruction pass over one method body.
///
/// Returns the number of rewrites applied.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut rewrites = 0;
    let mut i = 0;
    while i < stmts.len() {
        if let Some((disc, cases, default_body, span)) = try_match_string_switch(stmts, i) {
            let switch_stmt = Stmt::StringSwitch {
                disc: Box::new(disc),
                cases: cases
                    .into_iter()
                    .map(|(val, body)| (vec![val], body))
                    .collect(),
                tail: default_body,
            };
            stmts.splice(i..i + span, std::iter::once(switch_stmt));
            rewrites += 1;
            // Don't increment i — check for another switch at same position.
        } else {
            i += 1;
        }
    }
    // Recurse into nested blocks.
    rewrites += recurse_into_nested(stmts);
    rewrites
}

/// Instrumentation report for the string switch pass.
pub fn report() -> String {
    let examined = SITES_EXAMINED.load(std::sync::atomic::Ordering::Relaxed);
    let folds = SUCCESSFUL_FOLDS.load(std::sync::atomic::Ordering::Relaxed);
    if examined == 0 {
        return String::new();
    }
    let too_few = REFUSAL_TOO_FEW_ARMS.load(std::sync::atomic::Ordering::Relaxed);
    let diff_vars = REFUSAL_DIFFERENT_VARS.load(std::sync::atomic::Ordering::Relaxed);
    let non_lit = REFUSAL_NON_LITERAL.load(std::sync::atomic::Ordering::Relaxed);
    let side_fx = REFUSAL_SIDE_EFFECTS.load(std::sync::atomic::Ordering::Relaxed);
    let mut s = format!(
        "[reconstruct]   string_switch: examined {examined} sites, folded {folds}\n"
    );
    if too_few > 0 || diff_vars > 0 || non_lit > 0 || side_fx > 0 {
        s.push_str(&format!(
            "[reconstruct]     refusals: too_few_arms={too_few}, different_vars={diff_vars}, non_literal={non_lit}, side_effects={side_fx}\n"
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn str_lit(s: &str) -> Expr {
        Expr::Str(s.to_string())
    }

    /// Instance form: `s.Equals("literal")`
    fn instance_equals(var_name: &str, literal: &str) -> Expr {
        Expr::Call {
            recv: Some(Box::new(var(var_name))),
            name: "Equals".into(),
            args: vec![str_lit(literal)],
        }
    }

    /// Static form: `String.Equals(s, "literal")`
    fn static_equals(var_name: &str, literal: &str) -> Expr {
        Expr::Call {
            recv: None,
            name: "String.Equals".into(),
            args: vec![var(var_name), str_lit(literal)],
        }
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

    /// Build a 3-arm string switch chain with instance form.
    fn three_arm_string_chain() -> Stmt {
        make_if(
            instance_equals("s", "foo"),
            vec![body_stmt()],
            vec![make_if(
                instance_equals("s", "bar"),
                vec![body_stmt()],
                vec![make_if(
                    instance_equals("s", "baz"),
                    vec![body_stmt()],
                    vec![body_stmt()], // default case
                )],
            )],
        )
    }

    /// 1. Basic string switch with 3 arms.
    #[test]
    fn basic_three_arm_string_switch() {
        let mut stmts = vec![three_arm_string_chain()];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::StringSwitch { disc, cases, tail } => {
                // Discriminant is Var("s").
                assert!(matches!(disc.as_ref(), Expr::Var(n) if n == "s"));
                assert_eq!(cases.len(), 3);
                // Check case values.
                assert!(matches!(&cases[0].0[0], Expr::Str(s) if s == "foo"));
                assert!(matches!(&cases[1].0[0], Expr::Str(s) if s == "bar"));
                assert!(matches!(&cases[2].0[0], Expr::Str(s) if s == "baz"));
                // Default tail present.
                assert!(tail.is_some());
                assert_eq!(tail.as_ref().unwrap().len(), 1);
            }
            other => panic!("expected StringSwitch, got {other:?}"),
        }
    }

    /// 2. String switch with default branch.
    #[test]
    fn string_switch_with_default() {
        let chain = make_if(
            instance_equals("x", "a"),
            vec![body_stmt()],
            vec![make_if(
                instance_equals("x", "b"),
                vec![body_stmt()],
                vec![make_if(
                    instance_equals("x", "c"),
                    vec![body_stmt()],
                    vec![Stmt::Return(None)], // default: return;
                )],
            )],
        );
        let mut stmts = vec![chain];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::StringSwitch { tail, .. } => {
                assert!(tail.is_some());
                assert!(matches!(&tail.as_ref().unwrap()[0], Stmt::Return(None)));
            }
            other => panic!("expected StringSwitch, got {other:?}"),
        }
    }

    /// 3. String switch with 5+ arms.
    #[test]
    fn five_arm_string_switch() {
        let literals = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let mut stmt = make_if(
            instance_equals("s", literals[4]),
            vec![body_stmt()],
            vec![body_stmt()], // default
        );
        for i in (0..4).rev() {
            stmt = make_if(
                instance_equals("s", literals[i]),
                vec![body_stmt()],
                vec![stmt],
            );
        }
        let mut stmts = vec![stmt];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::StringSwitch { cases, .. } => {
                assert_eq!(cases.len(), 5);
                for (i, (vals, _)) in cases.iter().enumerate() {
                    assert!(matches!(&vals[0], Expr::Str(s) if s == literals[i]));
                }
            }
            other => panic!("expected StringSwitch, got {other:?}"),
        }
    }

    /// 4. String switch with static String.Equals(s, "literal") form.
    #[test]
    fn static_string_equals_form() {
        let chain = make_if(
            static_equals("s", "one"),
            vec![body_stmt()],
            vec![make_if(
                static_equals("s", "two"),
                vec![body_stmt()],
                vec![make_if(
                    static_equals("s", "three"),
                    vec![body_stmt()],
                    vec![],
                )],
            )],
        );
        let mut stmts = vec![chain];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::StringSwitch { cases, .. } => {
                assert_eq!(cases.len(), 3);
                assert!(matches!(&cases[0].0[0], Expr::Str(s) if s == "one"));
                assert!(matches!(&cases[1].0[0], Expr::Str(s) if s == "two"));
                assert!(matches!(&cases[2].0[0], Expr::Str(s) if s == "three"));
            }
            other => panic!("expected StringSwitch, got {other:?}"),
        }
    }

    /// 5. String switch with instance s.Equals("literal") form.
    #[test]
    fn instance_equals_form() {
        let chain = make_if(
            instance_equals("name", "alice"),
            vec![body_stmt()],
            vec![make_if(
                instance_equals("name", "bob"),
                vec![body_stmt()],
                vec![make_if(
                    instance_equals("name", "charlie"),
                    vec![body_stmt()],
                    vec![],
                )],
            )],
        );
        let mut stmts = vec![chain];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::StringSwitch { disc, cases, .. } => {
                assert!(matches!(disc.as_ref(), Expr::Var(n) if n == "name"));
                assert_eq!(cases.len(), 3);
            }
            other => panic!("expected StringSwitch, got {other:?}"),
        }
    }

    /// 6. Non-match: only 2 arms (should remain if-else).
    #[test]
    fn two_arms_not_converted() {
        let chain = make_if(
            instance_equals("s", "foo"),
            vec![body_stmt()],
            vec![make_if(
                instance_equals("s", "bar"),
                vec![body_stmt()],
                vec![],
            )],
        );
        let mut stmts = vec![chain];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    /// 7. Non-match: different variables tested in each arm.
    #[test]
    fn different_variables_not_converted() {
        let chain = make_if(
            instance_equals("s", "foo"),
            vec![body_stmt()],
            vec![make_if(
                instance_equals("t", "bar"), // different variable!
                vec![body_stmt()],
                vec![make_if(
                    instance_equals("s", "baz"),
                    vec![body_stmt()],
                    vec![],
                )],
            )],
        );
        let mut stmts = vec![chain];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    /// 8. Non-match: non-literal case values.
    #[test]
    fn non_literal_case_values_not_converted() {
        // First arm uses a variable instead of a string literal.
        let cond = Expr::Call {
            recv: Some(Box::new(var("s"))),
            name: "Equals".into(),
            args: vec![var("other")], // not a literal!
        };
        let chain = make_if(
            cond,
            vec![body_stmt()],
            vec![make_if(
                instance_equals("s", "bar"),
                vec![body_stmt()],
                vec![make_if(
                    instance_equals("s", "baz"),
                    vec![body_stmt()],
                    vec![],
                )],
            )],
        );
        let mut stmts = vec![chain];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    /// 9. Non-match: side effects in condition (call on a non-pure receiver).
    #[test]
    fn side_effects_in_condition_not_converted() {
        // Receiver is a call expression (has side effects).
        let cond = Expr::Call {
            recv: Some(Box::new(Expr::Call {
                recv: Some(Box::new(var("obj"))),
                name: "GetString".into(),
                args: vec![],
            })),
            name: "Equals".into(),
            args: vec![str_lit("foo")],
        };
        let chain = make_if(
            cond,
            vec![body_stmt()],
            vec![make_if(
                instance_equals("s", "bar"),
                vec![body_stmt()],
                vec![make_if(
                    instance_equals("s", "baz"),
                    vec![body_stmt()],
                    vec![],
                )],
            )],
        );
        let mut stmts = vec![chain];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    /// 10. Nested string switch (switch inside switch body).
    #[test]
    fn nested_string_switch() {
        let inner_chain = make_if(
            instance_equals("t", "x"),
            vec![body_stmt()],
            vec![make_if(
                instance_equals("t", "y"),
                vec![body_stmt()],
                vec![make_if(
                    instance_equals("t", "z"),
                    vec![body_stmt()],
                    vec![],
                )],
            )],
        );
        let outer_chain = make_if(
            instance_equals("s", "a"),
            vec![inner_chain], // inner string switch in this body
            vec![make_if(
                instance_equals("s", "b"),
                vec![body_stmt()],
                vec![make_if(
                    instance_equals("s", "c"),
                    vec![body_stmt()],
                    vec![],
                )],
            )],
        );
        let mut stmts = vec![outer_chain];
        // Both outer and inner should convert.
        assert_eq!(apply(&mut stmts), 2);
        // Outer should be a StringSwitch.
        match &stmts[0] {
            Stmt::StringSwitch { cases, .. } => {
                assert_eq!(cases.len(), 3);
                // First case body should contain the inner StringSwitch.
                match &cases[0].1[0] {
                    Stmt::StringSwitch {
                        disc: inner_disc,
                        cases: inner_cases,
                        ..
                    } => {
                        assert!(matches!(inner_disc.as_ref(), Expr::Var(n) if n == "t"));
                        assert_eq!(inner_cases.len(), 3);
                    }
                    other => panic!("expected inner StringSwitch, got {other:?}"),
                }
            }
            other => panic!("expected outer StringSwitch, got {other:?}"),
        }
    }
}
