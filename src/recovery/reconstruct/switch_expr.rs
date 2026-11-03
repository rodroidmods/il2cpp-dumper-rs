//! If-else chain → C# 8+ switch expression reconstruction.
//!
//! ## The pattern
//!
//! The C# compiler emits `if (discrim == val) target = expr; else if …` for
//! switch expressions. IL2CPP preserves this shape, so the recovered IR contains:
//!
//! ```text
//! if (discrim == val1) {
//!     target = expr1;
//! } else if (discrim == val2) {
//!     target = expr2;
//! } else {
//!     target = default_expr;
//! }
//! ```
//!
//! This pass detects those chains and converts them to switch expressions:
//!
//! ```text
//! target = discrim switch {
//!     val1 => expr1,
//!     val2 => expr2,
//!     _ => default_expr
//! };
//! ```
//!
//! ## Detection criteria
//!
//! The recognizer matches:
//! - A chain of `Stmt::If` where each `then_body` contains exactly one
//!   `Stmt::Assign(target, None, expr)` with the SAME target variable.
//! - Each condition is `Expr::Cmp(CmpKind::Eq, discrim, pattern)` where
//!   `discrim` is the same expression in every arm.
//! - The final `else_body` (if present) contains exactly one
//!   `Stmt::Assign(target, None, default_expr)`.
//!
//! ## Safety conditions
//!
//! The rewrite is safe when:
//! 1. All arms assign to the SAME target variable.
//! 2. All conditions test the SAME discriminant expression.
//! 3. Each arm's `then_body` has exactly one statement (the assignment).
//! 4. The discriminant has no side effects (no calls).
//!
//! ## Minimum arm count
//!
//! At least 2 arms are required. A single `if` with no else is just a
//! conditional assignment, not a switch expression.
//!
//! ## Null tests are NOT switches
//!
//! A single `null` arm plus a `_` default is the recovered shape of
//! `x?.M(args) ?? false` / `x != null ? … : …`, never of a source switch
//! expression. `try_match` refuses it. `rewrite_bool_null_tests` instead
//! rebuilds the boolean form:
//!
//! ```text
//! if (x == null) { t = 0; } else { t = x.Contains(a) & 1; }
//! …
//! if ((t | other) == 0) { A } else { B }
//! ```
//!
//! becomes
//!
//! ```text
//! t = x?.Contains(a) ?? false;
//! …
//! if (!(t || other)) { A } else { B }
//! ```
//!
//! The `& 1` is IL2CPP's bool-to-int canonicalisation; keeping it would
//! produce `bool & int` (invalid C#), and the later temp inlining used to
//! move the whole `switch` into the condition.
//!
//! ## Measured reality
//!
//! On the x64 corpus (47,924 methods), switch expressions are rare but present.
//! The pattern is most common in state machine dispatchers and enum-to-string
//! conversions.

use super::super::csharp::{BinOp, CmpKind};
use super::{children, children_mut, Expr, Stmt};
use std::collections::HashMap;

/// Run the switch expression recognizer over one method body.
///
/// A boolean null-test recovery runs first (it consumes shapes the generic
/// matcher would otherwise turn into a one-arm switch), then the switch
/// recognizer uses a **pre-order** walk: the current list is processed before
/// recursing into children. This is critical because an outer if-else chain
/// consumes inner chains as part of its arms; if we processed inner lists first
/// (post-order), they would be rewritten independently and break the outer
/// pattern.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    // Count each display name once over the ORIGINAL tree: the bool recovery
    // only rewrites a temp whose sole read is the condition it normalises, so
    // a name reused by another SSA live range blocks the rewrite.
    let mut mentions = HashMap::new();
    count_var_mentions(stmts, &mut mentions);
    let mut count = rewrite_bool_null_tests(stmts, &mentions);
    count += rewrite_switch_lists(stmts);
    count
}

/// Pre-order switch-expression scan (see [`apply`] for why pre-order).
fn rewrite_switch_lists(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = rewrite_list(stmts);
    for s in stmts.iter_mut() {
        for child in children_mut(s) {
            count += rewrite_switch_lists(child);
        }
    }
    count
}

/// Scan a statement list for switch expression patterns and rewrite in place.
fn rewrite_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < stmts.len() {
        if let Some(replacement) = try_match(&stmts[i]) {
            stmts[i] = replacement;
            count += 1;
        }
        i += 1;
    }
    count
}

// ===========================================================================
// Boolean null-test recovery (`x?.M() ?? false`)
// ===========================================================================

/// One `if (x == null) t = 0; else t = <bool>;` arm pair reduced to a bool.
struct BoolNullTest {
    target: String,
    /// Boolean expression equivalent to the whole if/else.
    value: Expr,
}

/// Rewrite bool-shaped null tests and their single boolean consumer.
///
/// Recurses into nested statement lists; the consumer may live in any later
/// statement of the same list (typically a nested `if` after the temp inliner
/// removed the intervening plumbing), so each candidate searches the rest of
/// its list recursively.
fn rewrite_bool_null_tests(stmts: &mut Vec<Stmt>, mentions: &HashMap<String, usize>) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < stmts.len() {
        if let Some(test) = match_bool_null_test(&stmts[i]) {
            // The temp must have exactly one read in the whole method: that
            // read is the condition we are about to make boolean-aware.
            if mentions.get(&test.target).copied() == Some(1)
                && rewrite_bool_use(&mut stmts[i + 1..], &test.target)
            {
                stmts[i] = Stmt::Assign(test.target, None, test.value);
                count += 1;
            }
        }
        i += 1;
    }
    for s in stmts.iter_mut() {
        for child in children_mut(s) {
            count += rewrite_bool_null_tests(child, mentions);
        }
    }
    count
}

/// Recognise `if (x == null) { t = 0; } else { t = e & 1; }` (or the mirror).
///
/// Both arms must be a single assignment to the SAME target, one arm must be
/// literal false (`0`/`false`) and the other must be a boolean expression
/// carrying the IL2CPP `& 1` canonicalisation (or a bool literal). Anything
/// else is an ordinary conditional assignment and is left alone.
fn match_bool_null_test(stmt: &Stmt) -> Option<BoolNullTest> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };
    if then_body.len() != 1 || else_body.len() != 1 {
        return None;
    }
    let (then_target, then_value) = single_assignment(&then_body[0])?;
    let (else_target, else_value) = single_assignment(&else_body[0])?;
    if then_target != else_target {
        return None;
    }

    let (receiver, null_literal, is_eq) = match_null_comparison(cond)?;
    let then_bool = bool_value(then_value)?;
    let else_bool = bool_value(else_value)?;

    // Which arm is the null branch, and which runs when the receiver exists?
    let (null_value, nonnull_value) = if is_eq {
        (then_bool, else_bool)
    } else {
        (else_bool, then_bool)
    };
    // Only the `false` / `value` pair is this idiom; a `true` null arm is not
    // what a null-conditional or a plain boolean test lowers to.
    if !matches!(null_value, Expr::Bool(false)) {
        return None;
    }
    // One side must be the stripped `& 1` mask (or an explicit bool), so an
    // ordinary integer `? 0 : 1` is not misclassified as boolean.
    let carries_bool_mask = has_bool_mask(then_value) || has_bool_mask(else_value);
    if !carries_bool_mask {
        return None;
    }

    // `x?.M() ?? false` when the non-null arm is a member access on the same
    // receiver and the test is a true null test.
    let value = if matches!(null_literal, Expr::Null) {
        if let Some(access) = member_access_on(&receiver, &nonnull_value) {
            Expr::Bin(
                BinOp::NullCoalesce,
                Box::new(Expr::NullConditional {
                    receiver: Box::new(receiver.clone()),
                    access: Box::new(access),
                }),
                Box::new(Expr::Bool(false)),
            )
        } else {
            conditional_bool(receiver, null_literal, nonnull_value)
        }
    } else {
        conditional_bool(receiver, null_literal, nonnull_value)
    };

    Some(BoolNullTest {
        target: then_target.clone(),
        value,
    })
}

/// `x == null`, `null == x`, `x != null`, `null != x`, and the same with the
/// integer zero spelling. Returns `(receiver, literal, is_eq)`.
fn match_null_comparison(cond: &Expr) -> Option<(Expr, Expr, bool)> {
    let Expr::Cmp(kind, a, b) = cond else {
        return None;
    };
    let is_eq = match kind {
        CmpKind::Eq => true,
        CmpKind::Ne => false,
        _ => return None,
    };
    let is_null_like = |e: &Expr| matches!(e, Expr::Null | Expr::Int(0));
    if let (Expr::Var(_), true) = (a.as_ref(), is_null_like(b)) {
        return Some(((**a).clone(), (**b).clone(), is_eq));
    }
    if let (true, Expr::Var(_)) = (is_null_like(a), b.as_ref()) {
        return Some(((**b).clone(), (**a).clone(), is_eq));
    }
    None
}

/// Translate an arm value into a boolean expression.
/// `0`/`false` -> `false`, `1`/`true` -> `true`, `e & 1` -> `e`.
fn bool_value(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Bool(v) => Some(Expr::Bool(*v)),
        Expr::Int(0) => Some(Expr::Bool(false)),
        Expr::Int(1) => Some(Expr::Bool(true)),
        Expr::Bin(BinOp::And, x, y) => {
            if matches!(x.as_ref(), Expr::Int(1)) {
                Some((**y).clone())
            } else if matches!(y.as_ref(), Expr::Int(1)) {
                Some((**x).clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn has_bool_mask(e: &Expr) -> bool {
    matches!(e, Expr::Bool(_))
        || matches!(e, Expr::Bin(BinOp::And, x, y)
            if matches!(x.as_ref(), Expr::Int(1)) || matches!(y.as_ref(), Expr::Int(1)))
}

/// `access` if it is a call or field read whose receiver is exactly `recv`.
fn member_access_on(recv: &Expr, access: &Expr) -> Option<Expr> {
    match access {
        Expr::Call {
            recv: Some(base), ..
        }
        | Expr::Field(base, _)
            if base.as_ref() == recv =>
        {
            Some(access.clone())
        }
        _ => None,
    }
}

/// Fallback ternary, always producing a bool:
/// `(x != null ? value : false)`.
fn conditional_bool(receiver: Expr, literal: Expr, value: Expr) -> Expr {
    let test = if matches!(literal, Expr::Null) {
        Expr::Cmp(CmpKind::Ne, Box::new(receiver), Box::new(Expr::Null))
    } else {
        Expr::Cmp(CmpKind::Ne, Box::new(receiver), Box::new(Expr::Int(0)))
    };
    Expr::Cond(Box::new(test), Box::new(value), Box::new(Expr::Bool(false)))
}

fn single_assignment(stmt: &Stmt) -> Option<(&String, &Expr)> {
    match stmt {
        Stmt::Assign(target, None, value) => Some((target, value)),
        _ => None,
    }
}

/// Rewrite the one boolean use of `target` in the remaining statements.
fn rewrite_bool_use(stmts: &mut [Stmt], target: &str) -> bool {
    for stmt in stmts.iter_mut() {
        if rewrite_bool_condition(stmt, target) {
            return true;
        }
    }
    false
}

/// Rewrite a condition that consumes `target` in a boolean context.
fn rewrite_bool_condition(stmt: &mut Stmt, target: &str) -> bool {
    let cond = match stmt {
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => Some(cond),
        Stmt::For {
            cond: Some(cond), ..
        } => Some(cond),
        _ => None,
    };
    if let Some(cond) = cond {
        if rewrite_bool_cond_expr(cond, target) {
            return true;
        }
    }
    for child in children_mut(stmt) {
        if rewrite_bool_use(child, target) {
            return true;
        }
    }
    false
}

/// `(t | other) == 0` -> `!(t || other)` (and the `!= 0` mirror),
/// `t == 0` -> `!t`, `t != 0` -> `t`.
///
/// `other` must be pure: `|` always evaluates it while `||` short-circuits.
/// It must also be syntactically boolean; mixed `int | bool` is not valid C#,
/// so the original source was boolean on both sides.
fn rewrite_bool_cond_expr(e: &mut Expr, target: &str) -> bool {
    if let Some(new) = bool_cond_rewrite(e, target) {
        *e = new;
        return true;
    }
    // The consumer may sit inside a larger boolean expression (`not`,
    // `&&`/`||` chains); rewrite the nested occurrence.
    for child in super::il2cpp_cleanup::expr_children_mut(e) {
        if rewrite_bool_cond_expr(child, target) {
            return true;
        }
    }
    false
}

/// Root-only form of the condition rewrite (see [`rewrite_bool_cond_expr`]).
fn bool_cond_rewrite(e: &Expr, target: &str) -> Option<Expr> {
    let Expr::Cmp(kind @ (CmpKind::Eq | CmpKind::Ne), a, b) = e else {
        return None;
    };
    let is_zero = |x: &Expr| matches!(x, Expr::Int(0) | Expr::Bool(false));
    let side = if is_zero(b) {
        a.as_ref()
    } else if is_zero(a) {
        b.as_ref()
    } else {
        return None;
    };

    if let Expr::Var(name) = side {
        if name == target {
            return Some(if matches!(kind, CmpKind::Eq) {
                Expr::Not(Box::new(Expr::Var(target.to_string())))
            } else {
                Expr::Var(target.to_string())
            });
        }
    }

    let Expr::Bin(BinOp::Or, l, r) = side else {
        return None;
    };
    let other = if matches!(l.as_ref(), Expr::Var(n) if n == target) {
        r.as_ref()
    } else if matches!(r.as_ref(), Expr::Var(n) if n == target) {
        l.as_ref()
    } else {
        return None;
    };
    if !pure_bool_operand(other) || !boolean_shaped(other) {
        return None;
    }
    let bin = Expr::Bin(
        BinOp::OrElse,
        Box::new(Expr::Var(target.to_string())),
        Box::new(other.clone()),
    );
    Some(if matches!(kind, CmpKind::Eq) {
        Expr::Not(Box::new(bin))
    } else {
        bin
    })
}

/// No calls, no allocations, no dereferences that can fault.
fn pure_bool_operand(e: &Expr) -> bool {
    match e {
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::StaticField(..) => true,
        Expr::Field(r, _) => pure_bool_operand(r),
        Expr::Cmp(_, a, b) | Expr::Bin(_, a, b) => pure_bool_operand(a) && pure_bool_operand(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) => pure_bool_operand(x),
        _ => false,
    }
}

/// Syntactically boolean operand (so `||` cannot mix in a raw int).
fn boolean_shaped(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Cmp(..)
            | Expr::Not(_)
            | Expr::Bool(_)
            | Expr::PatternMatch { .. }
            | Expr::Field(..)
            | Expr::NullConditional { .. }
            | Expr::Var(_)
    )
}

/// Count bare `Var` occurrences across the whole tree (assignment targets are
/// not reads and are not counted).
fn count_var_mentions(stmts: &[Stmt], out: &mut HashMap<String, usize>) {
    for stmt in stmts {
        count_stmt_mentions(stmt, out);
    }
}

fn count_stmt_mentions(stmt: &Stmt, out: &mut HashMap<String, usize>) {
    // Expressions owned by this statement.
    let mut exprs: Vec<&Expr> = Vec::new();
    match stmt {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => exprs.push(e),
        Stmt::Assign(_, _, e) => exprs.push(e),
        Stmt::Return(Some(e)) => exprs.push(e),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            exprs.push(cond)
        }
        Stmt::For { cond: Some(c), .. } => exprs.push(c),
        Stmt::Lock { expr, .. } => exprs.push(expr),
        Stmt::StringSwitch { disc, .. } => exprs.push(disc),
        Stmt::EventAssign { field, value, .. } => {
            exprs.push(field);
            exprs.push(value);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            exprs.push(collection);
            exprs.push(index);
            exprs.push(value);
        }
        _ => {}
    }
    for e in exprs {
        count_expr_mentions(e, out);
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(stmt, &mut kids);
    for k in kids {
        count_var_mentions(k, out);
    }
}

fn count_expr_mentions(e: &Expr, out: &mut HashMap<String, usize>) {
    if let Expr::Var(name) = e {
        *out.entry(name.clone()).or_insert(0) += 1;
    }
    for child in super::il2cpp_cleanup::expr_children(e) {
        count_expr_mentions(child, out);
    }
}

/// Try to match the switch expression pattern on a single statement.
///
/// Returns `Some(replacement)` when the statement is an if-else chain that
/// assigns to the same target variable, and `None` otherwise.
fn try_match(stmt: &Stmt) -> Option<Stmt> {
    // Collect all arms from the if-else chain.
    let arms = collect_arms(stmt)?;

    // Need at least 2 arms to form a switch expression.
    if arms.len() < 2 {
        return None;
    }

    // All arms must assign to the same target.
    let target = &arms[0].target;
    if !arms.iter().all(|arm| arm.target == *target) {
        return None;
    }

    // All conditions must test the same discriminant.
    let discrim = &arms[0].discrim;
    if !arms.iter().all(|arm| exprs_eq(&arm.discrim, discrim)) {
        return None;
    }

    // The discriminant must not have side effects.
    if expr_has_call(discrim) {
        return None;
    }

    // Build the switch expression.
    let mut switch_arms = Vec::new();
    let mut default_arm = None;

    for arm in &arms {
        match &arm.pattern {
            Some(pattern) => {
                switch_arms.push((pattern.clone(), arm.value.clone()));
            }
            None => {
                // This is the default arm (from the final else).
                default_arm = Some(Box::new(arm.value.clone()));
            }
        }
    }

    // A lone `null` arm plus `_` is a conditional/null-conditional, not a
    // source switch. Emitting `x switch { null => …, _ => … }` is unreadable
    // and, once the temp is inlined into a condition, can mix bool and int.
    // (`rewrite_bool_null_tests` should already have rebuilt a boolean form;
    // this guard keeps every other null-test shape out of a SwitchExpr.)
    if switch_arms.len() == 1 && arms.iter().any(|a| matches!(a.pattern, Some(Expr::Null))) {
        return None;
    }

    let switch_expr = Expr::SwitchExpr {
        discrim: Box::new(discrim.clone()),
        arms: switch_arms,
        default_arm,
    };

    Some(Stmt::Assign(target.clone(), None, switch_expr))
}

/// An arm of the if-else chain.
struct Arm {
    /// The discriminant expression (same in all arms).
    discrim: Expr,
    /// The pattern to match (None for the default arm).
    pattern: Option<Expr>,
    /// The target variable being assigned.
    target: String,
    /// The value being assigned.
    value: Expr,
}

/// Collect all arms from an if-else chain.
///
/// Returns `None` if the structure doesn't match the expected pattern.
fn collect_arms(stmt: &Stmt) -> Option<Vec<Arm>> {
    let mut arms = Vec::new();
    let mut current = stmt;

    loop {
        match current {
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                // Extract the condition: must be `discrim == pattern`.
                let (discrim, pattern) = extract_eq(cond)?;

                // The then_body must have exactly one statement: an assignment
                // or a nested if-else chain that can be converted to a switch.
                if then_body.len() != 1 {
                    return None;
                }
                let (target, value) = match &then_body[0] {
                    Stmt::Assign(target, None, value) => (target.clone(), value.clone()),
                    Stmt::If { .. } => {
                        // Try to recursively convert the inner if-else chain.
                        if let Some(Stmt::Assign(target, None, value)) = try_match(&then_body[0]) {
                            (target, value)
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                };

                arms.push(Arm {
                    discrim,
                    pattern: Some(pattern),
                    target,
                    value,
                });

                // If else_body is empty, we're done.
                if else_body.is_empty() {
                    break;
                }

                // If else_body has exactly one statement, check what kind.
                if else_body.len() == 1 {
                    // If it's another If, continue the chain.
                    if matches!(&else_body[0], Stmt::If { .. }) {
                        current = &else_body[0];
                        continue;
                    }

                    // If it's an assignment, it's the default arm.
                    if let Stmt::Assign(target, None, value) = &else_body[0] {
                        let discrim = arms[0].discrim.clone();
                        arms.push(Arm {
                            discrim,
                            pattern: None,
                            target: target.clone(),
                            value: value.clone(),
                        });
                        break;
                    }

                    // Otherwise, the structure doesn't match.
                    return None;
                }

                // Otherwise, the structure doesn't match.
                return None;
            }
            _ => return None,
        }
    }

    Some(arms)
}

/// Extract `(discriminant, pattern)` from `discrim == pattern`.
///
/// Returns `None` if the condition is not an equality comparison.
fn extract_eq(cond: &Expr) -> Option<(Expr, Expr)> {
    let Expr::Cmp(CmpKind::Eq, lhs, rhs) = cond else {
        return None;
    };
    Some((lhs.as_ref().clone(), rhs.as_ref().clone()))
}

/// Structural expression equality (copied from mod.rs to avoid circular dependency).
fn exprs_eq(a: &Expr, b: &Expr) -> bool {
    let mut sa = String::new();
    let mut sb = String::new();
    a.render(&mut sa);
    b.render(&mut sb);
    sa == sb
}

/// Does `e` have a call (copied from mod.rs to avoid circular dependency)?
fn expr_has_call(e: &Expr) -> bool {
    match e {
        Expr::Call { .. }
        | Expr::RawCall(..)
        | Expr::NewObj(..)
        | Expr::NewArr(..)
        | Expr::Await(..) => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_has_call(a) || expr_has_call(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) => expr_has_call(x),
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => expr_has_call(r),
        Expr::Cond(c, a, b) => expr_has_call(c) || expr_has_call(a) || expr_has_call(b),
        Expr::Lambda { body, .. } => expr_has_call(body),
        Expr::ArrayInit { items, .. } => items.iter().any(expr_has_call),
        Expr::CollectionInit { .. } | Expr::ObjInit { .. } => true,
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::Unknown
        | Expr::MissingAbiArg
        | Expr::StaticField(..)
        | Expr::MethodGroup(_) => false,
        Expr::Index { collection, index } => expr_has_call(collection) || expr_has_call(index),
        Expr::NullConditional { receiver, access } => {
            expr_has_call(receiver) || expr_has_call(access)
        }
        Expr::PatternMatch { expr, .. } => expr_has_call(expr),
        Expr::StringInterp { parts } => {
            use super::super::csharp::InterpPart;
            parts.iter().any(|p| match p {
                InterpPart::Literal(_) => false,
                InterpPart::Expr(e, _) => expr_has_call(e),
            })
        }
        Expr::Tuple { elements, .. } => elements.iter().any(expr_has_call),
        Expr::MethodChain { receiver, methods } => {
            expr_has_call(receiver)
                || methods
                    .iter()
                    .any(|m| m.args.iter().any(|a| expr_has_call(a)))
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            expr_has_call(discrim)
                || arms
                    .iter()
                    .any(|(p, r)| expr_has_call(p) || expr_has_call(r))
                || default_arm.as_deref().is_some_and(|e| expr_has_call(e))
        }
        Expr::Range { from, to } => {
            from.as_ref().is_some_and(|f| expr_has_call(f))
                || to.as_ref().is_some_and(|t| expr_has_call(t))
        }
        Expr::IndexFromEnd(e) => expr_has_call(e),
        Expr::TypeOf(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple variable expression.
    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    /// Helper to create an equality comparison.
    fn eq_cmp(lhs: Expr, rhs: Expr) -> Expr {
        Expr::Cmp(CmpKind::Eq, Box::new(lhs), Box::new(rhs))
    }

    /// Helper to create an if statement.
    fn if_stmt(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond,
            then_body,
            else_body,
        }
    }

    /// Helper to create an assignment statement.
    fn assign(target: &str, value: Expr) -> Stmt {
        Stmt::Assign(target.to_string(), None, value)
    }

    #[test]
    fn simple_switch_with_3_arms() {
        // if (x == 0) { result = "zero"; }
        // else if (x == 1) { result = "one"; }
        // else if (x == 2) { result = "two"; }
        let stmt = if_stmt(
            eq_cmp(var("x"), Expr::Int(0)),
            vec![assign("result", Expr::Str("zero".to_string()))],
            vec![if_stmt(
                eq_cmp(var("x"), Expr::Int(1)),
                vec![assign("result", Expr::Str("one".to_string()))],
                vec![if_stmt(
                    eq_cmp(var("x"), Expr::Int(2)),
                    vec![assign("result", Expr::Str("two".to_string()))],
                    vec![],
                )],
            )],
        );

        let result = try_match(&stmt);
        assert!(result.is_some(), "should match 3-arm switch");

        let replacement = result.unwrap();
        match replacement {
            Stmt::Assign(
                target,
                _,
                Expr::SwitchExpr {
                    discrim,
                    arms,
                    default_arm,
                },
            ) => {
                assert_eq!(target, "result");
                assert!(matches!(*discrim, Expr::Var(ref n) if n == "x"));
                assert_eq!(arms.len(), 3);
                assert!(default_arm.is_none());
            }
            _ => panic!("expected Assign with SwitchExpr"),
        }
    }

    #[test]
    fn switch_with_default_arm() {
        // if (x == 0) { result = "zero"; }
        // else if (x == 1) { result = "one"; }
        // else { result = "other"; }
        let stmt = if_stmt(
            eq_cmp(var("x"), Expr::Int(0)),
            vec![assign("result", Expr::Str("zero".to_string()))],
            vec![if_stmt(
                eq_cmp(var("x"), Expr::Int(1)),
                vec![assign("result", Expr::Str("one".to_string()))],
                vec![assign("result", Expr::Str("other".to_string()))],
            )],
        );

        let result = try_match(&stmt);
        assert!(result.is_some(), "should match switch with default");

        let replacement = result.unwrap();
        match replacement {
            Stmt::Assign(
                target,
                _,
                Expr::SwitchExpr {
                    arms, default_arm, ..
                },
            ) => {
                assert_eq!(target, "result");
                assert_eq!(arms.len(), 2);
                assert!(default_arm.is_some(), "should have default arm");
            }
            _ => panic!("expected Assign with SwitchExpr"),
        }
    }

    #[test]
    fn switch_with_complex_expressions_in_arms() {
        // if (x == 0) { result = Foo(1); }
        // else if (x == 1) { result = Bar(2); }
        let stmt = if_stmt(
            eq_cmp(var("x"), Expr::Int(0)),
            vec![assign(
                "result",
                Expr::Call {
                    recv: None,
                    name: "Foo".to_string(),
                    args: vec![Expr::Int(1)],
                },
            )],
            vec![if_stmt(
                eq_cmp(var("x"), Expr::Int(1)),
                vec![assign(
                    "result",
                    Expr::Call {
                        recv: None,
                        name: "Bar".to_string(),
                        args: vec![Expr::Int(2)],
                    },
                )],
                vec![],
            )],
        );

        let result = try_match(&stmt);
        assert!(result.is_some(), "should match switch with complex arms");

        let replacement = result.unwrap();
        match replacement {
            Stmt::Assign(_, _, Expr::SwitchExpr { arms, .. }) => {
                assert_eq!(arms.len(), 2);
                // First arm should have Foo(1).
                assert!(matches!(&arms[0].1, Expr::Call { name, .. } if name == "Foo"));
                // Second arm should have Bar(2).
                assert!(matches!(&arms[1].1, Expr::Call { name, .. } if name == "Bar"));
            }
            _ => panic!("expected Assign with SwitchExpr"),
        }
    }

    #[test]
    fn nested_switch_expressions() {
        // if (x == 0) {
        //     if (y == 0) { result = "00"; }
        //     else { result = "01"; }
        // }
        // else if (x == 1) { result = "1"; }
        let inner_switch = if_stmt(
            eq_cmp(var("y"), Expr::Int(0)),
            vec![assign("result", Expr::Str("00".to_string()))],
            vec![assign("result", Expr::Str("01".to_string()))],
        );

        let outer_stmt = if_stmt(
            eq_cmp(var("x"), Expr::Int(0)),
            vec![inner_switch],
            vec![if_stmt(
                eq_cmp(var("x"), Expr::Int(1)),
                vec![assign("result", Expr::Str("1".to_string()))],
                vec![],
            )],
        );

        let result = try_match(&outer_stmt);
        assert!(result.is_some(), "should match outer switch");

        let replacement = result.unwrap();
        match replacement {
            Stmt::Assign(_, _, Expr::SwitchExpr { arms, .. }) => {
                assert_eq!(arms.len(), 2);
                // First arm should have a nested SwitchExpr.
                assert!(matches!(&arms[0].1, Expr::SwitchExpr { .. }));
            }
            _ => panic!("expected Assign with SwitchExpr"),
        }
    }

    #[test]
    fn switch_in_assignment_context() {
        // if (status == 0) { value = 100; }
        // else if (status == 1) { value = 200; }
        // else { value = 0; }
        let stmt = if_stmt(
            eq_cmp(var("status"), Expr::Int(0)),
            vec![assign("value", Expr::Int(100))],
            vec![if_stmt(
                eq_cmp(var("status"), Expr::Int(1)),
                vec![assign("value", Expr::Int(200))],
                vec![assign("value", Expr::Int(0))],
            )],
        );

        let result = try_match(&stmt);
        assert!(result.is_some(), "should match assignment context");

        let replacement = result.unwrap();
        match replacement {
            Stmt::Assign(target, _, Expr::SwitchExpr { .. }) => {
                assert_eq!(target, "value");
            }
            _ => panic!("expected Assign with SwitchExpr"),
        }
    }

    #[test]
    fn switch_in_return_context() {
        // This tests that the pass works when the if-else is inside a function.
        // The if-else chain itself is the statement, not a return.
        let mut stmts = vec![if_stmt(
            eq_cmp(var("code"), Expr::Int(0)),
            vec![assign("msg", Expr::Str("ok".to_string()))],
            vec![if_stmt(
                eq_cmp(var("code"), Expr::Int(1)),
                vec![assign("msg", Expr::Str("error".to_string()))],
                vec![assign("msg", Expr::Str("unknown".to_string()))],
            )],
        )];

        let count = apply(&mut stmts);
        assert_eq!(count, 1, "should rewrite one switch");

        match &stmts[0] {
            Stmt::Assign(
                target,
                _,
                Expr::SwitchExpr {
                    arms, default_arm, ..
                },
            ) => {
                assert_eq!(target, "msg");
                assert_eq!(arms.len(), 2);
                assert!(default_arm.is_some());
            }
            _ => panic!("expected Assign with SwitchExpr"),
        }
    }

    #[test]
    fn mismatched_targets_not_rewritten() {
        // if (x == 0) { result1 = "a"; }
        // else if (x == 1) { result2 = "b"; }
        let stmt = if_stmt(
            eq_cmp(var("x"), Expr::Int(0)),
            vec![assign("result1", Expr::Str("a".to_string()))],
            vec![if_stmt(
                eq_cmp(var("x"), Expr::Int(1)),
                vec![assign("result2", Expr::Str("b".to_string()))],
                vec![],
            )],
        );

        let result = try_match(&stmt);
        assert!(result.is_none(), "mismatched targets should not match");
    }

    #[test]
    fn mismatched_discriminants_not_rewritten() {
        // if (x == 0) { result = "a"; }
        // else if (y == 1) { result = "b"; }
        let stmt = if_stmt(
            eq_cmp(var("x"), Expr::Int(0)),
            vec![assign("result", Expr::Str("a".to_string()))],
            vec![if_stmt(
                eq_cmp(var("y"), Expr::Int(1)),
                vec![assign("result", Expr::Str("b".to_string()))],
                vec![],
            )],
        );

        let result = try_match(&stmt);
        assert!(
            result.is_none(),
            "mismatched discriminants should not match"
        );
    }

    #[test]
    fn single_arm_not_rewritten() {
        // if (x == 0) { result = "a"; }
        let stmt = if_stmt(
            eq_cmp(var("x"), Expr::Int(0)),
            vec![assign("result", Expr::Str("a".to_string()))],
            vec![],
        );

        let result = try_match(&stmt);
        assert!(result.is_none(), "single arm should not match");
    }

    #[test]
    fn multiple_statements_in_then_body_not_rewritten() {
        // if (x == 0) { result = "a"; extra_stmt; }
        // else if (x == 1) { result = "b"; }
        let stmt = if_stmt(
            eq_cmp(var("x"), Expr::Int(0)),
            vec![
                assign("result", Expr::Str("a".to_string())),
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "extra".to_string(),
                    args: vec![],
                }),
            ],
            vec![if_stmt(
                eq_cmp(var("x"), Expr::Int(1)),
                vec![assign("result", Expr::Str("b".to_string()))],
                vec![],
            )],
        );

        let result = try_match(&stmt);
        assert!(
            result.is_none(),
            "multiple statements in then_body should not match"
        );
    }

    #[test]
    fn non_equality_condition_not_rewritten() {
        // if (x > 0) { result = "a"; }
        // else if (x > 1) { result = "b"; }
        let stmt = if_stmt(
            Expr::Cmp(CmpKind::Gt, Box::new(var("x")), Box::new(Expr::Int(0))),
            vec![assign("result", Expr::Str("a".to_string()))],
            vec![if_stmt(
                Expr::Cmp(CmpKind::Gt, Box::new(var("x")), Box::new(Expr::Int(1))),
                vec![assign("result", Expr::Str("b".to_string()))],
                vec![],
            )],
        );

        let result = try_match(&stmt);
        assert!(result.is_none(), "non-equality condition should not match");
    }

    #[test]
    fn switch_expr_render() {
        // Test the rendering of a SwitchExpr.
        let switch = Expr::SwitchExpr {
            discrim: Box::new(var("x")),
            arms: vec![
                (Expr::Int(0), Expr::Str("zero".to_string())),
                (Expr::Int(1), Expr::Str("one".to_string())),
            ],
            default_arm: Some(Box::new(Expr::Str("other".to_string()))),
        };

        let mut output = String::new();
        switch.render(&mut output);

        assert!(
            output.contains("x switch {"),
            "should contain switch keyword"
        );
        assert!(output.contains("0 => \"zero\""), "should contain first arm");
        assert!(output.contains("1 => \"one\""), "should contain second arm");
        assert!(
            output.contains("_ => \"other\""),
            "should contain default arm"
        );
    }

    fn field(base: Expr, name: &str) -> Expr {
        Expr::Field(Box::new(base), name.to_string())
    }

    fn contains_call(recv: &str) -> Expr {
        Expr::Call {
            recv: Some(Box::new(var(recv))),
            name: "Contains".into(),
            args: vec![field(var("storeItemData"), "ProductId")],
        }
    }

    fn bit_and_one(e: Expr) -> Expr {
        Expr::Bin(BinOp::And, Box::new(e), Box::new(Expr::Int(1)))
    }

    fn bool_null_if() -> Stmt {
        if_stmt(
            Expr::Cmp(CmpKind::Eq, Box::new(var("v159")), Box::new(Expr::Null)),
            vec![assign("v23", Expr::Int(0))],
            vec![assign("v23", bit_and_one(contains_call("v159")))],
        )
    }

    fn or_zero_cond() -> Expr {
        Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::Bin(
                BinOp::Or,
                Box::new(var("v23")),
                Box::new(field(var("v146"), "IsStarterPack")),
            )),
            Box::new(Expr::Int(0)),
        )
    }

    fn or_zero_consumer() -> Stmt {
        if_stmt(
            or_zero_cond(),
            vec![assign("v26", Expr::Int(1))],
            vec![assign("v26", Expr::Int(2))],
        )
    }

    /// Same consumer in a loop condition: the generic switch matcher ignores
    /// loops, so negative tests isolate the bool-null rewrite.
    fn or_zero_loop() -> Stmt {
        Stmt::While {
            cond: or_zero_cond(),
            body: vec![],
        }
    }

    #[test]
    fn null_bool_test_becomes_null_conditional() {
        let mut stmts = vec![bool_null_if(), or_zero_consumer()];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "one bool null-test rewrite");

        // `v23 = v159?.Contains(storeItemData.ProductId) ?? false`
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::Bin(BinOp::NullCoalesce, nc, def)) => {
                assert_eq!(target, "v23");
                assert!(matches!(def.as_ref(), Expr::Bool(false)));
                match nc.as_ref() {
                    Expr::NullConditional { receiver, access } => {
                        assert!(matches!(receiver.as_ref(), Expr::Var(n) if n == "v159"));
                        assert!(
                            matches!(access.as_ref(), Expr::Call { name, .. } if name == "Contains")
                        );
                    }
                    other => panic!("expected NullConditional, got {other:?}"),
                }
            }
            other => panic!("expected Assign(NullCoalesce), got {other:?}"),
        }

        // `if (!(v23 || v146.IsStarterPack))`
        match &stmts[1] {
            Stmt::If { cond, .. } => {
                let Expr::Not(inner) = cond else {
                    panic!("expected negated condition, got {cond:?}");
                };
                assert!(matches!(
                    inner.as_ref(),
                    Expr::Bin(BinOp::OrElse, l, r)
                        if matches!(l.as_ref(), Expr::Var(n) if n == "v23")
                            && matches!(r.as_ref(), Expr::Field(_, f) if f == "IsStarterPack")
                ));
                let mut s = String::new();
                cond.render(&mut s);
                assert_eq!(s, "!(v23 || v146.IsStarterPack)");
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn single_null_test_is_not_a_switch() {
        // No consumer: the bool rewrites cannot fire, and the generic matcher
        // must refuse the one-null-arm switch shape.
        let mut stmts = vec![bool_null_if()];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(
            matches!(&stmts[0], Stmt::If { .. }),
            "a lone null arm must stay an if/else, got {:?}",
            stmts[0]
        );
    }

    #[test]
    fn null_bool_test_not_rewritten_when_target_has_other_reads() {
        let mut stmts = vec![bool_null_if(), Stmt::Eval(var("v23")), or_zero_loop()];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::If { .. }));
        assert!(matches!(
            &stmts[2],
            Stmt::While {
                cond: Expr::Cmp(..),
                ..
            }
        ));
    }

    #[test]
    fn null_bool_test_with_int_zero_uses_ternary() {
        // `v159 == 0` (not proven to be a reference) must not grow a `?.`.
        let cond = Expr::Cmp(CmpKind::Eq, Box::new(var("v159")), Box::new(Expr::Int(0)));
        let mut stmts = vec![
            if_stmt(
                cond,
                vec![assign("v23", Expr::Int(0))],
                vec![assign("v23", bit_and_one(contains_call("v159")))],
            ),
            or_zero_consumer(),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, None, Expr::Cond(test, val, def)) => {
                assert!(matches!(
                    test.as_ref(),
                    Expr::Cmp(CmpKind::Ne, _, b) if matches!(b.as_ref(), Expr::Int(0))
                ));
                assert!(
                    matches!(val.as_ref(), Expr::Call { .. }),
                    "call without & 1"
                );
                assert!(matches!(def.as_ref(), Expr::Bool(false)));
            }
            other => panic!("expected Cond, got {other:?}"),
        }
    }

    #[test]
    fn null_bool_test_rejects_extra_arm_statements() {
        let mut if_stmt = bool_null_if();
        if let Stmt::If { then_body, .. } = &mut if_stmt {
            then_body.push(Stmt::Eval(var("side_effect")));
        }
        let mut stmts = vec![if_stmt, or_zero_loop()];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn null_conditional_method_chain_renderer_strips_receiver() {
        let expr = Expr::NullConditional {
            receiver: Box::new(var("v159")),
            access: Box::new(Expr::MethodChain {
                receiver: Box::new(var("v159")),
                methods: vec![super::super::super::csharp::ChainMethod {
                    name: "Contains".into(),
                    args: vec![var("id")],
                }],
            }),
        };
        let mut s = String::new();
        expr.render(&mut s);
        assert_eq!(s, "v159?.Contains(id)");
    }

    #[test]
    fn full_pipeline_integration() {
        // Test the full reconstruction pipeline.
        let mut stmts = vec![if_stmt(
            eq_cmp(var("state"), Expr::Int(0)),
            vec![assign("name", Expr::Str("init".to_string()))],
            vec![if_stmt(
                eq_cmp(var("state"), Expr::Int(1)),
                vec![assign("name", Expr::Str("running".to_string()))],
                vec![if_stmt(
                    eq_cmp(var("state"), Expr::Int(2)),
                    vec![assign("name", Expr::Str("done".to_string()))],
                    vec![assign("name", Expr::Str("unknown".to_string()))],
                )],
            )],
        )];

        let count = apply(&mut stmts);
        assert_eq!(count, 1, "should rewrite one switch");

        match &stmts[0] {
            Stmt::Assign(
                target,
                _,
                Expr::SwitchExpr {
                    discrim,
                    arms,
                    default_arm,
                },
            ) => {
                assert_eq!(target, "name");
                assert!(matches!(*discrim.as_ref(), Expr::Var(ref n) if n == "state"));
                assert_eq!(arms.len(), 3);
                assert!(default_arm.is_some());
            }
            _ => panic!("expected Assign with SwitchExpr"),
        }
    }
}
