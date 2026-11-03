//! Iterator / coroutine state-machine reconstruction.
//!
//! ## The shape the structurer actually emits
//!
//! The structurer emits two shapes for the state-machine dispatch:
//!
//! 1. **If-ladder** — the raw shape for a jump table the structurer could not
//!    rebuild:
//!
//!    ```text
//!    this.<>1__state = -1;
//!    if (this.<>1__state == 0) { … }
//!    else if (this.<>1__state == 1) { … }
//!    else if (this.<>1__state == 2) { … }
//!    ```
//!
//! 2. **`Stmt::Switch`** — produced by Phase 4A (`switch_recon`) when the
//!    if-ladder has 3+ arms. Phase 4A converts the chain to a proper switch,
//!    which this pass then recognizes as a state-machine dispatch.
//!
//! Both spellings are accepted here, and both must satisfy the SAME proof.
//!
//! ## The proof (relaxed constraints)
//!
//! Deleting the state assignments is what makes this rewrite worth doing, and
//! it is only allowed when the state slot carries no other meaning. That is a
//! property of the whole method, so:
//!
//!   1. **the dispatch must be the last *meaningful* statement.** The body may
//!      carry a bookkeeping prefix (comments and `<slot> = <int literal>`
//!      stores) before the dispatch, and a bookkeeping trailer (comments,
//!      `return` statements, and `<slot> = <int literal>` stores) after it.
//!      Anything else — real code before OR after the dispatch — is refused,
//!      because flattening the dispatch would reorder it;
//!   2. **the slot must be pure bookkeeping.** Anywhere in the whole body, the
//!      slot may appear only as `<slot> = <int literal>` or as a test of the
//!      dispatch itself. One read inside an arm, one store from a non-literal,
//!      one mention in a nested condition: all refuse the whole switch;
//!   3. **every arm must convert.** Each arm may contain a state store, a
//!      `<>2__current` store, a bool-valued return, and otherwise only pure
//!      statements. Any control flow in an arm (`if`/loop/`try`/`switch`/
//!      `goto`/`label`/`break`/`continue`) refuses the whole switch, because
//!      flattening an arm across the switch boundary would move control flow.
//!      An arm does NOT need a state store if it has other yield evidence
//!      (a `<>2__current` store or a boolean return);
//!   4. **at least one value must be yielded** — a state dispatch with no
//!      `yield return` is an ordinary switch that happens to test a `__state`
//!      field, and is left alone.
//!
//! A failed match is a no-op: the correct (lower-level) dispatch stays.

use super::super::csharp::{CmpKind, Expr, Stmt};
use super::render_text;

/// `yield return v;` / `yield break;` an arm reconstructs.
enum YieldKind {
    Return(Expr),
    Break,
}

/// Run the recognizer over one method body.
///
/// Only the TOP level is considered: a `yield` is a method-body-level construct
/// (it is what turns the method into a state machine), so a state dispatch that
/// is merely nested inside another construct is not this method's suspension
/// points and is left alone.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let Some(seq) = convert_dispatch(stmts) else {
        return 0;
    };
    *stmts = seq;
    1
}

/// Convert the whole-body state dispatch, or `None`.
fn convert_dispatch(list: &[Stmt]) -> Option<Vec<Stmt>> {
    // A bookkeeping prefix: comments and `<slot> = <int literal>` stores.
    let mut head = None;
    for (i, s) in list.iter().enumerate() {
        match s {
            Stmt::Comment(_) | Stmt::Attribute { .. } => continue,
            Stmt::Assign(n, _, Expr::Int(_)) if is_state_disc(n) => continue,
            Stmt::Switch { disc, .. } if is_state_disc(disc) => {
                head = Some(i);
                break;
            }
            Stmt::If { cond, .. } if state_test(cond).is_some() => {
                head = Some(i);
                break;
            }
            _ => return None,
        }
    }
    let head = head?;

    // The conversion replaces the dispatch (and its bookkeeping) with a flat
    // yield sequence. Any label inside that region would be deleted, so refuse:
    // labels are jump targets and are never dropped by a rewrite.
    let mut labels: std::collections::HashSet<String> = std::collections::HashSet::new();
    super::collect_labels(std::slice::from_ref(&list[head]), &mut labels);
    if !labels.is_empty() {
        return None;
    }

    // RELAXED: the dispatch need not be the absolute last statement. Trailing
    // bookkeeping (comments, returns, state stores) is allowed after it — these
    // are compiler machinery (`return false;` = MoveNext exhausted, state stores
    // = bookkeeping) that the yield sequence subsumes. Real code after the
    // dispatch is refused because flattening would reorder it.
    for s in &list[head + 1..] {
        if !is_trailing_bookkeeping(s) {
            return None;
        }
    }

    let (slot, cases) = dispatch_cases(&list[head])?;
    if cases.len() < 2 {
        return None;
    }
    // 2. the whole-slot purity proof.
    if !slot_is_pure_bookkeeping(list, head, &slot) {
        return None;
    }

    // 3. every arm must convert.
    let mut ordered = cases;
    ordered.sort_by_key(|(v, _)| *v);
    // Comments/attributes are not semantics; keep them in place instead of
    // deleting them with the dispatch machinery.
    let mut seq: Vec<Stmt> = list[..head]
        .iter()
        .filter(|s| matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. }))
        .cloned()
        .collect();
    let mut saw_return = false;
    for (_v, body) in &ordered {
        let (residual, kind) = convert_arm(body, &slot)?;
        seq.extend(residual);
        match kind {
            YieldKind::Return(e) => {
                saw_return = true;
                seq.push(Stmt::YieldReturn(e));
            }
            YieldKind::Break => seq.push(Stmt::YieldBreak),
        }
    }
    // 4. without a single `yield return` this is not an iterator.
    if !saw_return {
        return None;
    }
    seq.extend(
        list[head + 1..]
            .iter()
            .filter(|s| matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. }))
            .cloned(),
    );
    Some(seq)
}

/// The `(slot, cases)` of a dispatch statement — a `switch`, or an if-ladder
/// whose every test is `slot == <literal>`.
fn dispatch_cases(s: &Stmt) -> Option<(String, Vec<(u64, Vec<Stmt>)>)> {
    match s {
        Stmt::Switch {
            disc,
            cases,
            tail,
            default,
        } => {
            if !is_state_disc(disc) {
                return None;
            }
            // RELAXED: the tail (landing pads) and the default arm may contain
            // bookkeeping — returns and state stores — not just
            // comments/labels. Phase 4A puts the final else body (often
            // `return false;`) into the default arm. This is compiler machinery
            // that the yield sequence subsumes.
            for s in tail.iter().chain(default.iter().flatten()) {
                if !is_trailing_bookkeeping(s) && !matches!(s, Stmt::Label(_)) {
                    return None;
                }
            }
            Some((disc.clone(), cases.clone()))
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            let mut slot: Option<String> = None;
            let mut cases: Vec<(u64, Vec<Stmt>)> = Vec::new();
            if !ladder_cases(&mut slot, cond, then_body, else_body, &mut cases) {
                return None;
            }
            Some((slot?, cases))
        }
        _ => None,
    }
}

/// Walk one level of an if-ladder, collecting `(state value, arm)` pairs.
///
/// Every level must test the SAME slot with `==` against a distinct integer
/// literal. The chain must end in an empty else, or an else containing only
/// bookkeeping (returns, state stores, comments) — the terminal else is the
/// iterator's "exhausted" path (`return false;`).
/// A `!=` test is refused: the arm under it is not a single state.
fn ladder_cases(
    slot: &mut Option<String>,
    cond: &Expr,
    then_body: &[Stmt],
    else_body: &[Stmt],
    cases: &mut Vec<(u64, Vec<Stmt>)>,
) -> bool {
    let Some((s2, k)) = state_test(cond) else {
        return false;
    };
    // Every level must test the SAME slot. Checked before the update so the two
    // uses of `s2` never mix a move with a borrow.
    if slot.as_deref().is_some_and(|s| s != s2) {
        return false;
    }
    if slot.is_none() {
        *slot = Some(s2);
    }
    if cases.iter().any(|(v, _)| *v == k) {
        return false;
    }
    cases.push((k, then_body.to_vec()));
    let meaningful: Vec<&Stmt> = else_body
        .iter()
        .filter(|s| !matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. } | Stmt::Label(_)))
        .collect();
    match meaningful.as_slice() {
        [] => true,
        [
            Stmt::If {
                cond,
                then_body,
                else_body,
            },
        ] => ladder_cases(slot, cond, then_body, else_body, cases),
        // RELAXED: the terminal else may contain bookkeeping (returns, state
        // stores) — this is the iterator's "exhausted" path.
        _ if meaningful.iter().all(|s| is_trailing_bookkeeping(s)) => true,
        _ => false,
    }
}

/// `slot == <int literal>`, with the slot being a compiler state field.
fn state_test(cond: &Expr) -> Option<(String, u64)> {
    let Expr::Cmp(CmpKind::Eq, a, b) = cond else {
        return None;
    };
    let (slot, k) = match (a.as_ref(), b.as_ref()) {
        (slot, Expr::Int(k)) => (slot, *k),
        (Expr::Int(k), slot) => (slot, *k),
        _ => return None,
    };
    if k < 0 {
        return None;
    }
    let text = render_text(slot);
    if is_state_disc(&text) {
        Some((text, k as u64))
    } else {
        None
    }
}

/// Is this the compiler's iterator/coroutine state slot (`<>1__state`)?
///
/// The mangle marker is required so a user's own `switch (gameState)` can never
/// be mistaken for compiler machinery.
fn is_state_disc(name: &str) -> bool {
    name.contains("__state")
}

/// Is this the field a `yield return` value is parked in (`<>2__current`)?
fn is_current(name: &str) -> bool {
    name.to_ascii_lowercase().contains("current")
}

/// Is this the `MoveNext` return, which carries no yielded value?
fn is_bool_like(e: &Expr) -> bool {
    matches!(e, Expr::Bool(_) | Expr::Int(0) | Expr::Int(1))
}

/// Is `s` safe to appear as trailing bookkeeping after the dispatch?
///
/// Trailing bookkeeping is compiler machinery that the yield sequence subsumes:
/// - Comments (no effect)
/// - `return` statements (MoveNext's return value, subsumed by yield break)
/// - State stores `<slot> = <int literal>` (bookkeeping, subsumed by the
///   state-machine protocol)
///
/// Anything else — real code, control flow, calls — is NOT bookkeeping and
/// would be lost by the flattening, so it refuses the dispatch.
fn is_trailing_bookkeeping(s: &Stmt) -> bool {
    match s {
        Stmt::Comment(_) | Stmt::Attribute { .. } => true,
        // `return false;` / `return true;` is MoveNext's exhaustion result and
        // is subsumed by the yield sequence. A non-boolean return is REAL
        // code: dropping it would delete a value the method returns.
        Stmt::Return(None) => true,
        Stmt::Return(Some(e)) if is_bool_like(e) => true,
        Stmt::Assign(n, _, Expr::Int(_)) if is_state_disc(n) => true,
        _ => false,
    }
}

/// `true` when `hay` contains `needle` as a whole identifier.
fn contains_ident(hay: &str, needle: &str) -> bool {
    let mut from = 0usize;
    while let Some(pos) = hay[from..].find(needle) {
        let start = from + pos;
        let end = start + needle.len();
        let before_ok = start == 0
            || !hay[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after_ok = end >= hay.len()
            || !hay[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// Does `e` mention the slot (as an identifier)?
fn expr_mentions_slot(e: &Expr, slot: &str) -> bool {
    contains_ident(&render_text(e), slot)
}

/// Is `s` the bookkeeping store `<slot> = <int literal>`?
fn is_state_store(s: &Stmt, slot: &str) -> bool {
    matches!(s, Stmt::Assign(n, _, Expr::Int(_)) if n == slot)
}

/// Does `s` touch the slot in any way that is NOT a bookkeeping store?
fn touches_slot(s: &Stmt, slot: &str) -> bool {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => expr_mentions_slot(e, slot),
        Stmt::Assign(n, _, e) => n == slot || expr_mentions_slot(e, slot),
        Stmt::If { cond, .. } => expr_mentions_slot(cond, slot),
        Stmt::While { cond, .. } => expr_mentions_slot(cond, slot),
        Stmt::DoWhile { cond, .. } => expr_mentions_slot(cond, slot),
        Stmt::For {
            cond, init, incr, ..
        } => {
            cond.as_ref().is_some_and(|c| expr_mentions_slot(c, slot))
                || list_touches_slot_deep(init, slot)
                || list_touches_slot_deep(incr, slot)
        }
        Stmt::Lock { expr, .. } => expr_mentions_slot(expr, slot),
        Stmt::Using { decl, .. } => touches_slot(decl, slot),
        Stmt::Switch { disc, .. } => disc == slot,
        Stmt::ForEach {
            var_name,
            collection,
            ..
        } => var_name == slot || collection == slot,
        Stmt::Return(Some(e)) => expr_mentions_slot(e, slot),
        _ => false,
    }
}

/// Does any statement in `list`, or in any nested body, touch the slot in a way
/// that is not a bookkeeping store?
fn list_touches_slot_deep(list: &[Stmt], slot: &str) -> bool {
    for s in list {
        if is_state_store(s, slot) {
            continue;
        }
        if touches_slot(s, slot) {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            if list_touches_slot_deep(k, slot) {
                return true;
            }
        }
    }
    false
}

/// The whole-body purity proof: the slot may only be stored from an integer
/// literal, or tested by the dispatch itself.
fn slot_is_pure_bookkeeping(list: &[Stmt], head: usize, slot: &str) -> bool {
    for (i, s) in list.iter().enumerate() {
        if i == head {
            // The dispatch's own tests ARE the dispatch; its arms are checked
            // for any other mention of the slot.
            if !dispatch_arms_are_pure(s, slot) {
                return false;
            }
            continue;
        }
        if list_touches_slot_deep(std::slice::from_ref(s), slot) {
            return false;
        }
    }
    true
}

/// Are the arms of a dispatch statement pure w.r.t. the slot?
fn dispatch_arms_are_pure(s: &Stmt, slot: &str) -> bool {
    let Some((_slot, cases)) = dispatch_cases(s) else {
        return false;
    };
    cases
        .iter()
        .all(|(_, body)| !list_touches_slot_deep(body, slot))
}

/// Convert one state arm: drop the state assignment and the `MoveNext` return,
/// fold the parked `<>2__current` (or a non-boolean return value) into a
/// `yield`, and keep every other statement in front of it.
///
/// Refuses any arm containing control flow (nested `if`/loop/`try`/`switch`, or
/// a `goto`/`label`/`break`/`continue`) — flattening such an arm across a
/// switch boundary would move control flow, which rule 2 forbids.
///
/// RELAXED: an arm does NOT need a state store if it has other yield evidence:
/// - A `<>2__current` store is sufficient for `yield return`
/// - A boolean return (`return false;`) without any value is sufficient for
///   `yield break`
fn convert_arm(body: &[Stmt], slot: &str) -> Option<(Vec<Stmt>, YieldKind)> {
    let mut residual: Vec<Stmt> = Vec::new();
    let mut current: Option<Expr> = None;
    let mut ret_value: Option<Expr> = None;
    let mut state: Option<i64> = None;
    let mut has_bool_return = false;

    for s in body {
        match s {
            Stmt::Assign(n, _, Expr::Int(v)) if n == slot => state = Some(*v),
            Stmt::Assign(n, _, e) if is_current(n) => current = Some(e.clone()),
            Stmt::Return(Some(e)) => {
                if is_bool_like(e) {
                    has_bool_return = true;
                } else {
                    ret_value = Some(e.clone());
                }
            }
            Stmt::Return(None) => {}
            Stmt::Break
            | Stmt::Continue
            | Stmt::Goto(_)
            | Stmt::Label(_)
            | Stmt::If { .. }
            | Stmt::While { .. }
            | Stmt::DoWhile { .. }
            | Stmt::For { .. }
            | Stmt::ForEach { .. }
            | Stmt::Try { .. }
            | Stmt::Lock { .. }
            | Stmt::Using { .. }
            | Stmt::Switch { .. } => return None,
            other => residual.push(other.clone()),
        }
    }

    let kind = match current.or(ret_value) {
        // A `<>2__current` store or a non-boolean return value is a yield return.
        Some(v) => YieldKind::Return(v),
        None => {
            // No value parked. Determine yield break from state or return.
            match state {
                // A negative next state is the iterator's completion
                // (`yield break` / `MoveNext` returns false).
                Some(v) if v < 0 => YieldKind::Break,
                // RELAXED: no state store, but a boolean return is yield break
                // evidence. This handles arms that just do `return false;`
                // without an explicit state transition.
                None if has_bool_return => YieldKind::Break,
                // A non-negative state with no value is a plain transition we
                // do not model.
                _ => return None,
            }
        }
    };
    Some((residual, kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATE: &str = "this.<>1__state";
    const CUR: &str = "this.<>2__current";

    fn set_state(v: i64) -> Stmt {
        Stmt::Assign(STATE.to_string(), None, Expr::Int(v))
    }
    fn state_test_expr(v: i64) -> Expr {
        Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::Var(STATE.into())),
            Box::new(Expr::Int(v)),
        )
    }
    /// One arm that parks `x` and advances to `1`.
    fn yield_zero_arm() -> Vec<Stmt> {
        vec![
            set_state(1),
            Stmt::Assign(CUR.to_string(), None, Expr::Var("x".into())),
            Stmt::Return(Some(Expr::Bool(true))),
        ]
    }
    fn break_arm() -> Vec<Stmt> {
        vec![set_state(-1), Stmt::Return(Some(Expr::Bool(false)))]
    }

    /// The if-LADDER dispatch — the shape the structurer actually emits — folds
    /// to a yield sequence.
    #[test]
    fn a_state_ladder_becomes_yields() {
        let mut body = vec![
            Stmt::Assign(STATE.to_string(), None, Expr::Int(-1)),
            Stmt::If {
                cond: state_test_expr(0),
                then_body: yield_zero_arm(),
                else_body: vec![Stmt::If {
                    cond: state_test_expr(1),
                    then_body: break_arm(),
                    else_body: vec![],
                }],
            },
        ];
        assert_eq!(apply(&mut body), 1);
        assert_eq!(body.len(), 2, "the bookkeeping store is gone too");
        assert!(matches!(&body[0], Stmt::YieldReturn(e) if matches!(e, Expr::Var(n) if n == "x")));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// The `switch` spelling still folds.
    #[test]
    fn a_state_switch_becomes_yields() {
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
            tail: vec![],
        
            default: None,}];
        assert_eq!(apply(&mut body), 1);
        assert!(matches!(&body[0], Stmt::YieldReturn(_)));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// Real code AFTER the dispatch would be lost by flattening — refuse.
    /// (Note: `return false;` is now allowed as trailing bookkeeping; only
    /// non-bookkeeping statements like `Eval` are refused.)
    #[test]
    fn content_after_the_dispatch_refuses() {
        let mut body = vec![
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
                tail: vec![],
            
                default: None,},
            Stmt::Eval(Expr::Var("real_code".into())),
        ];
        assert_eq!(apply(&mut body), 0);
        assert!(matches!(&body[0], Stmt::Switch { .. }));
    }

    /// The state is consulted for REAL logic inside an arm: the bookkeeping
    /// store may not be deleted, so nothing is rewritten.
    #[test]
    fn a_real_state_test_inside_an_arm_refuses() {
        let mut arm = yield_zero_arm();
        arm.push(Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var(STATE.into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![],
            else_body: vec![],
        });
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, arm), (1u64, break_arm())],
            tail: vec![],
        
            default: None,}];
        assert_eq!(apply(&mut body), 0);
    }

    /// A non-state discriminant is never touched.
    #[test]
    fn an_ordinary_switch_is_left_alone() {
        let mut body = vec![Stmt::Switch {
            disc: "gameState".to_string(),
            cases: vec![
                (0u64, vec![Stmt::Assign("v".into(), None, Expr::Int(1))]),
                (1u64, vec![Stmt::Assign("v".into(), None, Expr::Int(2))]),
            ],
            tail: vec![],
        
            default: None,}];
        assert_eq!(apply(&mut body), 0);
    }

    /// A state dispatch with no `yield return` at all is an ordinary switch.
    #[test]
    fn a_dispatch_without_a_value_is_refused() {
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, break_arm()), (1u64, break_arm())],
            tail: vec![],
        
            default: None,}];
        assert_eq!(apply(&mut body), 0);
    }

    /// An arm with control flow would move meaning under the flattening.
    #[test]
    fn an_arm_with_control_flow_is_refused() {
        let mut arm = yield_zero_arm();
        arm.insert(
            0,
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![],
                else_body: vec![],
            },
        );
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, arm), (1u64, break_arm())],
            tail: vec![],
        
            default: None,}];
        assert_eq!(apply(&mut body), 0);
    }

    /// A ladder that tests a NON-state local is not compiler machinery.
    #[test]
    fn a_ladder_on_a_user_variable_is_refused() {
        let mut body = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Var("mode".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: yield_zero_arm(),
            else_body: vec![],
        }];
        assert_eq!(apply(&mut body), 0);
    }

    /// A `!=` test cannot name a single state, so the ladder is refused.
    #[test]
    fn a_negative_state_test_is_refused() {
        let mut body = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var(STATE.into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: yield_zero_arm(),
            else_body: vec![],
        }];
        assert_eq!(apply(&mut body), 0);
    }

    #[test]
    fn identifier_matching_respects_boundaries() {
        assert!(contains_ident("this.<>1__state", "<>1__state"));
        assert!(!contains_ident("this.<>1__stateExtra", "<>1__state"));
        assert!(!contains_ident("v12", "v1"));
    }

    // -----------------------------------------------------------------------
    // Phase 4B: relaxed constraints
    // -----------------------------------------------------------------------

    /// An if-ladder with 3 states folds to a yield sequence.
    #[test]
    fn three_state_ladder_becomes_yields() {
        let mut body = vec![
            Stmt::Assign(STATE.to_string(), None, Expr::Int(-1)),
            Stmt::If {
                cond: state_test_expr(0),
                then_body: yield_zero_arm(),
                else_body: vec![Stmt::If {
                    cond: state_test_expr(1),
                    then_body: vec![
                        set_state(2),
                        Stmt::Assign(CUR.to_string(), None, Expr::Var("y".into())),
                        Stmt::Return(Some(Expr::Bool(true))),
                    ],
                    else_body: vec![Stmt::If {
                        cond: state_test_expr(2),
                        then_body: break_arm(),
                        else_body: vec![],
                    }],
                }],
            },
        ];
        assert_eq!(apply(&mut body), 1);
        assert_eq!(body.len(), 3);
        assert!(matches!(&body[0], Stmt::YieldReturn(_)));
        assert!(matches!(&body[1], Stmt::YieldReturn(_)));
        assert!(matches!(&body[2], Stmt::YieldBreak));
    }

    /// Trailing `return false;` after the dispatch is bookkeeping — accepted.
    #[test]
    fn trailing_return_after_dispatch_is_accepted() {
        let mut body = vec![
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
                tail: vec![],
            
                default: None,},
            Stmt::Return(Some(Expr::Bool(false))),
        ];
        assert_eq!(apply(&mut body), 1);
        assert!(matches!(&body[0], Stmt::YieldReturn(_)));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// Trailing state store + return after the dispatch is bookkeeping.
    #[test]
    fn trailing_state_store_and_return_are_accepted() {
        let mut body = vec![
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
                tail: vec![],
            
                default: None,},
            Stmt::Assign(STATE.to_string(), None, Expr::Int(-1)),
            Stmt::Return(Some(Expr::Bool(false))),
        ];
        assert_eq!(apply(&mut body), 1);
        assert!(matches!(&body[0], Stmt::YieldReturn(_)));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// A switch with a non-empty tail (default case with `return false;`) is
    /// accepted — the tail is compiler machinery.
    #[test]
    fn switch_with_default_return_is_accepted() {
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
            tail: vec![Stmt::Return(Some(Expr::Bool(false)))],
        
            default: None,}];
        assert_eq!(apply(&mut body), 1);
        assert!(matches!(&body[0], Stmt::YieldReturn(_)));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// An if-ladder whose final else contains `return false;` is accepted.
    #[test]
    fn ladder_with_trailing_return_in_final_else() {
        let mut body = vec![
            Stmt::If {
                cond: state_test_expr(0),
                then_body: yield_zero_arm(),
                else_body: vec![Stmt::If {
                    cond: state_test_expr(1),
                    then_body: break_arm(),
                    else_body: vec![Stmt::Return(Some(Expr::Bool(false)))],
                }],
            },
        ];
        assert_eq!(apply(&mut body), 1);
        assert!(matches!(&body[0], Stmt::YieldReturn(_)));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// An arm with `<>2__current` but NO state store is accepted as yield return.
    #[test]
    fn arm_without_state_store_but_with_current_is_accepted() {
        let arm_no_state = vec![
            Stmt::Assign(CUR.to_string(), None, Expr::Var("val".into())),
            Stmt::Return(Some(Expr::Bool(true))),
        ];
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, arm_no_state), (1u64, break_arm())],
            tail: vec![],
        
            default: None,}];
        assert_eq!(apply(&mut body), 1);
        assert!(matches!(&body[0], Stmt::YieldReturn(e) if matches!(e, Expr::Var(n) if n == "val")));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// An arm with just `return false;` (no state store, no `<>2__current`) is
    /// accepted as yield break.
    #[test]
    fn arm_with_only_bool_return_is_yield_break() {
        let bare_return_arm = vec![Stmt::Return(Some(Expr::Bool(false)))];
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, yield_zero_arm()), (1u64, bare_return_arm)],
            tail: vec![],
        
            default: None,}];
        assert_eq!(apply(&mut body), 1);
        assert!(matches!(&body[0], Stmt::YieldReturn(_)));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// Multiple bookkeeping prefix statements before the dispatch are accepted,
    /// and a comment in the prefix is preserved rather than deleted with the
    /// state machinery.
    #[test]
    fn multiple_bookkeeping_prefix_statements() {
        let mut body = vec![
            Stmt::Assign(STATE.to_string(), None, Expr::Int(-1)),
            Stmt::Comment("state machine entry".into()),
            Stmt::Assign(STATE.to_string(), None, Expr::Int(0)),
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
                tail: vec![],
            
                default: None,},
        ];
        assert_eq!(apply(&mut body), 1);
        assert_eq!(body.len(), 3, "comment + two yields");
        assert!(matches!(&body[0], Stmt::Comment(c) if c == "state machine entry"));
        assert!(matches!(&body[1], Stmt::YieldReturn(_)));
        assert!(matches!(&body[2], Stmt::YieldBreak));
    }

    /// A non-state-machine method (no `__state` field) is never affected, even
    /// with an if-ladder on a user variable.
    #[test]
    fn non_state_machine_method_is_not_affected() {
        let mut body = vec![
            Stmt::Assign("mode".to_string(), None, Expr::Int(0)),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::Var("mode".into())),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Assign("v".into(), None, Expr::Int(1))],
                else_body: vec![Stmt::If {
                    cond: Expr::Cmp(
                        CmpKind::Eq,
                        Box::new(Expr::Var("mode".into())),
                        Box::new(Expr::Int(1)),
                    ),
                    then_body: vec![Stmt::Assign("v".into(), None, Expr::Int(2))],
                    else_body: vec![],
                }],
            },
        ];
        let original = body.clone();
        assert_eq!(apply(&mut body), 0);
        assert_eq!(body.len(), original.len());
    }

    /// A trailing non-bookkeeping statement (a call) after the dispatch refuses.
    #[test]
    fn trailing_call_after_dispatch_refuses() {
        let mut body = vec![
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
                tail: vec![],
            
                default: None,},
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Cleanup".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut body), 0);
    }

    /// A trailing NON-boolean return is real code: the flattening would delete
    /// the value the method returns. Refuse.
    #[test]
    fn trailing_non_bool_return_refuses() {
        let mut body = vec![
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
                tail: vec![],
            
                default: None,},
            Stmt::Return(Some(Expr::Int(42))),
        ];
        assert_eq!(apply(&mut body), 0);
        assert!(matches!(&body[0], Stmt::Switch { .. }));
    }

    /// A label inside the dispatch is a jump target: converting would delete
    /// it, so the dispatch is left alone.
    #[test]
    fn a_label_inside_the_dispatch_refuses() {
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
            tail: vec![Stmt::Label("L_shared".into())],
        
            default: None,}];
        assert_eq!(apply(&mut body), 0);
        assert!(matches!(&body[0], Stmt::Switch { .. }));
    }

    /// Trailing comments survive the conversion.
    #[test]
    fn trailing_comment_is_preserved() {
        let mut body = vec![
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
                tail: vec![],
            
                default: None,},
            Stmt::Comment("done".into()),
        ];
        assert_eq!(apply(&mut body), 1);
        assert_eq!(body.len(), 3);
        assert!(matches!(&body[2], Stmt::Comment(c) if c == "done"));
    }
}
