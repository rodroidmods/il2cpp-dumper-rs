//! Late guard / redundant-assignment de-duplication.
//!
//! The structuring passes (and the branch repairs that follow them) can leave
//! the same null test nested inside itself, and the same temp initialization
//! repeated at the head of every arm it dominates:
//!
//! ```text
//! v26 = 0;                       // preheader
//! if (v146 != null) {
//!     v26 = 0;                   // redundant: nothing reads/writes v26
//!     if (this.badge != 0) {
//!         v159 = ...;
//!         if (this.badge != 0) { // redundant: same test, pure, nothing writes badge
//!             ...
//!         }
//!     }
//! }
//! ```
//!
//! Three conservative rewrites are implemented:
//!
//! * **A. nested same-polarity guard folding** — an `if (COND)` nested inside
//!   `if (COND) { ... }` (through any chain of if/else arms, not just as a
//!   direct child) is replaced by its then-body when COND is pure,
//!   structurally equal after [`normalize_cond`], no statement on the path in
//!   between writes a name COND mentions, and (for a COND that reads memory)
//!   no call can have run in between. An inner `if` that carries an else is
//!   left as the nested `if` (dropping its else arm would change the program).
//! * **B. impossible else-if** — a direct child `if (COND)` inside the else arm
//!   of `if (COND)` is unreachable; it is deleted when its own else is empty,
//!   COND is pure, and nothing in between writes COND's operands.
//! * **C. redundant identical re-assignment** — `X = V; ...; X = V;` where the
//!   second statement is in a dominated branch, V is pure and stable (no
//!   calls, no memory reads), V does not mention X, and nothing between reads
//!   or writes X or a name V depends on.
//!
//! ## Safety rules
//!
//! * A statement that contains a [`Stmt::Label`] (directly or nested) is never
//!   deleted. A label is also an entry barrier for the value tracking: control
//!   can jump to it, so an assignment before the label is not known to have
//!   executed on that path.
//! * A condition containing [`Expr::Call`], [`Expr::RawCall`],
//!   [`Expr::NewObj`] (and, more conservatively, any allocation / await) is
//!   never folded or removed.
//! * A local whose address is taken anywhere in the method
//!   ([`Expr::AddressOf`] / [`Expr::RawAddr`]) is never used as a fold condition
//!   or a de-duplication target.
//! * Calls between the two positions: rule A allows them only when COND reads
//!   no memory. A call cannot rebind a local, but it can write a field (a
//!   reconstructed property getter is an [`Expr::Field`] too), so a
//!   field/static-field condition requires the whole intervening path to be
//!   call-free. This is stricter than the rest of the pipeline (which accepts
//!   that a call may mutate a field through another reference) and is
//!   deliberate: this pass has no aliasing proof. Rule B retains its existing
//!   looser call policy.
//!
//! Everything is a no-op when a proof fails: the flat, correct sequence stays.

use std::collections::{HashMap, HashSet};

use super::super::csharp::{BinOp, CmpKind, Expr, Stmt};
use super::{children, children_mut, expr_has_call, expr_mentions, il2cpp_cleanup};

/// Run the guard / redundant-assignment de-duplication over one method body.
///
/// Returns the number of rewrites (each folded guard, removed impossible guard
/// and deleted redundant assignment counts as one).
pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut addr_taken = HashSet::new();
    collect_addr_taken(stmts, &mut addr_taken);
    let mut known: HashMap<String, Expr> = HashMap::new();
    let mut count = 0;
    process_list(stmts, &mut known, &addr_taken, &mut count);
    count
}

// ===========================================================================
// Statement-list walker
// ===========================================================================

fn process_list(
    stmts: &mut Vec<Stmt>,
    known: &mut HashMap<String, Expr>,
    addr_taken: &HashSet<String>,
    count: &mut usize,
) {
    let mut i = 0;
    while i < stmts.len() {
        if matches!(stmts[i], Stmt::If { .. }) {
            // Names touched / written by the whole if, for the parent map.
            let touched = names_touched_deep(&stmts[i]);
            let written = names_written_deep(&stmts[i]);

            if let Stmt::If {
                cond,
                then_body,
                else_body,
            } = &mut stmts[i]
            {
                // A and B reshape the arms before C walks them.
                fold_nested_same_cond(then_body, cond, addr_taken, count);
                remove_impossible_else_if(else_body, cond, addr_taken, count);

                let cond_names = names_in_expr(cond);
                let mut then_known = known.clone();
                let mut else_known = known.clone();
                // The test reads these; a read between two equal assignments
                // is enough to disqualify the second one (rule C).
                invalidate_keys(&mut then_known, &cond_names);
                invalidate_keys(&mut else_known, &cond_names);
                process_list(then_body, &mut then_known, addr_taken, count);
                process_list(else_body, &mut else_known, addr_taken, count);
            }

            invalidate_keys(known, &touched);
            invalidate_dependents(known, &written);
            i += 1;
            continue;
        }

        // Rule C: an assignment may be a duplicate of an earlier one.
        let assign = match &stmts[i] {
            Stmt::Assign(name, ty, rhs) => Some((name.clone(), ty.clone(), rhs.clone())),
            _ => None,
        };
        if let Some((name, ty, rhs)) = assign {
            let simple = ty.is_none()
                && is_simple_local(&name)
                && !addr_taken.contains(&name)
                && is_stable_expr(&rhs)
                && !expr_mentions(&rhs, &name)
                && !expr_reads_addr_taken(&rhs, addr_taken);

            if simple && known.get(&name) == Some(&rhs) {
                // The target is unchanged between the two writes, V is a pure
                // stable expression, and no label can enter between them (a
                // label would have cleared `known`). The statement is dead.
                stmts.remove(i);
                *count += 1;
                continue;
            }

            // The statement itself reads the names in its RHS; those are
            // reads "between" the establishment and any later duplicate.
            invalidate_keys(known, &names_in_expr(&rhs));
            // It writes `name`, so earlier values and anything depending on
            // them are no longer established.
            invalidate_keys(known, &HashSet::from([name.clone()]));
            invalidate_dependents(known, &HashSet::from([name.clone()]));
            if simple {
                known.insert(name, rhs);
            }
            i += 1;
            continue;
        }

        match &mut stmts[i] {
            // A label is a re-entry point: nothing established before it is
            // guaranteed on a path that jumps here.
            Stmt::Label(_) => known.clear(),
            // Calls cannot rebind a local, and stable RHS expressions read no
            // fields, so a call cannot perturb what `known` records. Its
            // arguments are still reads of the names they mention.
            Stmt::Eval(expr) => {
                let reads = names_in_expr(expr);
                invalidate_keys(known, &reads);
            }
            Stmt::Comment(_) => {}
            Stmt::Attribute { args, .. } => {
                let mut reads = HashSet::new();
                for arg in args {
                    reads.extend(names_in_expr(arg));
                }
                invalidate_keys(known, &reads);
            }
            // Any other construct owns control flow (a loop can run its body
            // repeatedly, a switch can enter at a case). Its nested lists are
            // still processed, but only with fresh knowledge; nothing is
            // carried in or out.
            other => {
                let mut fresh: HashMap<String, Expr> = HashMap::new();
                for child in children_mut(other) {
                    process_list(child, &mut fresh, addr_taken, count);
                }
                known.clear();
            }
        }
        i += 1;
    }
}

// ===========================================================================
// A. Nested same-polarity guard folding
// ===========================================================================

/// Proof state for everything that executes between the outer test and the
/// point currently being visited on the same path.
///
/// `writes` is set by an assignment to a name COND reads; `calls` by any call /
/// raw call / allocation (a call may rebind a field, but not a local).
#[derive(Clone, Copy, Default)]
struct FoldState {
    writes: bool,
    calls: bool,
}

fn fold_nested_same_cond(
    list: &mut Vec<Stmt>,
    outer_cond: &Expr,
    addr_taken: &HashSet<String>,
    count: &mut usize,
) {
    if !expr_is_pure_test(outer_cond) || cond_reads_addr_taken(outer_cond, addr_taken) {
        return;
    }
    let outer_key = normalize_cond(outer_cond);
    let names = names_in_expr(outer_cond);
    // Calls can rebind a field / mutate memory an index reaches, but never a
    // plain local. Conditions that read no memory may therefore fold across
    // them; field conditions may not.
    let calls_unstable = cond_reads_memory(outer_cond);
    fold_descendant_same_cond(
        list,
        &outer_key,
        &names,
        calls_unstable,
        FoldState::default(),
        count,
    );
}

/// Delete a redundant same-condition `if` shell nested anywhere below the
/// outer test, through any chain of if/else arms.
///
/// The scan keeps the existing label/flow-barrier rule: a statement whose
/// subtree contains a label stops the scan (the label is an entry point, so
/// what the outer test proved need not hold there), and an inner `if` holding
/// a label is never folded. `state` describes the straight-line path from the
/// outer test to `list`; a write to COND's names or (for memory-reading COND)
/// a call poisons every later point on that path.
fn fold_descendant_same_cond(
    list: &mut Vec<Stmt>,
    outer_key: &Expr,
    names: &HashSet<String>,
    calls_unstable: bool,
    mut state: FoldState,
    count: &mut usize,
) {
    let outer_neg = negate_cond(outer_key);
    let mut i = 0;
    while i < list.len() {
        // Reuse the existing label rule: a label (directly or nested) is an
        // entry barrier, and an `if` carrying one is never folded.
        if stmt_has_label(&list[i]) || is_flow_barrier(&list[i]) {
            break;
        }

        // An inner `if` on the same condition is dead-on-the-else: the outer
        // test already proved COND true here, so the else arm can never run.
        // `stmt_has_label` above guarantees the arm holds no label another
        // goto could enter through, so dropping it is safe.
        let foldable = matches!(
            &list[i],
            Stmt::If { cond, .. } if normalize_cond(cond) == *outer_key
        );
        if foldable {
            if state.writes || (calls_unstable && state.calls) {
                break;
            }
            let Stmt::If { then_body, .. } = list.remove(i) else {
                unreachable!()
            };
            // Splice the guarded body at the position of the inner `if`. The
            // removed test itself is pure, so the proof state is unchanged.
            list.splice(i..i, then_body);
            *count += 1;
            // Re-scan from the splice: a deeper duplicate folds next.
            continue;
        }

        // The proof is already broken; nothing later on this path can fold.
        if state.writes || (calls_unstable && state.calls) {
            break;
        }

        // Descend through if/else nesting only. Effects inside an arm happen
        // after any duplicate inside that arm, so they do not disqualify it;
        // they do disqualify the siblings that follow the whole `if`.
        if let Stmt::If {
            cond,
            then_body,
            else_body,
        } = &mut list[i]
        {
            let key = normalize_cond(cond);
            let contradicts_then = key == outer_neg;
            let contradicts_else = key == *outer_key;
            let arm_state = FoldState {
                writes: state.writes,
                calls: state.calls || expr_has_call(cond),
            };
            if !contradicts_then {
                fold_descendant_same_cond(
                    then_body,
                    outer_key,
                    names,
                    calls_unstable,
                    arm_state,
                    count,
                );
            }
            if !contradicts_else {
                fold_descendant_same_cond(
                    else_body,
                    outer_key,
                    names,
                    calls_unstable,
                    arm_state,
                    count,
                );
            }
            // A write in either arm may execute before later siblings.
            if then_body.iter().any(|s| stmt_writes_any(s, names))
                || else_body.iter().any(|s| stmt_writes_any(s, names))
            {
                break;
            }
            if calls_unstable
                && (expr_has_call(cond)
                    || then_body.iter().any(stmt_has_call)
                    || else_body.iter().any(stmt_has_call))
            {
                break;
            }
            i += 1;
            continue;
        }

        // Any other construct owns control flow (and is not an if/else
        // chain), so its body is not part of the straight-line path.
        if stmt_writes_any(&list[i], names) {
            break;
        }
        if stmt_has_call(&list[i]) {
            state.calls = true;
            if calls_unstable {
                break;
            }
        }
        i += 1;
    }
}

// ===========================================================================
// B. Impossible else-if
// ===========================================================================

fn remove_impossible_else_if(
    list: &mut Vec<Stmt>,
    outer_cond: &Expr,
    addr_taken: &HashSet<String>,
    count: &mut usize,
) {
    // On this path the outer test was false. An inner test of the SAME pure
    // condition can never succeed; deleting it (when it has no else) removes
    // only unreachable code.
    if !expr_is_pure_test(outer_cond) || cond_reads_addr_taken(outer_cond, addr_taken) {
        return;
    }
    let outer_key = normalize_cond(outer_cond);
    let names = names_in_expr(outer_cond);

    let mut i = 0;
    while i < list.len() {
        if stmt_has_label(&list[i]) || is_flow_barrier(&list[i]) {
            break;
        }
        if stmt_writes_any(&list[i], &names) {
            break;
        }

        let removable = match &list[i] {
            Stmt::If {
                cond, else_body, ..
            } => {
                else_body.is_empty()
                    && !stmt_has_label(&list[i])
                    && normalize_cond(cond) == outer_key
            }
            _ => false,
        };
        if removable {
            list.remove(i);
            *count += 1;
            continue;
        }
        i += 1;
    }
}

// ===========================================================================
// Condition normalization
// ===========================================================================

/// Canonical form for "same-polarity" comparisons.
///
/// `!` is pushed into comparisons (`!(x == 0)` → `x != 0`, De Morgan for
/// `&&`/`||`), double negation is removed, and a literal is placed on the
/// right of a comparison. This is the structural equality the task asks for:
/// `if (!COND)`, `if (COND == 0)` and their un-negated spellings compare by
/// the test they perform, not by their surface syntax.
fn normalize_cond(e: &Expr) -> Expr {
    match e {
        Expr::Not(inner) => negate_cond(&normalize_cond(inner)),
        Expr::Cmp(kind, a, b) => {
            let a = normalize_cond(a);
            let b = normalize_cond(b);
            if is_literal(&a) && !is_literal(&b) {
                Expr::Cmp(swap_cmp(*kind), Box::new(b), Box::new(a))
            } else {
                Expr::Cmp(*kind, Box::new(a), Box::new(b))
            }
        }
        Expr::Bin(op @ (BinOp::AndAlso | BinOp::OrElse), a, b) => Expr::Bin(
            *op,
            Box::new(normalize_cond(a)),
            Box::new(normalize_cond(b)),
        ),
        _ => e.clone(),
    }
}

fn negate_cond(e: &Expr) -> Expr {
    match e {
        Expr::Cmp(kind, a, b) => Expr::Cmp(invert_cmp(*kind), a.clone(), b.clone()),
        Expr::Bool(v) => Expr::Bool(!v),
        Expr::Not(inner) => normalize_cond(inner),
        Expr::Bin(BinOp::AndAlso, a, b) => Expr::Bin(
            BinOp::OrElse,
            Box::new(negate_cond(a)),
            Box::new(negate_cond(b)),
        ),
        Expr::Bin(BinOp::OrElse, a, b) => Expr::Bin(
            BinOp::AndAlso,
            Box::new(negate_cond(a)),
            Box::new(negate_cond(b)),
        ),
        _ => Expr::Not(Box::new(e.clone())),
    }
}

fn invert_cmp(k: CmpKind) -> CmpKind {
    match k {
        CmpKind::Eq => CmpKind::Ne,
        CmpKind::Ne => CmpKind::Eq,
        CmpKind::Lt => CmpKind::Ge,
        CmpKind::Ge => CmpKind::Lt,
        CmpKind::Le => CmpKind::Gt,
        CmpKind::Gt => CmpKind::Le,
    }
}

/// Kind after swapping the two comparison operands (`a < b` ⇔ `b > a`).
fn swap_cmp(k: CmpKind) -> CmpKind {
    match k {
        CmpKind::Eq => CmpKind::Eq,
        CmpKind::Ne => CmpKind::Ne,
        CmpKind::Lt => CmpKind::Gt,
        CmpKind::Gt => CmpKind::Lt,
        CmpKind::Le => CmpKind::Ge,
        CmpKind::Ge => CmpKind::Le,
    }
}

fn is_literal(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Int(_)
            | Expr::Float(_)
            | Expr::Double(_)
            | Expr::Bool(_)
            | Expr::Str(_)
            | Expr::Null
    )
}

/// Pure test: no call, raw call, allocation, or await anywhere in it.
fn expr_is_pure_test(e: &Expr) -> bool {
    match e {
        Expr::Call { .. }
        | Expr::RawCall(..)
        | Expr::NewObj(..)
        | Expr::NewArr(..)
        | Expr::Await(..)
        | Expr::CollectionInit { .. }
        | Expr::ObjInit { .. }
        | Expr::MethodChain { .. } => false,
        _ => il2cpp_cleanup::expr_children(e)
            .into_iter()
            .all(expr_is_pure_test),
    }
}

/// Does the condition read memory a call could rewrite?
///
/// Fields and static fields (including reconstructed property getters, which
/// are spelled `Expr::Field` after `property_access`) and indexed/dereferenced
/// memory all qualify. A plain local (`Expr::Var`) does not: an intervening
/// call cannot rebind it.
fn cond_reads_memory(e: &Expr) -> bool {
    let mut found = false;
    visit_expr(e, &mut |x| {
        if matches!(
            x,
            Expr::Field(..)
                | Expr::StaticField(..)
                | Expr::Index { .. }
                | Expr::RawDeref(..)
        ) {
            found = true;
        }
    });
    found
}

// ===========================================================================
// C. Stable values
// ===========================================================================

/// Can this expression be re-evaluated later with the same value?
///
/// The strict version of "pure": no calls and no memory reads. Locals are not
/// rebound by calls (address-taken locals are refused separately), so a
/// `Var`/constant (or arithmetic over them) is stable across intervening
/// statements. Field/index reads are deliberately excluded: an intervening
/// call may store to them.
fn is_stable_expr(e: &Expr) -> bool {
    match e {
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            is_stable_expr(a) && is_stable_expr(b)
        }
        Expr::Not(x)
        | Expr::Neg(x)
        | Expr::Cast(_, x)
        | Expr::IndexFromEnd(x) => is_stable_expr(x),
        Expr::Cond(c, a, b) => {
            is_stable_expr(c) && is_stable_expr(a) && is_stable_expr(b)
        }
        Expr::Tuple { elements, .. } => elements.iter().all(is_stable_expr),
        _ => false,
    }
}

fn is_simple_local(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn invalidate_keys(known: &mut HashMap<String, Expr>, names: &HashSet<String>) {
    if names.is_empty() {
        return;
    }
    known.retain(|k, _| !names.contains(k));
}

/// Remove entries whose recorded value reads a name that was just written.
fn invalidate_dependents(known: &mut HashMap<String, Expr>, written: &HashSet<String>) {
    if written.is_empty() {
        return;
    }
    known.retain(|_, v| !written.iter().any(|n| expr_mentions(v, n)));
}

// ===========================================================================
// Expression / statement queries
// ===========================================================================

fn visit_expr<'a>(e: &'a Expr, f: &mut impl FnMut(&'a Expr)) {
    f(e);
    for child in il2cpp_cleanup::expr_children(e) {
        visit_expr(child, f);
    }
}

/// Every variable/field name mentioned by the expression.
///
/// Field names are included: a write to `this.badge` (spelled `badge` or
/// `this.badge`, depending on the producer) is a write to a name the condition
/// `this.badge != 0` reads.
fn names_in_expr(e: &Expr) -> HashSet<String> {
    let mut out = HashSet::new();
    visit_expr(e, &mut |x| match x {
        Expr::Var(n) => {
            out.insert(n.clone());
        }
        Expr::Field(_, f) | Expr::StaticField(_, f) => {
            out.insert(f.clone());
        }
        _ => {}
    });
    out
}

/// All expressions owned by a statement (not its nested statements).
fn stmt_exprs<'a>(s: &'a Stmt, out: &mut Vec<&'a Expr>) {
    match s {
        Stmt::Declare(..) => {}
        Stmt::Eval(e) | Stmt::YieldReturn(e) | Stmt::Throw(e) | Stmt::Return(Some(e)) => {
            out.push(e)
        }
        Stmt::Assign(_, _, e) => out.push(e),
        Stmt::EventAssign { field, value, .. } => {
            out.push(field);
            out.push(value);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            out.push(collection);
            out.push(index);
            out.push(value);
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            out.push(cond)
        }
        Stmt::For { cond: Some(cond), .. } => out.push(cond),
        Stmt::Lock { expr, .. } => out.push(expr),
        Stmt::StringSwitch { disc, .. } => out.push(disc),
        Stmt::Attribute { args, .. } => out.extend(args.iter()),
        Stmt::Using { decl, .. } => stmt_exprs(decl, out),
        Stmt::Return(None)
        | Stmt::For { cond: None, .. }
        | Stmt::ForEach { .. }
        | Stmt::Switch { .. }
        | Stmt::Try { .. }
        | Stmt::Break
        | Stmt::Continue
        | Stmt::Goto(_)
        | Stmt::GotoCase(_)
        | Stmt::Label(_)
        | Stmt::Comment(_)
        | Stmt::YieldBreak => {}
    }
}

fn stmt_has_label(s: &Stmt) -> bool {
    if matches!(s, Stmt::Label(_)) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    kids.iter().any(|k| k.iter().any(stmt_has_label))
}

/// Does evaluating the statement (directly or through a nested statement)
/// call anything?
fn stmt_has_call(s: &Stmt) -> bool {
    let mut exprs = Vec::new();
    stmt_exprs(s, &mut exprs);
    if exprs.into_iter().any(expr_has_call) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    kids.iter().any(|k| k.iter().any(stmt_has_call))
}

fn is_flow_barrier(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Label(_)
            | Stmt::Goto(_)
            | Stmt::GotoCase(_)
            | Stmt::Return(_)
            | Stmt::Throw(_)
            | Stmt::Break
            | Stmt::Continue
            | Stmt::YieldBreak
    )
}

/// Does `stmt` write any name mentioned by the condition?
fn stmt_writes_any(s: &Stmt, names: &HashSet<String>) -> bool {
    if names.is_empty() {
        return false;
    }
    let direct = match s {
        Stmt::Assign(target, _, _) => target_touches(target, names),
        Stmt::EventAssign { field, value, .. } => {
            expr_touches(field, names) || expr_touches(value, names)
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            expr_touches(collection, names)
                || expr_touches(index, names)
                || expr_touches(value, names)
        }
        Stmt::ForEach {
            var_name,
            collection,
            ..
        } => names.contains(var_name) || names.contains(collection),
        Stmt::Using { decl, .. } => stmt_writes_any(decl, names),
        _ => false,
    };
    if direct {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    kids.iter()
        .any(|k| k.iter().any(|child| stmt_writes_any(child, names)))
}

fn expr_touches(e: &Expr, names: &HashSet<String>) -> bool {
    let mut found = false;
    visit_expr(e, &mut |x| match x {
        Expr::Var(n) if names.contains(n) => found = true,
        Expr::Field(_, f) | Expr::StaticField(_, f) if names.contains(f) => found = true,
        _ => {}
    });
    found
}

/// `target` is the textual LHS of an `Assign`; it can be a local (`v1`) or a
/// path produced by a reconstruction (`this.badge`, `v.f`, `v[i]`). Any
/// identifier token in it that the condition mentions means the fold is
/// disqualified.
fn target_touches(target: &str, names: &HashSet<String>) -> bool {
    if names.contains(target) {
        return true;
    }
    target
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|tok| !tok.is_empty() && names.contains(tok))
}

/// Every name a statement reads or writes, including nested statements.
fn names_touched_deep(s: &Stmt) -> HashSet<String> {
    let mut out = names_written_deep(s);
    let mut exprs = Vec::new();
    stmt_exprs(s, &mut exprs);
    for e in exprs {
        out.extend(names_in_expr(e));
    }
    // A switch discriminant and a foreach collection are string names, not
    // expressions.
    match s {
        Stmt::Switch { disc, .. } => {
            out.insert(disc.clone());
        }
        Stmt::ForEach { collection, .. } => {
            out.insert(collection.clone());
        }
        _ => {}
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    for k in kids {
        for child in k {
            out.extend(names_touched_deep(child));
        }
    }
    if let Stmt::Using { decl, .. } = s {
        out.extend(names_touched_deep(decl));
    }
    out
}

/// Every name a statement definitely writes (or may write), including nested.
fn names_written_deep(s: &Stmt) -> HashSet<String> {
    let mut out = HashSet::new();
    match s {
        Stmt::Assign(target, _, _) => {
            for tok in target.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                if !tok.is_empty() {
                    out.insert(tok.to_string());
                }
            }
        }
        Stmt::EventAssign { field, .. } => out.extend(names_in_expr(field)),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            out.extend(names_in_expr(collection));
            out.extend(names_in_expr(index));
            out.extend(names_in_expr(value));
        }
        Stmt::ForEach {
            var_name,
            collection,
            ..
        } => {
            out.insert(var_name.clone());
            out.insert(collection.clone());
        }
        Stmt::Using { decl, .. } => out.extend(names_written_deep(decl)),
        _ => {}
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    for k in kids {
        for child in k {
            out.extend(names_written_deep(child));
        }
    }
    out
}

// ===========================================================================
// Address-taken locals
// ===========================================================================

fn collect_addr_taken(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        let mut exprs = Vec::new();
        stmt_exprs(s, &mut exprs);
        for e in exprs {
            visit_expr(e, &mut |x| {
                if let Expr::AddressOf(inner) | Expr::RawAddr(inner, _) = x {
                    if let Expr::Var(n) = inner.as_ref() {
                        out.insert(n.clone());
                    }
                }
            });
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for k in kids {
            collect_addr_taken(k, out);
        }
        if let Stmt::Using { decl, .. } = s {
            collect_addr_taken(std::slice::from_ref(decl.as_ref()), out);
        }
    }
}

fn cond_reads_addr_taken(cond: &Expr, addr_taken: &HashSet<String>) -> bool {
    names_in_expr(cond)
        .iter()
        .any(|n| addr_taken.contains(n))
}

fn expr_reads_addr_taken(e: &Expr, addr_taken: &HashSet<String>) -> bool {
    names_in_expr(e).iter().any(|n| addr_taken.contains(n))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn v(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    fn x() -> Expr {
        v("x")
    }

    fn ne_zero(e: Expr) -> Expr {
        Expr::Cmp(CmpKind::Ne, Box::new(e), Box::new(Expr::Int(0)))
    }

    fn assign(name: &str, rhs: Expr) -> Stmt {
        Stmt::Assign(name.to_string(), None, rhs)
    }

    fn if_stmt(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond,
            then_body,
            else_body,
        }
    }

    // --- A: nested same-polarity guards ---------------------------------

    #[test]
    fn folds_direct_duplicate_guard() {
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![
                assign("y", Expr::Int(1)),
                if_stmt(ne_zero(x()), vec![assign("z", Expr::Int(2))], vec![]),
            ],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 2);
                assert!(matches!(&then_body[1], Stmt::Assign(name, None, _) if name == "z"));
            }
            other => panic!("expected outer if, got {other:?}"),
        }
    }

    #[test]
    fn folds_not_and_zero_shapes() {
        // outer: if (!(x == 0)) { inner: if (x != 0) { y = 1; } }
        let outer = Expr::Not(Box::new(Expr::Cmp(
            CmpKind::Eq,
            Box::new(x()),
            Box::new(Expr::Int(0)),
        )));
        let mut stmts = vec![if_stmt(
            outer,
            vec![if_stmt(ne_zero(x()), vec![assign("y", Expr::Int(1))], vec![])],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(
            &stmts[0],
            Stmt::If { then_body, .. }
                if matches!(then_body.as_slice(), [Stmt::Assign(name, None, _)] if name == "y")
        ));
    }

    #[test]
    fn folds_inner_if_whose_else_arm_is_dead() {
        // Under the outer test the condition is true, so `z = 2` cannot run;
        // the whole inner shell (test AND dead else) goes away.
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![if_stmt(
                ne_zero(x()),
                vec![assign("y", Expr::Int(1))],
                vec![assign("z", Expr::Int(2))],
            )],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert!(matches!(&then_body[0], Stmt::Assign(name, _, _) if name == "y"));
            }
            other => panic!("expected outer if, got {other:?}"),
        }
    }

    #[test]
    fn refuses_to_fold_inner_if_whose_dead_else_holds_a_label() {
        // The else arm is unreachable, but a label inside it is an entry
        // point: a `goto` elsewhere may still land on it.
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![if_stmt(
                ne_zero(x()),
                vec![assign("y", Expr::Int(1))],
                vec![Stmt::Label("L9".into()), assign("z", Expr::Int(2))],
            )],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(
            &stmts[0],
            Stmt::If { then_body, .. } if matches!(then_body.as_slice(), [Stmt::If { .. }])
        ));
    }

    #[test]
    fn refuses_guard_when_condition_is_impure() {
        let cond = || {
            Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Call {
                    recv: None,
                    name: "GetBadge".into(),
                    args: vec![],
                }),
                Box::new(Expr::Int(0)),
            )
        };
        let mut stmts = vec![if_stmt(
            cond(),
            vec![if_stmt(cond(), vec![assign("y", Expr::Int(1))], vec![])],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(
            &stmts[0],
            Stmt::If { then_body, .. } if matches!(then_body.as_slice(), [Stmt::If { .. }])
        ));
    }

    #[test]
    fn refuses_guard_when_between_writes_condition() {
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![
                assign("x", Expr::Int(1)),
                if_stmt(ne_zero(x()), vec![assign("y", Expr::Int(1))], vec![]),
            ],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_guard_when_between_writes_a_field_read_by_condition() {
        let cond = || {
            Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Field(Box::new(Expr::This), "badge".into())),
                Box::new(Expr::Int(0)),
            )
        };
        let mut stmts = vec![if_stmt(
            cond(),
            vec![
                assign("this.badge", Expr::Int(1)),
                if_stmt(cond(), vec![assign("y", Expr::Int(1))], vec![]),
            ],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_guard_when_inner_body_contains_label() {
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![if_stmt(
                ne_zero(x()),
                vec![Stmt::Label("L".into()), assign("y", Expr::Int(1))],
                vec![],
            )],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_guard_when_condition_local_is_address_taken() {
        let mut stmts = vec![
            Stmt::Eval(Expr::AddressOf(Box::new(x()))),
            if_stmt(
                ne_zero(x()),
                vec![if_stmt(ne_zero(x()), vec![assign("y", Expr::Int(1))], vec![])],
                vec![],
            ),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn folds_through_intervening_call() {
        // Calls between may not rebind a local: the fold is still valid.
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![
                assign("y", Expr::Int(1)),
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Tick".into(),
                    args: vec![],
                }),
                if_stmt(ne_zero(x()), vec![assign("z", Expr::Int(2))], vec![]),
            ],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }

    fn field(name: &str) -> Expr {
        Expr::Field(Box::new(Expr::This), name.to_string())
    }

    fn ne_zero_field(name: &str) -> Expr {
        ne_zero(field(name))
    }

    fn call_stmt(name: &str) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: None,
            name: name.into(),
            args: vec![],
        })
    }

    #[test]
    fn folds_descendant_guard_behind_other_pure_guard() {
        // if (title != 0) { if (product != 0) { if (title != 0) { render } } }
        let cond = ne_zero_field("title");
        let mut stmts = vec![if_stmt(
            cond.clone(),
            vec![if_stmt(
                ne_zero_field("product"),
                vec![if_stmt(cond, vec![call_stmt("Render")], vec![])],
                vec![],
            )],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => match &then_body[0] {
                Stmt::If {
                    then_body: inner, ..
                } => {
                    assert_eq!(inner.len(), 1);
                    assert!(matches!(&inner[0], Stmt::Eval(Expr::Call { name, .. }) if name == "Render"));
                }
                other => panic!("expected product guard, got {other:?}"),
            },
            other => panic!("expected outer if, got {other:?}"),
        }
    }

    #[test]
    fn refuses_descendant_guard_when_call_between_field_tests() {
        // A call can rebind `title` through another reference: not provable.
        let cond = ne_zero_field("title");
        let mut stmts = vec![if_stmt(
            cond.clone(),
            vec![
                call_stmt("Mutate"),
                if_stmt(
                    ne_zero_field("product"),
                    vec![if_stmt(cond, vec![assign("y", Expr::Int(1))], vec![])],
                    vec![],
                ),
            ],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_descendant_guard_when_field_written_behind_other_guard() {
        let cond = ne_zero_field("title");
        let mut stmts = vec![if_stmt(
            cond.clone(),
            vec![if_stmt(
                ne_zero_field("product"),
                vec![
                    assign("this.title", Expr::Int(1)),
                    if_stmt(cond, vec![assign("y", Expr::Int(1))], vec![]),
                ],
                vec![],
            )],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn folds_descendant_local_guard_across_call() {
        // A call cannot rebind a local, so the inner local test is redundant.
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![
                call_stmt("Tick"),
                if_stmt(
                    ne_zero(v("product")),
                    vec![if_stmt(ne_zero(x()), vec![assign("y", Expr::Int(1))], vec![])],
                    vec![],
                ),
            ],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }

    #[test]
    fn drops_a_contradictory_else_along_with_its_dead_arm() {
        // The else arm requires x == 0 while the outer test proved x != 0, so
        // nothing in it can run: the inner test goes, and the arm with it.
        // `b` is dead and must never be spliced onto the live path.
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![if_stmt(
                ne_zero(x()),
                vec![assign("a", Expr::Int(1))],
                vec![if_stmt(ne_zero(x()), vec![assign("b", Expr::Int(2))], vec![])],
            )],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert!(matches!(&then_body[0], Stmt::Assign(name, None, _) if name == "a"));
            }
            other => panic!("expected outer if, got {other:?}"),
        }
    }

    // --- B: impossible else-if ------------------------------------------

    #[test]
    fn removes_impossible_else_if() {
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![assign("a", Expr::Int(1))],
            vec![
                assign("b", Expr::Int(2)),
                if_stmt(ne_zero(x()), vec![assign("c", Expr::Int(3))], vec![]),
            ],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { else_body, .. } => {
                assert_eq!(else_body.len(), 1);
                assert!(matches!(&else_body[0], Stmt::Assign(name, None, _) if name == "b"));
            }
            other => panic!("expected if, got {other:?}"),
        }
    }

    #[test]
    fn keeps_impossible_if_with_else() {
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![],
            vec![if_stmt(
                ne_zero(x()),
                vec![assign("c", Expr::Int(3))],
                vec![assign("d", Expr::Int(4))],
            )],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_impossible_if_after_write() {
        let mut stmts = vec![if_stmt(
            ne_zero(x()),
            vec![],
            vec![
                assign("x", Expr::Int(1)),
                if_stmt(ne_zero(x()), vec![assign("c", Expr::Int(3))], vec![]),
            ],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    // --- C: redundant re-assignment -------------------------------------

    #[test]
    fn removes_duplicate_constant_at_arm_head() {
        let mut stmts = vec![
            assign("x", Expr::Int(0)),
            if_stmt(
                ne_zero(v("t")),
                vec![assign("x", Expr::Int(0)), assign("y", Expr::Int(1))],
                vec![],
            ),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[1] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert!(matches!(&then_body[0], Stmt::Assign(name, None, _) if name == "y"));
            }
            other => panic!("expected if, got {other:?}"),
        }
    }

    #[test]
    fn removes_temp_copy_at_every_nested_arm_head() {
        // v = w; if (a) { v = w; if (b) { v = w; } }
        let mut stmts = vec![
            assign("v", v("w")),
            if_stmt(
                ne_zero(v("a")),
                vec![
                    assign("v", v("w")),
                    if_stmt(ne_zero(v("b")), vec![assign("v", v("w"))], vec![]),
                ],
                vec![],
            ),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2);
        match &stmts[1] {
            Stmt::If { then_body, .. } => match &then_body[0] {
                Stmt::If { then_body, .. } => assert!(then_body.is_empty()),
                other => panic!("expected nested if, got {other:?}"),
            },
            other => panic!("expected if, got {other:?}"),
        }
    }

    #[test]
    fn removes_duplicate_in_else_arm() {
        let mut stmts = vec![
            assign("x", Expr::Int(0)),
            if_stmt(
                ne_zero(v("t")),
                vec![assign("y", Expr::Int(1))],
                vec![assign("x", Expr::Int(0))],
            ),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[1] {
            Stmt::If { else_body, .. } => assert!(else_body.is_empty()),
            other => panic!("expected if, got {other:?}"),
        }
    }

    #[test]
    fn removes_straight_line_duplicate() {
        let mut stmts = vec![
            assign("x", Expr::Int(0)),
            assign("y", Expr::Int(1)),
            assign("x", Expr::Int(0)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn refuses_duplicate_when_read_between() {
        let mut stmts = vec![
            assign("x", Expr::Int(0)),
            assign("y", x()),
            if_stmt(ne_zero(v("t")), vec![assign("x", Expr::Int(0))], vec![]),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_duplicate_when_written_between() {
        let mut stmts = vec![
            assign("x", Expr::Int(0)),
            assign("x", Expr::Int(1)),
            if_stmt(ne_zero(v("t")), vec![assign("x", Expr::Int(0))], vec![]),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_duplicate_when_value_operand_changes() {
        let mut stmts = vec![
            assign("x", v("w")),
            assign("w", Expr::Int(5)),
            if_stmt(ne_zero(v("t")), vec![assign("x", v("w"))], vec![]),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_duplicate_when_condition_reads_target() {
        let mut stmts = vec![
            assign("x", Expr::Int(0)),
            if_stmt(ne_zero(x()), vec![assign("x", Expr::Int(0))], vec![]),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_duplicate_after_label() {
        let mut stmts = vec![
            assign("x", Expr::Int(0)),
            Stmt::Label("L".into()),
            if_stmt(ne_zero(v("t")), vec![assign("x", Expr::Int(0))], vec![]),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_duplicate_when_target_address_taken() {
        let mut stmts = vec![
            Stmt::Eval(Expr::AddressOf(Box::new(x()))),
            assign("x", Expr::Int(0)),
            if_stmt(ne_zero(v("t")), vec![assign("x", Expr::Int(0))], vec![]),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_duplicate_when_rhs_is_impure() {
        let call = || Expr::Call {
            recv: None,
            name: "F".into(),
            args: vec![],
        };
        let mut stmts = vec![
            assign("x", call()),
            if_stmt(ne_zero(v("t")), vec![assign("x", call())], vec![]),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn refuses_duplicate_when_rhs_reads_target() {
        // x = x + 1; if (t) { x = x + 1; } is NOT redundant: the second
        // statement increments the already-incremented value.
        let incr = || {
            Expr::Bin(
                BinOp::Add,
                Box::new(x()),
                Box::new(Expr::Int(1)),
            )
        };
        let mut stmts = vec![
            assign("x", incr()),
            if_stmt(ne_zero(v("t")), vec![assign("x", incr())], vec![]),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    #[test]
    fn nested_rewrites_inside_loop_body() {
        // The walker must reach lists owned by non-if constructs too.
        let mut stmts = vec![Stmt::While {
            cond: ne_zero(v("t")),
            body: vec![
                assign("x", Expr::Int(0)),
                if_stmt(ne_zero(v("t")), vec![assign("x", Expr::Int(0))], vec![]),
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }

    #[test]
    fn folds_a_duplicate_guard_over_an_index_with_zero() {
        // The exact shape the corpus leaves: `!= 0` (not `!= null`) over an
        // index, with two pure stores between the tests.
        let cond = || {
            Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Index {
                    collection: Box::new(v("v307")),
                    index: Box::new(v("v141")),
                }),
                Box::new(Expr::Int(0)),
            )
        };
        let mut stmts = vec![if_stmt(
            cond(),
            vec![
                assign("v620", Expr::Int(0)),
                assign("v109", Expr::Int(0)),
                if_stmt(cond(), vec![assign("z", Expr::Int(2))], vec![]),
            ],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:#?}");
    }

    #[test]
    fn folds_a_duplicate_guard_over_an_index() {
        // `if (list[i] != null) { a = 0; b = 0; if (list[i] != null) { ... } }`
        // — the shape the structurer leaves after IL2CPP's per-dereference
        // null check. Nothing between the two tests writes `list` or `i`.
        let cond = || {
            Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Index {
                    collection: Box::new(v("list")),
                    index: Box::new(v("i")),
                }),
                Box::new(Expr::Null),
            )
        };
        let mut stmts = vec![if_stmt(
            cond(),
            vec![
                assign("a", Expr::Int(0)),
                assign("b", Expr::Int(0)),
                if_stmt(cond(), vec![assign("z", Expr::Int(2))], vec![]),
            ],
            vec![],
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "{stmts:#?}");
    }
}
