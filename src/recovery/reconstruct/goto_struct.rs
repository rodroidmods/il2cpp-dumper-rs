//! Goto structuring — turn remaining jumps into structured C# control flow.
//!
//! The structurer emits a flat statement list with `goto`/label edges. The
//! earlier passes (`goto_simplify`, `goto_fold`, `idioms`) remove the jumps
//! whose target is the natural fall-through or the loop exit of the jump's own
//! block. What survives is the irreducible-looking part: back edges, forward
//! skips over a block, and exits that leave several nested loops at once.
//! This pass raises those edges into `while (true)` / `if` / `break` /
//! `continue` / `return`, and leaves a jump alone whenever the rewrite cannot
//! be proved equivalent.
//!
//! ## Rules
//!
//! **A. Back edge → loop.** For `L: …; goto L` in one statement list, the
//! region `[L … goto L]` is wrapped in `while (true) { … }`. Inside the loop a
//! jump to `L` becomes `continue`; a jump to a label in the run immediately
//! after the region becomes `break`; a region that can fall out of its end gets
//! a trailing `break`. Both rewrites fire only in "transparent" positions: a
//! jump inside a nested loop, switch, or exception region keeps its `goto`
//! (a `continue`/`break` there would bind to the wrong construct).
//!
//! **B. Forward join inversion.** `if (c) goto L; <rest>; L:` where `L`
//! directly follows `<rest>` becomes `if (!c) { <rest> }`, provided `c` is
//! side-effect free and nothing outside `<rest>` jumps to a label defined
//! inside it. When the label sits in a *nested* arm instead
//! (`if (A) { … L: S } else … goto L;`), the straight-line instruction run
//! `S` is copied into the jumping arm. Each path still executes one copy, so
//! local assignments are allowed; a call (a duplicated side-effecting
//! expression) or nested control flow refuses the rewrite.
//!
//! **C. Exit normalization.** A jump in the immediate tail label run of the
//! same list is dropped; any jump inside one loop to the label run directly
//! after that loop becomes `break`; a jump to the method-end label becomes
//! `return;` when the method is void-like.
//!
//! **D. Multi-level exits.** A jump from inside nested loops to a label after
//! the outermost loop uses a `bool done` flag: `done = true; break;` at the
//! jump site, `if (done) break;` after every enclosed loop. C# has no labelled
//! `break`, and duplicating the exit edge at each level is the exact
//! replacement for the multi-level jump.
//!
//! **E. Jump to the loop header → `continue`.** A jump inside a `while` body
//! to a label immediately before that `while` re-evaluates the condition,
//! which is exactly `continue`. Restricted to `while`: re-entering a `do`
//! executes the body, a `for` re-runs its initializer and a `foreach`
//! re-evaluates its collection, none of which `continue` does.
//!
//! ## Safety rules
//!
//! - Never cross a `try`/`catch`/`finally` boundary.
//! - Never rewrite a jump inside a switch case to `break`/`continue` (it would
//!   bind to the switch, not the loop).
//! - Never delete a label a remaining jump still targets.
//! - Refuse a region that can be entered from outside, or that contains a
//!   label referenced from outside (nesting the region would move the label out
//!   of the referencing scope).
//! - Refuse a region containing a free `break`/`continue`/`goto case`: moving
//!   it under the new loop would re-bind it.
//! - Refuse when a typed local declared inside the region is mentioned after
//!   it: nesting the declaration would take it out of scope.
//! - A failed match is a no-op. An incorrect rewrite is worse than a `goto`.
//!
//! ## Relationship to the other passes
//!
//! `goto_simplify` handles goto→return and label inlining; `goto_fold` handles
//! fall-through/chain/loop-tail shapes; `idioms` sweeps dangling jumps. This
//! pass adds the region-level structuring those per-statement rules cannot
//! express. Every rewrite is idempotent (the rewritten shape no longer
//! matches), so the pass runs to a small bounded fixpoint.

use crate::recovery::csharp::{Expr, InterpPart, Stmt};
use crate::recovery::reconstruct::{children, children_mut, expr_has_call};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

/// Bounded fixpoint. Each round removes at least one jump or label, so the
/// bound only guards a future rule that rewrites without progress.
const MAX_ROUNDS: usize = 32;

/// Per-rule counters and refusal counters. `ILR_GOTOCENSUS=1` prints the
/// counters once per method that changed or refused something.
pub struct GotoStructCensus {
    pub a_backward_loops: AtomicU64,
    pub b_forward_joins: AtomicU64,
    pub b_suffix_copies: AtomicU64,
    pub c_breaks: AtomicU64,
    pub c_returns: AtomicU64,
    pub c_drops: AtomicU64,
    pub d_multilevel_exits: AtomicU64,
    pub e_header_continues: AtomicU64,
    pub refuse_external_entry: AtomicU64,
    pub refuse_duplicate_label: AtomicU64,
    pub refuse_free_break: AtomicU64,
    pub refuse_scope_escape: AtomicU64,
    pub refuse_side_effect_cond: AtomicU64,
    pub refuse_impure_suffix: AtomicU64,
    pub refuse_try_boundary: AtomicU64,
    pub refuse_switch_boundary: AtomicU64,
    pub refuse_unproven_target: AtomicU64,
}

pub static CENSUS: GotoStructCensus = GotoStructCensus {
    a_backward_loops: AtomicU64::new(0),
    b_forward_joins: AtomicU64::new(0),
    b_suffix_copies: AtomicU64::new(0),
    c_breaks: AtomicU64::new(0),
    c_returns: AtomicU64::new(0),
    c_drops: AtomicU64::new(0),
    d_multilevel_exits: AtomicU64::new(0),
    e_header_continues: AtomicU64::new(0),
    refuse_external_entry: AtomicU64::new(0),
    refuse_duplicate_label: AtomicU64::new(0),
    refuse_free_break: AtomicU64::new(0),
    refuse_scope_escape: AtomicU64::new(0),
    refuse_side_effect_cond: AtomicU64::new(0),
    refuse_impure_suffix: AtomicU64::new(0),
    refuse_try_boundary: AtomicU64::new(0),
    refuse_switch_boundary: AtomicU64::new(0),
    refuse_unproven_target: AtomicU64::new(0),
};

fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

fn census_on() -> bool {
    std::env::var_os("ILR_GOTOCENSUS").is_some()
}

/// Human-readable counter dump (also printed per changed method when
/// `ILR_GOTOCENSUS=1`, since this module has no end-of-run hook of its own).
pub fn report() -> String {
    let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let applied = g(&CENSUS.a_backward_loops)
        + g(&CENSUS.b_forward_joins)
        + g(&CENSUS.b_suffix_copies)
        + g(&CENSUS.c_breaks)
        + g(&CENSUS.c_returns)
        + g(&CENSUS.c_drops)
        + g(&CENSUS.d_multilevel_exits)
        + g(&CENSUS.e_header_continues);
    let refused = g(&CENSUS.refuse_external_entry)
        + g(&CENSUS.refuse_duplicate_label)
        + g(&CENSUS.refuse_free_break)
        + g(&CENSUS.refuse_scope_escape)
        + g(&CENSUS.refuse_side_effect_cond)
        + g(&CENSUS.refuse_impure_suffix)
        + g(&CENSUS.refuse_try_boundary)
        + g(&CENSUS.refuse_switch_boundary)
        + g(&CENSUS.refuse_unproven_target);
    if applied == 0 && refused == 0 {
        return String::new();
    }
    format!(
        "[goto_struct] applied {applied} (loops {}, joins {}, suffix-copies {}, breaks {}, returns {}, drops {}, multi-exit {}, header-continues {}) \
         refused {refused} (external-entry {}, duplicate-label {}, free-break {}, scope {}, side-effect-cond {}, \
         impure-suffix {}, try {}, switch {}, unproven {})",
        g(&CENSUS.a_backward_loops),
        g(&CENSUS.b_forward_joins),
        g(&CENSUS.b_suffix_copies),
        g(&CENSUS.c_breaks),
        g(&CENSUS.c_returns),
        g(&CENSUS.c_drops),
        g(&CENSUS.d_multilevel_exits),
        g(&CENSUS.e_header_continues),
        g(&CENSUS.refuse_external_entry),
        g(&CENSUS.refuse_duplicate_label),
        g(&CENSUS.refuse_free_break),
        g(&CENSUS.refuse_scope_escape),
        g(&CENSUS.refuse_side_effect_cond),
        g(&CENSUS.refuse_impure_suffix),
        g(&CENSUS.refuse_try_boundary),
        g(&CENSUS.refuse_switch_boundary),
        g(&CENSUS.refuse_unproven_target),
    )
}

/// Entry point: structure the remaining jumps in one method body. Returns the
/// number of rewrites applied.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut used_names = HashSet::new();
    collect_names_stmts(stmts, &mut used_names);
    let before = report();
    let dbg = std::env::var_os("ILR_GSDBG").is_some();
    if dbg {
        eprintln!("[gsdbg] === before ===\n{stmts:#?}");
    }
    // Rule A is the one that regressed (`while (true) { return; }`). B–E are
    // forward joins, exit normalisation, multi-level exits and header
    // continues; test them independently before enabling A anywhere.
    let skip_a = std::env::var_os("ILR_GOTOSTRUCT_NOA").is_some()
        || std::env::var_os("ILR_GOTOSTRUCT_BCDE").is_some();
    let mut total = 0;
    for round_i in 0..MAX_ROUNDS {
        let ra = if skip_a { 0 } else { rule_a_backward_loops(stmts) };
        if dbg && ra > 0 {
            eprintln!("[gsdbg] === after round {round_i} rule A ({ra}) ===\n{stmts:#?}");
        }
        let rb = rule_b_forward_join(stmts);
        if dbg && rb > 0 {
            eprintln!("[gsdbg] === after round {round_i} rule B ({rb}) ===\n{stmts:#?}");
        }
        let rc = rule_c_exit_normalization(stmts);
        if dbg && rc > 0 {
            eprintln!("[gsdbg] === after round {round_i} rule C ({rc}) ===\n{stmts:#?}");
        }
        let rd = rule_d_multi_level_exits(stmts, &mut used_names);
        if dbg && rd > 0 {
            eprintln!("[gsdbg] === after round {round_i} rule D ({rd}) ===\n{stmts:#?}");
        }
        let re = rule_e_header_continues(stmts);
        if dbg && re > 0 {
            eprintln!("[gsdbg] === after round {round_i} rule E ({re}) ===\n{stmts:#?}");
        }
        let round = ra + rb + rc + rd + re;
        if round == 0 {
            break;
        }
        total += round;
    }
    if census_on() {
        let after = report();
        if !after.is_empty() && after != before {
            eprintln!("{after}");
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Shared tree helpers
// ---------------------------------------------------------------------------

fn walk<'a>(stmts: &'a [Stmt], f: &mut dyn FnMut(&'a Stmt)) {
    for s in stmts {
        f(s);
        let mut kids = Vec::new();
        children(s, &mut kids);
        for c in kids {
            walk(c, f);
        }
    }
}

fn count_refs(stmts: &[Stmt]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    walk(stmts, &mut |s| {
        if let Stmt::Goto(l) = s {
            *m.entry(l.clone()).or_insert(0) += 1;
        }
    });
    m
}

fn count_defs(stmts: &[Stmt]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    walk(stmts, &mut |s| {
        if let Stmt::Label(l) = s {
            *m.entry(l.clone()).or_insert(0) += 1;
        }
    });
    m
}

fn labels_in(stmts: &[Stmt]) -> HashSet<String> {
    let mut out = HashSet::new();
    walk(stmts, &mut |s| {
        if let Stmt::Label(l) = s {
            out.insert(l.clone());
        }
    });
    out
}

fn contains_goto(stmts: &[Stmt], label: &str) -> bool {
    let mut found = false;
    walk(stmts, &mut |s| {
        if matches!(s, Stmt::Goto(l) if l == label) {
            found = true;
        }
    });
    found
}

fn count_refs_in(stmts: &[Stmt], label: &str) -> usize {
    let mut n = 0;
    walk(stmts, &mut |s| {
        if matches!(s, Stmt::Goto(l) if l == label) {
            n += 1;
        }
    });
    n
}

/// Labels in the immediate label run after `i` (comments/attributes skipped).
fn label_run_after(list: &[Stmt], i: usize) -> Vec<String> {
    let mut out = Vec::new();
    for s in list.iter().skip(i + 1) {
        match s {
            Stmt::Comment(_) | Stmt::Attribute { .. } => continue,
            Stmt::Label(l) => out.push(l.clone()),
            _ => break,
        }
    }
    out
}

fn find_label_pos(list: &[Stmt], label: &str) -> Option<usize> {
    list.iter()
        .position(|s| matches!(s, Stmt::Label(l) if l == label))
}

/// Is `e` free of observable effects (calls, allocations, awaits)?
///
/// Rule B evaluates the condition exactly once in both spellings, so reads and
/// dereferences are allowed: they observe the same memory in the same order
/// (a faulting dereference faults identically). A call/allocation is a side
/// effect and disqualifies the inversion.
fn is_side_effect_free(e: &Expr) -> bool {
    !expr_has_call(e)
}

// ---------------------------------------------------------------------------
// Name collection (scope guard + `done` flag naming)
// ---------------------------------------------------------------------------

/// Sink for name collection: a set (does this name occur?) or a counter
/// (how often?). The scope guards need the counter form to decide whether a
/// name is used *outside* a region rather than merely inside it.
trait NameSink {
    fn add_name(&mut self, name: String);
}

impl NameSink for HashSet<String> {
    fn add_name(&mut self, name: String) {
        self.insert(name);
    }
}

impl NameSink for HashMap<String, usize> {
    fn add_name(&mut self, name: String) {
        *self.entry(name).or_insert(0) += 1;
    }
}

fn expr_names<S: NameSink>(e: &Expr, out: &mut S) {
    match e {
        Expr::Var(n) => {
            out.add_name(n.clone());
        }
        Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::This
        | Expr::MissingAbiArg
        | Expr::Unknown
        | Expr::TypeOf(_)
        | Expr::MethodGroup(_)
        | Expr::StaticField(_, _) => {}
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            expr_names(a, out);
            expr_names(b, out);
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) | Expr::Await(a) => expr_names(a, out),
        Expr::Field(base, _) => expr_names(base, out),
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                expr_names(r, out);
            }
            for a in args {
                expr_names(a, out);
            }
        }
        Expr::Cond(c, a, b) => {
            expr_names(c, out);
            expr_names(a, out);
            expr_names(b, out);
        }
        Expr::Lambda { body, .. } => expr_names(body, out),
        Expr::RawDeref(base, _, _) | Expr::RawAddr(base, _) | Expr::AddressOf(base) => {
            expr_names(base, out)
        }
        Expr::RawCall(_, args, _) => {
            for a in args {
                expr_names(a, out);
            }
        }
        Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                expr_names(a, out);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for a in items {
                expr_names(a, out);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter().chain(items.iter()) {
                expr_names(a, out);
            }
        }
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            for a in ctor_args.iter().chain(fields.iter().map(|(_, v)| v)) {
                expr_names(a, out);
            }
        }
        Expr::Index { collection, index } => {
            expr_names(collection, out);
            expr_names(index, out);
        }
        Expr::NullConditional { receiver, access } => {
            expr_names(receiver, out);
            expr_names(access, out);
        }
        Expr::PatternMatch { expr, .. } => expr_names(expr, out),
        Expr::StringInterp { parts } => {
            for p in parts {
                if let InterpPart::Expr(e, _) = p {
                    expr_names(e, out);
                }
            }
        }
        Expr::Tuple { elements, .. } => {
            for a in elements {
                expr_names(a, out);
            }
        }
        Expr::MethodChain { receiver, methods } => {
            expr_names(receiver, out);
            for m in methods {
                for a in &m.args {
                    expr_names(a, out);
                }
            }
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            expr_names(discrim, out);
            for (p, r) in arms {
                expr_names(p, out);
                expr_names(r, out);
            }
            if let Some(d) = default_arm {
                expr_names(d, out);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                expr_names(f, out);
            }
            if let Some(t) = to {
                expr_names(t, out);
            }
        }
        Expr::IndexFromEnd(a) => expr_names(a, out),
    }
}

fn collect_names_stmts<S: NameSink>(stmts: &[Stmt], out: &mut S) {
    for s in stmts {
        match s {
            Stmt::Eval(e) | Stmt::YieldReturn(e) | Stmt::Throw(e) | Stmt::Return(Some(e)) => {
                expr_names(e, out)
            }
            Stmt::Assign(name, _, e) => {
                out.add_name(name.clone());
                expr_names(e, out);
            }
            Stmt::EventAssign { field, value, .. } => {
                expr_names(field, out);
                expr_names(value, out);
            }
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                expr_names(collection, out);
                expr_names(index, out);
                expr_names(value, out);
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                expr_names(cond, out);
                collect_names_stmts(then_body, out);
                collect_names_stmts(else_body, out);
            }
            Stmt::While { cond, body } => {
                expr_names(cond, out);
                collect_names_stmts(body, out);
            }
            Stmt::DoWhile { body, cond } => {
                collect_names_stmts(body, out);
                expr_names(cond, out);
            }
            Stmt::ForEach {
                var_name,
                collection,
                body,
            } => {
                out.add_name(var_name.clone());
                out.add_name(collection.clone());
                collect_names_stmts(body, out);
            }
            Stmt::For {
                init,
                cond,
                incr,
                body,
            } => {
                collect_names_stmts(init, out);
                if let Some(c) = cond {
                    expr_names(c, out);
                }
                collect_names_stmts(incr, out);
                collect_names_stmts(body, out);
            }
            Stmt::Using { decl, body } => {
                collect_names_stmts(std::slice::from_ref(decl.as_ref()), out);
                collect_names_stmts(body, out);
            }
            Stmt::Lock { expr, body } => {
                expr_names(expr, out);
                collect_names_stmts(body, out);
            }
            Stmt::Switch {
                disc,
                cases,
                tail,
                default,
            } => {
                out.add_name(disc.clone());
                for (_, b) in cases {
                    collect_names_stmts(b, out);
                }
                collect_names_stmts(tail, out);
                if let Some(d) = default {
                    collect_names_stmts(d, out);
                }
            }
            Stmt::StringSwitch { disc, cases, tail } => {
                expr_names(disc, out);
                for (_, b) in cases {
                    collect_names_stmts(b, out);
                }
                if let Some(t) = tail {
                    collect_names_stmts(t, out);
                }
            }
            Stmt::Attribute { args, .. } => {
                for a in args {
                    expr_names(a, out);
                }
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                collect_names_stmts(try_body, out);
                for (_, b) in catches {
                    collect_names_stmts(b, out);
                }
                if let Some(f) = finally_body {
                    collect_names_stmts(f, out);
                }
            }
            _ => {}
        }
    }
}

fn unique_name(base: &str, used: &mut HashSet<String>) -> String {
    if !used.contains(base) {
        used.insert(base.to_string());
        return base.to_string();
    }
    let mut i = 1;
    loop {
        let cand = format!("{base}_{i}");
        if !used.contains(&cand) {
            used.insert(cand.clone());
            return cand;
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Rule A: back edge → `while (true)`
// ---------------------------------------------------------------------------

fn rule_a_backward_loops(root: &mut Vec<Stmt>) -> usize {
    let refs = count_refs(root);
    let defs = count_defs(root);
    // Occurrence counts over the whole method, for the scope guard: a
    // declaration inside a wrapped region may move to a *deeper* block, so a
    // use anywhere outside the region (not just the same list's suffix) would
    // fall out of scope. Comparing whole-method counts against the region's
    // own counts detects a use outside the region without needing the region's
    // position in the tree.
    let mut uses = HashMap::new();
    collect_names_stmts(root, &mut uses);
    rule_a_list(root, &refs, &defs, &uses)
}

fn rule_a_list(
    list: &mut Vec<Stmt>,
    refs: &HashMap<String, usize>,
    defs: &HashMap<String, usize>,
    uses: &HashMap<String, usize>,
) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        let name = match &list[i] {
            Stmt::Label(l) => l.clone(),
            _ => {
                i += 1;
                continue;
            }
        };
        // The last sibling whose subtree jumps back to this label. The whole
        // run from the label through that sibling is one loop body.
        let mut q = None;
        for j in (i + 1)..list.len() {
            if contains_goto(std::slice::from_ref(&list[j]), &name) {
                q = Some(j);
            }
        }
        let Some(q) = q else {
            i += 1;
            continue;
        };
        // A label name must identify exactly one definition, or the edge's
        // destination is ambiguous.
        if defs.get(&name).copied().unwrap_or(0) != 1 {
            bump(&CENSUS.refuse_duplicate_label);
            i += 1;
            continue;
        }

        // Every label defined inside the region must be uniquely defined and
        // have no incoming edge from outside the region: nesting the region
        // would move those labels out of the referencing scope.
        let inner = labels_in(&list[i..=q]);
        let mut duplicate = false;
        for l in &inner {
            if defs.get(l).copied().unwrap_or(0) != 1 {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            bump(&CENSUS.refuse_duplicate_label);
            i += 1;
            continue;
        }
        let inside_refs = count_refs(&list[i..=q]);
        let mut external = false;
        for l in &inner {
            if refs.get(l).copied().unwrap_or(0) != inside_refs.get(l).copied().unwrap_or(0) {
                external = true;
                break;
            }
        }
        if external {
            bump(&CENSUS.refuse_external_entry);
            i += 1;
            continue;
        }
        // A free `break`/`continue`/`goto case` in the region binds to a
        // construct outside it today; the new loop would capture it.
        if free_control_boundary(&list[i..=q]).is_some() {
            bump(&CENSUS.refuse_free_break);
            i += 1;
            continue;
        }
        if region_decls_escape(list, i, q, uses) {
            bump(&CENSUS.refuse_scope_escape);
            i += 1;
            continue;
        }

        // Only a label in the immediate run after the region can become
        // `break`: `break` lands right after the loop, before any statement
        // that originally sat between the region and the label. A duplicated
        // label name cannot be proven to name this landing pad.
        let after: HashSet<String> = label_run_after(list, q)
            .into_iter()
            .filter(|l| defs.get(l).copied().unwrap_or(0) == 1)
            .collect();
        let mut body: Vec<Stmt> = list[i + 1..=q].to_vec();
        let entry = name.clone();
        let (to_entry, _to_after) = rewrite_out_of_region(&mut body, &entry, &after, false);
        if to_entry == 0 {
            // Every jump back to the label sits inside a nested loop / switch
            // / exception region. Wrapping would add a loop level without
            // removing an edge (and would do so again on the next round), so
            // this shape is not structurable here.
            bump(&CENSUS.refuse_unproven_target);
            i += 1;
            continue;
        }
        // Jumps that could not be rewritten (inside nested loops / exception
        // regions) still need the label: keep it at the loop head.
        if count_refs_in(&body, &name) > 0 {
            body.insert(0, Stmt::Label(name.clone()));
        }
        drop_unreferenced_labels(&mut body);
        if block_falls_through(&body) {
            body.push(Stmt::Break);
        }
        let loop_stmt = Stmt::While {
            cond: Expr::Bool(true),
            body,
        };
        list.splice(i..=q, vec![loop_stmt]);
        bump(&CENSUS.a_backward_loops);
        n += 1;
        i += 1;
    }
    for s in list.iter_mut() {
        for c in children_mut(s) {
            n += rule_a_list(c, refs, defs, uses);
        }
    }
    n
}

/// Rewrite jumps in a freshly extracted loop body. Returns
/// `(jumps-to-entry, jumps-to-after)` converted.
///
/// `blocked` is set once the walk enters a nested loop, a switch, or an
/// exception region: `continue`/`break` there would bind to that construct
/// instead of the new loop, so the jump is left alone.
fn rewrite_out_of_region(
    list: &mut Vec<Stmt>,
    entry: &str,
    after: &HashSet<String>,
    blocked: bool,
) -> (usize, usize) {
    if blocked {
        return (0, 0);
    }
    let mut to_entry = 0;
    let mut to_after = 0;
    for s in list.iter_mut() {
        match s {
            Stmt::Goto(t) => {
                if t == entry {
                    *s = Stmt::Continue;
                    to_entry += 1;
                } else if after.contains(t.as_str()) {
                    *s = Stmt::Break;
                    to_after += 1;
                }
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                let (a, b) = rewrite_out_of_region(then_body, entry, after, false);
                let (c, d) = rewrite_out_of_region(else_body, entry, after, false);
                to_entry += a + c;
                to_after += b + d;
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                let (a, b) = rewrite_out_of_region(body, entry, after, false);
                to_entry += a;
                to_after += b;
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. } => {
                let (a, b) = rewrite_out_of_region(body, entry, after, true);
                to_entry += a;
                to_after += b;
            }
            Stmt::For {
                init, incr, body, ..
            } => {
                let (a, b) = rewrite_out_of_region(init, entry, after, true);
                let (c, d) = rewrite_out_of_region(incr, entry, after, true);
                let (e, f) = rewrite_out_of_region(body, entry, after, true);
                to_entry += a + c + e;
                to_after += b + d + f;
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases.iter_mut() {
                    let (a, c) = rewrite_out_of_region(b, entry, after, true);
                    to_entry += a;
                    to_after += c;
                }
                if let Some(arm) = default {
                    let (a, b) = rewrite_out_of_region(arm, entry, after, true);
                    to_entry += a;
                    to_after += b;
                }
                let (a, b) = rewrite_out_of_region(tail, entry, after, true);
                to_entry += a;
                to_after += b;
            }
            Stmt::StringSwitch { cases, tail, .. } => {
                for (_, b) in cases.iter_mut() {
                    let (a, c) = rewrite_out_of_region(b, entry, after, true);
                    to_entry += a;
                    to_after += c;
                }
                if let Some(t) = tail {
                    let (a, b) = rewrite_out_of_region(t, entry, after, true);
                    to_entry += a;
                    to_after += b;
                }
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                let (a, b) = rewrite_out_of_region(try_body, entry, after, true);
                to_entry += a;
                to_after += b;
                for (_, b) in catches.iter_mut() {
                    let (a, c) = rewrite_out_of_region(b, entry, after, true);
                    to_entry += a;
                    to_after += c;
                }
                if let Some(f) = finally_body {
                    let (a, b) = rewrite_out_of_region(f, entry, after, true);
                    to_entry += a;
                    to_after += b;
                }
            }
            _ => {}
        }
    }
    (to_entry, to_after)
}

/// The first `break`/`continue`/`goto case` in `stmts` that binds to a
/// construct OUTSIDE `stmts`, if any.
fn free_control_boundary(stmts: &[Stmt]) -> Option<&'static str> {
    fn rec(list: &[Stmt], loops: u32, switches: u32) -> Option<&'static str> {
        for s in list {
            match s {
                Stmt::Break if loops == 0 && switches == 0 => return Some("break"),
                Stmt::Continue if loops == 0 => return Some("continue"),
                Stmt::GotoCase(_) if switches == 0 => return Some("goto case"),
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::ForEach { body, .. } => {
                    if let Some(k) = rec(body, loops + 1, switches) {
                        return Some(k);
                    }
                }
                Stmt::For {
                    init, incr, body, ..
                } => {
                    if let Some(k) = rec(init, loops, switches) {
                        return Some(k);
                    }
                    if let Some(k) = rec(incr, loops, switches) {
                        return Some(k);
                    }
                    if let Some(k) = rec(body, loops + 1, switches) {
                        return Some(k);
                    }
                }
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if let Some(k) = rec(then_body, loops, switches) {
                        return Some(k);
                    }
                    if let Some(k) = rec(else_body, loops, switches) {
                        return Some(k);
                    }
                }
                Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                    if let Some(k) = rec(body, loops, switches) {
                        return Some(k);
                    }
                }
                Stmt::Switch { cases, tail, default, .. } => {
                    for (_, b) in cases {
                        if let Some(k) = rec(b, loops, switches + 1) {
                            return Some(k);
                        }
                    }
                    if let Some(arm) = default {
                        if let Some(k) = rec(arm, loops, switches + 1) {
                            return Some(k);
                        }
                    }
                    if let Some(k) = rec(tail, loops, switches + 1) {
                        return Some(k);
                    }
                }
                Stmt::StringSwitch { cases, tail, .. } => {
                    for (_, b) in cases {
                        if let Some(k) = rec(b, loops, switches + 1) {
                            return Some(k);
                        }
                    }
                    if let Some(t) = tail {
                        if let Some(k) = rec(t, loops, switches + 1) {
                            return Some(k);
                        }
                    }
                }
                Stmt::Try {
                    try_body,
                    catches,
                    finally_body,
                } => {
                    if let Some(k) = rec(try_body, loops, switches) {
                        return Some(k);
                    }
                    for (_, b) in catches {
                        if let Some(k) = rec(b, loops, switches) {
                            return Some(k);
                        }
                    }
                    if let Some(f) = finally_body {
                        if let Some(k) = rec(f, loops, switches) {
                            return Some(k);
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }
    rec(stmts, 0, 0)
}

/// Would nesting `list[i..=q]` inside a new block take a declaration out of
/// scope? Only direct typed declarations of the wrapped run can escape it, and
/// they escape when the name is used anywhere OUTSIDE the region — not just in
/// the same list's suffix. A use in an ancestor's tail (the region is the last
/// statement of an `if` arm) is exactly as out of scope after nesting, so the
/// whole-method occurrence count is compared with the region's own count.
fn region_decls_escape(
    list: &[Stmt],
    p: usize,
    q: usize,
    uses: &HashMap<String, usize>,
) -> bool {
    let mut decls = HashSet::new();
    for s in &list[p..=q] {
        if let Stmt::Assign(name, Some(_), _) = s {
            decls.insert(name.clone());
        }
    }
    if decls.is_empty() {
        return false;
    }
    let mut region_uses = HashMap::new();
    collect_names_stmts(&list[p..=q], &mut region_uses);
    decls.iter().any(|d| {
        uses.get(d).copied().unwrap_or(0) > region_uses.get(d).copied().unwrap_or(0)
    })
}

/// Remove labels in `stmts` that no jump in `stmts` targets. The caller has
/// already proved that no edge from outside reaches any of them.
fn drop_unreferenced_labels(stmts: &mut Vec<Stmt>) {
    let refs = count_refs(stmts);
    fn rec(list: &mut Vec<Stmt>, refs: &HashMap<String, usize>) {
        list.retain(|s| match s {
            Stmt::Label(l) => refs.get(l).copied().unwrap_or(0) > 0,
            _ => true,
        });
        for s in list.iter_mut() {
            for c in children_mut(s) {
                rec(c, refs);
            }
        }
    }
    rec(stmts, &refs);
}

// ---------------------------------------------------------------------------
// Fall-through analysis
// ---------------------------------------------------------------------------

/// Can execution reach the end of this block and continue after it?
fn block_falls_through(block: &[Stmt]) -> bool {
    let last = block.iter().rev().find(|s| {
        !matches!(
            s,
            Stmt::Label(_) | Stmt::Comment(_) | Stmt::Attribute { .. }
        )
    });
    match last {
        None => true,
        Some(s) => stmt_falls_through(s),
    }
}

fn stmt_falls_through(s: &Stmt) -> bool {
    match s {
        Stmt::Return(_)
        | Stmt::Throw(_)
        | Stmt::Break
        | Stmt::Continue
        | Stmt::Goto(_)
        | Stmt::GotoCase(_)
        | Stmt::YieldBreak => false,
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            else_body.is_empty()
                || block_falls_through(then_body)
                || block_falls_through(else_body)
        }
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Rule B: forward join inversion
// ---------------------------------------------------------------------------

fn rule_b_forward_join(root: &mut Vec<Stmt>) -> usize {
    let refs = count_refs(root);
    let defs = count_defs(root);
    rule_b_list(root, &refs, &defs)
}

/// Match `if (c) { <comments> goto L; }` with an empty/comment else arm.
fn inversion_match(s: &Stmt) -> Option<(Expr, String, Vec<Stmt>)> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = s
    else {
        return None;
    };
    let mut kept = Vec::new();
    for e in else_body {
        match e {
            Stmt::Comment(_) | Stmt::Attribute { .. } => kept.push(e.clone()),
            _ => return None,
        }
    }
    let mut target = None;
    for t in then_body {
        match t {
            Stmt::Comment(_) | Stmt::Attribute { .. } => kept.push(t.clone()),
            Stmt::Goto(l) if target.is_none() => target = Some(l.clone()),
            _ => return None,
        }
    }
    Some((cond.clone(), target?, kept))
}

/// The instruction run that executes after jumping to `target`, until control
/// leaves the arm. Recurses through `if` arms only: a label inside a loop,
/// switch or exception region is not reached by straight-line fall-through and
/// is not a candidate for duplication.
fn suffix_from_label(list: &[Stmt], target: &str) -> Option<Vec<Stmt>> {
    if let Some(pos) = find_label_pos(list, target) {
        return Some(list[pos + 1..].to_vec());
    }
    for (j, s) in list.iter().enumerate() {
        if let Stmt::If {
            then_body,
            else_body,
            ..
        } = s
        {
            let inner = suffix_from_label(then_body, target)
                .or_else(|| suffix_from_label(else_body, target));
            if let Some(mut suffix) = inner {
                // Falling out of the arm continues with the rest of the
                // enclosing list.
                suffix.extend_from_slice(&list[j + 1..]);
                return Some(suffix);
            }
        }
    }
    None
}

/// A suffix may be copied only when it is straight-line code whose duplication
/// does not double an effect *on one path*: each path still executes one copy.
/// Calls are refused (a duplicated call is a side-effecting expression in the
/// output), labels are refused (duplicate definitions), and any nested control
/// flow is refused (its `break`/`goto` would bind in the other arm).
fn cloneable_suffix(stmts: &[Stmt]) -> bool {
    stmts.iter().all(|s| match s {
        Stmt::Comment(_) | Stmt::Attribute { .. } => true,
        Stmt::Assign(_, None, rhs) => !expr_has_call(rhs),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => !expr_has_call(collection) && !expr_has_call(index) && !expr_has_call(value),
        Stmt::Eval(e) => !expr_has_call(e),
        _ => false,
    })
}

/// Target of the jump reached by falling through the trailing `else` chain of
/// `list`. `else if` chains render as nested `if`s, so the jumping arm is not
/// always the list's immediate last statement.
fn trailing_else_goto(list: &[Stmt]) -> Option<String> {
    let pos = list
        .iter()
        .rposition(|s| !matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. }))?;
    match &list[pos] {
        Stmt::Goto(t) => Some(t.clone()),
        Stmt::If { else_body, .. } if !else_body.is_empty() => trailing_else_goto(else_body),
        _ => None,
    }
}

/// Replace the jump found by [`trailing_else_goto`] with `suffix` (or remove
/// it when the suffix is empty). Returns whether a jump was replaced.
fn replace_trailing_else_goto(list: &mut Vec<Stmt>, suffix: &[Stmt]) -> bool {
    let Some(pos) = list
        .iter()
        .rposition(|s| !matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. }))
    else {
        return false;
    };
    match &mut list[pos] {
        Stmt::Goto(_) => {
            if suffix.is_empty() {
                list.remove(pos);
            } else {
                list.splice(pos..=pos, suffix.iter().cloned());
            }
            true
        }
        Stmt::If { else_body, .. } if !else_body.is_empty() => {
            replace_trailing_else_goto(else_body, suffix)
        }
        _ => false,
    }
}

/// Match the nested-target forward join:
/// `if (A) { … L: <pure suffix> } else … goto L;` where the jump is the
/// trailing edge of the else-if chain and `L` is uniquely defined.
fn suffix_copy_match(
    s: &Stmt,
    defs: &HashMap<String, usize>,
) -> Option<(String, Vec<Stmt>)> {
    let Stmt::If {
        then_body,
        else_body,
        ..
    } = s
    else {
        return None;
    };
    let target = trailing_else_goto(else_body)?;
    if defs.get(&target).copied().unwrap_or(0) != 1 {
        bump(&CENSUS.refuse_duplicate_label);
        return None;
    }
    // A declaration in the then arm may be in scope at the label but not in
    // the else arm; only a declaration-free arm is cloned.
    fn has_typed_decl(stmts: &[Stmt]) -> bool {
        let mut found = false;
        walk(stmts, &mut |s| {
            if matches!(s, Stmt::Assign(_, Some(_), _)) {
                found = true;
            }
        });
        found
    }
    if has_typed_decl(then_body) {
        bump(&CENSUS.refuse_scope_escape);
        return None;
    }
    let Some(suffix) = suffix_from_label(then_body, &target) else {
        bump(&CENSUS.refuse_unproven_target);
        return None;
    };
    if !cloneable_suffix(&suffix) {
        bump(&CENSUS.refuse_impure_suffix);
        return None;
    }
    Some((target, suffix))
}

fn rule_b_list(
    list: &mut Vec<Stmt>,
    refs: &HashMap<String, usize>,
    defs: &HashMap<String, usize>,
) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        let Some((cond, target, kept)) = inversion_match(&list[i]) else {
            // Nested-target variant: `if (A) { … if (B) { … } else { …; L: S } }
            // else { …; goto L; }` — copy the pure instruction run that starts
            // at `L` into the jumping arm. Duplication is limited to reads, so
            // the observable program is unchanged.
            if let Some((_target, suffix)) = suffix_copy_match(&list[i], defs) {
                let replaced = if let Stmt::If { else_body, .. } = &mut list[i] {
                    replace_trailing_else_goto(else_body, &suffix)
                } else {
                    false
                };
                if replaced {
                    bump(&CENSUS.b_suffix_copies);
                    n += 1;
                    continue;
                }
            }
            i += 1;
            continue;
        };
        if !is_side_effect_free(&cond) {
            bump(&CENSUS.refuse_side_effect_cond);
            i += 1;
            continue;
        }
        if defs.get(&target).copied().unwrap_or(0) != 1 {
            bump(&CENSUS.refuse_duplicate_label);
            i += 1;
            continue;
        }
        let Some(j) = list
            .iter()
            .enumerate()
            .skip(i + 1)
            .find(|(_, s)| matches!(s, Stmt::Label(l) if l == &target))
            .map(|(j, _)| j)
        else {
            i += 1;
            continue;
        };

        // `<rest>` is list[i+1..j]. Labels inside it may not be referenced
        // from outside: inversion nests them under the `if`.
        let rest_labels = labels_in(&list[i + 1..j]);
        let mut duplicate = false;
        for l in &rest_labels {
            if defs.get(l).copied().unwrap_or(0) != 1 {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            bump(&CENSUS.refuse_duplicate_label);
            i += 1;
            continue;
        }
        let inside_refs = count_refs(&list[i + 1..j]);
        let mut external = false;
        for l in &rest_labels {
            if refs.get(l).copied().unwrap_or(0) != inside_refs.get(l).copied().unwrap_or(0) {
                external = true;
                break;
            }
        }
        if external {
            bump(&CENSUS.refuse_external_entry);
            i += 1;
            continue;
        }
        // A typed declaration in `<rest>` used after the join would fall out
        // of scope when the statements are nested under the `if`.
        let mut decls = HashSet::new();
        for s in &list[i + 1..j] {
            if let Stmt::Assign(name, Some(_), _) = s {
                decls.insert(name.clone());
            }
        }
        if !decls.is_empty() {
            let mut used = HashSet::new();
            collect_names_stmts(&list[j + 1..], &mut used);
            if decls.iter().any(|d| used.contains(d)) {
                bump(&CENSUS.refuse_scope_escape);
                i += 1;
                continue;
            }
        }

        let mut rest: Vec<Stmt> = list.drain(i + 1..j).collect();
        let mut prefix = kept;
        prefix.append(&mut rest);
        if prefix.is_empty() {
            // Both routes reach L; the shell gated nothing.
            list.remove(i);
        } else {
            list[i] = Stmt::If {
                cond: Expr::Not(Box::new(cond)),
                then_body: prefix,
                else_body: Vec::new(),
            };
        }
        // The jump we just removed may have been the label's only reference.
        if refs.get(&target).copied().unwrap_or(0) == 1 {
            if let Some(k) = find_label_pos(list, &target) {
                list.remove(k);
            }
        }
        bump(&CENSUS.b_forward_joins);
        n += 1;
        if i >= list.len() {
            break;
        }
        i += 1;
    }
    for s in list.iter_mut() {
        for c in children_mut(s) {
            n += rule_b_list(c, refs, defs);
        }
    }
    n
}

// ---------------------------------------------------------------------------
// Rule C: exit normalization
// ---------------------------------------------------------------------------

fn rule_c_exit_normalization(root: &mut Vec<Stmt>) -> usize {
    let defs = count_defs(root);
    let mut n = 0;
    n += rule_c_drop_fallthrough(root, &defs);
    n += rule_c_loop_breaks(root, &defs);
    n += rule_c_method_end(root);
    n
}

/// A jump to a label in the immediate run after it in the same list is the
/// natural fall-through: drop the jump.
fn rule_c_drop_fallthrough(list: &mut Vec<Stmt>, defs: &HashMap<String, usize>) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        let target = match &list[i] {
            Stmt::Goto(t) => t.clone(),
            _ => {
                i += 1;
                continue;
            }
        };
        if defs.get(&target).copied().unwrap_or(0) != 1 {
            bump(&CENSUS.refuse_duplicate_label);
            i += 1;
            continue;
        }
        if label_run_after(list, i).iter().any(|l| *l == target) {
            list.remove(i);
            bump(&CENSUS.c_drops);
            n += 1;
            continue;
        }
        i += 1;
    }
    for s in list.iter_mut() {
        for c in children_mut(s) {
            n += rule_c_drop_fallthrough(c, defs);
        }
    }
    n
}

/// Any jump inside one loop to the label run directly after that loop becomes
/// `break`. Jumps nested in another loop/switch/try keep their `goto` (the
/// `break` would bind there; rule D handles the multi-level case).
fn rule_c_loop_breaks(list: &mut Vec<Stmt>, defs: &HashMap<String, usize>) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        if !is_loop_stmt(&list[i]) {
            i += 1;
            continue;
        }
        let exit = label_run_after(list, i);
        if !exit.is_empty() {
            let mut unique = true;
            for l in &exit {
                if defs.get(l).copied().unwrap_or(0) != 1 {
                    unique = false;
                    break;
                }
            }
            if unique {
                if let Some(body) = loop_body_mut(&mut list[i]) {
                    n += convert_exit_jumps(body, &exit, false);
                }
            } else {
                bump(&CENSUS.refuse_duplicate_label);
            }
        }
        i += 1;
    }
    for s in list.iter_mut() {
        for c in children_mut(s) {
            n += rule_c_loop_breaks(c, defs);
        }
    }
    n
}

fn is_loop_stmt(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::ForEach { .. } | Stmt::For { .. }
    )
}

fn loop_body_mut(s: &mut Stmt) -> Option<&mut Vec<Stmt>> {
    match s {
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::ForEach { body, .. }
        | Stmt::For { body, .. } => Some(body),
        _ => None,
    }
}

fn loop_body(s: &Stmt) -> Option<&Vec<Stmt>> {
    match s {
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::ForEach { body, .. }
        | Stmt::For { body, .. } => Some(body),
        _ => None,
    }
}

fn convert_exit_jumps(list: &mut Vec<Stmt>, exit: &[String], blocked: bool) -> usize {
    let mut n = 0;
    for s in list.iter_mut() {
        match s {
            Stmt::Goto(t) if !blocked && exit.iter().any(|l| l == t) => {
                *s = Stmt::Break;
                bump(&CENSUS.c_breaks);
                n += 1;
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                n += convert_exit_jumps(then_body, exit, blocked);
                n += convert_exit_jumps(else_body, exit, blocked);
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                n += convert_exit_jumps(body, exit, blocked);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. } => {
                n += convert_exit_jumps(body, exit, true);
            }
            Stmt::For {
                init, incr, body, ..
            } => {
                n += convert_exit_jumps(init, exit, true);
                n += convert_exit_jumps(incr, exit, true);
                n += convert_exit_jumps(body, exit, true);
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases.iter_mut() {
                    n += convert_exit_jumps(b, exit, true);
                }
                n += convert_exit_jumps(tail, exit, true);
                if let Some(arm) = default {
                    n += convert_exit_jumps(arm, exit, true);
                }
            }
            Stmt::StringSwitch { cases, tail, .. } => {
                for (_, b) in cases.iter_mut() {
                    n += convert_exit_jumps(b, exit, true);
                }
                if let Some(t) = tail {
                    n += convert_exit_jumps(t, exit, true);
                }
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                // Never cross an exception-region boundary.
                let before = count_exit_refs(try_body, exit)
                    + catches
                        .iter()
                        .map(|(_, b)| count_exit_refs(b, exit))
                        .sum::<usize>()
                    + finally_body
                        .as_ref()
                        .map(|f| count_exit_refs(f, exit))
                        .unwrap_or(0);
                for _ in 0..before {
                    bump(&CENSUS.refuse_try_boundary);
                }
            }
            _ => {}
        }
    }
    n
}

fn count_exit_refs(list: &[Stmt], exit: &[String]) -> usize {
    let mut n = 0;
    walk(list, &mut |s| {
        if matches!(s, Stmt::Goto(t) if exit.iter().any(|l| l == t)) {
            n += 1;
        }
    });
    n
}

/// `goto End;` to the method-end label becomes `return;` when the method is
/// void-like (no value return, no iterator yield anywhere).
fn rule_c_method_end(root: &mut Vec<Stmt>) -> usize {
    if !is_void_like(root) {
        return 0;
    }
    let mut ends = HashSet::new();
    for i in 0..root.len() {
        if let Stmt::Label(l) = &root[i] {
            if root[i + 1..].iter().all(|s| {
                matches!(
                    s,
                    Stmt::Label(_) | Stmt::Comment(_) | Stmt::Attribute { .. }
                )
            }) {
                ends.insert(l.clone());
            }
        }
    }
    if ends.is_empty() {
        return 0;
    }
    goto_end_to_return(root, &ends, false)
}

fn is_void_like(stmts: &[Stmt]) -> bool {
    let mut ok = true;
    walk(stmts, &mut |s| {
        if matches!(
            s,
            Stmt::Return(Some(_)) | Stmt::YieldReturn(_) | Stmt::YieldBreak
        ) {
            ok = false;
        }
    });
    ok
}

fn goto_end_to_return(list: &mut Vec<Stmt>, ends: &HashSet<String>, in_finally: bool) -> usize {
    let mut n = 0;
    for s in list.iter_mut() {
        match s {
            Stmt::Goto(t) if !in_finally && ends.contains(t.as_str()) => {
                *s = Stmt::Return(None);
                bump(&CENSUS.c_returns);
                n += 1;
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                n += goto_end_to_return(then_body, ends, in_finally);
                n += goto_end_to_return(else_body, ends, in_finally);
            }
            // A `return` is legal in a loop; a `finally` is not, and exception
            // regions are left entirely alone per the safety rule.
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. } => {
                n += goto_end_to_return(body, ends, in_finally);
            }
            Stmt::For {
                init, incr, body, ..
            } => {
                n += goto_end_to_return(init, ends, in_finally);
                n += goto_end_to_return(incr, ends, in_finally);
                n += goto_end_to_return(body, ends, in_finally);
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                n += goto_end_to_return(body, ends, in_finally);
            }
            _ => {}
        }
    }
    n
}

// ---------------------------------------------------------------------------
// Rule D: multi-level exits via a `done` flag
// ---------------------------------------------------------------------------

/// Which child list of a statement a path step descends into. Rule D only
/// walks transparent positions (`if` arms and loop bodies); switches and
/// exception regions are refused, so no selector exists for them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Sel {
    Then,
    Else,
    Body,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Step {
    idx: usize,
    sel: Sel,
}

fn child_list_mut<'a>(s: &'a mut Stmt, sel: Sel) -> Option<&'a mut Vec<Stmt>> {
    match (s, sel) {
        (Stmt::If { then_body, .. }, Sel::Then) => Some(then_body),
        (Stmt::If { else_body, .. }, Sel::Else) => Some(else_body),
        (
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. }
            | Stmt::For { body, .. },
            Sel::Body,
        ) => Some(body),
        (Stmt::Using { body, .. } | Stmt::Lock { body, .. }, Sel::Body) => Some(body),
        _ => None,
    }
}

/// Descend into the body of the loop at `loop_idx`, then through `path`.
fn descend_in_loop<'a>(
    root: &'a mut Vec<Stmt>,
    loop_idx: usize,
    path: &[Step],
) -> Option<&'a mut Vec<Stmt>> {
    let s = root.get_mut(loop_idx)?;
    let mut cur = loop_body_mut(s)?;
    for st in path {
        let s = cur.get_mut(st.idx)?;
        cur = child_list_mut(s, st.sel)?;
    }
    Some(cur)
}

#[derive(Clone)]
struct DeepJump {
    /// Path from the outer loop body to the list holding the jump.
    path: Vec<Step>,
    idx: usize,
    /// Ancestor loops between the jump and the outer loop, outermost first.
    /// Each entry is `(path from outer body to the loop's parent list, index)`.
    loops: Vec<(Vec<Step>, usize)>,
}

/// Find jumps to `targets` inside nested loops of the outer loop body.
/// Switch and exception bodies are not traversed: `break` there is not the
/// multi-level edge this rule knows how to express.
fn collect_deep_jumps(
    list: &[Stmt],
    path: &mut Vec<Step>,
    loops: &mut Vec<(Vec<Step>, usize)>,
    targets: &HashSet<String>,
    out: &mut Vec<DeepJump>,
) {
    for (idx, s) in list.iter().enumerate() {
        match s {
            Stmt::Goto(t) if targets.contains(t) && !loops.is_empty() => {
                out.push(DeepJump {
                    path: path.clone(),
                    idx,
                    loops: loops.clone(),
                });
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                for (sel, body) in [(Sel::Then, then_body), (Sel::Else, else_body)] {
                    path.push(Step { idx, sel });
                    collect_deep_jumps(body, path, loops, targets, out);
                    path.pop();
                }
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                path.push(Step { idx, sel: Sel::Body });
                collect_deep_jumps(body, path, loops, targets, out);
                path.pop();
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. } => {
                loops.push((path.clone(), idx));
                path.push(Step {
                    idx,
                    sel: Sel::Body,
                });
                collect_deep_jumps(body, path, loops, targets, out);
                path.pop();
                loops.pop();
            }
            Stmt::For { body, .. } => {
                loops.push((path.clone(), idx));
                path.push(Step {
                    idx,
                    sel: Sel::Body,
                });
                collect_deep_jumps(body, path, loops, targets, out);
                path.pop();
                loops.pop();
            }
            Stmt::Switch { .. } | Stmt::StringSwitch { .. } => {
                let n = count_targets_in(std::slice::from_ref(s), targets);
                for _ in 0..n {
                    bump(&CENSUS.refuse_switch_boundary);
                }
            }
            Stmt::Try { .. } => {
                let n = count_targets_in(std::slice::from_ref(s), targets);
                for _ in 0..n {
                    bump(&CENSUS.refuse_try_boundary);
                }
            }
            _ => {}
        }
    }
}

fn count_targets_in(stmts: &[Stmt], targets: &HashSet<String>) -> usize {
    let mut n = 0;
    walk(stmts, &mut |s| {
        if matches!(s, Stmt::Goto(t) if targets.contains(t)) {
            n += 1;
        }
    });
    n
}

fn guard_is_present(list: &[Stmt], idx: usize, done: &str) -> bool {
    match list.get(idx) {
        Some(Stmt::If {
            cond: Expr::Var(v),
            then_body,
            else_body,
        }) if v.as_str() == done && else_body.is_empty() => {
            matches!(then_body.as_slice(), [Stmt::Break])
        }
        _ => false,
    }
}

fn rule_d_multi_level_exits(root: &mut Vec<Stmt>, used: &mut HashSet<String>) -> usize {
    rule_d_list(root, used)
}

fn rule_d_list(list: &mut Vec<Stmt>, used: &mut HashSet<String>) -> usize {
    let defs = count_defs(list);
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        if !is_loop_stmt(&list[i]) {
            i += 1;
            continue;
        }
        let exit: HashSet<String> = label_run_after(list, i).into_iter().collect();
        if exit.is_empty() {
            i += 1;
            continue;
        }
        let targets: HashSet<String> = exit
            .iter()
            .filter(|l| defs.get(*l).copied().unwrap_or(0) == 1)
            .cloned()
            .collect();
        if targets.is_empty() {
            bump(&CENSUS.refuse_duplicate_label);
            i += 1;
            continue;
        }
        let body = match loop_body(&list[i]) {
            Some(b) => b,
            None => {
                i += 1;
                continue;
            }
        };
        let mut cands = Vec::new();
        {
            let mut path = Vec::new();
            let mut loops = Vec::new();
            collect_deep_jumps(body, &mut path, &mut loops, &targets, &mut cands);
        }
        if cands.is_empty() {
            i += 1;
            continue;
        }

        let done = unique_name("done", used);
        let mut decl_inserted = false;
        // One jump at a time, re-collecting indices each round: a rewrite
        // changes the statement count in the list that holds the jump.
        let mut rounds = 0usize;
        loop {
            rounds += 1;
            if rounds > 64 {
                bump(&CENSUS.refuse_unproven_target);
                break;
            }
            let loop_idx = if decl_inserted { i + 1 } else { i };
            let body = match loop_body(&list[loop_idx]) {
                Some(b) => b,
                None => break,
            };
            let mut fresh = Vec::new();
            {
                let mut path = Vec::new();
                let mut loops = Vec::new();
                collect_deep_jumps(body, &mut path, &mut loops, &targets, &mut fresh);
            }
            let Some(cand) = fresh.into_iter().next() else {
                break;
            };
            if !decl_inserted {
                list.insert(
                    i,
                    Stmt::Assign(done.clone(), Some("bool".into()), Expr::Bool(false)),
                );
                decl_inserted = true;
            }
            let loop_idx = i + 1;
            {
                let Some(targets_list) = descend_in_loop(list, loop_idx, &cand.path) else {
                    bump(&CENSUS.refuse_unproven_target);
                    break;
                };
                if !matches!(targets_list.get(cand.idx), Some(Stmt::Goto(_))) {
                    bump(&CENSUS.refuse_unproven_target);
                    break;
                }
                targets_list.splice(
                    cand.idx..=cand.idx,
                    vec![
                        Stmt::Assign(done.clone(), None, Expr::Bool(true)),
                        Stmt::Break,
                    ],
                );
            }
            // Propagate the exit one loop at a time, innermost first.
            for (path, loop_idx) in cand.loops.iter().rev() {
                let Some(parent) = descend_in_loop(list, i + 1, path) else {
                    bump(&CENSUS.refuse_unproven_target);
                    continue;
                };
                let at = loop_idx + 1;
                if at > parent.len() || guard_is_present(parent, at, &done) {
                    continue;
                }
                parent.insert(
                    at,
                    Stmt::If {
                        cond: Expr::Var(done.clone()),
                        then_body: vec![Stmt::Break],
                        else_body: Vec::new(),
                    },
                );
            }
            bump(&CENSUS.d_multilevel_exits);
            n += 1;
        }
        i = if decl_inserted { i + 2 } else { i + 1 };
    }
    for s in list.iter_mut() {
        for c in children_mut(s) {
            n += rule_d_list(c, used);
        }
    }
    n
}

// ---------------------------------------------------------------------------
// Rule E: jump to a loop header → `continue`
// ---------------------------------------------------------------------------

fn rule_e_header_continues(root: &mut Vec<Stmt>) -> usize {
    let defs = count_defs(root);
    rule_e_list(root, &defs)
}

fn rule_e_list(list: &mut Vec<Stmt>, defs: &HashMap<String, usize>) -> usize {
    let mut n = 0;
    for i in 0..list.len() {
        if !matches!(list[i], Stmt::While { .. }) {
            continue;
        }
        // Every label in the run immediately before the `while` is a header
        // label: jumping to any of them falls through to the same condition.
        let mut headers = Vec::new();
        let mut j = i;
        while j > 0 {
            j -= 1;
            match &list[j] {
                Stmt::Comment(_) | Stmt::Attribute { .. } => continue,
                Stmt::Label(l) => {
                    if defs.get(l).copied().unwrap_or(0) != 1 {
                        bump(&CENSUS.refuse_duplicate_label);
                        headers.clear();
                        break;
                    }
                    headers.push(l.clone());
                }
                _ => break,
            }
        }
        if headers.is_empty() {
            continue;
        }
        let Stmt::While { body, .. } = &mut list[i] else {
            unreachable!();
        };
        for h in &headers {
            n += convert_header_jumps(body, h, false);
        }
    }
    for s in list.iter_mut() {
        for c in children_mut(s) {
            n += rule_e_list(c, defs);
        }
    }
    n
}

/// Rewrite jumps to `header` into `continue`. `blocked` is set inside a nested
/// loop, switch or exception region, where `continue` would bind elsewhere (or
/// not be legal).
fn convert_header_jumps(list: &mut Vec<Stmt>, header: &str, blocked: bool) -> usize {
    if blocked {
        return 0;
    }
    let mut n = 0;
    for s in list.iter_mut() {
        match s {
            Stmt::Goto(t) if t == header => {
                *s = Stmt::Continue;
                bump(&CENSUS.e_header_continues);
                n += 1;
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                n += convert_header_jumps(then_body, header, false);
                n += convert_header_jumps(else_body, header, false);
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
                n += convert_header_jumps(body, header, false);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. } => {
                n += convert_header_jumps(body, header, true);
            }
            Stmt::For {
                init, incr, body, ..
            } => {
                n += convert_header_jumps(init, header, true);
                n += convert_header_jumps(incr, header, true);
                n += convert_header_jumps(body, header, true);
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases.iter_mut() {
                    n += convert_header_jumps(b, header, true);
                }
                n += convert_header_jumps(tail, header, true);
                if let Some(arm) = default {
                    n += convert_header_jumps(arm, header, true);
                }
            }
            Stmt::StringSwitch { cases, tail, .. } => {
                for (_, b) in cases.iter_mut() {
                    n += convert_header_jumps(b, header, true);
                }
                if let Some(t) = tail {
                    n += convert_header_jumps(t, header, true);
                }
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                for _ in 0..count_header_refs(try_body, header) {
                    bump(&CENSUS.refuse_try_boundary);
                }
                for (_, b) in catches.iter() {
                    for _ in 0..count_header_refs(b, header) {
                        bump(&CENSUS.refuse_try_boundary);
                    }
                }
                if let Some(f) = finally_body {
                    for _ in 0..count_header_refs(f, header) {
                        bump(&CENSUS.refuse_try_boundary);
                    }
                }
            }
            _ => {}
        }
    }
    n
}

fn count_header_refs(list: &[Stmt], header: &str) -> usize {
    let mut n = 0;
    walk(list, &mut |s| {
        if matches!(s, Stmt::Goto(t) if t == header) {
            n += 1;
        }
    });
    n
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::CmpKind;

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    fn eval(name: &str) -> Stmt {
        Stmt::Eval(var(name))
    }

    fn cond() -> Expr {
        Expr::Cmp(CmpKind::Eq, Box::new(var("c")), Box::new(Expr::Int(0)))
    }

    fn if_goto(target: &str) -> Stmt {
        Stmt::If {
            cond: cond(),
            then_body: vec![Stmt::Goto(target.to_string())],
            else_body: vec![],
        }
    }

    fn goto_count(stmts: &[Stmt]) -> usize {
        let mut n = 0;
        for s in stmts {
            if matches!(s, Stmt::Goto(_)) {
                n += 1;
            }
            let mut kids = Vec::new();
            children(s, &mut kids);
            for k in kids {
                n += goto_count(k);
            }
        }
        n
    }

    fn has_label(stmts: &[Stmt], name: &str) -> bool {
        let mut found = false;
        walk(stmts, &mut |s| {
            if matches!(s, Stmt::Label(l) if l == name) {
                found = true;
            }
        });
        found
    }

    fn label_count(stmts: &[Stmt], name: &str) -> usize {
        let mut n = 0;
        walk(stmts, &mut |s| {
            if matches!(s, Stmt::Label(l) if l == name) {
                n += 1;
            }
        });
        n
    }

    fn break_count(stmts: &[Stmt]) -> usize {
        let mut n = 0;
        walk(stmts, &mut |s| {
            if matches!(s, Stmt::Break) {
                n += 1;
            }
        });
        n
    }

    // ------------------------------------------------------------------
    // Rule A: back edge → loop
    // ------------------------------------------------------------------

    #[test]
    fn backward_edge_becomes_infinite_loop() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            eval("work"),
            Stmt::Goto("L".into()),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::While { cond, body } => {
                assert!(matches!(cond, Expr::Bool(true)));
                assert_eq!(body.len(), 2);
                assert!(matches!(&body[0], Stmt::Eval(_)));
                assert!(matches!(&body[1], Stmt::Continue));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    #[test]
    fn conditional_backward_edge_becomes_continue() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            eval("work"),
            if_goto("L"),
            eval("tail"),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[1], Stmt::If { .. }));
                match &body[1] {
                    Stmt::If { then_body, .. } => {
                        assert!(matches!(&then_body[0], Stmt::Continue))
                    }
                    _ => unreachable!(),
                }
                // The body can fall out of its end: a trailing break lands on
                // the same statement the fall-through did.
                assert!(matches!(body.last(), Some(Stmt::Break)));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    #[test]
    fn goto_after_region_becomes_break() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            eval("a"),
            if_goto("M"),
            eval("b"),
            Stmt::Goto("L".into()),
            Stmt::Label("M".into()),
            eval("after"),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                match &body[1] {
                    Stmt::If { then_body, .. } => {
                        assert!(matches!(&then_body[0], Stmt::Break))
                    }
                    other => panic!("expected If, got {other:?}"),
                }
                assert!(matches!(body.last(), Some(Stmt::Continue)));
            }
            other => panic!("expected While, got {other:?}"),
        }
        assert!(has_label(&stmts, "M"));
    }

    #[test]
    fn goto_to_label_after_intervening_statement_is_not_break() {
        // Break lands immediately after the loop, before `Eval(between)`;
        // the jump targets the label after it, so it must stay a goto.
        let mut stmts = vec![
            Stmt::Label("L".into()),
            if_goto("M"),
            Stmt::Goto("L".into()),
            eval("between"),
            Stmt::Label("M".into()),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[0], Stmt::If { .. }));
                assert!(goto_count(body) == 1, "the forward jump must remain");
            }
            other => panic!("expected While, got {other:?}"),
        }
        assert!(has_label(&stmts, "M"));
    }

    #[test]
    fn external_entry_into_region_is_refused() {
        let mut stmts = vec![
            Stmt::Goto("L".into()), // from outside the [L..goto L] region
            Stmt::Label("L".into()),
            eval("work"),
            Stmt::Goto("L".into()),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 0, "a jump from outside enters the region; refuse");
        assert!(matches!(&stmts[0], Stmt::Goto(_)));
        assert!(has_label(&stmts, "L"));
    }

    #[test]
    fn external_ref_to_inner_label_is_refused() {
        let mut stmts = vec![
            Stmt::Goto("X".into()),
            Stmt::Label("L".into()),
            Stmt::Label("X".into()),
            eval("a"),
            Stmt::Goto("L".into()),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 0);
        assert!(has_label(&stmts, "X"));
    }

    #[test]
    fn free_break_in_region_is_refused() {
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![
                    Stmt::Label("L".into()),
                    Stmt::Break, // belongs to the surrounding loop
                    Stmt::Goto("L".into()),
                ],
            },
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 0, "wrapping would re-bind the break to the new loop");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn typed_declaration_escaping_region_is_refused() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::Assign("x".into(), Some("int".into()), Expr::Int(1)),
            Stmt::Goto("L".into()),
            eval("x"),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 0, "x would leave the loop's scope");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn nested_loop_jump_back_is_not_wrapped_a_second_time() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::Goto("L".into())],
            },
            Stmt::Goto("L".into()),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        // The direct back edge forms the loop; the nested-loop edge keeps its
        // `goto` and the label. A second wrap would convert nothing (the only
        // remaining edge is inside the inner loop), so it is refused.
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 1);
        assert!(has_label(&stmts, "L"));
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[0], Stmt::Label(l) if l == "L"));
                assert!(matches!(&body[1], Stmt::While { .. }));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    #[test]
    fn wrappers_nest_and_both_back_edges_disappear() {
        let mut stmts = vec![
            Stmt::Label("A".into()),
            eval("a"),
            Stmt::Label("B".into()),
            eval("b"),
            Stmt::Goto("B".into()),
            Stmt::Goto("A".into()),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert!(n >= 1);
        assert_eq!(goto_count(&stmts), 0);
    }

    // ------------------------------------------------------------------
    // Rule B: forward join inversion
    // ------------------------------------------------------------------

    #[test]
    fn forward_join_is_inverted() {
        let mut stmts = vec![
            if_goto("L"),
            eval("a"),
            eval("b"),
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        match &stmts[0] {
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                assert!(matches!(cond, Expr::Not(_)));
                assert_eq!(then_body.len(), 2);
                assert!(else_body.is_empty());
            }
            other => panic!("expected If, got {other:?}"),
        }
        // The label had exactly one reference and is gone.
        assert!(!has_label(&stmts, "L"));
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn forward_join_with_other_reference_keeps_label() {
        let mut stmts = vec![
            if_goto("L"),
            eval("a"),
            Stmt::Label("L".into()),
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![],
            },
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 1);
        assert!(has_label(&stmts, "L"));
        assert_eq!(label_count(&stmts, "L"), 1);
    }

    #[test]
    fn empty_forward_join_shell_is_deleted() {
        let mut stmts = vec![
            if_goto("L"),
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        assert!(!has_label(&stmts, "L"));
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn dereferencing_condition_is_inverted() {
        // A dereference observes memory but has no side effect; both spellings
        // evaluate it exactly once (and would fault identically).
        let deref = Expr::Cmp(
            CmpKind::Lt,
            Box::new(Expr::RawDeref(Box::new(var("p")), 8, 4)),
            Box::new(Expr::Int(0)),
        );
        let mut stmts = vec![
            Stmt::If {
                cond: deref,
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![],
            },
            eval("a"),
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        assert!(matches!(&stmts[0], Stmt::If { cond: Expr::Not(_), .. }));
    }

    #[test]
    fn side_effecting_condition_is_not_inverted() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Call {
                    recv: None,
                    name: "Check".into(),
                    args: vec![],
                },
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![],
            },
            eval("a"),
            Stmt::Label("L".into()),
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn forward_join_with_external_inner_label_is_refused() {
        let mut stmts = vec![if_goto("L"), Stmt::Label("X".into()), eval("a"), Stmt::Label("L".into())];
        // An edge from outside the would-be `rest` region enters X.
        stmts.insert(0, Stmt::Goto("X".into()));
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 0);
        assert!(has_label(&stmts, "X"));
        assert_eq!(goto_count(&stmts), 2);
    }

    #[test]
    fn nested_forward_join_copies_pure_suffix() {
        // `if (A) { if (B) floor; else { prep; L: suffix } }
        //  else if (C) ceil; else goto L;`
        // The jump executes only `suffix`; copying that pure run into the
        // jumping arm removes the jump without changing any effect.
        let mut stmts = vec![
            Stmt::If {
                cond: var("A"),
                then_body: vec![Stmt::If {
                    cond: var("B"),
                    then_body: vec![eval("floor")],
                    else_body: vec![
                        eval("prep"),
                        Stmt::Label("L".into()),
                        eval("suffix"),
                    ],
                }],
                else_body: vec![Stmt::If {
                    cond: var("C"),
                    then_body: vec![eval("ceil")],
                    else_body: vec![Stmt::Goto("L".into())],
                }],
            },
            eval("after"),
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        match &stmts[0] {
            Stmt::If { else_body, .. } => match &else_body[0] {
                Stmt::If { else_body: inner, .. } => {
                    assert!(matches!(&inner[0], Stmt::Eval(e) if matches!(e, Expr::Var(v) if v == "suffix")));
                }
                other => panic!("expected nested If, got {other:?}"),
            },
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn nested_forward_join_with_call_suffix_is_refused() {
        let call = Stmt::Eval(Expr::Call {
            recv: None,
            name: "Touch".into(),
            args: vec![],
        });
        let mut stmts = vec![
            Stmt::If {
                cond: var("A"),
                then_body: vec![Stmt::If {
                    cond: var("B"),
                    then_body: vec![eval("floor")],
                    else_body: vec![Stmt::Label("L".into()), call],
                }],
                else_body: vec![Stmt::If {
                    cond: var("C"),
                    then_body: vec![eval("ceil")],
                    else_body: vec![Stmt::Goto("L".into())],
                }],
            },
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 0, "a call may not be duplicated");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn nested_forward_join_copies_straight_line_assignments() {
        // The real shape (ResizeWidget): the label's suffix is a short run of
        // local copies, not one expression. Each path still executes one copy.
        let mut stmts = vec![
            Stmt::If {
                cond: var("A"),
                then_body: vec![Stmt::If {
                    cond: var("B"),
                    then_body: vec![eval("floor")],
                    else_body: vec![
                        Stmt::Assign("prep".into(), None, Expr::Int(0)),
                        Stmt::Label("L".into()),
                        Stmt::Assign("x".into(), None, Expr::Var("u".into())),
                        Stmt::Assign("y".into(), None, Expr::Var("v".into())),
                    ],
                }],
                else_body: vec![Stmt::If {
                    cond: var("C"),
                    then_body: vec![eval("ceil")],
                    else_body: vec![Stmt::Goto("L".into())],
                }],
            },
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        match &stmts[0] {
            Stmt::If { else_body, .. } => match &else_body[0] {
                Stmt::If { else_body: inner, .. } => {
                    assert_eq!(inner.len(), 2);
                    assert!(matches!(&inner[0], Stmt::Assign(n, None, _) if n == "x"));
                }
                other => panic!("expected nested If, got {other:?}"),
            },
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn nested_forward_join_with_control_flow_suffix_is_refused() {
        let mut stmts = vec![
            Stmt::If {
                cond: var("A"),
                then_body: vec![Stmt::If {
                    cond: var("B"),
                    then_body: vec![eval("floor")],
                    else_body: vec![
                        Stmt::Label("L".into()),
                        Stmt::If {
                            cond: var("D"),
                            then_body: vec![Stmt::Break],
                            else_body: vec![],
                        },
                    ],
                }],
                else_body: vec![Stmt::If {
                    cond: var("C"),
                    then_body: vec![eval("ceil")],
                    else_body: vec![Stmt::Goto("L".into())],
                }],
            },
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 0, "control flow may not be cloned into another arm");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn nested_forward_join_with_empty_suffix_drops_the_jump() {
        let mut stmts = vec![
            Stmt::If {
                cond: var("A"),
                then_body: vec![Stmt::If {
                    cond: var("B"),
                    then_body: vec![eval("floor")],
                    else_body: vec![eval("prep"), Stmt::Label("L".into())],
                }],
                else_body: vec![Stmt::If {
                    cond: var("C"),
                    then_body: vec![eval("ceil")],
                    else_body: vec![Stmt::Goto("L".into())],
                }],
            },
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn forward_join_with_typed_decl_used_after_is_refused() {
        let mut stmts = vec![
            if_goto("L"),
            Stmt::Assign("x".into(), Some("int".into()), Expr::Int(1)),
            Stmt::Label("L".into()),
            eval("x"),
        ];
        let n = rule_b_forward_join(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    // ------------------------------------------------------------------
    // Rule C: exit normalization
    // ------------------------------------------------------------------

    #[test]
    fn loop_tail_goto_becomes_break() {
        let mut stmts = vec![
            Stmt::While {
                cond: cond(),
                body: vec![eval("work"), Stmt::Goto("L".into())],
            },
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rule_c_exit_normalization(&mut stmts);
        assert!(n >= 1);
        assert_eq!(goto_count(&stmts), 0);
        assert_eq!(break_count(&stmts), 1);
    }

    #[test]
    fn non_tail_goto_in_loop_becomes_break() {
        let mut stmts = vec![
            Stmt::While {
                cond: cond(),
                body: vec![
                    Stmt::If {
                        cond: cond(),
                        then_body: vec![Stmt::Goto("L".into())],
                        else_body: vec![],
                    },
                    eval("work"),
                ],
            },
            Stmt::Label("L".into()),
        ];
        let n = rule_c_exit_normalization(&mut stmts);
        assert!(n >= 1);
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn nested_loop_goto_is_left_for_rule_d() {
        let mut stmts = vec![
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::While {
                    cond: cond(),
                    body: vec![Stmt::Goto("L".into())],
                }],
            },
            Stmt::Label("L".into()),
            // Keep L away from the method end: a jump there would legitimately
            // become `return;`, hiding the rule under test.
            eval("after"),
        ];
        let n = rule_c_exit_normalization(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
        assert_eq!(break_count(&stmts), 0);
    }

    #[test]
    fn goto_after_same_list_label_is_dropped() {
        let mut stmts = vec![
            Stmt::Goto("L".into()),
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let defs = count_defs(&stmts);
        let n = rule_c_drop_fallthrough(&mut stmts, &defs);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn goto_to_method_end_becomes_return_in_void_method() {
        let mut stmts = vec![eval("work"), Stmt::Goto("L".into()), Stmt::Label("L".into())];
        let n = rule_c_method_end(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[1], Stmt::Return(None)));
        assert!(has_label(&stmts, "L"));
    }

    #[test]
    fn goto_to_method_end_is_kept_in_value_method() {
        let mut stmts = vec![
            Stmt::Goto("L".into()),
            Stmt::Return(Some(Expr::Int(1))),
            Stmt::Label("L".into()),
        ];
        let n = rule_c_method_end(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn goto_inside_try_is_not_normalized() {
        let mut stmts = vec![
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::Try {
                    try_body: vec![Stmt::Goto("L".into())],
                    catches: vec![],
                    finally_body: None,
                }],
            },
            Stmt::Label("L".into()),
        ];
        let n = rule_c_exit_normalization(&mut stmts);
        assert_eq!(n, 0, "never cross an exception-region boundary");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn goto_inside_finally_never_becomes_return() {
        let mut stmts = vec![
            Stmt::Try {
                try_body: vec![eval("t")],
                catches: vec![],
                finally_body: Some(vec![Stmt::Goto("L".into())]),
            },
            Stmt::Label("L".into()),
        ];
        let n = rule_c_method_end(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    // ------------------------------------------------------------------
    // Rule D: multi-level exits
    // ------------------------------------------------------------------

    #[test]
    fn multi_level_exit_uses_done_flag() {
        let mut stmts = vec![
            Stmt::While {
                cond: cond(),
                body: vec![
                    eval("outer_work"),
                    Stmt::While {
                        cond: cond(),
                        body: vec![Stmt::If {
                            cond: cond(),
                            then_body: vec![Stmt::Goto("L".into())],
                            else_body: vec![],
                        }],
                    },
                    eval("after_inner"),
                ],
            },
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1);
        assert_eq!(goto_count(&stmts), 0, "the multi-level goto is structured");
        // `bool done = false;` before the outer loop.
        assert!(matches!(
            &stmts[0],
            Stmt::Assign(name, Some(t), Expr::Bool(false)) if name == "done" && t == "bool"
        ));
        let outer = match &stmts[1] {
            Stmt::While { body, .. } => body,
            other => panic!("expected While, got {other:?}"),
        };
        // `if (done) break;` after the inner loop.
        match &outer[1] {
            Stmt::While { body, .. } => {
                // done = true; break; inside the if arm
                match &body[0] {
                    Stmt::If { then_body, .. } => {
                        assert!(matches!(&then_body[0], Stmt::Assign(n, None, Expr::Bool(true)) if n == "done"));
                        assert!(matches!(&then_body[1], Stmt::Break));
                    }
                    other => panic!("expected If, got {other:?}"),
                }
            }
            other => panic!("expected inner While, got {other:?}"),
        }
        match &outer[2] {
            Stmt::If { cond, then_body, .. } => {
                assert!(matches!(cond, Expr::Var(n) if n == "done"));
                assert!(matches!(&then_body[0], Stmt::Break));
            }
            other => panic!("expected guard after inner loop, got {other:?}"),
        }
    }

    #[test]
    fn multi_level_exit_from_try_is_refused() {
        let mut stmts = vec![
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::While {
                    cond: cond(),
                    body: vec![Stmt::Try {
                        try_body: vec![Stmt::Goto("L".into())],
                        catches: vec![],
                        finally_body: None,
                    }],
                }],
            },
            Stmt::Label("L".into()),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn multi_level_exit_reuses_one_done_flag_for_two_jumps() {
        let mut stmts = vec![
            Stmt::While {
                cond: cond(),
                body: vec![
                    Stmt::While {
                        cond: cond(),
                        body: vec![Stmt::Goto("L".into())],
                    },
                    Stmt::While {
                        cond: cond(),
                        body: vec![Stmt::Goto("L".into())],
                    },
                ],
            },
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 2);
        assert_eq!(goto_count(&stmts), 0);
        let decls = stmts
            .iter()
            .filter(|s| {
                matches!(s, Stmt::Assign(name, Some(t), Expr::Bool(false)) if name == "done" && t == "bool")
            })
            .count();
        assert_eq!(decls, 1, "one done flag serves both jumps");
    }

    #[test]
    fn done_name_avoids_collision() {
        let mut stmts = vec![
            Stmt::Assign("done".into(), None, Expr::Int(0)),
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::While {
                    cond: cond(),
                    body: vec![Stmt::Goto("L".into())],
                }],
            },
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1);
        assert_eq!(goto_count(&stmts), 0);
        assert!(stmts.iter().any(|s| matches!(
            s,
            Stmt::Assign(name, Some(t), Expr::Bool(false))
                if name == "done_1" && t == "bool"
        )));
    }

    // ------------------------------------------------------------------
    // Rule E: jump to the loop header → continue
    // ------------------------------------------------------------------

    #[test]
    fn goto_to_while_header_becomes_continue() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![eval("work"), if_goto("L")],
            },
            eval("after"),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1);
        assert_eq!(goto_count(&stmts), 0);
        match &stmts[1] {
            Stmt::While { body, .. } => match &body[1] {
                Stmt::If { then_body, .. } => {
                    assert!(matches!(&then_body[0], Stmt::Continue));
                }
                other => panic!("expected If, got {other:?}"),
            },
            other => panic!("expected While, got {other:?}"),
        }
        assert!(has_label(&stmts, "L"));
    }

    #[test]
    fn goto_to_loop_header_through_label_run_becomes_continue() {
        let mut stmts = vec![
            Stmt::Label("A".into()),
            Stmt::Comment("cold".into()),
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::Goto("A".into())],
            },
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn goto_to_outer_header_beside_nested_loop_becomes_continue() {
        // The jump is in the outer body (after an inner loop), so `continue`
        // addresses the loop the label heads.
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![
                    Stmt::While {
                        cond: cond(),
                        body: vec![eval("inner")],
                    },
                    Stmt::Goto("L".into()),
                ],
            },
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn goto_inside_nested_loop_to_outer_header_is_kept() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::While {
                    cond: cond(),
                    body: vec![if_goto("L")],
                }],
            },
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 0, "a `continue` would bind to the inner loop");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn goto_inside_switch_to_while_header_is_kept() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::Switch {
                    disc: "x".into(),
                    cases: vec![(0, vec![Stmt::Goto("L".into())])],
                    tail: vec![],
                    default: None,
                }],
            },
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 0, "switch arms are left alone conservatively");
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn goto_to_do_while_header_is_kept() {
        // Re-entering a `do` executes its body; `continue` jumps to the
        // condition at the bottom. Not equivalent.
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::DoWhile {
                body: vec![if_goto("L")],
                cond: cond(),
            },
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn goto_to_for_header_is_kept() {
        // Re-entering a `for` runs its initializer; `continue` does not.
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::For {
                init: vec![Stmt::Assign("i".into(), Some("int".into()), Expr::Int(0))],
                cond: Some(cond()),
                incr: vec![eval("i++")],
                body: vec![if_goto("L")],
            },
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    #[test]
    fn header_goto_becomes_continue_inside_if_and_using() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![
                    Stmt::If {
                        cond: cond(),
                        then_body: vec![Stmt::Goto("L".into())],
                        else_body: vec![],
                    },
                    Stmt::Using {
                        decl: Box::new(Stmt::Eval(var("h"))),
                        body: vec![Stmt::Goto("L".into())],
                    },
                ],
            },
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 2);
        assert_eq!(goto_count(&stmts), 0);
    }

    #[test]
    fn external_goto_to_header_keeps_the_label() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::Goto("L".into())],
            },
            eval("between"),
            Stmt::Goto("L".into()),
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 1, "only the in-loop jump becomes `continue`");
        assert_eq!(goto_count(&stmts), 1);
        assert!(has_label(&stmts, "L"));
    }

    #[test]
    fn duplicate_header_label_is_refused() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            Stmt::While {
                cond: cond(),
                body: vec![Stmt::Goto("L".into())],
            },
            eval("between"),
            Stmt::Label("L".into()),
            eval("after"),
        ];
        let n = rule_e_header_continues(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(goto_count(&stmts), 1);
    }

    // ------------------------------------------------------------------
    // Regression fixtures
    // ------------------------------------------------------------------

    #[test]
    fn wrapped_region_keeps_statements_after_the_back_edge() {
        let mut stmts = vec![
            Stmt::Label("L".into()),
            eval("body"),
            Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Goto("L".into())],
                else_body: vec![],
            },
            eval("tail_of_region"),
            Stmt::Label("M".into()),
            eval("after_region"),
        ];
        let n = rule_a_backward_loops(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(goto_count(&stmts), 0);
        // Every statement that used to follow the last back edge survives.
        let text = format!("{stmts:?}");
        assert!(text.contains("tail_of_region"));
        assert!(text.contains("after_region"));
        assert!(has_label(&stmts, "M"));
    }

    /// Mirrors `RewardChestHandler.ShowIconrevealingAnimation` (0x15A07DC):
    /// two early conditional jumps to a tail label, the UIHandler calls and a
    /// return, then a loop whose body jumps back past the label. The old
    /// structurer dropped the whole UIHandler tail when it wrapped the
    /// backward region; those calls are the canary.
    #[test]
    fn canary_ui_handler_tail_survives_structuring() {
        let mut stmts = vec![
            if_goto("L4"),
            if_goto("L4"),
            eval("work"),
            Stmt::Label("L34".into()),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "this.UIHandler.InitializeGridsAndDetailsScreen".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "this.UIHandler.DisableRVFromGridAndDetailsScreen".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "base.StartCoroutine".into(),
                args: vec![],
            }),
            Stmt::Return(None),
            Stmt::Label("L4".into()),
            Stmt::While {
                cond: var("moveNext"),
                body: vec![
                    eval("body_work"),
                    if_goto("L34"),
                    Stmt::While {
                        cond: var("inner"),
                        body: vec![eval("inner_work")],
                    },
                ],
            },
            eval("tail_after"),
            Stmt::Label("L91".into()),
            eval("after_label"),
        ];
        let before_calls = stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Eval(Expr::Call { name, .. }) if name.starts_with("this.UIHandler")))
            .count();
        assert_eq!(before_calls, 2);

        apply(&mut stmts);

        let mut init = 0;
        let mut disable = 0;
        walk(&stmts, &mut |s| {
            if let Stmt::Eval(Expr::Call { name, .. }) = s {
                if name == "this.UIHandler.InitializeGridsAndDetailsScreen" {
                    init += 1;
                }
                if name == "this.UIHandler.DisableRVFromGridAndDetailsScreen" {
                    disable += 1;
                }
            }
        });
        assert_eq!(init, 1, "the UIHandler tail must survive exactly once");
        assert_eq!(disable, 1, "the UIHandler tail must survive exactly once");
        let text = format!("{stmts:?}");
        assert!(text.contains("tail_after"), "code after the loop must survive");
        assert!(text.contains("base.StartCoroutine"));
        assert!(text.contains("Return"), "the early return must survive");
        assert!(has_label(&stmts, "L34"));
        assert!(has_label(&stmts, "L91"));
    }

    // ------------------------------------------------------------------
    // Whole-pass behaviours
    // ------------------------------------------------------------------

    #[test]
    fn empty_body_is_noop() {
        let mut stmts = vec![];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn apply_never_leaves_a_dangling_jump() {
        let mut stmts = vec![
            Stmt::Label("A".into()),
            eval("a"),
            Stmt::Goto("B".into()),
            Stmt::Label("B".into()),
            eval("b"),
            Stmt::Goto("A".into()),
            Stmt::Label("C".into()),
            Stmt::Return(None),
        ];
        apply(&mut stmts);
        let mut labels = HashSet::new();
        walk(&stmts, &mut |s| {
            if let Stmt::Label(l) = s {
                labels.insert(l.clone());
            }
        });
        walk(&stmts, &mut |s| {
            if let Stmt::Goto(l) = s {
                assert!(labels.contains(l), "goto {l} has no label");
            }
        });
    }

    #[test]
    fn apply_is_stable_on_structured_code() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![eval("work"), Stmt::If {
                cond: cond(),
                then_body: vec![Stmt::Break],
                else_body: vec![Stmt::Continue],
            }],
        }];
        let before = format!("{stmts:?}");
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(format!("{stmts:?}"), before);
    }
}
