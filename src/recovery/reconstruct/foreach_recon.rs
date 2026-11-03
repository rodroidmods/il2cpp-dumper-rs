//! `foreach` reconstruction from GetEnumerator/MoveNext/Dispose patterns.
//!
//! ## The shape that occurs
//!
//! The IL2CPP decompiler outputs ~10,384 `GetEnumerator` calls but only recovers
//! ~123 `foreach` loops. The dominant pattern is a 3-statement sequence:
//!
//! ```text
//! collection.GetEnumerator(&enumerator);
//! while ((&enumerator).MoveNext() & 1) != 0) {
//!   ... use (&enumerator).Current ...
//! }
//! (&enumerator).Dispose();
//! ```
//!
//! Two sub-patterns are recognized:
//!
//! 1. **Void-call form**: `Eval(Call { recv: Some(coll), name: "GetEnumerator",
//!    args: [addr] })` — the enumerator is initialized via an out-parameter
//!    (the address of a stack slot). MoveNext/Dispose/Current are called on the
//!    same address expression.
//!
//! 2. **Assignment form**: `Assign(var, _, Call { recv: Some(coll), name:
//!    "GetEnumerator", args: [] })` — GetEnumerator returns the enumerator
//!    directly. MoveNext/Dispose/Current are called on `Var(var)`.
//!
//! ## The rewrite
//!
//! The 3-statement sequence is replaced with:
//!
//! ```text
//! foreach (var item in collection) {
//!   ... use item ...
//! }
//! ```
//!
//! All accesses to `enumerator.Current` (field access or `get_Current()` call)
//! inside the while body are replaced with the fresh loop variable.
//!
//! ## Safety checks
//!
//! - GetEnumerator, MoveNext, and Dispose must all reference the SAME enumerator
//!   (compared by rendered expression text).
//! - The three statements must be consecutive (no gap for other enumerator uses).
//! - A failed match is a NO-OP: the flat sequence stays intact.
//!
//! ## Instrumentation
//!
//! | Counter | Meaning |
//! |---|---|
//! | `SITES_EXAMINED` | Every 3-statement window examined |
//! | `FOLDED` | Successfully folded into a `foreach` |
//! | `REFUSED_NO_GETENUM` | First statement was not a GetEnumerator call |
//! | `REFUSED_NO_WHILE` | Second statement was not a While with MoveNext |
//! | `REFUSED_NO_DISPOSE` | Third statement was not a Dispose on same enumerator |

use super::super::csharp::{BinOp, CmpKind};
use super::{
    expr_has_call, expr_mentions, list_writes_deep, render_text, target_names, walk_mut, Expr, Stmt,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Instrumentation
// ---------------------------------------------------------------------------

static SITES_EXAMINED: AtomicU64 = AtomicU64::new(0);
static FOLDED: AtomicU64 = AtomicU64::new(0);
static REFUSED_NO_GETENUM: AtomicU64 = AtomicU64::new(0);
static REFUSED_NO_WHILE: AtomicU64 = AtomicU64::new(0);
static REFUSED_NO_DISPOSE: AtomicU64 = AtomicU64::new(0);
static REFUSED_COLLECTION_MUTATED: AtomicU64 = AtomicU64::new(0);

// Index-loop (`i = 0; while (i < arr.Length) { item = arr[i]; …; i++ }`)
// instrumentation. Kept separate from the enumerator population: the two
// shapes share no gate, and a zero on one must not mask progress on the other.
static INDEX_SITES: AtomicU64 = AtomicU64::new(0);
static INDEX_FOLDED: AtomicU64 = AtomicU64::new(0);

/// Debug logging gated on `ILR_FEDBG`.
macro_rules! fed_dbg {
    ($($arg:tt)*) => {
        if std::env::var("ILR_FEDBG").is_ok() {
            eprintln!("[foreach-dbg] {}", format!($($arg)*));
        }
    };
}

/// Deterministic loop-variable naming: `item`, `item2`, … skipping every name
/// already present in the method body. The old version used a process-global
/// atomic, so the same method got `item2` or `item537` depending on which
/// worker thread folded first — the dump was not reproducible run to run
/// (measured: 124 of 317 sampled methods differed between two runs of the
/// same binary). Allocation order here is the textual fold order.
fn fresh_loop_name(used: &mut HashSet<String>) -> String {
    if used.insert("item".to_string()) {
        return "item".to_string();
    }
    let mut k = 2usize;
    loop {
        let c = format!("item{k}");
        if used.insert(c.clone()) {
            return c;
        }
        k += 1;
    }
}

fn collect_names(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Assign(n, _, _) => {
            out.insert(n.clone());
        }
        Stmt::ForEach { var_name, .. } => {
            out.insert(var_name.clone());
        }
        _ => {}
    }
    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
        collect_expr_names(e, out)
    });
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            collect_names(x, out);
        }
    }
}

fn collect_expr_names(e: &Expr, out: &mut HashSet<String>) {
    if let Expr::Var(n) = e {
        out.insert(n.clone());
    }
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
        collect_expr_names(c, out);
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the foreach recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut used = HashSet::new();
    for s in stmts.iter() {
        collect_names(s, &mut used);
    }
    let mut n = 0;
    walk_mut(stmts, &mut |list| {
        n += rewrite_list(list, &mut used);
    });
    n
}

// ---------------------------------------------------------------------------
// List-level rewriting
// ---------------------------------------------------------------------------

fn rewrite_list(list: &mut Vec<Stmt>, used: &mut HashSet<String>) -> usize {
    // Index loops first: `i = 0; while (i < arr.Length) { item = arr[i]; … }`.
    // Doing them before the enumerator scan means a `for` produced by
    // `for_loop` can still be re-expressed as `foreach`.
    let mut rewrites = rewrite_index_loops(list);
    // The compiler's own spelling: `GetEnumerator(); try { while (MoveNext)
    // { … } } finally { Dispose(); }`. Once the EH pass renders the compiler
    // finally (instead of an ABI catch wrapper), the Dispose is no longer a
    // third consecutive statement, so the three-statement matcher below can
    // never see this shape.
    {
        let mut i = 0;
        while i < list.len() {
            let Some((_collection, enumerator_expr)) = match_getenumerator(&list[i]) else {
                i += 1;
                continue;
            };
            let Some(base) = enum_base_name(&enumerator_expr) else {
                i += 1;
                continue;
            };
            // Sret/Current setup spills sit between GetEnumerator and the try
            // (`intPtrValue = enumerator;`, `*(&enumerator +16) = …`). The
            // compiler emits NO user statement between the enumerator's
            // creation and its foreach try, so a bounded run of plain
            // assignments/evals here is buffer initialisation and is dropped
            // with the wrapper. Control flow (labels/gotos/branches) is a
            // hard stop.
            let mut j = i + 1;
            while j < list.len() && j <= i + 8 && is_enum_setup(&list[j], &base) {
                j += 1;
            }
            fed_dbg!("wrapped scan i={i} j={j} try={}", j < list.len() && matches!(list[j], Stmt::Try { .. }));
            let matched = if j < list.len() && matches!(list[j], Stmt::Try { .. }) {
                try_match_foreach_wrapped(&list[i], &list[j])
            } else {
                None
            };
            let Some((collection_str, enum_text, loop_body, tail)) = matched else {
                i += 1;
                continue;
            };
            let current_var = fresh_loop_name(used);
            if list_writes_deep(&loop_body, &collection_str) {
                REFUSED_COLLECTION_MUTATED.fetch_add(1, Ordering::Relaxed);
                i += 1;
                continue;
            }
            let new_body = rewrite_current_accesses(loop_body, &enum_text, &current_var);
            let foreach_stmt = Stmt::ForEach {
                var_name: current_var,
                collection: collection_str,
                body: new_body,
            };
            let mut replacement = vec![foreach_stmt];
            replacement.extend(tail);
            list.splice(i..=j, replacement);
            rewrites += 1;
            FOLDED.fetch_add(1, Ordering::Relaxed);
        }
    }
    let mut i = 0;
    while i + 2 < list.len() {
        SITES_EXAMINED.fetch_add(1, Ordering::Relaxed);
        match try_match_foreach(&list[i..i + 3]) {
            Some(matched) => {
                let (collection_str, enum_text, loop_body) = matched;
                let current_var = fresh_loop_name(used);

                // Safety check: refuse rewrite if the loop body writes to the
                // collection variable. A foreach over a collection that is
                // modified inside the loop body would throw at runtime in C#
                // (IEnumerator contract), so the original while-loop form is
                // semantically different from a foreach and must not be folded.
                if list_writes_deep(&loop_body, &collection_str) {
                    REFUSED_COLLECTION_MUTATED.fetch_add(1, Ordering::Relaxed);
                    i += 1;
                    continue;
                }

                let new_body = rewrite_current_accesses(loop_body, &enum_text, &current_var);
                let foreach_stmt = Stmt::ForEach {
                    var_name: current_var,
                    collection: collection_str,
                    body: new_body,
                };
                list.splice(i..i + 3, std::iter::once(foreach_stmt));
                rewrites += 1;
                FOLDED.fetch_add(1, Ordering::Relaxed);
                // Don't increment i — check for another foreach at same position.
            }
            None => {
                i += 1;
            }
        }
    }
    rewrites
}

// ---------------------------------------------------------------------------
// foreach from an array/List index loop
// ---------------------------------------------------------------------------
//
// ## The shape
//
// ```text
// i = 0;
// while (i < arr.Length)          // or `*(int*)(arr +24)` / `*(nint*)(arr +24)`
// {
//     item = arr[i];              // the only use of `i` besides the update
//     …body…
//     i = i + 1;
// }
// ```
//
// becomes `foreach (var item in arr) { …body… }`.
//
// `for_loop` runs first, so the same loop may already have become
// `for (i = 0; i < arr.Length; i = i + 1) { item = arr[i]; … }`; both spellings
// are accepted here.
//
// ## The proof, gate by gate
//
//   * **the bound names the same collection the element is loaded from.**
//     `i < arr.Length` with the load `arr[i]`; the bound may also be the raw
//     IL2CPP length slot `*(int*)(arr +24)` / `*(nint*)(arr +24)` (array
//     `max_length` or `List<T>._size` — both types implement `IEnumerable`).
//   * **the collection expression is loop-invariant.** It must be a stable
//     path (`this.items`, a local, a static field — no calls, no indexers) and
//     the body must not write it. A reassigned `arr` would make `while` read
//     the NEXT array's length/index while `foreach` keeps iterating the
//     original reference.
//   * **the index is used for exactly three things:** the condition, the
//     single load, and the canonical `i = i + 1` update. Any other mention —
//     including an indexer/event write that hides it — refuses.
//   * **the load is the first executable statement and the update is the last**
//     (a trailing `continue` is dropped: it is a no-op in a `foreach`). A
//     `continue` before the update would skip it in the flat form but not in a
//     `foreach`, so it refuses.
//   * **the element local is read-only inside the loop and dead afterwards.**
//     A `foreach` iteration variable is not assignable; and its scope ends at
//     the loop, so a later read of the original variable would change.
//   * **the index is dead after the loop.** Its declaration (the `i = 0`) is
//     deleted, so every later mention — and every mention between the init and
//     the loop — would observe a different scope.

/// One planned index-loop rewrite. `init_at` is `None` for the `For` spelling,
/// whose initialiser lives in the loop header already.
struct IndexFold {
    at: usize,
    init_at: Option<usize>,
    foreach_stmt: Stmt,
}

fn rewrite_index_loops(list: &mut Vec<Stmt>) -> usize {
    // Count the population once: the fold loop rescans this list.
    for s in list.iter() {
        if matches!(s, Stmt::While { .. } | Stmt::For { .. }) {
            INDEX_SITES.fetch_add(1, Ordering::Relaxed);
        }
    }
    let mut n = 0usize;
    loop {
        let mut folded = false;
        for i in 0..list.len() {
            let Some(fold) = plan_index_fold(list, i) else {
                continue;
            };
            apply_index_fold(list, fold);
            n += 1;
            INDEX_FOLDED.fetch_add(1, Ordering::Relaxed);
            folded = true;
            break;
        }
        if !folded {
            return n;
        }
    }
}

fn apply_index_fold(list: &mut Vec<Stmt>, fold: IndexFold) {
    let mut out: Vec<Stmt> = Vec::with_capacity(list.len());
    for (idx, s) in std::mem::take(list).into_iter().enumerate() {
        if Some(idx) == fold.init_at {
            continue; // the `i = 0` declaration is subsumed by `foreach`
        }
        if idx == fold.at {
            out.push(fold.foreach_stmt.clone());
            continue;
        }
        out.push(s);
    }
    *list = out;
}

fn plan_index_fold(list: &[Stmt], i: usize) -> Option<IndexFold> {
    // `update_in_body`: the `while` spelling carries `i = i + 1` as the last
    // body statement, the `for` spelling carries it in the header. The header
    // form must not require (or look for) a body update.
    let (cond, body, init_at, update_in_body) = match &list[i] {
        Stmt::While { cond, body } => {
            let (idx, _) = index_condition(cond)?;
            let init_at = zero_init_before(list, i, &idx)?;
            (cond, body, Some(init_at), true)
        }
        Stmt::For {
            init,
            cond,
            incr,
            body,
        } => {
            let cond = cond.as_ref()?;
            let (idx, _) = index_condition(cond)?;
            let [Stmt::Assign(init_name, _, Expr::Int(0))] = init.as_slice() else {
                return None;
            };
            if init_name != &idx {
                return None;
            }
            let [Stmt::Assign(update_name, _, update_rhs)] = incr.as_slice() else {
                return None;
            };
            if update_name != &idx || !is_canonical_increment(update_rhs, &idx) {
                return None;
            }
            (cond, body, None, false)
        }
        _ => return None,
    };
    let (idx, collection) = index_condition(cond)?;

    // The load must be the first executable statement: `item = arr[i]`.
    let load_pos = first_meaningful(body)?;
    let (item, loaded_from) = match_index_load(&body[load_pos], &idx)?;
    if !super::exprs_eq(&loaded_from, &collection) {
        return None;
    }
    let collection_text = render_text(&collection);
    if item == idx || item == collection_text {
        return None;
    }

    // The update must be the last executable statement (one trailing
    // `continue` and trailing comments are allowed).
    let mut end = body.len();
    while end > 0
        && matches!(
            body[end - 1],
            Stmt::Comment(_) | Stmt::Attribute { .. }
        )
    {
        end -= 1;
    }
    let mut continue_at = None;
    if end > 0 && matches!(body[end - 1], Stmt::Continue) {
        continue_at = Some(end - 1);
        end -= 1;
        while end > 0
            && matches!(
                body[end - 1],
                Stmt::Comment(_) | Stmt::Attribute { .. }
            )
        {
            end -= 1;
        }
    }
    if end == 0 {
        return None;
    }
    let update_at = if update_in_body {
        let at = end - 1;
        if at == load_pos {
            return None;
        }
        let Stmt::Assign(update_name, _, update_rhs) = &body[at] else {
            return None;
        };
        if update_name != &idx || !is_canonical_increment(update_rhs, &idx) {
            return None;
        }
        Some(at)
    } else {
        None
    };

    // Everything that survives in the loop body.
    let mut new_body: Vec<Stmt> = Vec::with_capacity(body.len());
    for (j, s) in body.iter().enumerate() {
        if j == load_pos || Some(j) == update_at || Some(j) == continue_at {
            continue;
        }
        new_body.push(s.clone());
    }

    // No other use of the index, at any depth.
    if list_mentions_any(&new_body, &idx) {
        fed_dbg!("index {idx} used for more than the load/update/condition");
        return None;
    }
    // A `continue` before the update skips it in the flat form but would run
    // the next iteration in a `foreach`.
    if super::contains_continue(&new_body) {
        fed_dbg!("continue before the update");
        return None;
    }
    // The element local must survive the rewrite unchanged.
    if list_writes_var(&new_body, &item) {
        fed_dbg!("element local {item} written in the body");
        return None;
    }
    // A reassigned collection would change what the `while` iterates.
    if list_writes_deep(&new_body, &collection_text) {
        fed_dbg!("collection {collection_text} written in the body");
        return None;
    }
    // Both locals die at the loop: their declarations are deleted by the fold.
    let after = &list[i + 1..];
    if after.iter().any(|s| stmt_mentions_any(s, &idx))
        || after.iter().any(|s| stmt_mentions_any(s, &item))
    {
        fed_dbg!("index or element local live after the loop");
        return None;
    }
    if let Some(p) = init_at {
        // Deleting `i = 0` is only invisible when nothing between the init and
        // the loop mentions `i` at all (a read would otherwise see the old
        // value). The nearest-init scan already excludes an intervening write.
        if list[p + 1..i].iter().any(|s| stmt_mentions_any(s, &idx)) {
            fed_dbg!("index {idx} mentioned between init and loop");
            return None;
        }
    }

    fed_dbg!(
        "FOLDED index loop at {i}: foreach (var {item} in {collection_text}) \
         ({} loop(s) seen, {} folded)",
        INDEX_SITES.load(Ordering::Relaxed),
        INDEX_FOLDED.load(Ordering::Relaxed) + 1
    );

    Some(IndexFold {
        at: i,
        init_at,
        foreach_stmt: Stmt::ForEach {
            var_name: item,
            collection: collection_text,
            body: new_body,
        },
    })
}

/// `i < <collection>.Length` (or the raw IL2CPP length slot of `collection`).
fn index_condition(cond: &Expr) -> Option<(String, Expr)> {
    let Expr::Cmp(CmpKind::Lt, lhs, rhs) = cond else {
        return None;
    };
    let Expr::Var(idx) = lhs.as_ref() else {
        return None;
    };
    let collection = length_owner(rhs)?;
    if !stable_collection(collection) {
        return None;
    }
    Some((idx.clone(), collection.clone()))
}

/// The collection whose length this expression reads.
fn length_owner(e: &Expr) -> Option<&Expr> {
    match e {
        // `array_length` rewrites `*(nint*)(x +24)` to `x.Length`; the two raw
        // spellings below are what remains when it did not run (4-byte List
        // `_size`, 8-byte array `max_length`).
        Expr::Field(recv, field) if field == "Length" => Some(recv),
        Expr::RawDeref(base, 24, 4) | Expr::RawDeref(base, 24, 8) => Some(base),
        _ => None,
    }
}

/// A loop-invariant path: no calls and no indexed/dereferenced intermediate
/// that could change identity between two evaluations.
fn stable_collection(e: &Expr) -> bool {
    if expr_has_call(e) {
        return false;
    }
    match e {
        Expr::Var(_) | Expr::This | Expr::StaticField(..) => true,
        Expr::Field(recv, _) => stable_collection(recv),
        Expr::Cast(_, inner) => stable_collection(inner),
        _ => false,
    }
}

/// The nearest preceding assignment to `idx` before the loop, when it is the
/// literal zero.
fn zero_init_before(list: &[Stmt], i: usize, idx: &str) -> Option<usize> {
    let p = (0..i)
        .rev()
        .find(|&p| matches!(&list[p], Stmt::Assign(n, _, _) if n == idx))?;
    let Stmt::Assign(_, _, Expr::Int(0)) = &list[p] else {
        return None;
    };
    Some(p)
}

/// The first statement that is not a comment/attribute.
fn first_meaningful(body: &[Stmt]) -> Option<usize> {
    body.iter()
        .position(|s| !matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. }))
}

/// `item = arr[idx]` → `(item, arr)`.
fn match_index_load(stmt: &Stmt, idx: &str) -> Option<(String, Expr)> {
    let Stmt::Assign(name, _, Expr::Index { collection, index }) = stmt else {
        return None;
    };
    if !matches!(index.as_ref(), Expr::Var(n) if n == idx) {
        return None;
    }
    if !is_simple_local(name) {
        return None;
    }
    Some((name.clone(), collection.as_ref().clone()))
}

/// A plain C# local name (no field/index/pointer spelling).
fn is_simple_local(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        && !name.chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// `i = i + 1` / `i = 1 + i`.
fn is_canonical_increment(e: &Expr, idx: &str) -> bool {
    let Expr::Bin(BinOp::Add, a, b) = e else {
        return false;
    };
    let one = |e: &Expr| matches!(e, Expr::Int(1));
    let var = |e: &Expr| matches!(e, Expr::Var(n) if n == idx);
    (var(a) && one(b)) || (one(a) && var(b))
}

/// Does any statement (or nested body) read or write `name`?
///
/// Complete over the statement kinds — the shared `stmt_mentions_deep` does not
/// look at `IndexAssign`/`EventAssign`, and the deletion of the index's
/// declaration makes every such mention load-bearing.
fn stmt_mentions_any(s: &Stmt, name: &str) -> bool {
    let own = match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => expr_mentions(e, name),
        Stmt::Assign(lhs, _, e) => {
            lhs == name || target_names(lhs, name) || expr_mentions(e, name)
        }
        Stmt::EventAssign { field, value, .. } => {
            expr_mentions(field, name) || expr_mentions(value, name)
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            expr_mentions(collection, name)
                || expr_mentions(index, name)
                || expr_mentions(value, name)
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            expr_mentions(cond, name)
        }
        Stmt::For { init, cond, incr, .. } => {
            cond.as_ref().is_some_and(|c| expr_mentions(c, name))
                || list_mentions_any(init, name)
                || list_mentions_any(incr, name)
        }
        Stmt::ForEach {
            var_name,
            collection,
            ..
        } => var_name == name || collection == name,
        Stmt::Using { decl, .. } => stmt_mentions_any(decl, name),
        Stmt::Lock { expr, .. } => expr_mentions(expr, name),
        Stmt::Return(Some(e)) => expr_mentions(e, name),
        Stmt::Switch { disc, .. } => disc == name,
        Stmt::StringSwitch { disc, .. } => expr_mentions(disc, name),
        _ => false,
    };
    if own {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    kids.iter().any(|k| list_mentions_any(k, name))
}

fn list_mentions_any(list: &[Stmt], name: &str) -> bool {
    list.iter().any(|s| stmt_mentions_any(s, name))
}

/// Is the element local assigned anywhere in the list (including through a
/// target path like `item.field`)? A `foreach` iteration variable cannot be
/// the target of an assignment, and mutating a value-type copy is illegal.
fn list_writes_var(list: &[Stmt], name: &str) -> bool {
    list.iter().any(|s| {
        let own = match s {
            Stmt::Assign(lhs, _, _) => lhs == name || target_names(lhs, name),
            Stmt::IndexAssign { collection, .. } => expr_mentions(collection, name),
            Stmt::EventAssign { field, .. } => expr_mentions(field, name),
            _ => false,
        };
        if own {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        kids.iter().any(|k| list_writes_var(k, name))
    })
}

// ---------------------------------------------------------------------------
// Pattern matching
// ---------------------------------------------------------------------------

/// Result of a successful 3-statement pattern match.
/// `(collection_rendered, enumerator_rendered, while_body, loop_var_name)`
type MatchResult = (String, String, Vec<Stmt>);

/// Try to match a foreach pattern in a 3-statement window.
fn try_match_foreach(triple: &[Stmt]) -> Option<MatchResult> {
    if triple.len() < 3 {
        return None;
    }

    // Step 1: Extract GetEnumerator call from triple[0].
    let (collection_expr, enumerator_expr) = match match_getenumerator(&triple[0]) {
        Some(pair) => pair,
        None => {
            REFUSED_NO_GETENUM.fetch_add(1, Ordering::Relaxed);
            return None;
        }
    };

    // The flat triple has no enclosing try to prove that the MoveNext target
    // is the enumerator just created, so identity must stay textual here.
    // (The try/finally matcher can use the loop receiver because the
    // GetEnumerator directly precedes the Try.)
    let enum_text = render_text(&enumerator_expr);
    let loop_body = match match_while_movenext(&triple[1], &enum_text) {
        Some(body) => body,
        None => {
            REFUSED_NO_WHILE.fetch_add(1, Ordering::Relaxed);
            return None;
        }
    };

    if !match_dispose(&triple[2], &enum_text) {
        REFUSED_NO_DISPOSE.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    let collection_str = render_text(&collection_expr);
    Some((collection_str, enum_text, loop_body))
}

/// The compiler's `foreach` spelling: `GetEnumerator(); try { while
/// (MoveNext) { … } } finally { Dispose(); }`.
///
/// `foreach` compiles to exactly that shape, so once the EH pass renders the
/// compiler finally (rather than an ABI catch wrapper), the enumerator triple
/// is split across a Try and the three-statement matcher cannot fire. Every
/// gate here is the same proof the flat matcher makes: one GetEnumerator, one
/// MoveNext loop on that enumerator, one Dispose on it in the finally, and
/// nothing else executable in either list.
fn try_match_foreach_wrapped(
    getenum: &Stmt,
    try_stmt: &Stmt,
) -> Option<(String, String, Vec<Stmt>, Vec<Stmt>)> {
    let (collection_expr, _enumerator_expr) = match match_getenumerator(getenum) {
        Some(v) => v,
        None => { fed_dbg!("wrapped: not GetEnumerator"); return None; }
    };
    let Stmt::Try {
        try_body,
        catches,
        finally_body,
    } = try_stmt
    else {
        return None;
    };
    if !catches.is_empty() {
        fed_dbg!("wrapped: catches={}", catches.len());
        return None;
    }
    let Some(fin) = finally_body.as_ref() else { fed_dbg!("wrapped: no finally"); return None; };
    let mut try_exec = try_body
        .iter()
        .filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_)));
    // Identity is the LOOP's own MoveNext receiver, not the GetEnumerator
    // argument's spelling. They are the same address-taken cell, but the two
    // SSA paths can carry different display names (`&enumerator` vs
    // `&enumerator2`); comparing the GetEnumerator text to the While text
    // dropped 35 corpus loops after the buffer was coalesced.
    let (loop_body, enum_text) = match try_exec.next() {
        Some(Stmt::While { cond, body }) => match find_movenext_recv(cond) {
            Some(recv) => (body.clone(), render_text(&recv)),
            None => {
                fed_dbg!("wrapped: no MoveNext recv");
                REFUSED_NO_WHILE.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        },
        _ => {
            fed_dbg!("wrapped: first exec not While");
            REFUSED_NO_WHILE.fetch_add(1, Ordering::Relaxed);
            return None;
        }
    };
    let mut fin_exec = fin
        .iter()
        .filter(|s| !matches!(s, Stmt::Label(_) | Stmt::Comment(_)));
    // The pad is a separate SSA path, so the same address-taken cell can be
    // spelled with a second display name here (`enumerator` vs
    // `enumerator2`). In the compiler's foreach shape the finally contains
    // exactly one local address's Dispose; accept it by shape, not by text.
    match (fin_exec.next(), fin_exec.next()) {
        (Some(d), None) if match_dispose_any_local(d) => {}
        _ => {
            fed_dbg!("wrapped: finally not one local Dispose");
            REFUSED_NO_DISPOSE.fetch_add(1, Ordering::Relaxed);
            return None;
        }
    }
    // The structurer can place the post-loop code INSIDE the try (its range
    // often reaches the loop's secondary exit). It runs after the loop in the
    // source, so moving it after the `foreach` is exact — provided it is a
    // plain exit/assignment that does not touch the enumerator. A repeated
    // `<enum>.Dispose()` here is the ABI finally's copy; the `finally` clause
    // owns the dispose, so drop it.
    let mut tail: Vec<Stmt> = Vec::new();
    for s in try_exec {
        match s {
            Stmt::Eval(Expr::Call {
                recv: Some(_),
                name,
                args,
            }) if is_dispose_name(name) && args.is_empty() && match_dispose_any_local(s) => {}
            Stmt::Return(e) => {
                if e.as_ref().is_some_and(|x| render_text(x).contains(&enum_text)) {
                    return None;
                }
                tail.push(s.clone());
            }
            Stmt::Throw(e) => {
                if render_text(e).contains(&enum_text) {
                    return None;
                }
                tail.push(s.clone());
            }
            other => { fed_dbg!("wrapped: unsafe tail {:?}", other); return None; }
        }
    }
    let collection_str = render_text(&collection_expr);
    Some((collection_str, enum_text, loop_body, tail))
}

/// ABI initialisation of the enumerator buffer, safe to skip between
/// `GetEnumerator` and its `try/finally`: a copy of the buffer local, or a
/// store whose pre-rendered target addresses the buffer (`&e + off`). Any
/// OTHER statement there may be real source code; skipping it would delete it.
fn is_enum_setup(s: &Stmt, base: &str) -> bool {
    match s {
        Stmt::Assign(target, _, e) => {
            target.contains(&format!("&{base}")) || matches!(e, Expr::Var(n) if n == base)
        }
        _ => false,
    }
}

/// The local name behind an enumerator address expression (`&e` or `e`).
fn enum_base_name(e: &Expr) -> Option<String> {
    match e {
        Expr::AddressOf(inner) => match inner.as_ref() {
            Expr::Var(n) => Some(n.clone()),
            _ => None,
        },
        Expr::Var(n) => Some(n.clone()),
        _ => None,
    }
}

/// Extract the collection expression and enumerator expression from a
/// GetEnumerator statement. Returns `None` if the statement is not a
/// GetEnumerator call.
///
/// Two forms are recognized:
///
/// 1. Void-call: `Eval(Call { recv: Some(coll), name: "GetEnumerator",
///    args: [enumerator_addr] })`
/// 2. Assignment: `Assign(var, _, Call { recv: Some(coll), name:
///    "GetEnumerator", args: [] })` — enumerator is `Var(var)`.
fn match_getenumerator(stmt: &Stmt) -> Option<(Expr, Expr)> {
    match stmt {
        // Void-call form: collection.GetEnumerator(&enumerator)
        Stmt::Eval(Expr::Call {
            recv: Some(coll),
            name,
            args,
        }) if is_getenumerator_name(name) && args.len() == 1 => {
            Some((*coll.clone(), args[0].clone()))
        }
        // Assignment form: enumerator = collection.GetEnumerator()
        Stmt::Assign(var, _, Expr::Call {
            recv: Some(coll),
            name,
            args,
        }) if is_getenumerator_name(name) && args.is_empty() => {
            Some((*coll.clone(), Expr::Var(var.clone())))
        }
        _ => None,
    }
}

/// Is this method name a GetEnumerator call?
fn is_getenumerator_name(name: &str) -> bool {
    name == "GetEnumerator" || name.ends_with(".GetEnumerator")
}

/// Extract the while body if the statement is a While whose condition contains
/// a MoveNext call on the given enumerator.
fn match_while_movenext(stmt: &Stmt, enum_text: &str) -> Option<Vec<Stmt>> {
    match stmt {
        Stmt::While { cond, body } if expr_has_movenext_on(cond, enum_text) => {
            Some(body.clone())
        }
        _ => None,
    }
}

/// Does the Dispose statement call Dispose on the given enumerator?
fn match_dispose(stmt: &Stmt, enum_text: &str) -> bool {
    match stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(recv),
            name,
            args,
        }) if is_dispose_name(name) && args.is_empty() && render_text(recv) == enum_text => true,
        _ => false,
    }
}

/// Is this method name a Dispose call?
fn is_dispose_name(name: &str) -> bool {
    name == "Dispose" || name.ends_with(".Dispose")
}

/// `<local address>.Dispose()` — the compiler-finally shape. Used only by the
/// wrapped matcher, where the receiver's display name can differ from the
/// loop's spelling even though it is the same address-taken cell.
fn match_dispose_any_local(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::Eval(Expr::Call {
            recv: Some(r),
            name,
            args,
        }) if is_dispose_name(name)
            && args.is_empty()
            && matches!(r.as_ref(), Expr::AddressOf(inner) if matches!(inner.as_ref(), Expr::Var(_)))
    )
}

// ---------------------------------------------------------------------------
// Expression helpers
// ---------------------------------------------------------------------------

/// The receiver of the first `MoveNext` call on a LOCAL ADDRESS in `e`.
///
/// The compiler's foreach loop moves the enumerator whose address it took;
/// the display name on this path is the one the body's `Current` reads carry,
/// so it is the identity the Current rewrite must use.
fn find_movenext_recv(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Call {
            recv: Some(r),
            name,
            ..
        } if is_movenext_name(name)
            && matches!(r.as_ref(), Expr::AddressOf(inner) if matches!(inner.as_ref(), Expr::Var(_))) =>
        {
            Some((**r).clone())
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            find_movenext_recv(a).or_else(|| find_movenext_recv(b))
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) => {
            find_movenext_recv(x)
        }
        Expr::Cond(c, a, b) => find_movenext_recv(c)
            .or_else(|| find_movenext_recv(a))
            .or_else(|| find_movenext_recv(b)),
        _ => None,
    }
}

/// Does `e` contain a MoveNext call whose receiver renders as `enum_text`?
/// Recursively searches through binary operators, comparisons, casts, etc.
fn expr_has_movenext_on(e: &Expr, enum_text: &str) -> bool {
    match e {
        Expr::Call {
            recv: Some(recv),
            name,
            ..
        } if is_movenext_name(name) && render_text(recv) == enum_text => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            expr_has_movenext_on(a, enum_text) || expr_has_movenext_on(b, enum_text)
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) => {
            expr_has_movenext_on(x, enum_text)
        }
        Expr::Cond(c, a, b) => {
            expr_has_movenext_on(c, enum_text)
                || expr_has_movenext_on(a, enum_text)
                || expr_has_movenext_on(b, enum_text)
        }
        _ => false,
    }
}

/// Is this method name a MoveNext call?
fn is_movenext_name(name: &str) -> bool {
    name == "MoveNext" || name.ends_with(".MoveNext")
}

// ---------------------------------------------------------------------------
// Current access rewriting
// ---------------------------------------------------------------------------

/// Rewrite all statements in the body, replacing `enumerator.Current` accesses
/// with the fresh loop variable.
fn rewrite_current_accesses(body: Vec<Stmt>, enum_text: &str, loop_var: &str) -> Vec<Stmt> {
    body.into_iter()
        .map(|s| rewrite_stmt_current(s, enum_text, loop_var))
        .collect()
}

/// Rewrite one statement, replacing Current accesses in its expressions.
fn rewrite_stmt_current(stmt: Stmt, enum_text: &str, loop_var: &str) -> Stmt {
    match stmt {
        Stmt::Eval(e) => Stmt::Eval(rewrite_expr_current(e, enum_text, loop_var)),
        Stmt::Assign(name, ty, e) => {
            Stmt::Assign(name, ty, rewrite_expr_current(e, enum_text, loop_var))
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => Stmt::If {
            cond: rewrite_expr_current(cond, enum_text, loop_var),
            then_body: rewrite_current_accesses(then_body, enum_text, loop_var),
            else_body: rewrite_current_accesses(else_body, enum_text, loop_var),
        },
        Stmt::While { cond, body } => Stmt::While {
            cond: rewrite_expr_current(cond, enum_text, loop_var),
            body: rewrite_current_accesses(body, enum_text, loop_var),
        },
        Stmt::DoWhile { body, cond } => Stmt::DoWhile {
            body: rewrite_current_accesses(body, enum_text, loop_var),
            cond: rewrite_expr_current(cond, enum_text, loop_var),
        },
        Stmt::For {
            init,
            cond,
            incr,
            body,
        } => Stmt::For {
            init: rewrite_current_accesses(init, enum_text, loop_var),
            cond: cond.map(|c| rewrite_expr_current(c, enum_text, loop_var)),
            incr: rewrite_current_accesses(incr, enum_text, loop_var),
            body: rewrite_current_accesses(body, enum_text, loop_var),
        },
        Stmt::Return(e) => Stmt::Return(e.map(|e| rewrite_expr_current(e, enum_text, loop_var))),
        Stmt::Throw(e) => Stmt::Throw(rewrite_expr_current(e, enum_text, loop_var)),
        Stmt::YieldReturn(e) => {
            Stmt::YieldReturn(rewrite_expr_current(e, enum_text, loop_var))
        }
        Stmt::ForEach {
            var_name,
            collection,
            body,
        } => Stmt::ForEach {
            var_name,
            collection,
            body: rewrite_current_accesses(body, enum_text, loop_var),
        },
        Stmt::Using { decl, body } => Stmt::Using {
            decl: Box::new(rewrite_stmt_current(*decl, enum_text, loop_var)),
            body: rewrite_current_accesses(body, enum_text, loop_var),
        },
        Stmt::Lock { expr, body } => Stmt::Lock {
            expr: rewrite_expr_current(expr, enum_text, loop_var),
            body: rewrite_current_accesses(body, enum_text, loop_var),
        },
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => Stmt::Try {
            try_body: rewrite_current_accesses(try_body, enum_text, loop_var),
            catches: catches
                .into_iter()
                .map(|(name, body)| {
                    (
                        name,
                        rewrite_current_accesses(body, enum_text, loop_var),
                    )
                })
                .collect(),
            finally_body: finally_body
                .map(|fb| rewrite_current_accesses(fb, enum_text, loop_var)),
        },
        Stmt::Switch {
            disc,
            cases,
            tail,
            default,
        } => Stmt::Switch {
            disc,
            cases: cases
                .into_iter()
                .map(|(val, body)| {
                    (val, rewrite_current_accesses(body, enum_text, loop_var))
                })
                .collect(),
            tail: rewrite_current_accesses(tail, enum_text, loop_var),
            default: default.map(|d| rewrite_current_accesses(d, enum_text, loop_var)),
        },
        // Statements with no expressions to rewrite.
        other => other,
    }
}

/// Rewrite one expression, replacing `enumerator.Current` / `enumerator.get_Current()`
/// with the loop variable. Recurses into all sub-expressions.
fn rewrite_expr_current(e: Expr, enum_text: &str, loop_var: &str) -> Expr {
    // Check if THIS expression is a Current access on the enumerator.
    match &e {
        // Field access: enumerator.Current
        Expr::Field(recv, field) if field == "Current" && render_text(recv) == enum_text => {
            return Expr::Var(loop_var.to_string());
        }
        // Property getter call: enumerator.get_Current()
        Expr::Call {
            recv: Some(recv),
            name,
            args,
        } if is_get_current_name(name)
            && args.is_empty()
            && render_text(recv) == enum_text =>
        {
            return Expr::Var(loop_var.to_string());
        }
        // The inlined field read: `*(nint*)(&enumerator + 16)`. `Current`
        // is the third word of the enumerator buffer; before the buffer is
        // coalesced this was `&stack_66 +16` and no rewrite could see it.
        // The addend is the AArch64/IL2CPP buffer layout (base+16); a
        // different offset is not Current and is left alone.
        Expr::RawDeref(inner, 16, _) if render_text(inner) == enum_text => {
            return Expr::Var(loop_var.to_string());
        }
        _ => {}
    }
    // Otherwise recurse into sub-expressions.
    match e {
        Expr::Bin(op, a, b) => Expr::Bin(
            op,
            Box::new(rewrite_expr_current(*a, enum_text, loop_var)),
            Box::new(rewrite_expr_current(*b, enum_text, loop_var)),
        ),
        Expr::Cmp(kind, a, b) => Expr::Cmp(
            kind,
            Box::new(rewrite_expr_current(*a, enum_text, loop_var)),
            Box::new(rewrite_expr_current(*b, enum_text, loop_var)),
        ),
        Expr::Not(x) => Expr::Not(Box::new(rewrite_expr_current(*x, enum_text, loop_var))),
        Expr::Neg(x) => Expr::Neg(Box::new(rewrite_expr_current(*x, enum_text, loop_var))),
        Expr::Field(recv, f) => Expr::Field(
            Box::new(rewrite_expr_current(*recv, enum_text, loop_var)),
            f,
        ),
        Expr::StaticField(ty, f) => Expr::StaticField(ty, f),
        Expr::Call { recv, name, args } => Expr::Call {
            recv: recv.map(|r| Box::new(rewrite_expr_current(*r, enum_text, loop_var))),
            name,
            args: args
                .into_iter()
                .map(|a| rewrite_expr_current(a, enum_text, loop_var))
                .collect(),
        },
        Expr::Cond(c, a, b) => Expr::Cond(
            Box::new(rewrite_expr_current(*c, enum_text, loop_var)),
            Box::new(rewrite_expr_current(*a, enum_text, loop_var)),
            Box::new(rewrite_expr_current(*b, enum_text, loop_var)),
        ),
        Expr::Cast(t, x) => {
            Expr::Cast(t, Box::new(rewrite_expr_current(*x, enum_text, loop_var)))
        }
        Expr::AddressOf(x) => {
            Expr::AddressOf(Box::new(rewrite_expr_current(*x, enum_text, loop_var)))
        }
        Expr::Await(x) => {
            Expr::Await(Box::new(rewrite_expr_current(*x, enum_text, loop_var)))
        }
        Expr::RawDeref(r, off, sz) => Expr::RawDeref(
            Box::new(rewrite_expr_current(*r, enum_text, loop_var)),
            off,
            sz,
        ),
        Expr::RawAddr(r, off) => Expr::RawAddr(
            Box::new(rewrite_expr_current(*r, enum_text, loop_var)),
            off,
        ),
        Expr::Lambda { params, body } => Expr::Lambda {
            params,
            body: Box::new(rewrite_expr_current(*body, enum_text, loop_var)),
        },
        Expr::ArrayInit { elem_type, items } => Expr::ArrayInit {
            elem_type,
            items: items
                .into_iter()
                .map(|i| rewrite_expr_current(i, enum_text, loop_var))
                .collect(),
        },
        Expr::CollectionInit {
            type_name,
            ctor_args,
            items,
        } => Expr::CollectionInit {
            type_name,
            ctor_args: ctor_args
                .into_iter()
                .map(|a| rewrite_expr_current(a, enum_text, loop_var))
                .collect(),
            items: items
                .into_iter()
                .map(|i| rewrite_expr_current(i, enum_text, loop_var))
                .collect(),
        },
        Expr::ObjInit {
            type_name,
            ctor_args,
            fields,
        } => Expr::ObjInit {
            type_name,
            ctor_args: ctor_args
                .into_iter()
                .map(|a| rewrite_expr_current(a, enum_text, loop_var))
                .collect(),
            fields: fields
                .into_iter()
                .map(|(f, v)| (f, rewrite_expr_current(v, enum_text, loop_var)))
                .collect(),
        },
        Expr::Index { collection, index } => Expr::Index {
            collection: Box::new(rewrite_expr_current(*collection, enum_text, loop_var)),
            index: Box::new(rewrite_expr_current(*index, enum_text, loop_var)),
        },
        Expr::NullConditional { receiver, access } => Expr::NullConditional {
            receiver: Box::new(rewrite_expr_current(*receiver, enum_text, loop_var)),
            access: Box::new(rewrite_expr_current(*access, enum_text, loop_var)),
        },
        Expr::NewObj(t, args) => Expr::NewObj(
            t,
            args.into_iter()
                .map(|a| rewrite_expr_current(a, enum_text, loop_var))
                .collect(),
        ),
        Expr::NewArr(t, args) => Expr::NewArr(
            t,
            args.into_iter()
                .map(|a| rewrite_expr_current(a, enum_text, loop_var))
                .collect(),
        ),
        Expr::RawCall(addr, args, ret) => Expr::RawCall(
            addr,
            args.into_iter()
                .map(|a| rewrite_expr_current(a, enum_text, loop_var))
                .collect(),
            ret,
        ),
        // Leaf nodes — nothing to rewrite.
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
        | Expr::MethodGroup(_)
        | Expr::PatternMatch { .. }
        | Expr::StringInterp { .. }
        | Expr::Tuple { .. }
        | Expr::MethodChain { .. } | Expr::SwitchExpr { .. }
        | Expr::Range { .. } | Expr::IndexFromEnd(_) | Expr::TypeOf(_) => e,
    }
}

/// Is this method name a get_Current property accessor?
fn is_get_current_name(name: &str) -> bool {
    name == "get_Current" || name.ends_with(".get_Current")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::{BinOp, CmpKind};

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn addr_of(n: &str) -> Expr {
        Expr::AddressOf(Box::new(var(n)))
    }

    fn call(recv: Option<Expr>, name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            recv: recv.map(Box::new),
            name: name.to_string(),
            args,
        }
    }

    fn field(recv: Expr, name: &str) -> Expr {
        Expr::Field(Box::new(recv), name.to_string())
    }

    fn eval(e: Expr) -> Stmt {
        Stmt::Eval(e)
    }

    fn assign(name: &str, e: Expr) -> Stmt {
        Stmt::Assign(name.to_string(), None, e)
    }

    /// Build the standard 3-statement foreach pattern (void-call form).
    fn build_foreach_pattern(
        collection: Expr,
        enum_slot: &str,
        body: Vec<Stmt>,
    ) -> Vec<Stmt> {
        let enum_addr = addr_of(enum_slot);
        vec![
            // collection.GetEnumerator(&enumerator)
            eval(call(
                Some(collection),
                "GetEnumerator",
                vec![enum_addr.clone()],
            )),
            // while ((&enumerator).MoveNext() & 1) != 0
            Stmt::While {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(Expr::Bin(
                        BinOp::And,
                        Box::new(call(Some(enum_addr.clone()), "MoveNext", vec![])),
                        Box::new(Expr::Int(1)),
                    )),
                    Box::new(Expr::Int(0)),
                ),
                body,
            },
            // (&enumerator).Dispose()
            eval(call(Some(enum_addr), "Dispose", vec![])),
        ]
    }

    // -----------------------------------------------------------------------
    // 1. Basic foreach with simple collection
    // -----------------------------------------------------------------------

    #[test]
    fn basic_foreach_with_simple_collection() {
        let enum_addr = addr_of("stack_97");
        let mut stmts = build_foreach_pattern(
            var("myList"),
            "stack_97",
            vec![assign(
                "x",
                field(enum_addr.clone(), "Current"),
            )],
        );
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "should fold one foreach");
        assert_eq!(stmts.len(), 1, "three statements become one ForEach");
        match &stmts[0] {
            Stmt::ForEach {
                var_name,
                collection,
                body,
            } => {
                assert_eq!(collection, "myList");
                assert_eq!(body.len(), 1);
                // The body should reference the loop variable, not the enumerator.
                match &body[0] {
                    Stmt::Assign(name, _, Expr::Var(v)) => {
                        assert_eq!(name, "x");
                        assert_eq!(v, var_name);
                    }
                    other => panic!("expected Assign to loop var, got {other:?}"),
                }
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 2. Foreach with field access as collection
    // -----------------------------------------------------------------------

    #[test]
    fn foreach_with_field_access_collection() {
        let collection = field(Expr::This, "items");
        let mut stmts = build_foreach_pattern(
            collection,
            "enum1",
            vec![assign("v", field(addr_of("enum1"), "Current"))],
        );
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::ForEach { collection, .. } => {
                assert_eq!(collection, "this.items");
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 3. Foreach with method call as collection
    // -----------------------------------------------------------------------

    #[test]
    fn foreach_with_method_call_collection() {
        let collection = call(Some(Expr::This), "GetItems", vec![]);
        let mut stmts = build_foreach_pattern(
            collection,
            "e",
            vec![assign("v", field(addr_of("e"), "Current"))],
        );
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::ForEach { collection, .. } => {
                assert_eq!(collection, "this.GetItems()");
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 4. Nested foreach (outer and inner both recovered)
    // -----------------------------------------------------------------------

    #[test]
    fn nested_foreach_both_recovered() {
        // Inner foreach pattern as the body of the outer foreach.
        let inner_enum = addr_of("inner_e");
        let inner_pattern = build_foreach_pattern(
            field(addr_of("outer_e"), "Current"),
            "inner_e",
            vec![assign(
                "y",
                field(inner_enum.clone(), "Current"),
            )],
        );

        // The outer pattern wraps the inner pattern's 3 statements as its body.
        // But the inner pattern is itself 3 statements, so the outer while body
        // contains those 3 statements.
        let outer_enum = addr_of("outer_e");
        let mut stmts = vec![
            eval(call(Some(var("matrix")), "GetEnumerator", vec![outer_enum.clone()])),
            Stmt::While {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(Expr::Bin(
                        BinOp::And,
                        Box::new(call(Some(outer_enum.clone()), "MoveNext", vec![])),
                        Box::new(Expr::Int(1)),
                    )),
                    Box::new(Expr::Int(0)),
                ),
                body: inner_pattern,
            },
            eval(call(Some(outer_enum), "Dispose", vec![])),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both outer and inner foreach should fold");
        // The outer should be a ForEach.
        assert!(matches!(&stmts[0], Stmt::ForEach { .. }));
    }

    // -----------------------------------------------------------------------
    // 5. Multiple foreach in sequence
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_foreach_in_sequence() {
        let mut stmts = Vec::new();
        // First foreach.
        stmts.extend(build_foreach_pattern(
            var("list1"),
            "e1",
            vec![assign("a", field(addr_of("e1"), "Current"))],
        ));
        // Second foreach.
        stmts.extend(build_foreach_pattern(
            var("list2"),
            "e2",
            vec![assign("b", field(addr_of("e2"), "Current"))],
        ));
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both sequential foreach patterns should fold");
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::ForEach { collection, .. } if collection == "list1"));
        assert!(matches!(&stmts[1], Stmt::ForEach { collection, .. } if collection == "list2"));
    }

    // -----------------------------------------------------------------------
    // 6. Foreach with complex body (multiple statements)
    // -----------------------------------------------------------------------

    #[test]
    fn foreach_with_complex_body() {
        let enum_addr = addr_of("e");
        let mut stmts = build_foreach_pattern(
            var("items"),
            "e",
            vec![
                assign("x", field(enum_addr.clone(), "Current")),
                eval(call(None, "Process", vec![var("x")])),
                assign("sum", Expr::Bin(BinOp::Add, Box::new(var("sum")), Box::new(var("x")))),
            ],
        );
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::ForEach { body, .. } => {
                assert_eq!(body.len(), 3, "all body statements preserved");
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. Foreach where Current is used in expression (not just assignment)
    // -----------------------------------------------------------------------

    #[test]
    fn foreach_current_used_in_expression() {
        let enum_addr = addr_of("e");
        let mut stmts = build_foreach_pattern(
            var("items"),
            "e",
            vec![
                // sum = sum + enumerator.Current
                assign(
                    "sum",
                    Expr::Bin(
                        BinOp::Add,
                        Box::new(var("sum")),
                        Box::new(field(enum_addr.clone(), "Current")),
                    ),
                ),
                // Process(enumerator.Current)
                eval(call(None, "Process", vec![field(enum_addr, "Current")])),
            ],
        );
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::ForEach { body, var_name, .. } => {
                // First statement: sum = sum + item
                match &body[0] {
                    Stmt::Assign(name, _, Expr::Bin(BinOp::Add, lhs, rhs)) => {
                        assert_eq!(name, "sum");
                        assert!(matches!(lhs.as_ref(), Expr::Var(n) if n == "sum"));
                        assert!(matches!(rhs.as_ref(), Expr::Var(n) if n == var_name));
                    }
                    other => panic!("expected Assign with Bin, got {other:?}"),
                }
                // Second statement: Process(item)
                match &body[1] {
                    Stmt::Eval(Expr::Call { args, .. }) => {
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], Expr::Var(n) if n == var_name));
                    }
                    other => panic!("expected Eval(Call), got {other:?}"),
                }
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 8. Non-match: GetEnumerator without Dispose
    // -----------------------------------------------------------------------

    #[test]
    fn no_match_getenumerator_without_dispose() {
        let enum_addr = addr_of("e");
        let mut stmts = vec![
            eval(call(Some(var("list")), "GetEnumerator", vec![enum_addr.clone()])),
            Stmt::While {
                cond: call(Some(enum_addr.clone()), "MoveNext", vec![]),
                body: vec![assign("x", field(enum_addr.clone(), "Current"))],
            },
            // No Dispose call — just some other statement.
            eval(call(None, "DoSomething", vec![])),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "no Dispose means no fold");
        assert_eq!(stmts.len(), 3, "statements unchanged");
    }

    // -----------------------------------------------------------------------
    // 9. Non-match: MoveNext on different variable
    // -----------------------------------------------------------------------

    #[test]
    fn no_match_movenext_on_different_variable() {
        let enum_addr = addr_of("e1");
        let other_addr = addr_of("e2"); // different enumerator!
        let mut stmts = vec![
            eval(call(Some(var("list")), "GetEnumerator", vec![enum_addr])),
            Stmt::While {
                cond: call(Some(other_addr.clone()), "MoveNext", vec![]), // wrong enumerator
                body: vec![assign("x", field(other_addr, "Current"))],
            },
            eval(call(Some(addr_of("e1")), "Dispose", vec![])),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "MoveNext on different enumerator means no fold");
        assert_eq!(stmts.len(), 3, "statements unchanged");
    }

    // -----------------------------------------------------------------------
    // 10. Non-match: GetEnumerator with no following while
    // -----------------------------------------------------------------------

    #[test]
    fn no_match_getenumerator_no_following_while() {
        let enum_addr = addr_of("e");
        let mut stmts = vec![
            eval(call(Some(var("list")), "GetEnumerator", vec![enum_addr.clone()])),
            // Not a while — just an eval.
            eval(call(Some(enum_addr.clone()), "MoveNext", vec![])),
            eval(call(Some(enum_addr), "Dispose", vec![])),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "no While means no fold");
        assert_eq!(stmts.len(), 3, "statements unchanged");
    }

    // -----------------------------------------------------------------------
    // 11. Assignment form: enumerator = collection.GetEnumerator()
    // -----------------------------------------------------------------------

    #[test]
    fn assignment_form_foreach() {
        let mut stmts = vec![
            // enumerator = collection.GetEnumerator()
            assign(
                "enumerator",
                call(Some(var("collection")), "GetEnumerator", vec![]),
            ),
            // while (enumerator.MoveNext())
            Stmt::While {
                cond: call(Some(var("enumerator")), "MoveNext", vec![]),
                body: vec![assign(
                    "x",
                    field(var("enumerator"), "Current"),
                )],
            },
            // enumerator.Dispose()
            eval(call(Some(var("enumerator")), "Dispose", vec![])),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "assignment form should fold");
        match &stmts[0] {
            Stmt::ForEach {
                var_name,
                collection,
                body,
            } => {
                assert_eq!(collection, "collection");
                assert_eq!(body.len(), 1);
                match &body[0] {
                    Stmt::Assign(name, _, Expr::Var(v)) => {
                        assert_eq!(name, "x");
                        assert_eq!(v, var_name);
                    }
                    other => panic!("expected Assign to loop var, got {other:?}"),
                }
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 12. get_Current() property accessor form
    // -----------------------------------------------------------------------

    #[test]
    fn foreach_with_get_current_accessor() {
        let enum_addr = addr_of("e");
        let mut stmts = vec![
            eval(call(
                Some(var("list")),
                "GetEnumerator",
                vec![enum_addr.clone()],
            )),
            Stmt::While {
                cond: call(Some(enum_addr.clone()), "MoveNext", vec![]),
                body: vec![assign(
                    "x",
                    // enumerator.get_Current() instead of enumerator.Current
                    call(Some(enum_addr.clone()), "get_Current", vec![]),
                )],
            },
            eval(call(Some(enum_addr), "Dispose", vec![])),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "get_Current() accessor form should fold");
        match &stmts[0] {
            Stmt::ForEach { body, var_name, .. } => {
                match &body[0] {
                    Stmt::Assign(name, _, Expr::Var(v)) => {
                        assert_eq!(name, "x");
                        assert_eq!(v, var_name);
                    }
                    other => panic!("expected Assign to loop var, got {other:?}"),
                }
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Index-loop (`for`/`while` over arr[i]) → foreach
    // -----------------------------------------------------------------------

    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Bin(BinOp::Add, Box::new(a), Box::new(b))
    }
    fn lt(a: Expr, b: Expr) -> Expr {
        Expr::Cmp(CmpKind::Lt, Box::new(a), Box::new(b))
    }
    fn incr(i: &str) -> Stmt {
        assign(i, add(var(i), Expr::Int(1)))
    }
    fn length(coll: Expr) -> Expr {
        field(coll, "Length")
    }
    fn index(coll: Expr, i: &str) -> Expr {
        Expr::Index {
            collection: Box::new(coll),
            index: Box::new(var(i)),
        }
    }
    fn load(item: &str, coll: Expr, i: &str) -> Stmt {
        assign(item, index(coll, i))
    }
    fn while_index_loop(_coll: Expr, bound: Expr, body: Vec<Stmt>) -> Vec<Stmt> {
        vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), bound),
                body,
            },
        ]
    }

    #[test]
    fn index_loop_becomes_foreach() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                eval(call(None, "Use", vec![var("item")])),
                incr("i"),
            ],
        );
        assert_eq!(apply(&mut stmts), 1);
        assert_eq!(stmts.len(), 1, "the init is subsumed, the loop becomes one statement");
        match &stmts[0] {
            Stmt::ForEach {
                var_name,
                collection,
                body,
            } => {
                assert_eq!(var_name, "item");
                assert_eq!(collection, "arr");
                assert_eq!(body.len(), 1, "load and update are gone");
                assert!(!list_mentions_any(body, "i"), "no index left in the body");
                assert!(matches!(&body[0], Stmt::Eval(_)));
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    #[test]
    fn index_loop_with_raw_length_slot_becomes_foreach() {
        // `while (i < *(nint*)(arr +24))` — the IL2CPP array length slot when
        // `array_length` did not rewrite it (and the 4-byte List `_size`).
        for size in [4u8, 8u8] {
            let mut stmts = while_index_loop(
                var("arr"),
                Expr::RawDeref(Box::new(var("arr")), 24, size),
                vec![
                    load("item", var("arr"), "i"),
                    eval(call(None, "Use", vec![var("item")])),
                    incr("i"),
                ],
            );
            assert_eq!(apply(&mut stmts), 1, "size {size} must fold");
            assert!(matches!(&stmts[0], Stmt::ForEach { collection, .. } if collection == "arr"));
        }
    }

    #[test]
    fn for_spelling_becomes_foreach() {
        // `for_loop` runs before this pass and may already have folded the
        // while into a `for`; the index-loop recognizer must accept it.
        let mut stmts = vec![Stmt::For {
            init: vec![assign("i", Expr::Int(0))],
            cond: Some(lt(var("i"), length(var("arr")))),
            incr: vec![incr("i")],
            body: vec![
                load("item", var("arr"), "i"),
                eval(call(None, "Use", vec![var("item")])),
            ],
        }];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::ForEach { var_name, .. } if var_name == "item"));
    }

    #[test]
    fn trailing_continue_is_dropped() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                eval(call(None, "Use", vec![var("item")])),
                incr("i"),
                Stmt::Continue,
            ],
        );
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::ForEach { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(!body.iter().any(|s| matches!(s, Stmt::Continue)));
            }
            other => panic!("expected ForEach, got {other:?}"),
        }
    }

    #[test]
    fn index_used_for_another_load_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                eval(call(None, "Also", vec![index(var("arr"), "i")])),
                incr("i"),
            ],
        );
        assert_eq!(apply(&mut stmts), 0, "the index is used for more than the one load");
        assert!(matches!(&stmts[1], Stmt::While { .. }));
    }

    #[test]
    fn body_writing_the_collection_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                assign("arr", var("other")),
                incr("i"),
            ],
        );
        assert_eq!(apply(&mut stmts), 0, "a reassigned array is not loop-invariant");
    }

    #[test]
    fn item_live_after_the_loop_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                eval(call(None, "Use", vec![var("item")])),
                incr("i"),
            ],
        );
        stmts.push(eval(call(None, "Use", vec![var("item")])));
        assert_eq!(apply(&mut stmts), 0, "item's value after the loop would change");
    }

    #[test]
    fn index_live_after_the_loop_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![load("item", var("arr"), "i"), incr("i")],
        );
        stmts.push(eval(call(None, "Use", vec![var("i")])));
        assert_eq!(apply(&mut stmts), 0, "deleting `i = 0` would change the later read");
    }

    #[test]
    fn load_not_first_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                eval(call(None, "Before", vec![])),
                load("item", var("arr"), "i"),
                incr("i"),
            ],
        );
        assert_eq!(
            apply(&mut stmts),
            0,
            "the element must be the first thing the iteration reads"
        );
    }

    #[test]
    fn update_not_last_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                incr("i"),
                eval(call(None, "After", vec![])),
            ],
        );
        assert_eq!(apply(&mut stmts), 0, "the update must close the body");
    }

    #[test]
    fn continue_before_the_update_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                Stmt::If {
                    cond: var("c"),
                    then_body: vec![Stmt::Continue],
                    else_body: vec![],
                },
                incr("i"),
            ],
        );
        assert_eq!(
            apply(&mut stmts),
            0,
            "a continue before the update skips it in the flat form"
        );
    }

    #[test]
    fn writing_the_element_local_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                assign("item", var("other")),
                incr("i"),
            ],
        );
        assert_eq!(
            apply(&mut stmts),
            0,
            "a foreach iteration variable is not assignable"
        );
    }

    #[test]
    fn mismatched_collection_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("other"), "i"),
                eval(call(None, "Use", vec![var("item")])),
                incr("i"),
            ],
        );
        assert_eq!(apply(&mut stmts), 0, "the bound and the load disagree");
    }

    #[test]
    fn call_valued_collection_is_refused() {
        let mut stmts = while_index_loop(
            var("arr"),
            Expr::Field(
                Box::new(call(Some(Expr::This), "GetItems", vec![])),
                "Length".into(),
            ),
            vec![load("item", var("arr"), "i"), incr("i")],
        );
        assert_eq!(
            apply(&mut stmts),
            0,
            "a call bound would be evaluated once by foreach, many times by the while"
        );
    }

    #[test]
    fn non_zero_init_or_non_unit_update_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(1)),
            Stmt::While {
                cond: lt(var("i"), length(var("arr"))),
                body: vec![load("item", var("arr"), "i"), incr("i")],
            },
        ];
        assert_eq!(apply(&mut stmts), 0, "the index must start at zero");

        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                assign("i", add(var("i"), Expr::Int(2))),
            ],
        );
        assert_eq!(apply(&mut stmts), 0, "only a unit step is a foreach");
    }

    #[test]
    fn index_foreach_is_idempotent() {
        let mut stmts = while_index_loop(
            var("arr"),
            length(var("arr")),
            vec![
                load("item", var("arr"), "i"),
                eval(call(None, "Use", vec![var("item")])),
                incr("i"),
            ],
        );
        assert_eq!(apply(&mut stmts), 1);
        let after = stmts.clone();
        assert_eq!(apply(&mut stmts), 0, "a second run must not rewrite again");
        assert_eq!(stmts.len(), after.len());
    }

    /// The compiler's own `foreach` spelling: the Dispose lives in the
    /// `try`'s `finally`, so the enumerator triple is split 2 + loop.
    #[test]
    fn compiler_finally_foreach_is_folded() {
        let enum_addr = addr_of("stack_97");
        let while_stmt = Stmt::While {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Bin(
                    BinOp::And,
                    Box::new(call(Some(enum_addr.clone()), "MoveNext", vec![])),
                    Box::new(Expr::Int(1)),
                )),
                Box::new(Expr::Int(0)),
            ),
            body: vec![assign("v", field(enum_addr.clone(), "Current"))],
        };
        let mut stmts = vec![
            eval(call(
                Some(var("myList")),
                "GetEnumerator",
                vec![enum_addr.clone()],
            )),
            Stmt::Try {
                try_body: vec![while_stmt],
                catches: vec![],
                finally_body: Some(vec![eval(call(Some(enum_addr), "Dispose", vec![]))]),
            },
        ];
        assert_eq!(apply(&mut stmts), 1, "{stmts:?}");
        assert!(
            matches!(&stmts[0], Stmt::ForEach { collection, .. } if collection == "myList"),
            "{stmts:?}"
        );
    }

    /// The structurer can put the post-loop code inside the try; it is moved
    /// after the `foreach`, and the duplicated `<enum>.Dispose()` is dropped.
    #[test]
    fn compiler_finally_foreach_moves_post_loop_tail() {
        let enum_addr = addr_of("stack_97");
        let dispose = eval(call(Some(enum_addr.clone()), "Dispose", vec![]));
        let mut stmts = vec![
            eval(call(
                Some(var("myList")),
                "GetEnumerator",
                vec![enum_addr.clone()],
            )),
            Stmt::Try {
                try_body: vec![
                    Stmt::While {
                        cond: call(Some(enum_addr.clone()), "MoveNext", vec![]),
                        body: vec![assign("v", field(enum_addr, "Current"))],
                    },
                    dispose,
                    Stmt::Return(Some(Expr::Int(0))),
                ],
                catches: vec![],
                finally_body: Some(vec![eval(call(
                    Some(addr_of("stack_97")),
                    "Dispose",
                    vec![],
                ))]),
            },
        ];
        assert_eq!(apply(&mut stmts), 1, "{stmts:?}");
        assert_eq!(stmts.len(), 2, "foreach + moved tail: {stmts:?}");
        assert!(matches!(&stmts[0], Stmt::ForEach { .. }), "{stmts:?}");
        assert!(matches!(&stmts[1], Stmt::Return(Some(Expr::Int(0)))), "{stmts:?}");
    }

    /// `Current` read through the inlined enumerator buffer
    /// (`*(nint*)(&e +16)`) must become the loop variable like any other
    /// Current access.
    #[test]
    fn raw_current_slot_becomes_the_loop_variable() {
        let enum_addr = addr_of("e");
        let current = Expr::RawDeref(Box::new(enum_addr.clone()), 16, 0);
        let mut stmts = vec![
            eval(call(
                Some(var("list")),
                "GetEnumerator",
                vec![enum_addr.clone()],
            )),
            Stmt::Try {
                try_body: vec![Stmt::While {
                    cond: call(Some(enum_addr.clone()), "MoveNext", vec![]),
                    body: vec![assign("x", current)],
                }],
                catches: vec![],
                finally_body: Some(vec![eval(call(Some(enum_addr), "Dispose", vec![]))]),
            },
        ];
        assert_eq!(apply(&mut stmts), 1, "{stmts:?}");
        let text = format!("{stmts:?}");
        assert!(
            !text.contains("RawDeref"),
            "Current slot must be replaced: {text}"
        );
        assert!(text.contains("item"), "{text}");
    }

    /// The GetEnumerator, MoveNext and Dispose paths may carry different
    /// display names for the same address-taken cell. The wrapped matcher
    /// must key on the loop's receiver, not the GetEnumerator spelling.
    #[test]
    fn wrapped_foreach_identity_is_the_loop_receiver() {
        let mut stmts = vec![
            eval(call(Some(var("list")), "GetEnumerator", vec![addr_of("e1")])),
            Stmt::Try {
                try_body: vec![Stmt::While {
                    cond: call(Some(addr_of("e2")), "MoveNext", vec![]),
                    body: vec![assign(
                        "x",
                        Expr::RawDeref(Box::new(addr_of("e2")), 16, 0),
                    )],
                }],
                catches: vec![],
                finally_body: Some(vec![eval(call(Some(addr_of("e3")), "Dispose", vec![]))]),
            },
        ];
        assert_eq!(apply(&mut stmts), 1, "{stmts:?}");
        let text = format!("{stmts:?}");
        assert!(!text.contains("RawDeref"), "{text}");
        assert!(text.contains("item"), "{text}");
    }

    /// The sret/Current setup spills sit between GetEnumerator and the try
    /// after the enumerator buffer is coalesced; the matcher must look past
    /// them and drop them with the wrapper.
    #[test]
    fn wrapped_foreach_skips_enum_buffer_setup() {
        let enum_addr = addr_of("e");
        let mut stmts = vec![
            eval(call(
                Some(var("list")),
                "GetEnumerator",
                vec![enum_addr.clone()],
            )),
            Stmt::Assign("intPtrValue".into(), None, var("e")),
            Stmt::Assign(
                "*(nint*)((nint)&e +16)".into(),
                None,
                var("stack_99"),
            ),
            Stmt::Try {
                try_body: vec![Stmt::While {
                    cond: call(Some(enum_addr.clone()), "MoveNext", vec![]),
                    body: vec![assign("x", Expr::Int(1))],
                }],
                catches: vec![],
                finally_body: Some(vec![eval(call(Some(enum_addr), "Dispose", vec![]))]),
            },
        ];
        assert_eq!(apply(&mut stmts), 1, "{stmts:?}");
        assert_eq!(stmts.len(), 1, "setup spills must be dropped: {stmts:?}");
        assert!(matches!(&stmts[0], Stmt::ForEach { .. }), "{stmts:?}");
    }

    /// An arbitrary assignment inside the try after the loop may be real
    /// source code; moving it would change control flow, so the fold refuses.
    #[test]
    fn wrapped_foreach_refuses_an_unproven_tail() {
        let enum_addr = addr_of("e");
        let mut stmts = vec![
            eval(call(
                Some(var("list")),
                "GetEnumerator",
                vec![enum_addr.clone()],
            )),
            Stmt::Try {
                try_body: vec![
                    Stmt::While {
                        cond: call(Some(enum_addr.clone()), "MoveNext", vec![]),
                        body: vec![],
                    },
                    Stmt::Assign("v27".into(), None, Expr::Int(7)),
                ],
                catches: vec![],
                finally_body: Some(vec![eval(call(Some(enum_addr), "Dispose", vec![]))]),
            },
        ];
        assert_eq!(apply(&mut stmts), 0, "{stmts:?}");
        assert!(matches!(&stmts[0], Stmt::Eval(_)), "{stmts:?}");
    }

    /// A real typed catch around the loop is not a `foreach`; the fold must
    /// refuse rather than delete the handler.
    #[test]
    fn typed_catch_around_the_loop_is_not_folded() {
        let enum_addr = addr_of("stack_97");
        let mut stmts = vec![
            eval(call(
                Some(var("myList")),
                "GetEnumerator",
                vec![enum_addr.clone()],
            )),
            Stmt::Try {
                try_body: vec![Stmt::While {
                    cond: call(Some(enum_addr.clone()), "MoveNext", vec![]),
                    body: vec![],
                }],
                catches: vec![("Exception".into(), vec![Stmt::Throw(var("e"))])],
                finally_body: Some(vec![eval(call(Some(enum_addr), "Dispose", vec![]))]),
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }
}
