//! `for (init; cond; incr)` reconstruction.
//!
//! ## The shape that ACTUALLY occurs
//!
//! Measured on the x64 corpus (47,924 methods): `for (` appears **zero** times,
//! while `while (` appears 6,758 times. The structurer emits the canonical
//! counter loop as a flat `init; …; while (cond) { …; update; …; continue; }`.
//! Verbatim from the real dump (`dump.cs`, `get_Item` @~line 3010):
//!
//! ```text
//! v11 = 0;
//! v12 = this.attrNames;
//! v13 = 0;                       // the counter's init — NOT adjacent to the loop
//! v14 = name;
//! while (v13 < *(nint*)(v12 +24))
//! {
//!     …
//!     v11 = v11 + 1;
//!     v12 = this.attrNames;
//!     v13 = v11 + 1;             // updated through ANOTHER temp, not `v13 = v13 + 1`
//!     v14 = v18;                 // a live copy AFTER the update
//!     continue;                  // an explicit trailing `continue`
//! }
//! ```
//!
//! Three properties of that shape kill the naive rule ("the statement before
//! the `while` is the init, the body's LAST statement is the increment, and it
//! must mention the counter"): the init is not adjacent, the increment does not
//! mention the counter, and the body ends with `continue`. The previous version
//! of this file required all three and therefore never fired.
//!
//! ## The rewrite, and the proof for each gate
//!
//! `for (init; cond; UPDATE) { pre…, TAIL… }` where the flat form was
//! `init; …; while (cond) { pre…, UPDATE, TAIL… (continue) }`:
//!
//!   * **the counter** is an induction variable of the condition — the direct
//!     local operand of the comparison, or the left operand of arithmetic on it
//!     (`v33 + 1 < n`). A condition whose operands are raw derefs
//!     (`*(nint*)(v37 -1) < *(nint*)(buffer +8)`) names no local and is
//!     refused;
//!   * **the update** is the LAST assignment to the counter in the body. It
//!     need not mention the counter (`v13 = v11 + 1` is the real shape);
//!   * **the tail** (statements after the update, up to the end of the body) is
//!     left in the body, which moves it BEFORE the update. Rule 2 allows that
//!     only when nothing in it can observe the difference: every tail statement
//!     must be free of calls and of control flow, must not read the counter,
//!     and must not write anything the update's right-hand side reads;
//!   * **at most one `continue`**, and only as the very last statement: it is
//!     dropped (falling off the end of a `for` body also runs the increment, so
//!     a trailing `continue` is a no-op there). Any OTHER `continue` at this
//!     level skips the update in the flat form but would run it in a `for`, so
//!     it refuses the fold;
//!   * **the init** is the nearest preceding assignment to the counter, hoisted
//!     from its position `p` into the header. That is a move LATER (toward the
//!     loop), so: nothing between `p` and the loop may **write** the counter
//!     (a write between would be overwritten by the hoisted init, changing the
//!     value the loop sees). **Reads** of the counter between `p` and the loop
//!     are safe: the init is still the most recent write at every such read
//!     point, so the read sees the same value before and after the move. The
//!     init's right-hand side must be call-free and must not read anything
//!     those between statements write, and the init must have an inline form;
//!   * **the counter must not be read after the loop.** A `for` header's
//!     declaration is scoped to the loop, so hoisting the init while later
//!     statements still read the counter would change the program's scope;
//!   * **the header must be renderable.** `Renderer::stmt` builds a `for` header
//!     with `filter_map(stmt_inline)`, so an init/increment element without an
//!     inline form is silently DROPPED from the output — a data loss, not a
//!     style problem. Both are `Assign`s here, but the gate is checked anyway.
//!
//! ## The gate that was too strict (Phase 3A fix)
//!
//! The original between-init-and-loop check used `list_mentions_deep`, which
//! refuses when the counter is mentioned (read OR write) anywhere between the
//! init and the while. The write part is necessary (a write between would be
//! overwritten by the hoisted init), but the read part is not: the init is the
//! nearest preceding write at every point between itself and the loop (that is
//! exactly how `init_at` was chosen), so a read there already sees the init's
//! value. Moving the init later does not change what any such read observes.
//!
//! The fix replaces `list_mentions_deep(between, &counter)` with
//! `list_writes_deep(between, &counter)`. This is the gate that refused every
//! real corpus candidate where another statement between the init and the while
//! happened to mention the counter (e.g. a read in a condition or a copy),
//! which is why the pass fired 0 times despite the rest of the logic being
//! correct.
//!
//! A failed match is a no-op: the correct flat `while` stays.
//!
//! ## Instrumentation
//!
//! Following the `using_lock` model: atomic counters track the population of
//! while loops examined, the intermediate shape properties found, and the
//! reason each refused site was left alone. `report_refusals()` prints both
//! the population and the refusal breakdown through the run report, so a bare
//! zero can never again mean "the pass is dead" without saying why.
//!
//! | Counter | Meaning |
//! |---|---|
//! | `LOOPS_EXAMINED` | Every `while` statement examined (once per list) |
//! | `LOOPS_WITH_COUNTER` | Had an induction variable extractable from the condition |
//! | `LOOPS_WITH_UPDATE` | Also had an assignment to the counter in the body |
//! | `LOOPS_WITH_INIT` | Also had a preceding assignment to the counter |
//! | `REFUSED_NO_COUNTER` | Refused: condition names no local |
//! | `REFUSED_NO_UPDATE` | Refused: body never writes the counter |
//! | `REFUSED_CONTINUE` | Refused: a non-trailing `continue` in the body |
//! | `REFUSED_TAIL` | Refused: tail is impure or reads the counter |
//! | `REFUSED_COUNTER_WRITE` | Refused: a between-statement WRITES the counter |
//! | `REFUSED_INIT_CALL` | Refused: init RHS has a call |
//! | `REFUSED_AFTER` | Refused: counter is mentioned after the loop |
//! | `REFUSED_NOT_INLINABLE` | Refused: init or update lacks an inline form |
//! | `FOLDED` | Successfully folded into a `for` |
//!
//! Set `ILR_FORDBG=1` for per-candidate debug logging: each while loop
//! candidate, the shape properties found, and which gate refused it.

use super::{
    expr_has_call, expr_mentions, is_inlinable, list_writes_deep, walk_mut, Expr, Stmt,
};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Instrumentation: the difference between a silent zero and a measured pass
// ---------------------------------------------------------------------------

/// Every `while` statement examined (counted once per statement list, before
/// the fold loop rescans).
static LOOPS_EXAMINED: AtomicU64 = AtomicU64::new(0);
/// Had an induction variable extractable from the condition.
static LOOPS_WITH_COUNTER: AtomicU64 = AtomicU64::new(0);
/// Also had an assignment to the counter in the body.
static LOOPS_WITH_UPDATE: AtomicU64 = AtomicU64::new(0);
/// Also had a preceding assignment to the counter (a candidate init).
static LOOPS_WITH_INIT: AtomicU64 = AtomicU64::new(0);

// Refusal reasons — one counter per gate.
static REFUSED_NO_COUNTER: AtomicU64 = AtomicU64::new(0);
static REFUSED_NO_UPDATE: AtomicU64 = AtomicU64::new(0);
static REFUSED_CONTINUE: AtomicU64 = AtomicU64::new(0);
static REFUSED_TAIL: AtomicU64 = AtomicU64::new(0);
/// A statement between the init and the while WRITES the counter. This is the
/// relaxed version of the old `list_mentions_deep` gate that refused on reads
/// too — the gate that was blocking every real corpus candidate.
static REFUSED_COUNTER_WRITE: AtomicU64 = AtomicU64::new(0);
/// The body writes the counter somewhere other than the canonical update.
static REFUSED_COUNTER_WRITTEN: AtomicU64 = AtomicU64::new(0);
static REFUSED_INIT_CALL: AtomicU64 = AtomicU64::new(0);
static REFUSED_AFTER: AtomicU64 = AtomicU64::new(0);
static REFUSED_NOT_INLINABLE: AtomicU64 = AtomicU64::new(0);

/// Successfully folded into a `for`.
static FOLDED: AtomicU64 = AtomicU64::new(0);

/// Why the for-loop fold refused one site.
#[derive(Clone, Copy, Debug)]
enum Refusal {
    NoCounter,
    NoUpdate,
    Continue,
    Tail,
    /// A between-statement writes the counter (the relaxed gate).
    CounterWriteBetween,
    /// The body writes the counter outside the single canonical update.
    CounterWritten,
    InitCall,
    After,
    NotInlinable,
}

impl Refusal {
    fn count(self) {
        let c = match self {
            Refusal::NoCounter => &REFUSED_NO_COUNTER,
            Refusal::NoUpdate => &REFUSED_NO_UPDATE,
            Refusal::Continue => &REFUSED_CONTINUE,
            Refusal::Tail => &REFUSED_TAIL,
            Refusal::CounterWriteBetween => &REFUSED_COUNTER_WRITE,
            Refusal::CounterWritten => &REFUSED_COUNTER_WRITTEN,
            Refusal::InitCall => &REFUSED_INIT_CALL,
            Refusal::After => &REFUSED_AFTER,
            Refusal::NotInlinable => &REFUSED_NOT_INLINABLE,
        };
        c.fetch_add(1, Ordering::Relaxed);
    }
}

/// One line per non-empty counter, for the run report.
pub fn report_refusals() -> String {
    let sites = LOOPS_EXAMINED.load(Ordering::Relaxed);
    let folded = FOLDED.load(Ordering::Relaxed);
    if sites == 0 {
        return String::new();
    }
    format!(
        "[reconstruct]   for_loop: {sites} while loop(s) examined, {folded} folded; \
         no counter {}, no update {}, bad continue {}, bad tail {}, \
         counter write between {}, counter written in body {}, init has call {}, \
         counter used after {}, not inlinable {}\n",
        REFUSED_NO_COUNTER.load(Ordering::Relaxed),
        REFUSED_NO_UPDATE.load(Ordering::Relaxed),
        REFUSED_CONTINUE.load(Ordering::Relaxed),
        REFUSED_TAIL.load(Ordering::Relaxed),
        REFUSED_COUNTER_WRITE.load(Ordering::Relaxed),
        REFUSED_COUNTER_WRITTEN.load(Ordering::Relaxed),
        REFUSED_INIT_CALL.load(Ordering::Relaxed),
        REFUSED_AFTER.load(Ordering::Relaxed),
        REFUSED_NOT_INLINABLE.load(Ordering::Relaxed),
    )
}

/// Debug logging gated on `ILR_FORDBG`.
macro_rules! ford_dbg {
    ($($arg:tt)*) => {
        if std::env::var("ILR_FORDBG").is_ok() {
            eprintln!("[for-dbg] {}", format!($($arg)*));
        }
    };
}

/// One planned rewrite: where the `while` is, where its init lives, and the
/// pieces of the `for` header.
#[derive(Debug)]
struct Fold {
    /// Index of the `while` statement in the list.
    seq: usize,
    /// Index of the counter's init statement (hoisted into the header).
    init_at: usize,
    /// The `for` statement that replaces the `while`.
    for_stmt: Stmt,
}

/// Run the recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        n += rewrite_list(list);
    });
    n
}

fn rewrite_list(list: &mut Vec<Stmt>) -> usize {
    // Count this list's `while` sites ONCE, before the fold loop rescans.
    for s in list.iter() {
        if matches!(s, Stmt::While { .. }) {
            LOOPS_EXAMINED.fetch_add(1, Ordering::Relaxed);
        }
    }
    let mut n = 0usize;
    loop {
        match find_fold(list) {
            Ok(Some(fold)) => {
                let mut slot = Some(fold.for_stmt);
                let mut out: Vec<Stmt> = Vec::with_capacity(list.len());
                for (idx, s) in std::mem::take(list).into_iter().enumerate() {
                    if idx == fold.init_at {
                        continue; // hoisted into the `for` header
                    }
                    if idx == fold.seq {
                        out.push(slot.take().expect("the `for` is placed exactly once"));
                        continue;
                    }
                    out.push(s);
                }
                *list = out;
                n += 1;
                FOLDED.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            Ok(None) => return n,
            Err(reason) => {
                // Nothing foldable; record WHY the first candidate was refused.
                reason.count();
                return n;
            }
        }
    }
}

/// The first foldable counter loop in this list, if any. Returns `Err(reason)`
/// when at least one while loop was examined but every one was refused, so the
/// refusal breakdown can name the gate that blocked it.
fn find_fold(list: &[Stmt]) -> Result<Option<Fold>, Refusal> {
    let mut first_refusal: Option<Refusal> = None;
    for i in 0..list.len() {
        if !matches!(&list[i], Stmt::While { .. }) {
            continue;
        }
        match plan_at(list, i) {
            Ok(fold) => return Ok(Some(fold)),
            Err(r) => {
                if first_refusal.is_none() {
                    first_refusal = Some(r);
                }
            }
        }
    }
    match first_refusal {
        Some(r) => Err(r),
        None => Ok(None), // no while loops in this list at all
    }
}

/// Location of a counter update in the loop body.
#[derive(Debug, Clone)]
enum UpdateLocation {
    /// Update is at the top level of the loop body.
    TopLevel(usize),
    /// Update is nested inside an if branch.
    Nested {
        /// Index of the if statement in the loop body.
        if_index: usize,
        /// Which branch contains the update.
        branch: Branch,
        /// Index of the update within the branch.
        index_in_branch: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Branch {
    Then,
    Else,
}

/// Recursively search for the last assignment to `counter` in `body`.
/// Returns the location of the update, or None if not found.
fn find_update_deep(body: &[Stmt], counter: &str) -> Option<UpdateLocation> {
    // Search in reverse to find the LAST assignment.
    for (i, s) in body.iter().enumerate().rev() {
        match s {
            Stmt::Assign(n, _, _) if n == counter => {
                return Some(UpdateLocation::TopLevel(i));
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                // Search then_body first (it's later in the statement).
                if let Some(loc) = find_update_in_branch(then_body, counter) {
                    return Some(UpdateLocation::Nested {
                        if_index: i,
                        branch: Branch::Then,
                        index_in_branch: loc,
                    });
                }
                // Then search else_body.
                if let Some(loc) = find_update_in_branch(else_body, counter) {
                    return Some(UpdateLocation::Nested {
                        if_index: i,
                        branch: Branch::Else,
                        index_in_branch: loc,
                    });
                }
            }
            _ => {}
        }
    }
    None
}

/// Search for the last assignment to `counter` in a branch body.
/// Returns the index within the branch, or None if not found.
fn find_update_in_branch(branch: &[Stmt], counter: &str) -> Option<usize> {
    branch
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, s)| match s {
            Stmt::Assign(n, _, _) if n == counter => Some(i),
            _ => None,
        })
}

/// Check if a statement list always ends with a terminator (continue, break, return, goto).
/// This means no path falls through to the code after the list.
fn always_terminates(stmts: &[Stmt]) -> bool {
    match stmts.last() {
        Some(Stmt::Continue) | Some(Stmt::Break) | Some(Stmt::Return(_)) | Some(Stmt::Goto(_)) | Some(Stmt::Throw(_)) => true,
        Some(Stmt::If {
            then_body,
            else_body,
            ..
        }) => {
            // Both branches must terminate.
            always_terminates(then_body) && always_terminates(else_body)
        }
        _ => false,
    }
}

/// Check if a statement list contains a continue at any depth (but not inside nested loops).
fn contains_continue_deep(stmts: &[Stmt]) -> bool {
    for s in stmts {
        match s {
            Stmt::Continue => return true,
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                if contains_continue_deep(then_body) || contains_continue_deep(else_body) {
                    return true;
                }
            }
            // Nested loops own their own continues.
            Stmt::While { .. }
            | Stmt::DoWhile { .. }
            | Stmt::For { .. }
            | Stmt::ForEach { .. } => {}
            _ => {}
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Counter liveness / tail safety helpers
// ---------------------------------------------------------------------------

/// Number of writes to `name` anywhere in the list (including `for`/`using`
/// headers). Used to require EXACTLY ONE write — the canonical update.
fn count_writes_deep(list: &[Stmt], name: &str) -> usize {
    list.iter().map(|s| count_stmt_writes(s, name)).sum()
}

fn count_stmt_writes(s: &Stmt, name: &str) -> usize {
    let mut n = usize::from(matches!(s, Stmt::Assign(w, _, _) if w == name));
    // `children` does not include a `using` declaration.
    if let Stmt::Using { decl, .. } = s {
        n += count_stmt_writes(decl, name);
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        n += count_writes_deep(k, name);
    }
    n
}

/// Does `s` (or anything nested in it) read or write the local `name`?
///
/// This is a complete spelling: unlike `stmt_mentions_deep` it also covers
/// `EventAssign`/`IndexAssign` targets, so a counter hidden inside an indexer
/// write cannot slip past the liveness gate.
fn stmt_mentions_local(s: &Stmt, name: &str) -> bool {
    let own = match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => expr_mentions(e, name),
        Stmt::Assign(lhs, _, e) => {
            lhs == name || super::target_names(lhs, name) || expr_mentions(e, name)
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
        Stmt::If { cond, .. }
        | Stmt::While { cond, .. }
        | Stmt::DoWhile { cond, .. } => expr_mentions(cond, name),
        Stmt::For { init, cond, incr, .. } => {
            cond.as_ref().is_some_and(|c| expr_mentions(c, name))
                || list_mentions_local(init, name)
                || list_mentions_local(incr, name)
        }
        Stmt::ForEach {
            var_name,
            collection,
            ..
        } => var_name == name || collection == name,
        Stmt::Using { decl, .. } => stmt_mentions_local(decl, name),
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
    kids.iter().any(|k| list_mentions_local(k, name))
}

fn list_mentions_local(list: &[Stmt], name: &str) -> bool {
    list.iter().any(|s| stmt_mentions_local(s, name))
}

/// Identifier tokens in an assignment target (`this.items` → `this`, `items`).
fn target_tokens(lhs: &str) -> impl Iterator<Item = &str> {
    lhs.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
}

/// Does the tail write any variable the update's right-hand side reads?
///
/// The update is evaluated AFTER the tail once folded, so a tail write to one
/// of its inputs would feed it a different value than the flat code did. The
/// scan is recursive (an assignment inside an `if` in the tail counts).
fn tail_writes_update_inputs(list: &[Stmt], update_rhs: &Expr) -> bool {
    for s in list {
        match s {
            Stmt::Assign(lhs, _, _) => {
                if target_tokens(lhs).any(|t| expr_mentions(update_rhs, t)) {
                    return true;
                }
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                if tail_writes_update_inputs(then_body, update_rhs)
                    || tail_writes_update_inputs(else_body, update_rhs)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// The only statement kinds a relaxed tail may contain.
///
/// Labels and `goto` are excluded: a jump into/out of the tail changes how
/// many times the increment runs (in the flat form it already ran before the
/// tail; in a `for` a jump can skip it). `try`/`using`/`lock`/nested loops are
/// excluded too: they can observe or catch an exception from the relocated
/// increment, which is outside what this gate proves.
fn tail_stmt_allowed(s: &Stmt) -> bool {
    match s {
        Stmt::Comment(_)
        | Stmt::Attribute { .. }
        | Stmt::Assign(..)
        | Stmt::Eval(_)
        | Stmt::Return(_)
        | Stmt::Throw(_)
        | Stmt::Break
        | Stmt::Continue => true,
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            then_body.iter().all(tail_stmt_allowed) && else_body.iter().all(tail_stmt_allowed)
        }
        _ => false,
    }
}

/// Can executing the tail skip the `for` increment that the flat code ran
/// before the tail? `continue` runs it in both, so only `return`/`throw`/
/// `break` (at this loop level) qualify.
fn tail_can_skip_increment(list: &[Stmt]) -> bool {
    fn scan(s: &Stmt) -> bool {
        match s {
            Stmt::Return(_) | Stmt::Throw(_) | Stmt::Break => true,
            Stmt::If {
                then_body,
                else_body,
                ..
            } => then_body.iter().any(scan) || else_body.iter().any(scan),
            _ => false,
        }
    }
    list.iter().any(scan)
}

/// Is it safe to keep `tail` in the body while the update moves into the `for`
/// header? The flat form ran `update` BEFORE the tail; the folded form runs it
/// after, so the tail must not observe the counter, must not write an input of
/// the update, and must not jump around it. If a tail path skips the increment
/// (`return`/`throw`/`break`), the update must be side-effect free apart from
/// the counter write, which is dead after the loop (checked separately).
fn tail_is_safe(tail: &[Stmt], counter: &str, update_rhs: &Expr) -> bool {
    for t in tail {
        if !tail_stmt_allowed(t) {
            return false;
        }
        // The tail runs before the update in the folded form: it must not
        // observe the counter (read or write).
        if stmt_mentions_local(t, counter) {
            return false;
        }
    }
    if tail_writes_update_inputs(tail, update_rhs) {
        return false;
    }
    if tail_can_skip_increment(tail) && expr_has_call(update_rhs) {
        return false;
    }
    true
}

/// Try to fold the `while` at `i`.
fn plan_at(list: &[Stmt], i: usize) -> Result<Fold, Refusal> {
    let Stmt::While { cond, body } = &list[i] else {
        unreachable!("plan_at is only called on While statements");
    };
    ford_dbg!("candidate while at index {i}, cond={cond:?}");

    if body.is_empty() {
        return Err(Refusal::NoUpdate);
    }
    let counter = condition_counter(cond).ok_or_else(|| {
        ford_dbg!("  refused: no counter in condition");
        Refusal::NoCounter
    })?;
    LOOPS_WITH_COUNTER.fetch_add(1, Ordering::Relaxed);
    ford_dbg!("  counter={counter}");

    // The counter must be updated inside the loop body: a `while` whose
    // induction variable is never assigned in its own body is not a `for`.
    // Search recursively to handle ARM64 pattern where updates are inside if blocks.
    let update_loc = find_update_deep(body, &counter).ok_or_else(|| {
        ford_dbg!("  refused: no update to {counter} in body");
        Refusal::NoUpdate
    })?;
    LOOPS_WITH_UPDATE.fetch_add(1, Ordering::Relaxed);

    // The body may write the induction variable in exactly ONE place: the
    // canonical update that becomes the `for` increment. A second write (an
    // early initialisation, a nested re-assignment, a `counter = 7` in a
    // branch) would keep executing from inside the body, while the increment
    // moved to the header — the interleaving of those writes would change, so
    // refuse instead of reasoning about it.
    if count_writes_deep(body, &counter) != 1 {
        ford_dbg!("  refused: body writes {counter} outside the single update");
        return Err(Refusal::CounterWritten);
    }

    // Extract the update statement and build the new body.
    let (update, new_body) = match &update_loc {
        UpdateLocation::TopLevel(update_at) => {
            // Original logic for top-level updates.
            let mut end = body.len();
            if matches!(body[end - 1], Stmt::Continue) {
                end -= 1;
            }
            if *update_at >= end {
                return Err(Refusal::Continue);
            }
            // A `continue` BEFORE the update skips it in the flat form but
            // would run the `for` increment; refuse. A `continue` after the
            // update (in the tail) is safe: both forms increment exactly once
            // before re-testing the condition.
            if super::contains_continue(&body[..*update_at]) {
                ford_dbg!("  refused: non-trailing continue before the update");
                return Err(Refusal::Continue);
            }

            let update = body[*update_at].clone();
            let tail = &body[*update_at + 1..end];

            // The tail stays in the body, so it runs BEFORE the update once folded.
            // See `tail_is_safe` for the proof obligation: it may contain calls
            // and non-local control flow now, but never labels/gotos, never the
            // counter, and a path that skips the increment forces the update's
            // right-hand side to be call-free.
            let Stmt::Assign(_, _, update_rhs) = &update else {
                unreachable!("update_at is an Assign by construction")
            };
            if !tail_is_safe(tail, &counter, update_rhs) {
                ford_dbg!("  refused: unsafe tail after the update");
                return Err(Refusal::Tail);
            }

            let mut new_body: Vec<Stmt> = body[..*update_at].to_vec();
            new_body.extend_from_slice(tail);
            (update, new_body)
        }
        UpdateLocation::Nested {
            if_index,
            branch,
            index_in_branch,
        } => {
            // ARM64 pattern: update is inside an if branch.
            let Stmt::If {
                cond: if_cond,
                then_body,
                else_body,
            } = &body[*if_index]
            else {
                unreachable!("if_index points to an If by construction")
            };

            let branch_body = match branch {
                Branch::Then => then_body,
                Branch::Else => else_body,
            };

            // Safety check: the code after the `if` must EXIT the loop on every
            // path. The nested update runs only when the branch is taken; a
            // path that falls off the bottom re-tests the condition WITHOUT it
            // in the flat form, while the `for` would run its increment.
            let rest_exits = if *if_index + 1 >= body.len() {
                false
            } else {
                always_terminates(&body[*if_index + 1..])
            };

            if !rest_exits {
                ford_dbg!("  refused: nested update branch can fall through to the loop bottom");
                return Err(Refusal::Continue);
            }

            // Check that the branch contains a continue (at any depth).
            if !contains_continue_deep(branch_body) {
                ford_dbg!("  refused: nested update branch has no continue");
                return Err(Refusal::Continue);
            }

            // Check that there's no continue outside the if statement.
            let mut body_without_if = body[..*if_index].to_vec();
            body_without_if.extend_from_slice(&body[*if_index + 1..]);
            if super::contains_continue(&body_without_if) {
                ford_dbg!("  refused: continue outside the if with nested update");
                return Err(Refusal::Continue);
            }

            // Extract the update from the branch.
            let update = branch_body[*index_in_branch].clone();
            let Stmt::Assign(_, _, update_rhs) = &update else {
                unreachable!("update is an Assign by construction")
            };

            // Check the tail within the branch (statements after the update).
            // Same obligations as a top-level tail: it moves in front of the
            // update, so it must not observe the counter, must not write a
            // value the update reads, and must not contain labels/gotos.
            let tail_in_branch = &branch_body[*index_in_branch + 1..];
            if !tail_is_safe(tail_in_branch, &counter, update_rhs) {
                ford_dbg!("  refused: unsafe tail in nested branch");
                return Err(Refusal::Tail);
            }

            // Build the new branch without the update.
            let mut new_branch: Vec<Stmt> = branch_body[..*index_in_branch].to_vec();
            new_branch.extend_from_slice(tail_in_branch);

            // Build the new if statement with the update removed from the branch.
            let new_if = match branch {
                Branch::Then => Stmt::If {
                    cond: if_cond.clone(),
                    then_body: new_branch,
                    else_body: else_body.clone(),
                },
                Branch::Else => Stmt::If {
                    cond: if_cond.clone(),
                    then_body: then_body.clone(),
                    else_body: new_branch,
                },
            };

            // Build the new body with the modified if.
            let mut new_body: Vec<Stmt> = body[..*if_index].to_vec();
            new_body.push(new_if);
            new_body.extend_from_slice(&body[*if_index + 1..]);
            (update, new_body)
        }
    };

    // The init: the nearest preceding assignment to the counter.
    let init_at = (0..i)
        .rev()
        .find(|&p| matches!(&list[p], Stmt::Assign(n, _, _) if n == &counter))
        .ok_or_else(|| {
            ford_dbg!("  refused: no init for {counter} before the loop");
            Refusal::NoUpdate
        })?;
    let Stmt::Assign(_, _, init_rhs) = &list[init_at] else {
        unreachable!("the position was matched as an Assign")
    };
    LOOPS_WITH_INIT.fetch_add(1, Ordering::Relaxed);
    ford_dbg!("  init at {init_at}, between count={}", (i - init_at - 1));

    // Hoisting the init into the header moves it LATER, past everything between
    // its old position and the loop. Only WRITES to the counter between the two
    // positions are unsafe: a write there would be overwritten by the hoisted
    // init, changing the value the loop sees. READS are safe because the init
    // is still the most recent write at every such read point (that is exactly
    // how `init_at` was chosen — it is the NEAREST preceding write).
    //
    // This is the Phase 3A relaxation: the old check used `list_mentions_deep`
    // which refused on reads too, and that was the gate that blocked every
    // real corpus candidate.
    let between = &list[init_at + 1..i];
    if list_writes_deep(between, &counter) {
        ford_dbg!("  refused: between-statement writes counter {counter}");
        return Err(Refusal::CounterWriteBetween);
    }
    if expr_has_call(init_rhs) {
        ford_dbg!("  refused: init RHS has a call");
        return Err(Refusal::InitCall);
    }
    for s in between {
        if let Stmt::Assign(w, _, _) = s {
            if expr_mentions(init_rhs, w) {
                ford_dbg!("  refused: init RHS reads {w} written between");
                return Err(Refusal::CounterWriteBetween);
            }
        }
    }

    // A `for` header's declaration is scoped to the loop, so the counter must
    // not be mentioned after it. `stmt_mentions_local` covers every statement
    // kind (including indexer/event writes) so a use cannot hide from this gate.
    if list[i + 1..]
        .iter()
        .any(|s| stmt_mentions_local(s, &counter))
    {
        ford_dbg!("  refused: counter {counter} mentioned after the loop");
        return Err(Refusal::After);
    }

    // The renderer DROPS any header element without an inline form.
    if !is_inlinable(&list[init_at]) || !is_inlinable(&update) {
        ford_dbg!("  refused: init or update not inlinable");
        return Err(Refusal::NotInlinable);
    }

    ford_dbg!("  FOLDED: for ({counter} init; cond; update)");

    Ok(Fold {
        seq: i,
        init_at,
        for_stmt: Stmt::For {
            init: vec![list[init_at].clone()],
            cond: Some(cond.clone()),
            incr: vec![update],
            body: new_body,
        },
    })
}

/// The induction variable of a loop condition: a plain local, or a local
/// reached through arithmetic on it (`v33 + 1 < n`).
///
/// Returns `None` for a condition that names no local — e.g. the real
/// `while (*(nint*)(v37 -1) < *(nint*)(buffer +8))`, whose operands are raw
/// derefs of frame slots and a parameter. Refusing there is correct: a frame
/// slot has no assignment in the body to serve as the update.
fn condition_counter(cond: &Expr) -> Option<String> {
    match cond {
        Expr::Cmp(_, a, b) => induction_var(a).or_else(|| induction_var(b)),
        Expr::Var(n) => Some(n.clone()),
        Expr::Not(inner) => condition_counter(inner),
        Expr::Bin(_, a, _) => induction_var(a),
        _ => None,
    }
}

fn induction_var(e: &Expr) -> Option<String> {
    match e {
        Expr::Var(n) => Some(n.clone()),
        Expr::Bin(_, a, _) => induction_var(a),
        Expr::Cast(_, inner) | Expr::Neg(inner) => induction_var(inner),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::{BinOp, CmpKind};

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }
    fn assign(name: &str, e: Expr) -> Stmt {
        Stmt::Assign(name.to_string(), None, e)
    }
    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Bin(BinOp::Add, Box::new(a), Box::new(b))
    }
    /// `*(nint*)(base + addend)` — the real array-length spelling.
    fn deref(base: &str, addend: i64) -> Expr {
        Expr::RawDeref(Box::new(var(base)), addend, 0)
    }
    fn lt(a: Expr, b: Expr) -> Expr {
        Expr::Cmp(CmpKind::Lt, Box::new(a), Box::new(b))
    }

    /// The verbatim real shape from the corpus must fold.
    #[test]
    fn the_real_counter_loop_folds() {
        let mut stmts = vec![
            assign("v11", Expr::Int(0)),
            assign("v12", Expr::Field(Box::new(Expr::This), "attrNames".into())),
            assign("v13", Expr::Int(0)),
            assign("v14", var("name")),
            Stmt::While {
                // v13 < *(nint*)(v12 +24)
                cond: lt(var("v13"), deref("v12", 24)),
                body: vec![
                    assign("v25", var("this")),
                    assign("v11", add(var("v11"), Expr::Int(1))),
                    assign("v12", Expr::Field(Box::new(Expr::This), "attrNames".into())),
                    assign("v13", add(var("v11"), Expr::Int(1))),
                    assign("v14", var("v18")),
                    Stmt::Continue,
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        // The `for` sits where the `while` was; the init was hoisted out of
        // position 2, and the statements between it and the loop stay BEFORE
        // the loop.
        assert_eq!(stmts.len(), 4);
        assert!(matches!(&stmts[0], Stmt::Assign(n, _, _) if n == "v11"));
        assert!(matches!(&stmts[1], Stmt::Assign(n, _, _) if n == "v12"));
        assert!(
            matches!(&stmts[2], Stmt::Assign(n, _, _) if n == "v14"),
            "the copy between init and loop keeps its place"
        );
        match &stmts[3] {
            Stmt::For {
                init,
                cond,
                incr,
                body,
            } => {
                assert_eq!(init.len(), 1);
                assert!(matches!(&init[0], Stmt::Assign(n, _, Expr::Int(0)) if n == "v13"));
                assert!(
                    cond.is_some(),
                    "the `while` condition becomes the `for` condition"
                );
                assert_eq!(incr.len(), 1);
                assert!(
                    matches!(&incr[0], Stmt::Assign(n, _, rhs) if n == "v13" && expr_mentions(rhs, "v11")),
                    "the real update `v13 = v11 + 1` is the increment"
                );
                // Six statements went in. The update (`v13 = v11 + 1`) moved
                // into the header and the trailing `continue` was dropped, so
                // the body keeps the four statements that run per iteration,
                // with the tail (`v14 = v18`) still last.
                assert_eq!(body.len(), 4, "the update left the body, the tail stayed");
                assert!(matches!(&body[3], Stmt::Assign(n, _, _) if n == "v14"));
                assert!(
                    !body
                        .iter()
                        .any(|s| matches!(s, Stmt::Assign(n, _, _) if n == "v13")),
                    "no statement left in the body may still write the counter"
                );
                assert!(
                    !body.iter().any(|s| matches!(s, Stmt::Continue)),
                    "the trailing `continue` is dropped (falling off a `for` body runs the incr)"
                );
            }
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// An increment the OLD rule demanded (`v13 = v13 + 1`) must still fold.
    #[test]
    fn a_self_referencing_increment_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    Stmt::Eval(var("work")),
                    assign("i", add(var("i"), Expr::Int(1))),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::For { .. }));
    }

    /// A second `continue` at this level would skip the update in the flat form
    /// but run it in a `for` — refuse.
    #[test]
    fn another_continue_refuses_the_fold() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    Stmt::If {
                        cond: var("c"),
                        then_body: vec![Stmt::Continue],
                        else_body: vec![],
                    },
                    assign("i", add(var("i"), Expr::Int(1))),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[1], Stmt::While { .. }));
    }

    /// The counter is read after the loop: hoisting the declaration into the
    /// header would change its scope.
    #[test]
    fn counter_used_after_the_loop_refuses_the_fold() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Use".into(),
                args: vec![var("i")],
            }),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A body exactly one assignment long is still a `for` (the update is the
    /// whole body).
    #[test]
    fn a_body_that_is_only_the_update_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::For { body, .. } => assert!(body.is_empty()),
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// A raw-deref condition names no induction variable.
    #[test]
    fn a_condition_without_a_local_is_refused() {
        let mut stmts = vec![Stmt::While {
            cond: lt(deref("v37", -1), deref("buffer", 8)),
            body: vec![assign("v37", Expr::Int(1))],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// The tail moves before the update, so a tail that READS the counter is
    /// observing the pre-update value — refuse.
    #[test]
    fn a_tail_that_reads_the_counter_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign("i", add(var("i"), Expr::Int(1))),
                    assign("seen", var("i")),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A control-flow tail is allowed when the update is pure and the counter
    /// is dead after the loop: the skipped increment has no observable effect.
    #[test]
    fn a_break_tail_with_a_pure_update_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1))), Stmt::Break],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::For { body, .. } => assert!(matches!(body.first(), Some(Stmt::Break))),
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// The same `break` tail with a CALL in the update refuses: the `for` would
    /// skip that call on the break path.
    #[test]
    fn a_break_tail_with_a_call_in_the_update_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign(
                        "i",
                        Expr::Call {
                            recv: None,
                            name: "Next".into(),
                            args: vec![],
                        },
                    ),
                    Stmt::Break,
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A statement between the init and the loop that READS the counter does
    /// NOT block the hoist: the init is still the most recent write at every
    /// such read point, so the read sees the same value before and after.
    /// This is the Phase 3A relaxation — the old `list_mentions_deep` gate
    /// refused this, and that was the gate that blocked every real corpus
    /// candidate.
    #[test]
    fn a_read_between_init_and_loop_is_allowed() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            assign("saved", var("i")), // reads i, does not write it
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::Assign(n, _, _) if n == "saved"));
        assert!(matches!(&stmts[1], Stmt::For { .. }));
    }

    /// Without an init there is nothing to hoist into the header.
    #[test]
    fn a_loop_without_an_init_is_refused() {
        let mut stmts = vec![Stmt::While {
            cond: lt(var("i"), var("n")),
            body: vec![assign("i", add(var("i"), Expr::Int(1)))],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A `while` that never assigns its induction variable is not a `for`.
    #[test]
    fn a_loop_that_never_updates_the_counter_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![Stmt::Eval(var("f"))],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Nested loops each fold independently, and the recognizer runs again on
    /// the rewritten list.
    #[test]
    fn a_nested_pair_folds_twice() {
        let inner = Stmt::While {
            cond: lt(var("j"), var("m")),
            body: vec![assign("j", add(var("j"), Expr::Int(1)))],
        };
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign("j", Expr::Int(0)),
                    inner,
                    assign("i", add(var("i"), Expr::Int(1))),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 2);
        let outer = stmts
            .iter()
            .find_map(|s| match s {
                Stmt::For { body, .. } => Some(body),
                _ => None,
            })
            .expect("the outer loop folded");
        assert!(outer.iter().any(|s| matches!(s, Stmt::For { .. })));
    }

    // -----------------------------------------------------------------------
    // Phase 3A: tests for the relaxed gate and the real corpus shapes
    // -----------------------------------------------------------------------

    /// The real corpus shape where a statement between init and while READS the
    /// counter. The old `list_mentions_deep` gate refused this; the relaxed
    /// `list_writes_deep` gate allows it (correctly: the read sees the init
    /// value either way).
    #[test]
    fn real_corpus_shape_read_between_init_and_loop_folds() {
        // v13 = 0;
        // v14 = v13;         <-- reads the counter (a copy for later use)
        // v15 = name;
        // while (v13 < n) { ...; v13 = v11 + 1; continue; }
        let mut stmts = vec![
            assign("v11", Expr::Int(0)),
            assign("v13", Expr::Int(0)),
            assign("v14", var("v13")), // READS the counter — old gate refused here
            assign("v15", var("name")),
            Stmt::While {
                cond: lt(var("v13"), var("n")),
                body: vec![
                    assign("v11", add(var("v11"), Expr::Int(1))),
                    assign("v13", add(var("v11"), Expr::Int(1))),
                    Stmt::Continue,
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1, "reads between init and loop must not block the fold");
        // The for should be at the end, with v13's init hoisted.
        let last = stmts.last().unwrap();
        assert!(matches!(last, Stmt::For { .. }));
    }

    /// A statement between the init and the loop that WRITES the counter must
    /// still refuse: the hoisted init would overwrite the between-write,
    /// changing the value the loop sees.
    #[test]
    fn a_write_between_init_and_loop_is_still_refused() {
        // The nearest-init scan finds the SECOND assignment as init, so a
        // simple pair of assignments has no between-write. To test the gate,
        // we need a write to the counter that is NOT a direct assignment at
        // the top level — a nested write inside an `if` body. The init scan
        // only finds top-level `Assign`s, so `i = 0` is the init, and the
        // nested `i = 10` sits between it and the loop.
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![assign("i", Expr::Int(10))], // nested write to counter
                else_body: vec![],
            },
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
        ];
        assert_eq!(
            apply(&mut stmts), 0,
            "a nested write to the counter between init and loop must refuse"
        );
    }

    /// The exact real corpus shape: non-adjacent init, update through another
    /// temp, tail after the update, trailing continue. This is the shape that
    /// fired 0 times before Phase 3A.
    #[test]
    fn the_exact_corpus_shape_with_multiple_between_stmts() {
        // v11 = 0;
        // v12 = this.attrNames;
        // v13 = 0;                       // init
        // v14 = name;                    // between (does not mention v13)
        // v15 = v13;                     // between (READS v13 — old gate refused)
        // while (v13 < *(nint*)(v12 +24)) {
        //     v25 = this;
        //     v11 = v11 + 1;
        //     v12 = this.attrNames;
        //     v13 = v11 + 1;             // update through another temp
        //     v14 = v18;                 // tail
        //     continue;
        // }
        let mut stmts = vec![
            assign("v11", Expr::Int(0)),
            assign("v12", Expr::Field(Box::new(Expr::This), "attrNames".into())),
            assign("v13", Expr::Int(0)),
            assign("v14", var("name")),
            assign("v15", var("v13")), // reads the counter — the gate that was too strict
            Stmt::While {
                cond: lt(var("v13"), deref("v12", 24)),
                body: vec![
                    assign("v25", var("this")),
                    assign("v11", add(var("v11"), Expr::Int(1))),
                    assign("v12", Expr::Field(Box::new(Expr::This), "attrNames".into())),
                    assign("v13", add(var("v11"), Expr::Int(1))),
                    assign("v14", var("v18")),
                    Stmt::Continue,
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        match stmts.last().unwrap() {
            Stmt::For { init, incr, body, .. } => {
                assert!(matches!(&init[0], Stmt::Assign(n, _, Expr::Int(0)) if n == "v13"));
                assert!(matches!(&incr[0], Stmt::Assign(n, _, _) if n == "v13"));
                assert_eq!(body.len(), 4);
                assert!(!body.iter().any(|s| matches!(s, Stmt::Continue)));
            }
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// The refusal reason for a between-write is `CounterWriteBetween`, not
    /// some other gate. This verifies the instrumentation distinguishes the
    /// relaxed gate from the others.
    #[test]
    fn refusal_reason_distinguishes_counter_write_between() {
        // i = 0; if (true) { i = 10; } while (i < n) { i = i + 1; }
        // The nested write to `i` in the if-body is between the init and the
        // loop. The relaxed gate catches it as a write.
        let list = vec![
            assign("i", Expr::Int(0)),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![assign("i", Expr::Int(10))],
                else_body: vec![],
            },
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
        ];
        let result = plan_at(&list, 2);
        assert!(
            matches!(result, Err(Refusal::CounterWriteBetween)),
            "expected CounterWriteBetween, got {result:?}"
        );
    }

    /// A call between init and loop that reads the counter: the call is a read,
    /// not a write, so the relaxed gate allows it. The `expr_has_call` check on
    /// the INIT's RHS is a separate gate (and the init here has no call).
    #[test]
    fn a_call_between_init_and_loop_that_reads_the_counter_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Observe".into(),
                args: vec![var("i")], // reads i — but does not write it
            }),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
        ];
        assert_eq!(
            apply(&mut stmts), 1,
            "a call that reads the counter between init and loop is safe to fold"
        );
        assert!(matches!(&stmts[1], Stmt::For { .. }));
    }

    // -----------------------------------------------------------------------
    // ARM64 nested update pattern tests
    // -----------------------------------------------------------------------

    /// The ARM64 pattern: counter update inside an if branch with continue.
    /// This is the shape that fired 0 times before the nested update support.
    #[test]
    fn arm64_nested_update_with_continue_folds() {
        // v30 = 0;
        // while (v30 < n) {
        //     work;
        //     if (test) {
        //         v30 = v30 + 1;  // update INSIDE if
        //         if (null_check) {
        //             continue;
        //         }
        //     }
        //     goto L4;  // exit
        // }
        let mut stmts = vec![
            assign("v30", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("v30"), var("n")),
                body: vec![
                    Stmt::Eval(var("work")),
                    Stmt::If {
                        cond: var("test"),
                        then_body: vec![
                            assign("v30", add(var("v30"), Expr::Int(1))),
                            Stmt::If {
                                cond: var("null_check"),
                                then_body: vec![Stmt::Continue],
                                else_body: vec![],
                            },
                        ],
                        else_body: vec![],
                    },
                    Stmt::Goto("L4".into()),
                ],
            },
        ];
        assert_eq!(
            apply(&mut stmts),
            1,
            "ARM64 nested update pattern must fold"
        );
        assert_eq!(stmts.len(), 1, "init hoisted, only for-loop remains");
        match &stmts[0] {
            Stmt::For {
                init,
                cond,
                incr,
                body,
            } => {
                assert_eq!(init.len(), 1);
                assert!(matches!(&init[0], Stmt::Assign(n, _, Expr::Int(0)) if n == "v30"));
                assert!(cond.is_some());
                assert_eq!(incr.len(), 1);
                assert!(matches!(&incr[0], Stmt::Assign(n, _, _) if n == "v30"));
                // The body should have: work, if (with update removed), goto.
                assert_eq!(body.len(), 3);
                assert!(matches!(&body[0], Stmt::Eval(_)));
                assert!(matches!(&body[1], Stmt::If { .. }));
                assert!(matches!(&body[2], Stmt::Goto(_)));
            }
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// ARM64 pattern where the branch always ends with continue (no fall-through).
    #[test]
    fn arm64_nested_update_branch_always_continues() {
        // i = 0;
        // while (i < n) {
        //     work;
        //     if (test) {
        //         i = i + 1;  // update
        //         continue;   // always continues
        //     }
        //     break;  // exit
        // }
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    Stmt::Eval(var("work")),
                    Stmt::If {
                        cond: var("test"),
                        then_body: vec![
                            assign("i", add(var("i"), Expr::Int(1))),
                            Stmt::Continue,
                        ],
                        else_body: vec![],
                    },
                    Stmt::Break,
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::For { .. }));
    }

    /// ARM64 pattern refused when branch doesn't terminate and rest doesn't exit.
    #[test]
    fn arm64_nested_update_refused_when_unsafe() {
        // i = 0;
        // while (i < n) {
        //     work;
        //     if (test) {
        //         i = i + 1;  // update
        //         if (check) { continue; }
        //     }
        //     more_work;  // falls through, doesn't exit
        // }
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    Stmt::Eval(var("work")),
                    Stmt::If {
                        cond: var("test"),
                        then_body: vec![
                            assign("i", add(var("i"), Expr::Int(1))),
                            Stmt::If {
                                cond: var("check"),
                                then_body: vec![Stmt::Continue],
                                else_body: vec![],
                            },
                        ],
                        else_body: vec![],
                    },
                    Stmt::Eval(var("more_work")), // falls through, doesn't exit
                ],
            },
        ];
        assert_eq!(
            apply(&mut stmts),
            0,
            "unsafe nested update must be refused"
        );
        assert!(matches!(&stmts[1], Stmt::While { .. }));
    }

    /// ARM64 pattern with update in else branch.
    #[test]
    fn arm64_nested_update_in_else_branch() {
        // i = 0;
        // while (i < n) {
        //     work;
        //     if (test) {
        //         do_something;
        //     } else {
        //         i = i + 1;  // update in else
        //         continue;
        //     }
        //     break;
        // }
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    Stmt::Eval(var("work")),
                    Stmt::If {
                        cond: var("test"),
                        then_body: vec![Stmt::Eval(var("something"))],
                        else_body: vec![
                            assign("i", add(var("i"), Expr::Int(1))),
                            Stmt::Continue,
                        ],
                    },
                    Stmt::Break,
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::For { .. }));
    }

    /// A nested update whose false path falls off the bottom of the body: the
    /// flat form re-tested the condition without running the update, while a
    /// `for` would run its increment. Refuse.
    #[test]
    fn arm64_nested_update_with_empty_rest_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![Stmt::If {
                    cond: var("test"),
                    then_body: vec![
                        assign("i", add(var("i"), Expr::Int(1))),
                        Stmt::Continue,
                    ],
                    else_body: vec![],
                }],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[1], Stmt::While { .. }));
    }

    // -----------------------------------------------------------------------
    // Steps, conditions and update placement
    // -----------------------------------------------------------------------

    /// A descending counter: `for (i = n; i > 0; i = i - 1)`.
    #[test]
    fn a_negative_step_folds() {
        let mut stmts = vec![
            assign("i", var("n")),
            Stmt::While {
                cond: Expr::Cmp(CmpKind::Gt, Box::new(var("i")), Box::new(Expr::Int(0))),
                body: vec![
                    Stmt::Eval(var("work")),
                    assign(
                        "i",
                        Expr::Bin(BinOp::Sub, Box::new(var("i")), Box::new(Expr::Int(1))),
                    ),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::For { init, cond, incr, .. } => {
                assert!(matches!(&init[0], Stmt::Assign(n, _, _) if n == "i"));
                assert!(matches!(cond, Some(Expr::Cmp(CmpKind::Gt, ..))));
                assert!(matches!(
                    &incr[0],
                    Stmt::Assign(n, _, Expr::Bin(BinOp::Sub, ..)) if n == "i"
                ));
            }
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// An inequality test: `for (i = 0; i != n; i = i + 1)`.
    #[test]
    fn an_inequality_condition_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(var("i")), Box::new(var("n"))),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::For { cond, .. } => {
                assert!(matches!(cond, Some(Expr::Cmp(CmpKind::Ne, ..))))
            }
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// The update may sit at the TOP of the body; the statements after it move
    /// in front of it, which is safe when they are pure and do not read the
    /// counter.
    #[test]
    fn an_update_at_the_top_of_the_body_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign("i", add(var("i"), Expr::Int(1))),
                    assign("seen", var("work")),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::For { body, incr, .. } => {
                assert!(matches!(&incr[0], Stmt::Assign(n, _, _) if n == "i"));
                assert_eq!(body.len(), 1, "the tail stays in the body");
                assert!(matches!(&body[0], Stmt::Assign(n, _, _) if n == "seen"));
            }
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// A tail with a non-local exit is allowed when the update's RHS is
    /// call-free and the counter is dead after the loop: skipping the counter
    /// write is unobservable.
    #[test]
    fn a_return_tail_with_a_pure_update_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    Stmt::Eval(var("work")),
                    assign("i", add(var("i"), Expr::Int(1))),
                    Stmt::If {
                        cond: var("c"),
                        then_body: vec![Stmt::Return(None)],
                        else_body: vec![Stmt::Continue],
                    },
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::For { body, .. } => assert!(matches!(body.last(), Some(Stmt::If { .. }))),
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// The same tail with a CALL in the update is refused: a `return` path
    /// would skip that call in the folded form.
    #[test]
    fn a_return_tail_with_a_call_in_the_update_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign("i", Expr::Call {
                        recv: None,
                        name: "Next".into(),
                        args: vec![],
                    }),
                    Stmt::Return(None),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A tail that writes the counter is a semantic change: in the flat form
    /// that write happened AFTER the update; in the folded form it would be
    /// overwritten by the increment.
    #[test]
    fn a_tail_that_writes_the_counter_is_refused() {
        // v13's update reads v11, so the ordinary "update reads a tail-written
        // name" gate does not catch a tail write to v13 itself.
        let mut stmts = vec![
            assign("v11", Expr::Int(0)),
            assign("v13", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("v13"), var("n")),
                body: vec![
                    assign("v11", add(var("v11"), Expr::Int(1))),
                    assign("v13", add(var("v11"), Expr::Int(1))),
                    assign("v13", Expr::Int(7)),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Two writes to the induction variable refuse: only the single canonical
    /// update may move into the header.
    #[test]
    fn a_second_counter_write_in_the_body_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign("i", Expr::Int(5)),
                    Stmt::Eval(var("work")),
                    assign("i", add(var("i"), Expr::Int(1))),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A `goto` in the tail can jump over the relocated increment — refuse.
    #[test]
    fn a_goto_tail_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign("i", add(var("i"), Expr::Int(1))),
                    Stmt::Goto("L".into()),
                ],
            },
            Stmt::Label("L".into()),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }
}
