//! `lock (…)` and `using (…)` reconstruction.
//!
//! ## What the corpus actually contains (this is the whole point)
//!
//! Measured on the x64 corpus: `lock (` **0**, `using (` **0**, `try {` 18,
//! `finally` **0**, `Monitor.Enter` **145**.
//!
//! The previous version of this file only understood two shapes that do not
//! occur: `Monitor.Enter(x)` with ONE argument immediately followed by a
//! `try { … } finally { Monitor.Exit(x) }`, and `V = acquire()` followed by
//! `try { … } finally { V.Dispose(); }`. On this corpus `finally` is never
//! emitted at all, so both could never fire — the run report printed nothing.
//!
//! The shape that DOES occur is the flat form of the C# `lock` expansion. C#
//! compiles `lock (x) { body }` to
//!
//! ```text
//! bool lockTaken = false;
//! Monitor.Enter(x, ref lockTaken);
//! try { body } finally { if (lockTaken) Monitor.Exit(x); }
//! ```
//!
//! and when the EH structurer does not build a `finally` node (which is what
//! happens on this corpus), the region is emitted FLAT, with the cleanup
//! INLINED in front of every exit from the protected region. Verbatim from the
//! real dump (`dump.cs` line 7988):
//!
//! ```text
//! Monitor.Enter(*(nint*)(*(nint*)(v28 +152) +136), 0);
//! MSCompatUnicodeTable.FillCJKCore(…);
//! MSCompatUnicodeTable.SetCJKReferences(…);
//! Monitor.Exit(*(nint*)(*(nint*)(v28 +152) +136));
//! return;
//! ```
//!
//! ## The proof that lets this be folded
//!
//! Two independent facts identify the region and license the rewrite:
//!
//!   * **the second argument is the literal `0`.** The `lock` codegen passes a
//!     `lockTaken` out-parameter that it has just set to `false`; a hand-written
//!     `Monitor.Enter(o, ref flag)` passes a live local. A literal `0`/`false`
//!     is therefore positive evidence of `lock`, not of a hand-written monitor;
//!   * **the inlined-finally discipline.** Because the cleanup is INLINED, any
//!     exit from the protected region is preceded by exactly the matching
//!     `Monitor.Exit(<same object>)`. That is checkable, and it is what proves
//!     the native code had a `finally` in the first place. The fold then
//!     requires it BOTH ways:
//!       - every `Monitor.Exit(x)` in the region must be immediately followed by
//!         a statement that leaves the region (a `return`/`throw`, or a
//!         `goto`/`break`/`continue` whose target is outside it), and it is
//!         REMOVED as part of the fold — the `lock` statement performs it;
//!       - every statement in the region that leaves it must be immediately
//!         preceded by such a removed `Monitor.Exit(x)`. Without this direction
//!         the fold would ADD a release on a path where the flat code had none,
//!         i.e. it would change observable behaviour.
//!     A `break`/`continue` only leaves the region at loop depth 0 — inside a
//!     nested loop it belongs to that loop and stays in the region;
//!   * no nested `Monitor.Enter(x)` for the SAME object inside the region (a
//!     re-entrant pair is not one `lock`);
//!   * the pairing is unambiguous: scanning forward from the `Enter`, the next
//!     `Monitor.Enter(x)`/`Monitor.Exit(x)` on the same object must be the
//!     `Exit`. Two locks on the same object in one list
//!     (`Enter; A; Exit; Enter; B; Exit;`) therefore pair correctly;
//!   * a label inside the region that some `goto` OUTSIDE the region targets
//!     blocks the fold: C# does not permit jumping INTO a `lock` statement, so
//!     the label would become unreachable in the output while the `goto` stayed
//!     (the dangling-jump-target class). Labels targeted only from within the
//!     region are fine — they move with it.
//!
//! ## Status as MEASURED, not as intended
//!
//! The fold above did NOT fire on the x64 corpus despite that corpus containing
//! 145 `Monitor.Enter` statements (the run report printed no `using_lock` line
//! at all). Reading the gates cannot tell "the shape never occurs in one list"
//! from "a gate refuses everything", and those two need opposite responses, so
//! the pass is now INSTRUMENTED: it counts the sites it examines and the reason
//! each one was left alone, and `report_refusals` prints both through the run
//! report. Only after that breakdown exists is it responsible to relax a gate —
//! every gate here is sound, so a blind relaxation would be trading a silent
//! pass for a wrong program.
//!
//! The gate I would look at first is [`Refusal::NoPair`]: the fold pairs an
//! `Enter` with the next same-object event **in the same statement list**, and
//! the structurer emits unrecoverable regions as separately-labelled blocks. If
//! the measured breakdown is dominated by `NoPair`, the `Exit` is in a sibling
//! list and the fix is to pair across the two lists that form one region — not
//! to weaken the pairing rule.
//!
//! ## What this file deliberately does NOT do
//!
//! The `try/finally` spellings are kept, because they are the correct shape
//! wherever the structurer DOES build a `Try` node — but on the x64 corpus they
//! are unreachable (`finally` count 0) and must not be counted as working. The
//! `using` fold has no reachable spelling here at all: `Dispose()` appears
//! 2,050 times and almost every one of them is the *declaration*
//! `public void Dispose() {`, not a call, so there is no
//! `V = acquire(); try { … } finally { V.Dispose(); }` to find. There is no
//! second, weaker `using` shape to fold: without the `Dispose` call as the
//! cleanup, `V = acquire(); …; V.Dispose();` is indistinguishable from an
//! ordinary local that happens to be disposed at the end of the method, and
//! claiming a `using` there would invent a scope the native code does not have.

use super::{
    children, children_mut, exprs_eq, is_inlinable, leaves_region, list_writes_deep, monitor_stmt,
    monitor_stmt_on, stmt_reads_deep, stmt_writes, stmt_writes_deep, walk_mut, Expr, Stmt,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Why sites are refused (the difference between a measurement and a silence)
// ---------------------------------------------------------------------------
//
// This pass reported ZERO rewrites on a corpus that contains 145
// `Monitor.Enter` sites. A bare zero cannot distinguish "the shape never
// occurs" from "a gate refuses everything", and those two have opposite
// consequences: the first means the pass is dead code, the second means one
// gate is wrong. These counters answer which — the run report now prints the
// population of sites AND the reason each one was left alone.

/// Every `Monitor.Enter` statement examined (counted once per statement list).
static ENTER_SITES: AtomicU64 = AtomicU64::new(0);
static FOLDED: AtomicU64 = AtomicU64::new(0);
/// No second same-target monitor event in this list.
static REFUSED_NO_PAIR: AtomicU64 = AtomicU64::new(0);
/// A same-target event precedes the `Enter`, or the alternation is broken.
static REFUSED_ORDER: AtomicU64 = AtomicU64::new(0);
/// Neither the literal `lockTaken` nor an inlined cleanup is present.
static REFUSED_NO_EVIDENCE: AtomicU64 = AtomicU64::new(0);
/// A label in the region is targeted from outside it.
static REFUSED_LABEL: AtomicU64 = AtomicU64::new(0);
/// The inlined-finally discipline does not hold.
static REFUSED_DISCIPLINE: AtomicU64 = AtomicU64::new(0);

/// Why the inline-`lock` fold refused one site.
#[derive(Clone, Copy, Debug)]
enum Refusal {
    NoPair,
    Order,
    NoEvidence,
    LabelMovesIn,
    Discipline,
}

/// Debug logging gated on `ILR_LOCKDBG`.
macro_rules! lock_dbg {
    ($($arg:tt)*) => {
        if std::env::var("ILR_LOCKDBG").is_ok() {
            eprintln!("[lock-dbg] {}", format!($($arg)*));
        }
    };
}

impl Refusal {
    fn count(self) {
        let c = match self {
            Refusal::NoPair => &REFUSED_NO_PAIR,
            Refusal::Order => &REFUSED_ORDER,
            Refusal::NoEvidence => &REFUSED_NO_EVIDENCE,
            Refusal::LabelMovesIn => &REFUSED_LABEL,
            Refusal::Discipline => &REFUSED_DISCIPLINE,
        };
        c.fetch_add(1, Ordering::Relaxed);
    }
}

/// One line per non-empty counter, for the run report.
///
/// The refusal breakdown is the whole point: a gate that names no site is a
/// dead pass, and a gate that names every site is a bug.
pub fn report_refusals() -> String {
    let sites = ENTER_SITES.load(Ordering::Relaxed);
    let folded = FOLDED.load(Ordering::Relaxed);
    if sites == 0 {
        return String::new();
    }
    format!(
        "[reconstruct]   using_lock: {sites} Monitor.Enter site(s) examined, {folded} folded; \
         refused — no pairing Exit {}, event order {}, no lock-codegen evidence {}, \
         label reachable from outside {}, inlined-finally discipline {}\n",
        REFUSED_NO_PAIR.load(Ordering::Relaxed),
        REFUSED_ORDER.load(Ordering::Relaxed),
        REFUSED_NO_EVIDENCE.load(Ordering::Relaxed),
        REFUSED_LABEL.load(Ordering::Relaxed),
        REFUSED_DISCIPLINE.load(Ordering::Relaxed),
    )
}

/// Classify the FIRST refusable `Monitor.Enter` site in a list that folded
/// nothing, so its reason reaches the report. Called once per fully-refused
/// list (the fold loop returns immediately afterwards), which keeps the
/// counters a population rather than a retry count.
fn record_first_refusal(list: &[Stmt], global_gotos: &HashSet<String>) {
    for i in 0..list.len() {
        if monitor_stmt(&list[i], "Enter").is_none() {
            continue;
        }
        if let Err(r) = plan_inline_lock_at(list, i, global_gotos) {
            lock_dbg!("refused Enter at {i}: {r:?}");
            r.count();
        }
        return;
    }
}

/// Run the recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    // The fallthrough merge pre-pass splices child lists that fall through into
    // the parent list, so that cross-list lock/using patterns (e.g. Enter in an
    // if-then body with Exit in the parent list) become visible to the same-list
    // pairing logic.
    super::merge_fallthrough_lists(stmts);
    // Goto targets for the whole body: needed to tell a label a `goto` from
    // OUTSIDE the region reaches (which blocks the fold) from one that only
    // the region itself reaches (which moves with it).
    let mut global_gotos: HashSet<String> = HashSet::new();
    super::collect_goto_targets(stmts, &mut global_gotos);
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        n += rewrite_list(list, &global_gotos);
    });
    n
}

fn rewrite_list(list: &mut Vec<Stmt>, global_gotos: &HashSet<String>) -> usize {
    // Count this list's `Monitor.Enter` sites ONCE. The fold loop below rescans
    // the same list after every rewrite, so counting inside it would multiply
    // the population by the number of rewrites instead of measuring it.
    for s in list.iter() {
        if monitor_stmt(s, "Enter").is_some() {
            ENTER_SITES.fetch_add(1, Ordering::Relaxed);
        }
    }
    let mut n = 0usize;
    loop {
        if let Some(fold) = plan_inline_lock(list, global_gotos) {
            lock_dbg!(
                "fold Enter[{}]..Exit[{}] target={:?}",
                fold.enter,
                fold.exit,
                fold.target
            );
            apply_inline_lock(list, fold);
            n += 1;
            FOLDED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if let Some(fold) = plan_try_lock(list, global_gotos) {
            splice_pair(list, fold);
            n += 1;
            FOLDED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if let Some(fold) = plan_using(list, global_gotos) {
            splice_pair(list, fold);
            n += 1;
            FOLDED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if let Some(fold) = plan_flat_using(list, global_gotos) {
            splice_pair(list, fold);
            n += 1;
            FOLDED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        // Nothing left to fold. Record WHY the remaining sites were left alone
        // so the report can never show a bare zero for this pass again.
        record_first_refusal(list, global_gotos);
        return n;
    }
}

// ---------------------------------------------------------------------------
// lock, from the flat inlined-finally shape
// ---------------------------------------------------------------------------

/// A planned `Monitor.Enter`/`Monitor.Exit` fold.
struct InlineLock {
    enter: usize,
    exit: usize,
    target: Expr,
    body: Vec<Stmt>,
}

fn apply_inline_lock(list: &mut Vec<Stmt>, fold: InlineLock) {
    let lock = Stmt::Lock {
        expr: fold.target,
        body: fold.body,
    };
    let mut out: Vec<Stmt> = Vec::with_capacity(list.len());
    for (idx, s) in std::mem::take(list).into_iter().enumerate() {
        if idx == fold.enter {
            out.push(lock.clone());
            continue;
        }
        if idx > fold.enter && idx <= fold.exit {
            continue; // consumed by the `lock`
        }
        out.push(s);
    }
    *list = out;
}

fn plan_inline_lock(list: &[Stmt], global_gotos: &HashSet<String>) -> Option<InlineLock> {
    (0..list.len()).find_map(|i| plan_inline_lock_at(list, i, global_gotos).ok())
}

/// Plan the fold at `i`, or report the reason it cannot be done.
fn plan_inline_lock_at(
    list: &[Stmt],
    i: usize,
    global_gotos: &HashSet<String>,
) -> Result<InlineLock, Refusal> {
    let enter = monitor_stmt(&list[i], "Enter").ok_or(Refusal::NoPair)?;
    let target = enter.arg.clone();
    let lock_taken_var = enter.lock_taken_var.clone();

    // Every same-object monitor event at THIS level, in order. They must
    // alternate `Enter, Exit, Enter, …` starting at `i`: that is what makes the
    // pairing unambiguous (two locks on the same object in one list pair one
    // each, and a stray second `Exit` refuses instead of being merged).
    let mut events: Vec<usize> = Vec::new();
    for (k, s) in list.iter().enumerate() {
        if monitor_stmt_on(s, "Enter", &target) || monitor_exit_matches(s, &target, &lock_taken_var) {
            events.push(k);
        }
    }
    if events.first() != Some(&i) {
        return Err(Refusal::Order);
    }
    for (n, k) in events.iter().enumerate() {
        let is_enter = monitor_stmt_on(&list[*k], "Enter", &target);
        if is_enter != (n % 2 == 0) {
            return Err(Refusal::Order);
        }
    }
    
    // No second same-target event in THIS list: the `Exit` may be inside
    // nested control flow (if/while blocks). Search for it.
    let exit = if let Some(&exit_idx) = events.get(1) {
        exit_idx
    } else if lock_taken_var.is_some() || enter.lit_lock_taken {
        // ARM32 pattern: Exit is inside nested control flow.
        // First verify there IS an Exit somewhere in the region.
        if !region_has_exit_in_nested(list, i + 1, &target, &lock_taken_var) {
            return Err(Refusal::NoPair);
        }
        // Find the end of the region (first leaving statement at top level).
        find_region_end(list, i).ok_or(Refusal::NoPair)?
    } else {
        return Err(Refusal::NoPair);
    };

    let region = &list[i + 1..exit];

    // Positivity: either the `lock` codegen's literal `lockTaken`, or the
    // inlined-finally discipline inside the region, must be present. A bare
    // one-argument `Enter` with a perfectly flat body proves neither, so it is
    // refused rather than guessed.
    let mut region_labels: HashSet<String> = HashSet::new();
    super::collect_labels(region, &mut region_labels);
    let mut region_gotos: HashSet<String> = HashSet::new();
    super::collect_goto_targets(region, &mut region_gotos);

    let has_inlined_exit = region_has_inlined_exit(region, &target, &lock_taken_var);
    if !enter.lit_lock_taken && !has_inlined_exit {
        return Err(Refusal::NoEvidence);
    }

    // A label reached by a `goto` from outside the region cannot move into a
    // `lock` body.
    if region_labels
        .iter()
        .any(|l| global_gotos.contains(l) && !region_gotos.contains(l))
    {
        return Err(Refusal::LabelMovesIn);
    }

    // Vet and strip the inlined cleanups.
    let mut body = region.to_vec();
    if !strip_inlined_exits(&mut body, &target, &lock_taken_var, 0, &region_labels) {
        return Err(Refusal::Discipline);
    }

    Ok(InlineLock {
        enter: i,
        exit,
        target,
        body,
    })
}

/// Find the end of the lock region when Exit is inside nested control flow.
/// Returns the index of the first leaving statement (return/throw/goto) at the
/// top level after position `enter_idx`, or the end of the list.
fn find_region_end(list: &[Stmt], enter_idx: usize) -> Option<usize> {
    for k in (enter_idx + 1)..list.len() {
        match &list[k] {
            Stmt::Return(_) | Stmt::Throw(_) => return Some(k),
            Stmt::Goto(_) => return Some(k),
            _ => {}
        }
    }
    // No leaving statement found; use the end of the list.
    Some(list.len())
}

/// Does the region after `start_idx` contain any Exit call matching the target
/// or lockTaken variable, at any nesting depth?
fn region_has_exit_in_nested(
    list: &[Stmt],
    start_idx: usize,
    target: &Expr,
    lock_taken_var: &Option<Expr>,
) -> bool {
    for s in list.iter().skip(start_idx) {
        if monitor_exit_matches(s, target, lock_taken_var) {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for kid in kids {
            if region_has_exit_in_nested(kid, 0, target, lock_taken_var) {
                return true;
            }
        }
    }
    false
}

/// Does `s` match a Monitor.Exit on either the original target OR the lockTaken
/// variable (dereferenced)?
fn monitor_exit_matches(s: &Stmt, target: &Expr, lock_taken_var: &Option<Expr>) -> bool {
    if monitor_stmt_on(s, "Exit", target) {
        return true;
    }
    // ARM32 3-arg pattern: Exit is on the lockTaken variable (dereferenced).
    if let Some(lt_var) = lock_taken_var {
        let exit_target = Expr::RawDeref(Box::new(Expr::AddressOf(Box::new((*lt_var).clone()))), 0, 0);
        if monitor_stmt_on(s, "Exit", &exit_target) {
            return true;
        }
    }
    false
}

/// Does the region contain the positive evidence of an inlined `finally` — an
/// `Exit(target)` or `Exit(lockTakenVar)` that is either:
///   * at the top level and immediately followed by a statement that leaves the
///     region, OR
///   * inside nested control flow (if/while/for blocks), which is the ARM32
///     pattern where the Exit is part of the cleanup code inside an if block
///     that checks the lockTaken flag.
fn region_has_inlined_exit(list: &[Stmt], target: &Expr, lock_taken_var: &Option<Expr>) -> bool {
    region_has_inlined_exit_inner(list, target, lock_taken_var, 0)
}

fn region_has_inlined_exit_inner(
    list: &[Stmt],
    target: &Expr,
    lock_taken_var: &Option<Expr>,
    depth: usize,
) -> bool {
    for (k, s) in list.iter().enumerate() {
        let is_exit = monitor_stmt_on(s, "Exit", target)
            || lock_taken_var.as_ref().is_some_and(|lt_var| {
                let exit_target = Expr::RawDeref(Box::new(Expr::AddressOf(Box::new((*lt_var).clone()))), 0, 0);
                monitor_stmt_on(s, "Exit", &exit_target)
            });
        if is_exit {
            // An inlined cleanup is only evidence when the path it guards
            // actually leaves the protected region. At the region's own level
            // `return`/`throw`/`goto`/`break`/`continue` all leave it; inside a
            // nested LOOP only `return`/`throw` do (a `break` just ends that
            // loop and the region continues with the lock released).
            if k + 1 < list.len() && cleanup_followed_by_leave(&list[k + 1], depth) {
                return true;
            }
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for kid in kids {
            if region_has_inlined_exit_inner(kid, target, lock_taken_var, depth + loop_delta(s)) {
                return true;
            }
        }
    }
    false
}

/// Is the statement after an inlined `Monitor.Exit` one that leaves the lock
/// region at this nesting depth?
fn cleanup_followed_by_leave(next: &Stmt, depth: usize) -> bool {
    if depth == 0 {
        leaves_region(next)
    } else {
        // Inside a nested loop a `break`/`continue`/`goto` may stay inside the
        // region; only a method exit is unambiguous.
        matches!(next, Stmt::Return(_) | Stmt::Throw(_))
    }
}

/// Vet the region and remove every inlined `Monitor.Exit(target)` cleanup.
///
/// Returns `false` when the region does not obey the discipline, in which case
/// the caller leaves the flat form untouched:
///   * an `Exit(target)` at the TOP level NOT immediately followed by a statement
///     that leaves the region is not an inlined cleanup;
///   * an `Enter(target)` is a re-entrant pair;
///   * a leaving statement NOT preceded by a removed `Exit(target)` would gain a
///     release from the fold that it does not have today.
///
/// For Exit calls inside nested control flow (if/while/for blocks), the same
/// discipline holds with the region-exit spelling narrowed: at the region's own
/// level `break`/`continue`/`goto` leave it, but inside a nested loop only
/// `return`/`throw` do. A nested cleanup that is merely followed by more loop
/// body is NOT a region exit and refuses the fold.
fn strip_inlined_exits(
    list: &mut Vec<Stmt>,
    target: &Expr,
    lock_taken_var: &Option<Expr>,
    depth: usize,
    region_labels: &HashSet<String>,
) -> bool {
    let exit_matches = |s: &Stmt| -> bool {
        if monitor_stmt_on(s, "Exit", target) {
            return true;
        }
        if let Some(lt_var) = lock_taken_var {
            let exit_target = Expr::RawDeref(Box::new(Expr::AddressOf(Box::new((*lt_var).clone()))), 0, 0);
            if monitor_stmt_on(s, "Exit", &exit_target) {
                return true;
            }
        }
        false
    };
    let mut remove = vec![false; list.len()];
    for (k, s) in list.iter().enumerate() {
        if monitor_stmt_on(s, "Enter", target) {
            return false;
        }
        if exit_matches(s) {
            // The cleanup must actually guard a region exit. At the top level
            // any leaving statement qualifies; inside a nested loop only a
            // method exit does (a `break`/`continue` stays in the region, so
            // removing the Exit there would hold the lock longer).
            if !list
                .get(k + 1)
                .is_some_and(|next| cleanup_followed_by_leave(next, depth))
            {
                return false;
            }
            remove[k] = true;
        }
        // A statement that leaves the region must have its cleanup right in
        // front of it; otherwise the fold would ADD a release on that path.
        // `break`/`continue` only leave at loop depth 0; a `goto` to a label
        // inside the region stays inside it. This applies at every depth: a
        // `return` inside a nested loop still exits the method.
        let leaves = match s {
            Stmt::Return(_) | Stmt::Throw(_) => true,
            Stmt::Goto(l) => !region_labels.contains(l),
            Stmt::Break | Stmt::Continue => depth == 0,
            _ => false,
        };
        if leaves {
            let preceded = k > 0 && exit_matches(&list[k - 1]);
            if !preceded {
                return false;
            }
        }
    }
    if remove.iter().any(|r| *r) {
        let mut idx = 0usize;
        list.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
    }
    // Recurse into the surviving nested bodies, with the loop depth each of
    // them starts at.
    for s in list.iter_mut() {
        let child_depth = depth + loop_delta(s);
        for kid in children_mut(s) {
            if !strip_inlined_exits(kid, target, lock_taken_var, child_depth, region_labels) {
                return false;
            }
        }
    }
    true
}

/// How much a statement deepens the loop nesting for its own body.
fn loop_delta(s: &Stmt) -> usize {
    match s {
        Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::ForEach { .. } => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// lock, from a recovered `try/finally` (unreachable on the x64 corpus)
// ---------------------------------------------------------------------------

struct PairFold {
    at: usize,
    /// The last statement index consumed by this fold (inclusive). For
    /// try/finally patterns this is `at + 1`; for flat using it is the index
    /// of the Dispose/Exit call, which may be far past `at + 1`.
    range_end: usize,
    replacement: Stmt,
}

/// `Monitor.Enter(x)` then `try { … } finally { Monitor.Exit(x) }` → `lock (x)`.
fn plan_try_lock(list: &[Stmt], global_gotos: &HashSet<String>) -> Option<PairFold> {
    for i in 0..list.len().saturating_sub(1) {
        let Some(enter) = monitor_stmt(&list[i], "Enter") else {
            continue;
        };
        let target = enter.arg;
        let Stmt::Try {
            try_body,
            catches,
            finally_body,
        } = &list[i + 1]
        else {
            continue;
        };
        // A lock is a bare try/finally; any catch means this is a real try.
        if !catches.is_empty() {
            continue;
        }
        let Some(fin) = finally_body else { continue };
        let Some(exited) = single_monitor_arg(fin, "Exit") else {
            continue;
        };
        if !exprs_eq(&exited, target) {
            continue;
        }
        // C# forbids jumping INTO a `lock` statement, so a label in the
        // protected body that a `goto` from outside targets cannot move in.
        if labels_reached_from_outside(try_body, global_gotos)
            || labels_reached_from_outside(fin, global_gotos)
        {
            continue;
        }
        return Some(PairFold {
            at: i,
            range_end: i + 1,
            replacement: Stmt::Lock {
                expr: target.clone(),
                body: try_body.clone(),
            },
        });
    }
    None
}

/// Is a label DEFINED in `region` targeted by a `goto` that is not itself in
/// the region? Such a label would become unreachable after the region moves
/// inside a `lock`/`using` (C# forbids jumping into either).
fn labels_reached_from_outside(region: &[Stmt], global_gotos: &HashSet<String>) -> bool {
    let mut labels: HashSet<String> = HashSet::new();
    super::collect_labels(region, &mut labels);
    if labels.is_empty() {
        return false;
    }
    let mut local_targets: HashSet<String> = HashSet::new();
    super::collect_goto_targets(region, &mut local_targets);
    labels
        .iter()
        .any(|l| global_gotos.contains(l) && !local_targets.contains(l))
}

/// The argument of the one meaningful statement of a `finally` body when that
/// statement is `Monitor.Exit(x)` (comments/labels carry no semantics; a second
/// meaningful statement refuses the match).
fn single_monitor_arg(fin: &[Stmt], member: &str) -> Option<Expr> {
    let only = single_meaningful(fin)?;
    monitor_stmt(only, member).map(|c| c.arg.clone())
}

// ---------------------------------------------------------------------------
// using, from a recovered `try/finally` (unreachable on the x64 corpus)
// ---------------------------------------------------------------------------

/// `V = <acquire>` then `try { … } finally { V.Dispose(); }` →
/// `using (V = acquire)`.
fn plan_using(list: &[Stmt], global_gotos: &HashSet<String>) -> Option<PairFold> {
    for i in 0..list.len().saturating_sub(1) {
        let Stmt::Assign(var, ty, acquire) = &list[i] else {
            continue;
        };
        if !is_disposable_acquire(acquire) {
            continue;
        }
        let Stmt::Try {
            try_body,
            catches,
            finally_body,
        } = &list[i + 1]
        else {
            continue;
        };
        if !catches.is_empty() {
            continue;
        }
        let Some(fin) = finally_body else { continue };
        let Some(disposed) = single_dispose_receiver(fin) else {
            continue;
        };
        if disposed != *var {
            continue;
        }
        // The declaration must be renderable in a header (`using (<decl>)`).
        if !is_inlinable(&list[i]) {
            continue;
        }
        // A C# `using` declaration variable is read-only in the body: a
        // reassignment would dispose a different object than the flat code
        // disposed (and would not compile).
        if list_writes_deep(try_body, var) {
            continue;
        }
        // `var` must not be read after the try: a live-out resource is an
        // ordinary local, not a scoped `using`.
        if list[i + 2..].iter().any(|s| stmt_reads_deep(s, var)) {
            continue;
        }
        // No `goto` may jump into the `using` body.
        if labels_reached_from_outside(try_body, global_gotos)
            || labels_reached_from_outside(fin, global_gotos)
        {
            continue;
        }
        return Some(PairFold {
            at: i,
            range_end: i + 1,
            replacement: Stmt::Using {
                decl: Box::new(Stmt::Assign(var.clone(), ty.clone(), acquire.clone())),
                body: try_body.clone(),
            },
        });
    }
    None
}

/// Does `e` look like an acquire that yields an object with a `Dispose`?
/// A call, an allocation or a raw call (optionally behind a cast) qualifies;
/// a plain variable/literal does not.
fn is_disposable_acquire(e: &Expr) -> bool {
    match e {
        Expr::Call { .. } | Expr::NewObj(..) | Expr::RawCall(..) => true,
        Expr::Cast(_, inner) => is_disposable_acquire(inner),
        _ => false,
    }
}

/// The receiver name of `V.Dispose()` when the `finally` body's only meaningful
/// statement is exactly that call with no arguments.
fn single_dispose_receiver(fin: &[Stmt]) -> Option<String> {
    let only = single_meaningful(fin)?;
    let Stmt::Eval(Expr::Call { recv, name, args }) = only else {
        return None;
    };
    if !args.is_empty() {
        return None;
    }
    if !(name == "Dispose" || name.ends_with(".Dispose")) {
        return None;
    }
    match recv.as_deref() {
        Some(Expr::Var(r)) => Some(r.clone()),
        // `V.Dispose()` may also be spelled with the receiver folded into the
        // name; only accept it when there is no separate receiver.
        None => name.strip_suffix(".Dispose").map(str::to_string),
        _ => None,
    }
}

/// The single meaningful (non-comment, non-label) statement of `body`, or `None`
/// when there are zero or more than one.
fn single_meaningful(body: &[Stmt]) -> Option<&Stmt> {
    let mut it = body
        .iter()
        .filter(|s| !matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. } | Stmt::Label(_)));
    let only = it.next()?;
    if it.next().is_some() {
        return None;
    }
    Some(only)
}

// ---------------------------------------------------------------------------
// using, from the flat shape (no try/finally)
// ---------------------------------------------------------------------------

/// `V = <acquire>; …; V.Dispose();` → `using (V = acquire) { … }`.
///
/// This handles the flat spelling where the `Dispose()` call appears inline
/// rather than inside a `finally` block. On the x64 corpus `finally` is never
/// emitted (count 0), so the try/finally-based [`plan_using`] is unreachable
/// here; this function covers the shape that actually occurs.
///
/// Safety conditions:
///   * the acquire must look like a disposable construction (call, new, raw
///     call, or a cast of one);
///   * no early exit (`return`/`throw`/`goto`/`break`/`continue`) between the
///     acquire and the `Dispose` — without a `finally`, an exception on those
///     paths would skip the disposal;
///   * `var` is not reassigned between the acquire and the `Dispose`;
///   * the `Dispose` call is the last use of `var` in the list (a live-out
///     resource is an ordinary local, not a scoped `using`);
///   * the declaration is renderable inline (`Assign` or `Eval`).
fn plan_flat_using(list: &[Stmt], global_gotos: &HashSet<String>) -> Option<PairFold> {
    for i in 0..list.len().saturating_sub(1) {
        let Stmt::Assign(var, ty, acquire) = &list[i] else {
            continue;
        };
        if !is_disposable_acquire(acquire) {
            continue;
        }
        if !is_inlinable(&list[i]) {
            continue;
        }
        // If `var` was written earlier in the list, the Dispose we find may
        // belong to that earlier value, not this one — refuse to avoid a
        // spurious fold on a reassigned variable.
        if list[..i].iter().any(|s| stmt_writes(s, var)) {
            continue;
        }
        // Scan forward for V.Dispose() on the same variable.
        let mut dispose_idx = None;
        for j in (i + 1)..list.len() {
            let s = &list[j];
            // Any early exit between acquire and Dispose would skip the
            // disposal on that path — refuse. The scan is DEEP: a `return`
            // inside an `if` (or nested loop) leaves the method too, and a
            // `using` would newly dispose on that path. A `break`/`continue`
            // of a NESTED loop does not leave this region.
            if deeply_leaves_region(s, false) {
                dispose_idx = None;
                break;
            }
            // Reassignment of the variable breaks the link. This is also deep:
            // an assignment inside an `if` changes which object the Dispose
            // targets while the `using` would dispose the acquired one.
            if stmt_writes_deep(s, var) {
                dispose_idx = None;
                break;
            }
            if is_dispose_call(s, var) {
                dispose_idx = Some(j);
                break;
            }
        }
        let Some(d_idx) = dispose_idx else { continue };
        // The Dispose must be the last use of var in the remaining list:
        // a live-out resource is an ordinary local, not a scoped `using`.
        if list[d_idx + 1..].iter().any(|s| stmt_reads_deep(s, var)) {
            continue;
        }
        let body = list[i + 1..d_idx].to_vec();
        // No `goto` may jump into the `using` body.
        if labels_reached_from_outside(&body, global_gotos) {
            continue;
        }
        return Some(PairFold {
            at: i,
            range_end: d_idx,
            replacement: Stmt::Using {
                decl: Box::new(Stmt::Assign(var.clone(), ty.clone(), acquire.clone())),
                body,
            },
        });
    }
    None
}

/// Does `s` leave the statement list it sits in, at any nesting depth?
///
/// `return`/`throw`/`goto` always leave. `break`/`continue` leave only when
/// they are not owned by a nested loop (a `break` inside a `while` stays in
/// this region). Used by the flat `using` fold: an exit path that skips the
/// flat `Dispose` would gain one from the `using`.
fn deeply_leaves_region(s: &Stmt, in_loop: bool) -> bool {
    match s {
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Goto(_) => true,
        Stmt::Break | Stmt::Continue => !in_loop,
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            then_body.iter().any(|s| deeply_leaves_region(s, in_loop))
                || else_body.iter().any(|s| deeply_leaves_region(s, in_loop))
        }
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::ForEach { body, .. } => body.iter().any(|s| deeply_leaves_region(s, true)),
        Stmt::For { body, .. } => body.iter().any(|s| deeply_leaves_region(s, true)),
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            try_body.iter().any(|s| deeply_leaves_region(s, in_loop))
                || catches
                    .iter()
                    .any(|(_, b)| b.iter().any(|s| deeply_leaves_region(s, in_loop)))
                || finally_body
                    .as_ref()
                    .is_some_and(|f| f.iter().any(|s| deeply_leaves_region(s, in_loop)))
        }
        Stmt::Switch { cases, tail, default, .. } => {
            cases
                .iter()
                .any(|(_, b)| b.iter().any(|s| deeply_leaves_region(s, in_loop)))
                || tail.iter().any(|s| deeply_leaves_region(s, in_loop))
                || default
                    .as_ref()
                    .is_some_and(|arm| arm.iter().any(|s| deeply_leaves_region(s, in_loop)))
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            cases
                .iter()
                .any(|(_, b)| b.iter().any(|s| deeply_leaves_region(s, in_loop)))
                || tail
                    .as_ref()
                    .is_some_and(|t| t.iter().any(|s| deeply_leaves_region(s, in_loop)))
        }
        Stmt::Lock { body, .. } | Stmt::Using { body, .. } => {
            body.iter().any(|s| deeply_leaves_region(s, in_loop))
        }
        _ => false,
    }
}

/// Is `s` exactly `var.Dispose()` (no arguments)?
fn is_dispose_call(s: &Stmt, var: &str) -> bool {
    let Stmt::Eval(Expr::Call { recv, name, args }) = s else {
        return false;
    };
    if !args.is_empty() {
        return false;
    }
    if !(name == "Dispose" || name.ends_with(".Dispose")) {
        return false;
    }
    match recv.as_deref() {
        Some(Expr::Var(r)) => r == var,
        None => name.strip_suffix(".Dispose").is_some_and(|r| r == var),
        _ => false,
    }
}

/// Replace the statements from `at` through `range_end` (inclusive) with one.
fn splice_pair(list: &mut Vec<Stmt>, fold: PairFold) {
    let mut out: Vec<Stmt> = Vec::with_capacity(list.len());
    for (idx, s) in std::mem::take(list).into_iter().enumerate() {
        if idx == fold.at {
            out.push(fold.replacement.clone());
            continue;
        }
        if idx > fold.at && idx <= fold.range_end {
            continue; // consumed by the fold
        }
        out.push(s);
    }
    *list = out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::CmpKind;

    fn v(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }
    /// `Monitor.<member>(arg)` / `Monitor.<member>(arg, 0)` as a statement.
    fn monitor(member: &str, arg: Expr, lock_taken: bool) -> Stmt {
        let mut args = vec![arg];
        if lock_taken {
            args.push(Expr::Int(0));
        }
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("Monitor"))),
            name: member.into(),
            args,
        })
    }
    /// `*(nint*)(v28 +152)` — the real, non-trivial target expression.
    fn deref() -> Expr {
        Expr::RawDeref(Box::new(v("v28")), 152, 0)
    }

    /// The refusal REASONS are distinguishable, which is what makes the run
    /// report actionable: an `Exit` that lives in another statement list is a
    /// different defect from a broken inlined-finally discipline, and they need
    /// different fixes.
    #[test]
    fn refusal_reasons_are_distinguished() {
        // No second same-object event in this list: the `Exit` is elsewhere.
        let list = vec![monitor("Enter", v("gate"), true), Stmt::Eval(v("body"))];
        assert!(matches!(
            plan_inline_lock_at(&list, 0, &HashSet::new()),
            Err(Refusal::NoPair)
        ));

        // A one-argument Enter with no literal `lockTaken` and no inlined
        // cleanup proves nothing about the source.
        let list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
        ];
        assert!(matches!(
            plan_inline_lock_at(&list, 0, &HashSet::new()),
            Err(Refusal::NoEvidence)
        ));

        // An early exit with no cleanup in front of it means the fold would ADD
        // a release on that path.
        let list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![],
            },
            monitor("Exit", v("gate"), false),
        ];
        assert!(matches!(
            plan_inline_lock_at(&list, 0, &HashSet::new()),
            Err(Refusal::Discipline)
        ));
    }

    /// The verbatim real shape folds, and the Enter/Exit pair disappears.
    #[test]
    fn the_real_inline_monitor_pair_becomes_a_lock() {
        let mut list = vec![
            monitor("Enter", deref(), true),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "FillCJKCore".into(),
                args: vec![v("culture")],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SetCJKReferences".into(),
                args: vec![v("culture")],
            }),
            monitor("Exit", deref(), false),
            Stmt::Return(None),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "the lock must be recognized");
        assert_eq!(
            list.len(),
            2,
            "Enter+body+Exit collapse into one statement + the return"
        );
        match &list[0] {
            Stmt::Lock { expr, body } => {
                assert!(
                    matches!(expr, Expr::RawDeref(..)),
                    "the real target expression"
                );
                assert_eq!(body.len(), 2, "only the protected body is inside");
            }
            other => panic!("expected Stmt::Lock, got {other:?}"),
        }
        assert!(
            matches!(&list[1], Stmt::Return(None)),
            "the return stays outside the lock"
        );
    }

    /// Two sequential locks on the SAME object must pair correctly rather than
    /// have the first swallow the second.
    #[test]
    fn two_locks_on_the_same_object_pair_one_each() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("first")),
            monitor("Exit", v("gate"), false),
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("second")),
            monitor("Exit", v("gate"), false),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 2);
        assert_eq!(list.len(), 2);
        assert!(matches!(&list[0], Stmt::Lock { .. }));
        assert!(matches!(&list[1], Stmt::Lock { .. }));
    }

    /// An inlined cleanup in front of an early `return` is removed and the exit
    /// stays outside the lock.
    #[test]
    fn an_inlined_exit_before_an_early_return_is_removed() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    monitor("Exit", v("gate"), false),
                    Stmt::Return(Some(Expr::Int(1))),
                ],
                else_body: vec![],
            },
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
            Stmt::Return(Some(Expr::Int(2))),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1);
        match &list[0] {
            Stmt::Lock { body, .. } => {
                let Stmt::If { then_body, .. } = &body[0] else {
                    panic!("expected the guarded early exit inside the lock");
                };
                assert!(
                    !then_body.iter().any(|s| matches!(s, Stmt::Eval(_))),
                    "the inlined Exit was removed"
                );
                assert!(matches!(then_body.last(), Some(Stmt::Return(_))));
            }
            other => panic!("expected a lock, got {other:?}"),
        }
    }

    /// A leaving statement with NO inlined cleanup in front of it would GAIN a
    /// release from the fold — refuse.
    #[test]
    fn an_exit_without_a_cleanup_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Return(Some(Expr::Int(1)))],
                else_body: vec![],
            },
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(
            apply(&mut list),
            0,
            "the early return has no cleanup in the flat code"
        );
        assert_eq!(list.len(), 4);
    }

    /// A bare `Exit` that is not an inlined cleanup (nothing leaving follows it)
    /// is not the region's close.
    #[test]
    fn a_bare_exit_mid_region_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
            Stmt::Eval(v("more")),
            monitor("Exit", v("gate"), false),
        ];
        // The first Exit is followed by a statement that does NOT leave the
        // region, so the discipline does not hold.
        assert_eq!(apply(&mut list), 0);
    }

    /// Exit on a DIFFERENT object is a different monitor — never guessed.
    #[test]
    fn exit_on_a_different_object_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("body")),
            monitor("Exit", v("other"), false),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A re-entrant pair on the same object is not one `lock`.
    #[test]
    fn a_nested_same_object_enter_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("body")),
            monitor("Enter", v("gate"), true),
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A one-argument `Enter` with no inlined cleanup proves nothing about the
    /// source: a hand-written `Monitor.Enter(x)` looks identical.
    #[test]
    fn a_bare_enter_without_cleanup_evidence_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(
            apply(&mut list),
            0,
            "no literal lockTaken and no inlined cleanup"
        );
    }

    /// A one-argument `Enter` WITH an inlined cleanup is proof enough.
    #[test]
    fn a_bare_enter_with_cleanup_evidence_folds() {
        let mut list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![monitor("Exit", v("gate"), false), Stmt::Return(None)],
                else_body: vec![],
            },
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
            Stmt::Return(None),
        ];
        assert_eq!(apply(&mut list), 1);
    }

    /// A `break` inside a NESTED loop does not leave the region.
    #[test]
    fn a_break_in_a_nested_loop_needs_no_cleanup() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::While {
                cond: Expr::Cmp(CmpKind::Lt, Box::new(v("i")), Box::new(Expr::Int(4))),
                body: vec![Stmt::Break],
            },
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 1);
        match &list[0] {
            Stmt::Lock { body, .. } => assert!(matches!(body[0], Stmt::While { .. })),
            other => panic!("expected a lock, got {other:?}"),
        }
    }

    /// A `goto` out of the region with the inlined cleanup in front of it is
    /// fine; and the label it targets, being outside, is not moved.
    #[test]
    fn a_goto_out_of_the_region_with_a_cleanup_is_fine() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![monitor("Exit", v("gate"), false), Stmt::Goto("L".into())],
                else_body: vec![],
            },
            monitor("Exit", v("gate"), false),
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        assert_eq!(apply(&mut list), 1);
    }

    /// A label INSIDE the region targeted by a `goto` from OUTSIDE cannot move
    /// into a `lock` body.
    #[test]
    fn a_label_targeted_from_outside_blocks_the_fold() {
        let mut list = vec![
            Stmt::Goto("inner".into()),
            monitor("Enter", v("gate"), true),
            Stmt::Label("inner".into()),
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// The `try/finally` spelling still works where it occurs.
    #[test]
    fn the_try_finally_spelling_is_still_accepted() {
        let mut list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::Try {
                try_body: vec![Stmt::Comment("body".into())],
                catches: vec![],
                finally_body: Some(vec![monitor("Exit", v("gate"), false)]),
            },
        ];
        assert_eq!(apply(&mut list), 1);
        assert!(matches!(&list[0], Stmt::Lock { .. }));
    }

    /// A catch clause means this is a real `try`, not a `lock`.
    #[test]
    fn a_catch_clause_refuses_the_try_spelling() {
        let mut list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::Try {
                try_body: vec![],
                catches: vec![("Exception".into(), vec![])],
                finally_body: Some(vec![monitor("Exit", v("gate"), false)]),
            },
        ];
        assert_eq!(apply(&mut list), 0);
    }

    #[test]
    fn acquire_then_dispose_is_a_using() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Try {
                try_body: vec![Stmt::Comment("body".into())],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("tmp"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1);
        assert_eq!(list.len(), 1);
        match &list[0] {
            Stmt::Using { decl, body } => {
                assert!(matches!(decl.as_ref(), Stmt::Assign(n, None, _) if n == "tmp"));
                assert_eq!(body.len(), 1);
            }
            other => panic!("expected Stmt::Using, got {other:?}"),
        }
    }

    #[test]
    fn using_refused_when_the_resource_is_used_after_the_try() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Try {
                try_body: vec![],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("tmp"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("tmp"))),
                name: "Read".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut list), 0, "a live-out resource is not a `using`");
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn using_refused_when_dispose_is_not_on_the_acquired_var() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Try {
                try_body: vec![],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("other"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
        ];
        assert_eq!(apply(&mut list), 0);
    }

    #[test]
    fn a_non_acquire_assignment_is_not_a_using() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, v("existing")),
            Stmt::Try {
                try_body: vec![],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("tmp"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
        ];
        assert_eq!(apply(&mut list), 0, "a plain copy is not an acquire");
    }

    // -----------------------------------------------------------------------
    // ARM32 patterns: 3-arg Enter and nested Exit
    // -----------------------------------------------------------------------

    /// Helper: 3-arg Monitor.Enter(obj, ptr, &lockTakenVar)
    fn monitor3(member: &str, obj: Expr, ptr: Expr, lt_var: Expr) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("Monitor"))),
            name: member.into(),
            args: vec![obj, ptr, Expr::AddressOf(Box::new(lt_var))],
        })
    }

    /// Helper: Monitor.Exit on a dereferenced lockTaken variable
    fn exit_on_lt(lt_var: Expr) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("Monitor"))),
            name: "Exit".into(),
            args: vec![Expr::RawDeref(Box::new(Expr::AddressOf(Box::new(lt_var))), 0, 0)],
        })
    }

    /// Helper: 2-arg Monitor.Enter with pointer (ARM32 pattern)
    fn monitor_ptr(member: &str, obj: Expr, ptr: Expr) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("Monitor"))),
            name: member.into(),
            args: vec![obj, ptr],
        })
    }

    /// ARM32 3-arg pattern: Enter/Exit in the same list, Exit at top level.
    #[test]
    fn arm32_3arg_enter_exit_same_list() {
        let mut list = vec![
            monitor3("Enter", v("obj"), Expr::RawAddr(Box::new(v("stack")), 4), v("lt")),
            Stmt::Eval(v("body")),
            exit_on_lt(v("lt")),
            Stmt::Return(None),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "3-arg ARM32 lock must be recognized");
        assert!(matches!(&list[0], Stmt::Lock { .. }));
    }

    /// ARM32 3-arg pattern: Exit inside nested if block.
    #[test]
    fn arm32_3arg_exit_inside_nested_if() {
        let mut list = vec![
            monitor3("Enter", v("obj"), Expr::RawAddr(Box::new(v("stack")), 4), v("lt")),
            Stmt::Eval(v("body")),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    exit_on_lt(v("lt")),
                    Stmt::Return(None),
                ],
                else_body: vec![],
            },
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "3-arg ARM32 lock with nested Exit must fold");
        assert!(matches!(&list[0], Stmt::Lock { .. }));
    }

    /// ARM32 2-arg pointer pattern: Exit at top level.
    #[test]
    fn arm32_2arg_ptr_exit_same_list() {
        let mut list = vec![
            monitor_ptr("Enter", v("obj"), Expr::RawAddr(Box::new(v("stack")), 4)),
            Stmt::Eval(v("body")),
            monitor("Exit", v("obj"), false),
            Stmt::Return(None),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "2-arg pointer ARM32 lock must be recognized");
        assert!(matches!(&list[0], Stmt::Lock { .. }));
    }

    /// ARM32 2-arg pointer pattern: Exit inside nested if block.
    #[test]
    fn arm32_2arg_ptr_exit_inside_nested_if() {
        let mut list = vec![
            monitor_ptr("Enter", v("obj"), Expr::RawAddr(Box::new(v("stack")), 4)),
            Stmt::Eval(v("body")),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    monitor("Exit", v("obj"), false),
                    Stmt::Return(None),
                ],
                else_body: vec![],
            },
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "2-arg pointer ARM32 lock with nested Exit must fold");
        assert!(matches!(&list[0], Stmt::Lock { .. }));
    }

    /// ARM32 3-arg: Exit on a DIFFERENT lockTaken variable is refused.
    #[test]
    fn arm32_3arg_exit_on_different_lt_var_refused() {
        let mut list = vec![
            monitor3("Enter", v("obj"), Expr::RawAddr(Box::new(v("stack")), 4), v("lt")),
            Stmt::Eval(v("body")),
            exit_on_lt(v("other_lt")),  // different variable!
            Stmt::Return(None),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 0, "Exit on different lockTaken var must not fold");
    }

    // -----------------------------------------------------------------------
    // Flat using pattern (no try/finally)
    // -----------------------------------------------------------------------

    /// The simple flat using shape: acquire, body, Dispose → using statement.
    #[test]
    fn flat_using_simple_pattern() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r"))),
                name: "Read".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r"))),
                name: "Dispose".into(),
                args: vec![],
            }),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "flat using must be recognized");
        assert_eq!(list.len(), 1);
        match &list[0] {
            Stmt::Using { decl, body } => {
                assert!(matches!(decl.as_ref(), Stmt::Assign(n, None, _) if n == "r"));
                assert_eq!(body.len(), 1, "one body statement inside the using");
            }
            other => panic!("expected Stmt::Using, got {other:?}"),
        }
    }

    /// Flat using refused when the variable is read after the Dispose — a
    /// live-out resource is an ordinary local, not a scoped `using`.
    #[test]
    fn flat_using_refused_when_var_used_after_dispose() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r"))),
                name: "Read".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r"))),
                name: "Dispose".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r"))),
                name: "ToString".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut list), 0, "live-out resource is not a using");
    }

    /// Flat using refused when the variable is reassigned before Dispose.
    #[test]
    fn flat_using_refused_when_var_reassigned() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Assign("r".into(), None, Expr::NewObj("OtherReader".into(), vec![])),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r"))),
                name: "Dispose".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut list), 0, "reassignment breaks the link");
    }

    /// Flat using refused when an early return sits between acquire and Dispose
    /// — without a finally, that path would skip the disposal.
    #[test]
    fn flat_using_refused_with_early_return() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Return(None),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r"))),
                name: "Dispose".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut list), 0, "early return skips disposal");
    }

    /// Two sequential flat usings are both folded (the loop in rewrite_list
    /// rescans after each rewrite).
    #[test]
    fn flat_using_two_sequential_disposables() {
        let mut list = vec![
            Stmt::Assign("r1".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Eval(v("use1")),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r1"))),
                name: "Dispose".into(),
                args: vec![],
            }),
            Stmt::Assign("r2".into(), None, Expr::NewObj("Writer".into(), vec![])),
            Stmt::Eval(v("use2")),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r2"))),
                name: "Dispose".into(),
                args: vec![],
            }),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 2, "both sequential usings must fold");
        assert_eq!(list.len(), 2);
        assert!(matches!(&list[0], Stmt::Using { .. }));
        assert!(matches!(&list[1], Stmt::Using { .. }));
    }

    /// Flat using refused when Dispose targets a different variable.
    #[test]
    fn flat_using_refused_when_dispose_on_different_var() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Eval(v("body")),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("other"))),
                name: "Dispose".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut list), 0, "Dispose on different var is not a match");
    }

    /// Flat using refused when the acquire is a plain variable copy (not a
    /// disposable construction).
    #[test]
    fn flat_using_refused_non_disposable_acquire() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, v("existing")),
            Stmt::Eval(v("body")),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("r"))),
                name: "Dispose".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut list), 0, "a plain copy is not a disposable acquire");
    }

    // -----------------------------------------------------------------------
    // Cross-list lock pairing (via the fall-through merge pre-pass)
    // -----------------------------------------------------------------------

    /// Lock where Enter is in an if-then body and Exit is in the parent list
    /// after the if. The fall-through merge pre-pass splices them into one list
    /// before the lock fold runs.
    #[test]
    fn cross_list_lock_via_fallthrough_merge() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    monitor("Enter", v("gate"), true),
                    Stmt::Eval(v("body")),
                ],
                else_body: vec![Stmt::Return(None)],
            },
            Stmt::Label("L_exit".into()),
            monitor("Exit", v("gate"), false),
            Stmt::Return(None),
        ];
        apply(&mut stmts);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(
                    then_body.iter().any(|s| matches!(s, Stmt::Lock { .. })),
                    "cross-list lock must fold after fall-through merge"
                );
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// Flat using where the Dispose call uses the receiver-in-name spelling
    /// (`r.Dispose()` with no separate receiver).
    #[test]
    fn flat_using_with_receiver_in_name() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Eval(v("body")),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "r.Dispose".into(),
                args: vec![],
            }),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "receiver-in-name Dispose must be recognized");
        assert!(matches!(&list[0], Stmt::Using { .. }));
    }

    // -----------------------------------------------------------------------
    // Flat-using soundness: nested exits and reassignments
    // -----------------------------------------------------------------------

    fn dispose_r() -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("r"))),
            name: "Dispose".into(),
            args: vec![],
        })
    }

    /// A `return` inside an `if` between the acquire and the flat Dispose is an
    /// exit path on which the FLAT code does not dispose. A `using` would
    /// dispose there, so the fold must be refused.
    #[test]
    fn flat_using_refused_with_a_nested_return() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::If {
                cond: v("c"),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![],
            },
            dispose_r(),
        ];
        assert_eq!(
            apply(&mut list),
            0,
            "the early return would newly dispose under `using`"
        );
    }

    /// A `return` inside a NESTED loop is still an exit from the method.
    #[test]
    fn flat_using_refused_with_a_return_inside_a_loop() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![Stmt::Return(None)],
            },
            dispose_r(),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A nested reassignment changes which object the flat Dispose targets;
    /// wrapping the original acquire in `using` would dispose the wrong one.
    #[test]
    fn flat_using_refused_with_a_nested_reassignment() {
        let mut list = vec![
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::If {
                cond: v("c"),
                then_body: vec![Stmt::Assign(
                    "r".into(),
                    None,
                    Expr::NewObj("OtherReader".into(), vec![]),
                )],
                else_body: vec![],
            },
            dispose_r(),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A label in the prospective `using` body that a `goto` from before the
    /// acquire targets cannot move into the `using` (C# forbids jumping in).
    #[test]
    fn flat_using_refused_when_a_goto_jumps_into_the_body() {
        let mut list = vec![
            Stmt::Goto("inside".into()),
            Stmt::Assign("r".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Label("inside".into()),
            Stmt::Eval(v("body")),
            dispose_r(),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A reassignment in the `try` body of a recovered try/finally would make
    /// the `using` declaration non-readonly and dispose the wrong value.
    #[test]
    fn using_refused_when_the_try_body_reassigns_the_var() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Try {
                try_body: vec![Stmt::Assign(
                    "tmp".into(),
                    None,
                    Expr::NewObj("Other".into(), vec![]),
                )],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("tmp"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A label in a `try` body targeted from outside blocks the `lock` fold.
    #[test]
    fn try_lock_refused_when_a_goto_jumps_into_the_try() {
        let mut list = vec![
            Stmt::Goto("inside".into()),
            monitor("Enter", v("gate"), false),
            Stmt::Try {
                try_body: vec![Stmt::Label("inside".into()), Stmt::Eval(v("body"))],
                catches: vec![],
                finally_body: Some(vec![monitor("Exit", v("gate"), false)]),
            },
        ];
        assert_eq!(apply(&mut list), 0);
    }

    // -----------------------------------------------------------------------
    // Nested-loop cleanup discipline
    // -----------------------------------------------------------------------

    /// An `Exit` inside a nested loop that is merely followed by more loop
    /// body is NOT an inlined finally: removing it would release the lock for
    /// the next iteration while the flat code held it only per iteration.
    #[test]
    fn an_exit_inside_a_loop_without_a_following_exit_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![monitor("Exit", v("gate"), false), Stmt::Eval(v("work"))],
            },
            Stmt::Eval(v("body")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A `return` inside a nested loop with no `Exit` in front of it would
    /// gain a release from the fold — refuse.
    #[test]
    fn a_return_inside_a_loop_without_a_cleanup_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![Stmt::If {
                    cond: v("c"),
                    then_body: vec![Stmt::Return(None)],
                    else_body: vec![],
                }],
            },
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A `return` inside a nested loop WITH its inlined cleanup in front is a
    /// valid early-exit path; the lock keeps the return and drops the Exit.
    #[test]
    fn an_exit_before_a_return_inside_a_loop_folds() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![
                    Stmt::If {
                        cond: v("c"),
                        then_body: vec![monitor("Exit", v("gate"), false), Stmt::Return(None)],
                        else_body: vec![],
                    },
                    Stmt::Eval(v("work")),
                ],
            },
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 1);
        match &list[0] {
            Stmt::Lock { body, .. } => {
                let Stmt::While { body: loop_body, .. } = &body[0] else {
                    panic!("expected the loop inside the lock");
                };
                let Stmt::If { then_body, .. } = &loop_body[0] else {
                    panic!("expected the guarded early exit");
                };
                assert!(
                    !then_body.iter().any(|s| matches!(s, Stmt::Eval(_))),
                    "the inlined Exit was removed"
                );
                assert!(matches!(then_body.last(), Some(Stmt::Return(_))));
            }
            other => panic!("expected a lock, got {other:?}"),
        }
    }
}
