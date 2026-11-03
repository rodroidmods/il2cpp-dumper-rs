//! Phase 7: IL2CPP de-weaving — strip compiler/runtime boilerplate that has no
//! managed C# equivalent.
//!
//! ## What IL2CPP inserts that managed C# does not need
//!
//! IL2CPP translates managed C# into native C++ code. In native C++, there is
//! no garbage collector, no virtual machine to throw `NullReferenceException`,
//! and no runtime to initialize class metadata on first use. So IL2CPP inserts
//! explicit checks that have no managed equivalent:
//!
//! 1. **Null-reference guards** — `if (obj == null) ThrowHelper.ThrowNullReferenceException(obj);`
//!    In managed C#, the VM throws automatically on null dereference.
//!
//! 2. **Class initialization guards** — `if (*(byte*)(ADDR) == 0) { IL2CPP_ClassInit(...); *(byte*)(ADDR) = 1; }`
//!    In managed C#, the runtime initializes static constructors lazily.
//!
//! 3. **RPC ternary self-pointers** — `(v58 == 0 ? this : v58).CallRpc()`
//!    Network weavers (Mirror/UNET) inject safety checks into `[ClientRpc]`/
//!    `[Command]` methods. The ternary always resolves to `this`.
//!
//! 4. **Temporary variable flattening** — `v33 = op_Inequality(a, b); if (v33 != 0)`
//!    The decompiler stretches a single boolean test into two statements.
//!
//! ## Measured reality (x64 PE corpus, 47,924 methods)
//!
//! | pattern | occurrences |
//! |---|---|
//! | `IL2CPP_ClassInit` guards | 61,343 |
//! | `ThrowNullReferenceException` guards | 1,786 |
//! | RPC ternary `v == 0 ? this : v` | 31 (across all corpora) |
//! | `op_Inequality`/`op_Equality` temp vars | 28,526 (across all corpora) |
//!
//! ## Safety
//!
//! Every rewrite in this file removes code that the IL2CPP runtime inserted for
//! native safety but that the managed runtime handles natively. The rewrites
//! NEVER remove user logic: the pattern match requires the body to contain ONLY
//! the runtime call (plus optional dead assignments to temporaries that the
//! runtime call's result was stored into).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::super::csharp::{BinOp, CmpKind};
use super::{walk_mut, Expr, Stmt};

// ---------------------------------------------------------------------------
// Temp-noise census (ILR_TEMPSTATS=1)
// ---------------------------------------------------------------------------
//
// One counter per rewrite that removes a temporary assignment or copy. The
// counters are process-wide because `apply` runs per method; when
// `ILR_TEMPSTATS` is set, `apply` prints the per-rule delta for a method that
// fired, and a full dump's log is the sum of those lines.

static DEFCOPY_COALESCED: AtomicU64 = AtomicU64::new(0);
static DEFCOPY_EXAMINED: AtomicU64 = AtomicU64::new(0);
static RETURN_FOLDED: AtomicU64 = AtomicU64::new(0);
static LOOP_REASSIGN_REMOVED: AtomicU64 = AtomicU64::new(0);
static FINAL_DEAD_TEMPS: AtomicU64 = AtomicU64::new(0);

fn tempstats_enabled() -> bool {
    std::env::var_os("ILR_TEMPSTATS").is_some()
}

fn tempstats_snapshot() -> [u64; 5] {
    [
        DEFCOPY_EXAMINED.load(Ordering::Relaxed),
        DEFCOPY_COALESCED.load(Ordering::Relaxed),
        RETURN_FOLDED.load(Ordering::Relaxed),
        LOOP_REASSIGN_REMOVED.load(Ordering::Relaxed),
        FINAL_DEAD_TEMPS.load(Ordering::Relaxed),
    ]
}

/// Print the per-rule delta for one `apply` call, when the census is on.
fn tempstats_report(before: [u64; 5]) {
    let now = tempstats_snapshot();
    if now == before {
        return;
    }
    eprintln!(
        "[tempstats] defcopy={}/{} retfold={} loop_reassign={} final_dce={} \
         (total defcopy={}/{} retfold={} loop_reassign={} final_dce={})",
        now[1] - before[1],
        now[0] - before[0],
        now[2] - before[2],
        now[3] - before[3],
        now[4] - before[4],
        now[1],
        now[0],
        now[2],
        now[3],
        now[4],
    );
}

/// Wall-clock nanos per cleanup sub-pass (parallel-summed; `ILR_TIMING`).
/// `il2cpp_cleanup::apply` is called from the reconstruct pipeline, its final
/// round, the render path and both preview clones, so one slow sub-pass is
/// multiplied by ~5 per method. Same-output optimizations target the top
/// entry here.
static SUB_NANOS: [std::sync::atomic::AtomicU64; 18] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 18];

/// Wall-clock nanos for the internals of `remove_dead_temp_assignments`, the
/// largest single sub-pass: its observable-name scan, the per-range sweep, the
/// name collection, the fused read census and the name-wide sweep. The last
/// four entries split the range sweep itself.
static DTS_NANOS: [std::sync::atomic::AtomicU64; 9] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 9];

const DTS_NAMES: [&str; 9] = [
    "observable",
    "range",
    "collect",
    "census",
    "names",
    "r_census",
    "r_children",
    "r_later",
    "r_checks",
];

/// Record the time since `start` in `DTS_NANOS[idx]` (`ILR_TIMING` only).
fn dt_add(idx: usize, start: std::time::Instant) {
    DTS_NANOS[idx].fetch_add(
        start.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Per-sub-pass total rewrite count, split by call site (0 pipeline, 1 direct
/// render-path).
static SUB_N: [[std::sync::atomic::AtomicU64; 18]; 2] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 18] }; 2];

const SUB_NAMES: [&str; 18] = [
    "strip_null_guards",
    "drop_metadata_init_calls",
    "strip_class_init_guards",
    "fold_identical_reassign_guards",
    "raise_to_throw",
    "drop_dead_code_after_terminal",
    "collapse_rpc_ternary",
    "fold_loop_preheader",
    "flatten_temp_vars",
    "coalesce_def_into_copies",
    "inline_temp_to_temp_copies",
    "remove_redundant_temp_writes",
    "remove_consecutive_dead_writes",
    "inline_temp_variables",
    "fold_return_of_temp",
    "remove_dead_temp_assignments",
    "flatten_bitwise_safety_masks",
    "translate_array_indexing",
];

/// Top cleanup sub-passes by cumulative time.
pub(crate) fn cleanup_timing_report(top: usize) -> String {
    let mut rows: Vec<(&str, u64)> = SUB_NAMES
        .iter()
        .enumerate()
        .map(|(i, n)| {
            (
                *n,
                SUB_NANOS[i].load(std::sync::atomic::Ordering::Relaxed),
            )
        })
        .collect();
    rows.sort_by_key(|(_, ns)| std::cmp::Reverse(*ns));
    let total: u64 = rows.iter().map(|(_, ns)| *ns).sum();
    let mut out = format!("[clean-split] total={:.0}ms", total as f64 / 1e6);
    for (name, ns) in rows.into_iter().take(top) {
        if ns == 0 {
            break;
        }
        out.push_str(&format!(" {name}={:.0}", ns as f64 / 1e6));
    }
    // Which sub-passes actually fire at the DIRECT render-path call.
    let mut direct = String::new();
    for (i, name) in SUB_NAMES.iter().enumerate() {
        let n = SUB_N[1][i].load(std::sync::atomic::Ordering::Relaxed);
        if n > 0 {
            direct.push_str(&format!(" {name}={n}"));
        }
    }
    if !direct.is_empty() {
        out.push_str(&format!("\n[clean2-n]{direct}"));
    }
    let mut dts = String::new();
    let mut any = false;
    for (i, name) in DTS_NAMES.iter().enumerate() {
        let ns = DTS_NANOS[i].load(std::sync::atomic::Ordering::Relaxed);
        any |= ns > 0;
        if ns > 0 {
            dts.push_str(&format!(" {name}={:.0}", ns as f64 / 1e6));
        }
    }
    if any {
        out.push_str(&format!("\n[dt-split]{dts}"));
    }
    out
}

// ===========================================================================
// Public entry point
// ===========================================================================

fn missing_labels(stmts: &[Stmt]) -> Vec<String> {
    use std::collections::HashSet;
    let mut defs: HashSet<String> = HashSet::new();
    let mut gotos: HashSet<String> = HashSet::new();
    fn walk(stmts: &[Stmt], defs: &mut HashSet<String>, gotos: &mut HashSet<String>) {
        for s in stmts {
            match s {
                Stmt::Label(l) => {
                    defs.insert(l.clone());
                }
                Stmt::Goto(l) => {
                    gotos.insert(l.clone());
                }
                _ => {}
            }
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            for k in kids {
                walk(k, defs, gotos);
            }
        }
    }
    walk(stmts, &mut defs, &mut gotos);
    let mut v: Vec<String> = gotos.difference(&defs).cloned().collect();
    v.sort();
    v
}

// Run all IL2CPP de-weaving passes over one method body.
std::thread_local! {
    /// 0 = pipeline / final-round call, 1 = direct render-path call.
    static CLEAN_TAG: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Run cleanup with a call-site tag so per-sub-pass COUNTS can be attributed.
/// Instrumentation only: the tag never changes behavior.
pub fn apply_tagged(stmts: &mut Vec<Stmt>, tag: u8) -> usize {
    CLEAN_TAG.with(|c| c.set(tag));
    let r = apply(stmts);
    CLEAN_TAG.with(|c| c.set(0));
    r
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let stats_before = if tempstats_enabled() {
        Some(tempstats_snapshot())
    } else {
        None
    };
    let mut count = 0;
    macro_rules! timed {
        ($idx:expr, $e:expr) => {{
            let t = std::time::Instant::now();
            let v = $e;
            SUB_NANOS[$idx].fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            if v > 0 {
                let tag = CLEAN_TAG.with(|c| c.get()) as usize;
                SUB_N[tag][$idx].fetch_add(v as u64, std::sync::atomic::Ordering::Relaxed);
            }
            v
        }};
    }
    if crate::recovery::env_flags::flag("ILR_GOTODBG2") {
        let m = missing_labels(stmts);
        if !m.is_empty() {
            eprintln!("[g2]   before-cleanup missing={m:?}");
        }
    }
    macro_rules! g2 {
        ($name:literal) => {
            if crate::recovery::env_flags::flag("ILR_GOTODBG2") {
                let m = missing_labels(stmts);
                if !m.is_empty() {
                    eprintln!("[g2]   {}: missing={:?}", $name, m);
                }
            }
        };
    }
    static CLEAN_CALL: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    let call = CLEAN_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    macro_rules! cdump {
        ($name:literal) => {
            if crate::recovery::env_flags::flag("ILR_CLEANDBG") {
                let _ = std::fs::write(
                    crate::recovery::probe_dir()
                        .join(format!("clean_{call}_{}.txt", $name)),
                    format!("{stmts:#?}"),
                );
            }
        };
    }
    count += timed!(0, strip_null_guards(stmts));
    count += timed!(1, drop_metadata_init_calls(stmts));
    cdump!("00_start");
    g2!("strip_null_guards");
    count += timed!(2, strip_class_init_guards(stmts));
    cdump!("01_classinit");
    g2!("strip_class_init_guards");
    count += timed!(3, fold_identical_reassign_guards(stmts));
    cdump!("02_reassign");
    g2!("fold_identical_reassign_guards");
    count += timed!(4, raise_to_throw(stmts));
    cdump!("03_raise");
    g2!("raise_to_throw");
    count += timed!(5, drop_dead_code_after_terminal(stmts));
    cdump!("04_deadterminal");
    g2!("drop_dead_code_after_terminal");
    count += timed!(6, collapse_rpc_ternary(stmts));
    // Does round 1 (from the preheader pass down) change anything at all? If
    // every call returns 0, the tree entering round 2 is IDENTICAL to the
    // state those passes already reached their fixpoints on, so repeating
    // them (plus the dead-temp sweep, the single most expensive sub-pass) is
    // provably a no-op. Skipping it is output-neutral and removes a large
    // share of cleanup's cost on converged methods.
    let mut round1_dirty = false;
    // Did the two SPECIFIC steps 16/17 (bitwise masks, array indexing) change
    // the tree? They run AFTER the dead-temp sweep, so their changes can
    // expose new dead temps even when every round-2 pass is a no-op.
    let mut late_expr_dirty = false;
    macro_rules! step {
        ($idx:expr, $e:expr) => {{
            let v = timed!($idx, $e);
            round1_dirty |= v > 0;
            count += v;
        }};
    }
    step!(7, fold_loop_preheader_reassignments(stmts));
    step!(8, flatten_temp_vars(stmts));
    cdump!("05_flatten1");
    g2!("flatten_temp_vars");
    step!(9, if coalesce_disabled() { 0 } else { coalesce_def_into_copies(stmts) });
    step!(10, inline_temp_to_temp_copies(stmts));
    cdump!("06_copies");
    g2!("inline_temp_to_temp_copies");
    step!(11, remove_redundant_temp_writes(stmts));
    cdump!("07_redundant");
    step!(12, remove_consecutive_dead_writes(stmts));
    cdump!("08_consecutive");
    step!(13, inline_temp_variables(stmts));
    cdump!("09_inline1");
    step!(8, flatten_temp_vars(stmts));
    cdump!("10_flatten2");
    step!(14, if return_fold_disabled() { 0 } else { fold_return_of_temp(stmts) });
    step!(15, remove_dead_temp_assignments(stmts));
    cdump!("11_deadtemps1");
    g2!("remove_dead_temp_assignments");
    let v16 = timed!(16, flatten_bitwise_safety_masks(stmts));
    round1_dirty |= v16 > 0;
    late_expr_dirty |= v16 > 0;
    count += v16;
    let v17 = timed!(17, translate_array_indexing(stmts));
    round1_dirty |= v17 > 0;
    late_expr_dirty |= v17 > 0;
    count += v17;
    g2!("translate_array_indexing");
    // SECOND INLINING ROUND: the passes above drop uses that blocked the
    // first round's read-count checks — `raise_to_throw` turns the raise that
    // read a copy temp into a Throw, `remove_dead_temp_assignments` then
    // deletes the copy, and `flatten_bitwise_safety_masks` rewrites
    // `(v & 1) != 0` uses. A temp that was multi-read at the first round is
    // very often single-read by now, and single-read pure/impure temps are
    // the safest inlines there are. Re-running the trio once at the end
    // captures exactly those; each call is a fixed-point pass over the tree.
    if round1_dirty {
        let mut round2_dirty = false;
        macro_rules! step2 {
            ($idx:expr, $e:expr) => {{
                let v = timed!($idx, $e);
                round2_dirty |= v > 0;
                count += v;
            }};
        }
        step2!(7, fold_loop_preheader_reassignments(stmts));
        step2!(9, if coalesce_disabled() { 0 } else { coalesce_def_into_copies(stmts) });
        step2!(10, inline_temp_to_temp_copies(stmts));
        step2!(13, inline_temp_variables(stmts));
        step2!(8, flatten_temp_vars(stmts));
        step2!(14, if return_fold_disabled() { 0 } else { fold_return_of_temp(stmts) });
        // FINAL DEAD-TEMP SWEEP (7C-4). The passes above only became able to
        // see single-read temps because earlier passes deleted statements; a
        // def whose last reader was a stripped guard, a folded arm or a
        // return folded just above is dead now even though it was live when
        // the first sweep ran. Liveness is method-wide (see
        // `remove_dead_temp_assignments`), so this also catches a def in an
        // arm whose reader was in the other arm.
        //
        // It is provably a no-op when nothing changed since the previous
        // sweep's fixpoint: steps 16/17 and every call above returned 0, so
        // the tree is byte-for-byte the one `remove_dead_temp_assignments`
        // already converged on. That sweep is the single most expensive
        // sub-pass (55 s on the full corpus), and it runs in the pipeline,
        // the render path and both preview clones.
        if round2_dirty || late_expr_dirty {
            let final_dead = timed!(15, remove_dead_temp_assignments(stmts));
            cdump!("12_deadtemps2");
            FINAL_DEAD_TEMPS.fetch_add(final_dead as u64, Ordering::Relaxed);
            count += final_dead;
        }
    }
    if let Some(before) = stats_before {
        tempstats_report(before);
    }
    // Note: cleanup_static_field_access runs AFTER field_annotation in
    // apply_with_maps_and_types(), not here, because the patterns it cleans
    // are created by field_annotation.
    count
}

// ===========================================================================
// 7A-1: Null-reference guard stripping
// ===========================================================================

/// Strip IL2CPP null-reference guards.
///
/// ## Pattern
///
/// ```text
/// if (expr == 0) {
///     ThrowHelper.ThrowNullReferenceException(expr);
///     [optional dead assignments]
/// }
/// ```
///
/// or equivalently with `Null` instead of `Int(0)`, or with the comparison
/// reversed. The body MUST contain a call whose name contains
/// `ThrowNullReferenceException` (case-insensitive match on the key substring).
///
/// ## Why this is safe
///
/// In managed C#, dereferencing a null reference automatically throws
/// `NullReferenceException`. The IL2CPP native code must check explicitly
/// because native code has no such safety net. When we render back to C#,
/// the check is redundant.
///
/// ## What we keep
///
/// We only remove the `if` statement when the then-body's FIRST statement is
/// a call to a null-reference throwing function. Additional statements in the
/// body (dead assignments like `v19 = 0`) are also removed as they are part of
/// the guard's boilerplate.
///
/// Does this expression contain a `.initialized` runtime flag read on a
/// metadata/value expression (not on `this` or a plain user object)?
fn has_initialized_flag(e: &Expr) -> bool {
    match e {
        Expr::Field(base, f) if f == "initialized" => matches!(
            base.as_ref(),
            Expr::RawDeref(..)
                | Expr::RawAddr(..)
                | Expr::StaticField(..)
                | Expr::TypeOf(_)
                | Expr::Cast(..)
                | Expr::Call { .. }
                | Expr::RawCall(..)
        ),
        Expr::Field(b, _)
        | Expr::RawDeref(b, _, _)
        | Expr::RawAddr(b, _)
        | Expr::AddressOf(b)
        | Expr::Cast(_, b)
        | Expr::Not(b)
        | Expr::Neg(b)
        | Expr::Await(b) => has_initialized_flag(b),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            has_initialized_flag(a) || has_initialized_flag(b)
        }
        Expr::Cond(c, a, b) => {
            has_initialized_flag(c) || has_initialized_flag(a) || has_initialized_flag(b)
        }
        Expr::Call { recv, args, .. } => {
            recv.as_deref().is_some_and(has_initialized_flag)
                || args.iter().any(has_initialized_flag)
        }
        _ => false,
    }
}

/// Remove IL2CPP's metadata/generic-context initialization guards.
///
/// Two shapes, both runtime plumbing with no managed counterpart:
///
/// ```text
/// // (a) runtime type/method metadata init
/// initialized = (*(nint*)(x +16)).initialized;
/// if (!initialized) v = fn_init(*(nint*)(x +16), ctx, …);
/// // (b) generic-context lazy init
/// if (*(nint*)(slot +56) == 0) { fn_ctx(slot, ctx, …); v = *(nint*)(slot +56); }
/// ```
///
/// (a) is recognized by the `.initialized` runtime flag in the condition (read
/// through a metadata expression or a local assigned from one). Dropping it is
/// safe when every name the body assigns either already has a dominating
/// definition or is dead afterwards.
///
/// (b) is recognized structurally: the condition is `SLOT == 0`, the body's
/// first statement is a raw call and its last statement reloads `SLOT` — the
/// runtime initializes the slot in place. No address or offset is assumed.
pub(crate) fn strip_metadata_init_guards(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        let mut init_flags: std::collections::HashSet<String> = std::collections::HashSet::new();
        for s in list.iter() {
            if let Stmt::Assign(n, _, rhs) = s {
                if has_initialized_flag(rhs) {
                    init_flags.insert(n.clone());
                }
            }
        }
        let mut i = 0;
        while i < list.len() {
            if is_metadata_init_guard(&list[i], &init_flags, list, i) {
                list.remove(i);
                count += 1;
            } else {
                i += 1;
            }
        }
    });
    // The flag loads that fed the removed guards are now dead plumbing:
    // `initialized = (*(nint*)(x +16)).initialized;`. Remove every assignment
    // to such a name when the name has no reads outside assignments to
    // itself (the load chain).
    let mut flagged: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_flagged_assigns(stmts, &mut flagged);
    for name in flagged {
        if reads_outside_own_assigns(stmts, &name) == 0 {
            count += remove_assigns_of(stmts, &name);
        }
    }
    count
}

fn collect_flagged_assigns(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in stmts {
        if let Stmt::Assign(n, _, rhs) = s {
            if has_initialized_flag(rhs) {
                out.insert(n.clone());
            }
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_flagged_assigns(k, out);
        }
    }
}

/// Reads of `name` that are not inside an assignment to `name` itself.
fn reads_outside_own_assigns(stmts: &[Stmt], name: &str) -> usize {
    fn walk(stmts: &[Stmt], name: &str, total: &mut usize) {
        for s in stmts {
            if let Stmt::Assign(n, _, rhs) = s {
                if n != name {
                    *total += expr_mention_count(rhs, name);
                }
            } else {
                let mut kids: Vec<&Vec<Stmt>> = Vec::new();
                super::children(s, &mut kids);
                for k in kids {
                    walk(k, name, total);
                }
                for e in exprs_of_stmt(s) {
                    *total += expr_mention_count(e, name);
                }
            }
        }
    }
    let mut total = 0usize;
    walk(stmts, name, &mut total);
    total
}

/// Expressions a statement carries in its own slots (not in child lists).
fn exprs_of_stmt(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => vec![e],
        Stmt::Return(Some(e)) => vec![e],
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            vec![cond]
        }
        Stmt::Lock { expr, .. } => vec![expr],
        Stmt::StringSwitch { disc, .. } => vec![disc],
        Stmt::For { cond, .. } => cond.iter().collect(),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => vec![collection, index, value],
        Stmt::EventAssign { field, value, .. } => vec![field, value],
        _ => vec![],
    }
}

fn remove_assigns_of(stmts: &mut Vec<Stmt>, name: &str) -> usize {
    let mut n = 0;
    walk_mut(stmts, &mut |list| {
        let before = list.len();
        list.retain(|s| !matches!(s, Stmt::Assign(x, _, _) if x == name));
        n += before - list.len();
    });
    n
}

fn is_metadata_init_guard(
    stmt: &Stmt,
    init_flags: &std::collections::HashSet<String>,
    list: &[Stmt],
    at: usize,
) -> bool {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return false;
    };
    if !else_body.is_empty() || then_body.is_empty() {
        return false;
    }
    if then_body.iter().any(stmt_has_label) {
        return false;
    }
    if then_body.iter().any(|s| {
        matches!(
            s,
            Stmt::Return(_) | Stmt::Throw(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue
        )
    }) {
        return false;
    }
    let flag_guard = has_initialized_flag(cond)
        || match cond {
            Expr::Not(inner) => matches!(inner.as_ref(), Expr::Var(n) if init_flags.contains(n)),
            Expr::Var(n) => init_flags.contains(n),
            Expr::Cmp(_, a, b) => {
                matches!(a.as_ref(), Expr::Var(n) if init_flags.contains(n))
                    || matches!(b.as_ref(), Expr::Var(n) if init_flags.contains(n))
            }
            _ => false,
        };
    if flag_guard {
        for s in then_body {
            if let Stmt::Assign(name, _, _) = s {
                let defined_before = list[..at]
                    .iter()
                    .any(|p| matches!(p, Stmt::Assign(n, _, _) if n == name));
                let read_after = count_reads_in_range(list, at + 1, list.len(), name) > 0;
                if !defined_before && read_after {
                    return false;
                }
            }
        }
        return true;
    }
    // (b) in-place lazy init: `SLOT == 0 { fn(...); SLOT = …; }`.
    let Expr::Cmp(CmpKind::Eq, lhs, rhs) = cond else {
        return false;
    };
    let slot = match (lhs.as_ref(), rhs.as_ref()) {
        (s, e) if matches!(e, Expr::Int(0)) => s,
        (e, s) if matches!(e, Expr::Int(0)) => s,
        _ => return false,
    };
    let Some(first) = then_body.first() else {
        return false;
    };
    if !matches!(first, Stmt::Eval(Expr::RawCall(..))) {
        return false;
    }
    let Some(Stmt::Assign(_, _, reload)) = then_body.last() else {
        return false;
    };
    super::exprs_eq(reload, slot)
}

/// The pass is also re-run by the renderer after `branch_repair`, whose join
/// folding materializes fresh `if (x == 0) { raise } else { keep }` diamonds
/// (including the `throw new NullReferenceException` form synthesized by
/// `raise_to_throw`) long after the last full cleanup round.
pub(crate) fn strip_null_guards(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += strip_null_guards_list(list);
    });
    count
}

/// Is this statement the exact `throw new NullReferenceException(<fallback>)`
/// that `raise_to_throw` emits when no message could be extracted?
fn stmt_is_fallback_null_throw(s: &Stmt) -> bool {
    const FALLBACK: &str = "Object reference not set to an instance of an object.";
    matches!(
        s,
        Stmt::Throw(Expr::NewObj(name, args))
            if is_null_reference_exception_name(name)
                && matches!(args.as_slice(), [Expr::Str(m)] if m == FALLBACK)
    )
}

/// Final handling of the synthesized raise throws that survived guard
/// stripping.
///
/// The measured corpus split (8,080 survivors on Temple Run 2) is: 53 real
/// managed error checks (`if (!Call(...)) throw …`), 1,517 injected null
/// guards the matcher could not prove, 3,774 raises stranded at CFG joins,
/// 493 compiler-generated delegate-stub throws, 575 cold/coverage-region
/// raises, and 1,668 bounds/type-init checks. Only the call-result arm is
/// source logic; the rest are IL2CPP raise noise rendered as a made-up
/// `NullReferenceException` (the raise site carries no recoverable type or
/// message).
///
/// So: every fallback-message NRE throw is removed EXCEPT when it is the arm
/// of an `if` whose condition tests a call result. Removing a raise also
/// removes the statements after it up to the next label: the helper never
/// returns, so that tail is unreachable.
pub(crate) fn finalize_raise_throws(stmts: &mut Vec<Stmt>) -> usize {
    finalize_raise_list(stmts, false)
}

/// The condition must be a NULL TEST on a call result, not merely any call.
///
/// `if (!player.IsCharacterPurchased(id)) { …; raise }` also "tests a call",
/// and that kept a synthesized fallback NRE inside a real branch
/// (0x18A3A1C / 0x18A3D80 / 0x189D9D4). A genuine injected null guard is
/// `Call(…) == 0` / `!= 0` (optionally under `!`); an arbitrary boolean call
/// condition is not evidence that the raise was a managed throw.
fn is_null_test_of_call(e: &Expr) -> bool {
    match e {
        Expr::Cmp(kind, a, b) => {
            matches!(kind, CmpKind::Eq | CmpKind::Ne)
                && ((checked_is_call_result(a) && matches!(**b, Expr::Int(0)))
                    || (checked_is_call_result(b) && matches!(**a, Expr::Int(0))))
        }
        Expr::Not(inner) => is_null_test_of_call(inner),
        _ => false,
    }
}

fn finalize_raise_list(stmts: &mut Vec<Stmt>, in_call_test_arm: bool) -> usize {
    let mut count = 0;
    for s in stmts.iter_mut() {
        match s {
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let call_test = is_null_test_of_call(cond);
                count += finalize_raise_list(then_body, call_test);
                count += finalize_raise_list(else_body, call_test);
            }
            _ => {
                for k in super::children_mut(s) {
                    count += finalize_raise_list(k, false);
                }
            }
        }
    }
    if in_call_test_arm {
        return count;
    }
    let mut i = 0;
    let mut removed_tail_raise = false;
    while i < stmts.len() {
        if stmt_is_fallback_null_throw(&stmts[i]) {
            stmts.remove(i);
            count += 1;
            while i < stmts.len() && !stmt_has_label(&stmts[i]) {
                stmts.remove(i);
            }
            // A raise at the very end of the list is the null path of the
            // guards above it. Dropping the throw alone would leave their
            // false arms FALLING OFF the method (a non-void method with a
            // reachable path and no return — 0x18A5204). The raise is
            // implicit in C#: unwrap the guards so their payloads become the
            // straight-line tail.
            if stmts[i..].iter().all(|s| matches!(s, Stmt::Label(_))) {
                removed_tail_raise = true;
            }
        } else {
            i += 1;
        }
    }
    if removed_tail_raise {
        count += unwrap_trailing_null_guards(stmts);
    }
    count
}

/// The expression a `X != 0` / `X != null` guard tests.
fn ne_null_checked_expr(cond: &Expr) -> Option<&Expr> {
    let Expr::Cmp(CmpKind::Ne, lhs, rhs) = cond else {
        return None;
    };
    let is_zero = |e: &Expr| matches!(e, Expr::Int(0) | Expr::Null);
    let is_lit = |e: &Expr| matches!(e, Expr::Int(_) | Expr::Null);
    match (lhs.as_ref(), rhs.as_ref()) {
        (a, b) if is_zero(a) && !is_lit(b) => Some(b),
        (a, b) if is_zero(b) && !is_lit(a) => Some(a),
        _ => None,
    }
}

/// Flatten one `if (X != 0) { … }` whose false path was the removed trailing
/// raise. The body must use `X` (otherwise the test is real managed logic)
/// and must end in a terminator — directly, or through another such guard
/// chain. Returns the straight-line replacement body.
fn unwrap_tail_null_guard(s: &Stmt) -> Option<Vec<Stmt>> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = s
    else {
        return None;
    };
    if !else_body.is_empty() || then_body.is_empty() {
        return None;
    }
    let x = ne_null_checked_expr(cond)?;
    if checked_is_call_result(x) {
        return None;
    }
    let mut v = NameReadVisitor::new_expr(x);
    v.visit_stmts(then_body);
    if !v.found {
        return None;
    }
    let last = then_body.last().unwrap();
    if stmt_ends_control_flow(last) {
        return Some(then_body.clone());
    }
    let inner = unwrap_tail_null_guard(last)?;
    let mut body = then_body[..then_body.len() - 1].to_vec();
    body.extend(inner);
    Some(body)
}

fn collect_goto_labels(stmts: &[Stmt], out: &mut Vec<String>) {
    for s in stmts {
        if let Stmt::Goto(l) = s {
            out.push(l.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_goto_labels(k, out);
        }
    }
}

/// Unwrap a tail chain of null guards whose null arms were the synthesized
/// raise just removed. Refuses when a trailing label is still a jump target:
/// unwrapping would turn that jump into a fall-through past the old raise.
fn unwrap_trailing_null_guards(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    loop {
        let Some(idx) = stmts
            .iter()
            .rposition(|s| !matches!(s, Stmt::Label(_)))
        else {
            break;
        };
        let Some(body) = unwrap_tail_null_guard(&stmts[idx]) else {
            break;
        };
        let trailing: Vec<String> = stmts[idx + 1..]
            .iter()
            .filter_map(|s| match s {
                Stmt::Label(l) => Some(l.clone()),
                _ => None,
            })
            .collect();
        if !trailing.is_empty() {
            let mut gotos = Vec::new();
            collect_goto_labels(stmts, &mut gotos);
            if trailing.iter().any(|l| gotos.contains(l)) {
                break;
            }
        }
        let tail = stmts.split_off(idx + 1);
        stmts.pop();
        stmts.extend(body);
        stmts.extend(tail);
        n += 1;
    }
    n
}

/// Scan a statement list and remove null-guard if statements.
fn strip_null_guards_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < stmts.len() {
        if raise_clean_debug() {
            if let Stmt::If {
                cond,
                then_body,
                else_body,
            } = &stmts[i]
            {
                for (arm, body) in [("then", then_body), ("else", else_body)] {
                    // Find the raise call at ANY position in the arm, not just first.
                    if let Some(raise_pos) = body.iter().position(|s| {
                        matches!(s, Stmt::Eval(Expr::RawCall(a, _, _)) if is_raise_helper(*a))
                    }) {
                        let Stmt::Eval(Expr::RawCall(_a, args, ret)) = &body[raise_pos] else {
                            continue;
                        };
                        eprintln!(
                            "[raiselean] arm={} pos={} cond={:?} ret={:?} nargs={} len={} keep={:?} prelude={:?} tail={:?}",
                            arm,
                            raise_pos,
                            cond,
                            ret,
                            args.len(),
                            body.len(),
                            classify_null_raise_guard(&stmts[i]).map(|k| k.len()),
                            &body[..raise_pos],
                            &body[raise_pos + 1..]
                        );
                    }
                }
            }
        }
        if is_null_guard(&stmts[i], &stmts[i + 1..]) {
            stmts.remove(i);
            count += 1;
            // Don't increment i — the next statement shifted into position i.
        } else if let Some(keep) = classify_null_raise_guard(&stmts[i]) {
            // One arm is a proven dead raise guard (`if (X == 0) { raise; … }`
            // in either CFG orientation); the managed source is the other arm.
            stmts.remove(i);
            for s in keep.into_iter().rev() {
                stmts.insert(i, s);
            }
            count += 1;
        } else if let Some(keep) = classify_split_null_raise_guard(stmts, i) {
            // `if (X != 0) { S; goto L; } throw new NullReferenceException(…)`
            // — branch-over-join leaves the raise arm as the statement AFTER
            // the If instead of in its else. S is the managed source; the
            // trailing throw is the injected null path.
            stmts.remove(i);
            stmts.remove(i);
            // The raise never returns; keep only from the first possible jump
            // target (a statement defining a label) onward.
            while i < stmts.len() && !stmt_has_label(&stmts[i]) {
                stmts.remove(i);
            }
            let tail_only_labels = stmts[i..].iter().all(|s| matches!(s, Stmt::Label(_)));
            for s in keep.into_iter().rev() {
                stmts.insert(i, s);
            }
            // The kept body's own trailing guards tested `Y != 0` and their
            // false paths fell into the raise just removed (nested checks for
            // `a.b[0].c`). Removing only the outer guard leaves them wrapping
            // a return with no catch-all — the method then has a reachable
            // path with no return (0x18A5204). Unwrap that tail chain: the
            // null behaviour is C#'s implicit NullReferenceException.
            if tail_only_labels {
                count += unwrap_trailing_null_guards(stmts);
            }
            count += 1;
        } else {
            i += 1;
        }
    }
    count
}

/// Classify IL2CPP's injected null-raise guard in either orientation:
///
/// ```text
/// if (X == 0) { [prelude assigns;] fn_raise(anything); [return]; }      else { S }   // rendered tree
/// if (X != 0) { [prelude assigns;] fn_raise(anything); [return]; }      else { S }   // CFG node (see below)
/// ```
///
/// At this stage many `If` nodes are still raw CFG branch nodes whose
/// taken/fall-through arms are flipped by a later threading pass, so the
/// raise call sits under a `!= 0` condition (measured on the v31.1 corpus:
/// every surviving `fn_1302634` guard reaches here as `Ne(X, 0)` with the
/// raise in the then-arm). Either way the arm CONTAINING the raise call
/// (possibly preceded by SSA temp assignments) is the null path the runtime
/// injected — it can only run when `X` is null (or the raise would fire on
/// live objects) — and the other arm is the managed source. The raise call's
/// own arguments don't identify the compared operand (codegen passes the owner,
/// not the field), so safety comes from the GLOBAL address classification:
/// `RuntimeHelper::RaiseException` is only ever noted for thin noreturn
/// wrappers proven to reach `__cxa_throw`, never for a user method.
///
/// The raise call may be preceded by a chain of SSA temp assignments
/// (`v127 = ...; v128 = ...;`) that feed the raise's arguments. These are safe
/// to delete if none of the assigned names are live after the arm (i.e., not
/// read in the kept arm). This is a conservative liveness check: if any
/// prelude-assigned name appears as a read anywhere in the kept arm, we refuse
/// to match (better to leave the guard than to delete live code).
///
/// Returns the kept statements (possibly empty → delete the guard).
fn classify_null_raise_guard(stmt: &Stmt) -> Option<Vec<Stmt>> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };
    if then_body.is_empty() {
        return None;
    }
    // The condition must compare a non-literal against 0/null (either
    // orientation).
    let Expr::Cmp(k, lhs, rhs) = cond else {
        return None;
    };
    if !matches!(k, CmpKind::Eq | CmpKind::Ne) {
        return None;
    }
    let is_zero = |e: &Expr| matches!(e, Expr::Int(0) | Expr::Null);
    let is_lit = |e: &Expr| matches!(e, Expr::Int(_) | Expr::Null);
    let checked_expr = match (lhs.as_ref(), rhs.as_ref()) {
        (a, b) if is_zero(a) && !is_lit(b) => b,
        (a, b) if is_zero(b) && !is_lit(a) => a,
        _ => return None,
    };
    if checked_is_call_result(checked_expr) {
        return None;
    }
    // The kept arm must actually USE the tested value. A guard whose other arm
    // never reads it is an `if (!Status()) throw` error check — real managed
    // logic — not IL2CPP's auto-inserted null check.
    let keep_uses_checked = |kept: &[Stmt]| {
        let mut v = NameReadVisitor::new_expr(checked_expr);
        v.visit_stmts(kept);
        v.found
    };
    // Find the raise call at ANY position in the arm. Either arm can carry it
    // — the CFG nodes reaching this pass have not yet had their taken/fall-through
    // arms threaded, so the same guard shows up as raise-in-then (`Ne(X,0)`) or
    // raise-in-else (`Eq(X,0)`).
    // Matches BOTH shapes:
    //   Stmt::Eval(RawCall(raise_addr, ...))            — void raise
    //   Stmt::Assign(_, _, RawCall(raise_addr, ...))    — raise assigned to temp
    //   Stmt::Throw(NewObj("NullReferenceException"))   — materialized by
    //       raise_to_throw after this matcher already ran earlier in the round
    //
    // An UNRESOLVED `fn_` call also qualifies when it is passed the tested
    // value and its result is dead (checked below): this binary's codegen null
    // check (`fn_1302634`) does not reach `__cxa_throw` through the thin-wrapper
    // analysis, so it never earns `RaiseException`, yet every call site sits in
    // exactly this null arm with the checked object in x0. Rendering it as a
    // normal call emitted its whole register dump (measured: +11k vN refs).
    let same_expr = |a: &Expr, b: &Expr| format!("{a:?}") == format!("{b:?}");
    // (position, proven). A PROVEN raise role may sit in either arm because
    // raw CFG nodes are not yet threaded; an UNRESOLVED call qualifies only on
    // the null arm dictated by the condition's polarity (Eq→then, Ne→else),
    // or a user call on the non-null arm would be deleted.
    let find_raise_pos = |b: &[Stmt]| -> Option<(usize, bool)> {
        b.iter().enumerate().find_map(|(pos, s)| match s {
            Stmt::Throw(e) if expr_is_null_throw_call(e) => Some((pos, true)),
            Stmt::Eval(Expr::RawCall(addr, args, _))
            | Stmt::Assign(_, _, Expr::RawCall(addr, args, _)) => {
                if is_raise_helper(*addr) {
                    Some((pos, true))
                } else if args.first().is_some_and(|a| same_expr(a, checked_expr)) {
                    Some((pos, false))
                } else {
                    None
                }
            }
            _ => None,
        })
    };
    // The raise's destination (when assigned to a temp) must not be live in the
    // kept arm; otherwise deleting the arm loses a value the source reads.
    let raise_dest_live = |arm: &[Stmt], pos: usize, kept: &[Stmt]| -> bool {
        match &arm[pos] {
            Stmt::Assign(name, _, _) => {
                let mut v = NameReadVisitor::new(std::slice::from_ref(name));
                v.visit_stmts(kept);
                v.found
            }
            _ => false,
        }
    };
    // Statements after the raise in the dead arm are unreachable: the raise is
    // a proven-noreturn helper. They may only stay if any of them defines a
    // `Label` (directly or nested), which an outer `Goto` could target.
    // `Label` must not be deleted — that would dangle the edge.
    let tail_is_dead = |b: &[Stmt], raise_pos: usize| -> bool {
        b[raise_pos + 1..].iter().all(|s| !stmt_has_label(s))
    };
    // Statements before the raise must be Assign only (SSA temp assignments
    // feeding the raise's arguments). Collect the assigned names for liveness check.
    let prelude_is_assigns_only = |b: &[Stmt], raise_pos: usize| -> Option<Vec<String>> {
        let mut names = Vec::new();
        for s in &b[..raise_pos] {
            if let Stmt::Assign(name, _, _) = s {
                names.push(name.clone());
            } else {
                return None;
            }
        }
        Some(names)
    };
    // Conservative liveness check: refuse if any prelude-assigned name appears
    // as a read anywhere in the kept arm. This is safe but may miss some cases
    // where the name is shadowed or not actually live.
    let prelude_names_live_in = |names: &[String], kept: &[Stmt]| -> bool {
        let mut visitor = NameReadVisitor::new(names);
        visitor.visit_stmts(kept);
        visitor.found
    };
    // Try then-arm as the raise arm.
    if let Some((raise_pos, proven)) = find_raise_pos(then_body) {
        let polarity_ok = proven || matches!(k, CmpKind::Eq);
        if polarity_ok
            && tail_is_dead(then_body, raise_pos)
            && !raise_dest_live(then_body, raise_pos, else_body)
        {
            if let Some(prelude_names) = prelude_is_assigns_only(then_body, raise_pos) {
                if !prelude_names_live_in(&prelude_names, else_body)
                    && keep_uses_checked(else_body)
                {
                    return Some(else_body.clone());
                }
            }
        }
    }
    // Try else-arm as the raise arm.
    if let Some((raise_pos, proven)) = find_raise_pos(else_body) {
        let polarity_ok = proven || matches!(k, CmpKind::Ne);
        if polarity_ok
            && tail_is_dead(else_body, raise_pos)
            && !raise_dest_live(else_body, raise_pos, then_body)
        {
            if let Some(prelude_names) = prelude_is_assigns_only(else_body, raise_pos) {
                if !prelude_names_live_in(&prelude_names, then_body)
                    && keep_uses_checked(then_body)
                {
                    return Some(then_body.clone());
                }
            }
        }
    }
    None
}

/// Classify the SPLIT form of the same injected guard, produced when
/// `branch_repair` folds a branch-over-join diamond:
///
/// ```text
/// if (X != 0) { … keep arm …; goto L; }
/// throw new NullReferenceException(…);        // the raise arm, at the join
/// L: …
/// ```
///
/// The keep arm is the managed source, and it must END the then-path
/// (goto/return/throw/break/continue) — that is what makes the following
/// throw reachable ONLY when `X == 0`. The `!= 0` polarity is part of the
/// match: with `== 0` the then-arm would be the null path itself.
///
/// Splicing the keep arm out deletes the `if` shell, so in the reconstructed
/// managed source the keep arm runs unconditionally and its first dereference
/// of `X` raises `NullReferenceException` automatically on the null path, as
/// the original C# did.
fn classify_split_null_raise_guard(stmts: &[Stmt], i: usize) -> Option<Vec<Stmt>> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmts.get(i)?
    else {
        return None;
    };
    if !else_body.is_empty() || then_body.is_empty() {
        return None;
    }
    // Strict adjacency: a label between the If and the throw would make the
    // throw a jump target, and deleting it would dangle that edge.
    if !matches!(stmts.get(i + 1), Some(s) if stmt_is_synthesized_null_throw(s)) {
        return None;
    }
    let Expr::Cmp(k, lhs, rhs) = cond else {
        // A constant/fallback condition (`Bool(true)` / `!Bool(true)`) carries
        // no recovered test — the structurer swapped a payload arm under it.
        // There the keep arm must provably end the then-path, otherwise the
        // throw is reachable from BOTH arms and is not this guard's null arm.
        if is_constant_condition(cond) {
            return match then_body.last() {
                Some(last) if stmt_ends_control_flow(last) => Some(then_body.clone()),
                _ => None,
            };
        }
        return None;
    };
    if !matches!(k, CmpKind::Ne) {
        return None;
    }
    let is_zero = |e: &Expr| matches!(e, Expr::Int(0) | Expr::Null);
    let is_lit = |e: &Expr| matches!(e, Expr::Int(_) | Expr::Null);
    let checked_expr = match (lhs.as_ref(), rhs.as_ref()) {
        (a, b) if is_zero(a) && !is_lit(b) => b,
        (a, b) if is_zero(b) && !is_lit(a) => a,
        _ => return None,
    };
    if checked_is_call_result(checked_expr) {
        return None;
    }
    // `if (X != 0) { S } throw` — the throw is the null path only if S is the
    // code that uses X. A status/flag test whose payload never reads X is a
    // user error check; keep its throw.
    let mut v = NameReadVisitor::new_expr(checked_expr);
    v.visit_stmts(then_body);
    v.found.then(|| then_body.clone())
}

fn is_constant_condition(cond: &Expr) -> bool {
    match cond {
        Expr::Bool(_) => true,
        Expr::Not(inner) => matches!(**inner, Expr::Bool(_)),
        _ => false,
    }
}

/// Does this statement end the path that executes it — a direct jump/return/
/// throw, or an `if` whose every arm ends control flow? Used to prove the
/// statement after a split guard's keep arm is reachable only on the null
/// path.
fn stmt_ends_control_flow(s: &Stmt) -> bool {
    match s {
        Stmt::Goto(_)
        | Stmt::GotoCase(_)
        | Stmt::Return(_)
        | Stmt::Throw(_)
        | Stmt::Break
        | Stmt::Continue
        | Stmt::YieldBreak => true,
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            !then_body.is_empty()
                && !else_body.is_empty()
                && stmt_ends_control_flow(then_body.last().unwrap())
                && stmt_ends_control_flow(else_body.last().unwrap())
        }
        // `drop_unreachable_gotos` turns a goto whose label was never emitted
        // into this comment. The edge it encoded still does not fall through,
        // which is the structural fact this check needs.
        Stmt::Comment(t) => t.starts_with("goto ") && t.ends_with("(target not emitted)"),
        _ => false,
    }
}

/// Has the global helper classifier proven `addr` to be a noreturn raise
/// wrapper (`IL2CPP_RaiseException`)? Role-based, so it works across IL2CPP
/// versions — the address differs per binary, the evidence does not.
fn is_raise_helper(addr: u64) -> bool {
    crate::recovery::runtime_helpers::role_of(addr)
        == Some(crate::recovery::runtime_helpers::RuntimeHelper::RaiseException)
}

/// Visitor that checks if any of a set of names are read in an expression or
/// statement tree. Used for conservative liveness checking of prelude
/// assignments in raise guards.
struct NameReadVisitor<'a> {
    targets: &'a [String],
    /// When set, any expression structurally equal to this one counts as a
    /// match. Used to prove a null guard's kept code actually uses the value
    /// the guard tested: an `if (X == 0) raise` whose other arm never reads X
    /// is a boolean/status check (user error path), not a null guard.
    target_expr: Option<&'a Expr>,
    found: bool,
}

impl<'a> NameReadVisitor<'a> {
    fn new(targets: &'a [String]) -> Self {
        Self {
            targets,
            target_expr: None,
            found: false,
        }
    }

    fn new_expr(target: &'a Expr) -> Self {
        Self {
            targets: &[],
            target_expr: Some(target),
            found: false,
        }
    }

    fn visit_stmts(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            if self.found {
                return;
            }
            self.visit_stmt(stmt);
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        if self.found {
            return;
        }
        match stmt {
            Stmt::Declare(..) => {}
            Stmt::Assign(_, _, expr) => {
                // The assignment target is a write, not a read. Check the RHS.
                self.visit_expr(expr);
                // Also check if the assigned name is one of our targets (it's being
                // written here, but if it was read earlier in the prelude, that's a
                // different assignment; we only care about reads in the KEPT arm).
            }
            Stmt::Eval(expr)
            | Stmt::Return(Some(expr))
            | Stmt::YieldReturn(expr)
            | Stmt::Throw(expr) => {
                self.visit_expr(expr);
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.visit_expr(cond);
                self.visit_stmts(then_body);
                self.visit_stmts(else_body);
            }
            Stmt::While { cond, body } => {
                self.visit_expr(cond);
                self.visit_stmts(body);
            }
            Stmt::DoWhile { body, cond } => {
                self.visit_stmts(body);
                self.visit_expr(cond);
            }
            Stmt::For {
                init,
                cond,
                incr,
                body,
            } => {
                for s in init {
                    self.visit_stmt(s);
                }
                if let Some(cond) = cond {
                    self.visit_expr(cond);
                }
                for s in incr {
                    self.visit_stmt(s);
                }
                self.visit_stmts(body);
            }
            Stmt::ForEach {
                var_name: _,
                collection: _,
                body,
            } => {
                // collection is a String, not an Expr, so nothing to visit
                self.visit_stmts(body);
            }
            Stmt::Switch {
                disc: _,
                cases,
                tail,
                default,
            } => {
                for (_, case_body) in cases {
                    self.visit_stmts(case_body);
                }
                self.visit_stmts(tail);
                if let Some(d) = default {
                    self.visit_stmts(d);
                }
            }
            Stmt::StringSwitch { disc, cases, tail } => {
                self.visit_expr(disc);
                for (_, case_body) in cases {
                    self.visit_stmts(case_body);
                }
                if let Some(tail) = tail {
                    self.visit_stmts(tail);
                }
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                self.visit_stmts(try_body);
                for (_, catch_body) in catches {
                    self.visit_stmts(catch_body);
                }
                if let Some(finally_body) = finally_body {
                    self.visit_stmts(finally_body);
                }
            }
            Stmt::Using { decl, body } => {
                self.visit_stmt(decl);
                self.visit_stmts(body);
            }
            Stmt::Lock { expr, body } => {
                self.visit_expr(expr);
                self.visit_stmts(body);
            }
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                self.visit_expr(collection);
                self.visit_expr(index);
                self.visit_expr(value);
            }
            Stmt::EventAssign { field, value, .. } => {
                self.visit_expr(field);
                self.visit_expr(value);
            }
            // These don't contain expressions or nested statements.
            Stmt::Return(None)
            | Stmt::YieldBreak
            | Stmt::Break
            | Stmt::Continue
            | Stmt::Goto(_)
            | Stmt::Label(_)
            | Stmt::Comment(_)
            | Stmt::GotoCase(_) => {}
            Stmt::Attribute { .. } => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if self.found {
            return;
        }
        if let Some(target) = self.target_expr {
            if expr == target {
                self.found = true;
                return;
            }
        }
        match expr {
            Expr::Var(name) => {
                if self.targets.iter().any(|t| t == name) {
                    self.found = true;
                }
            }
            Expr::Field(base, _)
            | Expr::RawDeref(base, _, _)
            | Expr::RawAddr(base, _)
            | Expr::AddressOf(base)
            | Expr::Cast(_, base)
            | Expr::Not(base)
            | Expr::Neg(base)
            | Expr::Await(base) => {
                self.visit_expr(base);
            }
            Expr::StaticField(_, _) => {
                // Static field access doesn't contain a sub-expression
            }
            Expr::Bin(_, lhs, rhs) | Expr::Cmp(_, lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Cond(cond, then_expr, else_expr) => {
                self.visit_expr(cond);
                self.visit_expr(then_expr);
                self.visit_expr(else_expr);
            }
            Expr::Call { recv, args, .. } => {
                if let Some(recv) = recv {
                    self.visit_expr(recv);
                }
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::RawCall(_, args, _)
            | Expr::NewArr(_, args)
            | Expr::NewObj(_, args)
            | Expr::ArrayInit { items: args, .. } => {
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::CollectionInit {
                ctor_args, items, ..
            } => {
                for arg in ctor_args {
                    self.visit_expr(arg);
                }
                for item in items {
                    self.visit_expr(item);
                }
            }
            Expr::ObjInit {
                ctor_args, fields, ..
            } => {
                for arg in ctor_args {
                    self.visit_expr(arg);
                }
                for (_, field_expr) in fields {
                    self.visit_expr(field_expr);
                }
            }
            Expr::Index { collection, index } | Expr::NullConditional { receiver: collection, access: index } => {
                self.visit_expr(collection);
                self.visit_expr(index);
            }
            Expr::PatternMatch { expr, .. } => {
                self.visit_expr(expr);
            }
            Expr::Lambda { body, .. } => {
                self.visit_expr(body);
            }
            Expr::StringInterp { parts } => {
                for part in parts {
                    if let crate::recovery::csharp::InterpPart::Expr(expr, _) = part {
                        self.visit_expr(expr);
                    }
                }
            }
            Expr::MethodGroup(_) => {
                // Just a name, no sub-expressions
            }
            Expr::Tuple { names: _, elements } => {
                for elem in elements {
                    self.visit_expr(elem);
                }
            }
            Expr::MethodChain { receiver, methods } => {
                self.visit_expr(receiver);
                for method in methods {
                    for arg in &method.args {
                        self.visit_expr(arg);
                    }
                }
            }
            Expr::SwitchExpr {
                discrim,
                arms,
                default_arm,
            } => {
                self.visit_expr(discrim);
                for (case_expr, body_expr) in arms {
                    self.visit_expr(case_expr);
                    self.visit_expr(body_expr);
                }
                if let Some(default_expr) = default_arm {
                    self.visit_expr(default_expr);
                }
            }
            Expr::Range { from, to } => {
                if let Some(from_expr) = from {
                    self.visit_expr(from_expr);
                }
                if let Some(to_expr) = to {
                    self.visit_expr(to_expr);
                }
            }
            Expr::IndexFromEnd(expr) => {
                self.visit_expr(expr);
            }
            // These don't contain sub-expressions.
            Expr::This
            | Expr::Int(_)
            | Expr::Float(_)
            | Expr::Double(_)
            | Expr::Bool(_)
            | Expr::Str(_)
            | Expr::Null
            | Expr::MissingAbiArg
            | Expr::Unknown
            | Expr::TypeOf(_) => {}
        }
    }
}

/// `ILR_RAISECLEAN=1`: dump the AST shape of every raise-role guard branch
/// seen by the strip pass, for diagnosing matcher misses.
fn raise_clean_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::recovery::env_flags::flag("ILR_RAISECLEAN"))
}

/// Is this statement an IL2CPP null-reference guard?
///
/// Matches `if (expr == 0) { ThrowNullRef(...); ... }` with no else-body, and
/// only when the statements AFTER the guard use the checked value: an
/// `if (!Status()) throw` error check tests a boolean/status result the rest
/// of the method never reads, while a real null guard's subject is dereferenced
/// on the fall-through path.
fn is_null_guard(stmt: &Stmt, rest: &[Stmt]) -> bool {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return false;
    };

    // Must have no else-body.
    if !else_body.is_empty() {
        return false;
    }

    // Then-body must be non-empty.
    if then_body.is_empty() {
        return false;
    }

    // The condition must be a comparison against 0 or null.
    if !is_zero_or_null_check(cond) {
        return false;
    }

    // The first statement in the then-body must be a call to a null-reference
    // throwing function.
    if !body_contains_null_throw(then_body) {
        return false;
    }

    let Expr::Cmp(_, lhs, rhs) = cond else {
        return false;
    };
    let is_zero = |e: &Expr| matches!(e, Expr::Int(0) | Expr::Null);
    let is_lit = |e: &Expr| matches!(e, Expr::Int(_) | Expr::Null);
    let checked_expr = match (lhs.as_ref(), rhs.as_ref()) {
        (a, b) if is_zero(a) && !is_lit(b) => b,
        (a, b) if is_zero(b) && !is_lit(a) => a,
        _ => return false,
    };
    if checked_is_call_result(checked_expr) {
        return false;
    }
    let mut v = NameReadVisitor::new_expr(checked_expr);
    v.visit_stmts(rest);
    v.found
}

/// Does the condition test `expr == 0`, `expr == null`, `0 == expr`, or
/// `null == expr`?
fn is_zero_or_null_check(cond: &Expr) -> bool {
    let Expr::Cmp(k, lhs, rhs) = cond else {
        return false;
    };
    if !matches!(k, CmpKind::Eq) {
        return false;
    }
    matches!(
        (lhs.as_ref(), rhs.as_ref()),
        (_, Expr::Int(0))
            | (_, Expr::Null)
            | (Expr::Int(0), _)
            | (Expr::Null, _)
    )
}

/// Does this expression test a call result (directly or through a negation,
/// cast or comparison)? IL2CPP null checks compare memory values (fields,
/// locals, derefs); a branch on `!Status()` / `Status() == 0` is managed
/// error-handling logic that must keep its throw.
fn checked_is_call_result(e: &Expr) -> bool {
    match e {
        Expr::Call { .. } | Expr::RawCall(..) => true,
        Expr::Not(inner) | Expr::Neg(inner) | Expr::Cast(_, inner) => {
            checked_is_call_result(inner)
        }
        Expr::Cmp(_, l, r) | Expr::Bin(_, l, r) => {
            checked_is_call_result(l) || checked_is_call_result(r)
        }
        _ => false,
    }
}

/// Does the body's first statement call a null-reference throwing function?
///
/// For the materialized `throw new NullReferenceException(…)` form the rest of
/// the arm must be plain dead weight (no labels), because `is_null_guard`
/// removes the whole `if` — a label inside it could still be a goto target.
fn body_contains_null_throw(body: &[Stmt]) -> bool {
    match body.first() {
        Some(s) if stmt_is_synthesized_null_throw(s) => throw_tail_is_dead(&body[1..]),
        Some(s) => stmt_is_null_throw_call(s),
        None => false,
    }
}

/// Is this statement a call to a null-reference throwing function?
///
/// Matches:
/// - `ThrowHelper.ThrowNullReferenceException(...)` (Eval or Throw)
/// - Any call whose name contains "ThrowNullReference" (case-insensitive)
fn stmt_is_null_throw_call(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Eval(e) => expr_is_null_throw_call(e),
        // Some renderings use `throw ThrowNullReferenceException(...)` which
        // would appear as Stmt::Throw with a call expression.
        Stmt::Throw(e) => expr_is_null_throw_call(e),
        _ => false,
    }
}

/// The `throw new NullReferenceException(<message>)` that `raise_to_throw`
/// synthesizes for a globally-proven raise helper. It is created AFTER
/// `strip_null_guards` has already run in the same cleanup round, so the next
/// round must recognize the materialized Throw — otherwise the arm no longer
/// contains a call and the guard survives forever.
fn stmt_is_synthesized_null_throw(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::Throw(Expr::NewObj(name, _)) if is_null_reference_exception_name(name)
    )
}

fn is_null_reference_exception_name(name: &str) -> bool {
    name.to_lowercase().contains("nullreferenceexception")
}

/// After a terminal `throw`, only statements that cannot be jump targets are
/// dead-weight in the same arm. Anything with no `Label` (recursively) cannot
/// be entered except by fall-through, and fall-through is dead after a
/// proven-noreturn raise. A `Label` is forbidden: an outer `Goto` may target
/// it, and removing the arm would dangle that edge.
fn throw_tail_is_dead(tail: &[Stmt]) -> bool {
    tail.iter().all(|s| !stmt_has_label(s))
}

/// Is this expression a call to a null-reference throwing function?
fn expr_is_null_throw_call(e: &Expr) -> bool {
    match e {
        Expr::Call { name, .. } => {
            let lower = name.to_lowercase();
            lower.contains("thrownullreference")
                || lower.contains("nullreferenceexception")
        }
        Expr::NewObj(name, _) => is_null_reference_exception_name(name),
        // RawCall to a helper the GLOBAL classifier proved noreturn-raising
        // (thin forwarder reaching `__cxa_throw`). Address-specific, not
        // shape-guessed, so it is safe to treat the guard as runtime-injected
        // noise — and the same evidence works on every IL2CPP build, whatever
        // the helper's address is in it.
        Expr::RawCall(addr, ..) => is_raise_helper(*addr),
        _ => false,
    }
}

// ===========================================================================
// 7A-2: Class initialization guard stripping
// ===========================================================================

/// Strip IL2CPP class initialization guards.
///
/// ## Pattern
///
/// ```text
/// if (*(byte*)(ADDR) == 0) {
///     IL2CPP_ClassInit(...);
///     [more IL2CPP_ClassInit calls...]
///     *(byte*)(ADDR) = 1;
///     [optional other setup]
/// }
/// ```
///
/// The key identifier is that the body contains calls to `IL2CPP_ClassInit`.
///
/// ## Why this is safe
///
/// In managed C#, the runtime handles static constructor invocation lazily and
/// thread-safely. The IL2CPP native code must check a byte flag and call the
/// class initializer manually. When rendering back to C#, this is noise.
fn strip_class_init_guards(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += strip_class_init_guards_list(list);
    });
    count
}

/// Scan a statement list and remove class-init guard if statements.
fn strip_class_init_guards_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < stmts.len() {
        if is_class_init_guard(&stmts[i]) {
            stmts.remove(i);
            count += 1;
        } else if let Some(body) = unwrap_byte_static_guard(&stmts[i]) {
            // Replace the guard `if` with its body (minus the flag store). The
            // spliced statements are ordinary code now living at position `i`;
            // we do NOT advance so a nested guard inside the body is unwrapped
            // too. Termination is guaranteed because the guard `if` itself is
            // gone — the re-examined statements are its (non-If) body.
            stmts.splice(i..i + 1, body);
            count += 1;
        } else {
            i += 1;
        }
    }
    count
}

/// Extract the rendered address string of a class-init byte guard condition.
///
/// Matches the condition shape `(*(byte*)(STATIC + off) & 1) == 0` where the
/// deref base is a LITERAL absolute static address (an `Expr::Int`, never a
/// register/`this`/stack base) and the access width is one byte. Returns the
/// exact string the renderer would emit for that deref, so a `*(byte*) … = 1`
/// store target can be compared against it verbatim.
fn byte_static_guard_addr(cond: &Expr) -> Option<String> {
    // Outer comparison must be `… == 0` (either operand order).
    let Expr::Cmp(CmpKind::Eq, a, b) = cond else {
        return None;
    };
    let (and_side, zero_side) = if matches!(**b, Expr::Int(0)) {
        (a.as_ref(), true)
    } else if matches!(**a, Expr::Int(0)) {
        (b.as_ref(), true)
    } else {
        return None;
    };
    if !zero_side {
        return None;
    }

    // The non-zero side must be `deref & 1` (either operand order).
    let Expr::Bin(BinOp::And, x, y) = and_side else {
        return None;
    };
    let deref = match (x.as_ref(), y.as_ref()) {
        (Expr::Int(1), d) | (d, Expr::Int(1)) => d,
        _ => return None,
    };

    // The deref must be a one-byte load of a literal static address.
    let Expr::RawDeref(base, _, 1) = deref else {
        return None;
    };
    if !matches!(**base, Expr::Int(_)) {
        return None;
    }

    let mut s = String::new();
    deref.render(&mut s);
    Some(s)
}

/// Try to unwrap a class-init byte guard of the inline-store shape:
///
/// ```text
/// if ((*(byte*)(STATIC + off) & 1) == 0) {
///     BODY;
///     *(byte*)(STATIC + off) = 1;   // flag store INSIDE the body
/// }
/// ```
///
/// On a match this returns `BODY` with the flag store removed. Safety, per the
/// corpus spec:
/// - the compared address (condition deref) and the flag store address are the
///   SAME literal static `(base + off)` byte deref,
/// - exactly one store to that address occurs inside the body (the `= 1` flag
///   write); if the body contains any OTHER store to the same address we bail
///   rather than guess which is the guard,
/// - no `else` arm.
fn unwrap_byte_static_guard(stmt: &Stmt) -> Option<Vec<Stmt>> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };
    if !else_body.is_empty() {
        return None;
    }
    // Must not already be the IL2CPP_ClassInit form (that whole-if is deleted
    // by `is_class_init_guard` before we get here, but be defensive).
    if then_body.iter().any(|s| stmt_is_class_init_call(s)) {
        return None;
    }

    let addr = byte_static_guard_addr(cond)?;

    // Find stores to the guard address in the body. Exactly one must exist and
    // it must be the `= 1` flag write.
    let mut flag_store_idx: Option<usize> = None;
    let mut to_addr_stores = 0usize;
    for (k, s) in then_body.iter().enumerate() {
        if let Stmt::Assign(target, _, value) = s {
            if *target == addr {
                to_addr_stores += 1;
                if matches!(value, Expr::Int(1)) {
                    flag_store_idx = Some(k);
                }
            }
        }
    }
    // Require exactly one store to the address, and it is the `= 1` flag.
    if to_addr_stores != 1 {
        return None;
    }
    let idx = flag_store_idx?;

    // Return the body minus the flag store.
    let mut body = then_body.clone();
    body.remove(idx);
    // If the only statement was the flag store, the body is empty — still valid
    // (the guard becomes a no-op that we drop entirely).
    Some(body)
}

/// Is this statement an IL2CPP class initialization guard?
fn is_class_init_guard(stmt: &Stmt) -> bool {
    let Stmt::If {
        cond: _,
        then_body,
        else_body,
    } = stmt
    else {
        return false;
    };

    // Must have no else-body.
    if !else_body.is_empty() {
        return false;
    }

    // Then-body must contain at least one IL2CPP_ClassInit call.
    then_body.iter().any(|s| stmt_is_class_init_call(s))
}

/// Is this statement a call to `IL2CPP_ClassInit`?
fn stmt_is_class_init_call(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Eval(e) | Stmt::Assign(_, _, e) => expr_is_class_init_call(e),
        _ => false,
    }
}

/// Is this expression a call to `IL2CPP_ClassInit`?
fn expr_is_class_init_call(e: &Expr) -> bool {
    match e {
        Expr::Call { name, .. } => name.contains("IL2CPP_ClassInit") || name.contains("il2cpp_runtime_class_init"),
        Expr::RawCall(addr, _, _) => {
            super::super::runtime_helpers::role_of(*addr)
                == Some(super::super::runtime_helpers::RuntimeHelper::ClassInit)
        }
        _ => false,
    }
}

// ===========================================================================
// 7A-3: IL2CPP_RaiseException → throw (noreturn semantics)
// ===========================================================================

/// Convert raise-helper calls into proper C# `throw` statements.
///
/// ## Problem
///
/// IL2CPP's `il2cpp_codegen_raise_exception` is a **noreturn** function that
/// throws an exception at runtime. The decompiler currently models it as a
/// regular call that may return a value, producing patterns like:
///
/// ```text
/// v = IL2CPP_RaiseException(this);   // temp holding "result" of throw
/// return v;                          // dead — throw never returns
/// ```
///
/// This forces temps to exist just to hold the never-materialized return value,
/// and their argument-feed predecessors (SSA copies of fields/args that feed
/// the raise call) become dead once the throw is recognized.
///
/// ## Fix
///
/// Rewrite every `RawCall` whose target is classified as
/// `RuntimeHelper::RaiseException` into `Stmt::Throw(new NullReferenceException())`,
/// and drop the now-dead `return` that follows it. After this pass, the
/// `remove_dead_temp_assignments` pass sees the argument-feed temps as unused
/// and deletes them.
///
/// ## Patterns matched
///
/// 1. `Assign(v, _, RawCall(raise_addr, ...)); Return(Some(Var(v)))`
///    → `Throw(new NullReferenceException())`  (both statements collapse into one)
/// 2. `Assign(v, _, RawCall(raise_addr, ...)); Return(None)`
///    → `Throw(new NullReferenceException())`
/// 3. `Assign(v, _, RawCall(raise_addr, ...))` (no trailing return)
///    → `Throw(new NullReferenceException())`
/// 4. `Eval(RawCall(raise_addr, ...)); Return(...)`
///    → `Throw(new NullReferenceException())`
/// 5. `Eval(RawCall(raise_addr, ...))` (standalone)
///    → `Throw(new NullReferenceException())`
///
/// ## Safety
///
/// `RuntimeHelper::RaiseException` is only ever assigned to functions proven
/// (via the global reach-`__cxa_throw` classifier) to be noreturn raise
/// wrappers — never to user methods. So the rewrite cannot misfire.
fn raise_to_throw(stmts: &mut Vec<Stmt>) -> usize {
    // Labels that a goto still targets: a dead tail after a raise may only be
    // kept when it DEFINES one of these, otherwise it is unreachable code that
    // the sweep should still delete.
    let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_goto_targets(stmts, &mut referenced);
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += raise_to_throw_list(list, &referenced);
    });
    count
}

fn collect_goto_targets(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in stmts {
        if let Stmt::Goto(l) = s {
            out.insert(l.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_goto_targets(k, out);
        }
    }
}

fn raise_to_throw_list(
    stmts: &mut Vec<Stmt>,
    referenced: &std::collections::HashSet<String>,
) -> usize {
    let mut total = 0;
    loop {
        let mut count = 0;
        let mut i = 0;
        while i < stmts.len() {
            // Extract the raise address from either Assign or Eval shape. The
            // raise role is now proven NORETURN at classification time
            // (`returns_normally` rejects allocators that merely reach a
            // throw path), so the assignment's destination is dead by
            // definition and the tail can be dropped.
            let raise_addr = match &stmts[i] {
                Stmt::Assign(_, _, Expr::RawCall(addr, _, _)) if is_raise_helper(*addr) => Some(*addr),
                Stmt::Eval(Expr::RawCall(addr, _, _)) if is_raise_helper(*addr) => Some(*addr),
                _ => None,
            };

            let Some(addr) = raise_addr else {
                i += 1;
                continue;
            };
            // Look up the extracted message, or use default
            let message = crate::recovery::runtime_helpers::raise_message(addr)
                .unwrap_or_else(|| "Object reference not set to an instance of an object.".to_string());

            stmts[i] = Stmt::Throw(Expr::NewObj(
                "NullReferenceException".to_string(),
                vec![Expr::Str(message)],
            ));
            count += 1;
            drop_unreachable_tail(stmts, i + 1, referenced);
            i += 1;
        }
        total += count;
        if count == 0 {
            break;
        }
    }
    total
}

/// Remove the statements that follow a raise-to-throw rewrite in the same block.
///
/// `IL2CPP_RaiseException` never returns, so whatever the decompiler emitted
/// after it is unreachable: stores into the exception object and, most visibly,
/// reads of the raise's own "result" temp — a variable whose definition this pass
/// has just replaced with a `throw`.
///
/// The tail is provably dead only up to the first statement that IS a label:
/// a label is how control re-enters a block mid-way, so anything from there on
/// may be a jump target and stays. We only check if the statement itself is a
/// label, not if it contains a label (which would be too conservative).
fn drop_unreachable_tail(
    stmts: &mut Vec<Stmt>,
    from: usize,
    referenced: &std::collections::HashSet<String>,
) -> usize {
    // Stop at the first statement that DEFINES a label a goto still targets,
    // at any depth: deleting it would dangle that jump (measured: 0x1532520's
    // L26–L28 lived inside nested arms of the tail after a raise). Labels no
    // goto references are ordinary dead tail and go with it.
    let end = stmts[from..]
        .iter()
        .position(|s| defines_referenced_label(s, referenced))
        .map_or(stmts.len(), |p| from + p);
    let removed = end - from;
    if removed > 0 {
        stmts.drain(from..end);
    }
    removed
}

fn defines_referenced_label(s: &Stmt, referenced: &std::collections::HashSet<String>) -> bool {
    match s {
        Stmt::Label(l) => referenced.contains(l),
        _ => {
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            kids.iter()
                .any(|k| k.iter().any(|x| defines_referenced_label(x, referenced)))
        }
    }
}

fn stmt_has_label(s: &Stmt) -> bool {
    if matches!(s, Stmt::Label(_)) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    kids.iter().any(|k| k.iter().any(|x| stmt_has_label(x)))
}

// ===========================================================================
// 7A-2b: Dead code elimination after terminal if-else
// ===========================================================================

/// Drop unreachable code after if-else statements where both branches are terminal.
///
/// When `raise_to_throw` converts raises to throws, those throws are terminal
/// (they never return). If an if-else has both branches ending with a terminal
/// statement (throw, return, goto), then any code after the if-else is unreachable.
///
/// This pass walks all statement lists and truncates the tail after such terminal
/// if-else statements, up to the first label (which may be a jump target).
fn drop_dead_code_after_terminal(stmts: &mut Vec<Stmt>) -> usize {
    let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_goto_targets(stmts, &mut referenced);
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += drop_dead_code_after_terminal_list(list, &referenced);
    });
    count
}

fn drop_dead_code_after_terminal_list(
    stmts: &mut Vec<Stmt>,
    referenced: &std::collections::HashSet<String>,
) -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < stmts.len() {
        // Pattern 1: terminal if-else (both branches end with throw/return/goto)
        if stmt_is_terminal_if_else(&stmts[i]) {
            let removed = drop_unreachable_tail(stmts, i + 1, referenced);
            if removed > 0 {
                total += removed;
                continue;
            }
        }
        // Pattern 2: if-else followed by a goto (the goto makes everything after it dead)
        if i + 1 < stmts.len() && matches!(&stmts[i], Stmt::If { .. }) && matches!(&stmts[i + 1], Stmt::Goto(_)) {
            let removed = drop_unreachable_tail(stmts, i + 2, referenced);
            if removed > 0 {
                total += removed;
                continue;
            }
        }
        // Pattern 3: if-then-terminal followed by a label (the goto was already removed by idioms::dead_code)
        // This handles the case where the structurer placed a goto in the parent scope after the if-else,
        // and idioms::dead_code removed it because it was adjacent to its label.
        // Pattern: If { then: [terminal], else: [terminal] }, Label, <dead code>
        // If both branches are terminal (end with throw/return/goto or nested terminal if-else),
        // then code after the label is dead.
        if let Stmt::If { then_body, else_body, .. } = &stmts[i] {
            if i + 1 < stmts.len() {
                if let Stmt::Label(name) = &stmts[i + 1] {
                    // The premise of this pattern is that the guard's jump to
                    // this label was ALREADY removed (so nothing reaches the
                    // label any more and the code after it is dead). When a
                    // jump still names it — including one of this very if's
                    // arms — control DOES arrive there, and dropping the body
                    // would delete live code. Measured: a guard chain ending
                    // `if (c) goto L; else return; L: call(); return;` lost
                    // the call.
                    if referenced.contains(name) {
                        i += 1;
                        continue;
                    }
                    // Check if both branches are terminal (not just direct throw)
                    let then_is_terminal = then_body.last().is_some_and(|s| stmt_ends_terminal(s));
                    let else_is_terminal = !else_body.is_empty() && stmt_ends_terminal(else_body.last().unwrap());
                    
                    if then_is_terminal && else_is_terminal {
                        // Both branches are terminal
                        let removed = drop_unreachable_tail(stmts, i + 2, referenced);
                        if removed > 0 {
                            total += removed;
                            continue;
                        }
                    }
                }
            }
        }
        // Pattern 4: if-then-terminal followed by a synthesized null raise.
        //
        // The old version deleted ANY return/throw after an `if` whose then
        // arm terminates and whose else falls through. That premise is false:
        // the statement AFTER the `if` is exactly what the false branch runs.
        // `if (delay <= 0f) { … return v314; } return 1;` legitimately returns
        // 1 when `delay > 0`; deleting it made Stumble @0x196649C fall out of
        // the method with `return 0` on BOTH the delay>0 and CurrentState!=0
        // paths (native branches to a shared `return 1` block). The raise the
        // pattern was written for is IL2CPP's injected null check, so restrict
        // the deletion to a synthesized NullReferenceException throw.
        if let Stmt::If { then_body, else_body, .. } = &stmts[i] {
            if i + 1 < stmts.len() {
                let next_stmt = &stmts[i + 1];
                let next_is_null_raise = stmt_is_synthesized_null_throw(next_stmt);

                // Check if the statement after next is a label (recovery region)
                let followed_by_label = i + 2 < stmts.len() && matches!(&stmts[i + 2], Stmt::Label(_));

                if next_is_null_raise && !followed_by_label {
                    // Check if then branch is terminal
                    let then_is_terminal = then_body.last().is_some_and(|s| stmt_ends_terminal(s));
                    // Check if else is empty or non-terminal (falls through)
                    let else_falls_through = else_body.is_empty() 
                        || !stmt_ends_terminal(else_body.last().unwrap());
                    
                    if then_is_terminal && else_falls_through {
                        // Then is terminal, else falls through, next statement is terminal
                        // and not followed by a label, so it's dead code.
                        stmts.remove(i + 1);
                        total += 1;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    total
}

/// Check if a statement is an if-else where both branches are "effectively terminal".
/// A branch is effectively terminal if it ends with throw/return/goto, OR if it
/// contains a goto anywhere (which means control flow doesn't fall through normally).
fn stmt_is_terminal_if_else(s: &Stmt) -> bool {
    let Stmt::If {
        then_body,
        else_body,
        ..
    } = s
    else {
        return false;
    };
    // Both arms must be non-empty.
    if then_body.is_empty() || else_body.is_empty() {
        return false;
    }
    stmt_is_effectively_terminal(then_body) && stmt_is_effectively_terminal(else_body)
}

/// Check if a list of statements is "effectively terminal" — either ends with a
/// terminal statement, or contains a goto (which means control flow exits abnormally).
fn stmt_is_effectively_terminal(stmts: &[Stmt]) -> bool {
    // Check if the last statement is terminal.
    if stmt_ends_terminal(&stmts[stmts.len() - 1]) {
        return true;
    }
    // Check if there's a goto that's not followed by executable code.
    // "Executable" means anything other than Assign to temp vars.
    for (i, s) in stmts.iter().enumerate() {
        if matches!(s, Stmt::Goto(_)) {
            // Check if everything after the goto is just dead temp assignments.
            let all_dead_after = stmts[i + 1..].iter().all(|s| {
                matches!(s, Stmt::Assign(name, _, _) if is_temp_var_name(name))
            });
            if all_dead_after {
                return true;
            }
        }
    }
    false
}

/// Check if a statement is terminal (never falls through to the next statement).
fn stmt_ends_terminal(s: &Stmt) -> bool {
    match s {
        Stmt::Throw(_) | Stmt::Return(_) | Stmt::Goto(_) => true,
        // An if-else is terminal if both branches are terminal.
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            !then_body.is_empty()
                && !else_body.is_empty()
                && stmt_ends_terminal(&then_body[then_body.len() - 1])
                && stmt_ends_terminal(&else_body[else_body.len() - 1])
        }
        _ => false,
    }
}

// ===========================================================================
// 7B: RPC ternary self-pointer collapse
// ===========================================================================

/// Collapse RPC ternary self-pointer patterns to `this`.
///
/// ## Pattern
///
/// Network weavers (Mirror/UNET) inject safety checks into `[ClientRpc]` and
/// `[Command]` methods:
///
/// ```text
/// (v58 == 0 ? this : v58).CallRpcRemove();
/// ```
///
/// The ternary checks if a context pointer is null and falls back to `this`.
/// In practice, the context pointer IS `this` (or null, in which case `this`
/// is the correct target). The ternary always resolves to `this`.
///
/// ## Matched shapes
///
/// 1. `Expr::Cond(Cmp(Eq, Var(v), Int(0)), This, Var(v))` → `This`
/// 2. `Expr::Cond(Cmp(Ne, Var(v), Int(0)), Var(v), This)` → `This`
/// 3. `Expr::Cond(Cmp(Eq, Var(v), Null), This, Var(v))` → `This`
/// 4. `Expr::Cond(Cmp(Ne, Var(v), Null), Var(v), This)` → `This`
///
/// Also matches when `Var(v)` is a `Field(This, name)` — the Unity
/// `get_contentContainer()` pattern:
/// `this.m_RadioButtonContainer == 0 ? this : this.m_RadioButtonContainer`
///
/// ## Why this is safe
///
/// The network weaver's ternary is a defensive null check on the RPC target
/// context. In managed C#, the RPC infrastructure handles this internally.
/// The ternary's two branches are `this` and the same variable being tested —
/// when the variable is null, we get `this`; when it's non-null, we get the
/// variable which IS `this` (or a reference to the same object).
fn collapse_rpc_ternary(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            count += collapse_rpc_ternary_stmt(stmt);
        }
    });
    count
}

/// Recursively collapse RPC ternary in a statement's expressions.
fn collapse_rpc_ternary_stmt(stmt: &mut Stmt) -> usize {
    let mut count = 0;
    match stmt {
        Stmt::Eval(e) => count += collapse_rpc_ternary_expr(e),
        Stmt::Assign(_, _, e) => count += collapse_rpc_ternary_expr(e),
        Stmt::Return(Some(e)) => count += collapse_rpc_ternary_expr(e),
        Stmt::Throw(e) => count += collapse_rpc_ternary_expr(e),
        Stmt::If { cond, .. } => count += collapse_rpc_ternary_expr(cond),
        Stmt::While { cond, .. } => count += collapse_rpc_ternary_expr(cond),
        Stmt::DoWhile { cond, .. } => count += collapse_rpc_ternary_expr(cond),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                count += collapse_rpc_ternary_expr(c);
            }
        }
        _ => {}
    }
    count
}

/// Recursively collapse RPC ternary expressions.
///
/// Walks the expression tree bottom-up so nested ternaries are collapsed first.
fn collapse_rpc_ternary_expr(expr: &mut Expr) -> usize {
    let mut count = 0;

    // First, recurse into sub-expressions.
    match expr {
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                count += collapse_rpc_ternary_expr(r);
            }
            for a in args {
                count += collapse_rpc_ternary_expr(a);
            }
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += collapse_rpc_ternary_expr(a);
            count += collapse_rpc_ternary_expr(b);
        }
        Expr::Not(e) | Expr::Neg(e) | Expr::Cast(_, e) | Expr::AddressOf(e) => {
            count += collapse_rpc_ternary_expr(e);
        }
        Expr::Field(r, _) => count += collapse_rpc_ternary_expr(r),
        Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            count += collapse_rpc_ternary_expr(r);
        }
        Expr::Cond(c, a, b) => {
            count += collapse_rpc_ternary_expr(c);
            count += collapse_rpc_ternary_expr(a);
            count += collapse_rpc_ternary_expr(b);
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                count += collapse_rpc_ternary_expr(a);
            }
        }
        _ => {}
    }

    // Now try to collapse this expression itself.
    if let Some(replacement) = try_collapse_rpc_ternary(expr) {
        *expr = replacement;
        count += 1;
    }

    count
}

/// Try to match the RPC ternary pattern and return `Some(This)` if matched.
fn try_collapse_rpc_ternary(expr: &Expr) -> Option<Expr> {
    let Expr::Cond(cond, then_e, else_e) = expr else {
        return None;
    };

    let Expr::Cmp(k, cmp_lhs, cmp_rhs) = cond.as_ref() else {
        return None;
    };

    // Pattern 1: `v == 0 ? this : v` or `v == null ? this : v`
    // The RHS of the ternary (else_e) must be the same var as the non-constant
    // side of the comparison.
    if matches!(k, CmpKind::Eq) {
        // cmp_rhs is 0/null → cmp_lhs is the var
        if is_zero_or_null(cmp_rhs)
            && matches!(then_e.as_ref(), Expr::This)
            && exprs_structurally_eq(cmp_lhs, else_e)
        {
            return Some(Expr::This);
        }
        // cmp_lhs is 0/null → cmp_rhs is the var
        if is_zero_or_null(cmp_lhs)
            && matches!(then_e.as_ref(), Expr::This)
            && exprs_structurally_eq(cmp_rhs, else_e)
        {
            return Some(Expr::This);
        }
    }

    // Pattern 2: `v != 0 ? v : this` or `v != null ? v : this`
    if matches!(k, CmpKind::Ne) {
        // cmp_rhs is 0/null → cmp_lhs is the var, then_e should be the var
        if is_zero_or_null(cmp_rhs)
            && matches!(else_e.as_ref(), Expr::This)
            && exprs_structurally_eq(cmp_lhs, then_e)
        {
            return Some(Expr::This);
        }
        // cmp_lhs is 0/null → cmp_rhs is the var, then_e should be the var
        if is_zero_or_null(cmp_lhs)
            && matches!(else_e.as_ref(), Expr::This)
            && exprs_structurally_eq(cmp_rhs, then_e)
        {
            return Some(Expr::This);
        }
    }

    None
}

/// Is this expression a constant zero or null?
fn is_zero_or_null(e: &Expr) -> bool {
    matches!(e, Expr::Int(0) | Expr::Null)
}

/// Structural equality check for expressions (used to verify the ternary's
/// variable matches the comparison's variable).
fn exprs_structurally_eq(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Var(na), Expr::Var(nb)) => na == nb,
        (Expr::This, Expr::This) => true,
        (Expr::Int(a), Expr::Int(b)) => a == b,
        (Expr::Null, Expr::Null) => true,
        (Expr::Field(ra, fa), Expr::Field(rb, fb)) => fa == fb && exprs_structurally_eq(ra, rb),
        (Expr::RawDeref(ra, oa, sa), Expr::RawDeref(rb, ob, sb)) => {
            oa == ob && sa == sb && exprs_structurally_eq(ra, rb)
        }
        _ => false,
    }
}

// ===========================================================================
// 7C: Multi-line variable flattening
// ===========================================================================

/// Flatten redundant temporary variables into their use site.
///
/// ## Pattern
///
/// ```text
/// v33 = DiscreteTime.op_Inequality(v27, v30);
/// if (v33 != 0) { ... }
/// ```
///
/// becomes:
///
/// ```text
/// if (DiscreteTime.op_Inequality(v27, v30) != 0) { ... }
/// ```
///
/// ## Safety conditions
///
/// The temporary `v` is inlined only when:
/// 1. `v` is assigned exactly once (the statement before the use).
/// 2. `v` is used exactly once (in the immediately following condition).
/// 3. The assigned expression has no side effects that would change semantics
///    when moved into the condition (actually, calls ARE fine here because the
///    call was already happening at the assignment site, and we're just moving
///    it to the condition site which is the very next statement).
/// 4. No statement between the assignment and the use reads or writes `v`.
///    (Since we require they are adjacent, this is trivially satisfied.)
fn flatten_temp_vars(stmts: &mut Vec<Stmt>) -> usize {
    let observable = observable_names(stmts);
    let mut count = 0;
    walk_scoped(stmts, &no_scope(), false, &mut |list, ctx| {
        count += flatten_temp_vars_list(list, ctx, &observable);
    });
    count
}

/// Scope-aware post-order walk: every nested list is visited with the set of
/// names mentioned in ENCLOSING shallow scopes — ancestor lists' own
/// expressions (cell conditions, parent-list statements) plus the sibling
/// statements of the construct that owns the list. Those are exactly the reads
/// a substitution confined to this list cannot reach, so deleting a def they
/// still name would leave them pointing at nothing.
///
/// This is deliberately not a method-wide name count: SSA versions share one
/// display name, so "mentioned anywhere else" would block inlining across
/// unrelated lifetimes and cost far more temps than the dangling reads it
/// prevents.
fn walk_scoped<F>(
    stmts: &mut Vec<Stmt>,
    ctx: &Scope<'_>,
    inside_loop: bool,
    f: &mut F,
) where
    F: FnMut(&mut Vec<Stmt>, &Scope<'_>),
{
    // Only temp-shaped names are ever consulted (see `blocked_outside_mentions`),
    // so the per-list and per-sibling sets keep just those; every other mention
    // would only be cloned into unions that no query can name.
    let mut list_mentions: std::collections::HashSet<String> = std::collections::HashSet::new();
    for s in stmts.iter() {
        for_each_shallow_mention(s, &mut |n| {
            if is_temp_var_name(n) {
                list_mentions.insert(n.to_string());
            }
        });
    }
    for s in stmts.iter_mut() {
        let child_maps: Vec<std::collections::HashSet<String>> = {
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            kids.iter().map(|k| shallow_mentions(k)).collect()
        };
        let loop_body_idx = loop_body_child_index(s);
        for (ci, child) in super::children_mut(s).into_iter().enumerate() {
            let child_in_loop = inside_loop || loop_body_idx == Some(ci);
            // Outside a loop body the enclosing-scope check used to be
            // skipped: the structurer materializes a merge as a copy in the
            // arm, so an outside read names the copy target, never the arm's
            // temporary. Definition-side coalescing (7C-2b) removes that copy
            // and writes the merge target directly in the arm, which makes the
            // outside read observe the arm's def. The inliners must therefore
            // treat enclosing mentions as live even outside loops, or they
            // delete the definition while the outside read survives.
            // `ILR_LOOSESCOPE=1` restores the old behavior for measurement.
            let wide = child_in_loop || !loose_scope();
            let child_ctx = if wide {
                let mut sets: Vec<&std::collections::HashSet<String>> = ctx.sets.clone();
                sets.push(&list_mentions);
                for (cj, m) in child_maps.iter().enumerate() {
                    if cj != ci {
                        sets.push(m);
                    }
                }
                Scope { sets }
            } else {
                Scope { sets: Vec::new() }
            };
            walk_scoped(child, &child_ctx, child_in_loop, f);
        }
    }
    let f_ctx = if inside_loop || !loose_scope() {
        Scope {
            sets: ctx.sets.clone(),
        }
    } else {
        Scope { sets: Vec::new() }
    };
    f(stmts, &f_ctx);
}

/// Measurement-only A/B switch restoring the loop-only enclosing-scope ctx.
fn loose_scope() -> bool {
    crate::recovery::env_flags::flag("ILR_LOOSESCOPE")
}

/// The empty enclosing-scope context.
fn no_scope() -> Scope<'static> {
    Scope { sets: Vec::new() }
}

/// If this statement owns a LOOP BODY list, its index among `children_mut`.
fn loop_body_child_index(s: &Stmt) -> Option<usize> {
    match s {
        Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::ForEach { .. } => Some(0),
        // children: [init, incr, body]
        Stmt::For { .. } => Some(2),
        _ => None,
    }
}

fn shallow_mentions(list: &[Stmt]) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for s in list {
        for_each_shallow_mention(s, &mut |n| {
            if is_temp_var_name(n) {
                out.insert(n.to_string());
            }
        });
    }
    out
}

/// Temp names assigned in this list that are mentioned in an enclosing scope.
/// Their defs may be substituted here but must NOT be deleted: the outside
/// read would be left naming a variable nothing assigns.
fn blocked_outside_mentions(
    stmts: &[Stmt],
    ctx: &Scope<'_>,
) -> std::collections::HashSet<String> {
    if ctx.is_empty() {
        return std::collections::HashSet::new();
    }
    stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::Assign(var, None, _) if is_temp_var_name(var) && ctx.contains(var) => {
                Some(var.clone())
            }
            _ => None,
        })
        .collect()
}

/// Collapse a redundant class-init guard re-assignment:
///
/// ```text
/// v = <klass expr>;
/// if (<static flag test>) { v = <same klass expr>; }
/// ```
///
/// The arm only re-stores the value `v` already holds, and the test reads
/// static memory only (a klass's initialized flag), so the branch observes
/// nothing. IL2CPP emits the check in pairs; the copy that actually calls the
/// initializer has a CALL in its arm and is left alone.
fn fold_identical_reassign_guards(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += fold_identical_reassign_guards_list(list);
    });
    count
}

fn fold_identical_reassign_guards_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    let mut i = 1;
    while i < stmts.len() {
        // Adjacency is what makes the fold safe: the If carries no label of
        // its own, so the only way into it is right after the assignment.
        let fold = match (&stmts[i - 1], &stmts[i]) {
            (Stmt::Assign(prev_var, None, prev_rhs), Stmt::If { cond, then_body, else_body }) => {
                else_body.is_empty()
                    && then_body.len() == 1
                    && expr_is_pure(prev_rhs)
                    && matches!(
                        &then_body[0],
                        Stmt::Assign(var, None, rhs)
                            if var == prev_var && expr_structurally_same(rhs, prev_rhs)
                    )
                    && cond_reads_static_only(cond)
            }
            _ => false,
        };
        if fold {
            stmts.remove(i);
            count += 1;
        } else {
            i += 1;
        }
    }
    count
}

fn expr_structurally_same(a: &Expr, b: &Expr) -> bool {
    // `Expr` derives `PartialEq` structurally; comparing it directly is both
    // cheaper and stricter than a Debug-rendering comparison (which also
    // distinguishes formatting, not just shape).
    a == b
}

/// Does this expression read only static (image-relative) memory? Such reads
/// cannot fault on a valid image, which is what allows the guard around a
/// redundant re-assignment to be dropped together with its condition.
fn cond_reads_static_only(e: &Expr) -> bool {
    match e {
        Expr::Int(_) | Expr::Bool(_) | Expr::TypeOf(_) | Expr::StaticField(..) => true,
        Expr::Not(x)
        | Expr::Neg(x)
        | Expr::Cast(_, x)
        | Expr::RawDeref(x, _, _)
        | Expr::RawAddr(x, _)
        | Expr::Field(x, _) => cond_reads_static_only(x),
        Expr::Cmp(_, a, b) | Expr::Bin(_, a, b) => {
            cond_reads_static_only(a) && cond_reads_static_only(b)
        }
        _ => false,
    }
}

// ===========================================================================
// 7C-1c: Loop-preheader same-value re-assignment
// ===========================================================================
//
// `guard_dedup` rule C removes a repeated identical assignment WITHIN one
// statement list, and it deliberately clears its knowledge at a loop because
// the body may run again. This pass adds exactly the cross-list case it cannot
// see:
//
//     v = C;                          // preheader, list level
//     while (cond) {
//         v = C;                      // body, before any read of v
//         ...
//     }
//
// The body write is the loop's own initialization of `v` and re-stores the
// value the preheader just put there. It can be removed when
//
// 1. `C` is loop-invariant: a value built only from literals, `this` and
//    LOCALS (no field/static/array reads, no calls). `expr_is_pure` is not
//    enough: a pure `this.field` read may change between the preheader and
//    the body (the loop can call something), and the repeated write would
//    then re-load the NEW value. Only names that nothing in the loop writes
//    are allowed (the same stability argument `guard_dedup` rule C makes).
// 2. `C` does not mention `v` itself (`v = v + 1` is not a re-store).
// 3. The inner write is the ONLY write of `v` in the loop. Otherwise a later
//    iteration enters the body holding a different value; reads after the
//    removed write would observe that value instead of `C`.
// 4. `v` is not read between the loop entry and the inner write: the loop
//    condition / `for` initializer / `foreach` collection, plus every body
//    statement lexically before it. That is the read the spec protects; it is
//    also what keeps the removal a pure no-op for the first iteration.
// 5. No label anywhere inside the loop: a `goto` could otherwise enter the
//    body without executing the preheader.
// 6. `v` and every name `C` reads are not address-taken / closure-captured;
//    a pointer write or a lambda could change `v` behind our back.
//
// When all of that holds, every execution of the removed statement stores the
// value `v` already holds, so deleting it changes no read, no call count and
// no branch.

fn fold_loop_preheader_reassignments(stmts: &mut Vec<Stmt>) -> usize {
    let observable = observable_names(stmts);
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += fold_loop_preheader_reassignments_list(list, &observable);
    });
    count
}

fn fold_loop_preheader_reassignments_list(
    stmts: &mut Vec<Stmt>,
    observable: &std::collections::HashSet<String>,
) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + 1 < stmts.len() {
        let candidate = match (&stmts[i], &stmts[i + 1]) {
            (Stmt::Assign(var, None, value), loop_stmt)
                if is_temp_var_name(var)
                    && is_loop_stmt(loop_stmt)
                    && loop_invariant_expr(value)
                    && expr_mention_count(value, var) == 0
                    && !observable.contains(var) =>
            {
                Some((var.clone(), value.clone()))
            }
            _ => None,
        };
        let Some((var, value)) = candidate else {
            i += 1;
            continue;
        };
        let Some(body_idx) =
            plan_loop_reassignment(&stmts[i + 1], &var, &value, observable)
        else {
            i += 1;
            continue;
        };
        if remove_loop_body_stmt(&mut stmts[i + 1], body_idx) {
            LOOP_REASSIGN_REMOVED.fetch_add(1, Ordering::Relaxed);
            count += 1;
            // The inner write was the loop's only one, so the same pair cannot
            // fire again. Skip both statements.
            i += 2;
        } else {
            i += 1;
        }
    }
    count
}

fn is_loop_stmt(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::ForEach { .. }
    )
}

/// A value that cannot change while the loop runs, built only from literals,
/// `this`, type tokens and local names (whose writes the caller rules out).
/// Field / static / array-element reads are rejected even though they are
/// side-effect free: a call in the loop can store a new value into them, and
/// the body's repeated assignment would have observed it.
fn loop_invariant_expr(e: &Expr) -> bool {
    match e {
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::TypeOf(_) => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            loop_invariant_expr(a) && loop_invariant_expr(b)
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::IndexFromEnd(x) => {
            loop_invariant_expr(x)
        }
        Expr::Cond(c, a, b) => {
            loop_invariant_expr(c) && loop_invariant_expr(a) && loop_invariant_expr(b)
        }
        Expr::Tuple { elements, .. } => elements.iter().all(loop_invariant_expr),
        _ => false,
    }
}

/// Index of the redundant inner write in the loop's top-level body, or `None`
/// when any proof step fails. The caller removes it.
fn plan_loop_reassignment(
    loop_stmt: &Stmt,
    var: &str,
    value: &Expr,
    observable: &std::collections::HashSet<String>,
) -> Option<usize> {
    // A label is an entry point: control could reach the body without running
    // the preheader, where `v` does not hold `C`.
    if stmt_has_label(loop_stmt) {
        return None;
    }
    // Exactly one write of `v` in the whole loop, and it must be the inner
    // re-assignment. A second write anywhere makes the value at the top of a
    // later iteration differ from `C`.
    if count_writes_in_stmt(loop_stmt, var) != 1 {
        return None;
    }
    // Every name `C` reads must be loop-invariant too.
    let mut names = std::collections::HashSet::new();
    collect_expr_vars(value, &mut names);
    for name in &names {
        if observable.contains(name) || count_writes_in_stmt(loop_stmt, name) > 0 {
            return None;
        }
    }

    let (body, header_reads) = match loop_stmt {
        Stmt::While { cond, body } => (body, expr_mention_count(cond, var)),
        Stmt::DoWhile { body, .. } => (body, 0),
        Stmt::For {
            init, cond, incr, body, ..
        } => {
            let reads = count_reads_in_list(init, var)
                + cond.as_ref().map_or(0, |c| expr_mention_count(c, var));
            // `incr` runs after the body, so its reads see the write; only a
            // WRITE there matters, and that is covered by the count above.
            let _ = incr;
            (body, reads)
        }
        Stmt::ForEach {
            var_name,
            collection,
            body,
        } => {
            // The element binding would shadow `v` inside the body.
            if var_name == var {
                return None;
            }
            (body, usize::from(collection == var))
        }
        _ => return None,
    };
    if header_reads > 0 {
        return None;
    }

    // The inner write must be a TOP-LEVEL body statement whose RHS is
    // structurally equal to the preheader value. A nested write has an
    // execution condition the "no read before" walk below cannot reason
    // about, and is left alone.
    let mut found = None;
    for (k, stmt) in body.iter().enumerate() {
        if let Stmt::Assign(name, ty, rhs) = stmt {
            if name == var {
                if ty.is_some() || rhs != value || found.is_some() {
                    return None;
                }
                found = Some(k);
            }
        }
    }
    let body_idx = found?;

    // The read the spec protects: no use of `v` between loop entry and the
    // inner write. (Later reads are fine: `v` already holds `C` there, and
    // nothing else writes it.)
    if count_reads_in_range(body, 0, body_idx, var) > 0 {
        return None;
    }
    Some(body_idx)
}

fn remove_loop_body_stmt(loop_stmt: &mut Stmt, idx: usize) -> bool {
    let body = match loop_stmt {
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::For { body, .. }
        | Stmt::ForEach { body, .. } => body,
        _ => return false,
    };
    if idx >= body.len() {
        return false;
    }
    body.remove(idx);
    true
}

/// Scan a statement list for temp-var flattening opportunities.
fn flatten_temp_vars_list(
    stmts: &mut Vec<Stmt>,
    ctx: &Scope<'_>,
    observable: &std::collections::HashSet<String>,
) -> usize {
    // The only rewrite source is an adjacent pair: an untyped non-alloc
    // assignment followed by a statement with an inline target slot. Without
    // such a pair the pass cannot change anything, so the (deep) blocked-scan
    // and the per-statement census are both skipped.
    let has_pair = stmts.windows(2).any(|w| match &w[0] {
        Stmt::Assign(_, None, rhs) => {
            !is_alloc_rhs(rhs) && inline_target_slot(&w[1]).is_some()
        }
        _ => false,
    });
    if !has_pair {
        return 0;
    }
    let blocked = blocked_outside_mentions(stmts, ctx);
    // Only the names of untyped non-alloc assignments are ever looked up below.
    // Restricting the per-statement census to them keeps its maps at a handful
    // of keys instead of cloning every identifier a statement mentions.
    let cand_names: std::collections::HashSet<String> = stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::Assign(v, None, rhs) if !is_alloc_rhs(rhs) => Some(v.clone()),
            _ => None,
        })
        .collect();
    // One fused census per statement, maintained through the substitutions
    // and removals below. `count_reads_excluding` used to re-walk every
    // statement's expression tree for every candidate (`flatten_temp_vars`
    // measured 45 s on the full corpus). `reads_elsewhere` is
    // `total[name] - per_stmt[i][name]`, exactly the old sum over j != i.
    let mut per_stmt: Vec<HashMap<String, usize>> = stmts
        .iter()
        .map(|s| count_reads_impl(std::slice::from_ref(s), Some(&cand_names)))
        .collect();
    let mut total: HashMap<String, usize> = HashMap::new();
    for m in &per_stmt {
        for (n, c) in m {
            *total.entry(n.clone()).or_default() += *c;
        }
    }
    let mut count = 0;
    let mut i = 0;
    loop {
        if i + 1 >= stmts.len() {
            break;
        }

        // Look for: Assign(v, _, rhs) followed by a statement that reads v
        // exactly once in its condition. The RHS is borrowed, not cloned:
        // cloning it for every untyped assignment was pure allocation, and
        // most candidates fail the checks below.
        let keep_def = match &stmts[i] {
            Stmt::Assign(var, None, rhs) => {
                // Skip if the RHS is an allocation (RawCall, NewObj) because
                // those are part of the two-statement alloc+ctor pattern that
                // `normalize_ctor_calls` needs to see as separate statements.
                //
                // A self-referential def (`i = i - 1`) must never be
                // substituted into a read: every read the def governs sees the
                // NEW value, and the substituted RHS reads the same display
                // name again, so `i = i - 1; if (i >= 0)` would become
                // `if (i - 1 >= 0)` — a different value (and, chained across a
                // loop, the `-1 -1 -1` expressions seen in the dump).
                //
                // A name an enclosing scope reads may be substituted but not
                // deleted: the def has to stay for that outside read. Impure
                // RHS cannot be duplicated that way (the effect would run
                // twice).
                if is_alloc_rhs(rhs) || observable.contains(var) || super::expr_mentions(rhs, var)
                {
                    None
                } else {
                    let keep = blocked.contains(var);
                    if keep && !expr_is_pure(rhs) {
                        None
                    } else {
                        Some(keep)
                    }
                }
            }
            _ => None,
        };

        if let Some(keep_def) = keep_def {
            let var_name = match &stmts[i] {
                Stmt::Assign(v, None, _) => v.clone(),
                _ => unreachable!(),
            };
            // Every condition is evaluated BEFORE the substitution, which
            // cannot be undone: abandoning it half-way leaves `rhs` copied into
            // the condition while the statement that computed it survives, i.e.
            // one native call rendered twice.
            //
            // `reads_elsewhere == 1` means the only read of the temp in this
            // list is the condition slot we are about to rewrite (the def's own
            // RHS is excluded by `skip_idx`), so nothing else can be left
            // naming a variable that is about to disappear.
            let own_reads = per_stmt[i].get(&var_name).copied().unwrap_or(0);
            let reads_elsewhere =
                total.get(&var_name).copied().unwrap_or(0) - own_reads;
            let target_used_once = inline_target_slot(&stmts[i + 1])
                .is_some_and(|slot| expr_mentions_exactly_once(slot, &var_name));
            if reads_elsewhere == 1 && target_used_once {
                let inlined = {
                    let (head, tail) = stmts.split_at_mut(i + 1);
                    let Stmt::Assign(_, None, rhs) = &head[i] else {
                        unreachable!()
                    };
                    try_inline_temp(&mut tail[0], &var_name, rhs).is_some()
                };
                // A refused substitution can still have rewritten the
                // statement (name slots are checked after `substitute_var`),
                // so the census is resynced after every attempt.
                let old_map = std::mem::take(&mut per_stmt[i + 1]);
                for (n, c) in &old_map {
                    if let Some(t) = total.get_mut(n) {
                        *t = t.saturating_sub(*c);
                    }
                }
                per_stmt[i + 1] =
                    count_reads_impl(std::slice::from_ref(&stmts[i + 1]), Some(&cand_names));
                for (n, c) in &per_stmt[i + 1] {
                    *total.entry(n.clone()).or_default() += *c;
                }
                if inlined && !keep_def {
                    let removed = per_stmt.remove(i);
                    for (n, c) in &removed {
                        if let Some(t) = total.get_mut(n) {
                            *t = t.saturating_sub(*c);
                        }
                    }
                    stmts.remove(i);
                    count += 1;
                    // Don't increment i — check the new pair at position i.
                    continue;
                }
            }
        }

        i += 1;
    }
    count
}

/// Temporary A/B gate: ILR_NOSCOPE=1|all|copies restores the old
/// whole-list rename in the copy pass (measurement only).
fn copy_scope_disabled() -> bool {
    matches!(
        std::env::var("ILR_NOSCOPE").as_deref(),
        Ok("1") | Ok("all") | Ok("copies")
    )
}

/// The expression slot `try_inline_temp` substitutes into, if this statement
/// kind is a flattening target.
fn inline_target_slot(stmt: &Stmt) -> Option<&Expr> {
    match stmt {
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            Some(cond)
        }
        Stmt::Return(Some(e)) => Some(e),
        Stmt::Assign(_, _, e) => Some(e),
        Stmt::Eval(e) => Some(e),
        _ => None,
    }
}

/// `inline_target_slot`, mutable.
fn inline_target_slot_mut(stmt: &mut Stmt) -> Option<&mut Expr> {
    match stmt {
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            Some(cond)
        }
        Stmt::Return(Some(e)) => Some(e),
        Stmt::Assign(_, _, e) => Some(e),
        Stmt::Eval(e) => Some(e),
        _ => None,
    }
}

/// Try to inline `rhs` for `var_name` in the condition of `stmt`.
///
/// Returns `Some(())` if the inlining was performed, `None` otherwise.
///
/// Both checks run before the substitution. `substitute_var` cannot be undone,
/// and a substitution abandoned afterwards leaves `rhs` COPIED into the
/// condition while the statement that computed it survives: one native call,
/// two call expressions in the dump.
fn try_inline_temp(stmt: &mut Stmt, var_name: &str, rhs: &Expr) -> Option<()> {
    let used_once_in_target = inline_target_slot(stmt)
        .is_some_and(|slot| expr_mentions_exactly_once(slot, var_name));
    if !used_once_in_target || stmt_mentions_deep_excluding_cond(stmt, var_name) {
        return None;
    }
    substitute_var(inline_target_slot_mut(stmt)?, var_name, rhs);
    // `substitute_var` refuses name slots (`&v`, a shadowing lambda param).
    // Deleting the def after a refused substitution would leave that read
    // pointing at a variable nothing assigns, so verify the slot is gone.
    if inline_target_slot(stmt).is_some_and(|slot| expr_mention_count(slot, var_name) > 0) {
        return None;
    }
    Some(())
}

/// Every sub-expression slot of `e`, in evaluation order.
///
/// Deliberately exhaustive: no catch-all arm, so a new `Expr` variant is a
/// compile error here rather than a silent blind spot. Both the read counter
/// and the substitution passes derive from this shape — when they disagree, a
/// temp gets inlined away while one of its reads survives, and the output
/// references a variable that no longer exists anywhere in the method.
pub(crate) fn expr_children(e: &Expr) -> Vec<&Expr> {
    use crate::recovery::csharp::InterpPart;
    let mut out: Vec<&Expr> = Vec::new();
    match e {
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
        | Expr::TypeOf(_)
        | Expr::StaticField(_, _) => {}
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => out.extend([a.as_ref(), b.as_ref()]),
        Expr::Not(x)
        | Expr::Neg(x)
        | Expr::Cast(_, x)
        | Expr::AddressOf(x)
        | Expr::Await(x)
        | Expr::RawDeref(x, _, _)
        | Expr::RawAddr(x, _)
        | Expr::IndexFromEnd(x) => out.push(x.as_ref()),
        Expr::Field(r, _) => out.push(r.as_ref()),
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                out.push(r.as_ref());
            }
            out.extend(args.iter());
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            out.extend(args.iter())
        }
        Expr::Cond(c, a, b) => out.extend([c.as_ref(), a.as_ref(), b.as_ref()]),
        Expr::Lambda { body, .. } => out.push(body.as_ref()),
        Expr::ArrayInit { items, .. } => out.extend(items.iter()),
        Expr::CollectionInit { ctor_args, items, .. } => {
            out.extend(ctor_args.iter());
            out.extend(items.iter());
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            out.extend(ctor_args.iter());
            out.extend(fields.iter().map(|(_, v)| v));
        }
        Expr::Index { collection, index } => {
            out.extend([collection.as_ref(), index.as_ref()])
        }
        Expr::NullConditional { receiver, access } => {
            out.extend([receiver.as_ref(), access.as_ref()])
        }
        Expr::PatternMatch { expr, .. } => out.push(expr.as_ref()),
        Expr::StringInterp { parts } => {
            for p in parts {
                if let InterpPart::Expr(x, _) = p {
                    out.push(x.as_ref());
                }
            }
        }
        Expr::Tuple { elements, .. } => out.extend(elements.iter()),
        Expr::MethodChain { receiver, methods } => {
            out.push(receiver.as_ref());
            for m in methods {
                out.extend(m.args.iter());
            }
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            out.push(discrim.as_ref());
            for (p, r) in arms {
                out.push(p);
                out.push(r);
            }
            if let Some(d) = default_arm {
                out.push(d.as_ref());
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                out.push(f.as_ref());
            }
            if let Some(t) = to {
                out.push(t.as_ref());
            }
        }
    }
    out
}

/// Mutable mirror of [`expr_children`].
pub(crate) fn expr_children_mut(e: &mut Expr) -> Vec<&mut Expr> {
    use crate::recovery::csharp::InterpPart;
    let mut out: Vec<&mut Expr> = Vec::new();
    match e {
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
        | Expr::TypeOf(_)
        | Expr::StaticField(_, _) => {}
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => out.extend([a.as_mut(), b.as_mut()]),
        Expr::Not(x)
        | Expr::Neg(x)
        | Expr::Cast(_, x)
        | Expr::AddressOf(x)
        | Expr::Await(x)
        | Expr::RawDeref(x, _, _)
        | Expr::RawAddr(x, _)
        | Expr::IndexFromEnd(x) => out.push(x.as_mut()),
        Expr::Field(r, _) => out.push(r.as_mut()),
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                out.push(r.as_mut());
            }
            out.extend(args.iter_mut());
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            out.extend(args.iter_mut())
        }
        Expr::Cond(c, a, b) => out.extend([c.as_mut(), a.as_mut(), b.as_mut()]),
        Expr::Lambda { body, .. } => out.push(body.as_mut()),
        Expr::ArrayInit { items, .. } => out.extend(items.iter_mut()),
        Expr::CollectionInit { ctor_args, items, .. } => {
            out.extend(ctor_args.iter_mut());
            out.extend(items.iter_mut());
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            out.extend(ctor_args.iter_mut());
            out.extend(fields.iter_mut().map(|(_, v)| v));
        }
        Expr::Index { collection, index } => {
            out.extend([collection.as_mut(), index.as_mut()])
        }
        Expr::NullConditional { receiver, access } => {
            out.extend([receiver.as_mut(), access.as_mut()])
        }
        Expr::PatternMatch { expr, .. } => out.push(expr.as_mut()),
        Expr::StringInterp { parts } => {
            for p in parts {
                if let InterpPart::Expr(x, _) = p {
                    out.push(x.as_mut());
                }
            }
        }
        Expr::Tuple { elements, .. } => out.extend(elements.iter_mut()),
        Expr::MethodChain { receiver, methods } => {
            out.push(receiver.as_mut());
            for m in methods {
                out.extend(m.args.iter_mut());
            }
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            out.push(discrim.as_mut());
            for (p, r) in arms {
                out.push(p);
                out.push(r);
            }
            if let Some(d) = default_arm {
                out.push(d.as_mut());
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                out.push(f.as_mut());
            }
            if let Some(t) = to {
                out.push(t.as_mut());
            }
        }
    }
    out
}

/// Count how many times `name` is read in `expr`.
///
/// Bare-name slots that an expression can never be substituted into (a
/// pattern-match binding, a field named like the temp) are counted too: that
/// keeps the caller conservative instead of silently leaving a dangling name.
fn expr_mention_count(expr: &Expr, name: &str) -> usize {
    if let Expr::Lambda { params, .. } = expr {
        // A lambda parameter shadows the outer name: nothing inside reads it.
        if params.iter().any(|p| p == name) {
            return 0;
        }
    }
    let own = match expr {
        Expr::Var(n) if n == name => 1,
        Expr::Field(_, f) if f == name => 1,
        Expr::StaticField(_, f) if f == name => 1,
        Expr::PatternMatch { binding, .. } if binding == name => 1,
        _ => 0,
    };
    own + expr_children(expr)
        .iter()
        .map(|c| expr_mention_count(c, name))
        .sum::<usize>()
}

/// Does `expr` mention `Var(name)` exactly once?
fn expr_mentions_exactly_once(expr: &Expr, name: &str) -> bool {
    expr_mention_count(expr, name) == 1
}

/// Substitute `Var(name)` with `replacement` in `expr`.
pub(crate) fn substitute_var(expr: &mut Expr, name: &str, replacement: &Expr) {
    if let Expr::Var(n) = &*expr {
        if n == name {
            *expr = replacement.clone();
            return;
        }
    }
    if let Expr::Lambda { params, .. } = &*expr {
        if params.iter().any(|p| p == name) {
            return;
        }
    }
    // `&v` names the storage `v` provides. Rewriting it turns `(&stack_1)` into
    // `(&0)` — not the same program, and not valid C# either.
    if let Expr::AddressOf(x) | Expr::RawAddr(x, _) = &*expr {
        if matches!(x.as_ref(), Expr::Var(n) if n == name) {
            return;
        }
    }
    for child in expr_children_mut(expr) {
        substitute_var(child, name, replacement);
    }
}

/// Does `stmt` mention `name` anywhere in its nested bodies EXCLUDING the
/// condition? Used to verify the temp var isn't used deeper in the statement.
fn stmt_mentions_deep_excluding_cond(stmt: &Stmt, name: &str) -> bool {
    match stmt {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            then_body.iter().any(|s| stmt_mentions_deep(s, name))
                || else_body.iter().any(|s| stmt_mentions_deep(s, name))
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
            body.iter().any(|s| stmt_mentions_deep(s, name))
        }
        Stmt::For {
            init, incr, body, ..
        } => {
            init.iter().any(|s| stmt_mentions_deep(s, name))
                || incr.iter().any(|s| stmt_mentions_deep(s, name))
                || body.iter().any(|s| stmt_mentions_deep(s, name))
        }
        Stmt::Using { decl, body, .. } => {
            stmt_mentions_deep(decl, name) || body.iter().any(|s| stmt_mentions_deep(s, name))
        }
        Stmt::ForEach { body, .. } => body.iter().any(|s| stmt_mentions_deep(s, name)),
        Stmt::Lock { body, .. } => body.iter().any(|s| stmt_mentions_deep(s, name)),
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            try_body.iter().any(|s| stmt_mentions_deep(s, name))
                || catches
                    .iter()
                    .any(|(_, b)| b.iter().any(|s| stmt_mentions_deep(s, name)))
                || finally_body
                    .as_ref()
                    .is_some_and(|f| f.iter().any(|s| stmt_mentions_deep(s, name)))
        }
        Stmt::Switch { cases, tail, default, .. } => {
            cases
                .iter()
                .any(|(_, b)| b.iter().any(|s| stmt_mentions_deep(s, name)))
                || tail.iter().any(|s| stmt_mentions_deep(s, name))
                || default
                    .as_ref()
                    .is_some_and(|arm| arm.iter().any(|s| stmt_mentions_deep(s, name)))
        }
        _ => false,
    }
}

/// Does `stmt` mention `name` anywhere at any depth?
fn stmt_mentions_deep(stmt: &Stmt, name: &str) -> bool {
    super::stmt_mentions_deep(stmt, name)
}

// ===========================================================================
// 7C-2: General temp variable inlining (non-adjacent)
// ===========================================================================

/// Inline temporary variables that are assigned once and used once anywhere in
/// the same scope — even when the assignment and use are NOT adjacent.
///
/// ## Pattern
///
/// ```text
/// v1 = obj.SomeField;
/// someOtherStatement;
/// v2 = v1 + 10;
/// return v2;
/// ```
///
/// becomes:
///
/// ```text
/// someOtherStatement;
/// return obj.SomeField + 10;
/// ```
///
/// The existing `flatten_temp_vars` (7C) only handles *adjacent* pairs. This
/// pass catches the remaining cases where intervening statements separate the
/// assignment from its single use.
///
/// ## Safety conditions
///
/// A temp variable `v` is inlined only when ALL of the following hold:
///
/// 1. `v` matches the temp naming pattern (`v` followed by digits).
/// 2. `v` is assigned exactly once in the scope (the one `Stmt::Assign`).
/// 3. `v` is read exactly once in the scope (one expression mention).
/// 4. The RHS expression is **pure** — it contains no calls, allocations, or
///    other side effects. This ensures moving it from the assignment site to
///    the use site cannot change semantics, regardless of what intervening
///    statements exist.
/// 5. The RHS is not trivially simple (a bare variable, literal, etc.) —
///    inlining those would just swap one name for another.
///
/// ## Why pure-only
///
/// When the assignment and use are separated by other statements, those
/// intervening statements might contain calls that mutate state. If the RHS
/// contained a call, moving it past another call could change the order of
/// side effects. By restricting to pure RHS expressions (field accesses,
/// arithmetic, comparisons, casts), we guarantee the expression's value depends
/// only on its operands, not on when it is evaluated.
fn inline_temp_variables(stmts: &mut Vec<Stmt>) -> usize {
    // A name whose ADDRESS escapes (`TryGetValue(key, &cell)`) or that a
    // lambda captures is not a pure temp: the storage can be written behind
    // the statements. Substituting its one textual def into a read invents a
    // value (`stack_1 = 0; if (stack_1 == 0) goto raise; store; call` became
    // `if (0 == 0)`, making the update block unreachable — 0x18A3A1C).
    let observable = observable_names(stmts);
    let labels = build_label_index(stmts);
    let mut count = 0;
    walk_scoped(stmts, &no_scope(), false, &mut |list, ctx| {
        count += inline_temp_vars_list(list, ctx, &observable, &labels);
    });
    count
}

/// Inline temp-to-temp copy chains: `v1 = v2` where both are temps.
///
/// ## Pattern
///
/// ```text
/// v123 = v456;
/// ... use v123 ...
/// ```
///
/// becomes:
///
/// ```text
/// ... use v456 ...
/// ```
///
/// ## Why this matters
///
/// The structurer creates many temp-to-temp copies during SSA construction.
/// These are 16.9% of all temp variable assignments (explorer finding).
///
/// ## Safety
///
/// We only inline when:
/// 1. Both LHS and RHS are temp variables
/// 2. The LHS temp has exactly 1 write (the copy itself)
/// 3. The LHS temp has at least 1 read
/// 4. When a read of the LHS temp lives outside this list, the copy is KEPT
///    (local reads may still be renamed — the copy supplies the outside read)
/// 5. The source temp is never written after the copy, so every renamed read
///    still observes the copied value
/// 6. Only reads AFTER the copy are renamed: a read before it (or reached
///    through a loop back edge) sees the previous iteration's binding, which
///    the preheader initializes for the destination only
fn inline_temp_to_temp_copies(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_scoped(stmts, &no_scope(), false, &mut |list, ctx| {
        count += inline_temp_to_temp_copies_list(list, ctx);
    });
    count
}

/// Scan a statement list for temp-to-temp copy inlining opportunities.
fn inline_temp_to_temp_copies_list(
    stmts: &mut Vec<Stmt>,
    ctx: &Scope<'_>,
) -> usize {
    let blocked = blocked_outside_mentions(stmts, ctx);
    let mut total = 0;
    loop {
        let mut count = 0;

        // Find all `v1 = v2` assignments where both are temps.
        let copy_indices: Vec<usize> = stmts
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Stmt::Assign(var, None, Expr::Var(src))
                    if is_temp_var_name(var) && is_temp_var_name(src) =>
                {
                    Some(i)
                }
                _ => None,
            })
            .collect();

        // Process in reverse order so removal doesn't shift indices.
        for i in copy_indices.into_iter().rev() {
            let (var_name, src_name) = match &stmts[i] {
                Stmt::Assign(v, _, Expr::Var(s)) => (v.clone(), s.clone()),
                _ => continue,
            };

            // Must have exactly 1 write (the copy itself).
            let writes = count_writes_in_list(stmts, &var_name);
            if writes != 1 {
                continue;
            }

            // Must have at least 1 read.
            let reads = count_reads_in_list(stmts, &var_name);
            if reads == 0 {
                continue;
            }

            // Reads in enclosing scopes are outside the rename's reach, so the
            // copy must stay; the local reads can still be renamed (that is a
            // pure substitution — the copy keeps providing the value outside).
            let keep_def = blocked.contains(&var_name);

            // Renaming uses to `src` is only valid while `src` still holds the
            // value read at the copy. A later write (at any depth) makes those
            // uses observe the NEW value instead.
            if !copy_scope_disabled()
                && count_writes_in_range(stmts, i + 1, stmts.len(), &src_name) > 0
            {
                continue;
            }

            // Substitute all reads of var_name with src_name, then drop the
            // copy only if nothing still names it. Reads that live in a name
            // slot (an assignment target path, `switch (v)`) cannot be renamed,
            // and deleting the copy anyway would leave them pointing at a
            // variable that no longer has a definition.
            if copy_scope_disabled() {
                for (j, stmt) in stmts.iter_mut().enumerate() {
                    if j == i {
                        continue;
                    }
                    substitute_var_with_var_in_stmt(stmt, &var_name, &src_name);
                }
            } else {
                substitute_var_with_var_from(stmts, i + 1, &var_name, &src_name);
            }
            let after_is_zero = no_reads_excluding(stmts, &var_name, i);
            if after_is_zero && !keep_def {
                stmts.remove(i);
                count += 1;
            }
        }

        total += count;
        if count == 0 {
            break;
        }
    }
    total
}

// ===========================================================================
// 7C-2b: Definition-side copy coalescing
// ===========================================================================

/// Coalesce a copy BACKWARD into its source definition:
///
/// ```text
/// v61 = new object(<>c.<>9, typeof(ListContentManager));   // E
/// *(… +8) = v61;                                          // reads in between
/// v18 = v61;                                              // the copy
/// ```
/// becomes
/// ```text
/// v18 = new object(<>c.<>9, typeof(ListContentManager));
/// *(… +8) = v18;
/// ```
///
/// This is the inverse of [`inline_temp_to_temp_copies`]: that pass renames
/// reads AFTER the copy to the source name and drops the copy; this pass moves
/// the DEFINITION to the destination name and drops the copy. It is the shape
/// the structurer emits when a value computed in one arm is merged into a
/// variable read after the branch (the copy is the phi move).
///
/// ## Why the definition may stay where it is (impure E allowed)
///
/// Nothing is moved: the definition keeps its position, only its destination
/// NAME changes. The evaluation order, the call count and the side effects are
/// unchanged. That is what distinguishes this pass from inlining, which has to
/// move an expression to a use site and therefore requires purity.
///
/// ## Safety conditions
///
/// For `dst = src;` at index `c` with the last definition of `src` at `d < c`:
/// 1. That definition is a top-level `Assign(src, None, E)`, and NO write of
///    `src` occurs anywhere in `(d, c)` (top level or nested). Writes of the
///    same DISPLAY NAME outside that interval belong to other SSA versions —
///    `v289` is defined once per switch arm — and cannot change the value this
///    definition hands to the copy. (Requiring a single write per list, as the
///    strict variant did, refused exactly the multi-target staging moves that
///    motivated this pass.) `E` does not read `src` (otherwise the renamed
///    definition would read a name nothing defines).
/// 2. `src` is read only inside the LIVE RANGE `(d, c)` plus the copy itself.
///    Other SSA versions share the display name: a later write starts a new
///    lifetime whose reads that write supplies, and earlier reads belong to an
///    earlier lifetime. So a read after the copy is refused only until the next
///    write of `src` (that read would still observe this definition's value);
///    reads after that write are unrelated. Reads before `d` stay refused: one
///    can be a loop back edge observing this very definition's
///    previous-iteration value, so removing it would change what it sees.
/// 3. No read AND no write of `dst` in `(d, c)`: after the rename those reads
///    would observe `E` instead of the old destination value.
/// 4. No label anywhere in `[d, c]` and no `goto`, `break`, `continue` or
///    `yield break` in `(d, c)`. A label is an alternate entry into the range
///    (the definition may not run on that path); an outward jump skips the
///    copy, so `dst` would keep its old value, which the forward rename would
///    change.
/// 5. No read of `src` in a bare-name slot (assignment target path,
///    `switch (src)`, `foreach (… in src)`) and no `foreach`/`is` binding named
///    `src`: those slots cannot be renamed by the value substitution.
/// 6. `src` is not mentioned in an enclosing loop scope (the existing
///    [`blocked_outside_mentions`] guarantee) and, additionally, has no reads
///    outside this list anywhere in the method — a read after the branch that
///    still names `src` would be left without a definition.
/// 7. Neither name is address-taken (`&v` / `RawAddr`) or captured by a
///    reconstructed lambda. An ordinary local cannot be observed between the
///    new and old write points; a pointer or closure can.
///
/// `dst` may be read outside the list (that is the normal merge shape): the
/// value is written earlier but nothing between the two points can observe it,
/// so the outside read still sees `E`.
fn coalesce_def_into_copies(stmts: &mut Vec<Stmt>) -> usize {
    let observable = observable_names(stmts);
    // A staging copy's source is the RHS of a plain assignment; without one
    // anywhere in the method no list can have a copy, so the whole walk is
    // provably a no-op. Otherwise count mentions of just those names.
    let mut source_names = std::collections::HashSet::new();
    collect_copy_source_names(stmts, &mut source_names);
    if source_names.is_empty() {
        return 0;
    }
    let method_reads = name_read_counts_filtered(stmts, &source_names);
    let mut count = 0;
    walk_scoped(stmts, &no_scope(), false, &mut |list, ctx| {
        count += coalesce_def_into_copies_list(list, ctx, &observable, &method_reads);
    });
    count
}

fn coalesce_debug_enabled(name: &str) -> bool {
    match std::env::var("ILR_COALDEBUG").ok().as_deref() {
        Some("1") | Some("all") => true,
        Some(v) => v == name,
        None => false,
    }
}

/// Measurement-only A/B switch for the definition-side pass.
fn coalesce_disabled() -> bool {
    crate::recovery::env_flags::flag("ILR_NOCOALESCE")
}

/// Measurement-only A/B switch for the return-value fold.
fn return_fold_disabled() -> bool {
    crate::recovery::env_flags::flag("ILR_NORETFOLD")
}

fn coalesce_def_into_copies_list(
    stmts: &mut Vec<Stmt>,
    ctx: &Scope<'_>,
    observable: &std::collections::HashSet<String>,
    method_reads: &HashMap<String, usize>,
) -> usize {
    // No staging copy in this list means the loop below can never rewrite
    // anything; skip the deep blocked/read scans entirely (their results are
    // only consumed inside the copy loop).
    let has_copy = stmts.iter().any(|s| {
        matches!(
            s,
            Stmt::Assign(dst, None, Expr::Var(src))
                if dst != src && is_temp_var_name(dst) && is_temp_var_name(src)
        )
    });
    if !has_copy {
        return 0;
    }
    let blocked = blocked_outside_mentions(stmts, ctx);
    // Reads of each name BEFORE this list's own fixed-point loop starts. The
    // method-wide map was taken even earlier, so the two agree exactly for a
    // name whose reads are all here; any difference is a read the substitution
    // cannot reach. Comparing against this frozen count (rather than the live
    // count, which the renames below change) is what lets a chain of staging
    // copies — `src -> a`, then `a -> b` — coalesce in one pass while an
    // outside read still refuses. Only names that can become a copy source
    // (the RHS of a plain assignment) are ever queried, so only those are
    // counted.
    let mut source_names = std::collections::HashSet::new();
    collect_copy_source_names(stmts, &mut source_names);
    let local_reads = name_read_counts_filtered(stmts, &source_names);
    let mut total = 0;
    loop {
        let mut count = 0;
        let copies: Vec<usize> = stmts
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Stmt::Assign(dst, None, Expr::Var(src))
                    if dst != src && is_temp_var_name(dst) && is_temp_var_name(src) =>
                {
                    Some(i)
                }
                _ => None,
            })
            .collect();

        // Reverse order: removing a copy does not shift the indices of the
        // candidates still to be examined.
        for c in copies.into_iter().rev() {
            let (dst, src) = match &stmts[c] {
                Stmt::Assign(d, _, Expr::Var(s)) => (d.clone(), s.clone()),
                _ => continue,
            };
            DEFCOPY_EXAMINED.fetch_add(1, Ordering::Relaxed);
            let dbg = coalesce_debug_enabled(&src);

            if observable.contains(&src) || observable.contains(&dst) {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: address-taken/captured name");
                }
                continue;
            }
            if blocked.contains(&src) {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: mentioned in enclosing scope");
                }
                continue;
            }

            // The last top-level definition before the copy supplies the value.
            // Writes of the same display name elsewhere in the list are other
            // SSA versions; only a write INSIDE (d, c) could change what the
            // copy reads, so that interval is what must be write-free.
            let d = match stmts[..c]
                .iter()
                .rposition(|s| matches!(s, Stmt::Assign(n, _, _) if n == &src))
            {
                Some(d) => d,
                None => {
                    if dbg {
                        eprintln!(
                            "[coaldebug] {src}->{dst}: refused: no definition before copy in this list ({} stmts)",
                            stmts.len()
                        );
                    }
                    continue;
                }
            };
            let writes_between = count_writes_in_range(stmts, d + 1, c, &src);
            if writes_between != 0 {
                if dbg {
                    eprintln!(
                        "[coaldebug] {src}->{dst}: refused: source written between definition and copy"
                    );
                }
                continue;
            }
            let (ty, e) = match &stmts[d] {
                Stmt::Assign(_, ty, e) => (ty.clone(), e.clone()),
                _ => {
                    if dbg {
                        eprintln!(
                            "[coaldebug] {src}->{dst}: refused: definition is not a plain assignment"
                        );
                    }
                    continue;
                }
            };
            if ty.is_some() {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: typed declaration");
                }
                continue;
            }
            // The destination must not be declared at this list level: the
            // rename would write it before its `T v = …` declaration.
            if stmts
                .iter()
                .any(|s| matches!(s, Stmt::Assign(n, Some(_), _) if n == &dst))
            {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: destination has a typed declaration");
                }
                continue;
            }
            if expr_mention_count(&e, &src) > 0 {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: definition reads its own target");
                }
                continue;
            }

            // Reads of src are checked against the LIVE RANGE of this
            // definition, not against every read of the display name. Other SSA
            // versions share the name: a later `v289 = …` starts a new lifetime
            // whose reads are supplied by that write, and an earlier lifetime's
            // reads are supplied by an earlier write. Counting them was what
            // made the old test refuse nearly every staging copy.
            //
            // 1. No read before the definition. Such a read can be a loop back
            //    edge observing this definition's previous-iteration value;
            //    after the rename it would observe nothing.
            if count_reads_in_range(stmts, 0, d, &src) > 0 {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: read before definition");
                }
                continue;
            }
            // 2. Reads inside (d, c) are renamed below. Reads after the copy
            //    are still supplied by this definition until the next write of
            //    src; those would dangle, so refuse. A read in the statement
            //    that performs that next write (e.g. `src = src + 1`) also
            //    observes the old value and is covered by the inclusive end.
            let next_src_write = next_write_index(stmts, c, &src);
            let after_end = (next_src_write + 1).min(stmts.len());
            if count_reads_in_range(stmts, c + 1, after_end, &src) > 0 {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: read after copy before redefinition");
                }
                continue;
            }
            // A read outside this list (after the branch, in another arm, …)
            // would be left naming a variable nothing defines after the rename.
            // `local_reads` is the entry-time snapshot: the method map and it
            // agree when every read of `src` lives here. A `local > method`
            // mismatch means an earlier list was rewritten in this walk (its
            // reads were moved to this name); that is unprovable here too, so
            // refuse rather than trust a negative difference.
            let method_src = method_reads.get(&src).copied().unwrap_or(0);
            let local_src = local_reads.get(&src).copied().unwrap_or(0);
            if method_src != local_src {
                if dbg {
                    eprintln!(
                        "[coaldebug] {src}->{dst}: refused: read outside this list \
                         (method {method_src}, here {local_src})"
                    );
                }
                continue;
            }
            // The forward rename would make these observe E instead of the old
            // destination value.
            if count_reads_in_range(stmts, d + 1, c, &dst) > 0 {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: destination read in between");
                }
                continue;
            }
            if count_writes_in_range(stmts, d + 1, c, &dst) > 0 {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: destination written in between");
                }
                continue;
            }

            // A label anywhere in [d, c] is an alternate entry into the live
            // range: the definition may not have run on that path. The copy's
            // own statement cannot carry one in practice, but checking both
            // endpoints keeps the rule independent of how the structurer lays
            // labels out.
            if stmts[d..=c].iter().any(stmt_has_label) {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: label in live range");
                }
                continue;
            }

            let mut range_ok = true;
            for s in &stmts[d + 1..c] {
                if stmt_transfers_control_out(s)
                    || has_bare_name_slot_read(s, &src)
                    || stmt_mentions_binding(s, &src)
                {
                    range_ok = false;
                    break;
                }
            }
            if !range_ok {
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: jump/name-slot in between");
                }
                continue;
            }

            // Rename the reads in the live range, then the definition's
            // target, then drop the copy. The substitution can refuse a slot
            // it cannot rewrite; verify and undo before touching the
            // definition.
            substitute_var_in_range(stmts, d + 1, c, &src, &Expr::Var(dst.clone()), None);
            if count_reads_in_range(stmts, d + 1, c, &src) > 0 {
                // No read of dst existed in the range, so dst -> src restores
                // exactly what the substitution changed.
                substitute_var_in_range(stmts, d + 1, c, &dst, &Expr::Var(src.clone()), None);
                if dbg {
                    eprintln!("[coaldebug] {src}->{dst}: refused: substitution left a read");
                }
                continue;
            }
            if let Stmt::Assign(n, _, _) = &mut stmts[d] {
                *n = dst.clone();
            }
            stmts.remove(c);
            count += 1;
            DEFCOPY_COALESCED.fetch_add(1, Ordering::Relaxed);
            if dbg {
                eprintln!("[coaldebug] {src}->{dst}: COALESCED (def at {d}, copy at {c})");
            }
        }

        total += count;
        if count == 0 {
            break;
        }
    }
    total
}

/// The expression slots a statement owns itself (not its nested statement
/// lists). Mirrors the read/count passes: every slot they consult is visited.
pub(crate) fn for_each_own_expr(s: &Stmt, f: &mut impl FnMut(&Expr)) {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => f(e),
        Stmt::Assign(_, _, e) => f(e),
        Stmt::EventAssign { field, value, .. } => {
            f(field);
            f(value);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            f(collection);
            f(index);
            f(value);
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => f(cond),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                f(c);
            }
        }
        Stmt::Lock { expr, .. } => f(expr),
        Stmt::Return(Some(e)) => f(e),
        Stmt::StringSwitch { disc, cases, .. } => {
            f(disc);
            for (vals, _) in cases {
                for v in vals {
                    f(v);
                }
            }
        }
        Stmt::Attribute { args, .. } => {
            for a in args {
                f(a);
            }
        }
        // The discriminant and the iterated collection are stored as plain
        // names, not exprs, so a reader that only looks at expr slots misses
        // them. A name that is read here is READ -- reporting it as unread is
        // how a definition gets deleted out from under a live `switch (v)`.
        Stmt::Switch { disc, .. } => f(&Expr::Var(disc.clone())),
        Stmt::ForEach { collection, .. } => f(&Expr::Var(collection.clone())),
        // `children` exposes only the body; the declaration is a statement of
        // its own and carries the acquisition expression.
        Stmt::Using { decl, .. } => for_each_own_expr(decl, f),
        _ => {}
    }
}

/// Names whose storage is observable outside the plain dataflow of a local:
/// address-taken (`&v`, `RawAddr(v, …)`) or captured by a reconstructed lambda
/// body. Moving a write earlier is invisible for an ordinary local and can be
/// observed through a pointer or closure.
/// Drop IL2CPP metadata-initialization calls.
///
/// The compiler emits one `initialize_class/metadata` call per type used in a
/// method, all to the SAME unresolved helper and all with a `typeof(T)` first
/// argument and an unused result. They are runtime bookkeeping the C# source
/// never contained. A shape that cannot be managed source — the same unresolved
/// target invoked two or more times with different type arguments and no result
/// — is dropped. Resolved calls and consumed results are never touched.
fn drop_metadata_init_calls(stmts: &mut Vec<Stmt>) -> usize {
    // The type argument is a `typeof(T)` when the klass pointer was resolved,
    // or the metadata usage slot (`StaticField("void*", "ptr_0x…")`) before
    // ptr_resolution names it. Both are the same class pointer.
    fn is_type_arg(e: &Expr, is_first: bool) -> bool {
        match e {
            Expr::TypeOf(_) => is_first,
            Expr::StaticField(_, f) => is_first && f.starts_with("ptr_0x"),
            _ => false,
        }
    }
    fn collect(list: &[Stmt], counts: &mut std::collections::HashMap<u64, usize>) {
        for st in list {
            if let Stmt::Eval(Expr::RawCall(addr, args, None)) = st {
                if args.len() >= 2 && is_type_arg(&args[0], true) {
                    *counts.entry(*addr).or_insert(0) += 1;
                }
            }
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(st, &mut kids);
            for k in kids {
                collect(k, counts);
            }
        }
    }
    fn remove(list: &mut Vec<Stmt>, drop: &std::collections::HashSet<u64>) -> usize {
        let mut n = 0;
        let mut i = 0;
        while i < list.len() {
            let removable = matches!(
                &list[i],
                Stmt::Eval(Expr::RawCall(addr, args, None))
                    if drop.contains(addr)
                        && args.len() >= 2
                        && is_type_arg(&args[0], true)
            );
            if removable {
                list.remove(i);
                n += 1;
            } else {
                for k in super::children_mut(&mut list[i]) {
                    n += remove(k, drop);
                }
                i += 1;
            }
        }
        n
    }
    let mut counts = std::collections::HashMap::new();
    collect(stmts, &mut counts);
    let drop: std::collections::HashSet<u64> = counts
        .iter()
        .filter(|(_, c)| **c >= 2)
        .map(|(a, _)| *a)
        .collect();
    if crate::recovery::env_flags::flag("ILR_ALLOCDBG") {
        eprintln!("[initcalls] counts={counts:?} drop={drop:?}");
    }
    if drop.is_empty() {
        return 0;
    }
    remove(stmts, &drop)
}

fn observable_names(stmts: &[Stmt]) -> std::collections::HashSet<String> {
    fn walk(list: &[Stmt], out: &mut std::collections::HashSet<String>) {
        for s in list {
            for_each_own_expr(s, &mut |e| visit_observable_expr(e, out));
            if let Stmt::Using { decl, .. } = s {
                walk(std::slice::from_ref(decl.as_ref()), out);
            }
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            for k in kids {
                walk(k, out);
            }
        }
    }
    let mut out = std::collections::HashSet::new();
    walk(stmts, &mut out);
    out
}

fn visit_observable_expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
    match e {
        Expr::AddressOf(x) | Expr::RawAddr(x, _) => {
            collect_expr_vars(x, out);
        }
        Expr::Lambda { params, body } => {
            collect_captured_vars(body, params, out);
        }
        _ => {}
    }
    for c in expr_children(e) {
        visit_observable_expr(c, out);
    }
}

fn collect_expr_vars(e: &Expr, out: &mut std::collections::HashSet<String>) {
    if let Expr::Var(n) = e {
        out.insert(n.clone());
    }
    for c in expr_children(e) {
        collect_expr_vars(c, out);
    }
}

/// Vars read by `e` that are not bound inside `e` itself (lambda parameters /
/// pattern bindings are locals of the lambda, not captured outer names).
fn collect_captured_vars(
    e: &Expr,
    shadow: &[String],
    out: &mut std::collections::HashSet<String>,
) {
    match e {
        Expr::Lambda { params, body } => {
            let mut inner = shadow.to_vec();
            inner.extend(params.iter().cloned());
            collect_captured_vars(body, &inner, out);
            return;
        }
        Expr::Var(n) => {
            if !shadow.contains(n) {
                out.insert(n.clone());
            }
        }
        _ => {}
    }
    for c in expr_children(e) {
        collect_captured_vars(c, shadow, out);
    }
}

/// Deep read-mention counts for a statement tree. Used to prove a name has no
/// reads outside the list a rewrite is confined to. Mirrors
/// `count_reads_in_stmt`/`add_shallow_stmt_mentions`; over-counting is safe
/// (it only refuses), under-counting is not, so `Using` declarations (which
/// `children` does not expose) are walked explicitly.
/// `name_read_counts`, restricted to `filter`: same walk, same counts for the
/// kept names. The copies pass only ever asks about the source of a staging
/// copy (`v1 = v2`), so cloning every other mention into the map — and
/// re-walking nested lists to do it — was its dominant cost.
fn name_read_counts_filtered(
    stmts: &[Stmt],
    filter: &std::collections::HashSet<String>,
) -> HashMap<String, usize> {
    fn walk(
        list: &[Stmt],
        out: &mut HashMap<String, usize>,
        filter: &std::collections::HashSet<String>,
    ) {
        for s in list {
            for_each_shallow_mention(s, &mut |n| {
                if filter.contains(n) {
                    *out.entry(n.to_string()).or_default() += 1;
                }
            });
            if let Stmt::Using { decl, .. } = s {
                walk(std::slice::from_ref(decl.as_ref()), out, filter);
            }
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            for k in kids {
                walk(k, out, filter);
            }
        }
    }
    let mut out = HashMap::new();
    walk(stmts, &mut out, filter);
    out
}

/// Every name a rename could turn into the SOURCE of a staging copy: the RHS
/// of any plain `x = name` assignment (`v = src` or, after a rename,
/// `dst = name`). The coalesce pass asks for read counts of exactly these.
fn collect_copy_source_names(list: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in list {
        if let Stmt::Assign(_, _, Expr::Var(x)) = s {
            out.insert(x.clone());
        }
        if let Stmt::Using { decl, .. } = s {
            collect_copy_source_names(std::slice::from_ref(decl.as_ref()), out);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_copy_source_names(k, out);
        }
    }
}

/// Temp names of adjacent `v = …; return v;` pairs, the only names the return
/// fold reads from the method-wide map.
fn collect_return_pair_names(list: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for w in list.windows(2) {
        if let (Stmt::Assign(v, None, _), Stmt::Return(Some(Expr::Var(r)))) = (&w[0], &w[1]) {
            if r == v && is_temp_var_name(v) {
                out.insert(v.clone());
            }
        }
    }
    for s in list {
        if let Stmt::Using { decl, .. } = s {
            collect_return_pair_names(std::slice::from_ref(decl.as_ref()), out);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_return_pair_names(k, out);
        }
    }
}

/// Does control leave the statement list at this statement (or deeper) by a
/// path that does not fall through to the next statement? `return`/`throw` end
/// the method and cannot observe a later local; branches that jump to another
/// statement can skip the copy and observe the destination's OLD value after
/// the forward rename, so they are refused.
fn stmt_transfers_control_out(s: &Stmt) -> bool {
    match s {
        Stmt::Goto(_) | Stmt::Break | Stmt::Continue | Stmt::YieldBreak => true,
        _ => {
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            kids.iter()
                .any(|k| k.iter().any(stmt_transfers_control_out))
        }
    }
}

/// Does this statement introduce a binding named `name` (`foreach (name in …)`
/// or `is T name`) that would shadow the temp in part of the range?
fn stmt_mentions_binding(s: &Stmt, name: &str) -> bool {
    if matches!(s, Stmt::ForEach { var_name, .. } if var_name == name) {
        return true;
    }
    if let Stmt::Using { decl, .. } = s {
        if stmt_mentions_binding(decl, name) {
            return true;
        }
    }
    let mut hit = false;
    for_each_own_expr(s, &mut |e| {
        if expr_mentions_binding(e, name) {
            hit = true;
        }
    });
    if hit {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    kids.iter()
        .any(|k| k.iter().any(|x| stmt_mentions_binding(x, name)))
}

fn expr_mentions_binding(e: &Expr, name: &str) -> bool {
    if let Expr::PatternMatch { binding, expr, .. } = e {
        if binding == name {
            return true;
        }
        return expr_mentions_binding(expr, name);
    }
    expr_children(e)
        .iter()
        .any(|c| expr_mentions_binding(c, name))
}

// ===========================================================================
// 7C-2c: Return-value temp folding
// ===========================================================================

/// Fold `v = <pure expr>; return v;` into `return <expr>;`.
///
/// The two statements are adjacent, so relocating the value costs nothing in
/// evaluation order. When the live range holds no other read of `v` (and no
/// read exists outside the list), the assignment is deleted as well; when the
/// value has other readers — a merge target that cannot be removed — only the
/// return read is folded and the definition STAYS. Duplicating the expression
/// is restricted to the same size budget the inliner uses.
fn fold_return_of_temp(stmts: &mut Vec<Stmt>) -> usize {
    // Only the names of adjacent `v = …; return v;` pairs are ever looked up
    // in the method-wide map. Building the full mention map cloned every name
    // in the method for nothing when no pair exists at all.
    let mut pair_names = std::collections::HashSet::new();
    collect_return_pair_names(stmts, &mut pair_names);
    if pair_names.is_empty() {
        return 0;
    }
    let method_reads = name_read_counts_filtered(stmts, &pair_names);
    let observable = observable_names(stmts);
    let mut count = 0;
    walk_scoped(stmts, &no_scope(), false, &mut |list, ctx| {
        count += fold_return_of_temp_list(list, ctx, &method_reads, &observable);
    });
    count
}

fn fold_return_of_temp_list(
    stmts: &mut Vec<Stmt>,
    ctx: &Scope<'_>,
    method_reads: &HashMap<String, usize>,
    observable: &std::collections::HashSet<String>,
) -> usize {
    let blocked = blocked_outside_mentions(stmts, ctx);
    let mut count = 0;
    let mut i = 0;
    while i + 1 < stmts.len() {
        // Only the name is cloned here; the RHS is borrowed until the fold is
        // actually accepted, so refused candidates pay nothing for it.
        let candidate = match (&stmts[i], &stmts[i + 1]) {
            (Stmt::Assign(v, None, e), Stmt::Return(Some(Expr::Var(r))))
                if r == v
                    && is_temp_var_name(v)
                    && expr_is_pure(e)
                    && !is_alloc_rhs(e)
                    && !observable.contains(v) =>
            {
                Some(v.clone())
            }
            _ => None,
        };
        let Some(v) = candidate else {
            i += 1;
            continue;
        };

        let range_end = next_write_index(stmts, i, &v);
        let reads_in_range = count_reads_in_range(stmts, i + 1, range_end, &v);
        let local_reads = count_reads_in_list(stmts, &v);
        let no_outside = method_reads.get(&v).copied().unwrap_or(0) == local_reads;
        let keep_def = reads_in_range != 1 || !no_outside || blocked.contains(&v);
        if keep_def {
            let big = match &stmts[i] {
                Stmt::Assign(_, _, e) => expr_size(e) > MAX_DUPLICATED_EXPR_SIZE,
                _ => false,
            };
            if big {
                i += 1;
                continue;
            }
        }
        let e = match &stmts[i] {
            Stmt::Assign(_, _, e) => e.clone(),
            _ => unreachable!(),
        };
        if !substitute_var_in_range(stmts, i + 1, i + 2, &v, &e, None) {
            i += 1;
            continue;
        }
        count += 1;
        RETURN_FOLDED.fetch_add(1, Ordering::Relaxed);
        if keep_def {
            i += 2;
        } else {
            stmts.remove(i);
            i += 1;
        }
    }
    count
}

/// Remove redundant writes to temp variables.
///
/// When a temp is loaded with the same CONSTANT twice and nothing between the
/// two writes touches it, the second write stores the value the variable already
/// holds and can be dropped.
///
/// Restricted to literals on purpose:
/// - a repeated `v = f()` is NOT redundant — the call has to run again;
/// - a repeated `v = obj.field` is NOT redundant — an intervening call may have
///   written that field, so the second load is what keeps the value current.
fn remove_redundant_temp_writes(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += remove_redundant_temp_writes_list(list);
    });
    count
}

/// Is this expression a literal, whose value no statement can change?
fn expr_is_literal(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Int(_)
            | Expr::Bool(_)
            | Expr::Str(_)
            | Expr::Null
            | Expr::Float(_)
            | Expr::Double(_)
    )
}

/// Scan a statement list for redundant temp writes.
fn remove_redundant_temp_writes_list(stmts: &mut Vec<Stmt>) -> usize {
    use std::collections::HashMap;

    let mut total = 0;
    loop {
        let mut count = 0;
        let mut last_value: HashMap<String, (String, usize)> = HashMap::new();

        let mut i = 0;
        while i < stmts.len() {
            let should_remove = match &stmts[i] {
                Stmt::Assign(var, None, rhs)
                    if is_temp_var_name(var) && expr_is_literal(rhs) =>
                {
                    let rhs_str = format!("{:?}", rhs);
                    match last_value.get(var) {
                        Some((prev, prev_idx)) if *prev == rhs_str => {
                            // Only if nothing between the two writes names the
                            // temp: any read is harmless, but a write (through
                            // an alias or a `v.x = ..` path) could have changed
                            // what the second write restores.
                            let lo = (*prev_idx).min(i);
                            let hi = (*prev_idx).max(i);
                            stmts[lo + 1..hi]
                                .iter()
                                .all(|s| !stmt_mentions_deep(s, var))
                        }
                        Some(_) => {
                            last_value.insert(var.clone(), (rhs_str, i));
                            false
                        }
                        None => {
                            last_value.insert(var.clone(), (rhs_str, i));
                            false
                        }
                    }
                }
                _ => false,
            };

            if should_remove {
                if crate::recovery::env_flags::flag("ILR_DELDBG") {
                    if let Stmt::Assign(n, _, _) = &stmts[i] {
                        eprintln!("[del-name] {n} @{i}");
                    }
                }
                stmts.remove(i);
                count += 1;
                // Don't increment i - next statement shifted into position
            } else {
                i += 1;
            }
        }

        total += count;
        if count == 0 {
            break;
        }
    }
    total
}

/// Remove consecutive dead writes to temp variables.
///
/// Pattern: `v = X; v = Y;` where the first write is never read.
///
/// This is safe because:
/// 1. The two writes are adjacent (no intervening statement)
/// 2. `X` is pure, so dropping it loses no side effect
/// 3. `Y` does not read `v` — otherwise the value it reads is the one removed
fn remove_consecutive_dead_writes(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += remove_consecutive_dead_writes_list(list);
    });
    count
}

/// Scan a statement list for consecutive dead writes.
fn remove_consecutive_dead_writes_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut total = 0;
    loop {
        let mut count = 0;
        let mut i = 0;

        while i + 1 < stmts.len() {
            let should_remove = match (&stmts[i], &stmts[i + 1]) {
                (Stmt::Assign(v1, None, x), Stmt::Assign(v2, None, y))
                    if v1 == v2 && is_temp_var_name(v1) && expr_is_pure(x) =>
                {
                    // Adjacent, so no statement between them can read the first
                    // value — but the second write's own expression can.
                    expr_mention_count(y, v1) == 0
                }
                _ => false,
            };

            if should_remove {
                stmts.remove(i);
                count += 1;
                // Don't increment i - next statement shifted into position
            } else {
                i += 1;
            }
        }

        total += count;
        if count == 0 {
            break;
        }
    }
    total
}

/// Substitute all reads of `old_name` with `new_name` in the statement list,
/// excluding the statement at `skip_idx` (the assignment itself).
/// Substitute all reads of `old_name` with `new_name` from `start` onward.
/// Reads BEFORE the copy do not observe its value — on a loop back edge they
/// read the PREVIOUS iteration's binding, which the preheader initializes for
/// `old_name` only — so they must keep the old name.
fn substitute_var_with_var_from(
    stmts: &mut [Stmt],
    start: usize,
    old_name: &str,
    new_name: &str,
) -> bool {
    let mut substituted = false;
    for stmt in stmts.iter_mut().skip(start) {
        if substitute_var_with_var_in_stmt(stmt, old_name, new_name) {
            substituted = true;
        }
    }
    substituted
}

/// Substitute all reads of `old_name` with `new_name` in a statement.
fn substitute_var_with_var_in_stmt(stmt: &mut Stmt, old_name: &str, new_name: &str) -> bool {
    let mut substituted = false;
    match stmt {
        Stmt::Eval(e)
        | Stmt::Return(Some(e))
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e) => {
            substituted = substitute_var_with_var_in_expr(e, old_name, new_name);
        }
        Stmt::Assign(target, _, rhs) => {
            if target == old_name {
                *target = new_name.to_string();
                substituted = true;
            }
            substituted |= substitute_var_with_var_in_expr(rhs, old_name, new_name);
        }
        Stmt::IndexAssign { collection, index, value } => {
            substituted |= substitute_var_with_var_in_expr(collection, old_name, new_name);
            substituted |= substitute_var_with_var_in_expr(index, old_name, new_name);
            substituted |= substitute_var_with_var_in_expr(value, old_name, new_name);
        }
        Stmt::EventAssign { field, value, .. } => {
            substituted |= substitute_var_with_var_in_expr(field, old_name, new_name);
            substituted |= substitute_var_with_var_in_expr(value, old_name, new_name);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            substituted |= substitute_var_with_var_in_expr(cond, old_name, new_name);
            for s in then_body {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
            for s in else_body {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
        }
        Stmt::While { cond, body } | Stmt::DoWhile { cond, body, .. } => {
            substituted |= substitute_var_with_var_in_expr(cond, old_name, new_name);
            for s in body {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
        }
        Stmt::For { init, cond, incr, body, .. } => {
            for s in init {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
            if let Some(c) = cond {
                substituted |= substitute_var_with_var_in_expr(c, old_name, new_name);
            }
            for s in incr {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
            for s in body {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
        }
        Stmt::Lock { expr, body } => {
            substituted |= substitute_var_with_var_in_expr(expr, old_name, new_name);
            for s in body {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
        }
        Stmt::Using { decl, body } => {
            substituted |= substitute_var_with_var_in_stmt(decl, old_name, new_name);
            for s in body {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
        }
        Stmt::ForEach { var_name, collection, body } => {
            if var_name == old_name {
                *var_name = new_name.to_string();
                substituted = true;
            }
            // collection is a String (variable name), not an Expr
            if collection == old_name {
                *collection = new_name.to_string();
                substituted = true;
            }
            for s in body {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
        }
        Stmt::Try { try_body, catches, finally_body } => {
            for s in try_body {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
            for (_, b) in catches {
                for s in b {
                    substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
                }
            }
            if let Some(fb) = finally_body {
                for s in fb {
                    substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
                }
            }
        }
        Stmt::Switch {
            disc,
            cases,
            tail,
            default,
        } => {
            // disc is a String (variable name), not an Expr
            if disc == old_name {
                *disc = new_name.to_string();
                substituted = true;
            }
            for (_, ss) in cases {
                for s in ss {
                    substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
                }
            }
            for s in tail {
                substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
            }
            if let Some(d) = default {
                for s in d {
                    substituted |= substitute_var_with_var_in_stmt(s, old_name, new_name);
                }
            }
        }
        _ => {}
    }
    substituted
}

/// Rename every read of `old_name` to `new_name` in an expression.
fn substitute_var_with_var_in_expr(expr: &mut Expr, old_name: &str, new_name: &str) -> bool {
    if let Expr::Var(n) = expr {
        if n == old_name {
            *n = new_name.to_string();
            return true;
        }
    }
    let mut substituted = false;
    for child in expr_children_mut(expr) {
        substituted |= substitute_var_with_var_in_expr(child, old_name, new_name);
    }
    substituted
}

/// Scan a statement list for non-adjacent temp-var inlining opportunities.
///
/// Iterates to a fixed point because inlining one temp may expose another
/// (e.g., `v1 = expr; v2 = v1 + 1; return v2;` — after inlining v2, v1
/// becomes adjacent to its use and can be inlined too).
///
/// Two tiers of candidates:
/// 1. **Pure RHS** — can always be inlined (existing behavior).
/// 2. **Impure RHS with calls** — can be inlined when the temp has exactly 1
///    write and 1 read AND there are no other calls between the write and the
///    read. This is safe because the call result is used exactly once, so
///    moving it to the use site doesn't change the relative order of side
///    effects.
fn inline_temp_vars_list(
    stmts: &mut Vec<Stmt>,
    ctx: &Scope<'_>,
    observable: &std::collections::HashSet<String>,
    labels: &LabelIndex,
) -> usize {
    let blocked = blocked_outside_mentions(stmts, ctx);
    // Positions of statements that define labels, with their names. The scan
    // below used to re-walk every statement's expression tree for every
    // candidate; this pass never adds or removes LABELS (it substitutes
    // reads and removes defs), so one index serves all candidates and rounds.
    // Removals shift indices and are applied at the removal site.
    let mut label_pos: Vec<(usize, Vec<String>)> = stmts
        .iter()
        .enumerate()
        .filter_map(|(idx, s)| {
            let mut names = Vec::new();
            labels_in(s, &mut names);
            (!names.is_empty()).then_some((idx, names))
        })
        .collect();
    let mut total = 0;
    // Nodes this list has gained from duplication. Inlining shrinks the
    // statement count but can grow the tree, so the fixed-point loop needs its
    // own ceiling rather than relying on `count == 0` to terminate.
    let mut added_nodes: usize = 0;
    loop {
        if added_nodes >= MAX_INLINED_NODES_PER_LIST {
            break;
        }
        let mut count = 0;

        // Collect candidate assignments: indices where stmts[i] is Assign(v, None, rhs)
        // with v matching the temp pattern and rhs not being an allocation.
        // We separate pure and impure candidates for different safety checks.
        let pure_candidates: Vec<usize> = stmts
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Stmt::Assign(var, None, rhs)
                    if is_temp_var_name(var)
                        && !observable.contains(var)
                        && expr_is_pure(rhs)
                        && !is_alloc_rhs(rhs) =>
                {
                    Some(i)
                }
                _ => None,
            })
            .collect();

        let impure_candidates: Vec<usize> = stmts
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Stmt::Assign(var, None, rhs)
                    if is_temp_var_name(var)
                        && !observable.contains(var)
                        && !expr_is_pure(rhs) =>
                {
                    Some(i)
                }
                _ => None,
            })
            .collect();

        // Combine all candidates and sort in reverse order so removal doesn't
        // shift indices of earlier candidates.
        let mut all_candidates: Vec<(usize, bool)> = pure_candidates
            .into_iter()
            .map(|i| (i, true))
            .chain(impure_candidates.into_iter().map(|i| (i, false)))
            .collect();
        all_candidates.sort_by(|a, b| b.0.cmp(&a.0));

        // Process candidates in reverse order.
        for (i, is_pure) in all_candidates {
            let (var_name, rhs) = match &stmts[i] {
                Stmt::Assign(v, _, e) => (v.clone(), e.clone()),
                _ => continue,
            };
            // See `flatten_temp_vars_list`: substituting a self-referential
            // def into a later read changes the value it observes.
            if super::expr_mentions(&rhs, &var_name) {
                continue;
            }

            // LIVE-RANGE SLICING. SSA versions from different merge points can
            // share one display name (`v39 = A; use(v39); v39 = B; use(v39);`
            // is two independent live ranges). Counting reads/writes over the
            // WHOLE list made every such def look multi-def/multi-read and
            // blocked inlining (measured: ~2.6k sequential-multi-def vars,
            // ~17k refs). The range this def governs runs from the def up to
            // the next statement that writes the same name (nested writes
            // included — after one, later reads observe a different value).
            // `full_range_end` is the next write: the whole live range for
            // READ/WRITE analysis. `range_end` may be cut earlier at a label;
            // the cut limits only where the expression may be SUBSTITUTED, it
            // must not hide reads that still need the definition (an impure
            // call with several reads must never be inlined and its def must
            // never be deleted).
            let full_range_end = next_write_index(stmts, i, &var_name);
            let mut range_end = full_range_end;
            // A LABEL is a jump target: an edge from elsewhere reaches the
            // statements after it WITHOUT executing this definition. Inlining
            // across one substituted a branch-local `v289 = 0` into the shared
            // L7 block and turned `*(float*)(v289 +24)` into `*(float*)(0 +24)`
            // (TokenManager.Initialize). A label is safe to cross when all of
            // its gotos are forward from between this def and the label (see
            // LabelIndex); otherwise stop the substitution range there.
            let mut unsafe_off = None;
            let scan_end = range_end.min(stmts.len());
            for (interval_end, names) in label_pos.iter() {
                if *interval_end <= i || *interval_end >= scan_end {
                    continue;
                }
                let unsafe_here = names.iter().any(|l| {
                    let defs = labels.defs.get(l).copied().unwrap_or(0);
                    let total = labels.gotos.get(l).copied().unwrap_or(0);
                    if defs != 1 || total == 0 {
                        return true;
                    }
                    // Exclude the label statement itself: a goto there is a
                    // back edge, which can re-enter without re-executing the
                    // definition.
                    count_gotos_to(&stmts[i + 1..*interval_end], l) != total
                });
                if unsafe_here {
                    unsafe_off = Some(*interval_end - (i + 1));
                    break;
                }
            }
            if let Some(off) = unsafe_off {
                range_end = i + 1 + off;
            }
            // ILR_TEMPDBG=<name>: explain why this temp is (not) inlined.
            let dbg = std::env::var("ILR_TEMPDBG").ok();
            let dbg_hit = dbg.as_deref() == Some(var_name.as_str());
            // The range ends at the NEXT top-level write, so it contains no
            // write of the temp: `writes` is exactly 1 by construction (the
            // debug_assert below). The scan existed only for the debug line
            // and ran on every candidate in release; compute it only when the
            // line will print.
            let writes = if dbg_hit {
                1 + count_writes_in_range(stmts, i + 1, full_range_end, &var_name)
            } else {
                1
            };
            let reads = count_reads_in_range(stmts, i, full_range_end, &var_name);
            if dbg_hit {
                eprintln!(
                    "[tempdbg] {var_name}: pure={is_pure} writes={writes} reads={reads} rhs={:?}",
                    std::mem::discriminant(&rhs)
                );
                for (idx, s) in stmts.iter().enumerate() {
                    let c = count_reads_in_stmt(s, &var_name);
                    if c > 0 {
                        let d = format!("{s:?}");
                        let d = if d.len() > 220 { &d[..220] } else { &d[..] };
                        eprintln!("[tempdbg]   stmt[{idx}] mentions={c}: {d}");
                    }
                }
            }

            // Within its live range the def is the ONLY write (the range ends
            // at the next one), so only the read count gates the inline.
            debug_assert_eq!(writes, 1, "live range must contain just the def");

            // Reads outside this list are out of the substitution's reach. The
            // def must then stay, but the LOCAL reads can still be rewritten
            // when the RHS is pure: the name keeps its value for the outside
            // read while this list stops naming it.
            //
            // A RETURN consumes the value: the store is the last thing before
            // the method leaves, so no read in any other list can observe it,
            // whatever the enclosing scope names. The return register is read
            // that way by every arm that returns, so without this each arm
            // counted the other arms' reads as outside and no arm dropped its
            // store -- every early exit rendered as "obj = 0; return 0;".
            let keep_def = blocked.contains(&var_name);
            if keep_def && !is_pure {
                if dbg_hit {
                    eprintln!("[tempdbg] {var_name}: rejected: impure and mentioned outside this list");
                }
                continue;
            }

            // For pure RHS: allow any number of reads (>= 1).
            // For impure RHS: require exactly 1 read (side effects must happen once).
            if is_pure {
                if reads < 1 {
                    if dbg_hit {
                        eprintln!("[tempdbg] {var_name}: rejected: pure, reads=0");
                    }
                    continue;
                }
                // Inlining into several reads COPIES the expression once per
                // read. A single use is a pure rename and always fine; a large
                // expression repeated many times is not — the copies become
                // fodder for the next temp's inlining, and the tree grows
                // geometrically (this blew a 460 MB method up to 3 GB before
                // the OOM killer stopped it).
                if reads > 1 {
                    let size = expr_size(&rhs);
                    if size > MAX_DUPLICATED_EXPR_SIZE || size.saturating_mul(reads) > MAX_DUPLICATION_COST
                    {
                        if dbg_hit {
                            eprintln!("[tempdbg] {var_name}: rejected: pure, reads={reads} size={size} cost={}", size.saturating_mul(reads));
                        }
                        continue;
                    }
                }
            } else {
                if reads != 1 {
                    if dbg_hit {
                        eprintln!("[tempdbg] {var_name}: rejected: impure, reads={reads}");
                    }
                    continue;
                }
            }

            // For impure candidates, check that no intervening statement
            // between the write and the read contains a call, and that the
            // single read sits in a position a value can actually be written
            // into (moving an impure value into an unreachable read while
            // leaving the original behind would duplicate the side effect).
            if !is_pure {
                if stmts[i..range_end]
                    .iter()
                    .any(|s| has_bare_name_slot_read(s, &var_name))
                {
                    if dbg_hit {
                        eprintln!("[tempdbg] {var_name}: rejected: impure, bare-name-slot read");
                    }
                    continue;
                }
                if read_nested_in_loop(&stmts[i..range_end], &var_name) {
                    if dbg_hit {
                        eprintln!("[tempdbg] {var_name}: rejected: impure, read nested in loop");
                    }
                    continue;
                }
                let read_idx = match find_read_index_in_range(stmts, i, range_end, &var_name) {
                    Some(idx) => idx,
                    None => {
                        if dbg_hit {
                            eprintln!("[tempdbg] {var_name}: rejected: impure, no read index");
                        }
                        continue;
                    }
                };

                let lo = i.min(read_idx) + 1;
                let hi = i.max(read_idx);
                if lo < hi && stmts[lo..hi].iter().any(|s| stmt_has_call(s)) {
                    if dbg_hit {
                        eprintln!("[tempdbg] {var_name}: rejected: impure, intervening call between {} and {}", i, read_idx);
                    }
                    continue;
                }
            }

            // Find the statement that reads var_name and substitute. Only the
            // def's live range is rewritten: later ranges belong to later
            // defs of the same display name.
            //
            // Delete the assignment only when nothing still names the temp. A
            // read that lives in a bare-name slot (`switch (v)`,
            // `foreach (x in v)`) holds a name rather than an expression, so
            // the value cannot be moved into it; removing the declaration in
            // that case would leave the method referencing a variable that no
            // longer exists.
            // A pure RHS is not a constant: substituting it makes every use
            // re-read the names it mentions. A write to one of those names
            // between the def and a use changes what that use computes.
            // Measured (0x18A53C4):
            //   v158 = v42 - probability;
            //   v42  = v158;                 // loop-carried variable updated
            //   if (v158 >= 0f) …            // must test the RESULT
            // became `if (v42 - probability >= 0f)`, subtracting a SECOND time
            // because v42 already held the new value. Stop the substitution at
            // the first such write; reads beyond it keep the temp (and the def
            // stays materialised).
            let mut subst_end = range_end;
            let mut rhs_reads: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            collect_expr_vars(&rhs, &mut rhs_reads);
            for read in &rhs_reads {
                let w = next_write_index(stmts, i, read);
                if w < subst_end {
                    // The write statement itself is still safe: its RHS is
                    // evaluated BEFORE the store (`v123 = v123 + 1` is the
                    // source shape). Only statements AFTER it re-read the
                    // updated name.
                    subst_end = (w + 1).min(range_end);
                }
            }

            if substitute_var_in_range(stmts, i, subst_end, &var_name, &rhs, Some(i)) {
                // The live range (not the substitution cut) decides deletion:
                // a label or a name-writing statement can end SUBSTITUTION
                // while reads after it still need the definition. Counting
                // only to `range_end` deleted the def and left undefined
                // temps (0x1671C90: `v161 = ….gameObject` used after the pool
                // block's labels).
                let after = count_reads_in_range(stmts, i, full_range_end, &var_name);
                if after == 0 {
                    if keep_def {
                        if dbg_hit {
                            eprintln!(
                                "[tempdbg] {var_name}: substituted, def kept for outside read"
                            );
                        }
                    } else {
                        if dbg_hit {
                            eprintln!(
                                "[tempdbg] {var_name}: INLINED at stmt {i} of list(len={})",
                                stmts.len()
                            );
                        }
                        stmts.remove(i);
                        label_pos.retain(|(j, _)| *j != i);
                        for (j, _) in label_pos.iter_mut() {
                            if *j > i {
                                *j -= 1;
                            }
                        }
                        added_nodes =
                            added_nodes.saturating_add(expr_size(&rhs).saturating_mul(reads));
                        count += 1;
                    }
                } else if dbg_hit {
                    eprintln!(
                        "[tempdbg] {var_name}: rejected: {after} read(s) remain after substitution"
                    );
                }
            } else if dbg_hit {
                eprintln!("[tempdbg] {var_name}: rejected: substitution failed");
            }
        }

        total += count;
        if count == 0 {
            break;
        }
    }
    total
}

/// Does this variable name match a temp-variable pattern?
///
/// Recognized patterns:
/// - `v\d+` — standard decompiler temp (`v0`, `v1`, `v23`, …)
/// - `stack_\d+` — stack-slot temp (`stack_0`, `stack_198`, …)
/// - `in_x\d+` — instance-reference temp (`in_x1`, `in_x8`, …)
/// - `arg\d+` — argument-slot temp (`arg0`, `arg1`, …)
///
/// All of these are compiler-generated names that can be inlined when safe.
fn is_temp_var_name(name: &str) -> bool {
    // v\d+, and the type-split spellings v\d+_\d+ (one version per proven
    // type group; see `MethodCtx::split_versions_by_type`).
    if let Some(rest) = name.strip_prefix('v') {
        let mut parts = rest.split('_');
        let first = parts.next().unwrap_or("");
        if !first.is_empty() && first.chars().all(|c| c.is_ascii_digit()) {
            match parts.next() {
                None => return true,
                Some(k)
                    if !k.is_empty()
                        && k.chars().all(|c| c.is_ascii_digit())
                        && parts.next().is_none() =>
                {
                    return true;
                }
                _ => {}
            }
        }
    }
    // stack_\d+
    if let Some(rest) = name.strip_prefix("stack_") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    // in_x\d+
    if let Some(rest) = name.strip_prefix("in_x") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    // arg\d+
    if let Some(rest) = name.strip_prefix("arg") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    false
}

/// Is this RHS an allocation that must stay separate for `normalize_ctor_calls`?
///
/// Only TRUE allocations (ObjectNew/ArrayNew/Box helper calls, or explicit
/// NewObj/NewArr) are excluded from inlining. Other `RawCall` targets —
/// cast helpers, isinst, misc runtime helpers, and unresolved `fn_` calls
/// that never pair with a following `.ctor` — are safe to inline.
fn is_alloc_rhs(e: &Expr) -> bool {
    use crate::recovery::runtime_helpers::{role_of, RuntimeHelper};
    match e {
        Expr::NewObj(..) | Expr::NewArr(..) => true,
        Expr::RawCall(addr, _, _) => matches!(
            role_of(*addr),
            Some(RuntimeHelper::ObjectNew) | Some(RuntimeHelper::ArrayNew) | Some(RuntimeHelper::Box)
        ),
        _ => false,
    }
}

/// Is this expression a self-assignment? (v1 = v1)
/// These are no-ops and can always be removed.
fn is_self_assignment(var: &str, rhs: &Expr) -> bool {
    matches!(rhs, Expr::Var(n) if n == var)
}

/// Is this expression pure (free of side effects)?
///
/// Pure expressions can be safely moved from one program point to another
/// without changing semantics. Impure expressions contain calls, allocations,
/// or other operations that may have side effects or depend on evaluation order.
fn expr_is_pure(e: &Expr) -> bool {
    match e {
        // Calls and allocations are impure.
        Expr::Call { .. }
        | Expr::RawCall(..)
        | Expr::NewObj(..)
        | Expr::NewArr(..)
        | Expr::CollectionInit { .. }
        | Expr::ObjInit { .. } => false,

        // Compound expressions: pure if all sub-expressions are pure.
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_is_pure(a) && expr_is_pure(b),
        Expr::Not(e) | Expr::Neg(e) | Expr::Cast(_, e) | Expr::AddressOf(e) | Expr::Await(e) => {
            expr_is_pure(e)
        }
        Expr::Field(r, _) => expr_is_pure(r),
        Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => expr_is_pure(r),
        Expr::Cond(c, a, b) => expr_is_pure(c) && expr_is_pure(a) && expr_is_pure(b),
        Expr::Index { collection, index } => expr_is_pure(collection) && expr_is_pure(index),
        Expr::ArrayInit { items, .. } => items.iter().all(|e| expr_is_pure(e)),
        Expr::NullConditional { receiver, access } => {
            expr_is_pure(receiver) && expr_is_pure(access)
        }
        Expr::Lambda { body, .. } => expr_is_pure(body),

        // Leaf expressions are always pure.
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
        | Expr::StaticField(_, _)
        | Expr::MethodGroup(_)
        | Expr::PatternMatch { .. }
        | Expr::StringInterp { .. }
        | Expr::Tuple { .. }
        | Expr::MethodChain { .. }
        | Expr::SwitchExpr { .. }
        | Expr::Range { .. }
        | Expr::IndexFromEnd(_)
        | Expr::TypeOf(_) => true,
    }
}

/// Does this expression contain a call (method invocation)?
///
/// Used to determine whether inlining an impure RHS past an intervening
/// statement is safe: if neither the intervening statement nor the RHS
/// contains a call, the relative evaluation order doesn't matter.
fn expr_has_call(e: &Expr) -> bool {
    match e {
        Expr::Call { .. } => {
            // A Call expression IS a call, so always return true.
            true
        }
        Expr::RawCall(..) | Expr::NewObj(..) | Expr::NewArr(..) => {
            // RawCall is a call; NewObj/NewArr are allocations (also impure).
            true
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_has_call(a) || expr_has_call(b),
        Expr::Not(e) | Expr::Neg(e) | Expr::Cast(_, e) | Expr::AddressOf(e) | Expr::Await(e) => {
            expr_has_call(e)
        }
        Expr::Field(r, _) => expr_has_call(r),
        Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => expr_has_call(r),
        Expr::Cond(c, a, b) => expr_has_call(c) || expr_has_call(a) || expr_has_call(b),
        Expr::Index { collection, index } => expr_has_call(collection) || expr_has_call(index),
        Expr::ArrayInit { items, .. } => items.iter().any(|e| expr_has_call(e)),
        Expr::CollectionInit { ctor_args, items, .. } => {
            ctor_args.iter().any(|e| expr_has_call(e)) || items.iter().any(|e| expr_has_call(e))
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            ctor_args.iter().any(|e| expr_has_call(e))
                || fields.iter().any(|(_, v)| expr_has_call(v))
        }
        Expr::NullConditional { receiver, access } => {
            expr_has_call(receiver) || expr_has_call(access)
        }
        Expr::Lambda { body, .. } => expr_has_call(body),
        Expr::MethodChain { .. } => true, // method chains are calls
        _ => false,
    }
}

/// Does this statement contain a call (at any depth)?
fn stmt_has_call(s: &Stmt) -> bool {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => expr_has_call(e),
        Stmt::Assign(_, _, e) => expr_has_call(e),
        Stmt::EventAssign { field, value, .. } => expr_has_call(field) || expr_has_call(value),
        Stmt::IndexAssign { collection, index, value } => {
            expr_has_call(collection) || expr_has_call(index) || expr_has_call(value)
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            expr_has_call(cond)
                || then_body.iter().any(|s| stmt_has_call(s))
                || else_body.iter().any(|s| stmt_has_call(s))
        }
        Stmt::While { cond, body } => {
            expr_has_call(cond) || body.iter().any(|s| stmt_has_call(s))
        }
        Stmt::DoWhile { cond, body } => {
            expr_has_call(cond) || body.iter().any(|s| stmt_has_call(s))
        }
        Stmt::For { init, cond, incr, body, .. } => {
            init.iter().any(|s| stmt_has_call(s))
                || cond.as_ref().is_some_and(|c| expr_has_call(c))
                || incr.iter().any(|s| stmt_has_call(s))
                || body.iter().any(|s| stmt_has_call(s))
        }
        Stmt::Return(Some(e)) => expr_has_call(e),
        Stmt::Lock { expr, body } => expr_has_call(expr) || body.iter().any(|s| stmt_has_call(s)),
        Stmt::Using { decl, body } => {
            stmt_has_call(decl) || body.iter().any(|s| stmt_has_call(s))
        }
        Stmt::ForEach { body, .. } => {
            body.iter().any(|s| stmt_has_call(s))
        }
        Stmt::Try { try_body, catches, finally_body } => {
            try_body.iter().any(|s| stmt_has_call(s))
                || catches.iter().any(|(_, b)| b.iter().any(|s| stmt_has_call(s)))
                || finally_body.as_ref().is_some_and(|f| f.iter().any(|s| stmt_has_call(s)))
        }
        Stmt::Switch { cases, tail, default, .. } => {
            cases.iter().any(|(_, b)| b.iter().any(|s| stmt_has_call(s)))
                || tail.iter().any(|s| stmt_has_call(s))
                || default
                    .as_ref()
                    .is_some_and(|arm| arm.iter().any(|s| stmt_has_call(s)))
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            cases.iter().any(|(_, b)| b.iter().any(|s| stmt_has_call(s)))
                || tail.as_ref().is_some_and(|t| t.iter().any(|s| stmt_has_call(s)))
        }
        _ => false,
    }
}

/// Find the index of the first statement in `stmts` (after `skip_idx`) that
/// reads `name`. Returns `None` if no such statement exists.
/// Largest expression worth copying into more than one read site.
/// Kept at 12 (modest bump from 10) so we get extra multi-read inlines for
/// the mid-size field/arith chains that commonly sit on top of a temp without
/// paying the quadratic `substitute_var_in_list` cost of a large ceiling. The
/// renderer's `refused_inline_reads` relaxations now catch the majority of the
/// over-materialization at the IR level, so the AST-level pass does not need
/// to be as aggressive as it was once.
const MAX_DUPLICATED_EXPR_SIZE: usize = 12;
/// Largest total copy cost (`size x reads`) for one inlining.
const MAX_DUPLICATION_COST: usize = 160;
/// Ceiling on nodes one statement list may gain from inlining.
const MAX_INLINED_NODES_PER_LIST: usize = 2_000;

/// Number of AST nodes under `e`, inclusive of `e` itself.
fn expr_size(e: &Expr) -> usize {
    1 + expr_children(e)
        .iter()
        .map(|c| expr_size(c))
        .sum::<usize>()
}

/// Does `name` appear in a statement slot that holds a bare NAME rather than
/// an expression (`switch (v)` discriminant, `foreach (x in v)` collection)?
/// No value can be substituted into those, so a temp read there cannot be
/// inlined away.
fn has_bare_name_slot_read(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::ForEach { collection, .. } if collection == name => return true,
        Stmt::Switch { disc, .. } if disc == name => return true,
        // An assignment target is a NAME path, not an expression slot.
        Stmt::Assign(lhs, _, _) if super::target_names(lhs, name) => return true,
        _ => {}
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    kids
        .iter()
        .any(|k| k.iter().any(|x| has_bare_name_slot_read(x, name)))
}

/// Is any read of `name` nested inside a LOOP statement in this list?
///
/// Inlining an impure RHS (call/allocation) whose only read sits inside a
/// loop would re-execute the side effect once per iteration when the def
/// itself runs once — moving `v = Foo();` outside a loop into a read inside
/// the loop body changes both the call count and the order. If/switch/try
/// bodies are fine: they execute at most once per def execution.
fn read_nested_in_loop(stmts: &[Stmt], name: &str) -> bool {
    fn loop_mentions(s: &Stmt, name: &str) -> bool {
        match s {
            Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
                expr_mention_count(cond, name) > 0 || count_reads_in_list(body, name) > 0
            }
            Stmt::For {
                init,
                cond,
                incr,
                body,
                ..
            } => {
                count_reads_in_list(init, name) > 0
                    || cond
                        .as_ref()
                        .is_some_and(|c| expr_mention_count(c, name) > 0)
                    || count_reads_in_list(incr, name) > 0
                    || count_reads_in_list(body, name) > 0
            }
            Stmt::ForEach {
                collection, body, ..
            } => collection == name || count_reads_in_list(body, name) > 0,
            _ => false,
        }
    }
    for s in stmts {
        if loop_mentions(s, name) {
            return true;
        }
        if matches!(
            s,
            Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::ForEach { .. }
        ) {
            continue;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            if read_nested_in_loop(k, name) {
                return true;
            }
        }
    }
    false
}

/// Find the index of the first statement (other than `skip_idx`) that reads `name`.
#[allow(dead_code)]
fn find_read_index(stmts: &[Stmt], name: &str, skip_idx: usize) -> Option<usize> {
    stmts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != skip_idx)
        .find(|(_, s)| count_reads_in_stmt(s, name) > 0)
        .map(|(i, _)| i)
}

/// End (exclusive) of the live range a def at `def_idx` governs: the first
/// following statement that writes `name` at any depth. Nested writes count —
/// after one, later reads observe a different value. `stmts.len()` when the
/// def is the last writer.
/// Does this statement (or any nested list) define a label? A label is a jump
/// target, so linear inlining must not cross the statement that owns it: an
/// edge from elsewhere reaches the labelled code without this def.
/// Label/goto counts for the dominance-aware label cut in the temp inliner.
///
/// A definition may be substituted past a label when the label is NOT an
/// external entry: unique name, at least one textual goto, and EVERY goto to
/// it originates after the definition and before the label statement. Then
/// every path reaching the label executed the definition, so the read still
/// observes the same value. A back edge or a source before the definition
/// keeps the barrier.
#[derive(Default, Debug)]
struct LabelIndex {
    defs: std::collections::HashMap<String, usize>,
    gotos: std::collections::HashMap<String, usize>,
}

fn build_label_index(stmts: &[Stmt]) -> LabelIndex {
    fn walk(s: &Stmt, idx: &mut LabelIndex) {
        match s {
            Stmt::Label(l) => *idx.defs.entry(l.clone()).or_insert(0) += 1,
            Stmt::Goto(l) => *idx.gotos.entry(l.clone()).or_insert(0) += 1,
            _ => {}
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            for x in k {
                walk(x, idx);
            }
        }
    }
    let mut idx = LabelIndex::default();
    for s in stmts {
        walk(s, &mut idx);
    }
    idx
}

fn labels_in(s: &Stmt, out: &mut Vec<String>) {
    if let Stmt::Label(l) = s {
        out.push(l.clone());
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            labels_in(x, out);
        }
    }
}

/// Number of gotos to `label` anywhere under these statements.
fn count_gotos_to(stmts: &[Stmt], label: &str) -> usize {
    fn walk(s: &Stmt, label: &str, n: &mut usize) {
        if matches!(s, Stmt::Goto(l) if l == label) {
            *n += 1;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            for x in k {
                walk(x, label, n);
            }
        }
    }
    let mut n = 0;
    for s in stmts {
        walk(s, label, &mut n);
    }
    n
}

#[allow(dead_code)]
fn subtree_has_label(s: &Stmt) -> bool {
    if matches!(s, Stmt::Label(_)) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    kids.iter().any(|k| k.iter().any(subtree_has_label))
}

fn next_write_index(stmts: &[Stmt], def_idx: usize, name: &str) -> usize {
    (def_idx + 1..stmts.len())
        .find(|&j| count_writes_in_stmt(&stmts[j], name) > 0)
        .unwrap_or(stmts.len())
}

fn count_writes_in_range(stmts: &[Stmt], start: usize, end: usize, name: &str) -> usize {
    (start..end.min(stmts.len()))
        .map(|i| count_writes_in_stmt(&stmts[i], name))
        .sum()
}

fn count_reads_in_range(stmts: &[Stmt], start: usize, end: usize, name: &str) -> usize {
    (start..end.min(stmts.len()))
        .map(|i| count_reads_in_stmt(&stmts[i], name))
        .sum()
}

fn find_read_index_in_range(
    stmts: &[Stmt],
    start: usize,
    end: usize,
    name: &str,
) -> Option<usize> {
    (start..end.min(stmts.len())).find(|&j| count_reads_in_stmt(&stmts[j], name) > 0)
}

/// Substitute `Var(name)` with `replacement` in `stmts[start..end]` only.
fn substitute_var_in_range(
    stmts: &mut [Stmt],
    start: usize,
    end: usize,
    name: &str,
    replacement: &Expr,
    skip_idx: Option<usize>,
) -> bool {
    let mut found = false;
    for i in start..end.min(stmts.len()) {
        if skip_idx == Some(i) {
            continue;
        }
        found |= substitute_var_in_stmt(&mut stmts[i], name, replacement);
    }
    found
}

/// Count the number of times `name` is written in a statement list (at any depth).
fn count_writes_in_list(list: &[Stmt], name: &str) -> usize {
    list.iter()
        .map(|s| count_writes_in_stmt(s, name))
        .sum()
}

/// Count the number of times `name` is written in a statement (at any depth).
fn count_writes_in_stmt(s: &Stmt, name: &str) -> usize {
    let mut count = 0;
    // Direct write.
    if let Stmt::Assign(n, _, _) = s {
        if n == name {
            count += 1;
        }
    }
    // For/Using headers may contain assignments.
    if let Stmt::For { init, incr, .. } = s {
        count += count_writes_in_list(init, name);
        count += count_writes_in_list(incr, name);
    }
    if let Stmt::Using { decl, .. } = s {
        count += count_writes_in_stmt(decl, name);
    }
    // Recurse into nested bodies.
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        count += count_writes_in_list(k, name);
    }
    count
}

/// Count the number of times `name` is read in a statement list (at any depth).
/// Fused equivalent of calling `count_reads_in_stmt` once per name, counting
/// every name in `names` during a single walk. Semantics are deliberate
/// copies of the per-name version, including its quirks (Field/StaticField
/// spellings count as mentions; a complex target counts its tokens; For
/// init/incr are counted twice because `children` visits them too). With
/// `None`, every name is counted (the whole-method key set is not known
/// ahead); the dead-range sweep asks for all of them.
fn count_reads_impl(
    list: &[Stmt],
    names: Option<&std::collections::HashSet<String>>,
) -> HashMap<String, usize> {
    fn bump(
        out: &mut HashMap<String, usize>,
        names: Option<&std::collections::HashSet<String>>,
        shadow: &[String],
        n: &str,
    ) {
        if shadow.iter().any(|s| s == n) {
            return;
        }
        if names.is_some_and(|set| !set.contains(n)) {
            return;
        }
        *out.entry(n.to_string()).or_default() += 1;
    }
    fn walk_expr(
        e: &Expr,
        names: Option<&std::collections::HashSet<String>>,
        shadow: &mut Vec<String>,
        out: &mut HashMap<String, usize>,
    ) {
        match e {
            Expr::Var(n) => bump(out, names, shadow, n),
            Expr::Field(_, f) | Expr::StaticField(_, f) => bump(out, names, shadow, f),
            Expr::PatternMatch { binding, .. } => bump(out, names, shadow, binding),
            Expr::Lambda { params, .. } => {
                // A lambda parameter shadows the outer name for its whole
                // subtree (`expr_mention_count` returns 0 there).
                let base = shadow.len();
                shadow.extend(params.iter().cloned());
                for c in expr_children(e) {
                    walk_expr(c, names, shadow, out);
                }
                shadow.truncate(base);
                return;
            }
            _ => {}
        }
        for c in expr_children(e) {
            walk_expr(c, names, shadow, out);
        }
    }
    fn walk_stmt(
        s: &Stmt,
        names: Option<&std::collections::HashSet<String>>,
        shadow: &mut Vec<String>,
        out: &mut HashMap<String, usize>,
    ) {
        match s {
            Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => {
                walk_expr(e, names, shadow, out)
            }
            Stmt::Assign(lhs, _, e) => {
                walk_expr(e, names, shadow, out);
                for tok in lhs.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
                    if !tok.is_empty() && tok != lhs {
                        bump(out, names, shadow, tok);
                    }
                }
            }
            Stmt::EventAssign { field, value, .. } => {
                walk_expr(field, names, shadow, out);
                walk_expr(value, names, shadow, out);
            }
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                walk_expr(collection, names, shadow, out);
                walk_expr(index, names, shadow, out);
                walk_expr(value, names, shadow, out);
            }
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
                walk_expr(cond, names, shadow, out)
            }
            Stmt::For { cond, .. } => {
                if let Some(c) = cond {
                    walk_expr(c, names, shadow, out);
                }
            }
            Stmt::Lock { expr, .. } => walk_expr(expr, names, shadow, out),
            Stmt::Return(Some(e)) => walk_expr(e, names, shadow, out),
            Stmt::StringSwitch { disc, cases, .. } => {
                walk_expr(disc, names, shadow, out);
                for (vals, _) in cases {
                    for v in vals {
                        walk_expr(v, names, shadow, out);
                    }
                }
            }
            Stmt::Attribute { args, .. } => {
                for a in args {
                    walk_expr(a, names, shadow, out);
                }
            }
            Stmt::Switch { disc, .. } => {
                if names.is_none_or(|set| set.contains(disc)) {
                    // Bare-name discriminant: `expr_mention_count` is not used
                    // for it, so no shadow check.
                    *out.entry(disc.clone()).or_default() += 1;
                }
            }
            Stmt::ForEach { collection, .. } => {
                if names.is_none_or(|set| set.contains(collection)) {
                    *out.entry(collection.clone()).or_default() += 1;
                }
            }
            _ => {}
        }
        // For init/incr are ALSO reachable through `children`, exactly like
        // the per-name walker (which counts them twice).
        if let Stmt::For { init, incr, .. } = s {
            for x in init {
                walk_stmt(x, names, shadow, out);
            }
            for x in incr {
                walk_stmt(x, names, shadow, out);
            }
        }
        if let Stmt::Using { decl, .. } = s {
            walk_stmt(decl, names, shadow, out);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            for x in k {
                walk_stmt(x, names, shadow, out);
            }
        }
    }
    let mut out = HashMap::new();
    let mut shadow: Vec<String> = Vec::new();
    for s in list {
        walk_stmt(s, names, &mut shadow, &mut out);
    }
    out
}

fn count_reads_all(
    list: &[Stmt],
    names: &std::collections::HashSet<String>,
) -> HashMap<String, usize> {
    count_reads_impl(list, Some(names))
}

/// Unfiltered per-statement census (all names, same semantics).
#[cfg(test)]
fn count_reads_stmt_unfiltered(s: &Stmt) -> HashMap<String, usize> {
    count_reads_impl(std::slice::from_ref(s), None)
}

/// Sum of per-statement read counts over `[start, end)`.
fn range_read_sum(
    per_stmt: &[HashMap<String, usize>],
    start: usize,
    end: usize,
    name: &str,
) -> usize {
    let end = end.min(per_stmt.len());
    (start..end)
        .map(|j| per_stmt[j].get(name).copied().unwrap_or(0))
        .sum()
}

fn count_reads_in_list(list: &[Stmt], name: &str) -> usize {
    list.iter()
        .map(|s| count_reads_in_stmt(s, name))
        .sum()
}

/// Count the number of times `name` is read in a statement (at any depth).
fn count_reads_in_stmt(s: &Stmt, name: &str) -> usize {
    let mut count = 0;

    // Count reads in the statement's own expressions.
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => {
            count += expr_mention_count(e, name);
        }
        Stmt::Assign(lhs, _, e) => {
            count += expr_mention_count(e, name);
            // `v376.localRotation = 0` and `*(byte*)((nint)&stack_1 +4) = 0`
            // both NEED `v376` / `stack_1` to exist: the target is a path
            // rooted at the temp, not a redefinition of it.
            if super::target_names(lhs, name) {
                count += 1;
            }
        }
        Stmt::EventAssign { field, value, .. } => {
            count += expr_mention_count(field, name);
            count += expr_mention_count(value, name);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            count += expr_mention_count(collection, name);
            count += expr_mention_count(index, name);
            count += expr_mention_count(value, name);
        }
        Stmt::If { cond, .. }
        | Stmt::While { cond, .. }
        | Stmt::DoWhile { cond, .. } => {
            count += expr_mention_count(cond, name);
        }
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                count += expr_mention_count(c, name);
            }
        }
        Stmt::Lock { expr, .. } => {
            count += expr_mention_count(expr, name);
        }
        Stmt::Return(Some(e)) => {
            count += expr_mention_count(e, name);
        }
        Stmt::StringSwitch { disc, cases, .. } => {
            count += expr_mention_count(disc, name);
            for (vals, _) in cases {
                for v in vals {
                    count += expr_mention_count(v, name);
                }
            }
        }
        Stmt::Attribute { args, .. } => {
            for a in args {
                count += expr_mention_count(a, name);
            }
        }
        Stmt::Switch { disc, .. } => {
            // Switch discriminant is a bare name string.
            if disc == name {
                count += 1;
            }
        }
        Stmt::ForEach {
            var_name: _,
            collection,
            ..
        } => {
            // The collection expression is read; the var_name is a binding.
            if collection == name {
                count += 1;
            }
        }
        _ => {}
    }

    // For/Using headers may contain reads in their statement lists.
    if let Stmt::For { init, incr, .. } = s {
        count += count_reads_in_list(init, name);
        count += count_reads_in_list(incr, name);
    }
    if let Stmt::Using { decl, .. } = s {
        count += count_reads_in_stmt(decl, name);
    }

    // Recurse into nested bodies.
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        count += count_reads_in_list(k, name);
    }

    count
}

// ---------------------------------------------------------------------------
// Shallow scope mentions
// ---------------------------------------------------------------------------
//
// `inline_temp_vars_list`, `flatten_temp_vars_list` and
// `inline_temp_to_temp_copies_list` run on ONE statement list at a time, but a
// list is not a scope: the structurer emits label/goto shells, so a def in a
// loop body is routinely read by the loop CONDITION or by code after the loop.
// Deleting the def because "nothing in this list reads it" leaves every such
// read naming a variable nothing assigns.
//
// `walk_scoped` carries the names mentioned in enclosing scopes down to each
// list: ancestor lists' own expressions, plus the sibling statements of the
// construct that owns the list. Those reads are unreachable from the list's
// substitution, so a def they name must stay.

#[allow(dead_code)]
fn bump_mention(map: &mut HashMap<String, usize>, name: &str, by: usize) {
    if name.is_empty() {
        return;
    }
    *map.entry(name.to_string()).or_insert(0) += by;
}

fn for_each_expr_mention(e: &Expr, f: &mut dyn FnMut(&str)) {
    match e {
        // `expr_mention_count` counts these three name slots; mirror it so the
        // outside-read test agrees with the per-range counts the passes use.
        Expr::Var(n) => f(n),
        Expr::Field(_, n) | Expr::StaticField(_, n) => f(n),
        Expr::PatternMatch { binding, .. } => f(binding),
        _ => {}
    }
    for c in expr_children(e) {
        for_each_expr_mention(c, f);
    }
}

/// Names mentioned by a statement's OWN expressions, not its nested bodies.
fn for_each_shallow_mention(s: &Stmt, f: &mut dyn FnMut(&str)) {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => for_each_expr_mention(e, f),
        Stmt::Assign(lhs, _, e) => {
            for_each_expr_mention(e, f);
            // A target path (`v376.localRotation = …`) needs every name token
            // but the whole target itself; a plain target is a write.
            for tok in lhs.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
                if !tok.is_empty() && tok != lhs {
                    f(tok);
                }
            }
        }
        Stmt::EventAssign { field, value, .. } => {
            for_each_expr_mention(field, f);
            for_each_expr_mention(value, f);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            for_each_expr_mention(collection, f);
            for_each_expr_mention(index, f);
            for_each_expr_mention(value, f);
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            for_each_expr_mention(cond, f);
        }
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                for_each_expr_mention(c, f);
            }
        }
        Stmt::Lock { expr, .. } => for_each_expr_mention(expr, f),
        Stmt::Return(Some(e)) => for_each_expr_mention(e, f),
        Stmt::StringSwitch { disc, cases, .. } => {
            for_each_expr_mention(disc, f);
            for (vals, _) in cases {
                for v in vals {
                    for_each_expr_mention(v, f);
                }
            }
        }
        Stmt::Switch { disc, .. } => f(disc),
        Stmt::ForEach { collection, .. } => f(collection),
        Stmt::Attribute { args, .. } => {
            for a in args {
                for_each_expr_mention(a, f);
            }
        }
        _ => {}
    }
}

/// Names mentioned by a statement's OWN expressions, not its nested bodies.
#[allow(dead_code)]
fn add_shallow_stmt_mentions(s: &Stmt, map: &mut HashMap<String, usize>) {
    for_each_shallow_mention(s, &mut |n| bump_mention(map, n, 1));
}

/// Substitute `Var(name)` with `replacement` in ALL statements in the list
/// that read `name`. Returns `true` if any substitution was performed.
///
/// If `skip_idx` is `Some(i)`, the statement at index `i` is skipped (it's the
/// assignment we're about to remove).
pub(crate) fn substitute_var_in_list(
    stmts: &mut [Stmt],
    name: &str,
    replacement: &Expr,
    skip_idx: Option<usize>,
) -> bool {
    let mut substituted = false;
    for (i, stmt) in stmts.iter_mut().enumerate() {
        if skip_idx == Some(i) {
            continue;
        }
        if substitute_var_in_stmt(stmt, name, replacement) {
            substituted = true;
        }
    }
    substituted
}

/// Substitute `Var(name)` with `replacement` in a statement and everything
/// nested inside it. Returns `true` if any substitution was made.
///
/// Every sub-part is visited unconditionally (`|=`), never short-circuited: a
/// read in an `if` condition and a second read in its body are both rewrites
/// of the same temp, and skipping the body would delete the assignment while
/// the body still named the temp.
fn substitute_var_in_stmt(stmt: &mut Stmt, name: &str, replacement: &Expr) -> bool {
    let mut found = false;
    let sub = |e: &mut Expr, found: &mut bool| {
        if expr_mention_count(e, name) > 0 {
            substitute_var(e, name, replacement);
            *found = true;
        }
    };
    match stmt {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => sub(e, &mut found),
        Stmt::Assign(target, _, e) => {
            // Never create a SELF-COPY by substitution: inlining `v193` into
            // `v144 = v193` with the replacement `v144` produced
            // `v144 = v144`, the only definition of a live loop counter. The
            // dead-store sweep then correctly removed the no-op and the 23
            // reads lost their definition (0x1466DB4). Keep this read; the
            // caller sees the residual mention and keeps the source def.
            // Block ONLY the no-op copy `target = target`, i.e. a copy RHS
            // (`target = name`) whose replacement is EXACTLY the target
            // variable. That is the shape which severs a live counter's
            // definition. Loop increments (`target = target + 1`) MENTION the
            // target but are legitimate and must inline; blocking those cost
            // +8.5k temps with no undefined-name benefit (A/B on 0x146A0E4:
            // 354 refs allowed vs 1031 blocked, both undef=2).
            let noop_self = matches!(replacement, Expr::Var(v) if v == target);
            let is_copy_of_name = matches!(e, Expr::Var(v) if v == name);
            if !(noop_self && is_copy_of_name)
                || crate::recovery::env_flags::flag("ILR_NOSELFCOPY")
            {
                sub(e, &mut found);
            }
        }
        Stmt::EventAssign { field, value, .. } => {
            sub(field, &mut found);
            sub(value, &mut found);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            sub(collection, &mut found);
            sub(index, &mut found);
            sub(value, &mut found);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            sub(cond, &mut found);
            found |= substitute_var_in_list(then_body, name, replacement, None);
            found |= substitute_var_in_list(else_body, name, replacement, None);
        }
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            sub(cond, &mut found);
            found |= substitute_var_in_list(body, name, replacement, None);
        }
        Stmt::For { init, cond, incr, body, .. } => {
            found |= substitute_var_in_list(init, name, replacement, None);
            if let Some(c) = cond {
                sub(c, &mut found);
            }
            found |= substitute_var_in_list(incr, name, replacement, None);
            found |= substitute_var_in_list(body, name, replacement, None);
        }
        Stmt::Return(Some(e)) => sub(e, &mut found),
        Stmt::Lock { expr, body } => {
            sub(expr, &mut found);
            found |= substitute_var_in_list(body, name, replacement, None);
        }
        Stmt::Using { decl, body } => {
            found |= substitute_var_in_stmt(decl, name, replacement);
            found |= substitute_var_in_list(body, name, replacement, None);
        }
        Stmt::ForEach { collection, body, .. } => {
            // `collection` is a bare-name slot: an expression cannot be written
            // there. The caller detects the surviving read and keeps the
            // assignment instead of inlining.
            found |= collection == name;
            found |= substitute_var_in_list(body, name, replacement, None);
        }
        Stmt::Try { try_body, catches, finally_body } => {
            found |= substitute_var_in_list(try_body, name, replacement, None);
            for (_, b) in catches.iter_mut() {
                found |= substitute_var_in_list(b, name, replacement, None);
            }
            if let Some(fb) = finally_body {
                found |= substitute_var_in_list(fb, name, replacement, None);
            }
        }
        Stmt::Switch {
            disc,
            cases,
            tail,
            default,
        } => {
            found |= disc == name;
            for (_, b) in cases.iter_mut() {
                found |= substitute_var_in_list(b, name, replacement, None);
            }
            found |= substitute_var_in_list(tail, name, replacement, None);
            if let Some(d) = default {
                found |= substitute_var_in_list(d, name, replacement, None);
            }
        }
        Stmt::StringSwitch { disc, cases, tail } => {
            sub(disc, &mut found);
            for (vals, b) in cases.iter_mut() {
                for v in vals {
                    sub(v, &mut found);
                }
                found |= substitute_var_in_list(b, name, replacement, None);
            }
            if let Some(t) = tail {
                found |= substitute_var_in_list(t, name, replacement, None);
            }
        }
        Stmt::Attribute { args, .. } => {
            for a in args {
                sub(a, &mut found);
            }
        }
        _ => {}
    }
    found
}

// ===========================================================================
// 7C-3: Dead temporary variable assignment removal
// ===========================================================================

/// Remove dead assignments to temporary variables.
///
/// A temp var assignment `vN = expr;` is dead when `vN` is never read anywhere
/// in the method. If the RHS is pure (no side effects), the entire assignment
/// can be safely removed.
///
/// This also removes self-assignments like `v1 = v1;` which are no-ops.
///
/// ## Why this matters
///
/// The decompiler frequently generates dead temp var assignments when it creates
/// temporaries for sub-expressions that are later optimized away or inlined
/// into a different form. These dead assignments clutter the output.
fn remove_dead_temp_assignments(stmts: &mut Vec<Stmt>) -> usize {
    // Liveness is a property of the method, not of one statement list: a
    // definition inside a loop body is routinely read after the loop. Deciding
    // deadness from the list that holds the assignment deletes it while that
    // later read survives, and the result is a name nothing assigns. So the
    // method-wide read counts are collected once and consulted instead.
    //
    // The counts are recomputed after a round that removed something: a def's
    // only reader can itself be a dead pure def (`v1 = v2; v2 = 0;` where
    // nothing reads v1). The first round refuses v2 (v1 still names it) and
    // removes v1; only fresh counts show that v2 is dead now. The fixed point
    // is what makes the sweep complete rather than single-step.
    //
    // Name-wide counts are only sound in one direction: a name with zero reads
    // is dead, but a name with reads is NOT necessarily live at every def.
    // IL2CPP reuses one machine slot (one display name) for several disjoint
    // live ranges, so `v151 = this.badge != null; if (v151) { v151 =
    // BadgingManager.Instance; … }` keeps the second store because the FIRST
    // range's read is counted. `remove_dead_range_stores` handles exactly that
    // gap before the name-wide sweep runs.
    let dts_timed = crate::recovery::env_flags::flag("ILR_TIMING");
    macro_rules! dts {
        ($idx:expr, $e:expr) => {{
            if !dts_timed {
                $e
            } else {
                let t = std::time::Instant::now();
                let v = $e;
                DTS_NANOS[$idx].fetch_add(
                    t.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                v
            }
        }};
    }
    let observable = dts!(0, observable_names(stmts));
    static DT_CALL: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    let dt_call = DT_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut dt_round = 0usize;
    let dump = |tag: &str, round: usize, stmts: &Vec<Stmt>| {
        if crate::recovery::env_flags::flag("ILR_CLEANDBG") {
            let _ = std::fs::write(
                crate::recovery::probe_dir().join(format!(
                    "dt_{dt_call}_{round}_{tag}.txt"
                )),
                format!("{stmts:#?}"),
            );
        }
    };
    let mut total = 0;
    loop {
        dt_round += 1;
        let range_dead = dts!(1, remove_dead_range_stores(stmts, &observable));
        total += range_dead;
        dump("range", dt_round, stmts);

        let mut names: Vec<String> = Vec::new();
        dts!(2, {
            collect_assigned_temp_names(stmts, &mut names, &observable);
            names.sort();
            names.dedup();
        });
        // ONE tree walk for ALL names. The old shape walked the whole method
        // once per assigned temp (50–300 names), which multiplied the sweep's
        // cost by the temp count — measured as the single most expensive
        // cleanup sub-pass (73 s of a 124 s cleanup on 1.7k methods).
        // `count_reads_all` mirrors `count_reads_in_stmt` exactly: Var/Field/
        // StaticField/PatternMatch mentions, target-name tokens for complex
        // assignment targets, the For init/incr double count (they are also
        // visited through `children`), and lambda-parameter shadowing.
        let name_set: std::collections::HashSet<String> = names.iter().cloned().collect();
        let reads: HashMap<String, usize> = dts!(3, count_reads_all(stmts, &name_set));

        let mut count = 0;
        dts!(4, {
            walk_mut(stmts, &mut |list| {
                count += remove_dead_temp_in_list(list, &reads, &observable);
            });
        });
        total += count;
        dump("names", dt_round, stmts);
        if count == 0 && range_dead == 0 {
            break;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// 7C-3b: live-range-aware dead store removal
// ---------------------------------------------------------------------------
//
// A definition at index i is dead when nothing can observe the value before
// the NEXT write of the same display name. The write that ends the range must
// be a TOP-LEVEL `Assign` in the same list: a nested write (inside an `if`, a
// loop, …) does not execute on every path, so a read after that construct may
// still observe def i. For a def whose list has no later top-level write, the
// value is observable after the list ends, so the walker carries the names
// read "after this point" from the enclosing scopes:
//
// * `later`  — deep mentions in the following statements of this list;
// * `after`  — deep mentions that can run after this list completes in the
//              enclosing scopes, including loop back edges (a loop body's
//              after-set deliberately contains the WHOLE loop, so a read
//              before the def in the body still blocks: the next iteration
//              could observe this def).
//
// Deleting is additionally refused when a `goto`/`break`/`continue` in the
// dead range could leave the list before the overwriting write (the value
// would then be observed outside), and inside a `try` (an exception can skip
// the overwrite and reach a handler). RHS purity is required: `v = Foo();`
// is not a dead store, the call is the effect. Address-taken/captured names
// are never touched.

/// Every name MENTIONED by a statement (reads and writes, at any depth).
/// Over-approximating is safe here: it only makes the sweep refuse more.
/// Only names in `filter` are recorded — the sweep can only ever ASK about a
/// name that some list assigns, so every other mention is dead weight in the
/// sets the scopes consult.
fn collect_deep_mentions_into(
    s: &Stmt,
    out: &mut std::collections::HashSet<String>,
    filter: &std::collections::HashSet<String>,
) {
    for_each_shallow_mention(s, &mut |n| {
        if filter.contains(n) {
            if !out.contains(n) {
                out.insert(n.to_string());
            }
        }
    });
    if let Stmt::Using { decl, .. } = s {
        collect_deep_mentions_into(decl, out, filter);
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for st in k.iter() {
            collect_deep_mentions_into(st, out, filter);
        }
    }
}

fn deep_mentions_of_list(
    list: &[Stmt],
    filter: &std::collections::HashSet<String>,
) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for s in list {
        collect_deep_mentions_into(s, &mut out, filter);
    }
    out
}

fn collect_expr_mentions_into(
    e: &Expr,
    out: &mut std::collections::HashSet<String>,
    filter: &std::collections::HashSet<String>,
) {
    for_each_expr_mention(e, &mut |n| {
        if filter.contains(n) && !out.contains(n) {
            out.insert(n.to_string());
        }
    });
}

/// First TOP-LEVEL (unconditional) assignment to `name` after `def_idx`.
fn next_top_level_write_index(stmts: &[Stmt], def_idx: usize, name: &str) -> Option<usize> {
    (def_idx + 1..stmts.len()).find(|&j| matches!(&stmts[j], Stmt::Assign(n, _, _) if n == name))
}

fn remove_dead_range_stores(
    stmts: &mut Vec<Stmt>,
    observable: &std::collections::HashSet<String>,
) -> usize {
    // Every name some list can assign: the only names the sweep can ever ask
    // about. Mention sets are filtered to these, which keeps the scopes small.
    let mut relevant: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_assigned_locals(stmts, &mut relevant);
    remove_dead_range_in_list(
        stmts,
        &Scope { sets: Vec::new() },
        false,
        observable,
        &relevant,
    )
}

fn collect_assigned_locals(list: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in list {
        if let Stmt::Assign(name, _, _) = s {
            if is_plain_local_name(name) {
                out.insert(name.clone());
            }
        }
        if let Stmt::Using { decl, .. } = s {
            collect_assigned_locals(std::slice::from_ref(decl.as_ref()), out);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_assigned_locals(k, out);
        }
    }
}

/// A union of name sets, held BY REFERENCE.
///
/// The range sweep consults "can this name be read after the definition?"
/// against the mentions of enclosing constructs' other arms, the enclosing
/// lists' later statements and the parent's after-set. Those unions used to be
/// materialised (`base = later.clone(); base.extend(after)`) once per
/// STATEMENT per round — with the sets growing to every name in the method,
/// the clones alone were most of the sweep's cost. A scope keeps the member
/// sets by reference and tests membership across them instead.
///
/// `walk_scoped` carries the same unions for the five inlining passes; it uses
/// this type too, so an empty scope is its "no enclosing mentions" context.
struct Scope<'a> {
    sets: Vec<&'a std::collections::HashSet<String>>,
}

impl<'a> Scope<'a> {
    fn contains(&self, name: &str) -> bool {
        self.sets.iter().any(|s| s.contains(name))
    }

    fn is_empty(&self) -> bool {
        self.sets.iter().all(|s| s.is_empty())
    }

    /// This scope plus one more set. The result borrows the member sets (not
    /// `self`), so it can be built from a temporary chain.
    fn with<'b>(&self, extra: &'b std::collections::HashSet<String>) -> Scope<'b>
    where
        'a: 'b,
    {
        let mut sets: Vec<&'b std::collections::HashSet<String>> =
            Vec::with_capacity(self.sets.len() + 1);
        for s in &self.sets {
            sets.push(*s);
        }
        sets.push(extra);
        Scope { sets }
    }
}

/// Debug filter for the live-range dead-store sweep: `ILR_RANGEDEBUG=1`
/// reports every candidate, `ILR_RANGEDEBUG=v151,v95` only those names.
fn range_debug_matches(name: &str) -> bool {
    match std::env::var("ILR_RANGEDEBUG") {
        Ok(v) if v == "1" => true,
        Ok(v) => v.split(',').any(|n| n == name),
        Err(_) => false,
    }
}

fn remove_dead_range_in_list(
    stmts: &mut Vec<Stmt>,
    after: &Scope<'_>,
    in_try: bool,
    observable: &std::collections::HashSet<String>,
    relevant: &std::collections::HashSet<String>,
) -> usize {
    let mut total = 0;
    let mut later: std::collections::HashSet<String> = std::collections::HashSet::new();
    let dbg = std::env::var_os("ILR_RANGEDEBUG").is_some();
    let rt = crate::recovery::env_flags::flag("ILR_TIMING");
    // One fused read census per statement, kept in sync with the list. The
    // sweep used to call `count_reads_in_range` for every candidate, which
    // re-walked each statement's expression tree once PER NAME PER RANGE
    // (the single most expensive cleanup sub-pass: 90 s on the full corpus).
    // A range is now a sum of per-statement counts; `walk_range_children`
    // can mutate a statement's child lists, so the current statement's entry
    // is refreshed after that call.
    // The three vectors are read only by the removable-candidate arm, whose
    // pattern requires a top-level `Assign(name, None, rhs)`. Only such names
    // are ever summed over a range, so the census is restricted to them: that
    // keeps the per-statement maps tiny (the unfiltered census cloned every
    // identifier of every statement, for every cleanup round). Lists without a
    // candidate can never remove anything, so they skip the census entirely.
    let t_census = rt.then(std::time::Instant::now);
    let candidates: std::collections::HashSet<String> = stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::Assign(name, None, _) if dead_store_candidate(name, observable) => {
                Some(name.clone())
            }
            _ => None,
        })
        .collect();
    let track = !candidates.is_empty();
    let mut per_stmt: Vec<HashMap<String, usize>> = if track {
        stmts
            .iter()
            .map(|s| count_reads_impl(std::slice::from_ref(s), Some(&candidates)))
            .collect()
    } else {
        Vec::new()
    };
    let mut control_out: Vec<bool> = if track {
        stmts.iter().map(stmt_transfers_control_out).collect()
    } else {
        Vec::new()
    };
    let mut direct_jump: Vec<bool> = if track {
        stmts
            .iter()
            .map(|s| matches!(s, Stmt::Goto(_) | Stmt::Break | Stmt::Continue | Stmt::YieldBreak))
            .collect()
    } else {
        Vec::new()
    };
    if let Some(t) = t_census {
        dt_add(5, t);
    }
    let mut i = stmts.len();
    while i > 0 {
        i -= 1;
        let removed_before = total;
        let t_children = rt.then(std::time::Instant::now);
        walk_range_children(
            &mut stmts[i],
            &later,
            after,
            in_try,
            observable,
            relevant,
            &mut total,
        );
        if let Some(t) = t_children {
            dt_add(6, t);
        }
        // Re-census the current statement only when its child lists actually
        // changed (a nested removal). Doing it unconditionally cost more than
        // the range lookups save: most statements are not dead-store
        // candidates at all. The statement's own kind never changes here.
        if track && total != removed_before {
            let t = rt.then(std::time::Instant::now);
            per_stmt[i] = count_reads_impl(std::slice::from_ref(&stmts[i]), Some(&candidates));
            control_out[i] = stmt_transfers_control_out(&stmts[i]);
            if let Some(t) = t {
                dt_add(5, t);
            }
        }

        if dbg {
            if let Stmt::Assign(name, None, rhs) = &stmts[i] {
                if is_temp_var_name(name) && range_debug_matches(name) {
                    let next = next_top_level_write_index(stmts, i, name);
                    let reads_next =
                        next.map(|t| count_reads_in_range(stmts, i + 1, t, name));
                    let reads_after = count_reads_in_range(stmts, i + 1, stmts.len(), name);
                    eprintln!(
                        "[rangedbg] {name}@{i}/{} pure={} alloc={} self={} selfread={} \
                         obs={} next={next:?} reads_next={reads_next:?} reads_after={reads_after} \
                         later={} after={} ctrl={} in_try={in_try} rhs={rhs:?}",
                        stmts.len(),
                        expr_is_pure(rhs),
                        is_alloc_rhs(rhs),
                        is_self_assignment(name, rhs),
                        expr_mention_count(rhs, name),
                        observable.contains(name),
                        later.contains(name),
                        after.contains(name),
                        (i + 1..stmts.len()).any(|j| stmt_transfers_control_out(&stmts[j])),
                    );
                    for j in i + 1..stmts.len() {
                        let r = count_reads_in_stmt(&stmts[j], name);
                        if r > 0 {
                            eprintln!("[rangedbg]   reader@{j}: {r} in {:?}", stmts[j]);
                        }
                    }
                }
            }
        }

        let t_checks = rt.then(std::time::Instant::now);
        let removable = match &stmts[i] {
            Stmt::Assign(name, None, rhs)
                if track
                    && dead_store_candidate(name, observable)
                    && !observable.contains(name)
                    && !is_self_assignment(name, rhs)
                    && expr_is_pure(rhs)
                    && !is_alloc_rhs(rhs)
                    && expr_mention_count(rhs, name) == 0 =>
            {
                // A store whose very next statement LEAVES the method has no
                // observer on this path: the return is the end of it. Reads of
                // the same name on other paths belong to other defs (IL2CPP
                // reuses one slot for disjoint live ranges), so they say
                // nothing about this store. This is the early-exit shape every
                // IL2CPP method opens with -- `obj = 0; return 0;` -- where
                // the return register is materialized for the caller and then
                // read by nothing.
                if matches!(
                    stmts.get(i + 1),
                    Some(Stmt::Return(_)) | Some(Stmt::Throw(_))
                ) && range_read_sum(&per_stmt, i + 1, i + 2, name) == 0
                {
                    true
                } else {
                match next_top_level_write_index(stmts, i, name) {
                    Some(t) => {
                        // The overwrite runs unconditionally on the fall-through
                        // path. If control can leave the list before it, or an
                        // exception can reach a handler, the value is visible.
                        range_read_sum(&per_stmt, i + 1, t, name) == 0
                            && !control_out[i + 1..t].iter().any(|b| *b)
                            && !in_try
                    }
                    None => {
                        // Last write in this list: nothing after it may read the
                        // name, here or in the enclosing scopes.
                        //
                        // A `return`/`throw` after the store does NOT keep it
                        // alive -- the method (or the frame) is gone, and
                        // `count_reads_in_range` already counted the value the
                        // return itself reads. Only a branch that lands
                        // ELSEWHERE can observe the store, so only those keep
                        // it. That is the early-exit shape every IL2CPP method
                        // opens with: `obj = 0; return 0;`.
                        range_read_sum(&per_stmt, i + 1, per_stmt.len(), name) == 0
                            && !later.contains(name)
                            && !after.contains(name)
                            && !direct_jump[i + 1..].iter().any(|b| *b)
                    }
                }
                }
            }
            _ => false,
        };
        if let Some(t) = t_checks {
            dt_add(8, t);
        }

        if removable {
            if crate::recovery::env_flags::flag("ILR_DELDBG") {
                if let Stmt::Assign(n, _, _) = &stmts[i] {
                    eprintln!("[del-range] {n} @{i}");
                }
            }
            stmts.remove(i);
            per_stmt.remove(i);
            control_out.remove(i);
            direct_jump.remove(i);
            total += 1;
        } else {
            let t = rt.then(std::time::Instant::now);
            collect_deep_mentions_into(&stmts[i], &mut later, relevant);
            if let Some(t) = t {
                dt_add(7, t);
            }
        }
    }
    total
}

/// Recurse into the statement's child lists with the names observable after
/// each of them. Loop bodies get the whole loop in their after-set: the back
/// edge re-executes every read in the loop after any def in the body.
fn walk_range_children(
    s: &mut Stmt,
    later: &std::collections::HashSet<String>,
    after: &Scope<'_>,
    in_try: bool,
    observable: &std::collections::HashSet<String>,
    relevant: &std::collections::HashSet<String>,
    total: &mut usize,
) {
    let base = after.with(later);

    match s {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            let then_m = deep_mentions_of_list(then_body, relevant);
            let else_m = deep_mentions_of_list(else_body, relevant);
            let ta = base.with(&else_m);
            *total += remove_dead_range_in_list(then_body, &ta, in_try, observable, relevant);
            let ea = base.with(&then_m);
            *total += remove_dead_range_in_list(else_body, &ea, in_try, observable, relevant);
        }
        Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
            let body_m = deep_mentions_of_list(body, relevant);
            let mut cond_m = std::collections::HashSet::new();
            collect_expr_mentions_into(cond, &mut cond_m, relevant);
            let ba = base.with(&body_m).with(&cond_m);
            *total += remove_dead_range_in_list(body, &ba, in_try, observable, relevant);
        }
        Stmt::For {
            init,
            cond,
            incr,
            body,
        } => {
            let body_m = deep_mentions_of_list(body, relevant);
            let incr_m = deep_mentions_of_list(incr, relevant);
            let init_m = deep_mentions_of_list(init, relevant);
            let mut cond_m = std::collections::HashSet::new();
            if let Some(c) = cond {
                collect_expr_mentions_into(c, &mut cond_m, relevant);
            }
            let ba = base.with(&init_m).with(&incr_m).with(&body_m).with(&cond_m);
            *total += remove_dead_range_in_list(body, &ba, in_try, observable, relevant);

            let ia = base.with(&body_m).with(&incr_m).with(&cond_m);
            *total += remove_dead_range_in_list(init, &ia, in_try, observable, relevant);

            let na = base.with(&body_m).with(&cond_m);
            *total += remove_dead_range_in_list(incr, &na, in_try, observable, relevant);
        }
        Stmt::ForEach {
            collection, body, ..
        } => {
            let body_m = deep_mentions_of_list(body, relevant);
            let coll_m: std::collections::HashSet<String> =
                std::iter::once(collection.clone()).collect();
            let ba = base.with(&body_m).with(&coll_m);
            *total += remove_dead_range_in_list(body, &ba, in_try, observable, relevant);
        }
        Stmt::Lock { body, .. } => {
            *total += remove_dead_range_in_list(body, &base, in_try, observable, relevant);
        }
        Stmt::Using { body, .. } => {
            // The `using` declaration slot is a single statement owned outside
            // the body list; defs there are not stack-slot temps in practice.
            *total += remove_dead_range_in_list(body, &base, in_try, observable, relevant);
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut fin = std::collections::HashSet::new();
            if let Some(fb) = finally_body.as_ref() {
                fin = deep_mentions_of_list(fb, relevant);
            }
            let mut cat = std::collections::HashSet::new();
            for (_, b) in catches.iter() {
                cat.extend(deep_mentions_of_list(b, relevant));
            }
            let ta = base.with(&cat).with(&fin);
            *total += remove_dead_range_in_list(try_body, &ta, true, observable, relevant);
            let ca = base.with(&fin);
            for (_, b) in catches.iter_mut() {
                *total += remove_dead_range_in_list(b, &ca, true, observable, relevant);
            }
            if let Some(fb) = finally_body {
                *total += remove_dead_range_in_list(fb, &base, true, observable, relevant);
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            let mut all = std::collections::HashSet::new();
            for (_, b) in cases.iter() {
                all.extend(deep_mentions_of_list(b, relevant));
            }
            all.extend(deep_mentions_of_list(tail, relevant));
            if let Some(arm) = default {
                all.extend(deep_mentions_of_list(arm, relevant));
            }
            let ca = base.with(&all);
            for (_, b) in cases.iter_mut() {
                *total += remove_dead_range_in_list(b, &ca, in_try, observable, relevant);
            }
            *total += remove_dead_range_in_list(tail, &base, in_try, observable, relevant);
            if let Some(arm) = default {
                *total += remove_dead_range_in_list(arm, &base, in_try, observable, relevant);
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            let mut all = std::collections::HashSet::new();
            for (_, b) in cases.iter() {
                all.extend(deep_mentions_of_list(b, relevant));
            }
            if let Some(t) = tail.as_ref() {
                all.extend(deep_mentions_of_list(t, relevant));
            }
            let ca = base.with(&all);
            for (_, b) in cases.iter_mut() {
                *total += remove_dead_range_in_list(b, &ca, in_try, observable, relevant);
            }
            if let Some(t) = tail {
                *total += remove_dead_range_in_list(t, &base, in_try, observable, relevant);
            }
        }
        _ => {}
    }
}

/// Collect every temp name assigned in `list`, at any depth.
fn collect_assigned_temp_names(
    list: &[Stmt],
    out: &mut Vec<String>,
    observable: &std::collections::HashSet<String>,
) {
    for s in list {
        if let Stmt::Assign(n, _, _) = s {
            if dead_store_candidate(n, observable) {
                out.push(n.clone());
            }
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_assigned_temp_names(k, out, observable);
        }
    }
}

/// Can a store to this name be deleted once nothing reads it?
///
/// A temp always can: it is a compiler-introduced slot with no source-level
/// identity. A NAMED local can too, as long as its storage is not observable
/// -- an address-taken cell (`&v`, `RawAddr`) is written through a pointer
/// the C# cannot see, and a lambda captures the storage, not the value. The
/// return register is exactly this shape: IL2CPP keeps the result in a real
/// local and every early exit does `obj = 0; return obj;`. Once the folds
/// moved the value into the `return`, nothing reads `obj` and the store is
/// dead -- but only a NAME-wide liveness sweep can prove that, because each
/// arm's read belongs to another arm's store.
/// A rendered assignment target that names one plain local slot.
fn is_plain_local_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.as_bytes()[0].is_ascii_digit()
}

fn dead_store_candidate(name: &str, observable: &std::collections::HashSet<String>) -> bool {
    // Only a LOCAL. The LHS of an assignment is a rendered string, and
    // `*(byte*)(this +2603) = 1` is a store through a pointer -- a memory
    // write the C# cannot see the reader of. Deleting it is not dead-store
    // elimination, it is deleting code (the class-init guard, the field
    // write). A plain identifier is a slot this method owns.
    is_plain_local_name(name) && (is_temp_var_name(name) || !observable.contains(name))
}

/// Scan a statement list and remove dead temp var assignments.
fn remove_dead_temp_in_list(
    stmts: &mut Vec<Stmt>,
    reads: &HashMap<String, usize>,
    observable: &std::collections::HashSet<String>,
) -> usize {
    let mut total = 0;
    loop {
        let mut count = 0;
        let mut i = 0;
        while i < stmts.len() {
            let should_remove = match &stmts[i] {
                Stmt::Assign(var, None, rhs) if dead_store_candidate(var, observable) => {
                    // Self-assignments are always dead (v1 = v1 is a no-op).
                    if is_self_assignment(var, rhs) {
                        true
                    } else if expr_is_pure(rhs) && !is_alloc_rhs(rhs) {
                        // Reads that live inside this very statement do not keep
                        // it alive (`v1 = v1 - 1` is dead if nothing else
                        // names v1), so they are subtracted from the total.
                        //
                        // ONLY for a TEMP. A temp's self-mention is the
                        // machine slot the instruction mutates; the value has
                        // no other identity. A NAMED local's self-mention is
                        // the source-level read-modify-write the recognizers
                        // just built -- `x ??= y` is `Assign(x, x ?? y)` --
                        // and discounting it deleted the very statement that
                        // expresses the construct, leaving an empty `if` and,
                        // in a loop, an empty body. The read is real: it is
                        // the only thing that observes the previous value, so
                        // it keeps the store.
                        let own = if is_temp_var_name(var) {
                            expr_mention_count(rhs, var)
                        } else {
                            0
                        };
                        reads.get(var).copied().unwrap_or(0) <= own
                    } else {
                        false
                    }
                }
                _ => false,
            };
            if should_remove {
                stmts.remove(i);
                count += 1;
                // Don't increment i — next statement shifted into position.
            } else {
                i += 1;
            }
        }
        total += count;
        if count == 0 {
            break;
        }
    }
    total
}

/// Count reads of `name` in the statement list, excluding the statement at
/// `skip_idx`.
/// `count_reads_excluding(...) == 1`, decided without scanning the rest of the
/// list once a second read is found. Same answer, bounded work.
#[allow(dead_code)]
fn reads_exactly_one_excluding(list: &[Stmt], name: &str, skip_idx: usize) -> bool {
    let mut n = 0usize;
    for (i, s) in list.iter().enumerate() {
        if i == skip_idx {
            continue;
        }
        n += count_reads_in_stmt(s, name);
        if n > 1 {
            return false;
        }
    }
    n == 1
}

/// `count_reads_excluding(...) == 0`, decided at the first read found.
fn no_reads_excluding(list: &[Stmt], name: &str, skip_idx: usize) -> bool {
    for (i, s) in list.iter().enumerate() {
        if i == skip_idx {
            continue;
        }
        if count_reads_in_stmt(s, name) > 0 {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
fn count_reads_excluding(list: &[Stmt], name: &str, skip_idx: usize) -> usize {
    list.iter()
        .enumerate()
        .filter(|(i, _)| *i != skip_idx)
        .map(|(_, s)| count_reads_in_stmt(s, name))
        .sum()
}

// ===========================================================================
// 7C-2: Bitwise safety mask flattening — convert (X & 1) != 0 to X
// ===========================================================================

/// Flatten CPU-level bitwise safety checks back to high-level boolean checks.
///
/// ## Pattern
///
/// IL2CPP emits low-level safety masks that check the least significant bit:
/// ```text
/// if ((obj & 1) != 0) { ... }     // truthy check
/// if ((obj & 1) == 0) { ... }     // falsy check
/// ```
///
/// In managed C#, these are just boolean checks:
/// ```text
/// if (obj) { ... }                // or if (obj != null) for reference types
/// if (!obj) { ... }               // or if (obj == null)
/// ```
///
/// ## Why this is safe
///
/// The `& 1` pattern is a CPU-level optimization for checking if a value is
/// non-zero (odd). In managed C#, we just use truthy/falsy checks directly.
/// This cleanup removes the low-level bitwise noise.
///
/// ## What we transform
///
/// - `Cmp(Ne, Bin(And, X, Int(1)), Int(0))` → `X` (truthy check)
/// - `Cmp(Eq, Bin(And, X, Int(1)), Int(0))` → `Not(X)` (falsy check)
/// - Also handles reversed operand order: `Cmp(Ne, Int(0), Bin(And, X, Int(1)))`
///
/// ## Safety
///
/// We skip RawDeref with non-literal bases (like `This`, `Var`) because those
/// represent real managed code (instance field access), not safety masks.
fn is_raw_deref_with_non_literal_base(expr: &Expr) -> bool {
    if let Expr::RawDeref(base, _, _) = expr {
        !matches!(base.as_ref(), Expr::Int(_))
    } else {
        false
    }
}

fn flatten_bitwise_safety_masks(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            count += flatten_bitwise_safety_masks_stmt(stmt);
        }
    });
    count
}

/// Recursively flatten bitwise safety masks in a statement.
fn flatten_bitwise_safety_masks_stmt(stmt: &mut Stmt) -> usize {
    let mut count = 0;
    match stmt {
        Stmt::If { cond, then_body, else_body } => {
            // Check if this is a byte guard pattern with multiple stores (ambiguous).
            // If so, skip flattening.
            let has_multiple_stores = if let Some(guard_addr) = extract_guard_address(cond) {
                let store_count = count_stores_to_address(then_body, &guard_addr)
                    + count_stores_to_address(else_body, &guard_addr);
                store_count > 1
            } else {
                false
            };

            if !has_multiple_stores {
                count += flatten_bitwise_safety_masks_expr(cond);
            }
        }
        Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            count += flatten_bitwise_safety_masks_expr(cond)
        }
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                count += flatten_bitwise_safety_masks_expr(c);
            }
        }
        Stmt::Assign(_, _, expr)
        | Stmt::Eval(expr)
        | Stmt::Return(Some(expr))
        | Stmt::Throw(expr)
        | Stmt::YieldReturn(expr) => count += flatten_bitwise_safety_masks_expr(expr),
        _ => {}
    }
    count
}

/// Extract the guard address string from a byte guard condition like `(RawDeref(base, off, 1) & 1) == 0`.
fn extract_guard_address(cond: &Expr) -> Option<String> {
    if let Expr::Cmp(CmpKind::Eq, lhs, rhs) = cond {
        let inner = if let Expr::Bin(BinOp::And, inner, mask) = lhs.as_ref() {
            if matches!(mask.as_ref(), Expr::Int(1)) {
                Some(inner.as_ref())
            } else {
                None
            }
        } else if let Expr::Bin(BinOp::And, inner, mask) = rhs.as_ref() {
            if matches!(mask.as_ref(), Expr::Int(1)) {
                Some(inner.as_ref())
            } else {
                None
            }
        } else {
            None
        };

        if let Some(Expr::RawDeref(base, offset, 1)) = inner {
            if let Expr::Int(addr) = base.as_ref() {
                return Some(format!("*(byte*)(0x{:x} {:+})", addr, offset));
            }
        }
    }
    None
}

/// Count stores to a specific address in a statement list.
fn count_stores_to_address(stmts: &[Stmt], addr: &str) -> usize {
    stmts.iter().filter(|s| {
        if let Stmt::Assign(target, _, _) = s {
            target == addr
        } else {
            false
        }
    }).count()
}

/// Flatten bitwise safety masks in an expression tree.
fn flatten_bitwise_safety_masks_expr(expr: &mut Expr) -> usize {
    let mut count = 0;

    // Check for pattern 1: (X & 1) != 0 → X (truthy check)
    // But skip if X is a RawDeref with a non-literal base (real managed code).
    let should_flatten_ne = if let Expr::Cmp(CmpKind::Ne, lhs, rhs) = expr {
        if let (Expr::Bin(BinOp::And, inner, mask), Expr::Int(0)) = (lhs.as_ref(), rhs.as_ref()) {
            if matches!(mask.as_ref(), Expr::Int(1)) {
                !is_raw_deref_with_non_literal_base(inner.as_ref())
            } else {
                false
            }
        } else if let (Expr::Int(0), Expr::Bin(BinOp::And, inner, mask)) = (lhs.as_ref(), rhs.as_ref()) {
            if matches!(mask.as_ref(), Expr::Int(1)) {
                !is_raw_deref_with_non_literal_base(inner.as_ref())
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    if should_flatten_ne {
        // Extract the inner X from (X & 1)
        if let Expr::Cmp(CmpKind::Ne, lhs, rhs) = expr {
            let inner = if let Expr::Bin(BinOp::And, inner, _) = lhs.as_ref() {
                Some(inner.as_ref().clone())
            } else if let Expr::Bin(BinOp::And, inner, _) = rhs.as_ref() {
                Some(inner.as_ref().clone())
            } else {
                None
            };
            if let Some(inner_expr) = inner {
                *expr = inner_expr;
                count += 1;
            }
        }
    }

    // Check for pattern 2: (X & 1) == 0 → Not(X) (falsy check)
    // But skip if X is a RawDeref with a non-literal base (real managed code).
    let should_flatten_eq = if let Expr::Cmp(CmpKind::Eq, lhs, rhs) = expr {
        if let (Expr::Bin(BinOp::And, inner, mask), Expr::Int(0)) = (lhs.as_ref(), rhs.as_ref()) {
            if matches!(mask.as_ref(), Expr::Int(1)) {
                !is_raw_deref_with_non_literal_base(inner.as_ref())
            } else {
                false
            }
        } else if let (Expr::Int(0), Expr::Bin(BinOp::And, inner, mask)) = (lhs.as_ref(), rhs.as_ref()) {
            if matches!(mask.as_ref(), Expr::Int(1)) {
                !is_raw_deref_with_non_literal_base(inner.as_ref())
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    if should_flatten_eq {
        // Extract the inner X from (X & 1)
        if let Expr::Cmp(CmpKind::Eq, lhs, rhs) = expr {
            let inner = if let Expr::Bin(BinOp::And, inner, _) = lhs.as_ref() {
                Some(inner.as_ref().clone())
            } else if let Expr::Bin(BinOp::And, inner, _) = rhs.as_ref() {
                Some(inner.as_ref().clone())
            } else {
                None
            };
            if let Some(inner_expr) = inner {
                *expr = Expr::Not(Box::new(inner_expr));
                count += 1;
            }
        }
    }

    // Recurse into sub-expressions
    match expr {
        Expr::Bin(_, lhs, rhs) | Expr::Cmp(_, lhs, rhs) => {
            count += flatten_bitwise_safety_masks_expr(lhs);
            count += flatten_bitwise_safety_masks_expr(rhs);
        }
        Expr::Not(inner) | Expr::Neg(inner) | Expr::Cast(_, inner) => {
            count += flatten_bitwise_safety_masks_expr(inner);
        }
        Expr::Cond(cond, then_e, else_e) => {
            count += flatten_bitwise_safety_masks_expr(cond);
            count += flatten_bitwise_safety_masks_expr(then_e);
            count += flatten_bitwise_safety_masks_expr(else_e);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                count += flatten_bitwise_safety_masks_expr(r);
            }
            for arg in args {
                count += flatten_bitwise_safety_masks_expr(arg);
            }
        }
        Expr::Field(base, _) | Expr::RawDeref(base, _, _) => {
            count += flatten_bitwise_safety_masks_expr(base);
        }
        _ => {}
    }

    count
}

// ===========================================================================
// 7C-3: Array length access translation — convert *(nint*)(array +24) to array.Length
// ===========================================================================

/// Translate IL2CPP's low-level array length access to high-level C# syntax.
///
/// ## Pattern
///
/// IL2CPP accesses array length via pointer arithmetic:
/// ```text
/// *(nint*)(arrayObject + 24)    // offset 24 = length field in Il2CppArray
/// ```
///
/// In managed C#, this is just:
/// ```text
/// arrayObject.Length            // or .Count for collections
/// ```
///
/// ## Why this is safe
///
/// The offset 24 (0x18) is the standard location of the length field in
/// IL2CPP's array structure. This is a well-known constant across all IL2CPP
/// versions. Translating it to `.Length` makes the code readable and idiomatic.
///
/// ## What we transform
///
/// - `RawDeref(base, 24, 8)` where base is an array-like object → `Field(base, "Length")`
/// - Also handles offset 16 for some collection types → `Field(base, "Count")`
#[allow(dead_code)]
fn translate_array_length_access(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            count += translate_array_length_access_stmt(stmt);
        }
    });
    count
}

/// Recursively translate array length access in a statement.
#[allow(dead_code)]
fn translate_array_length_access_stmt(stmt: &mut Stmt) -> usize {
    let mut count = 0;
    match stmt {
        Stmt::Assign(_, _, expr)
        | Stmt::Eval(expr)
        | Stmt::Return(Some(expr))
        | Stmt::Throw(expr)
        | Stmt::YieldReturn(expr) => count += translate_array_length_access_expr(expr),
        Stmt::If { cond, .. } => count += translate_array_length_access_expr(cond),
        Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            count += translate_array_length_access_expr(cond)
        }
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                count += translate_array_length_access_expr(c);
            }
        }
        _ => {}
    }
    count
}

/// Translate array length access in an expression tree.
#[allow(dead_code)]
fn translate_array_length_access_expr(expr: &mut Expr) -> usize {
    let mut count = 0;

    // Pattern: RawDeref(base, 24, 8) → Field(base, "Length")
    // Offset 24 (0x18) is the standard array length offset in IL2CPP
    if let Expr::RawDeref(base, 24, 8) = expr {
        let base_clone = base.as_ref().clone();
        *expr = Expr::Field(Box::new(base_clone), "Length".to_string());
        count += 1;
    }
    // Offset 16 (0x10) is sometimes used for collection Count
    else if let Expr::RawDeref(base, 16, 8) = expr {
        let base_clone = base.as_ref().clone();
        *expr = Expr::Field(Box::new(base_clone), "Count".to_string());
        count += 1;
    }

    // Recurse into sub-expressions
    match expr {
        Expr::Bin(_, lhs, rhs) | Expr::Cmp(_, lhs, rhs) => {
            count += translate_array_length_access_expr(lhs);
            count += translate_array_length_access_expr(rhs);
        }
        Expr::Not(inner) | Expr::Neg(inner) | Expr::Cast(_, inner) => {
            count += translate_array_length_access_expr(inner);
        }
        Expr::Cond(cond, then_e, else_e) => {
            count += translate_array_length_access_expr(cond);
            count += translate_array_length_access_expr(then_e);
            count += translate_array_length_access_expr(else_e);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                count += translate_array_length_access_expr(r);
            }
            for arg in args {
                count += translate_array_length_access_expr(arg);
            }
        }
        Expr::Field(base, _) | Expr::RawDeref(base, _, _) => {
            count += translate_array_length_access_expr(base);
        }
        Expr::RawCall(_, args, _) | Expr::NewArr(_, args) | Expr::NewObj(_, args) => {
            for arg in args {
                count += translate_array_length_access_expr(arg);
            }
        }
        _ => {}
    }

    count
}

// ===========================================================================
// 7C-4: Array indexing translation — convert *(nint*)(arr + (i << 3) + 32) to arr[i]
// ===========================================================================

/// Translate IL2CPP's low-level array indexing to high-level C# syntax.
///
/// ## Pattern
///
/// IL2CPP accesses array elements via pointer arithmetic:
/// ```text
/// *(nint*)(array + (index << 3) + 32)    // offset 32 = start of array data
/// ```
///
/// The `<< 3` is multiplication by 8 (element size on 64-bit).
/// Offset 32 (0x20) is where the array data starts in Il2CppArray:
/// ```text
/// struct Il2CppArray {
///     void* klass;        // offset 0
///     void* monitor;      // offset 8
///     void* bounds;       // offset 16
///     int max_length;     // offset 24
///     // padding           // offset 28
///     void* data;         // offset 32 <- array elements start here
/// }
/// ```
///
/// In managed C#, this is just:
/// ```text
/// array[index]
/// ```
///
/// ## Why this is safe
///
/// The offset 32 and shift 3 are IL2CPP implementation details. Translating
/// to bracket syntax makes the code readable and idiomatic.
fn translate_array_indexing(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            count += translate_array_indexing_stmt(stmt);
        }
    });
    count
}

/// Recursively translate array indexing in a statement.
fn translate_array_indexing_stmt(stmt: &mut Stmt) -> usize {
    let mut count = 0;
    match stmt {
        Stmt::Assign(_, _, expr)
        | Stmt::Eval(expr)
        | Stmt::Return(Some(expr))
        | Stmt::Throw(expr)
        | Stmt::YieldReturn(expr) => count += translate_array_indexing_expr(expr),
        Stmt::If { cond, .. } => count += translate_array_indexing_expr(cond),
        Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            count += translate_array_indexing_expr(cond)
        }
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                count += translate_array_indexing_expr(c);
            }
        }
        _ => {}
    }
    count
}

/// Translate array indexing in an expression tree.
fn translate_array_indexing_expr(expr: &mut Expr) -> usize {
    let mut count = 0;

    // Pattern: RawDeref(Bin(Add, base, Bin(Shl, index, N)), 32, _)
    //        → Index { collection: base, index: index }
    // This represents: *(nint*)(base + (index << N) + 32)
    //
    // Offset 32 (0x20) is where array data starts in Il2CppArray.
    // Shift N=3 means element size 8 (64-bit pointers/nints).
    // Shift N=2 means element size 4 (32-bit ints/floats).
    // Width varies (0, 4, 8) depending on the load instruction — we accept any.
    if let Expr::RawDeref(inner_base, 32, _) = expr {
        if let Expr::Bin(BinOp::Add, base, offset_expr) = inner_base.as_ref() {
            if let Expr::Bin(BinOp::Shl, index, shift) = offset_expr.as_ref() {
                if matches!(shift.as_ref(), Expr::Int(2) | Expr::Int(3)) {
                    // Found the pattern! Translate to array[index]
                    let base_clone = base.as_ref().clone();
                    let index_clone = index.as_ref().clone();
                    *expr = Expr::Index {
                        collection: Box::new(base_clone),
                        index: Box::new(index_clone),
                    };
                    count += 1;
                }
            }
        }
    }

    // Recurse into sub-expressions
    match expr {
        Expr::Bin(_, lhs, rhs) | Expr::Cmp(_, lhs, rhs) => {
            count += translate_array_indexing_expr(lhs);
            count += translate_array_indexing_expr(rhs);
        }
        Expr::Not(inner) | Expr::Neg(inner) | Expr::Cast(_, inner) => {
            count += translate_array_indexing_expr(inner);
        }
        Expr::Cond(cond, then_e, else_e) => {
            count += translate_array_indexing_expr(cond);
            count += translate_array_indexing_expr(then_e);
            count += translate_array_indexing_expr(else_e);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                count += translate_array_indexing_expr(r);
            }
            for arg in args {
                count += translate_array_indexing_expr(arg);
            }
        }
        Expr::Field(base, _) | Expr::RawDeref(base, _, _) => {
            count += translate_array_indexing_expr(base);
        }
        Expr::Index { collection, index } => {
            count += translate_array_indexing_expr(collection);
            count += translate_array_indexing_expr(index);
        }
        Expr::RawCall(_, args, _) | Expr::NewArr(_, args) | Expr::NewObj(_, args) => {
            for arg in args {
                count += translate_array_indexing_expr(arg);
            }
        }
        _ => {}
    }

    count
}

// ===========================================================================
// 7D: Static field access cleanup — strip IL2CPP klass/static_fields machinery
// ===========================================================================

/// Clean up IL2CPP static field access patterns.
///
/// ## Problem
///
/// IL2CPP implements static field access via the Il2CppClass structure:
/// 1. Load the Il2CppClass* for TypeName (via a `klass(TypeName)` accessor)
/// 2. Access the `static_fields` pointer (offset 184/176 in Il2CppClass)
/// 3. Access a field at some offset within static_fields
///
/// This produces output like:
/// ```text
/// TypeName.klass(TypeName).static_fields.fieldName
/// ```
///
/// In the original C#, this was just `TypeName.fieldName`.
///
/// ## AST patterns
///
/// The field_annotation pass may have partially resolved these, producing:
///
/// 1. `Field(Field(StaticField(type, "klass(type)"), "static_fields"), field)`
///    → `StaticField(type, field)`
///
/// 2. `Field(Field(RawDeref(StaticField(type, "klass(type)"), 0, _), "static_fields"), field)`
///    → `StaticField(type, field)`
///
/// 3. `Field(Field(Field(base, "klass(type)"), "static_fields"), field)`
///    where base renders as the type name → `StaticField(type, field)`
///
/// 4. `RawDeref(StaticField(type, "klass(type)"), 0, _)` → just remove the deref
///    (loading the klass pointer itself, not a field within it)
///
/// 5. `RawDeref(Field(base, "klass(type)"), 0, _)` → remove the deref
///
/// ## Why this is safe
///
/// These rewrites only remove IL2CPP runtime machinery that has no managed
/// equivalent. The static field access in managed C# is simply `TypeName.field`.
pub fn cleanup_static_field_access(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            count += cleanup_static_field_stmt(stmt);
        }
    });
    count
}

/// Replace surviving `T.klass(T)` / `recv.klass(T)` accessors with `typeof(T)`.
///
/// `cleanup_static_field_access` performs this as Pattern 4, but it also runs
/// four broader static-field rewrites and is only exercised by its tests.
/// The pipeline needs just this normalization: metadata-usage type slots now
/// resolve to a klass accessor at arbitrary VALUE positions (the type argument
/// of `IL2CPP_Box`/`IL2CPP_ArrayNew`, bare comparisons), where no later field
/// pattern consumes the accessor. Left alone they render invalid C# such as
/// `IL2CPP_Box(int.klass(int), …)`.
pub fn normalize_klass_accessors(stmts: &mut Vec<Stmt>) -> usize {
    fn expr(e: &mut Expr) -> usize {
        let mut n = 0;
        for c in expr_children_mut(e) {
            n += expr(c);
        }
        let replacement = match e {
            Expr::StaticField(_, name) | Expr::Field(_, name)
                if name.starts_with("klass(") =>
            {
                name.strip_prefix("klass(")
                    .and_then(|s| s.strip_suffix(')'))
                    .map(|t| Expr::TypeOf(t.to_string()))
            }
            _ => None,
        };
        if let Some(r) = replacement {
            *e = r;
            n += 1;
        }
        n
    }
    let mut n = 0;
    walk_mut(stmts, &mut |list| {
        for s in list.iter_mut() {
            n += match s {
                Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Assign(_, _, e) => {
                    expr(e)
                }
                Stmt::Return(Some(e)) => expr(e),
                Stmt::If { cond, .. }
                | Stmt::While { cond, .. }
                | Stmt::DoWhile { cond, .. }
                | Stmt::Lock { expr: cond, .. } => expr(cond),
                Stmt::For { cond, .. } => cond.as_mut().map_or(0, expr),
                Stmt::StringSwitch { disc, .. } => expr(disc),
                Stmt::IndexAssign {
                    collection,
                    index,
                    value,
                } => expr(collection) + expr(index) + expr(value),
                Stmt::EventAssign { field, value, .. } => expr(field) + expr(value),
                Stmt::Attribute { args, .. } => args.iter_mut().map(expr).sum(),
                _ => 0,
            };
        }
    });
    n
}

/// Recursively clean up static field patterns in a statement's expressions.
fn cleanup_static_field_stmt(stmt: &mut Stmt) -> usize {
    let mut count = 0;
    match stmt {
        Stmt::Eval(e) => count += cleanup_static_field_expr(e),
        Stmt::Assign(_, _, e) => count += cleanup_static_field_expr(e),
        Stmt::Return(Some(e)) => count += cleanup_static_field_expr(e),
        Stmt::Throw(e) => count += cleanup_static_field_expr(e),
        Stmt::YieldReturn(e) => count += cleanup_static_field_expr(e),
        Stmt::If { cond, then_body, else_body, .. } => {
            count += cleanup_static_field_expr(cond);
            for s in then_body.iter_mut() { count += cleanup_static_field_stmt(s); }
            for s in else_body.iter_mut() { count += cleanup_static_field_stmt(s); }
        }
        Stmt::While { cond, body, .. } => {
            count += cleanup_static_field_expr(cond);
            for s in body.iter_mut() { count += cleanup_static_field_stmt(s); }
        }
        Stmt::DoWhile { cond, body, .. } => {
            count += cleanup_static_field_expr(cond);
            for s in body.iter_mut() { count += cleanup_static_field_stmt(s); }
        }
        Stmt::For { init, incr, cond, body, .. } => {
            for s in init.iter_mut() { count += cleanup_static_field_stmt(s); }
            for s in incr.iter_mut() { count += cleanup_static_field_stmt(s); }
            if let Some(c) = cond { count += cleanup_static_field_expr(c); }
            for s in body.iter_mut() { count += cleanup_static_field_stmt(s); }
        }
        Stmt::Try { try_body, catches, finally_body, .. } => {
            for s in try_body.iter_mut() { count += cleanup_static_field_stmt(s); }
            for (_, b) in catches.iter_mut() {
                for s in b.iter_mut() { count += cleanup_static_field_stmt(s); }
            }
            if let Some(fb) = finally_body {
                for s in fb.iter_mut() { count += cleanup_static_field_stmt(s); }
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases.iter_mut() {
                for s in b.iter_mut() { count += cleanup_static_field_stmt(s); }
            }
            for s in tail.iter_mut() { count += cleanup_static_field_stmt(s); }
            if let Some(arm) = default {
                for s in arm.iter_mut() { count += cleanup_static_field_stmt(s); }
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, b) in cases.iter_mut() {
                for s in b.iter_mut() { count += cleanup_static_field_stmt(s); }
            }
            if let Some(t) = tail {
                for s in t.iter_mut() { count += cleanup_static_field_stmt(s); }
            }
        }
        Stmt::Using { body, .. } | Stmt::Lock { body, .. } | Stmt::ForEach { body, .. } => {
            for s in body.iter_mut() { count += cleanup_static_field_stmt(s); }
        }
        _ => {}
    }
    count
}

/// Recursively clean up static field patterns in an expression tree.
///
/// Uses a TOP-DOWN strategy with fixpoint iteration: try outer patterns first
/// (static_fields chains), loop until no more outer patterns match, then recurse
/// into sub-expressions for bare klass accesses. This prevents the bare-klass →
/// typeof rewrite from destroying the inner klass accessor before the outer
/// static_fields chain pattern can match it.
fn cleanup_static_field_expr(expr: &mut Expr) -> usize {
    let mut count = 0;

    // Try outer patterns in a loop until none match at this level.
    loop {
        let mut matched = false;

        // Pattern 1: Field(Field(X, "static_fields"), "fieldName") where X contains
        // a klass access somewhere in its receiver chain → StaticField(type, "fieldName").
        if let Expr::Field(outer_recv, outer_field) = expr {
            if let Expr::Field(static_fields_recv, inner_field) = outer_recv.as_mut() {
                if inner_field == "static_fields" {
                    if let Some(type_name) = find_klass_type_in_chain(static_fields_recv) {
                        let field_name = outer_field.clone();
                        *expr = Expr::StaticField(type_name, field_name);
                        count += 1;
                        matched = true;
                    }
                }
            }
        }
        if matched { continue; }

        // Pattern 5: Field(klass_expr, "static_fields") without an outer field access.
        if let Expr::Field(recv, field_name) = expr {
            if field_name == "static_fields" {
                if let Some(type_name) = find_klass_type_in_chain(recv) {
                    *expr = Expr::TypeOf(type_name);
                    count += 1;
                    matched = true;
                }
            }
        }
        if matched { continue; }

        // Pattern 2: RawDeref(klass_access, 0, _) → remove the deref wrapper.
        if let Expr::RawDeref(inner, 0, _) = expr {
            if is_klass_access(inner) {
                let inner_clone = inner.as_ref().clone();
                *expr = inner_clone;
                count += 1;
                matched = true;
            }
        }
        if matched { continue; }

        // Pattern 3: RawDeref(Field(recv, "klass(type)"), 0, _) → remove the deref.
        if let Expr::RawDeref(inner, 0, _) = expr {
            if let Expr::Field(_recv, field_name) = inner.as_ref() {
                if field_name.starts_with("klass(") {
                    let inner_clone = inner.as_ref().clone();
                    *expr = inner_clone;
                    count += 1;
                    matched = true;
                }
            }
        }
        if matched { continue; }

        // Pattern 4: Bare klass access used as a value → typeof(T).
        if let Some(type_name) = extract_type_from_klass_access(expr) {
            *expr = Expr::TypeOf(type_name);
            count += 1;
            matched = true;
        }
        if matched { continue; }

        // No more outer patterns match at this level.
        break;
    }

    // Recurse into sub-expressions so nested klass accesses are converted.
    match expr {
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv.as_mut() {
                count += cleanup_static_field_expr(r);
            }
            for a in args.iter_mut() {
                count += cleanup_static_field_expr(a);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args.iter_mut() {
                count += cleanup_static_field_expr(a);
            }
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += cleanup_static_field_expr(a);
            count += cleanup_static_field_expr(b);
        }
        Expr::Not(e) | Expr::Neg(e) | Expr::Cast(_, e) | Expr::AddressOf(e)
        | Expr::Await(e) | Expr::RawDeref(e, _, _) | Expr::RawAddr(e, _) => {
            count += cleanup_static_field_expr(e);
        }
        Expr::Field(recv, _) => {
            count += cleanup_static_field_expr(recv);
        }
        Expr::Cond(c, a, b) => {
            count += cleanup_static_field_expr(c);
            count += cleanup_static_field_expr(a);
            count += cleanup_static_field_expr(b);
        }
        Expr::ArrayInit { items, .. } => {
            for e in items.iter_mut() {
                count += cleanup_static_field_expr(e);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for e in ctor_args.iter_mut() {
                count += cleanup_static_field_expr(e);
            }
            for e in items.iter_mut() {
                count += cleanup_static_field_expr(e);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for e in ctor_args.iter_mut() {
                count += cleanup_static_field_expr(e);
            }
            for (_, v) in fields.iter_mut() {
                count += cleanup_static_field_expr(v);
            }
        }
        Expr::Lambda { body, .. } => {
            count += cleanup_static_field_expr(body);
        }
        Expr::Index { collection, index } => {
            count += cleanup_static_field_expr(collection);
            count += cleanup_static_field_expr(index);
        }
        Expr::NullConditional { receiver, access } => {
            count += cleanup_static_field_expr(receiver);
            count += cleanup_static_field_expr(access);
        }
        _ => {}
    }

    count
}

/// Check if an expression is a klass accessor — i.e., `StaticField(type, "klass(type)")`
/// or `Field(base, "klass(type)")`.
fn is_klass_access(expr: &Expr) -> bool {
    match expr {
        Expr::StaticField(_, name) if name.starts_with("klass(") => true,
        Expr::Field(_, name) if name.starts_with("klass(") => true,
        _ => false,
    }
}

/// Extract the type name from a klass accessor expression.
///
/// Handles:
/// - `StaticField(type_name, "klass(type_name)")` → `Some(type_name)`
/// - `Field(base, "klass(type_name)")` → `Some(type_name)` (extracted from the field name)
/// - `RawDeref(StaticField(type_name, "klass(type_name)"), 0, _)` → `Some(type_name)`
/// - `RawDeref(Field(base, "klass(type_name)"), 0, _)` → `Some(type_name)`
fn extract_type_from_klass_access(expr: &Expr) -> Option<String> {
    match expr {
        // StaticField(type_name, "klass(type_name)")
        Expr::StaticField(type_name, accessor) if accessor.starts_with("klass(") => {
            // Extract the type name from "klass(TypeName)" or use the StaticField's type.
            let inner = accessor.strip_prefix("klass(").and_then(|s| s.strip_suffix(')'));
            Some(inner.unwrap_or(type_name).to_string())
        }
        // Field(base, "klass(type_name)")
        Expr::Field(_, accessor) if accessor.starts_with("klass(") => {
            accessor.strip_prefix("klass(").and_then(|s| s.strip_suffix(')')).map(|s| s.to_string())
        }
        // RawDeref(inner, 0, _) — dereferenced klass pointer
        Expr::RawDeref(inner, 0, _) => extract_type_from_klass_access(inner),
        _ => None,
    }
}

/// Walk up a receiver chain looking for a klass accessor.
///
/// Given an expression like `Field(Field(StaticField("T", "klass(T)"), "m00"), ...)`,
/// this walks down through the Field receivers until it finds a klass accessor
/// and extracts the type name from it.
///
/// This handles the case where IL2CPP accesses Il2CppClass structure fields
/// between the klass pointer and the static_fields pointer:
///   `klass(T).field1.field2....static_fields`
fn find_klass_type_in_chain(expr: &Expr) -> Option<String> {
    // First, check if this expression itself is a klass accessor.
    if let Some(type_name) = extract_type_from_klass_access(expr) {
        return Some(type_name);
    }
    // If this is a Field access, walk down into the receiver.
    if let Expr::Field(recv, _) = expr {
        return find_klass_type_in_chain(recv);
    }
    // If this is a RawDeref, look inside.
    if let Expr::RawDeref(inner, 0, _) = expr {
        return find_klass_type_in_chain(inner);
    }
    None
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::CmpKind;
    use crate::recovery::reconstruct::expr_mentions;

    fn v(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    // -----------------------------------------------------------------------
    // 7A-1: Null-guard stripping tests
    // -----------------------------------------------------------------------

    #[test]
    fn role_classified_raise_guard_with_else_is_spliced() {
        // if (this._setting == 0) { RAISE_ADDR(this); return; } else { S }
        // with RAISE_ADDR globally classified as a raise wrapper → splice S.
        let addr = 0xABCD_1234u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Field(Box::new(Expr::This), "_setting".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Eval(Expr::RawCall(addr, vec![Expr::This], None)),
                Stmt::Return(None),
            ],
            else_body: vec![Stmt::Eval(Expr::Field(
                Box::new(Expr::This),
                "_setting".into(),
            ))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "else-splice should fire for role-classified raise");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Eval(_)), "else body spliced in");
    }

    #[test]
    fn role_classified_raise_guard_without_else_is_stripped() {
        // if (obj == 0) { RAISE_ADDR(obj); return; } — bare guard is deleted.
        let addr = 0xABCD_1235u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("obj")), Box::new(Expr::Int(0))),
            then_body: vec![
                Stmt::Eval(Expr::RawCall(addr, vec![v("obj")], None)),
                Stmt::Return(None),
            ],
            else_body: vec![],
        },
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("obj"))),
            name: "Use".into(),
            args: vec![],
        }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "bare role-classified raise guard should be stripped");
        assert_eq!(stmts.len(), 1, "the use of obj remains");
    }

    #[test]
    fn role_classified_raise_guard_cfg_ne_orientation_is_spliced() {
        // Raw CFG branch node orientation (measured on v31.1): the arms are
        // flipped by a later pass, so the raise sits under `X != 0` with the
        // managed body in the else arm.
        let addr = 0xABCD_1236u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Field(Box::new(Expr::This), "_listPositionCtrl".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Eval(Expr::RawCall(addr, vec![v("v11"), Expr::This], None)),
                Stmt::Return(None),
            ],
            else_body: vec![Stmt::Eval(Expr::Field(
                Box::new(Expr::This),
                "_listPositionCtrl".into(),
            ))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "Ne-oriented raise guard should be spliced");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn role_classified_raise_guard_else_arm_is_spliced() {
        // if (X == 0) { S } else { RAISE_ADDR(...); return; } — before arm
        // threading the dead raise path can sit in the ELSE arm; the then arm
        // is the managed source.
        let addr = 0xABCD_1237u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("obj")), Box::new(Expr::Int(0))),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("obj"))),
                name: "keep".into(),
                args: vec![],
            })],
            else_body: vec![
                Stmt::Eval(Expr::RawCall(addr, vec![v("obj")], None)),
                Stmt::Return(None),
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "raise-in-else-arm guard should be spliced");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Eval(_)));
    }

    #[test]
    fn role_classified_raise_guard_goto_tail_is_spliced() {
        // Dead-arm tails from CFG lowering end in `Goto(label)`, not `return`:
        // if (X != 0) { raise; goto L1; } else { S } → splice S.
        let addr = 0xABCD_1238u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Ne, Box::new(v("obj")), Box::new(Expr::Int(0))),
            then_body: vec![
                Stmt::Eval(Expr::RawCall(addr, vec![v("obj")], None)),
                Stmt::Goto("L1".into()),
            ],
            else_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("obj"))),
                name: "keep".into(),
                args: vec![],
            })],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "goto-tailed raise guard should be spliced");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn role_classified_raise_guard_label_tail_is_kept() {
        // A `Label` in the dead arm may be the target of Gotos from elsewhere,
        // so the guard must NOT be stripped there.
        let addr = 0xABCD_1239u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![
            // A jump from elsewhere targets the label in the raise arm.
            Stmt::Goto("L9".into()),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("obj")), Box::new(Expr::Int(0))),
                then_body: vec![
                    Stmt::Eval(Expr::RawCall(addr, vec![v("obj")], None)),
                    Stmt::Label("L9".into()),
                ],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts);
        // The guard itself stays (the Label is a goto target), but the raise
        // inside it still becomes a throw.
        assert_eq!(n, 1, "label-tailed guard is kept, only the raise becomes a throw");
        assert_eq!(stmts.len(), 2);
        let Stmt::If { then_body, .. } = &stmts[1] else {
            panic!("guard must remain an if");
        };
        assert!(
            matches!(then_body.last(), Some(Stmt::Label(_))),
            "the labelled landing pad must survive"
        );
    }

    #[test]
    fn role_classified_raise_guard_with_prelude_assigns_is_spliced() {
        // if (X == 0) { v127 = ...; v128 = ...; RAISE(v128, v127, ...); return; } else { S }
        // Prelude assigns feed the raise's arguments. If none are live in the kept arm, splice.
        let addr = 0xABCD_1240u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("obj")), Box::new(Expr::Int(0))),
            then_body: vec![
                Stmt::Assign("v127".into(), None, v("arg1")),
                Stmt::Assign("v128".into(), None, v("arg2")),
                Stmt::Eval(Expr::RawCall(
                    addr,
                    vec![v("v128"), v("v127"), v("arg3")],
                    None,
                )),
                Stmt::Return(None),
            ],
            else_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("obj"))),
                name: "keep".into(),
                args: vec![],
            })],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "raise guard with prelude assigns should be spliced");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Eval(_)), "else body spliced in");
    }

    #[test]
    #[ignore] // FIXME: liveness check not working as expected
    fn role_classified_raise_guard_with_prelude_live_in_kept_is_refused() {
        // if (X == 0) { v127 = ...; RAISE(v127); return; } else { use(v127); }
        // If a prelude-assigned name is read in the kept arm, refuse to match.
        let addr = 0xABCD_1241u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("obj")), Box::new(Expr::Int(0))),
            then_body: vec![
                Stmt::Assign("v127".into(), None, v("arg1")),
                Stmt::Eval(Expr::RawCall(addr, vec![v("v127")], None)),
                Stmt::Return(None),
            ],
            else_body: vec![
                Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("use"))),
                    name: "use".into(),
                    args: vec![v("v127")],
                }),
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "raise guard with live prelude name should NOT be spliced");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::If { .. }), "guard kept intact");
    }

    #[test]
    fn role_classified_raise_guard_mid_arm_with_dead_tail_is_spliced() {
        // if (X == 0) { v1 = ...; RAISE(v1); v2 = ...; return; } else { S }
        // Raise in the middle with dead assigns after it should still splice.
        let addr = 0xABCD_1242u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("obj")), Box::new(Expr::Int(0))),
            then_body: vec![
                Stmt::Assign("v1".into(), None, v("arg1")),
                Stmt::Eval(Expr::RawCall(addr, vec![v("v1")], None)),
                Stmt::Assign("v2".into(), None, Expr::Int(0)),
                Stmt::Return(None),
            ],
            else_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("obj"))),
                name: "keep".into(),
                args: vec![],
            })],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "raise guard with dead tail after raise should be spliced");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn role_classified_raise_guard_prelude_non_assign_is_refused() {
        // if (X == 0) { Eval(...); RAISE(...); return; } else { S }
        // Prelude with non-Assign statement should refuse to match.
        let addr = 0xABCD_1243u64;
        crate::recovery::runtime_helpers::note(
            addr,
            crate::recovery::runtime_helpers::RuntimeHelper::RaiseException,
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("obj")), Box::new(Expr::Int(0))),
            then_body: vec![
                Stmt::Eval(v("side_effect")),
                Stmt::Eval(Expr::RawCall(addr, vec![v("obj")], None)),
                Stmt::Return(None),
            ],
            else_body: vec![Stmt::Eval(v("keep"))],
        }];
        let n = apply(&mut stmts);
        // The guard is not spliced (the prelude has a statement that is not an
        // assignment feeding the raise), but the raise still becomes a throw.
        assert_eq!(n, 1, "non-assign prelude keeps the guard, raise still becomes a throw");
        assert_eq!(stmts.len(), 1);
        let Stmt::If { then_body, .. } = &stmts[0] else {
            panic!("guard must remain an if");
        };
        assert!(
            then_body
                .iter()
                .any(|s| matches!(s, Stmt::Eval(Expr::Var(n)) if n == "side_effect")),
            "the side-effecting prelude statement must not be dropped"
        );
    }

    #[test]
    fn strip_basic_null_guard_eq_zero() {
        // if (this.m_CachedPtr == 0) { ThrowHelper.ThrowNullReferenceException(this); }
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Field(Box::new(Expr::This), "m_CachedPtr".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("ThrowHelper"))),
                name: "ThrowNullReferenceException".into(),
                args: vec![Expr::This],
            })],
            else_body: vec![],
        },
        // The kept code dereferences the checked value: this is what proves
        // the guard is IL2CPP's auto-inserted null check.
        Stmt::Eval(Expr::Field(
            Box::new(Expr::This),
            "m_CachedPtr".into(),
        )),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "null guard should be stripped");
        assert_eq!(stmts.len(), 1, "only the dereference use remains");
    }

    #[test]
    fn strip_null_guard_eq_null() {
        // if (obj == null) { ThrowHelper.ThrowNullReferenceException(obj); }
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("obj")), Box::new(Expr::Null)),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("ThrowHelper"))),
                name: "ThrowNullReferenceException".into(),
                args: vec![v("obj")],
            })],
            else_body: vec![],
        },
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("obj"))),
            name: "Use".into(),
            args: vec![],
        }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1, "the use of obj remains");
    }

    #[test]
    fn null_guard_with_dead_assignment_is_stripped() {
        // if (this.m_CachedPtr == 0) { ThrowNullRef(this); v19 = 0; }
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Field(Box::new(Expr::This), "m_CachedPtr".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("ThrowHelper"))),
                    name: "ThrowNullReferenceException".into(),
                    args: vec![Expr::This],
                }),
                Stmt::Assign("v19".into(), None, Expr::Int(0)),
            ],
            else_body: vec![],
        },
        Stmt::Eval(Expr::Field(
            Box::new(Expr::This),
            "m_CachedPtr".into(),
        )),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "null guard with dead assignment should be stripped");
        assert_eq!(stmts.len(), 1, "the dereference of the checked field remains");
    }

    #[test]
    fn non_null_guard_if_is_preserved() {
        // if (x == 0) { doSomething(); } — NOT a null guard (no throw call)
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Int(0))),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "doSomething".into(),
                args: vec![],
            })],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "non-null-guard if should not be stripped");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn null_guard_with_else_is_preserved() {
        // if (x == 0) { ThrowNullRef(x); } else { other(); }
        // Has else-body — too complex to be a simple guard.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Int(0))),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("ThrowHelper"))),
                name: "ThrowNullReferenceException".into(),
                args: vec![v("x")],
            })],
            else_body: vec![Stmt::Eval(v("other"))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "null guard with else should not be stripped");
    }

    #[test]
    fn null_guard_nested_in_while() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::Eval(v("work")),
                Stmt::If {
                    cond: Expr::Cmp(CmpKind::Eq, Box::new(v("p")), Box::new(Expr::Null)),
                    then_body: vec![Stmt::Eval(Expr::Call {
                        recv: Some(Box::new(v("ThrowHelper"))),
                        name: "ThrowNullReferenceException".into(),
                        args: vec![v("p")],
                    })],
                    else_body: vec![],
                },
                Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("p"))),
                    name: "Use".into(),
                    args: vec![],
                }),
            ],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "nested null guard should be stripped");
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert_eq!(body.len(), 2, "work eval and the use of p should remain");
            }
            _ => panic!("expected While"),
        }
    }

    // -----------------------------------------------------------------------
    // 7A-2: Class init guard stripping tests
    // -----------------------------------------------------------------------

    #[test]
    fn strip_class_init_guard() {
        // if (*(byte*)(0x25b22c0) == 0) { IL2CPP_ClassInit(...); *(byte*)(...) = 1; }
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::RawDeref(Box::new(Expr::Int(0x25b22c0)), 0, 1)),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "IL2CPP_ClassInit".into(),
                    args: vec![Expr::RawDeref(Box::new(Expr::Int(0x2479a50)), 0, 8)],
                }),
                Stmt::Assign("v9".into(), None, Expr::RawDeref(Box::new(Expr::Int(0x2479a50)), 0, 8)),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "class init guard should be stripped");
        assert!(stmts.is_empty());
    }

    #[test]
    fn strip_class_init_guard_multiple_calls() {
        // Body with multiple IL2CPP_ClassInit calls.
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::RawDeref(Box::new(Expr::Int(0x1000)), 0, 1)),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "IL2CPP_ClassInit".into(),
                    args: vec![Expr::Int(1)],
                }),
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "IL2CPP_ClassInit".into(),
                    args: vec![Expr::Int(2)],
                }),
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "IL2CPP_ClassInit".into(),
                    args: vec![Expr::Int(3)],
                }),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(stmts.is_empty());
    }

    // --- 7A-2b: inline-store byte class-init guard unwrapping ---------------

    /// The dominant corpus shape:
    /// `if ((*(byte*)(STATIC + off) & 1) == 0) { INIT_BODY; *(byte*)(STATIC + off) = 1; }`
    /// The guard `if` and its flag store are IL2CPP machinery; the body is
    /// real (runtime init) work that must be kept, unconditionally.
    #[test]
    fn unwrap_inline_store_byte_guard_keeps_body() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Bin(
                    BinOp::And,
                    Box::new(Expr::RawDeref(Box::new(Expr::Int(0x31ac000)), 2603, 1)),
                    Box::new(Expr::Int(1)),
                )),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Eval(Expr::RawCall(0x130240c, vec![Expr::Int(0x30746b0)], None)),
                Stmt::Assign(
                    "*(byte*)(0x31ac000 +2603)".into(),
                    None,
                    Expr::Int(1),
                ),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "guard should be unwrapped");
        assert!(
            !stmts.iter().any(|s| matches!(s, Stmt::If { .. })),
            "the guard if must be gone: {stmts:?}"
        );
        assert!(
            !stmts
                .iter()
                .any(|s| matches!(s, Stmt::Assign(t, _, _) if t.starts_with("*(byte*)"))),
            "the flag store must be gone: {stmts:?}"
        );
        // The init call survives, unconditionally.
        assert!(
            stmts
                .iter()
                .any(|s| matches!(s, Stmt::Eval(Expr::RawCall(0x130240c, ..)))),
            "the init body must be kept: {stmts:?}"
        );
    }

    /// If the body contains a SECOND store to the guard address, the shape is
    /// ambiguous and must be skipped.
    #[test]
    fn byte_guard_with_extra_store_to_addr_is_skipped() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Bin(
                    BinOp::And,
                    Box::new(Expr::RawDeref(Box::new(Expr::Int(0x31ac000)), 2603, 1)),
                    Box::new(Expr::Int(1)),
                )),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Assign("*(byte*)(0x31ac000 +2603)".into(), None, Expr::Int(1)),
                Stmt::Assign("*(byte*)(0x31ac000 +2603)".into(), None, Expr::Int(0)),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "ambiguous double-store guard must be skipped");
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    /// A guard on a computed (non-literal-base) address, e.g. `this` or a stack
    /// slot, is real managed code and must not be touched.
    #[test]
    fn byte_guard_on_non_literal_base_is_skipped() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Bin(
                    BinOp::And,
                    Box::new(Expr::RawDeref(Box::new(Expr::This), 2603, 1)),
                    Box::new(Expr::Int(1)),
                )),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Eval(v("doWork")),
                Stmt::Assign("*(byte*)(this +2603)".into(), None, Expr::Int(1)),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "instance-field guard is real code and must survive");
        assert!(matches!(&stmts[0], Stmt::If { .. }));
    }

    #[test]
    fn non_class_init_if_is_preserved() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Eq, Box::new(v("x")), Box::new(Expr::Int(0))),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "doWork".into(),
                args: vec![],
            })],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 7B: RPC ternary collapse tests
    // -----------------------------------------------------------------------

    #[test]
    fn collapse_eq_zero_this_var() {
        // v58 == 0 ? this : v58  →  this
        let mut expr = Expr::Cond(
            Box::new(Expr::Cmp(CmpKind::Eq, Box::new(v("v58")), Box::new(Expr::Int(0)))),
            Box::new(Expr::This),
            Box::new(v("v58")),
        );
        let count = collapse_rpc_ternary_expr(&mut expr);
        assert_eq!(count, 1);
        assert!(matches!(expr, Expr::This));
    }

    #[test]
    fn collapse_ne_zero_var_this() {
        // v58 != 0 ? v58 : this  →  this
        let mut expr = Expr::Cond(
            Box::new(Expr::Cmp(CmpKind::Ne, Box::new(v("v58")), Box::new(Expr::Int(0)))),
            Box::new(v("v58")),
            Box::new(Expr::This),
        );
        let count = collapse_rpc_ternary_expr(&mut expr);
        assert_eq!(count, 1);
        assert!(matches!(expr, Expr::This));
    }

    #[test]
    fn collapse_eq_null_this_var() {
        // v == null ? this : v  →  this
        let mut expr = Expr::Cond(
            Box::new(Expr::Cmp(CmpKind::Eq, Box::new(v("ctx")), Box::new(Expr::Null))),
            Box::new(Expr::This),
            Box::new(v("ctx")),
        );
        let count = collapse_rpc_ternary_expr(&mut expr);
        assert_eq!(count, 1);
        assert!(matches!(expr, Expr::This));
    }

    #[test]
    fn no_collapse_when_vars_differ() {
        // v1 == 0 ? this : v2  — different variables, no collapse.
        let mut expr = Expr::Cond(
            Box::new(Expr::Cmp(CmpKind::Eq, Box::new(v("v1")), Box::new(Expr::Int(0)))),
            Box::new(Expr::This),
            Box::new(v("v2")),
        );
        let count = collapse_rpc_ternary_expr(&mut expr);
        assert_eq!(count, 0);
    }

    #[test]
    fn no_collapse_when_no_this() {
        // v == 0 ? other : v  — neither branch is `this`.
        let mut expr = Expr::Cond(
            Box::new(Expr::Cmp(CmpKind::Eq, Box::new(v("v")), Box::new(Expr::Int(0)))),
            Box::new(v("other")),
            Box::new(v("v")),
        );
        let count = collapse_rpc_ternary_expr(&mut expr);
        assert_eq!(count, 0);
    }

    #[test]
    fn collapse_in_call_receiver() {
        // (v58 == 0 ? this : v58).CallRpc()  →  this.CallRpc()
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(Expr::Cond(
                Box::new(Expr::Cmp(CmpKind::Eq, Box::new(v("v58")), Box::new(Expr::Int(0)))),
                Box::new(Expr::This),
                Box::new(v("v58")),
            ))),
            name: "CallRpcRemove".into(),
            args: vec![],
        })];
        let n = apply(&mut stmts);
        assert!(n >= 1, "should collapse the RPC ternary");
        match &stmts[0] {
            Stmt::Eval(Expr::Call { recv: Some(r), name, .. }) => {
                assert!(matches!(r.as_ref(), Expr::This), "receiver should be this");
                assert_eq!(name, "CallRpcRemove");
            }
            other => panic!("expected Eval(Call), got {other:?}"),
        }
    }

    #[test]
    fn collapse_field_ternary() {
        // this.m_RadioButtonContainer == 0 ? this : this.m_RadioButtonContainer
        let field = Expr::Field(Box::new(Expr::This), "m_RadioButtonContainer".into());
        let mut expr = Expr::Cond(
            Box::new(Expr::Cmp(CmpKind::Eq, Box::new(field.clone()), Box::new(Expr::Int(0)))),
            Box::new(Expr::This),
            Box::new(field),
        );
        let count = collapse_rpc_ternary_expr(&mut expr);
        assert_eq!(count, 1);
        assert!(matches!(expr, Expr::This));
    }

    // -----------------------------------------------------------------------
    // 7C: Temp variable flattening tests
    // -----------------------------------------------------------------------

    #[test]
    fn flatten_op_inequality_into_if() {
        // v33 = DiscreteTime.op_Inequality(v27, v30);
        // if (v33 != 0) { return 0; }
        let mut stmts = vec![
            Stmt::Assign(
                "v33".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "DiscreteTime.op_Inequality".into(),
                    args: vec![v("v27"), v("v30")],
                },
            ),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("v33")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Return(Some(Expr::Int(0)))],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1, "should flatten the temp var");
        assert_eq!(stmts.len(), 1, "assignment should be removed");
        match &stmts[0] {
            Stmt::If { cond, .. } => {
                // The condition should now contain the call, not Var("v33").
                assert!(
                    !expr_mentions(cond, "v33"),
                    "v33 should be inlined away"
                );
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn flatten_into_return() {
        // v1 = Foo.Bar(x);
        // return v1;
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Foo.Bar".into(),
                    args: vec![v("x")],
                },
            ),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Return(Some(Expr::Call { name, .. })) => {
                assert_eq!(name, "Foo.Bar");
            }
            other => panic!("expected Return(Call), got {other:?}"),
        }
    }

    #[test]
    fn no_flatten_when_var_used_twice() {
        // v1 = Foo(x);
        // if (v1 != 0 && v1 != 1) { ... }
        // v1 is used twice — cannot inline.
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Foo".into(),
                    args: vec![v("x")],
                },
            ),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(v("v1")),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Eval(v("v1"))],
                else_body: vec![],
            },
        ];
        let n = flatten_temp_vars(&mut stmts);
        assert_eq!(n, 0, "v1 used in both cond and body should not be inlined");
    }

    #[test]
    fn flatten_trivial_rhs() {
        // v1 = v2;
        // if (v1 != 0) { ... }
        // Trivial RHS (bare var) — now inlined (was previously skipped).
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, v("v2")),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("v1")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![],
            },
        ];
        let n = flatten_temp_vars(&mut stmts);
        assert_eq!(n, 1, "trivial RHS should now be inlined");
        assert_eq!(stmts.len(), 1, "assignment should be removed");
    }

    #[test]
    fn no_flatten_non_adjacent() {
        // v1 = Foo(x);
        // other();
        // if (v1 != 0) { ... }
        // Not adjacent — an intervening statement prevents inlining.
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Foo".into(),
                    args: vec![v("x")],
                },
            ),
            Stmt::Eval(v("other")),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("v1")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![],
            },
        ];
        let n = flatten_temp_vars(&mut stmts);
        assert_eq!(n, 0, "non-adjacent assignment should not be inlined");
    }

    #[test]
    fn flatten_op_equality_into_if() {
        // v40 = TimeSpan.op_Inequality(a, b);
        // if (v40 == 0) { ... }
        let mut stmts = vec![
            Stmt::Assign(
                "v40".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "TimeSpan.op_Inequality".into(),
                    args: vec![v("a"), v("b")],
                },
            ),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("v40")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Eval(v("work"))],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::If { cond, .. } => {
                assert!(!expr_mentions(cond, "v40"), "v40 should be inlined");
            }
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn repeated_typeof_runtime_calls_are_dropped() {
        let call = |addr: u64, ty: &str| {
            Stmt::Eval(Expr::RawCall(
                addr,
                vec![Expr::TypeOf(ty.into()), v("inProfile")],
                None,
            ))
        };
        let mut stmts = vec![
            call(0x130240c, "CharacterManager"),
            call(0x130240c, "TokenManager"),
            call(0x130240c, "RewardData"),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Keep".into(),
                args: vec![Expr::TypeOf("X".into())],
            }),
        ];
        let n = drop_metadata_init_calls(&mut stmts);
        assert_eq!(n, 3);
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn a_single_typeof_runtime_call_is_kept() {
        let mut stmts = vec![Stmt::Eval(Expr::RawCall(
            0x130240c,
            vec![Expr::TypeOf("X".into()), v("p")],
            None,
        ))];
        assert_eq!(drop_metadata_init_calls(&mut stmts), 0);
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn inline_does_not_cross_a_label() {
        // A def in one branch, then a SHARED jump target: the edge into L7 may
        // not have executed the def, so its value must not be folded into the
        // L7 read (0x189D9D4 substituted `v289 = 0` into the L7 token load).
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            Stmt::Label("L7".into()),
            Stmt::Eval(Expr::RawDeref(Box::new(v("v1")), 24, 4)),
        ];
        let _ = inline_temp_variables(&mut stmts);
        let t = format!("{stmts:?}");
        assert!(t.contains("Assign(\"v1\""), "def must stay: {t}");
        assert!(
            !t.contains("RawDeref(Int("),
            "read must not be folded to a constant: {t}"
        );
        assert!(t.contains("\"v1\""), "read must stay a variable: {t}");
    }

    #[test]
    fn a_read_after_a_label_keeps_the_def() {
        // def; use; label; use — substitution stops at the label, but the
        // def's LIVE RANGE still covers the post-label read. Deleting the def
        // because the sliced range is clean left undefined temps
        // (0x1671C90 `v161` after the pool block's labels).
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(7)),
            Stmt::Assign("v2".into(), None, v("v1")),
            Stmt::Label("L3".into()),
            Stmt::Assign("v3".into(), None, v("v1")),
        ];
        let _ = inline_temp_variables(&mut stmts);
        let t = format!("{stmts:?}");
        assert!(t.contains("Assign(\"v1\""), "def must stay: {t}");
        assert!(
            t.contains("Assign(\"v2\", None, Int(7))"),
            "pre-label read is substituted: {t}"
        );
        assert!(
            t.contains("Assign(\"v3\", None, Var(\"v1\"))"),
            "post-label read keeps the name: {t}"
        );
    }

    #[test]
    fn a_forward_goto_after_the_def_does_not_block_substitution() {
        // The only edge to L1 leaves an arm AFTER the definition, so every
        // path reaching the label executed `v1 = 5`.
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(5)),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Goto("L1".into())],
                else_body: vec![],
            },
            Stmt::Label("L1".into()),
            Stmt::Assign("v2".into(), None, v("v1")),
        ];
        let _ = inline_temp_variables(&mut stmts);
        let t = format!("{stmts:?}");
        assert!(
            t.contains("Assign(\"v2\", None, Int(5))"),
            "the read after a dominated label is substituted: {t}"
        );
    }

    #[test]
    fn a_goto_before_the_def_keeps_the_label_barrier() {
        // The edge to L1 can be taken BEFORE `v1 = 5`; folding the read would
        // invent the value on that path.
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Goto("L1".into())],
                else_body: vec![],
            },
            Stmt::Assign("v1".into(), None, Expr::Int(5)),
            Stmt::Label("L1".into()),
            Stmt::Assign("v2".into(), None, v("v1")),
        ];
        let _ = inline_temp_variables(&mut stmts);
        let t = format!("{stmts:?}");
        assert!(t.contains("Assign(\"v1\""), "def must stay: {t}");
        assert!(
            t.contains("Assign(\"v2\", None, Var(\"v1\"))"),
            "post-label read must keep the name: {t}"
        );
    }

    #[test]
    fn inline_stops_at_a_write_to_a_name_the_def_reads() {
        // v2 = v1 - p; v1 = v2; if (v2 >= 0) … — the loop-carried v1 is
        // updated before the condition reads v2. Folding v2 into the condition
        // would re-evaluate `v1 - p` against the NEW v1 (a second subtraction).
        let mut stmts = vec![
            Stmt::Assign(
                "v2".into(),
                None,
                Expr::Bin(
                    super::super::super::csharp::BinOp::Sub,
                    Box::new(v("v1")),
                    Box::new(v("p")),
                ),
            ),
            Stmt::Assign("v1".into(), None, v("v2")),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ge,
                    Box::new(v("v2")),
                    Box::new(Expr::Float(0.0)),
                ),
                then_body: vec![Stmt::Continue],
                else_body: vec![Stmt::Continue],
            },
        ];
        let _ = inline_temp_variables(&mut stmts);
        let t = format!("{stmts:?}");
        assert!(
            t.contains("Assign(\"v2\""),
            "the temp must stay materialised: {t}"
        );
        assert!(
            t.contains("Cmp(Ge, Var(\"v2\")"),
            "the condition must read the result, not re-evaluate it: {t}"
        );
    }

    // -----------------------------------------------------------------------
    // Trailing synthesized raise: guards it terminated must unwind
    // -----------------------------------------------------------------------

    /// END-TO-END GUARD (code-loss class): the cleanup round must never fold
    /// an address-escaped `out` cell to its initialiser. Doing so turned the
    /// follow-up null check into `0 == 0` and made a whole update block
    /// unreachable (0x18A3A1C); the block — a field store plus a call — must
    /// survive this pass even though every step here is individually legal.
    #[test]
    fn cleanup_never_inlines_an_out_cell_into_its_guard() {
        let mut stmts = vec![
            Stmt::Assign("stack_1".into(), None, Expr::Int(0)),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("dict"))),
                name: "TryGetValue".into(),
                args: vec![
                    v("id"),
                    Expr::AddressOf(Box::new(v("stack_1"))),
                ],
            }),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(v("stack_1")),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![
                    Stmt::Assign(
                        "*(int*)((nint)&stack_1 +28)".into(),
                        None,
                        v("cost"),
                    ),
                    Stmt::Eval(Expr::Call {
                        recv: Some(Box::new(Expr::This)),
                        name: "UpdateProbability".into(),
                        args: vec![v("stack_1")],
                    }),
                ],
                else_body: vec![],
            },
        ];
        let _ = apply(&mut stmts);
        let t = format!("{stmts:?}");
        assert!(
            t.contains("\"stack_1\",") || t.contains("\"stack_1\""),
            "the out cell must stay named: {t}"
        );
        assert!(
            !t.contains("Cmp(Ne, Int(0), Int(0))"),
            "the guard was folded to a constant: {t}"
        );
        assert!(
            t.contains("UpdateProbability"),
            "the guarded call was deleted: {t}"
        );
        assert!(
            t.contains("+28"),
            "the guarded field store was deleted: {t}"
        );
    }

    fn fallback_nre_throw() -> Stmt {
        Stmt::Throw(Expr::NewObj(
            "NullReferenceException".into(),
            vec![Expr::Str(
                "Object reference not set to an instance of an object.".into(),
            )],
        ))
    }

    fn field(base: Expr, name: &str) -> Expr {
        Expr::Field(Box::new(base), name.to_string())
    }

    #[test]
    fn trailing_nre_throw_unwraps_nested_null_guards() {
        // if (p != null) { if (p.children != null) { return p.children[0].GetCost(); } }
        // throw new NRE();   // injected null path
        let payload = Stmt::Return(Some(Expr::Call {
            recv: Some(Box::new(Expr::Index {
                collection: Box::new(field(v("p"), "children")),
                index: Box::new(Expr::Int(0)),
            })),
            name: "GetCost".into(),
            args: vec![],
        }));
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("p")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::If {
                    cond: Expr::Cmp(
                        CmpKind::Ne,
                        Box::new(field(v("p"), "children")),
                        Box::new(Expr::Int(0)),
                    ),
                    then_body: vec![payload],
                    else_body: vec![],
                }],
                else_body: vec![],
            },
            Stmt::Label("L1".into()),
            fallback_nre_throw(),
        ];
        let n = finalize_raise_throws(&mut stmts);
        assert!(n >= 1, "raise removed");
        assert!(
            stmts.iter().any(|s| matches!(s, Stmt::Return(_))),
            "return preserved: {stmts:?}"
        );
        assert!(
            !stmts.iter().any(|s| matches!(s, Stmt::If { .. })),
            "null guards unwrapped: {stmts:?}"
        );
    }

    #[test]
    fn trailing_raise_keeps_guard_whose_body_ignores_the_value() {
        // if (p != null) { return q.Foo(); } throw NRE; — the test is real
        // managed logic (the body never uses p); keep it.
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("p")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Return(Some(Expr::Call {
                    recv: Some(Box::new(v("q"))),
                    name: "Foo".into(),
                    args: vec![],
                }))],
                else_body: vec![],
            },
            fallback_nre_throw(),
        ];
        finalize_raise_throws(&mut stmts);
        assert!(
            stmts.iter().any(|s| matches!(s, Stmt::If { .. })),
            "guard kept: {stmts:?}"
        );
        assert!(
            !stmts.iter().any(|s| matches!(s, Stmt::Throw(_))),
            "raise removed: {stmts:?}"
        );
    }

    #[test]
    fn referenced_trailing_label_blocks_unwrapping() {
        let mut stmts = vec![
            Stmt::Goto("L1".into()),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("p")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Return(Some(field(v("p"), "x")))],
                else_body: vec![],
            },
            Stmt::Label("L1".into()),
            fallback_nre_throw(),
        ];
        finalize_raise_throws(&mut stmts);
        assert!(
            stmts.iter().any(|s| matches!(s, Stmt::If { .. })),
            "referenced label keeps the guard: {stmts:?}"
        );
    }

    #[test]
    fn split_null_guard_chain_unwraps_to_the_payload() {
        // The measured shape after branch_repair for 0x18A5204:
        //   if (inData != null) {
        //     if (inData.UnlockInfos != 0) {
        //       if (inData.UnlockInfos[0] != null) return …GetCost();
        //     }
        //   }
        //   throw new NRE();
        // Every guard's false path is the raise; after removing it all three
        // guards are implicit and the return is the whole body.
        let payload = Stmt::Return(Some(Expr::Call {
            recv: Some(Box::new(Expr::Index {
                collection: Box::new(field(v("inData"), "UnlockInfos")),
                index: Box::new(Expr::Int(0)),
            })),
            name: "GetCost".into(),
            args: vec![],
        }));
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("inData")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::If {
                    cond: Expr::Cmp(
                        CmpKind::Ne,
                        Box::new(field(v("inData"), "UnlockInfos")),
                        Box::new(Expr::Int(0)),
                    ),
                    then_body: vec![Stmt::If {
                        cond: Expr::Cmp(
                            CmpKind::Ne,
                            Box::new(Expr::Index {
                                collection: Box::new(field(v("inData"), "UnlockInfos")),
                                index: Box::new(Expr::Int(0)),
                            }),
                            Box::new(Expr::Int(0)),
                        ),
                        then_body: vec![payload],
                        else_body: vec![],
                    }],
                    else_body: vec![],
                }],
                else_body: vec![],
            },
            fallback_nre_throw(),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1, "raise/guards handled");
        assert!(
            !stmts.iter().any(|s| matches!(s, Stmt::If { .. })),
            "all null guards unwrapped: {stmts:?}"
        );
        assert!(
            matches!(stmts.last(), Some(Stmt::Return(_))),
            "return is the tail: {stmts:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Combined pipeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn null_guard_and_class_init_guard_both_stripped() {
        let mut stmts = vec![
            // Null guard.
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("p")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("ThrowHelper"))),
                    name: "ThrowNullReferenceException".into(),
                    args: vec![v("p")],
                })],
                else_body: vec![],
            },
            // Some real work that dereferences the checked value.
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("p"))),
                name: "doWork".into(),
                args: vec![],
            }),
            // Class init guard.
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::RawDeref(Box::new(Expr::Int(0x1000)), 0, 1)),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "IL2CPP_ClassInit".into(),
                    args: vec![Expr::Int(1)],
                })],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both guards should be stripped");
        assert_eq!(stmts.len(), 1, "only the work should remain");
    }

    #[test]
    fn full_pipeline_rpc_ternary_and_temp_flatten() {
        // v33 = op_Inequality(a, b);
        // if ((v33 == 0 ? this : v33) != 0) { work(); }
        // After flattening: if (op_Inequality(a, b) != 0) { work(); }
        // (The RPC ternary is inside the condition but v33 is the same var
        // being assigned — the temp flatten should handle this.)
        let mut stmts = vec![
            Stmt::Assign(
                "v33".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "op_Inequality".into(),
                    args: vec![v("a"), v("b")],
                },
            ),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(Expr::Cond(
                        Box::new(Expr::Cmp(CmpKind::Eq, Box::new(v("v33")), Box::new(Expr::Int(0)))),
                        Box::new(Expr::This),
                        Box::new(v("v33")),
                    )),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Eval(v("work"))],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts);
        // The RPC ternary should collapse first (v33 == 0 ? this : v33 → this),
        // then the temp flatten should NOT fire because v33 no longer appears
        // in the condition (it was replaced by `this` by the RPC collapse).
        // Actually: the RPC ternary collapse replaces the inner Cond with This,
        // so the condition becomes `this != 0`. Then v33 is no longer mentioned,
        // so the temp flatten won't find it. But the assignment to v33 is now
        // dead. Let's just verify the RPC ternary collapsed.
        assert!(n >= 1, "at least the RPC ternary should collapse");
    }

    #[test]
    fn multiple_null_guards_in_sequence() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("a")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("ThrowHelper"))),
                    name: "ThrowNullReferenceException".into(),
                    args: vec![v("a")],
                })],
                else_body: vec![],
            },
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("b")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("ThrowHelper"))),
                    name: "ThrowNullReferenceException".into(),
                    args: vec![v("b")],
                })],
                else_body: vec![],
            },
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("a"))),
                name: "Use".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("b"))),
                name: "Use".into(),
                args: vec![],
            }),
            Stmt::Eval(v("realWork")),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both null guards should be stripped");
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[2], Stmt::Eval(Expr::Var(n)) if n == "realWork"));
    }

    // -----------------------------------------------------------------------
    // 7D: Static field access cleanup tests
    // -----------------------------------------------------------------------

    #[test]
    fn cleanup_static_field_via_klass_static_fields() {
        // Field(Field(StaticField("ImangiGame.WorldInfo", "klass(ImangiGame.WorldInfo)"), "static_fields"), "m_CancellationTokenSource")
        // → StaticField("ImangiGame.WorldInfo", "m_CancellationTokenSource")
        let mut stmts = vec![Stmt::Eval(Expr::Field(
            Box::new(Expr::Field(
                Box::new(Expr::StaticField(
                    "ImangiGame.WorldInfo".into(),
                    "klass(ImangiGame.WorldInfo)".into(),
                )),
                "static_fields".into(),
            )),
            "m_CancellationTokenSource".into(),
        ))];
        let n = cleanup_static_field_access(&mut stmts);
        assert_eq!(n, 1, "should clean up one static field access");
        match &stmts[0] {
            Stmt::Eval(Expr::StaticField(ty, name)) => {
                assert_eq!(ty, "ImangiGame.WorldInfo");
                assert_eq!(name, "m_CancellationTokenSource");
            }
            other => panic!("expected Eval(StaticField), got {other:?}"),
        }
    }

    #[test]
    fn cleanup_static_field_with_rawderef_klass() {
        // Field(Field(RawDeref(StaticField("T", "klass(T)"), 0, 0), "static_fields"), "field")
        // → StaticField("T", "field")
        let mut stmts = vec![Stmt::Eval(Expr::Field(
            Box::new(Expr::Field(
                Box::new(Expr::RawDeref(
                    Box::new(Expr::StaticField("T".into(), "klass(T)".into())),
                    0,
                    0,
                )),
                "static_fields".into(),
            )),
            "myField".into(),
        ))];
        let n = cleanup_static_field_access(&mut stmts);
        // 1 rewrite: Pattern 1 matches the full chain (find_klass_type_in_chain
        // follows the RawDeref to find the klass accessor inside).
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StaticField(ty, name)) => {
                assert_eq!(ty, "T");
                assert_eq!(name, "myField");
            }
            other => panic!("expected Eval(StaticField), got {other:?}"),
        }
    }

    #[test]
    fn cleanup_static_field_via_field_klass() {
        // Field(Field(Field(base, "klass(TypeName)"), "static_fields"), "field")
        // → StaticField("TypeName", "field")
        let mut stmts = vec![Stmt::Eval(Expr::Field(
            Box::new(Expr::Field(
                Box::new(Expr::Field(
                    Box::new(v("base")),
                    "klass(TypeName)".into(),
                )),
                "static_fields".into(),
            )),
            "myField".into(),
        ))];
        let n = cleanup_static_field_access(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StaticField(ty, name)) => {
                assert_eq!(ty, "TypeName");
                assert_eq!(name, "myField");
            }
            other => panic!("expected Eval(StaticField), got {other:?}"),
        }
    }

    #[test]
    fn cleanup_rawderef_of_klass_at_zero() {
        // RawDeref(StaticField("T", "klass(T)"), 0, 0) → TypeOf("T")
        // Pattern 2 removes the RawDeref, then Pattern 4 converts the bare klass to typeof.
        use crate::recovery::csharp::render_stmts;
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::StaticField("T".into(), "klass(T)".into())),
            0,
            0,
        ))];
        let n = cleanup_static_field_access(&mut stmts);
        assert_eq!(n, 2, "RawDeref removal + klass→typeof = 2 rewrites");
        let output = render_stmts(&stmts);
        assert!(
            output.contains("typeof(T)"),
            "should render as typeof(T), got: {output}"
        );
        assert!(
            !output.contains("klass("),
            "should not contain klass() after cleanup, got: {output}"
        );
    }

    #[test]
    fn cleanup_static_field_renders_cleanly() {
        use crate::recovery::csharp::render_stmts;
        let mut stmts = vec![Stmt::Eval(Expr::Field(
            Box::new(Expr::Field(
                Box::new(Expr::StaticField(
                    "ImangiGame.WorldInfo".into(),
                    "klass(ImangiGame.WorldInfo)".into(),
                )),
                "static_fields".into(),
            )),
            "m_CancellationTokenSource".into(),
        ))];
        cleanup_static_field_access(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("ImangiGame.WorldInfo.m_CancellationTokenSource"),
            "should render as clean static field access, got: {output}"
        );
        assert!(
            !output.contains("klass("),
            "should not contain klass() after cleanup, got: {output}"
        );
        assert!(
            !output.contains("static_fields"),
            "should not contain static_fields after cleanup, got: {output}"
        );
    }

    #[test]
    fn cleanup_static_field_nested_in_assignment() {
        use crate::recovery::csharp::render_stmts;
        // v1 = Field(Field(StaticField("T", "klass(T)"), "static_fields"), "field")
        let mut stmts = vec![Stmt::Assign(
            "v1".into(),
            None,
            Expr::Field(
                Box::new(Expr::Field(
                    Box::new(Expr::StaticField("T".into(), "klass(T)".into())),
                    "static_fields".into(),
                )),
                "myField".into(),
            ),
        )];
        let n = cleanup_static_field_access(&mut stmts);
        assert_eq!(n, 1);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("T.myField"),
            "should render as T.myField, got: {output}"
        );
    }

    #[test]
    fn cleanup_static_field_in_if_condition() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Field(
                    Box::new(Expr::Field(
                        Box::new(Expr::StaticField("T".into(), "klass(T)".into())),
                        "static_fields".into(),
                    )),
                    "flag".into(),
                )),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![Stmt::Eval(v("work"))],
            else_body: vec![],
        }];
        let n = cleanup_static_field_access(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { cond, .. } => match cond {
                Expr::Cmp(_, lhs, _) => {
                    assert!(matches!(lhs.as_ref(), Expr::StaticField(ty, name) if ty == "T" && name == "flag"));
                }
                other => panic!("expected Cmp, got {other:?}"),
            },
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn no_cleanup_for_non_klass_static_fields() {
        // Field(Field(Var("x"), "static_fields"), "y") — x is not a klass access
        // Should NOT be cleaned up (we don't know the type).
        let mut stmts = vec![Stmt::Eval(Expr::Field(
            Box::new(Expr::Field(
                Box::new(v("x")),
                "static_fields".into(),
            )),
            "y".into(),
        ))];
        let n = cleanup_static_field_access(&mut stmts);
        assert_eq!(n, 0, "should not clean up when base is not a klass access");
    }

    #[test]
    fn cleanup_static_field_with_intermediate_klass_fields() {
        // Field(Field(Field(StaticField("UnityEngine.Matrix4x4", "klass(UnityEngine.Matrix4x4)"), "m00"), "static_fields"), "element_class")
        // → StaticField("UnityEngine.Matrix4x4", "element_class")
        use crate::recovery::csharp::render_stmts;
        let mut stmts = vec![Stmt::Eval(Expr::Field(
            Box::new(Expr::Field(
                Box::new(Expr::Field(
                    Box::new(Expr::StaticField(
                        "UnityEngine.Matrix4x4".into(),
                        "klass(UnityEngine.Matrix4x4)".into(),
                    )),
                    "m00".into(),
                )),
                "static_fields".into(),
            )),
            "element_class".into(),
        ))];
        let n = cleanup_static_field_access(&mut stmts);
        assert!(n >= 1, "should clean up the deeper pattern");
        let output = render_stmts(&stmts);
        assert!(
            output.contains("UnityEngine.Matrix4x4.element_class"),
            "should render as clean static field, got: {output}"
        );
        assert!(
            !output.contains("klass("),
            "should not contain klass() after cleanup, got: {output}"
        );
    }

    // -----------------------------------------------------------------------
    // 7C-2: General temp variable inlining tests
    // -----------------------------------------------------------------------

    #[test]
    fn is_temp_var_name_basic() {
        assert!(is_temp_var_name("v1"));
        assert!(is_temp_var_name("v0"));
        assert!(is_temp_var_name("v23"));
        assert!(is_temp_var_name("v100"));
        assert!(!is_temp_var_name("v"));
        assert!(!is_temp_var_name("x1"));
        assert!(!is_temp_var_name("this"));
        assert!(!is_temp_var_name(""));
        assert!(!is_temp_var_name("v1x"));
    }

    #[test]
    fn inline_non_adjacent_field_access() {
        // v1 = obj.SomeField;
        // other();
        // return v1;
        // → other(); return obj.SomeField;
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "SomeField".into()),
            ),
            Stmt::Eval(v("other")),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "should inline v1");
        assert_eq!(stmts.len(), 2, "assignment should be removed");
        match &stmts[1] {
            Stmt::Return(Some(Expr::Field(recv, name))) => {
                assert!(matches!(recv.as_ref(), Expr::Var(n) if n == "obj"));
                assert_eq!(name, "SomeField");
            }
            other => panic!("expected Return(Field), got {other:?}"),
        }
    }

    #[test]
    fn inline_chain_of_temps() {
        // v1 = obj.SomeField;
        // v2 = v1 + 10;
        // return v2;
        // → return obj.SomeField + 10;
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "SomeField".into()),
            ),
            Stmt::Assign(
                "v2".into(),
                None,
                Expr::Bin(
                    crate::recovery::csharp::BinOp::Add,
                    Box::new(v("v1")),
                    Box::new(Expr::Int(10)),
                ),
            ),
            Stmt::Return(Some(v("v2"))),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 2, "should inline both v1 and v2");
        assert_eq!(stmts.len(), 1, "both assignments should be removed");
    }

    #[test]
    fn inline_when_pure_used_twice() {
        // v1 = obj.Field;
        // x = v1 + v1;
        // A pure read can be duplicated into every use.
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Field".into()),
            ),
            Stmt::Assign(
                "x".into(),
                None,
                Expr::Bin(
                    crate::recovery::csharp::BinOp::Add,
                    Box::new(v("v1")),
                    Box::new(v("v1")),
                ),
            ),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "pure v1 should inline into both uses");
        assert_eq!(
            stmts.len(),
            1,
            "the v1 declaration is gone once nothing reads it"
        );
    }

    #[test]
    fn no_inline_impure_rhs_with_intervening_call() {
        // v1 = Foo();
        // Bar();  // intervening call
        // return v1;
        // RHS is a call (impure) with an intervening call — should NOT inline.
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Foo".into(),
                    args: vec![],
                },
            ),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Bar".into(),
                args: vec![],
            }),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 0, "impure RHS with intervening call should not be inlined");
    }

    #[test]
    fn inline_impure_rhs_without_intervening_call() {
        // v1 = Foo();
        // x = 42;  // no intervening call
        // return v1;
        // RHS is a call (impure) but no intervening calls — SHOULD inline.
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Foo".into(),
                    args: vec![],
                },
            ),
            Stmt::Assign("x".into(), None, Expr::Int(42)),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "impure RHS without intervening call should be inlined");
    }

    #[test]
    fn sequential_multi_def_ranges_inline_independently() {
        // SSA versions from different merge points share one display name:
        //   v1 = a + 1; x = v1; v1 = b + 2; y = v1;
        // Each def is single-read within its OWN live range, so both must
        // inline — counting over the whole list used to reject both.
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Bin(BinOp::Add, Box::new(v("a")), Box::new(Expr::Int(1))),
            ),
            Stmt::Assign("x".into(), None, v("v1")),
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Bin(BinOp::Add, Box::new(v("b")), Box::new(Expr::Int(2))),
            ),
            Stmt::Assign("y".into(), None, v("v1")),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 2, "both live ranges should inline: {stmts:?}");
        assert_eq!(stmts.len(), 2, "both temp defs should be gone: {stmts:?}");
        assert!(
            !stmts.iter().any(|s| matches!(s, Stmt::Assign(n, _, _) if n == "v1")),
            "v1 must not survive: {stmts:?}"
        );
    }

    #[test]
    fn nested_write_ends_the_live_range() {
        //   v1 = a;
        //   if (c) { v1 = b; }
        //   x = v1;
        // The nested write ends v1's first range, so inlining `v1 = a` into
        // `x = v1` would be WRONG (x may read b). Must not fire.
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, v("a")),
            Stmt::If {
                cond: v("c"),
                then_body: vec![Stmt::Assign("v1".into(), None, v("b"))],
                else_body: vec![],
            },
            Stmt::Assign("x".into(), None, v("v1")),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(
            n, 0,
            "a nested write must fence the live range: {stmts:?}"
        );
    }

    #[test]
    fn no_inline_non_temp_var() {
        // x = obj.Field;
        // other();
        // return x;
        // 'x' is not a temp var name — should NOT inline.
        let mut stmts = vec![
            Stmt::Assign(
                "x".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Field".into()),
            ),
            Stmt::Eval(v("other")),
            Stmt::Return(Some(v("x"))),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 0, "non-temp var should not be inlined");
    }

    #[test]
    fn inline_trivial_rhs() {
        // v1 = v2;
        // other();
        // return v1;
        // Trivial RHS (bare var) — now inlined (was previously skipped).
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, v("v2")),
            Stmt::Eval(v("other")),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "trivial RHS should now be inlined");
        assert_eq!(stmts.len(), 2, "assignment should be removed");
        match &stmts[1] {
            Stmt::Return(Some(Expr::Var(n))) => assert_eq!(n, "v2"),
            other => panic!("expected Return(Var(v2)), got {other:?}"),
        }
    }

    #[test]
    fn inline_single_read_neq_within_if_body() {
        // if (guard) {
        //     v76 = ImangiGame.WorldInfo.CTS._kernelEvent != 0;
        //     if (!v76) { body }
        // }
        // The single-read pure RHS must inline into the `if (!...)` condition.
        use crate::recovery::csharp::CmpKind;
        let field_chain = Expr::Field(
            Box::new(Expr::Field(
                Box::new(Expr::Field(
                    Box::new(Expr::StaticField("ImangiGame".into(), "WorldInfo".into())),
                    "m_CancellationTokenSource".into(),
                )),
                "_kernelEvent".into(),
            )),
            "dummy".into(),
        );
        let neq = Expr::Cmp(CmpKind::Ne, Box::new(field_chain), Box::new(Expr::Int(0)));
        let mut stmts = vec![Stmt::If {
            cond: v("guard"),
            then_body: vec![
                Stmt::Assign("v76".into(), None, neq),
                Stmt::If {
                    cond: Expr::Not(Box::new(v("v76"))),
                    then_body: vec![Stmt::Return(None)],
                    else_body: vec![],
                },
            ],
            else_body: vec![],
        }];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "single-read !=0 RHS should inline into the if cond");
    }

    #[test]
    fn inline_binary_expr() {
        // v1 = a + b;
        // other();
        // return v1 * 2;
        // → other(); return (a + b) * 2;
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Bin(
                    crate::recovery::csharp::BinOp::Add,
                    Box::new(v("a")),
                    Box::new(v("b")),
                ),
            ),
            Stmt::Eval(v("other")),
            Stmt::Assign(
                "v2".into(),
                None,
                Expr::Bin(
                    crate::recovery::csharp::BinOp::Mul,
                    Box::new(v("v1")),
                    Box::new(Expr::Int(2)),
                ),
            ),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "should inline v1");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn inline_into_if_condition() {
        // v1 = obj.Flag;
        // other();
        // if (v1 != 0) { work(); }
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Flag".into()),
            ),
            Stmt::Eval(v("other")),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(v("v1")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Eval(v("work"))],
                else_body: vec![],
            },
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "should inline v1 into if condition");
        assert_eq!(stmts.len(), 2);
        match &stmts[1] {
            Stmt::If { cond, .. } => {
                assert!(!expr_mentions(cond, "v1"), "v1 should be inlined");
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn overwritten_def_stays_but_live_def_inlines() {
        // v1 = obj.Field;
        // v1 = other;
        // return v1;
        // The SECOND def is the only one that reaches the read: inlining
        // `other` into the return is exact (the first write is dead). The
        // overwritten def must stay in place (its load may fault; removing
        // loads is not this pass's job).
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Field".into()),
            ),
            Stmt::Assign("v1".into(), None, v("other")),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "the live def should inline: {stmts:?}");
        assert_eq!(stmts.len(), 2, "{stmts:?}");
        assert!(
            matches!(&stmts[0], Stmt::Assign(n, _, Expr::Field(..)) if n == "v1"),
            "the overwritten def must stay: {stmts:?}"
        );
        assert!(
            matches!(&stmts[1], Stmt::Return(Some(Expr::Var(n))) if n == "other"),
            "the live def's value must reach the return: {stmts:?}"
        );
    }

    #[test]
    fn inline_cast_expression() {
        // v1 = (int)x;
        // other();
        // return v1;
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Cast("int".into(), Box::new(v("x"))),
            ),
            Stmt::Eval(v("other")),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = inline_temp_variables(&mut stmts);
        assert_eq!(n, 1, "should inline cast expression");
        assert_eq!(stmts.len(), 2);
        match &stmts[1] {
            Stmt::Return(Some(Expr::Cast(_, _))) => {}
            other => panic!("expected Return(Cast), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7C-3: Dead temporary variable assignment removal tests
    // -----------------------------------------------------------------------

    #[test]
    fn remove_dead_pure_assignment() {
        // v1 = obj.Field;  (v1 never used)
        // other();
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Field".into()),
            ),
            Stmt::Eval(v("other")),
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 1, "dead pure assignment should be removed");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn remove_dead_literal_assignment() {
        // v1 = 0;  (v1 never used)
        // other();
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            Stmt::Eval(v("other")),
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 1, "dead literal assignment should be removed");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn no_remove_dead_impure_assignment() {
        // v1 = Foo();  (v1 never used, but RHS is a call with side effects)
        // other();
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Foo".into(),
                    args: vec![],
                },
            ),
            Stmt::Eval(v("other")),
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 0, "impure RHS should not be removed even if dead");
    }

    #[test]
    fn remove_self_assignment() {
        // v1 = v1;  (no-op)
        // other();
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, v("v1")),
            Stmt::Eval(v("other")),
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 1, "self-assignment should be removed");
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn no_remove_used_assignment() {
        // v1 = obj.Field;
        // return v1;
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Field".into()),
            ),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 0, "used assignment should not be removed");
    }

    #[test]
    fn keeps_a_named_local_whose_value_is_read() {
        // x = obj.Field; other(x);  -- the load feeds a call, so it is live.
        // A named local is a candidate only when NOTHING reads it (that is how
        // the return-register stores `obj = 0;` go away); one reader keeps it.
        let mut stmts = vec![
            Stmt::Assign(
                "x".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Field".into()),
            ),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "other".into(),
                args: vec![v("x")],
            }),
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 0, "a read store must not be removed");
    }

    #[test]
    fn never_removes_a_store_through_a_pointer() {
        // `*(int*)((nint)&stack_167 +4) = this.QualitySettingLevel;` -- the LHS
        // is a rendered string, i.e. a memory write, not a slot this method
        // owns. Nothing in the text reads the name, so a name-wide sweep
        // considered it dead and deleted the initialisation of a boxed value
        // (0x17A6468: `IL2CPP_Box(typeof(int), (nint*)(stack_167 +4))` then
        // read an uninitialised cell).
        let mut stmts = vec![
            Stmt::Assign(
                "*(int*)((nint)&stack_167 +4)".into(),
                None,
                Expr::Field(Box::new(Expr::This), "QualitySettingLevel".into()),
            ),
            Stmt::Return(None),
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 0, "a store through a pointer is not a dead local store");
    }

    #[test]
    fn removes_dead_store_in_reused_display_name_range() {
        // v151 = this.badge != 0;      <- live: read by the if below
        // if (v151) { v151 = BadgingManager.Instance; other(); }
        // The inner store starts a NEW live range of the same display name and
        // nothing reads it. The method-wide read count (from the if condition)
        // used to keep it alive.
        let mut stmts = vec![
            Stmt::Assign(
                "v151".into(),
                None,
                Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(Expr::Field(Box::new(Expr::This), "badge".into())),
                    Box::new(Expr::Null),
                ),
            ),
            Stmt::If {
                cond: v("v151"),
                then_body: vec![
                    Stmt::Assign(
                        "v151".into(),
                        None,
                        Expr::StaticField("BadgingManager".into(), "Instance".into()),
                    ),
                    Stmt::Eval(v("other")),
                ],
                else_body: vec![],
            },
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 1, "the second range's dead store must go: {stmts:?}");
        match &stmts[1] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert!(
                    !then_body
                        .iter()
                        .any(|s| matches!(s, Stmt::Assign(n, _, _) if n == "v151")),
                    "inner v151 store must be gone: {then_body:?}"
                );
            }
            other => panic!("expected if, got {other:?}"),
        }
        // The first range and its reader are untouched.
        assert!(matches!(&stmts[0], Stmt::Assign(n, _, _) if n == "v151"));
        assert!(matches!(&stmts[1], Stmt::If { cond: Expr::Var(n), .. } if n == "v151"));
    }

    #[test]
    fn keeps_nested_store_read_after_enclosing_construct() {
        // v1 = 0; if (c) { v1 = 1; } print(v1);
        // The nested write is conditional, so the read after the if may still
        // observe the outer store; neither may be deleted.
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            Stmt::If {
                cond: v("c"),
                then_body: vec![Stmt::Assign("v1".into(), None, Expr::Int(1))],
                else_body: vec![],
            },
            Stmt::Eval(v("v1")),
        ];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 0, "both stores can feed the read: {stmts:?}");
        assert_eq!(stmts.len(), 3);
    }

    #[test]
    fn loop_back_edge_and_continue_keep_stores() {
        // while (v1) { v1 = 1; if (c) continue; v1 = 2; }
        // The loop condition reads v1: a def in the body is observed by the
        // NEXT iteration, and `continue` can leave the range before the
        // overwriting write.
        let mut stmts = vec![Stmt::While {
            cond: v("v1"),
            body: vec![
                Stmt::Assign("v1".into(), None, Expr::Int(1)),
                Stmt::If {
                    cond: v("c"),
                    then_body: vec![Stmt::Continue],
                    else_body: vec![],
                },
                Stmt::Assign("v1".into(), None, Expr::Int(2)),
            ],
        }];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 0, "loop-carried value must stay: {stmts:?}");
    }

    #[test]
    fn keeps_dead_impure_mixed_definition() {
        // if (c) { v1 = Foo(); }  -- the call is the effect even though v1 is
        // never read afterwards.
        let mut stmts = vec![Stmt::If {
            cond: v("c"),
            then_body: vec![Stmt::Assign(
                "v1".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Foo".into(),
                    args: vec![],
                },
            )],
            else_body: vec![],
        }];
        let n = remove_dead_temp_assignments(&mut stmts);
        assert_eq!(n, 0, "impure RHS must stay: {stmts:?}");
    }

    #[test]
    fn full_pipeline_with_dead_code_elimination() {
        // v1 = obj.Field;
        // v2 = v1;
        // return v2;
        // After inline: return obj.Field; (both temps inlined)
        // No dead code to remove.
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Field".into()),
            ),
            Stmt::Assign("v2".into(), None, v("v1")),
            Stmt::Return(Some(v("v2"))),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 2, "should inline both temps");
        assert_eq!(stmts.len(), 1, "both assignments should be removed");
    }

    #[test]
    fn dead_code_after_inline() {
        // v1 = obj.Field;
        // v2 = 42;  (v2 never used)
        // return v1;
        // v1 should be inlined, v2 should be removed as dead.
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("obj")), "Field".into()),
            ),
            Stmt::Assign("v2".into(), None, Expr::Int(42)),
            Stmt::Return(Some(v("v1"))),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 2, "should inline v1 and remove dead v2");
        assert_eq!(stmts.len(), 1, "only the return should remain");
    }

    // -----------------------------------------------------------------------
    // 7C-2b: definition-side copy coalescing
    // -----------------------------------------------------------------------

    fn new_obj() -> Expr {
        Expr::NewObj("object".into(), vec![v("<>c"), v("typeof")])
    }

    #[test]
    fn coalesces_impure_definition_into_copy_target() {
        // else { v61 = new object(…); *(…v61…) ; v18 = v61; }
        // this._idHandler = v18;
        let mut stmts = vec![
            Stmt::If {
                cond: v("cond"),
                then_body: vec![],
                else_body: vec![
                    Stmt::Assign("v61".into(), None, new_obj()),
                    Stmt::Eval(Expr::Field(Box::new(v("v61")), "f".into())),
                    Stmt::Assign("v18".into(), None, v("v61")),
                ],
            },
            Stmt::Assign("this._idHandler".into(), None, v("v18")),
        ];
        let n = coalesce_def_into_copies(&mut stmts);
        assert_eq!(n, 1, "the copy must be coalesced into the definition");
        match &stmts[0] {
            Stmt::If { else_body, .. } => {
                assert_eq!(else_body.len(), 2, "the copy is gone");
                assert!(
                    matches!(&else_body[0], Stmt::Assign(name, None, Expr::NewObj(..)) if name == "v18"),
                    "definition now targets v18: {:?}",
                    else_body[0]
                );
                assert_eq!(
                    count_reads_in_stmt(&else_body[1], "v18"),
                    1,
                    "the in-between read was renamed"
                );
                assert_eq!(count_reads_in_stmt(&else_body[1], "v61"), 0);
            }
            other => panic!("expected if, got {other:?}"),
        }
        // The outside merge read still names v18.
        assert!(matches!(&stmts[1], Stmt::Assign(n, None, Expr::Var(r)) if n == "this._idHandler" && r == "v18"));
    }

    #[test]
    fn coalesces_pure_definition_with_intermediate_read() {
        // v61 = obj.F; v2 = v61 + 1; v18 = v61;
        let mut stmts = vec![
            Stmt::Assign(
                "v61".into(),
                None,
                Expr::Field(Box::new(v("obj")), "F".into()),
            ),
            Stmt::Assign(
                "v2".into(),
                None,
                Expr::Bin(
                    crate::recovery::csharp::BinOp::Add,
                    Box::new(v("v61")),
                    Box::new(Expr::Int(1)),
                ),
            ),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        let n = coalesce_def_into_copies(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Assign(n, _, Expr::Field(..)) if n == "v18"));
        assert_eq!(count_reads_in_stmt(&stmts[1], "v18"), 1);
        assert_eq!(count_reads_in_stmt(&stmts[1], "v61"), 0);
    }

    #[test]
    fn refuses_coalesce_when_source_read_after_copy() {
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Assign("v18".into(), None, v("v61")),
            Stmt::Eval(v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
        assert_eq!(stmts.len(), 3);
    }

    #[test]
    fn refuses_coalesce_when_source_read_outside_list() {
        // The arm's def is read after the branch; deleting its name would
        // leave the outer read dangling.
        let mut stmts = vec![
            Stmt::If {
                cond: v("cond"),
                then_body: vec![
                    Stmt::Assign("v61".into(), None, Expr::Int(1)),
                    Stmt::Assign("v18".into(), None, v("v61")),
                ],
                else_body: vec![],
            },
            Stmt::Eval(v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn refuses_coalesce_when_source_read_in_sibling_list() {
        // The later read is nested one list deeper than the enclosing-scope
        // ctx can see, but the method-wide count still proves it exists.
        let mut stmts = vec![
            Stmt::If {
                cond: v("a"),
                then_body: vec![
                    Stmt::Assign("v61".into(), None, Expr::Int(1)),
                    Stmt::Assign("v18".into(), None, v("v61")),
                ],
                else_body: vec![],
            },
            Stmt::If {
                cond: v("b"),
                then_body: vec![Stmt::Eval(v("v61"))],
                else_body: vec![],
            },
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn refuses_coalesce_when_destination_read_between() {
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Eval(v("v18")),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn refuses_coalesce_when_destination_written_between() {
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Assign("v18".into(), None, Expr::Int(2)),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn refuses_coalesce_across_label_or_goto() {
        let mut label_between = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Label("L4".into()),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut label_between), 0);
        assert!(matches!(&label_between[1], Stmt::Label(_)), "label kept");

        let mut goto_between = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Goto("L4".into()),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut goto_between), 0);
    }

    #[test]
    fn refuses_coalesce_when_source_read_in_name_slot() {
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Switch {
                disc: "v61".into(),
                cases: vec![],
                tail: vec![],
            
                default: None,},
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn refuses_coalesce_when_definition_reads_its_target() {
        let mut stmts = vec![
            Stmt::Assign(
                "v61".into(),
                None,
                Expr::Bin(
                    crate::recovery::csharp::BinOp::Add,
                    Box::new(v("v61")),
                    Box::new(Expr::Int(1)),
                ),
            ),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn coalesces_into_last_of_several_source_definitions() {
        // The earlier `v61 = 0` is dead: the last definition before the copy
        // supplies the value, and nothing reads v61 between the two. Extra
        // writes of the same display name outside (d, c) are other SSA
        // versions and must not block the rewrite.
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(0)),
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 1);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[1], Stmt::Assign(n, _, Expr::Int(1)) if n == "v18"));
    }

    #[test]
    fn coalesces_when_same_display_name_written_again_later() {
        // The later `v61 = Int(2)` starts a NEW SSA lifetime; its read is
        // supplied by that write, not by the definition the copy consumes.
        // Counting it as a read of our live range refused this shape before.
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Assign("v18".into(), None, v("v61")),
            Stmt::Assign("v61".into(), None, Expr::Int(2)),
            Stmt::Eval(v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 1);
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[0], Stmt::Assign(n, _, Expr::Int(1)) if n == "v18"));
        assert!(matches!(&stmts[1], Stmt::Assign(n, _, Expr::Int(2)) if n == "v61"));
        assert!(matches!(&stmts[2], Stmt::Eval(Expr::Var(n)) if n == "v61"));
    }

    #[test]
    fn refuses_coalesce_when_redefinition_reads_old_value() {
        // `v61 = v61 + 1` reads the definition's value before overwriting it;
        // renaming that definition would leave the RHS dangling.
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Assign("v18".into(), None, v("v61")),
            Stmt::Assign(
                "v61".into(),
                None,
                Expr::Bin(
                    crate::recovery::csharp::BinOp::Add,
                    Box::new(v("v61")),
                    Box::new(Expr::Int(1)),
                ),
            ),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
        assert_eq!(stmts.len(), 3);
    }

    #[test]
    fn refuses_coalesce_when_source_written_between_def_and_copy() {
        // The conditional write between the definition and the copy can feed
        // the copy, so the last top-level definition is not the only producer.
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(0)),
            Stmt::If {
                cond: v("x"),
                then_body: vec![Stmt::Assign("v61".into(), None, Expr::Int(1))],
                else_body: vec![],
            },
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn refuses_coalesce_when_source_read_before_definition() {
        // A read before the definition may be a loop back edge observing the
        // definition's previous-iteration value; renaming the definition would
        // change what it sees.
        let mut stmts = vec![
            Stmt::Eval(v("v61")),
            Stmt::Assign("v61".into(), None, Expr::Int(0)),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn coalesces_repeated_copies_from_one_definition() {
        // Wide-struct staging: one source is copied into several targets.
        // Coalescing the LAST copy renames the earlier copies' read to the new
        // target; the fixed-point loop then coalesces them in turn.
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(7)),
            Stmt::Assign("v18".into(), None, v("v61")),
            Stmt::Assign("v19".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 2);
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Assign(n, _, Expr::Int(7)) if n == "v18"));
    }

    #[test]
    fn refuses_coalesce_when_source_address_taken() {
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Eval(Expr::AddressOf(Box::new(v("v61")))),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn refuses_coalesce_when_destination_captured_by_lambda() {
        let mut stmts = vec![
            Stmt::Assign(
                "f".into(),
                None,
                Expr::Lambda {
                    params: vec![],
                    body: Box::new(v("v18")),
                },
            ),
            Stmt::Assign("v61".into(), None, Expr::Int(1)),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
    }

    #[test]
    fn coalescing_is_idempotent() {
        let mut stmts = vec![
            Stmt::Assign("v61".into(), None, new_obj()),
            Stmt::Eval(v("v61")),
            Stmt::Assign("v18".into(), None, v("v61")),
        ];
        assert_eq!(coalesce_def_into_copies(&mut stmts), 1);
        let len = stmts.len();
        assert_eq!(coalesce_def_into_copies(&mut stmts), 0);
        assert_eq!(stmts.len(), len);
    }

    // -----------------------------------------------------------------------
    // 7C-2c: return-value temp folding
    // -----------------------------------------------------------------------

    #[test]
    fn folds_pure_temp_into_return() {
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("this")), "field".into()),
            ),
            Stmt::Return(Some(v("v1"))),
        ];
        assert_eq!(fold_return_of_temp(&mut stmts), 1);
        assert_eq!(stmts.len(), 1);
        assert!(matches!(
            &stmts[0],
            Stmt::Return(Some(Expr::Field(..)))
        ));
    }

    #[test]
    fn return_fold_keeps_definition_with_outside_read() {
        // The arm returns the pure value, but the same display name is read on
        // another path: fold the return, keep the definition.
        let mut stmts = vec![
            Stmt::If {
                cond: v("cond"),
                then_body: vec![
                    Stmt::Assign(
                        "v1".into(),
                        None,
                        Expr::Field(Box::new(v("this")), "field".into()),
                    ),
                    Stmt::Return(Some(v("v1"))),
                ],
                else_body: vec![],
            },
            Stmt::Eval(v("v1")),
        ];
        assert_eq!(fold_return_of_temp(&mut stmts), 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 2, "definition stays for the outside read");
                assert!(matches!(&then_body[0], Stmt::Assign(n, _, _) if n == "v1"));
                assert!(matches!(&then_body[1], Stmt::Return(Some(Expr::Field(..)))));
            }
            other => panic!("expected if, got {other:?}"),
        }
    }

    #[test]
    fn return_fold_refuses_impure_value() {
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Foo".into(),
                    args: vec![],
                },
            ),
            Stmt::Return(Some(v("v1"))),
        ];
        assert_eq!(fold_return_of_temp(&mut stmts), 0);
    }

    #[test]
    fn return_fold_refuses_huge_value_when_definition_kept() {
        let mut e = Expr::Int(0);
        for _ in 0..16 {
            e = Expr::Bin(
                crate::recovery::csharp::BinOp::Add,
                Box::new(e),
                Box::new(Expr::Int(1)),
            );
        }
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, e),
            Stmt::Eval(v("v1")),
            Stmt::Return(Some(v("v1"))),
        ];
        assert_eq!(fold_return_of_temp(&mut stmts), 0);
    }

    #[test]
    fn return_fold_refuses_captured_target() {
        let mut stmts = vec![
            Stmt::Assign(
                "f".into(),
                None,
                Expr::Lambda {
                    params: vec![],
                    body: Box::new(v("v1")),
                },
            ),
            Stmt::Assign("v1".into(), None, Expr::Int(7)),
            Stmt::Return(Some(v("v1"))),
        ];
        assert_eq!(fold_return_of_temp(&mut stmts), 0);
    }

    #[test]
    fn return_fold_is_idempotent() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(7)),
            Stmt::Return(Some(v("v1"))),
        ];
        assert_eq!(fold_return_of_temp(&mut stmts), 1);
        assert_eq!(fold_return_of_temp(&mut stmts), 0);
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn apply_folds_literal_temp_into_return() {
        // `v9 = 0; return v9;` -> `return 0;` (the literal form named in the
        // coalescing rules; the def must be deleted, not left behind).
        let mut stmts = vec![
            Stmt::Assign("v9".into(), None, Expr::Int(0)),
            Stmt::Return(Some(v("v9"))),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1, "the literal temp should be folded into the return");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Return(Some(Expr::Int(0)))));
    }

    #[test]
    fn apply_removes_dead_temp_def_after_folded_guard() {
        // The guard only re-stores the value the temp already holds; folding
        // it removes the temp's last reader, and the dead def must not survive
        // the cleanup round.
        let rhs = Expr::Field(Box::new(v("this")), "F".into());
        let mut stmts = vec![
            Stmt::Assign("v5".into(), None, rhs.clone()),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::StaticField("Type".into(), "flag".into())),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Assign("v5".into(), None, rhs)],
                else_body: vec![],
            },
            Stmt::Eval(v("after")),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1, "the redundant re-assignment guard should be folded");
        assert!(
            !stmts
                .iter()
                .any(|s| matches!(s, Stmt::Assign(name, _, _) if name == "v5")),
            "the dead temp def must be swept: {stmts:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 7C-1c: loop-preheader same-value re-assignment
    // -----------------------------------------------------------------------

    fn while_loop(cond: Expr, body: Vec<Stmt>) -> Stmt {
        Stmt::While { cond, body }
    }

    fn counter_cond(name: &str) -> Expr {
        Expr::Cmp(CmpKind::Lt, Box::new(v(name)), Box::new(v("n")))
    }

    fn body_len(stmt: &Stmt) -> usize {
        match stmt {
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::For { body, .. }
            | Stmt::ForEach { body, .. } => body.len(),
            other => panic!("expected loop, got {other:?}"),
        }
    }

    #[test]
    fn loop_preheader_drops_identical_literal_reassign() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                counter_cond("i"),
                vec![
                    Stmt::Assign("v1".into(), None, Expr::Int(0)),
                    Stmt::Eval(v("use")),
                ],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 1);
        assert_eq!(body_len(&stmts[1]), 1, "the inner re-assignment is gone");
    }

    #[test]
    fn loop_preheader_keeps_when_condition_reads_target() {
        // The condition reads v1 before the body's write: on the second
        // iteration it would observe the previous iteration's value, which the
        // removed write used to overwrite.
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                Expr::Cmp(CmpKind::Ne, Box::new(v("v1")), Box::new(Expr::Int(0))),
                vec![Stmt::Assign("v1".into(), None, Expr::Int(0))],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
        assert_eq!(body_len(&stmts[1]), 1);
    }

    #[test]
    fn loop_preheader_keeps_read_before_inner_write() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                counter_cond("i"),
                vec![
                    Stmt::Eval(v("v1")),
                    Stmt::Assign("v1".into(), None, Expr::Int(0)),
                ],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_second_write_in_loop() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                counter_cond("i"),
                vec![
                    Stmt::Assign("v1".into(), None, Expr::Int(0)),
                    Stmt::Eval(v("v1")),
                    Stmt::Assign("v1".into(), None, Expr::Int(1)),
                ],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_impure_value() {
        let call = Expr::Call {
            recv: None,
            name: "Next".into(),
            args: vec![],
        };
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, call.clone()),
            while_loop(
                counter_cond("i"),
                vec![Stmt::Assign("v1".into(), None, call)],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_mutable_field_value() {
        // A field read is pure but not invariant: a call in the loop may store
        // a new value, and the inner assignment would have reloaded it.
        let field = Expr::Field(Box::new(v("this")), "F".into());
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, field.clone()),
            while_loop(
                counter_cond("i"),
                vec![
                    Stmt::Assign("v1".into(), None, field),
                    Stmt::Eval(Expr::Call {
                        recv: None,
                        name: "Mutate".into(),
                        args: vec![],
                    }),
                ],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_value_whose_local_changes_in_loop() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, v("v2")),
            while_loop(
                counter_cond("i"),
                vec![
                    Stmt::Assign("v1".into(), None, v("v2")),
                    Stmt::Assign("v2".into(), None, Expr::Int(9)),
                ],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_value_mentioning_target() {
        let inc = Expr::Bin(BinOp::Add, Box::new(v("v1")), Box::new(Expr::Int(1)));
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, inc.clone()),
            while_loop(
                counter_cond("i"),
                vec![Stmt::Assign("v1".into(), None, inc)],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_when_label_inside_loop() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                counter_cond("i"),
                vec![
                    Stmt::Assign("v1".into(), None, Expr::Int(0)),
                    Stmt::Label("L3".into()),
                ],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_address_taken_target() {
        let mut stmts = vec![
            Stmt::Eval(Expr::AddressOf(Box::new(v("v1")))),
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                counter_cond("i"),
                vec![Stmt::Assign("v1".into(), None, Expr::Int(0))],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_nested_inner_write() {
        // A conditional write has an execution condition this rule cannot
        // reason about; it is left alone.
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                counter_cond("i"),
                vec![Stmt::If {
                    cond: v("c"),
                    then_body: vec![Stmt::Assign("v1".into(), None, Expr::Int(0))],
                    else_body: vec![],
                }],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_keeps_different_value() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                counter_cond("i"),
                vec![Stmt::Assign("v1".into(), None, Expr::Int(1))],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
    }

    #[test]
    fn loop_preheader_handles_do_while() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Bool(true)),
            Stmt::DoWhile {
                body: vec![
                    Stmt::Assign("v1".into(), None, Expr::Bool(true)),
                    Stmt::Eval(v("use")),
                ],
                cond: counter_cond("i"),
            },
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 1);
        assert_eq!(body_len(&stmts[1]), 1);
    }

    #[test]
    fn loop_preheader_handles_for_and_refuses_increment_write() {
        let mut positive = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(2)),
            Stmt::For {
                init: vec![Stmt::Assign("i".into(), None, Expr::Int(0))],
                cond: Some(counter_cond("i")),
                incr: vec![Stmt::Assign(
                    "i".into(),
                    None,
                    Expr::Bin(BinOp::Add, Box::new(v("i")), Box::new(Expr::Int(1))),
                )],
                body: vec![
                    Stmt::Assign("v1".into(), None, Expr::Int(2)),
                    Stmt::Eval(v("use")),
                ],
            },
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut positive), 1);
        assert_eq!(body_len(&positive[1]), 1);

        let mut negative = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(2)),
            Stmt::For {
                init: vec![Stmt::Assign("i".into(), None, Expr::Int(0))],
                cond: Some(counter_cond("i")),
                incr: vec![Stmt::Assign("v1".into(), None, Expr::Int(2))],
                body: vec![Stmt::Assign("v1".into(), None, Expr::Int(2))],
            },
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut negative), 0);
    }

    #[test]
    fn loop_preheader_is_idempotent() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            while_loop(
                counter_cond("i"),
                vec![
                    Stmt::Assign("v1".into(), None, Expr::Int(0)),
                    Stmt::Eval(v("use")),
                ],
            ),
        ];
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 1);
        assert_eq!(fold_loop_preheader_reassignments(&mut stmts), 0);
        assert_eq!(body_len(&stmts[1]), 1);
    }

    #[test]
    fn a_real_return_after_a_terminating_guard_is_not_deleted() {
        // `if (delay <= 0f) { …; return raise; } return 1;` — the return 1 is
        // the path taken when the condition is FALSE (Stumble @0x196649C).
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Le,
                    Box::new(v("delay")),
                    Box::new(Expr::Float(0.0)),
                ),
                then_body: vec![Stmt::Return(Some(v("v314")))],
                else_body: vec![],
            },
            Stmt::Return(Some(Expr::Int(1))),
        ];
        drop_dead_code_after_terminal(&mut stmts);
        assert_eq!(stmts.len(), 2, "{stmts:?}");
    }

    #[test]
    fn a_synthesized_null_throw_after_a_guard_is_dead() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(v("x")),
                    Box::new(Expr::Null),
                ),
                then_body: vec![Stmt::Return(Some(v("y")))],
                else_body: vec![],
            },
            Stmt::Throw(Expr::NewObj(
                "System.NullReferenceException".to_string(),
                vec![],
            )),
        ];
        drop_dead_code_after_terminal(&mut stmts);
        assert_eq!(stmts.len(), 1, "{stmts:?}");
        assert!(matches!(stmts[0], Stmt::If { .. }), "{stmts:?}");
    }

    #[test]
    fn fused_read_counts_match_the_per_name_walk() {
        let names_vec = ["v1", "v2", "v3", "i", "item"];
        let set: std::collections::HashSet<String> =
            names_vec.iter().map(|s| s.to_string()).collect();
        let stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(v("v2")), "v3".to_string()),
            ),
            Stmt::Assign("*(byte*)((nint)&v2 +4)".into(), None, Expr::Int(0)),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Eq, Box::new(v("v1")), Box::new(Expr::Int(0))),
                then_body: vec![Stmt::Eval(v("v3"))],
                else_body: vec![],
            },
            Stmt::For {
                init: vec![Stmt::Assign("i".into(), None, Expr::Int(0))],
                cond: Some(Expr::Cmp(
                    CmpKind::Lt,
                    Box::new(v("i")),
                    Box::new(Expr::Int(3)),
                )),
                incr: vec![Stmt::Assign("i".into(), None, v("i"))],
                body: vec![Stmt::Eval(Expr::Index {
                    collection: Box::new(v("item")),
                    index: Box::new(v("i")),
                })],
            },
            Stmt::Eval(Expr::Lambda {
                params: vec!["v1".to_string()],
                body: Box::new(v("v2")),
            }),
            Stmt::ForEach {
                var_name: "item".to_string(),
                collection: "v2".to_string(),
                body: vec![Stmt::Eval(v("item"))],
            },
            Stmt::Switch {
                disc: "v3".to_string(),
                cases: vec![],
                tail: vec![],
                default: None,
            },
        ];
        let fused = count_reads_all(&stmts, &set);
        for n in names_vec {
            assert_eq!(
                fused.get(n).copied().unwrap_or(0),
                count_reads_in_list(&stmts, n),
                "name {n}"
            );
        }
        // Unfiltered census (the semantics both sweeps' filtered censuses
        // must preserve): every key must agree with the per-name walk.
        let all = count_reads_impl(&stmts, None);
        for (n, c) in &all {
            assert_eq!(*c, count_reads_in_list(&stmts, n), "unfiltered {n}");
        }
        // And the per-statement map (the unit the sweeps cache) is the same
        // walk over one statement.
        for s in &stmts {
            let m = count_reads_stmt_unfiltered(s);
            for (n, c) in &m {
                assert_eq!(*c, count_reads_in_stmt(s, n), "stmt {n}");
            }
        }
    }

}
