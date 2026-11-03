//! High-level C# CONSTRUCT reconstruction over the FINISHED statement tree.
//!
//! ## Why this is a sibling of `idioms`, not part of it
//!
//! `idioms` removes machine noise (dead gotos, unreferenced labels, static-init
//! guards). `reconstruct` does the opposite job: it rebuilds the *source-level
//! construct* the structurer could only emit in its flat machine form — a
//! `for` loop that arrived as `init; while (cond) { …; update; }`, a `lock`
//! that arrived as a bare `Monitor.Enter(x, 0)` / `Monitor.Exit(x)` pair, an
//! iterator that arrived as a `<>1__state` dispatch, a lambda that arrived as a
//! delegate construction. Those shapes are exactly what separates "recovered
//! machine code" from "readable C#".
//!
//! ## The two rules (same as `idioms`, and they are not negotiable)
//!
//! 1. **Never delete an observation.** A recognizer that removes a statement
//!    must prove the statement evaluates nothing and gates nothing.
//! 2. **Never move a write.** Hoisting/reordering is allowed only when nothing
//!    between the old and new position can observe the difference (no call, no
//!    construction, no write to a name the moved expression reads).
//!
//! A failed match is a NO-OP: the flat (correct, lower-level) sequence stays.
//! Accurate-but-ugly always beats plausible-but-wrong.
//!
//! ## Measured reality (x64 PE `Simple_6000_6_0_b1/GameAssembly.dll`, 47,924
//! methods, 2,454,023-line `dump.cs`) — this is what the shapes ACTUALLY look
//! like, and every rule below is written against it:
//!
//! | construct | in the real dump | real shape |
//! |---|---|---|
//! | `for (` | **0** | `init; …; while (cond) { …; update; …; continue; }` |
//! | `lock (` | **0** | `Monitor.Enter(X, 0);` … `Monitor.Exit(X);` INLINE — there is no `try/finally` (145 `Monitor.Enter` sites) |
//! | `using (` | **0** | `try` 18, `finally` **0** — the EH structurer emits no `finally` on this corpus, so the trait shape is unreachable here |
//! | `switch (` | **1** | the structurer emits goto-shaped case bodies, so a `Switch`-keyed recognizer cannot fire |
//! | `new T(` | **0** | an allocation renders as `T..ctor(args)` / `recv..ctor(args)` — an `Expr::Call` whose name ends in `.ctor`, never `Expr::NewObj` |
//!
//! The previous round of recognizers was written against *invented* shapes
//! (one-argument `Monitor.Enter` inside a `try/finally`, a `Switch` on
//! `<>1__state`, an adjacent counter init, `Expr::NewObj` for delegates) and
//! therefore fired **zero times in 47,924 methods** while `idioms` fired
//! hundreds of thousands of times. Each file below now documents, next to its
//! rules, which measured shape it exists for and which part of it provably
//! cannot fire here.

use super::csharp::{Expr, Stmt};
use super::resolve::ResolveMaps;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod algebraic_simplify;
pub mod alloc_types;
pub mod array_init;
pub mod array_element_sugar;
pub mod interface_scan;
pub mod collection_add;
pub mod array_length;
pub mod byref_deref;
pub mod auto_property;
pub mod base_call;
pub mod cast_elision;
pub mod closures;
pub mod merge_tail;
pub mod collection_init;
pub mod const_inline;
pub mod declare_locals;
pub mod copy_prop;
pub mod dead_store;
pub mod dead_tail;
pub mod merge_guards;
pub mod list_inline;
pub mod list_sugar;
pub mod compare_sugar;
pub mod clamp_sugar;
pub mod delegate_ctor;
pub mod equality_ops;
pub mod expr_cleanup;
pub mod operator_overload;
pub mod event_handler;
pub mod for_loop;
pub mod generic_inference;
pub mod dup_update_fold;
pub mod field_fixups;
pub mod fp_return_bind;
pub mod liveness_dse;
pub mod dead_write_only;
pub mod return_tail_fold;
pub mod return_ternary;
pub mod tail_guard_flatten;
pub mod closure_fields;
pub mod delegate_invoke;
pub mod class_init_stores;
pub mod role_naming;
pub mod tail_block_inline;
pub mod guard_dedup;
pub mod guard_invert;
pub mod il2cpp_cleanup;
pub mod indexer;
pub mod join_hoist;
pub mod property_access;
pub mod null_coalesce;
pub mod null_conditional;
pub mod nullability;
pub mod nullable_ops;
pub mod object_init;
pub mod pattern_match;
pub mod stack_naming;
pub mod store_sugar;
pub mod this_sugar;
pub mod stack_forward;
pub mod state_machine;
pub mod string_concat;
pub mod string_interp;
pub mod string_switch;
pub mod undef_args;
pub mod switch_recon;
pub mod switch_expr;
pub mod remove_continues;
pub mod foreach_recon;
pub mod goto_simplify;
pub mod goto_fold;
pub mod goto_fallthrough;
pub mod using_lock;
pub mod vtable_resolution;
pub mod field_annotation;
pub mod float_locals;
pub mod pointer_offsets;
pub mod enum_names;
pub mod attributes;
pub mod tuple_recon;
pub mod linq_recon;
pub mod local_naming;
pub mod local_types;
pub mod lambda_inline;
pub mod layout_sugar;
pub mod range_index;
pub mod ptr_resolution;

#[cfg(test)]
mod integration_tests;

/// A recognizer: rewrites one statement list, returning the number of rewrites
/// it applied (so a run report can show which recognizer actually fires).
pub type Pass = fn(&mut Vec<Stmt>) -> usize;

/// Ordering is deliberate. Peephole repairs (`nullability`) run first so the
/// structural recognizers see the cleanest conditions; the structural
/// recognizers run before `for_loop`, which consumes the `while` shapes
/// `using_lock`/`state_machine` may leave behind.
///
/// `merge_fallthrough` is a PRE-PASS: it runs before every recognizer so the
/// same-list pairing logic in `using_lock` (and any other pass that pairs
/// events within one `Vec<Stmt>`) can see across labelled-block boundaries.
///
/// Note: `closures` runs in the metadata-aware pipeline (`apply_with_maps`)
/// after `lambda_inline`, not in this standard array.
const PASSES: &[(&str, Pass)] = &[
    ("merge_fallthrough", merge_fallthrough_lists),
    ("normalize_ctor_calls", normalize_ctor_calls),
    ("nullability", nullability::apply),
    ("nullable_ops", nullable_ops::apply),
    ("null_coalesce", null_coalesce::apply),
    ("null_conditional", null_conditional::apply),
    ("il2cpp_cleanup", il2cpp_cleanup::apply),
    ("equality_ops", equality_ops::apply),
    ("algebraic_simplify", algebraic_simplify::apply),
    ("clamp_sugar", clamp_sugar::apply),
    ("cast_elision", cast_elision::apply),
    ("operator_overload", operator_overload::apply),
    ("array_length", array_length::apply),
    ("indexer", indexer::apply),
    ("property_access", property_access::apply),
    ("event_handler", event_handler::apply),
    ("pattern_match", pattern_match::apply),
    ("array_init", array_init::apply),
    ("list_inline", list_inline::apply),
    ("list_sugar", list_sugar::apply),
    ("collection_init", collection_init::apply),
    ("object_init", object_init::apply),
    ("base_call", base_call::apply),
    ("delegate_ctor", delegate_ctor::apply),
    ("using_lock", using_lock::apply),
    ("switch_recon", switch_recon::apply),
    ("switch_expr", switch_expr::apply),
    ("string_concat", string_concat::apply),
    ("string_interp", string_interp::apply),
    ("string_switch", string_switch::apply),
    ("state_machine", state_machine::apply),
    ("generic_inference", generic_inference::apply),
    ("dup_update_fold", dup_update_fold::apply),
    ("for_loop", for_loop::apply),
    ("foreach_recon", foreach_recon::apply),
    ("remove_continues", remove_continues::apply),
    ("join_hoist", join_hoist::apply),
    ("goto_simplify", goto_simplify::apply),
    ("goto_fold", goto_fold::apply),
    ("join_hoist_final", join_hoist::apply),
    ("goto_fallthrough", goto_fallthrough::apply),
    ("dead_tail", dead_tail::apply),
    ("tuple_recon", tuple_recon::apply),
    ("linq_recon", linq_recon::apply),
    ("range_index", range_index::apply),
    ("merge_tail", merge_tail::apply),
];

fn label_goto_census(stmts: &[Stmt]) -> (HashSet<String>, HashSet<String>, HashSet<String>) {
    let mut labels = HashSet::new();
    let mut gotos = HashSet::new();
    fn walk(
        stmts: &[Stmt],
        labels: &mut HashSet<String>,
        gotos: &mut HashSet<String>,
    ) {
        for s in stmts {
            match s {
                Stmt::Label(l) => {
                    labels.insert(l.clone());
                }
                Stmt::Goto(l) => {
                    gotos.insert(l.clone());
                }
                _ => {}
            }
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            children(s, &mut kids);
            for k in kids {
                walk(k, labels, gotos);
            }
        }
    }
    walk(stmts, &mut labels, &mut gotos);
    (labels.clone(), labels, gotos)
}

/// Run every reconstruct recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) {
    let verbose = crate::recovery::env_flags::flag("ILR_RECON");
    let skip_alg = crate::recovery::env_flags::flag("ILR_NOALG");
    let no_final = crate::recovery::env_flags::flag("ILR_NOFINAL");
    // PASSES index of the cleanup round. Everything AFTER it can expose new
    // dead temps; if none of those passes rewrote anything, the tree is still
    // the exact one cleanup converged on, so the final cleanup below is
    // provably a no-op and its (expensive) full-tree sweeps can be skipped.
    let cleanup_idx = PASSES
        .iter()
        .position(|(name, _)| *name == "il2cpp_cleanup");
    let mut changed_after_cleanup = false;
    if crate::recovery::env_flags::flag("ILR_GOTODBG2") {
        let (labels, defs, gotos) = label_goto_census(stmts);
        let missing: Vec<&String> = gotos.difference(&defs).collect();
        eprintln!("[g2] initial: labels={} gotos={} missing={:?}", labels.len(), gotos.len(), missing);
    }
    for (idx, (name, pass)) in PASSES.iter().enumerate() {
        if skip_alg && *name == "algebraic_simplify" {
            continue;
        }
        // `ILR_LABELDBG=1`: attribute new dangling gotos to the pass that
        // orphaned their labels. One extra label/goto census per pass; only
        // on when the flag is set.
        let labeldbg = crate::recovery::env_flags::flag("ILR_LABELDBG");
        // `ILR_TRACE=1`: per-label provenance — which pass removed a label
        // that a surviving `goto` still names, in which method.
        let trace = crate::recovery::env_flags::flag("ILR_TRACE");
        let labels_before = if trace {
            let (l, _, _) = label_goto_census(stmts);
            l
        } else {
            HashSet::new()
        };
        let missing_before = if labeldbg {
            let (_, defs, gotos) = label_goto_census(stmts);
            gotos.difference(&defs).count()
        } else {
            0
        };
        let t_pass = std::time::Instant::now();
        let n = pass(stmts);
        if labeldbg {
            let (_, defs, gotos) = label_goto_census(stmts);
            let after = gotos.difference(&defs).count();
            if after > missing_before {
                LABEL_LOST[idx].fetch_add(
                    (after - missing_before) as u64,
                    Ordering::Relaxed,
                );
            }
        }
        if trace && !labels_before.is_empty() {
            let (labels_after, defs, gotos) = label_goto_census(stmts);
            for l in labels_before.difference(&labels_after) {
                if gotos.contains(l) && !defs.contains(l) {
                    eprintln!("[labdrop] {:#x} {name} {l}", trace_rva());
                }
            }
        }
        PASS_NANOS[idx].fetch_add(
            t_pass.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        if crate::recovery::env_flags::flag("ILR_PASSDUMP") {
            let _ = std::fs::write(
                super::probe_dir().join(format!("recon_{idx:02}_{name}.txt")),
                format!("{stmts:#?}"),
            );
        }
        if crate::recovery::env_flags::flag("ILR_PASSDBG") {
            let d = format!("{stmts:?}");
            let defs = |want: &str| {
                d.match_indices("Assign(")
                    .filter(|(i, _)| d[*i + 7..].trim_start().starts_with(&format!("\"{want}\"")))
                    .count()
            };
            eprintln!(
                "[passdbg] {name}: n={n} v403def={} v231def={} rets={} assigns={}",
                defs("v403"),
                defs("v231"),
                d.matches("Return(").count(),
                d.matches("Assign(").count()
            );
        }
        if crate::recovery::env_flags::flag("ILR_GOTODBG2") {
            let (labels, defs, gotos) = label_goto_census(stmts);
            let missing: Vec<&String> = gotos.difference(&defs).collect();
            if !missing.is_empty() || *name == "goto_simplify" || *name == "join_hoist" {
                eprintln!(
                    "[g2] {name}: passes={n} labels={} gotos={} missing={:?}",
                    labels.len(),
                    gotos.len(),
                    missing
                );
            }
        }
        if n > 0 && cleanup_idx.is_some_and(|c| idx > c) {
            changed_after_cleanup = true;
        }
        if n > 0 {
            RECON_TOTAL.fetch_add(n as u64, Ordering::Relaxed);
            // PASSES and BY_PASS have identical entries in the same order, so
            // the index into PASSES is the index into BY_PASS. This avoids an
            // O(m) linear scan of BY_PASS per pass (was O(n*m) per method).
            BY_PASS[idx].1.fetch_add(n as u64, Ordering::Relaxed);
            if verbose {
                eprintln!("[recon] {name}: {n} rewrite(s)");
            }
        }
    }
    // FINAL SIMPLIFICATION + INLINING ROUND. The structural recognizers above
    // (`for_loop`, `goto_simplify`, `range_index`, …) create fresh conditions
    // and copy expressions late in the pipeline; the last algebraic pass folds
    // those, and a last cleanup round then removes temps whose read counts
    // only became single after those passes ran (measured: ~1,050 temps whose
    // leftover guard/raise reads were dropped after the first cleanup round).
    let n = if no_final {
        0
    } else {
        algebraic_simplify::apply(stmts)
    };
    let n2 = if no_final || (n == 0 && !changed_after_cleanup) {
        0
    } else {
        il2cpp_cleanup::apply(stmts)
    };
    if n + n2 > 0 {
        RECON_TOTAL.fetch_add((n + n2) as u64, Ordering::Relaxed);
        if verbose {
            eprintln!("[recon] final_simplify: {n}, final_cleanup: {n2}");
        }
    }
    if crate::recovery::env_flags::flag("ILR_PASSDUMP") {
        let _ = std::fs::write(
            super::probe_dir().join("recon_final.txt"),
            format!("{stmts:#?}"),
        );
    }
    if crate::recovery::env_flags::flag("ILR_GOTODBG2") {
        let (labels, _defs, gotos) = label_goto_census(stmts);
        let (_, defs2, _) = label_goto_census(stmts);
        let missing: Vec<&String> = gotos.difference(&defs2).collect();
        eprintln!("[g2] final: labels={} gotos={} missing={:?}", labels.len(), gotos.len(), missing);
    }
}

/// Run metadata-aware reconstruction passes that require `ResolveMaps`.
///
/// This is called from the main pipeline after the standard `apply` passes
/// have run. It runs passes that need metadata context:
/// - VTable-based virtual dispatch resolution
/// - String literal restoration (via generic_inference::apply_with_metadata)
/// - Post-hoc field annotation (RawDeref → Field when base type is known)
///
/// `local_types` maps variable display names to their TypeDef index (tdi),
/// built from the `LocalTable` after the typing fixpoint converges. Pass an
/// empty map if local type information is not available.
///
/// Returns the total number of rewrites across all metadata-aware passes.
pub fn apply_with_maps(stmts: &mut Vec<Stmt>, maps: &ResolveMaps) -> usize {
    apply_with_maps_and_types(stmts, maps, &std::collections::HashMap::new())
}

/// Run metadata-aware reconstruction passes with local type information.
///
/// Like [`apply_with_maps`] but also accepts a `local_types` map that enables
/// the post-hoc field annotation pass to resolve `RawDeref` patterns to named
/// fields when the base variable's type is known.
pub fn apply_with_maps_and_types(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    local_types: &std::collections::HashMap<String, i32>,
) -> usize {
    apply_with_maps_and_types_and_tdi(stmts, maps, local_types, -1, None)
}

/// Run metadata-aware reconstruction passes with local type information and declaring type.
///
/// Like [`apply_with_maps_and_types`] but also accepts the declaring type's tdi
/// for ptr_no_offset resolution context.
pub fn apply_with_maps_and_types_and_tdi(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    local_types: &std::collections::HashMap<String, i32>,
    declaring_tdi: i32,
    generic_map: Option<&generic_inference::GenericInstantiationMap>,
) -> usize {
    let verbose = crate::recovery::env_flags::flag("ILR_RECON");
    let mut total = 0;

    // VTable dispatch resolution
    let n = vtable_resolution::apply_with_maps(stmts, maps);
    if n > 0 {
        total += n;
        if verbose {
            eprintln!("[recon] vtable_resolution: {n} rewrite(s)");
        }
    }

    // Metadata-aware generic type parameter resolution (GetComponent<T>, etc.)
    // Use pre-built generic_map if provided, otherwise build on the fly.
    let n = if let Some(gim) = generic_map {
        generic_inference::apply_with_prebuilt_map(stmts, gim)
    } else {
        let gim = generic_inference::build_generic_instantiation_map(maps);
        generic_inference::apply_generic_inference_with_metadata(stmts, &gim)
    };
    if n > 0 {
        total += n;
        if verbose {
            eprintln!("[recon] generic_metadata_resolution: {n} rewrite(s)");
        }
    }

    // String literal restoration (metadata-aware)
    let n = generic_inference::apply_with_metadata(
        stmts,
        &maps.literal_index_by_addr,
        &maps.literal_ptr_index_by_addr,
        &maps.literals,
    );
    if n > 0 {
        total += n;
        if verbose {
            eprintln!("[recon] string_token_restoration: {n} rewrite(s)");
        }
    }

    // Post-hoc field annotation: resolve remaining RawDeref to named fields.
    // Locals that hold a klass pointer get their type recovered first: the
    // renderer materialises klass expressions into temps, and without the type
    // a `.static_fields` access on such a temp cannot be bridged back to the
    // named static field.
    let mut local_types_for_fields = local_types.clone();
    for (name, tdi) in field_annotation::infer_klass_var_types(stmts, maps, local_types) {
        local_types_for_fields.entry(name).or_insert(tdi);
    }
    if crate::recovery::env_flags::flag("ILR_GOTODBG2") {
        let (labels, _, gotos) = label_goto_census(stmts);
        let (_, defs2, _) = label_goto_census(stmts);
        let missing: Vec<&String> = gotos.difference(&defs2).collect();
        eprintln!("[g2] pre-field-annot: labels={} gotos={} missing={:?}", labels.len(), gotos.len(), missing);
    }
    let n = field_annotation::apply_with_maps(stmts, maps, &local_types_for_fields, declaring_tdi);
    if n > 0 {
        total += n;
        if verbose {
            eprintln!("[recon] field_annotation: {n} rewrite(s)");
        }
    }

    // IL2CPP static field cleanup: strip klass()/static_fields machinery that
    // field_annotation may have left behind when it couldn't fully resolve the
    // static field. Must run AFTER field_annotation.
    let n = if crate::recovery::env_flags::flag("ILR_NOSTATICSUGAR") {
        0
    } else {
        il2cpp_cleanup::cleanup_static_field_access(stmts)
    };
    if n > 0 {
        total += n;
        if verbose {
            eprintln!("[recon] il2cpp_static_field_cleanup: {n} rewrite(s)");
        }
    }

    // Enum member name recovery: replace integer constants with enum member names
    // when the variable has an enum type.
    let n = enum_names::apply(stmts, maps, local_types);
    if n > 0 {
        total += n;
        ENUM_NAMES_REWRITES.fetch_add(n as u64, Ordering::Relaxed);
        if verbose {
            eprintln!("[recon] enum_names: {n} rewrite(s)");
        }
    }

    // C# attribute recovery: recognise custom attribute constructor calls in
    // static initialisers and emit [AttributeName(args)] comments.
    let n = attributes::apply_with_maps(stmts, maps);
    if n > 0 {
        total += n;
        ATTRIBUTE_REWRITES.fetch_add(n as u64, Ordering::Relaxed);
        if verbose {
            eprintln!("[recon] attributes: {n} rewrite(s)");
        }
    }

    // Phase 2D: Lambda body inlining — uses the global lambda body registry
    // (populated by Phase 5A during rendering) to directly rewrite delegate
    // constructions referencing compiler-generated lambda methods into
    // Expr::Lambda. This is the cross-method complement to closures.rs.
    let n = lambda_inline::apply_with_maps(stmts, maps);
    if n > 0 {
        total += n;
        LAMBDA_INLINE_REWRITES.fetch_add(n as u64, Ordering::Relaxed);
        if verbose {
            eprintln!("[recon] lambda_inline: {n} rewrite(s)");
        }
    }

    // Phase 2E: Closure folding — runs AFTER lambda_inline so that Expr::Lambda
    // nodes created by lambda_inline are available for closures::apply to fold.
    // Previously this ran in the standard PASSES array where it was provably inert
    // (no Expr::Lambda existed before lambda_inline ran).
    let n = closures::apply(stmts);
    if n > 0 {
        total += n;
        CLOSURES_REWRITES.fetch_add(n as u64, Ordering::Relaxed);
        if verbose {
            eprintln!("[recon] closures: {n} rewrite(s)");
        }
    }

    total
}

/// Per-recognizer rewrite counts, reported with the structural census so a
/// recognizer that silently never fires is visible rather than assumed.
static BY_PASS: [(&str, AtomicU64); 46] = [
    ("merge_fallthrough", AtomicU64::new(0)),
    ("normalize_ctor_calls", AtomicU64::new(0)),
    ("nullability", AtomicU64::new(0)),
    ("nullable_ops", AtomicU64::new(0)),
    ("null_coalesce", AtomicU64::new(0)),
    ("null_conditional", AtomicU64::new(0)),
    ("il2cpp_cleanup", AtomicU64::new(0)),
    ("equality_ops", AtomicU64::new(0)),
    ("algebraic_simplify", AtomicU64::new(0)),
    ("clamp_sugar", AtomicU64::new(0)),
    ("cast_elision", AtomicU64::new(0)),
    ("operator_overload", AtomicU64::new(0)),
    ("array_length", AtomicU64::new(0)),
    ("indexer", AtomicU64::new(0)),
    ("property_access", AtomicU64::new(0)),
    ("event_handler", AtomicU64::new(0)),
    ("pattern_match", AtomicU64::new(0)),
    ("array_init", AtomicU64::new(0)),
    ("list_inline", AtomicU64::new(0)),
    ("list_sugar", AtomicU64::new(0)),
    ("collection_init", AtomicU64::new(0)),
    ("object_init", AtomicU64::new(0)),
    ("base_call", AtomicU64::new(0)),
    ("delegate_ctor", AtomicU64::new(0)),
    ("using_lock", AtomicU64::new(0)),
    ("switch_recon", AtomicU64::new(0)),
    ("switch_expr", AtomicU64::new(0)),
    ("string_concat", AtomicU64::new(0)),
    ("string_interp", AtomicU64::new(0)),
    ("string_switch", AtomicU64::new(0)),
    ("state_machine", AtomicU64::new(0)),
    ("generic_inference", AtomicU64::new(0)),
    ("dup_update_fold", AtomicU64::new(0)),
    ("for_loop", AtomicU64::new(0)),
    ("foreach_recon", AtomicU64::new(0)),
    ("remove_continues", AtomicU64::new(0)),
    ("join_hoist", AtomicU64::new(0)),
    ("goto_simplify", AtomicU64::new(0)),
    ("goto_fold", AtomicU64::new(0)),
    ("join_hoist_final", AtomicU64::new(0)),
    ("goto_fallthrough", AtomicU64::new(0)),
    ("dead_tail", AtomicU64::new(0)),
    ("tuple_recon", AtomicU64::new(0)),
    ("linq_recon", AtomicU64::new(0)),
    ("range_index", AtomicU64::new(0)),
    ("merge_tail", AtomicU64::new(0)),
];
static RECON_TOTAL: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Method RVA for `ILR_TRACE` provenance lines.
    static TRACE_RVA: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub fn set_trace_rva(rva: u64) {
    TRACE_RVA.with(|c| c.set(rva));
}

fn trace_rva() -> u64 {
    TRACE_RVA.with(|c| c.get())
}

/// The thread's current method RVA, for trace call sites in other modules.
pub fn trace_rva_pub() -> u64 {
    trace_rva()
}

/// PASSES[idx] must line up with BY_PASS[idx]; a missing stats entry used to
/// panic only on the full corpus (index 40 into len 39). Keep this in sync at
/// compile time.
const _: () = assert!(PASSES.len() == BY_PASS.len());

/// Wall-clock nanos per reconstruct pass, parallel-summed over methods. The
/// pipeline runs all 46 passes on every rendered body (and again on each
/// preview), so a single slow pass is multiplied by ~24k methods. Printed by
/// `timing::report` under `ILR_TIMING=1`; findings drive same-output
/// optimizations (avoid redundant tree clones/walks, not pass removal).
static PASS_NANOS: [std::sync::atomic::AtomicU64; 46] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 46];

/// Dangling-label growth per pass (`ILR_LABELDBG=1`): after the pass ran, how
/// many more `goto` targets lacked a definition than before it. The pass that
/// owns the increase consumed a label a surviving goto still names.
static LABEL_LOST: [std::sync::atomic::AtomicU64; 46] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 46];

/// Top passes by labels they orphaned, for the `ILR_LABELDBG` report.
pub(crate) fn label_loss_report(top: usize) -> String {
    let mut rows: Vec<(&str, u64)> = PASSES
        .iter()
        .enumerate()
        .map(|(i, (name, _))| {
            (
                *name,
                LABEL_LOST[i].load(std::sync::atomic::Ordering::Relaxed),
            )
        })
        .filter(|(_, n)| *n > 0)
        .collect();
    let idiom = crate::recovery::idioms::idiom_label_loss();
    for (i, name) in crate::recovery::idioms::IDIOM_STEP_NAMES.iter().enumerate() {
        if idiom[i] > 0 {
            rows.push((name, idiom[i]));
        }
    }
    if rows.is_empty() {
        return String::new();
    }
    rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let total: u64 = rows.iter().map(|(_, n)| *n).sum();
    let mut out = format!("[label-loss] total={total}");
    for (name, n) in rows.into_iter().take(top) {
        out.push_str(&format!(" {name}={n}"));
    }
    out
}

/// Top passes by cumulative time, one line each, highest first.
pub(crate) fn pass_timing_report(top: usize) -> String {
    let mut rows: Vec<(&str, u64)> = PASSES
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, PASS_NANOS[i].load(std::sync::atomic::Ordering::Relaxed)))
        .collect();
    rows.sort_by_key(|(_, ns)| std::cmp::Reverse(*ns));
    let total: u64 = rows.iter().map(|(_, ns)| *ns).sum();
    let mut out = format!("[pass-time] total={:.0}ms", total as f64 / 1e6);
    for (name, ns) in rows.into_iter().take(top) {
        if ns == 0 {
            break;
        }
        out.push_str(&format!(" {name}={:.0}", ns as f64 / 1e6));
    }
    out
}
/// Enum member name replacement count (tracked separately since it runs in
/// the metadata-aware pipeline, not the standard PASSES array).
static ENUM_NAMES_REWRITES: AtomicU64 = AtomicU64::new(0);
/// C# attribute recovery count (tracked separately since it runs in the
/// metadata-aware pipeline, not the standard PASSES array).
static ATTRIBUTE_REWRITES: AtomicU64 = AtomicU64::new(0);
/// Lambda body inlining count (Phase 2D: cross-method lambda reconstruction).
static LAMBDA_INLINE_REWRITES: AtomicU64 = AtomicU64::new(0);
/// Closure folding count (Phase 2E: runs after lambda_inline to fold Expr::Lambda).
static CLOSURES_REWRITES: AtomicU64 = AtomicU64::new(0);

/// The run summary line, or an empty string when nothing fired.
///
/// ## Metrics dashboard
///
/// The report includes:
/// - Total rewrite count across all passes
/// - Per-pass breakdown (only passes with non-zero counts)
/// - Refusal explanations from passes that instrument them
/// - A summary of high-level constructs recovered
pub fn report() -> String {
    let total = RECON_TOTAL.load(Ordering::Relaxed);
    // `total == 0` no longer means "say nothing": a pass can now have a
    // non-empty explanation (sites examined, and why each was refused) even
    // when nothing fired, and that explanation is exactly what a silent zero
    // was hiding.
    let refusals = format!(
        "{}{}{}",
        using_lock::report_refusals(),
        for_loop::report_refusals(),
        null_coalesce_report()
    );
    if total == 0 && refusals.is_empty() {
        return String::new();
    }
    let mut s = format!("[reconstruct] {total} construct rewrite(s)\n");
    for (name, c) in BY_PASS.iter() {
        let v = c.load(Ordering::Relaxed);
        if v > 0 {
            s.push_str(&format!("[reconstruct]   {name}: {v}\n"));
        }
    }
    // A recognizer with a ZERO count is the interesting case: it is either
    // provably dead or refusing everything. Those need opposite responses, and
    // a bare zero cannot tell them apart, so the passes that can explain
    // themselves print their site population and refusal reasons here.
    s.push_str(&refusals);
    // Add the metrics dashboard summary.
    s.push_str(&metrics_summary());
    // Add Phase 8 type propagation report.
    s.push_str(&generic_inference::report());
    // Add vtable resolution report.
    s.push_str(&vtable_resolution::report());
    // Add field annotation report.
    s.push_str(&field_annotation::report());
    // Add the local-declaration report (where locals landed, and why the
    // rest could not be declared).
    s.push_str(&declare_locals::report());
    // Add heuristic type inference report.
    s.push_str(&super::heuristic_types::report());
    // Add enum member name recovery report.
    let enum_count = ENUM_NAMES_REWRITES.load(Ordering::Relaxed);
    if enum_count > 0 {
        s.push_str(&format!("[reconstruct]   enum_names: {enum_count}\n"));
    }
    // Add C# attribute recovery report.
    let attr_count = ATTRIBUTE_REWRITES.load(Ordering::Relaxed);
    if attr_count > 0 {
        s.push_str(&format!("[reconstruct]   attributes: {attr_count}\n"));
    }
    // Add lambda body inlining report (Phase 2D).
    s.push_str(&lambda_inline::report());
    // Add closure folding report (Phase 2E).
    let closures_count = CLOSURES_REWRITES.load(Ordering::Relaxed);
    if closures_count > 0 {
        s.push_str(&format!("[reconstruct]   closures: {closures_count}\n"));
    }
    s
}

/// Null-coalesce refusal report: always empty (the pass does not instrument
/// refusals — the pattern is simple enough that non-matches are uninteresting).
fn null_coalesce_report() -> String {
    String::new()
}

/// Generate a metrics dashboard summary showing the reconstruction capabilities.
///
/// This provides a high-level overview of what the reconstruction pipeline
/// can detect and recover, useful for understanding the decompiler's output
/// quality on a given corpus.
pub fn metrics_summary() -> String {
    let mut s = String::from("[reconstruct] --- metrics dashboard ---\n");

    // Categorize passes by what they recover.
    let structural = [
        ("for_loop", "for-loops"),
        ("using_lock", "using/lock"),
        ("switch_recon", "switch"),
        ("state_machine", "state machines"),
    ];
    let initializer = [
        ("array_init", "array initializers"),
        ("collection_init", "collection initializers"),
        ("object_init", "object initializers"),
    ];
    let expression = [
        ("null_coalesce", "null-coalescing assignment"),
        ("null_conditional", "null-conditional access"),
        ("nullability", "null comparisons"),
        ("closures", "lambdas/closures"),
        ("indexer", "collection indexers"),
        ("property_access", "property accessors"),
    ];
    let call_recovery = [
        ("base_call", "base calls"),
        ("delegate_ctor", "delegate constructions"),
    ];
    let infrastructure = [
        ("merge_fallthrough", "fall-through merges"),
        ("normalize_ctor_calls", "ctor normalizations"),
        ("pattern_match", "pattern match annotations"),
        ("dup_update_fold", "duplicate-update temps"),
    ];
    let il2cpp_de_weaving = [
        ("il2cpp_cleanup", "IL2CPP de-weaving"),
    ];
    let type_propagation = [
        ("generic_inference", "generic type inference"),
    ];

    let emit_category = |s: &mut String, label: &str, passes: &[(&str, &str)]| {
        let mut parts = Vec::new();
        for (pass_name, description) in passes {
            let count = BY_PASS
                .iter()
                .find(|(n, _)| n == pass_name)
                .map(|(_, c)| c.load(Ordering::Relaxed))
                .unwrap_or(0);
            if count > 0 {
                parts.push(format!("{description}: {count}"));
            }
        }
        if !parts.is_empty() {
            s.push_str(&format!(
                "[reconstruct]   {label}: {}\n",
                parts.join(", ")
            ));
        }
    };

    emit_category(&mut s, "structural", &structural);
    emit_category(&mut s, "initializers", &initializer);
    emit_category(&mut s, "expressions", &expression);
    emit_category(&mut s, "call recovery", &call_recovery);
    emit_category(&mut s, "infrastructure", &infrastructure);
    emit_category(&mut s, "IL2CPP de-weaving", &il2cpp_de_weaving);
    emit_category(&mut s, "type propagation", &type_propagation);

    // Total pass count.
    let active_passes = BY_PASS
        .iter()
        .filter(|(_, c)| c.load(Ordering::Relaxed) > 0)
        .count();
    s.push_str(&format!(
        "[reconstruct]   active passes: {}/{}\n",
        active_passes,
        BY_PASS.len()
    ));

    s
}

/// Generate a comprehensive reconstruction report for the entire run.
///
/// This is a more detailed version of [`report`] that includes additional
/// context about the reconstruction pipeline's capabilities and limitations.
/// Suitable for logging or diagnostic output.
pub fn detailed_report() -> String {
    let mut s = report();

    // Add information about passes that are known to be inert on the current
    // corpus shape (documented in their module docs).
    s.push_str("[reconstruct] --- pass status ---\n");
    s.push_str("[reconstruct]   auto_property: cross-method analysis (not in pass pipeline)\n");
    s.push_str("[reconstruct]   pattern_match: simplified annotation (full reconstruction deferred)\n");
    s.push_str("[reconstruct]   generic_inference: AST-level type inference (metadata-aware version available via apply_with_metadata)\n");
    s.push_str("[reconstruct]   vtable_resolution: metadata-aware virtual dispatch resolution via apply_with_maps (resolves vtable slots to method names)\n");
    s.push_str("[reconstruct]   string_token_restoration: AST annotation + metadata-aware resolution via resolve_strings_with_metadata\n");
    s.push_str("[reconstruct]   field_annotation: post-hoc RawDeref→Field resolution via apply_with_maps_and_types (resolves typed pointer derefs to named fields)\n");
    s.push_str("[reconstruct]   lambda_inline: Phase 2D cross-method lambda body inlining via apply_with_maps (rewrites delegate constructions to Expr::Lambda using global lambda body registry)\n");

    s
}

/// Constructor call normalization: converts ctor calls to `Expr::NewObj`.
///
/// The IL2CPP corpus never writes `new T(` — every allocation is a TWO-STATEMENT
/// pattern:
///
/// ```text
/// v = fn_XXXX(typeInfo, ...);          // allocation (RawCall or Call)
/// v..ctor(args);                        // constructor (Eval(Call { recv: v, name: ".ctor" }))
/// ```
///
/// This pass merges those two statements into a single `NewObj`:
///
/// ```text
/// v = new object(args);                 // Assign(v, _, NewObj("object", args))
/// ```
///
/// The type name defaults to `"object"` because the allocation call's metadata
/// does not carry the C# type name in a directly extractable form. Downstream
/// passes (object_init, collection_init) can still key on the `NewObj` shape.
///
/// Additionally, the old single-statement shape (`Call { name: "Type..ctor" }`)
/// is still recognized for forward compatibility.
///
/// Returns the count of rewritten calls.
pub fn normalize_ctor_calls(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    // Every name this method assigns, at any depth: pass 1b below turns a ctor
    // call on a temp into a `new`, and that is only sound for a temp that has
    // no definition anywhere (see the comment there).
    let mut defined: HashSet<String> = HashSet::new();
    collect_assigned_deep(stmts, &mut defined);
    walk_mut(stmts, &mut |list| {
        // Pass 1: Merge two-statement allocation patterns.
        // Scan for consecutive pairs: Alloc(v) + Eval(.ctor on v).
        let mut i = 0;
        while i + 1 < list.len() {
            // Check if list[i] is an allocation: Assign(v, _, RawCall/Call/NewObj)
            //
            // The `NewObj` arm is the renderer's own allocation statement (see
            // `boundary_stmts_inner`). It carries the REAL allocated type, so the
            // merge below keeps `new Dictionary<int, Token>()` even when the ctor
            // name cannot be decoded.
            let alloc_info = match &list[i] {
                Stmt::Assign(v, _, Expr::RawCall(..)) => Some((v.clone(), None)),
                Stmt::Assign(v, _, Expr::Call { .. }) => Some((v.clone(), None)),
                Stmt::Assign(v, _, Expr::NewObj(ty, _)) => Some((v.clone(), Some(ty.clone()))),
                _ => None,
            };

            if let Some((ref var_name, ref alloc_ty)) = alloc_info {
                // Check if list[i+1] is Eval(Call { recv: Var(var_name), name: ".ctor", args })
                let ctor_args = match &list[i + 1] {
                    Stmt::Eval(Expr::Call { recv, name, args })
                        if (name == ".ctor" || name.ends_with(".ctor") || name.ends_with("..ctor")) =>
                    {
                        if let Some(Expr::Var(recv_name)) = recv.as_deref() {
                            if recv_name == var_name {
                                Some(args.clone())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(args) = ctor_args {
                    // Match! Merge into NewObj.
                    // Extract type name from the ctor name if it has a prefix.
                    // The ctor NAME carries the generic instantiation the
                    // allocation operand does not (`Dictionary`2` vs
                    // `Dictionary<object, object>..ctor`), so it wins when the
                    // allocation type is only the open definition.
                    let ctor_ty = match &list[i + 1] {
                        Stmt::Eval(Expr::Call { name, .. }) => {
                            ctor_type_name(name).map(|s| s.to_string())
                        }
                        _ => None,
                    };
                    let type_name = match (alloc_ty.clone(), ctor_ty) {
                        (Some(a), Some(c)) if !a.contains('<') && c.contains('<') => c,
                        (Some(a), _) => a,
                        (None, Some(c)) => c,
                        (None, None) => "object".to_string(),
                    };
                    // Replace list[i] with Assign(var_name, None, NewObj(type_name, args)).
                    list[i] = Stmt::Assign(
                        var_name.clone(),
                        None,
                        Expr::NewObj(type_name, args),
                    );
                    // Remove the ctor statement.
                    list.remove(i + 1);
                    count += 1;
                    // Don't advance i — check next pair.
                    continue;
                }
            }
            i += 1;
        }

        // Pass 1b: a ctor call on a temp the text never allocated.
        //
        // IL2CPP compiles `new T(args)` as `alloc; T..ctor(obj, args)`. When the
        // temp's only reader is the ctor's receiver, the lifter drops the
        // allocation statement (no bare read means "the use inlines the def"),
        // leaving the ctor call plus the store of the temp:
        //
        //     v271..ctor();   this._X = v271;
        //
        // `v271` has no definition anywhere in the method, so it can only be the
        // fresh object — the pair IS the construction:
        //
        //     this._X = new Dictionary<int, object>();
        //
        // A temp that IS defined is refused: its allocation is already spelled
        // elsewhere, and a second `new` would fabricate an allocation the
        // native code does not perform (see pass 2).
        let mut i = 0;
        while i + 1 < list.len() {
            let pair = match (&list[i], &list[i + 1]) {
                (
                    Stmt::Eval(Expr::Call {
                        recv: Some(r),
                        name,
                        args,
                    }),
                    Stmt::Assign(target, None, Expr::Var(src)),
                ) if is_ctor_name(name) => match (r.as_ref(), ctor_type_name(name)) {
                    (Expr::Var(rn), Some(ty))
                        if rn == src
                            && rn != target
                            && !defined.contains(rn)
                            && balanced_type(ty) =>
                    {
                        Some((target.clone(), ty.to_string(), args.clone()))
                    }
                    _ => None,
                },
                _ => None,
            };
            let Some((target, ty, args)) = pair else {
                i += 1;
                continue;
            };
            list[i] = Stmt::Assign(target, None, Expr::NewObj(ty, args));
            list.remove(i + 1);
            count += 1;
        }

        // Pass 2: Handle old single-statement ctor calls (Type..ctor).
        //
        // A ctor call WITH A RECEIVER is not an allocation. IL2CPP compiles
        // `new T(args)` as `alloc; T..ctor(obj, args)`, and the constructor
        // returns `this`, so an assignment `v = obj..ctor(args)` observes the
        // SAME object. Rendering it as `v = new T(args)` fabricated a SECOND
        // allocation (0x18A5338: two `new RewardData(...)` for one native
        // `bl alloc`). Alias the receiver instead; the single allocation is
        // already represented where `obj` got its value.
        for stmt in list.iter_mut() {
            match stmt {
                Stmt::Assign(_, _, expr) => {
                    let replacement: Option<Expr> = match &*expr {
                        Expr::Call {
                            recv: Some(r),
                            name,
                            ..
                        } if is_ctor_name(name) => match r.as_ref() {
                            Expr::Var(rn) => Some(Expr::Var(rn.clone())),
                            _ => None,
                        },
                        Expr::Call {
                            recv: None,
                            name,
                            args,
                        } if is_ctor_name(name) => ctor_type_name(name)
                            .map(|t| Expr::NewObj(t.to_string(), args.clone())),
                        _ => None,
                    };
                    if let Some(r) = replacement {
                        *expr = r;
                        count += 1;
                    }
                }
                Stmt::Eval(expr) => {
                    let replacement: Option<Expr> = match &*expr {
                        Expr::Call {
                            recv: None,
                            name,
                            args,
                        } if is_ctor_name(name) => ctor_type_name(name)
                            .map(|t| Expr::NewObj(t.to_string(), args.clone())),
                        _ => None,
                    };
                    if let Some(r) = replacement {
                        *expr = r;
                        count += 1;
                    }
                }
                _ => {}
            }
        }
    });
    count
}

// ---------------------------------------------------------------------------
// Fall-through list merging (pre-pass for same-list pairing recognizers)
// ---------------------------------------------------------------------------

/// Does this statement list end with a control-flow terminator?
///
/// A terminator is a statement after which no sequential execution reaches the
/// next statement in the list: `return`, `throw`, `goto`, `break`, `continue`.
/// An empty list does NOT end with a terminator (it falls through).
fn ends_with_terminator(stmts: &[Stmt]) -> bool {
    match stmts.last() {
        Some(Stmt::Return(_))
        | Some(Stmt::Throw(_))
        | Some(Stmt::Goto(_))
        | Some(Stmt::Break)
        | Some(Stmt::Continue) => true,
        _ => false,
    }
}

/// The fall-through merging pre-pass.
///
/// ## Why this exists
///
/// The structurer emits unrecoverable control-flow regions as separately-
/// labelled blocks. When a `Monitor.Enter(x, 0)` lands in one child list (e.g.
/// the `then_body` of an `if`) and the matching `Monitor.Exit(x)` lands in the
/// parent list after the `if`, the existing same-list pairing logic in
/// `using_lock` cannot see both events — they are in different `Vec<Stmt>`s.
///
/// ## What it does
///
/// Walks the statement tree. At each level, when a nested statement has child
/// lists, checks whether exactly one child list ends without a terminator while
/// every other child list ends with one. If the next statement in the parent
/// list (immediately after the nested statement) is a `Label`, the labelled
/// block's statements are spliced from the parent list into the falling-through
/// child list. The label is consumed (removed) as part of the splice.
///
/// ## Why the merge is safe
///
/// When all other child lists end with a terminator, no other branch can reach
/// the post-label code. The only path that reaches it is the fall-through from
/// the one non-terminating child list. Moving those statements into that child
/// list preserves the set of reachable executions exactly: the post-label code
/// was only reachable from that child's fall-through, and it remains so after
/// the merge. The label is a no-op (it was only needed as a jump target for the
/// now-eliminated boundary) and is consumed.
///
/// ## What it does NOT merge
///
/// - Two or more child lists that both end without a terminator: the post-label
///   code is reachable from multiple paths, so moving it into one would lose
///   the others.
/// - A child list that already ends with a terminator: there is no fall-through
///   to merge.
/// - Nested structures inside loops (`while`, `do_while`, `for`, `for_each`):
///   the loop body's exit goes back to the loop header, not to the post-loop
///   code, so the body does not "fall through" in the relevant sense.
pub fn merge_fallthrough_lists(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0usize;
    merge_fallthrough_walk(stmts, &mut count);
    count
}

/// Recursive worker: post-order walk that first descends into children, then
/// merges at the current level. The post-order ensures inner merges happen
/// before outer ones, so a chain of nested fall-throughs is fully collapsed.
fn merge_fallthrough_walk(stmts: &mut Vec<Stmt>, count: &mut usize) {
    // First, recurse into every child list so inner merges happen first.
    for s in stmts.iter_mut() {
        for child in children_mut(s) {
            merge_fallthrough_walk(child, count);
        }
    }
    // Now try to merge at this level: find a nested statement whose child list
    // falls through, while all other child lists terminate, and which is
    // immediately followed by a label in this list.
    //
    // We scan for the first eligible site, perform the merge, and restart the
    // scan (the list has changed). This handles chains of eligible sites.
    loop {
        let merge = find_fallthrough_merge(stmts);
        match merge {
            Some(m) => {
                apply_fallthrough_merge(stmts, m);
                *count += 1;
            }
            None => break,
        }
    }
}

/// Description of a pending fall-through merge.
struct FallthroughMerge {
    /// Index of the nested statement in the parent list.
    stmt_idx: usize,
    /// Index of the child list within the nested statement that falls through.
    child_idx: usize,
    /// Index of the label statement in the parent list (stmt_idx + 1).
    label_idx: usize,
}

/// Find the first eligible fall-through merge in `stmts`, or `None`.
fn find_fallthrough_merge(stmts: &[Stmt]) -> Option<FallthroughMerge> {
    for i in 0..stmts.len() {
        // The label must immediately follow the nested statement.
        let label_idx = i + 1;
        if label_idx >= stmts.len() {
            continue;
        }
        if !matches!(&stmts[label_idx], Stmt::Label(_)) {
            continue;
        }
        // Check if stmts[i] is a nested structure with eligible child lists.
        if let Some(child_idx) = eligible_fallthrough_child(&stmts[i]) {
            return Some(FallthroughMerge {
                stmt_idx: i,
                child_idx,
                label_idx,
            });
        }
    }
    None
}

/// If `s` is a nested structure where exactly one child list ends without a
/// terminator and all other child lists end with one, return the index of the
/// falling-through child. Otherwise return `None`.
///
/// Loop bodies (`while`, `do_while`, `for`, `for_each`) are excluded: the body
/// loops back to the header, so it does not "fall through" to the post-loop
/// code in the relevant sense.
fn eligible_fallthrough_child(s: &Stmt) -> Option<usize> {
    match s {
        // Loops: the body loops, not falls through.
        Stmt::While { .. }
        | Stmt::DoWhile { .. }
        | Stmt::For { .. }
        | Stmt::ForEach { .. } => None,
        // If: then_body (idx 0) and else_body (idx 1).
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            let then_term = ends_with_terminator(then_body);
            let else_term = ends_with_terminator(else_body);
            match (then_term, else_term) {
                (false, true) => Some(0), // then_body falls through
                (true, false) => Some(1), // else_body falls through
                _ => None,                // both fall through or both terminate
            }
        }
        // Using/Lock: body (idx 0). The body is scoped; after it exits, the
        // post-scope code runs. If the body ends without a terminator, it
        // falls through to the post-scope code.
        Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
            if ends_with_terminator(body) {
                None
            } else {
                Some(0)
            }
        }
        // Try: try_body (idx 0), catches (idx 1..n), finally_body (idx n+1).
        // The try_body falls through to the post-try code only when there are
        // no catches and no finally (a bare try is unusual). With catches or
        // finally, the control flow is more complex.
        // For safety, only merge when there's exactly one child that falls
        // through and all others terminate.
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut children: Vec<&Vec<Stmt>> = vec![try_body];
            for (_, b) in catches {
                children.push(b);
            }
            if let Some(fb) = finally_body {
                children.push(fb);
            }
            let mut falling = None;
            for (idx, c) in children.iter().enumerate() {
                if !ends_with_terminator(c) {
                    if falling.is_some() {
                        return None; // multiple falling-through children
                    }
                    falling = Some(idx);
                }
            }
            falling
        }
        // Switch: cases (idx 0..n) and tail (idx n).
        Stmt::Switch { cases, tail, default, .. } => {
            let mut children: Vec<&Vec<Stmt>> = Vec::new();
            for (_, b) in cases {
                children.push(b);
            }
            children.push(tail);
            if let Some(arm) = default {
                children.push(arm);
            }
            let mut falling = None;
            for (idx, c) in children.iter().enumerate() {
                if !ends_with_terminator(c) {
                    if falling.is_some() {
                        return None; // multiple falling-through children
                    }
                    falling = Some(idx);
                }
            }
            falling
        }
        // StringSwitch: cases (idx 0..n) and optional tail (idx n).
        Stmt::StringSwitch { cases, tail, .. } => {
            let mut children: Vec<&Vec<Stmt>> = Vec::new();
            for (_, b) in cases {
                children.push(b);
            }
            if let Some(t) = tail {
                children.push(t);
            }
            let mut falling = None;
            for (idx, c) in children.iter().enumerate() {
                if !ends_with_terminator(c) {
                    if falling.is_some() {
                        return None;
                    }
                    falling = Some(idx);
                }
            }
            falling
        }
        _ => None,
    }
}

/// Get a mutable reference to the child list at `child_idx` of statement `s`.
/// The indexing matches `eligible_fallthrough_child`.
fn get_child_mut<'a>(s: &'a mut Stmt, child_idx: usize) -> &'a mut Vec<Stmt> {
    match s {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => match child_idx {
            0 => then_body,
            1 => else_body,
            _ => unreachable!(),
        },
        Stmt::Using { body, .. } | Stmt::Lock { body, .. } => {
            assert_eq!(child_idx, 0);
            body
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut idx = 0;
            if child_idx == idx {
                return try_body;
            }
            idx += 1;
            for (_, b) in catches.iter_mut() {
                if child_idx == idx {
                    return b;
                }
                idx += 1;
            }
            if let Some(fb) = finally_body {
                if child_idx == idx {
                    return fb;
                }
            }
            unreachable!()
        }
        Stmt::Switch { cases, tail, default, .. } => {
            let mut idx = 0;
            for (_, b) in cases.iter_mut() {
                if child_idx == idx {
                    return b;
                }
                idx += 1;
            }
            if child_idx == idx {
                return tail;
            }
            idx += 1;
            if let Some(arm) = default {
                if child_idx == idx {
                    return arm;
                }
                idx += 1;
            }
            assert_eq!(child_idx, idx);
            tail
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            let mut idx = 0;
            for (_, b) in cases.iter_mut() {
                if child_idx == idx {
                    return b;
                }
                idx += 1;
            }
            // The tail (if present) is the last child.
            if let Some(t) = tail {
                assert_eq!(child_idx, idx);
                return t;
            }
            unreachable!()
        }
        _ => unreachable!(),
    }
}

/// Apply the merge: splice the post-label statements from the parent list into
/// the falling-through child list, and remove the label + spliced statements
/// from the parent list.
fn apply_fallthrough_merge(stmts: &mut Vec<Stmt>, m: FallthroughMerge) {
    // Move the LABEL together with its tail. The fall-through path does not
    // need the label, but OTHER gotos may target it: the old code dropped the
    // label unconditionally and left every such jump dangling (measured:
    // UpdateMaterial @0x1982AB4, where the label was the merge of two arms and
    // the whole diffuse-lit branch was then pruned as unreachable). An unused
    // moved label is removed later by the label sweeps.
    let tail: Vec<Stmt> = stmts[m.label_idx..].to_vec();
    // Remove the label and tail from the parent list.
    stmts.truncate(m.label_idx);
    // Append the label + tail to the falling-through child list.
    let child = get_child_mut(&mut stmts[m.stmt_idx], m.child_idx);
    child.extend(tail);
}

// ---------------------------------------------------------------------------
// Shared AST helpers (the recognizers must not each reinvent these)
// ---------------------------------------------------------------------------

/// Post-order walk: recurse into every nested statement list, then hand the
/// list to `f`. Recognizers call this so a construct nested inside an `if`,
/// a loop or a `try` is reached.
pub fn walk_mut<F: FnMut(&mut Vec<Stmt>)>(stmts: &mut Vec<Stmt>, f: &mut F) {
    for s in stmts.iter_mut() {
        for c in children_mut(s) {
            walk_mut(c, f);
        }
    }
    f(stmts);
}

/// Replace the structurer's coverage-repair marker with what survived.
///
/// `coverage_repair` runs before `branch_repair`/`goto_fold`; most flat
/// regions it emits are folded back into structure a few passes later and
/// their statements pruned. The comment written at repair time therefore
/// described a transient tree (measured on 0x1951D30: 190 regions claimed,
/// 8 present after cleanup). Every region that carries code is tagged with a
/// sentinel right after its label; this pass removes the sentinels, counts
/// the survivors, and rewrites or drops the marker. Called at the last AST
/// mutation point, so the number is what the reader can actually see.
pub fn finalize_recovery_markers(stmts: &mut Vec<Stmt>) {
    // The marker is a snapshot taken at repair time. It also SHIELDS the
    // whole repair tail from `idioms::prune_dead_code` (that sweep splits the
    // list at the marker to preserve cold pads). By this point the late
    // folds have removed most repair bodies; leaving the shield in place
    // kept dead tails (`goto L25;` eleven times on 0x1947380). Drop stale
    // markers first, collapse dead terminator runs, then report only what
    // survived. A full dead-code sweep is deliberately not run (cold pads
    // have no goto entry).
    walk_mut(stmts, &mut |list| {
        list.retain(|s| {
            !matches!(s, Stmt::Comment(c) if c.starts_with("[recovery] emitted"))
        });
    });
    // Do NOT run the full dead-code sweep here: a cold pad has no `goto`
    // entry (the unwinder is its predecessor), so after label cleanup its
    // labelless payload would be deleted. What the shield DID leave behind is
    // runs of dead terminators (`goto L25;` eleven times on 0x1947380) whose
    // real bodies the late folds removed. Collapse those only.
    walk_mut(stmts, &mut |list| {
        let mut dead = false;
        list.retain(|s| match s {
            Stmt::Label(_) => {
                dead = false;
                true
            }
            Stmt::Comment(_) | Stmt::Attribute { .. } => true,
            Stmt::Goto(_) | Stmt::Break | Stmt::Continue | Stmt::Return(_) | Stmt::Throw(_) => {
                if dead {
                    false
                } else {
                    dead = true;
                    true
                }
            }
            _ => {
                dead = false;
                true
            }
        });
    });
    let mut entry = 0usize;
    let mut cold = 0usize;
    // A sentinel marks the START of a repair region (right after its label).
    // Late passes routinely fold the region's body away while leaving the
    // sentinel and label behind (0x1951D30 ended with four consecutive
    // sentinels and one shared `return`). A region counts only when real,
    // non-terminator code still follows it before the next label: an alias
    // (`label: goto X`), an empty label, `break`/`continue`, or a shared
    // epilogue is not lost code.
    let mut anchor: Option<String> = None;
    walk_mut(stmts, &mut |list| {
        let mut pending: Vec<(bool, Option<String>)> = Vec::new();
        let mut last_label: Option<String> = None;
        let mut kept: Vec<Stmt> = Vec::with_capacity(list.len());
        for s in list.drain(..) {
            match &s {
                Stmt::Comment(c) if c.starts_with("[recovery-region]") => {
                    pending.push((c.contains("entry"), last_label.clone()));
                    continue;
                }
                Stmt::Label(l) => {
                    pending.clear();
                    last_label = Some(l.clone());
                }
                Stmt::Comment(_) => {}
                Stmt::Goto(_)
                | Stmt::Break
                | Stmt::Continue
                | Stmt::Return(_)
                | Stmt::Throw(_) => pending.clear(),
                _ => {
                    for (is_entry, label) in pending.drain(..) {
                        // A sentinel whose label was folded away marks code
                        // that the late passes bound into the structured body
                        // (join_hoist moves the tail and drops the label), or
                        // an alias that never needed one. It is not a
                        // labelled repair region any more.
                        if label.is_none() {
                            continue;
                        }
                        if is_entry {
                            entry += 1;
                        } else {
                            cold += 1;
                        }
                        if anchor.is_none() {
                            anchor = label;
                        }
                    }
                }
            }
            kept.push(s);
        }
        *list = kept;
    });
    let total = entry + cold;
    if total > 0 {
        let marker = Stmt::Comment(format!(
            "[recovery] emitted {total} labelled region(s): {cold} out-of-line cold, \
             {entry} entry-reachable region(s) not structured"
        ));
        let mut inserted = false;
        if let Some(a) = &anchor {
            walk_mut(stmts, &mut |list| {
                if inserted {
                    return;
                }
                if let Some(i) = list
                    .iter()
                    .position(|s| matches!(s, Stmt::Label(l) if l == a))
                {
                    list.insert(i + 1, marker.clone());
                    inserted = true;
                }
            });
        }
    }
}

/// Every nested statement list a statement owns, mutably.
pub fn children_mut(s: &mut Stmt) -> Vec<&mut Vec<Stmt>> {
    match s {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => vec![then_body, else_body],
        Stmt::While { body, .. } => vec![body],
        Stmt::DoWhile { body, .. } => vec![body],
        Stmt::For {
            init, incr, body, ..
        } => vec![init, incr, body],
        Stmt::Using { body, .. } => vec![body],
        Stmt::Lock { body, .. } => vec![body],
        Stmt::ForEach { body, .. } => vec![body],
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut v: Vec<&mut Vec<Stmt>> = vec![try_body];
            for (_, b) in catches.iter_mut() {
                v.push(b);
            }
            if let Some(fb) = finally_body {
                v.push(fb);
            }
            v
        }
        Stmt::Switch { cases, tail, default, .. } => {
            let mut v: Vec<&mut Vec<Stmt>> = Vec::new();
            for (_, b) in cases.iter_mut() {
                v.push(b);
            }
            v.push(tail);
            if let Some(arm) = default {
                v.push(arm);
            }
            v
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            let mut v: Vec<&mut Vec<Stmt>> = Vec::new();
            for (_, b) in cases.iter_mut() {
                v.push(b);
            }
            if let Some(t) = tail {
                v.push(t);
            }
            v
        }
        _ => Vec::new(),
    }
}

/// Every nested statement list a statement owns, immutably.
pub fn children<'a>(s: &'a Stmt, out: &mut Vec<&'a Vec<Stmt>>) {
    match s {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            out.push(then_body);
            out.push(else_body);
        }
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::Using { body, .. }
        | Stmt::Lock { body, .. }
        | Stmt::ForEach { body, .. } => out.push(body),
        Stmt::For {
            init, incr, body, ..
        } => {
            out.push(init);
            out.push(incr);
            out.push(body);
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            out.push(try_body);
            for (_, b) in catches {
                out.push(b);
            }
            if let Some(fb) = finally_body {
                out.push(fb);
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases {
                out.push(b);
            }
            out.push(tail);
            if let Some(arm) = default {
                out.push(arm);
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, b) in cases {
                out.push(b);
            }
            if let Some(t) = tail {
                out.push(t);
            }
        }
        _ => {}
    }
}

/// Does evaluating `e` call anything (a call, a raw call, an allocation, an
/// await)? A write-carrying expression — the test for "nothing observable can
/// happen between these two positions".
pub fn expr_has_call(e: &Expr) -> bool {
    match e {
        // These nodes are effectful in their own right, even when every input
        // is a plain local. Previously this helper only inspected their
        // children, so `Foo()` / `new T()` / `await task` were incorrectly
        // classified as pure. That let motion-sensitive recognizers move
        // initializers across observable work.
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
            use super::csharp::InterpPart;
            parts.iter().any(|p| match p {
                InterpPart::Literal(_) => false,
                InterpPart::Expr(e, _) => expr_has_call(e),
            })
        }
        Expr::Tuple { elements, .. } => elements.iter().any(expr_has_call),
        Expr::MethodChain { receiver, methods } => {
            expr_has_call(receiver)
                || methods.iter().any(|m| m.args.iter().any(|a| expr_has_call(a)))
        }
        Expr::SwitchExpr { discrim, arms, default_arm } => {
            expr_has_call(discrim)
                || arms.iter().any(|(p, r)| expr_has_call(p) || expr_has_call(r))
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

/// Does `e` mention the local `name` anywhere?
///
/// Deliberately OVER-approximate: an `Expr::Field(recv, f)` also reports a
/// mention when the field's own name equals `name`. A false mention can only
/// make a recognizer refuse (safe); a missed mention could make it rewrite
/// (unsafe), so the direction of the approximation is chosen on purpose.
pub fn expr_mentions(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Var(n) => n == name,
        Expr::Call {
            recv,
            name: cn,
            args,
        } => {
            cn == name
                || recv.as_deref().is_some_and(|r| expr_mentions(r, name))
                || args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::ArrayInit { items, .. } => items.iter().any(|a| expr_mentions(a, name)),
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            ctor_args.iter().any(|a| expr_mentions(a, name))
                || items.iter().any(|a| expr_mentions(a, name))
        }
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            ctor_args.iter().any(|a| expr_mentions(a, name))
                || fields.iter().any(|(_, v)| expr_mentions(v, name))
        }
        Expr::Await(x) => expr_mentions(x, name),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_mentions(a, name) || expr_mentions(b, name),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) => {
            expr_mentions(x, name)
        }
        Expr::Field(r, f) => f == name || expr_mentions(r, name),
        Expr::StaticField(_, f) => f == name,
        Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => expr_mentions(r, name),
        Expr::Cond(c, a, b) => {
            expr_mentions(c, name) || expr_mentions(a, name) || expr_mentions(b, name)
        }
        Expr::Lambda { params, body } => {
            !params.iter().any(|p| p.as_str() == name) && expr_mentions(body, name)
        }
        Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::Unknown
        | Expr::MissingAbiArg
        | Expr::MethodGroup(_) => false,
        Expr::Index { collection, index } => {
            expr_mentions(collection, name) || expr_mentions(index, name)
        }
        Expr::NullConditional { receiver, access } => {
            expr_mentions(receiver, name) || expr_mentions(access, name)
        }
        Expr::PatternMatch { expr, binding: _, .. } => {
            // The binding variable is a new declaration, not a reference to
            // an outer `name`. Only the tested expression can mention `name`.
            expr_mentions(expr, name)
        }
        Expr::StringInterp { parts } => {
            use super::csharp::InterpPart;
            parts.iter().any(|p| match p {
                InterpPart::Literal(_) => false,
                InterpPart::Expr(e, _) => expr_mentions(e, name),
            })
        }
        Expr::Tuple { elements, .. } => elements.iter().any(|e| expr_mentions(e, name)),
        Expr::MethodChain { receiver, methods } => {
            expr_mentions(receiver, name)
                || methods.iter().any(|m| m.args.iter().any(|a| expr_mentions(a, name)))
        }
        Expr::SwitchExpr { discrim, arms, default_arm } => {
            expr_mentions(discrim, name)
                || arms.iter().any(|(p, r)| expr_mentions(p, name) || expr_mentions(r, name))
                || default_arm.as_deref().is_some_and(|e| expr_mentions(e, name))
        }
        Expr::Range { from, to } => {
            from.as_ref().is_some_and(|f| expr_mentions(f, name))
                || to.as_ref().is_some_and(|t| expr_mentions(t, name))
        }
        Expr::IndexFromEnd(e) => expr_mentions(e, name),
        Expr::TypeOf(_) => false,
    }
}

/// Does an assignment TARGET name use `name`?
///
/// The target of `v376.localRotation = 0` is not a variable — it is a path that
/// STARTS at one, so `v376` has to keep existing. The same holds for raw
/// pointer targets like `*(byte*)((nint)&stack_1 +4)`, whose whole text is
/// built from a name. A target that is exactly `name` is the definition, not a
/// use, so it is excluded.
pub fn target_names(lhs: &str, name: &str) -> bool {
    lhs != name
        && lhs
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .any(|t| t == name)
}

/// Does this statement read `name`? Shallow: the statement's own expressions,
/// not the nested bodies. Used to prove a value is live-out.
pub fn stmt_reads(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => expr_mentions(e, name),
        Stmt::Assign(lhs, _, e) => expr_mentions(e, name) || target_names(lhs, name),
        Stmt::If { cond, .. } => expr_mentions(cond, name),
        Stmt::While { cond, .. } => expr_mentions(cond, name),
        Stmt::DoWhile { cond, .. } => expr_mentions(cond, name),
        Stmt::For { cond, .. } => cond.as_ref().is_some_and(|c| expr_mentions(c, name)),
        Stmt::Lock { expr, .. } => expr_mentions(expr, name),
        Stmt::Using { decl, .. } => stmt_reads(decl, name),
        Stmt::Return(Some(e)) => expr_mentions(e, name),
        Stmt::StringSwitch { disc, .. } => expr_mentions(disc, name),
        _ => false,
    }
}

/// Does this statement write `name` (an assignment target, a `for` header, or a
/// `using` declaration)?
pub fn stmt_writes(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Assign(n, _, _) => n == name,
        _ => false,
    }
}

/// Does this statement read `name`, ANYWHERE in its nested bodies?
pub fn stmt_reads_deep(s: &Stmt, name: &str) -> bool {
    if stmt_reads(s, name) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    kids.iter().any(|k| list_reads_deep(k, name))
}

/// Does any statement in `list` read `name`, at any depth?
pub fn list_reads_deep(list: &[Stmt], name: &str) -> bool {
    list.iter().any(|s| stmt_reads_deep(s, name))
}

/// Does this statement read OR write `name`, anywhere in its nested bodies?
///
/// This is the conservative liveness test: an assignment to `name` counts as a
/// mention (the name is live there), a read counts, both at any depth.
pub fn stmt_mentions_deep(s: &Stmt, name: &str) -> bool {
    if stmt_writes(s, name) || stmt_reads(s, name) {
        return true;
    }
    // `Stmt::Switch`'s discriminant is a bare name string, not an expression —
    // a switch on the local being tested is a mention.
    if let Stmt::Switch { disc, .. } = s {
        if disc == name {
            return true;
        }
    }
    // `Stmt::ForEach` names its element local as a string.
    if let Stmt::ForEach {
        var_name,
        collection,
        ..
    } = s
    {
        if var_name == name || collection == name {
            return true;
        }
    }
    // `Stmt::For`/`Using` carry statement lists that may assign the name.
    if let Stmt::For { init, incr, .. } = s {
        if list_mentions_deep(init, name) || list_mentions_deep(incr, name) {
            return true;
        }
    }
    if let Stmt::Using { decl, .. } = s {
        if stmt_mentions_deep(decl, name) {
            return true;
        }
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    kids.iter().any(|k| list_mentions_deep(k, name))
}

/// Does any statement in `list` read or write `name`, at any depth?
pub fn list_mentions_deep(list: &[Stmt], name: &str) -> bool {
    list.iter().any(|s| stmt_mentions_deep(s, name))
}

/// Does this statement write `name`, ANYWHERE in its nested bodies?
///
/// Unlike [`stmt_writes`] (shallow: only the statement's own assignment target),
/// this recurses into nested bodies — a `for`/`using` header assignment or an
/// assignment inside an `if` body counts. Used when the question is "does this
/// region change the value of `name`?" rather than "is this statement itself an
/// assignment to `name`?".
pub fn stmt_writes_deep(s: &Stmt, name: &str) -> bool {
    if stmt_writes(s, name) {
        return true;
    }
    // `Stmt::For`/`Using` carry header statement lists that may assign the name.
    if let Stmt::For { init, incr, .. } = s {
        if list_writes_deep(init, name) || list_writes_deep(incr, name) {
            return true;
        }
    }
    if let Stmt::Using { decl, .. } = s {
        if stmt_writes_deep(decl, name) {
            return true;
        }
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    kids.iter().any(|k| list_writes_deep(k, name))
}

/// Does any statement in `list` write `name`, at any depth?
pub fn list_writes_deep(list: &[Stmt], name: &str) -> bool {
    list.iter().any(|s| stmt_writes_deep(s, name))
}

/// The `(receiver, name, args)` of a call expression, or `None`.
pub fn split_call(e: &Expr) -> Option<(Option<&Expr>, &str, &[Expr])> {
    match e {
        Expr::Call { recv, name, args } => Some((recv.as_deref(), name.as_str(), args.as_slice())),
        _ => None,
    }
}

/// Is `name` a constructor's name? The renderer spells a construction
/// `Type..ctor(args)` (name `"Type..ctor"`) or, when the declaration is
/// unresolved, `recv..ctor(args)` (name `".ctor"`). Measured on the x64 corpus:
/// **`new T(` appears ZERO times** — every allocation in the real output is a
/// `.ctor` CALL, so `Expr::NewObj` is not the shape to key on.
pub fn is_ctor_name(name: &str) -> bool {
    name == ".ctor" || name.ends_with(".ctor") || name.ends_with("..ctor")
}

/// The type name a construction names, when it is spelled on the callee
/// (`"System.Action..ctor"` → `Some("System.Action")`). `recv..ctor` carries no
/// type here and yields `None` — the receiver's type is not in the tree.
/// Every name assigned anywhere in `list`, at any depth.
fn collect_assigned_deep(list: &[Stmt], out: &mut HashSet<String>) {
    for s in list {
        if let Stmt::Assign(target, _, _) = s {
            out.insert(target.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for k in kids {
            collect_assigned_deep(k, out);
        }
    }
}

/// A type name that can be written in `new T(…)`.
///
/// A generic instantiation recovered from a shared body can lose its opening
/// angle bracket (`Token>`), and `new Token>()` is not C#.
fn balanced_type(t: &str) -> bool {
    let mut depth = 0i32;
    for c in t.chars() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0 && !t.trim().is_empty()
}

pub fn ctor_type_name(name: &str) -> Option<&str> {
    let t = name
        .strip_suffix("..ctor")
        .or_else(|| name.strip_suffix(".ctor"))?;
    if t.is_empty() { None } else { Some(t) }
}

/// Structural expression equality — compares variant discriminants and
/// sub-expressions directly without rendering. This is O(n) in the size of the
/// expression tree with no allocations, and is correct by construction (unlike
/// render-based comparison, which risks collisions if the renderer is not
/// injective).
pub fn exprs_eq(a: &Expr, b: &Expr) -> bool {
    a == b
}

/// The rendered text of an expression — the key used wherever a name must be
/// compared textually (`Stmt::Assign` targets and `Stmt::Switch` discriminants
/// are plain strings, not expressions).
pub fn render_text(e: &Expr) -> String {
    let mut s = String::new();
    e.render(&mut s);
    s
}

/// Can this statement be rendered in a `for` header / `using` declaration?
///
/// **This is a data-loss gate, not a style choice.** `Renderer::stmt` for
/// `Stmt::For` builds the header with `filter_map(stmt_inline)`, so ANY init or
/// increment element without an inline form is silently DROPPED from the
/// emitted text (`Stmt::Using` renders `using ()` for the same reason). A
/// recognizer that put, say, a `Stmt::If` into `incr` would delete it. Only
/// `Assign` and `Eval` have an inline form; every recognizer that fills a header
/// must check this first.
pub fn is_inlinable(s: &Stmt) -> bool {
    matches!(s, Stmt::Assign(..) | Stmt::Eval(..))
}

/// Is `s` a `continue` AT THIS LOOP LEVEL? Nested loops own their own
/// `continue`s, so their bodies are not descended into.
pub fn contains_continue(list: &[Stmt]) -> bool {
    list.iter().any(|s| match s {
        Stmt::Continue => true,
        Stmt::If {
            then_body,
            else_body,
            ..
        } => contains_continue(then_body) || contains_continue(else_body),
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            contains_continue(try_body)
                || catches.iter().any(|(_, b)| contains_continue(b))
                || finally_body.as_ref().is_some_and(|f| contains_continue(f))
        }
        Stmt::Switch { cases, tail, default, .. } => {
            cases.iter().any(|(_, b)| contains_continue(b))
                || contains_continue(tail)
                || default.as_ref().is_some_and(|arm| contains_continue(arm))
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            cases.iter().any(|(_, b)| contains_continue(b))
                || tail.as_ref().is_some_and(|t| contains_continue(t))
        }
        // A nested loop's `continue` belongs to the nested loop.
        Stmt::While { .. }
        | Stmt::DoWhile { .. }
        | Stmt::For { .. }
        | Stmt::ForEach { .. }
        | Stmt::Using { .. }
        | Stmt::Lock { .. } => false,
        _ => false,
    })
}

/// Every label DEFINED anywhere in `list`, at any depth.
pub fn collect_labels(list: &[Stmt], out: &mut HashSet<String>) {
    for s in list {
        if let Stmt::Label(l) = s {
            out.insert(l.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for k in kids {
            collect_labels(k, out);
        }
    }
}

/// Every label TARGETED by a `goto` anywhere in `list`, at any depth.
pub fn collect_goto_targets(list: &[Stmt], out: &mut HashSet<String>) {
    for s in list {
        if let Stmt::Goto(l) = s {
            out.insert(l.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for k in kids {
            collect_goto_targets(k, out);
        }
    }
}

/// Is this statement an EXIT from the region it sits in — the statements a
/// compiler must inline a `finally` in front of?
///
/// `return`/`throw` leave the method; `goto`/`break`/`continue` leave the
/// protected region of the statement they sit in (the region is a statement
/// list, so any of them can leave it). This is the positive evidence of an
/// inlined `finally`: measured on the x64 corpus the lock shape is
/// `Monitor.Enter(X, 0); … Monitor.Exit(X); return;` with the `Exit` inlined in
/// front of each early exit, and never a `try/finally`.
pub fn leaves_region(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue
    )
}

/// The single argument of a `System.Threading.Monitor.<member>(arg[, 0])`
/// statement's call, or `None`.
///
/// Accepts the two spellings the renderer produces (a dotted name
/// `Monitor.Enter` with no receiver, or the owner as a receiver `Monitor`), and
/// reports whether the second argument — the `lockTaken` out-parameter of the
/// C# `lock` codegen — is present as a literal.
pub fn monitor_call<'a>(e: &'a Expr, member: &'a str) -> Option<MonitorCall<'a>> {
    let (recv, name, args) = split_call(e)?;
    let matches_member = name == member || name.ends_with(&format!(".{member}"));
    if !matches_member {
        return None;
    }
    // The qualifier must say Monitor: a bare `Enter`/`Exit` is far too common a
    // member name to rewrite on its own.
    match name.rsplit_once('.') {
        Some((qual, _)) => {
            if !qual.ends_with("Monitor") {
                return None;
            }
        }
        None => match recv {
            Some(Expr::Var(r)) if r.ends_with("Monitor") => {}
            Some(Expr::StaticField(_, f)) if f.ends_with("Monitor") => {}
            _ => return None,
        },
    }
    let arg = args.first()?;
    // The `lock` codegen's second argument is the `lockTaken` out-parameter.
    //
    // Three real shapes are measured across the corpora:
    //
    //   * x64 (windows-cs): `Monitor.Enter(obj, 0)` — literal `0`/`false`
    //   * ARM32 (temple-run-2-recon): `Monitor.Enter(obj, (nint*)(stack +4))`
    //     — a pointer to the lockTaken slot (RawAddr), not a literal
    //   * ARM32 (tomb-of-maze-cs): `Monitor.Enter(obj, ptr, &lockTakenVar)`
    //     — 3-arg form where the third arg is the lockTaken variable
    //
    // A live non-pointer local as the second arg is a hand-written
    // `Monitor.Enter(o, ref flag)` and is NOT lock codegen.
    let (lit_lock_taken, lock_taken_var) = match args.len() {
        1 => (false, None),
        2 => {
            if matches!(&args[1], Expr::Int(0) | Expr::Bool(false) | Expr::Null) {
                (true, None)
            } else if matches!(&args[1], Expr::RawAddr(..)) {
                // ARM32 pointer-to-lockTaken: still lock codegen.
                (true, None)
            } else {
                // A live local: hand-written Monitor.Enter, not lock codegen.
                (false, None)
            }
        }
        3 => {
            // ARM32 3-arg: Monitor.Enter(obj, lockTakenPtr, &lockTakenVar)
            // The third arg is the lockTaken variable (AddressOf).
            let lt_var = match &args[2] {
                Expr::AddressOf(inner) => Some(inner.as_ref().clone()),
                _ => None,
            };
            (true, lt_var)
        }
        _ => return None,
    };
    Some(MonitorCall {
        arg,
        lit_lock_taken,
        single_arg: args.len() == 1,
        lock_taken_var,
    })
}

/// See [`monitor_call`].
pub struct MonitorCall<'a> {
    /// The object the monitor is entered on / exited from.
    pub arg: &'a Expr,
    /// The second argument was the `lockTaken` literal — the C# `lock` codegen.
    pub lit_lock_taken: bool,
    /// The call had exactly one argument (`Monitor.Enter(x)`).
    pub single_arg: bool,
    /// The lockTaken variable from the 3-arg ARM32 pattern
    /// (`Monitor.Enter(obj, ptr, &var)`). When present, `Monitor.Exit` may
    /// target this variable (dereferenced) rather than the original object.
    pub lock_taken_var: Option<Expr>,
}

/// Is `s` exactly an `Eval(Monitor.<member>(arg…))` statement?
pub fn monitor_stmt<'a>(s: &'a Stmt, member: &'a str) -> Option<MonitorCall<'a>> {
    match s {
        Stmt::Eval(e) => monitor_call(e, member),
        _ => None,
    }
}

/// `true` when `s` is exactly `Monitor.<member>(...)` on an argument rendering
/// identically to `target`.
pub fn monitor_stmt_on(s: &Stmt, member: &str, target: &Expr) -> bool {
    monitor_stmt(s, member).is_some_and(|c| exprs_eq(c.arg, target))
}

/// Does this statement perform no observable action — no call, no construction,
/// no memory write, no control flow? The weaker half of the "nothing can observe
/// the reordering" test.
pub fn is_pure_local_stmt(s: &Stmt) -> bool {
    match s {
        Stmt::Comment(_) | Stmt::Attribute { .. } => true,
        Stmt::Assign(_, _, e) => !expr_has_call(e),
        _ => false,
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn ctor_names_are_recognized_in_both_spellings() {
        assert!(is_ctor_name(".ctor"));
        assert!(is_ctor_name("System.Action..ctor"));
        assert!(is_ctor_name("List<int>..ctor"));
        assert!(!is_ctor_name("get_Item"));
        assert_eq!(ctor_type_name("System.Action..ctor"), Some("System.Action"));
        assert_eq!(ctor_type_name(".ctor"), None);
    }

    #[test]
    fn monitor_call_needs_a_monitor_qualifier() {
        let qualified = Expr::Call {
            recv: None,
            name: "Monitor.Enter".into(),
            args: vec![Expr::Var("g".into()), Expr::Int(0)],
        };
        let c = monitor_call(&qualified, "Enter").expect("dotted name");
        assert!(c.lit_lock_taken);
        assert!(matches!(c.arg, Expr::Var(n) if n == "g"));

        let bare = Expr::Call {
            recv: None,
            name: "Enter".into(),
            args: vec![Expr::Var("g".into())],
        };
        assert!(
            monitor_call(&bare, "Enter").is_none(),
            "bare `Enter` proves nothing"
        );

        let recv_form = Expr::Call {
            recv: Some(Box::new(Expr::Var("Monitor".into()))),
            name: "Exit".into(),
            args: vec![Expr::Var("g".into())],
        };
        let c = monitor_call(&recv_form, "Exit").expect("receiver spelling");
        assert!(c.single_arg);
        assert!(!c.lit_lock_taken);
    }

    #[test]
    fn a_live_lock_taken_flag_is_not_the_lock_codegen() {
        let call = Expr::Call {
            recv: None,
            name: "Monitor.Enter".into(),
            args: vec![Expr::Var("g".into()), Expr::Var("flag".into())],
        };
        let c = monitor_call(&call, "Enter").expect("still a Monitor call");
        assert!(!c.lit_lock_taken, "a live flag means a hand-written Enter");
    }

    /// ARM32 2-arg pointer pattern: second arg is a RawAddr (pointer to lockTaken slot).
    #[test]
    fn arm32_2arg_pointer_is_lock_codegen() {
        let call = Expr::Call {
            recv: None,
            name: "Monitor.Enter".into(),
            args: vec![
                Expr::Var("obj".into()),
                Expr::RawAddr(Box::new(Expr::Var("stack".into())), 4),
            ],
        };
        let c = monitor_call(&call, "Enter").expect("ARM32 2-arg pointer");
        assert!(c.lit_lock_taken, "pointer to lockTaken is lock codegen");
        assert!(c.lock_taken_var.is_none());
    }

    /// ARM32 3-arg pattern: third arg is the lockTaken variable.
    #[test]
    fn arm32_3arg_extracts_lock_taken_var() {
        let call = Expr::Call {
            recv: None,
            name: "Monitor.Enter".into(),
            args: vec![
                Expr::Var("obj".into()),
                Expr::RawAddr(Box::new(Expr::Var("stack".into())), 4),
                Expr::AddressOf(Box::new(Expr::Var("lt".into()))),
            ],
        };
        let c = monitor_call(&call, "Enter").expect("ARM32 3-arg");
        assert!(c.lit_lock_taken, "3-arg is lock codegen");
        assert!(
            matches!(&c.lock_taken_var, Some(Expr::Var(n)) if n == "lt"),
            "lockTaken variable extracted from 3rd arg"
        );
    }

    #[test]
    fn continue_is_counted_at_this_level_only() {
        let nested = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Continue],
        }];
        assert!(
            !contains_continue(&nested),
            "a nested loop owns its own continue"
        );
        let here = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Continue],
            else_body: vec![],
        }];
        assert!(contains_continue(&here));
    }

    #[test]
    fn inline_form_is_only_assign_and_eval() {
        assert!(is_inlinable(&Stmt::Assign("i".into(), None, Expr::Int(0))));
        assert!(is_inlinable(&Stmt::Eval(Expr::Var("f".into()))));
        assert!(!is_inlinable(&Stmt::Comment("x".into())));
        assert!(!is_inlinable(&Stmt::Continue));
    }

    #[test]
    fn goto_targets_are_collected_at_any_depth() {
        let body = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Goto("L1".into())],
            else_body: vec![],
        }];
        let mut t = HashSet::new();
        collect_goto_targets(&body, &mut t);
        assert!(t.contains("L1"));
    }
}

#[cfg(test)]
mod normalize_ctor_calls_tests {
    use super::*;

    #[test]
    fn double_dot_ctor_becomes_newobj() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "System.Action..ctor".into(),
            args: vec![Expr::Int(42)],
        })];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::NewObj(ty, args)) => {
                assert_eq!(ty, "System.Action");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected Eval(NewObj), got {other:?}"),
        }
    }

    #[test]
    fn single_dot_ctor_becomes_newobj() {
        let mut stmts = vec![Stmt::Assign(
            "v1".into(),
            None,
            Expr::Call {
                recv: None,
                name: "List<int>.ctor".into(),
                args: vec![Expr::Int(0)],
            },
        )];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(name, _, Expr::NewObj(ty, args)) => {
                assert_eq!(name, "v1");
                assert_eq!(ty, "List<int>");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected Assign(_, _, NewObj), got {other:?}"),
        }
    }

    /// The renderer now emits the allocation as its own statement
    /// (`v1 = new JSONObject();`), so the merge must accept `NewObj` as the
    /// allocation side and keep the type it already carries — that type is
    /// exact, while the ctor name may not decode.
    #[test]
    fn an_allocation_statement_merges_with_its_ctor() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::NewObj("JSONObject".into(), vec![])),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v1".into()))),
                name: "..ctor".into(),
                args: vec![Expr::Int(7)],
            }),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        assert_eq!(stmts.len(), 1, "the ctor statement is merged away");
        match &stmts[0] {
            Stmt::Assign(name, _, Expr::NewObj(ty, args)) => {
                assert_eq!(name, "v1");
                assert_eq!(ty, "JSONObject");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected Assign(_, _, NewObj), got {other:?}"),
        }
    }

    #[test]
    fn bare_dot_ctor_is_left_unchanged() {
        // `.ctor` with no type prefix has no type info — must not rewrite.
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: ".ctor".into(),
            args: vec![],
        })];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == ".ctor"));
    }

    #[test]
    fn non_ctor_call_is_untouched() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Console.WriteLine".into(),
            args: vec![Expr::Str("hello".into())],
        })];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "Console.WriteLine"));
    }

    #[test]
    fn nested_ctor_in_if_body_is_rewritten() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "Foo..ctor".into(),
                args: vec![],
            })],
            else_body: vec![],
        }];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Eval(Expr::NewObj(ty, _)) if ty == "Foo"));
            }
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn multiple_ctors_in_one_method_all_converted() {
        let mut stmts = vec![
            Stmt::Assign(
                "a".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "A..ctor".into(),
                    args: vec![],
                },
            ),
            Stmt::Assign(
                "b".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "B..ctor".into(),
                    args: vec![Expr::Int(1)],
                },
            ),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "C..ctor".into(),
                args: vec![Expr::Int(2), Expr::Int(3)],
            }),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 3);
        for s in &stmts {
            match s {
                Stmt::Assign(_, _, Expr::NewObj(..)) | Stmt::Eval(Expr::NewObj(..)) => {}
                other => panic!("expected NewObj, got {other:?}"),
            }
        }
    }

    #[test]
    fn no_ctor_calls_is_noop() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Int(0)),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Foo".into(),
                args: vec![],
            }),
        ];
        let original = stmts.clone();
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 0);
        // Statements should be unchanged.
        assert_eq!(stmts.len(), original.len());
    }

    #[test]
    fn ctor_in_while_body_is_rewritten() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "Widget..ctor".into(),
                args: vec![Expr::Str("name".into())],
            })],
        }];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[0], Stmt::Eval(Expr::NewObj(ty, args)) if ty == "Widget" && args.len() == 1));
            }
            _ => panic!("expected While"),
        }
    }

    #[test]
    fn args_are_preserved_after_rewrite() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Multi..ctor".into(),
            args: vec![Expr::Int(1), Expr::Str("two".into()), Expr::Bool(true)],
        })];
        normalize_ctor_calls(&mut stmts);
        match &stmts[0] {
            Stmt::Eval(Expr::NewObj(ty, new_args)) => {
                assert_eq!(ty, "Multi");
                assert_eq!(new_args.len(), 3);
                // Verify the args were preserved correctly.
                assert!(matches!(&new_args[0], Expr::Int(1)));
                assert!(matches!(&new_args[1], Expr::Str(s) if s == "two"));
                assert!(matches!(&new_args[2], Expr::Bool(true)));
            }
            other => panic!("expected Eval(NewObj), got {other:?}"),
        }
    }

    #[test]
    fn ctor_in_try_catch_is_rewritten() {
        let mut stmts = vec![Stmt::Try {
            try_body: vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "Res..ctor".into(),
                args: vec![],
            })],
            catches: vec![(
                "ex".into(),
                vec![Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Err..ctor".into(),
                    args: vec![],
                })],
            )],
            finally_body: Some(vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "Cleanup..ctor".into(),
                args: vec![],
            })]),
        }];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 3);
    }

    // -----------------------------------------------------------------------
    // Two-statement allocation pattern tests (the REAL IL2CPP shape)
    // -----------------------------------------------------------------------

    /// The real IL2CPP pattern: RawCall allocation + .ctor eval.
    /// Should be merged into a single NewObj.
    #[test]
    fn two_stmt_rawcall_alloc_plus_ctor() {
        let mut stmts = vec![
            // v150 = fn_1302628(typeInfo, this, ...);
            Stmt::Assign(
                "v150".into(),
                None,
                Expr::RawCall(0x1302628, vec![
                    Expr::RawDeref(Box::new(Expr::Int(0x2f6a000)), 3896, 0),
                    Expr::This,
                ], None),
            ),
            // v150..ctor();
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v150".into()))),
                name: ".ctor".into(),
                args: vec![],
            }),
            // v150.OnEnter = v132;
            Stmt::Assign("v150.OnEnter".into(), None, Expr::Var("v132".into())),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1, "two-statement alloc+ctor should merge");
        assert_eq!(stmts.len(), 2, "alloc+ctor merged, field store remains");
        // First statement should now be Assign(v150, None, NewObj("object", []))
        match &stmts[0] {
            Stmt::Assign(v, _, Expr::NewObj(ty, args)) => {
                assert_eq!(v, "v150");
                assert_eq!(ty, "object");
                assert!(args.is_empty());
            }
            other => panic!("expected Assign(_, _, NewObj), got {other:?}"),
        }
        // Second statement should be the field store.
        assert!(matches!(&stmts[1], Stmt::Assign(t, _, _) if t == "v150.OnEnter"));
    }

    /// Two-statement pattern with a regular Call allocation (not RawCall).
    #[test]
    fn two_stmt_call_alloc_plus_ctor() {
        let mut stmts = vec![
            // v = il2cpp_alloc(size);
            Stmt::Assign(
                "v".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "il2cpp_alloc".into(),
                    args: vec![Expr::Int(32)],
                },
            ),
            // v..ctor(42);
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: ".ctor".into(),
                args: vec![Expr::Int(42)],
            }),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(v, _, Expr::NewObj(ty, args)) => {
                assert_eq!(v, "v");
                assert_eq!(ty, "object");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Int(42)));
            }
            other => panic!("expected Assign(_, _, NewObj), got {other:?}"),
        }
    }

    /// Two-statement pattern where ctor has a type prefix in the name.
    #[test]
    fn two_stmt_with_typed_ctor_name() {
        let mut stmts = vec![
            Stmt::Assign("v".into(), None, Expr::RawCall(0x1000, vec![], None)),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: "MyType..ctor".into(),
                args: vec![Expr::Str("hello".into())],
            }),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::NewObj(ty, args)) => {
                assert_eq!(ty, "MyType", "type extracted from ctor name");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected NewObj, got {other:?}"),
        }
    }

    /// A constructor's return value is the receiver object, not a fresh
    /// allocation: `v17 = v33..ctor(args)` aliases `v33`.
    #[test]
    fn ctor_result_assignment_aliases_the_receiver() {
        let mut stmts = vec![
            Stmt::Assign(
                "v33".into(),
                None,
                Expr::NewObj("RewardData".into(), vec![Expr::Var("type".into())]),
            ),
            Stmt::Assign(
                "v17".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::Var("v33".into()))),
                    name: "RewardData..ctor".into(),
                    args: vec![Expr::Var("type".into())],
                },
            ),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[1] {
            Stmt::Assign(v, _, Expr::Var(r)) => {
                assert_eq!(v, "v17");
                assert_eq!(r, "v33");
            }
            other => panic!("expected alias, got {other:?}"),
        }
        let text = format!("{stmts:?}");
        assert_eq!(text.matches("NewObj").count(), 1, "{text}");
    }

    /// Non-matching receiver: alloc on v1, ctor on v2 — should NOT merge.
    #[test]
    fn two_stmt_mismatched_receiver_not_merged() {        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::RawCall(0x1000, vec![], None)),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v2".into()))), // different var!
                name: ".ctor".into(),
                args: vec![],
            }),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 0, "mismatched receiver should not merge");
        assert_eq!(stmts.len(), 2);
    }

    /// Non-adjacent alloc and ctor: an intervening statement prevents merging.
    #[test]
    fn two_stmt_non_adjacent_not_merged() {
        let mut stmts = vec![
            Stmt::Assign("v".into(), None, Expr::RawCall(0x1000, vec![], None)),
            Stmt::Assign("x".into(), None, Expr::Int(42)), // intervening
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: ".ctor".into(),
                args: vec![],
            }),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 0, "non-adjacent alloc+ctor should not merge");
        assert_eq!(stmts.len(), 3);
    }

    /// Multiple two-statement patterns in one method.
    #[test]
    fn multiple_two_stmt_patterns() {
        let mut stmts = vec![
            Stmt::Assign("a".into(), None, Expr::RawCall(0x1000, vec![], None)),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("a".into()))),
                name: ".ctor".into(),
                args: vec![],
            }),
            Stmt::Assign("b".into(), None, Expr::RawCall(0x2000, vec![], None)),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("b".into()))),
                name: ".ctor".into(),
                args: vec![Expr::Int(1)],
            }),
        ];
        let n = normalize_ctor_calls(&mut stmts);
        assert_eq!(n, 2, "both two-statement patterns should merge");
        assert_eq!(stmts.len(), 2);
    }

    /// Full pipeline: two-statement alloc + ctor + field stores → object init.
    #[test]
    fn full_pipeline_two_stmt_to_object_init() {
        let mut stmts = vec![
            Stmt::Assign("v".into(), None, Expr::RawCall(0x1000, vec![], None)),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: ".ctor".into(),
                args: vec![],
            }),
            Stmt::Assign("v.X".into(), None, Expr::Int(1)),
            Stmt::Assign("v.Y".into(), None, Expr::Int(2)),
        ];
        // Run the full pipeline.
        apply(&mut stmts);
        // Should have: Assign(v, ObjInit)
        assert_eq!(stmts.len(), 1, "all should fold into one statement");
        match &stmts[0] {
            Stmt::Assign(v, _, Expr::ObjInit { type_name, fields, .. }) => {
                assert_eq!(v, "v");
                assert_eq!(type_name, "object");
                assert_eq!(fields.len(), 2);
            }
            other => panic!("expected Assign(_, _, ObjInit), got {other:?}"),
        }
    }

    /// Full pipeline: two-statement alloc + ctor + Add calls → collection init.
    #[test]
    fn full_pipeline_two_stmt_to_collection_init() {
        let mut stmts = vec![
            Stmt::Assign("v".into(), None, Expr::RawCall(0x1000, vec![], None)),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: ".ctor".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: "Add".into(),
                args: vec![Expr::Int(1)],
            }),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: "Add".into(),
                args: vec![Expr::Int(2)],
            }),
        ];
        apply(&mut stmts);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::CollectionInit { items, .. }) => {
                assert_eq!(items.len(), 2);
            }
            other => panic!("expected CollectionInit, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests for the fall-through merging pre-pass
// ---------------------------------------------------------------------------

#[cfg(test)]
mod merge_tests {
    use super::*;

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

    // -----------------------------------------------------------------------
    // ends_with_terminator
    // -----------------------------------------------------------------------

    #[test]
    fn empty_list_does_not_end_with_terminator() {
        assert!(!ends_with_terminator(&[]));
    }

    #[test]
    fn return_is_a_terminator() {
        assert!(ends_with_terminator(&[Stmt::Return(None)]));
        assert!(ends_with_terminator(&[Stmt::Return(Some(Expr::Int(1)))]));
    }

    #[test]
    fn throw_is_a_terminator() {
        assert!(ends_with_terminator(&[Stmt::Throw(Expr::Var("e".into()))]));
    }

    #[test]
    fn goto_is_a_terminator() {
        assert!(ends_with_terminator(&[Stmt::Goto("L".into())]));
    }

    #[test]
    fn break_and_continue_are_terminators() {
        assert!(ends_with_terminator(&[Stmt::Break]));
        assert!(ends_with_terminator(&[Stmt::Continue]));
    }

    #[test]
    fn eval_is_not_a_terminator() {
        assert!(!ends_with_terminator(&[Stmt::Eval(v("x"))]));
    }

    #[test]
    fn label_is_not_a_terminator() {
        assert!(!ends_with_terminator(&[Stmt::Label("L".into())]));
    }

    #[test]
    fn terminator_must_be_last() {
        // A return followed by another statement: the LAST stmt is not a
        // terminator, so the list falls through.
        assert!(!ends_with_terminator(&[
            Stmt::Return(None),
            Stmt::Eval(v("x")),
        ]));
    }

    // -----------------------------------------------------------------------
    // merge_fallthrough_lists — the main pre-pass
    // -----------------------------------------------------------------------

    /// The core scenario: `Monitor.Enter` in `then_body`, `Monitor.Exit` in the
    /// parent list after the `if`. The `else_body` ends with `return`, so the
    /// post-if code is only reachable from `then_body`'s fall-through. The
    /// pre-pass must splice the Exit into `then_body` so `using_lock` can pair.
    #[test]
    fn lock_exit_in_sibling_list_is_merged_into_then_body() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    monitor("Enter", v("gate"), true),
                    Stmt::Eval(v("body")),
                    // Falls through (no terminator)
                ],
                else_body: vec![Stmt::Return(None)],
            },
            Stmt::Label("L_exit".into()),
            monitor("Exit", v("gate"), false),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 1, "one merge must happen");
        // The parent list should now be just the If (label + tail removed).
        assert_eq!(stmts.len(), 1, "label + tail removed from parent");
        // The then_body should now contain the Exit and Return.
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 5, "Enter + body + Label + Exit + Return");
                assert!(
                    monitor_stmt_on(&then_body[3], "Exit", &v("gate")),
                    "Exit must be in then_body now"
                );
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// After the pre-pass merges the Exit into `then_body`, the `using_lock`
    /// pass must be able to pair Enter/Exit and produce a `lock`.
    #[test]
    fn lock_pattern_folds_after_merge() {
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
        // Run the full reconstruction pipeline.
        apply(&mut stmts);
        // A constant-true If may be spliced away, so the Lock can sit either
        // in then_body or at the top level; the pairing is what matters.
        let has_lock = match &stmts[0] {
            Stmt::Lock { .. } => true,
            Stmt::If { then_body, .. } => then_body.iter().any(|s| matches!(s, Stmt::Lock { .. })),
            _ => false,
        };
        assert!(has_lock, "a lock must be present after reconstruction: {stmts:?}");
    }

    /// The else_body falls through (no terminator), so the post-if code is
    /// reachable from BOTH paths. The merge must NOT happen.
    #[test]
    fn both_branches_fall_through_prevents_merge() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    monitor("Enter", v("gate"), true),
                    Stmt::Eval(v("body")),
                ],
                else_body: vec![Stmt::Eval(v("other"))],
            },
            Stmt::Label("L_exit".into()),
            monitor("Exit", v("gate"), false),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 0, "both branches fall through — merge is unsafe");
        assert_eq!(stmts.len(), 3, "nothing changed (If + Label + Exit)");
    }

    /// A non-label statement after the nested structure prevents the merge.
    #[test]
    fn non_label_after_nested_structure_prevents_merge() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(v("body"))],
                else_body: vec![Stmt::Return(None)],
            },
            Stmt::Eval(v("not_a_label")),
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 0, "the statement after the If is not a label");
    }

    /// A `Using` whose body falls through, followed by a label, is merged.
    #[test]
    fn using_body_fallthrough_is_merged() {
        let mut stmts = vec![
            Stmt::Using {
                decl: Box::new(Stmt::Assign(
                    "r".into(),
                    None,
                    Expr::NewObj("Reader".into(), vec![]),
                )),
                body: vec![Stmt::Eval(v("body"))],
            },
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Using { body, .. } => {
                assert_eq!(body.len(), 3, "body + Label + Return merged");
                assert!(matches!(body[2], Stmt::Return(None)));
            }
            other => panic!("expected Using, got {other:?}"),
        }
    }

    /// A `Lock` whose body falls through, followed by a label, is merged.
    #[test]
    fn lock_body_fallthrough_is_merged() {
        let mut stmts = vec![
            Stmt::Lock {
                expr: v("gate"),
                body: vec![Stmt::Eval(v("body"))],
            },
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Lock { body, .. } => {
                assert_eq!(body.len(), 3, "body + Label + Return merged");
            }
            other => panic!("expected Lock, got {other:?}"),
        }
    }

    /// A `While` body does NOT fall through (it loops), so no merge.
    #[test]
    fn while_body_does_not_fall_through() {
        let mut stmts = vec![
            Stmt::While {
                cond: Expr::Bool(true),
                body: vec![Stmt::Eval(v("body"))],
            },
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 0, "while body loops, not falls through");
    }

    /// The else_body falls through while the then_body terminates: the merge
    /// goes into the else_body.
    #[test]
    fn else_body_fallthrough_is_merged() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Return(Some(Expr::Int(1)))],
                else_body: vec![Stmt::Eval(v("body"))],
            },
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { else_body, .. } => {
                // The label moves WITH the tail: another goto may target it.
                assert_eq!(else_body.len(), 3, "body + Label + Return merged into else_body");
                assert!(matches!(else_body[1], Stmt::Label(ref l) if l == "L"));
                assert!(matches!(else_body[2], Stmt::Return(None)));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// Nested merges: an inner If's fall-through is merged first, then the
    /// outer If's fall-through is merged in a second pass.
    #[test]
    fn nested_fallthrough_merges_chain() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    Stmt::If {
                        cond: Expr::Bool(true),
                        then_body: vec![Stmt::Eval(v("inner"))],
                        else_body: vec![Stmt::Return(None)],
                    },
                    Stmt::Label("L_inner".into()),
                    Stmt::Eval(v("after_inner")),
                ],
                else_body: vec![Stmt::Return(None)],
            },
            Stmt::Label("L_outer".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 2, "both inner and outer merges happen");
        // The outer then_body should now be just the inner If + Return.
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 3, "inner If + Label + outer Return");
                assert!(matches!(&then_body[0], Stmt::If { .. }));
                assert!(matches!(&then_body[1], Stmt::Label(l) if l == "L_outer"));
                assert!(matches!(&then_body[2], Stmt::Return(None)));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// A label that is NOT immediately after the nested structure does not
    /// trigger a merge.
    #[test]
    fn label_not_immediately_after_prevents_merge() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(v("body"))],
                else_body: vec![Stmt::Return(None)],
            },
            Stmt::Comment("gap".into()),
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 0, "a comment between the If and the label blocks the merge");
    }

    /// The merge preserves the label's target semantics: the label is consumed
    /// because the fall-through now reaches the same code without a jump.
    #[test]
    fn label_moves_with_merged_tail() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(v("body"))],
                else_body: vec![Stmt::Return(None)],
            },
            Stmt::Label("L".into()),
            Stmt::Eval(v("after")),
        ];
        merge_fallthrough_lists(&mut stmts);
        // The label must SURVIVE: another goto may target it, and dropping it
        // here left the whole target branch dangling.
        let mut labels = HashSet::new();
        collect_labels(&stmts, &mut labels);
        assert!(labels.contains("L"), "the label must move with the tail");
    }

    /// A Try with a falling-through try_body and terminating catches/finally
    /// is eligible for merge.
    #[test]
    fn try_with_falling_through_body_is_merged() {
        let mut stmts = vec![
            Stmt::Try {
                try_body: vec![Stmt::Eval(v("body"))],
                catches: vec![("Exception".into(), vec![Stmt::Return(None)])],
                finally_body: None,
            },
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Try { try_body, .. } => {
                assert_eq!(try_body.len(), 3, "body + Label + Return merged into try_body");
            }
            other => panic!("expected Try, got {other:?}"),
        }
    }

    /// A Try where both try_body and a catch fall through is NOT eligible.
    #[test]
    fn try_with_multiple_falling_through_children_is_not_merged() {
        let mut stmts = vec![
            Stmt::Try {
                try_body: vec![Stmt::Eval(v("body"))],
                catches: vec![("Exception".into(), vec![Stmt::Eval(v("catch"))])],
                finally_body: None,
            },
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        let n = merge_fallthrough_lists(&mut stmts);
        assert_eq!(n, 0, "both try_body and catch fall through — unsafe");
    }
}
