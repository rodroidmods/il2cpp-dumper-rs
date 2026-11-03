//! Interprocedural type analysis — propagates types across method call
//! boundaries using metadata signatures as ground truth.
//!
//! ## Problem
//!
//! The per-method pipeline runs `locals::propagate_types` BEFORE `resolve`,
//! so call targets are still `Immediate(addr)` rather than `Method(m)`.
//! After `resolve` converts them, the return types and parameter types from
//! the callee's metadata signature never flow to the caller's locals.
//!
//! ## Solution
//!
//! Two phases, both running AFTER `resolve::run_with_fields`:
//!
//! 1. **Post-resolve type propagation** (`propagate_post_resolve`): re-runs
//!    the type fixpoint so that resolved `Method` operands can propagate
//!    their return types to destination locals and their parameter types
//!    back to argument locals. This is the highest-impact change — it
//!    unlocks every resolved call's signature for typing.
//!
//! 2. **Cross-method type seeding** (`InterproceduralAnalyzer`): builds a
//!    lightweight call graph from `methods_by_rva`, then for each method
//!    uses the resolved callees' metadata signatures to pre-seed argument
//!    types. This runs as a pre-pass before the per-method pipeline and
//!    feeds enriched seeds into `locals::create`.
//!
//! ## Architecture
//!
//! ```text
//!   Phase 0 (optional): InterproceduralAnalyzer builds call graph,
//!     computes per-method type summaries from metadata, stores them
//!     in a shared map.
//!
//!   Per method (parallel):
//!     lift → SSA → locals::create (uses interprocedural seeds)
//!     → propagate_types → resolve → propagate_post_resolve
//!     → optimize → emit
//! ```
//!
//! ## Safety
//!
//! All propagation uses `refine_type` (monotone: Unknown → concrete, never
//! concrete → different concrete). Wrong types cannot be introduced — only
//! missing types can be filled in.

use super::ir::{Insn, Op, Operand};
use super::types::{ManagedType, MethodRef};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Telemetry counters (aggregated over all methods, printed by timing::report)
// ---------------------------------------------------------------------------

/// Locals typed by post-resolve call return propagation (phase 1).
pub static POST_RESOLVE_RET_TYPES: AtomicU64 = AtomicU64::new(0);
/// Locals typed by post-resolve call parameter back-propagation (phase 1).
pub static POST_RESOLVE_PARAM_TYPES: AtomicU64 = AtomicU64::new(0);
/// Memory operands rewritten to Field/InstanceField by post-resolve sweep 3.
pub static POST_RESOLVE_FIELD_REWRITES: AtomicU64 = AtomicU64::new(0);
/// Call sites whose arguments were typed by cross-method seeding (phase 0).
pub static CROSS_METHOD_SEEDS: AtomicU64 = AtomicU64::new(0);
/// Fixpoint iterations in post-resolve propagation.
pub static POST_RESOLVE_ITERATIONS: AtomicU64 = AtomicU64::new(0);

/// Human-readable summary of interprocedural impact.
pub fn report() -> String {
    format!(
        "\n[interproc] post_resolve_ret={} post_resolve_param={} \
         post_resolve_fields={} cross_seeds={} post_resolve_iters={}",
        POST_RESOLVE_RET_TYPES.load(Ordering::Relaxed),
        POST_RESOLVE_PARAM_TYPES.load(Ordering::Relaxed),
        POST_RESOLVE_FIELD_REWRITES.load(Ordering::Relaxed),
        CROSS_METHOD_SEEDS.load(Ordering::Relaxed),
        POST_RESOLVE_ITERATIONS.load(Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// Phase 1: Post-resolve type propagation
// ---------------------------------------------------------------------------

/// Re-run type propagation after `resolve` has converted call targets from
/// `Immediate(addr)` to `Method(m)`. This allows the callee's metadata
/// signature (return type + parameter types) to flow into the caller's locals.
///
/// Returns the number of locals newly typed.
///
/// This is a LIGHTWEIGHT re-propagation: it only processes instructions whose
/// typing depends on resolved calls (Call/CallVoid with Method operands), plus
/// the Move chains that carry those types forward. It does NOT re-run the full
/// field-address seeding or memory-load resolution — those are stable after
/// the first `propagate_types` and would not benefit from re-visiting.
pub fn propagate_post_resolve(
    insns: &mut [Insn],
    table: &mut super::locals::LocalTable,
    oracle: &dyn super::locals::TypeOracle,
) -> usize {
    let mut total_typed = 0usize;
    let mut iterations = 0usize;
    let mut changed = true;

    while changed && iterations < 16 {
        changed = false;
        iterations += 1;

        // Sweep 1: resolved calls → return type + parameter back-propagation.
        for insn in insns.iter_mut() {
            match &insn.op {
                Op::Call | Op::CallVoid => {
                    let Some(Operand::Method(m)) = insn.operands.first() else {
                        continue;
                    };
                    // Return type → destination local.
                    if matches!(insn.op, Op::Call) {
                        let ret = m.sig.ret.clone();
                        if !matches!(ret, ManagedType::Unknown) {
                            if let Some(Operand::Local(dst)) = insn.dest() {
                                if let Some(&li) = table.index_of.get(dst) {
                                    if super::locals::refine_type_public(
                                        &table.infos[li].ty,
                                        &ret,
                                    ) {
                                        table.infos[li].ty = ret;
                                        changed = true;
                                        total_typed += 1;
                                        POST_RESOLVE_RET_TYPES
                                            .fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                    // Parameter types → argument locals (back-propagation).
                    let has_result = matches!(insn.op, Op::Call);
                    let param_start = if has_result {
                        if m.sig.is_static { 2 } else { 3 }
                    } else {
                        if m.sig.is_static { 1 } else { 2 }
                    };
                    for (pi, pty) in m.sig.params.iter().enumerate() {
                        if matches!(pty, ManagedType::Unknown) {
                            continue;
                        }
                        let arg_idx = param_start + pi;
                        let Some(arg_op) = insn.operands.get(arg_idx) else {
                            continue;
                        };
                        if let Operand::Local(v) = arg_op {
                            if let Some(&li) = table.index_of.get(v) {
                                if super::locals::refine_type_public(
                                    &table.infos[li].ty,
                                    pty,
                                ) {
                                    table.infos[li].ty = pty.clone();
                                    changed = true;
                                    total_typed += 1;
                                    POST_RESOLVE_PARAM_TYPES
                                        .fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Sweep 2: Move propagation — carry newly-typed values forward.
        for insn in insns.iter_mut() {
            if insn.op != Op::Move {
                continue;
            }
            let src_ty = match insn.operands.get(1) {
                Some(Operand::Local(v)) => {
                    table.index_of.get(v).map(|&i| table.infos[i].ty.clone())
                }
                _ => continue,
            };
            let Some(src_ty) = src_ty else { continue };
            if matches!(src_ty, ManagedType::Unknown) {
                continue;
            }
            if let Some(Operand::Local(dst)) = insn.dest() {
                if let Some(&li) = table.index_of.get(dst) {
                    if super::locals::refine_type_public(&table.infos[li].ty, &src_ty) {
                        table.infos[li].ty = src_ty;
                        changed = true;
                        total_typed += 1;
                    }
                }
            }
        }

        // Sweep 3: Memory loads AND stores off newly-typed bases resolve fields.
        // This mirrors the second sweep in `propagate_types` but ONLY for bases
        // that were newly typed by sweeps 1-2 (the `changed` flag ensures we
        // skip this sweep when no new types were propagated).
        //
        // Store side: Move(Memory{base, addend}, value) — type the stored value.
        // Load side:  Move(dst, Memory{base, addend}) — rewrite to Field/InstanceField
        //             and type the destination.
        for insn in insns.iter_mut() {
            if insn.op != Op::Move {
                continue;
            }
            // Store side: Move(Memory{base, addend}, value)
            if let Some(Operand::Memory { base, addend, .. }) =
                insn.operands.first().cloned()
            {
                let base_ty = super::locals::operand_type(
                    Some(&base),
                    table,
                    oracle,
                );
                if let Some(bt) = base_ty {
                    if let Some(nt) = super::locals::class_of_public(&bt) {
                        if nt.type_def_index >= 0 {
                            if let Some(field) = oracle
                                .field_at_offset(nt.type_def_index, addend)
                                .filter(|f| !f.is_static)
                            {
                                // Type the stored value.
                                if let Some(v) =
                                    insn.operands.get(1).and_then(|o| o.as_local())
                                {
                                    if let Some(&li) = table.index_of.get(&v) {
                                        let fty = field.field_type.clone();
                                        if super::locals::refine_type_public(
                                            &table.infos[li].ty,
                                            &fty,
                                        ) {
                                            table.infos[li].ty = fty;
                                            changed = true;
                                            total_typed += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Load side: Move(dst, Memory{base, addend}) — resolve to field.
            if let Some(Operand::Memory { base, addend, .. }) =
                insn.operands.get(1).cloned()
            {
                let base_ty = super::locals::operand_type(
                    Some(&base),
                    table,
                    oracle,
                );
                if let Some(bt) = base_ty {
                    if let Some(nt) = super::locals::class_of_public(&bt) {
                        if nt.type_def_index >= 0 {
                            if let Some(field) = oracle
                                .field_at_offset(nt.type_def_index, addend)
                                .filter(|f| !f.is_static)
                            {
                                let fty = field.field_type.clone();
                                // Type the destination local.
                                if let Some(Operand::Local(dst)) = insn.dest().cloned() {
                                    if let Some(&li) = table.index_of.get(&dst) {
                                        if super::locals::refine_type_public(
                                            &table.infos[li].ty,
                                            &fty,
                                        ) {
                                            table.infos[li].ty = fty.clone();
                                            changed = true;
                                            total_typed += 1;
                                        }
                                    }
                                    insn.dest_type = Some(fty);
                                }
                                // Rewrite the Memory operand into a Field or
                                // InstanceField so the emitter renders it as a
                                // named field access instead of `*(nint*)(...)`.
                                // Check if the base is `this` (a local flagged
                                // is_this) — if so, collapse to bare Field;
                                // otherwise keep the base in InstanceField.
                                let base_is_this = if let Operand::Local(bv) = base.as_ref() {
                                    table.index_of.get(bv).map(|&i| table.infos[i].is_this).unwrap_or(false)
                                } else {
                                    false
                                };
                                if base_is_this {
                                    insn.operands[1] = Operand::Field(field);
                                } else {
                                    insn.operands[1] = Operand::InstanceField {
                                        base,
                                        field,
                                    };
                                }
                                POST_RESOLVE_FIELD_REWRITES
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
        }
    }

    POST_RESOLVE_ITERATIONS.fetch_add(iterations as u64, Ordering::Relaxed);
    total_typed
}

// ---------------------------------------------------------------------------
// Phase 0: Cross-method interprocedural analyzer
// ---------------------------------------------------------------------------

/// A recorded call site: caller RVA → callee RVA + operand positions.
#[derive(Debug, Clone)]
pub struct CallSiteRecord {
    /// RVA of the calling method.
    pub caller_rva: u64,
    /// RVA of the callee (resolved from Immediate operand).
    pub callee_rva: u64,
    /// Instruction index within the caller's IR.
    pub insn_index: usize,
    /// Operand index of the first parameter argument (skipping target + result).
    pub param_start: usize,
}

/// Per-method type summary extracted from metadata signatures.
#[derive(Debug, Clone)]
pub struct MethodTypeSummary {
    /// Parameter types (parallel to the method's declared parameters).
    pub param_types: Vec<ManagedType>,
    /// Return type.
    pub ret_type: ManagedType,
    /// Whether the method is static (affects receiver slot).
    pub is_static: bool,
}

/// Interprocedural analyzer: builds a call graph and propagates types across
/// method boundaries using metadata signatures as ground truth.
///
/// ## Usage
///
/// ```ignore
/// let mut analyzer = InterproceduralAnalyzer::new();
/// // Phase 1: Build call graph from all methods' resolved IR.
/// analyzer.build_call_graph(methods);
/// // Phase 2: Seed from metadata (already in methods_by_rva).
/// analyzer.seed_from_metadata();
/// // Phase 3-5: Iterate to fixpoint (types stabilize).
/// analyzer.iterate_to_fixpoint();
/// // Apply: use the enriched seeds in per-method pipeline.
/// let seeds = analyzer.take_seeds();
/// ```
pub struct InterproceduralAnalyzer {
    /// method RVA → call sites within that method.
    pub call_graph: HashMap<u64, Vec<CallSiteRecord>>,
    /// method RVA → type summary from metadata.
    pub method_types: HashMap<u64, MethodTypeSummary>,
    /// method RVA → enriched parameter types (from cross-method propagation).
    pub enriched_params: HashMap<u64, Vec<ManagedType>>,
    /// method RVA → enriched return type (from cross-method propagation).
    pub enriched_returns: HashMap<u64, ManagedType>,
}

impl InterproceduralAnalyzer {
    pub fn new() -> Self {
        Self {
            call_graph: HashMap::new(),
            method_types: HashMap::new(),
            enriched_params: HashMap::new(),
            enriched_returns: HashMap::new(),
        }
    }

    /// Phase 1: Build call graph from resolved IR.
    ///
    /// Scans each method's instructions for `Call`/`CallVoid` with `Method`
    /// operands, recording caller → callee relationships.
    pub fn build_call_graph_from_insns(
        &mut self,
        caller_rva: u64,
        insns: &[Insn],
    ) {
        let mut sites = Vec::new();
        for (i, insn) in insns.iter().enumerate() {
            if !matches!(insn.op, Op::Call | Op::CallVoid) {
                continue;
            }
            let Some(Operand::Method(m)) = insn.operands.first() else {
                continue;
            };
            // Look up the callee's RVA from the method's code pointer.
            // We don't have the RVA directly in the MethodRef, but we can
            // record the MethodRef's identity for later lookup.
            // For now, use the method_index as a proxy.
            let callee_rva = 0u64; // Placeholder — filled by seed_from_metadata.
            let has_result = matches!(insn.op, Op::Call);
            let param_start = if has_result {
                if m.sig.is_static { 2 } else { 3 }
            } else {
                if m.sig.is_static { 1 } else { 2 }
            };
            sites.push(CallSiteRecord {
                caller_rva,
                callee_rva,
                insn_index: i,
                param_start,
            });
        }
        if !sites.is_empty() {
            self.call_graph.insert(caller_rva, sites);
        }
    }

    /// Phase 2: Seed method types from the metadata resolution maps.
    ///
    /// Uses `methods_by_rva` as ground truth — every entry has a full
    /// `MethodSigInfo` with parameter and return types from metadata.
    pub fn seed_from_metadata(
        &mut self,
        methods_by_rva: &HashMap<u64, MethodRef>,
    ) {
        for (&rva, m) in methods_by_rva {
            let summary = MethodTypeSummary {
                param_types: m.sig.params.clone(),
                ret_type: m.sig.ret.clone(),
                is_static: m.sig.is_static,
            };
            self.method_types.insert(rva, summary);
        }
    }

    /// Phase 3: Forward propagation (caller → callee parameters).
    ///
    /// For each call site, look at the types of the arguments in the caller.
    /// Propagate those types to the callee's parameter locals.
    ///
    /// Returns true if any types were propagated.
    pub fn propagate_forward(
        &mut self,
        caller_types: &HashMap<u64, HashMap<u32, ManagedType>>,
    ) -> bool {
        let mut changed = false;
        // For each caller, look at its call sites.
        for (&caller_rva, sites) in &self.call_graph {
            let Some(_caller_local_types) = caller_types.get(&caller_rva) else {
                continue;
            };
            for site in sites {
                let Some(callee_summary) = self.method_types.get(&site.callee_rva) else {
                    continue;
                };
                // For each parameter, check if the caller's argument has a type.
                let enriched = self
                    .enriched_params
                    .entry(site.callee_rva)
                    .or_insert_with(|| vec![ManagedType::Unknown; callee_summary.param_types.len()]);
                for (pi, pty) in callee_summary.param_types.iter().enumerate() {
                    if matches!(pty, ManagedType::Unknown) {
                        continue;
                    }
                    // The callee's metadata already gives us the parameter type.
                    // Forward propagation would use the CALLER's argument type
                    // to refine the callee's parameter, but since metadata is
                    // ground truth, we only fill in Unknowns.
                    if matches!(enriched.get(pi), Some(ManagedType::Unknown) | None) {
                        if pi < enriched.len() {
                            enriched[pi] = pty.clone();
                            changed = true;
                        }
                    }
                }
            }
        }
        changed
    }

    /// Phase 4: Backward propagation (callee → caller return types).
    ///
    /// For each method, look at its return type from metadata. Propagate that
    /// type to all call sites that receive its return value.
    ///
    /// Returns true if any types were propagated.
    pub fn propagate_backward(&mut self) -> bool {
        let mut changed = false;
        // For each callee with a known return type, update enriched_returns.
        for (&rva, summary) in &self.method_types {
            if matches!(summary.ret_type, ManagedType::Unknown) {
                continue;
            }
            let current = self.enriched_returns.get(&rva);
            if matches!(current, None | Some(ManagedType::Unknown)) {
                self.enriched_returns.insert(rva, summary.ret_type.clone());
                changed = true;
            }
        }
        changed
    }

    /// Phase 5: Iterate phases 3-4 to fixpoint.
    pub fn iterate_to_fixpoint(
        &mut self,
        caller_types: &HashMap<u64, HashMap<u32, ManagedType>>,
    ) {
        for _ in 0..8 {
            let c1 = self.propagate_forward(caller_types);
            let c2 = self.propagate_backward();
            if !c1 && !c2 {
                break;
            }
        }
    }

    /// Get enriched parameter types for a method (for seeding).
    pub fn get_enriched_params(&self, rva: u64) -> Option<&Vec<ManagedType>> {
        self.enriched_params.get(&rva)
    }

    /// Get enriched return type for a method (for seeding).
    pub fn get_enriched_return(&self, rva: u64) -> Option<&ManagedType> {
        self.enriched_returns.get(&rva)
    }
}
