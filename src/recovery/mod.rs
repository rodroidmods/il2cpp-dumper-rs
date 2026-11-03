//! Native → CIL method-body recovery for DummyDLLs (Cpp2IL port).
//!
//! Full pipeline per method, in MethodAnalysisContext.Analyze() order:
//!   lift_arm64 → Cfg::build → stack_analyzer → DominatorInfo
//!   → ssa::build + materialize → locals::create + propagate_types
//!   → resolve::run → optimization loop (propagation ⇄ folding + DCE)
//!   → branch folding → guard removal → arg trimming → ssa::eliminate
//!   → emit::generate → dotnetdll body
//!
//! Any failure returns None; the caller keeps the existing stub body —
//! recovery must never fail a dump (Cpp2IL IlRecovery policy).

pub mod cfg;
pub mod concord;
pub mod conform;
pub mod csharp;
pub mod dominator;
pub mod emit;
pub mod exceptions;
pub mod split_loops;
/// High-variable recovery (Ghidra `merge.cc` / `varmap.cc` / `unify.cc` port):
/// collapses SSA copy chains, resolves phi confluences whose arguments agree,
/// unifies identical pure computations, and merges the names of one register
/// slot's non-overlapping versions. Runs at the end of type propagation, before
/// metadata resolution, so every later pass sees one value per high variable.
pub mod highvar;
/// Statement-tree idiom recognition (dead-end blocks, structure-field
/// propagation). Runs after the structurer, on the finished pseudo-C.
pub mod idioms;
pub mod env_flags;
/// Interprocedural type analysis — propagates types across method call
/// boundaries using metadata signatures as ground truth. Two phases:
/// (1) post-resolve re-propagation within each method, (2) cross-method
/// call graph analysis for enriched seeding.
pub mod interprocedural;
pub mod ir;
pub mod lift;
pub mod lift_arm32;
pub mod lift_x86;
pub mod locals;
pub mod passes;
/// High-level C# CONSTRUCT reconstruction over the finished statement tree
/// (`for` / `using` / `lock` / lambdas / iterators / array initializers).
/// Sibling of `idioms`: that module removes machine noise, this one rebuilds
/// source-level constructs.
pub mod reconstruct;
pub mod resolve;
/// Identity for calls into the IL2CPP runtime itself (static-init guards,
/// dispatch trampolines), which no symbol table can name. See the module docs.
pub mod runtime_helpers;
pub mod ssa;
pub mod stack_analyzer;
pub mod types;
pub mod verify;
pub mod heuristic_types;

/// Per-stage wall-clock telemetry for the recovery pipeline (aggregated over
/// all methods; printed by the drivers when `ILR_TIMING=1`). Atomics keep the
/// parallel loops contention-cheap.
pub mod timing {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static LIFT: AtomicU64 = AtomicU64::new(0);
    pub static CFG: AtomicU64 = AtomicU64::new(0);
    pub static STACK: AtomicU64 = AtomicU64::new(0);
    pub static SSA: AtomicU64 = AtomicU64::new(0);
    pub static PROPAGATE: AtomicU64 = AtomicU64::new(0);
    pub static RESOLVE: AtomicU64 = AtomicU64::new(0);
    pub static GUARDS: AtomicU64 = AtomicU64::new(0);
    pub static OPT: AtomicU64 = AtomicU64::new(0);
    pub static STRUCTURAL: AtomicU64 = AtomicU64::new(0);
    pub static ELIMINATE: AtomicU64 = AtomicU64::new(0);
    pub static RENDER: AtomicU64 = AtomicU64::new(0);
    pub static METHODS: AtomicU64 = AtomicU64::new(0);
    pub static METHOD_NANOS: AtomicU64 = AtomicU64::new(0);
    pub static SLOW_100MS: AtomicU64 = AtomicU64::new(0);
    pub static SLOW_500MS: AtomicU64 = AtomicU64::new(0);
    pub static SLOW_1S: AtomicU64 = AtomicU64::new(0);
    pub static DEADLINE_HITS: AtomicU64 = AtomicU64::new(0);
    /// Calls whose operand list `locals::align_call_arguments` rebuilt into
    /// declared-parameter order (integer and FP classes placed by type).
    pub static ALIGNED_ARGS: AtomicU64 = AtomicU64::new(0);
    /// Indirect calls whose callee resolved to a METHOD (virtual dispatch,
    /// interface dispatch, delegate invoke) rather than staying `fnptr_0x…`.
    pub static VIRTUAL_CALLS: AtomicU64 = AtomicU64::new(0);
    /// Indirect calls the vtable resolver *reached* and could not name — the
    /// denominator for `VIRTUAL_CALLS`. A run where this is huge and the
    /// numerator is zero means the slot arithmetic or the receiver types are
    /// wrong, which is invisible from the dump alone (both ends render as an
    /// unnamed call).
    pub static VC_MISSES: AtomicU64 = AtomicU64::new(0);
    /// Calls to an identical-code-folded address whose member the CALL SITE
    /// named — its `NewObj(T)` klass or the receiver's class. These names are
    /// evidenced, not canonical.
    pub static SHARED_PICKED: AtomicU64 = AtomicU64::new(0);
    /// Calls to a folded address with no call-site evidence at all. The name in
    /// the output is the canonical member of the set, i.e. a GUESS — this is the
    /// size of the fabricated-declaring-type population, and it should never be
    /// assumed to be zero.
    pub static SHARED_AMBIGUOUS: AtomicU64 = AtomicU64::new(0);
    /// Why `SHARED_AMBIGUOUS` calls could not be named. The four reasons call
    /// for completely different fixes, and the total alone cannot tell them
    /// apart: no receiver operand, a receiver whose class was never recovered,
    /// a recovered class that does not declare the member, and a class that
    /// declares MORE THAN ONE of the merged members (so the name is still not
    /// determined).
    pub static SHARED_NO_RECV: AtomicU64 = AtomicU64::new(0);
    pub static SHARED_NO_CLASS: AtomicU64 = AtomicU64::new(0);
    pub static SHARED_NO_MEMBER: AtomicU64 = AtomicU64::new(0);
    pub static SHARED_MULTI: AtomicU64 = AtomicU64::new(0);
    /// Locals whose STORAGE address escapes — the one population our IR can
    /// express as Ghidra's `isAddrForce()` ("this value must live at an
    /// address"). Counted rather than assumed: the predicate was proposed as
    /// the fix for duplicate/never-read bindings, and a port of a RULE with no
    /// population is a no-op, so the size of the population has to be a number.
    pub static ADDR_ESCAPES: AtomicU64 = AtomicU64::new(0);
    /// `MissingAbiArg` placeholders written by `locals::align_call_arguments`,
    /// bucketed by the ARGUMENT POSITION they land in: `[0]` is the receiver,
    /// `[k]` for `k` in `1..=9` is declared parameter `k-1`, and `[10]` is
    /// parameter 9 or beyond.
    ///
    /// The SHAPE of this histogram is the diagnosis. Mass at LOW indices means
    /// a call site whose register dump was unreadable, or a value with no live
    /// register to name. Mass at HIGH indices means the ABI ran out of argument
    /// registers and the value went on the STACK, which no lifter records at
    /// all. Those two need completely different fixes and the total alone
    /// cannot tell them apart.
    pub static ABI_MISS_BY_POS: [AtomicU64; 11] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    /// Of the `ABI_MISS_BY_POS` placeholders, how many wanted a register the
    /// ABI's argument file does not have — a `stack_-N` slot, or (on x64) an
    /// `x6`/`x7` that SysV and MSVC pass in memory. Both mean "the ABI ran out
    /// of registers"; the remainder are registers that exist but held no value
    /// this call site could name.
    pub static ABI_MISS_STACK: AtomicU64 = AtomicU64::new(0);
    /// Call sites identified as the IL2CPP static-constructor guard by
    /// `runtime_helpers::detect`. They render as `IL2CPP_ClassInit(...)`
    /// instead of `fn_<address>(...)`, so this is the number of `fn_` sites
    /// removed by that classification.
    pub static RT_CLASSINIT_SITES: AtomicU64 = AtomicU64::new(0);
    /// Call-site argument locals that received a type from the callee's
    /// declared parameter signature (back-propagation in `propagate_types`).
    pub static CALLSITE_PARAM_TYPED: AtomicU64 = AtomicU64::new(0);
    /// Immediate call targets that `align_call_arguments` resolved to a method
    /// and embedded as `Operand::Method` so downstream passes see the signature.
    pub static CALLSITE_EMBEDDED: AtomicU64 = AtomicU64::new(0);
    /// Parameters seeded with a non-Unknown type from metadata at method entry
    /// (includes `this` typed with the declaring class).
    pub static PARAMS_SEEDED_TYPED: AtomicU64 = AtomicU64::new(0);
    /// Parameters seeded at method entry (total denominator).
    pub static PARAMS_SEEDED_TOTAL: AtomicU64 = AtomicU64::new(0);

    /// Render sub-stages (parallel-sum microseconds). `R_TOTAL` is the whole
    /// `render_method_csharp_full_chains` call; the rest are segments of it,
    /// so their sum plus loop bookkeeping is `R_TOTAL`. Printed by `report()`
    /// as `[render-split]` so a runtime regression can be attributed without
    /// guessing which of the three walks or two previews got slower.
    pub static R_TOTAL: AtomicU64 = AtomicU64::new(0);
    /// Duplicated listing: clone + shared-tail split + CFG/loops of both
    /// listings + EH block maps.
    pub static R_DUP: AtomicU64 = AtomicU64::new(0);
    /// Binding-discovery walk (throwaway body, keeps `bound_calls`).
    pub static R_DISCOVERY: AtomicU64 = AtomicU64::new(0);
    /// Structured trial: MethodCtx + emit_struct on the duplicated listing.
    pub static R_TRIAL: AtomicU64 = AtomicU64::new(0);
    /// Address-walker body emission (only when the trial was not jump-free).
    pub static R_WALK: AtomicU64 = AtomicU64::new(0);
    /// Both `preview_stats` runs (late-pass clones).
    pub static R_PREVIEW: AtomicU64 = AtomicU64::new(0);
    /// MethodCtx::build sub-steps (parallel-summed).
    pub static B_DEFS: AtomicU64 = AtomicU64::new(0);
    pub static B_REACHING: AtomicU64 = AtomicU64::new(0);
    pub static B_VARREADS: AtomicU64 = AtomicU64::new(0);
    pub static B_READCNT: AtomicU64 = AtomicU64::new(0);
    pub static B_ADDR: AtomicU64 = AtomicU64::new(0);
    pub static B_SHARED: AtomicU64 = AtomicU64::new(0);
    pub static B_REFUSED: AtomicU64 = AtomicU64::new(0);
    pub static B_NAMES: AtomicU64 = AtomicU64::new(0);
    /// The real `MethodCtx::build` (reaching-definition fixpoint over the
    /// original listing) between the discovery walk and the trial. The
    /// discovery walk builds an identical ctx for itself, so this is the
    /// duplicate-cost candidate.
    pub static R_CTX: AtomicU64 = AtomicU64::new(0);
    /// The real late-pass pipeline + header/body emission + trailing filters.
    pub static R_LATE: AtomicU64 = AtomicU64::new(0);
    /// Late-segment buckets: `R_L1` = header setup + rewrites/catalog/cleanup/
    /// idioms/reconstruct, `R_L2` = type maps + sugar passes, `R_L3` = the
    /// structural cleanup batch (sanitize/branch_repair/goto/for/guards),
    /// `R_L4` = naming, const/role/adjacent/liveness folds + kinds inference +
    /// `declare_locals`, `R_L5` = final emission and text filters.
    pub static R_L1: AtomicU64 = AtomicU64::new(0);
    pub static R_L2: AtomicU64 = AtomicU64::new(0);
    pub static R_L3: AtomicU64 = AtomicU64::new(0);
    pub static R_L4: AtomicU64 = AtomicU64::new(0);
    pub static R_L5: AtomicU64 = AtomicU64::new(0);

    fn ms(a: &AtomicU64) -> f64 {
        a.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Human-readable per-stage totals (microseconds are accumulated).
    pub fn report() -> String {
        let m = METHODS.load(Ordering::Relaxed);
        let total = ms(&METHOD_NANOS);
        let mut s = format!(
            "[timing] methods={m} total_pipeline_ms={total:.0} avg_ms={:.1} \
             slow>100ms={} >500ms={} >1s={} deadline_hits={} aligned_call_args={} \
             virtual_calls={} vc_misses={} folded_picked={} folded_guessed={} \
             folded_no_recv={} folded_no_class={} folded_no_member={} folded_multi={} \
             addr_escapes={} rt_classinit={} callsite_embedded={} callsite_params_typed={} \
             params_seeded_typed={}/{} \
             | lift={:.0} cfg={:.0} stack={:.0} ssa={:.0} propagate={:.0} \
             resolve={:.0} guards={:.0} opt={:.0} structural={:.0} eliminate={:.0} \
             render={:.0} (ms total)",
            if m > 0 { total / m as f64 } else { 0.0 },
            SLOW_100MS.load(Ordering::Relaxed),
            SLOW_500MS.load(Ordering::Relaxed),
            SLOW_1S.load(Ordering::Relaxed),
            DEADLINE_HITS.load(Ordering::Relaxed),
            ALIGNED_ARGS.load(Ordering::Relaxed),
            VIRTUAL_CALLS.load(Ordering::Relaxed),
            VC_MISSES.load(Ordering::Relaxed),
            SHARED_PICKED.load(Ordering::Relaxed),
            SHARED_AMBIGUOUS.load(Ordering::Relaxed),
            SHARED_NO_RECV.load(Ordering::Relaxed),
            SHARED_NO_CLASS.load(Ordering::Relaxed),
            SHARED_NO_MEMBER.load(Ordering::Relaxed),
            SHARED_MULTI.load(Ordering::Relaxed),
            ADDR_ESCAPES.load(Ordering::Relaxed),
            RT_CLASSINIT_SITES.load(Ordering::Relaxed),
            CALLSITE_EMBEDDED.load(Ordering::Relaxed),
            CALLSITE_PARAM_TYPED.load(Ordering::Relaxed),
            PARAMS_SEEDED_TYPED.load(Ordering::Relaxed),
            PARAMS_SEEDED_TOTAL.load(Ordering::Relaxed),
            ms(&LIFT),
            ms(&CFG),
            ms(&STACK),
            ms(&SSA),
            ms(&PROPAGATE),
            ms(&RESOLVE),
            ms(&GUARDS),
            ms(&OPT),
            ms(&STRUCTURAL),
            ms(&ELIMINATE),
            ms(&RENDER),
        );
        if R_TOTAL.load(Ordering::Relaxed) > 0 {
            s.push_str(&format!(
                "\n[render-split] total={:.0} dup={:.0} discovery={:.0} trial={:.0} \
                 walk={:.0} preview={:.0} late={:.0} (parallel-sum ms)",
                ms(&R_TOTAL),
                ms(&R_DUP),
                ms(&R_DISCOVERY),
                ms(&R_TRIAL),
                ms(&R_WALK),
                ms(&R_PREVIEW),
                ms(&R_LATE),
            ));
            s.push_str(&format!(
                "\n[ctx] method_ctx={:.0}ms (parallel-sum)",
                ms(&R_CTX)
            ));
            s.push_str(&format!(
                "\n[ctx-split] defs={:.0} reaching={:.0} varreads={:.0} readcnt={:.0} addr={:.0} shared={:.0} refused={:.0} names={:.0} (parallel-sum ms)",
                ms(&B_DEFS),
                ms(&B_REACHING),
                ms(&B_VARREADS),
                ms(&B_READCNT),
                ms(&B_ADDR),
                ms(&B_SHARED),
                ms(&B_REFUSED),
                ms(&B_NAMES)
            ));
            s.push_str(&format!(
                "\n[late-split] l1={:.0} l2={:.0} l3={:.0} l4={:.0} l5={:.0} (parallel-sum ms)",
                ms(&R_L1),
                ms(&R_L2),
                ms(&R_L3),
                ms(&R_L4),
                ms(&R_L5),
            ));
            s.push_str(&format!(
                "\n{}",
                crate::recovery::reconstruct::pass_timing_report(12)
            ));
            s.push_str(&format!(
                "\n{}",
                crate::recovery::reconstruct::il2cpp_cleanup::cleanup_timing_report(8)
            ));
            s.push_str(&crate::recovery::csharp::dangle_report());
            let ll = crate::recovery::reconstruct::label_loss_report(12);
            if !ll.is_empty() {
                s.push('\n');
                s.push_str(&ll);
            }
            let l3 = crate::recovery::csharp::l3_timing_report(14);
            if !l3.is_empty() {
                s.push('\n');
                s.push_str(&l3);
            }
        }
        // A second line only when there is something to report; see
        // `ABI_MISS_BY_POS` for how to read the bucket shape.
        let a: Vec<u64> = ABI_MISS_BY_POS
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let abi_total: u64 = a.iter().sum();
        if abi_total > 0 {
            s.push_str(&format!(
                "\n[abi-miss] total={abi_total} beyond_regfile={} by_pos=[recv={} p0={} \
                 p1={} p2={} p3={} p4={} p5={} p6={} p7={} p8={} p9+={}]",
                ABI_MISS_STACK.load(Ordering::Relaxed),
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                a[6],
                a[7],
                a[8],
                a[9],
                a[10],
            ));
        }
        // Guarded field-address seeds, always reported (a zero applied count is
        // the evidence the seed ran and refused, not a missing measurement).
        // Per fixpoint visit, not distinct sites — see `locals::field_addr_seed_counts`.
        {
            let (applied, ref_enum, no_layout, unproven) = super::locals::field_addr_seed_counts();
            s.push_str(&format!(
                "\n[field-addr] applied={applied} refused_ref_enum={ref_enum} refused_no_layout={no_layout} refused_unproven={unproven}"
            ));
        }
        // Interprocedural type propagation impact.
        s.push_str(&super::interprocedural::report());
        s
    }
}

// Convenience counters used inside recover_ir_deadline.
use timing as tm;

#[inline]
fn tm_add(c: &std::sync::atomic::AtomicU64, started: std::time::Instant) {
    c.fetch_add(
        started.elapsed().as_micros() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

use crate::executor::Il2CppExecutor;
use crate::il2cpp::base::Il2Cpp;
use crate::il2cpp::enums::Il2CppTypeEnum;
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::structures::Il2CppType;
use ir::{Insn, Op, Operand};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use types::{FieldRef, ManagedType, MethodRef, NamedType, PrimKind};

// ---------------------------------------------------------------------------
// il2cpp ↔ ManagedType bridge (Resolution-independent)
// ---------------------------------------------------------------------------

/// HFA component fields of a value type, as IR `FieldRef`s.
///
/// `hfa_float_components` gives the same layout as NAMES; retyping a call
/// return needs the full field reference so the renderer can emit
/// `result.x` and type propagation can see `float`.
pub fn hfa_field_refs(
    nt: &NamedType,
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
) -> Vec<FieldRef> {
    let tdi = nt.type_def_index;
    if tdi < 0 {
        return Vec::new();
    }
    let Some(td) = metadata.type_defs.get(tdi as usize) else {
        return Vec::new();
    };
    if !td.is_value_type() || td.is_enum() {
        return Vec::new();
    }
    let declaring = if nt.namespace.is_empty() {
        nt.name.clone()
    } else {
        format!("{}.{}", nt.namespace, nt.name)
    };
    let mut out: Vec<FieldRef> = Vec::new();
    for fi in 0..td.field_count as usize {
        let global_fi = td.field_start as usize + fi;
        let Some(field_def) = metadata.field_defs.get(global_fi) else {
            return Vec::new();
        };
        let Some(t) = il2cpp.types.get(field_def.type_index as usize) else {
            return Vec::new();
        };
        if (t.attrs & 0x0010) != 0 {
            continue;
        }
        match Il2CppTypeEnum::from_u8(t.type_enum) {
            Some(Il2CppTypeEnum::R4) => {
                let name = metadata
                    .get_string_from_index(field_def.name_index)
                    .unwrap_or_else(|_| format!("f{}", out.len()));
                out.push(FieldRef {
                    type_def_index: tdi,
                    field_index: global_fi as u32,
                    name,
                    declaring_type_name: declaring.clone(),
                    field_type: ManagedType::Primitive(PrimKind::R4),
                    is_static: false,
                });
                if out.len() > 4 {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    out
}

/// Convert an `Il2CppType` into the analysis-side `ManagedType`.
pub fn managed_type_of(
    t: &Il2CppType,
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
    depth: usize,
) -> ManagedType {
    managed_type_of_gn(t, metadata, il2cpp, depth, None)
}

/// Same, with a generic-name table (datapoint → display name).
pub fn managed_type_of_gn(
    t: &Il2CppType,
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
    depth: usize,
    gnames: Option<&HashMap<u64, String>>,
) -> ManagedType {
    if depth > 8 {
        return ManagedType::Unknown;
    }
    let te = Il2CppTypeEnum::from_u8(t.type_enum);
    let base = match te {
        Some(Il2CppTypeEnum::Void) => ManagedType::Primitive(PrimKind::Void),
        Some(Il2CppTypeEnum::Boolean) => ManagedType::Primitive(PrimKind::Boolean),
        Some(Il2CppTypeEnum::Char) => ManagedType::Primitive(PrimKind::Char),
        Some(Il2CppTypeEnum::I1) => ManagedType::Primitive(PrimKind::I1),
        Some(Il2CppTypeEnum::U1) => ManagedType::Primitive(PrimKind::U1),
        Some(Il2CppTypeEnum::I2) => ManagedType::Primitive(PrimKind::I2),
        Some(Il2CppTypeEnum::U2) => ManagedType::Primitive(PrimKind::U2),
        Some(Il2CppTypeEnum::I4) => ManagedType::Primitive(PrimKind::I4),
        Some(Il2CppTypeEnum::U4) => ManagedType::Primitive(PrimKind::U4),
        Some(Il2CppTypeEnum::I8) => ManagedType::Primitive(PrimKind::I8),
        Some(Il2CppTypeEnum::U8) => ManagedType::Primitive(PrimKind::U8),
        Some(Il2CppTypeEnum::R4) => ManagedType::Primitive(PrimKind::R4),
        Some(Il2CppTypeEnum::R8) => ManagedType::Primitive(PrimKind::R8),
        Some(Il2CppTypeEnum::I) => ManagedType::IntPtr,
        Some(Il2CppTypeEnum::U) => ManagedType::IntPtr,
        Some(Il2CppTypeEnum::String) => ManagedType::String,
        Some(Il2CppTypeEnum::Object) => ManagedType::Object,
        Some(Il2CppTypeEnum::TypedByRef) => opaque_named("System", "TypedReference", -1),
        Some(Il2CppTypeEnum::Class) | Some(Il2CppTypeEnum::ValueType) => {
            named_of_klass_index(t.klass_index(), metadata)
        }
        Some(Il2CppTypeEnum::SzArray) | Some(Il2CppTypeEnum::Array) => {
            let elem = il2cpp
                .get_il2cpp_type(t.type_ptr())
                .map(|et| Box::new(managed_type_of(&et, metadata, il2cpp, depth + 1)))
                .unwrap_or_else(|| Box::new(ManagedType::Object));
            ManagedType::SzArray(elem)
        }
        Some(Il2CppTypeEnum::Ptr) => {
            let inner = il2cpp
                .get_il2cpp_type(t.type_ptr())
                .map(|et| Box::new(managed_type_of(&et, metadata, il2cpp, depth + 1)))
                .unwrap_or_else(|| Box::new(ManagedType::Unknown));
            ManagedType::Pointer(inner)
        }
        Some(Il2CppTypeEnum::ByRef) => {
            let inner = il2cpp
                .get_il2cpp_type(t.type_ptr())
                .map(|et| Box::new(managed_type_of(&et, metadata, il2cpp, depth + 1)))
                .unwrap_or_else(|| Box::new(ManagedType::Unknown));
            ManagedType::ByRef(inner)
        }
        Some(Il2CppTypeEnum::GenericInst) => {
            // resolved display name (e.g. "List<int>") when the table is
            // provided by the caller
            match gnames.and_then(|g| g.get(&t.datapoint)) {
                Some(n) => ManagedType::Opaque(n.clone()),
                None => ManagedType::Opaque("generic".into()),
            }
        }
        Some(Il2CppTypeEnum::Var) => ManagedType::Opaque("T".into()),
        Some(Il2CppTypeEnum::MVar) => ManagedType::Opaque("M".into()),
        _ => ManagedType::Unknown,
    };
    // A `ref`/`out` parameter carries the BYREF BIT on its element type
    // (`ref float` has type_enum R4 with bit 29 set), not a separate
    // BYREF type_enum. Ignoring the bit classified such parameters as FP
    // scalars, so they were slotted into vsN instead of xN and never received
    // their parameter name (`*(nint*)(in_x2)` for `ref float inTotalProb`).
    if t.byref != 0 && !matches!(base, ManagedType::ByRef(_)) {
        ManagedType::ByRef(Box::new(base))
    } else {
        base
    }
}

fn named_of_klass_index(idx: u64, metadata: &Metadata) -> ManagedType {
    let i = idx as usize;
    if i >= metadata.type_defs.len() {
        return ManagedType::Object;
    }
    let td = &metadata.type_defs[i];
    let namespace = metadata
        .get_string_from_index(td.namespace_index)
        .unwrap_or_default();
    let name = metadata
        .get_string_from_index(td.name_index)
        .unwrap_or_default();
    ManagedType::Class(Arc::new(NamedType {
        namespace,
        name,
        assembly: None,
        type_def_index: i as i32,
    }))
}

fn opaque_named(ns: &str, name: &str, _tdi: i32) -> ManagedType {
    ManagedType::Opaque(format!("{ns}.{name}"))
    // note: tdi unused here by design
}

/// Metadata walk deciding whether a value type is an HFA float aggregate
/// (AAPCS64 §6.4.2: 1–4 members, all 32/64-bit floats) and, if so, the
/// names of its float fields in declaration (== layout) order.
///
/// Engine math types (Vector2/3/4, Color, Quaternion, ...) are passed
/// by-value in consecutive v-registers (Vector3 → s0,s1,s2); the field
/// names also let the renderer print `rootPosition.x` instead of `v26`.
/// Returns empty when the type is not a pure float aggregate — the caller
/// then treats the parameter as a GPR/memory argument.
pub fn hfa_float_components(tdi: i32, metadata: &Metadata, il2cpp: &Il2Cpp) -> Vec<String> {
    if tdi < 0 {
        return Vec::new();
    }
    let Some(td) = metadata.type_defs.get(tdi as usize) else {
        return Vec::new();
    };
    if !td.is_value_type() || td.is_enum() {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    for fi in 0..td.field_count as usize {
        let global_fi = td.field_start as usize + fi;
        let Some(field_def) = metadata.field_defs.get(global_fi) else {
            return Vec::new();
        };
        let Some(t) = il2cpp.types.get(field_def.type_index as usize) else {
            return Vec::new();
        };
        if (t.attrs & 0x0010) != 0 {
            continue; // static fields don't participate in the value layout
        }
        match Il2CppTypeEnum::from_u8(t.type_enum) {
            // 32-bit floats only: engine structs are pure R4; an R8 field
            // would need d-register packing handled separately (rare, and
            // a wrong v-register assignment is worse than no binding).
            Some(Il2CppTypeEnum::R4) => {
                let name = metadata
                    .get_string_from_index(field_def.name_index)
                    .unwrap_or_else(|_| format!("f{}", out.len()));
                out.push(name);
                if out.len() > 4 {
                    return Vec::new();
                }
            }
            _ => return Vec::new(), // mixed layout → not an HFA
        }
    }
    if out.is_empty() { Vec::new() } else { out }
}

/// Does an ARM32 method returning `ret` deliver its result through a
/// caller-allocated buffer (the AAPCS32 §5.4 hidden result pointer)?
///
/// A Composite Type result is returned in `r0` only while it is 4 bytes or
/// smaller; a larger one is written through a buffer whose address the caller
/// passes as an EXTRA FIRST argument. `r0` is then the BUFFER, `this` is `r1`,
/// and the declared parameters start at `r2` — which is exactly the shift that
/// `locals::create` applies when `SeedInfo::ret_hidden_buffer` is set.
///
/// Verified native (ARM32): `Transform.get_position` @0x15BA630 return `Vector3`
/// begins `mov r4, r0` (the buffer), `mov r0, r1` (this), `mov r1, &local`, and
/// ends `vstr d16, [r4]`, while `get_worldToLocalMatrix` @0x15BB95C copies a
/// 64-byte Matrix4x4 into `r0` the same way.
///
/// The size comes from the metadata's TypeDefinitionSizes (the same table the
/// struct generator reads). An UNKNOWN size (0) REFUSES the shift: guessing
/// here would move `this` for every method whose return type the size table
/// does not cover, turning a narrow fix into a broad regression.
pub fn arm32_ret_uses_hidden_buffer(
    ret: &ManagedType,
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
) -> bool {
    let ManagedType::Class(nt) = ret else {
        return false;
    };
    let tdi = nt.type_def_index;
    if tdi < 0 {
        return false;
    }
    let Some(td) = metadata.type_defs.get(tdi as usize) else {
        return false;
    };
    if !td.is_value_type() || td.is_enum() {
        return false;
    }
    let size = arm32_value_type_size(tdi, metadata, il2cpp).unwrap_or(0);
    size > 4
}

/// Managed value-type payload size on ARM32. `TypeDefinitionSizes::instance_size`
/// includes the 8-byte object header even for value types; AAPCS32 argument
/// classification needs the inline payload size (for example, the four shorts
/// in `AimShootProperties` occupy 8 bytes, while the metadata instance size is
/// 16). Unknown layouts return `None` rather than shifting later arguments on
/// a guessed size.
pub fn arm32_value_type_size(tdi: i32, metadata: &Metadata, il2cpp: &Il2Cpp) -> Option<usize> {
    if tdi < 0 {
        return None;
    }
    let td = metadata.type_defs.get(tdi as usize)?;
    if !td.is_value_type() || td.is_enum() {
        return None;
    }
    let instance_size = il2cpp
        .type_definition_sizes
        .get(tdi as usize)?
        .instance_size as usize;
    instance_size.checked_sub(8).filter(|&size| size > 0)
}

/// Metadata-backed lookups for the typing fixpoint.
///
/// `field_at_offset` is called for every Memory-based Move on every fixpoint
/// iteration; results are memoized (the metadata never changes mid-run),
/// turning the repeated chain-walk into one lookup per (type, offset) pair.
struct BridgeOracle<'a> {
    metadata: &'a Metadata,
    il2cpp: &'a Il2Cpp,
    chains: &'a HashMap<i32, Vec<i32>>,
    cache: std::cell::RefCell<HashMap<(i32, i64), Option<FieldRef>>>,
    /// Absolute static-storage address → global field index, from the FieldInfo
    /// metadata-usage scan. Consulted by [`TypeOracle::static_field_at`] so a
    /// bare-constant static access (`*(nint*)(0x63a… + off)`, the census's
    /// `staticabs`/`asingleton` families) resolves to `Type.field`.
    field_index_by_addr: &'a HashMap<u64, u32>,
    /// Global field index → owning TypeDef index (O(1) static-field naming).
    field_owner_tdi: &'a HashMap<u32, i32>,
}

impl<'a> locals::TypeOracle for BridgeOracle<'a> {
    fn field_at_offset(&self, tdi: i32, offset: i64) -> Option<FieldRef> {
        if tdi < 0 || tdi as usize >= self.metadata.type_defs.len() || offset < 0 {
            return None;
        }
        if let Some(hit) = self.cache.borrow().get(&(tdi, offset)) {
            return hit.clone();
        }
        // precomputed base chain (MonoBehaviour → … → Object)
        let empty = Vec::new();
        let chain = self.chains.get(&tdi).unwrap_or(&empty);
        let mut result = None;
        for &tdi_walk in chain {
            if let Some(f) = self.field_at_offset_single(tdi_walk, offset) {
                result = Some(f);
                break;
            }
        }
        // `ILR_FIELDDBG=<tdi>`: trace one class's offset→field lookups, and
        // alongside the chain-walked result show the SELF-ONLY lookup
        // (`field_at_offset_single`) and the chain length. A load whose base
        // is class-typed but whose destination stays `IntPtr` means the walk
        // returned `None`; this separates the three causes — no field at that
        // offset (`both NONE`), a field declared on a base class that the
        // chain is not reaching (`self Some, walk NONE`, i.e. `chains_len` or
        // the chain itself is wrong), or a field whose type is a plain
        // pointer. The empty-chain case is exactly how the arm32 `field_at_offset`
        // path was silently dead while `field_at_offset_single` still worked.
        if let Ok(v) = std::env::var("ILR_FIELDDBG") {
            if v.trim().parse::<i32>().ok() == Some(tdi) {
                let show = |o: &Option<FieldRef>| match o {
                    Some(f) => format!("{} : {:?}", f.name, f.field_type),
                    None => "NONE".to_string(),
                };
                let single = self.field_at_offset_single(tdi, offset);
                let chain_len = self.chains.get(&tdi).map(Vec::len).unwrap_or(0);
                eprintln!(
                    "[fielddbg] tdi {tdi} off {offset} -> {} (self-only: {}, chain_len: {chain_len})",
                    show(&result),
                    show(&single)
                );
            }
        }
        self.cache
            .borrow_mut()
            .insert((tdi, offset), result.clone());
        result
    }

    /// Resolve an ABSOLUTE static-storage address to the static field it names.
    ///
    /// `resolve::run_with_fields` names the form where the class's static
    /// storage block is materialised through a FieldInfo metadata-usage slot.
    /// The residue this covers is the form where that address is a bare constant
    /// — `*(nint*)(0x63a… + off)` — which is ~50% of the arm64 dump's methods.
    /// The FieldInfo usage scan already recorded every such slot in
    /// `field_index_by_addr`, so the lookup is O(1); the FieldRef is marked
    /// `is_static` so the renderer spells it `Type.field` from the declaring
    /// type alone (no receiver), which is why nothing is dropped.
    fn static_field_at(&self, addr: i64) -> Option<FieldRef> {
        if addr < 0 {
            return None;
        }
        let fi = *self.field_index_by_addr.get(&(addr as u64))?;
        let tdi = *self.field_owner_tdi.get(&fi)?;
        let td = self.metadata.type_defs.get(tdi as usize)?;
        let fdef = self.metadata.field_defs.get(fi as usize)?;
        let name = self
            .metadata
            .get_string_from_index(fdef.name_index)
            .unwrap_or_default();
        let field_type = self
            .il2cpp
            .types
            .get(fdef.type_index as usize)
            .map(|t| managed_type_of(t, self.metadata, self.il2cpp, 0))
            .unwrap_or(ManagedType::Unknown);
        let declaring_type_name = self
            .metadata
            .get_string_from_index(td.name_index)
            .unwrap_or_default();
        Some(FieldRef {
            type_def_index: tdi,
            field_index: fi,
            name,
            declaring_type_name,
            field_type,
            is_static: true,
        })
    }

    /// Value-type-ness from the `Il2CppTypeDefinition` bitfield: a struct is
    /// `is_value_type && !is_enum`. `None` for an unknown `tdi` — unknown
    /// means refuse, never guess.
    ///
    /// This is the narrowest metadata accessor the field-address guard needs:
    /// it answers one bit about one TypeDef, reads nothing else, and changes
    /// no existing lookup. Generic instantiations need no special case here —
    /// a `GenericInst` field type never reaches this as a `Class` with a real
    /// `tdi` (`managed_type_of` models it as `Opaque`), so such fields refuse
    /// in the guard's no-layout bucket by construction.
    fn is_struct_value_type(&self, tdi: i32) -> Option<bool> {
        if tdi < 0 {
            return None;
        }
        let td = self.metadata.type_defs.get(tdi as usize)?;
        Some(td.is_value_type() && !td.is_enum())
    }
}

impl<'a> BridgeOracle<'a> {
    fn field_at_offset_single(&self, tdi: i32, offset: i64) -> Option<FieldRef> {
        if tdi < 0 || tdi as usize >= self.metadata.type_defs.len() {
            return None;
        }
        let td = &self.metadata.type_defs[tdi as usize];
        let field_start = td.field_start as usize;
        let field_count = td.field_count as usize;
        for fi in 0..field_count {
            let global_fi = field_start + fi;
            let Some(field_def) = self.metadata.field_defs.get(global_fi) else {
                continue;
            };
            let is_static = (self
                .il2cpp
                .types
                .get(field_def.type_index as usize)
                .map(|t| t.attrs)
                .unwrap_or(0)
                & 0x0010)
                != 0;
            // STATIC fields do not live at instance offsets: their "offset"
            // is an index into the per-class static-storage block, where the
            // FIRST static field sits at 0. Matching them against an
            // instance-offset query fabricated member names for klass-pointer
            // loads — `LDR x8, [x0]` loads the object's `Il2CppClass*` (always
            // at instance offset 0), so `field_at_offset(XmlSchemaInference, 0)`
            // returned the static field `ST_boolean` (static offset 0x0) and
            // `*(nint*)(*(nint*)(this) + 302)` rendered as the provably wrong
            // `*(nint*)(XmlSchemaInference.ST_boolean + 302)`. Static reads
            // already have an exact mechanism of their own —
            // `resolve_static_fields`, keyed on the FieldInfo metadata-usage
            // slot address, which knows the static-storage pointer when it
            // sees one instead of guessing it from a colliding offset.
            if is_static {
                continue;
            }
            // Instance-offset convention: pass the TypeDef's own value-type-ness
            // so struct field offsets are STRUCT-relative. Every other consumer
            // of `get_field_offset_from_index` (`decompiler.rs` field-offset
            // comments, `dummy_assembly_generator.rs`) already passes
            // `type_def.is_value_type()`; this lookup passed a hardcoded
            // `false`, so a value type's fields compared header-included raw
            // offsets (e.g. 16/20/24) against struct-relative native offsets
            // (0/4/8) and NEVER matched — every `[&struct + sub]` subfield
            // silently fell back to `IntPtr`. Reference types are unaffected
            // (`is_value_type()` is false, identical input). Enum `value__`
            // moves 16→0, which is correct under native semantics: a
            // `Class(enum)`-typed local that is really an enum address (from an
            // `ldflda`-shaped sweep) reads its value at +0.
            let raw_offset = self.il2cpp.get_field_offset_from_index(
                tdi as usize,
                fi,
                global_fi,
                td.is_value_type(),
                is_static,
            );
            if raw_offset == offset as i32 {
                let name = self
                    .metadata
                    .get_string_from_index(field_def.name_index)
                    .unwrap_or_else(|_| format!("field_{global_fi}"));
                let fty = self
                    .il2cpp
                    .types
                    .get(field_def.type_index as usize)
                    .map(|t| managed_type_of(t, self.metadata, self.il2cpp, 0))
                    .unwrap_or(ManagedType::Unknown);
                let declaring = NamedType {
                    namespace: String::new(),
                    name: self
                        .metadata
                        .get_string_from_index(td.name_index)
                        .unwrap_or_default(),
                    assembly: None,
                    type_def_index: tdi,
                };
                let _ = declaring;
                return Some(FieldRef {
                    type_def_index: tdi,
                    field_index: global_fi as u32,
                    name,
                    declaring_type_name: self
                        .metadata
                        .get_string_from_index(td.name_index)
                        .unwrap_or_default(),
                    field_type: fty,
                    is_static,
                });
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// RecoveryContext
// ---------------------------------------------------------------------------

/// Everything the pipeline needs for one dump; built once, shared via Arc.
pub struct RecoveryContext {
    pub maps: resolve::ResolveMaps,
    pub max_method_bytes: usize,
    pub arch_is_arm64: bool,
    pub is_x64: bool,
    pub is_arm32: bool,
    /// Which x86-64 calling convention the ARGUMENT REGISTERS are named for.
    ///
    /// This has to be detected, not assumed. The lifter maps physical registers
    /// onto the shared positional vocabulary (`x0`…`x5` = argument 0…5), and the
    /// two x64 ABIs put those arguments in DIFFERENT registers:
    ///
    /// * SysV (Linux/macOS ELF/Mach-O): `rdi, rsi, rdx, rcx, r8, r9`
    /// * MSVC (Windows PE):            `rcx, rdx, r8,  r9`
    ///
    /// It was hardcoded to SysV everywhere, so on every Windows PE the argument
    /// registers were named as if they were `rdi`/`rsi` and `locals::create`
    /// seeded the WRONG slots as parameters. The real `this`/arguments then had
    /// no definition at all, and every read of them became an undefined local
    /// with an `x2`/`x3`/`x4`/`x5` origin — which under SysV naming are exactly
    /// the four MSVC argument registers. Measured on the v24.5 PE corpus, fixing
    /// this alone: clean methods 33 → 107, `suspect` 10,175 → 8,000, undefined
    /// locals 39,595 → 28,536, unresolved calls 646 → 352.
    pub x86_abi: lift_x86::X86Abi,
    /// type-def index → base-class chain (incl. self), precomputed
    pub parent_chains: HashMap<i32, Vec<i32>>,
    /// optional assembly-name allow-list (from --pseudo-include-assembly)
    pub include_assemblies: Vec<String>,
    /// metadata-usage slot RVA → global field index (FieldInfo usages)
    pub field_index_by_addr: HashMap<u64, u32>,
    /// metadata-usage slot RVA → global method index (MethodDef usages only;
    /// MethodRef usages index `method_specs` and live in
    /// `ResolveMaps::method_spec_index_by_addr`)
    pub method_index_by_addr: HashMap<u64, u32>,
    /// Il2CppType datapoint → resolved generic display name ("List<int>")
    pub generic_names: HashMap<u64, String>,
    /// global method index → owning TypeDef index
    pub method_owner_tdi: HashMap<u32, i32>,
    /// global field index → owning TypeDef index (replaces per-lookup scans)
    pub field_owner_tdi: HashMap<u32, i32>,
}

/// The spec indices to name instantiations for, ascending.
///
/// Callers that fill an address→name map MUST walk the SPEC INDEX SPACE, never
/// `method_spec_generic_method_pointers`. That map is keyed by spec index but
/// several specs share one code pointer — IL2CPP compiles every reference-type
/// instantiation of a generic method to a single shared body, and the
/// constructors of unrelated delegate types collapse onto one stub as well — so
/// it cannot answer "which instantiation is at this address", and iterating it
/// directly picked a different winner per process. The lowest spec index, the
/// engine's own emission order, is the one stable choice.
///
/// The index space is also the only COMPLETE source: under full generic sharing
/// only the shared instantiations are compiled, so a concrete instantiation a
/// call site passes may have no pointer entry at all.
fn spec_indices_to_name(spec_count: usize) -> impl Iterator<Item = usize> {
    0..spec_count
}

/// Retire the unreachable `b <pad>` the compiler leaves after a raise call.
///
/// A raise helper never returns, so the branch that follows it is dead code —
/// but the CFG has no way to know that, and the branch hands the landing pad a
/// SYNCHRONOUS predecessor from the raising block. SSA then renames the pad's
/// reads along that edge, and the handler renders the body's values as the
/// exception (the `IL2CPP_IsInst((&enumerator).MoveNext(), …)` class).
///
/// A `throw` of the exception argument (index 2 in the lifters' raw
/// `[target, ret, x0, x1…]` call layout) is INSERTED after the call. The
/// trampoline branch itself is left untouched: it stays a pad an EH row can
/// name and `pad_root` can still follow to the real handler, while the
/// inserted terminator removes it from the call's fall-through path. The
/// call's result slot is retired at the same time — the call does not produce
/// a value, and leaving it would bind the throw operand to the never-returning
/// call's result version.
///
/// Returns the addresses the throw was inserted at (for the verifier ledger).
fn neutralise_raise_trampolines(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    landing_pads: &[u64],
    raise_helpers: &[u64],
) -> Vec<u64> {
    let mut branch_targets: HashSet<u64> = HashSet::new();
    for insn in insns.iter() {
        match &insn.op {
            Op::Jump(t) | Op::CondJump(t) => {
                branch_targets.insert(*t as u64);
            }
            Op::Switch(ts) => {
                for t in ts {
                    branch_targets.insert(*t as u64);
                }
            }
            _ => {}
        }
    }
    let mut insert_after: Vec<usize> = Vec::new();
    for i in 0..insns.len().saturating_sub(1) {
        if !matches!(insns[i].op, Op::Call) {
            continue;
        }
        let Some(Operand::Immediate(target)) = insns[i].operands.first().cloned() else {
            continue;
        };
        // Two independent proofs of "never returns":
        //  - the explicit il2cpp_raise_exception family, and
        //  - a helper the body classifier proved is a thin forwarder that
        //    reaches a __cxa throw stub (the null-check raiser lives only
        //    here; it is deliberately NOT in `raise_exception` because the
        //    resolver must not turn its call sites into source `throw`s).
        let is_raise = raise_helpers.contains(&(target as u64))
            || matches!(
                runtime_helpers::role_of(target as u64),
                Some(runtime_helpers::RuntimeHelper::RaiseException)
            );
        if !is_raise {
            continue;
        }
        if !matches!(
            insns[i + 1].op,
            Op::Jump(t) if landing_pads.contains(&(t as u64))
        ) {
            continue;
        }
        // Another path really branches to the trampoline: running it is not
        // provably dead, so leave the shape alone.
        if branch_targets.contains(&addresses[i + 1]) {
            continue;
        }
        insert_after.push(i);
    }
    if insert_after.is_empty() {
        return Vec::new();
    }
    let mut old_insns = std::mem::take(insns);
    let old_addrs = std::mem::take(addresses);
    let mut out_insns = Vec::with_capacity(old_insns.len() + insert_after.len());
    let mut out_addrs = Vec::with_capacity(old_addrs.len() + insert_after.len());
    let mut rewritten = Vec::with_capacity(insert_after.len());
    let mut next_insert = 0usize;
    for (i, mut insn) in old_insns.drain(..).enumerate() {
        let addr = old_addrs[i];
        if next_insert < insert_after.len() && insert_after[next_insert] == i {
            let arg = insn.operands.get(2).cloned().unwrap_or(Operand::Unknown);
            if insn.operands.len() > 1 {
                insn.operands[1] = Operand::Unknown;
            }
            out_insns.push(insn);
            out_addrs.push(addr);
            out_insns.push(Insn::new(Op::Throw, vec![arg], addr));
            out_addrs.push(addr);
            rewritten.push(addr);
            next_insert += 1;
            continue;
        }
        out_insns.push(insn);
        out_addrs.push(addr);
    }
    *insns = out_insns;
    *addresses = out_addrs;
    rewritten
}

impl RecoveryContext {
    /// Build all resolution maps (parallel over images like decompiler.rs).
    pub fn build(executor: &mut Il2CppExecutor, metadata: &Metadata, il2cpp: &Il2Cpp) -> Self {
        crate::dbg_eprintln!("[debug] RecoveryContext::build: got_relative_relocations has {} entries", il2cpp.got_relative_relocations.len());
        // Resolve every GenericInst type through the executor's full naming
        // logic (nested args, namespaces) once; keyed by datapoint.
        let mut generic_names: HashMap<u64, String> = HashMap::new();
        for t in &il2cpp.types {
            if crate::il2cpp::enums::Il2CppTypeEnum::from_u8(t.type_enum)
                == Some(crate::il2cpp::enums::Il2CppTypeEnum::GenericInst)
            {
                let name = executor.get_type_name(t, metadata, il2cpp, false, false);
                generic_names.insert(t.datapoint, name);
            }
        }
        // (rva, MethodRef, Option<(tdi, vtable slot)>) — virtual methods carry
        // their slot (kInvalidIl2CppMethodSlot == u16::MAX) so vtable_slots can
        // be joined from the same pass. Generic methods are skipped: callvirt
        // on those needs a MethodSpec, not a plain memberref.
        let method_entries: Vec<(u64, MethodRef, Option<(i32, usize)>)> = metadata
            .image_defs
            .par_iter()
            .enumerate()
            .flat_map(|(image_idx, image_def)| {
                let image_name = metadata
                    .get_string_from_index(image_def.name_index)
                    .unwrap_or_default();
                let assembly = image_name.trim_end_matches(".dll").to_string();
                let type_end = image_def.type_start as usize + image_def.type_count as usize;
                (image_def.type_start as usize..type_end)
                    .into_par_iter()
                    .flat_map_iter(move |tdi| {
                        let td = &metadata.type_defs[tdi];
                        let type_ns = metadata
                            .get_string_from_index(td.namespace_index)
                            .unwrap_or_default();
                        let type_name = metadata
                            .get_string_from_index(td.name_index)
                            .unwrap_or_default();
                        let method_end = td.method_start as usize + td.method_count as usize;
                        let image_name = image_name.clone();
                        let assembly = assembly.clone();
                        (td.method_start as usize..method_end).map(move |mi| {
                            (
                                mi,
                                td.method_start,
                                method_end,
                                type_ns.clone(),
                                type_name.clone(),
                                image_name.clone(),
                                assembly.clone(),
                                tdi,
                            )
                        })
                    })
                    .filter_map(
                        move |(mi, _, _, type_ns, type_name, image_name, assembly, tdi)| {
                            let method_def = metadata.method_defs.get(mi)?;
                            let ptr = il2cpp.get_method_pointer(&image_name, method_def);
                            // A VIRTUAL method with NO native body (abstract /
                            // interface declaration) still DECLARES its vtable
                            // slot. `BaseListBank.GetListLength` is abstract, so
                            // the slot was never entered and a call through the
                            // base class's slot 5 stayed `fnptr_` even though
                            // the callee's NAME is in metadata. Such entries are
                            // collected with rva 0 and only feed `vtable_slots`;
                            // a real body at the same (tdi, slot) wins below.
                            // The abstract/virtual flag is REQUIRED: a bodyless
                            // method whose slot field is not meaningful (generic
                            // definitions leave it unset) would poison a real
                            // virtual slot — `List<T>.Add` sites rendered as
                            // `list...ctor` that way.
                            let is_virtual = (method_def.flags & 0x0040) != 0;
                            let rva = if ptr == 0 {
                                if !is_virtual {
                                    return None;
                                }
                                0
                            } else {
                                let r = il2cpp.get_rva(ptr);
                                if r == 0 {
                                    return None;
                                }
                                r
                            };
                            let name = metadata
                                .get_string_from_index(method_def.name_index as i32)
                                .unwrap_or_default();
                            let is_static = (method_def.flags & 0x0010) != 0;
                            let ret = il2cpp
                                .types
                                .get(method_def.return_type as usize)
                                .map(|t| managed_type_of(t, metadata, il2cpp, 0))
                                .unwrap_or(ManagedType::Unknown);
                            let params = (0..method_def.parameter_count as usize)
                                .filter_map(|j| {
                                    let pidx = method_def.parameter_start as usize + j;
                                    let pd = metadata.parameter_defs.get(pidx)?;
                                    il2cpp
                                        .types
                                        .get(pd.type_index as usize)
                                        .map(|t| managed_type_of(t, metadata, il2cpp, 0))
                                })
                                .collect();
                            let vslot = if method_def.slot != u16::MAX
                                && method_def.generic_container_index == -1
                            {
                                Some((tdi as i32, method_def.slot as usize))
                            } else {
                                None
                            };
                            Some((
                                rva,
                                MethodRef {
                                    image_index: image_idx,
                                    type_def_index: tdi as i32,
                                    method_index: mi as u32,
                                    name,
                                    declaring_namespace: type_ns,
                                    declaring_type_name: type_name,
                                    assembly,
                                    sig: types::MethodSigInfo {
                                        ret,
                                        params,
                                        is_static,
                                    },
                                },
                                vslot,
                            ))
                        },
                    )
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut methods_by_rva: HashMap<u64, MethodRef> =
            HashMap::with_capacity(method_entries.len());
        let mut vtable_slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        // REAL BODIES first, with the original last-wins ordering: two
        // definitions can declare the same (tdi, slot) in shared-generic
        // metadata, and the later entry is the one that used to win. Then the
        // bodyless declarations (abstract/interface) fill only slots no body
        // claimed, so they can never displace a concrete override.
        for (rva, m, vslot) in &method_entries {
            if *rva == 0 {
                continue;
            }
            methods_by_rva.insert(*rva, m.clone());
            if let Some(k) = vslot {
                vtable_slots.insert(*k, m.clone());
            }
        }
        if !crate::recovery::env_flags::flag("ILR_NOABSTRACT") {
            for (rva, m, vslot) in &method_entries {
                if *rva != 0 {
                    continue;
                }
                if let Some(k) = vslot {
                    vtable_slots.entry(*k).or_insert_with(|| m.clone());
                }
            }
        }
        // TypeDef → (image index, assembly). An interface's MethodRef has to name
        // the assembly its call site resolves against, and the loop below sees
        // TypeDefs without the image context the method pass above carries.
        // Metadata emits each image's TypeDefs as one contiguous range.
        let mut image_of_tdi: Vec<(usize, String)> =
            vec![(usize::MAX, String::new()); metadata.type_defs.len()];
        for (image_idx, image_def) in metadata.image_defs.iter().enumerate() {
            let image_name = metadata
                .get_string_from_index(image_def.name_index)
                .unwrap_or_default();
            let assembly = image_name.trim_end_matches(".dll").to_string();
            let start = image_def.type_start.max(0) as usize;
            let end = start
                .saturating_add(image_def.type_count.max(0) as usize)
                .min(image_of_tdi.len());
            if let Some(range) = image_of_tdi.get_mut(start..end) {
                for slot in range {
                    *slot = (image_idx, assembly.clone());
                }
            }
        }
        // DECLARED INTERFACE METHOD SLOTS — the second slot source.
        //
        // `vtable_slots` above holds only methods with a NATIVE BODY (`ptr == 0`
        // returns early), and an interface method is abstract, so an interface's
        // own methods never reach it and `build_interface_dispatch` had nothing to
        // place for its pairs: every interface call rendered `fnptr_0x…` forever.
        // Measured on the shipped corpora, that was 96% of all interface pairs
        // (7,192 of 7,331 lost pairs on x64, 11,574 of 12,122 on arm64) while the
        // metadata carried a usable slot for EVERY one of them — so this is a
        // repair, not a guess. Only the interface's own declared methods are read
        // here, and only methods whose slot the metadata states; the uniqueness
        // rule in the builder still decides whether a placement may be used.
        let mut iface_decl_slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        for (tdi, td) in metadata.type_defs.iter().enumerate() {
            if !td.is_interface() {
                continue;
            }
            let type_ns = metadata
                .get_string_from_index(td.namespace_index)
                .unwrap_or_default();
            let type_name = metadata
                .get_string_from_index(td.name_index)
                .unwrap_or_default();
            let (image_index, assembly) = image_of_tdi
                .get(tdi)
                .cloned()
                .unwrap_or((usize::MAX, String::new()));
            let mstart = td.method_start.max(0) as usize;
            let mend = mstart.saturating_add(td.method_count as usize);
            for mi in mstart..mend {
                let Some(md) = metadata.method_defs.get(mi) else {
                    continue;
                };
                // A generic-container method's signature mentions the interface's
                // own generic parameters, which no call site here can name: leaving
                // it out keeps the honest `fnptr_0x…` instead of a fabricated type.
                if md.slot == u16::MAX || md.generic_container_index != -1 {
                    continue;
                }
                let name = metadata
                    .get_string_from_index(md.name_index as i32)
                    .unwrap_or_default();
                let is_static = (md.flags & 0x0010) != 0;
                let ret = il2cpp
                    .types
                    .get(md.return_type as usize)
                    .map(|t| managed_type_of(t, metadata, il2cpp, 0))
                    .unwrap_or(ManagedType::Unknown);
                let params = (0..md.parameter_count as usize)
                    .filter_map(|j| {
                        let pidx = md.parameter_start as usize + j;
                        let pd = metadata.parameter_defs.get(pidx)?;
                        il2cpp
                            .types
                            .get(pd.type_index as usize)
                            .map(|t| managed_type_of(t, metadata, il2cpp, 0))
                    })
                    .collect();
                iface_decl_slots.insert(
                    (tdi as i32, md.slot as usize),
                    MethodRef {
                        image_index,
                        type_def_index: tdi as i32,
                        method_index: mi as u32,
                        name,
                        declaring_namespace: type_ns.clone(),
                        declaring_type_name: type_name.clone(),
                        assembly: assembly.clone(),
                        sig: types::MethodSigInfo {
                            ret,
                            params,
                            is_static,
                        },
                    },
                );
            }
        }
        // INTERFACE DISPATCH (once per dump).
        //
        // An interface call is byte-identical to a class virtual call
        // (`BLR [klass + vtable_base + slot*16]`) — only the SLOT differs. An
        // interface method lives at `Il2CppRuntimeInterfaceOffsetPair::offset +
        // method.slot` in the SAME vtable array, so it is not declared in the
        // class's own `vtable_slots` and `lookup_vtable_slot` returned `None`:
        // every interface call rendered `fnptr_0x…` forever (the largest single
        // residue on the arm64 corpus). The table is a pure function of the
        // class/interface layout, so it is built here, not per method.
        let iface_tdi_of = |ti: i32| -> Option<i32> {
            if ti < 0 {
                return None;
            }
            let t = il2cpp.types.get(ti as usize)?;
            match crate::il2cpp::enums::Il2CppTypeEnum::from_u8(t.type_enum) {
                Some(crate::il2cpp::enums::Il2CppTypeEnum::Class)
                | Some(crate::il2cpp::enums::Il2CppTypeEnum::ValueType) => {
                    let kidx = t.klass_index() as usize;
                    (kidx < metadata.type_defs.len()).then_some(kidx as i32)
                }
                // A GenericInst is not a TypeDef index; refuse it rather than
                // name an interface from an unrelated definition.
                _ => None,
            }
        };
        // `method_defs` adds no placement of its own — it lets the builder report
        // WHY a pair placed nothing (declares no methods / no usable slot / lost
        // in the join). That is the difference between a fact about the metadata
        // and a defect in this join, and only the defect arm is fixable.
        let (iface_dispatch, iface_stats) = resolve::build_interface_dispatch_detailed(
            &metadata.type_defs,
            &metadata.interface_offsets,
            &metadata.method_defs,
            &vtable_slots,
            &iface_decl_slots,
            &iface_tdi_of,
        );
        resolve::install_interface_dispatch(iface_dispatch);
        resolve::install_interface_dispatch_stats(iface_stats);
        // `ILR_IFACE=1`: how many interface slots resolved, AND why the rest
        // did not. The bare count cannot tell "the metadata declares no
        // interfaces" from "the interface-offset layout was mis-parsed", and
        // those two want opposite fixes. It matters most on the arm64 corpora,
        // where the table is small enough to look like the feature did nothing
        // (52 slots recovered against the 2,142 the metadata declares).
        if crate::recovery::env_flags::flag("ILR_IFACE") {
            if let Some(s) = resolve::interface_dispatch_summary() {
                eprintln!("{s}");
            }
        }
        // `ILR_RVACOLLIDE=1`: census of native addresses claimed by MORE THAN
        // ONE method definition.
        //
        // IL2CPP emits generic-shared bodies: `List<A>.Add` and `List<B>.Add`
        // share one native address, and a global generic method shares one
        // across every instantiation. `methods_by_rva` is a single-valued map,
        // so `insert` here keeps whichever definition happened to be last and
        // every direct call to that address is then NAMED after it. That is a
        // fabricated owner: the call site is real, the name is one arbitrary
        // member of a set the resolver cannot distinguish. This census sizes
        // that set so the cost of the choice is measurable instead of assumed.
        // Identical-code folding, indexed for the resolver.
        //
        // IL2CPP merges bodies that compile to the same machine code, so one
        // address routinely carries several method DEFINITIONS that have
        // nothing to do with each other. `methods_by_rva` is single-valued and
        // can only keep one, so every call to such an address is named after
        // that arbitrary member — a fabricated declaring type. `resolve_calls`
        // uses this index to pick the member the CALL SITE evidences instead.
        let mut shared_by_rva: HashMap<u64, Vec<MethodRef>> = HashMap::new();
        for (rva, m, _) in &method_entries {
            shared_by_rva.entry(*rva).or_default().push(m.clone());
        }
        shared_by_rva.retain(|_, v| v.len() > 1);
        // Differing type ARGUMENTS (`List<int>` vs `List<string>`) is benign —
        // the type named is still the right one. Differing type DEFINITIONS
        // (`List<T>` vs `Dictionary<K,V>`) is the fabricated-declaring-type
        // case, and only the latter is dangerous.
        let shared_cross_type: HashMap<u64, bool> = shared_by_rva
            .iter()
            .map(|(rva, ms)| {
                let defs: std::collections::HashSet<i32> =
                    ms.iter().map(|m| m.type_def_index).collect();
                (*rva, defs.len() > 1)
            })
            .collect();
        let shared_rva_of_method: HashMap<u32, u64> = shared_by_rva
            .iter()
            .flat_map(|(rva, ms)| ms.iter().map(move |m| (m.method_index, *rva)))
            .collect();

        if crate::recovery::env_flags::flag("ILR_RVACOLLIDE") {
            use std::collections::HashSet as _HS;
            let by_rva: HashMap<u64, Vec<&MethodRef>> = shared_by_rva
                .iter()
                .map(|(rva, ms)| (*rva, ms.iter().collect()))
                .collect();
            let distinct_rvas_total: usize = {
                let mut s: _HS<u64> = _HS::new();
                for (rva, _, _) in &method_entries {
                    s.insert(*rva);
                }
                s.len()
            };
            let mut colliding = 0usize;
            let mut cross_type = 0usize;
            let mut max_fanout = 0usize;
            let mut owners: u64 = 0;
            for (_rva, ms) in by_rva.iter() {
                if ms.len() > 1 {
                    colliding += 1;
                    max_fanout = max_fanout.max(ms.len());
                    owners += ms.len() as u64;
                    let tys: _HS<&str> =
                        ms.iter().map(|m| m.declaring_type_name.as_str()).collect();
                    if tys.len() > 1 {
                        cross_type += 1;
                    }
                }
            }
            eprintln!(
                "[rva-collide] method_defs_with_ptr={} distinct_rvas={}                shared_rvas={} ({:.1}%) cross_type_shared={} max_fanout={} defs_on_shared_rvas={}",
                method_entries.len(),
                distinct_rvas_total,
                colliding,
                100.0 * colliding as f64 / by_rva.len().max(1) as f64,
                cross_type,
                max_fanout,
                owners,
            );

            // `cross_type_shared` above compares INSTANTIATED type names, so
            // `List<int>` and `List<string>` count as "cross type" even though
            // they are the same generic type. The two are completely different
            // problems: differing type ARGUMENTS still name the right type,
            // while differing type DEFINITIONS (`List<T>` vs `Dictionary<K,V>`)
            // is a fabricated declaring type — the reader is told a method
            // belongs to a class it does not. Separate them, and show the worst
            // addresses, because no amount of arg-count evidence fixes the
            // second: the only thing that can is the `MethodInfo` the call site
            // loads.
            let mut diff_def = 0usize;
            let mut diff_name = 0usize;
            let mut worst: Vec<(usize, u64, Vec<String>)> = Vec::new();
            for (&rva, ms) in by_rva.iter() {
                if ms.len() < 2 {
                    continue;
                }
                let defs: _HS<i32> = ms.iter().map(|m| m.type_def_index).collect();
                let names: _HS<&str> = ms.iter().map(|m| m.name.as_str()).collect();
                if defs.len() > 1 {
                    diff_def += 1;
                }
                if names.len() > 1 {
                    diff_name += 1;
                }
                if std::env::var("ILR_RVACOLLIDE").ok().as_deref() == Some("2") && defs.len() > 1 {
                    let mut ex: Vec<String> = ms
                        .iter()
                        .take(4)
                        .map(|m| format!("{}::{}", m.declaring_type_name, m.name))
                        .collect();
                    ex.push(format!("… {} total", ms.len()));
                    worst.push((ms.len(), rva, ex));
                }
            }
            eprintln!(
                "[rva-collide] differing_type_DEFINITION={} differing_method_NAME={}",
                diff_def, diff_name
            );
            worst.sort_by(|a, b| b.0.cmp(&a.0));
            for (n, rva, ex) in worst.iter().take(12) {
                eprintln!("[rva-collide]   {rva:#x} fanout={n}: {}", ex.join(", "));
            }
        }

        let literals: HashMap<u32, String> = (0..metadata.string_literals.len())
            .filter_map(|i| {
                metadata
                    .get_string_literal_from_index(i)
                    .ok()
                    .map(|s| (i as u32, s))
            })
            .collect();

        let mut ctx = Self {
            maps: resolve::ResolveMaps {
                methods_by_rva,
                shared_by_rva,
                shared_cross_type,
                shared_rva_of_method,
                literal_index_by_addr: HashMap::new(),
                literal_ptr_index_by_addr: HashMap::new(),
                literals,
                method_spec_index_by_addr: HashMap::new(),
                specs_by_body_rva: HashMap::new(),
                specs_by_index: HashMap::new(),
                object_new: Vec::new(),
                string_new: Vec::new(),
                box_fn: Vec::new(),
                raise_exception: Vec::new(),
                metadata_init: Vec::new(),
                vtable_slots,
                klass_to_tdi: HashMap::new(),
                usage_type_by_addr: HashMap::new(),
                named_type_sizes: HashMap::new(),
                tdi_names: HashMap::new(),
                objc_selector_stubs: HashMap::new(),
                native_wrappers: HashMap::new(),
                field_layouts: HashMap::new(),
                field_types: HashMap::new(),
                field_widths: HashMap::new(),
                static_field_layouts: HashMap::new(),
                field_type_by_name: HashMap::new(),
                field_generic_text: HashMap::new(),
                interface_tdis: HashSet::new(),
                type_parent_chains: HashMap::new(),
                enum_members: HashMap::new(),
                attribute_ctor_names: HashMap::new(),
                got_addend_by_addr: il2cpp.got_relative_relocations.clone(),
                method_name_by_va: HashMap::new(),
                static_field_offsets: HashMap::new(),
                static_field_types: HashMap::new(),
                tdi_names_reverse: HashMap::new(),
                method_return_tdi: HashMap::new(),
                tdi_fields: HashMap::new(),
                value_type_tdis: HashSet::new(),
            },
            max_method_bytes: 64 * 1024,
            parent_chains: HashMap::new(),
            include_assemblies: Vec::new(),
            field_index_by_addr: HashMap::new(),
            method_index_by_addr: HashMap::new(),
            method_owner_tdi: HashMap::new(),
            field_owner_tdi: HashMap::new(),
            generic_names,
            arch_is_arm64: matches!(
                il2cpp.detect_architecture(),
                crate::disassembler::Architecture::Arm64
            ),
            is_x64: matches!(
                il2cpp.detect_architecture(),
                crate::disassembler::Architecture::X64
            ),
            is_arm32: matches!(
                il2cpp.detect_architecture(),
                crate::disassembler::Architecture::Arm32
            ),
            x86_abi: x86_abi_of(il2cpp),
        };
        // Populate method_name_by_va: convert RVAs to VAs for quick lookup
        for (&rva, method_ref) in &ctx.maps.methods_by_rva {
            let va = il2cpp.map_rtva(rva);
            if va != 0 {
                ctx.maps.method_name_by_va.insert(va, (method_ref.name.clone(), method_ref.declaring_type_name.clone()));
            }
        }
        
        // method/field index → owner type-def (single pass over types)
        for (ti, td) in metadata.type_defs.iter().enumerate() {
            let ms = td.method_start as usize;
            let me = ms + td.method_count as usize;
            for mi in ms..me {
                ctx.method_owner_tdi.insert(mi as u32, ti as i32);
            }
            let fs = td.field_start as usize;
            let fe = fs + td.field_count as usize;
            for fi in fs..fe {
                ctx.field_owner_tdi.insert(fi as u32, ti as i32);
            }
        }

        // Pre-v27 usage slots are described by the metadata file, not written
        // into the image, so they are read before (and independently of) the
        // arch-specific encoded scan below. `metadata_usage_dic` is only
        // populated for version < 27.0, so this is a no-op on newer builds.
        if il2cpp.version < 27.0 {
            ctx.scan_legacy_metadata_usages(metadata, il2cpp);
        }

        // The class-ancestry map is PURE METADATA and is consulted by two
        // architecture-independent consumers: the `field_at_offset` typing
        // oracle (`field_at_offset` walks this chain to name a field declared
        // on a BASE class) and the vtable slot resolver (`lookup_vtable_slot`
        // needs it to find an inherited virtual's entry). Building it only
        // inside the arm64/x64 block left it EMPTY on ARM32, which silently
        // disabled ALL metadata field typing there — every `field_at_offset`
        // returned `None`, so no `[obj + off]` load was ever typed from its
        // declared field, receivers stayed `IntPtr`, and every vtable dispatch
        // through a field-held receiver reported "receiver class unknown".
        // Inherited virtuals failed for the same reason. `scan_metadata_usages`
        // and `detect_helpers` stay arch-gated: those really are encoded-scan
        // passes for the 64-bit image formats. `build_parent_chains` is not.
        ctx.parent_chains = build_parent_chains(metadata, il2cpp);
        // Pre-compute field layouts for the post-hoc field annotation pass.
        // For every TypeDef, compute (tdi, byte_offset) → (field_name, declaring_type_name)
        // for each instance field, and copy the parent chains into ResolveMaps so the
        // field annotation pass can walk base-class chains without needing BridgeOracle.
        ctx.maps.type_parent_chains = ctx.parent_chains.clone();
        ctx.maps.interface_tdis = metadata
            .type_defs
            .iter()
            .enumerate()
            .filter(|(_, td)| td.is_interface())
            .map(|(i, _)| i as i32)
            .collect();
        // Array element sizes for the late element-store sugar. Reference
        // types are pointer-sized; value types take their metadata
        // `instance_size`. Ambiguous short names are dropped rather than
        // risk a wrong stride on one of the same-named types.
        {
            let ptr_size = if il2cpp.is_32bit { 4u64 } else { 8u64 };
            let sizes = &mut ctx.maps.named_type_sizes;
            sizes.insert("object".into(), ptr_size);
            sizes.insert("string".into(), ptr_size);
            sizes.insert("Type".into(), ptr_size);
            // short name → size, with `None` meaning "seen with two sizes".
            let mut short: HashMap<String, Option<u64>> = HashMap::new();
            for (i, td) in metadata.type_defs.iter().enumerate() {
                let name = metadata
                    .get_string_from_index(td.name_index)
                    .unwrap_or_default();
                if name.is_empty() {
                    continue;
                }
                let size = if td.is_value_type() {
                    il2cpp
                        .type_definition_sizes
                        .get(i)
                        .map(|s| s.instance_size as u64)
                        .filter(|s| *s > 0)
                } else {
                    Some(ptr_size)
                };
                let Some(size) = size else { continue };
                let ns = metadata
                    .get_string_from_index(td.namespace_index)
                    .unwrap_or_default();
                if !ns.is_empty() {
                    sizes.insert(format!("{ns}.{name}"), size);
                }
                short
                    .entry(name)
                    .and_modify(|slot| {
                        if *slot != Some(size) {
                            *slot = None;
                        }
                    })
                    .or_insert(Some(size));
            }
            for (name, size) in short {
                if let Some(size) = size {
                    sizes.entry(name).or_insert(size);
                }
            }
        }
        Self::populate_field_layouts(
            &mut ctx.maps,
            metadata,
            il2cpp,
            &ctx.parent_chains,
            &ctx.generic_names,
        );

        // CRITICAL: scan_metadata_usages MUST run before populate_static_field_layouts,
        // because the latter reads from field_index_by_addr which is populated by the scan.
        // The encoded metadata usage words live in data sections on all architectures,
        // so this scan is not arch-gated (unlike detect_helpers which is arch-specific).
        if il2cpp.version >= 27.0 {
            ctx.scan_metadata_usages(metadata, il2cpp);
        }

        // Follow the pointer that stands in front of a usage slot, so a site
        // that loads the slot ADDRESS and then the word resolves like a site
        // that loads the word directly (see
        // `ResolveMaps::literal_ptr_index_by_addr`).
        ctx.alias_usage_pointers(il2cpp);

        // Pre-compute static field layouts from the metadata-usage scan.
        // Maps absolute VA → (field_name, type_name) for every static field,
        // so the field annotation pass can resolve `RawDeref(Int(base), offset, _)`
        // to `StaticField(type, field)` when `base + offset` matches a known address.
        Self::populate_static_field_layouts(&mut ctx.maps, metadata, &ctx.field_index_by_addr, &ctx.field_owner_tdi);
        // Pre-compute static field offsets within the static_fields structure.
        // Maps (tdi, byte_offset) → (field_name, type_name) for every static field,
        // so the field annotation pass can resolve
        // `RawDeref(Field(klass, "static_fields"), offset)` to `StaticField(type, field)`.
        {
            let t = std::time::Instant::now();
            Self::populate_static_field_offsets(&mut ctx.maps, metadata, il2cpp);
            eprintln!("[perf] populate_static_field_offsets: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
        }
        // Pre-compute reverse maps used by infer_types_from_stmts (built once
        // globally instead of per-method — eliminates thousands of redundant
        // HashMap constructions).
        {
            let t = std::time::Instant::now();
            ctx.maps.tdi_names_reverse = ctx.maps.tdi_names.iter().map(|(&tdi, (ns, name))| {
                if ns.is_empty() {
                    (name.clone(), tdi)
                } else {
                    (format!("{}.{}", ns, name), tdi)
                }
            }).collect();
            {
                let mut conflict: HashMap<String, bool> = HashMap::new();
                for method_ref in ctx.maps.methods_by_rva.values() {
                    if let ManagedType::Class(named) = &method_ref.sig.ret {
                        if named.type_def_index >= 0 {
                            let name = &method_ref.name;
                            if let Some(&existing) = ctx.maps.method_return_tdi.get(name) {
                                if existing != named.type_def_index {
                                    conflict.insert(name.clone(), true);
                                }
                            } else {
                                ctx.maps.method_return_tdi.insert(name.clone(), named.type_def_index);
                            }
                        }
                    }
                }
                for name in conflict.keys() {
                    ctx.maps.method_return_tdi.remove(name);
                }
            }
            eprintln!("[perf] pre-compute tdi_names_reverse + method_return_tdi: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
        }
        // Pre-compute enum member names for the enum_names reconstruction pass.
        Self::populate_enum_members(&mut ctx.maps, metadata, il2cpp);
        // Pre-compute attribute constructor names for the attribute recovery pass.
        Self::populate_attribute_ctor_names(&mut ctx.maps, executor, metadata, il2cpp);
        // For Unity 2022+ (IL2CPP v29+), also try to populate static fields from data sections.
        // This handles the case where static fields are NOT resolved through metadata usages
        // but are stored directly in .bss/.data sections with direct address references.
        // ALWAYS run this for v29+ since metadata usages give different addresses than the
        // actual runtime static field storage locations.
        if il2cpp.version >= 29.0 {
            Self::populate_static_fields_from_data_sections(&mut ctx.maps, metadata, il2cpp);
        }
        // GOT relocation-based static field resolution. Uses R_AARCH64_RELATIVE /
        // R_X86_64_RELATIVE relocations to map GOT addresses to static field storage
        // addends, then matches addends to classes by static_fields_size.
        // (Now called from within populate_static_fields_from_data_sections)
        if ctx.arch_is_arm64 || ctx.is_x64 {
            ctx.detect_helpers(metadata, il2cpp);
            // Experimental (ILR_ICALL=1): wrapper→native-target catalog.
            // NOTE: on this binary every extern-wrapper route goes through
            // SHARED dispatch hubs (0x4a58xxx family has hundreds of
            // tail-callers), so direct-branch correlation does NOT name
            // engine implementations — kept as a tool for binaries where
            // externs link directly.
            if crate::recovery::env_flags::flag("ILR_ICALL") {
                ctx.build_native_wrappers(metadata, il2cpp);
            }
        }

        // Generic-method instantiations (MethodSpecs). IL2CPP compiles each
        // used instantiation to its OWN code pointer (e.g. List<object>
        // .AddWithResize at a distinct RVA); without this those calls render
        // as `fn_134d334(...)` even though the dumper PRINTS the name in the
        // GenericInstMethod block. Signature = the definition's; display
        // name = the instantiated one.
        if !il2cpp.method_spec_generic_method_pointers.is_empty() {
            // Ordered, never the map's own order — see `spec_indices_in_order`.
            // Shared generic bodies make RVA collisions common rather than rare
            // (869 of the 5,760 spec RVAs in the reference v24.5 build are
            // ambiguous, the worst covering 406 specs), so which spec wins decides
            // the name of hundreds of call sites and cannot be left to chance.
            let mut spec_entries: Vec<(u64, u32, MethodRef)> = Vec::new();
            // EVERY spec, in ascending index order — not only the ones the
            // pointer table indexes. Under FULL generic sharing (the Unity 6
            // builds) only the shared instantiations are given code pointers,
            // so the concrete instantiation a call site passes is very often
            // absent from `method_spec_generic_method_pointers` while still
            // being present in `method_specs`. Naming it is exactly what the
            // call site's own `MethodInfo` load is for, and it can only be done
            // from here, where the metadata is at hand.
            for spec_idx in spec_indices_to_name(il2cpp.method_specs.len()) {
                let Some(spec) = il2cpp.method_specs.get(spec_idx) else {
                    continue;
                };
                let mdi = spec.method_definition_index;
                if mdi < 0 {
                    continue;
                }
                // Definition MethodRef from metadata directly — generic
                // DEFINITIONS often have no dedicated native pointer, so the
                // definition-entry map above never contained them (the cause
                // of unresolved `Dictionary<object,object>..ctor` calls).
                let Some(mut m) = ctx.method_info_ref(metadata, il2cpp, mdi as u32) else {
                    continue;
                };
                let (tname, mname) =
                    executor.get_method_spec_name(spec_idx, metadata, il2cpp, false);
                m.declaring_type_name = tname;
                m.name = mname;
                // A spec the pointer table indexes is named from its address.
                // One it does NOT index — no pointer, a null pointer, or a
                // pointer that maps nowhere — can only ever be named from a
                // call site, so it goes into `specs_by_index` instead. Keeping
                // ONLY those holds the specs that need it (16k of the reference
                // v106 build's 144k) rather than a second copy of every
                // instantiation the pointer table already covers.
                let rva = match il2cpp.method_spec_generic_method_pointers.get(&spec_idx) {
                    Some(&ptr) if ptr != 0 => il2cpp.get_rva(ptr),
                    _ => 0,
                };
                if rva == 0 {
                    ctx.maps.specs_by_index.insert(spec_idx as u32, m);
                } else {
                    spec_entries.push((rva, spec_idx as u32, m));
                }
            }
            // Colliding RVAs are shared bodies: keep every instantiation that
            // compiles to one, so a call site can pick the one it actually
            // passes. Keyed in ascending spec index like everything else here.
            let mut collided: HashMap<u64, Vec<(u32, MethodRef)>> = HashMap::new();
            for (rva, spec_idx, m) in &spec_entries {
                collided
                    .entry(*rva)
                    .or_default()
                    .push((*spec_idx, m.clone()));
            }
            for (rva, cands) in collided {
                if cands.len() > 1 {
                    ctx.maps.specs_by_body_rva.insert(rva, cands);
                }
            }
            for (rva, _, m) in spec_entries {
                // Definition entries win on collision, and where two *specs* want
                // the same RVA the lower spec index wins, because `spec_entries` is
                // in index order. A spec pointer only fills the map where no
                // definition pointer exists.
                ctx.maps.methods_by_rva.entry(rva).or_insert(m);
            }
        }
        ctx
    }

    /// Resolve a global method index → MethodRef with full signature.
    pub fn method_info_ref(
        &self,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        mi: u32,
    ) -> Option<types::MethodRef> {
        let tdi = *self.method_owner_tdi.get(&mi)?;
        let md = metadata.method_defs.get(mi as usize)?;
        let td = metadata.type_defs.get(tdi as usize)?;
        let name = metadata
            .get_string_from_index(md.name_index as i32)
            .unwrap_or_default();
        let declaring_namespace = metadata
            .get_string_from_index(td.namespace_index)
            .unwrap_or_default();
        let declaring_type_name = metadata
            .get_string_from_index(td.name_index)
            .unwrap_or_default();
        let is_static = (md.flags & 0x0010) != 0;
        let ret = il2cpp
            .types
            .get(md.return_type as usize)
            .map(|t| managed_type_of(t, metadata, il2cpp, 0))
            .unwrap_or(ManagedType::Unknown);
        let params = (0..md.parameter_count as usize)
            .filter_map(|j| {
                let pidx = md.parameter_start as usize + j;
                let pd = metadata.parameter_defs.get(pidx)?;
                il2cpp
                    .types
                    .get(pd.type_index as usize)
                    .map(|t| managed_type_of(t, metadata, il2cpp, 0))
            })
            .collect();
        Some(types::MethodRef {
            image_index: usize::MAX,
            type_def_index: tdi,
            method_index: mi,
            name,
            declaring_namespace,
            declaring_type_name,
            assembly: String::new(),
            sig: types::MethodSigInfo {
                ret,
                params,
                is_static,
            },
        })
    }

    /// Port of decompiler.rs `build_v27_annotations` address decoding:
    /// scan data sections for encoded metadata-usage entries and record
    /// `usage-slot RVA → decoded index`. StringLiteral entries feed
    /// `ldstr` recovery; type/method/field kinds are kept for later passes.
    fn scan_metadata_usages(&mut self, metadata: &Metadata, il2cpp: &Il2Cpp) {
        use rayon::prelude::*;
        let pointer_size = if il2cpp.is_32bit { 4u64 } else { 8u64 };

        // parallel over data sections; results merged after
        let section_results: Vec<Vec<(u64, u8, u32)>> = il2cpp
            .data_sections
            .par_iter()
            .map(|sec| {
                let sec_end = sec
                    .offset_end
                    .min(il2cpp.stream.len() as u64)
                    .saturating_sub(pointer_size);
                let mut local: Vec<(u64, u8, u32)> = Vec::new();
                let mut pos = sec.offset;
                while pos < sec_end {
                    let metadata_value = if il2cpp.is_32bit {
                        il2cpp.stream.peek_u32_at(pos).unwrap_or(0) as u64
                    } else {
                        il2cpp.stream.peek_u64_at(pos).unwrap_or(0)
                    };
                    pos += pointer_size;

                    if metadata_value >= u32::MAX as u64 || metadata_value == 0 {
                        continue;
                    }
                    let encoded = metadata_value as u32;
                    let raw_usage = (encoded & 0xE000_0000) >> 29;
                    if raw_usage == 0 || raw_usage > 7 {
                        continue;
                    }
                    let decoded_index = (encoded & 0x1FFF_FFFE) >> 1;
                    let expected = ((raw_usage << 29) | (decoded_index << 1)) + 1;
                    if metadata_value != expected as u64 {
                        continue;
                    }
                    // Cpp2IL `MetadataUsage.DecodeMetadataUsage`: from 106.1 on,
                    // every kind ABOVE TypeInfo is written one number lower,
                    // because the code generator's enum no longer contains
                    // `kIl2CppMetadataUsageIl2CppType`. Measured against the
                    // tables — the StringLiteral slots are kind 4 on the 108
                    // build (10,475 of 10,483 literal-table indices in range),
                    // kind 5 on the 106, 31.1 and 29.1 builds (17,614 of 17,624
                    // on iOS) — so the shift is real and version-gated.
                    let usage = if il2cpp.version > 106.1 && raw_usage > 1 {
                        raw_usage + 1
                    } else {
                        raw_usage
                    };

                    let addr = pos - pointer_size;
                    let va = il2cpp.map_rtva(addr);
                    if va == 0 {
                        continue;
                    }
                    let rva = il2cpp.get_rva(va);
                    if rva == 0 {
                        continue;
                    }
                    local.push((rva, usage as u8, decoded_index));
                }
                local
            })
            .collect();

        if crate::recovery::env_flags::flag("ILR_USAGEDBG") {
            let mut kinds: HashMap<u8, usize> = HashMap::new();
            let mut arrays = 0usize;
            let mut array_slots = HashSet::new();
            let mut elems: HashMap<String, usize> = HashMap::new();
            for &(_, usage, index) in section_results.iter().flatten() {
                *kinds.entry(usage).or_default() += 1;
                if usage == 1 || usage == 2 {
                    if let Some(t) = il2cpp.types.get(index as usize) {
                        if let Some(crate::il2cpp::enums::Il2CppTypeEnum::SzArray) =
                            crate::il2cpp::enums::Il2CppTypeEnum::from_u8(t.type_enum)
                        {
                            arrays += 1;
                            let mt = managed_type_of(t, metadata, il2cpp, 0);
                            *elems.entry(mt.display_name()).or_default() += 1;
                            array_slots.insert(index);
                        }
                    }
                }
            }
            eprintln!(
                "[usage] encoded kinds {kinds:?} szarray slots {arrays} distinct types {}",
                array_slots.len()
            );
            let mut top: Vec<_> = elems.into_iter().collect();
            top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            for (name, n) in top.into_iter().take(15) {
                eprintln!("[usage]   array {name}: {n} slots");
            }
        }
        for (rva, usage, decoded_index) in section_results.into_iter().flatten() {
            self.apply_metadata_usage(rva, usage, decoded_index, metadata, il2cpp);
        }
    }

    /// Record the slots that POINT AT a metadata-usage word.
    ///
    /// `scan_metadata_usages` keys each encoded word by its OWN address. Where the
    /// toolchain reaches a word through a relocated pointer the code loads the
    /// pointer first, so the address in the lifted expression is the pointer's
    /// and every lookup missed. The relocation table already says which pointer
    /// points at which word, so the alias is exact — no heuristic, and it costs
    /// one pass over relocations the loader has already parsed.
    fn alias_usage_pointers(&mut self, il2cpp: &Il2Cpp) {
        if crate::recovery::env_flags::flag("ILR_NO_GOTALIAS") {
            return;
        }
        let mut aliased = 0usize;
        for (&slot, &target) in &il2cpp.got_relative_relocations {
            let idx = self.maps.literal_index_by_addr.get(&il2cpp.get_rva(target)).copied();
            if let Some(idx) = idx {
                self.maps
                    .literal_ptr_index_by_addr
                    .insert(il2cpp.get_rva(slot), idx);
                aliased += 1;
            }
        }
        if crate::recovery::env_flags::flag("ILR_USAGEDBG") {
            eprintln!("[usage] literal pointer aliases: {aliased}");
        }
    }

    /// Record ONE decoded metadata usage: `rva` (the slot's address) now names
    /// `decoded_index` in the table `usage` selects.
    ///
    /// Shared by the two ways a slot is discovered — the post-v27 ENCODED word
    /// scan (`scan_metadata_usages`) and the pre-v27 `(kind, destination) →`
    /// `source` table (`scan_legacy_metadata_usages`) — so the two index spaces
    /// cannot drift apart. `usage` is the FINAL kind; the post-106.1
    /// "one-number-down" shift belongs to the decoder of the encoded word, and
    /// the pre-v27 table already stores raw enum values.
    fn apply_metadata_usage(
        &mut self,
        rva: u64,
        usage: u8,
        decoded_index: u32,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) {
        match usage {
            5 => {
                if (decoded_index as usize) < metadata.string_literals.len() {
                    self.maps.literal_index_by_addr.insert(rva, decoded_index);
                }
            }
            // Il2CppMetadataUsage::FieldInfo
            4 => {
                if (decoded_index as usize) < metadata.field_defs.len() {
                    self.field_index_by_addr.insert(rva, decoded_index);
                }
            }
            // Il2CppMetadataUsage::MethodDef
            //
            // ONLY kind 3 indexes `method_defs`. Kind 6 (MethodRef) is
            // a DIFFERENT index space: it indexes `method_specs`, so
            // `method_defs[decoded]` was an unrelated method that
            // merely happened to be in range. Verified against the iOS
            // binary: the three class-init slots
            // `CollectWeeklyFacebookGift` @0x47014C loads and passes as
            // a MethodInfo (0x6066b18/0x6066b60/0x6066b78) are all
            // kind 6, and they rendered as
            // `AndroidJNISafe.CallFloatMethod`, `FromLongArray`,
            // `FromSByteArray` — three UnityEngine methods offered as
            // the dispatch target of a call on `this._schedulerManager`
            // / `_shopData` / `_shopModel`. Kind 6 is now resolved
            // through `method_specs` (see below), which is the table it
            // actually addresses.
            3 => {
                if (decoded_index as usize) < metadata.method_defs.len() {
                    self.method_index_by_addr.insert(rva, decoded_index);
                }
            }
            // Il2CppMetadataUsage::MethodRef → a MethodSpec index
            // (Cpp2IL `MetadataUsage.AsGenericMethodRef`). This is the
            // concrete instantiation IL2CPP passes as the hidden
            // leading argument of a shared generic body: the body is
            // one address for hundreds of instantiations, so the call
            // site's own load of this slot is the ONLY thing that says
            // which instantiation it means.
            6 => {
                if (decoded_index as usize) < il2cpp.method_specs.len() {
                    self.maps
                        .method_spec_index_by_addr
                        .insert(rva, decoded_index);
                }
            }
            1 | 2 => self.record_klass_usage(rva, decoded_index, metadata, il2cpp),
            _ => {}
        }
    }

    /// `Il2CppMetadataUsage::TypeInfo` / `Il2CppType`: resolve a klass slot to
    /// its TypeDef so virtual-call recovery can identify receivers; the display
    /// name is recorded too, so `NewObj` can be typed without a metadata
    /// callback in the resolve pass.
    ///
    /// Shared with the pre-v27 legacy table (`scan_legacy_metadata_usages`) so
    /// the two discovery paths can never disagree about what a slot means.
    fn record_klass_usage(
        &mut self,
        rva: u64,
        decoded_index: u32,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) {
        if let Some(t) = il2cpp.types.get(decoded_index as usize) {
            let te = crate::il2cpp::enums::Il2CppTypeEnum::from_u8(t.type_enum);
            // GenericInst slots (List<object> etc.) carry the
            // definition tdi through the Il2CppGenericClass —
            // skipping them left every generic NewObj/vtable
            // untyped and the class-init helper call unaliased.
            let tdi_opt: Option<i32> = match te {
                Some(crate::il2cpp::enums::Il2CppTypeEnum::Class)
                | Some(crate::il2cpp::enums::Il2CppTypeEnum::ValueType) => {
                    let kidx = t.klass_index() as usize;
                    (kidx < metadata.type_defs.len()).then_some(kidx as i32)
                }
                Some(crate::il2cpp::enums::Il2CppTypeEnum::GenericInst) => {
                    let gc_addr = t.generic_class();
                    il2cpp.peek_generic_class(gc_addr).ok().and_then(|gc| {
                        if il2cpp.version >= 27.0 {
                            let tp = gc.type_ptr;
                            il2cpp
                                .get_il2cpp_type(tp)
                                .map(|it| (it.klass_index() as usize, it.clone()))
                        } else {
                            let idx = gc.type_definition_index as usize;
                            Some((idx, t.clone()))
                        }
                        .and_then(|(kidx, _)| {
                            (kidx < metadata.type_defs.len()).then_some(kidx as i32)
                        })
                    })
                }
                _ => None,
            };
            if crate::recovery::env_flags::flag("ILR_USAGEDBG")
                && std::env::var("ILR_USAGESLOT").is_ok_and(|s| {
                    u64::from_str_radix(s.trim_start_matches("0x"), 16)
                        .is_ok_and(|want| want == rva)
                })
            {
                eprintln!(
                    "[usage] slot {rva:#x} idx {decoded_index} enum {:?} tdi {tdi_opt:?} managed {:?}",
                    te,
                    managed_type_of(t, metadata, il2cpp, 0)
                );
            }
            // Every type slot, not just those with a TypeDef. SzArray slots
            // carry the ARRAY class that `il2cpp_array_new_specific` receives;
            // rendering them as `typeof(T[])` is what gives the allocator its
            // element type. Primitive/generic-inst slots get their correct
            // typeof() spelling from the same conversion.
            let mt = managed_type_of_gn(t, metadata, il2cpp, 0, Some(&self.generic_names));
            let text = mt.display_name();
            if !matches!(mt, ManagedType::Unknown)
                && text != "generic"
                && !text.contains("<unknown>")
            {
                self.maps.usage_type_by_addr.insert(rva, mt);
            }
            if let Some(tdi) = tdi_opt {
                self.maps.klass_to_tdi.insert(rva, tdi);
                if !self.maps.tdi_names.contains_key(&tdi) {
                    let td = &metadata.type_defs[tdi as usize];
                    let ns = metadata
                        .get_string_from_index(td.namespace_index)
                        .unwrap_or_default();
                    let name = metadata
                        .get_string_from_index(td.name_index)
                        .unwrap_or_default();
                    self.maps.tdi_names.insert(tdi, (ns, name));
                }
            }
        }
    }

    /// Pre-v27 metadata-usage slots (Cpp2IL `MapGlobalIdentifiersPre27`).
    ///
    /// Before v27 the usage slots are NOT written into the image as encoded
    /// `(kind << 29) | (index << 1) | 1` words: they are ordinary zero-filled
    /// data and are filled at startup from
    /// `Il2CppMetadataRegistration::metadataUsages`. The metadata file instead
    /// describes them as `(kind, destination) → source` pairs
    /// (`metadata_usage_dic`), so the SLOT ADDRESS comes from
    /// `il2cpp.metadata_usages[destination]` and only the source index has to
    /// be decoded here.
    ///
    /// `scan_metadata_usages` cannot find these slots at all on such a build:
    /// the slots live in a zero-initialized section with no file data behind
    /// it, so there is nothing to scan, and on the reference v24.5 build every
    /// slot the call sites load (`0xC32A50` and friends) lies outside the raw
    /// range of the section that holds them.
    ///
    /// EVERY kind is mapped, not just kind 6. Kind 6 (MethodRef) was the only
    /// one wired up at first, because it is the one that names a shared generic
    /// body's instantiation — but the table describes all seven kinds, and a
    /// pre-27 PE build has NO encoded word scan to fall back on, so leaving the
    /// others out meant `field_index_by_addr`, `literal_index_by_addr`,
    /// `method_index_by_addr` and `klass_to_tdi` were all EMPTY there. The cost
    /// is visible in the v24.5 dump: every static field read renders as
    /// `*(nint*)(v0 +9909760)` instead of `Type.field`, no string literal in any
    /// method body is a `"…"`, and no virtual call resolves. The index spaces
    /// are the same ones the encoded scan uses (`apply_metadata_usage` order
    /// below), so both paths cannot disagree.
    fn scan_legacy_metadata_usages(&mut self, metadata: &Metadata, il2cpp: &Il2Cpp) {
        // `metadata_usage_dic` is a HashMap, so iterating it directly would make
        // the index a slot ends up naming depend on a per-process hash seed.
        // Sort by destination — the order the engine wrote the pairs in — and
        // let the LOWEST destination win, like the lowest spec index does
        // everywhere else here. Kinds are visited in ascending order so the
        // result does not depend on HashMap iteration order either.
        let mut applied = 0usize;
        for kind in 1..=7u32 {
            let Some(pairs) = metadata.metadata_usage_dic.get(&kind) else {
                continue;
            };
            let mut ordered: Vec<(u32, u32)> = pairs.iter().map(|(&d, &s)| (d, s)).collect();
            ordered.sort_unstable();
            let ordered_len = ordered.len();
            for (dest, src) in ordered {
                let Some(&address) = il2cpp.metadata_usages.get(dest as usize) else {
                    continue;
                };
                if address == 0 {
                    continue;
                }
                let rva = il2cpp.get_rva(address);
                if rva == 0 {
                    continue;
                }
                // The pre-v27 table stores the raw enum kind: 1 TypeInfo,
                // 2 Il2CppType, 3 MethodDef, 4 FieldInfo, 5 StringLiteral,
                // 6 MethodRef, 7 EventInfo (unused here). Unlike the encoded
                // word path there is no post-106.1 one-number-down shift to
                // apply — pre-27 metadata has no encoded words at all.
                applied += 1;
                if let Ok(want) = std::env::var("ILR_USAGEDBG") {
                    let near: i64 =
                        i64::from_str_radix(want.trim_start_matches("0x"), 16).unwrap_or(0);
                    if (rva as i64 - near).abs() <= 0x40 {
                        eprintln!(
                            "[usage] NEAR kind {kind} dest {dest} src {src} addr {address:#x} rva {rva:#x}"
                        );
                    }
                }
                self.apply_metadata_usage(rva, kind as u8, src, metadata, il2cpp);
            }
            if crate::recovery::env_flags::flag("ILR_USAGEDBG") {
                eprintln!("[usage] kind {kind}: {} pairs", ordered_len);
            }
        }
        if crate::recovery::env_flags::flag("ILR_USAGEDBG") {
            eprintln!(
                "[usage] legacy slots applied {applied} | literals {} fields {} methods {} specs {} klass {}",
                self.maps.literal_index_by_addr.len(),
                self.field_index_by_addr.len(),
                self.method_index_by_addr.len(),
                self.maps.method_spec_index_by_addr.len(),
                self.maps.klass_to_tdi.len(),
            );
        }
    }

    /// Runtime-helper detection, layered like Cpp2IL's KeyFunctionAddresses:
    ///
    /// 1. **Export anchors** — `il2cpp_object_new`, `il2cpp_value_box`,
    ///    `il2cpp_raise_exception`, `il2cpp_string_new(wrapper)` resolved
    ///    through the binary's export table (`api_export_rvas`).
    /// 2. **Codegen-variant expansion** — any frequently-called *unresolved*
    ///    target that is a single-branch thunk of the anchor's root, or that
    ///    reaches the anchor within ≤3 call hops, joins the anchor's family
    ///    (Cpp2IL `FindAllThunkFunctions` / `IsExceptionRaiser`). This is
    ///    what catches `il2cpp_codegen_object_new` etc.
    /// 3. **Signature fallback (no exports)** — per-target call-site stats
    ///    (does the first arg look like a klass pointer? is a second arg
    ///    defined?) classify object_new / box / string_new from raw lifted
    ///    bodies, preferring thunk-shaped helpers to avoid misclassifying
    ///    real functions like `il2cpp_runtime_class_init`.
    fn detect_helpers(&mut self, _metadata: &Metadata, il2cpp: &Il2Cpp) {
        let is_arm64 = self.arch_is_arm64;
        let is_x64 = self.is_x64;
        if !is_arm64 && !is_x64 {
            return;
        }
        let sorted = il2cpp.build_sorted_method_addresses();
        if sorted.is_empty() {
            return;
        }

        // ---- phase 1: everything that reads self.maps (immutable) ---------
        let (string_new, object_new, box_fn, raise_exception, metadata_init, objc_stubs) = {
            let maps = &self.maps;

            // 1. export anchors (name → RVA)
            let anchor = |name: &str| {
                il2cpp
                    .api_export_rvas
                    .get(name)
                    .filter(|&&r| r != 0)
                    .copied()
            };
            let obj_anchor = anchor("il2cpp_object_new");
            let box_anchor = anchor("il2cpp_value_box");
            let raise_anchor = anchor("il2cpp_raise_exception");
            let string_anchors: Vec<u64> = ["il2cpp_string_new", "il2cpp_string_new_wrapper"]
                .iter()
                .filter_map(|n| anchor(n))
                .collect();

            // data-section RVA ranges — klass pointers live in .data/.bss
            let data_ranges: Vec<(u64, u64)> = il2cpp
                .data_sections
                .iter()
                .filter_map(|s| {
                    let lo = il2cpp.get_rva(s.address);
                    let hi = il2cpp.get_rva(s.address_end);
                    (lo != 0 && hi > lo).then_some((lo, hi))
                })
                .collect();
            let in_data = |a: u64| data_ranges.iter().any(|&(lo, hi)| a >= lo && a < hi);

            // klass-usage slot RVAs (from scan_metadata_usages) — a load from
            // one of these is a klass argument
            let klass_slots: HashSet<u64> = maps.klass_to_tdi.keys().copied().collect();

            // 3. sample scan with a mini constant/address tracker
            let sample_count = sorted.len().min(2000);
            let stride = (sorted.len() / sample_count.max(1)).max(1);
            let mut counts: HashMap<u64, HelperCounts> = HashMap::new();
            let total_lits = maps.literals.len() as i64;

            for &rva in sorted.iter().step_by(stride).take(sample_count) {
                let size = il2cpp.get_method_body_size(rva, &sorted);
                let probe = size.min(256);
                let Some(bytes) = il2cpp.read_bytes_at_rva(rva, probe) else {
                    continue;
                };
                let Some(lifted) = lift_helper_body(il2cpp, rva, &bytes, 120, is_arm64, is_x64)
                else {
                    continue;
                };

                let mut consts: HashMap<String, ArgVal> = HashMap::new();
                consts.insert("sp".to_string(), ArgVal::SpRef);
                for insn in &lifted {
                    let mut handled = false;
                    match insn.op {
                        ir::Op::Move => {
                            if let Some(Operand::Register(d)) = insn.dest() {
                                let src = insn.operands.get(1).cloned().unwrap_or(Operand::Unknown);
                                consts.insert(d.clone(), eval_arg(&src, &consts, &data_ranges));
                            }
                            handled = true;
                        }
                        ir::Op::Arith(kind) => {
                            if let Some(Operand::Register(d)) = insn.dest() {
                                let a = insn.operands.get(1).cloned();
                                let b = insn.operands.get(2).cloned();
                                let v = match kind {
                                    ir::ArithKind::Add => fold_add(
                                        a.as_ref().map(|o| eval_arg(o, &consts, &data_ranges)),
                                        b.as_ref().map(|o| eval_arg(o, &consts, &data_ranges)),
                                    ),
                                    _ => ArgVal::Unknown,
                                };
                                consts.insert(d.clone(), v);
                            }
                            handled = true;
                        }
                        ir::Op::Call | ir::Op::CallVoid => {
                            if let Some(Operand::Immediate(t)) = insn.operands.first() {
                                let t = *t as u64;
                                // managed calls are resolved elsewhere — skip
                                if !maps.methods_by_rva.contains_key(&t) {
                                    let c = counts.entry(t).or_default();
                                    c.total += 1;
                                    let x0 = consts.get("x0").copied().unwrap_or(ArgVal::Unknown);
                                    match x0 {
                                        ArgVal::Const(v) if v >= 0 && v < total_lits => {
                                            c.string_arg += 1;
                                        }
                                        ArgVal::DataRef(a)
                                            if klass_slots.contains(&(a as u64))
                                                || in_data(a as u64) =>
                                        {
                                            c.klass_arg += 1;
                                        }
                                        _ => {}
                                    }
                                    if !matches!(
                                        consts.get("x1").copied().unwrap_or(ArgVal::Unknown),
                                        ArgVal::Unknown
                                    ) {
                                        c.x1_arg += 1;
                                    }
                                }
                            }
                            // caller-clobbered
                            for r in ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"] {
                                consts.insert(r.to_string(), ArgVal::Unknown);
                            }
                            handled = true;
                        }
                        _ => {}
                    }
                    if !handled {
                        if let Some(Operand::Register(d)) = insn.dest().cloned() {
                            consts.insert(d, ArgVal::Unknown);
                        }
                    }
                }
            }

            // ---- classification -----------------------------------------

            // string_new: literal-index call sites (unchanged rule) + anchors
            let mut string_new = select_string_new(&counts);
            string_new.extend(string_anchors.iter().copied());
            string_new.sort_unstable();
            string_new.dedup();
            let excluded_str: HashSet<u64> = string_new.iter().copied().collect();

            // object_new: anchors + codegen variants (thunk-root or reach)
            let mut object_new: Vec<u64> = Vec::new();
            if let Some(a) = obj_anchor {
                object_new.push(a);
                // Compare against the IMPLEMENTATION root, not the export's own
                // RVA: the export is a wrapper that CALLS the allocator, and
                // every codegen variant branches to that same allocator (see
                // `runtime_helpers::export_roots`). Using the export's RVA
                // matched nothing, so no allocator thunk was ever classified.
                let roots = runtime_helpers::export_roots(
                    il2cpp,
                    &maps.methods_by_rva,
                    "il2cpp_object_new",
                    is_arm64,
                    is_x64,
                );
                let mut budget = 512u32;
                for t in freq_targets_desc(&counts, &excluded_str, 128) {
                    if object_new.contains(&t) {
                        continue;
                    }
                    // THUNK-ROOT EQUALITY is the precise test: an
                    // `il2cpp_codegen_object_new` wrapper IS the anchor under
                    // one more hop of `b`.
                    //
                    // The second arm used to accept "reaches the anchor within
                    // 3 calls". Three hops is enough to reach `object_new`
                    // from nearly ANY helper that takes part in object
                    // construction, and it admitted shared runtime helpers — on
                    // the reference binary it put the class-init helper
                    // (`0x4A58960`) into the set. That is not cosmetic: every
                    // call to a member of this set is rewritten to
                    // `NewObj(None)`, which either renders as a phantom
                    // `new object()` or is deleted outright as a dead
                    // allocation. In `CollectWeeklyFacebookGift` @0x47014C it
                    // SILENTLY DELETED all four static-initializer calls from
                    // the `if (init_flag == 0)` guard, leaving an empty guard
                    // body plus the flag store — a side effect the native code
                    // performs and the dumped C# does not.
                    //
                    // One DIRECT `bl` is still a wrapper (that is exactly what
                    // `il2cpp_codegen_object_new` is); a chain of three is a
                    // generic helper, which the doc comment above this function
                    // already warns against admitting ("real functions like
                    // `il2cpp_runtime_class_init`").
                    if roots.contains(&chase_thunk_root(
                        il2cpp,
                        &maps.methods_by_rva,
                        t,
                        is_arm64,
                        is_x64,
                    )) || reaches_call(
                            il2cpp,
                            t,
                            a,
                            is_arm64,
                            is_x64,
                            1,
                            &mut HashSet::new(),
                            &mut budget,
                        )
                    {
                        object_new.push(t);
                    }
                }
            } else {
                object_new = select_object_new_fallback(&counts, &excluded_str, &|t| {
                    is_thunk_body(il2cpp, &maps.methods_by_rva, t, is_arm64, is_x64)
                });
            }
            let excluded_obj: HashSet<u64> = excluded_str
                .union(&object_new.iter().copied().collect())
                .copied()
                .collect();

            // box: value_box + variants; fallback requires a defined 2nd arg
            let mut box_fn: Vec<u64> = Vec::new();
            if let Some(a) = box_anchor {
                box_fn.push(a);
                // Same wrapper-vs-implementation defect as object_new above.
                let roots = runtime_helpers::export_roots(
                    il2cpp,
                    &maps.methods_by_rva,
                    "il2cpp_value_box",
                    is_arm64,
                    is_x64,
                );
                let mut budget = 512u32;
                for t in freq_targets_desc(&counts, &excluded_obj, 128) {
                    if box_fn.contains(&t) {
                        continue;
                    }
                    if roots.contains(&chase_thunk_root(
                        il2cpp,
                        &maps.methods_by_rva,
                        t,
                        is_arm64,
                        is_x64,
                    ))
                        || reaches_call(
                            il2cpp,
                            t,
                            a,
                            is_arm64,
                            is_x64,
                            3,
                            &mut HashSet::new(),
                            &mut budget,
                        )
                    {
                        box_fn.push(t);
                    }
                }
            } else {
                box_fn = select_box_fallback(&counts, &excluded_obj, &|t| {
                    is_thunk_body(il2cpp, &maps.methods_by_rva, t, is_arm64, is_x64)
                });
            }
            let excluded_box: HashSet<u64> = excluded_obj
                .union(&box_fn.iter().copied().collect())
                .copied()
                .collect();

            // raise: anchored helpers (arg-taking) only; per-type throw
            // helpers (zero-arg, constructed internally) are NOT added here —
            // rewriting those as Throw(arg) would emit garbage
            let mut raise_exception: Vec<u64> = Vec::new();
            if let Some(a) = raise_anchor {
                raise_exception.push(a);
                // Same wrapper-vs-implementation defect. This one is the most
                // dangerous of the three: a raise role turns the call into a CFG
                // terminator, so a false member deletes live code. It stays exact
                // (root equality against the export's own implementation), not a
                // reachability test — see the warning in classify_bodies.
                let roots = runtime_helpers::export_roots(
                    il2cpp,
                    &maps.methods_by_rva,
                    "il2cpp_raise_exception",
                    is_arm64,
                    is_x64,
                );
                let mut budget = 512u32;
                for t in freq_targets_desc(&counts, &excluded_box, 128) {
                    if raise_exception.contains(&t) {
                        continue;
                    }
                    if roots.contains(&chase_thunk_root(
                        il2cpp,
                        &maps.methods_by_rva,
                        t,
                        is_arm64,
                        is_x64,
                    ))
                        || reaches_call(
                            il2cpp,
                            t,
                            a,
                            is_arm64,
                            is_x64,
                            3,
                            &mut HashSet::new(),
                            &mut budget,
                        )
                    {
                        raise_exception.push(t);
                    }
                }
            }

            // ---- metadata-init family (behavioral, verified) --------------
            // il2cpp_codegen_initialize_class / InitializeMethodMetadata:
            // runtime metadata bookkeeping, never part of the C# source.
            // Signature (ARM64, verified on real dumps): the thunk root's body
            // loads the klass/cctor byte — `LDR x22,[x0]` + `LDRB w8,[x22,#0x132]`
            // + CBZ — or dereferences [x0,#0xe0] writeback for method slots.
            // Candidates: frequently-called unresolved targets whose ROOT body
            // matches either shape.
            let mut metadata_init: Vec<u64> = Vec::new();
            {
                let matches_init_shape =
                    |il2cpp: &Il2Cpp, root: u64, is_arm64: bool, is_x64: bool| -> bool {
                        // The shape can live at ANY hop of the thunk chain:
                        // 0x4a58b98 → 0x4a8c6b0 passes THROUGH 0x4a8c398 (the
                        // `LDRB w?, [x?, #0x132]` carrier) before settling. Walk
                        // each hop and test it; stop at managed methods.
                        let mut cur = root;
                        for _ in 0..8 {
                            if il2cpp
                                .read_bytes_at_rva(cur, 64)
                                .and_then(|bytes| {
                                    lift_helper_body(il2cpp, cur, &bytes, 12, is_arm64, is_x64)
                                })
                                .map(|insns| {
                                    // ARM64 shape A: LDRB w?, [x?, #0x132] (cctor/init byte)
                                    // ARM64 shape B: LDR w?, [x?, #0xe0]! (method slots)
                                    insns.iter().any(|i| {
                                        i.operands.iter().any(|o| {
                                            matches!(
                                                o,
                                                Operand::Memory {
                                                    addend: 0x132 | 0xe0,
                                                    ..
                                                }
                                            )
                                        })
                                    })
                                })
                                .unwrap_or(false)
                            {
                                return true;
                            }
                            let Some(bytes) = il2cpp.read_bytes_at_rva(cur, 64) else {
                                return false;
                            };
                            let Some(insns) =
                                lift_helper_body(il2cpp, cur, &bytes, 8, is_arm64, is_x64)
                            else {
                                return false;
                            };
                            match insns.first().map(|i| &i.op) {
                                Some(ir::Op::Jump(t)) if (*t as u64) != cur => cur = *t as u64,
                                _ => return false,
                            }
                        }
                        false
                    };
                for &t in freq_targets_desc(&counts, &HashSet::new(), 256).iter() {
                    if maps.methods_by_rva.contains_key(&t)
                        || string_new.contains(&t)
                        || object_new.contains(&t)
                        || box_fn.contains(&t)
                        || raise_exception.contains(&t)
                    {
                        continue;
                    }
                    let root = chase_thunk_root(il2cpp, &maps.methods_by_rva, t, is_arm64, is_x64);
                    if root != t && matches_init_shape(il2cpp, root, is_arm64, is_x64) {
                        metadata_init.push(t);
                    }
                }
                // the root itself (the shared initialize body) too
                for &t in freq_targets_desc(&counts, &HashSet::new(), 256).iter() {
                    if !maps.methods_by_rva.contains_key(&t)
                        && matches_init_shape(il2cpp, t, is_arm64, is_x64)
                        && !metadata_init.contains(&t)
                    {
                        metadata_init.push(t);
                    }
                }
                if crate::recovery::env_flags::flag("ILR_ALIASES") {
                    for &probe in &[0x4a58b98u64, 0x4a58bac, 0x4a58a74, 0x4a58a60] {
                        let root =
                            chase_thunk_root(il2cpp, &maps.methods_by_rva, probe, is_arm64, is_x64);
                        eprintln!(
                            "[init-dbg] {probe:#x} root={root:#x} in_counts={} shape_self={} shape_root={}",
                            counts.contains_key(&probe),
                            matches_init_shape(il2cpp, probe, is_arm64, is_x64),
                            matches_init_shape(il2cpp, root, is_arm64, is_x64),
                        );
                    }
                }
            }

            // ---- secondary RuntimeHelper classification --------------------
            // The HelperMaps system (object_new, box_fn, etc.) rewrites calls
            // to IR ops (NewObj, Box, etc.) which can have dangerous side
            // effects (dead code deletion). The RuntimeHelper system only
            // renames calls for display — safer for uncertain classifications.
            //
            // Scan the top unclassified targets and add RuntimeHelper notes
            // for ones that match known patterns based on call-site statistics.
            {
                let all_classified: HashSet<u64> = string_new
                    .iter()
                    .chain(object_new.iter())
                    .chain(box_fn.iter())
                    .chain(raise_exception.iter())
                    .chain(metadata_init.iter())
                    .copied()
                    .collect();

                // Collect unclassified targets sorted by frequency
                let mut unclassified: Vec<(u64, u32)> = counts
                    .iter()
                    .filter(|(t, c)| {
                        !maps.methods_by_rva.contains_key(*t)
                            && !all_classified.contains(*t)
                            && c.total >= 10
                    })
                    .map(|(&t, c)| (t, c.total))
                    .collect();
                unclassified.sort_by(|a, b| b.1.cmp(&a.1));

                // ObjectNew: high klass_arg ratio + significant x1 usage
                // (real allocators pass extra register args from the caller's
                // frame; metadata_init and similar bookkeeping helpers typically
                // only use x0). Require x1 ratio >= 0.1 to filter out helpers
                // that only take a klass pointer.
                let mut obj_new_count = 0;
                for &(t, total) in unclassified.iter().take(10) {
                    if obj_new_count >= 2 {
                        break;
                    }
                    let c = &counts[&t];
                    let klass_ratio = if total > 0 { c.klass_arg as f32 / total as f32 } else { 0.0 };
                    let x1_ratio = if total > 0 { c.x1_arg as f32 / total as f32 } else { 0.0 };
                    if c.klass_arg > 0 && klass_ratio >= 0.6 && x1_ratio >= 0.1 {
                        runtime_helpers::note(t, runtime_helpers::RuntimeHelper::ObjectNew);
                        obj_new_count += 1;
                    }
                }

                // Box: high klass_arg AND high x1_arg (value pointer)
                for &(t, total) in unclassified.iter().take(15) {
                    let c = &counts[&t];
                    if c.klass_arg > 0 && c.x1_arg > 0 {
                        let klass_ratio = c.klass_arg as f32 / total as f32;
                        let x1_ratio = c.x1_arg as f32 / total as f32;
                        if klass_ratio >= 0.4 && x1_ratio >= 0.4 {
                            runtime_helpers::note(t, runtime_helpers::RuntimeHelper::Box);
                            break; // Just the top one
                        }
                    }
                }
            }

            // ---- bind-symbol chain naming (authoritative) ----------------
            // Codegen helpers frequently reach a lazy-GOT stub
            // (`ADRP x16; LDR x16,[x16,#off]; BR x16`) deep in their call
            // chain. dyld's bind tables name those slots exactly, so chase
            // each unresolved target's chain to a stub and classify by the
            // imported symbol name — no behavioral guessing.
            // (No inner scope: the alias pass below reuses these closures.)
            {
                // resolve_stub handles BOTH slot kinds:
                //  - dyld-bound import slots (___cxa_throw, …)
                //  - INTERNAL VM slots: the slot's live pointer is an export
                //    of this binary (il2cpp_vm_object_new etc.) — read the
                //    pointer and match it against api_export_rvas.
                let rva_to_name: HashMap<u64, &String> = il2cpp
                    .api_export_rvas
                    .iter()
                    .map(|(n, r)| (*r, n))
                    .collect();
                let resolve_stub = |slot_rva: u64| -> Option<String> {
                    if let Some(n) = il2cpp.bind_symbols.get(&slot_rva) {
                        return Some(n.clone());
                    }
                    // live pointer at the slot → export name
                    let b = il2cpp.read_bytes_at_rva(slot_rva, 8)?;
                    let ptr = u64::from_le_bytes(b.try_into().ok()?);
                    let ptr_rva = il2cpp.get_rva(ptr);
                    if ptr_rva == 0 {
                        return None;
                    }
                    rva_to_name.get(&ptr_rva).map(|n| (*n).clone())
                };
                // detect lazy-GOT / IAT stub, per architecture:
                //  - ARM64:  ADRP x16; LDR x16,[x16,#imm]; BR x16  (lifted)
                //  - x86-64: [endbr64] jmp qword [rip+disp32]       (bytes FF 25)
                //  - x86-32: [endbr32] jmp dword [abs]              (bytes FF 25)
                //  - ARM32:  ldr pc, [pc, #-4] + literal            (bytes 00 F0 ..)
                // Byte-level matching is exact and cheap for x86; ARM64 keeps
                // the lifted form because ADRP/LDR need page math the lifter
                // already does.
                let stub_slot = |a: u64| -> Option<u64> {
                    let bytes = il2cpp.read_bytes_at_rva(a, 16)?;
                    if is_arm64 {
                        let insns = lift_helper_body(il2cpp, a, &bytes, 4, is_arm64, is_x64)?;
                        if insns.len() < 3 {
                            return None;
                        }
                        let base = match (&insns[0].op, insns[0].operands.first()) {
                            (ir::Op::Move, Some(Operand::Register(r))) if r == "x16" => {
                                match insns[0].operands.get(1) {
                                    Some(Operand::Immediate(v)) => *v as u64,
                                    _ => return None,
                                }
                            }
                            _ => return None,
                        };
                        let off = match insns[1].operands.get(1).as_deref() {
                            Some(Operand::Memory {
                                base: rb, addend, ..
                            }) if matches!(rb.as_ref(), Operand::Register(r) if r == "x16") => {
                                *addend as u64
                            }
                            _ => return None,
                        };
                        // Two Android/ELF PLT forms load the slot through
                        // x16: the 3-insn `adrp; ldr x16,[x16]; br x16`, and
                        // the 4-insn `adrp x16; ldr x17,[x16,#lo];
                        // add x16,x16,#lo; br x17` the NDK linker emits —
                        // the branch is the LAST lifted instruction, not
                        // necessarily insns[2]. (Verified on this binary:
                        // 0x2dff7f0/0x2dff800 are `__cxa_allocate_exception`
                        // / `__cxa_throw` entries of the 4-insn form.)
                        let br = insns[2..]
                            .iter()
                            .any(|i| matches!(i.op, ir::Op::IndirectCall | ir::Op::Jump(_)));
                        return br.then_some(base + off);
                    }
                    if is_x64 {
                        // optional endbr64 prefix (F3 0F 1E FA)
                        let (b, prefix) = if bytes.starts_with(&[0xF3, 0x0F, 0x1E, 0xFA]) {
                            (&bytes[4..], 4u64)
                        } else {
                            (&bytes[..], 0u64)
                        };
                        // jmp qword [rip+disp32]: FF 25 disp32
                        if b.len() >= 6 && b[0] == 0xFF && b[1] == 0x25 {
                            let disp = i32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                            let after = a + prefix + 6; // RIP = end of the jmp
                            return Some((after as i64 + disp as i64) as u64);
                        }
                        return None;
                    }
                    // x86-32: jmp dword [abs]: FF 25 abs32 (slot = absolute VA)
                    let b = if bytes.starts_with(&[0xF3, 0x0F, 0x1E, 0xFB]) {
                        &bytes[4..]
                    } else {
                        &bytes[..]
                    };
                    if b.len() >= 6 && b[0] == 0xFF && b[1] == 0x25 {
                        let abs = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                        return Some(abs as u64);
                    }
                    // ARM32 PLT: `add ip, pc, #hi; add ip, ip, #mid; ldr pc, [ip, #lo]!`
                    // — too much page math; use the lifted memory-load form when
                    // available instead.
                    None
                };
                // BFS through the call chain; classify when ANY reached body
                // is itself a bind stub (BR to a bound slot) or any of its
                // callees reaches one. Verified: 0x4a58a60 → 0x4a9b91c → BL
                // 0x4add0e8 (`___cxa_allocate_exception`) + BL 0x4add16c
                // (`___cxa_throw`) — a codegen throw helper.
                //
                // resolve_stub also handles INTERNAL VM slots: a stub may
                // load a pointer that is not a dyld bind but an export of
                // this very binary (il2cpp_vm_object_new etc.). Read the
                // live pointer at the slot and match it against
                // A codegen throw helper is a SHORT wrapper or a stub: either
                // it is itself a stub bound to a `__cxa` symbol, or its own
                // body calls one (`BL ___cxa_allocate_exception` +
                // `BL ___cxa_throw`). So the test looks at the candidate and at
                // its DIRECT callees — nothing further.
                //
                // It used to run a 64-node, 4-callee-wide BFS over the whole
                // call graph for a `__cxa` name. In a 100 MB binary almost
                // every function reaches `__cxa_throw` within a few hops, so
                // the test was effectively "is this function part of the
                // program" and ordinary helpers were classified as raise
                // sites. That is not cosmetic: every call to a misclassified
                // target is rewritten to `Op::Throw`, which is a CFG
                // TERMINATOR, so everything after it becomes unreachable and
                // silently disappears from the output. `Game`'s `.cctor`
                // @0x1cb257c — 1860 bytes, 465 native insns — came out as a
                // single `throw` because one ordinary `bl` three instructions
                // in had been rewritten this way (87% of the method gone).
                //
                // Depth ≤ 1 is the conservative direction: a throw helper that
                // hides behind a second wrapper is now left as a call
                // (`fn_xxx(...)`) instead of a `throw`, and a missed `throw`
                // costs one wrong-looking call, while a false one DELETES the
                // rest of the method.
                let cxa_stub_at = |cur: u64| -> bool {
                    let Some(slot) = stub_slot(cur) else {
                        return false;
                    };
                    resolve_stub(slot)
                        .is_some_and(|name| name.starts_with("__cxa") || name.starts_with("___cxa"))
                };
                // With libc++ statically linked (Android .so), `__cxa_throw` is
                // a DEFINED export of this image — helpers BL it directly and no
                // bind table names the target. The symbol table supplies the
                // name; only the terminating routines count, so a helper that
                // merely calls `__cxa_begin_catch`/`__cxa_demangle` is not
                // evidence of a raise.
                let is_cxa = |cur: u64| -> bool {
                    cxa_stub_at(cur)
                        || il2cpp
                            .cxa_func_rvas
                            .get(&cur)
                            .is_some_and(|n| {
                                n.contains("cxa_throw")
                                    || n.contains("cxa_rethrow")
                                    || n.contains("cxa_allocate_exception")
                            })
                };
                let chain_hits_throw = |start: u64| -> bool {
                    if is_cxa(start) {
                        return true;
                    }
                    callees_first_n(il2cpp, &maps.methods_by_rva, start, 8, is_arm64, is_x64)
                        .into_iter()
                        .any(is_cxa)
                };
                // ARRAY ALLOCATION beats "raise" when both are reachable.
                // `il2cpp_array_new` allocates (array_element_size * length,
                // header + memset) and can REACH a throw path for the negative-
                // length/overflow case without being noreturn. A thin wrapper
                // around it therefore satisfied the raise test and every call
                // was rewritten to `throw`, deleting a live array definition
                // (0x1302484 in 0x151F34C: 86 reads of the lost array). The
                // import name is authoritative. NOTE: `callees_first_n` uses
                // symbol-boundary heuristics that mis-attribute this packed
                // thunk table, so scan the actual instruction window and follow
                // a leading tail jump instead.
                let call_import_name = |c: u64| -> Option<String> {
                    stub_slot(c).and_then(resolve_stub)
                };
                let window_calls = |start: u64| -> Vec<u64> {
                    let mut calls = Vec::new();
                    let mut a = start;
                    for _ in 0..6 {
                        let Some(bytes) = il2cpp.read_bytes_at_rva(a, 128) else {
                            break;
                        };
                        let Some(insns) =
                            lift_helper_body(il2cpp, a, &bytes, 32, is_arm64, is_x64)
                        else {
                            break;
                        };
                        for i in &insns {
                            if matches!(
                                i.op,
                                ir::Op::Call | ir::Op::CallVoid | ir::Op::IndirectCall
                            ) {
                                if let Some(Operand::Immediate(c)) = i.operands.first() {
                                    calls.push(*c as u64);
                                }
                            }
                        }
                        let tail = insns.iter().take(3).find_map(|i| match i.op {
                            ir::Op::Jump(t) if t as u64 != a => Some(t as u64),
                            _ => None,
                        });
                        match tail {
                            Some(j) => a = j,
                            None => break,
                        }
                    }
                    calls
                };
                let chain_hits_array_new = |start: u64| -> bool {
                    window_calls(start).into_iter().any(|c| {
                        call_import_name(c).is_some_and(|n| {
                            n.contains("array_new")
                                || n.contains("array_element_size")
                                || n.contains("array_object_header")
                        })
                    })
                };
                // NOTE: throw classification runs AFTER the alias pass below
                // (allocation/string variants must be claimed by their real
                // family first — raise is checked first in the resolver, so
                // anything wrongly put here turns `new X()` into `throw`).

                // ---- export ALIASING: name codegen variants of the helpers ---
                // IL2CPP emits per-callsite wrapper variants (stp x29/x30; bl VM
                // fn) for object_new/box/string_new. Each variant's FIRST callee
                // is the same VM function the export reaches, so variants alias
                // to the export. This kills the 50k+ `fn_4a58xxx` renders that
                // are really `new List<object>()` style allocations.
                let alias_targets: Vec<u64> = freq_targets_desc(&counts, &HashSet::new(), 256);
                let mut alias_object_new: Vec<u64> = Vec::new();
                let mut alias_box: Vec<u64> = Vec::new();
                let mut alias_string_new: Vec<u64> = Vec::new();
                // (same scope as resolve_stub/stub_slot/chain_hits_throw above —
                // the throw pass at the end reuses those closures)
                if crate::recovery::env_flags::flag("ILR_ALIASES") {
                    // call-site stats for the biggest unresolved targets:
                    // klass_arg ratio ≈ object alloc family; string_arg ≈
                    // literal family. Drives behavioral (not structural)
                    // aliasing.
                    let mut top: Vec<(&u64, &HelperCounts)> = counts
                        .iter()
                        .filter(|(t, _)| !maps.methods_by_rva.contains_key(*t))
                        .collect();
                    top.sort_by(|a, b| b.1.total.cmp(&a.1.total));
                    eprintln!("[counts] top unresolved targets:");
                    for (t, c) in top.iter().take(15) {
                        eprintln!(
                            "  {:#x} total={} klass_arg={} string_arg={} x1_arg={}",
                            t, c.total, c.klass_arg, c.string_arg, c.x1_arg
                        );
                    }
                }
                let callee_root = |start: u64| -> Option<u64> {
                    let mut cur = start;
                    // Real convergence (verified on dumps): variant → shim →
                    // shim → VM fn → allocator core can be 5-6 hops. 8 covers
                    // it; the loop breaks early on fixpoint anyway.
                    for _ in 0..8 {
                        let Some(next) =
                            first_callee(il2cpp, &maps.methods_by_rva, cur, is_arm64, is_x64)
                        else {
                            break;
                        };
                        if next == cur {
                            break;
                        }
                        cur = next;
                    }
                    Some(cur)
                };
                let obj_root = obj_anchor
                    .and_then(|a| first_callee(il2cpp, &maps.methods_by_rva, a, is_arm64, is_x64))
                    .and_then(callee_root);
                let box_root = box_anchor
                    .and_then(|a| first_callee(il2cpp, &maps.methods_by_rva, a, is_arm64, is_x64))
                    .and_then(callee_root);
                let str_root = string_anchors
                    .iter()
                    .find_map(|a| first_callee(il2cpp, &maps.methods_by_rva, *a, is_arm64, is_x64))
                    .and_then(callee_root);
                for &t in &alias_targets {
                    if maps.methods_by_rva.contains_key(&t) || metadata_init.contains(&t) {
                        continue;
                    }
                    let Some(tc) = first_callee(il2cpp, &maps.methods_by_rva, t, is_arm64, is_x64)
                    else {
                        continue;
                    };
                    let Some(tc) = callee_root(tc) else { continue };
                    if Some(tc) == obj_root {
                        alias_object_new.push(t);
                    } else if Some(tc) == box_root {
                        alias_box.push(t);
                    } else if Some(tc) == str_root {
                        alias_string_new.push(t);
                    }
                }
                if crate::recovery::env_flags::flag("ILR_ALIASES") {
                    let fmt = |v: &[u64]| {
                        v.iter()
                            .take(8)
                            .map(|a| format!("{a:#x}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    eprintln!(
                        "[aliases] object_new+={}\n  box+={}\n  string_new+={}",
                        fmt(&alias_object_new),
                        fmt(&alias_box),
                        fmt(&alias_string_new)
                    );
                }
                object_new.extend(alias_object_new);
                box_fn.extend(alias_box);
                string_new.extend(alias_string_new);

                // ---- throw-helper classification (AFTER aliasing) ----------
                // Order matters: raise is checked FIRST in the resolver, so an
                // allocation variant wrongly classified as raise would turn every
                // `new X()` into `throw`. Aliasing above claims alloc/string/
                // box variants first; only leftovers can be raise helpers.
                //
                // Two raise shapes in codegen:
                //  - explicit: `il2cpp_raise_exception(obj)` — arg-taking
                //  - implicit checks: construct the exception internally. In C#
                //    those are NOT `throw` statements (null checks etc.), so
                //    they must stay unclassified — classify ONLY arg-taking
                //    sites to avoid flooding output with fake `throw`s.
                for t in freq_targets_desc(&counts, &HashSet::new(), 256) {
                    if maps.methods_by_rva.contains_key(&t)
                        || metadata_init.contains(&t)
                        || string_new.contains(&t)
                        || object_new.contains(&t)
                        || box_fn.contains(&t)
                        || raise_exception.contains(&t)
                    {
                        continue;
                    }
                    let Some(c) = counts.get(&t) else { continue };
                    let arg_taking = c.total >= 5 && (c.x1_arg as f32 / c.total as f32) >= 0.3;
                    let hit = arg_taking && chain_hits_throw(t);
                    if hit {
                        raise_exception.push(t);
                    }
                }

                // ---- noreturn panic wrappers: RENAME ONLY (no `Op::Throw`) --
                // The null-check raise the codegen emits (`if (x == 0) { fn(x);
                // return; }`) is a THIN WRAPPER that never returns and reaches a
                // `__cxa_throw` import through a few internal hops — deeper than
                // the depth-≤1 `chain_hits_throw` above will follow. That cap
                // stays, because a wrong entry there rewrites the call to
                // `Op::Throw`, a CFG terminator that DELETES every statement
                // after it. Here a wrong entry only renames the call, so the
                // reach may go deeper — the cost of a false positive is a wrong
                // label, never lost code. Two gates keep it honest: the target
                // must be NORETURN (`is_noreturn`, no `Return` in its own body)
                // and must reach a `__cxa` stub (`reaches_cxa`), and anything the
                // alloc/string/box/init families already claim is skipped so a
                // shared allocator never gets mislabelled as a raise.
                {
                    // A codegen raise wrapper is THIN: it saves the link
                    // register and delegates within the first couple of
                    // instructions (`stp x29,x30,[..]; bl <throw-core>`, or a
                    // bare `b <throw-core>`), with no `ret` on the way out.
                    // Ordinary helpers do work before their first call, so the
                    // immediate-delegation shape plus `reaches_cxa` below is
                    // what separates a raise wrapper from a function that
                    // merely happens to (transitively) reach `__cxa_throw`.
                    // Both predicates lift bytes per call; the BFS and the
                    // 256-target scan revisit the same engine nodes over and
                    // over, so memoize by address (classification is single
                    // threaded here).
                    let thin_memo: std::cell::RefCell<HashMap<u64, bool>> =
                        std::cell::RefCell::new(HashMap::new());
                    let callee_memo: std::cell::RefCell<HashMap<u64, Vec<u64>>> =
                        std::cell::RefCell::new(HashMap::new());
                    let callees_of = |n: u64| -> Vec<u64> {
                        if let Some(v) = callee_memo.borrow().get(&n) {
                            return v.clone();
                        }
                        let v = callees_first_n(il2cpp, &maps.methods_by_rva, n, 8, is_arm64, is_x64);
                        callee_memo.borrow_mut().insert(n, v.clone());
                        v
                    };
                    let is_thin_forwarder = |a: u64| -> bool {
                        if let Some(&v) = thin_memo.borrow().get(&a) {
                            return v;
                        }
                        let v = (|| -> bool {
                            if maps.methods_by_rva.contains_key(&a) {
                                return false;
                            }
                            let Some(bytes) = il2cpp.read_bytes_at_rva(a, 32) else {
                                return false;
                            };
                            let Some(insns) =
                                lift_helper_body(il2cpp, a, &bytes, 8, is_arm64, is_x64)
                            else {
                                return false;
                            };
                            // first instruction that leaves the function body.
                            let leave = insns.iter().position(|i| {
                                matches!(i.op, ir::Op::Call | ir::Op::CallVoid)
                                    || matches!(i.op, ir::Op::Jump(t) if (t as u64) != a)
                            });
                            match leave {
                                Some(k) => {
                                    // delegate within the first three instructions
                                    // and return to us in none of them.
                                    k <= 2
                                        && !insns[..k]
                                            .iter()
                                            .any(|i| matches!(i.op, ir::Op::Return))
                                }
                                None => false,
                            }
                        })();
                        thin_memo.borrow_mut().insert(a, v);
                        v
                    };
                    let reaches_cxa = |start: u64| -> bool {
                        let dbg = crate::recovery::env_flags::flag("ILR_RAISEDDBG") && start == 0x1302634;
                        let mut frontier = vec![start];
                        let mut seen: HashSet<u64> = HashSet::new();
                        seen.insert(start);
                        // The entry gate (`is_thin_forwarder` on `start`) is
                        // what separates a wrapper from ordinary code; inside,
                        // the chain runs thin forwarders AND fat IL2CPP panic
                        // helpers (verified: 0x1302634 → 0x133e628[fat] →
                        // 0x133e674 → 0x133e590[fat] → … → `__cxa_throw`), so
                        // every node expands. The cap keeps that honest: a
                        // genuine exception chain is small and deep-ish, while
                        // a helper that only reaches `__cxa_throw` through the
                        // whole program floods the frontier and gets cut off.
                        // (64 was too tight — real chains flooded on the fat
                        // helpers' callees; 1024 is the program-reachability
                        // trip wire.) This path is RENAME-ONLY: a wrong verdict
                        // costs a label, never deleted statements.
                        for depth in 0..7 {
                            let mut next = Vec::new();
                            for &n in &frontier {
                                let hit = is_cxa(n);
                                if dbg {
                                    let sl = stub_slot(n);
                                    eprintln!(
                                        "[raisedbg] depth={depth} node={n:#x} cxa={hit} thin={} slot={:?} name={:?}",
                                        is_thin_forwarder(n),
                                        sl,
                                        sl.and_then(|s| resolve_stub(s))
                                    );
                                }
                                if hit {
                                    return true;
                                }
                                for c in callees_of(n) {
                                    if seen.insert(c) {
                                        next.push(c);
                                        if seen.len() > 1024 {
                                            if dbg {
                                                eprintln!("[raisedbg] depth={depth} cap hit at node={n:#x}");
                                            }
                                            return false;
                                        }
                                    }
                                }
                            }
                            if next.is_empty() {
                                if dbg {
                                    eprintln!("[raisedbg] depth={depth} frontier exhausted");
                                }
                                return false;
                            }
                            frontier = next;
                        }
                        false
                    };
                    for t in freq_targets_desc(&counts, &HashSet::new(), 256) {
                        let dbg = crate::recovery::env_flags::flag("ILR_RAISEDDBG") && t == 0x1302634;
                        if maps.methods_by_rva.contains_key(&t)
                            || raise_exception.contains(&t)
                            || metadata_init.contains(&t)
                            || string_new.contains(&t)
                            || object_new.contains(&t)
                            || box_fn.contains(&t)
                        {
                            if dbg {
                                eprintln!("[raisedbg] {t:#x} EXCLUDED managed={} raise={} meta={} str={} obj={} box={}",
                                    maps.methods_by_rva.contains_key(&t), raise_exception.contains(&t),
                                    metadata_init.contains(&t), string_new.contains(&t), object_new.contains(&t), box_fn.contains(&t));
                            }
                            continue;
                        }
                        let thin = is_thin_forwarder(t);
                        if crate::recovery::env_flags::flag("ILR_ALIASES") && t == 0x1302484 {
                            let names: Vec<String> = window_calls(t)
                                .into_iter()
                                .map(|c| format!("{c:#x}:{:?}", call_import_name(c)))
                                .collect();
                            eprintln!(
                                "[arraydbg] t={t:#x} thin={thin} chain={} calls={}",
                                chain_hits_array_new(t),
                                names.join(",")
                            );
                        }
                        if thin && chain_hits_array_new(t) {
                            runtime_helpers::note(
                                t,
                                runtime_helpers::RuntimeHelper::ArrayNew,
                            );
                            if dbg {
                                eprintln!("[raisedbg] {t:#x} ARRAY_NEW (wins over raise)");
                            }
                            continue;
                        }
                        // short-circuit: only pay for the reach when thin.
                        let reach = thin && (dbg || reaches_cxa(t));
                        let reach = if dbg { reaches_cxa(t) } else { reach };
                        if dbg {
                            eprintln!("[raisedbg] {t:#x} thin={thin} reach={reach}");
                        }
                        if thin && reach {
                            runtime_helpers::note(
                                t,
                                runtime_helpers::RuntimeHelper::RaiseException,
                            );
                        }
                    }
                }

                // ---- authoritative BODY / IMPORT re-classification ---------
                // The statistical/alias passes above are deliberately broad;
                // read each frequent helper's own body and the exported IL2CPP
                // API (or PLT import) it calls. This overwrites the provisional
                // role, so a box helper claimed by object_new aliasing becomes
                // Box, and a class-init/metadata-init wrapper stops rendering
                // as an allocation. Keys on symbols, never on addresses.
                if !crate::recovery::env_flags::flag("ILR_NOBODYCLASS") {
                    runtime_helpers::classify_bodies(
                        il2cpp,
                        &maps.methods_by_rva,
                        &freq_targets_desc(&counts, &HashSet::new(), 256),
                        is_arm64,
                        is_x64,
                    );
                }
            } // end bind-symbol / stub-classification scope

            // ---- Objective-C selector-stub family (Apple builds) ----------
            // Shape (verified on UnityFramework arm64):
            //   outer: stp x29,x30,[sp,#-0x10]!; mov x29,sp; bl selector_stub
            //   selector_stub: adrp x1,GOT; ldr x1,[x1,off]; adrp x16; ldr x16,[x16,off]; br x16
            // The GOT slot's live pointer targets a __cstring selector
            // ("setBool:forKey:", …). A call to the outer stub IS a message
            // send with that fixed selector — name-recoverable where the raw
            // `fn_4a58bac` render was not. These stubs sit BEYOND the managed
            // RVA range (engine/native section), which is why every dump had
            // tens of thousands of unresolved `fn_` calls.
            let mut objc_stubs: HashMap<u64, String> = HashMap::new();
            if is_arm64 {
                for &t in freq_targets_desc(&counts, &HashSet::new(), 4096).iter() {
                    if maps.methods_by_rva.contains_key(&t)
                        || string_new.contains(&t)
                        || object_new.contains(&t)
                        || box_fn.contains(&t)
                        || raise_exception.contains(&t)
                        || metadata_init.contains(&t)
                    {
                        continue;
                    }
                    // walk the thunk chain; the selector stub is reached by BL
                    let mut cur = t;
                    let mut hops = 0;
                    let sel = loop {
                        if hops > 4 {
                            break None;
                        }
                        hops += 1;
                        let Some(bytes) = il2cpp.read_bytes_at_rva(cur, 64) else {
                            break None;
                        };
                        let Some(insns) =
                            lift_helper_body(il2cpp, cur, &bytes, 16, is_arm64, is_x64)
                        else {
                            break None;
                        };
                        // leading tail-jump keeps the chain going
                        if let Some(ir::Op::Jump(nt)) = insns.first().map(|i| &i.op) {
                            if (*nt as u64) != cur {
                                cur = *nt as u64;
                                continue;
                            }
                        }
                        // first BL in body → next hop
                        let mut bl_target: Option<u64> = None;
                        for i in &insns {
                            if matches!(i.op, ir::Op::Call | ir::Op::CallVoid) {
                                if let Some(Operand::Immediate(nt)) = i.operands.first() {
                                    bl_target = Some(*nt as u64);
                                }
                                break;
                            }
                        }
                        let Some(next) = bl_target else {
                            break None;
                        };
                        // BL target must be a selector stub:
                        // adrp x1; ldr x1,[x1+off]; …; br x16
                        let Some(sbytes) = il2cpp.read_bytes_at_rva(next, 32) else {
                            break None;
                        };
                        let Some(sinsns) =
                            lift_helper_body(il2cpp, next, &sbytes, 8, is_arm64, is_x64)
                        else {
                            break None;
                        };
                        // x1 load: Move(x1, Imm(page)) followed by
                        // Move(x1, Memory{base: x1, addend})
                        let mut sel_got: Option<u64> = None;
                        for w in sinsns.windows(2) {
                            if !matches!((&w[0].op, &w[1].op), (ir::Op::Move, ir::Op::Move)) {
                                continue;
                            }
                            let d_is_x1 =
                                matches!(w[0].dest(), Some(Operand::Register(r)) if r == "x1");
                            let page = match w[0].operands.get(1) {
                                Some(Operand::Immediate(v)) => *v as u64,
                                _ => 0,
                            };
                            let off = match w[1].operands.get(1) {
                                Some(Operand::Memory { base, addend, .. }) if matches!(base.as_ref(), Operand::Register(r) if r == "x1") => {
                                    *addend as u64
                                }
                                _ => 0,
                            };
                            if d_is_x1 && page != 0 {
                                sel_got = Some(page + off);
                                break;
                            }
                        }
                        let Some(got) = sel_got else {
                            break None;
                        };
                        // live pointer at the GOT slot → C-string selector
                        let Some(pb) = il2cpp.read_bytes_at_rva(got, 8) else {
                            break None;
                        };
                        let ptr = u64::from_le_bytes(pb.try_into().unwrap());
                        let ptr_rva = il2cpp.get_rva(ptr);
                        if ptr_rva == 0 {
                            break None;
                        }
                        let Some(sb) = il2cpp.read_bytes_at_rva(ptr_rva, 128) else {
                            break None;
                        };
                        let len = sb.iter().position(|&b| b == 0).unwrap_or(0);
                        if len == 0 || len > 120 {
                            break None;
                        }
                        let s = String::from_utf8_lossy(&sb[..len]).to_string();
                        // selector shape: colon-suffixed parts or camelCase
                        if s.ends_with(':')
                            || s.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                        {
                            break Some(s);
                        }
                        break None;
                    };
                    // skip: exception-wrap family already classified above by
                    // chain_hits_throw (0x4a58bac reaches the
                    // 'NullReferenceException' ctor chain); never a selector stub.
                    if raise_exception.contains(&t) {
                        continue;
                    }
                    if let Some(s) = sel {
                        objc_stubs.insert(t, s);
                    }
                }
                if crate::recovery::env_flags::flag("ILR_ALIASES") {
                    eprintln!("[objc] {} selector stubs recovered", objc_stubs.len());
                    for (rva, s) in objc_stubs.iter().take(10) {
                        eprintln!("[objc]   {rva:#x} → {s}");
                    }
                }
            }

            (
                string_new,
                object_new,
                box_fn,
                raise_exception,
                metadata_init,
                objc_stubs,
            )
        };

        // ---- phase 2: publish -------------------------------------------
        if crate::recovery::env_flags::flag("ILR_HELPERS") {
            eprintln!(
                "[helpers] string_new={} object_new={} box={} raise={}",
                fmt_rvas(&string_new),
                fmt_rvas(&object_new),
                fmt_rvas(&box_fn),
                fmt_rvas(&raise_exception),
            );
        }
        self.maps.string_new = string_new;
        self.maps.object_new = object_new;
        self.maps.box_fn = box_fn;
        self.maps.raise_exception = raise_exception;
        self.maps.metadata_init = metadata_init;
        self.maps.objc_selector_stubs = objc_stubs;
    }

    /// Recover one method body to CIL via dotnetdll tokens.
    pub fn recover_method<R: emit::TokenResolver>(
        &self,
        bytes: &[u8],
        rva: u64,
        seed: &locals::SeedInfo,
        resolver: &R,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> Option<emit::EmitOutput> {
        let ir = self.recover_ir(bytes, rva, seed, metadata, il2cpp)?;
        let unused = passes::unused_locals(&ir.insns, &ir.table);
        let emitter = emit::Emitter { resolver };
        Some(emitter.generate(&ir.insns, &ir.table, &unused, ir.returns_value))
    }

    /// Run the analysis pipeline through all passes, returning the final IR +
    /// local table. Shared by the CIL emitter and the native C# decompiler.
    pub fn recover_ir(
        &self,
        bytes: &[u8],
        rva: u64,
        seed: &locals::SeedInfo,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) -> Option<RecoveredIr> {
        self.recover_ir_deadline(bytes, rva, seed, metadata, il2cpp, None)
    }

    /// Same pipeline with a wall-clock budget checked between stages.
    pub fn recover_ir_deadline(
        &self,
        bytes: &[u8],
        rva: u64,
        seed: &locals::SeedInfo,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        deadline: Option<std::time::Instant>,
    ) -> Option<RecoveredIr> {
        let timed_out = |d: &Option<std::time::Instant>| -> bool {
            d.is_some_and(|d| std::time::Instant::now() > d)
        };
        let dbg_fail = crate::recovery::env_flags::flag("ILR_FAIL");
        let timing_on = crate::recovery::env_flags::flag("ILR_TIMING");
        let stage_trace = crate::recovery::env_flags::flag("ILR_STAGES");
        // `ILR_CONFORM` runs the structural conformance checker, which needs
        // the same per-method hook as the verify oracle but not its verbose
        // per-method log — a class census over the whole binary should print
        // one ranked table at the end, not one line per method.
        // `ILR_ANNOTATE` is the default-on mode: grade every method so the
        // emitted `dump.cs` is self-labelling, but keep the verbose per-method
        // log off (that stays ILR_VERIFY's job).
        let verify_on = crate::recovery::env_flags::flag("ILR_VERIFY")
            || crate::recovery::env_flags::flag("ILR_CONFORM")
            || crate::recovery::env_flags::flag("ILR_ANNOTATE");
        macro_rules! stage_begin {
            ($name:expr) => {{
                if stage_trace {
                    eprintln!("[stage] {:#x} {}", rva, $name);
                }
                std::time::Instant::now()
            }};
        }
        let method_t0 = std::time::Instant::now();
        tm::METHODS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // per-method stage microseconds: lift, cfg, stack, ssa, propagate,
        // resolve, guards, opt, eliminate
        let mut st = [0u64; 9];
        macro_rules! flush {
        () => {{
                let n = method_t0.elapsed().as_micros() as u64;
                use std::sync::atomic::Ordering::Relaxed;
                tm::LIFT.fetch_add(st[0], Relaxed);
                tm::CFG.fetch_add(st[1], Relaxed);
                tm::STACK.fetch_add(st[2], Relaxed);
                tm::SSA.fetch_add(st[3], Relaxed);
                tm::PROPAGATE.fetch_add(st[4], Relaxed);
                tm::RESOLVE.fetch_add(st[5], Relaxed);
                tm::GUARDS.fetch_add(st[6], Relaxed);
                tm::OPT.fetch_add(st[7], Relaxed);
                tm::ELIMINATE.fetch_add(st[8], Relaxed);
                tm::METHOD_NANOS.fetch_add(n, Relaxed);
                if n >= 1_000_000 { tm::SLOW_1S.fetch_add(1, Relaxed); }
                else if n >= 500_000 { tm::SLOW_500MS.fetch_add(1, Relaxed); }
                else if n >= 100_000 { tm::SLOW_100MS.fetch_add(1, Relaxed); }
                if n >= 300_000 && timing_on {
                    eprintln!(
                        "[slow] {:#x} {}B {}ms lift={} cfg={} stack={} ssa={} prop={} res={} guards={} opt={} elim={}",
                        rva, bytes.len(), n / 1000,
                        st[0] / 1000, st[1] / 1000, st[2] / 1000, st[3] / 1000,
                        st[4] / 1000, st[5] / 1000, st[6] / 1000, st[7] / 1000,
                        st[8] / 1000,
                    );
                }
            }};
        }
        macro_rules! fail {
            ($reason:expr) => {{
                flush!();
                if timed_out(&deadline) {
                    tm::DEADLINE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                if dbg_fail {
                    eprintln!("[fail] {:#x} {}B: {}", rva, bytes.len(), $reason);
                }
                return None;
            }};
        }

        if !self.arch_is_arm64 && !self.is_x64 && !self.is_arm32 {
            fail!("unsupported arch");
        }
        if bytes.is_empty() {
            fail!("empty body");
        }
        // Verifier ledger must never leak excuses between methods on the
        // same worker thread (rayon reuses threads across methods).
        if verify_on {
            verify::ledger_clear();
        }
        if bytes.len() > self.max_method_bytes {
            fail!("body exceeds cap");
        }

        // 1. lift — cap IR size so pathological methods fall back to stubs
        // instead of dominating runtime (SSA cost is superlinear)
        const MAX_IR_INSNS: usize = 6_000;
        let t = stage_begin!("lift");
        let lifted_any = if self.arch_is_arm64 {
            lift::lift_arm64(bytes, rva, MAX_IR_INSNS.min(16_384), None)
                .map(|l| (l.insns, l.addresses, "lift failed"))
        } else if self.is_arm32 {
            lift_arm32::lift_arm32(bytes, rva, MAX_IR_INSNS.min(16_384))
                .map(|l| (l.insns, l.addresses, "arm32 lift failed"))
        } else {
            lift_x86::lift_x86_64(bytes, rva, MAX_IR_INSNS.min(16_384), self.x86_abi)
                .map(|l| (l.insns, l.addresses, "x64 lift failed"))
        };
        let Some((mut insns, mut addresses)) = lifted_any.map(|(i, a, _)| (i, a)) else {
            st[0] += t.elapsed().as_micros() as u64;
            fail!("lift failed");
        };
        // Out-of-line ARM64 load veneers (`b 0x2dff…` → shared GOT load →
        // back into the body). Resolve before the CFG and before the verifier
        // baseline: the veneer is a faithful part of the native body, and an
        // unresolved target would leave the whole body unreachable.
        if self.arch_is_arm64 {
            let n = resolve_load_veneers(
                &mut insns,
                &mut addresses,
                rva,
                rva + bytes.len() as u64,
                |a, n| il2cpp.read_bytes_at_rva(a, n),
            );
            if n > 0 && crate::recovery::env_flags::flag("ILR_COV") {
                eprintln!("[veneer] {rva:#x}: resolved {n} out-of-line load veneer(s)");
            }
        }
        // Verifier A/A baseline: fingerprint the RAW lift before any pass
        // touches the listing (cloned — passes mutate insns in place).
        let pre_pass_insns = if verify_on { insns.clone() } else { Vec::new() };
        // `ILR_AFTERLIFT=<rva>`: the listing the LIFTER produced, before any
        // pass touches it. Needed because `ILR_RAWLIFT` (below) is written after
        // `rewrite_pinned_copies`, so a register-identity bug cannot be
        // attributed to the lifter or to the pass from that output alone.
        if let Ok(want) = std::env::var("ILR_AFTERLIFT") {
            let want = u64::from_str_radix(want.trim_start_matches("0x"), 16).unwrap_or(u64::MAX);
            if want == rva {
                let mut d = String::new();
                for (i, insn) in insns.iter().enumerate() {
                    d.push_str(&format!(
                        "  L{i} @{:#x}: op={:?} operands={:?}\n",
                        insn.address, insn.op, insn.operands
                    ));
                }
                let _ = std::fs::write(probe_dir().join(format!("afterlift.{rva:#x}.txt")), d);
            }
        }
        // 1.5 ABI pin-copy rewriting (pre-CFG, register operands): IL2CPP's
        // ARM64 codegen pins the receiver/hot params in callee-saved regs
        // (`MOV x20, x0`) and later loads `this.*` fields off the pin. The
        // pin register itself gets spilled by the prologue, so seeding must
        // not target the register — instead we canonicalize every USE of a
        // pinned register to the identity register (x0/x1/…) while the copy
        // association is live. Loads become `[x0 + 192]`, which the local
        // seeding + field oracle resolves as `this.<field>`.
        locals::rewrite_pinned_copies(&mut insns, seed);
        if crate::recovery::env_flags::flag("ILR_RAWLIFT") {
            let raw_ok = std::env::var("ILR_RAWLIFT")
                .ok()
                .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
                .map(|f| f == rva)
                .unwrap_or(true);
            if raw_ok {
                let mut d = String::new();
                for (i, insn) in insns.iter().enumerate() {
                    d.push_str(&format!(
                        "  r{i} @{:#x}: op={:?} operands={:?}\n",
                        insn.address, insn.op, insn.operands
                    ));
                }
                let _ = std::fs::write(probe_dir().join(format!("rawlift.{:#x}.txt", rva)), d);
            }
        }
        // `ILR_DISASM=<rva>`: native disassembly of this method beside the IR
        // lifted from it, address by address, plus the coverage gap. This is
        // the per-method answer to "does the output match the disassembly?" —
        // see `lift::disasm_align_report`.
        if let Ok(want) = std::env::var("ILR_DISASM") {
            let want = u64::from_str_radix(want.trim_start_matches("0x"), 16).unwrap_or(u64::MAX);
            if want == rva {
                let arch = if self.arch_is_arm64 {
                    lift::TargetArch::Arm64
                } else if self.is_arm32 {
                    lift::TargetArch::Arm32
                } else {
                    lift::TargetArch::X64
                };
                eprintln!("{}", lift::disasm_align_report(bytes, rva, &insns, arch));
            }
        }
        st[0] += t.elapsed().as_micros() as u64;
        // Single-instruction bodies are usually THUNKS: `B other_method`
        // (shared generics / icall forwarders). Resolve them into a real
        // delegating call instead of failing — this recovers ~98% of all
        // previously-unrecoverable methods.
        if insns.len() == 1 && matches!(insns[0].op, Op::Jump(_)) {
            // chase thunk chains (thunk → thunk → … → real method), ≤8 hops.
            // Pre-Cfg the Jump field holds an ABSOLUTE TARGET RVA, not an
            // index — indexing `addresses` with it silently resolved thunks
            // to the thunk itself.
            let mut target_rva = match insns[0].op {
                Op::Jump(t) => t as u64,
                _ => addresses.first().copied().unwrap_or(0),
            };
            {
                for _ in 0..8 {
                    if self.maps.methods_by_rva.contains_key(&target_rva) {
                        break;
                    }
                    // peek next body: is it again a single B?
                    let Some(bytes2) = il2cpp.read_bytes_at_rva(target_rva, 64) else {
                        break;
                    };
                    let Some(l2) = lift::lift_arm64(&bytes2, target_rva, 8, None) else {
                        break;
                    };
                    if l2.insns.len() != 1 {
                        break;
                    }
                    match l2.insns[0].op {
                        Op::Jump(t2) => target_rva = t2 as u64,
                        _ => break,
                    }
                }
            }
            if let Some(m) = self.maps.methods_by_rva.get(&target_rva).cloned() {
                let returns_value = !matches!(
                    seed.ret_type,
                    ManagedType::Primitive(PrimKind::Void) | ManagedType::Unknown
                );
                let mut operands = vec![Operand::Method(m.clone()), Operand::Unknown];
                if !m.sig.is_static {
                    operands.push(Operand::Unknown);
                }
                for _ in &m.sig.params {
                    operands.push(Operand::Unknown);
                }
                let table = locals::create(&HashMap::new(), seed);
                // This thunk is faithful by construction (one `B` → a delegating
                // call), not differential-verified. Mark it graded so the
                // renderer does not stamp it "not verified" — a forwarding stub
                // is not a body with unknown findings.
                if verify_on {
                    verify::mark_graded_trusted();
                }
                return Some(RecoveredIr {
                    base_rva: rva,
                    insns: vec![
                        Insn::new(Op::Call, operands, insns[0].address),
                        Insn::new(Op::Return, vec![Operand::reg("x0")], insns[0].address),
                    ],
                    table,
                    returns_value,
                });
            }
            // unresolvable target — still emit a VISIBLE forwarding stub
            let table = locals::create(&HashMap::new(), seed);
            let ir = RecoveredIr {
                base_rva: rva,
                insns: vec![
                    Insn::new(
                        Op::Call,
                        vec![Operand::imm(target_rva as i64), Operand::Unknown],
                        insns[0].address,
                    ),
                    Insn::new(Op::Return, vec![Operand::reg("x0")], insns[0].address),
                ],
                table,
                returns_value: !matches!(
                    seed.ret_type,
                    ManagedType::Primitive(PrimKind::Void) | ManagedType::Unknown
                ),
            };
            // Unresolvable thunk target: the body is a single `B` to an address
            // the resolver could not name, emitted as a visible forwarding stub.
            // Trusted by construction (it IS one branch), so mark it graded
            // rather than letting the renderer stamp it "not verified".
            if verify_on {
                verify::mark_graded_trusted();
            }
            return Some(ir);
        }
        // single-instruction non-thunk bodies (lone ret / icall dispatch /
        // padding) flow through the normal pipeline rather than failing
        if insns.is_empty() {
            fail!("no IR insns");
        }
        // Superlinear passes ahead — bound input size. This is the same cap
        // the lifter was given; the old separate 3,000 "text cap" rejected
        // three real methods (3.0–3.5k IR insns) as recovery failures even
        // though the pipeline handles them.
        if insns.len() > MAX_IR_INSNS {
            fail!("too many IR insns");
        }

        // X64 unmanaged tail calls (see `recover_x64_unmanaged_tail`): a trailing
        // `jmp` to a non-managed helper must become `Call + Return` BEFORE the
        // CFG/DCE stages, or the setup feeding it looks dead and the body comes
        // out empty. x64-only: ARM64/ARM32 keep the tail-merge splicer below.
        // Ledgered as `TailCall`, the same rewrite the managed path performs.
        // `ILR_NOTAIL=1` disables the recovery, so the effect of this pass can
        // be measured against an otherwise identical build (like the managed
        // tail-call pass it mirrors).
        if self.is_x64 && !crate::recovery::env_flags::flag("ILR_NOTAIL") {
            let body_lo = rva;
            let body_hi = rva + bytes.len() as u64;
            let arab = self.x86_abi;
            // Ledger address: the trailing jump's own address (the rewrite
            // preserves it on the synthesized call).
            let jump_addr = insns
                .iter()
                .rposition(|i| !matches!(i.op, Op::Nop))
                .map(|j| insns[j].address)
                .unwrap_or(rva);
            let did = recover_x64_unmanaged_tail(
                &mut insns,
                &mut addresses,
                body_lo,
                body_hi,
                &|t| self.maps.methods_by_rva.contains_key(&t),
                &|t| il2cpp.read_bytes_at_rva(t, 4).is_some(),
                &|t| {
                    let Some(tbytes) = il2cpp.read_bytes_at_rva(t, 64) else {
                        return true;
                    };
                    let win = tbytes.len().min(64);
                    let Some(tlift) = lift_x86::lift_x86_64(&tbytes[..win], t, 32, arab) else {
                        return true;
                    };
                    passes::tail_branches_back_into(&tlift.insns, body_lo, body_hi)
                },
                &seed.ret_type,
            );
            if did && verify_on {
                verify::ledger_note(jump_addr, verify::LedgerKind::TailCall);
            }
        }

        // TAIL-MERGE CONTINUATION: the ARM64 compiler hoists shared method
        // endings (final call + return) OUT of the body — the last B jumps
        // PAST the metadata-declared size into a shared tail. Without the
        // continuation, the merge jump resolves to a dead sentinel, the CFG
        // loses the edge into the call block, SSA renames the call block with
        // ENTRY register versions, and DCE deletes everything feeding it
        // (the "missing Vector3 math" — every FP computation looked unused).
        // Follow out-of-body jumps (one hop, bounded) and splice the lifted
        // tail onto the listing so the CFG stays connected.
        // (listing idx of the leaving jump, its RVA target) — retargeted AFTER
        // Cfg::build, which rewrites raw RVAs to indices. This MUST stay the
        // binding the splice loop pushes into: a second, shadowing declaration
        // inside the block left the outer list permanently empty, so the
        // retarget pass after Cfg::build never ran and a leaving `B` kept
        // pointing at the shared tail's own internal target instead of the
        // first instruction actually spliced into this body.
        let mut tail_fixups: Vec<(usize, u64)> = Vec::new();
        // Out-of-body jumps recovered as tail CALLS rather than spliced: the
        // target is a real method body that never re-enters this one.
        let mut out_of_body_tail_calls: Vec<usize> = Vec::new();
        // The loop below re-runs `find_map` over the (growing) listing, so it
        // would find the SAME unresolved jump on every iteration and splice the
        // same tail again — three times, up to `hop`. Those copies carry
        // identical addresses, and `cfg.rs` maps an RVA to the FIRST index with
        // it, so the second and third copies become unreachable blocks: the
        // structurer never walks them (`CFG regions missing entirely`).
        let mut tail_handled: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut tail_spliced_targets: std::collections::HashSet<u64> =
            std::collections::HashSet::new();
        {
            let mut hop = 0usize;
            loop {
                // any unconditional jump whose target is outside the address
                // table (tail-merge exit) — not just the last insn: the lifter
                // appends a synthetic Return after the leaving B.
                let Some((jidx, _)) = insns.iter().enumerate().find_map(|(i, insn)| {
                    if tail_handled.contains(&i) {
                        return None;
                    }
                    match insn.op {
                        Op::Jump(t)
                            if addresses.get(t).is_none()
                                && ((t as u64) < rva
                                    || (t as u64) >= rva + bytes.len() as u64) =>
                        {
                            // A raw RVA outside the body. In-body raw RVAs are
                            // ordinary branches the CFG resolves; indices are
                            // already-resolved branches. Testing `addresses.get`
                            // alone treated an early in-body `b` as a tail exit.
                            Some((i, t as u64))
                        }
                        _ => None,
                    }
                }) else {
                    break;
                };
                if hop >= 3 {
                    break;
                }
                hop += 1;
                tail_handled.insert(jidx);
                // The lifter stored raw RVAs; Cfg::build rewrote them to
                // indices AFTER this point — but sentinel survives as
                // `insns.len()`-at-build-time. We need the ORIGINAL RVA:
                // recover it from the instruction address + recompute is
                // impossible post-rewrite, so instead read the pre-rewrite
                // target from `addresses`: impossible too. Simplest correct
                // route: re-lift the jump's own address to get its RVA.
                let jump_addr = insns[jidx].address;
                let Some(jbytes) = il2cpp.read_bytes_at_rva(jump_addr, 4) else {
                    break;
                };
                let word = u32::from_le_bytes([jbytes[0], jbytes[1], jbytes[2], jbytes[3]]);
                if word >> 26 != 0b000101 {
                    break; // not `B`
                }
                let imm26 = (word & 0x03FF_FFFF) as i64;
                let sext = (imm26 << 6) >> 6; // sign-extend 26
                let target_rva = (jump_addr as i64 + (sext << 2)) as u64;
                // inside the declared body? then it resolves normally, stop
                if target_rva >= rva && target_rva < rva + bytes.len() as u64 {
                    break;
                }
                // A metadata method's entry is a real callee, never a shared
                // tail-merge epilogue. Splicing its first window cuts the rest
                // of the callee (including its return) and leaves a fragment
                // whose backward branch becomes a self-loop. The tail-call
                // path below keeps the call and the return.
                if self.maps.methods_by_rva.contains_key(&target_rva) {
                    out_of_body_tail_calls.push(jidx);
                    continue;
                }
                // Another jump already spliced this exact target. Re-splicing
                // would append a duplicate (dead, address-colliding) copy; just
                // point this jump at the copy that is already there.
                if tail_spliced_targets.contains(&target_rva) {
                    tail_fixups.push((jidx, target_rva));
                    continue;
                }
                let Some(tbytes) = il2cpp.read_bytes_at_rva(target_rva, 256) else {
                    break;
                };
                // The tail lives OUTSIDE the method's metadata body; data
                // (literal pools, padding) can follow the real RET and one
                // undecodable word would fail the whole lift. Lift the first
                // 64B window; on failure shorten progressively.
                let mut lifted_tail = None;
                for win in [64usize, 32, 16] {
                    if let Some(t) =
                        lift::lift_arm64(&tbytes[..win.min(tbytes.len())], target_rva, 512, None)
                    {
                        lifted_tail = Some(t);
                        break;
                    }
                }
                let Some(tail) = lifted_tail else { break };
                // Trim to the first terminator: the fixed-size window can run
                // past the tail's RET into unrelated code — splicing that
                // would glue another function's body into this method's IR.
                let mut tail_len = tail.insns.len();
                if let Some(term) = tail.insns.iter().position(|i| i.is_terminator()) {
                    tail_len = tail_len.min(term + 1);
                }
                // CoreLib targets are not in `methods_by_rva`, but a real
                // method entry is still recognisable by its frame allocation
                // (`str x30,[sp,#-N]!` / `stp x30,…`). A shared epilogue
                // restore never grows the frame. Refuse the splice and keep
                // the call + return instead.
                if tail_starts_with_frame_allocation(&tail.insns[..tail_len]) {
                    out_of_body_tail_calls.push(jidx);
                    continue;
                }
                // Dispatch-stub redirect (NOT a tail-merge epilogue): some
                // out-of-body targets are generic-method dispatch stubs that
                // only reload callee-saved regs, compare a token and branch
                // back INTO the body (the real tail-merged call site sits
                // right after our jump, still in-body). Splicing the stub
                // glues duplicate blocks and severs the def→use chain for
                // every argument of that in-body call (missing FP math).
                // Detect: no Call in the lifted tail ⇒ stub ⇒ don't splice;
                // Cfg::build adds the fallthrough edge into the payload call
                // block instead.
                let tail_has_call = tail
                    .insns
                    .iter()
                    .take(tail_len)
                    .any(|i| matches!(i.op, Op::Call | Op::IndirectCall));
                if !tail_has_call {
                    // A tail with no call is USUALLY the dispatch stub described
                    // above: it reloads the callee-saved registers, compares a
                    // token and branches BACK INTO this body, where the real
                    // payload call still sits. Splicing it would duplicate
                    // blocks and sever the def→use chain, so it is left alone.
                    //
                    // But some out-of-body targets are REAL method bodies that
                    // merely contain no call in their first window. A delegate
                    // `EndInvoke` compiles to `mov x0, x1; mov x1, #0; b <body>`
                    // and tail-jumps straight into the implementation. Leaving
                    // that as a raw `Op::Jump(rva)` made `Cfg::build` map the
                    // unresolvable target to its one-past-the-end exit sentinel,
                    // and the structurer then rendered the whole method as an
                    // empty `{ // body not recoverable }`.
                    //
                    // The stub is defined by re-entering this body; a tail that
                    // never does is a real transfer. Record those for tail-call
                    // recovery below instead of dropping them.
                    let body_lo = rva;
                    let body_hi = rva + bytes.len() as u64;
                    let branches_back =
                        passes::tail_branches_back_into(&tail.insns[..tail_len], body_lo, body_hi);
                    if !branches_back {
                        out_of_body_tail_calls.push(jidx);
                        // Keep scanning: a method can tail-branch on more
                        // than one path (Serializer.AddLine has two).
                        continue;
                    }
                    break;
                }
                // splice: record the fixup (keep the jump's target as an RVA —
                // Cfg::build rewrites all raw RVAs in one pass, which also
                // picks up our spliced-tail addresses).
                let _splice_at = insns.len();
                insns.extend(tail.insns.into_iter().take(tail_len));
                addresses.extend(tail.addresses.into_iter().take(tail_len));
                tail_fixups.push((jidx, target_rva));
                tail_spliced_targets.insert(target_rva);
                // keep looping: the tail itself may chain into another tail
            }
        }

        // Recover the out-of-body tails recorded above as tail CALLS. The
        // renderer already degrades an unnamed call target to `fn_<address>(…)`,
        // which keeps both the real address the reader can look up and the
        // argument registers. Rewrite after the splice loop, high index first:
        // inserting the trailing `Op::Return` shifts every later index.
        // The register a synthesised call's result lands in. On the ARM ABIs the
        // result register IS argument register 0 (`x0`), so one name is right for
        // both. On x86-64 they are different physical registers — `rax` versus
        // `rdi`/`rcx` — and naming both `x0` is what made a call result and the
        // receiver one local (see `ir::RET_REG`).
        let ret_reg = if self.is_x64 { ir::RET_REG } else { "x0" };
        out_of_body_tail_calls.sort_unstable();
        for &jidx in out_of_body_tail_calls.iter().rev() {
            let Some(insn) = insns.get(jidx) else {
                continue;
            };
            let target = match insn.op {
                Op::Jump(t) => t as u64,
                _ => continue,
            };
            let a = insn.address;
            // Same operand layout the direct tail call builds in
            // `passes::recover_tail_calls` → `[target, dest, x0 … x7]`.
            let mut ops = vec![Operand::imm(target as i64), Operand::reg(ret_reg)];
            for r in ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"] {
                ops.push(Operand::reg(r));
            }
            // FP argument registers, parallel to `passes::recover_tail_calls`:
            // a value carried unchanged through the tail branch lives in
            // s0..s7 and a bare `b` records nothing about it, so a float
            // parameter would otherwise become `MissingAbiArg` and be dropped
            // from the rendered call.
            // Flattened lane-0 names for s0..s7 (see lift::scalar_s_name):
            for r in ["vs0", "vs4", "vs8", "vs12", "vs16", "vs20", "vs24", "vs28"] {
                ops.push(Operand::reg(r));
            }
            insns[jidx] = Insn::new(Op::Call, ops, a);
            if jidx + 1 <= insns.len() {
                insns.insert(
                    jidx + 1,
                    Insn::new(Op::Return, vec![Operand::reg(ret_reg)], a),
                );
                addresses.insert(jidx + 1, a);
            }
            // ONE native `b <other body>` became a CALL **and** a RETURN, so the
            // A/A fingerprint legitimately sees one more control-transfer event
            // than the raw lift. This is the SAME rewrite `recover_tail_calls`
            // performs below, and it must be ledgered identically: without the
            // note the drift allowance in `SemanticFingerprint::drift` (which
            // budgets one extra call + branch per `TailCall` site) charged every
            // one of these as unexplained synthesis. On arm64 that was the
            // dominant A/A class — 9,743 `calls GREW` findings across the iOS
            // corpus, all from tail calls this loop recovered.
            if verify_on {
                verify::ledger_note(a, verify::LedgerKind::TailCall);
            }
        }

        if timed_out(&deadline) {
            flush!();
            tm::DEADLINE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        // 1.4 tail calls: `b <code outside this method>`. Must run BEFORE the
        // CFG builds edges, because a tail call has no successor inside the
        // method and the CFG would otherwise record the branch as a dead exit
        // edge and the renderer would drop the call entirely.
        {
            let t = stage_begin!("tail");
            // Only a branch to a KNOWN method's code address is a tail call.
            // A shared-epilogue `b` (tail-merge) is not a method start and is
            // left alone — the splicer further down recovers that one.
            // `ILR_NOTAIL=1` disables the recovery, so the effect of this one
            // pass can be measured against an otherwise identical build.
            let n = if crate::recovery::env_flags::flag("ILR_NOTAIL") {
                0
            } else {
                passes::recover_tail_calls(
                    &mut insns,
                    &mut addresses,
                    &|t| self.maps.methods_by_rva.contains_key(&t),
                    ret_reg,
                )
            };
            st[0] += t.elapsed().as_micros() as u64;
            if n > 0 && crate::recovery::env_flags::flag("ILR_TAILDBG") {
                eprintln!("[tail] {n} tail call(s) at {rva:#x}");
            }
        }

        // 1.5 switch/jump-table recovery using real binary bytes. Must run
        // BEFORE cfg/SSA: the recovered Switch edges make the case-body blocks
        // reachable from the start, so dominators/SSA/stack cover them and the
        // renderer can walk every arm. When recovery ran AFTER SSA the arms
        // stayed disconnected islands (raw registers, never renamed) and the
        // renderer dropped the entire case payload — the census `Evaluate`
        // class lost ~700 of ~709 ops to exactly this.
        {
            let addrs: Vec<u64> = insns.iter().map(|i| i.address).collect();
            passes::recover_switch_tables(&mut insns, &addrs, &|a| {
                let b = il2cpp.read_bytes_at_rva(a, 4)?;
                Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            });
        }

        // Retarget tail-merge jumps to the FIRST spliced insn, BEFORE the CFG
        // pass, and keep the operand a raw RVA so that pass resolves it.
        //
        // This used to run AFTER `Cfg::build`, as `Op::Jump(first)` plus a
        // SECOND `Cfg::build`. That second pass ran over an IR whose branch
        // operands were already instruction INDICES, and `resolve_target`
        // interprets its argument as an RVA — so every branch in the method
        // failed to resolve and was rewritten to the exit sentinel. One
        // spliced tail therefore deleted the method's entire control flow:
        // all 67 native conditions of `UpdateControls` @0x117674 rendered as
        // straight-line code with no `if` at all, and the blocks after the
        // splice became unreachable (`CFG regions missing entirely`).
        for (jidx, target_rva) in &tail_fixups {
            if addresses.contains(target_rva) {
                insns[*jidx].op = Op::Jump(*target_rva as usize);
            }
        }

        // 1.6 EH landing pads, decoded BEFORE the CFG. The unwinder edge is
        // not in the CFG; the only synchronous path into a pad is the
        // unreachable `b <pad>` the compiler leaves after a noreturn raise.
        // Knowing the pad RVAs here lets the CFG keep each pad a block even
        // when that trampoline is gone, and lets SSA treat the pad as a kill
        // point (see `ssa::build_with_handler_entries`).
        let eh_pad_addrs: Vec<u64> = if il2cpp.eh_tables.is_empty() {
            Vec::new()
        } else {
            il2cpp
                .eh_tables
                .regions_for(rva, bytes.len() as u64, &|_| None)
                .into_iter()
                .map(|r| r.landing_pad)
                .filter(|&a| a != 0)
                .collect()
        };

        // 1.7 The compiler leaves `bl <raise>; b <pad>` after a raise that
        // never returns. The `b` is unreachable, but the CFG reads it as an
        // ordinary successor of the call, so the landing pad gets a
        // SYNCHRONOUS predecessor from the loop body — the edge that made
        // handlers render body code as the exception. A raise helper is
        // noreturn by proof, so terminate the block at the call instead: the
        // trampoline becomes the first handler instruction (a `throw` of the
        // exception the call was given), which keeps the pad a block without
        // inventing an edge into it.
        let noreturn_trampolines = {
            let rewritten = if crate::recovery::env_flags::flag("ILR_NO_EHTRAMP") {
                Vec::new()
            } else {
                neutralise_raise_trampolines(
                    &mut insns,
                    &mut addresses,
                    &eh_pad_addrs,
                    &self.maps.raise_exception,
                )
            };
            if verify_on {
                // Ledger the rewritten trampolines so the differential
                // verifier excuses the branch the compiler never executes.
                for &addr in &rewritten {
                    verify::ledger_note(addr, verify::LedgerKind::FoldedBranch);
                }
            }
            rewritten.len()
        };
        if noreturn_trampolines > 0 && crate::recovery::env_flags::flag("ILR_EHSSA") {
            eprintln!(
                "[ehssa] {rva:#x} neutralised {noreturn_trampolines} raise trampoline(s)"
            );
        }

        // 2. CFG (rewrites RVA branch targets to indices)
        let t = stage_begin!("cfg");
        let Some(mut graph) = cfg::Cfg::build_with_landing_pads(
            &mut insns,
            &addresses,
            &eh_pad_addrs,
        ) else {
            fail!("cfg build");
        };
        st[1] += t.elapsed().as_micros() as u64;

        // 2b. cross-block flag liveness (ARM32). The lifter's `pending_cmp` is
        // a LINEAR model: it is cleared at every flag writer in address order,
        // so a `bl`/`adds` on the FALL-THROUGH path (not the branch's path)
        // killed a compare that a later block still re-tests, and the branch
        // rendered `if (? != 0)`. This pass carries the reaching comparison
        // across CFG edges and rewrites those Unknown operands. It only edits
        // operand slots — no insertion — so the CFG built above stays valid.
        // This was gated to ARM32 on the premise that "the x86/arm64 lifters
        // carry their pending compare across flag-preserving control transfers".
        // That premise is FALSE, and measurably so: every lifter shares the same
        // LINEAR `pending_cmp` model, so a flag writer on the fall-through path
        // (or a `call`, which clobbers EFLAGS and — before the Call arm was
        // fixed — did not clear it) kills a compare a later block still re-tests
        // on x86 exactly as on ARM32. The pass itself is architecture-agnostic:
        // it keys on `Op::Compare` + register operands + the CFG, and it only
        // rewrites an operand slot that is `Unknown`/`MissingAbiArg` when the
        // reaching comparison is PROVED, so enabling it on the other two lifters
        // cannot fabricate a condition — it can only replace an unknown with the
        // value the machine actually tested. The x64 corpus reports 2,247
        // conditions compared against `__unknown`, which is the population this
        // reaches.
        {
            let n = passes::resolve_cross_block_conditions(&mut insns, &graph);
            if n > 0 && crate::recovery::env_flags::flag("ILR_FLAGDBG") {
                eprintln!("[flags] {n} cross-block condition(s) recovered at {rva:#x}");
            }
        }

        // 3. native stack slots (pointer size matters for slot arithmetic)
        let ptr_size: u64 = if il2cpp.is_32bit { 4 } else { 8 };
        let t = stage_begin!("stack");
        stack_analyzer::analyze_with_ptr_size(&mut insns, &graph, ptr_size);
        st[2] += t.elapsed().as_micros() as u64;
        if let Ok(f) = std::env::var("ILR_POSTSTACK") {
            // ILR_POSTSTACK=1 → all; =0xRVA → that method only
            let ok = match f.as_str() {
                "1" => true,
                _ => u64::from_str_radix(f.trim_start_matches("0x"), 16).ok() == Some(rva),
            };
            if ok {
                let mut d = String::new();
                for (i, insn) in insns.iter().enumerate() {
                    d.push_str(&format!(
                        "  s{i} @{:#x}: op={:?} operands={:?}\n",
                        insn.address, insn.op, insn.operands
                    ));
                }
                let _ = std::fs::write(probe_dir().join("poststack.txt"), d);
            }
        }

        if timed_out(&deadline) {
            flush!();
            tm::DEADLINE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }

        // 4. dominance + SSA
        let t = stage_begin!("ssa");
        let dom = dominator::DominatorInfo::compute(&graph);
        if let Ok(f) = std::env::var("ILR_SSA_GRAPH") {
            if u64::from_str_radix(f.trim_start_matches("0x"), 16).ok() == Some(rva) {
                let mut d = String::new();
                for (bi, b) in graph.blocks.iter().enumerate() {
                    d.push_str(&format!(
                        "B{}: insns {}..{} succ={:?} idom={:?} df={:?} children={:?}\n",
                        bi,
                        b.start,
                        b.end,
                        graph.successors[bi],
                        dom.idom[bi],
                        dom.frontiers[bi],
                        dom.dom_children[bi]
                    ));
                }
                let _ = std::fs::write(probe_dir().join("ssagraph.txt"), d);
            }
        }
        // Caller-saved HFA return registers: s0,s1,s2,s3 are v0,v1,v2,v3
        // lane0, i.e. `vs0,vs4,vs8,vs12` under the lifter's lane indexing
        // (ARM32 uses consecutive `vs0..vs3`).
        let fp_clobbers: Vec<String> = if self.arch_is_arm64 {
            ["vs0", "vs4", "vs8", "vs12"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else if self.is_arm32 {
            ["vs0", "vs1", "vs2", "vs3"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };
        // Handler-entry blocks: the pad's block is a kill point for SSA.
        // Map each pad RVA through the same at-or-after slide the CFG used.
        let handler_blocks: std::collections::HashSet<usize> = if crate::recovery::env_flags::flag("ILR_NO_EHKILL") {
            std::collections::HashSet::new()
        } else {
            let mut set = std::collections::HashSet::new();
            for &pad in &eh_pad_addrs {
                let mut best: Option<usize> = None;
                let mut best_addr = u64::MAX;
                for (i, &a) in addresses.iter().enumerate() {
                    if a >= pad && a < best_addr {
                        best_addr = a;
                        best = Some(i);
                    }
                }
                if let Some(i) = best {
                    if let Some(b) = graph.blocks.iter().position(|blk| blk.start == i) {
                        set.insert(b);
                    }
                }
            }
            set
        };
        if crate::recovery::env_flags::flag("ILR_EHSSA") && !eh_pad_addrs.is_empty() {
            eprintln!(
                "[ehssa] {rva:#x} pads={:?} kill_blocks={:?}",
                eh_pad_addrs, handler_blocks
            );
        }
        let mut ssa_out =
            ssa::build_with_handler_entries(&insns, &graph, &dom, &fp_clobbers, &handler_blocks);
        ssa::materialize(&mut ssa_out, &mut graph);
        // FP-returning methods leave their result in v0 (s0/d0/xmm0), but
        // the lifter models every RET as returning x0 — rebind the Return to
        // the FP register's live version BEFORE DCE, or the whole float
        // computation is deleted as dead (`FallingFromRestY` rendered as
        // `return v3;` with its FMUL chain Nop'd away).
        passes::bind_fp_returns(&mut ssa_out.insns, &ssa_out.version_slot, &seed.ret_type);
        st[3] += t.elapsed().as_micros() as u64;
        if crate::recovery::env_flags::flag("ILR_POSTSSA") {
            // optional RVA filter: ILR_POSTSSA=0x2db38 dumps only that method
            let rva_ok = std::env::var("ILR_POSTSSA")
                .ok()
                .and_then(|v| {
                    u64::from_str_radix(v.trim_start_matches("0x"), 16)
                        .or_else(|_| v.parse::<u64>())
                        .ok()
                })
                .map(|f| f == rva)
                .unwrap_or(true);
            if rva_ok {
                let mut d = String::new();
                for (i, insn) in ssa_out.insns.iter().enumerate() {
                    d.push_str(&format!(
                        "  p{i} @{:#x}: op={:?} operands={:?}\n",
                        insn.address, insn.op, insn.operands
                    ));
                }
                let _ = std::fs::write(probe_dir().join(format!("postssa.{:#x}.txt", rva)), d);
            }
        }

        // 5. typed locals + propagation (metadata-backed field resolution)
        let t = stage_begin!("propagate");
        let mut table = locals::create(&ssa_out.version_slot, seed);
        // A call-clobber version has no listing definition; its value is
        // whatever the CALL delivered to that ABI register. That is an
        // incoming ABI value (the same category as an undeclared argument
        // register), so the undefined-local oracle must not report it.
        for &v in &ssa_out.call_defined {
            if let Some(&li) = table.index_of.get(&v) {
                table.infos[li].is_call_clobber = true;
            }
        }
        // `this` cannot be assigned: an entry version the body writes is a
        // plain local that starts out holding the this pointer (the
        // loop-carried register of a rotated loop).
        locals::unclaim_written_entry_this(&mut table, &ssa_out.insns);
        // CALL-ARGUMENT ALIGNMENT, before the scrub so the scrub sees every
        // argument still under its own ABI register name.
        //
        // The lifter records each `bl` as a REGISTER DUMP
        // (`[target, x0 result, x0…x7, fp-written…]`). That is AAPCS64 order,
        // not parameter order, so reading it positionally gave a float
        // parameter whatever the next INTEGER register held, and
        // `trim_call_arguments` — which truncates the dump to the declared
        // parameter count — deleted every FP argument outright (they sit at
        // the end of the dump). Both effects are silent: the call keeps a
        // plausible shape and loses one argument.
        let fp_arm64 = ssa_out
            .version_slot
            .values()
            .any(|o| o.starts_with("vs") || o.starts_with("vd"));
        //
        // A FOLDED address is deliberately NOT skipped here, though the
        // argument for it is real: `methods_by_rva` keeps one arbitrary member
        // of the merged set, so aligning to it rewrites the call's own ABI
        // register dump in a shape that belongs to a different method (when the
        // arbitrary member is STATIC, every argument is dropped outright).
        // Skipping them was implemented and MEASURED on the v24.5 dump:
        // `folded_no_recv` 50 -> 4 and `aligned_call_args` 39,304 -> 32,332 —
        // the pick really does then see the call's own register dump — but
        // `folded_picked` stayed at 234 and the dump traded 22 real operands for
        // 22 `__abi` holes across 1,565 relines. The reason the dump gets worse
        // is a SEPARATE, pre-existing defect this change only exposes: the x64
        // lifter names `rax` and `rdi` both `x0` (`gpr_name_for_ordinal`), so an
        // `x0` slot can resolve to a version that is really a discarded RETURN
        // value, and the raw dump then hands that version the receiver's place
        // (`frenchSort.Update()` for a `bool` parameter). Fixing the alias is the
        // prerequisite; until then, aligning is the lesser error.
        let aligned = locals::align_call_arguments(
            &mut ssa_out.insns,
            &table,
            |a| self.maps.methods_by_rva.get(&a).cloned(),
            fp_arm64,
        );
        if aligned > 0 {
            tm::ALIGNED_ARGS.fetch_add(aligned as u64, std::sync::atomic::Ordering::Relaxed);
        }
        // ABI-garbage scrub: unresolved calls render their FULL x0–x7
        // register-dump args (`fn_4a58bac(v28, v29, v15, …)`), and the
        // spill reads behind them create undefined entry-version locals
        // (`v41 = v10;` with v10 never defined). Args that are not declared
        // entry values become MissingAbiArg, which every render path drops.
        locals::scrub_abi_call_args(&mut ssa_out.insns, &table);
        // `ILR_DUMP_STAGE=0xRVA` — with `ILR_POSTSSA=0xRVA` this brackets the
        // ABI scrub: the two files show exactly which operand the scrub
        // rewrote, which is otherwise invisible because `MissingAbiArg` renders
        // as a bare token with no trace of its origin.
        dump_ir_stage(&ssa_out.insns, rva, "postscrub");
        // ILR_LOCALS=1 → all methods; ILR_LOCALS=0xRVA → just that method
        let locals_dump = match std::env::var("ILR_LOCALS") {
            Ok(v) if v != "1" => {
                u64::from_str_radix(v.trim_start_matches("0x"), 16).ok() == Some(rva)
            }
            Ok(_) => true,
            Err(_) => false,
        };
        if locals_dump {
            for (v, slot) in &ssa_out.version_slot {
                let li = table.index_of.get(v).copied().unwrap_or(usize::MAX);
                eprintln!(
                    "[locals] v{} slot={} idx={} this={} param={:?} origin={}",
                    v,
                    slot,
                    li,
                    li != usize::MAX && table.infos[li].is_this,
                    if li != usize::MAX {
                        table.infos[li].is_param
                    } else {
                        None
                    },
                    if li != usize::MAX {
                        table.infos[li].slot_origin.as_str()
                    } else {
                        "?"
                    }
                );
            }
        }
        let oracle = BridgeOracle {
            metadata,
            il2cpp,
            chains: &self.parent_chains,
            cache: std::cell::RefCell::new(HashMap::new()),
            field_index_by_addr: &self.field_index_by_addr,
            field_owner_tdi: &self.field_owner_tdi,
        };
        locals::propagate_types(&mut ssa_out.insns, &mut table, &oracle);
        dump_ir_stage(&ssa_out.insns, rva, "posttypes");
        st[4] += t.elapsed().as_micros() as u64;

        if timed_out(&deadline) {
            flush!();
            tm::DEADLINE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        // 6. metadata resolution (+ FieldInfo usage → named static fields)
        let mut field_owner = |fi: u32| -> Option<(String, String, ManagedType)> {
            self.field_info(metadata, il2cpp, fi)
        };
        let ps = if il2cpp.is_32bit { 4u64 } else { 8 };
        let t = stage_begin!("resolve");
        let mut stats = resolve::run_with_fields(
            &mut ssa_out.insns,
            &table,
            &self.maps,
            &self.field_index_by_addr,
            &mut field_owner,
            ps,
            &self.parent_chains,
        );
        // Delegate + dynamic-slot wiring (Worker K): `run_with_fields`
        // already proved `InstanceField invoke_impl` targets inside
        // `rewrite_virtual_calls` (`resolve_indirect_target` → `Invoke` with
        // the delegate base as receiver; `Arith`-base `klass + idx*16 + 312`
        // chains fold to a constant only with proof, else refuse and keep
        // `fnptr_` — see the `delegate_*`/`arith_slot_*` counters in
        // `ILR_VERIFY`). This call is the backstop for pre-SSA `Memory`
        // delegate loads and any typed `Call` the vtable walk refused; it
        // reuses the same strict proof without double-counting, so already-
        // resolved `Method(Invoke)` targets are skipped untouched.
        resolve::recover_delegate_invokes(&mut ssa_out.insns, &table, ps, &mut stats);
        resolve::resolve_methodinfo_usages(
            &mut ssa_out.insns,
            &self.method_index_by_addr,
            &mut |mi| self.method_info_ref(metadata, il2cpp, mi),
            &mut stats,
        );
        // GENERIC RETURN SPECIALIZATION: a shared generic body's MethodRef
        // returns `T`, but the call site's declaring type carries the concrete
        // instantiation (`List<Vector3>`). Without this, `get_Item` did not
        // even look like a struct return, so its HFA components were never
        // bound and later reads reused stale s-register versions — measured
        // as `(z - z)` collapsing real distance math to zero.
        let specialized =
            resolve::specialize_generic_return_sigs(&mut ssa_out.insns, &self.maps);
        if specialized > 0 {
            static GSPEC: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            GSPEC.fetch_add(specialized as u64, std::sync::atomic::Ordering::Relaxed);
        }
        dump_ir_stage(&ssa_out.insns, rva, "postresolve");
        // A float/double-returning call leaves its result in v0, but every
        // call only defines the integer return slot — bind the FP slot to the
        // call result now that `resolve` has filled each call's `dest_type`.
        // Before DCE: an unbound read makes the call look partially dead.
        // HFA components are bound as `result.<field>` (one DISTINCT value per
        // register): binding them all to the same local made every Vector3
        // component equal.
        let hfa_stride = if self.arch_is_arm64 { 4 } else { 1 };
        passes::bind_fp_call_returns(
            &mut ssa_out.insns,
            &ssa_out.version_slot,
            hfa_stride,
            |ty| match ty {
                ManagedType::Class(nt) if nt.type_def_index >= 0 => {
                    hfa_field_refs(nt, metadata, il2cpp)
                }
                _ => Vec::new(),
            },
        );
        st[5] += t.elapsed().as_micros() as u64;

        // 6b. Post-resolve type propagation.
        //
        // `propagate_types` (step 5) ran BEFORE `resolve`, so most call targets
        // were still `Immediate(addr)` rather than `Method(m)`. After resolve
        // converts them (including shared generic bodies, native wrappers, and
        // virtual dispatch), re-run a targeted type propagation that picks up
        // resolved calls and propagates their return types to destination locals.
        // This is more surgical than re-running the full `propagate_types` — it
        // only processes calls, moves, and memory loads, without re-doing the
        // arith field-address seeding that could interfere with existing resolutions.
        {
            let t = stage_begin!("post-resolve types");
            let oracle2 = BridgeOracle {
                metadata,
                il2cpp,
                chains: &self.parent_chains,
                cache: std::cell::RefCell::new(HashMap::new()),
                field_index_by_addr: &self.field_index_by_addr,
                field_owner_tdi: &self.field_owner_tdi,
            };
            interprocedural::propagate_post_resolve(&mut ssa_out.insns, &mut table, &oracle2);
            st[5] += t.elapsed().as_micros() as u64;
        }

        // 7. class-init guard excision (before folding so constants settle)
        let t = stage_begin!("guards");
        passes::remove_init_guards(&mut ssa_out.insns, &mut graph);
        dump_ir_stage(&ssa_out.insns, rva, "postguards");

        st[6] += t.elapsed().as_micros() as u64;

        if timed_out(&deadline) {
            flush!();
            tm::DEADLINE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        // 8. optimization loop (ConstantFolder ⇄ SsaSimplifier ≤8, then DCE)
        let t = stage_begin!("opt");
        for _ in 0..8 {
            let p = passes::copy_and_constant_propagation(&mut ssa_out.insns);
            let f = passes::constant_folding(&mut ssa_out.insns);
            if p + f == 0 {
                break;
            }
        }
        dump_ir_stage(&ssa_out.insns, rva, "postoptloop");
        passes::dead_code_elimination(&mut ssa_out.insns);
        dump_ir_stage(&ssa_out.insns, rva, "postdce");

        // 9. structural cleanup
        passes::simplify_boolean_flags(&mut ssa_out.insns);
        passes::trim_call_arguments(&mut ssa_out.insns);
        let removed_guards = passes::remove_null_check_guards(&mut ssa_out.insns, &graph);
        // NOTE: do NOT rebuild the cfg here. `ssa::build` recorded each phi's
        // predecessor BLOCK INDICES; a rebuild after Nop'ing guard branches
        // merges blocks, shifts those indices, and `eliminate` would index
        // out of range (the guard-removal Nops don't affect the phi edge
        // moves — each pred block's Nop'd terminator just reads as a non-
        // terminator and the move is placed after it). Nothing reads the CFG
        // after `eliminate`, so it stays stale rather than paying a rebuild.
        let _ = removed_guards;
        passes::dead_code_elimination(&mut ssa_out.insns);
        st[7] += t.elapsed().as_micros() as u64;
        dump_ir_stage(&ssa_out.insns, rva, "postopt");
        let t = stage_begin!("eliminate");
        ssa::eliminate(&mut ssa_out, &mut graph);
        // Bracket the phi-elimination pass: `eliminate` rewrites every Phi into
        // edge copies and the two DCEs below then delete dead ones. Without a
        // dump between them, "a version is read but never defined" cannot be
        // attributed — it is either `eliminate` never inserting the copy or a
        // DCE deleting a copy whose use survives. `stage.postedgemoves` answers
        // that in one comparison (`ILR_DUMP_STAGE=0xRVA`).
        dump_ir_stage(&ssa_out.insns, rva, "postedgemoves");
        passes::dead_code_elimination(&mut ssa_out.insns);
        dump_ir_stage(&ssa_out.insns, rva, "postelimdce");
        // Liveness DCE on the final (phi-free) listing: breaks self-cyclic
        // dead chains the use-count DCE cannot see (compiler-kept loop
        // counters that are written each iteration, read only by their own
        // next write, never read at any exit) — the `v31 -= 1` half of
        // `while (*(p-8) != 0) { v30 += 16; v31 -= 1; }` scans.
        passes::liveness_dce(&mut ssa_out.insns);
        dump_ir_stage(&ssa_out.insns, rva, "postlive");
        // Collapse no-op conditional branches. This MUST run after BOTH DCEs,
        // and that ordering is the whole point: the φ edge copies `eliminate`
        // materialized on the skipped edge look live until the DCEs prove them
        // dead. On `get_mainTexture` @0x18694 the guard at 0x18750 skips five
        // such copies at 0x18754 — they are unused, so `dead_code_elimination`
        // erases them, and ONLY THEN is the skipped slot empty and the branch
        // provably a no-op. Running one pass earlier sees a live-looking arm
        // and correctly declines, which is how this went unnoticed.
        passes::drop_vacuous_cond_branches(&mut ssa_out.insns);
        st[8] += t.elapsed().as_micros() as u64;
        dump_ir_stage(&ssa_out.insns, rva, "postelim");

        flush!();

        // 9b. SIMD lane selections outside their register's own lanes.
        //
        // NOT gated on `ILR_VERIFY`, and not a coverage statistic: the
        // register-granular SIMD model has no type that could catch a lifter
        // computing a lane index wrong, so an out-of-range pick is a silently
        // WRONG VALUE. The scan is one `match` per instruction and allocates
        // only on a violation, so it is affordable on every method; the
        // aggregate lands in the `[verify]` summary and the first violation
        // prints unconditionally.
        verify::record_out_of_range_lane_selections(
            rva,
            &verify::out_of_range_lane_selections(&ssa_out.insns),
        );

        // 10. differential verification (ILR_VERIFY=1): decoded-assembly
        // ground truth vs final IR + CFG invariants + A/A drift.
        if verify_on {
            let final_cfg = ssa::rebuild_cfg(&ssa_out.insns);
            verify::verify_and_log(
                bytes,
                rva,
                &pre_pass_insns,
                &ssa_out.insns,
                Some(&final_cfg),
                Some(&table),
                self.arch_is_arm64,
                self.is_x64,
                self.is_arm32,
            );
        }

        let returns_value = !matches!(
            seed.ret_type,
            ManagedType::Primitive(PrimKind::Void) | ManagedType::Unknown
        );

        Some(RecoveredIr {
            insns: ssa_out.insns,
            table,
            returns_value,
            base_rva: rva,
        })
    }
}

/// Directory for the `ILR_*` debug probes.
///
/// These probes wrote to a hardcoded POSIX `/tmp/tel`, which does not exist on
/// Windows: `std::fs::write` there targets a drive-relative path whose parent
/// is never created, so EVERY probe silently failed on the platform the dumps
/// are actually produced on. A diagnostic that cannot write is worse than no
/// diagnostic — it looks like the stage produced nothing. `temp_dir()` is the
/// OS's own answer on all three platforms.
pub fn probe_dir() -> std::path::PathBuf {
    // Scripts and agent runs must not scatter outputs into the system temp dir
    // (which is wiped on container recycle). Prefer an explicit override, then
    // the project-local `tmp/probe`, and only fall back to the OS temp dir.
    let d = match std::env::var_os("ILR_TMP_DIR") {
        Some(v) => std::path::PathBuf::from(v),
        None => std::path::PathBuf::from("tmp").join("probe"),
    };
    let _ = std::fs::create_dir_all(&d);
    d
}

/// Debug: dump a pipeline stage's IR for one method (ILR_DUMP_STAGE=0xRVA).
fn dump_ir_stage(insns: &[Insn], rva: u64, stage: &str) {
    let Ok(f) = std::env::var("ILR_DUMP_STAGE") else {
        return;
    };
    let t = f.trim_start_matches("0x");
    let Some(want) = u64::from_str_radix(t, 16)
        .or_else(|_| t.parse::<u64>())
        .ok()
    else {
        return;
    };
    if want != rva {
        return;
    }
    let mut d = String::new();
    for (i, insn) in insns.iter().enumerate() {
        d.push_str(&format!(
            "  {stage} {i} @{:#x}: op={:?} operands={:?}\n",
            insn.address, insn.op, insn.operands
        ));
    }
    let _ = std::fs::write(probe_dir().join(format!("stage.{stage}.{:#x}.txt", rva)), d);
}

/// Resolve out-of-line ARM64 load veneers before the CFG.
///
/// The linker shares the GOT load for a static/class pointer: a body ends in
/// `b 0x2dff1a8`, and that address is a 2-instruction stub
/// `ldr x19,[x19,#off]; b <continuation inside the method>`. The CFG cannot
/// follow an out-of-body target, so the entry edge was lost and the entire
/// real body became unreachable (`[covloss]` 49/50, 43/44, 35/36).
///
/// Splice the veneer's pure loads at the branch site (keeping the branch's
/// native address) and retarget to the in-body continuation. Anything that
/// allocates a frame, stores memory, calls, or branches anywhere but forward
/// into this body is refused and left as an exit edge.
fn resolve_load_veneers(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    body_lo: u64,
    body_hi: u64,
    read: impl Fn(u64, usize) -> Option<Vec<u8>>,
) -> usize {
    if insns.len() != addresses.len() {
        return 0;
    }
    let mut resolved = 0usize;
    let mut i = 0usize;
    while i < insns.len() {
        let Op::Jump(t) = insns[i].op else {
            i += 1;
            continue;
        };
        let target = t as u64;
        if target >= body_lo && target < body_hi {
            i += 1;
            continue;
        }
        let Some(vbytes) = read(target, 16) else {
            i += 1;
            continue;
        };
        let Some(lifted) = lift::lift_arm64(&vbytes, target, 16, None) else {
            i += 1;
            continue;
        };
        let mut end = lifted.insns.len();
        if let Some(term) = lifted.insns.iter().position(|x| x.is_terminator()) {
            end = end.min(term + 1);
        }
        let Some((last, init)) = lifted.insns[..end].split_last() else {
            i += 1;
            continue;
        };
        let Op::Jump(cont) = last.op else {
            i += 1;
            continue;
        };
        let cont = cont as u64;
        // The continuation must prove ownership: a forward in-body address.
        if cont < body_lo || cont >= body_hi || cont <= insns[i].address {
            i += 1;
            continue;
        }
        // A veneer is 1..=3 pure loads; a frame setup, store, call, compare or
        // in-stub branch is a real function body, not a linker stub.
        let mut ok = init.len() <= 3;
        for x in init {
            if !matches!(x.op, Op::Move | Op::Nop) {
                ok = false;
                break;
            }
            if let Some(pos) = Insn::dest_position(&x.op) {
                match x.operands.get(pos) {
                    Some(Operand::Register(r))
                        if !matches!(r.as_str(), "sp" | "x29" | "x30") => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if !ok {
            i += 1;
            continue;
        }
        let addr = insns[i].address;
        if init.is_empty() {
            insns[i] = Insn::new(Op::Jump(cont as usize), vec![], addr);
            resolved += 1;
            i += 1;
            continue;
        }
        let mut new_insns: Vec<Insn> = init
            .iter()
            .cloned()
            .map(|mut x| {
                x.address = addr;
                x
            })
            .collect();
        let mut new_addrs = vec![addr; new_insns.len()];
        let mut tail_jump = last.clone();
        tail_jump.address = addr;
        new_insns.push(tail_jump);
        new_addrs.push(addr);
        let n = new_insns.len();
        insns.splice(i..=i, new_insns);
        addresses.splice(i..=i, new_addrs);
        resolved += 1;
        i += n;
    }
    resolved
}

/// Whether a lifted out-of-body window begins like a real method: a frame
/// allocation before the first terminator. A shared tail-merge continuation
/// never allocates a frame, so splicing a window that does glues another
/// function's prologue and body into this method's IR — the CFG then turns
/// the fragment's backward branch into a self-loop (`while (true)`).
pub(crate) fn tail_starts_with_frame_allocation(insns: &[Insn]) -> bool {
    insns
        .iter()
        .take_while(|i| !i.is_terminator())
        .take(8)
        .any(|i| matches!(i.op, Op::ShiftStack(d) if d < 0))
}

/// Result of [`RecoveryContext::recover_ir`].
pub struct RecoveredIr {
    pub insns: Vec<Insn>,
    pub table: locals::LocalTable,
    pub returns_value: bool,
    /// Native RVA of the first instruction (base of the lifted body). EH
    /// try regions are RVA ranges; the renderer maps them onto statements
    /// through `insns[i].address` relative to this base.
    pub base_rva: u64,
}

/// Resolver used when no live Resolution exists (tests): emits diagnostics.
pub struct NullResolver;

impl emit::TokenResolver for NullResolver {
    fn user_method(&self, _m: &MethodRef) -> Option<dotnetdll::resolved::members::UserMethod> {
        None
    }
    fn field_source(&self, _f: &FieldRef) -> Option<dotnetdll::resolved::members::FieldSource> {
        None
    }
    fn method_type(&self, t: &ManagedType) -> Option<dotnetdll::resolved::types::MethodType> {
        managed_to_dotnetdll(t)
    }
}

/// Pure conversion ManagedType → dotnetdll MethodType WITHOUT a live
/// Resolution: class/array types degrade to System.Object (valid IL).
/// The production resolver inside dummy_assembly_generator pushes real
/// external type references instead.
pub fn managed_to_dotnetdll(t: &ManagedType) -> Option<dotnetdll::resolved::types::MethodType> {
    use dotnetdll::resolved::types::BaseType as BT;
    use dotnetdll::resolved::types::MethodType as MT;
    let base = match t {
        ManagedType::Primitive(p) => match p {
            PrimKind::Void => return None,
            PrimKind::Boolean => BT::Boolean,
            PrimKind::Char => BT::Char,
            PrimKind::I1 => BT::Int8,
            PrimKind::U1 => BT::UInt8,
            PrimKind::I2 => BT::Int16,
            PrimKind::U2 => BT::UInt16,
            PrimKind::I4 => BT::Int32,
            PrimKind::U4 => BT::UInt32,
            PrimKind::I8 => BT::Int64,
            PrimKind::U8 => BT::UInt64,
            PrimKind::R4 => BT::Float32,
            PrimKind::R8 => BT::Float64,
            PrimKind::I | PrimKind::U => BT::IntPtr,
        },
        ManagedType::Class(_) | ManagedType::Opaque(_) | ManagedType::SzArray(_) => BT::Object,
        ManagedType::Object => BT::Object,
        ManagedType::String => BT::String,
        ManagedType::IntPtr => BT::IntPtr,
        ManagedType::ByRef(_) | ManagedType::Pointer(_) => BT::IntPtr,
        ManagedType::Unknown => return None,
    };
    Some(MT::Base(Box::new(base)))
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod probe_tests;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

impl RecoveryContext {
    /// Resolve a global field index → (type name, field name, managed type).
    pub fn field_info(
        &self,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        fi: u32,
    ) -> Option<(String, String, ManagedType)> {
        // O(1) owner lookup via the prebuilt index (was a linear scan over
        // ALL type defs per call, inside the typing fixpoint loop)
        let tdi = *self.field_owner_tdi.get(&fi)? as usize;
        let td = metadata.type_defs.get(tdi)?;
        let fdef = metadata.field_defs.get(fi as usize)?;
        let name = metadata
            .get_string_from_index(fdef.name_index)
            .unwrap_or_default();
        let ty = il2cpp
            .types
            .get(fdef.type_index as usize)
            .map(|t| managed_type_of(t, metadata, il2cpp, 0))
            .unwrap_or(ManagedType::Unknown);
        let tname = metadata
            .get_string_from_index(td.name_index)
            .unwrap_or_default();
        Some((tname, name, ty))
    }

    /// Pre-compute instance field layouts for every TypeDef.
    ///
    /// For each type, iterates its instance fields and records
    /// `(tdi, byte_offset) → (field_name, declaring_type_name)` in `maps.field_layouts`,
    /// and `(tdi, byte_offset) → field_type_tdi` in `maps.field_types`.
    /// Walks parent chains so inherited fields are reachable from the child's tdi.
    pub fn populate_field_layouts(
        maps: &mut resolve::ResolveMaps,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        parent_chains: &HashMap<i32, Vec<i32>>,
        generic_names: &HashMap<u64, String>,
    ) {
        // Pre-compute well-known type def indices for String and Object.
        // These are needed because fields of type `string` have type_enum 0x0E
        // and fields of type `object` have type_enum 0x1C, neither of which
        // carries a klass_index directly.
        let mut string_tdi: Option<i32> = None;
        let mut object_tdi: Option<i32> = None;
        for (ti, td) in metadata.type_defs.iter().enumerate() {
            let ns = metadata.get_string_from_index(td.namespace_index).unwrap_or_default();
            let name = metadata.get_string_from_index(td.name_index).unwrap_or_default();
            if ns == "System" && name == "String" {
                string_tdi = Some(ti as i32);
            } else if ns == "System" && name == "Object" {
                object_tdi = Some(ti as i32);
            }
            if string_tdi.is_some() && object_tdi.is_some() {
                break;
            }
        }

        let empty = Vec::new();
        for tdi in 0..metadata.type_defs.len() {
            let tdi = tdi as i32;
            let chain = parent_chains.get(&tdi).unwrap_or(&empty);
            for &tdi_walk in chain {
                if tdi_walk < 0 || tdi_walk as usize >= metadata.type_defs.len() {
                    continue;
                }
                let td_walk = &metadata.type_defs[tdi_walk as usize];
                let field_start = td_walk.field_start as usize;
                let field_count = td_walk.field_count as usize;
                let declaring_name = metadata
                    .get_string_from_index(td_walk.name_index)
                    .unwrap_or_default();
                for fi in 0..field_count {
                    let global_fi = field_start + fi;
                    let Some(field_def) = metadata.field_defs.get(global_fi) else {
                        continue;
                    };
                    let Some(field_il2cpp_type) = il2cpp.types.get(field_def.type_index as usize) else {
                        continue;
                    };
                    let is_static = (field_il2cpp_type.attrs & 0x0010) != 0;
                    if is_static {
                        continue;
                    }
                    let raw_offset = il2cpp.get_field_offset_from_index(
                        tdi_walk as usize,
                        fi,
                        global_fi,
                        td_walk.is_value_type(),
                        is_static,
                    );
                    let name = metadata
                        .get_string_from_index(field_def.name_index)
                        .unwrap_or_else(|_| format!("field_{global_fi}"));
                    // The declared generic INSTANTIATION (`Dictionary<int,
                    // Token>`) is authoritative for a `new` stored into this
                    // field; shared-generic metadata only names the definition
                    // with `object` arguments.
                    if field_il2cpp_type.type_enum == 0x15 {
                        if let Some(text) = generic_names.get(&field_il2cpp_type.datapoint) {
                            maps.field_generic_text
                                .entry((tdi, name.clone()))
                                .or_insert_with(|| text.clone());
                        }
                    }
                    // Insert under the ORIGINAL tdi (the child), so a query
                    // for the child's tdi + this offset finds the inherited field.
                    // Do not overwrite a field already found on the child itself
                    // (child fields take priority over inherited ones).
                    let key = (tdi, raw_offset as i64);
                    maps.field_layouts
                        .entry(key)
                        .or_insert((name.clone(), declaring_name.clone()));
                    // Declared access width, where it is unambiguous.
                    let fw = match field_il2cpp_type.type_enum {
                        0x02 | 0x04 | 0x05 => Some(1u8),
                        0x03 | 0x06 | 0x07 => Some(2),
                        0x08 | 0x09 | 0x0C => Some(4),
                        0x0A | 0x0B | 0x0D | 0x18 | 0x19 => Some(8),
                        0x0E | 0x12 | 0x14 | 0x15 | 0x1C => Some(8),
                        0x11 => {
                            // Enum → its underlying integer width. A struct is
                            // refused: a 4-byte store at its offset is a
                            // sub-field, not the struct field itself.
                            let etdi = field_il2cpp_type.klass_index() as usize;
                            metadata
                                .type_defs
                                .get(etdi)
                                .filter(|td| td.is_enum())
                                .and_then(|td| {
                                    metadata
                                        .field_defs
                                        .get(td.field_start as usize)
                                        .and_then(|f| il2cpp.types.get(f.type_index as usize))
                                        .and_then(|t| match t.type_enum {
                                            0x04 | 0x05 => Some(1),
                                            0x06 | 0x07 => Some(2),
                                            0x08 | 0x09 => Some(4),
                                            0x0A | 0x0B => Some(8),
                                            _ => Some(4),
                                        })
                                })
                        }
                        _ => None,
                    };
                    if let Some(w) = fw {
                        maps.field_widths.entry(key).or_insert(w);
                    }
                    // Record the field's type tdi for pointer chain resolution.
                    // This enables resolving `*(nint*)(base.field + offset)` to
                    // the named field at `offset` within the field's type.
                    let type_enum = field_il2cpp_type.type_enum;
                    let field_type_tdi = match type_enum {
                        // Class or ValueType: datapoint is the klass_index (tdi).
                        0x12 | 0x11 => Some(field_il2cpp_type.klass_index() as i32),
                        // GenericInst (List<T>, Dictionary<K,V>, etc.):
                        // datapoint is a pointer to Il2CppGenericClass; resolve
                        // the generic type definition's tdi from it.
                        0x15 => {
                            let gc_ptr = field_il2cpp_type.generic_class();
                            if gc_ptr != 0 {
                                il2cpp.read_generic_class(gc_ptr).ok().and_then(|gc| {
                                    if il2cpp.version >= 27.0 {
                                        // v27+: type_ptr → Il2CppType whose klass_index is the tdi
                                        let type_ptr = gc.type_ptr;
                                        if type_ptr != 0 {
                                            il2cpp.get_il2cpp_type(type_ptr).map(|t| t.klass_index() as i32)
                                        } else {
                                            None
                                        }
                                    } else {
                                        // pre-v27: type_definition_index is the tdi directly
                                        let idx = gc.type_definition_index;
                                        if idx == u32::MAX as u64 || idx == u64::MAX || idx == 0 {
                                            None
                                        } else {
                                            Some(idx as i32)
                                        }
                                    }
                                })
                            } else {
                                None
                            }
                        }
                        // String: well-known type System.String
                        0x0E => string_tdi,
                        // Object: well-known type System.Object
                        0x1C => object_tdi,
                        _ => None,
                    };
                    if let Some(ftdi) = field_type_tdi {
                        maps.field_types.entry(key).or_insert(ftdi);
                    }
                    // Direct fields of the type itself (the chain loop would
                    // duplicate inherited ones under the child's key). The
                    // embedded-layout descent walks its own parent chain and
                    // needs each type's OWN fields with their declared types.
                    if tdi_walk == tdi {
                        maps.tdi_fields.entry(tdi).or_default().push((
                            key.1,
                            name.clone(),
                            field_type_tdi.unwrap_or(-1),
                        ));
                    }
                }
            }
        }
        for (ti, td) in metadata.type_defs.iter().enumerate() {
            if td.is_value_type() {
                maps.value_type_tdis.insert(ti as i32);
            }
        }
        // Build the reverse map: (tdi, field_name) → field_type_tdi
        // This is used by pointer chain resolution to look up field types by name.
        for ((tdi, offset), field_type_tdi) in &maps.field_types {
            if let Some((field_name, _decl_type)) = maps.field_layouts.get(&(*tdi, *offset)) {
                maps.field_type_by_name
                    .entry((*tdi, field_name.clone()))
                    .or_insert(*field_type_tdi);
            }
        }
    }

    /// Pre-compute static field layouts from the metadata-usage scan.
    ///
    /// For each entry in `field_index_by_addr` (which maps metadata-usage slot
    /// addresses to global field indices), resolve the field name and declaring
    /// type, then insert `address → (field_name, type_name)` into
    /// `maps.static_field_layouts`. The field annotation pass uses this to
    /// resolve `RawDeref(Int(base), offset, _)` to `StaticField(type, field)`
    /// when `base + offset` matches a known metadata-usage slot address.
    pub fn populate_static_field_layouts(
        maps: &mut resolve::ResolveMaps,
        metadata: &Metadata,
        field_index_by_addr: &HashMap<u64, u32>,
        field_owner_tdi: &HashMap<u32, i32>,
    ) {
        crate::dbg_eprintln!("[debug] populate_static_field_layouts: field_index_by_addr has {} entries", field_index_by_addr.len());
        // A slot address can carry several field candidates. `or_insert` keeps
        // the FIRST, and a HashMap iteration has no stable first: two runs of
        // the same binary rendered the same store as `StreamReader.Null` and
        // `X501.countryName`. Sort by (address, field index) so the winner is
        // the lowest field index every run.
        let mut addrs: Vec<(&u64, &u32)> = field_index_by_addr.iter().collect();
        addrs.sort_by_key(|(addr, fi)| (**addr, **fi));
        for (&addr, &fi) in addrs {
            let Some(&tdi) = field_owner_tdi.get(&fi) else {
                continue;
            };
            let Some(td) = metadata.type_defs.get(tdi as usize) else {
                continue;
            };
            let Some(fdef) = metadata.field_defs.get(fi as usize) else {
                continue;
            };
            let type_name = metadata
                .get_string_from_index(td.name_index)
                .unwrap_or_default();
            let field_name = metadata
                .get_string_from_index(fdef.name_index)
                .unwrap_or_else(|_| format!("field_{fi}"));
            maps.static_field_layouts
                .entry(addr)
                .or_insert((field_name, type_name));
        }
        crate::dbg_eprintln!("[debug] populate_static_field_layouts: static_field_layouts has {} entries", maps.static_field_layouts.len());
        // Debug: print first 10 static field addresses
        for (i, (addr, (field_name, type_name))) in maps.static_field_layouts.iter().enumerate().take(10) {
            crate::dbg_eprintln!("[debug]   static_field[{}]: 0x{:x} = {}.{}", i, addr, type_name, field_name);
        }
    }

    /// Pre-compute static field offsets within the static_fields structure.
    ///
    /// For each type, iterate its static fields and use `get_field_offset_from_index`
    /// to determine each field's byte offset within the static_fields block.
    /// Store `(type_name, byte_offset) → (field_name, type_name)` in `maps.static_field_offsets`.
    ///
    /// This enables resolving `RawDeref(Field(base, "static_fields"), offset)` to
    /// `StaticField(type_name, field_name)` when the type is known from the
    /// klass pointer or a variable.
    pub fn populate_static_field_offsets(
        maps: &mut resolve::ResolveMaps,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) {
        // Parallelize across type_defs: each produces a local Vec of entries,
        // then we merge once at the end. Avoids contending on the shared HashMap.
        let entries: Vec<(
            ((String, i64), (String, String)),
            Option<((String, String), i32)>,
        )> = metadata.type_defs.par_iter().enumerate().flat_map(|(tdi, td)| {
            let tdi = tdi as i32;
            let field_start = td.field_start as usize;
            let field_count = td.field_count as usize;
            let type_name = metadata
                .get_string_from_index(td.name_index)
                .unwrap_or_default();
            let ns = metadata
                .get_string_from_index(td.namespace_index)
                .unwrap_or_default();
            let full_type_name = if ns.is_empty() {
                type_name.clone()
            } else {
                format!("{}.{}", ns, type_name)
            };
            let mut local = Vec::new();
            use crate::il2cpp::enums::field_attributes;
            for fi in 0..field_count {
                let global_fi = field_start + fi;
                let Some(field_def) = metadata.field_defs.get(global_fi) else {
                    continue;
                };
                let Some(field_il2cpp_type) = il2cpp.types.get(field_def.type_index as usize) else {
                    continue;
                };
                let is_static = (field_il2cpp_type.attrs & 0x0010) != 0;
                // CONSTANTS and FieldRVA data are NOT in the static block:
                // giving them a storage offset collides with the real static
                // field at that offset. Camera's const `kMaxAperture` stole
                // the delegate field's slot and named `onPreCull` after it.
                let is_literal =
                    (field_il2cpp_type.attrs & field_attributes::LITERAL) != 0;
                let has_field_rva =
                    (field_il2cpp_type.attrs & field_attributes::HAS_FIELD_RVA) != 0;
                if !is_static || is_literal || has_field_rva {
                    continue;
                }
                let raw_offset = il2cpp.get_field_offset_from_index(
                    tdi as usize,
                    fi,
                    global_fi,
                    td.is_value_type(),
                    is_static,
                );
                if raw_offset < 0 {
                    continue;
                }
                let name = metadata
                    .get_string_from_index(field_def.name_index)
                    .unwrap_or_else(|_| format!("field_{global_fi}"));
                // The static field's DECLARED class type, so the field
                // annotator can continue a pointer chain through
                // `Type.StaticField +off`. Class/ValueType carry the tdi in
                // `klass_index` (same convention as instance field types).
                let field_type_tdi = match field_il2cpp_type.type_enum {
                    0x11 | 0x12 => Some(field_il2cpp_type.klass_index() as i32),
                    // GenericInst (List<T>, Dictionary<K,V>, …): same
                    // resolution as instance field types — the generic type
                    // DEFINITION's tdi is enough to recognise the layout.
                    0x15 => {
                        let gc_ptr = field_il2cpp_type.generic_class();
                        if gc_ptr != 0 {
                            il2cpp.read_generic_class(gc_ptr).ok().and_then(|gc| {
                                if il2cpp.version >= 27.0 {
                                    let type_ptr = gc.type_ptr;
                                    if type_ptr != 0 {
                                        il2cpp.get_il2cpp_type(type_ptr).map(|t| t.klass_index() as i32)
                                    } else {
                                        None
                                    }
                                } else {
                                    let idx = gc.type_definition_index;
                                    if idx == u32::MAX as u64 || idx == u64::MAX || idx == 0 {
                                        None
                                    } else {
                                        Some(idx as i32)
                                    }
                                }
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                let key = (full_type_name.clone(), raw_offset as i64);
                let ty = field_type_tdi
                    .map(|ftdi| ((full_type_name.clone(), name.clone()), ftdi));
                local.push(((key, (name, full_type_name.clone())), ty));
            }
            local
        }).collect();
        // The flat_map above runs in parallel; its Vec order (and therefore
        // which duplicate key wins `or_insert`) changed run to run. Sort by
        // (type, offset) and then by the candidate name so the winner is fixed.
        let mut entries = entries;
        entries.sort_by(|a, b| {
            (a.0).0.cmp(&(b.0).0).then_with(|| (a.0).1.cmp(&(b.0).1))
        });
        for ((key, val), ty) in entries {
            maps.static_field_offsets.entry(key).or_insert(val);
            if let Some((ty_key, ty_tdi)) = ty {
                maps.static_field_types.entry(ty_key).or_insert(ty_tdi);
            }
        }
    }

    /// Pre-compute enum member names: for every enum type in the metadata, build
    /// a map from constant value to member name. Stored in `maps.enum_members`
    /// as `tdi → (i64 → member_name)`.
    ///
    /// Enum types are identified by `Il2CppTypeDefinition::is_enum()`. Their
    /// fields (excluding the compiler-generated `value__` instance field) are
    /// static literal fields whose default values are stored in the metadata's
    /// `fieldAndParameterDefaultValueData` section.
    pub fn populate_enum_members(
        maps: &mut resolve::ResolveMaps,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) {
        use crate::il2cpp::field_layout::read_metadata_bytes;

        for (tdi, td) in metadata.type_defs.iter().enumerate() {
            let tdi = tdi as i32;
            if !td.is_enum() {
                continue;
            }
            let field_start = td.field_start as usize;
            let field_count = td.field_count as usize;
            let mut members: HashMap<i64, String> = HashMap::new();

            // Determine the underlying type size from element_type_index.
            // Default is Int32 (4 bytes) when element_type_index is -1.
            let underlying_size: usize = if td.element_type_index >= 0 {
                // Look up the element type to determine its size.
                if let Some(elem_type) = il2cpp.types.get(td.element_type_index as usize) {
                    use crate::il2cpp::enums::Il2CppTypeEnum;
                    match Il2CppTypeEnum::from_u8(elem_type.type_enum) {
                        Some(Il2CppTypeEnum::Boolean) | Some(Il2CppTypeEnum::I1) | Some(Il2CppTypeEnum::U1) => 1,
                        Some(Il2CppTypeEnum::Char) | Some(Il2CppTypeEnum::I2) | Some(Il2CppTypeEnum::U2) => 2,
                        Some(Il2CppTypeEnum::I4) | Some(Il2CppTypeEnum::U4) => 4,
                        Some(Il2CppTypeEnum::I8) | Some(Il2CppTypeEnum::U8) => 8,
                        Some(Il2CppTypeEnum::I) | Some(Il2CppTypeEnum::U) => if il2cpp.is_32bit { 4 } else { 8 },
                        _ => 4, // default to i32
                    }
                } else {
                    4
                }
            } else {
                4 // default underlying type is Int32
            };

            for fi in 0..field_count {
                let global_fi = field_start + fi;
                let Some(field_def) = metadata.field_defs.get(global_fi) else {
                    continue;
                };
                // Get the field name.
                let field_name = metadata
                    .get_string_from_index(field_def.name_index)
                    .unwrap_or_default();
                // Skip the compiler-generated `value__` instance field.
                if field_name == "value__" {
                    continue;
                }
                // Look up the default value for this field.
                let Some(default_val) = metadata.get_field_default_value(global_fi as i32) else {
                    continue;
                };
                let data_index = default_val.data_index;
                if data_index < 0 {
                    continue;
                }
                // Read the constant value bytes from metadata.
                let offset = metadata.get_default_value_offset(data_index);
                let Some(bytes) = read_metadata_bytes(metadata, offset, underlying_size) else {
                    continue;
                };
                // Interpret the bytes as a signed integer.
                let value: i64 = match underlying_size {
                    1 => {
                        if bytes.len() >= 1 {
                            bytes[0] as i8 as i64
                        } else {
                            continue;
                        }
                    }
                    2 => {
                        if bytes.len() >= 2 {
                            i16::from_le_bytes([bytes[0], bytes[1]]) as i64
                        } else {
                            continue;
                        }
                    }
                    4 => {
                        if bytes.len() >= 4 {
                            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                        } else {
                            continue;
                        }
                    }
                    8 => {
                        if bytes.len() >= 8 {
                            i64::from_le_bytes([
                                bytes[0], bytes[1], bytes[2], bytes[3],
                                bytes[4], bytes[5], bytes[6], bytes[7],
                            ])
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };
                members.insert(value, field_name);
            }
            if !members.is_empty() {
                maps.enum_members.insert(tdi, members);
            }
        }
    }

    /// Pre-compute attribute constructor names: for every custom attribute
    /// generator in the binary, map its RVA to the short attribute type name.
    ///
    /// This enables the attribute recovery pass to recognise calls to attribute
    /// constructors in static initialisers and emit `[AttributeName(args)]`
    /// comments. Only attributes from metadata version >= 21 are processed
    /// (custom attributes were introduced in that version).
    ///
    /// For metadata < 29, the generator function pointers are in
    /// `executor.custom_attribute_generators`. For >= 29, the attribute data
    /// is stored differently and we use `attribute_type_ranges` + `attribute_types`.
    pub fn populate_attribute_ctor_names(
        maps: &mut resolve::ResolveMaps,
        executor: &mut Il2CppExecutor,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) {
        if il2cpp.version < 21.0 {
            return;
        }

        if il2cpp.version < 29.0 {
            // Pre-v29: custom_attribute_generators is a flat array of function
            // pointers, indexed by attribute index. attribute_type_ranges gives
            // the type indices for each attribute index.
            // Collect (rva, type_idx) pairs first to avoid borrowing executor
            // immutably (for the generators vec) and mutably (for get_type_name)
            // in the same loop.
            let mut work: Vec<(u64, usize)> = Vec::new();
            for (attr_idx, &gen_ptr) in executor.custom_attribute_generators.iter().enumerate() {
                if gen_ptr == 0 {
                    continue;
                }
                let rva = il2cpp.get_rva(gen_ptr);
                if rva == 0 {
                    continue;
                }
                if attr_idx >= metadata.attribute_type_ranges.len() {
                    continue;
                }
                let attr_range = &metadata.attribute_type_ranges[attr_idx];
                if attr_range.count == 0 {
                    continue;
                }
                let type_index = attr_range.start as usize;
                if type_index >= metadata.attribute_types.len() {
                    continue;
                }
                work.push((rva, metadata.attribute_types[type_index] as usize));
            }
            for (rva, type_idx) in work {
                if let Some(attr_type) = il2cpp.types.get(type_idx) {
                    let full_name = executor.get_type_name(attr_type, metadata, il2cpp, false, false);
                    // Extract the short name (last component after '.').
                    let short_name = full_name.rsplit('.').next().unwrap_or(&full_name);
                    // Strip "Attribute" suffix for display if present.
                    let display_name = short_name.strip_suffix("Attribute").unwrap_or(short_name);
                    maps.attribute_ctor_names.insert(rva, display_name.to_string());
                }
            }
        }
        // For v29+, attribute data is in attribute_data_ranges with a different
        // encoding. The attribute type info is still in attribute_type_ranges
        // but the generator pointers are not directly available. We skip this
        // for now as the binary encoding is more complex.
    }

    /// Populate static field layouts from Il2CppClass structures in the binary (Unity 2022+ / IL2CPP v29+).
    ///
    /// In Unity 2022+, static fields are NOT resolved through metadata usages (kind 4).
    /// Instead, the static field data is stored in .bss/.data sections, and the
    /// code references these addresses directly with patterns like `*(nint*)(0x2f6a000 + 792)`.
    ///
    /// The Il2CppClass structure (see `assets/UnityHeaders/31-2022.3.33-.h`) contains a
    /// `void* static_fields;` member that points to the static field storage for that class.
    /// This function finds Il2CppClass instances in the binary and reads that pointer.
    ///
    /// Two discovery strategies are used:
    ///
    /// 1. **Metadata-usage slot dereference** (memory dumps): In a memory dump, the metadata
    ///    usage slots (kind 1 = TypeInfo) contain runtime pointers to Il2CppClass structures.
    ///    We dereference these to find the class and read `static_fields`.
    ///
    /// 2. **Data-section structure scan** (all binaries): We scan the binary's data sections
    ///    for Il2CppClass structures, identified by matching (field_count, method_count,
    ///    static_fields_size) at known offsets within the structure against the metadata's
    ///    TypeDef information. This works for both memory dumps (where `static_fields` is
    ///    populated) and static binaries (where `static_fields` may be null or pre-initialized).
    pub fn populate_static_fields_from_data_sections(
        maps: &mut resolve::ResolveMaps,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
    ) {
        use crate::il2cpp::enums::field_attributes;
        use crate::il2cpp::field_layout::decode_field_offset;

        crate::dbg_eprintln!("[debug] populate_static_fields_from_data_sections: starting");

        let ptr_size = if il2cpp.is_32bit { 4u64 } else { 8u64 };

        // Compute the offset of `static_fields` within Il2CppClass.
        // Layout from `assets/UnityHeaders/31-2022.3.33-.h`:
        //
        // Il2CppClass {
        //   image (ptr), gc_desc (ptr), name (ptr), namespaze (ptr),
        //   byval_arg (Il2CppType), this_arg (Il2CppType),
        //   element_class (ptr), castClass (ptr), declaringType (ptr), parent (ptr),
        //   generic_class (ptr), typeMetadataHandle (ptr), interopData (ptr), klass (ptr),
        //   fields (ptr), events (ptr), properties (ptr), methods (ptr),
        //   nestedTypes (ptr), implementedInterfaces (ptr),
        //   interfaceOffsets (ptr), static_fields (ptr), ...
        // }
        //
        // Il2CppType on 64-bit: data(8) + attrs:16 + type:8 + num_mods:5 + byref:1 + pinned:1 = 4 bytes
        //   packed into one unsigned int, total 12 bytes, padded to 16 for 8-byte alignment.
        // Il2CppType on 32-bit: data(4) + 4 = 8 bytes, 4-byte aligned.
        let static_fields_offset: u64 = if !il2cpp.is_32bit {
            // 64-bit: 4 pointers (32) + 2 * Il2CppType (32) + 10 pointers (80) + 2 pointers (16) + 1 pointer (8) = 200
            200
        } else {
            // 32-bit: 4 pointers (16) + 2 * Il2CppType (16) + 10 pointers (40) + 2 pointers (8) + 1 pointer (4) = 108
            108
        };

        // Offsets of identifying fields within Il2CppClass (for structure matching):
        // From the struct definition:
        //   ... static_fields (ptr), rgctx_data (ptr), typeHierarchy (ptr),
        //   unity_user_data (ptr), initializationExceptionGCHandle (u32),
        //   cctor_started (u32), cctor_finished_or_no_cctor (u32),
        //   [padding to pointer alignment], cctor_thread (ptr-sized),
        //   genericContainerHandle (ptr), instance_size (u32), actualSize (u32),
        //   element_size (u32), native_size (i32), static_fields_size (u32),
        //   thread_static_fields_size (u32), thread_static_fields_offset (i32),
        //   flags (u32), token (u32), method_count (u16), property_count (u16),
        //   field_count (u16), event_count (u16), ...
        //
        // 64-bit: static_fields at 200, then +8+8+8+4+4+4+4(pad)+8+8+4+4+4+4+4+4+4+4+4+4 = 240
        //   method_count at 240, field_count at 244, instance_size at 216, static_fields_size at 228
        // 32-bit: static_fields at 108, then +4+4+4+4+4+4+4+4+4+4+4+4+4+4+4+4+4+4 = 132
        //   method_count at 132, field_count at 136, instance_size at 120, static_fields_size at 128
        let (field_count_offset, method_count_offset, instance_size_offset, static_fields_size_offset): (u64, u64, u64, u64) = if !il2cpp.is_32bit {
            (244, 240, 216, 228)
        } else {
            (136, 132, 120, 128)
        };

        // Collect all TypeDefs with static fields
        let mut classes_with_static: Vec<(i32, u32)> = Vec::new();
        for (tdi, _td) in metadata.type_defs.iter().enumerate() {
            let tdi = tdi as i32;
            if let Some(sizes) = il2cpp.type_definition_sizes.get(tdi as usize) {
                if sizes.static_fields_size > 0 {
                    classes_with_static.push((tdi, sizes.static_fields_size));
                }
            }
        }

        crate::dbg_eprintln!("[debug] found {} classes with static fields", classes_with_static.len());

        if classes_with_static.is_empty() {
            return;
        }

        // Build a signature map for ALL TypeDefs (field_count, method_count) → Vec<tdi>,
        // so we can identify Il2CppClass structures by matching these fields against metadata.
        let mut all_sig_to_tdis: HashMap<(u16, u16), Vec<i32>> = HashMap::new();
        for (tdi, td) in metadata.type_defs.iter().enumerate() {
            let key = (td.field_count, td.method_count);
            all_sig_to_tdis.entry(key).or_default().push(tdi as i32);
        }

        // Map: class address (VA) → tdi
        let mut class_addr_to_tdi: HashMap<u64, i32> = HashMap::new();

        // --- Strategy 1: Metadata-usage slot dereference (memory dumps) ---
        // In a memory dump, the metadata usage slots (kind 1 = TypeInfo) contain runtime
        // pointers to Il2CppClass structures. The `klass_to_tdi` map keys are the RVAs of
        // these slots. We read the pointer at each slot and validate it.
        if !maps.klass_to_tdi.is_empty() {
            let mut deref_count = 0;
            for (&slot_rva, &tdi) in &maps.klass_to_tdi {
                // Convert RVA to file offset
                let slot_va = if il2cpp.image_base > 0 {
                    slot_rva.wrapping_add(il2cpp.image_base)
                } else {
                    slot_rva
                };
                let slot_file_off = match il2cpp.map_vatr(slot_va) {
                    Ok(o) => o,
                    Err(_) => continue,
                };

                // Read the pointer at the metadata usage slot
                let class_ptr = match il2cpp.stream.peek_ptr_at(slot_file_off) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                // Skip if it looks like an encoded metadata usage value rather than a pointer.
                // Encoded values have the form (kind << 29) | (index << 1) | 1, which is < u32::MAX.
                if class_ptr == 0 || class_ptr >= u32::MAX as u64 {
                    continue;
                }

                // The class_ptr should point to an Il2CppClass structure.
                // Validate by reading static_fields offset and checking it's plausible.
                let class_va = class_ptr;
                let sf_va = class_va.wrapping_add(static_fields_offset);
                let sf_file_off = match il2cpp.map_vatr(sf_va) {
                    Ok(o) => o,
                    Err(_) => continue,
                };

                // Read the static_fields pointer
                let _sf_ptr = match il2cpp.stream.peek_ptr_at(sf_file_off) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                class_addr_to_tdi.insert(class_va, tdi);
                deref_count += 1;
            }
            crate::dbg_eprintln!("[debug] strategy 1 (metadata-usage deref): found {} Il2CppClass pointers", deref_count);
        }

        // --- Strategy 2: Data-section structure scan ---
        // Scan data sections for Il2CppClass structures identified by their
        // (field_count, method_count) at known offsets. For each match, read the
        // static_fields pointer.
        {
            let mut scan_matches = 0usize;
            let mut scan_sf_nonzero = 0usize;

            for section in &il2cpp.data_sections {
                let sec_end = section.offset_end.min(il2cpp.stream.len() as u64);
                // The Il2CppClass structure is large (hundreds of bytes). We need at least
                // enough room for the fields we read (up to ~250 bytes on 64-bit).
                let min_struct_size = if !il2cpp.is_32bit { 260u64 } else { 140u64 };
                if sec_end.saturating_sub(section.offset) < min_struct_size {
                    continue;
                }
                let scan_end = sec_end.saturating_sub(min_struct_size);

                let mut pos = section.offset;
                while pos < scan_end {
                    // Read field_count and method_count at their known offsets (u16 each)
                    let fc_raw = match il2cpp.stream.peek_bytes_at(pos + field_count_offset, 2) {
                        Ok(b) => u16::from_le_bytes([b[0], b[1]]),
                        Err(_) => { pos += ptr_size; continue; },
                    };
                    let mc_raw = match il2cpp.stream.peek_bytes_at(pos + method_count_offset, 2) {
                        Ok(b) => u16::from_le_bytes([b[0], b[1]]),
                        Err(_) => { pos += ptr_size; continue; },
                    };

                    // Quick reject: both zero is almost certainly not a valid class
                    if fc_raw == 0 && mc_raw == 0 {
                        pos += ptr_size;
                        continue;
                    }

                    // Check if (field_count, method_count) matches any known TypeDef
                    let sig_key = (fc_raw, mc_raw);
                    let matching_tdis = match all_sig_to_tdis.get(&sig_key) {
                        Some(v) => v,
                        None => { pos += ptr_size; continue; },
                    };

                    // Additional validation: read instance_size and static_fields_size
                    // and check they're plausible (non-negative, within reasonable range).
                    let inst_size = match il2cpp.stream.peek_u32_at(pos + instance_size_offset) {
                        Ok(v) => v,
                        Err(_) => { pos += ptr_size; continue; },
                    };
                    let sf_size_in_struct = match il2cpp.stream.peek_u32_at(pos + static_fields_size_offset) {
                        Ok(v) => v,
                        Err(_) => { pos += ptr_size; continue; },
                    };

                    // instance_size should be > 0 for any real class (at minimum the object header)
                    if inst_size == 0 {
                        pos += ptr_size;
                        continue;
                    }

                    // Compute the virtual address of this candidate Il2CppClass
                    let class_addr = pos - section.offset + section.address;

                    // Find the best matching TypeDef
                    let mut best_tdi: Option<i32> = None;

                    // First, try to match with static_fields_size for disambiguation
                    for &tdi in matching_tdis {
                        if let Some(sizes) = il2cpp.type_definition_sizes.get(tdi as usize) {
                            if sizes.static_fields_size == sf_size_in_struct && sizes.instance_size == inst_size {
                                best_tdi = Some(tdi);
                                break;
                            }
                        }
                    }

                    // If no exact match, try matching just instance_size
                    if best_tdi.is_none() {
                        for &tdi in matching_tdis {
                            if let Some(sizes) = il2cpp.type_definition_sizes.get(tdi as usize) {
                                if sizes.instance_size == inst_size {
                                    best_tdi = Some(tdi);
                                    break;
                                }
                            }
                        }
                    }

                    // If still no match and there's only one candidate, use it
                    if best_tdi.is_none() && matching_tdis.len() == 1 {
                        best_tdi = Some(matching_tdis[0]);
                    }

                    if let Some(tdi) = best_tdi {
                        // Don't overwrite a mapping found by strategy 1
                        if !class_addr_to_tdi.contains_key(&class_addr) {
                            class_addr_to_tdi.insert(class_addr, tdi);
                            scan_matches += 1;

                            // Check if static_fields is non-zero
                            let sf_va = class_addr.wrapping_add(static_fields_offset);
                            if let Ok(sf_file_off) = il2cpp.map_vatr(sf_va) {
                                if let Ok(sf_ptr) = il2cpp.stream.peek_ptr_at(sf_file_off) {
                                    if sf_ptr != 0 {
                                        scan_sf_nonzero += 1;
                                    }
                                }
                            }
                        }
                    }

                    pos += ptr_size;
                }
            }

            crate::dbg_eprintln!(
                "[debug] strategy 2 (structure scan): {} matches, {} with non-null static_fields",
                scan_matches, scan_sf_nonzero
            );
        }

        crate::dbg_eprintln!("[debug] total Il2CppClass mappings: {}", class_addr_to_tdi.len());

        // --- Map static fields using the discovered Il2CppClass addresses ---
        let mut fields_mapped = 0;
        let mut classes_with_sf = 0usize;

        for (&class_addr, &tdi) in &class_addr_to_tdi {
            // Read the static_fields pointer from the Il2CppClass structure
            let sf_va = class_addr.wrapping_add(static_fields_offset);
            let sf_file_off = match il2cpp.map_vatr(sf_va) {
                Ok(o) => o,
                Err(_) => continue,
            };
            let sf_ptr = match il2cpp.stream.peek_ptr_at(sf_file_off) {
                Ok(p) => p,
                Err(_) => continue,
            };

            if sf_ptr == 0 {
                continue; // No static fields allocated (class not yet initialized)
            }

            classes_with_sf += 1;
            let td = &metadata.type_defs[tdi as usize];
            let type_name = metadata.get_string_from_index(td.name_index).unwrap_or_default();

            let field_start = td.field_start as usize;
            let field_count = td.field_count as usize;

            let computed_offsets = crate::il2cpp::field_layout::compute_static_field_offsets(td, metadata, il2cpp);

            for fi in 0..field_count {
                let global_fi = field_start + fi;
                let Some(field_def) = metadata.field_defs.get(global_fi) else {
                    continue;
                };
                let Some(field_type) = il2cpp.types.get(field_def.type_index as usize) else {
                    continue;
                };

                let is_static = (field_type.attrs & field_attributes::STATIC) != 0;
                let is_literal = (field_type.attrs & field_attributes::LITERAL) != 0;
                let has_field_rva = (field_type.attrs & field_attributes::HAS_FIELD_RVA) != 0;

                if !is_static || is_literal || has_field_rva {
                    continue;
                }

                let raw_offset = il2cpp.get_raw_field_offset(tdi as usize, fi, global_fi);
                let decoded = decode_field_offset(raw_offset);

                let effective = if decoded.effective <= 0 {
                    computed_offsets.as_ref()
                        .and_then(|v| v.get(fi).copied())
                        .unwrap_or(0) as i32
                } else {
                    decoded.effective
                };

                let abs_addr = sf_ptr.wrapping_add(effective as u64);

                let field_name = metadata
                    .get_string_from_index(field_def.name_index)
                    .unwrap_or_else(|_| format!("field_{global_fi}"));

                maps.static_field_layouts
                    .entry(abs_addr)
                    .or_insert((field_name, type_name.to_string()));

                fields_mapped += 1;
            }
        }

        crate::dbg_eprintln!(
            "[debug] populate_static_fields_from_data_sections: {} classes with non-null static_fields, {} fields mapped",
            classes_with_sf, fields_mapped
        );
        crate::dbg_eprintln!("[debug] populate_static_fields_from_data_sections: static_field_layouts now has {} entries", maps.static_field_layouts.len());

        // --- NEW: GOT relocation-based resolution ---
        // Parse ELF relocations to find what each GOT entry should point to at runtime.
        // The relocation addends are the actual static field storage addresses.
        let got_reloc_fields = Self::populate_static_fields_from_got_relocations(maps, metadata, il2cpp, &class_addr_to_tdi);

        // --- Fallback: if no static_fields pointers were found (static binary), ---
        // use sequential page assignment as a best-effort heuristic.
        if classes_with_sf == 0 && got_reloc_fields == 0 && !classes_with_static.is_empty() {
            crate::dbg_eprintln!("[debug] no non-null static_fields found, falling back to sequential page assignment");
            Self::populate_static_fields_sequential(maps, metadata, il2cpp, &classes_with_static);
        }
    }

    /// Parse ELF relocations to resolve static fields via GOT entries.
    ///
    /// In Unity 2022+, code accesses static fields through GOT entries like
    /// `*(nint*)(0x2f6a000 + 792)`. The GOT entry at 0x2f6a000 has a relocation
    /// that tells us what it should point to at runtime (the addend). The addend
    /// is the static field storage address. We map (GOT_address + field_offset)
    /// to (class_name, field_name).
    fn populate_static_fields_from_got_relocations(
        maps: &mut resolve::ResolveMaps,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        class_addr_to_tdi: &HashMap<u64, i32>,
    ) -> usize {
        // Use the pre-parsed GOT relocations from the ELF parser
        if il2cpp.got_relative_relocations.is_empty() {
            return 0;
        }

        crate::dbg_eprintln!("[debug] GOT relocations: {} entries", il2cpp.got_relative_relocations.len());

        // Collect unique addends (target addresses) from GOT relocations
        let mut unique_addends: Vec<u64> = il2cpp.got_relative_relocations.values().copied().collect();
        unique_addends.sort_unstable();
        unique_addends.dedup();

        crate::dbg_eprintln!("[debug] Unique addends: {}", unique_addends.len());

        // Collect classes with multi-constraint fingerprints
        #[allow(dead_code)] // class_addr and type_signature_hash are part of the fingerprint schema for future matching
        struct ClassFingerprint {
            class_addr: u64,
            tdi: i32,
            static_fields_size: u32,
            static_field_count: u32,
            type_signature_hash: u64,
        }

        let mut class_fingerprints: Vec<ClassFingerprint> = Vec::new();
        if !il2cpp.is_32bit {
            let static_fields_size_offset: u64 = 228;
            for (&class_addr, &tdi) in class_addr_to_tdi {
                let sfs_file_off = match il2cpp.map_vatr(class_addr.wrapping_add(static_fields_size_offset)) {
                    Ok(off) => off,
                    Err(_) => continue,
                };
                let static_fields_size = il2cpp.stream.peek_u32_at(sfs_file_off).unwrap_or(0);
                if static_fields_size == 0 {
                    continue;
                }

                // Compute static field count and type signature hash
                let Some(td) = metadata.type_defs.get(tdi as usize) else {
                    continue;
                };
                let field_start = td.field_start as usize;
                let field_count = td.field_count as usize;
                let is_value_type = td.is_value_type();

                let mut static_field_count = 0u32;
                let mut type_sig_hash: u64 = 0;

                for fi in 0..field_count {
                    let global_fi = field_start + fi;
                    let Some(field_def) = metadata.field_defs.get(global_fi) else {
                        continue;
                    };
                    let Some(field_il2cpp_type) = il2cpp.types.get(field_def.type_index as usize) else {
                        continue;
                    };
                    let is_static = (field_il2cpp_type.attrs & 0x0010) != 0;
                    if !is_static {
                        continue;
                    }

                    static_field_count += 1;

                    // Get field offset and type for signature
                    let raw_offset = il2cpp.get_field_offset_from_index(
                        tdi as usize, fi, global_fi, is_value_type, is_static,
                    );
                    let type_idx = field_def.type_index as u64;

                    // Simple hash: combine offset and type_index
                    type_sig_hash = type_sig_hash.wrapping_add(
                        (raw_offset as u64).wrapping_mul(31).wrapping_add(type_idx)
                    );
                }

                if static_field_count > 0 {
                    class_fingerprints.push(ClassFingerprint {
                        class_addr,
                        tdi,
                        static_fields_size,
                        static_field_count,
                        type_signature_hash: type_sig_hash,
                    });
                }
            }
        }

        crate::dbg_eprintln!("[debug] Classes with static fields: {}", class_fingerprints.len());

        // Partition addends into blocks using gap threshold
        let gap_threshold = 0x1000u64;
        let mut blocks: Vec<(u64, u64)> = Vec::new();
        if !unique_addends.is_empty() {
            let mut block_start = unique_addends[0];
            let mut block_end = unique_addends[0];
            for &addr in &unique_addends[1..] {
                if addr - block_end > gap_threshold {
                    blocks.push((block_start, block_end - block_start + 8));
                    block_start = addr;
                }
                block_end = addr;
            }
            blocks.push((block_start, block_end - block_start + 8));
        }

        crate::dbg_eprintln!("[debug] Blocks partitioned: {}", blocks.len());

        // Compute block fingerprints (size + entry count)
        #[allow(dead_code)] // block_idx retained for future block-indexed lookups
        struct BlockFingerprint {
            block_idx: usize,
            block_start: u64,
            block_size: u64,
            entry_count: u32,
        }

        let mut block_fingerprints: Vec<BlockFingerprint> = Vec::new();
        for (block_idx, &(block_start, block_size)) in blocks.iter().enumerate() {
            let entry_count = unique_addends.iter()
                .filter(|&&addr| addr >= block_start && addr < block_start + block_size)
                .count() as u32;

            block_fingerprints.push(BlockFingerprint {
                block_idx,
                block_start,
                block_size,
                entry_count,
            });
        }

        // Sort by size for efficient matching
        block_fingerprints.sort_by_key(|b| b.block_size);

        // Multi-constraint matching with confidence scoring
        let mut got_to_tdi: HashMap<u64, i32> = HashMap::new();
        let mut used_classes = std::collections::HashSet::new();
        let mut high_confidence_matches = 0;
        let mut low_confidence_rejected = 0;

        for block_fp in &block_fingerprints {
            let mut best_match: Option<(usize, f64)> = None; // (class_idx, score)
            let mut best_score = 0.0f64;

            for (class_idx, class_fp) in class_fingerprints.iter().enumerate() {
                if used_classes.contains(&class_idx) {
                    continue;
                }

                // Constraint 1: Size match (weight: 0.6)
                if block_fp.block_size > class_fp.static_fields_size as u64 {
                    continue; // Block can't be larger than class
                }
                let size_diff = class_fp.static_fields_size as u64 - block_fp.block_size;
                // Lenient size scoring: allow up to 512-byte difference with reasonable scores
                let size_score = if size_diff == 0 {
                    1.0
                } else {
                    1.0 / (1.0 + size_diff as f64 / 512.0)
                };

                // Constraint 2: Field count match (weight: 0.3)
                let count_diff = (class_fp.static_field_count as i32 - block_fp.entry_count as i32).abs();
                // More lenient: up to 8 fields difference still scores 0.5
                let count_score = if count_diff == 0 {
                    1.0
                } else {
                    1.0 / (1.0 + count_diff as f64 / 8.0)
                };

                // Constraint 3: Type signature similarity (weight: 0.1)
                // We can't directly compare signatures, but we can penalize large mismatches
                let sig_score = 0.5; // Neutral score since we can't easily compare

                // Weighted total score
                let total_score = 0.6 * size_score + 0.3 * count_score + 0.1 * sig_score;

                if total_score > best_score {
                    best_score = total_score;
                    best_match = Some((class_idx, total_score));
                }
            }

            // Confidence threshold: only accept matches with score >= 0.50
            // Lowered from 0.60 to improve coverage. Size matching is reliable,
            // so we trust it even with moderate size differences.
            if let Some((class_idx, score)) = best_match {
                if score >= 0.50 {
                    let class_fp = &class_fingerprints[class_idx];
                    used_classes.insert(class_idx);
                    high_confidence_matches += 1;

                    // Map GOT slots to this class
                    for &addr in &unique_addends {
                        if addr >= block_fp.block_start && addr < block_fp.block_start + block_fp.block_size {
                            for (&got_addr, &addend) in &il2cpp.got_relative_relocations {
                                if addend == addr {
                                    got_to_tdi.insert(got_addr, class_fp.tdi);
                                }
                            }
                        }
                    }
                } else {
                    low_confidence_rejected += 1;
                }
            }
        }

        crate::dbg_eprintln!("[debug] Multi-constraint matching: {} high-confidence matches, {} low-confidence rejected",
            high_confidence_matches, low_confidence_rejected);
        crate::dbg_eprintln!("[debug] GOT→class mapping: {} entries", got_to_tdi.len());

        // Build the final field map: (GOT_address + field_offset) → (field_name, type_name)
        let mut fields_mapped = 0;
        for (&got_addr, &tdi) in &got_to_tdi {
            let Some(td) = metadata.type_defs.get(tdi as usize) else {
                continue;
            };
            let type_name = metadata.get_string_from_index(td.name_index).unwrap_or_default();

            let field_start = td.field_start as usize;
            let field_count = td.field_count as usize;

            let computed_offsets = crate::il2cpp::field_layout::compute_static_field_offsets(td, metadata, il2cpp);

            for fi in 0..field_count {
                let global_fi = field_start + fi;
                let Some(field_def) = metadata.field_defs.get(global_fi) else {
                    continue;
                };
                let Some(field_il2cpp_type) = il2cpp.types.get(field_def.type_index as usize) else {
                    continue;
                };
                let is_static = (field_il2cpp_type.attrs & 0x0010) != 0;
                if !is_static {
                    continue;
                }

                let raw_offset_val = il2cpp.get_raw_field_offset(tdi as usize, fi, global_fi);
                let decoded = crate::il2cpp::field_layout::decode_field_offset(raw_offset_val);
                let raw_offset = if decoded.effective <= 0 {
                    computed_offsets.as_ref()
                        .and_then(|v| v.get(fi).copied())
                        .unwrap_or(0)
                } else {
                    decoded.effective as u32
                };
                let got_field_addr = got_addr.wrapping_add(raw_offset as u64);

                let field_name = metadata
                    .get_string_from_index(field_def.name_index)
                    .unwrap_or_else(|_| format!("field_{global_fi}"));

                if fields_mapped < 10 {
                    crate::dbg_eprintln!("[debug] GOT: got_addr=0x{:x}, fi={}, raw_offset={}, got_field_addr=0x{:x}, field={}",
                        got_addr, fi, raw_offset, got_field_addr, field_name);
                }

                maps.static_field_layouts
                    .entry(got_field_addr)
                    .or_insert((field_name.clone(), type_name.to_string()));
                fields_mapped += 1;
            }
        }

        crate::dbg_eprintln!("[debug] GOT-based mapping: {} fields mapped", fields_mapped);
        crate::dbg_eprintln!("[debug] After GOT: static_field_layouts has {} entries", maps.static_field_layouts.len());
        fields_mapped
    }

    /// Fallback: assign static field base addresses sequentially, one page per class.
    /// This is a best-effort heuristic for static binaries where Il2CppClass.static_fields
    /// is null (not yet initialized at dump time).
    fn populate_static_fields_sequential(
        maps: &mut resolve::ResolveMaps,
        metadata: &Metadata,
        il2cpp: &Il2Cpp,
        classes_with_static: &[(i32, u32)],
    ) {
        use crate::il2cpp::enums::field_attributes;
        use crate::il2cpp::field_layout::decode_field_offset;

        let page_size = 0x1000u64;

        let total_static_size: u64 = classes_with_static.iter().map(|(_, s)| *s as u64).sum();

        // Find the data section that could contain static fields
        let mut target_section: Option<&crate::search::SearchSection> = None;
        for section in &il2cpp.data_sections {
            let memsz = section.address_end - section.address;
            let filesz = section.offset_end - section.offset;
            if memsz >= total_static_size {
                let is_bss = filesz < memsz / 2;
                let is_better = match target_section {
                    None => true,
                    Some(prev) => {
                        let prev_memsz = prev.address_end - prev.address;
                        let prev_filesz = prev.offset_end - prev.offset;
                        let prev_is_bss = prev_filesz < prev_memsz / 2;
                        if prev_is_bss && !is_bss {
                            true
                        } else if !prev_is_bss && is_bss {
                            false
                        } else {
                            section.address > prev.address
                        }
                    }
                };
                if is_better {
                    target_section = Some(section);
                }
            }
        }

        let Some(section) = target_section else {
            crate::dbg_eprintln!("[debug] sequential fallback: no suitable data section found");
            return;
        };

        crate::dbg_eprintln!("[debug] sequential fallback: using data section {:#x} - {:#x}", section.address, section.address_end);

        let raw_start = section.address_end.saturating_sub(total_static_size);
        let first_page = raw_start & !(page_size - 1);
        let first_page = first_page.max((section.address + page_size - 1) & !(page_size - 1));

        let mut current_base = first_page;
        let mut fields_mapped = 0;

        for &(tdi, size) in classes_with_static {
            if current_base + size as u64 > section.address_end {
                break;
            }

            let td = &metadata.type_defs[tdi as usize];
            let type_name = metadata.get_string_from_index(td.name_index).unwrap_or_default();
            let field_start = td.field_start as usize;
            let field_count = td.field_count as usize;

            let computed_offsets = crate::il2cpp::field_layout::compute_static_field_offsets(td, metadata, il2cpp);

            for fi in 0..field_count {
                let global_fi = field_start + fi;
                let Some(field_def) = metadata.field_defs.get(global_fi) else { continue; };
                let Some(field_type) = il2cpp.types.get(field_def.type_index as usize) else { continue; };

                let is_static = (field_type.attrs & field_attributes::STATIC) != 0;
                let is_literal = (field_type.attrs & field_attributes::LITERAL) != 0;
                let has_field_rva = (field_type.attrs & field_attributes::HAS_FIELD_RVA) != 0;

                if !is_static || is_literal || has_field_rva {
                    continue;
                }

                let raw_offset = il2cpp.get_raw_field_offset(tdi as usize, fi, global_fi);
                let decoded = decode_field_offset(raw_offset);
                let effective = if decoded.effective <= 0 {
                    computed_offsets.as_ref()
                        .and_then(|v| v.get(fi).copied())
                        .unwrap_or(0) as i32
                } else {
                    decoded.effective
                };

                let abs_addr = current_base.wrapping_add(effective as u64);
                let field_name = metadata
                    .get_string_from_index(field_def.name_index)
                    .unwrap_or_else(|_| format!("field_{global_fi}"));

                maps.static_field_layouts
                    .entry(abs_addr)
                    .or_insert((field_name, type_name.to_string()));
                fields_mapped += 1;
            }

            current_base += page_size;
        }

        crate::dbg_eprintln!("[debug] sequential fallback: mapped {} static fields", fields_mapped);
    }


    /// Does an image/assembly name pass the configured include filters?
    pub fn assembly_allowed(&self, image_name: &str) -> bool {
        crate::output::pseudo_c::assembly_matches(image_name, &self.include_assemblies)
    }
}

// ---------------------------------------------------------------------------
// Helper-detection machinery (see RecoveryContext::detect_helpers)
// ---------------------------------------------------------------------------

/// Abstract value of a register at a call site, from a straight-line mini
/// simulation over the raw lifted listing.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum ArgVal {
    #[default]
    Unknown,
    /// known integer (may be a string-literal index)
    Const(i64),
    /// address inside a data section (klass pointers, rodata, vtables)
    DataRef(i64),
    /// derived from sp (address-of-local for box's value argument)
    SpRef,
}

/// Per-call-target statistics collected over sampled methods.
#[derive(Clone, Copy, Default, Debug)]
struct HelperCounts {
    total: u32,
    /// sites where x0 held a plausible string-literal index
    string_arg: u32,
    /// sites where x0 was a klass-ish data address
    klass_arg: u32,
    /// sites where x1 held any tracked value (box's value pointer)
    x1_arg: u32,
}

fn eval_arg(op: &Operand, consts: &HashMap<String, ArgVal>, data_ranges: &[(u64, u64)]) -> ArgVal {
    let is_data = |a: u64| a >= 0x10000 && data_ranges.iter().any(|&(lo, hi)| a >= lo && a < hi);
    match op {
        Operand::Immediate(v) => {
            if is_data(*v as u64) {
                ArgVal::DataRef(*v)
            } else {
                ArgVal::Const(*v)
            }
        }
        Operand::Register(n) => consts.get(n).copied().unwrap_or(ArgVal::Unknown),
        Operand::Memory { base, addend, .. } => match base.as_ref() {
            Operand::Immediate(b) => ArgVal::DataRef((*b as i64).wrapping_add(*addend)),
            Operand::Register(bn) => match consts.get(bn).copied().unwrap_or(ArgVal::Unknown) {
                ArgVal::Const(p) => ArgVal::DataRef(p.wrapping_add(*addend)),
                ArgVal::DataRef(p) => ArgVal::DataRef(p.wrapping_add(*addend)),
                other => other,
            },
            _ => ArgVal::Unknown,
        },
        _ => ArgVal::Unknown,
    }
}

fn fold_add(a: Option<ArgVal>, b: Option<ArgVal>) -> ArgVal {
    match (a, b) {
        (Some(ArgVal::Const(x)), Some(ArgVal::Const(y))) => ArgVal::Const(x.wrapping_add(y)),
        (Some(ArgVal::Const(x)), Some(ArgVal::DataRef(y)))
        | (Some(ArgVal::DataRef(y)), Some(ArgVal::Const(x))) => ArgVal::DataRef(x.wrapping_add(y)),
        (Some(ArgVal::DataRef(x)), Some(ArgVal::DataRef(y))) => ArgVal::DataRef(x.wrapping_add(y)),
        (Some(ArgVal::SpRef), _) | (_, Some(ArgVal::SpRef)) => ArgVal::SpRef,
        _ => ArgVal::Unknown,
    }
}

/// Which x86-64 calling convention this binary uses, from its container.
///
/// PE ⇒ MSVC (`rcx, rdx, r8, r9`), everything else x86 ⇒ SysV
/// (`rdi, rsi, rdx, rcx, r8, r9`). The container is the only reliable signal:
/// the two are indistinguishable instruction-for-instruction, and guessing
/// means seeding the wrong parameter slots. See `RecoveryContext::x86_abi` for
/// the measured cost of getting this wrong.
fn x86_abi_of(il2cpp: &Il2Cpp) -> lift_x86::X86Abi {
    if il2cpp.is_pe {
        lift_x86::X86Abi::Msvc
    } else {
        lift_x86::X86Abi::SysV
    }
}

/// Lift a (possibly non-managed) body for helper analysis, per architecture.
///
/// Takes the `Il2Cpp` handle rather than a pre-resolved ABI so every caller —
/// all of which already hold it — gets the same convention as the main
/// pipeline, and the argument-register names agree between the two passes.
fn lift_helper_body(
    il2cpp: &Il2Cpp,
    rva: u64,
    bytes: &[u8],
    max_insns: usize,
    is_arm64: bool,
    is_x64: bool,
) -> Option<Vec<ir::Insn>> {
    if is_arm64 {
        lift::lift_arm64(bytes, rva, max_insns, None).map(|l| l.insns)
    } else if is_x64 {
        lift_x86::lift_x86_64(bytes, rva, max_insns, x86_abi_of(il2cpp)).map(|l| l.insns)
    } else {
        None
    }
}

/// Follow single-branch thunk bodies (`B other`) up to 8 hops; stops at any
/// managed method or real (multi-insn) body. Mirrors Cpp2IL's
/// `FindFunctionThisIsAThunkOf` chaining. The result is the "root" that all
/// codegen variants of one helper also thunk to.
fn chase_thunk_root(
    il2cpp: &Il2Cpp,
    managed: &HashMap<u64, MethodRef>,
    start: u64,
    is_arm64: bool,
    is_x64: bool,
) -> u64 {
    let mut cur = start;
    for _ in 0..8 {
        if managed.contains_key(&cur) {
            return cur;
        }
        let Some(bytes) = il2cpp.read_bytes_at_rva(cur, 64) else {
            return cur;
        };
        let Some(insns) = lift_helper_body(il2cpp, cur, &bytes, 8, is_arm64, is_x64) else {
            return cur;
        };
        // Follow the LEADING Jump unconditionally. Requiring insns.len()==1
        // missed real thunks sitting next to jump tables: the lifter decodes
        // past the leading B into the adjacent table entries, yielding 19
        // insns for a 1-insn thunk (0x4a58b98 → 19). A body that STARTS with
        // an unconditional B is a tail-call regardless of what follows.
        if let Some(ir::Op::Jump(t)) = insns.first().map(|i| &i.op) {
            let next = *t as u64;
            if next != cur {
                cur = next;
                continue;
            }
        }
        return cur;
    }
    cur
}

/// True when the body at `rva` opens with an unconditional branch — the
/// codegen-helper shape (`il2cpp_codegen_object_new` etc.). Real functions
/// never start with `B`.
fn is_thunk_body(
    il2cpp: &Il2Cpp,
    managed: &HashMap<u64, MethodRef>,
    rva: u64,
    is_arm64: bool,
    is_x64: bool,
) -> bool {
    if managed.contains_key(&rva) {
        return false;
    }
    let Some(bytes) = il2cpp.read_bytes_at_rva(rva, 64) else {
        return false;
    };
    let Some(insns) = lift_helper_body(il2cpp, rva, &bytes, 8, is_arm64, is_x64) else {
        return false;
    };
    insns
        .first()
        .is_some_and(|i| matches!(i.op, ir::Op::Jump(_)))
}

/// Recover an x64 trailing tail call to a NON-managed helper as `Call + Return`.
///
/// The x64 shape `OutSine`@0x14D6540 exhibits: setup floating-point arithmetic
/// followed by a trailing unconditional `jmp` past the method body to a helper
/// the metadata never names (libm `sin`/`pow`, memcpy, …). The managed tail-call
/// pass only fires for known method starts, and the tail-merge splicer below is
/// ARM64-`B`-only, so on x64 these stayed bare `Jump`s: DCE deleted the setup
/// feeding them as dead, and the structurer emitted an empty body (`// body not
/// recoverable`) for a method that really is `return fn_<addr>(args)`.
///
/// The rewrite mirrors `passes::recover_tail_calls` (same `[target, dest,
/// x0 … x7, fp…]` layout, same `TailCall` ledger so the A/A drift allowance
/// applies identically) with four x64-specific points:
///
/// * the FP dump is `vfN` — the lifter's XMM names. Dumping ARM's `vsN` would
///   leave the `vfN` setup exactly as dead as before and fix nothing.
/// * the callee stays an immediate (`fn_<addr>`): the target is honest but
///   unnamed, so no value is invented.
/// * the call carries the method's own return type (`dest_type`), so the
///   float-result binders treat it like any lifted float call: without it the
///   call's `rv` result never reaches the FP slot and a float method renders
///   `fn_…(…); return <argument>` instead of `return fn_…(…)`.
/// * a `void` method returns nothing: the trailing instruction is a bare
///   `Return`, so the call renders as a statement and the body ends in
///   `return;` rather than a fabricated value.
///
/// Refusals (all leave the listing untouched, i.e. inert on well-formed IR):
/// fewer than two insns (length 1 is the thunk path above), last substantive
/// insn not a `Jump`, target inside the body (ordinary control flow), null
/// target, managed target (the managed path names it properly), unreadable
/// target bytes, or a target that branches back into this body (a dispatch
/// stub, not a call — checked by the caller, which can lift).
pub(crate) fn recover_x64_unmanaged_tail(
    insns: &mut Vec<Insn>,
    addresses: &mut Vec<u64>,
    body_lo: u64,
    body_hi: u64,
    is_managed: &dyn Fn(u64) -> bool,
    target_readable: &dyn Fn(u64) -> bool,
    target_branches_back: &dyn Fn(u64) -> bool,
    ret_type: &ManagedType,
) -> bool {
    if insns.len() < 2 {
        return false;
    }
    let Some(jidx) = insns.iter().rposition(|i| !matches!(i.op, Op::Nop)) else {
        return false;
    };
    let target = match insns[jidx].op {
        Op::Jump(t) => t as u64,
        _ => return false,
    };
    if target == 0 || (target >= body_lo && target < body_hi) {
        return false;
    }
    if is_managed(target) || !target_readable(target) || target_branches_back(target) {
        return false;
    }
    let a = insns[jidx].address;
    let mut ops = vec![Operand::imm(target as i64), Operand::reg(ir::RET_REG)];
    for r in ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"] {
        ops.push(Operand::reg(r));
    }
    for r in ["vf0", "vf1", "vf2", "vf3", "vf4", "vf5", "vf6", "vf7"] {
        ops.push(Operand::reg(r));
    }
    let mut call = Insn::new(Op::Call, ops, a);
    call.dest_type = Some(ret_type.clone());
    insns[jidx] = call;
    let ret = if matches!(ret_type, ManagedType::Primitive(PrimKind::Void)) {
        Insn::new(Op::Return, vec![], a)
    } else {
        Insn::new(Op::Return, vec![Operand::reg(ir::RET_REG)], a)
    };
    insns.insert(jidx + 1, ret);
    addresses.insert(jidx + 1, a);
    true
}

/// Does the body at `from` (or anything it calls, ≤depth hops) reach `want`?
/// Port of Cpp2IL `ThrowHelperRecovery::ReachesCall`. `budget` bounds the
/// total number of bodies inspected per query (the DFS can otherwise blow up
/// exponentially on call-dense code).
fn reaches_call(
    il2cpp: &Il2Cpp,
    from: u64,
    want: u64,
    is_arm64: bool,
    is_x64: bool,
    depth: u32,
    visited: &mut HashSet<u64>,
    budget: &mut u32,
) -> bool {
    if from == want {
        return true;
    }
    if depth == 0 || *budget == 0 {
        return false;
    }
    *budget -= 1;
    if !visited.insert(from) {
        return false;
    }
    let Some(bytes) = il2cpp.read_bytes_at_rva(from, 256) else {
        return false;
    };
    let Some(insns) = lift_helper_body(il2cpp, from, &bytes, 96, is_arm64, is_x64) else {
        return false;
    };
    for insn in &insns {
        if matches!(insn.op, ir::Op::Call | ir::Op::CallVoid) {
            if let Some(Operand::Immediate(t)) = insn.operands.first() {
                if reaches_call(
                    il2cpp,
                    *t as u64,
                    want,
                    is_arm64,
                    is_x64,
                    depth - 1,
                    visited,
                    budget,
                ) {
                    return true;
                }
            }
        }
    }
    false
}

/// Unresolved targets by descending call frequency (bounded probe list).
fn freq_targets_desc(
    counts: &HashMap<u64, HelperCounts>,
    excluded: &HashSet<u64>,
    cap: usize,
) -> Vec<u64> {
    let mut v: Vec<u64> = counts
        .iter()
        .filter(|(t, _)| !excluded.contains(*t))
        .map(|(&t, _)| t)
        .collect();
    v.sort_unstable_by(|a, b| counts[b].total.cmp(&counts[a].total));
    v.truncate(cap);
    v
}

/// The first function CALLED (or tail-jumped) by the body at `start`,
/// following unconditional tail chains (thunk → thunk → call). Returns the
/// VM/runtime function a codegen wrapper delegates to. This is the aliasing
/// key: every codegen variant of one runtime helper delegates to the SAME
/// VM function, so variant ≡ export when first_callee(variant) ==
/// first_callee(export) (or the variant directly reaches the export).
/// Collect the first N distinct BL call targets in a body (after following
/// leading-B tail-call chains). Used by bind-symbol chain classification:
/// throw helpers call the cxa stubs via a multi-BL body, so the FIRST callee
/// alone misses them.
fn callees_first_n(
    il2cpp: &Il2Cpp,
    managed: &HashMap<u64, MethodRef>,
    start: u64,
    n: usize,
    is_arm64: bool,
    is_x64: bool,
) -> Vec<u64> {
    let mut cur = start;
    let mut followed = 0;
    while followed < 8 {
        if managed.contains_key(&cur) {
            return Vec::new();
        }
        let Some(bytes) = il2cpp.read_bytes_at_rva(cur, 128) else {
            return Vec::new();
        };
        let Some(insns) = lift_helper_body(il2cpp, cur, &bytes, 32, is_arm64, is_x64) else {
            return Vec::new();
        };
        if let ir::Op::Jump(t) = insns[0].op {
            let next = t as u64;
            if next != cur {
                cur = next;
                followed += 1;
                continue;
            }
        }
        let mut out: Vec<u64> = Vec::new();
        for i in insns {
            match i.op {
                ir::Op::Call | ir::Op::CallVoid => {
                    if let Some(Operand::Immediate(t)) = i.operands.first() {
                        if !out.contains(&(*t as u64)) {
                            out.push(*t as u64);
                            if out.len() >= n {
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        return out;
    }
    Vec::new()
}

fn first_callee(
    il2cpp: &Il2Cpp,
    managed: &HashMap<u64, MethodRef>,
    start: u64,
    is_arm64: bool,
    is_x64: bool,
) -> Option<u64> {
    let mut cur = start;
    for _ in 0..8 {
        if managed.contains_key(&cur) {
            return None; // reached a managed method — not a helper
        }
        let Some(bytes) = il2cpp.read_bytes_at_rva(cur, 128) else {
            return None;
        };
        let Some(insns) = lift_helper_body(il2cpp, cur, &bytes, 32, is_arm64, is_x64) else {
            return None;
        };
        // tail-jump chains: a body OPENING with B is a tail-call regardless of
        // how much junk the lifter decoded after it (adjacent jump-table bytes
        // decode as extra "insns"). Same fix as chase_thunk_root.
        if let ir::Op::Jump(t) = insns[0].op {
            let next = t as u64;
            if next != cur {
                cur = next;
                continue;
            }
        }
        // first Call/CallVoid in body order
        for i in insns {
            match i.op {
                ir::Op::Call | ir::Op::CallVoid => {
                    if let Some(Operand::Immediate(t)) = i.operands.first() {
                        return Some(*t as u64);
                    }
                }
                _ => {}
            }
        }
        return None;
    }
    None
}

/// string_new without symbols: ≥60% of ≥5 call sites pass x0 holding a
/// plausible literal index (unchanged heuristic; mirrors decompiler.rs).
fn select_string_new(counts: &HashMap<u64, HelperCounts>) -> Vec<u64> {
    let mut candidates: Vec<(u64, u32, u32)> = counts
        .iter()
        .map(|(&rva, c)| (rva, c.string_arg, c.total))
        .filter(|(_, hits, total)| {
            *hits >= 5 && *total >= 5 && (*hits as f32 / *total as f32) >= 0.6
        })
        .collect();
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.truncate(2);
    candidates.into_iter().map(|(rva, _, _)| rva).collect()
}

/// object_new without symbols: dominant klass-arg target. Thunk-shaped
/// bodies are preferred (codegen_object_new is a tail-branch thunk; real
/// functions like il2cpp_runtime_class_init are not).
fn select_object_new_fallback(
    counts: &HashMap<u64, HelperCounts>,
    excluded: &HashSet<u64>,
    is_thunk: &dyn Fn(u64) -> bool,
) -> Vec<u64> {
    let mut cands: Vec<(u64, u32)> = counts
        .iter()
        .filter(|(t, c)| {
            !excluded.contains(*t) && c.total >= 6 && (c.klass_arg as f32 / c.total as f32) >= 0.7
        })
        .map(|(&t, c)| (t, c.total))
        .collect();
    cands.sort_by(|a, b| {
        is_thunk(b.0)
            .cmp(&is_thunk(a.0))
            .then_with(|| b.1.cmp(&a.1))
    });
    let thunk_top = cands.first().map(|(t, _)| is_thunk(*t)).unwrap_or(false);
    let keep = if thunk_top { 2 } else { 1 };
    cands.into_iter().take(keep).map(|(t, _)| t).collect()
}

/// box without symbols: klass in x0 AND a defined value pointer in x1,
/// high-confidence ratio over ≥8 sites, excluding other helper families.
fn select_box_fallback(
    counts: &HashMap<u64, HelperCounts>,
    excluded: &HashSet<u64>,
    is_thunk: &dyn Fn(u64) -> bool,
) -> Vec<u64> {
    let mut cands: Vec<(u64, u32)> = counts
        .iter()
        .filter(|(t, c)| {
            !excluded.contains(*t)
                && c.total >= 8
                && (c.klass_arg as f32 / c.total as f32) >= 0.8
                && (c.x1_arg as f32 / c.total as f32) >= 0.5
        })
        .map(|(&t, c)| (t, c.total))
        .collect();
    cands.sort_by(|a, b| {
        is_thunk(b.0)
            .cmp(&is_thunk(a.0))
            .then_with(|| b.1.cmp(&a.1))
    });
    cands.into_iter().take(2).map(|(t, _)| t).collect()
}

fn fmt_rvas(rvas: &[u64]) -> String {
    match rvas.len() {
        0 => "none".to_string(),
        _ => rvas
            .iter()
            .map(|r| format!("0x{r:X}"))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

impl RecoveryContext {
    /// Reverse index: native code target → the metadata methods whose
    /// compiled bodies branch directly to it.
    ///
    /// IL2CPP statically resolves extern engine (InternalCall) methods, so a
    /// game method's call to e.g. `CharacterController::Move` compiles as a
    /// DIRECT `bl`/`b` to the engine implementation — an address with no
    /// metadata entry. The engine method's own generated metadata wrapper
    /// (it does have a code pointer) is a tiny body that ends in the same
    /// direct branch to that implementation. Scanning every metadata body's
    /// first few direct branch targets therefore maps implementation
    /// addresses back to their managed names and signatures. A call to a
    /// target with exactly ONE wrapper can be named from metadata; multiple
    /// wrappers ⇒ shared dispatcher ⇒ leave `fn_…` (no wrong guesses).
    fn build_native_wrappers(&mut self, _metadata: &Metadata, il2cpp: &Il2Cpp) {
        let is_arm64 = self.arch_is_arm64;
        if !is_arm64 && !self.is_x64 {
            return;
        }
        if self.maps.methods_by_rva.is_empty() {
            return;
        }
        // Sort method-start RVAs so `get_method_body_size` can bound each
        // scan cheaply; bodies are tiny (wrappers are a few instructions).
        let mut starts: Vec<u64> = self.maps.methods_by_rva.keys().copied().collect();
        starts.sort_unstable();
        let mut wrappers: HashMap<u64, Vec<MethodRef>> = HashMap::new();
        for (rva, m) in self.maps.methods_by_rva.iter() {
            let size = il2cpp.get_method_body_size(*rva, &starts);
            if size == 0 || size > 512 {
                // size 0 = abstract/extern-without-body; huge = skip (we only
                // need the tiny wrappers)
                if size > 512 {
                    continue;
                }
            }
            let n = if size == 0 {
                64
            } else {
                size.min(128) as usize
            };
            let Some(bytes) = il2cpp.read_bytes_at_rva(*rva, n) else {
                continue;
            };
            if is_arm64 {
                // walk words; record direct B/BL; stop at first RET
                let mut i = 0usize;
                while i + 4 <= bytes.len() {
                    let word =
                        u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
                    let top = word >> 26;
                    if top == 0b000101 || top == 0b100101 {
                        // B / BL: pc-relative imm26
                        let imm = (word & 0x03FF_FFFF) as i64;
                        let sext = (imm << 6) >> 6;
                        let tgt = (*rva as i64 + (i as i64) + (sext << 2)) as u64;
                        // ignore branches to metadata methods (normal calls)
                        if !self.maps.methods_by_rva.contains_key(&tgt) {
                            wrappers.entry(tgt).or_default().push(m.clone());
                        }
                    } else if word == 0xD65F_03C0 {
                        break; // RET
                    }
                    i += 4;
                }
            } else {
                // x64: scan for E8 rel32 (call) / E9 rel32 (jmp)
                let mut i = 0usize;
                while i + 5 <= bytes.len() && i < 128 {
                    let b0 = bytes[i];
                    if b0 == 0xE8 || b0 == 0xE9 {
                        let rel = i32::from_le_bytes([
                            bytes[i + 1],
                            bytes[i + 2],
                            bytes[i + 3],
                            bytes[i + 4],
                        ]);
                        let tgt = (*rva as i64 + i as i64 + 5 + rel as i64) as u64;
                        if !self.maps.methods_by_rva.contains_key(&tgt) {
                            wrappers.entry(tgt).or_default().push(m.clone());
                        }
                    } else if b0 == 0xC3 {
                        break; // ret
                    }
                    i += 1;
                }
            }
        }
        let dbg = crate::recovery::env_flags::flag("ILR_ICALL");
        if dbg {
            let mut flat: Vec<(u64, usize)> = wrappers
                .iter()
                .map(|(&t, v)| {
                    let mut names: Vec<String> = v.iter().map(|m| m.name.clone()).collect();
                    names.sort();
                    names.dedup();
                    (t, names.len())
                })
                .collect();
            flat.sort_by(|a, b| b.1.cmp(&a.1));
            for (t, c) in flat.iter().take(20) {
                let names: Vec<String> = wrappers[t]
                    .iter()
                    .map(|m| format!("{}.{}", short_typename(&m.declaring_type_name), m.name))
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .take(6)
                    .collect();
                eprintln!("[icall] target {:#x} wrappers={} names={:?}", t, c, names);
            }
            eprintln!("[icall] total targets: {}", wrappers.len());
        }
        self.maps.native_wrappers = wrappers;
    }
}

fn short_typename(full: &str) -> &str {
    full.rsplit('.').next().unwrap_or(full)
}

/// Precompute base-class chains for every type (bounded walk).
fn build_parent_chains(metadata: &Metadata, il2cpp: &Il2Cpp) -> HashMap<i32, Vec<i32>> {
    use crate::il2cpp::enums::Il2CppTypeEnum;
    let n = metadata.type_defs.len();
    let mut chains: HashMap<i32, Vec<i32>> = HashMap::with_capacity(n);
    // resolve each type's parent once (parent_index → types entry → klass)
    let mut parents: Vec<Option<i32>> = vec![None; n];
    for tdi in 0..n {
        if let Some(pt) = il2cpp
            .types
            .get(metadata.type_defs[tdi].parent_index as usize)
        {
            let te = Il2CppTypeEnum::from_u8(pt.type_enum);
            if matches!(
                te,
                Some(Il2CppTypeEnum::Class) | Some(Il2CppTypeEnum::ValueType)
            ) {
                let k = pt.klass_index() as usize;
                if k < n {
                    parents[tdi] = Some(k as i32);
                }
            }
        }
    }
    for tdi in 0..n as i32 {
        let mut chain = vec![tdi];
        let mut cur = tdi;
        for _ in 0..16 {
            match parents[cur as usize] {
                Some(p) => {
                    chain.push(p);
                    cur = p;
                }
                None => break,
            }
        }
        chains.insert(tdi, chain);
    }
    chains
}
