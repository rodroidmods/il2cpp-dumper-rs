//! Differential semantic verifier for the recovery pipeline.
//!
//! The recovery engine can fail SILENTLY: a pass drops or reorders semantics
//! and the emitter still renders plausible-looking C#. This module is the
//! machine that makes those failures loud, method by method, with no manual
//! inspection:
//!
//! 1. **Decode-all ground truth** — every instruction of the method body is
//!    decoded linearly with the battle-tested crates (`yaxpeax-arm` /
//!    `iced-x86`), independent of the lifter's CFG. Nothing downstream can
//!    hide from this list.
//! 2. **Coverage oracle** — each decoded instruction must be accounted for by
//!    the final IR (a real op at that address, a load/store/arithmetic side
//!    effect, a control transfer, a known pseudo-insn) or be excused by the
//!    pipeline's *ledger* of legitimate rewrites (guard excision, DCE, ctor
//!    fusion, runtime-call dropping, branch folding). An uncovered
//!    instruction means semantics were silently dropped → reported with the
//!    address and the decoded text.
//! 3. **CFG invariants** — every block reaches an exit (catches unrecovered /
//!    impossible loops), no dangling successor edges, sane block bounds.
//! 4. **A/A pass diffing** — semantic fingerprint of the raw lift vs the
//!    final IR. Only growth is flagged (synthesized control flow); loss is
//!    policed by the coverage oracle through the ledger.
//!
//! Enabled by `ILR_VERIFY=1` (aggregates over all methods). `ILR_VERIFY_JSON=1`
//! additionally appends a per-method JSONL record to `verify_findings.jsonl`.
//!
//! Design note: the verifier never needs a second decompiler. The lifter's
//! decoders are already crate-grade; the bug zone is downstream of them, and
//! the oracle below is exactly what catches downstream loss mechanically.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use super::cfg::Cfg;
use super::ir::{ArithKind, CompareKind, Insn, Op, Operand};

// ---------------------------------------------------------------------------
// Aggregate stats (atomics: the dump pipeline runs under rayon)
// ---------------------------------------------------------------------------

pub struct VerifyStats {
    pub methods_checked: AtomicU64,
    pub methods_clean: AtomicU64,
    pub methods_flagged: AtomicU64,
    /// Methods the oracle PROVES disagree with their own native bytes
    /// ([`Verdict::WrongLogic`]). This — not `methods_flagged` — is the number
    /// of methods whose recovered logic is demonstrably not what the native
    /// code does. `methods_flagged` cannot answer that question: a single
    /// `uncovered` arithmetic instruction that a later pass legitimately folded
    /// flags a method, so on the v24.5 corpus it reports 12,585 of 12,615 and
    /// says nothing about correctness.
    pub methods_wrong_logic: AtomicU64,
    /// Methods with a corruption-SHAPED finding (a local read that is never
    /// defined, a branch with no real condition, an unresolved call, a raw
    /// register SSA never renamed, or control flow the optimizer synthesized)
    /// but no proven disagreement with the bytes.
    pub methods_suspect: AtomicU64,
    /// Methods whose ONLY findings are benign optimizer artefacts.
    pub methods_coverage_only: AtomicU64,
    pub uncovered_total: AtomicU64,
    /// Decoded instructions the census REFUSED to judge because they are not
    /// provably code — padding after the last `ret`, and the inline switch
    /// jump tables the compiler emits in the code section, both of which the
    /// linear sweep decodes into plausible-looking instructions. Reported so
    /// the exclusion is auditable: a spike here means the reachability walk
    /// lost the plot, not that the port got better.
    pub unreached_total: AtomicU64,
    pub cfg_violations_total: AtomicU64,
    pub aa_drift_total: AtomicU64,
    pub missing_conditions_total: AtomicU64,
    pub undefined_locals_total: AtomicU64,
    pub unresolved_calls_total: AtomicU64,
    /// Oracle #3: `Register(..)` operands still present in the FINAL listing.
    /// A raw register in the finished IR can only mean SSA never renamed it,
    /// which the renderer prints as `__unknown` (or a bogus field read).
    pub raw_registers_total: AtomicU64,
    pub decode_failures_total: AtomicU64,
    /// The ARM32 subset of `decode_failures_total` that the LIFTER still
    /// reconstructs from the raw word (the VFP coprocessor group and the
    /// signed-DSP multiply group). These are an ORACLE gap, not a
    /// reconstruction gap — the engine emitted real IR there — so the census
    /// reports them apart from the failures both sides dropped. See
    /// `lift_arm32::liftable_word`.
    pub decode_failures_lifted_total: AtomicU64,
    /// Methods whose FINAL listing selects a SIMD lane outside the register's
    /// own lanes, and the total number of such selections. See
    /// [`out_of_range_lane_selections`]: this is a VALUE-level invariant — a
    /// wrong lane index is a wrong value — and not a coverage statistic, so it
    /// is reported even when it is zero (the zero is the evidence the oracle
    /// ran).
    pub out_of_range_lane_methods: AtomicU64,
    pub out_of_range_lane_sites: AtomicU64,
}

pub static STATS: VerifyStats = VerifyStats {
    methods_checked: AtomicU64::new(0),
    methods_clean: AtomicU64::new(0),
    methods_flagged: AtomicU64::new(0),
    methods_wrong_logic: AtomicU64::new(0),
    methods_suspect: AtomicU64::new(0),
    methods_coverage_only: AtomicU64::new(0),
    uncovered_total: AtomicU64::new(0),
    unreached_total: AtomicU64::new(0),
    cfg_violations_total: AtomicU64::new(0),
    aa_drift_total: AtomicU64::new(0),
    missing_conditions_total: AtomicU64::new(0),
    undefined_locals_total: AtomicU64::new(0),
    unresolved_calls_total: AtomicU64::new(0),
    raw_registers_total: AtomicU64::new(0),
    decode_failures_total: AtomicU64::new(0),
    decode_failures_lifted_total: AtomicU64::new(0),
    out_of_range_lane_methods: AtomicU64::new(0),
    out_of_range_lane_sites: AtomicU64::new(0),
};

impl VerifyStats {
    /// Human-readable aggregate, printed at dump end when ILR_VERIFY=1.
    pub fn report() -> String {
        let checked = STATS.methods_checked.load(Ordering::Relaxed);
        let clean = STATS.methods_clean.load(Ordering::Relaxed);
        let flagged = STATS.methods_flagged.load(Ordering::Relaxed);
        let unc = STATS.uncovered_total.load(Ordering::Relaxed);
        let unreach = STATS.unreached_total.load(Ordering::Relaxed);
        let cfg = STATS.cfg_violations_total.load(Ordering::Relaxed);
        let aa = STATS.aa_drift_total.load(Ordering::Relaxed);
        let nocond = STATS.missing_conditions_total.load(Ordering::Relaxed);
        let undef = STATS.undefined_locals_total.load(Ordering::Relaxed);
        let unres = STATS.unresolved_calls_total.load(Ordering::Relaxed);
        let rawregs = STATS.raw_registers_total.load(Ordering::Relaxed);
        let dec = STATS.decode_failures_total.load(Ordering::Relaxed);
        let declift = STATS.decode_failures_lifted_total.load(Ordering::Relaxed);
        let wrong = STATS.methods_wrong_logic.load(Ordering::Relaxed);
        let suspect = STATS.methods_suspect.load(Ordering::Relaxed);
        let cov = STATS.methods_coverage_only.load(Ordering::Relaxed);
        let lane_m = STATS.out_of_range_lane_methods.load(Ordering::Relaxed);
        let lane_s = STATS.out_of_range_lane_sites.load(Ordering::Relaxed);
        let (abi_seen, abi_filled) = super::resolve::abi_recovery_counts();
        let mut s = format!(
            "[verify] methods: {checked} checked -> {clean} clean | {wrong} WRONG-LOGIC | {suspect} suspect | {cov} coverage-only ({flagged} flagged by the all-or-nothing rule) | findings: {unc} uncovered insns ({unreach} outside the reachable set, not judged), {cfg} cfg violations, {aa} A/A drifts, {nocond} missing conditions, {undef} undefined locals, {unres} unresolved calls, {rawregs} raw registers in final IR, {dec} decode failures ({declift} engine-lifted, oracle-only; the rest were dropped by the lifter too)"
        );
        let ledger = LEDGER_SUMMARY.load_summary();
        if !ledger.is_empty() {
            s.push_str(" | ledger: ");
            let parts: Vec<String> = ledger.iter().map(|(k, v)| format!("{k}={v}")).collect();
            s.push_str(&parts.join(", "));
        }
        // Value-level invariants, reported in EVERY dump (the counts are cheap
        // atomics) rather than only when they are non-zero: a zero is the
        // evidence that the oracle ran at all.
        s.push_str(&format!(
            "\n[verify] SIMD lanes: {lane_s} out-of-range lane selection(s) in {lane_m} method(s)"
        ));
        // `__abi` argument holes the recovery pass examined and filled. "ABI
        // recovery did nothing" has to be a number rather than an assumption —
        // a fill ratio near zero says the pass is inert on THIS corpus, which
        // no other counter would show.
        s.push_str(&format!(
            "\n[verify] abi holes: {abi_filled}/{abi_seen} filled"
        ));
        // Class-vtable vs interface-dispatch fallback attempts vs hits, kept as
        // two SEPARATE pairs so never-attempted (klass unknown or slot not
        // constant, so no `(tdi, slot)` key exists) is distinguishable from
        // attempted-missed (the table was consulted and held no entry). A bare
        // hit count conflates the two and cannot say whether the 50k-entry
        // interface table is unused because call sites never reach it (the iOS
        // `klass + idx*16 + 312` runtime-computed slots) or because it misses.
        // Always printed, like the ABI line above: `0/0` is the evidence the
        // counters ran, not a missing measurement. Inert on well-formed IR:
        // the counters are atomic increments in `resolve::lookup_*` that never
        // alter the returned `MethodRef` or the emitted C#.
        {
            let (v_attempted, v_hit) = super::resolve::vtable_lookup_counts();
            let (i_attempted, i_hit) = super::resolve::iface_lookup_counts();
            s.push_str(&format!(
                "\n[verify] vtable lookups: {v_hit}/{v_attempted} hit; iface lookups: {i_hit}/{i_attempted} hit"
            ));
        }
        // Delegate `invoke_impl` field loads (`InstanceField`/`Field` after
        // typing, `Memory` before it): attempts vs hits plus the unproven-
        // base refusals. A hit is a proven delegate base naming `Invoke`; a
        // refusal keeps the honest `fnptr_` (guessing `Invoke` for a
        // non-delegate would fabricate a callee). Always printed: `0/0` is
        // the evidence the counters ran, not a missing measurement. Inert:
        // atomic increments in `resolve::delegate_invoke_of_def` only.
        {
            let (d_attempted, d_hit) = super::resolve::delegate_lookup_counts();
            let d_refused = super::resolve::delegate_refused_unproven();
            s.push_str(&format!(
                "\n[verify] delegate lookups: {d_hit}/{d_attempted} hit; refused unproven={d_refused}"
            ));
        }
        // `Arith`-base dynamic slots (`klass + idx*16 + 312` through `v60 =
        // v59 + 312`): attempts vs hits plus the four refusal buckets and
        // the indexed-spelling refusals. A hit is a chain that folded to a
        // constant (`klass + const`) on the vtable grid with a known klass;
        // every dynamic/unknown-klass/bad-offset site refuses and keeps
        // `fnptr_` — a wrong slot names a wrong method, which is worse.
        // Also the two `klass_tdi_of` residue buckets (immediate klass with
        // no TypeDef, target with no SSA definition). Always printed, inert
        // (atomic increments only).
        {
            let (a_attempted, a_hit) = super::resolve::arith_slot_counts();
            let (a_dyn, a_nodef, a_unk, a_bad) = super::resolve::arith_slot_refusals();
            let idx = super::resolve::indexed_slot_refusals();
            let nodef = super::resolve::vc_no_def_count();
            let unkimm = super::resolve::vc_unknown_klass_imm_count();
            s.push_str(&format!(
                "\n[verify] arith slots: {a_hit}/{a_attempted} hit; refused dynamic={a_dyn} no-def={a_nodef} unknown-klass={a_unk} bad-offset={a_bad}; indexed refusals={idx}; vc no-def={nodef} unknown-klass-imm={unkimm}"
            ));
        }
        // Guarded field-address seeds (`locals::propagate_types`): applied
        // struct-address seeds vs the three refusal buckets. The guard's whole
        // safety story is that the reference/enum bucket dwarfs applied — an
        // applied count near zero says the seed is inert on THIS corpus, while
        // a refused-reference count of zero would say the guard never fires
        // (i.e. the measurement is broken, not clean). Counts are per fixpoint
        // visit (≤16 per site), not distinct sites. Always printed, like the
        // lines above.
        {
            let (applied, ref_enum, no_layout, unproven) = super::locals::field_addr_seed_counts();
            s.push_str(&format!(
                "\n[verify] field-addr seeds: {applied} applied; refused ref/enum={ref_enum} no-layout={no_layout} unproven={unproven}"
            ));
        }
        // Guard-chain folds (`csharp::fold_branch_over_join`): `if (c) { goto L; }
        // <body> L:` rewritten to `if (!c) { <body> }` with `L:` kept. Attempts
        // vs folds plus the two refusal buckets. A fold fires only when NO edge
        // from outside the region targets a label inside it (method-wide vs
        // inside-region `goto` counts); predicates are inverted, never deleted;
        // the join label is kept; switch arms are never merged (each arm folds
        // its own guard). A refused-outside count near zero with many candidates
        // would say the gate never fires (measurement broken); a folds count
        // near zero says the fold is inert on THIS corpus. Always printed,
        // inert (atomic increments only).
        {
            let (candidates, folds, refused_outside, refused_unclosed) =
                super::csharp::guard_chain_fold_counts();
            s.push_str(&format!(
                "\n[verify] guard-chain folds: {folds}/{candidates} folded; refused outside-edge={refused_outside} unclosed={refused_unclosed}"
            ));
        }
        // Switch-arm goto-to-case rewrites are measured in the structurer's
        // census. Report the three buckets alongside verification so zero
        // attempts cannot be mistaken for attempted-but-refused candidates.
        // These are read-only atomic counters and do not affect recovery.
        {
            let census = &super::csharp::STRUCT_CENSUS;
            let attempts = census.case_goto_attempts.load(Ordering::Relaxed);
            let hits = census.case_goto_hits.load(Ordering::Relaxed);
            let refused = census.case_goto_refused.load(Ordering::Relaxed);
            s.push_str(&format!(
                "\n[verify] switch-arm goto case: {hits}/{attempts} rewrites; refused non-case targets={refused}"
            ));
        }
        if let Some(iface) = super::resolve::interface_dispatch_summary() {
            s.push('\n');
            s.push_str(&iface);
        }
        // Top uncovered-mnemonic clusters: the root causes to fix first.
        // GLOBAL map (rayon splits methods across worker threads; a
        // thread-local histogram never merges into the reporting thread).
        let clusters = if let Ok(m) = TOP_CLUSTERS.lock() {
            let mut v: Vec<(String, u64)> = m.iter().map(|(k, n)| (k.clone(), *n)).collect();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            v
        } else {
            Vec::new()
        };
        if !clusters.is_empty() {
            // How many clusters to print. `ILR_CLUSTERS=<n>` raises it for
            // diagnosis (the tail of the distribution is where the next root
            // cause hides); the default keeps the summary readable.
            let show = std::env::var("ILR_CLUSTERS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(12);
            s.push_str("\n[verify] top uncovered clusters:");
            for (k, n) in clusters.iter().take(show) {
                s.push_str(&format!("\n[verify]   {n:>6}  {k}"));
            }
        }
        s
    }
}

static TOP_CLUSTERS: std::sync::LazyLock<std::sync::Mutex<HashMap<String, u64>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Cluster sink for non-verify stages (e.g. the SSA builder measuring how many
/// raw registers survived IT, before any later pass can introduce more).
pub fn bump_cluster_external(key: &str) {
    bump_cluster(key);
}

fn bump_cluster(key: &str) {
    if let Ok(mut m) = TOP_CLUSTERS.lock() {
        *m.entry(key.to_string()).or_insert(0) += 1;
    }
}

// Per-kind ledger census: atomics because notes fire on rayon workers but
// the summary prints on the driver thread.
//
// Indexed by `LedgerKind`, and keyed for the report through
// `LedgerKind::name()` — the names used to be written out twice (once in a
// match here, once in `name()`), so any new kind could be added to one and
// silently omitted from the other. One array + one name table cannot drift.
struct LedgerSummary {
    counts: [AtomicU64; LedgerKind::COUNT],
}

const ZERO: AtomicU64 = AtomicU64::new(0);

static LEDGER_SUMMARY: LedgerSummary = LedgerSummary {
    counts: [ZERO; LedgerKind::COUNT],
};

impl LedgerSummary {
    fn bump(&self, kind: LedgerKind) {
        self.counts[kind.index()].fetch_add(1, Ordering::Relaxed);
    }

    fn load_summary(&self) -> Vec<(&'static str, u64)> {
        LedgerKind::ALL
            .iter()
            .filter_map(|&k| {
                let v = self.counts[k.index()].load(Ordering::Relaxed);
                (v > 0).then_some((k.name(), v))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Per-thread excusal ledger
//
// Passes legitimately erase ops (guard excision, DCE, ctor fusion, runtime
// metadata-init dropping, constant-branch folding). Each site records the
// address BEFORE overwriting; the oracle then treats those addresses as
// excused instead of flagged. The ledger is per-thread: a method never
// migrates threads inside `recover_ir_deadline`, rayon workers each get
// their own cell, and `recover_ir_deadline` clears it on entry so a failed
// method cannot leak excuses into the next method on the same worker.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LedgerKind {
    /// Dead-code elimination removed an unused pure definition.
    Dce,
    /// Class-init / null-check guard excision removed a condition + feeders.
    GuardExcision,
    /// il2cpp_codegen_initialize_* bookkeeping call dropped.
    DroppedRuntimeCall,
    /// Constructor call fused into the preceding NewObj.
    CtorFuse,
    /// Constant-folded branch rewritten (always/never taken).
    FoldedBranch,
    /// Register pin copy (`mov x20, x0`) canonicalized to a self-move and
    /// erased by rewrite_pinned_copies (locals.rs) — identity semantics.
    PinCopyFold,
    /// Callee-saved register save/restore round-trip (prologue `stp d8,d9,[sp,…]`
    /// + matching epilogue `ldp d8,d9,[sp],…`) excised — pure ABI machinery,
    /// no source-level semantics.
    CalleeSaveRoundTrip,
    /// A conditional branch whose TAKEN path and FALL-THROUGH path both reach
    /// the same instruction having executed nothing in between. Both arms run
    /// the identical code, so the condition has no observable effect and the
    /// branch is not a branch. IL2CPP's null-check shape makes this common:
    /// `cbnz w8, +8` skips the `bl <raise_null>` at the next address, and once
    /// that runtime call is excised the skipped slot is empty — so
    /// `if (w8 != 0) goto L;` and the fall-through both land on L. Rendering
    /// that produced `if (cond) { goto L; } L:`, a shape no compiler emits and
    /// which reads as corruption. See `passes::drop_vacuous_cond_branches`.
    VacuousBranch,
    /// A native `bl` the renderer presented as a MEMBER ACCESS or an operator
    /// instead of as `callee(…)`, so the call is in the text but not as an
    /// `ident(` token.
    ///
    /// `csharp::MethodCtx::prettify_call` folds `obj.get_X()` into `obj.X`,
    /// `Type.get_X()` into `Type.X`, `obj.set_X(v)` into `obj.X = v` and
    /// `String.Concat(a, b)` into `a + b`. Every one of those is a real
    /// `bl` in the binary and a faithful presentation of it, and
    /// `concord` counts C# calls by scanning for `ident(`, so each fold was
    /// charged to the decompiler as a DROPPED call. Measured on
    /// `SwitchToRectTransform` @0xFA08: 14 native `bl`, 4 `ident(` in the
    /// text, and exactly 10 folds (`from.rect` ×4, `from.position`,
    /// `to.rect` ×4, `to.anchoredPosition`) — a 1:1 match. The note is keyed
    /// on the CALL INSTRUCTION's address, so the checker can subtract it
    /// from the native count exactly like `DroppedRuntimeCall`.
    PropertyAccessorCall,
    /// A TAIL TRANSFER rewritten into a `Call` + `Return`
    /// (`passes::recover_tail_calls`).
    ///
    /// `b <another method>` / `br xN` with the frame already torn down IS a
    /// call followed by a return; the IR makes the return explicit, which the
    /// transfer performed implicitly. So the A/A fingerprint legitimately sees
    /// ONE more control-transfer event than the raw lift:
    ///
    ///   * from a direct `b` (counted as a BRANCH) it becomes a call —
    ///     `calls` +1, the return replaces the branch;
    ///   * from a `br xN` (`Op::IndirectCall`, already counted as a CALL) it
    ///     becomes a call plus an explicit return — `branches` +1.
    ///
    /// It can also re-materialize a memory operand into the synthesized
    /// argument dump (the register dump is fresh, so copy propagation inlines a
    /// defining `Move`'s `Memory` source into the call's argument list).
    ///
    /// The fingerprint charges this as "unexpected synthesis" because it is a
    /// per-instruction-op count and cannot tell the two shapes apart; the note
    /// is keyed on the transfer's own address so the comparator can excuse
    /// exactly as many events as the pass created — and no more.
    TailCall,
}

impl LedgerKind {
    /// Every kind, in report order. Doubles as the index space of
    /// `LedgerSummary::counts`.
    const ALL: [LedgerKind; 10] = [
        LedgerKind::Dce,
        LedgerKind::GuardExcision,
        LedgerKind::DroppedRuntimeCall,
        LedgerKind::CtorFuse,
        LedgerKind::FoldedBranch,
        LedgerKind::PinCopyFold,
        LedgerKind::CalleeSaveRoundTrip,
        LedgerKind::VacuousBranch,
        LedgerKind::PropertyAccessorCall,
        LedgerKind::TailCall,
    ];
    const COUNT: usize = Self::ALL.len();

    fn index(self) -> usize {
        LedgerKind::ALL
            .iter()
            .position(|&k| k == self)
            .expect("LedgerKind::ALL must list every variant")
    }

    fn name(self) -> &'static str {
        match self {
            LedgerKind::Dce => "dce",
            LedgerKind::GuardExcision => "guard_excision",
            LedgerKind::DroppedRuntimeCall => "dropped_runtime_call",
            LedgerKind::CtorFuse => "ctor_fuse",
            LedgerKind::FoldedBranch => "folded_branch",
            LedgerKind::PinCopyFold => "pin_copy_fold",
            LedgerKind::CalleeSaveRoundTrip => "callee_save_roundtrip",
            LedgerKind::VacuousBranch => "vacuous_branch",
            LedgerKind::PropertyAccessorCall => "property_accessor_call",
            LedgerKind::TailCall => "tail_call",
        }
    }
}

thread_local! {
    static LEDGER: RefCell<HashSet<(u64, LedgerKind)>> = RefCell::new(HashSet::new());
}

/// Record that the instruction at `addr` is about to be legitimately erased.
pub fn ledger_note(addr: u64, kind: LedgerKind) {
    LEDGER.with(|l| {
        l.borrow_mut().insert((addr, kind));
    });
    LEDGER_SUMMARY.bump(kind);
}

/// Drop stale entries (called at the top of the per-method pipeline).
pub fn ledger_clear() {
    LEDGER.with(|l| l.borrow_mut().clear());
}

pub fn ledger_has(addr: u64) -> bool {
    LEDGER.with(|l| l.borrow().iter().any(|&(a, _)| a == addr))
}

/// Was `addr` ledgered under exactly this kind?
///
/// The checker needs the distinction: a `DroppedRuntimeCall` is an
/// `il2cpp_codegen_initialize_class` thunk the pipeline removes ON PURPOSE
/// (the C# source never had it), while a `Dce`/`GuardExcision` entry is a
/// structural rewrite. Counting the first as "a missing call" reported
/// deliberate plumbing removal as lost semantics — on 0x1fe2ca0 that was 112
/// of the 112 calls the concordance said were missing.
pub fn ledger_has_kind(addr: u64, kind: LedgerKind) -> bool {
    LEDGER.with(|l| l.borrow().contains(&(addr, kind)))
}

// ---------------------------------------------------------------------------
// Per-method verdict annotations (in-output surfacing)
// ---------------------------------------------------------------------------
//
// The census grades every method (Clean / WrongLogic / Suspect / CoverageOnly)
// but that grade lives only in the `ILR_VERIFY` log: the emitted `dump.cs`
// renders a flagged method exactly like a clean one, so a reader cannot tell a
// read of a never-written slot (`v10 = ...`) from a recovered value. This
// side-channel carries the grade from `verify_and_log` (inside the per-method
// pipeline, on a rayon worker) to the renderer (also on a worker) so each
// flagged body is stamped with its own grade. A global map rather than a
// thread-local: the renderer and the verifier run on the SAME worker for a
// given method, but a driver-thread consumer (the dump header) must be able to
// read it too, and only flagged methods are inserted — a few thousand entries
// on a 60k-method binary.

static ANNOTATIONS: std::sync::LazyLock<std::sync::Mutex<HashMap<u64, (Verdict, String)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

thread_local! {
    /// Did the pipeline reach the verification stage for the method currently
    /// being recovered on this worker? The renderer reads it to distinguish
    /// "graded clean" (no stamp) from "never graded" (pipeline exited early —
    /// deadline, IR cap — so the body must be stamped suspect, not silently
    /// passed off as proven). Thread-local because a worker handles one method
    /// at a time and `recover_ir_deadline` runs the pipeline on the calling
    /// thread.
    static GRADED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark that this thread's current method reached verification.
fn mark_graded() {
    GRADED.with(|c| c.set(true));
}

/// Mark a method as graded WITHOUT a differential-verifier report, because the
/// pipeline produced it by a construction that is trusted rather than checked.
/// The single case is the single-`B` forwarding thunk: `recover_ir_deadline`
/// synthesizes a delegating call from the one decoded branch, which is faithful
/// by construction, so the renderer must not stamp it "not verified".
pub fn mark_graded_trusted() {
    mark_graded();
}

/// Reset the marker before a method starts recovering.
pub fn clear_graded() {
    GRADED.with(|c| c.set(false));
}

/// Whether the current thread's last recovery reached verification.
pub fn graded_this_method() -> bool {
    GRADED.with(|c| c.get())
}

/// Record a method's grade for the renderer (see [`annotation_for`]).
pub fn annotate_method(addr: u64, verdict: Verdict, reason: String) {
    if let Ok(mut m) = ANNOTATIONS.lock() {
        m.insert(addr, (verdict, reason));
    }
}

/// The grade recorded for `addr`, if the oracle flagged it.
pub fn annotation_for(addr: u64) -> Option<(Verdict, String)> {
    ANNOTATIONS.lock().ok()?.get(&addr).cloned()
}

/// How many methods carry a grade (i.e. how many the oracle could not prove
/// clean). Reported in the dump banner so the reader knows the scale before
/// scrolling.
pub fn annotation_count() -> usize {
    ANNOTATIONS.lock().map(|m| m.len()).unwrap_or(0)
}

/// Human name for a verdict, as it appears in the output annotation.
pub fn verdict_name(v: Verdict) -> &'static str {
    match v {
        Verdict::Clean => "clean",
        Verdict::WrongLogic => "WRONG-LOGIC",
        Verdict::Suspect => "suspect",
        Verdict::CoverageOnly => "coverage-only",
    }
}

// ---------------------------------------------------------------------------
// Ground truth: decode-all with the crate decoders
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsnClass {
    /// Calls / throws (covered by Call-family IR ops).
    Call,
    /// Unconditional / conditional branches, returns.
    Branch,
    /// Memory load or store (any Memory operand in final IR covers).
    LoadStore,
    /// Register arithmetic / moves / compares.
    Arith,
    /// No semantic effect on managed logic (nop, barriers, ADRP pairs...).
    Pseudo,
    /// Float/SIMD — covered by FloatArith/Memory IR.
    Float,
    /// Unknown to the classifier (flagged unless the IR produced something).
    Other,
}

impl InsnClass {
    fn name(self) -> &'static str {
        match self {
            InsnClass::Call => "call",
            InsnClass::Branch => "branch",
            InsnClass::LoadStore => "loadstore",
            InsnClass::Arith => "arith",
            InsnClass::Pseudo => "pseudo",
            InsnClass::Float => "float",
            InsnClass::Other => "other",
        }
    }
}

/// Semantic facts extracted from the DECODED native instruction — the
/// ground-truth side of the per-address semantic diff. Typed extraction via
/// the yaxpeax AST (not text parsing), so the diff is exact.
#[derive(Debug, Clone, Default)]
pub struct NativeFacts {
    /// (condition code 0..15, branch target) for conditional branches.
    pub cond_branch: Option<(u8, u64)>,
    /// Unconditional branch target.
    pub branch_target: Option<u64>,
    /// (target, is_indirect) for calls.
    pub call: Option<(u64, bool)>,
    /// (base reg num, addend) for offset-addressed loads/stores.
    pub mem: Option<(u16, i64)>,
    /// (base reg num, index reg num) for register-indexed loads/stores.
    pub mem_indexed: Option<(u16, u16)>,
    /// TBZ/TBNZ bit index.
    pub tbz_bit: Option<u32>,
    /// Integer arithmetic kind (add/sub/and/orr/eor/mul/div) + dest reg num
    /// + (lhs reg num, rhs reg num) when the form is plain reg-op-reg.
    pub arith: Option<(ArithKind, u16, Option<(u16, u16)>)>,
    /// MOV-class constant: (dest reg num, value).
    pub mov_imm: Option<(u16, u64)>,
}

#[derive(Debug, Clone)]
pub struct DecodedInsn {
    pub addr: u64,
    /// Encoded length in bytes — i.e. the distance to this instruction's
    /// FALLTHROUGH, which is exactly what a conditional branch's not-taken
    /// successor address is.
    ///
    /// This has to come from the decoder rather than be assumed. The semantic
    /// diff's inverted-branch acceptance used to derive the not-taken successor
    /// as `addr + 4`, which is the ARM64 instruction size and nothing else. On
    /// x86-64 a `jcc rel8` is TWO bytes and a `jcc rel32` is SIX, so on that
    /// corpus the acceptance could never fire and every CORRECTLY-inverted
    /// branch was reported as a wrong branch target — by far the largest cluster
    /// in the wrong-logic census.
    pub size: usize,
    /// First token of the decoder's textual form ("add", "ldr", …).
    pub mnemonic: String,
    pub class: InsnClass,
    /// Typed ground-truth facts, from `facts_arm64` / `facts_arm32` / `facts_x86`
    /// — ALL THREE decode paths populate this, and both the semantic diff and
    /// the reachability walk consume it. (This comment used to claim "arm64
    /// decode path only", which is false and cost a misdiagnosis: the ARM32
    /// inline-table leak was blamed on `facts` being empty for ARM32 when
    /// `facts_arm32` had been setting `branch_target` for an unconditional `b`
    /// all along. A fact that is simply not set is still a fact — check the
    /// extractor, not this comment.)
    pub facts: NativeFacts,
}

fn classify_arm64(mnemonic: &str) -> InsnClass {
    match mnemonic {
        "bl" | "blr" => InsnClass::Call,
        "b" | "b.eq" | "b.ne" | "b.hs" | "b.lo" | "b.mi" | "b.pl" | "b.vs" | "b.vc" | "b.hi"
        | "b.ls" | "b.ge" | "b.lt" | "b.gt" | "b.le" | "b.al" | "cbz" | "cbnz" | "tbz" | "tbnz"
        | "ret" | "br" => InsnClass::Branch,
        "ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" | "ldrsb" | "ldrsh" | "ldrsw" | "ldp"
        | "stp" | "ldnp" | "stnp" | "ldur" | "stur" | "ldurb" | "sturb" | "ldurh" | "sturh"
        | "prfm" => InsnClass::LoadStore,
        "nop" | "brk" | "udf" | "dmb" | "dsb" | "isb" | "hint" | "msr" | "mrs" | "adr" | "adrp"
        | "svc" | "hvc" | "smc" | "wfe" | "wfi" | "sev" | "yield" => InsnClass::Pseudo,
        // `fcmpe`/`fccmp`/`fccmpe` belong here with `fcmp`: they are the FP
        // compares (the `-e` suffix only adds the "raise on quiet NaN" rule,
        // and `fccmp` is the conditional-compare that joins two float tests).
        // They were absent, so all three classified as `Other` and every one of
        // them was reported as uncovered semantics — 64 `fccmp` in the arm64
        // reference sample — even though the lifter folds them into the
        // consumer's `Compare` exactly like `fcmp`.
        "fmov" | "fcmp" | "fcmpe" | "fccmp" | "fccmpe" | "fcsel" | "fcvt" | "fcvtzs" | "fcvtzu"
        | "fcvtas" | "fcvtau" | "fcvtps" | "fcvtpu" | "fcvtms" | "fcvtmu" | "fcvtns" | "fcvtnu"
        | "scvtf" | "ucvtf" | "fadd" | "fsub" | "fmul" | "fdiv" | "fsqrt" | "fabs" | "fneg"
        | "fmax" | "fmin" | "fmadd" | "fmsub" | "fnmadd" | "fnmsub" | "frintn" | "frintp"
        | "frintm" | "frintz" | "frinta" | "frintx" | "frinti" => InsnClass::Float,
        "add" | "adds" | "sub" | "subs" | "mul" | "madd" | "msub" | "smull" | "umull" | "sdiv"
        | "udiv" | "and" | "ands" | "orr" | "orn" | "eor" | "eon" | "bic" | "mvn" | "lsl"
        | "lsr" | "asr" | "ror" | "clz" | "rbit" | "rev" | "rev16" | "rev32" | "sxtb" | "sxth"
        | "sxtw" | "uxtb" | "uxth" | "uxtw" | "cmp" | "cmn" | "tst" | "ccmp" | "ccmn" | "csel"
        | "csinc" | "csinv" | "csneg" | "cset" | "csetm" | "cinc" | "cinv" | "cneg" | "mov"
        | "movz" | "movk" | "movn" | "sbfm" | "ubfm" | "sbfiz" | "ubfiz" | "extr" | "bfi"
        | "bfm" | "bfxil" => InsnClass::Arith,
        // NEON vector ops: no managed-logic semantic today (the lifter
        // ignores them); classifying as Pseudo keeps the oracle honest
        // instead of mass-flagging SIMD-heavy methods.
        "movi" | "mvni" | "dup" | "sshll" | "ushll" | "shl" | "sshr" | "ushr" | "srshr"
        | "urshr" | "sqadd" | "uqadd" | "sqsub" | "uqsub" | "sqdmulh" | "sqrdmulh" | "smax"
        | "umax" | "smin" | "umin" | "smaxp" | "umaxp" | "addp" | "addv" | "saddlv" | "uaddlv"
        | "saddp" | "uaddp" | "xtl" | "xtl2" | "shrn" | "rshrn" | "sqrshrn" | "uqrshrn"
        | "sqshrn" | "uqshrn" | "sri" | "sli" | "tbl" | "tbx" | "uzp1" | "uzp2" | "zip1"
        | "zip2" | "trn1" | "trn2" | "ins" | "umov" | "smov" | "sqxtn" | "sqxtun" | "uqxtn"
        | "sqshrun" | "uqshrun" | "fcvtn" | "fcvtl" | "fcvtn2" | "fcvtl2" | "rev64" | "ext"
        | "pmul" | "pmull" | "ld1r" | "st1" | "ld1" | "ld2" | "ld3" | "ld4" | "st2" | "st3"
        | "st4" => InsnClass::Pseudo,
        _ => InsnClass::Other,
    }
}

/// Strip an ARM32 condition suffix ("beq" → "b", "movwge" stays "movw").
/// The base A32 mnemonic, with its condition AND its flag-setting `s` removed —
/// by delegating to the LIFTER'S OWN splitter rather than reimplementing it.
///
/// The previous local version only stripped a condition when the base was a
/// SINGLE character (it tested `mnemonic[1..]`), so `b`+`eq`/`ls` worked and
/// nothing else did. Every multi-character predicated or flag-setting mnemonic
/// therefore classified as `Other`: `cmps`/`tsts` (ARM appends `s` for the
/// flag-setting form, so `cmp`+`s`) and `ldreq`/`strne`/`ldrvs` (predicated
/// loads/stores). On the 1,941-method ARM32 sample that was 6,723 `cmps` +
/// 820 `tsts` real instructions reported as unknown and counted as uncovered
/// semantics — a pure measurement artifact that swamped the real findings.
/// The lifter already normalizes both suffixes in `split`, so sharing it is the
/// only way the oracle and the engine cannot disagree about what an instruction
/// IS.
fn base_mnemonic32(mnemonic: &str) -> &str {
    super::lift_arm32::split(mnemonic).0
}

/// Mnemonic identity for the mnemonic LIST tests below, normalised the same way
/// the lifter normalises it.
///
/// The lists are spelled in AArch64 (`cmp`, `tst`, `adds`), and ARM32 appends
/// the flag-setting `s` and a condition code: `cmps`, `tsteq`, `addseq`. So
/// `CMP_MNEMONICS.contains("cmps")` was false, the cmp→branch fold exemption
/// never fired, and 7,102 `cmp`/`tst` instructions that HAD been lifted (as the
/// `Compare` at the branch they feed) were reported as uncovered semantics — a
/// measurement artifact sitting at the top of the cluster table.
fn base_mnemonic(m: &str, is_arm32: bool) -> &str {
    if is_arm32 { base_mnemonic32(m) } else { m }
}

/// Does this decoded instruction READ the flags to produce a VALUE instead of
/// a control decision?
///
/// These are the comparison's OTHER consumers. The lifter materializes a
/// pending compare into a `Compare` at whichever instruction consumes it, so
/// the coverage oracle has to treat a select exactly like a branch when it
/// decides whether a decoded `cmp` was accounted for.
///
/// The list is by MNEMONIC rather than by decoding the operand pattern, for the
/// same reason the branch list is: on AArch64 the alias spellings (`cset`,
/// `cinc`, `cneg`, `csetm`) are what the decoder prints for the `CSINC`/`CSINV`
/// encodings, and those encodings are precisely the flags-reading `Xn : Xm ± 1`
/// forms. x86 reaches the same IR through `cmovcc`/`setcc`, whose spellings are
/// unbounded (`cmovae`, `setg`, …), so those match on prefix.
///
/// ARM32 has no select family in this IR: its conditional `mov` is the
/// consumer, and `base_mnemonic32` has already stripped the condition, so a
/// `mov` whose ORIGINAL spelling carried a condition is the one that reads the
/// flags (`movne`), while a plain `mov` writes them not at all.
fn consumes_flags(raw_mnemonic: &str, base_mnemonic: &str, is_arm32: bool) -> bool {
    const SELECT_CONSUMERS: &[&str] = &[
        "csel", "csinc", "csinv", "csneg", "fcsel", "cset", "csetm", "cinc", "cinv", "cneg",
    ];
    if SELECT_CONSUMERS.contains(&base_mnemonic) {
        return true;
    }
    if is_arm32 {
        // A32 predicates EVERY instruction, so any spelling carrying a
        // condition code READS the flags. But the walk may only STOP here when
        // the lifter actually MATERIALISED a `Compare` at this address —
        // otherwise the walk must continue to the real consumer. The lifter
        // materialises the compare for a predicated `mov`/`mvn` (as an
        // `Op::Select`, see `lift_arm32`), for `popeq {…, pc}` / `bx<cc> lr`
        // (`emit_predication_guard`), and for the EXT group. A predicated
        // `orrne`/`ldrgt`/`addne` does NOT write the flags, so a `cmp` before
        // it is still live at the next branch — stopping here would report a
        // comparison that is really accounted for at that branch as dropped.
        // The condition is read from the SAME splitter the lifter dispatches
        // on, so the two cannot disagree about whether an instruction is
        // predicated. `movs` (flag-setting, no condition) reads nothing here,
        // which is correct.
        // NOTE: `raw_mnemonic` here is only the first TOKEN (`popne`), not the
        // operand text, so the `pc` in `popne {…, pc}` is NOT visible. A
        // predicated `mov`/`mvn` is materialised as an `Op::Select` and a
        // predicated return `pop`/`bx` via `emit_predication_guard`, all at
        // their own address, so those are the forms that stop the walk. For
        // `pop` the clause is deliberately unconditional: a predicated pop
        // WITHOUT `pc` does not materialise a compare, but it also cannot end
        // the path, so the flags stay live and the next branch would credit the
        // comparison — that over-approximation costs at most the handful of
        // predicated non-return pops, against 1,288 real `pop{cc} {…, pc}`
        // returns mis-reported as dropped comparisons.
        let (base, cond, _) = super::lift_arm32::split(raw_mnemonic);
        if cond.is_empty() {
            return false;
        }
        // The value-producing forms that are materialised as an `Op::Select`
        // are `mov`/`mvn` ONLY. The ALU group (`add`/`sub`/`rsb`/`and`/`orr`/
        // `eor`/`bic`) was listed here while a predicated-ALU Select lowering
        // was under test, but that lowering was REVERTED (it forced a read of
        // the destination, so a predicated op that was the first writer of its
        // destination produced an undefined local — see `lift_arm32`). The
        // lifter now leaves those ops unconditional, so they materialise NO
        // `Compare`; stopping the walk at one would report the live comparison
        // as dropped. The oracle has to track the lifter, not the abandoned
        // experiment.
        //
        // The return forms (`pop`, `bx`) stop via `emit_predication_guard`. A
        // predicated LOAD/STORE is still lifted unconditionally, so it must NOT
        // stop the walk — the comparison is genuinely still live at the next
        // branch.
        return matches!(base, "mov" | "mvn") || base == "pop" || base == "bx";
    }
    base_mnemonic.starts_with("cmov") || base_mnemonic.starts_with("set")
}

fn classify_arm32(mnemonic: &str) -> InsnClass {
    match base_mnemonic32(mnemonic) {
        "bl" | "blx" => InsnClass::Call,
        // `bx` is a REGISTER BRANCH, not a call — the same split arm64 makes
        // between `blr` (Call) and `br` (Branch). Its overwhelmingly common
        // form `bx lr` is the ARM32 RETURN, which the lifter lowers to
        // `Op::Return`; classed as Call it was covered by nothing, and every
        // return in the binary was reported as dropped call semantics —
        // 11,626 `bx [call]` findings, the second-largest cluster in the
        // ARM32 census and entirely an artifact. `bx rN` lifts to
        // `Op::IndirectCall` (a virtual tail call) and is still covered,
        // because a Branch-class address accepts a Call-op (`Coverage::covers`,
        // the `BR xN` case). `conform::is_call_site` already excluded
        // `bx`/`br` from its call count for this exact reason.
        "bx" => InsnClass::Branch,
        "b" | "b.w" | "cbz" | "cbnz" => InsnClass::Branch,
        "ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" | "ldrd" | "strd" | "ldm"
        | "stm" | "push" | "pop" | "vldr" | "vstr" | "vldm" | "vstm" => InsnClass::LoadStore,
        "nop" | "bkpt" | "udf" | "dmb" | "dsb" | "isb" | "adr" | "svc" => InsnClass::Pseudo,
        // `cdp2`/`mcr2`/`smc` are the ARM32 no-op family the lifter deliberately
        // drops (they write no ARM architectural register and their effect is
        // only observable inside a coprocessor the recovered program cannot
        // see). Classed `Other` they demanded "something at this address" and
        // were reported as dropped semantics — 74 `cdp2` in the ARM32 census,
        // the top cluster that was not a real finding. `Pseudo` covers them the
        // same way `nop` is covered.
        "cdp2" | "mcr2" | "smc" => InsnClass::Pseudo,
        "vmov" | "vadd" | "vsub" | "vmul" | "vdiv" | "vcmp" | "vcvt" | "vsqrt"
        | "vabs" | "vneg" | "vmla" | "vmls" | "vnmla" | "vnmls" => InsnClass::Float,
        "mov" | "movw" | "movt" | "mvn" | "add" | "adds" | "sub" | "subs" | "mul"
        | "mla" | "mls" | "and" | "orr" | "eor" | "bic" | "orn" | "lsl" | "lsr"
        | "asr" | "ror" | "rrx" | "clz" | "rbit" | "rev" | "rev16" | "sdiv" | "udiv"
        | "cmp" | "cmn" | "tst" | "teq" | "sxtb" | "sxth" | "uxtb" | "uxth"
        // The two-lane spellings are part of the same group (`ext_op`) and were
        // reaching `Other`, which then demanded "something at this address"
        // instead of the arithmetic the lifter really emits there.
        | "sxtb16" | "uxtb16" | "addw" | "subw" => InsnClass::Arith,
        _ => InsnClass::Other,
    }
}

fn classify_x86(mnemonic: &str) -> InsnClass {
    match mnemonic {
        "call" => InsnClass::Call,
        "jmp" | "je" | "jne" | "jg" | "jl" | "jge" | "jle" | "ja" | "jb" | "jae" | "jbe" | "js"
        | "jns" | "jo" | "jno" | "jp" | "jnp" | "jrcxz" | "jecxz" | "ret" | "leave" => {
            InsnClass::Branch
        }
        "mov" | "movzx" | "movsx" | "movsxd" | "lea" | "xchg" | "sete" | "setne" | "setg"
        | "setl" => InsnClass::Arith,
        // Non-semantic filler / markers. `0xCC` (`int3`) is the inter-function
        // ALIGNMENT pad, and the body extent runs to the NEXT method's start,
        // so a measured body swallows every padding byte up to it. Classed as
        // Arith they were reported uncovered with no IR to produce (114,127
        // `int3` — the single largest cluster — plus 2,820 `nop`) even though
        // the lifter is right to drop them. `nop` stays in this group for the
        // same reason it is in `is_folded_into_successor`: it has no effect.
        "nop" | "int3" | "int" | "ud2" | "endbr64" | "endbr32" => InsnClass::Pseudo,
        "push" | "pop" => InsnClass::LoadStore,
        _ => {
            if mnemonic.starts_with("j") && mnemonic.len() >= 2 {
                InsnClass::Branch // conditional-move/setcc families fall here safely
            } else if mnemonic.starts_with("cmov") || mnemonic.starts_with("set") {
                InsnClass::Arith
            } else if mnemonic.starts_with('f')
                || mnemonic.starts_with("vadd")
                || mnemonic.starts_with("vsub")
                || mnemonic.starts_with("vmul")
                || mnemonic.starts_with("vdiv")
            {
                InsnClass::Float
            } else {
                InsnClass::Arith
            }
        }
    }
}

/// Decode the whole body linearly. Returns one entry per successfully decoded
/// instruction; addresses that fail to decode are counted, not fatal.
/// Extract semantic facts from one decoded arm64 instruction using the
/// yaxpeax AST. This is the INDEPENDENT side of the semantic diff — it reads
/// the native bytes, never the lifter's output.
fn facts_arm64(inst: &yaxpeax_arm::armv8::a64::Instruction, iaddr: u64) -> NativeFacts {
    use yaxpeax_arm::armv8::a64::{Opcode as Op64, Operand as O64, SizeCode};
    let mut f = NativeFacts::default();
    let ops = &inst.operands;
    let gpr = |o: &O64| -> Option<u16> {
        match o {
            // shift-0 shifted-register is a plain register (`add x0, x1, x2`
            // decodes its third operand as RegShift(LSL, 0, X, 2)); register 31
            // is the zero register, not addressable data.
            O64::Register(_, n) if *n != 31 => Some(*n),
            O64::RegShift(_, 0, _, n) if *n != 31 => Some(*n),
            _ => None,
        }
    };
    // any immediate operand (sign-corrected where the encoding is signed)
    let imm_of = |o: &O64| -> Option<i64> {
        match o {
            O64::Immediate(v) => Some(*v as i64),
            O64::Imm16(v) => Some(*v as i64),
            O64::Imm64(v) => Some(*v as i64),
            O64::PCOffset(_) => None,
            _ => None,
        }
    };
    match inst.opcode {
        Op64::Bcc(cond) => {
            let target = ops.iter().find_map(|o| match o {
                O64::PCOffset(off) => Some((iaddr as i64 + off) as u64),
                _ => None,
            });
            if let Some(t) = target {
                f.cond_branch = Some((cond & 0xF, t));
            }
        }
        Op64::CBZ | Op64::CBNZ => {
            let code = if inst.opcode == Op64::CBZ { 0 } else { 1 };
            let target = ops.iter().find_map(|o| match o {
                O64::PCOffset(off) => Some((iaddr as i64 + off) as u64),
                _ => None,
            });
            if let Some(t) = target {
                f.cond_branch = Some((code, t));
            }
        }
        Op64::TBZ | Op64::TBNZ => {
            let code = if inst.opcode == Op64::TBZ { 0 } else { 1 };
            let target = ops.iter().find_map(|o| match o {
                O64::PCOffset(off) => Some((iaddr as i64 + off) as u64),
                _ => None,
            });
            // yaxpeax encodes the tbz/tbnz bit index as Imm16, not
            // Immediate; reading only Immediate left tbz_bit at None, so the
            // bit-number oracle never ran on any site.
            f.tbz_bit = ops.iter().find_map(|o| imm_of(o).map(|v| v as u32));
            if let Some(t) = target {
                f.cond_branch = Some((code, t));
            }
        }
        Op64::B => {
            f.branch_target = ops.iter().find_map(|o| match o {
                O64::PCOffset(off) => Some((iaddr as i64 + off) as u64),
                _ => None,
            });
        }
        Op64::BL => {
            let target = ops.iter().find_map(|o| match o {
                O64::PCOffset(off) => Some((iaddr as i64 + off) as u64),
                _ => None,
            });
            f.call = Some((target.unwrap_or(0), false));
        }
        Op64::BLR => {
            f.call = ops.first().and_then(gpr).map(|r| (r as u64, true));
        }
        Op64::LDR
        | Op64::LDRB
        | Op64::LDRSB
        | Op64::LDRSW
        | Op64::LDRSH
        | Op64::LDRH
        | Op64::LDUR
        | Op64::LDURB
        | Op64::LDURSB
        | Op64::LDURSW
        | Op64::LDURSH
        | Op64::LDURH
        | Op64::STR
        | Op64::STRB
        | Op64::STRH
        | Op64::STRW
        | Op64::STUR
        | Op64::STURB
        | Op64::STURH => {
            // operand 1 is the address operand in every yaxpeax arm64 ldr/str form
            if let Some(addr_op) = ops.get(1) {
                match addr_op {
                    O64::RegPreIndex(base, off, _) => f.mem = Some((*base, *off as i64)),
                    // A post-indexed access reads or writes at the original
                    // base; its immediate updates the base only after the
                    // memory operation. Keep that writeback out of the
                    // access-offset fact, matching `memory_operand` in the
                    // lifter. Treating it as the access displacement falsely
                    // flagged every negative post-index store as wrong logic.
                    O64::RegPostIndex(base, _) => f.mem = Some((*base, 0)),
                    O64::RegRegOffset(b, i, _, _, _) => f.mem_indexed = Some((*b, *i)),
                    _ => {}
                }
            }
        }
        Op64::ADD
        | Op64::ADDS
        | Op64::SUB
        | Op64::SUBS
        | Op64::AND
        | Op64::ANDS
        | Op64::ORR
        | Op64::EOR
        | Op64::MUL
        | Op64::SDIV
        | Op64::UDIV
        | Op64::MADD
        | Op64::MSUB => {
            let kind = match inst.opcode {
                Op64::ADD | Op64::ADDS => ArithKind::Add,
                Op64::SUB | Op64::SUBS => ArithKind::Sub,
                Op64::AND | Op64::ANDS => ArithKind::And,
                Op64::ORR => ArithKind::Or,
                Op64::EOR => ArithKind::Xor,
                Op64::MUL | Op64::MADD => ArithKind::Mul,
                Op64::MSUB => ArithKind::Sub,
                Op64::SDIV | Op64::UDIV => ArithKind::Div,
                _ => unreachable!(),
            };
            let d = ops.first().and_then(gpr);
            // 3-operand reg-op-reg: (d, n, m). Flags-setting forms (adds/subs
            // with xzr dest) render through Compare in the lifter — skip
            // when dest is the zero register (gpr() returns None for 31).
            if let Some(d) = d {
                if ops.len() >= 3 {
                    let n = gpr(&ops[1]);
                    let m = gpr(&ops[2]);
                    if let (Some(n), Some(m)) = (n, m) {
                        f.arith = Some((kind, d, Some((n, m))));
                    }
                }
            }
        }
        Op64::MOVZ | Op64::MOVK | Op64::MOVN => {
            // textual MOVZ/MOVN map to the same opcodes with (d, imm, shift)
            let d = ops.first().and_then(gpr);
            let v = ops.get(1).and_then(imm_of).map(|v| v as u64).unwrap_or(0);
            let sh = ops
                .get(2)
                .and_then(|o| match o {
                    O64::Immediate(v) => Some(*v),
                    _ => None,
                })
                .unwrap_or(0) as u32;
            if let Some(d) = d {
                let val = if inst.opcode == Op64::MOVN {
                    !((v as u32) << sh) as u64
                } else {
                    (v << sh) as u64
                };
                f.mov_imm = Some((d, val));
            }
        }
        _ => {}
    }
    let _ = SizeCode::X; // silence unused-import when feature combos change
    f
}

/// Extract semantic facts from one decoded armv7/A32 instruction.
///
/// ARM32 is the architecture of every 32-bit Android il2cpp binary, and until
/// now it produced NO facts at all — `decode_all_arm32` filled
/// `NativeFacts::default()`, and the semantic diff only runs on facts. So on the
/// ARM32 corpus the question "does the recovered C# mean what the bytes mean"
/// was not merely unanswered, it was unaskable: coverage could say an op exists
/// at every address while nothing said whether it was the RIGHT op.
///
/// Branch targets mirror the lifter's arithmetic exactly. yaxpeax folds A32's
/// architectural PC bias INTO the operand (`(imm24 + 2)` before the display's
/// `<< 2`), so `iaddr + (off << 2)` is the byte target — the same value the
/// lifter's `parse_target` derives from the printed `$+0x…` text. Deriving it
/// here from the AST (rather than re-parsing text) keeps the oracle independent
/// of the lifter's string handling while agreeing on the arithmetic, so a
/// reported mismatch is a real downstream defect and never a convention dispute.
fn facts_arm32(inst: &yaxpeax_arm::armv7::Instruction, iaddr: u64) -> NativeFacts {
    use yaxpeax_arm::armv7::{ConditionCode, Opcode as Op32, Operand as O32};
    let mut f = NativeFacts::default();
    let ops = &inst.operands;

    let rn = |o: &O32| -> Option<u16> {
        match o {
            O32::Reg(r) => Some(r.number() as u16),
            _ => None,
        }
    };
    // Byte target of a pc-relative branch operand. The `<< 2` / `<< 1` match the
    // decoder's own display shift, which is where the PC bias already lives.
    let branch_at = |o: &O32| -> Option<u64> {
        match o {
            O32::BranchOffset(off) => Some((iaddr as i64 + ((*off as i64) << 2)) as u64),
            O32::BranchThumbOffset(off) => Some((iaddr as i64 + ((*off as i64) << 1)) as u64),
            _ => None,
        }
    };
    let any_target = || ops.iter().find_map(branch_at);
    // Condition code in the shared numeric space `lift::cond_to_kind` speaks.
    // `None` for AL: an unconditional instruction has no condition to check.
    let cond = |c: &ConditionCode| -> Option<u8> {
        Some(match c {
            ConditionCode::EQ => 0,
            ConditionCode::NE => 1,
            ConditionCode::HS => 2,
            ConditionCode::LO => 3,
            ConditionCode::MI => 4,
            ConditionCode::PL => 5,
            ConditionCode::VS => 6,
            ConditionCode::VC => 7,
            ConditionCode::HI => 8,
            ConditionCode::LS => 9,
            ConditionCode::GE => 10,
            ConditionCode::LT => 11,
            ConditionCode::GT => 12,
            ConditionCode::LE => 13,
            ConditionCode::AL => return None,
        })
    };
    // The `add` bool on every pre/post-index form is the first one (the second
    // is writeback, which no consumer of these facts reads). A subtracted offset
    // is signed here so the comparison against the IR addend is like-for-like.
    let mut mem_of = |o: &O32| -> bool {
        match o {
            O32::RegDeref(base) => {
                f.mem = Some((base.number() as u16, 0));
                true
            }
            O32::RegDerefPreindexOffset(base, off, add, _)
            | O32::RegDerefPostindexOffset(base, off, add, _) => {
                let v = if *add { *off as i64 } else { -(*off as i64) };
                f.mem = Some((base.number() as u16, v));
                true
            }
            O32::RegDerefPreindexReg(base, idx, _, _)
            | O32::RegDerefPostindexReg(base, idx, _, _) => {
                f.mem_indexed = Some((base.number() as u16, idx.number() as u16));
                true
            }
            // Shifted-register index: only the fact that the address IS indexed
            // is consumed, so the shift amount is deliberately not carried.
            O32::RegDerefPreindexRegShift(base, _, _, _)
            | O32::RegDerefPostindexRegShift(base, _, _, _) => {
                f.mem_indexed = Some((base.number() as u16, 0));
                true
            }
            _ => false,
        }
    };

    match inst.opcode {
        Op32::B => match cond(&inst.condition) {
            Some(c) => f.cond_branch = any_target().map(|t| (c, t)),
            None => f.branch_target = any_target(),
        },
        // `cbz`/`cbnz` test their register against zero and carry no condition
        // code of their own — the same two codes the arm64 path uses.
        Op32::CBZ | Op32::CBNZ => {
            let c = if inst.opcode == Op32::CBZ { 0 } else { 1 };
            f.cond_branch = any_target().map(|t| (c, t));
        }
        Op32::BL | Op32::BLX => {
            if let Some(t) = any_target() {
                f.call = Some((t, false));
            }
        }
        Op32::LDR
        | Op32::STR
        | Op32::LDRB
        | Op32::STRB
        | Op32::LDRH
        | Op32::STRH
        | Op32::LDRSH
        | Op32::LDRSB
        | Op32::LDRD
        | Op32::STRD => {
            // operand 0 is the data register; the address is operand 1
            if let Some(a) = ops.get(1) {
                mem_of(a);
            }
        }
        Op32::ADD
        | Op32::SUB
        | Op32::AND
        | Op32::ORR
        | Op32::EOR
        | Op32::MUL
        | Op32::SDIV
        | Op32::UDIV => {
            let kind = match inst.opcode {
                Op32::ADD => ArithKind::Add,
                Op32::SUB => ArithKind::Sub,
                Op32::AND => ArithKind::And,
                Op32::ORR => ArithKind::Or,
                Op32::EOR => ArithKind::Xor,
                Op32::MUL => ArithKind::Mul,
                Op32::SDIV | Op32::UDIV => ArithKind::Div,
                _ => unreachable!(),
            };
            // 3-operand reg-op-reg only (`d <= n op m`). Immediate and shifted
            // forms set no fact, which is the conservative direction: an ABSENT
            // fact never flags anything, so a miss costs nothing and a false
            // report is impossible.
            if let Some(d) = ops.first().and_then(rn) {
                if let (Some(n), Some(m)) = (rn(&ops[1]), rn(&ops[2])) {
                    f.arith = Some((kind, d, Some((n, m))));
                }
            }
        }
        Op32::MOV => {
            // `mov rd, #imm` — a real constant. Register moves set no fact.
            let d = ops.first().and_then(rn);
            let v = ops.get(1).and_then(|o| match o {
                O32::Imm12(v) => Some(*v as u64),
                O32::Imm32(v) => Some(*v as u64),
                _ => None,
            });
            if let (Some(d), Some(v)) = (d, v) {
                f.mov_imm = Some((d, v));
            }
        }
        _ => {}
    }
    f
}

/// Extract semantic facts from one decoded x86/x64 instruction (iced-x86 AST).
///
/// The third platform that previously had no facts at all. Condition codes are
/// canonicalized into the SAME numeric space as ARM (`lift::cond_to_kind`), one
/// code per x86 condition, chosen so the kind the x86 lifter's own `cond_kind`
/// emits is exactly what that space yields. That is not the same as ARM's
/// unsigned semantics: this lifter collapses `be`/`le` to a signed `LessThan`
/// and `a`/`g`/`ae`/`ge` to `GreaterOrEqual`, so those canonicalize to codes that
/// produce those kinds — comparing against ARM's `LS`/`LE`/`HI`/`GT` slots would
/// flag every correct branch.
fn facts_x86(inst: &iced_x86::Instruction) -> NativeFacts {
    use iced_x86::{Mnemonic as M86, OpKind, Register as R86};
    let mut f = NativeFacts::default();
    let m = inst.mnemonic();
    let near = || -> Option<u64> {
        matches!(
            inst.op0_kind(),
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64
        )
        .then(|| inst.near_branch_target())
    };
    let cond_code: Option<u8> = match m {
        M86::Je => Some(0),
        M86::Jne => Some(1),
        // iced decodes the shared opcodes to their canonical mnemonic, so the
        // `jnae`/`jc`/`jnb`/`jnc` spellings never reach here.
        M86::Jb => Some(3),
        M86::Jae => Some(2),
        M86::Ja => Some(2),
        M86::Jbe => Some(11),
        M86::Jl => Some(11),
        M86::Jle => Some(11),
        M86::Jg => Some(2),
        M86::Jge => Some(2),
        M86::Js => Some(4),
        M86::Jns => Some(5),
        M86::Jo => Some(6),
        M86::Jno => Some(7),
        _ => None,
    };
    if let Some(c) = cond_code {
        if let Some(t) = near() {
            f.cond_branch = Some((c, t));
        }
    } else if m == M86::Jmp {
        if let Some(t) = near() {
            f.branch_target = Some(t);
        }
    } else if m == M86::Call {
        if let Some(t) = near() {
            f.call = Some((t, false));
        }
    }
    // Memory operand: base/index/displacement from the decoder's own operand
    // analysis, so `[rax+rbx*8+0x10]` is described exactly. RIP-relative
    // operands have no base and set no fact — conservative, never a false hit.
    //
    // `lea` is excluded: it has a memory-SHAPED operand but performs no
    // access — it computes an address, and the stack analyzer legitimately
    // rebases that address into a slot (a prologue `mov rax,rsp; lea
    // rbp,[rax-0x5f]` lands in the IR as a `stack_N` rebase, not a load at
    // -95). Comparing its displacement to an IR memory addend flagged every
    // such prologue. The ARM facts never set `mem` for an address computation,
    // so this also keeps the architectures speaking the same language.
    if inst.memory_base() != R86::None && m != M86::Lea {
        let base = inst.memory_base() as u32 as u16;
        let idx = inst.memory_index();
        if idx != R86::None {
            f.mem_indexed = Some((base, idx as u32 as u16));
        } else {
            f.mem = Some((base, inst.memory_displacement64() as i64));
        }
    }
    let kind = match m {
        M86::Add => Some(ArithKind::Add),
        M86::Sub => Some(ArithKind::Sub),
        M86::And => Some(ArithKind::And),
        M86::Or => Some(ArithKind::Or),
        M86::Xor => Some(ArithKind::Xor),
        M86::Mul | M86::Imul => Some(ArithKind::Mul),
        M86::Div | M86::Idiv => Some(ArithKind::Div),
        M86::Shl => Some(ArithKind::Shl),
        M86::Shr | M86::Sar => Some(ArithKind::Shr),
        _ => None,
    };
    if let Some(k) = kind {
        // Destination/pair are not consumed by the diff (it compares the KIND
        // at an address); a register-valued `mov` is not a constant either.
        f.arith = Some((k, 0, None));
    }
    f
}

pub fn decode_all_arm64(bytes: &[u8], base_rva: u64) -> (Vec<DecodedInsn>, u64) {
    use yaxpeax_arch::{Decoder, LengthedInstruction, U8Reader};
    let decoder = yaxpeax_arm::armv8::a64::InstDecoder::default();
    let mut out = Vec::new();
    let mut fails = 0u64;
    let mut off = 0usize;
    while off + 4 <= bytes.len() {
        let mut reader = U8Reader::new(&bytes[off..]);
        match decoder.decode(&mut reader) {
            Ok(inst) => {
                let size = inst.len().to_const() as usize;
                let facts = facts_arm64(&inst, base_rva + off as u64);
                let text = format!("{inst}");
                let mnemonic = text.split_whitespace().next().unwrap_or("?").to_string();
                let class = classify_arm64(&mnemonic);
                out.push(DecodedInsn {
                    addr: base_rva + off as u64,
                    size,
                    mnemonic,
                    class,
                    facts,
                });
                off += size;
            }
            Err(_) => {
                fails += 1;
                off += 4;
            }
        }
    }
    (out, fails)
}

pub fn decode_all_arm32(bytes: &[u8], base_rva: u64, thumb: bool) -> (Vec<DecodedInsn>, u64) {
    use yaxpeax_arch::{Decoder, LengthedInstruction, U8Reader};
    // Honouring `thumb` is not cosmetic: it selects which byte stream we are
    // claiming to read. `InstDecoder::default()` is ARM (A32) mode, which is the
    // mode the LIFTER uses — and the ground-truth decoder must read the same
    // stream as the thing it is checking, or every finding is about a different
    // program.
    let mut decoder = yaxpeax_arm::armv7::InstDecoder::default();
    decoder.set_thumb_mode(thumb);
    let mut out = Vec::new();
    let mut fails = 0u64;
    let mut off = 0usize;
    // `ILR_DECFAIL=1` dumps every word BOTH sides drop (the ones the lifter
    // recovers are not gaps and are not interesting here) as `[decfail] 0x…`
    // on stderr, so the residual population can be tabulated by encoding.
    let decfail = std::env::var_os("ILR_DECFAIL").is_some();
    while off < bytes.len() {
        // Architectural instruction size. A FAILED decode still occupies a full
        // instruction slot, so the failure path must advance by the same amount
        // as the success path. It used to advance by `step`, which callers pass
        // as 2 for every ARM32 binary — so a single instruction this crate
        // cannot decode shifted the whole remaining walk by 2 bytes and every
        // later instruction was decoded from the wrong half-word. That is not a
        // theoretical worry: `0xED9B…`/`0xED84…` (VFP `vldr`/`vstr`, in every
        // float-using method) are among the encodings yaxpeax-arm 0.4 rejects,
        // and slipping on one at 0x53c840 fabricated ~2,400 phantom conditional
        // branches at half-word addresses like 0x53c846 — reported as "native
        // conditional branch; no IR at this address" because no such branch
        // exists.
        let step: usize = if thumb { 2 } else { 4 };
        let mut reader = U8Reader::new(&bytes[off..]);
        match decoder.decode(&mut reader) {
            Ok(inst) => {
                let size = (inst.len().to_const() as usize).max(step);
                let text = format!("{inst}");
                let mnemonic = text.split_whitespace().next().unwrap_or("?").to_string();
                let mut class = classify_arm32(&mnemonic);
                // An ALU or load whose DESTINATION is `pc` is a CONTROL
                // TRANSFER, not arithmetic: `add pc, r1, r2` is the ARM32
                // jump-table dispatch, `mov pc, lr` a return, `ldr pc, [..]` a
                // computed goto. The lifter emits a branch/return for it (see
                // lift_arm32's `pc`-destination handling), so classing it by
                // mnemonic alone made the coverage oracle demand an `Arith`
                // the correct IR never produces — 31 spurious `add [arith]`
                // uncovered findings on the 4,000-method sample. `pc` as a
                // SOURCE (`add r0, pc, #8`) is ordinary arithmetic and stays
                // `Arith`.
                if matches!(class, InsnClass::Arith | InsnClass::LoadStore)
                    && text
                        .split_whitespace()
                        .nth(1)
                        .map(|s| s.trim_end_matches(',').trim_start_matches('#'))
                        .is_some_and(|d| d == "pc" || d == "r15")
                {
                    class = InsnClass::Branch;
                }
                out.push(DecodedInsn {
                    addr: base_rva + off as u64,
                    size,
                    mnemonic,
                    class,
                    facts: facts_arm32(&inst, base_rva + off as u64),
                });
                off += size;
            }
            Err(_) => {
                // The EXT group (`sxtb`/`sxth`/`uxtb`/`uxth`/`uxtb16`/`sxtb16`)
                // is undecodable by this crate, but the LIFTER lifts it from the
                // raw word (`lift_arm32::ext_op`). The oracle and the engine must
                // not disagree about which words are instructions — otherwise
                // every one of these is counted as a decode failure AND, since
                // the lifter now emits real IR there, the census would have an
                // unexplained instruction in its own listing. So the oracle
                // synthesizes the same entry the lifter produces.
                if off + 4 <= bytes.len() {
                    let word = u32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]);
                    // NEON / Advanced SIMD is likewise undecodable here, and the
                    // lifter emits the catch-all `Op::Simd` for it (see
                    // `lift_arm32::neon_word`). The oracle synthesizes the same
                    // entry, classified `Pseudo` so the coverage census speaks
                    // the same language as the `Op::Simd` it will find in the IR.
                    if super::lift_arm32::neon_word(word) {
                        out.push(DecodedInsn {
                            addr: base_rva + off as u64,
                            size: step,
                            mnemonic: "neon".to_string(),
                            class: InsnClass::Pseudo,
                            facts: NativeFacts::default(),
                        });
                        off += step;
                        continue;
                    }
                    if let Some(e) = super::lift_arm32::ext_op(word) {
                        out.push(DecodedInsn {
                            addr: base_rva + off as u64,
                            size: step,
                            mnemonic: super::lift_arm32::EXT_MNEMONICS[e.op as usize].to_string(),
                            class: InsnClass::Arith,
                            facts: NativeFacts::default(),
                        });
                        off += step;
                        continue;
                    }
                    // VFP words the lifter ABSORBS into their consumer — a
                    // `vcmp`/`vcmpe` flag setter, or the `vmrs`/`vmsr` that
                    // moves FPSCR↔APSR. They emit no IR, so the oracle must
                    // still carry an entry or `absorbed_landing` cannot certify
                    // a branch that lands on one (the dominant ARM32
                    // `branch target mismatch` shape). Classified `Pseudo`, like
                    // the NEON/EXT synthesis, so the coverage census speaks the
                    // same language as the lifter that dropped them.
                    if super::lift_arm32::vfp_absorbed_word(word) {
                        out.push(DecodedInsn {
                            addr: base_rva + off as u64,
                            size: step,
                            mnemonic: "vfp".to_string(),
                            class: InsnClass::Pseudo,
                            facts: NativeFacts::default(),
                        });
                        off += step;
                        continue;
                    }
                    // Not the EXT group — but the LIFTER may still handle it
                    // from the raw word (VFP coprocessor / signed DSP). Those
                    // are an oracle gap, not a reconstruction gap, and the
                    // census reports them separately. See `liftable_word`.
                    if super::lift_arm32::liftable_word(word) {
                        STATS
                            .decode_failures_lifted_total
                            .fetch_add(1, Ordering::Relaxed);
                    } else if decfail {
                        eprintln!("[decfail] {word:#010x}");
                    }
                } else if decfail {
                    eprintln!("[decfail] <truncated tail>");
                }
                fails += 1;
                off += step;
            }
        }
    }
    (out, fails)
}

pub fn decode_all_x86(bytes: &[u8], base_rva: u64) -> (Vec<DecodedInsn>, u64) {
    use iced_x86::{Decoder, DecoderOptions};
    let mut out = Vec::new();
    let mut fails = 0u64;
    let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
    decoder.set_ip(base_rva);
    while decoder.can_decode() {
        let inst = decoder.decode();
        if inst.is_invalid() {
            fails += 1;
            continue;
        }
        // The mnemonic used for CLASSIFICATION must come from iced's own enum,
        // not from the formatter's text. The two disagree exactly where it
        // matters: iced decodes the multi-byte NOP pad `66 90` as `Nopw`
        // (mnemonic `Nop`, the same as `90`) but its formatter renders it
        // `xchg ax,ax` — objdump's spelling. A table keyed on the TEXT therefore
        // classified 387 padding instructions as a dropped ARITHMETIC `xchg`,
        // the largest false cluster on the v24.5 corpus. Prefixed instructions
        // had the mirror-image bug: `rep movsb` tokenises to `rep`, not
        // `movsb`.
        let mnemonic = format!("{:?}", inst.mnemonic()).to_ascii_lowercase();
        let class = classify_x86(&mnemonic);
        out.push(DecodedInsn {
            addr: inst.ip(),
            size: inst.len(),
            mnemonic,
            class,
            facts: facts_x86(&inst),
        });
    }
    (out, fails)
}

// ---------------------------------------------------------------------------
// Coverage: which classes of decoded insn does the final IR account for?
// ---------------------------------------------------------------------------

fn collect_memory_operands<'a>(op: &'a Operand, out: &mut Vec<&'a Operand>) {
    match op {
        Operand::Memory { base, index, .. } => {
            out.push(op);
            collect_memory_operands(base, out);
            if let Some(i) = index {
                collect_memory_operands(i, out);
            }
        }
        Operand::AddressOf(inner) | Operand::ArrayLength(inner) => {
            collect_memory_operands(inner, out)
        }
        Operand::ArrayAccess { array, index } => {
            collect_memory_operands(array, out);
            collect_memory_operands(index, out);
        }
        _ => {}
    }
}

/// Memory operands that are real ACCESSES — a read or a write of the location.
///
/// The difference from [`collect_memory_operands`] is `AddressOf`: `&x` is an
/// ADDRESS COMPUTATION, not an access to `x`. `stack_analyzer` creates exactly
/// that shape for `add xD, sp, #k` (`Move xD, AddressOf(Memory { base: sp })`),
/// so counting it charged a pure MODELLING change to the A/A fingerprint as if
/// the pass had invented a load.
fn collect_memory_accesses<'a>(op: &'a Operand, out: &mut Vec<&'a Operand>) {
    match op {
        Operand::Memory { .. } => out.push(op),
        // `&x` — the location is not touched here.
        Operand::AddressOf(_) => {}
        Operand::ArrayLength(inner) => collect_memory_accesses(inner, out),
        Operand::ArrayAccess { array, index } => {
            collect_memory_accesses(array, out);
            collect_memory_accesses(index, out);
        }
        _ => {}
    }
}

#[derive(Default)]
struct Coverage {
    call: HashSet<u64>,
    branch: HashSet<u64>,
    loadstore: HashSet<u64>,
    arith: HashSet<u64>,
    literal: HashSet<u64>,
    /// Addresses hosting a Move whose source was replaced by propagation
    /// (redundant-load elimination rewrites `ldr` into an identity copy
    /// IN PLACE — the load's semantics survive as the propagated value).
    moved: HashSet<u64>,
    /// Addresses that are branch-to-next targets in the final IR (semantic
    /// no-ops; a decoded branch at such an address is accounted).
    fallthrough: HashSet<u64>,
}

/// Native addresses that are provably EXECUTED.
///
/// `decoded` is a linear sweep, so it contains data. Judging that data as
/// "uncovered semantics" is what made the census a list of jump-table and
/// padding addresses, so the walk marks what the machine can actually reach:
/// recursive descent from the entry over direct branch targets and
/// fallthrough, UNION the targets of every `Op::Switch` the recovery produced
/// (a case body is reachable ONLY through the indirect `jmp [table + idx*8]`,
/// so descent alone would classify all of them as data and HIDE their real
/// losses), UNION the addresses of those final-IR instructions that belong to a
/// block the CFG can reach from the entry (`live_instructions`; a Nop, or a dead
/// block's leftovers, is not evidence that the machine executes it).
///
/// Conservatism is one-directional on purpose. Over-reporting reachability
/// keeps a doubtful instruction in the census, which is a false positive at
/// worst; under-reporting would hide a real loss, so every unrecognised flow
/// keeps its fallthrough edge.
fn reachable_native_addresses(
    decoded: &[DecodedInsn],
    final_insns: &[Insn],
    base_rva: u64,
    is_arm32: bool,
    cfg: Option<&Cfg>,
) -> HashSet<u64> {
    let index: HashMap<u64, usize> = decoded
        .iter()
        .enumerate()
        .map(|(i, d)| (d.addr, i))
        .collect();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut stack: Vec<u64> = Vec::new();
    stack.push(base_rva);
    // Seeds come from the IR — but ONLY from code the CFG can reach from the
    // entry block.
    //
    // "If the recovery emitted an instruction there it believes in it" is false
    // for code the CFG has ALREADY decided is dead. The pipeline Nops most of
    // that (a Nop is exactly how it records an instruction it found INERT: a
    // dropped runtime thunk, an excised guard, an ARM32 `cmp`/`teq` whose
    // semantic is the `Compare` materialized at its CONSUMER), and a Nop is
    // not a seed. But a spelling the lifter models as a REAL op survives the
    // data it decodes over — the absolute-address table a compiler lays out
    // after every function's last terminator lifts to `Arith`/`Move` as well as
    // `Nop` — and seeding from one word of it restored reachability to the whole
    // table, which the census then judged as dropped semantics (the residual
    // `cmpseq`/`cmps`/`teqseq` cluster).
    //
    // A block no edge enters cannot be executed, so nothing in it may be a seed.
    // Real code stays reachable through its own control flow and through the
    // `Op::Switch` targets, which `Cfg::build` already models as edges.
    let live = cfg.map(|c| live_instructions(c, final_insns.len()));
    // Addresses where the FINAL IR ends a block with a terminator. The walk
    // below cannot tell `pop {r4, pc}` from `pop {r4}` from the MNEMONIC alone
    // (the oracle keeps only the first token), so it deliberately keeps the
    // fallthrough edge for the whole `pop`/`ldm`/`mov pc, lr` return family —
    // which is why a function's trailing alignment/padding or a tail-duplicated
    // body was judged as dropped semantics (the residual `cmps`/`cmpseq`/
    // `teqseq` cluster, ~7.5k insns). But the recovery ALREADY made that
    // decision: if the final IR hosts a terminator at this address, nothing
    // after it executes, so the linear sweep past it is not executed code and
    // judging it can only invent a loss. Consulting the IR here makes the
    // oracle agree with the CFG it is checking rather than guessing.
    let ir_terminates: HashSet<u64> = final_insns
        .iter()
        .filter(|i| {
            // ONLY the ops that provably do NOT fall through. `CondJump` is
            // deliberately absent: it has a real fallthrough arm, and treating
            // it as terminal would under-report reachability and HIDE losses.
            // The unconditional `Jump` and the direct unconditional `b` are
            // already covered by the decoded facts above.
            matches!(i.op, Op::Return | Op::Throw | Op::IndirectCall)
        })
        .map(|i| i.address)
        .collect();
    for (i, insn) in final_insns.iter().enumerate() {
        if matches!(insn.op, Op::Nop) {
            continue;
        }
        if live.as_ref().is_some_and(|l| !l[i]) {
            continue;
        }
        stack.push(insn.address);
        if let Op::Switch(targets) = &insn.op {
            for t in targets {
                if let Some(ti) = final_insns.get(*t) {
                    stack.push(ti.address);
                }
            }
        }
    }
    while let Some(addr) = stack.pop() {
        if !seen.insert(addr) {
            continue;
        }
        let Some(&i) = index.get(&addr) else { continue };
        let d = &decoded[i];
        let fall = addr + d.size as u64;
        if let Some((_, t)) = d.facts.cond_branch {
            stack.push(t);
            stack.push(fall);
            continue;
        }
        if let Some(t) = d.facts.branch_target {
            // Direct unconditional branch: the target, never the fallthrough.
            stack.push(t);
            continue;
        }
        // A branch with NO direct target is a return, or an indirect transfer
        // whose destinations this walk cannot enumerate. Only the mnemonics
        // that provably do not fall through terminate the walk; anything
        // unrecognised keeps its fallthrough, so a miss here can only ADD a
        // judged instruction, never remove one.
        //
        // ARM32 has REAL facts (`facts_arm32` sets `branch_target` for an
        // unconditional `b` and `cond_branch` for `b<cc>`/`cbz`/`cbnz`), so the
        // `branch_target` test above already stops the walk at an ordinary tail
        // branch. What it does NOT cover is the register-transfer family, which
        // is exactly where an ARM32 function ends: `bx lr` (the return idiom),
        // a predicated `bx<cc> lr`, and `pop {…, pc}` / `ldm …{pc}` / `mov pc, lr`
        // all carry no fact, and only the bare `bx` spelling is in the mnemonic
        // list below. This clause is the backstop for the `bx` spellings; the
        // `pop`/`ldm` forms stay conservative (the mnemonic alone cannot tell
        // `pop {r4, pc}` from `pop {r4}`, and guessing would HIDE real losses
        // after every ordinary `pop`). `split` is the LIFTER'S splitter, so a
        // predicated branch (which DOES fall through) is never mistaken for an
        // unconditional one and the two sides cannot disagree.
        let arm32_nonfallthrough = is_arm32 && {
            let (base, cond, _) = super::lift_arm32::split(&d.mnemonic);
            // `bx` covers `bx lr` (return) and `bx rN` (indirect transfer);
            // neither falls through. `b` is redundant with the facts above but
            // costs nothing and guards an encoding the decoder leaves untyped.
            base == "bx" || (base == "b" && cond.is_empty())
        };
        let terminator = matches!(
            d.mnemonic.as_str(),
            "ret" | "retf" | "iret" | "iretq" | "jmp" | "br" | "eret" | "bx"
        ) || arm32_nonfallthrough
            // The recovery's own CFG says this address does not fall through.
            || ir_terminates.contains(&addr);
        if !terminator {
            stack.push(fall);
        }
    }
    seen
}

/// Which instructions of a listing belong to a block the CFG can reach from the
/// entry block.
///
/// The reachability walk seeds itself from the instructions the recovery
/// produced, which is only sound if they can EXECUTE. The CFG is the recovery's
/// own answer to that question, so it — not the mere presence of an instruction
/// — decides which seeds are admissible. A forward walk over
/// `cfg.successors` from block 0 is the exact dual of `check_cfg`'s backwards
/// walk from the exits.
fn live_instructions(cfg: &Cfg, n_insns: usize) -> Vec<bool> {
    let mut block_reach = vec![false; cfg.blocks.len()];
    if !cfg.blocks.is_empty() {
        let mut stack = vec![0usize];
        while let Some(b) = stack.pop() {
            if std::mem::replace(&mut block_reach[b], true) {
                continue;
            }
            if let Some(succ) = cfg.successors.get(b) {
                for &s in succ {
                    stack.push(s);
                }
            }
        }
    }
    let mut live = vec![false; n_insns];
    for (bi, b) in cfg.blocks.iter().enumerate() {
        if block_reach[bi] {
            for i in b.start..b.end.min(n_insns) {
                live[i] = true;
            }
        }
    }
    live
}

/// Byte addresses inside the measured body that hold an INLINE JUMP TABLE
/// rather than code.
///
/// LLVM lays a function's switch jump table out in the CODE section, right
/// after the body's last terminating instruction, as a run of 4-byte RVAs.
/// Those bytes decode as perfectly plausible instructions, so the linear sweep
/// lifts them and the census judged them as dropped semantics: `GetTextAnchorPivot`
/// @0x3b3900 ends in a tail `jmp` at 0x3b3a42, then `nop`, then
/// `47 39 3b 00  51 39 3b 00  63 39 3b 00 …` — the RVAs 0x3b3947, 0x3b3951,
/// 0x3b3963 — and those bytes lift as five `cmp` at 0x3b3a5a…0x3b3a69, which
/// was the bulk of the remaining `cmp`/`test` clusters. (This is the same
/// artifact class as Session 27, one layer deeper: S27 handled padding and
/// whole-table regions by reachability, but the IR the lifter emitted for the
/// table re-seeded those addresses as "reached".)
///
/// The signature is exact and narrow: a 4-byte little-endian value that points
/// INTO this method's own body, immediately followed by ANOTHER such value.
/// A real instruction stream does not contain two consecutive 32-bit constants
/// that both happen to be addresses inside the function, so this cannot remove
/// code — it only recognises data. Every byte of a proven run is marked, so it
/// does not matter how the decoder happens to align the slots.
fn inline_table_data(body: &[u8], base_rva: u64) -> HashSet<u64> {
    let hi = base_rva + body.len() as u64;
    let in_body = |v: u32| (v as u64) >= base_rva && (v as u64) < hi;
    let mut out: HashSet<u64> = HashSet::new();
    let mut o = 0usize;
    while o + 8 <= body.len() {
        let v0 = u32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
        if in_body(v0) {
            let v1 = u32::from_le_bytes([body[o + 4], body[o + 5], body[o + 6], body[o + 7]]);
            if in_body(v1) {
                // A run: extend while the 4-byte values keep pointing in-body.
                let start = o;
                let mut end = o + 8;
                while end + 4 <= body.len() {
                    let v = u32::from_le_bytes([
                        body[end],
                        body[end + 1],
                        body[end + 2],
                        body[end + 3],
                    ]);
                    if in_body(v) {
                        end += 4;
                    } else {
                        break;
                    }
                }
                for a in start..end {
                    out.insert(base_rva + a as u64);
                }
                o = end;
                continue;
            }
        }
        o += 1;
    }
    out
}

/// Lane selections that address a lane their own register does not have.
///
/// A `SimdOp::Arrange` is a list of lane NUMBERS, so a lifter that computes a
/// wrong index does not produce a wrong-looking stub — it produces a wrong
/// VALUE, silently, and the register-granular model has no type to catch it
/// with. This is the check that makes such a bug a REPORTED fact: the bound is
/// the register's lane count at the operation's own lane width (4 DWORDs, 16
/// bytes, 8 WORDs, …), and a legal extract selects from all of them while
/// producing only one.
///
/// Deliberately returns the violations instead of asserting: a corpus run must
/// be able to report them without aborting, and the count belongs next to the
/// other corruption oracles.
pub fn out_of_range_lane_selections(insns: &[Insn]) -> Vec<(u64, String)> {
    let mut out = Vec::new();
    for insn in insns {
        if let Op::SimdLane(op) = &insn.op {
            if !op.source_lanes_in_range() {
                out.push((
                    insn.address,
                    format!(
                        "{op} selects a lane outside its {}-byte register",
                        op.elem_bytes()
                    ),
                ));
            }
        }
    }
    out
}

/// Whether the first lane violation has already been announced. See
/// [`record_out_of_range_lane_selections`].
static LANE_VIOLATION_ANNOUNCED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fold one method's [`out_of_range_lane_selections`] findings into the run
/// aggregates, and ANNOUNCE the first one unconditionally.
///
/// The aggregate alone would only ever be read under `ILR_VERIFY`, which is off
/// by default — but an out-of-range lane pick is a WRONG VALUE, and a wrong
/// value must not depend on a verbosity flag to become visible. The first
/// violation therefore prints once, on any run that has one, with the running
/// total so the line is clearly a summary and not the whole story; the full
/// count, and which methods, are in [`VerifyStats::report`].
pub fn record_out_of_range_lane_selections(rva: u64, violations: &[(u64, String)]) {
    if violations.is_empty() {
        return;
    }
    STATS
        .out_of_range_lane_methods
        .fetch_add(1, Ordering::Relaxed);
    let total = STATS
        .out_of_range_lane_sites
        .fetch_add(violations.len() as u64, Ordering::Relaxed)
        + violations.len() as u64;
    if !LANE_VIOLATION_ANNOUNCED.swap(true, Ordering::Relaxed) {
        let (addr, why) = &violations[0];
        // NOTE: explicit arguments, not implicit capture — `format_args!` cannot
        // capture from the surrounding scope when the format string came from a
        // macro, and this file's CRLF endings make `concat!` the tempting shape.
        eprintln!(
            "[lane] {:#x}: {} (at {:#x}) — {} such selection(s) so far; run with ILR_VERIFY=1 for the full report",
            rva, why, addr, total
        );
    }
}

/// Addresses covered by semantic IR ops in the final listing.
fn covered_addresses(insns: &[Insn]) -> Coverage {
    let mut c = Coverage::default();
    for insn in insns {
        let a = insn.address;
        match &insn.op {
            Op::Call
            | Op::CallVoid
            | Op::IndirectCall
            | Op::NewObj(_)
            | Op::Throw
            // An ObjC message send IS a call — same value, same destination,
            // same address. Leaving it out scored every recovered
            // `[recv sel:…]` as a dropped call.
            | Op::ObjcMsgSend { .. } => {
                c.call.insert(a);
            }
            Op::Jump(_) | Op::CondJump(_) | Op::Switch(_) | Op::Return => {
                c.branch.insert(a);
            }
            _ => {}
        }
        match &insn.op {
            Op::Arith(_)
            | Op::FloatArith(_)
            | Op::Compare(_)
            | Op::Not
            | Op::Negate
            | Op::Convert(_)
            | Op::MulAcc { .. }
            | Op::Clz
            | Op::Move
            | Op::LoadLength
            | Op::NewArr(_)
            // A value-PRODUCING instruction the lifter carried as an
            // intrinsic defines its destination from its real operands
            // (`Op::Simd`'s own contract: "a missing NAME, never a lost
            // value"). It is therefore REPRESENTED, just not modelled, and
            // the census must not score it as dropped semantics — that put
            // `shufps` 221, `cvtdq2ps` 260, `unpcklps` 101, `psrldq`, `movi`
            // and every NEON intrinsic at the top of the x64 work list even
            // though each one has IR at its own address with its real
            // operands. `Op::Select` (`csinc`/`csel`/`fsel`) and `Op::Box`
            // are the same shape. Where the lifter genuinely DROPS an
            // instruction (no IR emitted at all) it is still reported
            // uncovered, which is the signal this census exists to give.
            | Op::Simd(_)
            // A MODELLED lane operation (`Op::SimdLane`) is the same shape and
            // strictly more: its semantics are carried as data. Leaving it out
            // of this table would score every `shufps`/`paddd` that the lifter
            // now models EXACTLY as a dropped instruction — the census must not
            // punish a fix.
            | Op::SimdLane(_)
            | Op::Select(_)
            | Op::Box(_) => {
                c.arith.insert(a);
                if matches!(insn.op, Op::Move) {
                    c.moved.insert(a);
                }
            }
            _ => {}
        }
        for op in &insn.operands {
            let mut mems = Vec::new();
            collect_memory_operands(op, &mut mems);
            if !mems.is_empty() {
                c.loadstore.insert(a);
                break;
            }
            // A resolved FieldRef IS the load/store: `ldr x0,[x19,#off]`
            // that resolve.rs named as `Type.field` at the same address.
            if matches!(op, Operand::Field(_)) {
                c.loadstore.insert(a);
                break;
            }
        }
        // In-place call rewrites: `bl string_new` becomes
        // Move(dst, StringLiteral) at the SAME address; likewise method/
        // type materialization. The call site survived — it just changed
        // shape, so it still covers the Call class.
        if matches!(insn.op, Op::Move) {
            match insn.operands.get(1) {
                Some(Operand::StringLiteral { .. })
                | Some(Operand::Method(_))
                | Some(Operand::Type(_)) => {
                    c.call.insert(a);
                    c.literal.insert(a);
                }
                _ => {}
            }
        }
    }
    // Branch-to-next is a semantic no-op: a Jump/CondJump/Switch whose target
    // is the fallthrough address carries no control decision. The final IR
    // keeps targets as INDICES (post-cfg::build), so resolve via the listing.
    for i in 0..insns.len() {
        let Some(next) = insns.get(i + 1) else { break };
        let targets: Vec<usize> = match &insns[i].op {
            Op::Jump(t) | Op::CondJump(t) => vec![*t],
            Op::Switch(ts) => ts.clone(),
            _ => continue,
        };
        for t in targets {
            if let Some(ti) = insns.get(t) {
                if ti.address == next.address {
                    c.fallthrough.insert(next.address);
                    break;
                }
            }
        }
    }
    c
}

impl Coverage {
    fn covers(&self, class: InsnClass, addr: u64) -> bool {
        match class {
            InsnClass::Call => self.call.contains(&addr),
            InsnClass::Branch => {
                // BR xN is a decoded branch but lifts to IndirectCall — the
                // virtual-dispatch semantic survived, so a Call-op at the
                // same address fully accounts for it.
                self.branch.contains(&addr)
                    || self.fallthrough.contains(&addr)
                    || self.call.contains(&addr)
            }
            InsnClass::LoadStore => self.loadstore.contains(&addr) || self.moved.contains(&addr),
            InsnClass::Arith => self.arith.contains(&addr),
            InsnClass::Pseudo => true,
            InsnClass::Float => {
                self.arith.contains(&addr)
                    || self.loadstore.contains(&addr)
                    || self.moved.contains(&addr)
            }
            InsnClass::Other => {
                // Unknown mnemonics are only covered when the IR produced
                // *something* at that address.
                self.arith.contains(&addr)
                    || self.call.contains(&addr)
                    || self.loadstore.contains(&addr)
                    || self.branch.contains(&addr)
                    || self.literal.contains(&addr)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CFG invariants
// ---------------------------------------------------------------------------

fn check_cfg(cfg: &Cfg, insns: &[Insn], findings: &mut Vec<String>) {
    let nblocks = cfg.blocks.len();
    if cfg.successors.len() != nblocks {
        findings.push(format!(
            "cfg: successors ({}) != blocks ({})",
            cfg.successors.len(),
            nblocks
        ));
        return;
    }
    for (bi, b) in cfg.blocks.iter().enumerate() {
        if b.start > b.end || b.end > insns.len() {
            findings.push(format!(
                "cfg: block {bi} bounds {}..{} outside listing {}",
                b.start,
                b.end,
                insns.len()
            ));
            continue;
        }
        for &s in &cfg.successors[bi] {
            if s > nblocks {
                findings.push(format!(
                    "cfg: block {bi} dangling successor {s} > {nblocks}"
                ));
            }
        }
    }
    // Every block must reach an exit: a region with no path to Return/Throw/
    // an open exit edge is the "unrecovered loop / impossible trap" signature.
    let term_is_exit = |bi: usize| -> bool {
        cfg.blocks
            .get(bi)
            .and_then(|b| insns.get(b.end.checked_sub(1)?))
            .map(|i| matches!(i.op, Op::Return | Op::Throw | Op::IndirectCall))
            .unwrap_or(false)
    };
    let mut exits: Vec<usize> = Vec::new();
    for bi in 0..nblocks {
        let open_edges = cfg.successors[bi].iter().any(|&s| s >= nblocks);
        if term_is_exit(bi) || open_edges || cfg.successors[bi].is_empty() {
            exits.push(bi);
        }
    }
    let mut preds: HashMap<usize, Vec<usize>> = HashMap::new();
    for (bi, succs) in cfg.successors.iter().enumerate() {
        for &s in succs {
            if s < nblocks {
                preds.entry(s).or_default().push(bi);
            }
        }
    }
    let mut reach = HashSet::new();
    let mut stack: Vec<usize> = exits;
    while let Some(b) = stack.pop() {
        if reach.insert(b) {
            if let Some(ps) = preds.get(&b) {
                stack.extend(ps.iter().copied());
            }
        }
    }
    for bi in 0..nblocks {
        if !reach.contains(&bi) {
            findings.push(format!(
                "cfg: block {bi} cannot reach an exit (possible unrecovered loop/trap)"
            ));
            break; // one report per method is enough to triage
        }
    }
}

// ---------------------------------------------------------------------------
// A/A fingerprinting: raw lift vs final IR
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy, PartialEq)]
pub struct SemanticFingerprint {
    pub calls: usize,
    pub branches: usize,
    pub loads_stores: usize,
    pub compares: usize,
    pub arith: usize,
    pub literals: usize,
}

impl SemanticFingerprint {
    pub fn of(insns: &[Insn]) -> Self {
        let mut f = Self::default();
        // Memory/string operands that are the SOURCE of a `Move`: the loads
        // themselves. See the re-materialization dedup below.
        let move_sources: Vec<&Operand> = insns
            .iter()
            .filter(|i| matches!(i.op, Op::Move))
            .filter_map(|i| i.operands.get(1))
            .filter(|o| matches!(o, Operand::Memory { .. }))
            .collect();
        let move_literals: Vec<&Operand> = insns
            .iter()
            .filter(|i| matches!(i.op, Op::Move))
            .filter_map(|i| i.operands.get(1))
            .filter(|o| matches!(o, Operand::StringLiteral { .. }))
            .collect();
        // String literals are CONSTANTS, not accesses. `resolve` turns ONE
        // metadata-blob read into a named literal and constant propagation then
        // pushes it into every use site, so a single raw load can surface at
        // three call arguments and would be charged three times. Counting the
        // DISTINCT literal values instead restores the invariant: one raw read
        // = one named constant (verified on 0x40E0F0: one load → three `"N"`
        // arguments). The raw lift never contains a `StringLiteral`, so this can
        // only ever lower the post count — it cannot hide synthesis.
        let mut lit_seen: Vec<&Operand> = Vec::new();
        for insn in insns {
            match &insn.op {
                Op::Call | Op::CallVoid | Op::IndirectCall | Op::NewObj(_) | Op::Throw => {
                    f.calls += 1
                }
                Op::Jump(_) | Op::CondJump(_) | Op::Switch(_) | Op::Return => f.branches += 1,
                _ => {}
            }
            if matches!(insn.op, Op::Compare(_)) {
                f.compares += 1;
            }
            if matches!(
                insn.op,
                Op::Arith(_)
                    | Op::FloatArith(_)
                    | Op::MulAcc { .. }
                    | Op::Convert(_)
                    | Op::Not
                    | Op::Negate
                    | Op::Clz
            ) {
                f.arith += 1;
            }
            // An operand that repeats a `Move` source VERBATIM is that same
            // value re-materialized, not a second access:
            // `copy_and_constant_propagation` substitutes a single-definition
            // local at its use, and the defining `Move` survives whenever the
            // value is read more than once (verified on 0x1794D0: the `Return`
            // carries the same `StringLiteral` its defining `Move` still stores,
            // so ONE named read was charged twice). The exception is the defining
            // `Move` itself — its source operand IS the access. The rule is
            // applied to BOTH fingerprints, so a raw lift that repeats a load
            // across two instructions is deduped on both sides and the
            // comparison stays fair while genuine synthesis (a new operand that
            // no `Move` defines) still grows the count.
            let is_move = matches!(insn.op, Op::Move);
            let move_dest = insn.operands.get(1);
            let defines = |o: &Operand| is_move && move_dest == Some(o);
            // The DESTINATION slot is a write, never a re-materialized read:
            // a store `[abs+K] = r` whose address an earlier `Move r, [abs+K]`
            // also loaded is a genuine SECOND access, so it must not be
            // deduped against that load.
            let dest_pos = Insn::dest_position(&insn.op);
            let mut has_mem = false;
            for (idx, op) in insn.operands.iter().enumerate() {
                let is_dest = dest_pos == Some(idx);
                let mut mems = Vec::new();
                collect_memory_accesses(op, &mut mems);
                if mems
                    .iter()
                    .any(|m| is_dest || defines(*m) || !move_sources.iter().any(|s| *s == *m))
                {
                    has_mem = true;
                }
                if matches!(op, Operand::StringLiteral { .. })
                    && (is_dest || defines(op) || !move_literals.iter().any(|s| *s == op))
                    && !lit_seen.iter().any(|s| *s == op)
                {
                    lit_seen.push(op);
                }
            }
            if has_mem {
                f.loads_stores += 1;
            }
        }
        f.literals = lit_seen.len();
        f
    }

    /// Fingerprint of `insns`, ignoring every instruction whose address lies
    /// OUTSIDE the method's declared body `[lo, hi)`.
    ///
    /// The tail-merge splicer legitimately appends a SHARED, out-of-body
    /// epilogue: the arm64 compiler hoists a method's final call+return PAST
    /// the metadata-declared size into a block several other methods branch
    /// into, and the splicer lifts that block into this listing so the CFG
    /// keeps its edge into the call block (without it the call's block is
    /// unreachable, SSA renames it with entry versions and DCE deletes every
    /// argument feed — the "missing FP math" class).
    ///
    /// That spliced code is real — this method executes it — but it was never
    /// part of the raw lift the A/A baseline fingerprints, so charging it to
    /// the comparison reports a faithful recovery as unexpected synthesis.
    /// Verified on `0x103ac` (iOS): raw lift 1 call / 1 compare / 0 arith; final
    /// 2 / 2 / 2, ALL of the growth coming from the spliced tail at `0x264b728`,
    /// which carries the shared `UpdateBounds()`, two `FloatArith(Add)`s and a
    /// `Compare`. Filtering the out-of-body insns makes the comparison
    /// apples-to-apples while still policing growth inside the body.
    pub fn of_inside(insns: &[Insn], lo: u64, hi: u64) -> Self {
        let kept: Vec<Insn> = insns
            .iter()
            .filter(|i| i.address >= lo && i.address < hi)
            .cloned()
            .collect();
        Self::of(&kept)
    }

    /// Human-readable drift report between pre-pass and final fingerprints.
    /// Only GROWTH is flagged (synthesis); loss is policed by the coverage
    /// oracle via the ledger.
    ///
    /// `tail_calls` is how many tail transfers this method converted into a
    /// `Call` + `Return` (the caller reads it off the excusal ledger). Each one
    /// adds exactly ONE control-transfer event (the `Jump` it replaces was one)
    /// and may re-materialize one memory operand in its synthesized argument
    /// dump, so the counters they move are allowed to grow by that much — and by
    /// no more, so a pass that fabricates two calls where the ledger explains
    /// one is still reported.
    ///
    /// Calls and branches are compared as ONE SUM, because two passes
    /// legitimately MOVE a transfer between the two buckets without adding one:
    /// `recover_switch_tables` rewrites an `IndirectCall` dispatch into a
    /// `Switch` (a call becomes a branch), and a tail call rewrites a `Jump`
    /// into a `Call` + `Return` (a branch becomes a call AND a branch). Bucket-
    /// wise comparison reported every switch recovery as `branches GREW` — the
    /// largest remaining arm64 A/A class — while the total never moved.
    ///
    /// `loads_stores` and `literals` are compared as a SUM. A metadata string
    /// reference is a MEMORY read that got NAMED: `resolve` replaces the
    /// `Memory { base: abs, addend }` operand with a `StringLiteral`, so the
    /// read moves from one bucket to the other and only their sum is
    /// invariant. Neither bucket can grow any other way — `StringLiteral`
    /// operands are only ever produced by `resolve`, never fabricated.
    ///
    /// Both counters are already de-duplicated in [`SemanticFingerprint::of`],
    /// so the residual growth this flags is a NEW operand no `Move` defines —
    /// genuine synthesis — rather than propagation moving one named value
    /// between its defining `Move` and its use sites. The measured effect on
    /// the v24.5 corpus: 4,210 drifts at the start of the session, 30 after the
    /// tail-call/memory-naming fixes, and 9 once re-materialization and literal
    /// fan-out were counted correctly.
    pub fn drift(&self, final_fp: &SemanticFingerprint, tail_calls: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut cmp = |name: &str, pre: usize, post: usize, allow: usize| {
            if post > pre + allow {
                out.push(format!(
                    "A/A: {name} GREW {pre}→{post} (unexpected synthesis)"
                ));
            }
        };
        cmp(
            "memory refs",
            self.loads_stores + self.literals,
            final_fp.loads_stores + final_fp.literals,
            tail_calls,
        );
        cmp("compares", self.compares, final_fp.compares, 0);
        cmp("arith", self.arith, final_fp.arith, 0);
        // ONE sum for control transfers; see the doc comment above (this has to
        // come AFTER the closure's last use — `cmp` borrows `out` mutably).
        // Verified on 0x1e99d8 (iOS): the raw `IndirectCall` becomes a `Switch`,
        // calls 1→0 and branches 8→9, sum 9→9, while the A/A used to flag
        // `branches GREW`.
        let pre_ct = self.calls + self.branches;
        let post_ct = final_fp.calls + final_fp.branches;
        if post_ct > pre_ct + tail_calls {
            out.push(format!(
                "A/A: control transfers GREW {pre_ct}→{post_ct} (calls {}→{}, branches {}→{}) (unexpected synthesis)",
                self.calls, final_fp.calls, self.branches, final_fp.branches
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// The verifier proper
// ---------------------------------------------------------------------------

/// How badly a method's recovery disagrees with its own native bytes.
///
/// The `clean` flag is all-or-nothing, and the most COMMON finding by far — an
/// `uncovered` arithmetic or flag-test instruction that a later pass folded, or
/// that the ledger explains (a `test` feeding a branch, a nop, a runtime call
/// dropped by DCE) — is benign. Lumping that in with a wrong branch target made
/// the aggregate useless for the one question worth asking: how many methods are
/// actually WRONG. It reported 12,585 of 12,615 flagged, which is consistent
/// with both "the pipeline is broken everywhere" and "the pipeline is fine and
/// the oracle is strict". The buckets below separate the two, using only kinds
/// of evidence the oracle can actually distinguish.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// No finding at all.
    Clean,
    /// The IR PROVABLY disagrees with the bytes: a semantic diff (inverted
    /// condition, wrong branch or call target, wrong memory offset, wrong
    /// arithmetic kind) or a structural CFG violation. Every one of these is a
    /// method whose recovered control flow or arithmetic is demonstrably not
    /// what the native code does.
    WrongLogic,
    /// Not provably wrong, but the output is corrupted in a way a reader would
    /// have to notice: a read of a local that is never defined, a branch with no
    /// real comparison at its site, a call the resolver could not name, a raw
    /// register SSA never renamed, or control flow that appeared where the raw
    /// lift had none. Each can hide a wrong value.
    Suspect,
    /// Every finding is of the kind a CORRECT pipeline produces: instructions
    /// the optimizer legitimately absorbed.
    CoverageOnly,
}

#[derive(Debug, Default)]
pub struct VerifyReport {
    pub method_rva: u64,
    pub decoded: usize,
    pub uncovered: Vec<DecodedInsn>,
    /// Decoded instructions inside the measured body that the reachability walk
    /// could NOT prove are code (padding, inline jump tables). Excluded from
    /// `uncovered` and from the verdict — see [`reachable_native_addresses`].
    pub unreached: Vec<DecodedInsn>,
    pub cfg_findings: Vec<String>,
    pub aa_findings: Vec<String>,
    /// Decoded conditional branches whose final IR has no REAL compare at
    /// the branch site (the `Compare(kind, Unknown, 0)` fallback →
    /// `if (? != 0)`: plausible text, wrong logic).
    pub missing_conditions: Vec<u64>,
    /// Corruption oracle #1: v-reg indices READ by the final listing but
    /// never defined in it and not a seeded entry value (this/param/retbuf).
    /// Renders as `v41 = v10;` with `v10` never written.
    pub undefined_locals: Vec<u32>,
    /// Parallel to `undefined_locals`: the SLOT ORIGIN of each (`g3`, `vf0`,
    /// `stack_12`, …). Printed in the per-method detail so a residual can be
    /// diagnosed from one method's line without a binary-wide cluster pass.
    pub undefined_local_origins: Vec<String>,
    /// Corruption oracle #2: (address, reason) for calls the resolver could
    /// not name (renders `fn_4a58bac(...)` thunks, `__unknown` args,
    /// `v807..ctor()` on unresolved ctors).
    pub unresolved_calls: Vec<(u64, String)>,
    /// Semantic-diff findings: the final IR at a native instruction's address
    /// DISAGREES with what the bytes actually say (inverted condition, wrong
    /// branch/call target, wrong memory offset, wrong arithmetic kind). Each
    /// entry is (address, description).
    pub semantic_diffs: Vec<(u64, String)>,
    /// Oracle #3: `Register(..)` operands still in the FINAL listing. SSA
    /// guarantees every register use becomes a versioned Local; a survivor
    /// means the block was never renamed (orphaned branch target, broken
    /// dominator tree) and the renderer emits `__unknown` for it.
    pub raw_registers: Vec<(u64, String)>,
    pub decode_failures: u64,
    /// Structural conformance findings (see `conform`): named classes where
    /// the IR performs a DIFFERENT SET of operations than the native bytes.
    /// Deliberately NOT part of `is_clean`: these describe a different (and
    /// larger) question than the per-site oracle, and folding them into the
    /// clean/flagged aggregate would blur both.
    pub conform_findings: Vec<(super::conform::DefectClass, u64, String)>,
}

impl VerifyReport {
    pub fn is_clean(&self) -> bool {
        self.uncovered.is_empty()
            && self.cfg_findings.is_empty()
            && self.aa_findings.is_empty()
            && self.missing_conditions.is_empty()
            && self.undefined_locals.is_empty()
            && self.unresolved_calls.is_empty()
            && self.semantic_diffs.is_empty()
            && self.raw_registers.is_empty()
    }

    /// One line naming WHY this method is not clean, in the same vocabulary the
    /// census summary uses. `None` when clean. Used for the in-output
    /// annotation: the renderer has no room for the full [`summarize`] dump, and
    /// "3 undefined local, 1 unresolved call" is exactly the context a reader
    /// needs to distrust the right lines.
    ///
    /// [`summarize`]: Self::summarize
    pub fn short_reason(&self) -> Option<String> {
        if self.is_clean() {
            return None;
        }
        let mut parts: Vec<String> = Vec::new();
        if !self.semantic_diffs.is_empty() {
            parts.push(format!("{} semantic diff", self.semantic_diffs.len()));
        }
        if !self.cfg_findings.is_empty() {
            parts.push(format!("{} cfg violation", self.cfg_findings.len()));
        }
        if !self.missing_conditions.is_empty() {
            parts.push(format!(
                "{} cond-branch without a real compare",
                self.missing_conditions.len()
            ));
        }
        if !self.undefined_locals.is_empty() {
            parts.push(format!("{} undefined local", self.undefined_locals.len()));
        }
        if !self.unresolved_calls.is_empty() {
            parts.push(format!("{} unresolved call", self.unresolved_calls.len()));
        }
        if !self.raw_registers.is_empty() {
            parts.push(format!("{} raw register", self.raw_registers.len()));
        }
        if !self.aa_findings.is_empty() {
            parts.push(format!("{} A/A drift", self.aa_findings.len()));
        }
        if !self.uncovered.is_empty() {
            parts.push(format!("{} uncovered insn", self.uncovered.len()));
        }
        Some(parts.join(", "))
    }

    /// The severity of this method's findings. See [`Verdict`] for why the
    /// `clean` flag alone cannot answer "is this method wrong".
    pub fn verdict(&self) -> Verdict {
        if self.is_clean() {
            return Verdict::Clean;
        }
        // A direct contradiction with the bytes, or a CFG the bytes cannot
        // produce: proven wrong.
        if !self.semantic_diffs.is_empty() || !self.cfg_findings.is_empty() {
            return Verdict::WrongLogic;
        }
        // Corruption-shaped, but not a proven contradiction.
        if !self.missing_conditions.is_empty()
            || !self.undefined_locals.is_empty()
            || !self.unresolved_calls.is_empty()
            || !self.raw_registers.is_empty()
            || !self.aa_findings.is_empty()
        {
            return Verdict::Suspect;
        }
        // Only `uncovered` remains — instructions the optimizer absorbed.
        Verdict::CoverageOnly
    }

    pub fn summarize(&self) -> String {
        if self.is_clean() {
            let mut s = format!(
                "[verify] {:#x}: clean ({} insns)",
                self.method_rva, self.decoded
            );
            // Conform findings are deliberately NOT part of `is_clean`, so a
            // method with only conform findings still reports "clean" — but the
            // per-method detail belongs here, not in a silenced field.
            for (class, site, what) in self.conform_findings.iter().take(4) {
                s.push_str(&format!(
                    "\n[verify]   conform {class:?} @{site:#x}: {what}"
                ));
            }
            return s;
        }
        let mut s = format!(
            "[verify] {:#x}: {} uncovered / {} cfg / {} aa / {} no-cond / {} undef-locals / {} unresolved-calls / {} sem-diff / {} raw-regs of {} decoded",
            self.method_rva,
            self.uncovered.len(),
            self.cfg_findings.len(),
            self.aa_findings.len(),
            self.missing_conditions.len(),
            self.undefined_locals.len(),
            self.unresolved_calls.len(),
            self.semantic_diffs.len(),
            self.raw_registers.len(),
            self.decoded
        );
        for (a, why) in self.raw_registers.iter().take(4) {
            s.push_str(&format!("\n[verify]   raw register @{a:#x}: {why}"));
        }
        for (a, why) in self.semantic_diffs.iter().take(6) {
            s.push_str(&format!("\n[verify]   sem-diff @{a:#x}: {why}"));
        }
        for (a, why) in self.unresolved_calls.iter().take(4) {
            s.push_str(&format!("\n[verify]   unresolved call @{a:#x}: {why}"));
        }
        if !self.undefined_locals.is_empty() {
            let names: Vec<String> = self
                .undefined_locals
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let o = self
                        .undefined_local_origins
                        .get(i)
                        .map(String::as_str)
                        .unwrap_or("?");
                    format!("v{v}({o})")
                })
                .collect();
            s.push_str(&format!(
                "\n[verify]   undefined locals (used, never defined): {}",
                names.join(", ")
            ));
        }
        for a in self.missing_conditions.iter().take(4) {
            s.push_str(&format!(
                "\n[verify]   conditional branch @{a:#x} has no real compare (wrong-logic class)"
            ));
        }
        for u in self.uncovered.iter().take(8) {
            s.push_str(&format!(
                "\n[verify]   uncovered @{:#x}: {} ({})",
                u.addr,
                u.mnemonic,
                u.class.name()
            ));
        }
        for f in self.cfg_findings.iter().take(4) {
            s.push_str(&format!("\n[verify]   {f}"));
        }
        for f in self.aa_findings.iter().take(4) {
            s.push_str(&format!("\n[verify]   {f}"));
        }
        // Structural conformance is deliberately NOT part of `is_clean`, but
        // it is still per-method data: surfacing it here makes the field read
        // (and the detail visible) without blurring the clean/flagged aggregate.
        for (class, site, what) in self.conform_findings.iter().take(4) {
            s.push_str(&format!(
                "\n[verify]   conform {class:?} @{site:#x}: {what}"
            ));
        }
        s
    }
}

/// Verify a finished method body. `pre_pass_insns` is the raw lift output
/// (before CFG/SSA/passes); `final_insns` + `cfg` are the pipeline result.
/// `body_bytes` is the raw method bytes and `base_rva` their RVA. `is_arm64` /
/// `is_x64` / `is_arm32` select the ground-truth decoder; `thumb` marks
/// ARM32 Thumb mode.
#[allow(clippy::too_many_arguments)]
/// Operand-swapped twin of a compare kind (constant folding may canonicalize
/// `5 > x` into `x < 5` — same semantics, different kind). A real inversion
/// (eq↔ne, lt↔ge …) is NOT a twin and stays flagged.
fn swapped_kind(k: CompareKind) -> CompareKind {
    match k {
        CompareKind::LessThan => CompareKind::GreaterThan,
        CompareKind::GreaterThan => CompareKind::LessThan,
        CompareKind::LessOrEqual => CompareKind::GreaterOrEqual,
        CompareKind::GreaterOrEqual => CompareKind::LessOrEqual,
        other => other,
    }
}

/// Human name for an arm64 condition code (yaxpeax's own spelling).
fn cond_name(code: u8) -> &'static str {
    match code & 0xF {
        0 => "eq",
        1 => "ne",
        2 => "hs",
        3 => "lo",
        4 => "mi",
        5 => "pl",
        6 => "vs",
        7 => "vc",
        8 => "hi",
        9 => "ls",
        10 => "ge",
        11 => "lt",
        12 => "gt",
        13 => "le",
        _ => "al",
    }
}

/// Per-address SEMANTIC comparison between the decoded native instruction
/// and the final IR group at the same address. Coverage (oracle above) only
/// proves an op EXISTS; this proves it MEANS the same thing:
///
/// 1. condition polarity — native cond code vs the IR Compare kind feeding
///    the CondJump (the `if (x == 0)` that should be `if (x != 0)` class);
/// 2. taken-edge branch target — the native PCOffset vs where the IR's taken
///    successor eventually lands;
/// 3. call target — decoded BL PCOffset vs the IR Call's immediate target
///    (a resolved Method operand is accepted as-is);
/// 4. memory addend — decoded load/store offset vs the IR Memory addend;
/// 5. arithmetic kind — native add/sub/and/orr/eor/mul/div vs the IR Arith
///    kind (the `add lifted as sub` class);
/// 6. TBZ bit number — the And mask must be exactly `1 << bit`.
///
/// Legitimate rewrites are accepted, never flagged: operand-swapped compare
/// kinds, Memory→Field rewrites, sp-based stack-slot rebasing, post-index
/// addressing (load + separate add), and ledgered excisions (checked by the
/// caller).
///
/// Architecture-independent: it reads `NativeFacts`, which `facts_arm64`,
/// `facts_arm32` and `facts_x86` all populate into one canonical space (in
/// particular the condition code is the same 4-bit numbering A32 and A64 share,/// which is exactly what `lift::cond_to_kind` speaks). `frame_regs` are the
/// architecture's FRAME registers — the stack pointer AND the frame pointer it
/// is aliased to. `stack_analyzer` rebases every offset-addressed access off
/// either one into a `stack_N` slot, and the rebasing legitimately CHANGES the
/// addend (it folds in the frame adjustment), so their addends are not
/// comparable and must not be flagged: sp 31 / fp 29 (A64), sp 13 / fp 11 (A32),
/// RSP / RBP (x86/x64). Sparing only the stack pointer left every
/// frame-pointer-relative local (`[rbp+0x14]` → `[stack_12+4]`) reported as a
/// mismatch the moment a `sub rsp, N` moved the frame.
/// Mnemonics a lifter FOLDS into the instruction that consumes their effect,
/// so the folded instruction's own address legitimately has no IR entry.
///
/// The flag setters are the mass case: `cmp`/`test` (x86) and `cmp`/`cmn`/`tst`
/// (arm64) do not become IR instructions — the lifter records a `PendingCmp`
/// and emits the `Compare` + `CondJump` pair at the address of the branch that
/// READS those flags. A branch that targets the flag setter therefore has no
/// exact landing site, which is why `cfg::resolve_target` resolves it to the
/// first instruction at or after it. The no-effect encodings are the trivial
/// other members of the same set.
///
/// This list is deliberately narrow. An instruction that is merely DROPPED
/// (a `mulss`, a `shufps`, an `xchg`) is not in it: branching past one of those
/// would skip real work, and such a landing must stay flagged.
fn is_folded_into_successor(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        // Synthetic token the ARM32 decoder assigns to a VFP word it cannot
        // decode but the lifter ABSORBS into its consumer (a `vcmp`/`vcmpe`
        // flag setter or a `vmrs`/`vmsr` flag transfer — see
        // `lift_arm32::vfp_absorbed_word`). It emits no IR, so entering the
        // successor IS entering it, exactly like the x86 SSE compares below.
        "vfp"
        // x86 flag setters — absorbed into the consuming `jcc` as `PendingCmp`
        | "cmp" | "test"
        // x86 bit TEST — absorbed into the consuming `jcc`/`cmovcc` via
        // `bit_scratch_if_carry`, exactly like `cmp`. The modify forms
        // (`bts`/`btr`/`btc`) are deliberately absent: they push their own IR
        // (an `Or`/`And`/`Xor`), so they have a real landing site and must not
        // be treated as folded.
        | "bt"
        // arm64 flag setters — same absorption via `materialize_cond`
        | "cmn" | "tst"
        // x86 SSE scalar FP compares — flag setters exactly like `cmp`/`test`.
        // The lifter records a `pending_fp` and emits the `Compare` at the
        // CONSUMER's address (`materialize_pending`), so the FP compare's own
        // address has no IR entry. Omitting them flagged every branch that
        // landed on the `ucomiss`/`comiss` that OPENED a compare/`jp`/`je`
        // chain — the whole `sem-diff branch target mismatch` class on x64.
        // They set EFLAGS only, so entering the consumer is entering them.
        | "comiss" | "comisd" | "ucomiss" | "ucomisd"
        // no observable effect on managed logic
        | "nop" | "endbr64" | "endbr32" | "hint" | "bti"
    )
}

/// Accept the ABSORBED landing `cfg::resolve_target` is contracted to produce:
/// a native target no IR instruction starts at is resolved to the FIRST IR
/// instruction at or after it.
///
/// That is exact — not approximate — when every decoded instruction skipped is
/// one the lifter folded into its successor (`is_folded_into_successor`): a
/// folded instruction has no observable effect of its own, so entering the
/// successor IS entering it. It is refused when any skipped instruction is
/// merely DROPPED rather than folded, because branching past that one would
/// skip real work.
///
/// Without this the oracle demanded `address == native_target` and reported
/// every restored edge as a mismatch: 6,742 of the 6,929 wrong-logic findings
/// on the v24.5 corpus were branch-target mismatches of exactly this form,
/// which made the wrong-logic method count barely move for the fix that
/// repaired them.
fn absorbed_landing(
    decoded: &[DecodedInsn],
    final_insns: &[Insn],
    native_target: u64,
    landing: u64,
) -> bool {
    if landing <= native_target {
        return false;
    }
    // `landing` must really be the FIRST IR instruction at or after the target
    // — otherwise the resolver picked something else and this is not the
    // contracted shape.
    let first_ge = final_insns
        .iter()
        .map(|i| i.address)
        .filter(|&a| a >= native_target)
        .min();
    if first_ge != Some(landing) {
        return false;
    }
    // Walk the native stream from the target: every instruction must be a
    // folded one, and the run must reach `landing` EXACTLY. Byte-contiguity is
    // the part that matters — a byte that failed to decode, or an instruction
    // missing from `decoded` for any other reason, breaks the chain and the
    // landing is refused. (Without this, an undecodable byte inside the range
    // would be stepped over silently, which is the one way this acceptance
    // could hide a real mismatch.)
    let mut cursor = native_target;
    loop {
        let Some(d) = decoded.iter().find(|d| d.addr == cursor) else {
            return false;
        };
        if !is_folded_into_successor(&d.mnemonic) || d.size == 0 {
            return false;
        }
        cursor += d.size as u64;
        if cursor >= landing {
            break;
        }
    }
    cursor == landing
}

/// Is this operand a local the stack analyzer created as a FRAME CELL
/// (`slot_origin` `stack_N`)? A `Memory` based on one has been rebased off a
/// frame alias, so its addend is not the native displacement — the same reason
/// sp/fp-relative accesses are not compared. See the memory-offset arm.
fn is_stack_cell_local(table: Option<&super::locals::LocalTable>, op: &Operand) -> bool {
    match op {
        Operand::Local(v) => table
            .and_then(|t| t.index_of.get(v).and_then(|&li| t.infos.get(li)))
            .is_some_and(|i| i.slot_origin.starts_with("stack_")),
        _ => false,
    }
}

fn semantic_diff(
    decoded: &[DecodedInsn],
    final_insns: &[Insn],
    finals_by_addr: &HashMap<u64, Vec<&Insn>>,
    cfg: Option<&Cfg>,
    frame_regs: &[u16],
    table_data: &HashSet<u64>,
    table: Option<&super::locals::LocalTable>,
    base_rva: u64,
    body_hi: u64,
    out: &mut Vec<(u64, String)>,
) {
    let Some(cfg) = cfg else { return };
    for d in decoded {
        // Inline jump tables are DATA, not code. The lifter decodes the whole
        // method body, so a 4-byte table entry that happens to encode a `jcc`
        // is lifted into a real CondJump — and then the native target it reads
        // out of the table (which is not a branch at all) mismatches. The
        // coverage census already skips these (see `inline_table_data`); the
        // semantic oracle must speak the same language about data-vs-code or
        // every dense table becomes a wrong-logic cluster.
        if table_data.contains(&d.addr) {
            continue;
        }
        let insns_at = match finals_by_addr.get(&d.addr) {
            Some(v) if !v.is_empty() => v,
            _ => continue, // absence is the coverage oracle's finding
        };
        // 0. TBZ/TBNZ bit number. Must run BEFORE the cond_branch gate below:
        // tbz/tbnz also set cond_branch, so the gate's `continue` skipped
        // this check on every site it was meant to guard.
        if let Some(bit) = d.facts.tbz_bit {
            let want = 1i64 << bit;
            let masks: Vec<i64> = insns_at
                .iter()
                .filter(|i| matches!(i.op, Op::Arith(ArithKind::And)))
                .filter_map(|i| {
                    i.operands.iter().find_map(|o| match o {
                        Operand::Immediate(m) => Some(*m),
                        _ => None,
                    })
                })
                .collect();
            // ABSENCE is legitimate and must never be reported: sign-test
            // sugar rewrites (x & 0x80000000) != 0 into x < 0, and DCE can
            // drop the mask entirely. Only a PRESENT-but-wrong mask proves
            // the bit index was misread.
            if !masks.is_empty() && !masks.contains(&want) {
                out.push((
                    d.addr,
                    format!("tbz bit {bit}: And mask {masks:?}, want {want}"),
                ));
            }
        }
        // 1+2. conditional branch: polarity + taken target
        if let Some((code, native_target)) = d.facts.cond_branch {
            let Some(cj) = insns_at.iter().find(|i| matches!(i.op, Op::CondJump(_))) else {
                continue; // "no real compare" oracle owns the lost-branch case
            };
            let expected = super::lift::cond_to_kind(code);
            let cmps: Vec<&Insn> = insns_at
                .iter()
                .filter(|i| matches!(i.op, Op::Compare(_)))
                .copied()
                .collect();
            // A lost pending-compare (Compare(kind, Unknown, 0)) is already
            // reported by missing_conditions — don't double-flag its kind.
            let has_unknown_cmp = cmps
                .iter()
                .any(|i| matches!(i.operands.get(1), Some(Operand::Unknown)));
            if !cmps.is_empty() && !has_unknown_cmp {
                let ok = cmps.iter().any(|i| match i.op {
                    Op::Compare(k) => k == expected || k == swapped_kind(expected),
                    _ => false,
                });
                // `bt x, n; jcc` is lowered by the x86 lifter into a SINGLE-BIT
                // test, not the carry kind the raw `jae`/`jb` code names:
                //   t = (x >> n) & 1;  if (t == 0) …   // CF == 0, i.e. `jae`
                //   t = (x >> n) & 1;  if (t != 0) …   // CF != 0, i.e. `jb`
                // so the Compare is Equal/NotEqual where `cond_to_kind` expects
                // GreaterOrEqual/LessThan. The tested VALUE is exact —
                // `bit_scratch_if_carry` shifts by the `bt` bit index and masks
                // one bit — so accept the shape rather than flag a correct
                // branch. Restricted to the CF-testing codes (2 = hs/ae, 3 =
                // lo/b): only the `bt` lowering produces this, and the polarity
                // is still checked below via the taken target.
                let bit_test_ok = (code == 2 || code == 3)
                    && cmps.iter().any(|i| match i.op {
                        Op::Compare(k) => {
                            matches!(k, CompareKind::Equal | CompareKind::NotEqual)
                                && matches!(i.operands.get(1), Some(Operand::Local(v))
                                    if insns_at.iter().any(|d| {
                                        matches!(d.op, Op::Arith(ArithKind::And))
                                            && matches!(d.operands.first(), Some(Operand::Local(dv)) if dv == v)
                                            && matches!(d.operands.get(2), Some(Operand::Immediate(1)))
                                    }))
                        }
                        _ => false,
                    });
                if !ok && !bit_test_ok {
                    let got: Vec<String> = cmps
                        .iter()
                        .filter_map(|i| match i.op {
                            Op::Compare(k) => Some(format!("{k:?}")),
                            _ => None,
                        })
                        .collect();
                    out.push((
                        d.addr,
                        format!(
                            "condition mismatch: native b.{} (expect {:?}), IR Compare {}",
                            cond_name(code),
                            expected,
                            got.join("/")
                        ),
                    ));
                    continue; // polarity already wrong; target check would be noise
                }
            }
            if let Op::CondJump(tidx) = &cj.op {
                // After Cfg::build the CondJump operand is the target
                // instruction INDEX (RVA→index rewrite). The taken edge lands
                // exactly there — a mid-block landing (e.g. on a Return inside
                // a bigger block) is normal, so compare the target ADDRESS of
                // the landing site, not the containing block's terminator.
                // Chase a short chain of jumps-to-jump (tail-merged re-tests,
                // branch-to-branch) before giving up.
                // A native target OUTSIDE the method body cannot have an IR
                // instruction: the IR is a closed listing over the body, so
                // `cfg` resolves such an edge to the exit sentinel. IL2CPP
                // emits exactly this for a shared out-of-body throw handler
                // (the null-check on a field accessor: `test rax,rax; je
                // <runtime throw>`), and managed C# re-introduces the check
                // implicitly, so nothing is lost. Only IN-BODY targets are
                // landings this oracle can judge.
                let in_body = native_target >= base_rva && native_target < body_hi;
                let mut ok = !in_body;
                let mut cur = Some(*tidx);
                for _ in 0..4 {
                    if !in_body {
                        break;
                    }
                    let Some(ti) = cur else { break };
                    let Some(ins) = final_insns.get(ti) else {
                        break;
                    };
                    // Check the CURRENT landing FIRST, before following any
                    // jump. The native taken target is very often an
                    // unconditional `jmp` to a shared exit (tail-merge), so
                    // the IR landing IS a `Jump` whose address is exactly the
                    // native target — following it before comparing made the
                    // chase compare the DEEPER address and flagged a correct
                    // edge. Following is only for the case where the landing
                    // is a jump-to-somewhere-else (branch-to-branch retarget).
                    if ins.address == native_target
                        || absorbed_landing(decoded, final_insns, native_target, ins.address)
                    {
                        ok = true;
                        break;
                    }
                    match &ins.op {
                        Op::Jump(t2) => cur = Some(*t2),
                        _ => break,
                    }
                }
                if !ok {
                    // LEGITIMATE INVERSION: a decompiler renders
                    // if-without-else by flipping the branch (taken↔fall),
                    // which pairs with the swapped compare polarity already
                    // accepted above. That is the same logic in a different
                    // layout — only an edge set that is NOT exactly the two
                    // native successors is real corruption.
                    let branch_block = cfg.blocks.iter().position(|b| {
                        final_insns
                            .get(b.end - 1)
                            .is_some_and(|t| t.address == d.addr && matches!(t.op, Op::CondJump(_)))
                    });
                    if let Some(tb) = branch_block {
                        let succs = cfg.successors.get(tb);
                        let taken_addr = succs
                            .and_then(|s| s.first())
                            .and_then(|&i| final_insns.get(i))
                            .map(|i| i.address);
                        let fall_addr = succs
                            .and_then(|s| s.last())
                            .and_then(|&i| final_insns.get(i))
                            .map(|i| i.address);
                        // The not-taken successor is the instruction AFTER this
                        // one, so its address is the decoder's byte length away
                        // — 4 only on ARM64. Assuming 4 made this acceptance
                        // path dead on x86-64, where `jcc` is 2 or 6 bytes.
                        // The IR branch's FALL is where the native branch's
                        // TAKEN target is. When that target is an absorbed
                        // instruction the IR landing slides to the first instruction at
                        // or after it, so the same at-or-after contract applies
                        // on this side — without it the inversion acceptance
                        // could only ever fire when the native target happened
                        // to be an exact instruction start, which is the rarer
                        // half of the population.
                        let fall_ok = match fall_addr {
                            Some(f) => {
                                f == native_target
                                    || absorbed_landing(decoded, final_insns, native_target, f)
                            }
                            None => false,
                        };
                        if taken_addr == Some(d.addr + d.size as u64) && fall_ok {
                            ok = true;
                        }
                    }
                }
                if !ok {
                    out.push((
                        d.addr,
                        format!("branch target mismatch: native taken → {native_target:#x}"),
                    ));
                }
            }
            continue;
        }
        // 3. direct call target
        if let Some((target, false)) = d.facts.call {
            let calls: Vec<&Insn> = insns_at
                .iter()
                .filter(|i| matches!(i.op, Op::Call))
                .copied()
                .collect();
            if !calls.is_empty() {
                let ok = calls.iter().any(|i| match i.operands.first() {
                    Some(Operand::Method(_)) => true, // resolver named it
                    Some(Operand::Immediate(v)) => *v as u64 == target,
                    _ => false,
                });
                if !ok {
                    out.push((
                        d.addr,
                        format!("call target mismatch: native → {target:#x}"),
                    ));
                }
            }
            continue;
        }
        // 4. memory addend
        if let Some((base, off)) = d.facts.mem {
            // sp/fp-based addressing is legitimately rebased into stack slots
            if !frame_regs.contains(&base) && off != 0 {
                let has_field = insns_at
                    .iter()
                    .any(|i| i.operands.iter().any(|o| matches!(o, Operand::Field(_))));
                // (is-this-a-stack-cell, addend) so a rebased access is
                // recognizable below.
                let mems: Vec<(bool, i64)> = insns_at
                    .iter()
                    .flat_map(|i| {
                        i.operands.iter().filter_map(|o| match o {
                            Operand::Memory { base, addend, .. } => {
                                Some((is_stack_cell_local(table, base), *addend))
                            }
                            _ => None,
                        })
                    })
                    .collect();
                if !mems.is_empty() && !has_field {
                    let plain_ok = mems.iter().any(|(_, a)| *a == off);
                    // post-index addressing lifts as load-at-0 + separate add
                    let post_ok = mems.iter().any(|(_, a)| *a == 0)
                        && insns_at.iter().any(|i| {
                            matches!(i.op, Op::Arith(ArithKind::Add))
                                && i.operands
                                    .iter()
                                    .any(|o| matches!(o, Operand::Immediate(v) if *v == off))
                        });
                    // The dynamic version of the sp/fp exemption above. The
                    // stack analyzer rebases accesses off ANY register it
                    // proves is a stack pointer alias (`mov reg, rsp` — RAX in
                    // `mov rax,rsp; lea rbp,[rax-K]` prologues, R11 in IL2CPP's
                    // outline frames), and those rebased accesses land in
                    // `stack_N` locals. That set is per-method and cannot be
                    // the fixed `frame_regs`, so the IR side is what proves it:
                    // a native offset compared against a rebased stack cell is
                    // not comparable, precisely as for `[rbp+N]`.
                    let frame_relative = mems.iter().all(|(stack, _)| *stack);
                    if !plain_ok && !post_ok && !frame_relative {
                        let show: Vec<String> = mems.iter().map(|(_, a)| format!("{a}")).collect();
                        out.push((
                            d.addr,
                            format!(
                                "memory offset mismatch: native base{base} +{off}, IR [{}]",
                                show.join(",")
                            ),
                        ));
                    }
                }
            }
            continue;
        }
        // 5. arithmetic kind (reg-op-reg forms only; cmp-class is skipped in
        // fact extraction, MADD/MSUB lower to MulAcc not Arith)
        if let Some((kind, _dest, _pair)) = d.facts.arith {
            let kinds: Vec<ArithKind> = insns_at
                .iter()
                .filter_map(|i| match i.op {
                    Op::Arith(k) => Some(k),
                    _ => None,
                })
                .collect();
            // An empty kind list is legitimate (folded into Compare, became
            // a Field/Call operand, …) — only a PRESENT-but-different kind
            // proves wrong logic.
            if !kinds.is_empty() && !kinds.iter().any(|k| *k == kind) {
                let show: Vec<String> = kinds.iter().map(|k| format!("{k:?}")).collect();
                out.push((
                    d.addr,
                    format!(
                        "arithmetic mismatch: native {kind:?}, IR {}",
                        show.join("/")
                    ),
                ));
            }
        }
    }
}

pub fn verify_method(
    body_bytes: &[u8],
    base_rva: u64,
    pre_pass_insns: &[Insn],
    final_insns: &[Insn],
    cfg: Option<&Cfg>,
    table: Option<&super::locals::LocalTable>,
    is_arm64: bool,
    is_x64: bool,
    is_arm32: bool,
    thumb: bool,
) -> VerifyReport {
    let mut report = VerifyReport {
        method_rva: base_rva,
        ..Default::default()
    };
    let (decoded, decode_failures) = if is_arm64 {
        decode_all_arm64(body_bytes, base_rva)
    } else if is_arm32 {
        decode_all_arm32(body_bytes, base_rva, thumb)
    } else if is_x64 {
        decode_all_x86(body_bytes, base_rva)
    } else {
        (Vec::new(), 0)
    };
    report.decoded = decoded.len();
    report.decode_failures = decode_failures;
    if let Ok(path) = std::env::var("ILR_TV_DUMP") {
        tv_dump_record(&path, base_rva, body_bytes, final_insns, &decoded);
    }
    STATS
        .decode_failures_total
        .fetch_add(decode_failures, Ordering::Relaxed);

    let mut coverage = covered_addresses(final_insns);

    // Folded compare pairs: the lifter emits `Compare` + `CondJump` BOTH at
    // the branch's address (the cmp site keeps a reserved Nop). Judge a
    // decoded cmp-class insn by the branch that follows it: if the next
    // decoded instruction's address hosts a final Compare, the pair's
    // semantics live there — excused, not uncovered.
    let final_cmp_addrs: HashSet<u64> = final_insns
        .iter()
        .filter(|i| matches!(i.op, Op::Compare(_)))
        .map(|i| i.address)
        .collect();
    // Addresses of branches the recovery folded to a CONSTANT condition.
    //
    // A folded compare is normally accounted for by the `Compare` the lifter
    // materialized at the consuming branch. But a pass may then prove the
    // branch itself constant and replace the condition — leaving a `CondJump`
    // whose operand is an `Immediate` and NO `Compare` anywhere, so the fold
    // walk found nothing and reported the compare as dropped. Verified on
    // `GetSpriteIndexFromUnicode` @0x14ec80: the `test rax,rax; je <throw>`
    // null guard at 0x14ecff/0x14ed02 survives as `CondJump(48)` over
    // `Immediate(1)`, and the `test` was reported uncovered (`test [arith]` is
    // 213 of them x64-wide).
    //
    // This cannot hide a genuinely lost compare: losing the pending compare
    // makes the lifter emit `Compare(kind, Unknown, 0)` AT the branch address,
    // whose address is in `final_cmp_addrs` — the `cond-branch without real
    // compare` corpus. A constant operand is the opposite fact: the decision
    // was PROVEN, so the fold was deliberate.
    let folded_branch_addrs: HashSet<u64> = final_insns
        .iter()
        .filter(|i| matches!(i.op, Op::CondJump(_)))
        .filter(|i| matches!(i.operands.first(), Some(Operand::Immediate(_))))
        .map(|i| i.address)
        .collect();
    // `fccmp`/`fccmpe` are in here for the same reason `ccmp` is: they are a
    // COMPARE that is folded into whatever consumes the flags, not an
    // instruction the IR is expected to represent at its own address. Leaving
    // them out meant the fold exemption below never fired for them, and every
    // `fccmp` was reported as a dropped comparison.
    const CMP_MNEMONICS: &[&str] = &[
        // `tst` is ARM64/ARM32; `test` is its x86 twin. Leaving `test` out of
        // this list meant every x86 `test` was reported uncovered (39,048 on
        // v24.5) even though the lifter folded it into the branch exactly as it
        // folds a `test`-then-`jcc` — the largest remaining artifact cluster.
        "cmp", "cmn", "tst", "test", "teq", "ccmp", "ccmn", "fcmp", "fcmpe", "fccmp", "fccmpe",
        // x86 SSE scalar FP compares: like `cmp`, they WRITE flags and the
        // lifter materializes the `Compare` at the consuming branch, so they
        // have no IR at their own address. Without them here every `comiss`
        // was reported uncovered (395 on v24.5) even though its semantics
        // survived at the branch.
        "comiss", "comisd", "ucomiss", "ucomisd",
        // x86 `bt` is the same shape: `lift_x86::bit_scratch_if_carry`
        // materializes a `Compare` at the branch that consumes CF, so a decoded
        // `bt` has no IR at its own address. Leaving it out meant all 175 `bt`
        // on the v24.5 corpus were reported uncovered even though their bit test
        // survived at the consumer. Only the TEST form is here: `bts`/`btr`/
        // `btc` also modify the bit and DO push real IR at their own address
        // (an `Or`/`And`/`Xor`), so they are not folded away.
        "bt",
    ];
    // Every AArch64 mnemonic that WRITES NZCV. Everything else preserves the
    // flags, which is why the compiler is free to shuffle registers between a
    // compare and the branch that reads it. This is the mnemonic-level twin of
    // `lift::preserves_nzcv` (which keys on the opcode); keeping it in sync is
    // what lets the walk below cross `cmp; mov; b.eq` at all.
    const FLAG_WRITERS: &[&str] = &[
        "cmp", "cmn", "tst", "ccmp", "ccmn", "fcmp", "fcmpe", "fccmp", "fccmpe", "subs", "adds",
        "ands", "bics", "adcs", "sbcs", "negs", "ngcs", "msr",
        // x86 bit tests all write CF (the selected bit). Without these in the
        // list a `cmp` before a `bt` walked THROUGH the bit test to the branch
        // and claimed the compare was covered — but the `bt` had already
        // destroyed the compare's flags, so that excuse hid a real finding.
        // Listing them makes the walk stop at the bit test, exactly as it stops
        // at any other flag writer.
        "bt", "bts", "btr", "btc",
    ];
    for (i, d) in decoded.iter().enumerate() {
        // Arch-normalised: an ARM32 `cmps`/`tsteq` must match the same list an
        // AArch64 `cmp`/`tst` does, or the exemption below silently never fires.
        if !CMP_MNEMONICS.contains(&base_mnemonic(&d.mnemonic, is_arm32)) {
            continue;
        }
        // Walk forward to the branch that CONSUMES these flags. The compare and
        // the branch are separate native instructions, and the lifter
        // materializes the `Compare` at the BRANCH (because the branch's
        // condition code is what selects the comparison kind), so a compare is
        // accounted for by a `Compare` at a later address.
        //
        // Two shapes have to be crossed:
        //   `cmp; ccmp; b.cc`  — a compound condition: the intermediate
        //                        instructions are cmp-class;
        //   `cmp; mov; b.eq`   — the compiler shuffling a register between the
        //                        compare and the branch: MOV does NOT touch
        //                        NZCV, so the flags survive it.
        // Any instruction that WRITES the flags ends the search.
        let mut j = i + 1;
        while j < decoded.len() && j - i <= 16 {
            let raw = &decoded[j].mnemonic;
            let m = base_mnemonic(raw, is_arm32);
            if CMP_MNEMONICS.contains(&m) {
                j += 1;
                continue;
            }
            if FLAG_WRITERS.contains(&m) {
                break;
            }
            // The consumer ends the walk: its own address hosts the
            // materialized `Compare` when the flags were not lost.
            //
            // A consumer is usually a BRANCH — but not always. The conditional
            // SELECT family reads the same NZCV and folds the comparison into
            // itself (`materialize_cond` is called from the CSEL/CSINC/FCSEL
            // arm exactly as it is from the Bcc arm), so `cmp w8, w20 ; cset
            // w0, eq` materializes the `Compare` at the `cset`. Walking past
            // the select to the next branch — or off the end of the listing —
            // found no `Compare` there and reported the compare as dropped
            // semantics. `CSET` is how a managed `bool` materialises and `CSEL`
            // is what `Math.Min`/`Max` and every clamp compile to, so this was
            // 251 of the 805 uncovered instructions on the arm64 reference
            // sample — the single largest false-alarm cluster, and one that
            // hid real findings underneath it.
            if consumes_flags(raw, m, is_arm32) || decoded[j].class == InsnClass::Branch {
                break;
            }
            j += 1;
        }
        if j < decoded.len()
            && (final_cmp_addrs.contains(&decoded[j].addr)
                || folded_branch_addrs.contains(&decoded[j].addr)
                // The CONSUMER itself was accounted for — its materialized
                // `Compare` dropped by DCE, folded to a constant, or excised
                // with a guard. The ledger already recorded that decision, so
                // the compare that fed it is accounted for too. Verified on
                // `cmp BYTE [rbx+0xe8], dl; setne dl` @0x3aece1 (method
                // 0x3aecb0): the setne chain at 0x3aece7 is `Nop` in the final
                // IR and ledgered, but the decoded `cmp` at 0x3aece1 lives at a
                // different address, so the walk's `final_cmp_addrs` lookup
                // found nothing and reported a dropped comparison.
                || ledger_has_kind(decoded[j].addr, LedgerKind::Dce)
                || ledger_has_kind(decoded[j].addr, LedgerKind::FoldedBranch)
                || ledger_has_kind(decoded[j].addr, LedgerKind::VacuousBranch)
                || ledger_has_kind(decoded[j].addr, LedgerKind::GuardExcision))
        {
            coverage.arith.insert(d.addr);
            coverage.loadstore.insert(d.addr);
        }
    }

    // Only PROVABLY-CODE instructions are judged. `decoded` is a linear sweep
    // of the measured body, which runs past the function's last `ret` into
    // alignment padding and — on x64 — the switch jump tables emitted inline in
    // the code section. Those bytes decode as perfectly plausible instructions,
    // so a census over the whole sweep was dominated by DATA: 1,164 of the
    // 2,460 "uncovered" insns on the v24.5 corpus (add 382, cmp 268, test 241,
    // bt 175, sub 98), and the cluster table was pointing the port at nothing.
    let reachable = reachable_native_addresses(&decoded, final_insns, base_rva, is_arm32, cfg);
    // Inline jump tables sit in the code section and decode as instructions;
    // they are DATA, never judged (see `inline_table_data`).
    let table_data = inline_table_data(body_bytes, base_rva);
    for d in &decoded {
        if coverage.covers(d.class, d.addr) || ledger_has(d.addr) {
            continue;
        }
        if !reachable.contains(&d.addr) || table_data.contains(&d.addr) {
            report.unreached.push(d.clone());
            continue;
        }
        report.uncovered.push(d.clone());
        bump_cluster(&format!("{} [{}]", d.mnemonic, d.class.name()));
    }

    // Wrong-logic sentinel: every decoded conditional branch must have a
    // REAL compare in the final IR at the branch site. The lifter falls back
    // to `Compare(kind, Unknown, 0)` when it lost the pending compare (the
    // if/else-if chain `b.ne L1; b.eq L2` — the second branch tests flags
    // the FIRST compare set), which renders as `if (? != 0)`: plausible
    // text, wrong logic. Each hit here is a genuine engine bug with an
    // exact address.
    let mut finals_by_addr: HashMap<u64, Vec<&Insn>> = HashMap::new();
    for i in final_insns {
        finals_by_addr.entry(i.address).or_default().push(i);
    }
    // Scratch local → does a REAL Compare define it (known left operand)?
    //
    // The compare and the branch are SEPARATE native instructions, so the
    // lifted `Compare` sits at the `cmp`'s address, not at the `b.cond`'s.
    // Testing for a Compare at the branch's own address therefore mis-flagged
    // every `cmp; b.cond` pair whose compare is intact — 3,519 of the 6,091
    // sites on the reference binary were this false alarm, drowning the real
    // `Compare(kind, Unknown, 0)` cases. Resolve the branch's condition
    // scratch to its defining compare instead, following a Phi merge (a
    // condition merged from several compares is real when any input is).
    let mut real_cmp: HashMap<u32, bool> = HashMap::new();
    for i in final_insns {
        if matches!(i.op, Op::Compare(_)) {
            if let Some(Operand::Local(v)) = i.operands.first() {
                let real = !matches!(i.operands.get(1), Some(Operand::Unknown))
                    && !matches!(i.operands.get(1), Some(Operand::MissingAbiArg));
                real_cmp
                    .entry(*v)
                    .and_modify(|e| *e |= real)
                    .or_insert(real);
            }
        }
    }
    // A condition scratch defined by a CONSTANT `Move` is a FOLDED condition,
    // not a lost one. `passes::constant_folding` rewrites `Compare(imm, imm)`
    // into `dst = imm`, which leaves the `CondJump` reading a constant — the
    // branch is resolved, so it must not be reported as "a branch without a
    // real compare". Before this the whole `0x17a5660` ARM32 residual was this
    // false alarm (`cmp r5, #1` folded once `r5` became a constant).
    let mut const_cond: HashSet<u32> = HashSet::new();
    for i in final_insns {
        if matches!(i.op, Op::Move) {
            if let (Some(Operand::Local(v)), Some(Operand::Immediate(_))) =
                (i.operands.first(), i.operands.get(1))
            {
                const_cond.insert(*v);
            }
        }
    }
    for _ in 0..4 {
        let mut grew = false;
        for i in final_insns {
            if i.op != Op::Phi {
                continue;
            }
            let Some(Operand::Local(dst)) = i.operands.first() else {
                continue;
            };
            let any = i
                .operands
                .iter()
                .skip(1)
                .any(|o| matches!(o, Operand::Local(v) if *real_cmp.get(v).unwrap_or(&false)));
            if any && !real_cmp.get(dst).copied().unwrap_or(false) {
                real_cmp.insert(*dst, true);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    for d in &decoded {
        // Ground truth for "this is a conditional branch" is the decoded
        // instruction's OWN condition fact, which every architecture now
        // populates. The old mnemonic test (`b.eq`, `cbz`, `tbz`) matched only
        // AArch64 spellings, so this oracle — the one that counts the
        // wrong-logic `if (? != 0)` class — silently examined ZERO sites on
        // ARM32 (`beq`, `cbz`) and x86 (`je`, `jne`).
        if d.facts.cond_branch.is_none() {
            continue;
        }
        // Inline jump-table DATA, not code (see `inline_table_data`): a 4-byte
        // entry that encodes a `jcc` is not a conditional branch, so it cannot
        // be a "conditional branch without a compare". The coverage census
        // already excludes these addresses; this oracle must too.
        if table_data.contains(&d.addr) {
            continue;
        }
        // Only PROVABLY-CODE branches are judged, for the same reason the
        // coverage census restricts itself to `reachable`: the body extent runs
        // past the last terminator into string/rodata literal pools, whose bytes
        // decode as plausible conditional branches. On ARM32 the ASCII string
        // table is dense enough that `...::GetPacket()` and friends decode as
        // `blo <addr>` (the ASCII `: :` byte pair is a `blo` encoding), and
        // every one of them was reported as a wrong-logic branch with no
        // compare — 21 of the residual 309 sites were this artifact. A branch
        // the CFG never placed in a reachable block is DATA, not code, and the
        // oracle must not judge it.
        if !reachable.contains(&d.addr) {
            continue;
        }
        // Real when EITHER a Compare sits at the branch address (the CBZ/TBZ
        // shape, where the test and the branch share an address) OR the
        // condition scratch the CondJump reads was defined by a real Compare.
        let real_cmp_at = finals_by_addr
            .get(&d.addr)
            .map(|v| {
                v.iter().any(|i| {
                    if matches!(i.op, Op::Compare(_)) {
                        return !matches!(i.operands.get(1), Some(Operand::Unknown));
                    }
                    if matches!(i.op, Op::CondJump(_)) {
                        if let Some(Operand::Local(v)) = i.operands.first() {
                            return *real_cmp.get(v).unwrap_or(&false) || const_cond.contains(v);
                        }
                        return false;
                    }
                    false
                })
            })
            .unwrap_or(false);
        if !real_cmp_at && !ledger_has(d.addr) {
            report.missing_conditions.push(d.addr);
            bump_cluster("cond-branch without real compare [wrong-logic]");
        }
    }

    // Semantic diff: decoded-native facts vs the final IR at the same
    // address. This is the check the coverage oracle cannot make: an op can
    // EXIST at every address yet mean the WRONG thing (inverted condition,
    // wrong branch/call target, wrong field offset, add lifted as sub).
    // Excused when the pipeline's ledger recorded a rewrite at the address.
    // Runs on every architecture now that all three fill `NativeFacts`; the
    // call is a no-op where the decoder produced no facts.
    if is_arm64 || is_arm32 || is_x64 {
        // Frame traffic is uncomparable: `stack_analyzer` rewrites accesses off
        // these registers into `stack_N` slots, folding the frame adjustment
        // into the addend. Spare BOTH the stack pointer and the frame pointer it
        // is aliased to — sparing only the former flagged every `[rbp+N]` local.
        let frame_regs: [u16; 2] = if is_arm64 {
            [31, 29] // A64 sp, x29 fp
        } else if is_arm32 {
            [13, 11] // A32 sp, r11 fp
        } else {
            [
                iced_x86::Register::RSP as u16,
                iced_x86::Register::RBP as u16,
            ]
        };
        semantic_diff(
            &decoded,
            final_insns,
            &finals_by_addr,
            cfg,
            &frame_regs,
            &table_data,
            table,
            base_rva,
            base_rva + body_bytes.len() as u64,
            &mut report.semantic_diffs,
        );
        for (a, why) in &report.semantic_diffs {
            if !ledger_has(*a) {
                // Cluster key must be address-free so single sites aggregate
                // into countable corruption classes.
                let key = why.split("→").next().unwrap_or(why).trim_end();
                bump_cluster(&format!("sem-diff {key} [wrong-logic]"));
            }
        }
        // Ledger-excused diffs are deliberate pipeline rewrites, not wrong
        // logic — drop them from the per-method report so every remaining
        // line is a provable divergence.
        report.semantic_diffs.retain(|(a, _)| !ledger_has(*a));
    }

    if let Some(cfg) = cfg {
        // ILR_CFGDBG=0xRVA — dump this method's blocks, their terminators and
        // their successors. The `cfg: block N cannot reach an exit` finding
        // names only a block INDEX, and with two pipeline runs per method
        // (dump.cs + the DummyDll) the index alone cannot say WHICH run it came
        // from: the per-pass dump is what showed this method's two runs
        // disagreeing, one ending a block at a resolved `jmp [klass+slot]` and
        // the other at a falling-through `Call`.
        if let Ok(v) = std::env::var("ILR_CFGDBG") {
            let t = v.trim_start_matches("0x");
            if u64::from_str_radix(t, 16).ok() == Some(base_rva) {
                eprintln!(
                    "[cfgdbg] rva={:#x} ninsns={} nblocks={}",
                    base_rva,
                    final_insns.len(),
                    cfg.blocks.len()
                );
                for (bi, b) in cfg.blocks.iter().enumerate() {
                    eprintln!(
                        "[cfgdbg] blk {bi} [{}..{}) last_op={:?} succ={:?}",
                        b.start,
                        b.end,
                        final_insns.get(b.end.saturating_sub(1)).map(|i| &i.op),
                        cfg.successors.get(bi)
                    );
                }
            }
        }
        check_cfg(cfg, final_insns, &mut report.cfg_findings);
    }

    // Corruption oracle #1: undefined locals. A v-reg index READ by the final
    // listing but never defined in it (and not a seeded entry value) renders
    // as `v41 = v10;` with `v10` never written — the output LOOKS plausible
    // but references nothing. Caused by passes erasing the defining op while
    // uses survive (or SSA version mismatches).
    if let Some(t) = table {
        // `Operand::Local(v)` ids are SSA VERSION ids; `LocalTable.infos` is
        // indexed by TABLE position. `index_of` maps version id → table
        // position — seed entry values through it, never by raw position.
        let mut defined: HashSet<u32> = HashSet::new();
        for (v, idx) in &t.index_of {
            let info = &t.infos[*idx];
            if info.is_this || info.is_param.is_some() || info.is_return_buffer {
                defined.insert(*v);
            }
            // An undeclared ABI ARGUMENT register's entry value is an incoming
            // value (a folded native body reached through a member that
            // declares fewer parameters), not a lost definition. See
            // `LocalInfo::is_incoming_arg`.
            if info.is_incoming_arg || info.is_call_clobber {
                defined.insert(*v);
            }
        }
        // The ENTRY version of the x86 ABI return register (`rv`). `rv` is an
        // OUTPUT only — nothing is ever passed in it — so its incoming value is
        // not a parameter and must not be reported. When `rv` is assigned on one
        // side of a branch (the ubiquitous `if (needs_init) call()` guard), SSA
        // builds a phi whose other arm is the entry version; without seeding it
        // that edge copy reads an undefined `rv` (364 residual findings, the
        // largest single origin). Only the lowest version per slot is the entry
        // value (versions are allocated 0-first per slot).
        if let Some(v) = t
            .index_of
            .iter()
            .filter(|(_, idx)| t.infos[**idx].slot_origin == "rv")
            .map(|(&v, _)| v)
            .min()
        {
            defined.insert(v);
        }
        // The STACK POINTER / FRAME POINTER entry values. `mov r5, sp` reads a
        // value the ABI always provides, and `locals::rooted_versions` roots
        // the stack-local ADDRESS (`ref`/`out` argument) so the scrub keeps it
        // — which means the oracle must treat the entry `sp`/`fp` as defined,
        // or every kept `v0(sp)` read reports as a lost definition. This is the
        // second half of the pair whose first half, rooting alone, was
        // measured as a regression (+111 undefined locals). Only the lowest
        // version per slot is the entry value; a later `sub sp, sp, #n` is a
        // real definition.
        for slot in ["sp", "fp"] {
            if let Some(v) = t
                .index_of
                .iter()
                .filter(|(_, idx)| t.infos[**idx].slot_origin == slot)
                .map(|(&v, _)| v)
                .min()
            {
                defined.insert(v);
            }
        }
        // The ABSOLUTE-ADDRESS base entry values (`abs`/`rip`). `lift_x86`
        // materialises `[rip+disp]`/`[abs+disp]` as `Memory { base, addend }`
        // where the base names nothing and the addend IS the address; once the
        // pseudo-register becomes a local, its entry version (lowest per slot,
        // never defined) IS that base value by construction, which
        // `locals::absolute_static_addr` and `resolve::absolute_address`
        // already rely on to name static storage. Reporting it as undefined
        // invents corruption in an address the output folds to a constant
        // (`is_abs_base` in `csharp.rs`), and disagrees with `highvar`'s
        // `seeded_versions`, which roots the same entry values. Only the
        // lowest version per slot is excused; a later defined `abs`/`rip`
        // holds a computed pointer and stays judged. Inert on well-formed IR:
        // it only ADDS two entry versions to the defined set, never removes a
        // real definition, alters no `Insn`, and changes no emitted C# except
        // dropping a spurious `suspect` annotation.
        for slot in ["abs", "rip"] {
            if let Some(v) = t
                .index_of
                .iter()
                .filter(|(_, idx)| t.infos[**idx].slot_origin == slot)
                .map(|(&v, _)| v)
                .min()
            {
                defined.insert(v);
            }
        }
        // Only instructions the CFG can REACH from the entry are judged. A
        // block no edge enters is dead code the renderer never emits: the
        // method-extent overrun past the last terminator (the next function's
        // body, or a rodata literal pool) builds exactly such an orphan, and
        // `liveness_dce` then removes the defining `fp = sp` inside it while a
        // later orphan instruction still READS that version — a "used, never
        // defined" that is really a read of a block the output dropped. The
        // coverage census and the missing-conditions oracle already restrict
        // themselves to `reachable`; this one must speak the same language or
        // it invents corruption in code no reader ever sees (the `x12`/`x11`/
        // `x10`/`x9` orphan tail).
        let live = cfg.map(|c| live_instructions(c, final_insns.len()));
        let mut used: HashSet<u32> = HashSet::new();
        let mut defd_here: HashSet<u32> = HashSet::new();
        for (ii, insn) in final_insns.iter().enumerate() {
            if live.as_ref().is_some_and(|l| !l[ii]) {
                continue;
            }
            if let Some(pos) = Insn::dest_position(&insn.op) {
                if let Some(Operand::Local(v)) = insn.operands.get(pos) {
                    defd_here.insert(*v);
                }
            }
            for o in &insn.operands {
                if let Operand::Local(v) = o {
                    // The ABI RETURN slot read by `Return` is not a data
                    // dependency: a void method's `ret` still carries it, and
                    // it is never an INCOMING value (it is only ever produced by
                    // a call or a computation). Counting it made every void
                    // method whose return register was left unmodified — the
                    // common case — report an undefined local. A method that
                    // really returns a value has DEFINED rv, so nothing real is
                    // hidden: the read is then not in `used.difference(defd)`.
                    if matches!(insn.op, Op::Return)
                        && t.index_of
                            .get(v)
                            .is_some_and(|&i| t.infos[i].slot_origin == "rv")
                    {
                        continue;
                    }
                    used.insert(*v);
                }
            }
            // Phis define all their sources (edge moves materialize them).
            if insn.op == Op::Phi {
                for o in insn.operands.iter().skip(1) {
                    if let Operand::Local(v) = o {
                        defd_here.insert(*v);
                    }
                }
            }
        }
        for v in used.difference(&defd_here) {
            if !defined.contains(v) {
                // A stack cell READ but never written anywhere is an INCOMING
                // STACK ARGUMENT, not corruption. x86-64 passes the arguments
                // past the register ones at positive `[rsp+N]`, and
                // `stack_analyzer` rebases those to `stack_N` — read-only by
                // construction. IL2CPP also zero-inits every managed local, so a
                // genuine uninitialised read does not exist; and a pass that
                // erased a store would show up in the A/A `memory refs GREW`
                // oracle instead. Treat it as an entry value like `is_param`.
                if let Some(&idx) = t.index_of.get(v) {
                    if t.infos[idx].slot_origin.starts_with("stack_") {
                        continue;
                    }
                }
                report.undefined_locals.push(*v);
                report
                    .undefined_local_origins
                    .push(match t.index_of.get(v) {
                        Some(&idx) => t.infos[idx].slot_origin.clone(),
                        None => "?".to_string(),
                    });
                STATS.undefined_locals_total.fetch_add(1, Ordering::Relaxed);
                // Cluster by SLOT ORIGIN (x8? stack_5? vd8?) — the binary-wide
                // ranking points straight at the next root cause.
                if let Some(&idx) = t.index_of.get(v) {
                    bump_cluster(&format!(
                        "undef-local origin {} [corruption]",
                        t.infos[idx].slot_origin
                    ));
                }
            }
        }
        report.undefined_locals.sort_unstable();
    }

    // Corruption oracle #2: unresolved calls. `IndirectCall` that never
    // resolved, and call args that are still `Operand::Unknown` (NOT
    // `MissingAbiArg`, which is the legitimate fewer-live-registers case the
    // renderer drops positionally). These render as `fn_4a58bac(...)`
    // thunks and `__unknown` arguments.
    for insn in final_insns {
        match insn.op {
            Op::IndirectCall => {
                report.unresolved_calls.push((
                    insn.address,
                    "indirect call target never resolved".to_string(),
                ));
                bump_cluster("unresolved indirect call [corruption]");
            }
            Op::Call | Op::CallVoid | Op::NewObj(_) => {
                let unk: Vec<usize> = insn
                    .operands
                    .iter()
                    .enumerate()
                    .filter(|(i, o)| {
                        matches!(o, Operand::Unknown)
                            // dest of Call / NewObj is operand 0 by convention
                            && !(insn.op == Op::Call && *i == 0)
                            && !(matches!(insn.op, Op::NewObj(_)) && *i == 0)
                    })
                    .map(|(i, _)| i)
                    .collect();
                if !unk.is_empty() {
                    report.unresolved_calls.push((
                        insn.address,
                        format!("operand(s) {unk:?} unresolved (__unknown)"),
                    ));
                    bump_cluster("call with __unknown operand [corruption]");
                }
            }
            _ => {}
        }
    }
    STATS
        .unresolved_calls_total
        .fetch_add(report.unresolved_calls.len() as u64, Ordering::Relaxed);

    // Corruption oracle #3: raw register operands in the FINAL listing.
    // SSA must rename EVERY register use into a versioned Local; a survivor
    // means that insn's block was never visited by the renaming DFS — the
    // dominator tree dropped it (a branch target resolving to the wrong insn
    // of a multi-insn address group used to orphan exactly these blocks).
    // The renderer has no expression for a bare register: it prints
    // `__unknown`, so every hit here is a visible corruption site.
    fn raw_regs_in(op: &Operand, out: &mut Vec<String>) {
        match op {
            Operand::Register(n) => out.push(n.clone()),
            Operand::Memory { base, index, .. } => {
                raw_regs_in(base, out);
                if let Some(i) = index {
                    raw_regs_in(i, out);
                }
            }
            // An instance-field access carries its receiver in a NESTED
            // operand. Without this arm a raw register surviving inside
            // `InstanceField { base }` was invisible to the corruption
            // oracle - the exact blind spot that let the in_xN family reach
            // the dump unreported (passes::collect_locals had the same hole).
            Operand::InstanceField { base, .. } => raw_regs_in(base, out),
            Operand::AddressOf(inner) | Operand::ArrayLength(inner) => raw_regs_in(inner, out),
            Operand::ArrayAccess { array, index } => {
                raw_regs_in(array, out);
                raw_regs_in(index, out);
            }
            // Leaves, listed explicitly so a new Operand variant is a compile
            // error here rather than a silent oracle blind spot.
            Operand::Local(_)
            | Operand::Immediate(_)
            | Operand::FloatLiteral(_)
            | Operand::DoubleLiteral(_)
            | Operand::StringLiteral { .. }
            | Operand::Method(_)
            | Operand::Field(_)
            | Operand::Type(_)
            | Operand::MissingAbiArg
            | Operand::Unknown => {}
        }
    }
    for insn in final_insns {
        let mut names = Vec::new();
        for op in &insn.operands {
            raw_regs_in(op, &mut names);
        }
        if !names.is_empty() {
            report.raw_registers.push((
                insn.address,
                format!("{:?} still holds raw register(s) {names:?} — never SSA-renamed, renders __unknown", insn.op),
            ));
            for n in &names {
                bump_cluster(&format!("raw register {n} in final IR [corruption]"));
            }
        }
    }
    STATS
        .raw_registers_total
        .fetch_add(report.raw_registers.len() as u64, Ordering::Relaxed);

    // A/A: raw-lift fingerprint vs final.
    let pre_fp = SemanticFingerprint::of(pre_pass_insns);
    // The listing may contain a spliced OUT-OF-BODY tail-merge epilogue (see
    // `SemanticFingerprint::of_inside`). It has no raw-lift counterpart, so it
    // is excluded rather than charged as synthesis — but only when one is
    // actually present, so the common case keeps the plain fingerprint.
    let body_lo = base_rva;
    let body_hi = base_rva + body_bytes.len() as u64;
    let post_fp = if final_insns
        .iter()
        .any(|i| i.address < body_lo || i.address >= body_hi)
    {
        SemanticFingerprint::of_inside(final_insns, body_lo, body_hi)
    } else {
        SemanticFingerprint::of(final_insns)
    };
    // Tail transfers this method turned into `Call` + `Return` (one ledger note
    // per site, keyed on the transfer's own address — deduped because the
    // synthesized call and return share it).
    let mut tail_calls = 0usize;
    let mut seen_tail: HashSet<u64> = HashSet::new();
    for insn in pre_pass_insns {
        if ledger_has_kind(insn.address, LedgerKind::TailCall) && seen_tail.insert(insn.address) {
            tail_calls += 1;
        }
    }
    report.aa_findings = pre_fp.drift(&post_fp, tail_calls);

    // Structural conformance: does the IR perform the same SET of operations
    // the native bytes demand? Per-site checks above accept a construct whose
    // semantics moved to another shape; this catches the constructs that are
    // simply GONE (dropped condition, lost return, duplicated call).
    report.conform_findings = super::conform::check(&decoded, final_insns, cfg, base_rva);

    STATS.methods_checked.fetch_add(1, Ordering::Relaxed);
    STATS
        .uncovered_total
        .fetch_add(report.uncovered.len() as u64, Ordering::Relaxed);
    STATS
        .unreached_total
        .fetch_add(report.unreached.len() as u64, Ordering::Relaxed);
    STATS
        .cfg_violations_total
        .fetch_add(report.cfg_findings.len() as u64, Ordering::Relaxed);
    STATS
        .aa_drift_total
        .fetch_add(report.aa_findings.len() as u64, Ordering::Relaxed);
    STATS
        .missing_conditions_total
        .fetch_add(report.missing_conditions.len() as u64, Ordering::Relaxed);
    if report.is_clean() {
        STATS.methods_clean.fetch_add(1, Ordering::Relaxed);
    } else {
        STATS.methods_flagged.fetch_add(1, Ordering::Relaxed);
    }
    // Severity, not just presence: a finding's KIND decides which question it
    // answers, and only `WrongLogic` answers "is the recovered logic wrong".
    match report.verdict() {
        Verdict::Clean => {}
        Verdict::WrongLogic => {
            STATS.methods_wrong_logic.fetch_add(1, Ordering::Relaxed);
        }
        Verdict::Suspect => {
            STATS.methods_suspect.fetch_add(1, Ordering::Relaxed);
        }
        Verdict::CoverageOnly => {
            STATS.methods_coverage_only.fetch_add(1, Ordering::Relaxed);
        }
    }
    report
}

/// Pipeline hook: verify and log interesting methods. Returns true when the
/// method is clean. Gated by nothing here — the caller checks ILR_VERIFY.
#[allow(clippy::too_many_arguments)]
pub fn verify_and_log(
    body_bytes: &[u8],
    base_rva: u64,
    pre_pass_insns: &[Insn],
    final_insns: &[Insn],
    cfg: Option<&Cfg>,
    table: Option<&super::locals::LocalTable>,
    is_arm64: bool,
    is_x64: bool,
    is_arm32: bool,
) -> bool {
    // MUST match the lifter's decode mode. `lift_arm32` builds
    // `A32Decoder::default()`, which is ARM mode (`thumb: false`) with no way to
    // select Thumb, so the oracle has to read A32 as well: decoding the same
    // bytes in a different mode makes the verifier describe a different program,
    // and every finding becomes noise. ARM mode is also independently confirmed
    // for this corpus — the bytes at 0x53c838 are `e28d1008` = `add r1, sp, #8`,
    // exactly what the lifter prints.
    let thumb = false;
    let report = verify_method(
        body_bytes,
        base_rva,
        pre_pass_insns,
        final_insns,
        cfg,
        table,
        is_arm64,
        is_x64,
        is_arm32,
        thumb,
    );
    let clean = report.is_clean();
    // The pipeline reached verification for this method (see `GRADED`): the
    // renderer uses this to tell a graded-clean body from an ungraded one.
    mark_graded();
    // Side-channel the grade to the renderer so the emitted method is
    // self-labelling: a reader must be able to tell a clean body from one the
    // oracle could not prove. Only non-clean methods are recorded (a clean one
    // needs no stamp), and only when the pipeline is verifying at all — a
    // `--no-annotate-recovery` run keeps the old output byte-for-byte.
    if !clean {
        if let Some(reason) = report.short_reason() {
            annotate_method(report.method_rva, report.verdict(), reason);
        }
    }
    // The verbose per-method log belongs to ILR_VERIFY. A conformance-only run
    // (ILR_CONFORM) uses this same hook to reach the checker but wants just the
    // aggregate table, so suppress the per-method lines unless asked for.
    let verbose = std::env::var("ILR_VERIFY").is_ok();
    if !clean && verbose {
        eprintln!("{}", report.summarize());
    }
    if !clean && std::env::var("ILR_VERIFY_JSON").is_ok() {
        append_jsonl(&report);
    }
    clean
}

/// Per-method TRANSLATION-VALIDATION dump, gated by `ILR_TV_DUMP=<path>`.
///
/// The differential verifier judges the IR against an oracle built from the
/// SAME decoder crates the lifter uses (`yaxpeax-arm`/`iced-x86`), so a decode
/// bug the two share is invisible to it. This dump hands an OUTSIDE checker the
/// two things it needs to be independent: the method's native bytes and the
/// final IR (op + operands, in listing order, so a `CondJump(t)` index resolves
/// to `ir[t].a`). `tools/tv_check.py` re-decodes the bytes with Capstone — a
/// different disassembler — and re-derives the control-flow facts from the
/// encoding, which is the closest thing to "translate and compare" that runs
/// at this scale.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

thread_local! {
    /// Which recovery consumer is currently running on this worker, for the TV
    /// dump. `ILR_TV_DUMP` fires from `verify_method`, and BOTH the C#/`dump.cs`
    /// path (`csharp.rs`) and the DummyDll path (`recovery_integration.rs`) drive
    /// the pipeline once per method — with DIFFERENT seeds, so the two records
    /// for one rva can carry different IR. Tagging lets the checker keep the
    /// authoritative one (the `dump.cs` rendering is the product).
    static TV_SRC: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Name the recovery consumer whose pipeline is about to run on this worker, so
/// `ILR_TV_DUMP` records can be attributed. Called by the two drivers.
pub fn set_tv_source(s: &'static str) {
    TV_SRC.with(|c| c.set(s));
}

fn tv_source() -> &'static str {
    TV_SRC.with(|c| c.get())
}

fn tv_dump_record(
    path: &str,
    base_rva: u64,
    body_bytes: &[u8],
    final_insns: &[Insn],
    decoded: &[DecodedInsn],
) {
    use std::io::Write;
    static TV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = TV_LOCK.lock();
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        return;
    };
    let mut line = String::with_capacity(body_bytes.len() * 2 + final_insns.len() * 48 + 64);
    line.push_str(&format!(
        "{{\"rva\":{base_rva},\"base\":{base_rva},\"src\":\"{}\",\"bytes\":\"",
        tv_source()
    ));
    for b in body_bytes {
        line.push_str(&format!("{b:02x}"));
    }
    line.push_str("\",\"ir\":[");
    for (i, ins) in final_insns.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let ops: Vec<String> = ins
            .operands
            .iter()
            .map(|o| format!("\"{}\"", json_escape(&format!("{o:?}"))))
            .collect();
        line.push_str(&format!(
            "{{\"i\":{i},\"a\":{},\"op\":\"{}\",\"ops\":[{}]}}",
            ins.address,
            json_escape(&format!("{:?}", ins.op)),
            ops.join(",")
        ));
    }
    line.push_str("],\"decoded\":[");
    for (i, d) in decoded.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        line.push_str(&format!(
            "{{\"a\":{},\"sz\":{},\"m\":\"{}\"}}",
            d.addr,
            d.size,
            json_escape(&d.mnemonic)
        ));
    }
    line.push_str("]}");
    let _ = writeln!(f, "{line}");
}

fn append_jsonl(report: &VerifyReport) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("verify_findings.jsonl")
    else {
        return;
    };
    let uncovered: Vec<String> = report
        .uncovered
        .iter()
        .map(|u| {
            format!(
                "{{\"addr\":{:#x},\"mnemonic\":\"{}\",\"class\":\"{}\"}}",
                u.addr,
                u.mnemonic,
                u.class.name()
            )
        })
        .collect();
    let _ = writeln!(
        f,
        "{{\"rva\":{:#x},\"decoded\":{},\"uncovered\":[{}],\"cfg\":{:?},\"aa\":{:?}}}",
        report.method_rva,
        report.decoded,
        uncovered.join(","),
        report.cfg_findings,
        report.aa_findings
    );
}

#[cfg(test)]
mod tests {
    use super::super::cfg::Block;
    use super::*;

    fn dec(addr: u64, mnemonic: &str) -> DecodedInsn {
        dec_sz(addr, mnemonic, 4)
    }

    fn dec_sz(addr: u64, mnemonic: &str, size: usize) -> DecodedInsn {
        DecodedInsn {
            addr,
            size,
            mnemonic: mnemonic.to_string(),
            class: classify_x86(mnemonic),
            facts: NativeFacts::default(),
        }
    }

    #[test]
    fn the_census_judges_only_provable_code() {
        // `ret` at 0x1000, then four bytes that decode as `add`/`jae`. A linear
        // sweep sees them; the machine cannot reach them. Judging them was the
        // single biggest distortion in the census — 1,164 of the 2,460
        // "uncovered" instructions on v24.5 were jump-table and padding bytes.
        let decoded = vec![
            dec_sz(0x1000, "ret", 1),
            dec_sz(0x1001, "add", 2),
            dec_sz(0x1003, "jae", 2),
        ];
        let insns = vec![Insn::new(Op::Return, vec![], 0x1000)];
        let r = reachable_native_addresses(&decoded, &insns, 0x1000, false, None);
        assert!(r.contains(&0x1000), "the entry is code: {r:?}");
        assert!(
            !r.contains(&0x1001),
            "padding after `ret` is not code: {r:?}"
        );
        assert!(!r.contains(&0x1003));
    }

    #[test]
    fn an_arm32_tail_branch_ends_the_walk() {
        // ARM32 `facts` DO carry a `branch_target` for an unconditional `b`
        // (pinned by `arm32_unconditional_branch_carries_a_branch_target_fact`),
        // so the walk stops at a tail branch on the FACTS path. This test pins
        // the backstop that covers what the facts do NOT type: the
        // register-transfer returns (`bx`, and the predicated `bx<cc>` spelled
        // here), which carry no fact and only partly appear in the mnemonic
        // list. The branch below has no recoverable target, so it is the fact-less
        // case the backstop exists for.
        let decoded = vec![
            dec_sz(0x1000, "b", 4),
            dec_sz(0x1004, "cmpseq", 4),
            dec_sz(0x1008, "bxeq", 4),
        ];
        let insns = vec![Insn::new(Op::Jump(usize::MAX), vec![], 0x1000)];
        let r = reachable_native_addresses(&decoded, &insns, 0x1000, true, None);
        assert!(r.contains(&0x1000), "the entry is code: {r:?}");
        assert!(
            !r.contains(&0x1004),
            "data after a tail branch is not code: {r:?}"
        );
        assert!(!r.contains(&0x1008));
    }

    #[test]
    fn a_predicated_arm32_branch_still_falls_through() {
        // The splitter must separate `beq` from `b`: a predicated branch DOES
        // continue to the next instruction. Treating it as a terminator would
        // hide real losses after every `if`, which is the failure mode this
        // filter must never have.
        let decoded = vec![dec_sz(0x1000, "beq", 4), dec_sz(0x1004, "add", 4)];
        let insns = vec![Insn::new(Op::CondJump(1), vec![], 0x1000)];
        let r = reachable_native_addresses(&decoded, &insns, 0x1000, true, None);
        assert!(
            r.contains(&0x1004),
            "a conditional branch falls through: {r:?}"
        );
    }

    #[test]
    fn a_nop_in_the_final_ir_does_not_seed_reachability() {
        // The pipeline Nops the inline data it decoded past a tail branch, so a
        // Nop's address in the final IR is not evidence of execution. Seeding
        // from it restored reachability to the whole table and re-flagged the
        // data as dropped semantics — the ARM32 `cmpseq`/`cmps`/`teqseq` cluster.
        let decoded = vec![dec_sz(0x1000, "b", 4), dec_sz(0x1004, "cmpseq", 4)];
        let insns = vec![
            Insn::new(Op::Jump(usize::MAX), vec![], 0x1000),
            Insn::new(Op::Nop, vec![], 0x1004),
        ];
        let r = reachable_native_addresses(&decoded, &insns, 0x1000, true, None);
        assert!(!r.contains(&0x1004), "a Nop is not executed code: {r:?}");
    }

    /// The A32 EXT group is undecodable by this crate, so BOTH sides have to read
    /// it from the raw word — and they must agree, or the four published censuses
    /// stop being comparable.
    ///
    /// `0xE6FF5074` is `uxth r5, r4`, a real word from libil2cpp.so method
    /// `0x545724`: it is the loop counter's truncation, the instruction whose
    /// absence made `blo 0x545814` resolve forward to `0x545818` and produced the
    /// `branch target mismatch: native taken` finding. These five words are 18,206
    /// of the ARM32 census's decode failures, 95% of them `UXTH`.
    #[test]
    fn arm32_ext_group_is_read_from_the_raw_word_by_both_sides() {
        for (word, mne) in [
            (0xE6FF5074u32, "uxth"),
            (0xE6EF0070, "uxtb"),
            (0xE6AF0070, "sxtb"),
            (0xE6BF0070, "sxth"),
            (0xE68F0F70, "sxtb16"),
            (0xE6CF0F70, "uxtb16"),
        ] {
            let (ins, fails) = decode_all_arm32(&word.to_le_bytes(), 0x1000, false);
            assert_eq!(fails, 0, "{word:#010x} is an instruction, not data");
            assert_eq!(ins.len(), 1, "{word:#010x}");
            assert_eq!(ins[0].mnemonic, mne);
            assert_eq!(ins[0].class, InsnClass::Arith);
            assert_eq!(ins[0].addr, 0x1000);
        }

        // The recognizer must reject the neighbours, or two different groups
        // would collapse into one: `add r11, sp, #8` and `cmp r1, #0` are the
        // same `cond 0110` prefix family and the crate decodes them itself.
        for word in [0xE28DB008u32, 0xE3510000, 0xE1A00000, 0xEA000000] {
            assert!(
                super::super::lift_arm32::ext_op(word).is_none(),
                "{word:#010x} is not an EXT instruction"
            );
        }
    }

    /// A word yaxpeax rejects is not automatically a hole the LIFTER left. The
    /// VFP coprocessor group and the signed-DSP multiply group are reconstructed
    /// from the raw bit fields by `lift_arm32`, so they are an ORACLE gap —
    /// recovered but uncheckable — and only the words BOTH sides drop count as
    /// reconstruction gaps. This pins the discriminator the census uses.
    #[test]
    fn arm32_decode_failures_separate_oracle_gaps_from_real_ones() {
        // `0xEE0E0A10` is `vmov s0, r0` (copro 10, Vn = Vd = 0, bit4 = 1):
        // yaxpeax cannot decode any VFP word, but the lifter does.
        assert!(
            super::super::lift_arm32::liftable_word(0xEE0E0A10),
            "the lifter recovers VFP from the raw word"
        );
        // The EXT group is likewise handled (and synthesized on the oracle side).
        assert!(super::super::lift_arm32::liftable_word(0xE6FF5074));
        // An ordinary decoded instruction and a data word are NOT liftable
        // exceptions: if either reaches the failure path the recovery really
        // did drop it.
        for word in [0xE28DB008u32, 0x0000_0000, 0xE3510000] {
            assert!(
                !super::super::lift_arm32::liftable_word(word),
                "{word:#010x} is not a recovered-from-word encoding"
            );
        }
    }

    /// NEON (Advanced SIMD) is undecodable by the oracle and was excluded from
    /// the hand VFP decoder, so it produced no IR at all. Both sides recognize
    /// it now: the oracle synthesizes a `Pseudo` entry (so a branch to the word
    /// has a landing and the coverage census is silent) and the lifter emits the
    /// catch-all `Op::Simd`.
    #[test]
    fn arm32_neon_word_is_recognized_by_both_sides() {
        for word in [0xf2c00050u32, 0xf4400acf, 0xf4600acf] {
            let (ins, fails) = decode_all_arm32(&word.to_le_bytes(), 0x1000, false);
            assert_eq!(fails, 0, "{word:#010x} must not be a decode failure");
            assert_eq!(ins.len(), 1, "{word:#010x}");
            assert_eq!(ins[0].class, InsnClass::Pseudo, "{word:#010x}");
            assert_eq!(ins[0].addr, 0x1000);
        }
        // Not NEON: the scalar VFP/EXT space (cond 0xE) and ordinary data.
        for word in [0xE28DB008u32, 0x0000_0000, 0xE6FF5074] {
            assert!(
                !super::super::lift_arm32::neon_word(word),
                "{word:#010x} is not NEON"
            );
        }
    }

    /// A VFP word the lifter ABSORBS into its consumer (a `vcmp`/`vcmpe` flag
    /// setter, or the `vmrs`/`vmsr` flag transfer) emits no IR of its own, so
    /// the oracle must synthesize a `Pseudo` entry for it — otherwise
    /// `absorbed_landing` cannot certify a branch that lands on one and reports
    /// a wrong branch target (the dominant ARM32 `branch target mismatch`
    /// class). Pinned from real bytes: `eeb59ac0`/`eeb58ac0` are
    /// `vcmpe.f32 sN, #0`, `eeb40ac8` is `vcmpe.f32 s0, s16`, and `eef1fa10` is
    /// `vmrs APSR_nzcv, fpscr`.
    #[test]
    fn arm32_absorbed_vfp_word_is_synthesized_for_the_landing_walk() {
        for word in [0xEEB59AC0u32, 0xEEB58AC0, 0xEEB40AC8, 0xEEF1FA10] {
            assert!(
                super::super::lift_arm32::vfp_absorbed_word(word),
                "{word:#010x} is a VFP flag setter/transfer the lifter absorbs"
            );
            let (ins, _fails) = decode_all_arm32(&word.to_le_bytes(), 0x1000, false);
            assert_eq!(ins.len(), 1, "{word:#010x} must have an oracle entry");
            assert_eq!(ins[0].mnemonic, "vfp", "{word:#010x}");
            assert_eq!(ins[0].class, InsnClass::Pseudo, "{word:#010x}");
            assert!(
                super::is_folded_into_successor(&ins[0].mnemonic),
                "the synthesized token must certify a folded landing"
            );
        }
        // A VFP word the lifter does NOT handle (`eeb10a49` = `vneg.f32 s0, s18`
        // by llvm's reading — the lifter answers `NotVfp` for this slice) is a
        // genuine DROP, not an absorbed flag setter: it must not be certified.
        assert!(!super::super::lift_arm32::vfp_absorbed_word(0xEEB10A49));
        // Nor is an ordinary instruction.
        assert!(!super::super::lift_arm32::vfp_absorbed_word(0xE28DB008));
    }

    #[test]
    fn arm32_unconditional_branch_carries_a_branch_target_fact() {
        // The `DecodedInsn::facts` contract for ARM32, pinned from real bytes:
        // `ea000000` is `b #8`, and the walk must stop there on the FACTS path.
        // (The stale claim that these facts are "arm64 only" is why the
        // inline-table leak was first misdiagnosed.)
        let (ins, fails) = decode_all_arm32(&0xEA00_0000u32.to_le_bytes(), 0x1000, false);
        assert_eq!(fails, 0);
        assert_eq!(ins[0].facts.branch_target, Some(0x1008));
        assert!(ins[0].facts.cond_branch.is_none());

        let (decoded, _) = decode_all_arm32(
            &[0x00, 0x00, 0x00, 0xEA, 0x00, 0x00, 0x51, 0xE3],
            0x1000,
            false,
        );
        let r = reachable_native_addresses(&decoded, &[], 0x1000, true, None);
        assert!(r.contains(&0x1000), "the entry is code: {r:?}");
        assert!(
            !r.contains(&0x1004),
            "a `b` has no fallthrough, so the word after it is data: {r:?}"
        );
    }

    #[test]
    fn a_dead_block_in_the_final_ir_is_not_a_seed() {
        // A spelling the lifter models as a REAL op survives on the bytes of an
        // inline data table (unlike the ones it Nops), so skipping Nops alone
        // left the table reachable. The CFG is the recovery's own answer to what
        // can execute: an `Arith` in a block no edge enters must not seed.
        let (decoded, _) = decode_all_arm32(
            &[0x00, 0x00, 0x00, 0xEA, 0x00, 0x00, 0x51, 0xE3],
            0x1000,
            false,
        );
        // Block 0 = [0x1000] (the `b`), block 1 = [0x1004] (data), and block 1
        // has NO predecessor: `b` has no fallthrough edge.
        let insns = vec![
            Insn::new(Op::Jump(usize::MAX), vec![], 0x1000),
            Insn::new(Op::Arith(ArithKind::And), vec![Operand::Local(1)], 0x1004),
        ];
        let cfg = Cfg {
            blocks: vec![Block { start: 0, end: 1 }, Block { start: 1, end: 2 }],
            successors: vec![vec![], vec![]],
        };
        let r = reachable_native_addresses(&decoded, &insns, 0x1000, true, Some(&cfg));
        assert!(r.contains(&0x1000));
        assert!(
            !r.contains(&0x1004),
            "an instruction in an unreachable block is not a seed: {r:?}"
        );
    }

    #[test]
    fn a_recovered_switch_body_is_still_judged() {
        // A case body is reachable only through the indirect `jmp [table]`, so
        // seeding from `Op::Switch` / the final IR is load-bearing: without it
        // every switch body would look like data and its REAL losses would
        // vanish from the report — the failure mode this filter must not have.
        let decoded = vec![
            dec_sz(0x1000, "jmp", 2),
            dec_sz(0x1010, "add", 2),
            dec_sz(0x1012, "mov", 2),
        ];
        let insns = vec![
            Insn::new(Op::Switch(vec![1]), vec![], 0x1000),
            Insn::new(Op::Arith(ArithKind::Add), vec![], 0x1010),
        ];
        let r = reachable_native_addresses(&decoded, &insns, 0x1000, false, None);
        assert!(
            r.contains(&0x1010),
            "a recovered switch body is code: {r:?}"
        );
        assert!(
            r.contains(&0x1012),
            "…and the walk continues through it: {r:?}"
        );
    }

    #[test]
    fn the_multibyte_nop_pad_is_pseudo_not_a_dropped_exchange() {
        // `66 90` is the multi-byte NOP MSVC emits between functions. Iced
        // decodes it as `Nopw` (mnemonic `Nop`) but its FORMATTER renders it
        // `xchg ax,ax`, and the classifier was fed the formatter's text — so
        // 387 padding instructions were reported as dropped ARITHMETIC `xchg`,
        // the largest false cluster on the v24.5 corpus.
        let (decoded, fails) = decode_all_x86(&[0x66, 0x90, 0x90, 0x48, 0x87, 0xD8], 0x1000);
        assert_eq!(fails, 0);
        assert_eq!(decoded[0].mnemonic, "nop", "`66 90` must classify as Nop");
        assert_eq!(
            decoded[0].class,
            InsnClass::Pseudo,
            "`66 90` is a no-op pad"
        );
        assert_eq!(decoded[1].class, InsnClass::Pseudo, "plain `90` is a no-op");
        // …and a REAL exchange must keep reporting, or the fix would hide
        // losses. `48 87 D8` = `xchg rax, rbx`, whose formatter text and
        // mnemonic agree, so it stays Arith.
        assert_eq!(decoded[2].mnemonic, "xchg");
        assert_eq!(
            decoded[2].class,
            InsnClass::Arith,
            "xchg rax, rbx exchanges"
        );
    }

    /// `bt eax, ecx` (0F A3 C8) then `jae` (73 00): the lifter hands CF to the
    /// consumer via `bit_scratch_if_carry`, which materializes the `Compare` at
    /// the BRANCH's address — so the `bt` has no IR at 0x1000, exactly like a
    /// folded `cmp`/`test`. The oracle must excuse it the same way: 175 `bt`
    /// were reported dropped on the v24.5 corpus before `bt` was added to the
    /// compare-fold list.
    #[test]
    fn a_folded_bit_test_is_not_a_dropped_comparison() {
        let bytes = [0x0F, 0xA3, 0xC8, 0x73, 0x00];
        let final_insns = vec![
            Insn::new(Op::Compare(CompareKind::Equal), vec![], 0x1003),
            Insn::new(Op::CondJump(2), vec![], 0x1003),
        ];
        let r = verify_method(
            &bytes,
            0x1000,
            &[],
            &final_insns,
            None,
            None,
            false,
            true,
            false,
            false,
        );
        assert!(
            !r.uncovered.iter().any(|d| d.addr == 0x1000),
            "a `bt` whose bit test survived at the consumer is not uncovered: {:?}",
            r.uncovered
                .iter()
                .map(|d| (d.addr, d.mnemonic.clone()))
                .collect::<Vec<_>>()
        );
    }

    /// `cmp eax, ecx` (39 C8), `bts eax, ecx` (0F AB C8), `jae`. `bts` WRITES
    /// CF (to the pre-modification bit), so the `cmp`'s flags do NOT reach the
    /// branch — the `Compare` at 0x1005 belongs to the `bts`. Excusing the
    /// `cmp` against it would hide a real dropped comparison, which is why the
    /// modify forms must stop the fold walk.
    #[test]
    fn a_bit_test_modify_form_ends_a_compare_fold_walk() {
        let bytes = [0x39, 0xC8, 0x0F, 0xAB, 0xC8, 0x73, 0x00];
        let final_insns = vec![
            // bts's own IR: it is NOT folded away
            Insn::new(Op::Arith(ArithKind::Or), vec![], 0x1002),
            Insn::new(Op::Compare(CompareKind::Equal), vec![], 0x1005),
            Insn::new(Op::CondJump(2), vec![], 0x1005),
        ];
        let r = verify_method(
            &bytes,
            0x1000,
            &[],
            &final_insns,
            None,
            None,
            false,
            true,
            false,
            false,
        );
        assert!(
            r.uncovered.iter().any(|d| d.addr == 0x1000),
            "a `cmp` whose flags a `bts` destroyed is a real dropped comparison"
        );
    }

    /// Only the TEST form is absorbed. The modify forms push real IR at their
    /// own address, so treating them as folded would let a branch land past
    /// them and skip the bit modification.
    #[test]
    fn only_the_bt_test_form_is_folded_into_its_successor() {
        assert!(is_folded_into_successor("bt"));
        assert!(!is_folded_into_successor("bts"));
        assert!(!is_folded_into_successor("btr"));
        assert!(!is_folded_into_successor("btc"));
    }

    /// A symbolic intrinsic is REPRESENTED, not dropped.
    ///
    /// `shufps xmm0, xmm1, 0x1b` reaches the lifter's catch-all, which emits
    /// `Op::Simd("__Shufps")` with the real operands at the instruction's own
    /// address (verified with `ILR_X86GAPS`: `X86GAP Shufps|ops=Register,
    /// Register,Immediate8`). The coverage table simply had no arm for
    /// `Op::Simd`, so `shufps` 221 + `cvtdq2ps` 260 + `unpcklps` 101 — plus
    /// `psrldq`, `movi` and every NEON intrinsic — were scored as dropped
    /// semantics on v24.5 despite having IR at their own address.
    /// A compare folded into a CONSTANT branch is accounted for.
    #[test]
    fn a_compare_folded_into_a_constant_branch_is_accounted() {
        // `test rax, rax` (48 85 C0) then `je` (74 0E). A later pass proved the
        // null guard constant, so the IR holds `CondJump` over an `Immediate`
        // and no `Compare` survives — the fold walk must still account for the
        // `test`, or every legitimately excised guard looks like a dropped
        // comparison (`test [arith]` was 213 of them x64-wide).
        let bytes = [0x48, 0x85, 0xC0, 0x74, 0x0E];
        let final_insns = vec![
            Insn::new(Op::Nop, vec![], 0x1003),
            Insn::new(Op::CondJump(1), vec![Operand::Immediate(1)], 0x1003),
        ];
        let r = verify_method(
            &bytes,
            0x1000,
            &[],
            &final_insns,
            None,
            None,
            false,
            true,
            false,
            false,
        );
        assert!(
            !r.uncovered.iter().any(|d| d.addr == 0x1000),
            "a compare folded into a constant branch is not a dropped comparison"
        );
    }

    /// The exact bytes from `GetTextAnchorPivot` @0x3b3900's switch table.
    #[test]
    fn an_inline_jump_table_is_data_not_code() {
        let base = 0x3b3900u64;
        let mut body = vec![0u8; 0x148]; // body up to the table (0x3b3a48)
        // three consecutive in-body RVAs: 0x3b3947, 0x3b3951, 0x3b3963
        body.extend_from_slice(&[
            0x47, 0x39, 0x3b, 0x00, 0x51, 0x39, 0x3b, 0x00, 0x63, 0x39, 0x3b, 0x00,
        ]);
        let d = inline_table_data(&body, base);
        assert!(
            d.contains(&(base + 0x148)),
            "the table's first byte is data"
        );
        assert!(d.contains(&(base + 0x149)), "…and its interior");
        // The leading zero padding is NOT an in-body pointer, so it must not be
        // marked: this only recognises the table, never ordinary bytes.
        assert!(!d.contains(&base), "padding before the table is not data");
    }

    #[test]
    fn a_symbolic_simd_op_still_covers_its_native_instruction() {
        let insns = vec![Insn::new(
            Op::Simd("__Shufps".into()),
            vec![Operand::Register("vf0".into())],
            0x1000,
        )];
        let c = covered_addresses(&insns);
        assert!(
            c.covers(InsnClass::Arith, 0x1000),
            "a symbolic SIMD op is represented, not a dropped instruction"
        );
        // The signal must survive: an address with NO IR at all is still
        // uncovered, so this only reclassifies what the lift actually emitted.
        assert!(!c.covers(InsnClass::Arith, 0x2000));
    }

    /// A MODELLED lane op covers its address too — and, because its lanes are
    /// data, its lane selection can be CHECKED rather than trusted.
    #[test]
    fn a_lane_op_is_covered_and_its_lanes_are_range_checked() {
        use crate::recovery::ir::{LanePick, SimdOp};
        let good = Insn::new(
            Op::SimdLane(SimdOp::Arrange {
                elem_bytes: 4,
                lanes: vec![
                    LanePick::A(3),
                    LanePick::A(2),
                    LanePick::B(1),
                    LanePick::B(0),
                ],
            }),
            vec![
                Operand::Register("vf0".into()),
                Operand::Register("vf0".into()),
                Operand::Register("vf1".into()),
            ],
            0x1000,
        );
        let c = covered_addresses(std::slice::from_ref(&good));
        assert!(
            c.covers(InsnClass::Arith, 0x1000),
            "an exactly-modelled lane op is represented, not a dropped instruction"
        );
        assert!(out_of_range_lane_selections(std::slice::from_ref(&good)).is_empty());

        // Four DWORD lanes exist; a fifth does not, and that lowering would
        // compute a wrong value rather than a missing one.
        let bad = Insn::new(
            Op::SimdLane(SimdOp::Arrange {
                elem_bytes: 4,
                lanes: vec![LanePick::A(4)],
            }),
            vec![
                Operand::Register("vf0".into()),
                Operand::Register("vf0".into()),
            ],
            0x2000,
        );
        let bads = out_of_range_lane_selections(std::slice::from_ref(&bad));
        assert_eq!(
            bads.len(),
            1,
            "an impossible lane must be reported: {bads:?}"
        );
        assert_eq!(bads[0].0, 0x2000);
        assert!(
            bads[0].1.contains("arrange32"),
            "the report must name the operation: {bads:?}"
        );
    }

    /// The x86 static-init guard, end to end at the oracle's level: a `jcc`
    /// whose taken target is a `test` the lifter folded into a LATER branch.
    /// `cfg::resolve_target` lands that edge on the first IR instruction at or
    /// after the folded address, and the oracle must accept it — demanding
    /// exact address equality reported every restored edge as a mismatch.
    #[test]
    fn a_landing_on_an_absorbed_flag_setter_is_accepted() {
        // `test al, al` is 2 bytes, so the instruction that absorbed it starts
        // at 0x1032 and the skipped run reaches the landing exactly.
        let decoded = vec![dec(0x1000, "je"), dec_sz(0x1030, "test", 2)];
        let final_insns = vec![
            Insn::new(Op::Compare(CompareKind::Equal), vec![], 0x1000),
            Insn::new(Op::CondJump(3), vec![], 0x1000),
            Insn::new(Op::Return, vec![], 0x1004),
            // the landing: nothing starts at 0x1030, the `test` is absorbed
            Insn::new(Op::Return, vec![], 0x1032),
        ];
        assert!(
            absorbed_landing(&decoded, &final_insns, 0x1030, 0x1032),
            "a landing past a folded flag setter is exact, not an approximation"
        );
    }

    /// The refusal that keeps the acceptance honest: if the instruction skipped
    /// was DROPPED rather than folded, branching past it would skip real work.
    #[test]
    fn a_landing_past_a_dropped_instruction_is_still_flagged() {
        let decoded = vec![dec(0x1000, "je"), dec_sz(0x1030, "mulss", 4)];
        let final_insns = vec![
            Insn::new(Op::Compare(CompareKind::Equal), vec![], 0x1000),
            Insn::new(Op::CondJump(3), vec![], 0x1000),
            Insn::new(Op::Return, vec![], 0x1004),
            Insn::new(Op::Return, vec![], 0x1032),
        ];
        assert!(
            !absorbed_landing(&decoded, &final_insns, 0x1030, 0x1032),
            "a dropped `mulss` must not be stepped over silently"
        );
    }

    /// `landing` must be the FIRST IR instruction at or after the target. If
    /// some other instruction sits in between, the resolver did not produce
    /// this shape and the edge is a real mismatch.
    #[test]
    fn a_landing_that_is_not_the_first_instruction_at_or_after_is_flagged() {
        let decoded = vec![dec(0x1000, "je"), dec_sz(0x1030, "test", 2)];
        let final_insns = vec![
            Insn::new(Op::Compare(CompareKind::Equal), vec![], 0x1000),
            Insn::new(Op::CondJump(3), vec![], 0x1000),
            // an IR instruction strictly between the target and the landing
            Insn::new(Op::Return, vec![], 0x1030),
            Insn::new(Op::Return, vec![], 0x1032),
        ];
        assert!(
            !absorbed_landing(&decoded, &final_insns, 0x1030, 0x1032),
            "0x1030 has an IR instruction of its own, so the landing is wrong"
        );
    }

    /// The exact match is not routed through the absorbed path at all.
    #[test]
    fn an_exact_landing_is_not_treated_as_absorbed() {
        let decoded = vec![dec(0x1000, "je"), dec_sz(0x1030, "test", 2)];
        let final_insns = vec![Insn::new(Op::Return, vec![], 0x1030)];
        assert!(
            !absorbed_landing(&decoded, &final_insns, 0x1030, 0x1030),
            "equal addresses are the exact case and must not be reported as absorbed"
        );
    }

    /// Byte-contiguity is what stops the acceptance from hiding a real
    /// mismatch: if a byte in the skipped range failed to decode, the chain
    /// does not reach the landing and the edge stays flagged.
    #[test]
    fn an_undecodable_byte_inside_the_skipped_run_breaks_the_acceptance() {
        // A 1-byte `test` claims to occupy 0x1030 only, so the run stops at
        // 0x1031 and never reaches the landing at 0x1032.
        let decoded = vec![dec(0x1000, "je"), dec_sz(0x1030, "test", 1)];
        let final_insns = vec![
            Insn::new(Op::Compare(CompareKind::Equal), vec![], 0x1000),
            Insn::new(Op::CondJump(3), vec![], 0x1000),
            Insn::new(Op::Return, vec![], 0x1004),
            Insn::new(Op::Return, vec![], 0x1032),
        ];
        assert!(
            !absorbed_landing(&decoded, &final_insns, 0x1030, 0x1032),
            "a gap in the decoded run must not be stepped over"
        );
    }

    /// A branch that lands on the `ucomiss`/`comiss` that OPENS a compare
    /// chain must be accepted: the x86 lifter folds the FP compare into the
    /// branch that consumes it (`pending_fp` → `materialize_pending`), so the
    /// FP compare's own address has no IR entry — exactly like `cmp`/`test`.
    ///
    /// Real site on the v24.5 build @0x3ac277: `jle 0x3ac2c0`, where 0x3ac2c0
    /// is `ucomiss xmm7, xmm6` followed by `jp`/`je`; the IR hosts the Compare
    /// at the `jp` (0x3ac2c3), so the native target 0x3ac2c0 has no landing.
    ///
    /// The negative half is the part that matters: the same shape with a real
    /// `mulss` in the skipped run must STAY refused, because branching past it
    /// would skip work that has an effect.
    #[test]
    fn a_branch_landing_on_a_folded_fp_compare_is_accepted() {
        // 0x1000: 0f 2e fe  ucomiss xmm7, xmm6
        // 0x1003: 7a 00     jp +0
        let bytes = [0x0f, 0x2e, 0xfe, 0x7a, 0x00];
        let (decoded, _) = decode_all_x86(&bytes, 0x1000);
        let final_insns = vec![
            Insn::new(Op::Compare(CompareKind::LessThan), vec![], 0x1003),
            Insn::new(Op::CondJump(2), vec![], 0x1003),
        ];
        assert!(
            absorbed_landing(&decoded, &final_insns, 0x1000, 0x1003),
            "an FP compare folded into its consumer is a legal landing"
        );

        // 0x1000: 0f 59 c1  mulss xmm0, xmm1   (real work — must break the chain)
        // 0x1003: 0f 2e fe  ucomiss xmm7, xmm6
        // 0x1006: 7a 00     jp +0
        let bytes = [0x0f, 0x59, 0xc1, 0x0f, 0x2e, 0xfe, 0x7a, 0x00];
        let (decoded, _) = decode_all_x86(&bytes, 0x1000);
        let final_insns = vec![
            Insn::new(Op::Compare(CompareKind::LessThan), vec![], 0x1006),
            Insn::new(Op::CondJump(2), vec![], 0x1006),
        ];
        assert!(
            !absorbed_landing(&decoded, &final_insns, 0x1000, 0x1006),
            "a real `mulss` in the skipped run must not be stepped over"
        );
    }

    #[test]
    fn only_folded_mnemonics_are_steppable() {
        for m in [
            "cmp", "test", "cmn", "tst", "nop", "endbr64", "ucomiss", "ucomisd", "comiss", "comisd",
        ] {
            assert!(is_folded_into_successor(m), "{m} is folded");
        }
        for m in ["mulss", "shufps", "xchg", "jmp", "call", "ret", "mov"] {
            assert!(!is_folded_into_successor(m), "{m} is not folded");
        }
    }

    fn ret_body() -> Vec<u8> {
        // ARM64: MOV X0, #42 ; RET
        let mov = 0xD2800540u32;
        let ret = 0xD65F03C0u32;
        let mut b = Vec::new();
        b.extend_from_slice(&mov.to_le_bytes());
        b.extend_from_slice(&ret.to_le_bytes());
        b
    }

    #[test]
    fn undefined_local_is_flagged() {
        let bytes = ret_body();
        // LocalTable with one entry (not this/param/retbuf → not seeded).
        let table = super::super::locals::LocalTable::default();
        let final_insns = vec![Insn::new(
            Op::Return,
            vec![Operand::Local(7)], // v7 read, never defined
            0x1000,
        )];
        let report = verify_method(
            &bytes,
            0x1000,
            &final_insns,
            &final_insns,
            None,
            Some(&table),
            true,
            false,
            false,
            false,
        );
        assert!(
            report.undefined_locals.contains(&7),
            "{}",
            report.summarize()
        );
    }

    #[test]
    fn undefined_local_in_an_unreachable_block_is_not_flagged() {
        // The method extent overruns the last terminator into the next
        // function; that orphan block is dropped by the renderer, so a read in
        // it is not corruption the reader can see. Block 1 has NO predecessor.
        let bytes = ret_body();
        let table = super::super::locals::LocalTable::default();
        let final_insns = vec![
            Insn::new(Op::Return, vec![], 0x1000),
            Insn::new(
                Op::Move,
                vec![Operand::Local(8), Operand::Local(7)], // v7 read, never defined
                0x1004,
            ),
        ];
        let cfg = super::super::cfg::Cfg {
            blocks: vec![
                super::super::cfg::Block { start: 0, end: 1 },
                super::super::cfg::Block { start: 1, end: 2 },
            ],
            successors: vec![vec![], vec![]],
        };
        let report = verify_method(
            &bytes,
            0x1000,
            &final_insns,
            &final_insns,
            Some(&cfg),
            Some(&table),
            true,
            false,
            false,
            false,
        );
        assert!(report.undefined_locals.is_empty(), "{}", report.summarize());
    }

    #[test]
    fn seeded_entry_locals_are_not_undefined() {
        let bytes = ret_body();
        let mut table = super::super::locals::LocalTable::default();
        table.infos.push(super::super::locals::LocalInfo {
            slot_origin: "x0".into(),
            ty: super::super::types::ManagedType::Unknown,
            is_this: true,
            is_param: None,
            param_comp: None,
            is_incoming_arg: false,
            is_call_clobber: false,
            is_return_buffer: false,
        });
        // Version id 25 maps to the seeded entry local (index_of, NOT position).
        table.index_of.insert(25, 0);
        let final_insns = vec![Insn::new(Op::Return, vec![Operand::Local(25)], 0x1000)];
        let report = verify_method(
            &bytes,
            0x1000,
            &final_insns,
            &final_insns,
            None,
            Some(&table),
            true,
            false,
            false,
            false,
        );
        assert!(report.undefined_locals.is_empty());
    }

    /// The absolute-address base entry (`abs`/`rip`) is a seeded value, not
    /// corruption. Its entry version (lowest per slot, never defined) IS the
    /// address the output folds to a constant (`is_abs_base`), which
    /// `locals::absolute_static_addr`, `resolve::absolute_address`, and
    /// `highvar::seeded_versions` already rely on. Fails pre-fix (reported as
    /// undefined); passes with the `abs`/`rip` excusal above.
    #[test]
    fn abs_rip_entry_versions_are_not_undefined() {
        let bytes = ret_body();
        for origin in ["abs", "rip"] {
            let mut table = super::super::locals::LocalTable::default();
            table.infos.push(super::super::locals::LocalInfo {
                slot_origin: origin.into(),
                ty: super::super::types::ManagedType::Unknown,
                is_this: false,
                is_param: None,
                param_comp: None,
                is_incoming_arg: false,
                is_call_clobber: false,
                is_return_buffer: false,
            });
            table.index_of.insert(25, 0);
            let final_insns = vec![Insn::new(Op::Return, vec![Operand::Local(25)], 0x1000)];
            let report = verify_method(
                &bytes,
                0x1000,
                &final_insns,
                &final_insns,
                None,
                Some(&table),
                true,
                false,
                false,
                false,
            );
            assert!(
                report.undefined_locals.is_empty(),
                "{origin} entry is seeded, not lost: {}",
                report.summarize()
            );
        }
    }

    #[test]
    fn unresolved_indirect_call_is_flagged() {
        let bytes = ret_body();
        let final_insns = vec![
            Insn::new(Op::IndirectCall, vec![Operand::reg("x8")], 0x1000),
            Insn::new(Op::Return, vec![], 0x1004),
        ];
        let report = verify_method(
            &bytes,
            0x1000,
            &final_insns,
            &final_insns,
            None,
            None,
            true,
            false,
            false,
            false,
        );
        assert!(
            !report.unresolved_calls.is_empty(),
            "{}",
            report.summarize()
        );
    }

    #[test]
    fn unknown_call_operand_is_flagged_but_missing_abi_arg_is_not() {
        let bytes = ret_body();
        let final_insns = vec![
            Insn::new(
                Op::CallVoid,
                vec![
                    Operand::Method(_mref()),
                    Operand::Unknown,
                    Operand::MissingAbiArg,
                ],
                0x1000,
            ),
            Insn::new(Op::Return, vec![], 0x1004),
        ];
        let report = verify_method(
            &bytes,
            0x1000,
            &final_insns,
            &final_insns,
            None,
            None,
            true,
            false,
            false,
            false,
        );
        assert_eq!(report.unresolved_calls.len(), 1);
        assert!(report.unresolved_calls[0].1.contains("operand(s) [1]"));
    }

    fn _mref() -> super::super::types::MethodRef {
        super::super::types::MethodRef {
            image_index: 0,
            type_def_index: 0,
            method_index: 0,
            name: "M".into(),
            declaring_namespace: String::new(),
            declaring_type_name: "T".into(),
            assembly: "A".into(),
            sig: super::super::types::MethodSigInfo {
                ret: super::super::types::ManagedType::Unknown,
                params: Vec::new(),
                is_static: false,
            },
        }
    }

    #[test]
    fn decode_all_arm64_counts_both() {
        let (insns, fails) = decode_all_arm64(&ret_body(), 0x1000);
        assert_eq!(fails, 0);
        assert_eq!(insns.len(), 2);
        assert_eq!(insns[0].class, InsnClass::Arith);
        assert_eq!(insns[1].class, InsnClass::Branch);
    }

    #[test]
    fn coverage_passes_on_matching_ir() {
        let bytes = ret_body();
        // Post-SSA IR: every register use is a versioned Local (raw registers
        // in the final listing are corruption — oracle #3).
        let final_insns = vec![
            Insn::move_(Operand::Local(1), Operand::imm(42), 0x1000),
            Insn::new(Op::Return, vec![Operand::Local(1)], 0x1004),
        ];
        let report = verify_method(
            &bytes,
            0x1000,
            &final_insns,
            &final_insns,
            None,
            None,
            true,
            false,
            false,
            false,
        );
        assert!(report.is_clean(), "{}", report.summarize());
    }

    #[test]
    fn coverage_flags_dropped_compare() {
        // cmp x0, x1 ; b.eq +8 ; ret  → IR has ONLY the return (dropped cmp)
        let cmp = 0xEB01001Fu32; // cmp x0, x1
        let beq = 0x54000100u32; // b.eq +8
        let ret = 0xD65F03C0u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&cmp.to_le_bytes());
        bytes.extend_from_slice(&beq.to_le_bytes());
        bytes.extend_from_slice(&ret.to_le_bytes());
        let final_insns = vec![Insn::new(Op::Return, vec![Operand::reg("x0")], 0x1008)];
        let report = verify_method(
            &bytes,
            0x1000,
            &final_insns,
            &final_insns,
            None,
            None,
            true,
            false,
            false,
            false,
        );
        assert!(!report.is_clean());
        assert_eq!(report.uncovered.len(), 2); // cmp + b.eq both uncovered
    }

    #[test]
    fn ledger_excuses_guard_removal() {
        // b.eq at 0x1000; the pipeline excised it. Final IR only has a
        // Return at a different address, so ONLY the ledger can excuse it.
        let b_eq = 0x54000100u32; // b.eq
        let bytes = b_eq.to_le_bytes().to_vec();
        ledger_clear();
        ledger_note(0x1000, LedgerKind::GuardExcision);
        let final_insns = vec![Insn::new(Op::Return, vec![Operand::Local(1)], 0x1004)];
        let report = verify_method(
            &bytes,
            0x1000,
            &final_insns,
            &final_insns,
            None,
            None,
            true,
            false,
            false,
            false,
        );
        ledger_clear();
        assert!(report.is_clean(), "{}", report.summarize());
    }

    #[test]
    fn cfg_trap_block_is_flagged() {
        // 3 blocks: 0 (cond → 1 or 2), 1 (self-loop via RVA 0x4, no exit),
        // 2 (return). Block 1 is the trap.
        let insns = vec![
            Insn::new(Op::CondJump(0x4), vec![Operand::reg("x0")], 0x0),
            Insn::new(Op::Jump(0x4), vec![], 0x4), // self-loop trap
            Insn::new(Op::Return, vec![Operand::reg("x0")], 0x8),
        ];
        let mut cfg_insns = insns.clone();
        let cfg = super::super::cfg::Cfg::build(&mut cfg_insns, &[0x0, 0x4, 0x8]).unwrap();
        let mut findings = Vec::new();
        check_cfg(&cfg, &cfg_insns, &mut findings);
        assert!(!findings.is_empty(), "self-loop trap must be flagged");
    }

    // ---- semantic diff: decoded native vs final IR (per-address) ----

    fn words(ws: &[u32]) -> Vec<u8> {
        let mut b = Vec::new();
        for w in ws {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b
    }

    fn cfg_of(insns: &[Insn]) -> (Vec<Insn>, super::super::cfg::Cfg) {
        let mut v = insns.to_vec();
        let addrs: Vec<u64> = v.iter().map(|i| i.address).collect();
        let cfg = super::super::cfg::Cfg::build(&mut v, &addrs).expect("cfg");
        (v, cfg)
    }

    #[test]
    fn semantic_diff_flags_inverted_condition() {
        // cmp x0, x1 ; b.eq +8 bytes ; ret — but the IR lifted it as LessThan.
        // Lifter shape: Compare+CondJump SHARE the branch address; the compare
        // site and the gap hold Nops so every decoded address has an IR op.
        let bytes = words(&[0xEB01001F, 0x54000020, 0xD65F03C0]);
        let final_insns = vec![
            Insn::new(
                Op::Compare(CompareKind::LessThan),
                vec![Operand::reg("t0"), Operand::reg("x0"), Operand::reg("x1")],
                0x1004,
            ),
            Insn::new(Op::CondJump(0x1008), vec![Operand::reg("t0")], 0x1004),
            Insn::new(Op::Return, vec![], 0x1008),
            Insn::nop(0x1000),
        ];
        let (finals, cfg) = cfg_of(&final_insns);
        let report = verify_method(
            &bytes,
            0x1000,
            &finals,
            &finals,
            Some(&cfg),
            None,
            true,
            false,
            false,
            false,
        );
        assert!(
            report
                .semantic_diffs
                .iter()
                .any(|(_, w)| w.contains("condition mismatch")),
            "{}",
            report.summarize()
        );
    }

    /// A taken edge whose native target IS an unconditional `jmp` (the
    /// tail-merge shape: `je shared_exit` where `shared_exit` is a `jmp` to the
    /// real exit) must be accepted. The IR landing is the `Jump` at exactly the
    /// native target; the jump-chain chase must compare the landing's own
    /// address FIRST, or it follows the jump and flags the deeper exit.
    #[test]
    fn semantic_diff_accepts_a_taken_target_that_is_itself_a_jump() {
        // 0x1000 je 0x1004 ; 0x1002 ret ; 0x1004 jmp 0x100a ; nops ; 0x100a ret
        let bytes = [
            0x74, 0x02, 0x90, 0x90, 0xEB, 0x04, 0x90, 0x90, 0x90, 0x90, 0xC3,
        ];
        let final_insns = vec![
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![Operand::Local(1), Operand::Local(2), Operand::imm(0)],
                0x1000,
            ),
            Insn::new(Op::CondJump(0x1004), vec![Operand::Local(1)], 0x1000),
            Insn::new(Op::Return, vec![], 0x1002),
            Insn::new(Op::Jump(0x100A), vec![], 0x1004),
            Insn::new(Op::Return, vec![], 0x100A),
        ];
        let (finals, cfg) = cfg_of(&final_insns);
        let report = verify_method(
            &bytes,
            0x1000,
            &finals,
            &finals,
            Some(&cfg),
            None,
            false,
            true,
            false,
            false,
        );
        assert!(
            report.semantic_diffs.is_empty(),
            "a landing on the exact native target must not be chased away: {}",
            report.summarize()
        );
    }

    /// A native taken target OUTSIDE the method body (IL2CPP's shared
    /// null-check throw: `test ecx,ecx; je <runtime handler>` outside the
    /// body) is resolved by `cfg` to the exit sentinel. The oracle cannot judge
    /// a landing the IR structurally cannot contain, so it must accept it.
    #[test]
    fn semantic_diff_accepts_an_out_of_body_taken_target() {
        // 0x1000 test ecx,ecx ; 0x1002 je 0x1083 (beyond the 5-byte body) ; ret
        let bytes = [0x85, 0xC9, 0x74, 0x7F, 0xC3];
        let final_insns = vec![
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![Operand::Local(1), Operand::Local(2), Operand::imm(0)],
                0x1000,
            ),
            Insn::new(Op::CondJump(0x1083), vec![Operand::Local(1)], 0x1000),
            Insn::new(Op::Return, vec![], 0x1004),
        ];
        let (finals, cfg) = cfg_of(&final_insns);
        let report = verify_method(
            &bytes,
            0x1000,
            &finals,
            &finals,
            Some(&cfg),
            None,
            false,
            true,
            false,
            false,
        );
        assert!(
            report.semantic_diffs.is_empty(),
            "an out-of-body target is not a landing the oracle can judge: {}",
            report.summarize()
        );
    }

    #[test]
    fn semantic_diff_clean_when_ir_matches_native() {
        // Same native body, but the Compare kind is Equal — no diffs expected.
        // b.eq imm19=2 → +8 bytes → 0x100C (the merge/return block).
        let bytes = words(&[0xEB01001F, 0x54000040, 0xD2800540, 0xD65F03C0]);
        let final_insns = vec![
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![Operand::reg("t0"), Operand::reg("x0"), Operand::reg("x1")],
                0x1004,
            ),
            Insn::new(Op::CondJump(0x100C), vec![Operand::reg("t0")], 0x1004),
            Insn::move_(Operand::reg("x0"), Operand::imm(42), 0x1008),
            Insn::new(Op::Return, vec![Operand::reg("x0")], 0x100C),
            Insn::nop(0x1000),
        ];
        let (finals, cfg) = cfg_of(&final_insns);
        let report = verify_method(
            &bytes,
            0x1000,
            &finals,
            &finals,
            Some(&cfg),
            None,
            true,
            false,
            false,
            false,
        );
        assert!(report.semantic_diffs.is_empty(), "{}", report.summarize());
    }

    #[test]
    fn semantic_diff_flags_wrong_arithmetic_kind() {
        // add x19, x0, x1 — but the IR lifted it as Sub. Note yaxpeax decodes
        // the third operand as RegShift(LSL, 0, X, 1): the shift-0 gpr rule.
        let bytes = words(&[0x8B010013, 0xD65F03C0]);
        let final_insns = vec![
            Insn::new(
                Op::Arith(ArithKind::Sub),
                vec![Operand::reg("x19"), Operand::reg("x0"), Operand::reg("x1")],
                0x1000,
            ),
            Insn::new(Op::Return, vec![], 0x1004),
        ];
        let (finals, cfg) = cfg_of(&final_insns);
        let report = verify_method(
            &bytes,
            0x1000,
            &finals,
            &finals,
            Some(&cfg),
            None,
            true,
            false,
            false,
            false,
        );
        assert!(
            report
                .semantic_diffs
                .iter()
                .any(|(_, w)| w.contains("arithmetic mismatch")),
            "{}",
            report.summarize()
        );
    }

    #[test]
    fn semantic_diff_flags_wrong_memory_offset() {
        // ldr w1, [x0, #0x10] — the IR says +0x20 (wrong field).
        let bytes = words(&[0xB9401001, 0xD65F03C0]);
        let final_insns = vec![
            Insn::move_(
                Operand::reg("w1"),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::reg("x0")),
                    index: None,
                    scale: 1,
                    addend: 0x20,
                },
                0x1000,
            ),
            Insn::new(Op::Return, vec![], 0x1004),
        ];
        let (finals, cfg) = cfg_of(&final_insns);
        let report = verify_method(
            &bytes,
            0x1000,
            &finals,
            &finals,
            Some(&cfg),
            None,
            true,
            false,
            false,
            false,
        );
        assert!(
            report
                .semantic_diffs
                .iter()
                .any(|(_, w)| w.contains("memory offset mismatch")),
            "{}",
            report.summarize()
        );
    }

    #[test]
    fn semantic_diff_flags_wrong_call_target() {
        // bl +8 (native target 0x1008) — the IR call points elsewhere and the
        // resolver did NOT name it (no Method operand).
        let bytes = words(&[0x94000002, 0xD65F03C0, 0xD65F03C0]);
        let final_insns = vec![
            Insn::new(Op::Call, vec![Operand::imm(0x9999)], 0x1000),
            Insn::new(Op::Return, vec![], 0x1004),
            Insn::new(Op::Return, vec![], 0x1008),
        ];
        let (finals, cfg) = cfg_of(&final_insns);
        let report = verify_method(
            &bytes,
            0x1000,
            &finals,
            &finals,
            Some(&cfg),
            None,
            true,
            false,
            false,
            false,
        );
        assert!(
            report
                .semantic_diffs
                .iter()
                .any(|(_, w)| w.contains("call target mismatch")),
            "{}",
            report.summarize()
        );
    }

    // -----------------------------------------------------------------------
    // ARM32 / x86 ground truth
    //
    // Both architectures used to fill `NativeFacts::default()`, so the semantic
    // diff — the only oracle that asks whether an op is the RIGHT op — could not
    // run on them at all. ARM32 is what every 32-bit Android il2cpp binary is
    // built for, so the entire Android corpus was outside the fidelity
    // measurement. These tests pin the extractors to real encodings.
    // -----------------------------------------------------------------------

    /// Branch arithmetic is the highest-risk part of an ARM32 extractor: A32
    /// branches are PC-relative with an architectural +8 bias, and getting the
    /// convention wrong would make every single target look corrupt.
    /// yaxpeax folds the bias in as `(imm24 + 2) << 2`, so `iaddr + (off << 2)`
    /// must equal the byte target.
    #[test]
    fn facts_arm32_branch_and_call_targets() {
        // `bne #16` (cond=NE, imm24=2) then `bl #16`
        let mut b = Vec::new();
        b.extend_from_slice(&0x1A00_0002u32.to_le_bytes());
        b.extend_from_slice(&0xEB00_0002u32.to_le_bytes());
        let (ins, fails) = decode_all_arm32(&b, 0x1000, false);
        assert_eq!(fails, 0);
        assert_eq!(ins.len(), 2);
        assert_eq!(ins[0].facts.cond_branch, Some((1, 0x1010)));
        assert_eq!(ins[1].facts.call, Some((0x1014, false)));
    }

    /// The A32 condition code must land in the same numeric space as A64, which
    /// is what lets one `lift::cond_to_kind` serve both. Every condition is
    /// checked, because a single mis-numbered code would make an entire class of
    /// correct branches look inverted.
    #[test]
    fn facts_arm32_conditions_match_lift_kinds() {
        let cases: &[(u32, CompareKind)] = &[
            (0x0, CompareKind::Equal),          // beq
            (0x1, CompareKind::NotEqual),       // bne
            (0x2, CompareKind::GreaterOrEqual), // bhs / bcs
            (0x3, CompareKind::LessThan),       // blo / bcc
            (0x8, CompareKind::GreaterThan),    // bhi
            (0x9, CompareKind::LessOrEqual),    // bls
            (0xA, CompareKind::GreaterOrEqual), // bge
            (0xB, CompareKind::LessThan),       // blt
            (0xC, CompareKind::GreaterThan),    // bgt
            (0xD, CompareKind::LessOrEqual),    // ble
        ];
        for &(cond, want) in cases {
            let word = (cond << 28) | 0x0A00_0000u32; // b<cc> #0
            let (ins, _) = decode_all_arm32(&word.to_le_bytes(), 0x4000, false);
            let (code, _) = ins[0]
                .facts
                .cond_branch
                .unwrap_or_else(|| panic!("cond {cond:#x} produced no condition"));
            assert_eq!(
                super::super::lift::cond_to_kind(code),
                want,
                "cond {cond:#x} canonicalized to {code}"
            );
        }
    }

    /// `ldr r1, [r0, #4]` / `ldr r1, [r0, #-4]`. The `add` flag on the pre-index
    /// form has to be signed into the addend, or every negative field offset
    /// would be compared against a positive native value.
    #[test]
    fn facts_arm32_memory_offset_is_signed() {
        let mut b = Vec::new();
        b.extend_from_slice(&0xE590_1004u32.to_le_bytes()); // ldr r1, [r0, #4]
        b.extend_from_slice(&0xE510_1004u32.to_le_bytes()); // ldr r1, [r0, #-4]
        let (ins, _) = decode_all_arm32(&b, 0x2000, false);
        assert_eq!(ins[0].facts.mem, Some((0, 4)));
        assert_eq!(ins[1].facts.mem, Some((0, -4)));
    }

    /// `strh w12, [x9], #-2` accesses the old x9 address and then updates x9.
    /// Its immediate is writeback, not a displacement on the memory access.
    /// This is the exact form used by the two iOS semantic-diff cases.
    #[test]
    fn facts_arm64_postindex_memory_offset_is_zero() {
        let (ins, fails) = decode_all_arm64(&0x781F_E52Cu32.to_le_bytes(), 0x1000);
        assert_eq!(fails, 0);
        assert_eq!(ins.len(), 1);
        assert_eq!(ins[0].facts.mem, Some((9, 0)));
    }

    /// x86 condition codes must canonicalize so that `lift::cond_to_kind`
    /// yields exactly what the x86 lifter's own `cond_kind` emits. This lifter
    /// collapses `be`/`le` to a signed `LessThan` and `a`/`g`/`ae`/`ge` to
    /// `GreaterOrEqual`, so mapping them to ARM's unsigned `LS`/`LE`/`HI`/`GT`
    /// slots would flag every correct branch as corruption.
    #[test]
    fn facts_x86_conditions_match_lift_kinds() {
        let cases: &[(u8, CompareKind)] = &[
            (0x74, CompareKind::Equal),          // je
            (0x75, CompareKind::NotEqual),       // jne
            (0x72, CompareKind::LessThan),       // jb
            (0x73, CompareKind::GreaterOrEqual), // jae
            (0x76, CompareKind::LessThan),       // jbe
            (0x77, CompareKind::GreaterOrEqual), // ja
            (0x7C, CompareKind::LessThan),       // jl
            (0x7D, CompareKind::GreaterOrEqual), // jge
            (0x7E, CompareKind::LessThan),       // jle
            (0x7F, CompareKind::GreaterOrEqual), // jg
        ];
        for &(op, want) in cases {
            let (ins, _) = decode_all_x86(&[op, 0x00], 0x3000);
            let (code, _) = ins[0]
                .facts
                .cond_branch
                .unwrap_or_else(|| panic!("opcode {op:#x} produced no condition"));
            assert_eq!(
                super::super::lift::cond_to_kind(code),
                want,
                "opcode {op:#x} canonicalized to {code}"
            );
        }
    }

    /// One instruction this crate cannot decode must not shift the walk.
    ///
    /// These are the real bytes of method `0x53c808`: `vldr s1,[fp,#12]`
    /// (`0xED9B1A03` — undecodable in yaxpeax-arm 0.4), `mov r2,#0`, then
    /// `bl #…`. Advancing 2 bytes on the failure made the walk slip to
    /// `0x1002`, so every later instruction was decoded from a half-word offset
    /// and phantom conditional branches appeared at `0x53c846`-style addresses —
    /// which is exactly what a 2,405-finding "condition dropped" census was
    /// built from.
    #[test]
    fn arm32_decode_does_not_desync_on_an_undecodable_instruction() {
        let mut b = Vec::new();
        for w in [0xED9B1A03u32, 0xE3A0_2000, 0xEB42_0994] {
            b.extend_from_slice(&w.to_le_bytes());
        }
        let (ins, fails) = decode_all_arm32(&b, 0x1000, false);
        // Pin the premise. If a future yaxpeax learns to decode VFP this test
        // stops exercising the failure path, and silently passing is worse than
        // failing — pick another encoding the crate rejects at that point.
        assert!(fails >= 1, "expected at least one undecodable instruction");
        for d in &ins {
            assert_eq!(
                (d.addr - 0x1000) % 4,
                0,
                "address {:#x} is not on a word boundary — the walk slipped",
                d.addr
            );
        }
        // Nothing in this sequence is a conditional branch, so any is invented.
        assert!(
            ins.iter().all(|d| d.facts.cond_branch.is_none()),
            "phantom conditional branch in {:?}",
            ins.iter()
                .map(|d| (d.addr, d.mnemonic.clone()))
                .collect::<Vec<_>>()
        );
    }

    /// The oracle must classify ARM32 with the LIFTER'S own splitter. The old
    /// local version only removed a condition when the base mnemonic was one
    /// character, so the flag-setting `s` (`cmps`, `tsts`) and every
    /// multi-character predicated form (`ldreq`, `strne`, `ldrvs`) reached the
    /// classifier intact and fell to `Other` — thousands of real instructions
    /// counted as unknown, and therefore as uncovered semantics.
    #[test]
    fn arm32_classifier_handles_flag_suffix_and_predication() {
        let cases: &[(&str, InsnClass)] = &[
            ("cmps", InsnClass::Arith),
            ("tsts", InsnClass::Arith),
            ("andeq", InsnClass::Arith),
            ("muleq", InsnClass::Arith),
            ("addne", InsnClass::Arith),
            ("ldreq", InsnClass::LoadStore),
            ("strne", InsnClass::LoadStore),
            ("ldrvs", InsnClass::LoadStore),
            // Coprocessor no-ops the lifter deliberately drops: `Pseudo` so the
            // coverage census does not demand an operation at their address.
            ("cdp2", InsnClass::Pseudo),
            ("mcr2", InsnClass::Pseudo),
            // `bls` is `b`+`ls` (a branch), NOT `bl`+`s` — the ambiguity the
            // lifter's splitter exists to resolve.
            ("bls", InsnClass::Branch),
            ("bleq", InsnClass::Call),
        ];
        for &(m, want) in cases {
            assert_eq!(classify_arm32(m), want, "mnemonic {m}");
        }
    }

    /// `call rel32=0` (lands on the next instruction) and `je rel8=1`.
    #[test]
    fn facts_x86_branch_and_call_targets() {
        let (ins, fails) = decode_all_x86(&[0xE8, 0, 0, 0, 0, 0x74, 0x01], 0x1000);
        assert_eq!(fails, 0);
        assert_eq!(ins.len(), 2);
        assert_eq!(ins[0].facts.call, Some((0x1005, false)));
        assert_eq!(ins[1].facts.cond_branch, Some((0, 0x1008)));
    }

    /// `mov rax, [rax + rbx*4 + 0x10]` — a register-indexed access, which is
    /// what the `IndexDropped` conformance class keys on.
    #[test]
    fn facts_x86_memory_base_and_index() {
        let (ins, _) = decode_all_x86(&[0x48, 0x8B, 0x44, 0x98, 0x10], 0x1000);
        assert_eq!(ins.len(), 1);
        let (base, index) = ins[0].facts.mem_indexed.expect("indexed operand");
        assert_eq!(base as u32, iced_x86::Register::RAX as u32);
        assert_eq!(index as u32, iced_x86::Register::RBX as u32);
    }

    /// The wiring proof: an ARM32 method whose IR has the wrong branch polarity
    /// must be reported. Before the ARM32 facts extractor existed this method
    /// produced NO semantic findings — the diff could not run without facts.
    #[test]
    fn arm32_semantic_diff_reports_wrong_condition() {
        // `cmp r0, #0` ; `bne .` (imm24 = 0xFFFFFE → offset 0 → own address)
        let mut b = Vec::new();
        b.extend_from_slice(&0xE350_0000u32.to_le_bytes());
        b.extend_from_slice(&0x1AFF_FFFEu32.to_le_bytes());
        let base = 0x5000;
        let final_insns = vec![
            // WRONG polarity: the native condition is NE, the IR says Equal.
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![Operand::Local(1), Operand::Local(2)],
                base + 4,
            ),
            // Raw RVA, as the lifter emits before `Cfg::build` rewrites it.
            Insn::new(
                Op::CondJump((base + 4) as usize),
                vec![Operand::Local(1)],
                base + 4,
            ),
            Insn::new(Op::Return, vec![], base + 8),
        ];
        let (finals, cfg) = cfg_of(&final_insns);
        let report = verify_method(
            &b,
            base,
            &finals,
            &finals,
            Some(&cfg),
            None,
            false,
            false,
            true,
            false,
        );
        assert!(
            report
                .semantic_diffs
                .iter()
                .any(|(_, w)| w.contains("condition mismatch")),
            "{}",
            report.summarize()
        );
    }

    /// `[abs + K]` — the metadata-blob addressing the lifter uses for
    /// absolute loads/stores.
    fn abs_mem(addend: i64) -> Operand {
        Operand::Memory {
            base: Box::new(Operand::Register("abs".into())),
            index: None,
            scale: 1,
            addend,
            width: 0,
        }
    }

    /// A store to an address an earlier `Move` also loaded is a genuine SECOND
    /// access. Regression: the re-materialization dedup charged it as a
    /// duplicate because the address matched a `Move` SOURCE, which made
    /// ordinary read-then-write code look like a lost access.
    #[test]
    fn a_store_to_a_loaded_address_is_not_deduped() {
        let insns = vec![
            Insn::move_(Operand::Local(1), abs_mem(9905849), 0x1000),
            Insn::move_(abs_mem(9905849), Operand::Immediate(1), 0x1004),
        ];
        assert_eq!(
            SemanticFingerprint::of(&insns).loads_stores,
            2,
            "the load and the store are two accesses"
        );
    }

    /// A CALL argument that repeats a `Move` source is that load
    /// re-materialized, not a second access — the defining `Move` is the one
    /// that counts.
    #[test]
    fn a_call_argument_repeating_a_move_source_is_deduped() {
        let insns = vec![
            Insn::move_(Operand::Register("x0".into()), abs_mem(7406364), 0x1000),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Immediate(1),
                    Operand::Register("rv".into()),
                    abs_mem(7406364),
                ],
                0x1004,
            ),
        ];
        assert_eq!(SemanticFingerprint::of(&insns).loads_stores, 1);
    }

    /// The dedup is not call-specific: ANY use site that repeats a `Move` source
    /// verbatim is the same load re-materialized (verified on 0x1794D0).
    #[test]
    fn a_memory_operand_repeated_at_a_non_call_use_site_is_not_a_second_access() {
        let insns = vec![
            Insn::move_(Operand::Register("x0".into()), abs_mem(9912137), 0x1000),
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![Operand::Local(3), abs_mem(9912137), Operand::Immediate(0)],
                0x1004,
            ),
        ];
        assert_eq!(SemanticFingerprint::of(&insns).loads_stores, 1);
    }

    /// ONE named metadata string fanned out to several call sites is one
    /// constant, not three accesses (verified on 0x40E0F0, where a single raw
    /// load surfaced as three `"N"` arguments after propagation).
    #[test]
    fn a_string_literal_fanned_out_to_several_call_sites_counts_once() {
        let lit = Operand::StringLiteral {
            index: 2466,
            value: "N".into(),
        };
        let call = |target: i64, addr: u64| {
            Insn::new(
                Op::Call,
                vec![
                    Operand::Immediate(target),
                    Operand::Register("rv".into()),
                    lit.clone(),
                ],
                addr,
            )
        };
        let insns = vec![call(1, 0x1000), call(2, 0x1004), call(3, 0x1008)];
        assert_eq!(
            SemanticFingerprint::of(&insns).literals,
            1,
            "one named constant, three uses"
        );
    }

    /// Distinct constants still count separately — the dedup is by VALUE, not a
    /// blanket one-literal-per-method cap.
    #[test]
    fn distinct_string_literals_are_counted_separately() {
        let a = Operand::StringLiteral {
            index: 1,
            value: "A".into(),
        };
        let b = Operand::StringLiteral {
            index: 2,
            value: "B".into(),
        };
        let call = |lit: Operand, addr: u64| {
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(1), Operand::Register("rv".into()), lit],
                addr,
            )
        };
        assert_eq!(
            SemanticFingerprint::of(&[call(a, 0x1000), call(b, 0x1004)]).literals,
            2
        );
    }

    /// The tail-call allowance is an exact budget: one extra access is excused
    /// per tail transfer, and the second is reported as synthesis.
    #[test]
    fn drift_reports_memory_growth_beyond_the_tail_call_allowance() {
        let pre = SemanticFingerprint::of(&[Insn::move_(
            Operand::Register("x0".into()),
            abs_mem(1),
            0x1000,
        )]);
        let post = SemanticFingerprint::of(&[
            Insn::move_(Operand::Register("x0".into()), abs_mem(1), 0x1000),
            Insn::move_(Operand::Register("x1".into()), abs_mem(2), 0x1004),
        ]);
        assert_eq!(
            pre.drift(&post, 0).len(),
            1,
            "{}→{}",
            pre.loads_stores,
            post.loads_stores
        );
        assert!(
            pre.drift(&post, 1).is_empty(),
            "one tail transfer excuses exactly one access"
        );
    }

    /// The tail-merge splicer appends a SHARED out-of-body epilogue to the
    /// listing. That code executes, but it was never in the raw lift, so the
    /// A/A comparison must not charge it as synthesised control flow.
    /// Reproduces the `0x103ac` shape (iOS): raw 1 call / 0 compare / 0 arith,
    /// final 2 / 1 / 1, ALL of the growth living past the declared body end.
    #[test]
    fn out_of_body_spliced_tail_is_not_charged_as_synthesis() {
        let body = [
            Insn::move_(
                Operand::Register("x0".into()),
                Operand::Register("x1".into()),
                0x1000,
            ),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(1), Operand::Register("x0".into())],
                0x1004,
            ),
            Insn::new(Op::Jump(0x2000), vec![], 0x1008),
        ];
        let mut final_insns = body.to_vec();
        final_insns.extend([
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(2), Operand::Register("x0".into())],
                0x2000,
            ),
            Insn::new(
                Op::Not,
                vec![
                    Operand::Register("x1".into()),
                    Operand::Register("x2".into()),
                ],
                0x2004,
            ),
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![Operand::Local(1), Operand::Local(2), Operand::Immediate(0)],
                0x2008,
            ),
        ]);
        let pre = SemanticFingerprint::of(&body);
        assert!(
            !pre.drift(&SemanticFingerprint::of(&final_insns), 0)
                .is_empty(),
            "without the body filter the spliced epilogue reads as synthesis"
        );
        let filtered = SemanticFingerprint::of_inside(&final_insns, 0x1000, 0x1020);
        let filtered_drift = pre.drift(&filtered, 0);
        assert!(
            filtered_drift.is_empty(),
            "out-of-body tail-merge code is not this method's synthesis: {filtered_drift:?}"
        );
        // Growth INSIDE the body is still policed.
        final_insns.push(Insn::new(
            Op::Call,
            vec![Operand::Immediate(3), Operand::Register("x0".into())],
            0x100c,
        ));
        let with_extra = SemanticFingerprint::of_inside(&final_insns, 0x1000, 0x1020);
        assert_eq!(
            pre.drift(&with_extra, 0).len(),
            1,
            "an extra call inside the body is still flagged"
        );
    }
}

/// The severity buckets, and the reason they exist: an `uncovered`
/// instruction is the single most common finding and is usually benign, so
/// "flagged" cannot be read as "wrong".
#[cfg(test)]
mod verdict_tests {
    use super::*;

    fn absorbed_arith() -> DecodedInsn {
        DecodedInsn {
            addr: 0x10,
            size: 4,
            mnemonic: "add".into(),
            class: InsnClass::Arith,
            facts: Default::default(),
        }
    }

    #[test]
    fn no_findings_at_all_is_clean() {
        assert_eq!(VerifyReport::default().verdict(), Verdict::Clean);
    }

    #[test]
    fn an_instruction_the_optimizer_absorbed_is_not_wrong_logic() {
        // `test` feeding a branch, a folded `add`, a nop: the IR covers these
        // by construction. On their own they say nothing about correctness,
        // and conflating them with a wrong branch target is exactly what made
        // the old aggregate report 12,585 of 12,615 flagged.
        let mut r = VerifyReport::default();
        r.uncovered.push(absorbed_arith());
        assert_eq!(r.verdict(), Verdict::CoverageOnly);
        // still not clean — the coverage oracle must keep reporting it
        assert!(!r.is_clean());
    }

    #[test]
    fn a_wrong_branch_target_is_wrong_logic() {
        let mut r = VerifyReport::default();
        r.uncovered.push(absorbed_arith());
        r.semantic_diffs
            .push((0x20, "branch target mismatch".into()));
        assert_eq!(r.verdict(), Verdict::WrongLogic);
    }

    #[test]
    fn a_cfg_the_bytes_cannot_produce_is_wrong_logic() {
        let mut r = VerifyReport::default();
        r.cfg_findings.push("successor out of range".into());
        assert_eq!(r.verdict(), Verdict::WrongLogic);
    }

    #[test]
    fn corruption_shaped_findings_are_suspect_not_proven_wrong() {
        let mut undef = VerifyReport::default();
        undef.undefined_locals.push(41);
        assert_eq!(undef.verdict(), Verdict::Suspect);

        let mut nocond = VerifyReport::default();
        nocond.missing_conditions.push(0x30);
        assert_eq!(nocond.verdict(), Verdict::Suspect);

        let mut unres = VerifyReport::default();
        unres.unresolved_calls.push((0x40, "no target".into()));
        assert_eq!(unres.verdict(), Verdict::Suspect);

        let mut synth = VerifyReport::default();
        synth.aa_findings.push("calls GREW 1->2".into());
        assert_eq!(synth.verdict(), Verdict::Suspect);
    }

    #[test]
    fn a_proven_contradiction_outranks_a_suspect_finding() {
        let mut r = VerifyReport::default();
        r.undefined_locals.push(41);
        r.semantic_diffs
            .push((0x20, "arithmetic kind mismatch".into()));
        assert_eq!(r.verdict(), Verdict::WrongLogic);
    }

    #[test]
    fn short_reason_is_none_only_when_clean() {
        assert_eq!(VerifyReport::default().short_reason(), None);

        let mut r = VerifyReport::default();
        r.undefined_locals.push(41);
        r.undefined_locals.push(42);
        r.unresolved_calls.push((0x40, "no target".into()));
        let why = r.short_reason().expect("a flagged method must say why");
        // The renderer embeds this verbatim; it must name the counts a reader
        // needs to distrust the right lines, without the full `summarize` dump.
        assert!(why.contains("2 undefined local"), "{why}");
        assert!(why.contains("1 unresolved call"), "{why}");
        assert!(!why.contains('\n'), "one header line only: {why}");
    }

    #[test]
    fn annotations_round_trip_and_are_absent_when_unset() {
        // Unique addresses: this static map is shared across the parallel test
        // threads, so a fixed address could collide with another test's entry.
        let clean = 0xDEAD_0000_0000_0001u64;
        let flagged = 0xDEAD_0000_0000_0002u64;
        assert_eq!(annotation_for(clean), None);
        annotate_method(flagged, Verdict::WrongLogic, "1 semantic diff".into());
        assert_eq!(
            annotation_for(clean),
            None,
            "only flagged methods are recorded"
        );
        let (v, why) = annotation_for(flagged).expect("recorded grade must round-trip");
        assert_eq!(v, Verdict::WrongLogic);
        assert_eq!(why, "1 semantic diff");
    }

    #[test]
    fn verdict_names_match_the_dump_banner_vocabulary() {
        assert_eq!(verdict_name(Verdict::Clean), "clean");
        assert_eq!(verdict_name(Verdict::WrongLogic), "WRONG-LOGIC");
        assert_eq!(verdict_name(Verdict::Suspect), "suspect");
        assert_eq!(verdict_name(Verdict::CoverageOnly), "coverage-only");
    }
}
