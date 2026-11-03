//! High-variable recovery — a port of Ghidra's `merge.cc` / `varmap.cc` value
//! merging (the `HighVariable` live range) onto this IR.
//!
//! An earlier revision of this comment also named Ghidra's `unify.cc` as the
//! model. It is not: `unify.cc` is the p-code RULE-DSL constraint matcher
//! (`#ifdef CPUI_RULECOMPILE`, driven by `rulecompile` / `ruleparse` — its
//! `RHSConstant` subclasses are `NZMask`, `Consumed`, `Offset`,
//! `VarnodeSize`, `IsConstant`), not a value- or type-unification pass. Of
//! the files the vendored Ghidra tree does contain, the merge machinery and
//! its pure-computation comparison are the real counterpart; rule C below is
//! this project's own proof over the listing.
//!
//! ## The defect it exists for
//!
//! `scripts/audit_dump.py` counts "undefined locals (v*)" as locals that are
//! READ in a method body but never ASSIGNED anywhere in it. On the x64 corpus
//! that is 351,360 references across 28,676 of 47,924 methods (59.8%) — the
//! largest single defect class in the dump. The verifier's own oracle
//! (`verify.rs`, "Corruption oracle #1") defines the same thing on the IR: a
//! `Local(v)` read by a reachable instruction that no reachable instruction
//! defines and that no seed excuses.
//!
//! Two mechanisms produce them, and both are *value* problems, not printing
//! problems:
//!
//! 1. **Copy chains.** SSA renames every register reuse into a fresh version,
//!    so a value the source held in one variable arrives as
//!    `v3 = v2; v2 = v1; use v3`. Each link that the renderer cannot inline
//!    prints a NAME, and if the head of the chain was never defined the whole
//!    chain prints names nothing writes.
//! 2. **Phi confluences.** A join whose two incoming values are the SAME value
//!    still gets a phi (SSA has no notion of "these two definitions agree"), so
//!    the merged version prints by name even though one argument is already the
//!    answer. `1` is where most refusals come from.
//!
//! Ghidra solves both in one place: it merges SSA varnodes into HIGH VARIABLES
//! (live ranges that provably hold one value), which is what lets it print a
//! value where a name would otherwise be forced.
//!
//! ## The rules, and why each is sound
//!
//! Every rule below is proven from the listing alone; a failed proof is a
//! NO-OP that leaves the (correct, uglier) machine form intact. The two
//! invariants of the surrounding pipeline apply unchanged and are not
//! negotiable:
//!
//! * **Never invent a value.** A wrong value is far worse than a bare `vN`.
//! * **Never move a write.** Substitution is applied only when the surviving
//!   definition provably DOMINATES every use of the version it replaces.
//!
//! * `rule A — trivial copies` (Ghidra `Merge::mergeTestRequired`, the
//!   `COPY` case): `d = s` where `d` has exactly ONE definition in the listing
//!   (this move), `s` dominates every use of `d`, and neither side is
//!   `this`/a parameter/an address-taken cell — then every use of `d` is the
//!   value of `s`, and the move is dead. Requires the dominance proof because
//!   our SSA is not perfect SSA: the rotated-loop transform and the
//!   pre-assigned phi destinations both re-write versions after renaming.
//! * `rule B — phi agreement` (Ghidra `Funcdata::opPhi` merge): a phi whose
//!   every non-`Unknown` argument resolves to ONE version, with no `Unknown`
//!   argument, and where that version dominates the phi's block — the phi IS
//!   that version. A single `Unknown` argument DISQUALIFIES the phi: the value
//!   differs on that path, and claiming otherwise would be an invention.
//! * `rule C — value unification` (our own proof — see the note at the top
//!   of this module): two versions with
//!   exactly one definition each, where the two definitions are the same PURE
//!   computation over the same (alias-resolved) operands, and the earlier one
//!   dominates every use of the later one — the later value is the earlier
//!   value. Purity is checked structurally: an operation that reads MEMORY
//!   (`Memory`, `Field`, `InstanceField`, `ArrayAccess`, `ArrayLength`) or
//!   that can have effects is never unified, because two identical loads are
//!   not the same value unless nothing wrote in between.
//! * `rule D — high-variable naming` (Ghidra `Merge::mergeTestAdjacent`): two
//!   DEFINED versions of the same register slot whose live spans do not
//!   overlap are one high variable, so they print one name. This is a NAMING
//!   decision only — the IR operands are untouched, so no value can change —
//!   and disjoint spans are what make the reuse visible-but-unambiguous, which
//!   is what `merge.cc` does with non-interfering varnodes of one storage.
//!
//! ## What this module does NOT do
//!
//! It does not name values, and it cannot fix a definition that a later pass
//! (the renderer's own dead-code idioms) drops: those act on the statement
//! tree after this pass. It reports what is left (see [`HighVarStats`]) and marks
//! the versions the ABI proves are INCOMING — an entry value of an
//! argument/callee-saved register is not a lost definition, which is exactly
//! what `LocalInfo::is_incoming_arg` documents.
//!
//! **The text-level `v*` count is a RENDER question, and it is only partly this
//! pass's job.** The audit counts a `vN` that the method never assigns; a
//! read-but-never-defined version renders `vN` only when the renderer declines
//! to name it. `MethodCtx::entry_register_name` used to name exactly ONE shape —
//! a plain `xN`/`wN` origin with N ≥ 1 and no definition anywhere — and return
//! `None` (hence `vN`) for EVERY other origin, which covered the FP views
//! (`vf`/`vb`/`vh`/`vs`/`vd`/`vq`/`s`/`d`/`q`), `rv`, `sp`, `fp` and the ARM32
//! `gN` names; that predicate has since been widened to every register family,
//! so those now render as `in_vs3` / `in_rv` / `in_sp` / `in_g14`.
//!
//! What is left as a bare `vN` is what this pass could NOT excuse, and the run
//! report now breaks that residue down **by origin** ([`report`]), because the
//! origin names the mechanism: a caller-saved register means a dropped write, a
//! local slot means a definition the pipeline lost, and an `abs`/`rip` read is
//! not a defect at all. Counting those three families as one number is what made
//! the residue impossible to act on.

use super::cfg::Cfg;
use super::dominator::DominatorInfo;
use super::ir::{Insn, Op, Operand};
use super::locals::LocalTable;
use super::types::ManagedType;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// How many times the rule set is re-run. Each round can enable another rule
/// (rule C can make two phi arguments agree, which then fires rule B), so the
/// set runs to a fixpoint — bounded, because every round strictly shrinks the
/// set of versions that still have a definition of their own.
const MAX_ROUNDS: usize = 8;

/// Longest alias chain `resolve_target` will follow before declaring a cycle.
const MAX_ALIAS_DEPTH: usize = 64;

/// What the pass did, so a run report can show it and a measurement can
/// attribute a delta. Every field is a count of *rewrites*, not of attempts.
#[derive(Debug, Default, Clone)]
pub struct HighVarStats {
    /// `d = s` copies folded away (rule A), plus the `Move(v, v)` identities
    /// erased afterwards (whether this pass created them or the listing already
    /// contained one).
    pub copies_folded: usize,
    /// Phis collapsed to one agreeing argument (rule B).
    pub phis_coalesced: usize,
    /// Versions unified with an earlier identical computation (rule C).
    pub values_unified: usize,
    /// Versions whose name was merged with a non-overlapping earlier version
    /// of the same register slot (rule D). Naming only — no IR change.
    pub names_merged: usize,
    /// Read-but-never-defined versions proven to be INCOMING ABI values and
    /// marked `is_incoming_arg` (so the oracle stops calling them corruption).
    pub incoming_marked: usize,
    /// Read-but-never-defined versions excused because the pipeline itself SEEDS
    /// the value on entry: `this`, a parameter, the indirect-result buffer, or
    /// the entry value of `sp`/`fp`/`rv`/`abs`/`rip`/a stack cell.
    ///
    /// These are not lost definitions, and counting them as such is how the
    /// residue number got inflated: `seeded_versions` roots the `abs`/`rip`
    /// entry values, but the origin test in the tally did not list them, so every
    /// absolute-address base read was reported as a lost definition by the very
    /// pass that seeds it.
    pub excused_on_entry: usize,
    /// Read-but-never-defined versions with NO `LocalTable` entry at all — SSA
    /// scratch slots that nothing renders by name. Counted apart so they can
    /// never be mistaken for a lost value.
    pub undefined_unattributed: usize,
    /// Read-but-never-defined versions that are NOT provably incoming or seeded,
    /// i.e. a genuinely lost definition. The residue this pass could not fix.
    pub undefined_remaining: usize,
    /// Slot origin → count, for `undefined_remaining`.
    pub undefined_origins: BTreeMap<String, usize>,
    /// Instructions removed by the alias application step (the union of rules A,
    /// B and C — every version rewritten to the version it provably equals).
    pub aliases_erased: usize,
    /// Rules that were applied in the last round (for the run report).
    pub rounds: usize,
}

// ---------------------------------------------------------------------------
// Run-report counters (house style: `reconstruct::report()`, `idioms::report()`)
// ---------------------------------------------------------------------------

static COPIES_FOLDED: AtomicU64 = AtomicU64::new(0);
static PHIS_COALESCED: AtomicU64 = AtomicU64::new(0);
static VALUES_UNIFIED: AtomicU64 = AtomicU64::new(0);
static NAMES_MERGED: AtomicU64 = AtomicU64::new(0);
static INCOMING_MARKED: AtomicU64 = AtomicU64::new(0);
static EXCUSED_ON_ENTRY: AtomicU64 = AtomicU64::new(0);
static UNDEFINED_UNATTRIBUTED: AtomicU64 = AtomicU64::new(0);
static UNDEFINED_LEFT: AtomicU64 = AtomicU64::new(0);
static METHODS: AtomicU64 = AtomicU64::new(0);
/// origin → count over the whole run. A `BTreeMap` behind one `Mutex`, because
/// the run report is printed from a single thread after the worker pool is done
/// and a `BTreeMap` merge is order-independent — so the printed breakdown is
/// deterministic however the methods interleave.
static UNDEFINED_ORIGINS: OnceLock<Mutex<BTreeMap<String, u64>>> = OnceLock::new();

fn record(s: &HighVarStats) {
    METHODS.fetch_add(1, Ordering::Relaxed);
    COPIES_FOLDED.fetch_add(s.copies_folded as u64, Ordering::Relaxed);
    PHIS_COALESCED.fetch_add(s.phis_coalesced as u64, Ordering::Relaxed);
    VALUES_UNIFIED.fetch_add(s.values_unified as u64, Ordering::Relaxed);
    NAMES_MERGED.fetch_add(s.names_merged as u64, Ordering::Relaxed);
    INCOMING_MARKED.fetch_add(s.incoming_marked as u64, Ordering::Relaxed);
    EXCUSED_ON_ENTRY.fetch_add(s.excused_on_entry as u64, Ordering::Relaxed);
    UNDEFINED_UNATTRIBUTED.fetch_add(s.undefined_unattributed as u64, Ordering::Relaxed);
    UNDEFINED_LEFT.fetch_add(s.undefined_remaining as u64, Ordering::Relaxed);
    if !s.undefined_origins.is_empty() {
        let m = UNDEFINED_ORIGINS.get_or_init(|| Mutex::new(BTreeMap::new()));
        if let Ok(mut g) = m.lock() {
            for (o, c) in &s.undefined_origins {
                *g.entry(o.clone()).or_insert(0) += *c as u64;
            }
        }
    }
}

/// One line per counter, or an empty string when nothing fired — so a
/// recognizer that is silently inert is visible instead of assumed to work.
pub fn report() -> String {
    let methods = METHODS.load(Ordering::Relaxed);
    if methods == 0 {
        return String::new();
    }
    let mut s = String::new();
    let c = COPIES_FOLDED.load(Ordering::Relaxed);
    if c > 0 {
        s.push_str(&format!("[highvar] copy chains folded: {c}\n"));
    }
    let p = PHIS_COALESCED.load(Ordering::Relaxed);
    if p > 0 {
        s.push_str(&format!("[highvar] phis coalesced to one argument: {p}\n"));
    }
    let v = VALUES_UNIFIED.load(Ordering::Relaxed);
    if v > 0 {
        s.push_str(&format!(
            "[highvar] values unified with an earlier computation: {v}\n"
        ));
    }
    let n = NAMES_MERGED.load(Ordering::Relaxed);
    if n > 0 {
        s.push_str(&format!(
            "[highvar] high-variable names merged (disjoint ranges): {n}\n"
        ));
    }
    let i = INCOMING_MARKED.load(Ordering::Relaxed);
    if i > 0 {
        s.push_str(&format!(
            "[highvar] incoming ABI entry versions marked: {i}\n"
        ));
    }
    let u = UNDEFINED_LEFT.load(Ordering::Relaxed);
    if u > 0 {
        s.push_str(&format!(
            "[highvar] read-but-never-defined versions NOT provably incoming: {u}\n"
        ));
    }
    let e = EXCUSED_ON_ENTRY.load(Ordering::Relaxed);
    let a = UNDEFINED_UNATTRIBUTED.load(Ordering::Relaxed);
    if e > 0 || a > 0 {
        s.push_str(&format!(
            "[highvar]   (excused on entry: {e}; no local-table entry: {a})\n"
        ));
    }
    // The breakdown is what makes the residue ACTIONABLE: the origin names the
    // register or slot the lost definition belonged to, and the families have
    // different causes and different fixes. Without it "41,471" is a number
    // nobody can act on.
    if u > 0 {
        if let Some(m) = UNDEFINED_ORIGINS.get() {
            if let Ok(g) = m.lock() {
                let mut top: Vec<(&String, &u64)> = g.iter().collect();
                // Most frequent first; ties by origin so the line is stable.
                top.sort_by(|x, y| y.1.cmp(x.1).then_with(|| x.0.cmp(y.0)));
                let line: Vec<String> = top
                    .iter()
                    .take(12)
                    .map(|(o, c)| format!("{o}={c}"))
                    .collect();
                s.push_str(&format!(
                    "[highvar]   undefined by origin: {}\n",
                    line.join(", ")
                ));
                if top.len() > 12 {
                    s.push_str(&format!(
                        "[highvar]   ({} origin families in total)\n",
                        top.len()
                    ));
                }
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// The value graph
// ---------------------------------------------------------------------------

/// Def/use/alias census of one method's listing, built once per round.
///
/// Every map is keyed by SSA VERSION id (the `u32` inside `Operand::Local`) and
/// every vector of instruction indices is ascending, so anything derived from
/// this graph is deterministic without sorting afterwards.
#[derive(Default)]
struct Graph {
    /// version → defining instruction indices (ascending).
    defs: HashMap<u32, Vec<usize>>,
    /// version → its single definition, when it has EXACTLY one.
    def_index: HashMap<u32, usize>,
    /// version → use instruction indices (ascending).
    uses: HashMap<u32, Vec<usize>>,
    /// version → the address of the cell it takes, when it appears inside an
    /// `AddressOf`. An address-taken cell must never be substituted: two
    /// versions can hold the same VALUE while their addresses differ.
    addr_taken: HashSet<u32>,
    /// `Move(Local dest, Local src)` sites as (instruction index, dest, src),
    /// ascending by index.
    copies: Vec<(usize, u32, u32)>,
    /// `Phi(Local dest, args…)` sites as (instruction index, dest, args),
    /// ascending by index. `None` marks a non-local argument.
    phis: Vec<(usize, u32, Vec<Option<u32>>)>,
    /// instruction index → CFG block index, or `usize::MAX` when unplaced.
    block_of: Vec<usize>,
    /// Versions defined by a CALL (`Op::Call`). Two call results are two
    /// DISTINCT source values even when one register slot is reused for both;
    /// merging their names made `Vector3` results render as the same variable
    /// (`vector3.x - vector3.x`).
    call_dests: HashSet<u32>,
}

impl Graph {
    fn build(insns: &[Insn], cfg: &Cfg) -> Graph {
        let mut g = Graph {
            block_of: vec![usize::MAX; insns.len()],
            ..Default::default()
        };
        for (bi, b) in cfg.blocks.iter().enumerate() {
            for i in b.start..b.end.min(insns.len()) {
                g.block_of[i] = bi;
            }
        }
        for (i, insn) in insns.iter().enumerate() {
            if matches!(insn.op, Op::Nop) {
                continue;
            }
            let dpos = Insn::dest_position(&insn.op);
            if let Some(pos) = dpos {
                if let Some(Operand::Local(d)) = insn.operands.get(pos) {
                    let n = {
                        let list = g.defs.entry(*d).or_default();
                        list.push(i);
                        list.len()
                    };
                    if n == 1 {
                        g.def_index.insert(*d, i);
                    } else {
                        // More than one definition: no longer "the" definition.
                        g.def_index.remove(d);
                    }
                    if matches!(insn.op, Op::Call) {
                        g.call_dests.insert(*d);
                    }
                }
            }
            for (k, op) in insn.operands.iter().enumerate() {
                if dpos == Some(k) && matches!(op, Operand::Local(_)) {
                    continue; // the destination is a def, not a use
                }
                collect_local_uses(op, &mut |v| {
                    g.uses.entry(v).or_default().push(i);
                });
                // Address-taken cells: refused by every substitution rule.
                collect_address_taken(op, &mut g.addr_taken);
            }
            match &insn.op {
                Op::Move => {
                    if let (Some(Operand::Local(d)), Some(Operand::Local(s))) =
                        (insn.operands.first(), insn.operands.get(1))
                    {
                        g.copies.push((i, *d, *s));
                    }
                }
                Op::Phi => {
                    if let Some(Operand::Local(d)) = insn.operands.first() {
                        let args = insn.operands.iter().skip(1).map(|o| o.as_local()).collect();
                        g.phis.push((i, *d, args));
                    }
                }
                _ => {}
            }
        }
        g
    }

    /// Live span of a version: (first mention, last mention) over defs and
    /// uses. `None` when the version appears nowhere.
    fn span(&self, v: u32) -> Option<(usize, usize)> {
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        let mut any = false;
        if let Some(ds) = self.defs.get(&v) {
            for &i in ds {
                lo = lo.min(i);
                hi = hi.max(i);
                any = true;
            }
        }
        if let Some(us) = self.uses.get(&v) {
            for &i in us {
                lo = lo.min(i);
                hi = hi.max(i);
                any = true;
            }
        }
        any.then_some((lo, hi))
    }
}

/// Walk every local a (possibly nested) operand reads.
fn collect_local_uses<F: FnMut(u32)>(op: &Operand, f: &mut F) {
    match op {
        Operand::Local(v) => f(*v),
        Operand::Memory { base, index, .. } => {
            collect_local_uses(base, &mut *f);
            if let Some(i) = index {
                collect_local_uses(i, &mut *f);
            }
        }
        Operand::InstanceField { base, .. } => collect_local_uses(base, &mut *f),
        Operand::AddressOf(inner) | Operand::ArrayLength(inner) => {
            collect_local_uses(inner, &mut *f)
        }
        Operand::ArrayAccess { array, index } => {
            collect_local_uses(array, &mut *f);
            collect_local_uses(index, &mut *f);
        }
        _ => {}
    }
}

/// Record every local whose ADDRESS is taken anywhere inside `op`.
fn collect_address_taken(op: &Operand, out: &mut HashSet<u32>) {
    match op {
        Operand::AddressOf(inner) => collect_local_uses(inner, &mut |v| {
            out.insert(v);
        }),
        Operand::Memory { base, index, .. } => {
            collect_address_taken(base, out);
            if let Some(i) = index {
                collect_address_taken(i, out);
            }
        }
        Operand::InstanceField { base, .. } => collect_address_taken(base, out),
        Operand::ArrayAccess { array, index } => {
            collect_address_taken(array, out);
            collect_address_taken(index, out);
        }
        Operand::ArrayLength(inner) => collect_address_taken(inner, out),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Aliases
// ---------------------------------------------------------------------------

/// `alias[dead] = replacement`. Built only by the rules, only in the direction
/// the rule PROVED, and never a cycle (see [`would_cycle`]).
#[derive(Default)]
struct Aliases {
    map: HashMap<u32, u32>,
}

impl Aliases {
    /// Follow a version to the version whose value it provably equals.
    ///
    /// Returns `v` itself when nothing is known about it, and `None` when the
    /// chain is too deep (a cycle, which the builders refuse — this is the belt
    /// to that suspender so a bug cannot hang a dump).
    fn resolve(&self, v: u32) -> Option<u32> {
        let mut cur = v;
        for _ in 0..MAX_ALIAS_DEPTH {
            match self.map.get(&cur) {
                Some(&next) if next != cur => cur = next,
                _ => return Some(cur),
            }
        }
        None
    }

    /// Is `v` already resolved to something else?
    fn is_aliased(&self, v: u32) -> bool {
        self.map.get(&v).is_some_and(|&t| t != v)
    }

    /// Would adding `dead → target` close a cycle? Walks the target's chain: if
    /// it reaches `dead`, the two are mutually recursive and neither may be
    /// eliminated (which is correct — a value that only ever comes from the
    /// other one is not defined by either).
    fn would_cycle(&self, dead: u32, target: u32) -> bool {
        let mut cur = target;
        for _ in 0..MAX_ALIAS_DEPTH {
            if cur == dead {
                return true;
            }
            match self.map.get(&cur) {
                Some(&next) if next != cur => cur = next,
                _ => return false,
            }
        }
        // Unresolvable chain: refuse rather than risk a cycle.
        true
    }

    /// Prove `dead ≡ target` and record it. Refuses a cycle, a self-alias, and
    /// an already-aliased source. Returns whether the alias was recorded.
    fn prove(&mut self, dead: u32, target: u32) -> bool {
        if dead == target || self.is_aliased(dead) || self.would_cycle(dead, target) {
            return false;
        }
        self.map.insert(dead, target);
        true
    }
}

// ---------------------------------------------------------------------------
// Guards shared by the rules
// ---------------------------------------------------------------------------

/// Versions the pipeline treats as EXISTING without a definition in this body:
/// `this`, parameters, the indirect-result buffer, values the ABI proves are
/// incoming, and the entry value of `sp`/`fp`/`rv`/the absolute-address
/// pseudo-register/a stack cell.
///
/// Mirrors `locals::rooted_versions` + `verify.rs`'s excused set, because a
/// substitution whose source is one of these is ALWAYS legal — the value is
/// live on entry to the method and therefore dominates every use by
/// construction.
fn seeded_versions(table: &LocalTable) -> HashSet<u32> {
    let mut seeded: HashSet<u32> = HashSet::new();
    let mut entry: HashMap<String, u32> = HashMap::new();
    for (v, idx) in &table.index_of {
        let Some(info) = table.infos.get(*idx) else {
            continue;
        };
        if info.is_this || info.is_param.is_some() || info.is_return_buffer || info.is_incoming_arg
        {
            seeded.insert(*v);
        }
        // SSA seeds one ENTRY version per slot, allocating versions in
        // increasing order, so the LOWEST id seen per origin is that entry
        // version (this is the same convention `locals::create` and
        // `rooted_versions` use).
        entry
            .entry(info.slot_origin.clone())
            .and_modify(|e| {
                if *v < *e {
                    *e = *v;
                }
            })
            .or_insert(*v);
    }
    for (origin, v) in entry {
        if origin == "sp"
            || origin == "fp"
            || origin == "rv"
            || origin == "abs"
            || origin == "rip"
            || origin.starts_with("stack_")
        {
            seeded.insert(v);
        }
    }
    seeded
}

/// The ENTRY version of every slot origin (see [`seeded_versions`]).
fn entry_versions(table: &LocalTable) -> HashMap<u32, String> {
    let mut entry: HashMap<String, u32> = HashMap::new();
    for (v, idx) in &table.index_of {
        let Some(info) = table.infos.get(*idx) else {
            continue;
        };
        entry
            .entry(info.slot_origin.clone())
            .and_modify(|e| {
                if *v < *e {
                    *e = *v;
                }
            })
            .or_insert(*v);
    }
    entry.into_iter().map(|(o, v)| (v, o)).collect()
}

/// Is this register slot one the ABI hands a value IN, so that a read with no
/// definition in the body is an incoming value rather than a lost one?
///
/// Deliberately the CONSERVATIVE list: the integer argument registers, `this`'s
/// slot, the callee-saved integer registers (the caller's value, spilled by the
/// prologue), the FP argument and callee-saved register views of both naming
/// schemes, and the pseudo-slots. Caller-saved temporaries (`x9..x18`) are
/// EXCLUDED: nothing is passed in them, so a read of one the body never wrote is
/// a genuine lost definition and must keep being reported.
fn is_incoming_register(origin: &str) -> bool {
    if origin.starts_with("stack_") {
        return true;
    }
    // `abs`/`rip` are NOT incoming: their entry version is the base of an
    // absolute address, which `locals::absolute_static_addr` already handles as
    // a value of its own, and marking them incoming would root a slot the
    // address logic expects to be definition-less.
    match origin {
        "sp" | "fp" | "rv" | "this" | "x8" => return true,
        _ => {}
    }
    // Integer: x0..x8 (arguments + sret pointer) and x19..x28 (callee-saved).
    // The `w`-view of the same registers is the same storage.
    let int = origin
        .strip_prefix('x')
        .or_else(|| origin.strip_prefix('w'))
        .and_then(|d| d.parse::<u32>().ok());
    if let Some(n) = int {
        return n <= 8 || (19..=28).contains(&n);
    }
    // Floating point, in every spelling the lifters use: arm64 names the views
    // `sN`/`dN`/`qN` and the 32-bit view `vsN`/`vdN`/`vqN`; x86-64 normalizes
    // XMM to `vfN`. v0..v7 are arguments, v8..v15 callee-saved, so the whole
    // 0..15 range is a register the ABI supplies a value in.
    for pre in ["vf", "vq", "vd", "q", "d", "s"] {
        if let Some(rest) = origin.strip_prefix(pre) {
            if let Ok(n) = rest.parse::<u32>() {
                return n <= 15;
            }
        }
    }
    // AArch64's 32-bit SIMD names are flattened lane identities:
    // `vs{4*reg + lane}`. v0-v7 are incoming argument registers; AAPCS64
    // preserves only the low 64 bits (lanes 0 and 1) of v8-v15. The flattened
    // names can therefore exceed 15 (`v9.s[1]` is `vs37`) while still being a
    // valid incoming value. Higher lanes of v8-v15 remain caller-saved.
    if let Some(n) = origin
        .strip_prefix("vs")
        .and_then(|d| d.parse::<u32>().ok())
    {
        let reg = n / 4;
        let lane = n % 4;
        return reg < 8 || ((8..16).contains(&reg) && lane < 2);
    }
    // ARM32 names its general-purpose registers `g0..g15` (`lift_arm32`), and
    // x86-64 uses the SAME spelling for the ordinals it does not rename
    // (`lift_x86::gpr_name_for_ordinal`). This function sees a NAME and not an
    // architecture, so only the origins that are ABI-supplied under BOTH
    // interpretations are accepted:
    //   * `g0..g5`, `g8`, `g9` cannot appear on x86-64 at all (those ordinals are
    //     renamed to `rv`/`x0..x5`/`sp`/`fp`), and on ARM32 they are r0..r5/r8/r9
    //     — arguments and callee-saved registers the caller supplies;
    //   * `g12..g15` are r12..r15 on x86-64 (callee-saved) and ip/`sp`/lr/pc on
    //     ARM32 — every one a value that exists on entry.
    // `g6`, `g7`, `g10` and `g11` are DELIBERATELY excluded: on x86-64 those are
    // rsi/rdi (scratch under MSVC, which passes in rcx/rdx/r8/r9) and r10/r11
    // (caller-saved), where a read with no definition really IS a lost one.
    // Excusing them would hide x86 losses to make an ARM32 number look better,
    // which is the wrong trade.
    if let Some(n) = origin.strip_prefix('g').and_then(|d| d.parse::<u32>().ok()) {
        return n <= 5 || n == 8 || n == 9 || (12..=15).contains(&n);
    }
    false
}

/// Does the single definition of `def_v` dominate the use at
/// (`use_block`, `use_index`)?
///
/// A seeded version dominates everything. Otherwise the definition's block must
/// dominate the use's block, or be the same block with an earlier index. An
/// unplaced instruction (no block) refuses: without CFG evidence there is
/// nothing to prove.
fn def_dominates_use(
    g: &Graph,
    dom: &DominatorInfo,
    seeded: &HashSet<u32>,
    def_v: u32,
    use_block: usize,
    use_index: usize,
) -> bool {
    if seeded.contains(&def_v) {
        return true;
    }
    let Some(&di) = g.def_index.get(&def_v) else {
        return false;
    };
    let Some(&db) = g.block_of.get(di) else {
        return false;
    };
    if db == usize::MAX || use_block == usize::MAX {
        return false;
    }
    if db == use_block {
        return di < use_index;
    }
    dom.dominates(db, use_block)
}

/// Is `v` a value the listing actually HAS: defined here, or supplied by the
/// ABI on entry?
///
/// Every substitution rule requires this of the value it substitutes. A version
/// that is neither defined nor seeded carries a value the pipeline lost, and
/// spreading that name into more use sites does not recover it — it only makes
/// the loss wider (measured: `/scripts/audit_dump.py` counts every read of a
/// never-defined name, so replacing `v42 = v10` by `v10` at three use sites
/// turns ONE reported site into three). Refusing keeps the binding, which is the
/// smaller, more readable defect.
fn is_rooted(g: &Graph, seeded: &HashSet<u32>, v: u32) -> bool {
    seeded.contains(&v) || g.def_index.contains_key(&v)
}

/// Does `def_v` dominate EVERY use of `v`?
fn def_dominates_all_uses(
    g: &Graph,
    dom: &DominatorInfo,
    seeded: &HashSet<u32>,
    def_v: u32,
    v: u32,
) -> bool {
    let Some(uses) = g.uses.get(&v) else {
        // No uses: nothing can be wrong, and eliminating the definition is the
        // job of dead-code elimination, not of this rule. Refuse so the pass
        // never reports a fold it did not actually observe.
        return false;
    };
    uses.iter().all(|&u| {
        g.block_of
            .get(u)
            .copied()
            .is_some_and(|b| def_dominates_use(g, dom, seeded, def_v, b, u))
    })
}

/// Is `v` a version this pass may REMOVE the definition of?
///
/// Never `this` (its identity is the receiver), never a parameter or one of its
/// float components (the renderer prints those by name and the ABI trims
/// arguments from them), never the indirect-result buffer, and never an
/// address-taken cell.
fn may_eliminate(g: &Graph, table: &LocalTable, v: u32) -> bool {
    if g.addr_taken.contains(&v) {
        return false;
    }
    match table.index_of.get(&v).and_then(|&i| table.infos.get(i)) {
        Some(info) => !info.is_this && info.is_param.is_none() && !info.is_return_buffer,
        // No table entry: an internal scratch slot. Nothing renders by name, so
        // there is nobody to break — but also no reason to touch it.
        None => false,
    }
}

/// Can two locals share one printed name without misleading the reader? Only
/// when one type is the other or strictly weaker (`Unknown`/`IntPtr`).
fn types_compatible(a: &ManagedType, b: &ManagedType) -> bool {
    use ManagedType::*;
    if a == b {
        return true;
    }
    matches!(
        (a, b),
        (Unknown, _) | (_, Unknown) | (IntPtr, _) | (_, IntPtr)
    )
}

// ---------------------------------------------------------------------------
// Canonical form (the structural comparison rule C is proved from)
// ---------------------------------------------------------------------------

/// Write a canonical, alias-resolved rendering of `op` into `out`.
///
/// Returns `false` for every operand whose value is NOT a function of the
/// values the key already mentions — i.e. anything that reads MEMORY (a load's
/// value can change between two identical loads), anything whose value is not
/// recovered (`Unknown`, `MissingAbiArg`), and an address-taken cell. A `false`
/// makes the whole computation unkeyable, which is exactly right: unification is
/// only for computations over values this listing already proves equal.
fn canon_operand(op: &Operand, alias: &Aliases, out: &mut String) -> bool {
    match op {
        Operand::Local(v) => {
            let r = alias.resolve(*v).unwrap_or(*v);
            out.push_str("L");
            out.push_str(&r.to_string());
            true
        }
        Operand::Register(r) => {
            out.push_str("R:");
            out.push_str(r);
            true
        }
        Operand::Immediate(i) => {
            out.push_str("I:");
            out.push_str(&i.to_string());
            true
        }
        Operand::FloatLiteral(f) => {
            out.push_str("F:");
            out.push_str(&f.to_bits().to_string());
            true
        }
        Operand::DoubleLiteral(d) => {
            out.push_str("D:");
            out.push_str(&d.to_bits().to_string());
            true
        }
        Operand::StringLiteral { index, .. } => {
            out.push_str("S:");
            out.push_str(&index.to_string());
            true
        }
        Operand::Method(m) => {
            out.push_str("M:");
            out.push_str(&m.full_name());
            true
        }
        Operand::Type(t) => {
            out.push_str("T:");
            out.push_str(&t.ty.to_string());
            true
        }
        Operand::Field(_) | Operand::InstanceField { .. } => false,
        Operand::Memory { .. } | Operand::ArrayAccess { .. } | Operand::ArrayLength(_) => false,
        Operand::AddressOf(_) => false,
        Operand::MissingAbiArg | Operand::Unknown => false,
    }
}

/// The canonical key of a computation, or `None` when it may not be unified.
///
/// The destination is excluded (it is what differs) and `dest_type` is
/// included, so two identical operations that produce different types are NOT
/// treated as one value.
fn value_key(insn: &Insn, alias: &Aliases) -> Option<String> {
    let pure = matches!(
        insn.op,
        Op::Arith(_)
            | Op::FloatArith(_)
            | Op::Compare(_)
            | Op::Not
            | Op::Negate
            | Op::Convert(_)
            | Op::MulAcc { .. }
            | Op::Clz
            | Op::Select(_)
            | Op::LoadLength
            | Op::Move
    );
    if !pure {
        return None;
    }
    let dpos = Insn::dest_position(&insn.op)?;
    let mut key = String::new();
    key.push_str(&format!("{:?}|", insn.op));
    for (i, op) in insn.operands.iter().enumerate() {
        if i == dpos {
            continue;
        }
        if !canon_operand(op, alias, &mut key) {
            return None;
        }
        key.push('|');
    }
    key.push_str(&format!("{:?}", insn.dest_type));
    Some(key)
}

// ---------------------------------------------------------------------------
// Rule A — trivial copies
// ---------------------------------------------------------------------------

/// `d = s` where `d` has exactly one definition in the listing and `s`
/// dominates every use of `d`: every use of `d` is `s`.
///
/// The single-definition requirement is what makes this legal without a
/// reaching-definition analysis: if the move is the only thing that ever wrote
/// `d`, then the value at every use of `d` is exactly the value `s` held at the
/// move — provided `s` itself did not change, which SSA guarantees for a
/// VERSION (`Local(v)` is immutable by construction, and where our SSA is not
/// perfect the dominance proof below catches it).
fn collect_trivial_copies(
    g: &Graph,
    table: &LocalTable,
    seeded: &HashSet<u32>,
    dom: &DominatorInfo,
    alias: &mut Aliases,
    stats: &mut HighVarStats,
) {
    for &(idx, d, s) in &g.copies {
        if g.def_index.get(&d) != Some(&idx) {
            continue; // not the only definition of `d`
        }
        if !may_eliminate(g, table, d) {
            continue;
        }
        let Some(target) = alias.resolve(s) else {
            continue; // alias chain too deep — refuse rather than risk a cycle
        };
        if target == d {
            continue;
        }
        if !is_rooted(g, seeded, target) {
            continue; // the source value was never recovered — do not spread it
        }
        if !def_dominates_all_uses(g, dom, seeded, target, d) {
            continue; // `s` does not provably hold this value at every use
        }
        if alias.prove(d, target) {
            stats.copies_folded += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Rule B — phi agreement
// ---------------------------------------------------------------------------

/// A phi whose every argument resolves to one version, with no `Unknown`
/// argument, is that version.
///
/// One `Unknown` argument disqualifies the whole phi: the value differs on that
/// path, so claiming the surviving version would be an invention. An argument
/// list of length < 2 is a malformed phi and is refused for the same reason.
fn collect_phi_agreement(
    g: &Graph,
    table: &LocalTable,
    seeded: &HashSet<u32>,
    dom: &DominatorInfo,
    alias: &mut Aliases,
    stats: &mut HighVarStats,
) {
    for &(idx, d, ref args) in &g.phis {
        if args.len() < 2 {
            continue;
        }
        if g.def_index.get(&d) != Some(&idx) {
            continue; // the phi is not the only definition of `d`
        }
        if !may_eliminate(g, table, d) {
            continue;
        }
        let mut agreed: Option<u32> = None;
        let mut ok = true;
        for a in args.iter() {
            let Some(a) = *a else {
                ok = false;
                break;
            };
            let Some(r) = alias.resolve(a) else {
                ok = false;
                break;
            };
            match agreed {
                None => agreed = Some(r),
                Some(prev) if prev == r => {}
                Some(_) => {
                    ok = false;
                    break;
                }
            }
        }
        let Some(target) = agreed else { continue };
        if !ok || target == d {
            continue;
        }
        if !is_rooted(g, seeded, target) {
            continue; // every path agrees on a value nobody recovered — refuse
        }
        // The agreeing value must reach the phi: it is either an entry value
        // (seeded, live into the method) or defined before the phi on every
        // path into the phi's block.
        let phi_block = g.block_of.get(idx).copied().unwrap_or(usize::MAX);
        if phi_block == usize::MAX || !def_dominates_use(g, dom, seeded, target, phi_block, idx) {
            continue;
        }
        if !def_dominates_all_uses(g, dom, seeded, target, d) {
            continue;
        }
        if alias.prove(d, target) {
            stats.phis_coalesced += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Rule C — value unification
// ---------------------------------------------------------------------------

/// Two versions with one pure definition each that are the SAME computation,
/// where the earlier definition dominates every use of the later one.
///
/// This is the rule that removes the `v14 = phi(v11, v11)` chains a merge of
/// two paths that computed the same thing leaves behind: the two arms are
/// different versions with identical definitions, which no amount of copy
/// propagation can equate because neither is a copy of the other.
fn collect_value_unification(
    insns: &[Insn],
    g: &Graph,
    table: &LocalTable,
    seeded: &HashSet<u32>,
    dom: &DominatorInfo,
    alias: &mut Aliases,
    stats: &mut HighVarStats,
) {
    // key → versions with exactly one pure definition having that key.
    let mut groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let mut versions: Vec<u32> = g.def_index.keys().copied().collect();
    versions.sort_unstable();
    for v in versions {
        if alias.is_aliased(v) || !may_eliminate(g, table, v) {
            continue;
        }
        let Some(&idx) = g.def_index.get(&v) else {
            continue;
        };
        let Some(insn) = insns.get(idx) else {
            continue;
        };
        let Some(key) = value_key(insn, alias) else {
            continue;
        };
        groups.entry(key).or_default().push(v);
    }
    for (_key, mut group) in groups {
        if group.len() < 2 {
            continue;
        }
        // Deterministic order: by (block, instruction index), then version id.
        group.sort_by_key(|&v| {
            let i = g.def_index.get(&v).copied().unwrap_or(usize::MAX);
            let b = g.block_of.get(i).copied().unwrap_or(usize::MAX);
            (b, i, v)
        });
        let kept = group[0];
        for &cand in &group[1..] {
            if cand == kept || alias.is_aliased(cand) {
                continue;
            }
            if !def_dominates_all_uses(g, dom, seeded, kept, cand) {
                continue;
            }
            if alias.prove(cand, kept) {
                stats.values_unified += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Applying the aliases
// ---------------------------------------------------------------------------

/// Rewrite every use of an eliminated version to the version it provably
/// equals, and remove its definition.
///
/// Order matters: the definitions go FIRST, so the rewriting walk can never see
/// an instruction whose destination is a version that is about to disappear.
/// Every removed instruction was proven pure by the rule that decided it (a copy
/// whose source dominates its uses, a phi whose arguments agree, or a duplicate
/// of an earlier identical computation), so removing it cannot lose an
/// observation.
fn apply_aliases(insns: &mut [Insn], g: &Graph, alias: &Aliases, stats: &mut HighVarStats) {
    let mut dead: Vec<u32> = alias.map.keys().copied().collect();
    dead.sort_unstable();
    if dead.is_empty() {
        return;
    }
    // Count what is actually erased, so the report can distinguish a rule that
    // proved something from one that silently stopped matching.
    stats.aliases_erased += dead.len();
    for &v in &dead {
        let Some(&idx) = g.def_index.get(&v) else {
            continue;
        };
        if idx >= insns.len() || matches!(insns[idx].op, Op::Nop) {
            continue;
        }
        let addr = insns[idx].address;
        // `Op::Nop` is skipped by the emitter (`emit.rs`) and by every census,
        // so this deletes the statement while keeping branch targets and the
        // address/instruction pairing intact.
        super::verify::ledger_note(addr, super::verify::LedgerKind::Dce);
        insns[idx] = Insn::nop(addr);
    }

    let dead_set: HashSet<u32> = dead.into_iter().collect();
    for insn in insns.iter_mut() {
        if matches!(insn.op, Op::Nop) {
            continue;
        }
        let dpos = Insn::dest_position(&insn.op);
        for (k, op) in insn.operands.iter_mut().enumerate() {
            // A destination position is never rewritten: a version that still
            // has a definition must keep it, and an eliminated version's own
            // definition is already gone.
            if dpos == Some(k) && matches!(op, Operand::Local(_)) {
                continue;
            }
            rewrite_uses(op, &dead_set, alias);
        }
    }
}

/// Erase `Move(v, v)` — a pure identity, and never a statement worth printing.
///
/// These are both pre-existing (a pin copy the lifter could not canonicalize)
/// and created by this pass (a rewrite turning `Move(d, s)` into `Move(s, s)`).
/// Either way the instruction defines a version from itself, which renders as an
/// assignment that says nothing and which the oracle reads as a definition that
/// exists while the value never changes.
fn erase_self_moves(insns: &mut [Insn]) -> usize {
    let mut n = 0usize;
    for insn in insns.iter_mut() {
        if insn.op != Op::Move {
            continue;
        }
        let same = matches!(
            (insn.operands.first(), insn.operands.get(1)),
            (Some(Operand::Local(a)), Some(Operand::Local(b))) if a == b
        );
        if same {
            let addr = insn.address;
            super::verify::ledger_note(addr, super::verify::LedgerKind::Dce);
            *insn = Insn::nop(addr);
            n += 1;
        }
    }
    n
}

/// Substitute eliminated versions inside one operand, recursively.
fn rewrite_uses(op: &mut Operand, dead: &HashSet<u32>, alias: &Aliases) {
    match op {
        Operand::Local(v) if dead.contains(v) => {
            if let Some(target) = alias.resolve(*v) {
                *op = Operand::Local(target);
            }
        }
        Operand::Memory { base, index, .. } => {
            rewrite_uses(base, dead, alias);
            if let Some(i) = index {
                rewrite_uses(i, dead, alias);
            }
        }
        Operand::InstanceField { base, .. } => rewrite_uses(base, dead, alias),
        Operand::AddressOf(inner) | Operand::ArrayLength(inner) => rewrite_uses(inner, dead, alias),
        Operand::ArrayAccess { array, index } => {
            rewrite_uses(array, dead, alias);
            rewrite_uses(index, dead, alias);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Rule D — high-variable naming
// ---------------------------------------------------------------------------

/// Give versions of one register slot whose live spans do NOT overlap one name.
///
/// This is a NAMING decision only: the IR keeps every `Local(v)` exactly as it
/// was, so no value can change. What it buys is the thing `merge.cc` is for —
/// two disjoint varnodes of one storage ARE one variable, and printing them as
/// two names invents a live range the source never had. (It also means a use of
/// a version whose own definition the renderer dropped now shares the name of
/// the slot's definition, which is the honest reading of the same storage.)
fn merge_names(g: &Graph, table: &mut LocalTable, stats: &mut HighVarStats) {
    let mut by_slot: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for (&v, idx) in table.index_of.iter() {
        let Some(info) = table.infos.get(*idx) else {
            continue;
        };
        if info.is_this || info.is_param.is_some() || info.is_return_buffer || info.is_incoming_arg
        {
            continue; // those render by name from their own identity
        }
        if !g.defs.contains_key(&v) || g.addr_taken.contains(&v) {
            continue; // only DEFINED, unescaped versions qualify
        }
        by_slot.entry(info.slot_origin.clone()).or_default().push(v);
    }
    for (_slot, mut versions) in by_slot {
        if versions.len() < 2 {
            continue;
        }
        versions.sort_by_key(|&v| (g.span(v).map(|s| s.0).unwrap_or(usize::MAX), v));
        let mut i = 0usize;
        while i < versions.len() {
            let rep = versions[i];
            if g.call_dests.contains(&rep) {
                // A call result is its own source value; see `call_dests`.
                i += 1;
                continue;
            }
            let Some((_, mut end)) = g.span(rep) else {
                i += 1;
                continue;
            };
            let Some(&rep_idx) = table.index_of.get(&rep) else {
                i += 1;
                continue;
            };
            let rep_ty = table.infos[rep_idx].ty.clone();
            let mut j = i + 1;
            while j < versions.len() {
                let cand = versions[j];
                if g.call_dests.contains(&cand) {
                    // Never fold a call result into another variable's name.
                    j += 1;
                    continue;
                }
                let Some((lo, hi)) = g.span(cand) else {
                    j += 1;
                    continue;
                };
                if lo <= end {
                    break; // the ranges interfere — different variables
                }
                let Some(&cand_idx) = table.index_of.get(&cand) else {
                    j += 1;
                    continue;
                };
                if cand_idx == rep_idx {
                    j += 1;
                    continue;
                }
                if !types_compatible(&rep_ty, &table.infos[cand_idx].ty) {
                    // One name for two different types would mislead the reader
                    // more than a second name does.
                    break;
                }
                table.index_of.insert(cand, rep_idx);
                stats.names_merged += 1;
                end = hi;
                j += 1;
            }
            i = j;
        }
    }
}

fn merge_phi_names(g: &Graph, table: &mut LocalTable, stats: &mut HighVarStats) {
    // Only x0 (the ARM64 result/argument-0 register) is merged. Its
    // phi-connected versions are call results and scratch values, where one
    // high variable is the honest reading and the naming win is large
    // (0x151F34C: 341→171 vN). Stack spills and SIMD versions are long-lived
    // struct/FP state: merging their names changed source names and typed
    // locals (0x17F2380 +20 vN, 0x1530F2C +1 nint), so they are left alone.
    fn mergeable_slot(slot: &str) -> bool {
        slot == "x0"
    }
    // Unlike rule D, phi-connected versions of one slot usually have LINEARLY
    // OVERLAPPING spans (a loop header's phi destination is live across the
    // body while the back-edge argument is defined at its end). They are still
    // one source variable: every edge hands that variable a value, and SSA
    // splits it into versions only to keep definitions unique. Naming them
    // together turns the phi edge copies into `x = x`, which the cleanup pass
    // drops, instead of printing a rotation of vN names.
    for _ in 0..3 {
        let mut changed = false;
        let mappings: Vec<(u32, usize, String)> = {
            let mut out = Vec::new();
            for &(_, d, ref args) in &g.phis {
                let Some(&di) = table.index_of.get(&d) else {
                    continue;
                };
                if !g.defs.contains_key(&d) || g.addr_taken.contains(&d) {
                    continue;
                }
                if g.call_dests.contains(&d) {
                    continue;
                }
                let d_info = &table.infos[di];
                if d_info.is_this
                    || d_info.is_param.is_some()
                    || d_info.is_return_buffer
                    || d_info.is_incoming_arg
                {
                    continue;
                }
                for a in args.iter().flatten() {
                    let Some(&ai) = table.index_of.get(a) else {
                        continue;
                    };
                    if ai == di {
                        continue;
                    }
                    if g.call_dests.contains(a) {
                        continue;
                    }
                    let a_info = &table.infos[ai];
                    if a_info.is_this
                        || a_info.is_param.is_some()
                        || a_info.is_return_buffer
                        || a_info.is_incoming_arg
                    {
                        continue;
                    }
                    if !g.defs.contains_key(a) || g.addr_taken.contains(a) {
                        continue;
                    }
                    if a_info.slot_origin != d_info.slot_origin {
                        continue;
                    }
                    if !mergeable_slot(&d_info.slot_origin) {
                        continue;
                    }
                    // Merge only UNTYPED versions. A merge unifies the table
                    // entry (hence the display name AND the type), so touching
                    // a known field/array type loses the field-derived name
                    // and regressed nint in 0x17FF3B0. Untyped phi groups are
                    // where the huge naming win is (0x151F34C: 341→171 vN);
                    // typed versions keep their own entries.
                    if !matches!(
                        (&d_info.ty, &a_info.ty),
                        (ManagedType::Unknown, ManagedType::Unknown)
                    ) {
                        continue;
                    }
                    out.push((*a, di, d_info.slot_origin.clone()));
                }
            }
            out
        };
        // Large phi groups are loop/merge state worth one high variable; a
        // handful of scratch x0 reuses in one method is incidental and merging
        // them changes local_naming's evidence (0x146F874 +9 vN, 0x1530F2C +1
        // nint). Only slots whose group is big enough are unified.
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (_v, _idx, slot) in &mappings {
            *counts.entry(slot.clone()).or_default() += 1;
        }
        for (v, idx, slot) in mappings {
            if counts.get(&slot).copied().unwrap_or(0) < 8 {
                continue;
            }
            if table.index_of.get(&v).copied() == Some(idx) {
                continue;
            }
            if std::env::var("ILR_HVDBG").is_ok() {
                eprintln!("[hvmerge] slot={slot} {v}->{idx}");
            }
            table.index_of.insert(v, idx);
            stats.names_merged += 1;
            changed = true;
        }
        if !changed {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Classification and reporting of what is left
// ---------------------------------------------------------------------------

/// Mark the read-but-never-defined versions the ABI proves are INCOMING, and
/// account for the rest in three buckets.
///
/// The distinction is the whole point. `LocalInfo::is_incoming_arg` documents
/// exactly this family — "a register the ABI hands a value in but the declared
/// signature does not name" — and the verifier's oracle already excuses it.
///
/// Everything else lands in one of three buckets, and conflating them is what
/// made the residue unnavigable:
///
///   * **excused on entry** — `this`, a parameter, the indirect-result buffer, or
///     a version `seeded_versions` roots (`sp`/`fp`/`rv`/`abs`/`rip`/a stack
///     cell). These are values the pipeline HAS; counting them as losses was a
///     bug in this function, not a defect in the binary.
///   * **unattributed** — no `LocalTable` entry at all, i.e. an SSA scratch slot
///     that nothing renders by name. Not a value anyone can read as lost.
///   * **undefined_remaining** — a version the table names and nothing defines,
///     on a slot the ABI does not supply. This is the real residue, and it is
///     reported per ORIGIN because each family has a different cause.
///
/// Takes the graph the caller just built: nothing between the two calls edits
/// the listing (`merge_names` only touches `table.index_of`), so rebuilding the
/// CFG here would be the same graph at the same cost.
fn mark_incoming_and_tally(
    g: &Graph,
    table: &mut LocalTable,
    seeded: &HashSet<u32>,
    stats: &mut HighVarStats,
) {
    let entry = entry_versions(table);
    // A surviving phi defines its arguments (its edge copies materialize them),
    // which is the same rule `verify.rs` states, so they are not a defect.
    let mut phi_defd: HashSet<u32> = HashSet::new();
    for &(_, _, ref args) in &g.phis {
        for a in args.iter().flatten() {
            phi_defd.insert(*a);
        }
    }
    let mut versions: Vec<u32> = g.uses.keys().copied().collect();
    versions.sort_unstable();
    for v in versions {
        if g.def_index.contains_key(&v) || phi_defd.contains(&v) {
            continue;
        }
        let Some(&idx) = table.index_of.get(&v) else {
            // No table entry at all: an internal scratch slot, which nothing
            // renders by name. Counted apart so it can never be read as a lost
            // VALUE.
            stats.undefined_unattributed += 1;
            continue;
        };
        let (origin, excused) = {
            let info = &table.infos[idx];
            (
                info.slot_origin.clone(),
                info.is_this
                    || info.is_param.is_some()
                    || info.is_return_buffer
                    || info.is_incoming_arg
                    || info.is_call_clobber,
            )
        };
        if excused {
            stats.excused_on_entry += 1;
            continue;
        }
        // A version the pipeline SEEDS on entry exists by construction — the
        // entry value of `sp`/`fp`/`rv`/`abs`/`rip`/a stack cell. `abs` and `rip`
        // are why this check is here: `seeded_versions` roots them, the origin
        // test below used not to list them, and so every absolute-address base
        // read was reported as a lost definition by the pass that seeds it.
        if seeded.contains(&v) {
            stats.excused_on_entry += 1;
            continue;
        }
        // Only the ENTRY version of a slot can be an ABI-provided value: a
        // later version of an argument register holds something the body
        // computed, so a read of one is a genuinely lost definition.
        let entry_slot_incoming = entry
            .get(&v)
            .map(|origin| is_incoming_register(origin))
            .unwrap_or(false);
        if entry_slot_incoming {
            table.infos[idx].is_incoming_arg = true;
            stats.incoming_marked += 1;
            continue;
        }
        if origin == "sp"
            || origin == "fp"
            || origin == "rv"
            || origin == "abs"
            || origin == "rip"
            || origin.starts_with("stack_")
        {
            stats.excused_on_entry += 1;
            continue;
        }
        stats.undefined_remaining += 1;
        *stats.undefined_origins.entry(origin).or_insert(0) += 1;
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Recover high variables over one method's listing, in place.
///
/// Runs after type propagation (where the CFG, the versions and the slot origins
/// are all final) and before metadata resolution and the optimization loop, so
/// every later pass sees the collapsed form: one value per high variable instead
/// of one per SSA version. A failed proof anywhere is a NO-OP for that site.
///
/// The returned [`HighVarStats`] is also accumulated into the module counters
/// ([`report`]).
///
pub fn recover(insns: &mut [Insn], table: &mut LocalTable) -> HighVarStats {
    let mut stats = HighVarStats::default();
    if insns.is_empty() {
        return stats;
    }
    let seeded = seeded_versions(table);
    let mut alias = Aliases::default();

    // Each round decides against the listing as the previous round left it. The
    // round that finds nothing new is the fixpoint, and the round that DOES find
    // something has already applied it, so no decision can be left pending.
    for round in 0..MAX_ROUNDS {
        stats.rounds = round + 1;
        let cfg = super::ssa::rebuild_cfg(insns);
        let dom = DominatorInfo::compute(&cfg);
        let g = Graph::build(insns, &cfg);
        let before = alias.map.len();
        // Rule order is deliberate: copies first (they collapse the chains that
        // hide agreement), then phi agreement (which the collapsed chains can
        // enable), then value unification (which equates the two arms a phi's
        // arguments come from), and each round sees the others' results.
        collect_trivial_copies(&g, table, &seeded, &dom, &mut alias, &mut stats);
        collect_phi_agreement(&g, table, &seeded, &dom, &mut alias, &mut stats);
        collect_value_unification(insns, &g, table, &seeded, &dom, &mut alias, &mut stats);
        if alias.map.len() == before {
            break; // fixpoint: nothing new can be proven
        }
        apply_aliases(insns, &g, &alias, &mut stats);
    }

    // Self-moves are pure identities whether this pass created them or the
    // lifted listing already contained one.
    stats.copies_folded += erase_self_moves(insns);

    // The naming rule and the census read the listing as it now stands.
    let cfg = super::ssa::rebuild_cfg(insns);
    let g = Graph::build(insns, &cfg);
    merge_names(&g, table, &mut stats);
    if std::env::var("ILR_NOPHIMERGE").is_err() {
        merge_phi_names(&g, table, &mut stats);
    }
    mark_incoming_and_tally(&g, table, &seeded, &mut stats);
    record(&stats);
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::locals::LocalInfo;

    const ADD: Op = Op::Arith(crate::recovery::ir::ArithKind::Add);

    fn info(origin: &str) -> LocalInfo {
        LocalInfo {
            slot_origin: origin.to_string(),
            ty: ManagedType::Unknown,
            is_this: false,
            is_param: None,
            param_comp: None,
            is_return_buffer: false,
            is_incoming_arg: false,
            is_call_clobber: false,
        }
    }

    /// One table entry per listed (origin, version), in listing order. The map
    /// is built the way `locals::create` builds it: version → table position.
    fn table_of(entries: &[(&str, u32)]) -> LocalTable {
        let mut t = LocalTable::default();
        for (i, (origin, v)) in entries.iter().enumerate() {
            t.infos.push(info(origin));
            t.index_of.insert(*v, i);
        }
        t
    }

    /// Mark table position `idx` as parameter `p` — the seed that makes a
    /// version dominate every use by construction.
    fn as_param(t: &mut LocalTable, idx: usize, p: usize) {
        t.infos[idx].is_param = Some(p);
    }

    fn l(v: u32) -> Operand {
        Operand::Local(v)
    }

    fn ret(v: u32) -> Insn {
        Insn::new(Op::Return, vec![l(v)], 0x2000)
    }

    fn mem(base: u32, addend: i64) -> Operand {
        Operand::Memory {
            base: Box::new(Operand::Local(base)),
            index: None,
            scale: 8,
            addend,
            width: 0,
        }
    }

    fn nop_at(insns: &[Insn], i: usize) -> bool {
        insns.get(i).map(|x| &x.op) == Some(&Op::Nop)
    }

    // ---- rule A: trivial copies ------------------------------------------

    #[test]
    fn copy_chain_collapses_to_the_root_and_the_moves_disappear() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![l(2), l(1)], 0x1000),
            Insn::new(Op::Move, vec![l(3), l(2)], 0x1004),
            ret(3),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 2), ("v", 3)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.copies_folded, 2, "both links of the chain must fold");
        assert!(nop_at(&insns, 0) && nop_at(&insns, 1));
        assert_eq!(
            insns[2].operands[0],
            l(1),
            "the use reads the chain's root, not a name nothing defines"
        );
        assert_eq!(stats.undefined_remaining, 0);
    }

    /// The source's definition does not dominate the use, so the fold is a lie.
    #[test]
    fn copy_whose_source_is_defined_later_is_refused() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![l(2), l(1)], 0x1000),
            ret(2),
            Insn::new(Op::Move, vec![l(1), Operand::Immediate(7)], 0x1008),
        ];
        let mut t = table_of(&[("v", 1), ("v", 2)]);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(
            stats.copies_folded, 0,
            "the source is defined after the use"
        );
        assert!(matches!(insns[0].op, Op::Move));
    }

    /// A destination with two definitions has no single value to substitute.
    #[test]
    fn multi_definition_destination_is_refused() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![l(2), l(1)], 0x1000),
            Insn::new(Op::Move, vec![l(2), Operand::Immediate(3)], 0x1004),
            ret(2),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 2)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.copies_folded, 0);
        assert!(matches!(insns[0].op, Op::Move));
    }

    /// A parameter is immutable in C#: its copy destination is a NEW variable,
    /// and collapsing it would rename a parameter's storage.
    #[test]
    fn copy_destination_that_is_a_parameter_is_refused() {
        let mut insns = vec![Insn::new(Op::Move, vec![l(2), l(1)], 0x1000), ret(2)];
        let mut t = table_of(&[("x1", 1), ("x2", 2)]);
        as_param(&mut t, 0, 0);
        as_param(&mut t, 1, 1);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.copies_folded, 0);
        assert!(matches!(insns[0].op, Op::Move));
    }

    /// An address-taken cell must never be substituted: two versions can hold
    /// one value while their addresses differ.
    #[test]
    fn address_taken_destination_is_refused() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![l(2), l(1)], 0x1000),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Immediate(0x1000),
                    l(4),
                    Operand::AddressOf(Box::new(l(2))),
                ],
                0x1004,
            ),
            ret(2),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 2), ("rv", 4)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.copies_folded, 0, "`&v2` makes the cell addressable");
        assert!(matches!(insns[0].op, Op::Move));
    }

    #[test]
    fn self_move_is_erased() {
        let mut insns = vec![Insn::new(Op::Move, vec![l(2), l(2)], 0x1000), ret(2)];
        let mut t = table_of(&[("v", 2)]);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.copies_folded, 1, "`v2 = v2` says nothing");
        assert!(nop_at(&insns, 0));
    }

    // ---- rule B: phi agreement -------------------------------------------

    #[test]
    fn phi_versions_of_one_slot_share_the_name() {
        let mut insns = Vec::new();
        for v in 2..=9 {
            insns.push(Insn::new(Op::Move, vec![l(v), l(1)], 0x1000 + v as u64 * 4));
        }
        let mut phi_ops = vec![l(10)];
        for v in 2..=9 {
            phi_ops.push(l(v));
        }
        insns.push(Insn::new(Op::Phi, phi_ops, 0x1010));
        insns.push(ret(10));
        let mut t = table_of(&[
            ("x0", 1),
            ("x0", 2),
            ("x0", 3),
            ("x0", 4),
            ("x0", 5),
            ("x0", 6),
            ("x0", 7),
            ("x0", 8),
            ("x0", 9),
            ("x0", 10),
        ]);
        // Unknown is the intended merge class; typed versions keep their own
        // table entries to preserve field-derived names.
        let cfg = crate::recovery::ssa::rebuild_cfg(&insns);
        let g = Graph::build(&insns, &cfg);
        let mut stats = HighVarStats::default();
        merge_phi_names(&g, &mut t, &mut stats);
        let rep = t.index_of[&10];
        for v in 2..=9 {
            assert_eq!(t.index_of[&v], rep, "phi arg {v} shares the name");
        }
        assert_eq!(stats.names_merged, 8);
    }

    #[test]
    fn a_small_x0_phi_group_is_left_alone() {
        // Three scratch reuses of x0 are incidental; merging them changes
        // local_naming's evidence (0x146F874 +9 vN, 0x1530F2C +1 nint).
        let mut insns = Vec::new();
        for v in 2..=4 {
            insns.push(Insn::new(Op::Move, vec![l(v), l(1)], 0x1000 + v as u64 * 4));
        }
        let mut phi_ops = vec![l(5)];
        for v in 2..=4 {
            phi_ops.push(l(v));
        }
        insns.push(Insn::new(Op::Phi, phi_ops, 0x1010));
        insns.push(ret(5));
        let mut t = table_of(&[
            ("x0", 1),
            ("x0", 2),
            ("x0", 3),
            ("x0", 4),
            ("x0", 5),
        ]);
        let cfg = crate::recovery::ssa::rebuild_cfg(&insns);
        let g = Graph::build(&insns, &cfg);
        let mut stats = HighVarStats::default();
        merge_phi_names(&g, &mut t, &mut stats);
        assert_eq!(stats.names_merged, 0);
        assert_ne!(t.index_of[&2], t.index_of[&5]);
    }

    #[test]
    fn callee_saved_gpr_phi_versions_keep_separate_names() {
        let mut insns = Vec::new();
        for v in 2..=9 {
            insns.push(Insn::new(Op::Move, vec![l(v), l(1)], 0x1000 + v as u64 * 4));
        }
        let mut phi_ops = vec![l(10)];
        for v in 2..=9 {
            phi_ops.push(l(v));
        }
        insns.push(Insn::new(Op::Phi, phi_ops, 0x1010));
        insns.push(ret(10));
        let mut t = table_of(&[
            ("x22", 1),
            ("x22", 2),
            ("x22", 3),
            ("x22", 4),
            ("x22", 5),
            ("x22", 6),
            ("x22", 7),
            ("x22", 8),
            ("x22", 9),
            ("x22", 10),
        ]);
        let cfg = crate::recovery::ssa::rebuild_cfg(&insns);
        let g = Graph::build(&insns, &cfg);
        let mut stats = HighVarStats::default();
        merge_phi_names(&g, &mut t, &mut stats);
        assert_eq!(
            stats.names_merged, 0,
            "pinned callee-saved state is left alone"
        );
        assert_ne!(t.index_of[&2], t.index_of[&10]);
    }

    #[test]
    fn phi_whose_arguments_agree_collapses_to_that_argument() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![l(2), l(1)], 0x1000),
            Insn::new(Op::Phi, vec![l(3), l(2), l(2)], 0x1004),
            ret(3),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 2), ("v", 3)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.phis_coalesced, 1, "both inputs are the same value");
        assert!(nop_at(&insns, 1));
        assert_eq!(insns[2].operands[0], l(1));
        assert_eq!(stats.undefined_remaining, 0);
    }

    /// One `Unknown` argument means the value DIFFERS on that path. Claiming the
    /// agreeing value would be an invention.
    #[test]
    fn phi_with_an_unknown_argument_is_refused() {
        let mut insns = vec![
            Insn::new(Op::Phi, vec![l(3), l(1), Operand::Unknown], 0x1000),
            ret(3),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 3)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.phis_coalesced, 0);
        assert!(matches!(insns[0].op, Op::Phi));
    }

    #[test]
    fn phi_with_disagreeing_arguments_is_refused() {
        let mut insns = vec![Insn::new(Op::Phi, vec![l(3), l(1), l(2)], 0x1000), ret(3)];
        let mut t = table_of(&[("x1", 1), ("x2", 2), ("v", 3)]);
        as_param(&mut t, 0, 0);
        as_param(&mut t, 1, 1);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.phis_coalesced, 0);
        assert!(matches!(insns[0].op, Op::Phi));
    }

    // ---- rule C: value unification ---------------------------------------

    #[test]
    fn identical_computations_unify_and_the_later_definition_is_removed() {
        let mut insns = vec![
            Insn::new(ADD, vec![l(2), l(1), Operand::Immediate(4)], 0x1000),
            Insn::new(ADD, vec![l(3), l(1), Operand::Immediate(4)], 0x1004),
            ret(3),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 2), ("v", 3)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(
            stats.values_unified, 1,
            "`v1 + 4` computed twice is one value"
        );
        assert!(nop_at(&insns, 1));
        assert_eq!(insns[2].operands[0], l(2));
    }

    #[test]
    fn different_computations_do_not_unify() {
        let mut insns = vec![
            Insn::new(ADD, vec![l(2), l(1), Operand::Immediate(4)], 0x1000),
            Insn::new(ADD, vec![l(3), l(1), Operand::Immediate(8)], 0x1004),
            ret(3),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 2), ("v", 3)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.values_unified, 0);
    }

    /// Two identical loads are NOT one value: memory can change in between.
    #[test]
    fn identical_memory_reads_never_unify() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![l(2), mem(1, 16)], 0x1000),
            Insn::new(Op::Move, vec![l(3), mem(1, 16)], 0x1004),
            ret(3),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 2), ("v", 3)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(
            stats.values_unified, 0,
            "a load's value is not a pure function"
        );
    }

    /// The later definition is in a SIBLING branch: it does not dominate the
    /// use, so the substitution would read a value that may not exist.
    #[test]
    fn unification_across_sibling_branches_is_refused() {
        let mut insns = vec![
            Insn::new(Op::CondJump(3), vec![l(1)], 0x1000),
            Insn::new(ADD, vec![l(3), l(1), Operand::Immediate(4)], 0x1004),
            Insn::new(Op::Jump(5), vec![], 0x1008),
            Insn::new(ADD, vec![l(2), l(1), Operand::Immediate(4)], 0x100c),
            Insn::nop(0x1010),
            ret(2),
        ];
        let mut t = table_of(&[("x1", 1), ("v", 2), ("v", 3)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(
            stats.values_unified, 0,
            "the earlier definition is in a branch that does not dominate the use"
        );
    }

    // ---- rule D: high-variable naming ------------------------------------

    #[test]
    fn disjoint_ranges_of_one_slot_share_one_name() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![l(1), Operand::Immediate(1)], 0x1000),
            Insn::new(Op::Move, vec![l(5), Operand::Immediate(2)], 0x1004),
            ret(5),
        ];
        let mut t = table_of(&[("x1", 1), ("x1", 5)]);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(
            stats.names_merged, 1,
            "one storage, two non-overlapping ranges"
        );
        assert_eq!(
            t.index_of[&5], t.index_of[&1],
            "both versions must print the same name"
        );
    }

    #[test]
    fn overlapping_ranges_of_one_slot_keep_their_own_names() {
        let mut insns = vec![
            Insn::new(Op::Move, vec![l(1), Operand::Immediate(1)], 0x1000),
            Insn::new(Op::Move, vec![l(5), Operand::Immediate(2)], 0x1004),
            ret(5),
            ret(1),
        ];
        let mut t = table_of(&[("x1", 1), ("x1", 5)]);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.names_merged, 0, "the live ranges interfere");
    }

    // ---- classification of what is left ----------------------------------

    #[test]
    fn read_of_an_incoming_argument_register_is_marked_not_reported() {
        let mut insns = vec![ret(9)];
        let mut t = table_of(&[("x5", 9)]);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.incoming_marked, 1);
        assert_eq!(stats.undefined_remaining, 0);
        assert!(
            t.infos[0].is_incoming_arg,
            "the ABI hands a value in x5; the body not writing it is not a defect"
        );
    }

    #[test]
    fn read_of_a_caller_saved_register_is_a_lost_definition() {
        let mut insns = vec![ret(9)];
        let mut t = table_of(&[("x12", 9)]);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.incoming_marked, 0);
        assert_eq!(stats.undefined_remaining, 1);
        assert_eq!(stats.undefined_origins.get("x12"), Some(&1));
        assert!(!t.infos[0].is_incoming_arg);
    }

    #[test]
    fn read_of_a_parameter_is_excused() {
        let mut insns = vec![ret(9)];
        let mut t = table_of(&[("x12", 9)]);
        as_param(&mut t, 0, 0);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(
            stats.undefined_remaining, 0,
            "a parameter is defined on entry"
        );
    }

    #[test]
    fn an_empty_listing_is_a_no_op() {
        let mut insns: Vec<Insn> = Vec::new();
        let mut t = LocalTable::default();
        let stats = recover(&mut insns, &mut t);
        assert_eq!(
            stats.copies_folded + stats.phis_coalesced + stats.values_unified,
            0
        );
    }

    // ---- classification of what is left ----------------------------------

    /// `abs`/`rip` are the base of an absolute address and `seeded_versions`
    /// roots their entry value, so a read of one is a value the pipeline HAS.
    /// Counting it as a lost definition is the bug that inflated the residue:
    /// the tally's origin test did not list them, so it contradicted the seed
    /// set of the same pass.
    #[test]
    fn address_space_entry_versions_are_excused_not_lost() {
        for origin in ["abs", "rip"] {
            let mut insns = vec![ret(9)];
            let mut t = table_of(&[(origin, 9)]);
            let stats = recover(&mut insns, &mut t);
            assert_eq!(
                stats.undefined_remaining, 0,
                "`{origin}` is seeded: {stats:?}"
            );
            assert_eq!(stats.excused_on_entry, 1, "excused, not lost: {stats:?}");
            assert!(stats.undefined_origins.is_empty());
        }
    }

    /// A callee-saved general-purpose register read with no definition in the
    /// body is the CALLER's value — the ABI supplies it on entry. `g14` is `r14`
    /// on x86-64 and `lr` on ARM32, so it is incoming under either naming.
    #[test]
    fn callee_saved_g_register_read_is_incoming() {
        let mut insns = vec![ret(9)];
        let mut t = table_of(&[("g14", 9)]);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.incoming_marked, 1, "{stats:?}");
        assert_eq!(stats.undefined_remaining, 0);
        assert!(t.infos[0].is_incoming_arg);
    }

    /// The `gN` classifier accepts only what is ABI-supplied under BOTH lifter
    /// namings. The caller-saved ones keep being reported, because a read of a
    /// scratch register the body never wrote is a genuine loss on x86-64.
    #[test]
    fn g_register_classification_is_conservative() {
        for o in [
            "g0", "g1", "g2", "g3", "g4", "g5", "g8", "g9", "g12", "g13", "g14", "g15",
        ] {
            assert!(
                is_incoming_register(o),
                "{o} is ABI-supplied under both namings"
            );
        }
        for o in ["g6", "g7", "g10", "g11"] {
            assert!(
                !is_incoming_register(o),
                "{o} is scratch on x86-64 (rsi/rdi under MSVC, r10/r11), so a \
                 read with no definition is a real loss there"
            );
        }
    }

    /// A version the table knows nothing about is an internal scratch slot: it is
    /// counted apart, never as a lost value.
    #[test]
    fn version_without_a_table_entry_is_unattributed() {
        let mut insns = vec![ret(77)];
        let mut t = LocalTable::default();
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.undefined_remaining, 0);
        assert_eq!(stats.undefined_unattributed, 1, "{stats:?}");
    }


    /// Two CALL results in one register slot are distinct source values.
    /// Rule D used to fold them into one name as soon as their spans did not
    /// overlap, which made two `Vector3` results render as the same variable.
    #[test]
    fn call_results_of_one_slot_keep_distinct_names() {
        let mut insns = vec![
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x1000), l(2)],
                0x1000,
            ),
            Insn::new(Op::Move, vec![l(4), l(2)], 0x1004),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x1000), l(3)],
                0x1008,
            ),
            Insn::new(Op::Move, vec![l(5), l(3)], 0x100c),
            ret(4),
        ];
        let mut t = table_of(&[("x0", 2), ("x0", 3), ("x0", 4), ("x0", 5)]);
        let stats = recover(&mut insns, &mut t);
        assert_ne!(
            t.index_of[&2], t.index_of[&3],
            "call results are different values, not one reused variable: {stats:?}"
        );
    }

    /// The genuinely lost family keeps being named by origin, so the report can
    /// say WHICH mechanism produced it.
    #[test]
    fn a_lost_definition_is_attributed_to_its_origin() {
        let mut insns = vec![ret(9)];
        let mut t = table_of(&[("x12", 9)]);
        let stats = recover(&mut insns, &mut t);
        assert_eq!(stats.undefined_remaining, 1);
        assert_eq!(stats.undefined_origins.get("x12"), Some(&1));
        assert_eq!(stats.excused_on_entry, 0);
        assert_eq!(stats.undefined_unattributed, 0);
    }
}
