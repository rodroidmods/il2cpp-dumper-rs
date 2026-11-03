//! C# ↔ DISASSEMBLY concordance.
//!
//! The rest of `recovery::*` verifies the pipeline against the native bytes at
//! the IR level. This module closes the last gap: it reads the **emitted C#
//! TEXT** and asks how much of what the disassembly says is actually VISIBLE in
//! what the user is handed. A loss introduced by the printer itself — a
//! condition folded to a constant, an arm dropped while structuring, a call
//! emitted with its result discarded — is invisible to an IR-level oracle and
//! visible here.
//!
//! ## What is compared
//!
//! Both sides are reduced to the same four *observable fact kinds*, chosen
//! because each is unambiguously present in the native encoding AND has an
//! unambiguous textual form if the semantics survived:
//!
//! | fact      | native side                                     | C# side                        |
//! |-----------|-------------------------------------------------|--------------------------------|
//! | `call`    | `bl` / `blr` (`InsnClass::Call`)                | a call expression `ident(`     |
//! | `cond`    | conditional branch (`b.cc`, `cbz`, `cbnz`, `tbz`)| `if`, `while`, or `? :`        |
//! | `offset`  | load/store addend, SP/FP-relative EXCLUDED      | an additive constant `+ N`     |
//! | `imm`     | `mov`-class constant (`NativeFacts::mov_imm`)   | an integer literal             |
//!
//! Matching is by COUNT per kind, not by identity: `native` facts are the
//! denominator and the C# count is what can be credited, so
//! `sum(min(native, cs)) / sum(native)` is the headline "how much of the
//! disassembly is visible in the C#".
//!
//! ## Documented limits (read these before believing a number)
//!
//! * **`offset` is a LOWER BOUND.** When metadata proves what a load is,
//!   the renderer prints a NAME — `this.transform`, `SomeType.SomeField` —
//!   and the numeric offset legitimately disappears from the text. There is no
//!   way to tell that apart from a lost offset by scanning text, so an
//!   unmatched offset is not automatically a defect. SP/FP-relative accesses
//!   (base 31/29) are excluded outright: the frame prologue/epilogue is
//!   deliberately elided and would otherwise dominate the denominator.
//! * **`imm` is also diluted** the same way (an immediate that is an address
//!   becomes a metadata name).
//! * **`call` and `cond` are NOT diluted**, which is why they carry the
//!   weight: a native call and a native conditional branch each have exactly
//!   one way to survive, and the two ways they can legitimately vanish
//!   (an excised IL2CPP runtime call, a vacuous branch) are counted separately
//!   as `dropped_calls` / `lost_conds` rather than hidden.
//! * The C# scan is LEXICAL — identifiers, numbers and operators, skipping
//!   comments, string and char literals. It is not a C# parser. Commas inside
//!   a comment cannot leak in; a literal inside a string cannot either.
//!
//! Enabled by `ILR_CONCORD=1`. `ILR_CONCORD_JSON=1` appends a JSONL record per
//! offending method to `concord_findings.jsonl`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::verify::{self, DecodedInsn, InsnClass};

// ---------------------------------------------------------------------------
// Aggregate stats
// ---------------------------------------------------------------------------

pub struct ConcordStats {
    pub methods: AtomicU64,
    pub native_calls: AtomicU64,
    pub cs_calls: AtomicU64,
    pub native_conds: AtomicU64,
    pub cs_conds: AtomicU64,
    pub native_offsets: AtomicU64,
    pub matched_offsets: AtomicU64,
    pub native_imms: AtomicU64,
    pub matched_imms: AtomicU64,
    /// Methods whose emitted C# contains FEWER calls than the native stream.
    pub methods_dropped_calls: AtomicU64,
    /// Methods whose emitted C# contains FEWER conditions than the native
    /// stream (a branch whose test vanished).
    pub methods_lost_conds: AtomicU64,
    /// CONDITION SITES in the emitted text whose tested value is an unknown
    /// (`(? <op>` / `__unknown <op>` / `__abi <op>`).
    ///
    /// This exists because the method-level `methods_lost_conds` above reported
    /// `0` on a run whose own dump contained real `(? != 0)` sites — a counter
    /// that cannot see the defect it names is worse than none. It is measured
    /// from the FINAL TEXT, the same source `scripts/audit_dump.py` reads, so
    /// the report and the audit cannot disagree about whether a site exists.
    pub lost_cond_sites: AtomicU64,
    /// `&&`/`||` emitted across all methods (short-circuit reconstruction).
    pub cs_logical_ops: AtomicU64,
    /// Methods whose emitted C# contains MORE calls than the native stream
    /// (one native `bl` emitted twice — the "synthesis" family).
    pub methods_extra_calls: AtomicU64,
    /// Methods the pipeline DECLINED to decompile (too large, unsupported
    /// pattern, unstructured body). Their C# is a marker, not a translation, so
    /// scoring them would report every one of their native facts as lost — and
    /// they are already reported by the passes that declined them.
    pub methods_declined: AtomicU64,
    /// CFG blocks the structured walk never entered, over every method checked,
    /// and the total block count. This is the shortfall the IR-level oracle
    /// cannot see: `verify` only asks whether a native instruction reached the
    /// IR, never whether the walk emitted it.
    pub blocks_unemitted: AtomicU64,
    pub blocks_total: AtomicU64,
    /// Methods with at least one never-walked block.
    pub methods_partial: AtomicU64,
    /// Native `bl` sites removed deliberately by the IL2CPP-runtime-call
    /// excision (each ledgered). Reported separately: these are in the binary
    /// but never in the C# source, so scoring them as lost semantics made the
    /// checker report plumbing removal as a decompiler defect.
    pub runtime_calls: AtomicU64,
    /// Native `bl` sites the renderer presented as a member access or operator
    /// (`PropertyAccessorCall`). Reported separately for the same reason as
    /// `runtime_calls`: the folded form IS in the text, so scoring it as a
    /// dropped call measures the scanner, not the decompiler.
    pub accessor_calls: AtomicU64,
    /// Native conditional branches erased on purpose (IL2CPP null-check
    /// collapse, `VacuousBranch`). Reported for the same reason.
    pub vacuous_conds: AtomicU64,
}

/// Marker the structurer emits when an indirect jump's dispatch was NOT
/// recovered (`Op::IndirectCall` reaching the walk means the jump-table
/// recognizer never fired). The case bodies that follow are still emitted, but
/// only reachable through a forced `goto`, so a reader must be told.
pub const DISPATCH_LOST_MARKER: &str = "indirect jump (dispatch not recovered)";

/// Per-method FIDELITY verdict, derived from the FINAL emitted text.
///
/// The fact-level concordance answers "how many native facts are visible". It
/// does NOT answer the question a reader actually has: **is the C# in front of
/// me trustworthy?** A method can show 100% of its calls and conditions and
/// still be unreadable because a value it uses has no name, because a whole
/// region was never walked, or because the control flow was reconstructed as
/// arbitrary `goto`s. These counters make each of those a separate, countable
/// defect class instead of the blanket judgement "looks wrong".
///
/// `clean` is deliberately strict: it requires that the text names every value
/// it uses, that no CFG block went unemitted, that no dispatch was lost, and
/// that the structurer needed no arbitrary `goto`. That makes it a LOWER bound
/// on how good the output is, which is the only kind of number worth reporting.
pub struct FidelityStats {
    /// Methods classified (excludes declined bodies — see `is_declined`).
    pub verdicts: AtomicU64,
    /// No defect class present at all.
    pub clean: AtomicU64,
    pub with_unknown: AtomicU64,
    pub unknown_refs: AtomicU64,
    /// `__unknown` used where a value is CONSUMED — a receiver or an argument.
    /// These are the ones that make a call site meaningless; an `__unknown` in
    /// an unused temporary does not.
    pub unknown_consumed: AtomicU64,
    /// Methods containing an unresolved indirect dispatch.
    pub dispatch_unrecovered: AtomicU64,
    /// Methods with at least one CFG block the structured walk never entered.
    pub with_unwalked_blocks: AtomicU64,
    pub with_goto: AtomicU64,
    pub goto_refs: AtomicU64,
    /// Methods whose text still shows raw register spellings (`x19`, `g14`) —
    /// a register the renderer could not map to a named local.
    pub with_raw_register: AtomicU64,
}

pub static FIDELITY: FidelityStats = FidelityStats {
    verdicts: AtomicU64::new(0),
    clean: AtomicU64::new(0),
    with_unknown: AtomicU64::new(0),
    unknown_refs: AtomicU64::new(0),
    unknown_consumed: AtomicU64::new(0),
    dispatch_unrecovered: AtomicU64::new(0),
    with_unwalked_blocks: AtomicU64::new(0),
    with_goto: AtomicU64::new(0),
    goto_refs: AtomicU64::new(0),
    with_raw_register: AtomicU64::new(0),
};

/// Classify one method's final text into the defect classes above.
///
/// Deliberately TEXT-level, not IR-level: it measures what a reader sees, and
/// it therefore also catches defects introduced by passes that run after the IR
/// checkers (the renderer, the structurer, the cleanup passes).
fn classify_fidelity(cs_text: &str, blocks_unemitted: usize) {
    let _ = FIDELITY.verdicts.fetch_add(1, Ordering::Relaxed);

    let unknown_refs = cs_text.matches("__unknown").count() as u64;
    // `__unknown` in a consumed position: as a receiver (`__unknown.`) or as an
    // argument after a comma. Both mean a real dependency was lost.
    let unknown_consumed = cs_text.matches("(__unknown").count() as u64
        + cs_text.matches(", __unknown").count() as u64;
    let goto_refs = cs_text.matches("goto L").count() as u64;
    let dispatch = cs_text.contains(DISPATCH_LOST_MARKER);
    let unwalked = blocks_unemitted > 0;
    // Raw register spellings the renderer prints verbatim when a register has no
    // local mapping: the frame pointer (`x19`), the link register (`g14`), or
    // one of the ABI argument registers beyond the named parameters.
    let raw_register = RAW_REGISTER_NAMES.iter().any(|r| {
        cs_text
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|tok| tok == *r)
    });

    if unknown_refs > 0 {
        let _ = FIDELITY.with_unknown.fetch_add(1, Ordering::Relaxed);
        let _ = FIDELITY
            .unknown_refs
            .fetch_add(unknown_refs, Ordering::Relaxed);
    }
    if unknown_consumed > 0 {
        let _ = FIDELITY
            .unknown_consumed
            .fetch_add(unknown_consumed, Ordering::Relaxed);
    }
    if goto_refs > 0 {
        let _ = FIDELITY.with_goto.fetch_add(1, Ordering::Relaxed);
        let _ = FIDELITY.goto_refs.fetch_add(goto_refs, Ordering::Relaxed);
    }
    if dispatch {
        let _ = FIDELITY
            .dispatch_unrecovered
            .fetch_add(1, Ordering::Relaxed);
    }
    if unwalked {
        let _ = FIDELITY
            .with_unwalked_blocks
            .fetch_add(1, Ordering::Relaxed);
    }
    if raw_register {
        let _ = FIDELITY.with_raw_register.fetch_add(1, Ordering::Relaxed);
    }
    if unknown_refs == 0 && goto_refs == 0 && !dispatch && !unwalked && !raw_register {
        let _ = FIDELITY.clean.fetch_add(1, Ordering::Relaxed);
    }
}

/// Register names the arm64/arm32 lifters hand the renderer verbatim. None of
/// them is a C# nameable value, so a token equal to one of these in the output
/// means the local-mapping step gave up on that register.
const RAW_REGISTER_NAMES: &[&str] = &[
    "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21",
    "x22", "x23", "x24", "x25", "x26", "x27", "x28", "g14", "g15",
];

pub static STATS: ConcordStats = ConcordStats {
    methods: AtomicU64::new(0),
    native_calls: AtomicU64::new(0),
    cs_calls: AtomicU64::new(0),
    native_conds: AtomicU64::new(0),
    cs_conds: AtomicU64::new(0),
    native_offsets: AtomicU64::new(0),
    matched_offsets: AtomicU64::new(0),
    native_imms: AtomicU64::new(0),
    matched_imms: AtomicU64::new(0),
    methods_dropped_calls: AtomicU64::new(0),
    methods_lost_conds: AtomicU64::new(0),
    lost_cond_sites: AtomicU64::new(0),
    cs_logical_ops: AtomicU64::new(0),
    methods_extra_calls: AtomicU64::new(0),
    methods_declined: AtomicU64::new(0),
    blocks_unemitted: AtomicU64::new(0),
    blocks_total: AtomicU64::new(0),
    methods_partial: AtomicU64::new(0),
    runtime_calls: AtomicU64::new(0),
    accessor_calls: AtomicU64::new(0),
    vacuous_conds: AtomicU64::new(0),
};

/// Per-method records, kept for the worst-offenders table. Bounded by the
/// caller (`ILR_CONCORD_TOP`, default 40) so a full-binary run cannot grow it
/// without limit.
static WORST: Mutex<Vec<MethodConcord>> = Mutex::new(Vec::new());

/// Methods whose structured walk never emitted at least one CFG block:
/// `(rva, unemitted, total)`. This is the explicit "code may have vanished"
/// ledger — the one class that must never be inferred from a percentage.
/// Bounded so a full-binary run cannot grow it without limit.
static BLOCK_LOSS: Mutex<Vec<(u64, usize, usize)>> = Mutex::new(Vec::new());
const BLOCK_LOSS_CAP: usize = 512;

#[derive(Debug, Clone)]
pub struct MethodConcord {
    pub rva: u64,
    pub native_calls: usize,
    pub cs_calls: usize,
    /// Native `bl` sites the pipeline removed ON PURPOSE, each one recorded in
    /// the verify ledger as `DroppedRuntimeCall`. Already SUBTRACTED from
    /// `native_calls`; kept so the report can show the deliberate part of the
    /// denominator instead of hiding it inside a percentage.
    pub runtime_calls: usize,
    /// Native `bl` sites the renderer presented as a member access or operator
    /// (each ledgered as `PropertyAccessorCall`). Already SUBTRACTED from
    /// `native_calls`; kept so the report shows them as their own number.
    pub accessor_calls: usize,
    /// Native conditional branches the pipeline erased on purpose (the IL2CPP
    /// null-check collapse, each ledgered as `VacuousBranch`). Already
    /// SUBTRACTED from `native_conds`; kept so the report shows them.
    pub vacuous_conds: usize,
    pub native_conds: usize,
    pub cs_conds: usize,
    /// `&&` / `||` occurrences in the emitted body. Deliberately NOT folded
    /// into `cs_conds` (see `CsFacts::logical_ops`), but carried here so the
    /// aggregate report can show how much short-circuit reconstruction the
    /// structurer did — a fused guard chain is ONE `if` token standing for
    /// several native branches, and without this the headline silently blames
    /// the fusion for "lost" conditions.
    pub cs_logical_ops: usize,
    pub native_offsets: usize,
    pub matched_offsets: usize,
    pub native_imms: usize,
    pub matched_imms: usize,
    /// CFG blocks the structured walk never entered, and the total. A non-zero
    /// `blocks_unemitted` means the text is missing whole regions regardless of
    /// what the fact counts say.
    pub blocks_unemitted: usize,
    pub blocks_total: usize,
}

impl MethodConcord {
    /// Native facts the emitted text does not account for. `call` and `cond`
    /// shortfalls are real; an offset/imm shortfall may be a named field
    /// access (see the module docs).
    pub fn missing(&self) -> usize {
        self.native_calls.saturating_sub(self.cs_calls)
            + self.native_conds.saturating_sub(self.cs_conds)
            + (self.native_offsets - self.matched_offsets)
            + (self.native_imms - self.matched_imms)
    }
}

// ---------------------------------------------------------------------------
// C# side: lexical fact extraction
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct CsFacts {
    pub calls: usize,
    pub conds: usize,
    /// Integer literals that appear in an ADDITIVE position (`+ 76`, `- 0x4`).
    /// Those are the ones that can be, and are, memory offsets.
    pub additive_ints: Vec<i64>,
    /// Every integer literal in the body.
    pub int_literals: Vec<i64>,
    /// `&&` / `||` occurrences. NOT part of `conds`.
    ///
    /// Each is a short-circuit branch in the C# but only one `if`/`while` token,
    /// so a body that renders two native branches as `a && b` would look like it
    /// lost a condition. It is reported SEPARATELY rather than added to `conds`
    /// so the headline number cannot be improved by a scanner rule; the measured
    /// corpus emits these almost never (3 `&&` and 2 `||` in a 3,886-method
    /// dump), so they explain none of the shortfall.
    pub logical_ops: usize,
    /// `switch (…)` statements, and `case`/`default` labels inside them.
    pub switches: usize,
    pub case_labels: usize,
    /// `goto` statements — the structurer's explicit control transfers, which
    /// carry a branch that is NOT an `if`.
    pub gotos: usize,
}

const KEYWORDS: &[&str] = &[
    "if",
    "else",
    "while",
    "for",
    "do",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "return",
    "goto",
    "new",
    "typeof",
    "sizeof",
    "nameof",
    "checked",
    "unchecked",
    "lock",
    "using",
    "try",
    "catch",
    "finally",
    "throw",
    "catch",
    "in",
    "is",
    "as",
    "this",
    "true",
    "false",
    "null",
    "and",
    "or",
    "not",
];

fn is_keyword(w: &str) -> bool {
    KEYWORDS.contains(&w)
}

/// Parse a C# integer literal: decimal (optionally `L`/`UL` suffixed) or `0x`.
fn parse_int_literal(word: &str) -> Option<i64> {
    let w = word.trim_end_matches(['L', 'l', 'U', 'u']);
    if w.is_empty() {
        return None;
    }
    if let Some(hex) = w.strip_prefix("0x").or_else(|| w.strip_prefix("0X")) {
        return i64::from_str_radix(hex, 16).ok();
    }
    if !w.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    w.parse::<i64>().ok()
}

/// Extract the comparable facts from an emitted method body.
pub fn extract_facts(text: &str) -> CsFacts {
    let b: Vec<char> = text.chars().collect();
    let mut f = CsFacts::default();
    let mut i = 0usize;
    // Sign of the pending additive operator; `None` = not in an additive
    // position. Whitespace deliberately does NOT clear it: the renderer emits
    // `+304` and `+ 76` both.
    let mut pending: Option<i64> = None;
    while i < b.len() {
        let c = b[i];
        // line comment
        if c == '/' && b.get(i + 1) == Some(&'/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // block comment
        if c == '/' && b.get(i + 1) == Some(&'*') {
            i += 2;
            while i + 1 < b.len() && !(b[i] == '*' && b[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
            continue;
        }
        // string / char literal
        if c == '"' || c == '\'' {
            let q = c;
            i += 1;
            while i < b.len() && b[i] != q {
                if b[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            pending = None;
            continue;
        }
        if c.is_alphanumeric() || c == '_' {
            let s = i;
            while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            let word: String = b[s..i].iter().collect();
            if word.as_bytes()[0].is_ascii_digit() {
                if let Some(v) = parse_int_literal(&word) {
                    if let Some(sign) = pending.take() {
                        f.additive_ints.push(sign * v);
                    }
                    f.int_literals.push(v);
                }
                pending = None;
            } else {
                // A call is an identifier directly followed by `(` — and not a
                // keyword (`if (`, `while (` …) nor a declaration's parameter
                // list, which is always preceded by whitespace-then-`(` too but
                // whose identifier is the method NAME. That one call site is
                // counted; it is constant per method and is subtracted by the
                // caller via `signature_call_bias`.
                if !is_keyword(&word) && b.get(i) == Some(&'(') {
                    f.calls += 1;
                }
                match word.as_str() {
                    "if" | "while" => f.conds += 1,
                    "switch" => f.switches += 1,
                    "case" | "default" => f.case_labels += 1,
                    "goto" => f.gotos += 1,
                    // `for`/`do` are deliberately NOT counted as conditions:
                    // `do { … } while (c);` already counts its `while`, and
                    // counting the `do` too would double it.
                    _ => {}
                }
                pending = None;
            }
            continue;
        }
        match c {
            '&' if b.get(i + 1) == Some(&'&') => {
                f.logical_ops += 1;
                pending = None;
                i += 2;
            }
            '|' if b.get(i + 1) == Some(&'|') => {
                f.logical_ops += 1;
                pending = None;
                i += 2;
            }
            '+' => {
                pending = Some(1);
                i += 1;
            }
            '-' => {
                pending = Some(-1);
                i += 1;
            }
            '?' if b.get(i + 1) == Some(&':') => {
                f.conds += 1;
                pending = None;
                i += 2;
            }
            _ => {
                // Whitespace keeps any pending additive sign; anything else
                // (`,`, `)`, `*`, `(` …) ends the additive position.
                if !c.is_whitespace() {
                    pending = None;
                }
                i += 1;
            }
        }
    }
    f
}

/// The rendered method always starts with `[modifiers] <ret> <Name>(params)`,
/// which the lexical scan counts as one call. It is not a native `bl`, so it
/// must be removed from the C# call count.
fn signature_call_bias(text: &str) -> usize {
    match text.find('{') {
        Some(br) => {
            let head = &text[..br];
            if head.contains('(') { 1 } else { 0 }
        }
        None => 0,
    }
}

// ---------------------------------------------------------------------------
// Native side
// ---------------------------------------------------------------------------

/// Is this decoded instruction a CONDITIONAL branch? (`ret`, `br` and plain
/// `b` transfer control without testing anything — they take no condition, so
/// they must not enter the denominator.)
fn is_conditional_branch(m: &str) -> bool {
    match m {
        "cbz" | "cbnz" | "tbz" | "tbnz" => true,
        _ => m.starts_with("b.") && !matches!(m, "b.al"),
    }
}

/// SP (31) and FP (29/`x29`) relative accesses are frame traffic: the prologue
/// and epilogue are deliberately elided from the pseudo-C, so counting them
/// would swamp the metric with facts that are correct to omit.
fn is_frame_base(reg: u16) -> bool {
    reg == 31 || reg == 29
}

/// Count only the native facts this module can judge. Returns the numbers
/// `(calls, conds, offsets, imms)`.
fn native_counts(decoded: &[DecodedInsn]) -> (usize, usize, usize, usize) {
    let mut calls = 0;
    let mut conds = 0;
    let mut offsets = 0;
    let mut imms = 0;
    for d in decoded {
        if d.class == InsnClass::Call {
            calls += 1;
        }
        if is_conditional_branch(&d.mnemonic) {
            conds += 1;
        }
        if let Some((base, addend)) = d.facts.mem {
            if addend != 0 && !is_frame_base(base) {
                offsets += 1;
            }
        }
        if d.facts.mov_imm.is_some() {
            imms += 1;
        }
    }
    (calls, conds, offsets, imms)
}

/// Multiset intersection size: how many of `wanted` are present in `have`.
fn multiset_hits(wanted: &[i64], have: &[i64]) -> usize {
    if wanted.is_empty() {
        return 0;
    }
    let mut pool: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for &v in have {
        *pool.entry(v).or_insert(0) += 1;
    }
    let mut hits = 0;
    for &w in wanted {
        if let Some(n) = pool.get_mut(&w) {
            if *n > 0 {
                *n -= 1;
                hits += 1;
            }
        }
    }
    hits
}

/// Native CALL sites the pipeline removed deliberately, each one recorded in
/// the verify ledger as [`verify::LedgerKind::DroppedRuntimeCall`].
///
/// `il2cpp_codegen_initialize_class` / `InitializeMethodMetadata` thunks are
/// runtime metadata bookkeeping: the C# source never contained them, and
/// `resolve::key_function_recovery` Nops them. They ARE in the binary, so a raw
/// count charges them to the decompiler as lost semantics — measured on
/// 0x1fe2ca0, ALL 112 calls the checker called "missing" were this class, while
/// the C# had every one of the 160 managed calls. A deliberate, ledgered drop is
/// not a defect; it must be visible in the report as its own number rather than
/// silently subtracted.
fn runtime_dropped_calls(decoded: &[DecodedInsn]) -> usize {
    decoded
        .iter()
        .filter(|d| {
            d.class == InsnClass::Call
                && verify::ledger_has_kind(d.addr, verify::LedgerKind::DroppedRuntimeCall)
        })
        .count()
}

/// Native CALL sites the renderer presented as a MEMBER ACCESS or an operator
/// rather than as `callee(…)`, each recorded in the verify ledger as
/// [`verify::LedgerKind::PropertyAccessorCall`].
///
/// The idioms are deliberate and faithful: `obj.get_X()` → `obj.X`,
/// `Type.get_X()` → `Type.X`, `obj.set_X(v)` → `obj.X = v`, `String.Concat(a,b)`
/// → `a + b`. The call IS in the text — just not as an `ident(` token, which is
/// what `extract_facts` counts. So a raw native count charged each of them to
/// the decompiler as a call it DROPPED. `SwitchToRectTransform` @0xFA08 is the
/// clean case: 14 native `bl`, 4 `ident(` in the text, and exactly 10 folds —
/// the method's whole deficit. Subtracting a presentation choice that is
/// visible in the output is not a concession; it is the same correction
/// `runtime_dropped_calls` makes, and the number is reported on its own line so
/// it can never hide a real loss silently.
fn folded_accessor_calls(decoded: &[DecodedInsn]) -> usize {
    decoded
        .iter()
        .filter(|d| {
            d.class == InsnClass::Call
                && verify::ledger_has_kind(d.addr, verify::LedgerKind::PropertyAccessorCall)
        })
        .count()
}

/// Native conditional branches the pipeline erased ON PURPOSE, each one recorded
/// in the verify ledger as [`verify::LedgerKind::VacuousBranch`].
///
/// IL2CPP's null check is `cbnz w8, +8` over a `bl <raise_null>`; the `bl` is a
/// runtime helper and is excised, which leaves BOTH arms of the branch empty.
/// The branch then has no observable effect and no expressible C# form —
/// rendering it anyway is exactly the "impossible / inverted if scope" defect —
/// so `passes::drop_vacuous_cond_branches` Nops it and ledgers the address. The
/// checker counted it as a LOST condition regardless: 1,872 of them in the
/// reference sample, 45% of the entire condition shortfall. `verify`'s own
/// wrong-logic oracle already excuses these (it reports 193 missing conditions
/// while `concord` reported 1,208 lost-condition METHODS on the same run).
fn vacuous_condition_branches(decoded: &[DecodedInsn]) -> usize {
    decoded
        .iter()
        .filter(|d| {
            is_conditional_branch(&d.mnemonic)
                && verify::ledger_has_kind(d.addr, verify::LedgerKind::VacuousBranch)
        })
        .count()
}

/// Count condition sites in the FINAL TEXT whose tested value is an unknown.
///
/// Two spellings, both real defects:
///   * `(? != 0)` — the compared operand rendered as nothing at all;
///   * `if (__unknown != 0)` / `while (__abi …)` — the explicit unknown marker
///     standing where the compared value belongs.
///
/// Measured from the TEXT on purpose. A CFG- or IR-level proxy answers a
/// different question, and the run report used exactly such a proxy: it printed
/// `lost-condition methods=0` while the dump contained real `(? != 0)` sites.
/// Counting the same source `scripts/audit_dump.py` reads is what stops the two
/// from disagreeing about whether a site exists.
fn unknown_condition_sites(text: &str) -> usize {
    const OPS: [&str; 6] = ["!=", "==", ">=", "<=", ">", "<"];
    let is_comparison = |after: &str| OPS.iter().any(|op| after.starts_with(op));
    let mut n = 0usize;
    for marker in ["(?", "__unknown", "__abi"] {
        let mut from = 0usize;
        while let Some(i) = text[from..].find(marker) {
            let end = from + i + marker.len();
            // Whitespace is allowed between the tested value and its operator
            // (`(?  != 0)`) and nothing else is: an `__unknown` used as an
            // ARGUMENT (`f(__unknown, 0)`) or a plain value is not a condition
            // and must not be counted here.
            if is_comparison(text[end..].trim_start()) {
                n += 1;
            }
            from = end;
        }
    }
    n
}

/// Compare one method's emitted C# against its decoded native stream.
pub fn compare(cs_text: &str, decoded: &[DecodedInsn]) -> MethodConcord {
    let facts = extract_facts(cs_text);
    let cs_calls = facts.calls.saturating_sub(signature_call_bias(cs_text));
    let (native_calls, native_conds, native_offsets, native_imms) = native_counts(decoded);
    let runtime_calls = runtime_dropped_calls(decoded);
    let accessor_calls = folded_accessor_calls(decoded);
    let vacuous_conds = vacuous_condition_branches(decoded);
    let native_calls = native_calls.saturating_sub(runtime_calls + accessor_calls);
    let native_conds = native_conds.saturating_sub(vacuous_conds);

    // The native constants that must still be visible as literals. ADRP/ADD
    // pairs are NOT included (`mov_imm` only records MOV-class forms), which
    // keeps out the addresses that legitimately become metadata names.
    let native_imm_vals: Vec<i64> = decoded
        .iter()
        .filter_map(|d| d.facts.mov_imm.map(|(_, v)| v as i64))
        .collect();

    let matched_offsets = multiset_hits(
        &decoded
            .iter()
            .filter_map(|d| d.facts.mem)
            .filter(|(base, addend)| *addend != 0 && !is_frame_base(*base))
            .map(|(_, addend)| addend)
            .collect::<Vec<_>>(),
        &facts.additive_ints,
    );
    let matched_imms = multiset_hits(&native_imm_vals, &facts.int_literals);

    MethodConcord {
        rva: 0,
        native_calls,
        cs_calls,
        runtime_calls,
        accessor_calls,
        vacuous_conds,
        native_conds,
        cs_conds: facts.conds,
        cs_logical_ops: facts.logical_ops,
        native_offsets,
        matched_offsets,
        native_imms,
        matched_imms,
        blocks_unemitted: 0,
        blocks_total: 0,
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// `true` when the emitted text is a marker rather than a translation — the
/// pipeline declined this method, so there is nothing to compare.
pub fn is_declined(cs_text: &str) -> bool {
    cs_text.contains("[recovery skipped:")
        || cs_text.contains("[recovery failed:")
        || cs_text.contains("// body not recoverable")
}

/// Run the concordance check for one method and fold it into the aggregates.
/// `cs_text` is the FINAL emitted body; `bytes` the method's native bytes.
pub fn check_and_log(cs_text: &str, bytes: &[u8], base_rva: u64, arch: Arch) {
    if is_declined(cs_text) {
        let _ = STATS.methods_declined.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let decoded = match arch {
        Arch::Arm64 => verify::decode_all_arm64(bytes, base_rva).0,
        Arch::Arm32 => verify::decode_all_arm32(bytes, base_rva, false).0,
        Arch::X64 => verify::decode_all_x86(bytes, base_rva).0,
    };
    // `ILR_CONCORD_METHOD=<rva|all>`: the per-method diff, printed for the one
    // method under investigation (or for every method with a shortfall). Read
    // once — it is consulted for every method in the run.
    static DIFF_FOR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let diff_for = DIFF_FOR.get_or_init(|| std::env::var("ILR_CONCORD_METHOD").ok());
    if let Some(want) = diff_for {
        if want.eq_ignore_ascii_case("all") {
            eprint!("{}", method_diff(cs_text, &decoded, base_rva));
        } else {
            let w = want.trim();
            let hit = w.eq_ignore_ascii_case(&format!("{base_rva:#x}"))
                || u64::from_str_radix(w.trim_start_matches("0x"), 16)
                    .map(|v| v == base_rva)
                    .unwrap_or(false);
            if hit {
                eprint!("{}", method_diff(cs_text, &decoded, base_rva));
            }
        }
    }

    // Structurer coverage for this method (set by the renderer on this
    // thread). Counted BEFORE the early bail-outs so a partial body is still
    // charged to the method that produced it.
    let (unemitted, blocks_total) = super::csharp::unemitted_blocks();
    let _ = STATS
        .blocks_unemitted
        .fetch_add(unemitted as u64, Ordering::Relaxed);
    let _ = STATS
        .blocks_total
        .fetch_add(blocks_total as u64, Ordering::Relaxed);
    if unemitted > 0 {
        let _ = STATS.methods_partial.fetch_add(1, Ordering::Relaxed);
        if std::env::var("ILR_CONCORD").is_ok() || std::env::var("ILR_COVLOSS").is_ok() {
            eprintln!(
                "[covloss] {base_rva:#x}: {unemitted}/{blocks_total} CFG blocks never emitted"
            );
        }
        if std::env::var("ILR_CONCORD").is_ok() {
            if let Ok(mut v) = BLOCK_LOSS.lock() {
                if v.len() < BLOCK_LOSS_CAP {
                    v.push((base_rva, unemitted, blocks_total));
                }
            }
        }
    }

    let mut c = compare(cs_text, &decoded);
    c.rva = base_rva;
    c.blocks_unemitted = unemitted;
    c.blocks_total = blocks_total;

    // The reader-facing verdict, recorded for every method that produced text.
    classify_fidelity(cs_text, unemitted);

    let _ = STATS.methods.fetch_add(1, Ordering::Relaxed);
    let _ = STATS
        .native_calls
        .fetch_add(c.native_calls as u64, Ordering::Relaxed);
    let _ = STATS
        .runtime_calls
        .fetch_add(c.runtime_calls as u64, Ordering::Relaxed);
    let _ = STATS
        .accessor_calls
        .fetch_add(c.accessor_calls as u64, Ordering::Relaxed);
    let _ = STATS
        .vacuous_conds
        .fetch_add(c.vacuous_conds as u64, Ordering::Relaxed);
    let _ = STATS
        .cs_calls
        .fetch_add(c.cs_calls as u64, Ordering::Relaxed);
    let _ = STATS
        .native_conds
        .fetch_add(c.native_conds as u64, Ordering::Relaxed);
    let _ = STATS
        .cs_conds
        .fetch_add(c.cs_conds as u64, Ordering::Relaxed);
    let _ = STATS
        .cs_logical_ops
        .fetch_add(c.cs_logical_ops as u64, Ordering::Relaxed);
    let _ = STATS
        .native_offsets
        .fetch_add(c.native_offsets as u64, Ordering::Relaxed);
    let _ = STATS
        .matched_offsets
        .fetch_add(c.matched_offsets as u64, Ordering::Relaxed);
    let _ = STATS
        .native_imms
        .fetch_add(c.native_imms as u64, Ordering::Relaxed);
    let _ = STATS
        .matched_imms
        .fetch_add(c.matched_imms as u64, Ordering::Relaxed);

    let dropped = c.native_calls > c.cs_calls;
    let extra = c.cs_calls > c.native_calls;
    let lost = c.native_conds > c.cs_conds;
    if dropped {
        let _ = STATS.methods_dropped_calls.fetch_add(1, Ordering::Relaxed);
    }
    if extra {
        let _ = STATS.methods_extra_calls.fetch_add(1, Ordering::Relaxed);
    }
    if lost {
        let _ = STATS.methods_lost_conds.fetch_add(1, Ordering::Relaxed);
    }
    // The site-level truth, from the text this method actually emitted.
    let unknown_cond_sites = unknown_condition_sites(cs_text);
    if unknown_cond_sites > 0 {
        let _ = STATS
            .lost_cond_sites
            .fetch_add(unknown_cond_sites as u64, Ordering::Relaxed);
    }

    if std::env::var("ILR_CONCORD_JSON").is_ok() && (dropped || extra || lost) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("concord_findings.jsonl")
        {
            let _ = writeln!(
                f,
                "{{\"rva\":{:#x},\"native_calls\":{},\"cs_calls\":{},\"native_conds\":{},\
                 \"cs_conds\":{},\"native_offsets\":{},\"matched_offsets\":{},\
                 \"native_imms\":{},\"matched_imms\":{}}}",
                c.rva,
                c.native_calls,
                c.cs_calls,
                c.native_conds,
                c.cs_conds,
                c.native_offsets,
                c.matched_offsets,
                c.native_imms,
                c.matched_imms
            );
        }
    }

    let top: usize = std::env::var("ILR_CONCORD_TOP")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(40);
    if top > 0 {
        if let Ok(mut w) = WORST.lock() {
            if w.len() < top {
                w.push(c.clone());
                w.sort_by(|a, b| b.missing().cmp(&a.missing()));
            } else if w.last().is_some_and(|l| c.missing() > l.missing()) {
                w.push(c.clone());
                w.sort_by(|a, b| b.missing().cmp(&a.missing()));
                w.truncate(top);
            }
        }
    }

    // `blocks` is part of the line on purpose: a method can show every call and
    // condition it has and still be missing whole regions, and without the
    // block count here the shortfall is only a global number no method can be
    // charged with (the same reason `dropped`/`lost` are per-method).
    if std::env::var("ILR_CONCORD").is_ok() && (dropped || lost || c.blocks_unemitted > 0) {
        eprintln!(
            "[concord] {:#x}: calls native={} cs={} | conds native={} cs={} | \
             offsets {}/{} | imms {}/{} | blocks {}/{} | acc={} vac={}",
            c.rva,
            c.native_calls,
            c.cs_calls,
            c.native_conds,
            c.cs_conds,
            c.matched_offsets,
            c.native_offsets,
            c.matched_imms,
            c.native_imms,
            c.blocks_total - c.blocks_unemitted,
            c.blocks_total,
            c.accessor_calls,
            c.vacuous_conds
        );
    }
}

// ---------------------------------------------------------------------------
// Per-method diagnosis
// ---------------------------------------------------------------------------

/// Native conditional branches, in address order: `(addr, mnemonic, target)`,
/// with `target == 0` for a form whose target this decoder does not expose.
fn native_cond_branches(decoded: &[DecodedInsn]) -> Vec<(u64, String, u64)> {
    decoded
        .iter()
        .filter(|d| is_conditional_branch(&d.mnemonic))
        .map(|d| {
            (
                d.addr,
                d.mnemonic.clone(),
                d.facts.cond_branch.map(|c| c.1).unwrap_or(0),
            )
        })
        .collect()
}

/// The C# lines that carry a branch, in order — the other half of the
/// side-by-side.
fn cs_cond_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let l = raw.trim();
        if l.contains("if (") || l.contains("while (") || l.starts_with("case ") {
            out.push(l.to_string());
        } else if l.contains("goto ") && !l.contains("//") {
            out.push(l.to_string());
        }
    }
    out
}

/// How concentrated the native branch targets are. One target taking most of
/// the branches is the compiler's unrolled early-exit form (a guard chain);
/// many near-equal counts is a switch lowering. The structurer's failure modes
/// are completely different for the two, so the diff names which one it is.
fn target_concentration(branches: &[(u64, String, u64)]) -> Vec<(u64, usize)> {
    let mut hist: HashMap<u64, usize> = HashMap::new();
    for (_, _, t) in branches {
        *hist.entry(*t).or_insert(0) += 1;
    }
    let mut v: Vec<(u64, usize)> = hist.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// Per-method C# ↔ disassembly diff, for `ILR_CONCORD_METHOD=<rva>`.
///
/// The aggregate report says HOW MUCH is missing across a run; this says, for
/// one method, WHICH native facts the emitted body does not account for and in
/// what SHAPE the loss is — the difference between a number and a diagnosis.
/// It is deliberately a Rust function over the decoded stream and the emitted
/// text, not a script over a dump: it runs inside the pipeline, so the text it
/// reads is the text that would be shipped.
///
/// `ILR_CONCORD_METHOD=all` prints it for every method with a shortfall.
pub fn method_diff(cs_text: &str, decoded: &[DecodedInsn], rva: u64) -> String {
    let facts = extract_facts(cs_text);
    // Every number below comes from `compare`, so the diagnosis can never drift
    // from the headline: the deliberate exclusions (runtime thunks, property
    // folds, the null-check collapse) are applied in exactly one place.
    let c = compare(cs_text, decoded);
    let cs_calls = c.cs_calls;
    let native_calls = c.native_calls;
    let native_conds = c.native_conds;
    let cs_conds = c.cs_conds;
    let native_offsets = c.native_offsets;
    let matched_offsets = c.matched_offsets;
    let native_imms = c.native_imms;
    let matched_imms = c.matched_imms;
    let branches = native_cond_branches(decoded);

    let mut o = String::new();
    let pct = |n: usize, d: usize| {
        if d == 0 {
            100.0
        } else {
            100.0 * n as f64 / d as f64
        }
    };
    o.push_str(&format!(
        "[concord-diff] {rva:#x}: {} decoded native insns\n",
        decoded.len()
    ));
    let (un, tot) = super::csharp::unemitted_blocks();
    if tot > 0 {
        o.push_str(&format!(
            "  walk    {un} of {tot} CFG blocks never entered{}\n",
            if un > 0 {
                " — the text is MISSING whole regions (a `br`/dispatch the structurer could not resolve ends its block, so the code after it is unreachable to the walk)"
            } else {
                ""
            }
        ));
    }

    // ---- calls ---------------------------------------------------------
    let mut native_targets: HashMap<u64, usize> = HashMap::new();
    let mut indirect = 0usize;
    for d in decoded {
        if let Some((t, ind)) = d.facts.call {
            if ind {
                indirect += 1;
            }
            *native_targets.entry(t).or_insert(0) += 1;
        }
    }
    let rt_calls = runtime_dropped_calls(decoded);
    let accessor_calls = folded_accessor_calls(decoded);
    let recoverable_calls = native_calls;
    o.push_str(&format!(
        "  calls   native={recoverable_calls} ({} distinct targets, {indirect} indirect)  \
         in-C#={cs_calls} ({:.1}%)\n",
        native_targets.len(),
        pct(recoverable_calls.min(cs_calls), recoverable_calls)
    ));
    if rt_calls > 0 {
        o.push_str(&format!(
            "          {rt_calls} further native `bl` are IL2CPP runtime-init thunks \
             — removed on purpose, excluded here\n"
        ));
    }
    if accessor_calls > 0 {
        o.push_str(&format!(
            "          {accessor_calls} further native `bl` are property/operator folds \
             (`obj.get_X()` → `obj.X`, `String.Concat` → `+`) — in the text, but not as \
             `ident(`, so excluded here rather than scored as dropped\n"
        ));
    }
    if cs_calls > recoverable_calls {
        o.push_str(&format!(
            "          OVER-EMISSION: {} call expressions for {recoverable_calls} native `bl`. \
             One native call emitted at several use sites = a call result that was never \
             bound to a name (see the call-binding pass).\n",
            cs_calls - recoverable_calls
        ));
    }

    // ---- conditions ----------------------------------------------------
    o.push_str(&format!(
        "  conds   native={native_conds}  in-C#={} ({:.1}%)  [&&/||={} switch={} case={} goto={}]\n",
        cs_conds,
        pct(native_conds.min(cs_conds), native_conds),
        facts.logical_ops,
        facts.switches,
        facts.case_labels,
        facts.gotos
    ));
    if c.vacuous_conds > 0 {
        o.push_str(&format!(
            "          {} native conditional branch(es) are the IL2CPP null-check \
             collapse — both arms identical, no expressible `if` form, excluded here\n",
            c.vacuous_conds
        ));
    }
    let mut by_mn: HashMap<&str, usize> = HashMap::new();
    for (_, m, _) in &branches {
        *by_mn.entry(m.as_str()).or_insert(0) += 1;
    }
    let mut mn: Vec<(&str, usize)> = by_mn.into_iter().collect();
    mn.sort_by(|a, b| b.1.cmp(&a.1));
    o.push_str(&format!(
        "          native forms: {}\n",
        mn.iter()
            .take(8)
            .map(|(m, c)| format!("{m}={c}"))
            .collect::<Vec<_>>()
            .join(" ")
    ));
    let conc = target_concentration(&branches);
    if let Some((t, c)) = conc.first() {
        let share = if branches.is_empty() {
            0.0
        } else {
            100.0 * *c as f64 / branches.len() as f64
        };
        o.push_str(&format!(
            "          {} distinct taken targets; most-taken {t:#x} by {c} branches ({share:.0}%)",
            conc.len()
        ));
        if share >= 25.0 {
            o.push_str(" → GUARD CHAIN (unrolled early-exit, one shared join)\n");
        } else {
            o.push_str(" → dispatch / switch lowering\n");
        }
    }

    // ---- offsets / imms ------------------------------------------------
    o.push_str(&format!(
        "  offsets native={native_offsets} matched={matched_offsets} ({:.1}%) \
         [LOWER BOUND: a named field access legitimately hides the number]\n",
        pct(matched_offsets, native_offsets)
    ));
    o.push_str(&format!(
        "  imms    native={native_imms} matched={matched_imms} ({:.1}%) \
         [diluted the same way]\n",
        pct(matched_imms, native_imms)
    ));

    // ---- the side-by-side ---------------------------------------------
    const SHOW: usize = 16;
    if native_conds > cs_conds {
        o.push_str(&format!(
            "  -- native branches (first {SHOW} of {}) --\n",
            branches.len()
        ));
        for (a, m, t) in branches.iter().take(SHOW) {
            o.push_str(&format!("     {a:#x}  {m} -> {t:#x}\n"));
        }
        let cl = cs_cond_lines(cs_text);
        o.push_str(&format!(
            "  -- C# branch lines (first {SHOW} of {}) --\n",
            cl.len()
        ));
        for l in cl.iter().take(SHOW) {
            let mut l = l.replace('\n', " ");
            if l.len() > 120 {
                l.truncate(120);
                l.push('…');
            }
            o.push_str(&format!("     {l}\n"));
        }
    }
    o
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Arm64,
    Arm32,
    X64,
}

/// Human-readable aggregate, printed at dump end when `ILR_CONCORD=1`.
pub fn report() -> String {
    let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let methods = g(&STATS.methods);
    if methods == 0 {
        return String::new();
    }
    let nc = g(&STATS.native_calls);
    let cc = g(&STATS.cs_calls);
    let ncond = g(&STATS.native_conds);
    let ccond = g(&STATS.cs_conds);
    let noff = g(&STATS.native_offsets);
    let moff = g(&STATS.matched_offsets);
    let nimm = g(&STATS.native_imms);
    let mimm = g(&STATS.matched_imms);

    // `call` + `cond` are the undiluted kinds; they are what the headline is
    // built from, so the number cannot be flattered by metadata naming.
    let undiluted_native = nc + ncond;
    let undiluted_visible = nc.min(cc) + ncond.min(ccond);
    let pct = |num: u64, den: u64| {
        if den == 0 {
            100.0
        } else {
            100.0 * num as f64 / den as f64
        }
    };

    let mut out = String::new();
    out.push_str(&format!(
        "[concord] methods={methods} (declined/not-decompiled, excluded: {})\n",
        g(&STATS.methods_declined)
    ));
    out.push_str(&format!(
        "[concord] C#-vs-disassembly (undiluted kinds): \
         {undiluted_visible}/{undiluted_native} facts visible = {:.1}%\n",
        pct(undiluted_visible, undiluted_native)
    ));
    out.push_str(&format!(
        "[concord]   calls:  native={nc} in-C#={cc} ({:.1}% visible)  \
         dropped-call methods={}  extra-call methods={}\n",
        pct(nc.min(cc), nc),
        g(&STATS.methods_dropped_calls),
        g(&STATS.methods_extra_calls)
    ));
    // The deliberate part of the call denominator, shown rather than netted
    // away: IL2CPP runtime-init thunks are removed on purpose (and ledgered),
    // and they are NOT in the C# source any more than the prologue is.
    let rt = g(&STATS.runtime_calls);
    if rt > 0 {
        out.push_str(&format!(
            "[concord]           of which {rt} native `bl` are IL2CPP runtime-init thunks \
             (removed on purpose, ledgered as dropped_runtime_call, EXCLUDED above)\n"
        ));
    }
    // Same treatment for the property/operator folds: the call is IN the text
    // as a member access (`obj.get_X()` → `obj.X`), so counting it as missing
    // measured the scanner's `ident(` rule rather than the decompiler.
    let acc = g(&STATS.accessor_calls);
    if acc > 0 {
        out.push_str(&format!(
            "[concord]           of which {acc} native `bl` are property/operator folds \
             (`obj.get_X()` → `obj.X`, `String.Concat` → `+`), presented as member \
             access rather than `ident(`, ledgered as property_accessor_call and EXCLUDED above\n"
        ));
    }
    out.push_str(&format!(
        "[concord]   conds:  native={ncond} in-C#={ccond} ({:.1}% visible)  \
         lost-condition methods={}\n",
        pct(ncond.min(ccond), ncond),
        g(&STATS.methods_lost_conds)
    ));
    // A method-level proxy is not allowed to be the only answer: if the TEXT
    // contains condition sites testing an unknown value, they are printed, so a
    // run can never claim a clean condition corpus while its own output
    // contradicts it. `lift_x86::lost_condition_materialized` is the same class
    // counted at its point of creation (a flag consumer that found no live
    // compare), which is what makes the residue actionable rather than merely
    // visible.
    let lost_sites = g(&STATS.lost_cond_sites);
    if lost_sites > 0 {
        out.push_str(&format!(
            "[concord]           {lost_sites} condition site(s) test an UNKNOWN value \
             (`(? <op>` / `__unknown <op>`): the branch is real, its test is not\n"
        ));
    }
    // Short-circuit reconstruction, reported separately and NOT folded into
    // `conds` above: `if (a == 0 || b == 0)` is one `if` token for two native
    // branches, so a rising count here explains (and offsets) a flat or falling
    // `in-C#` condition count after guard-chain fusion. Keeping the two apart is
    // what stops the headline being improvable by a scanner rule.
    out.push_str(&format!(
        "[concord]           of which {} are short-circuit operators (&&/||) \
         reconstructed from guard chains — one `if` token standing for several \
         native branches, counted separately so `conds` cannot be inflated\n",
        g(&STATS.cs_logical_ops)
    ));
    // The second deliberate share of the condition denominator: IL2CPP's
    // null-check collapse leaves a branch whose two arms are identical, which
    // has no expressible C# form (rendering one is the "impossible if scope"
    // defect). `passes::drop_vacuous_cond_branches` Nops and ledgers it.
    let vac = g(&STATS.vacuous_conds);
    if vac > 0 {
        out.push_str(&format!(
            "[concord]           of which {vac} native conditional branches are the \
             IL2CPP null-check collapse — both arms identical once the runtime \
             raise was excised, so they have no expressible `if` form, ledgered as \
             vacuous_branch and EXCLUDED above\n"
        ));
    }
    out.push_str(&format!(
        "[concord]   offsets (LOWER BOUND, named field accesses hide the \
         number): {moff}/{noff} = {:.1}%\n",
        pct(moff, noff)
    ));
    out.push_str(&format!(
        "[concord]   imms (diluted by metadata naming): {mimm}/{nimm} = {:.1}%\n",
        pct(mimm, nimm)
    ));

    // Reader-facing fidelity verdicts. The fact percentages above can look
    // healthy on a method whose text a human still cannot read, so these count
    // the defect classes that make the output unusable and are the numbers the
    // remaining work should be driving down.
    let fv = g(&FIDELITY.verdicts);
    if fv > 0 {
        let fc = g(&FIDELITY.clean);
        out.push_str(&format!(
            "[concord] fidelity verdicts ({fv} methods, from the FINAL text):\n"
        ));
        out.push_str(&format!(
            "[concord]   CLEAN (no __unknown, no goto, every CFG block emitted, no lost \
             dispatch): {fc} ({:.1}%)\n",
            pct(fc, fv)
        ));
        let (u, ur, uc) = (
            g(&FIDELITY.with_unknown),
            g(&FIDELITY.unknown_refs),
            g(&FIDELITY.unknown_consumed),
        );
        if u > 0 {
            out.push_str(&format!(
                "[concord]   unnameable value (`__unknown`): {u} methods, {ur} references \
                 — {uc} of them where a value is CONSUMED (receiver/argument), which is what \
                 makes a call site meaningless\n"
            ));
        }
        let du = g(&FIDELITY.dispatch_unrecovered);
        if du > 0 {
            out.push_str(&format!(
                "[concord]   unresolved indirect dispatch (case bodies behind a forced goto): {du}\n"
            ));
        }
        let uw = g(&FIDELITY.with_unwalked_blocks);
        if uw > 0 {
            out.push_str(&format!(
                "[concord]   CFG regions missing entirely (blocks never walked): {uw}\n"
            ));
            if let Ok(v) = BLOCK_LOSS.lock() {
                let list: Vec<String> = v
                    .iter()
                    .map(|(r, un, tot)| format!("{r:#x}:{un}/{tot}"))
                    .collect();
                out.push_str(&format!(
                    "[concord]     rvas (first {} of {}): {}\n",
                    list.len(),
                    uw,
                    list.join(" ")
                ));
            }
        }
        let (gg, gr) = (g(&FIDELITY.with_goto), g(&FIDELITY.goto_refs));
        if gg > 0 {
            out.push_str(&format!(
                "[concord]   irreducible control flow (arbitrary `goto`): {gg} methods, {gr} gotos\n"
            ));
        }
        let rr = g(&FIDELITY.with_raw_register);
        if rr > 0 {
            out.push_str(&format!(
                "[concord]   unmapped register printed verbatim (`x19`/`g14`/…): {rr}\n"
            ));
        }
    }

    // The verdicts above say HOW MANY methods are goto-shaped; this says WHY.
    // Without it "64% contain a goto" is not actionable.
    out.push_str(&super::csharp::struct_census_report());
    out.push_str(&super::csharp::fallback_census_report());
    out.push_str(&super::idioms::report());
    // Which high-level construct recognizers actually fired, so one that is
    // silently inert is visible instead of assumed to work.
    out.push_str(&super::reconstruct::report());
    // High-variable recovery: how many copy chains / phi confluences / duplicate
    // computations the Ghidra-style merge proved, and how many read-only locals
    // are STILL undefined after it (the residue it could not fix).
    out.push_str(&super::highvar::report());
    let nc_caps = super::csharp::nest_cap_count();
    if nc_caps > 0 {
        out.push_str(&format!(
            "[concord]   nesting cap (MAX_NEST_DEPTH) truncated {nc_caps} regions; \
             each is marked inline in the output\n"
        ));
    }
    let bu = g(&STATS.blocks_unemitted);
    let bt = g(&STATS.blocks_total);
    if bt > 0 {
        out.push_str(&format!(
            "[concord]   structurer coverage: {} of {bt} CFG blocks are in NO emitted \
             region ({:.1}% placed) in {} methods — the remainder are blocks with no \
             recovered edge in AND no edge back out to this method's body, so they \
             cannot be shown to be this method's code (a cold region that DOES branch \
             back is adopted and emitted)\n",
            bu,
            pct(bt - bu, bt),
            g(&STATS.methods_partial)
        ));
    }
    let (rep_blocks, rep_methods) = super::csharp::coverage_repair_totals();
    if rep_blocks > 0 {
        out.push_str(&format!(
            "[concord]   coverage repair: {rep_blocks} reachable region(s) the walker could \
             not place in a structured statement were emitted as labelled regions in \
             {rep_methods} methods\n"
        ));
    }

    if let Ok(w) = WORST.lock() {
        if !w.is_empty() {
            out.push_str("[concord] worst methods (unaccounted native facts):\n");
            for c in w.iter().take(15) {
                out.push_str(&format!(
                    "[concord]   {:#x}: calls {}/{} conds {}/{} offsets {}/{} imms {}/{}\n",
                    c.rva,
                    c.cs_calls,
                    c.native_calls,
                    c.cs_conds,
                    c.native_conds,
                    c.matched_offsets,
                    c.native_offsets,
                    c.matched_imms,
                    c.native_imms
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facts_are_extracted_from_emitted_csharp() {
        let text = "static void M()\n{\n    // a comment with 99 in it\n    \
                    if (*(nint*)(x23 + 76) != 0) { fn_1234(v9, \"not 7\"); }\n    \
                    while (v1 < 0x10) { g_5(1 + 4); }\n}\n";
        let f = extract_facts(text);
        // fn_1234, g_5 → 2 real call expressions, plus the method's own
        // signature `M(` — which `signature_call_bias` removes at the
        // comparison site, since the scan cannot know it is a declaration.
        assert_eq!(f.calls, 3, "call extraction: {f:?}");
        assert_eq!(signature_call_bias(text), 1, "signature not detected");
        assert_eq!(f.conds, 2, "cond extraction: {f:?}");
        // 76 (after `+`), 4 (after `+`). `1` is NOT additive here (`1 + 4`
        // makes 4 additive, not 1).
        assert!(f.additive_ints.contains(&76), "{f:?}");
        assert!(f.additive_ints.contains(&4), "{f:?}");
        // the 99 in the comment and the 7 in the string must not leak
        assert!(!f.additive_ints.contains(&99), "{f:?}");
        assert!(!f.int_literals.contains(&99), "comment leaked: {f:?}");
        assert!(!f.int_literals.contains(&7), "string leaked: {f:?}");
        // `if (` and `while (` are keywords, so they are not call sites
        assert_eq!(f.int_literals, vec![76, 0, 16, 1, 4], "{f:?}");
    }

    #[test]
    fn hex_and_suffixed_literals_parse() {
        assert_eq!(parse_int_literal("0x4c"), Some(0x4c));
        assert_eq!(parse_int_literal("304"), Some(304));
        assert_eq!(parse_int_literal("1234567890123L"), Some(1234567890123));
        assert_eq!(parse_int_literal("abc"), None);
    }

    #[test]
    fn frame_relative_accesses_are_not_counted_as_offsets() {
        // A stack save (`stp x28, x27, [sp, #0x10]`) is frame traffic and must
        // stay out of the denominator.
        let bytes = {
            let words: [u32; 1] = [0xA9017BFD]; // stp x29, x30, [sp, #0x10]
            words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let decoded = verify::decode_all_arm64(&bytes, 0x1000).0;
        let (calls, conds, offsets, imms) = native_counts(&decoded);
        assert_eq!((calls, conds), (0, 0));
        assert_eq!(offsets, 0, "SP-relative store entered the denominator");
        assert_eq!(imms, 0);
    }
}
