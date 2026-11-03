//! Structural conformance: does the IR contain the OPERATIONS the native
//! bytes demand?
//!
//! `verify.rs` answers two questions per instruction:
//!
//!   * *coverage* — is there any IR at this native address? (`uncovered`)
//!   * *site agreement* — where there IS IR, does it say the same thing the
//!     instruction does? (`semantic_diffs`)
//!
//! Neither catches the class that actually makes a decompiled method
//! *wrong-logic-but-plausible*: a native construct whose semantic survived in
//! a DIFFERENT shape, so both questions are answered "yes" while the method
//! does something else. Three real examples, all found by hand:
//!
//!   * `cmp r0,#0x3e; popeq {…,pc}` lifted as an UNCONDITIONAL stack-adjust +
//!     return. The address is covered (a Branch is there) and there is nothing
//!     at the site to disagree with — but the conditional return is gone, and
//!     with it everything reachable only through the other edge. The method
//!     either loses its loop body or never exits.
//!   * one native `bl` rendered as two call statements. Every site checks out;
//!     the method calls twice.
//!   * a native `ret` whose IR has no `Return` at all — the renderer emits
//!     `goto L27` and the function falls into an infinite loop.
//!
//! So this module checks the *population*, not the sites: for each method it
//! compares the multiset of side-effecting operations the native instruction
//! stream performs against the multiset the IR performs, and reports the
//! difference. Every class below is decidable from (decoded native stream,
//! final IR, CFG) — no heuristics on rendered text, no second decompiler.
//!
//! Enabled by `ILR_CONFORM=1`; `ILR_CONFORM_JSON=1` additionally appends a
//! per-method JSONL record. The ranked report is printed at dump end next to
//! the existing verify aggregate.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::cfg::Cfg;
use super::ir::{Insn, Op, Operand};
use super::verify::{DecodedInsn, InsnClass, ledger_has};

// ---------------------------------------------------------------------------
// Defect classes
// ---------------------------------------------------------------------------

/// A named, countable difference between what the native code does and what
/// the IR does. Each variant is a *class*, not an instance: the report ranks
/// classes so the largest systematic bug is the first thing on the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefectClass {
    /// One native call site rendered as MORE THAN ONE call statement.
    /// The native performs the call once; the IR performs it twice.
    CallDuplicated,
    /// A native conditional branch with no conditional branch in the IR at
    /// that address — folded into an unconditional edge without a
    /// `FoldedBranch` excuse. This is the ARM32 `popeq` class and, in
    /// general, the "condition silently dropped" class.
    CondDropped,
    /// The native stream contains `ret` but the IR contains no `Return` and
    /// does not end in a tail call: the method has lost its exit.
    ReturnLost,
    /// A narrow native access (`ldrb`/`strb`/`ldrh`/`strh`) whose IR performs
    /// a differently-WIDTH access at the same address: a one-byte field read
    /// as eight bytes, or vice versa.
    WidthMismatch,
    /// A register-indexed native access (`[base, index]`) whose IR address
    /// expression has no index and no expanded equivalent (`arr[i]`, `&x`, an
    /// explicit `Add`): the index vanished and the access reads the wrong
    /// element.
    IndexDropped,
    /// A CFG block that does not end in a terminator. The renderer either
    /// falls through into the next block or loops on itself.
    BlockUnterminated,
    /// A branch whose target instruction index is out of range.
    GotoOutOfRange,
    /// A call site present in the native stream whose address is absent from
    /// the final listing entirely (not merely changed shape).
    CallVanished,
}

impl DefectClass {
    pub const ALL: [DefectClass; 8] = [
        DefectClass::CallDuplicated,
        DefectClass::CondDropped,
        DefectClass::ReturnLost,
        DefectClass::WidthMismatch,
        DefectClass::IndexDropped,
        DefectClass::BlockUnterminated,
        DefectClass::GotoOutOfRange,
        DefectClass::CallVanished,
    ];

    fn index(self) -> usize {
        match self {
            DefectClass::CallDuplicated => 0,
            DefectClass::CondDropped => 1,
            DefectClass::ReturnLost => 2,
            DefectClass::WidthMismatch => 3,
            DefectClass::IndexDropped => 4,
            DefectClass::BlockUnterminated => 5,
            DefectClass::GotoOutOfRange => 6,
            DefectClass::CallVanished => 7,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DefectClass::CallDuplicated => "call duplicated (one bl, two calls)",
            DefectClass::CondDropped => {
                "condition dropped (conditional branch lowered unconditional)"
            }
            DefectClass::ReturnLost => "return lost (native ret, no Return in IR)",
            DefectClass::WidthMismatch => "access width mismatch (ldrb/strh rendered differently)",
            DefectClass::IndexDropped => "index dropped ([base,index] lost its index)",
            DefectClass::BlockUnterminated => "block has no terminator",
            DefectClass::GotoOutOfRange => "branch target out of range",
            DefectClass::CallVanished => "call site absent from IR",
        }
    }

    /// One line of guidance, printed with the ranked table so the report is a
    /// worklist rather than a complaint.
    pub fn hint(self) -> &'static str {
        match self {
            DefectClass::CallDuplicated => {
                "one native call emitted twice: a definition binding and a re-inline, or a block emitted from two entries"
            }
            DefectClass::CondDropped => {
                "the flags are set but the branch lost its predicate: check the lifter arm for the conditional form of this instruction"
            }
            DefectClass::ReturnLost => {
                "no exit edge reaches a Return: the structuring pass fused the exit into a back-edge"
            }
            DefectClass::WidthMismatch => {
                "the lifter produced the wrong access size for this mnemonic"
            }
            DefectClass::IndexDropped => {
                "the addressing mode was parsed but its index operand was not carried into Operand::Memory"
            }
            DefectClass::BlockUnterminated => {
                "CFG construction failed to give this block a terminator"
            }
            DefectClass::GotoOutOfRange => {
                "a branch target index was not remapped when the listing shrank; the edge and its block are lost"
            }
            DefectClass::CallVanished => {
                "an erased call with no ledger excuse: a pass deleted it without recording why"
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Global tallies
// ---------------------------------------------------------------------------

const ZERO: AtomicU64 = AtomicU64::new(0);

struct ConformStats {
    instances: [AtomicU64; DefectClass::ALL.len()],
    /// Methods in which the class occurs at least once. A class that hits
    /// 4,000 times in 12 methods has one root cause; one that hits 4,000
    /// times across 3,000 methods has another.
    methods: [AtomicU64; DefectClass::ALL.len()],
    methods_checked: AtomicU64,
    methods_conformant: AtomicU64,
}

static STATS: ConformStats = ConformStats {
    instances: [ZERO; DefectClass::ALL.len()],
    methods: [ZERO; DefectClass::ALL.len()],
    methods_checked: AtomicU64::new(0),
    methods_conformant: AtomicU64::new(0),
};

/// Sample sites per class (capped) so the report names addresses to inspect.
static SAMPLES: OnceLock<Mutex<HashMap<&'static str, Vec<(u64, String)>>>> = OnceLock::new();

fn samples() -> &'static Mutex<HashMap<&'static str, Vec<(u64, String)>>> {
    SAMPLES.get_or_init(|| Mutex::new(HashMap::new()))
}

const SAMPLE_CAP: usize = 8;

fn record(class: DefectClass, method_rva: u64, addr: u64, detail: String) {
    if let Ok(mut m) = samples().lock() {
        let v = m.entry(class.name()).or_default();
        if v.len() < SAMPLE_CAP {
            v.push((method_rva, format!("{addr:#x}: {detail}")));
        }
    }
}

/// Is structural conformance checking enabled?
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ILR_CONFORM").is_ok())
}

fn json_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ILR_CONFORM_JSON").is_ok())
}

// ---------------------------------------------------------------------------
// Native-side facts we need beyond verify.rs's NativeFacts
// ---------------------------------------------------------------------------

/// Access width in bytes implied by a narrow-load/store mnemonic, or `None`
/// when the width is not implied (plain `ldr`/`str` are register-width).
fn native_width(mnemonic: &str) -> Option<u8> {
    match mnemonic {
        "ldrb" | "strb" | "ldurb" | "sturb" | "ldrsb" | "ldursb" => Some(1),
        "ldrh" | "strh" | "ldurh" | "sturh" | "ldrsh" | "ldursh" => Some(2),
        "ldrsw" | "ldursw" | "strw" => Some(4),
        _ => None,
    }
}

/// A native CALL site. `bx`/`br` are excluded on purpose: they are register
/// jumps, and on ARM32 the overwhelmingly common `bx lr` is a RETURN. Counting
/// them as calls made the "call site absent from IR" class mostly consist of
/// perfectly good returns.

/// A native CALL site. `bx`/`br` are excluded on purpose: they are register
/// jumps, and on ARM32 the overwhelmingly common `bx lr` is a RETURN. Counting
/// them as calls made the "call site absent from IR" class consist mostly of
/// perfectly good returns.
fn is_call_site(d: &DecodedInsn) -> bool {
    matches!(d.mnemonic.as_str(), "bl" | "blr" | "blx")
        || (d.class == InsnClass::Call && d.mnemonic != "bx" && d.mnemonic != "br")
}

fn ir_is_call(op: &Op) -> bool {
    matches!(
        op,
        Op::Call | Op::CallVoid | Op::IndirectCall | Op::NewObj(_) | Op::Throw
    )
}

fn ir_is_terminator(op: &Op) -> bool {
    matches!(
        op,
        Op::Jump(_) | Op::CondJump(_) | Op::Switch(_) | Op::Return
    )
}

/// Does the IR at this address perform an access of `width` bytes?
fn ir_has_width(insns_at: &[&Insn], width: u8) -> bool {
    insns_at.iter().any(|i| {
        i.operands
            .iter()
            .any(|o| matches!(o, Operand::Memory { width: w, .. } if *w == width))
    })
}

/// Does the IR at this address contain any memory operand at all?
fn ir_has_memory(insns_at: &[&Insn]) -> bool {
    insns_at.iter().any(|i| {
        i.operands
            .iter()
            .any(|o| matches!(o, Operand::Memory { .. }))
    })
}

/// The "the index did not really vanish" escape hatches: forms the lifter
/// legitimately expands an indexed access into.
fn ir_expands_index(insns_at: &[&Insn]) -> bool {
    insns_at.iter().any(|i| {
        i.operands
            .iter()
            .any(|o| matches!(o, Operand::AddressOf(_) | Operand::ArrayAccess { .. }))
            || matches!(i.op, Op::Arith(super::ir::ArithKind::Add))
    })
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

/// Structural conformance for one method. Returns the findings; also folds
/// them into the global tallies (so the dump-end report covers the whole run).
pub fn check(
    decoded: &[DecodedInsn],
    final_insns: &[Insn],
    cfg: Option<&Cfg>,
    method_rva: u64,
) -> Vec<(DefectClass, u64, String)> {
    if !enabled() {
        return Vec::new();
    }
    let findings = analyze(decoded, final_insns, cfg);
    tally(method_rva, &findings);
    findings
}

/// Pure core: no global state, so unit tests can drive it directly and
/// deterministically (the tallying above is what the report reads).
pub fn analyze(
    decoded: &[DecodedInsn],
    final_insns: &[Insn],
    cfg: Option<&Cfg>,
) -> Vec<(DefectClass, u64, String)> {
    let mut by_addr: HashMap<u64, Vec<&Insn>> = HashMap::new();
    for i in final_insns {
        by_addr.entry(i.address).or_default().push(i);
    }
    let empty: Vec<&Insn> = Vec::new();

    let mut findings: Vec<(DefectClass, u64, String)> = Vec::new();
    let mut push = |class: DefectClass, addr: u64, detail: String| {
        findings.push((class, addr, detail));
    };

    // --- native-driven classes -------------------------------------------
    for d in decoded {
        let insns_at = by_addr.get(&d.addr).unwrap_or(&empty);

        // (1) call multiplicity: one native call, more than one IR call.
        if is_call_site(d) {
            let n_ir = insns_at.iter().filter(|i| ir_is_call(&i.op)).count();
            if n_ir > 1 {
                push(
                    DefectClass::CallDuplicated,
                    d.addr,
                    format!("native {} ×1, IR call statements ×{n_ir}", d.mnemonic),
                );
            }
            // A call site whose address is absent from the listing altogether,
            // with no ledger excuse: a pass deleted it silently.
            if insns_at.is_empty() && d.mnemonic != "br" && !ledger_has(d.addr) {
                push(
                    DefectClass::CallVanished,
                    d.addr,
                    format!("native {} — no IR at this address", d.mnemonic),
                );
            }
        }

        // (2) condition dropped: native conditional branch, no CondJump in IR.
        if d.facts.cond_branch.is_some() && !ledger_has(d.addr) {
            let has_cond = insns_at.iter().any(|i| matches!(i.op, Op::CondJump(_)));
            if !has_cond {
                let what = if insns_at.is_empty() {
                    "no IR at this address".to_string()
                } else {
                    let shapes: Vec<String> = insns_at
                        .iter()
                        .map(|i| format!("{:?}", std::mem::discriminant(&i.op)))
                        .collect();
                    format!("IR has {} insn(s) but no CondJump", shapes.len())
                };
                push(
                    DefectClass::CondDropped,
                    d.addr,
                    format!("native conditional branch; {what}"),
                );
            }
        }

        // (3) access width: narrow native access, IR performs another width.
        if let Some(w) = native_width(&d.mnemonic) {
            if ir_has_memory(insns_at) && !ir_has_width(insns_at, w) {
                let got: Vec<String> = insns_at
                    .iter()
                    .flat_map(|i| i.operands.iter())
                    .filter_map(|o| match o {
                        Operand::Memory { width, .. } => Some(format!("{width}")),
                        _ => None,
                    })
                    .collect();
                push(
                    DefectClass::WidthMismatch,
                    d.addr,
                    format!(
                        "native {} expects width {w}, IR memory width {}",
                        d.mnemonic,
                        got.join("/")
                    ),
                );
            }
        }

        // (4) index dropped: native indexed access, IR address has no index.
        if d.facts.mem_indexed.is_some() {
            let lost = insns_at.iter().any(|i| {
                i.operands.iter().any(|o| match o {
                    Operand::Memory { index, .. } => index.is_none(),
                    _ => false,
                })
            });
            if lost && !ir_expands_index(insns_at) {
                push(
                    DefectClass::IndexDropped,
                    d.addr,
                    format!("native {} uses [base,index]; IR has no index", d.mnemonic),
                );
            }
        }
    }

    // (5) return lost: native `ret` exists, IR never returns and has no tail
    // call standing in for one.
    let native_ret = decoded.iter().find(|d| d.mnemonic == "ret").map(|d| d.addr);
    if let Some(addr) = native_ret {
        let has_return = final_insns.iter().any(|i| matches!(i.op, Op::Return));
        // A tail call stands in for a return only when it is an actual CALL.
        // Ending in a `Jump` is exactly the failure mode this class exists to
        // catch (the exit edge was fused into a back-edge), so it must not
        // excuse anything.
        let tail_call = final_insns.last().is_some_and(|i| ir_is_call(&i.op));
        if !has_return && !tail_call {
            push(
                DefectClass::ReturnLost,
                addr,
                "native ret, IR has no Return and no trailing call".to_string(),
            );
        }
    }

    // (6) CFG integrity: a block whose CONTROL FLOW is not expressible
    // without a terminator. Three endings are legitimate and must not be
    // flagged, or the class becomes noise:
    //
    //   * the block already ends in a terminator;
    //   * it has NO successor — the trailing op is a call that does not
    //     return (a tail call, or a throw helper);
    //   * its only successor is the very next instruction — implicit
    //     fallthrough, which the renderer already emits correctly by listing
    //     the blocks in order.
    //
    // What is left is the real defect: a block whose recorded successors no
    // longer match the ops in it, i.e. a pass deleted or rewrote a terminator
    // and left the edges behind. That is the shape that renders as a `goto`
    // into a loop with no exit.
    if let Some(cfg) = cfg {
        for (bi, b) in cfg.blocks.iter().enumerate() {
            if b.end == 0 || b.end > final_insns.len() {
                continue;
            }
            // The IR emits PHI EDGE COPIES after the terminator, at the
            // terminator's own address, so the block's LAST index is often a
            // plain `Move` that is not the control transfer. Judge the block
            // by its last op that is not one of those copies.
            let mut term_idx = None;
            for idx in (b.start..b.end).rev() {
                if !matches!(final_insns[idx].op, Op::Move) {
                    term_idx = Some(idx);
                    break;
                }
            }
            if term_idx.is_some_and(|i| ir_is_terminator(&final_insns[i].op)) {
                continue;
            }
            let last = &final_insns[term_idx.unwrap_or(b.end - 1)];
            let succ = cfg.successors.get(bi).cloned().unwrap_or_default();
            if succ.is_empty() {
                continue;
            }
            // `successors` holds BLOCK indices (`rebuild_cfg` maps a
            // non-terminator block's edge through `block_of[b.end]`), so the
            // fallthrough test is "the very next block" — comparing against
            // an INSTRUCTION index (`b.end`) flagged nearly every block in the
            // binary and buried the signal under 8,000 false positives.
            if succ.len() == 1 && succ[0] == bi + 1 {
                continue;
            }
            // Report the successors by ADDRESS, not index: the report is read
            // next to the disassembly, and a block index means nothing there.
            let succ_addrs: Vec<String> = succ
                .iter()
                .map(|&s| match cfg.blocks.get(s).map(|blk| blk.start) {
                    Some(start) => match final_insns.get(start) {
                        Some(i) => format!("{:#x}", i.address),
                        None => format!("block #{s}"),
                    },
                    None => format!("block #{s} (past end)"),
                })
                .collect();
            let next_addr = final_insns
                .get(b.end)
                .map(|i| format!("{:#x}", i.address))
                .unwrap_or_else(|| "end".to_string());
            push(
                DefectClass::BlockUnterminated,
                last.address,
                format!(
                    "last non-copy insn {} ({:?}) at {:#x}, successors [{}], next insn is {} — control leaves another way",
                    term_idx.unwrap_or(b.end - 1),
                    last.op,
                    last.address,
                    succ_addrs.join(", "),
                    next_addr
                ),
            );
        }
    }
    // `Cfg::build` records a branch that LEAVES the method as
    // `Jump(insns.len())` — an EXIT SENTINEL one past the end, and a
    // documented convention (see the tail-merge test). Only a target
    // STRICTLY past that is dangling: nothing remapped it, so the renderer
    // emits a `goto` to a label that does not exist and the edge is lost.
    let n = final_insns.len();
    for i in final_insns {
        let (bad, what) = match &i.op {
            Op::Jump(t) | Op::CondJump(t) => {
                (*t > n, format!("target index {t} > listing length {n}"))
            }
            Op::Switch(ts) => {
                let oob: Vec<String> = ts
                    .iter()
                    .filter(|t| **t > n)
                    .map(|t| t.to_string())
                    .collect();
                (
                    !oob.is_empty(),
                    format!(
                        "switch targets [{}] exceed listing length {n}",
                        oob.join(",")
                    ),
                )
            }
            _ => (false, String::new()),
        };
        if bad {
            push(DefectClass::GotoOutOfRange, i.address, what);
        }
    }

    findings
}

/// Fold one method's findings into the global tallies + report samples.
fn tally(method_rva: u64, findings: &[(DefectClass, u64, String)]) {
    STATS.methods_checked.fetch_add(1, Ordering::Relaxed);
    if findings.is_empty() {
        STATS.methods_conformant.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let mut hit: [bool; DefectClass::ALL.len()] = [false; DefectClass::ALL.len()];
    for (class, addr, detail) in findings {
        STATS.instances[class.index()].fetch_add(1, Ordering::Relaxed);
        if !hit[class.index()] {
            hit[class.index()] = true;
            STATS.methods[class.index()].fetch_add(1, Ordering::Relaxed);
        }
        record(*class, method_rva, *addr, detail.clone());
    }
    if json_enabled() {
        append_jsonl(method_rva, findings);
    }
}

fn append_jsonl(method_rva: u64, findings: &[(DefectClass, u64, String)]) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("conform_findings.jsonl")
    else {
        return;
    };
    let items: Vec<String> = findings
        .iter()
        .map(|(c, a, d)| {
            let d = d.replace('\\', "\\\\").replace('"', "\\\"");
            format!(
                "{{\"class\":\"{}\",\"addr\":{:#x},\"detail\":\"{}\"}}",
                c.name(),
                a,
                d
            )
        })
        .collect();
    let _ = writeln!(
        f,
        "{{\"rva\":{:#x},\"findings\":[{}]}}",
        method_rva,
        items.join(",")
    );
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Ranked class table. Printed at dump end when `ILR_VERIFY=1` (the aggregate
/// report is the natural home: it already prints ledger and cluster tables).
pub fn report() -> String {
    if !enabled() {
        return String::new();
    }
    let checked = STATS.methods_checked.load(Ordering::Relaxed);
    if checked == 0 {
        return String::new();
    }
    let conformant = STATS.methods_conformant.load(Ordering::Relaxed);
    let mut rows: Vec<(DefectClass, u64, u64)> = DefectClass::ALL
        .iter()
        .map(|&c| {
            (
                c,
                STATS.instances[c.index()].load(Ordering::Relaxed),
                STATS.methods[c.index()].load(Ordering::Relaxed),
            )
        })
        .filter(|(_, n, _)| *n > 0)
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));

    let mut s = format!(
        "\n[conform] {checked} methods checked, {conformant} structurally conformant ({}%), {} CLASSES WITH DEFECTS",
        if checked == 0 {
            0
        } else {
            conformant * 100 / checked
        },
        rows.len()
    );
    if rows.is_empty() {
        s.push_str("\n[conform] no structural conformance defects found");
        return s;
    }
    let total: u64 = rows.iter().map(|(_, n, _)| *n).sum();
    s.push_str(&format!("\n[conform] {total} findings, ranked by class:"));
    let samples = samples().lock().ok();
    for (class, n, methods) in &rows {
        s.push_str(&format!(
            "\n[conform]   {n:>8}  in {methods:>6} methods  — {}\n[conform] {:>11}{}",
            class.name(),
            "",
            class.hint()
        ));
        if let Some(m) = samples.as_ref() {
            if let Some(v) = m.get(class.name()) {
                for (rva, site) in v.iter().take(3) {
                    s.push_str(&format!("\n[conform] {:>11}e.g. {rva:#x} @ {site}", ""));
                }
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::super::cfg::{Block, Cfg};
    use super::super::ir::{ArithKind, Insn, Op, Operand};
    use super::super::verify::{DecodedInsn, InsnClass, NativeFacts, ledger_clear};
    use super::*;

    fn decoded(addr: u64, mnemonic: &str, class: InsnClass, facts: NativeFacts) -> DecodedInsn {
        DecodedInsn {
            addr,
            // These helpers build ARM64-shaped sequences (4-byte words), so the
            // encoded length is 4. It matters: the semantic diff uses it to
            // derive a conditional branch's not-taken successor.
            size: 4,
            mnemonic: mnemonic.to_string(),
            class,
            facts,
        }
    }

    fn mem(width: u8, index: Option<Operand>) -> Operand {
        Operand::Memory {
            base: Box::new(Operand::Local(1)),
            index: index.map(Box::new),
            scale: 0,
            addend: 0,
            width,
        }
    }

    fn classes(f: &[(DefectClass, u64, String)]) -> Vec<DefectClass> {
        f.iter().map(|(c, _, _)| *c).collect()
    }

    /// A conditional branch whose IR carries an unconditional Jump at that
    /// address is the `cmp; popeq {…,pc}` class: the address is covered, there
    /// is nothing at the site to disagree with, and the other edge — with
    /// everything reachable only through it — is gone.
    #[test]
    fn conditional_branch_without_cond_jump_is_flagged() {
        ledger_clear();
        let dec = vec![decoded(
            0x1000,
            "b.eq",
            InsnClass::Branch,
            NativeFacts {
                cond_branch: Some((0, 0x2000)),
                ..Default::default()
            },
        )];
        let ir = vec![
            Insn::new(Op::Jump(1), vec![], 0x1000),
            Insn::new(Op::Return, vec![], 0x1004),
        ];
        let f = analyze(&dec, &ir, None);
        assert!(
            classes(&f).contains(&DefectClass::CondDropped),
            "expected CondDropped, got {:?}",
            f
        );
    }

    /// One native `bl` rendered as two call statements. Both sites check out
    /// against the bytes individually, which is why nothing else sees it.
    #[test]
    fn one_native_call_emitted_twice_is_flagged() {
        ledger_clear();
        let dec = vec![decoded(
            0x1000,
            "bl",
            InsnClass::Call,
            NativeFacts {
                call: Some((0x3000, false)),
                ..Default::default()
            },
        )];
        let ir = vec![
            Insn::new(Op::Call, vec![Operand::Immediate(0x3000)], 0x1000),
            Insn::new(Op::Call, vec![Operand::Immediate(0x3000)], 0x1000),
        ];
        let f = analyze(&dec, &ir, None);
        assert!(
            classes(&f).contains(&DefectClass::CallDuplicated),
            "expected CallDuplicated, got {:?}",
            f
        );
    }

    /// The native returns; the IR never does and ends in a jump — the
    /// `goto L27` fall-into-an-infinite-loop shape.
    #[test]
    fn native_return_with_no_ir_return_is_flagged() {
        ledger_clear();
        let dec = vec![
            decoded(0x1000, "mov", InsnClass::Arith, NativeFacts::default()),
            decoded(0x1004, "ret", InsnClass::Branch, NativeFacts::default()),
        ];
        let ir = vec![
            Insn::new(Op::Move, vec![Operand::Local(1)], 0x1000),
            Insn::new(Op::Jump(0), vec![], 0x1004),
        ];
        let f = analyze(&dec, &ir, None);
        assert!(
            classes(&f).contains(&DefectClass::ReturnLost),
            "expected ReturnLost, got {:?}",
            f
        );
    }

    /// `ldrb` is a one-byte access; an eight-byte read of the same field is a
    /// different program.
    #[test]
    fn byte_access_rendered_as_eight_bytes_is_flagged() {
        ledger_clear();
        let dec = vec![decoded(
            0x1000,
            "ldrb",
            InsnClass::LoadStore,
            NativeFacts {
                mem: Some((1, 8)),
                ..Default::default()
            },
        )];
        let ir = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(2), mem(8, None)],
            0x1000,
        )];
        let f = analyze(&dec, &ir, None);
        assert!(
            classes(&f).contains(&DefectClass::WidthMismatch),
            "expected WidthMismatch, got {:?}",
            f
        );
    }

    /// A register-indexed native access whose IR carries the base but not the
    /// index: the access reads the wrong element silently.
    #[test]
    fn indexed_access_without_index_is_flagged() {
        ledger_clear();
        let dec = vec![decoded(
            0x1000,
            "ldr",
            InsnClass::LoadStore,
            NativeFacts {
                mem_indexed: Some((1, 2)),
                ..Default::default()
            },
        )];
        let ir = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(2), mem(0, None)],
            0x1000,
        )];
        let f = analyze(&dec, &ir, None);
        assert!(
            classes(&f).contains(&DefectClass::IndexDropped),
            "expected IndexDropped, got {:?}",
            f
        );
    }

    /// The IR emits PHI edge copies AFTER the terminator, at the terminator's
    /// address, so a block's last index is routinely a `Move` while the block
    /// is perfectly terminated. Flagging that was the checker's own worst bug:
    /// it reported 6,000 `unterminated` blocks that were nothing of the kind.
    #[test]
    fn edge_copy_after_terminator_is_not_an_unterminated_block() {
        ledger_clear();
        let ir = vec![
            Insn::new(
                Op::Compare(super::super::ir::CompareKind::Equal),
                vec![Operand::Local(1)],
                0x1000,
            ),
            Insn::new(Op::CondJump(2), vec![], 0x1000),
            // φ edge copies, same address as the branch
            Insn::new(Op::Move, vec![Operand::Local(1), Operand::Local(2)], 0x1000),
        ];
        // Successors deliberately NOT the next block, so only the terminator
        // scan can excuse this — which is the behaviour under test.
        let cfg = Cfg {
            blocks: vec![
                Block { start: 0, end: 3 },
                Block { start: 3, end: 3 },
                Block { start: 3, end: 3 },
            ],
            successors: vec![vec![2], Vec::new(), Vec::new()],
        };
        let f = analyze(&[], &ir, Some(&cfg));
        assert!(
            !classes(&f).contains(&DefectClass::BlockUnterminated),
            "edge copies after a terminator must not be reported: {:?}",
            f
        );
    }

    /// Implicit fallthrough — the one case `rebuild_cfg` produces for a block
    /// with no terminator — must never be reported, no matter the indices.
    #[test]
    fn fallthrough_block_without_terminator_is_not_reported() {
        ledger_clear();
        let ir = vec![
            Insn::new(Op::Move, vec![Operand::Local(1)], 0x1000),
            Insn::new(Op::Move, vec![Operand::Local(2)], 0x1004),
            Insn::new(Op::Return, vec![], 0x1008),
        ];
        let cfg = Cfg {
            blocks: vec![Block { start: 0, end: 2 }, Block { start: 2, end: 3 }],
            successors: vec![vec![1], Vec::new()],
        };
        let f = analyze(&[], &ir, Some(&cfg));
        assert!(
            !classes(&f).contains(&DefectClass::BlockUnterminated),
            "fallthrough must be excused: {:?}",
            f
        );
    }

    /// `Jump(len)` is the documented EXIT SENTINEL (a branch leaving the
    /// method), not a dangling target. Flagging it would have reported
    /// thousands of perfectly good exits as corruption.
    #[test]
    fn exit_sentinel_is_not_a_dangling_target() {
        ledger_clear();
        let ir = vec![
            Insn::new(Op::Move, vec![Operand::Local(1)], 0x1000),
            Insn::new(Op::Jump(2), vec![], 0x1004), // 2 == len → exit sentinel
        ];
        let f = analyze(&[], &ir, None);
        assert!(
            !classes(&f).contains(&DefectClass::GotoOutOfRange),
            "the exit sentinel must not be flagged: {:?}",
            f
        );
    }

    /// A target strictly past the sentinel is dangling.
    #[test]
    fn target_past_the_sentinel_is_flagged() {
        ledger_clear();
        let ir = vec![
            Insn::new(Op::Move, vec![Operand::Local(1)], 0x1000),
            Insn::new(Op::Jump(9), vec![], 0x1004),
        ];
        let f = analyze(&[], &ir, None);
        assert!(
            classes(&f).contains(&DefectClass::GotoOutOfRange),
            "expected GotoOutOfRange, got {:?}",
            f
        );
    }

    /// The mirror: a block with NO terminator at all whose successor is not
    /// the next block — the shape that renders as a `goto` into a loop
    /// with no exit — must still be reported.
    #[test]
    fn block_without_terminator_and_other_successor_is_flagged() {
        ledger_clear();
        let ir = vec![
            Insn::new(Op::Move, vec![Operand::Local(1)], 0x1000),
            Insn::new(Op::Move, vec![Operand::Local(2)], 0x1004),
            Insn::new(Op::Move, vec![Operand::Local(3)], 0x1008),
            Insn::new(Op::Return, vec![], 0x100C),
        ];
        // The block ends at index 2 but control goes to index 3: not
        // fallthrough, and there is no terminator in the block to express it.
        let cfg = Cfg {
            blocks: vec![Block { start: 0, end: 2 }],
            successors: vec![vec![3]],
        };
        let f = analyze(&[], &ir, Some(&cfg));
        assert!(
            classes(&f).contains(&DefectClass::BlockUnterminated),
            "expected BlockUnterminated, got {:?}",
            f
        );
    }

    /// The same native access carried correctly must NOT be flagged — this is
    /// the falsification half of each class above: a checker that fires on
    /// conformant IR is worse than no checker.
    #[test]
    fn conformant_method_produces_no_findings() {
        ledger_clear();
        let dec = vec![
            decoded(
                0x1000,
                "ldrb",
                InsnClass::LoadStore,
                NativeFacts {
                    mem: Some((1, 8)),
                    ..Default::default()
                },
            ),
            decoded(
                0x1004,
                "b.eq",
                InsnClass::Branch,
                NativeFacts {
                    cond_branch: Some((0, 0x1010)),
                    ..Default::default()
                },
            ),
            decoded(0x100C, "ret", InsnClass::Branch, NativeFacts::default()),
        ];
        let ir = vec![
            Insn::new(Op::Move, vec![Operand::Local(2), mem(1, None)], 0x1000),
            Insn::new(
                Op::Compare(super::super::ir::CompareKind::Equal),
                vec![Operand::Local(3)],
                0x1004,
            ),
            Insn::new(Op::CondJump(3), vec![], 0x1004),
            Insn::new(Op::Return, vec![], 0x100C),
        ];
        let f = analyze(&dec, &ir, None);
        assert!(f.is_empty(), "conformant method flagged: {:?}", f);
        // Keep the ArithKind import honest: an unrelated Add at a site with no
        // indexed native access must not create an IndexDropped excuse path.
        let _ = ArithKind::Add;
    }
}

/// Reset between corpus runs inside one process (tests).
pub fn reset() {
    for c in DefectClass::ALL {
        STATS.instances[c.index()].store(0, Ordering::Relaxed);
        STATS.methods[c.index()].store(0, Ordering::Relaxed);
    }
    STATS.methods_checked.store(0, Ordering::Relaxed);
    STATS.methods_conformant.store(0, Ordering::Relaxed);
    if let Ok(mut m) = samples().lock() {
        m.clear();
    }
}
