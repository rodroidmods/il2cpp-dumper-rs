//! Identity for the IL2CPP runtime helpers that managed methods call.
//!
//! A managed method regularly calls into the IL2CPP **runtime** rather than into
//! other managed code: the static-constructor guard the compiler emits before
//! the first use of a type's statics, and the generic/virtual dispatch
//! trampolines that rgctx-driven calls go through.
//!
//! Those functions live in the executable's own `.text` section — the runtime's
//! C++ compiled into the same image — so they are NOT in the metadata's method
//! table and are NOT exported. `methods_by_rva` therefore cannot name them and
//! every one renders as `fn_<address>`. On the v24.5 reference build that is
//! 53,620 call sites from only **489** addresses, and the distribution is very
//! top-heavy: **50% of all sites are 4 addresses and 90% are 15**. One function
//! alone (`0x11c7f0`) accounts for 18% of them.
//!
//! Naming them from a symbol table is impossible by construction, so they are
//! identified from the SHAPE of their call sites instead — the same evidence
//! discipline the rest of the pipeline uses. The static-init helper is the
//! tractable one, because the guard it implements leaves three independent
//! traces that must all agree:
//!
//! ```text
//!   Compare(NotEqual)[c, *guard_slot, 0]      // the guard test …
//!   CondJump(c)                               // … folded into a branch
//!   Call  Immediate(<helper rva>) …           // the helper
//!   Move  [*guard_slot, 1]                    // … and the guard being set
//! ```
//!
//! The load, the branch and the store must all name the SAME slot, and the call
//! target must be an address no map could name. Requiring all four is what keeps
//! a coincidental call-next-to-a-guard from being mislabelled.
//!
//! ARM32 does not keep the slot literal: the guard byte is read out of the
//! metadata-usage table (`*(byte*)(tbl + *(nint*)(…))`), so the compare operand
//! is the load's local and `slot_of` names nothing. There the store half is
//! checked by shape instead — it must write an INDEXED byte
//! (`indexed_byte_store_to_one`) — which is the same evidence seen from the
//! other end. A bare compare-against-0 next to a plain `x = 1` is a null-check
//! raise or a type-init helper, not a static-init guard, and must stay unnamed.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use super::ir::{CompareKind, Insn, Op, Operand};

/// The role table a call target can be recorded under. See [`RuntimeHelper`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuntimeHelper {
    /// `if (guard == 0) { ClassInit(cls); guard = 1; }` — the static-constructor
    /// guard. `ClassInit` here is the runtime entry point that runs the type's
    /// `.cctor` once; the loaded class pointer travels in the helper's
    /// arguments (a metadata-slot load), not in this name.
    ClassInit,
    /// A codegen panic helper: `if (x == 0) { Raise(x); return; }`. Proven from
    /// the TARGET's own body (it never returns and reaches a `__cxa_throw`
    /// import), so the label is correct at every call site regardless of the
    /// local shape. This role is RENAME-ONLY: unlike the arg-taking
    /// `raise_exception` map it never rewrites the call to `Op::Throw`, so no
    /// trailing code is treated as unreachable.
    RaiseException,
    /// `il2cpp_object_new` / `Object::New` — the managed object allocator.
    /// Identified by body shape: the target calls a GC allocation primitive
    /// (or another helper that does) and returns the pointer. Call sites show
    /// the pattern `v = helper(typeof(T)); v..ctor(...)`.
    ObjectNew,
    /// `il2cpp_array_new` / `Array::New` — the managed array allocator.
    /// Identified by body shape: similar to ObjectNew but the allocation is
    /// array-specific (takes element class + size). Call sites show
    /// `v = helper(class, size)` or `v = helper(domain, class, size)`.
    ArrayNew,
    /// `il2cpp_value_box` / `Object::Box` — boxes a value type into a managed
    /// object. Identified by body shape: takes (class, value_pointer) and
    /// returns a new object with the value copied in. Call sites show
    /// `v = helper(class, &value)`.
    Box,
    /// `Object::IsInst` — returns the object when it is an instance of the
    /// class, else null. Identified from the body: it calls
    /// `il2cpp_class_is_assignable_from` and then either returns the input
    /// object or null. Call sites show `v = helper(object, class)`.
    IsInst,
    /// `il2cpp_class_is_assignable_from` itself (a bare assignability test).
    /// Same two arguments as [`IsInst`](Self::IsInst) but a bool result.
    ClassIsAssignable,
    /// `il2cpp_codegen_initialize_runtime_metadata` — decodes a metadata usage
    /// token (kind/index/1 word) into the runtime object. The call-site gives
    /// it the ADDRESS of a metadata usage slot and discards the result. This is
    /// runtime bookkeeping the managed source never contained.
    MetadataInit,
    /// A helper that CONSTRUCTS a managed exception object (calls
    /// `il2cpp_object_new` and `il2cpp_runtime_object_init`, formats from
    /// literal strings, and returns the exception). Paired with a raise helper
    /// it is the `throw new X(...)` the source wrote.
    ExceptionNew,
}

impl RuntimeHelper {
    /// The name the pseudo-C# shows for this helper. Deliberately spelled as a
    /// runtime call so a reader can tell "the runtime did this" from "a managed
    /// method with an unknown name did this" (`fn_…`). Neither name carries a
    /// hex digit after `fn_`, so both drop out of the `fn_[0-9a-f]` census.
    pub fn display_name(self) -> &'static str {
        match self {
            RuntimeHelper::ClassInit => "IL2CPP_ClassInit",
            RuntimeHelper::RaiseException => "IL2CPP_RaiseException",
            RuntimeHelper::ObjectNew => "IL2CPP_ObjectNew",
            RuntimeHelper::ArrayNew => "IL2CPP_ArrayNew",
            RuntimeHelper::Box => "IL2CPP_Box",
            RuntimeHelper::IsInst => "IL2CPP_IsInst",
            RuntimeHelper::ClassIsAssignable => "IL2CPP_ClassIsAssignable",
            RuntimeHelper::MetadataInit => "IL2CPP_InitMetadata",
            RuntimeHelper::ExceptionNew => "IL2CPP_NewException",
        }
    }
}

fn table() -> &'static Mutex<HashMap<u64, RuntimeHelper>> {
    static T: OnceLock<Mutex<HashMap<u64, RuntimeHelper>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record that `target` behaves as `role`. Called once per agreeing call site;
/// repeated notes for the same target are idempotent.
pub fn note(target: u64, role: RuntimeHelper) {
    if let Ok(mut t) = table().lock() {
        t.insert(target, role);
    }
}

/// The identified role of an unresolved call target, if any.
pub fn role_of(target: u64) -> Option<RuntimeHelper> {
    table().lock().ok().and_then(|t| t.get(&target).copied())
}

/// Note a role learned from a CALL-SITE SHAPE, unless the target is already a
/// PROVEN noreturn raise wrapper. Shapes see register contents, not the
/// callee's body; on 0x1302634 (thin wrapper proven to reach the throw core)
/// the "object in x0, class in x1" shape downgraded RaiseException to IsInst,
/// the tail raise rendered as `IL2CPP_IsInst(instance, instance2)`, and the
/// ABI staging stores feeding it stayed in the text.
pub fn note_shape(t: u64, role: RuntimeHelper) {
    if role_of(t) == Some(RuntimeHelper::RaiseException) {
        return;
    }
    note(t, role);
}

/// Targets proven NOT to be function entries (a jump-table slot / mid-function
/// address). A role must never be re-attached to these by a later call-site
/// shape heuristic: the shape only says how the call was written, not that the
/// target is a callable function.
fn invalid_targets() -> &'static Mutex<HashSet<u64>> {
    static T: std::sync::OnceLock<Mutex<HashSet<u64>>> = std::sync::OnceLock::new();
    T.get_or_init(|| Mutex::new(HashSet::new()))
}

pub fn mark_non_entry(target: u64) {
    if let Ok(mut inv) = invalid_targets().lock() {
        inv.insert(target);
    }
    if let Ok(mut t) = table().lock() {
        t.remove(&target);
    }
}

pub fn is_non_entry(target: u64) -> bool {
    invalid_targets()
        .lock()
        .map(|inv| inv.contains(&target))
        .unwrap_or(false)
}

/// Exact symbol names for unresolved call targets whose identity is in the
/// binary's own symbol table rather than in the metadata.
///
/// Two shapes land here, and neither needs an inference:
///
///  * the target IS a PLT/thunk stub, so the symbol bound to the GOT slot it
///    branches through (`bind_symbols`, from `.rela.plt`) is the function
///    being called — `__cxa_begin_catch`, `__cxa_end_catch`, `memcpy`, …;
///  * the target is a local ALIAS of an exported IL2CPP API (a `b impl` thunk
///    with no export of its own), so the API it reaches is the name.
///
/// Both facts are already computed by [`classify_bodies`] while it looks for a
/// semantic role, and both were then discarded: the role table has no entry
/// for "this is libc's `__cxa_end_catch`", so the `continue` left the call
/// rendering as `fn_…` even though the symbol table names it exactly. On the
/// temple-run-2 image that is 2,107 `fn_` occurrences for the PLT family alone
/// (`0x2dff7d0`–`0x2dff840`).
fn symbol_names() -> &'static Mutex<HashMap<u64, String>> {
    static M: OnceLock<Mutex<HashMap<u64, String>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record the exact symbol `target` calls. Idempotent.
fn note_symbol_name(target: u64, name: String) {
    if let Ok(mut m) = symbol_names().lock() {
        m.insert(target, name);
    }
}

/// The exact symbol an unresolved target calls, if the symbol table names it.
pub fn symbol_of(target: u64) -> Option<String> {
    symbol_names()
        .lock()
        .ok()
        .and_then(|m| m.get(&target).cloned())
}

/// The name to render for an unresolved call target: the runtime helper's name
/// when a role is proven, else the exact symbol when the binary names the
/// target, else `fn_<address>`.
///
/// The role wins over the symbol because a role says what the call MEANS
/// (`IL2CPP_ObjectNew`) where the symbol says which entry point implements it
/// (`il2cpp_object_new`); when both are known the meaning is the better name.
pub fn display_name(target: u64) -> String {
    if let Some(r) = role_of(target) {
        return r.display_name().to_string();
    }
    if let Some(n) = symbol_of(target) {
        return n;
    }
    format!("fn_{target:x}")
}

/// How many leading arguments the helper actually takes.
///
/// The lifter dumps every ABI register at an unresolved call; a runtime
/// helper's real parameters are the first ones. Rendering the rest produced
/// `IL2CPP_ArrayNew(typeof(T), 4, v70, v72, in_x4, in_x5, …)`, where
/// `v70`/`v72` were generic-context registers copied at method entry — two
/// undefined-looking temps per call site (10.4k helper call sites). Only roles
/// with an authoritative IL2CPP signature are trimmed; anything unknown keeps
/// the full dump (a hidden real argument is worse than noise).
pub fn arg_arity(target: u64) -> Option<usize> {
    match role_of(target) {
        // il2cpp_array_new(element_class, length)
        Some(RuntimeHelper::ArrayNew) => Some(2),
        // il2cpp_value_box(class, value_pointer)
        Some(RuntimeHelper::Box) => Some(2),
        // il2cpp_object_new(class)
        Some(RuntimeHelper::ObjectNew) => Some(1),
        // il2cpp_runtime_class_init(class)
        Some(RuntimeHelper::ClassInit) => Some(1),
        // il2cpp_codegen_initialize_runtime_metadata(usage_word)
        Some(RuntimeHelper::MetadataInit) => Some(1),
        // il2cpp_object_is_inst_of(object, class)
        Some(RuntimeHelper::IsInst) => Some(2),
        // il2cpp_class_is_assignable_from(class, class)
        Some(RuntimeHelper::ClassIsAssignable) => Some(2),
        // il2cpp_raise_exception(exception)
        Some(RuntimeHelper::RaiseException) => Some(1),
        // Exception::New() — the message is built from literals inside the
        // helper; the argument dump is generic-context noise.
        Some(RuntimeHelper::ExceptionNew) => Some(0),
        _ => None,
    }
}

/// Role-specific argument arity keyed by ROLE (not by address). Exposed for
/// tests/documentation; the renderer uses [`arg_arity`].
pub fn role_arity(role: RuntimeHelper) -> usize {
    match role {
        RuntimeHelper::ArrayNew | RuntimeHelper::Box | RuntimeHelper::IsInst
        | RuntimeHelper::ClassIsAssignable => 2,
        RuntimeHelper::ObjectNew
        | RuntimeHelper::ClassInit
        | RuntimeHelper::MetadataInit
        | RuntimeHelper::RaiseException => 1,
        RuntimeHelper::ExceptionNew => 0,
    }
}

fn raise_messages() -> &'static Mutex<HashMap<u64, String>> {
    static M: OnceLock<Mutex<HashMap<u64, String>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record the exception message for a raise helper.
pub fn note_raise_message(target: u64, message: String) {
    if let Ok(mut m) = raise_messages().lock() {
        m.insert(target, message);
    }
}

/// Get the exception message for a raise helper, if known.
pub fn raise_message(target: u64) -> Option<String> {
    raise_messages().lock().ok().and_then(|m| m.get(&target).cloned())
}

/// How many helpers were identified (for the `ILR_TIMING` line).
pub fn identified_count() -> usize {
    table().lock().map(|t| t.len()).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Body / import classification
// ---------------------------------------------------------------------------
//
// The call-site heuristics in `recovery::mod` are deliberately broad (they key
// on argument statistics), and a single alias chain can put unrelated runtime
// wrappers into the object_new family. The authoritative evidence lives in the
// helper's own BODY: which exported IL2CPP API (or imported libc/libc++ symbol)
// does it call, and what does it do with the result? `classify_bodies` reads
// that evidence and overwrites the provisional role. It never happens by
// address: the export names come from the ELF symbol table and the import names
// from `.rela.plt`/GOT (`Il2Cpp::bind_symbols`).

/// Map each exported IL2CPP API entry point to the implementation it reaches.
///
/// An export stub is either the implementation itself (a real body) or a thin
/// forwarder (`b impl`) / call wrapper (`bl impl; ret`). Mapping both the stub
/// address and its target lets a helper that calls an API be recognised even
/// when the linker emitted a local wrapper for it.
fn export_impl_names(
    il2cpp: &crate::il2cpp::base::Il2Cpp,
    is_arm64: bool,
    is_x64: bool,
) -> HashMap<u64, String> {
    let mut out = HashMap::new();
    for (name, &rva) in &il2cpp.api_export_rvas {
        if rva == 0 {
            continue;
        }
        out.insert(rva, name.clone());
        let Some(bytes) = il2cpp.read_bytes_at_rva(rva, 32) else {
            continue;
        };
        let Some(insns) =
            super::lift_helper_body(il2cpp, rva, &bytes, 8, is_arm64, is_x64)
        else {
            continue;
        };
        for i in insns.iter().take(4) {
            match &i.op {
                Op::Call | Op::CallVoid => {
                    if let Some(Operand::Immediate(t)) = i.operands.first() {
                        out.insert(*t as u64, name.clone());
                    }
                    break;
                }
                Op::Jump(t) if (*t as u64) != rva => {
                    out.insert(*t as u64, name.clone());
                    break;
                }
                _ => {}
            }
        }
    }
    out
}

/// Every RVA that names `export_name`, reduced to thunk roots.
///
/// An exported IL2CPP API is often a bare WRAPPER over its own implementation:
///
/// ```text
/// il2cpp_object_new:  str x30,[sp,#-0x10]!
///                     bl  0x1330ff8        ; the real allocator
///                     ldr x30,[sp],#0x10
///                     ret
/// 0x1330ff8:          b   0x1331300        ; codegen variant of the same body
/// 0x1302628:          b   0x1330ff8        ; the shared `new T()` thunk
/// ```
///
/// Comparing a candidate's thunk root against the EXPORT's own RVA therefore
/// matches only variants that branch straight INTO the export — on this binary
/// that is none of them, because the whole family roots at the implementation.
/// The consequence was that the shared allocator thunk was never a member of
/// `object_new`, so `key_function_recovery` never emitted `NewObj`, and the
/// renderer fell back to naming each object from its `.ctor`. Identical-code-
/// folding merges every trivial constructor onto ONE `ret` here, so that
/// fallback named an arbitrary class: `new Token()` printed as
/// `new Object()`, and because the local was then typed `Object` no field
/// could be named on it either — the whole `token.Reward = …` block degraded
/// to raw `*(nint*)(v + 16)` stores.
///
/// Both roots are returned: the export's own (covers a wrapper that branches)
/// and its implementation's (the normal wrapper-that-calls case).
pub fn export_roots(
    il2cpp: &crate::il2cpp::base::Il2Cpp,
    managed: &HashMap<u64, super::types::MethodRef>,
    export_name: &str,
    is_arm64: bool,
    is_x64: bool,
) -> std::collections::HashSet<u64> {
    export_impl_names(il2cpp, is_arm64, is_x64)
        .into_iter()
        .filter(|(_, n)| n == export_name)
        .map(|(rva, _)| super::chase_thunk_root(il2cpp, managed, rva, is_arm64, is_x64))
        .collect()
}

/// Decode a PLT entry to the GOT slot it branches through.
///
/// ARM64 (Android NDK, both 3- and 4-instruction forms):
///   `adrp x16, page; ldr x17, [x16, #off]; [add x16, x16, #off;] br x17`
/// x86-64: `[endbr64] jmp qword [rip + disp32]`.
fn plt_slot(il2cpp: &crate::il2cpp::base::Il2Cpp, a: u64, is_arm64: bool, is_x64: bool) -> Option<u64> {
    let b = il2cpp.read_bytes_at_rva(a, 16)?;
    if is_arm64 {
        let w0 = u32::from_le_bytes(b.get(0..4)?.try_into().ok()?);
        let w1 = u32::from_le_bytes(b.get(4..8)?.try_into().ok()?);
        // adrp x16, page
        if (w0 & 0x9F00_001F) != 0x9000_0010 {
            return None;
        }
        let immlo = ((w0 >> 29) & 0x3) as i64;
        let immhi = ((w0 >> 5) & 0x7_FFFF) as i64;
        let mut imm = (immhi << 2) | immlo;
        if imm & (1 << 20) != 0 {
            imm -= 1 << 21; // sign-extend the 21-bit page offset
        }
        let page = (a & !0xFFF) as i64 + (imm << 12);
        // ldr xN, [x16, #off]  (64-bit unsigned offset)
        if (w1 & 0xFFC0_0000) != 0xF940_0000 {
            return None;
        }
        let off = (((w1 >> 10) & 0xFFF) as i64) * 8;
        return Some((page + off) as u64);
    }
    if is_x64 {
        let (b, prefix) = if b.starts_with(&[0xF3, 0x0F, 0x1E, 0xFA]) {
            (&b[4..], 4u64)
        } else {
            (&b[..], 0u64)
        };
        if b.len() >= 6 && b[0] == 0xFF && b[1] == 0x25 {
            let disp = i32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            return Some(((a + prefix + 6) as i64 + disp as i64) as u64);
        }
    }
    None
}

/// The imported symbol a direct call target resolves to (libc/libc++/cxa).
fn imported_name(
    il2cpp: &crate::il2cpp::base::Il2Cpp,
    target: u64,
    is_arm64: bool,
    is_x64: bool,
) -> Option<String> {
    let slot = plt_slot(il2cpp, target, is_arm64, is_x64)?;
    il2cpp
        .bind_symbols
        .get(&slot)
        .or_else(|| il2cpp.bind_symbols.get(&il2cpp.get_rva(slot)))
        .cloned()
}

/// Every direct call target in the first few blocks of a helper body, following
/// unconditional TAIL jumps (a helper that ends in `b builder` still owns the
/// work the builder performs). Bounded: 3 hops, 32 instructions each; a jump is
/// only followed when it leaves the lifted window, which is what a tail call
/// looks like.
fn plausible_function_entry(
    il2cpp: &crate::il2cpp::base::Il2Cpp,
    addr: u64,
    is_arm64: bool,
    is_x64: bool,
) -> bool {
    let Some(bytes) = il2cpp.read_bytes_at_rva(addr, 32) else {
        return false;
    };
    if is_arm64 {
        if bytes.len() < 8 {
            return false;
        }
        for i in 0..(bytes.len() / 4).min(8) {
            let w = u32::from_le_bytes([
                bytes[4 * i],
                bytes[4 * i + 1],
                bytes[4 * i + 2],
                bytes[4 * i + 3],
            ]);
            // bti c / pacibsp
            if w == 0xD503245F || w == 0xD503233F {
                return true;
            }
            // stp x29, x30 / xN, x30, [sp, #-imm]!  (frame + return address)
            if w & 0xFFC07FE0 == 0xA9807BE0 {
                return true;
            }
            // sub sp, sp, #imm
            if w & 0xFFC003FF == 0xD10003FF {
                return true;
            }
            // str x30, [sp, #imm]
            if w & 0xFFC003FF == 0xF90003FE {
                return true;
            }
            // str x30, [sp, #-imm]! — the PRE-INDEX leaf prologue. This is a
            // real function entry and was the reason a proven raise helper
            // (0x1302634 in the temple-run-2 corpus) was marked a non-entry:
            // `classify_bodies` then DELETED its RaiseException role, every
            // guard/tail raise rendered as `fn_1302634`, and the ABI staging
            // stores feeding those calls stayed in the text.
            if w & 0xFFC00C00 == 0xF8000C00 && ((w >> 5) & 0x1F) == 31 && (w & 0x1F) == 30 {
                return true;
            }
            // mov x29, sp
            if w & 0xFFFFFC1F == 0x910003FD {
                return true;
            }
        }
        false
    } else if is_x64 {
        // endbr64, or a saved frame/return slot before any call.
        if bytes.windows(4).take(bytes.len().saturating_sub(3)).any(|w| w == [0xF3, 0x0F, 0x1E, 0xFA]) {
            return true;
        }
        bytes
            .iter()
            .take(12)
            .any(|b| matches!(b, 0x55 | 0x53 | 0x56 | 0x57))
            || bytes.windows(3).take(bytes.len().saturating_sub(2)).any(|w| w == [0x48, 0x83, 0xEC])
            || bytes.windows(3).take(bytes.len().saturating_sub(2)).any(|w| w == [0x48, 0x81, 0xEC])
    } else {
        true
    }
}

fn body_window_calls(
    il2cpp: &crate::il2cpp::base::Il2Cpp,
    start: u64,
    is_arm64: bool,
    is_x64: bool,
) -> Vec<u64> {
    let mut out = Vec::new();
    let mut a = start;
    for _ in 0..3 {
        let Some(bytes) = il2cpp.read_bytes_at_rva(a, 128) else {
            break;
        };
        let Some(insns) = super::lift_helper_body(il2cpp, a, &bytes, 32, is_arm64, is_x64)
        else {
            break;
        };
        for i in &insns {
            if matches!(i.op, Op::Call | Op::CallVoid | Op::IndirectCall) {
                if let Some(Operand::Immediate(t)) = i.operands.first() {
                    out.push(*t as u64);
                }
            }
        }
        let tail = insns.iter().take(16).find_map(|i| match i.op {
            Op::Jump(t) if (t as u64) < a || (t as u64) >= a + 0x80 => Some(t as u64),
            _ => None,
        });
        match tail {
            Some(t) => a = t,
            None => break,
        }
    }
    out
}

/// Re-classify `targets` from their bodies and imports. Returns how many roles
/// were written.
///
/// This runs after the statistical call-site classification, so it acts as the
/// authoritative correction: a target whose body is `il2cpp_value_box` becomes
/// [`RuntimeHelper::Box`] even if an alias chain had claimed it for object_new.
pub fn classify_bodies(
    il2cpp: &crate::il2cpp::base::Il2Cpp,
    managed: &HashMap<u64, super::types::MethodRef>,
    targets: &[u64],
    is_arm64: bool,
    is_x64: bool,
) -> usize {
    let mut impls = export_impl_names(il2cpp, is_arm64, is_x64);
    // A codegen variant thunks to the same root as the export: map the root too
    // so body scans can recognise "calls object_new" through a local wrapper.
    let roots: Vec<(u64, String)> = impls
        .iter()
        .map(|(&rva, name)| {
            (
                super::chase_thunk_root(il2cpp, managed, rva, is_arm64, is_x64),
                name.clone(),
            )
        })
        .collect();
    for (root, name) in roots {
        impls.entry(root).or_insert(name);
    }

    let mut found = 0usize;
    for &t in targets {
        if managed.contains_key(&t) {
            continue;
        }
        let clsdbg = std::env::var_os("ILR_CLSDBG").is_some() && t == 0x1302634;
        if clsdbg {
            eprintln!("[clsdbg] t={t:#x} role_before={:?} non_entry={}", role_of(t), is_non_entry(t));
        }
        // A wrapper the thin-forwarder analysis PROVED never returns must not
        // be reclassified from its body: the wrapper is a few instructions
        // (save LR; call the panic core) and `body_window_calls` legitimately
        // reads into the NEIGHBOURING function, where it found
        // `class_is_assignable_from` and renamed the raise to IL2CPP_IsInst
        // (`IL2CPP_IsInst(instance, instance2)` in 0x14DD344). The raise role
        // is the authority; keep it.
        if role_of(t) == Some(RuntimeHelper::RaiseException) {
            continue;
        }
        let root = super::chase_thunk_root(il2cpp, managed, t, is_arm64, is_x64);
        if managed.contains_key(&root) {
            continue;
        }
        // A PLT stub has no body to classify: the four instructions at the
        // target are `adrp/ldr/add/br` through a GOT slot, and the symbol bound
        // to that slot IS the callee. Reading further would walk into the
        // neighbouring stubs and invent a body out of them, so stop here.
        if let Some(n) = imported_name(il2cpp, t, is_arm64, is_x64) {
            note_symbol_name(t, n);
            continue;
        }
        // A call target that is not a function entry is not a helper. The
        // lifter sometimes resolves an indirect dispatch to an address inside
        // another function (a jump-table slot); classifying from there walks
        // the CONTAINING function's body and invents roles — measured:
        // `0x1302CB4` is a jump-table dispatcher inside another function, yet
        // its reachable body calls class_is_assignable_from and the call sites
        // rendered as `IL2CPP_IsInst(modelAnim, 16)` / one-argument IsInst.
        // Exports/symbols are trusted; otherwise a target that makes calls
        // must have a real prologue. Leaf helpers without a prologue have no
        // call evidence and still classify normally.
        let calls = body_window_calls(il2cpp, root, is_arm64, is_x64);
        let known = impls.contains_key(&t)
            || impls.contains_key(&root)
            || symbol_of(t).is_some();
        if !known
            && !calls.is_empty()
            && !plausible_function_entry(il2cpp, root, is_arm64, is_x64)
        {
            mark_non_entry(t);
            mark_non_entry(root);
            continue;
        }
        let mut names: Vec<String> = Vec::new();
        if let Some(n) = impls.get(&t) {
            names.push(n.clone());
        }
        if let Some(n) = impls.get(&root) {
            names.push(n.clone());
        }
        // A local alias of an exported API is that API. Recorded before the
        // role match so the name survives even when no role claims the target;
        // a proven role still wins in `display_name`.
        if let Some(n) = impls.get(&t).or_else(|| impls.get(&root)) {
            note_symbol_name(t, n.clone());
        }
        for &c in &calls {
            if let Some(n) = impls.get(&c) {                names.push(n.clone());
            }
            if let Some(n) = imported_name(il2cpp, c, is_arm64, is_x64) {
                names.push(n);
            }
            if let Some(n) = il2cpp.cxa_func_rvas.get(&c) {
                names.push(n.clone());
            }
            let cr = super::chase_thunk_root(il2cpp, managed, c, is_arm64, is_x64);
            if let Some(n) = impls.get(&cr) {
                names.push(n.clone());
            }
        }
        let has = |needle: &str| names.iter().any(|n| n.contains(needle));
        // ClassInit is deliberately NOT claimed here. Calling the class-init
        // core appears in many helpers that are not guard bodies; the cleaner
        // then deletes real `if (flag == 0) { init; flag = 1; }` code (and any
        // labels inside it) for every such target. The guard-SHAPE classifier
        // in recovery::mod and the metadata-init call shape remain the only
        // authorities for init roles.
        let role = if has("value_box") {
            RuntimeHelper::Box
        } else if has("array_new") || has("array_element_size") {
            RuntimeHelper::ArrayNew
        } else if has("object_new") {
            if has("memcpy") {
                // allocate + copy the value in: Object::Box's core.
                RuntimeHelper::Box
            } else if has("runtime_object_init") {
                // allocate + run a managed .ctor: an exception constructor.
                RuntimeHelper::ExceptionNew
            } else {
                RuntimeHelper::ObjectNew
            }
        } else if has("class_is_assignable_from") {
            if impls
                .get(&root)
                .is_some_and(|n| n.contains("class_is_assignable_from"))
            {
                RuntimeHelper::ClassIsAssignable
            } else {
                RuntimeHelper::IsInst
            }
        } else {
            // NEVER claim `RaiseException` from body reachability. Allocation
            // and dispatch helpers call the throw core on their FAILURE path
            // while returning normally; the cleanup turns a raise role into a
            // CFG terminator and deletes everything after the call (measured:
            // 68 methods / 198 dangling targets + 277 extra throws when this
            // branch existed). The conservative thin-wrapper classifier in
            // recovery::mod (rename-only) and the arg-taking raise map remain
            // the authority for throws.
            continue;
        };
        if let Ok(skip) = std::env::var("ILR_SKIPROLES") {
            let tag = format!("{role:?}");
            if skip.split(',').any(|r| r.trim() == tag) {
                continue;
            }
        }
        if clsdbg {
            eprintln!("[clsdbg] t={t:#x} matched_role={role:?} names={names:?}");
        }
        note(t, role);
        found += 1;
    }
    found
}

/// The guard slot a memory operand addresses, as `(base name, addend)`.
///
/// The base is carried as a name rather than an index so a slot versioned by
/// SSA still matches the store that sets it: at the point this runs the guard
/// is addressed through a stable local (`Local(0)`, the image base), and both
/// the load and the store name the SAME local id.
fn slot_of(op: &Operand) -> Option<(String, i64)> {
    match op {
        Operand::Memory {
            base,
            index: None,
            addend,
            ..
        } => match &**base {
            Operand::Local(v) => Some((format!("l{v}"), *addend)),
            Operand::Register(r) => Some((format!("r{r}"), *addend)),
            Operand::Immediate(b) => Some((format!("i{b}"), *addend)),
            _ => None,
        },
        _ => None,
    }
}

/// The guard byte slot a memory operand addresses, requiring the ONE-BYTE
/// access width that distinguishes a static-init flag from an eight-byte field
/// load. Same addressable cell as `slot_of`, narrowed to `width == 1`.
fn byte_slot_of(op: &Operand) -> Option<(String, i64)> {
    match op {
        Operand::Memory { width: 1, .. } => slot_of(op),
        _ => None,
    }
}

/// Trace the compared local `v` (whose compare-against-0 is at index `upto`)
/// back to the guard byte slot it reads.
///
/// The arm64 codegen guard is `(*(byte*)(base + off) & 1) == 0` — three
/// instructions: a byte load into a local, an `And(_, 1)`, then the compare.
/// `slot_of` on the compare's operand names nothing (the operand is the local
/// the load landed in), so the slot has to be recovered from the DEF chain:
///   * `v = *(byte*)(slot)`              → `slot` (a byte load), or
///   * `v = w & 1` where `w` traces to a byte slot → that slot.
/// Stops at any terminator and at a bounded window, so a compare whose operand
/// is not provably a masked byte load yields `None` and the call is left alone.
fn trace_byte_slot(insns: &[Insn], upto: usize, v: u32) -> Option<(String, i64)> {
    use super::ir::ArithKind;
    let mut need = v;
    let mut j = upto;
    for _ in 0..GUARD_LOOKBACK {
        if j == 0 {
            return None;
        }
        j -= 1;
        let insn = &insns[j];
        if insn.is_terminator() {
            return None;
        }
        if insn.operands.first().and_then(Operand::as_local) != Some(need) {
            continue;
        }
        match &insn.op {
            Op::Move => {
                let src = insn.operands.get(1)?;
                if let Some(s) = byte_slot_of(src) {
                    return Some(s);
                }
                need = src.as_local()?;
            }
            Op::Arith(ArithKind::And) => {
                // the mask must be the literal 1 — this is the flag-test `& 1`.
                if insn.operands.get(2) != Some(&Operand::Immediate(1)) {
                    return None;
                }
                let src = insn.operands.get(1)?;
                if let Some(s) = byte_slot_of(src) {
                    return Some(s);
                }
                need = src.as_local()?;
            }
            _ => return None,
        }
    }
    None
}

/// Is `insn` (at index `j`) the guard test, and if so which byte slot does it
/// compare against `0`? Handles both arm64 forms the codegen emits:
///   * `Compare(k, *(byte*)(slot), 0)`        — slot named directly,
///   * `Compare(k, (*(byte*)(slot) & 1), 0)`  — slot reached through the mask.
fn compare_guard_slot(insns: &[Insn], j: usize) -> Option<(String, i64)> {
    let insn = &insns[j];
    if let Op::Compare(k) = insn.op {
        if matches!(k, CompareKind::NotEqual | CompareKind::Equal)
            && insn.operands.get(2) == Some(&Operand::Immediate(0))
        {
            let operand = insn.operands.get(1)?;
            // Direct byte-slot compare (arm64 static-init `!= 0`).
            if let Some(s) = byte_slot_of(operand) {
                return Some(s);
            }
            // Masked/loaded form: the operand is a local tracing to a byte slot.
            if let Some(v) = operand.as_local() {
                return trace_byte_slot(insns, j, v);
            }
        }
    }
    None
}

/// Is this a zero-compare guard test whose compared operand `slot_of` cannot
/// name?
///
/// `slot_of` only matches a literal `base+addend` memory operand. On ARM32 the
/// guard byte is reached through the metadata-usage table (`*(byte*)(tbl +
/// *(nint*)(…))`), so the compared operand is the local that load landed in and
/// `slot_of` refuses it — which is why the strict rule never fired there. The
/// SHAPE is the same comparison against `0`; it is recognised as a FALLBACK
/// only, and the store that must close the shape is checked far more narrowly
/// (`indexed_byte_store_to_one`).
fn zero_guard_test(insn: &Insn) -> bool {
    matches!(
        &insn.op,
        Op::Compare(k)
            if matches!(k, CompareKind::NotEqual | CompareKind::Equal)
                && insn.operands.get(2) == Some(&Operand::Immediate(0))
    )
}

/// STRICT (arm64/x64): `insn` sets exactly the guard slot `g` to the literal
/// 1. The 1 travels either as the immediate or in a local set to `1` a few
/// instructions earlier (arm64 codegen spills `mov tmp, #1; strb tmp, [slot]`).
fn sets_slot_to_one(insns: &[Insn], j: usize, g: &(String, i64)) -> bool {
    if !matches!(insns[j].op, Op::Move) {
        return false;
    }
    if insns[j].operands.first().and_then(slot_of).as_ref() != Some(g) {
        return false;
    }
    let Some(val) = insns[j].operands.get(1) else {
        return false;
    };
    if *val == Operand::Immediate(1) {
        return true;
    }
    if let Some(l) = val.as_local() {
        for k in (0..j).rev().take(4) {
            if insns[k].is_terminator() {
                break;
            }
            if matches!(insns[k].op, Op::Move)
                && insns[k].operands.get(1) == Some(&Operand::Immediate(1))
                && insns[k].operands.first().and_then(Operand::as_local) == Some(l)
            {
                return true;
            }
        }
    }
    false
}

/// ARM32: `insns[j]` writes the literal 1 — held either as the immediate or in
/// a register set by a nearby `mov rN, #1` — to an INDEXED byte cell,
/// `*(byte*)(<table base> + <index>)`.
///
/// That indexed metadata-usage byte is what makes the guard the static-init
/// guard rather than just "some compare against 0 near some store of 1". Both
/// halves are required: the loose shape alone also matches the null-check raise
/// (`fn_5ae014`: `if (p == 0) panic(…)`, whose following `x = 1` lands in a
/// plain `base+off` field) and the type-init helper (`fn_5906b0`). Measured on
/// the 4k ARM32 dump: all 2,079 static-init sites carry the indexed write and
/// NONE of the 18 look-alike sites do.
fn indexed_byte_store_to_one(insns: &[Insn], j: usize) -> bool {
    if !matches!(insns[j].op, Op::Move) {
        return false;
    }
    let Some(Operand::Memory {
        base,
        index,
        addend,
        width,
        ..
    }) = insns[j].operands.first()
    else {
        return false;
    };
    if index.is_none() || *width != 1 || *addend != 0 || !matches!(**base, Operand::Immediate(_)) {
        return false;
    }
    let Some(val) = insns[j].operands.get(1) else {
        return false;
    };
    if *val == Operand::Immediate(1) {
        return true;
    }
    for k in (0..j).rev().take(4) {
        if insns[k].is_terminator() {
            break;
        }
        if matches!(insns[k].op, Op::Move)
            && insns[k].operands.get(1) == Some(&Operand::Immediate(1))
            && insns[k].operands.first() == Some(val)
        {
            return true;
        }
    }
    false
}

/// Is this the `[dest, *slot, 0]` guard test, and if so, which slot?
fn guard_test(insn: &Insn) -> Option<(String, i64)> {
    if let Op::Compare(k) = insn.op {
        if matches!(k, CompareKind::NotEqual | CompareKind::Equal) {
            // operands are [condition, lhs, rhs]
            if insn.operands.get(2) == Some(&Operand::Immediate(0)) {
                return insn.operands.get(1).and_then(slot_of);
            }
        }
    }
    None
}

/// How many instructions back the guard test may sit. The compiler emits the
/// test, the branch and then the helper; allowing a few slots of address
/// computation between them keeps the rule from depending on instruction
/// selection noise, while staying short enough that an unrelated comparison
/// earlier in the block cannot be reached.
const GUARD_LOOKBACK: usize = 8;
/// How many instructions forward the `slot = 1` may sit.
const GUARD_LOOKAHEAD: usize = 8;

/// Classify this method's unresolved call targets by call-site shape.
///
/// Returns the number of call sites that matched. Only targets that NO map
/// resolved are considered — a call whose operand is `Operand::Method` already
/// has a better name than anything here.
pub fn detect(insns: &[Insn]) -> usize {
    let mut found = 0usize;
    for i in 0..insns.len() {
        let insn = &insns[i];
        if !matches!(insn.op, Op::Call | Op::CallVoid) {
            continue;
        }
        // An unresolved target is still a bare address; a resolved one carries
        // its `MethodRef` and is not our business.
        let Some(Operand::Immediate(target)) = insn.operands.first() else {
            continue;
        };
        let target = *target as u64;
        if target == 0 {
            continue;
        }

        // (1) the guard test: the nearest preceding Compare against 0.
        //
        // The guard's OWN branch sits between the test and the call, so exactly
        // one `CondJump` is stepped over; a second control transfer means the
        // comparison found beyond it belongs to some other block.
        let mut guard: Option<(String, i64)> = None;
        // ARM32 reaches the guard byte through an INDEXED metadata-usage load
        // (`*(byte*)(base + *(nint*)(…))`), so `slot_of` — which needs a literal
        // base+addend — refuses it and the whole rule never fired on ARM32
        // (`IL2CPP_ClassInit` appears ZERO times on the corpus, while
        // `fn_5827e4` carries the shape at 99% of its 13,060 sites). The SHAPE
        // is unchanged: a byte compared against 0, the guarded call, then the
        // byte set to 1.
        let mut indexed_guard = false;
        let mut crossed_branch = false;
        for j in (0..i).rev().take(GUARD_LOOKBACK) {
            if let Some(s) = guard_test(&insns[j]) {
                guard = Some(s);
                break;
            }
            // arm64 codegen guard: `(*(byte*)(slot) & 1) == 0` / `!= 0`, where
            // the compare names the local the byte load (or its `& 1`) landed
            // in. Recover the slot from the def chain (see `compare_guard_slot`).
            if let Some(s) = compare_guard_slot(insns, j) {
                guard = Some(s);
                break;
            }
            if zero_guard_test(&insns[j]) {
                indexed_guard = true;
                break;
            }
            if insns[j].is_terminator() {
                if !crossed_branch && matches!(insns[j].op, Op::CondJump(_)) {
                    crossed_branch = true;
                    continue;
                }
                break;
            }
        }
        if guard.is_none() && !indexed_guard {
            continue;
        }

        // (2) the guard being set. STRICT: the same slot the test named, set to
        // the literal 1 (arm64/x64, unchanged). ARM32: the matching INDEXED byte
        // write — the only form that separates the static-init guard from a
        // null-check raise or a type-init helper (see
        // `indexed_byte_store_to_one`).
        let mut set = false;
        for j in i + 1..(i + 1 + GUARD_LOOKAHEAD).min(insns.len()) {
            let hit = match &guard {
                Some(g) => sets_slot_to_one(insns, j, g),
                None => indexed_byte_store_to_one(insns, j),
            };
            if hit {
                set = true;
                break;
            }
            if insns[j].is_terminator() {
                break;
            }
        }
        if !set {
            continue;
        }

        note(target, RuntimeHelper::ClassInit);
        super::timing::RT_CLASSINIT_SITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        found += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::ir::Insn;

    fn mem(base: Operand, addend: i64) -> Operand {
        Operand::Memory {
            base: Box::new(base),
            index: None,
            scale: 1,
            addend,
            width: 0,
        }
    }

    /// The full guard shape: test, branch, helper call, `guard = 1`.
    fn guard_body(target: i64, slot_base: Operand, addend: i64) -> Vec<Insn> {
        vec![
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![
                    Operand::Local(50),
                    mem(slot_base.clone(), addend),
                    Operand::Immediate(0),
                ],
                0x10,
            ),
            Insn::new(Op::CondJump(3), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Immediate(target),
                    Operand::Local(9),
                    Operand::Local(3),
                    mem(slot_base.clone(), addend + 16),
                ],
                0x18,
            ),
            Insn::new(
                Op::Move,
                vec![mem(slot_base, addend), Operand::Immediate(1)],
                0x20,
            ),
        ]
    }

    #[test]
    fn a_guard_that_is_tested_called_and_set_names_the_helper() {
        let insns = guard_body(0x11c7f0, Operand::Local(0), 9909776);
        assert_eq!(detect(&insns), 1);
        assert_eq!(role_of(0x11c7f0), Some(RuntimeHelper::ClassInit));
        assert_eq!(display_name(0x11c7f0), "IL2CPP_ClassInit");
        // and an address that was never classified keeps the old rendering
        assert_eq!(display_name(0xdeadbeef), "fn_deadbeef");
    }

    /// The ARM32 guard shape: the byte is reached through an INDEXED load, so
    /// `slot_of` names no slot, and the guard's read and write cells differ
    /// textually (`fn_5827e4` is 99% this shape on the corpus).
    #[test]
    fn an_indexed_guard_still_names_the_class_init_helper() {
        let idx = |base: i64, v: u32| Operand::Memory {
            base: Box::new(Operand::Immediate(base)),
            index: Some(Box::new(Operand::Local(v))),
            scale: 1,
            addend: 0,
            width: 1,
        };
        let insns = vec![
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![
                    Operand::Local(50),
                    idx(0x12eb194, 60),
                    Operand::Immediate(0),
                ],
                0x10,
            ),
            Insn::new(Op::CondJump(3), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Immediate(0x5827e4),
                    Operand::Local(9),
                    Operand::Local(3),
                ],
                0x18,
            ),
            // The write names a DIFFERENT cell (0x12eb1bc / local 61) — the
            // indexed form has no textual slot to compare, so the shape is what
            // carries the identification.
            Insn::new(
                Op::Move,
                vec![idx(0x12eb1bc, 61), Operand::Immediate(1)],
                0x20,
            ),
        ];
        assert_eq!(detect(&insns), 1);
        assert_eq!(display_name(0x5827e4), "IL2CPP_ClassInit");
    }

    /// A bare `== 0` compare with no `= 1` store is a null check, not a guard.
    #[test]
    fn an_indexed_zero_compare_without_a_store_is_not_a_class_init() {
        let insns = vec![
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![
                    Operand::Local(50),
                    Operand::Memory {
                        base: Box::new(Operand::Immediate(0x10)),
                        index: Some(Box::new(Operand::Local(60))),
                        scale: 1,
                        addend: 0,
                        width: 1,
                    },
                    Operand::Immediate(0),
                ],
                0x10,
            ),
            Insn::new(Op::CondJump(3), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x5ae014), Operand::Local(9)],
                0x18,
            ),
        ];
        assert_eq!(detect(&insns), 0);
        assert_eq!(display_name(0x5ae014), "fn_5ae014");
    }

    /// The measured ARM32 FALSE positive: a null-check raise (`fn_5ae014`:
    /// `if (p == 0) panic(…)`) followed by an unrelated `field = 1`. The store
    /// is a plain `base+off` byte, NOT the indexed metadata-usage cell, so the
    /// shape must not close.
    #[test]
    fn a_null_check_raise_with_a_plain_byte_store_is_not_a_class_init() {
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(70), Operand::Immediate(1)],
                0x08,
            ),
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![
                    Operand::Local(50),
                    Operand::Local(60),
                    Operand::Immediate(0),
                ],
                0x10,
            ),
            Insn::new(Op::CondJump(3), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x5ae014), Operand::Local(9)],
                0x18,
            ),
            Insn::new(
                Op::Move,
                vec![mem(Operand::Local(9), 88), Operand::Local(70)],
                0x20,
            ),
        ];
        assert_eq!(detect(&insns), 0);
        assert_eq!(display_name(0x5ae014), "fn_5ae014");
    }

    #[test]
    fn a_call_before_a_guard_is_not_a_class_init() {
        // The store sets a DIFFERENT slot, so the shape does not close.
        let mut insns = guard_body(0x2222, Operand::Local(0), 9909776);
        insns[3] = Insn::new(
            Op::Move,
            vec![mem(Operand::Local(0), 4242), Operand::Immediate(1)],
            0x20,
        );
        assert_eq!(detect(&insns), 0);
    }

    #[test]
    fn a_guarded_call_is_not_the_guard() {
        // Test and store agree, but the CALL follows the store instead of
        // sitting between them.
        let insns = vec![
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![
                    Operand::Local(50),
                    mem(Operand::Local(0), 88),
                    Operand::Immediate(0),
                ],
                0x10,
            ),
            Insn::new(Op::CondJump(4), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Move,
                vec![mem(Operand::Local(0), 88), Operand::Immediate(1)],
                0x14,
            ),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x3333), Operand::Local(9)],
                0x18,
            ),
        ];
        assert_eq!(detect(&insns), 0);
    }

    /// The arm64 byte-guard, DIRECT form: `if (*(byte*)(base + off) != 0) {
    /// ClassInit(...); *(byte*)(base + off) = 1; }`. The compare names the local
    /// the byte load landed in, so `slot_of` cannot see the slot from the compare
    /// alone — `trace_byte_slot` recovers it from the load's def.
    #[test]
    fn an_arm64_direct_byte_guard_names_the_class_init_helper() {
        let byte_slot = Operand::Memory {
            base: Box::new(Operand::Local(29)),
            index: None,
            scale: 8,
            addend: 2831,
            width: 1,
        };
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(51), byte_slot.clone()],
                0x10,
            ),
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![Operand::Local(60), Operand::Local(51), Operand::Immediate(0)],
                0x14,
            ),
            Insn::new(Op::CondJump(41), vec![Operand::Local(60)], 0x18),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x130240c), Operand::Local(54), Operand::Local(53)],
                0x1c,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(55), Operand::Immediate(1)],
                0x20,
            ),
            Insn::new(
                Op::Move,
                vec![byte_slot, Operand::Local(55)],
                0x24,
            ),
        ];
        assert_eq!(detect(&insns), 1);
        assert_eq!(role_of(0x130240c), Some(RuntimeHelper::ClassInit));
        assert_eq!(display_name(0x130240c), "IL2CPP_ClassInit");
    }

    /// The arm64 byte-guard, MASKED form: `if ((*(byte*)(base + off) & 1) == 0)
    /// { ClassInit(...); *(byte*)(base + off) = 1; }` — the load, then `And(_,
    /// 1)`, then the compare against 0.
    #[test]
    fn an_arm64_masked_byte_guard_names_the_class_init_helper() {
        use super::super::ir::ArithKind;
        let byte_slot = Operand::Memory {
            base: Box::new(Operand::Local(29)),
            index: None,
            scale: 8,
            addend: 2603,
            width: 1,
        };
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(31), byte_slot.clone()],
                0x10,
            ),
            Insn::new(
                Op::Arith(ArithKind::And),
                vec![Operand::Local(40), Operand::Local(31), Operand::Immediate(1)],
                0x14,
            ),
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![Operand::Local(41), Operand::Local(40), Operand::Immediate(0)],
                0x18,
            ),
            Insn::new(Op::CondJump(28), vec![Operand::Local(41)], 0x1c),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x1302410), Operand::Local(36), Operand::Local(35)],
                0x20,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(46), Operand::Immediate(1)],
                0x24,
            ),
            Insn::new(Op::Move, vec![byte_slot, Operand::Local(46)], 0x28),
        ];
        assert_eq!(detect(&insns), 1);
        assert_eq!(display_name(0x1302410), "IL2CPP_ClassInit");
    }

    /// A masked `& 1 == 0` compare whose operand does NOT trace to a byte slot
    /// is just a null check, not a guard: it stays `fn_`.
    #[test]
    fn a_masked_compare_without_a_byte_slot_is_not_a_class_init() {
        use super::super::ir::ArithKind;
        let insns = vec![
            Insn::new(
                Op::Arith(ArithKind::And),
                vec![Operand::Local(40), Operand::Local(31), Operand::Immediate(1)],
                0x14,
            ),
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![Operand::Local(41), Operand::Local(40), Operand::Immediate(0)],
                0x18,
            ),
            Insn::new(Op::CondJump(28), vec![Operand::Local(41)], 0x1c),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x1302634), Operand::Local(36)],
                0x20,
            ),
        ];
        assert_eq!(detect(&insns), 0);
        assert_eq!(display_name(0x1302634), "fn_1302634");
    }

    /// The panic-helper role renames a call to a non-`fn_` label without ever
    /// rewriting it to `Op::Throw` (that is done elsewhere, off `maps`).
    #[test]
    fn a_raise_helper_renames_without_the_fn_pattern() {
        note(0x9a11ce, RuntimeHelper::RaiseException);
        let name = display_name(0x9a11ce);
        assert_eq!(name, "IL2CPP_RaiseException");
        // crucially: no `fn_<hex>` remains, so it drops out of the census.
        assert!(!name.contains("fn_"));
    }

    /// ObjectNew role renames the allocator call so it drops out of the fn_
    /// census. Unlike the HelperMaps object_new (which rewrites to NewObj),
    /// this is RENAME-ONLY — no IR rewriting, no dead-code risk.
    #[test]
    fn an_object_new_helper_renames_without_the_fn_pattern() {
        note(0xb00001, RuntimeHelper::ObjectNew);
        let name = display_name(0xb00001);
        assert_eq!(name, "IL2CPP_ObjectNew");
        assert!(!name.contains("fn_"));
    }

    /// Box role renames the boxing helper call.
    #[test]
    fn a_box_helper_renames_without_the_fn_pattern() {
        note(0xb00002, RuntimeHelper::Box);
        let name = display_name(0xb00002);
        assert_eq!(name, "IL2CPP_Box");
        assert!(!name.contains("fn_"));
    }

    /// ArrayNew role renames the array allocator call.
    #[test]
    fn an_array_new_helper_renames_without_the_fn_pattern() {
        note(0xb00003, RuntimeHelper::ArrayNew);
        let name = display_name(0xb00003);
        assert_eq!(name, "IL2CPP_ArrayNew");
        assert!(!name.contains("fn_"));
    }

    #[test]
    fn a_resolved_target_is_never_relabelled() {
        use crate::recovery::types::{ManagedType, MethodRef, MethodSigInfo, PrimKind};
        let m = MethodRef {
            image_index: 0,
            type_def_index: 1,
            method_index: 2,
            name: "M".into(),
            declaring_namespace: "N".into(),
            declaring_type_name: "T".into(),
            assembly: "a".into(),
            sig: MethodSigInfo {
                ret: ManagedType::Primitive(PrimKind::Void),
                params: vec![],
                is_static: true,
            },
        };
        let mut insns = guard_body(0x11c7f0, Operand::Local(0), 9909776);
        insns[2].operands[0] = Operand::Method(m);
        assert_eq!(detect(&insns), 0);
    }

    /// An unresolved target the BINARY names (a PLT stub, or a local alias of
    /// an exported API) must render with that symbol, not `fn_…`.
    ///
    /// Measured on temple-run-2: 2,107 `fn_` occurrences were PLT stubs whose
    /// names are in `.rela.plt` (`__cxa_begin_catch` 745, `__cxa_end_catch`
    /// 838, plus `__cxa_allocate_exception`/`__cxa_throw`/`memcpy`/`modf`),
    /// and 1,025 more were local aliases of exported APIs
    /// (`0x1302514`→`il2cpp_runtime_class_init` 946, `0x1302520`→
    /// `il2cpp_object_unbox`, `0x130251c`→`il2cpp_value_box`).
    #[test]
    fn an_imported_symbol_names_its_target() {
        note_symbol_name(0xc0ffee, "__cxa_end_catch".into());
        assert_eq!(display_name(0xc0ffee), "__cxa_end_catch");
        assert_eq!(symbol_of(0xc0ffee).as_deref(), Some("__cxa_end_catch"));
        // and a target with no symbol and no role keeps the old rendering
        assert_eq!(display_name(0xc0ffef), "fn_c0ffef");
    }

    /// A proven role outranks the symbol: the role says what the call MEANS,
    /// the symbol only says which entry point implements it.
    ///
    /// Neutralised by swapping the two checks in `display_name` — this test
    /// then yields `il2cpp_object_new` instead of `IL2CPP_ObjectNew`.
    #[test]
    fn a_role_outranks_a_symbol() {
        note_symbol_name(0xc0fff0, "il2cpp_object_new".into());
        note(0xc0fff0, RuntimeHelper::ObjectNew);
        assert_eq!(display_name(0xc0fff0), "IL2CPP_ObjectNew");
    }
}
