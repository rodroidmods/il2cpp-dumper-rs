//! Typed local variables + type propagation fixpoint
//! (port of Cpp2IL `Analysis/LocalVariables.cs`).
//!
//! Locals are created from SSA versioned registers; types are seeded from
//! method signature (`this`, params) and propagated monotonically through
//! moves/arithmetic/calls until fixpoint. Field accesses `[base + offset]`
//! resolve to FieldRefs via a metadata-backed lookup callback.

use super::ir::{Insn, Op, Operand};
use super::types::{FieldRef, ManagedType, PrimKind};
use std::collections::HashMap;

/// Metadata lookups provided by RecoveryContext (no Resolution needed).
pub trait TypeOracle {
    /// Field at `offset` inside the type of `base_tdi` (walks base chain).
    fn field_at_offset(&self, tdi: i32, offset: i64) -> Option<FieldRef>;
    /// Static fields storage type name for klass pointer typing (optional).
    fn static_fields_type(&self, _tdi: i32) -> Option<i32> {
        None
    }
    /// Resolve an ABSOLUTE static-storage address to the static field it names.
    ///
    /// IL2CPP gives every type one static storage block in the data section, so
    /// `Type.field` loads/stores at `block + off`. `resolve::run_with_fields`
    /// names the form where the block address is materialised through a
    /// FieldInfo metadata-usage slot; the residue this method exists for is the
    /// form where that address is a BARE CONSTANT — `*(nint*)(0x63a… + off)`,
    /// the census's `staticabs`/`asingleton` families (53,384 methods on the
    /// arm64 corpus, ~50% of the dump). Returning the `FieldRef` (with
    /// `is_static = true`) lets [`propagate_types`] rewrite the access to
    /// `Type.field` and type its result, exactly as a resolved static slot
    /// would, and unlike an instance field a STATIC FieldRef renders from its
    /// own `declaring_type_name` — it needs no receiver, so nothing is dropped.
    /// `addr` already includes the in-block offset. Default `None`, so an
    /// oracle that does not implement it changes nothing.
    fn static_field_at(&self, _addr: i64) -> Option<FieldRef> {
        None
    }
    /// Is `tdi` a struct value type — `is_value_type && !is_enum` in the
    /// `Il2CppTypeDefinition` bitfield?
    ///
    /// This is the guard that keeps field-address seeding sound. `dst = base +
    /// off` is the field's ADDRESS, not its value: when the field at
    /// `(base, off)` is a reference type, typing `dst` as the field's class
    /// makes the next `[dst + off2]` resolve as a subfield of the wrong owner
    /// (measured pre-guard: 425k of 1,030,800 dump.cs lines changed, stores
    /// lost to comments, `(raw + 16).value__`-style misnames). Only the address
    /// of a VALUE-typed field may seed a struct-typed base — and then as
    /// `ByRef(field type)`, still an address — while a reference-typed field's
    /// address stays the integer guess. Enums are refused too: their only field
    /// is the compiler-named `value__`, so seeding buys no subfield and risks
    /// exactly that misname. Default `None` (unknown → refuse), so oracles
    /// without metadata change nothing.
    fn is_struct_value_type(&self, _tdi: i32) -> Option<bool> {
        None
    }
}

#[derive(Clone)]
pub struct LocalInfo {
    pub slot_origin: String,
    pub ty: ManagedType,
    pub is_this: bool,
    pub is_param: Option<usize>,
    /// When `is_param = Some(p)`: which consecutive float component of the
    /// parameter this local holds (None = the whole single-register param).
    /// An FP-aggregate (HFA struct) parameter like `Vector3 v` occupies one
    /// v-register per float field; comp 0 is v.x, comp 1 is v.y, ...
    pub param_comp: Option<usize>,
    pub is_return_buffer: bool,
    /// The ENTRY value of a register the ABI can hand a value to the function
    /// in, which the body never defines.
    ///
    /// Two families, both INCOMING values rather than corruption:
    ///   * an ARGUMENT register the declared signature does not cover. IL2CPP
    ///     folds several metadata methods onto one native body, and the member
    ///     this RVA is keyed to is an arbitrary one of them — so a body like
    ///     `sub x0, x1, x0 ; ret` can be reached through a member that declares
    ///     a single parameter while the code reads two registers.
    ///   * a CALLEE-SAVED register (`x19..x28`, `v8..v15`): the prologue spills
    ///     the caller's value and the epilogue restores it; the method itself
    ///     never defines it, and the pin-copy fold often erases the single
    ///     write that established it (`mov x19, x0`).
    ///
    /// Classifying these as incoming keeps them out of the undefined-local
    /// oracle and roots them in `rooted_versions`, while leaving `is_param`
    /// alone (an uncovered argument has no display name in the declared
    /// signature). Caller-saved temporaries (`x9..x18`) are NOT incoming, so a
    /// read of one the body never wrote is still a genuine lost definition.
    pub is_incoming_arg: bool,
    /// Defined by a CALL's ABI clobber rather than by a listing instruction.
    ///
    /// A call overwrites the caller-saved FP return registers; the SSA
    /// definition exists so post-call reads do not reuse pre-call versions,
    /// but there is no instruction to point at. The value is whatever the
    /// callee delivered (or garbage), so the undefined-local oracle excuses
    /// it; it is NOT an incoming argument and keeps its `vN` spelling.
    pub is_call_clobber: bool,
}

#[derive(Default, Clone)]
pub struct LocalTable {
    pub infos: Vec<LocalInfo>,
    /// version id → table index
    pub index_of: HashMap<u32, usize>,
    /// parameter display names (parallel to is_param indices)
    pub param_names: Vec<String>,
    /// For each parameter that is an FP aggregate: the metadata field names
    /// of its consecutive float components (Vector3 → ["x", "y", "z"]).
    /// Empty for non-FP parameters.
    pub param_comp_names: Vec<Vec<String>>,
}

impl LocalTable {
    pub fn len(&self) -> usize {
        self.infos.len()
    }
}

/// Build locals from SSA output: every distinct version becomes one local;
/// slots named after ABI registers get seeded types from the signature.
#[derive(Default)]
pub struct SeedInfo {
    pub param_types: Vec<ManagedType>,
    pub ret_type: ManagedType,
    pub is_static: bool,
    /// Declaring type of the method — seeds `this` with real class identity
    /// so instance-field loads resolve via the oracle.
    pub declaring: Option<std::sync::Arc<super::types::NamedType>>,
    /// Parameter display names parallel to `param_types`.
    pub param_names: Vec<String>,
    /// For each parameter: metadata field names of its consecutive float
    /// components when the parameter is an HFA float-aggregate value type
    /// (Vector3 → ["x", "y", "z"]); empty otherwise. Determines how many
    /// consecutive FP argument registers the ABI assigns to the parameter
    /// and how those registers render (`rootPosition.x` not `v26`).
    pub param_comp_names: Vec<Vec<String>>,
    /// AAPCS32 word count for each declared parameter. Value types can occupy
    /// multiple core registers (or the incoming stack area); references and
    /// scalars occupy one word. Zero means the size is unknown and uses the
    /// conservative one-word mapping.
    pub arm32_param_words: Vec<usize>,
    /// The target is 32-bit ARM (A32).
    ///
    /// This is a DECLARED fact, not a guess. `create` used to infer "is this a
    /// register-by-ordinal x86 caller?" from the slot NAMES alone, and the test
    /// (`rv`/`abs`/any `gN`) was wrong for ARM32: the ARM32 canonicalizer renders
    /// `lr`/`pc` as `g14`/`g15`, so every method that touched its link register
    /// was classified x86-64. That flipped `arm_like` to false and turned OFF the
    /// whole incoming-argument seeding — every read of `x0..x8` (all of which ARE
    /// ARM32's argument registers) became an undefined local. The architecture is
    /// already known to the caller, so it is passed rather than re-derived.
    pub is_arm32: bool,
    /// Does this method DELIVER ITS RESULT through a caller-allocated buffer?
    ///
    /// AAPCS32 §5.4: a Composite Type result is returned in `r0` only while it
    /// is 4 bytes or smaller; anything larger is written through a buffer whose
    /// address the caller passes as an EXTRA FIRST argument. `r0` is therefore
    /// the return buffer — not `this` — and every real argument shifts up one
    /// register (`this` in `r1`, parameters from `r2`).
    ///
    /// Without this, `Vector3 get_position()` (12-byte result) was seeded as
    /// if `r0` were `this`: the caller's result buffer got field stores
    /// attributed to the object (`*(nint*)(this + 20) = …` for a Matrix4x4
    /// copy) and the genuine `this` in `r1` was an unnamed `in_x1`.
    pub ret_hidden_buffer: bool,
}

/// Drop the `is_this` claim from any local the IR actually writes.
///
/// `this` is immutable, so an entry version that the body ASSIGNS is not the
/// this pointer — it is an ordinary local that merely *starts out* holding it.
/// That is exactly the loop-carried register of a rotated loop the C# compiler
/// emits for `while (inner != null) inner = inner.InnerException;`:
///
/// ```text
///   x8 = x0                     ; back = inner
///   x0 = [x0, #0x28]            ; inner = inner.InnerException
///   cbnz x0, -8                 ; repeat
/// ```
///
/// where `x0` is both the incoming `this` and the thing being reassigned. While
/// the entry version kept `is_this`, the renderer could not express the write
/// at all — `this = …` is not a statement, so it dropped the assignment, the
/// loop body came out empty and the test re-read the original field forever
/// (`while (this._innerException != 0) { // loop body not recovered }`, an
/// unkillable infinite loop). As a plain local the same write renders normally.
pub fn unclaim_written_entry_this(table: &mut LocalTable, insns: &[super::ir::Insn]) {
    let mut written: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for insn in insns {
        if let Some(super::ir::Operand::Local(v)) = insn.dest() {
            written.insert(*v);
        }
    }
    if written.is_empty() {
        return;
    }
    for (v, idx) in table.index_of.iter() {
        if !written.contains(v) {
            continue;
        }
        if let Some(info) = table.infos.get_mut(*idx) {
            info.is_this = false;
        }
    }
}

pub fn create(ssa_version_slot: &HashMap<u32, String>, seed: &SeedInfo) -> LocalTable {
    let mut table = LocalTable {
        param_names: seed.param_names.clone(),
        param_comp_names: seed.param_comp_names.clone(),
        ..Default::default()
    };
    let mut by_slot_order: Vec<u32> = ssa_version_slot.keys().copied().collect();
    by_slot_order.sort();

    // ---- ABI argument-register assignment (AAPCS64 / SysV x64) ----
    // Every parameter is allocated to the next free register of its class:
    // `this` takes x0, integer/reference parameters take x1..x7 (or the
    // stack spill area beyond), float (R4)/double (R8) parameters take the
    // SIMD/FP registers v0..v7, and small float-aggregate structs (HFA
    // value types — Vector2/3, Color, Quaternion, ...) occupy consecutive
    // v-registers. IL2CPP emits the platform ABI, so entry registers follow
    // this exactly. The OLD positional scheme bound param i to x(1+i) no
    // matter what class the parameter was — a `Vector3` parameter sits in
    // s0-s2, not x1, and the x-register it got instead was stale scratch.
    //
    // Slot origins: arm64 names SIMD registers by view width (flattened `vs`
    // lane indices for the 32-bit s-view, "vd0" for the 64-bit d-view); x86-64 SysV normalizes
    // XMM to "vf0".."vf7". Pick the scheme from the registers present.
    let fp_arm64 = by_slot_order.iter().any(|v| {
        let o = &ssa_version_slot[v];
        o.starts_with("vs") || o.starts_with("vd")
    });
    // The hidden indirect-result pointer is passed in the FIRST integer
    // argument register, so every declared argument shifts one slot up. Only
    // the ARM32 (`r0`-in-register) ABI does this: AArch64 passes the buffer in
    // `x8` (`SRET_OPERAND`), leaving `x0` as `this`. The flag is set ONLY for
    // ARM32 by `mod::arm32_ret_uses_hidden_buffer`, so it needs no arch test
    // here — and `fp_arm64` must NOT be used as one, because an ARM32 method
    // that touches VFP names its registers `vs`/`vd` exactly like AArch64.
    let sret_shift = usize::from(seed.ret_hidden_buffer);
    // origin → (param index, float component within an FP aggregate)
    let assign: HashMap<String, (usize, Option<usize>)> = if seed.is_arm32 {
        abi_arm32_param_locations(
            &seed.param_types,
            &seed.arm32_param_words,
            seed.is_static,
            sret_shift,
        )
    } else {
        abi_param_registers(
            &seed.param_types,
            &seed.param_comp_names,
            seed.is_static,
            fp_arm64,
            sret_shift,
        )
    }
    .into_iter()
    .map(|(reg, pi, comp)| (reg, (pi, comp)))
    .collect();

    // Every register an ABI can hand a value to a function in, so a slot the
    // declared signature does not cover can be recognised as incoming (see
    // [`LocalInfo::is_incoming_arg`]). Three families:
    //   * the argument registers `x0..x7` (plus `x8`, the indirect-result
    //     pointer — an incoming value, never computed by the callee);
    //   * the callee-saved `x19..x28`, whose ENTRY value is the caller's — the
    //     prologue spills it and the epilogue restores it, and the method
    //     itself never defines it;
    //   * the FP argument registers `v0..v7` (all four S lanes), and the
    //     callee-saved low 64 bits of `v8..v15` (S lanes 0 and 1).
    // Deliberately EXCLUDED are the caller-saved temporaries `x9..x18`: they
    // have no incoming meaning, so a read of one the body never wrote is a
    // genuine lost definition and must keep being reported.
    //
    // AAPCS64 ONLY. The x86-64 lifter names registers by ABI ORDINAL
    // (`x3`/`x0`/`x2`/`g3`/`g6`/`g7` are its argument registers, and `x4` is a
    // caller-saved temp), so the same strings would mark the wrong slots
    // incoming there and hide real lost definitions.
    // AAPCS64 vs x86-64. `fp_arm64` alone is NOT a reliable signal: a purely
    // integer arm64 method (`sub x0, x1, x0 ; ret`) has no v-registers at all.
    // The x86-64 lifter names registers by ABI ordinal instead, and `rv` (the
    // result), `g3`/`g6`/`g7` (arguments) and `abs` (an absolute address) are
    // names the ARM lifters never produce — so their presence proves x86-64.
    let x86_named = by_slot_order.iter().any(|v| {
        let o = ssa_version_slot[v].as_str();
        o == "rv" || o == "abs" || o.starts_with('g')
    });
    // `is_arm32` overrides the NAME heuristic: ARM32's `lr`/`pc` render as
    // `g14`/`g15`, which the `starts_with('g')` test reads as x86-64. See
    // `SeedInfo::is_arm32`.
    let arm_like = seed.is_arm32 || fp_arm64 || !x86_named;

    let mut abi_arg_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    if arm_like {
        for i in 0..=8 {
            // `this` is not an argument, but a static method's x0..x7 are;
            // either way xN is an incoming ABI slot. x8 is the indirect-result
            // (sret) pointer — incoming, never computed by the callee.
            abi_arg_names.insert(format!("x{i}"));
            if i < 8 {
                for lane in 0..4 {
                    abi_arg_names.insert(format!("vs{}", i * 4 + lane));
                }
                abi_arg_names.insert(format!("vd{i}"));
            }
        }
        for i in 19..=28 {
            abi_arg_names.insert(format!("x{i}"));
        }
        for i in 8..16 {
            // AAPCS64 preserves only the low 64 bits of v8-v15.
            for lane in 0..2 {
                abi_arg_names.insert(format!("vs{}", i * 4 + lane));
            }
            abi_arg_names.insert(format!("vd{i}"));
        }
    }

    // Only the ENTRY version of an ABI register is the incoming parameter.
    // The compiler reuses x0–x7 as scratch after the prologue; marking every
    // version of x1 as `inherit` produced garbage like `inherit = 0x6044000`
    // (a metadata pointer stored into a reused register) and `x = x` renders.
    // Versions are allocated 0-first per slot in the seeding loop, so the
    // first (lowest-id) version seen per origin IS the entry version.
    let mut seen_entry: std::collections::HashSet<String> = std::collections::HashSet::new();

    for v in by_slot_order {
        let origin = ssa_version_slot[&v].clone();
        let is_entry_version = seen_entry.insert(origin.clone());
        let mut info = LocalInfo {
            slot_origin: origin.clone(),
            ty: ManagedType::Unknown,
            is_this: false,
            is_param: None,
            param_comp: None,
            is_return_buffer: false,
            is_incoming_arg: false,
            is_call_clobber: false,
        };
        // The indirect-result buffer occupies `x0` and is an INCOMING value the
        // callee only ever WRITES through — never `this`, never a parameter.
        if is_entry_version && sret_shift == 1 && origin == "x0" {
            info.is_return_buffer = true;
        }
        if is_entry_version && !seed.is_static && origin == format!("x{sret_shift}") {
            info.ty = match &seed.declaring {
                Some(nt) => ManagedType::Class(nt.clone()),
                None => ManagedType::Class(super::types::NamedType::object()),
            };
            info.is_this = true;
            // Count `this` as a seeded typed parameter.
            {
                use std::sync::atomic::Ordering;
                super::timing::PARAMS_SEEDED_TOTAL.fetch_add(1, Ordering::Relaxed);
                super::timing::PARAMS_SEEDED_TYPED.fetch_add(1, Ordering::Relaxed);
            }
        }
        if is_entry_version {
            if let Some(&(pi, comp)) = assign.get(&origin) {
                // An HFA parameter occupies one FP register per float field.
                // The local is that scalar component, not a second copy of the
                // aggregate type. Giving (say) `s0` the type Vector3 let field
                // resolution treat it as an aggregate base, producing invalid
                // chains such as `position.x.y` when the ABI spill of the
                // parameter was analyzed as stack memory.
                info.ty = if comp.is_some() {
                    ManagedType::Primitive(PrimKind::R4)
                } else {
                    seed.param_types[pi].clone()
                };
                info.is_param = Some(pi);
                info.param_comp = comp;
                // Count parameter seeding for impact measurement.
                {
                    use std::sync::atomic::Ordering;
                    super::timing::PARAMS_SEEDED_TOTAL.fetch_add(1, Ordering::Relaxed);
                    if !matches!(info.ty, ManagedType::Unknown) {
                        super::timing::PARAMS_SEEDED_TYPED.fetch_add(1, Ordering::Relaxed);
                    }
                }
            } else if !info.is_this && !info.is_return_buffer && abi_arg_names.contains(&origin) {
                // An argument register the declared signature does not cover.
                // Only the ENTRY version can be incoming; a later version is
                // written by the body and is not a parameter.
                info.is_incoming_arg = true;
            }
        }
        let idx = table.infos.len();
        table.infos.push(info);
        table.index_of.insert(v, idx);
    }
    table
}

/// AAPCS32 core-register and stack assignment for incoming managed arguments.
/// Unlike AAPCS64, an aggregate is passed as consecutive core words, and a
/// stack argument is visible to the frame lifter as `stack_8`, `stack_12`, …
/// (entry SP is fp+8 in the common prologue).
fn abi_arm32_param_locations(
    param_types: &[ManagedType],
    param_words: &[usize],
    is_static: bool,
    sret_shift: usize,
) -> Vec<(String, usize, Option<usize>)> {
    let mut out = Vec::new();
    let mut r = sret_shift + usize::from(!is_static);
    let mut stack = 0usize;
    for pi in 0..param_types.len() {
        let words = param_words.get(pi).copied().unwrap_or(0).max(1);
        // A composite is allocated wholly to core registers when possible;
        // otherwise AAPCS32 places it in the stack argument area.
        if r + words <= 4 {
            // One IR local represents each physical register word. Only the
            // first word can carry the managed parameter's display name; the
            // remaining words are raw aggregate pieces and must not render as
            // duplicate copies of the whole C# parameter.
            out.push((format!("x{r}"), pi, None));
            r += words;
        } else {
            let aligned_stack = if words > 1 { (stack + 3) & !3 } else { stack };
            out.push((format!("stack_{}", 8 + aligned_stack), pi, None));
            stack = aligned_stack + words * 4;
            r = 4;
        }
    }
    out
}

/// ABI register assignment for a method's declared parameters (AAPCS64 / x64
/// SysV), returned in PARAMETER order as `(register, param index, float
/// component)`.
///
/// ONE source of truth for this rule. `create` uses it to name the ENTRY values
/// (which register the incoming parameter arrived in); `align_call_arguments`
/// uses the same table to place each argument at the operand position its own
/// signature requires. Two copies of this rule is precisely how a call could
/// be rendered with its float argument read out of an integer register.
pub fn abi_param_registers(
    param_types: &[ManagedType],
    param_comp_names: &[Vec<String>],
    is_static: bool,
    fp_arm64: bool,
    sret_shift: usize,
) -> Vec<(String, usize, Option<usize>)> {
    let fp_prefix = |w: PrimKind| match w {
        PrimKind::R8 if fp_arm64 => "vd",
        PrimKind::R8 => "vf",
        _ if fp_arm64 => "vs",
        _ => "vf",
    };
    let mut out: Vec<(String, usize, Option<usize>)> = Vec::with_capacity(param_types.len());
    // x0 is consumed by the hidden result buffer (when there is one) and then
    // by `this`.
    let mut gpr = sret_shift + usize::from(!is_static);
    let mut fp = 0usize;
    for (pi, ty) in param_types.iter().enumerate() {
        let comps = param_comp_names.get(pi).cloned().unwrap_or_default();
        let is_fp_scalar = matches!(ty, ManagedType::Primitive(PrimKind::R4 | PrimKind::R8));
        // Float aggregates ride consecutive v-registers ONLY on AAPCS64
        // (Vector3 → s0,s1,s2). x86-64 SysV packs 12-byte structs as two
        // eightbytes and MSVC passes them in memory, so the old GPR slotting
        // is no worse there — skip HFA binding and fall through to GPR.
        let is_hfa = !comps.is_empty() && fp_arm64;
        if is_fp_scalar {
            let w = match ty {
                ManagedType::Primitive(k) => *k,
                _ => PrimKind::R4,
            };
            if fp < 8 {
                let name = if fp_arm64 && w != PrimKind::R8 {
                    // Flattened lane-0 identity: scalar `sN` is lane 0 of `vN`
                    // and the lifter names lanes `vs{4*reg + lane}`. Naming
                    // the parameter `vs{fp}` (the ARGUMENT ORDINAL) never
                    // matched the register the callee actually read, so every
                    // float argument beyond s0 became `__abi`.
                    format!("vs{}", fp * 4)
                } else {
                    format!("{}{}", fp_prefix(w), fp)
                };
                out.push((name, pi, None));
            }
            fp = fp.saturating_add(1);
        } else if is_hfa {
            for (k, _name) in comps.iter().enumerate() {
                if fp + k < 8 {
                    // HFA component k is s{fp+k} = lane 0 of v{fp+k}.
                    out.push((format!("vs{}", (fp + k) * 4), pi, Some(k)));
                }
            }
            fp = fp.saturating_add(comps.len());
        } else {
            // GPR parameter (int/reference/pointer/opaque/unknown) — or the
            // stack spill area once x1..x7 (x0..x7 static) are exhausted.
            if gpr < 8 {
                out.push((format!("x{gpr}"), pi, None));
            } else {
                let si = (gpr - 8).min(4);
                out.push((
                    ["stack_-2", "stack_-3", "stack_-4", "stack_-5", "stack_-6"][si].to_string(),
                    pi,
                    None,
                ));
            }
            gpr = gpr.saturating_add(1);
        }
    }
    out
}

/// Canonical register-file key: any view of one physical register is ONE key.
///
/// `s3`, `d3`, `q3` and `v3.4s` are the same 128-bit register in different
/// widths, and AAPCS64 allocates an FP ARGUMENT by register NUMBER whatever
/// width the callee reads it at. Keying the argument lookup on the literal view
/// name would miss the match the moment a value written as `s0` is passed to a
/// `double` parameter (or the reverse) — i.e. exactly the case the mapping
/// exists to get right.
fn canon_reg(name: &str) -> String {
    let mut chars = name.chars();
    if chars.next() == Some('v') {
        let digits: String = name[1..]
            .chars()
            .skip(1)
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if !digits.is_empty() {
            return format!("v{digits}");
        }
    }
    name.to_string()
}

/// Put each call argument at the operand position its DECLARED SIGNATURE
/// requires.
///
/// The lifter records every `bl` as a REGISTER DUMP —
/// `[target, x0 result, x0…x7, fp-written…]` — in AAPCS64 register order, which
/// is NOT the callee's parameter order. Reading that dump positionally (what
/// `call_parts` did, and what `trim_call_arguments` truncates) is correct only
/// while every parameter is an integer or reference; the moment a signature
/// mixes register classes it hands a float parameter whatever the next INTEGER
/// register happened to hold, and — because the FP registers sit at the END of
/// the dump — truncation to the declared parameter count DELETES every FP
/// argument before the renderer ever sees it.
///
/// This pass rebuilds the operand list as
/// `[target, result, receiver?, param0, param1, …]`, resolving each parameter
/// by the register class its metadata type allocates (see
/// [`abi_param_registers`]). A parameter with no live register at this call site
/// becomes `MissingAbiArg` **in its own position** — which is now a truthful
/// statement about the ABI rather than an artifact of dump order.
///
/// `resolve` maps a raw call target address to its `MethodRef`; calls whose
/// target is not resolved are left untouched (there is no signature to align
/// against).
pub fn align_call_arguments(
    insns: &mut [Insn],
    table: &LocalTable,
    resolve: impl Fn(u64) -> Option<super::types::MethodRef>,
    fp_arm64: bool,
) -> usize {
    use super::ir::{Op, Operand as Op_};
    let mut aligned = 0usize;
    for insn in insns.iter_mut() {
        // Only the ARM64 `BL`/`BLR` shapes: `[target, x0 result, x0…x7, fp…]`.
        // The legacy arm32/x86 lifters emit different layouts and their
        // operands are positionally meaningful already.
        let has_result_slot = matches!(insn.op, Op::Call);
        if !matches!(insn.op, Op::Call | Op::CallVoid) {
            continue;
        }
        let (sig, resolved_method) = match insn.operands.first() {
            Some(Op_::Method(m)) => (m.sig.clone(), None),
            Some(Op_::Immediate(a)) => match resolve(*a as u64) {
                Some(m) => {
                    let sig = m.sig.clone();
                    (sig, Some(m))
                }
                None => continue,
            },
            _ => continue,
        };
        // Embed the resolved method so downstream passes (propagate_types)
        // can see the callee's parameter types without re-resolving.
        if let Some(m) = resolved_method {
            insn.operands[0] = Op_::Method(m);
            use std::sync::atomic::Ordering;
            super::timing::CALLSITE_EMBEDDED.fetch_add(1, Ordering::Relaxed);
        }
        let arg_start = if has_result_slot { 2 } else { 1 };
        // operand register-name → operand, over the whole dumped region
        let mut by_reg: HashMap<String, Op_> = HashMap::new();
        let dump: Vec<Op_> = insn.operands[arg_start.min(insn.operands.len())..].to_vec();
        // The dump is POSITIONAL in ABI register order: entry `i` holds the
        // value the caller passed in argument register `x{i}` (every lifter
        // dumps `x0…xN` first, then the FP registers, which carry their own
        // names). Keying the lookup on each operand's own `slot_origin` is
        // WRONG the moment SSA copy coalescing renames a value: at 0x73BE44 the
        // base call is `mov r0, r4 ; bl ChangeWeapon`, and the coalesced x0
        // slot therefore holds a version whose origin is `x4`. The receiver
        // lookup (`x0`) then MISSED — every receiver-with-a-rename rendered as
        // `__abi` — and the phantom `x4` key handed the real receiver to
        // whichever parameter allocates x4 (here `bool forceCreate`), producing
        // `__abi.ChangeWeapon(slotType, weapon, weaponTexture, this)`.
        // The ABI POSITION is the fact; the name of the version that carried
        // the value into that position is not.
        let int_entries = dump
            .iter()
            .take_while(|op| {
                op.as_local()
                    .and_then(|v| table.index_of.get(&v))
                    .is_some_and(|&li| {
                        let o = canon_reg(&table.infos[li].slot_origin);
                        o.len() > 1
                            && o.starts_with('x')
                            && o.as_bytes()[1..].iter().all(u8::is_ascii_digit)
                    })
            })
            .count();
        for (i, op) in dump.iter().enumerate() {
            let Some(v) = op.as_local() else { continue };
            let Some(&li) = table.index_of.get(&v) else {
                continue;
            };
            if i < int_entries {
                by_reg.insert(format!("x{i}"), op.clone());
            } else {
                by_reg.insert(canon_reg(&table.infos[li].slot_origin), op.clone());
            }
        }
        // The AAPCS64 indirect-result slot (x8) that the lifter attached when
        // the call has a hidden struct-return buffer. This rebuild is
        // positional by SIGNATURE, so it would drop it — but the renderer
        // reads it back from that fixed ARM64 slot index, so it must survive.
        // Only a slot the lifter actually attached is carried.
        //
        // Index 10 is the x8 slot ONLY when the lifter put x8 there. When it
        // recorded no x8, the FP argument registers start at that same index,
        // and reading it blind carried the FIRST FLOAT ARGUMENT as an
        // out-parameter — appending the same value a second time after the
        // signature's arguments (`set_pitch(x, x)`). The slot must prove its
        // origin is x8.
        let sret_slot = super::passes::SRET_OPERAND;
        let sret = insn
            .operands
            .get(sret_slot)
            .filter(|o| {
                !matches!(o, Op_::MissingAbiArg | Op_::Unknown)
                    && o.as_local()
                        .and_then(|v| table.index_of.get(&v))
                        .is_some_and(|&li| table.infos[li].slot_origin == "x8")
            })
            .cloned();
        // NOTE: `sret_shift` is 0 here — the CALLER-side shift depends on the
        // CALLEE's return type, which `MethodSigInfo` does not carry a size for.
        // A call to a value-returning method therefore still places the hidden
        // result buffer in the receiver slot; fixing that needs the callee's
        // return classification plumbed alongside its signature.
        let mut want = abi_param_registers(&sig.params, &[], sig.is_static, fp_arm64, 0);
        let mut new_ops: Vec<Op_> = insn.operands[..arg_start.min(insn.operands.len())].to_vec();
        // Placeholders are counted, not assumed: `(arg position, was the ABI
        // out of registers)` per miss. Committed only once the call passes the
        // unreadable-dump guard below, so a call we discard cannot inflate it.
        let mut miss_recv = false;
        let mut miss_params: Vec<(usize, bool)> = Vec::new();
        if !sig.is_static {
            match by_reg.get(&canon_reg("x0")) {
                Some(v) => new_ops.push(v.clone()),
                None => {
                    new_ops.push(Op_::MissingAbiArg);
                    miss_recv = true;
                }
            }
        }
        want.sort_by_key(|(_, pi, comp)| (*pi, *comp));
        for (reg, pi, _comp) in &want {
            match by_reg.get(&canon_reg(reg)) {
                Some(v) => new_ops.push(v.clone()),
                None => {
                    new_ops.push(Op_::MissingAbiArg);
                    // `stack_-N` is the ABI's own overflow slot; `x6`/`x7` are
                    // SysV/MSVC registers that do not exist (those args are
                    // passed in memory). Both mean "out of registers".
                    let beyond = reg.starts_with("stack_") || reg == "x6" || reg == "x7";
                    miss_params.push((*pi, beyond));
                }
            }
        }
        super::passes::place_sret(&mut new_ops, sret);
        // Nothing to say about a call whose dump we could not read at all.
        if by_reg.is_empty() {
            continue;
        }
        {
            use std::sync::atomic::Ordering;
            if miss_recv {
                super::timing::ABI_MISS_BY_POS[0].fetch_add(1, Ordering::Relaxed);
            }
            for (pi, beyond) in &miss_params {
                super::timing::ABI_MISS_BY_POS[(*pi + 1).min(10)].fetch_add(1, Ordering::Relaxed);
                if *beyond {
                    super::timing::ABI_MISS_STACK.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        if new_ops != insn.operands {
            insn.operands = new_ops;
            aligned += 1;
        }
    }
    aligned
}

/// The set of SSA versions whose value provably comes from SOMEWHERE.
///
/// A version is ROOTED when it is one of the ABI entry values the signature
/// seeds (`this`, a declared parameter, the indirect-result buffer), when a
/// real definition writes it (a load, arithmetic, call result, `Select`, …),
/// or when it is a `Move`/`Phi` whose SOURCES are themselves all rooted.
/// Everything else is a read of a register the method never wrote — the
/// register-dump filler the lifter records for every `bl`, which on x64 is
/// `rcx/rdx/r8/r9` (MSVC) and on arm64 `x0`–`x8`, none of which the callee's
/// own signature has to name.
///
/// The TRANSITIVE part is the whole point, and it is why "does this version
/// have a defining instruction" is the wrong test. `locals::create` seeds only
/// the ENTRY version of each register; a later version of the same register is
/// merely *defined*, so a pass that only asked for "any def" accepted it. That
/// is exactly the redundant-join-phi shape SSA leaves at a merge of two paths
/// that both read the same undeclared register (measured at `0x14eb20`,
/// `TMP_SpriteAsset.GetSpriteIndexFromHashcode`):
///
/// ```text
///   v11 = phi(v9, v9)      ; v9 = entry x2, a register this method never writes
///   v14 = phi(v11, v11)    ; every link "has a def", the value is still garbage
///   bl fn_11c930(…, v14, …)
/// ```
///
/// `ssa::eliminate` lowers the chain to `v14 = v9`, which reads an address
/// nothing defines and renders as a phantom `in_x2` argument. Following
/// `Move`/`Phi` sources back to their root makes the chain provably undefined
/// again, so the scrub below can drop it. A cycle (a loop phi that depends on
/// itself) never roots, which is correct: nothing in the loop ever defines it.
fn rooted_versions(insns: &[Insn], table: &LocalTable) -> std::collections::HashSet<u32> {
    use super::ir::{Op, Operand as Op_};
    let mut rooted: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // 1. Seeded entry values have no defining instruction by construction.
    for (v, idx) in &table.index_of {
        let li = &table.infos[*idx];
        if li.is_this || li.is_param.is_some() || li.is_return_buffer || li.is_incoming_arg {
            rooted.insert(*v);
        }
    }
    // The STACK POINTER and FRAME POINTER are defined at entry: `mov r5, sp`
    // reads a value the ABI always provides. Not rooting them left every
    // stack-local ADDRESS — a `ref`/`out` argument (`mov r1, sp`, the
    // `&local` of `GetRateOverTime(ref curve)`), a `sp`-based store — as ABI
    // garbage, so the scrub replaced the argument with `__abi` and the
    // renderer printed a phantom.
    //
    // Root only the ENTRY version per slot (SSA allocates versions 0-first
    // per slot); a later `sub sp, sp, #n` is a real definition and roots
    // through step 2 on its own. `fp` is included because ARM32 frames address
    // incoming stack arguments as `[fp, #n]`.
    //
    // ROOTING THEM ALONE IS NOT ENOUGH, and that is exactly why this was tried
    // once and reverted: it made the newly-KEPT `v0(sp)` reads visible to the
    // verifier's undefined-local oracle, which then flagged every one (+111 on
    // the 4,000-method ARM32 sample). The oracle must ALSO know that `sp` is a
    // seeded entry value; the two halves land together — see the `sp`/`fp`
    // exemption in `verify.rs`. (`stack_` cells, the x86 stack-argument rebase,
    // are already exempt for the same reason.)
    for slot in ["sp", "fp"] {
        if let Some(v) = table
            .index_of
            .iter()
            .filter(|(_, idx)| table.infos[**idx].slot_origin == slot)
            .map(|(&v, _)| v)
            .min()
        {
            rooted.insert(v);
        }
    }
    // 2. Split definitions into "real" (rooted immediately) and pure
    //    PROPAGATION (`Move`/`Phi` reading only locals — rootedness inherits).
    let mut inherit: Vec<(u32, Vec<u32>)> = Vec::new();
    for insn in insns {
        let Some(pos) = Insn::dest_position(&insn.op) else {
            continue;
        };
        let Some(Op_::Local(d)) = insn.operands.get(pos) else {
            continue;
        };
        match insn.op {
            Op::Move | Op::Phi => {
                let srcs: Vec<u32> = insn
                    .operands
                    .iter()
                    .skip(pos + 1)
                    .filter_map(|o| o.as_local())
                    .collect();
                // No local source (immediate, memory read, field, `AddressOf`)
                // is a real datum — taking an address or loading is a value.
                if srcs.is_empty() {
                    rooted.insert(*d);
                } else {
                    inherit.push((*d, srcs));
                }
            }
            _ => {
                rooted.insert(*d);
            }
        }
    }
    // 3. Fixpoint. Every source must be rooted for a propagation node to be;
    //    a node that can never satisfy that stays undefined.
    let mut progress = true;
    while progress {
        progress = false;
        inherit.retain(|(d, srcs)| {
            if srcs.iter().all(|s| rooted.contains(s)) {
                rooted.insert(*d);
                progress = true;
                false
            } else {
                true
            }
        });
    }
    rooted
}

/// Post-SSA ABI-garbage cleanup (`scrub_abi_call_args`).
///
/// The lifter records every `bl` with the FULL register dump as call operands
/// (AAPCS64/x64 argument registers), regardless of what the callee actually
/// declares. For RESOLVED calls, `trim_call_arguments` + the positional
/// renderer strip the excess. For UNRESOLVED calls (`Expr::RawCall` →
/// `fn_4a58bac(...)`), every operand renders — so the output shows phantom
/// arguments that are really just register-dump filler for values the method
/// never declared. Worse, those reads create SSA entry versions with no def
/// and no entry mark, which render as `v41 = v10;` with `v10` never written —
/// the single biggest corruption class in the dump.
///
/// Fix: for each call-family op, any arg operand that is a local which is NOT
/// [`rooted_versions`]-rooted is ABI garbage → rewrite to
/// `Operand::MissingAbiArg`, which every render path drops positionally.
/// Rooted operands include every declared parameter, which is what keeps real
/// arguments (e.g. the receiver `this._challengesModel` passed to SetAccessed)
/// intact. Runs AFTER `locals::create` (needs `index_of`) and BEFORE
/// `propagate_types` so the scrubbed operands never pick up types.
pub fn scrub_abi_call_args(insns: &mut [Insn], table: &LocalTable) -> usize {
    use super::ir::{Op, Operand as Op_};
    let rooted = rooted_versions(insns, table);
    let mut scrubbed = 0usize;
    for insn in insns.iter_mut() {
        match insn.op {
            Op::Call | Op::CallVoid | Op::IndirectCall | Op::NewObj(_) => {}
            _ => continue,
        }
        // arg region: [target, dest/ret] prefix skipped (matches render paths:
        // Call args start at operand 2; IndirectCall has no target slot).
        let arg_start = match insn.op {
            Op::IndirectCall => 1,
            _ => 2,
        };
        for op in insn.operands.iter_mut().skip(arg_start) {
            if let Op_::Local(v) = op {
                if !rooted.contains(v) {
                    *op = Op_::MissingAbiArg;
                    scrubbed += 1;
                }
            }
        }
        // The TRAILING run of argument registers the declared signature does
        // NOT cover is ABI register-dump filler, not arguments — and for an
        // UNRESOLVED target (`fn_<addr>`, `fnptr_…`, an indirect call) there is
        // no arity to trim it against, so `trim_call_arguments` cannot remove
        // it and the renderer printed phantom `in_x1…in_x3` arguments. Measured
        // on the ARM32 corpus that was 250k+ occurrences across 57% of methods,
        // dominated by `in_x3` (153k) passed to the class-init/runtime helpers —
        // the single largest readability defect in the dump.
        //
        // `is_incoming_arg` means "a register the ABI hands a value in but the
        // signature does not declare", so it is not proof the caller computed
        // an argument: an undeclared register reaches a call only as the same
        // leftover it arrived in. Scrubbing the TRAILING run to
        // `MissingAbiArg` lets the existing trailing trim delete it, while an
        // INTERIOR one — a genuine hole BETWEEN two real arguments — is left
        // untouched. A declared parameter or any computed value is rooted and
        // therefore never scrubbed here.
        if !matches!(insn.operands.first(), Some(Op_::Method(_))) {
            // Walk the trailing run from the end. A hole the loop above already
            // made is stepped over rather than stopping the walk: the filler
            // arrives as a contiguous run of registers, so `(arg, in_x2,
            // in_x3)` must lose BOTH tails, not just the last one.
            let mut end = insn.operands.len();
            while end > 0 {
                match &insn.operands[end - 1] {
                    Op_::Local(v)
                        if table
                            .index_of
                            .get(v)
                            .is_some_and(|&i| table.infos[i].is_incoming_arg) =>
                    {
                        insn.operands[end - 1] = Op_::MissingAbiArg;
                        scrubbed += 1;
                        end -= 1;
                    }
                    Op_::MissingAbiArg => end -= 1,
                    _ => break,
                }
            }
        }
    }
    scrubbed
}

/// Pre-SSA ABI pin-copy rewriting (`rewrite_pinned_copies`).
///
/// IL2CPP's ARM64 codegen pins the receiver and hot parameters in
/// callee-saved registers (x19–x28) plus the x8 indirect-result scratch:
/// the prologue does `MOV x20, x0` and every `this.field` load afterwards
/// reads `[x20 + off]`. Only x0 was seeded as `this`, so field resolution
/// never fired on the pinned copies and the renderer emitted
/// `*(nint*)(v0 + 192)` instead of `this.<field>`.
///
/// Runs right after lifting, BEFORE CFG/stack analysis, while operands are
/// still register names. Two linear passes:
///
/// 1. DISCOVER: track `MOV regA, regB` where regB currently holds an ABI
///    identity (receiver x0, parameter xN). Associations die when the
///    register is redefined by anything else; caller-saved registers die at
///    calls (callee-saved survive by ABI contract — exactly why IL2CPP
///    pins locals there). Also canonicalize `MOV dst, pin` back to
///    `MOV dst, identity` so chained pins collapse.
///
/// 2. REWRITE: while walking again with the same rules, every USE of a
///    pinned register becomes a USE of the identity register: Memory bases
///    `[x20 + 192]` turn into `[x0 + 192]`, and Compare/Call operands read
///    the receiver directly. Definitions of the pinned register itself are
///    rewritten to define the identity register ONLY for the initial pin
///    (`MOV x20, x0` becomes a no-op — the value already lives in x0);
///    later redefinitions of the pin kill the association instead, because
///    at that point the register holds something else and renaming would
///    be wrong.
///
/// The stack analyzer then spills/recovers normally (x0 is caller-saved, so
/// its spill handling is unchanged), and locals::create seeds the x0 entry
/// version as `this` — which is exactly what the field oracle needs.
pub fn rewrite_pinned_copies(insns: &mut [Insn], seed: &SeedInfo) {
    if insns.is_empty() {
        return;
    }
    let abi_arg_regs = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];
    let caller_saved: &[&str] = &[
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
        "x14", "x15", "x16", "x17",
    ];
    let this_off = usize::from(!seed.is_static);

    // identity of a register: Some(reg) where reg is "x0" (receiver) or a
    // parameter register currently holding that param's incoming value.
    let identity_of_src = |s: &str, holds: &HashMap<String, String>| -> Option<String> {
        // The receiver identity is LIVE, not eternal.
        //
        // This used to `return Some("x0")` unconditionally, asserting that x0
        // holds `this` for the whole body. It does not: x0 is also the
        // RETURN-VALUE register, so from the first call onward it holds
        // something else. Every post-call `MOV xd, x0` was therefore read as
        // "xd is a pinned copy of the receiver", and the next call's argument
        // register got rewritten to x0 — which had just been overwritten. Two
        // distinct arguments then read the SAME register and rendered
        // identically: `Type.op_Equality(x, x)`,
        // `x.GetType() == x.GetType()` (always true), and wrong call argument
        // registers generally.
        //
        // The real ARM64 this comes from (FindMatchingElement @0x1F4F660):
        //   0x01f4f7d0  bl  <callee>          ; x0 := result
        //   0x01f4f7d4  mov x1, x0            ; arg1 := THAT RESULT
        //   0x01f4f7d8  mov x0, x23           ; arg0 := receiver
        //   0x01f4f7e0  bl  <callee>          ; arg0 and arg1 must stay distinct
        //
        // Availability is tracked in the same map as every other identity, so
        // the existing clobber handling (redefinition / call) retires it
        // automatically.
        if s == "x0" && !seed.is_static {
            return holds.get("x0").cloned();
        }
        if let Some(pos) = abi_arg_regs.iter().position(|r| *r == s) {
            if pos >= this_off && pos - this_off < seed.param_types.len() {
                return Some(s.to_string());
            }
            return None;
        }
        holds.get(s).cloned()
    };

    // ---- pass 1: discover pins (no mutation) --------------------------------
    let mut holds: HashMap<String, String> = HashMap::new();
    // The receiver really does live in x0 at entry (x0 is the ABI `this`
    // slot); every redefinition of x0 and every call retires it below.
    if !seed.is_static {
        holds.insert("x0".to_string(), "x0".to_string());
    }
    // (insn idx, pinned reg, original src reg) — the src is kept so the pin
    // copy can be RESTORED as a real `MOV x20, x0` if its identity register
    // gets clobbered by a call before the pin's last use (canonicalizing the
    // use to the clobbered identity would alias the wrong value).
    let mut pins: Vec<(usize, String, String)> = Vec::new();
    for (i, insn) in insns.iter().enumerate() {
        let dst_reg = match insn.dest() {
            Some(Operand::Register(r)) => Some(r.clone()),
            _ => None,
        };
        let src_reg = if insn.op == Op::Move {
            match insn.operands.get(1) {
                Some(Operand::Register(r)) => Some(r.clone()),
                _ => None,
            }
        } else {
            None
        };
        if let (Some(d), Some(s)) = (&dst_reg, &src_reg) {
            if let Some(id) = identity_of_src(s, &holds) {
                holds.insert(d.clone(), id);
                if d != s {
                    pins.push((i, d.clone(), s.clone()));
                }
                continue;
            }
        }
        if let Some(d) = &dst_reg {
            holds.remove(d);
        }
        if matches!(
            insn.op,
            Op::Call | Op::IndirectCall | Op::NewObj(_) | Op::NewArr(_)
        ) {
            for r in caller_saved {
                holds.remove(*r);
            }
        }
    }
    if std::env::var("ILR_ABICOPIES").is_ok() {
        eprintln!(
            "[abicopy] rva={:#x} pins: {:?}",
            insns.first().map(|i| i.address).unwrap_or(0),
            pins.iter()
                .map(|(i, r, _)| (i, r.as_str()))
                .collect::<Vec<_>>()
        );
    }
    if pins.is_empty() {
        return;
    }

    // ---- pass 2: rewrite uses while re-simulating ---------------------------
    let mut rename: HashMap<String, String> = HashMap::new();
    // Same live-from-entry receiver fact as pass 1, so the pin-site handler
    // and the association update resolve `MOV pin, x0` while x0 really is
    // `this`. Dies at the first call / x0 redefinition, exactly as in pass 1.
    if !seed.is_static {
        rename.insert("x0".to_string(), "x0".to_string());
    }
    let mut pin_at: HashMap<usize, String> = HashMap::new();
    // pin SITE index -> the source register THAT SITE copied from.
    //
    // This used to be a `HashMap<pin reg, src>` ("the source of register R"),
    // which is the wrong key: IL2CPP pins a register MANY times with DIFFERENT
    // sources — the prologue `MOV x23, x7` (a parameter), then a later
    // `MOV x23, x0` (the receiver, re-established across a call). Keyed by
    // register, each insert overwrote the previous one, so every restore wrote
    // the LAST pin's source into the FIRST pin's site: the prologue copy of a
    // boolean parameter came back as a copy of `this`.
    //
    // That single wrong value is the origin of the whole "value identity"
    // family: a parameter rendering as `(this & 1) == 0`, two distinct
    // parameters comparing equal (`x.GetType() == x.GetType()`), wrong call
    // argument registers, and lost out-parameter values. The site selection
    // below was made per-site and deterministic earlier; the source has to be
    // per-site for the same reason.
    let mut pin_src_at: HashMap<usize, String> = HashMap::new();
    // (insn idx, pin reg, src reg) — pin copies to restore as real moves
    // after the loop (the pin site always precedes its clobber, so these
    // indices were already visited by the iter_mut borrow).
    let mut restores: Vec<(usize, String, String)> = Vec::new();
    for (i, p, s) in pins {
        pin_at.insert(i, p.clone());
        pin_src_at.insert(i, s);
    }
    // Pin site indices per register, in listing order.
    //
    // A register is pinned MORE THAN ONCE whenever IL2CPP re-establishes the
    // receiver after a call (`MOV x19, x0` in the prologue, then again later
    // in the body), so "which site do I restore" is a real question with a
    // real answer — and the old code answered it by walking `pin_at` (a
    // HashMap) with `.find()`, i.e. it returned an ARBITRARY site. HashMap
    // iteration order is randomised per process, so the same method
    // decompiled to DIFFERENT code on different runs: the restored
    // `MOV pin, src` jumped between the prologue pin and the later re-pin,
    // which reordered the listing, which changed SSA version ids, which
    // changed type propagation (a local typed `XmlSchemaInference` one run
    // and `object` the next), which changed whether a field resolved — e.g.
    // `*(nint*)(this.klass + 302)` becoming the fabricated static-field name
    // `XmlSchemaInference.ST_boolean`. Non-reproducible output also made
    // every before/after measurement of a fix half noise.
    let mut pin_sites: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, pin) in pin_at.iter() {
        pin_sites.entry(pin.clone()).or_default().push(*idx);
    }
    for sites in pin_sites.values_mut() {
        sites.sort_unstable();
    }
    for (i, insn) in insns.iter_mut().enumerate() {
        // the pin site itself: `MOV x20, x0` — canonicalize dst to x0 so the
        // move degenerates into a self-move and gets dropped.
        if let Some(pin_reg) = pin_at.get(&i).cloned() {
            // The pin's SOURCE identity: `MOV x20, x0` — canonicalize the
            // destination to the identity register so the move degenerates
            // into a self-move and is dropped, then record the association
            // for every later use of the pin.
            let src_identity = match insn.operands.get(1) {
                Some(Operand::Register(r)) => identity_of_src(r, &rename),
                _ => None,
            };
            if let Some(id) = src_identity {
                if let Some(d) = insn.dest_mut() {
                    if let Operand::Register(r) = d {
                        *r = id.clone();
                    }
                }
                if pin_reg != id {
                    rename.insert(pin_reg, id);
                }
            }
            let is_self = matches!(
                (insn.dest(), insn.operands.get(1)),
                (Some(Operand::Register(a)), Some(Operand::Register(b))) if a == b
            );
            if is_self {
                super::verify::ledger_note(insn.address, super::verify::LedgerKind::PinCopyFold);
                insn.op = Op::Nop;
                insn.operands.clear();
            }
            continue;
        }

        // rewrite USES of pinned registers to their identity.
        //
        // An instruction that DEFINES a pin's identity must not canonicalise
        // that pin within ITSELF. `ldr x1, [x19], #8` with `x19` pinned to the
        // argument `x1` both READS x1 (as the base, through the pin) and
        // OVERWRITES it (the loaded value) — rewriting the base to x1 made the
        // load read its own destination slot, and the post-index write-back
        // then advanced the just-loaded VALUE instead of the pointer, so an
        // element-copy loop never walked its list. Calls keep the plain
        // rewrite: every call argument is read BEFORE the call runs, so
        // canonicalising one to the live identity is correct there.
        let callish = matches!(
            insn.op,
            Op::Call | Op::IndirectCall | Op::NewObj(_) | Op::NewArr(_)
        );
        let self_identity = if callish {
            None
        } else {
            insn.dest().and_then(|d| match d {
                Operand::Register(r) => Some(r.clone()),
                _ => None,
            })
        };
        match self_identity {
            Some(id) if rename.values().any(|v| *v == id) => {
                let filtered: HashMap<String, String> = rename
                    .iter()
                    .filter(|(_, v)| **v != id)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                rewrite_reg_operands(insn, &filtered);
            }
            _ => rewrite_reg_operands(insn, &rename),
        }

        // Invalidate pin associations whose IDENTITY register this insn
        // defines/clobbers: canonicalizing a later pin use to a redefined
        // register would alias the NEW value (e.g. the load `[x20,#0x10]`
        // after `BL thunk` clobbers x0 — the pin x20 still holds `this`,
        // but rewriting it to the x0-phi would read the call's result).
        // Restore the pin copy as a real move so the pin register keeps a
        // definition and SSA tracks the correct value.
        let mut clobbered: Vec<String> = Vec::new();
        if matches!(
            insn.op,
            Op::Call | Op::IndirectCall | Op::NewObj(_) | Op::NewArr(_)
        ) {
            clobbered.extend(caller_saved.iter().map(|r| r.to_string()));
        } else if let Some(d) = insn.dest() {
            if let Operand::Register(r) = d {
                clobbered.push(r.clone());
            }
        }
        let mut dead_pins: Vec<String> = rename
            .iter()
            .filter(|(_, id)| clobbered.iter().any(|c| c == *id))
            .map(|(pin, _)| pin.clone())
            .collect();
        // Deterministic death order: `rename` is a HashMap, so its iteration
        // order is per-process random and would otherwise leak into the order
        // of `restores`.
        dead_pins.sort();
        for pin in dead_pins {
            rename.remove(&pin);
            // Restore the pin site whose VALUE is live at this clobber: the
            // latest site strictly before the current instruction (earliest
            // as a fallback when the clobber precedes every site). Picking
            // the earliest site unconditionally would re-materialise
            // `MOV pin, <identity>` with the identity's STALE value — the
            // source register has usually been reassigned between the
            // prologue pin and a later re-pin.
            let site = pin_sites.get(&pin).and_then(|sites| {
                sites
                    .iter()
                    .rev()
                    .find(|&&s| s < i)
                    .copied()
                    .or_else(|| sites.first().copied())
            });
            if let (Some(idx), Some(src)) = (site, site.and_then(|x| pin_src_at.get(&x))) {
                // restore `MOV pin, src` so the pin register has a real def
                // (applied after the loop — the pin site precedes this call).
                // `src` is looked up by SITE INDEX, so the restore reproduces
                // the copy that site actually performed.
                restores.push((idx, pin.clone(), src.clone()));
            }
        }

        // update the association state
        let dst_reg = match insn.dest() {
            Some(Operand::Register(r)) => Some(r.clone()),
            _ => None,
        };
        let src_reg = if insn.op == Op::Move {
            match insn.operands.get(1) {
                Some(Operand::Register(r)) => Some(r.clone()),
                _ => None,
            }
        } else {
            None
        };
        if let (Some(d), Some(s)) = (&dst_reg, &src_reg) {
            if let Some(id) = identity_of_src(s, &rename) {
                if d != &id {
                    rename.insert(d.clone(), id);
                    continue;
                }
            }
        }
        if let Some(d) = &dst_reg {
            rename.remove(d);
        }
        if matches!(
            insn.op,
            Op::Call | Op::IndirectCall | Op::NewObj(_) | Op::NewArr(_)
        ) {
            for r in caller_saved {
                rename.remove(*r);
            }
        }
    }
    for (idx, pin, src) in restores {
        insns[idx].op = Op::Move;
        insns[idx].operands = vec![Operand::Register(pin), Operand::Register(src)];
    }
}

/// Rename register operands (including Memory bases) per the association map.
fn rewrite_reg_operands(insn: &mut Insn, rename: &HashMap<String, String>) {
    if rename.is_empty() {
        return;
    }
    fn walk(op: &mut Operand, rename: &HashMap<String, String>) {
        match op {
            Operand::Register(n) => {
                if let Some(rep) = rename.get(n) {
                    *op = Operand::Register(rep.clone());
                }
            }
            Operand::Memory { base, index, .. } => {
                walk(base, rename);
                if let Some(i) = index {
                    walk(i, rename);
                }
            }
            Operand::AddressOf(inner) | Operand::ArrayLength(inner) => walk(inner, rename),
            Operand::ArrayAccess { array, index } => {
                walk(array, rename);
                walk(index, rename);
            }
            _ => {}
        }
    }
    let dest_first = matches!(
        insn.op,
        Op::Move
            | Op::Arith(_)
            | Op::FloatArith(_)
            | Op::Compare(_)
            | Op::Not
            | Op::Negate
            | Op::NewObj(_)
            | Op::NewArr(_)
            | Op::Box(_)
            | Op::LoadLength
            | Op::Select(_)
    );
    for (i, op) in insn.operands.iter_mut().enumerate() {
        // Only a PURE register destination is a def. A MEMORY destination is
        // a STORE (`STR x1, [x19, #8]` → `Move [x19+8], x1`): its base and
        // index registers are USES of the pin and MUST be rewritten. Skipping
        // operand#0 wholesale left every store through the pinned receiver
        // (`x19`) reading an erased register — the `*(nint*)(__unknown + off)`
        // sprawl in the rendered output.
        let pure_def = dest_first && i == 0 && matches!(op, Operand::Register(_));
        if pure_def {
            continue;
        }
        if insn.op == Op::Call && i == 1 {
            continue;
        }
        walk(op, rename);
    }
}

/// Outcome counters for guarded field-address seeding (see
/// [`arith_field_address_seed`]) — so "the guard did nothing" is numbers,
/// not an assumption. Process-global atomics, read-only w.r.t. the IR: each
/// increment precedes no early-return and changes no `Insn`, so the emitted
/// C# is byte-identical with or without the counting.
static FIELD_ADDR_SEEDS_APPLIED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Reference-typed (or enum) field: the address stayed the integer guess.
/// This is the pre-guard damage shape firing as a refusal instead.
static FIELD_ADDR_REFUSED_REF_ENUM: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Field type with no struct layout to resolve through (primitive — a value
/// type with no subfields — `IntPtr`, `ByRef`/`Pointer`, `Opaque`, `Unknown`,
/// or a `Class` the oracle cannot confirm as a struct).
static FIELD_ADDR_REFUSED_NO_LAYOUT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// No proven owner: untyped base, dynamic (non-constant) offset, or no field
/// at `(owner, offset)`.
static FIELD_ADDR_REFUSED_UNPROVEN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `(applied, refused_ref_or_enum, refused_no_layout, refused_unproven)` — see
/// the statics above.
pub fn field_addr_seed_counts() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering;
    (
        FIELD_ADDR_SEEDS_APPLIED.load(Ordering::Relaxed),
        FIELD_ADDR_REFUSED_REF_ENUM.load(Ordering::Relaxed),
        FIELD_ADDR_REFUSED_NO_LAYOUT.load(Ordering::Relaxed),
        FIELD_ADDR_REFUSED_UNPROVEN.load(Ordering::Relaxed),
    )
}

/// Guarded field-address seeding for `Arith(Add, base, const)`.
///
/// `dst = base + off` is the field's ADDRESS, never its value. The unguarded
/// version of this (seed `dst` with the field's own type for every class-typed
/// base) was tried and reverted: a reference-typed field's address mistyped as
/// its class makes the next `[dst + off2]` resolve as a subfield of the wrong
/// owner, snowballing through the fixpoint (425k of 1,030,800 dump.cs lines).
///
/// Returns the `ByRef` seed — still an address — when ALL of these hold, else
/// `None` and the caller keeps the generic integer guess:
///
/// * PROVEN OWNER: `base` is a `Local` whose type is `Class` with `tdi >= 0`,
///   or `ByRef(Class)` from an earlier seed (chained struct addressing: `&S`
///   plus the offset of a nested struct field seeds `&(nested)`). Raw
///   `Pointer` bases are refused — pointer arithmetic strides, it does not
///   name fields — as is anything but a `Local` (a `Field`/`Memory`/`Immediate`
///   base is a value or an unversioned constant, not a proven address).
/// * `this`-ALIASES ARE EXCLUDED: the `Arith(Add, <this-alias>, #k)` sweep
///   below already owns those (rewriting them to `Move(dst, Field)`), so
///   seeding here too would only churn their rendering for no new name.
/// * CONSTANT SLOT naming a real non-static instance field at `(owner, off)`.
///   Dynamic offsets and untyped bases refuse (pinned by the refusal tests at
///   the bottom of this file).
/// * STRUCT-VALUED FIELD: the field's type is `Class` with `tdi >= 0` AND the
///   oracle confirms `is_struct_value_type` (value type, not enum). A
///   reference-typed (or enum) field's address stays the integer guess —
///   never its class. Primitives refuse despite being value types: an `int`
///   field has no subfields, so seeding would add `ByRef` noise without ever
///   naming anything (`metadata_free_value_type_ness` documents the split).
///
/// The seed is monotone: it fires deterministically from (owner, offset,
/// field), and the caller only overwrites `Unknown` or the generic `I8` guess,
/// never a real type — so a later iteration reproduces the same seed and the
/// fixpoint still converges.
fn arith_field_address_seed(
    insn: &Insn,
    table: &LocalTable,
    oracle: &dyn TypeOracle,
) -> Option<ManagedType> {
    use std::sync::atomic::Ordering;
    let refuse = |which: &std::sync::atomic::AtomicU64| {
        which.fetch_add(1, Ordering::Relaxed);
        None
    };
    let base = insn.operands.get(1)?.clone();
    let Operand::Immediate(off) = insn.operands.get(2)? else {
        return refuse(&FIELD_ADDR_REFUSED_UNPROVEN);
    };
    // Only a versioned local carries a proven type; anything else (register,
    // immediate, memory, field value) has no owner to key on.
    let Operand::Local(bv) = base else {
        return refuse(&FIELD_ADDR_REFUSED_UNPROVEN);
    };
    let base_ty = table
        .index_of
        .get(&bv)
        .map(|&i| table.infos[i].ty.clone())?;
    // The owner: `Class` directly, or the struct inside one level of `ByRef`
    // (an address an earlier seed produced). `Pointer` is deliberately NOT
    // peeled — see above.
    let owner_tdi = match &base_ty {
        ManagedType::Class(nt) => nt.type_def_index,
        ManagedType::ByRef(inner) => match inner.as_ref() {
            ManagedType::Class(nt) => nt.type_def_index,
            _ => return refuse(&FIELD_ADDR_REFUSED_UNPROVEN),
        },
        _ => return refuse(&FIELD_ADDR_REFUSED_UNPROVEN),
    };
    if owner_tdi < 0 {
        return refuse(&FIELD_ADDR_REFUSED_UNPROVEN);
    }
    let Some(field) = oracle
        .field_at_offset(owner_tdi, *off)
        .filter(|f| !f.is_static)
    else {
        return refuse(&FIELD_ADDR_REFUSED_UNPROVEN);
    };
    // Value-type-ness gate: only a struct-typed field's address may seed.
    match field.field_type.metadata_free_value_type_ness() {
        // Always-reference kinds: the address must stay a pointer.
        Some(false) => refuse(&FIELD_ADDR_REFUSED_REF_ENUM),
        // Primitives / IntPtr: value types with no subfields — nothing would
        // ever resolve through the seed, so it would only be noise.
        Some(true) => refuse(&FIELD_ADDR_REFUSED_NO_LAYOUT),
        None => match &field.field_type {
            ManagedType::Class(nt) if nt.type_def_index >= 0 => {
                match oracle.is_struct_value_type(nt.type_def_index) {
                    // The seed stays an ADDRESS: `ByRef(struct)`, never the
                    // struct itself. `class_of` peels exactly this for a later
                    // `[dst + sub]` subfield load/store — the only consumer.
                    Some(true) => {
                        FIELD_ADDR_SEEDS_APPLIED.fetch_add(1, Ordering::Relaxed);
                        Some(ManagedType::ByRef(Box::new(field.field_type.clone())))
                    }
                    // A reference class — or an enum, whose only field is the
                    // compiler-named `value__` with no subfield to resolve.
                    Some(false) => refuse(&FIELD_ADDR_REFUSED_REF_ENUM),
                    // The oracle cannot tell: refuse, never guess.
                    None => refuse(&FIELD_ADDR_REFUSED_NO_LAYOUT),
                }
            }
            // `ByRef`/`Pointer`/`Opaque`/`Unknown` field type: no struct
            // layout to resolve a subfield against.
            _ => refuse(&FIELD_ADDR_REFUSED_NO_LAYOUT),
        },
    }
}

/// Monotone type-propagation fixpoint over the renamed listing.
///
/// Rules (each only ever *refines* Unknown → concrete, never downgrades):
/// - Move(local, TypedOperand): copy operand type
/// - Arith on ints → I8-ish; Compare → Boolean; FloatArith → R4/R8 by hint
/// - NewObj(t) → Class(t); NewArr(e) → SzArray(e); Box(t) → Class(t)
/// - Call: dest gets sig.ret
/// - Memory loads typed by field lookup when base has a known class type
///
/// Where `Field` refs enter the IR (measured, not assumed): the three lifters
/// (`lift.rs`, `lift_arm32.rs`, `lift_x86.rs`) emit ZERO `Operand::Field` /
/// `InstanceField` / `StringLiteral` — they have no metadata oracle, only
/// `Memory`/`Immediate`/registers. Every field name enters downstream: the
/// instance load/store sweeps below (`[base + off]` off a class-typed base →
/// `Field`/`InstanceField`), the `Arith(Add, this-alias, #k)` address sweep,
/// the absolute-static sweep (`static_field_at`), and — outside this file —
/// `resolve::resolve_static_fields` / `resolve_string_accessors` (FieldInfo /
/// literal usages). A type that lives ONLY in `operand_type`'s `Field` arm is
/// therefore unreachable for any value whose field was never named, which is
/// exactly the P0 interface-dispatch residue: the receiver's klass base is
/// runtime-computed (`v60 = v59 + 312`, dynamic index, no constant slot), so
/// no sweep names it and `klass_tdi_of` never reaches a table.
///
/// What this fixpoint deliberately does NOT do: type `Arith(Add, base, const)`
/// from the field at `(base, const)` for DERIVED (non-`this`) bases — EXCEPT
/// through the guarded seed in the `Arith` arm (see
/// [`arith_field_address_seed`]). The unguarded version of that (seed `dst`
/// with the field's own type for every class-typed base) was tried and
/// reverted: `dst = base + off` is the field's ADDRESS, and without
/// value-type-ness in the oracle a reference-typed field's address mistyped as
/// the field's own class makes the next `[dst + off2]` resolve as a subfield
/// of the wrong owner (measured: 425k of 1,030,800 dump.cs lines changed, with
/// stores rewritten to `// store to TTT.+0x20`-style comments and `(raw +
/// 16).value__` misnames — wrong values, the project's worst outcome). The
/// guard (proven `Class`/`ByRef` owner, constant slot, struct-valued field via
/// `TypeOracle::is_struct_value_type`, seed stays `ByRef`) re-attempts exactly
/// the sound sliver: gate-4000 measures 55 applied visits against 34 + 20 +
/// 50,639 refusals, 3,834 changed dump.cs lines all audited as corrections
/// (raw-collision misnames like `this.m_GradientMax = value.b` and
/// `this.<>4__this` sprawl become ground-truth names or honest gaps; the only
/// worse lines are 2 wrong-owner comments via a highvar merge, documented at
/// the seed helper), and `ILR_VERIFY` moves nowhere bad (undefined locals
/// 3,561 → 3,557). The `this`-alias address sweep below stays (its receiver
/// and owner are certain); every other `Add` keeps the generic integer guess,
/// and the refusal tests at the bottom of this file pin that. Raising
/// interface attempts needs the slot, which for these sites is dynamic — a
/// resolver-side (loop-pattern) change, never a guessed type here.
pub fn propagate_types(
    insns: &mut [Insn],
    table: &mut LocalTable,
    oracle: &dyn TypeOracle,
) -> usize {
    // copy-alias map for `this`-chasing in the memory→field rewrite: the
    // x0-entry local carries `is_this`, but IL2CPP pins the receiver in a
    // callee-saved register (`MOV x19, x0`) and derefs off the pin. Those
    // pin locals are ordinary (not flagged this), so a field rewrite gated
    // purely on `is_this` would stop firing for every event accessor and
    // other pin-`this` methods. Chase `Move v, Local(src)` copies (≤ 8
    // hops) to the flagged entry local. Loop-carried receivers are Phis of
    // this-aliases (the CAS retry loop re-pins x0 every iteration), so Phis
    // whose EVERY input aliases `this` also qualify. Ambiguous chains bail.
    let mut copy_src: HashMap<u32, u32> = HashMap::new();
    let mut phi_inputs: HashMap<u32, Vec<u32>> = HashMap::new();
    for insn in insns.iter() {
        match &insn.op {
            Op::Move => {
                let Some(dst) = insn.dest().and_then(|o| o.as_local()) else {
                    continue;
                };
                let Some(Operand::Local(src)) = insn.operands.get(1) else {
                    continue;
                };
                if copy_src.insert(dst, *src).is_some() {
                    copy_src.insert(dst, u32::MAX); // multi-def → ambiguous
                }
            }
            Op::Phi => {
                let Some(dst) = insn.dest().and_then(|o| o.as_local()) else {
                    continue;
                };
                let ins: Vec<u32> = insn
                    .operands
                    .iter()
                    .skip(1)
                    .filter_map(|o| o.as_local())
                    .collect();
                if !ins.is_empty() {
                    phi_inputs.insert(dst, ins);
                }
            }
            _ => {}
        }
    }
    // Absolute-address definitions, for the static-storage naming below. A slot
    // whose ONLY definition is `Move v, Imm(addr)` denotes that constant — the
    // same fact `resolve::absolute_address` relies on — and a slot with NO
    // definition at all whose origin is the `abs`/`rip` pseudo-register is the
    // bare RIP-relative/absolute base itself (its address IS the operand's
    // addend). Everything else is a computed runtime pointer and is refused.
    let mut def_count: HashMap<u32, u32> = HashMap::new();
    let mut imm_def_all: HashMap<u32, i64> = HashMap::new();
    for insn in insns.iter() {
        if let Some(Operand::Local(d)) = insn.dest() {
            *def_count.entry(*d).or_insert(0) += 1;
            if insn.op == Op::Move {
                if let Some(Operand::Immediate(a)) = insn.operands.get(1) {
                    imm_def_all.insert(*d, *a);
                }
            }
        }
    }
    let imm_def: HashMap<u32, i64> = imm_def_all
        .into_iter()
        .filter(|(d, _)| def_count.get(d).copied() == Some(1))
        .collect();
    let mut iterations = 0usize;
    let mut changed = true;
    while changed && iterations < 32 {
        changed = false;
        iterations += 1;

        // T2: a local the code DEREFERENCES is an address. Runs before the
        // per-op rules so an address cursor is `IntPtr` on the same visit that
        // first sees the deref, instead of staying `Unknown` (which made
        // `type_role_name` abstain and kept the `vN` spelling).
        if seed_address_uses(insns, table) {
            changed = true;
        }

        for insn in insns.iter_mut() {
            match &insn.op {
                Op::Move => {
                    let Some(src_ty) = operand_type(insn.operands.get(1), table, oracle) else {
                        continue;
                    };
                    apply_dest_type(insn, table, src_ty, &mut changed);
                }
                Op::Arith(kind) => {
                    // Guarded field-address seeding (see `arith_field_address_seed`):
                    // a proven `&struct_field` refines the destination to
                    // `ByRef(struct)` AHEAD of the generic integer guess below,
                    // which is what lets a later `[dst + sub]` resolve through
                    // `class_of`. Every refusal falls through to the guess, so
                    // the common case is byte-identical to before.
                    let mut seeded = false;
                    if matches!(kind, super::ir::ArithKind::Add) {
                        let base = insn.operands.get(1).cloned().unwrap_or(Operand::Unknown);
                        // `this`-aliases belong to the address sweep below (it
                        // rewrites them to `Move(dst, Field)`); seeding here
                        // too would only churn their rendering for no new name.
                        if !base_is_this(&base, table, &copy_src, &phi_inputs) {
                            if let Some(seed_ty) = arith_field_address_seed(insn, table, oracle) {
                                // Authoritative overwrite of the generic guess
                                // ONLY: `Unknown` or the `I8` this same arm
                                // would otherwise have installed (including the
                                // late-refinement case, where the base was
                                // untyped on an earlier visit). A destination
                                // holding any real type is never downgraded to
                                // an address. `refine_type` is deliberately NOT
                                // widened for this — the overwrite is local to
                                // this proven site.
                                if let Some(Operand::Local(dst)) = insn.dest().cloned() {
                                    if let Some(&li) = table.index_of.get(&dst) {
                                        if matches!(
                                            table.infos[li].ty,
                                            ManagedType::Unknown
                                                | ManagedType::Primitive(PrimKind::I8)
                                                // T2/T1 may have already typed the
                                                // address cursor `IntPtr`; a proven
                                                // `ByRef(struct)` is strictly more
                                                // informative, so the seed still
                                                // wins. A concrete type never does.
                                                | ManagedType::IntPtr
                                        ) {
                                            table.infos[li].ty = seed_ty;
                                            changed = true;
                                        }
                                    }
                                }
                                seeded = true;
                            }
                        }
                    }
                    if !seeded {
                        let ty = arith_result_type(
                            *kind,
                            insn.operands.get(1),
                            insn.operands.get(2),
                            table,
                        );
                        apply_dest_type(insn, table, ty, &mut changed);
                    }
                }
                Op::FloatArith(_) | Op::Negate => {
                    let src = operand_type(insn.operands.get(1), table, oracle)
                        .unwrap_or(ManagedType::Unknown);
                    let ty = if matches!(src, ManagedType::Primitive(PrimKind::R8)) {
                        ManagedType::Primitive(PrimKind::R8)
                    } else {
                        ManagedType::Primitive(PrimKind::R4)
                    };
                    apply_dest_type(insn, table, ty, &mut changed);
                }
                Op::Compare(_) => {
                    apply_dest_type(
                        insn,
                        table,
                        ManagedType::Primitive(PrimKind::Boolean),
                        &mut changed,
                    );
                }
                Op::Select(_) => {
                    // The `CSET` / `CSETM` aliases were folded by the lifter to
                    // `cond ? 1 : 0` / `cond ? -1 : 0`, and that IS how a
                    // managed `bool` materialises in optimized A64 — so type
                    // those Boolean, or the recovered boolean comes back as an
                    // `int` local and every `if (b)` reads an int. For a real
                    // select the result is the selected value's type.
                    let cset = matches!(
                        (insn.operands.get(2), insn.operands.get(3)),
                        (Some(Operand::Immediate(1)), Some(Operand::Immediate(0)))
                            | (Some(Operand::Immediate(-1)), Some(Operand::Immediate(0)))
                    );
                    let ty = if cset {
                        ManagedType::Primitive(PrimKind::Boolean)
                    } else {
                        operand_type(insn.operands.get(2), table, oracle)
                            .or_else(|| operand_type(insn.operands.get(3), table, oracle))
                            .unwrap_or(ManagedType::Unknown)
                    };
                    apply_dest_type(insn, table, ty, &mut changed);
                }
                Op::Not => {
                    // bool not vs bitwise — refine from source when known
                    let src = operand_type(insn.operands.get(1), table, oracle)
                        .unwrap_or(ManagedType::Unknown);
                    let ty = match src {
                        ManagedType::Primitive(p @ PrimKind::Boolean) => ManagedType::Primitive(p),
                        ManagedType::Unknown => ManagedType::Primitive(PrimKind::Boolean),
                        other => other,
                    };
                    apply_dest_type(insn, table, ty, &mut changed);
                }
                Op::NewObj(Some(t)) => {
                    let ty = ManagedType::Class(std::sync::Arc::new(t.ty.clone()));
                    apply_dest_type(insn, table, ty, &mut changed);
                }
                Op::NewArr(Some(elem)) => {
                    let ty = ManagedType::SzArray(Box::new(elem.clone()));
                    apply_dest_type(insn, table, ty, &mut changed);
                }
                Op::Box(Some(t)) => {
                    let ty = ManagedType::Class(std::sync::Arc::new(t.ty.clone()));
                    apply_dest_type(insn, table, ty, &mut changed);
                }
                Op::Call | Op::CallVoid => {
                    if let Some(Operand::Method(m)) = insn.operands.first() {
                        // Return type → destination local (Call only; CallVoid
                        // has no result slot, so apply_dest_type is a no-op).
                        let ret = m.sig.ret.clone();
                        apply_dest_type(insn, table, ret, &mut changed);
                        // Back-propagate parameter types from the callee's
                        // declared signature to the caller's argument locals.
                        // After align_call_arguments the operand layout is:
                        //   Call:    [target, result, receiver?, param0, …]
                        //   CallVoid:[target, receiver?, param0, …]
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
                                    if refine_type(&table.infos[li].ty, pty) {
                                        table.infos[li].ty = pty.clone();
                                        changed = true;
                                        use std::sync::atomic::Ordering;
                                        super::timing::CALLSITE_PARAM_TYPED.fetch_add(
                                            1,
                                            Ordering::Relaxed,
                                        );
                                    }
                                }
                            }
                        }
                        // The RECEIVER is the other half of the same fact: an
                        // instance call to `T.M` proves its receiver is a `T`.
                        // Only `Unknown`/`IntPtr`/`I8` locals move — a
                        // `this`, a declared parameter, a field-typed or
                        // call-typed receiver is a proven type and `refine_type`
                        // refuses to downgrade it.
                        if !m.sig.is_static {
                            let recv_idx = if has_result { 2 } else { 1 };
                            if let Some(Operand::Local(v)) = insn.operands.get(recv_idx) {
                                if let Some(&li) = table.index_of.get(v) {
                                    // A frame cell is STORAGE, not a value: the
                                    // renderer decides [cell + off] from the
                                    // cell type (mem_base_expr pointer-cell
                                    // test), so typing a ref/out cell as a class
                                    // would turn the value it holds back into
                                    // its own address.
                                    let frame_cell =
                                        table.infos[li].slot_origin.starts_with("stack_");
                                    if let Some(rty) =
                                        receiver_type(m, oracle).filter(|_| !frame_cell)
                                    {
                                        if refine_type(&table.infos[li].ty, &rty) {
                                            table.infos[li].ty = rty;
                                            changed = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Op::Convert(conv) => {
                    // Numeric conversions produce a known primitive type
                    // regardless of the source: the conversion's target
                    // determines the result type exactly.
                    let ty = match conv {
                        super::ir::Conversion::IntToFloat32 => {
                            ManagedType::Primitive(PrimKind::R4)
                        }
                        super::ir::Conversion::IntToFloat64 => {
                            ManagedType::Primitive(PrimKind::R8)
                        }
                        super::ir::Conversion::FloatToInt => {
                            ManagedType::Primitive(PrimKind::I8)
                        }
                        super::ir::Conversion::Float32ToFloat64 => {
                            ManagedType::Primitive(PrimKind::R8)
                        }
                        super::ir::Conversion::Float64ToFloat32 => {
                            ManagedType::Primitive(PrimKind::R4)
                        }
                    };
                    apply_dest_type(insn, table, ty, &mut changed);
                }
                Op::MulAcc { is_float, .. } => {
                    // FMA/FMS: result is float when is_float, else integer.
                    let ty = if *is_float {
                        ManagedType::Primitive(PrimKind::R4)
                    } else {
                        ManagedType::Primitive(PrimKind::I8)
                    };
                    apply_dest_type(insn, table, ty, &mut changed);
                }
                Op::Clz => {
                    // Count leading zeros always produces an integer result.
                    apply_dest_type(
                        insn,
                        table,
                        ManagedType::Primitive(PrimKind::I4),
                        &mut changed,
                    );
                }
                Op::LoadLength => {
                    apply_dest_type(
                        insn,
                        table,
                        ManagedType::Primitive(PrimKind::I4),
                        &mut changed,
                    );
                }
                Op::Phi => {
                    // merge arg types
                    let merged = insn
                        .operands
                        .iter()
                        .skip(1)
                        .filter_map(|o| operand_type(Some(o), table, oracle))
                        .fold(ManagedType::Unknown, |acc, t| merge_types(acc, t));
                    if !matches!(merged, ManagedType::Unknown) {
                        apply_dest_type(insn, table, merged, &mut changed);
                    }
                }
                _ => {}
            }
        }

        // second sweep: memory loads AND STORES off typed bases resolve fields
        for insn in insns.iter_mut() {
            if insn.op == Op::Move {
                // store side first: Move(Memory{base,addend}, value)
                if let Some(Operand::Memory { base, addend, .. }) = insn.operands.first().cloned() {
                    if let Some(base_ty) = operand_type(Some(&base), table, oracle) {
                        if let Some(nt) = class_of(&base_ty) {
                            if nt.type_def_index >= 0 {
                                // `!is_static`: this rewrite is gated on the
                                // base being provably `this`, and `this` is a
                                // managed OBJECT — its memory holds instance
                                // fields only. A static field's "offset" indexes
                                // the per-class static block (first field at 0),
                                // so accepting one here renamed klass-pointer
                                // derefs into fabricated `Type.staticField` reads.
                                if let Some(field) = oracle
                                    .field_at_offset(nt.type_def_index, addend)
                                    .filter(|f| !f.is_static)
                                {
                                    let fty = field.field_type.clone();
                                    // mark stored-field type on the VALUE local
                                    if let Some(v) = insn.operands.get(1).and_then(|o| o.as_local())
                                    {
                                        if let Some(&li) = table.index_of.get(&v) {
                                            if refine_type(&table.infos[li].ty, &fty) {
                                                table.infos[li].ty = fty;
                                                changed = true;
                                            }
                                        }
                                    }
                                    // A `this` receiver folds to the bare FieldRef
                                    // (`this.field`); a DERIVED receiver keeps its
                                    // base in an `InstanceField`, so the store
                                    // renders `base.field` instead of dropping
                                    // the receiver to `__unknown.field`.
                                    if base_is_this(&base, table, &copy_src, &phi_inputs) {
                                        insn.operands[0] = Operand::Field(field);
                                    } else {
                                        insn.operands[0] = Operand::InstanceField {
                                            base: base.clone(),
                                            field,
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(Operand::Memory { base, addend, .. }) = insn.operands.get(1).cloned() {
                    if let Some(base_ty) = operand_type(Some(&base), table, oracle) {
                        if let Some(nt) = class_of(&base_ty) {
                            if nt.type_def_index >= 0 {
                                // See the store side above: `this` only ever
                                // holds instance fields, so a static-field
                                // match at an instance offset is a collision
                                // (static block offset 0 vs the object's
                                // klass pointer), never a real read.
                                if let Some(field) = oracle
                                    .field_at_offset(nt.type_def_index, addend)
                                    .filter(|f| !f.is_static)
                                {
                                    let fty = field.field_type.clone();
                                    if let Some(Operand::Local(dst)) = insn.dest().cloned() {
                                        if let Some(&li) = table.index_of.get(&dst) {
                                            if refine_type(&table.infos[li].ty, &fty) {
                                                table.infos[li].ty = fty.clone();
                                                changed = true;
                                            }
                                        }
                                        insn.dest_type = Some(fty);
                                    }
                                    // rewrite load into a resolved field ref.
                                    // `this` folds to the bare FieldRef; a
                                    // derived receiver (a local holding
                                    // `this.Motor`) keeps its base in an
                                    // `InstanceField`, rendering `this.Motor.field`
                                    // instead of the old `__unknown._controller`
                                    // (bare FieldRef, receiver dropped).
                                    if base_is_this(&base, table, &copy_src, &phi_inputs) {
                                        insn.operands[1] = Operand::Field(field);
                                    } else {
                                        insn.operands[1] = Operand::InstanceField {
                                            base: base.clone(),
                                            field,
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // static-address sweep: `*(nint*)(0x63a… + off)` is a STATIC FIELD.
        //
        // `resolve` names a static access when the block address is
        // materialised through a FieldInfo metadata-usage slot; what it cannot
        // see is the form where that address is a BARE CONSTANT, which the
        // census counts as `staticabs`/`asingleton`. A static `FieldRef`
        // renders `Type.field` from its OWN declaring-type name — it carries no
        // receiver, so nothing is dropped when the operand is collapsed — so
        // the access can be named here directly once the address is proven.
        //
        // Additive by construction: the instance sweep above already ran, so a
        // load or store it named is no longer a `Memory` operand and cannot be
        // touched twice; the rewrite only fires when `absolute_static_addr`
        // proves a constant address AND `static_field_at` confirms a metadata
        // field, so every computed base is refused. Both typing arms are
        // monotone refinements, so a later visit cannot undo a concrete type.
        for insn in insns.iter_mut() {
            if insn.op != Op::Move {
                continue;
            }
            for slot in [0usize, 1usize] {
                let Some(Operand::Memory { base, addend, .. }) = insn.operands.get(slot).cloned()
                else {
                    continue;
                };
                let Some(addr) = absolute_static_addr(&base, addend, &imm_def, &def_count, table)
                else {
                    continue;
                };
                let Some(field) = oracle.static_field_at(addr).filter(|f| f.is_static) else {
                    continue;
                };
                let fty = field.field_type.clone();
                if slot == 0 {
                    // store: `Type.field = value` — type the stored value so the
                    // write-back and any later read of it agree.
                    if let Some(v) = insn.operands.get(1).and_then(|o| o.as_local()) {
                        if let Some(&li) = table.index_of.get(&v) {
                            if refine_type(&table.infos[li].ty, &fty) {
                                table.infos[li].ty = fty;
                                changed = true;
                            }
                        }
                    }
                } else {
                    // load: type the destination from the FIELD's own declared
                    // type. This is the forward typing that lets a chain
                    // continue — `v = Type.staticField;` now has a real type, so
                    // `v.something` resolves on the next iteration instead of
                    // stopping at an opaque pointer.
                    if let Some(Operand::Local(dst)) = insn.dest().cloned() {
                        if let Some(&li) = table.index_of.get(&dst) {
                            if refine_type(&table.infos[li].ty, &fty) {
                                table.infos[li].ty = fty.clone();
                                changed = true;
                            }
                        }
                        insn.dest_type = Some(fty);
                    }
                }
                insn.operands[slot] = Operand::Field(field);
            }
        }

        // third sweep: `Arith(Add, <this-alias>, #k)` is a field ADDRESS.
        //
        // IL2CPP computes the address of a nested struct field exactly this
        // way — `add x20, x19, #0x28` for `&this->ps`, and equally as the
        // write-back of `ldr x22, [x20, #0x28]!` (see `lift::memory_writeback`).
        // Left as an ADD, the receiver of a value-type instance call rendered
        // `this + 40.LinePos`; resolving it to the field the offset names is
        // what makes the call read `this.ps.LinePos`, exactly as the memory
        // sweep above already names a `[this+0x28]` LOAD `this.ps`. Gated on
        // the base being provably `this` and the offset naming a real
        // instance field, so an ordinary pointer bump is never renamed.
        for insn in insns.iter_mut() {
            if !matches!(&insn.op, Op::Arith(super::ir::ArithKind::Add)) {
                continue;
            }
            let Some(dst) = insn.dest().cloned() else {
                continue;
            };
            let base = insn.operands.get(1).cloned().unwrap_or(Operand::Unknown);
            let Some(Operand::Immediate(off)) = insn.operands.get(2).cloned() else {
                continue;
            };
            if !base_is_this(&base, table, &copy_src, &phi_inputs) {
                continue;
            }
            let base_ty = operand_type(Some(&base), table, oracle);
            let Some(nt) = base_ty.as_ref().and_then(class_of) else {
                continue;
            };
            if nt.type_def_index < 0 {
                continue;
            }
            let Some(field) = oracle
                .field_at_offset(nt.type_def_index, off)
                .filter(|f| !f.is_static)
            else {
                continue;
            };
            let fty = field.field_type.clone();
            let addr = insn.address;
            *insn = Insn::new(Op::Move, vec![dst, Operand::Field(field)], addr);
            insn.dest_type = Some(fty);
            changed = true;
        }
    }

    // ---- high-variable recovery (see `highvar.rs`) --------------------------
    //
    // Runs HERE, at the end of the type fixpoint, because that is the first
    // point at which everything the merge rules need is final: the CFG (all
    // branches folded), the SSA versions, and every local's slot origin and
    // type. It collapses copy chains, resolves phi confluences whose arguments
    // agree, unifies identical pure computations, gives one register slot's
    // non-overlapping versions one name, and classifies the reads that remain
    // (marking the ones the ABI proves are incoming). It is monotone and
    // idempotent, so the count it returns is deliberately NOT folded into the
    // fixpoint above — re-running it would find nothing, and the loop's
    // convergence must stay a function of type refinement alone.
    //
    // Phi-structure repair for `ssa::eliminate` (see `restore_phi_structure`):
    // `highvar` rewrites Phi arguments and nops their definitions in the
    // LISTING, but `SsaOutput.phis` (consumed by `eliminate` to place edge
    // copies) still holds the original arguments. `eliminate` therefore plants
    // moves reading versions whose definitions are gone — used-never-defined
    // locals (measured: the bulk of the undef-local suspect bucket on both
    // corpora, e.g. `Move(29,53)` after `Move(53,11)` was folded and
    // `Phi(29,53,55)` became `Phi(29,11,55)` in the listing). The repair keeps
    // every sound use-rewrite `highvar` proved and restores only the Phi
    // structure plus its transitive definitions, so `eliminate` sees the
    // arguments it recorded; the restored definitions are dead (their uses
    // were rewritten) and the DCEs after `eliminate` delete them with correct
    // edge moves. Table changes (naming, incoming marks) are kept: they never
    // reach `eliminate`.
    let pre_highvar = insns.to_vec();
    super::highvar::recover(insns, table);
    restore_phi_structure(insns, &pre_highvar);

    iterations
}

/// Restore the Phi structure `highvar::recover` rewrote so `ssa::eliminate`
/// (which lowers Phis from the out-of-band `SsaOutput.phis` records, not from
/// the listing) keeps reading the arguments it recorded.
///
/// `highvar`'s listing rewrites are SOUND for the listing: a folded copy
/// (`Move(53,11)` nopped, `Phi(29,53,55)` rewritten to `Phi(29,11,55)`) and a
/// coalesced phi chain (`Phi(48,35,35)` nopped, its uses rewritten to the
/// agreeing entry value) both preserve values under the dominance proofs the
/// pass records. But `eliminate` never sees the listing's new arguments — it
/// plants `Move(dest, old_arg)` edge copies from the stale records, reading
/// versions with no definition in the listing. Every such read is a
/// used-never-defined local in the final IR, i.e. a `suspect` verdict the
/// verifier reports and the renderer prints as a read of nothing.
///
/// The repair is purely restorative and therefore cannot invent a value: every
/// restored instruction is a verbatim pre-`highvar` instruction at its original
/// index (no write moves), and every KEPT rewrite is one `highvar` proved.
/// Concretely: restore each changed/nopped Phi to its original operands, then
/// restore (to a fixpoint) each nopped definition whose version a restored
/// instruction reads. Non-Phi use-rewrites (calls reading the folded target)
/// stand — their targets are defined by construction (`is_rooted` + dominance
/// in `highvar`), and the restored definitions are dead exactly because those
/// uses moved, so the post-`eliminate` DCEs collect them with correct edges.
///
/// Inert on well-formed IR: when `highvar` changed no Phi (the common case for
/// methods without redundant joins), the first loop restores nothing and the
/// function returns without touching the listing. Ledger staleness is
/// harmless: restored instructions keep any DCE note `highvar` recorded, but
/// they are original lifted ops (present in the pre-`highvar` listing that the
/// decoders produced), so no coverage or semantic-diff excusal they trigger
/// can hide a real divergence.
fn restore_phi_structure(insns: &mut [Insn], pre: &[Insn]) {
    if insns.len() != pre.len() {
        return;
    }
    use std::collections::HashSet;
    let mut restored: HashSet<usize> = HashSet::new();
    for (i, (cur, old)) in insns.iter_mut().zip(pre.iter()).enumerate() {
        if old.op == Op::Phi && (cur.op != old.op || cur.operands != old.operands) {
            *cur = old.clone();
            restored.insert(i);
        }
    }
    if restored.is_empty() {
        return;
    }
    let mut needed: HashSet<u32> = HashSet::new();
    for &i in &restored {
        for u in pre[i].use_locals() {
            needed.insert(u);
        }
    }
    let mut progress = true;
    while progress {
        progress = false;
        for (j, (cur, old)) in insns.iter_mut().zip(pre.iter()).enumerate() {
            if restored.contains(&j) {
                continue;
            }
            if cur.op != Op::Nop {
                continue;
            }
            let Some(Operand::Local(d)) = old.dest() else {
                continue;
            };
            if !needed.contains(d) {
                continue;
            }
            let srcs = old.use_locals();
            *cur = old.clone();
            restored.insert(j);
            for u in srcs {
                needed.insert(u);
            }
            progress = true;
        }
    }
}

fn apply_dest_type(insn: &Insn, table: &mut LocalTable, ty: ManagedType, changed: &mut bool) {
    if matches!(ty, ManagedType::Unknown) {
        return;
    }
    if let Some(Operand::Local(dst)) = insn.dest() {
        if let Some(&li) = table.index_of.get(dst) {
            if refine_type(&table.infos[li].ty, &ty) {
                table.infos[li].ty = ty;
                *changed = true;
            }
        }
    }
}

/// Monotone refinement for a local's type. Unknown seeds anything. The
/// generic `IntPtr` guess that untyped `Memory`-source loads seed ALSO
/// refines — a load destination whose Memory operand later resolves to a
/// metadata field (or whose def is a call/newobj/etc.) is authoritative: the
/// base of `*(T*)(v + off)` being a managed object pointer is exactly why
/// IL2CPP emits object-field derefs as memory reads, so refusing to refine
/// IntPtr keeps every one of those derefs permanently unnameable (e.g.
/// `v = this.Motor` typed nint loses `CharacterMotor` and `+192` never
/// resolves to `_controller`). Never downgrades an already-concrete type.
fn refine_type(cur: &ManagedType, inc: &ManagedType) -> bool {
    use ManagedType::*;
    match (cur, inc) {
        (Unknown, x) => !matches!(x, Unknown),
        // IntPtr is the weakest pointer guess; any authoritative concrete
        // type (class ref, array, string, primitive from a resolved field
        // load, …) is strictly better. Equal IntPtr/Unknown are not changes,
        // and neither is `I8` — the WIDTH-FREE integer guess is WEAKER than
        // "this is an address" (pointer arithmetic yields IntPtr, so an I8
        // reaching an IntPtr local came from a rule that knows less). Letting
        // I8 downgrade IntPtr also made the lattice oscillate
        // (IntPtr -> I8 -> IntPtr ...) until the 32-iteration cap, which cost
        // 3x render time on the 200-method sample.
        (IntPtr, x) => !matches!(x, Unknown | IntPtr | Primitive(PrimKind::I8)),
        // I8 is the WIDTH-FREE integer guess (`arith_result_type` returns it
        // for every integer operation), not a fact about the value. Treating
        // it as final made every later authoritative fact unreachable: a local
        // whose first visit was an `Add` could never become the field's
        // declared `bool`/`int`/`Enum`/`SomeClass`, nor a call's declared
        // parameter/return type. Measured on 0x15A5B6C: `v22 = static_fields +
        // 8` is `long` although the load that produced its base is
        // pointer-width, and `refine_type` refused every correction.
        //
        // Only a CONCRETE type may demote the guess (and I8->I8 is no
        // information); a proven type is still never downgraded.
        (Primitive(PrimKind::I8), x) => !matches!(x, Unknown | Primitive(PrimKind::I8)),
        _ => false,
    }
}

/// Public wrapper for [`refine_type`] — used by the interprocedural pass.
pub fn refine_type_public(cur: &ManagedType, inc: &ManagedType) -> bool {
    refine_type(cur, inc)
}

/// T2 — type every local the code DEREFERENCES as an address.
///
/// `Memory { base: Local(v) }` is a native load/store through `v`: whatever
/// `v` holds, the instruction treats it as a pointer. That is proof, not a
/// guess, and it is the largest single evidence source the lattice was
/// ignoring — measured on a 200-method sample, 46 % of surviving `vN` refs sit
/// on locals used as a deref base and 28 % of all refs sit on ones typed
/// `Unknown`.
///
/// Refusals, each for a reason:
///  * `this` and parameters — already carry identity (and never print `vN`);
///  * frame cells (`slot_origin` `stack_*`) — `[sp+off]` denotes the CELL's
///    address, not a pointer stored in it, and `mem_base_expr` already
///    distinguishes the two through `address_taken`;
///  * any local that already has a concrete type — `refine_type` only accepts
///    `Unknown`/`IntPtr`, so a proven `Class`/`SzArray`/`I4` is untouched.
fn seed_address_uses(insns: &[Insn], table: &mut LocalTable) -> bool {
    let mut changed = false;
    for insn in insns {
        for op in &insn.operands {
            seed_memory_base(op, table, &mut changed);
        }
    }
    changed
}

fn seed_memory_base(op: &Operand, table: &mut LocalTable, changed: &mut bool) {
    match op {
        Operand::Memory { base, index, .. } => {
            if let Operand::Local(v) = base.as_ref() {
                if let Some(&li) = table.index_of.get(v) {
                    let info = &mut table.infos[li];
                    let frame_cell = info.slot_origin.starts_with("stack_");
                    if !info.is_this
                        && info.is_param.is_none()
                        && !frame_cell
                        && refine_type(&info.ty, &ManagedType::IntPtr)
                    {
                        info.ty = ManagedType::IntPtr;
                        *changed = true;
                    }
                }
            }
            seed_memory_base(base, table, changed);
            if let Some(i) = index {
                seed_memory_base(i, table, changed);
            }
        }
        Operand::ArrayAccess { array, index } => {
            seed_memory_base(array, table, changed);
            seed_memory_base(index, table, changed);
        }
        Operand::InstanceField { base, .. } => seed_memory_base(base, table, changed),
        _ => {}
    }
}

/// The class a memory operand's base denotes, peeling one level of reference
/// indirection so a `ByRef`/`Pointer` receiver (an `out`/`ref` parameter, or a
/// value-type field address) resolves its fields exactly like a class reference.
/// Value types are already carried as `Class` by `managed_type_of`, so this only
/// has to unwrap the extra indirection the IR sometimes adds.
fn class_of(ty: &ManagedType) -> Option<std::sync::Arc<super::types::NamedType>> {
    match ty {
        ManagedType::Class(nt) => Some(nt.clone()),
        ManagedType::ByRef(inner) | ManagedType::Pointer(inner) => class_of(inner),
        _ => None,
    }
}

/// Public wrapper for [`class_of`] — used by the interprocedural pass.
pub fn class_of_public(ty: &ManagedType) -> Option<std::sync::Arc<super::types::NamedType>> {
    class_of(ty)
}

/// The ABSOLUTE address a memory operand denotes, when it can be proven.
///
/// Only the forms that name a FIXED location qualify: an `Immediate` base, a
/// local whose single definition is `Move v, Imm(addr)`, and the pre-SSA
/// `abs`/`rip` pseudo-register (whose base names nothing and whose addend IS the
/// address). Any base that is a computed runtime value is refused — a deref of a
/// computed pointer is not a static slot. This mirrors the distinction
/// `resolve::absolute_address` draws, repeated here because the static naming
/// below runs after that pass and sees only its leftovers.
fn absolute_static_addr(
    base: &Operand,
    addend: i64,
    imm_def: &HashMap<u32, i64>,
    def_count: &HashMap<u32, u32>,
    table: &LocalTable,
) -> Option<i64> {
    match base {
        Operand::Immediate(a) => Some(*a + addend),
        Operand::Local(v) => {
            if let Some(a) = imm_def.get(v) {
                return Some(*a + addend);
            }
            // A local the body NEVER writes, whose origin is the absolute/RIP
            // pseudo-register: the addend is the address itself.
            if def_count.get(v).is_none() {
                let origin = table
                    .index_of
                    .get(v)
                    .map(|&li| table.infos[li].slot_origin.as_str())?;
                if origin == "abs" || origin == "rip" {
                    return Some(addend);
                }
            }
            None
        }
        Operand::Register(r) if r == "abs" || r == "rip" => Some(addend),
        _ => None,
    }
}

/// Is this memory base provably the method's `this` pointer? Only then is
/// rewriting `Memory{base, off}` into a bare `FieldRef` faithful — a bare
/// FieldRef carries no receiver, so the renderer can only reconstruct
/// `this.field`. A base that is a DERIVED value (a local holding
/// `this.Motor`, or any other object-typed register/param) must keep its
/// Memory form: collapsing it would drop the receiver and render
/// `__unknown.field` (or a wrong `this.field` when the field's declaring
/// type merely shares the class chain).
fn base_is_this(
    base: &Operand,
    table: &LocalTable,
    copy_src: &HashMap<u32, u32>,
    phi_inputs: &HashMap<u32, Vec<u32>>,
) -> bool {
    match base {
        Operand::Local(v) => is_this_alias(
            *v,
            table,
            copy_src,
            phi_inputs,
            &mut std::collections::HashSet::new(),
        ),
        // entry `this` register (arm64 x0 / x86 this-ptr); non-renamed
        // registers are skipped by the type check anyway (type None)
        _ => false,
    }
}

/// Does `v` provably hold the method's `this`? Direct `is_this` flag OR an
/// unambiguous copy chain of `this` (`v31 = v26` after the x19-pin MOV — pin
/// copies are NOT flagged `is_this`, only the x0 entry version is), or a Phi
/// whose every input is a this-alias (loop-carried receiver). Chasing keeps
/// rewriting faithful for event accessors and other pin-`this` code that
/// IL2CPP compiles with callee-saved receivers.
fn is_this_alias(
    v: u32,
    table: &LocalTable,
    copy_src: &HashMap<u32, u32>,
    phi_inputs: &HashMap<u32, Vec<u32>>,
    seen: &mut std::collections::HashSet<u32>,
) -> bool {
    if !seen.insert(v) {
        return false; // cycle: not provably this
    }
    if table
        .index_of
        .get(&v)
        .map(|&li| table.infos[li].is_this)
        .unwrap_or(false)
    {
        return true;
    }
    match copy_src.get(&v).copied() {
        Some(src) if src != u32::MAX => is_this_alias(src, table, copy_src, phi_inputs, seen),
        Some(_) => false, // multi-def copy: ambiguous
        None => {
            // Phi: every input must be a this-alias (any non-this input
            // means the value varies across paths — not provably this)
            match phi_inputs.get(&v) {
                Some(inputs) if !inputs.is_empty() => inputs
                    .iter()
                    .all(|i| is_this_alias(*i, table, copy_src, phi_inputs, seen)),
                _ => false,
            }
        }
    }
}

pub fn operand_type(
    op: Option<&Operand>,
    table: &LocalTable,
    oracle: &dyn TypeOracle,
) -> Option<ManagedType> {
    match op? {
        Operand::Local(v) => table.index_of.get(v).map(|&i| table.infos[i].ty.clone()),
        Operand::Immediate(_) => Some(ManagedType::Primitive(PrimKind::I8)),
        Operand::FloatLiteral(_) => Some(ManagedType::Primitive(PrimKind::R4)),
        Operand::DoubleLiteral(_) => Some(ManagedType::Primitive(PrimKind::R8)),
        Operand::StringLiteral { .. } => Some(ManagedType::String),
        Operand::Method(m) => Some(ManagedType::Opaque(format!("method {}", m.full_name()))),
        Operand::Field(f) => Some(f.field_type.clone()),
        Operand::InstanceField { field, .. } => Some(field.field_type.clone()),
        Operand::Memory { base, addend, width, .. } => {
            // A load through a base whose class the fixpoint already knows is the
            // FIELD's own type, not an opaque pointer. This is what carries a
            // type across a chain: `obj = SomeCall();` types `obj` from the
            // call's return, and this makes `v = obj.field` take the field's
            // declared type on the next visit, so `v.subfield` resolves at depth
            // > 1 instead of stopping at the first deref. Falls back to the
            // WIDTH-derived shape — or `IntPtr`, the old weak pointer guess —
            // whenever the base or the offset is not provably a metadata field,
            // so it is never a downgrade.
            let base_ty = operand_type(Some(&**base), table, oracle);
            if let Some(nt) = base_ty.as_ref().and_then(class_of) {
                if nt.type_def_index >= 0 {
                    if let Some(f) = oracle.field_at_offset(nt.type_def_index, *addend) {
                        return Some(f.field_type);
                    }
                }
            }
            // The access WIDTH is a fact the lifter recorded (`ldr w0` vs
            // `ldr x0`), and without it a provably 32-bit integer load was
            // typed `IntPtr` — a pointer. That mis-typing is not cosmetic: the
            // next `Add` on it produced `I8` (see `arith_result_type`), and the
            // value stayed "long" for the rest of the method. Width 0 is "not
            // stated" and keeps the pointer guess; 8 is pointer-width, where
            // long-vs-pointer cannot be decided from the width alone.
            match width {
                1 => Some(ManagedType::Primitive(PrimKind::I1)),
                2 => Some(ManagedType::Primitive(PrimKind::I2)),
                4 => Some(ManagedType::Primitive(PrimKind::I4)),
                _ => Some(ManagedType::IntPtr),
            }
        }
        Operand::AddressOf(inner) => {
            operand_type(Some(&**inner), table, oracle).map(|t| ManagedType::ByRef(Box::new(t)))
        }
        _ => None,
    }
}

fn merge_types(a: ManagedType, b: ManagedType) -> ManagedType {
    use ManagedType::*;
    match (&a, &b) {
        (Unknown, x) => x.clone(),
        (x, Unknown) => x.clone(),
        (x, y) if x == y => x.clone(),
        // numeric widening fallbacks
        (Primitive(_), Primitive(_)) => Primitive(PrimKind::I8),
        _ => Object,
    }
}

/// The managed type a call's RECEIVER operand is known to have.
///
/// An instance call's receiver must be an instance of the DECLARING type: that
/// is what the callee's metadata says, and it is the only evidence the call
/// site carries about the object. `align_call_arguments` already uses the
/// signature to type every PARAMETER local, but it starts at `param0` and skips
/// the receiver slot — so a local whose only use was `obj.Foo()` stayed
/// `Unknown`, `type_role_name` abstained, and the `vN` spelling survived even
/// though the callee named the object's class.
///
/// Refusals: a STATIC call has no receiver; a `MethodRef` with no TypeDef index
/// (`type_def_index < 0`) or no declaring type name carries no evidence. For a
/// STRUCT declaring type the receiver holds the ADDRESS of the struct (that is
/// how IL2CPP passes a value-type `this`), so it is typed `ByRef(Class)` —
/// still an address, and `class_of` peels it to resolve the struct's fields.
fn receiver_type(m: &super::types::MethodRef, oracle: &dyn TypeOracle) -> Option<ManagedType> {
    if m.sig.is_static || m.type_def_index < 0 || m.declaring_type_name.is_empty() {
        return None;
    }
    let nt = std::sync::Arc::new(super::types::NamedType {
        namespace: m.declaring_namespace.clone(),
        name: m.declaring_type_name.clone(),
        assembly: (!m.assembly.is_empty()).then(|| m.assembly.clone()),
        type_def_index: m.type_def_index,
    });
    let cls = ManagedType::Class(nt);
    Some(match oracle.is_struct_value_type(m.type_def_index) {
        Some(true) => ManagedType::ByRef(Box::new(cls)),
        _ => cls,
    })
}

/// Does this type denote an ADDRESS in the native code?
///
/// A managed reference is an address in IL2CPP (objects, arrays, strings are
/// raw pointers), `ByRef`/`Pointer` are addresses by construction, and
/// `IntPtr` is the lattice's own pointer guess. `Unknown` is deliberately NOT
/// address-like: refusing is what keeps this rule from guessing.
fn is_address_like(t: &ManagedType) -> bool {
    use ManagedType::*;
    matches!(
        t,
        IntPtr | Class(_) | ByRef(_) | Pointer(_) | SzArray(_) | String | Object | Opaque(_)
    )
}

fn arith_result_type(
    kind: super::ir::ArithKind,
    lhs: Option<&Operand>,
    rhs: Option<&Operand>,
    table: &LocalTable,
) -> ManagedType {
    use super::ir::ArithKind;
    let lt = operand_type(lhs, table, &NullOracle);
    let rt = operand_type(rhs, table, &NullOracle);
    let floaty = [lt.as_ref(), rt.as_ref()]
        .into_iter()
        .flatten()
        .any(|t| t.is_float());
    if floaty {
        return ManagedType::Primitive(PrimKind::R4);
    }
    match kind {
        ArithKind::And
            if matches!(lt, Some(ManagedType::Primitive(PrimKind::Boolean)))
                || matches!(rt, Some(ManagedType::Primitive(PrimKind::Boolean))) =>
        {
            ManagedType::Primitive(PrimKind::Boolean)
        }
        // POINTER ARITHMETIC. `base + k` / `base - k` with exactly one
        // address-like side is an ADDRESS: it is how IL2CPP spells every
        // field, element and vtable-slot address computation
        // (`add x8, x19, #0x28`, `add x22, x0, w1, uxtw #4`). Returning
        // `I8` here threw the address-ness away, and since `refine_type` used
        // to treat `I8` as final nothing downstream could restore it, so the
        // cursor could never be named (type_role_name maps `IntPtr` -> `ptr`).
        //
        // `Sub(ptr, ptr)` is an element/byte DIFFERENCE — an integer — and
        // `Add` of two addresses is not an address; both keep `I8`.
        ArithKind::Add | ArithKind::Sub => {
            let lp = lt.as_ref().is_some_and(is_address_like);
            let rp = rt.as_ref().is_some_and(is_address_like);
            match (lp, rp) {
                (true, false) | (false, true) => ManagedType::IntPtr,
                _ => ManagedType::Primitive(PrimKind::I8),
            }
        }
        _ => ManagedType::Primitive(PrimKind::I8),
    }
}

pub struct NullOracle;
impl TypeOracle for NullOracle {
    fn field_at_offset(&self, _tdi: i32, _offset: i64) -> Option<FieldRef> {
        None
    }
}

/// Klass-pointer typing: `Move dst, *slot` where slot's address is a known
/// TypeInfo metadata-usage → dst (and the slot local) get the real class
/// type, enabling subsequent field-offset resolution.
pub fn type_klass_pointers(
    insns: &mut [Insn],
    table: &mut LocalTable,
    klass_to_tdi: &HashMap<u64, i32>,
    metadata: &crate::il2cpp::metadata::Metadata,
) -> usize {
    // slot-local → Imm(addr) defs
    let mut imm_defs: HashMap<u32, i64> = HashMap::new();
    for insn in insns.iter() {
        if !matches!(insn.op, Op::Move) {
            continue;
        }
        if let (Some(Operand::Local(d)), Some(Operand::Immediate(a))) =
            (insn.dest().cloned(), insn.operands.get(1))
        {
            imm_defs.insert(d, *a);
        }
    }
    let mut typed = 0usize;
    for insn in insns.iter_mut() {
        if !matches!(insn.op, Op::Move) {
            continue;
        }
        if let Some(Operand::Memory { base, .. }) = insn.operands.get(1).cloned() {
            let base_slot = match base.as_ref() {
                Operand::Local(b) => Some(*b),
                _ => None,
            };
            let Some(bs) = base_slot else { continue };
            let Some(addr) = imm_defs.get(&bs) else {
                continue;
            };
            let Some(&tdi) = klass_to_tdi.get(&(*addr as u64)) else {
                continue;
            };
            if tdi < 0 || tdi as usize >= metadata.type_defs.len() {
                continue;
            }
            let td = &metadata.type_defs[tdi as usize];
            let nt = std::sync::Arc::new(super::types::NamedType {
                namespace: metadata
                    .get_string_from_index(td.namespace_index)
                    .unwrap_or_default(),
                name: metadata
                    .get_string_from_index(td.name_index)
                    .unwrap_or_default(),
                assembly: None,
                type_def_index: tdi,
            });
            // type both the base local and this move's destination
            for slot in [Some(bs), insn.dest().and_then(|d| d.as_local())]
                .into_iter()
                .flatten()
            {
                if let Some(&li) = table.index_of.get(&slot) {
                    if matches!(table.infos[li].ty, ManagedType::Unknown) {
                        table.infos[li].ty = ManagedType::Class(nt.clone());
                        typed += 1;
                    }
                }
            }
        }
    }
    typed
}

#[cfg(test)]
mod typing_tests {
    use super::super::ir::{Insn, Op, Operand};
    use super::super::types::{FieldRef, ManagedType, NamedType, PrimKind};
    use super::*;
    use std::sync::Arc;

    fn field(name: &str, ty: ManagedType, tdi: i32, is_static: bool) -> FieldRef {
        FieldRef {
            type_def_index: tdi,
            field_index: 0,
            name: name.to_string(),
            declaring_type_name: "Test".to_string(),
            field_type: ty,
            is_static,
        }
    }

    struct Oracle {
        inst: Option<FieldRef>,
        statics: Vec<(i64, FieldRef)>,
    }
    impl TypeOracle for Oracle {
        fn field_at_offset(&self, tdi: i32, offset: i64) -> Option<FieldRef> {
            self.inst
                .clone()
                .filter(|f| f.type_def_index == tdi && offset == 24)
        }
        fn static_field_at(&self, addr: i64) -> Option<FieldRef> {
            self.statics
                .iter()
                .find(|(a, _)| *a == addr)
                .map(|(_, f)| f.clone())
        }
    }

    /// Oracle for the field-address seeding tests: owner type 7 holds one
    /// instance field at offset 24 whose type varies per test; `structs` is the
    /// set of TypeDefs the oracle confirms as struct value types (`None` =
    /// the oracle cannot tell, which must refuse).
    struct AddrOracle {
        field_type: ManagedType,
        structs: Option<Vec<i32>>,
    }
    impl TypeOracle for AddrOracle {
        fn field_at_offset(&self, tdi: i32, offset: i64) -> Option<FieldRef> {
            if tdi == 7 && offset == 24 {
                Some(field("nested", self.field_type.clone(), tdi, false))
            } else {
                None
            }
        }
        fn is_struct_value_type(&self, tdi: i32) -> Option<bool> {
            self.structs.as_ref().map(|s| s.contains(&tdi))
        }
    }

    fn nt(tdi: i32) -> Arc<NamedType> {
        Arc::new(NamedType {
            namespace: String::new(),
            name: format!("T{tdi}"),
            assembly: None,
            type_def_index: tdi,
        })
    }

    fn table_with(entries: &[(u32, ManagedType, bool)]) -> LocalTable {
        let mut t = LocalTable::default();
        for (v, ty, is_this) in entries {
            let idx = t.infos.len();
            t.infos.push(LocalInfo {
                slot_origin: format!("v{v}"),
                ty: ty.clone(),
                is_this: *is_this,
                is_param: None,
                param_comp: None,
                is_return_buffer: false,
                is_incoming_arg: false,
                is_call_clobber: false,
            });
            t.index_of.insert(*v, idx);
        }
        t
    }

    fn mem(base: Operand, addend: i64) -> Operand {
        Operand::Memory {
            base: Box::new(base),
            index: None,
            scale: 1,
            addend,
            width: 0,
        }
    }

    fn idx(t: &LocalTable, v: u32) -> usize {
        t.index_of[&v]
    }

    /// A load through a DERIVED receiver (a local holding another object) is
    /// typed from the field the oracle names AND named through the
    /// receiver-carrying `Operand::InstanceField`, so it renders
    /// `base.health` instead of the old raw `*(float*)(base + 24)`. A bare
    /// `Operand::Field` can only render `this.field`, so the receiver MUST ride
    /// along in the operand — that is what the `InstanceField` variant exists
    /// for, and why this no longer collapses to a receiver-less `Field`.
    #[test]
    fn derived_receiver_load_is_typed_not_collapsed() {
        let oracle = Oracle {
            inst: Some(field(
                "health",
                ManagedType::Primitive(PrimKind::R4),
                7,
                false,
            )),
            statics: vec![],
        };
        let mut table = table_with(&[
            (0, ManagedType::Class(nt(7)), false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(2), mem(Operand::Local(0), 24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::Primitive(PrimKind::R4)
        );
        assert_eq!(
            insns[0].dest_type,
            Some(ManagedType::Primitive(PrimKind::R4))
        );
        assert!(
            matches!(
                &insns[0].operands[1],
                Operand::InstanceField { field, .. } if field.name == "health"
            ),
            "a derived receiver must keep its base and name the field"
        );
    }

    /// The SAME load with a receiver that IS `this` collapses to `this.field`
    /// (existing behaviour, unchanged).
    #[test]
    fn this_receiver_load_collapses_to_field() {
        let oracle = Oracle {
            inst: Some(field(
                "health",
                ManagedType::Primitive(PrimKind::R4),
                7,
                false,
            )),
            statics: vec![],
        };
        let mut table = table_with(&[
            (0, ManagedType::Class(nt(7)), true),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(2), mem(Operand::Local(0), 24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        match &insns[0].operands[1] {
            Operand::Field(f) => assert_eq!(f.name, "health"),
            other => panic!("expected Field, got {other:?}"),
        }
    }

    /// An ABSOLUTE static-storage address is named `Type.field` and its result
    /// typed from the field — the `staticabs` family.
    #[test]
    fn absolute_static_load_is_named() {
        let sf = field("count", ManagedType::Primitive(PrimKind::I4), 3, true);
        let oracle = Oracle {
            inst: None,
            statics: vec![(0x63a_0010, sf)],
        };
        let mut table = table_with(&[(5, ManagedType::Unknown, false)]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(5), mem(Operand::Immediate(0x63a_0000), 16)],
            0x2000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        match &insns[0].operands[1] {
            Operand::Field(f) => {
                assert!(f.is_static);
                assert_eq!(f.name, "count");
            }
            other => panic!("expected static Field, got {other:?}"),
        }
        assert_eq!(
            insns[0].dest_type,
            Some(ManagedType::Primitive(PrimKind::I4))
        );
    }

    /// A static STORE `*(nint*)(0x63a…) = v` becomes `Type.field = v`.
    #[test]
    fn absolute_static_store_is_named() {
        let sf = field("count", ManagedType::Primitive(PrimKind::I4), 3, true);
        let oracle = Oracle {
            inst: None,
            statics: vec![(0x63a_0010, sf)],
        };
        let mut table = table_with(&[(9, ManagedType::Primitive(PrimKind::I4), false)]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![mem(Operand::Immediate(0x63a_0000), 16), Operand::Local(9)],
            0x2000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        match &insns[0].operands[0] {
            Operand::Field(f) => assert!(f.is_static),
            other => panic!("expected static Field dest, got {other:?}"),
        }
    }

    /// A local whose ONLY definition is `Move v, Imm(addr)` denotes that
    /// address, so a deref off it is the same static read.
    #[test]
    fn imm_defined_base_is_treated_as_address() {
        let sf = field("count", ManagedType::Primitive(PrimKind::I4), 3, true);
        let oracle = Oracle {
            inst: None,
            statics: vec![(0x63a_0010, sf)],
        };
        let mut table = table_with(&[]);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(1), Operand::Immediate(0x63a_0000)],
                0x1ff0,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(5), mem(Operand::Local(1), 16)],
                0x2000,
            ),
        ];
        propagate_types(&mut insns, &mut table, &oracle);
        match &insns[1].operands[1] {
            Operand::Field(f) => assert!(f.is_static),
            other => panic!("expected static Field, got {other:?}"),
        }
    }

    /// Negative: an untyped base stays a raw deref and is typed nothing.
    #[test]
    fn untyped_base_stays_raw() {
        let oracle = Oracle {
            inst: Some(field(
                "health",
                ManagedType::Primitive(PrimKind::R4),
                7,
                false,
            )),
            statics: vec![],
        };
        let mut table = table_with(&[
            (0, ManagedType::Unknown, false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(2), mem(Operand::Local(0), 24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        // The operand stays a RAW deref — no field was named, which is the
        // property under test.
        assert!(matches!(&insns[0].operands[1], Operand::Memory { .. }));
        // ...and the destination takes the weak pointer guess `IntPtr`, NOT the
        // field's real type. `operand_type(Memory{..})` deliberately falls back
        // to `IntPtr` when the base class or the offset is not provably a
        // metadata field (see its doc comment), so an untyped base must never
        // acquire the R4 field type the oracle would otherwise hand out.
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::IntPtr,
            "an untyped base must not be field-typed (must stay the IntPtr fallback)"
        );
    }

    /// A `ByRef`/value-type base resolves its fields exactly like a class ref.
    #[test]
    fn byref_value_type_base_resolves() {
        let oracle = Oracle {
            inst: Some(field(
                "LinePos",
                ManagedType::Primitive(PrimKind::R4),
                7,
                false,
            )),
            statics: vec![],
        };
        let mut table = table_with(&[
            (
                0,
                ManagedType::ByRef(Box::new(ManagedType::Class(nt(7)))),
                false,
            ),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(2), mem(Operand::Local(0), 24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::Primitive(PrimKind::R4)
        );
    }

    /// `operand_type` reports the field's type for a load through a typed base,
    /// which is what carries a type across a chain at depth > 1.
    #[test]
    fn operand_type_names_field_through_memory() {
        let oracle = Oracle {
            inst: Some(field(
                "health",
                ManagedType::Primitive(PrimKind::R4),
                7,
                false,
            )),
            statics: vec![],
        };
        let table = table_with(&[(0, ManagedType::Class(nt(7)), false)]);
        let ty = operand_type(Some(&mem(Operand::Local(0), 24)), &table, &oracle);
        assert_eq!(ty, Some(ManagedType::Primitive(PrimKind::R4)));
    }

    /// Refusal pin (P0 interface-dispatch class): `dst = base + w` with a
    /// DYNAMIC offset — the `klass + idx*16 (+ 312)` chain whose slot is
    /// computed at runtime — must NEVER take a field type. There is no slot to
    /// key without a constant, and guessing one would invent a type the
    /// resolver would then key a table on. Keeps the generic integer guess.
    ///
    /// Characterization, not a fix: passes before and after, and exists so the
    /// naive "propagate the base's field type through any Add" change that was
    /// tried and reverted (425k-line dump.cs damage, lost stores) cannot be
    /// reintroduced without this failing first.
    #[test]
    fn arith_add_with_dynamic_offset_is_never_field_typed() {
        use super::super::ir::ArithKind;
        let oracle = Oracle {
            inst: Some(field("nested", ManagedType::Class(nt(9)), 7, false)),
            statics: vec![],
        };
        let mut table = table_with(&[
            (0, ManagedType::Class(nt(7)), false),
            (1, ManagedType::Primitive(PrimKind::I8), false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(0), Operand::Local(1)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::IntPtr,
            "a dynamic offset names no slot, but `base + w` off a REFERENCE is still an address"
        );
        assert!(
            matches!(insns[0].op, Op::Arith(ArithKind::Add)),
            "the address computation must not be rewritten"
        );
    }

    /// Refusal pin (same class): `dst = base + const` with an UNTYPED base
    /// stays the integer guess — without a class there is no owner, so there
    /// is nothing sound to seed from the offset alone.
    #[test]
    fn arith_add_with_untyped_base_is_never_field_typed() {
        use super::super::ir::ArithKind;
        let oracle = Oracle {
            inst: Some(field("nested", ManagedType::Class(nt(9)), 7, false)),
            statics: vec![],
        };
        let mut table = table_with(&[
            (0, ManagedType::Unknown, false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(0), Operand::Immediate(24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::Primitive(PrimKind::I8),
            "an untyped base has no owner: the destination must stay the integer guess"
        );
    }

    /// Red-green (Worker F): `dst = base + 24` where 24 names a STRUCT-valued
    /// field seeds `ByRef(struct)` — the address, not the value.
    ///
    /// Fails pre-fix (the destination keeps the generic `I8` guess, so a later
    /// `[dst + sub]` can never resolve through `class_of`); passes with the
    /// guarded seed. The seed stays an address: a `ByRef`, never the struct
    /// itself, and the computation is never rewritten.
    #[test]
    fn arith_add_struct_field_seeds_byref_address() {
        use super::super::ir::ArithKind;
        let oracle = AddrOracle {
            field_type: ManagedType::Class(nt(9)),
            structs: Some(vec![9]),
        };
        let mut table = table_with(&[
            (0, ManagedType::Class(nt(7)), false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(0), Operand::Immediate(24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::ByRef(Box::new(ManagedType::Class(nt(9)))),
            "the address of a struct field must seed ByRef(struct), not the integer guess"
        );
        assert!(
            matches!(insns[0].op, Op::Arith(ArithKind::Add)),
            "the address computation itself must never be rewritten"
        );
    }

    /// A chained seed: `&S + 24` where 24 names a nested struct field seeds
    /// `&(nested)`. The `ByRef` base is itself a proven address from an earlier
    /// seed, so the owner is as proven as for a direct `Class` base.
    #[test]
    fn arith_add_through_byref_base_chains_struct_address() {
        use super::super::ir::ArithKind;
        let oracle = AddrOracle {
            field_type: ManagedType::Class(nt(9)),
            structs: Some(vec![9]),
        };
        let mut table = table_with(&[
            (
                0,
                ManagedType::ByRef(Box::new(ManagedType::Class(nt(7)))),
                false,
            ),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(0), Operand::Immediate(24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::ByRef(Box::new(ManagedType::Class(nt(9)))),
            "a struct offset off a proven struct address must chain the ByRef seed"
        );
    }

    /// Refusal pin (Worker F guard): a REFERENCE-typed field's address stays
    /// the integer guess — never its class. Typing the address as the class
    /// makes the next `[dst + off2]` resolve as a subfield of the wrong owner,
    /// the snowball the unguarded attempt shipped (425k-line dump.cs damage).
    #[test]
    fn arith_add_reference_field_stays_integer() {
        use super::super::ir::ArithKind;
        let oracle = AddrOracle {
            field_type: ManagedType::Class(nt(9)),
            structs: Some(vec![]),
        };
        let mut table = table_with(&[
            (0, ManagedType::Class(nt(7)), false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(0), Operand::Immediate(24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::IntPtr,
            "the address of a reference-typed field must stay an ADDRESS, never its class"
        );
        assert!(
            matches!(insns[0].op, Op::Arith(ArithKind::Add)),
            "a refused seed must leave the computation alone"
        );
    }

    /// Refusal pin: an enum-typed field's address stays the integer guess. An
    /// enum's only field is the compiler-named `value__`, so seeding could name
    /// no subfield — only reproduce the `(raw + 16).value__` misname.
    #[test]
    fn arith_add_enum_field_stays_integer() {
        use super::super::ir::ArithKind;
        // An enum answers `Some(false)` to the struct query, exactly like a
        // reference class: value type, but not a struct to resolve through.
        let oracle = AddrOracle {
            field_type: ManagedType::Class(nt(11)),
            structs: Some(vec![]),
        };
        let mut table = table_with(&[
            (0, ManagedType::Class(nt(7)), false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(0), Operand::Immediate(24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::IntPtr,
            "the address of an enum field names no subfield and must stay an address"
        );
    }

    /// Refusal pin: a PRIMITIVE-typed field's address stays the integer guess.
    /// An `int` field IS a value type, but it has no subfields, so a seed
    /// could never resolve anything through it — only add `ByRef` noise.
    #[test]
    fn arith_add_primitive_field_stays_integer() {
        use super::super::ir::ArithKind;
        let oracle = AddrOracle {
            field_type: ManagedType::Primitive(PrimKind::I4),
            structs: Some(vec![7]),
        };
        let mut table = table_with(&[
            (0, ManagedType::Class(nt(7)), false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(0), Operand::Immediate(24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::IntPtr,
            "a primitive field has no subfields: its address must stay an address"
        );
    }

    /// Refusal pin: when the oracle cannot tell struct-ness (`None`), the seed
    /// refuses rather than guessing. Unknown means unknown.
    #[test]
    fn arith_add_unknown_structness_stays_integer() {
        use super::super::ir::ArithKind;
        let oracle = AddrOracle {
            field_type: ManagedType::Class(nt(9)),
            structs: None,
        };
        let mut table = table_with(&[
            (0, ManagedType::Class(nt(7)), false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(0), Operand::Immediate(24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::IntPtr,
            "unprovable struct-ness must refuse the ByRef seed, never guess it"
        );
    }

    /// Phi-structure survives `highvar` so `ssa::eliminate` keeps its edge
    /// sources: a folded copy (`Move(53,11)` into `Phi(29,53,55)`) must not
    /// rewrite the Phi in the listing while `SsaOutput.phis` still records
    /// the original argument — `eliminate` would plant `Move(29,53)` reading
    /// a nopped definition (used-never-defined, i.e. suspect). `propagate`
    /// keeps the sound use-rewrites and restores the Phi plus its transitive
    /// definitions; the restored definitions are dead and the post-eliminate
    /// DCEs collect them with correct edges.
    #[test]
    fn phi_copy_fold_keeps_phi_sources_for_eliminate() {
        let oracle = Oracle {
            inst: None,
            statics: vec![],
        };
        let mut table = LocalTable::default();
        for (v, is_param) in [
            (11u32, true),
            (53u32, false),
            (55u32, false),
            (29u32, false),
        ] {
            let idx = table.infos.len();
            table.infos.push(LocalInfo {
                slot_origin: format!("s{v}"),
                ty: ManagedType::Unknown,
                is_this: false,
                is_param: is_param.then_some(0),
                param_comp: None,
                is_return_buffer: false,
                is_incoming_arg: false,
                is_call_clobber: false,
            });
            table.index_of.insert(v, idx);
        }
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(53), Operand::Local(11)],
                0x1000,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(55), Operand::Immediate(5)],
                0x1004,
            ),
            Insn::new(
                Op::Phi,
                vec![Operand::Local(29), Operand::Local(53), Operand::Local(55)],
                0x1008,
            ),
            Insn::new(Op::Return, vec![Operand::Local(29)], 0x100c),
        ];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            insns[2].operands,
            vec![Operand::Local(29), Operand::Local(53), Operand::Local(55)],
            "the Phi must still name its original sources for eliminate"
        );
        assert!(
            matches!(&insns[0].op, Op::Move),
            "the folded copy's definition must survive until eliminate+DCE"
        );
    }

    // -----------------------------------------------------------------
    // 4.2 — evidence-driven typing (docs/orientation/10-recon-typing.md)
    // -----------------------------------------------------------------

    fn empty_oracle() -> Oracle {
        Oracle {
            inst: None,
            statics: vec![],
        }
    }

    /// T1: `base + k` with one address-like side is an ADDRESS. IL2CPP spells
    /// every field/element/vtable address this way; typing it `I8` threw the
    /// address-ness away and nothing downstream could restore it, so the
    /// cursor could never be named (type_role_name maps IntPtr -> `ptr`).
    #[test]
    fn pointer_plus_constant_is_a_pointer() {
        let oracle = empty_oracle();
        let mut table = table_with(&[
            (1, ManagedType::IntPtr, false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(super::super::ir::ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(1), Operand::Immediate(8)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::IntPtr,
            "address + constant must stay an address"
        );
    }

    /// T1 refusal: the DIFFERENCE of two addresses is an integer, not an
    /// address. Typing it `IntPtr` would make every length/offset computation
    /// look like a pointer.
    #[test]
    fn pointer_minus_pointer_is_an_integer() {
        let oracle = empty_oracle();
        let mut table = table_with(&[
            (1, ManagedType::IntPtr, false),
            (2, ManagedType::IntPtr, false),
            (3, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Arith(super::super::ir::ArithKind::Sub),
            vec![Operand::Local(3), Operand::Local(1), Operand::Local(2)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 3)].ty,
            ManagedType::Primitive(PrimKind::I8),
            "pointer difference is an integer"
        );
    }

    /// T2: a local the code DEREFERENCES is an address. This is proof, not a
    /// guess — the native instruction treats whatever it holds as a pointer.
    #[test]
    fn dereferenced_local_is_typed_as_an_address() {
        let oracle = empty_oracle();
        let mut table = table_with(&[
            (0, ManagedType::Unknown, false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(2), mem(Operand::Local(0), 24)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 0)].ty,
            ManagedType::IntPtr,
            "a deref base is an address"
        );
    }

    /// T2 refusal: a FRAME cell is not a pointer. `[sp+off]` denotes the
    /// cell's address, not a pointer stored in it, and `mem_base_expr` already
    /// distinguishes the two through `address_taken`; typing the cell itself
    /// would corrupt that decision.
    #[test]
    fn stack_cell_deref_is_not_typed_as_a_pointer() {
        let oracle = empty_oracle();
        let mut table = table_with(&[(0, ManagedType::Unknown, false)]);
        table.infos[0].slot_origin = "stack_16".into();
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(2), mem(Operand::Local(0), 0)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 0)].ty,
            ManagedType::Unknown,
            "a frame cell must not be typed as a pointer"
        );
    }

    /// T3: `I8` is the WIDTH-FREE integer guess, not a proof. Field evidence
    /// must be able to correct it — before this, `refine_type` refused every
    /// move out of I8, so a local whose first visit was an `Add` could never
    /// become the field's declared class/struct type.
    #[test]
    fn field_evidence_demotes_the_i8_guess() {
        let oracle = Oracle {
            inst: Some(field("motor", ManagedType::Class(nt(7)), 7, false)),
            statics: vec![],
        };
        // Local 1 holds `this` (type T7); local 9 is the stored value, already
        // poisoned with the I8 guess by an earlier arithmetic visit.
        let mut table = table_with(&[
            (1, ManagedType::Class(nt(7)), true),
            (9, ManagedType::Primitive(PrimKind::I8), false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![mem(Operand::Local(1), 24), Operand::Local(9)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 9)].ty,
            ManagedType::Class(nt(7)),
            "the stored value must take the field's declared type"
        );
    }

    /// T3 refusal: a PROVEN type is never downgraded to the I8 guess.
    #[test]
    fn a_proven_type_is_not_downgraded_to_the_integer_guess() {
        let oracle = empty_oracle();
        let mut table = table_with(&[
            (1, ManagedType::Class(nt(7)), false),
            (2, ManagedType::Primitive(PrimKind::I4), false),
            (3, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![
            Insn::new(
                Op::Arith(super::super::ir::ArithKind::Add),
                vec![Operand::Local(3), Operand::Local(1), Operand::Immediate(4)],
                0x1000,
            ),
            Insn::new(
                Op::Arith(super::super::ir::ArithKind::Add),
                vec![Operand::Local(2), Operand::Local(1), Operand::Immediate(4)],
                0x1004,
            ),
        ];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 2)].ty,
            ManagedType::Primitive(PrimKind::I4),
            "a proven int32 must not be replaced by the width-free guess"
        );
    }

    /// T4: the ACCESS WIDTH is a type. The renderer already spells a 4-byte
    /// access `*(int*)`, a 2-byte one `*(short*)` and a 1-byte one
    /// `*(byte*)`; the lattice must agree, or the value is called a pointer.
    #[test]
    fn width_four_load_is_a_32_bit_integer() {
        let oracle = empty_oracle();
        let mut table = table_with(&[
            (0, ManagedType::Unknown, false),
            (1, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![
                Operand::Local(1),
                Operand::Memory {
                    base: Box::new(Operand::Local(0)),
                    index: None,
                    scale: 1,
                    addend: 0,
                    width: 4,
                },
            ],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 1)].ty,
            ManagedType::Primitive(PrimKind::I4),
            "a 4-byte load is a 32-bit integer, not a pointer"
        );
    }

    /// T4 refusal: width 0 is "not stated" and keeps the pointer guess.
    #[test]
    fn width_zero_load_stays_a_pointer() {
        let oracle = empty_oracle();
        let mut table = table_with(&[
            (0, ManagedType::Unknown, false),
            (1, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(1), mem(Operand::Local(0), 0)],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 1)].ty,
            ManagedType::IntPtr,
            "an unstated width keeps the pointer guess"
        );
    }

    // -----------------------------------------------------------------
    // 4.2 — receiver typing (the call site's other half of the evidence)
    // -----------------------------------------------------------------

    fn nt_asm(tdi: i32, asm: &str) -> Arc<NamedType> {
        Arc::new(NamedType {
            namespace: String::new(),
            name: format!("T{tdi}"),
            assembly: Some(asm.to_string()),
            type_def_index: tdi,
        })
    }

    fn method(name: &str, tdi: i32, is_static: bool) -> super::super::types::MethodRef {
        super::super::types::MethodRef {
            image_index: 0,
            type_def_index: tdi,
            method_index: 0,
            name: name.to_string(),
            declaring_namespace: String::new(),
            declaring_type_name: format!("T{tdi}"),
            assembly: "Test".to_string(),
            sig: super::super::types::MethodSigInfo {
                ret: ManagedType::Unknown,
                params: vec![],
                is_static,
            },
        }
    }

    /// A local whose ONLY use is `obj.Foo()` is proven to be a `Foo` by the
    /// callee's own metadata. Before this rule the receiver slot was skipped
    /// (parameter back-propagation starts at `param0`), so such a local stayed
    /// `Unknown`, `type_role_name` abstained, and the `vN` spelling survived
    /// although the call named the object's class.
    #[test]
    fn a_call_receiver_takes_the_callee_declaring_type() {
        let oracle = AddrOracle {
            field_type: ManagedType::Unknown,
            structs: Some(vec![]),
        };
        let mut table = table_with(&[
            (1, ManagedType::Unknown, false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Call,
            vec![
                Operand::Method(method("Update", 7, false)),
                Operand::Local(2), // result
                Operand::Local(1), // receiver
            ],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 1)].ty,
            ManagedType::Class(nt_asm(7, "Test")),
            "the receiver of T7.Update is a T7"
        );
    }

    /// Refusal: a STRUCT declaring type passes its `this` as a pointer, so the
    /// receiver is an ADDRESS of the struct (`ByRef`), never the struct value.
    #[test]
    fn a_struct_receiver_is_the_struct_address() {
        let oracle = AddrOracle {
            field_type: ManagedType::Unknown,
            structs: Some(vec![7]),
        };
        let mut table = table_with(&[
            (1, ManagedType::Unknown, false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Call,
            vec![
                Operand::Method(method("Normalize", 7, false)),
                Operand::Local(2),
                Operand::Local(1),
            ],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 1)].ty,
            ManagedType::ByRef(Box::new(ManagedType::Class(nt_asm(7, "Test")))),
            "a value-type receiver holds the struct's address"
        );
    }

    /// Refusal: a STATIC call has no receiver, and its first argument is not
    /// one — typing it would be a fabricated fact about an unrelated local.
    #[test]
    fn a_static_call_does_not_type_its_first_argument_as_a_receiver() {
        let oracle = AddrOracle {
            field_type: ManagedType::Unknown,
            structs: Some(vec![]),
        };
        let mut table = table_with(&[
            (1, ManagedType::Unknown, false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Call,
            vec![
                Operand::Method(method("Parse", 7, true)),
                Operand::Local(2),
                Operand::Local(1),
            ],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 1)].ty,
            ManagedType::Unknown,
            "a static call carries no receiver"
        );
    }

    /// Refusal: a FRAME cell is storage, not a value. `mem_base_expr` decides
    /// whether `[cell + off]` reads the cell's address or the reference the
    /// callee stored in it from the cell's TYPE, so typing a `ref`/`out` cell
    /// as a class would silently turn the reference back into its own address.
    #[test]
    fn a_frame_cell_receiver_is_not_retyped() {
        let oracle = AddrOracle {
            field_type: ManagedType::Unknown,
            structs: Some(vec![]),
        };
        let mut table = table_with(&[
            (1, ManagedType::Unknown, false),
            (2, ManagedType::Unknown, false),
        ]);
        table.infos[0].slot_origin = "stack_16".into();
        let mut insns = vec![Insn::new(
            Op::Call,
            vec![
                Operand::Method(method("Update", 7, false)),
                Operand::Local(2),
                Operand::Local(1),
            ],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 1)].ty,
            ManagedType::Unknown,
            "a frame cell keeps its storage type"
        );
    }

    /// Refusal: a receiver that is already a PROVEN type is never rewritten.
    #[test]
    fn a_proven_receiver_is_not_retyped_by_a_base_class_call() {
        let oracle = AddrOracle {
            field_type: ManagedType::Unknown,
            structs: Some(vec![]),
        };
        let mut table = table_with(&[
            (1, ManagedType::Class(nt(9)), false),
            (2, ManagedType::Unknown, false),
        ]);
        let mut insns = vec![Insn::new(
            Op::Call,
            vec![
                Operand::Method(method("ToString", 7, false)),
                Operand::Local(2),
                Operand::Local(1),
            ],
            0x1000,
        )];
        propagate_types(&mut insns, &mut table, &oracle);
        assert_eq!(
            table.infos[idx(&table, 1)].ty,
            ManagedType::Class(nt(9)),
            "a proven receiver keeps its own class"
        );
    }
}
