//! Metadata resolution pass — full port of Cpp2IL `Analysis/MetadataResolver.cs`
//! (essential subset, restructured for our IR):
//!
//! - `resolve_calls`: Immediate call targets → MethodRef via methods-by-RVA;
//!   raw ABI register args re-sliced onto the callee signature
//!   (`RemapRawArguments`); ambiguous/shared addresses left indirect
//! - `resolve_string_literal_accessors`: immediate literal-slot addresses →
//!   StringLiteral operands (covers `il2cpp_string_new` sites too)
//! - `key_function_recovery`: object_new/box/raise-exception/string-new
//!   families rewritten into NewObj/Box/Throw/StringLiteral ops
//! - `fuse_constructor_calls`: Call(.ctor) right after NewObj with matching
//!   receiver is fused (the .ctor's `this` traces back to the NewObj result)
//! - `resolve_getters`: single-load property getters become
//!   `Return FieldRef`
//! - `rewrite_virtual_calls`: `BLR [klass + vtable_offset + slot*psize]`
//!   patterns resolved through the vtable map when klass type is known

use super::ir::{Insn, Op, Operand};
use super::locals::LocalTable;
use super::types::{FieldRef, ManagedType, MethodRef, MethodSigInfo, NamedType, PrimKind};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default, Clone)]
pub struct ResolveMaps {
    /// method RVA → resolved managed method (with full signature)
    pub methods_by_rva: HashMap<u64, MethodRef>,
    /// metadata-usage / literal-slot address → string-literal index
    pub literal_index_by_addr: HashMap<u64, u32>,
    /// Address of a POINTER TO a literal slot → the same string-literal index.
    ///
    /// Where the literal table is not addressed directly the toolchain reaches
    /// a literal through a relocated pointer: `ldr x8,[slot]` loads the ADDRESS
    /// of the encoded word and the next load reads the word. The scan records
    /// the word's own address, so a site that goes through the pointer never
    /// resolved and the load was typed as the declaring class's klass —
    /// `return typeof(PotionsManager)` where `"icon_disabled_1"` belongs
    /// (`GetPotionIconName` @0x1990BCC). In the v108 build 19,435 of the 19,252
    /// literal words are reachable this way, so this is the common case.
    pub literal_ptr_index_by_addr: HashMap<u64, u32>,
    pub literals: HashMap<u32, String>,
    /// metadata-usage slot address → `MethodSpec` index (MethodRef usages).
    /// The slot holds the concrete instantiation of a generic method — what a
    /// call site loads to pass to a shared generic body.
    pub method_spec_index_by_addr: HashMap<u64, u32>,
    /// generic body RVA → every instantiation compiled to that address, as
    /// `(spec index, instantiated ref)`, in ascending spec-index order.
    /// One address carries many entries exactly when IL2CPP shared the body;
    /// `methods_by_rva` can only hold one of them.
    pub specs_by_body_rva: HashMap<u64, Vec<(u32, MethodRef)>>,
    /// Target RVA → every method DEFINITION compiled to that address, when
    /// more than one is. IL2CPP folds identical bodies: `RegistryKey::get_
    /// InternalHandle`, `LocalDataStoreElement::get_Cookie` and `OperatingSystem
    /// ::get_Version` are all "load the field at this offset", so all three
    /// share ONE address (fanout 88 on the v24.5 build). `methods_by_rva` is
    /// single-valued, so its entry there is one arbitrary member of the set and
    /// the name it produces is a FABRICATED declaring type — the reader is told
    /// a method belongs to a class it does not. Measured: 644 of 12,909
    /// addresses on v24.5, 2,142 of 37,644 on v108, and 74% of those span more
    /// than one type DEFINITION.
    pub shared_by_rva: HashMap<u64, Vec<MethodRef>>,
    /// The `shared_by_rva` keys whose members disagree about the declaring type
    /// DEFINITION. Differing type ARGUMENTS (`List<int>` vs `List<string>`) is
    /// a different, benign case: the type named is still the right one.
    pub shared_cross_type: HashMap<u64, bool>,
    /// method-definition index → the folded address it was compiled to. The
    /// reverse of `shared_by_rva`, so a rewritten call can find out whether the
    /// name it carries is one arbitrary member of a folded set.
    pub shared_rva_of_method: HashMap<u32, u64>,
    /// `method_specs` index → its instantiated ref, for the specs the code
    /// pointer table does NOT index (those it does are in `specs_by_body_rva`).
    /// Under full generic sharing only the shared instantiations are compiled,
    /// so a call site's concrete instantiation routinely has no pointer at all
    /// and cannot be among the candidates — `method_index` (the method
    /// DEFINITION) is what ties the two together: a spec whose definition is
    /// one of the definitions already known to live at the called address IS
    /// that call's instantiation.
    pub specs_by_index: HashMap<u32, MethodRef>,
    /// key function RVAs
    pub object_new: Vec<u64>,
    pub string_new: Vec<u64>,
    pub box_fn: Vec<u64>,
    pub raise_exception: Vec<u64>,
    /// il2cpp_codegen_initialize_class / InitializeMethodMetadata family —
    /// runtime metadata bookkeeping, dropped from recovered C# (the managed
    /// source never contained these calls).
    pub metadata_init: Vec<u64>,
    /// vtable slot → virtual method, per declaring TypeDef
    pub vtable_slots: HashMap<(i32, usize), MethodRef>,
    /// klass pointer RVA → TypeDef index
    pub klass_to_tdi: HashMap<u64, i32>,
    /// metadata-usage slot RVA → the FULL managed type the slot names.
    ///
    /// `klass_to_tdi` only covers named class/value-type/generic-instantiation
    /// slots, because that is what field/vtable resolution needs. The renderer
    /// needs the rest: `SzArray` slots (`il2cpp_array_new_specific` takes the
    /// ARRAY class), primitives (`typeof(int)`), and generic instantiations
    /// without a TypeDef. Without this map every such slot fell through to
    /// `void*.ptr_0x…`, which is why 1,259 of 1,316 array allocations had no
    /// element type and ~3.3k type comparisons read a raw pointer.
    pub usage_type_by_addr: HashMap<u64, ManagedType>,
    /// TypeDef index → (namespace, name) for klass slots seen in usages —
    /// lets key_function_recovery type NewObj without a metadata callback
    pub tdi_names: HashMap<i32, (String, String)>,
    /// Objective-C selector stubs: stub RVA → selector name. IL2CPP builds
    /// on Apple platforms bridge engine native calls through per-selector
    /// stubs (`stp x29/x30; mov x29,sp; bl objc_msgSend$sel; ret`) so a call
    /// to the stub IS a message send with that fixed selector. Recovered
    /// from the stub's BL target's GOT entry pointing into __objc_selrefs.
    pub objc_selector_stubs: HashMap<u64, String>,
    /// Engine/native code target → metadata WRAPPER methods whose compiled
    /// bodies branch DIRECTLY to it. IL2CPP statically resolves extern
    /// (InternalCall) engine methods, so game call sites BL the engine
    /// implementation directly — the implementation is not itself in
    /// metadata, but each engine method's generated wrapper body (which IS
    /// metadata, with a code pointer) ends in a direct `B`/`BL` to it. A
    /// native target with exactly ONE wrapper therefore has that wrapper's
    /// name + signature (positional ABI trimming becomes safe once the call
    /// is rendered through the signature).
    pub native_wrappers: HashMap<u64, Vec<MethodRef>>,
    /// Pre-computed instance field layouts: `(tdi, byte_offset) → (field_name, declaring_type_name)`.
    /// Populated during `RecoveryContext` construction from the metadata's type/field defs.
    /// Used by the post-hoc field annotation pass to resolve `RawDeref` to `Field`.
    pub field_layouts: HashMap<(i32, i64), (String, String)>,
    /// Pre-computed instance field types: `(tdi, byte_offset) → field_type_tdi`.
    /// Populated alongside `field_layouts`. Used by the pointer chain resolution
    /// in the field annotation pass to resolve multi-level dereference chains
    /// (e.g. `*(nint*)(*(nint*)(obj +16) +8)` → `obj.Field1.Field2`).
    pub field_types: HashMap<(i32, i64), i32>,
    /// `(tdi, byte_offset) → declared access width` for instance fields whose
    /// width is unambiguous (primitives, enums, references). A store may only
    /// be named as this field when the access width matches; a `ValueType`
    /// struct field is deliberately absent (a narrower store would be writing
    /// a sub-field, not the struct).
    pub field_widths: HashMap<(i32, i64), u8>,
    /// Pre-computed static field layouts: `global_VA → (field_name, type_name)`.
    /// Populated from `field_index_by_addr` (metadata-usage slot addresses) during
    /// `RecoveryContext` construction. Used by the post-hoc field annotation pass
    /// to resolve `RawDeref(Int(base), offset, _)` to `StaticField(type, field)`.
    pub static_field_layouts: HashMap<u64, (String, String)>,
    /// Type display name → byte size, for array-element stride checks.
    ///
    /// Arrays used to sugar element accesses only for the built-in element
    /// names (`object`, `string`, `int`, …); a `MyClass[]` or `MyStruct[]`
    /// element store was refused because an arbitrary name could be a struct
    /// of any size. Metadata knows the answer: references are pointer-sized,
    /// value types have an `instance_size`. Short names are kept only when
    /// unambiguous; full `Namespace.Name` keys are always present.
    pub named_type_sizes: HashMap<String, u64>,
    /// Pre-computed reverse map: `(tdi, field_name) → field_type_tdi`.
    /// Built once from field_layouts + field_types during RecoveryContext construction.
    /// Used by pointer chain resolution to look up field types by name.
    pub field_type_by_name: HashMap<(i32, String), i32>,
    /// Per-TypeDef instance fields in declaration layout:
    /// `tdi → [(byte_offset, field_name, declared_type_tdi_or_-1)]`.
    ///
    /// Only fields DECLARED by that type (inherited ones live on the parent's
    /// entry). This is the table the embedded-layout descent needs: an access
    /// at `obj + 0x1C` where the type has a `HAPI_GeoInfo` field at 0x14 is
    /// `_geoInfo.id`, and finding that requires the fields' offsets AND their
    /// declared type indexes, which `field_layouts`/`field_types` cannot be
    /// scanned by without walking the whole map per query.
    pub tdi_fields: HashMap<i32, Vec<(i64, String, i32)>>,
    /// TypeDefs that are VALUE types. The embedded-layout descent may only
    /// enter a value-type field: a class-typed field's storage holds a
    /// POINTER, so an offset measured from the outer object can never land
    /// inside the pointee.
    pub value_type_tdis: std::collections::HashSet<i32>,
    /// `(tdi, field_name) → full generic type text` for fields declared as a
    /// generic INSTANCE (`Dictionary<int, Token>`, `List<object>`). The
    /// allocation/ctor name from shared generic metadata only carries the
    /// generic definition with `object` arguments; the field's declared
    /// instantiation is authoritative for a `new` assigned to it.
    pub field_generic_text: HashMap<(i32, String), String>,
    /// TypeDef indices whose type is an INTERFACE. An allocation may not be
    /// typed as an interface (`new IFoo()` is invalid), so allocation retyping
    /// refuses these.
    pub interface_tdis: HashSet<i32>,
    /// TypeDef parent chains: `tdi → [tdi, parent_tdi, grandparent_tdi, …]`.
    /// Used by the field annotation pass to walk base-class chains when looking
    /// up a field at an offset (inherited fields live on the parent's layout).
    pub type_parent_chains: HashMap<i32, Vec<i32>>,
    /// Pre-computed enum member names: `tdi → (constant_value → member_name)`.
    /// Populated during `RecoveryContext` construction for every enum type in
    /// the metadata. Used by the enum_names reconstruction pass to replace
    /// integer constants with `EnumType.MemberName` when the context indicates
    /// an enum type.
    pub enum_members: HashMap<i32, HashMap<i64, String>>,
    /// Pre-computed attribute constructor names: `RVA → attribute_type_name`.
    /// Maps each custom attribute generator function pointer (RVA) to the
    /// short name of the attribute type it constructs (e.g. "SerializeField",
    /// "Header", "Tooltip"). Populated during `RecoveryContext` construction
    /// from `custom_attribute_generators` + `attribute_type_ranges`. Used by
    /// the attribute recovery pass to recognise attribute constructor calls
    /// in static initialisers and emit `[AttributeName(args)]` comments.
    pub attribute_ctor_names: HashMap<u64, String>,
    /// GOT address → relocation addend. Maps each GOT slot address to the
    /// value it will point to at runtime (from ELF relocations). Used to
    /// resolve `*(nint*)(got_base + offset)` patterns to their actual targets.
    pub got_addend_by_addr: HashMap<u64, u64>,
    /// Virtual address → method name. Pre-computed from methods_by_rva by
    /// converting RVAs to VAs. Used for quick lookup during field annotation
    /// when we have a VA (e.g., from GOT addends) instead of an RVA.
    pub method_name_by_va: HashMap<u64, (String, String)>,
    /// Pre-computed static field offsets within the static_fields structure:
    /// `(type_name, byte_offset) → (field_name, declaring_type_name)`.
    /// Populated during `RecoveryContext` construction. Used by the field
    /// annotation pass to resolve `RawDeref(Field(base, "static_fields"), offset)`
    /// to `StaticField(type_name, field_name)` when the type is known from the
    /// klass pointer or a variable and the offset matches a static field's position.
    pub static_field_offsets: HashMap<(String, i64), (String, String)>,
    /// Declared type of a static field: `(declaring_type_name, field_name) → tdi`.
    /// Lets pointer-chain resolution continue through `Type.StaticField`
    /// (`TRWorldInfo.RunnerGame +216`), which previously stopped at the
    /// static field and left the instance deref raw.
    pub static_field_types: HashMap<(String, String), i32>,
    /// Pre-computed reverse of `tdi_names`: `"TypeName"` or `"Namespace.TypeName"` → tdi.
    /// Built once during `RecoveryContext` construction so that `infer_types_from_stmts`
    /// does not rebuild it per method (thousands of times).
    pub tdi_names_reverse: HashMap<String, i32>,
    /// Pre-computed method name → return type tdi (only when unambiguous).
    /// Built once during `RecoveryContext` construction so that `infer_types_from_stmts`
    /// does not rebuild it per method.
    pub method_return_tdi: HashMap<String, i32>,
}

#[derive(Debug)]
pub struct ResolveStats {
    pub statics_resolved: usize,
    pub calls_resolved: usize,
    pub calls_unresolved: usize,
    pub calls_trimmed: usize,
    pub strings_resolved: usize,
    pub key_functions: usize,
    pub ctors_fused: usize,
    pub getters_resolved: usize,
    pub virtuals_resolved: usize,
    pub generic_bodies_pinned: usize,
    /// Calls to a folded (identical-code-merged) address that the CALL SITE
    /// itself named — the `NewObj(T)` klass or the receiver's class picked
    /// which of the merged definitions the call means.
    pub shared_calls_picked: usize,
    /// Calls to a folded address with no call-site evidence for any of the
    /// merged definitions. The name in the output is the canonical member, and
    /// that is a GUESS — counted so it is a number, not an assumption.
    pub shared_calls_ambiguous: usize,
}

impl Default for ResolveStats {
    fn default() -> Self {
        Self {
            statics_resolved: 0,
            calls_resolved: 0,
            calls_unresolved: 0,
            calls_trimmed: 0,
            strings_resolved: 0,
            key_functions: 0,
            ctors_fused: 0,
            getters_resolved: 0,
            virtuals_resolved: 0,
            generic_bodies_pinned: 0,
            shared_calls_picked: 0,
            shared_calls_ambiguous: 0,
        }
    }
}

/// Run the full resolution suite over the (renamed) listing.
pub fn run(insns: &mut [Insn], table: &LocalTable, maps: &ResolveMaps) -> ResolveStats {
    let no_fields = HashMap::new();
    let no_parents = HashMap::new();
    let mut no_owner = |_: u32| -> Option<(String, String, ManagedType)> { None };
    run_with_fields(
        insns,
        table,
        maps,
        &no_fields,
        &mut no_owner,
        8,
        &no_parents,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_with_fields(
    insns: &mut [Insn],
    table: &LocalTable,
    maps: &ResolveMaps,
    field_index_by_addr: &HashMap<u64, u32>,
    field_owner: &mut dyn FnMut(u32) -> Option<(String, String, ManagedType)>,
    ptr_size: u64,
    parents: &HashMap<i32, Vec<i32>>,
) -> ResolveStats {
    let mut stats = ResolveStats::default();
    resolve_static_fields(insns, field_index_by_addr, field_owner, &mut stats);
    resolve_string_accessors(insns, maps, &mut stats);
    resolve_call_targets(insns, maps, &mut stats);
    refine_component_type_from_store(insns);
    // Calls into the IL2CPP runtime itself (its own `.text`, which no metadata
    // map or export table can name) are identified from their call-site shape
    // so they render as named runtime calls instead of `fn_<address>`. Runs
    // here because this is the first point where a resolvable call carries its
    // `MethodRef` (so only genuinely unnameable targets are considered) while
    // the static-init guard this reads is still intact — `ssa::eliminate` and
    // the optimization loop both dissect it.
    super::runtime_helpers::detect(insns);
    key_function_recovery(insns, maps, field_index_by_addr, &mut stats);
    // AFTER `key_function_recovery`: the call-site evidence this pass needs is
    // the `NewObj(T)` a `.ctor` belongs to, and that op only exists once the
    // allocation call has been recognised. Running it before produced 82
    // evidenced picks against 5,755 guesses on the v24.5 build; the ordering,
    // not the evidence, was the limit.
    //
    // BEFORE `reslice_call_arguments`, and that ordering is load-bearing: this
    // pass reads the call's `x0` and its argument count, and the re-slice
    // replaces both with the CANONICAL member's — so a pick made afterwards was
    // deciding which member the call means using evidence that had already been
    // overwritten with "whatever the canonical member is".
    resolve_folded_names(insns, table, maps, parents, &mut stats);
    // Every call that metadata can name IS named by now, which is what makes
    // each hole's ABI carrier decidable: `__abi` slots whose value is provably
    // still sitting in its argument register are filled here.
    recover_abi_arguments(insns, table);
    // Only now, with every call named as precisely as the metadata allows, is
    // the raw ABI argument dump turned into the signature-shaped operand list
    // the renderer reads. Address-taken cells are excluded from operand
    // tracing: a callee can write them, so their initialiser is not their
    // value at a later call site (0x18A3A1C passed 0 instead of the token).
    let addressed = collect_addressed_locals(insns);
    reslice_call_arguments(insns, &addressed);
    fuse_constructor_calls(insns, &mut stats);
    resolve_getters(insns, &mut stats);
    rewrite_virtual_calls(insns, table, maps, parents, ptr_size, &mut stats);
    if std::env::var("ILR_NOIFACE").is_err() {
        rewrite_interface_invoke_calls(insns, table, maps, parents, &mut stats);
    }
    stats
}

/// Generic Unity `GetComponent<T>()` returns exactly `T`. When generic sharing
/// leaves its return as an opaque generic parameter, the immediately following
/// typed field store supplies the call site's concrete type. This is stronger
/// evidence than the shared body's arbitrary canonical instantiation (for
/// example, `GetComponent<Rigidbody>` on a body used by many component types).
fn refine_component_type_from_store(insns: &mut [Insn]) {
    for i in 0..insns.len() {
        let Some(Operand::Method(method)) = insns[i].operands.first() else {
            continue;
        };
        if !method.name.starts_with("GetComponent<")
            || !matches!(method.sig.ret, ManagedType::Opaque(_))
        {
            continue;
        }
        let method_name = method.name.clone();
        let Some(dst) = insns[i].operands.get(1).and_then(Operand::as_local) else {
            continue;
        };
        let field_type = insns[i + 1..].iter().take(3).find_map(|next| {
            if !matches!(next.op, Op::Move) {
                return None;
            }
            let target = next.operands.first()?;
            let value = next.operands.get(1)?;
            if value.as_local() != Some(dst) {
                return None;
            }
            match target {
                Operand::Field(field) => Some(field.field_type.clone()),
                _ => None,
            }
        });
        let member_receiver_type = insns[i + 1..].iter().take(24).find_map(|next| {
            if !matches!(next.op, Op::Call) {
                return None;
            }
            let Operand::Method(callee) = next.operands.first()? else {
                return None;
            };
            if !callee.name.starts_with("get_")
                || next.operands.get(2).and_then(Operand::as_local) != Some(dst)
                || callee.type_def_index < 0
            {
                return None;
            }
            Some(ManagedType::Class(std::sync::Arc::new(
                super::types::NamedType {
                    namespace: callee.declaring_namespace.clone(),
                    name: callee.declaring_type_name.clone(),
                    assembly: (!callee.assembly.is_empty()).then(|| callee.assembly.clone()),
                    type_def_index: callee.type_def_index,
                },
            )))
        });
        let Some(field_type) = field_type.or(member_receiver_type) else {
            continue;
        };
        let ManagedType::Class(nt) = &field_type else {
            continue;
        };
        let concrete_name = format!("GetComponent<{}>", nt.name);
        if let Some(Operand::Method(method)) = insns[i].operands.first_mut() {
            if method.name == method_name {
                method.name = concrete_name;
                method.sig.ret = field_type;
            }
        }
    }
}

// ---------------------------------------------------------------------------

/// Static-field resolution: `Move dst, Imm(slot_addr)` where slot_addr is a
/// FieldInfo metadata-usage → `dst` becomes `Type.field` (ldsfld value or
/// address depending on subsequent deref — we render the named form).
fn resolve_static_fields(
    insns: &mut [Insn],
    field_index_by_addr: &HashMap<u64, u32>,
    field_owner: &mut dyn FnMut(u32) -> Option<(String, String, ManagedType)>,
    stats: &mut ResolveStats,
) {
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Move) {
            continue;
        }
        // The slot address reaches here two ways: an `adrp` page or a folded
        // constant as an `Immediate`, or the x64 RIP-relative load the lifter
        // keeps as `Memory { base: abs, addend: <address> }`. Accepting only
        // `Immediate` left every static-field access on an x64 build unnamed —
        // the v24.5 dump renders them all as `*(nint*)(v0 +9909760)`.
        let Some(a) = insns[i]
            .operands
            .get(1)
            .and_then(|s| absolute_address(insns, i, s))
        else {
            continue;
        };
        let a = a as i64;
        let Some(d) = insns[i].dest().and_then(|d| d.as_local()) else {
            continue;
        };
        // An `adrp` page is not an address — the static field's slot is the
        // page PLUS the constant folded in at the use site. Naming the page
        // named a neighbouring field. See `page_base_offset_uses`.
        let page_uses = is_page_materialization(insns, i)
            .then(|| page_base_offset_uses(insns, i, d))
            .flatten();
        if let Some(uses) = page_uses {
            for (j, k) in uses {
                let Some(&fi) = field_index_by_addr.get(&(a.wrapping_add(k) as u64)) else {
                    continue;
                };
                let Some((tname, fname, fty)) = field_owner(fi) else {
                    continue;
                };
                let Some(w) = insns[j].dest().and_then(|x| x.as_local()) else {
                    continue;
                };
                let addr = insns[j].address;
                insns[j] = Insn::new(
                    Op::Move,
                    vec![
                        Operand::Local(w),
                        Operand::Field(FieldRef {
                            type_def_index: -1,
                            field_index: fi,
                            name: fname,
                            declaring_type_name: tname.clone(),
                            field_type: fty,
                            is_static: true,
                        }),
                    ],
                    addr,
                );
                stats.statics_resolved += 1;
            }
            continue;
        }
        let Some(&fi) = field_index_by_addr.get(&(a as u64)) else {
            continue;
        };
        let Some((tname, fname, fty)) = field_owner(fi) else {
            continue;
        };
        insns[i].operands[1] = Operand::Field(FieldRef {
            type_def_index: -1,
            field_index: fi,
            name: fname,
            declaring_type_name: tname.clone(),
            field_type: fty,
            is_static: true,
        });
        stats.strings_resolved += 0;
        stats.calls_resolved += 0;
        stats.ctors_fused += 0; // counter reuse avoided; tracked via strings
        stats.statics_resolved += 1;
    }
}

/// Literal-slot address materialized in a register/immediate → StringLiteral.
fn resolve_string_accessors(insns: &mut [Insn], maps: &ResolveMaps, stats: &mut ResolveStats) {
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Move) {
            continue;
        }
        let src = insns[i].operands.get(1).cloned();
        let Some(src) = src else { continue };
        // See `resolve_static_fields`: on x64 the literal slot is loaded
        // RIP-relative, so the address is the `Memory`'s addend and a bare
        // `Immediate` match named no string literal on any PE build.
        let Some(addr) = absolute_address(insns, i, &src) else {
            continue;
        };
        // An `adrp` page is not an address: the literal slot is the page plus
        // the constant folded in at the use site. `StringLiteral` carries the
        // literal VALUE (and sets the destination type to String), so naming
        // the page here would put a string where an address belongs. Skip and
        // leave the honest page constant. See `page_base_offset_uses`.
        if is_page_materialization(insns, i)
            && insns[i]
                .dest()
                .and_then(|d| d.as_local())
                .is_some_and(|d| page_base_offset_uses(insns, i, d).is_some())
        {
            continue;
        }
        let Some(&idx) = maps.literal_index_by_addr.get(&addr) else {
            continue;
        };
        let value = maps.literals.get(&idx).cloned().unwrap_or_default();
        insns[i].operands[1] = Operand::StringLiteral {
            index: idx,
            value: std::sync::Arc::from(value.as_str()),
        };
        insns[i].dest_type = Some(super::types::ManagedType::String);
        stats.strings_resolved += 1;
    }
}

/// Direct calls: target → MethodRef; then re-slice raw register arguments
/// onto the concrete signature (Cpp2IL `RemapRawArguments`).
/// Engine (extern/InternalCall) implementations have exactly ONE metadata
/// wrapper whose compiled body branches directly to them — IL2CPP statically
/// resolves externs, so game call sites BL the engine function itself. A
/// target with exactly one DISTINCT wrapper is therefore nameable: the call
/// site invokes that method's implementation. Multi-wrapper targets are
/// shared generic helpers (one address, many instantiations) — ambiguous.
fn single_native_wrapper(maps: &ResolveMaps, target: u64) -> Option<MethodRef> {
    let ws = maps.native_wrappers.get(&target)?;
    if ws.is_empty() {
        return None;
    }
    let mut distinct: Vec<&MethodRef> = Vec::new();
    for w in ws {
        if w.sig.params.len() > 8 {
            // beyond x0-x7 ABI slots; positional slicing would misalign
            return None;
        }
        if !distinct.iter().any(|d| {
            d.declaring_type_name == w.declaring_type_name
                && d.name == w.name
                && d.sig.is_static == w.sig.is_static
        }) {
            distinct.push(w);
        }
    }
    if distinct.len() == 1 {
        Some((*distinct[0]).clone())
    } else {
        None
    }
}

/// How far back a call's own argument setup is searched. The hidden
/// `MethodInfo` argument is reloaded (an `adrp`+`add`/`ldr` pair, plus the
/// class-init guard that reloads the same slot) within a handful of
/// instructions of the call.
const CALL_ARG_WINDOW: usize = 40;

/// The instantiation a call site passes to a shared generic body.
///
/// IL2CPP compiles every *reference-type* instantiation of a generic method to
/// ONE body and passes the concrete `MethodInfo` as a hidden leading argument
/// — a `MethodRef` metadata usage, whose decoded value indexes `method_specs`
/// (Cpp2IL `MetadataUsage.AsGenericMethodRef`). The body's address therefore
/// names hundreds of instantiations and the callee alone cannot say which one
/// it is; the call site can, because it is the one that loads the slot.
///
/// Two shapes reach a slot, and both are accepted:
///
/// 1. **The load feeds one of this call's own arguments.** `want` is the set of
///    locals the arguments come from, closed backwards over the instructions
///    that produced them, so a `MethodInfo` loaded for some other call nearby
///    cannot answer for this one. An `adrp` page is only meaningful together
///    with the constant folded in at the use site, so a page counts only when
///    a use of it is itself one of those locals.
/// 2. **The nearest preceding load of a slot, carried untouched into the
///    call.** This is the shape the reference builds actually have, and it is
///    *not* a loosening for convenience: by the time this pass runs the call's
///    operand list has already been re-sliced onto the CANONICAL callee's
///    signature, and for a shared body that signature belongs to one of the
///    wrong instantiations — so the hidden `MethodInfo` argument is often not
///    in the list at all. The load survives: IL2CPP emits it immediately
///    before the call and its value goes straight into the call's argument
///    register, so nothing between the two reads it. Real site on the v24.5
///    build @0x436d4c: `Local(1354) ← [abs+0xC32A50]`, then `Call
///    Immediate(0x266b80)` — a body shared by FIVE instantiations — with
///    `Local(1354)` never read again in the whole function.
pub(super) fn spec_loaded_by_call(insns: &[Insn], at: usize, maps: &ResolveMaps) -> Option<u32> {
    let by_addr = &maps.method_spec_index_by_addr;
    if by_addr.is_empty() {
        return None;
    }
    let lo = at.saturating_sub(CALL_ARG_WINDOW);
    if let Some(spec) = spec_loaded_through_args(insns, at, lo, by_addr) {
        return Some(spec);
    }
    // (2) the load that is carried untouched into the call.
    for k in (lo..at).rev() {
        if !matches!(insns[k].op, Op::Move) {
            continue;
        }
        let Some(d) = insns[k].dest().and_then(|x| x.as_local()) else {
            continue;
        };
        let Some(src) = insns[k].operands.get(1) else {
            continue;
        };
        // A page is not an address; adding its folded constant belongs to (1),
        // where a use that feeds this call is evidence of what it passes.
        if is_page_materialization(insns, k) && page_base_offset_uses(insns, k, d).is_some() {
            continue;
        }
        let Some(addr) = absolute_address(insns, k, src) else {
            continue;
        };
        let Some(&spec) = by_addr.get(&addr) else {
            continue;
        };
        if insns[k + 1..at].iter().any(|n| n.use_locals().contains(&d)) {
            continue;
        }
        return Some(spec);
    }
    None
}

/// Shape (1) of [`spec_loaded_by_call`].
fn spec_loaded_through_args(
    insns: &[Insn],
    at: usize,
    lo: usize,
    by_addr: &HashMap<u64, u32>,
) -> Option<u32> {
    // locals the call's arguments come from, closed over the instructions that
    // define them (backwards, so a definition is always reached before a use)
    let mut want: Vec<u32> = Vec::new();
    for op in insns[at].operands.iter().skip(2) {
        op.uses(&mut want);
    }
    if want.is_empty() {
        return None;
    }
    for k in (lo..at).rev() {
        let Some(d) = insns[k].dest().and_then(|x| x.as_local()) else {
            continue;
        };
        if !want.contains(&d) {
            continue;
        }
        for u in insns[k].use_locals() {
            if !want.contains(&u) {
                want.push(u);
            }
        }
    }
    let feeds_call = |j: usize| {
        insns[j]
            .dest()
            .and_then(|x| x.as_local())
            .is_some_and(|l| want.contains(&l))
    };
    for k in (lo..at).rev() {
        let Some(d) = insns[k].dest().and_then(|x| x.as_local()) else {
            continue;
        };
        if !want.contains(&d) {
            continue;
        }
        let Some(addr) = insns[k]
            .operands
            .get(1)
            .and_then(|s| absolute_address(insns, k, s))
        else {
            continue;
        };
        let page_uses = is_page_materialization(insns, k)
            .then(|| page_base_offset_uses(insns, k, d))
            .flatten();
        match page_uses {
            // A complete address in the instruction itself.
            None => {
                if let Some(&spec) = by_addr.get(&addr) {
                    return Some(spec);
                }
            }
            // A page: the slot is the SUM, and only a use that feeds this
            // call is evidence of what the call passes.
            Some(uses) => {
                for (j, off) in uses {
                    if !feeds_call(j) {
                        continue;
                    }
                    if let Some(&spec) = by_addr.get(&addr.wrapping_add(off as u64)) {
                        return Some(spec);
                    }
                }
            }
        }
    }
    None
}

/// The absolute address a load source encodes, when it encodes one.
///
/// Two spellings reach the metadata-usage slots:
///   * the `adrp` page (or the folded constant) — an `Immediate`;
///   * the x64 RIP-relative access, which the lifter keeps as
///     `Memory { base: abs, index: None, addend: absolute }`. `abs` is an
///     ordinary local by the time SSA has materialized registers into
///     versions, and a class pointer folded into a base appears as an
///     `Immediate`, so both are accepted as "the address is the addend".
///
/// The `Local` base is accepted ONLY when that local has no definition in
/// this function. That is exactly the `abs`/`rip` pseudo-register (an ENTRY
/// value — nothing ever writes it), and it is what separates "the addend is
/// the address" from "the addend is an offset from a COMPUTED pointer".
/// Accepting every local made `[r1 + 0xD30]` — where `r1` holds a computed
/// address — resolve as if `0xD30` alone were a slot address: on the ARM32
/// corpus that rewrote an `il2cpp_codegen_initialize_method` byte-flag guard
/// into `if (MethodInfo("…") != 0)`.
fn absolute_address(insns: &[Insn], at: usize, src: &Operand) -> Option<u64> {
    match src {
        Operand::Immediate(a) => Some(*a as u64),
        Operand::Memory {
            base,
            index: None,
            addend,
            ..
        } => match base.as_ref() {
            Operand::Register(r) if r == "abs" || r == "rip" => Some(*addend as u64),
            Operand::Local(v) if ssa_def_of(insns, at, *v).is_none() => Some(*addend as u64),
            Operand::Immediate(b) => Some((*b as u64).wrapping_add(*addend as u64)),
            _ => None,
        },
        _ => None,
    }
}

/// Pass 1 of call resolution: a direct call's immediate target → the managed
/// method it names.
///
/// The ARGUMENTS are deliberately left alone here. Fixing the argument list
/// means fixing how many arguments the call has, and for a call to a FOLDED
/// address that number belongs to the canonical member of the merged set rather
/// than to the call — so slicing here destroyed the call site's own ABI shape
/// (whose slot 0 is the receiver) before the pass that has to pick a member
/// could look at it. `reslice_call_arguments` does it afterwards, with the
/// member that this pass and `resolve_folded_names` settled on.
fn resolve_call_targets(insns: &mut [Insn], maps: &ResolveMaps, stats: &mut ResolveStats) {
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Call) {
            continue;
        }
        let Some(Operand::Immediate(target)) = insns[i].operands.first().cloned() else {
            continue;
        };
        let m = match maps.methods_by_rva.get(&(target as u64)).cloned() {
            Some(default) => {
                // Shared generic body: `methods_by_rva` holds the canonical
                // instantiation. Where the call site loads the `MethodInfo` it
                // is passing, that is not a tie-break — it is the answer.
                maps.specs_by_body_rva
                    .get(&(target as u64))
                    .and_then(|cands| {
                        let spec = spec_loaded_by_call(insns, i, maps)?;
                        if let Some((_, m)) = cands.iter().find(|(idx, _)| *idx == spec) {
                            return Some(m.clone());
                        }
                        // The slot names a spec the POINTER table does not
                        // index — every instantiation in `specs_by_body_rva`
                        // got here through a pointer, so a spec without one
                        // cannot be among them even when its code IS this
                        // address. `method_definition_index` is the link that
                        // survives: only accept the spec when the called
                        // address already carries that same definition, which
                        // is what keeps an unrelated `MethodInfo` load standing
                        // next to the call from answering for it.
                        let m = maps.specs_by_index.get(&spec)?;
                        cands
                            .iter()
                            .any(|(_, c)| c.method_index == m.method_index)
                            .then(|| m.clone())
                    })
                    .map(|m| {
                        stats.generic_bodies_pinned += 1;
                        m
                    })
                    .unwrap_or(default)
            }
            None => {
                // Native-wrapper fallback: `fn_4a58bac`-style engine calls
                // whose implementation has exactly one metadata wrapper. On
                // this binary the hot engine targets are SHARED generic
                // trampolines (thousands of wrappers each) so the fallback
                // rarely fires — it is exact when it does (single wrapper =
                // that method's own compiled extern body).
                match single_native_wrapper(maps, target as u64) {
                    Some(m) => m,
                    None => {
                        stats.calls_unresolved += 1;
                        continue;
                    }
                }
            }
        };

        insns[i].operands[0] = Operand::Method(m.clone());
        insns[i].dest_type = Some(m.sig.ret.clone());
        stats.calls_resolved += 1;
    }
}

/// Pass 2: re-slice a resolved call's argument list onto its callee's declared
/// signature (Cpp2IL `RemapRawArguments`).
///
/// Coming in, the operands are the argument slots `locals::align_call_arguments`
/// placed by REGISTER CLASS, plus the destination slot: `[target, dest,
/// position…]`. Coming out they are `[Method, dest, this?, param…]`, one operand
/// per declared parameter, in order.
///
/// The reason this runs LAST is the FOLDED case. `align_call_arguments` had to
/// pick a signature, and for a folded address `methods_by_rva` holds one
/// arbitrary member of the merged set — so the slot layout it left behind
/// describes a method the call may not be. Slicing against that layout and then
/// discovering which member the call really means leaves the argument list
/// shaped by the WRONG signature (measured: `OnDeserialization(object)` with its
/// parameter silently dropped because the arbitrary member took none). Every
/// pass that reads a call's arguments therefore runs first, `resolve_folded_names`
/// settles the member, and only then is the list rebuilt.
/// Every local whose storage is addressed by an operand (`&local`).
fn collect_addressed_locals(insns: &[Insn]) -> std::collections::HashSet<u32> {
    fn walk(op: &Operand, out: &mut std::collections::HashSet<u32>) {
        match op {
            Operand::AddressOf(inner) => {
                match inner.as_ref() {
                    Operand::Local(v) => {
                        out.insert(*v);
                    }
                    Operand::Memory { base, .. } => {
                        if let Operand::Local(v) = base.as_ref() {
                            out.insert(*v);
                        }
                    }
                    _ => {}
                }
                walk(inner, out);
            }
            Operand::ArrayLength(inner) => walk(inner, out),
            Operand::Memory { base, index, .. } => {
                walk(base, out);
                if let Some(i) = index {
                    walk(i, out);
                }
            }
            Operand::ArrayAccess { array, index } => {
                walk(array, out);
                walk(index, out);
            }
            Operand::InstanceField { base, .. } => walk(base, out),
            _ => {}
        }
    }
    let mut out = std::collections::HashSet::new();
    for insn in insns {
        for op in &insn.operands {
            walk(op, &mut out);
        }
    }
    out
}

fn reslice_call_arguments(insns: &mut [Insn], addressed: &std::collections::HashSet<u32>) {
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Call) {
            continue;
        }
        let Some(Operand::Method(m)) = insns[i].operands.first().cloned() else {
            continue;
        };
        //
        // `resolve_through_moves` is a BEST-EFFORT tracer: it returns `None`
        // whenever it cannot follow a value to its source — a `Phi` at a join,
        // a load, an arithmetic definition. That is a completely different fact
        // from "the ABI carried no register for this slot", and conflating the
        // two deleted live operands: the receiver of every virtual call whose
        // receiver local is a join merge was replaced with a hole, so
        // `Lazy<object>.get_Value` rendered `__abi.Value` even though the IR
        // held a perfectly good `Local(20)` defined by the phi. Precedence is
        // therefore: what the tracer found, else the UNTOUCHED original, else
        // `default_for` for a slot that genuinely has no operand at all.
        let orig: Vec<Operand> = insns[i].operands.iter().skip(2).cloned().collect();
        let raw: Vec<Option<Operand>> = orig
            .iter()
            .map(|op| {
                Some(match resolve_through_moves(insns, i, op, Some(addressed)) {
                    // a tracer that gave up, or came back empty-handed, must
                    // never overwrite a live operand
                    None | Some(Operand::Unknown) => op.clone(),
                    Some(v) => v,
                })
            })
            .collect();

        let mut new_ops = Vec::with_capacity(m.sig.params.len() + 3);
        new_ops.push(Operand::Method(m.clone()));
        // result position (Call only)
        new_ops.push(
            insns[i]
                .operands
                .get(1)
                .cloned()
                .unwrap_or(Operand::Unknown),
        );

        let mut cursor = 0usize;
        if !m.sig.is_static {
            // this = first reference-typed arg register holding a class local.
            // A missing receiver is an ABI fact — mark it, never fake a value.
            let this_op = raw.first().and_then(|o| o.clone());
            new_ops.push(this_op.unwrap_or_else(|| {
                default_for(&super::types::ManagedType::Class(
                    super::types::NamedType::object(),
                ))
            }));
            cursor += 1;
        }
        for pty in &m.sig.params {
            let op = raw
                .get(cursor)
                .and_then(|o| o.clone())
                .unwrap_or_else(|| default_for(pty));
            new_ops.push(op);
            cursor += 1;
        }

        // AAPCS64 indirect-result slot (x8), kept across this rebuild for the
        // same reason as in `align_call_arguments` / `trim_call_arguments`:
        // this list is rebuilt positionally by SIGNATURE, but the slot index is
        // an ABI position the renderer reads back, so it must be restored at
        // that index rather than dropped with the register dump.
        let sret = insns[i]
            .operands
            .get(super::passes::SRET_OPERAND)
            .filter(|o| {
                !matches!(
                    o,
                    Operand::Unknown | Operand::MissingAbiArg | Operand::Register(_)
                )
            })
            .cloned();
        super::passes::place_sret(&mut new_ops, sret);
        insns[i].operands = new_ops;
        insns[i].dest_type = Some(m.sig.ret.clone());
    }
}

// ---------------------------------------------------------------------------
// ABI argument recovery: filling `__abi` holes from the call's own ABI
// ---------------------------------------------------------------------------
//
// `locals::scrub_abi_call_args` replaces any call argument that is not a
// "rooted" local with `Operand::MissingAbiArg`, which the renderer prints as
// `__abi` — 51,439 of them on the x64 PE corpus (47,924 methods), the largest
// defect family after the `v*` locals. The value is not necessarily lost: the
// CALLER wrote it into the ABI's argument register, and that write is still in
// the IR.
//
// This pass re-derives the value from two facts that already exist in the repo:
//
//   * the lifters name argument registers POSITIONALLY on every architecture —
//     `x{i}` for the integer class and `vf{i}`/`vs{i}`/`vd{i}`… for the FP class
//     (`lift_x86::gpr_name_for_ordinal` states the contract: "argument register
//     *i* takes the name `x{i}` whatever physical register the ABI puts it in,
//     so every downstream pass speaks one positional vocabulary across ARM64,
//     ARM32, x64 SysV and x64 MSVC");
//   * `LocalTable.infos[..].slot_origin` records which register a local came
//     from, so "the value in argument register C at this call" is "the nearest
//     preceding definition of a local whose slot origin is C".
//
// The gates, in the order they refuse:
//
//   * the parameter's class (`float`/`double` -> FP) decides which register file
//     the position uses, and the integer and FP positions are counted
//     SEPARATELY — every ABI here allocates the two classes independently;
//   * a position past that class's register capacity is NOT recovered. The
//     stack-argument model is a separate mechanism (`stack_{n}` locals assigned
//     by `stack_analyzer` from frame offsets, not from argument index) and
//     guessing its slot arithmetic would fabricate an argument;
//   * a CALL between the definition and this call means the carrier was
//     clobbered — every argument register is caller-saved under both AAPCS64
//     and the x64 conventions: REFUSE;
//   * the definition must sit within `ABI_LIVE_WINDOW` instructions, so a def
//     from far earlier in the method is not read as "still live";
//   * a definition whose SOURCE is itself an argument register is the
//     register-rename shape (`mov r0, r4 ; bl ChangeWeapon`) whose SSA copy can
//     have been coalesced away — precisely the shape that made
//     `align_call_arguments` hand the receiver to the wrong parameter and
//     render `__abi.ChangeWeapon(slotType, weapon, weaponTexture, this)`: REFUSE;
//   * a known type of the wrong CLASS refuses (`float` filled from an integer
//     register and vice versa).
//
// Everything else stays `__abi`: accurate-but-ugly, and always better than a
// fabricated argument.

/// `__abi` holes examined and filled, process-wide — so "ABI recovery did
/// nothing" is a number instead of an assumption.
static ABI_HOLES_SEEN: AtomicU64 = AtomicU64::new(0);
static ABI_HOLES_FILLED: AtomicU64 = AtomicU64::new(0);

/// `(examined, filled)` `__abi` argument holes.
pub fn abi_recovery_counts() -> (u64, u64) {
    (
        ABI_HOLES_SEEN.load(Ordering::Relaxed),
        ABI_HOLES_FILLED.load(Ordering::Relaxed),
    )
}

/// `(attempted, hit)` counters for the two vtable lookup fallbacks — so
/// "interface dispatch did nothing" is a pair of numbers instead of an
/// assumption.
///
/// A bare hit count cannot distinguish NEVER-ATTEMPTED (the call site never
/// reached a table: its klass type was unknown or its slot was not a constant,
/// so no table could key) from ATTEMPTED-MISSED (the table was consulted and
/// held no entry for that `(tdi, slot)`). The two want opposite fixes — a
/// typing/slot-shape gap versus a table-coverage gap — so each fallback keeps
/// both numbers:
///
/// * `VTABLE_*`: `lookup_vtable_slot` calls (the class's own ancestry). An
///   attempt means the slot was a constant on the vtable grid AND the klass
///   resolved to a TypeDef; a hit means that `(tdi, slot)` was declared
///   somewhere in the ancestry.
/// * `IFACE_*`: `lookup_interface_slot` calls (the interface-dispatch table).
///   It runs ONLY after the class lookup missed, so its attempts count exactly
///   the class-misses with a known `tdi`; a hit means the interface table
///   placed that slot.
///
/// On the iOS arm64 corpus the interface table holds 50,166 entries yet
/// `virtual_calls` stays 0: if `VTABLE_ATTEMPTS`/`IFACE_ATTEMPTS` stay near
/// zero while `timing::VC_MISSES` counts thousands, the residue is
/// never-attempted (the sampled `klass + idx*16 + 312` sites compute the slot
/// at runtime through an `Arith` chain, so the `Memory` shape this resolver
/// keys on never forms and no table is ever consulted) — not a table miss.
///
/// Counting is inert on well-formed IR: the counters are process-global
/// `AtomicU64` increments that do not read, write, or reorder any `Insn`,
/// `Operand`, or table entry. Each `lookup_*` returns exactly what the
/// uncounted version returned for the same `(tdi, slot)` (the increment
/// precedes the map probe and never early-returns), and the call order
/// (class first, interface only on class miss) is unchanged — so the emitted
/// C# is byte-identical with or without the counts; only the `ILR_VERIFY`
/// report gains lines.
static VTABLE_LOOKUP_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static VTABLE_LOOKUP_HITS: AtomicU64 = AtomicU64::new(0);
static IFACE_LOOKUP_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static IFACE_LOOKUP_HITS: AtomicU64 = AtomicU64::new(0);

/// `(attempted, hit)` delegate `invoke_impl` lookups — so "delegate recovery
/// did nothing" is a pair of numbers instead of an assumption.
///
/// An attempt means the call's target definition proved to be an `invoke_impl`
/// field load (`InstanceField` off a delegate-typed base, or a bare `Field`
/// when `this` is delegate-typed); a hit means the base's delegate type was
/// proven and an `Invoke` was synthesized. A bare `Field`/`InstanceField`
/// count cannot distinguish "the base was never proven delegate" (typing gap)
/// from "the shape was never an invoke_impl load" — so the unproven bucket is
/// kept separately (see [`delegate_refused_unproven`]).
///
/// Counting is inert on well-formed IR: process-global `AtomicU64` increments
/// that never read, write, or reorder any `Insn`, `Operand`, or table entry.
static DELEGATE_LOOKUP_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static DELEGATE_LOOKUP_HITS: AtomicU64 = AtomicU64::new(0);
/// `invoke_impl` field loads whose delegate base could not be proven (untyped
/// base, or a bare `Field` with no delegate-typed `this`). Kept as `fnptr_`,
/// never guessed.
static DELEGATE_REFUSED_UNPROVEN: AtomicU64 = AtomicU64::new(0);

/// `(attempted, hit)` `Arith`-base vtable slot computations — the
/// `klass + idx*16 + 312` loop shape whose slot is computed at runtime.
///
/// An attempt means the call's slot load base proved to be an `Arith` address
/// computation (`v60 = v59 + 312`); a hit means the whole chain folded to a
/// constant (`klass + const`) whose slot is on the vtable grid AND whose klass
/// resolved. Every other shape refuses with its own bucket below, so a hit
/// count of zero against thousands of attempts is the measurement that the
/// residue is genuinely dynamic — not a table miss.
///
/// Counting is inert: increments only, no IR change, no early-return.
static ARITH_SLOT_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static ARITH_SLOT_HITS: AtomicU64 = AtomicU64::new(0);
/// The `Arith` chain contains a dynamic (non-constant) offset — `klass +
/// scaled-idx`, a `Shl`/`Mul`, or two locals added. No constant slot exists
/// to key, so guessing one would invent a callee.
static ARITH_SLOT_REFUSED_DYNAMIC: AtomicU64 = AtomicU64::new(0);
/// The `Arith` base chain has no definition (entry value with no SSA def).
static ARITH_SLOT_REFUSED_NO_DEF: AtomicU64 = AtomicU64::new(0);
/// The `Arith` chain folded to `klass + const` but the klass resolved to no
/// TypeDef (unknown immediate klass, or an untyped receiver).
static ARITH_SLOT_REFUSED_UNKNOWN_KLASS: AtomicU64 = AtomicU64::new(0);
/// The folded constant is not on the vtable grid (`(off - base) % entry !=
/// 0`, e.g. the `MethodInfo*` half at `+8`).
static ARITH_SLOT_REFUSED_BAD_OFFSET: AtomicU64 = AtomicU64::new(0);

/// `Memory` slot loads with a register index (`[klass + idx*scale + off]`) —
/// the indexed spelling of the same dynamic-slot shape. No constant slot
/// exists, so the site stays `fnptr_`, never guessed. Inert counter.
static INDEXED_SLOT_REFUSALS: AtomicU64 = AtomicU64::new(0);

/// Call targets with no SSA definition at all (entry values, `ICall`
/// leftovers). Inert counter — the site stays `fnptr_`.
static VC_NO_DEF_SITES: AtomicU64 = AtomicU64::new(0);

/// Slot loads whose klass is an `Immediate` naming no TypeDef (a klass
/// pointer the `klass_to_tdi` map does not cover). Inert counter.
static VC_UNKNOWN_KLASS_IMM_SITES: AtomicU64 = AtomicU64::new(0);

/// `(attempted, hit)` class-vtable lookups.
pub fn vtable_lookup_counts() -> (u64, u64) {
    (
        VTABLE_LOOKUP_ATTEMPTS.load(Ordering::Relaxed),
        VTABLE_LOOKUP_HITS.load(Ordering::Relaxed),
    )
}

/// `(attempted, hit)` interface-dispatch fallback lookups.
pub fn iface_lookup_counts() -> (u64, u64) {
    (
        IFACE_LOOKUP_ATTEMPTS.load(Ordering::Relaxed),
        IFACE_LOOKUP_HITS.load(Ordering::Relaxed),
    )
}

/// `(attempted, hit)` delegate `invoke_impl` lookups.
pub fn delegate_lookup_counts() -> (u64, u64) {
    (
        DELEGATE_LOOKUP_ATTEMPTS.load(Ordering::Relaxed),
        DELEGATE_LOOKUP_HITS.load(Ordering::Relaxed),
    )
}

/// `invoke_impl` loads refused for lack of a proven delegate base.
pub fn delegate_refused_unproven() -> u64 {
    DELEGATE_REFUSED_UNPROVEN.load(Ordering::Relaxed)
}

/// `(attempted, hit)` `Arith`-base slot computations.
pub fn arith_slot_counts() -> (u64, u64) {
    (
        ARITH_SLOT_ATTEMPTS.load(Ordering::Relaxed),
        ARITH_SLOT_HITS.load(Ordering::Relaxed),
    )
}

/// `(dynamic, no-def, unknown-klass, bad-offset)` `Arith`-base refusals.
pub fn arith_slot_refusals() -> (u64, u64, u64, u64) {
    (
        ARITH_SLOT_REFUSED_DYNAMIC.load(Ordering::Relaxed),
        ARITH_SLOT_REFUSED_NO_DEF.load(Ordering::Relaxed),
        ARITH_SLOT_REFUSED_UNKNOWN_KLASS.load(Ordering::Relaxed),
        ARITH_SLOT_REFUSED_BAD_OFFSET.load(Ordering::Relaxed),
    )
}

/// Indexed (`[klass + idx*scale + off]`) dynamic-slot refusals.
pub fn indexed_slot_refusals() -> u64 {
    INDEXED_SLOT_REFUSALS.load(Ordering::Relaxed)
}

/// Targets with no SSA definition.
pub fn vc_no_def_count() -> u64 {
    VC_NO_DEF_SITES.load(Ordering::Relaxed)
}

/// Slot loads whose immediate klass names no TypeDef.
pub fn vc_unknown_klass_imm_count() -> u64 {
    VC_UNKNOWN_KLASS_IMM_SITES.load(Ordering::Relaxed)
}

/// How far back the definition of an argument carrier is searched for.
///
/// The caller computes an argument immediately before the call; the window only
/// has to cover the register dump, the class-init guard and the argument setup
/// that precede it.
const ABI_SEARCH_WINDOW: usize = 64;

/// How far back a definition may sit and still be trusted as "the value still
/// in the register". Narrower than the search window on purpose: a def at the
/// far end of a 64-instruction window is far more likely to be a stale write
/// whose register was later reused through a copy SSA coalesced away.
const ABI_LIVE_WINDOW: usize = 16;

/// Argument registers per class, per ABI. FP is 8 everywhere here (x86 has
/// `xmm0`-`xmm15` and both ARM profiles have 8 argument FP registers); the
/// integer capacity is detected from the dump, because this file serves
/// AAPCS64 (8), x64 SysV (6), x64 MSVC (4) and ARM32 (4) and the modes cannot
/// be told apart from the instruction stream alone (see
/// `detect_int_arg_capacity`).
const FP_ARG_CAPACITY: usize = 8;

/// The register class an argument position is carried in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArgClass {
    Int,
    Fp,
}

/// Is this a float/double parameter — the FP register class?
///
/// Only the scalar primitives are classified. An HFA aggregate (`Vector3`, a
/// 12-byte struct of floats) is passed in consecutive FP registers too, but
/// deciding that needs the type's FIELD layout, which is not in this table; a
/// struct is therefore classified as `Int`, and the type-class gate below then
/// refuses the fill when the carrier's own type says otherwise. Missing an HFA
/// argument is a refusal, not a wrong answer.
fn is_fp_param(t: &ManagedType) -> bool {
    matches!(
        t,
        ManagedType::Primitive(super::types::PrimKind::R4)
            | ManagedType::Primitive(super::types::PrimKind::R8)
    )
}

/// The FP argument POSITION a SIMD register name denotes, if it is one.
///
/// One physical register is spelled at several widths by the lifters:
/// `vf0` (x86 SSE/AVX), and `vb0`/`vh0`/`vs0`/`vd0`/`vq0` (AArch64 lane views)
/// and `vs0`/`vd0` (ARM32). The digits are the SIMD register index, which is
/// the FP argument position under every one of those ABIs.
fn fp_register_slot(origin: &str) -> Option<usize> {
    let bytes = origin.as_bytes();
    if bytes.len() < 3 || bytes[0] != b'v' {
        return None;
    }
    if !matches!(bytes[1], b'f' | b'b' | b'h' | b's' | b'd' | b'q') {
        return None;
    }
    let digits = &origin[2..];
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The integer argument position an `x{i}` register name denotes.
fn int_register_slot(origin: &str) -> Option<usize> {
    if !is_int_arg_register(origin) {
        return None;
    }
    origin[1..].parse().ok()
}

/// Any argument register, either class — used to recognise the rename/
/// coalescing shape that must not be filled through.
fn is_any_arg_register(origin: &str) -> bool {
    int_register_slot(origin).is_some() || fp_register_slot(origin).is_some()
}

/// The slot origin of a local version, when the table knows it.
fn local_origin<'a>(table: &'a LocalTable, v: u32) -> Option<&'a str> {
    let li = table.index_of.get(&v)?;
    table.infos.get(*li).map(|i| i.slot_origin.as_str())
}

/// Does `origin` carry argument position `slot` of `class`?
fn origin_is_carrier(origin: &str, class: ArgClass, slot: usize) -> bool {
    match class {
        ArgClass::Int => int_register_slot(origin) == Some(slot),
        ArgClass::Fp => fp_register_slot(origin) == Some(slot),
    }
}

/// How many integer argument registers this dump's ABI uses.
///
/// Detected from the register names the lifters actually created locals for:
/// a function that passes six arguments writes `x5`, one that passes eight
/// writes `x7`. The guess can only be too LOW (a method that never fills its
/// last argument register), and too low can only cost a refusal — a position
/// past capacity is never filled.
fn detect_int_arg_capacity(table: &LocalTable) -> usize {
    let mut cap = 4usize;
    for info in &table.infos {
        if let Some(slot) = int_register_slot(&info.slot_origin) {
            cap = cap.max(slot + 1);
        }
    }
    cap.min(8)
}

/// The FP-ness of a value, when the IR states one. `None` = not known, which is
/// a refusal rather than a guess.
fn value_is_fp(insns: &[Insn], table: &LocalTable, at: usize, v: u32) -> Option<bool> {
    if let Some(t) = insns.get(at).and_then(|i| i.dest_type.as_ref()) {
        return Some(is_fp_param(t));
    }
    let li = *table.index_of.get(&v)?;
    match &table.infos.get(li)?.ty {
        ManagedType::Unknown => None,
        other => Some(is_fp_param(other)),
    }
}

/// The value provably sitting in `class`/`slot` at instruction `at`, or `None`
/// when that cannot be proven (see the module section above for each gate).
fn carrier_value_at(
    insns: &[Insn],
    table: &LocalTable,
    at: usize,
    class: ArgClass,
    slot: usize,
    want_fp: bool,
) -> Option<Operand> {
    let lo = at.saturating_sub(ABI_SEARCH_WINDOW);
    for k in (lo..at).rev() {
        let Some(d) = insns[k].dest().and_then(|x| x.as_local()) else {
            continue;
        };
        let Some(origin) = local_origin(table, d) else {
            continue;
        };
        if !origin_is_carrier(origin, class, slot) {
            continue;
        }
        // The nearest definition of the carrier is the reaching one — but only
        // if nothing in between could have clobbered it. Every argument register
        // is caller-saved, so a call in between ends the proof.
        if insns[k + 1..at]
            .iter()
            .any(|n| matches!(n.op, Op::Call | Op::CallVoid | Op::IndirectCall))
        {
            return None;
        }
        if at - k > ABI_LIVE_WINDOW {
            return None;
        }
        let value = match insns[k].operands.get(1).cloned() {
            // `mov x1, x5` — a rename. The SSA copy pass coalesces these, which
            // means the local that really carried the value has the OTHER
            // register's origin, and reading this definition instead hands the
            // call a stale value. Refuse.
            Some(Operand::Local(w)) if local_origin(table, w).is_some_and(is_any_arg_register) => {
                return None;
            }
            Some(src)
                if matches!(
                    src,
                    Operand::MissingAbiArg | Operand::Unknown | Operand::Register(_)
                ) =>
            {
                return None;
            }
            Some(src) => src,
            // A producing instruction (a call's result, an arithmetic op) wrote
            // the carrier directly: the value IS that local.
            None => Operand::Local(d),
        };
        // Class agreement, when either end states a type.
        if let Some(is_fp) = value_is_fp(insns, table, k, d) {
            if is_fp != want_fp {
                return None;
            }
        }
        if let Operand::Local(w) = &value {
            if let Some(is_fp) = value_is_fp(insns, table, k, *w) {
                if is_fp != want_fp {
                    return None;
                }
            }
        }
        return Some(value);
    }
    None
}

/// Fill the `__abi` holes whose value is provably still in its argument
/// register (see the section above). Returns how many were filled.
pub fn recover_abi_arguments(insns: &mut [Insn], table: &LocalTable) -> usize {
    let int_capacity = detect_int_arg_capacity(table);
    let mut filled = 0usize;
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Call | Op::CallVoid) {
            continue;
        }
        let Some(Operand::Method(m)) = insns[i].operands.first().cloned() else {
            continue;
        };
        // Position -> (class, slot) for every argument operand. The list is
        // `[target, dest, this?, param…]`, the layout `align_call_arguments`
        // produced and `reslice_call_arguments` consumes.
        let mut plan: Vec<Option<(ArgClass, usize, bool)>> = vec![None, None];
        let mut int_used = 0usize;
        let mut fp_used = 0usize;
        if !m.sig.is_static {
            plan.push(Some((ArgClass::Int, int_used, false)));
            int_used += 1;
        }
        for pty in &m.sig.params {
            if is_fp_param(pty) {
                plan.push(Some((ArgClass::Fp, fp_used, true)));
                fp_used += 1;
            } else {
                plan.push(Some((ArgClass::Int, int_used, false)));
                int_used += 1;
            }
        }
        for p in 2..insns[i].operands.len() {
            let Some(&Some((class, slot, want_fp))) = plan.get(p) else {
                continue;
            };
            let hole = insns[i]
                .operands
                .get(p)
                .is_some_and(|o| matches!(o, Operand::MissingAbiArg | Operand::Unknown));
            if !hole {
                continue;
            }
            let capacity = match class {
                ArgClass::Int => int_capacity,
                ArgClass::Fp => FP_ARG_CAPACITY,
            };
            if slot >= capacity {
                continue;
            }
            ABI_HOLES_SEEN.fetch_add(1, Ordering::Relaxed);
            if let Some(v) = carrier_value_at(insns, table, i, class, slot, want_fp) {
                insns[i].operands[p] = v;
                ABI_HOLES_FILLED.fetch_add(1, Ordering::Relaxed);
                filled += 1;
            }
        }
    }
    filled
}

fn default_for(t: &super::types::ManagedType) -> Operand {
    // A reference-typed parameter whose operand was not recovered from the
    // call site is a POSITION fact, not an unknown value: the ABI carries no
    // live register for it here. `Operand::Unknown` conflated that with "the
    // lifter lost this", and — because this pass runs AFTER
    // `locals::scrub_abi_call_args` — the substituted slot was never scrubbed
    // either, so it surfaced in the output as another `__unknown`.
    if t.is_reference_like() {
        Operand::MissingAbiArg
    } else {
        Operand::Immediate(0)
    }
}

/// Trace an operand back through Moves to its most-resolved form.
fn resolve_through_moves(
    insns: &[Insn],
    at: usize,
    op: &Operand,
    addressed: Option<&std::collections::HashSet<u32>>,
) -> Option<Operand> {
    match op {
        Operand::Register(name) => {
            // trace register defs backwards; follow Moves AND call results
            // (receiver often = result of the preceding managed call)
            for insn in insns[..at].iter().rev().take(96) {
                let Some(dest) = insn.dest().cloned() else {
                    continue;
                };
                match dest {
                    Operand::Local(_) => {}
                    Operand::Register(ref n) if *n == *name => {
                        match insn.op {
                            Op::Move => return insn.operands.get(1).cloned(),
                            Op::Call => {
                                // value produced by the preceding call: KEEP the
                                // register reference — SSA renaming + def-map will inline
                                // the producing call expression at render time.
                                return Some(Operand::Register(name.clone()));
                            }
                            _ => return None,
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        Operand::Local(v) => {
            // An addressed cell is mutated through its address (an `out`
            // parameter write). Its last textual definition is NOT its value
            // at this use: 0x18A3A1C traced `TryGetValue(key, &cell)`'s out
            // cell back to the `cell = 0` initialiser and passed 0 where the
            // decoded token belongs.
            if addressed.is_some_and(|a| a.contains(v)) {
                return None;
            }
            for insn in insns[..at].iter().rev().take(96) {
                if let Some(Operand::Local(dv)) = insn.dest() {
                    if *dv == *v {
                        return match insn.op {
                            Op::Move => insn.operands.get(1).cloned(),
                            // value defined by an earlier resolved call:
                            // keep the local — the C# def-map inlines the
                            // producing call expression automatically.
                            Op::Call => Some(op.clone()),
                            _ => None,
                        };
                    }
                }
            }
            None
        }
        other => Some(other.clone()),
    }
}

/// Is this instruction's source a bare address CONSTANT — the
/// `adrp`/`movw+movt` PAGE materialization [`page_base_offset_uses`] is written
/// for?
///
/// A `Memory` source is a VALUE load (`[base + off]`), not a page, and running
/// the page fixup on one treats the load's own addend as a page address and
/// then ADDS the next instruction's constant to it — so on the ARM32 corpus
/// `add rX, #0x1609000` resolved as if `0x1609000` alone were a usage-slot
/// address, and an `il2cpp_codegen_initialize_method` guard (a byte-flag read)
/// was rewritten into `if (MethodInfo("…") != 0)`.
fn is_page_materialization(insns: &[Insn], at: usize) -> bool {
    matches!(insns[at].operands.get(1), Some(Operand::Immediate(_)))
}

/// Does this instruction read local `v` anywhere in its operands?
///
/// Every variant is listed EXPLICITLY. The old body ended in `_ => false` and
/// so could not see a local used as an `InstanceField` receiver or as an
/// `ArrayAccess` base/index. `page_base_offset_uses` is the only caller and it
/// uses this answer to decide whether a page materialization is "only ever used
/// with a constant offset": a use it cannot see is a use it treats as absent, so
/// the page was resolved as a complete address and the `#lo` became an offset
/// into an unrelated slot. Adding an Operand variant must break this match at
/// compile time rather than silently answer "no".
fn operand_reads_local(op: &Operand, v: u32) -> bool {
    match op {
        Operand::Local(x) => *x == v,
        Operand::AddressOf(i) | Operand::ArrayLength(i) => operand_reads_local(i, v),
        Operand::Memory { base, index, .. } => {
            operand_reads_local(base, v)
                || index.as_deref().is_some_and(|i| operand_reads_local(i, v))
        }
        Operand::InstanceField { base, .. } => operand_reads_local(base, v),
        Operand::ArrayAccess { array, index } => {
            operand_reads_local(array, v) || operand_reads_local(index, v)
        }
        Operand::Register(_)
        | Operand::Immediate(_)
        | Operand::FloatLiteral(_)
        | Operand::DoubleLiteral(_)
        | Operand::StringLiteral { .. }
        | Operand::Method(_)
        | Operand::Field(_)
        | Operand::Type(_)
        | Operand::MissingAbiArg
        | Operand::Unknown => false,
    }
}

/// `adrp` materializes a PAGE, not an address.
///
/// IL2CPP's metadata-usage idiom is `adrp xN, page; add xN, xN, #lo` (or
/// `ldr xN, [xN, #lo]`), and the SLOT ADDRESS is the SUM. The page on its own
/// is very often the address of a *different* usage slot that happens to begin
/// that page, so matching the page named the wrong member and the `#lo` was
/// then rendered as an offset into it. Real site `DrawGizmo` @0x14048:
///
/// ```text
/// 0x140b0  adrp x0, 0x6091000    ; a PAGE, not a slot address
/// 0x140b4  add  x0, x0, #0xce0   ; the slot is 0x6091ce0
/// 0x140b8  bl   <init thunk>
/// ```
///
/// which rendered `*(nint*)(MethodInfo("KeyList.Clear") + 3296)` — the name of
/// whatever slot sits at 0x6091000, carrying an offset that belongs to a
/// different method entirely. 38,861 sites in the dump used a `MethodInfo` as
/// a pointer base from exactly this collision.
///
/// A page is only meaningful together with the constant that follows it, so
/// this returns the `(use instruction, constant)` pairs when EVERY use of the
/// local in a short window is a constant add/sub or a constant-offset load off
/// it. `None` means the value is used as a complete address after all, and the
/// caller may resolve it directly.
pub(super) fn page_base_offset_uses(
    insns: &[Insn],
    at: usize,
    page: u32,
) -> Option<Vec<(usize, i64)>> {
    use super::ir::ArithKind;
    let mut uses: Vec<(usize, i64)> = Vec::new();
    for (j, insn) in insns.iter().enumerate().skip(at + 1).take(64) {
        if !insn.operands.iter().any(|o| operand_reads_local(o, page)) {
            continue;
        }
        match &insn.op {
            // `add xN, xN, #lo` / `sub xN, xN, #lo`
            Op::Arith(k @ (ArithKind::Add | ArithKind::Sub)) => {
                let imm = insn.operands.iter().find_map(|o| match o {
                    Operand::Immediate(v) => Some(*v),
                    _ => None,
                })?;
                uses.push((
                    j,
                    if matches!(k, ArithKind::Sub) {
                        -imm
                    } else {
                        imm
                    },
                ));
            }
            // `ldr xN, [xN, #lo]` — the low bits folded into the addressing mode
            Op::Move => {
                let off = insn.operands.iter().find_map(|o| match o {
                    Operand::Memory {
                        base,
                        index: None,
                        addend,
                        ..
                    } if matches!(base.as_ref(), Operand::Local(x) if *x == page) => Some(*addend),
                    _ => None,
                })?;
                if off == 0 {
                    return None; // `[page]` — a complete address after all
                }
                uses.push((j, off));
            }
            _ => return None,
        }
    }
    if uses.is_empty() { None } else { Some(uses) }
}

// ---------------------------------------------------------------------------

/// Follow `Move` chains (`v1 = v2; v2 = imm`) from `op` back to a stable
/// operand. `resolve_through_moves` follows ONE def; metadata-slot addresses
/// routinely travel through two or three copies before a call.
fn resolve_operand_deep(insns: &[Insn], at: usize, op: &Operand, depth: usize) -> Operand {
    let mut cur = op.clone();
    for _ in 0..depth {
        let Some(next) = resolve_through_moves(insns, at, &cur, None) else {
            break;
        };
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

/// The static address an operand names, when it is one:
///   * `Immediate(a)`            — an address constant, or
///   * `[Imm(b) + addend]`       — a load from the fixed address `b+addend`.
///
/// A nested load (`[ [Imm(b)] ]`) is not a static address; `klass_operand_type`
/// peels those one level at a time.
fn static_addr_of(op: &Operand) -> Option<u64> {
    match op {
        Operand::Immediate(a) => Some(*a as u64),
        Operand::Memory {
            base,
            index: None,
            addend,
            ..
        } => match base.as_ref() {
            Operand::Immediate(b) => Some((*b as u64).wrapping_add(*addend as u64)),
            _ => None,
        },
        _ => None,
    }
}

/// [`static_addr_of`], with the page LOCAL resolved.
///
/// `adrp x9, page; ldr x0, [x9, #off]` reaches the IR as
/// `x0 = [Local(page) + off]` as soon as SSA has materialized the `adrp` into
/// its own local version, and `static_addr_of` only accepts
/// `base: Immediate`. So the ordinary Android spelling of "the address of a
/// metadata-usage cell" matched nothing, and every
/// `il2cpp_codegen_initialize_runtime_metadata(&slot)` call was invisible to
/// the detector that drops it (`key_function_recovery`), which is why those
/// calls survived as `fn_…(&slot)`.
///
/// Only an ENTRY-constant base is folded: a local whose definition is a
/// constant, possibly through register copies. A computed base is an address
/// the analysis does not know, and is refused.
fn static_addr_of_deep(insns: &[Insn], at: usize, op: &Operand) -> Option<u64> {
    if let Some(a) = static_addr_of(op) {
        return Some(a);
    }
    let Operand::Memory {
        base,
        index: None,
        addend,
        ..
    } = op
    else {
        return None;
    };
    let Operand::Local(v) = base.as_ref() else {
        return None;
    };
    let mut cur = *v;
    for _ in 0..4 {
        let d = ssa_def_of(insns, at, cur)?;
        if !matches!(insns[d].op, Op::Move) {
            return None;
        }
        match insns[d].operands.get(1)? {
            Operand::Immediate(b) => return Some((*b as u64).wrapping_add(*addend as u64)),
            Operand::Local(w) if *w != cur => cur = *w,
            _ => return None,
        }
    }
    None
}

/// Resolve a metadata-usage slot (or a GOT slot that points at one) to its
/// TypeDef and display name.
fn type_at_slot(maps: &ResolveMaps, slot: u64) -> Option<super::ir::NamedTypeRef> {
    if let Some(t) = klass_tdi_to_type(maps, slot) {
        return Some(t);
    }
    // Metadata-usage slots are routinely reached through the GOT: the slot
    // holds the runtime address of the usage cell, which is what
    // `klass_to_tdi` keys.
    if let Some(&addend) = maps.got_addend_by_addr.get(&slot) {
        return klass_tdi_to_type(maps, addend);
    }
    None
}

/// The instruction that last defines register `name` strictly before `at`.
fn def_of_reg_before(insns: &[Insn], at: usize, name: &str) -> Option<usize> {
    (0..at.min(insns.len()))
        .rev()
        .find(|&j| matches!(insns[j].dest(), Some(Operand::Register(n)) if n == name))
}

/// The constant value an operand provably holds.
///
/// Folds the constant-carrying shapes the ARM64 lifter produces:
///   * `adrp`/`movz`          → `Immediate`,
///   * a register copy          → follow the defining `Move`,
///   * `ldr x, [page, #off]`   → the slot at `page + off`; when that slot is
///     RELATIVE-relocated its live value IS the relocation addend, the only
///     pointer value knowable without a running process.
///
/// Anything else refuses: a computed address is not an address this analysis
/// knows, and answering a plausible-looking constant for one would name the
/// wrong class.
fn static_value_of(
    maps: &ResolveMaps,
    insns: &[Insn],
    at: usize,
    op: &Operand,
    depth: usize,
) -> Option<u64> {
    if depth == 0 {
        return None;
    }
    match op {
        Operand::Immediate(a) => Some(*a as u64),
        Operand::Memory {
            base,
            index: None,
            addend,
            ..
        } => {
            let slot = static_value_of(maps, insns, at, base, depth - 1)?
                .wrapping_add(*addend as u64);
            maps.got_addend_by_addr.get(&slot).copied()
        }
        Operand::Register(name) => {
            let d = def_of_reg_before(insns, at, name)?;
            if !matches!(insns[d].op, Op::Move) {
                return None;
            }
            static_value_of(maps, insns, d, insns[d].operands.get(1)?, depth - 1)
        }
        Operand::Local(v) => {
            let d = ssa_def_of(insns, at, *v)?;
            if !matches!(insns[d].op, Op::Move) {
                return None;
            }
            static_value_of(maps, insns, d, insns[d].operands.get(1)?, depth - 1)
        }
        _ => None,
    }
}

/// The class an operand provably names, from metadata evidence only.
///
/// Handles every shape the code generator emits for a `Type_t*` value:
///   * the usage-slot address itself (`Immediate(slot)`),
///   * a load from the slot (`[slot]`),
///   * a load from a GOT slot that points at the usage slot (`[[got]]`), which
///     is the normal cross-`adrp` form on Android,
///   * the same reached through the ordinary `adrp; ldr; ldr` chain, where the
///     middle register is NOT a constant:
///
///     ```text
///     adrp x8, page
///     ldr  x8, [x8, #off]   ; x8 = the relocated pointer stored at page+off
///     ldr  x0, [x8]         ; x0 = klass
///     ```
///
/// The old body consulted only `static_addr_of` (an `Immediate` base) and then
/// recursed into `Memory.base`, so the fourth shape — where `base` is a bare
/// register — answered `None`. Every allocation through it lost its klass, and
/// `fuse_constructor_calls` then named the object from the `.ctor` instead.
/// Identical-code-folding merges every trivial constructor onto ONE `ret` in
/// this binary, so that fallback named the wrong class: `new Token()` printed
/// as `new Object()`, and because the local was then typed `Object` no field
/// could be named on it either — the whole `token.Reward = …` block degraded
/// to raw `*(nint*)(v + 16)` stores.
fn klass_operand_type(
    maps: &ResolveMaps,
    insns: &[Insn],
    at: usize,
    op: &Operand,
    depth: usize,
) -> Option<super::ir::NamedTypeRef> {
    if depth == 0 {
        return None;
    }
    match op {
        // The operand IS the cell address (`&slot`), or a plain constant.
        Operand::Immediate(a) => type_at_slot(maps, *a as u64),
        Operand::Memory {
            base,
            index: None,
            addend,
            ..
        } => {
            // The address this load reads from.
            if let Some(addr) = static_value_of(maps, insns, at, base, depth - 1)
                .map(|a| a.wrapping_add(*addend as u64))
            {
                if let Some(t) = type_at_slot(maps, addr) {
                    return Some(t);
                }
            }
            klass_operand_type(maps, insns, at, base, depth - 1)
        }
        Operand::Register(name) => {
            let d = def_of_reg_before(insns, at, name)?;
            if !matches!(insns[d].op, Op::Move) {
                return None;
            }
            klass_operand_type(maps, insns, d, insns[d].operands.get(1)?, depth - 1)
        }
        Operand::Local(v) => {
            let d = ssa_def_of(insns, at, *v)?;
            if !matches!(insns[d].op, Op::Move) {
                return None;
            }
            klass_operand_type(maps, insns, d, insns[d].operands.get(1)?, depth - 1)
        }
        _ => None,
    }
}

/// Is this address a metadata-usage cell? That is the argument of
/// `il2cpp_codegen_initialize_runtime_metadata` — it passes the slot, not the
/// value, so the operand is an address constant (or a GOT-wrapped one).
fn is_metadata_usage_slot(
    maps: &ResolveMaps,
    field_index_by_addr: &HashMap<u64, u32>,
    addr: u64,
) -> bool {
    let hit = |a: u64| {
        maps.klass_to_tdi.contains_key(&a)
            || maps.literal_index_by_addr.contains_key(&a)
            || maps.method_spec_index_by_addr.contains_key(&a)
            || field_index_by_addr.contains_key(&a)
    };
    hit(addr)
        || maps
            .got_addend_by_addr
            .get(&addr)
            .is_some_and(|a| hit(*a))
}

/// `&value` in the two shapes the lifter produces for an out-address.
fn is_value_address(op: &Operand) -> bool {
    matches!(op, Operand::AddressOf(_))
}

/// The next non-Nop instruction at or after `i + 1`.
fn next_real_insn(insns: &[Insn], i: usize) -> Option<usize> {
    (i + 1..insns.len()).find(|&j| !matches!(insns[j].op, Op::Nop))
}

/// The constructor call immediately following a NewObj result, when it is one:
/// `next_call` is a resolved `.ctor` whose receiver is `dst`. Returns whether
/// the constructor takes NO declared parameters (so fusing it cannot drop a
/// real argument).
fn ctor_after(insns: &[Insn], i: usize, dst: u32) -> Option<bool> {
    let j = next_real_insn(insns, i)?;
    if !matches!(insns[j].op, Op::Call | Op::CallVoid) {
        return None;
    }
    let Some(Operand::Method(m)) = insns[j].operands.first() else {
        return None;
    };
    if m.name != ".ctor" {
        return None;
    }
    let receiver_ok = insns[j]
        .operands
        .get(2)
        .or_else(|| insns[j].operands.get(1))
        .and_then(Operand::as_local)
        == Some(dst);
    receiver_ok.then_some(m.sig.params.is_empty())
}

/// [`ctor_after`] with the callee resolved from a RAW call target as well.
///
/// `ctor_after` accepts only `Operand::Method`, i.e. a call the resolve pass
/// already named. The constructor that follows an allocation is frequently NOT
/// named: on this binary identical-code-folding merges every trivial `.ctor`
/// onto one shared `ret`, so the target arrives as a bare `Immediate` and the
/// allocate/construct pair was invisible. Resolving through
/// `maps.methods_by_rva` — the same table the pass uses everywhere else —
/// recovers the pairing without guessing.
fn ctor_after_resolved(insns: &[Insn], i: usize, dst: u32, maps: &ResolveMaps) -> bool {
    if let Some(known) = ctor_after(insns, i, dst) {
        return known;
    }
    let Some(j) = next_real_insn(insns, i) else {
        return false;
    };
    if !matches!(insns[j].op, Op::Call | Op::CallVoid) {
        return false;
    }
    let Some(Operand::Immediate(t)) = insns[j].operands.first() else {
        return false;
    };
    let Some(m) = maps.methods_by_rva.get(&(*t as u64)) else {
        return false;
    };
    if m.name != ".ctor" || !m.sig.params.is_empty() {
        return false;
    }
    insns[j]
        .operands
        .get(2)
        .or_else(|| insns[j].operands.get(1))
        .and_then(Operand::as_local)
        == Some(dst)
}

fn key_function_recovery(
    insns: &mut [Insn],
    maps: &ResolveMaps,
    field_index_by_addr: &HashMap<u64, u32>,
    stats: &mut ResolveStats,
) {
    // Every local any instruction READS, computed once. A call's destination
    // slot is a write, so it is excluded by `dest_position`.
    //
    // This exists because the shape detectors below asked `is_void` (i.e.
    // `Op::CallVoid`) whether a call's result is discarded — and **no lifter
    // ever emits `Op::CallVoid`** (`lift*.rs` contains no constructor for it;
    // the ARM64 `BL`/`BLR` arms both build `Op::Call`). The question they
    // actually need answered is "is this result used", which is a property of
    // the dataflow, not of the opcode.
    let mut read_locals: HashSet<u32> = HashSet::new();
    for insn in insns.iter() {
        let dpos = Insn::dest_position(&insn.op);
        for (k, op) in insn.operands.iter().enumerate() {
            if Some(k) == dpos {
                continue;
            }
            let mut u = Vec::new();
            op.uses(&mut u);
            read_locals.extend(u);
        }
    }
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Call | Op::CallVoid) {
            continue;
        }
        let Some(Operand::Immediate(target)) = insns[i].operands.first().cloned() else {
            continue;
        };
        let t = target as u64;
        let is_void = matches!(insns[i].op, Op::CallVoid);
        // Does anything read this call's result? See `read_locals` above: this
        // is the fact `is_void` was standing in for.
        let result_unread = match insns[i].dest() {
            None => true,
            Some(Operand::Local(v)) => !read_locals.contains(v),
            // A still-unnamed register: be conservative and keep the call.
            Some(_) => false,
        };
        // Operand layout: Call [target, result, x0, x1, …],
        //                 CallVoid [target, x0, x1, …].
        let x0_at = if is_void { 1 } else { 2 };
        let x0 = insns[i]
            .operands
            .get(x0_at)
            .map(|o| resolve_operand_deep(insns, i, o, 4));
        let x1 = insns[i]
            .operands
            .get(x0_at + 1)
            .map(|o| resolve_operand_deep(insns, i, o, 4));
        let x0_klass = x0.as_ref().and_then(|o| klass_operand_type(maps, insns, i, o, 8));
        let x1_klass = x1.as_ref().and_then(|o| klass_operand_type(maps, insns, i, o, 8));
        // A target proven not to be a function entry must not acquire a
        // helper role from a call-site SHAPE. This is how a resolved indirect
        // dispatch into a jump table became `IL2CPP_IsInst(obj, 16)` /
        // one-argument IsInst in every method that called it.
        let non_entry = super::runtime_helpers::is_non_entry(t);
        // An allocation is also provable by SHAPE at the call site, without
        // naming the helper at all:
        //   * the target is not a managed method (it is a runtime helper),
        //   * x0 is a KLASS VALUE (a load through a metadata-usage cell),
        //   * the result is immediately constructed by a zero-argument `.ctor`.
        //
        // Nothing else in IL2CPP takes a klass value in x0 and is followed by a
        // constructor on its own result. This evidence is stronger than the
        // helper's name, and unlike the name it does not depend on the target
        // appearing in the frequency sample `classify_bodies` walks.
        let alloc_by_shape = !is_void
            && !maps.methods_by_rva.contains_key(&t)
            && x0_klass.is_some()
            && insns[i]
                .dest()
                .and_then(Operand::as_local)
                .is_some_and(|d| ctor_after_resolved(insns, i, d, maps));
        // PRECEDENCE, and it is load-bearing. An allocation is the most specific
        // shape in this function, and several broader families below would
        // otherwise claim it first and `continue` — most importantly
        // `maps.metadata_init`, whose detector matches "the body reads the
        // class-init byte `[klass+0x135]`". `il2cpp_object_new` reads exactly
        // that byte (it runs the cctor before allocating), so on this binary the
        // shared allocator thunk was classified as metadata bookkeeping and its
        // every call site hit the `continue` in the metadata-init block below.
        // The call then stayed a plain `Call`, `fuse_constructor_calls` named
        // the object from its `.ctor`, and because identical-code-folding merges
        // every trivial `.ctor` onto one `ret`, that name was an arbitrary
        // class: `new Token()` printed as `new Object()`, and since the local
        // was then typed `Object` no field could be named on it either — the
        // whole `token.Reward = …` block degraded to `*(nint*)(v + 16)` stores.
        let wants_object_new = !non_entry
            && (maps.object_new.contains(&t)
                || super::runtime_helpers::role_of(t)
                    == Some(super::runtime_helpers::RuntimeHelper::ObjectNew)
                || alloc_by_shape);

        // Runtime metadata initialization: a VOID call whose first argument is
        // the ADDRESS of a metadata-usage cell. The helper decodes the cell
        // into the runtime object graph; the managed source never contained
        // this call, and its result is discarded by construction. This is the
        // only shape that passes a usage-slot ADDRESS (a class-init helper is
        // passed the resolved class VALUE).
        // The gate is `result_unread`, NOT `is_void`: see `read_locals` above.
        // A RESOLVED target is excluded outright — this helper is a VM internal
        // that never has a metadata body, so a nameable method taking a usage
        // cell is a real call and dropping it would delete managed code.
        if !non_entry && !wants_object_new && result_unread && !maps.methods_by_rva.contains_key(&t)
        {
            if let Some(slot) = x0
                .as_ref()
                .and_then(|o| static_addr_of_deep(insns, i, o))
            {
                if is_metadata_usage_slot(maps, field_index_by_addr, slot) {
                    super::runtime_helpers::note_shape(
                        t,
                        super::runtime_helpers::RuntimeHelper::MetadataInit,
                    );
                    super::verify::ledger_note(
                        insns[i].address,
                        super::verify::LedgerKind::DroppedRuntimeCall,
                    );
                    insns[i] = Insn::new(Op::Nop, Vec::new(), insns[i].address);
                    stats.key_functions += 1;
                    continue;
                }
            }
        }

        // Box call shape: `v = helper(class, &value)`. Kept as a call (rather
        // than rewritten to `Op::Box`, which cannot carry the value into the
        // renderer); the statement-stage fold turns it into the value
        // expression. Recording the role here also overrides a broad
        // object_new alias for the same target.
        if !non_entry && !is_void && x0_klass.is_some() && x1.as_ref().is_some_and(is_value_address)
        {
            super::runtime_helpers::note_shape(t, super::runtime_helpers::RuntimeHelper::Box);
            continue;
        }

        // Instance-of shape: object in x0, class in x1. A broad object_new
        // alias is overridden here only when x0 is provably NOT a class: an
        // allocator's first argument is always the class.
        if !non_entry && !is_void && x0_klass.is_none() && x1_klass.is_some() {
            super::runtime_helpers::note_shape(t, super::runtime_helpers::RuntimeHelper::IsInst);
            continue;
        }

        // il2cpp_codegen_initialize_class / InitializeMethodMetadata-family:
        // runtime metadata bookkeeping (verified: body checks [klass+0x132]
        // cctor byte / method slots). NOT managed logic — the C# source never
        // had these calls. Drop them entirely instead of rendering
        // `fn_4a58b98(...)` noise that misreads as a real operation.
        if !wants_object_new && maps.metadata_init.contains(&t) {
            // NEVER drop a call whose return value is consumed downstream.
            // The init-shape detector matches shared helpers that ALSO
            // allocate (`il2cpp_codegen_object_new` checks the class-init
            // byte then returns the new object). When the x0 result feeds a
            // later `.ctor` receiver / null-check / argument, dropping the
            // call orphans every use into undefined locals (`v53..ctor()`
            // with v53 never defined — the plausible-but-wrong class).
            // Conservative use-scan; the block is bounded by body length.
            let ret_used = match insns[i].operands.get(1) {
                // SSA phi inputs can name a definition on a loop backedge even
                // when the phi appears earlier in listing order. Search the
                // whole body: the version is unique, so any non-destination
                // read proves the result is consumed, including an edge use.
                Some(Operand::Local(v)) => insns
                    .iter()
                    .enumerate()
                    .any(|(j, n)| j != i && n.use_locals().contains(v)),
                _ => false,
            };
            if !ret_used {
                super::verify::ledger_note(
                    insns[i].address,
                    super::verify::LedgerKind::DroppedRuntimeCall,
                );
                insns[i] = Insn::new(Op::Nop, Vec::new(), insns[i].address);
                stats.key_functions += 1;
            }
            continue;
        }

        // il2cpp_string_new*(str) → Move dst, StringLiteral
        if maps.string_new.contains(&t) {
            let lit = find_arg_operand(insns, i, &maps.literal_index_by_addr, &maps.literals);
            if let Some(lit) = lit {
                let dst = insns[i].dest().cloned().unwrap_or(Operand::Unknown);
                insns[i] = Insn::move_(dst, lit, insns[i].address);
                insns[i].dest_type = Some(super::types::ManagedType::String);
                stats.key_functions += 1;
                stats.strings_resolved += 1;
                continue;
            }
        }

        // Objective-C selector stub: the callee IS `msgSend(receiver, sel…)`
        // with the selector fixed by the stub. Render as `[recv sel:args]` —
        // the selector carries the real operation name (setBool:forKey: etc.)
        // that raw `fn_4a58bac` renders hide.
        if let Some(sel) = maps.objc_selector_stubs.get(&t).cloned() {
            let raw_args: Vec<Operand> = insns[i]
                .operands
                .iter()
                .skip(2)
                .map(|op| resolve_through_moves(insns, i, op, None))
                .map(|o| o.unwrap_or(Operand::MissingAbiArg))
                .collect();
            let recv = raw_args.first().cloned().unwrap_or(Operand::Unknown);
            // objc selector `sel:a:b:` maps to args (x2, x3, x4…)
            let margs: Vec<Operand> = raw_args.into_iter().skip(1).collect();
            let dst = insns[i].dest().cloned().unwrap_or(Operand::Unknown);
            insns[i] = Insn::new(
                Op::ObjcMsgSend { selector: sel },
                std::iter::once(dst)
                    .chain(std::iter::once(recv))
                    .chain(margs)
                    .collect(),
                insns[i].address,
            );
            stats.key_functions += 1;
            continue;
        }

        // raise exception family → Throw(arg)
        if maps.raise_exception.contains(&t) {
            let arg = insns[i]
                .operands
                .get(if matches!(insns[i].op, Op::CallVoid) {
                    1
                } else {
                    2
                })
                .cloned()
                .and_then(|o| resolve_through_moves(insns, i, &o, None))
                .unwrap_or(Operand::Unknown);
            
            // Try to extract the exception message by tracing the exception object back to its NewObj constructor
            if let Operand::Local(local_id) = &arg {
                // Search backwards for the instruction that defines this local
                for j in (0..i).rev() {
                    if let Some(dest) = insns[j].dest() {
                        if dest.as_local() == Some(*local_id) {
                            // Found the defining instruction
                            if matches!(insns[j].op, Op::Call | Op::CallVoid) {
                                // It's a call - check if it's a constructor call with a string argument
                                let first_arg_idx = if matches!(insns[j].op, Op::CallVoid) { 1 } else { 2 };
                                if let Some(first_arg) = insns[j].operands.get(first_arg_idx) {
                                    if let Some(resolved) = resolve_through_moves(insns, j, first_arg, None) {
                                        if let Operand::StringLiteral { value, .. } = resolved {
                                            super::runtime_helpers::note_raise_message(t, value.to_string());
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
            
            insns[i] = Insn::new(Op::Throw, vec![arg], insns[i].address);
            stats.key_functions += 1;
            continue;
        }

        // object_new(klass) → NewObj(type when klass known). A target whose
        // BODY was proven to be il2cpp_object_new participates even when the
        // statistical classifier missed it (the largest allocator variants are
        // not exported and do not thunk to the exported anchor).
        // An allocation is also provable by SHAPE at the call site, without
        // naming the helper at all:
        //   * the target is not a managed method (it is a runtime helper),
        //   * x0 is a KLASS VALUE (a load through a metadata-usage cell),
        //   * the result is immediately constructed by a zero-argument `.ctor`.
        //
        // Nothing else in IL2CPP takes a klass value in x0 and is followed by a
        // constructor on its own result. This evidence is stronger than the
        // helper's name, and unlike the name it does not depend on the target
        // appearing in the frequency sample `classify_bodies` walks (top 128).
        // The shared `new T()` allocator is a `b`-chain thunk that is common
        // but not top-128, so the name route missed it and every allocation fell
        // through to `fuse_constructor_calls`, which names the object from its
        // `.ctor` — and every trivial `.ctor` in this binary is folded onto one
        // `ret`, so that name was arbitrary. `new Token()` printed as
        // `new Object()`, which then made every `token.X = …` store unnameable.
        if wants_object_new {
            // A void call cannot be an allocation use: the result is the
            // object. Broad alias chains put bookkeeping wrappers in the
            // object_new family; they stay calls.
            if is_void {
                continue;
            }
            // The result slot is still a REGISTER at this point (SSA renames
            // registers to locals later), so it must be carried through
            // verbatim.
            let Some(dst) = insns[i].dest().cloned() else {
                continue;
            };
            let dst_local = dst.as_local();
            // The klass is x0. Metadata evidence: a constant slot address, a
            // load from it, or a GOT load that points at the usage cell.
            let tyref = x0
                .as_ref()
                .and_then(|o| klass_operand_type(maps, insns, i, o, 8));
            // ONLY claim a constructor when the klass actually resolves.
            //
            // An UNTYPED `NewObj(None)` is not a harmless placeholder: the
            // statement renderer prints a bare `new object()`, and the dead
            // code pass DELETES it as an unused allocation. So rewriting a
            // plain call here can erase a side effect that really happens.
            //
            // On the reference binary `0x4A58960` — a thin wrapper
            // (`bl 0x4abb2fc; ret`) whose RESULT IS DISCARDED by every caller —
            // was a member of this set, and in `CollectWeeklyFacebookGift`
            // @0x47014C all four `il2cpp_codegen_initialize_runtime_metadata`
            // calls inside the `if (init_flag == 0)` guard were rewritten to
            // `NewObj(None)` and then deleted, leaving the guard EMPTY. Four
            // real calls, gone, with no marker in the output.
            //
            // The one safe untyped rewrite is when the allocation is provably
            // paired with a ZERO-ARGUMENT `.ctor`: `fuse_constructor_calls`
            // immediately names the type from the constructor and nothing is
            // dropped.
            if let Some(ty) = tyref {
                insns[i] = Insn::new(
                    Op::NewObj(Some(ty)),
                    vec![dst.clone()],
                    insns[i].address,
                );
                stats.key_functions += 1;
            } else if dst_local.is_some_and(|d| ctor_after(insns, i, d) == Some(true)) {
                insns[i] = Insn::new(Op::NewObj(None), vec![dst.clone()], insns[i].address);
                stats.key_functions += 1;
            }
            continue;
        }

        // box family: box(klass, T* value). The call is LEFT IN PLACE: the
        // renderer can slice it to its two proven arguments, and the
        // statement-stage pass folds it into the value expression. `Op::Box`
        // cannot carry the value into the renderer, so emitting it here would
        // lose the only interesting operand.
        if maps.box_fn.contains(&t) {
            super::runtime_helpers::note_shape(t, super::runtime_helpers::RuntimeHelper::Box);
            continue;
        }
    }
}

/// klass slot address → TypeDef index → NamedTypeRef (namespace/name from
/// the metadata strings recorded in `tdi_names` by RecoveryContext).
fn klass_tdi_to_type(maps: &ResolveMaps, klass_addr: u64) -> Option<super::ir::NamedTypeRef> {
    let tdi = *maps.klass_to_tdi.get(&klass_addr)?;
    tdi_name_hint(maps, tdi).map(|name| super::ir::NamedTypeRef {
        ty: super::types::NamedType {
            namespace: name.0,
            name: name.1,
            assembly: None,
            type_def_index: tdi,
        },
    })
}

fn tdi_name_hint(maps: &ResolveMaps, tdi: i32) -> Option<(String, String)> {
    maps.tdi_names.get(&tdi).cloned()
}

/// Receiver local's seeded managed type → TypeDef index (for vtable lookups).
fn seed_class_tdi(table: &LocalTable, local: u32) -> Option<i32> {
    let &li = table.index_of.get(&local)?;
    match &table.infos[li].ty {
        ManagedType::Class(nt) if nt.type_def_index >= 0 => Some(nt.type_def_index),
        _ => None,
    }
}

/// Receiver type when the receiver is a FIELD (or a copy of one):
/// `v5 = this._ctrl` carries the field's declared type in its `FieldRef`,
/// even when the LocalTable never typed `v5`. Without this every virtual
/// call on a field receiver (`this._ctrl.Method()`) stayed `fnptr_` unless
/// the field happened to also be in the type table.
fn receiver_tdi_through_fields(insns: &[Insn], upto: usize, v: u32, depth: u8) -> Option<i32> {
    if depth == 0 {
        return None;
    }
    let d = ssa_def_of(insns, upto, v)?;
    if !matches!(insns[d].op, Op::Move) {
        return None;
    }
    match insns[d].operands.get(1)? {
        Operand::InstanceField { field, .. } | Operand::Field(field) => match &field.field_type {
            ManagedType::Class(nt) if nt.type_def_index >= 0 => Some(nt.type_def_index),
            _ => None,
        },
        Operand::Local(w) if *w != v => receiver_tdi_through_fields(insns, d, *w, depth - 1),
        _ => None,
    }
}

fn find_arg_operand(
    insns: &[Insn],
    call_idx: usize,
    lit_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) -> Option<Operand> {
    // walk back looking for the value moved into x0/w0
    for k in (0..call_idx).rev().take(16) {
        let insn = &insns[k];
        if !matches!(insn.op, Op::Move) {
            continue;
        }
        let is_x0 = matches!(insn.dest(), Some(Operand::Register(n)) if n == "x0")
            || matches!(insn.dest(), Some(Operand::Local(_)));
        if !is_x0 {
            continue;
        }
        // `absolute_address`, not a bare `Immediate`: on x64 the literal slot
        // is loaded RIP-relative, so the address is the `Memory`'s addend.
        match insn.operands.get(1) {
            Some(Operand::StringLiteral { .. }) => return insn.operands.get(1).cloned(),
            Some(src) => {
                let addr = absolute_address(insns, k, src)?;
                if let Some(&idx) = lit_addr.get(&addr) {
                    return literals.get(&idx).map(|s| Operand::StringLiteral {
                        index: idx,
                        value: std::sync::Arc::from(s.as_str()),
                    });
                }
                return None;
            }
            None => return None,
        }
    }
    None
}

// ---------------------------------------------------------------------------

/// Fuse `Call(.ctor)` immediately following `NewObj` whose receiver traces to
/// the NewObj result — Cpp2IL `MetadataResolver.ResolveConstructorCalls`.
fn fuse_constructor_calls(insns: &mut [Insn], stats: &mut ResolveStats) {
    for i in 0..insns.len().saturating_sub(1) {
        // accept BOTH typed and untyped NewObj: key_function_recovery emits
        // NewObj(None) when the klass type is unknown, and the upgrade below
        // is exactly what fills the type in from the fused .ctor — the old
        // `Some(_)` gate made that branch unreachable dead code
        if !matches!(insns[i].op, Op::NewObj(_)) {
            continue;
        }
        let Some(Some(new_dst)) = insns[i].dest().cloned().as_ref().map(|d| d.as_local()) else {
            continue;
        };
        // Walk to the `.ctor`, stepping over Nops AND over copies that merely
        // rename the allocation result.
        //
        // `mov x25, x0` after the allocation reaches the IR as
        // `Move [alias, dst]`, and requiring the ctor to be the IMMEDIATELY
        // next real instruction left the pair unfused at every such site. An
        // unfused pair is not cosmetic: the `.ctor` survives as its own
        // statement, the renderer inlines the `NewObj` definition at each use
        // of the local, and the result is `new Token()...ctor()` plus a fresh
        // `new Token()` for every field store. Only copies OF THE ALLOCATION
        // are stepped over, at most four, so an unrelated `Move` still stops
        // the walk.
        let mut aliases: std::collections::HashSet<u32> = std::collections::HashSet::new();
        aliases.insert(new_dst);
        let mut j = i + 1;
        let mut hops = 0usize;
        loop {
            while j < insns.len() && matches!(insns[j].op, Op::Nop) {
                j += 1;
            }
            if j >= insns.len() || hops > 4 {
                break;
            }
            if matches!(insns[j].op, Op::Call | Op::CallVoid) {
                break;
            }
            if matches!(insns[j].op, Op::Move) {
                let copy = insns[j]
                    .dest()
                    .and_then(|d| d.as_local())
                    .zip(insns[j].operands.get(1).and_then(|o| o.as_local()));
                if let Some((a, src)) = copy {
                    if aliases.contains(&src) {
                        aliases.insert(a);
                        j += 1;
                        hops += 1;
                        continue;
                    }
                }
            }
            break;
        }
        if j >= insns.len() || !matches!(insns[j].op, Op::Call | Op::CallVoid) {
            continue;
        }
        // receiver must be the newobj result, or one of its copies (position
        // depends on static-ness which we cannot know until resolved — accept
        // both layouts)
        let receiver_ok = insns[j]
            .operands
            .get(2)
            .or_else(|| insns[j].operands.get(1))
            .and_then(|op| op.as_local())
            .is_some_and(|v| aliases.contains(&v));
        // ctor heuristic: resolved to a method literally named ".ctor"
        let is_ctor =
            matches!(insns[j].operands.first(), Some(Operand::Method(m)) if m.name == ".ctor");
        // Fuse ARGUMENT-TAKING constructors only at the statement stage. The
        // `NewObj` op cannot carry constructor arguments, so fusing one here
        // would silently delete them (`new Foo(a, b)` → `new Foo()`). The
        // expression-stage fold (`reconstruct::expr_cleanup::fold_ctor_calls`)
        // merges `v = new T(); v..ctor(args)` into `new T(args)` instead.
        let zero_arg_ctor = matches!(
            insns[j].operands.first(),
            Some(Operand::Method(m)) if m.sig.params.is_empty()
        );
        if receiver_ok && is_ctor && zero_arg_ctor {
            // merge: upgrade NewObj's type knowledge and drop the ctor call
            if let (Op::NewObj(ty), Some(Operand::Method(m))) =
                (&insns[i].op, insns[j].operands.first())
            {
                if ty.is_none() {
                    insns[i].op = Op::NewObj(Some(super::ir::NamedTypeRef {
                        ty: super::types::NamedType {
                            namespace: m.declaring_namespace.clone(),
                            name: m.declaring_type_name.clone(),
                            assembly: Some(m.assembly.clone()),
                            type_def_index: m.type_def_index,
                        },
                    }));
                }
            }
            super::verify::ledger_note(insns[j].address, super::verify::LedgerKind::CtorFuse);
            insns[j] = Insn::nop(insns[j].address);
            stats.ctors_fused += 1;
        }
    }
}

// ---------------------------------------------------------------------------

/// Single-load getter pattern: `Return FieldRef([this + off])` becomes
/// `Return Field(...)` — Cpp2IL `MetadataResolver.ResolveGetter`.
fn resolve_getters(insns: &mut [Insn], stats: &mut ResolveStats) {
    // handled naturally once field loads carry FieldRef operands; count them
    for insn in insns.iter_mut() {
        if matches!(insn.op, Op::Return) {
            if let Some(Operand::Field(_)) = insn.operands.first_mut() {
                stats.getters_resolved += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------

/// The single SSA definition of version `v` at or before `upto`.
///
/// Exact, not a heuristic: this runs after `ssa::materialize` and before
/// `ssa::eliminate`, so every version has exactly ONE defining instruction
/// (a `Phi` for merge versions, an ordinary op otherwise). It must NOT be
/// used after phi elimination, where the per-edge copies give a version
/// several defs.
/// Name a call to an identical-code-folded address from the CALL SITE.
///
/// IL2CPP merges bodies that compile to the same machine code, so one address
/// can be claimed by hundreds of unrelated method definitions — on the v24.5
/// build `0x14dae0` is 272 methods, `0x14d000` is 238 constructors, and 74% of
/// the 644 folded addresses span more than one type DEFINITION. `methods_by_rva`
/// keeps one of them, so without this pass every call to such an address is
/// named after an arbitrary member: the dump told the reader a method belonged
/// to a class it does not (one type's property getter reported as another
/// type's; a `List<T>` method reported as a `Dictionary<K,V>` one).
///
/// Runs after `key_function_recovery` so the `NewObj` evidence exists, and
/// before `fuse_constructor_calls` so the fused `new T(…)` inherits the ctor the
/// ALLOCATION identifies instead of the canonical member of the folded set.
fn resolve_folded_names(
    insns: &mut [Insn],
    table: &LocalTable,
    maps: &ResolveMaps,
    parents: &HashMap<i32, Vec<i32>>,
    stats: &mut ResolveStats,
) {
    if maps.shared_by_rva.is_empty() {
        return;
    }
    for i in 0..insns.len() {
        let Some(Operand::Method(cur)) = insns[i].operands.first().cloned() else {
            continue;
        };
        if !matches!(insns[i].op, Op::Call | Op::CallVoid) {
            continue;
        }
        // `shared_rva_of_method` answers "is this name one member of a folded
        // set" in one lookup: a call has no `Immediate` target left once it has
        // been rewritten to a `Method` operand, so the ref itself has to carry
        // the address back.
        let Some(&rva) = maps.shared_rva_of_method.get(&cur.method_index) else {
            continue;
        };
        let Some(cands) = maps.shared_by_rva.get(&rva).filter(|c| c.len() > 1) else {
            continue;
        };
        if maps.shared_cross_type.get(&rva).copied() != Some(true) {
            continue; // only type ARGUMENTS differ: the name is already right
        }
        // The call's own ABI argument dump, still untouched: `[x0 … xN, fp…]`.
        // `resolve_folded_names` runs before `reslice_call_arguments` precisely
        // so that this is still the CALL's shape and not the canonical member's.
        let start = 2.min(insns[i].operands.len());
        let raw: &[Operand] = &insns[i].operands[start..];
        match pick_folded_definition(cands, insns, i, table, parents, raw) {
            Ok(k) => {
                let m = cands[k].clone();
                insns[i].operands[0] = Operand::Method(m.clone());
                insns[i].dest_type = Some(m.sig.ret.clone());
                stats.shared_calls_picked += 1;
                super::timing::SHARED_PICKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Err(why) => {
                fold_site_dbg(cands, insns, i, &cur, table, why, raw);
                stats.shared_calls_ambiguous += 1;
                use std::sync::atomic::Ordering::Relaxed;
                super::timing::SHARED_AMBIGUOUS.fetch_add(1, Relaxed);
                let c = match why {
                    SharedMiss::NoReceiver => &super::timing::SHARED_NO_RECV,
                    SharedMiss::NoClass => &super::timing::SHARED_NO_CLASS,
                    SharedMiss::NoMember => &super::timing::SHARED_NO_MEMBER,
                    SharedMiss::Multiple => &super::timing::SHARED_MULTI,
                };
                c.fetch_add(1, Relaxed);
            }
        }
    }
}

/// Which of the definitions folded onto one address a given call means.
///
/// IL2CPP folds identical bodies, so one address can be claimed by up to 857
/// method definitions that have nothing to do with each other (`RegistryKey::
/// get_InternalHandle`, `LocalDataStoreElement::get_Cookie`,
/// `OperatingSystem::get_Version` — all "load the field at this offset"). The
/// address cannot say which; the CALL SITE can, in two ways, and both are exact
/// rather than heuristic:
///
/// 1. **The allocation's klass.** A `.ctor` reached from a `NewObj(T)` is the
///    constructor of `T` — `T` came from a TypeInfo metadata-usage slot, so it
///    is metadata, not inference. 238 constructors share one address on the
///    v24.5 build, and every one of them is named by its own `NewObj`.
/// 2. **The receiver's class.** For a trivial accessor the member is the one
///    DECLARED ON the receiver's type (or inherited by it), so the receiver's
///    recovered class picks it out of the set.
///
/// Anything else returns `Err(reason)` and the caller keeps the canonical
/// member — but counts it, because that name is then a guess.
fn pick_folded_definition(
    cands: &[MethodRef],
    insns: &[Insn],
    i: usize,
    table: &LocalTable,
    parents: &HashMap<i32, Vec<i32>>,
    raw: &[Operand],
) -> Result<usize, SharedMiss> {
    // ---- ABI shape first: it is exact and needs no type ---------------------
    //
    // The merged candidates are methods folded onto one address, and a set
    // routinely spans DIFFERENT ARITIES: `FirstPassBehaviour::Update`'s address
    // carries **272** definitions with 0..4 parameters, 16 of them static
    // (`static=16/272 params={0,1,2,3,4}`). The call site's own argument count
    // removes candidates without a single type, a receiver, or any inference,
    // and it is available at EVERY folded call site.
    //
    // It has to come from the call's own ABI argument list, taken before the
    // operand list was rebuilt by SIGNATURE — that rebuild uses the CANONICAL
    // member's signature, so the arity it leaves behind is the canonical
    // member's. Reading that back here was this filter answering its own
    // question: whatever the canonical member's arity was, the compatible set
    // came back as "the members with that same arity" at every site,
    // independently of the call.
    //
    // `n_vals` is an UPPER BOUND on the call's real argument count. Every
    // argument register the callee reads has to be written by the caller, so it
    // holds a defined version and is never the trailing `MissingAbiArg` (or
    // `Unknown`) this strips. Leftover register contents and the FP dump inflate
    // it; nothing deflates it. So a merged member that needs MORE arguments than
    // the dump carries cannot be the callee — a rejection that holds even when
    // the number is only an upper bound.
    let n_vals = if raw.is_empty() {
        call_arg_values(&insns[i])
    } else {
        abi_arg_count(raw)
    };
    // An upper bound only PROVES that, though, while the dump shows an integer
    // register the caller never wrote. A callee with more arguments than the
    // register capacity passes the rest on the STACK — and then it necessarily
    // reads every argument register, so the caller writes all of them and the
    // run is full. A missing slot therefore proves the callee fits inside the
    // registers it was handed, and the count is exact for it. With a full run it
    // is only an upper bound that may be understating a stack-passed argument,
    // so nothing is filtered.
    let narrowable = abi_int_slots(raw, table).is_some_and(|cap| n_vals < cap);
    // The ABI-compatible set. `want == n_vals` is the reading this picker has
    // always used; it BECOMES a proof when the count is provable, and stays what
    // it was (the count is an upper bound, so a member needing fewer arguments is
    // merely unproven, not excluded) when it is not. Widening it to `want <=
    // n_vals` in the unprovable case was tried and measured on the v24.5 dump: it
    // turned arm (a) into arm (b)'s test and DROPPED an evidenced pick
    // (`this.Flush()` became the canonical `this.Update()`), so the restriction
    // is left exactly as strong as the evidence behind it.
    let exact: Vec<usize> = (0..cands.len())
        .filter(|&k| accepts_arg_count(&cands[k], n_vals))
        .collect();

    // The receiver is ABI slot 0 of the CALL — `x0`, which is the receiver for
    // an instance callee and the first declared parameter for a static one. It
    // is NOT `operands[2]`. When the canonical member is a zero-parameter
    // static the re-slice emits no arguments at all, `get(2)` is `None`, and the
    // old fallback read `get(1)` — this call's OWN result slot, whose single
    // definition is the call itself, so `receiver_tdi` resolved the receiver's
    // class from a `Call` whose return type was `Void` and refused. That is
    // where the 460 void-returning "receivers" in the census came from.
    let recv = raw
        .first()
        .filter(|o| !matches!(**o, Operand::MissingAbiArg | Operand::Unknown))
        .or_else(|| insns[i].operands.get(2))
        .or_else(|| insns[i].operands.get(1));

    // (1) the allocation this call belongs to.
    //
    // Match the receiver LOCAL, not a fixed look-back window. In x64 IL2CPP the
    // allocation and the `.ctor` call it feeds are two separate instructions and
    // `key_function_recovery` tags only the allocation site, so the two can sit
    // dozens of instructions apart whenever the constructor takes arguments —
    // exactly the `List<T>`/`Dictionary<K,V>` constructors this is meant to
    // name. A ±4 window silently missed every one of those.
    if let Some(Operand::Local(rv)) = recv {
        for k in (0..i).rev() {
            match &insns[k].op {
                Op::NewObj(Some(ty)) => {
                    if matches!(insns[k].dest(), Some(Operand::Local(d)) if *d == *rv) {
                        // An ABI-compatible candidate wins over one that cannot
                        // accept this call's arguments; the unrestricted search
                        // stays as the fallback so no existing pick is lost.
                        if let Some(p) = pick_by_named_type_idx(cands, &exact, &ty.ty) {
                            return Ok(p);
                        }
                        if let Some(p) = pick_by_named_type(cands, &ty.ty) {
                            return Ok(p);
                        }
                    }
                }
                // A `NewObj` with no type is the allocation of an UNKNOWN
                // class; if it defines our receiver, nothing downstream can
                // name it.
                Op::NewObj(None) => {
                    if matches!(insns[k].dest(), Some(Operand::Local(d)) if *d == *rv) {
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    // (2) the receiver's recovered class
    let Some(Operand::Local(v)) = recv else {
        return Err(SharedMiss::NoReceiver);
    };
    // What to report when nothing picks. "The receiver's class was never
    // recovered" and "the receiver's class owns no member of the merged set"
    // are different gaps, and the arity arm below can still answer for the
    // first — which is the LARGER one (4,227 of 4,231 folded calls on the v24.5
    // build), so refusing to reach it there would leave the argument evidence
    // unused at almost every site that needs it.
    let mut miss = SharedMiss::NoClass;
    if let Some(tdi) = receiver_tdi(insns, i, table, *v) {
        let chain = parents.get(&tdi).map(|c| c.as_slice()).unwrap_or(&[]);
        let owns = |c: &MethodRef| c.type_def_index == tdi || chain.contains(&c.type_def_index);
        // (a) the class identifies exactly one ABI-COMPATIBLE member
        if let Some(k) = unique_on_chain(&exact, cands, &owns) {
            return Ok(k);
        }
        let mut hits = cands.iter().enumerate().filter(|(_, c)| owns(c));
        match (hits.next(), hits.next()) {
            // (b) exactly one definition in the set is declared on this class's
            // chain. Note this arm deliberately does NOT require ABI
            // compatibility: it is the behaviour that already picked these
            // names, and the class evidence is the stronger fact whenever it
            // exists.
            (Some((k, _)), None) => return Ok(k),
            // Two or more: the class declares several of the merged members (an
            // indexer and its `get_`/`set_`, or overloads). Still a guess.
            (Some(_), Some(_)) => return Err(SharedMiss::Multiple),
            _ => miss = SharedMiss::NoMember,
        }
    }
    // (c) the class says nothing, but the ARGUMENT SHAPE leaves one candidate.
    // Gated on `narrowable`: a singleton `exact` is only meaningful when the
    // count it was compared against is the call's real one, and that is exactly
    // what `narrowable` establishes. Without it the singleton is just "one member
    // happens to share the canonical member's arity", which names nothing.
    if narrowable && exact.len() == 1 {
        return Ok(exact[0]);
    }
    Err(miss)
}

/// The ARGUMENT VALUES a call site carries, read back off an already re-sliced
/// operand list. Fallback only: `abi_arg_count` over the pre-re-slice dump is
/// the faithful reading, and this one is the canonical member's arity.
fn call_arg_values(insn: &Insn) -> usize {
    let start = 2.min(insn.operands.len());
    abi_arg_count(&insn.operands[start..])
}

/// How many argument VALUES an ABI operand dump carries.
///
/// Trailing `MissingAbiArg` is always padding — that token means "the ABI had
/// no live register for this slot" — and a trailing `Unknown` is a value whose
/// recovery failed, so both are dropped. Everything else counts, including
/// trailing `Register` operands: the lifter appends the FP argument registers
/// that were written since the last call, and those ARE values.
///
/// The result is an UPPER BOUND on the call's real argument count (see
/// `pick_folded_definition`), so it is only ever fed to a filter that REFUSES
/// candidates.
fn abi_arg_count(raw: &[Operand]) -> usize {
    let mut n = raw.len();
    while n > 0 && matches!(raw[n - 1], Operand::MissingAbiArg | Operand::Unknown) {
        n -= 1;
    }
    n
}

/// Does this definition need exactly `n` argument values?
///
/// An instance method's first ABI argument is its receiver, so it needs
/// `params + 1`; a static method needs exactly `params`. `n` is an upper bound
/// on what the call really passes (see `pick_folded_definition`), so an exact
/// match is a proof only where `narrowable` holds.
fn accepts_arg_count(c: &MethodRef, n: usize) -> bool {
    let want = if c.sig.is_static {
        c.sig.params.len()
    } else {
        c.sig.params.len() + 1
    };
    want == n
}

/// How many INTEGER argument slots an ABI dump opens with — the register
/// capacity the call passes its arguments in, as far as the dump shows it.
///
/// The lifter emits the integer argument registers first and the FP argument
/// registers after them, so the leading run of `x` slots IS the capacity (8 on
/// AAPCS64, 6 on x64). A slot scrubbed to `MissingAbiArg` has lost its origin
/// and is counted as an integer slot while the run lasts — which is exactly the
/// case that matters, because a missing slot is what the proof in
/// `pick_folded_definition` reads. `None` when no slot's origin is visible at
/// all; the ARM32 contract names its registers `r0`–`r3`, which this does not
/// recognise, and refusing there is the honest answer.
fn abi_int_slots(raw: &[Operand], table: &LocalTable) -> Option<usize> {
    let mut n = 0usize;
    let mut identified = false;
    for op in raw {
        match op {
            // A hole: no origin to check, so it is taken as an integer slot.
            Operand::MissingAbiArg => {}
            Operand::Local(v) => match table
                .index_of
                .get(v)
                .map(|&li| &table.infos[li].slot_origin)
            {
                Some(origin) if is_int_arg_register(origin) => identified = true,
                // an FP slot: the integer run ended
                Some(_) => break,
                None => break,
            },
            Operand::Register(r) if is_int_arg_register(r) => identified = true,
            _ => break,
        }
        n += 1;
    }
    identified.then_some(n)
}

/// `x0`…`x7` — an integer argument register, by the name BOTH IL2CPP lifters
/// use for one. The FP names are `vfN` (x64) and `vsN`/`vdN` (AAPCS64), so
/// requiring a DIGIT after the `x` is what keeps a stray symbol from counting.
fn is_int_arg_register(name: &str) -> bool {
    let mut cs = name.chars();
    cs.next() == Some('x') && cs.all(|c| c.is_ascii_digit())
}

/// The index of the one candidate in `idx` that the class chain owns, if there
/// is exactly one. Two or more is NOT a match — that is the overload/indexer
/// case the caller reports as `Multiple`.
fn unique_on_chain(
    idx: &[usize],
    cands: &[MethodRef],
    owns: &impl Fn(&MethodRef) -> bool,
) -> Option<usize> {
    let mut hit: Option<usize> = None;
    for &k in idx {
        if owns(&cands[k]) {
            if hit.is_some() {
                return None;
            }
            hit = Some(k);
        }
    }
    hit
}

/// [`pick_by_named_type`] restricted to the candidates in `idx`.
fn pick_by_named_type_idx(
    cands: &[MethodRef],
    idx: &[usize],
    ty: &super::types::NamedType,
) -> Option<usize> {
    let restricted: Vec<MethodRef> = idx.iter().map(|&k| cands[k].clone()).collect();
    pick_by_named_type(&restricted, ty).map(|p| idx[p])
}

/// Why a folded call site could not pick one definition out of the merged set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SharedMiss {
    /// The call has no receiver operand to read a type from.
    NoReceiver,
    /// The receiver is a local whose managed class was never recovered.
    NoClass,
    /// The class is known, but does not (and does not inherit) any member of
    /// the merged set — usually a wrong class, so it is safer to guess than to
    /// name it after the wrong `tdi`.
    NoMember,
    /// The class declares MORE than one member of the merged set.
    Multiple,
}

/// The class to search for the receiver's member: the receiver local's own
/// type, or — when the table has no class for it — the managed type of the
/// instruction that DEFINES that SSA version.
///
/// The table is filled by `propagate` from the same evidence, but a local that
/// is only ever written by a call is typed on the `Insn`, not always copied
/// back into the table, and those are precisely the receivers that arrive from
/// `new T(...)` and from a factory method.
fn receiver_tdi(insns: &[Insn], upto: usize, table: &LocalTable, v: u32) -> Option<i32> {
    if let Some(tdi) = receiver_tdi_inner(insns, upto, table, v, 8) {
        return Some(tdi);
    }
    fold_dbg(insns, upto, v);
    None
}

/// The class a local definitely holds, resolved through the SSA graph.
///
/// This is a TYPE JOIN over the def web, and it has to be exact — a wrong class
/// here does not degrade the name, it replaces the canonical member of a folded
/// set with a DIFFERENT wrong member, which is worse than leaving the guess
/// visible. So every step either proves one class or refuses:
///
/// 1. the local table's seeded type (parameters, `this`, annotated slots);
/// 2. the defining instruction's own `dest_type` (a call result, a cast);
/// 3. a CALL whose target is already a `Method` — its `sig.ret` IS the type,
///    no inference involved;
/// 4. a COPY chain (`v30 ← v28`) — followed, because after phi elimination and
///    DCE the value a receiver actually comes from is routinely an edge copy;
/// 5. a MERGE — accepted only when EVERY incoming value resolves to the SAME
///    class. A merge of two classes carries no single class, and refusing there
///    is what keeps a wrong name out of the output.
///
/// Measured on the v24.5 dump, the producers of the 4,231 folded calls whose
/// receiver class was unknown were: 1,653 a merge, 1,357 a call to a constant
/// address, 805 a call already resolved to a `Method`, 213 an entry value, and
/// the rest a vtable/function-pointer load or arithmetic. Steps 3–5 therefore
/// cover the majority of a population that was previously refused wholesale.
fn receiver_tdi_inner(
    insns: &[Insn],
    upto: usize,
    table: &LocalTable,
    v: u32,
    depth: u8,
) -> Option<i32> {
    if depth == 0 {
        return None;
    }
    if let Some(tdi) = seed_class_tdi(table, v) {
        return Some(tdi);
    }
    let d = ssa_def_of(insns, upto, v)?;
    let op = &insns[d];
    // 2. the defining instruction's own type
    if let Some(super::types::ManagedType::Class(nt)) = op.dest_type.as_ref() {
        if nt.type_def_index >= 0 {
            return Some(nt.type_def_index);
        }
    }
    // 3. a resolved callee's declared return type
    if matches!(op.op, Op::Call | Op::CallVoid) {
        if let Some(Operand::Method(m)) = op.operands.first() {
            if let super::types::ManagedType::Class(nt) = &m.sig.ret {
                if nt.type_def_index >= 0 {
                    return Some(nt.type_def_index);
                }
            }
        }
    }
    // 4. a copy: the SSA versions of ONE value, so the class is the source's
    if matches!(op.op, Op::Move) {
        if let Some(Operand::Local(w)) = op.operands.get(1) {
            if *w != v {
                return receiver_tdi_inner(insns, d, table, *w, depth - 1);
            }
        }
    }
    // 5. a merge: every incoming value must agree
    if matches!(op.op, Op::Phi) {
        let mut agreed: Option<i32> = None;
        for o in op.operands.iter().skip(1) {
            let Operand::Local(w) = o else { return None };
            // A phi input is defined in a PREDECESSOR block, so its def may sit
            // anywhere in the listing — search the whole method, not up to here.
            let t = receiver_tdi_inner(insns, insns.len(), table, *w, depth - 1)?;
            match agreed {
                None => agreed = Some(t),
                Some(a) if a == t => {}
                Some(_) => return None,
            }
        }
        return agreed;
    }
    None
}

/// `ILR_FOLDDBG=1` dumps the whole EVIDENCE at a folded call site that could not
/// be resolved: the merged candidate set (each member's identity, staticness,
/// parameter and return types) and the call's own operand list.
///
/// The per-receiver census said the producer of the unknown-class receiver is
/// usually a call that returns `Void`/`R4`/`Boolean` — which cannot be producing
/// a receiver — so what has to be seen is whether the candidate set is even
/// compatible with reading operand 2 as a receiver: if the merged members
/// disagree about `is_static`, one slicing cannot fit them all, and the fix is
/// to choose the member BEFORE slicing the argument list rather than after.
fn fold_site_dbg(
    cands: &[MethodRef],
    insns: &[Insn],
    i: usize,
    cur: &MethodRef,
    table: &LocalTable,
    why: SharedMiss,
    raw: &[Operand],
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    if !dbg_enabled("ILR_FOLDDBG") {
        return;
    }
    if N.fetch_add(1, Ordering::Relaxed) >= 12 {
        return;
    }
    let statics = cands.iter().filter(|c| c.sig.is_static).count();
    let params: std::collections::BTreeSet<usize> =
        cands.iter().map(|c| c.sig.params.len()).collect();
    // `n` is the ABI dump's value count — the evidence the pick is keyed on —
    // and `abi` is that dump. `operands` is the RE-SLICED list, printed next to
    // it so the two can be compared directly: they are the same site, and only
    // one of them is the call's own shape.
    let n = if raw.is_empty() {
        call_arg_values(&insns[i])
    } else {
        abi_arg_count(raw)
    };
    eprintln!(
        "[foldsite] {why:?} canonical={} static={}/{} params={:?} n={n} \
         abi={:?} operands={} sites={:?}",
        cur.full_name(),
        statics,
        cands.len(),
        params,
        raw.iter()
            .take(9)
            .map(|o| format!("{o}"))
            .collect::<Vec<_>>(),
        insns[i].operands.len(),
        insns[i]
            .operands
            .iter()
            .take(6)
            .map(|o| format!("{o}"))
            .collect::<Vec<_>>(),
    );
    for c in cands.iter().take(6) {
        eprintln!(
            "[foldsite]    candidate {} static={} params={:?} ret={:?} tdi={}",
            c.full_name(),
            c.sig.is_static,
            c.sig.params,
            c.sig.ret,
            c.type_def_index
        );
    }
    if let Some(Operand::Local(rv)) = raw
        .first()
        .filter(|o| !matches!(**o, Operand::MissingAbiArg | Operand::Unknown))
    {
        let ty = table
            .index_of
            .get(rv)
            .map(|li| format!("{:?}", table.infos[*li].ty));
        eprintln!("[foldsite]    receiver v{rv} seeded={ty:?}");
    }
}

/// `ILR_FOLDDBG=1` prints the shape of receiver locals whose class could not be
/// recovered, so the propagation gap is a list of concrete patterns to fix
/// rather than a number.
fn fold_dbg(insns: &[Insn], upto: usize, v: u32) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    if !dbg_enabled("ILR_FOLDDBG") {
        return;
    }
    if N.fetch_add(1, Ordering::Relaxed) >= 100_000 {
        return;
    }
    // CLASSIFY, do not just dump. The four populations call for completely
    // different fixes: a call whose target is already a resolved `Method`
    // (return type known, the renderer just did not use it), a call to a
    // constant address (folded/indirect address the resolver could name), a
    // call through a local (an indirect call the resolver could not name), a
    // field/pointer LOAD (a delegate or a vtbl slot), a phi (a merge of
    // receivers — resolvable if every incoming value agrees), and a parameter
    // (genuinely untyped `this`/`pN`).
    let def = match ssa_def_of(insns, upto, v) {
        Some(d) => {
            let op = &insns[d];
            let kind = match &op.op {
                Op::Call | Op::CallVoid => match op.operands.first() {
                    Some(Operand::Method(_)) => "call:method",
                    Some(Operand::Immediate(_)) => "call:addr",
                    Some(Operand::Local(_)) => "call:local",
                    Some(Operand::Memory { .. }) => "call:mem",
                    _ => "call:other",
                },
                Op::IndirectCall => "indirect",
                Op::Move => match op.operands.get(1) {
                    Some(Operand::Memory { .. }) => "load:mem",
                    Some(Operand::Local(_)) => "move:local",
                    Some(Operand::Field(_)) => "load:field",
                    Some(Operand::Method(_)) => "move:method",
                    _ => "move:other",
                },
                Op::Phi => "phi",
                Op::Arith(_) | Op::Convert(_) => "arith",
                Op::NewObj(_) => "newobj",
                _ => "other",
            };
            // For the two shapes the SSA-graph join is supposed to answer, show
            // the fact it keys on: the resolved callee's return type, or the
            // phi's operands. `folded_no_class` barely moved when the join was
            // added, so the REASON has to be visible.
            let why = match (&op.op, op.operands.first()) {
                (Op::Call | Op::CallVoid, Some(Operand::Method(m))) => {
                    format!(" ret={:?}", m.sig.ret)
                }
                (Op::Phi, _) => {
                    // Where each incoming value is defined. The join refuses
                    // unless every one of them resolves to the SAME class, so
                    // the def sites are what shows whether that is even
                    // possible (one producer per arm, or unrelated values).
                    let where_from: Vec<String> = op
                        .operands
                        .iter()
                        .skip(1)
                        .map(|o| {
                            match o.as_local().and_then(|w| ssa_def_of(insns, insns.len(), w)) {
                                Some(d) => format!("{}", insns[d].op),
                                None => "nodef".into(),
                            }
                        })
                        .collect();
                    format!(
                        " n={} from=[{}]",
                        op.operands.len() - 1,
                        where_from.join(", ")
                    )
                }
                _ => String::new(),
            };
            format!("{kind} {:#x}{why}", op.address)
        }
        None => "no-def".into(),
    };
    eprintln!("[fold] recv v{v} <- {def}");
}

fn dbg_enabled(var: &str) -> bool {
    std::env::var_os(var).is_some()
}

/// Pick the merged definition declared on a named type (a `NewObj`'s klass).
fn pick_by_named_type(cands: &[MethodRef], ty: &super::types::NamedType) -> Option<usize> {
    if ty.type_def_index >= 0 {
        if let Some(p) = cands
            .iter()
            .position(|c| c.type_def_index == ty.type_def_index)
        {
            return Some(p);
        }
    }
    // Fall back to the NAME: a generic instantiation `List<string>` shares the
    // type-def index of `List<T>` but is spelled differently in the dump, and
    // the merged set may hold either spelling.
    cands.iter().position(|c| c.declaring_type_name == ty.name)
}

fn ssa_def_of(insns: &[Insn], upto: usize, v: u32) -> Option<usize> {
    (0..upto.min(insns.len()))
        .rev()
        .find(|&j| matches!(insns[j].dest(), Some(Operand::Local(d)) if *d == v))
}

/// Is this the `invoke_impl` code-pointer field of `Il2CppDelegate`?
///
/// The typing sweep names `[delegate + off]` as `InstanceField` (or a bare
/// `Field` when the base is provably `this`), so by resolve time the delegate
/// target is no longer a `Memory` with `addend == 3*ptr` — it is a named
/// field. The name + declaring type are the proof: `invoke_impl` exists only
/// on `Delegate` (tdi 5422 on the iOS corpus), and a bare `Field` without a
/// base is only produced for a `this` base. Anything else (a different field,
/// a static field) refuses.
fn is_invoke_impl_field(f: &FieldRef) -> bool {
    !f.is_static && f.name == "invoke_impl" && f.declaring_type_name.contains("Delegate")
}

/// Is this managed type provably a delegate (so an `invoke_impl` load off it
/// is a delegate invoke, not some other object's pointer field)?
///
/// Same predicate `recover_delegate_invokes` uses for its `Memory` shape
/// (`Delegate`/`Action`/`Func`/`Event` in the type name). A `ByRef`/untyped
/// base refuses — the address could be anything, and guessing `Invoke` for a
/// non-delegate would fabricate a callee, which is worse than `fnptr_`.
fn is_delegate_class_ty(ty: &ManagedType) -> bool {
    match ty {
        ManagedType::Class(nt) => {
            nt.name.contains("Delegate")
                || nt.name.contains("Action")
                || nt.name.contains("Func")
                || nt.name.contains("Event")
        }
        _ => false,
    }
}

/// The delegate-typed `NamedType` a local provably holds, if it holds one.
fn delegate_named_ty(
    table: &LocalTable,
    v: u32,
) -> Option<std::sync::Arc<super::types::NamedType>> {
    let li = *table.index_of.get(&v)?;
    match &table.infos.get(li)?.ty {
        ManagedType::Class(nt) if is_delegate_class_ty(&ManagedType::Class(nt.clone())) => {
            Some(nt.clone())
        }
        _ => None,
    }
}

/// The method's `this` when it provably holds a delegate.
///
/// A bare `Field(invoke_impl)` carries no base — the locals sweep only folds
/// a `this` base into that form. `this` never changes value (every `is_this`
/// version holds the same object; `highvar` merges them to one name), so any
/// `is_this` version is the same receiver. The proof is a conjunction: at
/// least one `is_this` version exists, and EVERY `is_this` version with a
/// known class agrees on the same delegate type. One untyped `is_this` among
/// delegate-typed ones still proves (untyped is lack of evidence, not counter-
/// evidence); two DIFFERENT delegate types refuse.
fn this_delegate_base(
    table: &LocalTable,
) -> Option<(u32, std::sync::Arc<super::types::NamedType>)> {
    let mut found: Option<(u32, std::sync::Arc<super::types::NamedType>)> = None;
    // Deterministic order: sort versions so the chosen receiver does not
    // depend on `HashMap` iteration (see `det_gate.sh`).
    let mut versions: Vec<u32> = table.index_of.keys().copied().collect();
    versions.sort_unstable();
    for v in versions {
        let li = table.index_of.get(&v).copied().unwrap_or(usize::MAX);
        if li == usize::MAX {
            continue;
        }
        let Some(info) = table.infos.get(li) else {
            continue;
        };
        if !info.is_this {
            continue;
        }
        let nt = match &info.ty {
            ManagedType::Class(nt) if is_delegate_class_ty(&ManagedType::Class(nt.clone())) => {
                nt.clone()
            }
            // Untyped `this` is no evidence either way — skip, do not refuse.
            ManagedType::Unknown | ManagedType::IntPtr => continue,
            _ => return None,
        };
        match &found {
            None => found = Some((v, nt)),
            Some((_, prev)) if prev.name == nt.name && prev.namespace == nt.namespace => {}
            Some(_) => return None,
        }
    }
    found
}

/// Synthesize the `Invoke` `MethodRef` for a proven delegate type.
///
/// Same shape `recover_delegate_invokes` emits: `image_index == usize::MAX`
/// forces an external memberref, `type_def_index == -1` marks it synthesized,
/// `params == []` keeps the call's captured operands (the emitter preserves
/// them for an unknown-arity callee). The name is always `Invoke` — the one
/// virtual every delegate carries — so no slot or TypeDef is guessed.
fn delegate_invoke_ref(ns: String, tname: String) -> MethodRef {
    MethodRef {
        image_index: usize::MAX,
        type_def_index: -1,
        method_index: 0,
        name: "Invoke".into(),
        declaring_namespace: ns,
        declaring_type_name: tname,
        assembly: String::new(),
        sig: super::types::MethodSigInfo {
            ret: ManagedType::Unknown,
            params: vec![],
            is_static: false,
        },
    }
}

/// The delegate invoke a field-load definition provably performs.
///
/// `src` is the defining instruction's source operand (`operands[1]` of the
/// target's `Move`): either `InstanceField { base, field: invoke_impl }` or a
/// bare `Field(invoke_impl)`. Returns the synthesized `Invoke` plus the
/// delegate-base version that must become the call's receiver.
///
/// Proof (conjunction, never a guess):
/// * the field IS `invoke_impl` on `Delegate` (see [`is_invoke_impl_field`]);
/// * the base is a `Local` whose seeded type IS a delegate class
///   (`InstanceField`), or `this` is provably delegate (`Field`).
///
/// Anything else bumps `DELEGATE_REFUSED_UNPROVEN` and refuses — the site
/// stays `fnptr_`. Inert: atomic increments only, no IR change.
fn delegate_invoke_of_def(src: &Operand, table: &LocalTable) -> Option<(MethodRef, u32)> {
    match src {
        Operand::InstanceField { base, field } if is_invoke_impl_field(field) => {
            DELEGATE_LOOKUP_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
            let Operand::Local(d) = base.as_ref() else {
                DELEGATE_REFUSED_UNPROVEN.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let Some(nt) = delegate_named_ty(table, *d) else {
                DELEGATE_REFUSED_UNPROVEN.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let m = delegate_invoke_ref(nt.namespace.clone(), nt.name.clone());
            DELEGATE_LOOKUP_HITS.fetch_add(1, Ordering::Relaxed);
            Some((m, *d))
        }
        Operand::Field(field) if is_invoke_impl_field(field) => {
            DELEGATE_LOOKUP_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
            let (v, nt) = this_delegate_base(table).or_else(|| {
                DELEGATE_REFUSED_UNPROVEN.fetch_add(1, Ordering::Relaxed);
                None
            })?;
            let m = delegate_invoke_ref(nt.namespace.clone(), nt.name.clone());
            DELEGATE_LOOKUP_HITS.fetch_add(1, Ordering::Relaxed);
            Some((m, v))
        }
        _ => None,
    }
}

/// Why an `Arith`-base slot computation could not be keyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArithSlotMiss {
    Dynamic,
    NoDef,
    UnknownKlass,
    BadOffset,
}

/// Resolve `(klass + const)` through an `Arith(Add, base, const)` chain.
///
/// `base_local` is the `Memory` load's base (e.g. `v60` in `v60 = v59 + 312;
/// fn = [v60]`), `mem_addend` is that load's own addend (usually 0). The
/// chain is peeled Add-by-Immediate steps (`v60 = v59 + 312` → `acc = 312`,
/// `cur = v59`), following plain `Move` copies, until a non-`Arith` def is
/// reached. That remainder must be the klass (`[recv]`, an immediate klass,
/// or a copy thereof) resolved through [`klass_tdi_of`]; the accumulated
/// constant is the vtable offset, keyed exactly like a constant `Memory`
/// load (`(off - base) % entry == 0`).
///
/// ONLY `Add` with one `Immediate` side is peeled. `klass + scaled-idx` (two
/// locals), any `Shl`/`Mul`/`Sub`, or a `Phi` address is the dynamic-slot
/// shape (`klass + idx*16 + 312` with a runtime `idx`): there is no constant
/// slot to key, so it refuses as `Dynamic` rather than guessing one. A wrong
/// slot names a wrong method, which is worse than `fnptr_`.
///
/// Inert: atomic increments only; returns exactly what the constant-`Memory`
/// path would return for the same `(tdi, slot)`.
fn resolve_arith_slot(
    insns: &[Insn],
    upto: usize,
    base_local: u32,
    mem_addend: i64,
    table: &LocalTable,
    maps: &ResolveMaps,
    parents: &HashMap<i32, Vec<i32>>,
    ptr_size: u64,
    depth: u8,
) -> Option<MethodRef> {
    ARITH_SLOT_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    let miss = |m: ArithSlotMiss| {
        match m {
            ArithSlotMiss::Dynamic => ARITH_SLOT_REFUSED_DYNAMIC.fetch_add(1, Ordering::Relaxed),
            ArithSlotMiss::NoDef => ARITH_SLOT_REFUSED_NO_DEF.fetch_add(1, Ordering::Relaxed),
            ArithSlotMiss::UnknownKlass => {
                ARITH_SLOT_REFUSED_UNKNOWN_KLASS.fetch_add(1, Ordering::Relaxed)
            }
            ArithSlotMiss::BadOffset => {
                ARITH_SLOT_REFUSED_BAD_OFFSET.fetch_add(1, Ordering::Relaxed)
            }
        };
        None::<MethodRef>
    };
    let mut acc = mem_addend;
    let mut cur = base_local;
    let mut remaining = depth.max(8);
    loop {
        if remaining == 0 {
            return miss(ArithSlotMiss::Dynamic);
        }
        remaining -= 1;
        let Some(d) = ssa_def_of(insns, upto, cur) else {
            return miss(ArithSlotMiss::NoDef);
        };
        if matches!(insns[d].op, Op::Phi) {
            return miss(ArithSlotMiss::Dynamic);
        }
        match &insns[d].op {
            Op::Arith(super::ir::ArithKind::Add) => {
                let lhs = insns[d]
                    .operands
                    .get(1)
                    .cloned()
                    .unwrap_or(Operand::Unknown);
                let rhs = insns[d]
                    .operands
                    .get(2)
                    .cloned()
                    .unwrap_or(Operand::Unknown);
                match (lhs, rhs) {
                    (Operand::Local(w), Operand::Immediate(c)) => {
                        acc = acc.wrapping_add(c);
                        if w == cur {
                            return miss(ArithSlotMiss::Dynamic);
                        }
                        cur = w;
                        continue;
                    }
                    (Operand::Immediate(c), Operand::Local(w)) => {
                        acc = acc.wrapping_add(c);
                        if w == cur {
                            return miss(ArithSlotMiss::Dynamic);
                        }
                        cur = w;
                        continue;
                    }
                    // `klass + scaled-idx`: two live values, no constant.
                    // This is the `v59 = v50 + v58` step of the interface-
                    // dispatch scan loop — the slot is runtime-computed.
                    _ => return miss(ArithSlotMiss::Dynamic),
                }
            }
            // Any scaled/shifted step (`v58 = v57 << 4`) is the dynamic index
            // itself. Refuse — do not fold a runtime value into a slot.
            Op::Arith(_) => return miss(ArithSlotMiss::Dynamic),
            Op::Move => {
                let src = insns[d]
                    .operands
                    .get(1)
                    .cloned()
                    .unwrap_or(Operand::Unknown);
                match src {
                    Operand::Local(w) if w != cur => {
                        cur = w;
                        continue;
                    }
                    _ => break,
                }
            }
            _ => break,
        }
    }
    let klass_op = Operand::Local(cur);
    let Some(tdi) = klass_tdi_of(insns, upto, &klass_op, table, maps, depth) else {
        return miss(ArithSlotMiss::UnknownKlass);
    };
    // Same two candidate bases as `vtable_dispatch`: the accumulated constant is
    // the entry offset, and which base it is relative to depends on the class
    // layout this image was built with (see `vtable_slot_candidates`).
    let (cands, n) = vtable_slot_candidates(acc, ptr_size);
    if n == 0 {
        return miss(ArithSlotMiss::BadOffset);
    }
    for &slot in cands.iter().take(n) {
        if let Some(hit) = lookup_vtable_slot(maps, parents, tdi, slot)
            .or_else(|| lookup_interface_slot(parents, tdi, slot))
        {
            ARITH_SLOT_HITS.fetch_add(1, Ordering::Relaxed);
            return Some(hit);
        }
    }
    miss(ArithSlotMiss::BadOffset)
}

/// Resolve the TypeDef of the object a vtable load is dispatched through.
///
/// `klass_slot` is the operand of the slot load's base, i.e. the il2cpp
/// `Il2CppClass*`. Three ways it materializes:
///   * `Immediate(kaddr)` — the class pointer folded into an `adrp/add` pair;
///   * `Local(kl)` with `kl ← [recv]` — the ordinary `obj → klass` load, whose
///     TypeDef comes from the receiver's recovered managed type;
///   * `Local(kl)` with `kl ← Immediate(kaddr)` — a class pointer moved
///     through a register;
/// plus plain register moves in between, which are followed.
fn klass_tdi_of(
    insns: &[Insn],
    upto: usize,
    klass_slot: &Operand,
    table: &LocalTable,
    maps: &ResolveMaps,
    depth: u8,
) -> Option<i32> {
    match klass_slot {
        Operand::Immediate(k) => maps.klass_to_tdi.get(&(*k as u64)).copied(),
        Operand::Local(kl) if depth > 0 => {
            let d = ssa_def_of(insns, upto, *kl)?;
            // A merge version carries many different klasses — nothing exact
            // to key on. Refusing here keeps a wrong name out of the output.
            if matches!(insns[d].op, Op::Phi) {
                return None;
            }
            // ONLY a `Move` defines a plain value copy. An `Arith` def
            // (`v60 = v59 + 312`) is an ADDRESS computation, not the klass
            // itself: following its LHS (`v59`) as "the klass" drops the
            // constant/dynamic offset the slot needs and keys the table on a
            // slot the site never loads. Those bases belong to
            // `resolve_arith_slot`, which accounts for the constant — never
            // here.
            if !matches!(insns[d].op, Op::Move) {
                return None;
            }
            match insns[d].operands.get(1)? {
                Operand::Memory {
                    base: obj,
                    addend: 0,
                    index: None,
                    ..
                } => match obj.as_ref() {
                    Operand::Local(recv) => seed_class_tdi(table, *recv)
                        .or_else(|| receiver_tdi_through_fields(insns, d, *recv, 4)),
                    Operand::Immediate(k) => maps.klass_to_tdi.get(&(*k as u64)).copied(),
                    _ => None,
                },
                Operand::Immediate(k) => maps.klass_to_tdi.get(&(*k as u64)).copied(),
                src @ Operand::Local(w) if *w != *kl => {
                    klass_tdi_of(insns, d, src, table, maps, depth - 1)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Resolve a call TARGET operand to the managed method it dispatches to.
///
/// `target` is a *value* (the callee address), not a method reference: after
/// SSA it is a `Local` version whose definition is the load of a vtable slot
/// — `x9 ← [klass + 0x138 + slot*8]` — which the `BLR` lifter arm no longer
/// keeps in a dedicated opcode. Shapes handled: the slot load itself, a plain
/// register move of it, and a constant address (`Immediate`), which is a
/// direct call whose target the lifter could not name at lift time.
/// Byte size of one `Il2CppClass` vtable entry on 64-bit targets.
///
/// The vtable is an array of `VirtualInvokeData { methodPtr; method; }` — TWO
/// pointers per entry, not one. Striding by 8 therefore mis-numbered every
/// slot: `XmlSchemaObjectCollection`'s indexer entry sits at byte 776 of the
/// class, which is slot 29 (`(776-0x138)/16`) and IS declared, but `/8` read it
/// as slot 58, which no class in the hierarchy declares — so the resolver
/// missed it and reported "no such slot" rather than "found it".
///
/// It also explains the second load the call site performs: at entry+8 it
/// fetches `method` (`MethodInfo*`) and passes it as the trailing hidden
/// argument of the virtual invoke, which is exactly what the ABI does.
const VTABLE_ENTRY_64: i64 = 16;

// ---------------------------------------------------------------------------
// Interface dispatch
// ---------------------------------------------------------------------------
//
// An interface call is the SAME machine shape as a class virtual call:
// `BLR [klass + vtable_base + slot*entry]`. The only difference is WHICH slot:
// IL2CPP keeps ONE `Il2CppClass::vtable` array holding the class's own virtuals
// first, then the methods of every interface it implements, each interface's
// block beginning at `Il2CppRuntimeInterfaceOffsetPair::offset` SLOTS into that
// array. So an interface call's slot is `interfaceOffset + method.slot`, and it
// is not declared in the class's own `vtable_slots` — `lookup_vtable_slot`
// returns `None`, and before this the call rendered `fnptr_0x…` forever.
//
// The table is a process-global rather than a `ResolveMaps` field on purpose:
// `RecoveryContext::build` constructs `ResolveMaps` with an EXHAUSTIVE struct
// literal, so adding a field would force a change in another module. A global
// matches how `runtime_helpers` already publishes per-dump identity, keeps this
// slice self-contained, and — because it is empty until installed — leaves the
// resolver's behaviour byte-for-byte unchanged when it is not wired in.

/// `(class TypeDef, vtable slot) → interface method` for the whole dump.
static INTERFACE_DISPATCH: OnceLock<HashMap<(i32, usize), MethodRef>> = OnceLock::new();

/// Install the interface table for this dump.
///
/// Called ONCE from `RecoveryContext::build` (see the integration note with this
/// change) with the value of [`build_interface_dispatch`]. Idempotent: the
/// first install wins, a later one is ignored — a dump is analysed once per
/// process.
pub fn install_interface_dispatch(map: HashMap<(i32, usize), MethodRef>) {
    let _ = INTERFACE_DISPATCH.set(map);
}

/// The breakdown that explains the installed table's size.
///
/// `interface_dispatch_len()` is a bare count, and a bare count cannot answer
/// "is the table small because the metadata declares few interfaces, or because
/// the interface-offset layout was mis-parsed?" — two questions with opposite
/// fixes. [`build_interface_dispatch_detailed`] already computes the breakdown
/// that separates them; this keeps it so a caller other than the builder can
/// report it (the builder is called once, from `RecoveryContext::build`, long
/// before any run summary exists).
static INTERFACE_DISPATCH_STATS: OnceLock<InterfaceDispatchStats> = OnceLock::new();

/// Install the breakdown that accompanies [`install_interface_dispatch`].
///
/// Idempotent for the same reason as the table itself: a dump is analysed once
/// per process.
pub fn install_interface_dispatch_stats(stats: InterfaceDispatchStats) {
    let _ = INTERFACE_DISPATCH_STATS.set(stats);
}

/// How many interface slots resolved — a reportable number, so "interface
/// dispatch did nothing" is visible instead of silent.
pub fn interface_dispatch_len() -> usize {
    INTERFACE_DISPATCH.get().map_or(0, |m| m.len())
}

/// The installed interface-dispatch breakdown, rendered for a report.
///
/// One text, shared by the build-time `ILR_IFACE` probe and the run report, so
/// the two can never disagree about what the numbers mean. `None` when no table
/// was installed (a caller that never built one, e.g. a unit test).
///
/// Fixed order, no iteration: the output is byte-stable by construction, which
/// `det_gate.sh` requires.
pub fn interface_dispatch_summary() -> Option<String> {
    let stats = INTERFACE_DISPATCH_STATS.get().copied()?;
    Some(format!(
        concat!(
            "[iface] {} interface dispatch slot(s) installed\n",
            "[iface]   from {} pair(s) over {} type(s) (counts derived from the pair ",
            "array because the stored count was unusable: {})\n",
            "[iface]   pairs: {} with a type_index naming no TypeDef, {} where the ",
            "interface placed at no slot\n",
            "[iface]     no slot: {} declares no methods, {} no usable slot, {} LOST ",
            "IN THE JOIN ({} plain, {} generic-only), {} unclassified\n",
            "[iface]   slots built: {} ({} from interfaces' DECLARED methods, which ",
            "have no native body), dropped because two interfaces claimed the ",
            "same slot: {}",
        ),
        interface_dispatch_len(),
        stats.pairs_seen,
        stats.types_with_pairs,
        stats.counts_derived,
        stats.pairs_unresolved_type,
        stats.pairs_without_slots,
        stats.pairs_without_slots_no_methods,
        stats.pairs_without_slots_slotless,
        stats.pairs_without_slots_unindexed_plain + stats.pairs_without_slots_unindexed_generic,
        stats.pairs_without_slots_unindexed_plain,
        stats.pairs_without_slots_unindexed_generic,
        stats.pairs_without_slots_unclassified,
        stats.slots_built,
        stats.slots_from_declared_interfaces,
        stats.conflicts_dropped,
    ))
}

/// True when two refs name the same method. Used to fold a slot two DIFFERENT
/// interfaces both claim: the runtime has exactly one entry there, and without
/// an explicit-implementation record neither is provable, so both are dropped
/// rather than guessed (`same_method` keeps the benign case where the same
/// method is reached twice through a diamond).
fn same_method(a: &MethodRef, b: &MethodRef) -> bool {
    a.declaring_namespace == b.declaring_namespace
        && a.declaring_type_name == b.declaring_type_name
        && a.name == b.name
        && a.sig.is_static == b.sig.is_static
}

/// Build the interface-dispatch table from metadata.
///
/// Kept OUT of the per-method pass so it runs once per dump: it is a pure
/// function of the class/interface layout, not of any method body.
///
/// `iface_tdi_of` maps an `Il2CppInterfaceOffset::type_index` (an `Il2CppType`
/// index) to the TypeDef index of the interface it names; the caller supplies it
/// because that conversion belongs to the executor. The interface's OWN method
/// slots come from the same `vtable_slots` map the class virtuals use — an
/// interface method carries the slot metadata assigned it, exactly like a class
/// virtual — so no new metadata walk is needed for the slots themselves.
///
/// A `(class, slot)` that two different interfaces both claim is REMOVED, not
/// guessed. That refusal is the whole safety story: an interface call site and a
/// class virtual call site are indistinguishable from the bytes, so the only
/// thing that keeps a wrong name out of the output is requiring the placement to
/// be unique.
/// THE TABLE-LESS ENTRY POINT: it receives no method table, so the breakdown
/// cannot say WHY a pair found no slot and every such pair is reported
/// `unclassified`. Callers that report the breakdown use
/// [`build_interface_dispatch_detailed`] directly; this exists for callers that
/// only want the table.
pub fn build_interface_dispatch(
    type_defs: &[crate::il2cpp::structures::Il2CppTypeDefinition],
    interface_offsets: &[crate::il2cpp::structures::Il2CppInterfaceOffset],
    vtable_slots: &HashMap<(i32, usize), MethodRef>,
    iface_tdi_of: &dyn Fn(i32) -> Option<i32>,
) -> HashMap<(i32, usize), MethodRef> {
    build_interface_dispatch_detailed(
        type_defs,
        interface_offsets,
        &[],
        vtable_slots,
        &HashMap::new(),
        iface_tdi_of,
    )
    .0
}

/// WHICH WAY a lookup for an interface pair came up empty.
///
/// A bare count cannot tell a FACT about the metadata from a DEFECT in this
/// crate's join, and the two want opposite responses. `DeclaresNoMethods` and
/// `NoUsableSlot` are facts: there is nothing to place and dropping the pair is
/// correct. `Unindexed*` is a defect: the interface's own method carries a slot
/// in the metadata, so `(class, pair.offset + method.slot)` is a placement the
/// metadata proves, and the join lost it. Every lost placement is a call site
/// that renders `fnptr_0x…` forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterfaceSlotLoss {
    /// The interface TypeDef declares no methods at all.
    DeclaresNoMethods,
    /// It declares methods, none carrying a usable slot
    /// (`kInvalidIl2CppMethodSlot`).
    NoUsableSlot,
    /// A method with a usable slot and NO generic container exists — so the
    /// loss is the body filter, not the generic one: the slot map only holds
    /// methods with a native body, and an interface method is normally abstract.
    UnindexedPlain,
    /// Only generic-container methods carry a usable slot: the generic filter in
    /// the slot-map construction excludes them before the body question arises.
    UnindexedGeneric,
    /// No method table was supplied, so no claim is made.
    Unclassified,
}

/// Classify an interface that placed at no slot — see [`InterfaceSlotLoss`].
///
/// The interface's OWN method list is what is examined, because that is where
/// the slot that would place the pair comes from: `Il2CppRuntimeInterfaceOffsetPair`
/// gives the base, and the interface method's own `slot` gives the lane.
fn interface_slot_loss(
    type_defs: &[crate::il2cpp::structures::Il2CppTypeDefinition],
    method_defs: &[crate::il2cpp::structures::Il2CppMethodDefinition],
    itdi: i32,
) -> InterfaceSlotLoss {
    let Some(td) = type_defs.get(itdi as usize) else {
        return InterfaceSlotLoss::Unclassified;
    };
    let start = td.method_start.max(0) as usize;
    let count = td.method_count.max(0) as usize;
    if count == 0 {
        return InterfaceSlotLoss::DeclaresNoMethods;
    }
    if method_defs.is_empty() {
        return InterfaceSlotLoss::Unclassified;
    }
    let (mut plain, mut generic) = (0usize, 0usize);
    for mi in start..start.saturating_add(count) {
        let Some(md) = method_defs.get(mi) else {
            continue;
        };
        if md.slot == u16::MAX {
            continue;
        }
        if md.generic_container_index == -1 {
            plain += 1;
        } else {
            generic += 1;
        }
    }
    if plain > 0 {
        InterfaceSlotLoss::UnindexedPlain
    } else if generic > 0 {
        InterfaceSlotLoss::UnindexedGeneric
    } else {
        InterfaceSlotLoss::NoUsableSlot
    }
}

/// Why an interface-dispatch table is as small (or as large) as it is.
///
/// `interface_dispatch_len()` alone cannot distinguish "the metadata declares
/// no interfaces" from "the interface-offsets layout was mis-parsed", and the
/// two want opposite fixes. This is the breakdown that separates them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InterfaceDispatchStats {
    /// Types whose pair slice was non-empty.
    pub types_with_pairs: usize,
    /// Slices whose count came from the neighbour derivation rather than the
    /// stored field — the signature of a layout whose count field is absent or
    /// unusable.
    pub counts_derived: usize,
    /// Interface-offset pairs examined.
    pub pairs_seen: usize,
    /// Pairs whose `type_index` did not name a TypeDef (`iface_tdi_of` refused:
    /// a generic instantiation, a sentinel, or an out-of-range index).
    pub pairs_unresolved_type: usize,
    /// Pairs whose interface placed at no slot — the total of the
    /// `pairs_without_slots_*` counters below, and the number that matters when
    /// it is large, because each such pair is a call site left as `fnptr_0x…`
    /// forever.
    pub pairs_without_slots: usize,
    /// Of those, interfaces that declare no method at all — a fact about the
    /// metadata, and a correct drop.
    pub pairs_without_slots_no_methods: usize,
    /// Of those, interfaces whose methods carry no usable slot — also a fact:
    /// the metadata offers nothing to place.
    pub pairs_without_slots_slotless: usize,
    /// Of those, interfaces whose own method carries a usable slot and NO
    /// generic container, yet was absent from the slot map. This is the defect
    /// arm: the slot map is built from methods that have a native BODY, and an
    /// interface method is normally abstract, so the join loses a placement the
    /// metadata proves exists.
    pub pairs_without_slots_unindexed_plain: usize,
    /// Of those, interfaces where only generic-container methods carry a usable
    /// slot — excluded from the slot map a second time, by the generic filter.
    pub pairs_without_slots_unindexed_generic: usize,
    /// Of those, pairs with no method table to judge (the table-less entry
    /// point), so no claim is made either way.
    pub pairs_without_slots_unclassified: usize,
    /// Placements that came from an interface's DECLARED methods — the ones with
    /// no native body, which the class slot map cannot hold. Before
    /// `iface_slots` existed every one of these was a dropped pair, so this
    /// counter is the size of that defect where it was measured: 7,192 of 7,331
    /// lost pairs on the x64 Unity-6 corpus, 11,574 of 12,122 on the iOS arm64
    /// one (see `interface_dispatch_summary`).
    pub slots_from_declared_interfaces: usize,
    /// `(class, vtable slot)` entries placed.
    pub slots_built: usize,
    /// Entries dropped because two different interfaces claimed the slot.
    pub conflicts_dropped: usize,
}

/// How many interface-offset pairs a TypeDef owns, derived defensively.
///
/// `Il2CppTypeDefinition::interface_offsets_count` is authoritative when it is
/// present AND fits the pair array. It is not present in every metadata layout
/// this reader accepts — the field only exists from v27 on, while the reader
/// reads it unconditionally — so a count of 0, or one running past the array,
/// can be an artefact of the layout rather than a fact about the type. (A
/// 40x gap between two corpora — 2,142 slots on one, 52 on the other — is the
/// signature of exactly this, not of a corpus that implements 40x fewer
/// interfaces.)
///
/// For those two cases the slice is re-derived from the array itself: metadata
/// emits TypeDefs in order and each type's pairs are the contiguous slice
/// `[start, next.start)`, so the next type's start delimits this type's pairs,
/// and the last type owns the tail of the array. The derivation can only ADD
/// pairs the stored count refused, never remove one it claimed — so it cannot
/// turn a correct table into a wrong one, because every added pair still has to
/// resolve through `iface_tdi_of` and still has to land on a method slot the
/// interface itself declares.
///
/// Returns `(count, derived)`.
fn interface_pair_count(
    type_defs: &[crate::il2cpp::structures::Il2CppTypeDefinition],
    index: usize,
    pairs_len: usize,
) -> (usize, bool) {
    let td = &type_defs[index];
    if td.interface_offsets_start < 0 {
        return (0, false);
    }
    let start = td.interface_offsets_start as usize;
    if start >= pairs_len {
        return (0, false);
    }
    let stored = td.interface_offsets_count as usize;
    // The count field is authoritative, INCLUDING when it is zero. `0` is the
    // ordinary "this type implements no interfaces", and in valid metadata the
    // derivation below would yield 0 for it anyway (a type that contributed no
    // pairs has the same start as the next type). Using the stored count first
    // is therefore free where the metadata is sound, and it is the only thing
    // that stops a plain class being handed its neighbour's pairs where the
    // metadata is not.
    if start + stored <= pairs_len {
        return (stored, false);
    }
    // The stored count is provably inconsistent with the array (it runs past the
    // end), so the array is the only trustworthy witness: metadata emits
    // TypeDefs in order, so this type's pairs are the contiguous slice up to the
    // next type's start, and the last type owns the tail.
    let derived = match type_defs.get(index + 1) {
        Some(next) => {
            let n = next.interface_offsets_start - td.interface_offsets_start;
            if n > 0 { n as usize } else { 0 }
        }
        None => pairs_len - start,
    };
    (derived.min(pairs_len - start), true)
}

/// [`build_interface_dispatch`] with the breakdown that explains its size.
/// `method_defs` is consulted ONLY to classify a pair that placed at no slot
/// (`interface_slot_loss`): the interface's own method list is what proves
/// whether a placement existed to lose. It never contributes a placement
/// itself, so passing an empty table cannot change the returned map — only the
/// honesty of the breakdown.
///
/// `iface_slots` is the SECOND, metadata-built slot source: the interface's own
/// declared methods, which have no native body and are therefore absent from
/// `vtable_slots` (that map is filled from methods with a body). It is consulted
/// only when an interface places nothing from `vtable_slots`, and it is what
/// keeps an interface call site from rendering as `fnptr_0x…`; see the counters
/// on [`InterfaceDispatchStats`] for the measured size of the loss it repairs.
pub fn build_interface_dispatch_detailed(
    type_defs: &[crate::il2cpp::structures::Il2CppTypeDefinition],
    interface_offsets: &[crate::il2cpp::structures::Il2CppInterfaceOffset],
    method_defs: &[crate::il2cpp::structures::Il2CppMethodDefinition],
    vtable_slots: &HashMap<(i32, usize), MethodRef>,
    iface_slots: &HashMap<(i32, usize), MethodRef>,
    iface_tdi_of: &dyn Fn(i32) -> Option<i32>,
) -> (HashMap<(i32, usize), MethodRef>, InterfaceDispatchStats) {
    // Group the slots by declaring TypeDef once, so the pair walk touches only
    // the slots of the interface it is placing — the flat maps hold every method
    // in the binary (tens of thousands), and scanning them per pair would be
    // quadratic. Both sources are grouped the same way; the pair walk prefers
    // `vtable_slots` because a body-bearing declaration is the stronger witness.
    let group_by_tdi = |slots: &HashMap<(i32, usize), MethodRef>| {
        let mut by: HashMap<i32, Vec<(usize, MethodRef)>> = HashMap::new();
        for ((tdi, slot), m) in slots.iter() {
            by.entry(*tdi).or_default().push((*slot, m.clone()));
        }
        for v in by.values_mut() {
            v.sort_unstable_by_key(|(s, _)| *s);
        }
        by
    };
    let by_tdi = group_by_tdi(vtable_slots);
    let iface_by_tdi = group_by_tdi(iface_slots);

    let mut stats = InterfaceDispatchStats::default();
    let mut out: HashMap<(i32, usize), MethodRef> = HashMap::new();
    let mut conflicted: std::collections::HashSet<(i32, usize)> = Default::default();
    for (tdi, _) in type_defs.iter().enumerate() {
        let (count, derived) = interface_pair_count(type_defs, tdi, interface_offsets.len());
        if count == 0 {
            continue;
        }
        stats.types_with_pairs += 1;
        if derived {
            stats.counts_derived += 1;
        }
        let start = type_defs[tdi].interface_offsets_start as usize;
        for k in start..start.saturating_add(count) {
            let Some(pair) = interface_offsets.get(k) else {
                break;
            };
            stats.pairs_seen += 1;
            let Some(itdi) = iface_tdi_of(pair.type_index) else {
                stats.pairs_unresolved_type += 1;
                continue;
            };
            let (slots, from_decl) = match (by_tdi.get(&itdi), iface_by_tdi.get(&itdi)) {
                (Some(s), _) => (s, false),
                (None, Some(s)) => (s, true),
                (None, None) => {
                    stats.pairs_without_slots += 1;
                    match interface_slot_loss(type_defs, method_defs, itdi) {
                        InterfaceSlotLoss::DeclaresNoMethods => {
                            stats.pairs_without_slots_no_methods += 1
                        }
                        InterfaceSlotLoss::NoUsableSlot => stats.pairs_without_slots_slotless += 1,
                        InterfaceSlotLoss::UnindexedPlain => {
                            stats.pairs_without_slots_unindexed_plain += 1
                        }
                        InterfaceSlotLoss::UnindexedGeneric => {
                            stats.pairs_without_slots_unindexed_generic += 1
                        }
                        InterfaceSlotLoss::Unclassified => {
                            stats.pairs_without_slots_unclassified += 1
                        }
                    }
                    continue;
                }
            };
            let base = pair.offset.max(0) as usize;
            for (s, m) in slots {
                let key = (tdi as i32, base + *s);
                if conflicted.contains(&key) {
                    continue;
                }
                match out.get(&key) {
                    Some(existing) if !same_method(existing, m) => {
                        conflicted.insert(key);
                        out.remove(&key);
                        stats.conflicts_dropped += 1;
                    }
                    Some(_) => {}
                    None => {
                        out.insert(key, m.clone());
                        stats.slots_built += 1;
                        if from_decl {
                            stats.slots_from_declared_interfaces += 1;
                        }
                    }
                }
            }
        }
    }
    (out, stats)
}

/// Interface fallback for a slot the class's own ancestry did not declare.
///
/// Runs ONLY after `lookup_vtable_slot` missed, so a slot the class really does
/// declare (an explicit interface implementation reuses the class's own slot
/// number) is never shadowed by the interface table. The ancestry walk mirrors
/// `lookup_vtable_slot` because the interface block sits in the vtable of the
/// type that declares it — an inherited interface implementation is placed on a
/// BASE class and a derived class has no entry of its own.
fn lookup_interface_slot(
    parents: &HashMap<i32, Vec<i32>>,
    tdi: i32,
    slot: usize,
) -> Option<MethodRef> {
    // Counted BEFORE the table probe: an attempt is "the class lookup missed
    // with a known tdi, so the interface table was consulted", even when the
    // table was never installed (`None` below) or holds no entry. Counting
    // only hits would conflate never-attempted with attempted-missed.
    // Inert: the increment does not change the return value for any input.
    IFACE_LOOKUP_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    let map = INTERFACE_DISPATCH.get()?;
    let hit = interface_slot_in(map, parents, tdi, slot);
    if hit.is_some() {
        IFACE_LOOKUP_HITS.fetch_add(1, Ordering::Relaxed);
    }
    hit
}

/// The pure core of [`lookup_interface_slot`], taking the table explicitly so it
/// is unit-testable without touching the process-global.
fn interface_slot_in(
    map: &HashMap<(i32, usize), MethodRef>,
    parents: &HashMap<i32, Vec<i32>>,
    tdi: i32,
    slot: usize,
) -> Option<MethodRef> {
    if let Some(m) = map.get(&(tdi, slot)) {
        return Some(m.clone());
    }
    for &base in parents.get(&tdi)?.iter() {
        if base == tdi {
            continue;
        }
        if let Some(m) = map.get(&(base, slot)) {
            return Some(m.clone());
        }
    }
    None
}

/// Resolve `(tdi, slot)` through the class's ancestry.
///
/// A runtime vtable entry for a derived class holds whatever the hierarchy
/// puts at that slot. Metadata records the slot on the type that DECLARES the
/// method, so a class that does not override an inherited virtual has no
/// `(tdi, slot)` entry of its own — the entry is on a base class. Matching
/// only the exact declaring type therefore missed EVERY inherited virtual:
/// `XmlSchemaObjectCollection` declares slots 20–23/29/30 and inherits its
/// indexer at slot 58, so `elements[i]` could never be named.
///
/// Walking the chain is exact rather than speculative: if the derived type
/// does not declare that slot, the vtable entry IS the base's method, and an
/// override reuses the base's slot number (`parent_chains` is precomputed
/// with the type itself first, so an override is found before any base).
fn lookup_vtable_slot(
    maps: &ResolveMaps,
    parents: &HashMap<i32, Vec<i32>>,
    tdi: i32,
    slot: usize,
) -> Option<MethodRef> {
    // Counted BEFORE the ancestry probe for the same reason as the interface
    // fallback: an attempt is "a constant slot with a known klass tdi reached
    // a table", a hit is "that (tdi, slot) was declared". Inert: the map is
    // only read, never written, and the return for any input is unchanged.
    VTABLE_LOOKUP_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    if let Some(m) = maps.vtable_slots.get(&(tdi, slot)) {
        VTABLE_LOOKUP_HITS.fetch_add(1, Ordering::Relaxed);
        return Some(m.clone());
    }
    for &base in parents.get(&tdi)?.iter() {
        if base == tdi {
            continue;
        }
        if let Some(m) = maps.vtable_slots.get(&(base, slot)) {
            VTABLE_LOOKUP_HITS.fetch_add(1, Ordering::Relaxed);
            return Some(m.clone());
        }
    }
    None
}

fn resolve_indirect_target(
    insns: &[Insn],
    upto: usize,
    target: &Operand,
    table: &LocalTable,
    maps: &ResolveMaps,
    parents: &HashMap<i32, Vec<i32>>,
    ptr_size: u64,
    depth: u8,
) -> Option<MethodRef> {
    let _ = vtable_layout(ptr_size);
    match target {
        Operand::Method(m) => Some(m.clone()),
        Operand::Immediate(a) => maps.methods_by_rva.get(&(*a as u64)).cloned(),
        // A register-indexed slot (`[klass + idx*scale + off]`) is the
        // dynamic-slot shape in its indexed spelling: the slot is a runtime
        // value, so no `(tdi, slot)` key exists. Refuse and keep `fnptr_`.
        Operand::Memory { index: Some(_), .. } => {
            INDEXED_SLOT_REFUSALS.fetch_add(1, Ordering::Relaxed);
            None
        }
        // The delegate's `invoke_impl` code pointer, already named by the
        // typing sweep as an `InstanceField`/`Field` (see
        // [`is_invoke_impl_field`]). Proven through
        // [`delegate_invoke_of_def`]; unproven stays `fnptr_`.
        Operand::InstanceField { .. } | Operand::Field(_) => {
            delegate_invoke_of_def(target, table).map(|(m, _)| m)
        }
        // `call [klass + 0x138 + slot*16]` — on x86 the slot load is INLINE in
        // the call, so the call's own operand 0 IS the vtable access and there
        // is no separate `Move` to walk back to. Without this arm every x64
        // virtual call stayed unnamed even once the lifter stopped discarding
        // the target.
        Operand::Memory {
            base,
            addend,
            index: None,
            ..
        } => {
            // An `Arith`-computed base (`v60 = v59 + 312`) never forms the
            // constant `Memory` shape this arm keys on. Route it to the
            // proof-only folder instead of mis-keying it here.
            if let Operand::Local(bv) = base.as_ref() {
                if depth > 0 {
                    if let Some(d) = ssa_def_of(insns, upto, *bv) {
                        if matches!(insns[d].op, Op::Arith(_)) {
                            return resolve_arith_slot(
                                insns,
                                upto,
                                *bv,
                                *addend,
                                table,
                                maps,
                                parents,
                                ptr_size,
                                depth,
                            );
                        }
                    }
                }
            }
            vtable_dispatch(
                insns, upto, base, *addend, table, maps, parents, ptr_size, depth,
            )
        }
        Operand::Local(v) if depth > 0 => {
            let Some(d) = ssa_def_of(insns, upto, *v) else {
                VC_NO_DEF_SITES.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            if matches!(insns[d].op, Op::Phi) {
                return None;
            }
            // A target defined by an address computation itself (not a load)
            // is not a callee value. The `Arith`-base slot load is handled
            // one level down (the `Memory` arm above routes its base here);
            // a bare `Arith` target has no slot to key.
            if matches!(insns[d].op, Op::Arith(_)) {
                ARITH_SLOT_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
                ARITH_SLOT_REFUSED_DYNAMIC.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            match insns[d].operands.get(1)? {
                Operand::Memory { index: Some(_), .. } => {
                    INDEXED_SLOT_REFUSALS.fetch_add(1, Ordering::Relaxed);
                    None
                }
                Operand::Memory {
                    base,
                    addend,
                    index: None,
                    ..
                } => {
                    // Same `Arith`-base routing as the inline arm: `fn =
                    // [v60]` with `v60 = v59 + 312` never forms a keyable
                    // `(tdi, slot)` here.
                    if let Operand::Local(bv) = base.as_ref() {
                        if let Some(bd) = ssa_def_of(insns, d, *bv) {
                            if matches!(insns[bd].op, Op::Arith(_)) {
                                return resolve_arith_slot(
                                    insns,
                                    d,
                                    *bv,
                                    *addend,
                                    table,
                                    maps,
                                    parents,
                                    ptr_size,
                                    depth - 1,
                                );
                            }
                        }
                    }
                    vtable_dispatch(
                        insns,
                        d,
                        base,
                        *addend,
                        table,
                        maps,
                        parents,
                        ptr_size,
                        depth - 1,
                    )
                }
                // The delegate shape after typing: `t = base.invoke_impl`.
                // Proven through [`delegate_invoke_of_def`]; unproven stays
                // `fnptr_`. One small step from the `Memory` shape
                // `recover_delegate_invokes` still handles: the field name is
                // the same proof, only the spelling changed.
                src @ (Operand::InstanceField { .. } | Operand::Field(_)) => {
                    delegate_invoke_of_def(src, table).map(|(m, _)| m)
                }
                // `mov x9, x10` — the callee pointer moved through a register.
                src @ Operand::Local(w) if *w != *v => resolve_indirect_target(
                    insns,
                    d,
                    src,
                    table,
                    maps,
                    parents,
                    ptr_size,
                    depth - 1,
                ),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Count an unresolvable klass that traces to an immediate klass pointer.
///
/// `klass_to_tdi` misses for two very different reasons — an immediate klass
/// address the map never covered (a metadata gap, 72 sites on the gate
/// corpus) versus a computed/untyped receiver (a typing/shape gap). Only the
/// first is counted here, so the residue splits into addressable buckets
/// instead of one opaque "unknown klass".
fn count_unknown_klass_imm(base: &Operand, insns: &[Insn], upto: usize, maps: &ResolveMaps) {
    let imm = match base {
        Operand::Immediate(k) => Some(*k as u64),
        Operand::Local(kl) => {
            ssa_def_of(insns, upto, *kl).and_then(|d| match insns[d].operands.get(1) {
                Some(Operand::Immediate(k)) => Some(*k as u64),
                _ => None,
            })
        }
        _ => None,
    };
    if let Some(k) = imm {
        if !maps.klass_to_tdi.contains_key(&k) {
            VC_UNKNOWN_KLASS_IMM_SITES.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The delegate base for a call target, WITHOUT counting.
///
/// Same proof as [`delegate_invoke_of_def`] but inert for the rewriter: the
/// resolver already counted the attempt/hit, and counting again would double
/// every delegate site. Used by `rewrite_virtual_calls` to fix the receiver
/// of a synthesized `Invoke` to the proven delegate local.
fn delegate_base_nocount(src: &Operand, table: &LocalTable) -> Option<u32> {
    match src {
        Operand::InstanceField { base, field } if is_invoke_impl_field(field) => {
            let Operand::Local(d) = base.as_ref() else {
                return None;
            };
            delegate_named_ty(table, *d).map(|_| *d)
        }
        Operand::Field(field) if is_invoke_impl_field(field) => {
            this_delegate_base(table).map(|(v, _)| v)
        }
        _ => None,
    }
}

/// Is this call argument one of the delegate's hidden runtime fields?
///
/// The native delegate invoke passes `method_code`/`method`/`invoke_impl`
/// (all `InstanceField`/`Field` loads off the SAME delegate base) as leading
/// ABI arguments; the managed `Invoke` takes only the USER parameters.
/// Keeping the hidden loads as `Invoke` params would render
/// `Invoke(method_code, method, …)` — the exact `fnptr_` shape this fix
/// removes — while dropping a USER argument would lose semantics. The proof
/// is the definition: a `Local` whose `Move` source is an `InstanceField`
/// off the delegate base, or a `Field` with a `Delegate` declaring type (the
/// `this`-delegate hidden loads). Anything else — including the delegate
/// base itself — is kept. Never drops a write, never invents a value.
fn is_hidden_delegate_arg(insns: &[Insn], upto: usize, op: &Operand, delegate_base: u32) -> bool {
    // Direct field operand off the same base (not yet materialized).
    match op {
        Operand::InstanceField { base, field }
            if field.declaring_type_name.contains("Delegate") =>
        {
            if matches!(base.as_ref(), Operand::Local(b) if *b == delegate_base) {
                return true;
            }
        }
        Operand::Field(field) if field.declaring_type_name.contains("Delegate") => {
            return true;
        }
        _ => {}
    }
    let Operand::Local(v) = op else { return false };
    let Some(d) = ssa_def_of(insns, upto, *v) else {
        return false;
    };
    if !matches!(insns[d].op, Op::Move) {
        return false;
    }
    match insns[d].operands.get(1) {
        Some(Operand::InstanceField { base, field })
            if field.declaring_type_name.contains("Delegate") =>
        {
            matches!(base.as_ref(), Operand::Local(b) if *b == delegate_base)
        }
        Some(Operand::Field(field)) if field.declaring_type_name.contains("Delegate") => true,
        _ => false,
    }
}

/// The delegate base for a whole call target operand, WITHOUT counting.
///
/// Follows `Move` copies exactly like [`resolve_indirect_target`] so a
/// `t = u; u = base.invoke_impl` chain still finds `base`. Returns the
/// delegate-local version that must become the rewritten call's receiver.
fn delegate_base_of_target_nocount(
    insns: &[Insn],
    upto: usize,
    target: &Operand,
    table: &LocalTable,
    depth: u8,
) -> Option<u32> {
    match target {
        Operand::InstanceField { .. } | Operand::Field(_) => delegate_base_nocount(target, table),
        Operand::Local(v) if depth > 0 => {
            let d = ssa_def_of(insns, upto, *v)?;
            if matches!(insns[d].op, Op::Phi | Op::Arith(_)) {
                return None;
            }
            match insns[d].operands.get(1)? {
                src @ (Operand::InstanceField { .. } | Operand::Field(_)) => {
                    delegate_base_nocount(src, table)
                }
                src @ Operand::Local(w) if *w != *v => {
                    delegate_base_of_target_nocount(insns, d, src, table, depth - 1)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Offset of `Il2CppClass::vtable` (64-bit layout).
///
/// **0x130, not 0x138.** Every `Il2CppClass` layout table in circulation
/// (including this repo's `assets/UnityHeaders/*` and Cpp2IL's
/// `Il2CppClassUsefulOffsets`) says 0x138 — but 0x138 ≡ 8 (mod 16) while a
/// `VirtualInvokeData` array must start on a 16-byte boundary, so under 0x138
/// the `methodPtr` half of every entry lands off the entry grid and the
/// resolver's `(off - base) % 16 == 0` test rejects it. That is not a
/// theoretical worry: it is why **zero** virtual calls resolved on every x64
/// dump and on the ARM64 iOS dump, with thousands of sites reported as
/// "offset N is not a vtable methodPtr" — all of them ≡ 0 (mod 16).
///
/// The two candidate bases accept DISJOINT offset sets (0x130 ⟺ ≡0, 0x138 ⟺
/// ≡8), so the right one is decidable rather than a guess, and the decisive
/// measurement is the resolver's own validated hit count — every accepted
/// offset still has to land on a slot the receiver's class DECLARES:
///
/// ```text
/// v108 PE  (37,644 methods):  base 0x138 → 0 resolved / 217,504 reached
///                             base 0x130 → 64 resolved / 217,440 reached
/// Simple_2019_4_34 (x64):     base 0x138 → 0 resolved
/// Simple_2022_3_35 (x64):     base 0x138 → 0 resolved
/// libs/ios/UnityFramework:    base 0x138 → 0 resolved
/// ```
///
/// The x64/ARM64 native pairing confirms it independently: the callee load is
/// always `call [[obj] + N]` with the companion `MethodInfo` at `N + 8` — the
/// `VirtualInvokeData { methodPtr; method; }` halves — so `N` is ≡ 0 (mod 16)
/// and the base is too. (Under 0x138 the same `N` gives `(N - 0x138) / 16` =
/// slot + 0.5, i.e. it is not an entry boundary at all.)
///
/// `ILR_VBASE=0x…` overrides it, so the claim stays checkable.
///
/// The 32-bit base is **0xC0**, and unlike the 64-bit one it is NOT taken from
/// any layout table in circulation: Cpp2IL's `Il2CppClassUsefulOffsets` only
/// covers the 64-bit class. It was measured instead. Every ARM32 virtual
/// dispatch is the pair `ldr rD,[klass, #N]; ldr rD2,[klass, #N+4]` (the
/// `VirtualInvokeData { methodPtr; method }` halves) followed by `blx rD`, so
/// decoding those pairs across `libil2cpp.so` yields the set of real `N`
/// values. That set starts at exactly 0xC0 with an 8-byte stride (0xC0, 0xD0,
/// 0xD8, 0xE0, …, 0x110, …) — the same 0xC0 the 32-bit `Il2CppClass` struct
/// gives for `vtable[32]` (4-byte pointers put it 0x70 bytes before the 64-bit
/// array). The earlier guess of 0x110 was a slot, not the base; with it every
/// slot number was off and nothing resolved.
fn vtable_layout(ptr_size: u64) -> (i64, i64) {
    let ovr = std::env::var("ILR_VBASE")
        .ok()
        .and_then(|v| i64::from_str_radix(v.trim_start_matches("0x"), 16).ok());
    if ptr_size == 4 {
        (ovr.unwrap_or(0xC0), 8)
    } else {
        (ovr.unwrap_or(0x130), VTABLE_ENTRY_64)
    }
}

/// The vtable SLOT numbers an entry offset can denote, in preference order.
///
/// `Il2CppClass::vtable` is at **0x130 on the builds the constant was measured
/// on, and at 0x138 on the v29+ layout**, where four `uint16_t` counters
/// (method/property/field/event counts) were inserted before the array. This
/// image is the second kind, and the machine code says so directly:
///
/// ```text
/// 0x148a4f4  ldr x9, [x0]              ; x9 = klass
/// 0x148a4fc  ldp x8, x1, [x9, #0x188] ; methodPtr = [klass+0x188], MethodInfo = [klass+0x190]
/// 0x148a500  blr x8
/// ```
///
/// The IR agrees: `Local(84) = [Local(82) + 392]`, `Local(85) = [Local(82) +
/// 400]`, `Call [Local(84), …, Local(81), Local(85), …]` — the `methodPtr` /
/// `MethodInfo` halves of ONE `VirtualInvokeData` (stride 16). 392 is
/// `8 (mod 16)`, so the 0x130 rule rejects every site of this shape as "not a
/// vtable methodPtr" — which is exactly what `ILR_VC` reports, at 100 % of the
/// computed-callee misses on this corpus.
///
/// Trying both bases cannot mis-key a call, because the two accept DISJOINT
/// offset sets (`base + k*16` for 0x130 is `≡0`; for 0x138 it is `≡8`) and a
/// CALL TARGET is always the `methodPtr` half: the `MethodInfo` half is passed
/// as an argument, never branched to. Under a 0x130 image the `≡8` offsets are
/// therefore MethodInfo pointers that no `BLR` targets; under a 0x138 image
/// the `≡0` offsets are not entry starts at all. Each candidate still has to
/// land on a `(tdi, slot)` the receiver's class DECLARES, which is the same
/// validation the single base had, so the accepted set only grows where the
/// metadata can confirm it.
///
/// 32-bit keeps its single measured base: the ARM32 entry stride/offset set was
/// measured, not derived, and no ARM32 image is available here to re-measure.
fn vtable_slot_candidates(addend: i64, ptr_size: u64) -> ([usize; 2], usize) {
    let (base, entry) = vtable_layout(ptr_size);
    let mut out = [0usize; 2];
    let mut n = 0;
    let step = addend - base;
    if step >= 0 && step % entry == 0 {
        out[n] = (step / entry) as usize;
        n += 1;
    }
    if ptr_size == 8 && entry % 2 == 0 {
        let alt = base + entry / 2;
        let step = addend - alt;
        if step >= 0 && step % entry == 0 {
            out[n] = (step / entry) as usize;
            n += 1;
        }
    }
    (out, n)
}

/// Key a vtable entry offset against the receiver's class (and its ancestry).
///
/// The single place that turns "a call through `[klass + off]`" into a method,
/// shared by the constant-`Memory` arm and the `Arith`-base arm so the two can
/// never disagree about which bases are tried or how a hit is validated.
fn vtable_dispatch(
    insns: &[Insn],
    upto: usize,
    klass_op: &Operand,
    addend: i64,
    table: &LocalTable,
    maps: &ResolveMaps,
    parents: &HashMap<i32, Vec<i32>>,
    ptr_size: u64,
    depth: u8,
) -> Option<MethodRef> {
    let (cands, n) = vtable_slot_candidates(addend, ptr_size);
    if n == 0 {
        return None;
    }
    let Some(tdi) = klass_tdi_of(insns, upto, klass_op, table, maps, depth) else {
        count_unknown_klass_imm(klass_op, insns, upto, maps);
        return None;
    };
    for &slot in cands.iter().take(n) {
        if let Some(m) =
            lookup_vtable_slot(maps, parents, tdi, slot).or_else(|| lookup_interface_slot(parents, tdi, slot))
        {
            return Some(m);
        }
    }
    None
}

/// `ILR_VC=1` diagnostic: why an indirect call could not be named.
fn report_vc_miss(
    insns: &[Insn],
    i: usize,
    target: &Operand,
    table: &LocalTable,
    maps: &ResolveMaps,
    parents: &HashMap<i32, Vec<i32>>,
    ptr_size: u64,
) {
    let (vtable_base, entry) = vtable_layout(ptr_size);
    let addr = insns[i].address;
    // Delegate shape first: the typing sweep names `[delegate + off]` as an
    // `InstanceField`/`Field`, so the "is not a memory load" line below would
    // otherwise hide that the site IS a delegate invoke whose base was (or
    // was not) proven. Say which, so the delegate counters have a per-site
    // witness.
    let delegate_note = |v: u32, ops: &[Operand]| {
        let src = ops.get(1).cloned().unwrap_or(Operand::Unknown);
        match &src {
            Operand::InstanceField { base, field } if is_invoke_impl_field(field) => {
                let proven = match base.as_ref() {
                    Operand::Local(d) => delegate_named_ty(table, *d)
                        .map(|nt| format!("{}.{}", nt.namespace, nt.name))
                        .unwrap_or_else(|| "(unproven base)".into()),
                    other => format!("(non-local base {other:?})"),
                };
                eprintln!(
                    "[vc] miss {addr:#x} v{v} delegate invoke_impl off {proven} base: {:?}",
                    ops
                );
                true
            }
            Operand::Field(field) if is_invoke_impl_field(field) => {
                let proven = this_delegate_base(table)
                    .map(|(_, nt)| format!("{}.{}", nt.namespace, nt.name))
                    .unwrap_or_else(|| "(no delegate this)".into());
                eprintln!(
                    "[vc] miss {addr:#x} v{v} delegate invoke_impl (this-field) {proven}: {:?}",
                    ops
                );
                true
            }
            _ => false,
        }
    };
    // Direct field target (no `Move` in between).
    if matches!(target, Operand::InstanceField { .. } | Operand::Field(_)) {
        if delegate_note(u32::MAX, &[Operand::Unknown, target.clone()]) {
            return;
        }
    }
    // Indexed slot: dynamic by construction.
    if matches!(target, Operand::Memory { index: Some(_), .. }) {
        eprintln!("[vc] miss {addr:#x} [mem] call has a register index (dynamic slot): {target:?}");
        return;
    }
    // The x86 `call [klass + vtable]` form carries the slot access INLINE, so
    // the call's operand 0 IS the `Memory` and there is no local to report a
    // definition from. Reporting only "target is not a value" for it hid the
    // reason for every one of those sites (the largest single class of
    // unresolved indirect call on an x64 build). Walk the Memory form the same
    // way `resolve_indirect_target` does and say what was missing.
    let (base, addend, at, what) = match target {
        Operand::Local(v) => {
            let Some(d) = ssa_def_of(insns, i, *v) else {
                eprintln!("[vc] miss {addr:#x} v{v} has no definition (delegate/ICall)");
                return;
            };
            // Delegate field load (see above): report the proof state, not a
            // generic "not a memory load".
            if delegate_note(*v, &insns[d].operands) {
                return;
            }
            // Indexed load through a versioned local.
            if matches!(
                insns[d].operands.get(1),
                Some(Operand::Memory { index: Some(_), .. })
            ) {
                eprintln!(
                    "[vc] miss {addr:#x} v{v} indexed slot load (dynamic slot): {:?}",
                    insns[d].operands
                );
                return;
            }
            // `Arith`-base slot load (`fn = [v60]`, `v60 = v59 + 312`): the
            // slot is computed at runtime through the chain. Show the chain
            // head so the dynamic shape is auditable (the full chain is in
            // the stage dump; the counters say how many such sites exist).
            if let Some(Operand::Memory {
                base,
                addend: _,
                index: None,
                ..
            }) = insns[d].operands.get(1).cloned()
            {
                if let Operand::Local(bv) = base.as_ref() {
                    if let Some(bd) = ssa_def_of(insns, d, *bv) {
                        if matches!(insns[bd].op, Op::Arith(_)) {
                            eprintln!(
                                "[vc] miss {addr:#x} v{v} Arith-base slot: base v{bv} <- {:?} (dynamic klass+idx*16+312 shape; see arith-slot counters)",
                                insns[bd].operands
                            );
                            return;
                        }
                    }
                }
            }
            let Operand::Memory {
                base,
                addend,
                index: None,
                ..
            } = insns[d]
                .operands
                .get(1)
                .cloned()
                .unwrap_or(Operand::Unknown)
            else {
                eprintln!(
                    "[vc] miss {addr:#x} v{v} is not a memory load: {:?}",
                    insns[d].operands
                );
                return;
            };
            (base, addend, d, format!("v{v}"))
        }
        Operand::Memory {
            base,
            addend,
            index: None,
            ..
        } => (base.clone(), *addend, i, "[mem] call".to_string()),
        other => {
            eprintln!("[vc] miss {addr:#x} target is not a value: {other:?}");
            return;
        }
    };
    let (base, addend, d) = (base, addend, at);
    let _ = (vtable_base, entry);
    // Same candidate arithmetic the resolver uses (both vtable bases), so the
    // report can never call an offset "not a methodPtr" while the real pass
    // was trying a valid slot derived from the alternate base.
    let (cands, ncand) = vtable_slot_candidates(addend, ptr_size);
    if ncand == 0 {
        eprintln!("[vc] miss {addr:#x} {what}: offset {addend} is not a vtable methodPtr");
        return;
    }
    let slot = cands[0];
    let tdi = klass_tdi_of(insns, d, &base, table, maps, 4);
    let Some(tdi) = tdi else {
        // Say WHY the class is unknown, not just that it is: the klass-pointer
        // load and the local's seeded type are the two things `klass_tdi_of`
        // consumes, and distinguishing "the load shape is not `[obj]`" from
        // "the receiver has no seeded type" is the difference between a
        // recoverable gap and a genuinely dynamic receiver.
        let seeded = match base.as_ref() {
            Operand::Local(v) => table
                .index_of
                .get(v)
                .and_then(|i| table.infos.get(*i))
                .map(|i| format!("{:?}", i.ty))
                .unwrap_or_else(|| "(not in table)".into()),
            _ => "-".into(),
        };
        let def = match base.as_ref() {
            Operand::Local(v) => ssa_def_of(insns, d, *v)
                .map(|k| format!("{:?}", insns[k].operands))
                .unwrap_or_else(|| "(no def)".into()),
            _ => "-".into(),
        };
        // The receiver itself is one hop past the klass-pointer load: `kl ←
        // [recv]`. Report recv's seeded type and whether it is `this`, because
        // "receiver class unknown" has two very different causes — a receiver
        // whose managed type was never recovered (a typing gap) versus one the
        // IR simply does not trace to `[recv]` (a load-shape gap).
        let recv = match base.as_ref() {
            Operand::Local(v) => ssa_def_of(insns, d, *v).and_then(|k| {
                let op = insns[k].operands.get(1)?;
                if let Operand::Memory {
                    base, addend: 0, ..
                } = op
                {
                    if let Operand::Local(r) = base.as_ref() {
                        return Some(*r);
                    }
                }
                None
            }),
            _ => None,
        };
        let recv_info = recv.map(|r| {
            // Walk the pointer-load chain RECV -> base -> base -> … and print
            // each hop's recovered type plus the offset it was loaded at, so
            // the exact hop where typing stops being a `Class` is visible.
            // `recv ty=IntPtr` alone cannot say whether the break is an
            // untyped parameter, a call return, or a field load whose base was
            // never typed.
            let chain_desc = {
                let mut s = String::new();
                let mut cur = r;
                for _ in 0..6 {
                    let li = table.index_of.get(&cur).and_then(|i| table.infos.get(*i));
                    let (ty, is_this, origin) = match li {
                        Some(i) => (format!("{:?}", i.ty), i.is_this, i.slot_origin.clone()),
                        None => ("(not in table)".to_string(), false, "?".to_string()),
                    };
                    let k = ssa_def_of(insns, d, cur);
                    let (shape, next) = match k.map(|k| &insns[k].operands) {
                        Some(ops) => match ops.get(1) {
                            Some(Operand::Memory {
                                base: b, addend, ..
                            }) => match b.as_ref() {
                                Operand::Local(w) => (format!("[v{w} + {addend}]"), Some(*w)),
                                other => (format!("[{other:?} + {addend}]"), None),
                            },
                            // A pure register copy (`v20 = v19`, the `mov r4,
                            // r0` pin) is part of the chain, not its end:
                            // follow it so an untyped receiver is traced to
                            // whatever actually produced the pointer.
                            Some(Operand::Local(w)) => (format!("copy v{w}"), Some(*w)),
                            Some(o) => (format!("{o:?}"), None),
                            None => ("(no operands)".into(), None),
                        },
                        None => ("(no def)".into(), None),
                    };
                    s.push_str(&format!(
                        " v{cur}({ty},this={is_this},slot={origin})={shape}"
                    ));
                    match next {
                        Some(w) if w != cur => cur = w,
                        _ => break,
                    }
                }
                s
            };
            format!("chain{chain_desc}")
        });
        eprintln!(
            "[vc] miss {addr:#x} {what} slot {slot}: receiver class unknown ({base:?}, \
             seeded {seeded}, def {def}, {})",
            recv_info.as_deref().unwrap_or("recv: no [recv] load")
        );
        return;
    };
    let chain = parents.get(&tdi).cloned().unwrap_or_else(|| vec![tdi]);
    eprintln!(
        "[vc] miss {addr:#x} slot {slot} tdi {tdi}: not declared in any of {chain:?} (declared slots: {:?})",
        {
            let mut s: Vec<usize> = maps
                .vtable_slots
                .keys()
                .filter(|(t, _)| chain.contains(t))
                .map(|(_, s)| *s)
                .collect();
            s.sort_unstable();
            s.dedup();
            s.truncate(24);
            s
        }
    );
}

/// Does any instruction other than `skip` READ version `v`?
///
/// A rewritten call replaces the indirect expression with a direct call; when
/// the resolved method returns VOID the renderer emits a bare statement and
/// the destination assignment disappears. If something reads that
/// destination — the IR modelled the x0 value although the managed method has
/// none — naming it void leaves an undefined reader (`ptr8 = v349`). Such a
/// site is refused and keeps its raw call, which still carries the value.
fn version_read_elsewhere(insns: &[Insn], skip: usize, v: u32) -> bool {
    let mut work = vec![v];
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    while let Some(cur) = work.pop() {
        if !seen.insert(cur) {
            continue;
        }
        for (k, insn) in insns.iter().enumerate() {
            if k == skip {
                continue;
            }
            for (j, op) in insn.operands.iter().enumerate() {
                if !matches!(op, Operand::Local(w) if *w == cur) {
                    continue;
                }
                let is_write = match insn.op {
                    Op::Move | Op::Phi | Op::Arith(_) => j == 0,
                    Op::Call | Op::CallVoid => j == 1,
                    _ => false,
                };
                if is_write {
                    continue;
                }
                // A copy or phi FORWARDS the value; follow its destination.
                // A RETURN ends the chain: the renderer prints no value for a
                // void return, so a value that only reaches returns is not a
                // rendered use. Any other consumer is a real read and the
                // site refuses.
                match insn.op {
                    Op::Move | Op::Phi => match insn.operands.first() {
                        Some(Operand::Local(d)) => work.push(*d),
                        _ => return true,
                    },
                    Op::Return => {}
                    _ => return true,
                }
            }
        }
    }
    false
}

/// `(klass local, entry addend)` of the vtable slot load behind a call target.
fn slot_load_of(insns: &[Insn], upto: usize, target: &Operand) -> Option<(u32, i64)> {
    let Operand::Local(v) = target else {
        return None;
    };
    let d = ssa_def_of(insns, upto, *v)?;
    match insns[d].operands.get(1)? {
        Operand::Memory {
            base,
            addend,
            index: None,
            ..
        } => match base.as_ref() {
            Operand::Local(b) => Some((*b, *addend)),
            _ => None,
        },
        _ => None,
    }
}

/// Is `op` the `MethodInfo` half of the SAME `VirtualInvokeData` entry?
fn is_companion_slot_load(
    insns: &[Insn],
    upto: usize,
    op: &Operand,
    vid_base: u32,
    vt_off: i64,
) -> bool {
    let Operand::Local(v) = op else {
        return false;
    };
    let Some(d) = ssa_def_of(insns, upto, *v) else {
        return false;
    };
    matches!(
        insns[d].operands.get(1),
        Some(Operand::Memory {
            base,
            addend,
            index: None,
            ..
        }) if *addend == vt_off + 8
            && matches!(base.as_ref(), Operand::Local(b) if *b == vid_base)
    )
}

/// Absolute address an operand denotes, following register moves and
/// `base + constant` address arithmetic. Small immediates are values, not
/// addresses, and are refused.
#[allow(dead_code)]
fn addr_of_operand(insns: &[Insn], upto: usize, op: &Operand, depth: u8) -> Option<u64> {
    if depth == 0 {
        return None;
    }
    match op {
        Operand::Immediate(a) if (*a as u64) >= 0x10000 => Some(*a as u64),
        Operand::Local(v) => {
            let d = ssa_def_of(insns, upto, *v)?;
            match insns[d].op {
                Op::Move => match insns[d].operands.get(1)? {
                    Operand::Immediate(a) if (*a as u64) >= 0x10000 => Some(*a as u64),
                    Operand::Memory {
                        base,
                        addend,
                        index: None,
                        ..
                    } => Some(addr_of_operand(insns, d, base, depth - 1)? + *addend as u64),
                    src @ Operand::Local(w) if *w != *v => {
                        addr_of_operand(insns, d, src, depth - 1)
                    }
                    _ => None,
                },
                _ => None,
            }
        }
        _ => None,
    }
}

/// Constant value an operand carries, following register moves.
fn operand_const(insns: &[Insn], upto: usize, op: &Operand, depth: u8) -> Option<i64> {
    if depth == 0 {
        return None;
    }
    match op {
        Operand::Immediate(k) => Some(*k),
        Operand::Local(v) => {
            let d = ssa_def_of(insns, upto, *v)?;
            if matches!(insns[d].op, Op::Phi) {
                if insns[d].operands.len() > 9 {
                    return None;
                }
                let mut found: Option<i64> = None;
                for arm in insns[d].operands.iter().skip(1) {
                    let k = operand_const(insns, d, arm, depth - 1)?;
                    match found {
                        None => found = Some(k),
                        Some(prev) if prev == k => {}
                        Some(_) => return None,
                    }
                }
                return found;
            }
            if !matches!(insns[d].op, Op::Move) {
                return None;
            }
            match insns[d].operands.get(1)? {
                src @ Operand::Local(w) if *w != *v => operand_const(insns, d, src, depth - 1),
                other => operand_const(insns, d, other, depth - 1),
            }
        }
        _ => None,
    }
}

/// TypeDef index named by a metadata cell at `addr`, following the GOT.
fn klass_tdi_at_addr(maps: &ResolveMaps, addr: u64) -> Option<i32> {
    if let Some(&tdi) = maps.klass_to_tdi.get(&addr) {
        return Some(tdi);
    }
    if let Some(mt) = maps.usage_type_by_addr.get(&addr) {
        if let ManagedType::Class(nt) = mt {
            if nt.type_def_index >= 0 {
                return Some(nt.type_def_index);
            }
        }
    }
    if let Some(&target) = maps.got_addend_by_addr.get(&addr) {
        return klass_tdi_at_addr(maps, target);
    }
    None
}

/// TypeDef index of a KLASS-token operand: an immediate klass address or the
/// value loaded from a metadata cell (an interface's `typeof` argument).
/// TypeDef named by a metadata cell reached by folding `op + off` as an
/// address. A PHI of cells is one type when every arm names the same one.
fn cell_tdi_addr_expr(
    insns: &[Insn],
    upto: usize,
    op: &Operand,
    off: i64,
    maps: &ResolveMaps,
    depth: u8,
) -> Option<i32> {
    if depth == 0 {
        return None;
    }
    match op {
        Operand::Immediate(a) if (*a as u64) >= 0x10000 => {
            klass_tdi_at_addr(maps, (*a as u64).wrapping_add(off as u64))
        }
        Operand::Local(v) => {
            let d = ssa_def_of(insns, upto, *v)?;
            match insns[d].op {
                Op::Phi => {
                    // A big phi is a data merge, not a materialised type
                    // token; walking it multiplies work for no proof.
                    if insns[d].operands.len() > 9 {
                        return None;
                    }
                    let mut found: Option<i32> = None;
                    for arm in insns[d].operands.iter().skip(1) {
                        let t = cell_tdi_addr_expr(insns, d, arm, off, maps, depth - 1)?;
                        match found {
                            None => found = Some(t),
                            Some(prev) if prev == t => {}
                            Some(_) => return None,
                        }
                    }
                    found
                }
                Op::Move => match insns[d].operands.get(1)? {
                    // A load of a POINTER to the cell: keep folding.
                    Operand::Memory {
                        base,
                        addend,
                        index: None,
                        ..
                    } => cell_tdi_addr_expr(
                        insns,
                        d,
                        base,
                        addend.wrapping_add(off),
                        maps,
                        depth - 1,
                    ),
                    src => cell_tdi_addr_expr(insns, d, src, off, maps, depth - 1),
                },
                _ => None,
            }
        }
        _ => None,
    }
}

fn token_tdi_of(insns: &[Insn], upto: usize, op: &Operand, maps: &ResolveMaps) -> Option<i32> {
    let mut memo: HashMap<u32, Option<i32>> = HashMap::new();
    let mut busy: std::collections::HashSet<u32> = std::collections::HashSet::new();
    token_tdi_of_m(insns, upto, op, maps, 12, &mut memo, &mut busy)
}

/// Depth-bounded core. A loop-carried phi can reference versions defined
/// LATER in the listing, so the phi walk is a graph, not a DAG; without the
/// bound a cycle re-explores both arms at every level (exponential) and the
/// whole dump dies.
fn token_tdi_of_m(
    insns: &[Insn],
    upto: usize,
    op: &Operand,
    maps: &ResolveMaps,
    depth: u8,
    memo: &mut HashMap<u32, Option<i32>>,
    busy: &mut std::collections::HashSet<u32>,
) -> Option<i32> {
    if depth == 0 {
        return None;
    }
    match op {
        Operand::Immediate(a) => klass_tdi_at_addr(maps, *a as u64),
        Operand::Local(v) => {
            if let Some(hit) = memo.get(v) {
                return *hit;
            }
            // A loop-carried phi references versions defined LATER; a cycle
            // is a refusal for this query, not a reason to spin.
            if !busy.insert(*v) {
                return None;
            }
            let d = ssa_def_of(insns, upto, *v);
            let result = match d {
                Some(d) => match insns[d].op {
                    // A type token merged across branches is still ONE type:
                    // every arm must name it (the cells may differ).
                    Op::Phi => {
                        if insns[d].operands.len() > 9 {
                            busy.remove(v);
                            memo.insert(*v, None);
                            return None;
                        }
                        let mut found: Option<i32> = None;
                        let mut ok = true;
                        for arm in insns[d].operands.iter().skip(1) {
                            match token_tdi_of_m(insns, d, arm, maps, depth - 1, memo, busy) {
                                Some(t) => match found {
                                    None => found = Some(t),
                                    Some(prev) if prev == t => {}
                                    Some(_) => {
                                        ok = false;
                                        break;
                                    }
                                },
                                None => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok {
                            found
                        } else {
                            None
                        }
                    }
                    Op::Move => match insns[d].operands.get(1) {
                        Some(Operand::Memory {
                            base,
                            addend,
                            index: None,
                            ..
                        }) => {
                            cell_tdi_addr_expr(insns, d, base, *addend, maps, depth.min(6))
                        }
                        Some(src @ Operand::Local(w)) if *w != *v => {
                            token_tdi_of_m(insns, d, src, maps, depth - 1, memo, busy)
                        }
                        Some(Operand::Immediate(a)) => klass_tdi_at_addr(maps, *a as u64),
                        _ => None,
                    },
                    _ => None,
                },
                None => None,
            };
            busy.remove(v);
            memo.insert(*v, result);
            result
        }
        _ => None,
    }
}

/// Interface invoke-data dispatch:
/// `vid = get_interface_invoke_data(obj, typeof(I), slot); call [vid]`.
///
/// The helper returns a `VirtualInvokeData*` whose halves the call site loads
/// (`[vid]` is the callee, `[vid + 8]` the hidden MethodInfo argument). The
/// interface type comes from the helper's own `typeof(I)` argument and the
/// slot from its constant argument, so the call resolves to the interface's
/// DECLARED method without trusting any helper address. `vid` usually reaches
/// the call through a phi (the inline interface-offset scan merges with the
/// helper's result), so copies and phi arms are followed; arms that name
/// different helpers make the site refuse.
fn rewrite_interface_invoke_calls(
    insns: &mut [Insn],
    _table: &LocalTable,
    maps: &ResolveMaps,
    parents: &HashMap<i32, Vec<i32>>,
    stats: &mut ResolveStats,
) {
    use std::collections::HashMap;
    // dest version → (receiver version, interface tdi, slot)
    let mut helpers: HashMap<u32, (u32, i32, usize)> = HashMap::new();
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Call) {
            continue;
        }
        if !matches!(insns[i].operands.first(), Some(Operand::Immediate(_))) {
            continue;
        }
        let ops = &insns[i].operands;
        let Some(Operand::Local(dest)) = ops.get(1) else {
            continue;
        };
        let Some(Operand::Local(obj)) = ops.get(2) else {
            continue;
        };
        let Some(iface) = ops.get(3).and_then(|o| token_tdi_of(insns, i, o, maps)) else {
            continue;
        };
        if !maps.interface_tdis.contains(&iface) {
            continue;
        }
        let Some(slot) = ops.get(4).and_then(|o| operand_const(insns, i, o, 6)) else {
            continue;
        };
        if !(0..4096).contains(&slot) {
            continue;
        }
        helpers.insert(*dest, (*obj, iface, slot as usize));
    }
    if helpers.is_empty() {
        return;
    }
    // Distinct helper shapes reaching one invoke-data local.
    fn helper_of(
        insns: &[Insn],
        upto: usize,
        v: u32,
        helpers: &HashMap<u32, (u32, i32, usize)>,
        depth: u8,
        out: &mut Vec<(u32, i32, usize)>,
    ) {
        if depth == 0 {
            return;
        }
        if let Some(h) = helpers.get(&v) {
            out.push(*h);
            return;
        }
        let Some(d) = ssa_def_of(insns, upto, v) else {
            return;
        };
        match insns[d].op {
            Op::Phi => {
                for src in insns[d].operands.iter().skip(1) {
                    if let Operand::Local(w) = src {
                        helper_of(insns, d, *w, helpers, depth - 1, out);
                    }
                }
            }
            Op::Move => {
                if let Some(Operand::Local(w)) = insns[d].operands.get(1) {
                    if *w != v {
                        helper_of(insns, d, *w, helpers, depth - 1, out);
                    }
                }
            }
            _ => {}
        }
    }
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Call) {
            continue;
        }
        if matches!(insns[i].operands.first(), Some(Operand::Method(_))) {
            continue;
        }
        let Some((vid, off)) = insns[i]
            .operands
            .first()
            .and_then(|t| slot_load_of(insns, i, t))
        else {
            continue;
        };
        if off != 0 {
            continue;
        }
        let Some(Operand::Local(recv)) = insns[i].operands.get(2) else {
            continue;
        };
        let mut found = Vec::new();
        helper_of(insns, i, vid, &helpers, 8, &mut found);
        found.sort_unstable();
        found.dedup();
        let [(obj, iface, slot)] = found.as_slice() else {
            continue;
        };
        if obj != recv {
            continue;
        }
        let Some(m) = lookup_vtable_slot(maps, parents, *iface, *slot)
            .or_else(|| maps.vtable_slots.get(&(*iface, *slot)).cloned())
        else {
            continue;
        };
        if m.sig.is_void_like() {
            if let Some(Operand::Local(dv)) = insns[i].operands.get(1) {
                if version_read_elsewhere(insns, i, *dv) {
                    continue;
                }
            }
        }
        // Rebuild: receiver + the outer user arguments, dropping the hidden
        // MethodInfo half of the invoke-data pair.
        let mut ops: Vec<Operand> = Vec::with_capacity(insns[i].operands.len());
        ops.push(Operand::Method(m.clone()));
        ops.push(insns[i].operands.get(1).cloned().unwrap_or(Operand::Unknown));
        ops.push(Operand::Local(*recv));
        for op in insns[i].operands.iter().skip(3) {
            if is_companion_slot_load(insns, i, op, vid, 0) {
                continue;
            }
            ops.push(op.clone());
        }
        insns[i].operands = ops;
        insns[i].dest_type = Some(m.sig.ret.clone());
        stats.virtuals_resolved += 1;
    }
}

/// Virtual dispatch: rewrite a call through a computed callee into a direct
/// call once the callee's vtable slot is identified.
/// (Structural subset of Cpp2IL `ResolveVirtualCalls`; interface dispatch
/// recovery intentionally deferred.)
///
/// This used to key on `Op::IndirectCall` alone, and to rebuild the operand
/// list as `[Method, Unknown, Unknown…]`. Both were wrong by the time the
/// ARM64 lifter was complete:
///
///   1. `BLR`/`BLRAA*` lift to `Op::Call` with the ordinary
///      `[target, ret, args…]` layout (they RETURN, so modelling them as the
///      `IndirectCall` CFG terminator made everything after the call
///      unreachable). So the resolver's input never arrived, and *no* virtual
///      call was ever resolved — which is where the resolved callees went.
///   2. The rebuilt list threw the arguments away. Even when it did fire, the
///      receivers and parameters it had just been handed were replaced with
///      `Unknown`, so a resolved call rendered with no arguments at all.
///
/// Now the target is resolved from the VALUE operand, and the existing
/// operand list is preserved: the rewritten call keeps its real receiver and
/// arguments.
fn rewrite_virtual_calls(
    insns: &mut [Insn],
    table: &LocalTable,
    maps: &ResolveMaps,
    parents: &HashMap<i32, Vec<i32>>,
    ptr_size: u64,
    stats: &mut ResolveStats,
) {
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Call | Op::IndirectCall) {
            continue;
        }
        // A direct call (resolved `BL`/metadata thunk) is already as good as
        // it gets.
        if matches!(insns[i].operands.first(), Some(Operand::Method(_))) {
            continue;
        }
        let Some(target) = insns[i].operands.first().cloned() else {
            continue;
        };
        let Some(m) = resolve_indirect_target(insns, i, &target, table, maps, parents, ptr_size, 4)
        else {
            // Count only COMPUTED callees. A `BL` to a constant address that
            // `methods_by_rva` does not know is an unnamed direct call, not an
            // unresolved dispatch, and mixing the two made the counter read
            // 217k against 64 resolutions — i.e. it hid the real denominator.
            if !matches!(target, Operand::Immediate(_)) {
                super::timing::VC_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // `ILR_VC=1` reports every indirect call the resolver could not
            // name, with the *reason* (unknown class, unknown slot, slot not
            // in the ancestry) — so "unresolved calls" stops being a single
            // opaque number and becomes an addressable work list.
            if std::env::var("ILR_VC").is_ok() {
                report_vc_miss(insns, i, &target, table, maps, parents, ptr_size);
            }
            continue;
        };
        super::timing::VIRTUAL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // A resolved call is a DIRECT call that RETURNS. `IndirectCall` is
        // the CFG terminator used for non-returning dispatch, so leaving it
        // there would end the block at a call the callee comes back from.
        insns[i].op = Op::Call;
        // A synthesized delegate `Invoke` carries its receiver separately
        // from the vtable path: the call's ABI arguments are the delegate's
        // FIELDS (`method_code`, `method`, `invoke_impl`), not the delegate
        // object itself, so preserving the list would keep `method_code` as
        // the receiver — a wrong value. The proven delegate base (see
        // `delegate_base_of_target_nocount`, same proof the resolver counted,
        // re-derived here WITHOUT counting so no site is double-counted)
        // becomes the receiver; the delegate's hidden runtime fields
        // (`method_code`/`method`/`invoke_impl` loads off the same base) are
        // DROPPED, while every USER argument is kept. Keeping the hidden
        // loads would render `Invoke(method_code, method, …)` — the exact
        // `fnptr_` shape this fix removes; dropping a user argument would lose
        // semantics. See `is_hidden_delegate_arg`. Never moves a write.
        if m.name == "Invoke" && m.type_def_index == -1 && m.image_index == usize::MAX {
            if let Some(dl) = delegate_base_of_target_nocount(insns, i, &target, table, 4) {
                if insns[i].operands.len() >= 2 {
                    let dest = insns[i]
                        .operands
                        .get(1)
                        .cloned()
                        .unwrap_or(Operand::MissingAbiArg);
                    let rest: Vec<Operand> = insns[i]
                        .operands
                        .iter()
                        .skip(2)
                        .cloned()
                        .filter(|op| !is_hidden_delegate_arg(insns, i, op, dl))
                        .collect();
                    // Size the synthesized signature to the call site's own
                    // USER arity so `trim_call_arguments` (which runs after
                    // this pass and truncates `Call` to `1 + dest + this +
                    // params`) keeps exactly those operands. `params == []`
                    // would truncate a 2-user-arg `Invoke(v66, v69)` to
                    // `Invoke()` — the trim, not the proof, would delete live
                    // arguments. Trailing `MissingAbiArg` padding is NOT
                    // counted (the trim drops it, as for every other call);
                    // interior holes are kept. The count is the call's own
                    // filtered operand count, never a guessed slot; the types
                    // stay `Unknown`.
                    let mut effective = rest.len();
                    while effective > 0 && matches!(rest[effective - 1], Operand::MissingAbiArg) {
                        effective -= 1;
                    }
                    let mut m = m.clone();
                    m.sig.params = vec![ManagedType::Unknown; effective];
                    let mut ops = vec![Operand::Method(m.clone()), dest, Operand::Local(dl)];
                    ops.extend(rest);
                    insns[i].operands = ops;
                } else {
                    let mut ops = vec![
                        Operand::Method(m.clone()),
                        Operand::MissingAbiArg,
                        Operand::Local(dl),
                    ];
                    ops.extend(std::iter::repeat(Operand::MissingAbiArg).take(m.sig.params.len()));
                    insns[i].operands = ops;
                }
                insns[i].dest_type = Some(m.sig.ret.clone());
                stats.virtuals_resolved += 1;
                continue;
            }
            // Proof vanished between resolve and rewrite (same IR, so this
            // cannot happen): fall through to the preserving path rather than
            // fabricating a receiver.
        }
        // DROP THE COMPANION `MethodInfo`. A vtable entry is
        // `VirtualInvokeData { methodPtr; method; }`; the lift loads BOTH
        // halves (`ldp x9, x1, [klass + N]`) and passes `method` as x1, so the
        // operand list is `[callee, dest, this, methodInfo, args…]`. Keeping it
        // makes the signature trim consume the MethodInfo AS the first
        // argument. The companion is proved, never guessed: a load from the
        // SAME base local as the callee slot at an addend exactly 8 past it.
        let target0 = insns[i].operands.first().cloned().unwrap_or(Operand::Unknown);
        if let Some((vid_base, vt_off)) = slot_load_of(insns, i, &target0) {
            let companion = insns[i]
                .operands
                .iter()
                .enumerate()
                .skip(2)
                .find(|(_, op)| is_companion_slot_load(insns, i, op, vid_base, vt_off))
                .map(|(j, _)| j);
            if let Some(j) = companion {
                insns[i].operands.remove(j);
            }
        }
        // PRESERVE the operand list. For the `BLR`-shaped `Op::Call` it is
        // already `[target, ret-slot, receiver, args…]`, exactly what
        // `call_parts` and `dest_position` expect, so only operand 0 changes —
        // the receiver and the arguments that the type/SSA passes just
        // recovered stay attached to the call.
        if insns[i].operands.len() >= 2 {
            insns[i].operands[0] = Operand::Method(m.clone());
        } else {
            // Legacy arm32/x86 `IndirectCall` shape (a bare target and no
            // recovered argument registers): fill positionally with
            // `MissingAbiArg`, the operand that exists precisely for "this ABI
            // slot has no recovered value". Filling with `Unknown` instead put
            // a raw unresolved marker in every argument position, which the
            // renderer prints as `__unknown` — indistinguishable from a value
            // the lifter actually lost.
            let mut ops = vec![Operand::Method(m.clone()), Operand::MissingAbiArg];
            if !m.sig.is_static {
                ops.push(Operand::MissingAbiArg);
            }
            ops.extend(std::iter::repeat(Operand::MissingAbiArg).take(m.sig.params.len()));
            insns[i].operands = ops;
        }
        insns[i].dest_type = Some(m.sig.ret.clone());
        stats.virtuals_resolved += 1;
    }
}

// convenience ----------------------------------------------------------------

impl ResolveMaps {
    pub fn sig_of(m: &MethodRef) -> &MethodSigInfo {
        &m.sig
    }
}

/// DelegateInvokeRecovery (Cpp2IL port, structural subset):
/// `BLR [delegate + 3*ptr]` loads `invoke_impl` — rewrite as a direct call
/// to `Invoke` with the delegate local as receiver. Works without knowing
/// the concrete delegate type: the synthesized MethodRef resolves through
/// cross-image memberref emission.
///
/// Two spellings reach here, and both are handled — but in different places,
/// because they exist at different pipeline stages:
///
/// * the pre-typing `Memory` spelling (`[delegate + 3*ptr]`, offset checked
///   below) is what this pass has always handled: it runs AFTER
///   `resolve::run_with_fields`, i.e. after the typing sweep may already have
///   rewritten the load, so a `Memory` that survives to here is one the sweep
///   could not name (untyped base). The window scan below is the only evidence
///   for pre-SSA `Register` targets, which have no definition chain.
/// * the post-typing `InstanceField`/`Field` spelling (`base.invoke_impl`,
///   named by `locals::propagate_types`) never reaches the window below — and
///   its call is a `Call`, not an `IndirectCall`, so this pass's opcode gate
///   never saw it either. Those sites are proven upstream in
///   `resolve_indirect_target` (see `delegate_invoke_of_def`), which runs
///   inside `run_with_fields` BEFORE this pass and already rewrote every
///   provable one to `Method(Invoke)`. The chain pass at the top here is the
///   backstop for any such `Call` that survived (unproven base, or a depth
///   the vtable walk refused): it uses the SAME strict proof but WITHOUT
///   counting, so no site is double-counted — the resolver already counted
///   the attempt.
///
/// Inert on well-formed IR: every new arm either proves a delegate base or
/// refuses and keeps `fnptr_`; no write is moved (the receiver becomes the
/// proven delegate local, hidden delegate fields are dropped, every USER
/// argument is kept — see `is_hidden_delegate_arg`).
pub fn recover_delegate_invokes(
    insns: &mut [Insn],
    table: &LocalTable,
    ptr_size: u64,
    stats: &mut ResolveStats,
) {
    // Backstop for typed field shapes on `Call` (see above). Uses the
    // no-count proof so the resolver's attempt/hit pair stays exact.
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Call) {
            continue;
        }
        if matches!(insns[i].operands.first(), Some(Operand::Method(_))) {
            continue;
        }
        let Some(target) = insns[i].operands.first().cloned() else {
            continue;
        };
        let Some(dl) = delegate_base_of_target_nocount(insns, i, &target, table, 4) else {
            continue;
        };
        let nt =
            delegate_named_ty(table, dl).or_else(|| this_delegate_base(table).map(|(_, nt)| nt));
        let Some(nt) = nt else { continue };
        let mut m = delegate_invoke_ref(nt.namespace.clone(), nt.name.clone());
        if insns[i].operands.len() >= 2 {
            let dest = insns[i]
                .operands
                .get(1)
                .cloned()
                .unwrap_or(Operand::MissingAbiArg);
            let rest: Vec<Operand> = insns[i]
                .operands
                .iter()
                .skip(2)
                .cloned()
                .filter(|op| !is_hidden_delegate_arg(insns, i, op, dl))
                .collect();
            // Same user-arity sizing as the vtable path above: without it the
            // later trim to `params == []` would delete live user arguments.
            // Trailing padding is not counted (the trim drops it).
            let mut effective = rest.len();
            while effective > 0 && matches!(rest[effective - 1], Operand::MissingAbiArg) {
                effective -= 1;
            }
            m.sig.params = vec![ManagedType::Unknown; effective];
            let mut ops = vec![Operand::Method(m.clone()), dest, Operand::Local(dl)];
            ops.extend(rest);
            insns[i].operands = ops;
        } else {
            let mut ops = vec![
                Operand::Method(m.clone()),
                Operand::MissingAbiArg,
                Operand::Local(dl),
            ];
            ops.extend(std::iter::repeat(Operand::MissingAbiArg).take(m.sig.params.len()));
            insns[i].operands = ops;
        }
        insns[i].dest_type = Some(m.sig.ret.clone());
        stats.virtuals_resolved += 1;
    }
    let invoke_impl_off: i64 = 3 * ptr_size as i64;

    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::IndirectCall) {
            continue;
        }
        // find target load: Move(t, Memory{base: d_local, addend: INVOKE_IMPL_OFF})
        // in the 6 instructions right before the call (the unconditional
        // `break` that used to sit at the end of this loop reduced it to
        // inspecting exactly ONE instruction)
        let mut target_load = None;
        for k in (i.saturating_sub(6)..i).rev() {
            if let (Some(Operand::Local(_)), Some(src)) =
                (insns[k].dest().cloned(), insns[k].operands.get(1))
            {
                if let Operand::Memory { base, addend, .. } = src {
                    if *addend == invoke_impl_off {
                        if let Operand::Local(d) = base.as_ref() {
                            // receiver must be a delegate-typed local
                            let is_delegate = table
                                .index_of
                                .get(d)
                                .map(|&li| {
                                    matches!(
                                        &table.infos[li].ty,
                                        ManagedType::Class(nt)
                                            if nt.name.contains("Delegate")
                                                || nt.name.contains("Action")
                                                || nt.name.contains("Func")
                                                || nt.name.contains("Event")
                                    )
                                })
                                .unwrap_or(false);
                            // accept even untyped when offset matches exactly
                            target_load = Some((k, *d, is_delegate));
                            break;
                        }
                    }
                }
            }
        }
        let Some((_, d_local, _is_delegate)) = target_load else {
            continue;
        };
        let recv_ty = table
            .index_of
            .get(&d_local)
            .and_then(|&li| match &table.infos[li].ty {
                ManagedType::Class(nt) => Some(nt.clone()),
                _ => None,
            });
        let (ns, tname) = match &recv_ty {
            Some(nt) => (nt.namespace.clone(), nt.name.clone()),
            None => (String::new(), "Delegate".into()),
        };
        let m = MethodRef {
            image_index: usize::MAX, // forces external memberref
            type_def_index: -1,
            method_index: 0,
            name: "Invoke".into(),
            declaring_namespace: ns,
            declaring_type_name: tname,
            assembly: String::new(),
            sig: super::types::MethodSigInfo {
                ret: ManagedType::Unknown,
                params: vec![], // unknown arity — emitter keeps captured args
                is_static: false,
            },
        };
        let mut ops = vec![
            Operand::Method(m),
            Operand::Unknown,
            Operand::Local(d_local),
        ];
        ops.extend(insn_operands_after(insns, i));
        insns[i] = Insn::new(Op::CallVoid, ops, insns[i].address);
        stats.virtuals_resolved += 1;
    }
}

fn insn_operands_after(insns: &[Insn], i: usize) -> impl Iterator<Item = Operand> + '_ {
    insns[i].operands.iter().skip(1).cloned()
}


/// Split the generic argument list of a rendered type (`List<Vector3>` →
/// `["Vector3"]`). Nested instantiations are split at top-level commas only.
fn generic_args(text: &str) -> Option<Vec<String>> {
    let start = text.find('<')?;
    let end = text.rfind('>')?;
    if end <= start {
        return None;
    }
    let inner = &text[start + 1..end];
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut from = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(inner[from..i].trim().to_string());
                from = i + 1;
            }
            _ => {}
        }
    }
    out.push(inner[from..].trim().to_string());
    if out.iter().any(|s| s.is_empty()) {
        return None;
    }
    Some(out)
}

fn named_managed(tdi: i32, maps: &ResolveMaps) -> Option<ManagedType> {
    let (ns, name) = maps.tdi_names.get(&tdi)?;
    Some(ManagedType::Class(std::sync::Arc::new(NamedType {
        namespace: ns.clone(),
        name: name.clone(),
        assembly: None,
        type_def_index: tdi,
    })))
}

/// Resolve a rendered generic ARGUMENT (`Vector3`, `System.Int32`, `string`)
/// to a managed type. The short name is accepted only when it identifies a
/// single TypeDef.
fn managed_from_type_text(text: &str, maps: &ResolveMaps) -> Option<ManagedType> {
    let text = text.trim();
    let prim = match text {
        "bool" | "Boolean" | "System.Boolean" => Some(PrimKind::Boolean),
        "char" | "Char" | "System.Char" => Some(PrimKind::Char),
        "sbyte" | "SByte" | "System.SByte" => Some(PrimKind::I1),
        "byte" | "Byte" | "System.Byte" => Some(PrimKind::U1),
        "short" | "Int16" | "System.Int16" => Some(PrimKind::I2),
        "ushort" | "UInt16" | "System.UInt16" => Some(PrimKind::U2),
        "int" | "Int32" | "System.Int32" => Some(PrimKind::I4),
        "uint" | "UInt32" | "System.UInt32" => Some(PrimKind::U4),
        "long" | "Int64" | "System.Int64" => Some(PrimKind::I8),
        "ulong" | "UInt64" | "System.UInt64" => Some(PrimKind::U8),
        "float" | "Single" | "System.Single" => Some(PrimKind::R4),
        "double" | "Double" | "System.Double" => Some(PrimKind::R8),
        _ => None,
    };
    if let Some(p) = prim {
        return Some(ManagedType::Primitive(p));
    }
    if matches!(text, "string" | "String" | "System.String") {
        return Some(ManagedType::String);
    }
    if matches!(text, "object" | "Object" | "System.Object") {
        return Some(ManagedType::Object);
    }
    if let Some(&tdi) = maps.tdi_names_reverse.get(text) {
        return named_managed(tdi, maps);
    }
    let suffix = format!(".{text}");
    let mut hit: Option<i32> = None;
    for (name, &tdi) in &maps.tdi_names_reverse {
        if name.ends_with(&suffix) {
            if hit.is_some_and(|h| h != tdi) {
                return None; // ambiguous short name — refuse rather than guess
            }
            hit = Some(tdi);
        }
    }
    hit.and_then(|tdi| named_managed(tdi, maps))
}

/// Specialize a shared generic call's `T` return from the call site's
/// declaring-type instantiation.
///
/// IL2CPP compiles one shared body for `List<T>.get_Item`, so its MethodRef
/// returns `Opaque("T")` even when the receiver is `List<Vector3>`. That lost
/// the HFA return layout: the call was treated as a scalar, its s-register
/// components were never bound, and later reads reused pre-call SSA versions
/// (measured as `(z - z)` in real distance math). The declaring type text is
/// the call site's own evidence — `List<Vector3>` — so substituting its single
/// argument is a proof, not a guess.
pub fn specialize_generic_return_sigs(insns: &mut [Insn], maps: &ResolveMaps) -> usize {
    let mut n = 0usize;
    for insn in insns.iter_mut() {
        if !matches!(insn.op, Op::Call) {
            continue;
        }
        let Some(Operand::Method(m)) = insn.operands.first() else {
            continue;
        };
        if !matches!(&m.sig.ret, ManagedType::Opaque(t) if t == "T") {
            continue;
        }
        let Some(args) = generic_args(&m.declaring_type_name) else {
            continue;
        };
        // Only a single-argument instantiation: managed_type_of names every
        // generic parameter "T", so a multi-parameter definition cannot say
        // WHICH parameter this is; substituting arg0 there would be a guess.
        if args.len() != 1 {
            continue;
        }
        let Some(mt) = managed_from_type_text(&args[0], maps) else {
            continue;
        };
        if let Some(Operand::Method(m)) = insn.operands.first_mut() {
            m.sig.ret = mt.clone();
        }
        insn.dest_type = Some(mt);
        n += 1;
    }
    n
}

/// ResolveCallsViaMethodInfo (Cpp2IL port, usage-table variant):
/// 1. `Move(local, Imm(slot))` where slot is a MethodInfo metadata-usage →
///    the local becomes a resolved Method operand.
/// 2. `IndirectCall` whose register traces to such a local becomes a direct
///    named Call (emitter fills missing args from the signature).
pub fn resolve_methodinfo_usages(
    insns: &mut [Insn],
    method_index_by_addr: &HashMap<u64, u32>,
    method_ref_of: &mut dyn FnMut(u32) -> Option<super::types::MethodRef>,
    stats: &mut ResolveStats,
) {
    use super::types::MethodRef;

    // pass 1: materialize method operands at the load sites
    let mut method_locals: HashMap<u32, MethodRef> = HashMap::new();
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::Move) {
            continue;
        }
        // See `resolve_static_fields`: on x64 the MethodInfo slot is loaded
        // RIP-relative (`Memory { base: abs, addend }`), so a bare `Immediate`
        // match meant no MethodInfo usage was ever materialized on a PE build.
        let Some(a) = insns[i]
            .operands
            .get(1)
            .and_then(|s| absolute_address(insns, i, s))
        else {
            continue;
        };
        let a = a as i64;
        let Some(d) = insns[i].dest().and_then(|d| d.as_local()) else {
            continue;
        };
        // An `adrp` page is not an address: resolve the SUM at each use, and
        // leave the page alone. See `page_base_offset_uses`.
        let page_uses = is_page_materialization(insns, i)
            .then(|| page_base_offset_uses(insns, i, d))
            .flatten();
        if let Some(uses) = page_uses {
            for (j, k) in uses {
                let Some(&mi) = method_index_by_addr.get(&(a.wrapping_add(k) as u64)) else {
                    continue;
                };
                let Some(m) = method_ref_of(mi) else { continue };
                let Some(w) = insns[j].dest().and_then(|x| x.as_local()) else {
                    continue;
                };
                let addr = insns[j].address;
                // `add` made `w` the slot's ADDRESS; `ldr` made `w` the slot's
                // CONTENTS, which is this MethodInfo. Either way the use now
                // names the member instead of an offset into its neighbour.
                insns[j] = Insn::new(
                    Op::Move,
                    vec![Operand::Local(w), Operand::Method(m.clone())],
                    addr,
                );
                method_locals.insert(w, m);
                stats.calls_resolved += 1;
            }
            continue;
        }
        let Some(&mi) = method_index_by_addr.get(&(a as u64)) else {
            continue;
        };
        let Some(m) = method_ref_of(mi) else { continue };
        method_locals.insert(d, m.clone());
        insns[i].operands[1] = Operand::Method(m);
        stats.calls_resolved += 1;
    }

    // pass 2: a call whose target traces to a materialized MethodInfo
    //
    // Two facts make this exact instead of a look-back guess:
    //
    //   1. `mi = <MethodInfo*>` then `fn = [mi + 0]`: offset 0 of `MethodInfo`
    //      IS `methodPointer`, so the deref NAMES the method. This is the shape
    //      IL2CPP emits for every generic-shared call — the body is shared, the
    //      concrete instantiation arrives as the hidden `MethodInfo` argument,
    //      and the code pointer is read back out of it — which is why it is the
    //      dominant source of unresolved `fn_`/`fnptr_` calls on both corpora
    //      (`fnptr_` 10,582 sites on the x64 PE, 50,788 on the iOS arm64 one).
    //   2. pass 1 writes the MethodInfo cell itself as `Move(local, Method(…))`,
    //      so the chain is walkable from the CALL'S OWN target operand instead
    //      of from a fixed instruction window — and the window is what used to
    //      let an unrelated MethodInfo load nearby name the callee.
    //
    // The window scan survives ONLY for a `Register` target: that is the
    // pre-SSA shape the legacy arm32/x86 lifters emit, where no definition chain
    // exists and the register name is the only evidence there is.
    //
    // ORDERING: `mod.rs` calls this AFTER `resolve::run_with_fields`, i.e. after
    // `reslice_call_arguments` has already turned each NAMED call's operand list
    // into the signature-shaped form. `signature_shaped_operands` is the
    // IDENTITY on such a list (it re-derives the same receiver/parameter slots
    // from the same entries), so the rebuild below is a no-op for a call that
    // pass already shaped, and a real fix for the `Op::IndirectCall` shape,
    // which `reslice_call_arguments` never touched because it has no callee to
    // size the list from.
    for i in 0..insns.len() {
        if !matches!(insns[i].op, Op::IndirectCall | Op::Call) {
            continue;
        }
        let Some(target) = insns[i].operands.first().cloned() else {
            continue;
        };
        // A direct call is already as precise as metadata allows.
        if matches!(target, Operand::Method(_)) {
            continue;
        }
        let found: Option<MethodRef> = match &target {
            Operand::Immediate(_) => None,
            Operand::Register(_) => loose_method_local(insns, i, &method_locals),
            _ => method_for_operand(insns, i, &target, METHOD_CHAIN_DEPTH),
        };
        let Some(m) = found else { continue };

        // PRESERVE the argument operands. Rebuilding the list with `Unknown`
        // made every argument of every call this pass named render as
        // `__unknown` — a value the IR still had, deleted by the naming pass
        // (9,260 `__unknown` on the x64 corpus, plus every condition that
        // compared one). The list is trimmed to the callee's arity instead, and
        // a slot with no operand is a HOLE (`MissingAbiArg`, printed `__abi`),
        // which keeps the distinction between "the ABI had no value here" and
        // "the lifter lost this value" visible in the output.
        let is_indirect = matches!(insns[i].op, Op::IndirectCall);
        let mut ops = signature_shaped_operands(&m, &insns[i].operands, is_indirect);
        // AAPCS64 indirect-result slot (x8), kept across the rebuild for the
        // same reason as in `reslice_call_arguments`: this list is rebuilt
        // positionally by SIGNATURE, but the slot index is an ABI position the
        // renderer reads back, so it must be restored at that index rather than
        // dropped with the register dump.
        let sret = insns[i]
            .operands
            .get(super::passes::SRET_OPERAND)
            .filter(|o| {
                !matches!(
                    o,
                    Operand::Unknown | Operand::MissingAbiArg | Operand::Register(_)
                )
            })
            .cloned();
        super::passes::place_sret(&mut ops, sret);
        let addr = insns[i].address;
        insns[i] = Insn::new(Op::Call, ops, addr);
        insns[i].dest_type = Some(m.sig.ret.clone());
        stats.virtuals_resolved += 1;
    }
}

/// How many hops a call target is followed through copies, spills and
/// `MethodInfo` loads.
///
/// The bound exists ONLY to terminate the walk on a cycle (a version whose chain
/// leads back to itself). Every hop is individually PROVED — a `Move` of the same
/// value, a `[mi + 0]` `methodPointer` load, a phi whose arms agree, or a frame
/// slot's reaching store — so a higher bound can only recover chains that were
/// previously refused; it cannot invent a callee. 6 was too low for real code:
/// the compiler re-materialises a shared-generic `MethodInfo*` through a copy, a
/// spill, a reload and a second copy across a block boundary, which is already 4
/// hops before the `[mi+0]` load and 2 more after it, so the old bound silently
/// refused the tail of exactly the shape this pass exists for.
const METHOD_CHAIN_DEPTH: u8 = 16;

/// The method a call target provably invokes, or `None`.
///
/// The decisive shape is IL2CPP's generic-shared call: the concrete
/// instantiation is passed as a `MethodInfo*` and the body reads the code
/// pointer back out of it —
///
/// ```text
/// mi = <MethodInfo*>   ; materialized by pass 1 as `Operand::Method`
/// fn = [mi + 0]        ; `MethodInfo::methodPointer`, offset 0
/// call fn
/// ```
///
/// — so the deref at offset 0 IS the method and the chain from the call's own
/// target operand reaches it. Everything else REFUSES: a `[mi + n]` for n != 0
/// (`methodMetadataHandle`, `invoker` and `genericMethod` are other fields and
/// none of them is a callable code pointer), a `Phi` whose arms do NOT agree
/// (see [`phi_method`]), a frame slot whose reaching store cannot be proved (see
/// [`spill_value_at`]), or a chain longer than [`METHOD_CHAIN_DEPTH`]. Naming a
/// call after an unrelated `MethodInfo` load nearby would fabricate a callee,
/// which is strictly worse than the honest `fnptr_<addr>`.
fn method_for_operand(insns: &[Insn], at: usize, target: &Operand, depth: u8) -> Option<MethodRef> {
    if depth == 0 {
        return None;
    }
    match target {
        Operand::Method(m) => Some(m.clone()),
        Operand::Local(v) => {
            let d = ssa_def_of(insns, at, *v)?;
            // A merge version is provable when its arms AGREE. Refusing every
            // phi (what this walk did) cost every call whose `MethodInfo*`
            // crossed a join its name — and that is a REGRESSION against the
            // window scan it replaced: `fnptr_` went from 10,582 to 10,832 on
            // the x64 corpus when this arm started refusing them.
            if matches!(insns[d].op, Op::Phi) {
                return phi_method(insns, d, depth - 1);
            }
            match insns[d].operands.get(1)? {
                // The definition is itself a LOAD: either the `methodPointer`
                // field of a materialized `MethodInfo`, or a code pointer that
                // was SPILLED to a frame slot and reloaded.
                src @ Operand::Memory { .. } => loaded_method(insns, d, src, depth - 1),
                // `fn = mi` — the MethodInfo itself (a shared body called
                // indirectly through its own cell).
                src => method_for_operand(insns, d, src, depth - 1),
            }
        }
        // The target was loaded from memory: a spilled code pointer.
        Operand::Memory { .. } => loaded_method(insns, at, target, depth),
        _ => None,
    }
}

/// The method a `Move(dst, <load>)` provably loads.
///
/// Two load shapes name a callable method, and they are told apart by the BASE:
///
///   * a FRAME slot (`[sp ± k]`) is a SPILL of the code pointer — the value is
///     whatever the nearest store to that slot wrote, so it is resolved through
///     [`spill_value_at`];
///   * `[mi + 0]` on anything else is `MethodInfo::methodPointer`, the field
///     IL2CPP reads to call a shared-generic body. A MethodInfo field at any
///     OTHER offset (`methodMetadataHandle`, `invoker`, `genericMethod`) is not
///     a code pointer, so it refuses.
fn loaded_method(insns: &[Insn], at: usize, load: &Operand, depth: u8) -> Option<MethodRef> {
    if depth == 0 {
        return None;
    }
    let Operand::Memory {
        base,
        addend,
        index,
        ..
    } = load
    else {
        return None;
    };
    // Frame slot FIRST: `[sp ± k]` is a stack slot whatever its addend is, and
    // reading it as a `MethodInfo` field would be a category error.
    if is_frame_base(base) {
        return spill_value_at(insns, at, load, depth);
    }
    if index.is_none() && *addend == 0 {
        return method_for_operand(insns, at, base, depth);
    }
    None
}

/// The method a merged (phi) version provably holds: EVERY incoming argument
/// must name the SAME method.
///
/// SSA inserts a phi wherever two paths join, so a version that holds one
/// `MethodInfo*` on both paths still becomes a phi. Refusing every phi made that
/// shape fall back to `fnptr_<addr>` even though the merge ITSELF proves the
/// answer. The proof is a conjunction, never a vote: one `Unknown` or unresolved
/// argument, or two arguments that name different methods, refuses — a phi whose
/// arms disagree is exactly the case where the callee is not a single method and
/// a name would be a fabrication.
fn phi_method(insns: &[Insn], at: usize, depth: u8) -> Option<MethodRef> {
    if depth == 0 {
        return None;
    }
    let args = insns[at].operands.get(1..)?;
    // A phi with fewer than two arguments is not a merge. (One arm is a
    // malformed listing, not evidence.)
    if args.len() < 2 {
        return None;
    }
    let mut found: Option<MethodRef> = None;
    for a in args {
        if matches!(a, Operand::Unknown | Operand::MissingAbiArg) {
            return None;
        }
        let m = method_for_operand(insns, at, a, depth)?;
        match &found {
            None => found = Some(m),
            Some(prev) if !same_method(prev, &m) => return None,
            Some(_) => {}
        }
    }
    found
}

/// The value a spilled call target provably holds.
///
/// A `MethodInfo*` that crosses a block boundary is often written to a FRAME
/// SLOT and reloaded (`mov [rbp-8], rax … mov rcx, [rbp-8] … call [rcx]`), so the
/// call's target is a memory load rather than a versioned local, and the chain
/// walk above (which follows locals) refuses it. The proof is the nearest
/// preceding `Move` whose DESTINATION is that exact slot — nearest, because the
/// most recent store is the reaching one — and it is gated so a wrong callee can
/// never come out of it:
///
///   * the slot's base must be a FRAME POINTER (`sp`/`fp`, or a named `stack…`
///     slot). A heap or static slot can be written through an alias this pass
///     cannot see, so only a stack slot has a single, textual writer;
///   * nothing between the store and the load may be a CALL — a call can take
///     the slot's address and write it. Refusing costs a name; guessing costs
///     the truth about which method is being called;
///   * the store must be a plain `Move` (an RMW writes a computed value, not
///     the value being spilled), and the value it stores must itself resolve.
fn spill_value_at(insns: &[Insn], at: usize, slot: &Operand, depth: u8) -> Option<MethodRef> {
    if depth == 0 || !is_frame_base_of(slot) {
        return None;
    }
    for k in (0..at).rev() {
        if matches!(insns[k].op, Op::Call | Op::CallVoid | Op::IndirectCall) {
            return None;
        }
        if insns[k].dest() == Some(slot) {
            if !matches!(insns[k].op, Op::Move) {
                return None;
            }
            let src = insns[k].operands.get(1)?;
            return method_for_operand(insns, k, src, depth - 1);
        }
    }
    None
}

/// Is this operand a FRAME POINTER (or a named stack slot) — the only kind of
/// slot the compiler spills a value to.
fn is_frame_base_of(slot: &Operand) -> bool {
    match slot {
        Operand::Memory { base, .. } => is_frame_base(base),
        _ => false,
    }
}

fn is_frame_base(base: &Operand) -> bool {
    match base {
        Operand::Register(r) => {
            let r = r.trim();
            r == "sp" || r == "fp" || r == "rsp" || r == "rbp" || r.starts_with("stack")
        }
        _ => false,
    }
}

/// The legacy pre-SSA fallback: with a `Register` target there is no definition
/// chain to walk, so the nearest preceding method materialization is the only
/// evidence available. Nearest-first (the value most recently written is the one
/// the call goes through), unlike the forward scan this replaced, which took the
/// FURTHEST match in the window.
fn loose_method_local(
    insns: &[Insn],
    at: usize,
    method_locals: &HashMap<u32, MethodRef>,
) -> Option<MethodRef> {
    for k in (at.saturating_sub(12)..at).rev() {
        let Some(d) = insns[k].dest().and_then(|x| x.as_local()) else {
            continue;
        };
        if let Some(m) = method_locals.get(&d) {
            return Some(m.clone());
        }
        if let Some(Operand::Method(m)) = insns[k].operands.get(1) {
            return Some(m.clone());
        }
    }
    None
}

/// Rebuild a call's operand list to the callee's declared shape, KEEPING the
/// argument operands the IR already has.
///
/// `[Method, dest, this?, param…]` is the layout every consumer reads
/// (`Insn::dest_position` puts `Op::Call`'s destination at index 1;
/// `csharp::call_parts` reads the receiver at 2 and the params after it), and
/// the incoming dump is the lifter's ABI register dump, which is positionally
/// ordered — so its first `receiver? + params` entries are this call's own
/// arguments in order.
///
/// IDEMPOTENT, and that is load-bearing: this pass runs after
/// `reslice_call_arguments`, so for a call that pass already shaped, the
/// "incoming dump" here IS its output — the receiver and parameters in the same
/// order — and re-deriving the same slots from it produces the same list. A
/// second shape pass that dropped a trailing parameter or appended holes would
/// corrupt every named call in the dump. Entries beyond that are register-dump filler (the
/// `in_x1`-style phantom arguments) and are dropped. A slot with no operand is
/// a HOLE, never `Unknown`: the renderer prints `__abi` for the first and
/// `__unknown` for the second, and only the first says "the ABI carried no
/// value here" instead of "the lifter lost this value".
fn signature_shaped_operands(
    m: &MethodRef,
    old: &[Operand],
    old_is_indirect: bool,
) -> Vec<Operand> {
    // `Op::IndirectCall` is the legacy bare `[target, args…]` list; `Op::Call`
    // is `[target, dest, args…]`.
    let arg_start = if old_is_indirect { 1 } else { 2 };
    let dump: Vec<Operand> = old.iter().skip(arg_start.min(old.len())).cloned().collect();
    let mut out: Vec<Operand> = Vec::with_capacity(m.sig.params.len() + 3);
    out.push(Operand::Method(m.clone()));
    out.push(if old_is_indirect {
        // No result operand in the incoming layout; the slot starts empty.
        Operand::Unknown
    } else {
        old.get(1).cloned().unwrap_or(Operand::Unknown)
    });
    let mut take = 0usize;
    if !m.sig.is_static {
        // A missing receiver is an ABI fact — a hole, never a fabricated value.
        out.push(dump.first().cloned().unwrap_or(Operand::MissingAbiArg));
        take = 1;
    }
    for _ in &m.sig.params {
        out.push(dump.get(take).cloned().unwrap_or(Operand::MissingAbiArg));
        take += 1;
    }
    out
}

#[cfg(test)]
static TEST_COUNTER_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod folded_pick_tests {
    use super::*;
    use crate::recovery::ir::NamedTypeRef;
    use crate::recovery::types::{ManagedType, MethodRef, MethodSigInfo, NamedType};

    fn mref(method_index: u32, tdi: i32, tyname: &str, name: &str) -> MethodRef {
        MethodRef {
            image_index: 0,
            type_def_index: tdi,
            method_index,
            name: name.into(),
            declaring_namespace: "System.Collections.Generic".into(),
            declaring_type_name: tyname.into(),
            assembly: "mscorlib".into(),
            sig: MethodSigInfo {
                ret: ManagedType::Unknown,
                params: vec![],
                is_static: false,
            },
        }
    }

    /// The ALLOCATION names the constructor, and it is matched by IDENTITY of
    /// the receiver local — not by a look-back window.
    ///
    /// `List<T>::.ctor` and `Dictionary<K,V>::.ctor` compile to the same body
    /// (both just call the base), so they share one address and
    /// `methods_by_rva` keeps one of them — which turned every `new`
    /// expression in the dump into the same generic class. The call site can
    /// tell them apart because the object it constructs came from a `NewObj`
    /// carrying the TYPE.
    ///
    /// Measured on the v24.5 PE dump: with a fixed 4-instruction window the
    /// picker named 78 of 5,551 folded call sites. The allocation and the
    /// `.ctor` call it feeds are two separate instructions, and any constructor
    /// that takes arguments puts them further apart than that.
    #[test]
    fn allocation_names_the_constructor_across_a_gap() {
        let cands = vec![
            mref(1, 10, "List", ".ctor"),
            mref(2, 20, "Dictionary", ".ctor"),
        ];
        let mut insns: Vec<Insn> = Vec::new();
        insns.push(Insn::new(
            Op::NewObj(Some(NamedTypeRef {
                ty: NamedType {
                    namespace: "System.Collections.Generic".into(),
                    name: "List".into(),
                    assembly: Some("mscorlib".into()),
                    type_def_index: 10,
                },
            })),
            vec![Operand::Local(7)],
            0x1000,
        ));
        // six instructions of argument setup between the allocation and the call
        for k in 0..6u32 {
            insns.push(Insn::new(
                Op::Move,
                vec![Operand::Local(20 + k), Operand::imm(k as i64)],
                0x1004 + u64::from(k) * 4,
            ));
        }
        insns.push(Insn::new(
            Op::Call,
            vec![
                Operand::Method(cands[0].clone()),
                Operand::Unknown,
                Operand::Local(7),
            ],
            0x1020,
        ));
        let i = insns.len() - 1;
        let table = LocalTable::default();
        let parents: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();
        let picked = pick_folded_definition(&cands, &insns, i, &table, &parents, abi(&insns[i]));
        assert_eq!(
            picked,
            Ok(0),
            "the allocation's type must pick `List::.ctor`, not the canonical member"
        );
    }

    /// The call's ABI argument dump, exactly as `resolve_folded_names` reads it
    /// out of the (not yet re-sliced) operand list.
    fn abi(insn: &Insn) -> &[Operand] {
        let start = 2.min(insn.operands.len());
        &insn.operands[start..]
    }

    /// A local table whose versions carry the ABI register origins that
    /// `abi_int_slots` reads — without them it refuses (correctly) and the
    /// tests would be exercising that refusal instead of the predicate.
    fn origin_table(pairs: &[(u32, &str)]) -> LocalTable {
        let mut t = LocalTable::default();
        for (v, origin) in pairs {
            t.infos.push(crate::recovery::locals::LocalInfo {
                slot_origin: (*origin).into(),
                ty: ManagedType::Unknown,
                is_this: false,
                is_param: None,
                param_comp: None,
                is_incoming_arg: false,
                is_call_clobber: false,
                is_return_buffer: false,
            });
            t.index_of.insert(*v, t.infos.len() - 1);
        }
        t
    }

    /// The ARGUMENT SHAPE alone names the callee when the receiver's class is
    /// unknown — the largest evidence gap in the dump, and one no amount of
    /// receiver typing can close.
    ///
    /// `FirstPassBehaviour::Update`'s address carries 272 folded definitions
    /// with 0..4 parameters, and a call site that passes two argument values
    /// cannot be one of the four-parameter ones. Requiring the class to own the
    /// member refused every one of those sites (`NoClass`), because the shapes
    /// that reach a folded address are exactly the ones whose receiver has no
    /// recovered class.
    #[test]
    fn abi_shape_alone_names_a_folded_definition() {
        let mut wide = mref(2, 20, "Dictionary", "Wide");
        wide.sig.is_static = true;
        wide.sig.params = vec![ManagedType::Unknown; 4];
        let mut narrow = mref(3, 30, "Hashtable", "Narrow");
        narrow.sig.is_static = true;
        narrow.sig.params = vec![ManagedType::Unknown; 2];
        let cands = vec![wide, narrow];
        // A call whose ABI dump carries TWO values and shows an unused integer
        // register: slots x2..x7 were never written by the caller.
        let mut ops = vec![Operand::Method(cands[0].clone()), Operand::Unknown];
        ops.push(Operand::Local(3));
        ops.push(Operand::Local(4));
        for _ in 0..6 {
            ops.push(Operand::MissingAbiArg);
        }
        let insns = vec![Insn::new(Op::Call, ops, 0x3000)];
        let table = origin_table(&[(3, "x0"), (4, "x1")]);
        let parents: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();
        assert_eq!(
            pick_folded_definition(&cands, &insns, 0, &table, &parents, abi(&insns[0])),
            Ok(1),
            "only the two-parameter member can be reached by a two-argument call"
        );
    }

    /// ...and it must NOT fire while the integer argument registers are all in
    /// use. A callee with more arguments than the register capacity passes the
    /// rest on the STACK, so a full run is consistent with an arity the dump
    /// cannot see, and picking on it would name a member the call is not.
    #[test]
    fn a_full_argument_register_run_proves_nothing() {
        let mut wide = mref(2, 20, "Dictionary", "Wide");
        wide.sig.is_static = true;
        wide.sig.params = vec![ManagedType::Unknown; 4];
        let mut narrow = mref(3, 30, "Hashtable", "Narrow");
        narrow.sig.is_static = true;
        narrow.sig.params = vec![ManagedType::Unknown; 2];
        let cands = vec![wide, narrow];
        // Every register holds a value: the caller could be passing two and
        // spilling nothing, or reading a stack argument this dump cannot show.
        let mut ops = vec![Operand::Method(cands[1].clone()), Operand::Unknown];
        let mut origins: Vec<(u32, &str)> = Vec::new();
        let names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];
        for (k, name) in names.iter().enumerate() {
            ops.push(Operand::Local(3 + k as u32));
            origins.push((3 + k as u32, name));
        }
        let insns = vec![Insn::new(Op::Call, ops, 0x3000)];
        let table = origin_table(&origins);
        let parents: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();
        assert_eq!(
            pick_folded_definition(&cands, &insns, 0, &table, &parents, abi(&insns[0])),
            Err(SharedMiss::NoClass),
            "a saturated register run must not be read as an arity"
        );
    }

    /// With no call-site evidence at all the picker must REFUSE, so the caller
    /// can count the guess instead of presenting it as a fact.
    #[test]
    fn no_evidence_is_reported_as_a_miss_not_a_guess() {
        let cands = vec![
            mref(1, 10, "List", "get_Count"),
            mref(2, 20, "Dictionary", "get_Count"),
        ];
        let insns = vec![
            Insn::new(
                Op::Call,
                vec![
                    Operand::Method(cands[1].clone()),
                    Operand::Unknown,
                    Operand::Local(3),
                ],
                0x2000,
            ),
            Insn::new(Op::Return, vec![Operand::Local(3)], 0x2004),
        ];
        let table = LocalTable::default();
        let parents: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();
        assert_eq!(
            pick_folded_definition(&cands, &insns, 0, &table, &parents, abi(&insns[0])),
            Err(SharedMiss::NoClass),
            "a receiver whose class was never recovered is a MISS"
        );
    }
}

#[cfg(test)]
mod abi_recovery_tests {
    use super::*;
    use crate::il2cpp::structures::{
        Il2CppInterfaceOffset, Il2CppTypeDefinition,
    };
    use crate::recovery::types::{ManagedType, MethodSigInfo, PrimKind};

    fn method(name: &str, ty: &str, params: Vec<ManagedType>, is_static: bool) -> MethodRef {
        MethodRef {
            image_index: 0,
            type_def_index: 1,
            method_index: 0,
            name: name.into(),
            declaring_namespace: "NS".into(),
            declaring_type_name: ty.into(),
            assembly: "asm".into(),
            sig: MethodSigInfo {
                ret: ManagedType::Unknown,
                params,
                is_static,
            },
        }
    }

    /// A local table whose versions carry the ABI register origins and types the
    /// recovery pass reads.
    fn table_with(entries: &[(u32, &str, ManagedType)]) -> LocalTable {
        let mut t = LocalTable::default();
        for (v, origin, ty) in entries {
            t.infos.push(crate::recovery::locals::LocalInfo {
                slot_origin: (*origin).into(),
                ty: ty.clone(),
                is_this: false,
                is_param: None,
                param_comp: None,
                is_incoming_arg: false,
                is_call_clobber: false,
                is_return_buffer: false,
            });
            t.index_of.insert(*v, t.infos.len() - 1);
        }
        t
    }

    fn i4() -> ManagedType {
        ManagedType::Primitive(PrimKind::I4)
    }

    fn r8() -> ManagedType {
        ManagedType::Primitive(PrimKind::R8)
    }

    /// The value a caller put into argument register `x0` is still there at the
    /// call, so the `__abi` hole is filled from it.
    #[test]
    fn an_integer_hole_is_filled_from_its_live_argument_register() {
        let callee = method("Use", "C", vec![i4()], true);
        let mut insns = vec![
            Insn::new(Op::Move, vec![Operand::Local(7), Operand::imm(41)], 0x1000),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Method(callee),
                    Operand::Unknown,
                    Operand::MissingAbiArg,
                ],
                0x1004,
            ),
        ];
        let table = table_with(&[(7, "x0", i4())]);
        assert_eq!(recover_abi_arguments(&mut insns, &table), 1);
        assert_eq!(insns[1].operands[2], Operand::imm(41));
    }

    /// The receiver of an instance call is argument 0, and a hole there is
    /// filled the same way — an instance call whose receiver was holed used to
    /// render `__abi.Method(...)`.
    #[test]
    fn a_missing_receiver_is_filled_from_x0() {
        let callee = method("Tick", "Enemy", vec![], false);
        let mut insns = vec![
            Insn::new(Op::Move, vec![Operand::Local(4), Operand::Local(9)], 0x2000),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Method(callee),
                    Operand::Unknown,
                    Operand::MissingAbiArg,
                ],
                0x2004,
            ),
        ];
        // `v9` is a computed value, not an argument register, so the rename gate
        // does not fire; `v4` is the x0 carrier.
        let table = table_with(&[
            (4, "x0", ManagedType::Object),
            (9, "g3", ManagedType::Object),
        ]);
        assert_eq!(recover_abi_arguments(&mut insns, &table), 1);
        assert_eq!(insns[1].operands[2], Operand::Local(9));
    }

    /// A CALL between the definition and this call clobbers every argument
    /// register — the value is gone, so the slot must stay a hole.
    #[test]
    fn a_call_in_between_refuses_the_fill() {
        let callee = method("Use", "C", vec![i4()], true);
        let other = method("Other", "C", vec![], true);
        let mut insns = vec![
            Insn::new(Op::Move, vec![Operand::Local(7), Operand::imm(41)], 0x3000),
            Insn::new(
                Op::Call,
                vec![Operand::Method(other), Operand::Unknown],
                0x3004,
            ),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Method(callee),
                    Operand::Unknown,
                    Operand::MissingAbiArg,
                ],
                0x3008,
            ),
        ];
        let table = table_with(&[(7, "x0", i4())]);
        assert_eq!(
            recover_abi_arguments(&mut insns, &table),
            0,
            "x0 is caller-saved: a call in between destroys the proof"
        );
        assert_eq!(insns[2].operands[2], Operand::MissingAbiArg);
    }

    /// A definition far from the call is not "still live" — the register may
    /// have been reused through a copy SSA coalesced away, which leaves no
    /// instruction exhibiting the reuse.
    #[test]
    fn a_stale_definition_outside_the_live_window_refuses() {
        let callee = method("Use", "C", vec![i4()], true);
        let mut insns = vec![Insn::new(
            Op::Move,
            vec![Operand::Local(7), Operand::imm(41)],
            0x4000,
        )];
        // 17 unrelated moves: no call, but far enough that the def is stale.
        for k in 0..17u32 {
            insns.push(Insn::new(
                Op::Move,
                vec![Operand::Local(100 + k), Operand::imm(k as i64)],
                0x4004 + u64::from(k) * 4,
            ));
        }
        insns.push(Insn::new(
            Op::Call,
            vec![
                Operand::Method(callee),
                Operand::Unknown,
                Operand::MissingAbiArg,
            ],
            0x4100,
        ));
        let table = table_with(&[(7, "x0", i4())]);
        let at = insns.len() - 1;
        assert_eq!(recover_abi_arguments(&mut insns, &table), 0);
        assert_eq!(insns[at].operands[2], Operand::MissingAbiArg);
    }

    /// `mov x0, x1` is the register-rename shape: the SSA copy pass coalesces
    /// these, so the definition of `x0` here is a stale value dressed as the
    /// carrier. This is exactly the shape that made `align_call_arguments` hand
    /// the receiver to the wrong parameter.
    #[test]
    fn a_rename_from_another_argument_register_refuses() {
        let callee = method("Use", "C", vec![i4()], true);
        let mut insns = vec![
            Insn::new(Op::Move, vec![Operand::Local(5), Operand::imm(7)], 0x5000),
            Insn::new(Op::Move, vec![Operand::Local(7), Operand::Local(5)], 0x5004),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Method(callee),
                    Operand::Unknown,
                    Operand::MissingAbiArg,
                ],
                0x5008,
            ),
        ];
        let table = table_with(&[(7, "x0", i4()), (5, "x1", i4())]);
        assert_eq!(recover_abi_arguments(&mut insns, &table), 0);
        assert_eq!(insns[2].operands[2], Operand::MissingAbiArg);
    }

    /// The two register classes are allocated independently: a `double`
    /// parameter is carried in the FP file, so an integer carrier proves
    /// nothing about it.
    #[test]
    fn a_float_parameter_is_not_filled_from_an_integer_register() {
        let callee = method("SetScale", "C", vec![r8()], true);
        let mut insns = vec![
            Insn::new(Op::Move, vec![Operand::Local(7), Operand::imm(3)], 0x6000),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Method(callee),
                    Operand::Unknown,
                    Operand::MissingAbiArg,
                ],
                0x6004,
            ),
        ];
        let table = table_with(&[(7, "x0", i4())]);
        assert_eq!(recover_abi_arguments(&mut insns, &table), 0);
        assert_eq!(insns[1].operands[2], Operand::MissingAbiArg);
    }

    /// ...and the FP file's own carrier works, when the class agrees.
    #[test]
    fn a_float_parameter_is_filled_from_its_fp_register() {
        let callee = method("SetScale", "C", vec![r8()], true);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(6), Operand::DoubleLiteral(1.5)],
                0x7000,
            ),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Method(callee),
                    Operand::Unknown,
                    Operand::MissingAbiArg,
                ],
                0x7004,
            ),
        ];
        let table = table_with(&[(6, "vd0", r8())]);
        assert_eq!(recover_abi_arguments(&mut insns, &table), 1);
        assert_eq!(insns[1].operands[2], Operand::DoubleLiteral(1.5));
    }

    /// A mixed signature counts the two classes separately. The method is
    /// static (no `this`), so the operand list is
    /// `[target, dest, param0(i4), param1(r8), param2(i4)]`: the integer
    /// parameters are `x0` and `x1` — the float in between does NOT shift the
    /// integer numbering — and the float is FP slot 0.
    #[test]
    fn the_two_classes_are_counted_separately_across_mixed_parameters() {
        let callee = method("Mix", "C", vec![i4(), r8(), i4()], true);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(1), Operand::Immediate(7)],
                0x7ff0,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::DoubleLiteral(2.0)],
                0x8000,
            ),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Method(callee),
                    Operand::Unknown,
                    Operand::MissingAbiArg,
                    Operand::MissingAbiArg,
                    Operand::MissingAbiArg,
                ],
                0x8004,
            ),
        ];
        let table = table_with(&[(1, "x0", i4()), (3, "vd0", r8())]);
        assert_eq!(recover_abi_arguments(&mut insns, &table), 2);
        assert_eq!(
            insns[2].operands[2],
            Operand::Immediate(7),
            "parameter 0 is the first INT parameter, i.e. x0"
        );
        assert_eq!(
            insns[2].operands[3],
            Operand::DoubleLiteral(2.0),
            "the double is parameter 1 (FP slot 0), not parameter 0"
        );
        assert_eq!(
            insns[2].operands[4],
            Operand::MissingAbiArg,
            "parameter 2 is the second INT parameter, i.e. x1, which is not defined"
        );
    }

    // ---- MethodInfo chain -------------------------------------------------

    fn memory(base: u32, addend: i64) -> Operand {
        Operand::Memory {
            base: Box::new(Operand::Local(base)),
            index: None,
            scale: 1,
            addend,
            width: 0,
        }
    }

    /// Run the pass. Pass 1's map is built by pass 1 itself from the
    /// `Move(_, Method)` instructions, so the caller only has to lay the IR out
    /// the way the metadata-usage materialization would have.
    fn resolve_indirect(insns: &mut Vec<Insn>) -> ResolveStats {
        let by_addr: HashMap<u64, u32> = HashMap::new();
        let mut of = |_mi: u32| -> Option<MethodRef> { None };
        let mut stats = ResolveStats::default();
        resolve_methodinfo_usages(insns, &by_addr, &mut of, &mut stats);
        stats
    }

    /// `mi = MethodInfo(Foo.Bar) ; fn = [mi + 0] ; call fn` — the deref at
    /// offset 0 is `methodPointer`, so the call names `Foo.Bar`.
    #[test]
    fn a_call_through_a_methodinfo_code_pointer_is_named() {
        let target = method("Bar", "Foo", vec![i4()], false);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target.clone())],
                0x9000,
            ),
            Insn::new(Op::Move, vec![Operand::Local(4), memory(3, 0)], 0x9004),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(4), Operand::Local(9)],
                0x9008,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 1);
        assert!(matches!(insns[2].op, Op::Call));
        match insns[2].operands.first() {
            Some(Operand::Method(m)) => assert_eq!(m.name, "Bar"),
            other => panic!("expected the named method, got {other:?}"),
        }
        assert_eq!(
            insns[2].operands[2],
            Operand::Local(9),
            "the receiver operand must survive the rename"
        );
    }

    /// A different `MethodInfo` field is not a code pointer: `[mi + 8]` is
    /// `methodMetadataHandle`/`invoker`, so the call must stay unresolved.
    #[test]
    fn a_load_from_another_methodinfo_field_refuses() {
        let target = method("Bar", "Foo", vec![], false);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target.clone())],
                0xA000,
            ),
            Insn::new(Op::Move, vec![Operand::Local(4), memory(3, 8)], 0xA004),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(4), Operand::Local(9)],
                0xA008,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 0);
        assert!(matches!(insns[2].op, Op::IndirectCall));
    }

    /// The bug the window scan had: a `MethodInfo` load that has nothing to do
    /// with the call used to name it. The chain from the call's OWN target now
    /// proves the link, so an unrelated load cannot answer for it.
    #[test]
    fn an_unrelated_methodinfo_load_does_not_name_the_call() {
        let target = method("Bar", "Foo", vec![], false);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target.clone())],
                0xB000,
            ),
            // v4 is some other pointer with no definition at all.
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(4), Operand::Local(9)],
                0xB004,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 0);
        assert!(matches!(insns[1].op, Op::IndirectCall));
    }

    /// Naming a call must not delete the arguments the IR still has.
    #[test]
    fn naming_a_call_keeps_its_argument_operands() {
        let target = method("Bar", "Foo", vec![i4(), i4()], true);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target.clone())],
                0xC000,
            ),
            Insn::new(Op::Move, vec![Operand::Local(4), memory(3, 0)], 0xC004),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(4), Operand::Local(11), Operand::Local(12)],
                0xC008,
            ),
        ];
        resolve_indirect(&mut insns);
        assert_eq!(
            insns[2].operands[2],
            Operand::Local(11),
            "argument 0 must survive"
        );
        assert_eq!(insns[2].operands[3], Operand::Local(12));
    }

    /// A phi whose arms all name the SAME method names the call.
    ///
    /// SSA inserts a merge wherever two paths join, so a `MethodInfo*` holding
    /// one method on both paths still becomes a phi. Refusing every phi was a
    /// regression: the call fell back to `fnptr_<addr>` even though the merge
    /// itself proves the callee.
    #[test]
    fn a_phi_of_agreeing_arms_names_the_call() {
        let target = method("Bar", "Foo", vec![], false);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target.clone())],
                0xD000,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(5), Operand::Method(target.clone())],
                0xD004,
            ),
            Insn::new(
                Op::Phi,
                vec![Operand::Local(6), Operand::Local(3), Operand::Local(5)],
                0xD008,
            ),
            Insn::new(Op::Move, vec![Operand::Local(7), memory(6, 0)], 0xD00C),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(7), Operand::Local(9)],
                0xD010,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 1, "the merge proves one callee");
        match insns[4].operands.first() {
            Some(Operand::Method(m)) => assert_eq!(m.name, "Bar"),
            other => panic!("expected the named method, got {other:?}"),
        }
    }

    /// A phi whose arms name DIFFERENT methods is not one callee: the honest
    /// `fnptr_<addr>` stays.
    #[test]
    fn a_phi_of_disagreeing_arms_refuses() {
        let a = method("Bar", "Foo", vec![], false);
        let b = method("Baz", "Foo", vec![], false);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(a)],
                0xE000,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(5), Operand::Method(b)],
                0xE004,
            ),
            Insn::new(
                Op::Phi,
                vec![Operand::Local(6), Operand::Local(3), Operand::Local(5)],
                0xE008,
            ),
            Insn::new(Op::Move, vec![Operand::Local(7), memory(6, 0)], 0xE00C),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(7), Operand::Local(9)],
                0xE010,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 0, "two callees prove neither");
        assert!(matches!(insns[4].op, Op::IndirectCall));
    }

    /// One unresolved arm is enough to refuse: the conjunction is what makes the
    /// phi proof sound.
    #[test]
    fn a_phi_with_an_unknown_arm_refuses() {
        let target = method("Bar", "Foo", vec![], false);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target)],
                0xF000,
            ),
            Insn::new(
                Op::Phi,
                vec![Operand::Local(6), Operand::Local(3), Operand::Unknown],
                0xF004,
            ),
            Insn::new(Op::Move, vec![Operand::Local(7), memory(6, 0)], 0xF008),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(7), Operand::Local(9)],
                0xF00C,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 0);
        assert!(matches!(insns[3].op, Op::IndirectCall));
    }

    /// A frame slot the code pointer was spilled to: the nearest store proves
    /// the reload.
    fn frame_slot(addend: i64) -> Operand {
        Operand::Memory {
            base: Box::new(Operand::Register("sp".into())),
            index: None,
            scale: 1,
            addend,
            width: 8,
        }
    }

    #[test]
    fn a_reloaded_frame_slot_is_named_from_its_store() {
        let target = method("Bar", "Foo", vec![], false);
        let slot = frame_slot(-8);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target.clone())],
                0x10000,
            ),
            // mov [sp-8], mi
            Insn::new(Op::Move, vec![slot.clone(), Operand::Local(3)], 0x10004),
            // mov rcx, [sp-8]
            Insn::new(Op::Move, vec![Operand::Local(4), slot.clone()], 0x10008),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(4), Operand::Local(9)],
                0x1000C,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 1, "the store proves the reload");
        match insns[3].operands.first() {
            Some(Operand::Method(m)) => assert_eq!(m.name, "Bar"),
            other => panic!("expected the named method, got {other:?}"),
        }
    }

    /// A call between the spill and the reload can take the slot's address and
    /// write it, so the link is unprovable.
    #[test]
    fn a_call_between_the_spill_and_the_reload_refuses() {
        let target = method("Bar", "Foo", vec![], false);
        let slot = frame_slot(-16);
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target)],
                0x11000,
            ),
            Insn::new(Op::Move, vec![slot.clone(), Operand::Local(3)], 0x11004),
            Insn::new(
                Op::CallVoid,
                vec![Operand::Method(method("Touch", "Foo", vec![], true))],
                0x11008,
            ),
            Insn::new(Op::Move, vec![Operand::Local(4), slot.clone()], 0x1100C),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(4), Operand::Local(9)],
                0x11010,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 0);
        assert!(matches!(insns[4].op, Op::IndirectCall));
    }

    /// A slot whose base is not a frame pointer is not a spill: a heap or static
    /// slot can be written through an alias.
    #[test]
    fn a_load_from_a_non_frame_slot_is_not_a_spill() {
        let target = method("Bar", "Foo", vec![], false);
        let slot = Operand::Memory {
            base: Box::new(Operand::Local(1)),
            index: None,
            scale: 1,
            addend: -8,
            width: 8,
        };
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(3), Operand::Method(target)],
                0x12000,
            ),
            Insn::new(Op::Move, vec![slot.clone(), Operand::Local(3)], 0x12004),
            Insn::new(Op::Move, vec![Operand::Local(4), slot.clone()], 0x12008),
            Insn::new(
                Op::IndirectCall,
                vec![Operand::Local(4), Operand::Local(9)],
                0x1200C,
            ),
        ];
        let stats = resolve_indirect(&mut insns);
        assert_eq!(stats.virtuals_resolved, 0);
        assert!(matches!(insns[3].op, Op::IndirectCall));
    }

    // ---- interface pair counting ------------------------------------------

    /// The stored count wins when it fits the pair array.
    #[test]
    fn a_valid_stored_count_is_used_as_is() {
        let mut td = Il2CppTypeDefinition::default();
        td.interface_offsets_start = 2;
        td.interface_offsets_count = 3;
        let defs = vec![td];
        assert_eq!(interface_pair_count(&defs, 0, 9), (3, false));
    }

    /// A count of ZERO is authoritative: it is the ordinary `this type
    /// implements no interfaces`, and treating it as "field absent" would hand
    /// the type its neighbour's pairs.
    #[test]
    fn a_zero_count_contributes_nothing_even_with_pairs_after_it() {
        let mut a = Il2CppTypeDefinition::default();
        a.interface_offsets_start = 0;
        a.interface_offsets_count = 0;
        let mut b = Il2CppTypeDefinition::default();
        b.interface_offsets_start = 3;
        let defs = vec![a, b];
        assert_eq!(
            interface_pair_count(&defs, 0, 5),
            (0, false),
            "a stored 0 means no interfaces; the [0,3) slice is NOT ours"
        );
    }

    /// A count that runs past the pair array is unusable, so the slice is
    /// derived from the next type's start instead.
    #[test]
    fn an_inconsistent_count_is_derived_from_the_next_type_start() {
        let mut a = Il2CppTypeDefinition::default();
        a.interface_offsets_start = 0;
        a.interface_offsets_count = 999;
        let mut b = Il2CppTypeDefinition::default();
        b.interface_offsets_start = 3;
        let defs = vec![a, b];
        assert_eq!(
            interface_pair_count(&defs, 0, 5),
            (3, true),
            "the slice [0,3) belongs to the first type"
        );
    }

    /// A count that runs past the pair array is unusable, so it is derived too
    /// — and the derivation can never exceed the array.
    #[test]
    fn an_out_of_range_count_is_derived_and_clamped() {
        let mut a = Il2CppTypeDefinition::default();
        a.interface_offsets_start = 0;
        a.interface_offsets_count = 999;
        let b = Il2CppTypeDefinition::default(); // start 0 => derived 0
        let defs = vec![a, b];
        assert_eq!(interface_pair_count(&defs, 0, 4), (0, true));
        // ...and the last type owns the tail of the array.
        let defs2 = vec![{
            let mut t = Il2CppTypeDefinition::default();
            t.interface_offsets_start = 1;
            t.interface_offsets_count = 999;
            t
        }];
        assert_eq!(interface_pair_count(&defs2, 0, 4), (3, true));
    }

    /// A start past the end contributes nothing, and a negative start is a
    /// parse artefact rather than an index.
    #[test]
    fn a_start_outside_the_array_contributes_nothing() {
        let mut a = Il2CppTypeDefinition::default();
        a.interface_offsets_start = 7;
        a.interface_offsets_count = 4;
        let mut b = Il2CppTypeDefinition::default();
        b.interface_offsets_start = -1;
        let defs = vec![a, b];
        assert_eq!(interface_pair_count(&defs, 0, 4), (0, false));
        assert_eq!(interface_pair_count(&defs, 1, 4), (0, false));
    }

    /// The derived count must actually produce slots: the whole point of the
    /// derivation is that a table whose count field is inconsistent with the
    /// array still resolves.
    #[test]
    fn a_derived_count_still_produces_a_dispatch_slot() {
        let mut td = Il2CppTypeDefinition::default();
        td.interface_offsets_start = 0;
        td.interface_offsets_count = 999; // provably inconsistent with 2 pairs
        let mut next = Il2CppTypeDefinition::default();
        next.interface_offsets_start = 1;
        next.interface_offsets_count = 999; // and so is this one
        let type_defs = vec![td, next];
        let offsets = vec![
            Il2CppInterfaceOffset {
                type_index: 7,
                offset: 5,
            },
            Il2CppInterfaceOffset {
                type_index: 9,
                offset: 6,
            },
        ];
        let mut slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        slots.insert((42, 2), method("Dispose", "IDisposable", vec![], false));
        let (map, stats) = build_interface_dispatch_detailed(
            &type_defs,
            &offsets,
            &[],
            &slots,
            &HashMap::new(),
            &|ti| (ti == 7).then_some(42),
        );
        assert_eq!(map.get(&(0, 7)).map(|m| m.name.as_str()), Some("Dispose"));
        // BOTH types derive a count (neither stored a usable one): the first
        // owns [0,1), the second owns the tail [1,2). Only the first pair names
        // a TypeDef, so the second is counted as unresolvable rather than
        // guessed.
        assert_eq!(stats.counts_derived, 2);
        assert_eq!(stats.pairs_seen, 2);
        assert_eq!(stats.pairs_unresolved_type, 1);
        assert_eq!(stats.slots_built, 1);
    }
}

#[cfg(test)]
mod interface_dispatch_tests {
    use super::*;
    use crate::il2cpp::structures::{
        Il2CppInterfaceOffset, Il2CppMethodDefinition, Il2CppTypeDefinition,
    };
    use crate::recovery::types::{ManagedType, MethodRef, MethodSigInfo};

    fn m(name: &str, ty: &str, tdi: i32) -> MethodRef {
        MethodRef {
            image_index: tdi.max(0) as usize,
            type_def_index: tdi,
            method_index: 0,
            name: name.into(),
            declaring_namespace: String::new(),
            declaring_type_name: ty.into(),
            assembly: String::new(),
            sig: MethodSigInfo {
                ret: ManagedType::Unknown,
                params: vec![],
                is_static: false,
            },
        }
    }

    /// An interface's block begins at `offset` SLOTS into the class vtable, and
    /// its method at `method.slot` therefore lands at `offset + slot`.
    #[test]
    fn an_interface_slot_lands_at_base_plus_method_slot() {
        let mut td = Il2CppTypeDefinition::default();
        td.interface_offsets_start = 0;
        td.interface_offsets_count = 1;
        let type_defs = vec![td];
        let offsets = vec![Il2CppInterfaceOffset {
            type_index: 7,
            offset: 5,
        }];
        let mut slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        slots.insert((42, 2), m("Dispose", "IDisposable", 42));

        let map =
            build_interface_dispatch(&type_defs, &offsets, &slots, &|ti| (ti == 7).then_some(42));
        assert_eq!(
            map.get(&(0, 7)).map(|m| m.name.as_str()),
            Some("Dispose"),
            "interface block at slot 5 + method slot 2 = vtable slot 7"
        );
    }

    /// A slot two DIFFERENT interfaces both claim carries no evidence, so it
    /// must be dropped — naming the wrong interface method is worse than the
    /// honest `fnptr_0x…`.
    #[test]
    fn a_slot_two_interfaces_claim_is_dropped() {
        let mut td = Il2CppTypeDefinition::default();
        td.interface_offsets_start = 0;
        td.interface_offsets_count = 2;
        let type_defs = vec![td];
        let offsets = vec![
            Il2CppInterfaceOffset {
                type_index: 7,
                offset: 5,
            },
            Il2CppInterfaceOffset {
                type_index: 8,
                offset: 5,
            },
        ];
        let mut slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        slots.insert((42, 2), m("Dispose", "IDisposable", 42));
        slots.insert((43, 2), m("Reset", "IResettable", 43));

        let map = build_interface_dispatch(&type_defs, &offsets, &slots, &|ti| match ti {
            7 => Some(42),
            8 => Some(43),
            _ => None,
        });
        assert!(
            map.get(&(0, 7)).is_none(),
            "two interfaces cannot both own vtable slot 7"
        );
    }

    /// A type with no interface offsets contributes nothing (the common case: a
    /// plain class), and an unresolvable interface type index is skipped rather
    /// than guessed.
    #[test]
    fn a_class_without_interfaces_contributes_nothing() {
        let td = Il2CppTypeDefinition::default();
        let type_defs = vec![td];
        let offsets = vec![Il2CppInterfaceOffset {
            type_index: 7,
            offset: 5,
        }];
        let mut slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        slots.insert((42, 2), m("Dispose", "IDisposable", 42));
        let map = build_interface_dispatch(&type_defs, &offsets, &slots, &|_| Some(42));
        assert!(map.is_empty());
    }

    /// The inherited case: the interface block is placed on a BASE class, so the
    /// lookup must walk the ancestry the way the class-vtable lookup does.
    #[test]
    fn interface_slot_is_found_through_the_base_chain() {
        let mut map: HashMap<(i32, usize), MethodRef> = HashMap::new();
        map.insert((5, 7), m("Dispose", "IDisposable", 42));
        let mut parents: HashMap<i32, Vec<i32>> = HashMap::new();
        parents.insert(9, vec![9, 5]); // derived 9, then base 5
        assert!(interface_slot_in(&map, &parents, 9, 7).is_some());
        assert!(interface_slot_in(&map, &parents, 9, 8).is_none());
    }

    /// A pair that found no slot is not one thing, and the difference decides
    /// whether there is a defect at all. "Declares no methods" and "no usable
    /// slot" are facts about the metadata; "unindexed" means the metadata proves
    /// a placement and the join lost it — every such pair is a call site left as
    /// `fnptr_0x…` forever.
    #[test]
    fn a_slotless_pair_reports_why_it_has_no_slot() {
        // Three plain classes, one interface pair each, then the three
        // interfaces those pairs name.
        let class = |pair_at: i32| {
            let mut t = Il2CppTypeDefinition::default();
            t.interface_offsets_start = pair_at;
            t.interface_offsets_count = 1;
            t
        };
        let (c0, c1, c2) = (class(0), class(1), class(2));
        // 3: an interface whose method carries slot 3 and no generic container,
        // absent from the slot map because an interface method has no body.
        let mut iface_plain = Il2CppTypeDefinition::default();
        iface_plain.method_start = 0;
        iface_plain.method_count = 1;
        // 4: an interface whose only method carries a generic container.
        let mut iface_generic = Il2CppTypeDefinition::default();
        iface_generic.method_start = 1;
        iface_generic.method_count = 1;
        // 5: an interface that declares no method at all.
        let iface_empty = Il2CppTypeDefinition::default();
        let type_defs = vec![c0, c1, c2, iface_plain, iface_generic, iface_empty];

        let mut plain = Il2CppMethodDefinition::default();
        plain.slot = 3;
        plain.generic_container_index = -1;
        let mut generic = Il2CppMethodDefinition::default();
        generic.slot = 3;
        generic.generic_container_index = 0;
        let method_defs = vec![plain, generic];

        let offsets: Vec<Il2CppInterfaceOffset> = (0..3)
            .map(|k| Il2CppInterfaceOffset {
                type_index: 7 + k,
                offset: 5,
            })
            .collect();
        // The slot map holds NOTHING — the observed corpus shape: an interface
        // method is abstract, so it never reaches a map built from methods with a
        // native body.
        let slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        let iface_of = |ti: i32| match ti {
            7 => Some(3),
            8 => Some(4),
            _ => Some(5),
        };
        let (map, stats) = build_interface_dispatch_detailed(
            &type_defs,
            &offsets,
            &method_defs,
            &slots,
            &HashMap::new(),
            &iface_of,
        );
        assert!(map.is_empty(), "nothing was placeable");
        assert_eq!(stats.pairs_without_slots, 3);
        assert_eq!(
            stats.pairs_without_slots_unindexed_plain, 1,
            "a non-generic method with a usable slot was lost in the join"
        );
        assert_eq!(
            stats.pairs_without_slots_unindexed_generic, 1,
            "only a generic-container method carried a usable slot"
        );
        assert_eq!(
            stats.pairs_without_slots_no_methods, 1,
            "an interface that declares no methods is a fact, not a loss"
        );
        assert_eq!(stats.pairs_without_slots_slotless, 0);
        assert_eq!(stats.pairs_without_slots_unclassified, 0);

        // Without the method table no claim is made — and the table can never
        // change the MAP, only the honesty of the breakdown.
        let (map2, stats2) = build_interface_dispatch_detailed(
            &type_defs,
            &offsets,
            &[],
            &slots,
            &HashMap::new(),
            &iface_of,
        );
        assert_eq!(map2.len(), map.len());
        assert_eq!(stats2.pairs_without_slots, 3);
        assert_eq!(
            stats2.pairs_without_slots_unclassified, 2,
            "only the two slot-bearing cases need the method table; an interface \
             that declares no methods is decidable from the type list alone"
        );
        assert_eq!(stats2.pairs_without_slots_no_methods, 1);
    }

    /// The repair itself. An interface's DECLARED method carries a slot but has
    /// no native body, so it is absent from the body-bearing slot map and the
    /// pair used to be dropped (7,192 of 7,331 lost pairs on the x64 corpus).
    /// It must place from the metadata-built source — while the uniqueness rule
    /// keeps refusing a slot two interfaces claim.
    #[test]
    fn a_pair_places_from_the_interfaces_declared_methods() {
        let mut owner = Il2CppTypeDefinition::default();
        owner.interface_offsets_start = 0;
        owner.interface_offsets_count = 2;
        let mut iface_a = Il2CppTypeDefinition::default();
        iface_a.flags = 0x20; // TypeAttributes.Interface
        iface_a.method_start = 0;
        iface_a.method_count = 1;
        let mut iface_b = Il2CppTypeDefinition::default();
        iface_b.flags = 0x20;
        iface_b.method_start = 1;
        iface_b.method_count = 1;
        let type_defs = vec![owner, iface_a, iface_b];
        let offsets = vec![
            Il2CppInterfaceOffset {
                type_index: 7,
                offset: 5,
            },
            Il2CppInterfaceOffset {
                type_index: 8,
                offset: 5,
            },
        ];
        let mut a = Il2CppMethodDefinition::default();
        a.slot = 2;
        a.generic_container_index = -1;
        let mut b = Il2CppMethodDefinition::default();
        b.slot = 2;
        b.generic_container_index = -1;
        let method_defs = vec![a, b];
        // The body-bearing map holds NOTHING: both methods are abstract.
        let body_slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        // The declared source holds both, at each interface's own slot 2.
        let mut iface_slots: HashMap<(i32, usize), MethodRef> = HashMap::new();
        iface_slots.insert((1, 2), m("Dispose", "IDisposable", 1));
        iface_slots.insert((2, 2), m("Reset", "IResettable", 2));

        // Both pairs land on class vtable slot 5 + 2 = 7 and the two interfaces
        // disagree about what lives there, so it is refused: naming either would
        // be a guess, which is worse than the honest `fnptr_0x…`.
        let (map, stats) = build_interface_dispatch_detailed(
            &type_defs,
            &offsets,
            &method_defs,
            &body_slots,
            &iface_slots,
            &|ti| match ti {
                7 => Some(1),
                _ => Some(2),
            },
        );
        assert!(
            map.is_empty(),
            "two interfaces cannot both own class vtable slot 7"
        );
        assert_eq!(stats.conflicts_dropped, 1);

        // With a single pair the declared method places the slot — and the
        // counter says the placement came from that source, not the body map.
        let (map1, stats1) = build_interface_dispatch_detailed(
            &type_defs[..1],
            &offsets[..1],
            &method_defs,
            &body_slots,
            &iface_slots,
            &|ti| (ti == 7).then_some(1),
        );
        assert_eq!(
            map1.get(&(0, 7)).map(|r| r.name.as_str()),
            Some("Dispose"),
            "interface block at slot 5 + method slot 2 = class vtable slot 7"
        );
        assert_eq!(stats1.slots_built, 1);
        assert_eq!(
            stats1.slots_from_declared_interfaces, 1,
            "the placement came from the interface's declared method"
        );
        assert_eq!(stats1.pairs_without_slots, 0);
        assert_eq!(stats1.conflicts_dropped, 0);
    }

    /// The two fallback counters distinguish NEVER-ATTEMPTED from
    /// ATTEMPTED-MISSED — the property §6.1 requires before any interface
    /// residue can be called "explained".
    ///
    /// A bare hit count cannot tell "the call site never reached a table"
    /// (klass unknown or slot not constant, so no `(tdi, slot)` key exists)
    /// from "the table was consulted and held no entry". The first wants a
    /// typing/slot-shape fix, the second a table-coverage fix. This test pins
    /// the distinction: a constant slot with a known klass MUST increment
    /// attempts (and hits when the entry exists), while a site whose klass
    /// cannot resolve MUST NOT increment either — proving the iOS residue
    /// (`klass + idx*16 + 312` computed through an `Arith` chain, plus the
    /// `MethodInfo`-half `+8` loads) never reaches a table at all.
    ///
    /// Fails pre-fix (no `*_lookup_counts`, no increments in `lookup_*`);
    /// passes after.
    #[test]
    fn vtable_and_iface_lookups_count_attempts_vs_hits_separately() {
        use crate::recovery::ir::{Insn, Op, Operand};
        use crate::recovery::locals::LocalTable;
        // Serialized: `resolve_arith_slot` hits below also bump the vtable
        // pair, so an exact delta here is only meaningful in isolation.
        let _guard = super::TEST_COUNTER_SERIAL.lock().unwrap();
        let (v_attempted_before, v_hit_before) = super::vtable_lookup_counts();
        let (i_attempted_before, i_hit_before) = super::iface_lookup_counts();
        // A declared class slot: one attempt, one hit.
        let mut maps = super::ResolveMaps::default();
        maps.vtable_slots.insert((5, 3), m("Tick", "Enemy", 5));
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        assert!(super::lookup_vtable_slot(&maps, &parents, 5, 3).is_some());
        // A constant slot no ancestry declares: attempted, missed.
        assert!(super::lookup_vtable_slot(&maps, &parents, 5, 99).is_none());
        // The interface fallback is consulted (attempted) even when the
        // process-global table is empty — that is exactly the
        // attempted-missed case the bare hit count hides.
        let _ = super::lookup_interface_slot(&parents, 5, 3);
        let (v_attempted, v_hit) = super::vtable_lookup_counts();
        let (i_attempted, i_hit) = super::iface_lookup_counts();
        assert_eq!(
            v_attempted - v_attempted_before,
            2,
            "both constant-slot klass-known sites must count as vtable attempts"
        );
        assert_eq!(
            v_hit - v_hit_before,
            1,
            "only the declared slot must count as a vtable hit"
        );
        assert_eq!(
            i_attempted - i_attempted_before,
            1,
            "the fallback must count as an iface attempt even on a miss"
        );
        assert_eq!(
            i_hit - i_hit_before,
            0,
            "with no installed table entry the fallback must record no hit"
        );
        // NEVER-ATTEMPTED: a `Memory` slot load whose klass is an unknown
        // immediate never reaches either table, so neither pair moves.
        let table = LocalTable::default();
        let insns = vec![Insn::new(
            Op::Call,
            vec![
                Operand::Memory {
                    base: Box::new(Operand::Immediate(0xdead_beef)),
                    index: None,
                    scale: 1,
                    addend: 0x130,
                    width: 8,
                },
                Operand::Unknown,
            ],
            0x1000,
        )];
        let target = insns[0].operands[0].clone();
        let before_v = super::vtable_lookup_counts();
        let before_i = super::iface_lookup_counts();
        // `0xdead_beef` names no klass in an empty map, so the lookup cannot
        // be keyed — `None` without touching either counter.
        assert!(
            super::resolve_indirect_target(&insns, 0, &target, &table, &maps, &parents, 8, 4)
                .is_none()
        );
        assert_eq!(
            super::vtable_lookup_counts(),
            before_v,
            "an unresolvable klass must not count as a vtable attempt"
        );
        assert_eq!(
            super::iface_lookup_counts(),
            before_i,
            "an unreached fallback must not count as an iface attempt"
        );
    }
}

#[cfg(test)]
mod delegate_arith_tests {
    use super::*;
    use crate::recovery::ir::{ArithKind, Insn, Op, Operand};
    use crate::recovery::locals::{LocalInfo, LocalTable};
    use crate::recovery::types::{ManagedType, NamedType};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn nt(ns: &str, name: &str, tdi: i32) -> Arc<NamedType> {
        Arc::new(NamedType {
            namespace: ns.into(),
            name: name.into(),
            assembly: None,
            type_def_index: tdi,
        })
    }

    fn table_with_typed(entries: &[(u32, ManagedType, bool)]) -> LocalTable {
        let mut t = LocalTable::default();
        for (v, ty, is_this) in entries {
            t.infos.push(LocalInfo {
                slot_origin: "x0".into(),
                ty: ty.clone(),
                is_this: *is_this,
                is_param: None,
                param_comp: None,
                is_incoming_arg: false,
                is_call_clobber: false,
                is_return_buffer: false,
            });
            t.index_of.insert(*v, t.infos.len() - 1);
        }
        t
    }

    fn invoke_impl_field() -> FieldRef {
        FieldRef {
            type_def_index: 5422,
            field_index: 29303,
            name: "invoke_impl".into(),
            declaring_type_name: "Delegate".into(),
            field_type: ManagedType::IntPtr,
            is_static: false,
        }
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

    /// An `InstanceField invoke_impl` off a proven delegate base resolves to
    /// `Invoke` with that delegate as the declaring type.
    ///
    /// Fails pre-fix (`resolve_indirect_target` only accepted `Memory`/copy
    /// targets, so this returned `None` and the site stayed `fnptr_`); passes
    /// after. This is the 0x525c64 `Action` shape (`v63 = v62.invoke_impl`
    /// with `v62: Action`).
    #[test]
    fn delegate_instancefield_with_delegate_base_resolves_to_invoke() {
        let field = invoke_impl_field();
        let delegate_ty = ManagedType::Class(nt("System", "Action", 5150));
        let table = table_with_typed(&[(62, delegate_ty, false), (63, ManagedType::IntPtr, false)]);
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(63),
                    Operand::InstanceField {
                        base: Box::new(Operand::Local(62)),
                        field: field.clone(),
                    },
                ],
                0x525d08,
            ),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Local(63),
                    Operand::Unknown,
                    Operand::Local(64),
                    Operand::Local(65),
                ],
                0x525d1c,
            ),
        ];
        let target = insns[1].operands[0].clone();
        let maps = ResolveMaps::default();
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        let m = super::resolve_indirect_target(&insns, 1, &target, &table, &maps, &parents, 8, 4)
            .expect("a proven delegate base must name Invoke");
        assert_eq!(m.name, "Invoke");
        assert_eq!(m.declaring_type_name, "Action");
        // The rewriter must fix the receiver to the delegate base, not keep
        // the field operand (`method_code`) as the receiver.
        let dl = super::delegate_base_of_target_nocount(&insns, 1, &target, &table, 4)
            .expect("the same proof must yield the receiver");
        assert_eq!(dl, 62);
    }

    /// An `invoke_impl` load off an untyped base refuses — the honest
    /// `fnptr_` stays. Characterization pin: passes before and after, and
    /// exists so a future "accept any invoke_impl offset" change fails here
    /// first (a wrong `Invoke` is worse than an unnamed call).
    #[test]
    fn delegate_instancefield_with_unproven_base_refuses() {
        let field = invoke_impl_field();
        let table = table_with_typed(&[
            (62, ManagedType::IntPtr, false),
            (63, ManagedType::IntPtr, false),
        ]);
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(63),
                    Operand::InstanceField {
                        base: Box::new(Operand::Local(62)),
                        field,
                    },
                ],
                0x3000,
            ),
            Insn::new(Op::Call, vec![Operand::Local(63), Operand::Unknown], 0x3004),
        ];
        let target = insns[1].operands[0].clone();
        let maps = ResolveMaps::default();
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        assert!(
            super::resolve_indirect_target(&insns, 1, &target, &table, &maps, &parents, 8, 4)
                .is_none(),
            "an untyped base proves no delegate: must stay fnptr_"
        );
    }

    /// A bare `Field(invoke_impl)` resolves when `this` is provably delegate
    /// (the locals sweep only folds a `this` base into that form).
    ///
    /// Fails pre-fix (bare `Field` targets returned `None`); passes after.
    /// This is the 0x1bc4eac `MulticastDelegate.Invoke` shape (`t =
    /// this.invoke_impl` with delegate `this`).
    #[test]
    fn delegate_this_field_with_delegate_this_resolves() {
        let field = invoke_impl_field();
        let this_ty = ManagedType::Class(nt("System", "MulticastDelegate", 5400));
        let table = table_with_typed(&[(6, this_ty, true), (18, ManagedType::IntPtr, false)]);
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(18), Operand::Field(field)],
                0x1000,
            ),
            Insn::new(Op::Call, vec![Operand::Local(18), Operand::Unknown], 0x1004),
        ];
        let target = insns[1].operands[0].clone();
        let maps = ResolveMaps::default();
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        let m = super::resolve_indirect_target(&insns, 1, &target, &table, &maps, &parents, 8, 4)
            .expect("delegate this must name Invoke");
        assert_eq!(m.name, "Invoke");
        assert_eq!(m.declaring_type_name, "MulticastDelegate");
    }

    /// A bare `Field(invoke_impl)` with no delegate `this` refuses.
    #[test]
    fn delegate_this_field_without_delegate_this_refuses() {
        let field = invoke_impl_field();
        let table = table_with_typed(&[
            (
                6,
                ManagedType::Primitive(crate::recovery::types::PrimKind::I4),
                true,
            ),
            (18, ManagedType::IntPtr, false),
        ]);
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(18), Operand::Field(field)],
                0x1000,
            ),
            Insn::new(Op::Call, vec![Operand::Local(18), Operand::Unknown], 0x1004),
        ];
        let target = insns[1].operands[0].clone();
        let maps = ResolveMaps::default();
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        assert!(
            super::resolve_indirect_target(&insns, 1, &target, &table, &maps, &parents, 8, 4)
                .is_none(),
            "without a delegate this there is no proven receiver"
        );
    }

    /// `klass_tdi_of` must NOT peel `Arith` address computations.
    ///
    /// Fails pre-fix (the `Local`-follow arm ignored the opcode, so `v60 =
    /// v59 + 312` returned the klass of `v59` and let the caller key a slot
    /// the site never loads); passes after (refuses, so `Arith` bases route
    /// to `resolve_arith_slot`, which accounts for the constant).
    #[test]
    fn klass_tdi_of_refuses_an_arith_address() {
        // v50 = [v46] (klass), v60 = v50 + 312 (address, NOT the klass).
        let recv_ty = ManagedType::Class(nt("", "IRandomGen", 18));
        let table = table_with_typed(&[
            (46, recv_ty, false),
            (50, ManagedType::IntPtr, false),
            (60, ManagedType::IntPtr, false),
        ]);
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(50), mem(Operand::Local(46), 0)],
                0x1000,
            ),
            Insn::new(
                Op::Arith(ArithKind::Add),
                vec![
                    Operand::Local(60),
                    Operand::Local(50),
                    Operand::Immediate(312),
                ],
                0x1004,
            ),
        ];
        let maps = ResolveMaps::default();
        assert!(
            super::klass_tdi_of(&insns, 2, &Operand::Local(60), &table, &maps, 4).is_none(),
            "an address computation is not the klass"
        );
        // ...while the plain klass load still resolves.
        assert_eq!(
            super::klass_tdi_of(&insns, 2, &Operand::Local(50), &table, &maps, 4),
            Some(18)
        );
    }

    /// A constant `Arith` base (`klass + 320` = base 304 + slot 1 * 16)
    /// resolves through the chain.
    ///
    /// Fails pre-fix (the `Memory` arm saw `addend 0`, which is off-grid, and
    /// returned before `klass_tdi_of`); passes after via `resolve_arith_slot`.
    /// Serialized: the hit bumps the shared vtable pair (see the existing
    /// counter test).
    #[test]
    fn arith_constant_base_resolves_through_the_chain() {
        let _guard = super::TEST_COUNTER_SERIAL.lock().unwrap();
        let recv_ty = ManagedType::Class(nt("", "Enemy", 5));
        let table = table_with_typed(&[
            (46, recv_ty, false),
            (50, ManagedType::IntPtr, false),
            (60, ManagedType::IntPtr, false),
            (61, ManagedType::IntPtr, false),
        ]);
        let mut maps = ResolveMaps::default();
        maps.vtable_slots.insert(
            (5, 1),
            crate::recovery::types::MethodRef {
                image_index: 0,
                type_def_index: 5,
                method_index: 99,
                name: "Tick".into(),
                declaring_namespace: "".into(),
                declaring_type_name: "Enemy".into(),
                assembly: "".into(),
                sig: crate::recovery::types::MethodSigInfo {
                    ret: ManagedType::Unknown,
                    params: vec![],
                    is_static: false,
                },
            },
        );
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        // v50 = [v46] (klass), v60 = v50 + 320, fn = [v60].
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(50), mem(Operand::Local(46), 0)],
                0x1000,
            ),
            Insn::new(
                Op::Arith(ArithKind::Add),
                vec![
                    Operand::Local(60),
                    Operand::Local(50),
                    Operand::Immediate(320),
                ],
                0x1004,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(61), mem(Operand::Local(60), 0)],
                0x1008,
            ),
            Insn::new(Op::Call, vec![Operand::Local(61), Operand::Unknown], 0x100c),
        ];
        let target = insns[3].operands[0].clone();
        let m = super::resolve_indirect_target(&insns, 3, &target, &table, &maps, &parents, 8, 4)
            .expect("klass + constant on the grid must resolve");
        assert_eq!(m.name, "Tick");
    }

    /// The interface-scan loop shape (`v59 = klass + scaled-idx; v60 = v59 +
    /// 312; fn = [v60]`) refuses: the slot is runtime-computed, so no single
    /// method is provable. Characterization pin — passes before and after —
    /// pinning that the resolver keeps `fnptr_` instead of guessing a slot.
    #[test]
    fn arith_dynamic_base_refuses_without_guessing() {
        let recv_ty = ManagedType::Class(nt("", "IRandomGen", 18));
        let table = table_with_typed(&[
            (46, recv_ty, false),
            (50, ManagedType::IntPtr, false),
            (58, ManagedType::IntPtr, false),
            (59, ManagedType::IntPtr, false),
            (60, ManagedType::IntPtr, false),
            (61, ManagedType::IntPtr, false),
        ]);
        let maps = ResolveMaps::default();
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        let insns = vec![
            Insn::new(
                Op::Move,
                vec![Operand::Local(50), mem(Operand::Local(46), 0)],
                0x1000,
            ),
            Insn::new(
                Op::Arith(ArithKind::Add),
                vec![Operand::Local(59), Operand::Local(50), Operand::Local(58)],
                0x1004,
            ),
            Insn::new(
                Op::Arith(ArithKind::Add),
                vec![
                    Operand::Local(60),
                    Operand::Local(59),
                    Operand::Immediate(312),
                ],
                0x1008,
            ),
            Insn::new(
                Op::Move,
                vec![Operand::Local(61), mem(Operand::Local(60), 0)],
                0x100c,
            ),
            Insn::new(Op::Call, vec![Operand::Local(61), Operand::Unknown], 0x1010),
        ];
        let target = insns[4].operands[0].clone();
        assert!(
            super::resolve_indirect_target(&insns, 4, &target, &table, &maps, &parents, 8, 4)
                .is_none(),
            "a runtime-computed slot must stay fnptr_, never a guessed method"
        );
    }

    /// A register-indexed slot (`[klass + idx*16 + off]`) refuses for the same
    /// reason: no constant slot exists to key.
    #[test]
    fn indexed_slot_refuses_without_guessing() {
        let recv_ty = ManagedType::Class(nt("", "Enemy", 5));
        let table = table_with_typed(&[(46, recv_ty, false)]);
        let mut maps = ResolveMaps::default();
        maps.vtable_slots.insert(
            (5, 1),
            crate::recovery::types::MethodRef {
                image_index: 0,
                type_def_index: 5,
                method_index: 99,
                name: "Tick".into(),
                declaring_namespace: "".into(),
                declaring_type_name: "Enemy".into(),
                assembly: "".into(),
                sig: crate::recovery::types::MethodSigInfo {
                    ret: ManagedType::Unknown,
                    params: vec![],
                    is_static: false,
                },
            },
        );
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        let insns = vec![Insn::new(
            Op::Call,
            vec![
                Operand::Memory {
                    base: Box::new(Operand::Local(46)),
                    index: Some(Box::new(Operand::Local(58))),
                    scale: 16,
                    addend: 312,
                    width: 0,
                },
                Operand::Unknown,
            ],
            0x1000,
        )];
        let target = insns[0].operands[0].clone();
        assert!(
            super::resolve_indirect_target(&insns, 0, &target, &table, &maps, &parents, 8, 4)
                .is_none(),
            "an indexed slot is dynamic by construction"
        );
    }

    /// Hidden delegate fields are dropped, USER arguments are kept.
    ///
    /// The native passes `method_code`/`method`/`invoke_impl` (all off the
    /// same delegate base) ahead of the real parameters. Keeping them renders
    /// `Invoke(method_code, method, …)` — the `fnptr_` shape with a new name;
    /// dropping a user argument loses semantics. Fails pre-fix (the rewrite
    /// kept everything, so the later trim to `params == []` dropped the user
    /// args too and `OnTokenRefreshed` rendered `Invoke()`); passes after.
    #[test]
    fn delegate_rewrite_keeps_user_args_drops_hidden_fields() {
        use crate::recovery::types::{FieldRef, PrimKind};
        let base_ty = ManagedType::Class(nt("System", "Action", 5150));
        let table = table_with_typed(&[
            (62, base_ty, false),
            (63, ManagedType::IntPtr, false),
            (64, ManagedType::IntPtr, false),
            (65, ManagedType::IntPtr, false),
            (80, ManagedType::Primitive(PrimKind::I4), false),
        ]);
        let hidden = |name: &str| FieldRef {
            type_def_index: 5422,
            field_index: 29303,
            name: name.into(),
            declaring_type_name: "Delegate".into(),
            field_type: ManagedType::IntPtr,
            is_static: false,
        };
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(64),
                    Operand::InstanceField {
                        base: Box::new(Operand::Local(62)),
                        field: hidden("method_code"),
                    },
                ],
                0x1000,
            ),
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(65),
                    Operand::InstanceField {
                        base: Box::new(Operand::Local(62)),
                        field: hidden("method"),
                    },
                ],
                0x1004,
            ),
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(63),
                    Operand::InstanceField {
                        base: Box::new(Operand::Local(62)),
                        field: hidden("invoke_impl"),
                    },
                ],
                0x1008,
            ),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Local(63),
                    Operand::Local(70),
                    Operand::Local(64),
                    Operand::Local(65),
                    Operand::Local(80),
                ],
                0x100c,
            ),
        ];
        let maps = ResolveMaps::default();
        let parents: HashMap<i32, Vec<i32>> = HashMap::new();
        let mut stats = super::ResolveStats::default();
        super::rewrite_virtual_calls(&mut insns, &table, &maps, &parents, 8, &mut stats);
        assert_eq!(stats.virtuals_resolved, 1);
        let call = &insns[3];
        assert!(
            matches!(call.operands.first(), Some(Operand::Method(m)) if m.name == "Invoke"),
            "must name Invoke, got {:?}",
            call.operands.first()
        );
        assert_eq!(
            call.operands.get(2),
            Some(&Operand::Local(62)),
            "receiver must be the delegate base"
        );
        assert!(
            call.operands
                .iter()
                .skip(3)
                .any(|o| *o == Operand::Local(80)),
            "the USER argument must survive, got {:?}",
            call.operands
        );
        assert!(
            !call
                .operands
                .iter()
                .skip(3)
                .any(|o| *o == Operand::Local(64) || *o == Operand::Local(65)),
            "hidden delegate fields must not render as Invoke params, got {:?}",
            call.operands
        );
    }
}

#[cfg(test)]
mod metadata_init_phi_use_tests {
    use super::*;

    #[test]
    fn metadata_init_result_used_by_prior_phi_is_preserved() {
        let mut insns = vec![
            Insn::new(
                Op::Phi,
                vec![Operand::Local(8), Operand::Local(7), Operand::Local(9)],
                0x1000,
            ),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x4a58a74), Operand::Local(9)],
                0x1004,
            ),
        ];
        let maps = ResolveMaps {
            metadata_init: vec![0x4a58a74],
            ..ResolveMaps::default()
        };
        let mut stats = ResolveStats::default();
        key_function_recovery(&mut insns, &maps, &HashMap::new(), &mut stats);
        assert!(
            matches!(insns[1].op, Op::Call),
            "phi edge use keeps the result-producing call"
        );
        assert_eq!(stats.key_functions, 0);
    }
}

#[cfg(test)]
mod indirect_target_tests {
    use super::*;
    use crate::recovery::locals::LocalInfo;

    fn widget_table() -> LocalTable {
        LocalTable {
            infos: vec![LocalInfo {
                slot_origin: "x0".into(),
                ty: ManagedType::Class(std::sync::Arc::new(NamedType {
                    namespace: "Test".into(),
                    name: "Widget".into(),
                    assembly: None,
                    type_def_index: 7,
                })),
                is_this: true,
                is_param: None,
                param_comp: None,
                is_incoming_arg: false,
                is_call_clobber: false,
                is_return_buffer: false,
            }],
            index_of: [(0u32, 0usize)].into_iter().collect(),
            param_names: vec![],
            param_comp_names: vec![],
        }
    }

    fn slot_method(name: &str) -> MethodRef {
        MethodRef {
            image_index: 0,
            type_def_index: 7,
            method_index: 0,
            name: name.into(),
            declaring_namespace: "Test".into(),
            declaring_type_name: "Widget".into(),
            assembly: String::new(),
            sig: super::super::types::MethodSigInfo {
                ret: ManagedType::Unknown,
                params: vec![],
                is_static: false,
            },
        }
    }

    /// The dispatch shape, at an arbitrary entry offset.
    ///
    /// `obj → [obj] = klass → [klass + off] = callee → call callee`, exactly the
    /// three instructions `CircularScrollingList.Refresh` @0x148A47C contains.
    fn dispatch_at(addend: i64) -> Vec<Insn> {
        vec![
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(1),
                    Operand::Memory {
                        width: 0,
                        base: Box::new(Operand::Local(0)),
                        index: None,
                        scale: 8,
                        addend: 0,
                    },
                ],
                0x1000,
            ),
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(2),
                    Operand::Memory {
                        width: 0,
                        base: Box::new(Operand::Local(1)),
                        index: None,
                        scale: 8,
                        addend,
                    },
                ],
                0x1004,
            ),
            Insn::new(Op::IndirectCall, vec![Operand::Local(2)], 0x1008),
        ]
    }

    /// An entry whose `methodPtr` half sits at `0x138 + slot*16` — the v29+
    /// `Il2CppClass` layout this image is built with.
    ///
    /// Real site (0x148A47C): `ldp x8, x1, [x9, #0x188]` loads the `methodPtr`
    /// at 0x188 and the `MethodInfo` at 0x190, so 0x188 is `0x138 + 5*16` and
    /// 392 is `8 (mod 16)`. Under the hardcoded 0x130 base the offset is
    /// rejected outright ("not a vtable methodPtr") and the site renders
    /// `fnptr_0x…` forever.
    ///
    /// Fails before the fix: 392 - 0x130 = 88, 88 % 16 = 8 → no slot.
    #[test]
    fn vtable_entry_at_0x138_keys_slot_five() {
        let table = widget_table();
        let mut insns = dispatch_at(0x188);
        let mut maps = ResolveMaps::default();
        maps.vtable_slots.insert((7, 5), slot_method("GetListLength"));

        let stats = run(&mut insns, &table, &maps);

        assert_eq!(
            stats.virtuals_resolved, 1,
            "the 0x138 grid must key slot 5, got {:?}",
            insns[2].op
        );
        match insns[2].operands.first() {
            Some(Operand::Method(m)) => assert_eq!(m.name, "GetListLength"),
            other => panic!("expected the declared method, got {other:?}"),
        }
    }

    /// A vtable call on a FIELD receiver whose local was never typed: the
    /// FieldRef's declared type is enough to key the slot.
    #[test]
    fn a_field_receiver_supplies_its_declared_type() {
        let table = widget_table();
        let mut maps = ResolveMaps::default();
        maps.vtable_slots.insert((7, 5), slot_method("GetListLength"));
        let field = FieldRef {
            type_def_index: 7,
            field_index: 1,
            name: "_ctrl".into(),
            declaring_type_name: "Holder".into(),
            field_type: ManagedType::Class(std::sync::Arc::new(NamedType {
                namespace: "Test".into(),
                name: "Widget".into(),
                assembly: None,
                type_def_index: 7,
            })),
            is_static: false,
        };
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(0),
                    Operand::InstanceField {
                        base: Box::new(Operand::Local(8)),
                        field,
                    },
                ],
                0x1000,
            ),
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(1),
                    Operand::Memory {
                        width: 0,
                        base: Box::new(Operand::Local(0)),
                        index: None,
                        scale: 8,
                        addend: 0,
                    },
                ],
                0x1004,
            ),
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(2),
                    Operand::Memory {
                        width: 8,
                        base: Box::new(Operand::Local(1)),
                        index: None,
                        scale: 8,
                        addend: 0x188,
                    },
                ],
                0x1008,
            ),
            Insn::new(
                Op::Call,
                vec![Operand::Local(2), Operand::Local(9), Operand::Local(0)],
                0x100c,
            ),
        ];
        let stats = run(&mut insns, &table, &maps);
        assert_eq!(stats.virtuals_resolved, 1, "{:?}", insns[3].op);
        match insns[3].operands.first() {
            Some(Operand::Method(m)) => assert_eq!(m.name, "GetListLength"),
            other => panic!("expected the declared method, got {other:?}"),
        }
    }

    /// Both entry grids in ONE run: an `≡0` offset keys the 0x130 grid and an
    /// `≡8` offset keys the 0x138 grid, and neither may steal the other's slot.
    ///
    /// This is the disjointness the two-candidate design rests on: a call
    /// target is always the `methodPtr` half, so the two bases can only add
    /// sites, never reinterpret one the other already keyed.
    #[test]
    fn the_two_entry_grids_are_disjoint_and_both_resolve() {
        let table = widget_table();
        let mut maps = ResolveMaps::default();
        maps.vtable_slots.insert((7, 1), slot_method("FromTheFirstGrid"));
        maps.vtable_slots.insert((7, 5), slot_method("FromTheSecondGrid"));

        // 0x140 = 0x130 + 1*16 (≡0 grid) → slot 1.
        let mut a = dispatch_at(0x140);
        let sa = run(&mut a, &table, &maps);
        assert_eq!(sa.virtuals_resolved, 1);
        match a[2].operands.first() {
            Some(Operand::Method(m)) => assert_eq!(m.name, "FromTheFirstGrid"),
            other => panic!("0x130 grid lost, got {other:?}"),
        }

        // 0x188 = 0x138 + 5*16 (≡8 grid) → slot 5.
        let mut b = dispatch_at(0x188);
        let sb = run(&mut b, &table, &maps);
        assert_eq!(sb.virtuals_resolved, 1);
        match b[2].operands.first() {
            Some(Operand::Method(m)) => assert_eq!(m.name, "FromTheSecondGrid"),
            other => panic!("0x138 grid lost, got {other:?}"),
        }
    }

    /// A resolved virtual call drops the MethodInfo companion the entry's
    /// SECOND half supplies, so the real argument list survives the
    /// signature-shaped trim.
    #[test]
    fn a_resolved_vtable_call_drops_the_companion_method_info() {
        let table = widget_table();
        let mut insns = vec![
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(1),
                    Operand::Memory {
                        width: 0,
                        base: Box::new(Operand::Local(0)),
                        index: None,
                        scale: 8,
                        addend: 0,
                    },
                ],
                0x1000,
            ),
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(2),
                    Operand::Memory {
                        width: 8,
                        base: Box::new(Operand::Local(1)),
                        index: None,
                        scale: 8,
                        addend: 0x188,
                    },
                ],
                0x1004,
            ),
            // Companion MethodInfo half of the same entry.
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(3),
                    Operand::Memory {
                        width: 8,
                        base: Box::new(Operand::Local(1)),
                        index: None,
                        scale: 8,
                        addend: 0x190,
                    },
                ],
                0x1008,
            ),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Local(2),
                    Operand::Local(9),
                    Operand::Local(0),
                    Operand::Local(3),
                    Operand::Local(4),
                ],
                0x100c,
            ),
        ];
        let mut maps = ResolveMaps::default();
        let mut m = slot_method("GetRegionName");
        m.sig.params = vec![ManagedType::Primitive(super::super::types::PrimKind::I4)];
        maps.vtable_slots.insert((7, 5), m);

        let stats = run(&mut insns, &table, &maps);
        assert_eq!(stats.virtuals_resolved, 1, "{:?}", insns[3].op);
        match insns[3].operands.as_slice() {
            [Operand::Method(m), Operand::Local(9), Operand::Local(0), Operand::Local(4)] => {
                assert_eq!(m.name, "GetRegionName");
            }
            other => panic!("expected the companion to be dropped, got {other:?}"),
        }
    }

    /// A slot load at an offset that is on NEITHER grid stays unresolved.
    #[test]
    fn an_offset_off_both_grids_stays_unresolved() {
        let table = widget_table();
        let mut insns = dispatch_at(0x144); // ≡4 (mod 16)
        let mut maps = ResolveMaps::default();
        maps.vtable_slots.insert((7, 1), slot_method("A"));
        maps.vtable_slots.insert((7, 5), slot_method("B"));

        let stats = run(&mut insns, &table, &maps);

        assert_eq!(stats.virtuals_resolved, 0);
        assert!(matches!(insns[2].op, Op::IndirectCall));
    }

    /// The `il2cpp_codegen_initialize_runtime_metadata(&slot)` shape: the
    /// argument is the ADDRESS of a metadata-usage cell, reached through a GOT
    /// load whose page SSA has materialized into its own local.
    ///
    /// ```text
    /// 0x148a514  adrp x0, 0x2f69000
    /// 0x148a518  ldr  x0, [x0, #0x7b0]   ; x0 = &cell
    /// 0x148a51c  bl   0x1302420
    /// ```
    ///
    /// Fails before the fix twice over: the gate was `Op::CallVoid` (no lifter
    /// emits it) and `static_addr_of` cannot see through the page local.
    fn init_call_shape(with_reader: bool) -> (Vec<Insn>, ResolveMaps) {
        let page: i64 = 0x2f69000;
        let cell: u64 = 0x3065b18;
        let mut insns = vec![
            // adrp x0, page
            Insn::new(Op::Move, vec![Operand::Local(1), Operand::Immediate(page)], 0x1000),
            // ldr x0, [x0, #0x7b0]  →  &cell
            Insn::new(
                Op::Move,
                vec![
                    Operand::Local(2),
                    Operand::Memory {
                        width: 0,
                        base: Box::new(Operand::Local(1)),
                        index: None,
                        scale: 8,
                        addend: 0x7b0,
                    },
                ],
                0x1004,
            ),
            // bl 0x1302420   [target, result, x0]
            Insn::new(
                Op::Call,
                vec![
                    Operand::Immediate(0x1302420),
                    Operand::Local(3),
                    Operand::Local(2),
                ],
                0x1008,
            ),
        ];
        if with_reader {
            // Someone consumes the result: the call is not void.
            insns.push(Insn::new(
                Op::Move,
                vec![Operand::Local(4), Operand::Local(3)],
                0x100c,
            ));
        }
        let mut maps = ResolveMaps::default();
        maps.got_addend_by_addr
            .insert(page as u64 + 0x7b0, cell);
        maps.literal_index_by_addr.insert(cell, 0);
        maps.literals.insert(0, "text".into());
        (insns, maps)
    }

    #[test]
    fn metadata_init_call_with_an_unread_result_is_dropped() {
        let (mut insns, maps) = init_call_shape(false);
        let mut stats = ResolveStats::default();
        key_function_recovery(&mut insns, &maps, &HashMap::new(), &mut stats);

        assert!(
            matches!(insns[2].op, Op::Nop),
            "the init call must be dropped, got {:?}",
            insns[2].op
        );
        assert_eq!(stats.key_functions, 1);
        assert_eq!(
            super::super::runtime_helpers::role_of(0x1302420),
            Some(super::super::runtime_helpers::RuntimeHelper::MetadataInit)
        );
    }

    /// The same call with a live result must be KEPT: dropping it would delete
    /// the value every later statement reads.
    #[test]
    fn metadata_init_call_with_a_read_result_is_kept() {
        let (mut insns, maps) = init_call_shape(true);
        let mut stats = ResolveStats::default();
        key_function_recovery(&mut insns, &maps, &HashMap::new(), &mut stats);

        assert!(
            matches!(insns[2].op, Op::Call),
            "a consumed result must keep the call, got {:?}",
            insns[2].op
        );
        assert_eq!(stats.key_functions, 0);
    }

    /// A call with the same argument shape but a RESOLVED target is managed
    /// code, not a runtime helper: it must never be dropped.
    #[test]
    fn a_resolved_method_taking_a_cell_is_not_dropped() {
        let (mut insns, mut maps) = init_call_shape(false);
        maps.methods_by_rva
            .insert(0x1302420, slot_method("RealManagedMethod"));
        let mut stats = ResolveStats::default();
        key_function_recovery(&mut insns, &maps, &HashMap::new(), &mut stats);

        assert!(matches!(insns[2].op, Op::Call));
        assert_eq!(stats.key_functions, 0);
    }
}

