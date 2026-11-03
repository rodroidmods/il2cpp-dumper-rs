//! Post-hoc field annotation pass.
//!
//! Runs AFTER the typing fixpoint has converged and after the standard
//! reconstruct passes. Scans all remaining `RawDeref` expressions in the
//! statement tree and resolves them to named `Field` accesses when the base
//! variable's type is known.
//!
//! ## Pattern
//!
//! ```text
//! *(nint*)(obj + 184)   →   obj.FieldName
//! ```
//!
//! where `obj` has a known type (from the local table) and offset 184 maps to
//! a named field via the metadata's field layout.
//!
//! ## Pointer chain resolution
//!
//! The pass also resolves multi-level pointer chains:
//!
//! ```text
//! *(nint*)(*(nint*)(obj +16) +8)   →   obj.Field1.Field2
//! ```
//!
//! This works bottom-up: the inner `RawDeref` is resolved first (to
//! `obj.Field1`), then the outer `RawDeref` sees a `Field` base. The pass
//! looks up the field's type via `field_types` to continue the chain.
//! Arbitrary depth is supported (2, 3, 4+ levels).
//!
//! ## How it works
//!
//! 1. The caller provides a `local_types` map: variable display name → TypeDef index.
//!    This is built from the `LocalTable` after the typing fixpoint converges.
//! 2. For each `Expr::RawDeref(Expr::Var(name), offset, _)`, look up `name` in
//!    `local_types` to get the tdi.
//! 3. Look up `(tdi, offset)` in `maps.field_layouts` to get the field name.
//! 4. Rewrite `RawDeref` to `Field(Var(name), field_name)`.
//! 5. For chains: when the base is `Field(recv, field_name)`, look up the
//!    field's type tdi via the reverse map `(tdi, field_name) → type_tdi`,
//!    then resolve the outer offset on that type.
//!
//! ## Safety
//!
//! Only rewrites when BOTH the variable's type AND the field at the offset are
//! definitively known. Unresolved patterns are left as `RawDeref` (conservative).

use super::super::csharp::{BinOp, Expr, Stmt};
use super::super::ir::IrType;
use super::super::resolve::ResolveMaps;
use super::super::types::{ManagedType, MethodRef};
use super::walk_mut;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Counters (process-wide, reported via `report()`)
// ---------------------------------------------------------------------------

static FIELDS_RESOLVED: AtomicU64 = AtomicU64::new(0);
static STATIC_FIELDS_RESOLVED: AtomicU64 = AtomicU64::new(0);
static STATIC_BYTES_RESOLVED: AtomicU64 = AtomicU64::new(0);

/// Cost split of the provenance work (see `timing_report`).
static TOKEN_NANOS: AtomicU64 = AtomicU64::new(0);
static DESCENT_NANOS: AtomicU64 = AtomicU64::new(0);
static DESCENT_CALLS: AtomicU64 = AtomicU64::new(0);

/// `ILR_TIMING` line: time spent classifying token variables and walking
/// embedded layouts, plus how often the descent ran.
pub(crate) fn timing_report() -> String {
    format!(
        "[newtime] token={:.0}ms descent={:.0}ms calls={}",
        TOKEN_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
        DESCENT_NANOS.load(Ordering::Relaxed) as f64 / 1e6,
        DESCENT_CALLS.load(Ordering::Relaxed),
    )
}

/// Reset counters (for testing).
#[cfg(test)]
fn reset_counters() {
    FIELDS_RESOLVED.store(0, Ordering::Relaxed);
    STATIC_FIELDS_RESOLVED.store(0, Ordering::Relaxed);
    STATIC_BYTES_RESOLVED.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Post-hoc field annotation pass.
///
/// Walks the statement tree looking for `RawDeref` patterns where the base
/// variable has a known type, and resolves them to named `Field` accesses
/// using the pre-computed field layouts from metadata.
///
/// `local_types` maps variable display names to their TypeDef index (tdi),
/// built from the `LocalTable` after the typing fixpoint converges.
///
/// `declaring_tdi` is the TypeDef index of the method's declaring type, used
/// for ptr_no_offset resolution context.
///
/// Returns the number of `RawDeref` → `Field` rewrites applied.
pub fn apply_with_maps(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
) -> usize {
    if maps.field_layouts.is_empty() && maps.static_field_layouts.is_empty() {
        return 0;
    }
    if std::env::var("ILR_FADBG").is_ok() {
        let mut rows: Vec<_> = local_types.iter().collect();
        rows.sort();
        for (name, tdi) in rows {
            let tyname = maps
                .tdi_names
                .get(tdi)
                .map(|(ns, n)| if ns.is_empty() { n.clone() } else { format!("{ns}.{n}") })
                .unwrap_or_else(|| "?".into());
            let mut fields = Vec::new();
            for off in [16i64, 24, 32, 48, 56, 108] {
                if let Some((f, _)) = maps.field_layouts.get(&(*tdi, off)) {
                    fields.push(format!("{off}=>{f}"));
                }
            }
            eprintln!("[fadbg] local {name} tdi={tdi} ({tyname}) fields: {}", fields.join(", "));
        }
    }
    // Klass/static-block provenance is computed on the UNANNOTATED tree: the
    // shapes that prove a token (`RawDeref(slot,0)`, `Field(x,"static_fields")`,
    // `RawDeref(klass,184)`) are exactly the ones this pass rewrites away, so
    // classifying later would see only the output of earlier guesses.
    let tok_start = std::time::Instant::now();
    let tokens = if std::env::var("ILR_NOTOK").is_ok() {
        TokenMaps::default()
    } else {
        build_token_maps(stmts, maps)
    };
    TOKEN_NANOS.fetch_add(
        tok_start.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );
    // Use the pre-computed reverse map from maps (built once globally).
    let mut rewrites = 0;
    let mut static_rewrites = 0;
    let mut static_byte_rewrites = 0;
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            let (r, sr, sbr) = rewrite_stmt(stmt, maps, local_types, &maps.field_type_by_name, declaring_tdi, &tokens);
            rewrites += r;
            static_rewrites += sr;
            static_byte_rewrites += sbr;
        }
    });
    FIELDS_RESOLVED.fetch_add(rewrites as u64, Ordering::Relaxed);
    STATIC_FIELDS_RESOLVED.fetch_add(static_rewrites as u64, Ordering::Relaxed);
    STATIC_BYTES_RESOLVED.fetch_add(static_byte_rewrites as u64, Ordering::Relaxed);
    rewrites + static_rewrites + static_byte_rewrites
}

/// Rewrite a single statement, recursing into nested bodies.
/// Returns (instance_rewrites, static_rewrites, static_byte_rewrites).
fn rewrite_stmt(
    stmt: &mut Stmt,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    field_type_by_name: &HashMap<(i32, String), i32>,
    declaring_tdi: i32,
    tokens: &TokenMaps,
) -> (usize, usize, usize) {
    let mut count = 0;
    let mut static_count = 0;
    let mut static_byte_count = 0;

    match stmt {
        Stmt::Assign(_, _, expr) | Stmt::Eval(expr) | Stmt::Return(Some(expr))
        | Stmt::Throw(expr) | Stmt::YieldReturn(expr) => {
            let (r, sr, sbr) = rewrite_expr(expr, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            let (r, sr, sbr) = rewrite_expr(cond, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
            for s in then_body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
            for s in else_body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Stmt::While { cond, body, .. } => {
            let (r, sr, sbr) = rewrite_expr(cond, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
            for s in body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Stmt::DoWhile { body, cond, .. } => {
            for s in body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
            let (r, sr, sbr) = rewrite_expr(cond, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
        }
        Stmt::For { init, incr, body, cond, .. } => {
            for s in init.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
            for s in incr.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
            if let Some(c) = cond {
                let (r, sr, sbr) = rewrite_expr(c, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
                static_byte_count += sbr;
            }
            for s in body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Stmt::Try { try_body, catches, finally_body, .. } => {
            for s in try_body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
            for (_, b) in catches.iter_mut() {
                for s in b.iter_mut() {
                    let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                    count += r;
                    static_count += sr;
                    static_byte_count += sbr;
                }
            }
            if let Some(fb) = finally_body {
                for s in fb.iter_mut() {
                    let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                    count += r;
                    static_count += sr;
                    static_byte_count += sbr;
                }
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases.iter_mut() {
                for s in b.iter_mut() {
                    let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                    count += r;
                    static_count += sr;
                    static_byte_count += sbr;
                }
            }
            for s in tail.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
            if let Some(arm) = default {
                for s in arm.iter_mut() {
                    let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                    count += r;
                    static_count += sr;
                    static_byte_count += sbr;
                }
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, b) in cases.iter_mut() {
                for s in b.iter_mut() {
                    let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                    count += r;
                    static_count += sr;
                    static_byte_count += sbr;
                }
            }
            if let Some(t) = tail {
                for s in t.iter_mut() {
                    let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                    count += r;
                    static_count += sr;
                    static_byte_count += sbr;
                }
            }
        }
        Stmt::ForEach { body, .. } => {
            for s in body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        // These statement kinds carry expressions the pass must still name.
        // The old walk covered bodies/conditions only, so an event target
        // such as `(*(nint*)(static_fields +16) +216).Event -= value` was
        // never visited: the whole class-init/klass chain survived as
        // `*(nint*)` text even though every level was resolvable
        // (UnRegisterForRunnerPawnEvents, 0x198AFB8, 17 event assignments).
        Stmt::EventAssign { field, value, .. } => {
            for e in [field, value] {
                let (r, sr, sbr) = rewrite_expr(e, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            for e in [collection, index, value] {
                let (r, sr, sbr) = rewrite_expr(e, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Stmt::Lock { expr, body } => {
            let (r, sr, sbr) = rewrite_expr(expr, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
            for s in body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Stmt::Using { decl, body } => {
            let (r, sr, sbr) = rewrite_stmt(decl, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
            for s in body.iter_mut() {
                let (r, sr, sbr) = rewrite_stmt(s, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        _ => {}
    }
    (count, static_count, static_byte_count)
}

/// Rewrite field-access patterns within an expression tree.
///
/// Recurses into sub-expressions first (bottom-up), then tries to resolve
/// the current expression as a `RawDeref` with a typed base.
/// Returns (instance_rewrites, static_rewrites, static_byte_rewrites).
fn rewrite_expr(
    expr: &mut Expr,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    field_type_by_name: &HashMap<(i32, String), i32>,
    declaring_tdi: i32,
    tokens: &TokenMaps,
) -> (usize, usize, usize) {
    let mut count = 0;
    let mut static_count = 0;
    let mut static_byte_count = 0;

    // First, recurse into sub-expressions (bottom-up so inner derefs resolve first).
    match expr {
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv.as_mut() {
                let (r, sr, sbr) = rewrite_expr(r, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
                static_byte_count += sbr;
            }
            for arg in args.iter_mut() {
                let (r, sr, sbr) = rewrite_expr(arg, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Expr::RawCall(_, args, _) => {
            for arg in args.iter_mut() {
                let (r, sr, sbr) = rewrite_expr(arg, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            let (r, sr, sbr) = rewrite_expr(a, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
                static_byte_count += sbr;
            let (r, sr, sbr) = rewrite_expr(b, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
                static_byte_count += sbr;
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x)
        | Expr::Await(x) | Expr::RawDeref(x, _, _) | Expr::RawAddr(x, _) => {
            let (r, sr, sbr) = rewrite_expr(x, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
                static_byte_count += sbr;
        }
        Expr::Field(recv, _) => {
            let (r, sr, sbr) = rewrite_expr(recv, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
                static_byte_count += sbr;
        }
        Expr::Cond(c, a, b) => {
            let (r, sr, sbr) = rewrite_expr(c, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
                static_byte_count += sbr;
            let (r, sr, sbr) = rewrite_expr(a, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
                static_byte_count += sbr;
            let (r, sr, sbr) = rewrite_expr(b, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
                static_byte_count += sbr;
        }
        Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for arg in args.iter_mut() {
                let (r, sr, sbr) = rewrite_expr(arg, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Expr::ArrayInit { items, .. } => {
            for item in items.iter_mut() {
                let (r, sr, sbr) = rewrite_expr(item, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for arg in ctor_args.iter_mut() {
                let (r, sr, sbr) = rewrite_expr(arg, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
            for item in items.iter_mut() {
                let (r, sr, sbr) = rewrite_expr(item, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for arg in ctor_args.iter_mut() {
                let (r, sr, sbr) = rewrite_expr(arg, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
            for (_, v) in fields.iter_mut() {
                let (r, sr, sbr) = rewrite_expr(v, maps, local_types, field_type_by_name, declaring_tdi, tokens);
                count += r;
                static_count += sr;
                static_byte_count += sbr;
            }
        }
        Expr::Lambda { body, .. } => {
            let (r, sr, sbr) = rewrite_expr(body, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
                static_byte_count += sbr;
        }
        Expr::Index { collection, index } => {
            let (r, sr, sbr) = rewrite_expr(collection, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
            let (r, sr, sbr) = rewrite_expr(index, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
        }
        Expr::NullConditional { receiver, access } => {
            let (r, sr, sbr) = rewrite_expr(receiver, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
            let (r, sr, sbr) = rewrite_expr(access, maps, local_types, field_type_by_name, declaring_tdi, tokens);
            count += r;
            static_count += sr;
            static_byte_count += sbr;
        }
        _ => {}
    }

    // A METADATA USAGE load: `*(nint*)(slot)` (or through the GOT) is the
    // value the slot names, not a raw pointer read.
    //
    // The GOT path used to stop at `void*.ptr_0x<addend>` for every slot kind
    // it did not model, and the outer dereference was then folded back to that
    // accessor (offset-0 simplification below). On this corpus that lost
    // 9,713 STRING LITERALS (kind 5) and 3,325 type references (kind 1/2) to
    // raw addresses. Resolve the VALUE here, where the slot is still visible:
    // a string literal is `"…"`, a type slot is `typeof(T)` — including
    // `typeof(T[])` for the SzArray slots `il2cpp_array_new_specific` takes.
    if let Expr::RawDeref(base, off, _) = expr {
        if let Some(e) = usage_slot_value_for_base(base, *off, maps) {
            *expr = e;
            return (count + 1, static_count, static_byte_count);
        }
    }
    // `(klass_of_array).element_class` is the ELEMENT type. The class-pointer
    // value resolves to `typeof(T[])` (above or via the GOT branch); reading
    // `element_class` off it is `typeof(T)`.
    if let Expr::Field(base, name) = expr {
        if name == "element_class" {
            if let Some(t) = usage_element_type(base, maps) {
                *expr = Expr::TypeOf(t);
                return (count + 1, static_count, static_byte_count);
            }
        }
    }

    // Now try to resolve the current expression as a typed RawDeref.
    if let Expr::RawDeref(base, offset, width) = expr {
        // TOKEN PROVENANCE. A variable whose definitions are all the same
        // KLASS token holds a class pointer, and one whose definitions are all
        // the same `static_fields` load holds a static-fields block. Neither
        // is an object: running the INSTANCE layouts on them named unrelated
        // fields at colliding offsets (`sf0.GameControllerPrefabNames` for the
        // static `TRWorldInfo.RegionMgr`, both at 0x38). The block case is
        // resolved here FIRST, before the collection/instance layouts, for the
        // same collision reason (`List<T>._items` also lives at +16).
        let (tok_root, tok_folded) = strip_const_addend(base);
        let tok_name = match tok_root {
            Expr::Var(n) => Some(n.as_str()),
            _ => None,
        };
        let block_owner = tok_name.and_then(|n| tokens.block.get(n)).cloned();
        let klass_token_var = tok_name.is_some_and(|n| tokens.klass.contains_key(n));
        if let Some(owner) = &block_owner {
            let off = tok_folded.wrapping_add(*offset);
            if let Some((field_name, type_name)) = maps.static_field_offsets.get(&(owner.clone(), off))
            {
                *expr = Expr::StaticField(type_name.clone(), field_name.clone());
                static_count += 1;
            }
        } else if klass_token_var {
            // Only Il2CppClass STRUCTURE offsets name anything on a klass
            // token; its managed instance layout describes a different object.
            let total = tok_folded.wrapping_add(*offset);
            if let Some(field_name) = try_resolve_il2cpp_class_field(total) {
                *expr = Expr::Field(Box::new(tok_root.clone()), field_name);
                count += 1;
            }
        }
        // First, try a known generic-collection internals layout. List<T>'s
        // metadata layout is absent from the generic type definition, so a
        // plain `try_resolve_field` lookup fails and every inlined
        // `List<T>.Add`/indexer keeps `*(nint*)(list +16)` and
        // `*(int*)(list +24/+28)` (0x179C950: 272 nint; 0x1772808: 41).
        else if let Some(hit) = try_resolve_collection_internal(
            base,
            *offset,
            *width,
            maps,
            local_types,
            field_type_by_name,
            tokens,
        ) {
            *expr = Expr::Field(Box::new(hit.receiver), hit.field_name);
            count += 1;
        }
        // A klass token (`typeof(T)`, or the dereferenced `klass(T)` accessor)
        // addressed at an offset is a STATIC field slot: `typeof(NGUIText)+56`
        // is `NGUIText.<field>`, not a field of System.Type. The existing
        // static resolver only handled the `.static_fields` shape.
        else if let Some((field_name, type_name)) =
            try_resolve_static_field_via_typed_base(base, *offset, *width, maps)
        {
            *expr = Expr::StaticField(type_name, field_name);
            static_count += 1;
        }
        // Then, try instance field resolution (typed base, including chains).
        else if let Some(hit) =
            try_resolve_field(base, *offset, *width, maps, local_types, field_type_by_name, tokens)
        {
            // The receiver is the ROOT of the address expression, not the
            // address itself: `*(nint*)(obj + 24)` is `obj.Field`, never
            // `(obj + 24).Field` — the latter does not even compile as C#,
            // and naming it off the interior pointer would be a different
            // object. `try_resolve_field` returns the stripped receiver.
            *expr = Expr::Field(Box::new(hit.receiver), hit.field_name);
            count += 1;
        }
        // Then, try static field resolution (Int base with known address).
        else if let Some((field_name, type_name)) = try_resolve_static_field(base, *offset, maps, declaring_tdi) {
            if std::env::var("ILR_FADBG").is_ok() {
                eprintln!("[fa-static] base={base:?} off={offset} → {type_name}.{field_name}");
            }
            *expr = Expr::StaticField(type_name, field_name);
            static_count += 1;
        }
        // Next, try Il2CppClass static_fields access:
        // RawDeref(Field(Int(klass_addr), "static_fields"), offset)
        // → StaticField(type_name, field_name)
        else if let Some((field_name, type_name)) = try_resolve_static_field_via_klass(base, *offset, maps, local_types, tokens) {
            if std::env::var("ILR_FADBG").is_ok() {
                eprintln!("[fa-klass] base={base:?} off={offset} → {type_name}.{field_name}");
            }
            *expr = Expr::StaticField(type_name, field_name);
            static_count += 1;
        }
        // Finally, try Il2CppClass structure field recognition.
        // When the base is a dereferenced pointer (likely an Il2CppClass*),
        // recognize well-known offsets like 184 = static_fields.
        else if let Some(field_name) = try_resolve_il2cpp_class_field(*offset) {
            let base_clone = base.as_ref().clone();
            *expr = Expr::Field(Box::new(base_clone), field_name);
            count += 1;
        }
        // Class initialization bitmap fallback: BYTE-width accesses to static
        // addresses that don't match any known field/static-field/klass mapping
        // are IL2CPP's class initialization bitmap checks. Each page-aligned
        // base (e.g. 0x31ac000) is a bitmap region; each byte offset is one
        // class's initialization flag.
        // `*(byte*)(0x31ac000 +2603)` → `ClassInit.init_31ac000_2603`
        //
        // Width matters: a wider read of an unmapped static address is NOT a
        // bitmap probe, and naming it `ClassInit.init_…` invents a field.
        else if *offset > 0 && *width == 1 {
            if let Expr::Int(addr) = base.as_ref() {
                let field_name = format!("init_{:x}_{}", *addr, *offset);
                *expr = Expr::StaticField("ClassInit".to_string(), field_name);
                static_byte_count += 1;
            }
        }
        // Simplification: RawDeref(Field(...), 0) → Field(...)
        // When we have a deref at offset 0 of a field access, and we couldn't
        // resolve it to a more specific field, simplify it to just the field
        // access. In C#, field access already implies loading the field value,
        // so the explicit deref is redundant.
        else if *offset == 0 {
            if matches!(base.as_ref(), Expr::Field(..) | Expr::StaticField(..)) {
                let base_clone = base.as_ref().clone();
                *expr = base_clone;
                count += 1;
            }
        }
    }

    // Refusal census (ILR_FAREFUSE=1): every RawDeref that survived all
    // resolvers, with the structural reason it could not be named. Kept
    // out-of-line so it does not extend the borrow of expr into the
    // resolution arms above (that would make every '*expr = ...' illegal).
    if std::env::var("ILR_FAREFUSE").is_ok() {
        if let Expr::RawDeref(base, offset, width) = expr {
            let (root, folded) = strip_const_addend(base);
            let rt = resolve_expr_type(root, maps, local_types, field_type_by_name, tokens);
            let reason = match rt {
                None => "notype".to_string(),
                Some(tdi) => {
                    let total = folded.wrapping_add(*offset);
                    let inl = layout_field_at(maps, tdi, total).is_some();
                    format!("typed(tdi={tdi},inlayout={inl})")
                }
            };
            eprintln!(
                "[farefuse] w={} off={} base={} {reason}",
                width, offset, shape(base)
            );
        }
    }

    (count, static_count, static_byte_count)
}

/// Compact structural shape of an expression, for the refusal census.
fn shape(e: &Expr) -> String {
    fn go(e: &Expr, d: usize) -> String {
        if d > 3 {
            return "…".into();
        }
        match e {
            Expr::Var(n) => format!("Var({n})"),
            Expr::This => "This".into(),
            Expr::Int(k) => format!("Int({k})"),
            Expr::Field(b, n) => format!("Field({},{n})", go(b, d + 1)),
            Expr::StaticField(t, n) => format!("Static({t}.{n})"),
            Expr::RawDeref(b, o, w) => format!("Deref({},+{o},w{w})", go(b, d + 1)),
            Expr::RawAddr(b, o) => format!("Addr({},+{o})", go(b, d + 1)),
            Expr::AddressOf(b) => format!("&({})", go(b, d + 1)),
            Expr::Cast(t, _) => format!("Cast({t})"),
            Expr::Call { name, .. } => format!("Call({name})"),
            Expr::RawCall(a, _, _) => format!("RawCall({a:x})"),
            Expr::Bin(op, a, b) => format!("Bin({op:?},{},{})", go(a, d + 1), go(b, d + 1)),
            Expr::Index { .. } => "Index".into(),
            Expr::TypeOf(t) => format!("typeof({t})"),
            _ => "other".into(),
        }
    }
    go(e, 0)
}

/// A resolved instance-field access.
struct FieldResolve {
    field_name: String,
    /// Root object expression the field belongs to. For a folded address
    /// (`obj + 24`) this is `obj`, NOT the address expression.
    receiver: Expr,
}

// ---------------------------------------------------------------------------
// Token provenance: klass pointers and static-fields blocks
// ---------------------------------------------------------------------------

/// Per-method type identity of variables that hold a KLASS pointer or the
/// static-fields block of one.
///
/// SSA destruction reuses one display name across disjoint live ranges, so a
/// name is typed only when every PROVABLE definition agrees — a definition
/// that proves nothing is ignored, a definition that proves a different type
/// vetoes the whole name. `klass` names hold an Il2CppClass* (`typeof(T)` in
/// this dialect, `T.klass(T)`, or a metadata-slot load of a type usage);
/// `block` names hold that type's `static_fields` block.
#[derive(Default)]
struct TokenMaps {
    klass: HashMap<String, String>,
    block: HashMap<String, String>,
}

fn plain_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn collect_assign_defs<'a>(list: &'a [Stmt], out: &mut HashMap<String, Vec<&'a Expr>>) {
    for s in list {
        if let Stmt::Assign(name, _, rhs) = s {
            if plain_ident(name) {
                out.entry(name.clone()).or_default().push(rhs);
            }
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_assign_defs(k, out);
        }
    }
}

fn full_type_name(maps: &ResolveMaps, tdi: i32) -> Option<String> {
    maps.tdi_names.get(&tdi).map(|(ns, n)| {
        if ns.is_empty() {
            n.clone()
        } else {
            format!("{ns}.{n}")
        }
    })
}

/// The managed type a metadata usage cell names, when that cell is a TYPE
/// usage. A cell that names a method/field/string is not a klass token.
///
/// A usage slot is often reached through the GOT: the cell holds the target's
/// VA, which the klass/usage maps key directly. Without that hop the token
/// stayed unclassified and the later `+184` was read through an INSTANCE
/// layout.
fn klass_slot_type_name(maps: &ResolveMaps, addr: u64) -> Option<String> {
    if let Some(mt) = maps.usage_type_by_addr.get(&addr) {
        return Some(mt.display_name());
    }
    if let Some(&tdi) = maps.klass_to_tdi.get(&addr) {
        if let Some(t) = full_type_name(maps, tdi) {
            return Some(t);
        }
    }
    if let Some(&target) = maps.got_addend_by_addr.get(&addr) {
        if let Some(mt) = maps.usage_type_by_addr.get(&target) {
            return Some(mt.display_name());
        }
        if let Some(&tdi) = maps.klass_to_tdi.get(&target) {
            if let Some(t) = full_type_name(maps, tdi) {
                return Some(t);
            }
        }
    }
    None
}

/// Type name when `e` denotes an Il2CppClass* token.
fn klass_token_type(e: &Expr, maps: &ResolveMaps, t: &TokenMaps) -> Option<String> {
    match e {
        Expr::TypeOf(ty) => Some(ty.clone()),
        Expr::StaticField(ty, acc)
            if acc.starts_with("klass(") || acc.starts_with("typeof(") =>
        {
            Some(ty.clone())
        }
        // A load from a constant table cell: the cell names a TYPE usage, so
        // the value loaded is that type's klass.
        Expr::Int(addr) => klass_slot_type_name(maps, *addr as u64),
        Expr::RawDeref(inner, off, _) => match inner.as_ref() {
            Expr::Int(a) => {
                klass_slot_type_name(maps, (*a as u64).wrapping_add(*off as u64))
            }
            // The lazy-init idiom reads the klass through the cell's pointer:
            // `*(*(cell))` is the same token as `*(cell)`.
            Expr::RawDeref(under, uoff, _) if *off == 0 => match under.as_ref() {
                Expr::Int(a) => {
                    klass_slot_type_name(maps, (*a as u64).wrapping_add(*uoff as u64))
                }
                _ => None,
            },
            _ => None,
        },
        Expr::Var(n) => t.klass.get(n).cloned(),
        _ => None,
    }
}

/// Owning type name when `e` denotes the STATIC-FIELDS BLOCK of a type.
fn block_token_type(e: &Expr, maps: &ResolveMaps, t: &TokenMaps) -> Option<String> {
    match e {
        Expr::Field(inner, name) if name == "static_fields" => {
            klass_token_type(inner, maps, t)
        }
        Expr::RawDeref(inner, off, _)
            if try_resolve_il2cpp_class_field(*off).is_some_and(|f| f == "static_fields") =>
        {
            klass_token_type(inner, maps, t)
        }
        Expr::Var(n) => t.block.get(n).cloned(),
        _ => None,
    }
}

/// Build the token maps by a small fixpoint over assignment definitions.
///
/// A definition may become provable only after another name is typed (a copy,
/// or `block = *(klass_var + 184)`), so the scan repeats until nothing new is
/// added. Each name is committed at most once and never changes its story:
/// conflicting provable definitions leave it untyped.
fn build_token_maps(stmts: &[Stmt], maps: &ResolveMaps) -> TokenMaps {
    let mut defs: HashMap<String, Vec<&Expr>> = HashMap::new();
    collect_assign_defs(stmts, &mut defs);
    let mut t = TokenMaps::default();
    for _round in 0..4 {
        let mut changed = false;
        for (name, rhss) in &defs {
            if t.klass.contains_key(name) || t.block.contains_key(name) {
                continue;
            }
            let mut klass: Option<String> = None;
            let mut block: Option<String> = None;
            let mut conflict = false;
            for rhs in rhss {
                if let Some(ty) = klass_token_type(rhs, maps, &t) {
                    match &klass {
                        None => klass = Some(ty),
                        Some(prev) if *prev == ty => {}
                        Some(_) => {
                            conflict = true;
                            break;
                        }
                    }
                }
                if let Some(ty) = block_token_type(rhs, maps, &t) {
                    match &block {
                        None => block = Some(ty),
                        Some(prev) if *prev == ty => {}
                        Some(_) => {
                            conflict = true;
                            break;
                        }
                    }
                }
            }
            if conflict || (klass.is_some() && block.is_some()) {
                continue;
            }
            if let Some(ty) = klass {
                t.klass.insert(name.clone(), ty);
                changed = true;
            } else if let Some(ty) = block {
                t.block.insert(name.clone(), ty);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if std::env::var("ILR_KLASSDBG").is_ok() {
        let mut rows: Vec<_> = t
            .klass
            .iter()
            .map(|(n, ty)| format!("{n}=klass({ty})"))
            .chain(t.block.iter().map(|(n, ty)| format!("{n}=block({ty})")))
            .collect();
        rows.sort();
        eprintln!("[tokens] {}", rows.join(" "));
    }
    t
}

/// Resolve the internals of a known generic collection.
///
/// `List<T>`'s fields live in the generic type DEFINITION's metadata, but the
/// definition carries no usable instance offsets (`_items`/`_size`/`_version`
/// come from the runtime instantiation). The AArch64/IL2CPP layout is fixed
/// and already documented in HANDOFF: +16 `_items`, +24 `_size` (Count),
/// +28 `_version`. Naming them removes the raw derefs left by inlined
/// `List<T>.Add` / indexer bodies, even where the element type is generic.
///
/// Width-gated: +16 is a pointer (width 0/8); +24/+28 are 4-byte ints. A
/// different width at those offsets is a different object and is refused.
fn try_resolve_collection_internal(
    base: &Expr,
    offset: i64,
    width: u8,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    field_type_by_name: &HashMap<(i32, String), i32>,
    tokens: &TokenMaps,
) -> Option<FieldResolve> {
    let tdi = resolve_expr_type(base, maps, local_types, field_type_by_name, tokens)?;
    let (ns, name) = maps.tdi_names.get(&tdi)?;
    let is_list = ns == "System.Collections.Generic"
        && (name == "List" || name.starts_with("List`"));
    if !is_list {
        return None;
    }
    let field_name = match (offset, width) {
        (16, 0 | 8) => "_items",
        // Width 0 is "pointer-width / not stated", not proof of an 8-byte
        // access; the List layout still proves these two fields are 4 bytes.
        (24, 0 | 4) => "Count",
        (28, 0 | 4) => "_version",
        _ => return None,
    };
    Some(FieldResolve {
        field_name: field_name.to_string(),
        receiver: base.clone(),
    })
}

/// Try to resolve a `RawDeref(base, offset)` to a named field.
///
/// Returns `Some(..)` if:
/// 1. The base denotes an instance of a known type: a `Var` with a known type
///    in `local_types`, a field access whose type is known, a cast, or a
///    **constant offset folded into the expression** (`obj + 24`, any nesting
///    of `+`/`-` with integer literals) — the renderer inlines a `base + off`
///    IR instruction exactly this way, so without stripping the constant the
///    field offset never reached this lookup and every such site stayed a raw
///    dereference even when both the receiver type and the offset were known.
/// 2. The `(tdi, offset)` maps to a field in `maps.field_layouts`.
///
/// Also handles chained field access: if the base is itself a `Field(recv, field_name)`,
/// we look up the field's type via `field_type_by_name` to continue the chain.
/// This enables multi-level pointer chain resolution:
/// `*(nint*)(*(nint*)(obj +16) +8)` → `obj.Inner.Count` (2 levels),
/// and arbitrary depth beyond that.
/// The type itself followed by its ancestors. Metadata layouts are often
/// stored per DECLARING type, so an inherited field is only visible through
/// the parent chain (store_sugar has always used the same rule).
fn parent_chain_tdis(maps: &ResolveMaps, tdi: i32) -> Vec<i32> {
    let mut chain = maps.type_parent_chains.get(&tdi).cloned().unwrap_or_default();
    if chain.is_empty() {
        chain.push(tdi);
    } else if !chain.contains(&tdi) {
        chain.push(tdi);
    }
    chain
}

/// `(field_name, owner_tdi)` at `off` in the type or one of its ancestors.
fn layout_field_at(maps: &ResolveMaps, tdi: i32, off: i64) -> Option<(&str, i32)> {
    for p in parent_chain_tdis(maps, tdi) {
        if let Some((name, _)) = maps.field_layouts.get(&(p, off)) {
            return Some((name.as_str(), p));
        }
    }
    None
}

/// Resolve an access that lands INSIDE an embedded VALUE-TYPE field.
///
/// `get_GeoID()` reads `*(nint*)(this +28)` because `HEU_GeoInfo _geoInfo`
/// sits at 0x14 and its `id` member at +8: the metadata declares only the
/// struct field's start offset, so the exact lookup refuses. Walking the
/// declared fields solves it at any depth: find a value-type field whose
/// start precedes the access, recurse into that type with the remainder, and
/// join the names (`_geoInfo.id`).
///
/// Refusals, in order of importance:
///  * only VALUE-type fields are entered — a class field's storage is a
///    pointer, so the remainder could never be inside the pointee;
///  * the remainder must resolve EXACTLY (a mid-field access is not a field);
///  * two candidates naming different paths are ambiguous and refused;
///  * a leaf whose declared width disagrees with the access width is refused
///    (a narrower store would be writing a sub-field, not this field).
/// The ROOT's exact-offset rule stays as it was: this only adds the descent.
pub(crate) fn layout_path_at(maps: &ResolveMaps, tdi: i32, off: i64, width: u8) -> Option<String> {
    if std::env::var("ILR_NODESC").is_ok() {
        return None;
    }
    let started = std::time::Instant::now();
    let mut visiting: Vec<(i32, i64)> = Vec::new();
    let out = layout_path_rec(maps, tdi, off, width, 0, &mut visiting);
    DESCENT_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    DESCENT_CALLS.fetch_add(1, Ordering::Relaxed);
    out
}

fn layout_path_rec(
    maps: &ResolveMaps,
    tdi: i32,
    off: i64,
    width: u8,
    depth: usize,
    visiting: &mut Vec<(i32, i64)>,
) -> Option<String> {
    if depth > 6 || visiting.contains(&(tdi, off)) {
        return None;
    }
    if let Some((name, owner)) = layout_field_at(maps, tdi, off) {
        if depth > 0 {
            if let Some(w) = maps.field_widths.get(&(owner, off)) {
                if width != 0 && *w != width {
                    return None;
                }
            }
        }
        return Some(name.to_string());
    }
    let mut hits: Vec<String> = Vec::new();
    for p in parent_chain_tdis(maps, tdi) {
        let Some(fields) = maps.tdi_fields.get(&p) else {
            continue;
        };
        for (fo, fname, ftdi) in fields {
            if *ftdi < 0 || !maps.value_type_tdis.contains(ftdi) {
                continue;
            }
            let rem = off - *fo;
            if rem <= 0 {
                continue;
            }
            visiting.push((tdi, off));
            if let Some(sub) = layout_path_rec(maps, *ftdi, rem, width, depth + 1, visiting) {
                hits.push(format!("{fname}.{sub}"));
            }
            visiting.pop();
        }
    }
    hits.sort();
    hits.dedup();
    if hits.len() == 1 {
        hits.pop()
    } else {
        None
    }
}

/// A float access INSIDE a `Vector2`/`Vector3` field: `this + (field+4)` is
/// `field.y`, `+8` is `field.z`. Metadata lists only the struct field's start
/// offset, so the exact lookup refuses and the access stayed a raw
/// dereference (`var sf2 = *(int*)(this +460)` where `PlayerVelocity` is a
/// Vector3 at 456 — TRRunnerPawn.VerifySpeed @0x1968EF8). Width is checked by
/// the caller; the component is only returned for UnityEngine vectors.
fn layout_field_component(
    maps: &ResolveMaps,
    tdi: i32,
    off: i64,
) -> Option<(String, i32)> {
    for k in 1..=2i64 {
        let base_off = off - k * 4;
        let Some((name, owner)) = layout_field_at(maps, tdi, base_off) else {
            continue;
        };
        let Some(&ftdi) = maps.field_types.get(&(owner, base_off)) else {
            continue;
        };
        let Some((ns, tname)) = maps.tdi_names.get(&ftdi) else {
            continue;
        };
        if ns != "UnityEngine" {
            continue;
        }
        let comp = match (tname.as_str(), k) {
            ("Vector3", 1) => "y",
            ("Vector3", 2) => "z",
            ("Vector2", 1) => "y",
            _ => continue,
        };
        return Some((format!("{name}.{comp}"), owner));
    }
    None
}

/// Declared type tdi of `field` on the type or an ancestor.
fn inherited_field_type(
    maps: &ResolveMaps,
    tdi: i32,
    field: &str,
    field_type_by_name: &HashMap<(i32, String), i32>,
) -> Option<i32> {
    for p in parent_chain_tdis(maps, tdi) {
        if let Some(&ft) = field_type_by_name.get(&(p, field.to_string())) {
            return Some(ft);
        }
    }
    None
}

fn try_resolve_field(
    base: &Expr,
    offset: i64,
    width: u8,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    field_type_by_name: &HashMap<(i32, String), i32>,
    tokens: &TokenMaps,
) -> Option<FieldResolve> {
    // ---- constant-folded address: `(obj + 24)` / `((obj + 8) + 16)` -------
    //
    // The offset lives in the EXPRESSION here, so the field lookup must use
    // the root's type plus the folded total, and the emitted receiver must be
    // the root (`obj`), not the address.
    let (root, folded) = strip_const_addend(base);
    let root_is_static_fields =
        matches!(root, Expr::Field(_, name) if name == "static_fields");
    if !root_is_static_fields {
        if let Some(root_tdi) = resolve_expr_type(root, maps, local_types, field_type_by_name, tokens) {
            let total = folded.wrapping_add(offset);
            if let Some((name, _decl)) = layout_field_at(maps, root_tdi, total) {
                return Some(FieldResolve {
                    field_name: name.to_string(),
                    receiver: root.clone(),
                });
            }
            // Known Il2CppClass structure offsets (176/184 static_fields,
            // 224 initialized, …) are never embedded struct members. A local
            // that actually holds a KLASS can carry an instance type from the
            // type map, and descending its managed layout at one of these
            // offsets fabricated `typeof(T).DirectionOnTrack.y` for the
            // static field read `*(klass+184)+16`. The class-field resolver
            // below owns these offsets, as it did before the descent existed.
            if !is_klass_structure_offset(total, width) {
                if let Some(path) = layout_path_at(maps, root_tdi, total, width) {
                    return Some(FieldResolve {
                        field_name: path,
                        receiver: root.clone(),
                    });
                }
            }
            if width == 4 {
                if let Some((name, _decl)) = layout_field_component(maps, root_tdi, total) {
                    return Some(FieldResolve {
                        field_name: name,
                        receiver: root.clone(),
                    });
                }
            }
        }
    }

    // ---- legacy path: the base itself is the object ----------------------
    //
    // `.static_fields` is a pseudo-field: its offsets index the declaring
    // type's STATIC layout, not its instance layout. Resolving it through
    // `field_layouts` named an instance field that happens to share the
    // offset (0x198AFB8: static +16 rendered as the instance `m_CachedPtr`)
    // and prevented the correct static resolver from running. Leave the whole
    // static-fields shape to `try_resolve_static_field_via_klass`.
    if !root_is_static_fields && std::ptr::eq(root, base) {
        let base_tdi = resolve_expr_type(base, maps, local_types, field_type_by_name, tokens)?;
        if let Some((name, _decl)) = layout_field_at(maps, base_tdi, offset) {
            return Some(FieldResolve {
                field_name: name.to_string(),
                receiver: base.clone(),
            });
        }
        if !is_klass_structure_offset(offset, width) {
            if let Some(path) = layout_path_at(maps, base_tdi, offset, width) {
                return Some(FieldResolve {
                    field_name: path,
                    receiver: base.clone(),
                });
            }
        }
        if width == 4 {
            if let Some((name, _decl)) = layout_field_component(maps, base_tdi, offset) {
                return Some(FieldResolve {
                    field_name: name,
                    receiver: base.clone(),
                });
            }
        }
    }
    None
}

/// Peel nested constant additions/subtractions off an address expression.
///
/// `((obj + 8) + 16) - 4` → `(obj, 20)`. Only integer literals are folded; a
/// computed index stays in the returned root, where `resolve_expr_type`
/// refuses to type it (an indexed address is not an instance).
fn strip_const_addend(expr: &Expr) -> (&Expr, i64) {
    let mut cur = expr;
    let mut off: i64 = 0;
    loop {
        match cur {
            Expr::Bin(BinOp::Add, a, b) => {
                if let Expr::Int(k) = b.as_ref() {
                    off = off.wrapping_add(*k);
                    cur = a;
                } else if let Expr::Int(k) = a.as_ref() {
                    off = off.wrapping_add(*k);
                    cur = b;
                } else {
                    break;
                }
            }
            Expr::Bin(BinOp::Sub, a, b) => {
                if let Expr::Int(k) = b.as_ref() {
                    off = off.wrapping_sub(*k);
                    cur = a;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    (cur, off)
}

/// Determine the TypeDef index (tdi) of an expression, if known.
///
/// This is the key enabler for pointer chain resolution: given an expression
/// like `Field(Var("obj"), "Inner")`, we can determine that its type is the
/// type of the "Inner" field on "obj"'s type, by looking up the reverse map.
///
/// Handles:
/// - `Var(name)`: look up in `local_types`
/// - `This`: look up "this" in `local_types`
/// - `Int(addr)`: look up in `klass_to_tdi`
/// - `Field(recv, field_name)`: recursively get recv's type, then look up
///   the field's type via `field_type_by_name`
/// - `Field(recv, "static_fields")`: returns recv's type (static_fields is a
///   pseudo-field whose layout matches the declaring type's static fields)
/// - `StaticField(type, "klass(type)")`: returns the tdi of that type (the
///   klass static field holds an Il2CppClass* of that type)
/// - `Cast(type_name, _)`: returns the tdi of the cast target type
fn resolve_expr_type(
    expr: &Expr,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    field_type_by_name: &HashMap<(i32, String), i32>,
    tokens: &TokenMaps,
) -> Option<i32> {
    match expr {
        Expr::Var(name) => {
            // A KLASS token or a static-fields block is not an instance: its
            // offsets index the Il2CppClass structure / the type's static
            // layout, never an object body. Answering with the declared type
            // here is what named `TRWorldInfo`'s static block offset 0x38 as
            // `WorldInfo.GameControllerPrefabNames` (an instance field that
            // happens to share the offset).
            if tokens.klass.contains_key(name) || tokens.block.contains_key(name) {
                return None;
            }
            local_types.get(name).copied()
        }
        Expr::This => local_types.get("this").copied(),
        Expr::Int(addr) => maps.klass_to_tdi.get(&(*addr as u64)).copied(),
        Expr::Field(recv, field_name) => {
            // "static_fields" is a pseudo-field: its layout is the static
            // fields structure of the receiver's type, so the "type" of
            // static_fields is the type itself (for offset resolution).
            if field_name == "static_fields" {
                return resolve_expr_type(recv, maps, local_types, field_type_by_name, tokens);
            }
            // Recursively determine the receiver's type.
            let recv_tdi = resolve_expr_type(recv, maps, local_types, field_type_by_name, tokens)?;
            // Look up the field's type via the reverse map.
            inherited_field_type(maps, recv_tdi, field_name, field_type_by_name)
        }
        // `StaticField(type_name, "klass(type_name)")` holds an Il2CppClass* of
        // that type. A plain static field carries its declared type, which
        // lets a chain continue through the singleton object
        // (`TRWorldInfo.RunnerGame +216`).
        Expr::StaticField(type_name, accessor) => {
            if accessor.starts_with("klass(") {
                tdi_for_type_name(type_name, maps)
            } else {
                maps.static_field_types
                    .get(&(type_name.clone(), accessor.clone()))
                    .copied()
            }
        }
        // Cast(type_name, _) — the result type is the cast target.
        Expr::Cast(type_name, _) => {
            tdi_for_type_name(type_name, maps)
        }
        // Calls whose declared return type is a class. `method_return_tdi`
        // only keeps names whose return type is unambiguous binary-wide, so an
        // overloaded name never resolves here. This is what lets a call result
        // used directly as a field base (`GetThing().field`) be named; the
        // table-based propagation already types the MATERIALISED temp, so
        // without this the same fact is lost at exactly the inlined sites.
        Expr::Call { name, .. } => maps.method_return_tdi.get(name).copied(),
        Expr::NewObj(type_name, _) => tdi_for_type_name(type_name, maps),
        Expr::RawCall(addr, _, _) => maps.methods_by_rva.get(addr).and_then(|m| {
            match &m.sig.ret {
                ManagedType::Class(nt) if nt.type_def_index >= 0 => Some(nt.type_def_index),
                _ => None,
            }
        }),
        _ => None,
    }
}

/// Look up a type's tdi by its fully-qualified name.
/// Uses the pre-computed reverse map to avoid O(types) scan.
fn tdi_for_type_name(type_name: &str, maps: &ResolveMaps) -> Option<i32> {
    maps.tdi_names_reverse.get(type_name).copied()
}

/// Infer the TypeDef of locals that hold a KLASS POINTER (`T.klass(T)`, the
/// dereferenced form, or a raw klass address).
///
/// The renderer materialises such a klass into a temp whenever the chain has
/// more than one def (a class-init guard re-assigns it), and then the
/// `.static_fields` access on that temp can no longer be resolved: the
/// resolver sees a `Var`, not the klass expression. With the type recovered
/// here, `try_resolve_static_field_via_klass` bridges
/// `v.static_fields + off` back to `T.field` and the whole klass bookkeeping
/// disappears from the text.
pub(crate) fn infer_klass_var_types(
    stmts: &[Stmt],
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
) -> HashMap<String, i32> {
    let mut defs: HashMap<String, Vec<&Expr>> = HashMap::new();
    fn walk<'a>(list: &'a [Stmt], out: &mut HashMap<String, Vec<&'a Expr>>) {
        for s in list {
            if let Stmt::Assign(name, None, rhs) = s {
                if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    out.entry(name.clone()).or_default().push(rhs);
                }
            }
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            for k in kids {
                walk(k, out);
            }
        }
    }
    walk(stmts, &mut defs);

    let mut out: HashMap<String, i32> = HashMap::new();
    let dbg = std::env::var("ILR_KLASSDBG").is_ok();
    // Fixpoint: a copy `v = w` can only be typed after `w` is. The chains are
    // short (klass -> guard copy -> static_fields receiver), so a few rounds
    // settle; the round cap keeps a pathological cycle from spinning.
    let mut changed = true;
    let mut rounds = 0;
    while changed && rounds < 8 {
        changed = false;
        rounds += 1;
        for (name, rhss) in &defs {
            if rhss.is_empty() || local_types.contains_key(name) || out.contains_key(name) {
                continue;
            }
            if dbg {
                let res: Vec<_> = rhss
                    .iter()
                    .map(|e| klass_pointer_type_name(e, maps, local_types, &out))
                    .collect();
                eprintln!("[klassdbg] {name}: res={res:?}");
            }
            // Every def must name the SAME type; a temp that is assigned two
            // different klass pointers has no single type to attribute.
            let mut agreed: Option<String> = None;
            let mut consistent = true;
            for rhs in rhss {
                let Some(t) = klass_pointer_type_name(rhs, maps, local_types, &out) else {
                    consistent = false;
                    break;
                };
                match &agreed {
                    None => agreed = Some(t),
                    Some(prev) if *prev == t => {}
                    _ => {
                        consistent = false;
                        break;
                    }
                }
            }
            if !consistent {
                continue;
            }
            if let Some(t) = agreed {
                if let Some(&tdi) = maps.tdi_names_reverse.get(&t) {
                    out.insert(name.clone(), tdi);
                    changed = true;
                }
            }
        }
    }
    let mut out = out;
    // Defs are only needed for typing; do not keep borrows alive.
    out.shrink_to_fit();
    out
}

/// Map the IR's post-SSA type propagation onto the variable-display-name map
/// this pass consumes.
///
/// [`super::super::ir::infer_ir_types`] proves a type for every SSA version it
/// can (copy chains, equal-only merges, field loads, store→load cells and
/// `base + off` field addresses). The annotation pass works on the rendered
/// statement tree, whose `Var`s are `LocalTable` display names, so the
/// version-keyed result is translated exactly the way
/// `csharp::build_local_tdi_map` names a local:
///
/// * `is_this` → `"this"`;
/// * `is_param = Some(p)` → `table.param_names[p]` (fallback `arg{p}`);
/// * otherwise → `v{table index}`.
///
/// Only `Tdi` entries are emitted. `TdiOff` deliberately has no equivalent in
/// this map: an address is not an instance, and telling the statement resolver
/// that `v` (which holds `obj + 24`) is an instance of the field's type would
/// make the NEXT access name a field of the wrong owner. Those addresses are
/// consumed inside the IR pass (load typing) and, in the statement tree, by
/// the folded-offset path of [`try_resolve_field`].
///
/// The caller merges this into the pre-annotation map with `or_insert`, so
/// types already proved by the `LocalTable` always win; this only fills holes.
/// Per-VERSION type provenance from the IR fixpoint.
///
/// The name-level map (`infer_ir_type_map`) drops a display name whose
/// versions disagree — which is exactly the multi-def stack-cell case. The
/// per-version types still exist; this exposes them so the renderer can give
/// disagreeing versions distinct names (`MethodCtx::split_versions_by_type`),
/// the variable-recovery-before-typing order Ghidra/TIE/Retypd use.
pub fn infer_ir_version_map(
    ir: &super::super::RecoveredIr,
    maps: &ResolveMaps,
) -> HashMap<u32, i32> {
    let mut seeds: HashMap<u32, IrType> = HashMap::new();
    for (&version, &li) in &ir.table.index_of {
        let Some(info) = ir.table.infos.get(li) else {
            continue;
        };
        if let Some(ty) = lattice_type_of_managed(&info.ty) {
            seeds.insert(version, ty);
        }
    }
    let info = super::super::ir::infer_ir_types_seeded(&ir.insns, maps, &seeds);
    let mut out = HashMap::new();
    for (&version, &ty) in &info.version_types {
        if let Some(tdi) = ty.tdi() {
            out.insert(version, tdi);
        }
    }
    out
}

pub fn infer_ir_type_map(
    ir: &super::super::RecoveredIr,
    maps: &ResolveMaps,
) -> HashMap<String, i32> {
    // Seed the fixpoint from the table's own metadata-derived types so `this`
    // and typed parameters can start a chain. Only Class/ByRef/Class-shaped
    // types carry a TypeDef index; primitives and Unknown stay unseeded.
    let mut seeds: HashMap<u32, IrType> = HashMap::new();
    for (&version, &li) in &ir.table.index_of {
        let Some(info) = ir.table.infos.get(li) else {
            continue;
        };
        if let Some(ty) = lattice_type_of_managed(&info.ty) {
            seeds.insert(version, ty);
        }
    }
    let info = super::super::ir::infer_ir_types_seeded(&ir.insns, maps, &seeds);

    let mut out: HashMap<String, i32> = HashMap::new();
    for (&version, &ty) in &info.version_types {
        let Some(tdi) = ty.tdi() else {
            if std::env::var("ILR_FADBG").is_ok() {
                eprintln!("[irdbg] version {version} = {ty:?} (no tdi)");
            }
            continue;
        };
        let Some(&li) = ir.table.index_of.get(&version) else {
            continue;
        };
        let Some(local) = ir.table.infos.get(li) else {
            continue;
        };
        // Mirrors `csharp::build_local_tdi_map` / `MethodCtx::var_name`.
        let name = if local.is_this {
            "this".to_string()
        } else if let Some(pi) = local.is_param {
            ir.table
                .param_names
                .get(pi)
                .cloned()
                .unwrap_or_else(|| format!("arg{pi}"))
        } else {
            format!("v{li}")
        };
        // Equal-only at the name level too: two versions sharing a display
        // name (only possible for `this`/a parameter) must agree.
        match out.get(&name) {
            None => {
                out.insert(name, tdi);
            }
            Some(&prev) if prev == tdi => {}
            Some(_) => {
                out.remove(&name);
            }
        }
    }
    if std::env::var("ILR_FADBG").is_ok() {
        let mut rows: Vec<_> = out.iter().collect();
        rows.sort();
        for (name, tdi) in rows {
            let tyname = maps
                .tdi_names
                .get(tdi)
                .map(|(ns, n)| if ns.is_empty() { n.clone() } else { format!("{ns}.{n}") })
                .unwrap_or_else(|| "?".into());
            let mut fields = Vec::new();
            for off in [16i64, 24, 32, 48, 56, 108] {
                if let Some((f, _)) = maps.field_layouts.get(&(*tdi, off)) {
                    fields.push(format!("{off}=>{f}"));
                }
            }
            eprintln!("[irdbg] name {name} tdi={tdi} ({tyname}) fields: {}", fields.join(", "));
        }
    }
    out
}

/// `ManagedType` → the IR lattice's instance/address type, when it carries a
/// TypeDef index. `ByRef`/`Pointer` mean "address of the type", which the
/// lattice's `Tdi(t)` convention (base of `[v + off]`) already describes.
fn lattice_type_of_managed(ty: &ManagedType) -> Option<IrType> {
    match ty {
        ManagedType::Class(nt) if nt.type_def_index >= 0 => {
            Some(IrType::Tdi(nt.type_def_index))
        }
        ManagedType::ByRef(inner) | ManagedType::Pointer(inner) => {
            lattice_type_of_managed(inner)
        }
        _ => None,
    }
}

/// Type name of an expression that is a KLASS pointer, not an instance.
fn klass_pointer_type_name(
    e: &Expr,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    inferred: &HashMap<String, i32>,
) -> Option<String> {
    match e {
        Expr::Var(n) => {
            let tdi = local_types.get(n).or_else(|| inferred.get(n))?;
            let (ns, name) = maps.tdi_names.get(tdi)?;
            Some(if ns.is_empty() {
                name.clone()
            } else {
                format!("{}.{}", ns, name)
            })
        }
        Expr::StaticField(t, a) if a.starts_with("klass(") => a
            .strip_prefix("klass(")
            .and_then(|s| s.strip_suffix(')'))
            .map(str::to_string)
            .or_else(|| Some(t.clone())),
        Expr::Field(_, a) if a.starts_with("klass(") => a
            .strip_prefix("klass(")
            .and_then(|s| s.strip_suffix(')'))
            .map(str::to_string),
        Expr::TypeOf(t) => Some(t.clone()),
        Expr::StaticField(t, a) if a.starts_with("typeof(") => Some(t.clone()),
        // `*(nint*)(*(nint*)(0x2f6a000 + off))` — the metadata-usage slot at
        // `base + off` holds the Il2CppClass* of a type; the outer deref is
        // part of the klass accessor (cleanup renders the pair as typeof(T)).
        Expr::RawDeref(inner, off, _) => match inner.as_ref() {
            Expr::Int(base) => {
                let full = (*base as u64).wrapping_add(*off as u64);
                if std::env::var("ILR_KLASSDBG").is_ok() {
                    eprintln!(
                        "[klassdbg]   slot {full:#x}: klass={:?} got={:?} got_klass={:?} sfl={:?}",
                        maps.klass_to_tdi.get(&full),
                        maps.got_addend_by_addr.get(&full),
                        maps.got_addend_by_addr
                            .get(&full)
                            .and_then(|a| maps.klass_to_tdi.get(a)),
                        maps.static_field_layouts.get(&full),
                    );
                }
                let tdi = maps.klass_to_tdi.get(&full).copied().or_else(|| {
                    // Metadata-usage slots are often reached through the GOT:
                    // the slot holds the addend (the runtime klass VA), which
                    // `klass_to_tdi` keys directly.
                    maps.got_addend_by_addr
                        .get(&full)
                        .and_then(|a| maps.klass_to_tdi.get(a).copied())
                })?;
                let (ns, name) = maps.tdi_names.get(&tdi)?;
                Some(if ns.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", ns, name)
                })
            }
            _ => klass_pointer_type_name(inner, maps, local_types, inferred),
        },
        Expr::Int(addr) => {
            let tdi = maps.klass_to_tdi.get(&(*addr as u64))?;
            let (ns, name) = maps.tdi_names.get(tdi)?;
            Some(if ns.is_empty() {
                name.clone()
            } else {
                format!("{}.{}", ns, name)
            })
        }
        _ => None,
    }
}

/// The metadata-usage slot address carried by a `void*.ptr_0x…` accessor.
///
/// `try_resolve_static_field`'s GOT branch emits that accessor for a slot it
/// could not model, and the offset-0 simplification below folds a dereference
/// of it back to the accessor itself. The slot address in the name is the only
/// remaining evidence of WHAT the load reads, so the usage maps are consulted
/// through it.
fn ptr_accessor_slot(e: &Expr) -> Option<u64> {
    match e {
        Expr::StaticField(ty, name) if ty == "void*" => name
            .strip_prefix("ptr_0x")
            .and_then(|hex| u64::from_str_radix(hex, 16).ok()),
        _ => None,
    }
}

/// The source-level VALUE of a metadata usage slot.
fn usage_slot_value(slot: u64, maps: &ResolveMaps) -> Option<Expr> {
    if let Some(&idx) = maps.literal_index_by_addr.get(&slot) {
        if let Some(text) = maps.literals.get(&idx) {
            return Some(Expr::Str(text.clone()));
        }
    }
    if let Some(mt) = maps.usage_type_by_addr.get(&slot) {
        return Some(Expr::TypeOf(mt.display_name()));
    }
    // kind 6 -- Il2CppMetadataUsage::MethodRef. The slot holds the concrete
    // instantiation of a generic method (an Il2CppMethodInfo*), i.e. a METHOD
    // USED AS A VALUE. Rendering it as MethodInfo("Type.Method") is the same
    // spelling Operand::Method already gets (csharp.rs:3652 -> MethodGroup ->
    // csharp.rs:709), so a usage value and a method operand can never disagree
    // about how a method is written.
    //
    // Measured on OUT_R74: 3,159 of the 4,819 standalone ptr_0x occurrences
    // (214 distinct slots) are kind-6 cells, and every one of them fell through
    // this function to void*.ptr_0x...
    if let Some(&spec) = maps.method_spec_index_by_addr.get(&slot) {
        if let Some(m) = method_ref_of_spec(spec, maps) {
            return Some(Expr::MethodGroup(method_group_name(m)));
        }
    }
    None
}

/// Type.Method with the declaring type shortened exactly the way the emitter
/// shortens it (csharp.rs:2107), so a usage value and an Operand::Method render
/// identically for the same method.
fn method_group_name(m: &MethodRef) -> String {
    let short = m
        .declaring_type_name
        .rsplit('.')
        .next()
        .unwrap_or(&m.declaring_type_name);
    format!("{}.{}", short, m.name)
}

/// A MethodSpec index -> its instantiated ref.
///
/// specs_by_index holds only the specs the code-pointer table does NOT index
/// (see its doc comment); the rest live in specs_by_body_rva under the body
/// they were compiled to, so the index is searched there. Both maps name the
/// same instantiation, so either answer is exact -- and a spec in neither map
/// resolves to nothing rather than to a neighbour.
fn method_ref_of_spec(spec: u32, maps: &ResolveMaps) -> Option<&MethodRef> {
    if let Some(m) = maps.specs_by_index.get(&spec) {
        return Some(m);
    }
    maps.specs_by_body_rva
        .values()
        .flat_map(|v| v.iter())
        .find(|(idx, _)| *idx == spec)
        .map(|(_, m)| m)
}

/// Resolve `base` of a dereference to a usage slot and emit its value.
///
/// Accepted shapes:
///  * the `ptr_0x…` accessor the GOT branch already produced;
///  * a still-unresolved `RawDeref(Int(got), off)` (slot = GOT addend);
///  * a direct `Int(slot)` (no GOT in the way).
fn usage_slot_value_for_base(base: &Expr, off: i64, maps: &ResolveMaps) -> Option<Expr> {
    // A ptr_0x... accessor names a GOT target; an offset off it walks a table
    // of usage cells (slot + 8k). The sum is looked up EXACTLY, so this only
    // fires when the metadata really registers that address as a cell -- an
    // offset that lands anywhere else resolves to nothing, never to a
    // neighbour's value.
    if let Some(slot) = ptr_accessor_slot(base) {
        return usage_slot_value(slot.wrapping_add(off as u64), maps);
    }
    if off != 0 {
        return None;
    }
    let slot = match base {
        Expr::Int(addr) => *addr as u64,
        Expr::RawDeref(inner, off, _) => match inner.as_ref() {
            Expr::Int(got) => *maps
                .got_addend_by_addr
                .get(&(*got as u64).wrapping_add(*off as u64))?,
            _ => return None,
        },
        _ => return None,
    };
    usage_slot_value(slot, maps)
}

/// Element type of an array class-pointer expression.
fn usage_element_type(base: &Expr, maps: &ResolveMaps) -> Option<String> {
    let strip = |t: &str| {
        let elem = t.strip_suffix("[]")?;
        (!elem.contains('[')).then(|| elem.to_string())
    };
    match base {
        Expr::TypeOf(t) => strip(t),
        Expr::StaticField(_, a) if a.starts_with("klass(") => {
            strip(a.strip_prefix("klass(")?.strip_suffix(')')?)
        }
        _ => {
            let slot = ptr_accessor_slot(base)?;
            match maps.usage_type_by_addr.get(&slot)? {
                ManagedType::SzArray(e) => strip(&e.display_name()),
                _ => None,
            }
        }
    }
}

/// Try to resolve a `RawDeref(Int(base), offset)` to a named static field, metadata usage slot, or GOT target.
///
/// Returns `Some((field_name, type_name))` if:
/// 1. The base is an `Int(addr)` (a constant address)
/// 2. The computed address `addr + offset` maps to:
///    - A static field in `maps.static_field_layouts`
///    - A metadata usage slot (klass reference or string literal)
///    - A GOT entry (resolved via relocation addend)
///
/// This handles the pattern `*(nint*)(0x2f6a000 + 792)` where `0x2f6a000` is
/// the base of a static fields block, the metadata usage table, or the GOT.
///
/// `declaring_tdi` is the TypeDef index of the method's declaring type, used
/// for ptr_no_offset resolution context.
fn try_resolve_static_field(
    base: &Expr,
    offset: i64,
    maps: &ResolveMaps,
    declaring_tdi: i32,
) -> Option<(String, String)> {
    if let Expr::Int(addr) = base {
        let full_addr = (*addr as u64).wrapping_add(offset as u64);

        // First, try static field resolution
        if let Some(result) = maps.static_field_layouts.get(&full_addr) {
            return Some(result.clone());
        }

        // Then, try metadata usage slot resolution. The full type map covers
        // array/primitive/generic slots that have no TypeDef; `klass_to_tdi`
        // stays as the fallback for slots recorded without a managed type.
        if let Some(mt) = maps.usage_type_by_addr.get(&full_addr) {
            let type_name = mt.display_name();
            return Some((format!("klass({})", type_name), type_name));
        }
        if let Some(&tdi) = maps.klass_to_tdi.get(&full_addr) {
            if let Some((ns, name)) = maps.tdi_names.get(&tdi) {
                let type_name = if ns.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", ns, name)
                };
                return Some((format!("typeof({})", type_name), type_name));
            }
        }

        // Check if this is a string literal
        if let Some(&lit_idx) = maps.literal_index_by_addr.get(&full_addr) {
            if let Some(lit) = maps.literals.get(&lit_idx) {
                return Some((format!("\"{}\"", lit.replace('\\', "\\\\").replace('"', "\\\"")), "string".to_string()));
            }
        }

        // Finally, try GOT relocation resolution
        // If this address is a GOT slot, resolve to its runtime target
        if let Some(&addend) = maps.got_addend_by_addr.get(&full_addr) {
            // The addend is the runtime target address (VA)
            // Check if it's a known method
            if let Some((method_name, type_name)) = maps.method_name_by_va.get(&addend) {
                return Some((format!("{}.{}", type_name, method_name), type_name.clone()));
            }
            // Check if the addend names a type usage. Arrays and primitives
            // have no `klass_to_tdi` entry, and without this branch the
            // allocation token and every `typeof` argument loaded through the
            // GOT stayed `void*.ptr_0x…`.
            if let Some(mt) = maps.usage_type_by_addr.get(&addend) {
                let type_name = mt.display_name();
                return Some((format!("klass({})", type_name), type_name));
            }
            // Check if the addend is a known klass (type) pointer
            if let Some(&tdi) = maps.klass_to_tdi.get(&addend) {
                if let Some((ns, name)) = maps.tdi_names.get(&tdi) {
                    let type_name = if ns.is_empty() {
                        name.clone()
                    } else {
                        format!("{}.{}", ns, name)
                    };
                    return Some((format!("klass({})", type_name), type_name));
                }
            }
            // Check if the addend is a static field storage address
            if let Some(result) = maps.static_field_layouts.get(&addend) {
                return Some(result.clone());
            }
            // Unresolved GOT entry - record for ptr_no_offset resolution
            // Get the declaring type name for context
            let declaring_type_name = if declaring_tdi >= 0 {
                if let Some((ns, name)) = maps.tdi_names.get(&declaring_tdi) {
                    if ns.is_empty() {
                        name.clone()
                    } else {
                        format!("{}.{}", ns, name)
                    }
                } else {
                    format!("Type{}", declaring_tdi)
                }
            } else {
                "Unknown".to_string()
            };
            // Record the usage for later resolution
            super::ptr_resolution::record_usage(addend, &declaring_type_name);
            // Otherwise, just show the target address
            return Some((format!("ptr_0x{:x}", addend), "void*".to_string()));
        }
    }
    None
}

/// Recognize well-known Il2CppClass structure field offsets.
///
/// The Il2CppClass structure has fields at fixed offsets that vary by Unity/IL2CPP version:
/// - v24 (Unity 2017-2020): static_fields at offset 184
/// - v27+ (Unity 2020.2+): static_fields at offset 176
///
/// When we see a dereference like `*(klass_ptr + 184)` or `*(klass_ptr + 176)`,
/// we can recognize it as accessing `static_fields`. This makes the decompiled
/// code more readable even when we can't resolve the full type name.
///
/// We check both common offsets to support multiple Unity versions.
/// True for offsets that index the Il2CppClass structure rather than an
/// instance body. The class-field table covers the older layout; this corpus
/// (v29+) keeps the init flag at byte offset 224, which collides numerically
/// with a managed field component (`TRRunnerPawn.TintToColor` at 0xD4 puts
/// `Color.a` at 224). The flag is read as a byte/pointer, so only those
/// widths are claimed: a 4-byte read there is a real float component.
fn is_klass_structure_offset(off: i64, width: u8) -> bool {
    if try_resolve_il2cpp_class_field(off).is_some() {
        return true;
    }
    matches!(off, 224 | 232) && matches!(width, 0 | 1)
}

fn try_resolve_il2cpp_class_field(offset: i64) -> Option<String> {
    match offset {
        // Common across versions
        64 => Some("element_class".to_string()),
        72 => Some("castClass".to_string()),
        80 => Some("declaringType".to_string()),
        88 => Some("parent".to_string()),
        96 => Some("generic_class".to_string()),
        128 => Some("fields".to_string()),
        136 => Some("events".to_string()),
        144 => Some("properties".to_string()),
        152 => Some("methods".to_string()),
        160 => Some("nestedTypes".to_string()),
        168 => Some("implementedInterfaces".to_string()),
        // Version-specific: static_fields
        176 => Some("static_fields".to_string()),  // v27+ (Unity 2020.2+)
        184 => Some("static_fields".to_string()),  // v24 (Unity 2017-2020)
        // More version-specific offsets
        192 => Some("rgctx_data".to_string()),
        200 => Some("typeHierarchy".to_string()),
        // Class initialization flag (byte-width access)
        // This is the `initialized` field in Il2CppClass, accessed as *(byte*)(klass + offset)
        // Used in class-init guard patterns: if (*(byte*)(typeof(Type) +304) == 0) { init; }
        // The offset varies by Unity/IL2CPP version:
        //   v24 (Unity 2017-2020): offset 304
        //   v24.5-v27 (Unity 2020-2021): offset 309
        //   v29+ (Unity 2021+): offset 312
        304 => Some("initialized".to_string()),
        309 => Some("initialized".to_string()),
        312 => Some("initialized".to_string()),
        320 => Some("initialized".to_string()),
        _ => None,
    }
}

/// Try to resolve a `RawDeref(Field(base, "static_fields"), offset)` to a named static field.
///
/// This handles the pattern where the inner `RawDeref(Int(klass_addr), 184)` has already
/// been resolved to `Field(Int(klass_addr), "static_fields")` by `try_resolve_il2cpp_class_field`,
/// and now the outer `RawDeref` needs to resolve the offset within the static_fields structure.
///
/// Handles multiple base expression forms:
/// - `Field(StaticField(type_name, "klass(...)"), "static_fields")` — klass accessor
/// - `Field(RawDeref(StaticField(type_name, "klass(...)"), 0, 0), "static_fields")` — dereferenced klass
/// - `Field(Int(klass_addr), "static_fields")` — raw klass pointer address
/// - `Field(Var(name), "static_fields")` — variable holding klass pointer (via local_types)
///
/// Returns `Some((field_name, type_name))` if the type can be determined and the
/// (type_name, offset) maps to a static field in `static_field_offsets`.
/// Resolve a static field addressed through a KLASS token:
/// `*(int*)(typeof(NGUIText) +56)` is a slot in NGUIText's static-fields
/// structure, not an instance field of `System.Type`. The `.static_fields`
/// shape is handled by [`try_resolve_static_field_via_klass`]; this covers
/// the rendered `typeof(T) +off` / `klass(T) +off` form.
fn try_resolve_static_field_via_typed_base(
    base: &Expr,
    offset: i64,
    width: u8,
    maps: &ResolveMaps,
) -> Option<(String, String)> {
    let ty = match base {
        // Only a VALUE-form klass token is a static-field base. The
        // `klass(T)` accessor is the base of the `.static_fields` chain and
        // is handled by try_resolve_static_field_via_klass; treating it here
        // hijacked the singleton chain and produced
        // `TRWorldInfo.TRPersistentData.static_fields +16`.
        Expr::TypeOf(t) => t.clone(),
        Expr::StaticField(t, accessor) if accessor.starts_with("typeof(") => t.clone(),        // The renderer folds a klass load into an Int token before it prints
        // `typeof(T)`: `*(int*)(0x2f… +56)` is a static slot of the type whose
        // klass the address holds. Resolve the BASE address as a klass (or a
        // GOT slot pointing at one), never as a static-field address.
        Expr::Int(addr) => {
            let addr = *addr as u64;
            let tdi = maps.klass_to_tdi.get(&addr).copied().or_else(|| {
                maps.got_addend_by_addr
                    .get(&addr)
                    .and_then(|a| maps.klass_to_tdi.get(a).copied())
            })?;
            let (ns, name) = maps.tdi_names.get(&tdi)?;
            if ns.is_empty() {
                name.clone()
            } else {
                format!("{ns}.{name}")
            }
        }
        _ => return None,
    };
    if std::env::var("ILR_FADBG").is_ok() && ty.contains("Vector3") {
        eprintln!(
            "[fa31] ty={ty} off={offset} w={width} keys0={} keys4={} keys8={} n={}",
            maps.static_field_offsets.contains_key(&(ty.clone(), 0)),
            maps.static_field_offsets.contains_key(&(ty.clone(), 4)),
            maps.static_field_offsets.contains_key(&(ty.clone(), 8)),
            maps.static_field_offsets.len()
        );
    }
    // A 4-byte read INSIDE a static Vector2/Vector3 field: `typeof(Vector3)+8`
    // is `Vector3.zero.z` (the static block's first field is `zero`, size 12).
    // Without this the access renders as `*(int*)(typeof(Vector3) +8)` and the
    // scalar multiply looks like an int -> float miracle (VerifySpeed).
    if width == 4 {
        for k in 1..=2i64 {
            let base_off = offset - k * 4;
            let Some((name, _)) = maps.static_field_offsets.get(&(ty.clone(), base_off)) else {
                continue;
            };
            let Some(&ftdi) = maps.static_field_types.get(&(ty.clone(), name.clone())) else {
                continue;
            };
            let Some((ns, tname)) = maps.tdi_names.get(&ftdi) else {
                continue;
            };
            if ns != "UnityEngine" {
                continue;
            }
            let comp = match (tname.as_str(), k) {
                ("Vector3", 1) => "y",
                ("Vector3", 2) => "z",
                ("Vector2", 1) => "y",
                _ => continue,
            };
            return Some((format!("{name}.{comp}"), ty));
        }
    }
    // A klass-structure offset (176/184 static_fields, 224 typeHierarchy, …)
    // is not a static field even if a static layout happens to have a field
    // at that offset. Those shapes are handled by the dedicated resolvers.
    // (Checked AFTER the vector-component rule: `typeof(Vector3)+8` is
    // Vector3.zero.z, and offset 8 is also a klass field name.)
    if try_resolve_il2cpp_class_field(offset).is_some() {
        return None;
    }
    maps.static_field_offsets.get(&(ty, offset)).cloned()
}

fn try_resolve_static_field_via_klass(
    base: &Expr,
    offset: i64,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    tokens: &TokenMaps,
) -> Option<(String, String)> {
    // Check if base is Field(inner, "static_fields")
    if let Expr::Field(inner_base, field_name) = base {
        if field_name != "static_fields" {
            return None;
        }
        // Try to extract the type name from the inner base expression.
        let type_name = extract_type_from_static_base(inner_base, maps, local_types, tokens)?;
        // Look up the static field at this offset for this type.
        return maps.static_field_offsets.get(&(type_name, offset)).cloned();
    }
    None
}

/// Extract the declaring type name from the base expression of a `.static_fields` access.
///
/// Handles:
/// - `StaticField(type_name, "klass(...)")` — direct klass accessor
/// - `RawDeref(StaticField(type_name, "klass(...)"), 0, 0)` — dereferenced klass accessor
/// - `RawDeref(Var(name), 0, 0)` — dereferenced variable holding klass pointer
/// - `RawDeref(Field(recv, field_name), 0, 0)` — dereferenced field holding klass pointer
/// - `Int(klass_addr)` — raw klass pointer, resolved via klass_to_tdi + tdi_names
/// - `Var(name)` — variable, resolved via local_types + tdi_names
/// - `Field(recv, field_name)` — field access where the field type is known
/// - Any expression with a known type via `resolve_expr_type`
fn extract_type_from_static_base(
    expr: &Expr,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    tokens: &TokenMaps,
) -> Option<String> {
    // First, try specific patterns for efficiency
    match expr {
        // StaticField(type_name, "klass(...)") — the klass accessor directly
        Expr::StaticField(type_name, accessor_name) if accessor_name.starts_with("klass(") => {
            return Some(type_name.clone());
        }
        // RawDeref(inner, 0, _) — dereferenced pointer
        Expr::RawDeref(inner, 0, _) => {
            match inner.as_ref() {
                // RawDeref(StaticField(type_name, "klass(...)"), 0, 0)
                Expr::StaticField(type_name, accessor_name) if accessor_name.starts_with("klass(") => {
                    return Some(type_name.clone());
                }
                _ => {}
            }
        }
        // Int(klass_addr) — raw klass pointer address
        Expr::Int(addr) => {
            if let Some(&tdi) = maps.klass_to_tdi.get(&(*addr as u64)) {
                if let Some((ns, name)) = maps.tdi_names.get(&tdi) {
                    return Some(if ns.is_empty() {
                        name.clone()
                    } else {
                        format!("{}.{}", ns, name)
                    });
                }
            }
        }
        // A variable proved to hold a KLASS token names the token's type.
        // `local_types` cannot answer for these: the token map is built from
        // the variable's DEFINITIONS, not from the local's declared type.
        Expr::Var(n) if tokens.klass.contains_key(n) => {
            return tokens.klass.get(n).cloned();
        }
        _ => {}
    }
    
    // General fallback: use resolve_expr_type to determine the type of any expression.
    // This handles Var, This, Field, Cast, and other expressions with known types.
    if let Some(tdi) = resolve_expr_type(expr, maps, local_types, &maps.field_type_by_name, tokens) {
        if let Some((ns, name)) = maps.tdi_names.get(&tdi) {
            return Some(if ns.is_empty() {
                name.clone()
            } else {
                format!("{}.{}", ns, name)
            });
        }
    }
    
    None
}

/// Build a local variable → tdi map from the statement tree's type annotations.
///
/// Scans `Stmt::Assign(name, Some(type_decl), _)` patterns and extracts the
/// TypeDef index from the type declaration string by looking it up in the
/// `tdi_names` reverse map. This is a fallback for when the `LocalTable` is not
/// available — it catches explicitly typed assignments.
pub fn build_local_types_from_stmts(
    stmts: &[Stmt],
    tdi_names_reverse: &HashMap<String, i32>,
) -> HashMap<String, i32> {
    let mut result = HashMap::new();
    collect_typed_assigns(stmts, &mut result, tdi_names_reverse);
    result
}

/// Recursively collect typed assignments from the statement tree.
fn collect_typed_assigns(
    stmts: &[Stmt],
    out: &mut HashMap<String, i32>,
    tdi_names_reverse: &HashMap<String, i32>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign(name, Some(type_decl), _) => {
                if !type_decl.is_empty() && type_decl != "var" {
                    if let Some(&tdi) = tdi_names_reverse.get(type_decl) {
                        out.entry(name.clone()).or_insert(tdi);
                    }
                }
            }
            Stmt::If { then_body, else_body, .. } => {
                collect_typed_assigns(then_body, out, tdi_names_reverse);
                collect_typed_assigns(else_body, out, tdi_names_reverse);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::Using { body, .. }
            | Stmt::Lock { body, .. }
            | Stmt::ForEach { body, .. } => {
                collect_typed_assigns(body, out, tdi_names_reverse);
            }
            Stmt::For { init, incr, body, .. } => {
                collect_typed_assigns(init, out, tdi_names_reverse);
                collect_typed_assigns(incr, out, tdi_names_reverse);
                collect_typed_assigns(body, out, tdi_names_reverse);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                collect_typed_assigns(try_body, out, tdi_names_reverse);
                for (_, b) in catches {
                    collect_typed_assigns(b, out, tdi_names_reverse);
                }
                if let Some(fb) = finally_body {
                    collect_typed_assigns(fb, out, tdi_names_reverse);
                }
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases {
                    collect_typed_assigns(b, out, tdi_names_reverse);
                }
                collect_typed_assigns(tail, out, tdi_names_reverse);
                if let Some(arm) = default {
                    collect_typed_assigns(arm, out, tdi_names_reverse);
                }
            }
            _ => {}
        }
    }
}

/// Infer variable types from statement-level patterns.
///
/// Scans assignments in the statement tree and propagates types:
/// 1. `v1 = v2` where v2 has known type → v1 gets same type
/// 2. `v1 = expr.field` where expr has known type → v1 gets field's type
/// 3. `v1 = MethodName(...)` where MethodName has known return type → v1 gets return type
/// 4. `v1 = RawCall(addr, ...)` where addr maps to a method → v1 gets return type
///
/// Iterates to a fixpoint so chains like `v1 = this.playerStats; v2 = v1` propagate.
/// Returns the number of new types inferred (not counting those already in local_types).
pub fn infer_types_from_stmts(
    stmts: &[Stmt],
    maps: &ResolveMaps,
    local_types: &mut HashMap<String, i32>,
) -> usize {
    // Use pre-computed reverse maps from ResolveMaps (built once globally).
    let tdi_names_reverse = &maps.tdi_names_reverse;
    let method_return_tdi = &maps.method_return_tdi;

    let mut count = 0;
    // Iterate to fixpoint (typically 2-3 iterations suffice)
    for _iteration in 0..10 {
        // AGREEMENT, not first-wins. A display name can carry several SSA
        // live ranges / join arms with different runtime types; committing the
        // type inferred from ONE arm made later field lookups name unrelated
        // fields on the other arm (`BonusItem.defaultLayerMask` used as a
        // CharacterData in TokenManager.Initialize). A type is committed only
        // when EVERY definition agrees, or when an explicit source declaration
        // names it.
        let mut cand: HashMap<String, Option<i32>> = HashMap::new();
        let mut defs: HashMap<String, usize> = HashMap::new();
        let mut declared: HashMap<String, Option<i32>> = HashMap::new();
        collect_type_candidates(
            stmts,
            maps,
            local_types,
            tdi_names_reverse,
            method_return_tdi,
            &mut cand,
            &mut defs,
            &mut declared,
        );
        let mut changed = false;
        for (name, n) in &defs {
            if local_types.contains_key(name) {
                continue;
            }
            if let Some(decl) = declared.get(name) {
                if let Some(tdi) = decl {
                    local_types.insert(name.clone(), *tdi);
                    count += 1;
                    changed = true;
                }
                // A conflicting/unknown declaration gives no evidence.
                continue;
            }
            if let Some(Some(tdi)) = cand.get(name) {
                if *n > 0 {
                    local_types.insert(name.clone(), *tdi);
                    count += 1;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    count
}

/// Gather per-name type evidence from every definition in the method.
///
/// `cand[name]` is `Some(tdi)` only when all definitions infer the SAME type;
/// any unknown or disagreeing definition clears it. `declared[name]` records an
/// explicit rendered type declaration (`Some(tdi)`, or `None` on conflict).
#[allow(clippy::too_many_arguments)]
fn collect_type_candidates(
    stmts: &[Stmt],
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    tdi_names_reverse: &HashMap<String, i32>,
    method_return_tdi: &HashMap<String, i32>,
    cand: &mut HashMap<String, Option<i32>>,
    defs: &mut HashMap<String, usize>,
    declared: &mut HashMap<String, Option<i32>>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign(name, type_decl, expr) => {
                // A name already typed by the IR/table is authoritative.
                if local_types.contains_key(name) {
                    continue;
                }
                *defs.entry(name.clone()).or_insert(0) += 1;
                let mut has_decl = false;
                if let Some(td) = type_decl {
                    if !td.is_empty() && td != "var" {
                        match tdi_names_reverse.get(td) {
                            Some(&tdi) => {
                                has_decl = true;
                                match declared.get(name).copied() {
                                    None => {
                                        declared.insert(name.clone(), Some(tdi));
                                    }
                                    Some(Some(prev)) if prev == tdi => {}
                                    _ => {
                                        declared.insert(name.clone(), None);
                                    }
                                }
                            }
                            None => {
                                // An unresolvable declaration is not evidence.
                                declared.entry(name.clone()).or_insert(None);
                            }
                        }
                    }
                }
                if has_decl {
                    continue;
                }
                let inferred =
                    infer_expr_type(expr, maps, local_types, tdi_names_reverse, method_return_tdi);
                match cand.get(name).copied() {
                    None => {
                        cand.insert(name.clone(), inferred);
                    }
                    Some(prev) if prev == inferred => {}
                    Some(_) => {
                        cand.insert(name.clone(), None);
                    }
                }
            }
            Stmt::Using { decl, body } => {
                collect_type_candidates(
                    std::slice::from_ref(decl.as_ref()),
                    maps,
                    local_types,
                    tdi_names_reverse,
                    method_return_tdi,
                    cand,
                    defs,
                    declared,
                );
                collect_type_candidates(
                    body,
                    maps,
                    local_types,
                    tdi_names_reverse,
                    method_return_tdi,
                    cand,
                    defs,
                    declared,
                );
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                for k in [then_body, else_body] {
                    collect_type_candidates(
                        k,
                        maps,
                        local_types,
                        tdi_names_reverse,
                        method_return_tdi,
                        cand,
                        defs,
                        declared,
                    );
                }
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::Lock { body, .. }
            | Stmt::ForEach { body, .. } => {
                collect_type_candidates(
                    body,
                    maps,
                    local_types,
                    tdi_names_reverse,
                    method_return_tdi,
                    cand,
                    defs,
                    declared,
                );
            }
            Stmt::For {
                init, incr, body, ..
            } => {
                for k in [init, incr, body] {
                    collect_type_candidates(
                        k,
                        maps,
                        local_types,
                        tdi_names_reverse,
                        method_return_tdi,
                        cand,
                        defs,
                        declared,
                    );
                }
            }
            Stmt::Try {
                try_body,
                catches,
                finally_body,
            } => {
                collect_type_candidates(
                    try_body,
                    maps,
                    local_types,
                    tdi_names_reverse,
                    method_return_tdi,
                    cand,
                    defs,
                    declared,
                );
                for (_, b) in catches {
                    collect_type_candidates(
                        b,
                        maps,
                        local_types,
                        tdi_names_reverse,
                        method_return_tdi,
                        cand,
                        defs,
                        declared,
                    );
                }
                if let Some(fb) = finally_body {
                    collect_type_candidates(
                        fb,
                        maps,
                        local_types,
                        tdi_names_reverse,
                        method_return_tdi,
                        cand,
                        defs,
                        declared,
                    );
                }
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases {
                    collect_type_candidates(
                        b,
                        maps,
                        local_types,
                        tdi_names_reverse,
                        method_return_tdi,
                        cand,
                        defs,
                        declared,
                    );
                }
                if let Some(arm) = default {
                    collect_type_candidates(
                        arm,
                        maps,
                        local_types,
                        tdi_names_reverse,
                        method_return_tdi,
                        cand,
                        defs,
                        declared,
                    );
                }
                collect_type_candidates(
                    tail,
                    maps,
                    local_types,
                    tdi_names_reverse,
                    method_return_tdi,
                    cand,
                    defs,
                    declared,
                );
            }
            Stmt::StringSwitch { cases, tail, .. } => {
                for (_, b) in cases {
                    collect_type_candidates(
                        b,
                        maps,
                        local_types,
                        tdi_names_reverse,
                        method_return_tdi,
                        cand,
                        defs,
                        declared,
                    );
                }
                if let Some(t) = tail {
                    collect_type_candidates(
                        t,
                        maps,
                        local_types,
                        tdi_names_reverse,
                        method_return_tdi,
                        cand,
                        defs,
                        declared,
                    );
                }
            }
            _ => {}
        }
    }
}

/// Infer the type (tdi) of an expression, given the current local_types map.
///
/// Handles:
/// - `Var(name)`: look up in local_types
/// - `This`: look up "this" in local_types
/// - `Int(addr)`: look up in klass_to_tdi
/// - `Field(recv, field_name)`: recursively get recv's type, then look up field's type
/// - `Field(recv, "static_fields")`: returns recv's type (pseudo-field)
/// - `StaticField(type, "klass(type)")`: returns the type's tdi
/// - `RawDeref(expr, 0, _)` where expr is a klass StaticField: returns the type's tdi
/// - `Call { name, .. }`: look up method name in method_return_tdi
/// - `RawCall(addr, ..)`: look up addr in methods_by_rva
/// - `NewObj(type_name, ..)`: look up type_name in tdi_names_reverse
/// - `Cast(type_name, _)`: returns the cast target type's tdi
/// - `RawDeref(base, offset, _)` where base has known type and offset maps to a
///   field with known type: returns that field's type
fn infer_expr_type(
    expr: &Expr,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    tdi_names_reverse: &HashMap<String, i32>,
    method_return_tdi: &HashMap<String, i32>,
) -> Option<i32> {
    match expr {
        Expr::Var(name) => local_types.get(name).copied(),
        Expr::This => local_types.get("this").copied(),
        Expr::Int(addr) => maps.klass_to_tdi.get(&(*addr as u64)).copied(),
        Expr::Field(recv, field_name) => {
            // "static_fields" is a pseudo-field whose "type" is the receiver's type.
            if field_name == "static_fields" {
                return infer_expr_type(recv, maps, local_types, tdi_names_reverse, method_return_tdi);
            }
            let recv_tdi = infer_expr_type(recv, maps, local_types, tdi_names_reverse, method_return_tdi)?;
            maps.field_type_by_name.get(&(recv_tdi, field_name.clone())).copied()
        }
        // StaticField(type, "klass(type)") — the klass static field holds an
        // Il2CppClass* of that type. The "type" of this expression is the named type.
        Expr::StaticField(type_name, accessor) if accessor.starts_with("klass(") => {
            tdi_names_reverse.get(type_name).copied()
        }
        // StaticField(type, field_name) — NOT inferred from
        // `static_field_types`: that map is keyed by the FULL `Namespace.Type`
        // while the rendered field carries the short/accessor name, and a
        // measured A/B showed the lookup resolves no raw site (R194 neutral).
        // Keeping the arm explicit prevents a silent wrong-type commit if the
        // key space ever changes.
        Expr::StaticField(_type_name, _field_name) => None,
        Expr::Call { name, .. } => {
            method_return_tdi.get(name).copied()
        }
        Expr::RawCall(addr, _, _ret_type_str) => {
            maps.methods_by_rva.get(addr).and_then(|m| {
                if let ManagedType::Class(named) = &m.sig.ret {
                    if named.type_def_index >= 0 {
                        return Some(named.type_def_index);
                    }
                }
                None
            })
        }
        Expr::NewObj(type_name, _) => {
            tdi_names_reverse.get(type_name).copied()
        }
        // `Cond`/`Await` agreement inference was measured (R194) and is
        // neutral: the untyped mass is multi-def cells, not ternary/await
        // shapes. Do not re-add without a census delta.
        Expr::Cast(type_name, inner) => {
            // If casting to a known type, use that
            if let Some(&tdi) = tdi_names_reverse.get(type_name) {
                return Some(tdi);
            }
            // Otherwise, try the inner expression
            infer_expr_type(inner, maps, local_types, tdi_names_reverse, method_return_tdi)
        }
        // RawDeref(base, offset, _) — if the base has a known type and the
        // offset maps to a field with a known type, the result has that type.
        // Also handles RawDeref(StaticField(type, "klass(type)"), 0) → type.
        Expr::RawDeref(base, offset, _) => {
            // Special case: dereferencing a klass static field at offset 0
            // gives the Il2CppClass* of that type.
            if *offset == 0 {
                if let Expr::StaticField(type_name, accessor) = base.as_ref() {
                    if accessor.starts_with("klass(") {
                        return tdi_names_reverse.get(type_name).copied();
                    }
                }
            }
            // General case: if the base denotes a known type and the total
            // constant offset maps to a typed field, the load has that type.
            // Constant addends folded into the BASE expression (`obj + 16`)
            // count exactly like the RawDeref's own offset — the renderer
            // produces that shape by inlining an IR `base + off`.
            let (root, folded) = strip_const_addend(base);
            if let Some(base_tdi) =
                infer_expr_type(root, maps, local_types, tdi_names_reverse, method_return_tdi)
            {
                let total = folded.wrapping_add(*offset);
                if let Some(&field_type_tdi) = maps.field_types.get(&(base_tdi, total)) {
                    if field_type_tdi >= 0 {
                        return Some(field_type_tdi);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Report the field annotation rewrite count.
pub fn report() -> String {
    let resolved = FIELDS_RESOLVED.load(Ordering::Relaxed);
    let static_resolved = STATIC_FIELDS_RESOLVED.load(Ordering::Relaxed);
    let static_bytes = STATIC_BYTES_RESOLVED.load(Ordering::Relaxed);
    if resolved == 0 && static_resolved == 0 && static_bytes == 0 {
        return String::new();
    }
    let mut out = String::new();
    if resolved > 0 {
        out.push_str(&format!("[field_annotation] {resolved} raw deref(s) resolved to named fields\n"));
    }
    if static_resolved > 0 {
        out.push_str(&format!("[field_annotation] {static_resolved} static field access(es) resolved\n"));
    }
    if static_bytes > 0 {
        out.push_str(&format!("[field_annotation] {static_bytes} class-init bitmap byte(s) resolved\n"));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::render_stmts;

    /// Build a minimal ResolveMaps with field_layouts for testing.
    fn make_test_maps() -> ResolveMaps {
        let mut field_layouts = HashMap::new();
        // TypeDef 5 (MyClass): offset 184 → "Health", offset 24 → "Name",
        // offset 32 → "Position", offset 16 → "Inner"
        field_layouts.insert((5, 184), ("Health".to_string(), "MyClass".to_string()));
        field_layouts.insert((5, 24), ("Name".to_string(), "MyClass".to_string()));
        field_layouts.insert((5, 32), ("Position".to_string(), "MyClass".to_string()));
        field_layouts.insert((5, 16), ("Inner".to_string(), "MyClass".to_string()));
        // TypeDef 10 (OtherClass): offset 8 → "Count", offset 48 → "Data"
        field_layouts.insert((10, 8), ("Count".to_string(), "OtherClass".to_string()));
        field_layouts.insert((10, 48), ("Data".to_string(), "OtherClass".to_string()));
        // TypeDef 20 (DeepClass): offset 8 → "Value"
        field_layouts.insert((20, 8), ("Value".to_string(), "DeepClass".to_string()));

        let mut klass_to_tdi = HashMap::new();
        klass_to_tdi.insert(0x1000, 5);

        let mut tdi_names = HashMap::new();
        tdi_names.insert(
            30,
            (
                "System.Collections.Generic".to_string(),
                "List`1".to_string(),
            ),
        );

        // Field types: (tdi, offset) → field_type_tdi
        // Enables pointer chain resolution.
        let mut field_types = HashMap::new();
        field_types.insert((5, 16), 10);   // MyClass.Inner → OtherClass (tdi=10)
        field_types.insert((5, 32), 10);   // MyClass.Position → OtherClass (tdi=10)
        field_types.insert((10, 8), 20);   // OtherClass.Count → DeepClass (tdi=20)
        field_types.insert((10, 48), 5);   // OtherClass.Data → MyClass (tdi=5)

        let mut static_field_layouts = HashMap::new();
        // Static fields: address → (field_name, type_name)
        // 0x2f6a000 + 792 = 0x2f6a318 → "StaticInt" on "StaticClass"
        static_field_layouts.insert(0x2f6a318, ("StaticInt".to_string(), "StaticClass".to_string()));
        // 0x2f6a000 + 800 = 0x2f6a320 → "StaticObj" on "StaticClass"
        static_field_layouts.insert(0x2f6a320, ("StaticObj".to_string(), "StaticClass".to_string()));
        // 0x3000000 + 16 = 0x3000010 → "Counter" on "Globals"
        static_field_layouts.insert(0x3000010, ("Counter".to_string(), "Globals".to_string()));
        // A klass-token static slot: typeof(MyClass) + 56 → MyClass.CounterStatic
        let mut static_field_offsets = HashMap::new();
        static_field_offsets.insert(
            ("MyClass".to_string(), 56i64),
            ("CounterStatic".to_string(), "MyClass".to_string()),
        );

        // Inherited layout: a field declared on Base (tdi 1) must be visible
        // through MyClass (tdi 5), and its declared type drives chained
        // resolution.
        let mut type_parent_chains = HashMap::new();
        type_parent_chains.insert(5, vec![5, 1]);
        type_parent_chains.insert(10, vec![10, 1]);
        field_layouts.insert((1, 200), ("BaseField".to_string(), "Base".to_string()));
        field_types.insert((1, 200), 20);

        // Embedded VALUE-TYPE layout: GeoCache (tdi 40) declares `_geoInfo`
        // at 0x14 as a struct (tdi 41); HAPI_GeoInfo (41) declares `id` at
        // +8. `*(nint*)(cache + 28)` is `cache._geoInfo.id` — the exact-offset
        // lookup alone cannot see it.
        field_layouts.insert((40, 0x14), ("_geoInfo".to_string(), "GeoCache".to_string()));
        field_types.insert((40, 0x14), 41);
        field_layouts.insert((41, 8), ("id".to_string(), "HAPI_GeoInfo".to_string()));
        // A CLASS-typed field at 0x100: descent must NOT enter it, because the
        // storage at 0x100 holds a pointer, not the object's body.
        field_layouts.insert((5, 0x100), ("Child".to_string(), "MyClass".to_string()));
        field_types.insert((5, 0x100), 10);
        let mut value_type_tdis = std::collections::HashSet::new();
        value_type_tdis.insert(41);

        // Build tdi_fields from the declaration layout, mirroring
        // populate_field_layouts.
        let mut tdi_fields: HashMap<i32, Vec<(i64, String, i32)>> = HashMap::new();
        for (&(tdi, off), (name, _)) in &field_layouts {
            let ftdi = field_types.get(&(tdi, off)).copied().unwrap_or(-1);
            tdi_fields.entry(tdi).or_default().push((off, name.clone(), ftdi));
        }

        // Build field_type_by_name from field_types + field_layouts,
        // mirroring the production code in recovery/mod.rs.
        let mut field_type_by_name = HashMap::new();
        for ((tdi, offset), field_type_tdi) in &field_types {
            if let Some((field_name, _decl_type)) = field_layouts.get(&(*tdi, *offset)) {
                field_type_by_name
                    .entry((*tdi, field_name.clone()))
                    .or_insert(*field_type_tdi);
            }
        }

        ResolveMaps {
            methods_by_rva: HashMap::new(),
            literal_index_by_addr: HashMap::new(),
            literal_ptr_index_by_addr: HashMap::new(),
            literals: HashMap::new(),
            method_spec_index_by_addr: HashMap::new(),
            specs_by_body_rva: HashMap::new(),
            shared_by_rva: HashMap::new(),
            shared_cross_type: HashMap::new(),
            shared_rva_of_method: HashMap::new(),
            specs_by_index: HashMap::new(),
            object_new: vec![],
            string_new: vec![],
            box_fn: vec![],
            raise_exception: vec![],
            metadata_init: vec![],
            vtable_slots: HashMap::new(),
            klass_to_tdi,
            usage_type_by_addr: HashMap::new(),
            named_type_sizes: HashMap::new(),
            tdi_names,
            objc_selector_stubs: HashMap::new(),
            native_wrappers: HashMap::new(),
            field_layouts,
            field_types,
            field_widths: HashMap::new(),
            static_field_layouts,
            field_type_by_name,
            field_generic_text: HashMap::new(),
            interface_tdis: std::collections::HashSet::new(),
            type_parent_chains,
            enum_members: HashMap::new(),
            attribute_ctor_names: HashMap::new(),
            got_addend_by_addr: HashMap::new(),
            method_name_by_va: HashMap::new(),
            static_field_offsets,
            static_field_types: HashMap::new(),
            tdi_names_reverse: HashMap::new(),
            method_return_tdi: HashMap::new(),
            tdi_fields,
            value_type_tdis,
        }
    }

    fn make_local_types() -> HashMap<String, i32> {
        let mut m = HashMap::new();
        m.insert("obj".to_string(), 5);       // obj : MyClass (tdi=5)
        m.insert("other".to_string(), 10);     // other : OtherClass (tdi=10)
        m.insert("this".to_string(), 5);       // this : MyClass (tdi=5)
        m.insert("list".to_string(), 30);      // list : List<T> (generic def tdi)
        m
    }

    // -----------------------------------------------------------------------
    // Test 1: Simple field access
    // -----------------------------------------------------------------------

    #[test]
    fn simple_field_access() {
        reset_counters();
        // *(nint*)(obj + 184) → obj.Health
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("obj".into())),
            184,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve one field access");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(base, name)) => {
                assert_eq!(name, "Health");
                assert!(matches!(base.as_ref(), Expr::Var(n) if n == "obj"));
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    /// Fields declared on an ANCESTOR are named too, and their declared type
    /// continues the chain: `obj.BaseField.Value`.
    #[test]
    fn inherited_field_layout_is_named() {
        reset_counters();
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::Var("obj".into())),
                200,
                0,
            )),
            8,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "both the inherited field and its member resolve");
        let text = render_stmts(&stmts);
        assert!(text.contains("obj.BaseField.Value"), "{text}");
    }

    /// A klass token addressed at a static offset is a STATIC field, not an
    /// instance field of System.Type: `*(int*)(typeof(NGUIText) +56)`.
    #[test]
    fn klass_token_static_field_is_named() {
        reset_counters();
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::TypeOf("MyClass".into())),
            56,
            4,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StaticField(t, f)) => {
                assert_eq!(t, "MyClass");
                assert_eq!(f, "CounterStatic");
            }
            other => panic!("expected a static field, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: Chained field access — 2-level pointer chain resolution
    // -----------------------------------------------------------------------

    #[test]
    fn chained_field_access_two_levels() {
        reset_counters();
        // *(nint*)(*(nint*)(obj + 16) + 8)
        // Inner: *(nint*)(obj + 16) → obj.Inner (MyClass.Inner → OtherClass)
        // Outer: *(nint*)(obj.Inner + 8) → obj.Inner.Count (OtherClass.Count)
        // Both levels resolve via chain resolution.
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::Var("obj".into())),
                16,
                0,
            )),
            8,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "both inner and outer deref should resolve");
        // Should be: obj.Inner.Count
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, name)) => {
                assert_eq!(name, "Count", "outer field should be Count");
                match recv.as_ref() {
                    Expr::Field(inner_recv, inner_name) => {
                        assert_eq!(inner_name, "Inner", "inner field should be Inner");
                        assert!(matches!(inner_recv.as_ref(), Expr::Var(n) if n == "obj"));
                    }
                    other => panic!("expected Field(Field(Var, _), _), got {other:?}"),
                }
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    #[test]
    fn embedded_struct_component_resolves_to_dotted_path() {
        reset_counters();
        // *(nint*)(cache + 28): GeoCache._geoInfo is a struct at 0x14 whose
        // `id` sits at +8, so this is cache._geoInfo.id.
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("cache".into())),
            28,
            0,
        ))];
        let maps = make_test_maps();
        let mut locals = make_local_types();
        locals.insert("cache".to_string(), 40);
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, name)) => {
                assert_eq!(name, "_geoInfo.id");
                assert!(matches!(recv.as_ref(), Expr::Var(v) if v == "cache"));
            }
            other => panic!("expected Eval(Field(_geoInfo.id)), got {other:?}"),
        }
    }

    #[test]
    fn embedded_component_with_wrong_leaf_width_is_refused() {
        reset_counters();
        // The leaf `id` is declared 4 bytes; an 8-byte access there is not
        // that field (it spans the next member), so it must stay raw.
        let mut maps = make_test_maps();
        maps.field_widths.insert((41, 8), 4);
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("cache".into())),
            28,
            8,
        ))];
        let mut locals = make_local_types();
        locals.insert("cache".to_string(), 40);
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 0, "{stmts:?}");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(..))));
    }

    #[test]
    fn embedded_descent_does_not_enter_class_typed_fields() {
        reset_counters();
        // `Child` at 0x100 is a CLASS field; obj + 0x108 is not a field of
        // the outer object, and the pointer at 0x100 has no +8 member stored
        // in the outer object. Must stay raw.
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("obj".into())),
            0x108,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 0, "{stmts:?}");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(..))));
    }

    #[test]
    fn static_block_temp_resolves_through_its_owner() {
        reset_counters();
        // k = klass (metadata slot); block = *(k + 184); v = *(block + 56).
        // The block belongs to TRWorldInfo, so +56 is its static field.
        let mut maps = make_test_maps();
        maps.klass_to_tdi.insert(0xABC, 50);
        maps.tdi_names.insert(50, (String::new(), "TRWorldInfo".to_string()));
        maps.static_field_offsets.insert(
            ("TRWorldInfo".to_string(), 56),
            ("RegionMgr".to_string(), "TRWorldInfo".to_string()),
        );
        let mut stmts = vec![
            Stmt::Assign(
                "k".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Int(0xABC)), 0, 0),
            ),
            Stmt::Assign(
                "block".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Var("k".into())), 184, 0),
            ),
            Stmt::Assign(
                "v".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Var("block".into())), 56, 0),
            ),
        ];
        let locals = HashMap::new();
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        match &stmts[2] {
            Stmt::Assign(_, _, Expr::StaticField(ty, name)) => {
                assert_eq!(ty, "TRWorldInfo");
                assert_eq!(name, "RegionMgr");
            }
            other => panic!("expected StaticField(TRWorldInfo, RegionMgr), got {other:?}"),
        }
    }

    #[test]
    fn static_block_temp_never_uses_instance_layouts() {
        reset_counters();
        // The block temp shares its NAME with an object view via local_types
        // (MyClass, "Name" at 24). Reading +24 from a static block is NOT
        // MyClass.Name; with no static field at that offset it must stay raw.
        let mut maps = make_test_maps();
        maps.klass_to_tdi.insert(0xABC, 50);
        maps.tdi_names.insert(50, (String::new(), "TRWorldInfo".to_string()));
        let mut stmts = vec![
            Stmt::Assign(
                "k".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Int(0xABC)), 0, 0),
            ),
            Stmt::Assign(
                "block".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Var("k".into())), 184, 0),
            ),
            Stmt::Eval(Expr::RawDeref(Box::new(Expr::Var("block".into())), 24, 0)),
        ];
        let mut locals = HashMap::new();
        locals.insert("block".to_string(), 5); // MyClass, Name at 24
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert!(
            matches!(&stmts[2], Stmt::Eval(Expr::RawDeref(..))),
            "{stmts:?}"
        );
    }

    #[test]
    fn klass_var_never_uses_instance_layouts() {
        reset_counters();
        // `k` is a klass token; +24 is not MyClass.Name even though the
        // local-type map says the name is a MyClass.
        let mut maps = make_test_maps();
        maps.klass_to_tdi.insert(0xABC, 50);
        maps.tdi_names.insert(50, (String::new(), "TRWorldInfo".to_string()));
        let mut stmts = vec![
            Stmt::Assign(
                "k".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Int(0xABC)), 0, 0),
            ),
            Stmt::Eval(Expr::RawDeref(Box::new(Expr::Var("k".into())), 24, 0)),
        ];
        let mut locals = HashMap::new();
        locals.insert("k".to_string(), 5);
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert!(
            matches!(&stmts[1], Stmt::Eval(Expr::RawDeref(..))),
            "{stmts:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Field in assignment
    // -----------------------------------------------------------------------

    #[test]
    fn field_in_assignment() {
        reset_counters();
        // x = *(nint*)(obj + 24) → x = obj.Name
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::RawDeref(Box::new(Expr::Var("obj".into())), 24, 0),
        )];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Field(base, name)) => {
                assert_eq!(target, "x");
                assert_eq!(name, "Name");
                assert!(matches!(base.as_ref(), Expr::Var(n) if n == "obj"));
            }
            other => panic!("expected Assign(x, _, Field), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: Field in condition
    // -----------------------------------------------------------------------

    #[test]
    fn field_in_condition() {
        reset_counters();
        // if (*(nint*)(obj + 184) != 0) { ... }
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                crate::recovery::csharp::CmpKind::Ne,
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Var("obj".into())),
                    184,
                    0,
                )),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![Stmt::Eval(Expr::Var("x".into()))],
            else_body: vec![],
        }];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve field in condition");
        match &stmts[0] {
            Stmt::If { cond, .. } => {
                match cond {
                    Expr::Cmp(_, lhs, _) => {
                        assert!(matches!(lhs.as_ref(), Expr::Field(_, name) if name == "Health"));
                    }
                    other => panic!("expected Cmp, got {other:?}"),
                }
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Non-match — base type unknown
    // -----------------------------------------------------------------------

    #[test]
    fn non_match_unknown_type() {
        reset_counters();
        // *(nint*)(unknown_var + 24) — unknown_var not in local_types
        // Use offset 24 (not an Il2CppClass well-known offset) so only
        // instance field resolution is tested.
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("unknown_var".into())),
            24,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 0, "should not resolve when base type is unknown");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(..))));
    }

    // -----------------------------------------------------------------------
    // Test 6: Non-match — offset doesn't match any field
    // -----------------------------------------------------------------------

    #[test]
    fn non_match_unknown_offset() {
        reset_counters();
        // *(nint*)(obj + 999) — offset 999 not in field_layouts for tdi=5
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("obj".into())),
            999,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 0, "should not resolve when offset has no field");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(..))));
    }

    // -----------------------------------------------------------------------
    // Test 7: Multiple fields in one statement
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_fields_in_one_statement() {
        reset_counters();
        // x = *(nint*)(obj + 24) + *(nint*)(other + 8)
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Bin(
                crate::recovery::csharp::BinOp::Add,
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Var("obj".into())),
                    24,
                    0,
                )),
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Var("other".into())),
                    8,
                    0,
                )),
            ),
        )];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "should resolve both field accesses");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Bin(_, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Field(_, name) if name == "Name"));
                assert!(matches!(rhs.as_ref(), Expr::Field(_, name) if name == "Count"));
            }
            other => panic!("expected Assign with Bin, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: Field in nested blocks (if body)
    // -----------------------------------------------------------------------

    #[test]
    fn field_in_nested_blocks() {
        reset_counters();
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Assign(
                "y".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Var("obj".into())), 32, 0),
            )],
            else_body: vec![Stmt::Assign(
                "z".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Var("other".into())), 48, 0),
            )],
        }];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "should resolve fields in both branches");
        match &stmts[0] {
            Stmt::If { then_body, else_body, .. } => {
                match &then_body[0] {
                    Stmt::Assign(_, _, Expr::Field(_, name)) => assert_eq!(name, "Position"),
                    other => panic!("expected Field in then_body, got {other:?}"),
                }
                match &else_body[0] {
                    Stmt::Assign(_, _, Expr::Field(_, name)) => assert_eq!(name, "Data"),
                    other => panic!("expected Field in else_body, got {other:?}"),
                }
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 9: This reference
    // -----------------------------------------------------------------------

    #[test]
    fn this_reference_field() {
        reset_counters();
        // *(nint*)(this + 184) → this.Health
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::This),
            184,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve this.Field");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(base, name)) => {
                assert_eq!(name, "Health");
                assert!(matches!(base.as_ref(), Expr::This));
            }
            other => panic!("expected Eval(Field(This, _)), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 10: Klass pointer constant
    // -----------------------------------------------------------------------

    #[test]
    fn klass_pointer_constant() {
        reset_counters();
        // *(nint*)(0x1000 + 184) → 0x1000.Health (klass pointer → tdi=5)
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Int(0x1000)),
            184,
            0,
        ))];
        let maps = make_test_maps();
        let locals = HashMap::new(); // no local types needed for klass pointer
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve klass pointer field");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(base, name)) => {
                assert_eq!(name, "Health");
                assert!(matches!(base.as_ref(), Expr::Int(0x1000)));
            }
            other => panic!("expected Eval(Field(Int, _)), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 11: Empty field_layouts is a no-op
    // -----------------------------------------------------------------------

    #[test]
    fn empty_field_layouts_is_noop() {
        reset_counters();
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("obj".into())),
            184,
            0,
        ))];
        let maps = ResolveMaps::default(); // empty field_layouts
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 0, "empty field_layouts should be a no-op");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(..))));
    }

    // -----------------------------------------------------------------------
    // Test 12: Report is empty when nothing fires
    // -----------------------------------------------------------------------

    #[test]
    fn report_empty_when_nothing_fires() {
        reset_counters();
        let r = report();
        assert!(r.is_empty(), "report should be empty when nothing fires");
    }

    // -----------------------------------------------------------------------
    // Test 13: Rendering produces correct output
    // -----------------------------------------------------------------------

    #[test]
    fn renders_correctly_after_resolution() {
        reset_counters();
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::RawDeref(Box::new(Expr::Var("obj".into())), 24, 0),
        )];
        let maps = make_test_maps();
        let locals = make_local_types();
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("obj.Name"),
            "rendered output should contain 'obj.Name', got: {output}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 14: Non-Var base (e.g. Bin expression) is not resolved
    // -----------------------------------------------------------------------

    #[test]
    fn non_var_base_not_resolved() {
        reset_counters();
        // *(nint*)((obj + other) + 24) — base is a Bin, not a Var
        // Use offset 24 (not an Il2CppClass well-known offset) so only
        // instance field resolution is tested.
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Bin(
                crate::recovery::csharp::BinOp::Add,
                Box::new(Expr::Var("obj".into())),
                Box::new(Expr::Var("other".into())),
            )),
            24,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 0, "should not resolve when base is not a simple Var/This/Int");
    }

    // -----------------------------------------------------------------------
    // Test 15: Static field — simple resolution
    // -----------------------------------------------------------------------

    #[test]
    fn static_field_simple_resolution() {
        reset_counters();
        // *(nint*)(0x2f6a000 + 792) → StaticClass.StaticInt
        // 0x2f6a000 + 792 = 0x2f6a318
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Int(0x2f6a000)),
            792,
            0,
        ))];
        let maps = make_test_maps();
        let locals = HashMap::new();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve one static field access");
        match &stmts[0] {
            Stmt::Eval(Expr::StaticField(ty, name)) => {
                assert_eq!(ty, "StaticClass");
                assert_eq!(name, "StaticInt");
            }
            other => panic!("expected Eval(StaticField), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 16: Static field — different offset, same base
    // -----------------------------------------------------------------------

    #[test]
    fn static_field_different_offset() {
        reset_counters();
        // *(nint*)(0x2f6a000 + 800) → StaticClass.StaticObj
        // 0x2f6a000 + 800 = 0x2f6a320
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Int(0x2f6a000)),
            800,
            0,
        ))];
        let maps = make_test_maps();
        let locals = HashMap::new();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve static field at offset 800");
        match &stmts[0] {
            Stmt::Eval(Expr::StaticField(ty, name)) => {
                assert_eq!(ty, "StaticClass");
                assert_eq!(name, "StaticObj");
            }
            other => panic!("expected Eval(StaticField), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 17: Static field — different base address
    // -----------------------------------------------------------------------

    #[test]
    fn static_field_different_base() {
        reset_counters();
        // *(nint*)(0x3000000 + 16) → Globals.Counter
        // 0x3000000 + 16 = 0x3000010
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Int(0x3000000)),
            16,
            0,
        ))];
        let maps = make_test_maps();
        let locals = HashMap::new();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve static field with different base");
        match &stmts[0] {
            Stmt::Eval(Expr::StaticField(ty, name)) => {
                assert_eq!(ty, "Globals");
                assert_eq!(name, "Counter");
            }
            other => panic!("expected Eval(StaticField), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 18: Static field — non-matching address
    // -----------------------------------------------------------------------

    #[test]
    fn static_field_non_matching_address() {
        reset_counters();
        // *(nint*)(0x2f6a000 + 999) — offset 999 not in static_field_layouts
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Int(0x2f6a000)),
            999,
            0,
        ))];
        let maps = make_test_maps();
        let locals = HashMap::new();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(
            n, 0,
            "should not resolve when address is not in static_field_layouts: {stmts:?}"
        );
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(..))));
    }

    // -----------------------------------------------------------------------
    // Test 19: Static field — in assignment
    // -----------------------------------------------------------------------

    #[test]
    fn static_field_in_assignment() {
        reset_counters();
        // x = *(nint*)(0x2f6a000 + 792) → x = StaticClass.StaticInt
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::RawDeref(Box::new(Expr::Int(0x2f6a000)), 792, 0),
        )];
        let maps = make_test_maps();
        let locals = HashMap::new();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::StaticField(ty, name)) => {
                assert_eq!(target, "x");
                assert_eq!(ty, "StaticClass");
                assert_eq!(name, "StaticInt");
            }
            other => panic!("expected Assign(x, _, StaticField), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 20: Static field — rendering produces correct output
    // -----------------------------------------------------------------------

    #[test]
    fn static_field_renders_correctly() {
        reset_counters();
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::RawDeref(Box::new(Expr::Int(0x2f6a000)), 792, 0),
        )];
        let maps = make_test_maps();
        let locals = HashMap::new();
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("StaticClass.StaticInt"),
            "rendered output should contain 'StaticClass.StaticInt', got: {output}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 21: Mixed instance and static fields in one statement
    // -----------------------------------------------------------------------

    #[test]
    fn mixed_instance_and_static_fields() {
        reset_counters();
        // x = obj.Health + *(nint*)(0x2f6a000 + 792)
        //   = obj.Health + StaticClass.StaticInt
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Bin(
                crate::recovery::csharp::BinOp::Add,
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Var("obj".into())),
                    184,
                    0,
                )),
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Int(0x2f6a000)),
                    792,
                    0,
                )),
            ),
        )];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "should resolve both instance and static field");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Bin(_, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Field(_, name) if name == "Health"));
                assert!(matches!(rhs.as_ref(), Expr::StaticField(ty, name) if ty == "StaticClass" && name == "StaticInt"));
            }
            other => panic!("expected Assign with Bin, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 22: Static field — non-Int base is not resolved
    // -----------------------------------------------------------------------

    #[test]
    fn static_field_non_int_base_not_resolved() {
        reset_counters();
        // *(nint*)(var + 792) — base is a Var, not an Int, so static resolution doesn't apply
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("some_var".into())),
            792,
            0,
        ))];
        let maps = make_test_maps();
        let locals = HashMap::new(); // some_var not in locals
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 0, "should not resolve static field when base is not Int");
    }

    // -----------------------------------------------------------------------
    // Test 23: Report includes static field count
    // -----------------------------------------------------------------------

    #[test]
    fn report_includes_static_field_count() {
        reset_counters();
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Int(0x2f6a000)),
            792,
            0,
        ))];
        let maps = make_test_maps();
        let locals = HashMap::new();
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        let r = report();
        assert!(
            r.contains("1 static field access(es) resolved"),
            "report should mention static field resolution, got: {r}"
        );
    }

    // -----------------------------------------------------------------------
    // Pointer chain resolution tests
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Test 24: 3-level pointer chain
    // -----------------------------------------------------------------------

    #[test]
    fn three_level_pointer_chain() {
        reset_counters();
        // *(nint*)(*(nint*)(*(nint*)(obj + 16) + 8) + 8)
        // Level 1: *(nint*)(obj + 16) → obj.Inner (MyClass.Inner → OtherClass)
        // Level 2: *(nint*)(obj.Inner + 8) → obj.Inner.Count (OtherClass.Count → DeepClass)
        // Level 3: *(nint*)(obj.Inner.Count + 8) → obj.Inner.Count.Value (DeepClass.Value)
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Var("obj".into())),
                    16,
                    0,
                )),
                8,
                0,
            )),
            8,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 3, "all three levels should resolve");
        // Should be: obj.Inner.Count.Value
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv3, name3)) => {
                assert_eq!(name3, "Value");
                match recv3.as_ref() {
                    Expr::Field(recv2, name2) => {
                        assert_eq!(name2, "Count");
                        match recv2.as_ref() {
                            Expr::Field(recv1, name1) => {
                                assert_eq!(name1, "Inner");
                                assert!(matches!(recv1.as_ref(), Expr::Var(n) if n == "obj"));
                            }
                            other => panic!("expected Field(Var, _), got {other:?}"),
                        }
                    }
                    other => panic!("expected Field(Field, _), got {other:?}"),
                }
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 25: Chain where intermediate type is unknown — stops at known level
    // -----------------------------------------------------------------------

    #[test]
    fn chain_stops_when_intermediate_type_unknown() {
        reset_counters();
        // *(nint*)(*(nint*)(obj + 184) + 8)
        // Level 1: *(nint*)(obj + 184) → obj.Health (MyClass.Health, type not in field_types)
        // Level 2: *(nint*)(obj.Health + 8) — cannot resolve because Health's type is unknown
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::Var("obj".into())),
                184,
                0,
            )),
            8,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        // Only inner resolves; outer cannot because Health's type is not in field_types.
        assert_eq!(n, 1, "only inner deref should resolve");
        match &stmts[0] {
            Stmt::Eval(Expr::RawDeref(base, 8, _)) => {
                assert!(matches!(base.as_ref(), Expr::Field(_, name) if name == "Health"));
            }
            other => panic!("expected RawDeref(Field(Health), 8), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 26: Chain with `this` as base
    // -----------------------------------------------------------------------

    #[test]
    fn chain_with_this_base() {
        reset_counters();
        // *(nint*)(*(nint*)(this + 16) + 8)
        // Level 1: *(nint*)(this + 16) → this.Inner (MyClass.Inner → OtherClass)
        // Level 2: *(nint*)(this.Inner + 8) → this.Inner.Count (OtherClass.Count)
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::This),
                16,
                0,
            )),
            8,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "both levels should resolve with this base");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, name)) => {
                assert_eq!(name, "Count");
                match recv.as_ref() {
                    Expr::Field(inner_recv, inner_name) => {
                        assert_eq!(inner_name, "Inner");
                        assert!(matches!(inner_recv.as_ref(), Expr::This));
                    }
                    other => panic!("expected Field(This, _), got {other:?}"),
                }
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 27: Chain in assignment RHS
    // -----------------------------------------------------------------------

    #[test]
    fn chain_in_assignment_rhs() {
        reset_counters();
        // x = *(nint*)(*(nint*)(obj + 16) + 8)
        // → x = obj.Inner.Count
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::RawDeref(
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Var("obj".into())),
                    16,
                    0,
                )),
                8,
                0,
            ),
        )];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "both levels should resolve in assignment");
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Field(recv, name)) => {
                assert_eq!(target, "x");
                assert_eq!(name, "Count");
                assert!(matches!(recv.as_ref(), Expr::Field(_, n) if n == "Inner"));
            }
            other => panic!("expected Assign(x, _, Field), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 28: Chain rendering produces correct output
    // -----------------------------------------------------------------------

    #[test]
    fn chain_renders_correctly() {
        reset_counters();
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::RawDeref(
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Var("obj".into())),
                    16,
                    0,
                )),
                8,
                0,
            ),
        )];
        let maps = make_test_maps();
        let locals = make_local_types();
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("obj.Inner.Count"),
            "rendered output should contain 'obj.Inner.Count', got: {output}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 29: Two independent chains in one statement
    // -----------------------------------------------------------------------

    #[test]
    fn two_independent_chains_in_one_statement() {
        reset_counters();
        // x = *(nint*)(*(nint*)(obj + 16) + 8) + *(nint*)(*(nint*)(other + 48) + 184)
        // Chain 1: obj.Inner.Count (MyClass→OtherClass→DeepClass... wait, Count→DeepClass,
        //          but offset 8 on DeepClass → Value)
        // Actually: obj.Inner (offset 16, tdi 5→10), then .Count (offset 8, tdi 10→20)
        // Chain 2: other.Data (offset 48, tdi 10→5), then .Health (offset 184, tdi 5)
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Bin(
                crate::recovery::csharp::BinOp::Add,
                Box::new(Expr::RawDeref(
                    Box::new(Expr::RawDeref(
                        Box::new(Expr::Var("obj".into())),
                        16,
                        0,
                    )),
                    8,
                    0,
                )),
                Box::new(Expr::RawDeref(
                    Box::new(Expr::RawDeref(
                        Box::new(Expr::Var("other".into())),
                        48,
                        0,
                    )),
                    184,
                    0,
                )),
            ),
        )];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 4, "all four derefs should resolve (2 chains x 2 levels)");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Bin(_, lhs, rhs)) => {
                // LHS: obj.Inner.Count
                assert!(matches!(lhs.as_ref(), Expr::Field(_, name) if name == "Count"));
                // RHS: other.Data.Health
                assert!(matches!(rhs.as_ref(), Expr::Field(_, name) if name == "Health"));
            }
            other => panic!("expected Assign with Bin, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 30: Chain with klass pointer base
    // -----------------------------------------------------------------------

    #[test]
    fn chain_with_klass_pointer_base() {
        reset_counters();
        // *(nint*)(*(nint*)(0x1000 + 16) + 8)
        // Level 1: *(nint*)(0x1000 + 16) → 0x1000.Inner (klass→tdi=5, MyClass.Inner → OtherClass)
        // Level 2: *(nint*)(0x1000.Inner + 8) → 0x1000.Inner.Count (OtherClass.Count)
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::Int(0x1000)),
                16,
                0,
            )),
            8,
            0,
        ))];
        let maps = make_test_maps();
        let locals = HashMap::new();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "both levels should resolve with klass pointer base");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, name)) => {
                assert_eq!(name, "Count");
                match recv.as_ref() {
                    Expr::Field(inner_recv, inner_name) => {
                        assert_eq!(inner_name, "Inner");
                        assert!(matches!(inner_recv.as_ref(), Expr::Int(0x1000)));
                    }
                    other => panic!("expected Field(Int, _), got {other:?}"),
                }
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 31: Chain where outer offset has no field — only inner resolves
    // -----------------------------------------------------------------------

    #[test]
    fn chain_outer_offset_unknown() {
        reset_counters();
        // *(nint*)(*(nint*)(obj + 16) + 999)
        // Level 1: *(nint*)(obj + 16) → obj.Inner (resolves)
        // Level 2: *(nint*)(obj.Inner + 999) — offset 999 not in field_layouts for tdi=10
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::Var("obj".into())),
                16,
                0,
            )),
            999,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "only inner deref should resolve");
        match &stmts[0] {
            Stmt::Eval(Expr::RawDeref(base, 999, _)) => {
                assert!(matches!(base.as_ref(), Expr::Field(_, name) if name == "Inner"));
            }
            other => panic!("expected RawDeref(Field, 999), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 32: Chain in condition expression
    // -----------------------------------------------------------------------

    #[test]
    fn chain_in_condition() {
        reset_counters();
        // if (*(nint*)(*(nint*)(obj + 16) + 8) != 0) { ... }
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                crate::recovery::csharp::CmpKind::Ne,
                Box::new(Expr::RawDeref(
                    Box::new(Expr::RawDeref(
                        Box::new(Expr::Var("obj".into())),
                        16,
                        0,
                    )),
                    8,
                    0,
                )),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![Stmt::Eval(Expr::Var("x".into()))],
            else_body: vec![],
        }];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "both levels should resolve in condition");
        match &stmts[0] {
            Stmt::If { cond, .. } => {
                match cond {
                    Expr::Cmp(_, lhs, _) => {
                        assert!(matches!(lhs.as_ref(), Expr::Field(_, name) if name == "Count"));
                    }
                    other => panic!("expected Cmp, got {other:?}"),
                }
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Tests for infer_types_from_stmts
    // -----------------------------------------------------------------------

    /// Helper: build ResolveMaps with tdi_names for type inference tests.
    fn make_test_maps_with_tdi_names() -> ResolveMaps {
        let mut maps = make_test_maps();
        // Add tdi_names so we can resolve type names
        maps.tdi_names.insert(5, ("".into(), "MyClass".into()));
        maps.tdi_names.insert(10, ("".into(), "OtherClass".into()));
        maps.tdi_names.insert(20, ("".into(), "DeepClass".into()));
        // Populate pre-computed reverse maps (mirrors production init in mod.rs)
        maps.tdi_names_reverse.insert("MyClass".into(), 5);
        maps.tdi_names_reverse.insert("OtherClass".into(), 10);
        maps.tdi_names_reverse.insert("DeepClass".into(), 20);
        maps
    }

    // -----------------------------------------------------------------------
    // Test 33: Variable-to-variable assignment propagation
    // -----------------------------------------------------------------------

    #[test]
    fn infer_type_var_to_var_assignment() {
        reset_counters();
        // v1 = obj; *(nint*)(v1 + 184) should resolve to v1.Health
        // because v1 gets obj's type (MyClass, tdi=5)
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Var("obj".into())),
            Stmt::Eval(Expr::RawDeref(
                Box::new(Expr::Var("v1".into())),
                184,
                0,
            )),
        ];
        let maps = make_test_maps_with_tdi_names();
        let mut locals = make_local_types();
        // Before inference: v1 not in locals
        assert!(!locals.contains_key("v1"));
        // Run type inference
        let n = infer_types_from_stmts(&stmts, &maps, &mut locals);
        assert_eq!(n, 1, "should infer v1's type from obj");
        assert_eq!(locals.get("v1"), Some(&5), "v1 should have tdi=5 (MyClass)");
        // Now apply field annotation
        let rewrites = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(rewrites, 1, "should resolve v1.Health after inference");
    }

    // -----------------------------------------------------------------------
    // Test 34: Field access type inference
    // -----------------------------------------------------------------------

    #[test]
    fn infer_type_from_field_access() {
        reset_counters();
        // v1 = obj.Inner; *(nint*)(v1 + 8) should resolve to v1.Count
        // because obj.Inner has type OtherClass (tdi=10)
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(Expr::Var("obj".into())), "Inner".into()),
            ),
            Stmt::Eval(Expr::RawDeref(
                Box::new(Expr::Var("v1".into())),
                8,
                0,
            )),
        ];
        let maps = make_test_maps_with_tdi_names();
        let mut locals = make_local_types();
        let n = infer_types_from_stmts(&stmts, &maps, &mut locals);
        assert_eq!(n, 1, "should infer v1's type from obj.Inner");
        assert_eq!(locals.get("v1"), Some(&10), "v1 should have tdi=10 (OtherClass)");
        let rewrites = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(rewrites, 1, "should resolve v1.Count after inference");
    }

    // -----------------------------------------------------------------------
    // Test 35: Chain propagation through multiple assignments
    // -----------------------------------------------------------------------

    #[test]
    fn infer_type_chain_propagation() {
        reset_counters();
        // v1 = obj; v2 = v1; *(nint*)(v2 + 184) should resolve
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Var("obj".into())),
            Stmt::Assign("v2".into(), None, Expr::Var("v1".into())),
            Stmt::Eval(Expr::RawDeref(
                Box::new(Expr::Var("v2".into())),
                184,
                0,
            )),
        ];
        let maps = make_test_maps_with_tdi_names();
        let mut locals = make_local_types();
        let n = infer_types_from_stmts(&stmts, &maps, &mut locals);
        assert_eq!(n, 2, "should infer both v1 and v2 types");
        assert_eq!(locals.get("v1"), Some(&5));
        assert_eq!(locals.get("v2"), Some(&5));
        let rewrites = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(rewrites, 1, "should resolve v2.Health");
    }

    // -----------------------------------------------------------------------
    // Test 35b: mixed-type definitions must not type the local
    // -----------------------------------------------------------------------

    #[test]
    fn disagreeing_defs_do_not_infer_a_type() {
        reset_counters();
        // v1 holds an OtherClass on one arm and a MyClass on the other. Typing
        // it from either arm fabricated field names on the other arm
        // (`BonusItem.defaultLayerMask` used as a CharacterData in
        // TokenManager.Initialize).
        let stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(Expr::Var("obj".into())), "Inner".into()),
            ),
            Stmt::Assign("v1".into(), None, Expr::Var("obj".into())),
        ];
        let maps = make_test_maps_with_tdi_names();
        let mut locals = make_local_types();
        let n = infer_types_from_stmts(&stmts, &maps, &mut locals);
        assert_eq!(n, 0, "disagreeing definitions must not infer a type");
        assert_eq!(locals.get("v1"), None, "v1 must stay untyped");
    }

    // -----------------------------------------------------------------------
    // Test 36: This field access inference
    // -----------------------------------------------------------------------

    #[test]
    fn infer_type_from_this_field() {
        reset_counters();
        // v1 = this.Inner; *(nint*)(v1 + 8) should resolve to v1.Count
        let stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(Expr::This), "Inner".into()),
            ),
            Stmt::Eval(Expr::RawDeref(
                Box::new(Expr::Var("v1".into())),
                8,
                0,
            )),
        ];
        let maps = make_test_maps_with_tdi_names();
        let mut locals = make_local_types();
        locals.insert("this".to_string(), 5); // this: MyClass
        let n = infer_types_from_stmts(&stmts, &maps, &mut locals);
        assert_eq!(n, 1, "should infer v1 from this.Inner");
        assert_eq!(locals.get("v1"), Some(&10), "v1 should be OtherClass");
    }

    // -----------------------------------------------------------------------
    // Test 37: No inference when RHS type is unknown
    // -----------------------------------------------------------------------

    #[test]
    fn no_inference_when_rhs_unknown() {
        reset_counters();
        // v1 = unknown_var; — unknown_var not in locals, so v1 stays unknown
        let stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Var("unknown_var".into())),
        ];
        let maps = make_test_maps_with_tdi_names();
        let mut locals = make_local_types();
        let n = infer_types_from_stmts(&stmts, &maps, &mut locals);
        assert_eq!(n, 0, "should not infer when RHS is unknown");
        assert!(!locals.contains_key("v1"));
    }

    // -----------------------------------------------------------------------
    // Test 38: Explicit type declaration takes precedence
    // -----------------------------------------------------------------------

    #[test]
    fn explicit_type_decl_takes_precedence() {
        reset_counters();
        // v1 : OtherClass = obj; — explicit type should be used
        let stmts = vec![
            Stmt::Assign("v1".into(), Some("OtherClass".into()), Expr::Var("obj".into())),
        ];
        let maps = make_test_maps_with_tdi_names();
        let mut locals = make_local_types();
        let n = infer_types_from_stmts(&stmts, &maps, &mut locals);
        assert_eq!(n, 1, "should infer from explicit type decl");
        assert_eq!(locals.get("v1"), Some(&10), "v1 should be OtherClass from decl");
    }

    // -----------------------------------------------------------------------
    // Test 39: Byte-width deref resolution (width=1)
    // -----------------------------------------------------------------------

    #[test]
    fn byte_width_deref_resolution() {
        reset_counters();
        // *(byte*)(obj + 184) → obj.Health (same as nint, width should be ignored)
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Var("obj".into())),
            184,
            1, // width=1 (byte)
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve byte-width deref same as nint");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(base, name)) => {
                assert_eq!(name, "Health");
                assert!(matches!(base.as_ref(), Expr::Var(n) if n == "obj"));
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 40: Byte-width deref with this reference
    // -----------------------------------------------------------------------

    #[test]
    fn byte_width_deref_with_this() {
        reset_counters();
        // *(byte*)(this + 24) → this.Name (width=1)
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::This),
            24,
            1, // width=1 (byte)
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "should resolve byte-width deref with this");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(base, name)) => {
                assert_eq!(name, "Name");
                assert!(matches!(base.as_ref(), Expr::This));
            }
            other => panic!("expected Eval(Field(This, _)), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Folded-base-offset resolution (`*(nint*)(obj + 24)`)
    // -----------------------------------------------------------------------

    fn add(var: &str, k: i64) -> Expr {
        Expr::Bin(
            BinOp::Add,
            Box::new(Expr::Var(var.into())),
            Box::new(Expr::Int(k)),
        )
    }

    #[test]
    fn folded_base_offset_resolves_and_rebases_the_receiver() {
        reset_counters();
        // *(nint*)(obj + 16 + 8)  ->  obj.Name  (offset 24 on MyClass)
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Bin(
                BinOp::Add,
                Box::new(add("obj", 16)),
                Box::new(Expr::Int(8)),
            )),
            0,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1, "folded offset must reach the field layout lookup");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(base, name)) => {
                assert_eq!(name, "Name");
                assert!(
                    matches!(base.as_ref(), Expr::Var(v) if v == "obj"),
                    "receiver must be the ROOT object, not the address: {base:?}"
                );
            }
            other => panic!("expected Eval(Field(Var(obj), Name)), got {other:?}"),
        }
    }

    #[test]
    fn folded_base_offset_combines_with_the_deref_addend() {
        reset_counters();
        // `(obj + 8)` addressed at +16 and `((obj + 32) + 8) - 16` both land on
        // MyClass+24.
        let mut stmts = vec![
            Stmt::Eval(Expr::RawDeref(Box::new(add("obj", 8)), 16, 0)),
            Stmt::Eval(Expr::RawDeref(
                Box::new(Expr::Bin(
                    BinOp::Sub,
                    Box::new(Expr::Bin(
                        BinOp::Add,
                        Box::new(add("obj", 32)),
                        Box::new(Expr::Int(8)),
                    )),
                    Box::new(Expr::Int(16)),
                )),
                0,
                0,
            )),
        ];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2);
        for s in &stmts {
            match s {
                Stmt::Eval(Expr::Field(base, name)) => {
                    assert_eq!(name, "Name");
                    assert!(matches!(base.as_ref(), Expr::Var(v) if v == "obj"));
                }
                other => panic!("expected Eval(Field), got {other:?}"),
            }
        }
    }

    #[test]
    fn folded_base_offset_continues_a_field_chain() {
        reset_counters();
        // obj.Inner has type OtherClass (tdi 10); +8 is OtherClass.Count.
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::Bin(
                BinOp::Add,
                Box::new(Expr::Field(
                    Box::new(Expr::Var("obj".into())),
                    "Inner".into(),
                )),
                Box::new(Expr::Int(8)),
            )),
            0,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Field(base, name)) => {
                assert_eq!(name, "Count");
                match base.as_ref() {
                    Expr::Field(recv, inner) => {
                        assert_eq!(inner, "Inner");
                        assert!(matches!(recv.as_ref(), Expr::Var(v) if v == "obj"));
                    }
                    other => panic!("expected Field(obj, Inner), got {other:?}"),
                }
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    #[test]
    fn folded_base_offset_on_an_unknown_receiver_stays_raw() {
        reset_counters();
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(add("mystery", 24)),
            0,
            0,
        ))];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(..))));
    }

    // -----------------------------------------------------------------------
    // IR type map (`infer_ir_type_map`)
    // -----------------------------------------------------------------------

    use crate::recovery::ir::{ArithKind, Insn, NamedTypeRef, Op, Operand};
    use crate::recovery::locals::{LocalInfo, LocalTable};
    use crate::recovery::types::NamedType;
    use crate::recovery::RecoveredIr;
    use std::sync::Arc;

    fn class_type(tdi: i32) -> ManagedType {
        ManagedType::Class(Arc::new(NamedType {
            namespace: String::new(),
            name: format!("T{tdi}"),
            assembly: None,
            type_def_index: tdi,
        }))
    }

    fn local_info(slot: &str, ty: ManagedType) -> LocalInfo {
        LocalInfo {
            slot_origin: slot.to_string(),
            ty,
            is_this: false,
            is_param: None,
            param_comp: None,
            is_return_buffer: false,
            is_incoming_arg: false,
            is_call_clobber: false,
        }
    }

    fn move_local(dst: u32, src: u32) -> Insn {
        Insn::move_(Operand::Local(dst), Operand::Local(src), 0x1000)
    }

    fn add_imm(dst: u32, src: u32, k: i64) -> Insn {
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(dst), Operand::Local(src), Operand::Immediate(k)],
            0x1000,
        )
    }

    fn load(dst: u32, base: u32, addend: i64) -> Insn {
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(dst),
                Operand::Memory {
                    base: Box::new(Operand::Local(base)),
                    index: None,
                    scale: 1,
                    addend,
                    width: 0,
                },
            ],
            0x1000,
        )
    }

    /// End-to-end: a field address (`this + 16`) flows into a load whose
    /// metadata field type then reaches the display-name map, while the
    /// address version itself is NOT reported as an instance type.
    #[test]
    fn ir_map_types_a_load_through_a_field_address() {
        let maps = make_test_maps();
        let mut table = LocalTable::default();
        let mut this = local_info("x0", class_type(5));
        this.is_this = true;
        table.infos.push(this); // li 0 = this
        table.infos.push(local_info("x1", ManagedType::Unknown)); // li 1
        table.infos.push(local_info("x2", ManagedType::Unknown)); // li 2
        table.index_of.insert(0, 0);
        table.index_of.insert(1, 1);
        table.index_of.insert(2, 2);

        let ir = RecoveredIr {
            insns: vec![add_imm(1, 0, 16), load(2, 1, 0)],
            table,
            returns_value: false,
            base_rva: 0,
        };
        let map = infer_ir_type_map(&ir, &maps);
        assert_eq!(map.get("v2"), Some(&10), "MyClass.Inner is OtherClass");
        assert_eq!(
            map.get("v1"),
            None,
            "TdiOff must never be emitted as an instance type"
        );
        assert_eq!(map.get("this"), Some(&5), "the seed still types this");
    }

    /// A parameter's display name (not `vN`) is used for its inferred type.
    #[test]
    fn ir_map_uses_parameter_display_names() {
        let maps = make_test_maps();
        let mut table = LocalTable::default();
        table.param_names.push("other".to_string());
        let mut param = local_info("x1", ManagedType::Unknown);
        param.is_param = Some(0);
        table.infos.push(local_info("x0", ManagedType::Unknown)); // li 0
        table.infos.push(param); // li 1
        table.index_of.insert(0, 0);
        table.index_of.insert(1, 1);

        let newobj = Insn::new(
            Op::NewObj(Some(NamedTypeRef {
                ty: NamedType {
                    namespace: String::new(),
                    name: "T10".into(),
                    assembly: None,
                    type_def_index: 10,
                },
            })),
            vec![Operand::Local(0)],
            0x1000,
        );
        let ir = RecoveredIr {
            insns: vec![newobj, move_local(1, 0)],
            table,
            returns_value: false,
            base_rva: 0,
        };
        let map = infer_ir_type_map(&ir, &maps);
        assert_eq!(map.get("other"), Some(&10));
        assert_eq!(map.get("v0"), Some(&10));
    }

    /// `List<T>` internals: +16 `_items`, +24 Count, +28 `_version`, with the
    /// access width gating each name.
    #[test]
    fn list_internal_fields_are_named_with_their_widths() {
        reset_counters();
        let mut stmts = vec![
            Stmt::Eval(Expr::RawDeref(Box::new(Expr::Var("list".into())), 16, 0)),
            Stmt::Eval(Expr::RawDeref(Box::new(Expr::Var("list".into())), 24, 4)),
            Stmt::Eval(Expr::RawDeref(Box::new(Expr::Var("list".into())), 28, 4)),
            // A 64-bit access at the Count offset is a different object.
            Stmt::Eval(Expr::RawDeref(Box::new(Expr::Var("list".into())), 24, 8)),
        ];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 3, "only the width-valid List internals resolve");
        let text = render_stmts(&stmts);
        assert!(text.contains("list._items"), "{text}");
        assert!(text.contains("list.Count"), "{text}");
        assert!(text.contains("list._version"), "{text}");
    }

    /// Event assignments, index writes and lock/using expressions carry
    /// expressions that must be annotated too. The old statement walk skipped
    /// them, so a resolvable static/class chain used as an event target kept
    /// every `*(nint*)` level (0x198AFB8 had 17 such assignments).
    /// A string literal reached through the GOT used to be an opaque
    /// `void*.ptr_0x…`; 9.7k corpus occurrences were exactly this shape.
    #[test]
    fn a_got_loaded_string_literal_slot_becomes_a_string() {
        reset_counters();
        let mut maps = make_test_maps();
        maps.literal_index_by_addr.insert(0x3053e80, 0);
        maps.literals.insert(0, "hello".into());
        maps.got_addend_by_addr.insert(0x2F6A410, 0x3053e80);
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(Box::new(Expr::Int(0x2F6A410)), 0, 0)),
            0,
            0,
        ))];
        let locals = make_local_types();
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert!(
            matches!(&stmts[0], Stmt::Eval(Expr::Str(s)) if s == "hello"),
            "{stmts:?}"
        );
    }

    /// `il2cpp_array_new_specific` receives the ARRAY class. The slot must
    /// resolve to that class, and `element_class` off it to the element.
    #[test]
    fn a_got_loaded_array_type_slot_resolves_the_class_and_its_element() {
        use crate::recovery::types::NamedType;
        use std::sync::Arc;
        reset_counters();
        let mut maps = make_test_maps();
        let keyframe = NamedType {
            namespace: "UnityEngine".into(),
            name: "Keyframe".into(),
            assembly: None,
            type_def_index: 7515,
        };
        maps.usage_type_by_addr.insert(
            0x3053e80,
            ManagedType::SzArray(Box::new(ManagedType::Class(Arc::new(keyframe)))),
        );
        maps.got_addend_by_addr.insert(0x2F6A410, 0x3053e80);
        let deref = || {
            Expr::RawDeref(
                Box::new(Expr::RawDeref(Box::new(Expr::Int(0x2F6A410)), 0, 0)),
                0,
                0,
            )
        };
        let mut stmts = vec![
            Stmt::Eval(deref()),
            Stmt::Eval(Expr::Field(
                Box::new(deref()),
                "element_class".into(),
            )),
        ];
        let locals = make_local_types();
        apply_with_maps(&mut stmts, &maps, &locals, 0);
        let text = render_stmts(&stmts);
        assert!(
            text.contains("klass(UnityEngine.Keyframe[])"),
            "array class accessor missing: {text}"
        );
        assert!(
            text.contains("typeof(UnityEngine.Keyframe)"),
            "element_class not folded: {text}"
        );
    }

    #[test]
    fn event_assignment_target_chain_is_rewritten() {
        reset_counters();
        let mut stmts = vec![Stmt::EventAssign {
            field: Expr::RawDeref(
                Box::new(Expr::RawDeref(Box::new(Expr::Var("obj".into())), 16, 0)),
                8,
                0,
            ),
            value: Expr::Int(1),
            add: false,
        }];
        let maps = make_test_maps();
        let locals = make_local_types();
        let n = apply_with_maps(&mut stmts, &maps, &locals, 0);
        assert_eq!(n, 2, "both levels of the event target must resolve");
        let text = render_stmts(&stmts);
        assert!(text.contains("obj.Inner.Count"), "{text}");
    }
}

#[cfg(test)]
mod usage_slot_tests {
    use super::*;
    use crate::recovery::types::MethodSigInfo;

    fn spec_ref(name: &str, ty: &str) -> MethodRef {
        MethodRef {
            image_index: 0,
            type_def_index: 12,
            method_index: 3,
            name: name.into(),
            declaring_namespace: "UnityEngine".into(),
            declaring_type_name: ty.into(),
            assembly: String::new(),
            sig: MethodSigInfo {
                ret: ManagedType::Unknown,
                params: vec![],
                is_static: false,
            },
        }
    }

    fn rendered(e: &Expr) -> String {
        let mut s = String::new();
        e.render(&mut s);
        s
    }

    /// A kind-6 usage slot is a METHOD USED AS A VALUE: it must render with the
    /// same spelling Operand::Method gets, not as a raw pointer.
    #[test]
    fn a_method_ref_slot_renders_as_a_method_info() {
        let mut maps = ResolveMaps::default();
        maps.method_spec_index_by_addr.insert(0x3065b18, 250);
        maps.specs_by_index
            .insert(250, spec_ref("FromLongArray", "UnityEngine.AndroidJNISafe"));

        let v = usage_slot_value(0x3065b18, &maps).expect("kind 6 must resolve");
        assert_eq!(rendered(&v), "MethodInfo(\"AndroidJNISafe.FromLongArray\")");
    }

    /// Specs the code-pointer table DOES index live in specs_by_body_rva; the
    /// same slot must resolve through there, or half the kind-6 cells stay raw.
    #[test]
    fn a_spec_indexed_by_its_body_still_resolves() {
        let mut maps = ResolveMaps::default();
        maps.method_spec_index_by_addr.insert(0x3065b18, 7);
        maps.specs_by_body_rva.insert(
            0x47014C,
            vec![(7, spec_ref("CallFloatMethod", "UnityEngine.AndroidJNISafe"))],
        );

        let v = usage_slot_value(0x3065b18, &maps).expect("a body-indexed spec must resolve");
        assert_eq!(rendered(&v), "MethodInfo(\"AndroidJNISafe.CallFloatMethod\")");
    }

    /// A ptr_0x accessor plus an offset walks a table of usage cells. The sum
    /// is looked up EXACTLY, so an offset onto no cell resolves to nothing
    /// instead of borrowing the neighbouring cell's value.
    #[test]
    fn a_ptr_accessor_offset_resolves_only_onto_a_real_cell() {
        let mut maps = ResolveMaps::default();
        maps.literal_index_by_addr.insert(0x3065b18, 0);
        maps.literals.insert(0, "text".into());
        let base = Expr::StaticField("void*".into(), "ptr_0x3065b00".into());

        let hit = usage_slot_value_for_base(&base, 0x18, &maps).expect("cell at base+0x18");
        assert_eq!(rendered(&hit), "\"text\"");
        assert!(
            usage_slot_value_for_base(&base, 0x20, &maps).is_none(),
            "an offset with no cell behind it must not resolve"
        );
    }
}

/// `ILR_RAWDBG=1`: per-site root-cause census of every `RawDeref` that
/// survives annotation. Joins the site with the inferred type and the exact
/// maps the resolver consults, so each line names the REASON it stayed raw:
/// `[raw] rva base_kind off width tdi reason`.
pub fn raw_census(
    stmts: &[Stmt],
    rva: u64,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
) {
    fn kind(e: &Expr) -> &'static str {
        match e {
            Expr::Var(n) if n.starts_with("stack_") => "stack_cell",
            Expr::Var(n)
                if n.len() > 1
                    && n.starts_with('v')
                    && n[1..].bytes().all(|b| b.is_ascii_digit()) =>
            {
                "temp_var"
            }
            Expr::Var(_) => "var",
            Expr::This => "this",
            Expr::Field(..) => "field_chain",
            Expr::StaticField(..) => "static_slot",
            Expr::RawDeref(..) => "double_deref",
            Expr::RawAddr(..) | Expr::AddressOf(_) => "addr_local",
            Expr::Int(_) => "absolute",
            Expr::Call { .. } | Expr::RawCall(..) => "call",
            Expr::Cast(..) => "cast",
            Expr::Bin(..) | Expr::Cmp(..) => "arith",
            Expr::Cond(..) => "cond",
            Expr::Index { .. } => "index",
            _ => "other",
        }
    }

    /// Why `(tdi, off, width)` did not become a field.
    fn field_reason(maps: &ResolveMaps, tdi: i32, off: i64, width: u8) -> String {
        if let Some((f, _)) = maps.field_layouts.get(&(tdi, off)) {
            if is_klass_structure_offset(off, width) {
                // The Il2CppClass table arm runs first and consumes this
                // offset; a real field here can never be named.
                return format!("field_but_klass_offset({f})");
            }
            return format!("field_exists({f})");
        }
        if let Some(p) = layout_path_at(maps, tdi, off, width) {
            return format!("embedded_path({p})");
        }
        if let Some(&w) = maps.field_widths.get(&(tdi, off)) {
            if width != 0 && w != width {
                return format!("width_mismatch(decl{w})");
            }
        }
        // Drill-down: an empty layout (unknown/foreign type) vs a populated
        // layout that simply has nothing at this offset (embedded struct,
        // generic instantiation, or a wrong inferred tdi).
        let n = maps.tdi_fields.get(&tdi).map(|v| v.len()).unwrap_or(0);
        if n == 0 {
            "no_field_tdi_empty".into()
        } else {
            format!("no_field_at_offset(fields={n})")
        }
    }

    /// Innermost variable of a pointer expression.
    fn root_var(e: &Expr) -> Option<&str> {
        match e {
            Expr::Var(n) => Some(n),
            Expr::Field(r, _) => root_var(r),
            Expr::RawDeref(b, _, _) => root_var(b),
            Expr::RawAddr(b, _) | Expr::AddressOf(b) | Expr::Cast(_, b) => root_var(b),
            _ => None,
        }
    }

    fn classify(
        base: &Expr,
        off: i64,
        width: u8,
        maps: &ResolveMaps,
        local_types: &HashMap<String, i32>,
    ) -> (i32, String) {
        match base {
            Expr::Var(name) => {
                if name.starts_with("stack_") {
                    return (-1, "stack_cell_no_type".into());
                }
                match local_types.get(name) {
                    None => (-1, "type_unknown".into()),
                    Some(&tdi) => (tdi, field_reason(maps, tdi, off, width)),
                }
            }
            Expr::RawAddr(inner, _) | Expr::AddressOf(inner) => {
                if let Expr::Var(name) = &**inner {
                    if name.starts_with("stack_") {
                        return (-1, "addr_stack_cell".into());
                    }
                    match local_types.get(name) {
                        None => (-1, "addr_type_unknown".into()),
                        Some(&tdi) => (tdi, format!("addr_{}", field_reason(maps, tdi, off, width))),
                    }
                } else {
                    (-1, "addr_complex".into())
                }
            }
            Expr::RawDeref(inner, ioff, _) => {
                let (tdi, r) = classify(inner, *ioff, 0, maps, local_types);
                (tdi, format!("chain({r})"))
            }
            Expr::Field(recv, _) => {
                let tdi = root_var(recv)
                    .and_then(|n| local_types.get(n).copied())
                    .unwrap_or(-1);
                let r = if tdi >= 0 {
                    "field_chain_typed"
                } else {
                    "field_chain_untyped"
                };
                (tdi, r.into())
            }
            Expr::StaticField(ty, field) => {
                match maps
                    .static_field_types
                    .get(&(ty.clone(), field.clone()))
                {
                    Some(&tdi) => (
                        tdi,
                        format!("static_value_{}", field_reason(maps, tdi, off, width)),
                    ),
                    None => (-1, "static_value_unknown".into()),
                }
            }
            Expr::Int(a) => {
                let addr = (*a as u64).wrapping_add(off as u64);
                let r = if maps.static_field_layouts.contains_key(&addr) {
                    "absolute_static_found"
                } else if maps.klass_to_tdi.contains_key(&addr) {
                    "absolute_klass"
                } else if maps.usage_type_by_addr.contains_key(&addr) {
                    "absolute_usage"
                } else {
                    "absolute_unknown"
                };
                (-1, r.into())
            }
            _ => (-1, "complex_base".into()),
        }
    }

    fn walk_expr(e: &Expr, rva: u64, maps: &ResolveMaps, local_types: &HashMap<String, i32>) {
        if let Expr::RawDeref(b, off, w) = e {
            let (tdi, reason) = classify(b, *off, *w, maps, local_types);
            eprintln!("[raw] {rva:#x} {} {off} {w} {tdi} {reason}", kind(b));
        }
        for c in super::il2cpp_cleanup::expr_children(e) {
            walk_expr(c, rva, maps, local_types);
        }
    }
    fn walk_stmt(s: &Stmt, rva: u64, maps: &ResolveMaps, local_types: &HashMap<String, i32>) {
        super::float_locals::for_each_expr(s, &mut |e| walk_expr(e, rva, maps, local_types));
        if let Stmt::Using { decl, .. } = s {
            walk_stmt(decl, rva, maps, local_types);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            for x in k.iter() {
                walk_stmt(x, rva, maps, local_types);
            }
        }
    }
    for s in stmts {
        walk_stmt(s, rva, maps, local_types);
    }
}
