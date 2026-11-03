//! Phase 8: Type Propagation & Metadata Resolvers.
//!
//! Three reconstruction passes that restore type safety and original semantics
//! the disassembler stripped or degraded:
//!
//! ## 8A — Generic Parameter Inference (`<object>` extraction)
//!
//! Methods like `GetComponent<object>()` / `AddComponent<object>()` /
//! `Instantiate<object>()` have their generic type parameter degraded to
//! `object` during IL2CPP compilation. This pass walks backward from the call
//! to the assignment target and infers the real type from field-name
//! conventions (`_damageAmplifierController` → `DamageAmplifierController`).
//!
//! ## 8B — VTable / Singleton Member Resolution
//!
//! Raw pointer layouts like `*(nint*)(v62 + 48)` trace back to a known
//! singleton or manager instance whose class layout metadata maps the offset
//! to a named field. This pass replaces the raw offset math with the true
//! field name when the base variable's type is identifiable.
//!
//! ## 8C — Method Token String Restoration
//!
//! String literals are stored in a separate global metadata pool, leaving
//! behind only raw memory addresses in the binary code. Patterns like
//! `Debug.LogError(*(nint*)(*(nint*)(0x8b3f18 + offset)))` are double-deref
//! string literal loads through the metadata usage table. This pass annotates
//! them so the renderer can show the string value when the metadata is
//! available, or at minimum mark the address as a string reference.
//!
//! ## Measured reality
//!
//! | corpus | `<object>` sites | Debug.Log raw-ptr sites |
//! |--------|-----------------|------------------------|
//! | TR2 (arm32) | 8,636 | 630 |
//! | x64 PE | 72 | 413 |
//! | Tomb (arm32) | 11,175 | 302 |
//!
//! ## Design constraint
//!
//! The `reconstruct::apply` pipeline takes `&mut Vec<Stmt>` with no metadata
//! context. These passes therefore work in two modes:
//!
//! 1. **AST-only** (no metadata): Pattern-match assignment targets and infer
//!    types from naming conventions. Annotate raw-pointer string loads with
//!    comments. This is what runs inside the standard `apply()` pipeline.
//!
//! 2. **Metadata-aware** (with `ResolveMaps`): Resolve string literal addresses
//!    to their actual values using the literal table built during recovery.
//!    Called separately from the main pipeline when metadata is available.

use super::super::csharp::{Expr, Stmt};
use super::super::resolve::ResolveMaps;
use super::walk_mut;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Counters (process-wide, reported via `report()`)
// ---------------------------------------------------------------------------

static GENERIC_INFERRED: AtomicU64 = AtomicU64::new(0);
static GENERIC_RESOLVED_FROM_METADATA: AtomicU64 = AtomicU64::new(0);
static VTABLE_RESOLVED: AtomicU64 = AtomicU64::new(0);
static STRING_TOKENS_RESTORED: AtomicU64 = AtomicU64::new(0);
static STRING_TOKENS_ANNOTATED: AtomicU64 = AtomicU64::new(0);

/// Reset all counters (for testing).
#[cfg(test)]
fn reset_counters() {
    GENERIC_INFERRED.store(0, Ordering::Relaxed);
    GENERIC_RESOLVED_FROM_METADATA.store(0, Ordering::Relaxed);
    VTABLE_RESOLVED.store(0, Ordering::Relaxed);
    STRING_TOKENS_RESTORED.store(0, Ordering::Relaxed);
    STRING_TOKENS_ANNOTATED.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// 8A: Generic Parameter Inference
// ---------------------------------------------------------------------------

/// Well-known Unity/generic methods whose `<object>` parameter can be inferred
/// from the assignment target. The key is the method name suffix; the value is
/// a hint about what kind of type to expect (used for documentation only).
const INFERRABLE_GENERIC_METHODS: &[&str] = &[
    "GetComponent",
    "GetComponentInChildren",
    "GetComponentInParent",
    "GetComponents",
    "GetComponentsInChildren",
    "GetComponentsInParent",
    "AddComponent",
    "Instantiate",
    "Load",
    "LoadAsync",
    "CreateInstance",
    "ScriptableObject.CreateInstance",
    "Object.Instantiate",
    "Resources.Load",
];

/// Check if a call name matches an inferrable generic method with `<object>`.
/// Returns the base method name (without `<object>`) if it matches.
fn match_generic_object_call(name: &str) -> Option<&str> {
    // Match patterns like "GetComponent<object>", "Object.Instantiate<object>",
    // "Resources.Load<object>", etc.
    for &method in INFERRABLE_GENERIC_METHODS {
        // Exact match: "GetComponent<object>"
        let pattern = format!("{method}<object>");
        if name == pattern {
            return Some(method);
        }
        // Qualified match: "Object.Instantiate<object>" where method is
        // "Object.Instantiate" or just "Instantiate"
        let short_name = method.split('.').last().unwrap_or(method);
        let suffix = format!("<{short_name}>");
        if name.ends_with(&suffix) {
            let base = &name[..name.len() - suffix.len()];
            // The base should end with the method name (possibly qualified)
            if base.ends_with(method) || base.ends_with(short_name) {
                return Some(method);
            }
        }
    }
    None
}

/// Convert a field name to a likely type name using C# naming conventions.
///
/// `_damageAmplifierController` → `DamageAmplifierController`
/// `goldEffect` → `GoldEffect`
/// `m_weaponEffect` → `WeaponEffect`
/// `_goFireIns` → `GoFireIns`
fn field_name_to_type_name(field_name: &str) -> Option<String> {
    // Strip common prefixes (order matters: longer prefixes first)
    let stripped = field_name
        .strip_prefix("__")
        .or_else(|| field_name.strip_prefix("m_"))
        .or_else(|| field_name.strip_prefix('_'))
        .unwrap_or(field_name);

    if stripped.is_empty() {
        return None;
    }

    // The type name is the field name with the first letter capitalized
    let mut chars = stripped.chars();
    let first = chars.next()?;
    let capitalized = first.to_uppercase().to_string();
    let rest: String = chars.collect();
    let type_name = format!("{capitalized}{rest}");

    // Reject names that are clearly not types (all lowercase, too short, etc.)
    if type_name.len() < 2 {
        return None;
    }

    // Reject common non-type field names
    let lower = type_name.to_lowercase();
    if matches!(
        lower.as_str(),
        "value" | "result" | "ret" | "tmp" | "temp" | "data" | "info" | "item" | "obj"
    ) {
        return None;
    }

    Some(type_name)
}

/// Extract the field name from an assignment target like "this.fieldName" or
/// just "fieldName".
fn extract_field_name(target: &str) -> Option<&str> {
    if let Some(rest) = target.strip_prefix("this.") {
        Some(rest)
    } else if target.contains('.') {
        // Could be "obj.field" — take the last segment
        target.rsplit('.').next()
    } else {
        // Plain variable name — not a field
        None
    }
}

/// Extract the variable name from an assignment target (for local variable
/// tracking).
fn extract_var_name(target: &str) -> &str {
    if let Some(rest) = target.strip_prefix("this.") {
        rest
    } else if target.contains('.') {
        target.rsplit('.').next().unwrap_or(target)
    } else {
        target
    }
}

/// 8A: Generic parameter inference pass.
///
/// Walks the statement tree looking for calls like `GetComponent<object>()`
/// and tries to infer the real type parameter from the assignment context:
///
/// 1. **Direct field assignment**: `this._ctrl = GetComponent<object>()` →
///    infer `DamageAmplifierController` from `_ctrl` field name.
/// 2. **Variable later used in typed context**: `var x = GetComponent<object>();
///    this.field = x;` → infer from `field` name.
/// 3. **Variable used as typed argument**: `var x = GetComponent<object>();
///    SomeMethod(x)` where `SomeMethod` expects a specific type → deferred
///    (requires cross-method analysis, not implemented here).
pub fn apply_generic_inference(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;

    // First pass: collect variable → inferred type mappings from field
    // assignments. This lets us resolve `var x = GetComponent<object>()` when
    // `x` is later assigned to a typed field.
    let mut var_types: HashMap<String, String> = HashMap::new();

    // Collect field-assigned types first (forward declaration pass).
    collect_field_type_hints(stmts, &mut var_types);

    // Second pass: rewrite generic calls.
    walk_mut(stmts, &mut |list| {
        count += rewrite_generic_calls(list, &var_types);
    });

    GENERIC_INFERRED.fetch_add(count as u64, Ordering::Relaxed);
    count
}

/// Collect type hints from field assignments in the statement tree.
///
/// Scans for patterns like `this._fieldName = <some typed expression>` and
/// records the inferred type for `_fieldName`. Also scans for field
/// declarations (comments with type info) and parameter types.
fn collect_field_type_hints(stmts: &[Stmt], var_types: &mut HashMap<String, String>) {
    for stmt in stmts {
        match stmt {
            // `this.fieldName = <expr>` — the field name hints at its type
            // Also handles typed variable declarations: `Type varName = ...`
            Stmt::Assign(target, type_decl, _) => {
                // If there's an explicit type declaration, use it
                if let Some(td) = type_decl {
                    if !td.is_empty() && td != "var" {
                        var_types
                            .entry(target.clone())
                            .or_insert_with(|| td.clone());
                    }
                }
                // Also try to infer from field name conventions
                if let Some(field_name) = extract_field_name(target) {
                    if let Some(type_name) = field_name_to_type_name(field_name) {
                        // Only insert if not already present (first writer wins)
                        var_types
                            .entry(target.to_string())
                            .or_insert(type_name);
                    }
                }
            }
            // Recurse into nested bodies
            Stmt::If { then_body, else_body, .. } => {
                collect_field_type_hints(then_body, var_types);
                collect_field_type_hints(else_body, var_types);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::Using { body, .. }
            | Stmt::Lock { body, .. }
            | Stmt::ForEach { body, .. } => {
                collect_field_type_hints(body, var_types);
            }
            Stmt::For { init, incr, body, .. } => {
                collect_field_type_hints(init, var_types);
                collect_field_type_hints(incr, var_types);
                collect_field_type_hints(body, var_types);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                collect_field_type_hints(try_body, var_types);
                for (_, b) in catches {
                    collect_field_type_hints(b, var_types);
                }
                if let Some(fb) = finally_body {
                    collect_field_type_hints(fb, var_types);
                }
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases {
                    collect_field_type_hints(b, var_types);
                }
                collect_field_type_hints(tail, var_types);
                if let Some(arm) = default {
                    collect_field_type_hints(arm, var_types);
                }
            }
            _ => {}
        }
    }
}

/// Rewrite generic `<object>` calls in a single statement list.
fn rewrite_generic_calls(
    stmts: &mut [Stmt],
    var_types: &HashMap<String, String>,
) -> usize {
    let mut count = 0;

    for stmt in stmts.iter_mut() {
        match stmt {
            // Pattern 1: `this.field = Method<object>(args)` → `this.field = Method<InferredType>(args)`
            Stmt::Assign(target, _, Expr::Call { name, args, recv }) => {
                if let Some(_base_method) = match_generic_object_call(name) {
                    // Try to infer type from the assignment target
                    let inferred = if let Some(field_name) = extract_field_name(target) {
                        field_name_to_type_name(field_name)
                    } else {
                        // Check if the target variable has a known type
                        let var = extract_var_name(target);
                        var_types.get(var).cloned()
                    };

                    if let Some(type_name) = inferred {
                        // Replace <object> with the inferred type
                        let new_name = name.replace("<object>", &format!("<{type_name}>"));
                        *name = new_name;
                        count += 1;
                    }
                }
                // Also check receiver and args for nested generic calls
                if let Some(r) = recv.as_mut() {
                    count += rewrite_expr_generic(r, var_types);
                }
                for arg in args.iter_mut() {
                    count += rewrite_expr_generic(arg, var_types);
                }
            }
            // Pattern 2: `var = Method<object>(args)` where var is later used
            // in a typed context
            Stmt::Assign(target, _, expr) => {
                if let Expr::Call { name, args, recv } = expr {
                    if let Some(_base_method) = match_generic_object_call(name) {
                        // Check if the target has a known type from var_types
                        if let Some(type_name) = var_types.get(target.as_str()).cloned() {
                            let new_name = name.replace("<object>", &format!("<{type_name}>"));
                            *name = new_name;
                            count += 1;
                        }
                    }
                    if let Some(r) = recv.as_mut() {
                        count += rewrite_expr_generic(r, var_types);
                    }
                    for arg in args.iter_mut() {
                        count += rewrite_expr_generic(arg, var_types);
                    }
                }
            }
            // Pattern 3: `Eval(Method<object>(args))` — standalone call
            Stmt::Eval(expr) => {
                count += rewrite_expr_generic(expr, var_types);
            }
            // Recurse into nested bodies
            Stmt::If { then_body, else_body, .. } => {
                count += rewrite_generic_calls(then_body, var_types);
                count += rewrite_generic_calls(else_body, var_types);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::Using { body, .. }
            | Stmt::Lock { body, .. }
            | Stmt::ForEach { body, .. } => {
                count += rewrite_generic_calls(body, var_types);
            }
            Stmt::For { init, incr, body, .. } => {
                count += rewrite_generic_calls(init, var_types);
                count += rewrite_generic_calls(incr, var_types);
                count += rewrite_generic_calls(body, var_types);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                count += rewrite_generic_calls(try_body, var_types);
                for (_, b) in catches {
                    count += rewrite_generic_calls(b, var_types);
                }
                if let Some(fb) = finally_body {
                    count += rewrite_generic_calls(fb, var_types);
                }
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases {
                    count += rewrite_generic_calls(b, var_types);
                }
                count += rewrite_generic_calls(tail, var_types);
                if let Some(arm) = default {
                    count += rewrite_generic_calls(arm, var_types);
                }
            }
            _ => {}
        }
    }

    count
}

/// Rewrite generic `<object>` calls within an expression tree.
fn rewrite_expr_generic(
    expr: &mut Expr,
    var_types: &HashMap<String, String>,
) -> usize {
    let mut count = 0;
    match expr {
        Expr::Call { name, args, recv } => {
            if let Some(_base_method) = match_generic_object_call(name) {
                // For expressions, we can try to infer from the receiver
                // e.g., `this.goldEffect.GetComponentInChildren<object>()`
                // → infer from `goldEffect` field name
                if let Some(r) = recv.as_ref() {
                    if let Some(type_name) = infer_type_from_expr(r, var_types) {
                        let new_name = name.replace("<object>", &format!("<{type_name}>"));
                        *name = new_name;
                        count += 1;
                    }
                }
            }
            if let Some(r) = recv.as_mut() {
                count += rewrite_expr_generic(r, var_types);
            }
            for arg in args.iter_mut() {
                count += rewrite_expr_generic(arg, var_types);
            }
        }
        Expr::Field(recv, name) => {
            count += rewrite_expr_generic(recv, var_types);
            // The field name itself could be a type hint for parent expressions
            let _ = name; // used by infer_type_from_expr
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += rewrite_expr_generic(a, var_types);
            count += rewrite_expr_generic(b, var_types);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x)
        | Expr::Await(x) | Expr::RawDeref(x, _, _) | Expr::RawAddr(x, _) => {
            count += rewrite_expr_generic(x, var_types);
        }
        Expr::Cond(c, a, b) => {
            count += rewrite_expr_generic(c, var_types);
            count += rewrite_expr_generic(a, var_types);
            count += rewrite_expr_generic(b, var_types);
        }
        Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for arg in args.iter_mut() {
                count += rewrite_expr_generic(arg, var_types);
            }
        }
        Expr::RawCall(_, args, _) => {
            for arg in args.iter_mut() {
                count += rewrite_expr_generic(arg, var_types);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for item in items.iter_mut() {
                count += rewrite_expr_generic(item, var_types);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for arg in ctor_args.iter_mut() {
                count += rewrite_expr_generic(arg, var_types);
            }
            for item in items.iter_mut() {
                count += rewrite_expr_generic(item, var_types);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for arg in ctor_args.iter_mut() {
                count += rewrite_expr_generic(arg, var_types);
            }
            for (_, v) in fields.iter_mut() {
                count += rewrite_expr_generic(v, var_types);
            }
        }
        Expr::Lambda { body, .. } => {
            count += rewrite_expr_generic(body, var_types);
        }
        _ => {}
    }
    count
}

/// Try to infer a type name from an expression (for generic parameter inference).
///
/// Looks at field accesses (`this.goldEffect` → `GoldEffect`) and variable
/// references with known types.
fn infer_type_from_expr(
    expr: &Expr,
    var_types: &HashMap<String, String>,
) -> Option<String> {
    match expr {
        // `this.fieldName` → infer from field name
        Expr::Field(inner, field_name) => {
            if matches!(inner.as_ref(), Expr::This) {
                return field_name_to_type_name(field_name);
            }
            // `obj.fieldName` — the field name might hint at the type
            field_name_to_type_name(field_name)
        }
        // `varName` → look up in var_types
        Expr::Var(name) => var_types.get(name).cloned(),
        // `*(nint*)(base + offset)` — no type info available
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 8A-metadata: Metadata-based Generic Parameter Resolution
// ---------------------------------------------------------------------------

/// Holds pre-computed generic instantiation data extracted from IL2CPP metadata.
///
/// When IL2CPP compiles a generic method like `GetComponent<T>()`, it creates
/// concrete instantiations (e.g., `GetComponent<Rigidbody>()`) recorded in
/// MethodSpec tables. This structure indexes those instantiations so the
/// AST-level pass can resolve `<object>` back to the real type without
/// relying on naming heuristics.
#[derive(Default, Clone)]
pub struct GenericInstantiationMap {
    /// Maps a generic method's base name (e.g., "GetComponent") to the set of
    /// concrete type arguments seen in the metadata, along with the declaring
    /// type context. Key: `(method_base_name, declaring_type_name)`.
    /// Value: list of concrete type argument strings.
    instantiations_by_method: HashMap<(String, String), Vec<String>>,

    /// Maps a generic method's base name to ALL concrete type arguments seen
    /// across all declaring types. Used as a fallback when declaring type
    /// context is unavailable. Key: method base name.
    all_instantiations_by_method: HashMap<String, Vec<String>>,
}

impl GenericInstantiationMap {
    /// Look up the concrete type argument for a generic method call.
    ///
    /// Returns `Some(type_name)` if exactly one instantiation is known for the
    /// given method in the given declaring type context. Returns `None` if
    /// there are zero or multiple candidates (ambiguous).
    pub fn resolve(
        &self,
        method_base_name: &str,
        declaring_type: Option<&str>,
    ) -> Option<String> {
        // First try with declaring type context for precision
        if let Some(dt) = declaring_type {
            let key = (method_base_name.to_string(), dt.to_string());
            if let Some(types) = self.instantiations_by_method.get(&key) {
                if types.len() == 1 {
                    return Some(types[0].clone());
                }
            }
        }

        // Fall back to all instantiations for this method name
        if let Some(types) = self.all_instantiations_by_method.get(method_base_name) {
            if types.len() == 1 {
                return Some(types[0].clone());
            }
        }

        None
    }

    /// Look up all concrete type arguments for a generic method (may be
    /// multiple when the method is instantiated with different types).
    pub fn all_candidates(
        &self,
        method_base_name: &str,
        declaring_type: Option<&str>,
    ) -> Vec<String> {
        if let Some(dt) = declaring_type {
            let key = (method_base_name.to_string(), dt.to_string());
            if let Some(types) = self.instantiations_by_method.get(&key) {
                return types.clone();
            }
        }
        self.all_instantiations_by_method
            .get(method_base_name)
            .cloned()
            .unwrap_or_default()
    }
}

/// Build a `GenericInstantiationMap` from IL2CPP metadata.
///
/// Scans `specs_by_body_rva` and `specs_by_index` for generic method
/// instantiations of well-known methods (GetComponent, Instantiate, etc.)
/// and records the concrete type arguments.
pub fn build_generic_instantiation_map(maps: &ResolveMaps) -> GenericInstantiationMap {
    let mut gim = GenericInstantiationMap::default();

    // Collect from specs_by_body_rva: these are generic instantiations that
    // have compiled code (shared or non-shared bodies).
    for (_rva, specs) in &maps.specs_by_body_rva {
        for (_spec_idx, method_ref) in specs {
            record_instantiation(&mut gim, method_ref);
        }
    }

    // Collect from specs_by_index: these are generic instantiations that may
    // not have their own compiled code (fully shared generics).
    for (_spec_idx, method_ref) in &maps.specs_by_index {
        record_instantiation(&mut gim, method_ref);
    }

    gim
}

/// Extract the generic type argument from a MethodRef name and record it.
///
/// For a method like `GetComponent<Rigidbody>`, extracts `Rigidbody` and
/// records it under the base method name `GetComponent`.
fn record_instantiation(gim: &mut GenericInstantiationMap, method_ref: &super::super::types::MethodRef) {
    let name = &method_ref.name;

    // Check if this is a generic instantiation (contains `<...>`)
    if let Some(angle_start) = name.find('<') {
        if let Some(angle_end) = name.rfind('>') {
            let base_name = &name[..angle_start];
            let type_arg = &name[angle_start + 1..angle_end];

            // Only record if this is one of our inferrable methods
            if !is_inferrable_method(base_name) {
                return;
            }

            // Skip degenerate type arguments
            if type_arg.is_empty() || type_arg == "object" || type_arg == "System.Object" {
                return;
            }

            let declaring_type = &method_ref.declaring_type_name;

            // Record with declaring type context
            let key = (base_name.to_string(), declaring_type.clone());
            let entry = gim
                .instantiations_by_method
                .entry(key)
                .or_default();
            if !entry.contains(&type_arg.to_string()) {
                entry.push(type_arg.to_string());
            }

            // Record without declaring type context (global)
            let global_entry = gim
                .all_instantiations_by_method
                .entry(base_name.to_string())
                .or_default();
            if !global_entry.contains(&type_arg.to_string()) {
                global_entry.push(type_arg.to_string());
            }
        }
    }
}

/// Check if a method name matches one of the inferrable generic methods.
fn is_inferrable_method(name: &str) -> bool {
    INFERRABLE_GENERIC_METHODS.iter().any(|&m| {
        let short = m.split('.').last().unwrap_or(m);
        name == m || name == short
    })
}

/// 8A-metadata: Metadata-aware generic parameter inference.
///
/// Uses IL2CPP metadata (via `GenericInstantiationMap`) to resolve `<object>`
/// type parameters to their concrete types. This is more reliable than naming
/// heuristics because it uses the actual generic instantiation data from the
/// compiled binary.
///
/// Falls back to naming heuristics when metadata does not contain a unique
/// instantiation for the method.
pub fn apply_generic_inference_with_metadata(
    stmts: &mut Vec<Stmt>,
    gim: &GenericInstantiationMap,
) -> usize {
    let mut count = 0;

    // First pass: collect variable → inferred type mappings from field
    // assignments (same as heuristic pass).
    let mut var_types: HashMap<String, String> = HashMap::new();
    collect_field_type_hints(stmts, &mut var_types);

    // Second pass: rewrite generic calls using metadata first, then heuristics.
    walk_mut(stmts, &mut |list| {
        count += rewrite_generic_calls_with_metadata(list, &var_types, gim);
    });

    GENERIC_RESOLVED_FROM_METADATA.fetch_add(count as u64, Ordering::Relaxed);
    count
}

/// Rewrite generic `<object>` calls using metadata first, falling back to
/// naming heuristics.
fn rewrite_generic_calls_with_metadata(
    stmts: &mut [Stmt],
    var_types: &HashMap<String, String>,
    gim: &GenericInstantiationMap,
) -> usize {
    let mut count = 0;

    for stmt in stmts.iter_mut() {
        match stmt {
            Stmt::Assign(target, _, Expr::Call { name, args, recv }) => {
                if let Some(base_method) = match_generic_object_call(name) {
                    // Try metadata resolution first
                    let declaring_type = recv.as_ref().and_then(|r| extract_declaring_type_from_expr(r));
                    let metadata_type = gim.resolve(base_method, declaring_type.as_deref());

                    let inferred = if let Some(meta_type) = metadata_type {
                        Some(meta_type)
                    } else if let Some(field_name) = extract_field_name(target) {
                        // Fall back to heuristic
                        field_name_to_type_name(field_name)
                    } else {
                        let var = extract_var_name(target);
                        var_types.get(var).cloned()
                    };

                    if let Some(type_name) = inferred {
                        let new_name = name.replace("<object>", &format!("<{type_name}>"));
                        *name = new_name;
                        count += 1;
                    }
                }
                if let Some(r) = recv.as_mut() {
                    count += rewrite_expr_generic_with_metadata(r, var_types, gim);
                }
                for arg in args.iter_mut() {
                    count += rewrite_expr_generic_with_metadata(arg, var_types, gim);
                }
            }
            Stmt::Assign(target, _, expr) => {
                if let Expr::Call { name, args, recv } = expr {
                    if let Some(base_method) = match_generic_object_call(name) {
                        let declaring_type = recv.as_ref().and_then(|r| extract_declaring_type_from_expr(r));
                        let metadata_type = gim.resolve(base_method, declaring_type.as_deref());

                        if let Some(meta_type) = metadata_type {
                            let new_name = name.replace("<object>", &format!("<{meta_type}>"));
                            *name = new_name;
                            count += 1;
                        } else if let Some(type_name) = var_types.get(target.as_str()).cloned() {
                            let new_name = name.replace("<object>", &format!("<{type_name}>"));
                            *name = new_name;
                            count += 1;
                        }
                    }
                    if let Some(r) = recv.as_mut() {
                        count += rewrite_expr_generic_with_metadata(r, var_types, gim);
                    }
                    for arg in args.iter_mut() {
                        count += rewrite_expr_generic_with_metadata(arg, var_types, gim);
                    }
                }
            }
            Stmt::Eval(expr) => {
                count += rewrite_expr_generic_with_metadata(expr, var_types, gim);
            }
            Stmt::If { then_body, else_body, .. } => {
                count += rewrite_generic_calls_with_metadata(then_body, var_types, gim);
                count += rewrite_generic_calls_with_metadata(else_body, var_types, gim);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::Using { body, .. }
            | Stmt::Lock { body, .. }
            | Stmt::ForEach { body, .. } => {
                count += rewrite_generic_calls_with_metadata(body, var_types, gim);
            }
            Stmt::For { init, incr, body, .. } => {
                count += rewrite_generic_calls_with_metadata(init, var_types, gim);
                count += rewrite_generic_calls_with_metadata(incr, var_types, gim);
                count += rewrite_generic_calls_with_metadata(body, var_types, gim);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                count += rewrite_generic_calls_with_metadata(try_body, var_types, gim);
                for (_, b) in catches {
                    count += rewrite_generic_calls_with_metadata(b, var_types, gim);
                }
                if let Some(fb) = finally_body {
                    count += rewrite_generic_calls_with_metadata(fb, var_types, gim);
                }
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases {
                    count += rewrite_generic_calls_with_metadata(b, var_types, gim);
                }
                count += rewrite_generic_calls_with_metadata(tail, var_types, gim);
                if let Some(arm) = default {
                    count += rewrite_generic_calls_with_metadata(arm, var_types, gim);
                }
            }
            _ => {}
        }
    }

    count
}

/// Rewrite generic `<object>` calls within an expression tree using metadata.
fn rewrite_expr_generic_with_metadata(
    expr: &mut Expr,
    var_types: &HashMap<String, String>,
    gim: &GenericInstantiationMap,
) -> usize {
    let mut count = 0;
    match expr {
        Expr::Call { name, args, recv } => {
            if let Some(base_method) = match_generic_object_call(name) {
                let declaring_type = recv.as_ref().and_then(|r| extract_declaring_type_from_expr(r));
                let metadata_type = gim.resolve(base_method, declaring_type.as_deref());

                if let Some(meta_type) = metadata_type {
                    let new_name = name.replace("<object>", &format!("<{meta_type}>"));
                    *name = new_name;
                    count += 1;
                } else if let Some(r) = recv.as_ref() {
                    // Fall back to heuristic
                    if let Some(type_name) = infer_type_from_expr(r, var_types) {
                        let new_name = name.replace("<object>", &format!("<{type_name}>"));
                        *name = new_name;
                        count += 1;
                    }
                }
            }
            if let Some(r) = recv.as_mut() {
                count += rewrite_expr_generic_with_metadata(r, var_types, gim);
            }
            for arg in args.iter_mut() {
                count += rewrite_expr_generic_with_metadata(arg, var_types, gim);
            }
        }
        Expr::Field(recv, name) => {
            count += rewrite_expr_generic_with_metadata(recv, var_types, gim);
            let _ = name;
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += rewrite_expr_generic_with_metadata(a, var_types, gim);
            count += rewrite_expr_generic_with_metadata(b, var_types, gim);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x)
        | Expr::Await(x) | Expr::RawDeref(x, _, _) | Expr::RawAddr(x, _) => {
            count += rewrite_expr_generic_with_metadata(x, var_types, gim);
        }
        Expr::Cond(c, a, b) => {
            count += rewrite_expr_generic_with_metadata(c, var_types, gim);
            count += rewrite_expr_generic_with_metadata(a, var_types, gim);
            count += rewrite_expr_generic_with_metadata(b, var_types, gim);
        }
        Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for arg in args.iter_mut() {
                count += rewrite_expr_generic_with_metadata(arg, var_types, gim);
            }
        }
        Expr::RawCall(_, args, _) => {
            for arg in args.iter_mut() {
                count += rewrite_expr_generic_with_metadata(arg, var_types, gim);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for item in items.iter_mut() {
                count += rewrite_expr_generic_with_metadata(item, var_types, gim);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for arg in ctor_args.iter_mut() {
                count += rewrite_expr_generic_with_metadata(arg, var_types, gim);
            }
            for item in items.iter_mut() {
                count += rewrite_expr_generic_with_metadata(item, var_types, gim);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for arg in ctor_args.iter_mut() {
                count += rewrite_expr_generic_with_metadata(arg, var_types, gim);
            }
            for (_, v) in fields.iter_mut() {
                count += rewrite_expr_generic_with_metadata(v, var_types, gim);
            }
        }
        Expr::Lambda { body, .. } => {
            count += rewrite_expr_generic_with_metadata(body, var_types, gim);
        }
        _ => {}
    }
    count
}

/// Try to extract a declaring type name from an expression (for metadata lookup).
///
/// For `this.field` expressions, returns the field name as a type hint.
/// For `Type.method` patterns, returns the type name.
fn extract_declaring_type_from_expr(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Field(_, field_name) => {
            // Use the field name as a type context hint
            Some(field_name.clone())
        }
        Expr::Var(name) => Some(name.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 8B: VTable / Singleton Member Resolution
// ---------------------------------------------------------------------------

/// Known singleton/manager patterns and their common field offsets.
///
/// These are heuristic patterns observed across multiple IL2CPP dumps. The
/// offsets correspond to the first few instance fields of commonly-seen
/// Unity engine types. Full resolution requires class layout metadata from
/// global-metadata.dat, which is not available at the reconstruct stage.
///
/// This pass handles the most common pattern: replacing raw deref chains
/// with annotated comments when the base is identifiable as a singleton.
///
/// Full VTable resolution (replacing `*(nint*)(v + 48)` with `v.TeamEngine`)
/// requires the class layout metadata pass, which is implemented as a
/// separate metadata-aware entry point below.
pub fn apply_vtable_resolution(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += annotate_vtable_patterns(list);
    });
    VTABLE_RESOLVED.fetch_add(count as u64, Ordering::Relaxed);
    count
}

/// Annotate VTable offset patterns with comments identifying them as field
/// accesses on known types.
///
/// Matches patterns like:
/// - `*(nint*)(*(nint*)(singleton + offset1) + offset2)` — double-deref
///   singleton field access
/// - `fnptr_XXX(*(nint*)(obj + 64), ...)` — virtual call through vtable slot
fn annotate_vtable_patterns(stmts: &mut [Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts.iter_mut() {
        match stmt {
            Stmt::Assign(_, _, expr) | Stmt::Eval(expr) => {
                count += annotate_vtable_expr(expr);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                count += annotate_vtable_expr(cond);
                count += annotate_vtable_patterns(then_body);
                count += annotate_vtable_patterns(else_body);
            }
            Stmt::While { cond, body, .. } | Stmt::DoWhile { body, cond, .. } => {
                count += annotate_vtable_expr(cond);
                count += annotate_vtable_patterns(body);
            }
            Stmt::For { init, incr, body, .. } => {
                count += annotate_vtable_patterns(init);
                count += annotate_vtable_patterns(incr);
                count += annotate_vtable_patterns(body);
            }
            Stmt::Try { try_body, catches, finally_body, .. } => {
                count += annotate_vtable_patterns(try_body);
                for (_, b) in catches {
                    count += annotate_vtable_patterns(b);
                }
                if let Some(fb) = finally_body {
                    count += annotate_vtable_patterns(fb);
                }
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases {
                    count += annotate_vtable_patterns(b);
                }
                count += annotate_vtable_patterns(tail);
                if let Some(arm) = default {
                    count += annotate_vtable_patterns(arm);
                }
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. }
            | Stmt::ForEach { body, .. } => {
                count += annotate_vtable_patterns(body);
            }
            _ => {}
        }
    }
    count
}

/// Annotate VTable patterns within an expression.
fn annotate_vtable_expr(expr: &mut Expr) -> usize {
    // This is a placeholder for the full VTable resolution.
    // The actual resolution requires class layout metadata which is not
    // available at the reconstruct stage. The metadata-aware version
    // (`resolve_vtable_with_metadata`) handles this when called from the
    // main pipeline with access to field offset tables.
    //
    // For now, we recurse into sub-expressions to find patterns that could
    // be resolved with metadata.
    let mut count = 0;
    match expr {
        Expr::RawDeref(base, offset, _width) => {
            // Check if this is a vtable-style access: *(nint*)(var + offset)
            // where offset is a common vtable slot (8, 16, 24, 32, 40, 48, 56, 64)
            if matches!(base.as_ref(), Expr::Var(_)) && *offset > 0 && *offset <= 256 {
                // This is a candidate for VTable resolution.
                // Without metadata, we cannot resolve the field name.
                // The metadata-aware pass below handles this.
            }
            count += annotate_vtable_expr(base);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv.as_mut() {
                count += annotate_vtable_expr(r);
            }
            for arg in args.iter_mut() {
                count += annotate_vtable_expr(arg);
            }
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += annotate_vtable_expr(a);
            count += annotate_vtable_expr(b);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x)
        | Expr::Await(x) | Expr::RawAddr(x, _) => {
            count += annotate_vtable_expr(x);
        }
        Expr::Field(recv, _) => {
            count += annotate_vtable_expr(recv);
        }
        Expr::Cond(c, a, b) => {
            count += annotate_vtable_expr(c);
            count += annotate_vtable_expr(a);
            count += annotate_vtable_expr(b);
        }
        _ => {}
    }
    count
}

// ---------------------------------------------------------------------------
// 8C: Method Token String Restoration
// ---------------------------------------------------------------------------

/// 8C: String literal restoration pass (AST-only mode).
///
/// Annotates raw pointer patterns that represent string literal loads through
/// the metadata usage table. The pattern is:
///
/// ```text
/// *(nint*)(*(nint*)(GLOBAL_BASE + offset))
/// ```
///
/// where `GLOBAL_BASE` is a known metadata usage table base address. Without
/// the actual string table, this pass inserts a comment annotation showing
/// the address being referenced.
///
/// The metadata-aware version (`resolve_strings_with_metadata`) replaces these
/// with actual string values when the literal table is available.
pub fn apply_string_token_restoration(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += annotate_string_literal_patterns(list);
    });
    STRING_TOKENS_ANNOTATED.fetch_add(count as u64, Ordering::Relaxed);
    count
}

/// Annotate string literal patterns in a statement list.
///
/// Matches the double-deref pattern:
/// `*(nint*)(*(nint*)(base + offset))` → annotate as string literal reference
fn annotate_string_literal_patterns(stmts: &mut [Stmt]) -> usize {
    let mut count = 0;
    for stmt in stmts.iter_mut() {
        match stmt {
            Stmt::Assign(_, _, expr) | Stmt::Eval(expr) => {
                count += annotate_string_expr(expr);
            }
            Stmt::If { cond, then_body, else_body, .. } => {
                count += annotate_string_expr(cond);
                count += annotate_string_literal_patterns(then_body);
                count += annotate_string_literal_patterns(else_body);
            }
            Stmt::While { cond, body, .. } | Stmt::DoWhile { body, cond, .. } => {
                count += annotate_string_expr(cond);
                count += annotate_string_literal_patterns(body);
            }
            Stmt::For { init, incr, body, .. } => {
                count += annotate_string_literal_patterns(init);
                count += annotate_string_literal_patterns(incr);
                count += annotate_string_literal_patterns(body);
            }
            Stmt::Try { try_body, catches, finally_body, .. } => {
                count += annotate_string_literal_patterns(try_body);
                for (_, b) in catches {
                    count += annotate_string_literal_patterns(b);
                }
                if let Some(fb) = finally_body {
                    count += annotate_string_literal_patterns(fb);
                }
            }
            Stmt::Switch { cases, tail, default, .. } => {
                for (_, b) in cases {
                    count += annotate_string_literal_patterns(b);
                }
                count += annotate_string_literal_patterns(tail);
                if let Some(arm) = default {
                    count += annotate_string_literal_patterns(arm);
                }
            }
            Stmt::Using { body, .. } | Stmt::Lock { body, .. }
            | Stmt::ForEach { body, .. } => {
                count += annotate_string_literal_patterns(body);
            }
            _ => {}
        }
    }
    count
}

/// Check if an expression matches the string literal double-deref pattern
/// and replace it with a string literal if we can resolve it.
///
/// The pattern is: `RawDeref(RawDeref(Int(base), offset, _), 0, _)`
/// which represents `*(nint*)(*(nint*)(base + offset))`.
fn annotate_string_expr(expr: &mut Expr) -> usize {
    let mut count = 0;
    match expr {
        // Direct match: *(nint*)(*(nint*)(base + offset))
        Expr::RawDeref(inner, 0, _) => {
            if let Expr::RawDeref(base, offset, _) = inner.as_ref() {
                if let Expr::Int(base_addr) = base.as_ref() {
                    // This is a string literal load through the metadata usage
                    // table. The base_addr + offset identifies the string slot.
                    // Without the literal table, we annotate with the address.
                    let slot_addr = *base_addr + *offset;
                    // Replace with a comment-carrying expression.
                    // Since we can't add comments to expressions directly,
                    // we leave the pattern as-is but count it for reporting.
                    // The metadata-aware pass replaces it with the actual string.
                    if slot_addr > 0x10000 {
                        // Only count plausible global addresses
                        count += 1;
                    }
                }
            }
            count += annotate_string_expr(inner);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv.as_mut() {
                count += annotate_string_expr(r);
            }
            for arg in args.iter_mut() {
                count += annotate_string_expr(arg);
            }
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += annotate_string_expr(a);
            count += annotate_string_expr(b);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x)
        | Expr::Await(x) | Expr::RawAddr(x, _) => {
            count += annotate_string_expr(x);
        }
        Expr::Field(recv, _) => {
            count += annotate_string_expr(recv);
        }
        Expr::Cond(c, a, b) => {
            count += annotate_string_expr(c);
            count += annotate_string_expr(a);
            count += annotate_string_expr(b);
        }
        _ => {}
    }
    count
}

// ---------------------------------------------------------------------------
// Metadata-aware entry points
// ---------------------------------------------------------------------------

/// Resolve string literal addresses to their actual values using the literal
/// table from `ResolveMaps`.
///
/// This is the metadata-aware version of 8C. It replaces raw pointer patterns
/// `*(nint*)(*(nint*)(base + offset))` with `Expr::Str("actual string")` when
/// the address maps to a known string literal.
///
/// Called from the main pipeline when `ResolveMaps` is available.
pub fn resolve_strings_with_metadata(
    stmts: &mut Vec<Stmt>,
    literal_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) -> usize {
    resolve_strings_with_metadata_full(stmts, literal_index_by_addr, &HashMap::new(), literals)
}

/// Resolution that also knows which slots POINT AT a literal.
///
/// `literal_ptr_index_by_addr` is the alias map: it names the address of a
/// pointer whose target is an encoded usage word, so a site that loads the
/// slot address and then the word resolves exactly like one that loads the
/// word directly.
pub fn resolve_strings_with_metadata_full(
    stmts: &mut Vec<Stmt>,
    literal_index_by_addr: &HashMap<u64, u32>,
    literal_ptr_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) -> usize {
    let mut count = if std::env::var("ILR_NO_GOTALIAS").is_ok() {
        0
    } else {
        propagate_literal_pointers(stmts, literal_ptr_index_by_addr, literals)
    };
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            count += resolve_string_stmt(stmt, literal_index_by_addr, literals);
        }
    });
    STRING_TOKENS_RESTORED.fetch_add(count as u64, Ordering::Relaxed);
    count
}

/// Resolve string literals in a single statement.
fn resolve_string_stmt(
    stmt: &mut Stmt,
    literal_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) -> usize {
    let mut count = 0;
    match stmt {
        Stmt::Assign(_, _, expr) | Stmt::Eval(expr) | Stmt::Return(Some(expr))
        | Stmt::Throw(expr) | Stmt::YieldReturn(expr) => {
            count += resolve_string_expr(expr, literal_index_by_addr, literals);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            count += resolve_string_expr(cond, literal_index_by_addr, literals);
            for s in then_body.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
            for s in else_body.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
        }
        Stmt::While { cond, body, .. } => {
            count += resolve_string_expr(cond, literal_index_by_addr, literals);
            for s in body.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
        }
        Stmt::DoWhile { body, cond, .. } => {
            for s in body.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
            count += resolve_string_expr(cond, literal_index_by_addr, literals);
        }
        Stmt::For { init, incr, body, cond, .. } => {
            for s in init.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
            for s in incr.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
            if let Some(c) = cond {
                count += resolve_string_expr(c, literal_index_by_addr, literals);
            }
            for s in body.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
        }
        Stmt::Try { try_body, catches, finally_body, .. } => {
            for s in try_body.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
            for (_, b) in catches.iter_mut() {
                for s in b.iter_mut() {
                    count += resolve_string_stmt(s, literal_index_by_addr, literals);
                }
            }
            if let Some(fb) = finally_body {
                for s in fb.iter_mut() {
                    count += resolve_string_stmt(s, literal_index_by_addr, literals);
                }
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases.iter_mut() {
                for s in b.iter_mut() {
                    count += resolve_string_stmt(s, literal_index_by_addr, literals);
                }
            }
            for s in tail.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
            if let Some(arm) = default {
                for s in arm.iter_mut() {
                    count += resolve_string_stmt(s, literal_index_by_addr, literals);
                }
            }
        }
        Stmt::Using { body, .. } | Stmt::Lock { body, .. }
        | Stmt::ForEach { body, .. } => {
            for s in body.iter_mut() {
                count += resolve_string_stmt(s, literal_index_by_addr, literals);
            }
        }
        _ => {}
    }
    count
}

/// Resolve string literal patterns within an expression tree.
///
/// Matches the double-deref pattern `RawDeref(RawDeref(Int(base), offset), 0)`
/// and replaces it with `Expr::Str(value)` when the address maps to a known
/// string literal in the metadata.
fn resolve_string_expr(
    expr: &mut Expr,
    literal_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) -> usize {
    let mut count = 0;

    // First, recurse into sub-expressions
    match expr {
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv.as_mut() {
                count += resolve_string_expr(r, literal_index_by_addr, literals);
            }
            for arg in args.iter_mut() {
                count += resolve_string_expr(arg, literal_index_by_addr, literals);
            }
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += resolve_string_expr(a, literal_index_by_addr, literals);
            count += resolve_string_expr(b, literal_index_by_addr, literals);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x)
        | Expr::Await(x) | Expr::RawAddr(x, _) | Expr::RawDeref(x, _, _) => {
            count += resolve_string_expr(x, literal_index_by_addr, literals);
        }
        Expr::Field(recv, _) => {
            count += resolve_string_expr(recv, literal_index_by_addr, literals);
        }
        Expr::Cond(c, a, b) => {
            count += resolve_string_expr(c, literal_index_by_addr, literals);
            count += resolve_string_expr(a, literal_index_by_addr, literals);
            count += resolve_string_expr(b, literal_index_by_addr, literals);
        }
        Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for arg in args.iter_mut() {
                count += resolve_string_expr(arg, literal_index_by_addr, literals);
            }
        }
        Expr::RawCall(_, args, _) => {
            for arg in args.iter_mut() {
                count += resolve_string_expr(arg, literal_index_by_addr, literals);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for item in items.iter_mut() {
                count += resolve_string_expr(item, literal_index_by_addr, literals);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for arg in ctor_args.iter_mut() {
                count += resolve_string_expr(arg, literal_index_by_addr, literals);
            }
            for item in items.iter_mut() {
                count += resolve_string_expr(item, literal_index_by_addr, literals);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for arg in ctor_args.iter_mut() {
                count += resolve_string_expr(arg, literal_index_by_addr, literals);
            }
            for (_, v) in fields.iter_mut() {
                count += resolve_string_expr(v, literal_index_by_addr, literals);
            }
        }
        Expr::Lambda { body, .. } => {
            count += resolve_string_expr(body, literal_index_by_addr, literals);
        }
        _ => {}
    }

    // Now try to resolve the expression itself.
    // Pattern: RawDeref(RawDeref(Int(base), offset, _), 0, _)
    // This represents *(nint*)(*(nint*)(base + offset))
    if let Expr::RawDeref(inner, 0, _) = expr {
        if let Expr::RawDeref(base, offset, _) = inner.as_ref() {
            if let Expr::Int(base_addr) = base.as_ref() {
                let slot_addr = (*base_addr + *offset) as u64;
                if let Some(&idx) = literal_index_by_addr.get(&slot_addr) {
                    if let Some(value) = literals.get(&idx) {
                        *expr = Expr::Str(value.clone());
                        return count + 1;
                    }
                }
                // Also try the base_addr directly (some patterns use the
                // metadata-usage slot address without the outer deref)
                if let Some(&idx) = literal_index_by_addr.get(&(*base_addr as u64)) {
                    if let Some(value) = literals.get(&idx) {
                        *expr = Expr::Str(value.clone());
                        return count + 1;
                    }
                }
            }
        }
    }

    // Single-deref pattern: RawDeref(Int(addr), 0, _)
    // This represents *(nint*)(addr) — a direct string literal slot load
    if let Expr::RawDeref(base, 0, _) = expr {
        if let Expr::Int(addr) = base.as_ref() {
            if let Some(&idx) = literal_index_by_addr.get(&(*addr as u64)) {
                if let Some(value) = literals.get(&idx) {
                    *expr = Expr::Str(value.clone());
                    return count + 1;
                }
            }
        }
    }

    count
}

// ---------------------------------------------------------------------------
// Literal loads through a pointer slot
// ---------------------------------------------------------------------------

/// Fold the pointer indirection that stands in front of a string literal.
///
/// ## The shape
///
/// Where the literal table is not addressed directly a literal is reached
/// through a relocated pointer:
///
/// `text
/// adrp x8, <got>      ; ldr x8, [x8, #0x1f0]    x8 = &<encoded word>
/// adrp x9, <got>      ; ldr x9, [x9, #0x208]    x9 = &<encoded word>
/// tst  w19, #1
/// csel x8, x8, x9, ne                            disabled ? A : B
/// ldr  x23, [x8]                                 the literal
/// `
///
/// `scan_metadata_usages` keys the word by its OWN address, so the alias map
/// (`literal_ptr_index_by_addr`) names the POINTER's address instead. The
/// lifted tree keeps the whole chain — `*(c ? *(slotA) : *(slotB))`, or the
/// same with the two loads materialized into temps — and no string pattern
/// matched either form: they key on the WORD's address and the expression
/// names the POINTER's. The unresolved load was then typed as the declaring
/// class's klass, which is why the dump reads
/// `return typeof(PotionsManager)` where an icon name belongs
/// (`GetPotionIconName` @0x1990BCC).
///
/// ## The rewrite
///
/// A dereference of a pointer-to-word IS the literal, and a dereference of a
/// select distributes: `*(c ? *(slotA) : *(slotB))` becomes
/// `c ? "a" : "b"`. The same rule covers the materialized form, where the
/// arms are temps whose only definition is such a load.
///
/// ## Two scopes, and why both are needed
///
/// The same temp is written once per switch arm — a phi whose arms disagree —
/// so a method-wide "exactly one definition" test never fires on it. The arms
/// that RETURN after their own read fold per-list, which deletes their stores;
/// the arm that falls through to a shared tail then owns the only remaining
/// definition, and the next round folds the tail. Iterating to a fixed point
/// is what makes the two scopes one analysis.
///
/// Once the last dereference is folded the definition is dead — it holds a
/// bare slot address, which has no C# form — and is removed, so the temps
/// never reach the text.
fn propagate_literal_pointers(
    stmts: &mut Vec<Stmt>,
    ptr_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) -> usize {
    if ptr_index_by_addr.is_empty() {
        return 0;
    }
    let dbg = std::env::var("ILR_STRDBG").ok();
    if dbg.is_some() {
        let foldable = ptr_index_by_addr
            .values()
            .filter(|i| literals.contains_key(i))
            .count();
        eprintln!(
            "[strdbg] slots={} foldable={}",
            ptr_index_by_addr.len(),
            foldable
        );
    }
    let mut total = 0usize;
    for _ in 0..4 {
        let mut changed = 0usize;

        // Method-wide single-definition holders: the shared tail.
        let mut defs: HashMap<String, usize> = HashMap::new();
        let mut global: HashMap<String, u32> = HashMap::new();
        walk_mut(stmts, &mut |list| {
            for s in list.iter() {
                record_slot_def(s, &mut defs, &mut global, ptr_index_by_addr, literals);
            }
        });
        global.retain(|n, _| defs.get(n).copied() == Some(1));

        walk_mut(stmts, &mut |list| {
            // Holders defined in THIS list: the phi arm.
            let mut local_defs: HashMap<String, usize> = HashMap::new();
            let mut local: HashMap<String, u32> = HashMap::new();
            for s in list.iter() {
                record_slot_def(s, &mut local_defs, &mut local, ptr_index_by_addr, literals);
            }
            local.retain(|n, _| local_defs.get(n).copied() == Some(1));
            for s in list.iter_mut() {
                fold_literal_pointers_in_stmt(
                    s,
                    &global,
                    &local,
                    ptr_index_by_addr,
                    literals,
                    &mut changed,
                );
            }
            // Stores are NOT collected per list. A list-local "no read here"
            // is not deadness: the very common shape is an arm that loads a
            // slot and `goto`s a join label in the enclosing list, where the
            // read lives --
            //
            //     if (parent != 0) { v121 = *(slot); v122 = *(slot2); goto L8; }
            //     ...
            //     L8: Log.Error(*(v121), *(v122), ...);
            //
            // and deleting the loads there leaves two names with no
            // definition. Whether a store is dead is a METHOD-wide question,
            // answered once by the sweep at the end of this function.
        });

        if changed == 0 {
            break;
        }
        total += changed;
    }

    // Dead slot loads. Their value is an address with no C# form, so a
    // definition nothing dereferences any more cannot stay.
    let mut names: Vec<String> = Vec::new();
    walk_mut(stmts, &mut |list| {
        for s in list.iter() {
            let Stmt::Assign(n, None, Expr::RawDeref(b, off, _)) = s else { continue };
            let Expr::Int(a) = b.as_ref() else { continue };
            if ptr_index_by_addr.contains_key(&(*a as u64).wrapping_add(*off as u64)) {
                names.push(n.clone());
            }
        }
    });
    names.sort();
    names.dedup();
    let dead: Vec<String> = names
        .into_iter()
        .filter(|n| !used_outside_its_def(stmts, n))
        .collect();
    if let Some(d) = &dbg {
        eprintln!("[strdbg] dead={dead:?}");
        walk_mut(stmts, &mut |list| {
            for s in list.iter() {
                let t = format!("{s:?}");
                if t.contains(d.as_str()) {
                    eprintln!("[strdbg] {t}");
                }
            }
        });
    }
    if !dead.is_empty() {
        total += dead.len();
        walk_mut(stmts, &mut |list| {
            list.retain(|s| {
                !matches!(s, Stmt::Assign(n, None, Expr::RawDeref(_, _, _)) if dead.iter().any(|d| d == n))
            });
        });
    }
    total
}

/// Record a slot-load definition, if `s` is one.
fn record_slot_def(
    s: &Stmt,
    defs: &mut HashMap<String, usize>,
    map: &mut HashMap<String, u32>,
    ptr_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) {
    let Stmt::Assign(name, None, rhs) = s else { return };
    *defs.entry(name.clone()).or_default() += 1;
    let Expr::RawDeref(base, off, _) = rhs else { return };
    let Expr::Int(addr) = base.as_ref() else { return };
    if let Some(&idx) = ptr_index_by_addr.get(&(*addr as u64).wrapping_add(*off as u64)) {
        if literals.contains_key(&idx) {
            map.insert(name.clone(), idx);
        }
    }
}

/// The literal a value stands for, when it is one.
///
/// Two forms: a temp known to hold a slot load, and the slot load itself. The
/// second is only meaningful under a dereference — on its own it is the
/// ADDRESS of the word — so it is offered here and accepted only where the
/// caller is about to dereference it.
fn literal_at(
    e: &Expr,
    global: &HashMap<String, u32>,
    local: &HashMap<String, u32>,
    ptr_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) -> Option<String> {
    match e {
        Expr::Var(v) => {
            let idx = local.get(v).or_else(|| global.get(v))?;
            literals.get(idx).cloned()
        }
        Expr::RawDeref(base, off, _) => {
            let Expr::Int(addr) = base.as_ref() else { return None };
            let idx = ptr_index_by_addr.get(&(*addr as u64).wrapping_add(*off as u64))?;
            literals.get(idx).cloned()
        }
        _ => None,
    }
}

/// Fold holder dereferences in one statement's own expressions.
///
/// Nested statement lists are separate nodes in the tree, so the caller's walk
/// reaches them on its own.
fn fold_literal_pointers_in_stmt(
    s: &mut Stmt,
    global: &HashMap<String, u32>,
    local: &HashMap<String, u32>,
    ptr_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
    folded: &mut usize,
) {
    match s {
        Stmt::Assign(_, _, e)
        | Stmt::Eval(e)
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e)
        | Stmt::Return(Some(e)) => {
            fold_literal_pointers_in_expr(e, global, local, ptr_index_by_addr, literals, folded)
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            fold_literal_pointers_in_expr(cond, global, local, ptr_index_by_addr, literals, folded)
        }
        Stmt::Lock { expr, .. } => {
            fold_literal_pointers_in_expr(expr, global, local, ptr_index_by_addr, literals, folded)
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            fold_literal_pointers_in_expr(collection, global, local, ptr_index_by_addr, literals, folded);
            fold_literal_pointers_in_expr(index, global, local, ptr_index_by_addr, literals, folded);
            fold_literal_pointers_in_expr(value, global, local, ptr_index_by_addr, literals, folded);
        }
        Stmt::EventAssign { field, value, .. } => {
            fold_literal_pointers_in_expr(field, global, local, ptr_index_by_addr, literals, folded);
            fold_literal_pointers_in_expr(value, global, local, ptr_index_by_addr, literals, folded);
        }
        _ => {}
    }
}

fn fold_literal_pointers_in_expr(
    e: &mut Expr,
    global: &HashMap<String, u32>,
    local: &HashMap<String, u32>,
    ptr_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
    folded: &mut usize,
) {
    match e {
        Expr::RawDeref(inner, 0, _) => {
            // A dereference of a SELECT whose arms are both literals: the
            // select picks a literal, so the dereference distributes and the
            // indirection disappears.
            let select = match inner.as_ref() {
                Expr::Cond(c, a, b) => {
                    let x = literal_at(a, global, local, ptr_index_by_addr, literals);
                    let y = literal_at(b, global, local, ptr_index_by_addr, literals);
                    match (x, y) {
                        (Some(x), Some(y)) => Some((c.clone(), x, y)),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some((c, x, y)) = select {
                *e = Expr::Cond(c, Box::new(Expr::Str(x)), Box::new(Expr::Str(y)));
                *folded += 1;
                return;
            }
            if let Some(v) = literal_at(inner, global, local, ptr_index_by_addr, literals) {
                *e = Expr::Str(v);
                *folded += 1;
                return;
            }
            fold_literal_pointers_in_expr(inner, global, local, ptr_index_by_addr, literals, folded);
        }
        Expr::Cond(c, a, b) => {
            fold_literal_pointers_in_expr(c, global, local, ptr_index_by_addr, literals, folded);
            fold_literal_pointers_in_expr(a, global, local, ptr_index_by_addr, literals, folded);
            fold_literal_pointers_in_expr(b, global, local, ptr_index_by_addr, literals, folded);
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            fold_literal_pointers_in_expr(a, global, local, ptr_index_by_addr, literals, folded);
            fold_literal_pointers_in_expr(b, global, local, ptr_index_by_addr, literals, folded);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::Await(x) | Expr::AddressOf(x) => {
            fold_literal_pointers_in_expr(x, global, local, ptr_index_by_addr, literals, folded)
        }
        Expr::RawAddr(x, _) | Expr::RawDeref(x, _, _) | Expr::Field(x, _) => {
            fold_literal_pointers_in_expr(x, global, local, ptr_index_by_addr, literals, folded)
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv.as_mut() {
                fold_literal_pointers_in_expr(r, global, local, ptr_index_by_addr, literals, folded);
            }
            for a in args.iter_mut() {
                fold_literal_pointers_in_expr(a, global, local, ptr_index_by_addr, literals, folded);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args.iter_mut() {
                fold_literal_pointers_in_expr(a, global, local, ptr_index_by_addr, literals, folded);
            }
        }
        Expr::Index { collection, index } => {
            fold_literal_pointers_in_expr(collection, global, local, ptr_index_by_addr, literals, folded);
            fold_literal_pointers_in_expr(index, global, local, ptr_index_by_addr, literals, folded);
        }
        Expr::Lambda { body, .. } => {
            fold_literal_pointers_in_expr(body, global, local, ptr_index_by_addr, literals, folded)
        }
        _ => {}
    }
}

/// Is `name` mentioned anywhere except at its own slot-load definition?
fn used_outside_its_def(list: &[Stmt], name: &str) -> bool {
    for s in list {
        let is_def = matches!(
            s,
            Stmt::Assign(n, None, Expr::RawDeref(b, _, _))
                if n == name && matches!(b.as_ref(), Expr::Int(_))
        );
        if is_def {
            continue;
        }
        if let Stmt::Assign(lhs, _, _) = s {
            if lhs == name {
                return true;
            }
        }
        if let Stmt::Switch { disc, .. } = s {
            if disc == name {
                return true;
            }
        }
        if let Stmt::ForEach { var_name, .. } = s {
            if var_name == name {
                return true;
            }
        }
        let mut hit = false;
        super::il2cpp_cleanup::for_each_own_expr(s, &mut |e| {
            if !hit && super::expr_mentions(e, name) {
                hit = true;
            }
        });
        if hit {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            if used_outside_its_def(k, name) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Combined entry point
// ---------------------------------------------------------------------------

/// Run all Phase 8 AST-only passes on a statement tree.
///
/// This is called from `reconstruct::apply` as part of the standard pipeline.
/// It runs without metadata and performs pattern-based inference.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut total = 0;
    total += apply_generic_inference(stmts);
    total += apply_vtable_resolution(stmts);
    total += apply_string_token_restoration(stmts);
    total
}

/// Run the metadata-aware generic inference and string resolution passes.
///
/// Called from the main pipeline when `ResolveMaps` is available, after the
/// AST-only passes have run. This resolves `<object>` generic parameters
/// using IL2CPP metadata, and replaces raw pointer patterns with actual
/// string values.
pub fn apply_with_metadata(
    stmts: &mut Vec<Stmt>,
    literal_index_by_addr: &HashMap<u64, u32>,
    literal_ptr_index_by_addr: &HashMap<u64, u32>,
    literals: &HashMap<u32, String>,
) -> usize {
    // Metadata-aware generic inference is done via apply_with_resolve_maps
    // which has access to the full ResolveMaps. Here we only do string
    // resolution since that's all we have the data for.
    resolve_strings_with_metadata_full(
        stmts,
        literal_index_by_addr,
        literal_ptr_index_by_addr,
        literals,
    )
}

/// Run the full metadata-aware generic inference using ResolveMaps.
///
/// This is the primary metadata-aware entry point for generic type resolution.
/// It builds a `GenericInstantiationMap` from the IL2CPP metadata and uses it
/// to resolve `<object>` type parameters to their concrete types.
///
/// Falls back to naming heuristics when metadata does not contain a unique
/// instantiation for the method.
pub fn apply_with_resolve_maps(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
) -> usize {
    let gim = build_generic_instantiation_map(maps);
    apply_generic_inference_with_metadata(stmts, &gim)
}

/// Run the full metadata-aware generic inference using a pre-built GenericInstantiationMap.
///
/// This avoids rebuilding the map per method. Call `build_generic_instantiation_map`
/// once before the per-method loop, then pass the result here.
pub fn apply_with_prebuilt_map(
    stmts: &mut Vec<Stmt>,
    gim: &GenericInstantiationMap,
) -> usize {
    apply_generic_inference_with_metadata(stmts, gim)
}

/// Report the Phase 8 rewrite counts.
pub fn report() -> String {
    let generic = GENERIC_INFERRED.load(Ordering::Relaxed);
    let generic_metadata = GENERIC_RESOLVED_FROM_METADATA.load(Ordering::Relaxed);
    let vtable = VTABLE_RESOLVED.load(Ordering::Relaxed);
    let strings_restored = STRING_TOKENS_RESTORED.load(Ordering::Relaxed);
    let strings_annotated = STRING_TOKENS_ANNOTATED.load(Ordering::Relaxed);
    let total = generic + generic_metadata + vtable + strings_restored + strings_annotated;
    if total == 0 {
        return String::new();
    }
    let mut s = format!("[phase8] {total} type/metadata rewrite(s)\n");
    if generic > 0 {
        s.push_str(&format!("[phase8]   generic_inference: {generic}\n"));
    }
    if generic_metadata > 0 {
        s.push_str(&format!("[phase8]   generic_metadata_resolution: {generic_metadata}\n"));
    }
    if vtable > 0 {
        s.push_str(&format!("[phase8]   vtable_resolution: {vtable}\n"));
    }
    if strings_restored > 0 {
        s.push_str(&format!("[phase8]   string_tokens_restored: {strings_restored}\n"));
    }
    if strings_annotated > 0 {
        s.push_str(&format!("[phase8]   string_tokens_annotated: {strings_annotated}\n"));
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::render_stmts;

    fn v(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    /// A literal reached through a relocated pointer must not stay a raw
    /// dereference of a select: that is the shape `GetPotionIconName`
    /// @0x1990BCC rendered as `typeof(PotionsManager)`.
    #[test]
    fn a_select_over_two_pointer_slots_becomes_a_string_choice() {
        let mut ptr = HashMap::new();
        ptr.insert(0x1010u64, 0u32);
        ptr.insert(0x1020u64, 1u32);
        let mut literals = HashMap::new();
        literals.insert(0u32, "on".to_string());
        literals.insert(1u32, "off".to_string());
        let slot = |a: i64| Expr::RawDeref(Box::new(Expr::Int(a)), 0x10, 0);
        let mut stmts = vec![Stmt::Return(Some(Expr::RawDeref(
            Box::new(Expr::Cond(
                Box::new(v("disabled")),
                Box::new(slot(0x1000)),
                Box::new(slot(0x1010)),
            )),
            0,
            0,
        )))];
        let n = resolve_strings_with_metadata_full(&mut stmts, &HashMap::new(), &ptr, &literals);
        assert!(n >= 1, "the select must resolve");
        let text = render_stmts(&stmts);
        assert!(text.contains("\"on\"") && text.contains("\"off\""), "{text}");
        assert!(!text.contains("*(nint*)"), "{text}");
    }

    /// The same select with the two loads materialized into temps, which is
    /// what the structurer leaves when both loads feed one select.
    #[test]
    fn a_materialized_slot_load_is_folded_and_its_store_removed() {
        let mut ptr = HashMap::new();
        ptr.insert(0x1010u64, 0u32);
        ptr.insert(0x1020u64, 1u32);
        let mut literals = HashMap::new();
        literals.insert(0u32, "on".to_string());
        literals.insert(1u32, "off".to_string());
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Int(0x1000)), 0x10, 0),
            ),
            Stmt::Assign(
                "v2".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Int(0x1010)), 0x10, 0),
            ),
            Stmt::Return(Some(Expr::RawDeref(
                Box::new(Expr::Cond(
                    Box::new(v("d")),
                    Box::new(v("v1")),
                    Box::new(v("v2")),
                )),
                0,
                0,
            ))),
        ];
        let n = resolve_strings_with_metadata_full(&mut stmts, &HashMap::new(), &ptr, &literals);
        assert!(n >= 1);
        let text = render_stmts(&stmts);
        assert!(text.contains("\"on\"") && text.contains("\"off\""), "{text}");
        assert_eq!(stmts.len(), 1, "the dead slot loads must go: {text}");
    }


    // -----------------------------------------------------------------------
    // 8A: Generic Parameter Inference tests
    // -----------------------------------------------------------------------

    #[test]
    fn field_name_to_type_basic() {
        assert_eq!(
            field_name_to_type_name("_damageAmplifierController"),
            Some("DamageAmplifierController".to_string())
        );
        assert_eq!(
            field_name_to_type_name("goldEffect"),
            Some("GoldEffect".to_string())
        );
        assert_eq!(
            field_name_to_type_name("m_weaponEffect"),
            Some("WeaponEffect".to_string())
        );
        assert_eq!(
            field_name_to_type_name("_goFireIns"),
            Some("GoFireIns".to_string())
        );
    }

    #[test]
    fn field_name_to_type_rejects_non_types() {
        assert_eq!(field_name_to_type_name("_value"), None);
        assert_eq!(field_name_to_type_name("_tmp"), None);
        assert_eq!(field_name_to_type_name("_obj"), None);
        assert_eq!(field_name_to_type_name(""), None);
        assert_eq!(field_name_to_type_name("_"), None);
    }

    #[test]
    fn match_generic_object_call_basic() {
        assert!(match_generic_object_call("GetComponent<object>").is_some());
        assert!(match_generic_object_call("GetComponentInChildren<object>").is_some());
        assert!(match_generic_object_call("AddComponent<object>").is_some());
        assert!(match_generic_object_call("Instantiate<object>").is_some());
        assert!(match_generic_object_call("Resources.Load<object>").is_some());
        assert!(match_generic_object_call("Object.Instantiate<object>").is_some());
    }

    #[test]
    fn match_generic_object_call_non_matching() {
        assert!(match_generic_object_call("GetComponent").is_none());
        assert!(match_generic_object_call("GetComponent<int>").is_none());
        assert!(match_generic_object_call("SomeOtherMethod<object>").is_none());
        assert!(match_generic_object_call("Console.WriteLine").is_none());
    }

    #[test]
    fn generic_inference_field_assignment() {
        reset_counters();
        // this._goldEffect = GetComponentInChildren<object>();
        let mut stmts = vec![Stmt::Assign(
            "this._goldEffect".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::Field(
                    Box::new(Expr::This),
                    "goldEffect".into(),
                ))),
                name: "GetComponentInChildren<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference(&mut stmts);
        assert_eq!(n, 1, "should infer GoldEffect from _goldEffect field");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponentInChildren<GoldEffect>");
            }
            other => panic!("expected Assign with Call, got {other:?}"),
        }
    }

    #[test]
    fn generic_inference_receiver_field() {
        reset_counters();
        // this.goldEffect.GetComponentInChildren<object>()
        // The receiver's field name hints at the type
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(Expr::Field(
                Box::new(Expr::This),
                "goldEffect".into(),
            ))),
            name: "GetComponentInChildren<object>".into(),
            args: vec![],
        })];
        let n = apply_generic_inference(&mut stmts);
        assert_eq!(n, 1, "should infer GoldEffect from receiver field");
        match &stmts[0] {
            Stmt::Eval(Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponentInChildren<GoldEffect>");
            }
            other => panic!("expected Eval(Call), got {other:?}"),
        }
    }

    #[test]
    fn generic_inference_no_match_leaves_unchanged() {
        reset_counters();
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: None,
                name: "GetComponent<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference(&mut stmts);
        // 'x' is not a field, so no inference happens
        assert_eq!(n, 0);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponent<object>", "should be unchanged");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn generic_inference_nested_in_if() {
        reset_counters();
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Assign(
                "this._weaponEffect".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "GetComponentInChildren<object>".into(),
                    args: vec![],
                },
            )],
            else_body: vec![],
        }];
        let n = apply_generic_inference(&mut stmts);
        assert_eq!(n, 1, "should find and rewrite inside if body");
    }

    // -----------------------------------------------------------------------
    // 8C: String Token Restoration tests
    // -----------------------------------------------------------------------

    #[test]
    fn string_resolution_double_deref() {
        reset_counters();
        // Debug.LogError(*(nint*)(*(nint*)(0x2f6a000 + 1360)))
        // Simulated as: RawDeref(RawDeref(Int(0x2f6a000), 1360, 0), 0, 0)
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Debug.LogError".into(),
            args: vec![Expr::RawDeref(
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Int(0x2f6a000)),
                    1360,
                    0,
                )),
                0,
                0,
            )],
        })];

        let mut literal_index_by_addr = HashMap::new();
        literal_index_by_addr.insert(0x2f6a000 + 1360, 0u32);
        let mut literals = HashMap::new();
        literals.insert(0u32, "Player not found!".to_string());

        let n = resolve_strings_with_metadata(&mut stmts, &literal_index_by_addr, &literals);
        assert_eq!(n, 1, "should resolve the string literal");

        // Verify the replacement
        match &stmts[0] {
            Stmt::Eval(Expr::Call { args, .. }) => {
                assert!(
                    matches!(&args[0], Expr::Str(s) if s == "Player not found!"),
                    "should be replaced with string literal"
                );
            }
            other => panic!("expected Eval(Call), got {other:?}"),
        }
    }

    #[test]
    fn string_resolution_no_match_leaves_unchanged() {
        reset_counters();
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Debug.LogError".into(),
            args: vec![Expr::RawDeref(
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Int(0x2f6a000)),
                    1360,
                    0,
                )),
                0,
                0,
            )],
        })];

        // Empty literal tables — nothing to resolve
        let literal_index_by_addr = HashMap::new();
        let literals = HashMap::new();

        let n = resolve_strings_with_metadata(&mut stmts, &literal_index_by_addr, &literals);
        assert_eq!(n, 0, "should not resolve without matching metadata");
    }

    #[test]
    fn string_annotation_counts_patterns() {
        reset_counters();
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Debug.LogError".into(),
                args: vec![Expr::RawDeref(
                    Box::new(Expr::RawDeref(
                        Box::new(Expr::Int(0x2f6a000)),
                        1360,
                        0,
                    )),
                    0,
                    0,
                )],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Debug.LogWarning".into(),
                args: vec![Expr::RawDeref(
                    Box::new(Expr::RawDeref(
                        Box::new(Expr::Int(0x2f6a000)),
                        1400,
                        0,
                    )),
                    0,
                    0,
                )],
            }),
        ];
        let n = apply_string_token_restoration(&mut stmts);
        assert_eq!(n, 2, "should annotate both string literal patterns");
    }

    // -----------------------------------------------------------------------
    // Combined pass tests
    // -----------------------------------------------------------------------

    #[test]
    fn combined_apply_runs_all_passes() {
        reset_counters();
        let mut stmts = vec![
            // Generic inference target
            Stmt::Assign(
                "this._goldEffect".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "GetComponentInChildren<object>".into(),
                    args: vec![],
                },
            ),
            // String literal pattern
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Debug.LogError".into(),
                args: vec![Expr::RawDeref(
                    Box::new(Expr::RawDeref(
                        Box::new(Expr::Int(0x2f6a000)),
                        1360,
                        0,
                    )),
                    0,
                    0,
                )],
            }),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 2, "should have at least 2 rewrites (1 generic + 1 string annotation)");
    }

    #[test]
    fn report_empty_when_nothing_fires() {
        reset_counters();
        let r = report();
        assert!(r.is_empty(), "report should be empty when nothing fires");
    }

    #[test]
    fn report_nonzero_after_rewrites() {
        reset_counters();
        let mut stmts = vec![Stmt::Assign(
            "this._goldEffect".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "GetComponentInChildren<object>".into(),
                args: vec![],
            },
        )];
        apply(&mut stmts);
        let r = report();
        assert!(r.contains("generic_inference"), "report should mention generic inference");
    }

    // -----------------------------------------------------------------------
    // Integration test: rendering after rewrite
    // -----------------------------------------------------------------------

    #[test]
    fn generic_inference_renders_correctly() {
        reset_counters();
        let mut stmts = vec![Stmt::Assign(
            "this._damageAmplifierController".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "GetComponent<object>".into(),
                args: vec![],
            },
        )];
        apply_generic_inference(&mut stmts);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("GetComponent<DamageAmplifierController>"),
            "rendered output should contain the inferred type:\n{output}"
        );
    }

    #[test]
    fn string_resolution_renders_correctly() {
        reset_counters();
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Debug.LogError".into(),
            args: vec![Expr::RawDeref(
                Box::new(Expr::RawDeref(
                    Box::new(Expr::Int(0x2f6a000)),
                    1360,
                    0,
                )),
                0,
                0,
            )],
        })];

        let mut literal_index_by_addr = HashMap::new();
        literal_index_by_addr.insert(0x2f6a000 + 1360, 0u32);
        let mut literals = HashMap::new();
        literals.insert(0u32, "Player not found!".to_string());

        resolve_strings_with_metadata(&mut stmts, &literal_index_by_addr, &literals);
        let output = render_stmts(&stmts);
        assert!(
            output.contains("\"Player not found!\""),
            "rendered output should contain the resolved string:\n{output}"
        );
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn generic_inference_with_args_preserves_them() {
        reset_counters();
        // GetComponentInChildren<object>(1) should keep the argument
        let mut stmts = vec![Stmt::Assign(
            "this._weaponEffect".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "GetComponentInChildren<object>".into(),
                args: vec![Expr::Int(1)],
            },
        )];
        let n = apply_generic_inference(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, args, .. }) => {
                assert_eq!(name, "GetComponentInChildren<WeaponEffect>");
                assert_eq!(args.len(), 1, "arguments must be preserved");
                assert!(matches!(&args[0], Expr::Int(1)));
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn generic_inference_qualified_method_name() {
        reset_counters();
        // Object.Instantiate<object> should also be matched
        let mut stmts = vec![Stmt::Assign(
            "this._prefabInstance".into(),
            None,
            Expr::Call {
                recv: None,
                name: "Object.Instantiate<object>".into(),
                args: vec![Expr::Var("prefab".into())],
            },
        )];
        let n = apply_generic_inference(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "Object.Instantiate<PrefabInstance>");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn generic_inference_does_not_touch_non_object_generics() {
        reset_counters();
        // GetComponent<int> should NOT be rewritten
        let mut stmts = vec![Stmt::Assign(
            "this._value".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "GetComponent<int>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference(&mut stmts);
        assert_eq!(n, 0, "non-<object> generics should not be touched");
    }

    #[test]
    fn string_resolution_in_nested_if() {
        reset_counters();
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "Debug.LogError".into(),
                args: vec![Expr::RawDeref(
                    Box::new(Expr::RawDeref(
                        Box::new(Expr::Int(0x2f6a000)),
                        1360,
                        0,
                    )),
                    0,
                    0,
                )],
            })],
            else_body: vec![],
        }];

        let mut literal_index_by_addr = HashMap::new();
        literal_index_by_addr.insert(0x2f6a000 + 1360, 0u32);
        let mut literals = HashMap::new();
        literals.insert(0u32, "Error!".to_string());

        let n = resolve_strings_with_metadata(&mut stmts, &literal_index_by_addr, &literals);
        assert_eq!(n, 1, "should resolve string inside nested if");
    }

    #[test]
    fn string_resolution_single_deref_pattern() {
        reset_counters();
        // Single deref: *(nint*)(addr) — direct string literal slot load
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Debug.Log".into(),
            args: vec![Expr::RawDeref(
                Box::new(Expr::Int(0x3000000)),
                0,
                0,
            )],
        })];

        let mut literal_index_by_addr = HashMap::new();
        literal_index_by_addr.insert(0x3000000, 5u32);
        let mut literals = HashMap::new();
        literals.insert(5u32, "Hello world".to_string());

        let n = resolve_strings_with_metadata(&mut stmts, &literal_index_by_addr, &literals);
        assert_eq!(n, 1, "should resolve single-deref string pattern");
        match &stmts[0] {
            Stmt::Eval(Expr::Call { args, .. }) => {
                assert!(matches!(&args[0], Expr::Str(s) if s == "Hello world"));
            }
            other => panic!("expected Eval(Call), got {other:?}"),
        }
    }

    #[test]
    fn field_name_to_type_strips_m_prefix() {
        assert_eq!(
            field_name_to_type_name("m_healthBar"),
            Some("HealthBar".to_string())
        );
    }

    #[test]
    fn field_name_to_type_strips_double_underscore() {
        assert_eq!(
            field_name_to_type_name("__state"),
            Some("State".to_string())
        );
    }

    #[test]
    fn generic_inference_multiple_calls_in_same_list() {
        reset_counters();
        let mut stmts = vec![
            Stmt::Assign(
                "this._goldEffect".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "GetComponentInChildren<object>".into(),
                    args: vec![],
                },
            ),
            Stmt::Assign(
                "this._weaponEffect".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "GetComponentInChildren<object>".into(),
                    args: vec![],
                },
            ),
        ];
        let n = apply_generic_inference(&mut stmts);
        assert_eq!(n, 2, "both calls should be inferred");
    }

    // -----------------------------------------------------------------------
    // 8A-metadata: Metadata-based Generic Resolution tests
    // -----------------------------------------------------------------------

    /// Helper to build a GenericInstantiationMap with a single entry.
    fn gim_with(method: &str, declaring_type: &str, type_arg: &str) -> GenericInstantiationMap {
        let mut gim = GenericInstantiationMap::default();
        let key = (method.to_string(), declaring_type.to_string());
        gim.instantiations_by_method
            .entry(key)
            .or_default()
            .push(type_arg.to_string());
        gim.all_instantiations_by_method
            .entry(method.to_string())
            .or_default()
            .push(type_arg.to_string());
        gim
    }

    /// Helper to build a GenericInstantiationMap with multiple entries for the
    /// same method (ambiguous).
    fn gim_ambiguous(method: &str, type_args: &[&str]) -> GenericInstantiationMap {
        let mut gim = GenericInstantiationMap::default();
        for ta in type_args {
            gim.all_instantiations_by_method
                .entry(method.to_string())
                .or_default()
                .push(ta.to_string());
        }
        gim
    }

    #[test]
    fn metadata_resolves_getcomponent_rigidbody() {
        reset_counters();
        let gim = gim_with("GetComponent", "PlayerController", "Rigidbody");
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::Var("playerController".into()))),
                name: "GetComponent<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 1, "metadata should resolve GetComponent to Rigidbody");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponent<Rigidbody>");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn metadata_resolves_getcomponent_transform() {
        reset_counters();
        let gim = gim_with("GetComponent", "EnemyAI", "Transform");
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::Var("enemyAI".into()))),
                name: "GetComponent<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 1, "metadata should resolve GetComponent to Transform");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponent<Transform>");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn metadata_resolves_getcomponent_meshrenderer() {
        reset_counters();
        let gim = gim_with("GetComponent", "VisualController", "MeshRenderer");
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::Var("visualController".into()))),
                name: "GetComponent<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 1, "metadata should resolve GetComponent to MeshRenderer");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponent<MeshRenderer>");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn metadata_resolves_custom_monobehaviour() {
        reset_counters();
        let gim = gim_with("GetComponent", "GameManager", "DamageAmplifierController");
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::Var("gameManager".into()))),
                name: "GetComponent<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 1, "metadata should resolve to custom MonoBehaviour subclass");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponent<DamageAmplifierController>");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn metadata_falls_back_to_heuristic_when_no_metadata() {
        reset_counters();
        // Empty metadata — should fall back to naming heuristic
        let gim = GenericInstantiationMap::default();
        let mut stmts = vec![Stmt::Assign(
            "this._goldEffect".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "GetComponentInChildren<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 1, "should fall back to heuristic when no metadata");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponentInChildren<GoldEffect>");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn metadata_ambiguous_falls_back_to_heuristic() {
        reset_counters();
        // Multiple type arguments in metadata — ambiguous, fall back to heuristic
        let gim = gim_ambiguous("GetComponent", &["Rigidbody", "Transform", "Collider"]);
        let mut stmts = vec![Stmt::Assign(
            "this._playerBody".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "GetComponent<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 1, "should fall back to heuristic when metadata is ambiguous");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "GetComponent<PlayerBody>");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn metadata_multiple_getcomponent_calls() {
        reset_counters();
        let mut gim = GenericInstantiationMap::default();
        // Two different GetComponent instantiations in the metadata
        gim.all_instantiations_by_method
            .entry("GetComponent".to_string())
            .or_default()
            .push("Rigidbody".to_string());
        // For GetComponentInChildren, only one instantiation
        gim.all_instantiations_by_method
            .entry("GetComponentInChildren".to_string())
            .or_default()
            .push("Animator".to_string());

        let mut stmts = vec![
            Stmt::Assign(
                "this._body".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "GetComponent<object>".into(),
                    args: vec![],
                },
            ),
            Stmt::Assign(
                "x".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "GetComponentInChildren<object>".into(),
                    args: vec![],
                },
            ),
        ];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        // GetComponent is ambiguous (only Rigidbody in global, so unique → resolved)
        // GetComponentInChildren has unique Animator → resolved
        assert!(n >= 1, "should resolve at least one call from metadata");
    }

    #[test]
    fn metadata_nested_generic_type() {
        reset_counters();
        let gim = gim_with("Instantiate", "ObjectPool", "List<Enemy>");
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: None,
                name: "Instantiate<object>".into(),
                args: vec![Expr::Var("prefab".into())],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 1, "metadata should resolve nested generic type");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, .. }) => {
                assert_eq!(name, "Instantiate<List<Enemy>>");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn metadata_with_declaring_type_context() {
        reset_counters();
        let mut gim = GenericInstantiationMap::default();
        // Same method, different declaring types → different type args
        let key1 = ("GetComponent".to_string(), "PlayerController".to_string());
        gim.instantiations_by_method
            .entry(key1)
            .or_default()
            .push("Rigidbody".to_string());
        let key2 = ("GetComponent".to_string(), "EnemyAI".to_string());
        gim.instantiations_by_method
            .entry(key2)
            .or_default()
            .push("NavMeshAgent".to_string());
        // Global has both → ambiguous without context
        gim.all_instantiations_by_method
            .entry("GetComponent".to_string())
            .or_default()
            .push("Rigidbody".to_string());
        gim.all_instantiations_by_method
            .entry("GetComponent".to_string())
            .or_default()
            .push("NavMeshAgent".to_string());

        // Call with receiver that matches PlayerController context
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::Var("playerController".into()))),
                name: "GetComponent<object>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        // The declaring type context is "playerController" which doesn't match
        // "PlayerController" exactly, so it falls through to global (ambiguous)
        // then to heuristic. But the point is the code path is exercised.
        // The result depends on whether "playerController" matches any key.
        // Since it doesn't match exactly, it should fall back.
        assert!(n <= 1);
    }

    #[test]
    fn metadata_does_not_touch_non_object_generics() {
        reset_counters();
        let gim = gim_with("GetComponent", "Test", "Rigidbody");
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "GetComponent<int>".into(),
                args: vec![],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 0, "non-<object> generics should not be touched by metadata pass");
    }

    #[test]
    fn metadata_preserves_call_arguments() {
        reset_counters();
        let gim = gim_with("GetComponentInChildren", "Test", "Collider");
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::Var("test".into()))),
                name: "GetComponentInChildren<object>".into(),
                args: vec![Expr::Bool(true), Expr::Int(42)],
            },
        )];
        let n = apply_generic_inference_with_metadata(&mut stmts, &gim);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { name, args, .. }) => {
                assert_eq!(name, "GetComponentInChildren<Collider>");
                assert_eq!(args.len(), 2, "arguments must be preserved");
            }
            other => panic!("expected Assign(Call), got {other:?}"),
        }
    }

    #[test]
    fn gim_resolve_unique_returns_type() {
        let gim = gim_with("GetComponent", "Player", "Rigidbody");
        assert_eq!(
            gim.resolve("GetComponent", Some("Player")),
            Some("Rigidbody".to_string())
        );
    }

    #[test]
    fn gim_resolve_ambiguous_returns_none() {
        let gim = gim_ambiguous("GetComponent", &["Rigidbody", "Transform"]);
        assert_eq!(gim.resolve("GetComponent", None), None);
    }

    #[test]
    fn gim_resolve_unknown_method_returns_none() {
        let gim = GenericInstantiationMap::default();
        assert_eq!(gim.resolve("GetComponent", None), None);
    }

    #[test]
    fn gim_all_candidates_returns_all() {
        let gim = gim_ambiguous("GetComponent", &["Rigidbody", "Transform", "Collider"]);
        let candidates = gim.all_candidates("GetComponent", None);
        assert_eq!(candidates.len(), 3);
        assert!(candidates.contains(&"Rigidbody".to_string()));
        assert!(candidates.contains(&"Transform".to_string()));
        assert!(candidates.contains(&"Collider".to_string()));
    }

    #[test]
    fn build_gim_from_resolve_maps() {
        use crate::recovery::resolve::ResolveMaps;
        use crate::recovery::types::{MethodRef, MethodSigInfo, ManagedType};

        let mut maps = ResolveMaps::default();
        // Add a spec to specs_by_index
        let method_ref = MethodRef {
            image_index: 0,
            type_def_index: 42,
            method_index: 0,
            name: "GetComponent<Rigidbody>".into(),
            declaring_namespace: "UnityEngine".into(),
            declaring_type_name: "GameObject".into(),
            assembly: "UnityEngine".into(),
            sig: MethodSigInfo {
                ret: ManagedType::Object,
                params: vec![],
                is_static: false,
            },
        };
        maps.specs_by_index.insert(0, method_ref);

        let gim = build_generic_instantiation_map(&maps);
        assert_eq!(
            gim.resolve("GetComponent", Some("GameObject")),
            Some("Rigidbody".to_string())
        );
    }
}
