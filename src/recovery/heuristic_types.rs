//! Heuristic type inference from usage patterns.
//!
//! When metadata-based type propagation (`locals::propagate_types`) cannot
//! determine a local's type, we can still infer it from HOW the local is used:
//!
//! - **Pattern 1 (method call receiver):** `v.Method()` where `Method` is
//!   declared on exactly one type → `v` is that type.
//! - **Pattern 2 (field access receiver):** `v.Field` where `Field` is declared
//!   on exactly one type → `v` is that type.
//! - **Pattern 3 (Unity API):** Well-known Unity methods have known return
//!   types (`GetComponent<T>()` → T, `Find()` → GameObject, etc.).
//!
//! ## Design principles
//!
//! - **High confidence only.** When a method/field name is ambiguous (shared by
//!   multiple types), we skip it rather than guess wrong. Better to type 10k
//!   locals correctly than 100k locals with 50% accuracy.
//! - **Never override metadata.** If `propagate_types` already determined a
//!   type, the heuristic pass leaves it alone.
//! - **Works on the C# AST.** The patterns are statement-level shapes
//!   (`Expr::Call` with a receiver, `Expr::Field` on a variable), so this runs
//!   after the statement tree is built but before field annotation consumes the
//!   type map.
//!
//! ## Integration
//!
//! Called from `csharp::render_method_csharp_full_chains` between
//! `reconstruct::apply` and `build_local_tdi_map`. The inferred types are
//! returned as a `HashMap<String, i32>` (display name → tdi) that is merged
//! into the local_types map before `apply_with_maps_and_types`.

use super::csharp::{Expr, Stmt};
use super::resolve::ResolveMaps;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

static HEURISTIC_METHOD_HITS: AtomicU64 = AtomicU64::new(0);
static HEURISTIC_FIELD_HITS: AtomicU64 = AtomicU64::new(0);
static HEURISTIC_UNITY_HITS: AtomicU64 = AtomicU64::new(0);
static HEURISTIC_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Human-readable report of heuristic typing impact.
pub fn report() -> String {
    let total = HEURISTIC_TOTAL.load(Ordering::Relaxed);
    if total == 0 {
        return String::new();
    }
    let method = HEURISTIC_METHOD_HITS.load(Ordering::Relaxed);
    let field = HEURISTIC_FIELD_HITS.load(Ordering::Relaxed);
    let unity = HEURISTIC_UNITY_HITS.load(Ordering::Relaxed);
    format!(
        "[heuristic-types] {total} local(s) typed: {method} from method calls, \
         {field} from field accesses, {unity} from Unity API patterns\n"
    )
}

// ---------------------------------------------------------------------------
// Index: method-name → declaring types, field-name → declaring types
// ---------------------------------------------------------------------------

/// Precomputed indexes for O(1) lookup of "which types declare a member with
/// this name?". Built once globally and reused across all method renders.
pub struct HeuristicIndex {
    /// method name → set of (tdi, fully_qualified_type_name) that declare it.
    methods_by_name: HashMap<String, Vec<TypeInfo>>,
    /// field name → set of (tdi, fully_qualified_type_name) that declare it.
    fields_by_name: HashMap<String, Vec<TypeInfo>>,
    /// TypeDef index of `System.Object`. A local typed Object has no usable
    /// name (`objN`); unique member usage is stronger evidence than the
    /// widened Object type and may refine it.
    object_tdi: Option<i32>,
}

#[derive(Clone)]
struct TypeInfo {
    tdi: i32,
    /// Declaring type's simple name, used as the display role when a unique
    /// member identifies the receiver's type.
    name: String,
}

impl HeuristicIndex {
    /// Build the indexes from `ResolveMaps`.
    ///
    /// Uses `methods_by_rva` for method names (every resolved method carries
    /// its declaring type) and `field_layouts` for field names (every resolved
    /// field offset carries its name and declaring type).
    pub fn build(maps: &ResolveMaps) -> Self {
        let mut methods_by_name: HashMap<String, Vec<TypeInfo>> = HashMap::new();
        let mut fields_by_name: HashMap<String, Vec<TypeInfo>> = HashMap::new();

        // Method index: from every resolved method, record (name → declaring type).
        for m in maps.methods_by_rva.values() {
            let info = TypeInfo {
                tdi: m.type_def_index,
                name: m.declaring_type_name.clone(),
            };
            methods_by_name
                .entry(m.name.clone())
                .or_default()
                .push(info);
        }
        // Also index methods from shared_by_rva (folded bodies).
        for methods in maps.shared_by_rva.values() {
            for m in methods {
                let info = TypeInfo {
                tdi: m.type_def_index,
                name: m.declaring_type_name.clone(),
            };
                methods_by_name
                    .entry(m.name.clone())
                    .or_default()
                    .push(info);
            }
        }
        // Also index from specs_by_body_rva.
        for specs in maps.specs_by_body_rva.values() {
            for (_spec_idx, m) in specs {
                let info = TypeInfo {
                tdi: m.type_def_index,
                name: m.declaring_type_name.clone(),
            };
                methods_by_name
                    .entry(m.name.clone())
                    .or_default()
                    .push(info);
            }
        }

        // Field index: from field_layouts, record (name → declaring type).
        // field_layouts: (tdi, offset) → (field_name, declaring_type_name)
        // We need the namespace too, but field_layouts only stores the type name.
        // We'll use the type name as-is (it's sufficient for uniqueness checks).
        let mut seen_fields: HashMap<String, Vec<(i32, String)>> = HashMap::new();
        for ((tdi, _offset), (field_name, decl_type_name)) in &maps.field_layouts {
            seen_fields
                .entry(field_name.clone())
                .or_default()
                .push((*tdi, decl_type_name.clone()));
        }
        for (field_name, entries) in seen_fields {
            let infos: Vec<TypeInfo> = entries
                .into_iter()
                .map(|(tdi, type_name)| TypeInfo {
                    tdi,
                    name: type_name,
                })
                .collect();
            fields_by_name.insert(field_name, infos);
        }

        // Deduplicate each entry by tdi (a method can appear in multiple maps).
        for entries in methods_by_name.values_mut() {
            entries.sort_by_key(|e| e.tdi);
            entries.dedup_by_key(|e| e.tdi);
        }
        for entries in fields_by_name.values_mut() {
            entries.sort_by_key(|e| e.tdi);
            entries.dedup_by_key(|e| e.tdi);
        }

        let object_tdi = maps
            .tdi_names_reverse
            .get("System.Object")
            .copied()
            .or_else(|| maps.tdi_names_reverse.get("Object").copied());
        HeuristicIndex {
            methods_by_name,
            fields_by_name,
            object_tdi,
        }
    }

    /// TypeDef index of `System.Object`, when the metadata names it.
    pub fn object_tdi(&self) -> Option<i32> {
        self.object_tdi
    }

    /// If `method_name` is declared on exactly one type, return that type.
    fn unique_type_for_method(&self, method_name: &str) -> Option<&TypeInfo> {
        let entries = self.methods_by_name.get(method_name)?;
        if entries.len() == 1 {
            Some(&entries[0])
        } else {
            None
        }
    }

    /// If `field_name` is declared on exactly one type, return that type.
    fn unique_type_for_field(&self, field_name: &str) -> Option<&TypeInfo> {
        let entries = self.fields_by_name.get(field_name)?;
        if entries.len() == 1 {
            Some(&entries[0])
        } else {
            None
        }
    }

    /// The unique declaring type's simple name for a method, when one exists.
    pub(crate) fn unique_type_name_for_method(&self, method: &str) -> Option<&str> {
        self.unique_type_for_method(method)
            .map(|t| t.name.as_str())
    }

    /// The unique declaring type's simple name for a field, when one exists.
    pub(crate) fn unique_type_name_for_field(&self, field: &str) -> Option<&str> {
        self.unique_type_for_field(field)
            .map(|t| t.name.as_str())
    }
}

// ---------------------------------------------------------------------------
// Well-known Unity API patterns
// ---------------------------------------------------------------------------

/// Check if a call matches a well-known Unity API pattern and return the
/// inferred type for the result.
///
/// Returns a `UnityInference` when a pattern matches.
fn check_unity_pattern(
    method_name: &str,
    _recv: &Option<Box<Expr>>,
    args: &[Expr],
    already_typed: &HashMap<String, i32>,
) -> Option<UnityInference> {
    // Instantiate(obj) → returns same type as obj.
    // If obj is typed, we can type the result.
    if method_name == "Instantiate" || method_name.ends_with(".Instantiate") {
        if let Some(first_arg) = args.first() {
            if let Some(src_name) = expr_var_name(first_arg) {
                if let Some(&src_tdi) = already_typed.get(src_name) {
                    if src_tdi >= 0 {
                        return Some(UnityInference {
                            target: InferenceTarget::CallResult,
                            tdi: src_tdi,
                        });
                    }
                }
            }
        }
    }

    // Transform.Find(name) / GameObject.Find(name) → returns Transform/GameObject.
    // We don't know the exact tdi without metadata lookup, so skip these.
    // They would need a tdi → name reverse map to resolve.

    _ = args;
    None
}

struct UnityInference {
    target: InferenceTarget,
    tdi: i32,
}

enum InferenceTarget {
    /// The result of the call (assigned to a local).
    CallResult,
}

// ---------------------------------------------------------------------------
// Statement tree walking
// ---------------------------------------------------------------------------

/// Extract the variable name from an expression, if it's a simple `Expr::Var`.
fn expr_var_name(e: &Expr) -> Option<&str> {
    match e {
        Expr::Var(name) => Some(name.as_str()),
        _ => None,
    }
}

/// Run heuristic type inference on the statement tree.
///
/// Returns a map of `local_display_name → tdi` for each local whose type was
/// inferred. The caller merges this into the `local_types` map before field
/// annotation.
///
/// `already_typed` is the set of locals that already have a type from metadata
/// propagation (the output of `build_local_tdi_map`). We never override these.
pub fn infer_types_heuristically(
    stmts: &[Stmt],
    maps: &ResolveMaps,
    already_typed: &HashMap<String, i32>,
) -> HashMap<String, i32> {
    let index = HeuristicIndex::build(maps);
    infer_types_with_index(stmts, &index, already_typed)
}

/// Infer types using a pre-built index (avoids O(n²) rebuild cost).
pub fn infer_types_with_index(
    stmts: &[Stmt],
    index: &HeuristicIndex,
    already_typed: &HashMap<String, i32>,
) -> HashMap<String, i32> {
    let mut inferred: HashMap<String, i32> = HashMap::new();
    let mut result_locals: HashMap<String, String> = HashMap::new(); // call result → assigned local

    // First pass: collect assignment targets so we can map call results to locals.
    collect_call_result_assignments(stmts, &mut result_locals);

    // Second pass: walk the tree and apply patterns.
    walk_stmts(stmts, index, already_typed, &mut inferred, &result_locals);

    let total = inferred.len();
    if total > 0 {
        HEURISTIC_TOTAL.fetch_add(total as u64, Ordering::Relaxed);
    }

    inferred
}

/// Collect `v = expr.call(...)` patterns so we know which local holds a call result.
fn collect_call_result_assignments(stmts: &[Stmt], out: &mut HashMap<String, String>) {
    for stmt in stmts {
        if let Stmt::Assign(target, _, Expr::Call { recv: _, name, args: _ }) = stmt {
            // The target local holds the result of this call.
            // We use the call's rendered name as the key.
            // Actually, we just need to know: for each Call expression that is
            // assigned, what local gets the result. We'll handle this inline
            // during the walk instead.
            let _ = target;
            let _ = name;
        }
        // Recurse into nested statement lists.
        let mut children: Vec<&Vec<Stmt>> = Vec::new();
        super::reconstruct::children(stmt, &mut children);
        for child in children {
            collect_call_result_assignments(child, out);
        }
    }
}

/// Walk all statements, applying heuristic patterns.
fn walk_stmts(
    stmts: &[Stmt],
    index: &HeuristicIndex,
    already_typed: &HashMap<String, i32>,
    inferred: &mut HashMap<String, i32>,
    result_locals: &HashMap<String, String>,
) {
    for stmt in stmts {
        walk_stmt(stmt, index, already_typed, inferred, result_locals);
    }
}

fn walk_stmt(
    stmt: &Stmt,
    index: &HeuristicIndex,
    already_typed: &HashMap<String, i32>,
    inferred: &mut HashMap<String, i32>,
    result_locals: &HashMap<String, String>,
) {
    match stmt {
        Stmt::Eval(expr) => {
            check_expr(expr, index, already_typed, inferred, result_locals, None);
        }
        Stmt::Assign(target, _, expr) => {
            check_expr(expr, index, already_typed, inferred, result_locals, Some(target.as_str()));
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            check_expr(cond, index, already_typed, inferred, result_locals, None);
            walk_stmts(then_body, index, already_typed, inferred, result_locals);
            walk_stmts(else_body, index, already_typed, inferred, result_locals);
        }
        Stmt::While { cond, body, .. } => {
            check_expr(cond, index, already_typed, inferred, result_locals, None);
            walk_stmts(body, index, already_typed, inferred, result_locals);
        }
        Stmt::DoWhile { body, cond, .. } => {
            walk_stmts(body, index, already_typed, inferred, result_locals);
            check_expr(cond, index, already_typed, inferred, result_locals, None);
        }
        Stmt::For { init, cond, incr, body, .. } => {
            walk_stmts(init, index, already_typed, inferred, result_locals);
            if let Some(c) = cond {
                check_expr(c, index, already_typed, inferred, result_locals, None);
            }
            walk_stmts(incr, index, already_typed, inferred, result_locals);
            walk_stmts(body, index, already_typed, inferred, result_locals);
        }
        Stmt::ForEach { body, .. } => {
            walk_stmts(body, index, already_typed, inferred, result_locals);
        }
        Stmt::Using { decl, body, .. } => {
            walk_stmt(decl, index, already_typed, inferred, result_locals);
            walk_stmts(body, index, already_typed, inferred, result_locals);
        }
        Stmt::Lock { expr, body, .. } => {
            check_expr(expr, index, already_typed, inferred, result_locals, None);
            walk_stmts(body, index, already_typed, inferred, result_locals);
        }
        Stmt::Try { try_body, catches, finally_body, .. } => {
            walk_stmts(try_body, index, already_typed, inferred, result_locals);
            for (_, body) in catches {
                walk_stmts(body, index, already_typed, inferred, result_locals);
            }
            if let Some(fb) = finally_body {
                walk_stmts(fb, index, already_typed, inferred, result_locals);
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, body) in cases {
                walk_stmts(body, index, already_typed, inferred, result_locals);
            }
            walk_stmts(tail, index, already_typed, inferred, result_locals);
            if let Some(arm) = default {
                walk_stmts(arm, index, already_typed, inferred, result_locals);
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, body) in cases {
                walk_stmts(body, index, already_typed, inferred, result_locals);
            }
            if let Some(t) = tail {
                walk_stmts(t, index, already_typed, inferred, result_locals);
            }
        }
        Stmt::Return(Some(expr)) | Stmt::Throw(expr) | Stmt::YieldReturn(expr) => {
            check_expr(expr, index, already_typed, inferred, result_locals, None);
        }
        Stmt::EventAssign { field, value, .. } => {
            check_expr(field, index, already_typed, inferred, result_locals, None);
            check_expr(value, index, already_typed, inferred, result_locals, None);
        }
        Stmt::IndexAssign {
            collection,
            index: idx,
            value,
        } => {
            check_expr(collection, index, already_typed, inferred, result_locals, None);
            check_expr(idx, index, already_typed, inferred, result_locals, None);
            check_expr(value, index, already_typed, inferred, result_locals, None);
        }
        _ => {}
    }
}

/// Check an expression for heuristic typing opportunities.
fn check_expr(
    expr: &Expr,
    index: &HeuristicIndex,
    already_typed: &HashMap<String, i32>,
    inferred: &mut HashMap<String, i32>,
    result_locals: &HashMap<String, String>,
    assign_target: Option<&str>,
) {
    match expr {
        // Pattern 1: Method call receiver typing.
        Expr::Call { recv, name, args } => {
            // Check Unity API patterns first.
            if let Some(unity_inf) = check_unity_pattern(name, recv, args, already_typed) {
                match unity_inf.target {
                    InferenceTarget::CallResult => {
                        // The result is assigned to assign_target.
                        if let Some(target) = assign_target {
                            if !already_typed.contains_key(target) && !inferred.contains_key(target) {
                                if unity_inf.tdi >= 0 {
                                    inferred.insert(target.to_string(), unity_inf.tdi);
                                    HEURISTIC_UNITY_HITS.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            }

            // Instance method call: receiver.Method(args)
            if let Some(recv_expr) = recv.as_ref() {
                if let Some(recv_name) = expr_var_name(recv_expr) {
                    // An OBJECT-typed local is exactly the `objN` population:
                    // metadata widened it, so no name could be derived. A
                    // member with a unique declaring type identifies the
                    // receiver better than the widened type does.
                    let fixable = !already_typed.contains_key(recv_name)
                        || (index.object_tdi.is_some()
                            && already_typed.get(recv_name) == index.object_tdi.as_ref());
                    if fixable && !inferred.contains_key(recv_name) {
                        let bare_name = extract_bare_method_name(name);
                        if let Some(type_info) = index.unique_type_for_method(bare_name) {
                            inferred.insert(recv_name.to_string(), type_info.tdi);
                            HEURISTIC_METHOD_HITS.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }

            // Recurse into arguments.
            for arg in args {
                check_expr(arg, index, already_typed, inferred, result_locals, None);
            }
        }

        // Pattern 2: Field access typing.
        Expr::Field(recv, field_name) => {
            if let Some(recv_name) = expr_var_name(recv) {
                let fixable = !already_typed.contains_key(recv_name)
                    || (index.object_tdi.is_some()
                        && already_typed.get(recv_name) == index.object_tdi.as_ref());
                if fixable && !inferred.contains_key(recv_name) {
                    if let Some(type_info) = index.unique_type_for_field(field_name) {
                        inferred.insert(recv_name.to_string(), type_info.tdi);
                        HEURISTIC_FIELD_HITS.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            // Recurse into receiver.
            check_expr(recv, index, already_typed, inferred, result_locals, None);
        }

        // Pattern 3: Collection indexing — if collection[i] is used and the
        // result is assigned, we can infer the collection is an array/list.
        // This is a weaker signal; skip for now (high-confidence only).
        Expr::Index { collection, index: idx } => {
            check_expr(collection, index, already_typed, inferred, result_locals, None);
            check_expr(idx, index, already_typed, inferred, result_locals, None);
        }

        // Recurse into sub-expressions.
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            check_expr(a, index, already_typed, inferred, result_locals, None);
            check_expr(b, index, already_typed, inferred, result_locals, None);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            check_expr(x, index, already_typed, inferred, result_locals, None);
        }
        Expr::Cond(c, a, b) => {
            check_expr(c, index, already_typed, inferred, result_locals, None);
            check_expr(a, index, already_typed, inferred, result_locals, None);
            check_expr(b, index, already_typed, inferred, result_locals, None);
        }
        Expr::RawDeref(base, _, _) | Expr::RawAddr(base, _) => {
            check_expr(base, index, already_typed, inferred, result_locals, None);
        }
        Expr::ArrayInit { items, .. } => {
            for item in items {
                check_expr(item, index, already_typed, inferred, result_locals, None);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for arg in ctor_args {
                check_expr(arg, index, already_typed, inferred, result_locals, None);
            }
            for item in items {
                check_expr(item, index, already_typed, inferred, result_locals, None);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for arg in ctor_args {
                check_expr(arg, index, already_typed, inferred, result_locals, None);
            }
            for (_, v) in fields {
                check_expr(v, index, already_typed, inferred, result_locals, None);
            }
        }
        Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for arg in args {
                check_expr(arg, index, already_typed, inferred, result_locals, None);
            }
        }
        Expr::RawCall(_, args, _) => {
            for arg in args {
                check_expr(arg, index, already_typed, inferred, result_locals, None);
            }
        }
        Expr::Lambda { body, .. } => {
            check_expr(body, index, already_typed, inferred, result_locals, None);
        }
        Expr::NullConditional { receiver, access } => {
            check_expr(receiver, index, already_typed, inferred, result_locals, None);
            check_expr(access, index, already_typed, inferred, result_locals, None);
        }
        // Terminals — nothing to infer.
        Expr::Var(_) | Expr::This | Expr::Int(_) | Expr::Float(_) | Expr::Double(_)
        | Expr::Bool(_) | Expr::Str(_) | Expr::Null | Expr::Unknown
        | Expr::MissingAbiArg | Expr::StaticField(_, _) | Expr::MethodGroup(_)
        | Expr::PatternMatch { .. } | Expr::StringInterp { .. }
        | Expr::Tuple { .. } | Expr::MethodChain { .. } | Expr::SwitchExpr { .. }
        | Expr::Range { .. } | Expr::IndexFromEnd(_) | Expr::TypeOf(_) => {}
    }
}

/// Extract the bare method name from a possibly-qualified name.
///
/// `"UnityEngine.Transform::GetPosition"` → `"GetPosition"`
/// `"GetComponent"` → `"GetComponent"`
/// `"System.Object..ctor"` → `"ctor"` (won't match any index entry)
/// `"Console.WriteLine"` → `"WriteLine"`
fn extract_bare_method_name(full_name: &str) -> &str {
    // Try "::" separator first (IL2CPP convention).
    if let Some(pos) = full_name.rfind("::") {
        return &full_name[pos + 2..];
    }
    // Try "." separator (C# convention for static calls like "Console.WriteLine").
    if let Some(pos) = full_name.rfind('.') {
        return &full_name[pos + 1..];
    }
    full_name
}

// ---------------------------------------------------------------------------
// Integration helper
// ---------------------------------------------------------------------------

/// Merge heuristic inferences into an existing local_types map.
///
/// Does not override entries already present in `local_types`.
pub fn merge_inferred_types(
    local_types: &mut HashMap<String, i32>,
    inferred: &HashMap<String, i32>,
    object_tdi: Option<i32>,
) {
    for (name, tdi) in inferred {
        match local_types.get(name) {
            None => {
                local_types.insert(name.clone(), *tdi);
            }
            // Refine only the widened Object type; a concrete class/struct
            // from metadata is never overridden.
            Some(existing) if Some(*existing) == object_tdi => {
                local_types.insert(name.clone(), *tdi);
            }
            Some(_) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::ManagedType;

    fn make_method_ref(name: &str, tdi: i32, ns: &str, type_name: &str) -> super::super::types::MethodRef {
        super::super::types::MethodRef {
            image_index: 0,
            type_def_index: tdi,
            method_index: 0,
            name: name.to_string(),
            declaring_namespace: ns.to_string(),
            declaring_type_name: type_name.to_string(),
            assembly: "Assembly".to_string(),
            sig: super::super::types::MethodSigInfo {
                ret: ManagedType::Unknown,
                params: vec![],
                is_static: false,
            },
        }
    }

    #[test]
    fn unique_method_names_resolve() {
        let mut maps = ResolveMaps::default();
        // Only one type has "DoSomething".
        maps.methods_by_rva.insert(
            0x1000,
            make_method_ref("DoSomething", 42, "UnityEngine", "Transform"),
        );
        let index = HeuristicIndex::build(&maps);
        let ti = index.unique_type_for_method("DoSomething").unwrap();
        assert_eq!(ti.tdi, 42);
    }

    #[test]
    fn ambiguous_method_names_are_skipped() {
        let mut maps = ResolveMaps::default();
        // Two types have "Update".
        maps.methods_by_rva.insert(
            0x1000,
            make_method_ref("Update", 10, "Game", "Player"),
        );
        maps.methods_by_rva.insert(
            0x2000,
            make_method_ref("Update", 20, "Game", "Enemy"),
        );
        let index = HeuristicIndex::build(&maps);
        assert!(index.unique_type_for_method("Update").is_none());
    }

    #[test]
    fn unique_field_names_resolve() {
        let mut maps = ResolveMaps::default();
        maps.field_layouts.insert((5, 8), ("health".to_string(), "Player".to_string()));
        let index = HeuristicIndex::build(&maps);
        let ti = index.unique_type_for_field("health").unwrap();
        assert_eq!(ti.tdi, 5);
    }

    #[test]
    fn ambiguous_field_names_are_skipped() {
        let mut maps = ResolveMaps::default();
        maps.field_layouts.insert((5, 8), ("name".to_string(), "Player".to_string()));
        maps.field_layouts.insert((6, 8), ("name".to_string(), "Enemy".to_string()));
        let index = HeuristicIndex::build(&maps);
        assert!(index.unique_type_for_field("name").is_none());
    }

    #[test]
    fn method_call_on_untyped_local_infers_type() {
        let mut maps = ResolveMaps::default();
        maps.methods_by_rva.insert(
            0x1000,
            make_method_ref("DoSomething", 42, "UnityEngine", "Transform"),
        );
        let already_typed = HashMap::new();
        let stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v1".into()))),
                name: "DoSomething".into(),
                args: vec![],
            }),
        ];
        let inferred = infer_types_heuristically(&stmts, &maps, &already_typed);
        assert_eq!(inferred.get("v1"), Some(&42));
    }

    #[test]
    fn already_typed_local_is_not_overridden() {
        let mut maps = ResolveMaps::default();
        maps.methods_by_rva.insert(
            0x1000,
            make_method_ref("DoSomething", 42, "UnityEngine", "Transform"),
        );
        let mut already_typed = HashMap::new();
        already_typed.insert("v1".to_string(), 99); // Already typed as something else.
        let stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v1".into()))),
                name: "DoSomething".into(),
                args: vec![],
            }),
        ];
        let inferred = infer_types_heuristically(&stmts, &maps, &already_typed);
        // Should NOT infer v1 since it's already typed.
        assert!(inferred.get("v1").is_none());
    }

    #[test]
    fn field_access_on_untyped_local_infers_type() {
        let mut maps = ResolveMaps::default();
        maps.field_layouts.insert((7, 16), ("health".to_string(), "Player".to_string()));
        let already_typed = HashMap::new();
        let stmts = vec![
            Stmt::Assign(
                "v2".into(),
                None,
                Expr::Field(Box::new(Expr::Var("v1".into())), "health".into()),
            ),
        ];
        let inferred = infer_types_heuristically(&stmts, &maps, &already_typed);
        assert_eq!(inferred.get("v1"), Some(&7));
    }

    #[test]
    fn qualified_method_name_is_stripped() {
        assert_eq!(extract_bare_method_name("UnityEngine.Transform::GetPosition"), "GetPosition");
        assert_eq!(extract_bare_method_name("GetComponent"), "GetComponent");
        // "..ctor" → the part after the last dot is "ctor"; won't match any index entry.
        assert_eq!(extract_bare_method_name("System.Object..ctor"), "ctor");
    }

    #[test]
    fn nested_stmts_are_walked() {
        let mut maps = ResolveMaps::default();
        maps.methods_by_rva.insert(
            0x1000,
            make_method_ref("UniqueMethod", 55, "Game", "Widget"),
        );
        let already_typed = HashMap::new();
        let stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    Stmt::Eval(Expr::Call {
                        recv: Some(Box::new(Expr::Var("v3".into()))),
                        name: "UniqueMethod".into(),
                        args: vec![],
                    }),
                ],
                else_body: vec![],
            },
        ];
        let inferred = infer_types_heuristically(&stmts, &maps, &already_typed);
        assert_eq!(inferred.get("v3"), Some(&55));
    }

    #[test]
    fn object_typed_receiver_is_refined_by_unique_member() {
        let mut maps = ResolveMaps::default();
        maps.tdi_names_reverse.insert("System.Object".into(), 1);
        maps.methods_by_rva
            .insert(0x1000, make_method_ref("DoSomething", 2, "", "Thing"));
        let index = HeuristicIndex::build(&maps);
        let stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(Expr::Var("v1".into()))),
            name: "DoSomething".into(),
            args: vec![],
        })];
        let mut already = HashMap::new();
        already.insert("v1".to_string(), 1);
        let inferred = infer_types_with_index(&stmts, &index, &already);
        assert_eq!(inferred.get("v1"), Some(&2));
        let mut local_types = already.clone();
        merge_inferred_types(&mut local_types, &inferred, index.object_tdi());
        assert_eq!(local_types.get("v1"), Some(&2));
    }

    #[test]
    fn merge_does_not_override_existing() {
        let mut local_types = HashMap::new();
        local_types.insert("v1".to_string(), 10);
        let mut inferred = HashMap::new();
        inferred.insert("v1".to_string(), 99); // Conflict.
        inferred.insert("v2".to_string(), 42); // New.
        merge_inferred_types(&mut local_types, &inferred, None);
        assert_eq!(local_types["v1"], 10); // Not overridden.
        assert_eq!(local_types["v2"], 42); // Added.
    }
}
