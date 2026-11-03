//! Auto-property detection: identifies getter/setter pairs that form
//! C# auto-properties, and computed (expression-body) properties.
//!
//! ## Pattern
//!
//! C# auto-properties compile to getter/setter method pairs:
//! ```csharp
//! public int X { get; set; }
//! ```
//! becomes:
//! ```text
//! int get_X() { return this._x; }
//! void set_X(int value) { this._x = value; }
//! ```
//!
//! Computed (expression-body) properties compile to getters with small
//! statement bodies that compute and return a value:
//! ```csharp
//! public int DoubleX => this._x * 2;
//! public bool IsReady => this._state != 0 && this._enabled;
//! ```
//!
//! ## Detection
//!
//! This module provides a post-processing analysis that examines a collection
//! of methods and identifies:
//!
//! 1. **Simple auto-properties**: getter/setter pairs sharing a backing field.
//! 2. **Computed properties**: getters with 1-3 statements that compute a
//!    value from field accesses and return it.
//! 3. **Backing field detection**: identifies the `<X>k__BackingField` pattern
//!    used by the C# compiler for auto-properties.
//!
//! ## Implementation approach
//!
//! This is a CROSS-METHOD analysis — it requires examining multiple methods
//! together, not just one statement list. The existing reconstruction passes
//! operate on a single method's statement tree. This module therefore provides
//! a standalone analysis function that takes a collection of method bodies
//! and returns a set of auto-property descriptors.
//!
//! The actual rendering transformation (replacing the two methods with a
//! single property declaration) is deferred to the emitter, which has the
//! full type context needed to produce correct C# property syntax.
//!
//! ## Safety conditions
//!
//! The detection is conservative:
//! - Simple auto-properties: both getter and setter must have EXACTLY one
//!   statement. The getter must return a field of `this`. The setter must
//!   assign to a field of `this`.
//! - Computed properties: the getter has 1-3 statements, the last of which
//!   is `return expr;`. Intermediate statements must be simple assignments
//!   of local temporaries from field accesses or arithmetic on fields.
//! - The backing field must be the SAME in both methods (for auto-properties).
//! - The property name is derived from the method names (`get_X` → `X`).
//!
//! Any deviation from these conditions causes the pair to be rejected, leaving
//! the methods as-is. Accurate-but-ugly beats plausible-but-wrong.
//!
//! ## Measured reality
//!
//! Auto-properties are common in the x64 corpus. Unity game code uses them
//! extensively for data classes. The `get_`/`set_` naming convention is
//! standard IL2CPP output for C# properties.

use super::super::csharp::{Expr, Stmt};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Instrumentation counters
// ---------------------------------------------------------------------------

/// Counters for property detection statistics, useful for tuning and reporting.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PropertyCounters {
    /// Total getter methods examined.
    pub getters_examined: u64,
    /// Total setter methods examined.
    pub setters_examined: u64,
    /// Simple auto-property pairs detected (getter + setter sharing a field).
    pub simple_auto_properties: u64,
    /// Getter-only simple auto-properties detected.
    pub getter_only_properties: u64,
    /// Setter-only simple auto-properties detected.
    pub setter_only_properties: u64,
    /// Computed (expression-body) properties detected.
    pub computed_properties: u64,
    /// Getter candidates rejected due to too many statements.
    pub rejected_too_many_stmts: u64,
    /// Getter candidates rejected because the return expression is not
    /// recognisable (e.g. involves calls, complex nesting).
    pub rejected_complex_expr: u64,
    /// Getter candidates rejected because intermediate statements are not
    /// simple temp assignments.
    pub rejected_complex_intermediates: u64,
    /// Setter candidates rejected because the assigned value is not `value`.
    pub rejected_setter_non_value: u64,
    /// Backing fields detected via the `<X>k__BackingField` naming convention.
    pub compiler_backing_fields: u64,
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A detected auto-property: a getter/setter pair sharing a backing field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoProperty {
    /// The property name (derived from `get_X` → `X`).
    pub name: String,
    /// The backing field name (e.g., `_x` or `<X>k__BackingField`).
    pub backing_field: String,
    /// The getter method name (e.g., `get_X`).
    pub getter_name: String,
    /// The setter method name (e.g., `set_X`).
    pub setter_name: String,
    /// Whether the property has a getter.
    pub has_getter: bool,
    /// Whether the property has a setter.
    pub has_setter: bool,
}

/// A detected computed (expression-body) property.
///
/// This represents a getter whose body computes a value from field accesses
/// and returns it, corresponding to C# expression-body syntax:
/// ```csharp
/// public int DoubleX => this._x * 2;
/// ```
#[derive(Debug, Clone)]
pub struct ComputedProperty {
    /// The property name (derived from `get_X` → `X`).
    pub name: String,
    /// The getter method name (e.g., `get_X`).
    pub getter_name: String,
    /// The return expression (the computed value).
    pub expr: Expr,
    /// Names of `this`-fields read by the expression.
    pub fields_read: Vec<String>,
    /// Names of local temporaries used in intermediate statements (if any).
    pub temporaries: Vec<String>,
}

/// Result of the full property detection pass.
#[derive(Debug, Clone)]
pub struct PropertyDetectionResult {
    /// Simple auto-properties (backed by a single field with getter/setter).
    pub auto_properties: Vec<AutoProperty>,
    /// Computed (expression-body) properties.
    pub computed_properties: Vec<ComputedProperty>,
    /// Instrumentation counters.
    pub counters: PropertyCounters,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Analyze a collection of methods and identify auto-properties and computed
/// properties.
///
/// `methods` maps method names to their statement bodies. The function
/// returns a [`PropertyDetectionResult`] containing all detected properties
/// and instrumentation counters.
///
/// A method that is part of an auto-property pair should be rendered as a
/// property accessor rather than a standalone method. The caller is
/// responsible for the actual rendering transformation.
pub fn detect_auto_properties(methods: &HashMap<String, Vec<Stmt>>) -> Vec<AutoProperty> {
    detect_auto_properties_with_counters(methods).auto_properties
}

/// Extended detection that also returns computed properties and counters.
pub fn detect_auto_properties_with_counters(
    methods: &HashMap<String, Vec<Stmt>>,
) -> PropertyDetectionResult {
    let mut counters = PropertyCounters::default();
    let mut getters: HashMap<String, GetterInfo> = HashMap::new();
    let mut setters: HashMap<String, SetterInfo> = HashMap::new();
    let mut computed_getters: Vec<ComputedProperty> = Vec::new();

    // First pass: identify candidate getters, setters, and computed properties.
    for (name, body) in methods {
        if let Some(prop_name) = name.strip_prefix("get_") {
            counters.getters_examined += 1;

            // Try simple auto-property getter first.
            if let Some(field) = extract_getter_field(body) {
                // Check for compiler-generated backing field naming.
                if is_compiler_backing_field(&field) {
                    counters.compiler_backing_fields += 1;
                }
                getters.insert(
                    prop_name.to_string(),
                    GetterInfo {
                        method_name: name.clone(),
                        field,
                    },
                );
            } else if let Some(computed) = try_extract_computed_getter(prop_name, name, body, &mut counters) {
                // Fall back to computed property detection.
                counters.computed_properties += 1;
                computed_getters.push(computed);
            }
        } else if let Some(prop_name) = name.strip_prefix("set_") {
            counters.setters_examined += 1;
            if let Some(field) = extract_setter_field(body, &mut counters) {
                // Check for compiler-generated backing field naming.
                if is_compiler_backing_field(&field) {
                    counters.compiler_backing_fields += 1;
                }
                setters.insert(
                    prop_name.to_string(),
                    SetterInfo {
                        method_name: name.clone(),
                        field,
                    },
                );
            }
        }
    }

    // Second pass: pair getters and setters that share a backing field.
    let mut properties = Vec::new();

    // Collect all property names from both getters and setters.
    let mut all_names: Vec<String> = getters.keys().cloned().collect();
    for name in setters.keys() {
        if !all_names.contains(name) {
            all_names.push(name.clone());
        }
    }
    all_names.sort();

    for prop_name in all_names {
        let getter = getters.get(&prop_name);
        let setter = setters.get(&prop_name);

        match (getter, setter) {
            (Some(g), Some(s)) => {
                // Both exist — they must access the same field to be paired.
                if g.field == s.field {
                    counters.simple_auto_properties += 1;
                    properties.push(AutoProperty {
                        name: prop_name,
                        backing_field: g.field.clone(),
                        getter_name: g.method_name.clone(),
                        setter_name: s.method_name.clone(),
                        has_getter: true,
                        has_setter: true,
                    });
                } else {
                    // Fields differ — NOT an auto-property pair. Report them
                    // as separate getter-only and setter-only entries.
                    counters.getter_only_properties += 1;
                    properties.push(AutoProperty {
                        name: prop_name.clone(),
                        backing_field: g.field.clone(),
                        getter_name: g.method_name.clone(),
                        setter_name: String::new(),
                        has_getter: true,
                        has_setter: false,
                    });
                    counters.setter_only_properties += 1;
                    properties.push(AutoProperty {
                        name: prop_name,
                        backing_field: s.field.clone(),
                        getter_name: String::new(),
                        setter_name: s.method_name.clone(),
                        has_getter: false,
                        has_setter: true,
                    });
                }
            }
            (Some(g), None) => {
                // Getter-only property (read-only).
                counters.getter_only_properties += 1;
                properties.push(AutoProperty {
                    name: prop_name,
                    backing_field: g.field.clone(),
                    getter_name: g.method_name.clone(),
                    setter_name: String::new(),
                    has_getter: true,
                    has_setter: false,
                });
            }
            (None, Some(s)) => {
                // Setter-only property (write-only, rare but legal).
                counters.setter_only_properties += 1;
                properties.push(AutoProperty {
                    name: prop_name,
                    backing_field: s.field.clone(),
                    getter_name: String::new(),
                    setter_name: s.method_name.clone(),
                    has_getter: false,
                    has_setter: true,
                });
            }
            (None, None) => unreachable!(),
        }
    }

    PropertyDetectionResult {
        auto_properties: properties,
        computed_properties: computed_getters,
        counters,
    }
}

// ---------------------------------------------------------------------------
// Backing field detection
// ---------------------------------------------------------------------------

/// Check if a field name matches the compiler-generated backing field pattern
/// `<PropertyName>k__BackingField`.
pub fn is_compiler_backing_field(field: &str) -> bool {
    field.starts_with('<') && field.ends_with(">k__BackingField")
}

/// Extract the property name from a compiler-generated backing field name.
/// Returns `None` if the field doesn't match the pattern.
pub fn property_name_from_backing_field(field: &str) -> Option<String> {
    if !is_compiler_backing_field(field) {
        return None;
    }
    // `<X>k__BackingField` → extract between `<` and `>`.
    let inner = &field[1..field.len() - ">k__BackingField".len()];
    if inner.is_empty() {
        None
    } else {
        Some(inner.to_string())
    }
}

// ---------------------------------------------------------------------------
// Simple getter / setter extraction
// ---------------------------------------------------------------------------

/// If `body` is a getter (single `return this.field;`), return the field name.
fn extract_getter_field(body: &[Stmt]) -> Option<String> {
    // Must be exactly one statement.
    if body.len() != 1 {
        return None;
    }

    let Stmt::Return(Some(Expr::Field(recv, field_name))) = &body[0] else {
        return None;
    };

    // The receiver must be `this`.
    if !matches!(recv.as_ref(), Expr::This) {
        return None;
    }

    Some(field_name.clone())
}

/// If `body` is a setter (single `this.field = value;`), return the field name.
fn extract_setter_field(body: &[Stmt], counters: &mut PropertyCounters) -> Option<String> {
    // Must be exactly one statement.
    if body.len() != 1 {
        return None;
    }

    let Stmt::Assign(target, None, value_expr) = &body[0] else {
        return None;
    };

    // The value must be a Var named "value" (the standard C# setter parameter).
    let Expr::Var(value_name) = value_expr else {
        counters.rejected_setter_non_value += 1;
        return None;
    };

    if value_name != "value" {
        counters.rejected_setter_non_value += 1;
        return None;
    }

    if target.is_empty() {
        return None;
    }

    Some(target.clone())
}

// ---------------------------------------------------------------------------
// Computed property extraction
// ---------------------------------------------------------------------------

/// Maximum number of statements allowed in a computed property getter body.
const MAX_COMPUTED_STMTS: usize = 3;

/// Try to extract a computed (expression-body) property from a getter body.
///
/// Accepted shapes:
/// - 1 stmt: `return expr;` where `expr` involves `this.field` accesses
///   (but is NOT a simple field access — that's handled by the auto-property
///   path).
/// - 2 stmts: `tmp = this.field; return expr_using_tmp;`
/// - 3 stmts: two temp assignments + return.
///
/// Intermediate statements must be simple assignments of the form
/// `tmp = this.field` or `tmp = binop(this.field, literal)` etc.
fn try_extract_computed_getter(
    prop_name: &str,
    method_name: &str,
    body: &[Stmt],
    counters: &mut PropertyCounters,
) -> Option<ComputedProperty> {
    if body.is_empty() || body.len() > MAX_COMPUTED_STMTS {
        return None;
    }

    // The last statement must be a return.
    let last = body.last()?;
    let Stmt::Return(Some(ret_expr)) = last else {
        return None;
    };

    // The return expression must not be a simple field access on `this`
    // (that case is handled by the simple auto-property path).
    if matches!(ret_expr, Expr::Field(recv, _) if matches!(recv.as_ref(), Expr::This)) {
        return None;
    }

    // The return expression must be "computable" — i.e. it must not
    // contain calls or other side-effecting operations.
    let mut fields_read = Vec::new();
    if !expr_is_computable(ret_expr, &mut fields_read) {
        counters.rejected_complex_expr += 1;
        return None;
    }

    // Validate intermediate statements (all but the last).
    let intermediates = &body[..body.len() - 1];
    let mut temporaries = Vec::new();

    for stmt in intermediates {
        match stmt {
            Stmt::Assign(name, _, rhs) => {
                // Must be a simple temp assignment. The RHS must be
                // computable (no calls, etc.).
                let mut intermediate_fields = Vec::new();
                if !expr_is_computable(rhs, &mut intermediate_fields) {
                    counters.rejected_complex_intermediates += 1;
                    return None;
                }
                // Record the field reads from intermediates too.
                for f in intermediate_fields {
                    if !fields_read.contains(&f) {
                        fields_read.push(f);
                    }
                }
                temporaries.push(name.clone());
            }
            _ => {
                // Non-assignment intermediate — reject.
                counters.rejected_complex_intermediates += 1;
                return None;
            }
        }
    }

    // Must read at least one field (from return expr or intermediates)
    // to qualify as a computed property.
    if fields_read.is_empty() {
        counters.rejected_complex_expr += 1;
        return None;
    }

    fields_read.sort();
    fields_read.dedup();

    Some(ComputedProperty {
        name: prop_name.to_string(),
        getter_name: method_name.to_string(),
        expr: ret_expr.clone(),
        fields_read,
        temporaries,
    })
}

/// Check whether an expression is "computable" — i.e. safe to fold into an
/// expression-body property. This means:
/// - No calls (instance or static) — these could have side effects.
/// - No assignments, awaits, or other side-effecting constructs.
/// - At least one `this.field` access somewhere in the tree (checked by
///   the caller via `fields_read`).
///
/// Returns `true` if the expression is safe, and populates `fields_read`
/// with any `this.field` names found.
fn expr_is_computable(expr: &Expr, fields_read: &mut Vec<String>) -> bool {
    match expr {
        Expr::Field(recv, name) => {
            if matches!(recv.as_ref(), Expr::This) {
                fields_read.push(name.clone());
                true
            } else {
                // Field access on non-this — still computable if the receiver
                // is computable (e.g. a temporary variable).
                expr_is_computable(recv, fields_read)
            }
        }
        Expr::Var(_) | Expr::This | Expr::Int(_) | Expr::Float(_) | Expr::Double(_)
        | Expr::Bool(_) | Expr::Str(_) | Expr::Null | Expr::Unknown
        | Expr::MissingAbiArg => true,
        Expr::Bin(_, lhs, rhs) => {
            expr_is_computable(lhs, fields_read) && expr_is_computable(rhs, fields_read)
        }
        Expr::Cmp(_, lhs, rhs) => {
            expr_is_computable(lhs, fields_read) && expr_is_computable(rhs, fields_read)
        }
        Expr::Not(inner) | Expr::Neg(inner) => expr_is_computable(inner, fields_read),
        Expr::Cond(cond, then_e, else_e) => {
            expr_is_computable(cond, fields_read)
                && expr_is_computable(then_e, fields_read)
                && expr_is_computable(else_e, fields_read)
        }
        Expr::StaticField(_, _) => true,
        // Everything else (calls, awaits, allocations, etc.) is not safe.
        _ => false,
    }
}

/// Collect all `this.field` names referenced in an expression.
pub fn collect_this_fields(expr: &Expr) -> Vec<String> {
    let mut fields = Vec::new();
    collect_this_fields_rec(expr, &mut fields);
    fields.sort();
    fields.dedup();
    fields
}

fn collect_this_fields_rec(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Field(recv, name) if matches!(recv.as_ref(), Expr::This) => {
            out.push(name.clone());
        }
        Expr::Field(recv, _) => {
            collect_this_fields_rec(recv, out);
        }
        Expr::Bin(_, l, r) => {
            collect_this_fields_rec(l, out);
            collect_this_fields_rec(r, out);
        }
        Expr::Cmp(_, l, r) => {
            collect_this_fields_rec(l, out);
            collect_this_fields_rec(r, out);
        }
        Expr::Not(e) | Expr::Neg(e) | Expr::Await(e) | Expr::AddressOf(e) => {
            collect_this_fields_rec(e, out);
        }
        Expr::Cond(c, t, f) => {
            collect_this_fields_rec(c, out);
            collect_this_fields_rec(t, out);
            collect_this_fields_rec(f, out);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                collect_this_fields_rec(r, out);
            }
            for a in args {
                collect_this_fields_rec(a, out);
            }
        }
        Expr::RawDeref(base, _, _) | Expr::RawAddr(base, _) => {
            collect_this_fields_rec(base, out);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Internal helper types
// ---------------------------------------------------------------------------

/// Internal: getter candidate info.
struct GetterInfo {
    method_name: String,
    field: String,
}

/// Internal: setter candidate info.
struct SetterInfo {
    method_name: String,
    field: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::csharp::{BinOp, CmpKind};

    fn make_getter(field: &str) -> Vec<Stmt> {
        vec![Stmt::Return(Some(Expr::Field(
            Box::new(Expr::This),
            field.to_string(),
        )))]
    }

    fn make_setter(field: &str) -> Vec<Stmt> {
        vec![Stmt::Assign(field.to_string(), None, Expr::Var("value".into()))]
    }

    // -- Simple auto-property tests --

    #[test]
    fn detect_simple_auto_property() {
        let mut methods = HashMap::new();
        methods.insert("get_Name".into(), make_getter("_name"));
        methods.insert("set_Name".into(), make_setter("_name"));

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.auto_properties.len(), 1);
        assert_eq!(result.auto_properties[0].name, "Name");
        assert_eq!(result.auto_properties[0].backing_field, "_name");
        assert!(result.auto_properties[0].has_getter);
        assert!(result.auto_properties[0].has_setter);
        assert_eq!(result.auto_properties[0].getter_name, "get_Name");
        assert_eq!(result.auto_properties[0].setter_name, "set_Name");
        assert_eq!(result.counters.simple_auto_properties, 1);
    }

    #[test]
    fn getter_only_property() {
        let mut methods = HashMap::new();
        methods.insert("get_Count".into(), make_getter("_count"));

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.auto_properties.len(), 1);
        assert_eq!(result.auto_properties[0].name, "Count");
        assert!(result.auto_properties[0].has_getter);
        assert!(!result.auto_properties[0].has_setter);
        assert_eq!(result.counters.getter_only_properties, 1);
    }

    #[test]
    fn setter_only_property() {
        let mut methods = HashMap::new();
        methods.insert("set_Value".into(), make_setter("_value"));

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.auto_properties.len(), 1);
        assert_eq!(result.auto_properties[0].name, "Value");
        assert!(!result.auto_properties[0].has_getter);
        assert!(result.auto_properties[0].has_setter);
        assert_eq!(result.counters.setter_only_properties, 1);
    }

    #[test]
    fn mismatched_fields_not_paired() {
        let mut methods = HashMap::new();
        methods.insert("get_X".into(), make_getter("_x"));
        methods.insert("set_X".into(), make_setter("_y")); // Different field!

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.auto_properties.len(), 2);
        for p in &result.auto_properties {
            assert!(!(p.has_getter && p.has_setter));
        }
    }

    #[test]
    fn non_getter_setter_methods_ignored() {
        let mut methods = HashMap::new();
        methods.insert("DoWork".into(), vec![Stmt::Eval(Expr::Var("x".into()))]);
        methods.insert(
            "Calculate".into(),
            vec![Stmt::Return(Some(Expr::Int(42)))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.auto_properties.len(), 0);
        assert_eq!(result.computed_properties.len(), 0);
    }

    #[test]
    fn setter_with_non_value_param_not_matched() {
        let mut methods = HashMap::new();
        methods.insert(
            "set_X".into(),
            vec![Stmt::Assign("_x".into(), None, Expr::Int(0))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(
            result
                .auto_properties
                .iter()
                .filter(|p| p.name == "X")
                .count(),
            0,
            "setter not using 'value' param should not match"
        );
        assert_eq!(result.counters.rejected_setter_non_value, 1);
    }

    #[test]
    fn getter_on_non_this_receiver_not_matched() {
        let mut methods = HashMap::new();
        methods.insert(
            "get_X".into(),
            vec![Stmt::Return(Some(Expr::Field(
                Box::new(Expr::Var("other".into())),
                "_x".into(),
            )))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(
            result
                .auto_properties
                .iter()
                .filter(|p| p.name == "X")
                .count(),
            0,
            "getter on non-this receiver should not match"
        );
    }

    #[test]
    fn multiple_properties_detected() {
        let mut methods = HashMap::new();
        methods.insert("get_A".into(), make_getter("_a"));
        methods.insert("set_A".into(), make_setter("_a"));
        methods.insert("get_B".into(), make_getter("_b"));
        methods.insert("set_B".into(), make_setter("_b"));
        methods.insert("get_C".into(), make_getter("_c"));

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.auto_properties.len(), 3);

        let a = result.auto_properties.iter().find(|p| p.name == "A").unwrap();
        assert!(a.has_getter && a.has_setter);

        let b = result.auto_properties.iter().find(|p| p.name == "B").unwrap();
        assert!(b.has_getter && b.has_setter);

        let c = result.auto_properties.iter().find(|p| p.name == "C").unwrap();
        assert!(c.has_getter && !c.has_setter);

        assert_eq!(result.counters.simple_auto_properties, 2);
        assert_eq!(result.counters.getter_only_properties, 1);
    }

    #[test]
    fn empty_methods_map() {
        let methods = HashMap::new();
        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.auto_properties.len(), 0);
        assert_eq!(result.computed_properties.len(), 0);
        assert_eq!(result.counters, PropertyCounters::default());
    }

    // -- Computed property tests --

    #[test]
    fn computed_property_binop_field_and_literal() {
        // get_DoubleX() { return this._x * 2; }
        let mut methods = HashMap::new();
        methods.insert(
            "get_DoubleX".into(),
            vec![Stmt::Return(Some(Expr::Bin(
                BinOp::Mul,
                Box::new(Expr::Field(Box::new(Expr::This), "_x".into())),
                Box::new(Expr::Int(2)),
            )))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        // Should NOT be a simple auto-property (return is not a plain field).
        assert_eq!(
            result
                .auto_properties
                .iter()
                .filter(|p| p.name == "DoubleX")
                .count(),
            0
        );
        // Should be a computed property.
        assert_eq!(result.computed_properties.len(), 1);
        let cp = &result.computed_properties[0];
        assert_eq!(cp.name, "DoubleX");
        assert_eq!(cp.getter_name, "get_DoubleX");
        assert_eq!(cp.fields_read, vec!["_x"]);
        assert_eq!(result.counters.computed_properties, 1);
    }

    #[test]
    fn computed_property_two_fields_added() {
        // get_Sum() { return this._a + this._b; }
        let mut methods = HashMap::new();
        methods.insert(
            "get_Sum".into(),
            vec![Stmt::Return(Some(Expr::Bin(
                BinOp::Add,
                Box::new(Expr::Field(Box::new(Expr::This), "_a".into())),
                Box::new(Expr::Field(Box::new(Expr::This), "_b".into())),
            )))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 1);
        let cp = &result.computed_properties[0];
        assert_eq!(cp.name, "Sum");
        assert_eq!(cp.fields_read, vec!["_a", "_b"]);
    }

    #[test]
    fn computed_property_with_temp_assignment() {
        // get_IsReady() {
        //   tmp = this._state;
        //   return tmp != 0;
        // }
        let mut methods = HashMap::new();
        methods.insert(
            "get_IsReady".into(),
            vec![
                Stmt::Assign(
                    "tmp".into(),
                    None,
                    Expr::Field(Box::new(Expr::This), "_state".into()),
                ),
                Stmt::Return(Some(Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(Expr::Var("tmp".into())),
                    Box::new(Expr::Int(0)),
                ))),
            ],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 1);
        let cp = &result.computed_properties[0];
        assert_eq!(cp.name, "IsReady");
        assert_eq!(cp.fields_read, vec!["_state"]);
        assert_eq!(cp.temporaries, vec!["tmp"]);
    }

    #[test]
    fn computed_property_three_stmts() {
        // get_Computed() {
        //   t1 = this._x;
        //   t2 = this._y;
        //   return t1 + t2;
        // }
        let mut methods = HashMap::new();
        methods.insert(
            "get_Computed".into(),
            vec![
                Stmt::Assign(
                    "t1".into(),
                    None,
                    Expr::Field(Box::new(Expr::This), "_x".into()),
                ),
                Stmt::Assign(
                    "t2".into(),
                    None,
                    Expr::Field(Box::new(Expr::This), "_y".into()),
                ),
                Stmt::Return(Some(Expr::Bin(
                    BinOp::Add,
                    Box::new(Expr::Var("t1".into())),
                    Box::new(Expr::Var("t2".into())),
                ))),
            ],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 1);
        let cp = &result.computed_properties[0];
        assert_eq!(cp.name, "Computed");
        assert_eq!(cp.fields_read, vec!["_x", "_y"]);
        assert_eq!(cp.temporaries, vec!["t1", "t2"]);
    }

    #[test]
    fn computed_property_rejected_with_call() {
        // get_X() { return this.GetValue(); }  — has a call, not safe.
        let mut methods = HashMap::new();
        methods.insert(
            "get_X".into(),
            vec![Stmt::Return(Some(Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "GetValue".into(),
                args: vec![],
            }))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 0);
        assert_eq!(result.counters.rejected_complex_expr, 1);
    }

    #[test]
    fn computed_property_rejected_too_many_stmts() {
        // 4 statements — exceeds MAX_COMPUTED_STMTS.
        let mut methods = HashMap::new();
        methods.insert(
            "get_X".into(),
            vec![
                Stmt::Assign(
                    "a".into(),
                    None,
                    Expr::Field(Box::new(Expr::This), "_a".into()),
                ),
                Stmt::Assign(
                    "b".into(),
                    None,
                    Expr::Field(Box::new(Expr::This), "_b".into()),
                ),
                Stmt::Assign(
                    "c".into(),
                    None,
                    Expr::Bin(
                        BinOp::Add,
                        Box::new(Expr::Var("a".into())),
                        Box::new(Expr::Var("b".into())),
                    ),
                ),
                Stmt::Return(Some(Expr::Var("c".into()))),
            ],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 0);
    }

    #[test]
    fn computed_property_no_field_read_rejected() {
        // get_X() { return 42; } — no field reads, not a property.
        let mut methods = HashMap::new();
        methods.insert(
            "get_X".into(),
            vec![Stmt::Return(Some(Expr::Int(42)))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 0);
    }

    #[test]
    fn computed_property_negated_field() {
        // get_NegX() { return -this._x; }
        let mut methods = HashMap::new();
        methods.insert(
            "get_NegX".into(),
            vec![Stmt::Return(Some(Expr::Neg(Box::new(Expr::Field(
                Box::new(Expr::This),
                "_x".into(),
            )))))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 1);
        let cp = &result.computed_properties[0];
        assert_eq!(cp.name, "NegX");
        assert_eq!(cp.fields_read, vec!["_x"]);
    }

    #[test]
    fn computed_property_ternary() {
        // get_Display() { return this._enabled ? this._onText : this._offText; }
        let mut methods = HashMap::new();
        methods.insert(
            "get_Display".into(),
            vec![Stmt::Return(Some(Expr::Cond(
                Box::new(Expr::Field(Box::new(Expr::This), "_enabled".into())),
                Box::new(Expr::Field(Box::new(Expr::This), "_onText".into())),
                Box::new(Expr::Field(Box::new(Expr::This), "_offText".into())),
            )))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 1);
        let cp = &result.computed_properties[0];
        assert_eq!(cp.name, "Display");
        assert_eq!(cp.fields_read, vec!["_enabled", "_offText", "_onText"]);
    }

    #[test]
    fn computed_property_with_intermediate_call_rejected() {
        // get_X() {
        //   tmp = this.GetValue();  // call in intermediate — reject
        //   return tmp + 1;
        // }
        let mut methods = HashMap::new();
        methods.insert(
            "get_X".into(),
            vec![
                Stmt::Assign(
                    "tmp".into(),
                    None,
                    Expr::Call {
                        recv: Some(Box::new(Expr::This)),
                        name: "GetValue".into(),
                        args: vec![],
                    },
                ),
                Stmt::Return(Some(Expr::Bin(
                    BinOp::Add,
                    Box::new(Expr::Var("tmp".into())),
                    Box::new(Expr::Int(1)),
                ))),
            ],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 0);
    }

    // -- Backing field detection tests --

    #[test]
    fn compiler_backing_field_detection() {
        let mut methods = HashMap::new();
        methods.insert(
            "get_Name".into(),
            make_getter("<Name>k__BackingField"),
        );
        methods.insert(
            "set_Name".into(),
            make_setter("<Name>k__BackingField"),
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.auto_properties.len(), 1);
        assert_eq!(result.auto_properties[0].backing_field, "<Name>k__BackingField");
        assert_eq!(result.counters.compiler_backing_fields, 2); // getter + setter
    }

    #[test]
    fn is_compiler_backing_field_works() {
        assert!(is_compiler_backing_field("<Name>k__BackingField"));
        assert!(is_compiler_backing_field("<X>k__BackingField"));
        assert!(!is_compiler_backing_field("_name"));
        assert!(!is_compiler_backing_field("<Name>"));
        assert!(!is_compiler_backing_field("k__BackingField"));
    }

    #[test]
    fn property_name_from_backing_field_works() {
        assert_eq!(
            property_name_from_backing_field("<Name>k__BackingField"),
            Some("Name".into())
        );
        assert_eq!(
            property_name_from_backing_field("<X>k__BackingField"),
            Some("X".into())
        );
        assert_eq!(property_name_from_backing_field("_name"), None);
        assert_eq!(property_name_from_backing_field("<>k__BackingField"), None);
    }

    // -- Instrumentation counter tests --

    #[test]
    fn counters_track_examined_methods() {
        let mut methods = HashMap::new();
        methods.insert("get_A".into(), make_getter("_a"));
        methods.insert("set_A".into(), make_setter("_a"));
        methods.insert("get_B".into(), make_getter("_b"));
        methods.insert("DoWork".into(), vec![Stmt::Eval(Expr::Var("x".into()))]);

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.counters.getters_examined, 2);
        assert_eq!(result.counters.setters_examined, 1);
    }

    #[test]
    fn counters_distinguish_simple_vs_computed() {
        let mut methods = HashMap::new();
        // Simple auto-property.
        methods.insert("get_A".into(), make_getter("_a"));
        methods.insert("set_A".into(), make_setter("_a"));
        // Computed property.
        methods.insert(
            "get_B".into(),
            vec![Stmt::Return(Some(Expr::Bin(
                BinOp::Add,
                Box::new(Expr::Field(Box::new(Expr::This), "_x".into())),
                Box::new(Expr::Int(1)),
            )))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.counters.simple_auto_properties, 1);
        assert_eq!(result.counters.computed_properties, 1);
        assert_eq!(result.auto_properties.len(), 1);
        assert_eq!(result.computed_properties.len(), 1);
    }

    // -- collect_this_fields tests --

    #[test]
    fn collect_this_fields_simple() {
        let expr = Expr::Field(Box::new(Expr::This), "_x".into());
        assert_eq!(collect_this_fields(&expr), vec!["_x"]);
    }

    #[test]
    fn collect_this_fields_nested() {
        let expr = Expr::Bin(
            BinOp::Add,
            Box::new(Expr::Field(Box::new(Expr::This), "_a".into())),
            Box::new(Expr::Field(Box::new(Expr::This), "_b".into())),
        );
        assert_eq!(collect_this_fields(&expr), vec!["_a", "_b"]);
    }

    #[test]
    fn collect_this_fields_ignores_non_this() {
        let expr = Expr::Field(Box::new(Expr::Var("other".into())), "_x".into());
        assert_eq!(collect_this_fields(&expr), Vec::<String>::new());
    }

    // -- Backwards compatibility --

    #[test]
    fn backwards_compat_detect_auto_properties() {
        // The old API `detect_auto_properties` should still work.
        let mut methods = HashMap::new();
        methods.insert("get_Name".into(), make_getter("_name"));
        methods.insert("set_Name".into(), make_setter("_name"));

        let props = detect_auto_properties(&methods);
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].name, "Name");
    }

    #[test]
    fn getter_with_computed_return_not_matched_as_simple() {
        // A getter that returns a computed expression should not match as
        // a simple auto-property, but should match as a computed property.
        let mut methods = HashMap::new();
        methods.insert(
            "get_X".into(),
            vec![Stmt::Return(Some(Expr::Bin(
                BinOp::Add,
                Box::new(Expr::Field(Box::new(Expr::This), "_x".into())),
                Box::new(Expr::Int(1)),
            )))],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(
            result
                .auto_properties
                .iter()
                .filter(|p| p.name == "X")
                .count(),
            0,
            "computed getter should not match as simple auto-property"
        );
        assert_eq!(result.computed_properties.len(), 1);
        assert_eq!(result.computed_properties[0].name, "X");
    }

    #[test]
    fn getter_with_multiple_non_temp_statements_not_matched() {
        // A getter with Eval (side-effect) + return — not a computed property.
        let mut methods = HashMap::new();
        methods.insert(
            "get_X".into(),
            vec![
                Stmt::Eval(Expr::Var("side_effect".into())),
                Stmt::Return(Some(Expr::Field(Box::new(Expr::This), "_x".into()))),
            ],
        );

        let result = detect_auto_properties_with_counters(&methods);
        assert_eq!(result.computed_properties.len(), 0);
    }
}
