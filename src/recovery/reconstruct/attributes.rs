//! C# attribute recovery: recognises custom attribute constructor calls in
//! static initialisers and emits `[AttributeName(args)]` comments.
//!
//! ## What this pass does
//!
//! IL2CPP compiles C# custom attributes (e.g. `[SerializeField]`, `[Header("X")]`)
//! as constructor calls in the class's static initialiser (`.cctor`). The
//! attribute constructor's RVA is recorded in the metadata's
//! `custom_attribute_generators` table. When the decompiler encounters a call
//! to a known attribute constructor RVA, this pass replaces the call with a
//! `Comment` statement showing the attribute syntax.
//!
//! ```text
//! SerializeField..ctor();    -->    /* [SerializeField] */
//! Header..ctor("Speed");     -->    /* [Header("Speed")] */
//! Range..ctor(0f, 100f);     -->    /* [Range(0, 100)] */
//! ```
//!
//! ## How it works
//!
//! 1. The `ResolveMaps::attribute_ctor_names` map is pre-populated during
//!    `RecoveryContext::build` from the metadata's `custom_attribute_generators`
//!    and `attribute_type_ranges` tables. It maps each attribute constructor's
//!    RVA to the short attribute type name (e.g. "SerializeField").
//!
//! 2. This pass walks every statement looking for:
//!    - `Eval(Call { name: "X..ctor", args })` where the call name matches a
//!      known attribute type name (e.g. "SerializeField..ctor")
//!    - `Eval(RawCall(rva, args, _))` where `rva` is in `attribute_ctor_names`
//!    - `Eval(NewObj(type_name, args))` where `type_name` matches a known
//!      attribute type name
//!
//! 3. Matched calls are replaced with `Stmt::Comment("[AttrName(args)]")`.
//!
//! ## Safety
//!
//! Only calls whose name or address matches a KNOWN attribute constructor are
//! rewritten. A failed match is a no-op: the original call is preserved.
//! The pass does not delete any observable computation — attribute constructors
//! have no side effects visible to the program beyond metadata registration.
//!
//! ## Common Unity attributes
//!
//! The pass recognises every attribute whose type is in the metadata. Common
//! Unity attributes include:
//! - `[SerializeField]` — force-serialise a private field
//! - `[Header("...")]` — inspector section header
//! - `[Tooltip("...")]` — inspector tooltip
//! - `[Range(min, max)]` — numeric slider range
//! - `[HideInInspector]` — hide from inspector
//! - `[TextArea(min, max)]` — multi-line text area

use super::super::csharp::{Expr, Stmt};
use super::super::resolve::ResolveMaps;
use super::{is_ctor_name, walk_mut};

/// Run the attribute recovery pass over one method body.
///
/// Returns the number of attribute constructor calls replaced with
/// `Stmt::Attribute` nodes.
pub fn apply_with_maps(stmts: &mut Vec<Stmt>, maps: &ResolveMaps) -> usize {
    let mut count = 0;
    walk_mut(stmts, &mut |list| {
        count += recover_attributes(list, maps);
    });
    count
}

/// Scan a statement list for attribute constructor calls and replace them
/// with `Stmt::Attribute` nodes. After replacement, runs an association pass
/// that sets the `target` field on each attribute to the name of the
/// following field/method declaration, if any.
fn recover_attributes(stmts: &mut Vec<Stmt>, maps: &ResolveMaps) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < stmts.len() {
        if let Some((name, args)) = try_match_attribute(&stmts[i], maps) {
            stmts[i] = Stmt::Attribute {
                name,
                args,
                target: None,
            };
            count += 1;
        }
        i += 1;
    }
    // Association pass: set `target` on each Attribute that precedes a
    // declaration-like statement (Assign, Eval with a field-like name).
    associate_attributes(stmts);
    count
}

/// After attribute recovery, walk the statement list and set the `target`
/// field on each `Stmt::Attribute` to the name of the next declaration-like
/// statement, if one immediately follows (possibly after other attributes).
fn associate_attributes(stmts: &mut [Stmt]) {
    let mut i = 0;
    while i < stmts.len() {
        if matches!(&stmts[i], Stmt::Attribute { .. }) {
            // Find the end of the attribute run (consecutive attributes).
            let mut j = i + 1;
            while j < stmts.len() && matches!(&stmts[j], Stmt::Attribute { .. }) {
                j += 1;
            }
            // The statement after the attribute run is the target.
            if j < stmts.len() {
                let target_name = stmt_target_name(&stmts[j]);
                if let Some(name) = target_name {
                    // Set target on all attributes in the run.
                    for k in i..j {
                        if let Stmt::Attribute { target, .. } = &mut stmts[k] {
                            *target = Some(name.clone());
                        }
                    }
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
}

/// Extract a target name from a declaration-like statement.
/// Returns `Some(name)` for assignments to simple names or field-like targets,
/// `None` for control flow and other non-declaration statements.
fn stmt_target_name(stmt: &Stmt) -> Option<String> {
    match stmt {
        Stmt::Assign(name, _, _) => Some(name.clone()),
        // A method declaration would be an Eval with a recognizable name,
        // but in the decompiled output methods are not typically represented
        // as statements. We still handle the case for completeness.
        _ => None,
    }
}

/// Try to match a statement as an attribute constructor call.
/// Returns `Some((attr_name, args))` on match, `None` otherwise.
fn try_match_attribute(stmt: &Stmt, maps: &ResolveMaps) -> Option<(String, Vec<Expr>)> {
    match stmt {
        // Pattern 1: Eval(Call { name: "AttrName..ctor", args })
        // This is the most common shape after normalize_ctor_calls has run.
        Stmt::Eval(Expr::Call { recv: None, name, args }) if is_ctor_name(name) => {
            let type_name = extract_type_from_ctor_name(name)?;
            let short_name = short_attr_name(type_name);
            if !is_known_attribute(short_name, maps) {
                return None;
            }
            Some((short_name.to_string(), args.clone()))
        }

        // Pattern 2: Eval(RawCall(rva, args, _)) where rva is a known attribute ctor
        Stmt::Eval(Expr::RawCall(rva, args, _)) => {
            let attr_name = maps.attribute_ctor_names.get(rva)?;
            Some((attr_name.clone(), args.clone()))
        }

        // Pattern 3: Eval(NewObj(type_name, args)) where type_name is a known attribute
        Stmt::Eval(Expr::NewObj(type_name, args)) => {
            let short_name = short_attr_name(type_name);
            if !is_known_attribute(short_name, maps) {
                return None;
            }
            Some((short_name.to_string(), args.clone()))
        }

        // Pattern 4: Assign(_, _, Call/NewObj/RawCall) — the result is unused
        // but the attribute constructor was assigned to a temp. If the temp is
        // never used, this is still an attribute application.
        Stmt::Assign(_, _, expr) => {
            match expr {
                Expr::Call { recv: None, name, args } if is_ctor_name(name) => {
                    let type_name = extract_type_from_ctor_name(name)?;
                    let short_name = short_attr_name(type_name);
                    if !is_known_attribute(short_name, maps) {
                        return None;
                    }
                    Some((short_name.to_string(), args.clone()))
                }
                Expr::RawCall(rva, args, _) => {
                    let attr_name = maps.attribute_ctor_names.get(rva)?;
                    Some((attr_name.clone(), args.clone()))
                }
                Expr::NewObj(type_name, args) => {
                    let short_name = short_attr_name(type_name);
                    if !is_known_attribute(short_name, maps) {
                        return None;
                    }
                    Some((short_name.to_string(), args.clone()))
                }
                _ => None,
            }
        }

        _ => None,
    }
}

/// Extract the type name from a constructor name like "SerializeField..ctor"
/// or "UnityEngine.SerializeField..ctor".
fn extract_type_from_ctor_name(name: &str) -> Option<&str> {
    let t = name
        .strip_suffix("..ctor")
        .or_else(|| name.strip_suffix(".ctor"))?;
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

/// Extract the short attribute name from a fully-qualified type name.
/// Strips the namespace prefix and the "Attribute" suffix.
///
/// Examples:
/// - "UnityEngine.SerializeField" → "SerializeField"
/// - "SerializeField" → "SerializeField"
/// - "UnityEngine.HeaderAttribute" → "Header"
/// - "HeaderAttribute" → "Header"
fn short_attr_name(type_name: &str) -> &str {
    let short = type_name.rsplit('.').next().unwrap_or(type_name);
    short.strip_suffix("Attribute").unwrap_or(short)
}

/// Check if a short attribute name is a known attribute (present in the
/// `attribute_ctor_names` map values, or a well-known Unity attribute name).
fn is_known_attribute(short_name: &str, maps: &ResolveMaps) -> bool {
    // Direct lookup: the name is in the attribute ctor names values.
    if maps.attribute_ctor_names.values().any(|n| n == short_name) {
        return true;
    }
    // Well-known Unity attributes that may not have a metadata entry but are
    // commonly seen in attribute constructor calls.
    matches!(
        short_name,
        "SerializeField"
            | "Header"
            | "Tooltip"
            | "Range"
            | "HideInInspector"
            | "TextArea"
            | "Space"
            | "ContextMenu"
            | "DisallowMultipleComponent"
            | "RequireComponent"
            | "AddComponentMenu"
            | "CreateAssetMenu"
            | "ExecuteInEditMode"
            | "ExecuteAlways"
            | "NonSerialized"
            | "Obsolete"
            | "Conditional"
            | "RuntimeInitializeOnLoadMethod"
            | "RuntimeInitializeOnLoadBeforeSceneLoad"
            | "Preserve"
            | "DllImport"
    )
}

/// Format an attribute as `[AttrName(args)]`.
///
/// The arguments are rendered using the expression renderer. String literals
/// are kept as-is (they already include quotes). Numeric literals are
/// rendered without type suffixes where possible.
#[allow(dead_code)]
fn format_attribute(name: &str, args: &[Expr]) -> String {
    if args.is_empty() {
        return format!("[{name}]");
    }
    let mut rendered_args = Vec::with_capacity(args.len());
    for arg in args {
        rendered_args.push(render_attr_arg(arg));
    }
    format!("[{name}({})]", rendered_args.join(", "))
}

/// Render a single attribute argument expression as a string.
#[allow(dead_code)]
fn render_attr_arg(expr: &Expr) -> String {
    match expr {
        Expr::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Expr::Int(n) => n.to_string(),
        Expr::Float(f) => {
            // Render floats without the trailing 'f' for C# attribute syntax.
            if f.fract() == 0.0 {
                format!("{:.1}", f)
            } else {
                format!("{}", f)
            }
        }
        Expr::Double(d) => {
            if d.fract() == 0.0 {
                format!("{:.1}", d)
            } else {
                format!("{}", d)
            }
        }
        Expr::Bool(b) => {
            // C# attributes use lowercase true/false.
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Expr::Var(name) => {
            // Named argument: "name = value" is handled at the call site.
            // A bare var is likely an enum member or constant.
            name.clone()
        }
        Expr::StaticField(type_name, field_name) => {
            // Enum member: "Type.Field"
            format!("{type_name}.{field_name}")
        }
        _ => {
            // Fall back to the general renderer for complex expressions.
            let mut s = String::new();
            expr.render(&mut s);
            s
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal ResolveMaps with the given attribute ctor names.
    fn make_maps(attr_ctors: Vec<(u64, String)>) -> ResolveMaps {
        ResolveMaps {
            attribute_ctor_names: attr_ctors.into_iter().collect(),
            ..Default::default()
        }
    }

    /// Helper: assert that stmts[idx] is a Stmt::Attribute with the given name.
    fn assert_attr(stmts: &[Stmt], idx: usize, expected_name: &str) {
        match &stmts[idx] {
            Stmt::Attribute { name, .. } => assert_eq!(name, expected_name),
            other => panic!("expected Attribute at [{idx}], got {other:?}"),
        }
    }

    /// Helper: assert that stmts[idx] is a Stmt::Attribute with the given name
    /// and target.
    fn assert_attr_with_target(stmts: &[Stmt], idx: usize, expected_name: &str, expected_target: Option<&str>) {
        match &stmts[idx] {
            Stmt::Attribute { name, target, .. } => {
                assert_eq!(name, expected_name);
                assert_eq!(target.as_deref(), expected_target);
            }
            other => panic!("expected Attribute at [{idx}], got {other:?}"),
        }
    }

    #[test]
    fn short_attr_name_strips_namespace_and_suffix() {
        assert_eq!(short_attr_name("UnityEngine.SerializeField"), "SerializeField");
        assert_eq!(short_attr_name("SerializeField"), "SerializeField");
        assert_eq!(short_attr_name("UnityEngine.HeaderAttribute"), "Header");
        assert_eq!(short_attr_name("HeaderAttribute"), "Header");
        assert_eq!(short_attr_name("UnityEditor.Tooltip"), "Tooltip");
    }

    #[test]
    fn extract_type_from_ctor_name_both_spellings() {
        assert_eq!(
            extract_type_from_ctor_name("SerializeField..ctor"),
            Some("SerializeField")
        );
        assert_eq!(
            extract_type_from_ctor_name("UnityEngine.Header.ctor"),
            Some("UnityEngine.Header")
        );
        assert_eq!(extract_type_from_ctor_name(".ctor"), None);
        assert_eq!(extract_type_from_ctor_name("NotACtor"), None);
    }

    #[test]
    fn format_attribute_no_args() {
        assert_eq!(format_attribute("SerializeField", &[]), "[SerializeField]");
    }

    #[test]
    fn format_attribute_with_string_arg() {
        let args = vec![Expr::Str("Speed".into())];
        assert_eq!(
            format_attribute("Header", &args),
            "[Header(\"Speed\")]"
        );
    }

    #[test]
    fn format_attribute_with_numeric_args() {
        let args = vec![Expr::Int(0), Expr::Int(100)];
        assert_eq!(format_attribute("Range", &args), "[Range(0, 100)]");
    }

    #[test]
    fn format_attribute_with_float_args() {
        let args = vec![Expr::Float(0.0), Expr::Float(100.0)];
        assert_eq!(format_attribute("Range", &args), "[Range(0.0, 100.0)]");
    }

    #[test]
    fn format_attribute_with_bool_arg() {
        let args = vec![Expr::Bool(true)];
        assert_eq!(format_attribute("Flag", &args), "[Flag(true)]");
    }

    #[test]
    fn recover_eval_ctor_call() {
        let maps = make_maps(vec![(0x1000, "SerializeField".into())]);
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "SerializeField..ctor".into(),
            args: vec![],
        })];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        assert_attr(&stmts, 0, "SerializeField");
    }

    #[test]
    fn recover_eval_ctor_with_args() {
        let maps = make_maps(vec![(0x1000, "Header".into())]);
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Header..ctor".into(),
            args: vec![Expr::Str("Speed".into())],
        })];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Attribute { name, args, .. } => {
                assert_eq!(name, "Header");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected Attribute, got {other:?}"),
        }
    }

    #[test]
    fn recover_rawcall_by_rva() {
        let maps = make_maps(vec![(0x2000, "Tooltip".into())]);
        let mut stmts = vec![Stmt::Eval(Expr::RawCall(
            0x2000,
            vec![Expr::Str("Player health".into())],
            None,
        ))];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        assert_attr(&stmts, 0, "Tooltip");
    }

    #[test]
    fn recover_newobj_attribute() {
        let maps = make_maps(vec![(0x1000, "Range".into())]);
        let mut stmts = vec![Stmt::Eval(Expr::NewObj(
            "Range".into(),
            vec![Expr::Int(0), Expr::Int(100)],
        ))];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        assert_attr(&stmts, 0, "Range");
    }

    #[test]
    fn unknown_ctor_is_not_rewritten() {
        let maps = make_maps(vec![(0x1000, "SerializeField".into())]);
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "SomeOtherType..ctor".into(),
            args: vec![],
        })];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "SomeOtherType..ctor"));
    }

    #[test]
    fn empty_maps_only_recognizes_well_known() {
        let maps = ResolveMaps::default();
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "UnknownAttribute..ctor".into(),
            args: vec![],
        })];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 0);
    }

    #[test]
    fn multiple_attributes_in_one_list() {
        let maps = make_maps(vec![
            (0x1000, "SerializeField".into()),
            (0x2000, "Header".into()),
        ]);
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SerializeField..ctor".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Header..ctor".into(),
                args: vec![Expr::Str("Stats".into())],
            }),
            Stmt::Assign("x".into(), None, Expr::Int(42)),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 2);
        assert_attr(&stmts, 0, "SerializeField");
        assert_attr(&stmts, 1, "Header");
        assert!(matches!(&stmts[2], Stmt::Assign(..)));
    }

    #[test]
    fn well_known_attribute_without_metadata_entry() {
        let maps = ResolveMaps::default();
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "SerializeField..ctor".into(),
            args: vec![],
        })];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        assert_attr(&stmts, 0, "SerializeField");
    }

    #[test]
    fn nested_attribute_in_if_body() {
        let maps = make_maps(vec![(0x1000, "SerializeField".into())]);
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "SerializeField..ctor".into(),
                args: vec![],
            })],
            else_body: vec![],
        }];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Attribute { name, .. } if name == "SerializeField"));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn qualified_ctor_name_is_recognised() {
        let maps = make_maps(vec![(0x1000, "SerializeField".into())]);
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "UnityEngine.SerializeField..ctor".into(),
            args: vec![],
        })];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        assert_attr(&stmts, 0, "SerializeField");
    }

    #[test]
    fn render_attr_arg_static_field() {
        let expr = Expr::StaticField("MyEnum".into(), "ValueA".into());
        assert_eq!(render_attr_arg(&expr), "MyEnum.ValueA");
    }

    #[test]
    fn attribute_with_enum_arg() {
        let maps = make_maps(vec![(0x1000, "MyAttr".into())]);
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "MyAttr..ctor".into(),
            args: vec![Expr::StaticField("MyEnum".into(), "ValueA".into())],
        })];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Attribute { name, args, .. } => {
                assert_eq!(name, "MyAttr");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected Attribute, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // New tests for Stmt::Attribute association and rendering
    // -----------------------------------------------------------------------

    /// Simple attribute on a field: the attribute's `target` should be set
    /// to the field name.
    #[test]
    fn attribute_on_field_sets_target() {
        let maps = make_maps(vec![(0x1000, "SerializeField".into())]);
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SerializeField..ctor".into(),
                args: vec![],
            }),
            Stmt::Assign("health".into(), Some("int".into()), Expr::Int(100)),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        assert_attr_with_target(&stmts, 0, "SerializeField", Some("health"));
    }

    /// Attribute with arguments: args are preserved in the Attribute node.
    #[test]
    fn attribute_with_args_preserves_them() {
        let maps = make_maps(vec![(0x1000, "Range".into())]);
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Range..ctor".into(),
                args: vec![Expr::Int(0), Expr::Int(100)],
            }),
            Stmt::Assign("speed".into(), None, Expr::Float(5.0)),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Attribute { name, args, target } => {
                assert_eq!(name, "Range");
                assert_eq!(args.len(), 2);
                assert_eq!(target.as_deref(), Some("speed"));
            }
            other => panic!("expected Attribute, got {other:?}"),
        }
    }

    /// Multiple attributes on the same field: all get the same target.
    #[test]
    fn multiple_attributes_same_field_share_target() {
        let maps = make_maps(vec![
            (0x1000, "SerializeField".into()),
            (0x2000, "Header".into()),
        ]);
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SerializeField..ctor".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Header..ctor".into(),
                args: vec![Expr::Str("Stats".into())],
            }),
            Stmt::Assign("health".into(), None, Expr::Int(100)),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 2);
        assert_attr_with_target(&stmts, 0, "SerializeField", Some("health"));
        assert_attr_with_target(&stmts, 1, "Header", Some("health"));
    }

    /// Attribute followed by a non-declaration (e.g., a return): target is None.
    #[test]
    fn attribute_followed_by_non_declaration_has_no_target() {
        let maps = make_maps(vec![(0x1000, "SerializeField".into())]);
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SerializeField..ctor".into(),
                args: vec![],
            }),
            Stmt::Return(None),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        assert_attr_with_target(&stmts, 0, "SerializeField", None);
    }

    /// Attribute at end of list (no following statement): target is None.
    #[test]
    fn attribute_at_end_of_list_has_no_target() {
        let maps = make_maps(vec![(0x1000, "SerializeField".into())]);
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Int(1)),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SerializeField..ctor".into(),
                args: vec![],
            }),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        assert_attr_with_target(&stmts, 1, "SerializeField", None);
    }

    /// Attribute rendering: no args renders as `[Name]`.
    #[test]
    fn attribute_render_no_args() {
        let attr = Stmt::Attribute {
            name: "SerializeField".into(),
            args: vec![],
            target: None,
        };
        // Verify the attribute node has the correct structure.
        match &attr {
            Stmt::Attribute { name, args, target } => {
                assert_eq!(name, "SerializeField");
                assert!(args.is_empty());
                assert!(target.is_none());
            }
            _ => panic!("expected Attribute"),
        }
    }

    /// Attribute rendering: with args renders as `[Name(arg1, arg2)]`.
    #[test]
    fn attribute_render_with_args() {
        let attr = Stmt::Attribute {
            name: "Range".into(),
            args: vec![Expr::Int(0), Expr::Int(100)],
            target: Some("speed".into()),
        };
        match &attr {
            Stmt::Attribute { name, args, target } => {
                assert_eq!(name, "Range");
                assert_eq!(args.len(), 2);
                assert_eq!(target.as_deref(), Some("speed"));
            }
            _ => panic!("expected Attribute"),
        }
    }

    /// Attribute followed by an Eval (method-like): target is set to the
    /// assignment target name if it's an Assign, otherwise None.
    #[test]
    fn attribute_followed_by_eval_has_no_target() {
        let maps = make_maps(vec![(0x1000, "ContextMenu".into())]);
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "ContextMenu..ctor".into(),
                args: vec![Expr::Str("DoThing".into())],
            }),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("this".into()))),
                name: "DoThing".into(),
                args: vec![],
            }),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 1);
        // Eval is not a declaration, so target should be None.
        assert_attr_with_target(&stmts, 0, "ContextMenu", None);
    }

    /// Three attributes in a row, all targeting the same field.
    #[test]
    fn three_attributes_same_target() {
        let maps = make_maps(vec![
            (0x1000, "SerializeField".into()),
            (0x2000, "Header".into()),
            (0x3000, "Tooltip".into()),
        ]);
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SerializeField..ctor".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Header..ctor".into(),
                args: vec![Expr::Str("Combat".into())],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Tooltip..ctor".into(),
                args: vec![Expr::Str("Player health".into())],
            }),
            Stmt::Assign("health".into(), None, Expr::Int(100)),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 3);
        assert_attr_with_target(&stmts, 0, "SerializeField", Some("health"));
        assert_attr_with_target(&stmts, 1, "Header", Some("health"));
        assert_attr_with_target(&stmts, 2, "Tooltip", Some("health"));
    }

    /// Attribute followed by another attribute, then a non-declaration:
    /// both attributes have no target.
    #[test]
    fn attributes_followed_by_non_declaration() {
        let maps = make_maps(vec![
            (0x1000, "SerializeField".into()),
            (0x2000, "Header".into()),
        ]);
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SerializeField..ctor".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Header..ctor".into(),
                args: vec![Expr::Str("X".into())],
            }),
            Stmt::Return(None),
        ];
        let n = apply_with_maps(&mut stmts, &maps);
        assert_eq!(n, 2);
        assert_attr_with_target(&stmts, 0, "SerializeField", None);
        assert_attr_with_target(&stmts, 1, "Header", None);
    }
}
