//! Retype allocations from the declaration they are stored into.
//!
//! Shared-generic metadata names an allocation's constructor with the generic
//! DEFINITION and `object` arguments — `Dictionary<int, object>..ctor` — even
//! when the field it is stored into is declared `Dictionary<int, Token>`.
//! Likewise an obfuscated/default constructor folds onto `Object..ctor`, so a
//! `Token` allocation renders `new Object()` although the IR typed the local
//! `Token` by value flow.
//!
//! This pass retypes the constructed type from:
//!  * the declared generic instantiation of the field it is assigned to
//!    (`maps.field_generic_text`), when both are the same generic definition;
//!  * the local's class type (`local_types`), only when the constructed type is
//!    a bare placeholder (`object`/`Object`/`__unknown`) and the type is not an
//!    interface (a `new I()` would be invalid).
//!
//! Only the type NAME changes, and only in the direction the metadata proves.

use super::super::csharp::{Expr, Stmt};
use super::super::resolve::ResolveMaps;
use super::children_mut;
use std::collections::HashMap;

fn is_placeholder(ty: &str) -> bool {
    matches!(ty, "object" | "Object" | "System.Object" | "__unknown" | "")
}

/// Generic definition without arguments or namespace: `System.Collections.
/// Generic.Dictionary<int, X>` → `Dictionary`.
fn generic_base(text: &str) -> &str {
    let t = text.trim();
    // Strip the namespace from the DEFINITION only; the last dot may sit
    // inside the argument list (`Dictionary<int, TokenManager.Token>`).
    let head = t.split('<').next().unwrap_or(t);
    head.rsplit('.').next().unwrap_or(head)
}

fn clean_generic_text(text: &str) -> String {
    // `Dictionary`2<int, T>` → `Dictionary<int, T>` (backtick is metadata
    // arity syntax, not C#).
    text.replace('`', "")
}

fn is_simple_local(n: &str) -> bool {
    !n.is_empty()
        && !n.contains('.')
        && !n.contains('[')
        && !n.contains('*')
        && !n.contains('&')
        && !n.contains('<')
        && !n.contains('>')
        && n.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !n.chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// `this.<field>` — the only receiver whose declared type this method knows
/// without extra type plumbing.
fn this_field_name(target: &str) -> Option<&str> {
    let rest = target.strip_prefix("this.")?;
    if rest.is_empty()
        || rest.contains('.')
        || rest.contains('[')
        || rest.contains(' ')
        || rest.contains('*')
        || rest.contains('&')
    {
        return None;
    }
    Some(rest)
}

fn field_text<'a>(target: &str, maps: &'a ResolveMaps, declaring_tdi: i32) -> Option<&'a String> {
    let field = this_field_name(target)?;
    if let Some(chain) = maps.type_parent_chains.get(&declaring_tdi) {
        for &tdi in chain {
            if let Some(t) = maps.field_generic_text.get(&(tdi, field.to_string())) {
                return Some(t);
            }
        }
    }
    maps.field_generic_text
        .get(&(declaring_tdi, field.to_string()))
}

fn class_name_for(tdi: i32, maps: &ResolveMaps) -> Option<String> {
    if maps.interface_tdis.contains(&tdi) {
        return None;
    }
    let (_, name) = maps.tdi_names.get(&tdi)?;
    if name.is_empty() {
        return None;
    }
    Some(name.clone())
}

fn retype(expr: &mut Expr, new_ty: &str) -> bool {
    let slot = match expr {
        Expr::NewObj(ty, _)
        | Expr::CollectionInit { type_name: ty, .. }
        | Expr::ObjInit { type_name: ty, .. } => ty,
        _ => return false,
    };
    if *slot == new_ty {
        return false;
    }
    *slot = new_ty.to_string();
    true
}

fn apply_stmt(
    s: &mut Stmt,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
    n: &mut usize,
) {
    if let Stmt::Assign(target, _, rhs) = s {
        if std::env::var("ILR_ALLOCDBG").is_ok()
            && matches!(rhs, Expr::NewObj(..) | Expr::CollectionInit { .. } | Expr::ObjInit { .. })
        {
            eprintln!(
                "[alloc] candidate target={target:?} rhs_ty={:?} field_text={:?}",
                match rhs {
                    Expr::NewObj(t, _) => t.as_str(),
                    Expr::CollectionInit { type_name, .. } => type_name.as_str(),
                    Expr::ObjInit { type_name, .. } => type_name.as_str(),
                    _ => "",
                },
                field_text(target, maps, declaring_tdi)
            );
        }
        // 1. The declared field's generic instantiation is authoritative.
        if let Some(text) = field_text(target, maps, declaring_tdi) {
            let same_def = match rhs {
                Expr::NewObj(ty, _)
                | Expr::CollectionInit { type_name: ty, .. }
                | Expr::ObjInit { type_name: ty, .. } => {
                    is_placeholder(ty) || generic_base(ty) == generic_base(text)
                }
                _ => false,
            };
            if same_def {
                let text = clean_generic_text(text);
                if text.contains('<') && retype(rhs, &text) {
                    if std::env::var("ILR_ALLOCDBG").is_ok() {
                        eprintln!("[alloc] field type → {text}");
                    }
                    *n += 1;
                }
            }
        }
        // 2. A bare `new Object()` whose local the IR typed as a class.
        if is_simple_local(target) {
            if let Expr::NewObj(ty, _) = rhs {
                if is_placeholder(ty) {
                    if let Some(&tdi) = local_types.get(target.as_str()) {
                        if let Some(class) = class_name_for(tdi, maps) {
                            if retype(rhs, &class) {
                                if std::env::var("ILR_ALLOCDBG").is_ok() {
                                    eprintln!("[alloc] {target}: placeholder → {class}");
                                }
                                *n += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    for k in children_mut(s) {
        for x in k.iter_mut() {
            apply_stmt(x, maps, local_types, declaring_tdi, n);
        }
    }
}

/// Run the pass. Returns the number of retyped allocations.
pub fn apply(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
) -> usize {
    let mut n = 0;
    for s in stmts.iter_mut() {
        apply_stmt(s, maps, local_types, declaring_tdi, &mut n);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps() -> ResolveMaps {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(301, ("".into(), "Token".into()));
        m.tdi_names.insert(700, ("".into(), "IDamageable".into()));
        m.interface_tdis.insert(700);
        m.type_parent_chains.insert(302, vec![302, 1]);
        m.field_generic_text
            .insert((302, "_tokens".into()), "Dictionary<int, Token>".into());
        m
    }

    #[test]
    fn field_generic_instantiation_retypes_allocation() {
        let mut stmts = vec![Stmt::Assign(
            "this._tokens".into(),
            None,
            Expr::NewObj("Dictionary<int, object>".into(), vec![]),
        )];
        assert_eq!(apply(&mut stmts, &maps(), &HashMap::new(), 302), 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("Dictionary<int, Token>"), "{t}");
    }

    #[test]
    fn inherited_field_text_is_found_through_the_parent_chain() {
        let mut m = maps();
        m.field_generic_text
            .insert((1, "_items".into()), "List<Slot>".into());
        let mut stmts = vec![Stmt::Assign(
            "this._items".into(),
            None,
            Expr::NewObj("List<object>".into(), vec![]),
        )];
        assert_eq!(apply(&mut stmts, &m, &HashMap::new(), 302), 1);
        assert!(format!("{stmts:?}").contains("List<Slot>"));
    }

    #[test]
    fn placeholder_local_type_retypes_but_interfaces_are_refused() {
        let mut local_types = HashMap::new();
        local_types.insert("v1".to_string(), 301);
        let mut stmts = vec![Stmt::Assign(
            "v1".into(),
            None,
            Expr::NewObj("Object".into(), vec![]),
        )];
        assert_eq!(apply(&mut stmts, &maps(), &local_types, 302), 1);
        assert!(format!("{stmts:?}").contains("\"Token\""));

        let mut local_types = HashMap::new();
        local_types.insert("v2".to_string(), 700);
        let mut stmts = vec![Stmt::Assign(
            "v2".into(),
            None,
            Expr::NewObj("Object".into(), vec![]),
        )];
        assert_eq!(apply(&mut stmts, &maps(), &local_types, 302), 0);
    }

    #[test]
    fn a_different_generic_definition_is_not_retyped() {
        let mut stmts = vec![Stmt::Assign(
            "this._tokens".into(),
            None,
            Expr::NewObj("List<object>".into(), vec![]),
        )];
        assert_eq!(apply(&mut stmts, &maps(), &HashMap::new(), 302), 0);
    }
}
