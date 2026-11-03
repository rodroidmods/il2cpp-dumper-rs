//! Recover closure (display-class) field accesses from raw offsets.
//!
//! A lambda captures its environment in a compiler-generated display class:
//!
//! ```text
//! var v351 = new <>c__DisplayClass23_0();
//! *(nint*)(v351 +16) = this;
//! *(nint*)(v351 +32) = GameProfile.GetActiveUserData();
//! … (*(nint*)(v351 +32)).IsArtifactPurchased(…)
//! ```
//!
//! The construction names the TypeDef, but the method's name-level local type
//! map can prove a different type for the reused slot (here an array type, so
//! `+0x18` was even sugared to `.Length`), and the generic field annotator
//! never sees the display class's layout. The layout IS in the metadata:
//! `<>4__this` at 0x10, then the captured locals in declaration order.
//!
//! This pass resolves the display class's TypeDef from the lambda bodies it
//! declares (`<EnclosingMethod>b__N` in the method table — exact even when
//! several display classes share a name), then rewrites the raw accesses whose
//! base is the display local, or a field of it whose type the layout proves.
//!
//! ## Safety
//!
//! * The local must have exactly ONE assignment in the whole method, and it
//!   must be the `new` of a `<>…__DisplayClass…` type: no other value can flow
//!   into the name, so every read observes that object.
//! * The TypeDef must be unambiguous (one lambda owner, or one TypeDef name).
//! * `field_at` requires an exact offset and, when the declared width is
//!   known, an exact width, so a different-width access is never renamed.
//! * The raw access is rewritten only to a field that exists at that offset;
//!   anything else stays raw.

use std::collections::HashMap;

use crate::recovery::csharp::{Expr, Stmt};
use crate::recovery::resolve::ResolveMaps;

pub(crate) fn apply(stmts: &mut Vec<Stmt>, maps: &ResolveMaps, method_name: &str) -> usize {
    let mut assigns: HashMap<String, usize> = HashMap::new();
    let mut classes: HashMap<String, String> = HashMap::new();
    collect_candidates(stmts, &mut assigns, &mut classes);
    if classes.is_empty() {
        return 0;
    }
    let mut resolved: HashMap<String, i32> = HashMap::new();
    for (name, cls) in &classes {
        if assigns.get(name).copied() != Some(1) {
            continue;
        }
        if let Some(tdi) = resolve_tdi(maps, cls, method_name) {
            if maps.tdi_fields.get(&tdi).is_some_and(|f| !f.is_empty()) {
                resolved.insert(name.clone(), tdi);
            }
        }
    }
    if resolved.is_empty() {
        return 0;
    }
    let mut n = 0;
    super::walk_mut(stmts, &mut |list| {
        for s in list.iter_mut() {
            if let Stmt::Assign(target, _, _) = s {
                n += rewrite_target(target, &resolved, maps);
            }
            super::copy_prop::for_each_expr_mut(s, &mut |e| {
                n += rewrite_expr(e, &resolved, maps);
            });
        }
    });
    n
}

fn is_plain_local(n: &str) -> bool {
    !n.is_empty()
        && n.chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !matches!(
            n,
            "this" | "base" | "null" | "true" | "false" | "new" | "default"
        )
}

fn collect_candidates(
    stmts: &[Stmt],
    assigns: &mut HashMap<String, usize>,
    classes: &mut HashMap<String, String>,
) {
    for s in stmts {
        if let Stmt::Assign(t, _, e) = s {
            if is_plain_local(t) {
                *assigns.entry(t.clone()).or_insert(0) += 1;
                if let Expr::NewObj(cls, _) = e {
                    if cls.starts_with("<>") && cls.contains("__DisplayClass") {
                        classes.insert(t.clone(), cls.clone());
                    }
                }
            }
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_candidates(k, assigns, classes);
        }
        if let Stmt::Using { decl, .. } = s {
            collect_candidates(std::slice::from_ref(decl.as_ref()), assigns, classes);
        }
    }
}

/// The display class's TypeDef: prefer the lambda bodies it declares for THIS
/// method (`<Name>b__N`), which pin the exact nested type; fall back to a
/// globally unique TypeDef name.
fn resolve_tdi(maps: &ResolveMaps, cls: &str, method_name: &str) -> Option<i32> {
    let want = format!("<{method_name}>b__");
    let mut found: Option<i32> = None;
    for m in maps.methods_by_rva.values() {
        if m.declaring_type_name == cls && m.name.starts_with(&want) {
            match found {
                None => found = Some(m.type_def_index),
                Some(t) if t == m.type_def_index => {}
                Some(_) => return None,
            }
        }
    }
    if found.is_some() {
        return found;
    }
    let mut tdi = None;
    for (&t, (_, name)) in &maps.tdi_names {
        if name == cls {
            match tdi {
                None => tdi = Some(t),
                Some(prev) if prev == t => {}
                Some(_) => return None,
            }
        }
    }
    tdi
}

fn field_at(maps: &ResolveMaps, tdi: i32, off: i64, width: u8) -> Option<(String, i32)> {
    let fields = maps.tdi_fields.get(&tdi)?;
    let (o, name, ty) = fields.iter().find(|(o, _, _)| *o == off)?;
    if let Some(&decl) = maps.field_widths.get(&(tdi, *o)) {
        if width != 0 && decl != width {
            return None;
        }
    }
    Some((name.clone(), *ty))
}

fn class_has_field(maps: &ResolveMaps, tdi: i32, name: &str) -> bool {
    maps.tdi_fields
        .get(&tdi)
        .is_some_and(|f| f.iter().any(|(_, n, _)| n == name))
}

/// Type of a base expression whose layout the display class proves: the local
/// itself, or a field reached through it.
fn tdi_of(e: &Expr, resolved: &HashMap<String, i32>, maps: &ResolveMaps) -> Option<i32> {
    match e {
        Expr::Var(n) => resolved.get(n).copied(),
        Expr::Field(b, name) => {
            let bt = tdi_of(b, resolved, maps)?;
            maps.tdi_fields
                .get(&bt)?
                .iter()
                .find(|(_, fname, _)| fname == name)
                .map(|(_, _, t)| *t)
        }
        _ => None,
    }
}

fn rewrite_target(
    target: &mut String,
    resolved: &HashMap<String, i32>,
    maps: &ResolveMaps,
) -> usize {
    let Some(rest) = target.strip_prefix("*(nint*)(") else {
        return 0;
    };
    let Some(inner) = rest.strip_suffix(')') else {
        return 0;
    };
    let Some((base, off_s)) = inner.rsplit_once(" +") else {
        return 0;
    };
    let base = base.trim();
    if !is_plain_local(base) {
        return 0;
    }
    let Ok(off) = off_s.trim().parse::<i64>() else {
        return 0;
    };
    let Some(&tdi) = resolved.get(base) else {
        return 0;
    };
    let Some((name, _)) = field_at(maps, tdi, off, 8) else {
        return 0;
    };
    *target = format!("{base}.{name}");
    1
}

fn rewrite_expr(e: &mut Expr, resolved: &HashMap<String, i32>, maps: &ResolveMaps) -> usize {
    let mut n = 0;
    for c in super::il2cpp_cleanup::expr_children_mut(e) {
        n += rewrite_expr(c, resolved, maps);
    }
    // A `.Length`/`.Count` on a proven display local is a misfired array/List
    // sugar from the +0x18 slot: the class has no such member, so the real
    // access is the field at 0x18.
    if let Expr::Field(base, name) = e {
        if name == "Length" || name == "Count" {
            if let Expr::Var(local) = base.as_ref() {
                if let Some(&tdi) = resolved.get(local) {
                    if !class_has_field(maps, tdi, name) {
                        if let Some((fname, _)) = field_at(maps, tdi, 0x18, 8) {
                            *name = fname;
                            n += 1;
                        }
                    }
                }
            }
        }
    }
    if matches!(e, Expr::RawDeref(..)) {
        let old = std::mem::replace(e, Expr::Unknown);
        if let Expr::RawDeref(base, off, width) = old {
            if let Some(tdi) = tdi_of(&base, resolved, maps) {
                if let Some((fname, _)) = field_at(maps, tdi, off, width) {
                    *e = Expr::Field(base, fname);
                    return n + 1;
                }
            }
            *e = Expr::RawDeref(base, off, width);
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps_with_display() -> ResolveMaps {
        let mut maps = ResolveMaps::default();
        maps.tdi_fields.insert(
            1228,
            vec![
                (0x10, "<>4__this".to_string(), 999),
                (0x18, "cellData".to_string(), 2194),
                (0x20, "playerStats".to_string(), 516),
            ],
        );
        for off in [0x10, 0x18, 0x20] {
            maps.field_widths.insert((1228, off), 8);
        }
        maps.tdi_names
            .insert(1228, (String::new(), "<>c__DisplayClass23_0".to_string()));
        maps
    }

    fn new_display(local: &str) -> Stmt {
        Stmt::Assign(
            local.into(),
            Some("var".into()),
            Expr::NewObj("<>c__DisplayClass23_0".into(), vec![]),
        )
    }

    #[test]
    fn a_display_class_raw_store_becomes_a_named_field() {
        let maps = maps_with_display();
        let mut s = vec![
            new_display("v351"),
            Stmt::Assign("*(nint*)(v351 +32)".into(), None, Expr::Int(0)),
        ];
        assert_eq!(apply(&mut s, &maps, "M"), 1);
        assert!(matches!(&s[1], Stmt::Assign(t, _, _) if t == "v351.playerStats"));
    }

    #[test]
    fn a_display_class_raw_load_becomes_a_named_field() {
        let maps = maps_with_display();
        let mut s = vec![
            new_display("v351"),
            Stmt::Eval(Expr::RawDeref(
                Box::new(Expr::Var("v351".into())),
                0x18,
                8,
            )),
        ];
        assert_eq!(apply(&mut s, &maps, "M"), 1);
        assert!(matches!(
            &s[1],
            Stmt::Eval(Expr::Field(base, name)) if name == "cellData"
        ));
    }

    #[test]
    fn a_reassigned_local_is_not_typed() {
        let maps = maps_with_display();
        let mut s = vec![
            new_display("v351"),
            Stmt::Assign("v351".into(), None, Expr::Null),
            Stmt::Assign("*(nint*)(v351 +32)".into(), None, Expr::Int(0)),
        ];
        assert_eq!(apply(&mut s, &maps, "M"), 0);
    }

    #[test]
    fn an_unknown_offset_stays_raw() {
        let maps = maps_with_display();
        let mut s = vec![
            new_display("v351"),
            Stmt::Eval(Expr::RawDeref(
                Box::new(Expr::Var("v351".into())),
                0x40,
                8,
            )),
        ];
        assert_eq!(apply(&mut s, &maps, "M"), 0);
    }
}
