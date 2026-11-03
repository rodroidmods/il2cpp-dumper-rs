//! Name raw field-store targets from metadata layouts.
//!
//! The structurer has no field map, so an instance-field store whose base is a
//! typed local comes out as a raw memory write in the assignment TARGET
//! (a string, not an AST expression):
//!
//! ```text
//! *(nint*)(v261 +16) = rewardData;   // Token.Reward
//! *(int*)(v268 +30)  = index;        // RewardData.Value
//! ```
//!
//! Once the base local's class is proven (`local_types`, which the IR fixpoint
//! filled from value flow) and the offset is in that class's layout
//! (`maps.field_layouts`), the target is exactly `base.Field`. The access width
//! is checked against the declared field width where it is knowable, so a
//! wrong base type cannot silently name an unrelated field.
//!
//! This used to be a `// store to Type.+0xNN` comment that DROPPED the store;
//! the comment is gone and the write is preserved.

use super::super::csharp::{Expr, Stmt};
use super::super::resolve::ResolveMaps;
use super::children_mut;
use std::collections::{HashMap, HashSet};

/// `*(int*)(v261 +28)` → (`v261`, 28, 4).
fn parse_raw_target(target: &str) -> Option<(&str, i64, u8)> {
    let rest = target.strip_prefix("*(")?;
    let close = rest.find("*)(")?;
    let ty = &rest[..close];
    let width = match ty {
        "byte" => 1,
        "short" => 2,
        "int" | "float" => 4,
        "nint" | "long" | "ulong" | "double" => 8,
        _ => 0,
    };
    let body = rest[close + 3..].strip_suffix(')')?;
    let bytes = body.as_bytes();
    let mut depth = 0i32;
    let mut split = None;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'+' | b'-' if depth == 0 => split = Some(i),
            _ => {}
        }
    }
    if let Some(p) = split {
        let (base, off) = body.split_at(p);
        let off: i64 = off.trim().parse().ok()?;
        let base = base.trim();
        if !base.is_empty() {
            return Some((base, off, width));
        }
    }
    Some((body.trim(), 0, width))
}

fn is_simple_local(n: &str) -> bool {
    !n.is_empty()
        && !n.contains('.')
        && !n.contains('[')
        && !n.contains('*')
        && !n.contains('&')
        && !n.contains('<')
        && !n.contains('>')
        && !n.contains(' ')
        && !n.chars().next().is_some_and(|c| c.is_ascii_digit())
}

fn lookup_layout<'a>(
    maps: &'a ResolveMaps,
    tdi: i32,
    off: i64,
) -> Option<(&'a str, i32)> {
    let mut chain: Vec<i32> = maps
        .type_parent_chains
        .get(&tdi)
        .cloned()
        .unwrap_or_else(|| vec![tdi]);
    if !chain.contains(&tdi) {
        chain.push(tdi);
    }
    chain
        .iter()
        .find_map(|p| maps.field_layouts.get(&(*p, off)).map(|(name, _)| (name.as_str(), *p)))
}

/// A 4-byte store inside a `Vector2`/`Vector3` field: `this + (field+4)` is
/// `field.y`, `+8` is `field.z`. Metadata has no field start at those offsets
/// (they are struct components), so the exact lookup refuses and the store
/// stayed `*(float*)(this +464) = …` in VerifySpeed @0x1968EF8.
fn component_layout(
    maps: &ResolveMaps,
    tdi: i32,
    off: i64,
) -> Option<(String, i32, &'static str)> {
    for k in 1..=2i64 {
        let base_off = off - k * 4;
        let Some((name, owner)) = lookup_layout(maps, tdi, base_off) else {
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
        return Some((name.to_string(), owner, comp));
    }
    None
}

/// The store's access width must equal the field's DECLARED width. Width 0
/// means "pointer-width / not stated" (`nint`), i.e. 8.
fn width_matches(maps: &ResolveMaps, owner_tdi: i32, off: i64, access: u8) -> bool {
    let access = if access == 0 { 8 } else { access };
    maps.field_widths
        .get(&(owner_tdi, off))
        .is_some_and(|w| *w == access)
}

/// List<T> internals, whose offsets are not in the generic definition's
/// metadata layout: +16 `_items` (pointer), +24 `_size` (Count), +28
/// `_version` (both 4-byte). The same width gates as the expression annotator:
/// a wider access at these offsets is a different object and is refused.
fn collection_internal_name(
    maps: &ResolveMaps,
    tdi: i32,
    off: i64,
    width: u8,
) -> Option<&'static str> {
    let (ns, name) = maps.tdi_names.get(&tdi)?;
    if ns != "System.Collections.Generic" || !(name == "List" || name.starts_with("List`")) {
        return None;
    }
    match (off, width) {
        (16, 0 | 8) => Some("_items"),
        (24, 0 | 4) => Some("Count"),
        (28, 0 | 4) => Some("_version"),
        _ => None,
    }
}

fn base_tdi(
    base: &str,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
    value_tdis: &HashSet<i32>,
) -> Option<i32> {
    let tdi = if base == "this" {
        declaring_tdi
    } else if is_simple_local(base) {
        *local_types.get(base)?
    } else if let Some(field) = base.strip_prefix("this.") {
        if field.contains('.') || field.contains(' ') {
            return None;
        }
        let mut chain: Vec<i32> = maps
            .type_parent_chains
            .get(&declaring_tdi)
            .cloned()
            .unwrap_or_else(|| vec![declaring_tdi]);
        if !chain.contains(&declaring_tdi) {
            chain.push(declaring_tdi);
        }
        chain.iter().find_map(|p| {
            maps.field_type_by_name
                .get(&(*p, field.to_string()))
                .copied()
        })?
    } else {
        return None;
    };
    // A value-type local's `[ptr+off]` is not an instance field access on a
    // reference; only class layouts start at the 16-byte object header.
    if value_tdis.contains(&tdi) {
        return None;
    }
    Some(tdi)
}

fn rewrite_stmt(
    s: &mut Stmt,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
    value_tdis: &HashSet<i32>,
    klass_vars: &HashMap<String, String>,
    n: &mut usize,
) {
    for k in children_mut(s) {
        for x in k.iter_mut() {
            rewrite_stmt(
                x,
                maps,
                local_types,
                declaring_tdi,
                value_tdis,
                klass_vars,
                n,
            );
        }
    }
    if let Stmt::Assign(target, _, _) = s {
        // Rendered chains (`*(int*)(*(nint*)(…klass…)+8)+28`) are rebuilt and
        // resolved before the single-level path; that path cannot see a
        // static-metadata/List chain and left the whole store raw.
        if let Some(expr) = resolve_store_target(
            target,
            maps,
            local_types,
            declaring_tdi,
            value_tdis,
            klass_vars,
        ) {
            if crate::recovery::env_flags::flag("ILR_STOREDBG") {
                eprintln!("[store] {target} → {expr} (chain)");
            }
            *target = expr;
            *n += 1;
            return;
        }
        // A complex element store (`items + (chain << 4) +32`) is not a
        // single field, but the chains inside it still resolve.
        let nested = rewrite_nested_chains(
            target,
            maps,
            local_types,
            declaring_tdi,
            value_tdis,
            klass_vars,
        );
        if nested > 0 {
            *n += nested;
        }
        let Some((base, off, width)) = parse_raw_target(target) else {
            return;
        };
        let base = base.to_string();
        let Some(tdi) = base_tdi(&base, maps, local_types, declaring_tdi, value_tdis) else {
            if crate::recovery::env_flags::flag("ILR_STOREDBG") && is_simple_local(&base) {
                eprintln!("[store-dbg] no type for base {base} in {target}");
            }
            return;
        };
        if let Some(fname) = collection_internal_name(maps, tdi, off, width) {
            if crate::recovery::env_flags::flag("ILR_STOREDBG") {
                eprintln!("[store] {target} → {base}.{fname}");
            }
            *target = format!("{base}.{fname}");
            *n += 1;
            return;
        }
        let Some((fname, owner)) = lookup_layout(maps, tdi, off) else {
            if crate::recovery::env_flags::flag("ILR_STOREDBG") {
                eprintln!("[store-dbg] no layout tdi={tdi} off={off} in {target}");
            }
            return;
        };
        if !width_matches(maps, owner, off, width) {
            if crate::recovery::env_flags::flag("ILR_STOREDBG") {
                eprintln!("[store-dbg] width {width} mismatch field {fname} in {target}");
            }
            return;
        }
        let fname = fname.to_string();
        if crate::recovery::env_flags::flag("ILR_STOREDBG") {
            eprintln!("[store] {target} → {base}.{fname}");
        }
        *target = format!("{base}.{fname}");
        *n += 1;
    }
}

/// `true` for a known generic collection whose metadata definition may carry
/// zero-based field offsets (and would otherwise trip the value-type
/// heuristic). These are reference types.
fn is_known_collection_type(maps: &ResolveMaps, tdi: i32) -> bool {
    maps.tdi_names.get(&tdi).is_some_and(|(ns, name)| {
        ns == "System.Collections.Generic" && (name == "List" || name.starts_with("List`"))
    })
}

/// A partially resolved rendered address chain.
///
/// Store targets are strings produced by the structurer, not expressions, so
/// the pass rebuilds just enough of the expression tree to run the same
/// metadata lookups as `field_annotation`:
///
///   `*(nint*)(*(nint*)(*(nint*)(0xA +K)) +184) +OFF`
///   = klass at `A+K` → `.static_fields` → static field at `OFF`.
///
/// `last_field` marks the value as a real source-level field, which is what
/// makes it a valid store target (a bare `.static_fields` or klass is not).
struct RVal {
    expr: String,
    tdi: Option<i32>,
    owner_type: Option<String>,
    kind: RKind,
    last_field: bool,
    /// The value came from a proven STATIC field, so a value-type layout's
    /// offsets are relative to the struct's own start (not an object header).
    value_ok: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum RKind {
    Address(u64),
    Klass,
    StaticFields,
    Object,
}

fn type_name_of(maps: &ResolveMaps, tdi: i32) -> Option<String> {
    let (ns, name) = maps.tdi_names.get(&tdi)?;
    Some(if ns.is_empty() {
        name.clone()
    } else {
        format!("{ns}.{name}")
    })
}

/// Managed type name a metadata cell/address names, when it names a TYPE.
///
/// `klass_to_tdi` covers TypeDef-bearing slots; `usage_type_by_addr` is the
/// full map field_annotation uses (arrays, primitives, generic instances).
/// The store-target resolver only consulted the first, so a static field
/// written through a slot that exists only in the second stayed raw
/// (`*(nint*)(*(nint*)(*(nint*)(0x2f6a000+1264)) +184) +16 = …`).
fn klass_type_name_at(maps: &ResolveMaps, addr: u64) -> Option<String> {
    if let Some(&tdi) = maps.klass_to_tdi.get(&addr) {
        if let Some(t) = type_name_of(maps, tdi) {
            return Some(t);
        }
    }
    maps.usage_type_by_addr.get(&addr).map(|mt| mt.display_name())
}

/// Locals whose every definition is a class/type token (`v = typeof(T)` or
/// the `klass(T)` accessor, possibly through copies). These are klass
/// POINTERS, so a `+176/184` deref is `.static_fields`, not an instance
/// field. Only names with at least one def and no conflicting def are kept.
fn collect_klass_vars(stmts: &[Stmt]) -> HashMap<String, String> {
    #[derive(Clone)]
    enum Def {
        Known(String),
        Copy(String),
        Other,
    }
    fn walk(stmts: &[Stmt], defs: &mut HashMap<String, Vec<Def>>) {
        for s in stmts {
            if let Stmt::Assign(name, _, rhs) = s {
                if is_simple_local(name) {
                    let d = match rhs {
                        Expr::TypeOf(t) => Def::Known(t.clone()),
                        Expr::StaticField(t, a)
                            if a.starts_with("typeof(") || a.starts_with("klass(") =>
                        {
                            Def::Known(t.clone())
                        }
                        Expr::Var(src) => Def::Copy(src.clone()),
                        _ => Def::Other,
                    };
                    defs.entry(name.clone()).or_default().push(d);
                }
            }
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            for k in kids {
                walk(k, defs);
            }
        }
    }
    let mut defs = HashMap::new();
    walk(stmts, &mut defs);
    let mut out: HashMap<String, String> = HashMap::new();
    for _ in 0..4 {
        let mut changed = false;
        for (name, ds) in &defs {
            if out.contains_key(name) {
                continue;
            }
            let mut ty: Option<String> = None;
            let mut ok = true;
            for d in ds {
                let t = match d {
                    Def::Known(t) => Some(t.clone()),
                    Def::Copy(src) => out.get(src).cloned(),
                    Def::Other => None,
                };
                match (ty.as_ref(), t) {
                    (None, Some(t)) => ty = Some(t),
                    (Some(prev), Some(t)) if *prev == t => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                if let Some(t) = ty {
                    out.insert(name.clone(), t);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    out
}

fn resolve_chain(
    s: &str,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
    value_tdis: &HashSet<i32>,
    klass_vars: &HashMap<String, String>,
    depth: usize,
) -> Option<RVal> {
    let s = s.trim();
    if depth > 16 || s.is_empty() {
        return None;
    }
    if let Some((base, off, width)) = parse_raw_target(s) {
        // `parse_raw_target` also parses plain `x +N`; only treat a real
        // deref as a chain node. A cast string always begins `*(`.
        if s.starts_with("*(") {
            let v = resolve_chain(
                base,
                maps,
                local_types,
                declaring_tdi,
                value_tdis,
                klass_vars,
                depth + 1,
            )?;
            return apply_chain_offset(v, off, width, maps, value_tdis);
        }
    }
    if let Some(addr) = s
        .strip_prefix("0x")
        .and_then(|h| u64::from_str_radix(h, 16).ok())
    {
        return Some(RVal {
            expr: s.to_string(),
            tdi: None,
            owner_type: None,
            kind: RKind::Address(addr),
            last_field: false,
            value_ok: false,
        });
    }
    if let Some(inner) = s.strip_prefix("typeof(").and_then(|x| x.strip_suffix(')')) {
        return Some(RVal {
            expr: s.to_string(),
            tdi: None,
            owner_type: Some(inner.to_string()),
            kind: RKind::Klass,
            last_field: false,
            value_ok: false,
        });
    }
    // A local that always holds a class/type token is a KLASS pointer, not an
    // instance. This is the `v = typeof(T)` / `type3` shape left by static
    // cleanup; `+184` must take the static-fields branch, not an instance
    // field at the same offset.
    if is_simple_local(s) {
        if let Some(ty) = klass_vars.get(s) {
            return Some(RVal {
                expr: s.to_string(),
                tdi: None,
                owner_type: Some(ty.clone()),
                kind: RKind::Klass,
                last_field: false,
            value_ok: false,
            });
        }
    }
    // A simple local / this.field with a proven type.
    if is_simple_local(s) || s.starts_with("this.") {
        if let Some(tdi) = base_tdi(
            s,
            maps,
            local_types,
            declaring_tdi,
            value_tdis,
        ) {
            return Some(RVal {
                expr: s.to_string(),
                tdi: Some(tdi),
                owner_type: None,
                kind: RKind::Object,
                last_field: false,
            value_ok: false,
            });
        }
    }
    // A static field path `Type.Field` (no parentheses/brackets).
    if let Some((ty, field)) = s.rsplit_once('.') {
        if is_simple_local(ty) && is_simple_local(field) {
            if let Some(&tdi) = maps.static_field_types.get(
                &(ty.to_string(), field.to_string()),
            ) {
                return Some(RVal {
                    expr: s.to_string(),
                    tdi: Some(tdi),
                    owner_type: None,
                    kind: RKind::Object,
                    last_field: false,
                    value_ok: true,
                });
            }
        }
    }
    None
}

fn apply_chain_offset(
    v: RVal,
    off: i64,
    width: u8,
    maps: &ResolveMaps,
    value_tdis: &HashSet<i32>,
) -> Option<RVal> {
    // A load at offset 0 of a KLASS value is the redundant klass deref the
    // expression annotator folds to the klass itself. A StaticFields base is
    // the static-fields POINTER: a load at offset 0 there is the static field
    // at slot 0 and must be looked up below.
    if off == 0 && matches!(v.kind, RKind::Klass) {
        return Some(v);
    }
    match v.kind {
        RKind::Address(addr) => {
            let full = addr.wrapping_add(off as u64);
            if let Some((name, ty)) = maps.static_field_layouts.get(&full) {
                if name.starts_with("typeof(") || name.starts_with("klass(") {
                    return Some(RVal {
                        expr: format!("typeof({ty})"),
                        tdi: None,
                        owner_type: Some(ty.clone()),
                        kind: RKind::Klass,
                        last_field: false,
            value_ok: false,
                    });
                }
                if !is_simple_local(name) {
                    return None;
                }
                let tdi = maps
                    .static_field_types
                    .get(&(ty.clone(), name.clone()))
                    .copied();
                return Some(RVal {
                    expr: format!("{ty}.{name}"),
                    tdi,
                    owner_type: None,
                    kind: RKind::Object,
                    last_field: true,
                    value_ok: true,
                });
            }
            if let Some(ty) = klass_type_name_at(maps, full) {
                return Some(RVal {
                    expr: format!("typeof({ty})"),
                    tdi: None,
                    owner_type: Some(ty),
                    kind: RKind::Klass,
                    last_field: false,
                    value_ok: false,
                });
            }
            // A metadata-usage slot is often reached through the GOT: the slot
            // holds the addend (the runtime klass/field VA), which the maps
            // key directly. `field_annotation` already takes this path.
            if let Some(&addend) = maps.got_addend_by_addr.get(&full) {
                if let Some(ty) = klass_type_name_at(maps, addend) {
                    return Some(RVal {
                        expr: format!("typeof({ty})"),
                        tdi: None,
                        owner_type: Some(ty),
                        kind: RKind::Klass,
                        last_field: false,
                        value_ok: false,
                    });
                }
                if let Some((name, ty)) = maps.static_field_layouts.get(&addend) {
                    if !name.starts_with("typeof(") && !name.starts_with("klass(") {
                        if !is_simple_local(name) {
                            return None;
                        }
                        let tdi = maps
                            .static_field_types
                            .get(&(ty.clone(), name.clone()))
                            .copied();
                        return Some(RVal {
                            expr: format!("{ty}.{name}"),
                            tdi,
                            owner_type: None,
                            kind: RKind::Object,
                            last_field: true,
                    value_ok: true,
                        });
                    }
                }
            }
            None
        }
        RKind::Klass => {
            let ty = v.owner_type.clone()?;
            // Il2CppClass.static_fields (v24: +184, v27+: +176).
            if off == 176 || off == 184 {
                return Some(RVal {
                    expr: format!("{ty}.static_fields"),
                    tdi: None,
                    owner_type: Some(ty),
                    kind: RKind::StaticFields,
                    last_field: false,
            value_ok: false,
                });
            }
            // The klass token itself can carry a static-field offset:
            // `typeof(NGUIText) +56` addresses NGUIText's static slot 56.
            if let Some((name, owner)) = maps.static_field_offsets.get(&(ty, off)) {
                if !is_simple_local(name) {
                    return None;
                }
                let tdi = maps
                    .static_field_types
                    .get(&(owner.clone(), name.clone()))
                    .copied();
                return Some(RVal {
                    expr: format!("{owner}.{name}"),
                    tdi,
                    owner_type: None,
                    kind: RKind::Object,
                    last_field: true,
                    value_ok: true,
                });
            }
            None
        }
        RKind::StaticFields => {
            let ty = v.owner_type?;
            let (name, owner) = maps.static_field_offsets.get(&(ty, off))?;
            if !is_simple_local(name) {
                return None;
            }
            let tdi = maps
                .static_field_types
                .get(&(owner.clone(), name.clone()))
                .copied();
            Some(RVal {
                expr: format!("{owner}.{name}"),
                tdi,
                owner_type: None,
                kind: RKind::Object,
                last_field: true,
                value_ok: true,
            })
        }
        RKind::Object => {
            // Offset 0 of a managed object is its klass pointer, not a field.
            if off == 0 {
                return None;
            }
            let tdi = v.tdi?;
            if let Some(fname) = collection_internal_name(maps, tdi, off, width) {
                return Some(RVal {
                    expr: format!("{}.{fname}", v.expr),
                    tdi: None,
                    owner_type: None,
                    kind: RKind::Object,
                    last_field: true,
                    value_ok: v.value_ok,
                });
            }
            if value_tdis.contains(&tdi) && !v.value_ok {
                return None;
            }
            let exact = lookup_layout(maps, tdi, off)
                .filter(|(_, owner)| width_matches(maps, *owner, off, width));
            let (fname, field_tdi) = match exact {
                Some((f, owner)) => (
                    f.to_string(),
                    maps.field_type_by_name
                        .get(&(owner, f.to_string()))
                        .copied(),
                ),
                None if width == 4 => match component_layout(maps, tdi, off) {
                    Some((base, _owner, comp)) => (format!("{base}.{comp}"), None),
                    None => return None,
                },
                None => return None,
            };
            Some(RVal {
                expr: format!("{}.{fname}", v.expr),
                tdi: field_tdi,
                owner_type: None,
                kind: RKind::Object,
                last_field: true,
                value_ok: v.value_ok,
            })
        }
    }
}

/// Matching `)` for a `(` at `open`, ignoring string literals.
fn matching_paren(s: &str, open: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() {
                    if b[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if b[i] == b'"' {
                        break;
                    }
                    i += 1;
                }
            }
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Find every rendered `*(T*)(…)` sub-expression (start..=end), innermost
/// first. Store targets can contain chains inside an index/base expression
/// (`items + (*(int*)(…_version… ) << 4) +32`), which the whole-target
/// resolver cannot see.
fn deref_candidates(s: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i + 2 < bytes.len() {
        if bytes[i] == b'*' && bytes[i + 1] == b'(' {
            if let Some(cast_close) = s[i..].find("*)(") {
                let cast_close = i + cast_close;
                let body_open = cast_close + 2;
                if let Some(end) = matching_paren(s, body_open) {
                    out.push((i, end));
                    // Keep scanning INSIDE this deref: nested chains are the
                    // ones that resolve first (`*(nint*)(items + (chain<<4))`).
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }
    out.sort_by_key(|(a, b)| b - a);
    out
}

/// Rewrite resolvable nested chains inside a larger store target. Returns the
/// number of replaced sub-expressions. Bounded; the outer array store may stay
/// raw, but every metadata/List link inside it becomes a field.
fn rewrite_nested_chains(
    target: &mut String,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
    value_tdis: &HashSet<i32>,
    klass_vars: &HashMap<String, String>,
) -> usize {
    let mut n = 0usize;
    for _ in 0..32 {
        let mut replaced = false;
        for (start, end) in deref_candidates(target) {
            let candidate = target[start..=end].to_string();
            if let Some(expr) = resolve_store_target(
                &candidate,
                maps,
                local_types,
                declaring_tdi,
                value_tdis,
                klass_vars,
            ) {
                if crate::recovery::env_flags::flag("ILR_STOREDBG") {
                    eprintln!("[store] {candidate} → {expr} (nested)");
                }
                target.replace_range(start..=end, &expr);
                n += 1;
                replaced = true;
                break;
            }
        }
        if !replaced {
            break;
        }
    }
    n
}

/// Resolve a fully rendered store target. Returns the named lvalue.
fn resolve_store_target(
    target: &str,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
    value_tdis: &HashSet<i32>,
    klass_vars: &HashMap<String, String>,
) -> Option<String> {
    let dbg = crate::recovery::env_flags::flag("ILR_STOREDBG");
    let (base, off, width) = parse_raw_target(target)?;
    if !target.starts_with("*(") {
        return None;
    }
    let Some(v) = resolve_chain(
        base,
        maps,
        local_types,
        declaring_tdi,
        value_tdis,
        klass_vars,
        0,
    ) else {
        if dbg {
            eprintln!("[chain-dbg] unresolved base={base} target={target}");
        }
        return None;
    };
    let Some(v) = apply_chain_offset(v, off, width, maps, value_tdis) else {
        if dbg {
            eprintln!("[chain-dbg] unresolved offset={off} width={width} base={base}");
        }
        return None;
    };
    if dbg {
        eprintln!("[chain-dbg] candidate={} target={target}", v.expr);
    }
    if v.last_field {
        Some(v.expr)
    } else {
        None
    }
}
pub fn apply(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
) -> usize {
    let value_tdis: HashSet<i32> = maps
        .field_layouts
        .keys()
        .filter(|(_, off)| (0..16).contains(off))
        .map(|(tdi, _)| *tdi)
        .filter(|tdi| !is_known_collection_type(maps, *tdi))
        .collect();
    let mut n = 0;
    let klass_vars = collect_klass_vars(stmts);
    for s in stmts.iter_mut() {
        rewrite_stmt(
            s,
            maps,
            local_types,
            declaring_tdi,
            &value_tdis,
            &klass_vars,
            &mut n,
        );
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps() -> ResolveMaps {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(301, ("".into(), "Token".into()));
        m.tdi_names.insert(531, ("".into(), "RewardData".into()));
        m.tdi_names.insert(540, ("".into(), "Int32".into()));
        m.type_parent_chains.insert(301, vec![301, 1]);
        m.field_layouts
            .insert((301, 16), ("Reward".into(), "Token".into()));
        m.field_types.insert((301, 16), 531);
        m.field_widths.insert((301, 16), 8);
        m.field_layouts
            .insert((301, 32), ("TotalCost".into(), "Token".into()));
        m.field_types.insert((301, 32), 540);
        m.field_widths.insert((301, 32), 4);
        m
    }

    #[test]
    fn reference_store_is_named() {
        let mut lt = HashMap::new();
        lt.insert("v1".to_string(), 301);
        let mut stmts = vec![Stmt::Assign(
            "*(nint*)(v1 +16)".into(),
            None,
            Expr::Var("reward".into()),
        )];
        assert_eq!(apply(&mut stmts, &maps(), &lt, 302), 1);
        assert_eq!(target(&stmts), "v1.Reward");
    }

    #[test]
    fn int_store_is_named_and_width_checked() {
        let mut lt = HashMap::new();
        lt.insert("v1".to_string(), 301);
        let mut ok = vec![Stmt::Assign(
            "*(int*)(v1 +32)".into(),
            None,
            Expr::Int(3),
        )];
        assert_eq!(apply(&mut ok, &maps(), &lt, 302), 1);
        assert_eq!(target(&ok), "v1.TotalCost");

        // A one-byte store at the int field is not that field's write.
        let mut bad = vec![Stmt::Assign(
            "*(byte*)(v1 +32)".into(),
            None,
            Expr::Int(1),
        )];
        assert_eq!(apply(&mut bad, &maps(), &lt, 302), 0);
    }

    #[test]
    fn value_type_local_is_refused() {
        let mut m = maps();
        m.field_layouts
            .insert((600, 8), ("x".into(), "Vec3".into()));
        m.field_types.insert((600, 8), 540);
        m.field_widths.insert((600, 8), 4);
        let mut lt = HashMap::new();
        lt.insert("v9".to_string(), 600);
        let mut stmts = vec![Stmt::Assign(
            "*(int*)(v9 +8)".into(),
            None,
            Expr::Int(1),
        )];
        assert_eq!(apply(&mut stmts, &m, &lt, 302), 0);
    }

    #[test]
    fn list_internal_stores_are_named_with_widths() {
        let mut m = maps();
        m.tdi_names.insert(
            30,
            ("System.Collections.Generic".into(), "List`1".into()),
        );
        let mut lt = HashMap::new();
        lt.insert("list".to_string(), 30);
        let mut stmts = vec![
            Stmt::Assign("*(int*)(list +24)".into(), None, Expr::Int(1)),
            Stmt::Assign("*(int*)(list +28)".into(), None, Expr::Int(2)),
            Stmt::Assign("*(nint*)(list +16)".into(), None, Expr::Null),
            // A 64-bit write at Count is not Count.
            Stmt::Assign("*(nint*)(list +24)".into(), None, Expr::Int(0)),
        ];
        assert_eq!(apply(&mut stmts, &m, &lt, 302), 3);
        let targets: Vec<&str> = stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::Assign(t, _, _) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            targets,
            vec!["list.Count", "list._version", "list._items", "*(nint*)(list +24)"]
        );
    }

    #[test]
    fn a_static_value_type_field_keeps_resolving_inner_fields() {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(600, ("".into(), "Matrix".into()));
        m.static_field_offsets
            .insert(("T".into(), 0), ("State".into(), "T".into()));
        m.static_field_types
            .insert(("T".into(), "State".into()), 600);
        // A field at offset 0 makes the heuristic classify 600 as a value
        // type; a static value-type base must still resolve inner offsets.
        m.field_layouts
            .insert((600, 0), ("m00".into(), "Matrix".into()));
        m.field_widths.insert((600, 0), 4);
        m.field_layouts
            .insert((600, 32), ("m20".into(), "Matrix".into()));
        m.field_widths.insert((600, 32), 4);
        let mut stmts = vec![
            Stmt::Assign("k".into(), None, Expr::TypeOf("T".into())),
            Stmt::Assign(
                "*(int*)(*(nint*)(*(nint*)(k +184)) +32)".into(),
                None,
                Expr::Int(1),
            ),
        ];
        assert_eq!(apply(&mut stmts, &m, &HashMap::new(), 0), 1);
        assert_eq!(target(&stmts[1..]), "T.State.m20");
    }

    #[test]
    fn a_klass_token_local_chain_store_is_named() {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(5, ("".into(), "T".into()));
        m.static_field_offsets
            .insert(("T".into(), 0), ("Instance".into(), "T".into()));
        m.static_field_types
            .insert(("T".into(), "Instance".into()), 5);
        m.field_layouts
            .insert((5, 32), ("Value".into(), "T".into()));
        m.field_widths.insert((5, 32), 4);

        let mut stmts = vec![
            Stmt::Assign("k".into(), None, Expr::TypeOf("T".into())),
            Stmt::Assign(
                "*(int*)(*(nint*)(*(nint*)(k +184)) +32)".into(),
                None,
                Expr::Int(1),
            ),
        ];
        assert_eq!(apply(&mut stmts, &m, &HashMap::new(), 0), 1);
        assert_eq!(target(&stmts[1..]), "T.Instance.Value");
    }

    #[test]
    fn a_static_singleton_chain_store_is_named() {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(5, ("".into(), "TargetType".into()));
        m.static_field_layouts.insert(
            0x1000,
            ("klass(OwnerClass)".into(), "OwnerClass".into()),
        );
        m.static_field_offsets.insert(
            ("OwnerClass".into(), 8),
            ("Instance".into(), "OwnerClass".into()),
        );
        m.static_field_types
            .insert(("OwnerClass".into(), "Instance".into()), 5);
        m.field_layouts
            .insert((5, 216), ("Target".into(), "TargetType".into()));
        m.field_widths.insert((5, 216), 8);

        let mut stmts = vec![Stmt::Assign(
            "*(nint*)(*(nint*)(*(nint*)(*(nint*)(0x1000) +184) +8) +216)".into(),
            None,
            Expr::Int(1),
        )];
        assert_eq!(apply(&mut stmts, &m, &HashMap::new(), 0), 1);
        assert_eq!(target(&stmts), "OwnerClass.Instance.Target");
    }

    #[test]
    fn a_static_list_internal_store_is_named_through_the_klass_chain() {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(
            30,
            ("System.Collections.Generic".into(), "List`1".into()),
        );
        m.static_field_layouts.insert(
            0x2000,
            ("klass(Holder)".into(), "Holder".into()),
        );
        m.static_field_offsets
            .insert(("Holder".into(), 8), ("Items".into(), "Holder".into()));
        m.static_field_types
            .insert(("Holder".into(), "Items".into()), 30);

        let mut stmts = vec![Stmt::Assign(
            "*(int*)(*(nint*)(*(nint*)(*(nint*)(0x2000) +184) +8) +24)".into(),
            None,
            Expr::Int(1),
        )];
        assert_eq!(apply(&mut stmts, &m, &HashMap::new(), 0), 1);
        assert_eq!(target(&stmts), "Holder.Items.Count");
    }

    #[test]
    fn chains_inside_an_element_index_are_named() {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(
            30,
            ("System.Collections.Generic".into(), "List`1".into()),
        );
        m.static_field_layouts.insert(
            0x1000,
            ("klass(Holder)".into(), "Holder".into()),
        );
        m.static_field_offsets.insert(
            ("Holder".into(), 8),
            ("Items".into(), "Holder".into()),
        );
        m.static_field_types
            .insert(("Holder".into(), "Items".into()), 30);

        let mut stmts = vec![Stmt::Assign(
            "*(nint*)(items + (*(int*)(*(nint*)(*(nint*)(*(nint*)(0x1000) +184) +8) +24) << 4) +32)"
                .into(),
            None,
            Expr::Int(0),
        )];
        assert_eq!(apply(&mut stmts, &m, &HashMap::new(), 0), 2);
        assert_eq!(
            target(&stmts),
            "*(nint*)(items + (Holder.Items.Count << 4) +32)"
        );
    }

    fn target(stmts: &[Stmt]) -> &str {
        match &stmts[0] {
            Stmt::Assign(t, _, _) => t,
            _ => "",
        }
    }
}
