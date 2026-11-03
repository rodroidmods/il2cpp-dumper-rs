//! Rewrite raw array-element stores to source-level indexing.
//!
//! After array-initializer folding, the remaining element stores are still
//! rendered as raw pointer writes:
//!
//! ```text
//! *(nint*)(buttonStrings + (i << 3) +32) = value;
//! ```
//!
//! A local declared as `object[]`/`T[]` has a known element size, so this is
//! `buttonStrings[i] = value` (or a constant index for `+32+8k`). Only
//! stores whose base is a declared array local, whose width matches the
//! element size, and whose offset starts at the 32-byte array data header
//! are rewritten.

use super::super::csharp::{BinOp, Expr, Stmt};
use super::super::resolve::ResolveMaps;
use super::children_mut;
use std::collections::HashMap;

fn elem_size_of_name(base: &str, sizes: &HashMap<String, u64>) -> Option<u64> {
    let base = base.trim();
    Some(match base {
        "bool" | "byte" | "sbyte" => 1,
        "short" | "ushort" | "char" => 2,
        "int" | "uint" | "float" => 4,
        "long" | "ulong" | "double" | "nint" | "nuint" => 8,
        // Only KNOWN reference arrays may be treated as pointer-sized. Any
        // other name is looked up in the metadata size map (references are
        // pointer-sized, value types carry their instance_size); an unknown
        // name stays `None` and only stride-proven accesses are rewritten.
        "object" | "string" | "Type" | "System.Object" | "System.String" | "System.Type" => 8,
        _ => return sizes.get(base).copied(),
    })
}

fn elem_size_of_array_type(ty: &str, sizes: &HashMap<String, u64>) -> Option<u64> {
    elem_size_of_name(ty.strip_suffix("[]")?, sizes)
}

/// What is known about one array local.
#[derive(Default, Clone)]
struct ArrayInfo {
    /// Element byte size; `None` = known array, size unproved.
    size: Option<u64>,
    /// Element type display name, for `(array.klass).element_class`.
    elem: Option<String>,
}

impl ArrayInfo {
    fn merge(&mut self, other: ArrayInfo) {
        self.size = self.size.or(other.size);
        self.elem = self.elem.take().or(other.elem);
    }
}

/// Collect array locals and their element size.
///
/// Three sources, because the element type is proved in three places:
///  * a declared `T[]` local (params, typed assignments);
///  * the `new T[n]` the allocation-helper fold produced;
///  * a folded `ArrayInit`.
/// `None` means "known array, element size unproved"; constant-index stores
/// to those are refused, but strided accesses carry their own stride.
fn collect_arrays(
    stmts: &[Stmt],
    out: &mut HashMap<String, ArrayInfo>,
    sizes: &HashMap<String, u64>,
) {
    fn note(out: &mut HashMap<String, ArrayInfo>, name: &str, info: ArrayInfo) {
        out.entry(name.to_string())
            .or_default()
            .merge(info);
    }
    for s in stmts {
        match s {
            Stmt::Assign(name, Some(ty), _) if ty.contains("[]") => {
                let elem = ty.trim_end_matches("[]").trim();
                note(
                    out,
                    name,
                    ArrayInfo {
                        size: elem_size_of_array_type(ty, sizes),
                        elem: (!elem.is_empty()).then(|| elem.to_string()),
                    },
                );
            }
            Stmt::Assign(name, _, Expr::NewArr(elem, _)) => {
                note(
                    out,
                    name,
                    ArrayInfo {
                        size: elem_size_of_name(elem, sizes),
                        elem: Some(elem.clone()),
                    },
                );
            }
            Stmt::Assign(name, _, Expr::ArrayInit { elem_type, .. }) => {
                note(
                    out,
                    name,
                    ArrayInfo {
                        size: elem_size_of_name(elem_type, sizes),
                        elem: Some(elem_type.clone()),
                    },
                );
            }
            // An allocation call carries the array class as its FIRST
            // argument (`il2cpp_array_new_specific(typeof(string[]), len)`).
            // The allocator is often not named in the symbol table, but the
            // type token alone proves what the result is: a call handed the
            // T[] class and returning an object produces a T[].
            Stmt::Assign(name, _, Expr::Call { args, .. })
            | Stmt::Assign(name, _, Expr::RawCall(_, args, _)) => {
                if let Some(Expr::TypeOf(t)) = args.first() {
                    if let Some(elem) = t.strip_suffix("[]") {
                        if !elem.is_empty() && !elem.contains('[') {
                            note(
                                out,
                                name,
                                ArrayInfo {
                                    size: elem_size_of_name(elem, sizes),
                                    elem: Some(elem.to_string()),
                                },
                            );
                        }
                    }
                }
            }
            _ => {}
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_arrays(k, out, sizes);
        }
    }
}

fn cast_width(t: &str) -> Option<u8> {
    let rest = t.strip_prefix("*(")?;
    let close = rest.find("*)")?;
    let ty = &rest[..close];
    Some(match ty {
        "byte" | "sbyte" | "bool" => 1,
        "short" | "ushort" | "char" => 2,
        "int" | "uint" | "float" => 4,
        "nint" | "nuint" | "long" | "ulong" | "double" => 8,
        _ => 8,
    })
}

/// Split a rendered address body at top-level `+`/`-` signs.
fn split_terms(body: &str) -> Vec<&str> {
    let b = body.as_bytes();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut out = Vec::new();
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'+' | b'-' if depth == 0 && i > 0 => {
                out.push(body[start..i].trim());
                start = i;
            }
            _ => {}
        }
    }
    out.push(body[start..].trim());
    out
}

/// Parse `(<expr> << SHIFT)` / `<expr> << SHIFT` / `(<expr> * SIZE)`.
///
/// With a proved element size the stride must equal it. With an unproved
/// size (a named struct whose metadata size is unavailable) the stride in
/// the ACCESS is the proof: the machine computed element addresses at that
/// stride, so the index is the same either way.
fn index_from_term(term: &str, elem: Option<u64>) -> Option<String> {
    let t = term.trim().trim_start_matches('+').trim();
    let t = t.strip_prefix('(').and_then(|x| x.strip_suffix(')')).unwrap_or(t);
    for op in ["<<", "*"] {
        let Some(pos) = t.rfind(op) else { continue };
        let lhs = t[..pos].trim();
        let rhs = t[pos + op.len()..].trim();
        let n: u64 = rhs.parse().ok()?;
        let matches = match (op, elem) {
            ("<<", Some(e)) => n < 64 && (1u64 << n) == e,
            ("<<", None) => n <= 6,
            ("*", Some(e)) => n == e,
            ("*", None) => (1..=64).contains(&n),
            _ => false,
        };
        if !matches || lhs.is_empty() {
            return None;
        }
        return Some(lhs.to_string());
    }
    None
}

fn rewrite_target(target: &str, arrays: &HashMap<String, ArrayInfo>) -> Option<String> {
    let width = cast_width(target)?;
    // The OUTER cast is the first `*)(`; the body can contain nested casts
    // (`*(nint*)(*(nint*)(arr +16) +16)`) which `rfind` would pick instead.
    let close = target.find("*)(")?;
    let body_start = close + 3;
    let body = target[body_start..].strip_suffix(')')?.trim();

    let (var, info) = arrays
        .iter()
        .filter(|(k, _)| body == k.as_str() || body.starts_with(&format!("{k} ")) || body.starts_with(&format!("{k}+")) || body.starts_with(&format!("{k} +")))
        .max_by_key(|(k, _)| k.len())?;
    let elem = info.size;
    // A store narrower than the element would write a FIELD of the element,
    // not the element; only full-width element stores are sugarable.
    match elem {
        Some(e) if width as u64 == e => {}
        None if width == 8 => {}
        _ => return None,
    }

    let rest = body[var.len()..].trim();
    if rest.is_empty() {
        return None;
    }
    let body_after_var = rest.trim_start_matches('+').trim();
    let with_plus = format!("+{body_after_var}");
    let terms = split_terms(&with_plus);
    let mut const_sum: i64 = 0;
    let mut index: Option<String> = None;
    for term in terms {
        let t = term.trim();
        if t.is_empty() || t == "+" || t == "-" {
            continue;
        }
        if let Some(v) = t.strip_prefix('+').map(str::trim) {
            if let Ok(n) = v.parse::<i64>() {
                const_sum += n;
                continue;
            }
            if let Some(idx) = index_from_term(v, elem) {
                if index.is_some() {
                    return None;
                }
                index = Some(idx);
                continue;
            }
            // A bare variable/index expression (no stride) is not an array
            // element address.
            return None;
        }
        if let Some(v) = t.strip_prefix('-').map(str::trim) {
            if let Ok(n) = v.parse::<i64>() {
                const_sum -= n;
                continue;
            }
            return None;
        }
        return None;
    }
    const_sum -= 32;
    let idx = match index {
        Some(idx) => {
            // `array + (i << 3) + 40` is element `i + 1` (data begins at +32).
            let k = match elem {
                Some(e) => {
                    if const_sum < 0 || (const_sum as u64) % e != 0 {
                        return None;
                    }
                    const_sum as u64 / e
                }
                None if const_sum == 0 => 0,
                // Without the element size an offset beyond the header
                // cannot be attributed to an element index.
                None => return None,
            };
            if k == 0 {
                idx
            } else {
                format!("{idx} + {k}")
            }
        }
        None => {
            let e = elem?;
            if const_sum < 0 || const_sum as u64 % e != 0 {
                return None;
            }
            (const_sum as u64 / e).to_string()
        }
    };
    Some(format!("{var}[{idx}]"))
}

/// Element type of an array LOCAL behind `(*(nint*)(array)).element_class`.
fn local_array_elem(base: &Expr, arrays: &HashMap<String, ArrayInfo>) -> Option<String> {
    match base {
        Expr::RawDeref(inner, 0, _) | Expr::AddressOf(inner) => local_array_elem(inner, arrays),
        Expr::Var(name) => arrays.get(name).and_then(|i| i.elem.clone()),
        _ => None,
    }
}

/// Constant value of an integer literal node.
fn int_const(e: &Expr) -> Option<i64> {
    if let Expr::Int(k) = e {
        Some(*k)
    } else {
        None
    }
}

/// Split an array-element address into `(local name, byte offset from the
/// DATA start, optional (index expr, stride))`.
///
/// The address is a sum of: the array local, integer constants (whose total
/// is combined with the deref's own `addend`), and at most one scaled index
/// term (`i << k` / `i * stride`). Anything else is not an element address.
fn array_addr(base: &Expr, addend: i64) -> Option<(String, i64, Option<(Expr, u64)>)> {
    fn walk(
        e: &Expr,
        name: &mut Option<String>,
        total: &mut i64,
        idx: &mut Option<(Expr, u64)>,
    ) -> bool {
        match e {
            Expr::Int(k) => {
                *total += *k;
                true
            }
            Expr::Var(n) => {
                if name.is_some() {
                    return false;
                }
                *name = Some(n.clone());
                true
            }
            Expr::Bin(BinOp::Add, a, b) => {
                walk(a, name, total, idx) && walk(b, name, total, idx)
            }
            Expr::Bin(BinOp::Sub, a, b) => {
                if let Expr::Int(k) = b.as_ref() {
                    *total -= *k;
                    walk(a, name, total, idx)
                } else {
                    false
                }
            }
            Expr::Bin(BinOp::Shl, x, s) => {
                if idx.is_some() {
                    return false;
                }
                let Some(sh) = int_const(s) else {
                    return false;
                };
                if !(1..63).contains(&sh) {
                    return false;
                }
                *idx = Some(((**x).clone(), 1u64 << sh));
                true
            }
            Expr::Bin(BinOp::Mul, a, b) => {
                if idx.is_some() {
                    return false;
                }
                let (x, k) = match (int_const(a), int_const(b)) {
                    (Some(k), None) => (b, k),
                    (None, Some(k)) => (a, k),
                    _ => return false,
                };
                if k <= 0 {
                    return false;
                }
                *idx = Some(((**x).clone(), k as u64));
                true
            }
            _ => false,
        }
    }
    let mut name = None;
    let mut total = addend;
    let mut idx = None;
    if !walk(base, &mut name, &mut total, &mut idx) {
        return None;
    }
    let data = total.checked_sub(32)?;
    if data < 0 {
        return None;
    }
    Some((name?, data, idx))
}

/// True when the access width can be the WHOLE element: `0` is "not stated",
/// `5` is the 4-byte float lane. A narrower access inside a wider element is
/// a component, not the element.
fn width_is_element(width: u8, elem: u64) -> bool {
    let w = match width {
        // "Not stated" is a pointer-width load: it can only be a whole
        // element when the element fits in one register.
        0 => return elem <= 8,
        1 => 1,
        2 => 2,
        4 | 5 => 4,
        _ => 8,
    };
    w == elem
}

fn rewrite_expr(
    e: &mut Expr,
    arrays: &HashMap<String, ArrayInfo>,
    maps: Option<&ResolveMaps>,
) -> usize {
    let mut n = 0;
    for c in super::il2cpp_cleanup::expr_children_mut(e) {
        n += rewrite_expr(c, arrays, maps);
    }
    // ARRAY ELEMENT LOAD: `*(T*)(arr + (i << k) +32)` is `arr[i]`. The same
    // element-size rules as the store path apply; an access that lands
    // INSIDE a value-type element resolves to that element's component
    // (`*(int*)(vecs + (i << 4) +36)` → `vecs[i].y`) when metadata proves the
    // element's layout.
    if let Expr::RawDeref(base, addend, width) = e {
        if let Some((name, data, idx)) = array_addr(base, *addend) {
            if let Some(info) = arrays.get(&name) {
                if let Some(elem) = info.size.filter(|e| *e > 0) {
                    let stride_ok = match &idx {
                        None => true,
                        Some((_, stride)) => *stride == elem,
                    };
                    if stride_ok {
                        let k = data / elem as i64;
                        let rem = data % elem as i64;
                        if rem == 0 && width_is_element(*width, elem) {
                            let index = match (&idx, k) {
                                (Some((ie, _)), 0) => ie.clone(),
                                (Some((ie, _)), k) => Expr::Bin(
                                    BinOp::Add,
                                    Box::new(ie.clone()),
                                    Box::new(Expr::Int(k)),
                                ),
                                (None, k) => Expr::Int(k),
                            };
                            *e = Expr::Index {
                                collection: Box::new(Expr::Var(name)),
                                index: Box::new(index),
                            };
                            n += 1;
                        } else if rem > 0 && *width != 0 {
                            // A component read needs its PROVEN width; an
                            // unstated width could span two members, and
                            // naming it after the first would misreport it.
                            if let (Some(maps), Some(elem_name)) = (maps, info.elem.as_deref()) {
                                if let Some(&tdi) = maps.tdi_names_reverse.get(elem_name) {
                                    if maps.value_type_tdis.contains(&tdi) {
                                        if let Some(path) =
                                            super::field_annotation::layout_path_at(
                                                maps, tdi, rem, *width,
                                            )
                                        {
                                            let index = match (&idx, k) {
                                                (Some((ie, _)), 0) => ie.clone(),
                                                (Some((ie, _)), k) => Expr::Bin(
                                                    BinOp::Add,
                                                    Box::new(ie.clone()),
                                                    Box::new(Expr::Int(k)),
                                                ),
                                                (None, k) => Expr::Int(k),
                                            };
                                            *e = Expr::Field(
                                                Box::new(Expr::Index {
                                                    collection: Box::new(Expr::Var(name)),
                                                    index: Box::new(index),
                                                }),
                                                path,
                                            );
                                            n += 2;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // `(array.klass).element_class` on an array whose element type is known
    // is exactly `typeof(T)`. It was the base of 4.5k `element_class` reads.
    if let Expr::Field(base, name) = e {
        if name == "element_class" {
            if let Some(t) = local_array_elem(base, arrays) {
                *e = Expr::TypeOf(t);
                n += 1;
            }
        }
    }
    n
}

fn rewrite_stmt_exprs(
    s: &mut Stmt,
    arrays: &HashMap<String, ArrayInfo>,
    maps: Option<&ResolveMaps>,
) -> usize {
    let mut n = 0;
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Assign(_, _, e) => {
            n += rewrite_expr(e, arrays, maps);
        }
        Stmt::Return(Some(e)) => n += rewrite_expr(e, arrays, maps),
        Stmt::If { cond, .. }
        | Stmt::While { cond, .. }
        | Stmt::DoWhile { cond, .. }
        | Stmt::Lock { expr: cond, .. } => n += rewrite_expr(cond, arrays, maps),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                n += rewrite_expr(c, arrays, maps);
            }
        }
        Stmt::StringSwitch { disc, .. } => n += rewrite_expr(disc, arrays, maps),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            n += rewrite_expr(collection, arrays, maps);
            n += rewrite_expr(index, arrays, maps);
            n += rewrite_expr(value, arrays, maps);
        }
        Stmt::EventAssign { field, value, .. } => {
            n += rewrite_expr(field, arrays, maps);
            n += rewrite_expr(value, arrays, maps);
        }
        Stmt::Attribute { args, .. } => {
            for a in args {
                n += rewrite_expr(a, arrays, maps);
            }
        }
        _ => {}
    }
    n
}

fn rewrite_stmt(
    s: &mut Stmt,
    arrays: &HashMap<String, ArrayInfo>,
    maps: Option<&ResolveMaps>,
) -> usize {
    let mut n = 0;
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += rewrite_stmt(x, arrays, maps);
        }
    }
    n += rewrite_stmt_exprs(s, arrays, maps);
    if let Stmt::Assign(target, _, _) = s {
        if let Some(new) = rewrite_target(target, arrays) {
            *target = new;
            n += 1;
        }
    }
    n
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    apply_with_sizes(stmts, &HashMap::new())
}

/// `sizes` maps a type name to its byte size (see
/// `ResolveMaps::named_type_sizes`). Empty means "primitive and universal
/// reference names only", which keeps unit tests and metadata-less callers
/// sound.
pub fn apply_with_sizes(stmts: &mut Vec<Stmt>, sizes: &HashMap<String, u64>) -> usize {
    apply_inner(stmts, sizes, None)
}

/// Like [`apply_with_sizes`] but also resolves VALUE-TYPE element layouts:
/// `*(int*)(vecs + (i << 4) +36)` is `vecs[i].y` (16-byte Vector2 elements,
/// component at +4). Without metadata the pass can only see whole elements.
pub fn apply_with_sizes_and_maps(
    stmts: &mut Vec<Stmt>,
    sizes: &HashMap<String, u64>,
    maps: &ResolveMaps,
) -> usize {
    apply_inner(stmts, sizes, Some(maps))
}

fn apply_inner(
    stmts: &mut Vec<Stmt>,
    sizes: &HashMap<String, u64>,
    maps: Option<&ResolveMaps>,
) -> usize {
    let mut arrays = HashMap::new();
    collect_arrays(stmts, &mut arrays, sizes);
    if arrays.is_empty() {
        return 0;
    }
    let mut n = 0;
    for s in stmts.iter_mut() {
        n += rewrite_stmt(s, &arrays, maps);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::Expr;

    #[test]
    fn constant_and_shifted_element_stores_are_indexed() {
        let mut stmts = vec![
            Stmt::Assign(
                "arr".into(),
                Some("object[]".into()),
                Expr::ArrayInit {
                    elem_type: "object".into(),
                    items: vec![],
                },
            ),
            Stmt::Assign("*(nint*)(arr +32)".into(), None, Expr::Int(1)),
            Stmt::Assign("*(nint*)(arr +40)".into(), None, Expr::Int(2)),
            Stmt::Assign("*(nint*)(arr + (i << 3) +32)".into(), None, Expr::Int(3)),
            Stmt::Assign("*(nint*)(arr + (j * 8) +32)".into(), None, Expr::Int(4)),
            // Constant element offset after the index.
            Stmt::Assign("*(nint*)(arr + (k << 3) +48)".into(), None, Expr::Int(5)),
            // The index expression itself contains nested casts.
            Stmt::Assign(
                "*(nint*)(arr + (*(nint*)(*(nint*)(arr +16) +16) << 3) +32)".into(),
                None,
                Expr::Int(6),
            ),
        ];
        assert_eq!(apply(&mut stmts), 6);
        let targets: Vec<&str> = stmts
            .iter()
            .skip(1) // the array declaration itself
            .filter_map(|s| match s {
                Stmt::Assign(t, _, _) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            targets,
            vec![
                "arr[0]",
                "arr[1]",
                "arr[i]",
                "arr[j]",
                "arr[k + 2]",
                "arr[*(nint*)(*(nint*)(arr +16) +16)]"
            ]
        );
    }

    #[test]
    fn shifted_and_constant_element_loads_are_indexed() {
        let mut stmts = vec![
            Stmt::Assign(
                "arr".into(),
                Some("object[]".into()),
                Expr::ArrayInit {
                    elem_type: "object".into(),
                    items: vec![],
                },
            ),
            Stmt::Assign(
                "x".into(),
                None,
                Expr::RawDeref(
                    Box::new(Expr::Bin(
                        BinOp::Add,
                        Box::new(Expr::Var("arr".into())),
                        Box::new(Expr::Bin(
                            BinOp::Shl,
                            Box::new(Expr::Var("i".into())),
                            Box::new(Expr::Int(3)),
                        )),
                    )),
                    32,
                    0,
                ),
            ),
            Stmt::Assign(
                "y".into(),
                None,
                Expr::RawDeref(Box::new(Expr::Var("arr".into())), 40, 8),
            ),
        ];
        assert_eq!(apply(&mut stmts), 2);
        match &stmts[1] {
            Stmt::Assign(_, _, Expr::Index { collection, index }) => {
                assert!(matches!(collection.as_ref(), Expr::Var(v) if v == "arr"));
                assert!(matches!(index.as_ref(), Expr::Var(v) if v == "i"));
            }
            other => panic!("expected arr[i], got {other:?}"),
        }
        match &stmts[2] {
            Stmt::Assign(_, _, Expr::Index { index, .. }) => {
                assert!(matches!(index.as_ref(), Expr::Int(1)));
            }
            other => panic!("expected arr[1], got {other:?}"),
        }
    }

    fn vector_maps() -> ResolveMaps {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(61, (String::new(), "Vector3".to_string()));
        m.tdi_names_reverse.insert("Vector3".to_string(), 61);
        m.value_type_tdis.insert(61);
        m.field_layouts
            .insert((61, 0), ("x".into(), "Vector3".into()));
        m.field_layouts
            .insert((61, 4), ("y".into(), "Vector3".into()));
        m.field_layouts
            .insert((61, 8), ("z".into(), "Vector3".into()));
        m.field_widths.insert((61, 0), 4);
        m.field_widths.insert((61, 4), 4);
        m.field_widths.insert((61, 8), 4);
        m.tdi_fields.insert(
            61,
            vec![
                (0, "x".to_string(), -1),
                (4, "y".to_string(), -1),
                (8, "z".to_string(), -1),
            ],
        );
        m
    }

    #[test]
    fn struct_element_component_resolves_when_width_is_proven() {
        let mut sizes = HashMap::new();
        sizes.insert("Vector3".to_string(), 12u64);
        let mut stmts = vec![
            Stmt::Assign(
                "vecs".into(),
                Some("Vector3[]".into()),
                Expr::ArrayInit {
                    elem_type: "Vector3".into(),
                    items: vec![],
                },
            ),
            // *(int*)(vecs + i*12 +36) = vecs[i].y
            Stmt::Assign(
                "v".into(),
                None,
                Expr::RawDeref(
                    Box::new(Expr::Bin(
                        BinOp::Add,
                        Box::new(Expr::Var("vecs".into())),
                        Box::new(Expr::Bin(
                            BinOp::Mul,
                            Box::new(Expr::Var("i".into())),
                            Box::new(Expr::Int(12)),
                        )),
                    )),
                    36,
                    4,
                ),
            ),
            // *(nint*)(vecs + i*12 +36): unstated width can span y and z.
            Stmt::Assign(
                "w".into(),
                None,
                Expr::RawDeref(
                    Box::new(Expr::Bin(
                        BinOp::Add,
                        Box::new(Expr::Var("vecs".into())),
                        Box::new(Expr::Bin(
                            BinOp::Mul,
                            Box::new(Expr::Var("i".into())),
                            Box::new(Expr::Int(12)),
                        )),
                    )),
                    36,
                    0,
                ),
            ),
        ];
        let maps = vector_maps();
        let n = apply_with_sizes_and_maps(&mut stmts, &sizes, &maps);
        assert_eq!(n, 2, "{n}: {stmts:?}");
        match &stmts[1] {
            Stmt::Assign(_, _, Expr::Field(recv, name)) => {
                assert_eq!(name, "y");
                assert!(matches!(recv.as_ref(), Expr::Index { .. }));
            }
            other => panic!("expected vecs[i].y, got {other:?}"),
        }
        assert!(matches!(&stmts[2], Stmt::Assign(_, _, Expr::RawDeref(..))));
    }

    #[test]
    fn allocation_call_with_an_array_type_token_types_the_local() {
        // `strings = ArrayNew(typeof(string[]), 5)` — the allocator may render
        // as an unnamed `fn_…`; the type token alone proves the element type,
        // so `strings +32/+40` are `strings[0]`/`strings[1]`.
        let mut stmts = vec![
            Stmt::Assign(
                "strings".into(),
                None,
                Expr::RawCall(
                    0x1302484,
                    vec![Expr::TypeOf("string[]".into()), Expr::Int(5)],
                    None,
                ),
            ),
            Stmt::Assign("*(nint*)(strings +32)".into(), None, Expr::Str("a".into())),
            Stmt::Assign("*(nint*)(strings +40)".into(), None, Expr::Str("b".into())),
        ];
        assert_eq!(apply(&mut stmts), 2);
        assert!(matches!(&stmts[1], Stmt::Assign(t, _, _) if t == "strings[0]"));
        assert!(matches!(&stmts[2], Stmt::Assign(t, _, _) if t == "strings[1]"));
    }

    #[test]
    fn named_class_array_stores_use_the_metadata_element_size() {
        let mut sizes = HashMap::new();
        sizes.insert("CoinChainDebug".to_string(), 8u64);
        let mut stmts = vec![
            Stmt::Assign(
                "arr".into(),
                None,
                Expr::NewArr("CoinChainDebug".into(), vec![Expr::Int(3)]),
            ),
            Stmt::Assign(
                "*(nint*)(arr + (i << 3) +32)".into(),
                None,
                Expr::Var("x".into()),
            ),
        ];
        assert_eq!(apply_with_sizes(&mut stmts, &sizes), 1);
        assert!(matches!(&stmts[1], Stmt::Assign(t, _, _) if t == "arr[i]"));
    }

    #[test]
    fn element_class_of_a_known_array_local_is_the_element_type() {
        let mut stmts = vec![
            Stmt::Assign(
                "arr".into(),
                None,
                Expr::NewArr("UnityEngine.Keyframe".into(), vec![Expr::Int(2)]),
            ),
            Stmt::Assign(
                "v".into(),
                None,
                Expr::Field(
                    Box::new(Expr::RawDeref(Box::new(Expr::Var("arr".into())), 0, 8)),
                    "element_class".into(),
                ),
            ),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(
            &stmts[1],
            Stmt::Assign(_, _, Expr::TypeOf(t)) if t == "UnityEngine.Keyframe"
        ));
    }

    #[test]
    fn width_mismatch_is_refused() {
        let mut stmts = vec![
            Stmt::Assign(
                "ints".into(),
                Some("int[]".into()),
                Expr::ArrayInit {
                    elem_type: "int".into(),
                    items: vec![],
                },
            ),
            // An 8-byte store into an int[] is not an element write.
            Stmt::Assign("*(nint*)(ints +32)".into(), None, Expr::Int(1)),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(
            match &stmts[1] {
                Stmt::Assign(t, _, _) => t.as_str(),
                _ => "",
            },
            "*(nint*)(ints +32)"
        );
    }
}
