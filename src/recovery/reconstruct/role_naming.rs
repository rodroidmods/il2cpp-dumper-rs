//! Evidence-based role names for the generated locals that survived the
//! folding passes.
//!
//! Metadata has no local names, so a name is only assigned when the USE gives
//! one, and never when the evidence is ambiguous:
//!
//! * **counter** — the local is updated from itself (`v = v ± k`, `v++`) AND
//!   compared against a `Length`/`Count` or used as an index → `i`/`j`/`k`.
//! * **index** — read inside `collection[v]` → `index`, `index2`, …
//! * **field destination** — stored to exactly ONE field (`this.X = v`) →
//!   the field name (`X`, leading `_`/`m_` stripped).
//! * **call result** — exactly one definition, a call: `GetFoo`/`get_Foo` →
//!   `foo`, `MoveNext` → `moveNext`, `IL2CPP_IsInst` → `instance`.
//! * **returned** — exactly one definition, read by `return` → `result`.
//!
//! Everything else keeps its generated name rather than inventing one. The
//! pass runs after every folding pass and before `declare_locals`, so it only
//! sees the names that actually reach the text. Names are allocated in first-
//! occurrence order, never collide with an identifier already present in the
//! method (including member/method names, because the renderer sometimes
//! spells a field without `this.`), and lambda bodies are never rewritten
//! (their parameters are a separate scope).

use crate::recovery::csharp::{BinOp, Expr, Stmt};
use crate::recovery::heuristic_types::HeuristicIndex;
use std::collections::{HashMap, HashSet};

fn is_generated(n: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if let Some(rest) = n.strip_prefix('v') {
        return digits(rest);
    }
    for prefix in ["obj", "ptr", "num", "flag"] {
        if let Some(rest) = n.strip_prefix(prefix) {
            if digits(rest) {
                return true;
            }
        }
    }
    false
}

fn is_ident(n: &str) -> bool {
    !n.is_empty()
        && n.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !matches!(
            n,
            "this" | "base" | "null" | "true" | "false" | "new" | "default" | "class"
                | "int" | "long" | "float" | "double" | "bool" | "byte" | "short" | "uint"
                | "string" | "object" | "void" | "return" | "break" | "continue" | "if"
                | "else" | "while" | "for" | "foreach" | "do" | "switch" | "case" | "try"
                | "catch" | "finally" | "goto" | "in" | "out" | "ref" | "is" | "as"
        )
}

#[derive(Default)]
struct Facts {
    order: usize,
    reads: usize,
    defs: usize,
    /// How many of the definitions are call results (all to the same callee
    /// when `calls.len() == 1`).
    call_defs: usize,
    self_step: bool,
    cmp_len: bool,
    index_use: bool,
    fields: HashSet<String>,
    calls: HashSet<String>,
    /// Methods called ON this local as the direct receiver (`local.Foo()`).
    member_methods: HashSet<String>,
    /// Fields read on this local as the direct receiver (`local.foo`).
    member_fields: HashSet<String>,
    returned: bool,
}

fn expr_children(e: &Expr) -> Vec<&Expr> {
    crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e)
}

fn expr_children_mut(e: &mut Expr) -> Vec<&mut Expr> {
    crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e)
}

fn last_segment(s: &str) -> &str {
    s.rsplit('.').next().unwrap_or(s)
}

fn strip_generic(s: &str) -> &str {
    s.split('<').next().unwrap_or(s)
}

fn note_used_expr(e: &Expr, used: &mut HashSet<String>) {
    match e {
        Expr::Var(n) if !is_generated(n) => {
            used.insert(n.clone());
        }
        Expr::Field(_, f) | Expr::StaticField(_, f) => {
            used.insert(f.clone());
        }
        Expr::Call { name, .. } => {
            used.insert(last_segment(strip_generic(name)).to_string());
        }
        Expr::MethodGroup(n) | Expr::TypeOf(n) => {
            used.insert(last_segment(strip_generic(n)).to_string());
        }
        Expr::Lambda { params, .. } => {
            for p in params {
                used.insert(p.clone());
            }
        }
        _ => {}
    }
    for c in expr_children(e) {
        note_used_expr(c, used);
    }
}

fn gen_vars_in(e: &Expr, out: &mut HashSet<String>) {
    if let Expr::Var(n) = e {
        if is_generated(n) {
            out.insert(n.clone());
        }
    }
    for c in expr_children(e) {
        gen_vars_in(c, out);
    }
}

fn has_len_member(e: &Expr) -> bool {
    match e {
        Expr::Field(_, f) if f == "Length" || f == "Count" => return true,
        _ => {}
    }
    expr_children(e).into_iter().any(has_len_member)
}

/// A generated local multiplied by an element scale inside an address
/// expression is an array index, even when field/array sugar did not fire
/// (`*(nint*)(base + v364 * 8 + 32)`).
fn scaled_index_vars(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Bin(op, a, b) if matches!(op, BinOp::Mul) => {
            let (v, k) = match (a.as_ref(), b.as_ref()) {
                (Expr::Var(n), Expr::Int(k)) | (Expr::Int(k), Expr::Var(n))
                    if is_generated(n) =>
                {
                    (Some(n), *k)
                }
                _ => (None, 0),
            };
            if let Some(v) = v {
                if matches!(k, 2 | 4 | 8 | 16) {
                    out.insert(v.clone());
                }
            }
        }
        Expr::Bin(op, a, _) if matches!(op, BinOp::Shl) => {
            if let Expr::Var(n) = a.as_ref() {
                if is_generated(n) {
                    out.insert(n.clone());
                }
            }
        }
        _ => {}
    }
    for c in expr_children(e) {
        scaled_index_vars(c, out);
    }
}

fn scan_expr(e: &Expr, facts: &mut HashMap<String, Facts>, used: &mut HashSet<String>, order: &mut usize) {
    note_used_expr(e, used);
    match e {
        Expr::Var(n) => {
            if is_generated(n) {
                let f = facts.entry(n.clone()).or_default();
                if f.order == 0 {
                    f.order = *order;
                }
                f.reads += 1;
                *order += 1;
            }
        }
        Expr::Index { index, .. } => {
            let mut gs = HashSet::new();
            gen_vars_in(index, &mut gs);
            for g in gs {
                facts.entry(g).or_default().index_use = true;
            }
        }
        Expr::Call {
            recv: Some(recv),
            name,
            ..
        } => {
            if let Expr::Var(g) = recv.as_ref() {
                if is_generated(g) {
                    facts
                        .entry(g.clone())
                        .or_default()
                        .member_methods
                        .insert(last_segment(strip_generic(name)).to_string());
                }
            }
        }
        Expr::Field(base, field) => {
            if let Expr::Var(g) = base.as_ref() {
                if is_generated(g) {
                    facts
                        .entry(g.clone())
                        .or_default()
                        .member_fields
                        .insert(field.clone());
                }
            }
        }
        Expr::Cmp(_, a, b) => {
            let (la, lb) = (has_len_member(a), has_len_member(b));
            if la || lb {
                let mut gs = HashSet::new();
                if la {
                    gen_vars_in(b, &mut gs);
                }
                if lb {
                    gen_vars_in(a, &mut gs);
                }
                for g in gs {
                    facts.entry(g).or_default().cmp_len = true;
                }
            }
        }
        Expr::RawDeref(base, _, _) | Expr::RawAddr(base, _) => {
            let mut gs = HashSet::new();
            scaled_index_vars(base, &mut gs);
            for g in gs {
                facts.entry(g).or_default().index_use = true;
            }
        }
        Expr::Lambda { params, .. } => {
            for p in params {
                used.insert(p.clone());
            }
            // A lambda body is a separate scope; do not gather rename facts
            // from it (the renamer will not descend there either).
            return;
        }
        _ => {}
    }
    for c in expr_children(e) {
        scan_expr(c, facts, used, order);
    }
}

fn note_text_used(text: &str, used: &mut HashSet<String>) {
    for tok in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if !tok.is_empty() && !tok.chars().next().unwrap().is_ascii_digit() {
            used.insert(tok.to_string());
        }
    }
}

fn scan_list(stmts: &[Stmt], facts: &mut HashMap<String, Facts>, used: &mut HashSet<String>, order: &mut usize) {
    for s in stmts {
        crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
            scan_expr(e, facts, used, order)
        });
        match s {
            Stmt::Assign(target, _, rhs) => {
                if is_generated(target) {
                    let f = facts.entry(target.clone()).or_default();
                    if f.order == 0 {
                        f.order = *order;
                    }
                    f.defs += 1;
                    *order += 1;
                    if let Expr::Bin(op, a, b) = rhs {
                        if matches!(op, BinOp::Add | BinOp::Sub) {
                            if matches!(a.as_ref(), Expr::Var(n) if n == target)
                                && matches!(b.as_ref(), Expr::Int(_))
                            {
                                f.self_step = true;
                            }
                        }
                    }
                    match rhs {
                        Expr::Call { name, .. } => {
                            f.calls.insert(last_segment(strip_generic(name)).to_string());
                            f.call_defs += 1;
                        }
                        _ => {}
                    }
                } else {
                    note_text_used(target, used);
                    if let Expr::Var(v) = rhs {
                        if is_generated(v) && target.contains('.') {
                            if let Some(field) = target.rsplit('.').next() {
                                if is_ident(field) {
                                    facts.entry(v.clone()).or_default().fields.insert(field.to_string());
                                }
                            }
                        }
                    }
                }
            }
            Stmt::Declare(name, _) => {
                if is_generated(name) {
                    facts.entry(name.clone()).or_default().defs += 1;
                } else {
                    used.insert(name.clone());
                }
            }
            Stmt::Switch { disc, .. } => {
                if is_generated(disc) {
                    let f = facts.entry(disc.clone()).or_default();
                    f.reads += 1;
                } else {
                    note_text_used(disc, used);
                }
            }
            Stmt::ForEach {
                var_name,
                collection,
                ..
            } => {
                if is_generated(var_name) {
                    facts.entry(var_name.clone()).or_default().defs += 1;
                } else {
                    used.insert(var_name.clone());
                }
                if is_generated(collection) {
                    facts.entry(collection.clone()).or_default().reads += 1;
                } else {
                    note_text_used(collection, used);
                }
            }
            Stmt::Return(Some(Expr::Var(n))) if is_generated(n) => {
                facts.entry(n.clone()).or_default().returned = true;
            }
            _ => {}
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            scan_list(k, facts, used, order);
        }
    }
}

fn field_base(field: &str) -> Option<String> {
    let mut s = field.trim_start_matches('_');
    if let Some(rest) = s.strip_prefix("m_") {
        s = rest;
    }
    if s.is_empty() || !is_ident(s) {
        return None;
    }
    let mut c = s.chars();
    let first = c.next().unwrap().to_ascii_lowercase();
    let out: String = std::iter::once(first).chain(c).collect();
    if out.len() < 2 {
        None
    } else {
        Some(out)
    }
}

fn callee_base(callee: &str) -> Option<String> {
    let last = strip_generic(last_segment(callee));
    if last == "IL2CPP_IsInst" {
        return Some("instance".into());
    }
    let base = if let Some(r) = last.strip_prefix("get_") {
        r
    } else if let Some(r) = last.strip_prefix("Get") {
        r
    } else {
        last
    };
    if base.is_empty() || !base.chars().next().unwrap().is_ascii_alphabetic() {
        return None;
    }
    let mut c = base.chars();
    let first = c.next().unwrap().to_ascii_lowercase();
    let out: String = std::iter::once(first).chain(c).collect();
    if out.len() < 3 || !is_ident(&out) {
        return None;
    }
    Some(out)
}

fn lower_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => std::iter::once(f.to_ascii_lowercase()).chain(c).collect(),
        None => String::new(),
    }
}

fn type_name_base(name: &str) -> Option<String> {
    let simple = last_segment(strip_generic(name));
    if simple.is_empty() {
        return None;
    }
    let out = lower_first(simple);
    (out.len() >= 3 && is_ident(&out)).then_some(out)
}

/// A name from the receiver's USE: every method called on the local and every
/// field read from it must resolve to ONE metadata-declared type. Generic
/// `object` members (`Equals`, `ToString`, …) are ignored; any ambiguous
/// member or disagreeing type abstains.
fn member_type_role(f: &Facts, types: &HeuristicIndex) -> Option<String> {
    let mut names: HashSet<String> = HashSet::new();
    let mut any = false;
    for m in &f.member_methods {
        if matches!(
            m.as_str(),
            "Equals" | "GetHashCode" | "ToString" | "GetType" | "MemberwiseClone"
                | "ReferenceEquals"
        ) {
            continue;
        }
        let t = types.unique_type_name_for_method(m)?;
        names.insert(type_name_base(t)?);
        any = true;
    }
    for fld in &f.member_fields {
        let t = types.unique_type_name_for_field(fld)?;
        names.insert(type_name_base(t)?);
        any = true;
    }
    if any && names.len() == 1 {
        names.into_iter().next()
    } else {
        None
    }
}

fn pick_base(f: &Facts, types: Option<&HeuristicIndex>) -> Option<String> {
    if f.self_step && (f.cmp_len || f.index_use || f.reads >= 2) {
        return Some("i".into());
    }
    if f.index_use {
        return Some("index".into());
    }
    if f.fields.len() == 1 {
        if let Some(base) = f.fields.iter().next().and_then(|x| field_base(x)) {
            return Some(base);
        }
    }
    if f.defs > 0 && f.call_defs == f.defs && f.calls.len() == 1 {
        if let Some(base) = f.calls.iter().next().and_then(|x| callee_base(x)) {
            return Some(base);
        }
    }
    if let Some(types) = types {
        if let Some(base) = member_type_role(f, types) {
            return Some(base);
        }
    }
    if f.returned {
        return Some("result".into());
    }
    None
}

const COUNTERS: &[&str] = &["i", "j", "k", "l", "m", "n"];

fn allocate(base: &str, used: &HashSet<String>, counters: &mut usize) -> String {
    if base == "i" {
        let idx = *counters;
        *counters += 1;
        if idx < COUNTERS.len() && !used.contains(COUNTERS[idx]) {
            return COUNTERS[idx].to_string();
        }
    }
    if !used.contains(base) {
        return base.to_string();
    }
    for k in 2.. {
        let cand = format!("{base}{k}");
        if !used.contains(&cand) {
            return cand;
        }
    }
    unreachable!()
}

fn rename_expr(e: &mut Expr, map: &HashMap<String, String>) -> usize {
    if let Expr::Var(n) = e {
        if let Some(new) = map.get(n.as_str()) {
            *n = new.clone();
            return 1;
        }
        return 0;
    }
    if matches!(e, Expr::Lambda { .. }) {
        return 0;
    }
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += rename_expr(c, map);
    }
    n
}

fn rename_text(text: &str, map: &HashMap<String, String>) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut changed = false;
    let mut prev_nonspace: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_alphanumeric() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let tok = &text[start..i];
            if prev_nonspace != Some(b'.') {
                if let Some(new) = map.get(tok) {
                    out.push_str(new);
                    changed = true;
                    prev_nonspace = new.as_bytes().last().copied();
                    continue;
                }
            }
            out.push_str(tok);
            prev_nonspace = tok.as_bytes().last().copied();
            continue;
        }
        out.push(c as char);
        if !c.is_ascii_whitespace() {
            prev_nonspace = Some(c);
        }
        i += 1;
    }
    if changed {
        Some(out)
    } else {
        None
    }
}

fn rename_list(stmts: &mut [Stmt], map: &HashMap<String, String>, n: &mut usize) {
    for s in stmts {
        crate::recovery::reconstruct::copy_prop::for_each_expr_mut(s, &mut |e| {
            *n += rename_expr(e, map);
        });
        match s {
            Stmt::Assign(target, _, _) => {
                if let Some(new) = map.get(target.as_str()) {
                    *target = new.clone();
                    *n += 1;
                } else if let Some(nt) = rename_text(target, map) {
                    *target = nt;
                    *n += 1;
                }
            }
            Stmt::Declare(name, _) => {
                if let Some(new) = map.get(name.as_str()) {
                    *name = new.clone();
                    *n += 1;
                }
            }
            Stmt::Switch { disc, .. } => {
                if let Some(nt) = rename_text(disc, map) {
                    *disc = nt;
                    *n += 1;
                }
            }
            Stmt::ForEach {
                var_name,
                collection,
                ..
            } => {
                if let Some(new) = map.get(var_name.as_str()) {
                    *var_name = new.clone();
                    *n += 1;
                }
                if let Some(nt) = rename_text(collection, map) {
                    *collection = nt;
                    *n += 1;
                }
            }
            _ => {}
        }
        for k in super::children_mut(s) {
            rename_list(k, map, n);
        }
    }
}

pub fn apply(stmts: &mut Vec<Stmt>, types: Option<&HeuristicIndex>) -> usize {
    let mut facts: HashMap<String, Facts> = HashMap::new();
    let mut used: HashSet<String> = HashSet::new();
    let mut order = 1usize;
    scan_list(stmts, &mut facts, &mut used, &mut order);
    let mut names: Vec<String> = facts.keys().cloned().collect();
    names.sort_by_key(|n| (facts[n].order, n.clone()));
    let mut map: HashMap<String, String> = HashMap::new();
    let mut counters = 0usize;
    for name in names {
        if !is_generated(&name) {
            continue;
        }
        let Some(base) = pick_base(&facts[&name], types) else {
            continue;
        };
        let cand = allocate(&base, &used, &mut counters);
        used.insert(cand.clone());
        map.insert(name, cand);
    }
    if map.is_empty() {
        return 0;
    }
    let mut n = 0usize;
    rename_list(stmts, &map, &mut n);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    #[test]
    fn a_counter_and_an_index_get_role_names() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Bin(BinOp::Add, Box::new(var("v1")), Box::new(Expr::Int(1))),
            ),
            Stmt::If {
                cond: Expr::Cmp(
                    crate::recovery::csharp::CmpKind::Lt,
                    Box::new(var("v1")),
                    Box::new(Expr::Field(
                        Box::new(var("items")),
                        "Length".into(),
                    )),
                ),
                then_body: vec![Stmt::Eval(Expr::Index {
                    collection: Box::new(var("items")),
                    index: Box::new(var("v1")),
                })],
                else_body: vec![],
            },
        ];
        apply(&mut stmts, None);
        let d = format!("{stmts:?}");
        assert!(!d.contains("v1"), "{d}");
        assert!(d.contains("\"i\"") || d.contains("i "), "{d}");
    }

    #[test]
    fn a_plain_value_keeps_its_generated_name() {
        let mut stmts = vec![
            Stmt::Assign("v2".into(), None, Expr::Int(5)),
            Stmt::Eval(var("v2")),
        ];
        apply(&mut stmts, None);
        let d = format!("{stmts:?}");
        assert!(d.contains("v2"), "{d}");
    }

    #[test]
    fn a_unique_field_store_names_the_temp() {
        let mut stmts = vec![
            Stmt::Assign("v3".into(), None, Expr::Call {
                recv: None,
                name: "GetFoo".into(),
                args: vec![],
            }),
            Stmt::Assign("this.fooBar".into(), None, var("v3")),
            Stmt::Eval(var("v3")),
        ];
        apply(&mut stmts, None);
        let d = format!("{stmts:?}");
        assert!(!d.contains("v3"), "{d}");
        assert!(d.contains("fooBar"), "{d}");
    }
}
