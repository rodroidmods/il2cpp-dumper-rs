//! Give `stack_N` cells semantic names when a call site proves the role.
//!
//! ## Why these exist
//!
//! IL2CPP stores value-type locals on the frame and passes their ADDRESS to
//! instance methods. The structurer names the frame cell `stack_N`
//! (`stack_cell_name`); the local's `LocalInfo.ty` is usually `Unknown`
//! because the cell is only used as a memory base. 17k references in the
//! corpus are such cells.
//!
//! ## Proven roles (call-shape evidence, no guessing)
//!
//! * `(&cell).MoveNext()` / `(&cell).get_Current` / `(&cell).Dispose` →
//!   `enumerator` (a struct enumerator stored on the frame).
//! * `(&cell).GetResult()` / `(&cell).get_IsCompleted` → `awaiter`.
//! * `TryGetValue(…, &cell)` / `TryParse(…, &cell)` (any call whose name
//!   begins with `Try` and takes the cell by address) → `outValue`.
//!
//! A cell with no matching call keeps its stable `stack_N` name: the pass
//! never renames from a type guess, only from a method the cell is provably
//! passed to. Addresses are renamed consistently (`&stack_129` →
//! `&enumerator`), including complex assignment targets such as
//! `*(int*)((nint)&stack_133 +4)`.
//!
//! Renaming a stack cell is a pure substitution of one display name; values
//! and control flow are untouched.

use super::super::csharp::{BinOp, Expr, Stmt};
use super::super::resolve::ResolveMaps;
use super::super::types::{ManagedType, MethodRef, PrimKind};
use super::children_mut;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

fn is_stack_cell(n: &str) -> bool {
    n.starts_with("stack_") && n[6..].chars().all(|c| c.is_ascii_digit()) && n.len() > 6
}

/// The stack cell an expression addresses (`&stack_N`, possibly cast).
fn addressed_cell(e: &Expr) -> Option<&str> {
    match e {
        // A struct instance method on a frame cell: the AST receiver is the
        // cell itself; the RENDERER prints it as `(&stack_N)` because a stack
        // local names memory, not a value.
        Expr::Var(n) if is_stack_cell(n) => Some(n),
        Expr::AddressOf(inner) => match inner.as_ref() {
            Expr::Var(n) if is_stack_cell(n) => Some(n),
            _ => None,
        },
        Expr::Cast(_, inner) => addressed_cell(inner),
        Expr::RawAddr(inner, _) => match inner.as_ref() {
            Expr::Var(n) if is_stack_cell(n) => Some(n),
            _ => None,
        },
        _ => None,
    }
}

fn classify_call(call_name: &str) -> Option<&'static str> {
    match call_name {
        "MoveNext" | "get_Current" | "Dispose" => Some("enumerator"),
        "GetResult" | "get_IsCompleted" => Some("awaiter"),
        n if n.starts_with("Try") => Some("outValue"),
        _ => None,
    }
}

/// Members that identify a BCL value type carried in a frame cell. The cell
/// has no metadata type, but a struct whose reads are `Year`/`Month`/`Day`/
/// `Hour`/`Minute`/`Second` is a DateTime, and one read through `Ticks`/
/// `Total*` is a TimeSpan. Two DISTINCT members are required, so a lone
/// `get_Day` on an unrelated struct cannot trigger a rename.
const DATE_MEMBERS: &[&str] = &["Year", "Month", "Day", "Hour", "Minute", "Second"];
const SPAN_MEMBERS: &[&str] = &[
    "Ticks",
    "TotalHours",
    "TotalMinutes",
    "TotalSeconds",
    "TotalMilliseconds",
];

fn member_base(name: &str) -> &str {
    name.strip_prefix("get_").unwrap_or(name)
}

fn collect_type_evidence(stmts: &[Stmt], ev: &mut HashMap<String, HashSet<&'static str>>) {
    for s in stmts {
        for e in stmt_exprs(s) {
            collect_type_evidence_expr(e, ev);
        }
        for k in children_lists(s) {
            collect_type_evidence(k, ev);
        }
    }
}

fn collect_type_evidence_expr(e: &Expr, ev: &mut HashMap<String, HashSet<&'static str>>) {
    if let Expr::Call { recv, name, .. } = e {
        if let Some(r) = recv {
            if let Some(cell) = addressed_cell(r) {
                let base = member_base(name);
                let known = DATE_MEMBERS
                    .iter()
                    .chain(SPAN_MEMBERS.iter())
                    .find(|m| **m == base);
                if let Some(m) = known {
                    ev.entry(cell.to_string()).or_default().insert(m);
                }
            }
        }
    }
    for c in expr_children(e) {
        collect_type_evidence_expr(c, ev);
    }
}

fn classify_arg_role(call_name: &str) -> Option<&'static str> {
    match call_name {
        // `collection.GetEnumerator(&cell)`: the address is the enumerator
        // the callee fills in.
        "GetEnumerator" => Some("enumerator"),
        // `list.get_Item(i, &cell)`: a value-type element is returned through
        // the hidden pointer argument.
        "get_Item" => Some("item"),
        _ => None,
    }
}

fn collect_roles_expr(e: &Expr, roles: &mut HashMap<String, &'static str>) {
    if let Expr::Call { recv, name, args } = e {
        if let Some(r) = recv {
            if let Some(cell) = addressed_cell(r) {
                if let Some(role) = classify_call(name) {
                    roles.entry(cell.to_string()).or_insert(role);
                }
            }
        }
        if name.starts_with("Try") {
            for a in args {
                if let Some(cell) = addressed_cell(a) {
                    roles.entry(cell.to_string()).or_insert("outValue");
                }
            }
        }
        if let Some(role) = classify_arg_role(name) {
            for a in args {
                if let Some(cell) = addressed_cell(a) {
                    roles.entry(cell.to_string()).or_insert(role);
                }
            }
        }
    }
    for c in expr_children(e) {
        collect_roles_expr(c, roles);
    }
}

fn expr_children(e: &Expr) -> Vec<&Expr> {
    // The canonical visitor: a local copy here silently missed compound
    // expressions (object initializers folded after this pass was written,
    // string interpolations, method chains), leaving machine stack names
    // un-renamed inside them.
    super::il2cpp_cleanup::expr_children(e)
}

fn stmt_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => vec![e],
        Stmt::Return(Some(e)) => vec![e],
        Stmt::Assign(_, _, e) => vec![e],
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            vec![cond]
        }
        Stmt::Lock { expr, .. } => vec![expr],
        Stmt::StringSwitch { disc, .. } => vec![disc],
        Stmt::For { cond, .. } => cond.iter().collect(),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => vec![collection, index, value],
        Stmt::EventAssign { field, value, .. } => vec![field, value],
        _ => vec![],
    }
}

fn collect_roles(stmts: &[Stmt], roles: &mut HashMap<String, &'static str>) {
    for s in stmts {
        for e in stmt_exprs(s) {
            collect_roles_expr(e, roles);
        }
        for k in children_lists(s) {
            collect_roles(k, roles);
        }
    }
}

fn children_lists(s: &Stmt) -> Vec<&Vec<Stmt>> {
    let mut v: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut v);
    v
}

/// Replace `old` with `new` when it appears as a whole identifier token.
fn replace_token(text: &str, old: &str, new: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if text[i..].starts_with(old) {
            let before_ok = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let end = i + old.len();
            let after_ok = end == bytes.len()
                || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
            if before_ok && after_ok {
                out.push_str(new);
                i = end;
                continue;
            }
        }
        out.push(text[i..].chars().next().unwrap());
        i += text[i..].chars().next().unwrap().len_utf8();
    }
    out
}

fn rename_expr(e: &mut Expr, map: &HashMap<String, String>) -> usize {
    let mut n = 0;
    match e {
        Expr::Var(name) => {
            if let Some(new) = map.get(name.as_str()) {
                *name = new.clone();
                n += 1;
            }
        }
        _ => {
            for c in expr_children_mut(e) {
                n += rename_expr(c, map);
            }
        }
    }
    n
}

fn expr_children_mut(e: &mut Expr) -> Vec<&mut Expr> {
    super::il2cpp_cleanup::expr_children_mut(e)
}

fn rename_stmt(s: &mut Stmt, map: &HashMap<String, String>) -> usize {
    let mut n = 0;
    // Assignment targets are plain strings and may embed the cell inside an
    // expression (`*(int*)((nint)&stack_133 +4)`), so rewrite tokens there.
    if let Stmt::Assign(target, _, _) = s {
        let mut t = target.clone();
        for (old, new) in map {
            t = replace_token(&t, old, new);
        }
        if t != *target {
            *target = t;
            n += 1;
        }
    }
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += rename_stmt(x, map);
        }
    }
    let mut fold = |e: &mut Expr| n += rename_expr(e, map);
    match s {
        Stmt::Eval(e)
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e)
        | Stmt::Return(Some(e))
        | Stmt::Assign(_, _, e) => fold(e),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => fold(cond),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                fold(c);
            }
        }
        Stmt::Lock { expr, .. } => fold(expr),
        Stmt::StringSwitch { disc, .. } => fold(disc),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            fold(collection);
            fold(index);
            fold(value);
        }
        Stmt::EventAssign { field, value, .. } => {
            fold(field);
            fold(value);
        }
        _ => {}
    }
    n
}

/// Name proven stack cells. Returns the number of cells renamed.
///
/// This is the map-free entry point: call-role evidence only (the historical
/// behaviour, used by tests). The production pipeline calls
/// [`apply_with_maps`], which additionally types cells from callee signatures
/// and metadata field layouts.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    apply_cells(stmts, None)
}

fn collect_role_names(stmts: &[Stmt]) -> HashMap<String, String> {
    let mut roles: HashMap<String, &'static str> = HashMap::new();
    collect_roles(stmts, &mut roles);
    // Call-role evidence wins; type evidence only names cells no call proved.
    let mut typed: HashMap<String, HashSet<&'static str>> = HashMap::new();
    collect_type_evidence(stmts, &mut typed);
    for (cell, members) in typed {
        if roles.contains_key(&cell) {
            continue;
        }
        let date = members.iter().filter(|m| DATE_MEMBERS.contains(m)).count();
        let span = members.iter().filter(|m| SPAN_MEMBERS.contains(m)).count();
        match (date >= 2, span >= 2) {
            (true, false) => {
                roles.insert(cell, "dateTime");
            }
            (false, true) => {
                roles.insert(cell, "timeSpan");
            }
            _ => {}
        }
    }
    roles
        .into_iter()
        .map(|(cell, role)| (cell, role.to_string()))
        .collect()
}

fn allocate_names(
    stmts: &[Stmt],
    wanted: Vec<(String, String)>,
) -> HashMap<String, String> {
    let mut used: HashSet<String> = HashSet::new();
    for s in stmts.iter() {
        collect_existing_names(s, &mut used);
    }
    let mut map: HashMap<String, String> = HashMap::new();
    for (cell, base) in wanted {
        let mut candidate = base.clone();
        let mut k = 2usize;
        while used.contains(&candidate) {
            candidate = format!("{base}{k}");
            k += 1;
        }
        used.insert(candidate.clone());
        map.insert(cell, candidate);
    }
    map
}

fn apply_cells(stmts: &mut Vec<Stmt>, typed: Option<&CellCtx<'_>>) -> usize {
    let roles = collect_role_names(stmts);
    let role_cells: HashSet<String> = roles.keys().cloned().collect();
    // `roles` is a HashMap: iterating it directly handed `enumerator`,
    // `enumerator2`, ... to sibling cells in RANDOM hash order, so two runs of
    // the same binary swapped the names between methods (measured: 124 of 317
    // sampled methods differed run to run). Sort by cell before allocation.
    let mut wanted: Vec<(String, String)> = roles.into_iter().collect();
    wanted.sort();
    let mut field_plan: HashMap<(String, i64, u8), String> = HashMap::new();
    let mut unwrap_cells: HashSet<String> = HashSet::new();
    let mut split_kinds: HashMap<String, HashSet<UseKind>> = HashMap::new();
    let mut split_strong: HashSet<String> = HashSet::new();
    if let Some(ctx) = typed {
        let (cell_types, evidence) = infer_cells(stmts, ctx);
        for (cell, ty) in &cell_types {
            let tdi = ty.tdi;
            if tdi >= 0 && ctx.is_value_type(tdi) {
                unwrap_cells.insert(cell.clone());
                collect_field_plan(&evidence, cell, tdi, ctx, &mut field_plan);
            }
        }
        let typed_cells: HashSet<String> = cell_types.keys().cloned().collect();
        // READ-ONLY SPLIT: a cell with no store and no address escape cannot
        // be written by anything the body can see. When its reads are used as
        // two different kinds (a string receiver and an array index), the
        // slot was reused; name each use region from the proven use instead
        // of forcing one wrong type onto the whole cell.
        let mut excluded: HashSet<String> = role_cells.clone();
        excluded.extend(typed_cells);
        excluded.extend(evidence.array_cells.iter().cloned());
        excluded.extend(evidence.address_uses.iter().cloned());
        for (cell, _, _) in &evidence.raw_stores {
            excluded.insert(cell.clone());
        }
        for (cell, _) in &evidence.defs {
            excluded.insert(cell.clone());
        }
        let candidates: HashSet<String> = evidence
            .seen_cells
            .difference(&excluded)
            .cloned()
            .collect();
        if !candidates.is_empty() {
            let (mut use_kinds, strong) = collect_use_kinds(stmts, &candidates, ctx);
            use_kinds.retain(|_, set| set.len() >= 2);
            if !use_kinds.is_empty() {
                split_kinds = use_kinds;
                split_strong = strong;
            }
        }
        let mut names: Vec<(String, String)> = Vec::new();
        for (cell, ty) in cell_types {
            let name = cell_name_for(&ty);
            names.push((cell, name));
        }
        names.sort();
        wanted.extend(names);
    }
    // Role names claim their cells first; typed names skip those cells so a
    // proven call shape (`enumerator`, `outValue`) is never replaced by a
    // type-shaped name.
    let mut seen: HashSet<String> = HashSet::new();
    wanted.retain(|(cell, _)| seen.insert(cell.clone()));
    if wanted.is_empty()
        && field_plan.is_empty()
        && unwrap_cells.is_empty()
        && split_kinds.is_empty()
    {
        return 0;
    }
    let map = allocate_names(stmts, wanted);
    let split_names = allocate_split_names(stmts, &split_kinds, &map);
    let mut n = 0;
    for s in stmts.iter_mut() {
        if let Some(ctx) = typed {
            if !split_names.is_empty() {
                n += split_uses_stmt(s, &split_names, &split_strong, ctx);
            }
        }
        n += rewrite_cell_stmt(s, &map, &field_plan, &unwrap_cells);
    }
    n
}

/// Role names are the fixed words the call-shape classifier emits.
fn collect_existing_names(s: &Stmt, used: &mut HashSet<String>) {
    match s {
        Stmt::Assign(n, _, _) => {
            used.insert(n.clone());
        }
        Stmt::ForEach { var_name, .. } => {
            used.insert(var_name.clone());
        }
        _ => {}
    }
    for e in stmt_exprs(s) {
        collect_var_names(e, used);
    }
    for k in children_lists(s) {
        for x in k {
            collect_existing_names(x, used);
        }
    }
}

fn collect_var_names(e: &Expr, used: &mut HashSet<String>) {
    if let Expr::Var(n) = e {
        used.insert(n.clone());
    }
    for c in expr_children(e) {
        collect_var_names(c, used);
    }
}

// ===========================================================================
// Type-directed cell classification
// ===========================================================================
//
// Metadata has no local names, but it does have method signatures and field
// layouts. A frame cell is typed only when one of these proofs holds:
//
// 1. The cell's ADDRESS is passed to a resolved method (`&cell`, `RawAddr` of
//    the cell, `(nint*)(cell + off)`) and EVERY overload of that call at the
//    same arity declares that parameter `ref`/`out` T. All overloads must
//    agree, otherwise the call abstains.
// 2. The cell is the receiver of an instance method whose name resolves to
//    exactly one declaring type in the whole metadata.
// 3. A field read `cell.F` names a field declared by exactly one type.
// 4. A copy `cell = other` whose source type is proven propagates the type.
// 5. A whole-cell store whose RHS expression has a provable primitive/class
//    type (metadata local type, NewObj, cast, float/bool literal, or a unique
//    method return type) types the cell.
//
// Conflicting proofs abstain: a cell reused for two types keeps its stable
// `stack_N` name rather than inventing a merge. Field sugar is only applied
// to types PROVEN value types (the metadata layout has a field below the
// 16-byte object header) because `&cell + off` on a reference local is not
// `cell.field` — it is the local's own storage.

const TAG_CLASS: u8 = 1;
const TAG_PRIM: u8 = 2;
const TAG_STRING: u8 = 3;
const TAG_OBJECT: u8 = 4;
const TAG_ARRAY: u8 = 5;
const TAG_OPAQUE: u8 = 6;
const TAG_INTPTR: u8 = 7;
const TAG_POINTER: u8 = 8;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct TypeKey {
    /// Short display name (`TerrainData`, `int`, `String`, `Vector3[]`).
    name: String,
    /// TypeDef index when metadata knows one, else -1.
    tdi: i32,
    tag: u8,
}

fn type_key(ty: &ManagedType) -> Option<TypeKey> {
    let (name, tdi, tag) = match ty {
        ManagedType::Primitive(p) => (p.name().to_string(), -1, TAG_PRIM),
        ManagedType::String => ("String".into(), -1, TAG_STRING),
        ManagedType::Object => ("Object".into(), -1, TAG_OBJECT),
        ManagedType::SzArray(inner) => {
            let elem = type_key(inner).map(|k| k.name).unwrap_or_else(|| "object".into());
            (format!("{elem}[]"), -1, TAG_ARRAY)
        }
        ManagedType::Class(nt) => (nt.name.clone(), nt.type_def_index, TAG_CLASS),
        ManagedType::IntPtr => ("IntPtr".into(), -1, TAG_INTPTR),
        ManagedType::Pointer(_) => ("Pointer".into(), -1, TAG_POINTER),
        ManagedType::Opaque(name) => {
            if name == "T" || name == "M" || name == "generic" {
                return None;
            }
            (name.clone(), -1, TAG_OPAQUE)
        }
        ManagedType::ByRef(inner) => return type_key(inner),
        ManagedType::Unknown => return None,
    };
    Some(TypeKey { name, tdi, tag })
}

fn provable_type(ty: &ManagedType) -> bool {
    match ty {
        ManagedType::Unknown => false,
        ManagedType::Opaque(n) if n == "T" || n == "M" || n == "generic" => false,
        ManagedType::ByRef(_) | ManagedType::Pointer(_) => false,
        _ => true,
    }
}

/// Peel `ref`/`out` wrappers; `None` when the parameter is not by-reference.
fn byref_inner(ty: &ManagedType) -> Option<&ManagedType> {
    match ty {
        ManagedType::ByRef(inner) => Some(inner),
        _ => None,
    }
}

fn camel_type(name: &str) -> Option<String> {
    let base = name.split('`').next().unwrap_or(name);
    let base = base.rsplit('+').next().unwrap_or(base);
    let base = base.trim_start_matches('_');
    if base.is_empty() {
        return None;
    }
    let mut chars = base.chars();
    let first = chars.next()?;
    if !first.is_alphabetic() {
        return None;
    }
    let mut out = String::new();
    out.extend(first.to_lowercase());
    out.push_str(chars.as_str());
    let ok = out
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_')
        && !matches!(
            out.as_str(),
            "abstract" | "as" | "base" | "bool" | "break" | "byte" | "case" | "catch"
                | "char" | "checked" | "class" | "const" | "continue" | "decimal"
                | "default" | "delegate" | "do" | "double" | "else" | "enum" | "event"
                | "explicit" | "extern" | "false" | "finally" | "fixed" | "float"
                | "for" | "foreach" | "goto" | "if" | "implicit" | "in" | "int"
                | "interface" | "internal" | "is" | "lock" | "long" | "namespace"
                | "new" | "null" | "object" | "operator" | "out" | "override"
                | "params" | "private" | "protected" | "public" | "readonly" | "ref"
                | "return" | "sbyte" | "sealed" | "short" | "sizeof" | "stackalloc"
                | "static" | "string" | "struct" | "switch" | "this" | "throw"
                | "true" | "try" | "typeof" | "uint" | "ulong" | "unchecked"
                | "unsafe" | "ushort" | "using" | "virtual" | "void" | "volatile"
                | "while"
        );
    ok.then_some(out)
}

fn cell_name_for(ty: &TypeKey) -> String {
    match ty.tag {
        TAG_PRIM => format!("{}Value", ty.name),
        TAG_STRING => "stringValue".into(),
        TAG_OBJECT => "objectValue".into(),
        TAG_ARRAY => "arrayValue".into(),
        TAG_INTPTR => "intPtrValue".into(),
        TAG_POINTER => "pointerValue".into(),
        _ => camel_type(&ty.name).unwrap_or_else(|| "value".into()),
    }
}

/// A resolved constructor/method signature considered as call-site evidence.
#[derive(Clone, Debug)]
struct CalleeSig {
    decl_name: String,
    decl_tdi: i32,
    name: String,
    is_static: bool,
    params: Vec<ManagedType>,
    ret: ManagedType,
}

#[derive(Default)]
struct CallSigIndex {
    by_name: HashMap<String, Vec<CalleeSig>>,
    by_qualified: HashMap<String, Vec<CalleeSig>>,
    by_addr: HashMap<u64, Vec<CalleeSig>>,
    /// field name → distinct `(declaring type name, tdi)` declarations.
    field_owner: HashMap<String, Vec<(String, i32)>>,
    /// method name → return type when every overload agrees; absent/None ties.
    ret_by_name: HashMap<String, Option<ManagedType>>,
}

fn sig_key(m: &MethodRef) -> String {
    format!(
        "{}|{}|{}|{:?}|{:?}",
        m.type_def_index, m.name, m.sig.is_static, m.sig.params, m.sig.ret
    )
}

fn push_sig(index: &mut CallSigIndex, m: &MethodRef, seen: &mut HashSet<String>) {
    if m.name.is_empty() || m.name.starts_with('\0') {
        return;
    }
    if !seen.insert(sig_key(m)) {
        return;
    }
    let sig = CalleeSig {
        decl_name: m.declaring_type_name.clone(),
        decl_tdi: m.type_def_index,
        name: m.name.clone(),
        is_static: m.sig.is_static,
        params: m.sig.params.clone(),
        ret: m.sig.ret.clone(),
    };
    index.by_name.entry(sig.name.clone()).or_default().push(sig.clone());
    if !sig.decl_name.is_empty() {
        index
            .by_qualified
            .entry(format!("{}.{}", sig.decl_name, sig.name))
            .or_default()
            .push(sig.clone());
        if !m.declaring_namespace.is_empty() {
            index
                .by_qualified
                .entry(format!(
                    "{}.{}.{}",
                    m.declaring_namespace, sig.decl_name, sig.name
                ))
                .or_default()
                .push(sig.clone());
        }
    }
    // Return-type agreement per method name: a name with two different return
    // types cannot type a store.
    match index.ret_by_name.get(&sig.name) {
        None => {
            index.ret_by_name.insert(sig.name.clone(), Some(sig.ret.clone()));
        }
        Some(Some(prev)) if *prev != sig.ret => {
            index.ret_by_name.insert(sig.name.clone(), None);
        }
        _ => {}
    }
}

impl CallSigIndex {
    fn build(maps: &ResolveMaps) -> Self {
        let mut index = CallSigIndex::default();
        let mut seen: HashSet<String> = HashSet::new();
        for m in maps.methods_by_rva.values() {
            push_sig(&mut index, m, &mut seen);
        }
        for ms in maps.shared_by_rva.values() {
            for m in ms {
                push_sig(&mut index, m, &mut seen);
            }
        }
        for ms in maps.specs_by_body_rva.values() {
            for (_, m) in ms {
                push_sig(&mut index, m, &mut seen);
            }
        }
        for m in maps.specs_by_index.values() {
            push_sig(&mut index, m, &mut seen);
        }
        for m in maps.vtable_slots.values() {
            push_sig(&mut index, m, &mut seen);
        }
        for ms in maps.native_wrappers.values() {
            for m in ms {
                push_sig(&mut index, m, &mut seen);
            }
        }
        // Address-keyed view for `RawCall` targets that metadata already named.
        for (rva, m) in &maps.methods_by_rva {
            if let Some(sig) = index
                .by_name
                .get(&m.name)
                .and_then(|v| v.iter().find(|s| s.decl_tdi == m.type_def_index && s.name == m.name))
            {
                index.by_addr.entry(*rva).or_default().push(sig.clone());
            }
        }
        for (rva, ms) in &maps.shared_by_rva {
            for m in ms {
                if let Some(sig) = index
                    .by_name
                    .get(&m.name)
                    .and_then(|v| v.iter().find(|s| s.decl_tdi == m.type_def_index && s.name == m.name))
                {
                    index.by_addr.entry(*rva).or_default().push(sig.clone());
                }
            }
        }
        // Field-name ownership, used only when a name is declared by exactly
        // one type (common names such as `Length`/`Count` abstain).
        for ((tdi, _off), (fname, decl)) in &maps.field_layouts {
            if fname.is_empty() {
                continue;
            }
            let list = index.field_owner.entry(fname.clone()).or_default();
            if !list.iter().any(|(d, t)| d == decl && t == tdi) {
                list.push((decl.clone(), *tdi));
            }
        }
        index
    }
}

/// Whole-program `value type` set, cached per `ResolveMaps` instance.
///
/// Building it from every `field_layouts` key on EVERY method was the single
/// largest per-method cost in this pass (10^5 map entries × 24k methods). The
/// key includes two map sizes so a recycled address cannot alias an older
/// map's set.
static VALUE_TYPES_CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), Arc<HashSet<i32>>>>> =
    OnceLock::new();

fn value_types(maps: &ResolveMaps) -> Arc<HashSet<i32>> {
    let key = (
        maps as *const ResolveMaps as usize,
        maps.field_layouts.len(),
        maps.tdi_names.len(),
    );
    let cache = VALUE_TYPES_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(v) = guard.get(&key) {
        return v.clone();
    }
    // A field below the 16-byte object header can only belong to a value
    // type. That is the metadata-backed proof that makes `cell.field` legal
    // for `(&cell + off)`.
    let set: HashSet<i32> = maps
        .field_layouts
        .keys()
        .filter(|(_, off)| (0..16).contains(off))
        .map(|(tdi, _)| *tdi)
        .collect();
    let arc = Arc::new(set);
    guard.insert(key, arc.clone());
    arc
}

static CALL_INDEX_CACHE: OnceLock<Mutex<HashMap<usize, Arc<CallSigIndex>>>> = OnceLock::new();

fn call_index(maps: &ResolveMaps) -> Arc<CallSigIndex> {
    let key = maps as *const ResolveMaps as usize;
    let cache = CALL_INDEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(idx) = guard.get(&key) {
        return idx.clone();
    }
    let idx = Arc::new(CallSigIndex::build(maps));
    guard.insert(key, idx.clone());
    idx
}

/// Everything the typing rules need from the pipeline.
struct CellCtx<'a> {
    index: &'a CallSigIndex,
    maps: &'a ResolveMaps,
    local_types: &'a HashMap<String, i32>,
    local_mty: &'a HashMap<String, ManagedType>,
    declaring_tdi: i32,
    value_types: Arc<HashSet<i32>>,
}

impl<'a> CellCtx<'a> {
    fn new(
        index: &'a CallSigIndex,
        maps: &'a ResolveMaps,
        local_types: &'a HashMap<String, i32>,
        local_mty: &'a HashMap<String, ManagedType>,
        declaring_tdi: i32,
    ) -> Self {
        Self {
            index,
            maps,
            local_types,
            local_mty,
            declaring_tdi,
            value_types: value_types(maps),
        }
    }

    fn is_value_type(&self, tdi: i32) -> bool {
        self.value_types.contains(&tdi)
    }

    fn type_name(&self, tdi: i32) -> Option<&str> {
        self.maps.tdi_names.get(&tdi).map(|(_, n)| n.as_str())
    }
}

/// Build the field-sugar plan from metadata layouts. `field_plan` maps
/// `(cell, byte offset, normalized width)` to the proven field name.
fn collect_field_plan(
    evidence: &Evidence,
    cell: &str,
    tdi: i32,
    ctx: &CellCtx<'_>,
    out: &mut HashMap<(String, i64, u8), String>,
) {
    let mut chain: Vec<i32> = ctx
        .maps
        .type_parent_chains
        .get(&tdi)
        .cloned()
        .unwrap_or_else(|| vec![tdi]);
    if !chain.contains(&tdi) {
        chain.push(tdi);
    }
    let mut keys: Vec<(i64, u8)> = evidence
        .interior
        .iter()
        .filter(|(c, _, _)| c == cell)
        .map(|(_, off, w)| (*off, normalize_width(*w)))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    for (off, width) in keys {
        let Some((fname, ftdi)) = chain.iter().find_map(|p| {
            ctx.maps
                .field_layouts
                .get(&(*p, off))
                .map(|(n, _)| (n.clone(), ctx.maps.field_types.get(&(*p, off)).copied()))
        }) else {
            continue;
        };
        let Some(ftdi) = ftdi else { continue };
        if !field_width_matches(ftdi, width, ctx) {
            continue;
        }
        out.insert((cell.to_string(), off, width), fname);
    }
}

fn normalize_width(w: u8) -> u8 {
    if w == 0 {
        8
    } else {
        w
    }
}

fn field_width_matches(ftdi: i32, access_width: u8, ctx: &CellCtx<'_>) -> bool {
    let Some(name) = ctx.type_name(ftdi) else {
        return false;
    };
    let name = name.trim();
    let expected = match name {
        "Boolean" | "Byte" | "SByte" => 1,
        "Char" | "Int16" | "UInt16" => 2,
        "Int32" | "UInt32" | "Single" => 4,
        "Int64" | "UInt64" | "Double" | "IntPtr" | "UIntPtr" => 8,
        "String" | "Object" => 8,
        n if n.ends_with("[]") => 8,
        _ => {
            // A field whose type has fields below the object header is a
            // struct value; its width is not derivable here. Otherwise the
            // only safe reading of an 8-byte access is a reference.
            if ctx.is_value_type(ftdi) {
                return false;
            }
            8
        }
    };
    access_width == expected
}

#[derive(Default)]
struct Evidence {
    /// Address arguments: `(cell, offset, callee key, has_recv, nargs, argidx)`.
    address_args: Vec<AddrArg>,
    receiver_methods: Vec<(String, String)>,
    field_reads: Vec<(String, String)>,
    /// `cell = other` copies (other may be a cell or an ordinary local).
    copies: Vec<(String, String)>,
    /// `cell = rhs` whole-value stores, kept for RHS type inference.
    defs: Vec<(String, Expr)>,
    /// Raw loads/stores into a cell: `(cell, offset, width)`.
    interior: Vec<(String, i64, u8)>,
    /// Raw STORES only (assignment targets), for read-only detection.
    raw_stores: HashSet<(String, i64, u8)>,
    /// Type forced by the assignment target's cast (`*(int*)` → int).
    store_kinds: Vec<(String, i64, TypeKey)>,
    /// Every `&cell` / `RawAddr(cell, …)` occurrence (address escape).
    address_uses: HashSet<String>,
    /// Cells used as arrays: `.Length` or `cell[index]`.
    array_cells: HashSet<String>,
    /// Every frame cell name seen anywhere in the method.
    seen_cells: HashSet<String>,
}

struct AddrArg {
    cell: String,
    off: i64,
    name: String,
    addr: Option<u64>,
    has_recv: bool,
    nargs: usize,
    idx: usize,
}

fn is_stack_name(n: &str) -> bool {
    is_stack_cell(n)
}

/// `&cell`, `(nint*)(cell + off)`, `RawAddr(cell, off)` → `(cell, off)`.
fn addr_expr_cell(e: &Expr) -> Option<(String, i64)> {
    match e {
        Expr::AddressOf(inner) => cell_plus_offset(inner),
        Expr::RawAddr(inner, addend) => {
            let (cell, off) = cell_plus_offset(inner)?;
            Some((cell, off + *addend))
        }
        Expr::Cast(_, inner) => addr_expr_cell(inner),
        Expr::Bin(BinOp::Add, _, _) => cell_plus_offset(e),
        _ => None,
    }
}

fn cell_plus_offset(e: &Expr) -> Option<(String, i64)> {
    match e {
        Expr::Var(n) if is_stack_name(n) => Some((n.clone(), 0)),
        Expr::Cast(_, inner) => cell_plus_offset(inner),
        Expr::Bin(BinOp::Add, a, b) => match (a.as_ref(), b.as_ref()) {
            (Expr::Var(n), Expr::Int(k)) if is_stack_name(n) => Some((n.clone(), *k)),
            (Expr::Int(k), Expr::Var(n)) if is_stack_name(n) => Some((n.clone(), *k)),
            _ => None,
        },
        _ => None,
    }
}

/// A cell used as a call receiver (`cell.Method` / `(&cell).Method`).
fn receiver_cell(e: &Expr) -> Option<(String, i64)> {
    match e {
        Expr::Var(n) if is_stack_name(n) => Some((n.clone(), 0)),
        Expr::AddressOf(inner) => cell_plus_offset(inner),
        Expr::Cast(_, inner) => receiver_cell(inner),
        Expr::RawAddr(inner, addend) => {
            let (c, o) = cell_plus_offset(inner)?;
            Some((c, o + *addend))
        }
        _ => None,
    }
}

fn collect_evidence(stmts: &[Stmt], ev: &mut Evidence) {
    for s in stmts {
        if let Stmt::Assign(target, _, rhs) = s {
            if is_stack_name(target) {
                ev.seen_cells.insert(target.clone());
                ev.defs.push((target.clone(), rhs.clone()));
                if let Expr::Var(src) = rhs {
                    if is_stack_name(src) {
                        ev.seen_cells.insert(src.clone());
                    }
                    ev.copies.push((target.clone(), src.clone()));
                }
            } else if let Some((cell, off, width)) = parse_raw_target(target) {
                ev.seen_cells.insert(cell.clone());
                ev.raw_stores.insert((cell.clone(), off, width));
                if let Some(ty) = raw_store_type(target) {
                    ev.store_kinds.push((cell.clone(), off, ty));
                }
                ev.interior.push((cell, off, width));
            }
        }
        for e in stmt_exprs(s) {
            collect_evidence_expr(e, ev);
        }
        for k in children_lists(s) {
            collect_evidence(k, ev);
        }
    }
}

fn collect_evidence_expr(e: &Expr, ev: &mut Evidence) {
    if let Expr::Var(n) = e {
        if is_stack_name(n) {
            ev.seen_cells.insert(n.clone());
        }
    }
    match e {
        Expr::Call { recv, name, args } => {
            if let Some(r) = recv {
                if let Some((cell, 0)) = receiver_cell(r) {
                    ev.receiver_methods.push((cell, name.clone()));
                }
            }
            for (i, a) in args.iter().enumerate() {
                if let Some((cell, off)) = addr_expr_cell(a) {
                    ev.address_args.push(AddrArg {
                        cell,
                        off,
                        name: name.clone(),
                        addr: None,
                        has_recv: recv.is_some(),
                        nargs: args.len(),
                        idx: i,
                    });
                }
            }
        }
        Expr::RawCall(addr, args, _) => {
            for (i, a) in args.iter().enumerate() {
                if let Some((cell, off)) = addr_expr_cell(a) {
                    ev.address_args.push(AddrArg {
                        cell,
                        off,
                        name: String::new(),
                        addr: Some(*addr),
                        has_recv: false,
                        nargs: args.len(),
                        idx: i,
                    });
                }
            }
        }
        Expr::Field(base, fname) => {
            if let Some((cell, 0)) = receiver_cell(base) {
                if fname == "Length" {
                    ev.array_cells.insert(cell.clone());
                }
                ev.field_reads.push((cell, fname.clone()));
            }
        }
        Expr::Index { collection, .. } => {
            if let Expr::Var(cell) = collection.as_ref() {
                if is_stack_name(cell) {
                    ev.array_cells.insert(cell.clone());
                }
            }
        }
        Expr::AddressOf(inner) => {
            if let Some((cell, _)) = cell_plus_offset(inner) {
                ev.address_uses.insert(cell);
            }
        }
        Expr::RawAddr(inner, _) => {
            if let Some((cell, _)) = cell_plus_offset(inner) {
                ev.address_uses.insert(cell);
            }
        }
        Expr::RawDeref(base, off, width) => {
            if let Some((cell, base_off)) = addr_expr_cell(base) {
                ev.interior.push((cell, base_off + *off, *width));
            }
        }
        _ => {}
    }
    for c in expr_children(e) {
        collect_evidence_expr(c, ev);
    }
}

#[derive(Clone, Debug)]
struct Vote {
    ty: TypeKey,
    weight: u32,
}

fn add_vote(
    votes: &mut HashMap<String, Vec<Vote>>,
    cell: &str,
    ty: TypeKey,
    weight: u32,
) {
    let list = votes.entry(cell.to_string()).or_default();
    if let Some(v) = list.iter_mut().find(|v| v.ty == ty) {
        v.weight = v.weight.max(weight);
    } else {
        list.push(Vote { ty, weight });
    }
}

/// The heaviest non-tied type. A tie between two distinct types abstains: the
/// cell is reused and no single name/type is true for all accesses.
fn unique_vote(votes: &HashMap<String, Vec<Vote>>, cell: &str) -> Option<TypeKey> {
    let list = votes.get(cell)?;
    let max = list.iter().map(|v| v.weight).max()?;
    let mut winners = list.iter().filter(|v| v.weight == max);
    let first = winners.next()?;
    if winners.any(|v| v.ty != first.ty) {
        return None;
    }
    Some(first.ty.clone())
}

fn call_bucket<'a>(
    index: &'a CallSigIndex,
    name: &str,
) -> Option<&'a Vec<CalleeSig>> {
    if let Some(v) = index.by_qualified.get(name) {
        return Some(v);
    }
    if let Some(short) = name.rsplit('.').next() {
        if let Some(v) = index.by_qualified.get(short) {
            return Some(v);
        }
        if let Some(v) = index.by_name.get(short) {
            return Some(v);
        }
    }
    index.by_name.get(name)
}

/// Type proven for one address argument, or `None` when any candidate
/// signature disagrees / is not by-reference.
fn addr_arg_type(arg: &AddrArg, ctx: &CellCtx<'_>) -> Option<TypeKey> {
    let candidates: &[CalleeSig] = match arg.addr {
        Some(addr) => ctx.index.by_addr.get(&addr)?,
        None => call_bucket(ctx.index, &arg.name)?,
    };
    let matching: Vec<&CalleeSig> = candidates
        .iter()
        .filter(|s| s.is_static == !arg.has_recv)
        .filter(|s| s.params.len() == arg.nargs)
        .collect();
    if matching.is_empty() {
        return None;
    }
    let mut out: Option<TypeKey> = None;
    for sig in matching {
        let param = sig.params.get(arg.idx)?;
        let inner = byref_inner(param)?;
        if !provable_type(inner) {
            return None;
        }
        let key = type_key(inner)?;
        match &out {
            None => out = Some(key),
            Some(prev) if *prev == key => {}
            Some(_) => return None,
        }
    }
    out
}

fn receiver_method_type(cell: &str, name: &str, ctx: &CellCtx<'_>) -> Option<TypeKey> {
    let short = name.rsplit('.').next().unwrap_or(name);
    let bucket = ctx.index.by_name.get(short)?;
    let mut out: Option<TypeKey> = None;
    let mut any = false;
    for sig in bucket.iter().filter(|s| !s.is_static) {
        any = true;
        if sig.decl_name.is_empty() {
            return None;
        }
        let key = TypeKey {
            name: sig.decl_name.clone(),
            tdi: sig.decl_tdi,
            tag: TAG_CLASS,
        };
        match &out {
            None => out = Some(key),
            Some(prev) if *prev == key => {}
            Some(_) => return None,
        }
    }
    let _ = cell;
    any.then_some(out).flatten()
}

fn field_name_type(name: &str, ctx: &CellCtx<'_>) -> Option<TypeKey> {
    let owners = ctx.index.field_owner.get(name)?;
    let mut out: Option<&(String, i32)> = None;
    for owner in owners {
        match out {
            None => out = Some(owner),
            Some(prev) if prev.0 == owner.0 => {}
            Some(_) => return None,
        }
    }
    let (decl, tdi) = out?;
    Some(TypeKey {
        name: decl.rsplit('.').next().unwrap_or(decl).to_string(),
        tdi: *tdi,
        tag: TAG_CLASS,
    })
}

fn type_from_local(name: &str, ctx: &CellCtx<'_>) -> Option<TypeKey> {
    if let Some(tdi) = ctx.local_types.get(name) {
        let display = ctx.type_name(*tdi)?;
        return Some(TypeKey {
            name: display.to_string(),
            tdi: *tdi,
            tag: TAG_CLASS,
        });
    }
    ctx.local_mty.get(name).and_then(type_key)
}

fn type_from_name(name: &str, ctx: &CellCtx<'_>) -> Option<TypeKey> {
    let short = name.rsplit('.').next().unwrap_or(name);
    if let Some(tdi) = ctx.maps.tdi_names_reverse.get(name).or_else(|| {
        ctx.maps
            .tdi_names_reverse
            .get(name.rsplit('.').next().unwrap_or(name))
    }) {
        return Some(TypeKey {
            name: short.to_string(),
            tdi: *tdi,
            tag: TAG_CLASS,
        });
    }
    let ty = match short {
        "Boolean" | "bool" => ManagedType::Primitive(PrimKind::Boolean),
        "Byte" | "byte" => ManagedType::Primitive(PrimKind::U1),
        "SByte" | "sbyte" => ManagedType::Primitive(PrimKind::I1),
        "Int16" | "short" => ManagedType::Primitive(PrimKind::I2),
        "UInt16" | "ushort" => ManagedType::Primitive(PrimKind::U2),
        "Int32" | "int" => ManagedType::Primitive(PrimKind::I4),
        "UInt32" | "uint" => ManagedType::Primitive(PrimKind::U4),
        "Int64" | "long" => ManagedType::Primitive(PrimKind::I8),
        "UInt64" | "ulong" => ManagedType::Primitive(PrimKind::U8),
        "Single" | "float" => ManagedType::Primitive(PrimKind::R4),
        "Double" | "double" => ManagedType::Primitive(PrimKind::R8),
        "Char" | "char" => ManagedType::Primitive(PrimKind::Char),
        "IntPtr" | "nint" => ManagedType::IntPtr,
        "String" | "string" => ManagedType::String,
        "Object" | "object" => ManagedType::Object,
        _ => return None,
    };
    type_key(&ty)
}

fn field_type_by_name_chain(ctx: &CellCtx<'_>, tdi: i32, name: &str) -> Option<i32> {
    if let Some(t) = ctx.maps.field_type_by_name.get(&(tdi, name.to_string())) {
        return Some(*t);
    }
    // Inherited fields live on a parent's layout.
    if let Some(chain) = ctx.maps.type_parent_chains.get(&tdi) {
        for p in chain {
            if let Some(t) = ctx.maps.field_type_by_name.get(&(*p, name.to_string())) {
                return Some(*t);
            }
        }
    }
    None
}

/// Structural RHS typing. Only expressions whose type is forced by the AST
/// shape are accepted; anything else abstains.
fn infer_expr_type(
    e: &Expr,
    ctx: &CellCtx<'_>,
    votes: &HashMap<String, Vec<Vote>>,
) -> Option<TypeKey> {
    match e {
        Expr::Var(n) if is_stack_name(n) => unique_vote(votes, n),
        Expr::Var(n) => type_from_local(n, ctx),
        Expr::Cast(t, _) => type_from_name(t, ctx),
        Expr::Float(_) => type_key(&ManagedType::Primitive(PrimKind::R4)),
        Expr::Double(_) => type_key(&ManagedType::Primitive(PrimKind::R8)),
        Expr::Bool(_) => type_key(&ManagedType::Primitive(PrimKind::Boolean)),
        Expr::Str(_) => type_key(&ManagedType::String),
        Expr::Cmp(..) => type_key(&ManagedType::Primitive(PrimKind::Boolean)),
        Expr::Not(_) => Some(TypeKey {
            name: "bool".into(),
            tdi: -1,
            tag: TAG_PRIM,
        }),
        Expr::Neg(inner) => infer_expr_type(inner, ctx, votes),
        Expr::Bin(op, a, b) => match op {
            BinOp::AndAlso | BinOp::OrElse => Some(TypeKey {
                name: "bool".into(),
                tdi: -1,
                tag: TAG_PRIM,
            }),
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
            | BinOp::And | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::Shr => {
                let ta = infer_expr_type(a, ctx, votes)?;
                let tb = infer_expr_type(b, ctx, votes)?;
                (ta == tb).then_some(ta)
            }
            _ => None,
        },
        Expr::Cond(_, a, b) => {
            let ta = infer_expr_type(a, ctx, votes)?;
            let tb = infer_expr_type(b, ctx, votes)?;
            (ta == tb).then_some(ta)
        }
        Expr::NewObj(t, _) => type_from_name(t, ctx),
        Expr::Call { name, .. } => ctx
            .index
            .ret_by_name
            .get(name.rsplit('.').next().unwrap_or(name))
            .cloned()
            .flatten()
            .and_then(|t| type_key(&t)),
        Expr::Field(base, f) => {
            let base_tdi = match base.as_ref() {
                Expr::This => Some(ctx.declaring_tdi),
                Expr::Var(n) => {
                    if is_stack_name(n) {
                        unique_vote(votes, n).and_then(|k| (k.tdi >= 0).then_some(k.tdi))
                    } else {
                        ctx.local_types.get(n).copied()
                    }
                }
                _ => None,
            }?;
            let ftdi = field_type_by_name_chain(ctx, base_tdi, f)?;
            let name = ctx.type_name(ftdi)?.to_string();
            Some(TypeKey {
                name,
                tdi: ftdi,
                tag: TAG_CLASS,
            })
        }
        _ => None,
    }
}

fn infer_cells(
    stmts: &[Stmt],
    ctx: &CellCtx<'_>,
) -> (HashMap<String, TypeKey>, Evidence) {
    let mut ev = Evidence::default();
    collect_evidence(stmts, &mut ev);
    let mut votes: HashMap<String, Vec<Vote>> = HashMap::new();
    let dbg = crate::recovery::env_flags::flag("ILR_CELLDBG");
    for arg in &ev.address_args {
        let got = addr_arg_type(arg, ctx);
        if dbg {
            eprintln!(
                "[cell] argv cell={} off={} call={} addr={:?} recv={} nargs={} idx={} -> {:?}",
                arg.cell, arg.off, arg.name, arg.addr, arg.has_recv, arg.nargs, arg.idx, got
            );
        }
        if let Some(key) = got {
            add_vote(&mut votes, &arg.cell, key, 4);
        }
    }
    for (cell, name) in &ev.receiver_methods {
        if let Some(key) = receiver_method_type(cell, name, ctx) {
            if dbg {
                eprintln!("[cell] recv cell={cell} method={name} -> {key:?}");
            }
            add_vote(&mut votes, cell, key, 2);
        }
    }
    for (cell, fname) in &ev.field_reads {
        if let Some(key) = field_name_type(fname, ctx) {
            if dbg {
                eprintln!("[cell] field cell={cell} name={fname} -> {key:?}");
            }
            add_vote(&mut votes, cell, key, 2);
        }
    }
    // `.Length` / `cell[index]` is the exact managed array shape produced by
    // `layout_sugar` and the renderer; the cell holds a managed array
    // reference. Structural, not name-guessed: both shapes are only emitted
    // for array element arithmetic.
    for cell in &ev.array_cells {
        add_vote(
            &mut votes,
            cell,
            TypeKey {
                name: "Array".into(),
                tdi: -1,
                tag: TAG_ARRAY,
            },
            4,
        );
    }
    // A store's cast is a type statement: `*(int*)(&cell +4) = v` writes an
    // int. Several stores of different primitive kinds at one cell tie and
    // abstain (a reused/aggregated cell), while a metadata signature still
    // outweighs this structural evidence.
    for (cell, _off, ty) in &ev.store_kinds {
        add_vote(&mut votes, cell, ty.clone(), 2);
    }
    // Copy / RHS propagation to a fixpoint. Copies between cells need the
    // source's vote, so a few rounds are required for chains; the bound
    // keeps a cyclic copy ring from spinning.
    let mut changed = true;
    let mut rounds = 0;
    while changed && rounds < 8 {
        changed = false;
        rounds += 1;
        for (dst, src) in &ev.copies {
            let src_ty = if is_stack_name(src) {
                unique_vote(&votes, src)
            } else {
                type_from_local(src, ctx)
            };
            if let Some(ty) = src_ty {
                if dbg {
                    eprintln!("[cell] copy dst={dst} src={src} -> {ty:?}");
                }
                let before = votes.get(dst).map(|v| v.len()).unwrap_or(0);
                add_vote(&mut votes, dst, ty, 1);
                if votes.get(dst).map(|v| v.len()).unwrap_or(0) != before {
                    changed = true;
                }
            }
        }
        for (cell, rhs) in &ev.defs {
            if let Some(ty) = infer_expr_type(rhs, ctx, &votes) {
                if dbg {
                    eprintln!("[cell] def cell={cell} rhs={rhs:?} -> {ty:?}");
                }
                let before = votes.get(cell).map(|v| v.len()).unwrap_or(0);
                let weight = if matches!(rhs, Expr::Var(_)) { 1 } else { 2 };
                add_vote(&mut votes, cell, ty, weight);
                if votes.get(cell).map(|v| v.len()).unwrap_or(0) != before {
                    changed = true;
                }
            }
        }
    }
    let mut out = HashMap::new();
    for (cell, list) in &votes {
        if dbg {
            eprintln!("[cell] votes {cell}: {:?}", list);
        }
    }
    for cell in votes.keys() {
        if let Some(ty) = unique_vote(&votes, cell) {
            out.insert(cell.clone(), ty);
        }
    }
    (out, ev)
}

/// Production entry point: type cells from metadata signatures/layouts.
///
/// Returns the number of cell occurrences rewritten (renames + field sugar).
pub fn apply_with_maps(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    local_mty: &HashMap<String, ManagedType>,
    declaring_tdi: i32,
) -> usize {
    let index = call_index(maps);
    let ctx = CellCtx::new(&index, maps, local_types, local_mty, declaring_tdi);
    apply_cells(stmts, Some(&ctx))
}

/// A use-region type proven from the immediate syntactic context of ONE read.
/// Used only for read-only cells (no store, no address escape), where two
/// non-overlapping typings can be split by occurrence without moving a value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum UseKind {
    Str,
    Int,
    Float,
    #[allow(dead_code)]
    Bool,
}



const STRING_METHODS: &[&str] = &[
    "StartsWith",
    "EndsWith",
    "Substring",
    "Trim",
    "TrimStart",
    "TrimEnd",
    "ToUpper",
    "ToLower",
    "Split",
    "IndexOfAny",
    "PadLeft",
    "PadRight",
    "Replace",
    "Contains",
];

/// True when the method name is string-ONLY (`Contains` also exists on
/// collections, so it is accepted only for cells with another string proof).
fn strong_string_method(name: &str) -> bool {
    STRING_METHODS.contains(&name) && name != "Contains"
}

enum Ctx<'a> {
    None,
    CallRecv(&'a str),
    CallArg(&'a str, usize),
    Index,
    CmpOther(&'a Expr),
    BinOther(BinOp, &'a Expr),
}

fn expr_is_float(e: &Expr, ctx: &CellCtx<'_>) -> bool {
    match e {
        Expr::Float(_) | Expr::Double(_) => true,
        Expr::Var(n) => matches!(
            ctx.local_mty.get(n),
            Some(ManagedType::Primitive(PrimKind::R4 | PrimKind::R8))
        ),
        Expr::Cast(t, _) => matches!(t.as_str(), "float" | "double" | "Single" | "Double"),
        Expr::Neg(inner) => expr_is_float(inner, ctx),
        Expr::Bin(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, a, b) => {
            expr_is_float(a, ctx) || expr_is_float(b, ctx)
        }
        _ => false,
    }
}

fn expr_is_int(e: &Expr, ctx: &CellCtx<'_>) -> bool {
    match e {
        Expr::Int(_) => true,
        Expr::Var(n) => matches!(
            ctx.local_mty.get(n),
            Some(ManagedType::Primitive(
                PrimKind::I1
                    | PrimKind::U1
                    | PrimKind::I2
                    | PrimKind::U2
                    | PrimKind::I4
                    | PrimKind::U4
                    | PrimKind::I8
                    | PrimKind::U8
                    | PrimKind::Char
            ))
        ),
        Expr::Cast(t, _) => matches!(
            t.as_str(),
            "int" | "uint" | "long" | "ulong" | "short" | "ushort" | "byte" | "sbyte" | "char"
        ),
        Expr::Neg(inner) => expr_is_int(inner, ctx),
        Expr::Bin(_, a, b) => expr_is_int(a, ctx) && expr_is_int(b, ctx),
        _ => false,
    }
}

fn classify_use(cell: &str, ctx: &Ctx<'_>, strong_string: &HashSet<String>, cctx: &CellCtx<'_>) -> Option<UseKind> {
    match ctx {
        Ctx::CallRecv(name) => {
            let short = name.rsplit('.').next().unwrap_or(name);
            if strong_string_method(short)
                || (short == "Contains" && strong_string.contains(cell))
            {
                Some(UseKind::Str)
            } else {
                None
            }
        }
        Ctx::CallArg(name, idx) => {
            let short = name.rsplit('.').next().unwrap_or(name);
            if *idx == 0 && (short == "IsNullOrEmpty" || short == "IsNullOrWhiteSpace") {
                Some(UseKind::Str)
            } else {
                None
            }
        }
        Ctx::Index => Some(UseKind::Int),
        Ctx::CmpOther(other) => {
            if expr_is_float(other, cctx) {
                Some(UseKind::Float)
            } else if expr_is_int(other, cctx) {
                Some(UseKind::Int)
            } else {
                None
            }
        }
        Ctx::BinOther(op, other) => {
            if expr_is_float(other, cctx)
                || matches!(op, BinOp::Mul | BinOp::Div)
                    && expr_is_float(other, cctx)
            {
                Some(UseKind::Float)
            } else if expr_is_int(other, cctx)
                || matches!(op, BinOp::Shl | BinOp::Shr | BinOp::And | BinOp::Or | BinOp::Xor)
            {
                Some(UseKind::Int)
            } else {
                None
            }
        }
        Ctx::None => None,
    }
}

fn walk_uses(
    e: &Expr,
    ctx: &Ctx<'_>,
    cands: &HashSet<String>,
    strong_string: &HashSet<String>,
    cctx: &CellCtx<'_>,
    out: &mut HashMap<String, HashSet<UseKind>>,
) {
    if let Expr::Var(n) = e {
        if cands.contains(n) {
            if let Some(kind) = classify_use(n, ctx, strong_string, cctx) {
                out.entry(n.clone()).or_default().insert(kind);
            }
        }
    }
    match e {
        Expr::Call { recv, name, args } => {
            if let Some(r) = recv {
                walk_uses(r, &Ctx::CallRecv(name), cands, strong_string, cctx, out);
            }
            for (i, a) in args.iter().enumerate() {
                walk_uses(a, &Ctx::CallArg(name, i), cands, strong_string, cctx, out);
            }
        }
        Expr::Index { collection, index } => {
            walk_uses(collection, &Ctx::None, cands, strong_string, cctx, out);
            walk_uses(index, &Ctx::Index, cands, strong_string, cctx, out);
        }
        Expr::Cmp(_, a, b) => {
            walk_uses(a, &Ctx::CmpOther(b), cands, strong_string, cctx, out);
            walk_uses(b, &Ctx::CmpOther(a), cands, strong_string, cctx, out);
        }
        Expr::Bin(op, a, b) => {
            walk_uses(a, &Ctx::BinOther(*op, b), cands, strong_string, cctx, out);
            walk_uses(b, &Ctx::BinOther(*op, a), cands, strong_string, cctx, out);
        }
        _ => {
            for c in expr_children(e) {
                walk_uses(c, &Ctx::None, cands, strong_string, cctx, out);
            }
        }
    }
}

fn collect_use_kinds(
    stmts: &[Stmt],
    cands: &HashSet<String>,
    cctx: &CellCtx<'_>,
) -> (HashMap<String, HashSet<UseKind>>, HashSet<String>) {
    // First pass: cells with a string-only method prove `Contains` is the
    // string overload.
    let mut strong_string: HashSet<String> = HashSet::new();
    fn find_strong(e: &Expr, cands: &HashSet<String>, out: &mut HashSet<String>) {
        if let Expr::Call { recv, name, .. } = e {
            if let Some(r) = recv {
                if let Expr::Var(cell) = r.as_ref() {
                    if cands.contains(cell)
                        && strong_string_method(name.rsplit('.').next().unwrap_or(name))
                    {
                        out.insert(cell.clone());
                    }
                }
            }
        }
        for c in expr_children(e) {
            find_strong(c, cands, out);
        }
    }
    // Rust closures cannot recurse; the helper above covers one level, so do
    // the recursion explicitly.
    fn scan_strong(stmts: &[Stmt], cands: &HashSet<String>, out: &mut HashSet<String>) {
        for s in stmts {
            for e in stmt_exprs(s) {
                find_strong(e, cands, out);
            }
            for k in children_lists(s) {
                scan_strong(k, cands, out);
            }
        }
    }
    scan_strong(stmts, cands, &mut strong_string);
    let mut out: HashMap<String, HashSet<UseKind>> = HashMap::new();
    fn scan(
        stmts: &[Stmt],
        cands: &HashSet<String>,
        strong: &HashSet<String>,
        cctx: &CellCtx<'_>,
        out: &mut HashMap<String, HashSet<UseKind>>,
    ) {
        for s in stmts {
            for e in stmt_exprs(s) {
                walk_uses(e, &Ctx::None, cands, strong, cctx, out);
            }
            for k in children_lists(s) {
                scan(k, cands, strong, cctx, out);
            }
        }
    }
    scan(stmts, cands, &strong_string, cctx, &mut out);
    (out, strong_string)
}

fn allocate_split_names(
    stmts: &[Stmt],
    kinds: &HashMap<String, HashSet<UseKind>>,
    whole: &HashMap<String, String>,
) -> HashMap<(String, UseKind), String> {
    let mut used: HashSet<String> = HashSet::new();
    for s in stmts.iter() {
        collect_existing_names(s, &mut used);
    }
    for name in whole.values() {
        used.insert(name.clone());
    }
    let mut entries: Vec<(&String, &HashSet<UseKind>)> = kinds.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = HashMap::new();
    for (cell, set) in entries {
        let mut sorted: Vec<UseKind> = set.iter().copied().collect();
        sorted.sort_by_key(|k| format!("{k:?}"));
        for kind in sorted {
            let base = match kind {
                UseKind::Str => "stringValue",
                UseKind::Int => "intValue",
                UseKind::Float => "floatValue",
                UseKind::Bool => "boolValue",
            };
            let mut candidate = base.to_string();
            let mut k = 2usize;
            while used.contains(&candidate) {
                candidate = format!("{base}{k}");
                k += 1;
            }
            used.insert(candidate.clone());
            out.insert((cell.clone(), kind), candidate);
        }
    }
    out
}

fn split_uses_expr(
    e: &mut Expr,
    ctx: &Ctx<'_>,
    kinds: &HashMap<(String, UseKind), String>,
    strong_string: &HashSet<String>,
    cctx: &CellCtx<'_>,
) -> usize {
    let mut n = 0;
    if let Expr::Var(name) = e {
        if let Some(kind) = classify_use(name, ctx, strong_string, cctx) {
            if let Some(repl) = kinds.get(&(name.clone(), kind)) {
                *name = repl.clone();
                return 1;
            }
        }
    }
    match e {
        Expr::Call { recv, name, args } => {
            if let Some(r) = recv {
                let cname = name.clone();
                n += split_uses_expr(r, &Ctx::CallRecv(&cname), kinds, strong_string, cctx);
            }
            for (i, a) in args.iter_mut().enumerate() {
                let cname = name.clone();
                n += split_uses_expr(a, &Ctx::CallArg(&cname, i), kinds, strong_string, cctx);
            }
        }
        Expr::Index { collection, index } => {
            n += split_uses_expr(collection, &Ctx::None, kinds, strong_string, cctx);
            n += split_uses_expr(index, &Ctx::Index, kinds, strong_string, cctx);
        }
        Expr::Cmp(_, a, b) => {
            // Borrow both sides independently: split the left against the
            // right's ORIGINAL shape, then the right against the left's.
            let b_clone = b.as_ref().clone();
            n += split_uses_expr(a, &Ctx::CmpOther(&b_clone), kinds, strong_string, cctx);
            let a_clone = a.as_ref().clone();
            n += split_uses_expr(b, &Ctx::CmpOther(&a_clone), kinds, strong_string, cctx);
        }
        Expr::Bin(op, a, b) => {
            let b_clone = b.as_ref().clone();
            n += split_uses_expr(a, &Ctx::BinOther(*op, &b_clone), kinds, strong_string, cctx);
            let a_clone = a.as_ref().clone();
            n += split_uses_expr(b, &Ctx::BinOther(*op, &a_clone), kinds, strong_string, cctx);
        }
        _ => {
            for c in expr_children_mut(e) {
                n += split_uses_expr(c, &Ctx::None, kinds, strong_string, cctx);
            }
        }
    }
    n
}

fn split_uses_stmt(
    s: &mut Stmt,
    kinds: &HashMap<(String, UseKind), String>,
    strong_string: &HashSet<String>,
    cctx: &CellCtx<'_>,
) -> usize {
    let mut n = 0;
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += split_uses_stmt(x, kinds, strong_string, cctx);
        }
    }
    let mut fold = |e: &mut Expr| {
        n += split_uses_expr(e, &Ctx::None, kinds, strong_string, cctx);
    };
    match s {
        Stmt::Eval(e)
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e)
        | Stmt::Return(Some(e))
        | Stmt::Assign(_, _, e) => fold(e),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => fold(cond),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                fold(c);
            }
        }
        Stmt::Lock { expr, .. } => fold(expr),
        Stmt::StringSwitch { disc, .. } => fold(disc),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            fold(collection);
            fold(index);
            fold(value);
        }
        Stmt::EventAssign { field, value, .. } => {
            fold(field);
            fold(value);
        }
        _ => {}
    }
    n
}

/// The primitive a raw-store cast names (`*(int*)(…) = v` stores an int).
fn raw_store_type(target: &str) -> Option<TypeKey> {
    let ty = if target.starts_with("*(byte*)") {
        ManagedType::Primitive(PrimKind::U1)
    } else if target.starts_with("*(short*)") {
        ManagedType::Primitive(PrimKind::I2)
    } else if target.starts_with("*(int*)") {
        ManagedType::Primitive(PrimKind::I4)
    } else if target.starts_with("*(float*)") {
        ManagedType::Primitive(PrimKind::R4)
    } else if target.starts_with("*(double*)") {
        ManagedType::Primitive(PrimKind::R8)
    } else if target.starts_with("*(long*)") {
        ManagedType::Primitive(PrimKind::I8)
    } else if target.starts_with("*(nint*)") {
        ManagedType::IntPtr
    } else {
        return None;
    };
    type_key(&ty)
}

/// `*(int*)((nint)&stack_424 +4)` → `("stack_424", 4, 4)`.
fn parse_raw_target(target: &str) -> Option<(String, i64, u8)> {
    if !target.starts_with("*(") {
        return None;
    }
    let width = if target.starts_with("*(byte*)") {
        1
    } else if target.starts_with("*(short*)") {
        2
    } else if target.starts_with("*(int*)") || target.starts_with("*(float*)") {
        4
    } else if target.starts_with("*(double*)") || target.starts_with("*(long*)") {
        8
    } else if target.starts_with("*(nint*)") {
        0
    } else {
        return None;
    };
    let amp = target.find("&stack_")?;
    let rest = &target[amp + 1..];
    let end = rest
        .char_indices()
        .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    let cell = &rest[..end];
    if !is_stack_name(cell) {
        return None;
    }
    let tail = &rest[end..];
    let off = tail
        .find('+')
        .and_then(|p| {
            let digits: String = tail[p + 1..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse::<i64>().ok()
        })
        .unwrap_or(0);
    Some((cell.to_string(), off, width))
}

fn sugar_expr(
    e: &mut Expr,
    map: &HashMap<String, String>,
    fields: &HashMap<(String, i64, u8), String>,
    unwrap_cells: &HashSet<String>,
) -> usize {
    let mut n = 0;
    // Recurse first so a replaced node is not walked again.
    match e {
        Expr::RawDeref(base, off, width) => {
            if let Some((cell, base_off)) = addr_expr_cell(base) {
                let key = (cell.clone(), base_off + *off, normalize_width(*width));
                if let Some(fname) = fields.get(&key) {
                    let newcell = map.get(&cell).cloned().unwrap_or(cell);
                    *e = Expr::Field(Box::new(Expr::Var(newcell)), fname.clone());
                    return 1;
                }
            }
        }
        Expr::Field(base, _) => {
            if let Expr::AddressOf(inner) = base.as_ref() {
                if let Expr::Var(cell) = inner.as_ref() {
                    if unwrap_cells.contains(cell) {
                        let newcell = map.get(cell).cloned().unwrap_or_else(|| cell.clone());
                        *base = Box::new(Expr::Var(newcell));
                        n += 1;
                    }
                }
            }
        }
        _ => {}
    }
    for c in expr_children_mut(e) {
        n += sugar_expr(c, map, fields, unwrap_cells);
    }
    n
}

fn sugar_stmt(
    s: &mut Stmt,
    map: &HashMap<String, String>,
    fields: &HashMap<(String, i64, u8), String>,
    unwrap_cells: &HashSet<String>,
) -> usize {
    let mut n = 0;
    if let Stmt::Assign(target, _, _) = s {
        if let Some((cell, off, width)) = parse_raw_target(target) {
            if let Some(fname) = fields.get(&(cell.clone(), off, normalize_width(width))) {
                let newcell = map.get(&cell).cloned().unwrap_or(cell);
                *target = format!("{newcell}.{fname}");
                n += 1;
            }
        }
    }
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += sugar_stmt(x, map, fields, unwrap_cells);
        }
    }
    let mut fold = |e: &mut Expr| n += sugar_expr(e, map, fields, unwrap_cells);
    match s {
        Stmt::Eval(e)
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e)
        | Stmt::Return(Some(e))
        | Stmt::Assign(_, _, e) => fold(e),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => fold(cond),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                fold(c);
            }
        }
        Stmt::Lock { expr, .. } => fold(expr),
        Stmt::StringSwitch { disc, .. } => fold(disc),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            fold(collection);
            fold(index);
            fold(value);
        }
        Stmt::EventAssign { field, value, .. } => {
            fold(field);
            fold(value);
        }
        _ => {}
    }
    n
}

fn rewrite_cell_stmt(
    s: &mut Stmt,
    map: &HashMap<String, String>,
    fields: &HashMap<(String, i64, u8), String>,
    unwrap_cells: &HashSet<String>,
) -> usize {
    let mut n = sugar_stmt(s, map, fields, unwrap_cells);
    n += rename_stmt(s, map);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn call(recv: Expr, name: &str, args: Vec<Expr>) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(recv)),
            name: name.to_string(),
            args,
        })
    }

    #[test]
    fn move_next_names_the_cell_enumerator() {
        let mut stmts = vec![call(
            Expr::AddressOf(Box::new(var("stack_129"))),
            "MoveNext",
            vec![],
        )];
        assert!(apply(&mut stmts) >= 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("enumerator"), "{t}");
        assert!(!t.contains("stack_129"), "{t}");
    }

    #[test]
    fn by_value_struct_receiver_names_the_cell() {
        let mut stmts = vec![call(var("stack_129"), "MoveNext", vec![])];
        assert!(apply(&mut stmts) >= 1);
        assert!(format!("{stmts:?}").contains("enumerator"));
    }

    #[test]
    fn get_result_names_awaiter() {
        let mut stmts = vec![call(
            Expr::AddressOf(Box::new(var("stack_67"))),
            "GetResult",
            vec![],
        )];
        assert!(apply(&mut stmts) >= 1);
        assert!(format!("{stmts:?}").contains("awaiter"));
    }

    #[test]
    fn try_get_names_out_value() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("dict"))),
            name: "TryGetValue".into(),
            args: vec![var("key"), Expr::AddressOf(Box::new(var("stack_1")))],
        })];
        assert!(apply(&mut stmts) >= 1);
        assert!(format!("{stmts:?}").contains("outValue"));
    }

    #[test]
    fn assignment_target_strings_are_renamed() {
        let mut stmts = vec![
            call(Expr::AddressOf(Box::new(var("stack_133"))), "MoveNext", vec![]),
            Stmt::Assign(
                "*(int*)((nint)&stack_133 +4)".into(),
                None,
                var("x"),
            ),
        ];
        apply(&mut stmts);
        assert!(format!("{stmts:?}").contains("&enumerator"));
    }

    #[test]
    fn unnamed_cell_keeps_its_name() {
        let mut stmts = vec![Stmt::Eval(var("stack_9"))];
        assert_eq!(apply(&mut stmts), 0);
        assert!(format!("{stmts:?}").contains("stack_9"));
    }

    #[test]
    fn get_enumerator_argument_names_enumerator() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("listBoxes"))),
            name: "GetEnumerator".into(),
            args: vec![Expr::AddressOf(Box::new(var("stack_161")))],
        })];
        assert!(apply(&mut stmts) >= 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("enumerator"), "{t}");
        assert!(!t.contains("stack_161"), "{t}");
    }

    #[test]
    fn two_date_members_name_a_date_time() {
        let s = "stack_80";
        let mut stmts = vec![
            call(Expr::AddressOf(Box::new(var(s))), "get_Year", vec![]),
            call(Expr::AddressOf(Box::new(var(s))), "get_Month", vec![]),
        ];
        assert!(apply(&mut stmts) >= 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("dateTime"), "{t}");
        assert!(!t.contains("stack_80"), "{t}");
    }

    #[test]
    fn one_date_member_is_not_type_evidence() {
        let mut stmts = vec![call(
            Expr::AddressOf(Box::new(var("stack_80"))),
            "get_Day",
            vec![],
        )];
        assert_eq!(apply(&mut stmts), 0);
        assert!(format!("{stmts:?}").contains("stack_80"));
    }

    #[test]
    fn two_span_members_name_a_time_span() {
        let s = "stack_60";
        let mut stmts = vec![
            call(Expr::AddressOf(Box::new(var(s))), "get_Ticks", vec![]),
            call(Expr::AddressOf(Box::new(var(s))), "get_TotalHours", vec![]),
        ];
        assert!(apply(&mut stmts) >= 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("timeSpan"), "{t}");
        assert!(!t.contains("stack_60"), "{t}");
    }
}
