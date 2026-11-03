//! Give machine-slot locals source-like names.
//!
//! ## Why names are `vN`
//!
//! SSA versions are keyed by their MACHINE SLOT (`x0`, `x21`, `sp`, …). For
//! parameters and `this` the table carries identity, so the renderer prints the
//! real name; every other defined slot falls through to `v{table index}`.
//! `highvar::merge_names` unifies non-overlapping versions of the same slot
//! (so `v16 = v16 - 1` is one source variable), but the name is still an index.
//! Measured on `InitialImageSorting` (0x1489EC4): `v16` is slot `x21`, `v64`
//! is `x0`, `v19` is `x19`; none carries a source name.
//!
//! This pass studies each surviving local's DEFINITIONS and derives a name:
//!
//! * every def loads the same field → the field name (`_settings` → `settings`);
//! * every def calls the same getter/method → the property name
//!   (`GetActiveDeal()` → `activeDeal`, `get_Count()` → `count`);
//! * every def reads the same static field → the field name;
//! * a loop induction (defs are an integer seed and `v ± const`) → `i`, `j`, …;
//! * a def that copies another local (after that local's rename) → its name;
//! * a sole metadata-type candidate → the camel-cased type name, unless the
//!   type is primitive (`Int32`, `Boolean`, …).
//!
//! Renaming is a pure substitution of a local name across `Var` references,
//! `Assign` targets, switch discriminants and `foreach` collections. The last
//! three are stored as PRE-RENDERED strings, so a complex target such as
//! `*(float*)(((v124 + (v124 << 1)) …` embeds the display name as a bare
//! token that no expression walker can see; those tokens are rewritten
//! token-wise. Missing that rewrite renamed a definition (`v124 = 0` →
//! `j = 0`) while every read inside a target string kept `v124`, which is the
//! shape of the undefined-local census. Names that would collide with an
//! existing identifier, a parameter, a `foreach` variable or a C# keyword are
//! skipped. Locals with mixed or unknown provenance keep `vN`.

use super::super::csharp::{BinOp, Expr, Stmt};
use super::super::locals::LocalTable;
use super::super::types::{ManagedType, PrimKind};
use super::children_mut;
use std::collections::{HashMap, HashSet};

#[derive(Default)]
struct LocalFacts {
    defs: usize,
    /// (field) for `Field(base, field)` definitions.
    fields: HashSet<String>,
    /// (method) for `Call { name }` definitions.
    calls: HashSet<String>,
    /// Runtime-helper roles resolved from a RawCall target address.
    helper_roles: HashSet<&'static str>,
    /// (field) for `StaticField(_, field)` definitions.
    static_fields: HashSet<String>,
    /// Candidate names for `base[index]` definitions, with their weight.
    index_loads: HashMap<String, f32>,
    /// (source local) for `Var(src)` definitions.
    copies: HashSet<String>,
    /// Number of defs that are an integer literal (induction seed).
    int_defs: usize,
    /// Defs that are a type/klass token (`typeof(T)` / `klass(T)`).
    typeof_defs: usize,
    /// True when every def is an integer literal or `<any> ± const` (a
    /// numeric cursor).
    numeric_cursor: bool,
    /// The local is used as an array/List index.
    index_use: bool,
    /// The local is compared against an integer constant.
    cmp_int_use: bool,
    /// Number of defs that are `v = v ± const` (loop self-steps).
    self_steps: usize,
    /// `this.<field> = v` consumers: how often each field receives this local.
    field_stores: HashMap<String, u32>,
    /// `Set<Prop>(v, …)` consumers: how often each property receives it.
    setter_args: HashMap<String, u32>,
    seen_def: bool,
    refs: usize,
}

/// A machine-slot name that carries no source evidence: `vN`, `in_xN`,
/// `stack_N`, `__abi`, or an unresolved pointer/function-pointer token.
/// An SSA/table version name (`v12`), not merely an identifier starting with
/// `v`. The old filter was `starts_with('v')`, so a source name produced by an
/// earlier round (`value`, `volumeCache`, `verts`) was re-scanned as a temp and
/// could be renamed AGAIN from its own field/call evidence.
fn is_version_name(n: &str) -> bool {
    n.starts_with('v') && n.len() > 1 && n[1..].chars().all(|c| c.is_ascii_digit())
}

fn looks_like_machine_name(n: &str) -> bool {
    (n.starts_with('v') && n.len() > 1 && n[1..].chars().all(|c| c.is_ascii_digit()))
        || n.starts_with("in_")
        || n.starts_with("stack_")
        || n == "__abi"
        || n.starts_with("fnptr_")
        || n.starts_with("ptr_")
}

/// Name evidence for an element load `collection[index]`.
///
/// The element itself has no syntactic name, but its container often does:
/// `this._unlockInfos[i]` is an unlock info, `path[j]` is a path element. Only
/// a container with a real name contributes: a `this.<field>` base is strong
/// (3.0); a named local is weaker (2.0) because it may hold a mix. A `vN` /
/// `in_x` / `stack` / `__abi` base carries no source identity and must
/// abstain (the fixpoint in `apply` may rename it in a later round).
fn index_load_hint(collection: &Expr) -> Option<(String, f32)> {
    match collection {
        Expr::Field(recv, field) => {
            let field = unwrap_backing(field);
            let weight = match recv.as_ref() {
                Expr::This => 3.0,
                Expr::Var(base) if !looks_like_machine_name(base) => 2.0,
                _ => return None,
            };
            camel(&field).map(|x| (x, weight))
        }
        Expr::StaticField(_, field) => camel(field).map(|x| (x, 3.0)),
        Expr::Var(base) if !looks_like_machine_name(base) => camel(base).map(|x| (x, 2.0)),
        _ => None,
    }
}

fn is_simple_local(n: &str) -> bool {
    !n.is_empty()
        && !n.contains('.')
        && !n.contains('[')
        && !n.contains('*')
        && !n.contains('&')
        && !n.contains('<')
        && n.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !n.chars().next().is_some_and(|c| c.is_ascii_digit())
}

fn collect(stmts: &[Stmt], out: &mut HashMap<String, LocalFacts>, used: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign(name, _, rhs) => {
                // A rendered lvalue path (`*(int*)(v12 +16)`, `v12.field`)
                // embeds display names as tokens; reserve them all so a
                // candidate name can never collide with one.
                if !is_simple_local(name) {
                    note_text_names(name, used);
                }
                if is_simple_local(name) {
                    let f = out.entry(name.clone()).or_default();
                    if !f.seen_def {
                        f.numeric_cursor = true;
                    }
                    f.defs += 1;
                    f.seen_def = true;
                    match rhs {
                        Expr::Field(_, field) => {
                            f.fields.insert(field.clone());
                        }
                        Expr::StaticField(_, field) => {
                            f.static_fields.insert(field.clone());
                        }
                        Expr::TypeOf(_) => {
                            f.typeof_defs += 1;
                        }
                        Expr::Call { name: m, .. } => {
                            f.calls.insert(m.clone());
                        }
                        Expr::RawCall(addr, _, _) => {
                            if let Some(role) =
                                super::super::runtime_helpers::role_of(*addr)
                            {
                                f.helper_roles.insert(role.display_name());
                            }
                        }
                        Expr::Var(src) => {
                            f.copies.insert(src.clone());
                        }
                        Expr::Index { collection, .. } => {
                            if let Some((hint, weight)) = index_load_hint(collection) {
                                f.index_loads.entry(hint).or_insert(weight);
                            }
                        }
                        Expr::Int(_) => {
                            f.int_defs += 1;
                        }
                        Expr::Bin(op, a, b) => {
                            let self_step = match (op, a.as_ref(), b.as_ref()) {
                                (BinOp::Add | BinOp::Sub, Expr::Var(v), Expr::Int(_)) => v == name,
                                (BinOp::Add, Expr::Int(_), Expr::Var(v)) => v == name,
                                _ => false,
                            };
                            if self_step {
                                f.self_steps += 1;
                            }
                            // A cursor: arithmetic with an integer constant on
                            // one side (`n - 1`, `i + 1`). Not necessarily an
                            // induction variable, but numeric.
                            let cursor_step = matches!(
                                (op, a.as_ref(), b.as_ref()),
                                (BinOp::Add | BinOp::Sub, _, Expr::Int(_))
                                    | (BinOp::Add, Expr::Int(_), _)
                            );
                            if !cursor_step {
                                f.numeric_cursor = false;
                            }
                        }
                        _ => {
                            f.numeric_cursor = false;
                        }
                    }
                    // A cursor's every def is numeric (literal or +/- const).
                    // Field/copy/call defs fall through the arms above without
                    // touching the flag, so close it here.
                    if !matches!(rhs, Expr::Int(_) | Expr::Bin(..)) {
                        f.numeric_cursor = false;
                    }
                }
                used.insert(name.clone());
                mark_used(rhs, used);
                if let Expr::Var(src) = rhs {
                    if let Some(field) = field_store_target(name) {
                        if let Some(f) = out.get_mut(src.as_str()) {
                            *f.field_stores.entry(field).or_insert(0) += 1;
                        }
                    }
                }
            }
            Stmt::ForEach {
                var_name,
                collection,
                body,
            } => {
                used.insert(var_name.clone());
                note_text_names(collection, used);
                collect(body, out, used);
            }
            Stmt::Switch { disc, .. } => {
                note_text_names(disc, used);
                for k in children_list(s) {
                    collect(k, out, used);
                }
            }
            Stmt::EventAssign { field, value, .. } => {
                mark_used(field, used);
                mark_used(value, used);
            }
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                mark_used(collection, used);
                mark_used(index, used);
                mark_used(value, used);
            }
            _ => {
                for e in stmt_exprs(s) {
                    mark_used(e, used);
                    note_calls(e, out);
                }
                for k in children_list(s) {
                    collect(k, out, used);
                }
            }
        }
    }
}

/// All expressions a statement mentions directly (for the used-name set).
fn stmt_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => vec![e],
        Stmt::Return(Some(e)) => vec![e],
        Stmt::Assign(_, _, e) => vec![e],
        Stmt::If { cond, .. } => vec![cond],
        Stmt::While { cond, .. } => vec![cond],
        Stmt::DoWhile { cond, .. } => vec![cond],
        Stmt::Lock { expr, .. } => vec![expr],
        Stmt::StringSwitch { disc, .. } => vec![disc],
        Stmt::For { cond, .. } => cond.iter().collect(),
        _ => vec![],
    }
}

fn children_list(s: &Stmt) -> Vec<&Vec<Stmt>> {
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    kids
}

fn mark_used(e: &Expr, used: &mut HashSet<String>) {
    match e {
        Expr::Var(n) => {
            used.insert(n.clone());
        }
        Expr::Field(b, _)
        | Expr::RawDeref(b, _, _)
        | Expr::RawAddr(b, _)
        | Expr::AddressOf(b)
        | Expr::Cast(_, b)
        | Expr::Not(b)
        | Expr::Neg(b)
        | Expr::Await(b) => mark_used(b, used),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            mark_used(a, used);
            mark_used(b, used);
        }
        Expr::Cond(c, a, b) => {
            mark_used(c, used);
            mark_used(a, used);
            mark_used(b, used);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                mark_used(r, used);
            }
            for a in args {
                mark_used(a, used);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                mark_used(a, used);
            }
        }
        Expr::Index { collection, index } => {
            mark_used(collection, used);
            mark_used(index, used);
        }
        Expr::NullConditional { receiver, access } => {
            mark_used(receiver, used);
            mark_used(access, used);
        }
        _ => {}
    }
}

const KEYWORDS: &[&str] = &[
    "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char",
    "checked", "class", "const", "continue", "decimal", "default", "delegate",
    "do", "double", "else", "enum", "event", "explicit", "extern", "false",
    "finally", "fixed", "float", "for", "foreach", "goto", "if", "implicit",
    "in", "int", "interface", "internal", "is", "lock", "long", "namespace",
    "new", "null", "object", "operator", "out", "override", "params", "private",
    "protected", "public", "readonly", "ref", "return", "sbyte", "sealed",
    "short", "sizeof", "stackalloc", "static", "string", "struct", "switch",
    "this", "throw", "true", "try", "typeof", "uint", "ulong", "unchecked",
    "unsafe", "ushort", "using", "virtual", "void", "volatile", "while",
];

fn camel(s: &str) -> Option<String> {
    let s = s.trim_start_matches('_');
    let s = s.strip_prefix("get_").or_else(|| s.strip_prefix("Get")).unwrap_or(s);
    if s.is_empty() {
        return None;
    }
    let mut c = s.chars();
    let first = c.next()?;
    let mut out = String::new();
    out.extend(first.to_lowercase());
    out.push_str(c.as_str());
    if !is_simple_local(&out) || KEYWORDS.contains(&out.as_str()) {
        return None;
    }
    // GetActiveDeal → activeDeal; UIData → uIData is acceptable but odd; keep.
    Some(out)
}

/// The field a `this.<field> = v` target names, with compiler backing fields
/// (`<contentID>k__BackingField`) unwrapped to their source property.
fn field_store_target(target: &str) -> Option<String> {
    let field = target.rsplit('.').next()?;
    if field == target {
        return None; // not a field path
    }
    Some(unwrap_backing(field))
}

fn unwrap_backing(field: &str) -> String {
    if let Some(rest) = field.strip_prefix('<') {
        if let Some(end) = rest.find('>') {
            let inner = &rest[..end];
            if !inner.is_empty() {
                return inner.to_string();
            }
        }
    }
    field.to_string()
}

/// `Set<Prop>(v, …)` uses `v` as the value of property `<Prop>`.
fn note_calls(e: &Expr, out: &mut HashMap<String, LocalFacts>) {
    if let Expr::Call { name, args, .. } = e {
        if let Some(prop) = name.strip_prefix("Set") {
            if !prop.is_empty() {
                for a in args {
                    if let Expr::Var(v) = a {
                        if let Some(f) = out.get_mut(v.as_str()) {
                            *f.setter_args.entry(prop.to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
    }
    for c in expr_children(e) {
        note_calls(c, out);
    }
}

/// Score every candidate name from the full evidence set and return the
/// unambiguous winner. Weights: an exact producer (field/getter/static) is
/// worth 3, a consumer that names the value's role 2, the metadata type 1.
/// A tie or a missing strong candidate keeps the stable `vN`.
fn infer_name(name: &str, f: &LocalFacts, type_names: &HashMap<String, String>) -> Option<String> {
    let mut cands: HashMap<String, f32> = HashMap::new();
    let mut add = |hint: String, w: f32| {
        *cands.entry(hint).or_insert(0.0) += w;
    };
    if f.defs > 0 {
        if f.fields.len() == 1 {
            if let Some(x) = f.fields.iter().next().and_then(|x| camel(&unwrap_backing(x))) {
                add(x, 3.0);
            }
        }
        if f.static_fields.len() == 1 {
            if let Some(x) = f.static_fields.iter().next().and_then(|x| camel(x)) {
                add(x, 3.0);
            }
        }
        if f.index_loads.len() == 1 {
            if let Some((hint, weight)) = f.index_loads.iter().next() {
                if let Some(x) = camel(hint) {
                    add(x, *weight);
                }
            }
        }
        // A local whose every definition is a class/type token is the
        // recovered source-level `Type` value (`v = typeof(T)`); name it
        // `type` rather than a machine index. These tokens are field bases
        // and arguments across the corpus (≈3.6k refs in the R63 census).
        if f.defs > 0 && f.typeof_defs == f.defs {
            add("type".to_string(), 3.0);
        }
        // A value used as an array/List subscript is an index by ROLE. It is
        // weaker than a producer (a field load can override it) but it names
        // the remaining anonymous cursors.
        if f.index_use {
            add("index".to_string(), 2.0);
        }
        if f.helper_roles.len() == 1 {
            let role = *f.helper_roles.iter().next().unwrap();
            let hint = match role {
                "IL2CPP_ArrayNew" => Some("array"),
                "IL2CPP_IsInst" => Some("instance"),
                "IL2CPP_ObjectNew" => Some("instance"),
                "IL2CPP_Box" => Some("boxed"),
                "IL2CPP_ClassIsAssignable" => Some("isAssignable"),
                _ => None,
            };
            if let Some(h) = hint {
                add(h.to_string(), 3.0);
            }
        }
        if f.calls.len() == 1 {
            let m = f.calls.iter().next().unwrap();
            if m.starts_with("Find") {
                add("index".to_string(), 3.0);
            } else {
                // Name a cached call result from the callee itself, not only
                // from getter-shaped names. `CalculateReward()` → calculateReward,
                // `GetComponent<Rigidbody>()` → component. Raw runtime helpers
                // (`fn_`, `IL2CPP_`), constructors and mutators are not values
                // the source named this way.
                let base = m.split('<').next().unwrap_or(m);
                let last = base.rsplit('.').next().unwrap_or(base);
                let skip = base.starts_with("fn_")
                    || base.starts_with("IL2CPP_")
                    || base.contains(".ctor")
                    || base.starts_with("set_")
                    || matches!(
                        last,
                        "Add" | "Remove" | "Clear" | "Equals" | "GetHashCode"
                            | "ToString" | "Dispose" | "Invoke" | "MoveNext"
                            | "get_Current" | "MoveNextAsync"
                    );
                if !skip {
                    if let Some(x) = camel(last) {
                        add(x, 3.0);
                    }
                }
            }
        }
        for (field, n) in &f.field_stores {
            if let Some(x) = camel(&unwrap_backing(field)) {
                add(x, 2.0 + (*n).min(2) as f32 * 0.1);
            }
        }
        for (prop, n) in &f.setter_args {
            if let Some(x) = camel(prop) {
                add(x, 2.0 + (*n).min(2) as f32 * 0.1);
            }
        }
        if f.copies.len() == 1 {
            let src = f.copies.iter().next().unwrap();
            // A MACHINE name is not a source name. `looks_like_machine_name`
            // (not just `vN`) is the predicate: `in_x3`, `stack_147`, `__abi`
            // and the pointer tokens all name STORAGE, so a copy of one is a
            // different value with no evidence of its own. The old test
            // excluded only `vN`, so `in_x32 = in_x3` took the source's name,
            // the collision suffixer appended a digit, and the dump grew
            // variables called `in_x32`, `in_x322`, `stack_1472` — register
            // numbers that do not exist and frame cells that are not cells
            // (1,284 defs of an `in_xN` name binary-wide, 707 of them copies
            // of another `in_xN`). Keeping `vN` lets the fixpoint name the
            // value from real producer/consumer evidence in a later round.
            if !looks_like_machine_name(src) {
                if let Some(x) = camel(src) {
                    // A sole non-temp copy source names the value as strongly
                    // as a producer: `v = playerStats;` is playerStats.
                    add(x, 3.0);
                }
            }
        }
        // A loop counter. The seed is usually 0, but 1/4/52 are just as
        // common (a start index or a byte offset), and a long chain of
        // self-steps with a single literal seed is unambiguous.
        let pure_induction = f.defs >= 2
            && f.self_steps >= 1
            && f.int_defs + f.self_steps == f.defs;
        let mixed_induction = f.defs >= 2
            && f.self_steps >= 1
            && f.int_defs >= 1
            && (f.index_use || f.cmp_int_use);
        // Numeric cursors need evidence of index use; two different literal
        // defs (`v = 5; v = 10`) are a select, not a counter.
        let cursor = f.numeric_cursor
            && f.defs >= 2
            && (f.index_use || f.cmp_int_use)
            && !(f.int_defs >= 2 && f.self_steps == 0);
        if pure_induction || mixed_induction || cursor {
            // `i` is the source's own spelling for the commonest cursor; the
            // collision suffixer turns a second one into `i2`.
            add("i".to_string(), 3.0);
        }
        if let Some(raw) = type_names.get(name) {
            let simple = raw.rsplit('.').next().unwrap_or(raw);
            if !is_primitive_type_name(simple) {
                if let Some(t) = camel(raw) {
                    // A COMMITTED class/struct type (every definition agrees,
                    // `infer_types_from_stmts`) is a source-TYPE role, not a
                    // hint: it names an `obj`/`vN` local after what it holds.
                    // Weak BCL roots stay at the old low weight so this never
                    // mints `object`/`valueType`/`array` variables.
                    let weak = matches!(
                        simple,
                        "Object"
                            | "object"
                            | "ValueType"
                            | "Enum"
                            | "Delegate"
                            | "MulticastDelegate"
                            | "Array"
                    );
                    // Do not create a 3.0/3.0 tie with a producer: a tie
                    // abstains and leaves the generated `objN` name, which is
                    // how raising this weight unconditionally made the corpus
                    // WORSE (+1,355 `objN`). The type is strong only when no
                    // producer-shaped evidence exists; then it beats a lone
                    // index (2.0) role.
                    let producer_exists = f.fields.len() == 1
                        || f.static_fields.len() == 1
                        || f.calls.len() == 1
                        || (f.defs > 0 && f.typeof_defs == f.defs)
                        || f.index_loads.len() == 1
                        || f.helper_roles.len() == 1
                        || (f.defs >= 2 && f.self_steps >= 1)
                        || f.copies.len() == 1;
                    add(t, if weak || producer_exists { 1.0 } else { 3.0 });
                }
            }
        }
    }
    let mut ranked: Vec<(String, f32)> = cands.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (best, top) = ranked.first()?;
    let second = ranked.get(1).map(|(_, s)| *s).unwrap_or(0.0);
    if *top >= 3.0 && *top > second {
        Some(best.clone())
    } else if ranked.len() == 1 {
        // A SINGLE candidate is still evidence, even below the producer
        // weight: a local whose only hint is its metadata type, or a sole
        // copy of an already-named variable, has no competing story to tell.
        // The weighted threshold exists to break CONFLICTS (field vs type,
        // two setters); with one candidate there is none.
        Some(best.clone())
    } else {
        None
    }
}

/// Last-resort source-style name from the local's managed TYPE.
///
/// Only reached when no producer/consumer/copy evidence named the value. The
/// name states the value's role, never a claim about the source identifier:
/// a `List<T>` is `list`, a `Transform` is `transform`, a float is `num`, an
/// untyped value is `value`. Collisions get the normal numeric suffix. This is
/// what removes the remaining machine names after every evidence rule has
/// abstained; it is deliberately a ROLE, not a fabricated field/method name.
fn type_role_name(name: &str, table: &LocalTable) -> Option<String> {
    let li: usize = name.strip_prefix('v')?.parse().ok()?;
    let info = table.infos.get(li)?;
    let word = match &info.ty {
        ManagedType::Object => "obj".to_string(),
        ManagedType::String => "text".to_string(),
        ManagedType::IntPtr => "ptr".to_string(),
        // Primitives and untyped values have no informative role word: the
        // source identifier is unrecoverable and `num`/`value` would be a
        // guess. They keep `vN` unless evidence names them.
        ManagedType::Unknown | ManagedType::Primitive(_) => return None,
        ManagedType::SzArray(e) => match e.as_ref() {
            ManagedType::Primitive(PrimKind::U1) => "bytes".to_string(),
            ManagedType::Primitive(PrimKind::I1) => "sbytes".to_string(),
            ManagedType::Primitive(PrimKind::Boolean) => "flags".to_string(),
            ManagedType::String => "strings".to_string(),
            ManagedType::Object => "items".to_string(),
            _ => "array".to_string(),
        },
        ManagedType::Class(nt) => camel(&nt.name)?,
        ManagedType::Opaque(s) => {
            let base = s.split('<').next()?.rsplit('.').next()?;
            camel(base)?
        }
        ManagedType::ByRef(inner) | ManagedType::Pointer(inner) => match inner.as_ref() {
            ManagedType::Class(nt) => camel(&nt.name)?,
            ManagedType::String => "text".to_string(),
            _ => "ptr".to_string(),
        },
    };
    Some(word)
}

/// Type names that would make a variable read like its CLR type rather than a
/// source identifier (`int32`, `boolean`, …). A type-only rename must never
/// produce these; the local keeps its `vN`.
fn is_primitive_type_name(name: &str) -> bool {
    matches!(
        name,
        "Int16"
            | "Int32"
            | "Int64"
            | "UInt16"
            | "UInt32"
            | "UInt64"
            | "Byte"
            | "SByte"
            | "Single"
            | "Double"
            | "Boolean"
            | "Char"
            | "IntPtr"
            | "UIntPtr"
            | "Decimal"
            | "Void"
            | "Object"
    )
}

/// Mark index/cmp uses of simple locals, which disambiguates numeric cursors.
fn collect_contexts(stmts: &[Stmt], facts: &mut HashMap<String, LocalFacts>) {
    for s in stmts {
        for e in stmt_exprs(s) {
            note_expr(e, facts);
        }
        for k in children_list(s) {
            collect_contexts(k, facts);
        }
    }
}

fn note_expr(e: &Expr, facts: &mut HashMap<String, LocalFacts>) {
    match e {
        Expr::Index { index, .. } => {
            if let Expr::Var(n) = index.as_ref() {
                if let Some(f) = facts.get_mut(n) {
                    f.index_use = true;
                }
            }
        }
        Expr::Cmp(_, a, b) => {
            let is_int = |x: &Expr| matches!(x, Expr::Int(_) | Expr::Null);
            if let Expr::Var(n) = a.as_ref() {
                if is_int(b) {
                    if let Some(f) = facts.get_mut(n) {
                        f.cmp_int_use = true;
                    }
                }
            }
            if let Expr::Var(n) = b.as_ref() {
                if is_int(a) {
                    if let Some(f) = facts.get_mut(n) {
                        f.cmp_int_use = true;
                    }
                }
            }
        }
        _ => {}
    }
    for c in expr_children(e) {
        note_expr(c, facts);
    }
}

fn expr_children(e: &Expr) -> Vec<&Expr> {
    // Lambda bodies are a separate binding scope; the canonical visitor does
    // not expose their parameters, so a shadowed local could not be excluded.
    // Everything else — including `SwitchExpr`, whose discriminant and arms
    // were invisible here and therefore never renamed (the undefined `v26`
    // read in a structured switch-expression body) — uses the ONE canonical
    // visitor so this scan cannot drift from the renaming walk below.
    if matches!(e, Expr::Lambda { .. }) {
        return Vec::new();
    }
    crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e)
}

/// Insert every BARE identifier token of a rendered string operand into
/// `used`. Member names (`obj.field`) are skipped: they are field reads, not
/// locals, and C# lets a local shadow a field name without changing meaning.
fn note_text_names(text: &str, used: &mut HashSet<String>) {
    let mut cur = String::new();
    let mut member = false;
    let mut prev = '\0';
    let flush = |cur: &mut String, member: bool, used: &mut HashSet<String>| {
        if cur.is_empty() {
            return;
        }
        let starts_ok = cur
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        if starts_ok && !member {
            used.insert(std::mem::take(cur));
        } else {
            cur.clear();
        }
    };
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if cur.is_empty() {
                member = prev == '.';
            }
            cur.push(ch);
        } else {
            flush(&mut cur, member, used);
        }
        if !ch.is_whitespace() {
            prev = ch;
        }
    }
    flush(&mut cur, member, used);
}

/// Rewrite mapped local names that appear as WHOLE IDENTIFIER TOKENS inside a
/// pre-rendered string operand (assignment target, switch discriminant,
/// foreach collection). These strings were rendered from expressions before
/// this pass, so `rename_expr` cannot reach the names embedded in them.
/// Returns `None` when nothing changed.
fn rename_text(text: &str, map: &HashMap<String, String>) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut changed = false;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let tok = &text[start..i];
            match map.get(tok) {
                Some(new) => {
                    out.push_str(new);
                    changed = true;
                }
                None => out.push_str(tok),
            }
        } else {
            let ch = text[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    if changed {
        Some(out)
    } else {
        None
    }
}

/// Replace every simple `Var(name)` and every embedded name token of a
/// pre-rendered `Assign` target / `Switch` discriminant / `ForEach` collection.
fn rename(stmts: &mut [Stmt], map: &HashMap<String, String>) -> usize {
    let mut n = 0;
    for s in stmts.iter_mut() {
        match s {
            Stmt::Assign(name, _, rhs) => {
                if let Some(new) = map.get(name.as_str()) {
                    *name = new.clone();
                    n += 1;
                } else if let Some(t) = rename_text(name, map) {
                    // A complex lvalue path: `*(float*)(v12 + 32)`.
                    *name = t;
                    n += 1;
                }
                rename_expr(rhs, map);
            }
            Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => rename_expr(e, map),
            Stmt::Return(Some(e)) => rename_expr(e, map),
            Stmt::If { cond, then_body, else_body } => {
                rename_expr(cond, map);
                n += rename(then_body, map);
                n += rename(else_body, map);
            }
            Stmt::While { cond, body } => {
                rename_expr(cond, map);
                n += rename(body, map);
            }
            Stmt::DoWhile { body, cond } => {
                n += rename(body, map);
                rename_expr(cond, map);
            }
            Stmt::For { init, cond, incr, body } => {
                n += rename(init, map);
                if let Some(c) = cond {
                    rename_expr(c, map);
                }
                n += rename(incr, map);
                n += rename(body, map);
            }
            Stmt::Lock { expr, body } => {
                rename_expr(expr, map);
                n += rename(body, map);
            }
            Stmt::IndexAssign { collection, index, value } => {
                rename_expr(collection, map);
                rename_expr(index, map);
                rename_expr(value, map);
            }
            Stmt::EventAssign { field, value, .. } => {
                rename_expr(field, map);
                rename_expr(value, map);
            }
            Stmt::StringSwitch { disc, cases, tail } => {
                rename_expr(disc, map);
                for (_, b) in cases.iter_mut() {
                    n += rename(b, map);
                }
                if let Some(t) = tail {
                    n += rename(t, map);
                }
            }
            Stmt::Switch {
                disc,
                cases,
                tail,
                default,
            } => {
                if let Some(t) = rename_text(disc, map) {
                    *disc = t;
                    n += 1;
                }
                for (_, b) in cases.iter_mut() {
                    n += rename(b, map);
                }
                n += rename(tail, map);
                if let Some(d) = default {
                    n += rename(d, map);
                }
            }
            Stmt::ForEach { collection, body, .. } => {
                if let Some(t) = rename_text(collection, map) {
                    *collection = t;
                    n += 1;
                }
                n += rename(body, map);
            }
            _ => {
                for k in children_mut(s) {
                    n += rename(k, map);
                }
            }
        }
    }
    n
}

fn rename_expr(e: &mut Expr, map: &HashMap<String, String>) {
    if let Expr::Var(n) = e {
        if let Some(new) = map.get(n.as_str()) {
            *n = new.clone();
        }
        return;
    }
    // Lambda bodies are a separate binding scope (see `expr_children`).
    if matches!(e, Expr::Lambda { .. }) {
        return;
    }
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
        rename_expr(c, map);
    }
}

/// Name locals whose provenance is unambiguous. Returns the number of locals
/// renamed (all their refs are rewritten).
///
/// Naming is iterated to a small fixpoint: a copy `v = w` is only nameable
/// once `w` carries a source name, and the facts for `v` are collected before
/// `w` is renamed in the same pass. One round left `v17 = instance;` behind
/// whenever `v38` was renamed to `instance` in that same round
/// (0x15A6A10). Each round is a pure rename, so no value or order changes;
/// three rounds bound a copy cycle.
pub fn apply(stmts: &mut [Stmt], type_names: &HashMap<String, String>) -> usize {
    apply_with_rva(stmts, type_names, 0, &LocalTable::default())
}

/// Same, with the method RVA for the naming census (`ILR_NAMECENSUS`) and the
/// local table for TYPE-ROLE fallback names.
pub fn apply_with_rva(
    stmts: &mut [Stmt],
    type_names: &HashMap<String, String>,
    rva: u64,
    table: &LocalTable,
) -> usize {
    let mut renames: HashMap<String, String> = HashMap::new();
    apply_with_rva_map(stmts, type_names, rva, table, &mut renames)
}

/// [`apply_with_rva`], reporting the rename map it applied.
///
/// Callers that need to keep a name-keyed side table (the local TYPE map,
/// which is captured before renaming) must rekey it through this map —
/// otherwise every renamed local loses its metadata type.
pub fn apply_with_rva_map(
    stmts: &mut [Stmt],
    type_names: &HashMap<String, String>,
    rva: u64,
    table: &LocalTable,
    renames: &mut HashMap<String, String>,
) -> usize {
    let mut total = 0usize;
    for _ in 0..3 {
        let n = apply_once(stmts, type_names, rva, table, renames);
        total += n;
        if n == 0 {
            break;
        }
    }
    total
}

fn apply_once(
    stmts: &mut [Stmt],
    type_names: &HashMap<String, String>,
    rva: u64,
    table: &LocalTable,
    renames: &mut HashMap<String, String>,
) -> usize {
    let mut facts: HashMap<String, LocalFacts> = HashMap::new();
    let mut used: HashSet<String> = HashSet::new();
    collect(stmts, &mut facts, &mut used);
    collect_contexts(stmts, &mut facts);
    // Only `vN` names are candidates: parameters/fields already have names and
    // renaming them would fight the renderer's identity rules.
    let mut map: HashMap<String, String> = HashMap::new();
    let mut taken: HashSet<String> = used.clone();
    // Two SSA versions of ONE stack cell are the same C# variable (the cell
    // is memory, not a value). They must share a display name; suffixing the
    // second (`enumerator`, `enumerator2`) invents a second variable and, in
    // the enumerator case, breaks the foreach matcher's text identity.
    let origin_of = |name: &str| -> Option<String> {
        let v: u32 = name.strip_prefix('v')?.parse().ok()?;
        table
            .index_of
            .get(&v)
            .map(|&li| table.infos[li].slot_origin.clone())
    };
    let mut assigned_origin: HashMap<String, String> = HashMap::new();
    let mut renamed = 0usize;
    let mut entries: Vec<(String, LocalFacts)> = facts
        .into_iter()
        .filter(|(name, f)| is_version_name(name) && f.defs > 0 && f.seen_def)
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut renamed_names: HashSet<String> = HashSet::new();
    for (name, f) in entries.iter() {
        let name = name.as_str();
        let Some(hint) =
            infer_name(name, f, type_names).or_else(|| type_role_name(name, table))
        else {
            continue;
        };
        // Induction locals all want `i`; give later ones j, k, … deterministically.
        let mut candidate = hint.clone();
        if candidate == "i" {
            for alt in ["i", "j", "k", "n", "m"] {
                if !taken.contains(alt) {
                    candidate = alt.to_string();
                    break;
                }
            }
        }
        if candidate == name {
            continue;
        }
        // Collisions take a deterministic numeric suffix (`array`, `array2`,
        // …) instead of abstaining. The evidence still names the value; only
        // the display identifier had to be disambiguated. This is what lets
        // helper-role and call-result names apply in methods that already use
        // the natural word.
        if taken.contains(&candidate) {
            let cur_origin = origin_of(name);
            let same_cell = cur_origin.as_ref().is_some_and(|o| {
                o.starts_with("stack_") && assigned_origin.get(&candidate) == Some(o)
            });
            if !same_cell {
                let base = candidate.clone();
                let mut k = 2usize;
                loop {
                    let c = format!("{base}{k}");
                    if !taken.contains(&c) && c != name {
                        candidate = c;
                        break;
                    }
                    k += 1;
                    if k > 64 {
                        candidate.clear();
                        break;
                    }
                }
                if candidate.is_empty() {
                    continue;
                }
            }
        }
        taken.insert(candidate.clone());
        if let Some(o) = origin_of(name) {
            assigned_origin.insert(candidate.clone(), o);
        }
        renamed_names.insert(name.to_string());
        renames.insert(name.to_string(), candidate.clone());
        map.insert(name.to_string(), candidate);
        renamed += 1;
    }
    namensus_record(rva, &entries, &renamed_names, type_names);
    if std::env::var("ILR_NAMING").is_ok() && !map.is_empty() {
        eprintln!("[naming] renamed {renamed} local(s): {map:?}");
    }
    if map.is_empty() {
        return 0;
    }
    rename(stmts, &map);
    renamed
}

/// `ILR_NAMECENSUS=<tsv>`: why each `vN` name was (or was not) named.
///
/// One row per candidate with its full evidence set. Aggregating this is the
/// answer to "which naming rule would capture the remaining 62k refs": the
/// columns are exactly the inputs `infer_name` scores.
fn namensus_record(
    rva: u64,
    entries: &[(String, LocalFacts)],
    renamed: &HashSet<String>,
    type_names: &HashMap<String, String>,
) {
    use std::io::Write;
    let Ok(path) = std::env::var("ILR_NAMECENSUS") else {
        return;
    };
    static WRITER: std::sync::OnceLock<
        std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
    > = std::sync::OnceLock::new();
    let writer = WRITER.get_or_init(|| {
        let f = std::fs::File::create(&path).unwrap_or_else(|e| {
            panic!("ILR_NAMECENSUS: cannot create {path}: {e}")
        });
        let mut w = std::io::BufWriter::new(f);
        let _ = writeln!(
            w,
            "rva\tname\tdefs\trefs\tfields\tcalls\tstatic\tcopies\tidxload\thelpers\tfstores\tsetters\tints\tsteps\ttypeof\tidxuse\tcmpint\ttype\trenamed"
        );
        std::sync::Mutex::new(w)
    });
    let set = |s: &HashSet<String>| {
        let mut v: Vec<&str> = s.iter().map(|x| x.as_str()).collect();
        v.sort_unstable();
        v.join("|")
    };
    let counts = |m: &HashMap<String, u32>| {
        let mut v: Vec<String> = m.iter().map(|(k, n)| format!("{k}:{n}")).collect();
        v.sort_unstable();
        v.join("|")
    };
    let mut out = String::new();
    for (name, f) in entries {
        let idx = {
            let mut v: Vec<String> = f
                .index_loads
                .iter()
                .map(|(k, w)| format!("{k}:{w}"))
                .collect();
            v.sort_unstable();
            v.join("|")
        };
        let ty = type_names.get(name).cloned().unwrap_or_default();
        let mut helpers: Vec<&str> = f.helper_roles.iter().copied().collect();
        helpers.sort_unstable();
        let helpers = helpers.join("|");
        out.push_str(&format!(
            "{rva:#x}\t{name}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            f.defs,
            f.refs,
            set(&f.fields),
            set(&f.calls),
            set(&f.static_fields),
            set(&f.copies),
            idx,
            helpers,
            counts(&f.field_stores),
            counts(&f.setter_args),
            f.int_defs,
            f.self_steps,
            f.typeof_defs,
            f.index_use as u8,
            f.cmp_int_use as u8,
            ty,
            renamed.contains(name) as u8,
        ));
    }
    if let Ok(mut w) = writer.lock() {
        let _ = w.write_all(out.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn field(base: Expr, f: &str) -> Expr {
        Expr::Field(Box::new(base), f.to_string())
    }

    fn text(stmts: &[Stmt]) -> String {
        let mut s = String::new();
        for st in stmts {
            s.push_str(&format!("{st:?}\n"));
        }
        s
    }

    #[test]
    fn getter_result_gets_property_name() {
        let mut stmts = vec![
            Stmt::Assign(
                "v5".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "GetActiveDeal".into(),
                    args: vec![],
                },
            ),
            Stmt::If {
                cond: Expr::Cmp(
                    super::super::super::csharp::CmpKind::Ne,
                    Box::new(var("v5")),
                    Box::new(Expr::Null),
                ),
                then_body: vec![Stmt::Eval(var("v5"))],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        let t = text(&stmts);
        assert!(t.contains("activeDeal"), "{t}");
        assert!(!t.contains("\"v5\""), "{t}");
    }

    #[test]
    fn field_load_gets_field_name_without_underscore() {
        let mut stmts = vec![
            Stmt::Assign("v3".into(), None, field(Expr::This, "_settings")),
            Stmt::Eval(field(var("v3"), "Count")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        assert!(text(&stmts).contains("\"settings\""));
    }

    #[test]
    fn induction_local_becomes_i() {
        let mut stmts = vec![
            Stmt::Assign("v9".into(), None, Expr::Int(0)),
            Stmt::While {
                cond: Expr::Cmp(
                    super::super::super::csharp::CmpKind::Lt,
                    Box::new(var("v9")),
                    Box::new(var("n")),
                ),
                body: vec![
                    Stmt::Assign(
                        "v9".into(),
                        None,
                        Expr::Bin(BinOp::Add, Box::new(var("v9")), Box::new(Expr::Int(1))),
                    ),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        assert!(text(&stmts).contains("i"));
    }

    #[test]
    fn collision_with_existing_name_takes_a_suffix() {
        let mut stmts = vec![
            Stmt::Assign("v3".into(), None, field(Expr::This, "_settings")),
            Stmt::Eval(var("settings")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        assert!(text(&stmts).contains("settings2"), "{}", text(&stmts));
    }

    #[test]
    fn mixed_provenance_keeps_v_name() {
        let mut stmts = vec![
            Stmt::Assign("v3".into(), None, field(Expr::This, "_a")),
            Stmt::Assign("v3".into(), None, field(Expr::This, "_b")),
            Stmt::Eval(var("v3")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 0);
    }

    #[test]
    fn array_new_result_is_named_array() {
        use crate::recovery::runtime_helpers::{note, RuntimeHelper};
        note(0x991001, RuntimeHelper::ArrayNew);
        let mut stmts = vec![
            Stmt::Assign(
                "v5".into(),
                None,
                Expr::RawCall(0x991001, vec![Expr::Int(4)], None),
            ),
            Stmt::Eval(var("v5")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        assert!(text(&stmts).contains("array"), "{}", text(&stmts));
    }

    #[test]
    fn nonsemantic_call_names_the_result() {
        let mut stmts = vec![
            Stmt::Assign(
                "v5".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "CalculateReward".into(),
                    args: vec![],
                },
            ),
            Stmt::Eval(var("v5")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        assert!(text(&stmts).contains("calculateReward"), "{}", text(&stmts));
    }

    #[test]
    fn generic_getter_name_drops_type_arguments() {
        let mut stmts = vec![
            Stmt::Assign(
                "v7".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::This)),
                    name: "GetComponent<Rigidbody>".into(),
                    args: vec![],
                },
            ),
            Stmt::Eval(var("v7")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        assert!(text(&stmts).contains("component"), "{}", text(&stmts));
    }

    #[test]
    fn mixed_induction_defs_name_the_index() {
        let mut stmts = vec![
            Stmt::Assign("v9".into(), None, Expr::Int(0)),
            Stmt::Assign(
                "v9".into(),
                None,
                Expr::Bin(BinOp::Add, Box::new(var("v9")), Box::new(Expr::Int(1))),
            ),
            Stmt::Eval(Expr::Index {
                collection: Box::new(var("arr")),
                index: Box::new(var("v9")),
            }),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        assert!(text(&stmts).contains("\"i\""), "{}", text(&stmts));
    }

    #[test]
    fn nonzero_seed_induction_is_still_a_counter() {
        // `v = 4` then many `v = v + 1` — a start index, not an offset guess.
        let mut stmts = vec![
            Stmt::Assign("v9".into(), None, Expr::Int(4)),
            Stmt::Assign(
                "v9".into(),
                None,
                Expr::Bin(BinOp::Add, Box::new(var("v9")), Box::new(Expr::Int(1))),
            ),
            Stmt::Assign(
                "v9".into(),
                None,
                Expr::Bin(BinOp::Add, Box::new(var("v9")), Box::new(Expr::Int(1))),
            ),
            Stmt::Eval(var("v9")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        assert!(text(&stmts).contains("\"i\""), "{}", text(&stmts));
    }

    #[test]
    fn for_loop_parent_field_names_the_cursor() {
        // for (v15 = this; v15 != null; v15 = v15.mParentPanel)
        let mut stmts = vec![Stmt::For {
            init: vec![Stmt::Assign("v15".into(), None, Expr::Var("this".into()))],
            cond: Some(Expr::Cmp(
                super::super::super::csharp::CmpKind::Ne,
                Box::new(var("v15")),
                Box::new(Expr::Int(0)),
            )),
            incr: vec![Stmt::Assign("v15".into(), None, field(var("v15"), "mParentPanel"))],
            body: vec![Stmt::Assign("v18".into(), None, Expr::Var("v17".into()))],
        }];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1, "{}", text(&stmts));
        let t = text(&stmts);
        assert!(!t.contains("\"v15\""), "{t}");
        assert!(t.contains("\"mParentPanel\", None, Var(\"this\")"), "{t}");
    }

    #[test]
    fn two_literal_defs_are_not_a_counter() {
        let mut stmts = vec![
            Stmt::Assign("v9".into(), None, Expr::Int(5)),
            Stmt::Assign("v9".into(), None, Expr::Int(10)),
            Stmt::Eval(Expr::Index {
                collection: Box::new(var("arr")),
                index: Box::new(var("v9")),
            }),
        ];
        // Not a counter (`i`) — but its USE as a subscript is a role, so the
        // index fallback names it `index` rather than leaving `v9`.
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        let t = text(&stmts);
        assert!(t.contains("index"), "{t}");
        assert!(!t.contains("\"i\""), "{t}");
    }

    #[test]
    fn sole_type_candidate_names_the_local() {
        let mut stmts = vec![
            Stmt::Assign(
                "v7".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "fn_12ab34".into(),
                    args: vec![],
                },
            ),
            Stmt::Eval(var("v7")),
        ];
        let mut types = HashMap::new();
        types.insert("v7".to_string(), "ListBox".to_string());
        assert_eq!(apply(&mut stmts, &types), 1);
        assert!(text(&stmts).contains("\"listBox\""), "{}", text(&stmts));
    }

    /// A committed concrete class type outweighs a lone weaker role (an index
    /// use). Before this weight, the type hint (1.0) lost to `index` (2.0) and
    /// the local kept its generated name.
    #[test]
    fn committed_class_type_outweighs_a_lone_index_role() {
        let mut stmts = vec![
            // A null def gives no producer evidence, so the only competitors
            // are the committed type (3.0) and the index role (2.0).
            Stmt::Assign("v7".into(), None, Expr::Null),
            Stmt::Eval(Expr::Index {
                collection: Box::new(var("arr")),
                index: Box::new(var("v7")),
            }),
        ];
        let mut types = HashMap::new();
        types.insert("v7".to_string(), "PotionData".to_string());
        assert_eq!(apply(&mut stmts, &types), 1);
        assert!(
            text(&stmts).contains("\"potionData\""),
            "{}",
            text(&stmts)
        );
    }

    #[test]
    fn primitive_type_does_not_name_the_local() {
        let mut stmts = vec![
            Stmt::Assign("v7".into(), None, Expr::Int(0)),
            Stmt::Eval(var("v7")),
        ];
        let mut types = HashMap::new();
        types.insert("v7".to_string(), "System.Int32".to_string());
        assert_eq!(apply(&mut stmts, &types), 0);
    }

    fn loop_counter_body(target: &str) -> Vec<Stmt> {
        vec![
            Stmt::Assign("v124".into(), None, Expr::Int(0)),
            Stmt::While {
                cond: Expr::Cmp(
                    super::super::super::csharp::CmpKind::Lt,
                    Box::new(var("v124")),
                    Box::new(var("length")),
                ),
                body: vec![
                    // The lvalue is a PRE-RENDERED path: the read of v124 lives
                    // inside the string, not in an Expr.
                    Stmt::Assign(
                        target.to_string(),
                        None,
                        Expr::Int(1),
                    ),
                    Stmt::Assign(
                        "v124".into(),
                        None,
                        Expr::Bin(BinOp::Add, Box::new(var("v124")), Box::new(Expr::Int(1))),
                    ),
                ],
            },
        ]
    }

    #[test]
    fn complex_assignment_target_string_is_renamed_with_its_def() {
        // Regression for the undefined-local census: `v124 = 0` was renamed to
        // `j = 0` while every read inside a rendered target stayed `v124`.
        let mut stmts = loop_counter_body("*(int*)(v124 +32)");
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        let t = text(&stmts);
        assert!(t.contains("\"*(int*)(i +32)\""), "{t}");
        assert!(!t.contains("v124"), "{t}");
    }

    #[test]
    fn switch_discriminant_string_is_renamed() {
        let mut stmts = vec![
            Stmt::Assign("v7".into(), None, field(Expr::This, "_settings")),
            Stmt::Eval(var("v7")),
            Stmt::Switch {
                disc: "v7.level".into(),
                cases: vec![(0, vec![Stmt::Break])],
                tail: vec![],
            
                default: None,},
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        let t = text(&stmts);
        assert!(t.contains("\"settings.level\""), "{t}");
        assert!(!t.contains("v7"), "{t}");
    }

    #[test]
    fn foreach_collection_string_is_renamed() {
        let mut stmts = vec![
            Stmt::Assign("v5".into(), None, field(Expr::This, "_items")),
            Stmt::ForEach {
                var_name: "item".into(),
                collection: "v5".into(),
                body: vec![Stmt::Eval(var("item"))],
            },
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        let t = text(&stmts);
        assert!(t.contains("collection: \"items\""), "{t}");
    }

    #[test]
    fn name_present_inside_a_target_string_blocks_the_candidate() {
        let mut stmts = vec![
            Stmt::Assign("v3".into(), None, field(Expr::This, "_settings")),
            Stmt::Eval(var("v3")),
            Stmt::Assign("*(int*)(settings +8)".into(), None, var("v3")),
        ];
        // The target path mentions a DIFFERENT identifier `settings`; the
        // local takes the disambiguated `settings2`, and the raw target path
        // is never rewritten (it names the existing variable).
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        let t = text(&stmts);
        assert!(t.contains("settings2"), "{t}");
        assert!(t.contains("*(int*)(settings +8)"), "{t}");
    }

    /// A copy of MACHINE STORAGE carries no source identity.
    ///
    /// The sole-copy-source rule excluded only `vN`, so `in_x32 = in_x3` (a
    /// local that copies the incoming `x3` register) took the register's own
    /// name, the collision suffixer appended a digit, and the dump grew
    /// variables called `in_x32`/`in_x322` — register numbers that do not
    /// exist on any target. 1,284 defs of an `in_xN` name binary-wide, 707 of
    /// them copies of another `in_xN`.
    #[test]
    fn a_copy_of_a_machine_name_is_not_named_after_the_storage() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, var("in_x3")),
            Stmt::Return(Some(var("v1"))),
        ];
        apply(&mut stmts, &HashMap::new());
        let t = text(&stmts);
        // The RHS legitimately still names the register it reads; what must not
        // happen is the local being NAMED after it (and then suffixed).
        assert!(t.contains("Assign(\"v1\""), "the value keeps its temp name: {t}");
        assert!(!t.contains("in_x32"), "no suffixed register name: {t}");
    }

    #[test]
    fn a_copy_of_a_local_named_in_the_same_pass_is_named_by_the_fixpoint() {
        // Round 1 names v2 from its field; v1's copy source still looks like a
        // temp. Round 2 sees `v1 = playerStats;` and names v1 too (suffixed).
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, var("v2")),
            Stmt::Assign("v2".into(), None, field(Expr::This, "_playerStats")),
        ];
        apply(&mut stmts, &HashMap::new());
        let t = text(&stmts);
        assert!(!t.contains("v1\n") && !t.contains("v2\n"), "{t}");
        assert!(t.contains("playerStats"), "{t}");
    }

    #[test]
    fn an_index_load_is_named_from_its_named_container() {
        let load = Expr::Index {
            collection: Box::new(field(Expr::This, "_unlockInfos")),
            index: Box::new(Expr::Int(0)),
        };
        let mut stmts = vec![
            Stmt::Assign("v3".into(), None, load),
            Stmt::Return(Some(var("v3"))),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        let t = text(&stmts);
        assert!(t.contains("unlockInfos"), "{t}");
    }

    #[test]
    fn an_index_load_over_a_machine_base_abstains() {
        let load = Expr::Index {
            collection: Box::new(var("v9")),
            index: Box::new(Expr::Int(0)),
        };
        let mut stmts = vec![
            Stmt::Assign("v3".into(), None, load),
            Stmt::Return(Some(var("v3"))),
        ];
        assert_eq!(
            apply(&mut stmts, &HashMap::new()),
            0,
            "a vN container carries no source identity"
        );
        assert!(text(&stmts).contains("v3"));
    }

    #[test]
    fn a_type_token_local_is_named_type() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::TypeOf("UIBasicSprite".into())),
            Stmt::Eval(var("v1")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 1);
        let t = text(&stmts);
        assert!(t.contains("\"type\""), "{t}");
    }

    #[test]
    fn a_type_token_mixed_with_a_value_def_is_not_named_type() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::TypeOf("UIBasicSprite".into())),
            Stmt::Assign("v1".into(), None, Expr::Int(3)),
            Stmt::Eval(var("v1")),
        ];
        assert_eq!(apply(&mut stmts, &HashMap::new()), 0);
    }
}
