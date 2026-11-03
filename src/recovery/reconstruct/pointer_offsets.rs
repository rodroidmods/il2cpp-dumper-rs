//! Pointer-offset reconstruction: name the field or element that an address
//! computation proves, using the MANAGED TYPE of the base.
//!
//! ## Why this pass exists (and why field_annotation could not do it)
//!
//! field_annotation resolves a raw deref through ONE map:
//! local_types: name -> TypeDef index, built by build_local_tdi_map, which
//! keeps only ManagedType::Class locals. Everything else is invisible to it:
//!
//!   * string            -- no TypeDef at all (ManagedType::String);
//!   * T[]               -- ManagedType::SzArray, no TypeDef;
//!   * List<object> and friends -- ManagedType::Opaque, because a generic
//!     INSTANCE has no TypeDef of its own.
//!
//! So a local whose type the resolver DID prove still arrives at
//! field_annotation as "notype" whenever that type is an array, a string or a
//! generic instance. Measured on R74 (41,127 raw nint derefs, 15,791 of them
//! with a constant addend): the 'list +16' / 'list +24' / 'list +28' cluster
//! is 1,000+ sites and every one of them is _items / Count / _version.
//!
//! This pass answers the questions the tdi map cannot:
//!
//!   * is the base a ref/out PARAMETER read through its own address (offset 0)?
//!   * is the base a string, and is the offset the length slot (+16)?
//!   * is the base an array, and is the offset the length slot (+24) or an
//!     element (+32 + k*elem_size)?
//!   * is the base a List<T> instance, and is the offset one of its three
//!     instantiation-independent fields?
//!
//! Every rewrite is gated on a fact from metadata or on a layout the runtime
//! fixes (string and array headers, List<T>'s field set). An offset that
//! cannot be proved stays raw.
//!
//! LOAD POSITION ONLY. A store target is a STRING in this IR
//! (Stmt::Assign(name, ..)), not an expression, so store targets are
//! store_sugar's job; this pass never touches them.

use super::super::csharp::{BinOp, Expr, Stmt};
use super::super::resolve::ResolveMaps;
use super::super::types::ManagedType;
use super::walk_mut;
use std::collections::{HashMap, HashSet};

/// Arity separator in a metadata generic type name (List + sep + 1).
const ARITY_SEP: char = '\x60';

// ---------------------------------------------------------------------------
// Resolved type of an expression, in the form the layout questions need
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Ty {
    /// string
    Str,
    /// T[] -- element type and its byte size when proved.
    Array(Box<Ty>, Option<u64>),
    /// System.Collections.Generic.List<T>, any instantiation.
    List,
    /// A class or value type with a TypeDef, plus the display name.
    Named { tdi: i32, name: String },
    /// A klass token (typeof(T)) -- the base of an Il2CppClass offset.
    Klass { tdi: i32, name: String },
    /// Nothing proved.
    Other,
}

/// Split a generic display name into its definition name and arguments.
/// "Ns.Name<A, B>" -> ("Ns.Name", ["A", "B"]).
fn split_generic(text: &str) -> Option<(String, Vec<String>)> {
    let open = text.find('<')?;
    let close = text.rfind('>')?;
    if close < open {
        return None;
    }
    let base = text[..open].trim().to_string();
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in text[open + 1..close].chars() {
        match ch {
            '<' | '[' => {
                depth += 1;
                cur.push(ch);
            }
            '>' | ']' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                args.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        args.push(cur.trim().to_string());
    }
    Some((base, args))
}

/// Is this display name System.Collections.Generic.List<T> (or List)?
///
/// The definition's own name is List + arity sep + 1; a display name from
/// ManagedType::Opaque is the fully spelled instance (List<object>).
fn is_list_name(text: &str) -> bool {
    let arity_name = format!("List{ARITY_SEP}1");
    match split_generic(text) {
        // An INSTANCE (List<object>): the definition's short name is "List".
        Some((base, _)) => {
            let short = base.rsplit('.').next().unwrap_or(&base);
            short == "List" || short == arity_name
        }
        // A DEFINITION is only List when the arity separator is present. A
        // bare "List" could be an unrelated user type, so it is refused.
        None => text.rsplit('.').next() == Some(arity_name.as_str()),
    }
}

/// Byte size of a type by display name, from metadata (named_type_sizes).
fn size_of_name(name: &str, maps: &ResolveMaps) -> Option<u64> {
    match name {
        "bool" | "byte" | "sbyte" | "System.Boolean" | "System.Byte" | "System.SByte" => Some(1),
        "short" | "ushort" | "char" | "System.Int16" | "System.UInt16" | "System.Char" => Some(2),
        "int" | "uint" | "float" | "System.Int32" | "System.UInt32" | "System.Single" => Some(4),
        "long" | "ulong" | "double" | "System.Int64" | "System.UInt64" | "System.Double" => Some(8),
        "IntPtr" | "UIntPtr" | "System.IntPtr" | "System.UIntPtr" => Some(8),
        "object" | "string" | "System.Object" | "System.String" => Some(8),
        _ => maps.named_type_sizes.get(name).copied(),
    }
}

/// ManagedType -> Ty.
pub(crate) fn ty_of_managed(mt: &ManagedType, maps: &ResolveMaps) -> Ty {
    match mt {
        ManagedType::String => Ty::Str,
        ManagedType::SzArray(elem) => {
            let inner = ty_of_managed(elem, maps);
            let size = size_of_name(&elem.display_name(), maps);
            Ty::Array(Box::new(inner), size)
        }
        // A byref holds the ADDRESS of a T, so an offset off it addresses a
        // field of that same T. Only offset 0 is ambiguous (value vs. first
        // field) and that case is gated on the parameter modifier instead.
        ManagedType::ByRef(inner) => ty_of_managed(inner, maps),
        ManagedType::Class(named) if named.type_def_index >= 0 => Ty::Named {
            tdi: named.type_def_index,
            name: named.to_string(),
        },
        ManagedType::Class(named) => match tdi_for_name(&named.to_string(), maps) {
            Some(tdi) => Ty::Named {
                tdi,
                name: named.to_string(),
            },
            None => Ty::Other,
        },
        ManagedType::Object => Ty::Named {
            tdi: -1,
            name: "object".into(),
        },
        ManagedType::Opaque(text) => {
            // A generic INSTANCE: List<T> has three fields, none of them
            // inline, so the definition's layout IS the instance's layout for
            // every instantiation. Any other generic instance would need the
            // definition's TypeDef, which ResolveMaps cannot name (see the
            // design doc, section 7).
            if is_list_name(text) {
                Ty::List
            } else {
                Ty::Other
            }
        }
        _ => Ty::Other,
    }
}

fn tdi_for_name(name: &str, maps: &ResolveMaps) -> Option<i32> {
    maps.tdi_names_reverse.get(name).copied()
}

/// Field name at off on the type or one of its ancestors.
fn field_at(maps: &ResolveMaps, tdi: i32, off: i64) -> Option<String> {
    if tdi < 0 {
        return None;
    }
    let mut chain = maps
        .type_parent_chains
        .get(&tdi)
        .cloned()
        .unwrap_or_default();
    if !chain.contains(&tdi) {
        chain.push(tdi);
    }
    for p in chain {
        if let Some((name, _)) = maps.field_layouts.get(&(p, off)) {
            return Some(name.clone());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Rewrite
// ---------------------------------------------------------------------------

/// Everything the rewrite needs to know about one method.
pub(crate) struct Ctx<'a> {
    pub maps: &'a ResolveMaps,
    pub local_types: &'a HashMap<String, i32>,
    pub local_mty: &'a HashMap<String, ManagedType>,
    /// Names of the method's byref parameters (ref/out/in in the signature).
    /// A deref at offset 0 of one of these is the parameter's own value.
    pub ref_params: &'a HashSet<String>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct PoCounts {
    pub ref_param: usize,
    pub str_len: usize,
    pub arr_len: usize,
    pub arr_elem: usize,
    pub list: usize,
}

impl PoCounts {
    pub fn total(&self) -> usize {
        self.ref_param + self.str_len + self.arr_len + self.arr_elem + self.list
    }
}

/// Peel + constant / - constant off an address expression.
pub(crate) fn strip_const_addend(expr: &Expr) -> (&Expr, i64) {
    let mut cur = expr;
    let mut off: i64 = 0;
    loop {
        match cur {
            Expr::Bin(BinOp::Add, a, b) => {
                if let Expr::Int(k) = b.as_ref() {
                    off = off.wrapping_add(*k);
                    cur = a;
                } else if let Expr::Int(k) = a.as_ref() {
                    off = off.wrapping_add(*k);
                    cur = b;
                } else {
                    break;
                }
            }
            Expr::Bin(BinOp::Sub, a, b) => {
                if let Expr::Int(k) = b.as_ref() {
                    off = off.wrapping_sub(*k);
                    cur = a;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    (cur, off)
}

/// Width 0 in the IR means "pointer width / not stated", i.e. 8 bytes.
const W_PTR: u8 = 0;

fn width_ok(w: u8, want: u64) -> bool {
    if w == W_PTR {
        want == 8
    } else {
        w as u64 == want
    }
}

/// List<T> internals. List<T> has exactly three instance fields and none of
/// them is an inline T, so these offsets do not depend on the instantiation.
/// The same table is used by field_annotation::try_resolve_collection_internal
/// and store_sugar::collection_internal_name.
fn list_field(total: i64, w: u8) -> Option<&'static str> {
    match (total, w) {
        (16, W_PTR | 8) => Some("_items"),
        (24, W_PTR | 4) => Some("Count"),
        (28, W_PTR | 4) => Some("_version"),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Rule {
    RefParam,
    StrLen,
    ArrLen,
    ArrElem,
    List,
}

/// Try to rewrite one deref. Returns the replacement and which rule fired.
fn rewrite_deref(e: &Expr, ctx: &Ctx) -> Option<(Expr, Rule)> {
    let Expr::RawDeref(base, off, w) = e else {
        return None;
    };
    // R1: offset 0 of a byref PARAMETER is the parameter itself. A holder of
    // &local at offset 0 would be that local's first field instead, so this is
    // gated on the signature's modifier, never on ByRef-ness alone.
    if *off == 0 {
        if let Expr::Var(n) = base.as_ref() {
            if ctx.ref_params.contains(n) {
                let reference_pointee = matches!(
                    ctx.local_mty.get(n),
                    Some(ManagedType::ByRef(inner)) if inner.unknown_is_reference()
                );
                if reference_pointee && (*w == W_PTR || *w == 8) {
                    return Some((Expr::Var(n.clone()), Rule::RefParam));
                }
            }
        }
    }
    let (root, addend) = strip_const_addend(base);
    let total = addend.wrapping_add(*off);
    if total == 0 {
        return None;
    }
    match ty_of_expr(root, ctx) {
        // Il2CppString::length and Il2CppArray::max_length are 4-byte ints.
        // A pointer-width read at those offsets would span the following
        // field, so the width must say "int" exactly.
        Ty::Str if total == 16 && *w == 4 => Some((
            Expr::Field(Box::new(root.clone()), "Length".into()),
            Rule::StrLen,
        )),
        Ty::Array(_, _) if total == 24 && *w == 4 => Some((
            Expr::Field(Box::new(root.clone()), "Length".into()),
            Rule::ArrLen,
        )),
        Ty::Array(_, Some(size)) if total >= 32 && (total - 32) % (size as i64) == 0 => {
            if !width_ok(*w, size) {
                return None;
            }
            let k = (total - 32) / size as i64;
            Some((
                Expr::Index {
                    collection: Box::new(root.clone()),
                    index: Box::new(Expr::Int(k)),
                },
                Rule::ArrElem,
            ))
        }
        Ty::List => list_field(total, *w).map(|name| {
            (
                Expr::Field(Box::new(root.clone()), name.to_string()),
                Rule::List,
            )
        }),
        _ => None,
    }
}

fn rewrite_expr(e: &mut Expr, ctx: &Ctx, out: &mut PoCounts) {
    for k in expr_children_mut(e) {
        rewrite_expr(k, ctx, out);
    }
    let Some((new, rule)) = rewrite_deref(e, ctx) else {
        return;
    };
    *e = new;
    match rule {
        Rule::RefParam => out.ref_param += 1,
        Rule::StrLen => out.str_len += 1,
        Rule::ArrLen => out.arr_len += 1,
        Rule::ArrElem => out.arr_elem += 1,
        Rule::List => out.list += 1,
    }
}

/// Rewrite every load-position raw deref in one method body.
pub(crate) fn rewrite_stmts(stmts: &mut Vec<Stmt>, ctx: &Ctx) -> PoCounts {
    let mut out = PoCounts::default();
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            rewrite_stmt(stmt, ctx, &mut out);
        }
    });
    out
}

fn rewrite_stmt(stmt: &mut Stmt, ctx: &Ctx, out: &mut PoCounts) {
    match stmt {
        Stmt::Assign(_, _, e)
        | Stmt::Eval(e)
        | Stmt::Return(Some(e))
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e) => rewrite_expr(e, ctx, out),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            rewrite_expr(cond, ctx, out)
        }
        Stmt::Lock { expr, .. } => rewrite_expr(expr, ctx, out),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            rewrite_expr(collection, ctx, out);
            rewrite_expr(index, ctx, out);
            rewrite_expr(value, ctx, out);
        }
        Stmt::EventAssign { field, value, .. } => {
            rewrite_expr(field, ctx, out);
            rewrite_expr(value, ctx, out);
        }
        _ => {}
    }
}

fn expr_children_mut(expr: &mut Expr) -> Vec<&mut Expr> {
    match expr {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => vec![a, b],
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) | Expr::AddressOf(a) | Expr::Await(a)
        | Expr::RawAddr(a, _) => vec![a],
        Expr::RawDeref(a, _, _) | Expr::Field(a, _) => vec![a],
        Expr::Cond(c, a, b) => vec![c, a, b],
        Expr::Call { recv, args, .. } => {
            let mut v: Vec<&mut Expr> = Vec::new();
            if let Some(r) = recv {
                v.push(r);
            }
            for a in args.iter_mut() {
                v.push(a);
            }
            v
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            args.iter_mut().collect()
        }
        Expr::Index { collection, index } => vec![collection, index],
        _ => Vec::new(),
    }
}

/// Type of an expression, in layout terms.
pub(crate) fn ty_of_expr(expr: &Expr, ctx: &Ctx) -> Ty {
    match expr {
        Expr::This => ty_of_var("this", ctx),
        Expr::Var(name) => ty_of_var(name, ctx),
        Expr::Cast(name, _) => match tdi_for_name(name, ctx.maps) {
            Some(tdi) => Ty::Named {
                tdi,
                name: name.clone(),
            },
            None => {
                if is_list_name(name) {
                    Ty::List
                } else {
                    Ty::Other
                }
            }
        },
        Expr::TypeOf(name) => match tdi_for_name(name, ctx.maps) {
            Some(tdi) => Ty::Klass {
                tdi,
                name: name.clone(),
            },
            None => Ty::Other,
        },
        Expr::NewArr(elem, _) => {
            let size = size_of_name(elem, ctx.maps);
            Ty::Array(Box::new(ty_of_name(elem, ctx.maps)), size)
        }
        _ => Ty::Other,
    }
}

fn ty_of_var(name: &str, ctx: &Ctx) -> Ty {
    if let Some(mt) = ctx.local_mty.get(name) {
        let t = ty_of_managed(mt, ctx.maps);
        if t != Ty::Other {
            return t;
        }
    }
    match ctx.local_types.get(name) {
        Some(&tdi) if tdi >= 0 => Ty::Named {
            tdi,
            name: ctx
                .maps
                .tdi_names
                .get(&tdi)
                .map(|(ns, nm)| {
                    if ns.is_empty() {
                        nm.clone()
                    } else {
                        format!("{ns}.{nm}")
                    }
                })
                .unwrap_or_default(),
        },
        _ => Ty::Other,
    }
}

fn ty_of_name(name: &str, maps: &ResolveMaps) -> Ty {
    if name == "string" || name == "System.String" {
        return Ty::Str;
    }
    if is_list_name(name) {
        return Ty::List;
    }
    match tdi_for_name(name, maps) {
        Some(tdi) => Ty::Named {
            tdi,
            name: name.to_string(),
        },
        None => Ty::Other,
    }
}

// ---------------------------------------------------------------------------
// Census (diagnostic, ILR_POCENSUS=1)
// ---------------------------------------------------------------------------

fn census_enabled() -> bool {
    std::env::var("ILR_POCENSUS").is_ok()
}

/// Method RVA of the body being censused (diagnostic only).
static CENSUS_RVA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn shape(e: &Expr) -> String {
    fn go(e: &Expr, d: usize) -> String {
        if d > 3 {
            return "…".into();
        }
        match e {
            Expr::Var(n) => format!("Var({n})"),
            Expr::This => "This".into(),
            Expr::Int(k) => format!("Int({k})"),
            Expr::Field(b, n) => format!("Field({},{n})", go(b, d + 1)),
            Expr::StaticField(t, n) => format!("Static({t}.{n})"),
            Expr::RawDeref(b, o, w) => format!("Deref({},+{o},w{w})", go(b, d + 1)),
            Expr::RawAddr(b, o) => format!("Addr({},+{o})", go(b, d + 1)),
            Expr::AddressOf(b) => format!("&({})", go(b, d + 1)),
            Expr::Cast(t, _) => format!("Cast({t})"),
            Expr::Call { name, .. } => format!("Call({name})"),
            Expr::Bin(op, a, b) => format!("Bin({op:?},{},{})", go(a, d + 1), go(b, d + 1)),
            Expr::TypeOf(t) => format!("typeof({t})"),
            _ => "other".into(),
        }
    }
    go(e, 0)
}

fn census(bucket: &str, e: &Expr) {
    let rva = CENSUS_RVA.load(std::sync::atomic::Ordering::Relaxed);
    if !census_enabled() {
        return;
    }
    if let Expr::RawDeref(base, off, w) = e {
        let mut text = String::new();
        e.render(&mut text);
        eprintln!(
            "[po] {rva:#x} {bucket} off={} w={} root={} base={} text={}",
            off,
            w,
            shape(strip_const_addend(base).0),
            shape(base),
            &text[..text.len().min(120)]
        );
    }
}

/// Classify every SURVIVING raw deref in one method body. Runs after
/// rewrite_stmts, so the buckets are what the pass could NOT prove — the
/// ceiling measurement.
pub(crate) fn census_stmts(stmts: &mut Vec<Stmt>, ctx: &Ctx, rva: u64) -> usize {
    if !census_enabled() {
        return 0;
    }
    CENSUS_RVA.store(rva, std::sync::atomic::Ordering::Relaxed);
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            n += census_stmt(stmt, ctx);
        }
    });
    n
}

fn census_stmt(stmt: &mut Stmt, ctx: &Ctx) -> usize {
    let mut n = 0;
    match stmt {
        Stmt::Assign(_, _, e) | Stmt::Eval(e) | Stmt::Return(Some(e)) | Stmt::Throw(e)
        | Stmt::YieldReturn(e) => n += census_expr(e, ctx),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            n += census_expr(cond, ctx)
        }
        Stmt::Lock { expr, .. } => n += census_expr(expr, ctx),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            n += census_expr(collection, ctx);
            n += census_expr(index, ctx);
            n += census_expr(value, ctx);
        }
        Stmt::EventAssign { field, value, .. } => {
            n += census_expr(field, ctx);
            n += census_expr(value, ctx);
        }
        _ => {}
    }
    // NO recursion into nested lists here: walk_mut in census_stmts already
    // visits every statement list exactly once, so recursing again would count
    // a deeply nested deref once per enclosing block.
    n
}

fn census_expr(expr: &mut Expr, ctx: &Ctx) -> usize {
    let mut n = 0;
    for k in expr_children_mut(expr) {
        n += census_expr(k, ctx);
    }
    if let Expr::RawDeref(base, off, _w) = expr {
        let (root, folded) = strip_const_addend(base);
        let total = folded.wrapping_add(*off);
        let ty = ty_of_expr(root, ctx);
        // For an untyped base, say WHY: no managed type at all, or one the
        // layout questions cannot use (ByRef, Pointer, primitive, Unknown).
        let why = match root {
            Expr::Var(n) => match ctx.local_mty.get(n) {
                None => "no_mty".to_string(),
                Some(mt) => format!("mty={mt:?}"),
            },
            _ => String::new(),
        };
        let bucket = match &ty {
            Ty::Other => format!("notype({why})"),
            Ty::Str => match total {
                16 => "str_len".to_string(),
                _ => "str_other".to_string(),
            },
            Ty::Array(_, size) => match (*size, total) {
                (_, 24) => "arr_len".to_string(),
                (Some(es), o) if o >= 32 && (o - 32) % (es as i64) == 0 => "arr_data".to_string(),
                _ => "arr_other".to_string(),
            },
            Ty::List => "list_other".to_string(),
            Ty::Klass { .. } => "klass_meta".to_string(),
            Ty::Named { tdi, .. } => {
                if total == 0 {
                    "ref_klass0".to_string()
                } else if field_at(ctx.maps, *tdi, total).is_some() {
                    "field_exists".to_string()
                } else {
                    "no_field".to_string()
                }
            }
        };
        census(&bucket, expr);
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::types::PrimKind;

    fn run(
        stmts: &mut Vec<Stmt>,
        maps: &ResolveMaps,
        lt: &HashMap<String, i32>,
        mty: &HashMap<String, ManagedType>,
        refs: &HashSet<String>,
    ) -> PoCounts {
        let ctx = Ctx {
            maps,
            local_types: lt,
            local_mty: mty,
            ref_params: refs,
        };
        rewrite_stmts(stmts, &ctx)
    }

    fn text(e: &Expr) -> String {
        let mut s = String::new();
        e.render(&mut s);
        s
    }

    fn value(stmts: &Vec<Stmt>) -> String {
        match &stmts[0] {
            Stmt::Assign(_, _, e) => text(e),
            other => panic!("expected an assign, got {other:?}"),
        }
    }

    fn one(e: Expr) -> Vec<Stmt> {
        vec![Stmt::Assign("dst".into(), None, e)]
    }

    fn deref(base: Expr, off: i64, w: u8) -> Expr {
        Expr::RawDeref(Box::new(base), off, w)
    }

    fn var(n: &str) -> Expr {
        Expr::Var(n.into())
    }

    fn typed(name: &str, mt: ManagedType) -> HashMap<String, ManagedType> {
        let mut m = HashMap::new();
        m.insert(name.to_string(), mt);
        m
    }

    // ---- R1: a byref parameter at offset 0 is the parameter ----------------

    #[test]
    fn ref_param_deref_collapses() {
        let maps = ResolveMaps::default();
        let lt = HashMap::new();
        let mty = typed(
            "values",
            ManagedType::ByRef(Box::new(ManagedType::Opaque("List<string>".into()))),
        );
        let mut refs = HashSet::new();
        refs.insert("values".to_string());

        let mut stmts = one(deref(var("values"), 0, 0));
        assert_eq!(run(&mut stmts, &maps, &lt, &mty, &refs).ref_param, 1);
        assert_eq!(value(&stmts), "values");

        // NEUTRALISATION: the same shape without the signature's modifier is
        // not a byref parameter — a local holding an address at offset 0 is
        // that local's first field, not the local.
        let none = HashSet::new();
        let mut raw = one(deref(var("values"), 0, 0));
        assert_eq!(run(&mut raw, &maps, &lt, &mty, &none).total(), 0);
        assert_eq!(value(&raw), "*(nint*)(values)");
    }

    #[test]
    fn ref_param_of_a_value_type_is_refused() {
        let maps = ResolveMaps::default();
        let lt = HashMap::new();
        let mty = typed(
            "n",
            ManagedType::ByRef(Box::new(ManagedType::Primitive(PrimKind::I4))),
        );
        let mut refs = HashSet::new();
        refs.insert("n".to_string());
        let mut stmts = one(deref(var("n"), 0, 0));
        assert_eq!(run(&mut stmts, &maps, &lt, &mty, &refs).total(), 0);
    }

    // ---- R2: string length -------------------------------------------------

    #[test]
    fn string_length_slot() {
        let maps = ResolveMaps::default();
        let lt = HashMap::new();
        let mty = typed("s", ManagedType::String);

        let mut ok = one(deref(var("s"), 16, 4));
        assert_eq!(run(&mut ok, &maps, &lt, &mty, &HashSet::new()).str_len, 1);
        assert_eq!(value(&ok), "s.Length");

        // +24 is not a slot of Il2CppString.
        let mut bad = one(deref(var("s"), 24, 4));
        assert_eq!(run(&mut bad, &maps, &lt, &mty, &HashSet::new()).total(), 0);

        // An 8-byte read at the 4-byte length slot is not the length.
        let mut wide = one(deref(var("s"), 16, 0));
        assert_eq!(run(&mut wide, &maps, &lt, &mty, &HashSet::new()).total(), 0);
    }

    // ---- R3/R4: array length and element -----------------------------------

    #[test]
    fn array_length_and_element() {
        let maps = ResolveMaps::default();
        let lt = HashMap::new();

        // object[]: element size 8, length at +24.
        let mty = typed(
            "a",
            ManagedType::SzArray(Box::new(ManagedType::Object)),
        );
        let mut len = one(deref(var("a"), 24, 4));
        assert_eq!(run(&mut len, &maps, &lt, &mty, &HashSet::new()).arr_len, 1);
        assert_eq!(value(&len), "a.Length");

        let mut el = one(deref(var("a"), 48, 0));
        assert_eq!(run(&mut el, &maps, &lt, &mty, &HashSet::new()).arr_elem, 1);
        assert_eq!(value(&el), "a[2]");

        // int[]: element size 4.
        let mty_i = typed(
            "b",
            ManagedType::SzArray(Box::new(ManagedType::Primitive(PrimKind::I4))),
        );
        let mut el_i = one(deref(var("b"), 40, 4));
        assert_eq!(run(&mut el_i, &maps, &lt, &mty_i, &HashSet::new()).arr_elem, 1);
        assert_eq!(value(&el_i), "b[2]");

        // NEUTRALISATION: a 4-byte read inside an 8-byte element is a sub-field
        // of the element, not the element.
        let mut wrong = one(deref(var("a"), 48, 4));
        assert_eq!(run(&mut wrong, &maps, &lt, &mty, &HashSet::new()).total(), 0);

        // A non-element offset (the bounds pointer at +16) stays raw.
        let mut bounds = one(deref(var("a"), 16, 0));
        assert_eq!(run(&mut bounds, &maps, &lt, &mty, &HashSet::new()).total(), 0);
    }

    // ---- R5: List<T> internals --------------------------------------------

    #[test]
    fn list_internals() {
        let maps = ResolveMaps::default();
        let lt = HashMap::new();
        let mty = typed("list", ManagedType::Opaque("List<object>".into()));

        let mut items = one(deref(var("list"), 16, 0));
        assert_eq!(run(&mut items, &maps, &lt, &mty, &HashSet::new()).list, 1);
        assert_eq!(value(&items), "list._items");

        let mut count = one(deref(var("list"), 24, 4));
        assert_eq!(run(&mut count, &maps, &lt, &mty, &HashSet::new()).list, 1);
        assert_eq!(value(&count), "list.Count");

        let mut ver = one(deref(var("list"), 28, 4));
        assert_eq!(run(&mut ver, &maps, &lt, &mty, &HashSet::new()).list, 1);
        assert_eq!(value(&ver), "list._version");

        // NEUTRALISATION: +32 is not a List<T> field, and a non-List generic
        // instance is not claimed at all.
        let mut beyond = one(deref(var("list"), 32, 0));
        assert_eq!(run(&mut beyond, &maps, &lt, &mty, &HashSet::new()).total(), 0);

        let dict = typed(
            "d",
            ManagedType::Opaque("Dictionary<int, object>".into()),
        );
        let mut not_list = one(deref(var("d"), 16, 0));
        assert_eq!(run(&mut not_list, &maps, &lt, &dict, &HashSet::new()).total(), 0);
    }

    #[test]
    fn byref_unwrap_for_fields_but_not_for_offset_zero() {
        let maps = ResolveMaps::default();
        let lt = HashMap::new();
        let mty = typed(
            "l",
            ManagedType::ByRef(Box::new(ManagedType::Opaque("List<int>".into()))),
        );

        // A byref holds the address of a List<T>, so +16 is its _items.
        let mut items = one(deref(var("l"), 16, 0));
        assert_eq!(run(&mut items, &maps, &lt, &mty, &HashSet::new()).list, 1);
        assert_eq!(value(&items), "l._items");

        // Offset 0 without the parameter modifier is NOT the value.
        let mut zero = one(deref(var("l"), 0, 0));
        assert_eq!(run(&mut zero, &maps, &lt, &mty, &HashSet::new()).total(), 0);
    }

    #[test]
    fn constant_addend_is_folded_before_the_lookup() {
        let maps = ResolveMaps::default();
        let lt = HashMap::new();
        let mty = typed("list", ManagedType::Opaque("List<object>".into()));
        // (list + 8) + 8 == +16.
        let base = Expr::Bin(
            BinOp::Add,
            Box::new(Expr::Bin(
                BinOp::Add,
                Box::new(var("list")),
                Box::new(Expr::Int(8)),
            )),
            Box::new(Expr::Int(8)),
        );
        let mut stmts = one(deref(base, 0, 0));
        assert_eq!(run(&mut stmts, &maps, &lt, &mty, &HashSet::new()).list, 1);
        assert_eq!(value(&stmts), "list._items");
    }
}
