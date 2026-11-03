//! Recover the C# type of a local from the statements that define and use it.
//!
//! `local_mty` (the metadata map) only knows a local's type when the IL local
//! signature was carried through — for most recovered temporaries it is
//! `Unknown`. That leaves the renderer with no declaration to emit and the
//! pointer passes with no pointee, which is why 23k locals were undeclared and
//! 45k derefs stayed `*(nint*)`.
//!
//! This pass reads the type off the DEFINITIONS instead. A local written only
//! from `Expr::Int` is an `int`; one written from `new Foo(…)` is a `Foo`; one
//! written from a `Cast("Vector3", …)` is a `Vector3`. The rules are:
//!
//! * Every TYPED write must agree. An untyped write (a call with no recovered
//!   return type, a field load) is ignored — it carries no information, and
//!   refusing the whole local for it would throw away the typed ones.
//! * A local that is ever used as a pointer base (`*(nint*)(v + 8)`,
//!   `(nint*)(v + 8)`) cannot be a numeric primitive: it becomes `nint`,
//!   which is the C# type pointer arithmetic needs.
//! * A local used as a field base (`v.field`) cannot be a primitive; when the
//!   definitions say `int` and a use says "has fields", the local is skipped
//!   rather than declared wrong.
//!
//! Everything else is left alone: a wrong declaration is worse than none,
//! because it turns a compile error into a silently different program.

use std::collections::{HashMap, HashSet};

use crate::recovery::csharp::{Expr, Stmt};

use super::float_locals::{for_each_expr, F32_WIDTH};
use super::il2cpp_cleanup::expr_children;

#[derive(Default)]
struct Collector {
    /// Inferred type of every write to each name (`None` = the expression
    /// carries no type information).
    writes: HashMap<String, Vec<Option<String>>>,
    /// Names used as the BASE of a raw pointer deref / address computation.
    pointer_base: HashSet<String>,
    /// Names used as the receiver of a field access.
    field_base: HashSet<String>,
    /// `a = b` edges, so a type can flow through a copy chain.
    copies: HashMap<String, HashSet<String>>,
    /// Types proven by a USE position that admits only one C# type.
    use_ty: HashMap<String, Vec<String>>,
}


/// Method name -> return type, for names whose every definition agrees.
///
/// `Expr::Call` carries no return type, which is why a local written only from
/// calls could not be declared. The resolved signatures are already in
/// `ResolveMaps`; this indexes them by the method name the renderer emits and
/// keeps only names where every overload returns the same type (so `Add`
/// stays unusable while `get_Length` becomes `int`).
/// Cached: the map is identical for every method, and building it walks
/// every resolved signature (~40k). Recomputing it per method was 25k*40k.
pub fn call_return_types_cached(
    maps: &crate::recovery::resolve::ResolveMaps,
) -> &'static HashMap<String, String> {
    static CACHE: std::sync::OnceLock<HashMap<String, String>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| call_return_types(maps))
}

pub fn call_return_types(maps: &crate::recovery::resolve::ResolveMaps) -> HashMap<String, String> {
    let mut acc: HashMap<String, Option<String>> = HashMap::new();
    let mut add = |name: &str, ret: &crate::recovery::types::ManagedType| {
        let Some(ty) = crate::recovery::csharp::csharp_type(ret) else {
            acc.insert(name.to_string(), None);
            return;
        };
        match acc.get_mut(name) {
            None => {
                acc.insert(name.to_string(), Some(ty));
            }
            Some(slot) => match slot {
                Some(cur) if *cur == ty => {}
                _ => *slot = None,
            },
        }
    };
    for m in maps.methods_by_rva.values() {
        add(&m.name, &m.sig.ret);
    }
    for list in maps.shared_by_rva.values() {
        for m in list {
            add(&m.name, &m.sig.ret);
        }
    }
    acc.into_iter()
        .filter_map(|(k, v)| v.map(|t| (k, t)))
        .collect()
}


/// Infer a C# type for every local the statements prove one for.
pub fn infer(stmts: &[Stmt], ret: &HashMap<String, String>) -> HashMap<String, String> {
    let mut c = Collector::default();
    walk(stmts, &mut c, ret);
    c.finish()
}

fn walk(stmts: &[Stmt], c: &mut Collector, ret: &HashMap<String, String>) {
    for s in stmts {
        if let Stmt::Assign(name, _, rhs) = s {
            let t = ty_of(rhs, ret);
            c.writes.entry(name.clone()).or_default().push(t);
            // `a = b` carries b's type to a; the fixpoint in `finish` walks
            // these edges. Most recovered locals are copies of another local.
            if let Expr::Var(src) = rhs {
                if src != "this" && src != "base" && src != name {
                    c.copies.entry(name.clone()).or_default().insert(src.clone());
                }
            }
        }
        // A bare local in a CONDITION must be a `bool`: C# has no implicit
        // truthiness. Definitions leave such a local untyped whenever every
        // write to it is a field load or an unresolved call.
        match s {
            Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
                note_cond(cond, c)
            }
            Stmt::For { cond: Some(cond), .. } => note_cond(cond, c),
            _ => {}
        }
        for_each_expr(s, &mut |e| note_uses(e, c));
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            walk(k, c, ret);
        }
    }
}

fn note_uses(e: &Expr, c: &mut Collector) {
    match e {
        Expr::RawDeref(base, _, _) | Expr::RawAddr(base, _) => {
            if let Expr::Var(n) = base.as_ref() {
                c.pointer_base.insert(n.clone());
            }
        }
        Expr::Field(base, _) => {
            if let Expr::Var(n) = base.as_ref() {
                c.field_base.insert(n.clone());
            }
        }
        // `!v` and `c ? a : b` are bool positions too.
        Expr::Not(inner) => {
            if let Expr::Var(n) = inner.as_ref() {
                note_var_bool(n, c);
            }
        }
        Expr::Cond(cond, _, _) => {
            if let Expr::Var(n) = cond.as_ref() {
                note_var_bool(n, c);
            }
        }
        _ => {}
    }
    for k in expr_children(e) {
        note_uses(k, c);
    }
}

fn note_cond(cond: &Expr, c: &mut Collector) {
    if let Expr::Var(n) = cond {
        note_var_bool(n, c);
    }
}

fn note_var_bool(name: &str, c: &mut Collector) {
    if name == "this" || name == "base" {
        return;
    }
    c.use_ty
        .entry(name.to_string())
        .or_default()
        .push("bool".to_string());
}

impl Collector {
    fn finish(self) -> HashMap<String, String> {
        let mut out = self.by_writes();
        // COPY CHAINS: `a = b` where `b` is typed. Iterated to a fixpoint
        // because the chains nest (`a = b; b = c`). A name that already has a
        // type is never overwritten, and a name whose sources disagree stays
        // untyped.
        self.propagate(&mut out);
        // USES that admit exactly one C# type. Only after the definitions
        // have spoken, and only for names they left untyped.
        for (name, ty) in self.by_uses(&out) {
            out.insert(name, ty);
        }
        self.propagate(&mut out);
        out
    }

    /// Flow types along `a = b` edges until nothing new is learned.
    fn propagate(&self, out: &mut HashMap<String, String>) {
        for _ in 0..8 {
            let mut grew = false;
            for (name, srcs) in &self.copies {
                if out.contains_key(name) {
                    continue;
                }
                let mut ty: Option<&String> = None;
                let mut ok = true;
                for s in srcs {
                    let Some(t) = out.get(s) else {
                        ok = false;
                        break;
                    };
                    match ty {
                        None => ty = Some(t),
                        Some(cur) if cur == t => {}
                        Some(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    if let Some(t) = ty {
                        out.insert(name.clone(), t.clone());
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
        }
    }

    /// Types proven by USE, for names no definition types.
    ///
    /// The evidence here is deliberately narrow: only positions where C#
    /// admits a single type. A bare local in a condition must be `bool`, so a
    /// `bool` declaration can never be the wrong type in a way that compiles —
    /// a recovered `if (v)` that was really an integer comparison fails to
    /// compile, which is loud. Anything wider (a local compared with `0` could
    /// be `int`, `uint`, `long` or a pointer) is NOT inferred: a silently
    /// truncated `long` would be a different program.
    fn by_uses(&self, out: &HashMap<String, String>) -> Vec<(String, String)> {
        let mut add = Vec::new();
        for (name, ev) in &self.use_ty {
            if out.contains_key(name) {
                continue;
            }
            let mut ty: Option<&String> = None;
            let mut ok = true;
            for t in ev {
                match ty {
                    None => ty = Some(t),
                    Some(cur) if cur == t => {}
                    Some(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let Some(ty) = ty else {
                continue;
            };
            // A local the code also uses as a pointer or a field base cannot
            // be the primitive a condition suggests.
            if is_numeric(ty) && (self.pointer_base.contains(name) || self.field_base.contains(name)) {
                continue;
            }
            add.push((name.clone(), ty.clone()));
        }
        add
    }

    /// The type each local's own DEFINITIONS prove, before copy propagation.
    fn by_writes(&self) -> HashMap<String, String> {
        let mut out: HashMap<String, String> = HashMap::new();
        for (name, writes) in &self.writes {
            // Every TYPED write must agree; untyped writes are ignored.
            let mut ty: Option<&String> = None;
            let mut ok = true;
            for w in writes.iter().flatten() {
                match ty {
                    None => ty = Some(w),
                    Some(t) if t == w => {}
                    Some(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let Some(ty) = ty else {
                continue;
            };
            let ty = if self.pointer_base.contains(name) && is_numeric(ty) {
                // Pointer arithmetic needs a pointer type; `nint` is the one
                // C# spelling that supports `+ offset` and a cast to `T*`.
                "nint".to_string()
            } else if self.field_base.contains(name) && is_numeric(ty) {
                // Definitions say primitive, a use says "has fields" — the
                // definitions lost. Declaring either way would be wrong.
                continue;
            } else {
                ty.clone()
            };
            out.insert(name.clone(), ty);
        }
        out
    }
}

fn is_numeric(t: &str) -> bool {
    matches!(
        t,
        "byte" | "sbyte" | "short" | "ushort" | "int" | "uint" | "long" | "ulong" | "float" | "double"
    )
}

/// The C# type an expression's VALUE has, when it is unambiguous.
fn ty_of(e: &Expr, ret: &HashMap<String, String>) -> Option<String> {
    match e {
        Expr::Int(i) => Some(
            if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 {
                "int"
            } else {
                "long"
            }
            .to_string(),
        ),
        Expr::Float(_) => Some("float".to_string()),
        Expr::Double(_) => Some("double".to_string()),
        Expr::Bool(_) => Some("bool".to_string()),
        Expr::Str(_) => Some("string".to_string()),
        Expr::Not(_) | Expr::Cmp(..) => Some("bool".to_string()),
        Expr::Cast(t, _) => Some(t.clone()),
        Expr::NewObj(t, _) => Some(t.clone()),
        Expr::NewArr(t, _) => Some(format!("{t}[]")),
        Expr::StaticField(t, _) => Some(t.clone()),
        Expr::RawCall(_, _, Some(t)) => Some(t.clone()),
        // A RESOLVED call: the callee's return type, when every method with
        // that name agrees on one.
        Expr::Call { name, .. } => {
            let bare = name.rsplit('.').next().unwrap_or(name.as_str());
            ret.get(bare).cloned()
        }
        Expr::RawDeref(_, _, w) => Some(
            match *w {
                1 => "byte",
                2 => "short",
                4 => "int",
                F32_WIDTH => "float",
                _ => "nint",
            }
            .to_string(),
        ),
        Expr::AddressOf(_) => Some("nint".to_string()),
        // `a + b` has a type only when both sides agree on it — the same
        // rule the write merge uses, applied one level down.
        Expr::Bin(_, a, b) => match (ty_of(a, ret), ty_of(b, ret)) {
            (Some(x), Some(y)) if x == y => Some(x),
            _ => None,
        },
        Expr::Cond(_, a, b) => match (ty_of(a, ret), ty_of(b, ret)) {
            (Some(x), Some(y)) if x == y => Some(x),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn infer1(s: &[Stmt]) -> HashMap<String, String> {
        infer(s, &HashMap::new())
    }

    fn assign(n: &str, e: Expr) -> Stmt {
        Stmt::Assign(n.to_string(), None, e)
    }

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    #[test]
    fn an_integer_literal_local_is_an_int() {
        let m = infer1(&[assign("v1", Expr::Int(512))]);
        assert_eq!(m.get("v1").map(String::as_str), Some("int"));
    }

    #[test]
    fn a_large_literal_is_a_long() {
        let m = infer1(&[assign("v1", Expr::Int(0x1_0000_0000))]);
        assert_eq!(m.get("v1").map(String::as_str), Some("long"));
    }

    #[test]
    fn a_new_object_local_takes_the_allocated_type() {
        let m = infer1(&[assign("v1", Expr::NewObj("List<string>".into(), vec![]))]);
        assert_eq!(m.get("v1").map(String::as_str), Some("List<string>"));
    }

    #[test]
    fn disagreeing_writes_are_not_declared() {
        let m = infer1(&[
            assign("v1", Expr::Int(1)),
            assign("v1", Expr::Str("s".into())),
        ]);
        assert!(m.get("v1").is_none());
    }

    #[test]
    fn an_untyped_write_does_not_block_a_typed_one() {
        let m = infer1(&[
            assign("v1", Expr::Int(1)),
            assign(
                "v1",
                Expr::Call {
                    recv: None,
                    name: "f".into(),
                    args: vec![],
                },
            ),
        ]);
        assert_eq!(m.get("v1").map(String::as_str), Some("int"));
    }

    #[test]
    fn a_pointer_base_becomes_nint() {
        let m = infer1(&[
            assign("v1", Expr::Int(0)),
            Stmt::Eval(Expr::RawDeref(Box::new(var("v1")), 8, 4)),
        ]);
        assert_eq!(m.get("v1").map(String::as_str), Some("nint"));
    }

    #[test]
    fn a_field_base_blocks_a_primitive_declaration() {
        let m = infer1(&[
            assign("v1", Expr::Int(0)),
            Stmt::Eval(Expr::Field(Box::new(var("v1")), "x".into())),
        ]);
        assert!(m.get("v1").is_none());
    }

    #[test]
    fn a_local_used_as_a_condition_is_a_bool() {
        // Every write is a field load, which proves nothing; the USE proves
        // the type, and C# requires a bool here.
        let m = infer1(&[
            assign(
                "v1",
                Expr::Field(Box::new(Expr::Var("this".into())), "flag".into()),
            ),
            Stmt::If {
                cond: var("v1"),
                then_body: vec![],
                else_body: vec![],
            },
        ]);
        assert_eq!(m.get("v1").map(String::as_str), Some("bool"));
    }

    #[test]
    fn a_local_only_compared_with_a_literal_is_not_typed_by_use() {
        // `v != 0` compiles for int, uint, long and pointer types alike — a
        // wrong choice here would truncate silently, so nothing is inferred.
        let m = infer1(&[
            assign(
                "v1",
                Expr::Field(Box::new(Expr::Var("this".into())), "handle".into()),
            ),
            Stmt::If {
                cond: Expr::Cmp(
                    crate::recovery::csharp::CmpKind::Ne,
                    Box::new(var("v1")),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![],
                else_body: vec![],
            },
        ]);
        assert!(m.get("v1").is_none());
    }
}
