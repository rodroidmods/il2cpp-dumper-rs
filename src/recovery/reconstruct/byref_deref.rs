//! Use a byref parameter as the managed value it names.
//!
//! IL2CPP passes a `ref`/`out` parameter as a raw address. The lifter keeps
//! that address in the parameter local, so every managed use lowers to a
//! dereference:
//!
//! ```text
//! *(float*)(inTotalProb) = *(float*)(inTotalProb) - inToken.Probability;
//! ```
//!
//! C# writes the same program on the parameter itself:
//!
//! ```text
//! inTotalProb = inTotalProb - inToken.Probability;
//! ```
//!
//! This pass rewrites `RawDeref(Var(p), 0, _)` to `Var(p)` and the store
//! target `*(T*)(p)` to `p` for parameters declared `ref`/`out` with a
//! PRIMITIVE pointee (`ref float`, `ref int`, `ref bool`, `ref string`, …).
//!
//! ## Why only primitive pointees, and only offset 0
//!
//! `*(float*)(p)` for `ref Vector3 p` is the struct's FIRST FIELD, not the
//! struct; substituting `p` there would silently change the type from float
//! to Vector3. Reference pointees would be safe, but the rendered type string
//! does not distinguish a class from a struct, and a wrong guess is worse
//! than the raw deref. Offset != 0 is a field of the pointee in every case
//! and keeps its own layout recovery.
//!
//! Address-taking (`&p`, forwarding the ref) is untouched: passing the
//! parameter already names it.

use super::super::csharp::{Expr, Stmt};
use super::super::types::ManagedType;
use super::children_mut;
use std::collections::{HashMap, HashSet};

/// Byref parameters whose pointee is a primitive value or a reference value.
/// A `ref SomeStruct` is excluded: its offset-0 load is the struct's first
/// field on the rendered node, not the struct itself. A named pointee is
/// inclusive only when its TypeDef is NOT a value type (`ref List<T>`,
/// `ref SomeClass`), which the metadata table answers exactly.
pub fn primitive_byref_names(
    mty: &HashMap<String, ManagedType>,
    value_types: &HashSet<i32>,
) -> HashSet<String> {
    let mut out = HashSet::new();
    for (name, ty) in mty {
        if let ManagedType::ByRef(inner) = ty {
            let reference = match inner.as_ref() {
                ManagedType::Primitive(_)
                | ManagedType::IntPtr
                | ManagedType::String
                | ManagedType::Object
                | ManagedType::SzArray(_) => true,
                ManagedType::Class(nt) => !value_types.contains(&nt.type_def_index),
                ManagedType::Opaque(s) => opaque_is_known_class(s),
                _ => false,
            };
            if reference {
                out.insert(name.clone());
            }
        }
    }
    out
}

/// An `Opaque` display name whose generic definition is a KNOWN BCL CLASS.
/// Generic instances reach the pass as rendered names (`List<int>`,
/// `System.Collections.Generic.Dictionary<K,V>`), so the definition cannot be
/// looked up by TypeDef here. Treating every opaque name as a reference would
/// substitute over `ref KeyValuePair<K,V>` (a struct, whose offset-0 load is
/// its first field); this allowlist contains only classes and interfaces.
fn opaque_is_known_class(display: &str) -> bool {
    let base = display.split('<').next().unwrap_or(display);
    let short = base.rsplit('.').next().unwrap_or(base);
    let short = short.split('`').next().unwrap_or(short);
    matches!(
        short,
        "List"
            | "Dictionary"
            | "HashSet"
            | "Queue"
            | "Stack"
            | "SortedList"
            | "SortedDictionary"
            | "SortedSet"
            | "LinkedList"
            | "Collection"
            | "ReadOnlyCollection"
            | "ObservableCollection"
            | "IList"
            | "IDictionary"
            | "ISet"
            | "ICollection"
            | "IEnumerable"
            | "IReadOnlyList"
            | "IReadOnlyCollection"
    )
}

/// `*(T*)(p)` → `p` when the whole target is one deref of a byref parameter.
fn deref_target_of_param(target: &str, names: &HashSet<String>) -> Option<String> {
    if !target.starts_with("*(") || !target.ends_with(')') {
        return None;
    }
    let mid = target.find("*)(")?;
    let name = &target[mid + 3..target.len() - 1];
    names.contains(name).then(|| name.to_string())
}

fn expr_children_mut(e: &mut Expr) -> Vec<&mut Expr> {
    match e {
        Expr::Field(b, _)
        | Expr::RawDeref(b, _, _)
        | Expr::RawAddr(b, _)
        | Expr::AddressOf(b)
        | Expr::Cast(_, b)
        | Expr::Not(b)
        | Expr::Neg(b)
        | Expr::Await(b)
        | Expr::IndexFromEnd(b) => vec![b],
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => vec![a, b],
        Expr::Cond(c, a, b) => vec![c, a, b],
        Expr::Index { collection, index } => vec![collection, index],
        Expr::NullConditional { receiver, access } => vec![receiver, access],
        Expr::Call { recv, args, .. } => {
            let mut v: Vec<&mut Expr> = args.iter_mut().collect();
            if let Some(r) = recv {
                v.push(r);
            }
            v
        }
        Expr::RawCall(_, args, _)
        | Expr::NewObj(_, args)
        | Expr::NewArr(_, args)
        | Expr::ArrayInit { items: args, .. } => args.iter_mut().collect(),
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            let mut v: Vec<&mut Expr> = ctor_args.iter_mut().collect();
            v.extend(items.iter_mut());
            v
        }
        Expr::ObjInit {
            ctor_args,
            fields,
            ..
        } => {
            let mut v: Vec<&mut Expr> = ctor_args.iter_mut().collect();
            v.extend(fields.iter_mut().map(|(_, x)| x));
            v
        }
        Expr::PatternMatch { expr, .. } => vec![expr],
        Expr::StringInterp { parts } => {
            let mut v: Vec<&mut Expr> = Vec::new();
            for p in parts {
                if let super::super::csharp::InterpPart::Expr(x, _) = p {
                    v.push(x.as_mut());
                }
            }
            v
        }
        Expr::Tuple { elements, .. } => elements.iter_mut().collect(),
        Expr::MethodChain { receiver, methods } => {
            let mut v: Vec<&mut Expr> = vec![receiver];
            for m in methods.iter_mut() {
                v.extend(m.args.iter_mut());
            }
            v
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            let mut v: Vec<&mut Expr> = vec![discrim];
            for (pat, res) in arms.iter_mut() {
                v.push(pat);
                v.push(res);
            }
            if let Some(d) = default_arm {
                v.push(d);
            }
            v
        }
        Expr::Range { from, to } => {
            let mut v: Vec<&mut Expr> = Vec::new();
            if let Some(f) = from {
                v.push(f.as_mut());
            }
            if let Some(t) = to {
                v.push(t.as_mut());
            }
            v
        }
        Expr::Lambda { body, .. } => vec![body],
        _ => vec![],
    }
}

fn rewrite_expr(e: &mut Expr, names: &HashSet<String>) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += rewrite_expr(c, names);
    }
    if let Expr::RawDeref(base, 0, _) = e {
        if let Expr::Var(p) = base.as_ref() {
            if names.contains(p) {
                *e = Expr::Var(p.clone());
                n += 1;
            }
        }
    }
    n
}

fn rewrite_stmt(s: &mut Stmt, names: &HashSet<String>) -> usize {
    let mut n = 0;
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += rewrite_stmt(x, names);
        }
    }
    if let Stmt::Assign(t, _, _) = s {
        if let Some(p) = deref_target_of_param(t, names) {
            *t = p;
            n += 1;
        }
    }
    let mut fold = |e: &mut Expr| n += rewrite_expr(e, names);
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

/// Run the pass. Returns the number of rewritten accesses.
pub fn apply(stmts: &mut Vec<Stmt>, byref_names: &HashSet<String>) -> usize {
    if byref_names.is_empty() {
        return 0;
    }
    let mut n = 0;
    for s in stmts.iter_mut() {
        n += rewrite_stmt(s, byref_names);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn names() -> HashSet<String> {
        HashSet::from(["inTotalProb".to_string()])
    }

    #[test]
    fn ref_primitive_load_and_store_fold() {
        let mut stmts = vec![Stmt::Assign(
            "*(float*)(inTotalProb)".into(),
            None,
            Expr::Bin(
                super::super::super::csharp::BinOp::Sub,
                Box::new(Expr::RawDeref(
                    Box::new(var("inTotalProb")),
                    0,
                    4,
                )),
                Box::new(var("probability")),
            ),
        )];
        assert_eq!(apply(&mut stmts, &names()), 2);
        let t = format!("{stmts:?}");
        assert!(t.contains("Assign(\"inTotalProb\""), "{t}");
        assert!(!t.contains("RawDeref"), "{t}");
    }

    #[test]
    fn offset_access_is_kept() {
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(var("inTotalProb")),
            4,
            4,
        ))];
        assert_eq!(apply(&mut stmts, &names()), 0);
    }

    #[test]
    fn ref_struct_pointee_is_not_listed() {
        // `ref Vector3` parses as ByRef(Class/opaque), never as a primitive.
        let mut mty = HashMap::new();
        mty.insert(
            "v".to_string(),
            ManagedType::ByRef(Box::new(ManagedType::Class(std::sync::Arc::new(
                super::super::super::types::NamedType {
                    namespace: String::new(),
                    name: "Vector3".into(),
                    assembly: None,
                    type_def_index: 7,
                },
            )))),
        );
        let mut value_types = HashSet::new();
        value_types.insert(7);
        assert!(primitive_byref_names(&mty, &value_types).is_empty());
    }

    #[test]
    fn opaque_generic_class_is_listed_but_a_generic_struct_is_not() {
        assert!(opaque_is_known_class("List<int>"));
        assert!(opaque_is_known_class(
            "System.Collections.Generic.Dictionary<System.String, object>"
        ));
        assert!(!opaque_is_known_class(
            "System.Collections.Generic.KeyValuePair<int, int>"
        ));
        assert!(!opaque_is_known_class("Nullable<int>"));
        assert!(!opaque_is_known_class("generic"));
    }

    #[test]
    fn ref_class_pointee_is_listed() {
        let mut mty = HashMap::new();
        mty.insert(
            "p".to_string(),
            ManagedType::ByRef(Box::new(ManagedType::Class(std::sync::Arc::new(
                super::super::super::types::NamedType {
                    namespace: String::new(),
                    name: "List".into(),
                    assembly: None,
                    type_def_index: 1234,
                },
            )))),
        );
        // `List` is not a value type, so `*(nint*)(p)` is p's value.
        let listed = primitive_byref_names(&mty, &HashSet::new());
        assert!(listed.contains("p"));
    }
}
