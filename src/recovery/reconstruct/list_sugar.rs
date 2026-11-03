//! Recover `List<T>.Count` from the inlined `_size` field.
//!
//! `List<T>` keeps `_size` at offset 24 (4 bytes). When the compiler inlines
//! list operations, the decompiler prints the raw load
//! `*(int*)(list + 24)` in conditions and arithmetic. The receiver is proven
//! to be a List by a call to one of its instance methods (`Add`,
//! `AddWithResize`, `Remove`, `Clear`, `Contains`, `IndexOf`, `Insert`,
//! `RemoveAt`, `get_Count`, `get_Item`) in the same method — the managed API
//! only exists on `List<T>`-shaped receivers in this corpus. Only then is the
//! 4-byte +24 load rewritten to `list.Count`; an unproven receiver keeps the
//! raw expression.

use super::super::csharp::{Expr, Stmt};
use super::children_mut;
use std::collections::HashSet;

const LIST_METHODS: &[&str] = &[
    "Add",
    "AddWithResize",
    "Remove",
    "RemoveAt",
    "Clear",
    "Contains",
    "IndexOf",
    "Insert",
    "get_Count",
    "get_Item",
    "set_Item",
];

fn collect_bases_expr(e: &Expr, out: &mut Vec<Expr>) {
    if let Expr::Call { recv: Some(r), name, .. } = e {
        if LIST_METHODS.contains(&name.as_str()) {
            out.push((**r).clone());
        }
    }
    for c in expr_children(e) {
        collect_bases_expr(c, out);
    }
}

fn collect_bases(stmts: &[Stmt], out: &mut Vec<Expr>) {
    for s in stmts {
        for e in direct_exprs(s) {
            collect_bases_expr(e, out);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_bases(k, out);
        }
    }
}

fn direct_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Return(Some(e)) => vec![e],
        Stmt::Return(None) => vec![],
        Stmt::Assign(_, _, e) => vec![e],
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => vec![cond],
        Stmt::For { cond, .. } => cond.iter().collect(),
        Stmt::Lock { expr, .. } => vec![expr],
        Stmt::StringSwitch { disc, .. } => vec![disc],
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => vec![collection, index, value],
        Stmt::EventAssign { field, value, .. } => vec![field, value],
        _ => vec![],
    }
}

fn expr_children(e: &Expr) -> Vec<&Expr> {
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
            let mut v: Vec<&Expr> = args.iter().collect();
            if let Some(r) = recv {
                v.push(r);
            }
            v
        }
        Expr::RawCall(_, args, _)
        | Expr::NewObj(_, args)
        | Expr::NewArr(_, args)
        | Expr::ArrayInit { items: args, .. } => args.iter().collect(),
        _ => vec![],
    }
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
        _ => vec![],
    }
}

fn rewrite_expr(e: &mut Expr, bases: &[Expr]) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += rewrite_expr(c, bases);
    }
    if let Expr::RawDeref(b, 24, 4) = e {
        if bases.iter().any(|x| x == b.as_ref()) {
            let base = (**b).clone();
            *e = Expr::Field(Box::new(base), "Count".into());
            n += 1;
        }
    }
    // `AddWithResize` is List<T>'s private implementation detail; the source
    // call was `Add` (which performs the same resize when needed). The name
    // itself proves the receiver, so no separate evidence is required.
    if let Expr::Call { name, args, .. } = e {
        if name == "AddWithResize" && args.len() == 1 {
            *name = "Add".into();
            n += 1;
        }
    }
    n
}

fn rewrite_stmt(s: &mut Stmt, bases: &[Expr]) -> usize {
    let mut n = 0;
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += rewrite_stmt(x, bases);
        }
    }
    let mut fold = |e: &mut Expr| n += rewrite_expr(e, bases);
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

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut bases = Vec::new();
    collect_bases(stmts, &mut bases);
    if bases.is_empty() {
        return 0;
    }
    // Deduplicate structurally to keep the linear scan short.
    let mut seen = HashSet::new();
    bases.retain(|b| seen.insert(format!("{b:?}")));
    let mut n = 0;
    for s in stmts.iter_mut() {
        n += rewrite_stmt(s, &bases);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    #[test]
    fn proven_list_size_becomes_count() {
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("list"))),
                name: "Add".into(),
                args: vec![var("item")],
            }),
            Stmt::If {
                cond: Expr::Cmp(
                    super::super::super::csharp::CmpKind::Gt,
                    Box::new(Expr::RawDeref(Box::new(var("list")), 24, 4)),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("Count"), "{t}");
    }

    #[test]
    fn add_with_resize_is_the_public_add() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("list"))),
            name: "AddWithResize".into(),
            args: vec![var("item")],
        })];
        assert_eq!(apply(&mut stmts), 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("\"Add\""), "{t}");
        assert!(!t.contains("AddWithResize"), "{t}");
    }

    #[test]
    fn unproven_receiver_keeps_raw_load() {
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(Box::new(var("o")), 24, 4))];
        assert_eq!(apply(&mut stmts), 0);
    }
}
