//! Drop stale-register arguments from UNRESOLVED raw calls.
//!
//! At a call whose target has no metadata, the lifter dumps every ABI register
//! (x0..x7, s0..s7) as an argument. The trailing ones are often registers the
//! callee never reads; in the AST they arrive as `Var("vN")` names that have
//! NO definition anywhere in the method (and are not parameters) — stale
//! register garbage, rendered as undefined locals. Measured on 0x168BB94:
//! `fn_1302420(…, v154, v155, v156, v157)` on every helper call, with
//! `v156`/`v157` defined nowhere; they accounted for 40 of that method's
//! undefined references.
//!
//! The pass trims TRAILING arguments that are:
//!   * `MissingAbiArg`, or
//!   * `Var(name)` where `name` is neither defined anywhere in the method
//!     (assignment target / foreach variable / lambda parameter) nor a method
//!     parameter.
//!
//! Only resolved calls (`Expr::Call`) are untouched: their arguments were
//! sliced against a metadata signature. An interior undefined argument stays
//! (dropping it would silently shift the declared parameters).

use super::super::csharp::{Expr, Stmt};
use super::children_mut;
use std::collections::HashSet;

fn collect_defined(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign(t, _, _) => {
                if !t.contains('.') && !t.contains('*') && !t.contains('[') {
                    out.insert(t.clone());
                }
            }
            Stmt::ForEach { var_name, .. } => {
                out.insert(var_name.clone());
            }
            _ => {}
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_defined(k, out);
        }
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
        Expr::Lambda { params, body } => {
            let _ = params;
            vec![body]
        }
        _ => vec![],
    }
}

fn collect_lambda_params(e: &Expr, out: &mut HashSet<String>) {
    if let Expr::Lambda { params, .. } = e {
        for p in params {
            out.insert(p.clone());
        }
    }
    for c in expr_children(e) {
        collect_lambda_params(c, out);
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

fn walk_ctx_args(
    args: &[Expr],
    pos: &mut std::collections::HashMap<String, Vec<usize>>,
    other: &mut HashSet<String>,
) {
    for (i, a) in args.iter().enumerate() {
        if matches!(a, Expr::Var(_)) {
            walk_ctx_expr(a, Some(i), pos, other);
        } else {
            walk_ctx_expr(a, None, pos, other);
        }
    }
}

/// Positions of every Var used directly as a call argument, and names used in
/// any other way (arithmetic, receiver, condition, target, …).
fn walk_ctx_expr(
    e: &Expr,
    direct_arg: Option<usize>,
    pos: &mut std::collections::HashMap<String, Vec<usize>>,
    other: &mut HashSet<String>,
) {
    match e {
        Expr::Var(n) => {
            if let Some(i) = direct_arg {
                pos.entry(n.clone()).or_default().push(i);
            } else {
                other.insert(n.clone());
            }
            return;
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                walk_ctx_expr(r, None, pos, other);
            }
            walk_ctx_args(args, pos, other);
        }
        Expr::RawCall(_, args, _) => walk_ctx_args(args, pos, other),
        _ => {
            for c in expr_children(e) {
                walk_ctx_expr(c, None, pos, other);
            }
        }
    }
}

fn ctx_stmt(
    s: &Stmt,
    pos: &mut std::collections::HashMap<String, Vec<usize>>,
    other: &mut HashSet<String>,
) {
    // A variable named inside a COMPLEX assignment target (`v70.field`,
    // `*(nint*)(v70+4)`) is a value/address base. A plain `v70 = …` target is
    // a DEFINITION, not a use — counting it here made every defined temp look
    // “used elsewhere”, so context-register args could never be trimmed.
    if let Stmt::Assign(t, _, _) = s {
        if t.chars().any(|c| !(c.is_ascii_alphanumeric() || c == '_')) {
            for n in re_find_vars(t) {
                other.insert(n);
            }
        }
    }
    for e in direct_exprs(s) {
        walk_ctx_expr(e, None, pos, other);
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            ctx_stmt(x, pos, other);
        }
    }
}

fn re_find_vars(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'v' {
            let start = i;
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 {
                out.push(text[start..j].to_string());
                i = j;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn is_temp_name(n: &str) -> bool {
    n.len() > 1
        && n.starts_with('v')
        && n[1..].chars().all(|c| c.is_ascii_digit())
}

fn trim_raw_args(e: &mut Expr, defined: &HashSet<String>, arg_only: &HashSet<String>) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += trim_raw_args(c, defined, arg_only);
    }
    if let Expr::RawCall(_, args, _) = e {
        // Context registers: a var used ONLY as a direct call argument and
        // never in the first two positions of any call is register-dump
        // plumbing (`v70/v72` at `fn_…(obj, class, v70, v72, in_x4)`), not a
        // source value. Drop it wherever it appears after the first two args.
        let mut idx = 0;
        let before = args.len();
        args.retain(|a| {
            let keep = !(idx >= 2
                && matches!(a, Expr::Var(name) if arg_only.contains(name)));
            idx += 1;
            keep
        });
        n += before - args.len();
        while let Some(last) = args.last() {
            let drop = matches!(last, Expr::MissingAbiArg)
                || matches!(last, Expr::Var(name) if !defined.contains(name));
            if drop {
                args.pop();
                n += 1;
            } else {
                break;
            }
        }
    }
    n
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
        | Expr::ArrayInit { items: args, .. }
        | Expr::Tuple { elements: args, .. } => args.iter_mut().collect(),
        Expr::CollectionInit { ctor_args, items, .. } => {
            let mut v: Vec<&mut Expr> = ctor_args.iter_mut().collect();
            v.extend(items.iter_mut());
            v
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            let mut v: Vec<&mut Expr> = ctor_args.iter_mut().collect();
            v.extend(fields.iter_mut().map(|(_, e)| e));
            v
        }
        Expr::Lambda { body, .. } => vec![body],
        _ => vec![],
    }
}

fn trim_stmt(s: &mut Stmt, defined: &HashSet<String>, arg_only: &HashSet<String>) -> usize {
    let mut n = 0;
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += trim_stmt(x, defined, arg_only);
        }
    }
    let mut fold = |e: &mut Expr| n += trim_raw_args(e, defined, arg_only);
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

/// `params` are the method's parameter display names (never trimmed).
pub fn apply(stmts: &mut Vec<Stmt>, params: &HashSet<String>) -> usize {
    let mut defined = params.clone();
    collect_defined(stmts, &mut defined);
    for s in stmts.iter() {
        for e in direct_exprs(s) {
            collect_lambda_params(e, &mut defined);
        }
    }
    // Context-only names: used exclusively as direct call arguments, never
    // before position 2, and not parameters.
    let mut pos: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    let mut other: HashSet<String> = HashSet::new();
    for s in stmts.iter() {
        ctx_stmt(s, &mut pos, &mut other);
    }
    let arg_only: HashSet<String> = pos
        .into_iter()
        .filter(|(n, ps)| {
            is_temp_name(n)
                && !params.contains(n)
                && !other.contains(n)
                && ps.iter().copied().min().unwrap_or(usize::MAX) >= 2
        })
        .map(|(n, _)| n)
        .collect();
    let mut n = 0;
    for s in stmts.iter_mut() {
        n += trim_stmt(s, &defined, &arg_only);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_undefined_and_abi_args_are_trimmed() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(1)),
            Stmt::Eval(Expr::RawCall(
                0x1000,
                vec![
                    Expr::Var("v1".into()),
                    Expr::Var("v9".into()),
                    Expr::MissingAbiArg,
                ],
                None,
            )),
        ];
        assert_eq!(apply(&mut stmts, &HashSet::new()), 2);
        let t = format!("{stmts:?}");
        assert!(!t.contains("v9"), "{t}");
        assert!(t.contains("v1"), "{t}");
    }

    #[test]
    fn context_only_arg_is_trimmed_but_parameters_are_not() {
        let mut stmts = vec![
            Stmt::Assign("v70".into(), None, Expr::Var("in_x2".into())),
            Stmt::Assign("v72".into(), None, Expr::Var("in_x3".into())),
            Stmt::Eval(Expr::RawCall(
                0x1302420,
                vec![
                    Expr::Var("obj".into()),
                    Expr::Var("cls".into()),
                    Expr::Var("v70".into()),
                    Expr::Var("v72".into()),
                ],
                None,
            )),
        ];
        let mut params = HashSet::new();
        params.insert("obj".to_string());
        params.insert("cls".to_string());
        assert!(apply(&mut stmts, &params) >= 2);
        let args = stmts
            .iter()
            .find_map(|s| match s {
                Stmt::Eval(Expr::RawCall(_, a, _)) => Some(a.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(args.len(), 2, "{args:?}");

        // A parameter at position 2 is a real value and must survive.
        let mut params = HashSet::new();
        params.insert("obj".to_string());
        params.insert("cls".to_string());
        params.insert("v70".to_string());
        let mut stmts = vec![
            Stmt::Assign("v70".into(), None, Expr::Var("in_x2".into())),
            Stmt::Eval(Expr::RawCall(
                0x1302420,
                vec![
                    Expr::Var("obj".into()),
                    Expr::Var("cls".into()),
                    Expr::Var("v70".into()),
                ],
                None,
            )),
        ];
        apply(&mut stmts, &params);
        let args = stmts
            .iter()
            .find_map(|s| match s {
                Stmt::Eval(Expr::RawCall(_, a, _)) => Some(a.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(args.len(), 3, "{args:?}");
        assert!(matches!(&args[2], Expr::Var(n) if n == "v70"));
    }

    #[test]
    fn interior_undefined_argument_is_kept() {
        let mut params = HashSet::new();
        params.insert("v1".to_string());
        let mut stmts = vec![Stmt::Eval(Expr::RawCall(
            0x1000,
            vec![Expr::Var("v9".into()), Expr::Var("v1".into())],
            None,
        ))];
        assert_eq!(apply(&mut stmts, &params), 0);
        let t = format!("{stmts:?}");
        assert!(t.contains("v9"), "{t}");
    }
}
