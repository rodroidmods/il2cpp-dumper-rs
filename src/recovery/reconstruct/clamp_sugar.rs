//! Recover `Mathf.Clamp01` from the compiler's ternary expansion.
//!
//! IL2CPP inlines `Mathf.Clamp01(x)` as
//!
//! ```text
//! x >= 0f ? (x < 1f ? x : 1f) : 0f
//! ```
//!
//! (and the mirrored `x < 0f ? 0f : (x > 1f ? 1f : x)` order). Repeated
//! recovery of the inner expression made `SetVelocityAfterStumble`
//! @0x196B960 one 700-character statement with the whole speed-clamp tree
//! printed three times. The pattern is exact: the SAME expression `x` is the
//! low branch, the high branch and the condition operand, and the constants
//! are 0f/1f. The rewrite evaluates `x` once — what the source call did.
//!
//! A mismatch is a NO-OP; expressions are compared structurally, never by
//! text.

use super::super::csharp::{CmpKind, Expr, Stmt};
use super::children_mut;

fn is_f(e: &Expr, v: f32) -> bool {
    matches!(e, Expr::Float(f) if *f == v)
}

fn cmp_const<'a>(e: &'a Expr, kind: CmpKind, v: f32) -> Option<&'a Expr> {
    let swap = |k: CmpKind| match k {
        CmpKind::Lt => CmpKind::Gt,
        CmpKind::Gt => CmpKind::Lt,
        CmpKind::Le => CmpKind::Ge,
        CmpKind::Ge => CmpKind::Le,
        k => k,
    };
    if let Expr::Cmp(k, a, b) = e {
        if *k == kind && is_f(b, v) {
            return Some(a);
        }
        if *k == swap(kind) && is_f(a, v) {
            return Some(b);
        }
    }
    None
}

/// The clamped expression, when `e` is exactly a Clamp01 expansion.
fn clamp01_arg(e: &Expr) -> Option<&Expr> {
    let Expr::Cond(c1, t, f) = e else {
        return None;
    };
    // x >= 0 ? (x < 1 ? x : 1) : 0
    if is_f(f, 0.0) {
        if let Expr::Cond(c2, t2, f2) = t.as_ref() {
            if is_f(f2, 1.0) {
                let x1 = cmp_const(c1, CmpKind::Ge, 0.0)?;
                let x2 = cmp_const(c2, CmpKind::Lt, 1.0)?;
                if x1 == t2.as_ref() && x1 == x2 {
                    return Some(x1);
                }
            }
        }
    }
    // x < 0 ? 0 : (x > 1 ? 1 : x)
    if is_f(t, 0.0) {
        if let Expr::Cond(c2, t2, f2) = f.as_ref() {
            if is_f(t2, 1.0) {
                let x1 = cmp_const(c1, CmpKind::Lt, 0.0)?;
                let x2 = cmp_const(c2, CmpKind::Gt, 1.0)?;
                if x1 == x2 && x1 == f2.as_ref() {
                    return Some(x1);
                }
            }
        }
    }
    None
}

fn rewrite_expr(e: &mut Expr) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += rewrite_expr(c);
    }
    if let Some(arg) = clamp01_arg(e) {
        let arg = arg.clone();
        *e = Expr::Call {
            recv: None,
            name: "Mathf.Clamp01".into(),
            args: vec![arg],
        };
        n += 1;
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
        | Expr::ArrayInit { items: args, .. } => args.iter_mut().collect(),
        _ => vec![],
    }
}

fn fold_stmt(s: &mut Stmt) -> usize {
    let mut n = 0;
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += fold_stmt(x);
        }
    }
    let mut fold = |e: &mut Expr| n += rewrite_expr(e);
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
    let mut n = 0;
    for s in stmts.iter_mut() {
        n += fold_stmt(s);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }
    fn cmp(k: CmpKind, a: Expr, b: Expr) -> Expr {
        Expr::Cmp(k, Box::new(a), Box::new(b))
    }

    #[test]
    fn clamp01_expansion_folds() {
        let x = v("x");
        let e = Expr::Cond(
            Box::new(cmp(CmpKind::Ge, x.clone(), Expr::Float(0.0))),
            Box::new(Expr::Cond(
                Box::new(cmp(CmpKind::Lt, x.clone(), Expr::Float(1.0))),
                Box::new(x.clone()),
                Box::new(Expr::Float(1.0)),
            )),
            Box::new(Expr::Float(0.0)),
        );
        let mut stmts = vec![Stmt::Eval(e)];
        assert_eq!(apply(&mut stmts), 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("Mathf.Clamp01"), "{t}");
    }

    #[test]
    fn different_operand_is_refused() {
        let e = Expr::Cond(
            Box::new(cmp(CmpKind::Ge, v("x"), Expr::Float(0.0))),
            Box::new(Expr::Cond(
                Box::new(cmp(CmpKind::Lt, v("y"), Expr::Float(1.0))),
                Box::new(v("y")),
                Box::new(Expr::Float(1.0)),
            )),
            Box::new(Expr::Float(0.0)),
        );
        let mut stmts = vec![Stmt::Eval(e)];
        assert_eq!(apply(&mut stmts), 0);
    }
}
