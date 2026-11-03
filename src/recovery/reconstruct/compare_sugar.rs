//! Normalize `== 0` / `!= 0` comparisons to their managed meaning.
//!
//! IL2CPP lowers every boolean and every reference test to an integer compare.
//! The decompiled text then reads `if (this.badge != 0)` for `if (badge != null)`
//! and `if ((a > b) != 0)` for `if (a > b)` — 31,538 `!= 0)` and 7,494 `== 0)`
//! in the reference dump (R25).
//!
//! ## Evidence, not guesses
//!
//! Without type information the pass uses *usage*:
//!
//! * **References**: an expression that is used as a receiver (`x.M()`), a
//!   field/deref base (`x.field`, `*(T*)(x)`), a null-conditional receiver, or
//!   is a `new` allocation/typeof — those are objects. A zero test on such an
//!   expression becomes `== null` / `!= null`.
//! * **Booleans**: an expression that is itself a comparison, a `Not`, a
//!   boolean literal, or a local that is only ever assigned from one of those
//!   and read in conditions. `bool != 0` becomes `bool`, `bool == 0` becomes
//!   `!bool`.
//! * `cond ? 1 : 0` becomes `cond`, `cond ? 0 : 1` becomes `!cond`.
//!
//! Everything else keeps its integer comparison: a numeric zero test is left
//! alone.

use super::super::csharp::{CmpKind, Expr, Stmt};
use super::children_mut;
use std::collections::HashSet;

/// Structural key for an expression (Debug text is injective for our AST).
fn key(e: &Expr) -> String {
    format!("{e:?}")
}

fn is_zero(e: &Expr) -> bool {
    matches!(e, Expr::Int(0))
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
        | Expr::Await(b) => vec![b],
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

/// Statements' own expressions (conditions, values), for the evidence scan.
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

/// Collect expressions proven to be REFERENCES (receivers / deref bases) and
/// locals proven to be BOOLEANS (assigned from a boolean expression).
fn collect_evidence(
    stmts: &[Stmt],
    refs: &mut HashSet<String>,
    bool_locals: &mut HashSet<String>,
    ref_locals: &mut HashSet<String>,
) {
    for s in stmts {
        if let Stmt::Assign(name, _, rhs) = s {
            match rhs {
                Expr::Cmp(..) | Expr::Not(_) | Expr::Bool(_) => {
                    bool_locals.insert(name.clone());
                }
                Expr::NewObj(..) | Expr::NewArr(..) | Expr::CollectionInit { .. }
                | Expr::ObjInit { .. } | Expr::Lambda { .. } => {
                    ref_locals.insert(name.clone());
                }
                _ => {}
            }
        }
        for e in stmt_exprs(s) {
            collect_expr_evidence(e, refs);
        }
        for k in children_lists(s) {
            collect_evidence(k, refs, bool_locals, ref_locals);
        }
    }
}

fn children_lists(s: &Stmt) -> Vec<&Vec<Stmt>> {
    let mut v: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut v);
    v
}

fn collect_expr_evidence(e: &Expr, refs: &mut HashSet<String>) {
    match e {
        Expr::Field(base, _) | Expr::RawDeref(base, _, _) | Expr::RawAddr(base, _) => {
            refs.insert(key(base));
        }
        Expr::Call { recv: Some(r), .. } => {
            // A receiver that is a plain local could be a numeric value
            // (`count.ToString()`); locals only become references through an
            // allocation/copy assignment (`ref_locals`). Field/deref receivers
            // are objects.
            if !matches!(r.as_ref(), Expr::Var(_)) {
                refs.insert(key(r));
            }
        }
        Expr::NullConditional { receiver, .. } => {
            refs.insert(key(receiver));
        }
        Expr::MethodChain { receiver, .. } => {
            refs.insert(key(receiver));
        }
        _ => {}
    }
    for c in expr_children(e) {
        collect_expr_evidence(c, refs);
    }
}

/// `cond ? 1 : 0` ↔ boolean; zero-tests on proven booleans/references.
fn sugar_expr(e: &mut Expr, refs: &HashSet<String>, bool_locals: &HashSet<String>) -> usize {
    let mut n = 0usize;
    for c in expr_children_mut(e) {
        n += sugar_expr(c, refs, bool_locals);
    }
    // `cond ? 1 : 0` → cond, `cond ? 0 : 1` → !cond
    if let Expr::Cond(c, a, b) = e {
        let one = |x: &Expr| matches!(x, Expr::Int(1));
        let zero = |x: &Expr| matches!(x, Expr::Int(0));
        if one(a) && zero(b) {
            *e = (**c).clone();
            return n + 1;
        }
        if zero(a) && one(b) {
            *e = Expr::Not(c.clone());
            return n + 1;
        }
    }
    let Expr::Cmp(op @ (CmpKind::Eq | CmpKind::Ne), lhs, rhs) = e else {
        return n;
    };
    let (other, zero_on_left) = match (lhs.as_ref(), rhs.as_ref()) {
        (x, z) if is_zero(z) => (x, false),
        (_, x) if is_zero(x) => (x, true),
        _ => return n,
    };
    if is_zero(other) {
        return n;
    }
    // Boolean expression or boolean local: `b != 0` → b, `b == 0` → !b.
    let is_bool = matches!(
        other,
        Expr::Cmp(..) | Expr::Not(_) | Expr::Bool(_)
    ) || matches!(other, Expr::Var(n) if bool_locals.contains(n));
    if is_bool {
        let ne = matches!(op, CmpKind::Ne);
        let replacement = if ne {
            other.clone()
        } else {
            Expr::Not(Box::new(other.clone()))
        };
        *e = replacement;
        return n + 1;
    }
    // Reference: `x != 0` → `x != null`.
    let is_ref = match other {
        Expr::NewObj(..) | Expr::NewArr(..) | Expr::This => true,
        Expr::Var(n) => refs.contains(&key(other)) || refs.contains(&format!("Var(\"{n}\")")),
        _ => refs.contains(&key(other)),
    };
    if is_ref {
        let null = Box::new(Expr::Null);
        if zero_on_left {
            *e = Expr::Cmp(*op, null, lhs.clone());
        } else {
            *e = Expr::Cmp(*op, lhs.clone(), null);
        }
        return n + 1;
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
        | Expr::Await(b) => vec![b],
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
        _ => vec![],
    }
}

fn sugar_stmts(stmts: &mut [Stmt], refs: &HashSet<String>, bools: &HashSet<String>) -> usize {
    let mut total = 0usize;
    for s in stmts.iter_mut() {
        match s {
            Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => total += sugar_expr(e, refs, bools),
            Stmt::Return(Some(e)) => total += sugar_expr(e, refs, bools),
            Stmt::Assign(_, _, e) => total += sugar_expr(e, refs, bools),
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                total += sugar_expr(cond, refs, bools);
                total += sugar_stmts(then_body, refs, bools);
                total += sugar_stmts(else_body, refs, bools);
            }
            Stmt::While { cond, body } => {
                total += sugar_expr(cond, refs, bools);
                total += sugar_stmts(body, refs, bools);
            }
            Stmt::DoWhile { body, cond } => {
                total += sugar_stmts(body, refs, bools);
                total += sugar_expr(cond, refs, bools);
            }
            Stmt::For {
                init,
                cond,
                incr,
                body,
            } => {
                total += sugar_stmts(init, refs, bools);
                if let Some(c) = cond {
                    total += sugar_expr(c, refs, bools);
                }
                total += sugar_stmts(incr, refs, bools);
                total += sugar_stmts(body, refs, bools);
            }
            Stmt::Lock { expr, body } => {
                total += sugar_expr(expr, refs, bools);
                total += sugar_stmts(body, refs, bools);
            }
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                total += sugar_expr(collection, refs, bools);
                total += sugar_expr(index, refs, bools);
                total += sugar_expr(value, refs, bools);
            }
            Stmt::EventAssign { field, value, .. } => {
                total += sugar_expr(field, refs, bools);
                total += sugar_expr(value, refs, bools);
            }
            Stmt::StringSwitch { disc, cases, tail } => {
                total += sugar_expr(disc, refs, bools);
                for (_, b) in cases.iter_mut() {
                    total += sugar_stmts(b, refs, bools);
                }
                if let Some(t) = tail {
                    total += sugar_stmts(t, refs, bools);
                }
            }
            _ => {
                for k in children_mut(s) {
                    total += sugar_stmts(k, refs, bools);
                }
            }
        }
    }
    total
}

/// Run the normalization. Returns the number of rewritten comparisons.
pub fn apply(stmts: &mut [Stmt]) -> usize {
    let mut refs: HashSet<String> = HashSet::new();
    let mut bool_locals: HashSet<String> = HashSet::new();
    let mut ref_locals: HashSet<String> = HashSet::new();
    collect_evidence(stmts, &mut refs, &mut bool_locals, &mut ref_locals);
    for n in &ref_locals {
        refs.insert(format!("Var(\"{n}\")"));
    }
    sugar_stmts(stmts, &refs, &bool_locals)
}

#[allow(dead_code)]
fn count_zero_cmps(stmts: &[Stmt]) -> usize {
    let mut n = 0;
    for s in stmts {
        for e in stmt_exprs(s) {
            n += count_in_expr(e);
        }
        for k in children_lists(s) {
            n += count_zero_cmps(k);
        }
    }
    n
}

#[allow(dead_code)]
fn count_in_expr(e: &Expr) -> usize {
    let mut n = match e {
        Expr::Cmp(_, a, b) if is_zero(a) || is_zero(b) => 1,
        _ => 0,
    };
    for c in expr_children(e) {
        n += count_in_expr(c);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(base: Expr, f: &str) -> Expr {
        Expr::Field(Box::new(base), f.to_string())
    }

    fn call(recv: Expr, name: &str) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(recv)),
            name: name.to_string(),
            args: vec![],
        })
    }

    #[test]
    fn reference_receiver_becomes_null_test() {
        let badge = field(Expr::This, "_badge");
        let mut stmts = vec![
            call(badge.clone(), "SetActive"),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(badge), Box::new(Expr::Int(0))),
                then_body: vec![],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(format!("{stmts:?}").contains("Null"));
    }

    #[test]
    fn boolean_expression_zero_test_is_dropped() {
        let cmp = Expr::Cmp(
            CmpKind::Gt,
            Box::new(Expr::Var("a".into())),
            Box::new(Expr::Var("b".into())),
        );
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Ne, Box::new(cmp.clone()), Box::new(Expr::Int(0))),
            then_body: vec![],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 1);
        assert_eq!(
            format!("{:?}", stmts[0]),
            format!("{:?}", Stmt::If { cond: cmp, then_body: vec![], else_body: vec![] })
        );
    }

    #[test]
    fn numeric_zero_test_is_kept() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var("count".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn ternary_int_to_bool() {
        let cond = Expr::Cmp(
            CmpKind::Lt,
            Box::new(Expr::Var("a".into())),
            Box::new(Expr::Var("b".into())),
        );
        let mut stmts = vec![Stmt::Assign(
            "ok".into(),
            None,
            Expr::Cond(Box::new(cond.clone()), Box::new(Expr::Int(1)), Box::new(Expr::Int(0))),
        )];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::Cmp(..))));
    }
}
