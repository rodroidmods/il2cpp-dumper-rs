//! Float lanes: retype a local that is only ever compared against a float
//! immediate.
//!
//! The lifter types every 32-bit load by its ACCESS WIDTH, so a `ldr s0, [x1,
//! #0x18]` — a single-precision load — arrives as `RawDeref(base, 0x18, 4)`
//! and renders `*(int*)(x1 + 24)`. When the value is then tested against a
//! float immediate the output contradicts itself:
//!
//! ```text
//! sf8 = *(int*)(t + 24);
//! if (sf8 < 0f)                       // `int < float` — CS0019
//! ```
//!
//! The comparison is the proof: an integer local can never be the operand of
//! a comparison whose other side is a `float`/`double` literal (C# would have
//! inserted a conversion, and the lifter renders conversions as `Expr::Cast`,
//! which this pass looks through). So the local IS single-precision, and every
//! 32-bit load that defines it is a float load.
//!
//! The rewrite changes the access WIDTH CODE from `4` (int) to
//! [`F32_WIDTH`], which the `RawDeref` renderer and `cast_elision::width_type`
//! both spell `float`. Nothing else is touched: the address expression, the
//! offset and the surrounding control flow stay exactly as they were.

use std::collections::HashSet;

use crate::recovery::csharp::{Expr, Stmt};

use super::il2cpp_cleanup::{expr_children, expr_children_mut};

/// Access-width code for a 4-byte FLOAT load. `4` is the 4-byte INT code; the
/// renderer maps `5` to `float` and `cast_elision::width_type` agrees.
pub const F32_WIDTH: u8 = 5;

/// Retype every local this method compares against a float immediate.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut floats: HashSet<String> = HashSet::new();
    collect(stmts, &mut floats);
    if floats.is_empty() {
        return 0;
    }
    let mut n = 0usize;
    rewrite(stmts, &floats, &mut n);
    n
}

fn collect(stmts: &[Stmt], floats: &mut HashSet<String>) {
    for s in stmts {
        for_each_expr(s, &mut |e| walk_expr(e, &mut |node| note_float(node, floats)));
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect(k, floats);
        }
    }
}

/// Record the variable operand of a float-tested comparison or a float
/// arithmetic operand.
fn note_float(e: &Expr, floats: &mut HashSet<String>) {
    match e {
        Expr::Cmp(_, a, b) => {
            if is_float_lit(a) {
                if let Some(n) = var_of(b) {
                    floats.insert(n.to_string());
                }
            }
            if is_float_lit(b) {
                if let Some(n) = var_of(a) {
                    floats.insert(n.to_string());
                }
            }
        }
        // `v + 0.5f`, `v * 2f` — the same proof, through arithmetic.
        Expr::Bin(_, a, b) => {
            if is_float_lit(a) {
                if let Some(n) = var_of(b) {
                    floats.insert(n.to_string());
                }
            }
            if is_float_lit(b) {
                if let Some(n) = var_of(a) {
                    floats.insert(n.to_string());
                }
            }
        }
        _ => {}
    }
}

fn is_float_lit(e: &Expr) -> bool {
    matches!(e, Expr::Float(_) | Expr::Double(_))
}

/// The variable an operand names, looking through a numeric conversion.
fn var_of(e: &Expr) -> Option<&str> {
    match e {
        Expr::Var(n) => Some(n.as_str()),
        Expr::Cast(_, inner) => var_of(inner),
        _ => None,
    }
}

fn rewrite(stmts: &mut Vec<Stmt>, floats: &HashSet<String>, n: &mut usize) {
    for s in stmts.iter_mut() {
        let mut target: Option<String> = None;
        if let Stmt::Assign(name, _, _) = s {
            if floats.contains(name.as_str()) {
                target = Some(name.clone());
            }
        }
        if target.is_some() {
            super::copy_prop::for_each_expr_mut(s, &mut |e| {
                walk_expr_mut(e, &mut |node| {
                    if let Expr::RawDeref(_, _, width) = node {
                        if *width == 4 {
                            *width = F32_WIDTH;
                            *n += 1;
                        }
                    }
                });
            });
        }
        for child in super::children_mut(s) {
            rewrite(child, floats, n);
        }
    }
}

/// Expressions directly owned by this statement (nested statement lists are
/// visited separately). Mirrors `copy_prop::for_each_expr_mut`; a position it
/// misses can only cost a detection, never produce a wrong rewrite.
pub(crate) fn for_each_expr(s: &Stmt, f: &mut dyn FnMut(&Expr)) {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Assign(_, _, e) => f(e),
        Stmt::Return(Some(e)) => f(e),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => f(cond),
        Stmt::Lock { expr, .. } => f(expr),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                f(c);
            }
        }
        Stmt::StringSwitch { disc, .. } => f(disc),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            f(collection);
            f(index);
            f(value);
        }
        Stmt::EventAssign { field, value, .. } => {
            f(field);
            f(value);
        }
        Stmt::Attribute { args, .. } => {
            for a in args {
                f(a);
            }
        }
        _ => {}
    }
}

fn walk_expr<'a>(e: &'a Expr, f: &mut impl FnMut(&'a Expr)) {
    f(e);
    for c in expr_children(e) {
        walk_expr(c, f);
    }
}

fn walk_expr_mut(e: &mut Expr, f: &mut impl FnMut(&mut Expr)) {
    f(e);
    for c in expr_children_mut(e) {
        walk_expr_mut(c, f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::CmpKind;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn load(n: &str, off: i64, width: u8) -> Expr {
        Expr::RawDeref(Box::new(var(n)), off, width)
    }

    #[test]
    fn a_float_tested_local_gets_a_float_load() {
        let mut stmts = vec![
            Stmt::Assign("sf8".into(), None, load("t", 24, 4)),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Lt,
                    Box::new(var("sf8")),
                    Box::new(Expr::Float(0.0)),
                ),
                then_body: vec![Stmt::Return(Some(Expr::Bool(true)))],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        let Stmt::Assign(_, _, Expr::RawDeref(_, _, w)) = &stmts[0] else {
            panic!("assign")
        };
        assert_eq!(*w, F32_WIDTH);
    }

    #[test]
    fn an_int_tested_local_keeps_its_int_load() {
        let mut stmts = vec![
            Stmt::Assign("n".into(), None, load("t", 24, 4)),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Lt,
                    Box::new(var("n")),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Return(Some(Expr::Bool(true)))],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
        let Stmt::Assign(_, _, Expr::RawDeref(_, _, w)) = &stmts[0] else {
            panic!("assign")
        };
        assert_eq!(*w, 4);
    }

    #[test]
    fn a_double_conversion_does_not_hide_the_proof() {
        let mut stmts = vec![
            Stmt::Assign("sf8".into(), None, load("t", 24, 4)),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::Cast("double".into(), Box::new(var("sf8")))),
                    Box::new(Expr::Double(-0.5)),
                ),
                then_body: vec![Stmt::Return(Some(Expr::Bool(true)))],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
    }

    #[test]
    fn a_byte_load_is_left_alone() {
        let mut stmts = vec![
            Stmt::Assign("b".into(), None, load("t", 24, 1)),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Lt,
                    Box::new(var("b")),
                    Box::new(Expr::Float(0.0)),
                ),
                then_body: vec![Stmt::Return(Some(Expr::Bool(true)))],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }
}
