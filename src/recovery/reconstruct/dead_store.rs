//! Dead stores to plain locals.
//!
//! ## Why this pass exists
//!
//! SSA destruction leaves "clear" stores that the very next definition
//! overwrites:
//!
//! ```text
//! ptr2 = MethodInfo("Token>..ctor");   // never read
//! ...
//! ptr2 = 0;
//! ```
//!
//! and the machine's per-iteration re-initialisation of a merged slot
//! (`obj12 = 0;` at the top of a body whose first real definition follows).
//! They are invisible to `copy_prop` (which only drops `t = <var>` copies) and
//! to `prune_dead_code` (which only drops statements after a terminator), yet
//! they are the bulk of the remaining `= 0;` / `= MethodInfo(...)` noise.
//!
//! ## The rule
//!
//! `x = <value-pure expr>;` is removed when, before any read of `x` on the
//! same straight-line run, another statement writes `x` outright. The run may
//! contain assignments, `Eval`s, comments and attributes; a label, a jump, a
//! branch, a loop or a terminator ends it (control could enter between the two
//! stores from elsewhere, or leave before the overwrite runs).
//!
//! ## Safety
//!
//! * The RHS must be VALUE-PURE: a local, a literal, a `MethodInfo(...)` token,
//!   `typeof`, `&x`, or arithmetic over those. A call, an allocation, a memory
//!   read (`field`, `[i]`, `*(T*)(p + n)`) or a lambda disqualifies the store,
//!   because dropping it would drop an effect (or a fault).
//! * Only plain identifiers are considered; `this.f`, `a[i]` and
//!   `*(int*)(p + 8)` are memory, not locals.
//! * Only untyped assignments are removed: a typed one is a declaration, and
//!   dropping it would leave the surviving write undeclared.

use super::super::csharp::{Expr, Stmt};
use super::{children_mut, expr_mentions};

/// Remove dead stores to plain locals. Returns the number removed.
pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut count = 0;
    sweep_list(stmts, &mut count);
    count
}

fn sweep_list(list: &mut Vec<Stmt>, count: &mut usize) {
    let mut i = 0;
    while i < list.len() {
        let dead = match &list[i] {
            Stmt::Assign(target, ty, value) => {
                ty.is_none()
                    && plain_local(target)
                    && value_pure(value)
                    && overwritten_before_read(list, i, target)
            }
            _ => false,
        };
        if dead {
            list.remove(i);
            *count += 1;
            continue;
        }
        i += 1;
    }
    for stmt in list.iter_mut() {
        for child in children_mut(stmt) {
            sweep_list(child, count);
        }
    }
}

/// Is `name` overwritten before anything reads it, scanning forward from
/// `from + 1` over a straight-line run?
fn overwritten_before_read(list: &[Stmt], from: usize, name: &str) -> bool {
    for stmt in &list[from + 1..] {
        match stmt {
            Stmt::Assign(target, _, value) => {
                // A plain target equal to NAME is the overwrite itself; any
                // other target (`a[i]`, `*(T*)(p + 8)`, `this.f`) may READ
                // NAME through its rendered text.
                if target == name {
                    return true;
                }
                if expr_mentions(value, name) || target_mentions(target, name) {
                    return false;
                }
            }
            Stmt::Eval(e) => {
                if expr_mentions(e, name) {
                    return false;
                }
            }
            Stmt::Attribute { args, .. } => {
                if args.iter().any(|a| expr_mentions(a, name)) {
                    return false;
                }
            }
            Stmt::Comment(_) => {}
            // Control flow, a label, a nested construct or a terminator: the
            // overwrite is no longer on every path from the store.
            _ => return false,
        }
    }
    false
}

/// A rendered assignment target can embed local names as bare tokens
/// (`*(int*)((nint)&intValue +4)`), which no expression walker sees.
fn target_mentions(target: &str, name: &str) -> bool {
    target
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|tok| tok == name)
}

/// A plain local identifier — not `this.x`, `a[i]`, `*(T*)(p + n)`.
fn plain_local(target: &str) -> bool {
    let mut chars = target.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// No call, no allocation, no memory read: dropping the store drops no
/// effect and cannot suppress a fault.
fn value_pure(e: &Expr) -> bool {
    match e {
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::MissingAbiArg
        | Expr::Unknown
        | Expr::MethodGroup(_)
        | Expr::TypeOf(_) => true,
        Expr::Cast(_, inner) | Expr::AddressOf(inner) | Expr::Not(inner) | Expr::Neg(inner) => {
            value_pure(inner)
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => value_pure(a) && value_pure(b),
        Expr::Cond(c, a, b) => value_pure(c) && value_pure(a) && value_pure(b),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::csharp::{BinOp, CmpKind};
    use super::*;

    fn v(name: &str) -> Expr {
        Expr::Var(name.into())
    }

    fn assign(name: &str, rhs: Expr) -> Stmt {
        Stmt::Assign(name.into(), None, rhs)
    }

    #[test]
    fn removes_a_store_the_next_statement_overwrites() {
        let mut stmts = vec![
            assign("x", Expr::MethodGroup("T..ctor".into())),
            assign("y", Expr::Int(1)),
            assign("x", Expr::Int(0)),
            Stmt::Eval(v("x")),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::Assign(t, _, Expr::Int(1)) if t == "y"));
    }

    #[test]
    fn keeps_a_store_that_is_read_first() {
        let mut stmts = vec![
            assign("x", Expr::Int(1)),
            Stmt::Eval(v("x")),
            assign("x", Expr::Int(2)),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(stmts.len(), 3);
    }

    #[test]
    fn keeps_a_store_whose_rhs_has_an_effect() {
        let call = Expr::Call {
            recv: None,
            name: "Compute".into(),
            args: vec![],
        };
        let mut stmts = vec![assign("x", call), assign("x", Expr::Int(0))];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn keeps_a_store_whose_rhs_reads_memory() {
        let load = Expr::Field(Box::new(v("this")), "count".into());
        let mut stmts = vec![assign("x", load), assign("x", Expr::Int(0))];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn a_label_between_the_two_stores_keeps_the_first() {
        let mut stmts = vec![
            assign("x", Expr::Int(1)),
            Stmt::Label("L3".into()),
            assign("x", Expr::Int(0)),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn an_index_write_counts_as_a_read_of_the_base() {
        let mut stmts = vec![
            assign("x", Expr::Int(1)),
            assign("*(int*)((nint)&x +4)", Expr::Int(0)),
            assign("x", Expr::Int(0)),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn removes_a_dead_store_inside_a_nested_arm() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(CmpKind::Ne, Box::new(v("a")), Box::new(Expr::Int(0))),
            then_body: vec![assign("x", Expr::Int(1)), assign("x", Expr::Int(2))],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 1);
    }

    #[test]
    fn a_typed_assignment_is_a_declaration_and_is_kept() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), Some("var".into()), Expr::Int(1)),
            assign("x", Expr::Int(0)),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn arithmetic_over_locals_is_pure() {
        let mut stmts = vec![
            assign(
                "x",
                Expr::Bin(BinOp::Add, Box::new(v("a")), Box::new(Expr::Int(1))),
            ),
            assign("x", Expr::Int(0)),
        ];
        assert_eq!(apply(&mut stmts), 1);
    }
}