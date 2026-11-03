//! Recover the inlined `List<T>.Add` fast/slow path as `list.Add(x)`.
//!
//! IL2CPP inlines small collection methods. The inlined body of `Add` is the
//! capacity check plus the two paths:
//!
//! ```text
//! if (list.Count >= list._items.Length)
//! {
//!     list.Add(x);                       // the slow path CALLS Add itself
//! } else
//! {
//!     list.Count = list.Count + 1;       // _size++ (Count is a property read)
//!     *(nint*)(list._items + (list.Count << 3) +32) = x;
//!     …tail…                             // duplicated continuation
//! }
//! …tail…                                 // the real continuation
//! ```
//!
//! The source is `list.Add(x); …tail…`. The recognizer keeps ONE `Add` call
//! and leaves the continuation in place, which removes the whole diamond and
//! the raw element store. It fires only when every part matches exactly:
//! receiver renders equal in the condition, the call and the store; the stored
//! value equals the call's argument; and the else-branch tail (if any) is
//! structurally identical to the statements that follow the `if`.

use super::super::csharp::{BinOp, CmpKind, Expr, Stmt};
use super::children_mut;

fn expr_render(e: &Expr) -> String {
    let mut s = String::new();
    e.render(&mut s);
    s
}

fn stmt_dbg(s: &Stmt) -> String {
    format!("{s:?}")
}

/// `*(T*)(BODY)` → BODY.
fn deref_body(target: &str) -> Option<&str> {
    let rest = target.strip_prefix("*(")?;
    let close = rest.find("*)")?;
    let after = &rest[close + 2..];
    after.strip_prefix('(')?.strip_suffix(')')
}

/// The `Add` call in the then-arm: (receiver rendering, argument).
fn add_call(then_body: &[Stmt]) -> Option<(String, Expr)> {
    let [Stmt::Eval(Expr::Call { recv: Some(r), name, args })] = then_body else {
        return None;
    };
    if name.rsplit('.').next() != Some("Add") || args.len() != 1 {
        return None;
    }
    Some((expr_render(r), args[0].clone()))
}

fn cond_matches(cond: &Expr, recv: &str) -> bool {
    let Expr::Cmp(kind, a, b) = cond else {
        return false;
    };
    if !matches!(kind, CmpKind::Ge | CmpKind::Gt) {
        return false;
    }
    expr_render(a) == format!("{recv}.Count") && expr_render(b) == format!("{recv}._items.Length")
}

/// Does `s` increment `recv.Count` by one?
fn count_increment(s: &Stmt, recv: &str) -> bool {
    let Stmt::Assign(t, None, Expr::Bin(BinOp::Add, a, b)) = s else {
        return false;
    };
    *t == format!("{recv}.Count")
        && expr_render(a) == format!("{recv}.Count")
        && matches!(b.as_ref(), Expr::Int(1))
}

/// Does `s` store `value` as the next element of `recv._items`?
fn element_store(s: &Stmt, recv: &str, value: &Expr) -> bool {
    match s {
        Stmt::Assign(t, None, rhs) => {
            let Some(body) = deref_body(t) else {
                return false;
            };
            body == format!("{recv}._items + ({recv}.Count << 3) +32")
                && expr_render(rhs) == expr_render(value)
        }
        Stmt::IndexAssign {
            collection,
            index,
            value: v,
        } => {
            expr_render(collection) == format!("{recv}._items")
                && expr_render(index) == format!("{recv}.Count")
                && expr_render(v) == expr_render(value)
        }
        _ => false,
    }
}

/// Match at `list[i]`; returns the replacement statement when the whole
/// duplicated continuation is consistent with what follows.
fn match_add(list: &[Stmt], i: usize) -> Option<Stmt> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = &list[i]
    else {
        return None;
    };
    let (recv, value) = add_call(then_body)?;
    if !cond_matches(cond, &recv) {
        return None;
    }
    if else_body.len() < 2 || !count_increment(&else_body[0], &recv) {
        return None;
    }
    if !element_store(&else_body[1], &recv, &value) {
        return None;
    }
    // The else arm may duplicate the continuation that follows the `if`.
    let tail = &else_body[2..];
    if !tail.is_empty() {
        let following = list.get(i + 1..i + 1 + tail.len())?;
        if tail.len() != following.len()
            || tail
                .iter()
                .zip(following)
                .any(|(a, b)| stmt_dbg(a) != stmt_dbg(b))
        {
            return None;
        }
    }
    Some(then_body[0].clone())
}

fn walk(list: &mut Vec<Stmt>, n: &mut usize) {
    let mut i = 0;
    while i < list.len() {
        if let Some(add) = match_add(list, i) {
            list[i] = add;
            *n += 1;
            continue;
        }
        for k in children_mut(&mut list[i]) {
            walk(k, n);
        }
        i += 1;
    }
}

/// Returns the number of inlined `Add` bodies collapsed.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    if crate::recovery::env_flags::flag("ILR_NOCOLLADD") {
        return 0;
    }
    let mut n = 0;
    walk(stmts, &mut n);
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::Expr;

    fn v(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }
    fn f(e: Expr, name: &str) -> Expr {
        Expr::Field(Box::new(e), name.to_string())
    }

    fn add_block(tail_dup: bool, value_name: &str) -> Vec<Stmt> {
        let recv = Expr::Field(Box::new(Expr::This), "list".into());
        let add = Stmt::Eval(Expr::Call {
            recv: Some(Box::new(recv.clone())),
            name: "Add".into(),
            args: vec![v(value_name)],
        });
        let mut fast = vec![
            Stmt::Assign(
                "this.list.Count".into(),
                None,
                Expr::Bin(
                    BinOp::Add,
                    Box::new(f(f(Expr::This, "list"), "Count")),
                    Box::new(Expr::Int(1)),
                ),
            ),
            Stmt::Assign(
                "*(nint*)(this.list._items + (this.list.Count << 3) +32)".into(),
                None,
                v(value_name),
            ),
        ];
        if tail_dup {
            fast.push(Stmt::Eval(v("after")));
        }
        vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ge,
                    Box::new(f(f(Expr::This, "list"), "Count")),
                    Box::new(f(f(f(Expr::This, "list"), "_items"), "Length")),
                ),
                then_body: vec![add],
                else_body: fast,
            },
            Stmt::Eval(v("after")),
        ]
    }

    #[test]
    fn an_inlined_add_without_tail_duplication_collapses() {
        let mut stmts = add_block(false, "x");
        stmts.pop();
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "Add"));
    }

    #[test]
    fn an_inlined_add_with_a_duplicated_tail_collapses_to_one_tail() {
        let mut stmts = add_block(true, "x");
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2, "{stmts:?}");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "Add"));
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::Var(n)) if n == "after"));
    }

    #[test]
    fn a_different_stored_value_is_kept() {
        let mut stmts = add_block(false, "x");
        stmts.pop();
        if let Stmt::If { else_body, .. } = &mut stmts[0] {
            else_body[1] = Stmt::Assign(
                "*(nint*)(this.list._items + (this.list.Count << 3) +32)".into(),
                None,
                v("y"),
            );
        }
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn a_mismatched_duplicated_tail_is_kept() {
        let mut stmts = add_block(true, "x");
        stmts.pop();
        stmts.push(Stmt::Eval(v("other")));
        assert_eq!(apply(&mut stmts), 0);
    }
}
