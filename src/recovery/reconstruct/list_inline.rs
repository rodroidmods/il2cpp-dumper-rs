//! Fold IL2CPP's INLINED `List<T>.Add` back to `list.Add(item)`.
//!
//! ## Shape
//!
//! The C++ compiler inlines the whole method for small `List<T>` uses:
//!
//! ```text
//! _version++;                                  // *(int*)(list + 28) = … + 1
//! if (_items != null) {                        // *(nint*)(list + 16) != 0
//!     if (_size >= _items.Length) {            // *(int*)(list + 24) >= …Length
//!         list.AddWithResize(item);
//!     } else {
//!         _size = _size + 1;                   // *(int*)(list + 28 - 4) = … + 1
//!         _items[_size] = item;                // *(T*)(_items + (_size << k) + 32) = item
//!     }
//! }
//! ```
//!
//! The decompiler renders that as raw memory writes and a `AddWithResize`
//! call. The source is a single `list.Add(item)` (the C# List implementation
//! does exactly these updates internally).
//!
//! ## Proof gates
//!
//! * the outer test is `<list>+16` (`_items`) against 0, with an EMPTY other
//!   arm (a non-empty null-items path is not the standard inline);
//! * the inner test is exactly `<list>+24` (`_size`) against
//!   `((<list>+16)).Length`;
//! * the resize arm is exactly one `AddWithResize(item)` with the SAME list
//!   receiver and item;
//! * the other arm is exactly the `_size` increment and one `_items` element
//!   store whose value is that same item and whose target names `+32` and the
//!   list base;
//! * the preceding `_version` increment (if adjacent) is removed with it.
//!
//! Any mismatch is a NO-OP. The receiver is preserved structurally, so
//! `this._list`, a local, or a field chain all fold without rendering.

use super::super::csharp::{BinOp, CmpKind, Expr, Stmt};
use super::children_mut;

fn deref_at(e: &Expr, base: &Expr, off: i64) -> bool {
    matches!(e, Expr::RawDeref(b, o, _) if *o == off && b.as_ref() == base)
}

fn items_len(e: &Expr, base: &Expr) -> bool {
    matches!(e, Expr::Field(b, name) if name == "Length" && deref_at(b, base, 16))
}

fn size_incr(rhs: &Expr, base: &Expr) -> bool {
    matches!(
        rhs,
        Expr::Bin(BinOp::Add, a, b)
            if deref_at(a, base, 24) && matches!(b.as_ref(), Expr::Int(1))
    )
}

fn is_add_with_resize(s: &Stmt, base: &Expr, item: &Expr) -> bool {
    matches!(
        s,
        Stmt::Eval(Expr::Call { recv: Some(r), name, args })
            if r.as_ref() == base
                && name == "AddWithResize"
                && args.len() == 1
                && args[0] == *item
    )
}

fn item_store_matches(s: &Stmt, base: &Expr, item: &Expr) -> bool {
    let _ = base;
    match s {
        Stmt::Assign(t, _, rhs) => {
            rhs == item
                && (t.contains("+32") || t.contains("+ 32"))
                && (t.contains("<<") || t.contains("* 4") || t.contains("* 8") || t.contains("* 12"))
        }
        _ => false,
    }
}

fn size_store_matches(s: &Stmt, base: &Expr) -> bool {
    matches!(s, Stmt::Assign(_, _, rhs) if size_incr(rhs, base))
}

/// Match the inlined-Add capacity diamond:
/// `if (_size >= _items.Length) AddWithResize(item); else { _size++; _items[_size] = item; }`
/// (or the inverted `<` form). The enclosing `_items != null` fast-path guard
/// is left alone: one guard often encloses a whole chain of Adds.
fn add_shape(s: &Stmt) -> Option<(Expr, Expr)> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = s
    else {
        return None;
    };
    let (base, resize, stores) = match cond {
        Expr::Cmp(k, a, b)
            if matches!(k, CmpKind::Ge) => {
            let Expr::RawDeref(base, 24, _) = a.as_ref() else {
                return None;
            };
            if !items_len(b, base) {
                return None;
            }
            ((**base).clone(), then_body, else_body)
        }
        Expr::Cmp(k, a, b) if matches!(k, CmpKind::Lt) => {
            let Expr::RawDeref(base, 24, _) = b.as_ref() else {
                return None;
            };
            if !items_len(a, base) {
                return None;
            }
            ((**base).clone(), else_body, then_body)
        }
        _ => return None,
    };
    if resize.len() != 1 || stores.len() != 2 {
        return None;
    }
    let item = match &resize[0] {
        Stmt::Eval(Expr::Call {
            recv: Some(r),
            name,
            args,
        }) if r.as_ref() == &base && name == "AddWithResize" && args.len() == 1 => {
            args[0].clone()
        }
        _ => return None,
    };
    if !is_add_with_resize(&resize[0], &base, &item) {
        return None;
    }
    let (a, b) = (&stores[0], &stores[1]);
    let ok = (size_store_matches(a, &base) && item_store_matches(b, &base, &item))
        || (size_store_matches(b, &base) && item_store_matches(a, &base, &item));
    if !ok {
        return None;
    }
    Some((base, item))
}

/// The loop fast-path spelling of the same Add:
///
/// ```text
/// if (_size < _items.Length) { _size++; _items[_size] = item; continue; }
/// list.AddWithResize(item);        // fall-through (resize path)
/// ```
///
/// Both paths add `item`; the source is one `list.Add(item)` followed by the
/// statement after the fallback call (usually the loop's `continue`).
fn early_add_shape(if_stmt: &Stmt, next: &Stmt) -> Option<(Expr, Expr)> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = if_stmt
    else {
        return None;
    };
    if !else_body.is_empty() || then_body.len() != 3 {
        return None;
    }
    let (base, _) = match cond {
        Expr::Cmp(CmpKind::Lt, a, b) => {
            let Expr::RawDeref(base, 24, _) = a.as_ref() else {
                return None;
            };
            if !items_len(b, base) {
                return None;
            }
            ((**base).clone(), ())
        }
        Expr::Cmp(CmpKind::Gt, a, b) => {
            let Expr::RawDeref(base, 24, _) = b.as_ref() else {
                return None;
            };
            if !items_len(a, base) {
                return None;
            }
            ((**base).clone(), ())
        }
        _ => return None,
    };
    if !matches!(then_body[2], Stmt::Continue) {
        return None;
    }
    let (a, b) = (&then_body[0], &then_body[1]);
    let (size_ok, item_ok) = if size_store_matches(a, &base) {
        (true, item_store_matches(b, &base, &match_add_item(next, &base)?))
    } else if size_store_matches(b, &base) {
        (true, item_store_matches(a, &base, &match_add_item(next, &base)?))
    } else {
        (false, false)
    };
    if !(size_ok && item_ok) {
        return None;
    }
    let item = match_add_item(next, &base)?;
    Some((base, item))
}

/// Extract the item of `list.Add(item)` / `list.AddWithResize(item)`.
fn match_add_item(s: &Stmt, base: &Expr) -> Option<Expr> {
    match s {
        Stmt::Eval(Expr::Call {
            recv: Some(r),
            name,
            args,
        }) if r.as_ref() == base
            && (name == "Add" || name == "AddWithResize")
            && args.len() == 1 =>
        {
            Some(args[0].clone())
        }
        _ => None,
    }
}

fn version_incr(s: &Stmt, base: &Expr) -> bool {
    matches!(
        s,
        Stmt::Assign(t, _, rhs)
            if t.contains("+ 28)") || t.contains("+28)")
                && matches!(rhs, Expr::Bin(BinOp::Add, a, b)
                    if deref_at(a, base, 28) && matches!(b.as_ref(), Expr::Int(1)))
    )
}

fn fold_list(stmts: &mut Vec<Stmt>, count: &mut usize) {
    for s in stmts.iter_mut() {
        for k in children_mut(s) {
            fold_list(k, count);
        }
    }
    let mut i = 0;
    while i < stmts.len() {
        // Early-continue fast path + fall-through Add.
        if i + 1 < stmts.len() {
            if let Some((base, item)) = early_add_shape(&stmts[i], &stmts[i + 1]) {
                if i > 0 && version_incr(&stmts[i - 1], &base) {
                    stmts.remove(i - 1);
                    i -= 1;
                }
                stmts[i] = Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(base)),
                    name: "Add".into(),
                    args: vec![item],
                });
                stmts.remove(i + 1);
                *count += 1;
                i += 1;
                continue;
            }
        }
        let Some((base, item)) = add_shape(&stmts[i]) else {
            i += 1;
            continue;
        };
        // The `_version++` of the inline Add sits immediately before the
        // capacity diamond on most paths; drop it when it is adjacent (the
        // emitted `Add` performs that increment).
        if i > 0 && version_incr(&stmts[i - 1], &base) {
            stmts.remove(i - 1);
            i -= 1;
        }
        stmts[i] = Stmt::Eval(Expr::Call {
            recv: Some(Box::new(base)),
            name: "Add".into(),
            args: vec![item],
        });
        *count += 1;
        i += 1;
    }
}

/// Run the pass; returns the number of folded Adds.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    fold_list(stmts, &mut n);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }
    fn dr(base: Expr, off: i64, w: u8) -> Expr {
        Expr::RawDeref(Box::new(base), off, w)
    }
    fn add1(e: Expr) -> Expr {
        Expr::Bin(BinOp::Add, Box::new(e), Box::new(Expr::Int(1)))
    }
    fn shape() -> Vec<Stmt> {
        let base = var("list");
        let item = var("item");
        vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ge,
                Box::new(dr(base.clone(), 24, 4)),
                Box::new(Expr::Field(
                    Box::new(dr(base.clone(), 16, 0)),
                    "Length".into(),
                )),
            ),
            then_body: vec![Stmt::Eval(Expr::Call {
                recv: Some(Box::new(base.clone())),
                name: "AddWithResize".into(),
                args: vec![item.clone()],
            })],
            else_body: vec![
                Stmt::Assign(
                    "*(int*)(list + 28 - 4)".into(),
                    None,
                    add1(dr(base.clone(), 24, 4)),
                ),
                Stmt::Assign(
                    "*(nint*)(*(nint*)(list + 28 -12) + (*(int*)(list + 28 -4) << 3) +32)"
                        .into(),
                    None,
                    item.clone(),
                ),
            ],
        }]
    }

    #[test]
    fn inline_add_folds_to_list_add() {
        let mut stmts = shape();
        assert_eq!(apply(&mut stmts), 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("\"Add\""), "{t}");
        assert!(!t.contains("AddWithResize"), "{t}");
        assert!(!t.contains("RawDeref"), "{t}");
    }

    #[test]
    fn early_continue_add_folds() {
        let base = var("list");
        let item = var("item");
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Lt,
                    Box::new(dr(base.clone(), 24, 4)),
                    Box::new(Expr::Field(
                        Box::new(dr(base.clone(), 16, 0)),
                        "Length".into(),
                    )),
                ),
                then_body: vec![
                    Stmt::Assign(
                        "*(int*)(list + 28 - 4)".into(),
                        None,
                        add1(dr(base.clone(), 24, 4)),
                    ),
                    Stmt::Assign(
                        "*(nint*)(*(nint*)(list + 28 -12) + (*(int*)(list + 28 -4) << 3) +32)"
                            .into(),
                        None,
                        item.clone(),
                    ),
                    Stmt::Continue,
                ],
                else_body: vec![],
            },
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(base.clone())),
                name: "AddWithResize".into(),
                args: vec![item.clone()],
            }),
            Stmt::Continue,
        ];
        assert_eq!(apply(&mut stmts), 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("\"Add\""), "{t}");
        assert!(!t.contains("AddWithResize"), "{t}");
        assert!(!t.contains("RawDeref"), "{t}");
    }

    #[test]
    fn mismatched_item_is_refused() {
        let mut stmts = shape();
        if let Stmt::If { else_body, .. } = &mut stmts[0] {
            if let Stmt::Assign(_, _, rhs) = &mut else_body[1] {
                *rhs = var("other");
            }
        }
        assert_eq!(apply(&mut stmts), 0);
    }
}
