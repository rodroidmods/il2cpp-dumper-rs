//! Recover array-element expressions from IL2CPP's address arithmetic.
//!
//! ## Shape
//!
//! A value-type instance method takes its receiver BY REFERENCE, so the
//! native code passes the element's ADDRESS:
//!
//! ```text
//! Call { recv: (contents + (index << 2)) + 32, name: "ToString" }
//! RawDeref((contents + (index << 2)) + 32, 0, 4)
//! ```
//!
//! `+32` is the first element's offset in `Il2CppArray`; `index << 2` is the
//! element stride (4 bytes here, 8 for references). The managed source is
//! `contents[index]` — the address is never a C# value.
//!
//! ## Safety
//!
//! The rewrite requires exactly: an additive tree with one `+32` term, one
//! `index << k` term (k = 0..4), and one remaining term that must be a
//! memory path (`this.field`, a local, a static field, a deref or another
//! index). Strides are checked against the load width where one is known.
//! Anything else — user arithmetic that merely contains `+32` and a shift —
//! is left untouched because its base is not a memory path.
//!
//! ## Why a late pass
//!
//! The shape is produced by the renderer only after algebraic simplification
//! folded the data-pointer load and the byte offset into one expression, so
//! the early indexer pass (`translate_array_indexing`, which expects a literal
//! `RawDeref(base + (i<<k), 32, w)`) no longer matches it.

use super::children_mut;
use super::super::csharp::{BinOp, Expr, Stmt};

fn is_memory_path(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Var(_)
            | Expr::This
            | Expr::Field(..)
            | Expr::StaticField(..)
            | Expr::RawDeref(..)
            | Expr::Index { .. }
            | Expr::Cast(..)
            | Expr::Call { .. }
    )
}

fn add_terms<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) -> bool {
    match e {
        Expr::Bin(BinOp::Add, a, b) => add_terms(a, out) && add_terms(b, out),
        _ => {
            out.push(e);
            true
        }
    }
}

/// `base + (index << k) + 32` → `(base, index)`.
///
/// The stride is also emitted as a plain multiply (`index * 4`) — ARM64's
/// scaled addressing and algebraic simplification both produce it, and the
/// spilled-array shape `stack_465 + 32 + j * 4` was the dominant remaining
/// raw-address form. Only power-of-two strides are accepted, and the same
/// `+32`/memory-path requirements apply.
fn array_element_address(e: &Expr) -> Option<(Expr, Expr)> {
    let mut terms = Vec::new();
    if !add_terms(e, &mut terms) || terms.len() != 3 {
        return None;
    }
    let mut base: Option<&Expr> = None;
    let mut index: Option<Expr> = None;
    let mut saw_32 = false;
    for &t in &terms {
        if matches!(t, Expr::Int(32)) && !saw_32 {
            saw_32 = true;
            continue;
        }
        if let Expr::Bin(BinOp::Shl, idx, sh) = t {
            if let Expr::Int(k) = sh.as_ref() {
                if (0..=4).contains(k) && index.is_none() {
                    index = Some(idx.as_ref().clone());
                    continue;
                }
            }
        }
        if let Expr::Bin(BinOp::Mul, a, b) = t {
            let (idx, w) = match (a.as_ref(), b.as_ref()) {
                (idx, Expr::Int(w)) => (idx, *w),
                (Expr::Int(w), idx) => (idx, *w),
                _ => return None,
            };
            if matches!(w, 1 | 2 | 4 | 8 | 16) && index.is_none() {
                index = Some(idx.clone());
                continue;
            }
        }
        if base.is_none() && is_memory_path(t) {
            base = Some(t);
            continue;
        }
        return None;
    }
    let base = base?;
    let index = index?;
    Some((base.clone(), index))
}

/// Cells proven to be managed arrays: an element address with `+32` and a
/// power-of-two stride. Their `.Length` is the array length, and the renderer
/// prints it through the frame-cell `&cell`; unwrap that address afterwards.
fn collect_array_cells(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    fn expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
        match e {
            Expr::RawDeref(b, 0, _) => {
                if let Some((Expr::Var(cell), _)) = array_element_address(b) {
                    out.insert(cell);
                }
            }
            Expr::Call { recv: Some(r), .. } => {
                if let Some((Expr::Var(cell), _)) = array_element_address(r) {
                    out.insert(cell);
                }
            }
            _ => {}
        }
        for c in expr_children(e) {
            expr(c, out);
        }
    }
    for s in stmts {
        for e in direct_exprs(s) {
            expr(e, out);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_array_cells(k, out);
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

fn rewrite_expr(e: &mut Expr, cells: &std::collections::HashSet<String>) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += rewrite_expr(c, cells);
    }
    // Value-type receiver: `Call { recv: <element address> }` → `contents[i]`.
    if let Expr::Call { recv: Some(r), .. } = e {
        if let Some((base, idx)) = array_element_address(r) {
            *r = Box::new(Expr::Index {
                collection: Box::new(base),
                index: Box::new(idx),
            });
            n += 1;
        }
    }
    // The same address dereferenced: `*(T*)(<element address>)` → `contents[i]`.
    if let Expr::RawDeref(b, 0, _w) = e {
        if let Some((base, idx)) = array_element_address(b) {
            *e = Expr::Index {
                collection: Box::new(base),
                index: Box::new(idx),
            };
            n += 1;
        }
    }
    // `(&cell).Length` on a proven array cell → `cell.Length` (the `&` is the
    // frame-cell representation, not an address the source ever takes).
    if let Expr::Field(base, name) = e {
        if name == "Length" {
            if let Expr::AddressOf(inner) = base.as_ref() {
                if let Expr::Var(cell) = inner.as_ref() {
                    if cells.contains(cell) {
                        *e = Expr::Field(
                            Box::new(Expr::Var(cell.clone())),
                            name.clone(),
                        );
                        n += 1;
                    }
                }
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
        Expr::RawCall(_, args, _)
        | Expr::NewObj(_, args)
        | Expr::NewArr(_, args)
        | Expr::ArrayInit { items: args, .. } => args.iter_mut().collect(),
        _ => vec![],
    }
}

fn rewrite_stmt(s: &mut Stmt, cells: &std::collections::HashSet<String>) -> usize {
    let mut n = 0;
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += rewrite_stmt(x, cells);
        }
    }
    let mut fold = |e: &mut Expr| n += rewrite_expr(e, cells);
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

/// Run the pass. Returns the number of rewritten expressions.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut cells: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_array_cells(stmts, &mut cells);
    let mut n = 0;
    for s in stmts.iter_mut() {
        n += rewrite_stmt(s, &cells);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn field(base: Expr, f: &str) -> Expr {
        Expr::Field(Box::new(base), f.to_string())
    }

    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Bin(BinOp::Add, Box::new(a), Box::new(b))
    }

    fn shl(a: Expr, k: i64) -> Expr {
        Expr::Bin(BinOp::Shl, Box::new(a), Box::new(Expr::Int(k)))
    }

    #[test]
    fn value_receiver_becomes_index() {
        // (contents + (index << 2)) + 32 .ToString()
        let addr = add(add(field(Expr::This, "contents"), shl(var("index"), 2)), Expr::Int(32));
        let mut stmts = vec![Stmt::Return(Some(Expr::Call {
            recv: Some(Box::new(addr)),
            name: "ToString".into(),
            args: vec![],
        }))];
        assert_eq!(apply(&mut stmts), 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("Index"), "{t}");
        assert!(t.contains("contents"), "{t}");
        assert!(!t.contains("Int(32)"), "{t}");
    }

    #[test]
    fn element_load_becomes_index() {
        let addr = add(Expr::Int(32), add(field(Expr::This, "contents"), shl(var("i"), 2)));
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(Box::new(addr), 0, 4))];
        assert_eq!(apply(&mut stmts), 1);
        assert!(format!("{stmts:?}").contains("Index"));
    }

    #[test]
    fn multiply_stride_becomes_index() {
        // Spilled array cell: `stack_465 + 32 + j * 4`.
        let addr = add(
            add(var("stack_465"), Expr::Int(32)),
            Expr::Bin(BinOp::Mul, Box::new(var("j")), Box::new(Expr::Int(4))),
        );
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(Box::new(addr), 0, 4))];
        assert_eq!(apply(&mut stmts), 1);
        let t = format!("{stmts:?}");
        assert!(t.contains("Index"), "{t}");
        assert!(t.contains("stack_465"), "{t}");
    }

    #[test]
    fn array_cell_length_loses_the_address_of() {
        let addr = add(
            add(var("stack_465"), Expr::Int(32)),
            Expr::Bin(BinOp::Mul, Box::new(var("j")), Box::new(Expr::Int(4))),
        );
        let mut stmts = vec![
            Stmt::Eval(Expr::RawDeref(Box::new(addr), 0, 4)),
            Stmt::Eval(field(
                Expr::AddressOf(Box::new(var("stack_465"))),
                "Length",
            )),
        ];
        apply(&mut stmts);
        let t = format!("{stmts:?}");
        assert!(!t.contains("AddressOf"), "{t}");
    }

    #[test]
    fn length_of_a_non_array_cell_is_kept() {
        let mut stmts = vec![Stmt::Eval(field(
            Expr::AddressOf(Box::new(var("stack_9"))),
            "Length",
        ))];
        assert_eq!(apply(&mut stmts), 0);
        assert!(format!("{stmts:?}").contains("AddressOf"));
    }

    #[test]
    fn user_arithmetic_with_32_and_shift_is_kept() {
        // The base is not a memory path; the sum is plain arithmetic.
        let e = add(add(Expr::Int(1000), shl(var("i"), 2)), Expr::Int(32));
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(Box::new(e), 0, 4))];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn non_zero_deref_addend_is_kept() {
        let addr = add(add(field(Expr::This, "contents"), shl(var("i"), 2)), Expr::Int(32));
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(Box::new(addr), 4, 4))];
        assert_eq!(apply(&mut stmts), 0);
    }
}
