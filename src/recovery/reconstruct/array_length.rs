//! Array length reconstruction: `*(nint*)(arr +24)` -> `arr.Length`.
//!
//! ## What this pass does
//!
//! IL2CPP stores the array length at offset +24 from the array pointer. The
//! decompiler emits this as a raw pointer dereference:
//!
//! ```text
//! *(nint*)(arr +24)     ->     arr.Length
//! ```
//!
//! This appears thousands of times in loop bounds like
//! `while (i < *(nint*)(arr +24))` and is one of the highest-impact
//! expression-level rewrites for readability.
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - Offset must be exactly 24 (the IL2CPP array length slot)
//! - Size must be exactly 8 (nint is 8 bytes on 64-bit)
//!
//! A failed match is a no-op: the original `RawDeref` form is preserved.

use super::super::csharp::{Expr, Stmt};

/// Run the array length reconstruction over one method body.
///
/// Returns the number of `RawDeref(_, 24, 8)` expressions rewritten to
/// `Field(_, "Length")`.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut rewrites = 0;
    for stmt in stmts.iter_mut() {
        rewrites += rewrite_stmt(stmt);
    }
    rewrites
}

/// Rewrite a single statement, recursing into nested bodies.
fn rewrite_stmt(stmt: &mut Stmt) -> usize {
    let mut rewrites = 0;
    match stmt {
        Stmt::Declare(..) => {}
        Stmt::Eval(expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::Assign(_, _, expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::Return(Some(expr)) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::Throw(expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::YieldReturn(expr) => {
            rewrites += rewrite_expr(expr);
        }
        Stmt::YieldBreak => {}
        Stmt::EventAssign { field, value, .. } => {
            rewrites += rewrite_expr(field);
            rewrites += rewrite_expr(value);
        }
        Stmt::IndexAssign { collection, index, value } => {
            rewrites += rewrite_expr(collection);
            rewrites += rewrite_expr(index);
            rewrites += rewrite_expr(value);
        }
        Stmt::GotoCase(_) => {}
        Stmt::StringSwitch { disc: _, cases, tail } => {
            for (_, b) in cases {
                rewrites += apply(b);
            }
            if let Some(t) = tail {
                rewrites += apply(t);
            }
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            rewrites += rewrite_expr(cond);
            rewrites += apply(then_body);
            rewrites += apply(else_body);
        }
        Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
            rewrites += rewrite_expr(cond);
            rewrites += apply(body);
        }
        Stmt::For {
            init,
            cond,
            incr,
            body,
        } => {
            for s in init {
                rewrites += rewrite_stmt(s);
            }
            if let Some(c) = cond {
                rewrites += rewrite_expr(c);
            }
            for s in incr {
                rewrites += rewrite_stmt(s);
            }
            rewrites += apply(body);
        }
        Stmt::ForEach { collection: _, body, .. } => {
            rewrites += apply(body);
        }
        Stmt::Using { decl, body } => {
            rewrites += rewrite_stmt(decl);
            rewrites += apply(body);
        }
        Stmt::Lock { expr, body } => {
            rewrites += rewrite_expr(expr);
            rewrites += apply(body);
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            rewrites += apply(try_body);
            for (_, b) in catches {
                rewrites += apply(b);
            }
            if let Some(fb) = finally_body {
                rewrites += apply(fb);
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases {
                rewrites += apply(b);
            }
            rewrites += apply(tail);
            if let Some(arm) = default {
                rewrites += apply(arm);
            }
        }
        Stmt::Return(None)
        | Stmt::Goto(_)
        | Stmt::Label(_)
        | Stmt::Comment(_)
        | Stmt::Attribute { .. }
        | Stmt::Break
        | Stmt::Continue => {}
    }
    rewrites
}

/// Recursively rewrite `RawDeref(base, 24, 8)` to `Field(base, "Length")`
/// within an expression tree.
///
/// Post-order traversal: children are rewritten first, so nested derefs
/// are handled naturally.
fn rewrite_expr(expr: &mut Expr) -> usize {
    let mut rewrites = 0;

    // First, recurse into subexpressions (post-order).
    match expr {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            rewrites += rewrite_expr(a);
            rewrites += rewrite_expr(b);
        }
        Expr::Not(e) | Expr::Neg(e) | Expr::Cast(_, e) | Expr::AddressOf(e) | Expr::Await(e) => {
            rewrites += rewrite_expr(e);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            rewrites += rewrite_expr(r);
        }
        Expr::Cond(c, a, b) => {
            rewrites += rewrite_expr(c);
            rewrites += rewrite_expr(a);
            rewrites += rewrite_expr(b);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                rewrites += rewrite_expr(r);
            }
            for a in args {
                rewrites += rewrite_expr(a);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                rewrites += rewrite_expr(a);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for e in items {
                rewrites += rewrite_expr(e);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for e in ctor_args {
                rewrites += rewrite_expr(e);
            }
            for e in items {
                rewrites += rewrite_expr(e);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for e in ctor_args {
                rewrites += rewrite_expr(e);
            }
            for (_, v) in fields {
                rewrites += rewrite_expr(v);
            }
        }
        Expr::Index { collection, index } => {
            rewrites += rewrite_expr(collection);
            rewrites += rewrite_expr(index);
        }
        Expr::Lambda { body, .. } => {
            rewrites += rewrite_expr(body);
        }
        Expr::NullConditional { receiver, access } => {
            rewrites += rewrite_expr(receiver);
            rewrites += rewrite_expr(access);
        }
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::Unknown
        | Expr::MissingAbiArg
        | Expr::StaticField(..)
        | Expr::MethodGroup(_)
        | Expr::PatternMatch { .. }
        | Expr::StringInterp { .. }
        | Expr::Tuple { .. }
        | Expr::MethodChain { .. } | Expr::SwitchExpr { .. }
        | Expr::Range { .. } | Expr::IndexFromEnd(_) | Expr::TypeOf(_) => {}
    }

    // Now check if THIS expression is a RawDeref with offset 24 and size 8.
    //
    // Exactly that shape, and nothing wider, is provably a length: in
    // Il2CppArray the slots are klass(0) / monitor(8) / bounds(16) /
    // max_length(24). Offset 16 is `bounds` — a POINTER, non-null only for
    // multi-dimensional or sliced arrays — so rewriting it as a count invents
    // a `.Count` on ordinary objects whose own field lives at 0x10. And a
    // 4-byte read at 0x18 is `List<T>._size`, not `max_length`; telling the
    // two apart needs the receiver's type from metadata, not a width guess.
    if let Expr::RawDeref(base, 24, w) = expr {
        // The RENDERER prints both width 8 and width 0 as `nint` (0 means
        // "pointer width / not stated"). A nint-width read at +24 is the array
        // `max_length` slot; the 4-byte List `_size` renders as `*(int*)` and
        // stays untouched. Matching only 8 left 8,423 `*(nint*)(x +24)` sites
        // in the dump.
        if matches!(w, 0 | 8) {
            let base = std::mem::replace(base.as_mut(), Expr::Null);
            *expr = Expr::Field(Box::new(base), "Length".to_string());
            rewrites += 1;
        }
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::{BinOp, CmpKind};

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    /// Helper: build `*(nint*)(base + offset)` with given size.
    fn raw_deref(base: Expr, offset: i64, size: u8) -> Expr {
        Expr::RawDeref(Box::new(base), offset, size)
    }

    // -----------------------------------------------------------------------
    // Positive rewrites
    // -----------------------------------------------------------------------

    /// 1. Simple array length: `*(nint*)(arr +24)` -> `arr.Length`
    #[test]
    fn simple_array_length() {
        let mut stmts = vec![Stmt::Eval(raw_deref(var("arr"), 24, 8))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, field)) => {
                assert!(matches!(recv.as_ref(), Expr::Var(name) if name == "arr"));
                assert_eq!(field, "Length");
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    /// 2. Array length in while condition: `while (i < *(nint*)(arr +24))`
    #[test]
    fn array_length_in_while_condition() {
        let cond = Expr::Cmp(
            CmpKind::Lt,
            Box::new(var("i")),
            Box::new(raw_deref(var("arr"), 24, 8)),
        );
        let mut stmts = vec![Stmt::While {
            cond,
            body: vec![Stmt::Eval(var("body"))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { cond, .. } => match cond {
                Expr::Cmp(_, _, rhs) => match rhs.as_ref() {
                    Expr::Field(recv, field) => {
                        assert!(matches!(recv.as_ref(), Expr::Var(name) if name == "arr"));
                        assert_eq!(field, "Length");
                    }
                    other => panic!("expected Field, got {other:?}"),
                },
                other => panic!("expected Cmp, got {other:?}"),
            },
            other => panic!("expected While, got {other:?}"),
        }
    }

    /// 3. Array length in assignment: `len = *(nint*)(arr +24)`
    #[test]
    fn array_length_in_assignment() {
        let mut stmts = vec![Stmt::Assign("len".into(), None, raw_deref(var("arr"), 24, 8))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Field(recv, field)) => {
                assert_eq!(target, "len");
                assert!(matches!(recv.as_ref(), Expr::Var(name) if name == "arr"));
                assert_eq!(field, "Length");
            }
            other => panic!("expected Assign(_, _, Field), got {other:?}"),
        }
    }

    /// 4. Array length with field receiver: `*(nint*)(this.items +24)` -> `this.items.Length`
    #[test]
    fn array_length_with_field_receiver() {
        let base = Expr::Field(Box::new(Expr::This), "items".into());
        let mut stmts = vec![Stmt::Eval(raw_deref(base, 24, 8))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, field)) => {
                assert!(matches!(recv.as_ref(), Expr::Field(_, _)));
                assert_eq!(field, "Length");
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    /// 5. Array length with complex receiver: `*(nint*)(GetArray() +24)` -> `GetArray().Length`
    #[test]
    fn array_length_with_call_receiver() {
        let base = Expr::Call {
            recv: None,
            name: "GetArray".into(),
            args: vec![],
        };
        let mut stmts = vec![Stmt::Eval(raw_deref(base, 24, 8))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, field)) => {
                assert!(matches!(recv.as_ref(), Expr::Call { .. }));
                assert_eq!(field, "Length");
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    /// 6. Array length in binary expression: `*(nint*)(arr +24) - 1` -> `arr.Length - 1`
    #[test]
    fn array_length_in_binary_expression() {
        let expr = Expr::Bin(
            BinOp::Sub,
            Box::new(raw_deref(var("arr"), 24, 8)),
            Box::new(Expr::Int(1)),
        );
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(_, lhs, rhs)) => {
                assert!(matches!(lhs.as_ref(), Expr::Field(_, _)));
                assert!(matches!(rhs.as_ref(), Expr::Int(1)));
            }
            other => panic!("expected Eval(Bin), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    /// 7. Non-match: offset 32 (not 24) — must NOT rewrite.
    #[test]
    fn non_match_offset_32() {
        let mut stmts = vec![Stmt::Eval(raw_deref(var("arr"), 32, 8))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "offset 32 must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(_, 32, 8))));
    }

    /// 8. Non-match: offset 16 (not 24) — must NOT rewrite.
    #[test]
    fn non_match_offset_16() {
        let mut stmts = vec![Stmt::Eval(raw_deref(var("arr"), 16, 8))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "offset 16 must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(_, 16, 8))));
    }

    /// 9. Non-match: size 4 (not 8) — must NOT rewrite.
    #[test]
    fn non_match_size_4() {
        let mut stmts = vec![Stmt::Eval(raw_deref(var("arr"), 24, 4))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "size 4 must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::RawDeref(_, 24, 4))));
    }

    /// 10. Multiple array lengths in one statement.
    #[test]
    fn multiple_array_lengths_in_one_statement() {
        let lhs = raw_deref(var("a"), 24, 8);
        let rhs = raw_deref(var("b"), 24, 8);
        let expr = Expr::Bin(BinOp::Add, Box::new(lhs), Box::new(rhs));
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both RawDeref(_, 24, 8) should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(_, a, b)) => {
                assert!(matches!(a.as_ref(), Expr::Field(_, _)));
                assert!(matches!(b.as_ref(), Expr::Field(_, _)));
            }
            other => panic!("expected Eval(Bin), got {other:?}"),
        }
    }

    /// 11. Rendering: verify that the rewritten expression renders as `arr.Length`.
    #[test]
    fn array_length_renders_correctly() {
        let expr = Expr::Field(Box::new(var("arr")), "Length".to_string());
        let mut s = String::new();
        expr.render(&mut s);
        assert_eq!(s, "arr.Length");
    }

    /// 12. Nested: array length inside a nested structure (if body).
    #[test]
    fn array_length_in_nested_if_body() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Assign(
                "len".into(),
                None,
                raw_deref(var("items"), 24, 8),
            )],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(
                    &then_body[0],
                    Stmt::Assign(_, _, Expr::Field(_, f)) if f == "Length"
                ));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }
}
