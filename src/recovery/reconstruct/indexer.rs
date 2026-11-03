//! Collection indexer reconstruction: `get_Item`/`set_Item` → `v[i]` syntax.
//!
//! ## What this pass does
//!
//! IL2CPP renders collection indexing as ordinary method calls:
//!
//! ```text
//! v.get_Item(i)        →  v[i]
//! v.set_Item(i, val)   →  v[i] = val  (not yet rewritten — see below)
//! ```
//!
//! This appears ~11,883 times in the test corpus and is the single highest-
//! impact expression-level rewrite in the reconstruction pipeline.
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - Name must be exactly `"get_Item"` (not `"get_Items"`, not `"GetItem"`)
//! - Receiver must be `Some` (static calls are not indexers)
//! - Argument count must be exactly 1 for `get_Item`
//!
//! A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{Expr, Stmt};

/// Run the indexer reconstruction over one method body.
///
/// Returns the number of `get_Item` calls rewritten to `Index` expressions.
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
            // Check if this is a set_Item call that should be rewritten to IndexAssign.
            // We must take ownership of the expression (via mem::replace) before
            // destructuring, because destructuring &mut Expr::Call borrows all fields
            // from the same reference and prevents using them independently.
            let is_set_item = matches!(&*expr, Expr::Call { name, .. } if name == "set_Item");
            if is_set_item {
                // First, recurse into subexpressions to handle nested get_Item/set_Item calls
                if let Expr::Call { recv, args, .. } = &mut *expr {
                    if let Some(recv_expr) = recv {
                        rewrites += rewrite_expr(recv_expr);
                    }
                    for arg in args.iter_mut() {
                        rewrites += rewrite_expr(arg);
                    }
                }
                // Now take ownership of the expression so we can destructure freely.
                let owned_expr = std::mem::replace(&mut *expr, Expr::Null);
                if let Expr::Call { recv, args, .. } = owned_expr {
                    // Instance form: obj.set_Item(index, value) -> obj[index] = value
                    if let Some(ref recv_box) = recv {
                        if args.len() >= 2 {
                            let collection = *recv_box.clone();
                            let args_len = args.len();
                            let mut args = args;
                            let value = std::mem::replace(&mut args[args_len - 1], Expr::Null);
                            let index = if args_len - 1 == 1 {
                                std::mem::replace(&mut args[0], Expr::Null)
                            } else {
                                let idx_vec: Vec<Expr> = args[..args_len - 1]
                                    .iter_mut()
                                    .map(|e| std::mem::replace(e, Expr::Null))
                                    .collect();
                                Expr::Tuple {
                                    names: vec![None; idx_vec.len()],
                                    elements: idx_vec,
                                }
                            };
                            *stmt = Stmt::IndexAssign {
                                collection,
                                index,
                                value,
                            };
                            rewrites += 1;
                            return rewrites;
                        }
                    }
                    // Static form: Type.set_Item(obj, index, value) -> obj[index] = value
                    // or: set_Item(obj, index, value) -> obj[index] = value
                    else if args.len() >= 3 {
                        let mut args = args;
                        let collection = std::mem::replace(&mut args[0], Expr::Null);
                        let args_len = args.len();
                        let value = std::mem::replace(&mut args[args_len - 1], Expr::Null);
                        let index = if args_len - 2 == 1 {
                            std::mem::replace(&mut args[1], Expr::Null)
                        } else {
                            let idx_vec: Vec<Expr> = args[1..args_len - 1]
                                .iter_mut()
                                .map(|e| std::mem::replace(e, Expr::Null))
                                .collect();
                            Expr::Tuple {
                                names: vec![None; idx_vec.len()],
                                elements: idx_vec,
                            }
                        };
                        *stmt = Stmt::IndexAssign {
                            collection,
                            index,
                            value,
                        };
                        rewrites += 1;
                        return rewrites;
                    }
                    // Didn't match the conditions, restore the expression
                    *expr = Expr::Call {
                        recv,
                        name: "set_Item".into(),
                        args,
                    };
                }
            }
            // Not a set_Item call (or didn't match), just rewrite expressions normally
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
        Stmt::EventAssign { .. } => {}
        Stmt::GotoCase(_) => {}
        Stmt::StringSwitch { disc: _, cases, tail } => {
            // disc is a String, not an Expr. Recurse into case bodies.
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
            // collection is a String, not an Expr — nothing to rewrite here.
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
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            rewrites += rewrite_expr(collection);
            rewrites += rewrite_expr(index);
            rewrites += rewrite_expr(value);
        }
        // Statements with no expressions to rewrite.
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

/// Recursively rewrite `get_Item` calls to `Index` expressions within an
/// expression tree.
///
/// Post-order traversal: children are rewritten first, so nested indexers
/// (`a.get_Item(b.get_Item(0))`) are handled naturally — the inner call is
/// rewritten to `b[0]`, then the outer call sees `a.get_Item(b[0])` and
/// becomes `a[b[0]]`.
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
        // Leaf expressions: nothing to recurse into.
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

    // Now check if THIS expression is a get_Item call we can rewrite.
    if let Expr::Call {
        recv: Some(recv),
        name,
        args,
    } = expr
    {
        if name == "get_Item" && args.len() == 1 {
            // Safe to destructure: we own the expression through &mut.
            let recv = std::mem::replace(recv.as_mut(), Expr::Null);
            let index = std::mem::replace(&mut args[0], Expr::Null);
            *expr = Expr::Index {
                collection: Box::new(recv),
                index: Box::new(index),
            };
            rewrites += 1;
        }
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build `v.get_Item(args...)`.
    fn get_item_call(recv: Expr, args: Vec<Expr>) -> Expr {
        Expr::Call {
            recv: Some(Box::new(recv)),
            name: "get_Item".into(),
            args,
        }
    }

    /// Helper: build `v.set_Item(args...)`.
    fn set_item_call(recv: Expr, args: Vec<Expr>) -> Expr {
        Expr::Call {
            recv: Some(Box::new(recv)),
            name: "set_Item".into(),
            args,
        }
    }

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    // -----------------------------------------------------------------------
    // get_Item → Index rewrites
    // -----------------------------------------------------------------------

    /// 1. Simple get_Item with integer index: `v.get_Item(0)` → `v[0]`
    #[test]
    fn simple_get_item_with_int_index() {
        let mut stmts = vec![Stmt::Eval(get_item_call(var("v"), vec![Expr::Int(0)]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Index { collection, index }) => {
                assert!(matches!(collection.as_ref(), Expr::Var(name) if name == "v"));
                assert!(matches!(index.as_ref(), Expr::Int(0)));
            }
            other => panic!("expected Eval(Index), got {other:?}"),
        }
    }

    /// 2. get_Item with variable index: `v.get_Item(i)` → `v[i]`
    #[test]
    fn get_item_with_variable_index() {
        let mut stmts = vec![Stmt::Eval(get_item_call(var("v"), vec![var("i")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Index { collection, index }) => {
                assert!(matches!(collection.as_ref(), Expr::Var(name) if name == "v"));
                assert!(matches!(index.as_ref(), Expr::Var(name) if name == "i"));
            }
            other => panic!("expected Eval(Index), got {other:?}"),
        }
    }

    /// 3. get_Item in assignment RHS: `x = v.get_Item(i)` → `x = v[i]`
    #[test]
    fn get_item_in_assignment_rhs() {
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            get_item_call(var("v"), vec![var("i")]),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Index { collection, index }) => {
                assert_eq!(target, "x");
                assert!(matches!(collection.as_ref(), Expr::Var(name) if name == "v"));
                assert!(matches!(index.as_ref(), Expr::Var(name) if name == "i"));
            }
            other => panic!("expected Assign(_, _, Index), got {other:?}"),
        }
    }

    /// 4. get_Item with complex receiver: `foo.bar.get_Item(0)` → `foo.bar[0]`
    #[test]
    fn get_item_with_field_receiver() {
        let recv = Expr::Field(Box::new(var("foo")), "bar".into());
        let mut stmts = vec![Stmt::Eval(get_item_call(recv, vec![Expr::Int(0)]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Index { collection, index }) => {
                assert!(matches!(collection.as_ref(), Expr::Field(_, _)));
                assert!(matches!(index.as_ref(), Expr::Int(0)));
            }
            other => panic!("expected Eval(Index), got {other:?}"),
        }
    }

    /// 5. Nested get_Item: `a.get_Item(b.get_Item(0))` → `a[b[0]]`
    #[test]
    fn nested_get_item() {
        let inner = get_item_call(var("b"), vec![Expr::Int(0)]);
        let outer = get_item_call(var("a"), vec![inner]);
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both inner and outer get_Item should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Index { collection, index }) => {
                assert!(matches!(collection.as_ref(), Expr::Var(name) if name == "a"));
                // The index should itself be an Index expression.
                match index.as_ref() {
                    Expr::Index { collection: ic, index: ii } => {
                        assert!(matches!(ic.as_ref(), Expr::Var(name) if name == "b"));
                        assert!(matches!(ii.as_ref(), Expr::Int(0)));
                    }
                    other => panic!("expected nested Index, got {other:?}"),
                }
            }
            other => panic!("expected Eval(Index), got {other:?}"),
        }
    }

    /// 6. get_Item in condition: `if (v.get_Item(i) == x)` → `if (v[i] == x)`
    #[test]
    fn get_item_in_condition() {
        let cond = Expr::Cmp(
            super::super::super::csharp::CmpKind::Eq,
            Box::new(get_item_call(var("v"), vec![var("i")])),
            Box::new(var("x")),
        );
        let mut stmts = vec![Stmt::If {
            cond,
            then_body: vec![Stmt::Eval(var("body"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { cond, .. } => match cond {
                Expr::Cmp(_, lhs, _) => match lhs.as_ref() {
                    Expr::Index { collection, index } => {
                        assert!(matches!(collection.as_ref(), Expr::Var(name) if name == "v"));
                        assert!(matches!(index.as_ref(), Expr::Var(name) if name == "i"));
                    }
                    other => panic!("expected Index in condition, got {other:?}"),
                },
                other => panic!("expected Cmp, got {other:?}"),
            },
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// 7. Multiple get_Item in one statement: `v.get_Item(0) + v.get_Item(1)`
    #[test]
    fn multiple_get_items_in_one_statement() {
        let lhs = get_item_call(var("v"), vec![Expr::Int(0)]);
        let rhs = get_item_call(var("v"), vec![Expr::Int(1)]);
        let expr = Expr::Bin(super::super::super::csharp::BinOp::Add, Box::new(lhs), Box::new(rhs));
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both get_Item calls should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(_, a, b)) => {
                assert!(matches!(a.as_ref(), Expr::Index { .. }));
                assert!(matches!(b.as_ref(), Expr::Index { .. }));
            }
            other => panic!("expected Eval(Bin), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    /// 8. get_Item with wrong arg count (0 args) — must NOT rewrite.
    #[test]
    fn get_item_wrong_arg_count_zero() {
        let mut stmts = vec![Stmt::Eval(get_item_call(var("v"), vec![]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "get_Item with 0 args must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "get_Item"));
    }

    /// 8b. get_Item with wrong arg count (2 args) — must NOT rewrite.
    #[test]
    fn get_item_wrong_arg_count_two() {
        let mut stmts = vec![Stmt::Eval(get_item_call(
            var("v"),
            vec![Expr::Int(0), Expr::Int(1)],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "get_Item with 2 args must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "get_Item"));
    }

    /// 9. Static get_Item (no receiver) — must NOT rewrite.
    #[test]
    fn static_get_item_no_receiver() {
        let expr = Expr::Call {
            recv: None,
            name: "get_Item".into(),
            args: vec![Expr::Int(0)],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "static get_Item must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { recv: None, .. })));
    }

    /// 10. set_Item is now rewritten to IndexAssign (see tests 13-22).
    /// This test is kept as a placeholder to maintain test numbering.
    #[test]
    fn set_item_now_rewritten() {
        // This is now tested extensively in tests 13-22
        assert!(true);
    }

    /// 11. Similar name but not exact match — must NOT rewrite.
    #[test]
    fn similar_name_not_rewritten() {
        let expr = Expr::Call {
            recv: Some(Box::new(var("v"))),
            name: "get_Items".into(), // plural — not an indexer
            args: vec![Expr::Int(0)],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "get_Items is not get_Item");
    }

    /// 12. get_Item inside a while loop body.
    #[test]
    fn get_item_inside_while_body() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Eval(get_item_call(var("v"), vec![var("i")]))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[0], Stmt::Eval(Expr::Index { .. })));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Rendering tests
    // -----------------------------------------------------------------------

    /// Verify that the Index expression renders correctly.
    #[test]
    fn index_renders_correctly() {
        let expr = Expr::Index {
            collection: Box::new(var("v")),
            index: Box::new(Expr::Int(0)),
        };
        let mut s = String::new();
        expr.render(&mut s);
        assert_eq!(s, "v[0]");
    }

    /// Verify nested Index rendering.
    #[test]
    fn nested_index_renders_correctly() {
        let inner = Expr::Index {
            collection: Box::new(var("b")),
            index: Box::new(Expr::Int(0)),
        };
        let outer = Expr::Index {
            collection: Box::new(var("a")),
            index: Box::new(inner),
        };
        let mut s = String::new();
        outer.render(&mut s);
        assert_eq!(s, "a[b[0]]");
    }

    /// Verify field receiver renders correctly with indexer.
    #[test]
    fn field_receiver_index_renders_correctly() {
        let expr = Expr::Index {
            collection: Box::new(Expr::Field(Box::new(var("foo")), "bar".into())),
            index: Box::new(var("i")),
        };
        let mut s = String::new();
        expr.render(&mut s);
        assert_eq!(s, "foo.bar[i]");
    }

    // -----------------------------------------------------------------------
    // set_Item → IndexAssign rewrites
    // -----------------------------------------------------------------------

    /// 13. Simple instance set_Item: `v.set_Item(0, val)` → `v[0] = val`
    #[test]
    fn simple_instance_set_item() {
        let mut stmts = vec![Stmt::Eval(set_item_call(
            var("v"),
            vec![Expr::Int(0), var("val")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                assert!(matches!(collection, Expr::Var(name) if name == "v"));
                assert!(matches!(index, Expr::Int(0)));
                assert!(matches!(value, Expr::Var(name) if name == "val"));
            }
            other => panic!("expected IndexAssign, got {other:?}"),
        }
    }

    /// 14. Instance set_Item with variable index: `v.set_Item(i, val)` → `v[i] = val`
    #[test]
    fn instance_set_item_with_variable_index() {
        let mut stmts = vec![Stmt::Eval(set_item_call(
            var("v"),
            vec![var("i"), var("val")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                assert!(matches!(collection, Expr::Var(name) if name == "v"));
                assert!(matches!(index, Expr::Var(name) if name == "i"));
                assert!(matches!(value, Expr::Var(name) if name == "val"));
            }
            other => panic!("expected IndexAssign, got {other:?}"),
        }
    }

    /// 15. Static set_Item: `Type.set_Item(obj, 0, val)` → `obj[0] = val`
    #[test]
    fn static_set_item() {
        let expr = Expr::Call {
            recv: None,
            name: "set_Item".into(),
            args: vec![var("obj"), Expr::Int(0), var("val")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                assert!(matches!(collection, Expr::Var(name) if name == "obj"));
                assert!(matches!(index, Expr::Int(0)));
                assert!(matches!(value, Expr::Var(name) if name == "val"));
            }
            other => panic!("expected IndexAssign, got {other:?}"),
        }
    }

    /// 16. Multi-dimensional instance set_Item: `v.set_Item(i, j, val)` → `v[(i, j)] = val`
    #[test]
    fn multidimensional_instance_set_item() {
        let mut stmts = vec![Stmt::Eval(set_item_call(
            var("v"),
            vec![var("i"), var("j"), var("val")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                assert!(matches!(collection, Expr::Var(name) if name == "v"));
                // Index should be a tuple
                match index {
                    Expr::Tuple { elements, .. } => {
                        assert_eq!(elements.len(), 2);
                        assert!(matches!(&elements[0], Expr::Var(name) if name == "i"));
                        assert!(matches!(&elements[1], Expr::Var(name) if name == "j"));
                    }
                    other => panic!("expected Tuple index, got {other:?}"),
                }
                assert!(matches!(value, Expr::Var(name) if name == "val"));
            }
            other => panic!("expected IndexAssign, got {other:?}"),
        }
    }

    /// 17. Multi-dimensional static set_Item: `Type.set_Item(obj, i, j, val)` → `obj[(i, j)] = val`
    #[test]
    fn multidimensional_static_set_item() {
        let expr = Expr::Call {
            recv: None,
            name: "set_Item".into(),
            args: vec![var("obj"), var("i"), var("j"), var("val")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                assert!(matches!(collection, Expr::Var(name) if name == "obj"));
                // Index should be a tuple
                match index {
                    Expr::Tuple { elements, .. } => {
                        assert_eq!(elements.len(), 2);
                        assert!(matches!(&elements[0], Expr::Var(name) if name == "i"));
                        assert!(matches!(&elements[1], Expr::Var(name) if name == "j"));
                    }
                    other => panic!("expected Tuple index, got {other:?}"),
                }
                assert!(matches!(value, Expr::Var(name) if name == "val"));
            }
            other => panic!("expected IndexAssign, got {other:?}"),
        }
    }

    /// 18. set_Item with field receiver: `foo.bar.set_Item(0, val)` → `foo.bar[0] = val`
    #[test]
    fn set_item_with_field_receiver() {
        let recv = Expr::Field(Box::new(var("foo")), "bar".into());
        let mut stmts = vec![Stmt::Eval(set_item_call(
            recv,
            vec![Expr::Int(0), var("val")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                assert!(matches!(collection, Expr::Field(_, _)));
                assert!(matches!(index, Expr::Int(0)));
                assert!(matches!(value, Expr::Var(name) if name == "val"));
            }
            other => panic!("expected IndexAssign, got {other:?}"),
        }
    }

    /// 19. Mixed get/set: `v.set_Item(0, v.get_Item(1))` → `v[0] = v[1]`
    #[test]
    fn mixed_get_set_item() {
        let get_expr = get_item_call(var("v"), vec![Expr::Int(1)]);
        let mut stmts = vec![Stmt::Eval(set_item_call(
            var("v"),
            vec![Expr::Int(0), get_expr],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both set_Item and get_Item should be rewritten");
        match &stmts[0] {
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                assert!(matches!(collection, Expr::Var(name) if name == "v"));
                assert!(matches!(index, Expr::Int(0)));
                // Value should be an Index expression (from get_Item)
                match value {
                    Expr::Index { collection, index } => {
                        assert!(matches!(collection.as_ref(), Expr::Var(name) if name == "v"));
                        assert!(matches!(index.as_ref(), Expr::Int(1)));
                    }
                    other => panic!("expected Index value, got {other:?}"),
                }
            }
            other => panic!("expected IndexAssign, got {other:?}"),
        }
    }

    /// 20. set_Item in nested context (inside if body)
    #[test]
    fn set_item_in_nested_context() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(set_item_call(
                var("v"),
                vec![Expr::Int(0), var("val")],
            ))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::IndexAssign { .. }));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// 21. set_Item with wrong arg count (1 arg) — must NOT rewrite.
    #[test]
    fn set_item_wrong_arg_count_one() {
        let mut stmts = vec![Stmt::Eval(set_item_call(var("v"), vec![Expr::Int(0)]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "set_Item with 1 arg must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "set_Item"));
    }

    /// 22. Verify IndexAssign rendering.
    #[test]
    fn index_assign_renders_correctly() {
        let stmts = vec![Stmt::IndexAssign {
            collection: var("v"),
            index: Expr::Int(0),
            value: var("val"),
        }];
        let rendered = super::super::super::csharp::render_stmts(&stmts);
        assert_eq!(rendered, "v[0] = val;\n");
    }
}
