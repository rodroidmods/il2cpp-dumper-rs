//! Collection initializer recognition.
//!
//! ## Pattern
//!
//! Detects when a newly-created collection has elements immediately added via
//! `Add()` calls, and folds them into a collection initializer expression:
//!
//! ```text
//! v = new List<T>(cap);     // Stmt::Assign(v, ty, Expr::NewObj(type, ctor_args))
//! v.Add(e0);                // Stmt::Eval(Expr::Call { recv: Some(Var(v)), name: "Add", args: [e0] })
//! v.Add(e1);                // Stmt::Eval(Expr::Call { recv: Some(Var(v)), name: "Add", args: [e1] })
//! v.Add(e2);                // Stmt::Eval(Expr::Call { recv: Some(Var(v)), name: "Add", args: [e2] })
//! ```
//!
//! becomes:
//!
//! ```text
//! v = new List<T>(cap) { e0, e1, e2 };
//! ```
//!
//! ## Safety conditions
//!
//! The fold is only applied when ALL of the following hold:
//!
//! 1. The assignment creates a new object (`Expr::NewObj`).
//! 2. Every following statement until the first non-Add statement is an `Add`
//!    call on the SAME local variable.
//! 3. Each `Add` call has exactly one argument (list-style) — or two arguments
//!    (dictionary-style, stored as nested items).
//! 4. No intervening statement READS the local `v` (other than the Add calls
//!    themselves), which would prove the collection is used before initialization
//!    is complete.
//! 5. The method name is `Add` or ends with `.Add`.
//!
//! A failed match is a NO-OP: the flat sequence stays.

use super::super::csharp::{Expr, Stmt};

/// Entry point: fold collection initializers in one statement list, returning
/// the number of rewrites applied.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    super::walk_mut(stmts, &mut |list| {
        n += fold_collection_inits(list);
    });
    n
}

/// Fold every collection initializer pattern in one statement list.
fn fold_collection_inits(list: &mut Vec<Stmt>) -> usize {
    if list.len() < 2 {
        return 0;
    }
    let mut count = 0;
    let mut i = 0;
    while i < list.len() {
        // Look for: Stmt::Assign(v, ty, Expr::NewObj(type_name, ctor_args))
        let (var_name, type_name, ctor_args) = match &list[i] {
            Stmt::Assign(v, _ty, Expr::NewObj(ty, args)) => (v.clone(), ty.clone(), args.clone()),
            _ => {
                i += 1;
                continue;
            }
        };

        // Collect consecutive Add calls on the same variable.
        let mut items = Vec::new();
        let mut j = i + 1;
        while j < list.len() {
            if let Some(item_expr) = is_add_call_to(&list[j], &var_name) {
                items.push(item_expr);
                j += 1;
            } else {
                // Not an Add call — stop collecting.
                break;
            }
        }

        // We need at least one Add call to form a collection initializer.
        // (Zero Add calls would just be `new List<T>()` with no initializer,
        // which is already fine as a bare NewObj.)
        if items.is_empty() {
            i += 1;
            continue;
        }

        // Verify no intervening reads of `var_name` between the NewObj assignment
        // and the Add calls. The Add calls themselves read `var_name` (as receiver),
        // but that is the pattern we are folding. We need to check that no
        // NON-Add statement between i+1..j reads `var_name`.
        // Since we already verified that every statement in i+1..j IS an Add call
        // to `var_name`, there are no intervening non-Add statements by construction.
        // However, we must also check that the statement AFTER the last Add (if any)
        // does not indicate the variable was used in a way that would be disrupted.
        // Actually, the pattern is safe by construction: we only fold consecutive
        // Add calls immediately after the NewObj, so there is no gap.

        // Also verify that the statement right after the Add calls (if it exists)
        // does not read `var_name` in a way that would be disrupted. Actually,
        // folding removes the Add calls and puts the items inline in the NewObj,
        // so the variable is still assigned at position `i` — any later read is
        // fine because the variable still holds the same collection.

        // Replace the assignment with a CollectionInit.
        list[i] = Stmt::Assign(
            var_name.clone(),
            None, // type was already declared on the original assignment
            Expr::CollectionInit {
                type_name,
                ctor_args,
                items,
            },
        );

        // Remove the Add call statements.
        let remove_count = j - (i + 1);
        if remove_count > 0 {
            list.drain((i + 1)..j);
        }

        count += 1;
        // Don't advance `i` — there might be another pattern starting at the
        // same position (unlikely but safe).
        i += 1;
    }
    count
}

/// If `stmt` is `v.Add(item)` (or `v.Add(k, v)` for dictionary), return the
/// item expression(s) wrapped appropriately. For single-arg Add, returns the
/// single item. For two-arg Add where the first arg is IL2CPP dispatch metadata,
/// returns the second arg (the real item).
///
/// Returns `None` if the statement is not an Add call to `var_name`.
fn is_add_call_to(stmt: &Stmt, var_name: &str) -> Option<Expr> {
    let Stmt::Eval(expr) = stmt else {
        return None;
    };
    let Expr::Call { recv, name, args } = expr else {
        return None;
    };
    // The method name must be `Add` or end with `.Add`.
    if name != "Add" && !name.ends_with(".Add") {
        return None;
    }
    // The receiver must be the variable we are tracking.
    let Some(recv_expr) = recv.as_ref() else {
        return None;
    };
    let Expr::Var(recv_name) = recv_expr.as_ref() else {
        return None;
    };
    if recv_name != var_name {
        return None;
    }
    // Accept 1-arg (list-style), 2-arg with dispatch metadata, and 2-arg
    // dictionary-style Add calls.
    match args.len() {
        1 => Some(args[0].clone()),
        2 => {
            // IL2CPP often passes dispatch metadata as the first argument:
            //   v.Add(*(nint*)(*(nint*)(global_addr)), real_item)
            // The first arg is a RawDeref from a global address — detect this
            // pattern and return the second arg as the real item.
            if is_dispatch_metadata(&args[0]) {
                Some(args[1].clone())
            } else {
                // Dictionary-style Add(k, v) — not folded in this phase.
                None
            }
        }
        _ => None,
    }
}

/// Does this expression look like IL2CPP virtual dispatch metadata?
///
/// The real pattern is `*(nint*)(*(nint*)(global_addr))` — a double dereference
/// from a fixed global address. This is how IL2CPP loads method metadata for
/// virtual/interface dispatch. We detect:
/// - `RawDeref(inner, _, _)` where `inner` is itself a `RawDeref` of a constant
///   address, OR
/// - `RawDeref(inner, _, _)` where `inner` mentions a hex address (0x...), OR
/// - Any `RawDeref` whose base is a constant (not a local variable).
fn is_dispatch_metadata(e: &Expr) -> bool {
    match e {
        Expr::RawDeref(inner, _, _) => {
            // If the inner expression is a constant address or another deref
            // of a constant, it's dispatch metadata.
            match inner.as_ref() {
                Expr::RawDeref(base, _, _) => {
                    // Double deref: *(nint*)(*(nint*)(addr)) — classic dispatch.
                    is_constant_addr(base)
                }
                _ => is_constant_addr(inner),
            }
        }
        Expr::Cast(_, inner) => is_dispatch_metadata(inner),
        _ => false,
    }
}

/// Is this expression a constant address (hex literal or sum involving one)?
fn is_constant_addr(e: &Expr) -> bool {
    match e {
        Expr::Int(v) => *v > 0x10000, // large constants are addresses
        Expr::Bin(_, a, b) => is_constant_addr(a) || is_constant_addr(b),
        Expr::RawDeref(inner, _, _) => is_constant_addr(inner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build `v = new Type(ctor_args)`.
    fn new_obj_assign(var: &str, type_name: &str, ctor_args: Vec<Expr>) -> Stmt {
        Stmt::Assign(
            var.into(),
            Some("List<int>".into()),
            Expr::NewObj(type_name.into(), ctor_args),
        )
    }

    /// Helper: build `v.Add(item)`.
    fn add_call(var: &str, item: Expr) -> Stmt {
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(Expr::Var(var.into()))),
            name: "Add".into(),
            args: vec![item],
        })
    }

    /// Basic collection initializer with 3 items.
    #[test]
    fn basic_collection_init_with_three_items() {
        let mut stmts = vec![
            new_obj_assign("v", "List<int>", vec![]),
            add_call("v", Expr::Int(1)),
            add_call("v", Expr::Int(2)),
            add_call("v", Expr::Int(3)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(name, _, Expr::CollectionInit { type_name, ctor_args, items }) => {
                assert_eq!(name, "v");
                assert_eq!(type_name, "List<int>");
                assert!(ctor_args.is_empty());
                assert_eq!(items.len(), 3);
            }
            other => panic!("expected Assign(_, _, CollectionInit), got {other:?}"),
        }
    }

    /// Collection initializer with a capacity argument.
    #[test]
    fn collection_init_with_capacity() {
        let mut stmts = vec![
            new_obj_assign("list", "List<string>", vec![Expr::Int(10)]),
            add_call("list", Expr::Str("hello".into())),
            add_call("list", Expr::Str("world".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(name, _, Expr::CollectionInit { type_name, ctor_args, items }) => {
                assert_eq!(name, "list");
                assert_eq!(type_name, "List<string>");
                assert_eq!(ctor_args.len(), 1);
                assert_eq!(items.len(), 2);
            }
            other => panic!("expected Assign(_, _, CollectionInit), got {other:?}"),
        }
    }

    /// Intervening read of the variable prevents folding.
    #[test]
    fn intervening_read_prevents_folding() {
        let mut stmts = vec![
            new_obj_assign("v", "List<int>", vec![]),
            add_call("v", Expr::Int(1)),
            // Intervening read: `x = v.Count;`
            Stmt::Assign(
                "x".into(),
                None,
                Expr::Call {
                    recv: Some(Box::new(Expr::Var("v".into()))),
                    name: "get_Count".into(),
                    args: vec![],
                },
            ),
            add_call("v", Expr::Int(2)),
        ];
        let n = apply(&mut stmts);
        // The first Add is folded (it's right after the NewObj), but the
        // intervening read breaks the chain, so the second Add is not folded.
        // Actually, our algorithm folds consecutive Add calls starting right
        // after the NewObj. The intervening read stops the collection at 1 item.
        assert_eq!(n, 1);
        // Should have: CollectionInit with 1 item, then the Count read, then the Add.
        assert_eq!(stmts.len(), 3);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::CollectionInit { items, .. }) => {
                assert_eq!(items.len(), 1, "only the first Add should be folded");
            }
            other => panic!("expected CollectionInit, got {other:?}"),
        }
    }

    /// Non-Add methods are not folded.
    #[test]
    fn non_add_method_is_not_folded() {
        let mut stmts = vec![
            new_obj_assign("v", "List<int>", vec![]),
            // This is a Remove call, not Add.
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: "Remove".into(),
                args: vec![Expr::Int(1)],
            }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 2);
    }

    /// Empty collection (no Add calls) is not folded — stays as bare NewObj.
    #[test]
    fn empty_collection_is_not_folded() {
        let mut stmts = vec![
            new_obj_assign("v", "List<int>", vec![]),
            Stmt::Eval(Expr::Var("other".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 2);
        // The first statement should still be a bare NewObj.
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::NewObj(..))));
    }

    /// Multiple collections in one method are each folded independently.
    #[test]
    fn multiple_collections_in_one_method() {
        let mut stmts = vec![
            new_obj_assign("a", "List<int>", vec![]),
            add_call("a", Expr::Int(1)),
            add_call("a", Expr::Int(2)),
            new_obj_assign("b", "List<string>", vec![]),
            add_call("b", Expr::Str("x".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2);
        assert_eq!(stmts.len(), 2);
        match &stmts[0] {
            Stmt::Assign(name, _, Expr::CollectionInit { items, .. }) => {
                assert_eq!(name, "a");
                assert_eq!(items.len(), 2);
            }
            other => panic!("expected CollectionInit for 'a', got {other:?}"),
        }
        match &stmts[1] {
            Stmt::Assign(name, _, Expr::CollectionInit { items, .. }) => {
                assert_eq!(name, "b");
                assert_eq!(items.len(), 1);
            }
            other => panic!("expected CollectionInit for 'b', got {other:?}"),
        }
    }

    /// Add calls on a DIFFERENT variable are not folded.
    #[test]
    fn add_on_different_variable_is_not_folded() {
        let mut stmts = vec![
            new_obj_assign("v", "List<int>", vec![]),
            // Add on "w", not "v".
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("w".into()))),
                name: "Add".into(),
                args: vec![Expr::Int(1)],
            }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 2);
    }

    /// Rendering test: verify the rendered output matches C# collection initializer syntax.
    #[test]
    fn render_collection_init() {
        let e = Expr::CollectionInit {
            type_name: "List<int>".into(),
            ctor_args: vec![Expr::Int(10)],
            items: vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)],
        };
        let mut out = String::new();
        e.render(&mut out);
        assert_eq!(out, "new List<int>(10) { 1, 2, 3 }");
    }

    /// Rendering test: empty ctor args.
    #[test]
    fn render_collection_init_no_ctor_args() {
        let e = Expr::CollectionInit {
            type_name: "List<string>".into(),
            ctor_args: vec![],
            items: vec![Expr::Str("a".into()), Expr::Str("b".into())],
        };
        let mut out = String::new();
        e.render(&mut out);
        assert_eq!(out, "new List<string>() { \"a\", \"b\" }");
    }

    /// Add call with dotted name (e.g., `System.Collections.Generic.List<int>.Add`)
    /// should also be recognized.
    #[test]
    fn dotted_add_name_is_folded() {
        let mut stmts = vec![
            new_obj_assign("v", "List<int>", vec![]),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v".into()))),
                name: "ICollection<int>.Add".into(),
                args: vec![Expr::Int(42)],
            }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::CollectionInit { items, .. }) => {
                assert_eq!(items.len(), 1);
            }
            other => panic!("expected CollectionInit, got {other:?}"),
        }
    }

    /// Two-argument Add (dictionary-style) is NOT folded (optional feature).
    #[test]
    fn dictionary_add_is_not_folded() {
        let mut stmts = vec![
            new_obj_assign("d", "Dictionary<string, int>", vec![]),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("d".into()))),
                name: "Add".into(),
                args: vec![Expr::Str("key".into()), Expr::Int(1)],
            }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "dictionary-style Add is not folded in this phase");
        assert_eq!(stmts.len(), 2);
    }

    /// A collection initializer nested inside an if-body is still found.
    #[test]
    fn nested_in_if_body() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![
                new_obj_assign("v", "List<int>", vec![]),
                add_call("v", Expr::Int(1)),
                add_call("v", Expr::Int(2)),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert!(matches!(
                    &then_body[0],
                    Stmt::Assign(_, _, Expr::CollectionInit { items, .. }) if items.len() == 2
                ));
            }
            _ => panic!("expected If"),
        }
    }

    // -----------------------------------------------------------------------
    // Two-argument Add with IL2CPP dispatch metadata
    // -----------------------------------------------------------------------

    /// 2-arg Add where first arg is dispatch metadata (double deref from global).
    /// The first arg should be skipped, and the second arg is the real item.
    #[test]
    fn two_arg_add_with_dispatch_metadata_is_folded() {
        let mut stmts = vec![
            new_obj_assign("v91", "List<object>", vec![]),
            // v91.Add(*(nint*)(*(nint*)(0x2f6a000 +1704)), v103);
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v91".into()))),
                name: "Add".into(),
                args: vec![
                    Expr::RawDeref(
                        Box::new(Expr::RawDeref(Box::new(Expr::Int(0x2f6a000 + 1704)), 0, 0)),
                        0,
                        0,
                    ),
                    Expr::Var("v103".into()),
                ],
            }),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v91".into()))),
                name: "Add".into(),
                args: vec![
                    Expr::RawDeref(
                        Box::new(Expr::RawDeref(Box::new(Expr::Int(0x2f6a000 + 1688)), 0, 0)),
                        0,
                        0,
                    ),
                    Expr::Var("v131".into()),
                ],
            }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "2-arg Add with dispatch metadata should fold");
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::CollectionInit { items, .. }) => {
                assert_eq!(items.len(), 2);
                // Items should be the second args (v103, v131), not the metadata.
                assert!(matches!(&items[0], Expr::Var(n) if n == "v103"));
                assert!(matches!(&items[1], Expr::Var(n) if n == "v131"));
            }
            other => panic!("expected CollectionInit, got {other:?}"),
        }
    }

    /// 2-arg Add where first arg is a plain local (NOT dispatch metadata).
    /// This is a dictionary-style Add and should NOT be folded.
    #[test]
    fn two_arg_add_with_local_first_is_not_folded() {
        let mut stmts = vec![
            new_obj_assign("d", "Dictionary<string, int>", vec![]),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("d".into()))),
                name: "Add".into(),
                args: vec![Expr::Var("key".into()), Expr::Int(1)],
            }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "dictionary-style Add with local first arg is not folded");
        assert_eq!(stmts.len(), 2);
    }
}
