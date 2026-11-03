//! Event handler reconstruction: `add_X(h)` / `remove_X(h)` to `+=` / `-=`.
//!
//! ## What this pass does
//!
//! IL2CPP renders C# event subscription as ordinary method calls:
//!
//! ```text
//! obj.add_OnClick(handler)     ->  obj.OnClick += handler
//! obj.remove_OnClick(handler)  ->  obj.OnClick -= handler
//! ```
//!
//! This appears hundreds of times in Unity codebases and is one of the most
//! visible reconstruction passes for readable C# output.
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - Name must start with `"add_"` or `"remove_"` and have a non-empty suffix
//! - Receiver must be `Some` (static calls are not event subscriptions)
//! - Argument count must be exactly 1 (the handler)
//! - The statement must be `Stmt::Eval` (expression evaluated for side effects)
//!
//! A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{Expr, Stmt};

/// Run the event handler reconstruction over one method body.
///
/// Returns the number of `add_X` / `remove_X` calls rewritten to
/// `EventAssign` statements.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut rewrites = 0;
    for stmt in stmts.iter_mut() {
        rewrites += rewrite_stmt(stmt);
    }
    rewrites
}

/// Rewrite a single statement, recursing into nested bodies.
fn rewrite_stmt(stmt: &mut Stmt) -> usize {
    // Check if this is an Eval(Call { add_X/remove_X }) pattern.
    if let Stmt::Eval(Expr::Call { recv: Some(_), name, args }) = stmt {
        if args.len() == 1 {
            if let Some(event_name) = name.strip_prefix("add_") {
                if !event_name.is_empty() {
                    let event_name = event_name.to_string();
                    // Destructure to take ownership of parts.
                    if let Stmt::Eval(Expr::Call { recv: Some(recv), args: mut call_args, .. }) =
                        std::mem::replace(stmt, Stmt::Comment(String::new()))
                    {
                        let recv_expr = *recv;
                        let handler = std::mem::replace(&mut call_args[0], Expr::Null);
                        *stmt = Stmt::EventAssign {
                            field: Expr::Field(Box::new(recv_expr), event_name),
                            value: handler,
                            add: true,
                        };
                        return 1;
                    }
                }
            } else if let Some(event_name) = name.strip_prefix("remove_") {
                if !event_name.is_empty() {
                    let event_name = event_name.to_string();
                    if let Stmt::Eval(Expr::Call { recv: Some(recv), args: mut call_args, .. }) =
                        std::mem::replace(stmt, Stmt::Comment(String::new()))
                    {
                        let recv_expr = *recv;
                        let handler = std::mem::replace(&mut call_args[0], Expr::Null);
                        *stmt = Stmt::EventAssign {
                            field: Expr::Field(Box::new(recv_expr), event_name),
                            value: handler,
                            add: false,
                        };
                        return 1;
                    }
                }
            }
        }
    }

    // Recurse into nested blocks so event handlers inside if/while/for/try
    // bodies are also recovered.
    match stmt {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            let mut r = 0;
            r += apply(then_body);
            r += apply(else_body);
            r
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::ForEach { body, .. } => {
            apply(body)
        }
        Stmt::For {
            init,
            incr,
            body,
            ..
        } => {
            let mut r = 0;
            for s in init {
                r += rewrite_stmt(s);
            }
            for s in incr {
                r += rewrite_stmt(s);
            }
            r += apply(body);
            r
        }
        Stmt::Using { decl, body } => {
            let mut r = rewrite_stmt(decl);
            r += apply(body);
            r
        }
        Stmt::Lock { body, .. } => apply(body),
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut r = apply(try_body);
            for (_, b) in catches {
                r += apply(b);
            }
            if let Some(fb) = finally_body {
                r += apply(fb);
            }
            r
        }
        Stmt::Switch { cases, tail, default, .. } => {
            let mut r = 0;
            for (_, b) in cases {
                r += apply(b);
            }
            r += apply(tail);
            if let Some(arm) = default {
                r += apply(arm);
            }
            r
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            let mut r = 0;
            for (_, b) in cases {
                r += apply(b);
            }
            if let Some(t) = tail {
                r += apply(t);
            }
            r
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build `recv.name(args...)`.
    fn call(recv: Expr, name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            recv: Some(Box::new(recv)),
            name: name.into(),
            args,
        }
    }

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    // -----------------------------------------------------------------------
    // Positive rewrites
    // -----------------------------------------------------------------------

    /// 1. Simple add handler: `obj.add_OnClick(handler)` -> `obj.OnClick += handler`
    #[test]
    fn simple_add_handler() {
        let mut stmts = vec![Stmt::Eval(call(
            var("obj"),
            "add_OnClick",
            vec![var("handler")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::EventAssign { field, value, add } => {
                assert!(matches!(field, Expr::Field(recv, name) if matches!(recv.as_ref(), Expr::Var(v) if v == "obj") && name == "OnClick"));
                assert!(matches!(value, Expr::Var(v) if v == "handler"));
                assert!(*add);
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    /// 2. Simple remove handler: `obj.remove_OnClick(handler)` -> `obj.OnClick -= handler`
    #[test]
    fn simple_remove_handler() {
        let mut stmts = vec![Stmt::Eval(call(
            var("obj"),
            "remove_OnClick",
            vec![var("handler")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::EventAssign { field, value, add } => {
                assert!(matches!(field, Expr::Field(recv, name) if matches!(recv.as_ref(), Expr::Var(v) if v == "obj") && name == "OnClick"));
                assert!(matches!(value, Expr::Var(v) if v == "handler"));
                assert!(!(*add));
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    /// 3. Add with field receiver: `this.button.add_Click(handler)` -> `this.button.Click += handler`
    #[test]
    fn add_with_field_receiver() {
        let recv = Expr::Field(Box::new(Expr::This), "button".into());
        let mut stmts = vec![Stmt::Eval(call(recv, "add_Click", vec![var("handler")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::EventAssign { field, value, add } => {
                // field should be Field(Field(This, "button"), "Click")
                assert!(matches!(field, Expr::Field(recv, name) if matches!(recv.as_ref(), Expr::Field(_, _)) && name == "Click"));
                assert!(matches!(value, Expr::Var(v) if v == "handler"));
                assert!(*add);
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    /// 4. Add with lambda handler: `obj.add_OnClick(() => foo())` -> `obj.OnClick += () => foo()`
    #[test]
    fn add_with_lambda_handler() {
        let lambda = Expr::Lambda {
            params: vec![],
            body: Box::new(Expr::Call {
                recv: None,
                name: "foo".into(),
                args: vec![],
            }),
        };
        let mut stmts = vec![Stmt::Eval(call(var("obj"), "add_OnClick", vec![lambda]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::EventAssign { field, value, add } => {
                assert!(matches!(field, Expr::Field(_, name) if name == "OnClick"));
                assert!(matches!(value, Expr::Lambda { .. }));
                assert!(*add);
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    /// 5. Add with method reference: `obj.add_OnClick(this.HandleClick)` -> `obj.OnClick += this.HandleClick`
    #[test]
    fn add_with_method_reference() {
        let handler = Expr::Field(Box::new(Expr::This), "HandleClick".into());
        let mut stmts = vec![Stmt::Eval(call(var("obj"), "add_OnClick", vec![handler]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::EventAssign { field, value, add } => {
                assert!(matches!(field, Expr::Field(recv, name) if matches!(recv.as_ref(), Expr::Var(v) if v == "obj") && name == "OnClick"));
                assert!(matches!(value, Expr::Field(recv, name) if matches!(recv.as_ref(), Expr::This) && name == "HandleClick"));
                assert!(*add);
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    /// 6. Remove with complex receiver: `GetButton().remove_Click(handler)` -> `GetButton().Click -= handler`
    #[test]
    fn remove_with_complex_receiver() {
        let recv = Expr::Call {
            recv: None,
            name: "GetButton".into(),
            args: vec![],
        };
        let mut stmts = vec![Stmt::Eval(call(recv, "remove_Click", vec![var("handler")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::EventAssign { field, value, add } => {
                assert!(matches!(field, Expr::Field(recv, name) if matches!(recv.as_ref(), Expr::Call { .. }) && name == "Click"));
                assert!(matches!(value, Expr::Var(v) if v == "handler"));
                assert!(!(*add));
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    /// 7. Non-match: add_X with 0 arguments.
    #[test]
    fn add_with_zero_args_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(call(var("obj"), "add_OnClick", vec![]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "add_OnClick with 0 args must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "add_OnClick"));
    }

    /// 8. Non-match: add_X with 2 arguments.
    #[test]
    fn add_with_two_args_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(call(
            var("obj"),
            "add_OnClick",
            vec![var("h1"), var("h2")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "add_OnClick with 2 args must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "add_OnClick"));
    }

    /// 9. Non-match: static add_X (no receiver).
    #[test]
    fn static_add_not_rewritten() {
        let expr = Expr::Call {
            recv: None,
            name: "add_OnClick".into(),
            args: vec![var("handler")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "static add_X must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { recv: None, .. })));
    }

    /// 10. Non-match: addX (no underscore — not an event accessor pattern).
    #[test]
    fn no_underscore_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(call(var("obj"), "addOnClick", vec![var("handler")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "addOnClick without underscore must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "addOnClick"));
    }

    /// 11. Multiple event handlers in sequence.
    #[test]
    fn multiple_event_handlers() {
        let mut stmts = vec![
            Stmt::Eval(call(var("obj"), "add_OnClick", vec![var("h1")])),
            Stmt::Eval(call(var("obj"), "remove_OnClick", vec![var("h2")])),
            Stmt::Eval(call(var("obj"), "add_OnDestroy", vec![var("h3")])),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 3, "all three event handlers should be rewritten");
        // Check first: add
        match &stmts[0] {
            Stmt::EventAssign { field, add, .. } => {
                assert!(matches!(field, Expr::Field(_, name) if name == "OnClick"));
                assert!(*add);
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
        // Check second: remove
        match &stmts[1] {
            Stmt::EventAssign { field, add, .. } => {
                assert!(matches!(field, Expr::Field(_, name) if name == "OnClick"));
                assert!(!(*add));
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
        // Check third: add different event
        match &stmts[2] {
            Stmt::EventAssign { field, add, .. } => {
                assert!(matches!(field, Expr::Field(_, name) if name == "OnDestroy"));
                assert!(*add);
            }
            other => panic!("expected EventAssign, got {other:?}"),
        }
    }

    /// 12. Event handler in if body.
    #[test]
    fn event_handler_in_if_body() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(call(
                var("obj"),
                "add_OnClick",
                vec![var("handler")],
            ))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => match &then_body[0] {
                Stmt::EventAssign { field, value, add } => {
                    assert!(matches!(field, Expr::Field(_, name) if name == "OnClick"));
                    assert!(matches!(value, Expr::Var(v) if v == "handler"));
                    assert!(*add);
                }
                other => panic!("expected EventAssign in then_body, got {other:?}"),
            },
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// Non-match: bare "add_" with empty suffix.
    #[test]
    fn bare_add_prefix_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(call(var("obj"), "add_", vec![var("handler")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "bare add_ with empty suffix must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "add_"));
    }

    /// Non-match: bare "remove_" with empty suffix.
    #[test]
    fn bare_remove_prefix_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(call(var("obj"), "remove_", vec![var("handler")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "bare remove_ with empty suffix must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "remove_"));
    }

    /// Event handler inside a while loop body.
    #[test]
    fn event_handler_in_while_body() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Eval(call(
                var("obj"),
                "add_OnUpdate",
                vec![var("handler")],
            ))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => match &body[0] {
                Stmt::EventAssign { field, add, .. } => {
                    assert!(matches!(field, Expr::Field(_, name) if name == "OnUpdate"));
                    assert!(*add);
                }
                other => panic!("expected EventAssign in while body, got {other:?}"),
            },
            other => panic!("expected While, got {other:?}"),
        }
    }

    /// Non-event call is left untouched.
    #[test]
    fn regular_call_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(call(var("obj"), "DoSomething", vec![var("arg")]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "DoSomething"));
    }
}
