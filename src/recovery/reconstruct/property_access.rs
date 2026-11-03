//! Property access reconstruction: `get_X()` → `X`, `set_X(v)` → `X = v`.
//!
//! ## What this pass does
//!
//! IL2CPP renders property access as ordinary method calls:
//!
//! ```text
//! obj.get_Name()           →  obj.Name
//! obj.set_Name(value)      →  obj.Name = value
//! Type.set_Property(o, v)  →  o.Property = v   (static-form setter)
//! ```
//!
//! Getters are expression-level rewrites (`Call` → `Field`).
//! Setters are statement-level rewrites (`Eval(Call)` → `Assign`).
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - Name must start with `"get_"` / `"set_"` and have a non-empty suffix
//! - Getter: receiver must be `Some`, argument count must be exactly 0
//! - Setter instance form: receiver must be `Some`, exactly 1 argument
//! - Setter static form: receiver must be `None`, exactly 2 arguments
//!   (first arg is the target object, second is the value)
//! - `"get_Item"` / `"set_Item"` are excluded (handled by the indexer pass)
//!
//! A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{Expr, Stmt};

/// Run the property access reconstruction over one method body.
///
/// Returns the number of `get_X` calls rewritten to `Field` expressions.
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
            // Try setter rewrite FIRST (statement-level: Eval(Call) → Assign).
            // Check if this looks like a setter before borrowing expr.
            let is_setter = matches!(&*expr, Expr::Call { name, .. } if name.starts_with("set_"));
            if is_setter {
                rewrites += try_rewrite_setter(stmt);
            }
            if rewrites == 0 {
                // Not a setter — rewrite getter sub-expressions normally.
                if let Stmt::Eval(expr) = stmt {
                    rewrites += rewrite_expr(expr);
                }
            }
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
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            rewrites += rewrite_expr(collection);
            rewrites += rewrite_expr(index);
            rewrites += rewrite_expr(value);
        }
    }
    rewrites
}

/// Try to rewrite a setter call in statement context.
///
/// Handles two forms:
/// 1. Instance setter: `obj.set_Property(value)` → `obj.Property = value`
/// 2. Static setter: `Type.set_Property(obj, value)` → `obj.Property = value`
///
/// Returns 1 if rewritten, 0 otherwise.
fn try_rewrite_setter(stmt: &mut Stmt) -> usize {
    if let Stmt::Eval(expr) = stmt {
        if let Expr::Call { recv, name, args } = expr {
            // Safety: only rewrite when:
            // 1. Name starts with "set_"
            // 2. Has a non-empty property name after the prefix
            // 3. Not "set_Item" (handled by the indexer pass)
            if name == "set_Item" {
                return 0;
            }
            if let Some(prop_name) = name.strip_prefix("set_") {
                if prop_name.is_empty() {
                    return 0;
                }

                // Instance form: recv.set_Property(value)
                if let Some(recv_expr) = recv {
                    if args.len() == 1 {
                        let mut target = String::new();
                        // Add parentheses if receiver has lower precedence than field access
                        let needs_parens = matches!(recv_expr.as_ref(), Expr::Cast(_, _) | Expr::Bin(_, _, _) | Expr::Cmp(_, _, _));
                        if needs_parens {
                            target.push('(');
                        }
                        recv_expr.render(&mut target);
                        if needs_parens {
                            target.push(')');
                        }
                        target.push('.');
                        target.push_str(prop_name);
                        let value = args.remove(0);
                        *stmt = Stmt::Assign(target, None, value);
                        return 1;
                    }
                }

                // Static form: Type.set_Property(obj, value)
                if recv.is_none() && args.len() == 2 {
                    let mut target = String::new();
                    args[0].render(&mut target);
                    target.push('.');
                    target.push_str(prop_name);
                    let _obj = args.remove(0); // Already rendered into target
                    let value = args.remove(0);
                    *stmt = Stmt::Assign(target, None, value);
                    return 1;
                }
            }
        }
    }
    0
}

/// Recursively rewrite `get_X()` calls to `Field` expressions within an
/// expression tree.
///
/// Post-order traversal: children are rewritten first, so nested property
/// accesses (`a.get_X().get_Y()`) are handled naturally — the inner call is
/// rewritten to `a.X`, then the outer call sees `a.X.get_Y()` and becomes
/// `a.X.Y`.
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

    // Now check if THIS expression is a get_X call we can rewrite.
    if let Expr::Call {
        recv: Some(recv),
        name,
        args,
    } = expr
    {
        // Safety: only rewrite when:
        // 1. Name starts with "get_"
        // 2. Has a non-empty property name after the prefix
        // 3. Exactly 0 arguments (parameterless getter)
        // 4. Not "get_Item" (handled by the indexer pass)
        if args.is_empty()
            && name != "get_Item"
            && let Some(prop_name) = name.strip_prefix("get_")
            && !prop_name.is_empty()
        {
            let prop_name = prop_name.to_string();
            let recv = std::mem::replace(recv.as_mut(), Expr::Null);
            *expr = Expr::Field(Box::new(recv), prop_name);
            rewrites += 1;
        }
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build `recv.get_X(args...)`.
    fn getter_call(recv: Expr, name: &str, args: Vec<Expr>) -> Expr {
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

    /// 1. Simple getter: `v.get_Name()` → `v.Name`
    #[test]
    fn simple_getter() {
        let mut stmts = vec![Stmt::Eval(getter_call(var("v"), "get_Name", vec![]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, name)) => {
                assert!(matches!(recv.as_ref(), Expr::Var(v) if v == "v"));
                assert_eq!(name, "Name");
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    /// 2. Getter with field receiver: `this.obj.get_Value()` → `this.obj.Value`
    #[test]
    fn getter_with_field_receiver() {
        let recv = Expr::Field(Box::new(Expr::This), "obj".into());
        let mut stmts = vec![Stmt::Eval(getter_call(recv, "get_Value", vec![]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, name)) => {
                assert!(matches!(recv.as_ref(), Expr::Field(_, _)));
                assert_eq!(name, "Value");
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    /// 3. Getter in assignment: `x = obj.get_Count()` → `x = obj.Count`
    #[test]
    fn getter_in_assignment() {
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            getter_call(var("obj"), "get_Count", vec![]),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::Field(recv, name)) => {
                assert_eq!(target, "x");
                assert!(matches!(recv.as_ref(), Expr::Var(v) if v == "obj"));
                assert_eq!(name, "Count");
            }
            other => panic!("expected Assign(_, _, Field), got {other:?}"),
        }
    }

    /// 4. Getter in condition: `if (obj.get_IsValid())` → `if (obj.IsValid)`
    #[test]
    fn getter_in_condition() {
        let mut stmts = vec![Stmt::If {
            cond: getter_call(var("obj"), "get_IsValid", vec![]),
            then_body: vec![Stmt::Eval(var("body"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { cond, .. } => match cond {
                Expr::Field(recv, name) => {
                    assert!(matches!(recv.as_ref(), Expr::Var(v) if v == "obj"));
                    assert_eq!(name, "IsValid");
                }
                other => panic!("expected Field in condition, got {other:?}"),
            },
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// 5. Getter in method call: `foo(obj.get_Data())` → `foo(obj.Data)`
    #[test]
    fn getter_in_method_call() {
        let inner = getter_call(var("obj"), "get_Data", vec![]);
        let outer = Expr::Call {
            recv: Some(Box::new(var("foo"))),
            name: "Bar".into(),
            args: vec![inner],
        };
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Call { args, .. }) => {
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Field(_, name) if name == "Data"));
            }
            other => panic!("expected Eval(Call), got {other:?}"),
        }
    }

    /// 6. Nested getters: `a.get_X().get_Y()` → `a.X.Y`
    #[test]
    fn nested_getters() {
        let inner = getter_call(var("a"), "get_X", vec![]);
        let outer = getter_call(inner, "get_Y", vec![]);
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both inner and outer getters should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, name)) => {
                assert_eq!(name, "Y");
                match recv.as_ref() {
                    Expr::Field(inner_recv, inner_name) => {
                        assert_eq!(inner_name, "X");
                        assert!(matches!(inner_recv.as_ref(), Expr::Var(v) if v == "a"));
                    }
                    other => panic!("expected nested Field, got {other:?}"),
                }
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    /// 7. Getter with complex receiver: `((Type)obj).get_Field()` → `((Type)obj).Field`
    #[test]
    fn getter_with_cast_receiver() {
        let recv = Expr::Cast("Type".into(), Box::new(var("obj")));
        let mut stmts = vec![Stmt::Eval(getter_call(recv, "get_Field", vec![]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::Field(recv, name)) => {
                assert!(matches!(recv.as_ref(), Expr::Cast(_, _)));
                assert_eq!(name, "Field");
            }
            other => panic!("expected Eval(Field), got {other:?}"),
        }
    }

    /// 8. Getter in return: `return obj.get_Result()` → `return obj.Result`
    #[test]
    fn getter_in_return() {
        let mut stmts = vec![Stmt::Return(Some(getter_call(
            var("obj"),
            "get_Result",
            vec![],
        )))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Return(Some(Expr::Field(recv, name))) => {
                assert!(matches!(recv.as_ref(), Expr::Var(v) if v == "obj"));
                assert_eq!(name, "Result");
            }
            other => panic!("expected Return(Field), got {other:?}"),
        }
    }

    /// 9. Multiple getters in one statement: `obj.get_A() + obj.get_B()`
    #[test]
    fn multiple_getters_in_one_statement() {
        let lhs = getter_call(var("obj"), "get_A", vec![]);
        let rhs = getter_call(var("obj"), "get_B", vec![]);
        let expr = Expr::Bin(
            super::super::super::csharp::BinOp::Add,
            Box::new(lhs),
            Box::new(rhs),
        );
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both getters should be rewritten");
        match &stmts[0] {
            Stmt::Eval(Expr::Bin(_, a, b)) => {
                assert!(matches!(a.as_ref(), Expr::Field(_, name) if name == "A"));
                assert!(matches!(b.as_ref(), Expr::Field(_, name) if name == "B"));
            }
            other => panic!("expected Eval(Bin), got {other:?}"),
        }
    }

    /// 10. Getter inside a while loop body.
    #[test]
    fn getter_inside_while_body() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Eval(getter_call(var("v"), "get_Count", vec![]))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[0], Stmt::Eval(Expr::Field(_, name)) if name == "Count"));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    /// 11. Non-match: get_Item (handled by indexer pass).
    #[test]
    fn get_item_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(getter_call(
            var("v"),
            "get_Item",
            vec![Expr::Int(0)],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "get_Item must not be rewritten by this pass");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "get_Item"));
    }

    /// 12. Non-match: get_X with arguments (not a parameterless getter).
    #[test]
    fn getter_with_args_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(getter_call(
            var("v"),
            "get_Value",
            vec![Expr::Int(0)],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "get_Value with args must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "get_Value"));
    }

    /// 13. Non-match: static get_X (no receiver).
    #[test]
    fn static_getter_not_rewritten() {
        let expr = Expr::Call {
            recv: None,
            name: "get_Static".into(),
            args: vec![],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "static get_X must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { recv: None, .. })));
    }

    /// 14. Non-match: getX (no underscore — not a property getter pattern).
    #[test]
    fn no_underscore_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(getter_call(var("v"), "getX", vec![]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "getX without underscore must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "getX"));
    }

    /// 15. Non-match: bare "get_" with empty suffix.
    #[test]
    fn bare_get_prefix_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(getter_call(var("v"), "get_", vec![]))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "bare get_ with empty suffix must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "get_"));
    }

    // -----------------------------------------------------------------------
    // Rendering tests
    // -----------------------------------------------------------------------

    /// Verify that the Field expression from a getter renders correctly.
    #[test]
    fn property_renders_correctly() {
        let expr = Expr::Field(Box::new(var("obj")), "Name".into());
        let mut s = String::new();
        expr.render(&mut s);
        assert_eq!(s, "obj.Name");
    }

    /// Verify nested property rendering.
    #[test]
    fn nested_property_renders_correctly() {
        let inner = Expr::Field(Box::new(var("a")), "X".into());
        let outer = Expr::Field(Box::new(inner), "Y".into());
        let mut s = String::new();
        outer.render(&mut s);
        assert_eq!(s, "a.X.Y");
    }

    // -----------------------------------------------------------------------
    // Setter tests
    // -----------------------------------------------------------------------

    /// Helper: build `recv.set_X(args...)`.
    fn setter_call(recv: Option<Expr>, name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            recv: recv.map(Box::new),
            name: name.into(),
            args,
        }
    }

    /// 16. Instance setter: `obj.set_Name("foo")` → `obj.Name = "foo"`
    #[test]
    fn instance_setter() {
        let mut stmts = vec![Stmt::Eval(setter_call(
            Some(var("obj")),
            "set_Name",
            vec![Expr::Str("foo".into())],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::Str(s)) => {
                assert_eq!(target, "obj.Name");
                assert_eq!(s, "foo");
            }
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    /// 17. Static setter: `Type.set_Property(obj, value)` → `obj.Property = value`
    #[test]
    fn static_setter() {
        let mut stmts = vec![Stmt::Eval(setter_call(
            None,
            "set_Property",
            vec![var("obj"), var("value")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::Var(v)) => {
                assert_eq!(target, "obj.Property");
                assert_eq!(v, "value");
            }
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    /// 18. Setter with complex value expression.
    #[test]
    fn setter_with_complex_value() {
        let complex_value = Expr::Bin(
            super::super::super::csharp::BinOp::Add,
            Box::new(Expr::Int(1)),
            Box::new(Expr::Int(2)),
        );
        let mut stmts = vec![Stmt::Eval(setter_call(
            Some(var("obj")),
            "set_Count",
            vec![complex_value],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::Bin(_, _, _)) => {
                assert_eq!(target, "obj.Count");
            }
            other => panic!("expected Assign with Bin, got {other:?}"),
        }
    }

    /// 19. Setter with field receiver: `this.obj.set_Value(42)` → `this.obj.Value = 42`
    #[test]
    fn setter_with_field_receiver() {
        let recv = Expr::Field(Box::new(Expr::This), "obj".into());
        let mut stmts = vec![Stmt::Eval(setter_call(
            Some(recv),
            "set_Value",
            vec![Expr::Int(42)],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::Int(v)) => {
                assert_eq!(target, "this.obj.Value");
                assert_eq!(*v, 42);
            }
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    /// 20. Setter for auto-property vs computed property (both rewrite the same way).
    #[test]
    fn setter_auto_vs_computed() {
        // Auto-property setter
        let mut stmts1 = vec![Stmt::Eval(setter_call(
            Some(var("obj")),
            "set_Name",
            vec![Expr::Str("test".into())],
        ))];
        let n1 = apply(&mut stmts1);
        assert_eq!(n1, 1);
        assert!(matches!(&stmts1[0], Stmt::Assign(t, _, _) if t == "obj.Name"));

        // Computed property setter (same rewrite)
        let mut stmts2 = vec![Stmt::Eval(setter_call(
            Some(var("obj")),
            "set_FullName",
            vec![Expr::Str("John Doe".into())],
        ))];
        let n2 = apply(&mut stmts2);
        assert_eq!(n2, 1);
        assert!(matches!(&stmts2[0], Stmt::Assign(t, _, _) if t == "obj.FullName"));
    }

    /// 21. Setter with namespaced type: `System.String.set_Length(10)` → `System.String.Length = 10`
    #[test]
    fn setter_with_namespaced_type() {
        let recv = Expr::Var("System.String".into());
        let mut stmts = vec![Stmt::Eval(setter_call(
            Some(recv),
            "set_Length",
            vec![Expr::Int(10)],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::Int(v)) => {
                assert_eq!(target, "System.String.Length");
                assert_eq!(*v, 10);
            }
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    /// 22. Setter in if-body context.
    #[test]
    fn setter_in_if_body() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(setter_call(
                Some(var("obj")),
                "set_Flag",
                vec![Expr::Bool(true)],
            ))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Assign(t, _, _) if t == "obj.Flag"));
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    /// 23. Non-match: set_Item (handled by indexer pass).
    #[test]
    fn set_item_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(setter_call(
            Some(var("arr")),
            "set_Item",
            vec![Expr::Int(0), Expr::Str("value".into())],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "set_Item must not be rewritten by this pass");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "set_Item"));
    }

    /// 24. Non-match: set_X with wrong arg count (instance form needs exactly 1 arg).
    #[test]
    fn setter_wrong_arg_count() {
        let mut stmts = vec![Stmt::Eval(setter_call(
            Some(var("obj")),
            "set_Value",
            vec![Expr::Int(1), Expr::Int(2)],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "setter with 2 args in instance form must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "set_Value"));
    }

    /// 25. Non-match: bare "set_" with empty suffix.
    #[test]
    fn bare_set_prefix_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(setter_call(
            Some(var("obj")),
            "set_",
            vec![Expr::Int(1)],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "bare set_ with empty suffix must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "set_"));
    }

    /// 26. Setter with cast receiver: `((Type)obj).set_Field(v)` → `((Type)obj).Field = v`
    #[test]
    fn setter_with_cast_receiver() {
        let recv = Expr::Cast("Type".into(), Box::new(var("obj")));
        let mut stmts = vec![Stmt::Eval(setter_call(
            Some(recv),
            "set_Field",
            vec![var("v")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, None, Expr::Var(v)) => {
                assert_eq!(target, "((Type)obj).Field");
                assert_eq!(v, "v");
            }
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    /// 27. Static setter with wrong arg count (needs exactly 2 args).
    #[test]
    fn static_setter_wrong_arg_count() {
        let mut stmts = vec![Stmt::Eval(setter_call(
            None,
            "set_Property",
            vec![var("value")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "static setter with 1 arg must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { recv: None, name, .. }) if name == "set_Property"));
    }
}
