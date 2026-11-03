//! Operator overloading reconstruction:
//! `Type.op_Addition(a, b)` -> `a + b`,
//! `Type.op_Subtraction(a, b)` -> `a - b`, etc.
//! `Type.op_Explicit(value)` -> `(Type)value`,
//! `Type.op_Implicit(value)` -> `value`.
//!
//! ## What this pass does
//!
//! IL2CPP renders C# operator overloads as static method calls:
//!
//! ```text
//! Vector3.op_Addition(a, b)     ->  a + b
//! Vector3.op_Subtraction(a, b)  ->  a - b
//! Vector3.op_Multiply(a, b)     ->  a * b
//! SafeDouble.op_Explicit(v)     ->  (SafeDouble)v
//! SafeDouble.op_Implicit(v)     ->  v
//! ```
//!
//! This pass rewrites ~20 standard C# operator names to their binary operator form,
//! plus conversion operators (op_Explicit / op_Implicit).
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - Binary operators: static call only (`recv` is `None`), method name must match
//!   `Type.op_OperatorName` pattern, must have exactly 2 arguments.
//! - Conversion operators: static call with exactly 1 argument, or instance call
//!   with 0 arguments. For op_Explicit the declaring type (extracted from the
//!   dotted name) becomes the cast target. For op_Implicit the value is returned
//!   as-is (no visible cast needed).
//! - A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{BinOp, CmpKind, Expr, Stmt};

/// Run the operator-overload reconstruction over one method body.
///
/// Returns the number of operator calls rewritten to binary operators.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut rewrites = 0;
    for stmt in stmts.iter_mut() {
        rewrites += rewrite_stmt(stmt);
    }
    rewrites
}

/// Which `CmpKind` a dotted `Type.op_*` comparison operator denotes.
///
/// The longer names are tested FIRST: `op_LessThan` is a prefix of
/// `op_LessThanOrEqual`, and `op_GreaterThan` of `op_GreaterThanOrEqual`, so a
/// shortest-first table would silently render every `<=` as `<`.
///
/// Only the DOTTED form (`Type.op_Equality`) is accepted. A bare
/// `op_Equality` carries no declaring type, and `==` on an unknown operand
/// type is not the same operator call.
fn comparison_operator(name: &str) -> Option<CmpKind> {
    const TABLE: &[(&str, CmpKind)] = &[
        (".op_Equality", CmpKind::Eq),
        (".op_Inequality", CmpKind::Ne),
        (".op_LessThanOrEqual", CmpKind::Le),
        (".op_LessThan", CmpKind::Lt),
        (".op_GreaterThanOrEqual", CmpKind::Ge),
        (".op_GreaterThan", CmpKind::Gt),
    ];
    TABLE
        .iter()
        .find_map(|(suffix, kind)| name.ends_with(suffix).then_some(*kind))
}

/// Rewrite a single statement, recursing into nested bodies.
fn rewrite_stmt(stmt: &mut Stmt) -> usize {
    let mut rewrites = 0;
    match stmt {
        Stmt::Declare(..) => {}
        Stmt::Eval(expr) => {
            // IL2CPP output-pointer convention for conversion operators:
            // `Type.op_Implicit(value, &stack_N)` -> `stack_N = value`
            // `Type.op_Explicit(value, &stack_N)` -> `stack_N = (Type)value`
            if let Expr::Call {
                recv: None,
                name,
                args,
            } = expr
            {
                if args.len() == 2 {
                    let is_implicit = name.ends_with(".op_Implicit");
                    let is_explicit = name.ends_with(".op_Explicit");
                    if is_implicit || is_explicit {
                        // Second arg should be AddressOf(Var(...)) — the output slot.
                        if let Expr::AddressOf(inner) = &args[1] {
                            if let Expr::Var(slot_name) = inner.as_ref() {
                                let slot = slot_name.clone();
                                let value = std::mem::replace(&mut args[0], Expr::Null);
                                let type_name = if is_explicit {
                                    // Extract declaring type for the cast.
                                    name.strip_suffix(".op_Explicit")
                                        .filter(|t| !t.is_empty())
                                        .unwrap_or("object")
                                        .to_string()
                                } else {
                                    // Implicit: no visible cast needed.
                                    String::new()
                                };
                                let rhs = if is_explicit {
                                    Expr::Cast(type_name, Box::new(value))
                                } else {
                                    value
                                };
                                *stmt = Stmt::Assign(slot, Some("auto".to_string()), rhs);
                                return rewrites + 1;
                            }
                        }
                    }
                }
            }
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
        Stmt::StringSwitch { cases, tail, .. } => {
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
        Stmt::ForEach { body, .. } => {
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

/// Recursively rewrite operator calls to binary operators within an expression tree.
///
/// Post-order traversal: children are rewritten first, so nested operator calls
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

    // Now check if THIS expression is an operator call we can rewrite.

    // --- Binary operators (static call, 2 args) ---
    if let Expr::Call {
        recv: None,
        name,
        args,
    } = expr
    {
        if args.len() == 2 {
            // Extract operator name from "Type.op_OperatorName" pattern
            if let Some(op_name) = name.strip_suffix(".op_Addition") {
                let _ = op_name; // Type name not needed for rewrite
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Add, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_Subtraction") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Sub, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_Multiply") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Mul, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_Division") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Div, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_Modulus") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Rem, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_BitwiseAnd") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::And, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_BitwiseOr") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Or, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_ExclusiveOr") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Xor, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_LeftShift") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Shl, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if name.ends_with(".op_RightShift") {
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Bin(BinOp::Shr, Box::new(a), Box::new(b));
                return rewrites + 1;
            } else if let Some(kind) = comparison_operator(name) {
                // COMPARISON OPERATORS. C# REQUIRES `==`, `!=`, `<`, `<=`,
                // `>`, `>=` to be declared as `op_Equality`, `op_Inequality`,
                // `op_LessThan`, `op_LessThanOrEqual`, `op_GreaterThan`,
                // `op_GreaterThanOrEqual`, so a call to one of those methods on
                // operands of its own type IS that operator: `T.op_Equality(a,b)`
                // and `a == b` denote the same call. Nothing is inferred.
                //
                // This arm is why `CmpKind` was imported at the top of this
                // module and never used — the comparison half of the table was
                // the one that never got written. Measured on R76: 726
                // `String.op_Equality`, 270 `op_Inequality`, 64 `op_LessThan`,
                // 62 `op_LessThanOrEqual`, 53 `op_GreaterThan`,
                // 52 `op_GreaterThanOrEqual` — 1,227 operator calls rendered as
                // method calls.
                //
                // The literal `0` argument is left as `Int(0)`: `compare_sugar`
                // runs after this pass and turns `x == 0` into `x == null` when
                // the operand is provably a reference. Deciding it here would
                // need the parameter's declared type, which this pass does not
                // have, and a value-type operator legitimately takes a zero.
                let a = std::mem::replace(&mut args[0], Expr::Null);
                let b = std::mem::replace(&mut args[1], Expr::Null);
                *expr = Expr::Cmp(kind, Box::new(a), Box::new(b));
                return rewrites + 1;
            }
        }

        // --- Conversion operators (static call, 1 arg) ---
        // `Type.op_Explicit(value)` -> `(Type)value`
        // `Type.op_Implicit(value)` -> `value`
        if args.len() == 1 {
            if let Some(type_name) = name.strip_suffix(".op_Explicit") {
                if !type_name.is_empty() {
                    let value = std::mem::replace(&mut args[0], Expr::Null);
                    *expr = Expr::Cast(type_name.to_string(), Box::new(value));
                    return rewrites + 1;
                }
            } else if let Some(type_name) = name.strip_suffix(".op_Implicit") {
                if !type_name.is_empty() {
                    let value = std::mem::replace(&mut args[0], Expr::Null);
                    *expr = value;
                    return rewrites + 1;
                }
            } else if name.ends_with(".op_UnaryNegation") {
                let value = std::mem::replace(&mut args[0], Expr::Null);
                *expr = Expr::Neg(Box::new(value));
                return rewrites + 1;
            } else if name.ends_with(".op_UnaryPlus") {
                let value = std::mem::replace(&mut args[0], Expr::Null);
                *expr = value;
                return rewrites + 1;
            } else if name.ends_with(".op_LogicalNot") {
                let value = std::mem::replace(&mut args[0], Expr::Null);
                *expr = Expr::Not(Box::new(value));
                return rewrites + 1;
            } else if name.ends_with(".op_Increment") {
                let value = std::mem::replace(&mut args[0], Expr::Null);
                *expr = Expr::Bin(BinOp::Add, Box::new(value), Box::new(Expr::Int(1)));
                return rewrites + 1;
            } else if name.ends_with(".op_Decrement") {
                let value = std::mem::replace(&mut args[0], Expr::Null);
                *expr = Expr::Bin(BinOp::Sub, Box::new(value), Box::new(Expr::Int(1)));
                return rewrites + 1;
            } else if name.ends_with(".op_True") {
                let value = std::mem::replace(&mut args[0], Expr::Null);
                *expr = value;
                return rewrites + 1;
            } else if name.ends_with(".op_False") {
                let value = std::mem::replace(&mut args[0], Expr::Null);
                *expr = Expr::Not(Box::new(value));
                return rewrites + 1;
            }
        }
    }

    // --- Conversion operators (instance call, 0 args) ---
    // `recv.op_Explicit()` -> we can't determine target type without metadata,
    //   but if the name is dotted (e.g. recv is a field access and name is
    //   "Type.op_Explicit"), extract the type. For bare `op_Explicit` with a
    //   receiver, leave as-is (no type info).
    // `recv.op_Implicit()` -> `recv` (implicit conversion, no cast needed).
    // Unary operators (instance form, 0 args):
    // `recv.op_UnaryNegation()` -> `-recv`
    // `recv.op_UnaryPlus()` -> `recv`
    // `recv.op_LogicalNot()` -> `!recv`
    // `recv.op_Increment()` -> `recv + 1`
    // `recv.op_Decrement()` -> `recv - 1`
    // `recv.op_True()` -> `recv`
    // `recv.op_False()` -> `!recv`
    if let Expr::Call {
        recv: Some(receiver),
        name,
        args,
    } = expr
    {
        if args.is_empty() {
            if name == "op_Implicit" || name == "op_UnaryPlus" || name == "op_True" {
                let value = std::mem::replace(receiver, Box::new(Expr::Null));
                *expr = *value;
                return rewrites + 1;
            } else if name == "op_UnaryNegation" {
                let value = std::mem::replace(receiver, Box::new(Expr::Null));
                *expr = Expr::Neg(value);
                return rewrites + 1;
            } else if name == "op_LogicalNot" || name == "op_False" {
                let value = std::mem::replace(receiver, Box::new(Expr::Null));
                *expr = Expr::Not(value);
                return rewrites + 1;
            } else if name == "op_Increment" {
                let value = std::mem::replace(receiver, Box::new(Expr::Null));
                *expr = Expr::Bin(BinOp::Add, value, Box::new(Expr::Int(1)));
                return rewrites + 1;
            } else if name == "op_Decrement" {
                let value = std::mem::replace(receiver, Box::new(Expr::Null));
                *expr = Expr::Bin(BinOp::Sub, value, Box::new(Expr::Int(1)));
                return rewrites + 1;
            } else if name == "op_Explicit" {
                // Instance-form explicit cast: no target type available from the
                // name alone. Leave as-is to avoid inventing a wrong type.
                // (Static form with dotted name is handled above.)
            }
        }
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    /// Helper: build `Type.op_Addition(a, b)` (static dotted form).
    fn op_add(a: Expr, b: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "Vector3.op_Addition".into(),
            args: vec![a, b],
        }
    }

    #[test]
    fn test_op_addition_rewrite() {
        let mut expr = op_add(var("a"), var("b"));
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Bin(BinOp::Add, _, _) => {}
            _ => panic!("expected Bin(Add, ...)"),
        }
    }

    #[test]
    fn test_op_subtraction_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "Vector3.op_Subtraction".into(),
            args: vec![var("a"), var("b")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Bin(BinOp::Sub, _, _) => {}
            _ => panic!("expected Bin(Sub, ...)"),
        }
    }

    #[test]
    fn test_op_multiply_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "Vector3.op_Multiply".into(),
            args: vec![var("a"), var("b")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Bin(BinOp::Mul, _, _) => {}
            _ => panic!("expected Bin(Mul, ...)"),
        }
    }

    // --- Comparison operator tests ---

    fn cmp_call(name: &str, a: Expr, b: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: name.into(),
            args: vec![a, b],
        }
    }

    /// `String.op_Equality(a, b)` IS `a == b`: C# requires the `==` operator to
    /// be declared under that exact name, so the call and the operator are the
    /// same method. 726 of these were rendered as method calls on R76.
    #[test]
    fn string_equality_becomes_double_equals() {
        let mut expr = cmp_call("String.op_Equality", var("a"), var("b"));
        assert_eq!(rewrite_expr(&mut expr), 1);
        assert!(matches!(expr, Expr::Cmp(CmpKind::Eq, ..)));
        let mut text = String::new();
        expr.render(&mut text);
        assert_eq!(text, "a == b");
    }

    /// `<= ` and `>=` must not be swallowed by the shorter `<`/`>` suffixes.
    /// This is the ordering bug the table's longest-first order exists for.
    #[test]
    fn less_or_equal_is_not_less_than() {
        let mut le = cmp_call("Int32.op_LessThanOrEqual", var("a"), var("b"));
        assert_eq!(rewrite_expr(&mut le), 1);
        assert!(matches!(le, Expr::Cmp(CmpKind::Le, ..)));
        let mut ge = cmp_call("Int32.op_GreaterThanOrEqual", var("a"), var("b"));
        assert_eq!(rewrite_expr(&mut ge), 1);
        assert!(matches!(ge, Expr::Cmp(CmpKind::Ge, ..)));
        let mut lt = cmp_call("Int32.op_LessThan", var("a"), var("b"));
        assert_eq!(rewrite_expr(&mut lt), 1);
        assert!(matches!(lt, Expr::Cmp(CmpKind::Lt, ..)));
    }

    /// A bare `op_Equality` names no declaring type, so the operator it denotes
    /// is not proven. It must stay a call.
    #[test]
    fn undotted_equality_is_not_rewritten() {
        let mut expr = cmp_call("op_Equality", var("a"), var("b"));
        assert_eq!(rewrite_expr(&mut expr), 0);
        assert!(matches!(expr, Expr::Call { .. }));
    }

    /// `TrackedReference.op_Inequality(x, 0)` becomes `x != 0` here; `compare_sugar`
    /// runs later and rewrites the zero to `null` once the operand is proven to
    /// be a reference. The literal is NOT guessed at in this pass.
    #[test]
    fn inequality_keeps_the_zero_literal_for_compare_sugar() {
        let mut expr = cmp_call("TrackedReference.op_Inequality", var("x"), Expr::Int(0));
        assert_eq!(rewrite_expr(&mut expr), 1);
        assert!(matches!(expr, Expr::Cmp(CmpKind::Ne, _, ref b) if **b == Expr::Int(0)));
    }

    #[test]
    fn test_non_operator_not_rewritten() {
        let mut expr = Expr::Call {
            recv: None,
            name: "SomeClass.SomeMethod".into(),
            args: vec![var("a"), var("b")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 0);
        match expr {
            Expr::Call { name, .. } if name == "SomeClass.SomeMethod" => {}
            _ => panic!("expected Call to be unchanged"),
        }
    }

    #[test]
    fn test_instance_call_not_rewritten() {
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("obj"))),
            name: "op_Addition".into(),
            args: vec![var("a"), var("b")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 0);
    }

    // --- Conversion operator tests ---

    #[test]
    fn test_op_explicit_static_rewrite() {
        // SafeDouble.op_Explicit(value) -> (SafeDouble)value
        let mut expr = Expr::Call {
            recv: None,
            name: "SafeDouble.op_Explicit".into(),
            args: vec![var("value")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match &expr {
            Expr::Cast(type_name, inner) => {
                assert_eq!(type_name, "SafeDouble");
                match inner.as_ref() {
                    Expr::Var(name) => assert_eq!(name, "value"),
                    _ => panic!("expected Var inside Cast"),
                }
            }
            _ => panic!("expected Cast, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_implicit_static_rewrite() {
        // SafeDouble.op_Implicit(value) -> value
        let mut expr = Expr::Call {
            recv: None,
            name: "SafeDouble.op_Implicit".into(),
            args: vec![var("value")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match &expr {
            Expr::Var(name) => assert_eq!(name, "value"),
            _ => panic!("expected Var after implicit conversion, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_implicit_instance_rewrite() {
        // value.op_Implicit() -> value (instance form, 0 args)
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("myVal"))),
            name: "op_Implicit".into(),
            args: vec![],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match &expr {
            Expr::Var(name) => assert_eq!(name, "myVal"),
            _ => panic!("expected Var after instance implicit, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_explicit_instance_not_rewritten() {
        // value.op_Explicit() -> left as-is (no target type info)
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("myVal"))),
            name: "op_Explicit".into(),
            args: vec![],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 0);
        match &expr {
            Expr::Call { name, args, .. } => {
                assert_eq!(name, "op_Explicit");
                assert!(args.is_empty());
            }
            _ => panic!("expected Call to be unchanged"),
        }
    }

    #[test]
    fn test_op_explicit_no_type_prefix_not_rewritten() {
        // A bare "op_Explicit" as a static call (no type prefix) should not be
        // rewritten since we can't determine the cast target type.
        let mut expr = Expr::Call {
            recv: None,
            name: "op_Explicit".into(),
            args: vec![var("value")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_op_implicit_nested_in_assign() {
        // Test that conversion operators are rewritten inside statements.
        let mut stmts = vec![Stmt::Assign(
            "result".to_string(),
            Some("double".to_string()),
            Expr::Call {
                recv: None,
                name: "SafeDouble.op_Implicit".into(),
                args: vec![var("sd")],
            },
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, expr) => match expr {
                Expr::Var(name) => assert_eq!(name, "sd"),
                _ => panic!("expected Var after implicit in assign"),
            },
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn test_op_explicit_nested_in_cast_context() {
        // Test op_Explicit inside a Return statement.
        let mut stmts = vec![Stmt::Return(Some(Expr::Call {
            recv: None,
            name: "MyType.op_Explicit".into(),
            args: vec![var("input")],
        }))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Return(Some(expr)) => match expr {
                Expr::Cast(type_name, inner) => {
                    assert_eq!(type_name, "MyType");
                    match inner.as_ref() {
                        Expr::Var(name) => assert_eq!(name, "input"),
                        _ => panic!("expected Var inside Cast"),
                    }
                }
                _ => panic!("expected Cast in return"),
            },
            _ => panic!("expected Return(Some(...))"),
        }
    }

    #[test]
    fn test_op_implicit_output_pointer_form() {
        // IL2CPP output-pointer convention:
        // `MinMaxCurve.op_Implicit(0f, &stack_100)` -> `stack_100 = 0f`
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "MinMaxCurve.op_Implicit".into(),
            args: vec![Expr::Float(0.0), Expr::AddressOf(Box::new(var("stack_100")))],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(slot, _, rhs) => {
                assert_eq!(slot, "stack_100");
                match rhs {
                    Expr::Float(f) => assert_eq!(*f, 0.0),
                    _ => panic!("expected Float on rhs"),
                }
            }
            _ => panic!("expected Assign, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn test_op_explicit_output_pointer_form() {
        // `SafeDouble.op_Explicit(value, &stack_N)` -> `stack_N = (SafeDouble)value`
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "SafeDouble.op_Explicit".into(),
            args: vec![var("value"), Expr::AddressOf(Box::new(var("stack_50")))],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(slot, _, rhs) => {
                assert_eq!(slot, "stack_50");
                match rhs {
                    Expr::Cast(type_name, inner) => {
                        assert_eq!(type_name, "SafeDouble");
                        match inner.as_ref() {
                            Expr::Var(name) => assert_eq!(name, "value"),
                            _ => panic!("expected Var inside Cast"),
                        }
                    }
                    _ => panic!("expected Cast on rhs"),
                }
            }
            _ => panic!("expected Assign, got {:?}", stmts[0]),
        }
    }

    // --- Comparison operator tests ---

    // --- Unary operator tests (static form) ---

    #[test]
    fn test_op_unary_negation_static_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "Vector3.op_UnaryNegation".into(),
            args: vec![var("a")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Neg(inner) => match *inner {
                Expr::Var(name) => assert_eq!(name, "a"),
                _ => panic!("expected Var inside Neg"),
            },
            _ => panic!("expected Neg, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_unary_plus_static_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "Vector3.op_UnaryPlus".into(),
            args: vec![var("a")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Var(name) => assert_eq!(name, "a"),
            _ => panic!("expected Var after unary plus, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_logical_not_static_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "MyType.op_LogicalNot".into(),
            args: vec![var("flag")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Not(inner) => match *inner {
                Expr::Var(name) => assert_eq!(name, "flag"),
                _ => panic!("expected Var inside Not"),
            },
            _ => panic!("expected Not, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_increment_static_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "Counter.op_Increment".into(),
            args: vec![var("count")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Bin(BinOp::Add, _, _) => {}
            _ => panic!("expected Bin(Add, ...) for increment, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_decrement_static_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "Counter.op_Decrement".into(),
            args: vec![var("count")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Bin(BinOp::Sub, _, _) => {}
            _ => panic!("expected Bin(Sub, ...) for decrement, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_true_static_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "MyType.op_True".into(),
            args: vec![var("obj")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Var(name) => assert_eq!(name, "obj"),
            _ => panic!("expected Var after op_True, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_false_static_rewrite() {
        let mut expr = Expr::Call {
            recv: None,
            name: "MyType.op_False".into(),
            args: vec![var("obj")],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Not(inner) => match *inner {
                Expr::Var(name) => assert_eq!(name, "obj"),
                _ => panic!("expected Var inside Not"),
            },
            _ => panic!("expected Not after op_False, got {:?}", expr),
        }
    }

    // --- Unary operator tests (instance form) ---

    #[test]
    fn test_op_unary_negation_instance_rewrite() {
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("obj"))),
            name: "op_UnaryNegation".into(),
            args: vec![],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Neg(inner) => match *inner {
                Expr::Var(name) => assert_eq!(name, "obj"),
                _ => panic!("expected Var inside Neg"),
            },
            _ => panic!("expected Neg, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_logical_not_instance_rewrite() {
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("flag"))),
            name: "op_LogicalNot".into(),
            args: vec![],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Not(inner) => match *inner {
                Expr::Var(name) => assert_eq!(name, "flag"),
                _ => panic!("expected Var inside Not"),
            },
            _ => panic!("expected Not, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_increment_instance_rewrite() {
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("counter"))),
            name: "op_Increment".into(),
            args: vec![],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Bin(BinOp::Add, _, _) => {}
            _ => panic!("expected Bin(Add, ...) for instance increment, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_decrement_instance_rewrite() {
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("counter"))),
            name: "op_Decrement".into(),
            args: vec![],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Bin(BinOp::Sub, _, _) => {}
            _ => panic!("expected Bin(Sub, ...) for instance decrement, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_true_instance_rewrite() {
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("obj"))),
            name: "op_True".into(),
            args: vec![],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Var(name) => assert_eq!(name, "obj"),
            _ => panic!("expected Var after instance op_True, got {:?}", expr),
        }
    }

    #[test]
    fn test_op_false_instance_rewrite() {
        let mut expr = Expr::Call {
            recv: Some(Box::new(var("obj"))),
            name: "op_False".into(),
            args: vec![],
        };
        let n = rewrite_expr(&mut expr);
        assert_eq!(n, 1);
        match expr {
            Expr::Not(inner) => match *inner {
                Expr::Var(name) => assert_eq!(name, "obj"),
                _ => panic!("expected Var inside Not"),
            },
            _ => panic!("expected Not after instance op_False, got {:?}", expr),
        }
    }
}
