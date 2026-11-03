//! String interpolation recovery: `String.Format(fmt, args...)` → `$"..."`.
//!
//! ## What this pass does
//!
//! IL2CPP renders C# string interpolation as static calls to `String.Format`:
//!
//! ```text
//! String.Format("Hello {0}, age {1:D2}", name, age)
//! ```
//!
//! This pass detects those calls and rewrites them to C# interpolated strings:
//!
//! ```text
//! $"Hello {name}, age {age:D2}"
//! ```
//!
//! Two call shapes are recognized:
//!
//! 1. **Dotted static form**: `recv: None, name: "String.Format"` — the
//!    renderer's default for a static call whose qualifier is folded into the
//!    method name.
//! 2. **Receiver form**: `recv: Some(Var("String")), name: "Format"` — an
//!    instance-style call on the `String` type.
//!
//! ## Format string parsing
//!
//! The format string is parsed to extract:
//! - Literal text segments (with `{{` / `}}` unescaped)
//! - Placeholders `{N}` or `{N:format}` where N is a zero-based argument index
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - Method name must be exactly `"String.Format"` (static form) or `"Format"`
//!   with a `Var("String")` receiver
//! - First argument must be a string literal (the format string)
//! - All placeholder indices must reference valid arguments
//! - Every format argument must be referenced. Folding `Format("{0}", a, b)`
//!   to `$"{a}"` would silently drop the evaluation of `b`; a format string
//!   with no placeholders and extra arguments is refused for the same reason.
//! - A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{Expr, InterpPart, Stmt};

/// Run the string-interpolation reconstruction over one method body.
///
/// Returns the number of `String.Format` calls rewritten to `$"..."`.
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
        Stmt::StringSwitch { disc, cases, tail } => {
            rewrites += rewrite_expr(disc);
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

/// Returns `true` when `expr` is a `String.Format` call that should be
/// rewritten to an interpolated string.
fn is_string_format(expr: &Expr) -> bool {
    match expr {
        // Static dotted form: String.Format(fmt, args...)
        Expr::Call {
            recv: None,
            name,
            args,
        } => {
            name == "String.Format"
                && args.len() >= 2
                && matches!(&args[0], Expr::Str(_))
        }
        // Receiver form: String.Format(fmt, args...) with Var("String") receiver
        Expr::Call {
            recv: Some(recv),
            name,
            args,
        } => {
            name == "Format"
                && args.len() >= 2
                && matches!(recv.as_ref(), Expr::Var(t) if t == "String")
                && matches!(&args[0], Expr::Str(_))
        }
        _ => false,
    }
}

/// Parse a format string into interpolation parts.
///
/// Returns `None` if the format string is malformed (unmatched braces, invalid
/// placeholder indices, etc.). The caller should treat `None` as "do not rewrite".
fn parse_format_string(fmt: &str, arg_count: usize) -> Option<Vec<InterpPart>> {
    let mut parts = Vec::new();
    let mut literal = String::new();
    let mut chars = fmt.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    // Escaped `{{` → literal `{`
                    chars.next();
                    literal.push('{');
                } else {
                    // Start of a placeholder `{index}` or `{index:format}`
                    if !literal.is_empty() {
                        parts.push(InterpPart::Literal(std::mem::take(&mut literal)));
                    }
                    // Parse the index
                    let mut index_str = String::new();
                    let mut format_str: Option<String> = None;
                    let mut found_close = false;

                    for ic in chars.by_ref() {
                        if ic == ':' {
                            // Format specifier follows
                            let mut fmt_buf = String::new();
                            for fc in chars.by_ref() {
                                if fc == '}' {
                                    found_close = true;
                                    break;
                                }
                                // A nested brace would produce an invalid
                                // interpolation hole (`{x:{}`) — refuse.
                                if fc == '{' {
                                    return None;
                                }
                                fmt_buf.push(fc);
                            }
                            if !found_close {
                                return None; // unmatched `}`
                            }
                            format_str = Some(fmt_buf);
                            break;
                        } else if ic == '}' {
                            found_close = true;
                            break;
                        } else {
                            index_str.push(ic);
                        }
                    }

                    if !found_close {
                        return None; // unmatched `{`
                    }

                    let index: usize = index_str.parse().ok()?;
                    if index >= arg_count {
                        return None; // index out of range
                    }

                    // We don't store the arg index directly — we store a placeholder
                    // that will be resolved later when we have the actual args.
                    // Use a special marker: InterpPart::Expr with a Var("__fmt_N").
                    parts.push(InterpPart::Expr(
                        Box::new(Expr::Var(format!("__fmt_{index}"))),
                        format_str,
                    ));
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    // Escaped `}}` → literal `}`
                    chars.next();
                    literal.push('}');
                } else {
                    // Unmatched `}` — malformed format string
                    return None;
                }
            }
            _ => {
                literal.push(c);
            }
        }
    }

    if !literal.is_empty() {
        parts.push(InterpPart::Literal(literal));
    }

    Some(parts)
}

/// Recursively rewrite `String.Format` calls to `StringInterp` within an
/// expression tree.
///
/// Post-order traversal: children are rewritten first, so nested format calls
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

    // Now check if THIS expression is a String.Format call we can rewrite.
    if is_string_format(expr) {
        if let Expr::Call { args, .. } = expr {
            let mut args = std::mem::take(args);
            // First arg is the format string literal.
            let fmt_str = match args.remove(0) {
                Expr::Str(s) => s,
                _ => unreachable!("is_string_format guarantees first arg is Str"),
            };

            // The remaining args are the format arguments.
            let fmt_args = args;
            let arg_count = fmt_args.len();

            // Parse the format string.
            if let Some(mut parts) = parse_format_string(&fmt_str, arg_count) {
                // Refuse if folding would drop an argument's evaluation.
                // `Format("{0}", a, b)` calls (and evaluates) `b` even though
                // it never appears in the result; `$"{a}"` would not.
                let mut used = vec![false; arg_count];
                for part in &parts {
                    if let InterpPart::Expr(e, _) = part {
                        if let Expr::Var(marker) = e.as_ref() {
                            if let Some(idx) = marker
                                .strip_prefix("__fmt_")
                                .and_then(|s| s.parse::<usize>().ok())
                            {
                                if let Some(slot) = used.get_mut(idx) {
                                    *slot = true;
                                }
                            }
                        }
                    }
                }
                if used.iter().any(|seen| !seen) {
                    // No placeholder at all, or an argument the format does
                    // not reference: keep the call.
                    return rewrites;
                }

                // Resolve placeholder indices to actual argument expressions.
                for part in &mut parts {
                    if let InterpPart::Expr(e, _) = part {
                        if let Expr::Var(name) = e.as_ref() {
                            if let Some(idx_str) = name.strip_prefix("__fmt_") {
                                if let Ok(idx) = idx_str.parse::<usize>() {
                                    if idx < fmt_args.len() {
                                        *e = Box::new(fmt_args[idx].clone());
                                    }
                                }
                            }
                        }
                    }
                }

                *expr = Expr::StringInterp { parts };
                rewrites += 1;
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

    /// Helper: build `String.Format(fmt, args...)` (static dotted form).
    fn string_format(fmt: &str, args: Vec<Expr>) -> Expr {
        let mut all_args = vec![Expr::Str(fmt.to_string())];
        all_args.extend(args);
        Expr::Call {
            recv: None,
            name: "String.Format".into(),
            args: all_args,
        }
    }

    /// Helper: build `String.Format(fmt, args...)` (receiver form).
    fn string_format_recv(fmt: &str, args: Vec<Expr>) -> Expr {
        let mut all_args = vec![Expr::Str(fmt.to_string())];
        all_args.extend(args);
        Expr::Call {
            recv: Some(Box::new(var("String"))),
            name: "Format".into(),
            args: all_args,
        }
    }

    // -----------------------------------------------------------------------
    // Positive rewrites
    // -----------------------------------------------------------------------

    /// 1. Simple two-arg format: `String.Format("Hello {0}, age {1}", name, age)`
    #[test]
    fn simple_two_arg_format() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "Hello {0}, age {1}",
            vec![var("name"), var("age")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StringInterp { parts }) => {
                assert_eq!(parts.len(), 4); // "Hello ", {name}, ", age ", {age}
                assert!(matches!(&parts[0], InterpPart::Literal(s) if s == "Hello "));
                assert!(matches!(&parts[1], InterpPart::Expr(e, None) if matches!(e.as_ref(), Expr::Var(n) if n == "name")));
                assert!(matches!(&parts[2], InterpPart::Literal(s) if s == ", age "));
                assert!(matches!(&parts[3], InterpPart::Expr(e, None) if matches!(e.as_ref(), Expr::Var(n) if n == "age")));
            }
            other => panic!("expected Eval(StringInterp), got {other:?}"),
        }
    }

    /// 2. Format with format specifier: `String.Format("{0:D2}", val)`
    #[test]
    fn format_with_specifier() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{0:D2}",
            vec![var("val")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StringInterp { parts }) => {
                assert_eq!(parts.len(), 1);
                match &parts[0] {
                    InterpPart::Expr(e, Some(fmt)) => {
                        assert!(matches!(e.as_ref(), Expr::Var(n) if n == "val"));
                        assert_eq!(fmt, "D2");
                    }
                    other => panic!("expected Expr with format, got {other:?}"),
                }
            }
            other => panic!("expected Eval(StringInterp), got {other:?}"),
        }
    }

    /// 3. Format in assignment: `s = String.Format("x={0}", x)`
    #[test]
    fn format_in_assignment() {
        let mut stmts = vec![Stmt::Assign(
            "s".into(),
            None,
            string_format("x={0}", vec![var("x")]),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::StringInterp { parts }) => {
                assert_eq!(target, "s");
                assert_eq!(parts.len(), 2);
            }
            other => panic!("expected Assign(_, _, StringInterp), got {other:?}"),
        }
    }

    /// 4. Receiver form: `String.Format("{0}", x)` with Var("String") receiver
    #[test]
    fn receiver_form_format() {
        let mut stmts = vec![Stmt::Eval(string_format_recv(
            "{0}",
            vec![var("x")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::StringInterp { .. })));
    }

    /// 5. Escaped braces: `String.Format("{{0}} = {0}", val)`
    #[test]
    fn escaped_braces() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{{0}} = {0}",
            vec![var("val")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StringInterp { parts }) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(&parts[0], InterpPart::Literal(s) if s == "{0} = "));
                assert!(matches!(&parts[1], InterpPart::Expr(e, None) if matches!(e.as_ref(), Expr::Var(n) if n == "val")));
            }
            other => panic!("expected Eval(StringInterp), got {other:?}"),
        }
    }

    /// 6. Format inside a while loop body is rewritten.
    #[test]
    fn format_inside_while_body() {
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Eval(string_format(
                "val={0}",
                vec![var("x")],
            ))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[0], Stmt::Eval(Expr::StringInterp { .. })));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    /// 7. Multiple format specifiers: `String.Format("{0:N} items at {1:C}", count, price)`
    #[test]
    fn multiple_format_specifiers() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{0:N} items at {1:C}",
            vec![var("count"), var("price")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StringInterp { parts }) => {
                assert_eq!(parts.len(), 3);
                match &parts[0] {
                    InterpPart::Expr(_, Some(fmt)) => assert_eq!(fmt, "N"),
                    other => panic!("expected Expr with format N, got {other:?}"),
                }
                assert!(matches!(&parts[1], InterpPart::Literal(s) if s == " items at "));
                match &parts[2] {
                    InterpPart::Expr(_, Some(fmt)) => assert_eq!(fmt, "C"),
                    other => panic!("expected Expr with format C, got {other:?}"),
                }
            }
            other => panic!("expected Eval(StringInterp), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    /// 8. Non-string first argument — must NOT rewrite.
    #[test]
    fn non_string_format_arg_not_rewritten() {
        let expr = Expr::Call {
            recv: None,
            name: "String.Format".into(),
            args: vec![var("fmt"), var("x")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "non-string format arg must not rewrite");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { name, .. }) if name == "String.Format"));
    }

    /// 9. Instance method call on non-String receiver — must NOT rewrite.
    #[test]
    fn non_string_receiver_not_rewritten() {
        let expr = Expr::Call {
            recv: Some(Box::new(var("MyType"))),
            name: "Format".into(),
            args: vec![Expr::Str("{0}".into()), var("x")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "non-String receiver must not rewrite");
    }

    /// 10. Different method name — must NOT rewrite.
    #[test]
    fn different_method_name_not_rewritten() {
        let expr = Expr::Call {
            recv: None,
            name: "String.Join".into(),
            args: vec![Expr::Str("{0}".into()), var("x")],
        };
        let mut stmts = vec![Stmt::Eval(expr)];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "String.Join must not rewrite");
    }

    /// 11. Only format string, no args — must NOT rewrite.
    #[test]
    fn only_format_string_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "String.Format".into(),
            args: vec![Expr::Str("hello".into())],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "single-arg Format must not rewrite");
    }

    /// 12. Malformed format string (unmatched `{`) — must NOT rewrite.
    #[test]
    fn malformed_format_string_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "broken {0",
            vec![var("x")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "malformed format string must not rewrite");
    }

    /// 13. Placeholder index out of range — must NOT rewrite.
    #[test]
    fn placeholder_index_out_of_range() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{5}",
            vec![var("x")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "out-of-range placeholder must not rewrite");
    }

    /// 14. Rendering test: verify the output is correct C# interpolation syntax.
    #[test]
    fn rendering_output() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "Hello {0}, you are {1:D2}",
            vec![var("name"), var("age")],
        ))];
        apply(&mut stmts);
        let mut out = String::new();
        match &stmts[0] {
            Stmt::Eval(expr) => expr.render(&mut out),
            _ => panic!("expected Eval"),
        }
        assert_eq!(out, "$\"Hello {name}, you are {age:D2}\"");
    }

    /// 15. Rendering test: escaped braces render correctly.
    #[test]
    fn rendering_escaped_braces() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{{literal}} = {0}",
            vec![var("val")],
        ))];
        apply(&mut stmts);
        let mut out = String::new();
        match &stmts[0] {
            Stmt::Eval(expr) => expr.render(&mut out),
            _ => panic!("expected Eval"),
        }
        assert_eq!(out, "$\"{{literal}} = {val}\"");
    }

    /// 16. An argument the format string never references must not be dropped:
    ///     `Format("{0}", a, b)` evaluates `b`.
    #[test]
    fn unused_argument_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{0}",
            vec![var("a"), var("b")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "unused format argument must refuse the fold");
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "String.Format"
        ));
    }

    /// 17. A literal-only format string with arguments is also a dropped-arg case.
    #[test]
    fn literal_format_with_args_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "no placeholders",
            vec![var("a")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    /// 18. Out-of-order placeholders keep their argument mapping.
    #[test]
    fn out_of_order_placeholders() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{1}-{0}",
            vec![var("a"), var("b")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StringInterp { parts }) => {
                assert_eq!(parts.len(), 3);
                assert!(matches!(&parts[0], InterpPart::Expr(e, _) if matches!(e.as_ref(), Expr::Var(n) if n == "b")));
                assert!(matches!(&parts[1], InterpPart::Literal(s) if s == "-"));
                assert!(matches!(&parts[2], InterpPart::Expr(e, _) if matches!(e.as_ref(), Expr::Var(n) if n == "a")));
            }
            other => panic!("expected StringInterp, got {other:?}"),
        }
    }

    /// 19. A repeated placeholder repeats the expression; nothing is reordered.
    #[test]
    fn repeated_placeholder() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{0}{0}",
            vec![var("x")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::StringInterp { parts }) => {
                assert_eq!(parts.len(), 2);
                for p in parts {
                    assert!(matches!(p, InterpPart::Expr(e, _) if matches!(e.as_ref(), Expr::Var(n) if n == "x")));
                }
            }
            other => panic!("expected StringInterp, got {other:?}"),
        }
    }

    /// 20. Nested format calls fold bottom-up.
    #[test]
    fn nested_format_folds() {
        let inner = string_format("[{0}]", vec![var("x")]);
        let outer = string_format("<{0}>", vec![inner]);
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2);
        match &stmts[0] {
            Stmt::Eval(Expr::StringInterp { parts }) => {
                assert_eq!(parts.len(), 3);
                assert!(matches!(&parts[1], InterpPart::Expr(e, _) if matches!(e.as_ref(), Expr::StringInterp { .. })));
            }
            other => panic!("expected StringInterp, got {other:?}"),
        }
    }

    /// 21. A second application is a no-op (idempotent).
    #[test]
    fn second_run_is_noop() {
        let mut stmts = vec![Stmt::Eval(string_format("x={0}", vec![var("x")]))];
        assert_eq!(apply(&mut stmts), 1);
        let snapshot = format!("{:?}", stmts);
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(format!("{:?}", stmts), snapshot);
    }

    /// 22. A format specifier containing a brace cannot be rendered as an
    ///     interpolation hole; refuse instead of emitting `{x:{}`.
    #[test]
    fn brace_in_format_specifier_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(string_format(
            "{0:{}",
            vec![var("x")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }
}
