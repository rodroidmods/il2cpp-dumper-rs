//! LINQ method-chain reconstruction:
//! `Enumerable.Select(Enumerable.Where(source, pred), transform)`
//!   -> `source.Where(pred).Select(transform)`
//!
//! ## What this pass does
//!
//! IL2CPP renders C# LINQ extension methods as static calls on
//! `System.Linq.Enumerable`. A chained LINQ expression like:
//!
//! ```csharp
//! source.Where(x => x > 0).Select(x => x * 2).OrderBy(x => x)
//! ```
//!
//! arrives as nested static calls:
//!
//! ```text
//! Enumerable.OrderBy(Enumerable.Select(Enumerable.Where(source, pred), transform), key)
//! ```
//!
//! This pass detects those nested patterns and rewrites them into a
//! `Expr::MethodChain` that renders as fluent dot-chained method calls.
//!
//! ## Recognized patterns
//!
//! 1. **Static extension-method nesting**: `Enumerable.Method(inner_call, args...)`
//!    where `inner_call` is itself an `Enumerable.Method(...)` or any expression.
//!    The first argument is the source/receiver; remaining arguments are the
//!    method's parameters.
//!
//! 2. **Instance method chaining**: `Call { recv: Call { recv: source, name: "Where", args }, name: "Select", args }`
//!    Already in chain form but flattened into `MethodChain` for consistent rendering.
//!
//! ## Recognized LINQ methods
//!
//! All standard `System.Linq.Enumerable` extension methods:
//! Where, Select, SelectMany, OrderBy, OrderByDescending, ThenBy, ThenByDescending,
//! First, FirstOrDefault, Last, LastOrDefault, Single, SingleOrDefault,
//! Any, All, Contains, Count, Sum, Min, Max, Average,
//! Take, Skip, TakeWhile, SkipWhile, Distinct, GroupBy, Join, GroupJoin,
//! Aggregate, Reverse, ToList, ToArray, ToDictionary, ToHashSet,
//! Zip, DefaultIfEmpty, Empty, Range, Repeat, Concat, Union, Intersect, Except,
//! Cast, OfType, AsEnumerable, AsQueryable.
//!
//! ## Safety
//!
//! Only proven layouts are rewritten:
//! - The qualifier must be exactly `Enumerable` (or end in `.Enumerable`); the
//!   method name may carry a generic instantiation (`Any<object>`,
//!   `Select<object, long>`) because IL2CPP spells nearly every LINQ call that
//!   way. The generic suffix is stripped before matching.
//! - `Range` / `Repeat` / `Empty` are static factories, not extension methods:
//!   their first argument is NOT a receiver, so they are never folded.
//! - The static form must carry the source plus the minimum number of
//!   parameters the known overload needs. Argument order is preserved.
//! - A call whose first parameter is structurally the source again is refused
//!   (it would produce `src.Method(src)` from a shape we did not prove).
//! - An ordinary instance call (`list.Contains(x)`) is left alone: only calls
//!   that already have a LINQ-shaped receiver are folded into a chain.
//! - A failed match is a no-op: the original call form is preserved.

use super::super::csharp::{ChainMethod, Expr, Stmt};

/// The set of `System.Linq.Enumerable` extension method names we recognize.
///
/// These are the first-argument-is-source extension methods that the C# compiler
/// lowers to static calls on `System.Linq.Enumerable`.
const LINQ_METHODS: &[&str] = &[
    "Where",
    "Select",
    "SelectMany",
    "OrderBy",
    "OrderByDescending",
    "ThenBy",
    "ThenByDescending",
    "First",
    "FirstOrDefault",
    "Last",
    "LastOrDefault",
    "Single",
    "SingleOrDefault",
    "Any",
    "All",
    "Contains",
    "Count",
    "LongCount",
    "Sum",
    "Min",
    "Max",
    "Average",
    "Take",
    "Skip",
    "TakeWhile",
    "SkipWhile",
    "Distinct",
    "GroupBy",
    "Join",
    "GroupJoin",
    "Aggregate",
    "Reverse",
    "ToList",
    "ToArray",
    "ToDictionary",
    "ToHashSet",
    "Zip",
    "DefaultIfEmpty",
    "Concat",
    "Union",
    "Intersect",
    "Except",
    "Cast",
    "OfType",
    "AsEnumerable",
    "AsQueryable",
];

/// Static `Enumerable` factories: valid LINQ entry points, but not extension
/// methods. Their first argument is a value (range start, repeated element),
/// never a source sequence, so folding them would invert the call direction.
const NON_EXTENSION_METHODS: &[&str] = &["Range", "Repeat", "Empty"];

/// Returns `true` if `name` is a recognized LINQ extension method name.
///
/// Accepts bare names (`"Where"`), qualified names
/// (`"Enumerable.Where"`, `"System.Linq.Enumerable.Where"`) and generic
/// instantiations (`"Enumerable.Any<object>"`).
#[allow(dead_code)]
fn is_linq_method(name: &str) -> bool {
    linq_method_name(name).is_some()
}

/// Extract the bare extension-method name from a possibly-qualified name.
///
/// IL2CPP spells generic methods with their instantiation attached, e.g.
/// `System.Linq.Enumerable.Select<object, long>`. Splitting on the last `.`
/// of the raw string would yield `Select<object, long>`, which matches no
/// known method, so the generic suffix is removed first. `Range`, `Repeat`
/// and `Empty` deliberately return `None` (see [`NON_EXTENSION_METHODS`]).
fn linq_method_name(name: &str) -> Option<&str> {
    let head = name.split('<').next().unwrap_or(name);
    let bare = head.rsplit('.').next().unwrap_or(head);
    if LINQ_METHODS.contains(&bare) && !NON_EXTENSION_METHODS.contains(&bare) {
        Some(bare)
    } else {
        None
    }
}

/// Is `qualifier` the LINQ owner type (`Enumerable` or `*.Enumerable`)?
///
/// Exact-match instead of `contains("Enumerable")`: a `MyEnumerableHelper.Where`
/// static helper has the same method name but no source-first argument layout.
fn is_enumerable_qualifier(qualifier: &str) -> bool {
    qualifier == "Enumerable" || qualifier.ends_with(".Enumerable")
}

/// If `name` is a dotted static call on `[System.Linq.]Enumerable`, return the
/// bare extension-method name. The source is `args[0]`.
fn static_linq_method(name: &str) -> Option<&str> {
    let head = name.split('<').next().unwrap_or(name);
    let (qualifier, bare) = head.rsplit_once('.')?;
    if !is_enumerable_qualifier(qualifier) {
        return None;
    }
    if !LINQ_METHODS.contains(&bare) || NON_EXTENSION_METHODS.contains(&bare) {
        return None;
    }
    Some(bare)
}

/// Minimum number of parameters after the source for a known overload.
///
/// We only need the minimum: the rewrite keeps every argument in order, so a
/// longer overload is still the same source-first layout. A shorter call is a
/// shape we cannot prove (`Enumerable.Select(source)` is not a Select call).
fn min_params_after_source(method: &str) -> usize {
    match method {
        "Join" | "GroupJoin" => 4,
        "Where" | "Select" | "SelectMany" | "OrderBy" | "OrderByDescending"
        | "ThenBy" | "ThenByDescending" | "GroupBy" | "Aggregate" | "Contains"
        | "Take" | "Skip" | "TakeWhile" | "SkipWhile" | "All" | "Concat"
        | "Union" | "Intersect" | "Except" | "Zip" | "ToDictionary" => 1,
        _ => 0,
    }
}

/// Does `args` (source first) have the proven extension-call layout?
///
/// The first parameter must not be the source again: a chain built from that
/// shape would render `src.Method(src)`, repeating the receiver as its first
/// argument. Such a call is left in its original static form.
fn source_first_layout_ok(args: &[Expr], method: &str) -> bool {
    if args.len() <= min_params_after_source(method) {
        return false;
    }
    if args.len() > 1 && args[1] == args[0] {
        return false;
    }
    true
}

/// Build a `MethodChain` from a proven source-first argument list, extending an
/// existing chain instead of nesting a second receiver.
fn make_chain(source: Expr, method: String, method_args: Vec<Expr>) -> Expr {
    debug_assert!(
        method_args.first() != Some(&source),
        "LINQ chain must never repeat the receiver as its first argument"
    );
    if let Expr::MethodChain {
        receiver,
        mut methods,
    } = source
    {
        methods.push(ChainMethod {
            name: method,
            args: method_args,
        });
        Expr::MethodChain { receiver, methods }
    } else {
        Expr::MethodChain {
            receiver: Box::new(source),
            methods: vec![ChainMethod {
                name: method,
                args: method_args,
            }],
        }
    }
}

/// Run the LINQ chain reconstruction over one method body.
///
/// Returns the number of LINQ call chains rewritten to `MethodChain` expressions.
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

/// Recursively rewrite LINQ call patterns to `MethodChain` within an expression tree.
///
/// Post-order traversal: children are rewritten first, so nested LINQ calls
/// are handled naturally — the innermost call becomes a `MethodChain`, then the
/// outer call sees a `MethodChain` as its source argument and extends the chain.
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
        Expr::Tuple { elements, .. } => {
            for e in elements {
                rewrites += rewrite_expr(e);
            }
        }
        Expr::MethodChain { receiver, methods } => {
            rewrites += rewrite_expr(receiver);
            for m in methods {
                for a in &mut m.args {
                    rewrites += rewrite_expr(a);
                }
            }
        }
        Expr::SwitchExpr { discrim, arms, default_arm } => {
            rewrites += rewrite_expr(discrim);
            for (p, r) in arms {
                rewrites += rewrite_expr(p);
                rewrites += rewrite_expr(r);
            }
            if let Some(d) = default_arm {
                rewrites += rewrite_expr(d);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                rewrites += rewrite_expr(f);
            }
            if let Some(t) = to {
                rewrites += rewrite_expr(t);
            }
        }
        Expr::IndexFromEnd(inner) => {
            rewrites += rewrite_expr(inner);
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
        | Expr::StringInterp { .. } | Expr::TypeOf(_) => {}
    }

    // Now check if THIS expression is a LINQ call we can rewrite.
    rewrites += try_rewrite_linq(expr);

    rewrites
}

/// Try to rewrite a single expression into a `MethodChain`.
///
/// Three proven shapes are recognized:
///
/// 1. **Static dotted call**: `Call { recv: None, name: "Enumerable.Where", args: [source, rest...] }`.
///    The qualifier must be `Enumerable`/`*.Enumerable`; `args[0]` is the
///    source, the remaining arguments keep their order.
///
/// 2. **Static receiver form**: `Call { recv: Some(Var("[System.Linq.]Enumerable")), name: "Where", args: [source, rest...] }`.
///    Same source-first layout; the qualifier expression is not a value.
///
/// 3. **Instance chain**: `Call { recv: Some(inner), name, args }` where the
///    receiver is already a `MethodChain` or another LINQ instance call. A
///    plain receiver is an ordinary instance method (`list.Contains(x)`) and
///    is left alone — folding it buys nothing and can duplicate the receiver
///    when a later pass wraps the call in a null-conditional.
fn try_rewrite_linq(expr: &mut Expr) -> usize {
    // Pattern 1: static `Enumerable.Method(source, ...)`.
    {
        let (ok, method) = match expr {
            Expr::Call {
                recv: None,
                name,
                args,
            } => match static_linq_method(name) {
                Some(m) => (source_first_layout_ok(args, m), m.to_string()),
                None => (false, String::new()),
            },
            _ => (false, String::new()),
        };
        if ok {
            if let Expr::Call { args, .. } = expr {
                let source = args.remove(0);
                let method_args = std::mem::take(args);
                *expr = make_chain(source, method, method_args);
                return 1;
            }
        }
    }

    // Pattern 2: static receiver form `Enumerable.Method(source, ...)` where
    // the renderer kept the owning type as a receiver expression.
    {
        let (ok, method) = match expr {
            Expr::Call {
                recv: Some(recv),
                name,
                args,
            } => match recv.as_ref() {
                Expr::Var(qualifier) if is_enumerable_qualifier(qualifier) => {
                    match linq_method_name(name) {
                        Some(m) => (source_first_layout_ok(args, m), m.to_string()),
                        None => (false, String::new()),
                    }
                }
                _ => (false, String::new()),
            },
            _ => (false, String::new()),
        };
        if ok {
            if let Expr::Call { recv, args, .. } = expr {
                // Drop the type-qualifier "receiver"; it is not evaluated.
                recv.take();
                let source = args.remove(0);
                let method_args = std::mem::take(args);
                *expr = make_chain(source, method, method_args);
                return 1;
            }
        }
    }

    // Pattern 3: instance chain. Only receiver shapes that are already LINQ
    // calls are folded; an ordinary `obj.LinqNamedMethod()` stays untouched.
    {
        let (ok, method) = match expr {
            Expr::Call {
                recv: Some(recv),
                name,
                args,
            } => {
                let method = match linq_method_name(name) {
                    Some(m) => m.to_string(),
                    None => return 0,
                };
                // Never let the first argument repeat the receiver.
                if args.first() == Some(recv.as_ref()) {
                    return 0;
                }
                let chainish = match recv.as_ref() {
                    Expr::MethodChain { .. } => true,
                    Expr::Call {
                        recv: Some(inner_recv),
                        name: inner_name,
                        args: inner_args,
                    } => {
                        linq_method_name(inner_name).is_some()
                            // ... and the inner call itself must not repeat its
                            // own receiver in its first argument.
                            && inner_args.first() != Some(inner_recv.as_ref())
                            && args.first() != Some(inner_recv.as_ref())
                    }
                    _ => false,
                };
                (chainish, method)
            }
            _ => (false, String::new()),
        };
        if ok {
            if let Expr::Call { recv, args, .. } = expr {
                let recv_owned = recv.take().expect("checked Some above");
                let method_args = std::mem::take(args);
                *expr = match *recv_owned {
                    Expr::MethodChain {
                        receiver,
                        mut methods,
                    } => {
                        methods.push(ChainMethod {
                            name: method,
                            args: method_args,
                        });
                        Expr::MethodChain { receiver, methods }
                    }
                    Expr::Call {
                        recv: Some(inner_recv),
                        name: inner_name,
                        args: inner_args,
                    } => {
                        // Post-order normally turns the inner call into a
                        // chain already; this branch covers a plain-receiver
                        // inner call that Pattern 3 intentionally leaves alone.
                        let inner_method = linq_method_name(&inner_name)
                            .expect("pre-checked LINQ name")
                            .to_string();
                        Expr::MethodChain {
                            receiver: inner_recv,
                            methods: vec![
                                ChainMethod {
                                    name: inner_method,
                                    args: inner_args,
                                },
                                ChainMethod {
                                    name: method,
                                    args: method_args,
                                },
                            ],
                        }
                    }
                    _ => unreachable!("pre-checked chain-shaped receiver"),
                };
                return 1;
            }
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    /// Helper: build a static `Enumerable.Method(source, args...)` call.
    fn linq_static(method: &str, source: Expr, args: Vec<Expr>) -> Expr {
        let mut full_args = vec![source];
        full_args.extend(args);
        Expr::Call {
            recv: None,
            name: format!("Enumerable.{method}"),
            args: full_args,
        }
    }

    /// Helper: build a lambda expression `(x) => body`.
    fn lambda(param: &str, body: Expr) -> Expr {
        Expr::Lambda {
            params: vec![param.to_string()],
            body: Box::new(body),
        }
    }

    // -----------------------------------------------------------------------
    // Positive rewrites: static LINQ calls
    // -----------------------------------------------------------------------

    /// 1. Simple Where: `Enumerable.Where(source, pred)` -> `source.Where(pred)`
    #[test]
    fn simple_where() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::Eval(linq_static(
            "Where",
            var("source"),
            vec![pred],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { receiver, methods }) => {
                assert!(matches!(receiver.as_ref(), Expr::Var(name) if name == "source"));
                assert_eq!(methods.len(), 1);
                assert_eq!(methods[0].name, "Where");
                assert_eq!(methods[0].args.len(), 1);
            }
            other => panic!("expected Eval(MethodChain), got {other:?}"),
        }
    }

    /// 2. Chained Where+Select: `Enumerable.Select(Enumerable.Where(source, pred), transform)`
    ///    -> `source.Where(pred).Select(transform)`
    #[test]
    fn chained_where_select() {
        let pred = lambda("x", Expr::Bool(true));
        let transform = lambda("x", var("x"));
        let inner = linq_static("Where", var("source"), vec![pred]);
        let outer = linq_static("Select", inner, vec![transform]);
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "inner Where and outer Select each rewrite");
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { receiver, methods }) => {
                assert!(matches!(receiver.as_ref(), Expr::Var(name) if name == "source"));
                assert_eq!(methods.len(), 2, "Where + Select");
                assert_eq!(methods[0].name, "Where");
                assert_eq!(methods[1].name, "Select");
            }
            other => panic!("expected Eval(MethodChain), got {other:?}"),
        }
    }

    /// 3. Triple chain: Where.Select.OrderBy
    #[test]
    fn triple_chain() {
        let pred = lambda("x", Expr::Bool(true));
        let transform = lambda("x", var("x"));
        let key = lambda("x", var("x"));
        let inner = linq_static("Where", var("source"), vec![pred]);
        let mid = linq_static("Select", inner, vec![transform]);
        let outer = linq_static("OrderBy", mid, vec![key]);
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 3, "each nested LINQ call is rewritten individually");
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { receiver, methods }) => {
                assert!(matches!(receiver.as_ref(), Expr::Var(name) if name == "source"));
                assert_eq!(methods.len(), 3, "Where + Select + OrderBy");
                assert_eq!(methods[0].name, "Where");
                assert_eq!(methods[1].name, "Select");
                assert_eq!(methods[2].name, "OrderBy");
            }
            other => panic!("expected Eval(MethodChain), got {other:?}"),
        }
    }

    /// 4. LINQ in assignment: `result = Enumerable.Where(source, pred)`
    #[test]
    fn linq_in_assignment() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::Assign(
            "result".into(),
            None,
            linq_static("Where", var("source"), vec![pred]),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(target, _, Expr::MethodChain { .. }) => {
                assert_eq!(target, "result");
            }
            other => panic!("expected Assign(_, _, MethodChain), got {other:?}"),
        }
    }

    /// 5. LINQ in return: `return Enumerable.Count(source)`
    #[test]
    fn linq_in_return() {
        let mut stmts = vec![Stmt::Return(Some(linq_static(
            "Count",
            var("items"),
            vec![],
        )))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Return(Some(Expr::MethodChain { methods, .. })) => {
                assert_eq!(methods.len(), 1);
                assert_eq!(methods[0].name, "Count");
            }
            other => panic!("expected Return(Some(MethodChain)), got {other:?}"),
        }
    }

    /// 6. Fully qualified name: `System.Linq.Enumerable.Where(source, pred)`
    #[test]
    fn fully_qualified_enumerable() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "System.Linq.Enumerable.Where".into(),
            args: vec![var("source"), pred],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { methods, .. }) => {
                assert_eq!(methods[0].name, "Where");
            }
            other => panic!("expected MethodChain, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Positive rewrites: instance method chains
    // -----------------------------------------------------------------------

    /// 7. Instance chain: `source.Where(pred).Select(transform)`
    #[test]
    fn instance_chain() {
        let pred = lambda("x", Expr::Bool(true));
        let transform = lambda("x", var("x"));
        let inner = Expr::Call {
            recv: Some(Box::new(var("source"))),
            name: "Where".into(),
            args: vec![pred],
        };
        let outer = Expr::Call {
            recv: Some(Box::new(inner)),
            name: "Select".into(),
            args: vec![transform],
        };
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(
            n, 1,
            "only the outer call folds; a plain-receiver instance call stays a Call"
        );
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { receiver, methods }) => {
                assert!(matches!(receiver.as_ref(), Expr::Var(name) if name == "source"));
                assert_eq!(methods.len(), 2);
                assert_eq!(methods[0].name, "Where");
                assert_eq!(methods[1].name, "Select");
            }
            other => panic!("expected MethodChain, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Non-matches (safety tests)
    // -----------------------------------------------------------------------

    /// 8. Non-LINQ static call — must NOT rewrite.
    #[test]
    fn non_linq_static_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Console.WriteLine".into(),
            args: vec![Expr::Str("hello".into())],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "Console.WriteLine"
        ));
    }

    /// 9. Enumerable method with no args — must NOT rewrite (no source).
    #[test]
    fn enumerable_no_args_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Enumerable.Empty".into(),
            args: vec![],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    /// 10. Non-Enumerable qualifier — must NOT rewrite even if method name matches.
    #[test]
    fn non_enumerable_qualifier_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "MyList.Where".into(),
            args: vec![var("source"), lambda("x", Expr::Bool(true))],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "non-Enumerable qualifier must not rewrite");
    }

    /// 11. Instance call with non-LINQ method name — must NOT rewrite.
    #[test]
    fn instance_non_linq_not_rewritten() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("obj"))),
            name: "ToString".into(),
            args: vec![],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "ToString"
        ));
    }

    /// 12. LINQ in nested expression (inside a BinOp).
    #[test]
    fn linq_nested_in_binop() {
        let pred = lambda("x", Expr::Bool(true));
        let linq = linq_static("Where", var("source"), vec![pred]);
        let combined = Expr::Bin(
            super::super::super::csharp::BinOp::AndAlso,
            Box::new(Expr::Bool(true)),
            Box::new(linq),
        );
        let mut stmts = vec![Stmt::Eval(combined)];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }

    /// 13. LINQ in if condition.
    #[test]
    fn linq_in_if_condition() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::If {
            cond: linq_static("Any", var("source"), vec![pred]),
            then_body: vec![Stmt::Eval(var("x"))],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If {
                cond: Expr::MethodChain { .. },
                ..
            } => {}
            other => panic!("expected If with MethodChain condition, got {other:?}"),
        }
    }

    /// 14. Rendering test: verify the chain renders correctly.
    #[test]
    fn render_method_chain() {
        let chain = Expr::MethodChain {
            receiver: Box::new(var("items")),
            methods: vec![
                ChainMethod {
                    name: "Where".into(),
                    args: vec![lambda("x", Expr::Cmp(
                        super::super::super::csharp::CmpKind::Gt,
                        Box::new(var("x")),
                        Box::new(Expr::Int(0)),
                    ))],
                },
                ChainMethod {
                    name: "Select".into(),
                    args: vec![lambda("x", var("x"))],
                },
            ],
        };
        let mut s = String::new();
        chain.render(&mut s);
        assert!(
            s.contains("items.Where("),
            "rendering should contain items.Where(, got: {s}"
        );
        assert!(
            s.contains(").Select("),
            "rendering should contain ).Select(, got: {s}"
        );
    }

    /// 15. Multiple independent LINQ calls in the same method.
    #[test]
    fn multiple_independent_chains() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![
            Stmt::Eval(linq_static("Where", var("a"), vec![pred.clone()])),
            Stmt::Eval(linq_static("Select", var("b"), vec![lambda("x", var("x"))])),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "both independent chains should be rewritten");
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::MethodChain { .. })));
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::MethodChain { .. })));
    }

    /// 16. LINQ chain inside a loop body.
    #[test]
    fn linq_in_loop_body() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Eval(linq_static(
                "Where",
                var("source"),
                vec![pred],
            ))],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While { body, .. } => {
                assert!(matches!(&body[0], Stmt::Eval(Expr::MethodChain { .. })));
            }
            other => panic!("expected While, got {other:?}"),
        }
    }

    /// 17. LINQ chain inside a try body.
    #[test]
    fn linq_in_try_body() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::Try {
            try_body: vec![Stmt::Eval(linq_static(
                "Where",
                var("source"),
                vec![pred],
            ))],
            catches: vec![],
            finally_body: None,
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
    }

    /// 18. No LINQ calls — no-op.
    #[test]
    fn no_linq_calls_is_noop() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::Int(0)),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Foo".into(),
                args: vec![],
            }),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
    }

    /// 19. is_linq_method helper tests.
    #[test]
    fn is_linq_method_helper() {
        assert!(is_linq_method("Where"));
        assert!(is_linq_method("Select"));
        assert!(is_linq_method("Enumerable.Where"));
        assert!(is_linq_method("System.Linq.Enumerable.Where"));
        assert!(!is_linq_method("ToString"));
        assert!(!is_linq_method("GetEnumerator"));
        assert!(!is_linq_method("Add"));
    }

    /// 20. LINQ in condition of a while loop.
    #[test]
    fn linq_in_while_condition() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::While {
            cond: linq_static("Any", var("items"), vec![pred]),
            body: vec![Stmt::Break],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::While {
                cond: Expr::MethodChain { .. },
                ..
            } => {}
            other => panic!("expected While with MethodChain condition, got {other:?}"),
        }
    }

    /// Helper: build a static LINQ call with a generic instantiation suffix.
    fn linq_static_generic(method: &str, instantiation: &str, source: Expr, args: Vec<Expr>) -> Expr {
        let mut full_args = vec![source];
        full_args.extend(args);
        Expr::Call {
            recv: None,
            name: format!("Enumerable.{method}<{instantiation}>"),
            args: full_args,
        }
    }

    /// 21. Generic-instantiated static call folds; the suffix is not part of
    ///     the rendered chain method name.
    #[test]
    fn generic_static_name_is_stripped() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::Eval(linq_static_generic(
            "Where",
            "Foo",
            var("source"),
            vec![pred],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { methods, .. }) => {
                assert_eq!(methods[0].name, "Where");
            }
            other => panic!("expected MethodChain, got {other:?}"),
        }
    }

    /// 22. Fully-qualified generic names fold too (`System.Linq.Enumerable.Select<A, B>`).
    #[test]
    fn fully_qualified_generic_name() {
        let transform = lambda("x", var("x"));
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "System.Linq.Enumerable.Select<object, long>".into(),
            args: vec![var("source"), transform],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { methods, .. }) => {
                assert_eq!(methods[0].name, "Select");
                assert_eq!(methods[0].args.len(), 1);
            }
            other => panic!("expected MethodChain, got {other:?}"),
        }
    }

    /// 23. `Enumerable.Range` / `Repeat` are static factories, not extension
    ///     methods: folding would invert the call (`0.Range(10)`).
    #[test]
    fn static_factories_not_folded() {
        for name in ["Enumerable.Range", "Enumerable.Repeat", "Enumerable.Empty"] {
            let mut stmts = vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: name.into(),
                args: vec![Expr::Int(0), Expr::Int(10)],
            })];
            let n = apply(&mut stmts);
            assert_eq!(n, 0, "{name} must not fold");
            assert!(matches!(&stmts[0], Stmt::Eval(Expr::Call { .. })));
        }
    }

    /// 24. A static call whose first parameter repeats the source must be left
    ///     alone — a chain would render `src.Contains(src)` from an unproven
    ///     layout.
    #[test]
    fn duplicated_source_argument_not_folded() {
        let mut stmts = vec![Stmt::Eval(linq_static(
            "Contains",
            var("src"),
            vec![var("src")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        match &stmts[0] {
            Stmt::Eval(Expr::Call { name, args, .. }) => {
                assert_eq!(name, "Enumerable.Contains");
                assert_eq!(args.len(), 2);
            }
            other => panic!("plain static call expected, got {other:?}"),
        }
    }

    /// 25. Ordinary instance call: `list.Contains(x)` is not LINQ source-first
    ///     syntax and must stay a Call (this is what caused `v?.v.Contains(x)`
    ///     once null-conditionals wrapped a chain access).
    #[test]
    fn plain_instance_call_not_folded() {
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("list"))),
            name: "Contains".into(),
            args: vec![var("x")],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert!(matches!(
            &stmts[0],
            Stmt::Eval(Expr::Call { name, .. }) if name == "Contains"
        ));
    }

    /// 26. Argument order after the source is preserved verbatim.
    #[test]
    fn argument_order_preserved() {
        let mut stmts = vec![Stmt::Eval(linq_static(
            "Join",
            var("outer"),
            vec![var("inner"), var("ok"), var("ik"), var("result")],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { methods, .. }) => {
                assert_eq!(methods[0].name, "Join");
                let names: Vec<_> = methods[0]
                    .args
                    .iter()
                    .map(|a| match a {
                        Expr::Var(n) => n.as_str(),
                        _ => panic!("expected Var"),
                    })
                    .collect();
                assert_eq!(names, ["inner", "ok", "ik", "result"]);
            }
            other => panic!("expected MethodChain, got {other:?}"),
        }
    }

    /// 27. A call missing the minimum parameter count is not the shape we know.
    #[test]
    fn missing_required_parameters_not_folded() {
        let mut stmts = vec![Stmt::Eval(linq_static(
            "Select",
            var("source"),
            vec![],
        ))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "Select needs at least one parameter");
    }

    /// 28. Receiver form `Enumerable.Where(source, pred)` folds like the dotted form.
    #[test]
    fn enumerable_receiver_form() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("System.Linq.Enumerable"))),
            name: "Where".into(),
            args: vec![var("source"), pred],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { receiver, methods }) => {
                assert!(matches!(receiver.as_ref(), Expr::Var(n) if n == "source"));
                assert_eq!(methods[0].name, "Where");
            }
            other => panic!("expected MethodChain, got {other:?}"),
        }
    }

    /// 29. A second application is a no-op (idempotent).
    #[test]
    fn second_run_is_noop() {
        let pred = lambda("x", Expr::Bool(true));
        let mut stmts = vec![Stmt::Eval(linq_static(
            "Where",
            var("source"),
            vec![pred],
        ))];
        assert_eq!(apply(&mut stmts), 1);
        let snapshot = format!("{:?}", stmts);
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(format!("{:?}", stmts), snapshot);
    }

    /// 30. Regression for `v159?.v159.Contains(x)`: the access side of a
    ///     null-conditional is an ordinary instance call, so linq_recon must
    ///     not wrap it in a chain that repeats the `?.` receiver.
    #[test]
    fn null_conditional_access_stays_instance_call() {
        let access = Expr::Call {
            recv: Some(Box::new(var("v159"))),
            name: "Contains".into(),
            args: vec![var("x")],
        };
        let mut stmts = vec![Stmt::Eval(Expr::NullConditional {
            receiver: Box::new(var("v159")),
            access: Box::new(access),
        })];
        assert_eq!(apply(&mut stmts), 0);
        let mut out = String::new();
        match &stmts[0] {
            Stmt::Eval(e) => e.render(&mut out),
            other => panic!("expected Eval, got {other:?}"),
        }
        assert_eq!(out, "v159?.Contains(x)");
    }

    /// 31. Generic instance method names are stripped when extending a chain.
    #[test]
    fn chain_method_generic_name_stripped() {
        let pred = lambda("x", Expr::Bool(true));
        let inner = linq_static_generic("Where", "Foo", var("source"), vec![pred.clone()]);
        let outer = Expr::Call {
            recv: Some(Box::new(inner)),
            name: "Select<Foo, Bar>".into(),
            args: vec![lambda("x", var("x"))],
        };
        let mut stmts = vec![Stmt::Eval(outer)];
        let n = apply(&mut stmts);
        assert_eq!(n, 2, "inner and outer generic calls both fold");
        match &stmts[0] {
            Stmt::Eval(Expr::MethodChain { methods, .. }) => {
                assert_eq!(methods.len(), 2);
                assert_eq!(methods[0].name, "Where");
                assert_eq!(methods[1].name, "Select");
            }
            other => panic!("expected MethodChain, got {other:?}"),
        }
    }
}
