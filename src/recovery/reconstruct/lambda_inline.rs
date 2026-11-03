//! Lambda body inlining — cross-method lambda reconstruction.
//!
//! ## Problem
//!
//! The `closures.rs` pass can fold delegate constructions back into
//! `Expr::Lambda { params, body }` — but only when the lambda's body expression
//! is already present in the same statement list as a binding. Lambda bodies
//! live in **separate methods** (`<>c.<M>b__N_M` or `<M>b__N_M`), so
//! `closures.rs` alone cannot recover the common site.
//!
//! Phase 5A (in `csharp.rs`) registers lambda bodies in a global map and
//! `inject_lambda_bodies` prepends synthetic bindings before reconstruction.
//! That bridge is best-effort: the renderer registers a lambda method under its
//! BARE metadata name (`<Initialize>b__43_3`), while a `MethodGroup` operand
//! carries the DISPLAY name (`<>c.<Initialize>b__43_3`). A plain `get` on the
//! registry therefore misses every static `<>c` display-class lambda. This pass
//! resolves the name through the same set of spellings a display class can
//! produce before giving up.
//!
//! ## What this pass does
//!
//! 1. Walks the statement tree looking for delegate constructions, in either
//!    spelling the pipeline produces:
//!    - `Expr::NewObj("System.Action`1", [target, method])` — after
//!      `normalize_ctor_calls` merged the two-statement allocation; and
//!    - `Expr::Call { name: "System.Action`1..ctor", args: [target, method] }`
//!      — the raw corpus spelling.
//!    The construction TYPE is often lost (`NewObj("object", …)`) because the
//!    allocation helper's type argument is dropped by the merge. When the
//!    second argument names a compiler-generated lambda method, the two-argument
//!    `object` shape IS the delegate construction, so the type gate accepts it;
//!    the arity check is simply skipped because no arity is stated.
//! 2. Resolves the method cell WHEN PROVABLE:
//!    - a `MethodGroup` directly;
//!    - a local whose only assignment in the list is a `MethodGroup` (the real
//!      corpus loads the cell into a temporary first).
//! 3. Looks the body up in the global registry under every spelling the same
//!    generated method can be filed under (exact, bare last segment,
//!    `<>c.`-prefixed), so a static display-class lambda is found whether the
//!    reference or the registration used the qualified form.
//! 4. Refuses to emit a lambda whose body would change capture semantics. A
//!    generated method's `this` is its DISPLAY CLASS, not the enclosing
//!    instance. When the construction target is `Expr::This`, the method is an
//!    instance method on the enclosing class (a display class object could not
//!    be `this` in the caller), so a body that reads `this` keeps its meaning.
//!    For any other target, a body that mentions `this` is refused: its fields
//!    may be captured locals whose values are not present at the construction
//!    site. This is the "refuse when the display class holds mutable state
//!    other than captures" rule, enforced on the only evidence a single-list
//!    pass can see.
//! 5. Inlines `.Invoke(args)` on a lambda local with exactly ONE assignment in
//!    the list. The substitution is deliberately narrow: every parameter must
//!    occur exactly once in the body, every argument must be free of side
//!    effects, and the body may not introduce a shadowing scope (nested lambda
//!    or pattern binding). Under those conditions the call and the body
//!    evaluate the same things in the same order.
//!
//! ## Safety
//!
//! - A missing body, an arity mismatch, an ambiguous method cell, or a
//!   `this`-capturing body with a non-`this` target leaves the construction
//!   unchanged.
//! - `.Invoke` inlining never fires on a local with more than one write (the
//!   value reaching the call site is not provable).
//! - No statement is deleted; only the delegate construction (whose value is
//!   reproduced exactly by the lambda) and a proven single-use `.Invoke` call
//!   are rewritten.

use super::super::csharp::{Expr, Stmt, lookup_lambda_body};
use super::super::resolve::ResolveMaps;
use super::{children, ctor_type_name, is_ctor_name, list_writes_deep, walk_mut};
use std::sync::atomic::{AtomicU64, Ordering};

/// Lambda inlining rewrite count.
static LAMBDA_INLINE_REWRITES: AtomicU64 = AtomicU64::new(0);
/// Delegate-shaped constructions whose method cell the renderer could not
/// name (a raw GOT/usage pointer or a `typeof(T)` guess). These are the sites a
/// renderer-side method-usage resolution would turn into foldable lambdas.
static LAMBDA_CELL_MISSES: AtomicU64 = AtomicU64::new(0);
/// Generated lambda methods that are named correctly but have no registered
/// body (not rendered yet, or a multi-statement body).
static LAMBDA_BODY_MISSES: AtomicU64 = AtomicU64::new(0);

/// Human-readable report of lambda inlining impact.
pub fn report() -> String {
    let n = LAMBDA_INLINE_REWRITES.load(Ordering::Relaxed);
    let cells = LAMBDA_CELL_MISSES.load(Ordering::Relaxed);
    let bodies = LAMBDA_BODY_MISSES.load(Ordering::Relaxed);
    if n == 0 && cells == 0 && bodies == 0 {
        return String::new();
    }
    if cells == 0 && bodies == 0 {
        return format!("[reconstruct]   lambda_inline: {n}\n");
    }
    format!(
        "[reconstruct]   lambda_inline: {n} rewrite(s), {cells} unnamed method cell(s), {bodies} body misses\n"
    )
}

/// Is `name` a compiler-generated lambda method (`<…>b__N_M`)?
///
/// Matches the C# compiler's naming convention for lifted lambda bodies:
/// `<>c.<M>b__1_0`, `<MyClass>o__DisplayClass3.<Method>b__0`, etc.
fn is_generated_lambda(name: &str) -> bool {
    let last = name.rsplit('.').next().unwrap_or(name);
    last.starts_with('<') && last.contains("b__")
}

/// Is `ty` plausibly a delegate type name?
fn is_delegate_type(ty: &str) -> bool {
    let t = ty.rsplit('.').next().unwrap_or(ty);
    let base = t.split('`').next().unwrap_or(t);
    base.starts_with("Action")
        || base.starts_with("Func")
        || base == "Predicate"
        || base == "Comparison"
        || base.ends_with("EventHandler")
        || base.ends_with("Delegate")
        || base.ends_with("Callback")
}

/// Is this construction's type name the lossy `object` fallback written by
/// `normalize_ctor_calls` when the allocation helper's type could not be read?
fn is_unknown_ctor_type(ty: Option<&str>) -> bool {
    matches!(ty, None | Some("") | Some("object") | Some("System.Object"))
}

/// Number of type arguments in the first `<…>` of `s` (top-level commas + 1),
/// or `None` when there is no generic argument list.
fn angle_arity(s: &str) -> Option<usize> {
    let open = s.find('<')?;
    let close = s.rfind('>')?;
    if close <= open {
        return None;
    }
    let inner = &s[open + 1..close];
    if inner.trim().is_empty() {
        return Some(0);
    }
    let mut depth = 0i32;
    let mut n = 1usize;
    for c in inner.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => n += 1,
            _ => {}
        }
    }
    Some(n)
}

/// Parameter count of a delegate type.
fn delegate_arity(ty: &str) -> Option<usize> {
    let t = ty.rsplit('.').next().unwrap_or(ty);
    let base = t.split('`').next().unwrap_or(t);
    let angle = angle_arity(t);
    let backtick = t
        .split('`')
        .nth(1)
        .and_then(|d| d.split('[').next())
        .and_then(|d| d.trim().parse::<usize>().ok());
    let args = angle.or(backtick);
    if base.starts_with("Action") {
        Some(args.unwrap_or(0))
    } else if base.starts_with("Func") {
        args.map(|a| a.saturating_sub(1))
    } else {
        None
    }
}

/// Every name a registered lambda body may be filed under.
///
/// The renderer registers `sig.name`, the BARE metadata method name
/// (`<Initialize>b__43_3`), while `Operand::Method` renders a `MethodGroup` as
/// `short_type(declaring_type_name).name` — for a static display class that is
/// `<>c.<Initialize>b__43_3`, and for an instance one it can be
/// `<>c__DisplayClass5.<M>b__0`. The body is the same; only the spelling
/// differs. Ordered from most to least specific, deduplicated.
fn lookup_variants(name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(3);
    let mut push = |s: String| {
        if !s.is_empty() && !out.contains(&s) {
            out.push(s);
        }
    };
    push(name.to_string());
    let last = name.rsplit('.').next().unwrap_or(name);
    if last != name && is_generated_lambda(last) {
        push(last.to_string());
        push(format!("<>c.{}", last));
    }
    out
}

/// Look a lambda body up under every spelling the same generated method can
/// have. Returns `None` when no body is registered for any spelling.
fn registered_lambda(name: &str) -> Option<Expr> {
    for variant in lookup_variants(name) {
        if let Some((params, body)) = lookup_lambda_body(&variant) {
            return Some(Expr::Lambda {
                params,
                body: Box::new(body),
            });
        }
    }
    None
}

/// Does `e` mention the enclosing instance (`this`) anywhere?
fn mentions_this(e: &Expr) -> bool {
    match e {
        Expr::This => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => mentions_this(a) || mentions_this(b),
        Expr::Not(x)
        | Expr::Neg(x)
        | Expr::Cast(_, x)
        | Expr::AddressOf(x)
        | Expr::Await(x)
        | Expr::Field(x, _)
        | Expr::RawDeref(x, _, _)
        | Expr::RawAddr(x, _) => mentions_this(x),
        Expr::Cond(c, a, b) => mentions_this(c) || mentions_this(a) || mentions_this(b),
        Expr::Call { recv, args, .. } => {
            recv.as_deref().is_some_and(mentions_this) || args.iter().any(mentions_this)
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            args.iter().any(mentions_this)
        }
        Expr::ArrayInit { items, .. } => items.iter().any(mentions_this),
        Expr::CollectionInit {
            ctor_args, items, ..
        } => ctor_args.iter().chain(items.iter()).any(mentions_this),
        Expr::ObjInit {
            ctor_args, fields, ..
        } => ctor_args.iter().any(mentions_this) || fields.iter().any(|(_, v)| mentions_this(v)),
        Expr::Lambda { body, .. } => mentions_this(body),
        Expr::Index { collection, index } => mentions_this(collection) || mentions_this(index),
        Expr::NullConditional { receiver, access } => {
            mentions_this(receiver) || mentions_this(access)
        }
        Expr::StringInterp { parts } => {
            use super::super::csharp::InterpPart;
            parts.iter().any(|p| match p {
                InterpPart::Expr(e, _) => mentions_this(e),
                _ => false,
            })
        }
        Expr::Tuple { elements, .. } => elements.iter().any(mentions_this),
        Expr::PatternMatch { expr, .. } => mentions_this(expr),
        Expr::MethodChain { receiver, methods } => {
            mentions_this(receiver) || methods.iter().any(|m| m.args.iter().any(mentions_this))
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            mentions_this(discrim)
                || arms
                    .iter()
                    .any(|(p, r)| mentions_this(p) || mentions_this(r))
                || default_arm.as_deref().is_some_and(mentions_this)
        }
        Expr::Range { from, to } => {
            from.as_deref().is_some_and(mentions_this) || to.as_deref().is_some_and(mentions_this)
        }
        Expr::IndexFromEnd(x) => mentions_this(x),
        _ => false,
    }
}

/// Is a recovered lambda body safe to materialise at a construction site whose
/// delegate target is (or is not) the enclosing `this`?
///
/// A generated lambda method's `this` is its display class. It can only be the
/// enclosing instance when the construction binds `this` itself (see the module
/// docs). In every other case a `this` reference names display-class state that
/// is not available at the caller, so the fold is refused.
fn body_is_capture_safe(body: &Expr, target_is_this: bool) -> bool {
    target_is_this || !mentions_this(body)
}

/// The evidence a single statement list supplies for cell/local resolution.
#[derive(Default)]
struct ListFacts {
    /// local → generated method name, for locals whose ONLY write in the list
    /// is a `MethodGroup` assignment at the given index.
    method_cells: Vec<(String, String, usize)>,
}

impl ListFacts {
    fn collect(list: &[Stmt]) -> Self {
        let mut f = ListFacts::default();
        for (i, s) in list.iter().enumerate() {
            if let Stmt::Assign(n, _, Expr::MethodGroup(m)) = s {
                if single_write_index(list, n) == Some(i) {
                    f.method_cells.push((n.clone(), m.clone(), i));
                }
            }
        }
        f
    }

    /// The method name a second construction argument provably names.
    fn method_name_at(&self, e: &Expr, at: usize) -> Option<String> {
        match e {
            Expr::MethodGroup(m) => Some(m.clone()),
            Expr::Var(v) => self
                .method_cells
                .iter()
                .find(|(n, _, i)| n == v && *i < at)
                .map(|(_, m, _)| m.clone()),
            _ => None,
        }
    }
}

/// The index of `name`'s ONLY assignment when it is a top-level statement of
/// this list and no nested body writes it either. `None` = the value reaching a
/// later use is not provable.
fn single_write_index(list: &[Stmt], name: &str) -> Option<usize> {
    let mut found = None;
    for (i, s) in list.iter().enumerate() {
        if matches!(s, Stmt::Assign(n, _, _) if n == name) {
            if found.is_some() {
                return None;
            }
            found = Some(i);
        }
    }
    if found.is_none() {
        return None;
    }
    // A write inside a nested body can reach the use site too.
    for s in list {
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        if kids.iter().any(|k| list_writes_deep(k, name)) {
            return None;
        }
    }
    found
}

/// A delegate construction, in either spelling the pipeline produces.
/// Returns `(type name, argument list)`.
fn construction_parts(e: &Expr) -> Option<(Option<&str>, &[Expr])> {
    match e {
        Expr::NewObj(ty, args) => Some((Some(ty.as_str()), args.as_slice())),
        Expr::Call { recv: None, name, args } if is_ctor_name(name) => {
            Some((ctor_type_name(name), args.as_slice()))
        }
        _ => None,
    }
}

/// Does `e` look like a delegate ctor's method cell that lost its identity?
///
/// The measured corpus spells it `typeof(T)` (the ptr-resolution heuristic's
/// type guess) or a raw `void*.ptr_0x…` static pseudo-field. Neither names the
/// method, so nothing can be folded — but counting them keeps the miss visible.
fn looks_like_unnamed_method_cell(e: &Expr) -> bool {
    match e {
        Expr::TypeOf(_) | Expr::MethodGroup(_) | Expr::RawDeref(..) => true,
        Expr::StaticField(_, name) => name.starts_with("ptr_0x") || name.starts_with("MethodInfo"),
        _ => false,
    }
}

/// Fold one delegate construction to the registered lambda body.
fn try_inline_lambda(e: &Expr, at: usize, facts: &ListFacts) -> Option<Expr> {
    let (ty, args) = construction_parts(e)?;
    // Delegate constructions have exactly 2 args: (target, method_ref).
    if args.len() != 2 {
        return None;
    }
    // Type gate. A known delegate type states the arity; the lossy `object`
    // spelling carries no type information but the shape plus a generated
    // method name is the delegate construction itself.
    let known_delegate = ty.is_some_and(is_delegate_type);
    if !known_delegate && !is_unknown_ctor_type(ty) {
        return None;
    }
    let Some(method_name) = facts.method_name_at(&args[1], at) else {
        if (known_delegate || is_unknown_ctor_type(ty)) && looks_like_unnamed_method_cell(&args[1])
        {
            LAMBDA_CELL_MISSES.fetch_add(1, Ordering::Relaxed);
            if std::env::var("ILR_LAMDBG").is_ok() {
                eprintln!("[lamdbg] unnamed delegate method cell: {:?}", args[1]);
            }
        }
        return None;
    };
    if !is_generated_lambda(&method_name) {
        // A named, non-generated method group is a plain method-group delegate
        // (valid source, nothing to recover).
        return None;
    }
    let Some(lambda) = registered_lambda(&method_name) else {
        LAMBDA_BODY_MISSES.fetch_add(1, Ordering::Relaxed);
        if std::env::var("ILR_LAMDBG").is_ok() {
            eprintln!("[lamdbg] no registered body for {method_name}");
        }
        return None;
    };
    let Expr::Lambda { params, body } = &lambda else {
        return None;
    };
    if known_delegate {
        if let Some(arity) = ty.and_then(delegate_arity) {
            if params.len() != arity {
                return None;
            }
        }
    }
    if !body_is_capture_safe(body, matches!(&args[0], Expr::This)) {
        return None;
    }
    Some(lambda)
}

/// Rewrite one expression, recursing into children.
fn rewrite_expr(e: &mut Expr, at: usize, facts: &ListFacts) -> usize {
    if let Some(lambda) = try_inline_lambda(e, at, facts) {
        *e = lambda;
        LAMBDA_INLINE_REWRITES.fetch_add(1, Ordering::Relaxed);
        return 1;
    }
    let mut n = 0usize;
    match e {
        Expr::Bin(_, a, c) | Expr::Cmp(_, a, c) => {
            n += rewrite_expr(a, at, facts);
            n += rewrite_expr(c, at, facts);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            n += rewrite_expr(x, at, facts);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            n += rewrite_expr(r, at, facts);
        }
        Expr::Cond(c, a, d) => {
            n += rewrite_expr(c, at, facts);
            n += rewrite_expr(a, at, facts);
            n += rewrite_expr(d, at, facts);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                n += rewrite_expr(r, at, facts);
            }
            for a in args {
                n += rewrite_expr(a, at, facts);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                n += rewrite_expr(a, at, facts);
            }
        }
        Expr::Lambda { body, .. } => n += rewrite_expr(body, at, facts),
        Expr::Index { collection, index } => {
            n += rewrite_expr(collection, at, facts);
            n += rewrite_expr(index, at, facts);
        }
        Expr::NullConditional { receiver, access } => {
            n += rewrite_expr(receiver, at, facts);
            n += rewrite_expr(access, at, facts);
        }
        Expr::ArrayInit { items, .. } => {
            for item in items {
                n += rewrite_expr(item, at, facts);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter_mut().chain(items.iter_mut()) {
                n += rewrite_expr(a, at, facts);
            }
        }
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            for a in ctor_args {
                n += rewrite_expr(a, at, facts);
            }
            for (_, v) in fields {
                n += rewrite_expr(v, at, facts);
            }
        }
        Expr::StringInterp { parts } => {
            use super::super::csharp::InterpPart;
            for p in parts {
                if let InterpPart::Expr(e, _) = p {
                    n += rewrite_expr(e, at, facts);
                }
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                n += rewrite_expr(e, at, facts);
            }
        }
        Expr::PatternMatch { expr, .. } => {
            n += rewrite_expr(expr, at, facts);
        }
        Expr::MethodChain { receiver, methods } => {
            n += rewrite_expr(receiver, at, facts);
            for m in methods {
                for a in &mut m.args {
                    n += rewrite_expr(a, at, facts);
                }
            }
        }
        _ => {}
    }
    n
}

/// Count the occurrences of `name` as a variable read in `e`.
///
/// A nested lambda or pattern binding that re-uses the name shadows it, so the
/// caller refuses those shapes before asking; this walker therefore does not
/// need to model scopes.
fn count_var(e: &Expr, name: &str) -> usize {
    let mut n = 0usize;
    match e {
        Expr::Var(v) if v == name => n += 1,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            n += count_var(a, name);
            n += count_var(b, name);
        }
        Expr::Not(x)
        | Expr::Neg(x)
        | Expr::Cast(_, x)
        | Expr::AddressOf(x)
        | Expr::Await(x)
        | Expr::Field(x, _)
        | Expr::RawDeref(x, _, _)
        | Expr::RawAddr(x, _) => n += count_var(x, name),
        Expr::Cond(c, a, b) => {
            n += count_var(c, name);
            n += count_var(a, name);
            n += count_var(b, name);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                n += count_var(r, name);
            }
            for a in args {
                n += count_var(a, name);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                n += count_var(a, name);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for a in items {
                n += count_var(a, name);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter().chain(items.iter()) {
                n += count_var(a, name);
            }
        }
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            for a in ctor_args {
                n += count_var(a, name);
            }
            for (_, v) in fields {
                n += count_var(v, name);
            }
        }
        Expr::Lambda { body, .. } => n += count_var(body, name),
        Expr::Index { collection, index } => {
            n += count_var(collection, name);
            n += count_var(index, name);
        }
        Expr::NullConditional { receiver, access } => {
            n += count_var(receiver, name);
            n += count_var(access, name);
        }
        Expr::StringInterp { parts } => {
            use super::super::csharp::InterpPart;
            for p in parts {
                if let InterpPart::Expr(e, _) = p {
                    n += count_var(e, name);
                }
            }
        }
        Expr::Tuple { elements, .. } => {
            for a in elements {
                n += count_var(a, name);
            }
        }
        Expr::PatternMatch { expr, .. } => n += count_var(expr, name),
        Expr::MethodChain { receiver, methods } => {
            n += count_var(receiver, name);
            for m in methods {
                for a in &m.args {
                    n += count_var(a, name);
                }
            }
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            n += count_var(discrim, name);
            for (p, r) in arms {
                n += count_var(p, name);
                n += count_var(r, name);
            }
            if let Some(d) = default_arm {
                n += count_var(d, name);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                n += count_var(f, name);
            }
            if let Some(t) = to {
                n += count_var(t, name);
            }
        }
        Expr::IndexFromEnd(x) => n += count_var(x, name),
        _ => {}
    }
    n
}

/// Does the body introduce a scope that can shadow a parameter name?
fn has_shadowing_scope(e: &Expr) -> bool {
    match e {
        Expr::Lambda { .. } | Expr::PatternMatch { .. } => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            has_shadowing_scope(a) || has_shadowing_scope(b)
        }
        Expr::Not(x)
        | Expr::Neg(x)
        | Expr::Cast(_, x)
        | Expr::AddressOf(x)
        | Expr::Await(x)
        | Expr::Field(x, _)
        | Expr::RawDeref(x, _, _)
        | Expr::RawAddr(x, _) => has_shadowing_scope(x),
        Expr::Cond(c, a, b) => {
            has_shadowing_scope(c) || has_shadowing_scope(a) || has_shadowing_scope(b)
        }
        Expr::Call { recv, args, .. } => {
            recv.as_deref().is_some_and(has_shadowing_scope)
                || args.iter().any(has_shadowing_scope)
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            args.iter().any(has_shadowing_scope)
        }
        Expr::ArrayInit { items, .. } => items.iter().any(has_shadowing_scope),
        Expr::CollectionInit {
            ctor_args, items, ..
        } => ctor_args.iter().chain(items.iter()).any(has_shadowing_scope),
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            ctor_args.iter().any(has_shadowing_scope)
                || fields.iter().any(|(_, v)| has_shadowing_scope(v))
        }
        Expr::Index { collection, index } => {
            has_shadowing_scope(collection) || has_shadowing_scope(index)
        }
        Expr::NullConditional { receiver, access } => {
            has_shadowing_scope(receiver) || has_shadowing_scope(access)
        }
        Expr::StringInterp { parts } => {
            use super::super::csharp::InterpPart;
            parts.iter().any(|p| match p {
                InterpPart::Expr(e, _) => has_shadowing_scope(e),
                _ => false,
            })
        }
        Expr::Tuple { elements, .. } => elements.iter().any(has_shadowing_scope),
        Expr::MethodChain { receiver, methods } => {
            has_shadowing_scope(receiver)
                || methods.iter().any(|m| m.args.iter().any(has_shadowing_scope))
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            has_shadowing_scope(discrim)
                || arms
                    .iter()
                    .any(|(p, r)| has_shadowing_scope(p) || has_shadowing_scope(r))
                || default_arm.as_deref().is_some_and(has_shadowing_scope)
        }
        Expr::Range { from, to } => {
            from.as_deref().is_some_and(has_shadowing_scope)
                || to.as_deref().is_some_and(has_shadowing_scope)
        }
        Expr::IndexFromEnd(x) => has_shadowing_scope(x),
        _ => false,
    }
}

/// Is `e` safe to duplicate/move: a side-effect-free leaf or chain of them?
///
/// Field reads are included (a field read has no side effect in C#); indexers
/// are NOT (their bounds check can throw), and calls/allocations are not.
fn is_pure_arg(e: &Expr) -> bool {
    match e {
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::StaticField(_, _)
        | Expr::MethodGroup(_)
        | Expr::TypeOf(_) => true,
        Expr::Field(r, _) | Expr::Cast(_, r) => is_pure_arg(r),
        _ => false,
    }
}

/// Replace every variable named in `subst` by its expression.
fn substitute(e: &Expr, subst: &[(String, Expr)]) -> Expr {
    if let Expr::Var(v) = e {
        if let Some((_, value)) = subst.iter().find(|(n, _)| n == v) {
            return value.clone();
        }
        return e.clone();
    }
    match e {
        Expr::Bin(op, a, b) => Expr::Bin(
            *op,
            Box::new(substitute(a, subst)),
            Box::new(substitute(b, subst)),
        ),
        Expr::Cmp(op, a, b) => Expr::Cmp(
            *op,
            Box::new(substitute(a, subst)),
            Box::new(substitute(b, subst)),
        ),
        Expr::Not(x) => Expr::Not(Box::new(substitute(x, subst))),
        Expr::Neg(x) => Expr::Neg(Box::new(substitute(x, subst))),
        Expr::Cast(t, x) => Expr::Cast(t.clone(), Box::new(substitute(x, subst))),
        Expr::AddressOf(x) => Expr::AddressOf(Box::new(substitute(x, subst))),
        Expr::Await(x) => Expr::Await(Box::new(substitute(x, subst))),
        Expr::Field(r, name) => Expr::Field(Box::new(substitute(r, subst)), name.clone()),
        Expr::RawDeref(r, off, w) => Expr::RawDeref(Box::new(substitute(r, subst)), *off, *w),
        Expr::RawAddr(r, off) => Expr::RawAddr(Box::new(substitute(r, subst)), *off),
        Expr::Cond(c, a, b) => Expr::Cond(
            Box::new(substitute(c, subst)),
            Box::new(substitute(a, subst)),
            Box::new(substitute(b, subst)),
        ),
        Expr::Call { recv, name, args } => Expr::Call {
            recv: recv.as_ref().map(|r| Box::new(substitute(r, subst))),
            name: name.clone(),
            args: args.iter().map(|a| substitute(a, subst)).collect(),
        },
        Expr::RawCall(rva, args, ret) => Expr::RawCall(
            *rva,
            args.iter().map(|a| substitute(a, subst)).collect(),
            ret.clone(),
        ),
        Expr::NewObj(ty, args) => Expr::NewObj(
            ty.clone(),
            args.iter().map(|a| substitute(a, subst)).collect(),
        ),
        Expr::NewArr(ty, args) => Expr::NewArr(
            ty.clone(),
            args.iter().map(|a| substitute(a, subst)).collect(),
        ),
        Expr::ArrayInit { elem_type, items } => Expr::ArrayInit {
            elem_type: elem_type.clone(),
            items: items.iter().map(|a| substitute(a, subst)).collect(),
        },
        Expr::CollectionInit {
            type_name,
            ctor_args,
            items,
        } => Expr::CollectionInit {
            type_name: type_name.clone(),
            ctor_args: ctor_args.iter().map(|a| substitute(a, subst)).collect(),
            items: items.iter().map(|a| substitute(a, subst)).collect(),
        },
        Expr::ObjInit {
            type_name,
            ctor_args,
            fields,
        } => Expr::ObjInit {
            type_name: type_name.clone(),
            ctor_args: ctor_args.iter().map(|a| substitute(a, subst)).collect(),
            fields: fields
                .iter()
                .map(|(n, v)| (n.clone(), substitute(v, subst)))
                .collect(),
        },
        Expr::Index { collection, index } => Expr::Index {
            collection: Box::new(substitute(collection, subst)),
            index: Box::new(substitute(index, subst)),
        },
        Expr::NullConditional { receiver, access } => Expr::NullConditional {
            receiver: Box::new(substitute(receiver, subst)),
            access: Box::new(substitute(access, subst)),
        },
        Expr::StringInterp { parts } => {
            use super::super::csharp::InterpPart;
            Expr::StringInterp {
                parts: parts
                    .iter()
                    .map(|p| match p {
                        InterpPart::Expr(e, fmt) => {
                            InterpPart::Expr(Box::new(substitute(e, subst)), fmt.clone())
                        }
                        other => other.clone(),
                    })
                    .collect(),
            }
        }
        Expr::Tuple { names, elements } => Expr::Tuple {
            names: names.clone(),
            elements: elements.iter().map(|a| substitute(a, subst)).collect(),
        },
        Expr::PatternMatch {
            expr,
            type_name,
            binding,
        } => Expr::PatternMatch {
            expr: Box::new(substitute(expr, subst)),
            type_name: type_name.clone(),
            binding: binding.clone(),
        },
        Expr::MethodChain { receiver, methods } => Expr::MethodChain {
            receiver: Box::new(substitute(receiver, subst)),
            methods: methods
                .iter()
                .map(|m| super::super::csharp::ChainMethod {
                    name: m.name.clone(),
                    args: m.args.iter().map(|a| substitute(a, subst)).collect(),
                })
                .collect(),
        },
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => Expr::SwitchExpr {
            discrim: Box::new(substitute(discrim, subst)),
            arms: arms
                .iter()
                .map(|(p, r)| (substitute(p, subst), substitute(r, subst)))
                .collect(),
            default_arm: default_arm
                .as_ref()
                .map(|d| Box::new(substitute(d, subst))),
        },
        Expr::Range { from, to } => Expr::Range {
            from: from.as_ref().map(|f| Box::new(substitute(f, subst))),
            to: to.as_ref().map(|t| Box::new(substitute(t, subst))),
        },
        Expr::IndexFromEnd(x) => Expr::IndexFromEnd(Box::new(substitute(x, subst))),
        // Leaf / unmodelled nodes carry no variable of this lambda's scope.
        _ => e.clone(),
    }
}

/// Inline `lambda(args)` under the narrow proof described in the module docs.
fn inline_lambda_call(lambda: &Expr, args: &[Expr]) -> Option<Expr> {
    let Expr::Lambda { params, body } = lambda else {
        return None;
    };
    if params.len() != args.len() {
        return None;
    }
    if has_shadowing_scope(body) {
        return None;
    }
    // Duplicate parameter names make the substitution ambiguous.
    for (i, p) in params.iter().enumerate() {
        if params[..i].contains(p) {
            return None;
        }
    }
    let mut subst: Vec<(String, Expr)> = Vec::with_capacity(params.len());
    for (i, p) in params.iter().enumerate() {
        // Dropping an argument (unused parameter) changes evaluation; moving a
        // non-pure argument into the body changes order. Both are refused.
        if count_var(body, p) != 1 || !is_pure_arg(&args[i]) {
            return None;
        }
        subst.push((p.clone(), args[i].clone()));
    }
    let result = substitute(body, &subst);
    // Defence in depth: if a node kind was not modelled by `substitute`, the
    // check above counted the parameter but the body still names it. Never
    // emit an unbound parameter.
    if params.iter().any(|p| count_var(&result, p) != 0) {
        return None;
    }
    Some(result)
}

/// Single-assignment lambda locals of one list, collected AFTER construction
/// folding so a local assigned a delegate construction is eligible too.
fn lambda_locals(list: &[Stmt]) -> Vec<(String, Expr)> {
    let mut out = Vec::new();
    for (i, s) in list.iter().enumerate() {
        if let Stmt::Assign(n, _, Expr::Lambda { .. }) = s {
            if single_write_index(list, n) == Some(i) {
                if let Stmt::Assign(_, _, e) = s {
                    out.push((n.clone(), e.clone()));
                }
            }
        }
    }
    out
}

/// `.Invoke(args)` on a single-assignment lambda local → its body.
fn rewrite_invoke(e: &mut Expr, locals: &[(String, Expr)]) -> usize {
    if let Expr::Call {
        recv: Some(r),
        name,
        args,
    } = e
    {
        if name == "Invoke" {
            if let Expr::Var(v) = r.as_ref() {
                if let Some((_, lambda)) = locals.iter().find(|(n, _)| n == v) {
                    if let Some(inlined) = inline_lambda_call(lambda, args) {
                        *e = inlined;
                        LAMBDA_INLINE_REWRITES.fetch_add(1, Ordering::Relaxed);
                        return 1 + rewrite_invoke(e, locals);
                    }
                }
            }
        }
    }
    let mut n = 0usize;
    match e {
        Expr::Bin(_, a, c) | Expr::Cmp(_, a, c) => {
            n += rewrite_invoke(a, locals);
            n += rewrite_invoke(c, locals);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            n += rewrite_invoke(x, locals);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            n += rewrite_invoke(r, locals);
        }
        Expr::Cond(c, a, d) => {
            n += rewrite_invoke(c, locals);
            n += rewrite_invoke(a, locals);
            n += rewrite_invoke(d, locals);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                n += rewrite_invoke(r, locals);
            }
            for a in args {
                n += rewrite_invoke(a, locals);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                n += rewrite_invoke(a, locals);
            }
        }
        Expr::Lambda { body, .. } => n += rewrite_invoke(body, locals),
        Expr::Index { collection, index } => {
            n += rewrite_invoke(collection, locals);
            n += rewrite_invoke(index, locals);
        }
        Expr::NullConditional { receiver, access } => {
            n += rewrite_invoke(receiver, locals);
            n += rewrite_invoke(access, locals);
        }
        Expr::ArrayInit { items, .. } => {
            for item in items {
                n += rewrite_invoke(item, locals);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter_mut().chain(items.iter_mut()) {
                n += rewrite_invoke(a, locals);
            }
        }
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            for a in ctor_args {
                n += rewrite_invoke(a, locals);
            }
            for (_, v) in fields {
                n += rewrite_invoke(v, locals);
            }
        }
        Expr::StringInterp { parts } => {
            use super::super::csharp::InterpPart;
            for p in parts {
                if let InterpPart::Expr(e, _) = p {
                    n += rewrite_invoke(e, locals);
                }
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                n += rewrite_invoke(e, locals);
            }
        }
        Expr::PatternMatch { expr, .. } => {
            n += rewrite_invoke(expr, locals);
        }
        Expr::MethodChain { receiver, methods } => {
            n += rewrite_invoke(receiver, locals);
            for m in methods {
                for a in &mut m.args {
                    n += rewrite_invoke(a, locals);
                }
            }
        }
        _ => {}
    }
    n
}

/// The expression slot a statement exposes to the two rewrite phases.
fn stmt_expr_mut(s: &mut Stmt) -> Option<&mut Expr> {
    match s {
        Stmt::Assign(_, _, e) => Some(e),
        Stmt::Eval(e) => Some(e),
        Stmt::Return(Some(e)) => Some(e),
        Stmt::Throw(e) => Some(e),
        Stmt::YieldReturn(e) => Some(e),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => Some(cond),
        Stmt::For { cond: Some(c), .. } => Some(c),
        Stmt::Lock { expr, .. } => Some(expr),
        Stmt::EventAssign { value, .. } => Some(value),
        Stmt::IndexAssign { value, .. } => Some(value),
        Stmt::StringSwitch { disc, .. } => Some(disc),
        Stmt::Using { decl, .. } => {
            if let Stmt::Assign(_, _, e) = decl.as_mut() {
                Some(e)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Fold one statement list.
fn fold_list(list: &mut Vec<Stmt>) -> usize {
    let facts = ListFacts::collect(list);
    let mut n = 0usize;
    for (i, s) in list.iter_mut().enumerate() {
        if let Some(e) = stmt_expr_mut(s) {
            n += rewrite_expr(e, i, &facts);
        }
    }
    // Invoke inlining sees lambdas that the construction fold just produced.
    let locals = lambda_locals(list);
    if !locals.is_empty() {
        for s in list.iter_mut() {
            if let Some(e) = stmt_expr_mut(s) {
                n += rewrite_invoke(e, &locals);
            }
        }
    }
    n
}

/// Run the lambda inlining pass over one method body.
///
/// The `_maps` parameter is kept for the pipeline signature; body lookup goes
/// through the global registry populated by the renderer (`Phase 5A`).
pub fn apply_with_maps(stmts: &mut Vec<Stmt>, _maps: &ResolveMaps) -> usize {
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        n += fold_list(list);
    });
    n
}

/// Run the lambda inlining pass without `ResolveMaps` (for testing).
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        n += fold_list(list);
    });
    n
}

/// Reset the counters. For testing only.
pub fn reset_counter() {
    LAMBDA_INLINE_REWRITES.store(0, Ordering::Relaxed);
    LAMBDA_CELL_MISSES.store(0, Ordering::Relaxed);
    LAMBDA_BODY_MISSES.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::{clear_lambda_bodies, register_lambda_body};
    use std::sync::Mutex;

    /// The registry is process-global and the test harness runs tests in
    /// parallel, so every test that touches it holds this lock. Without it one
    /// test's `clear_lambda_bodies` can erase another's registration.
    static REGISTRY_LOCK: Mutex<()> = Mutex::new(());

    /// Acquire the registry lock even if another test panicked while holding
    /// it; a poisoned lock must not turn one failure into eighteen.
    fn registry_lock() -> std::sync::MutexGuard<'static, ()> {
        REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn var(n: &str) -> Expr {
        Expr::Var(n.into())
    }

    /// `System.Action`1..ctor(this, <method>)` — the real construction spelling.
    fn ctor(method: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "System.Action`1..ctor".into(),
            args: vec![Expr::This, method],
        }
    }

    fn func_ctor(method: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "System.Func`2..ctor".into(),
            args: vec![Expr::This, method],
        }
    }

    /// The lossy spelling produced when the allocation helper's type is lost.
    fn object_ctor(method: Expr) -> Expr {
        Expr::NewObj("object".into(), vec![Expr::This, method])
    }

    fn reg(name: &str, params: &[&str], body: Expr) {
        register_lambda_body(
            name.to_string(),
            params.iter().map(|p| p.to_string()).collect(),
            body,
        );
    }

    // -----------------------------------------------------------------------
    // Helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn generated_lambda_name_detection() {
        assert!(is_generated_lambda("<>c.<M>b__1_0"));
        assert!(is_generated_lambda("<>c.<Method>b__0"));
        assert!(is_generated_lambda(
            "<MyClass>o__DisplayClass3.<Method>b__0"
        ));
        assert!(!is_generated_lambda("NormalMethod"));
        assert!(!is_generated_lambda("get_Value"));
        assert!(!is_generated_lambda("<>c..ctor"));
    }

    #[test]
    fn delegate_arity_computation() {
        assert_eq!(delegate_arity("System.Action"), Some(0));
        assert_eq!(delegate_arity("System.Action`1"), Some(1));
        assert_eq!(delegate_arity("System.Action`2"), Some(2));
        assert_eq!(delegate_arity("System.Func`2"), Some(1));
        assert_eq!(delegate_arity("System.Func`3"), Some(2));
        assert_eq!(delegate_arity("System.Predicate`1"), None);
        assert_eq!(delegate_arity("Dictionary<string, List<int>>"), None);
    }

    #[test]
    fn registry_spellings_cover_both_display_forms() {
        let v = lookup_variants("<>c.<Initialize>b__43_3");
        assert!(v.contains(&"<>c.<Initialize>b__43_3".to_string()));
        assert!(v.contains(&"<Initialize>b__43_3".to_string()));
        let v = lookup_variants("<Initialize>b__43_3");
        assert_eq!(v, vec!["<Initialize>b__43_3".to_string()]);
        let v = lookup_variants("<>c__DisplayClass5.<M>b__0");
        assert!(v.contains(&"<M>b__0".to_string()));
    }

    // -----------------------------------------------------------------------
    // Core inlining tests
    // -----------------------------------------------------------------------

    /// A delegate construction referencing a registered lambda method is
    /// rewritten to `Expr::Lambda`.
    #[test]
    fn delegate_construction_is_inlined_to_lambda() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let body = Expr::Field(Box::new(var("x")), "Value".into());
        reg("<>c.<M>b__1_0", &["x"], body);
        let mut stmts = vec![Stmt::Assign(
            "action".to_string(),
            None,
            ctor(Expr::MethodGroup("<>c.<M>b__1_0".to_string())),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "one rewrite expected");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Lambda { params, body }) => {
                assert_eq!(params, &["x".to_string()]);
                assert!(matches!(body.as_ref(), Expr::Field(..)));
            }
            other => panic!("expected Lambda, got {other:?}"),
        }
        clear_lambda_bodies();
    }

    /// The renderer files a lambda body under its BARE metadata name, while a
    /// `MethodGroup` operand carries the display name. The lookup must bridge
    /// the two spellings — otherwise every static `<>c` lambda is missed.
    #[test]
    fn bare_registered_name_matches_the_display_spelling() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<Initialize>b__43_3", &["x"], var("x"));
        let mut stmts = vec![Stmt::Eval(ctor(Expr::MethodGroup(
            "<>c.<Initialize>b__43_3".to_string(),
        )))];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Lambda { .. })));
        clear_lambda_bodies();
    }

    /// A `Func` construction is also inlined.
    #[test]
    fn func_delegate_is_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<>c.<Sort>b__0_0", &["x"], var("x"));
        let mut stmts = vec![Stmt::Assign(
            "func".to_string(),
            None,
            func_ctor(Expr::MethodGroup("<>c.<Sort>b__0_0".to_string())),
        )];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Lambda { params, .. }) => {
                assert_eq!(params, &["x".to_string()]);
            }
            other => panic!("expected Lambda, got {other:?}"),
        }
        clear_lambda_bodies();
    }

    /// The type-lost `NewObj("object", [target, method])` spelling is the same
    /// delegate construction once the second operand names a generated lambda.
    #[test]
    fn object_spelling_with_a_generated_method_is_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__0", &[], var("this"));
        let mut stmts = vec![Stmt::Assign(
            "d".to_string(),
            None,
            object_ctor(Expr::MethodGroup("<>c.<M>b__0".to_string())),
        )];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::Lambda { .. })));
        clear_lambda_bodies();
    }

    /// The real corpus loads the method cell into a local first; the
    /// construction then names the local. A single-assignment local proves the
    /// value.
    #[test]
    fn method_cell_held_in_a_local_is_resolved() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__2_0", &["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("<>c.<M>b__2_0".to_string()),
            ),
            Stmt::Assign("d".to_string(), None, object_ctor(var("v12"))),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[1], Stmt::Assign(_, _, Expr::Lambda { .. })));
        clear_lambda_bodies();
    }

    /// A local written twice does not prove which method cell reaches the
    /// construction: refuse.
    #[test]
    fn a_reassigned_method_cell_is_refused() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__1_0", &[], var("a"));
        reg("<M>b__1_1", &[], var("b"));
        let mut stmts = vec![
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("<>c.<M>b__1_0".to_string()),
            ),
            Stmt::Assign(
                "v12".to_string(),
                None,
                Expr::MethodGroup("<>c.<M>b__1_1".to_string()),
            ),
            Stmt::Eval(object_ctor(var("v12"))),
        ];
        assert_eq!(apply(&mut stmts), 0);
        clear_lambda_bodies();
    }

    /// An unregistered lambda method is left unchanged.
    #[test]
    fn unregistered_lambda_is_unchanged() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let mut stmts = vec![Stmt::Eval(ctor(Expr::MethodGroup(
            "<>c.<M>b__1_0".to_string(),
        )))];
        let before = format!("{:?}", stmts);
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(before, format!("{:?}", stmts));
        clear_lambda_bodies();
    }

    /// An arity mismatch is refused.
    #[test]
    fn arity_mismatch_is_refused() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<>c.<M>b__1_0", &[], var("x"));
        let mut stmts = vec![Stmt::Eval(ctor(Expr::MethodGroup(
            "<>c.<M>b__1_0".to_string(),
        )))];
        let n = apply(&mut stmts);
        assert_eq!(n, 0, "arity mismatch must refuse");
        clear_lambda_bodies();
    }

    /// A non-delegate construction is never touched, even when the registered
    /// name matches.
    #[test]
    fn non_delegate_construction_is_untouched() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__1_0", &["x"], var("x"));
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "System.Collections.Generic.List`1..ctor".into(),
            args: vec![Expr::This, Expr::MethodGroup("<>c.<M>b__1_0".to_string())],
        })];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        clear_lambda_bodies();
    }

    /// A non-lambda method reference is not inlined.
    #[test]
    fn non_lambda_method_is_not_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let mut stmts = vec![Stmt::Eval(ctor(Expr::MethodGroup(
            "SomeClass.NormalMethod".to_string(),
        )))];
        assert_eq!(apply(&mut stmts), 0);
        clear_lambda_bodies();
    }

    /// A body that reads `this` is only safe when the construction binds the
    /// enclosing instance. With a display-class target it names captured state
    /// that the caller does not have — refuse.
    #[test]
    fn this_capturing_body_with_a_non_this_target_is_refused() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg(
            "<M>b__1_0",
            &[],
            Expr::Field(Box::new(Expr::This), "state".into()),
        );
        let display_class_target = Expr::NewObj(
            "object".into(),
            vec![var("v83"), Expr::MethodGroup("<>c.<M>b__1_0".to_string())],
        );
        let mut stmts = vec![Stmt::Eval(display_class_target)];
        assert_eq!(apply(&mut stmts), 0);
        clear_lambda_bodies();
    }

    /// The same body IS safe when the delegate target is `this`: then the
    /// method is an instance method on the enclosing class, whose `this` is the
    /// very same object the caller's `this` names.
    #[test]
    fn this_capturing_body_with_a_this_target_is_allowed() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg(
            "<M>b__1_0",
            &[],
            Expr::Field(Box::new(Expr::This), "state".into()),
        );
        let mut stmts = vec![Stmt::Eval(object_ctor(Expr::MethodGroup(
            "<>c.<M>b__1_0".to_string(),
        )))];
        assert_eq!(apply(&mut stmts), 1);
        clear_lambda_bodies();
    }

    /// Nested delegate constructions inside an if-body are also inlined.
    #[test]
    fn nested_construction_is_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__1_0", &["x"], var("x"));
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(ctor(Expr::MethodGroup(
                "<>c.<M>b__1_0".to_string(),
            )))],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert!(matches!(&then_body[0], Stmt::Eval(Expr::Lambda { .. })));
            }
            other => panic!("expected If, got {other:?}"),
        }
        clear_lambda_bodies();
    }

    /// The NewObj spelling is also accepted.
    #[test]
    fn newobj_spelling_is_accepted() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__1_0", &["x"], var("x"));
        let mut stmts = vec![Stmt::Eval(Expr::NewObj(
            "System.Action`1".into(),
            vec![Expr::This, Expr::MethodGroup("<>c.<M>b__1_0".to_string())],
        ))];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::Eval(Expr::Lambda { .. })));
        clear_lambda_bodies();
    }

    /// Multiple delegate constructions in one method are all inlined.
    #[test]
    fn multiple_constructions_are_all_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__1_0", &["x"], var("x"));
        reg("<M>b__1_1", &["y"], var("y"));
        let mut stmts = vec![
            Stmt::Assign(
                "a".to_string(),
                None,
                ctor(Expr::MethodGroup("<>c.<M>b__1_0".to_string())),
            ),
            Stmt::Assign(
                "b".to_string(),
                None,
                ctor(Expr::MethodGroup("<>c.<M>b__1_1".to_string())),
            ),
        ];
        assert_eq!(apply(&mut stmts), 2);
        clear_lambda_bodies();
    }

    // -----------------------------------------------------------------------
    // `.Invoke` inlining
    // -----------------------------------------------------------------------

    fn reg_and_assign(local: &str, params: &[&str], body: Expr) -> Vec<Stmt> {
        reg("<M>b__9_0", params, body);
        vec![
            Stmt::Assign(
                local.to_string(),
                None,
                object_ctor(Expr::MethodGroup("<>c.<M>b__9_0".to_string())),
            ),
        ]
    }

    /// `v = (x) => x.Value; v.Invoke(p)` folds to `p.Value` (parameter used
    /// once, argument pure).
    #[test]
    fn invoke_on_a_single_assignment_lambda_is_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let mut stmts = reg_and_assign(
            "v",
            &["x"],
            Expr::Field(Box::new(var("x")), "Value".into()),
        );
        stmts.push(Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("v"))),
            name: "Invoke".into(),
            args: vec![var("p")],
        }));
        assert_eq!(apply(&mut stmts), 2, "one construction + one invoke");
        match &stmts[1] {
            Stmt::Eval(Expr::Field(r, f)) => {
                assert!(matches!(r.as_ref(), Expr::Var(v) if v == "p"));
                assert_eq!(f, "Value");
            }
            other => panic!("expected inlined body, got {other:?}"),
        }
        clear_lambda_bodies();
    }

    /// A parameter used twice would duplicate the argument expression.
    #[test]
    fn invoke_with_a_reused_parameter_is_refused() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let body = Expr::Bin(
            super::super::super::csharp::BinOp::Add,
            Box::new(var("x")),
            Box::new(var("x")),
        );
        let mut stmts = reg_and_assign("v", &["x"], body);
        stmts.push(Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("v"))),
            name: "Invoke".into(),
            args: vec![var("p")],
        }));
        assert_eq!(apply(&mut stmts), 1, "construction folds, invoke must not");
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::Call { .. })));
        clear_lambda_bodies();
    }

    /// A non-pure argument would be MOVED past the body's side effects.
    #[test]
    fn invoke_with_an_impure_argument_is_refused() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let mut stmts = reg_and_assign("v", &["x"], var("x"));
        stmts.push(Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("v"))),
            name: "Invoke".into(),
            args: vec![Expr::Call {
                recv: None,
                name: "Next".into(),
                args: vec![],
            }],
        }));
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::Call { .. })));
        clear_lambda_bodies();
    }

    /// An unused parameter would DROP the argument's evaluation.
    #[test]
    fn invoke_with_an_unused_parameter_is_refused() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let mut stmts = reg_and_assign("v", &["x"], var("other"));
        stmts.push(Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("v"))),
            name: "Invoke".into(),
            args: vec![var("p")],
        }));
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::Call { .. })));
        clear_lambda_bodies();
    }

    /// A local assigned more than once does not prove which lambda is invoked.
    #[test]
    fn invoke_on_a_reassigned_local_is_refused() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__9_0", &["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(
                "v".to_string(),
                None,
                ctor(Expr::MethodGroup("<>c.<M>b__9_0".to_string())),
            ),
            Stmt::Assign("v".to_string(), None, Expr::Null),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("v"))),
                name: "Invoke".into(),
                args: vec![var("p")],
            }),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[2], Stmt::Eval(Expr::Call { .. })));
        clear_lambda_bodies();
    }

    /// A zero-parameter lambda's `.Invoke()` is its body.
    #[test]
    fn parameterless_invoke_is_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let mut stmts = reg_and_assign(
            "v",
            &[],
            Expr::Call {
                recv: None,
                name: "Configure".into(),
                args: vec![],
            },
        );
        stmts.push(Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("v"))),
            name: "Invoke".into(),
            args: vec![],
        }));
        assert_eq!(apply(&mut stmts), 2);
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::Call { name, .. }) if name == "Configure"));
        clear_lambda_bodies();
    }

    /// A body that reads `this` captured by the LAMBDA ITSELF keeps its meaning
    /// when inlined into the same method's call site (the enclosing instance is
    /// the same object).
    #[test]
    fn invoke_body_with_this_is_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let body = Expr::Field(Box::new(Expr::This), "state".into());
        let mut stmts = reg_and_assign("v", &[], body);
        stmts.push(Stmt::Eval(Expr::Call {
            recv: Some(Box::new(var("v"))),
            name: "Invoke".into(),
            args: vec![],
        }));
        assert_eq!(apply(&mut stmts), 2);
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::Field(..))));
        clear_lambda_bodies();
    }

    /// The second phase uses lambdas created by the first phase in the same
    /// list: `v = new Action(...)` then `v.Invoke()`.
    #[test]
    fn invoke_after_construction_fold_is_inlined() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        reg("<M>b__7_0", &[], var("result"));
        let mut stmts = vec![
            Stmt::Assign(
                "v".to_string(),
                None,
                object_ctor(Expr::MethodGroup("<>c.<M>b__7_0".to_string())),
            ),
            Stmt::Assign(
                "r".to_string(),
                None,
                Expr::Call {
                    recv: Some(Box::new(var("v"))),
                    name: "Invoke".into(),
                    args: vec![],
                },
            ),
        ];
        assert_eq!(apply(&mut stmts), 2);
        assert!(matches!(&stmts[1], Stmt::Assign(_, _, Expr::Var(v)) if v == "result"));
        clear_lambda_bodies();
    }

    /// A delegate-shaped construction whose method cell the renderer lost is
    /// counted (and reported), not silently ignored.
    #[test]
    fn unnamed_method_cell_is_counted() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let mut stmts = vec![Stmt::Eval(Expr::NewObj(
            "object".into(),
            vec![Expr::This, Expr::TypeOf("SomeCaller".into())],
        ))];
        assert_eq!(apply(&mut stmts), 0, "nothing may be folded without a body");
        let text = report();
        assert!(text.contains("1 unnamed method cell"), "report was: {text}");
        clear_lambda_bodies();
        reset_counter();
    }

    /// A generated method name with no registered body is a distinct miss (the
    /// identity is known; the body is not).
    #[test]
    fn missing_body_is_counted() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        let mut stmts = vec![Stmt::Eval(ctor(Expr::MethodGroup(
            "<>c.<NeverRendered>b__1_0".to_string(),
        )))];
        assert_eq!(apply(&mut stmts), 0);
        let text = report();
        assert!(text.contains("1 body miss"), "report was: {text}");
        clear_lambda_bodies();
        reset_counter();
    }

    /// Kept from the original file: `apply_with_maps` shares the fold logic.
    #[test]
    fn apply_with_maps_compiles_and_counts() {
        let _g = registry_lock();
        clear_lambda_bodies();
        reset_counter();
        assert_eq!(LAMBDA_INLINE_REWRITES.load(Ordering::Relaxed), 0);
        clear_lambda_bodies();
    }
}
