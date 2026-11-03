//! Late expression/structure cleanup driven by return-type context.
//!
//! Four shapes measured on the `TokenManager` class:
//!
//! 1. **Conditional return**
//!    `v = init; if (c) v = a; return v;` and
//!    `if (c) v = a; else v = b; return v;`
//!    → `return c ? a : b;` (then the boolean-ternary fold turns
//!    `c ? e : false` into `c && e`, `c ? false : e` into `!c && e`, etc.).
//! 2. **Constructor call duplication / typing**
//!    `v = new object(args); v..ctor(args2);` is one managed construction.
//!    When `v` is the method's returned local and the return type is a class,
//!    the allocation is typed with it (`new RewardData(...)`). A null test on
//!    a freshly allocated local is then always true and its `if` shell folds.
//! 3. **`return 0` for a reference result** → `return null;`.
//! 4. **Duplicate indexer write**
//!    `c[i] = v; c.set_Item(i, v);` → the low-level call is dropped.
//!
//! All rewrites are value-preserving; conditional folding only accepts boolean
//! or literal arms, and refuses when the merged local is read elsewhere.

use super::super::csharp::{BinOp, CmpKind, Expr, Stmt};
use super::children_mut;
use std::collections::HashSet;

/// Method return-type context.
#[derive(Default, Clone)]
pub struct RetInfo {
    pub is_ref: bool,
    pub is_bool: bool,
    /// Simple C# type name for a class return (`RewardData`).
    pub type_name: Option<String>,
    /// The method is a constructor (`.ctor`).
    pub is_ctor: bool,
}

fn is_boolish(e: &Expr, ret_is_bool: bool) -> bool {
    match e {
        Expr::Cmp(..) | Expr::Not(_) | Expr::Bool(_) => true,
        Expr::Int(0) | Expr::Int(1) => ret_is_bool,
        Expr::Bin(BinOp::AndAlso | BinOp::OrElse, a, b) => {
            is_boolish(a, ret_is_bool) && is_boolish(b, ret_is_bool)
        }
        _ => false,
    }
}

fn as_bool_lit(e: &Expr, want: bool) -> bool {
    match e {
        Expr::Bool(b) => *b == want,
        Expr::Int(0) => !want,
        Expr::Int(1) => want,
        _ => false,
    }
}

/// Invert a comparison instead of wrapping it in `Not`.
fn invert_cmp(e: &mut Expr) {
    if let Expr::Not(inner) = e {
        if let Expr::Cmp(k, a, b) = inner.as_ref() {
            let inv = match k {
                CmpKind::Eq => CmpKind::Ne,
                CmpKind::Ne => CmpKind::Eq,
                CmpKind::Lt => CmpKind::Ge,
                CmpKind::Le => CmpKind::Gt,
                CmpKind::Gt => CmpKind::Le,
                CmpKind::Ge => CmpKind::Lt,
            };
            *e = Expr::Cmp(inv, a.clone(), b.clone());
        }
    }
}

/// Boolean ternary folds: `c ? e : false` → `c && e`, etc.
/// `!(a <= b)` → `a > b` anywhere in the tree.
fn invert_not_cmp(e: &mut Expr) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += invert_not_cmp(c);
    }
    let before = format!("{e:?}");
    invert_cmp(e);
    if format!("{e:?}") != before {
        n += 1;
    }
    n
}

fn fold_bool_cond(e: &mut Expr, ret_is_bool: bool) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += fold_bool_cond(c, ret_is_bool);
    }
    let Expr::Cond(cond, a, b) = e else { return n };
    if !is_boolish(a, ret_is_bool) || !is_boolish(b, ret_is_bool) {
        return n;
    }
    if as_bool_lit(b, false) {
        *e = Expr::Bin(BinOp::AndAlso, cond.clone(), a.clone());
        n += 1;
    } else if as_bool_lit(a, false) {
        let mut neg = Expr::Not(cond.clone());
        invert_cmp(&mut neg);
        *e = Expr::Bin(BinOp::AndAlso, Box::new(neg), b.clone());
        n += 1;
    } else if as_bool_lit(b, true) {
        *e = Expr::Bin(BinOp::OrElse, cond.clone(), a.clone());
        n += 1;
    } else if as_bool_lit(a, true) {
        let mut neg = Expr::Not(cond.clone());
        invert_cmp(&mut neg);
        *e = Expr::Bin(BinOp::OrElse, Box::new(neg), b.clone());
        n += 1;
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
        Expr::Call { recv, args, .. } => {
            let mut v: Vec<&mut Expr> = args.iter_mut().collect();
            if let Some(r) = recv {
                v.push(r);
            }
            v
        }
        _ => vec![],
    }
}

fn is_noise(s: &Stmt) -> bool {
    matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. })
}

fn next_exec(list: &[Stmt], from: usize) -> Option<usize> {
    (from..list.len()).find(|&i| !is_noise(&list[i]))
}

/// `if (c) v = a; else v = b;` → returns (c, a, b).
fn both_arms_assign(stmt: &Stmt, v: &str) -> Option<(Expr, Expr, Expr)> {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = stmt
    else {
        return None;
    };
    let pick = |b: &[Stmt]| -> Option<Expr> {
        let mut out = None;
        for s in b {
            if is_noise(s) {
                continue;
            }
            match s {
                Stmt::Assign(n, _, e) if n == v && out.is_none() => out = Some(e.clone()),
                _ => return None,
            }
        }
        out
    };
    Some((cond.clone(), pick(then_body)?, pick(else_body)?))
}

fn fold_conditional_returns(list: &mut Vec<Stmt>, ret: &RetInfo) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        // Shape A: `v = init; if (c) { v = a } return v;`
        if let Stmt::Assign(v, _, init) = &list[i] {
            if let Some(j) = next_exec(list, i + 1) {
                if let Stmt::If {
                    cond,
                    then_body,
                    else_body,
                } = &list[j]
                {
                    let single = else_body.iter().all(|s| is_noise(s));
                    let then_assign = then_body.iter().filter(|s| !is_noise(s)).count() == 1
                        && then_body.iter().any(|s| matches!(s, Stmt::Assign(n, _, _) if n == v));
                    let val = then_body
                        .iter()
                        .find_map(|s| match s {
                            Stmt::Assign(n, _, e) if n == v => Some(e.clone()),
                            _ => None,
                        });
                    if single && then_assign {
                        if let (Some(val), Some(k)) = (val, next_exec(list, j + 1)) {
                            let is_ret_v = matches!(&list[k], Stmt::Return(Some(Expr::Var(r))) if r.as_str() == v.as_str());
                            let typed_ok = (is_boolish(init, ret.is_bool)
                                && is_boolish(&val, ret.is_bool))
                                || matches!(init, Expr::Int(_) | Expr::Null | Expr::Bool(_));
                            if is_ret_v && typed_ok {
                                let merged = Expr::Cond(
                                    Box::new(cond.clone()),
                                    Box::new(val),
                                    Box::new(init.clone()),
                                );
                                list[k] = Stmt::Return(Some(merged));
                                // Remove the If and the initial assignment.
                                list.remove(j);
                                list.remove(i);
                                n += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        // Shape B: `if (c) v = a; else v = b; return v;`
        let shape_b = match &list[i] {
            Stmt::If { .. } => {
                // Determine the assigned name from the then arm.
                let v = match &list[i] {
                    Stmt::If { then_body, .. } => then_body.iter().find_map(|s| match s {
                        Stmt::Assign(n, _, _) => Some(n.clone()),
                        _ => None,
                    }),
                    _ => None,
                };
                v.and_then(|v| both_arms_assign(&list[i], &v).map(|t| (v, t)))
            }
            _ => None,
        };
        if let Some((v, (cond, a, b))) = shape_b {
            if let Some(k) = next_exec(list, i + 1) {
                let is_ret_v = matches!(&list[k], Stmt::Return(Some(Expr::Var(r))) if r.as_str() == v.as_str());
                let typed_ok = (is_boolish(&a, ret.is_bool) && is_boolish(&b, ret.is_bool))
                    || matches!(a, Expr::Int(_) | Expr::Null | Expr::Bool(_))
                    || matches!(b, Expr::Int(_) | Expr::Null | Expr::Bool(_));
                if is_ret_v && typed_ok {
                    let merged = Expr::Cond(Box::new(cond), Box::new(a), Box::new(b));
                    list[k] = Stmt::Return(Some(merged));
                    list.remove(i);
                    n += 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    n
}

/// Name of the local a method returns (`Return(Var(v))` anywhere).
fn returned_local(stmts: &[Stmt]) -> Option<String> {
    for s in stmts {
        if let Stmt::Return(Some(Expr::Var(v))) = s {
            return Some(v.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            if let Some(v) = returned_local(k) {
                return Some(v);
            }
        }
    }
    None
}

fn simple_type_name(t: &str) -> Option<String> {
    let t = t.rsplit('.').next().unwrap_or(t);
    if t.is_empty()
        || t == "object"
        || t == "Object"
        || !t.chars().next()?.is_ascii_uppercase()
        || t.contains('<')
        || t.contains('[')
    {
        return None;
    }
    Some(t.to_string())
}

fn fold_ctor_calls(list: &mut Vec<Stmt>, ret: &RetInfo, returned: Option<&str>) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        let (v, args) = match &list[i] {
            Stmt::Assign(v, _, Expr::NewObj(_ty, args)) => (v.clone(), args.clone()),
            _ => {
                i += 1;
                continue;
            }
        };
        // Merge an adjacent `v..ctor(...)` (comments allowed between).
        let mut ctor_at = None;
        let mut j = i + 1;
        while j < list.len() && is_noise(&list[j]) {
            j += 1;
        }
        // The constructor call reaches us either as a bare Eval or as the
        // RHS of a temp assignment (`v17 = v..ctor(args)`).
        let cargs = match list.get(j) {
            Some(Stmt::Eval(Expr::Call {
                recv: Some(r),
                name,
                args,
            })) if name.ends_with(".ctor")
                && matches!(r.as_ref(), Expr::Var(x) if x == &v) =>
            {
                Some(args.clone())
            }
            Some(Stmt::Assign(_, _, Expr::Call {
                recv: Some(r),
                name,
                args,
            })) if name.ends_with(".ctor")
                && matches!(r.as_ref(), Expr::Var(x) if x == &v) =>
            {
                Some(args.clone())
            }
            _ => None,
        };
        if let Some(cargs) = cargs {
            ctor_at = Some(j);
            // Prefer the allocation args; fall back to the ctor's.
            let use_args = if args.is_empty() { cargs } else { args.clone() };
            if let Stmt::Assign(_, _, Expr::NewObj(ty2, a2)) = &mut list[i] {
                *a2 = use_args;
                // Type the allocation when the value is the method result
                // and the metadata return type is a concrete class.
                if (ty2 == "object" || ty2 == "Object" || ty2 == "__unknown")
                    && ret.is_ref
                    && returned == Some(v.as_str())
                {
                    if let Some(t) = ret.type_name.as_deref().and_then(simple_type_name) {
                        *ty2 = t;
                    }
                }
            }
            n += 1;
        }
        if let Some(j) = ctor_at {
            list.remove(j);
            // i unchanged: inspect the merged allocation again.
            continue;
        }
        i += 1;
    }
    n
}

/// `if (v != null) { … }` where `v` was allocated just above and never
/// reassigned ⇒ condition always true; splice the body.
fn fold_alloc_null_checks(list: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    let mut allocs: HashSet<String> = HashSet::new();
    let mut i = 0;
    while i < list.len() {
        if let Stmt::Assign(v, _, Expr::NewObj(..)) = &list[i] {
            allocs.insert(v.clone());
            i += 1;
            continue;
        }
        if let Stmt::If {
            cond,
            then_body,
            else_body,
        } = &list[i]
        {
            if else_body.is_empty() && !then_body.iter().any(|s| matches!(s, Stmt::Label(_))) {
                let tested = match cond {
                    Expr::Cmp(CmpKind::Ne, a, b)
                        if matches!(b.as_ref(), Expr::Null | Expr::Int(0))
                            && matches!(a.as_ref(), Expr::Var(_)) =>
                    {
                        match a.as_ref() {
                            Expr::Var(v) => Some(v.clone()),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some(v) = tested {
                    if allocs.contains(&v) {
                        let body = then_body.clone();
                        list.splice(i..=i, body);
                        n += 1;
                        continue;
                    }
                }
            }
        }
        // A write to a name invalidates its allocation proof.
        if let Stmt::Assign(v, _, _) = &list[i] {
            allocs.remove(v);
        }
        i += 1;
    }
    n
}

/// Drop a `set_Item` call immediately following the equivalent index write.
fn drop_duplicate_set_item(list: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i + 1 < list.len() {
        let dup = match (&list[i], &list[i + 1]) {
            (
                Stmt::IndexAssign {
                    collection: c1,
                    index: i1,
                    value: v1,
                },
                Stmt::Eval(Expr::Call {
                    recv: Some(r),
                    name,
                    args,
                }),
            ) if name == "set_Item" && args.len() == 2 => {
                matcher_eq(r, c1) && matcher_eq(&args[0], i1) && matcher_eq(&args[1], v1)
            }
            _ => false,
        };
        if dup {
            list.remove(i + 1);
            n += 1;
        } else {
            i += 1;
        }
    }
    n
}

fn matcher_eq(a: &Expr, b: &Expr) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

fn walk_lists(stmts: &mut Vec<Stmt>, f: &mut dyn FnMut(&mut Vec<Stmt>) -> usize) -> usize {
    let mut total = 0;
    for s in stmts.iter_mut() {
        for k in children_mut(s) {
            total += walk_lists(k, f);
        }
    }
    total + f(stmts)
}

/// Apply the cleanup. Returns the number of rewrites.
pub fn apply(stmts: &mut Vec<Stmt>, ret: &RetInfo) -> usize {
    let mut total = 0;
    // A constructor's implicit base `System.Object..ctor()` is not source.
    // IL2CPP compiles an empty `.ctor` as a tail jump to `Object..ctor`, which
    // renders as a bare `new Object();` — a fabricated allocation
    // (0x18A50DC). C# already performs this call implicitly.
    if ret.is_ctor {
        total += drop_implicit_object_ctor(stmts);
    }
    // RUNTIME-HELPER CALLS. `IL2CPP_ArrayNew(typeof(T), n)` is `new T[n]` and
    // `IL2CPP_Box(typeof(T), &v)` is the value it boxes. This runs first so the
    // folds below (and the renderer) never see the raw helper shape.
    total += fold_runtime_helpers(stmts);
    // Conditional-return folding creates the ternaries the boolean fold wants.
    total += walk_lists(stmts, &mut |list| fold_conditional_returns(list, ret));
    let returned = returned_local(stmts);
    let returned = returned.as_deref();
    total += walk_lists(stmts, &mut |list| fold_ctor_calls(list, ret, returned));
    total += walk_lists(stmts, &mut |list| fold_alloc_null_checks(list));
    total += walk_lists(stmts, &mut |list| drop_duplicate_set_item(list));

    // Boolean folds and `return 0` → `null` over every expression.
    total += walk_lists(stmts, &mut |list| {
        let mut n = 0;
        for s in list.iter_mut() {
            n += fold_stmt_exprs(s, ret);
        }
        n
    });
    if ret.is_ref {
        total += walk_lists(stmts, &mut |list| {
            let mut n = 0;
            for s in list.iter_mut() {
                if matches!(s, Stmt::Return(Some(Expr::Int(0)))) {
                    *s = Stmt::Return(Some(Expr::Null));
                    n += 1;
                }
            }
            n
        });
    }
    total
}

/// Drop a bare `new Object()` / `Object..ctor()` statement at the top level of
/// a constructor body. Only the parameterless System.Object base call is
/// matched: a real allocation of another type, or any call with arguments, is
/// left alone.
fn drop_implicit_object_ctor(stmts: &mut Vec<Stmt>) -> usize {
    fn is_object_ctor(e: &Expr) -> bool {
        match e {
            Expr::NewObj(ty, args) => {
                matches!(ty.as_str(), "object" | "Object" | "System.Object") && args.is_empty()
            }
            Expr::Call { recv: None, name, args } => {
                matches!(
                    name.as_str(),
                    "object..ctor" | "Object..ctor" | "System.Object..ctor"
                ) && args.is_empty()
            }
            _ => false,
        }
    }
    let mut n = 0;
    let mut i = 0;
    while i < stmts.len() {
        if matches!(&stmts[i], Stmt::Eval(e) if is_object_ctor(e)) {
            stmts.remove(i);
            n += 1;
        } else {
            i += 1;
        }
    }
    n
}

fn fold_stmt_exprs(s: &mut Stmt, ret: &RetInfo) -> usize {
    let mut n = 0;
    let mut fold = |e: &mut Expr| {
        n += fold_bool_cond(e, ret.is_bool);
        n += invert_not_cmp(e);
    };
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => fold(e),
        Stmt::Return(Some(e)) => fold(e),
        Stmt::Assign(_, _, e) => fold(e),
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

/// Fold allocation helper calls early, before the array passes.
///
/// `apply` runs after array reconstruction, so `IL2CPP_ArrayNew(typeof(T[]),n)`
/// reached `array_init`/`array_element_sugar` as a raw call and could never
/// seed an element type. The late array passes call this first; the final
/// `apply` remains idempotent.
pub fn fold_allocation_helpers(stmts: &mut Vec<Stmt>) -> usize {
    fold_runtime_helpers(stmts)
}

/// Fold proven IL2CPP runtime-helper call shapes and name typed allocations.
fn fold_runtime_helpers(stmts: &mut Vec<Stmt>) -> usize {
    // Renaming is a readability rewrite, not a structural one; it is not
    // counted in `apply`'s return value (existing callers treat that count as
    // "folds applied").
    let _ = rename_alloc_locals(stmts);
    let mut n = 0;
    super::walk_mut(stmts, &mut |list| {
        for s in list.iter_mut() {
            n += fold_runtime_stmt(s);
        }
    });
    n
}

/// Recursively fold helper calls in every expression a statement owns.
fn fold_runtime_stmt(s: &mut Stmt) -> usize {
    let mut n = 0;
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => n += fold_runtime_expr(e),
        Stmt::Return(Some(e)) => n += fold_runtime_expr(e),
        Stmt::Assign(_, _, e) => n += fold_runtime_expr(e),
        Stmt::If { cond, .. }
        | Stmt::While { cond, .. }
        | Stmt::DoWhile { cond, .. }
        | Stmt::Lock { expr: cond, .. } => n += fold_runtime_expr(cond),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                n += fold_runtime_expr(c);
            }
        }
        Stmt::StringSwitch { disc, .. } => n += fold_runtime_expr(disc),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            n += fold_runtime_expr(collection);
            n += fold_runtime_expr(index);
            n += fold_runtime_expr(value);
        }
        Stmt::EventAssign { field, value, .. } => {
            n += fold_runtime_expr(field);
            n += fold_runtime_expr(value);
        }
        Stmt::Attribute { args, .. } => {
            for a in args {
                n += fold_runtime_expr(a);
            }
        }
        _ => {}
    }
    n
}

fn fold_runtime_expr(e: &mut Expr) -> usize {
    let mut n = 0;
    match e {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            n += fold_runtime_expr(a);
            n += fold_runtime_expr(b);
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Await(a) | Expr::Cast(_, a) => {
            n += fold_runtime_expr(a);
        }
        Expr::Field(a, _) | Expr::RawDeref(a, _, _) | Expr::RawAddr(a, _) => {
            n += fold_runtime_expr(a);
        }
        Expr::AddressOf(a) | Expr::IndexFromEnd(a) => n += fold_runtime_expr(a),
        Expr::Cond(c, a, b) => {
            n += fold_runtime_expr(c);
            n += fold_runtime_expr(a);
            n += fold_runtime_expr(b);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                n += fold_runtime_expr(r);
            }
            for a in args {
                n += fold_runtime_expr(a);
            }
        }
        Expr::Lambda { body, .. } => n += fold_runtime_expr(body),
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                n += fold_runtime_expr(a);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for i in items {
                n += fold_runtime_expr(i);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter_mut().chain(items.iter_mut()) {
                n += fold_runtime_expr(a);
            }
        }
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            for a in ctor_args {
                n += fold_runtime_expr(a);
            }
            for (_, v) in fields {
                n += fold_runtime_expr(v);
            }
        }
        Expr::Index { collection, index } => {
            n += fold_runtime_expr(collection);
            n += fold_runtime_expr(index);
        }
        Expr::NullConditional { receiver, access } => {
            n += fold_runtime_expr(receiver);
            n += fold_runtime_expr(access);
        }
        Expr::PatternMatch { expr, .. } => n += fold_runtime_expr(expr),
        Expr::StringInterp { parts } => {
            for p in parts {
                if let super::super::csharp::InterpPart::Expr(e, _) = p {
                    n += fold_runtime_expr(e);
                }
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                n += fold_runtime_expr(e);
            }
        }
        Expr::MethodChain { receiver, methods } => {
            n += fold_runtime_expr(receiver);
            for m in methods {
                for a in &mut m.args {
                    n += fold_runtime_expr(a);
                }
            }
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            n += fold_runtime_expr(discrim);
            for (p, r) in arms {
                n += fold_runtime_expr(p);
                n += fold_runtime_expr(r);
            }
            if let Some(d) = default_arm {
                n += fold_runtime_expr(d);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                n += fold_runtime_expr(f);
            }
            if let Some(t) = to {
                n += fold_runtime_expr(t);
            }
        }
        _ => {}
    }
    // `new object(typeof(T), value)`: a RESOLVED object_new whose klass
    // argument was a managed type and whose second argument is the boxed
    // value. `new object(...)` is not valid C#; the source was `(object)value`.
    // The first argument must be the type — allocations that carry a display
    // class / context pointer as arg0 (`new object(<>c.<>9, typeof(T))`) stay.
    if let Expr::NewObj(name, args) = e {
        if name == "object"
            && args.len() >= 2
            && matches!(args[0], Expr::TypeOf(_))
            && !matches!(args[1], Expr::MissingAbiArg | Expr::Unknown)
        {
            let value = args[1].clone();
            *e = Expr::Cast("object".into(), Box::new(value));
            n += 1;
            return n;
        }
    }
    if let Some(replacement) = helper_replacement(e) {
        *e = replacement;
        n += 1;
    }
    n
}

fn mentions_typeof(e: &Expr) -> bool {
    match e {
        Expr::TypeOf(_) => true,
        Expr::Field(b, _)
        | Expr::Cast(_, b)
        | Expr::AddressOf(b)
        | Expr::RawAddr(b, _)
        | Expr::RawDeref(b, _, _)
        | Expr::Not(b)
        | Expr::Neg(b) => mentions_typeof(b),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => mentions_typeof(a) || mentions_typeof(b),
        _ => false,
    }
}

/// The high-level shape of a proven helper call, or `None` when the operands
/// do not prove one. Types are taken from the call's own `typeof(T)` argument
/// (metadata evidence), never guessed.
fn helper_replacement(e: &Expr) -> Option<Expr> {
    use super::super::runtime_helpers::{self, RuntimeHelper};
    let Expr::RawCall(addr, args, _) = e else {
        return None;
    };
    match runtime_helpers::role_of(*addr)? {
        RuntimeHelper::ArrayNew => {
            let ty = args.first()?;
            let len = args.get(1)?;
            let Expr::TypeOf(elem) = ty else {
                return None;
            };
            if matches!(len, Expr::MissingAbiArg | Expr::Unknown) {
                return None;
            }
            // `il2cpp_array_new_specific` takes the ARRAY class
            // (`typeof(Keyframe[])`); `new` spells the ELEMENT type. Without
            // this strip the fold produced `new Keyframe[][n]`, so every
            // array allocation kept the raw helper call.
            let elem = elem.strip_suffix("[]").unwrap_or(elem);
            if elem.contains('[') || elem.is_empty() {
                return None; // jagged array: `Expr::NewArr` cannot spell it
            }
            Some(Expr::NewArr(elem.to_string(), vec![len.clone()]))
        }
        RuntimeHelper::Box => {
            // `box(class, &value)`: the first two arguments are authoritative
            // (the rest of the register dump is not). The value is the boxed
            // expression; `(object)value` is exactly what the C# source wrote.
            let ty = args.first()?;
            // The klass expression can be a member of the type
            // (`typeof(EnumT).value__`) — the fold to `(object)value` does not
            // depend on which form it takes, only that the type is named.
            if !matches!(ty, Expr::TypeOf(_)) && !mentions_typeof(ty) {
                return None;
            }
            let inner = match args.get(1)? {
                Expr::AddressOf(v) => match v.as_ref() {
                    Expr::Var(_) | Expr::Field(..) | Expr::StaticField(..) => (**v).clone(),
                    _ => return None,
                },
                _ => return None,
            };
            Some(Expr::Cast("object".into(), Box::new(inner)))
        }
        _ => None,
    }
}

/// Give a typed allocation local the type-derived camelCase name when the
/// normal naming pass could not (generic definitions like `List`1` do not pass
/// its identifier check). Purely lexical and unique: a candidate already used
/// by any local is rejected with a numeric suffix.
fn rename_alloc_locals(stmts: &mut Vec<Stmt>) -> usize {
    let mut used = HashSet::new();
    collect_used_names(stmts, &mut used);
    let mut candidates: Vec<(String, String)> = Vec::new();
    collect_alloc_candidates(stmts, &used, &mut candidates);
    if candidates.is_empty() {
        return 0;
    }
    let mut n = 0;
    for (old, new) in candidates {
        replace_name_in_stmts(stmts, &old, &new);
        n += 1;
    }
    n
}

fn collect_alloc_candidates(
    stmts: &[Stmt],
    used: &HashSet<String>,
    out: &mut Vec<(String, String)>,
) {
    for s in stmts {
        match s {
            Stmt::Assign(v, _, Expr::NewObj(ty, _))
                if v.starts_with('v') && v[1..].chars().all(|c| c.is_ascii_digit()) && v.len() > 1 =>
            {
                if let Some(base) = type_local_name(ty) {
                    let mut cand = base.clone();
                    let mut k = 2;
                    while used.contains(&cand) || out.iter().any(|(_, n)| n == &cand) {
                        cand = format!("{base}{k}");
                        k += 1;
                    }
                    out.push((v.clone(), cand));
                }
            }
            _ => {}
        }
        let mut kids = Vec::new();
        super::children(s, &mut kids);
        for c in kids {
            collect_alloc_candidates(c, used, out);
        }
    }
}

/// Simple camelCase identifier from a metadata type name. Generic arity
/// (`List`1`) and namespaces are stripped; anything that is not a plain
/// identifier is rejected.
fn type_local_name(ty: &str) -> Option<String> {
    let simple = ty.rsplit('.').next().unwrap_or(ty);
    let simple = simple.split('`').next().unwrap_or(simple);
    // Generic arguments are not part of the identifier: `List<object>` and
    // `Dictionary<object, object>` are just `List` / `Dictionary` here. This
    // was rejecting every generic allocation local (`new List<object>()`),
    // leaving 11.3k refs on names like v480.
    let simple = simple.split('<').next().unwrap_or(simple);
    let simple = simple.trim_end_matches("[]");
    let first = simple.chars().next()?;
    if !first.is_ascii_uppercase() {
        return None;
    }
    let out: String = first
        .to_ascii_lowercase()
        .to_string()
        .chars()
        .chain(simple.chars().skip(1))
        .collect();
    if !out
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return None;
    }
    const RESERVED: &[&str] = &[
        "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
        "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
        "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
        "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock",
        "long", "namespace", "new", "null", "object", "operator", "out", "override", "params",
        "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short",
        "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true",
        "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "virtual",
        "void", "volatile", "while",
    ];
    if RESERVED.contains(&out.as_str()) {
        return None;
    }
    Some(out)
}

fn collect_used_names(stmts: &[Stmt], used: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign(target, _, e) => {
                used.insert(target.clone());
                collect_expr_names(e, used);
            }
            Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => collect_expr_names(e, used),
            Stmt::Return(Some(e)) => collect_expr_names(e, used),
            Stmt::If { cond, .. }
            | Stmt::While { cond, .. }
            | Stmt::DoWhile { cond, .. }
            | Stmt::Lock { expr: cond, .. } => collect_expr_names(cond, used),
            Stmt::For { cond, .. } => {
                if let Some(c) = cond {
                    collect_expr_names(c, used);
                }
            }
            Stmt::StringSwitch { disc, .. } => collect_expr_names(disc, used),
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                collect_expr_names(collection, used);
                collect_expr_names(index, used);
                collect_expr_names(value, used);
            }
            Stmt::EventAssign { field, value, .. } => {
                collect_expr_names(field, used);
                collect_expr_names(value, used);
            }
            Stmt::ForEach { var_name, .. } => {
                used.insert(var_name.clone());
            }
            _ => {}
        }
        let mut kids = Vec::new();
        super::children(s, &mut kids);
        for c in kids {
            collect_used_names(c, used);
        }
    }
}

fn collect_expr_names(e: &Expr, used: &mut HashSet<String>) {
    match e {
        Expr::Var(v) => {
            used.insert(v.clone());
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            collect_expr_names(a, used);
            collect_expr_names(b, used);
        }
        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Await(a)
        | Expr::Cast(_, a)
        | Expr::Field(a, _)
        | Expr::RawDeref(a, _, _)
        | Expr::RawAddr(a, _)
        | Expr::AddressOf(a)
        | Expr::IndexFromEnd(a) => collect_expr_names(a, used),
        Expr::Cond(c, a, b) => {
            collect_expr_names(c, used);
            collect_expr_names(a, used);
            collect_expr_names(b, used);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                collect_expr_names(r, used);
            }
            for a in args {
                collect_expr_names(a, used);
            }
        }
        Expr::Lambda { body, .. } => collect_expr_names(body, used),
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                collect_expr_names(a, used);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for i in items {
                collect_expr_names(i, used);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter().chain(items.iter()) {
                collect_expr_names(a, used);
            }
        }
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            for a in ctor_args {
                collect_expr_names(a, used);
            }
            for (_, v) in fields {
                collect_expr_names(v, used);
            }
        }
        Expr::Index { collection, index } => {
            collect_expr_names(collection, used);
            collect_expr_names(index, used);
        }
        Expr::NullConditional { receiver, access } => {
            collect_expr_names(receiver, used);
            collect_expr_names(access, used);
        }
        Expr::PatternMatch { expr, .. } => collect_expr_names(expr, used),
        Expr::StringInterp { parts } => {
            for p in parts {
                if let super::super::csharp::InterpPart::Expr(e, _) = p {
                    collect_expr_names(e, used);
                }
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                collect_expr_names(e, used);
            }
        }
        Expr::MethodChain { receiver, methods } => {
            collect_expr_names(receiver, used);
            for m in methods {
                for a in &m.args {
                    collect_expr_names(a, used);
                }
            }
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            collect_expr_names(discrim, used);
            for (p, r) in arms {
                collect_expr_names(p, used);
                collect_expr_names(r, used);
            }
            if let Some(d) = default_arm {
                collect_expr_names(d, used);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                collect_expr_names(f, used);
            }
            if let Some(t) = to {
                collect_expr_names(t, used);
            }
        }
        _ => {}
    }
}

fn replace_name_in_stmts(stmts: &mut Vec<Stmt>, old: &str, new: &str) {
    for s in stmts.iter_mut() {
        match s {
            Stmt::Assign(target, _, e) => {
                if target == old {
                    *target = new.to_string();
                } else if target.starts_with(old)
                    && !target[old.len()..]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    *target = format!("{}{}", new, &target[old.len()..]);
                } else {
                    *target = replace_name_in_text(target, old, new);
                }
                replace_name_in_expr(e, old, new);
            }
            Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => replace_name_in_expr(e, old, new),
            Stmt::Return(Some(e)) => replace_name_in_expr(e, old, new),
            Stmt::If { cond, .. }
            | Stmt::While { cond, .. }
            | Stmt::DoWhile { cond, .. }
            | Stmt::Lock { expr: cond, .. } => replace_name_in_expr(cond, old, new),
            Stmt::For { cond, .. } => {
                if let Some(c) = cond {
                    replace_name_in_expr(c, old, new);
                }
            }
            Stmt::StringSwitch { disc, .. } => replace_name_in_expr(disc, old, new),
            Stmt::IndexAssign {
                collection,
                index,
                value,
            } => {
                replace_name_in_expr(collection, old, new);
                replace_name_in_expr(index, old, new);
                replace_name_in_expr(value, old, new);
            }
            Stmt::EventAssign { field, value, .. } => {
                replace_name_in_expr(field, old, new);
                replace_name_in_expr(value, old, new);
            }
            _ => {}
        }
        for c in super::children_mut(s) {
            replace_name_in_stmts(c, old, new);
        }
    }
}

fn replace_name_in_text(text: &str, old: &str, new: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(old) {
        let before_ok = rest[..pos]
            .chars()
            .next_back()
            .map_or(true, |c| !(c.is_ascii_alphanumeric() || c == '_'));
        let after = &rest[pos + old.len()..];
        let after_ok = after
            .chars()
            .next()
            .map_or(true, |c| !(c.is_ascii_alphanumeric() || c == '_'));
        out.push_str(&rest[..pos]);
        if before_ok && after_ok {
            out.push_str(new);
        } else {
            out.push_str(old);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

fn replace_name_in_expr(e: &mut Expr, old: &str, new: &str) {
    if let Expr::Var(v) = e {
        if v == old {
            *v = new.to_string();
        }
        return;
    }
    match e {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            replace_name_in_expr(a, old, new);
            replace_name_in_expr(b, old, new);
        }
        Expr::Not(a)
        | Expr::Neg(a)
        | Expr::Await(a)
        | Expr::Cast(_, a)
        | Expr::Field(a, _)
        | Expr::RawDeref(a, _, _)
        | Expr::RawAddr(a, _)
        | Expr::AddressOf(a)
        | Expr::IndexFromEnd(a) => replace_name_in_expr(a, old, new),
        Expr::Cond(c, a, b) => {
            replace_name_in_expr(c, old, new);
            replace_name_in_expr(a, old, new);
            replace_name_in_expr(b, old, new);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                replace_name_in_expr(r, old, new);
            }
            for a in args {
                replace_name_in_expr(a, old, new);
            }
        }
        Expr::Lambda { body, .. } => replace_name_in_expr(body, old, new),
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                replace_name_in_expr(a, old, new);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for i in items {
                replace_name_in_expr(i, old, new);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter_mut().chain(items.iter_mut()) {
                replace_name_in_expr(a, old, new);
            }
        }
        Expr::ObjInit {
            ctor_args, fields, ..
        } => {
            for a in ctor_args {
                replace_name_in_expr(a, old, new);
            }
            for (_, v) in fields {
                replace_name_in_expr(v, old, new);
            }
        }
        Expr::Index { collection, index } => {
            replace_name_in_expr(collection, old, new);
            replace_name_in_expr(index, old, new);
        }
        Expr::NullConditional { receiver, access } => {
            replace_name_in_expr(receiver, old, new);
            replace_name_in_expr(access, old, new);
        }
        Expr::PatternMatch { expr, .. } => replace_name_in_expr(expr, old, new),
        Expr::StringInterp { parts } => {
            for p in parts {
                if let super::super::csharp::InterpPart::Expr(e, _) = p {
                    replace_name_in_expr(e, old, new);
                }
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                replace_name_in_expr(e, old, new);
            }
        }
        Expr::MethodChain { receiver, methods } => {
            replace_name_in_expr(receiver, old, new);
            for m in methods {
                for a in &mut m.args {
                    replace_name_in_expr(a, old, new);
                }
            }
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            replace_name_in_expr(discrim, old, new);
            for (p, r) in arms {
                replace_name_in_expr(p, old, new);
                replace_name_in_expr(r, old, new);
            }
            if let Some(d) = default_arm {
                replace_name_in_expr(d, old, new);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                replace_name_in_expr(f, old, new);
            }
            if let Some(t) = to {
                replace_name_in_expr(t, old, new);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn text(stmts: &[Stmt]) -> String {
        format!("{stmts:?}")
    }

    #[test]
    fn ctor_drops_implicit_object_base_call() {
        let mut stmts = vec![
            Stmt::Eval(Expr::NewObj("Object".into(), vec![])),
            Stmt::Return(None),
        ];
        let n = apply(
            &mut stmts,
            &RetInfo {
                is_ctor: true,
                ..Default::default()
            },
        );
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1, "{}", text(&stmts));
    }

    #[test]
    fn non_ctor_keeps_object_allocation() {
        let mut stmts = vec![
            Stmt::Eval(Expr::NewObj("Object".into(), vec![])),
            Stmt::Return(None),
        ];
        let n = apply(&mut stmts, &RetInfo::default());
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn array_new_helper_folds_to_new_array() {
        use crate::recovery::runtime_helpers::{self, RuntimeHelper};
        runtime_helpers::note(0xabc100, RuntimeHelper::ArrayNew);
        let mut stmts = vec![Stmt::Assign(
            "arr".into(),
            None,
            Expr::RawCall(
                0xabc100,
                vec![
                    Expr::TypeOf("CoinChainDebug".into()),
                    Expr::Int(4),
                    Expr::Var("v70".into()),
                ],
                None,
            ),
        )];
        assert!(apply(&mut stmts, &RetInfo::default()) >= 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::NewArr(elem, dims)) => {
                assert_eq!(elem, "CoinChainDebug");
                assert!(matches!(dims.as_slice(), [Expr::Int(4)]));
            }
            other => panic!("expected new[] fold, got {other:?}"),
        }
    }

    #[test]
    fn array_new_helper_folds_a_szarray_class_token_to_the_element_allocation() {
        use crate::recovery::runtime_helpers::{self, RuntimeHelper};
        runtime_helpers::note(0xabc102, RuntimeHelper::ArrayNew);
        let mut stmts = vec![Stmt::Assign(
            "arr".into(),
            None,
            Expr::RawCall(
                0xabc102,
                vec![
                    Expr::TypeOf("UnityEngine.Keyframe[]".into()),
                    Expr::Int(2),
                ],
                None,
            ),
        )];
        assert!(apply(&mut stmts, &RetInfo::default()) >= 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::NewArr(elem, dims)) => {
                assert_eq!(elem, "UnityEngine.Keyframe");
                assert!(matches!(dims.as_slice(), [Expr::Int(2)]));
            }
            other => panic!("expected new[] fold, got {other:?}"),
        }
    }

    #[test]
    fn a_jagged_class_token_is_not_folded_into_an_unspellable_new() {
        use crate::recovery::runtime_helpers::{self, RuntimeHelper};
        runtime_helpers::note(0xabc103, RuntimeHelper::ArrayNew);
        let mut stmts = vec![Stmt::Assign(
            "arr".into(),
            None,
            Expr::RawCall(
                0xabc103,
                vec![Expr::TypeOf("float[][]".into()), Expr::Int(8)],
                None,
            ),
        )];
        assert_eq!(apply(&mut stmts, &RetInfo::default()), 0);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::RawCall(..))));
    }

    #[test]
    fn box_helper_folds_to_object_value() {
        use crate::recovery::runtime_helpers::{self, RuntimeHelper};
        runtime_helpers::note(0xabc101, RuntimeHelper::Box);
        let mut stmts = vec![Stmt::Assign(
            "v1".into(),
            None,
            Expr::RawCall(
                0xabc101,
                vec![
                    Expr::TypeOf("DeathTypes".into()),
                    Expr::AddressOf(Box::new(Expr::Var("value".into()))),
                    Expr::Var("v70".into()),
                ],
                None,
            ),
        )];
        assert!(apply(&mut stmts, &RetInfo::default()) >= 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Cast(ty, inner)) => {
                assert_eq!(ty, "object");
                assert!(matches!(inner.as_ref(), Expr::Var(v) if v == "value"));
            }
            other => panic!("expected box fold, got {other:?}"),
        }
    }

    #[test]
    fn box_helper_without_an_address_stays_a_call() {
        use crate::recovery::runtime_helpers::{self, RuntimeHelper};
        runtime_helpers::note(0xabc102, RuntimeHelper::Box);
        let call = Expr::RawCall(
            0xabc102,
            vec![
                Expr::TypeOf("DeathTypes".into()),
                Expr::Var("v9".into()),
            ],
            None,
        );
        let mut stmts = vec![Stmt::Assign("v1".into(), None, call.clone())];
        apply(&mut stmts, &RetInfo::default());
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, e) if e == &call));
    }

    #[test]
    fn typed_allocation_local_takes_the_type_name() {
        let mut stmts = vec![Stmt::Assign(
            "v480".into(),
            None,
            Expr::NewObj("List`1".into(), vec![]),
        )];
        apply(&mut stmts, &RetInfo::default());
        assert!(
            matches!(&stmts[0], Stmt::Assign(n, _, _) if n == "list"),
            "{stmts:?}"
        );
    }

    #[test]
    fn generic_allocation_type_strips_the_arguments() {
        let mut stmts = vec![Stmt::Assign(
            "v480".into(),
            None,
            Expr::NewObj("System.Collections.Generic.List<object>".into(), vec![]),
        )];
        apply(&mut stmts, &RetInfo::default());
        assert!(
            matches!(&stmts[0], Stmt::Assign(n, _, _) if n == "list"),
            "{stmts:?}"
        );
    }

    #[test]
    fn typed_allocation_with_ctor_args_keeps_the_args() {
        let mut stmts = vec![
            Stmt::Assign("v7".into(), None, Expr::NewObj("Widget".into(), vec![])),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("v7"))),
                name: "..ctor".into(),
                args: vec![var("a"), var("b")],
            }),
        ];
        apply(&mut stmts, &RetInfo::default());
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::NewObj(ty, args)) => {
                assert_eq!(ty, "Widget");
                assert_eq!(args.len(), 2, "{stmts:?}");
            }
            other => panic!("expected merged ctor args, got {other:?}"),
        }
        assert_eq!(stmts.len(), 1, "{stmts:?}");
    }

    #[test]
    fn both_arms_return_folds_to_conditional() {
        let cond = Expr::Cmp(CmpKind::Lt, Box::new(var("n")), Box::new(Expr::Int(1)));
        let mut stmts = vec![
            Stmt::If {
                cond: cond.clone(),
                then_body: vec![Stmt::Assign("v".into(), None, Expr::Int(0))],
                else_body: vec![Stmt::Assign("v".into(), None, Expr::Cmp(
                    CmpKind::Gt,
                    Box::new(var("p")),
                    Box::new(Expr::Float(0.0)),
                ))],
            },
            Stmt::Return(Some(var("v"))),
        ];
        let ret = RetInfo { is_bool: true, ..Default::default() };
        assert!(apply(&mut stmts, &ret) >= 1);
        let t = text(&stmts);
        assert!(t.contains("AndAlso"), "{t}");
        assert!(t.contains("Ge"), "condition should be inverted: {t}");
    }

    #[test]
    fn single_arm_return_folds() {
        let mut stmts = vec![
            Stmt::Assign("v".into(), None, Expr::Int(0)),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(var("x")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Assign("v".into(), None, var("y"))],
                else_body: vec![],
            },
            Stmt::Return(Some(var("v"))),
        ];
        let ret = RetInfo::default();
        assert_eq!(apply(&mut stmts, &ret), 1);
        assert!(text(&stmts).contains("Cond"));
    }

    #[test]
    fn ctor_call_is_merged_and_typed_from_return() {
        let mut stmts = vec![
            Stmt::Assign(
                "v".into(),
                None,
                Expr::NewObj("object".into(), vec![var("t")]),
            ),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("v"))),
                name: "..ctor".into(),
                args: vec![var("t")],
            }),
            Stmt::Return(Some(var("v"))),
        ];
        let ret = RetInfo { is_ref: true, type_name: Some("RewardData".into()), ..Default::default() };
        assert!(apply(&mut stmts, &ret) >= 1);
        let t = text(&stmts);
        assert!(t.contains("RewardData"), "{t}");
        assert!(!t.contains("..ctor"), "{t}");
    }

    #[test]
    fn return_zero_becomes_null_for_reference() {
        let mut stmts = vec![Stmt::Return(Some(Expr::Int(0)))];
        let ret = RetInfo { is_ref: true, ..Default::default() };
        assert_eq!(apply(&mut stmts, &ret), 1);
        assert!(matches!(&stmts[0], Stmt::Return(Some(Expr::Null))));
    }

    #[test]
    fn duplicate_set_item_is_dropped() {
        let mut stmts = vec![
            Stmt::IndexAssign {
                collection: var("c"),
                index: var("i"),
                value: var("v"),
            },
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("c"))),
                name: "set_Item".into(),
                args: vec![var("i"), var("v")],
            }),
        ];
        assert_eq!(apply(&mut stmts, &RetInfo::default()), 1);
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn alloc_null_check_is_spliced() {
        let mut stmts = vec![
            Stmt::Assign(
                "v33".into(),
                None,
                Expr::NewObj("RewardData".into(), vec![var("t")]),
            ),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(var("v33")), Box::new(Expr::Null)),
                then_body: vec![Stmt::Return(Some(var("v33")))],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts, &RetInfo { is_ref: true, ..Default::default() });
        assert_eq!(n, 1, "{stmts:?}");
        assert_eq!(stmts.len(), 2, "{stmts:?}");
    }
}
