//! Inline constant and stable-alias locals at every read.
//!
//! ## Constant locals
//!
//! A local whose definitions are all the same pure constant (`0f`, `12`,
//! `null`, `typeof(T)`, …) carries no information: replacing each read with
//! that constant and deleting the definitions is value-identical and removes a
//! temp. Measured on 0x17F2380: `v435 = 0f` (two defs, 35 reads); on
//! 0x146A0E4: `v120 = 12` (80 reads), `v121 = typeof(Vector3)` (50 reads).
//!
//! Definitions are not always textually identical. The renderer keeps the
//! SSA-destruction copies of a value that never changes:
//!
//! ```text
//! v120 = 12;      // seed
//! ... v154 = v120; ...
//! v120 = v154;    // join copies the value back
//! ```
//!
//! Such definitions form an equality component: every alias edge must agree
//! with the component's single constant, so `v120` and `v154` are both `12`.
//! Components are resolved with union-find; a component stays unresolved when
//! it contains a name with no known definitions (a parameter or an impure
//! def), which is what keeps `v = 12; v = ptr;` from folding to 12.
//!
//! ## Stable aliases
//!
//! A local whose every definition is a copy of the SAME source name is a
//! pure alias: `v274 = v227` twice (0x1727E90) says nothing the source does
//! not. It folds only when the source cannot change value: an unassigned name
//! (parameter, `foreach` variable) or a name with one top-level pure
//! definition. A source that is itself folded as an alias is left alone in
//! this round so no read can reference a deleted definition.
//!
//! ## Refusals
//!
//! Generated display names (`vN`, `objN`, `ptrN`, `numN`, `flagN`) are the
//! foldable class; metadata-named locals are not. A name used as an ADDRESS
//! (`&vN`, `(nint)&vN`) or in a name-only slot (`switch (vN)`, `foreach (… in
//! vN)`, `using` declaration, attribute argument) is left untouched. A
//! constant may not become a receiver (`0f.Field`, `12.Method()`) — those uses
//! keep the temp; an alias may, because a name is a valid receiver. A value
//! use inside a complex assignment target (`*(float*)(i * v120 + array) = …`)
//! is rewritten in place with the constant spelling (or the alias name).
//!
//! The constant is pure, so duplicating it at reads changes no evaluation
//! effects. A read that occurs before any definition was already an
//! undefined-local bug; folding keeps it visible as the source expression.

use super::super::csharp::{Expr, Stmt};
use super::children_mut;
use std::collections::{HashMap, HashSet};

fn is_temp_name(n: &str) -> bool {
    // All generated display classes share the same proof: they are SSA
    // locals, never metadata names. Restricting this to `vN` left the same
    // alias behind whenever local naming had called it `objN`/`ptrN`/`numN`
    // (`var obj = j;`, `var ptr2 = in_x2;`), which is the bulk of the
    // remaining trivial copies.
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if let Some(rest) = n.strip_prefix('v') {
        return digits(rest);
    }
    for prefix in ["obj", "ptr", "num", "flag"] {
        if let Some(rest) = n.strip_prefix(prefix) {
            if digits(rest) {
                return true;
            }
        }
    }
    false
}

fn is_pure_const(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Int(_)
            | Expr::Float(_)
            | Expr::Double(_)
            | Expr::Bool(_)
            | Expr::Null
            | Expr::Str(_)
            | Expr::TypeOf(_)
            | Expr::MethodGroup(_)
    )
}

fn expr_children(e: &Expr) -> Vec<&Expr> {
    let mut v: Vec<&Expr> = Vec::new();
    match e {
        Expr::Field(b, _)
        | Expr::RawDeref(b, _, _)
        | Expr::RawAddr(b, _)
        | Expr::AddressOf(b)
        | Expr::Cast(_, b)
        | Expr::Not(b)
        | Expr::Neg(b)
        | Expr::Await(b)
        | Expr::IndexFromEnd(b) => v.push(b),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => v.extend([a.as_ref(), b.as_ref()]),
        Expr::Cond(c, a, b) => v.extend([c.as_ref(), a.as_ref(), b.as_ref()]),
        Expr::Index { collection, index } => v.extend([collection.as_ref(), index.as_ref()]),
        Expr::NullConditional { receiver, access } => {
            v.extend([receiver.as_ref(), access.as_ref()]);
        }
        Expr::Call { recv, args, .. } => {
            v.extend(args.iter());
            if let Some(r) = recv {
                v.push(r);
            }
        }
        Expr::RawCall(_, args, _)
        | Expr::NewObj(_, args)
        | Expr::NewArr(_, args)
        | Expr::ArrayInit { items: args, .. } => v.extend(args.iter()),
        Expr::CollectionInit {
            ctor_args, items, ..
        } => v.extend(ctor_args.iter().chain(items.iter())),
        Expr::ObjInit {
            ctor_args,
            fields,
            ..
        } => {
            v.extend(ctor_args.iter());
            v.extend(fields.iter().map(|(_, x)| x));
        }
        Expr::PatternMatch { expr, .. } => v.push(expr),
        Expr::StringInterp { parts } => {
            for p in parts {
                if let super::super::csharp::InterpPart::Expr(x, _) = p {
                    v.push(x);
                }
            }
        }
        Expr::Tuple { elements, .. } => v.extend(elements.iter()),
        Expr::MethodChain { receiver, methods } => {
            v.push(receiver);
            v.extend(methods.iter().flat_map(|m| m.args.iter()));
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            v.push(discrim);
            for (pat, res) in arms {
                v.push(pat);
                v.push(res);
            }
            if let Some(d) = default_arm {
                v.push(d);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                v.push(f);
            }
            if let Some(t) = to {
                v.push(t);
            }
        }
        Expr::Lambda { body, .. } => v.push(body),
        _ => {}
    }
    v
}

fn expr_children_mut(e: &mut Expr) -> Vec<&mut Expr> {
    let mut v: Vec<&mut Expr> = Vec::new();
    match e {
        Expr::Field(b, _)
        | Expr::RawDeref(b, _, _)
        | Expr::RawAddr(b, _)
        | Expr::AddressOf(b)
        | Expr::Cast(_, b)
        | Expr::Not(b)
        | Expr::Neg(b)
        | Expr::Await(b)
        | Expr::IndexFromEnd(b) => v.push(b),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => v.extend([a.as_mut(), b.as_mut()]),
        Expr::Cond(c, a, b) => v.extend([c.as_mut(), a.as_mut(), b.as_mut()]),
        Expr::Index { collection, index } => {
            v.extend([collection.as_mut(), index.as_mut()]);
        }
        Expr::NullConditional { receiver, access } => {
            v.extend([receiver.as_mut(), access.as_mut()]);
        }
        Expr::Call { recv, args, .. } => {
            v.extend(args.iter_mut());
            if let Some(r) = recv {
                v.push(r);
            }
        }
        Expr::RawCall(_, args, _)
        | Expr::NewObj(_, args)
        | Expr::NewArr(_, args)
        | Expr::ArrayInit { items: args, .. } => v.extend(args.iter_mut()),
        Expr::CollectionInit {
            ctor_args, items, ..
        } => v.extend(ctor_args.iter_mut().chain(items.iter_mut())),
        Expr::ObjInit {
            ctor_args,
            fields,
            ..
        } => {
            v.extend(ctor_args.iter_mut());
            v.extend(fields.iter_mut().map(|(_, x)| x));
        }
        Expr::PatternMatch { expr, .. } => v.push(expr),
        Expr::StringInterp { parts } => {
            for p in parts {
                if let super::super::csharp::InterpPart::Expr(x, _) = p {
                    v.push(x);
                }
            }
        }
        Expr::Tuple { elements, .. } => v.extend(elements.iter_mut()),
        Expr::MethodChain { receiver, methods } => {
            v.push(receiver);
            for m in methods.iter_mut() {
                v.extend(m.args.iter_mut());
            }
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            v.push(discrim);
            for (pat, res) in arms.iter_mut() {
                v.push(pat);
                v.push(res);
            }
            if let Some(d) = default_arm {
                v.push(d);
            }
        }
        Expr::Range { from, to } => {
            if let Some(f) = from {
                v.push(f);
            }
            if let Some(t) = to {
                v.push(t);
            }
        }
        Expr::Lambda { body, .. } => v.push(body),
        _ => {}
    }
    v
}

fn direct_exprs(s: &Stmt) -> Vec<&Expr> {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Return(Some(e)) => vec![e],
        Stmt::Return(None) => vec![],
        Stmt::Assign(_, _, e) => vec![e],
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => vec![cond],
        Stmt::For { cond, .. } => cond.iter().collect(),
        Stmt::Lock { expr, .. } => vec![expr],
        Stmt::StringSwitch { disc, .. } => vec![disc],
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => vec![collection, index, value],
        Stmt::EventAssign { field, value, .. } => vec![field, value],
        _ => vec![],
    }
}

/// Every identifier token in a target/discriminant string with its byte span,
/// so refusals do not depend on whether the definition was scanned yet.
fn ident_tokens(text: &str) -> Vec<(usize, usize, &str)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let tok = &text[start..i];
            if is_temp_name(tok) {
                out.push((start, i, tok));
            }
            continue;
        }
        i += 1;
    }
    out
}

/// A temp token inside a complex target string. An address (`&vN`) can never
/// be replaced. A member/index/call base (`vN.f`, `vN[`, `vN(`) is a valid
/// position for an alias name but not for a constant literal.
fn target_token_class(text: &str, start: usize, end: usize) -> TokenUse {
    let prev = text[..start].chars().last();
    let next = text[end..].chars().next();
    if prev == Some('&') {
        return TokenUse::Address;
    }
    if prev == Some('.') || matches!(next, Some('.') | Some('[') | Some('(')) {
        return TokenUse::Receiver;
    }
    TokenUse::Value
}

enum TokenUse {
    Address,
    Receiver,
    Value,
}

fn note_mentions(
    text: &str,
    bad_shared: &mut HashSet<String>,
    bad_const: &mut HashSet<String>,
    address_taken: &mut HashSet<String>,
) {
    for (start, end, tok) in ident_tokens(text) {
        match target_token_class(text, start, end) {
            TokenUse::Address => {
                bad_shared.insert(tok.to_string());
                address_taken.insert(tok.to_string());
            }
            TokenUse::Receiver => {
                bad_const.insert(tok.to_string());
            }
            TokenUse::Value => {}
        }
    }
}

/// One simple-name definition: the RHS, whether it sits at the method's top
/// level, and whether it is a pure constant.
struct DefInfo {
    top: bool,
    pure: bool,
}

#[derive(Default)]
struct Collected {
    /// Every simple-name assignment in the method.
    assigns: HashMap<String, Vec<DefInfo>>,
    /// Temp names whose defs are all pure constants or bare copies; the raw
    /// def list is the constraint system `resolve_consts` solves.
    defs: HashMap<String, Vec<Expr>>,
    /// Temp names with at least one def of another shape (value unknown).
    unknown: HashSet<String>,
    /// Names that may never be folded (address escapes, name-only slots).
    bad_shared: HashSet<String>,
    /// Names that may not become a literal receiver (`0f.Field`).
    bad_const: HashSet<String>,
    /// Names whose memory is written through their address; their value can
    /// change without an `Assign`, so they are never a stable alias source.
    address_taken: HashSet<String>,
}

/// Collect qualified names and reject address/name-slot uses. Value uses in a
/// complex target string are allowed: `rewrite_stmt` spells the replacement
/// into the target.
fn scan(stmts: &[Stmt], c: &mut Collected, depth: usize) {
    for s in stmts {
        match s {
            Stmt::Assign(t, _, rhs) => {
                let pure = is_pure_const(rhs) || matches!(rhs, Expr::Var(_));
                if is_temp_name(t) {
                    c.assigns.entry(t.clone()).or_default().push(DefInfo {
                        top: depth == 0,
                        pure,
                    });
                    if pure && !c.unknown.contains(t) {
                        c.defs.entry(t.clone()).or_default().push(rhs.clone());
                    } else {
                        c.defs.remove(t);
                        c.unknown.insert(t.clone());
                    }
                } else if is_simple_name(t) {
                    c.assigns.entry(t.clone()).or_default().push(DefInfo {
                        top: depth == 0,
                        pure,
                    });
                } else {
                    // A complex target may root at any temp; address roots and
                    // constant member bases are refused, bare value roots are
                    // rewritten.
                    note_mentions(
                        t,
                        &mut c.bad_shared,
                        &mut c.bad_const,
                        &mut c.address_taken,
                    );
                }
            }
            Stmt::ForEach {
                var_name,
                collection,
                ..
            } => {
                // The loop header rebinds the variable every iteration; it is
                // not a stable source for an alias.
                c.assigns.entry(var_name.clone()).or_default().push(DefInfo {
                    top: false,
                    pure: false,
                });
                for (_, _, tok) in ident_tokens(collection) {
                    c.bad_shared.insert(tok.to_string());
                }
            }
            Stmt::Switch { disc, .. } => {
                for (_, _, tok) in ident_tokens(disc) {
                    c.bad_shared.insert(tok.to_string());
                }
            }
            Stmt::Using { decl, .. } => {
                // `decl` is not part of `children`; a local bound there can be
                // read before any scanned def, so refuse any temp it mentions
                // and record its binding as a definition.
                let mut names: Vec<String> = Vec::new();
                stmt_mentions(decl, &mut names);
                c.bad_shared.extend(names);
                if let Stmt::Assign(n, _, rhs) = decl.as_ref() {
                    if is_simple_name(n) {
                        c.assigns.entry(n.clone()).or_default().push(DefInfo {
                                top: false,
                            pure: is_pure_const(rhs) || matches!(rhs, Expr::Var(_)),
                        });
                    }
                }
            }
            Stmt::Attribute { args, .. } => {
                for a in args {
                    let mut names: Vec<String> = Vec::new();
                    expr_var_names(a, &mut names);
                    c.bad_shared.extend(names);
                }
            }
            _ => {}
        }
        // Receiver-position slots that live on a STATEMENT (not inside an
        // `Expr::Index`/`Field`): a literal collection/event receiver would
        // render as `5[i] = v` / `5 += h`.
        match s {
            Stmt::IndexAssign { collection, .. } => {
                refuse_direct(collection, &mut c.bad_const);
            }
            Stmt::EventAssign { field, .. } => {
                refuse_direct(field, &mut c.bad_const);
            }
            _ => {}
        }
        for e in direct_exprs(s) {
            scan_expr(e, c);
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            scan(k, c, depth + 1);
        }
    }
}

fn is_simple_name(n: &str) -> bool {
    !n.is_empty()
        && !n.contains('.')
        && !n.contains('[')
        && !n.contains('*')
        && !n.contains('&')
        && !n.contains(' ')
        && !n.chars().next().is_some_and(|c| c.is_ascii_digit())
}

fn stmt_mentions(s: &Stmt, out: &mut Vec<String>) {
    for e in direct_exprs(s) {
        expr_var_names(e, out);
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            stmt_mentions(x, out);
        }
    }
}

fn expr_var_names(e: &Expr, out: &mut Vec<String>) {
    if let Expr::Var(n) = e {
        if is_temp_name(n) {
            out.push(n.clone());
        }
    }
    if let Expr::Lambda { params, .. } = e {
        // A lambda parameter shadows any outer local with the same spelling.
        out.extend(params.iter().filter(|p| is_temp_name(p)).cloned());
    }
    for c in expr_children(e) {
        expr_var_names(c, out);
    }
}

fn refuse_direct(e: &Expr, bad: &mut HashSet<String>) {
    if let Expr::Var(n) = e {
        if is_temp_name(n) {
            bad.insert(n.clone());
        }
    }
}

fn scan_expr(e: &Expr, c: &mut Collected) {
    match e {
        // The cell's address escapes.
        Expr::AddressOf(inner) | Expr::RawAddr(inner, _) => {
            refuse_direct(inner, &mut c.bad_shared);
            if let Expr::Var(n) = inner.as_ref() {
                c.address_taken.insert(n.clone());
            }
        }
        // A constant would become a receiver (`0f.Field`, `12.Method()`,
        // `5[0]` — not valid C#); an alias name is fine.
        Expr::Field(base, _) => refuse_direct(base, &mut c.bad_const),
        Expr::Call { recv: Some(r), .. } => refuse_direct(r, &mut c.bad_const),
        Expr::Index { collection, .. } => refuse_direct(collection, &mut c.bad_const),
        Expr::NullConditional { receiver, .. } => refuse_direct(receiver, &mut c.bad_const),
        Expr::MethodChain { receiver, .. } => refuse_direct(receiver, &mut c.bad_const),
        Expr::Await(inner) => refuse_direct(inner, &mut c.bad_const),
        Expr::Lambda { params, .. } => {
            for p in params {
                if is_temp_name(p) {
                    c.bad_shared.insert(p.clone());
                }
            }
        }
        _ => {}
    }
    for child in expr_children(e) {
        scan_expr(child, c);
    }
}

/// Union-find over temp names joined by alias definitions.
struct Dsu {
    parent: HashMap<String, String>,
}

impl Dsu {
    fn find(&mut self, x: &str) -> String {
        let mut root = x.to_string();
        while let Some(p) = self.parent.get(&root) {
            if *p == root {
                break;
            }
            root = p.clone();
        }
        let mut cur = x.to_string();
        while let Some(p) = self.parent.get(&cur).cloned() {
            if p == cur || p == root {
                break;
            }
            self.parent.insert(cur.clone(), root.clone());
            cur = p;
        }
        root
    }

    fn union(&mut self, a: &str, b: &str) {
        self.parent.entry(a.to_string()).or_insert_with(|| a.to_string());
        self.parent.entry(b.to_string()).or_insert_with(|| b.to_string());
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(rb, ra);
        }
    }
}

/// Solve `name = const` / `name = other` constraints per equality component.
/// A component resolves only when every referenced name has known definitions
/// (closed) and all of its constants are equal.
fn resolve_consts(defs: &HashMap<String, Vec<Expr>>) -> HashMap<String, Expr> {
    let mut dsu = Dsu {
        parent: HashMap::new(),
    };
    for (name, ds) in defs {
        for d in ds {
            if let Expr::Var(src) = d {
                dsu.union(name, src);
            }
        }
    }
    let mut consts: HashMap<String, Vec<Expr>> = HashMap::new();
    let mut open: HashSet<String> = HashSet::new();
    for (name, ds) in defs {
        let root = dsu.find(name);
        for d in ds {
            match d {
                Expr::Var(src) => {
                    if !defs.contains_key(src) {
                        open.insert(root.clone());
                    }
                }
                _ => consts.entry(root.clone()).or_default().push(d.clone()),
            }
        }
    }
    let mut out = HashMap::new();
    for name in defs.keys() {
        let root = dsu.find(name);
        if open.contains(&root) {
            continue;
        }
        let Some(cs) = consts.get(&root) else {
            continue;
        };
        let first = &cs[0];
        if cs.iter().all(|c| c == first) {
            out.insert(name.clone(), first.clone());
        }
    }
    out
}

/// A source that cannot change value inside this method: never assigned, or
/// assigned exactly once at top level with a pure expression.
fn stable_source(src: &str, c: &Collected, removed: &HashSet<String>) -> bool {
    if removed.contains(src) || c.address_taken.contains(src) {
        return false;
    }
    match c.assigns.get(src) {
        None => true,
        Some(ds) => ds.len() == 1 && ds[0].top && ds[0].pure,
    }
}

/// Names whose every def is a copy of the same stable source.
fn resolve_aliases(c: &Collected, removed: &HashSet<String>) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::new();
    for (name, ds) in &c.defs {
        if removed.contains(name) || c.bad_shared.contains(name) {
            continue;
        }
        let mut src: Option<&str> = None;
        let mut ok = true;
        for d in ds {
            match d {
                Expr::Var(s) if s != name => match src {
                    None => src = Some(s),
                    Some(prev) if prev == s.as_str() => {}
                    _ => {
                        ok = false;
                        break;
                    }
                },
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let Some(src) = src else { continue };
        if stable_source(src, c, removed) {
            out.insert(name.clone(), src.to_string());
        }
    }
    // Do not alias to a name that is itself folded in this round: its
    // definitions disappear, so the substituted read would dangle.
    let alias_keys: HashSet<String> = out.keys().cloned().collect();
    out.retain(|_, src| !alias_keys.contains(src));
    out
}

fn subst_expr(e: &mut Expr, map: &HashMap<String, Expr>) -> usize {
    let mut n = 0;
    if let Expr::Var(name) = e {
        if let Some(c) = map.get(name) {
            *e = c.clone();
            return 1;
        }
    }
    for c in expr_children_mut(e) {
        n += subst_expr(c, map);
    }
    n
}

fn rewrite_target(text: &str, map: &HashMap<String, Expr>) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    let mut n = 0;
    let mut last = 0;
    for (start, end, tok) in ident_tokens(text) {
        if let Some(c) = map.get(tok) {
            out.push_str(&text[last..start]);
            let mut rendered = String::new();
            c.render(&mut rendered);
            out.push_str(&rendered);
            last = end;
            n += 1;
        }
    }
    if n == 0 {
        return (text.to_string(), 0);
    }
    out.push_str(&text[last..]);
    (out, n)
}

fn rewrite_stmt(s: &mut Stmt, map: &HashMap<String, Expr>) -> usize {
    let mut n = 0;
    for k in children_mut(s) {
        for x in k.iter_mut() {
            n += rewrite_stmt(x, map);
        }
    }
    if let Stmt::Assign(t, _, _) = s {
        // Only COMPLEX targets embed the name as an operand
        // (`*(nint*)(v5 +16) = …`). A simple target IS the folded definition
        // and is deleted by `drop_defs`; rewriting it would turn
        // `v5 = 0f;` into `0f = 0f;`.
        if !is_simple_name(t) {
            let (nt, k) = rewrite_target(t, map);
            if k > 0 {
                *t = nt;
                n += k;
            }
        }
    }
    let mut fold = |e: &mut Expr| n += subst_expr(e, map);
    match s {
        Stmt::Eval(e)
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e)
        | Stmt::Return(Some(e))
        | Stmt::Assign(_, _, e) => fold(e),
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

fn drop_defs(stmts: &mut Vec<Stmt>, map: &HashMap<String, Expr>) -> usize {
    let mut n = 0;
    for s in stmts.iter_mut() {
        for k in children_mut(s) {
            n += drop_defs(k, map);
        }
    }
    let before = stmts.len();
    stmts.retain(|s| !matches!(s, Stmt::Assign(t, _, _) if map.contains_key(t)));
    n + before - stmts.len()
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut c = Collected::default();
    scan(stmts, &mut c, 0);
    let consts = resolve_consts(&c.defs);
    let const_keys: HashSet<String> = consts.keys().cloned().collect();
    let aliases = resolve_aliases(&c, &const_keys);

    let mut map: HashMap<String, Expr> = HashMap::new();
    for (name, val) in consts {
        if c.bad_shared.contains(&name) || c.bad_const.contains(&name) {
            continue;
        }
        map.insert(name, val);
    }
    for (name, src) in aliases {
        if map.contains_key(&name) {
            continue;
        }
        map.insert(name, Expr::Var(src));
    }
    if map.is_empty() {
        return 0;
    }
    let removed: HashSet<String> = map.keys().cloned().collect();
    debug_assert!(map
        .values()
        .all(|v| !matches!(v, Expr::Var(n) if removed.contains(n))));
    let mut n = 0;
    for s in stmts.iter_mut() {
        n += rewrite_stmt(s, &map);
    }
    n += drop_defs(stmts, &map);
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::csharp::{BinOp, CmpKind};


    #[test]
    fn same_constant_defs_fold_and_disappear() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Float(0.0)),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Assign("v1".into(), None, Expr::Float(0.0))],
                else_body: vec![],
            },
            Stmt::Eval(Expr::Var("v1".into())),
        ];
        assert!(apply(&mut stmts) >= 2);
        let t = format!("{stmts:?}");
        assert!(!t.contains("\"v1\""), "{t}");
        assert!(!t.contains("Assign("), "{t}");
        assert!(t.contains("Float"), "{t}");
    }

    #[test]
    fn different_constant_defs_are_refused() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Float(0.0)),
            Stmt::Assign("v1".into(), None, Expr::Float(1.0)),
            Stmt::Eval(Expr::Var("v1".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn alias_cycle_agrees_with_the_constant() {
        // v1 = 12; ... v2 = v1; ... v1 = v2;  → both are 12.
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(12)),
            Stmt::Assign("v2".into(), None, Expr::Var("v1".into())),
            Stmt::Assign("v1".into(), None, Expr::Var("v2".into())),
            Stmt::Eval(Expr::Bin(
                BinOp::Mul,
                Box::new(Expr::Var("v1".into())),
                Box::new(Expr::Var("v2".into())),
            )),
        ];
        assert!(apply(&mut stmts) >= 1);
        let t = format!("{stmts:?}");
        assert!(!t.contains("\"v1\""), "{t}");
        assert!(!t.contains("\"v2\""), "{t}");
        assert!(!t.contains("Assign("), "{t}");
        assert!(t.contains("Int(12)"), "{t}");
    }

    #[test]
    fn alias_to_unknown_source_keeps_the_constant_unresolved() {
        // v1 = 12; v1 = param;  → cannot fold.
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(12)),
            Stmt::Assign("v1".into(), None, Expr::Var("param".into())),
            Stmt::Eval(Expr::Var("v1".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn stable_alias_folds() {
        // v2 = in_vs1 twice; in_vs1 is a parameter (never assigned).
        let mut stmts = vec![
            Stmt::Assign("v2".into(), None, Expr::Var("in_vs1".into())),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Assign("v2".into(), None, Expr::Var("in_vs1".into()))],
                else_body: vec![],
            },
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v2".into()))),
                name: "Foo".into(),
                args: vec![],
            }),
        ];
        assert!(apply(&mut stmts) >= 1);
        let t = format!("{stmts:?}");
        assert!(!t.contains("\"v2\""), "{t}");
        assert!(!t.contains("Assign("), "{t}");
    }

    #[test]
    fn alias_to_reassigned_source_is_refused() {
        // v2 = v1 (twice) but v1 is reassigned elsewhere → not stable.
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(1)),
            Stmt::Assign("v2".into(), None, Expr::Var("v1".into())),
            Stmt::Assign("v1".into(), None, Expr::Int(2)),
            Stmt::Assign("v2".into(), None, Expr::Var("v1".into())),
            Stmt::Eval(Expr::Var("v2".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn address_taken_constant_is_refused() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, Expr::Int(0)),
            Stmt::Eval(Expr::AddressOf(Box::new(Expr::Var("v1".into())))),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn complex_target_before_def_still_refuses() {
        // The address use appears BEFORE the definition is scanned; the
        // refusal must not depend on traversal order.
        let mut stmts = vec![
            Stmt::Assign("*(int*)((nint)&v7 +4)".into(), None, Expr::Int(1)),
            Stmt::Assign("v7".into(), None, Expr::Int(0)),
            Stmt::Eval(Expr::Var("v7".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    #[test]
    fn nested_reads_are_substituted_and_defs_dropped() {
        let mut stmts = vec![
            Stmt::Assign("v3".into(), None, Expr::Int(2)),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::Var("v3".into())),
                    Box::new(Expr::Int(2)),
                ),
                then_body: vec![Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Foo".into(),
                    args: vec![Expr::Var("v3".into())],
                })],
                else_body: vec![],
            },
        ];
        assert!(apply(&mut stmts) >= 3);
        let t = format!("{stmts:?}");
        assert!(!t.contains("\"v3\""), "{t}");
        assert!(!t.contains("Assign("), "{t}");
        assert_eq!(stmts.len(), 1, "{t}");
        assert!(t.contains("Int(2)"), "{t}");
    }

    #[test]
    fn value_in_complex_target_is_rewritten() {
        let mut stmts = vec![
            Stmt::Assign("v4".into(), None, Expr::Int(12)),
            Stmt::Assign(
                "*(float*)(i * v4 + array +32)".into(),
                None,
                Expr::Var("v4".into()),
            ),
        ];
        assert!(apply(&mut stmts) >= 2);
        let t = format!("{stmts:?}");
        assert!(!t.contains("\"v4\""), "{t}");
        assert_eq!(stmts.len(), 1, "{t}");
        assert!(t.contains("i * 12 + array +32"), "{t}");
    }

    #[test]
    fn receiver_use_keeps_a_constant_but_not_an_alias() {
        // `v6` would become `0.Length` — refuse the constant.
        let mut stmts = vec![
            Stmt::Assign("v6".into(), None, Expr::Int(0)),
            Stmt::Eval(Expr::Field(Box::new(Expr::Var("v6".into())), "Length".into())),
        ];
        assert_eq!(apply(&mut stmts), 0);
        // An alias name is a valid receiver.
        let mut stmts = vec![
            Stmt::Assign("v6".into(), None, Expr::Var("obj".into())),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(Expr::Var("v6".into()))),
                name: "Foo".into(),
                args: vec![],
            }),
        ];
        assert!(apply(&mut stmts) >= 1);
        assert!(format!("{stmts:?}").contains("\"obj\""));
    }

    #[test]
    fn lambda_param_shadow_is_refused() {
        let mut stmts = vec![
            Stmt::Assign("v5".into(), None, Expr::Int(1)),
            Stmt::Eval(Expr::Lambda {
                params: vec!["v5".into()],
                body: Box::new(Expr::Var("v5".into())),
            }),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }
}
