//! Source-level copy propagation and dead-copy removal.
//!
//! SSA destruction and the merge rules in `highvar` leave long chains of
//! `t = s` copies in the reconstructed body: the R69 census has 2,802 of them,
//! and they are the largest single evidence class behind the remaining `vN`
//! names (55% of the unrenamed refs in the top methods). `highvar` cannot fold
//! most of them because after a phi/merge a source version has several
//! definitions (one per predecessor edge), which its single-definition alias
//! proof refuses. At the AST level the same copies are removable with a
//! reaching-copy analysis:
//!
//!  * a copy `t = s` makes `s` the value of `t` until either name is
//!    redefined;
//!  * any control-flow construct is handled with a private environment (a copy
//!    made before it is visible inside, nothing made inside escapes);
//!  * a copy whose destination has no remaining reads is deleted (the RHS is a
//!    bare local, so deletion has no effects).
//!
//! Address-taken locals and `foreach` variables are protected: their identity
//! is observable and their reads are not rewritten.

use super::super::csharp::{Expr, Stmt};
use std::collections::{HashMap, HashSet};

type Env = HashMap<String, String>;

fn is_simple_local(n: &str) -> bool {
    !n.is_empty()
        && !n.contains('.')
        && !n.contains('[')
        && !n.contains('*')
        && !n.contains('&')
        && !n.contains('<')
        && !n.contains('>')
        && n.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !n.chars().next().is_some_and(|c| c.is_ascii_digit())
}

#[allow(dead_code)]
fn resolve(env: &Env, name: &str) -> String {
    let mut cur = name.to_string();
    for _ in 0..8 {
        match env.get(&cur) {
            Some(next) => cur = next.clone(),
            None => break,
        }
    }
    cur
}

/// Record that `name` now holds a new value: every copy OF it and every copy
/// INTO it stops being valid.
fn invalidate(env: &mut Env, name: &str) {
    env.remove(name);
    env.retain(|_, v| v != name);
}

/// Expressions directly owned by this statement (nested statement lists are
/// visited separately). `Using.decl` is a statement and `Switch.disc` /
/// `ForEach.collection` are rendered paths, not expression nodes.
pub(crate) fn for_each_expr_mut(s: &mut Stmt, f: &mut dyn FnMut(&mut Expr)) {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Assign(_, _, e) => f(e),
        Stmt::Return(Some(e)) => f(e),
        Stmt::If { cond, .. }
        | Stmt::While { cond, .. }
        | Stmt::DoWhile { cond, .. } => f(cond),
        Stmt::Lock { expr, .. } => f(expr),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                f(c);
            }
        }
        Stmt::StringSwitch { disc, .. } => f(disc),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            f(collection);
            f(index);
            f(value);
        }
        Stmt::EventAssign { field, value, .. } => {
            f(field);
            f(value);
        }
        Stmt::Attribute { args, .. } => {
            for a in args {
                f(a);
            }
        }
        _ => {}
    }
}

fn substitute(expr: &mut Expr, env: &Env, fin: &Env) {
    for c in super::il2cpp_cleanup::expr_children_mut(expr) {
        substitute(c, env, fin);
    }
    if let Expr::Var(name) = expr {
        if let Some(src) = env.get(name).or_else(|| fin.get(name)) {
            if src == "this" {
                *expr = Expr::This;
            } else {
                *name = src.clone();
            }
        }
    }
}

/// Replace identifier tokens in a rendered read path (`target.Field`,
/// `ForEach.collection`, `Switch.disc`).
fn subst_tokens(text: &mut String, env: &Env, fin: &Env) -> usize {
    let mut out = String::with_capacity(text.len());
    let mut changed = 0usize;
    let mut tok = String::new();
    let flush = |tok: &mut String, out: &mut String, changed: &mut usize| {
        if tok.is_empty() {
            return;
        }
        match env.get(tok.as_str()).or_else(|| fin.get(tok.as_str())) {
            Some(src) => {
                out.push_str(src);
                *changed += 1;
            }
            None => out.push_str(tok),
        }
        tok.clear();
    };
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            tok.push(c);
        } else {
            flush(&mut tok, &mut out, &mut changed);
            out.push(c);
        }
    }
    flush(&mut tok, &mut out, &mut changed);
    if changed > 0 {
        *text = out;
    }
    changed
}

struct Ctx {
    protected: HashSet<String>,
    /// Position of the LAST definition of each plain local, in the same
    /// pre-order the walker uses. A copy whose source is already final — no
    /// definition of it at or after the copy — stays true for the rest of the
    /// method, so it can be propagated across branches and labels.
    last_def: HashMap<String, usize>,
    pos: usize,
}

fn visit_list(list: &mut Vec<Stmt>, env: &mut Env, fin: &mut Env, ctx: &mut Ctx) -> usize {
    let mut n = 0;
    for s in list.iter_mut() {
        n += visit_stmt(s, env, fin, ctx);
    }
    n
}

fn visit_stmt(s: &mut Stmt, env: &mut Env, fin: &mut Env, ctx: &mut Ctx) -> usize {
    let mut n = 0;
    ctx.pos += 1;
    let here = ctx.pos;
    // Names the whole construct may write. A final-value claim on one of
    // them dies after the construct, because an arm may have replaced it.
    let written = match s {
        Stmt::If { .. }
        | Stmt::While { .. }
        | Stmt::DoWhile { .. }
        | Stmt::For { .. }
        | Stmt::ForEach { .. }
        | Stmt::Switch { .. }
        | Stmt::StringSwitch { .. }
        | Stmt::Try { .. }
        | Stmt::Using { .. }
        | Stmt::Lock { .. } => written_names(s),
        _ => HashSet::new(),
    };
    // Control flow: the environment is visible inside each arm, but nothing
    // defined inside may escape, and the parent environment is dropped after
    // the construct (a write in any arm could invalidate a source). FINAL
    // claims survive it: their source is never written again, so the only
    // way they can stop holding is a write to the DESTINATION, which is
    // exactly what `written` reports.
    match s {
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            substitute(cond, env, fin);
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            n += visit_list(then_body, &mut inner, &mut inner_fin, ctx);
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            n += visit_list(else_body, &mut inner, &mut inner_fin, ctx);
            leave(env, fin, &written);
            return n;
        }
        Stmt::While { cond, body, .. } | Stmt::DoWhile { cond, body, .. } => {
            substitute(cond, env, fin);
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            n += visit_list(body, &mut inner, &mut inner_fin, ctx);
            leave(env, fin, &written);
            return n;
        }
        Stmt::For {
            init,
            incr,
            cond,
            body,
        } => {
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            n += visit_list(init, &mut inner, &mut inner_fin, ctx);
            n += visit_list(incr, &mut inner, &mut inner_fin, ctx);
            if let Some(c) = cond {
                substitute(c, &mut inner, &mut inner_fin);
            }
            n += visit_list(body, &mut inner, &mut inner_fin, ctx);
            leave(env, fin, &written);
            return n;
        }
        Stmt::ForEach {
            collection,
            body,
            var_name,
        } => {
            n += subst_tokens(collection, env, fin);
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            inner.remove(var_name.as_str());
            inner_fin.remove(var_name.as_str());
            n += visit_list(body, &mut inner, &mut inner_fin, ctx);
            invalidate(env, var_name);
            fin.remove(var_name.as_str());
            leave(env, fin, &written);
            return n;
        }
        Stmt::Switch {
            disc,
            cases,
            tail,
            default,
        } => {
            n += subst_tokens(disc, env, fin);
            for (_, b) in cases.iter_mut() {
                let mut inner = env.clone();
                let mut inner_fin = fin.clone();
                n += visit_list(b, &mut inner, &mut inner_fin, ctx);
            }
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            n += visit_list(tail, &mut inner, &mut inner_fin, ctx);
            if let Some(d) = default {
                let mut inner = env.clone();
                let mut inner_fin = fin.clone();
                n += visit_list(d, &mut inner, &mut inner_fin, ctx);
            }
            leave(env, fin, &written);
            return n;
        }
        Stmt::StringSwitch { disc, cases, tail } => {
            substitute(disc, env, fin);
            for (_, b) in cases.iter_mut() {
                let mut inner = env.clone();
                let mut inner_fin = fin.clone();
                n += visit_list(b, &mut inner, &mut inner_fin, ctx);
            }
            if let Some(t) = tail {
                let mut inner = env.clone();
                let mut inner_fin = fin.clone();
                n += visit_list(t, &mut inner, &mut inner_fin, ctx);
            }
            leave(env, fin, &written);
            return n;
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            n += visit_list(try_body, &mut inner, &mut inner_fin, ctx);
            for (_, b) in catches.iter_mut() {
                let mut inner = env.clone();
                let mut inner_fin = fin.clone();
                n += visit_list(b, &mut inner, &mut inner_fin, ctx);
            }
            if let Some(fb) = finally_body {
                let mut inner = env.clone();
                let mut inner_fin = fin.clone();
                n += visit_list(fb, &mut inner, &mut inner_fin, ctx);
            }
            leave(env, fin, &written);
            return n;
        }
        Stmt::Using { decl, body } => {
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            n += visit_stmt(decl, &mut inner, &mut inner_fin, ctx);
            n += visit_list(body, &mut inner, &mut inner_fin, ctx);
            leave(env, fin, &written);
            return n;
        }
        Stmt::Lock { expr, body } => {
            substitute(expr, env, fin);
            let mut inner = env.clone();
            let mut inner_fin = fin.clone();
            n += visit_list(body, &mut inner, &mut inner_fin, ctx);
            leave(env, fin, &written);
            return n;
        }
        Stmt::Label(_) | Stmt::Goto(_) => {
            // A label is an entry edge and a goto is an edge out: no
            // path-sensitive copy claim may cross either. A FINAL claim can:
            // its source never changes again, so every path that reaches the
            // read agrees with the copy (a path that skips the copy reads an
            // unassigned slot in the original too).
            env.clear();
        }
        _ => {}
    }

    // Every read in this statement sees the current copies.
    let mut assign_copy: Option<(String, String)> = None;
    if let Stmt::Assign(target, None, rhs) = s {
        substitute(rhs, env, fin);
        if is_simple_local(target) {
            if matches!(rhs, Expr::This) {
                assign_copy = Some((target.clone(), "this".to_string()));
            } else if let Expr::Var(src) = rhs {
                if src == target {
                    assign_copy = Some((target.clone(), src.clone()));
                } else {
                    let resolved = resolve2(env, fin, src);
                    assign_copy = Some((target.clone(), resolved));
                }
            }
        }
    }
    for_each_expr_mut(s, &mut |e| substitute(e, env, fin));
    if let Stmt::Assign(target, _, _) = s {
        if !is_simple_local(target) {
            n += subst_tokens(target, env, fin);
        }
    }
    if let Stmt::Assign(target, _, _) = s {
        if is_simple_local(target) {
            invalidate(env, target);
            fin.remove(target.as_str());
            if let Some((t, src)) = assign_copy {
                if t == *target && src != t && !ctx.protected.contains(&t) {
                    // A source with NO definition at all (a parameter, an entry
                    // value, `this`) is final everywhere.
                    if ctx.last_def.get(&src).is_none_or(|&p| p < here) {
                        fin.insert(t, src);
                    } else {
                        env.insert(t, src);
                    }
                }
            }
        }
    }
    n
}

/// Drop the path-sensitive claims after a construct; keep the final ones
/// except where the construct may have written the destination.
fn leave(env: &mut Env, fin: &mut Env, written: &HashSet<String>) {
    env.clear();
    for name in written {
        fin.remove(name);
    }
}

/// Every plain-local write anywhere inside `s`.
fn written_names(s: &Stmt) -> HashSet<String> {
    let mut out = HashSet::new();
    fn walk(s: &Stmt, out: &mut HashSet<String>) {
        match s {
            Stmt::Assign(t, _, _) if is_simple_local(t) => {
                out.insert(t.clone());
            }
            Stmt::ForEach { var_name, .. } => {
                out.insert(var_name.clone());
            }
            _ => {}
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            for c in k {
                walk(c, out);
            }
        }
        if let Stmt::Using { decl, .. } = s {
            walk(decl, out);
        }
    }
    walk(s, &mut out);
    out
}

/// Position of the last definition of every plain local, numbered in the
/// same pre-order `visit_stmt` walks.
fn number_list(list: &[Stmt], pos: &mut usize, last: &mut HashMap<String, usize>) {
    for s in list {
        number_stmt(s, pos, last);
    }
}

fn number_stmt(s: &Stmt, pos: &mut usize, last: &mut HashMap<String, usize>) {
    *pos += 1;
    let here = *pos;
    match s {
        Stmt::Assign(t, _, _) if is_simple_local(t) => {
            last.insert(t.clone(), here);
        }
        Stmt::ForEach { var_name, .. } => {
            last.insert(var_name.clone(), here);
        }
        _ => {}
    }
    match s {
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            number_list(then_body, pos, last);
            number_list(else_body, pos, last);
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => number_list(body, pos, last),
        Stmt::For {
            init,
            incr,
            body,
            ..
        } => {
            number_list(init, pos, last);
            number_list(incr, pos, last);
            number_list(body, pos, last);
        }
        Stmt::ForEach { body, .. } => number_list(body, pos, last),
        Stmt::Switch {
            cases,
            tail,
            default,
            ..
        } => {
            for (_, b) in cases {
                number_list(b, pos, last);
            }
            number_list(tail, pos, last);
            if let Some(d) = default {
                number_list(d, pos, last);
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, b) in cases {
                number_list(b, pos, last);
            }
            if let Some(t) = tail {
                number_list(t, pos, last);
            }
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            number_list(try_body, pos, last);
            for (_, b) in catches {
                number_list(b, pos, last);
            }
            if let Some(f) = finally_body {
                number_list(f, pos, last);
            }
        }
        Stmt::Using { decl, body } => {
            number_stmt(decl, pos, last);
            number_list(body, pos, last);
        }
        Stmt::Lock { body, .. } => number_list(body, pos, last),
        _ => {}
    }
}

/// Follow copy chains through both maps.
fn resolve2(env: &Env, fin: &Env, name: &str) -> String {
    let mut cur = name.to_string();
    for _ in 0..8 {
        match env.get(&cur).or_else(|| fin.get(&cur)) {
            Some(next) => cur = next.clone(),
            None => break,
        }
    }
    cur
}

fn expr_reads(expr: &Expr, out: &mut Vec<String>) {
    if let Expr::Var(n) = expr {
        out.push(n.clone());
    }
    for c in super::il2cpp_cleanup::expr_children(expr) {
        expr_reads(c, out);
    }
}

/// Identifier tokens in a read-only rendered path (`target.Field`, a switch
/// discriminator, a foreach collection). A lone identifier that is an
/// assignment target is a WRITE and is not counted by the caller.
fn token_reads(text: &str, out: &mut Vec<String>) {
    let mut tok = String::new();
    for c in text.chars().chain(std::iter::once(' ')) {
        if c.is_alphanumeric() || c == '_' {
            tok.push(c);
        } else if !tok.is_empty() {
            out.push(std::mem::take(&mut tok));
        }
    }
}

fn count_reads(list: &[Stmt], counts: &mut HashMap<String, usize>) {
    for s in list {
        let mut reads: Vec<String> = Vec::new();
        let mut clone = s.clone();
        for_each_expr_mut(&mut clone, &mut |e| expr_reads(e, &mut reads));
        match s {
            Stmt::Assign(target, _, _) if !is_simple_local(target) => {
                token_reads(target, &mut reads);
            }
            Stmt::ForEach { collection, .. } => token_reads(collection, &mut reads),
            Stmt::Switch { disc, .. } => token_reads(disc, &mut reads),
            _ => {}
        }
        for r in reads {
            *counts.entry(r).or_default() += 1;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            count_reads(k, counts);
        }
        if let Stmt::Using { decl, .. } = s {
            let one = std::slice::from_ref(decl.as_ref());
            count_reads(one, counts);
        }
    }
}

fn protected_names(list: &[Stmt], out: &mut HashSet<String>) {
    fn walk(e: &Expr, out: &mut HashSet<String>) {
        if let Expr::AddressOf(inner) | Expr::RawAddr(inner, _) = e {
            if let Expr::Var(n) = inner.as_ref() {
                out.insert(n.clone());
            }
        }
        for c in super::il2cpp_cleanup::expr_children(e) {
            walk(c, out);
        }
    }
    for s in list {
        if let Stmt::ForEach { var_name, .. } = s {
            out.insert(var_name.clone());
        }
        let mut clone = s.clone();
        for_each_expr_mut(&mut clone, &mut |e| walk(e, out));
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            protected_names(k, out);
        }
        if let Stmt::Using { decl, .. } = s {
            protected_names(std::slice::from_ref(decl.as_ref()), out);
        }
    }
}

/// Remove copy assignments whose destination nobody reads.
///
/// Only `Assign(target, None, Var(src))` is considered: the RHS is a bare
/// local, so its evaluation has no effects.
fn drop_dead_copies(list: &mut Vec<Stmt>, reads: &HashMap<String, usize>) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        let drop = matches!(
            &list[i],
            Stmt::Assign(target, None, Expr::Var(_))
                if is_simple_local(target) && reads.get(target).copied().unwrap_or(0) == 0
        );
        if drop {
            list.remove(i);
            n += 1;
        } else {
            i += 1;
        }
    }
    for s in list.iter_mut() {
        for k in super::children_mut(s) {
            n += drop_dead_copies(k, reads);
        }
    }
    n
}

/// Coalesce a local whose every definition is a copy of the SAME immutable
/// source.
///
/// `x = y;` is only a temporary name for `y` when nothing can ever change `y`
/// and nothing else ever writes `x`: then every read of `x` is `y` on every
/// path, whatever control flow sits between them. The reaching-copy analysis
/// below cannot prove that — it drops its environment at every branch and
/// label — so the copies that straddle a join survive it. This rule needs no
/// path reasoning at all, which is what makes it immune to control flow.
///
/// The source is a PARAMETER, `this`, or an entry value: a name the body never
/// writes. Address-taken locals are excluded on both sides, because their
/// identity is observable through the address.
fn coalesce_immutable(stmts: &mut Vec<Stmt>) -> usize {
    let mut protected = HashSet::new();
    protected_names(stmts, &mut protected);
    let mut written: HashSet<String> = HashSet::new();
    let mut defs: HashMap<String, Vec<Option<String>>> = HashMap::new();
    collect_defs(stmts, &mut written, &mut defs);
    let mut dominated = HashSet::new();
    collect_dominated(stmts, &mut dominated);
    let mut env: Env = HashMap::new();
    for (name, sites) in &defs {
        if sites.is_empty() || protected.contains(name) || !dominated.contains(name) {
            continue;
        }
        let mut source: Option<&str> = None;
        let mut ok = true;
        for site in sites {
            match (site.as_deref(), source) {
                (Some(y), None) => source = Some(y),
                (Some(y), Some(prev)) if prev == y => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        let Some(source) = source else { continue };
        if !ok || source == name || written.contains(source) || protected.contains(source) {
            continue;
        }
        env.insert(name.clone(), source.to_string());
    }
    if env.is_empty() {
        return 0;
    }
    rewrite_reads(stmts, &env);
    drop_copies_of(stmts, &env)
}

/// Every plain-local write in the body, and the shape of each definition
/// (`Some(src)` when it is `name = <local src>`).
fn collect_defs(
    stmts: &[Stmt],
    written: &mut HashSet<String>,
    defs: &mut HashMap<String, Vec<Option<String>>>,
) {
    for s in stmts {
        match s {
            Stmt::Assign(target, _, rhs) => {
                if is_simple_local(target) {
                    written.insert(target.clone());
                    let src = match rhs {
                        Expr::Var(y) if is_simple_local(y) => Some(y.clone()),
                        _ => None,
                    };
                    defs.entry(target.clone()).or_default().push(src);
                }
            }
            Stmt::ForEach { var_name, .. } => {
                written.insert(var_name.clone());
                defs.entry(var_name.clone()).or_default().push(None);
            }
            Stmt::Using { decl, .. } => {
                if let Stmt::Assign(target, _, _) = decl.as_ref() {
                    if is_simple_local(target) {
                        written.insert(target.clone());
                        defs.entry(target.clone()).or_default().push(None);
                    }
                }
            }
            _ => {}
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for child in kids {
            collect_defs(child, written, defs);
        }
    }
}

/// Names whose single definition sits at the METHOD ROOT and is preceded by
/// no read of the name anywhere.
///
/// That is the cheap dominance proof this rule needs: root statements run in
/// order, so a definition with no earlier read cannot be bypassed on the way
/// to any read that follows it. A definition inside a branch is NOT included
/// — the branch may not run, and the surviving read would then have no value
/// to stand for (the test `a_copy_inside_a_branch_does_not_escape` pins this).
fn collect_dominated(list: &[Stmt], out: &mut HashSet<String>) {
    let mut seen: HashMap<String, usize> = HashMap::new();
    for s in list {
        if let Stmt::Assign(target, None, Expr::Var(_)) = s {
            if is_simple_local(target) && seen.get(target.as_str()).copied().unwrap_or(0) == 0 {
                out.insert(target.clone());
            }
        }
        count_reads(std::slice::from_ref(s), &mut seen);
    }
}

fn rewrite_reads(stmts: &mut Vec<Stmt>, env: &Env) {
    for s in stmts.iter_mut() {
        for_each_expr_mut(s, &mut |e| substitute(e, env, env));
        match s {
            // A rendered lvalue path (`v1.Size`, `*(int*)(v1 + 4)`) embeds
            // the name as a token. A PLAIN target is a write of that local,
            // not a read, and must keep its own name.
            Stmt::Assign(target, _, _) if !is_simple_local(target) => {
                subst_tokens(target, env, env);
            }
            Stmt::ForEach { collection, .. } => {
                subst_tokens(collection, env, env);
            }
            Stmt::Switch { disc, .. } => {
                subst_tokens(disc, env, env);
            }
            Stmt::StringSwitch { disc, .. } => {
                substitute(disc, env, env);
            }
            _ => {}
        }
        for child in super::children_mut(s) {
            rewrite_reads(child, env);
        }
    }
}

fn drop_copies_of(stmts: &mut Vec<Stmt>, env: &Env) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < stmts.len() {
        let drop = matches!(
            &stmts[i],
            Stmt::Assign(target, None, Expr::Var(src)) if env.get(target).is_some_and(|s| s == src)
        );
        if drop {
            stmts.remove(i);
            n += 1;
        } else {
            i += 1;
        }
    }
    for s in stmts.iter_mut() {
        for child in super::children_mut(s) {
            n += drop_copies_of(child, env);
        }
    }
    n
}

/// Propagate copies through the body and delete dead copy statements.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut protected = HashSet::new();
    protected_names(stmts, &mut protected);
    // Protected names are never tracked as copy sources; their value identity
    // is observable through the address.
    // Barrier-immune coalescing first: it removes the copies the
    // path-sensitive walk below cannot see past a join, and leaves fewer
    // names for it to track.
    let mut n = coalesce_immutable(stmts);
    let mut last_def: HashMap<String, usize> = HashMap::new();
    let mut pos = 0usize;
    number_list(stmts, &mut pos, &mut last_def);
    let mut ctx = Ctx {
        protected: protected.clone(),
        last_def,
        pos: 0,
    };
    let mut env: Env = HashMap::new();
    let mut fin: Env = HashMap::new();
    n += visit_list(stmts, &mut env, &mut fin, &mut ctx);
    // Deleting one dead copy can make its source copy dead too.
    loop {
        let mut reads: HashMap<String, usize> = HashMap::new();
        count_reads(stmts, &mut reads);
        let d = drop_dead_copies(stmts, &reads);
        n += d;
        if d == 0 {
            break;
        }
    }
    n
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
    fn a_straight_line_copy_is_propagated_and_removed() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, var("config")),
            Stmt::Assign(
                "out".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "Use".into(),
                    args: vec![var("v1")],
                },
            ),
            Stmt::Eval(var("v1")),
        ];
        apply(&mut stmts);
        let t = text(&stmts);
        assert!(!t.contains("Var(\"v1\")"), "{t}");
        assert!(t.contains("Var(\"config\")"), "{t}");
    }

    #[test]
    fn a_copy_of_a_local_field_path_is_not_a_simple_copy() {
        let mut stmts = vec![
            Stmt::Assign(
                "v1".into(),
                None,
                Expr::Field(Box::new(Expr::This), "config".into()),
            ),
            Stmt::Eval(var("v1")),
        ];
        apply(&mut stmts);
        assert!(text(&stmts).contains("Var(\"v1\")"));
    }

    #[test]
    fn a_redefinition_of_the_source_stops_propagation() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, var("a")),
            Stmt::Eval(Expr::Field(Box::new(var("v1")), "X".into())),
            Stmt::Assign("a".into(), None, Expr::Int(1)),
            Stmt::Eval(Expr::Field(Box::new(var("v1")), "Y".into())),
        ];
        apply(&mut stmts);
        let t = text(&stmts);
        assert!(t.contains("Field(Var(\"a\"), \"X\")"), "{t}");
        assert!(t.contains("Field(Var(\"v1\"), \"Y\")"), "{t}");
    }

    #[test]
    fn a_copy_inside_a_branch_does_not_escape() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Assign("v1".into(), None, var("a"))],
                else_body: vec![],
            },
            Stmt::Eval(var("v1")),
        ];
        apply(&mut stmts);
        assert!(text(&stmts).contains("Var(\"v1\")"));
    }

    #[test]
    fn a_dead_copy_is_dropped_without_any_use() {
        let mut stmts = vec![
            Stmt::Assign("v9".into(), None, var("config")),
            Stmt::Eval(Expr::Field(Box::new(var("config")), "X".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
    }

    #[test]
    fn an_address_taken_local_is_never_removed_or_rewritten() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, var("config")),
            Stmt::Eval(Expr::AddressOf(Box::new(var("v1")))),
        ];
        apply(&mut stmts);
        assert!(text(&stmts).contains("Var(\"v1\")"));
    }

    #[test]
    fn reads_inside_an_lvalue_path_are_rewritten() {
        let mut stmts = vec![
            Stmt::Assign("v1".into(), None, var("config")),
            Stmt::Assign("v1.Size".into(), None, Expr::Int(3)),
        ];
        apply(&mut stmts);
        let t = text(&stmts);
        assert!(t.contains("config.Size"), "{t}");
    }

    // --- final-source propagation (crosses control flow) ------------------

    #[test]
    fn a_copy_of_a_parameter_crosses_a_branch() {
        // `p` is never defined, so it is final: `x` is `p` on every path that
        // reaches a read, whatever control flow sits between them.
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, var("p")),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(var("x"))],
                else_body: vec![],
            },
            Stmt::Eval(var("x")),
        ];
        apply(&mut stmts);
        let t = text(&stmts);
        assert!(!t.contains("Var(\"x\")"), "{t}");
        assert_eq!(t.matches("Var(\"p\")").count(), 2, "{t}");
    }

    #[test]
    fn a_copy_of_a_value_defined_later_does_not_cross() {
        // The read is only reachable through FINAL claims: the branch drops the
        // path-sensitive claim, and `y` is defined after the copy, so the copy
        // must survive.
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, var("y")),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![],
                else_body: vec![],
            },
            Stmt::Eval(var("x")),
            Stmt::Assign("y".into(), None, Expr::Int(5)),
        ];
        apply(&mut stmts);
        assert!(text(&stmts).contains("Var(\"x\")"));
    }

    #[test]
    fn a_final_copy_dies_when_an_arm_writes_its_destination() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, var("p")),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Assign("x".into(), None, Expr::Int(1))],
                else_body: vec![],
            },
            Stmt::Eval(var("x")),
        ];
        apply(&mut stmts);
        let t = text(&stmts);
        assert!(t.contains("Var(\"x\")"), "{t}");
    }

    #[test]
    fn a_copy_of_this_becomes_this() {
        let mut stmts = vec![
            Stmt::Assign("x".into(), None, Expr::This),
            Stmt::Eval(Expr::Field(Box::new(var("x")), "count".into())),
        ];
        apply(&mut stmts);
        let t = text(&stmts);
        assert!(t.contains("Field(This"), "{t}");
    }
}