//! Fold a temp that duplicates a counter update, and drop dead generated
//! declarations.
//!
//! The arm64 lowering materialises `sub wN, wX, #1` into a temp and stores it
//! back, then tests the temp:
//!
//! ```text
//! v331 = v46 - 1;
//! ptr5 = ptr5 + 16;
//! v46 = v46 - 1;
//! if (v331 != 0) continue;
//! ```
//!
//! At the test, `v331` and `v46` hold the same value: the update recomputed
//! exactly the expression the temp was defined with, and nothing wrote `v46`
//! in between. Replacing the read with `v46` and deleting the def removes the
//! temp. The proof is deliberately local and conservative:
//!
//!  * the temp has exactly one definition (globally) and exactly one read;
//!  * def and update are in the SAME statement list, separated only by
//!    straight-line statements that write neither operand;
//!  * the read is after the update, again with no control-flow statement and
//!    no write to the counter in between;
//!  * the temp is never address-taken or used in a name-only slot.
//!
//! The same scan drops definitions whose value is never read, one definition
//! at a time when its own right-hand side is pure (a pure, unread write has no
//! observable effect). This is the `var vN = default;` / `vN = in_xN;` text
//! that carries no information. Impure definitions (calls) stay even when
//! unread; they are not covered by the per-def purity check.

use crate::recovery::csharp::{BinOp, Expr, Stmt};
use std::collections::{HashMap, HashSet};

fn is_plain_name(n: &str) -> bool {
    !n.is_empty()
        && n.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !matches!(n, "this" | "base" | "null" | "true" | "false" | "new" | "default")
}

fn is_generated(n: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if let Some(rest) = n.strip_prefix('v') {
        return digits(rest);
    }
    for prefix in ["obj", "ptr", "num", "flag", "sf"] {
        if let Some(rest) = n.strip_prefix(prefix) {
            if digits(rest) {
                return true;
            }
        }
    }
    false
}

fn is_pure(e: &Expr) -> bool {
    match e {
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
        | Expr::MethodGroup(_)
        | Expr::TypeOf(_)
        | Expr::StaticField(_, _) => true,
        // A field/deref/address READ has no observable effect: dropping an
        // unread one cannot change behaviour, and moving it to its single
        // adjacent read cannot reorder it past anything. This covers the
        // repeated `instance = this.RewardName;` receiver materialisations.
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => is_pure(r),
        Expr::AddressOf(x) => is_pure(x),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => is_pure(a) && is_pure(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) => is_pure(x),
        _ => false,
    }
}

#[derive(Default)]
struct Facts {
    defs: HashMap<String, usize>,
    reads: HashMap<String, usize>,
    /// Names that may not be folded: address-taken, name-only slots, complex
    /// targets, or read inside a lambda (a separate binding scope).
    bad: HashSet<String>,
}

fn note_target(text: &str, f: &mut Facts) {
    // A complex assignment target such as `*(T*)(&v1 + 8)` reads/escapes the
    // local; treat every plain-name token it mentions as unsafe and as a read.
    for tok in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if is_plain_name(tok) {
            *f.reads.entry(tok.to_string()).or_default() += 1;
            f.bad.insert(tok.to_string());
        }
    }
}

fn scan_expr(e: &Expr, f: &mut Facts, in_lambda: bool) {
    match e {
        Expr::Var(n) => {
            if is_plain_name(n) {
                *f.reads.entry(n.clone()).or_default() += 1;
                if in_lambda {
                    f.bad.insert(n.clone());
                }
            }
        }
        Expr::AddressOf(x) | Expr::RawAddr(x, _) => {
            if let Expr::Var(n) = x.as_ref() {
                if is_plain_name(n) {
                    f.bad.insert(n.clone());
                }
            }
        }
        Expr::Lambda { params, .. } => {
            for p in params {
                if is_plain_name(p) {
                    f.bad.insert(p.clone());
                }
            }
            for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
                scan_expr(c, f, true);
            }
            return;
        }
        _ => {}
    }
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
        scan_expr(c, f, in_lambda);
    }
}

fn scan(stmts: &[Stmt], f: &mut Facts) {
    for s in stmts {
        match s {
            Stmt::Assign(t, _, _rhs) => {
                if is_plain_name(t) {
                    *f.defs.entry(t.clone()).or_default() += 1;
                } else {
                    note_target(t, f);
                }
            }
            Stmt::ForEach {
                var_name,
                collection,
                ..
            } => {
                if is_plain_name(var_name) {
                    *f.defs.entry(var_name.clone()).or_default() += 1;
                    f.bad.insert(var_name.clone());
                }
                if is_plain_name(collection) {
                    *f.reads.entry(collection.clone()).or_default() += 1;
                    f.bad.insert(collection.clone());
                }
            }
            Stmt::Switch { disc, .. } => {
                if is_plain_name(disc) {
                    *f.reads.entry(disc.clone()).or_default() += 1;
                    f.bad.insert(disc.clone());
                }
            }
            Stmt::Using { decl, .. } => scan(std::slice::from_ref(decl.as_ref()), f),
            _ => {}
        }
        crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
            scan_expr(e, f, false)
        });
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            scan(k, f);
        }
    }
}

fn def_pair(s: &Stmt) -> Option<(String, String, BinOp, i64)> {    let Stmt::Assign(t, _, rhs) = s else {
        return None;
    };
    if !is_generated(t) {
        return None;
    }
    let Expr::Bin(op, a, b) = rhs else { return None };
    if !matches!(op, BinOp::Add | BinOp::Sub) {
        return None;
    }
    let Expr::Var(x) = a.as_ref() else { return None };
    let Expr::Int(k) = b.as_ref() else {
        return None;
    };
    Some((t.clone(), x.clone(), *op, *k))
}

fn update_pair(s: &Stmt) -> Option<(String, BinOp, i64)> {    let Stmt::Assign(x, _, rhs) = s else {
        return None;
    };
    let Expr::Bin(op, a, b) = rhs else { return None };
    if !matches!(op, BinOp::Add | BinOp::Sub) {
        return None;
    }
    let Expr::Var(src) = a.as_ref() else {
        return None;
    };
    if src != x {
        return None;
    }
    let Expr::Int(k) = b.as_ref() else {
        return None;
    };
    Some((x.clone(), *op, *k))
}

fn token_mentions(text: &str, name: &str) -> bool {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|tok| tok == name)
}

fn writes(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Assign(t, _, _) => t == name || token_mentions(t, name),
        Stmt::ForEach { var_name, .. } => var_name == name,
        Stmt::EventAssign { .. } | Stmt::IndexAssign { .. } => false,
        _ => false,
    }
}

/// A read in this statement's own expressions happens exactly where the
/// statement does. `If` qualifies through its condition; a loop does not,
/// because a later iteration can see a different counter value.
fn direct_read_executes(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::If { .. }
            | Stmt::Return(Some(_))
            | Stmt::Throw(_)
            | Stmt::Eval(_)
            | Stmt::Assign(..)
            | Stmt::IndexAssign { .. }
            | Stmt::EventAssign { .. }
            | Stmt::YieldReturn(_)
            | Stmt::Lock { .. }
    )
}

/// Statements that may execute between the def, the update and the read
/// without introducing another path. Anything with control flow (branch,
/// loop, label, jump, terminator) ends the straight-line window.
fn is_straight(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Assign(..)
            | Stmt::EventAssign { .. }
            | Stmt::IndexAssign { .. }
            | Stmt::Eval(_)
            | Stmt::Comment(_)
            | Stmt::Attribute { .. }
    )
}

fn replace_var(e: &mut Expr, from: &str, to: &str) -> usize {
    if let Expr::Var(n) = e {
        if n == from {
            *n = to.to_string();
            return 1;
        }
    }
    let mut n = 0;
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
        n += replace_var(c, from, to);
    }
    n
}

fn direct_reads(s: &Stmt, name: &str) -> bool {
    let mut found = false;
    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
        if !found {
            let mut hit = false;
            let mut stack: Vec<&Expr> = vec![e];
            while let Some(x) = stack.pop() {
                if matches!(x, Expr::Var(n) if n == name) {
                    hit = true;
                    break;
                }
                stack.extend(
                    crate::recovery::reconstruct::il2cpp_cleanup::expr_children(x),
                );
            }
            found = hit;
        }
    });
    found
}

fn expr_vars(e: &Expr, out: &mut HashSet<String>) {
    if let Expr::Var(n) = e {
        out.insert(n.clone());
    }
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
        expr_vars(c, out);
    }
}

/// A pure value that can be recomputed at a single later read without
/// changing effects: constants, names, and arithmetic/comparison/cast over
/// them. Memory reads and calls are excluded — moving those across statements
/// could observe a different value.
fn inline_expr_ok(e: &Expr) -> bool {
    match e {
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::MethodGroup(_)
        | Expr::TypeOf(_) => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => inline_expr_ok(a) && inline_expr_ok(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) => inline_expr_ok(x),
        _ => false,
    }
}

fn inline_candidate(s: &Stmt) -> Option<(String, Expr)> {
    let Stmt::Assign(t, _, rhs) = s else {
        return None;
    };
    if !is_generated(t) || !inline_expr_ok(rhs) {
        return None;
    }
    Some((t.clone(), rhs.clone()))
}

fn subst_with_expr(e: &mut Expr, from: &str, rhs: &Expr) -> usize {
    if matches!(e, Expr::Var(n) if n == from) {
        *e = rhs.clone();
        return 1;
    }
    let mut n = 0;
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
        n += subst_with_expr(c, from, rhs);
    }
    n
}

fn expr_size(e: &Expr) -> usize {
    let mut n = 1;
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
        n += expr_size(c);
    }
    n
}

fn count_direct_reads(s: &Stmt, name: &str) -> usize {
    fn count(e: &Expr, name: &str) -> usize {
        let mut n = usize::from(matches!(e, Expr::Var(x) if x == name));
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
            n += count(c, name);
        }
        n
    }
    let mut n = 0;
    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
        n += count(e, name);
    });
    n
}

fn process_list(list: &mut Vec<Stmt>, f: &Facts) -> usize {
    let mut rewrites = 0usize;
    let mut remove: Vec<usize> = Vec::new();
    for i in 0..list.len() {
        let mut folded = false;
        if let Some((t, x, op, k)) = def_pair(&list[i]) {
            if f.defs.get(&t) == Some(&1)
                && f.reads.get(&t) == Some(&1)
                && !f.bad.contains(&t)
                && x != t
            {
                // The update that recomputes the temp's value, with a
                // straight-line window from the def and no intervening write
                // to the counter.
                let mut u = None;
                for j in i + 1..list.len() {
                    if !is_straight(&list[j]) {
                        break;
                    }
                    if writes(&list[j], &x) || writes(&list[j], &t) {
                        if let Some((ux, uop, uk)) = update_pair(&list[j]) {
                            if ux == x && uop == op && uk == k {
                                u = Some(j);
                            }
                        }
                        break;
                    }
                }
                if let Some(u) = u {
                    // The single read must be a direct expression of a later
                    // statement in the same straight-line run.
                    let mut p = None;
                    for j in u + 1..list.len() {
                        if writes(&list[j], &x) || writes(&list[j], &t) {
                            break;
                        }
                        if direct_reads(&list[j], &t) {
                            if !direct_read_executes(&list[j]) {
                                break;
                            }
                            p = Some(j);
                            break;
                        }
                        if !is_straight(&list[j]) {
                            break;
                        }
                    }
                    if let Some(p) = p {
                        let mut n = 0usize;
                        super::copy_prop::for_each_expr_mut(&mut list[p], &mut |e| {
                            n += replace_var(e, &t, &x);
                        });
                        if n > 0 {
                            rewrites += n;
                            remove.push(i);
                            folded = true;
                        }
                    }
                }
            }
        }
        if folded {
            continue;
        }
        // SINGLE-USE PURE VALUE: `t = e; … use(t)` with one global def and
        // one global read, no control flow between them, and no write to any
        // name `e` reads on the way. The read recomputes `e` where the value
        // is used; the temp and its declaration disappear.
        let Some((t, rhs)) = inline_candidate(&list[i]) else {
            continue;
        };
        let Some(&total_reads) = f.reads.get(&t) else {
            continue;
        };
        if f.defs.get(&t) != Some(&1)
            || total_reads == 0
            || total_reads > 4
            || expr_size(&rhs) > 24
            || f.bad.contains(&t)
        {
            continue;
        }
        let mut vars = HashSet::new();
        expr_vars(&rhs, &mut vars);
        if vars.contains(&t) {
            continue;
        }
        // Collect EVERY read of the temp in one straight-line window. An
        // operand write ends the window after the statement that contains it
        // (a statement's RHS is evaluated before its own write).
        let mut ps: Vec<usize> = Vec::new();
        let mut seen = 0usize;
        for j in i + 1..list.len() {
            if direct_reads(&list[j], &t) {
                if !direct_read_executes(&list[j]) {
                    break;
                }
                ps.push(j);
                seen += count_direct_reads(&list[j], &t);
                if vars.iter().any(|v| writes(&list[j], v)) {
                    break;
                }
                continue;
            }
            if !is_straight(&list[j]) {
                break;
            }
            if vars.iter().any(|v| writes(&list[j], v)) {
                break;
            }
        }
        if ps.is_empty() || seen != total_reads {
            continue;
        }
        let mut n = 0usize;
        for &p in &ps {
            super::copy_prop::for_each_expr_mut(&mut list[p], &mut |e| {
                n += subst_with_expr(e, &t, &rhs);
            });
        }
        if n > 0 {
            rewrites += n;
            remove.push(i);
        }
    }
    for &i in remove.iter().rev() {
        list.remove(i);
        rewrites += 1;
    }
    rewrites
}

fn drop_dead(stmts: &mut Vec<Stmt>, dead: &HashSet<String>) -> usize {
    let before = stmts.len();
    stmts.retain(|s| match s {
        Stmt::Assign(t, _, rhs) if dead.contains(t) && is_pure(rhs) => false,
        _ => true,
    });
    let mut n = before - stmts.len();
    for s in stmts.iter_mut() {
        for k in super::children_mut(s) {
            n += drop_dead(k, dead);
        }
    }
    n
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    apply_protected(stmts, &HashSet::new())
}

/// As [`apply`], with names whose stores must never be dropped even when the
/// body reads them nowhere (`ref`/`out` parameters: the caller reads them).
pub fn apply_protected(stmts: &mut Vec<Stmt>, protected: &HashSet<String>) -> usize {
    let mut f = Facts::default();
    scan(stmts, &mut f);
    let dead: HashSet<String> = f
        .defs
        .iter()
        .filter(|(name, count)| {
            // A name with one impure def (a call result) used to be excluded
            // ENTIRELY, which kept every other dead pure store of that name:
            // `instance = this.RewardName;` nineteen times next to a single
            // `instance = IL2CPP_Box(...)`. Purity is judged PER DEF inside
            // `drop_dead`; the impure def itself is never removed.
            **count > 0
                && f.reads.get(*name).copied().unwrap_or(0) == 0
                && !f.bad.contains(*name)
                && !protected.contains(*name)
        })
        .map(|(name, _)| name.clone())
        .collect();
    let mut n = 0usize;
    {
        let mut stack: Vec<&mut Vec<Stmt>> = vec![stmts];
        while let Some(list) = stack.pop() {
            n += process_list(list, &f);
            for s in list.iter_mut() {
                for k in super::children_mut(s) {
                    stack.push(k);
                }
            }
        }
    }
    n += drop_dead(stmts, &dead);
    n
}

/// Inline a plain local assigned a PURE value and read exactly once in the
/// statement immediately after it (comments/attributes transparent):
///
/// ```text
/// instance = this.RewardName;      instance = this.RewardName;
/// if (instance != null) …     →    if (this.RewardName != null) …
/// ```
///
/// The split Object hubs materialise one receiver per check; when the value is
/// a field/name chain and the single read is adjacent, moving the load to that
/// read is control-equivalent (no call/write between) and removes both the
/// assignment and the declaration. Refuses address-taken / name-only names,
/// self-referential RHS, calls, index loads and nested-only reads.
pub fn inline_adjacent_single_use(stmts: &mut Vec<Stmt>) -> usize {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut bad: HashSet<String> = HashSet::new();
    scan_plain(stmts, &mut counts, &mut bad);
    let mut n = 0usize;
    inline_adjacent_list(stmts, &counts, &bad, &mut n);
    n
}

fn rhs_adjacent_ok(e: &Expr) -> bool {
    match e {
        Expr::Var(_) | Expr::This | Expr::Int(_) | Expr::Float(_) | Expr::Double(_)
        | Expr::Bool(_) | Expr::Str(_) | Expr::Null | Expr::TypeOf(_)
        | Expr::MethodGroup(_) | Expr::StaticField(_, _) => true,
        Expr::Field(recv, _) | Expr::RawDeref(recv, _, _) | Expr::RawAddr(recv, _) => {
            rhs_adjacent_ok(recv)
        }
        Expr::Cast(_, inner) => rhs_adjacent_ok(inner),
        _ => false,
    }
}

fn scan_plain(
    stmts: &[Stmt],
    counts: &mut HashMap<String, usize>,
    bad: &mut HashSet<String>,
) {
    for s in stmts {
        if let Stmt::Switch { disc, .. } = s {
            if is_plain_name(disc) {
                *counts.entry(disc.clone()).or_default() += 1;
                bad.insert(disc.clone());
            }
        }
        if let Stmt::ForEach {
            var_name,
            collection,
            ..
        } = s
        {
            if is_plain_name(collection) {
                *counts.entry(collection.clone()).or_default() += 1;
                bad.insert(collection.clone());
            }
            if is_plain_name(var_name) {
                bad.insert(var_name.clone());
            }
        }
        if let Stmt::Assign(target, _, _) = s {
            if !is_plain_name(target) {
                for tok in target.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                    if is_plain_name(tok) {
                        *counts.entry(tok.to_string()).or_default() += 1;
                        bad.insert(tok.to_string());
                    }
                }
            }
        }
        crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
            expr_plain_reads(e, counts, bad);
        });
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            scan_plain(k, counts, bad);
        }
    }
}

fn expr_plain_reads(e: &Expr, counts: &mut HashMap<String, usize>, bad: &mut HashSet<String>) {
    match e {
        Expr::Var(n) => {
            if is_plain_name(n) {
                *counts.entry(n.clone()).or_default() += 1;
            }
        }
        Expr::AddressOf(x) | Expr::RawAddr(x, _) => {
            if let Expr::Var(n) = x.as_ref() {
                if is_plain_name(n) {
                    bad.insert(n.clone());
                }
            }
        }
        Expr::Lambda { params, .. } => {
            for p in params {
                bad.insert(p.clone());
            }
        }
        _ => {}
    }
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
        expr_plain_reads(c, counts, bad);
    }
}

fn adjacent_use_stmt(list: &[Stmt], start: usize) -> Option<usize> {
    for (j, s) in list.iter().enumerate().skip(start) {
        if is_trivia_local(s) {
            continue;
        }
        return Some(j);
    }
    None
}

fn is_trivia_local(s: &Stmt) -> bool {
    matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. })
}

fn expr_reads_plain(e: &Expr, name: &str) -> bool {
    if matches!(e, Expr::Var(n) if n == name) {
        return true;
    }
    crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e)
        .into_iter()
        .any(|c| expr_reads_plain(c, name))
}

fn count_direct_reads_in(s: &Stmt, name: &str) -> usize {
    fn count(e: &Expr, name: &str) -> usize {
        let mut n = usize::from(matches!(e, Expr::Var(x) if x == name));
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
            n += count(c, name);
        }
        n
    }
    let mut n = 0;
    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
        n += count(e, name);
    });
    n
}

#[allow(dead_code)]
fn direct_reads_plain(s: &Stmt, name: &str) -> bool {
    let mut hit = false;
    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
        if !hit && expr_reads_plain(e, name) {
            hit = true;
        }
    });
    hit
}

fn inline_adjacent_list(
    list: &mut Vec<Stmt>,
    counts: &HashMap<String, usize>,
    bad: &HashSet<String>,
    n: &mut usize,
) {
    let mut i = 0usize;
    while i + 1 < list.len() {
        let cand = match &list[i] {
            Stmt::Assign(t, _, rhs)
                if is_plain_name(t)
                    && !bad.contains(t)
                    && counts.get(t).copied().is_some_and(|c| c >= 1)
                    && rhs_adjacent_ok(rhs)
                    && !expr_reads_plain(rhs, t) =>
            {
                Some((t.clone(), rhs.clone()))
            }
            _ => None,
        };
        if let Some((t, rhs)) = cand {
            let total = counts.get(&t).copied().unwrap_or(0);
            let use_idx = adjacent_use_stmt(list, i + 1);
            if let Some(j) = use_idx {
                // ALL reads of the temp must sit in this one adjacent
                // statement (a component load read twice in the same
                // expression, e.g. `sf3 * sf3` after
                // `sf3 = this.PlayerVelocity.z`). Moving a PURE field chain
                // into the statement that immediately follows the def is
                // control-equivalent; a read in any later statement refuses.
                let in_stmt = count_direct_reads_in(&list[j], &t);
                if in_stmt == total && in_stmt > 0 {
                    let mut replaced = 0usize;
                    super::copy_prop::for_each_expr_mut(&mut list[j], &mut |e| {
                        replaced += subst_with_expr(e, &t, &rhs);
                    });
                    if replaced >= total {
                        list.remove(i);
                        *n += replaced;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    for s in list.iter_mut() {
        for k in super::children_mut(s) {
            inline_adjacent_list(k, counts, bad, n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }
    fn sub(a: Expr, k: i64) -> Expr {
        Expr::Bin(BinOp::Sub, Box::new(a), Box::new(Expr::Int(k)))
    }
    fn assign(n: &str, e: Expr) -> Stmt {
        Stmt::Assign(n.to_string(), None, e)
    }

    #[test]
    fn duplicate_decrement_is_folded_into_the_counter() {
        let mut stmts = vec![
            assign("v331", sub(var("v46"), 1)),
            assign("v46", sub(var("v46"), 1)),
            Stmt::If {
                cond: Expr::Cmp(
                    crate::recovery::csharp::CmpKind::Ne,
                    Box::new(var("v331")),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Continue],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts);
        assert!(n >= 2, "n={n} {stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("v331"), "{d}");
        assert!(d.contains("v46"), "{d}");
    }

    #[test]
    fn control_flow_between_update_and_read_refuses() {
        let mut stmts = vec![
            assign("v1", sub(var("i"), 1)),
            assign("i", sub(var("i"), 1)),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![],
                else_body: vec![],
            },
            Stmt::Eval(var("v1")),
        ];
        apply(&mut stmts);
        let d = format!("{stmts:?}");
        assert!(d.contains("v1"), "{d}");
    }

    #[test]
    fn another_write_to_the_counter_refuses() {
        let mut stmts = vec![
            assign("v1", sub(var("i"), 1)),
            assign("i", sub(var("i"), 1)),
            assign("i", Expr::Int(7)),
            Stmt::Eval(var("v1")),
        ];
        apply(&mut stmts);
        let d = format!("{stmts:?}");
        assert!(d.contains("v1"), "{d}");
    }

    #[test]
    fn dead_generated_declaration_is_removed_but_address_taken_is_kept() {
        let mut stmts = vec![
            assign("v1", var("in_x1")),
            assign("v2", var("in_x2")),
            Stmt::Eval(Expr::AddressOf(Box::new(var("v2")))),
        ];
        apply(&mut stmts);
        let d = format!("{stmts:?}");
        assert!(!d.contains("v1"), "{d}");
        assert!(d.contains("v2"), "{d}");
    }

    #[test]
    fn nested_use_is_not_rewritten() {
        let mut stmts = vec![
            assign("v1", sub(var("i"), 1)),
            assign("i", sub(var("i"), 1)),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(var("v1"))],
                else_body: vec![],
            },
        ];
        apply(&mut stmts);
        let d = format!("{stmts:?}");
        assert!(d.contains("v1"), "{d}");
    }

    #[test]
    fn single_use_pure_temp_is_inlined() {
        let mut stmts = vec![
            assign("v1", Expr::Bin(BinOp::Add, Box::new(var("i")), Box::new(Expr::Int(1)))),
            assign("x", var("v1")),
            Stmt::Eval(var("x")),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 2, "n={n} {stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("v1"), "{d}");
        assert!(d.contains("Add"), "{d}");
    }

    #[test]
    fn single_use_temp_is_kept_when_an_operand_is_written_between() {
        let mut stmts = vec![
            assign("v1", Expr::Bin(BinOp::Add, Box::new(var("i")), Box::new(Expr::Int(1)))),
            assign("i", Expr::Int(7)),
            assign("x", var("v1")),
        ];
        apply(&mut stmts);
        let d = format!("{stmts:?}");
        assert!(d.contains("v1"), "{d}");
    }

    #[test]
    fn dead_pure_defs_next_to_an_impure_def_are_removed() {
        let mut stmts = vec![
            assign("instance", Expr::Field(Box::new(Expr::This), "A".into())),
            assign("instance", Expr::Field(Box::new(Expr::This), "B".into())),
            assign(
                "instance",
                Expr::Call {
                    recv: None,
                    name: "Box".into(),
                    args: vec![],
                },
            ),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 2, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("\"A\""), "{d}");
        assert!(!d.contains("\"B\""), "{d}");
        assert!(d.contains("Box"), "{d}");
    }

    #[test]
    fn adjacent_single_use_field_receiver_moves_into_the_check() {
        let mut stmts = vec![
            assign(
                "instance",
                Expr::Field(Box::new(Expr::This), "RewardName".into()),
            ),
            Stmt::If {
                cond: Expr::Cmp(
                    crate::recovery::csharp::CmpKind::Ne,
                    Box::new(var("instance")),
                    Box::new(Expr::Null),
                ),
                then_body: vec![Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Use".into(),
                    args: vec![var("instance")],
                })],
                else_body: vec![],
            },
        ];
        //`instance` is read TWICE here (cond + body), so the count refuses.
        let n = inline_adjacent_single_use(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
        let mut stmts = vec![
            assign(
                "instance",
                Expr::Field(Box::new(Expr::This), "RewardName".into()),
            ),
            Stmt::If {
                cond: Expr::Cmp(
                    crate::recovery::csharp::CmpKind::Ne,
                    Box::new(var("instance")),
                    Box::new(Expr::Null),
                ),
                then_body: vec![Stmt::Eval(Expr::Int(1))],
                else_body: vec![],
            },
        ];
        let n = inline_adjacent_single_use(&mut stmts);
        assert_eq!(n, 1, "{stmts:?}");
        assert_eq!(stmts.len(), 1, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("instance"), "{d}");
        assert!(d.contains("RewardName"), "{d}");
    }

    #[test]
    fn adjacent_inline_refuses_a_non_trivia_statement_between() {
        let mut stmts = vec![
            assign(
                "instance",
                Expr::Field(Box::new(Expr::This), "RewardName".into()),
            ),
            assign("x", Expr::Int(7)),
            Stmt::Eval(var("instance")),
        ];
        let n = inline_adjacent_single_use(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
        assert!(format!("{stmts:?}").contains("instance"));
    }

    #[test]
    fn adjacent_inline_refuses_calls_and_address_taken() {
        let mut stmts = vec![
            assign(
                "instance",
                Expr::Call {
                    recv: None,
                    name: "Get".into(),
                    args: vec![],
                },
            ),
            Stmt::Eval(var("instance")),
        ];
        let n = inline_adjacent_single_use(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
        let mut stmts = vec![
            assign("x", Expr::Int(1)),
            Stmt::Eval(Expr::AddressOf(Box::new(var("x")))),
            Stmt::Eval(var("x")),
        ];
        let n = inline_adjacent_single_use(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
    }
    #[test]
    fn adjacent_multi_read_component_load_is_inlined() {
        // `sf3 = this.PlayerVelocity.z; Sqrt(sf3*sf3 + …)` — both reads in the
        // immediately following statement, so moving the pure field chain
        // there is control-equivalent and removes the temp (VerifySpeed).
        let field = || {
            Expr::Field(
                Box::new(Expr::Field(Box::new(Expr::This), "PlayerVelocity".into())),
                "z".into(),
            )
        };
        let mut stmts = vec![
            assign("sf3", field()),
            Stmt::Eval(Expr::Bin(
                BinOp::Mul,
                Box::new(var("sf3")),
                Box::new(var("sf3")),
            )),
        ];
        let n = inline_adjacent_single_use(&mut stmts);
        assert_eq!(n, 2, "{stmts:?}");
        assert!(!format!("{stmts:?}").contains("sf3"));
    }

    #[test]
    fn a_read_in_a_later_statement_refuses_multi_read_inline() {
        let mut stmts = vec![
            assign("x", Expr::Field(Box::new(Expr::This), "A".into())),
            Stmt::Eval(var("x")),
            Stmt::Eval(var("x")),
        ];
        let n = inline_adjacent_single_use(&mut stmts);
        assert_eq!(n, 0, "{stmts:?}");
    }

}
