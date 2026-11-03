//! High-levelize frame-cell spills.
//!
//! IL2CPP spills locals to the native frame; the structurer renders every
//! store as a memory write through a `stack_N` cell:
//!
//! ```text
//! *(int*)((nint)&stack_420 +4) = bGenerateNormals & 1;
//! …
//! if ((*(nint*)((nint)&stack_420 +4) & !geoCache._normalAttrInfo) != 0)
//! ```
//!
//! Two rewrites, both proof-gated:
//!
//! 1. **Dead spill store.** The cell is never read (`RawDeref(&cell, off)`),
//!    never used as a whole value, and its address never escapes. The store is
//!    unobservable; a side-effecting RHS survives as `Eval(rhs)`.
//! 2. **Forwarding.** The stored RHS is a SIMPLE LOCAL EXPRESSION (`Var`,
//!    constants, casts, arithmetic/compare over them — no fields, calls or raw
//!    memory) and every read of that `(cell, off)` in the METHOD appears in the
//!    scanned straight-line window and is rewritten with that expression. If
//!    any read remains (branch body, label, loop), the store stays.
//!
//! Reads are `RawDeref(AddressOf(Var(cell)), off)`. Bare `stack_N` value reads
//! and any address use (call argument, struct receiver, `.Length`) disable both
//! rewrites for that cell — a callee could observe the frame memory.
//!
//! RHS safety: forwarding stops if an intervening assignment writes a name the
//! RHS reads. Calls are allowed because the RHS is composed only of locals and
//! constants (not fields or memory), so a call cannot change its value; a call
//! also cannot reach the cell because its address never escapes.

use super::super::csharp::{Expr, Stmt};
use super::children_mut;
use std::collections::{HashMap, HashSet};

fn is_stack_name(n: &str) -> bool {
    n.starts_with("stack_") && n[6..].chars().all(|c| c.is_ascii_digit()) && n.len() > 6
}

/// `*(int*)((nint)&stack_424 +4)` → ("stack_424", 4).
fn parse_store_target(target: &str) -> Option<(String, i64)> {
    if !target.starts_with("*(") {
        return None;
    }
    let amp = target.find("&stack_")?;
    let rest = &target[amp + 1..];
    let end = rest
        .char_indices()
        .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_'))
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    let cell = &rest[..end];
    if !is_stack_name(cell) {
        return None;
    }
    let tail = &rest[end..];
    let off = tail
        .find('+')
        .and_then(|p| {
            let digits: String = tail[p + 1..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse::<i64>().ok()
        })
        .unwrap_or(0);
    Some((cell.to_string(), off))
}

#[derive(Default)]
struct Reads {
    loads: HashMap<(String, i64), usize>,
    value_read: HashSet<String>,
    address_used: HashSet<String>,
    /// Plain-name assignment targets anywhere in the method (for the
    /// dominating-spill rule: a stored simple RHS must never be rewritten).
    assigned: HashSet<String>,
}

fn scan_expr(e: &Expr, r: &mut Reads) {
    if let Expr::RawDeref(base, off, _) = e {
        if let Expr::AddressOf(inner) = base.as_ref() {
            if let Expr::Var(cell) = inner.as_ref() {
                if is_stack_name(cell) {
                    *r.loads.entry((cell.clone(), *off)).or_insert(0) += 1;
                    return;
                }
            }
        }
    }
    match e {
        Expr::Var(n) if is_stack_name(n) => {
            r.value_read.insert(n.clone());
        }
        Expr::AddressOf(inner) => {
            if let Expr::Var(cell) = inner.as_ref() {
                if is_stack_name(cell) {
                    r.address_used.insert(cell.clone());
                }
            }
            scan_expr(inner, r);
            return;
        }
        _ => {}
    }
    for c in expr_children(e) {
        scan_expr(c, r);
    }
}

fn scan_stmt(s: &Stmt, r: &mut Reads) {
    if let Stmt::Assign(t, _, _) = s {
        if !t.contains('.') && !t.contains('*') && !t.contains('[') {
            if let Some(b) = target_base(t) {
                r.assigned.insert(b);
            }
        }
    }
    for e in direct_exprs(s) {
        scan_expr(e, r);
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            scan_stmt(x, r);
        }
    }
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

fn expr_children(e: &Expr) -> Vec<&Expr> {
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
        Expr::Index { collection, index } => vec![collection, index],
        Expr::NullConditional { receiver, access } => vec![receiver, access],
        Expr::Call { recv, args, .. } => {
            let mut v: Vec<&Expr> = args.iter().collect();
            if let Some(r) = recv {
                v.push(r);
            }
            v
        }
        Expr::RawCall(_, args, _)
        | Expr::NewObj(_, args)
        | Expr::NewArr(_, args)
        | Expr::ArrayInit { items: args, .. } => args.iter().collect(),
        _ => vec![],
    }
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
        Expr::Index { collection, index } => vec![collection, index],
        Expr::NullConditional { receiver, access } => vec![receiver, access],
        Expr::Call { recv, args, .. } => {
            let mut v: Vec<&mut Expr> = args.iter_mut().collect();
            if let Some(r) = recv {
                v.push(r);
            }
            v
        }
        Expr::RawCall(_, args, _)
        | Expr::NewObj(_, args)
        | Expr::NewArr(_, args)
        | Expr::ArrayInit { items: args, .. } => args.iter_mut().collect(),
        _ => vec![],
    }
}

/// Names a simple local RHS depends on. `None` when the RHS is not safe to
/// duplicate at a later point (field/memory/call expressions can change).
fn simple_rhs_names(e: &Expr, out: &mut HashSet<String>) -> bool {
    match e {
        Expr::Var(n) => {
            if is_stack_name(n) || n.contains('.') {
                return false;
            }
            out.insert(n.clone());
            true
        }
        Expr::Int(_) | Expr::Bool(_) | Expr::Null | Expr::Str(_) => true,
        Expr::Cast(_, inner) | Expr::Not(inner) | Expr::Neg(inner) => {
            simple_rhs_names(inner, out)
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            simple_rhs_names(a, out) && simple_rhs_names(b, out)
        }
        Expr::Cond(c, a, b) => {
            simple_rhs_names(c, out)
                && simple_rhs_names(a, out)
                && simple_rhs_names(b, out)
        }
        _ => false,
    }
}

fn target_base(target: &str) -> Option<String> {
    let t = target.trim_start_matches("this.").trim();
    let base: String = t
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    (!base.is_empty()).then_some(base)
}

struct Pending {
    rhs: Expr,
    names: HashSet<String>,
    store_idx: usize,
    rewritten: usize,
}

/// Replace `*(&cell + off)` loads with `rhs`; returns how many were replaced.
fn rewrite_loads_expr(e: &mut Expr, cell: &str, off: i64, rhs: &Expr) -> usize {
    if let Expr::RawDeref(base, o, _) = e {
        if *o == off {
            if let Expr::AddressOf(inner) = base.as_ref() {
                if let Expr::Var(c) = inner.as_ref() {
                    if c == cell {
                        *e = rhs.clone();
                        return 1;
                    }
                }
            }
        }
    }
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += rewrite_loads_expr(c, cell, off, rhs);
    }
    n
}

fn rewrite_loads_stmt(s: &mut Stmt, pending: &mut HashMap<(String, i64), Pending>) {
    let mut fold = |e: &mut Expr| {
        for ((cell, off), p) in pending.iter_mut() {
            p.rewritten += rewrite_loads_expr(e, cell, *off, &p.rhs);
        }
    };
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Return(Some(e)) => fold(e),
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
}

fn finalize(p: Pending, key: &(String, i64), reads: &Reads, drop: &mut HashSet<usize>) {
    let total = reads.loads.get(key).copied().unwrap_or(0);
    if total > 0 && p.rewritten == total {
        drop.insert(p.store_idx);
    }
}

fn forwarding_list(stmts: &mut Vec<Stmt>, reads: &Reads, removed: &mut usize) {
    let mut pending: HashMap<(String, i64), Pending> = HashMap::new();
    let mut drop: HashSet<usize> = HashSet::new();
    let mut i = 0;
    while i < stmts.len() {
        // A new store to a key finalizes the pending one.
        if let Stmt::Assign(t, _, rhs) = &stmts[i] {
            if let Some((cell, off)) = parse_store_target(t) {
                let key = (cell.clone(), off);
                if let Some(prev) = pending.remove(&key) {
                    finalize(prev, &key, reads, &mut drop);
                }
                let mut names = HashSet::new();
                let simple = simple_rhs_names(rhs, &mut names)
                    && !reads.value_read.contains(&cell)
                    && !reads.address_used.contains(&cell);
                if simple {
                    pending.insert(
                        key,
                        Pending {
                            rhs: rhs.clone(),
                            names,
                            store_idx: i,
                            rewritten: 0,
                        },
                    );
                }
                i += 1;
                continue;
            }
        }
        // Any assignment to a name the pending RHS reads invalidates it.
        // A memory write through a complex target CANNOT change a simple-local
        // RHS (fields/calls are rejected by `simple_rhs_names`), so it must not
        // clear pendings — the spill-forwarding window spans the method's
        // pointer stores and helper calls by design.
        if let Stmt::Assign(t, _, _) = &stmts[i] {
            if let Some(base) = target_base(t) {
                if !t.contains('.') && !t.contains('*') && !t.contains('[') {
                    pending.retain(|_, p| !p.names.contains(&base));
                }
            }
        }
        let barrier = matches!(
            stmts[i],
            Stmt::Label(_)
                | Stmt::Goto(_)
                | Stmt::GotoCase(_)
                | Stmt::Break
                | Stmt::Continue
                | Stmt::Return(_)
                | Stmt::Throw(_)
                | Stmt::YieldReturn(_)
                | Stmt::YieldBreak
        );
        if barrier {
            for (key, p) in pending.drain() {
                finalize(p, &key, reads, &mut drop);
            }
            i += 1;
            continue;
        }
        if !pending.is_empty() {
            rewrite_loads_stmt(&mut stmts[i], &mut pending);
        }
        // Loop tests may re-run after the RHS local changes.
        if matches!(
            stmts[i],
            Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. }
        ) {
            for (key, p) in pending.drain() {
                finalize(p, &key, reads, &mut drop);
            }
        }
        i += 1;
    }
    for (key, p) in pending.drain() {
        finalize(p, &key, reads, &mut drop);
    }
    if !drop.is_empty() {
        let mut idx = 0;
        stmts.retain(|_| {
            let keep = !drop.contains(&idx);
            idx += 1;
            keep
        });
        *removed += drop.len();
    }
}

fn remove_dead_stores(stmts: &mut Vec<Stmt>, reads: &Reads, removed: &mut usize) {
    let mut i = 0;
    while i < stmts.len() {
        let dead = if let Stmt::Assign(t, _, _) = &stmts[i] {
            parse_store_target(t).is_some_and(|(cell, off)| {
                !reads.loads.contains_key(&(cell.clone(), off))
                    && !reads.address_used.contains(&cell)
                    && !reads.value_read.contains(&cell)
            })
        } else {
            false
        };
        if dead {
            let rhs = match &stmts[i] {
                Stmt::Assign(_, _, rhs) => rhs.clone(),
                _ => unreachable!(),
            };
            if super::super::csharp::cleanup::expr_has_side_effect(&rhs) {
                stmts[i] = Stmt::Eval(rhs);
                i += 1;
            } else {
                stmts.remove(i);
                *removed += 1;
            }
        } else {
            i += 1;
        }
    }
}

fn process_list(stmts: &mut Vec<Stmt>, reads: &Reads, removed: &mut usize) {
    forwarding_list(stmts, reads, removed);
    remove_dead_stores(stmts, reads, removed);
    for s in stmts.iter_mut() {
        for k in children_mut(s) {
            process_list(k, reads, removed);
        }
    }
}

/// Stores at the top of the method (before any label) execute on every path;
/// their value is available to loads anywhere, including nested branches.
fn global_stores(
    stmts: &[Stmt],
    reads: &Reads,
) -> HashMap<(String, i64), (Expr, usize)> {
    let mut map: HashMap<(String, i64), (Expr, usize)> = HashMap::new();
    for (i, s) in stmts.iter().enumerate() {
        if matches!(s, Stmt::Label(_) | Stmt::Goto(_) | Stmt::GotoCase(_)) {
            break;
        }
        let Stmt::Assign(t, _, rhs) = s else { continue };
        let Some((cell, off)) = parse_store_target(t) else { continue };
        let key = (cell.clone(), off);
        if reads.address_used.contains(&cell) || reads.value_read.contains(&cell) {
            continue;
        }
        let mut names = HashSet::new();
        if !simple_rhs_names(rhs, &mut names) {
            continue;
        }
        if names.iter().any(|n| reads.assigned.contains(n)) {
            continue;
        }
        // Last store in the dominating region wins; an earlier one is
        // overwritten before every load.
        map.insert(key, (rhs.clone(), i));
    }
    map
}

fn rewrite_global_expr(
    e: &mut Expr,
    global: &HashMap<(String, i64), (Expr, usize)>,
    counts: &mut HashMap<(String, i64), usize>,
) {
    for ((cell, off), (rhs, _)) in global.iter() {
        let n = rewrite_loads_expr(e, cell, *off, rhs);
        if n > 0 {
            *counts.entry((cell.clone(), *off)).or_insert(0) += n;
        }
    }
    for c in expr_children_mut(e) {
        rewrite_global_expr(c, global, counts);
    }
}

fn rewrite_global_list(
    stmts: &mut Vec<Stmt>,
    global: &HashMap<(String, i64), (Expr, usize)>,
    counts: &mut HashMap<(String, i64), usize>,
    skip: &HashSet<usize>,
    top: bool,
) {
    for (i, s) in stmts.iter_mut().enumerate() {
        if top && skip.contains(&i) {
            continue;
        }
        let mut fold = |e: &mut Expr| rewrite_global_expr(e, global, counts);
        match s {
            Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Return(Some(e)) => fold(e),
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
        for k in children_mut(s) {
            rewrite_global_list(k, global, counts, skip, false);
        }
    }
}

fn expr_mentions_stack(e: &Expr) -> bool {
    if let Expr::Var(n) = e {
        if is_stack_name(n) {
            return true;
        }
    }
    expr_children(e).into_iter().any(expr_mentions_stack)
}

fn has_stack_work(stmts: &[Stmt]) -> bool {
    for s in stmts {
        if let Stmt::Assign(t, _, _) = s {
            if t.contains("&stack_") {
                return true;
            }
        }
        if direct_exprs(s).into_iter().any(|e| expr_mentions_stack(e)) {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            if has_stack_work(k) {
                return true;
            }
        }
    }
    false
}

/// Run the pass. Returns the number of removed spill stores.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    // 90% of methods contain no frame cell at all; skip the five AST walks.
    if !has_stack_work(stmts) {
        return 0;
    }
    let mut reads = Reads::default();
    for s in stmts.iter() {
        scan_stmt(s, &mut reads);
    }
    let mut removed = 0;
    // Dominating top-of-method spills: rewrite their loads anywhere.
    let global = global_stores(stmts, &reads);
    if !global.is_empty() {
        let skip: HashSet<usize> = global.values().map(|(_, i)| *i).collect();
        let mut counts: HashMap<(String, i64), usize> = HashMap::new();
        rewrite_global_list(stmts, &global, &mut counts, &skip, true);
        let mut drop: HashSet<usize> = HashSet::new();
        for (key, (_, idx)) in &global {
            let total = reads.loads.get(key).copied().unwrap_or(0);
            if total > 0 && counts.get(key).copied().unwrap_or(0) == total {
                drop.insert(*idx);
            }
        }
        if !drop.is_empty() {
            let mut i = 0;
            stmts.retain(|_| {
                let keep = !drop.contains(&i);
                i += 1;
                keep
            });
            removed += drop.len();
        }
    }
    process_list(stmts, &reads, &mut removed);
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(cell: &str, off: i64, rhs: Expr) -> Stmt {
        Stmt::Assign(format!("*(int*)((nint)&{cell} +{off})"), None, rhs)
    }
    fn load(cell: &str, off: i64) -> Expr {
        Expr::RawDeref(
            Box::new(Expr::AddressOf(Box::new(Expr::Var(cell.to_string())))),
            off,
            0,
        )
    }

    #[test]
    fn unobserved_store_is_removed() {
        let mut stmts = vec![store("stack_1", 4, Expr::Int(0))];
        assert_eq!(apply(&mut stmts), 1);
        assert!(stmts.is_empty());
    }

    #[test]
    fn store_with_escaped_cell_is_kept() {
        let mut stmts = vec![
            store("stack_1", 4, Expr::Int(0)),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "foo".into(),
                args: vec![Expr::AddressOf(Box::new(Expr::Var("stack_1".into())))],
            }),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn simple_value_is_forwarded_into_a_condition() {
        let mut stmts = vec![
            store("stack_1", 4, Expr::Var("flag".into())),
            Stmt::If {
                cond: Expr::Cmp(
                    super::super::super::csharp::CmpKind::Ne,
                    Box::new(load("stack_1", 4)),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        let t = format!("{stmts:?}");
        assert!(!t.contains("RawDeref"), "{t}");
        assert!(t.contains("flag"), "{t}");
    }

    #[test]
    fn dominating_store_forwards_into_a_branch_body() {
        // The store is the first statement (no label before it), so it runs on
        // every path; the load inside the branch is dominated by it.
        let mut stmts = vec![
            store("stack_1", 4, Expr::Var("flag".into())),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(load("stack_1", 4))],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(!format!("{stmts:?}").contains("RawDeref"));
    }

    #[test]
    fn store_after_a_label_keeps_a_branch_body_load() {
        let mut stmts = vec![
            Stmt::Label("L0".into()),
            store("stack_1", 4, Expr::Var("flag".into())),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Eval(load("stack_1", 4))],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }
}
