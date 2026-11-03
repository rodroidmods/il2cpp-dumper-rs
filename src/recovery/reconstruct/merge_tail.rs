//! Merge a tail duplicated into BOTH arms of an `if` back out of the `if`.
//!
//! ## Shape
//!
//! The IR duplicator (`split_loops::split_shared_tails`) makes the CFG a tree
//! by giving every branch its own copy of a shared continuation, so the
//! structurer can place that code without a jump. The copies are identical, so
//! the emitted tree has:
//!
//! ```text
//! if (c) { A; T } else { B; T }
//! ```
//!
//! which is exactly the source shape the compiler produced before it merged
//! `T`. Moving the shared suffix after the `if` restores the source:
//!
//! ```text
//! if (c) { A } else { B }
//! T;
//! ```
//!
//! ## Why this is safe
//!
//! * The suffix is a contiguous tail of each arm, so every path that executed
//!   it before still does — now by falling out of the `if`.
//! * Both arms keep their own code, and the suffix is executed once on the
//!   merged path. Nothing is duplicated and nothing is deleted.
//! * Refused when the suffix contains a label, a jump, `break`/`continue`,
//!   or a typed declaration: those change meaning or scope when moved.
//! * Refused when a suffix statement mentions a name DECLARED inside either
//!   arm before the suffix — that declaration would no longer be in scope.
//! * Refused when both arms would become empty (the `if` would be dropped,
//!   taking a condition that may have side effects with it).

use super::super::csharp::Stmt;
use super::children_mut;
use std::collections::HashSet;

/// Longest suffix this pass will move. A long shared tail is still one tail;
/// the cap only keeps the comparison from scanning huge arms.
const MAX_TAIL: usize = 64;

fn key(s: &Stmt) -> String {
    format!("{s:?}")
}

/// May this statement be moved out of an arm unchanged?
fn movable(s: &Stmt) -> bool {
    !matches!(
        s,
        Stmt::Label(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue | Stmt::GotoCase(_)
    ) && !matches!(s, Stmt::Assign(_, Some(_), _))
}

/// Names declared (typed assignment) anywhere in `body`.
fn declared(body: &[Stmt]) -> HashSet<String> {
    let mut out = HashSet::new();
    fn walk(stmts: &[Stmt], out: &mut HashSet<String>) {
        for s in stmts {
            if let Stmt::Assign(name, Some(_), _) = s {
                out.insert(name.clone());
            }
            let mut ch: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut ch);
            for c in ch {
                walk(c, out);
            }
        }
    }
    walk(body, &mut out);
    out
}

/// Does `s` mention `name` as a token?
fn mentions(s: &Stmt, name: &str) -> bool {
    key(s)
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|t| t == name)
}

fn apply_list(stmts: &mut Vec<Stmt>) -> usize {
    let mut total = 0;
    for s in stmts.iter_mut() {
        for body in children_mut(s) {
            total += apply_list(body);
        }
    }
    let mut i = 0;
    while i < stmts.len() {
        let mut hoisted: Option<Vec<Stmt>> = None;
        if let Stmt::If {
            then_body,
            else_body,
            ..
        } = &mut stmts[i]
        {
            if !then_body.is_empty() && !else_body.is_empty() {
                let limit = then_body.len().min(else_body.len()).min(MAX_TAIL);
                let mut n = 0usize;
                while n < limit {
                    let a = &then_body[then_body.len() - 1 - n];
                    let b = &else_body[else_body.len() - 1 - n];
                    if key(a) != key(b) || !movable(a) {
                        break;
                    }
                    n += 1;
                }
                // Never merge the WHOLE of both arms: the `if` would vanish
                // and its condition (possibly a call) with it.
                if n > 0 && n < then_body.len().max(else_body.len()) {
                    let tail = then_body[then_body.len() - n..].to_vec();
                    let decl: HashSet<String> = declared(&then_body[..then_body.len() - n])
                        .into_iter()
                        .chain(declared(&else_body[..else_body.len() - n]))
                        .collect();
                    let scope_ok = !tail
                        .iter()
                        .any(|s| decl.iter().any(|d| mentions(s, d)));
                    if scope_ok {
                        then_body.truncate(then_body.len() - n);
                        else_body.truncate(else_body.len() - n);
                        hoisted = Some(tail);
                    }
                }
            }
        }
        match hoisted {
            Some(tail) => {
                total += 1;
                stmts.splice(i + 1..i + 1, tail);
                i += 1;
            }
            None => i += 1,
        }
    }
    total
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    apply_list(stmts)
}