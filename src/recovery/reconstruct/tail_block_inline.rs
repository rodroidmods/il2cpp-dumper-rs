//! Inline a single-reference tail block that a branch jumps to.
//!
//! The compiler hoists a small block out of a branch and jumps to it from one
//! arm, then back to the join:
//!
//! ```text
//! if (jump == 0) {
//!     if (stumble == 0) { v = …; }
//!     else { goto L2; }          // ← one arm jumps out
//!     L3:
//!     … rest …
//! }
//! L2:
//! v = 0;
//! goto L3;                       // ← back to the join
//! ```
//!
//! C# cannot jump into the middle of the block. The block is reached by
//! EXACTLY one goto, so moving its statements into that arm is
//! control-equivalent (only that path executes them, in the same order), and
//! when its final `goto` targets the label immediately following the `if`,
//! the goto disappears and the arm falls through. The emptied label block is
//! removed; the join label is dropped later by `drop_unreferenced_labels`.
//!
//! Refusals: a target referenced more than once, a block containing labels,
//! nested control flow, or non-straight statements, a block that is not a
//! single tail (must end in goto/return/throw), and any block living inside
//! the arm being rewritten.

use crate::recovery::csharp::Stmt;
use std::collections::HashMap;

type Path = Vec<(usize, usize)>;

struct Block {
    path: Path,
    label_idx: usize,
    end_idx: usize,
    final_goto: Option<String>,
}

fn is_trivia(s: &Stmt) -> bool {
    matches!(s, Stmt::Comment(_) | Stmt::Attribute { .. })
}

fn single_goto(body: &[Stmt]) -> Option<&str> {
    let mut hit = None;
    for s in body {
        if is_trivia(s) {
            continue;
        }
        if hit.is_some() {
            return None;
        }
        match s {
            Stmt::Goto(l) => hit = Some(l.as_str()),
            _ => return None,
        }
    }
    hit
}

fn straight(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Assign(..)
            | Stmt::IndexAssign { .. }
            | Stmt::EventAssign { .. }
            | Stmt::Eval(_)
            | Stmt::Comment(_)
    )
}

fn terminator(s: &Stmt) -> Option<Option<String>> {
    match s {
        Stmt::Goto(l) => Some(Some(l.clone())),
        Stmt::Return(_) | Stmt::Throw(_) => Some(None),
        _ => None,
    }
}

fn next_label(list: &[Stmt], after: usize) -> Option<String> {
    for s in &list[after + 1..] {
        if is_trivia(s) {
            continue;
        }
        return match s {
            Stmt::Label(l) => Some(l.clone()),
            _ => None,
        };
    }
    None
}

fn collect(
    list: &[Stmt],
    path: &Path,
    goto_counts: &mut HashMap<String, usize>,
    blocks: &mut HashMap<String, Block>,
    candidates: &mut Vec<(Path, usize, bool, String, Option<String>)>,
) {
    for (i, s) in list.iter().enumerate() {
        if let Stmt::Goto(l) = s {
            *goto_counts.entry(l.clone()).or_default() += 1;
        }
        if let Stmt::Label(l) = s {
            let mut end = None;
            for (j, t) in list.iter().enumerate().skip(i + 1) {
                if is_trivia(t) {
                    continue;
                }
                if straight(t) {
                    continue;
                }
                if terminator(t).is_some() {
                    end = Some(j);
                }
                break;
            }
            if let Some(end_idx) = end {
                blocks.insert(
                    l.clone(),
                    Block {
                        path: path.clone(),
                        label_idx: i,
                        end_idx,
                        final_goto: terminator(&list[end_idx]).flatten(),
                    },
                );
            }
        }
        if let Stmt::If {
            then_body,
            else_body,
            ..
        } = s
        {
            let next = next_label(list, i);
            if let Some(a) = single_goto(then_body) {
                candidates.push((path.clone(), i, true, a.to_string(), next.clone()));
            }
            if let Some(a) = single_goto(else_body) {
                candidates.push((path.clone(), i, false, a.to_string(), next));
            }
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for (slot, child) in kids.into_iter().enumerate() {
            let mut child_path = path.clone();
            child_path.push((i, slot));
            collect(child, &child_path, goto_counts, blocks, candidates);
        }
    }
}

fn with_list_mut(stmts: &mut Vec<Stmt>, path: &[(usize, usize)], f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    if path.is_empty() {
        f(stmts);
        return;
    }
    let (i, slot) = path[0];
    if let Some(s) = stmts.get_mut(i) {
        let mut kids: Vec<&mut Vec<Stmt>> = super::children_mut(s);
        if slot < kids.len() {
            let child = kids.swap_remove(slot);
            with_list_mut(child, &path[1..], f);
        }
    }
}

fn prefix(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
    a.len() <= b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// Apply the fold once; returns true when a block was inlined.
fn try_once(stmts: &mut Vec<Stmt>) -> bool {
    let mut goto_counts: HashMap<String, usize> = HashMap::new();
    let mut blocks: HashMap<String, Block> = HashMap::new();
    let mut candidates: Vec<(Path, usize, bool, String, Option<String>)> = Vec::new();
    let root_path: Path = Vec::new();
    collect(
        stmts,
        &root_path,
        &mut goto_counts,
        &mut blocks,
        &mut candidates,
    );
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    for (if_path, if_idx, then_arm, a, next) in candidates {
        if goto_counts.get(&a).copied().unwrap_or(0) != 1 {
            continue;
        }
        let Some(blk) = blocks.get(&a) else {
            continue;
        };
        // A block inside the arm being rewritten would be moved into itself.
        if prefix(&if_path, &blk.path) {
            continue;
        }
        let (block_path, label_idx, end_idx, final_goto) = (
            blk.path.clone(),
            blk.label_idx,
            blk.end_idx,
            blk.final_goto.clone(),
        );
        let mut body: Vec<Stmt> = {
            // Collect the block's statements through the path lookup.
            let mut out = Vec::new();
            with_list_mut(stmts, &block_path, &mut |l| {
                out.extend(l[label_idx + 1..=end_idx].iter().cloned());
            });
            out
        };
        if body.is_empty() {
            continue;
        }
        if let Some(g) = &final_goto {
            if next.as_deref() == Some(g.as_str()) {
                body.pop();
            }
        }
        // Rewrite the arm.
        let mut ok = false;
        with_list_mut(stmts, &if_path, &mut |l| {
            let Some(Stmt::If {
                then_body,
                else_body,
                ..
            }) = l.get_mut(if_idx)
            else {
                return;
            };
            if then_arm {
                *then_body = body.clone();
            } else {
                *else_body = body.clone();
            }
            ok = true;
        });
        if !ok {
            continue;
        }
        // Remove the emptied block (label + tail) from its list.
        with_list_mut(stmts, &block_path, &mut |l| {
            l.drain(label_idx..=end_idx);
        });
        return true;
    }
    false
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    for _ in 0..256 {
        if !try_once(stmts) {
            break;
        }
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> crate::recovery::csharp::Expr {
        crate::recovery::csharp::Expr::Var(n.to_string())
    }

    #[test]
    fn a_single_reference_tail_block_is_inlined_and_the_join_goto_drops() {
        let inner = Stmt::If {
            cond: crate::recovery::csharp::Expr::Bool(true),
            then_body: vec![Stmt::Assign("v".into(), None, crate::recovery::csharp::Expr::Int(1))],
            else_body: vec![Stmt::Goto("L2".into())],
        };
        let jump = Stmt::If {
            cond: crate::recovery::csharp::Expr::Bool(true),
            then_body: vec![inner, Stmt::Label("L3".into()), Stmt::Eval(var("x"))],
            else_body: vec![],
        };
        let mut stmts = vec![
            jump,
            Stmt::Label("L2".into()),
            Stmt::Assign("v".into(), None, crate::recovery::csharp::Expr::Int(0)),
            Stmt::Goto("L3".into()),
            Stmt::Return(None),
        ];
        let n = apply(&mut stmts);
        assert!(n >= 1, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(!d.contains("Goto(\"L2\")"), "{d}");
        assert!(!d.contains("Goto(\"L3\")"), "{d}");
        assert!(d.contains("Int(0)"), "{d}");
    }

    #[test]
    fn a_shared_tail_block_is_not_moved() {
        let inner = Stmt::If {
            cond: crate::recovery::csharp::Expr::Bool(true),
            then_body: vec![],
            else_body: vec![Stmt::Goto("L2".into())],
        };
        let mut stmts = vec![
            inner,
            Stmt::Goto("L2".into()),
            Stmt::Label("L2".into()),
            Stmt::Assign("v".into(), None, crate::recovery::csharp::Expr::Int(0)),
            Stmt::Return(None),
        ];
        apply(&mut stmts);
        let d = format!("{stmts:?}");
        assert!(d.contains("Goto(\"L2\")"), "{d}");
    }
}
