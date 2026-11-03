//! Drop statements that no path can reach.
//!
//! A list is a straight-line sequence, so anything after an unconditional
//! transfer at the same level is dead -- unless a later statement carries a
//! LABEL, because a branch from elsewhere can enter there. The rule is
//! therefore: after \`return\`/\`throw\`/\`goto\`/\`break\`/\`continue\`, delete the
//! statements up to (not including) the next \`Label\`, then keep scanning.
//!
//! ## Why these survive to the end of the pipeline
//!
//! Every structuring pass emits "the block, then its terminator" and several
//! of them splice a shared epilogue behind an early exit. When the last branch
//! to an epilogue is folded away the epilogue itself is deleted, but the
//! statements that followed the \`return\` inside the same block are not
//! anybody's business -- nothing downstream rewrites them, so they render:
//!
//! \`\`\`text
//! return *(nint*)((nint)&stack_66 +16);
//! goto L6;                 // target never emitted
//! obj2 = 0;                // dead
//! \`\`\`
//!
//! Deleting them is pure: an unreachable statement has no effect on the
//! emitted program, and the pass never touches a statement that follows a
//! label.
//!
//! ## What it refuses
//!
//! Only statements in the SAME list. A terminator nested in an \`if\` does not
//! make its siblings dead, so the walk descends into every nested list and
//! applies the rule there, independently.

use super::super::csharp::Stmt;
use super::children_mut;

fn is_terminator(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue | Stmt::YieldBreak
    )
}

fn drop_in_list(list: &mut Vec<Stmt>) -> usize {
    let mut removed = 0;
    let mut i = 0;
    while i < list.len() {
        if is_terminator(&list[i]) {
            let mut j = i + 1;
            while j < list.len() && !matches!(list[j], Stmt::Label(_)) {
                j += 1;
            }
            if j > i + 1 {
                removed += j - i - 1;
                list.drain(i + 1..j);
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    removed
}

fn walk(list: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    for s in list.iter_mut() {
        for k in children_mut(s) {
            n += walk(k);
        }
    }
    n + drop_in_list(list)
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    // The coverage-repair tail is appended to the method body AFTER its last
    // terminator on purpose: it holds blocks the structured walk could not
    // place, including the function's out-of-line cold region (exception
    // landing pads entered by the unwinder, not by a `goto`). Those statements
    // are unreachable over recovered edges but they are the method's code, and
    // they carry no label when nothing jumps to them -- so this sweep deleted
    // them and the cold region vanished from the text
    // (`a_trailing_region_is_emitted_only_when_an_edge_proves_ownership`).
    // `idioms::prune_dead_code` already splits the list at the repair marker
    // for exactly this reason; do the same here.
    let repair_start = stmts
        .iter()
        .position(|s| matches!(s, Stmt::Comment(c) if c.starts_with("[recovery] emitted ")));
    let n = match repair_start {
        Some(at) => {
            let mut tail = stmts.split_off(at);
            let n = walk(stmts);
            stmts.append(&mut tail);
            n
        }
        None => walk(stmts),
    };
    if n > 0 && std::env::var("ILR_DEADTAIL").is_ok() {
        eprintln!("[deadtail] dropped {n}");
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::Expr;

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    #[test]
    fn statements_after_return_are_dropped() {
        let mut s = vec![
            Stmt::Return(Some(var("x"))),
            Stmt::Assign("dead".into(), None, Expr::Int(0)),
            Stmt::Eval(var("y")),
        ];
        assert_eq!(apply(&mut s), 2);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn a_label_reopens_reachability() {
        let mut s = vec![
            Stmt::Return(Some(var("x"))),
            Stmt::Assign("dead".into(), None, Expr::Int(0)),
            Stmt::Label("L1".into()),
            Stmt::Return(Some(var("y"))),
        ];
        assert_eq!(apply(&mut s), 1);
        assert_eq!(s.len(), 3);
        assert!(matches!(s[1], Stmt::Label(_)));
    }

    #[test]
    fn nested_terminators_only_clear_their_own_list() {
        let mut s = vec![Stmt::If {
            cond: var("c"),
            then_body: vec![
                Stmt::Return(Some(Expr::Int(1))),
                Stmt::Assign("dead".into(), None, Expr::Int(0)),
            ],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut s), 1);
        assert_eq!(s.len(), 1);
    }
}
