//! `base.Method(...)` restoration — the NARROW, provable subset.
//!
//! ## STATUS: Two complementary approaches
//!
//! ### 1. Render-time rewriting in `csharp.rs` (Phase 5B — ACTIVE)
//!
//! `MethodCtx::call_parts` detects base calls at expression construction time
//! using metadata that was previously unavailable at this layer:
//!
//!   - The callee's `MethodRef.type_def_index` (the TDI of the type that
//!     DECLARES the callee);
//!   - The current method's `parent_chain` (the ancestor chain of the type
//!     that declares the current method);
//!   - The receiver expression (resolved from the `this` parameter local).
//!
//! When the callee's TDI is a STRICT ANCESTOR of the declaring type AND the
//! receiver is `this`, the receiver is rewritten to `Expr::Var("base")`, which
//! renders as `base.Method(args)`. This is provable, not guessed: the metadata
//! proves the callee lives on an ancestor type, and the receiver proves the
//! call is on the current object. The virtual flag is deliberately NOT checked
//! because `base.NonVirtual()` is legal C#.
//!
//! When the parent chain is empty (no ancestry metadata), `is_strict_ancestor`
//! returns false and the call is left untouched — conservative by construction.
//!
//! ### 2. Marker-based rewriting (THIS FILE — INERT BY CONSTRUCTION)
//!
//! This must not be read as a working recognizer. On the x64 corpus it reports
//! **0**, and it will keep reporting 0 until something produces the marker it
//! keys on: no stage of this pipeline emits a `base` comment. That is checkable
//! rather than assumed — every `Stmt::Comment` whose text begins with the `base`
//! token in the whole crate is written by THIS file's own tests, so the marker
//! has no producer and the pass is inert by construction, not by rarity of the
//! corpus shape. It is kept because
//! the rewrite below is exactly the transformation a metadata-aware producer
//! needs, and because the alternative — guessing the receiver — fabricates
//! callers, which is the worst class of error this port can make.
//!
//! ## Why the marker-based approach cannot decide the general case here
//!
//! IL2CPP compiles `base.M(x)` to a direct call on the base implementation. In
//! the recovered tree that surfaces as a **static-looking call whose first
//! argument is the receiver** (`M(this, x)`), because the receiver was consumed
//! as argument 0 rather than left on the call.
//!
//! Deciding that such a call really WAS `base.M` needs two facts this layer does
//! not have:
//!
//!   1. the declaring type's ANCESTRY CHAIN — the call is a base call only when
//!      the target is an ancestor's override of the same-named virtual; and
//!   2. the callee's resolved `MethodRef`, i.e. its declaring type.
//!
//! `Expr` carries neither: there is no `base` receiver variant and no
//! `MethodRef`, and a `M(this, x)` shape is equally consistent with an ordinary
//! STATIC method that takes an object (`SomeType.M(obj, x)`), which the corpus
//! renders identically. Renaming the receiver on that evidence would claim the
//! call reaches an ancestor when it does not.
//!
//! ## What the marker-based pass *can* do without inventing anything
//!
//! Act on an explicit MARKER: a `Stmt::Comment` immediately above the call whose
//! text begins with the whole token `base` (`base call`, `base: SomeBase`,
//! `base.Method`). A producer that already knew the ancestry — a metadata-aware
//! pass, or a human — writes that marker. Given it, the transformation is
//! mechanical and faithful:
//!
//!   * `M(this, x)`  ->  `base.M(x)` — the receiver moves into the call's `recv`
//!     (`Expr::Var("base")`, which renders `base.M(...)`) and the duplicated
//!     leading `this` argument is removed, because `base` already names it;
//!   * `base.M(this, x)` -> `base.M(x)` — the same de-duplication where the
//!     receiver is already `base`.
//!
//! In every other case the pass is a no-op.

use super::{Expr, Stmt, split_call, walk_mut};

/// True when a comment is a `base`-call marker.
///
/// Matches only a `base` TOKEN at the start: `base`, `base call`, `base: X`,
/// `base.X`, `base(x)`. `baseline`/`baselineShift` are NOT markers — the
/// character after `base` must be a separator or nothing.
fn is_base_marker(c: &str) -> bool {
    let t = c.trim();
    let rest = match t.strip_prefix("base").or_else(|| t.strip_prefix("Base")) {
        Some(r) => r,
        None => return false,
    };
    rest.chars()
        .next()
        .is_none_or(|ch| !(ch.is_alphanumeric() || ch == '_'))
}

/// `M(this, x)` / `base.M(this, x)` -> `base.M(x)`. Returns true when the call
/// was rewritten.
fn narrow_call(e: &mut Expr) -> bool {
    let Some((recv, _, args)) = split_call(e) else {
        return false;
    };
    // The receiver-as-first-argument shape is the whole basis of the rewrite;
    // without a leading `this` there is nothing provable to move.
    if !matches!(args.first(), Some(Expr::This)) {
        return false;
    }
    let recv_is_base = matches!(recv, Some(Expr::Var(n)) if n == "base");
    let recv_is_none = recv.is_none();
    if !(recv_is_none || recv_is_base) {
        // Any other receiver is not ours: leave it exactly as it is.
        return false;
    }
    let Expr::Call { recv, args, .. } = e else {
        return false;
    };
    // A `base.M(…)` call that still carries the receiver duplicated as arg 0:
    // de-duplicate. A static-looking call: promote the `this` into the receiver.
    if recv_is_none {
        *recv = Some(Box::new(Expr::Var("base".to_string())));
    }
    args.remove(0);
    true
}

/// Apply [`narrow_call`] to the call an expression statement carries.
fn rewrite_stmt(s: &mut Stmt) -> bool {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => narrow_call(e),
        Stmt::Return(Some(e)) => narrow_call(e),
        Stmt::Assign(_, _, e) => narrow_call(e),
        // Conditions are not `base.M(…)` call statements; skip them.
        _ => false,
    }
}

/// One statement list: a marker comment applies to the statement directly after
/// it, and to nothing else.
fn apply_list(list: &mut [Stmt]) -> usize {
    let mut n = 0;
    for i in 1..list.len() {
        let marked = matches!(&list[i - 1], Stmt::Comment(c) if is_base_marker(c.as_str()));
        if marked && rewrite_stmt(&mut list[i]) {
            n += 1;
        }
    }
    n
}

/// Rewrite `M(this, x)` to `base.M(x)` where a `base` marker comment proves the
/// receiver is the base implementation.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        n += apply_list(list);
    });
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::reconstruct::is_inlinable;

    fn call(recv: Option<Expr>, name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            recv: recv.map(Box::new),
            name: name.to_string(),
            args,
        }
    }

    /// `base: SomeBase` + `OnEnter(this, prev)` -> `base.OnEnter(prev)`.
    #[test]
    fn marker_makes_a_static_looking_call_a_base_call() {
        let mut stmts = vec![
            Stmt::Comment("base: SomeBase".to_string()),
            Stmt::Eval(call(
                None,
                "OnEnter",
                vec![Expr::This, Expr::Var("prev".to_string())],
            )),
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[1] {
            Stmt::Eval(Expr::Call { recv, args, .. }) => {
                assert!(
                    matches!(recv.as_deref(), Some(Expr::Var(v)) if v == "base"),
                    "receiver must become `base`, got {recv:?}"
                );
                assert_eq!(args.len(), 1, "the duplicated `this` must be dropped");
                assert!(matches!(&args[0], Expr::Var(v) if v == "prev"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    /// A redundant `this` on an already-`base` call is dropped.
    #[test]
    fn a_redundant_this_is_dropped_from_an_already_base_call() {
        let mut stmts = vec![
            Stmt::Comment("base call".to_string()),
            Stmt::Eval(call(
                Some(Expr::Var("base".to_string())),
                "M",
                vec![Expr::This, Expr::Var("x".to_string())],
            )),
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[1] {
            Stmt::Eval(Expr::Call { args, .. }) => assert_eq!(args.len(), 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    /// Without a marker the pass must not touch the call — a static helper
    /// taking an object looks identical.
    #[test]
    fn without_a_marker_the_call_is_untouched() {
        let mut stmts = vec![Stmt::Eval(call(None, "M", vec![Expr::This]))];
        let before = format!("{:?}", stmts);
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(before, format!("{:?}", stmts));
        match &stmts[0] {
            Stmt::Eval(Expr::Call { recv, args, .. }) => {
                assert!(recv.is_none());
                assert_eq!(args.len(), 1);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    /// A marker over a call whose first argument is NOT `this` proves nothing —
    /// refuse rather than invent a receiver.
    #[test]
    fn a_marker_over_a_call_without_a_receiver_argument_is_refused() {
        let mut stmts = vec![
            Stmt::Comment("base".to_string()),
            Stmt::Eval(call(None, "M", vec![Expr::Var("x".to_string())])),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A marker over a call on a DIFFERENT receiver is not a base call.
    #[test]
    fn a_marker_over_a_foreign_receiver_is_refused() {
        let mut stmts = vec![
            Stmt::Comment("base".to_string()),
            Stmt::Eval(call(
                Some(Expr::Var("other".to_string())),
                "M",
                vec![Expr::This],
            )),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// `baseline…` is not a `base` token.
    #[test]
    fn a_non_base_comment_is_not_a_marker() {
        let mut stmts = vec![
            Stmt::Comment("baseline shift".to_string()),
            Stmt::Eval(call(None, "M", vec![Expr::This])),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert!(is_base_marker("base"));
        assert!(is_base_marker("Base call"));
        assert!(!is_base_marker("baseName"));
    }

    /// The marker applies to the NEXT statement only.
    #[test]
    fn a_marker_does_not_leak_past_the_next_statement() {
        let mut stmts = vec![
            Stmt::Comment("base".to_string()),
            Stmt::Eval(Expr::Var("noop".into())),
            Stmt::Eval(call(None, "M", vec![Expr::This])),
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert!(is_inlinable(&stmts[1]));
    }
}
