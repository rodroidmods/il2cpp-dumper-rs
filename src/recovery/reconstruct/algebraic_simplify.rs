//! Algebraic, boolean and constant-folding simplifications on the statement
//! tree — the C#-level counterpart of Ghidra's simplification rule pool
//! (ported here from Rudra's `ruleaction.rs` / `condexe.rs`).
//!
//! ## Why a separate pass
//!
//! The structurer emits the *machine* shape of every condition and arithmetic
//! chain: `(v & 1) != 0` for boolean tests, `!(a == b)` for negated guards,
//! `0 == x` from an inverted compare, `x + 0` / `x * 1` from addressing
//! artifacts, `(a <= b) && (a != b)` from `IComparable` lowering. The
//! construct recognizers (`equality_ops`, `null_coalesce`, …) each know one
//! idiom; this pass applies the general algebra the same way Ghidra's
//! `oppool1` does, iterated to a fixed point.
//!
//! ## Rules (all soundness gates are deliberate — see below)
//!
//! Constant folding:
//!   * `Bin(Int a, Int b)` for wrapping arithmetic + masks/shifts in range.
//!   * `Cmp(Int a, Int b)` → `Bool`.
//!   * `Neg(Int)` → `Int`.
//!   * `Bin(AndAlso/OrElse, Bool, Bool)` → `Bool`.
//!
//! Identities (the KEPT operand may be impure — it is still evaluated; only a
//! DROPPED operand has to be pure):
//!   * `x + 0`, `x - 0`, `x * 1`, `x / 1`, `x | 0`, `x ^ 0`, `x << 0`,
//!     `x >> 0` → `x`.
//!   * `x * 0`, `x & 0` → `0` (needs `x` pure: the whole operand is dropped).
//!   * `x - x`, `x ^ x` → `0` (needs `x` pure).
//!   * `x && true`, `x || false` → `x`;
//!     `x && false` → `false`, `x || true` → `true` (needs `x` pure).
//!
//! Comparisons:
//!   * `Cmp(k, Int c, rhs)` → `Cmp(swap(k), rhs, Int c)` — constant on the
//!     right (Ghidra `RuleTermOrder`; comparison-operand swap is exact,
//!     including NaN).
//!   * `Eq(x, x)`, `Le(x, x)`, `Ge(x, x)` → `true`; `Ne/Lt/Gt(x, x)` →
//!     `false` — ONLY for operands that are provably non-float (`x == x` is
//!     false for NaN; a Var could be a float, so Vars are excluded).
//!   * `(a <= b) && (a != b)` → `a < b`; `(a >= b) && (a != b)` → `a > b`;
//!     `(a < b) || (a == b)` → `a <= b`; `(a > b) || (a == b)` → `a >= b`
//!     (Ghidra `RuleLessNotEqual`/`RuleLessEqual`; these hold for NaN too).
//!   * `!(a == b)` → `a != b`, `!(a != b)` → `a == b` (`RuleBoolNegate`,
//!     restricted to equality: negating an ORDERING is not NaN-safe).
//!   * `!!x` → `x`.
//!   * `Eq(e, true)`, `Ne(e, false)` → `e` for boolean-shaped `e`.
//!
//! Boolean hygiene:
//!   * `Bin(And, Cmp|Not|Bool, …)` → `&&`; `Bin(Or, …)` → `||` when both
//!     sides are boolean-shaped and pure (`&`/`|` on bools reads as bitwise).
//!
//! Constant-condition statements:
//!   * `if (true) A else B` → `A` (and the mirror), `while (false) B` → the
//!     loop disappears — but ONLY when the dropped arm defines no labels, so
//!     a `goto` target can never lose its definition. (The renderer folds
//!     these too; doing it in the tree keeps dead-arm reads out of the temp
//!     inliner's read counts.)
//!
//! ## Soundness gates
//!
//! * **Impure operands are never dropped.** `x * 0 → 0` is only valid when
//!   evaluating `x` has no side effects (no calls, no throwing derefs).
//! * **`Not(Bool)` is never touched.** The structurer uses a negated literal
//!   as its arm-swap marker; folding it would invert the arm it protects
//!   (see `folded_bool` in `csharp.rs`).
//! * **Float hazards.** Reflexive folds and ordering negations are skipped:
//!   NaN breaks `x == x`, `!(a < b) ⇒ a >= b`, and identity drops are safe
//!   but reflexive folds are not. Only comparison-operand swaps and the
//!   range rules above are NaN-exact.
//! * **Shift counts** outside `0..64` are left alone: C# masks the count
//!   (`x << 32 == x` for `int`), p-code zeroes; computing it here would be
//!   a wrong fold for one of the two.
//! * **Division by zero** and `i64::MIN / -1` are never folded.

use super::super::csharp::{BinOp, CmpKind, Expr, Stmt};
use super::il2cpp_cleanup::expr_children_mut;

// The cast/pointer-spelling pass lives in its own file. It is declared here
// (rather than in `reconstruct/mod.rs`) so this file can be reviewed as one
// change; promoting it to a top-level pass needs only the module declaration
// and a `PASSES` entry.
#[path = "cast_elision.rs"]
mod cast_elision;

/// Run the pass over one method body. Returns the number of rewrites.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut total = 0;
    // ILR_NOCAST=<anything> disables the cast/pointer-spelling companion pass,
    // for A/B measurement of its effect (same pattern as ILR_NOALG).
    let run_cast = std::env::var("ILR_NOCAST").is_err();
    // Fixpoint: a folded `If(true)` splices an arm whose statements then fold
    // again, an identity rewrite can expose a constant comparison, and a cast
    // removal can expose an identity (`(int)0 + x`).
    for _ in 0..8 {
        let mut n = if run_cast {
            cast_elision::apply(stmts)
        } else {
            0
        };
        super::walk_mut(stmts, &mut |list| {
            n += simplify_list(list);
        });
        total += n;
        if n == 0 {
            break;
        }
    }
    total
}

fn simplify_list(list: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < list.len() {
        n += simplify_stmt_exprs(&mut list[i]);
        // Constant-condition fold. Only when the DEAD arm defines no labels:
        // a label is a goto target, and dropping its definition would leave
        // `goto L<missing>` behind (the renderer's `labels_only` keeps them
        // for exactly this reason).
        let splice: Option<Vec<Stmt>> = match &mut list[i] {
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => match cond {
                Expr::Bool(true) if !has_label(else_body) => Some(std::mem::take(then_body)),
                Expr::Bool(false) if !has_label(then_body) => Some(std::mem::take(else_body)),
                _ => None,
            },
            Stmt::While { cond, body } => match cond {
                Expr::Bool(false) if !has_label(body) => Some(Vec::new()),
                _ => None,
            },
            _ => None,
        };
        if let Some(new) = splice {
            list.splice(i..=i, new);
            n += 1;
            // Re-examine the spliced statements at position i.
            continue;
        }
        i += 1;
    }
    n
}

fn has_label(body: &[Stmt]) -> bool {
    body.iter().any(|s| {
        if matches!(s, Stmt::Label(_)) {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        kids.iter().any(|k| has_label(k))
    })
}

/// Simplify every expression owned by `stmt` (recursing into nested bodies).
fn simplify_stmt_exprs(stmt: &mut Stmt) -> usize {
    let mut n = 0;
    match stmt {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) | Stmt::Assign(_, _, e) => {
            n += simplify_expr(e);
        }
        Stmt::Return(Some(e)) => n += simplify_expr(e),
        Stmt::EventAssign { field, value, .. } => {
            n += simplify_expr(field);
            n += simplify_expr(value);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            n += simplify_expr(collection);
            n += simplify_expr(index);
            n += simplify_expr(value);
        }
        Stmt::If { cond, .. }
        | Stmt::While { cond, .. }
        | Stmt::DoWhile { cond, .. } => {
            n += simplify_expr(cond);
        }
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                n += simplify_expr(c);
            }
        }
        Stmt::Return(None) | Stmt::YieldBreak | Stmt::Goto(_) | Stmt::Label(_) => {}
        Stmt::Lock { expr, .. } => n += simplify_expr(expr),
        Stmt::Switch { disc, .. } => {
            // `disc` is a bare name, not an expression to rewrite.
            let _ = disc;
        }
        Stmt::StringSwitch { disc, .. } => n += simplify_expr(disc),
        _ => {}
    }
    // Nested bodies (If/While/For/ForEach/Switch/Try/lambda-free wrappers).
    if matches!(
        stmt,
        Stmt::If { .. }
            | Stmt::While { .. }
            | Stmt::DoWhile { .. }
            | Stmt::For { .. }
            | Stmt::ForEach { .. }
            | Stmt::Switch { .. }
            | Stmt::StringSwitch { .. }
            | Stmt::Try { .. }
            | Stmt::Lock { .. }
            | Stmt::Using { .. }
    ) {
        for c in super::children_mut(stmt) {
            n += simplify_list(c);
        }
    }
    n
}

/// Post-order expression simplification, iterated at each node so a rewrite
/// that exposes another (e.g. `Eq(Bool, Bool)` after a term swap) converges.
fn simplify_expr(e: &mut Expr) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += simplify_expr(c);
    }
    // A handful of rounds is plenty: each round either rewrites the root or
    // stops, and rules only shrink the tree.
    for _ in 0..4 {
        let (new, changed) = rewrite_root(e);
        if !changed {
            break;
        }
        *e = new;
        n += 1;
    }
    n
}

/// Rewrite the root node only. Returns `(replacement, changed)`.
fn rewrite_root(e: &Expr) -> (Expr, bool) {
    match e {
        Expr::Bin(op, a, b) => rewrite_bin(*op, a, b),
        Expr::Cmp(k, a, b) => rewrite_cmp(*k, a, b),
        Expr::Not(inner) => rewrite_not(inner),
        Expr::Neg(inner) => {
            if let Expr::Int(v) = **inner {
                (Expr::Int(v.wrapping_neg()), true)
            } else {
                (e.clone(), false)
            }
        }
        _ => (e.clone(), false),
    }
}

fn rewrite_not(inner: &Expr) -> (Expr, bool) {
    match inner {
        // `!!x` → `x`.
        Expr::Not(x) => ((**x).clone(), true),
        // Equality negations are exact for every value, NaN included.
        Expr::Cmp(CmpKind::Eq, a, b) => (Expr::Cmp(CmpKind::Ne, a.clone(), b.clone()), true),
        Expr::Cmp(CmpKind::Ne, a, b) => (Expr::Cmp(CmpKind::Eq, a.clone(), b.clone()), true),
        // ORDERING negation is not: for NaN, `!(a<b)` is true while `a>=b`
        // is false. Left as `!(…)`.
        // `Not(Bool)` is the structurer's arm-swap marker — untouched.
        _ => (Expr::Not(Box::new(inner.clone())), false),
    }
}

fn swap_cmp(k: CmpKind) -> CmpKind {
    match k {
        CmpKind::Lt => CmpKind::Gt,
        CmpKind::Gt => CmpKind::Lt,
        CmpKind::Le => CmpKind::Ge,
        CmpKind::Ge => CmpKind::Le,
        CmpKind::Eq => CmpKind::Eq,
        CmpKind::Ne => CmpKind::Ne,
    }
}

fn is_boolish(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Cmp(..) | Expr::Not(_) | Expr::Bool(_) | Expr::PatternMatch { .. }
    )
}

/// Is this expression provably NOT a float/double value? Only these operands
/// may take part in reflexive folds (`x == x` is false for NaN).
fn definitely_non_float(e: &Expr) -> bool {
    match e {
        Expr::Int(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Null | Expr::This => true,
        Expr::Cmp(..) | Expr::Not(_) | Expr::PatternMatch { .. } => true,
        Expr::Cast(t, inner) => {
            let tl = t.to_ascii_lowercase();
            !(tl.contains("float") || tl.contains("double")) && definitely_non_float(inner)
        }
        Expr::Neg(x) => definitely_non_float(x),
        Expr::Bin(_, a, b) => definitely_non_float(a) && definitely_non_float(b),
        _ => false,
    }
}

/// No calls, no allocations, no memory reads that can fault. Dropping such an
/// expression cannot change observable behavior.
/// Can this expression be re-read once instead of twice without changing
/// observable behavior? Calls/allocations cannot; a variable, field load or
/// raw memory read is the SAME SSA value rendered twice (the renderer inlines
/// uses), so collapsing `x + (x << k)` to `x * 3` evaluates it once — exactly
/// what the native code did. This is deliberately weaker than [`pure`], which
/// guards DROPPING an operand (a fault must not disappear).
fn repeatable(e: &Expr) -> bool {
    match e {
        Expr::Call { .. }
        | Expr::RawCall(..)
        | Expr::NewObj(..)
        | Expr::NewArr(..)
        | Expr::CollectionInit { .. }
        | Expr::ObjInit { .. }
        | Expr::Await(_)
        | Expr::Lambda { .. }
        | Expr::MethodChain { .. }
        | Expr::StringInterp { .. }
        | Expr::SwitchExpr { .. } => false,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => repeatable(a) && repeatable(b),
        Expr::Cond(c, a, b) => repeatable(c) && repeatable(a) && repeatable(b),
        Expr::Not(x)
        | Expr::Neg(x)
        | Expr::Cast(_, x)
        | Expr::AddressOf(x)
        | Expr::RawAddr(x, _)
        | Expr::RawDeref(x, _, _)
        | Expr::Field(x, _)
        | Expr::Index { collection: x, .. } => repeatable(x),
        Expr::NullConditional { receiver, access } => {
            repeatable(receiver) && repeatable(access)
        }
        _ => true,
    }
}

fn pure(e: &Expr) -> bool {
    match e {
        Expr::Call { .. }
        | Expr::RawCall(..)
        | Expr::NewObj(..)
        | Expr::NewArr(..)
        | Expr::CollectionInit { .. }
        | Expr::ObjInit { .. }
        | Expr::Index { .. }
        | Expr::Await(_) => false,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => pure(a) && pure(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::RawAddr(x, _) => {
            pure(x)
        }
        Expr::Field(r, _) => pure(r),
        // A raw load can fault on a bad pointer; dropping it is observable.
        Expr::RawDeref(..) => false,
        Expr::Cond(c, a, b) => pure(c) && pure(a) && pure(b),
        Expr::NullConditional { receiver, access } => pure(receiver) && pure(access),
        Expr::Lambda { .. } => true,
        Expr::StringInterp { parts } => parts.iter().all(|p| match p {
            super::super::csharp::InterpPart::Literal(_) => true,
            super::super::csharp::InterpPart::Expr(x, _) => pure(x),
        }),
        Expr::Tuple { elements, .. } => elements.iter().all(pure),
        Expr::ArrayInit { items, .. } => items.iter().all(pure),
        Expr::MethodChain { .. } => false,
        Expr::SwitchExpr { discrim, arms, default_arm } => {
            pure(discrim)
                && arms.iter().all(|(p, r)| pure(p) && pure(r))
                && default_arm.as_ref().map_or(true, |d| pure(d))
        }
        _ => true,
    }
}

fn is_int(e: &Expr, v: i64) -> bool {
    matches!(e, Expr::Int(x) if *x == v)
}

fn is_u32_mask(e: &Expr) -> bool {
    is_int(e, 0xFFFF_FFFF)
}

fn rewrite_bin(op: BinOp, a: &Expr, b: &Expr) -> (Expr, bool) {
    // ---- constant folding ------------------------------------------------
    if let (Expr::Int(x), Expr::Int(y)) = (a, b) {
        let folded = match op {
            BinOp::Add => Some(x.wrapping_add(*y)),
            BinOp::Sub => Some(x.wrapping_sub(*y)),
            BinOp::Mul => Some(x.wrapping_mul(*y)),
            BinOp::Div if *y != 0 && !(*x == i64::MIN && *y == -1) => Some(x / y),
            BinOp::Rem if *y != 0 && !(*x == i64::MIN && *y == -1) => Some(x % y),
            BinOp::And => Some(x & y),
            BinOp::Or => Some(x | y),
            BinOp::Xor => Some(x ^ y),
            // C# masks the shift count; only fold counts Rust can evaluate
            // and C# agrees on.
            BinOp::Shl if (0..64).contains(y) => Some(x.wrapping_shl(*y as u32)),
            BinOp::Shr if (0..64).contains(y) => Some(x.wrapping_shr(*y as u32)),
            _ => None,
        };
        if let Some(v) = folded {
            return (Expr::Int(v), true);
        }
    }
    if let (Expr::Bool(x), Expr::Bool(y)) = (a, b) {
        let folded = match op {
            BinOp::AndAlso => Some(*x && *y),
            BinOp::OrElse => Some(*x || *y),
            _ => None,
        };
        if let Some(v) = folded {
            return (Expr::Bool(v), true);
        }
    }

    // ---- identities ------------------------------------------------------
    // Dropping an operand is only legal when evaluating it has no effects.
    let (a_pure, b_pure) = (pure(a), pure(b));
    let bin = |a: Expr, b: Expr| Expr::Bin(op, Box::new(a), Box::new(b));
    let rw = |e: Expr| (e, true);

    match op {
        BinOp::Add => {
            // `x + 0` keeps `x`; only the constant disappears, so `x` may be
            // impure (`*(nint*)(p) + 0` is still a load, just spelled shorter).
            if is_int(b, 0) {
                return rw(a.clone());
            }
            if is_int(a, 0) {
                return rw(b.clone());
            }
            // Strength reduction: `x + (x << k)` is `x * (2^k + 1)` in
            // wrapping two's-complement arithmetic. The source compiler emits
            // this for `i * 3` / `i * 5` array strides; the structurer then
            // repeats the whole index expression in every term, so the dump
            // showed `stack[i] + (stack[i] << 1)` for a single `i * 3`.
            // `x` must be repeatable: the rewrite evaluates it once, which is
            // what the native code did (the duplicate was render-time inlining).
            if repeatable(a) {
                if let Expr::Bin(BinOp::Shl, inner, sh) = b {
                    if inner.as_ref() == a {
                        if let Expr::Int(k) = sh.as_ref() {
                            if (1..=30).contains(k) {
                                return rw(Expr::Bin(BinOp::Mul, Box::new(a.clone()), Box::new(Expr::Int((1i64 << k) + 1))));
                            }
                        }
                    }
                }
                if let Expr::Bin(BinOp::Shl, inner, sh) = a {
                    if inner.as_ref() == b {
                        if let Expr::Int(k) = sh.as_ref() {
                            if (1..=30).contains(k) {
                                return rw(Expr::Bin(BinOp::Mul, Box::new(b.clone()), Box::new(Expr::Int((1i64 << k) + 1))));
                            }
                        }
                    }
                }
            }
            if a == b && repeatable(a) {
                return rw(Expr::Bin(BinOp::Mul, Box::new(a.clone()), Box::new(Expr::Int(2))));
            }
            // Constant on the right, so every matcher sees one shape.
            if matches!(a, Expr::Int(_)) && !matches!(b, Expr::Int(_)) {
                return rw(bin(b.clone(), a.clone()));
            }
        }
        BinOp::Sub => {
            if is_int(b, 0) {
                return rw(a.clone());
            }
            if a == b && a_pure && definitely_non_float(a) {
                return rw(Expr::Int(0));
            }
            if repeatable(a) {
                if let Expr::Bin(BinOp::Shl, inner, sh) = b {
                    if inner.as_ref() == a {
                        if let Expr::Int(k) = sh.as_ref() {
                            if (1..=30).contains(k) {
                                return rw(Expr::Bin(BinOp::Mul, Box::new(a.clone()), Box::new(Expr::Int(1 - (1i64 << k)))));
                            }
                        }
                    }
                }
            }
        }
        BinOp::Mul => {
            if is_int(b, 1) {
                return rw(a.clone());
            }
            if is_int(a, 1) {
                return rw(b.clone());
            }
            if (is_int(b, 0) && a_pure) || (is_int(a, 0) && b_pure) {
                return rw(Expr::Int(0));
            }
            if matches!(a, Expr::Int(_)) && !matches!(b, Expr::Int(_)) {
                return rw(bin(b.clone(), a.clone()));
            }
        }
        BinOp::Div => {
            if is_int(b, 1) {
                return rw(a.clone());
            }
        }
        BinOp::And => {
            // `x & 0` is 0, not x — this must be decided BEFORE the neutral
            // element of the sibling bitwise operators. (The shared arm used
            // to return `x` for `x & 0`.)
            if (is_int(b, 0) && a_pure) || (is_int(a, 0) && b_pure) {
                return rw(Expr::Int(0));
            }
            if is_boolish(a) && is_boolish(b) && a_pure && b_pure {
                return rw(Expr::Bin(
                    BinOp::AndAlso,
                    Box::new(a.clone()),
                    Box::new(b.clone()),
                ));
            }
        }
        BinOp::Or => {
            // 0 is the neutral element for `|`; dropping it keeps every side
            // effect of the other operand.
            if is_int(b, 0) {
                return rw(a.clone());
            }
            if is_int(a, 0) {
                return rw(b.clone());
            }
            if is_boolish(a) && is_boolish(b) && a_pure && b_pure {
                return rw(Expr::Bin(
                    BinOp::OrElse,
                    Box::new(a.clone()),
                    Box::new(b.clone()),
                ));
            }
        }
        BinOp::Xor => {
            if is_int(b, 0) {
                return rw(a.clone());
            }
            if is_int(a, 0) {
                return rw(b.clone());
            }
            if a == b && a_pure && definitely_non_float(a) {
                return rw(Expr::Int(0));
            }
        }
        BinOp::Shl | BinOp::Shr => {
            if is_int(b, 0) {
                return rw(a.clone());
            }
            // ARM64 sign-extension of a 32-bit write: `(x << 32) >> 32` and
            // `((x & 0xFFFFFFFF) << 32) >> 32` both truncate x to its low 32
            // bits and sign-extend — exactly C#'s `(int)x`. The lifter models
            // every `w`-register write this way, and the structurer repeats
            // the index in every term, so a hot loop read as
            // `((i + j) & 4294967295L) << 32 >> 32` on nearly every line.
            // A `long` cast is not assumed: the cast renders only the width
            // the shift pair proved.
            if op == BinOp::Shr && is_int(b, 32) {
                if let Expr::Bin(BinOp::Shl, inner, sh) = a {
                    if is_int(sh, 32) {
                        let mut src = &**inner;
                        if let Expr::Bin(BinOp::And, x, y) = src {
                            if is_u32_mask(y) {
                                src = x;
                            } else if is_u32_mask(x) {
                                src = y;
                            }
                        }
                        return rw(Expr::Cast("int".into(), Box::new(src.clone())));
                    }
                }
            }
        }
        BinOp::AndAlso | BinOp::OrElse => {
            // Range combination (Ghidra RuleLessNotEqual / RuleLessEqual).
            // Exact for floats too: for NaN every comparison is false, and
            // both the pair and the collapsed form evaluate to false.
            if pure(a) && pure(b) {
                if let Some(combined) = range_rule(a, b, op == BinOp::AndAlso) {
                    return rw(combined);
                }
            }
            let absorbing = if op == BinOp::AndAlso {
                Expr::Bool(false)
            } else {
                Expr::Bool(true)
            };
            let neutral = if op == BinOp::AndAlso {
                Expr::Bool(true)
            } else {
                Expr::Bool(false)
            };
            if *b == neutral && a_pure {
                return rw(a.clone());
            }
            if *a == neutral && b_pure {
                return rw(b.clone());
            }
            if *b == absorbing && a_pure {
                return rw(absorbing);
            }
            if *a == absorbing && b_pure {
                return rw(absorbing);
            }
        }
        _ => {}
    }
    (e_clone_bin(op, a, b), false)
}

fn e_clone_bin(op: BinOp, a: &Expr, b: &Expr) -> Expr {
    Expr::Bin(op, Box::new(a.clone()), Box::new(b.clone()))
}

fn is_zero_ish(e: &Expr) -> bool {
    matches!(e, Expr::Int(0) | Expr::Bool(false))
}
fn is_one_ish(e: &Expr) -> bool {
    matches!(e, Expr::Int(1) | Expr::Bool(true))
}
fn neg_bool(e: &Expr) -> Expr {
    match e {
        Expr::Cmp(CmpKind::Eq, a, b) => Expr::Cmp(CmpKind::Ne, a.clone(), b.clone()),
        Expr::Cmp(CmpKind::Ne, a, b) => Expr::Cmp(CmpKind::Eq, a.clone(), b.clone()),
        Expr::Not(x) => (**x).clone(),
        _ => Expr::Not(Box::new(e.clone())),
    }
}

/// `(c ? 0 : x) == 0` is `c || x == 0`, `(c ? 1 : x) == 0` is `!c && x == 0`
/// (and the `!=`/zero-on-the-left/mirrored-constant variants). This is the
/// shape branch lowering leaves in `Celebrate()` etc. Evaluation counts are
/// preserved: `c` once, `x` only on the branch that took it.
fn cond_zero_cmp(k: CmpKind, a: &Expr, b: &Expr) -> Option<Expr> {
    if !matches!(k, CmpKind::Eq | CmpKind::Ne) {
        return None;
    }
    let (cond, t, f) = match (a, b) {
        (Expr::Cond(c, t, f), other) if is_zero_ish(other) => ((**c).clone(), t, f),
        (other, Expr::Cond(c, t, f)) if is_zero_ish(other) => ((**c).clone(), t, f),
        _ => return None,
    };
    // Normalize so the CONSTANT branch is the true arm and `x` the false arm.
    let (cond, x, constant) = if is_zero_ish(t) && !is_zero_ish(f) {
        (cond, f, 0)
    } else if is_zero_ish(f) && !is_zero_ish(t) {
        (neg_bool(&cond), t, 0)
    } else if is_one_ish(t) && !is_one_ish(f) {
        (cond, f, 1)
    } else if is_one_ish(f) && !is_one_ish(t) {
        (neg_bool(&cond), t, 1)
    } else {
        return None;
    };
    let cmp = if k == CmpKind::Eq {
        Expr::Cmp(CmpKind::Eq, x.clone(), Box::new(Expr::Int(0)))
    } else {
        Expr::Cmp(CmpKind::Ne, x.clone(), Box::new(Expr::Int(0)))
    };
    match (constant, k) {
        (0, CmpKind::Eq) => Some(Expr::Bin(BinOp::OrElse, Box::new(cond), Box::new(cmp))),
        (0, CmpKind::Ne) => Some(Expr::Bin(
            BinOp::AndAlso,
            Box::new(neg_bool(&cond)),
            Box::new(cmp),
        )),
        (1, CmpKind::Eq) => Some(Expr::Bin(
            BinOp::AndAlso,
            Box::new(neg_bool(&cond)),
            Box::new(cmp),
        )),
        (1, CmpKind::Ne) => Some(Expr::Bin(BinOp::OrElse, Box::new(cond), Box::new(cmp))),
        _ => None,
    }
}

fn rewrite_cmp(k: CmpKind, a: &Expr, b: &Expr) -> (Expr, bool) {
    if let Some(e) = cond_zero_cmp(k, a, b) {
        return (e, true);
    }
    // Sign-test sugar: (x & 0x8000_0000) != 0 is exactly (int)x < 0, and
    // == 0 is (int)x >= 0. ARM64 tbnz/tbz on bit 31 lifts to this mask form;
    // the identity holds for every value (truncate to 32 bits, then test the
    // sign bit), so it is a rewrite, not a guess. The 64-bit mask is the same
    // test without truncation. Every OTHER mask is a real bit probe and stays
    // as (x & 1 << bit).
    if let Expr::Bin(BinOp::And, inner, mask) = a {
        if is_int(b, 0) {
            if let Expr::Int(m) = mask.as_ref() {
                let wide = *m == i64::MIN;
                let narrow = *m == 0x8000_0000;
                if (wide || narrow) && matches!(k, CmpKind::Eq | CmpKind::Ne) {
                    let base = if narrow {
                        Expr::Cast("int".into(), Box::new(inner.as_ref().clone()))
                    } else {
                        inner.as_ref().clone()
                    };
                    let kind = if matches!(k, CmpKind::Ne) {
                        CmpKind::Lt
                    } else {
                        CmpKind::Ge
                    };
                    return (Expr::Cmp(kind, Box::new(base), Box::new(Expr::Int(0))), true);
                }
            }
        }
    }
    // Constant folding.
    if let (Expr::Int(x), Expr::Int(y)) = (a, b) {
        let v = match k {
            CmpKind::Eq => x == y,
            CmpKind::Ne => x != y,
            CmpKind::Lt => x < y,
            CmpKind::Gt => x > y,
            CmpKind::Le => x <= y,
            CmpKind::Ge => x >= y,
        };
        return (Expr::Bool(v), true);
    }

    // Reflexive comparisons, non-float operands only (`x == x` is false for a
    // NaN float, and a Var's type is not known here).
    if a == b && pure(a) && definitely_non_float(a) {
        let v = matches!(k, CmpKind::Eq | CmpKind::Le | CmpKind::Ge);
        return (Expr::Bool(v), true);
    }

    // Constant on the right (Ghidra RuleTermOrder). Comparison-operand swap
    // is exact for every value including NaN.
    if matches!(a, Expr::Int(_)) && !matches!(b, Expr::Int(_)) {
        return (
            Expr::Cmp(swap_cmp(k), Box::new(b.clone()), Box::new(a.clone())),
            true,
        );
    }

    // Boolean operand normalization: `e == true` → `e`, `e != false` → `e`.
    if is_boolish(a) && pure(a) {
        match (k, b) {
            (CmpKind::Eq, Expr::Bool(true)) => return (a.clone(), true),
            (CmpKind::Ne, Expr::Bool(false)) => return (a.clone(), true),
            _ => {}
        }
    }
    if is_boolish(b) && pure(b) {
        match (k, a) {
            (CmpKind::Eq, Expr::Bool(true)) => return (b.clone(), true),
            (CmpKind::Ne, Expr::Bool(false)) => return (b.clone(), true),
            _ => {}
        }
    }

    (Expr::Cmp(k, Box::new(a.clone()), Box::new(b.clone())), false)
}

/// Range combination over two comparisons with the SAME operand pair.
///
/// `(a <= b) && (a != b)` → `a < b`; `(a >= b) && (a != b)` → `a > b`;
/// `(a < b) || (a == b)` → `a <= b`; `(a > b) || (a == b)` → `a >= b`.
/// Exact for floats: for NaN every comparison is false, and both sides of
/// each equivalence evaluate to false.
fn range_rule(a: &Expr, b: &Expr, _and: bool) -> Option<Expr> {
    let (Expr::Cmp(k1, x1, y1), Expr::Cmp(k2, x2, y2)) = (a, b) else {
        return None;
    };
    if x1 != x2 || y1 != y2 {
        return None;
    }
    match (k1, k2) {
        (CmpKind::Le, CmpKind::Ne) | (CmpKind::Ne, CmpKind::Le) => {
            Some(Expr::Cmp(CmpKind::Lt, x1.clone(), y1.clone()))
        }
        (CmpKind::Ge, CmpKind::Ne) | (CmpKind::Ne, CmpKind::Ge) => {
            Some(Expr::Cmp(CmpKind::Gt, x1.clone(), y1.clone()))
        }
        (CmpKind::Lt, CmpKind::Eq) | (CmpKind::Eq, CmpKind::Lt) => {
            Some(Expr::Cmp(CmpKind::Le, x1.clone(), y1.clone()))
        }
        (CmpKind::Gt, CmpKind::Eq) | (CmpKind::Eq, CmpKind::Gt) => {
            Some(Expr::Cmp(CmpKind::Ge, x1.clone(), y1.clone()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: &str) -> Expr {
        Expr::Var(n.into())
    }

    fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
        Expr::Bin(op, Box::new(a), Box::new(b))
    }

    fn cmp(k: CmpKind, a: Expr, b: Expr) -> Expr {
        Expr::Cmp(k, Box::new(a), Box::new(b))
    }

    fn simplify(e: Expr) -> Expr {
        let mut stmts = vec![Stmt::Eval(e)];
        apply(&mut stmts);
        match &mut stmts[0] {
            Stmt::Eval(x) => x.clone(),
            other => panic!("unexpected stmt: {other:?}"),
        }
    }

    #[test]
    fn negated_equality_flips_the_operator() {
        let e = Expr::Not(Box::new(cmp(CmpKind::Eq, v("a"), v("b"))));
        assert_eq!(simplify(e), cmp(CmpKind::Ne, v("a"), v("b")));
    }

    #[test]
    fn negated_ordering_is_left_alone_for_nan_safety() {
        let e = Expr::Not(Box::new(cmp(CmpKind::Lt, v("a"), v("b"))));
        let out = simplify(e);
        assert!(
            matches!(&out, Expr::Not(inner) if matches!(**inner, Expr::Cmp(CmpKind::Lt, ..))),
            "ordering negation must stay: {out:?}"
        );
    }

    #[test]
    fn double_negation_collapses() {
        let e = Expr::Not(Box::new(Expr::Not(Box::new(v("flag")))));
        assert_eq!(simplify(e), v("flag"));
    }

    #[test]
    fn not_of_literal_is_the_structurer_marker_and_must_survive() {
        let e = Expr::Not(Box::new(Expr::Bool(true)));
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn constant_comparison_folds_to_bool() {
        let e = cmp(CmpKind::Ne, Expr::Int(0), Expr::Int(0));
        assert_eq!(simplify(e), Expr::Bool(false));
    }

    #[test]
    fn identities_apply_when_operands_are_pure() {
        assert_eq!(simplify(bin(BinOp::Add, v("x"), Expr::Int(0))), v("x"));
        assert_eq!(simplify(bin(BinOp::Mul, v("x"), Expr::Int(1))), v("x"));
        assert_eq!(simplify(bin(BinOp::Sub, v("x"), Expr::Int(0))), v("x"));
        assert_eq!(simplify(bin(BinOp::Xor, v("x"), Expr::Int(0))), v("x"));
        assert_eq!(simplify(bin(BinOp::Mul, v("x"), Expr::Int(0))), Expr::Int(0));
    }

    #[test]
    fn ternary_zero_compare_becomes_boolean_or() {
        let cond = cmp(CmpKind::Eq, v("c"), Expr::Int(0));
        let e = cmp(
            CmpKind::Eq,
            Expr::Cond(Box::new(cond.clone()), Box::new(Expr::Int(0)), Box::new(v("x"))),
            Expr::Int(0),
        );
        let out = simplify(e);
        let t = format!("{out:?}");
        assert!(t.contains("OrElse"), "{t}");
        assert!(t.contains("\"c\""), "{t}");
    }

    #[test]
    fn x_plus_x_shift_becomes_multiply() {
        let e = bin(
            BinOp::Add,
            v("i"),
            bin(BinOp::Shl, v("i"), Expr::Int(1)),
        );
        assert_eq!(simplify(e), bin(BinOp::Mul, v("i"), Expr::Int(3)));
        let e = bin(BinOp::Sub, v("i"), bin(BinOp::Shl, v("i"), Expr::Int(1)));
        assert_eq!(simplify(e), bin(BinOp::Mul, v("i"), Expr::Int(-1)));
        assert_eq!(
            simplify(bin(BinOp::Add, v("i"), v("i"))),
            bin(BinOp::Mul, v("i"), Expr::Int(2))
        );
    }


    /// A tbnz/tbz on bit 31/63 is a SIGN test, not a bit probe. Rendering it
    /// as (x & 2147483648) == 0 is exact but unreadable; the identity
    /// (x & 0x80000000) == 0  <=>  (int)x >= 0 holds for every value.
    #[test]
    fn sign_bit_mask_becomes_a_sign_comparison() {
        let m = |mask: i64, k: CmpKind| {
            Expr::Cmp(
                k,
                Box::new(bin(BinOp::And, v("x"), Expr::Int(mask))),
                Box::new(Expr::Int(0)),
            )
        };
        assert_eq!(
            simplify(m(0x8000_0000, CmpKind::Ne)),
            Expr::Cmp(
                CmpKind::Lt,
                Box::new(Expr::Cast("int".into(), Box::new(v("x")))),
                Box::new(Expr::Int(0))
            )
        );
        assert_eq!(
            simplify(m(0x8000_0000, CmpKind::Eq)),
            Expr::Cmp(
                CmpKind::Ge,
                Box::new(Expr::Cast("int".into(), Box::new(v("x")))),
                Box::new(Expr::Int(0))
            )
        );
        assert_eq!(
            simplify(m(i64::MIN, CmpKind::Ne)),
            Expr::Cmp(CmpKind::Lt, Box::new(v("x")), Box::new(Expr::Int(0)))
        );
        // A real bit probe is untouched: bit 1 is not the sign bit.
        assert_eq!(
            simplify(m(2, CmpKind::Ne)),
            m(2, CmpKind::Ne),
            "only the sign bit folds"
        );
    }

    #[test]
    fn w_register_sign_extension_folds_to_an_int_cast() {
        let inner = bin(BinOp::Add, v("i"), v("j"));
        let e = bin(
            BinOp::Shr,
            bin(BinOp::Shl, inner.clone(), Expr::Int(32)),
            Expr::Int(32),
        );
        assert_eq!(
            simplify(e),
            Expr::Cast("int".into(), Box::new(inner.clone()))
        );
        let masked = bin(BinOp::And, inner.clone(), Expr::Int(0xFFFF_FFFF));
        let e = bin(
            BinOp::Shr,
            bin(BinOp::Shl, masked, Expr::Int(32)),
            Expr::Int(32),
        );
        assert_eq!(simplify(e), Expr::Cast("int".into(), Box::new(inner)));
    }

    #[test]
    fn other_shift_widths_keep_the_sign_extension() {
        let e = bin(
            BinOp::Shr,
            bin(BinOp::Shl, v("x"), Expr::Int(16)),
            Expr::Int(16),
        );
        let out = simplify(e.clone());
        assert_eq!(out, e, "only the 32-bit width is proven here");
    }

    #[test]
    fn identity_never_drops_an_impure_operand() {
        let call = Expr::Call {
            recv: None,
            name: "Foo".into(),
            args: vec![],
        };
        let e = bin(BinOp::Mul, call.clone(), Expr::Int(0));
        let out = simplify(e.clone());
        assert_eq!(out, e, "a call must not be dropped by x * 0");
    }

    #[test]
    fn one_sided_identities_keep_an_impure_operand() {
        // The load stays; only the constant goes — so purity is not required.
        let load = Expr::RawDeref(Box::new(v("p")), 0, 8);
        assert_eq!(simplify(bin(BinOp::Add, load.clone(), Expr::Int(0))), load);
        assert_eq!(simplify(bin(BinOp::Add, Expr::Int(0), load.clone())), load);
        assert_eq!(simplify(bin(BinOp::Sub, load.clone(), Expr::Int(0))), load);
        assert_eq!(simplify(bin(BinOp::Mul, load.clone(), Expr::Int(1))), load);
        assert_eq!(simplify(bin(BinOp::Mul, Expr::Int(1), load.clone())), load);
        assert_eq!(simplify(bin(BinOp::Shl, load.clone(), Expr::Int(0))), load);
        assert_eq!(simplify(bin(BinOp::Shr, load.clone(), Expr::Int(0))), load);
        assert_eq!(simplify(bin(BinOp::Or, load.clone(), Expr::Int(0))), load);
        assert_eq!(simplify(bin(BinOp::Xor, load.clone(), Expr::Int(0))), load);
    }

    #[test]
    fn and_with_zero_is_zero_not_the_other_operand() {
        assert_eq!(simplify(bin(BinOp::And, v("x"), Expr::Int(0))), Expr::Int(0));
        assert_eq!(simplify(bin(BinOp::And, Expr::Int(0), v("x"))), Expr::Int(0));
        // An impure operand must be kept, so the fold does not fire.
        let call = Expr::Call {
            recv: None,
            name: "Foo".into(),
            args: vec![],
        };
        let e = bin(BinOp::And, call, Expr::Int(0));
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn reflexive_comparison_folds_only_for_non_float_operands() {
        assert_eq!(
            simplify(cmp(CmpKind::Eq, Expr::Int(7), Expr::Int(7))),
            Expr::Bool(true)
        );
        // A Var could be a float; do not fold.
        let e = cmp(CmpKind::Eq, v("x"), v("x"));
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn constant_moves_to_the_right_of_a_comparison() {
        let e = cmp(CmpKind::Eq, Expr::Int(0), v("x"));
        assert_eq!(simplify(e), cmp(CmpKind::Eq, v("x"), Expr::Int(0)));
        let e = cmp(CmpKind::Lt, Expr::Int(5), v("x"));
        assert_eq!(simplify(e), cmp(CmpKind::Gt, v("x"), Expr::Int(5)));
    }

    #[test]
    fn fused_boolean_ops_become_short_circuit() {
        let e = bin(
            BinOp::And,
            cmp(CmpKind::Eq, v("a"), Expr::Int(0)),
            cmp(CmpKind::Eq, v("b"), Expr::Int(0)),
        );
        let out = simplify(e);
        assert!(
            matches!(out, Expr::Bin(BinOp::AndAlso, ..)),
            "expected &&, got {out:?}"
        );
    }

    #[test]
    fn constant_if_splices_the_live_arm() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Return(Some(Expr::Int(1)))],
            else_body: vec![Stmt::Return(Some(Expr::Int(2)))],
        }];
        apply(&mut stmts);
        assert_eq!(stmts.len(), 1);
        assert!(matches!(&stmts[0], Stmt::Return(Some(Expr::Int(1)))));
    }

    #[test]
    fn constant_if_keeps_a_labelled_dead_arm() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Return(Some(Expr::Int(1)))],
            else_body: vec![Stmt::Label("L9".into())],
        }];
        apply(&mut stmts);
        assert!(
            matches!(&stmts[0], Stmt::If { .. }),
            "a dead arm that defines a label must not be dropped"
        );
    }
}
