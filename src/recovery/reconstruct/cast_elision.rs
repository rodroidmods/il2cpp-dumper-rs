//! Cast elision and pointer-spelling cleanup on the finished statement tree.
//!
//! The lifter materialises every width change it cannot prove away and every
//! address computation as its raw shape. That is the honest form, but three
//! families of it are pure spelling, and they dominate the remaining
//! "this is obviously C#" noise in the dump:
//!
//! * **Redundant casts.** `(int)5`, `(nint)(nint)x`, `(int)((int)x)`,
//!   `(float)1.5f`, `(string)"s"`, and casts whose operand already has the
//!   cast's type (`(int)*(int*)(p)`). Also exact widenings that the compiler
//!   would insert anyway: `(long)(int)x` → `(int)x`, `(double)(float)x` →
//!   `(float)x`.
//! * **Deref / address cancellation.** `*(nint*)(&v)` is just `v`: the
//!   address-of operator names a storage cell, and the deref immediately
//!   reads it back. Only value-storage bases (`Var`, `This`, `Field` on a
//!   pure receiver) qualify; a call result has no stable cell, so its
//!   address cannot cancel. `AddressOf(RawDeref(..))` is the *other*
//!   direction and is never cancelled.
//! * **Address arithmetic spelling.** `RawDeref`/`RawAddr` carry their byte
//!   offset outside the base expression, so a base that is itself
//!   `x + k` renders as `x + k + off`. Folding `k` into `off`
//!   (`*(int*)(p + 8 + 16)` → `*(int*)(p + 24)`) changes only the spelling.
//!
//! ## Soundness gates
//!
//! * **No type guesses.** A deref of an address cancels only when the access
//!   width is compatible with the cell: pointer-width loads (`width` 0/8)
//!   from a base with no narrower declared type, or a width that exactly
//!   matches the local's recorded declaration (`int`/`uint` for width 4, …).
//!   An `int` local read through `nint*` is *not* the int value, so it is
//!   left alone.
//! * **Side-effecting bases are never cancelled.** `&M()` has no storage
//!   cell; a `Field` receiver must itself be pure (`this`, a local, or a
//!   field chain of those).
//! * **Cast removal is value-preserving.** Literals are dropped only when
//!   the literal already has the target type. Nested casts keep the inner
//!   cast. A widening layer is dropped only when it is exact for the inner
//!   type (`int`→`long`, `float`→`double`), not when the intermediate type
//!   loses information (`(double)(float)someInt` keeps the float rounding —
//!   only the outer cast is dropped, the float cast stays).
//! * **The 32-bit mask only goes where 32 bits are consumed.**
//!   `x & 0xFFFFFFFF` is deleted only under an `(int)`/`(uint)` cast or in an
//!   assignment to a declared 32-bit local. It is never deleted from a
//!   64-bit expression: `(v & 0xFFFFFFFFL) << 32 >> 32` is the
//!   zero-extension idiom, and dropping the mask would change its value.
//! * **Idempotent.** Every rewrite is local and shrinks the tree; a second
//!   application on the emitted form performs no rewrite.

use crate::recovery::csharp::{BinOp, Expr, Stmt};
use crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut;
use crate::recovery::reconstruct::{children, walk_mut};
use std::collections::HashMap;

/// Run the pass over one method body. Returns the number of rewrites.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    // Declared local types are the only width evidence in the AST: a typed
    // first assignment (`Stmt::Assign(name, Some("int"), …)`). They gate the
    // non-pointer-width deref cancellation.
    let decls = declared_types(stmts);
    let mut total = 0;
    // A cast removal can expose another cast (`(int)((long)5)`); three rounds
    // reach a fixed point for any expression the lifter produces.
    for _ in 0..3 {
        let mut n = 0;
        walk_mut(stmts, &mut |list| {
            for s in list.iter_mut() {
                n += simplify_stmt(s, &decls);
            }
        });
        total += n;
        if n == 0 {
            break;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Declared local types
// ---------------------------------------------------------------------------

fn declared_types(stmts: &[Stmt]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    collect_declared(stmts, &mut out);
    out
}

fn collect_declared(list: &[Stmt], out: &mut HashMap<String, String>) {
    for s in list {
        if let Stmt::Assign(name, Some(ty), _) = s {
            out.insert(name.clone(), ty.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for kid in kids {
            collect_declared(kid, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Statement / expression walk
// ---------------------------------------------------------------------------

fn simplify_stmt(stmt: &mut Stmt, decls: &HashMap<String, String>) -> usize {
    let mut n = 0;
    match stmt {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => n += simplify_expr(e, decls),
        Stmt::Assign(name, _, e) => {
            n += simplify_expr(e, decls);
            // `int x = 1; x = <expr & 0xFFFFFFFF>;` consumes 32 bits: the mask
            // is redundant. 64-bit destinations keep their mask.
            if decl_is_32_bit(decls.get(name.as_str()).map(String::as_str)) {
                n += strip_root_u32_mask(e, decls) as usize;
            }
        }
        Stmt::Return(Some(e)) => n += simplify_expr(e, decls),
        Stmt::EventAssign { field, value, .. } => {
            n += simplify_expr(field, decls);
            n += simplify_expr(value, decls);
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            n += simplify_expr(collection, decls);
            n += simplify_expr(index, decls);
            n += simplify_expr(value, decls);
        }
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            n += simplify_expr(cond, decls);
        }
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                n += simplify_expr(c, decls);
            }
        }
        Stmt::Return(None) | Stmt::YieldBreak | Stmt::Goto(_) | Stmt::Label(_) => {}
        Stmt::Lock { expr, .. } => n += simplify_expr(expr, decls),
        // `disc` is a bare name; `Stmt::Switch` owns no expressions.
        Stmt::Switch { disc, .. } => {
            let _ = disc;
        }
        Stmt::StringSwitch { disc, .. } => n += simplify_expr(disc, decls),
        _ => {}
    }
    n
}

/// Bottom-up expression rewrite, iterated at the root so a rule that exposes
/// another (`(int)((int)(int)x)`) converges in one visit.
fn simplify_expr(e: &mut Expr, decls: &HashMap<String, String>) -> usize {
    let mut n = 0;
    for c in expr_children_mut(e) {
        n += simplify_expr(c, decls);
    }
    for _ in 0..6 {
        let (new, changed) = rewrite_root(e, decls);
        if !changed {
            break;
        }
        *e = new;
        n += 1;
    }
    n
}

fn rewrite_root(e: &Expr, decls: &HashMap<String, String>) -> (Expr, bool) {
    match e {
        Expr::Cast(t, inner) => rewrite_cast(e, t, inner, decls),
        Expr::RawDeref(base, addend, width) => rewrite_deref(e, base, *addend, *width, decls),
        Expr::RawAddr(base, addend) => match fold_base_addend(base, *addend) {
            Some((base, addend)) => (Expr::RawAddr(Box::new(base), addend), true),
            None => (e.clone(), false),
        },
        _ => (e.clone(), false),
    }
}

// ---------------------------------------------------------------------------
// Casts
// ---------------------------------------------------------------------------

fn rewrite_cast(
    whole: &Expr,
    target: &str,
    inner: &Expr,
    decls: &HashMap<String, String>,
) -> (Expr, bool) {
    let ty = norm_type(target);

    // `(T)literal` where the literal already has type T.
    if literal_is_natural(ty, inner) {
        return (inner.clone(), true);
    }

    // `(int)*(int*)(p)`, `(nint)*(nint*)(p)`, `(bool)(a == b)` — the operand's
    // type is already the cast's target.
    if source_type(inner, decls).is_some_and(|src| norm_type(&src) == ty) {
        return (inner.clone(), true);
    }

    if let Expr::Cast(src, operand) = inner {
        let src = norm_type(src);
        // `(T)(T)x` → `(T)x` is covered by `source_type` above; this arm is
        // for an exact WIDENING layer: `(long)(int)x` → `(int)x` and
        // `(double)(float)x` → `(float)x`. The inner cast is what determines
        // the value; the outer only converts it to a wider type, which C#
        // does implicitly at the use site.
        if is_exact_widening(src, ty) {
            return (Expr::Cast(src.to_string(), operand.clone()), true);
        }
    }

    // `(int)(x & 0xFFFFFFFF)` → `(int)x`: truncating to 32 bits after masking
    // to 32 bits is the same truncation. Only under a 32-bit cast — in a
    // 64-bit expression the mask is the zero-extension idiom.
    if matches!(ty, "int" | "uint") {
        if let Expr::Bin(BinOp::And, a, b) = inner {
            if is_u32_mask(b) {
                return (
                    Expr::Cast(target.to_string(), Box::new((**a).clone())),
                    true,
                );
            }
            if is_u32_mask(a) {
                return (
                    Expr::Cast(target.to_string(), Box::new((**b).clone())),
                    true,
                );
            }
        }
    }

    (whole.clone(), false)
}

/// Does `e` already have the type named by `ty` on its own?
fn literal_is_natural(ty: &str, e: &Expr) -> bool {
    match e {
        // A decimal literal is `int` when it fits and `long` above that range
        // (the renderer appends the `L` suffix there); it is implicitly
        // convertible to a wider integral type at the use site. `nint` is
        // deliberately absent: dropping `(nint)5` from pointer arithmetic
        // would leave `5 + ptr`, which is not C#. `uint`/`ulong` are absent
        // too: `4294967295` renders with an `L` suffix, so it is a `long`,
        // and a long constant does not implicitly convert to `uint`.
        Expr::Int(v) => match ty {
            "int" => i32::try_from(*v).is_ok(),
            "long" => true,
            _ => false,
        },
        Expr::Float(_) => ty == "float",
        Expr::Double(_) => ty == "double",
        Expr::Bool(_) => ty == "bool",
        Expr::Str(_) => ty == "string",
        // `null` has no type of its own; it is implicitly convertible to a
        // reference type (and to Nullable<T> — so `int?` keeps its cast).
        Expr::Null => matches!(ty, "string" | "object"),
        _ => false,
    }
}

/// The C# type of `e` when it is evident from the expression alone. `None`
/// means "unknown — do not decide anything from it".
fn source_type(e: &Expr, decls: &HashMap<String, String>) -> Option<String> {
    match e {
        Expr::Int(v) => Some(if i32::try_from(*v).is_ok() {
            "int".to_string()
        } else {
            "long".to_string()
        }),
        Expr::Float(_) => Some("float".to_string()),
        Expr::Double(_) => Some("double".to_string()),
        Expr::Bool(_) | Expr::Cmp(..) | Expr::Not(_) | Expr::PatternMatch { .. } => {
            Some("bool".to_string())
        }
        Expr::Str(_) => Some("string".to_string()),
        Expr::Cast(t, _) => Some(norm_type(t).to_string()),
        Expr::RawAddr(..) => Some("nint*".to_string()),
        Expr::RawDeref(_, _, width) => Some(width_type(*width).to_string()),
        Expr::NewObj(t, _) | Expr::CollectionInit { type_name: t, .. } => {
            Some(norm_type(t).to_string())
        }
        Expr::ObjInit { type_name, .. } => Some(norm_type(type_name).to_string()),
        Expr::ArrayInit { elem_type, .. } => Some(format!("{}[]", norm_type(elem_type))),
        Expr::NewArr(elem, dims) if dims.len() == 1 => {
            Some(format!("{}[]", norm_type(elem)))
        }
        Expr::Neg(inner) => match source_type(inner, decls)?.as_str() {
            t @ ("int" | "long" | "float" | "double") => Some(t.to_string()),
            _ => None,
        },
        Expr::Var(name) => decls.get(name).map(|t| norm_type(t).to_string()),
        Expr::Bin(op, a, b) => match op {
            BinOp::AndAlso | BinOp::OrElse => Some("bool".to_string()),
            BinOp::Shl | BinOp::Shr => {
                let t = source_type(a, decls)?;
                matches!(t.as_str(), "int" | "long" | "uint" | "ulong" | "nint" | "nuint")
                    .then_some(t)
            }
            _ => {
                let (ta, tb) = (source_type(a, decls)?, source_type(b, decls)?);
                // C#'s numeric promotion for same-type integer operands keeps
                // that type (int+int is int, long+long is long).
                let t = if ta == tb {
                    ta
                } else if matches!(ta.as_str(), "int" | "uint" | "long" | "ulong")
                    && matches!(tb.as_str(), "int" | "uint" | "long" | "ulong")
                {
                    promote_int(&ta, &tb)
                } else if ta == "float" && tb == "double" || ta == "double" && tb == "float" {
                    "double".to_string()
                } else {
                    return None;
                };
                Some(t)
            }
        },
        // `cond ? a : b` has a type only when both arms agree (an int literal
        // against an `int` local is the same type; anything else is left).
        Expr::Cond(_, a, b) => {
            let (ta, tb) = (source_type(a, decls)?, source_type(b, decls)?);
            (ta == tb).then_some(ta)
        }
        _ => None,
    }
}

/// C# binary numeric promotion for two integral types.
fn promote_int(a: &str, b: &str) -> String {
    let rank = |t: &str| match t {
        "int" => 0,
        "uint" => 1,
        "long" => 2,
        "ulong" => 3,
        _ => 4,
    };
    if rank(a) >= rank(b) {
        a.to_string()
    } else {
        b.to_string()
    }
}

/// `RawDeref`'s access width → the C# type the renderer prints for it.
fn width_type(width: u8) -> &'static str {
    match width {
        1 => "byte",
        2 => "short",
        4 => "int",
        crate::recovery::reconstruct::float_locals::F32_WIDTH => "float",
        _ => "nint",
    }
}

/// Is dropping `outer` around a value already of type `inner` exact?
fn is_exact_widening(inner: &str, outer: &str) -> bool {
    matches!(
        (inner, outer),
        ("int", "long") | ("uint", "long") | ("uint", "ulong") | ("float", "double")
    )
}

// ---------------------------------------------------------------------------
// Deref / address cancellation
// ---------------------------------------------------------------------------

fn rewrite_deref(
    whole: &Expr,
    base: &Expr,
    addend: i64,
    width: u8,
    decls: &HashMap<String, String>,
) -> (Expr, bool) {
    // `*(T*)(&x)` reads exactly the storage `&x` names; for a width the cell
    // actually holds, the value is `x` itself.
    if addend == 0 {
        if let Expr::AddressOf(target) = base {
            if address_target_ok(target, width, decls) {
                return (target.as_ref().clone(), true);
            }
        }
    }
    // `*(T*)((p + 8) + 16)` ≡ `*(T*)(p + 24)`: byte offsets combine.
    if let Some((base, addend)) = fold_base_addend(base, addend) {
        return (
            Expr::RawDeref(Box::new(base), addend, width),
            true,
        );
    }
    (whole.clone(), false)
}

/// Is `target` a pure storage cell whose width matches the access?
fn address_target_ok(
    target: &Expr,
    width: u8,
    decls: &HashMap<String, String>,
) -> bool {
    let (cell, declared) = match target {
        Expr::Var(name) => (true, decls.get(name).map(|t| norm_type(t))),
        Expr::This => (true, None),
        // `&this.field` / `&obj.field`: the cell exists, but reading it
        // through a possibly side-effecting receiver would move work.
        Expr::Field(receiver, _) => (storage_pure(receiver), None),
        _ => (false, None),
    };
    if !cell {
        return false;
    }
    match width {
        // Pointer-width access (0 = not stated, 8 = nint): the cell holds a
        // pointer/reference unless a narrower type was declared for it.
        0 | 8 => !declared.is_some_and(|t| {
            matches!(
                t,
                "int" | "uint" | "short" | "ushort" | "byte" | "sbyte" | "bool" | "char"
                    | "float" | "double"
            )
        }),
        4 => matches!(declared, Some("int" | "uint")),
        2 => matches!(declared, Some("short" | "ushort" | "char")),
        1 => matches!(declared, Some("byte" | "sbyte" | "bool")),
        _ => false,
    }
}

/// A receiver that names storage and cannot run code: `this`, a local, or a
/// field chain rooted in one of those.
fn storage_pure(e: &Expr) -> bool {
    match e {
        Expr::Var(_) | Expr::This | Expr::Int(_) | Expr::Str(_) | Expr::Bool(_) | Expr::Null => {
            true
        }
        Expr::Field(recv, _) => storage_pure(recv),
        _ => false,
    }
}

/// Fold a constant `± k` already inside a raw-deref/address base into the
/// node's own byte offset: `(p + 8) + 16` → `p + 24`.
fn fold_base_addend(base: &Expr, addend: i64) -> Option<(Expr, i64)> {
    let Expr::Bin(op @ (BinOp::Add | BinOp::Sub), inner, k) = base else {
        return None;
    };
    let Expr::Int(kv) = &**k else {
        return None;
    };
    // Leave literal-vs-literal arithmetic to constant folding.
    if matches!(**inner, Expr::Int(_)) {
        return None;
    }
    let delta = if *op == BinOp::Add {
        *kv
    } else {
        kv.checked_neg()?
    };
    let folded = addend.checked_add(delta)?;
    Some(((**inner).clone(), folded))
}

// ---------------------------------------------------------------------------
// 32-bit mask
// ---------------------------------------------------------------------------

fn is_u32_mask(e: &Expr) -> bool {
    matches!(e, Expr::Int(v) if *v == 0xFFFF_FFFF)
}

/// Remove a root-level `& 0xFFFFFFFF` from a value assigned to a 32-bit
/// destination. The operand must be provably 32-bit; a 64-bit value keeps its
/// mask (the mask is what narrows it).
fn strip_root_u32_mask(e: &mut Expr, decls: &HashMap<String, String>) -> bool {
    let Expr::Bin(BinOp::And, a, b) = e else {
        return false;
    };
    let operand = if is_u32_mask(b) {
        a
    } else if is_u32_mask(a) {
        b
    } else {
        return false;
    };
    if !expr_is_32_bit(operand, decls) {
        return false;
    }
    let replacement = operand.as_ref().clone();
    *e = replacement;
    true
}

fn expr_is_32_bit(e: &Expr, decls: &HashMap<String, String>) -> bool {
    match e {
        // Only in-range decimals: a large literal is a 64-bit value.
        Expr::Int(v) => i32::try_from(*v).is_ok(),
        Expr::Var(name) => decl_is_32_bit(decls.get(name.as_str()).map(String::as_str)),
        Expr::Cast(t, _) => matches!(norm_type(t), "int" | "uint"),
        Expr::Neg(x) => expr_is_32_bit(x, decls),
        Expr::Bin(op, a, b) => match op {
            BinOp::Shl | BinOp::Shr => {
                expr_is_32_bit(a, decls) && matches!(**b, Expr::Int(_))
            }
            BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Rem
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor => expr_is_32_bit(a, decls) && expr_is_32_bit(b, decls),
            _ => false,
        },
        _ => false,
    }
}

fn decl_is_32_bit(ty: Option<&str>) -> bool {
    ty.is_some_and(|t| matches!(norm_type(t), "int" | "uint"))
}

/// Map metadata spelling onto the C# keyword the renderer emits, so casts
/// and declarations compare equal even when the names came from different
/// tables (`Int32` vs `int`).
fn norm_type(t: &str) -> &str {
    match t.trim() {
        "System.Int32" | "Int32" => "int",
        "System.UInt32" | "UInt32" => "uint",
        "System.Int64" | "Int64" => "long",
        "System.UInt64" | "UInt64" => "ulong",
        "System.Int16" | "Int16" => "short",
        "System.UInt16" | "UInt16" => "ushort",
        "System.Byte" | "Byte" => "byte",
        "System.SByte" | "SByte" => "sbyte",
        "System.Boolean" | "Boolean" => "bool",
        "System.Single" | "Single" => "float",
        "System.Double" | "Double" => "double",
        "System.Char" | "Char" => "char",
        "System.String" | "String" => "string",
        "System.Object" | "Object" => "object",
        "System.IntPtr" | "IntPtr" => "nint",
        "System.UIntPtr" | "UIntPtr" => "nuint",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: &str) -> Expr {
        Expr::Var(n.into())
    }

    fn cast(t: &str, e: Expr) -> Expr {
        Expr::Cast(t.into(), Box::new(e))
    }

    fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
        Expr::Bin(op, Box::new(a), Box::new(b))
    }

    fn deref(base: Expr, off: i64, width: u8) -> Expr {
        Expr::RawDeref(Box::new(base), off, width)
    }

    fn addr(e: Expr) -> Expr {
        Expr::AddressOf(Box::new(e))
    }

    /// Run on `e`, returning the rewritten expression.
    fn simplify(e: Expr) -> Expr {
        let mut stmts = vec![Stmt::Eval(e)];
        apply(&mut stmts);
        match &mut stmts[0] {
            Stmt::Eval(x) => x.clone(),
            other => panic!("unexpected stmt: {other:?}"),
        }
    }

    /// Run on `stmts`, returning (rewrites, statements).
    fn simplify_stmts(mut stmts: Vec<Stmt>) -> (usize, Vec<Stmt>) {
        let n = apply(&mut stmts);
        (n, stmts)
    }

    fn assign(name: &str, ty: Option<&str>, e: Expr) -> Stmt {
        Stmt::Assign(name.into(), ty.map(str::to_string), e)
    }

    // ---- literal casts ---------------------------------------------------

    #[test]
    fn integer_literal_cast_is_dropped() {
        assert_eq!(simplify(cast("int", Expr::Int(5))), Expr::Int(5));
        assert_eq!(simplify(cast("long", Expr::Int(5))), Expr::Int(5));
        assert_eq!(simplify(cast("long", Expr::Int(5_000_000_000))), Expr::Int(5_000_000_000));
    }

    #[test]
    fn a_literal_out_of_int_range_keeps_its_narrowing_cast() {
        let e = cast("int", Expr::Int(0x1_0000_0000));
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn pointer_and_small_unsigned_casts_are_not_literal_natural_types() {
        // `(nint)5` in pointer arithmetic would become `5 + ptr`.
        let e = cast("nint", Expr::Int(5));
        assert_eq!(simplify(e.clone()), e);
        // 5 is an `int` literal, not a `uint` one; 4294967295 renders with
        // an `L` suffix (a long), so `(uint)` is doing real work there too.
        let e = cast("uint", Expr::Int(5));
        assert_eq!(simplify(e.clone()), e);
        let e = cast("uint", Expr::Int(0xFFFF_FFFF));
        assert_eq!(simplify(e.clone()), e);
        let e = cast("ulong", Expr::Int(5_000_000_000));
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn float_double_bool_string_and_null_casts_are_dropped() {
        assert_eq!(simplify(cast("float", Expr::Float(1.5))), Expr::Float(1.5));
        assert_eq!(simplify(cast("double", Expr::Double(2.5))), Expr::Double(2.5));
        assert_eq!(simplify(cast("bool", Expr::Bool(true))), Expr::Bool(true));
        assert_eq!(
            simplify(cast("string", Expr::Str("hi".into()))),
            Expr::Str("hi".into())
        );
        assert_eq!(simplify(cast("object", Expr::Null)), Expr::Null);
        // Nullable<T> is a value type: `(int?)null` is not `null` in a
        // non-nullable context, so the cast stays.
        let nullable = cast("int?", Expr::Null);
        assert_eq!(simplify(nullable.clone()), nullable);
    }

    #[test]
    fn a_literal_of_a_different_type_keeps_its_cast() {
        // `(double)1.5f` widens; the float literal is not a double literal.
        let e = cast("double", Expr::Float(1.5));
        assert_eq!(simplify(e.clone()), e);
        // `(float)5` is a real conversion (int -> float), not a spelling.
        let e = cast("float", Expr::Int(5));
        assert_eq!(simplify(e.clone()), e);
    }

    // ---- nested / same-type casts ---------------------------------------

    #[test]
    fn nested_same_type_cast_collapses_to_one() {
        let e = cast("int", cast("int", v("x")));
        assert_eq!(simplify(e), cast("int", v("x")));
        let e = cast("nint", cast("nint", v("x")));
        assert_eq!(simplify(e), cast("nint", v("x")));
    }

    #[test]
    fn cast_of_a_value_already_of_the_target_type_is_dropped() {
        // `(int)*(int*)(p)`.
        let e = cast("int", deref(v("p"), 0, 4));
        assert_eq!(simplify(e), deref(v("p"), 0, 4));
        // `(nint)*(nint*)(p)`.
        let e = cast("nint", deref(v("p"), 8, 8));
        assert_eq!(simplify(e), deref(v("p"), 8, 8));
        // `(bool)(a == b)`.
        let e = cast(
            "bool",
            Expr::Cmp(
                crate::recovery::csharp::CmpKind::Eq,
                Box::new(v("a")),
                Box::new(v("b")),
            ),
        );
        assert!(matches!(simplify(e), Expr::Cmp(..)));
    }

    #[test]
    fn cast_of_a_new_object_of_the_same_type_is_dropped() {
        let e = cast("Foo", Expr::NewObj("Foo".into(), vec![]));
        assert_eq!(simplify(e), Expr::NewObj("Foo".into(), vec![]));
    }

    #[test]
    fn exact_widening_layer_is_dropped() {
        // `(long)(int)x` truncates x to int; the long layer adds nothing.
        let e = cast("long", cast("int", v("x")));
        assert_eq!(simplify(e), cast("int", v("x")));
        // `(double)(float)x` keeps the float rounding, but not the double cast.
        let e = cast("double", cast("float", v("x")));
        assert_eq!(simplify(e), cast("float", v("x")));
    }

    #[test]
    fn a_lossy_intermediate_cast_is_never_dropped() {
        // `(int)(float)x` is a truncation with float precision loss; removing
        // either layer would change the value.
        let e = cast("int", cast("float", v("x")));
        assert_eq!(simplify(e.clone()), e);
        // `(double)(float)bigInt` loses precision in the float; the outer
        // double cast is exact, so only it is dropped — that is the rule
        // above, and the float cast must survive.
        let out = simplify(cast("double", cast("float", cast("int", v("x")))));
        assert_eq!(out, cast("float", cast("int", v("x"))));
    }

    // ---- deref / address -------------------------------------------------

    #[test]
    fn deref_of_address_of_a_local_cancels() {
        let e = deref(addr(v("v46")), 0, 8);
        assert_eq!(simplify(e), v("v46"));
        // width 0 means "pointer width, not stated".
        let e = deref(addr(v("stack_3")), 0, 0);
        assert_eq!(simplify(e), v("stack_3"));
    }

    #[test]
    fn deref_of_address_of_this_and_fields_cancels() {
        let e = deref(addr(Expr::This), 0, 8);
        assert_eq!(simplify(e), Expr::This);
        let field = Expr::Field(Box::new(Expr::This), "f".into());
        let e = deref(addr(field.clone()), 0, 8);
        assert_eq!(simplify(e), field);
    }

    #[test]
    fn deref_of_a_side_effecting_receiver_does_not_cancel() {
        let call = Expr::Call {
            recv: None,
            name: "M".into(),
            args: vec![],
        };
        let field = Expr::Field(Box::new(call), "f".into());
        let e = deref(addr(field.clone()), 0, 8);
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn a_narrower_access_than_the_cell_is_not_cancelled_without_type_evidence() {
        // `*(int*)(&x)` is only x when x really is a 32-bit integer.
        let e = deref(addr(v("x")), 0, 4);
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn a_narrow_access_cancels_when_the_local_is_declared_at_that_width() {
        let (_, stmts) = simplify_stmts(vec![
            assign("x", Some("int"), Expr::Int(0)),
            Stmt::Eval(deref(addr(v("x")), 0, 4)),
        ]);
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::Var(n)) if n == "x"));
    }

    #[test]
    fn a_pointer_width_read_of_an_int_local_is_not_cancelled() {
        let (_, stmts) = simplify_stmts(vec![
            assign("x", Some("int"), Expr::Int(0)),
            Stmt::Eval(deref(addr(v("x")), 0, 8)),
        ]);
        assert!(matches!(&stmts[1], Stmt::Eval(Expr::RawDeref(..))));
    }

    #[test]
    fn address_of_a_deref_is_not_cancelled() {
        // `&*(T*)(p)` is the address `p`, but the two operators are not a
        // no-op in this direction and the pointer type may differ — leave it.
        let e = addr(deref(v("p"), 0, 8));
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn deref_of_a_nonzero_addend_is_not_cancelled() {
        let e = deref(addr(v("stack_7")), 28, 4);
        assert_eq!(simplify(e.clone()), e);
    }

    #[test]
    fn addends_inside_the_base_are_folded_into_the_node_offset() {
        let base = bin(BinOp::Add, v("p"), Expr::Int(8));
        let e = deref(base, 16, 4);
        assert_eq!(simplify(e), deref(v("p"), 24, 4));
        // Subtraction combines with a negative direction.
        let base = bin(BinOp::Sub, v("p"), Expr::Int(8));
        let e = deref(base, 16, 4);
        assert_eq!(simplify(e), deref(v("p"), 8, 4));
        // RawAddr carries its offset the same way.
        let base = bin(BinOp::Add, v("p"), Expr::Int(8));
        let e = Expr::RawAddr(Box::new(base), 16);
        assert_eq!(simplify(e), Expr::RawAddr(Box::new(v("p")), 24));
    }

    // ---- 32-bit mask -----------------------------------------------------

    #[test]
    fn a_32_bit_mask_under_an_int_cast_is_redundant() {
        let masked = bin(BinOp::And, v("x"), Expr::Int(0xFFFF_FFFF));
        let out = simplify(cast("int", masked));
        // The mask goes; the (int) stays because `x`'s own type is unknown.
        assert_eq!(out, cast("int", v("x")));
    }

    #[test]
    fn a_64_bit_shift_idiom_keeps_its_mask() {
        let masked = bin(BinOp::And, v("x"), Expr::Int(0xFFFF_FFFF));
        let out = simplify(bin(BinOp::Shr, bin(BinOp::Shl, masked, Expr::Int(32)), Expr::Int(32)));
        assert!(
            matches!(
                out,
                Expr::Bin(BinOp::Shr, ref l, _)
                    if matches!(**l, Expr::Bin(BinOp::Shl, ref m, _)
                        if matches!(**m, Expr::Bin(BinOp::And, ..)))
            ),
            "the zero-extension idiom must keep its mask: {out:?}"
        );
    }

    #[test]
    fn a_root_mask_on_an_int_local_assignment_is_redundant() {
        let masked = bin(BinOp::And, v("x"), Expr::Int(0xFFFF_FFFF));
        let (n, stmts) = simplify_stmts(vec![
            assign("x", Some("int"), Expr::Int(0)),
            assign("y", Some("int"), masked),
        ]);
        assert!(n >= 1);
        assert!(matches!(&stmts[1], Stmt::Assign(_, _, Expr::Var(v)) if v == "x"));
    }

    #[test]
    fn a_root_mask_on_a_long_local_is_kept() {
        let masked = bin(BinOp::And, v("x"), Expr::Int(0xFFFF_FFFF));
        let (_, stmts) = simplify_stmts(vec![
            assign("x", Some("long"), Expr::Int(0)),
            assign("y", Some("long"), masked.clone()),
        ]);
        assert!(matches!(
            &stmts[1],
            Stmt::Assign(_, _, Expr::Bin(BinOp::And, ..))
        ));
    }

    // ---- idempotency -----------------------------------------------------

    #[test]
    fn a_second_application_is_a_no_op() {
        let exprs = vec![
            cast("int", Expr::Int(5)),
            cast("int", cast("int", v("x"))),
            cast("long", cast("int", v("x"))),
            deref(addr(v("v46")), 0, 8),
            Expr::RawAddr(Box::new(bin(BinOp::Add, v("p"), Expr::Int(8))), 16),
            cast("int", deref(v("p"), 0, 4)),
        ];
        for e in exprs {
            let mut stmts = vec![Stmt::Eval(e)];
            assert!(apply(&mut stmts) > 0);
            assert_eq!(apply(&mut stmts), 0, "second apply changed: {stmts:?}");
        }
    }

    #[test]
    fn rewrites_inside_nested_bodies_are_found() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(deref(addr(v("v1")), 0, 8))],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(
            &stmts[0],
            Stmt::If { then_body, .. } if matches!(&then_body[0], Stmt::Eval(Expr::Var(n)) if n == "v1")
        ));
    }

    #[test]
    fn evaluation_order_is_never_changed_by_this_pass() {
        // No rule moves work: the deref is replaced by the SAME storage cell,
        // and a cast around a call is never dropped (a call has no type we
        // can prove, so the conversion stays).
        let call = Expr::Call {
            recv: None,
            name: "M".into(),
            args: vec![],
        };
        let e = deref(addr(v("x")), 0, 8);
        assert_eq!(simplify(e), v("x"));
        let e = cast("object", call.clone());
        assert_eq!(simplify(e.clone()), e);
    }
}
