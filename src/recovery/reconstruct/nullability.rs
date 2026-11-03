//! Nullability peepholes: `x == 0` -> `x == null` for provably reference-typed
//! values.
//!
//! IL2CPP tests a managed reference for null with an integer compare against
//! `0`; the source wrote `== null` / `!= null`. An integer local's real test
//! compiles to the *same* shape (`if (count != 0)`), so the pass cannot rewrite
//! on shape alone: `count != null` would not even be valid C#. It therefore
//! requires proof that the compared value is a managed reference.
//!
//! ## Where the proof comes from
//!
//! Measured on the x64 corpus, the output declares almost nothing —
//! `v13 = 0;` carries no type — so a typed-declaration-only rule (the previous
//! version) can essentially never fire. Three independent sources are combined
//! instead, and a local is a reference when ANY of them holds and it is not
//! excluded:
//!
//!   1. **a declared reference type.** `Stmt::Assign(name, Some(ty), _)` is the
//!      renderer's declaration site, so `Some(ty)` naming a non-primitive,
//!      non-`var` type proves the local is a reference. (`var` proves nothing
//!      and is refused; `int`/`nint`/`float?`/`void*` are value types.)
//!   2. **the renderer used it as a MANAGED receiver.** `recv.M(args)` and
//!      `recv.f` are only formed when the recovery resolved a managed member
//!      named `M`/`f` on `recv` — a raw pointer would have been emitted as
//!      `*(nint*)(recv + off)`. So a local that is the receiver of a `Call` or
//!      the base of a `Field` is an object reference.
//!   3. **a copy chain.** `a = b` where `b` is a proven reference makes `a` one,
//!      to a fixpoint (IL2CPP copies object references through temporaries
//!      constantly).
//!
//! ## The exclusion that keeps (2) honest
//!
//! A *struct* local also gets `v.M(...)` / `v.f` spellings, and a struct is a
//! value type — so (2) alone would classify one as an object. The one signal
//! the tree carries for a value type is that its ADDRESS is taken: a struct is
//! passed/returned by reference, which binds as `&v` (`Expr::AddressOf`) or
//! `(nint*)(v)` (`Expr::RawAddr`). Any local whose address is taken anywhere in
//! the body is therefore removed from the reference set. This is deliberately
//! one-sided: it can only make the pass refuse.
//!
//! Everything else is REFUSED — a bare undeclared local, a `Call`'s result (it
//! may be an `int`), a `Field` read (`this.attrNames` — the field's own type is
//! not in the tree), and every arithmetic shape (`Int`, `Float`, `Double`,
//! `Bin`, `Cmp`, `Neg`, `Cast`). A refusal leaves the flat `== 0` exactly as
//! recovered: accurate-but-ugly beats a wrong `null`.
//!
//! The strictly better version is TYPED (ask the `LocalTable` via `MethodCtx`
//! at statement-render time); this pass is the provable subset that needs no
//! `MethodCtx`.

use super::super::csharp::CmpKind;
use super::{Expr, Stmt, children, walk_mut};
use std::collections::HashSet;

/// The evidence collected over a whole method body.
#[derive(Default)]
struct ReferenceProof {
    /// Locals declared with a reference-shaped type (source 1).
    declared: HashSet<String>,
    /// Locals used as a managed receiver / field base (source 2).
    receiver: HashSet<String>,
    /// Locals whose ADDRESS is taken (value-type signal, excludes).
    addressed: HashSet<String>,
    /// `a = b` copy pairs (source 3).
    copies: Vec<(String, String)>,
    /// The solved set.
    refs: HashSet<String>,
}

impl ReferenceProof {
    fn scan_expr(&mut self, e: &Expr) {
        match e {
            Expr::Field(recv, _) => {
                if let Expr::Var(n) = recv.as_ref() {
                    self.receiver.insert(n.clone());
                }
                self.scan_expr(recv);
            }
            Expr::Call { recv, args, .. } => {
                if let Some(r) = recv {
                    if let Expr::Var(n) = r.as_ref() {
                        self.receiver.insert(n.clone());
                    }
                    self.scan_expr(r);
                }
                for a in args {
                    self.scan_expr(a);
                }
            }
            Expr::AddressOf(inner) | Expr::RawAddr(inner, _) => {
                if let Expr::Var(n) = inner.as_ref() {
                    self.addressed.insert(n.clone());
                }
                self.scan_expr(inner);
            }
            // The base of a raw deref is NOT an address-of: `*(nint*)(v + 8)` is
            // the ordinary way a recovered field read is spelled when the field
            // name was not resolved.
            Expr::RawDeref(base, _, _) => self.scan_expr(base),
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
                self.scan_expr(a);
                self.scan_expr(b);
            }
            Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::Await(x) => self.scan_expr(x),
            Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
                for a in args {
                    self.scan_expr(a);
                }
            }
            Expr::ArrayInit { items, .. } => {
                for a in items {
                    self.scan_expr(a);
                }
            }
            Expr::CollectionInit {
                ctor_args, items, ..
            } => {
                for a in ctor_args.iter().chain(items) {
                    self.scan_expr(a);
                }
            }
            Expr::Cond(c, a, b) => {
                self.scan_expr(c);
                self.scan_expr(a);
                self.scan_expr(b);
            }
            Expr::Lambda { body, .. } => self.scan_expr(body),
            _ => {}
        }
    }

    fn scan_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Assign(n, Some(ty), e) => {
                if declared_type_is_reference(ty) {
                    self.declared.insert(n.clone());
                }
                self.scan_expr(e);
            }
            Stmt::Assign(n, None, e) => {
                if let Expr::Var(m) = e {
                    self.copies.push((n.clone(), m.clone()));
                }
                self.scan_expr(e);
            }
            Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => self.scan_expr(e),
            Stmt::If { cond, .. } => self.scan_expr(cond),
            Stmt::While { cond, .. } => self.scan_expr(cond),
            Stmt::DoWhile { cond, .. } => self.scan_expr(cond),
            Stmt::For { cond, .. } => {
                if let Some(c) = cond {
                    self.scan_expr(c);
                }
            }
            Stmt::Lock { expr, .. } => self.scan_expr(expr),
            Stmt::Return(Some(e)) => self.scan_expr(e),
            Stmt::Using { decl, .. } => self.scan_stmt(decl),
            _ => {}
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for k in kids {
            for s2 in k {
                self.scan_stmt(s2);
            }
        }
    }

    /// Solve the reference set: (declared ∪ receiver) − addressed, closed over
    /// the copy chains to a fixpoint.
    fn solve(&mut self) {
        for n in self.declared.iter().chain(self.receiver.iter()) {
            if !self.addressed.contains(n) {
                self.refs.insert(n.clone());
            }
        }
        loop {
            let mut changed = false;
            for (a, b) in self.copies.clone() {
                if self.refs.contains(&b) && !self.addressed.contains(&a) && self.refs.insert(a) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
}

/// C# built-in value-type keywords plus the pointer/`void` spellings.
fn is_value_type_keyword(ty: &str) -> bool {
    let t = ty.trim();
    // `int?` is a value type too; `T[]`/`List<T>` are references.
    let t = t.strip_suffix('?').unwrap_or(t);
    matches!(
        t,
        "bool"
            | "byte"
            | "sbyte"
            | "char"
            | "short"
            | "ushort"
            | "int"
            | "uint"
            | "long"
            | "ulong"
            | "float"
            | "double"
            | "decimal"
            | "nint"
            | "nuint"
            | "IntPtr"
            | "UIntPtr"
            | "void"
            | "void*"
    )
}

/// Is a declaration type name a managed reference (so `== 0` means `== null`)?
///
/// `var` is deliberately NOT a reference: it states no type, so it proves
/// nothing and must not enable the rewrite.
fn declared_type_is_reference(ty: &str) -> bool {
    let t = ty.trim();
    if t.is_empty() || t == "var" {
        return false;
    }
    // An array type is a reference regardless of its element type.
    if t.ends_with("[]") {
        return true;
    }
    // Strip a generic argument list: `List<int>` -> `List`.
    let bare = t.split('<').next().unwrap_or(t).trim();
    let bare = bare.split('`').next().unwrap_or(bare).trim();
    !is_value_type_keyword(bare)
}

/// A value proven to be a managed reference: a fresh allocation, or a local the
/// proof classified.
fn is_proven_reference(e: &Expr, refs: &HashSet<String>) -> bool {
    match e {
        Expr::NewObj(..) | Expr::NewArr(..) | Expr::ArrayInit { .. } | Expr::CollectionInit { .. } => {
            true
        }
        Expr::Var(n) => refs.contains(n),
        _ => false,
    }
}

/// Recursively rewrite null tests inside one expression, returning the count.
fn rewrite_expr(e: &mut Expr, refs: &HashSet<String>) -> usize {
    let mut n = 0usize;
    match e {
        Expr::Cmp(k, a, b) => {
            n += rewrite_expr(a, refs);
            n += rewrite_expr(b, refs);
            if matches!(k, CmpKind::Eq | CmpKind::Ne) {
                // Exactly one side is the literal `0` and the other is a proven
                // reference. The literal side is trivially numeric, so it must
                // NOT take part in the arithmetic check.
                let (ref_zero, zero_is_b) = match (a.as_ref(), b.as_ref()) {
                    (_, Expr::Int(0)) if is_proven_reference(a, refs) => (true, true),
                    (Expr::Int(0), _) if is_proven_reference(b, refs) => (true, false),
                    _ => (false, false),
                };
                if ref_zero {
                    if zero_is_b {
                        **b = Expr::Null;
                    } else {
                        **a = Expr::Null;
                    }
                    n += 1;
                }
            }
        }
        Expr::Bin(_, a, b) => {
            n += rewrite_expr(a, refs);
            n += rewrite_expr(b, refs);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            n += rewrite_expr(x, refs);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            n += rewrite_expr(r, refs);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                n += rewrite_expr(r, refs);
            }
            for a in args.iter_mut() {
                n += rewrite_expr(a, refs);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args.iter_mut() {
                n += rewrite_expr(a, refs);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for a in items.iter_mut() {
                n += rewrite_expr(a, refs);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter_mut().chain(items) {
                n += rewrite_expr(a, refs);
            }
        }
        Expr::Cond(c, a, b) => {
            n += rewrite_expr(c, refs);
            n += rewrite_expr(a, refs);
            n += rewrite_expr(b, refs);
        }
        Expr::Lambda { body, .. } => n += rewrite_expr(body, refs),
        _ => {}
    }
    n
}

fn rewrite_stmt(s: &mut Stmt, refs: &HashSet<String>) -> usize {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => rewrite_expr(e, refs),
        Stmt::Assign(_, _, e) => rewrite_expr(e, refs),
        Stmt::If { cond, .. } => rewrite_expr(cond, refs),
        Stmt::While { cond, .. } => rewrite_expr(cond, refs),
        Stmt::DoWhile { cond, .. } => rewrite_expr(cond, refs),
        Stmt::For { cond, .. } => cond.as_mut().map_or(0, |c| rewrite_expr(c, refs)),
        Stmt::Lock { expr, .. } => rewrite_expr(expr, refs),
        Stmt::Return(Some(e)) => rewrite_expr(e, refs),
        _ => 0,
    }
}

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    // Phase 1 (whole body): prove which locals are managed references.
    let mut proof = ReferenceProof::default();
    for s in stmts.iter() {
        proof.scan_stmt(s);
    }
    proof.solve();
    // Phase 2: rewrite the null tests those proofs license.
    let mut count = 0usize;
    walk_mut(stmts, &mut |list| {
        for s in list.iter_mut() {
            count += rewrite_stmt(s, &proof.refs);
        }
    });
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmp_eq(a: Expr, b: Expr) -> Expr {
        Expr::Cmp(CmpKind::Eq, Box::new(a), Box::new(b))
    }
    fn var(n: &str) -> Expr {
        Expr::Var(n.into())
    }
    fn if_zero(n: &str) -> Stmt {
        Stmt::If {
            cond: cmp_eq(var(n), Expr::Int(0)),
            then_body: vec![],
            else_body: vec![],
        }
    }
    /// Did the statement's condition become `… == null`?
    fn became_null(s: &Stmt) -> bool {
        matches!(s, Stmt::If { cond: Expr::Cmp(_, _, b), .. } if matches!(**b, Expr::Null))
    }

    /// A declared reference local compared with zero becomes a null test.
    #[test]
    fn declared_reference_compared_with_zero_becomes_null() {
        let mut stmts = vec![
            Stmt::Assign(
                "v12".into(),
                Some("Enemy".into()),
                Expr::NewObj("Enemy".into(), vec![]),
            ),
            if_zero("v12"),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(became_null(&stmts[1]));
    }

    /// A fresh allocation is a reference no matter how it is spelled.
    #[test]
    fn allocation_compared_with_zero_becomes_null() {
        let mut stmts = vec![Stmt::If {
            cond: cmp_eq(Expr::NewObj("List".into(), vec![]), Expr::Int(0)),
            then_body: vec![],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 1);
    }

    /// The real corpus spelling: the local is used as a managed receiver
    /// somewhere in the body, so it is an object reference.
    #[test]
    fn a_managed_receiver_is_a_reference() {
        let mut stmts = vec![
            // v39.get_Item(v11) — an instance call on a recovered type.
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("v39"))),
                name: "get_Item".into(),
                args: vec![var("v11")],
            }),
            if_zero("v39"),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(became_null(&stmts[1]));
    }

    /// A local that is only ever a raw-deref base is NOT classified: the base of
    /// `*(nint*)(v +8)` is not evidence of a managed member.
    #[test]
    fn a_raw_deref_base_is_not_a_reference() {
        let mut stmts = vec![
            Stmt::Assign(
                "v39".into(),
                None,
                Expr::RawDeref(Box::new(var("v0")), 16, 0),
            ),
            if_zero("v39"),
        ];
        assert_eq!(apply(&mut stmts), 0, "no managed-member evidence");
    }

    /// A copy chain carries the proof: `a = b` where `b` is a receiver.
    #[test]
    fn a_copy_chain_carries_the_proof() {
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("src"))),
                name: "get_Name".into(),
                args: vec![],
            }),
            Stmt::Assign("mid".into(), None, var("src")),
            Stmt::Assign("v9".into(), None, var("mid")),
            if_zero("v9"),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(became_null(&stmts[3]));
    }

    /// A local whose ADDRESS is taken is (very likely) a value type — excluded.
    #[test]
    fn an_address_taken_local_is_excluded() {
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("v9"))),
                name: "Normalize".into(),
                args: vec![],
            }),
            Stmt::Eval(Expr::AddressOf(Box::new(var("v9")))),
            if_zero("v9"),
        ];
        assert_eq!(apply(&mut stmts), 0, "a struct gets member calls too");
    }

    /// An `int` local's `== 0` is a real integer test; `count == null` would not
    /// even compile.
    #[test]
    fn integer_local_compared_with_zero_is_refused() {
        let mut stmts = vec![
            Stmt::Assign("count".into(), Some("int".into()), Expr::Int(0)),
            if_zero("count"),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// An undeclared, otherwise-unused local proves nothing.
    #[test]
    fn undeclared_local_compared_with_zero_is_refused() {
        let mut stmts = vec![if_zero("v7")];
        assert_eq!(apply(&mut stmts), 0, "a bare `vN == 0` is ambiguous");
    }

    /// The declaration may live in a nested list; the proof still counts.
    #[test]
    fn declaration_inside_a_nested_body_is_seen() {
        let mut stmts = vec![
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Assign(
                    "o".into(),
                    Some("object".into()),
                    Expr::NewObj("object".into(), vec![]),
                )],
                else_body: vec![],
            },
            if_zero("o"),
        ];
        assert_eq!(apply(&mut stmts), 1);
    }

    #[test]
    fn arithmetic_zero_test_is_refused() {
        let mut stmts = vec![Stmt::If {
            cond: cmp_eq(
                Expr::Bin(
                    super::super::super::csharp::BinOp::Add,
                    Box::new(var("v")),
                    Box::new(Expr::Int(8)),
                ),
                Expr::Int(0),
            ),
            then_body: vec![],
            else_body: vec![],
        }];
        assert_eq!(apply(&mut stmts), 0, "`(v + 8) == 0` is not a null test");
    }

    /// `!= 0` is rewritten just like `== 0`.
    #[test]
    fn not_equal_zero_is_rewritten_too() {
        let mut stmts = vec![
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("o"))),
                name: "M".into(),
                args: vec![],
            }),
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ne, Box::new(var("o")), Box::new(Expr::Int(0))),
                then_body: vec![],
                else_body: vec![],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(
            &stmts[1],
            Stmt::If { cond: Expr::Cmp(CmpKind::Ne, _, b), .. } if matches!(**b, Expr::Null)
        ));
    }

    #[test]
    fn var_type_declaration_proves_nothing() {
        assert!(!declared_type_is_reference("var"));
        assert!(declared_type_is_reference("Enemy"));
        assert!(declared_type_is_reference("List<int>"));
        assert!(declared_type_is_reference("int[]"));
        assert!(!declared_type_is_reference("int"));
        assert!(!declared_type_is_reference("nint"));
        assert!(!declared_type_is_reference("float?"));
        assert!(declared_type_is_reference("string"));
    }
}
