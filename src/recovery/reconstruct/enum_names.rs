//! Enum member name recovery: replaces integer constants with enum member names
//! when the context indicates an enum type.
//!
//! ## What this pass does
//!
//! When a local variable or parameter has an enum type (determined by the type
//! propagation fixpoint or heuristic type inference), integer constants assigned
//! to it are replaced with the corresponding enum member name:
//!
//! ```text
//! v3 = 4;    -->    v3 = ConnectionState.Connected;
//! ```
//!
//! The pass also handles:
//! - Integer constants used as arguments to calls where the parameter has an enum type
//! - Integer constants in comparison expressions where one side has an enum type
//! - Integer constants in return statements where the return type is an enum
//!
//! ## Safety
//!
//! Only exact matches are rewritten:
//! - The variable must have a known enum type (from `local_types` map)
//! - The enum type must have a member with the exact integer value
//! - A failed match is a no-op: the original integer is preserved.

use super::super::csharp::{Expr, Stmt};
use super::super::resolve::ResolveMaps;
use std::collections::HashMap;

/// Run the enum member name recovery over one method body.
///
/// `local_types` maps variable display names to their TypeDef index (tdi).
/// `maps` provides the `enum_members` lookup table.
///
/// Returns the number of integer constants replaced with enum member names.
pub fn apply(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
) -> usize {
    if maps.enum_members.is_empty() || local_types.is_empty() {
        return 0;
    }
    let mut rewrites = 0;
    for stmt in stmts.iter_mut() {
        rewrites += rewrite_stmt(stmt, maps, local_types);
    }
    rewrites
}

/// Rewrite a single statement, recursing into nested bodies.
fn rewrite_stmt(
    stmt: &mut Stmt,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
) -> usize {
    let mut rewrites = 0;
    match stmt {
        Stmt::Declare(..) => {}
        Stmt::Assign(name, _, expr) => {
            // Check if the assigned variable has an enum type.
            if let Some(&tdi) = local_types.get(name.as_str()) {
                rewrites += rewrite_expr_for_enum(expr, tdi, maps);
            }
            // Also recurse into subexpressions for nested enum contexts.
            rewrites += rewrite_expr_children(expr, maps, local_types);
        }
        Stmt::Eval(expr) => {
            rewrites += rewrite_expr_children(expr, maps, local_types);
        }
        Stmt::Return(Some(expr)) => {
            rewrites += rewrite_expr_children(expr, maps, local_types);
        }
        Stmt::Throw(expr) => {
            rewrites += rewrite_expr_children(expr, maps, local_types);
        }
        Stmt::YieldReturn(expr) => {
            rewrites += rewrite_expr_children(expr, maps, local_types);
        }
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            rewrites += rewrite_expr_children(cond, maps, local_types);
            for s in then_body {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
            for s in else_body {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
        }
        Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
            rewrites += rewrite_expr_children(cond, maps, local_types);
            for s in body {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
        }
        Stmt::For {
            init,
            cond,
            incr,
            body,
        } => {
            for s in init {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
            if let Some(c) = cond {
                rewrites += rewrite_expr_children(c, maps, local_types);
            }
            for s in incr {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
            for s in body {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
        }
        Stmt::ForEach { body, .. } => {
            for s in body {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
        }
        Stmt::Using { decl, body } => {
            rewrites += rewrite_stmt(decl, maps, local_types);
            for s in body {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
        }
        Stmt::Lock { expr, body } => {
            rewrites += rewrite_expr_children(expr, maps, local_types);
            for s in body {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
        }
        Stmt::Try {
            try_body,
            catches,
            finally_body,
        } => {
            for s in try_body {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
            for (_, b) in catches {
                for s in b {
                    rewrites += rewrite_stmt(s, maps, local_types);
                }
            }
            if let Some(fb) = finally_body {
                for s in fb {
                    rewrites += rewrite_stmt(s, maps, local_types);
                }
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases {
                for s in b {
                    rewrites += rewrite_stmt(s, maps, local_types);
                }
            }
            for s in tail {
                rewrites += rewrite_stmt(s, maps, local_types);
            }
            if let Some(arm) = default {
                for s in arm {
                    rewrites += rewrite_stmt(s, maps, local_types);
                }
            }
        }
        Stmt::StringSwitch { cases, tail, .. } => {
            for (_, b) in cases {
                for s in b {
                    rewrites += rewrite_stmt(s, maps, local_types);
                }
            }
            if let Some(t) = tail {
                for s in t {
                    rewrites += rewrite_stmt(s, maps, local_types);
                }
            }
        }
        Stmt::EventAssign { field, value, .. } => {
            rewrites += rewrite_expr_children(field, maps, local_types);
            rewrites += rewrite_expr_children(value, maps, local_types);
        }
        Stmt::IndexAssign { collection, index, value } => {
            rewrites += rewrite_expr_children(collection, maps, local_types);
            rewrites += rewrite_expr_children(index, maps, local_types);
            rewrites += rewrite_expr_children(value, maps, local_types);
        }
        Stmt::GotoCase(_) => {}
        // Statements with no expressions or children to rewrite.
        Stmt::Return(None)
        | Stmt::Goto(_)
        | Stmt::Label(_)
        | Stmt::Comment(_)
        | Stmt::Attribute { .. }
        | Stmt::Break
        | Stmt::Continue
        | Stmt::YieldBreak => {}
    }
    rewrites
}

/// If `expr` is an `Int(n)` and `tdi` is an enum type with member `n`,
/// replace it with `Var("EnumType.MemberName")`.
fn rewrite_expr_for_enum(expr: &mut Expr, tdi: i32, maps: &ResolveMaps) -> usize {
    if let Expr::Int(n) = expr {
        let n_val = *n;
        if let Some(members) = maps.enum_members.get(&tdi) {
            if let Some(member_name) = members.get(&n_val) {
                // Get the enum type name from tdi_names.
                let type_name = maps
                    .tdi_names
                    .get(&tdi)
                    .map(|(ns, name)| {
                        if ns.is_empty() {
                            name.clone()
                        } else {
                            format!("{}.{}", ns, name)
                        }
                    })
                    .unwrap_or_else(|| format!("Enum_{}", tdi));
                *expr = Expr::Var(format!("{}.{}", type_name, member_name));
                return 1;
            }
        }
    }
    0
}

/// Recurse into the children of an expression, looking for enum replacement
/// opportunities. This handles cases like call arguments, comparison operands,
/// binary expression operands, etc.
fn rewrite_expr_children(expr: &mut Expr, maps: &ResolveMaps, local_types: &HashMap<String, i32>) -> usize {
    let mut rewrites = 0;
    match expr {
        Expr::Cmp(_, a, b) => {
            // For comparisons, if one side is a variable with an enum type,
            // try to replace the other side (if it's an Int) with the enum member.
            rewrites += try_rewrite_cmp_operands(a, b, maps, local_types);
            rewrites += try_rewrite_cmp_operands(b, a, maps, local_types);
            rewrites += rewrite_expr_children(a, maps, local_types);
            rewrites += rewrite_expr_children(b, maps, local_types);
        }
        Expr::Bin(_, a, b) => {
            rewrites += rewrite_expr_children(a, maps, local_types);
            rewrites += rewrite_expr_children(b, maps, local_types);
        }
        Expr::Not(e) | Expr::Neg(e) | Expr::Cast(_, e) | Expr::AddressOf(e) | Expr::Await(e) => {
            rewrites += rewrite_expr_children(e, maps, local_types);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            rewrites += rewrite_expr_children(r, maps, local_types);
        }
        Expr::Cond(c, a, b) => {
            rewrites += rewrite_expr_children(c, maps, local_types);
            rewrites += rewrite_expr_children(a, maps, local_types);
            rewrites += rewrite_expr_children(b, maps, local_types);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                rewrites += rewrite_expr_children(r, maps, local_types);
            }
            for a in args {
                rewrites += rewrite_expr_children(a, maps, local_types);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                rewrites += rewrite_expr_children(a, maps, local_types);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for e in items {
                rewrites += rewrite_expr_children(e, maps, local_types);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for e in ctor_args {
                rewrites += rewrite_expr_children(e, maps, local_types);
            }
            for e in items {
                rewrites += rewrite_expr_children(e, maps, local_types);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for e in ctor_args {
                rewrites += rewrite_expr_children(e, maps, local_types);
            }
            for (_, v) in fields {
                rewrites += rewrite_expr_children(v, maps, local_types);
            }
        }
        Expr::Index { collection, index } => {
            rewrites += rewrite_expr_children(collection, maps, local_types);
            rewrites += rewrite_expr_children(index, maps, local_types);
        }
        Expr::Lambda { body, .. } => {
            rewrites += rewrite_expr_children(body, maps, local_types);
        }
        Expr::NullConditional { receiver, access } => {
            rewrites += rewrite_expr_children(receiver, maps, local_types);
            rewrites += rewrite_expr_children(access, maps, local_types);
        }
        // Leaf expressions: nothing to recurse into.
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
        | Expr::StaticField(..)
        | Expr::MethodGroup(_)
        | Expr::PatternMatch { .. }
        | Expr::StringInterp { .. }
        | Expr::Tuple { .. }
        | Expr::MethodChain { .. } | Expr::SwitchExpr { .. }
        | Expr::Range { .. } | Expr::IndexFromEnd(_) | Expr::TypeOf(_) => {}
    }
    rewrites
}

/// For a comparison expression, if one operand is a variable with an enum type
/// and the other is an integer constant, replace the integer with the enum member.
fn try_rewrite_cmp_operands(
    enum_side: &Expr,
    int_side: &mut Expr,
    maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
) -> usize {
    if let Expr::Var(name) = enum_side {
        if let Some(&tdi) = local_types.get(name.as_str()) {
            return rewrite_expr_for_enum(int_side, tdi, maps);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_maps() -> ResolveMaps {
        let mut maps = ResolveMaps::default();
        // Create a fake enum type at tdi=100 with members:
        // 0 -> "None", 1 -> "First", 2 -> "Second", 4 -> "Connected"
        let mut members = HashMap::new();
        members.insert(0, "None".to_string());
        members.insert(1, "First".to_string());
        members.insert(2, "Second".to_string());
        members.insert(4, "Connected".to_string());
        maps.enum_members.insert(100, members);
        maps.tdi_names.insert(100, ("TestNS".to_string(), "ConnectionState".to_string()));
        maps
    }

    #[test]
    fn simple_assignment_replacement() {
        let maps = make_maps();
        let mut local_types = HashMap::new();
        local_types.insert("v3".to_string(), 100);
        let mut stmts = vec![Stmt::Assign("v3".into(), None, Expr::Int(4))];
        let n = apply(&mut stmts, &maps, &local_types);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::Assign(name, _, Expr::Var(var_name)) => {
                assert_eq!(name, "v3");
                assert_eq!(var_name, "TestNS.ConnectionState.Connected");
            }
            other => panic!("expected Assign(v3, _, Var(...)), got {other:?}"),
        }
    }

    #[test]
    fn comparison_replacement() {
        let maps = make_maps();
        let mut local_types = HashMap::new();
        local_types.insert("state".to_string(), 100);
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                super::super::super::csharp::CmpKind::Eq,
                Box::new(Expr::Var("state".into())),
                Box::new(Expr::Int(1)),
            ),
            then_body: vec![],
            else_body: vec![],
        }];
        let n = apply(&mut stmts, &maps, &local_types);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { cond, .. } => match cond {
                Expr::Cmp(_, lhs, rhs) => {
                    assert!(matches!(lhs.as_ref(), Expr::Var(n) if n == "state"));
                    assert!(matches!(rhs.as_ref(), Expr::Var(n) if n == "TestNS.ConnectionState.First"));
                }
                other => panic!("expected Cmp, got {other:?}"),
            },
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn no_match_leaves_integer_unchanged() {
        let maps = make_maps();
        let mut local_types = HashMap::new();
        local_types.insert("v3".to_string(), 100);
        // Value 99 has no enum member.
        let mut stmts = vec![Stmt::Assign("v3".into(), None, Expr::Int(99))];
        let n = apply(&mut stmts, &maps, &local_types);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::Int(99))));
    }

    #[test]
    fn non_enum_type_not_replaced() {
        let maps = make_maps();
        let mut local_types = HashMap::new();
        local_types.insert("v3".to_string(), 200); // tdi=200 is not an enum
        let mut stmts = vec![Stmt::Assign("v3".into(), None, Expr::Int(4))];
        let n = apply(&mut stmts, &maps, &local_types);
        assert_eq!(n, 0);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::Int(4))));
    }

    #[test]
    fn empty_enum_members_is_noop() {
        let maps = ResolveMaps::default();
        let mut local_types = HashMap::new();
        local_types.insert("v3".to_string(), 100);
        let mut stmts = vec![Stmt::Assign("v3".into(), None, Expr::Int(4))];
        let n = apply(&mut stmts, &maps, &local_types);
        assert_eq!(n, 0);
    }
}
