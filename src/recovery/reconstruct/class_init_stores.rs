//! Name class-initialization bitmap STORES the way their guard is already named.
//!
//! The lifter emits the classic IL2CPP lazy-init sequence as
//!
//! ```text
//! if (ClassInit.init_31ac000_2831 == 0)
//!     *(byte*)(0x31ac000 +2831) = 1;
//! ```
//!
//! The `field_annotation` fallback names the byte LOAD (the guard), but raw
//! STORE targets are rendered strings and that pass never sees them, so the
//! store keeps its raw address. This pass rewrites a store target
//! `*(byte*)(0xADDR +OFF)` to `ClassInit.init_{addr}_{off}` when the SAME
//! method already names that exact flag — the two sites are the test and the
//! set of one initialization flag, so the name is not a guess.
//!
//! Nothing is renamed when the method carries no such name: an absolute byte
//! write that the annotator did not recognize stays raw.

use std::collections::HashSet;

use crate::recovery::csharp::{Expr, Stmt};

pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut named: HashSet<String> = HashSet::new();
    collect_named(stmts, &mut named);
    if named.is_empty() {
        return 0;
    }
    let mut n = 0;
    super::walk_mut(stmts, &mut |list| {
        for s in list.iter_mut() {
            if let Stmt::Assign(target, _, _) = s {
                n += rewrite_target(target, &named);
            }
        }
    });
    n
}

fn collect_named(stmts: &[Stmt], out: &mut HashSet<String>) {
    fn walk_expr(e: &Expr, out: &mut HashSet<String>) {
        if let Expr::StaticField(ty, name) = e {
            if ty == "ClassInit" {
                out.insert(name.clone());
            }
        }
        for c in super::il2cpp_cleanup::expr_children(e) {
            walk_expr(c, out);
        }
    }
    for s in stmts {
        super::float_locals::for_each_expr(s, &mut |e| walk_expr(e, out));
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            collect_named(k, out);
        }
        if let Stmt::Using { decl, .. } = s {
            collect_named(std::slice::from_ref(decl.as_ref()), out);
        }
    }
}

/// `*(byte*)(0xADDR +OFF)` → `ClassInit.init_{addr:x}_{off}` when that flag is
/// named in this method.
fn rewrite_target(target: &mut String, named: &HashSet<String>) -> usize {
    let Some(rest) = target.strip_prefix("*(byte*)(") else {
        return 0;
    };
    let Some(inner) = rest.strip_suffix(')') else {
        return 0;
    };
    let Some((addr_s, off_s)) = inner.split_once(" +") else {
        return 0;
    };
    let Some(hex) = addr_s.trim().strip_prefix("0x") else {
        return 0;
    };
    let Ok(addr) = u64::from_str_radix(hex, 16) else {
        return 0;
    };
    let Ok(off) = off_s.trim().parse::<u64>() else {
        return 0;
    };
    let flag = format!("init_{addr:x}_{off}");
    if !named.contains(&flag) {
        return 0;
    }
    *target = format!("ClassInit.{flag}");
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard(name: &str) -> Stmt {
        Stmt::If {
            cond: Expr::Cmp(
                crate::recovery::csharp::CmpKind::Eq,
                Box::new(Expr::StaticField("ClassInit".into(), name.into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![],
            else_body: vec![],
        }
    }

    #[test]
    fn a_guard_paired_store_takes_the_flag_name() {
        let mut s = vec![
            guard("init_31ac000_2831"),
            Stmt::Assign(
                "*(byte*)(0x31ac000 +2831)".into(),
                None,
                Expr::Int(1),
            ),
        ];
        assert_eq!(apply(&mut s), 1);
        assert!(matches!(&s[1], Stmt::Assign(t, _, _) if t == "ClassInit.init_31ac000_2831"));
    }

    #[test]
    fn a_store_without_the_named_guard_stays_raw() {
        let mut s = vec![Stmt::Assign(
            "*(byte*)(0x31ac000 +999)".into(),
            None,
            Expr::Int(1),
        )];
        assert_eq!(apply(&mut s), 0);
        assert!(matches!(&s[0], Stmt::Assign(t, _, _) if t == "*(byte*)(0x31ac000 +999)"));
    }
}
