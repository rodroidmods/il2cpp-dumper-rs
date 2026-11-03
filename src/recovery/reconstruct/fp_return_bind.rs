//! Bind a dropped FLOAT-returning call result to the ABI register reads that
//! actually use it.
//!
//! ARM64 returns R4/R8 in `s0`/`d0`, and the compiler may PRESERVE that value
//! in another FP register across intervening calls (`fmov d2, s0`), an
//! instruction the lifter models as a Nop. The SSA then has the call writing a
//! normal destination local while every use reads the register pseudo-locals
//! (`in_vs0`, `in_vd2`), which have no definition; the call itself renders as a
//! bare `this.GetMaxModifiedSpeed();` and the values as `in_vs0 * scale`.
//!
//! The rewrite is deliberately narrow: it fires only when the whole method has
//! EXACTLY ONE call whose result is discarded and at least one otherwise
//! undefined FP register read, and it replaces EVERY such undefined FP pseudo.
//! A broader per-region binding was tried and MEASURED AS A REGRESSION
//! (FP refs 790 → 957, 91 methods worse), so it was reverted: attributing a
//! register read to a call requires the IR/ABI model, not a textual guess.

use crate::recovery::csharp::{Expr, Stmt};
use std::collections::HashSet;

fn is_fp_pseudo(n: &str) -> bool {
    n.strip_prefix("in_vs")
        .or_else(|| n.strip_prefix("in_vd"))
        .or_else(|| n.strip_prefix("in_s"))
        .or_else(|| n.strip_prefix("in_d"))
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn collect_names(s: &Stmt, defined: &mut HashSet<String>, all: &mut HashSet<String>) {
    match s {
        Stmt::Assign(t, _, _) => {
            if is_ident(t) {
                defined.insert(t.clone());
                all.insert(t.clone());
            } else {
                for tok in t.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                    if is_ident(tok) {
                        all.insert(tok.to_string());
                    }
                }
            }
        }
        Stmt::ForEach { var_name, .. } => {
            defined.insert(var_name.clone());
            all.insert(var_name.clone());
        }
        _ => {}
    }
    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
        fn walk(e: &Expr, all: &mut HashSet<String>) {
            if let Expr::Var(n) = e {
                all.insert(n.clone());
            }
            for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
                walk(c, all);
            }
        }
        walk(e, all);
    });
    let mut kids = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for x in k {
            collect_names(x, defined, all);
        }
    }
}

fn undefined_fp_pseudos(stmts: &[Stmt], defined: &HashSet<String>, out: &mut HashSet<String>) {
    for s in stmts {
        crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
            fn walk(e: &Expr, defined: &HashSet<String>, out: &mut HashSet<String>) {
                if let Expr::Var(n) = e {
                    if is_fp_pseudo(n) && !defined.contains(n) {
                        out.insert(n.clone());
                    }
                }
                for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
                    walk(c, defined, out);
                }
            }
            walk(e, defined, out);
        });
        let mut kids = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            undefined_fp_pseudos(k, defined, out);
        }
    }
}

fn unused_calls(stmts: &mut [Stmt], hits: &mut Vec<String>) {
    for s in stmts.iter_mut() {
        if let Stmt::Eval(Expr::Call { name, .. }) = s {
            let short = name.rsplit('.').next().unwrap_or(name);
            if !short.starts_with("fn_")
                && !short.starts_with("IL2CPP_")
                && !short.starts_with("il2cpp_")
                && short != "Sqrt"
            {
                hits.push(short.to_string());
            }
        }
        let kids: Vec<&mut Vec<Stmt>> = super::children_mut(s);
        for k in kids {
            unused_calls(k, hits);
        }
    }
}

fn replace_pseudo_expr(e: &mut Expr, pseudos: &HashSet<String>, tmp: &str) -> usize {
    let mut n = 0;
    if let Expr::Var(name) = e {
        if pseudos.contains(name) {
            *name = tmp.to_string();
            n += 1;
        }
    }
    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
        n += replace_pseudo_expr(c, pseudos, tmp);
    }
    n
}

fn bind_calls(stmts: &mut [Stmt], pseudos: &HashSet<String>, tmp: &str, bound: &mut bool) -> usize {
    let mut n = 0;
    for s in stmts.iter_mut() {
        if let Stmt::Eval(Expr::Call { .. }) = s {
            let Stmt::Eval(call) = std::mem::replace(s, Stmt::Comment(String::new())) else {
                continue;
            };
            *s = Stmt::Assign(tmp.to_string(), None, call);
            *bound = true;
            n += 1;
        }
        crate::recovery::reconstruct::copy_prop::for_each_expr_mut(s, &mut |e| {
            n += replace_pseudo_expr(e, pseudos, tmp);
        });
        let kids: Vec<&mut Vec<Stmt>> = super::children_mut(s);
        for k in kids {
            n += bind_calls(k, pseudos, tmp, bound);
        }
    }
    n
}

/// Returns the number of rewritten references (1 for the call + one per
/// pseudo read), or 0 when the shape does not qualify.
pub fn apply(stmts: &mut Vec<Stmt>, param_names: &HashSet<String>) -> usize {
    let mut defined = HashSet::new();
    let mut all = HashSet::new();
    for s in stmts.iter() {
        collect_names(s, &mut defined, &mut all);
    }
    let mut pseudos = HashSet::new();
    undefined_fp_pseudos(stmts, &defined, &mut pseudos);
    pseudos.retain(|p| !param_names.contains(p));
    if pseudos.is_empty() {
        return 0;
    }
    let mut calls = Vec::new();
    unused_calls(stmts, &mut calls);
    if calls.len() != 1 {
        return 0;
    }
    let short = &calls[0];
    let base = short.strip_prefix("Get").unwrap_or(short);
    let base = base.strip_prefix("Is").unwrap_or(base);
    let mut tmp: String = match base.chars().next() {
        Some(c) => c.to_lowercase().collect::<String>() + &base[c.len_utf8()..],
        None => "callResult".to_string(),
    };
    if !is_ident(&tmp) {
        tmp = "callResult".to_string();
    }
    if all.contains(&tmp) {
        let mut k = 2;
        loop {
            let c = format!("{tmp}{k}");
            if !all.contains(&c) {
                tmp = c;
                break;
            }
            k += 1;
        }
    }
    let mut bound = false;
    let n = bind_calls(stmts, &pseudos, &tmp, &mut bound);
    if bound {
        n
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }
    fn call(n: &str) -> Expr {
        Expr::Call {
            recv: Some(Box::new(v("this"))),
            name: n.to_string(),
            args: vec![],
        }
    }

    #[test]
    fn a_dropped_float_call_binds_its_register_reads() {
        let mut stmts = vec![
            Stmt::Eval(call("GetMaxModifiedSpeed")),
            Stmt::Assign("a".into(), None, v("in_vs0")),
            Stmt::Assign("b".into(), None, v("in_vd2")),
        ];
        let n = apply(&mut stmts, &HashSet::new());
        assert!(n >= 3, "{stmts:?}");
        let d = format!("{stmts:?}");
        assert!(d.contains("maxModifiedSpeed"), "{d}");
        assert!(!d.contains("in_vs0"), "{d}");
        assert!(!d.contains("in_vd2"), "{d}");
    }

    #[test]
    fn two_dropped_calls_refuse() {
        let mut stmts = vec![
            Stmt::Eval(call("GetA")),
            Stmt::Eval(call("GetB")),
            Stmt::Assign("a".into(), None, v("in_vs0")),
        ];
        assert_eq!(apply(&mut stmts, &HashSet::new()), 0);
    }

    #[test]
    fn a_defined_pseudo_refuses() {
        let mut stmts = vec![
            Stmt::Assign("in_vs0".into(), None, Expr::Float(1.0)),
            Stmt::Eval(call("GetMaxModifiedSpeed")),
            Stmt::Assign("a".into(), None, v("in_vs0")),
        ];
        assert_eq!(apply(&mut stmts, &HashSet::new()), 0);
    }
}
