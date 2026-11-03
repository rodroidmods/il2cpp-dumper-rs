//! Fold a delegate tail-invocation into a normal delegate call.
//!
//! A delegate invoked through its runtime fields reaches the renderer as
//!
//! ```text
//! fnptr_0x…(del.method_code, userArgs…, del.method, del.invoke_impl)
//! ```
//!
//! (the exact field set varies with the IL2CPP version). The call target is
//! the delegate's `invoke_impl`; the hidden target/method loads belong to the
//! runtime calling convention, not to the source. C# writes `del(userArgs)`.
//!
//! Detection is textual on the ALREADY-RESOLVED member names: at least two
//! arguments are field loads off the SAME rendered base expression and one of
//! them is `method_code`/`invoke_impl` — the delegate ABI. A false positive
//! would need a non-delegate object exposing `method_code`/`invoke_impl` in a
//! raw `fnptr_` call, which the runtime never writes.

use super::super::csharp::{Expr, Stmt};
use std::collections::HashSet;

const HIDDEN: &[&str] = &[
    "method_code",
    "invoke_impl",
    "method_ptr",
    "method",
    "target",
];

pub(crate) fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    super::walk_mut(stmts, &mut |list| {
        for s in list.iter_mut() {
            super::copy_prop::for_each_expr_mut(s, &mut |e| {
                n += rewrite(e);
            });
        }
    });
    n
}

fn render(e: &Expr) -> String {
    let mut s = String::new();
    e.render(&mut s);
    s
}

fn rewrite(e: &mut Expr) -> usize {
    let mut n = 0;
    for c in super::il2cpp_cleanup::expr_children_mut(e) {
        n += rewrite(c);
    }
    let Expr::Call {
        recv: None,
        name,
        args,
    } = e
    else {
        return n;
    };
    if !name.starts_with("fnptr_") || args.len() < 3 {
        return n;
    }
    // Base candidate: the text before the last '.' of every dotted argument.
    let mut bases: Vec<(String, String)> = Vec::new();
    for a in args.iter() {
        let text = render(a);
        if let Some((base, member)) = text.rsplit_once('.') {
            if HIDDEN.contains(&member) {
                bases.push((base.to_string(), member.to_string()));
            }
        }
    }
    let Some((base, _)) = bases.first().cloned() else {
        return n;
    };
    // At least two hidden loads off the SAME base, one of them method_code or
    // invoke_impl (the delegate calling convention, not a plain struct).
    let hidden: HashSet<String> = bases
        .iter()
        .filter(|(b, _)| b == &base)
        .map(|(_, m)| m.clone())
        .collect();
    if hidden.len() < 2 || !(hidden.contains("method_code") || hidden.contains("invoke_impl")) {
        return n;
    }
    // Keep the user arguments: everything that is not a load off the delegate
    // base (the hidden ABI fields, including a misnamed one such as `.Length`).
    let prefix = format!("{base}.");
    let kept: Vec<Expr> = args
        .iter()
        .filter(|a| {
            let t = render(a);
            t != base && !t.starts_with(&prefix)
        })
        .cloned()
        .collect();
    *e = Expr::Call {
        recv: None,
        name: base,
        args: kept,
    };
    n + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(base: Expr, name: &str) -> Expr {
        Expr::Field(Box::new(base), name.into())
    }

    fn static_delegate() -> Expr {
        Expr::StaticField("UnityEngine.Camera".into(), "onPreCull".into())
    }

    #[test]
    fn a_delegate_tail_call_becomes_a_delegate_call() {
        let call = Expr::Call {
            recv: None,
            name: "fnptr_0x3c8153c".into(),
            args: vec![
                field(static_delegate(), "method_code"),
                Expr::Var("cam".into()),
                field(static_delegate(), "method"),
                field(static_delegate(), "Length"),
            ],
        };
        let mut stmts = vec![Stmt::Eval(call)];
        assert_eq!(apply(&mut stmts), 1);
        let Stmt::Eval(Expr::Call { name, args, .. }) = &stmts[0] else {
            panic!("{stmts:#?}");
        };
        assert_eq!(name, "UnityEngine.Camera.onPreCull");
        assert_eq!(args.len(), 1);
        assert!(matches!(&args[0], Expr::Var(v) if v == "cam"));
    }

    #[test]
    fn a_plain_struct_field_call_is_not_a_delegate() {
        let base = Expr::Var("x".into());
        let call = Expr::Call {
            recv: None,
            name: "fnptr_0x1234".into(),
            args: vec![field(base.clone(), "a"), field(base, "b"), Expr::Var("v".into())],
        };
        let mut stmts = vec![Stmt::Eval(call)];
        assert_eq!(apply(&mut stmts), 0);
    }
}
