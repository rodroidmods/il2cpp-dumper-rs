//! Render provable `this` aliases as `this`.
//!
//! The compiler keeps `this` in a callee-saved register across calls
//! (`mov x27, x0`), so the recovered body receives it as an entry value of that
//! register and prints `in_x27._AllPossibleCharacterTokens` — an ABI name where
//! the source says `this._AllPossibleCharacterTokens`.
//!
//! An `in_xNN` local whose proven type is EXACTLY the declaring type is this
//! method's receiver, provided the method declares no parameter of that type
//! (otherwise the value could be that parameter). The rewrite is a pure
//! substitution of a proven receiver, and addresses are left alone (`&in_x27`
//! must not become `&this`).

use super::super::csharp::{Expr, Stmt};
use super::super::resolve::ResolveMaps;
use super::children_mut;
use std::collections::{HashMap, HashSet};

fn is_entry_reg(n: &str) -> bool {
    n.strip_prefix("in_x")
        .is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
}

fn token_in(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let mut from = 0;
    while let Some(pos) = text[from..].find(name) {
        let at = from + pos;
        let before_ok =
            at == 0 || !(bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_');
        let end = at + name.len();
        let after_ok =
            end == bytes.len() || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

fn replace_token(text: &str, old: &str, new: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if text[i..].starts_with(old) {
            let before_ok =
                i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let end = i + old.len();
            let after_ok =
                end == bytes.len() || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
            if before_ok && after_ok {
                out.push_str(new);
                i = end;
                continue;
            }
        }
        let c = text[i..].chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    out
}

fn rewrite_expr(e: &mut Expr, aliases: &HashSet<String>, n: &mut usize) {
    match e {
        Expr::Field(base, _) => {
            if let Expr::Var(name) = base.as_ref() {
                if aliases.contains(name) {
                    **base = Expr::This;
                    *n += 1;
                }
            }
        }
        Expr::Call { recv: Some(r), .. } => {
            if let Expr::Var(name) = r.as_ref() {
                if aliases.contains(name) {
                    **r = Expr::This;
                    *n += 1;
                }
            }
        }
        _ => {}
    }
    if let Expr::Lambda { .. } = e {
        // A lambda body captures by name; leave it to later passes.
        return;
    }
    for c in expr_children_mut(e) {
        rewrite_expr(c, aliases, n);
    }
}

fn expr_children_mut(e: &mut Expr) -> Vec<&mut Expr> {
    match e {
        Expr::Field(b, _)
        | Expr::RawDeref(b, _, _)
        | Expr::RawAddr(b, _)
        | Expr::AddressOf(b)
        | Expr::Cast(_, b)
        | Expr::Not(b)
        | Expr::Neg(b)
        | Expr::Await(b)
        | Expr::IndexFromEnd(b) => vec![b],
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => vec![a, b],
        Expr::Cond(c, a, b) => vec![c, a, b],
        Expr::Index { collection, index } => vec![collection, index],
        Expr::NullConditional { receiver, access } => vec![receiver, access],
        Expr::Call { recv, args, .. } => {
            let mut v: Vec<&mut Expr> = args.iter_mut().collect();
            if let Some(r) = recv {
                v.push(r);
            }
            v
        }
        Expr::RawCall(_, args, _)
        | Expr::NewObj(_, args)
        | Expr::NewArr(_, args)
        | Expr::ArrayInit { items: args, .. } => args.iter_mut().collect(),
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            let mut v: Vec<&mut Expr> = ctor_args.iter_mut().collect();
            v.extend(items.iter_mut());
            v
        }
        Expr::ObjInit {
            ctor_args,
            fields,
            ..
        } => {
            let mut v: Vec<&mut Expr> = ctor_args.iter_mut().collect();
            v.extend(fields.iter_mut().map(|(_, x)| x));
            v
        }
        Expr::PatternMatch { expr, .. } => vec![expr],
        Expr::StringInterp { parts } => {
            let mut v: Vec<&mut Expr> = Vec::new();
            for p in parts {
                if let super::super::csharp::InterpPart::Expr(x, _) = p {
                    v.push(x.as_mut());
                }
            }
            v
        }
        Expr::Tuple { elements, .. } => elements.iter_mut().collect(),
        Expr::MethodChain { receiver, methods } => {
            let mut v: Vec<&mut Expr> = vec![receiver];
            for m in methods.iter_mut() {
                v.extend(m.args.iter_mut());
            }
            v
        }
        Expr::SwitchExpr {
            discrim,
            arms,
            default_arm,
        } => {
            let mut v: Vec<&mut Expr> = vec![discrim];
            for (pat, res) in arms.iter_mut() {
                v.push(pat);
                v.push(res);
            }
            if let Some(d) = default_arm {
                v.push(d);
            }
            v
        }
        Expr::Range { from, to } => {
            let mut v: Vec<&mut Expr> = Vec::new();
            if let Some(f) = from {
                v.push(f.as_mut());
            }
            if let Some(t) = to {
                v.push(t.as_mut());
            }
            v
        }
        Expr::Lambda { body, .. } => vec![body],
        _ => vec![],
    }
}

fn rewrite_stmt(s: &mut Stmt, aliases: &HashSet<String>, n: &mut usize) {
    for k in children_mut(s) {
        for x in k.iter_mut() {
            rewrite_stmt(x, aliases, n);
        }
    }
    if let Stmt::Assign(t, _, _) = s {
        for a in aliases {
            if token_in(t, a) {
                *t = replace_token(t, a, "this");
                *n += 1;
            }
        }
    }
    let mut fold = |e: &mut Expr| rewrite_expr(e, aliases, n);
    match s {
        Stmt::Eval(e)
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e)
        | Stmt::Return(Some(e))
        | Stmt::Assign(_, _, e) => fold(e),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => fold(cond),
        Stmt::For { cond, .. } => {
            if let Some(c) = cond {
                fold(c);
            }
        }
        Stmt::Lock { expr, .. } => fold(expr),
        Stmt::StringSwitch { disc, .. } => fold(disc),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            fold(collection);
            fold(index);
            fold(value);
        }
        Stmt::EventAssign { field, value, .. } => {
            fold(field);
            fold(value);
        }
        _ => {}
    }
}

/// Run the pass. Returns the number of rewritten receivers/aliases.
pub fn apply(
    stmts: &mut Vec<Stmt>,
    _maps: &ResolveMaps,
    local_types: &HashMap<String, i32>,
    declaring_tdi: i32,
    param_names: &HashSet<String>,
) -> usize {
    if declaring_tdi < 0 {
        return 0;
    }
    // If a PARAMETER is typed as the declaring class, an entry register of that
    // type could be that parameter rather than `this`.
    let param_same_type = param_names.iter().any(|p| {
        local_types
            .get(p)
            .is_some_and(|&tdi| tdi == declaring_tdi)
    });
    if param_same_type {
        return 0;
    }
    // An addressed alias cannot become `this` (`&this` does not exist).
    let mut addressed: HashSet<String> = HashSet::new();
    fn scan_addr(stmts: &[Stmt], out: &mut HashSet<String>) {
        for s in stmts {
            if let Stmt::Assign(t, _, _) = s {
                if let Some(pos) = t.find('&') {
                    let rest = &t[pos + 1..];
                    let end = rest
                        .char_indices()
                        .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_'))
                        .map(|(i, _)| i)
                        .unwrap_or(rest.len());
                    out.insert(rest[..end].to_string());
                }
            }
            let mut kids: Vec<&Vec<Stmt>> = Vec::new();
            super::children(s, &mut kids);
            for k in kids {
                scan_addr(k, out);
            }
        }
    }
    scan_addr(stmts, &mut addressed);

    let aliases: HashSet<String> = local_types
        .iter()
        .filter(|(name, tdi)| **tdi == declaring_tdi && is_entry_reg(name))
        .map(|(name, _)| name.clone())
        .filter(|name| !addressed.contains(name))
        .collect();
    if aliases.is_empty() {
        return 0;
    }
    if std::env::var("ILR_THISDBG").is_ok() {
        eprintln!("[this] aliases={aliases:?}");
    }
    let mut n = 0;
    for s in stmts.iter_mut() {
        rewrite_stmt(s, &aliases, &mut n);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps() -> ResolveMaps {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(302, ("".into(), "TokenManager".into()));
        m
    }

    #[test]
    fn field_receiver_becomes_this() {
        let mut lt = HashMap::new();
        lt.insert("in_x27".to_string(), 302);
        let mut stmts = vec![Stmt::Assign(
            "in_x27._Total".into(),
            None,
            Expr::Field(Box::new(Expr::Var("in_x27".into())), "_Total".into()),
        )];
        assert_eq!(apply(&mut stmts, &maps(), &lt, 302, &HashSet::new()), 2);
        let t = format!("{stmts:?}");
        assert!(!t.contains("in_x27"), "{t}");
        assert!(t.contains("Assign(\"this._Total\""), "{t}");
    }

    #[test]
    fn a_same_type_parameter_blocks_the_rewrite() {
        let mut lt = HashMap::new();
        lt.insert("in_x27".to_string(), 302);
        lt.insert("other".to_string(), 302);
        let params: HashSet<String> = ["other".to_string()].into_iter().collect();
        let mut stmts = vec![Stmt::Eval(Expr::Field(
            Box::new(Expr::Var("in_x27".into())),
            "_Total".into(),
        ))];
        assert_eq!(apply(&mut stmts, &maps(), &lt, 302, &params), 0);
    }

    #[test]
    fn an_addressed_alias_stays_a_register() {
        let mut lt = HashMap::new();
        lt.insert("in_x27".to_string(), 302);
        let mut stmts = vec![
            Stmt::Eval(Expr::Field(
                Box::new(Expr::Var("in_x27".into())),
                "_Total".into(),
            )),
            Stmt::Assign("*(int*)((nint)&in_x27 +4)".into(), None, Expr::Int(1)),
        ];
        assert_eq!(apply(&mut stmts, &maps(), &lt, 302, &HashSet::new()), 0);
    }
}
