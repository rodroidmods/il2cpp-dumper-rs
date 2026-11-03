//! Late, layout-grounded field fixups that the general annotators cannot see.
//!
//! 1. COMPONENT-CHAIN STORES. IL2CPP writes a struct field component by
//!    component (`str s0, [x0,#0x1C8]`). The IR keeps the first store as a
//!    whole-field `Move Field(PlayerVelocity)` (width lost), followed by
//!    `+4/+8` stores. Emitting `this.PlayerVelocity = <float>;` is not valid
//!    C# and hides that the first store is `.x`. When the NEXT statement
//!    writes `field.y`/`field.z` of the same UnityEngine Vector2/Vector3
//!    field, the current store is rewritten to `field.x`.
//!
//! 2. FLOAT-TYPED ABSOLUTE SLOTS. A 4-byte deref of an absolute address
//!    compared against a float expression is a single-precision load; the
//!    renderer prints width 4 as `int`. Retype only that comparison operand to
//!    width 5 (`*(float*)`). Local retyping is `float_locals`' job; this is
//!    the inline-deref case it cannot reach.

use crate::recovery::csharp::{CmpKind, Expr, Stmt};
use crate::recovery::resolve::ResolveMaps;

pub fn apply(stmts: &mut Vec<Stmt>, maps: &ResolveMaps, declaring_tdi: i32) -> usize {
    let mut n = 0;
    fix_whole_field_components(stmts, maps, declaring_tdi, &mut n);
    componentize_vector_operands(stmts, maps, declaring_tdi, &mut n);
    static_vector_components(stmts, maps, &mut n);
    absolute_static_fields(stmts, maps, &mut n);
    if !crate::recovery::env_flags::flag("ILR_NOA4") {
        drop_class_init_guards(stmts, &mut n);
    }
    name_scalar_factors(stmts, &mut n);
    retype_float_comparison_derefs(stmts, &mut n);
    n
}

/// Type name tested by the klass initialized flag (`+224`).
/// A helper call whose arguments include `typeof(T)` — the class-init helper
/// under its unresolved `RawCall` form (`fn 0x1302514(typeof(Math))`).
fn helper_takes_type(e: &Expr, ty: &str) -> bool {
    let args = match e {
        Expr::Call { args, .. } => args.as_slice(),
        Expr::RawCall(_, args, _) => args.as_slice(),
        _ => return false,
    };
    args.iter().any(|a| matches!(a, Expr::TypeOf(t) if t == ty))
}

/// Name an absolute static slot whose type was never recorded as a metadata
/// usage, by inferring its static-block BASE from the type's OTHER usage
/// slots. `static_field_layouts` gives absolute slot addresses with
/// (field, declaring type); matching that field against the type's computed
/// static offsets gives `base = addr - offset`. A base confirmed by >= 2
/// slots identifies the block, so `base+644` is `(type, 644)`.
fn absolute_static_fields(stmts: &mut Vec<Stmt>, maps: &ResolveMaps, n: &mut usize) {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static BASES: OnceLock<HashMap<u64, String>> = OnceLock::new();
    let bases = BASES.get_or_init(|| {
        let mut off_by_name: HashMap<(String, String), i64> = HashMap::new();
        for ((ty, off), (name, _)) in maps.static_field_offsets.iter() {
            off_by_name.insert((ty.clone(), name.clone()), *off);
        }
        let mut tally: HashMap<(String, u64), usize> = HashMap::new();
        for (addr, (name, ty)) in maps.static_field_layouts.iter() {
            let Some(off) = off_by_name.get(&(ty.clone(), name.clone())).copied() else {
                continue;
            };
            if off < 0 || (*addr as i64) < off {
                continue;
            }
            *tally.entry((ty.clone(), *addr - off as u64)).or_default() += 1;
        }
        let mut best: HashMap<u64, (usize, String)> = HashMap::new();
        for ((ty, base), count) in tally {
            if count < 2 {
                continue;
            }
            match best.get(&base) {
                Some((c, _)) if *c >= count => {}
                _ => {
                    best.insert(base, (count, ty));
                }
            }
        }
        best.into_iter().map(|(b, (_, ty))| (b, ty)).collect()
    });
    fn visit(e: &mut Expr, maps: &ResolveMaps, bases: &HashMap<u64, String>, n: &mut usize) {
        if let Expr::RawDeref(base, off, w) = e {
            if (*w == 4 || *w == 5) && *off > 0 {
                if let Expr::Int(b) = base.as_ref() {
                    if let Some(ty) = bases.get(&(*b as u64)) {
                        if let Some((name, _)) = maps.static_field_offsets.get(&(ty.clone(), *off)) {
                            *e = Expr::StaticField(ty.clone(), name.clone());
                            *n += 1;
                            return;
                        }
                    }
                }
            }
        }
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
            visit(c, maps, bases, n);
        }
    }
    for s in stmts.iter_mut() {
        crate::recovery::reconstruct::copy_prop::for_each_expr_mut(s, &mut |e| visit(e, maps, bases, n));
    }
}

fn expr_is_class_init(e: &Expr) -> bool {
    match e {
        Expr::Call { name, .. } => {
            name.contains("il2cpp_runtime_class_init") || name.contains("IL2CPP_ClassInit")
        }
        Expr::RawCall(addr, _, _) => {
            crate::recovery::runtime_helpers::role_of(*addr)
                == Some(crate::recovery::runtime_helpers::RuntimeHelper::ClassInit)
        }
        _ => false,
    }
}

/// A late class-init guard: `if (klassFlag == 0) { ptr = il2cpp_runtime_class_init(typeof(T)); }`.
/// The early cleanup only sees the raw helper address; by the time the call
/// renders by name the guard is already past it (VerifySpeed).
///
/// Second accepted shape, and the one the late tree actually reaches: the
/// "init" result was already folded into the type token, so the body only
/// RE-ASSIGNS the klass (`var ptr = type; if (*(nint*)(type+224)==0) ptr =
/// typeof(T);`). The assignment is the identity — the value is the token the
/// variable already holds — so the whole `if` is removable. This needs a
/// small forward flow: the tested variable is only "already holding T" when
/// the nearest assignment in the same statement list assigned that token.
fn drop_class_init_guards(stmts: &mut Vec<Stmt>, n: &mut usize) {
    use std::collections::HashMap;

    fn klass_tok(e: &Expr) -> Option<String> {
        match e {
            Expr::TypeOf(t) => Some(t.clone()),
            Expr::StaticField(t, a)
                if a.starts_with("klass(") || a.starts_with("typeof(") =>
            {
                Some(t.clone())
            }
            _ => None,
        }
    }

    /// `*(base + 224) ==/!= 0` → (base, offset).
    fn init_flag_test(cond: &Expr) -> Option<(&Expr, i64)> {
        let Expr::Cmp(kind, a, b) = cond else {
            return None;
        };
        if !matches!(kind, CmpKind::Eq | CmpKind::Ne) {
            return None;
        }
        for (x, y) in [(a, b), (b, a)] {
            if matches!(y.as_ref(), Expr::Int(0)) {
                if let Expr::RawDeref(base, off, _) = x.as_ref() {
                    return Some((base.as_ref(), *off));
                }
            }
        }
        None
    }

    fn scan_list(
        list: &mut Vec<Stmt>,
        tokens: &mut HashMap<String, String>,
        n: &mut usize,
    ) {
        let mut i = 0;
        while i < list.len() {
            let mut drop = false;
            if let Stmt::If {
                cond,
                then_body,
                else_body,
            } = &list[i]
            {
                if else_body.is_empty() && !then_body.is_empty() {
                    if let Some((base, off)) = init_flag_test(cond) {
                        if off == 224 {
                            let guard_ty = match base {
                                Expr::TypeOf(t) => Some(t.clone()),
                                Expr::Var(v) => tokens.get(v).cloned(),
                                _ => None,
                            };
                            if let Some(ty) = guard_ty {
                                let mut exec = 0usize;
                                let mut ok = 0usize;
                                for s in then_body {
                                    match s {
                                        Stmt::Comment(_) | Stmt::Attribute { .. } => {}
                                        Stmt::Eval(e) => {
                                            exec += 1;
                                            if expr_is_class_init(e)
                                                || helper_takes_type(e, &ty)
                                            {
                                                ok += 1;
                                            }
                                        }
                                        Stmt::Assign(name, _, e) => {
                                            exec += 1;
                                            // The body writes the SAME token the
                                            // variable already holds, and the
                                            // condition only READ the flag: the
                                            // arm is the identity, whatever
                                            // variable it names.
                                            let identity =
                                                klass_tok(e).as_deref() == Some(ty.as_str())
                                                    && tokens
                                                        .get(name)
                                                        .is_some_and(|t| *t == ty);
                                            if identity
                                                || expr_is_class_init(e)
                                                || helper_takes_type(e, &ty)
                                            {
                                                ok += 1;
                                            }
                                        }
                                        _ => exec += 2, // anything else is real code
                                    }
                                }
                                drop = exec == ok && ok > 0;
                            }
                        }
                    }
                }
            }
            if drop {
                list.remove(i);
                *n += 1;
                continue;
            }
            if let Stmt::Assign(name, _, rhs) = &list[i] {
                match rhs {
                    // A copy of a token-bearing variable carries the token.
                    Expr::Var(src) => match tokens.get(src) {
                        Some(t) => {
                            let t = t.clone();
                            tokens.insert(name.clone(), t);
                        }
                        None => {
                            tokens.remove(name);
                        }
                    },
                    _ => match klass_tok(rhs) {
                        Some(t) => {
                            tokens.insert(name.clone(), t);
                        }
                        None => {
                            tokens.remove(name);
                        }
                    },
                }
            }
            // Child lists get a COPY of the flow: an assignment made inside an
            // arm is not known to reach the statements after the arm, so it
            // must not license an identity drop there.
            let mut kids: Vec<&mut Vec<Stmt>> = super::children_mut(&mut list[i]);
            for k in kids.drain(..) {
                let mut child_tokens = tokens.clone();
                scan_list(k, &mut child_tokens, n);
            }
            i += 1;
        }
    }

    let mut tokens = HashMap::new();
    scan_list(stmts, &mut tokens, n);
}

/// Name a generated temp that exists only as a multiplication factor
/// (`v40` in VerifySpeed: `maxSpeed * v40` in both arms) `scale`. Requires
/// >= 2 definitions (a phi-like temp) and EVERY read to be an operand of a
/// Mul, so ordinary products that feed other expressions are untouched.
fn name_scalar_factors(stmts: &mut Vec<Stmt>, n: &mut usize) {
    use std::collections::HashMap;
    let mut defs: HashMap<String, usize> = HashMap::new();
    let mut reads: HashMap<String, usize> = HashMap::new();
    let mut mul_reads: HashMap<String, usize> = HashMap::new();
    let mut others: std::collections::HashSet<String> = std::collections::HashSet::new();
    fn expr_reads(e: &Expr, name: &str, total: &mut usize, mul: &mut usize) {
        if matches!(e, Expr::Var(v) if v == name) {
            *total += 1;
        }
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
            expr_reads(c, name, total, mul);
        }
    }
    fn scan(
        stmts: &[Stmt],
        defs: &mut HashMap<String, usize>,
        reads: &mut HashMap<String, usize>,
        mul_reads: &mut HashMap<String, usize>,
        others: &mut std::collections::HashSet<String>,
    ) {
        for s in stmts {
            if let Stmt::Assign(t, _, _) = s {
                if t.starts_with('v') && t[1..].bytes().all(|b| b.is_ascii_digit()) {
                    *defs.entry(t.clone()).or_default() += 1;
                } else {
                    others.insert(t.clone());
                }
            }
            crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
                if let Expr::Bin(crate::recovery::csharp::BinOp::Mul, a, b) = e {
                    for (x, y) in [(a, b), (b, a)] {
                        if let Expr::Var(v) = x.as_ref() {
                            if v.starts_with('v') && v[1..].bytes().all(|c| c.is_ascii_digit()) {
                                *mul_reads.entry(v.clone()).or_default() += 1;
                            }
                        }
                        let _ = y;
                    }
                }
                if let Expr::Var(v) = e {
                    let _ = v;
                }
            });
            // total reads per generated name
            let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
            crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
                fn walk(e: &Expr, out: &mut std::collections::HashSet<String>) {
                    if let Expr::Var(v) = e {
                        out.insert(v.clone());
                    }
                    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e) {
                        walk(c, out);
                    }
                }
                walk(e, &mut names);
            });
            for nm in names {
                if nm.starts_with('v') && nm[1..].bytes().all(|c| c.is_ascii_digit()) {
                    let mut t = 0usize;
                    let mut m = 0usize;
                    crate::recovery::reconstruct::float_locals::for_each_expr(s, &mut |e| {
                        expr_reads(e, &nm, &mut t, &mut m);
                    });
                    *reads.entry(nm).or_default() += t;
                }
            }
            let mut kids = Vec::new();
            super::children(s, &mut kids);
            for k in kids {
                scan(k, defs, reads, mul_reads, others);
            }
        }
    }
    scan(stmts, &mut defs, &mut reads, &mut mul_reads, &mut others);
    let mut rename_to: Option<String> = None;
    for (name, d) in &defs {
        if *d < 2 || others.contains(name) || others.contains("scale") {
            continue;
        }
        let total = reads.get(name).copied().unwrap_or(0);
        let mul = mul_reads.get(name).copied().unwrap_or(0);
        if total > 0 && mul == total {
            rename_to = Some(name.clone());
            break;
        }
    }
    let Some(from) = rename_to else { return };
    let mut target = "scale".to_string();
    if others.contains(&target) || defs.contains_key(&target) {
        let mut k = 2;
        loop {
            let c = format!("scale{k}");
            if !others.contains(&c) && !defs.contains_key(&c) {
                target = c;
                break;
            }
            k += 1;
        }
    }
    fn ren(stmts: &mut [Stmt], from: &str, to: &str) {
        for s in stmts.iter_mut() {
            match s {
                Stmt::Assign(t, _, _) if t == from => *t = to.to_string(),
                _ => {}
            }
            crate::recovery::reconstruct::copy_prop::for_each_expr_mut(s, &mut |e| {
                fn w(e: &mut Expr, from: &str, to: &str) {
                    if let Expr::Var(n) = e {
                        if n == from {
                            *n = to.to_string();
                        }
                    }
                    for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
                        w(c, from, to);
                    }
                }
                w(e, from, to);
            });
            for k in super::children_mut(s) {
                ren(k, from, to);
            }
        }
    }
    ren(stmts, &from, &target);
    *n += 1;
}

/// A component read of a static vector field: `*(TypeOf(T)+8)` inside the
/// first static Vector3 (e.g. `Vector3.zero.z`). This runs AFTER the sugar
/// passes, where the klass token has become `TypeOf(T)`; `field_annotation`
/// runs earlier and sees the raw klass chain.
fn static_vector_components(stmts: &mut Vec<Stmt>, maps: &ResolveMaps, n: &mut usize) {
    fn component(maps: &ResolveMaps, ty: &str, off: i64) -> Option<String> {
        // Unity's Vector3 static block, proven by this metadata's own offsets:
        // zero at +0 (a 12-byte Vector3; a colliding float alias is what
        // `static_field_offsets` happens to name there), one +12, up +24, …
        // The public source names are `Vector3.zero`, `.one`, … — a read of
        // +8 bytes is `zero.z`. This is the constant VerifySpeed multiplies
        // (`*(int*)(typeof(Vector3)+8)`).
        if ty == "UnityEngine.Vector3" {
            const NAMES: [&str; 8] = [
                "zero", "one", "up", "down", "left", "right", "forward", "back",
            ];
            for (i, name) in NAMES.iter().enumerate() {
                let base = i as i64 * 12;
                if off >= base && off < base + 12 && (off - base) % 4 == 0 {
                    let comp = ["x", "y", "z"][((off - base) / 4) as usize];
                    return Some(format!("{name}.{comp}"));
                }
            }
        }
        if crate::recovery::env_flags::flag("ILR_FIELDFIX") && ty.contains("Vector3") {
            let mut rows: Vec<(i64, String, Option<i32>)> = maps
                .static_field_offsets
                .iter()
                .filter(|((t, _), _)| t == ty)
                .map(|((_, o), (n, _))| {
                    (*o, n.clone(), maps.static_field_types.get(&(ty.to_string(), n.clone())).copied())
                })
                .collect();
            rows.sort();
            for (o, n, t) in rows.iter().take(20) {
                eprintln!("[fx-layout] {ty} +{o} = {n} tdi={t:?}");
            }
        }
        for k in 1..=2i64 {
            let base_off = off - k * 4;
            let Some((name, _)) = maps.static_field_offsets.get(&(ty.to_string(), base_off)) else {
                continue;
            };
            if crate::recovery::env_flags::flag("ILR_FIELDFIX") {
                eprintln!("[fx-comp] ty={ty} base_off={base_off} field={name} ftdi={:?} has_type={}",
                    maps.static_field_types.get(&(ty.to_string(), name.clone())),
                    maps.static_field_types.contains_key(&(ty.to_string(), name.clone())));
            }
            let Some(&ftdi) = maps.static_field_types.get(&(ty.to_string(), name.clone())) else {
                continue;
            };
            let Some((ns, tname)) = maps.tdi_names.get(&ftdi) else {
                continue;
            };
            if ns != "UnityEngine" {
                continue;
            }
            let comp = match (tname.as_str(), k) {
                ("Vector3", 1) => "y",
                ("Vector3", 2) => "z",
                ("Vector2", 1) => "y",
                _ => continue,
            };
            return Some(format!("{name}.{comp}"));
        }
        None
    }
    fn visit(e: &mut Expr, maps: &ResolveMaps, n: &mut usize) {
        if let Expr::RawDeref(base, off, w) = e {
            if crate::recovery::env_flags::flag("ILR_FIELDFIX") {
                if let Expr::TypeOf(ty) = base.as_ref() {
                    eprintln!("[fx-static] ty={ty} off={off} w={w} has0={} has4={} n={}",
                        maps.static_field_offsets.contains_key(&(ty.clone(), 0)),
                        maps.static_field_offsets.contains_key(&(ty.clone(), 4)),
                        maps.static_field_offsets.len());
                }
            }
            if *w == 4 {
                if let Expr::TypeOf(ty) = base.as_ref() {
                    if let Some(field) = component(maps, ty, *off) {
                        // Vector components are float; the Cast makes the
                        // type unambiguous for the locals inference (scale
                        // was `int` because the component read had no type).
                        *e = Expr::Cast(
                            "float".to_string(),
                            Box::new(Expr::StaticField(ty.clone(), field)),
                        );
                        *n += 1;
                        return;
                    }
                }
            }
        }
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
            visit(c, maps, n);
        }
    }
    for s in stmts.iter_mut() {
        crate::recovery::reconstruct::copy_prop::for_each_expr_mut(s, &mut |e| visit(e, maps, n));
        for k in super::children_mut(s) {
            static_vector_components(k, maps, n);
        }
    }
}

/// A UnityEngine vector FIELD used directly in float arithmetic is its `.x`
/// component: a whole `Vector3` has no `*`/`+`/`<` operator, so `v * v` can
/// only mean `v.x * v.x`. Rewrites `Field(this, Name)` operands of Bin/Cmp/
/// Neg into `Field(Field(this, Name), "x")`.
fn componentize_vector_operands(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    declaring_tdi: i32,
    n: &mut usize,
) {
    fn scalar_operand(e: &mut Expr, maps: &ResolveMaps, tdi: i32, n: &mut usize) {
        // A field chain is inspected ONLY at its head: descending would wrap
        // the receiver of `PlayerVelocity.x` again and print `.x.x.x.x`
        // (measured once on VerifySpeed).
        if let Expr::Field(recv, name) = e {
            if matches!(recv.as_ref(), Expr::This) {
                if let Some(&ftdi) = maps.field_type_by_name.get(&(tdi, name.clone())) {
                    if is_unity_vector(maps, ftdi) {
                        *e = Expr::Field(Box::new(e.clone()), "x".to_string());
                        *n += 1;
                    }
                }
            }
            return;
        }
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
            scalar_operand(c, maps, tdi, n);
        }
    }
    fn visit(e: &mut Expr, maps: &ResolveMaps, tdi: i32, n: &mut usize) {
        match e {
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
                scalar_operand(a, maps, tdi, n);
                scalar_operand(b, maps, tdi, n);
            }
            Expr::Neg(a) => scalar_operand(a, maps, tdi, n),
            _ => {}
        }
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
            visit(c, maps, tdi, n);
        }
    }
    for s in stmts.iter_mut() {
        crate::recovery::reconstruct::copy_prop::for_each_expr_mut(s, &mut |e| visit(e, maps, declaring_tdi, n));
        for k in super::children_mut(s) {
            componentize_vector_operands(k, maps, declaring_tdi, n);
        }
    }
}

fn is_unity_vector(maps: &ResolveMaps, tdi: i32) -> bool {
    maps.tdi_names
        .get(&tdi)
        .is_some_and(|(ns, name)| ns == "UnityEngine" && (name == "Vector3" || name == "Vector2"))
}

/// `this.Name` whose field on the declaring type is a UnityEngine vector.
fn vector_field_name(maps: &ResolveMaps, declaring_tdi: i32, target: &str) -> Option<String> {
    let name = target.strip_prefix("this.")?;
    if name.contains('.') || name.is_empty() {
        return None;
    }
    let ftdi = *maps
        .field_type_by_name
        .get(&(declaring_tdi, name.to_string()))?;
    is_unity_vector(maps, ftdi).then(|| name.to_string())
}

fn fix_whole_field_components(
    stmts: &mut Vec<Stmt>,
    maps: &ResolveMaps,
    declaring_tdi: i32,
    n: &mut usize,
) {
    for i in 0..stmts.len() {
        // Peek at the next EXECUTABLE statement (comments/attributes are not
        // emitted lines of the chain).
        let mut j = i + 1;
        while j < stmts.len()
            && matches!(stmts[j], Stmt::Comment(_) | Stmt::Attribute { .. })
        {
            j += 1;
        }
        let Some(next_target) = stmts.get(j).and_then(|s| match s {
            Stmt::Assign(t, _, _) => Some(t.as_str()),
            _ => None,
        }) else {
            continue;
        };
        let Stmt::Assign(target, _, _) = &stmts[i] else {
            continue;
        };
        let Some(field) = vector_field_name(maps, declaring_tdi, target) else {
            continue;
        };
        let x = format!("this.{field}.x");
        let y = format!("this.{field}.y");
        let z = format!("this.{field}.z");
        if next_target == z || next_target == y {
            let is_x_store = matches!(stmts[i], Stmt::Assign(ref t, _, _) if *t == format!("this.{field}"));
            if is_x_store {
                if let Stmt::Assign(t, _, _) = &mut stmts[i] {
                    *t = x;
                    *n += 1;
                }
            }
        }
    }
    for s in stmts.iter_mut() {
        for k in super::children_mut(s) {
            fix_whole_field_components(k, maps, declaring_tdi, n);
        }
    }
}

fn floatish(e: &Expr) -> bool {
    match e {
        Expr::Float(_) | Expr::Double(_) => true,
        Expr::Cast(t, _) => matches!(t.as_str(), "float" | "double" | "Single" | "Double"),
        Expr::Call { name, .. } => {
            let short = name.rsplit('.').next().unwrap_or(name);
            short == "Sqrt" || short == "Sqrtf" || short == "Mathf.Sqrt"
        }
        Expr::Bin(_, a, b) => floatish(a) || floatish(b),
        Expr::Neg(x) => floatish(x),
        _ => false,
    }
}

fn retype_float_comparison_derefs(stmts: &mut Vec<Stmt>, n: &mut usize) {
    fn visit_expr(e: &mut Expr, n: &mut usize) {
        if let Expr::Cmp(_, a, b) = e {
            let a_float = floatish(a);
            let b_float = floatish(b);
            if a_float && !b_float {
                if let Expr::RawDeref(_, _, w) = b.as_mut() {
                    if *w == 4 {
                        *w = 5;
                        *n += 1;
                    }
                }
            } else if b_float && !a_float {
                if let Expr::RawDeref(_, _, w) = a.as_mut() {
                    if *w == 4 {
                        *w = 5;
                        *n += 1;
                    }
                }
            }
        }
        for c in crate::recovery::reconstruct::il2cpp_cleanup::expr_children_mut(e) {
            visit_expr(c, n);
        }
    }
    for s in stmts.iter_mut() {
        crate::recovery::reconstruct::copy_prop::for_each_expr_mut(s, &mut |e| visit_expr(e, n));
        for k in super::children_mut(s) {
            retype_float_comparison_derefs(k, n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::BinOp;

    fn v(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    #[test]
    fn a_whole_vector_store_followed_by_a_z_store_is_x() {
        let mut m = ResolveMaps::default();
        m.tdi_names.insert(388, (String::new(), "TRRunnerPawn".into()));
        m.tdi_names.insert(7651, ("UnityEngine".into(), "Vector3".into()));
        m.field_type_by_name
            .insert((388, "PlayerVelocity".into()), 7651);
        let mut stmts = vec![
            Stmt::Assign("this.PlayerVelocity".into(), None, v("in_vd2")),
            Stmt::Assign("this.PlayerVelocity.z".into(), None, v("in_vs0")),
        ];
        let n = apply(&mut stmts, &m, 388);
        assert_eq!(n, 1, "{stmts:?}");
        assert!(matches!(&stmts[0], Stmt::Assign(t, _, _) if t == "this.PlayerVelocity.x"));
    }

    #[test]
    fn a_class_init_guard_that_reassigns_the_same_token_is_removed() {
        let m = ResolveMaps::default();
        let mut stmts = vec![
            Stmt::Assign("type".into(), None, Expr::TypeOf("Foo".into())),
            Stmt::Assign("ptr".into(), None, v("type")),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::RawDeref(Box::new(v("type")), 224, 0)),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Assign("ptr".into(), None, Expr::TypeOf("Foo".into()))],
                else_body: vec![],
            },
            Stmt::Eval(v("x")),
        ];
        let n = apply(&mut stmts, &m, 0);
        assert_eq!(n, 1, "{stmts:?}");
        assert_eq!(stmts.len(), 3);
    }

    #[test]
    fn a_class_init_guard_assigning_a_new_value_is_kept() {
        let m = ResolveMaps::default();
        let mut stmts = vec![
            Stmt::Assign("type".into(), None, Expr::TypeOf("Foo".into())),
            Stmt::Assign("ptr".into(), None, v("type")),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::RawDeref(Box::new(v("type")), 224, 0)),
                    Box::new(Expr::Int(0)),
                ),
                // Assigns a DIFFERENT token: not the identity.
                then_body: vec![Stmt::Assign("ptr".into(), None, Expr::TypeOf("Bar".into()))],
                else_body: vec![],
            },
        ];
        let n = apply(&mut stmts, &m, 0);
        assert_eq!(n, 0, "{stmts:?}");
    }

    #[test]
    fn a_float_comparison_retypes_an_absolute_int_deref() {
        let m = ResolveMaps::default();
        let mut stmts = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Le,
                Box::new(Expr::Call {
                    recv: None,
                    name: "Sqrt".into(),
                    args: vec![Expr::Float(1.0)],
                }),
                Box::new(Expr::RawDeref(Box::new(Expr::Int(0x799000)), 644, 4)),
            ),
            then_body: vec![],
            else_body: vec![],
        }];
        let n = apply(&mut stmts, &m, 0);
        assert_eq!(n, 1);
        let d = format!("{stmts:?}");
        assert!(d.contains("RawDeref") && d.contains("644") && d.contains("5"), "{d}");
        let _ = BinOp::Mul;
    }
}
