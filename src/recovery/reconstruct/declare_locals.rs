//! Declare the recovered locals.
//!
//! [`Stmt::Assign`] carries an optional declared type: `Some(ty)` renders
//! `ty name = value;`, `None` renders a bare `name = value;`. The renderer,
//! `stmt_inline` (for-loop headers) and several guards already understand the
//! declaration form, but nothing else in the pipeline sets it — so without
//! this pass every recovered local is a bare assignment and the method
//! declares none of its locals at all.
//!
//! ## Where the declaration goes
//!
//! The declaration belongs in the **lowest common ancestor** of the local's
//! mentions: the innermost statement list that contains every one of them.
//! That is the tightest C# scope that still covers every use.
//!
//! ```text
//! if (a) {                       if (a) {
//!     v3 = f();                      var v3 = f();
//!     use(v3);           not         use(v3);
//! }                              }
//! ```
//!
//! Two shapes are emitted:
//!
//! * `var name = <first write>;` when the first statement of the LCA list that
//!   mentions the name *is* a plain write of it. `var` is always the right
//!   type for its initializer, so it can never contradict a later use.
//! * `ty name = default;` when the first mention is a read, or a write nested
//!   inside a branch. CIL locals are zero-initialised and function-scoped, so
//!   this is the faithful spelling; the type comes from the metadata local
//!   signature or from `local_types`' definition-based inference.
//!
//! ## Safety
//!
//! * **Loops.** A typed declaration must not sit inside a loop body: it would
//!   re-initialise the local on every iteration, and CIL locals carry their
//!   value across iterations. A `var` declaration inside a loop body is fine —
//!   its initializer *is* the original write, so re-running it is exactly what
//!   the original code does.
//! * **Jumps.** A `goto` must not be able to cross the declaration: entering
//!   the list past it leaves the local unassigned (CS0165), and jumping back
//!   over a typed declaration re-initialises it. The test is exact — the jump
//!   table is built per list, so a label no jump uses costs nothing. A `var`
//!   is only at risk from a FORWARD jump (a backward one re-runs the original
//!   write, which is what the CIL did); a typed declaration is at risk from
//!   both, and escapes to an ancestor list. At the head of the body no jump
//!   can cross, so index 0 is always available as a last resort.
//!
//! ## What is never declared
//!
//! * Parameters (they are already in the signature) — passed in by the caller.
//! * Non-identifiers (`this.field`, `arr[0]`, `v12..ctor()`), which are not
//!   declaration targets.
//! * ABI/runtime spellings the renderer uses for things that are not locals
//!   (`stack_N` frame cells, `__abi` slots, `in_xN` incoming registers,
//!   `fn_…`/`ptr_…`/`fnptr_…` degraded addresses).
//! * A local with no known type whose first mention is a read: there is no
//!   initializer to write and no type to declare, and guessing `object` would
//!   break every arithmetic use. These are counted in [`report`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::recovery::csharp::{self, Expr, Stmt};

// ---------------------------------------------------------------------------
// Counters (`ILR_RECON` report)
// ---------------------------------------------------------------------------

static VAR_LCA: AtomicU64 = AtomicU64::new(0);
static VAR_BLOCK: AtomicU64 = AtomicU64::new(0);
static TYPED_LCA: AtomicU64 = AtomicU64::new(0);
static TYPED_BLOCK: AtomicU64 = AtomicU64::new(0);
/// Declarations emitted as `ty name;` because every read is definitely
/// assigned on every path (no `= default` initializer needed).
static BARE: AtomicU64 = AtomicU64::new(0);
static UNDECL_UNTYPED: AtomicU64 = AtomicU64::new(0);
static UNDECL_NOT_LOCAL: AtomicU64 = AtomicU64::new(0);
static UNDECL_PARAM: AtomicU64 = AtomicU64::new(0);
static UNDECL_NO_SITE: AtomicU64 = AtomicU64::new(0);

fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

/// One line for the `ILR_RECON` dashboard: how many locals this pass declared,
/// where they landed, and why the rest could not be declared.
pub fn report() -> String {
    let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
    let var = g(&VAR_LCA) + g(&VAR_BLOCK);
    let typed = g(&TYPED_LCA) + g(&TYPED_BLOCK);
    let undeclared =
        g(&UNDECL_UNTYPED) + g(&UNDECL_NOT_LOCAL) + g(&UNDECL_PARAM) + g(&UNDECL_NO_SITE);
    if var + typed + undeclared == 0 {
        return String::new();
    }
    format!(
        "[reconstruct]   declare_locals: {var} var ({} at the LCA, {} in a block), \
         {typed} typed ({} at the LCA, {} in a block), {} bare (definite assignment); undeclared: \
         {} untyped, {} not-a-local, {} parameter, {} no-site\n",
        g(&VAR_LCA),
        g(&VAR_BLOCK),
        g(&TYPED_LCA),
        g(&TYPED_BLOCK),
        g(&BARE),
        g(&UNDECL_UNTYPED),
        g(&UNDECL_NOT_LOCAL),
        g(&UNDECL_PARAM),
        g(&UNDECL_NO_SITE),
    )
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// A list position: one `(statement index, child slot)` step per nesting level.
type Path = Vec<(usize, usize)>;

/// A decided declaration: where it goes, which shape, and for which name.
struct Decl {
    level: Path,
    idx: usize,
    /// `None` renders `var name = <the statement's own write>;`.
    /// `Some(ty)` renders `ty name = default;` — or `ty name;` when `bare`.
    ty: Option<String>,
    name: String,
    /// `Some(ty)` with `bare` renders a declaration with NO initializer: the
    /// definite-assignment proof showed every read is written first.
    bare: bool,
}

/// Declare every recovered local at the lowest common ancestor of its mentions.
/// `params` are the method's parameter names (never redeclared) and `types` is
/// the best known C# type per local (metadata signature, then inference).
pub fn apply(
    stmts: &mut Vec<Stmt>,
    params: &HashSet<String>,
    types: &HashMap<String, String>,
) -> usize {
    // A name that is never written is not a local this pass owns: it is a
    // parameter, a field the renderer spelled without `this.`, or noise.
    let mut targets: HashSet<String> = HashSet::new();
    for s in stmts.iter() {
        targets_into(s, &mut targets);
    }
    if targets.is_empty() {
        return 0;
    }
    // Every mention of every target, with the path of the list holding it.
    // `legal` is the set of lists a declaration may live in, `loops` the set
    // of lists that are re-entered (a typed declaration must escape those).
    let mut sites: HashMap<String, Vec<Path>> = HashMap::new();
    let mut legal: HashSet<Path> = HashSet::new();
    let mut loops: HashSet<Path> = HashSet::new();
    let mut path: Path = Vec::new();
    collect_sites(
        stmts,
        &mut path,
        &targets,
        &mut sites,
        &mut legal,
        &mut loops,
        true,
    );
    // Sorted so the output never depends on hash iteration order.
    let mut names: Vec<String> = targets.iter().cloned().collect();
    names.sort();
    let mut decls: Vec<Decl> = Vec::new();
    let mut firsts: HashMap<Path, HashMap<String, usize>> = HashMap::new();
    let mut jump_cache: HashMap<Path, std::rc::Rc<Vec<(usize, usize)>>> = HashMap::new();
    for name in names {
        if !plain_local(&name) {
            bump(&UNDECL_NOT_LOCAL);
            continue;
        }
        if params.contains(&name) {
            bump(&UNDECL_PARAM);
            continue;
        }
        let Some(mentions) = sites.get(&name) else {
            bump(&UNDECL_NO_SITE);
            continue;
        };
        let mut level = common_prefix(mentions);
        while !legal.contains(&level) {
            level.pop();
        }
        match decide(
            stmts,
            &level,
            &name,
            types,
            &loops,
            &targets,
            &mut firsts,
            &mut jump_cache,
        ) {
            Some(d) => decls.push(d),
            None => bump(&UNDECL_UNTYPED),
        }
    }
    if decls.is_empty() {
        return 0;
    }
    let n = decls.len();
    // DEEPEST LIST FIRST. Inserting into a list never changes the statement
    // count of its ancestors, so the paths decided above stay valid.
    let mut by_level: std::collections::BTreeMap<Path, Vec<Decl>> =
        std::collections::BTreeMap::new();
    for d in decls {
        by_level.entry(d.level.clone()).or_default().push(d);
    }
    let mut groups: Vec<(Path, Vec<Decl>)> = by_level.into_iter().collect();
    groups.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (level, mut items) in groups {
        items.sort_by(|a, b| a.idx.cmp(&b.idx).then_with(|| a.name.cmp(&b.name)));
        with_list_at(stmts, &level, &mut |list| {
            let mut inserts: HashMap<usize, Vec<(String, String, bool)>> = HashMap::new();
            let mut vars: HashMap<usize, String> = HashMap::new();
            for d in items.iter() {
                match &d.ty {
                    Some(ty) => inserts
                        .entry(d.idx)
                        .or_default()
                        .push((d.name.clone(), ty.clone(), d.bare)),
                    None => {
                        vars.insert(d.idx, d.name.clone());
                    }
                }
            }
            let old = std::mem::take(list);
            let mut out: Vec<Stmt> = Vec::with_capacity(old.len() + inserts.len());
            for (i, mut s) in old.into_iter().enumerate() {
                if let Some(d) = inserts.remove(&i) {
                    for (name, ty, bare) in d {
                        if bare {
                            out.push(Stmt::Declare(name, ty));
                        } else {
                            out.push(Stmt::Assign(
                                name,
                                Some(ty),
                                Expr::Var("default".to_string()),
                            ));
                        }
                    }
                }
                if let Some(name) = vars.get(&i) {
                    if let Stmt::Assign(target, ty, _) = &mut s {
                        if target == name {
                            *ty = Some("var".to_string());
                        }
                    }
                }
                out.push(s);
            }
            *list = out;
        });
    }
    n
}

/// Decide where (and in which shape) `name` is declared, starting at `level`.
///
/// Tries the level it was given first — the tightest scope — and walks out one
/// list at a time when that list cannot hold the declaration (it is re-entered
/// by a loop, or a label lets a jump skip past it).
fn decide(
    stmts: &Vec<Stmt>,
    start: &Path,
    name: &str,
    types: &HashMap<String, String>,
    loops: &HashSet<Path>,
    targets: &HashSet<String>,
    firsts: &mut HashMap<Path, HashMap<String, usize>>,
    jump_cache: &mut HashMap<Path, std::rc::Rc<Vec<(usize, usize)>>>,
) -> Option<Decl> {
    let mut level = start.clone();
    loop {
        let list = list_ref(stmts, &level)?;
        let first = firsts
            .entry(level.clone())
            .or_insert_with(|| mention_census(list, targets));
        let Some(&k) = first.get(name) else {
            return None;
        };
        let root = level.is_empty();
        // A declaration is only skippable by a `goto` that lands in THIS list.
        // The crossing test below is exact for those, so a label that no jump
        // uses no longer blocks a declaration.
        let jumps = jump_cache
            .entry(level.clone())
            .or_insert_with(|| std::rc::Rc::new(jumps(list)))
            .clone();
        // SHAPE 1: the first mention is a plain write, so it becomes the
        // declaration. Safe wherever a `var` can be spelled, and inside a loop
        // too (the initializer is the original write).
        if let Stmt::Assign(target, None, e) = &list[k] {
            let skipped = jumps.iter().any(|&(g, j)| g < k && j >= k);
            if target == name && !is_unknown(e) && !mentions(e, name) && !skipped {
                if root {
                    bump(&VAR_LCA);
                } else {
                    bump(&VAR_BLOCK);
                }
                return Some(Decl {
                    level,
                    idx: k,
                    ty: None,
                    name: name.to_string(),
                    bare: false,
                });
            }
        }
        // SHAPE 2: a typed declaration carrying `default`.
        if !loops.contains(&level) {
            // A typed declaration must not be crossed by a jump: forward it
            // would be skipped (CS0165), backward it would re-initialise a
            // local the CIL carries across the jump.
            let crossed = jumps
                .iter()
                .any(|&(g, j)| (g < k && j >= k) || (j <= k && g >= k));
            if crossed && !root {
                // Only an ancestor list can hold a declaration no jump skips.
            } else if let Some(ty) = types.get(name).filter(|t| usable(t)) {
                // At the head of the body a declaration is always executed
                // first; further in it can be jumped over, so use index 0.
                let idx = if crossed { 0 } else { k };
                // A bare declaration (`ty name;`) is only spec-valid when
                // every read is definitely assigned first. The proof walks
                // the statement tree from the declaration position; it
                // refuses address-taken locals and any list with jumps.
                let bare = jumps.is_empty() && !address_taken(list, name)
                    && seq_assigns(&list[idx..], name, false).is_some();
                if bare {
                    bump(&BARE);
                } else if root {
                    bump(&TYPED_LCA);
                } else {
                    bump(&TYPED_BLOCK);
                }
                return Some(Decl {
                    level,
                    idx,
                    ty: Some(ty.clone()),
                    name: name.to_string(),
                    bare,
                });
            }
        }
        // Widen the scope: step out of this list (and out of every loop that
        // still encloses the new position).
        if level.is_empty() {
            return None;
        }
        let mut up = level.clone();
        up.pop();
        while !up.is_empty() && loops.contains(&up) {
            up.pop();
        }
        if up == level {
            return None;
        }
        level = up;
    }
}

/// The longest prefix shared by every mention's path.
fn common_prefix(sites: &[Path]) -> Path {
    let mut lca = sites[0].clone();
    for s in sites.iter().skip(1) {
        let mut k = 0;
        while k < lca.len() && k < s.len() && lca[k] == s[k] {
            k += 1;
        }
        lca.truncate(k);
        if lca.is_empty() {
            break;
        }
    }
    lca
}

fn collect_sites(
    list: &[Stmt],
    path: &mut Path,
    targets: &HashSet<String>,
    sites: &mut HashMap<String, Vec<Path>>,
    legal: &mut HashSet<Path>,
    loops: &mut HashSet<Path>,
    declare_here: bool,
) {
    if declare_here {
        legal.insert(path.clone());
    }
    for (i, s) in list.iter().enumerate() {
        // Mentions in THIS statement's own expressions, not its child lists:
        // the children are visited next and record their own sites.
        let mut own: HashMap<String, usize> = HashMap::new();
        count_stmt(s, &mut own);
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            let mut sub: HashMap<String, usize> = HashMap::new();
            count_list(k, &mut sub);
            for (name, c) in sub {
                if let Some(slot) = own.get_mut(&name) {
                    *slot = slot.saturating_sub(c);
                }
            }
        }
        for (name, c) in own {
            if c == 0 || !targets.contains(&name) {
                continue;
            }
            sites.entry(name).or_default().push(path.clone());
        }
        let is_loop = matches!(
            s,
            Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::For { .. } | Stmt::ForEach { .. }
        );
        let mut kids2: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids2);
        for (slot, child) in kids2.into_iter().enumerate() {
            // A `for` incrementor cannot hold a declaration, so it is not a
            // legal site (its mentions still count, so the LCA escapes it).
            let incr = matches!(s, Stmt::For { .. }) && slot == 1;
            path.push((i, slot));
            if is_loop {
                loops.insert(path.clone());
            }
            collect_sites(child, path, targets, sites, legal, loops, !incr);
            path.pop();
        }
    }
}

/// Every name one top-level statement mentions anywhere inside it: reads in
/// its expressions (or its nested lists), any assignment target, and the
/// identifier tokens of an lvalue the expression walkers cannot see.
fn seen_in_stmt(s: &Stmt, out: &mut HashSet<String>) {
    csharp::collect_used_vars_stmt(s, out);
    targets_into(s, out);
    lvalue_mentions(s, &mut |n| {
        out.insert(n.to_string());
    });
}

/// First top-level index that mentions each target name, built in ONE pass
/// over the list. `decide` asks this per name and level; rescanning the list
/// with three fresh sets per name was the dominant cost of this pass.
fn mention_census(list: &[Stmt], targets: &HashSet<String>) -> HashMap<String, usize> {
    let mut first: HashMap<String, usize> = HashMap::new();
    if targets.is_empty() {
        return first;
    }
    let mut names: HashSet<String> = HashSet::new();
    for (i, s) in list.iter().enumerate() {
        names.clear();
        seen_in_stmt(s, &mut names);
        for n in &names {
            if targets.contains(n) {
                first.entry(n.clone()).or_insert(i);
            }
        }
        if first.len() == targets.len() {
            break;
        }
    }
    first
}

fn mentions(e: &Expr, name: &str) -> bool {
    let mut used: HashSet<String> = HashSet::new();
    csharp::collect_used_vars(e, &mut used);
    used.contains(name)
}

fn is_unknown(e: &Expr) -> bool {
    matches!(e, Expr::Unknown | Expr::MissingAbiArg)
}

/// A type this pass can write in a declaration.
fn usable(ty: &str) -> bool {
    !matches!(ty, "" | "object" | "__unknown" | "var" | "void")
}

/// `goto` positions whose target label lives in this same list:
/// `(index of the statement holding the goto, index of the label)`.
///
/// Only same-list targets matter. A `goto` INTO a nested block is not valid
/// C# to begin with, and a `goto` out of one cannot skip a declaration in this
/// list.
fn jumps(list: &[Stmt]) -> Vec<(usize, usize)> {
    let mut labels: HashMap<&str, usize> = HashMap::new();
    for (i, s) in list.iter().enumerate() {
        if let Stmt::Label(name) = s {
            labels.insert(name.as_str(), i);
        }
    }
    if labels.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, s) in list.iter().enumerate() {
        collect_jumps(s, i, &labels, &mut out);
    }
    out
}

fn collect_jumps(
    s: &Stmt,
    at: usize,
    labels: &HashMap<&str, usize>,
    out: &mut Vec<(usize, usize)>,
) {
    if let Stmt::Goto(target) = s {
        if let Some(&j) = labels.get(target.as_str()) {
            out.push((at, j));
        }
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for c in k {
            collect_jumps(c, at, labels, out);
        }
    }
}

/// The list at `path`, immutably.
fn list_ref<'a>(stmts: &'a Vec<Stmt>, path: &[(usize, usize)]) -> Option<&'a Vec<Stmt>> {
    let mut cur: &'a Vec<Stmt> = stmts;
    for &(i, slot) in path {
        let s = cur.get(i)?;
        let mut kids: Vec<&'a Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        cur = *kids.get(slot)?;
    }
    Some(cur)
}

/// Run `f` on the list at `path`, mutably.
fn with_list_at(stmts: &mut Vec<Stmt>, path: &[(usize, usize)], f: &mut dyn FnMut(&mut Vec<Stmt>)) {
    if path.is_empty() {
        f(stmts);
        return;
    }
    let (i, slot) = path[0];
    if let Some(s) = stmts.get_mut(i) {
        let mut kids: Vec<&mut Vec<Stmt>> = super::children_mut(s);
        if slot < kids.len() {
            let child = kids.swap_remove(slot);
            with_list_at(child, &path[1..], f);
        }
    }
}

/// A name that can appear on the left of a C# declaration.
fn plain_local(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return false;
    }
    // C# keywords the renderer emits as expressions, plus the ABI/runtime
    // spellings listed in the module docs — none of these is a local.
    const NOT_A_LOCAL: &[&str] = &["this", "base", "null", "true", "false", "new", "default"];
    if NOT_A_LOCAL.contains(&name) {
        return false;
    }
    const NOT_A_LOCAL_PREFIX: &[&str] = &["stack_", "__", "in_", "fn_", "fnptr_", "ptr_"];
    !NOT_A_LOCAL_PREFIX.iter().any(|p| name.starts_with(p))
}

/// Count every mention of every name in `list`, reads AND assignment targets.
///
/// Reads come from `csharp::collect_used_vars_stmt` (the same walker the
/// dead-local sweep uses); targets are added on top because a name that is
/// only ever *written* still occupies a scope slot.
fn count_list(list: &[Stmt], out: &mut HashMap<String, usize>) {
    for s in list {
        count_stmt(s, out);
    }
}

fn count_stmt(s: &Stmt, out: &mut HashMap<String, usize>) {
    let mut reads: HashSet<String> = HashSet::new();
    csharp::collect_used_vars_stmt(s, &mut reads);
    for r in reads {
        *out.entry(r).or_insert(0) += 1;
    }
    count_targets(s, out);
    lvalue_mentions(s, &mut |n| {
        *out.entry(n.to_string()).or_insert(0) += 1;
    });
}

/// Names mentioned inside a NON-identifier assignment target.
///
/// `Stmt::Assign` carries its target as a rendered string, so
/// `*(byte*)((nint)&v1 +4) = 0;` mentions `v1` in a place no expression
/// walker can see. Counting those is what keeps a declaration from landing
/// after its first use (CS0841). Over-counting only widens the scope the
/// declaration is hoisted to, so the token split may be coarse.
fn lvalue_mentions(s: &Stmt, f: &mut dyn FnMut(&str)) {
    let Stmt::Assign(name, _, _) = s else {
        return;
    };
    if plain_local(name) {
        return;
    }
    for tok in name.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if !tok.is_empty() && !tok.starts_with(|c: char| c.is_ascii_digit()) {
            f(tok);
        }
    }
}

fn count_targets(s: &Stmt, out: &mut HashMap<String, usize>) {
    if let Stmt::Assign(name, _, _) = s {
        *out.entry(name.clone()).or_insert(0) += 1;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for c in k {
            count_targets(c, out);
        }
    }
}

fn targets_into(s: &Stmt, out: &mut HashSet<String>) {
    if let Stmt::Assign(name, _, _) = s {
        out.insert(name.clone());
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(s, &mut kids);
    for k in kids {
        for c in k {
            targets_into(c, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Definite assignment (bare declarations)
// ---------------------------------------------------------------------------

fn expr_reads_name(e: &Expr, name: &str) -> bool {
    if matches!(e, Expr::Var(n) if n == name) {
        return true;
    }
    if let Expr::Lambda { params, .. } = e {
        // A lambda parameter with the same spelling shadows the local.
        if params.iter().any(|p| p == name) {
            return false;
        }
    }
    crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e)
        .into_iter()
        .any(|c| expr_reads_name(c, name))
}

/// Reads of `name` in this statement's own expressions. A mention inside a
/// complex assignment target (`*(T*)(&x +4) = …`) is a read too, and
/// `address_taken` separately refuses the bare form for it.
fn own_reads(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Switch { disc, .. } if disc == name => return true,
        Stmt::ForEach { collection, .. } if collection == name => return true,
        _ => {}
    }
    let mut hit = false;
    lvalue_mentions(s, &mut |n| {
        if n == name {
            hit = true;
        }
    });
    if hit {
        return true;
    }
    super::float_locals::for_each_expr(s, &mut |e| {
        if !hit && expr_reads_name(e, name) {
            hit = true;
        }
    });
    hit
}

fn expr_takes_address(e: &Expr, name: &str) -> bool {
    match e {
        Expr::AddressOf(x) | Expr::RawAddr(x, _) => {
            if matches!(x.as_ref(), Expr::Var(n) if n == name) {
                return true;
            }
        }
        _ => {}
    }
    crate::recovery::reconstruct::il2cpp_cleanup::expr_children(e)
        .into_iter()
        .any(|c| expr_takes_address(c, name))
}

/// A bare `ty name;` is never emitted for an address-taken local: its memory
/// can be written without a visible assignment, so C# definite assignment
/// cannot see the write.
fn address_taken(list: &[Stmt], name: &str) -> bool {
    fn stmt(s: &Stmt, name: &str) -> bool {
        if let Stmt::Assign(target, _, _) = s {
            if target != name && target.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).any(|t| t == name) {
                return true;
            }
        }
        let mut hit = false;
        super::float_locals::for_each_expr(s, &mut |e| {
            if !hit && expr_takes_address(e, name) {
                hit = true;
            }
        });
        if hit {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        kids.into_iter().any(|k| k.iter().any(|s| stmt(s, name)))
    }
    list.iter().any(|s| stmt(s, name))
}

/// Does this list end with a path that cannot fall out (return/throw),
/// including an if whose both arms terminate?
fn always_terminates(list: &[Stmt]) -> bool {
    match list.last() {
        Some(Stmt::Return(_)) | Some(Stmt::Throw(_)) => true,
        Some(Stmt::If {
            then_body,
            else_body,
            ..
        }) => !else_body.is_empty() && always_terminates(then_body) && always_terminates(else_body),
        Some(Stmt::Switch {
            cases, default, ..
        }) => {
            default.as_ref().is_some_and(|d| always_terminates(d))
                && cases.iter().all(|(_, b)| always_terminates(b))
        }
        Some(Stmt::StringSwitch { cases, tail, .. }) => {
            tail.as_ref().is_some_and(|d| always_terminates(d))
                && cases.iter().all(|(_, b)| always_terminates(b))
        }
        _ => false,
    }
}

/// Walk the statements after a declaration with an `assigned` flag. Returns
/// `None` as soon as a read can happen with the local unassigned, or when a
/// construct cannot be modelled; `Some(_)` means every read on every path is
/// preceded by a write.
fn seq_assigns(list: &[Stmt], name: &str, mut assigned: bool) -> Option<bool> {
    for s in list {
        if own_reads(s, name) && !assigned {
            return None;
        }
        match s {
            Stmt::Assign(t, _, _) if t == name => assigned = true,
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                let at = seq_assigns(then_body, name, assigned)?;
                let ae = if else_body.is_empty() {
                    None
                } else {
                    Some(seq_assigns(else_body, name, assigned)?)
                };
                let at_merge = if always_terminates(then_body) {
                    assigned
                } else {
                    at
                };
                let ae_merge = match ae {
                    Some(a) if !always_terminates(else_body) => a,
                    _ => assigned,
                };
                assigned = at_merge && ae_merge;
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => {
                let _ = seq_assigns(body, name, assigned)?;
            }
            Stmt::DoWhile { body, cond } => {
                let ab = seq_assigns(body, name, assigned)?;
                if expr_reads_name(cond, name) && !ab {
                    return None;
                }
                if !assigned {
                    return None;
                }
            }
            Stmt::ForEach { body, .. } => {
                let _ = seq_assigns(body, name, assigned)?;
            }
            Stmt::Switch { .. } | Stmt::StringSwitch { .. } | Stmt::Try { .. } => {
                if !assigned {
                    return None;
                }
            }
            Stmt::Return(_) | Stmt::Throw(_) => return Some(assigned),
            Stmt::Break | Stmt::Continue | Stmt::Goto(_) | Stmt::Label(_) => {
                if !assigned {
                    return None;
                }
            }
            Stmt::Lock { body, .. } => {
                let _ = seq_assigns(body, name, assigned)?;
            }
            Stmt::Using { decl, body } => {
                if own_reads(decl, name) && !assigned {
                    return None;
                }
                if let Stmt::Assign(t, _, _) = decl.as_ref() {
                    if t == name {
                        assigned = true;
                    }
                }
                let _ = seq_assigns(body, name, assigned)?;
            }
            _ => {}
        }
    }
    Some(assigned)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assign(name: &str, e: Expr) -> Stmt {
        Stmt::Assign(name.to_string(), None, e)
    }

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }

    fn params() -> HashSet<String> {
        HashSet::new()
    }

    fn typed(name: &str, ty: &str) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(name.to_string(), ty.to_string());
        m
    }

    fn if_stmt(then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond: Expr::Bool(true),
            then_body,
            else_body,
        }
    }

    #[test]
    fn both_arms_assigning_emits_a_bare_declaration() {
        let mut stmts = vec![
            if_stmt(
                vec![assign("v3", Expr::Int(1))],
                vec![assign("v3", Expr::Int(2))],
            ),
            Stmt::Eval(var("v3")),
        ];
        let n = apply(&mut stmts, &params(), &typed("v3", "long"));
        assert!(n >= 1);
        assert!(
            matches!(&stmts[0], Stmt::Declare(n, t) if n == "v3" && t == "long"),
            "{stmts:?}"
        );
    }

    #[test]
    fn conditional_assignment_keeps_the_default() {
        let mut stmts = vec![
            if_stmt(vec![assign("v3", Expr::Int(1))], vec![]),
            Stmt::Eval(var("v3")),
        ];
        apply(&mut stmts, &params(), &typed("v3", "long"));
        let d = format!("{stmts:?}");
        assert!(d.contains("default"), "{d}");
        assert!(!matches!(&stmts[0], Stmt::Declare(..)), "{d}");
    }

    #[test]
    fn first_write_at_top_level_becomes_a_declaration() {
        let mut stmts = vec![
            assign("v1", Expr::Int(1)),
            assign(
                "v2",
                Expr::Bin(
                    csharp::BinOp::Add,
                    Box::new(var("v1")),
                    Box::new(Expr::Int(2)),
                ),
            ),
        ];
        assert_eq!(apply(&mut stmts, &params(), &HashMap::new()), 2);
        assert!(matches!(&stmts[0], Stmt::Assign(_, Some(t), _) if t == "var"));
        assert!(matches!(&stmts[1], Stmt::Assign(_, Some(t), _) if t == "var"));
    }

    #[test]
    fn a_second_write_is_not_a_declaration() {
        let mut stmts = vec![assign("v1", Expr::Int(1)), assign("v1", Expr::Int(2))];
        assert_eq!(apply(&mut stmts, &params(), &HashMap::new()), 1);
        assert!(matches!(&stmts[0], Stmt::Assign(_, Some(_), _)));
        assert!(matches!(&stmts[1], Stmt::Assign(_, None, _)));
    }

    #[test]
    fn a_read_before_the_first_write_blocks_the_declaration() {
        // `if (v1) …; v1 = 1;` — a `var` at the assignment would leave the
        // earlier read undeclared, and there is no type to hoist.
        let mut stmts = vec![if_stmt(vec![Stmt::Eval(var("v2"))], vec![]), assign("v1", Expr::Int(1))];
        stmts[0] = Stmt::If {
            cond: var("v1"),
            then_body: vec![Stmt::Eval(var("v2"))],
            else_body: vec![],
        };
        assert_eq!(apply(&mut stmts, &params(), &HashMap::new()), 0);
        assert!(matches!(&stmts[1], Stmt::Assign(_, None, _)));
    }

    #[test]
    fn a_local_used_outside_the_block_is_not_declared_inside_it() {
        let mut stmts = vec![
            if_stmt(vec![assign("v1", Expr::Int(1))], vec![]),
            Stmt::Eval(var("v1")),
        ];
        apply(&mut stmts, &params(), &HashMap::new());
        let Stmt::If { then_body, .. } = &stmts[0] else {
            panic!("if")
        };
        assert!(matches!(&then_body[0], Stmt::Assign(_, None, _)));
    }

    #[test]
    fn a_local_confined_to_a_block_is_declared_there() {
        let mut stmts = vec![if_stmt(
            vec![assign("v1", Expr::Int(1)), Stmt::Eval(var("v1"))],
            vec![],
        )];
        apply(&mut stmts, &params(), &HashMap::new());
        let Stmt::If { then_body, .. } = &stmts[0] else {
            panic!("if")
        };
        assert!(matches!(&then_body[0], Stmt::Assign(_, Some(t), _) if t == "var"));
    }

    #[test]
    fn the_declaration_carries_the_real_initializer() {
        let call = Expr::Call {
            recv: None,
            name: "f".to_string(),
            args: vec![],
        };
        let mut stmts = vec![if_stmt(
            vec![assign("v1", call.clone()), Stmt::Eval(var("v1"))],
            vec![],
        )];
        apply(&mut stmts, &params(), &HashMap::new());
        let Stmt::If { then_body, .. } = &stmts[0] else {
            panic!("if")
        };
        assert_eq!(then_body.len(), 2);
        assert!(matches!(&then_body[0], Stmt::Assign(n, Some(t), e)
            if n == "v1" && t == "var" && matches!(e, Expr::Call { name, .. } if name == "f")));
    }

    #[test]
    fn a_read_first_local_is_declared_typed_where_it_lives() {
        let mut stmts = vec![if_stmt(
            vec![Stmt::Eval(var("v1")), assign("v1", Expr::Int(1))],
            vec![],
        )];
        assert_eq!(apply(&mut stmts, &params(), &typed("v1", "int")), 1);
        let Stmt::If { then_body, .. } = &stmts[0] else {
            panic!("if")
        };
        assert_eq!(then_body.len(), 3);
        assert!(matches!(&then_body[0], Stmt::Assign(n, Some(t), Expr::Var(d))
            if n == "v1" && t == "int" && d == "default"));
        assert!(matches!(&then_body[2], Stmt::Assign(_, None, _)));
    }

    #[test]
    fn a_typed_declaration_never_lands_inside_a_loop() {
        // A typed declaration inside the body would re-initialise a local the
        // CIL carries across iterations.
        let mut stmts = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Eval(var("v1")), assign("v1", Expr::Int(1))],
        }];
        assert_eq!(apply(&mut stmts, &params(), &typed("v1", "int")), 1);
        assert!(matches!(&stmts[0], Stmt::Assign(n, Some(t), _) if n == "v1" && t == "int"));
        assert!(matches!(&stmts[1], Stmt::While { .. }));
        let Stmt::While { body, .. } = &stmts[1] else {
            panic!("while")
        };
        assert_eq!(body.len(), 2);
    }

    #[test]
    fn a_declaration_escapes_a_jump_that_crosses_it() {
        // `goto L1; use(v1); L1: v1 = 1;` — entering at the label skips the
        // declaration, which C# rejects as an unassigned read.
        let mut stmts = vec![if_stmt(
            vec![
                Stmt::Goto("L1".to_string()),
                Stmt::Eval(var("v1")),
                Stmt::Label("L1".to_string()),
                assign("v1", Expr::Int(1)),
            ],
            vec![],
        )];
        assert_eq!(apply(&mut stmts, &params(), &typed("v1", "int")), 1);
        assert!(matches!(&stmts[0], Stmt::Assign(n, Some(t), _) if n == "v1" && t == "int"));
        assert!(matches!(&stmts[1], Stmt::If { .. }));
        let Stmt::If { then_body, .. } = &stmts[1] else {
            panic!("if")
        };
        assert_eq!(then_body.len(), 4);
        assert!(matches!(&then_body[1], Stmt::Eval(_)));
    }

    #[test]
    fn a_label_no_jump_uses_does_not_block_the_declaration() {
        let mut stmts = vec![if_stmt(
            vec![
                Stmt::Label("L1".to_string()),
                assign("v1", Expr::Int(1)),
                Stmt::Eval(var("v1")),
            ],
            vec![],
        )];
        assert_eq!(apply(&mut stmts, &params(), &HashMap::new()), 1);
        let Stmt::If { then_body, .. } = &stmts[0] else {
            panic!("if")
        };
        assert!(matches!(&then_body[1], Stmt::Assign(_, Some(t), _) if t == "var"));
    }

    #[test]
    fn a_local_written_in_two_arms_is_declared_above_the_branch() {
        let mut stmts = vec![if_stmt(
            vec![assign("v1", Expr::Int(1))],
            vec![assign("v1", Expr::Int(2))],
        )];
        assert_eq!(apply(&mut stmts, &params(), &typed("v1", "int")), 1);
        assert_eq!(stmts.len(), 2);
        // Both arms assign, so the declaration needs no `= default`
        // initializer: `int v1;`
        assert!(matches!(&stmts[0], Stmt::Declare(n, t) if n == "v1" && t == "int"));
        let Stmt::If {
            then_body,
            else_body,
            ..
        } = &stmts[1]
        else {
            panic!("if")
        };
        assert!(matches!(&then_body[0], Stmt::Assign(_, None, _)));
        assert!(matches!(&else_body[0], Stmt::Assign(_, None, _)));
    }

    #[test]
    fn a_self_referential_write_is_not_a_var_declaration() {
        let mut stmts = vec![assign(
            "v1",
            Expr::Bin(
                csharp::BinOp::Add,
                Box::new(var("v1")),
                Box::new(Expr::Int(1)),
            ),
        )];
        assert_eq!(apply(&mut stmts, &params(), &HashMap::new()), 0);
        assert!(matches!(&stmts[0], Stmt::Assign(_, None, _)));
    }

    #[test]
    fn an_unknown_initializer_is_not_a_var_declaration() {
        let mut stmts = vec![assign("v1", Expr::Unknown)];
        assert_eq!(apply(&mut stmts, &params(), &HashMap::new()), 0);
        assert!(matches!(&stmts[0], Stmt::Assign(_, None, _)));
    }

    #[test]
    fn a_reference_inside_an_lvalue_string_counts_as_a_mention() {
        // `*(byte*)((nint)&v1 +4) = 0;` mentions v1 before its first write, so
        // a declaration at the write would be a use-before-declaration.
        let mut stmts = vec![
            Stmt::Assign("*(byte*)((nint)&v1 +4)".to_string(), None, Expr::Int(0)),
            assign("v1", Expr::Int(0)),
        ];
        assert_eq!(apply(&mut stmts, &params(), &typed("v1", "int")), 1);
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[0], Stmt::Assign(n, Some(t), _) if n == "v1" && t == "int"));
    }

    #[test]
    fn parameters_and_non_identifiers_are_never_declared() {
        let mut p = HashSet::new();
        p.insert("in_x2".to_string());
        p.insert("count".to_string());
        let mut stmts = vec![
            assign("in_x2", Expr::Int(1)),
            assign("count", Expr::Int(1)),
            assign("this.f", Expr::Int(1)),
            assign("stack_3", Expr::Int(1)),
        ];
        assert_eq!(apply(&mut stmts, &p, &HashMap::new()), 0);
    }
}
