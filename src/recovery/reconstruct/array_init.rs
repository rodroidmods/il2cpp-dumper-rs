//! Array initializers + CLR bounds-check folding.
//!
//! ## The two halves
//!
//! ### 1. The bounds-check fold
//!
//! The CLR's array bounds check compiles to `if ((uint)i >= (uint)len) throw
//! new IndexOutOfRangeException();`. Where the IR-level passes
//! (`passes::remove_null_check_guards`, `passes::drop_vacuous_cond_branches`)
//! deliberately left one behind, it carries no C#-source semantics and is
//! folded away here.
//!
//! The condition's spelling on the real corpus is NOT `.Length`. Measured
//! output (`dump.cs`) writes the length as a raw deref of the array object:
//!
//! ```text
//! while (v13 < *(nint*)(v12 +24))      // Il2CppArray::max_length at 0x18
//! if (v7 < *(nint*)(this.ranges +24) && …)
//! ```
//!
//! `Il2CppArray` is `{ klass; monitor; bounds; max_length; }`, so
//! `*(nint*)(a + 24)` IS the array's length — and for `List<T>`,
//! `{ …; _items; _size; }` puts `_size` at the same 0x18. Both are length reads,
//! and both must be recognized or the pass can never fire (`.Length` appears
//! nowhere in the real output). The `.Length`/`.Count`/`get_Length` spellings
//! are kept for other architectures/pipelines that do resolve the property.
//!
//! Every gate is kept tight, because the cost of a wrong fold is deleting a real
//! check:
//!   * the arm is `if (cond) { throw <range-exception>; }` with NO else and
//!     exactly ONE statement in the arm (so removing it cannot orphan a
//!     label — a label in the arm would have to stay);
//!   * the thrown value names an out-of-range exception
//!     (`…OutOfRange…`) — a user `throw new Exception()` is never touched;
//!   * `cond` compares against an array-length reference on exactly ONE side,
//!     and the other side yields a local (the index);
//!   * the index is still READ later in the same list, so the access the guard
//!     protects (and the index's own only other use) survives.
//!
//! ### 2. The array initializer fold (implemented)
//!
//! When an array allocation is followed by consecutive element stores starting
//! at index 0, the stores are folded into an `Expr::ArrayInit`:
//!
//! ```text
//! v0 = new int[3];       v0 = new int[] { 10, 20, 30 };
//! v0[0] = 10;     →
//! v0[1] = 20;            // (stores removed)
//! v0[2] = 30;
//! ```
//!
//! Two element-store shapes are recognized:
//!
//!   * **Indexed assignment**: `Stmt::Assign("v0[0]", None, value)` — the
//!     assignment target's name contains the literal index.
//!
//!   * **Raw pointer write**: the IL2CPP codegen never uses managed array
//!     indexing for element stores. The real spelling is a raw pointer write:
//!     `*(nint*)(v0 + idx*elem_size + base_offset) = value`, modeled as
//!     `Stmt::Assign(_, _, Expr::RawDeref(base, addend, width))` where
//!     `addend = base_offset + idx * elem_size`. The base offset is the distance
//!     from the array object to the first element (typically 32 bytes on 64-bit:
//!     16 bytes object header + 8 bytes `Length` + 8 bytes alignment/padding).
//!
//! Safety conditions — ALL must hold for the fold to fire:
//!   * The allocation must be `NewObj` or `NewArr` with an array type name
//!     (contains `[]`) and exactly one dimension argument.
//!   * Element stores must start at index 0 and be consecutive.
//!   * Between stores, the array local must not be read or overwritten.
//!   * Intervening statements that do not mention the array are kept in place;
//!     calls between stores are tolerated only if they do not mention the array.
//!   * At least one element store must be collected (no empty `ArrayInit`).

use super::super::csharp::{CmpKind, Expr, Stmt};

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    super::walk_mut(stmts, &mut |list| {
        n += fold_bounds_guards(list);
        n += fold_array_initializers(list);
        // A first pass may have folded only the elements that shared the
        // allocation's name. Copy propagation later makes the remaining
        // element stores use that name; append them to the existing literal.
        n += extend_array_initializers(list);
    });
    n
}

/// Statements that cannot affect the array between its element stores.
fn is_harmless_between(s: &Stmt) -> bool {
    match s {
        Stmt::Comment(_) => true,
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => then_body.is_empty() && else_body.is_empty() && !expr_has_call(cond),
        _ => false,
    }
}

/// Append raw element stores that follow an existing `ArrayInit`.
///
/// The early pass can only fold stores adjacent to the allocation statement
/// and already naming the allocation temp. After copy propagation the rest of
/// a `.cctor`'s element stores name the same local, but the allocation has
/// already become `ArrayInit` — this pass extends it instead of requiring a
/// fresh allocation (0x160A6A0: 5-element initializers kept 4 raw stores).
fn extend_array_initializers(list: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    let mut i = 0usize;
    while i < list.len() {
        let (var, elem_type, existing_len) = match &list[i] {
            Stmt::Assign(var, Some(ty), Expr::ArrayInit { elem_type, items })
                if ty.contains("[]") =>
            {
                (var.clone(), elem_type.clone(), items.len() as u64)
            }
            _ => {
                i += 1;
                continue;
            }
        };
        let mut next = existing_len;
        let mut new_items: Vec<Expr> = Vec::new();
        let mut positions: Vec<usize> = Vec::new();
        let mut j = i + 1;
        if std::env::var("ILR_ARRDBG").is_ok() {
            eprintln!(
                "[arrdbg] init {var} items={existing_len} next={next} stmt={:?}",
                list.get(i + 1)
            );
        }
        while j < list.len() {
            let Some((idx, val, _bo, _es)) =
                try_decode_raw_pointer_store(&list[j], &var, None, None)
            else {
                // The structurer leaves an empty bounds-guard shell (and
                // occasionally comments) between element stores:
                //   if (array.Length == 1) { }   // then/else empty
                //   *(nint*)(array +40) = …
                // It has no effect, so keep scanning; a real statement still
                // stops the fold.
                if is_harmless_between(&list[j]) {
                    j += 1;
                    continue;
                }
                if std::env::var("ILR_ARRDBG").is_ok() {
                    eprintln!("[arrdbg]   stop at {j} (no store): {:?}", list[j]);
                }
                break;
            };
            if std::env::var("ILR_ARRDBG").is_ok() {
                eprintln!("[arrdbg]   store {j} idx={idx} next={next}");
            }
            if idx != next {
                break;
            }
            new_items.push(val);
            positions.push(j);
            next += 1;
            j += 1;
        }
        if new_items.is_empty() {
            i += 1;
            continue;
        }
        if let Stmt::Assign(_, _, Expr::ArrayInit { items, .. }) = &mut list[i] {
            items.extend(new_items);
        }
        for &pos in positions.iter().rev() {
            list.remove(pos);
        }
        let _ = elem_type;
        n += 1;
        i += 1;
    }
    n
}

/// Remove every provable CLR bounds guard from one statement list.
fn fold_bounds_guards(list: &mut Vec<Stmt>) -> usize {
    if list.is_empty() {
        return 0;
    }
    let mut remove = vec![false; list.len()];
    let mut n = 0;
    for i in 0..list.len() {
        if !is_bounds_guard(&list[i]) {
            continue;
        }
        // The index the guard tests must still be observed afterwards: folding
        // the guard must never remove the only reference to an index.
        let Some(idx) = guard_index_name(&list[i]) else {
            continue;
        };
        if !list[i + 1..].iter().any(|s| super::stmt_reads(s, &idx)) {
            continue;
        }
        remove[i] = true;
        n += 1;
    }
    if n > 0 {
        let mut i = 0usize;
        list.retain(|_| {
            let keep = !remove[i];
            i += 1;
            keep
        });
    }
    n
}

/// `if (cond) { throw <out-of-range>; }` with no else and a single statement.
fn is_bounds_guard(s: &Stmt) -> bool {
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = s
    else {
        return false;
    };
    if !else_body.is_empty() || then_body.len() != 1 {
        return false;
    }
    let Stmt::Throw(e) = &then_body[0] else {
        return false;
    };
    expr_is_range_exception(e) && expr_references_length(cond)
}

/// The index operand of the length comparison (`i` in `i >= arr.Length`).
fn guard_index_name(s: &Stmt) -> Option<String> {
    let Stmt::If { cond, .. } = s else {
        return None;
    };
    let Expr::Cmp(k, a, b) = cond else {
        return None;
    };
    // Only equality/ordering against a length is the CLR shape; require the
    // length on exactly one side and take the other side's local.
    if !matches!(
        k,
        CmpKind::Eq | CmpKind::Ne | CmpKind::Lt | CmpKind::Gt | CmpKind::Le | CmpKind::Ge
    ) {
        return None;
    }
    let a_len = expr_references_length(a);
    let b_len = expr_references_length(b);
    match (a_len, b_len) {
        // Exactly one side must be the length: both sides being a length is not
        // an index test.
        (false, true) => first_var(a),
        (true, false) => first_var(b),
        _ => None,
    }
}

/// Does the thrown value name an out-of-range exception?
fn expr_is_range_exception(e: &Expr) -> bool {
    match e {
        Expr::NewObj(name, _) => names_range_exception(name),
        Expr::Call { recv, name, .. } => {
            names_range_exception(name) || recv.as_deref().is_some_and(expr_is_range_exception)
        }
        Expr::RawCall(_, _, Some(ty)) => names_range_exception(ty),
        Expr::Cast(_, inner) => expr_is_range_exception(inner),
        Expr::Var(n) => names_range_exception(n),
        _ => false,
    }
}

fn names_range_exception(name: &str) -> bool {
    name.contains("OutOfRange")
}

/// Does `e` reference an array/collection length?
///
/// Includes the **raw IL2CPP spelling**: `*(nint*)(base + 24)` is
/// `Il2CppArray::max_length` (0x18) and `List<T>::_size` (also 0x18) — the only
/// length spelling the real corpus emits.
fn expr_references_length(e: &Expr) -> bool {
    match e {
        Expr::Field(_, f) => matches!(f.as_str(), "Length" | "LongLength" | "Count"),
        Expr::StaticField(_, f) => matches!(f.as_str(), "Length" | "LongLength" | "Count"),
        // The width matters here and 4 is NOT a typo. `Il2CppArray` is
        // `{ klass; monitor; bounds; max_length; }` and `max_length` is an
        // `il2cpp_array_size_t`, which the runtime defines as a `uint32_t` — so
        // the compiler loads it with a 32-bit `mov` before widening for the
        // compare, and the REAL spelling of an array length on x64 is
        // `*(int*)(a + 24)`, not `*(nint*)(a + 24)`. Accepting only 0/8 here
        // refused every 4-byte length read, i.e. most of them, once the x86
        // lifter started carrying access widths. 1 and 2 stay refused: a
        // byte/word read at 0x18 would be a different field.
        Expr::RawDeref(_, 24, w) => *w == 0 || *w == 4 || *w == 8,
        Expr::Call { recv, name, args } => {
            matches!(
                name.as_str(),
                "get_Length" | "get_LongLength" | "get_Count" | "Length" | "Count"
            ) || recv.as_deref().is_some_and(expr_references_length)
                || args.iter().any(expr_references_length)
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            expr_references_length(a) || expr_references_length(b)
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::Await(x) => {
            expr_references_length(x)
        }
        Expr::Cond(c, a, b) => {
            expr_references_length(c) || expr_references_length(a) || expr_references_length(b)
        }
        Expr::RawDeref(x, _, _) | Expr::RawAddr(x, _) | Expr::AddressOf(x) => {
            expr_references_length(x)
        }
        _ => false,
    }
}

fn first_var(e: &Expr) -> Option<String> {
    let mut out = Vec::new();
    collect_vars(e, &mut out);
    out.into_iter().next()
}

/// Collect the local names an expression reads (lambda params shadow).
fn collect_vars(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Var(n) => out.push(n.clone()),
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) | Expr::AddressOf(r) => {
            collect_vars(r, out)
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::Await(x) => collect_vars(x, out),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Cond(c, a, b) => {
            collect_vars(c, out);
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                collect_vars(r, out);
            }
            for a in args {
                collect_vars(a, out);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                collect_vars(a, out);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for a in items {
                collect_vars(a, out);
            }
        }
        Expr::CollectionInit {
            ctor_args, items, ..
        } => {
            for a in ctor_args.iter().chain(items) {
                collect_vars(a, out);
            }
        }
        Expr::Lambda { params, body } => {
            let mut inner = Vec::new();
            collect_vars(body, &mut inner);
            for v in inner {
                if !params.iter().any(|p| p == &v) {
                    out.push(v);
                }
            }
        }
        _ => {}
    }
}

/// Fold consecutive array element stores into an `ArrayInit` expression.
///
/// Scans for array allocations followed by consecutive element stores starting
/// at index 0, and replaces them with a single `ArrayInit` expression.
/// Intervening statements that do not mention the array are kept in place.
///
/// Two allocation shapes are recognized:
/// 1. `Assign(var, Some(ty), NewObj/NewArr(...))` — the high-level shape
/// 2. `Assign(var, None, RawCall(...))` — the real IL2CPP shape (allocation
///    function call, later merged with `.ctor` by normalize_ctor_calls if a
///    ctor follows; for arrays there is no ctor, so the RawCall stays)
///
/// Two element-store shapes are recognized:
/// 1. Indexed assignment: `Assign("var[N]", None, value)`
/// 2. Raw pointer write: `Assign("*(T*)(...)", None, value)` where the target
///    encodes `*(T*)(*(nint*)(var +16) + base_offset + idx*elem_size)`
fn fold_array_initializers(list: &mut Vec<Stmt>) -> usize {
    if list.is_empty() {
        return 0;
    }

    let mut n = 0;
    let mut i = 0;

    while i < list.len() {
        // Look for array allocation.
        // Shape 1: Assign(var, Some(ty), NewObj/NewArr) with array type.
        // Shape 2: Assign(var, None, RawCall(...)) — potential array allocation.
        let (var_name, elem_type, is_raw_alloc) = match &list[i] {
            Stmt::Assign(var, Some(ty), Expr::NewObj(ctor_ty, args))
                if ty.contains("[]") && args.len() == 1 =>
            {
                (var.clone(), extract_elem_type(ctor_ty), false)
            }
            Stmt::Assign(var, Some(ty), Expr::NewArr(elem_ty, args))
                if ty.contains("[]") && args.len() == 1 =>
            {
                (var.clone(), elem_ty.clone(), false)
            }
            // RawCall allocation — potential array. We don't know the element
            // type yet; it will be inferred from the element stores (if any).
            Stmt::Assign(var, _, Expr::RawCall(..)) => {
                (var.clone(), "object".to_string(), true)
            }
            Stmt::Assign(var, _, Expr::Call { .. }) => {
                // A regular call could also be an allocation function.
                (var.clone(), "object".to_string(), true)
            }
            _ => {
                i += 1;
                continue;
            }
        };

        // Collect consecutive element stores starting at index 0.
        // For RawCall allocations, we need to auto-detect base_offset and elem_size.
        let mut items = Vec::new();
        let mut store_positions = Vec::new();
        let mut next_idx = 0u64;
        let mut j = i + 1;
        let mut detected_base_offset: Option<i64> = None;
        let mut detected_elem_size: Option<u64> = None;

        while j < list.len() {
            // Try to extract (index, value) from this statement.
            // First try indexed assignment (v[N] = value).
            if let Some((idx, val)) = is_array_element_store(&list[j], &var_name) {
                if idx == next_idx {
                    items.push(val);
                    store_positions.push(j);
                    next_idx += 1;
                    j += 1;
                    continue;
                }
            }

            // Then try raw pointer stores with auto-detected parameters.
            // This handles the real IL2CPP pattern: *(nint*)(v +OFFSET) = value.
            if let Some((idx, val, base_off, elem_sz)) =
                try_decode_raw_pointer_store(&list[j], &var_name, detected_base_offset, detected_elem_size)
            {
                if idx == next_idx {
                    // Lock in the detected parameters on first success.
                    // base_offset and elem_size are tracked independently because
                    // the first store (index 0) may only reveal the base_offset
                    // (displacement=0 is divisible by all elem_sizes), while the
                    // second store reveals the actual elem_size.
                    if detected_base_offset.is_none() {
                        detected_base_offset = Some(base_off);
                    }
                    if detected_elem_size.is_none() && elem_sz > 0 {
                        detected_elem_size = Some(elem_sz);
                    }
                    items.push(val);
                    store_positions.push(j);
                    next_idx += 1;
                    j += 1;
                    continue;
                }
            }

            // Any non-store statement stops the scan — strictly consecutive.
            break;
        }

        // Only fold if we collected at least one element store.
        if items.is_empty() {
            i += 1;
            continue;
        }

        // For raw allocations with auto-detected parameters, verify we have
        // enough stores to be confident this is an array init.
        if is_raw_alloc && items.len() < 2 && detected_base_offset.is_none() {
            i += 1;
            continue;
        }

        // Replace the allocation with an ArrayInit assignment.
        let type_decl = format!("{elem_type}[]");
        list[i] = Stmt::Assign(
            var_name.clone(),
            Some(type_decl),
            Expr::ArrayInit {
                elem_type,
                items,
            },
        );

        // Remove element store statements in reverse order to keep indices valid.
        for &pos in store_positions.iter().rev() {
            list.remove(pos);
        }

        n += 1;
        i += 1;
    }

    n
}

/// Try to decode a raw pointer store to an array element, auto-detecting
/// base_offset and elem_size from the first store if they are not yet known.
///
/// Returns `(index, value, base_offset, elem_size)` on success.
fn try_decode_raw_pointer_store(
    stmt: &Stmt,
    var_name: &str,
    known_base_offset: Option<i64>,
    known_elem_size: Option<u64>,
) -> Option<(u64, Expr, i64, u64)> {
    let Stmt::Assign(target, _, value) = stmt else {
        return None;
    };

    // The target must be a dereference mentioning the variable.
    if !target.starts_with('*') {
        return None;
    }
    if !target.contains(var_name) {
        return None;
    }

    // Try to extract the total offset from the target string.
    let total_offset = extract_trailing_offset(target)?;

    // If we already know both parameters, use them directly.
    if let (Some(base_off), Some(elem_sz)) = (known_base_offset, known_elem_size) {
        if total_offset < base_off || elem_sz == 0 {
            return None;
        }
        let displacement = (total_offset - base_off) as u64;
        if displacement % elem_sz != 0 {
            return None;
        }
        let index = displacement / elem_sz;
        return Some((index as u64, value.clone(), base_off, elem_sz));
    }

    // Auto-detect: try common IL2CPP array layouts.
    // On 64-bit: base_offset=32 (16 header + 8 length + 8 padding), elem_size varies.
    // On 32-bit: base_offset=16, elem_size varies.
    // If base_offset is already known, only try that value.
    let base_offsets: &[i64] = if let Some(bo) = known_base_offset {
        // Safety: create a slice from a single value on the stack.
        // We use a small array to avoid lifetime issues.
        &[bo]
    } else {
        &[32i64, 16, 24]
    };
    for base_off in base_offsets {
        let base_off = *base_off;
        if total_offset < base_off {
            continue;
        }
        let displacement = (total_offset - base_off) as u64;
        if displacement == 0 {
            // Index 0 — can't determine elem_size yet (0 is divisible by everything).
            // Return with elem_size=0 to signal "parameters not locked in".
            // The caller will keep trying subsequent stores to determine the real size.
            return Some((0, value.clone(), base_off, 0));
        }
        // Try common element sizes: 8 (nint/object), 4 (int/float), 2 (short), 1 (byte).
        for elem_sz in [8u64, 4u64, 2u64, 1u64] {
            if displacement % elem_sz == 0 {
                let index = displacement / elem_sz;
                // Sanity check: index should be small (likely < 100 for an initializer).
                if index < 1000 {
                    return Some((index, value.clone(), base_off, elem_sz));
                }
            }
        }
    }

    None
}

/// Extract element type from array type name: `"int[]"` → `"int"`.
fn extract_elem_type(array_type: &str) -> String {
    array_type
        .find('[')
        .map_or_else(|| array_type.to_string(), |idx| array_type[..idx].to_string())
}

/// If `stmt` is an element store to `var_name`, return `(index, stored_value)`.
///
/// Only indexed assignment is recognized here: `Stmt::Assign("var[N]", None, value)`.
/// Raw pointer stores are handled by `try_decode_raw_pointer_store` which
/// tracks the detected base_offset and elem_size across consecutive stores.
fn is_array_element_store(stmt: &Stmt, var_name: &str) -> Option<(u64, Expr)> {
    let Stmt::Assign(target, _, value) = stmt else {
        return None;
    };

    // Pattern: Indexed assignment — target is "var_name[literal]"
    if let Some(idx) = parse_indexed_target(target, var_name) {
        return Some((idx, value.clone()));
    }

    None
}

/// Specialized raw pointer store detection with known base offset and element size.
///
/// Matches `Stmt::Assign(target, _, value)` where `target` encodes a raw pointer
/// write through `var_name` at `addend = base_offset + index * elem_size`.
///
/// The IL2CPP codegen produces targets like:
///   `*(nint*)(v0 +0+32)` (index 0, elem_size=4, base_offset=32)
///   `*(nint*)(v0 +4+32)` (index 1, elem_size=4, base_offset=32)
/// Checks whether `stmt` is a raw-pointer store through `var_name` at an offset
/// matching `base_offset + index * elem_size`. Returns `(index, value)` on match.
#[allow(dead_code)] // retained for future array-init reconstruction passes
fn is_raw_pointer_store_to_array(
    stmt: &Stmt,
    var_name: &str,
    base_offset: i64,
    elem_size: u64,
) -> Option<(u64, Expr)> {
    let Stmt::Assign(target, _, value) = stmt else {
        return None;
    };
    if elem_size == 0 {
        return None;
    }

    // The target string must start with `*` (dereference marker) and mention
    // the variable name.
    if !target.starts_with('*') {
        return None;
    }
    if !target.contains(var_name) {
        return None;
    }

    // Try to extract the numeric addend from the target string.
    // Look for a pattern like `+OFFSET)` at the end, where OFFSET = base_offset + idx*elem_size.
    let addend = extract_trailing_offset(target)?;

    if addend < base_offset {
        return None;
    }
    let displacement = (addend - base_offset) as u64;
    if displacement % elem_size != 0 {
        return None;
    }
    let index = displacement / elem_size;
    Some((index as u64, value.clone()))
}

/// Parse `"var_name[N]"` → `Some(N)`.
fn parse_indexed_target(target: &str, var_name: &str) -> Option<u64> {
    if !target.starts_with(var_name) {
        return None;
    }
    let rest = &target[var_name.len()..];
    if !rest.starts_with('[') || !rest.ends_with(']') || rest.len() < 3 {
        return None;
    }
    rest[1..rest.len() - 1].parse().ok()
}

/// Extract the total numeric offset from a raw pointer target string.
///
/// Parses all `+NUMBER` terms inside the outermost parentheses and sums them.
/// For example, `*(nint*)(v0 +4+32)` → 4 + 32 = 36.
fn extract_trailing_offset(target: &str) -> Option<i64> {
    // Find the content inside the outermost parens that contain the offset.
    // The pattern is `*(TYPE*)(EXPR)`, so find the last `(...)` group.
    let open = target.rfind('(')?;
    let close = target.rfind(')')?;
    if open >= close {
        return None;
    }
    let inner = &target[open + 1..close];

    // Sum all +NUMBER terms in the inner expression.
    let mut total: i64 = 0;
    for segment in inner.split('+') {
        let trimmed = segment.trim();
        if let Ok(n) = trimmed.parse::<i64>() {
            total += n;
        }
        // Non-numeric segments (like the variable name) are silently skipped.
    }
    Some(total)
}

/// Does `stmt` read or write `var_name`?
///
/// This is broader than `super::stmt_reads` — it also catches assignments
/// whose *target* is the variable (not just mentions in the value).
/// Reserved for future use when the scan relaxes the strictly-consecutive
/// requirement to allow harmless intervening statements.
#[allow(dead_code)]
fn stmt_mentions_var(stmt: &Stmt, var_name: &str) -> bool {
    match stmt {
        Stmt::Assign(target, _, value) => {
            target == var_name || super::expr_mentions(value, var_name)
        }
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::Return(Some(e)) | Stmt::YieldReturn(e) => {
            super::expr_mentions(e, var_name)
        }
        Stmt::If { cond, .. } => super::expr_mentions(cond, var_name),
        Stmt::While { cond, .. } => super::expr_mentions(cond, var_name),
        Stmt::DoWhile { cond, .. } => super::expr_mentions(cond, var_name),
        Stmt::For { cond, .. } => {
            cond.as_ref().is_some_and(|c| super::expr_mentions(c, var_name))
        }
        Stmt::Lock { expr, .. } => super::expr_mentions(expr, var_name),
        Stmt::Using { decl, .. } => stmt_mentions_var(decl, var_name),
        _ => false,
    }
}

/// Does `stmt` contain a call or allocation (side-effecting operation)?
/// Reserved for future use when the scan relaxes the strictly-consecutive
/// requirement.
#[allow(dead_code)]
fn stmt_is_call(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Eval(e) => expr_has_call(e),
        Stmt::Assign(_, _, e) => expr_has_call(e),
        _ => false,
    }
}

/// Does `e` contain a call, newobj, newarr, or rawcall?
/// Reserved for future use when the scan relaxes the strictly-consecutive
/// requirement.
#[allow(dead_code)]
fn expr_has_call(e: &Expr) -> bool {
    match e {
        Expr::Call { .. } | Expr::RawCall(..) | Expr::NewObj(..) | Expr::NewArr(..) => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_has_call(a) || expr_has_call(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::Await(x) => expr_has_call(x),
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) | Expr::AddressOf(r) => {
            expr_has_call(r)
        }
        Expr::Cond(c, a, b) => expr_has_call(c) || expr_has_call(a) || expr_has_call(b),
        Expr::ArrayInit { items, .. } => items.iter().any(expr_has_call),
        Expr::CollectionInit { ctor_args, items, .. } => {
            ctor_args.iter().any(expr_has_call) || items.iter().any(expr_has_call)
        }
        Expr::Lambda { body, .. } => expr_has_call(body),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(n: &str) -> Expr {
        Expr::Var(n.into())
    }
    fn len_of(arr: &str) -> Expr {
        Expr::Field(Box::new(Expr::Var(arr.into())), "Length".into())
    }
    /// `*(nint*)(arr +24)` — the real IL2CPP length spelling.
    fn raw_len(arr: &str) -> Expr {
        Expr::RawDeref(Box::new(Expr::Var(arr.into())), 24, 0)
    }
    /// `if (v2 >= v0.Length) throw new IndexOutOfRangeException();`
    fn guard() -> Stmt {
        Stmt::If {
            cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(len_of("v0"))),
            then_body: vec![Stmt::Throw(Expr::NewObj(
                "System.IndexOutOfRangeException".into(),
                vec![],
            ))],
            else_body: vec![],
        }
    }
    /// The real spelling of the same guard.
    fn raw_guard() -> Stmt {
        Stmt::If {
            cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(raw_len("v0"))),
            then_body: vec![Stmt::Throw(Expr::NewObj(
                "System.IndexOutOfRangeException".into(),
                vec![],
            ))],
            else_body: vec![],
        }
    }

    /// The CLR bounds guard disappears when its index is still used afterwards.
    #[test]
    fn bounds_guard_is_folded() {
        let mut list = vec![guard(), Stmt::Eval(idx("v2"))];
        assert_eq!(apply(&mut list), 1);
        assert_eq!(list.len(), 1);
        assert!(matches!(&list[0], Stmt::Eval(Expr::Var(n)) if n == "v2"));
    }

    /// The REAL spelling — a raw deref at offset 0x18 — must fold too.
    #[test]
    fn the_raw_il2cpp_length_spelling_is_folded() {
        let mut list = vec![raw_guard(), Stmt::Eval(idx("v2"))];
        assert_eq!(apply(&mut list), 1);
        assert_eq!(list.len(), 1);
    }

    /// A raw deref at a DIFFERENT offset is a field, not a length.
    #[test]
    fn a_raw_deref_at_another_offset_is_not_a_length() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ge,
                    Box::new(idx("v2")),
                    Box::new(Expr::RawDeref(Box::new(idx("v0")), 16, 0)),
                ),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// The 32-bit load of `max_length` — the spelling the x64 lifter actually
    /// produces once access widths are carried — is a length read too.
    #[test]
    fn a_32_bit_deref_at_24_is_a_length() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ge,
                    Box::new(idx("v2")),
                    Box::new(Expr::RawDeref(Box::new(idx("v0")), 24, 4)),
                ),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 1);
    }

    /// A byte-width deref at 24 is not an object's `max_length`.
    #[test]
    fn a_byte_width_deref_at_24_is_not_a_length() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ge,
                    Box::new(idx("v2")),
                    Box::new(Expr::RawDeref(Box::new(idx("v0")), 24, 1)),
                ),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A guard with no `.Length` reference is an ordinary branch — refused.
    #[test]
    fn guard_without_length_is_refused() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(Expr::Int(8))),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
        assert_eq!(list.len(), 2);
    }

    /// A non-range throw is a REAL user throw — refused.
    #[test]
    fn non_range_throw_is_refused() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(len_of("v0"))),
                then_body: vec![Stmt::Throw(Expr::NewObj("System.Exception".into(), vec![]))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// Removing the guard must not drop the index's only reference.
    #[test]
    fn guard_whose_index_is_not_used_after_is_refused() {
        let mut list = vec![guard(), Stmt::Eval(idx("v9"))];
        assert_eq!(apply(&mut list), 0);
        assert_eq!(list.len(), 2);
    }

    /// An arm with more than one statement may contain content (and labels) —
    /// folding it could orphan a jump target.
    #[test]
    fn a_multi_statement_arm_is_refused() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(len_of("v0"))),
                then_body: vec![
                    Stmt::Throw(Expr::NewObj(
                        "System.IndexOutOfRangeException".into(),
                        vec![],
                    )),
                    Stmt::Label("L".into()),
                ],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// An `else` arm means the guard has a real second path — refused.
    #[test]
    fn an_else_arm_is_refused() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(len_of("v0"))),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![Stmt::Label("L".into())],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    // ── Array initializer fold tests ──────────────────────────────────────

    /// Helper: `v0 = new int[3];` (NewObj spelling, as on the real corpus).
    fn array_alloc_newobj(var: &str, elem_ty: &str, len: i64) -> Stmt {
        Stmt::Assign(
            var.into(),
            Some(format!("{elem_ty}[]")),
            Expr::NewObj(format!("{elem_ty}[]"), vec![Expr::Int(len)]),
        )
    }

    /// Helper: `v0 = new int[3];` (NewArr spelling).
    fn array_alloc_newarr(var: &str, elem_ty: &str, len: i64) -> Stmt {
        Stmt::Assign(
            var.into(),
            Some(format!("{elem_ty}[]")),
            Expr::NewArr(elem_ty.into(), vec![Expr::Int(len)]),
        )
    }

    /// Helper: `v0[N] = value;` (indexed assignment).
    fn indexed_store(var: &str, idx: u64, value: Expr) -> Stmt {
        Stmt::Assign(format!("{var}[{idx}]"), None, value)
    }

    /// Match an ArrayInit assignment and return (elem_type, items).
    fn as_array_init(s: &Stmt) -> Option<(&str, &[Expr])> {
        match s {
            Stmt::Assign(_, _, Expr::ArrayInit { elem_type, items }) => {
                Some((elem_type, items))
            }
            _ => None,
        }
    }

    /// Basic fold: 3 consecutive indexed stores → ArrayInit.
    #[test]
    fn fold_three_indexed_stores() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 3),
            indexed_store("v0", 0, Expr::Int(10)),
            indexed_store("v0", 1, Expr::Int(20)),
            indexed_store("v0", 2, Expr::Int(30)),
        ];
        assert_eq!(apply(&mut list), 1);
        assert_eq!(list.len(), 1);
        let (ty, items) = as_array_init(&list[0]).expect("expected ArrayInit");
        assert_eq!(ty, "int");
        assert_eq!(items.len(), 3);
        assert!(matches!(items[0], Expr::Int(10)));
        assert!(matches!(items[1], Expr::Int(20)));
        assert!(matches!(items[2], Expr::Int(30)));
    }

    /// Five elements fold correctly.
    #[test]
    fn fold_five_indexed_stores() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 5),
            indexed_store("v0", 0, Expr::Int(1)),
            indexed_store("v0", 1, Expr::Int(2)),
            indexed_store("v0", 2, Expr::Int(3)),
            indexed_store("v0", 3, Expr::Int(4)),
            indexed_store("v0", 4, Expr::Int(5)),
        ];
        assert_eq!(apply(&mut list), 1);
        assert_eq!(list.len(), 1);
        let (_, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(items.len(), 5);
    }

    /// NewArr allocation is recognized the same as NewObj.
    #[test]
    fn fold_newarr_allocation() {
        let mut list = vec![
            array_alloc_newarr("v0", "float", 2),
            indexed_store("v0", 0, Expr::Float(1.0)),
            indexed_store("v0", 1, Expr::Float(2.0)),
        ];
        assert_eq!(apply(&mut list), 1);
        assert_eq!(list.len(), 1);
        let (ty, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(ty, "float");
        assert_eq!(items.len(), 2);
    }

    /// Non-consecutive indices (gap at index 2) → only the prefix folds.
    #[test]
    fn non_consecutive_indices_fold_prefix_only() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 5),
            indexed_store("v0", 0, Expr::Int(10)),
            indexed_store("v0", 1, Expr::Int(20)),
            // gap: index 2 is missing
            indexed_store("v0", 3, Expr::Int(40)),
        ];
        assert_eq!(apply(&mut list), 1);
        // The first two stores fold; the allocation becomes ArrayInit with 2 items.
        // The store at index 3 remains because index 2 was never stored.
        let (ty, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(ty, "int");
        assert_eq!(items.len(), 2);
        // The remaining statement is the store at index 3.
        assert_eq!(list.len(), 2);
        assert!(matches!(&list[1], Stmt::Assign(t, _, _) if t == "v0[3]"));
    }

    /// A call between stores stops the scan — the consecutive prefix folds.
    #[test]
    fn intervening_call_stops_scan() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 3),
            indexed_store("v0", 0, Expr::Int(10)),
            // Call between stores — stops the scan.
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "DoWork".into(),
                args: vec![],
            }),
            indexed_store("v0", 1, Expr::Int(20)),
            indexed_store("v0", 2, Expr::Int(30)),
        ];
        // The prefix (index 0) folds; the call and remaining stores are kept.
        assert_eq!(apply(&mut list), 1);
        let (_, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(list.len(), 4); // ArrayInit + call + 2 remaining stores
    }

    /// An intervening statement between stores stops the scan — only the
    /// prefix before the gap folds.
    #[test]
    fn intervening_stmt_stops_scan() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 3),
            indexed_store("v0", 0, Expr::Int(10)),
            // This assignment to a DIFFERENT variable breaks consecutiveness.
            Stmt::Assign("v1".into(), None, Expr::Int(99)),
            indexed_store("v0", 1, Expr::Int(20)),
            indexed_store("v0", 2, Expr::Int(30)),
        ];
        // Only the first store (index 0) folds — the rest are not consecutive.
        assert_eq!(apply(&mut list), 1);
        let (_, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(items.len(), 1);
        // The v1 assignment and remaining stores are kept.
        assert_eq!(list.len(), 4);
    }

    /// Reading the array between stores stops the scan — the prefix folds.
    #[test]
    fn array_read_between_stores_stops_scan() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 3),
            indexed_store("v0", 0, Expr::Int(10)),
            // Reading v0 (e.g., passing it to a field access)
            Stmt::Eval(Expr::Field(Box::new(Expr::Var("v0".into())), "Length".into())),
            indexed_store("v0", 1, Expr::Int(20)),
        ];
        // The prefix (index 0) folds; the read and remaining store are kept.
        assert_eq!(apply(&mut list), 1);
        let (_, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(list.len(), 3); // ArrayInit + read + remaining store
    }

    /// Reassigning the array variable between stores stops the scan.
    #[test]
    fn array_reassignment_between_stores_stops_scan() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 3),
            indexed_store("v0", 0, Expr::Int(10)),
            // Reassign v0 to something else
            Stmt::Assign("v0".into(), None, Expr::Null),
            indexed_store("v0", 1, Expr::Int(20)),
        ];
        // The prefix (index 0) folds; the reassignment and remaining store kept.
        assert_eq!(apply(&mut list), 1);
        let (_, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(list.len(), 3); // ArrayInit + reassignment + remaining store
    }

    /// Empty array: allocation with no element stores → no fold.
    #[test]
    fn empty_array_no_fold() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 0),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "DoWork".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut list), 0);
        assert_eq!(list.len(), 2);
    }

    /// Multiple independent arrays in one method both fold.
    #[test]
    fn multiple_arrays_both_fold() {
        let mut list = vec![
            array_alloc_newobj("v0", "int", 2),
            indexed_store("v0", 0, Expr::Int(1)),
            indexed_store("v0", 1, Expr::Int(2)),
            array_alloc_newobj("v1", "string", 2),
            indexed_store("v1", 0, Expr::Str("a".into())),
            indexed_store("v1", 1, Expr::Str("b".into())),
        ];
        assert_eq!(apply(&mut list), 2);
        assert_eq!(list.len(), 2);
        let (ty0, items0) = as_array_init(&list[0]).unwrap();
        assert_eq!(ty0, "int");
        assert_eq!(items0.len(), 2);
        let (ty1, items1) = as_array_init(&list[1]).unwrap();
        assert_eq!(ty1, "string");
        assert_eq!(items1.len(), 2);
    }

    /// The raw pointer store helper decodes the index from the offset.
    #[test]
    fn raw_pointer_store_helper_decodes_index() {
        // Target: *(nint*)(v0 +0+32)  →  trailing offset = 32
        // With base_offset=32, elem_size=4: index = (32-32)/4 = 0
        let stmt = Stmt::Assign(
            "*(nint*)(v0 +0+32)".into(),
            None,
            Expr::Int(10),
        );
        let result = is_raw_pointer_store_to_array(&stmt, "v0", 32, 4);
        assert!(result.is_some(), "expected Some, got None");
        assert_eq!(result.unwrap().0, 0);

        // Target: *(nint*)(v0 +4+32)  →  trailing offset = 36
        // With base_offset=32, elem_size=4: index = (36-32)/4 = 1
        let stmt2 = Stmt::Assign(
            "*(nint*)(v0 +4+32)".into(),
            None,
            Expr::Int(20),
        );
        let result2 = is_raw_pointer_store_to_array(&stmt2, "v0", 32, 4);
        assert!(result2.is_some(), "expected Some, got None");
        assert_eq!(result2.unwrap().0, 1);

        // Target: *(nint*)(v0 +8+32)  →  trailing offset = 40
        // With base_offset=32, elem_size=4: index = (40-32)/4 = 2
        let stmt3 = Stmt::Assign(
            "*(nint*)(v0 +8+32)".into(),
            None,
            Expr::Int(30),
        );
        let result3 = is_raw_pointer_store_to_array(&stmt3, "v0", 32, 4);
        assert!(result3.is_some(), "expected Some, got None");
        assert_eq!(result3.unwrap().0, 2);
    }

    /// The raw pointer store helper refuses a mismatched variable.
    #[test]
    fn raw_pointer_store_helper_refuses_wrong_var() {
        let stmt = Stmt::Assign(
            "*(nint*)(v1 +0+32)".into(),
            None,
            Expr::Int(10),
        );
        let result = is_raw_pointer_store_to_array(&stmt, "v0", 32, 4);
        assert!(result.is_none());
    }

    /// The raw pointer store helper refuses an offset below the base.
    #[test]
    fn raw_pointer_store_helper_refuses_offset_below_base() {
        let stmt = Stmt::Assign(
            "*(nint*)(v0 +0+16)".into(),
            None,
            Expr::Int(10),
        );
        // base_offset=32 but addend is only 16 → refused
        let result = is_raw_pointer_store_to_array(&stmt, "v0", 32, 4);
        assert!(result.is_none());
    }

    /// Bounds-check fold and array init fold coexist: both fire in one pass.
    #[test]
    fn both_folds_fire_in_one_pass() {
        let mut list = vec![
            // bounds guard (will fold)
            guard(),
            // array alloc + stores (will fold)
            array_alloc_newobj("v0", "int", 2),
            indexed_store("v0", 0, Expr::Int(1)),
            indexed_store("v0", 1, Expr::Int(2)),
            // index still used after guard removal
            Stmt::Eval(idx("v2")),
        ];
        // 1 bounds guard + 1 array init = 2 folds
        assert_eq!(apply(&mut list), 2);
        // guard removed, alloc+stores → ArrayInit, Eval kept
        assert_eq!(list.len(), 2);
        let (_, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(items.len(), 2);
    }

    // -----------------------------------------------------------------------
    // RawCall allocation + raw pointer store tests (the REAL IL2CPP shape)
    // -----------------------------------------------------------------------

    /// RawCall allocation followed by raw pointer stores with constant offsets.
    /// The real pattern: `v = fn_XXXX(...)` followed by `*(nint*)(v +OFFSET) = val`.
    #[test]
    fn rawcall_alloc_with_raw_pointer_stores() {
        let mut list = vec![
            // v0 = fn_1302484(typeInfo, 3);
            Stmt::Assign(
                "v0".into(),
                None,
                Expr::RawCall(0x1302484, vec![
                    Expr::RawDeref(Box::new(Expr::Int(0x2f6a000 + 1040)), 0, 0),
                    Expr::Int(3),
                ], None),
            ),
            // *(nint*)(v0 +0+32) = 10;  (index 0, base=32, elem_size=8)
            Stmt::Assign("*(nint*)(v0 +0+32)".into(), None, Expr::Int(10)),
            // *(nint*)(v0 +8+32) = 20;  (index 1)
            Stmt::Assign("*(nint*)(v0 +8+32)".into(), None, Expr::Int(20)),
            // *(nint*)(v0 +16+32) = 30;  (index 2)
            Stmt::Assign("*(nint*)(v0 +16+32)".into(), None, Expr::Int(30)),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "raw pointer stores should fold into ArrayInit");
        assert_eq!(list.len(), 1);
        let (ty, items) = as_array_init(&list[0]).expect("expected ArrayInit");
        assert_eq!(ty, "object"); // element type defaults to "object" for raw allocs
        assert_eq!(items.len(), 3);
    }

    /// An existing (partially folded) ArrayInit absorbs the remaining raw
    /// element stores that copy propagation has just renamed to the same var.
    #[test]
    fn existing_array_init_is_extended_by_remaining_raw_stores() {
        let mut list = vec![
            Stmt::Assign(
                "v0".into(),
                Some("object[]".into()),
                Expr::ArrayInit {
                    elem_type: "object".into(),
                    items: vec![Expr::Int(10)],
                },
            ),
            Stmt::Assign("*(nint*)(v0 +40)".into(), None, Expr::Int(20)),
            Stmt::Assign("*(nint*)(v0 +48)".into(), None, Expr::Int(30)),
        ];
        assert_eq!(apply(&mut list), 1);
        let (_, items) = as_array_init(&list[0]).expect("expected ArrayInit");
        assert_eq!(items.len(), 3);
        assert_eq!(list.len(), 1);
    }

    /// `IL2CPP_ArrayNew` Call (the rendered runtime-helper shape) with
    /// consecutive raw element stores folds like the RawCall shape.
    #[test]
    fn il2cpp_array_new_call_with_stores_folds() {
        let mut list = vec![
            Stmt::Assign(
                "v0".into(),
                None,
                Expr::Call {
                    recv: None,
                    name: "IL2CPP_ArrayNew".into(),
                    args: vec![Expr::TypeOf("X".into()), Expr::Int(3)],
                },
            ),
            Stmt::Assign(
                "*(nint*)(v0 +32)".into(),
                None,
                Expr::TypeOf("X".into()),
            ),
            Stmt::Assign(
                "*(nint*)(v0 +40)".into(),
                None,
                Expr::TypeOf("X".into()),
            ),
            Stmt::Assign(
                "*(nint*)(v0 +48)".into(),
                None,
                Expr::TypeOf("X".into()),
            ),
        ];
        assert_eq!(apply(&mut list), 1);
        let (ty, items) = as_array_init(&list[0]).expect("expected ArrayInit");
        assert_eq!(ty, "object");
        assert_eq!(items.len(), 3);
    }

    /// RawCall allocation with no element stores → no fold.
    #[test]
    fn rawcall_alloc_without_stores_no_fold() {
        let mut list = vec![
            Stmt::Assign("v0".into(), None, Expr::RawCall(0x1000, vec![], None)),
            Stmt::Eval(Expr::Var("other".into())),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 0);
        assert_eq!(list.len(), 2);
    }

    /// Raw pointer stores with elem_size=4 (int array).
    #[test]
    fn raw_pointer_stores_elem_size_4() {
        let mut list = vec![
            Stmt::Assign("v0".into(), None, Expr::RawCall(0x1000, vec![], None)),
            // base_offset=32, elem_size=4: offsets 32, 36, 40
            Stmt::Assign("*(nint*)(v0 +0+32)".into(), None, Expr::Int(100)),
            Stmt::Assign("*(nint*)(v0 +4+32)".into(), None, Expr::Int(200)),
            Stmt::Assign("*(nint*)(v0 +8+32)".into(), None, Expr::Int(300)),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1);
        let (_, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(items.len(), 3);
    }

    /// Mixed indexed and raw pointer stores: indexed stores are recognized first.
    #[test]
    fn rawcall_with_indexed_stores() {
        let mut list = vec![
            Stmt::Assign("v0".into(), None, Expr::RawCall(0x1000, vec![], None)),
            indexed_store("v0", 0, Expr::Int(10)),
            indexed_store("v0", 1, Expr::Int(20)),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1);
        let (_, items) = as_array_init(&list[0]).unwrap();
        assert_eq!(items.len(), 2);
    }
}
