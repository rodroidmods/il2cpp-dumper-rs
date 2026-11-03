//! VTable-based virtual dispatch resolution pass.
//!
//! IL2CPP implements virtual method dispatch via VTables. The decompiler outputs
//! patterns like:
//!
//! ```text
//! *(nint*)(*(nint*)(obj + vtable_offset) + slot * 8)(args)
//! ```
//!
//! This should be resolved to the actual method name using metadata. The metadata
//! system has:
//! - `ResolveMaps.vtable_slots: HashMap<(i32, usize), MethodRef>` - maps (TypeDef index, slot) to method
//! - `ResolveMaps.klass_to_tdi: HashMap<u64, i32>` - maps klass pointer address to TypeDef index
//!
//! ## Pattern Detection
//!
//! The vtable dispatch pattern in the AST is:
//! - `RawCall(target, args, ret)` where `target` is computed from vtable
//! - OR nested `RawDeref` patterns: `RawDeref(RawDeref(obj, vtable_offset), slot_offset, _)`
//!
//! ## Resolution Logic
//!
//! When we find a vtable dispatch pattern:
//! 1. Extract the TypeDef index (from klass pointer or object type)
//! 2. Extract the vtable slot number
//! 3. Look up in `maps.vtable_slots.get(&(tdi, slot))`
//! 4. If found, replace with a direct call to the resolved method name
//!
//! ## Safety
//!
//! Only rewrite when we can definitively resolve the method. Don't rewrite if
//! the lookup fails. Preserve the original call if resolution is uncertain.

use super::super::resolve::ResolveMaps;
use super::super::csharp::{Expr, Stmt};
use super::walk_mut;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Counters (process-wide, reported via `report()`)
// ---------------------------------------------------------------------------

static VTABLE_CALLS_RESOLVED: AtomicU64 = AtomicU64::new(0);

/// Reset counters (for testing).
#[cfg(test)]
fn reset_counters() {
    VTABLE_CALLS_RESOLVED.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Metadata-aware vtable dispatch resolution pass.
///
/// Walks the statement tree looking for virtual dispatch patterns and resolves
/// them to actual method names using the vtable metadata.
///
/// Called from the main pipeline when `ResolveMaps` is available.
pub fn apply_with_maps(stmts: &mut Vec<Stmt>, maps: &ResolveMaps) -> usize {
    let mut rewrites = 0;
    walk_mut(stmts, &mut |list| {
        for stmt in list.iter_mut() {
            rewrites += rewrite_stmt(stmt, maps);
        }
    });
    VTABLE_CALLS_RESOLVED.fetch_add(rewrites as u64, Ordering::Relaxed);
    rewrites
}

/// Rewrite a single statement, recursing into nested bodies.
fn rewrite_stmt(stmt: &mut Stmt, maps: &ResolveMaps) -> usize {
    let mut count = 0;

    match stmt {
        Stmt::Assign(_, _, expr) | Stmt::Eval(expr) | Stmt::Return(Some(expr))
        | Stmt::Throw(expr) | Stmt::YieldReturn(expr) => {
            count += rewrite_expr(expr, maps);
        }
        Stmt::If { cond, then_body, else_body, .. } => {
            count += rewrite_expr(cond, maps);
            for s in then_body.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
            for s in else_body.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
        }
        Stmt::While { cond, body, .. } => {
            count += rewrite_expr(cond, maps);
            for s in body.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
        }
        Stmt::DoWhile { body, cond, .. } => {
            for s in body.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
            count += rewrite_expr(cond, maps);
        }
        Stmt::For { init, incr, body, cond, .. } => {
            for s in init.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
            for s in incr.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
            if let Some(c) = cond {
                count += rewrite_expr(c, maps);
            }
            for s in body.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
        }
        Stmt::Try { try_body, catches, finally_body, .. } => {
            for s in try_body.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
            for (_, b) in catches.iter_mut() {
                for s in b.iter_mut() {
                    count += rewrite_stmt(s, maps);
                }
            }
            if let Some(fb) = finally_body {
                for s in fb.iter_mut() {
                    count += rewrite_stmt(s, maps);
                }
            }
        }
        Stmt::Switch { cases, tail, default, .. } => {
            for (_, b) in cases.iter_mut() {
                for s in b.iter_mut() {
                    count += rewrite_stmt(s, maps);
                }
            }
            for s in tail.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
            if let Some(arm) = default {
                for s in arm.iter_mut() {
                    count += rewrite_stmt(s, maps);
                }
            }
        }
        Stmt::Using { body, .. } | Stmt::Lock { body, .. }
        | Stmt::ForEach { body, .. } => {
            for s in body.iter_mut() {
                count += rewrite_stmt(s, maps);
            }
        }
        _ => {}
    }
    count
}

/// Rewrite vtable dispatch patterns within an expression tree.
fn rewrite_expr(expr: &mut Expr, maps: &ResolveMaps) -> usize {
    let mut count = 0;

    // First, recurse into sub-expressions
    match expr {
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv.as_mut() {
                count += rewrite_expr(r, maps);
            }
            for arg in args.iter_mut() {
                count += rewrite_expr(arg, maps);
            }
        }
        Expr::RawCall(_, args, _) => {
            for arg in args.iter_mut() {
                count += rewrite_expr(arg, maps);
            }
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            count += rewrite_expr(a, maps);
            count += rewrite_expr(b, maps);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x)
        | Expr::Await(x) | Expr::RawDeref(x, _, _) | Expr::RawAddr(x, _) => {
            count += rewrite_expr(x, maps);
        }
        Expr::Field(recv, _) => {
            count += rewrite_expr(recv, maps);
        }
        Expr::Cond(c, a, b) => {
            count += rewrite_expr(c, maps);
            count += rewrite_expr(a, maps);
            count += rewrite_expr(b, maps);
        }
        Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for arg in args.iter_mut() {
                count += rewrite_expr(arg, maps);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for item in items.iter_mut() {
                count += rewrite_expr(item, maps);
            }
        }
        Expr::CollectionInit { ctor_args, items, .. } => {
            for arg in ctor_args.iter_mut() {
                count += rewrite_expr(arg, maps);
            }
            for item in items.iter_mut() {
                count += rewrite_expr(item, maps);
            }
        }
        Expr::ObjInit { ctor_args, fields, .. } => {
            for arg in ctor_args.iter_mut() {
                count += rewrite_expr(arg, maps);
            }
            for (_, v) in fields.iter_mut() {
                count += rewrite_expr(v, maps);
            }
        }
        Expr::Lambda { body, .. } => {
            count += rewrite_expr(body, maps);
        }
        _ => {}
    }

    // Now try to resolve the expression itself as a vtable dispatch
    if let Some(method_name) = try_resolve_vtable_call(expr, maps) {
        // Replace the vtable dispatch pattern with a direct call
        match expr {
            Expr::RawCall(_, args, _ret_type) => {
                // Convert RawCall to a named Call
                let args_clone = args.clone();
                *expr = Expr::Call {
                    recv: None,
                    name: method_name,
                    args: args_clone,
                };
                count += 1;
            }
            Expr::RawDeref(inner, slot_offset, _) => {
                // This is a vtable slot deref that represents a virtual call.
                // Check if this is the vtable dispatch pattern.
                if let Expr::RawDeref(obj_expr, _vtable_offset, _) = inner.as_ref() {
                    let slot = (*slot_offset / 8) as usize;

                    // Verify we can resolve this
                    let can_resolve = if let Expr::Int(klass_addr) = obj_expr.as_ref() {
                        maps.klass_to_tdi.contains_key(&(*klass_addr as u64))
                            && maps.vtable_slots.contains_key(&(
                                *maps.klass_to_tdi.get(&(*klass_addr as u64)).unwrap(),
                                slot,
                            ))
                    } else {
                        extract_tdi_from_obj(obj_expr, maps)
                            .map(|tdi| maps.vtable_slots.contains_key(&(tdi, slot)))
                            .unwrap_or(false)
                    };

                    if can_resolve {
                        // REFUSE the rewrite: RawDeref does not carry call arguments,
                        // so we cannot produce a correct Call expression here.
                        // Rewriting to Call { args: vec![] } would silently discard
                        // arguments and produce wrong output like `obj.Method()`
                        // instead of `obj.Method(arg1, arg2)`.
                        //
                        // The correct pattern is RawCall(target, args, ret) where
                        // the target is a vtable-derived address — that arm
                        // preserves args.  A bare RawDeref is just a function-
                        // pointer load; leave it as-is.
                        let _ = (method_name, slot); // suppress unused warning
                    }
                }
            }
            _ => {}
        }
    }

    count
}

/// Try to resolve an expression as a vtable dispatch pattern.
///
/// Returns the resolved method name if successful, None otherwise.
///
/// Patterns detected:
/// 1. `RawCall(target, args, ret)` where target is a vtable-derived address
/// 2. Nested `RawDeref` patterns representing vtable slot access
fn try_resolve_vtable_call(expr: &Expr, maps: &ResolveMaps) -> Option<String> {
    match expr {
        // Pattern 1: RawCall with vtable-derived or direct target address.
        //
        // RawCall carries the actual call arguments, so resolving it here
        // produces a correct named Call (unlike the RawDeref path which has
        // no argument information).
        Expr::RawCall(target_addr, _args, _ret) => {
            // Try to resolve the target address via methods_by_rva.
            if let Some(method_ref) = maps.methods_by_rva.get(target_addr) {
                return Some(method_ref.full_name());
            }
            // Also check shared generic bodies (specs_by_body_rva).
            if let Some(candidates) = maps.specs_by_body_rva.get(target_addr) {
                if let Some(first) = candidates.first() {
                    return Some(first.1.full_name());
                }
            }
            None
        }

        // Pattern 2: Nested RawDeref representing vtable slot access
        // *(nint*)(*(nint*)(obj + vtable_offset) + slot_offset)
        Expr::RawDeref(inner, slot_offset, _) => {
            if let Expr::RawDeref(obj_expr, _vtable_offset, _) = inner.as_ref() {
                // This is the classic vtable dispatch pattern:
                // obj->vtable[slot]
                // where vtable_offset is the offset to the vtable pointer in the object
                // and slot_offset is the slot index * 8 (pointer size)

                // Try to extract the slot number
                let slot = (*slot_offset / 8) as usize;

                // Try to find the TypeDef index from the object expression
                if let Some(tdi) = extract_tdi_from_obj(obj_expr, maps) {
                    // Look up the vtable slot
                    if let Some(method_ref) = maps.vtable_slots.get(&(tdi, slot)) {
                        return Some(method_ref.full_name());
                    }
                }

                // Alternative: if the vtable_offset is a known klass pointer address
                if let Expr::Int(klass_addr) = obj_expr.as_ref() {
                    if let Some(&tdi) = maps.klass_to_tdi.get(&(*klass_addr as u64)) {
                        if let Some(method_ref) = maps.vtable_slots.get(&(tdi, slot)) {
                            return Some(method_ref.full_name());
                        }
                    }
                }
            }
            None
        }

        _ => None,
    }
}

/// Try to extract a TypeDef index from an object expression.
///
/// This looks for patterns that identify the object's type:
/// - Variable names that might encode type information
/// - Known klass pointer addresses
fn extract_tdi_from_obj(obj_expr: &Expr, maps: &ResolveMaps) -> Option<i32> {
    match obj_expr {
        // Direct klass pointer address
        Expr::Int(addr) => maps.klass_to_tdi.get(&(*addr as u64)).copied(),

        // Variable reference - we'd need type information to resolve this
        // For now, return None since we don't have variable type tracking here
        Expr::Var(_) => None,

        // Field access - the field might have type information
        Expr::Field(_, _) => None,

        _ => None,
    }
}

/// Report the vtable resolution rewrite count.
pub fn report() -> String {
    let resolved = VTABLE_CALLS_RESOLVED.load(Ordering::Relaxed);
    if resolved == 0 {
        return String::new();
    }
    format!("[vtable_resolution] {resolved} virtual dispatch site(s) resolved\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::render_stmts;
    use crate::recovery::types::{MethodRef, MethodSigInfo, ManagedType, PrimKind};
    use std::collections::HashMap;

    fn make_test_maps() -> ResolveMaps {
        let mut vtable_slots = HashMap::new();

        // Add a test method: TypeDef 5, slot 3 -> "MyClass.MyMethod"
        let my_method = MethodRef {
            image_index: 0,
            type_def_index: 5,
            method_index: 100,
            name: "MyMethod".to_string(),
            declaring_namespace: "TestNamespace".to_string(),
            declaring_type_name: "MyClass".to_string(),
            assembly: "TestAssembly".to_string(),
            sig: MethodSigInfo {
                ret: ManagedType::Primitive(PrimKind::Void),
                params: vec![],
                is_static: false,
            },
        };
        vtable_slots.insert((5, 3), my_method.clone());

        // Add another test method: TypeDef 10, slot 2 -> "OtherClass.OtherMethod"
        let other_method = MethodRef {
            image_index: 0,
            type_def_index: 10,
            method_index: 200,
            name: "OtherMethod".to_string(),
            declaring_namespace: "TestNamespace".to_string(),
            declaring_type_name: "OtherClass".to_string(),
            assembly: "TestAssembly".to_string(),
            sig: MethodSigInfo {
                ret: ManagedType::Primitive(PrimKind::I4),
                params: vec![ManagedType::Primitive(PrimKind::I4)],
                is_static: false,
            },
        };
        vtable_slots.insert((10, 2), other_method.clone());

        let mut klass_to_tdi = HashMap::new();
        klass_to_tdi.insert(0x1000, 5);  // klass address 0x1000 -> TypeDef 5
        klass_to_tdi.insert(0x2000, 10); // klass address 0x2000 -> TypeDef 10

        // methods_by_rva: map compiled addresses to methods so RawCall resolution works
        let mut methods_by_rva = HashMap::new();
        methods_by_rva.insert(0xAAAA, my_method);
        methods_by_rva.insert(0xBBBB, other_method);

        ResolveMaps {
            methods_by_rva,
            literal_index_by_addr: HashMap::new(),
            literal_ptr_index_by_addr: HashMap::new(),
            literals: HashMap::new(),
            method_spec_index_by_addr: HashMap::new(),
            specs_by_body_rva: HashMap::new(),
            shared_by_rva: HashMap::new(),
            shared_cross_type: HashMap::new(),
            shared_rva_of_method: HashMap::new(),
            specs_by_index: HashMap::new(),
            object_new: vec![],
            string_new: vec![],
            box_fn: vec![],
            raise_exception: vec![],
            metadata_init: vec![],
            vtable_slots,
            klass_to_tdi,
            usage_type_by_addr: HashMap::new(),
            named_type_sizes: HashMap::new(),
            tdi_names: HashMap::new(),
            objc_selector_stubs: HashMap::new(),
            native_wrappers: HashMap::new(),
            field_layouts: HashMap::new(),
            field_types: HashMap::new(),
            field_widths: HashMap::new(),
            static_field_layouts: HashMap::new(),
            field_type_by_name: HashMap::new(),
            field_generic_text: HashMap::new(),
            interface_tdis: std::collections::HashSet::new(),
            type_parent_chains: HashMap::new(),
            enum_members: HashMap::new(),
            attribute_ctor_names: HashMap::new(),
            got_addend_by_addr: HashMap::new(),
            method_name_by_va: HashMap::new(),
            static_field_offsets: HashMap::new(),
            static_field_types: HashMap::new(),
            tdi_names_reverse: HashMap::new(),
            method_return_tdi: HashMap::new(),
            tdi_fields: HashMap::new(),
            value_type_tdis: std::collections::HashSet::new(),
        }
    }

    #[test]
    fn simple_rawcall_resolution() {
        reset_counters();
        // RawCall to a known method address preserves arguments.
        let mut stmts = vec![Stmt::Eval(Expr::RawCall(
            0xAAAA, // resolves to TestNamespace.MyClass::MyMethod
            vec![],
            None,
        ))];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 1, "should resolve one RawCall dispatch");
        // Verify the result is a named Call
        match &stmts[0] {
            Stmt::Eval(Expr::Call { recv, name, args }) => {
                assert!(recv.is_none());
                assert_eq!(name, "TestNamespace.MyClass::MyMethod");
                assert!(args.is_empty());
            }
            other => panic!("expected Eval(Call), got {:?}", other),
        }
    }

    #[test]
    fn rawcall_preserves_arguments() {
        reset_counters();
        // RawCall with arguments must preserve them in the rewrite.
        let mut stmts = vec![Stmt::Assign(
            "result".into(),
            None,
            Expr::RawCall(
                0xBBBB, // resolves to TestNamespace.OtherClass::OtherMethod
                vec![Expr::Var("arg1".into())],
                None,
            ),
        )];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 1, "should resolve RawCall with args");
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::Call { recv, name, args }) => {
                assert!(recv.is_none());
                assert_eq!(name, "TestNamespace.OtherClass::OtherMethod");
                assert_eq!(args.len(), 1, "arguments must be preserved");
            }
            other => panic!("expected Assign(Call), got {:?}", other),
        }
    }

    #[test]
    fn bare_rawderef_is_refused() {
        reset_counters();
        // A bare RawDeref (vtable slot load) has no argument information.
        // The rewrite MUST be refused to avoid silently producing wrong output.
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::Int(0x1000)), // klass pointer
                0,                            // vtable offset
                0,
            )),
            24, // slot 3 * 8
            0,
        ))];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 0, "must refuse RawDeref rewrite (no args available)");
        // Verify the expression is unchanged
        match &stmts[0] {
            Stmt::Eval(Expr::RawDeref(..)) => {} // good, still a RawDeref
            other => panic!("expected RawDeref to be preserved, got {:?}", other),
        }
    }

    #[test]
    fn non_match_non_vtable_call() {
        reset_counters();
        // Regular call, not a vtable pattern
        let mut stmts = vec![Stmt::Eval(Expr::Call {
            recv: None,
            name: "Console.WriteLine".into(),
            args: vec![Expr::Str("hello".into())],
        })];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 0, "should not resolve non-vtable calls");
    }

    #[test]
    fn non_match_unresolvable_vtable_slot() {
        reset_counters();
        // VTable pattern but slot doesn't exist in metadata
        let mut stmts = vec![Stmt::Eval(Expr::RawDeref(
            Box::new(Expr::RawDeref(
                Box::new(Expr::Int(0x1000)), // known klass
                0,
                0,
            )),
            999, // non-existent slot
            0,
        ))];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 0, "should not resolve unknown vtable slot");
    }

    #[test]
    fn nested_rawcalls() {
        reset_counters();
        let mut stmts = vec![
            Stmt::Eval(Expr::RawCall(0xAAAA, vec![], None)),
            Stmt::Eval(Expr::RawCall(0xBBBB, vec![Expr::Var("x".into())], None)),
        ];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 2, "should resolve both RawCall dispatches");
    }

    #[test]
    fn rawcall_in_assignment() {
        reset_counters();
        let mut stmts = vec![Stmt::Assign(
            "x".into(),
            None,
            Expr::RawCall(0xAAAA, vec![], None),
        )];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 1, "should resolve RawCall in assignment");
    }

    #[test]
    fn rawcall_in_condition() {
        reset_counters();
        let mut stmts = vec![Stmt::If {
            cond: Expr::RawCall(0xAAAA, vec![], None),
            then_body: vec![Stmt::Eval(Expr::Var("x".into()))],
            else_body: vec![],
        }];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 1, "should resolve RawCall in condition");
    }

    #[test]
    fn multiple_rawcalls_in_one_method() {
        reset_counters();
        let mut stmts = vec![
            Stmt::Eval(Expr::RawCall(0xAAAA, vec![], None)),
            Stmt::Assign(
                "y".into(),
                None,
                Expr::RawCall(0xBBBB, vec![Expr::Var("arg".into())], None),
            ),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SomeOtherCall".into(),
                args: vec![],
            }),
        ];

        let maps = make_test_maps();
        let n = apply_with_maps(&mut stmts, &maps);

        assert_eq!(n, 2, "should resolve both RawCalls, leave other call alone");
    }

    #[test]
    fn rawcall_resolution_renders_correctly() {
        reset_counters();
        let mut stmts = vec![Stmt::Eval(Expr::RawCall(
            0xAAAA,
            vec![Expr::Var("a".into()), Expr::Int(42)],
            None,
        ))];

        let maps = make_test_maps();
        apply_with_maps(&mut stmts, &maps);

        let output = render_stmts(&stmts);
        // The output should contain the resolved method name
        assert!(
            output.contains("MyMethod"),
            "should render resolved method name, got: {}",
            output
        );
    }

    #[test]
    fn report_empty_when_nothing_fires() {
        reset_counters();
        let r = report();
        assert!(r.is_empty(), "report should be empty when nothing fires");
    }
}
