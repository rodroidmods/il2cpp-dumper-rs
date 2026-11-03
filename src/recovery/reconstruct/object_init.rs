//! Object initializer recognition.
//!
//! ## Pattern
//!
//! Detects when a newly-created object has its fields immediately set, and
//! folds those stores into an object initializer expression:
//!
//! ```text
//! v = new T(args);          // Stmt::Assign(v, ty, Expr::NewObj(type, ctor_args))
//! v.Field1 = e1;            // Stmt::Assign("v.Field1", None, e1)
//! v.Field2 = e2;            // Stmt::Assign("v.Field2", None, e2)
//! ```
//!
//! becomes:
//!
//! ```text
//! v = new T(args) { Field1 = e1, Field2 = e2 };
//! ```
//!
//! ## How field stores appear in the IR
//!
//! The lifter renders a resolved field store `recv.field = value` as
//! `Stmt::Assign(rendered_target, None, value)` where `rendered_target` is the
//! rendered string `"recv.field"` (e.g. `"v.Field1"` or `"this._size"`). This
//! is because the IR lowering in `csharp.rs` calls `field_expr(recv, f).render()`
//! to produce the assignment target string.
//!
//! We detect field stores to a local `v` by checking whether the target string
//! starts with `v.` (the variable name followed by a dot).
//!
//! ## Safety conditions
//!
//! The run of hoistable field stores ENDS (fold the prefix collected so far)
//! at the first statement that:
//!
//! 1. Mentions `v` at any depth in a non-field-store position — a read, a
//!    call argument, or a direct write. Statements after that point observe
//!    the initialized object and must stay where they are.
//! 2. Is a `Label` or a control-flow exit: a jump can arrive after the
//!    allocation without running it, so the run must not span the target.
//! 3. Is a field store whose value reads `v`, contains a call, or reads a
//!    variable defined by a statement the fold would hoist over.
//!
//! Statements in between that do not mention `v` are transparent only when
//! they cannot change anything a hoisted field value reads: no calls at any
//! depth (a call can run a class initializer or write statics), and
//! assignments only define or mutate fresh locals introduced by those same
//! skipped statements (`hoistable_over` below). Field values are additionally
//! never hoisted above definitions they read, so the reordering cannot change
//! a value.
//!
//! A failed match is a NO-OP: the flat sequence stays.

use std::collections::HashSet;

use super::super::csharp::{Expr, Stmt};

/// Entry point: fold object initializers in one statement list, returning the
/// number of rewrites applied.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    super::walk_mut(stmts, &mut |list| {
        n += fold_object_inits(list);
    });
    n
}

/// Fold every object initializer pattern in one statement list.
fn fold_object_inits(list: &mut Vec<Stmt>) -> usize {
    if list.len() < 2 {
        return 0;
    }
    let mut count = 0;
    let mut i = 0;
    while i < list.len() {
        // Look for: Stmt::Assign(v, ty, Expr::NewObj(type_name, ctor_args)).
        // Only a plain local is a trigger: a field target such as `this._cache`
        // renders as a string, while uses of it are `Expr::Field` nodes that
        // `expr_mentions` cannot match against that string — the run would
        // hoist stores past an observing read.
        let (var_name, type_name, ctor_args) = match &list[i] {
            Stmt::Assign(v, _ty, Expr::NewObj(ty, args)) if is_plain_name(v) => {
                (v.clone(), ty.clone(), args.clone())
            }
            _ => {
                i += 1;
                continue;
            }
        };

        // Collect the hoistable field-store prefix of this allocation. The
        // run ends at the first statement that uses the object or that the
        // stores cannot be reordered across; the prefix collected until then
        // is still foldable.
        let mut fields: Vec<(String, Expr)> = Vec::new();
        let mut field_indices: Vec<usize> = Vec::new();
        let mut skip_defs: HashSet<String> = HashSet::new();
        let mut j = i + 1;

        while j < list.len() {
            // Check for field store to var_name.
            if let Some((field_name, value)) = is_field_store_to(&list[j], &var_name) {
                // The value must not read `v` (self-referential init is
                // unsafe), must not contain calls (a call could observe the
                // partially-initialized object), and must not read a variable
                // defined by a statement the fold hoists over (its evaluation
                // would move before that definition).
                if stmt_reads_var(&list[j], &var_name)
                    || super::expr_has_call(&value)
                    || value_reads_any(&value, &skip_defs)
                {
                    break;
                }
                fields.push((field_name, value));
                field_indices.push(j);
                j += 1;
                continue;
            }

            // Labels are jump targets — a goto can land after the allocation
            // without running it, so the run cannot span them. Control-flow
            // exits end the linear region the same way.
            if matches!(&list[j], Stmt::Label(_)) || super::leaves_region(&list[j]) {
                break;
            }

            // Any mention of `v` at any depth (in a non-field-store context)
            // means `v` is observed or modified: the fold covers only the
            // field stores before that point.
            if stmt_mentions_deep(&list[j], &var_name) {
                break;
            }

            // The statement is transparent only if it cannot observe the
            // object AND cannot change anything a hoisted field value reads:
            // no calls (a call can run a class initializer or mutate
            // statics), and assignments only define or mutate fresh locals
            // introduced by earlier skipped statements.
            if !hoistable_over(&list[j], &mut skip_defs) {
                break;
            }
            j += 1;
        }

        if fields.is_empty() {
            i += 1;
            continue;
        }

        // Build the ObjInit expression.
        let obj_init = Expr::ObjInit {
            type_name,
            ctor_args,
            fields,
        };

        // Replace the allocation with the ObjInit.
        list[i] = Stmt::Assign(var_name, None, obj_init);

        // Remove the field store statements (iterate in reverse to keep
        // indices valid).
        for &idx in field_indices.iter().rev() {
            list.remove(idx);
        }

        count += 1;
        i += 1;
    }
    count
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// If `stmt` is a field store to `var_name` (i.e. `Stmt::Assign(target, _, value)`
/// where `target` starts with `"var_name."`), return `(field_name, value)`.
///
/// Field stores in the IR are rendered as `Stmt::Assign(rendered_target, None, value)`
/// where `rendered_target` is the string form of the field access (e.g.
/// `"v.Field1"` or `"this._size"`). We detect stores to a specific variable by
/// checking the target string prefix.
fn is_field_store_to(stmt: &Stmt, var_name: &str) -> Option<(String, Expr)> {
    let Stmt::Assign(target, _, value) = stmt else {
        return None;
    };
    let prefix = format!("{var_name}.");
    if let Some(field_name) = target.strip_prefix(&prefix) {
        // An object initializer assigns one simple member at a time:
        // `v.Inner.X = …` has no initializer spelling and must end the run
        // (the statement itself still mentions `v` through `target_names`).
        if is_plain_name(field_name) {
            return Some((field_name.to_string(), value.clone()));
        }
    }
    None
}

/// Is `s` a single C# identifier (a local variable or a plain field name,
/// no dots, brackets, or punctuation)?
fn is_plain_name(s: &str) -> bool {
    let mut chars = s.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic());
    first_ok && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// Does this statement read `var_name`? Only checks the value expression of
/// assignments, not the target (so a field store `v.Field = expr` reports
/// whether `expr` reads `v`, not whether the target mentions `v`).
fn stmt_reads_var(stmt: &Stmt, var_name: &str) -> bool {
    match stmt {
        Stmt::Assign(_, _, value) => super::expr_mentions(value, var_name),
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => {
            super::expr_mentions(e, var_name)
        }
        Stmt::Return(Some(e)) => super::expr_mentions(e, var_name),
        Stmt::If { cond, .. } => super::expr_mentions(cond, var_name),
        Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            super::expr_mentions(cond, var_name)
        }
        Stmt::Lock { expr, .. } => super::expr_mentions(expr, var_name),
        _ => false,
    }
}

/// Does this statement write `var_name`? Only a direct assignment to the exact
/// name counts — a field store `"v.Field1"` is NOT a write to `"v"`.
fn _stmt_writes_var(stmt: &Stmt, var_name: &str) -> bool {
    match stmt {
        Stmt::Assign(target, _, _) => target == var_name,
        _ => false,
    }
}

/// Does this statement mention `var_name` at any depth (including nested
/// bodies)? Includes reads, writes, and field stores to `v`.
fn stmt_mentions_deep(stmt: &Stmt, var_name: &str) -> bool {
    if super::stmt_reads(stmt, var_name) || super::stmt_writes(stmt, var_name) {
        return true;
    }
    // Check for field stores: target starts with "v."
    if let Stmt::Assign(target, _, _) = stmt {
        let prefix = format!("{var_name}.");
        if target.starts_with(&prefix) {
            return true;
        }
    }
    // Check Switch discriminant.
    if let Stmt::Switch { disc, .. } = stmt {
        if disc == var_name {
            return true;
        }
    }
    // Check ForEach variable/collection.
    if let Stmt::ForEach {
        var_name: vn,
        collection,
        ..
    } = stmt
    {
        if vn == var_name || collection == var_name {
            return true;
        }
    }
    // Recurse into nested statement lists.
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(stmt, &mut kids);
    kids.iter().any(|k| k.iter().any(|s| stmt_mentions_deep(s, var_name)))
}

/// Does `value` read any of `names`?
fn value_reads_any(value: &Expr, names: &HashSet<String>) -> bool {
    names.iter().any(|n| super::expr_mentions(value, n))
}

/// Can the fold hoist field stores over `stmt`? Only over statements that
/// cannot change anything a field value may read: no calls at any depth, and
/// every assignment must define a fresh local or mutate one of the fresh
/// locals an earlier skipped statement introduced (`skip_defs`). A write to
/// any outer location (a `this` field, an outer local's field, raw memory,
/// an event, a nested body) ends the run.
fn hoistable_over(stmt: &Stmt, skip_defs: &mut HashSet<String>) -> bool {
    match stmt {
        Stmt::Assign(target, _, value) => {
            if super::expr_has_call(value) {
                return false;
            }
            let t = target.trim();
            if is_plain_name(t) {
                skip_defs.insert(t.to_string());
                return true;
            }
            base_of_target(t).is_some_and(|b| skip_defs.contains(b))
        }
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            !super::expr_has_call(collection)
                && !super::expr_has_call(index)
                && !super::expr_has_call(value)
                && matches!(collection, Expr::Var(b) if skip_defs.contains(b))
        }
        Stmt::Declare(name, _) => {
            skip_defs.insert(name.clone());
            true
        }
        _ => !stmt_has_call(stmt) && !stmt_has_assign(stmt),
    }
}

/// The base local of an assignment target string (`q[0]` / `q.F` → `q`), or
/// `None` when the target is not rooted at a plain identifier (raw memory,
/// `this.…`, a call result).
fn base_of_target(target: &str) -> Option<&str> {
    let t = target.trim();
    let mut end = 0;
    for (i, c) in t.char_indices() {
        if i == 0 {
            if c != '_' && !c.is_ascii_alphabetic() {
                return None;
            }
        } else if c != '_' && !c.is_ascii_alphanumeric() {
            break;
        }
        end = i + c.len_utf8();
    }
    (end > 0).then(|| &t[..end])
}

/// Does this statement evaluate a call, at any depth?
fn stmt_has_call(stmt: &Stmt) -> bool {
    let direct = match stmt {
        Stmt::Assign(_, _, e)
        | Stmt::Eval(e)
        | Stmt::Throw(e)
        | Stmt::YieldReturn(e) => super::expr_has_call(e),
        Stmt::Return(Some(e)) => super::expr_has_call(e),
        Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
            super::expr_has_call(cond)
        }
        Stmt::Lock { expr, .. } => super::expr_has_call(expr),
        Stmt::For { cond, .. } => cond.as_ref().is_some_and(|c| super::expr_has_call(c)),
        Stmt::StringSwitch { disc, .. } => super::expr_has_call(disc),
        Stmt::IndexAssign {
            collection,
            index,
            value,
        } => {
            super::expr_has_call(collection)
                || super::expr_has_call(index)
                || super::expr_has_call(value)
        }
        _ => false,
    };
    if direct {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(stmt, &mut kids);
    kids.iter().any(|k| k.iter().any(stmt_has_call))
}

/// Does this statement assign anything, at any depth?
fn stmt_has_assign(stmt: &Stmt) -> bool {
    if matches!(
        stmt,
        Stmt::Assign(..) | Stmt::IndexAssign { .. } | Stmt::EventAssign { .. } | Stmt::Declare(..)
    ) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    super::children(stmt, &mut kids);
    kids.iter().any(|k| k.iter().any(stmt_has_assign))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build `v = new Type(ctor_args)` with a type declaration.
    fn new_obj_assign(var: &str, type_name: &str, ctor_args: Vec<Expr>) -> Stmt {
        Stmt::Assign(
            var.into(),
            Some(format!("var")),
            Expr::NewObj(type_name.into(), ctor_args),
        )
    }

    /// Helper: build `v.FieldName = value` (a field store).
    fn field_store(var: &str, field: &str, value: Expr) -> Stmt {
        Stmt::Assign(format!("{var}.{field}"), None, value)
    }

    /// Helper: render an expression to string.
    fn render_expr(e: &Expr) -> String {
        let mut out = String::new();
        e.render(&mut out);
        out
    }

    // -----------------------------------------------------------------------
    // Basic recognition tests
    // -----------------------------------------------------------------------

    /// Basic object initializer with 2 fields.
    #[test]
    fn basic_object_init_with_two_fields() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Int(1)),
            field_store("v", "Y", Expr::Int(2)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(name, _, Expr::ObjInit { type_name, ctor_args, fields }) => {
                assert_eq!(name, "v");
                assert_eq!(type_name, "Point");
                assert!(ctor_args.is_empty());
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "X");
                assert_eq!(fields[1].0, "Y");
            }
            other => panic!("expected Assign(_, _, ObjInit), got {other:?}"),
        }
    }

    /// Object initializer with 5 fields.
    #[test]
    fn object_init_with_five_fields() {
        let mut stmts = vec![
            new_obj_assign("obj", "Config", vec![Expr::Str("name".into())]),
            field_store("obj", "Width", Expr::Int(100)),
            field_store("obj", "Height", Expr::Int(200)),
            field_store("obj", "Depth", Expr::Int(3)),
            field_store("obj", "Label", Expr::Str("test".into())),
            field_store("obj", "Visible", Expr::Bool(true)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::ObjInit { type_name, ctor_args, fields }) => {
                assert_eq!(type_name, "Config");
                assert_eq!(ctor_args.len(), 1);
                assert_eq!(fields.len(), 5);
                assert_eq!(fields[0].0, "Width");
                assert_eq!(fields[4].0, "Visible");
            }
            other => panic!("expected Assign(_, _, ObjInit), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Safety condition tests
    // -----------------------------------------------------------------------

    /// Intervening read of the variable prevents folding.
    #[test]
    fn intervening_read_stops_the_fold_at_the_read() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Int(1)),
            // Intervening read: `x = v.Y;`
            Stmt::Assign(
                "x".into(),
                None,
                Expr::Field(Box::new(Expr::Var("v".into())), "Y".into()),
            ),
            field_store("v", "Y", Expr::Int(2)),
        ];
        let n = apply(&mut stmts);
        // The read observes the object, so the run ends there and only the
        // stores before it are folded.
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::ObjInit { .. })));
        assert!(matches!(&stmts[2], Stmt::Assign(t, _, _) if t == "v.Y"));
    }

    /// Calls that read `v` between allocation and field stores prevent folding.
    #[test]
    fn calls_reading_v_prevent_folding() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            // Intervening call that reads v: `Console.WriteLine(v);`
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Console.WriteLine".into(),
                args: vec![Expr::Var("v".into())],
            }),
            field_store("v", "X", Expr::Int(1)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 3);
    }

    /// A call (even one that does not mention v) is a boundary: it can run a
    /// class initializer or mutate statics that a hoisted field value reads.
    #[test]
    fn calls_end_the_run() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            // Unrelated call: `Console.WriteLine("init");`
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Console.WriteLine".into(),
                args: vec![Expr::Str("init".into())],
            }),
            field_store("v", "X", Expr::Int(1)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 3);
    }

    /// Field store whose value reads v prevents folding.
    #[test]
    fn self_referential_field_store_prevents_folding() {
        let mut stmts = vec![
            new_obj_assign("v", "Node", vec![]),
            // v.Next = v; — the value reads v
            field_store("v", "Next", Expr::Var("v".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 2);
    }

    /// Field store whose value contains a call prevents folding.
    #[test]
    fn field_store_value_with_call_prevents_folding() {
        let mut stmts = vec![
            new_obj_assign("v", "Widget", vec![]),
            field_store(
                "v",
                "Name",
                Expr::Call {
                    recv: None,
                    name: "GetName".into(),
                    args: vec![],
                },
            ),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Transparent statement tests
    // -----------------------------------------------------------------------

    /// Comments between field stores are transparent.
    #[test]
    fn comments_are_transparent() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Int(1)),
            Stmt::Comment("set Y coordinate".into()),
            field_store("v", "Y", Expr::Int(2)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        // The comment remains, but the field stores are folded.
        // Result: Assign(v, ObjInit), Comment
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::ObjInit { .. })));
        assert!(matches!(&stmts[1], Stmt::Comment(_)));
    }

    /// Unrelated statements (not mentioning v, no calls) are transparent.
    #[test]
    fn unrelated_statements_are_transparent() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Int(1)),
            // Unrelated: `x = 42;`
            Stmt::Assign("x".into(), None, Expr::Int(42)),
            field_store("v", "Y", Expr::Int(2)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::ObjInit { .. })));
    }

    // -----------------------------------------------------------------------
    // Label and control flow tests
    // -----------------------------------------------------------------------

    /// Labels between field stores prevent folding.
    #[test]
    fn labels_stop_the_fold_before_the_label() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Int(1)),
            Stmt::Label("L1".into()),
            field_store("v", "Y", Expr::Int(2)),
        ];
        // A jump can land at the label without running the allocation: the
        // stores after it must stay visible on that path.
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[1], Stmt::Label(l) if l == "L1"));
    }

    /// Goto between field stores prevents folding.
    #[test]
    fn goto_stops_the_fold_before_the_exit() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Int(1)),
            Stmt::Goto("L1".into()),
            field_store("v", "Y", Expr::Int(2)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[1], Stmt::Goto(l) if l == "L1"));
    }

    // -----------------------------------------------------------------------
    // Use after the run
    // -----------------------------------------------------------------------

    /// A use after the last field store is the normal shape: the object is
    /// initialized first and only then consumed, so it does not block the fold.
    #[test]
    fn a_use_after_the_stores_does_not_block_the_fold() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Int(1)),
            field_store("v", "Y", Expr::Int(2)),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "Take".into(),
                args: vec![Expr::Var("v".into())],
            }),
            field_store("v", "Z", Expr::Int(3)),
        ];
        let n = apply(&mut stmts);
        // The call consumes the fully-initialized object; the later store
        // must not be hoisted above the call.
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 3);
        assert!(matches!(&stmts[2], Stmt::Assign(t, _, _) if t == "v.Z"));
    }

    /// A field value reading a variable defined by a statement the fold would
    /// hoist over must not be folded: evaluating it at the allocation site
    /// would read the previous value.
    #[test]
    fn a_field_value_may_not_read_a_hoisted_over_definition() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            // Skipped statement defines `x`.
            new_obj_assign("x", "Size", vec![]),
            field_store("v", "X", Expr::Var("x".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 3);
    }

    /// The same value is fine when the variable was defined before the
    /// allocation: nothing moves past its definition.
    #[test]
    fn a_field_value_defined_before_the_allocation_still_folds() {
        let mut stmts = vec![
            new_obj_assign("x", "Size", vec![]),
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Var("x".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[1], Stmt::Assign(_, _, Expr::ObjInit { .. })));
    }

    // -----------------------------------------------------------------------
    // Nested pattern test
    // -----------------------------------------------------------------------

    /// Object initializer nested inside an if-body is still found.
    #[test]
    fn nested_in_if_body() {
        let mut stmts = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![
                new_obj_assign("v", "Point", vec![]),
                field_store("v", "X", Expr::Int(1)),
                field_store("v", "Y", Expr::Int(2)),
            ],
            else_body: vec![],
        }];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        match &stmts[0] {
            Stmt::If { then_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert!(matches!(
                    &then_body[0],
                    Stmt::Assign(_, _, Expr::ObjInit { fields, .. }) if fields.len() == 2
                ));
            }
            _ => panic!("expected If"),
        }
    }

    // -----------------------------------------------------------------------
    // Rendering tests
    // -----------------------------------------------------------------------

    /// Render a basic object initializer.
    #[test]
    fn render_object_init() {
        let e = Expr::ObjInit {
            type_name: "Point".into(),
            ctor_args: vec![],
            fields: vec![
                ("X".into(), Expr::Int(1)),
                ("Y".into(), Expr::Int(2)),
            ],
        };
        assert_eq!(render_expr(&e), "new Point() { X = 1, Y = 2 }");
    }

    /// Render with constructor arguments.
    #[test]
    fn render_object_init_with_ctor_args() {
        let e = Expr::ObjInit {
            type_name: "Config".into(),
            ctor_args: vec![Expr::Str("name".into())],
            fields: vec![
                ("Width".into(), Expr::Int(100)),
                ("Visible".into(), Expr::Bool(true)),
            ],
        };
        assert_eq!(
            render_expr(&e),
            "new Config(\"name\") { Width = 100, Visible = true }"
        );
    }

    // -----------------------------------------------------------------------
    // Multiple patterns in one method
    // -----------------------------------------------------------------------

    /// Multiple object initializers in one method are each folded independently.
    #[test]
    fn multiple_object_inits_in_one_method() {
        let mut stmts = vec![
            new_obj_assign("a", "Point", vec![]),
            field_store("a", "X", Expr::Int(1)),
            field_store("a", "Y", Expr::Int(2)),
            new_obj_assign("b", "Size", vec![]),
            field_store("b", "W", Expr::Int(10)),
            field_store("b", "H", Expr::Int(20)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 2);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Assign(name, _, Expr::ObjInit { .. }) if name == "a"));
        assert!(matches!(&stmts[1], Stmt::Assign(name, _, Expr::ObjInit { .. }) if name == "b"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    /// No field stores → no fold (bare NewObj stays).
    #[test]
    fn no_field_stores_is_not_folded() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            Stmt::Eval(Expr::Var("other".into())),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 2);
        assert!(matches!(&stmts[0], Stmt::Assign(_, _, Expr::NewObj(..))));
    }

    /// Field stores on a DIFFERENT variable are not folded.
    #[test]
    fn field_store_on_different_variable() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            // Field store on "w", not "v"
            field_store("w", "X", Expr::Int(1)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 0);
        assert_eq!(stmts.len(), 2);
    }

    /// A single field store is folded (minimum viable pattern).
    #[test]
    fn single_field_store_is_folded() {
        let mut stmts = vec![
            new_obj_assign("v", "Point", vec![]),
            field_store("v", "X", Expr::Int(42)),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1);
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Stmt::Assign(_, _, Expr::ObjInit { fields, .. }) => {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].0, "X");
            }
            other => panic!("expected ObjInit, got {other:?}"),
        }
    }
}
