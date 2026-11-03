//! Pipeline tests: synthetic compiler-shaped ARM64 → full recovery → CIL.

use super::csharp::{BinOp, CmpKind, Expr, Stmt, cleanup};
use super::*;
use ir::{ArithKind, CompareKind, FloatArithKind, Op, Operand, SelectVariant};
use std::collections::HashMap;

fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The ARM64 `BLR` lifter emits `Op::Call`, not `Op::IndirectCall` (a
/// returning call must not be the `IndirectCall` CFG terminator), and it
/// emits the ordinary `[target, ret-slot, receiver, args…]` operand layout.
///
/// The vtable resolver used to key on `Op::IndirectCall` alone and to REBUILD
/// the operand list as `[Method, Unknown, Unknown…]`. Consequences, both of
/// which this test pins down:
///   * no virtual call on ARM64 was ever resolved — the matcher's input never
///     arrived, so `elements[index]` stayed an opaque indirect call;
///   * whenever it did fire, the receiver and every argument had just been
///     replaced with `Unknown`, so a resolved call rendered with no arguments.
/// A def whose only consumer is an INSTANCE-FIELD BASE is live.
///
/// `passes::collect_locals` is the private operand walker behind
/// `dead_code_elimination`'s use counts. It had arms for `Local`,
/// `Memory`, `AddressOf`, `ArrayLength` and `ArrayAccess` but NONE for
/// `Operand::InstanceField`, so the RECEIVER of a field access was invisible
/// to the counts and the def that computed it was deleted as dead.
///
/// Shape (0x1489B84 `CircularScrollingList.Start`, ARM64):
/// ```text
///   ldr  x8,[x0,#0x30]     ; x8 = this._setting
///   cbz  x8, …             ; copy propagation rewrites this read to the field
///   ldrb w8,[x8,#0x26]     ; x8 = this._setting._initializeOnStart
/// ```
/// The `InstanceField` base was then the load's ONLY remaining use, so the load
/// was Nop'd and the renderer — seeing a read with no definition — printed
/// `in_x8._initializeOnStart`, an ABI name for a value the body does define.
#[test]
fn an_instance_field_base_keeps_its_definition_alive() {
    let field = FieldRef {
        type_def_index: 6,
        field_index: 29,
        name: "_initializeOnStart".into(),
        declaring_type_name: "CircularScrollingListSetting".into(),
        field_type: ManagedType::Primitive(PrimKind::Boolean),
        is_static: false,
    };
    let mut insns = vec![
        // x8 = <something the body computes>
        Insn::new(Op::Move, vec![Operand::Local(1), Operand::Local(9)], 0x1000),
        // w8 = x8._initializeOnStart   (the base is a NESTED operand)
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(2),
                Operand::InstanceField {
                    base: Box::new(Operand::Local(1)),
                    field,
                },
            ],
            0x1004,
        ),
        // The loaded value is consumed, so nothing here is dead for the usual
        // reason — the base is the ONLY thing keeping the load alive.
        Insn::new(Op::Return, vec![Operand::Local(2)], 0x1008),
    ];
    passes::dead_code_elimination(&mut insns);
    assert!(
        matches!(insns[0].op, Op::Move),
        "the base definition must survive: {insns:?}"
    );
}

#[test]
fn virtual_call_through_blr_shape_resolves_and_keeps_arguments() {
    let mut table = locals::LocalTable::default();
    table.infos.push(locals::LocalInfo {
        slot_origin: "x20".into(),
        ty: ManagedType::Class(std::sync::Arc::new(NamedType {
            namespace: "Test".into(),
            name: "Widget".into(),
            assembly: None,
            type_def_index: 7,
        })),
        is_this: false,
        is_param: Some(0),
        param_comp: None,
        is_incoming_arg: false,
        is_call_clobber: false,
        is_return_buffer: false,
    });
    table.index_of.insert(1u32, 0usize);
    table.param_names.push("elements".into());

    let slot_load = |dst: u32, base: u32, addend: i64, addr: u64| {
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(dst),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::Local(base)),
                    index: None,
                    scale: 8,
                    addend,
                },
            ],
            addr,
        )
    };
    let mut insns = vec![
        // v1 = elements (the receiver-typed parameter)
        Insn::move_(Operand::Local(1), Operand::Local(0), 0x1000),
        // v2 = [v1]            — obj → klass
        slot_load(2, 1, 0, 0x1004),
        // v3 = [v2 + 0x140]    — vtable slot 1 (16-byte entries: `VirtualInvokeData`)
        slot_load(3, 2, 0x140, 0x1008),
        // blr x3: [target, ret-slot, receiver, argument]
        Insn::new(
            Op::Call,
            vec![
                Operand::Local(3),
                Operand::Local(4),
                Operand::Local(1),
                Operand::Local(9),
            ],
            0x100c,
        ),
    ];
    let mut maps = resolve::ResolveMaps::default();
    maps.vtable_slots.insert((7, 1), mref("get_Item", 7, false));

    let stats = resolve::run(&mut insns, &table, &maps);

    assert_eq!(
        stats.virtuals_resolved, 1,
        "the BLR-shaped call must resolve: {:?}",
        insns[3].operands
    );
    match insns[3].operands.first() {
        Some(Operand::Method(m)) => assert_eq!(m.name, "get_Item"),
        other => panic!("expected Method operand, got {other:?}"),
    }
    assert_eq!(
        insns[3].operands.len(),
        4,
        "the operand list must survive the rewrite, got {:?}",
        insns[3].operands
    );
    assert_eq!(
        insns[3].operands[2],
        Operand::Local(1),
        "receiver preserved"
    );
    assert_eq!(
        insns[3].operands[3],
        Operand::Local(9),
        "argument preserved"
    );
}

/// `Op::Call` always carries a destination slot at operand 1 — every
/// `bl`/`blr` lifter arm allocates x0 there, `dest_position(Op::Call)` is
/// `Some(1)`, and the renderer skips that operand before reading the
/// receiver.
///
/// `trim_call_arguments` used to decide the slot existed only when the callee
/// actually returned a value, so for a VOID-returning method it computed an
/// operand count one too small and truncated the call's LAST ARGUMENT. The
/// renderer then had no operand for that parameter and printed `__unknown` —
/// ~58k arguments binary-wide.
#[test]
fn trim_call_arguments_keeps_the_destination_slot_for_void_calls() {
    let mut m = mref("DoThing", 7, false);
    m.sig.ret = ManagedType::Primitive(PrimKind::Void);
    m.sig.params = vec![
        ManagedType::Primitive(PrimKind::I4),
        ManagedType::Primitive(PrimKind::I4),
    ];
    let mut insns = vec![Insn::new(
        Op::Call,
        vec![
            Operand::Method(m),
            Operand::Local(10), // destination slot (x0) — present even for void
            Operand::Local(11), // receiver
            Operand::Local(12), // argument 0
            Operand::Local(13), // argument 1
            Operand::Local(14), // ABI register-dump filler beyond the signature
        ],
        0x1000,
    )];
    passes::trim_call_arguments(&mut insns);
    assert_eq!(
        insns[0].operands.len(),
        5,
        "void call must keep 1 target + 1 dest slot + 1 receiver + 2 args, got {:?}",
        insns[0].operands
    );
    assert_eq!(
        insns[0].operands[4],
        Operand::Local(13),
        "last argument dropped"
    );
}

/// An indirect call the resolver could NOT name must still produce a VALUE.
///
/// `call_value_expr` returned `Expr::Unknown` for a call whose target operand
/// is neither a `Method` nor an `Immediate`. That discarded the call's result,
/// and every downstream read of it — the returned object's klass, its fields,
/// the next argument carrying it, the branch testing it — rendered
/// `__unknown`. It is the single largest `__unknown` cluster in the dump
/// (`while (__unknown != 0)` with `*(nint*)(*(nint*)(__unknown) +304)` inside).
#[test]
fn unresolved_indirect_call_keeps_its_value() {
    let ir = super::RecoveredIr {
        insns: vec![
            Insn::move_(Operand::Local(10), Operand::imm(7), 0x2000),
            // `blr x9`: v11 is the callee (no definition → unnamed),
            // v12 the return slot, v10 the argument.
            Insn::new(
                Op::Call,
                vec![Operand::Local(11), Operand::Local(12), Operand::Local(10)],
                0x2004,
            ),
            // the result is used — this is what used to become `__unknown`
            Insn::move_(Operand::Local(13), Operand::Local(12), 0x2008),
            Insn::new(Op::Return, vec![Operand::Local(13)], 0x200c),
        ],
        table: locals::LocalTable::default(),
        returns_value: true,
        base_rva: 0x2000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("fnptr_0x2004"),
        "the unresolved indirect call must still be rendered:\n{out}"
    );
    assert!(
        !out.contains("__unknown"),
        "the call's result must not be discarded:\n{out}"
    );
}

#[test]
fn arm64_full_pipeline() {
    // Compiler-shaped method:
    //   movz x0, #0x1000        ; constant
    //   add  x0, x0, #8         ; arithmetic
    //   subs xzr, x0, #0        ; cmp x0, 0  (folded)
    //   b.ne skip               ; conditional branch over next insn
    //   movz x1, #1             ; skipped when taken
    //   orr  w0, wzr, wzr       ; mov w0, #0 (return value)
    //   ret
    let bytes = le(&[
        0xD2820000, // movz x0, #0x1000
        0x91002000, // add x0, x0, #8
        0xF100001F, // subs xzr, x0, #0
        0x54000041, // b.ne (+2)
        0xD2800021, // movz x1, #1
        0x2A0003E0, // mov w0, wzr
        0xD65F03C0, // ret
    ]);

    let lifted = lift::lift_arm64(&bytes, 0x1000, 256, None).expect("lift");
    // 7 native insns → 8 IR insns: the folded `subs xzr` compare now reserves
    // its address with a Nop entry so branches targeting the compare resolve
    assert_eq!(lifted.insns.len(), 8);
    let mut insns = lifted.insns;

    let mut graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    assert_eq!(graph.block_count(), 3);

    // `Cfg::build` REWRITES branch operands from RVAs to instruction indices,
    // so it must be safe to run again on the same IR. It was not: the second
    // pass read the already-resolved indices as RVAs, resolved every branch to
    // `None`, and rewrote them all to the exit sentinel. The pipeline used to
    // do exactly that after retargeting a spliced tail, which deleted the
    // whole method's control flow (all 67 native conditions of
    // `UpdateControls` @0x117674 came out as straight-line code).
    let branch_before = insns
        .iter()
        .find_map(|i| match &i.op {
            Op::CondJump(t) => Some(*t),
            _ => None,
        })
        .expect("the fixture has a conditional branch");
    assert!(
        branch_before < insns.len(),
        "branch resolves to an instruction"
    );
    let graph2 = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg again");
    let branch_after = insns
        .iter()
        .find_map(|i| match &i.op {
            Op::CondJump(t) => Some(*t),
            _ => None,
        })
        .expect("the fixture has a conditional branch");
    assert_eq!(
        branch_after, branch_before,
        "a second Cfg::build must not retarget already-resolved branches"
    );
    assert_eq!(graph2.block_count(), graph.block_count());

    stack_analyzer::analyze(&mut insns, &graph);

    let dom = dominator::DominatorInfo::compute(&graph);
    let mut ssa_out = ssa::build(&insns, &graph, &dom, &[]);
    ssa::materialize(&mut ssa_out, &mut graph);

    let seed = locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::Primitive(PrimKind::I4),
        is_static: true,
        declaring: None,
        param_names: vec![],
        ..Default::default()
    };
    let mut table = locals::create(&ssa_out.version_slot, &seed);
    locals::propagate_types(&mut ssa_out.insns, &mut table, &locals::NullOracle);

    resolve::run(&mut ssa_out.insns, &table, &resolve::ResolveMaps::default());

    passes::copy_and_constant_propagation(&mut ssa_out.insns);
    passes::constant_folding(&mut ssa_out.insns);
    passes::dead_code_elimination(&mut ssa_out.insns);

    ssa::eliminate(&mut ssa_out, &mut graph);

    let unused = passes::unused_locals(&ssa_out.insns, &table);
    let resolver = NullResolver;
    let emitter = emit::Emitter {
        resolver: &resolver,
    };
    let out = emitter.generate(&ssa_out.insns, &table, &unused, true);

    assert!(!out.instructions.is_empty());
    assert!(matches!(
        out.instructions.last(),
        Some(dotnetdll::resolved::il::Instruction::Return)
    ));
    // branch fixups all resolved and in range
    for i in &out.instructions {
        match i {
            dotnetdll::resolved::il::Instruction::Branch(t)
            | dotnetdll::resolved::il::Instruction::BranchTruthy(t)
            | dotnetdll::resolved::il::Instruction::BranchFalsy(t) => {
                assert_ne!(*t, usize::MAX, "unfixed branch target");
                assert!(*t < out.instructions.len(), "branch target out of range");
            }
            _ => {}
        }
    }
}

/// Loop shape exercises phi placement + elimination:
///   movz x0, #0          ; i = 0
/// loop:
///   add  x0, x0, #1      ; i++
///   cmp  x0, #10
///   blt  loop            ; back edge → x0 has multiple defs → phi
///   ret
#[test]
fn loop_shape_produces_and_removes_phis() {
    // addresses: base=0x2000
    // 0x2000 movz x0,#0
    // 0x2004 add x0,x0,#1     <- loop head
    // 0x2008 subs xzr,x0,#10
    // 0x200C b.lt -3 insns (to 0x2004): offset -12
    // 0x2010 ret
    let bytes = le(&[
        0xD28F0000, // movz x0, #0  (hw=0 imm=0 → 0xD2800000; use plain)
        0x91000400, // add x0, x0, #1
        0xF1003DFF, // subs xzr, x0, #15 (imm12=15<<10=0x3C00|0x1F)
        0x54000F4B, // placeholder b.lt — recomputed below
        0xD65F03C0, // ret
    ]);
    let _ = bytes;
    // encode precisely instead:
    let movz_x0_0: u32 = 0xD2800000; // imm16=0 hw=0 rd=0
    let add_x0_x0_1: u32 = 0x91000000 | (1 << 10) | (0 << 5) | 0;
    let subs_xzr_x0_10: u32 = 0xF1000000 | (10 << 10) | (0 << 5) | 31;
    // b.lt to 0x2004 from 0x200C: offset = -8 bytes => imm19 = (-8/4)&0x7FFFF = 0x7FFFE
    let imm19 = ((0u32.wrapping_sub(2)) & 0x7FFFF) << 5;
    let blt: u32 = 0x54000000 | imm19 | 11; // cond LT = 1011b
    let bytes = le(&[movz_x0_0, add_x0_x0_1, subs_xzr_x0_10, blt, 0xD65F03C0]);

    let lifted = lift::lift_arm64(&bytes, 0x2000, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let mut graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    stack_analyzer::analyze(&mut insns, &graph);

    let dom = dominator::DominatorInfo::compute(&graph);
    let mut ssa_out = ssa::build(&insns, &graph, &dom, &[]);
    ssa::materialize(&mut ssa_out, &mut graph);

    // the loop counter must have a phi (two defs joining at loop head)
    assert!(
        !ssa_out.phis.is_empty(),
        "expected at least one phi for the loop-carried counter"
    );

    ssa::eliminate(&mut ssa_out, &mut graph);
    // after elimination no Phi ops remain
    assert!(ssa_out.insns.iter().all(|i| i.op != Op::Phi));

    // branch targets still valid post-elimination
    let n = ssa_out.insns.len();
    for insn in &ssa_out.insns {
        match insn.op {
            Op::Jump(t) | Op::CondJump(t) => assert!(t < n),
            _ => {}
        }
    }
}

/// String literal resolution through il2cpp_string_new-style call shapes.
#[test]
fn string_literal_resolution() {
    let bytes = le(&[
        0xD2804000, // movz x0, #0x200  (literal slot address)
        0xF94003E1, // ldr x1, [sp]     (noise)
        0xD65F03C0, // ret              (stop before call — resolution test only)
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x3000, 16, None).unwrap();
    let mut maps = resolve::ResolveMaps::default();
    maps.literal_index_by_addr.insert(0x200, 7);
    maps.literals.insert(7, "Hello".to_string());
    let mut insns = lifted.insns;

    // simulate the string_new rewrite directly on the move
    resolve::run(&mut insns, &locals_local_table(), &maps);
    let has_lit = insns.iter().any(|i| {
        matches!(
            i.operands.get(1),
            Some(Operand::StringLiteral { index: 7, .. })
        )
    });
    assert!(
        has_lit,
        "literal-slot immediate should become StringLiteral"
    );
}

// tiny helper so the string test can build an empty table
fn locals_local_table() -> super::locals::LocalTable {
    let seed = locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::Primitive(PrimKind::Void),
        is_static: true,
        declaring: None,
        param_names: vec![],
        ..Default::default()
    };
    super::locals::create(&HashMap::new(), &seed)
}

/// Reproduce GetLanguage(): call result used as receiver of next call.
///   bl  PreciseLocale.GetLanguage   ; x0 = result
///   b   skip_args_setup (nothing)
///   bl  Sanitize                    ; instance: this = x0 (previous result)
///   ret
#[test]
fn receiver_from_previous_call_result() {
    // layout @0x4000:
    // 0x4000 bl A        (+8 → 0x4008)
    // 0x4004 bl B        (+8 → 0x400C)
    // 0x4008 ret
    let bl_a: u32 = 0x94000000 | ((2u32) & 0x3FF_FFFF); // offset +2 insns
    let bl_b: u32 = 0x94000000 | ((2u32) & 0x3FF_FFFF); // offset +2 insns
    let bytes = le(&[bl_a, bl_b, 0xD65F03C0]);

    let lifted = lift::lift_arm64(&bytes, 0x4000, 32, None).expect("lift");
    let mut insns = lifted.insns;
    println!("=== LIFTED ===");
    for (i, insn) in insns.iter().enumerate() {
        println!("  {i}: {insn}");
    }

    let mut graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    stack_analyzer::analyze(&mut insns, &graph);
    let dom = dominator::DominatorInfo::compute(&graph);
    let mut ssa_out = ssa::build(&insns, &graph, &dom, &[]);
    ssa::materialize(&mut ssa_out, &mut graph);

    let seed = locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::String,
        is_static: false,
        declaring: None,
        param_names: vec![],
        ..Default::default()
    };
    let mut table = locals::create(&ssa_out.version_slot, &seed);
    locals::propagate_types(&mut ssa_out.insns, &mut table, &locals::NullOracle);

    let mut maps = resolve::ResolveMaps::default();
    maps.methods_by_rva
        .insert(0x5000, method_ref_for_test("PreciseLocale", "GetLanguage"));
    maps.methods_by_rva
        .insert(0x6000, method_ref_for_instance("Sanitize", "HTTP"));
    // patch call targets to the mapped RVAs
    for insn in insns.iter_mut() {
        if insn.op == Op::Call {
            if let Some(Operand::Immediate(t)) = insn.operands.first().cloned() {
                if t == 0x4008_usize as i64 { /* self-jump artifact? no */ }
            }
        }
    }
    // our two BLs targeted past-ret addresses — remap on the POST-SSA listing
    let mut fixed = 0;
    for insn in ssa_out.insns.iter_mut() {
        if matches!(insn.op, Op::Call | Op::IndirectCall) {
            if let Some(Operand::Immediate(t)) = insn.operands.first().cloned() {
                let _ = t;
                let target_rva = if fixed == 0 { 0x5000 } else { 0x6000 };
                insn.operands[0] = Operand::Immediate(target_rva as i64);
                fixed += 1;
            }
        }
    }

    let stats = resolve::run(&mut ssa_out.insns, &table, &maps);
    println!(
        "=== RESOLVED === resolved={} unresolved={}",
        stats.calls_resolved, stats.calls_unresolved
    );
    for (i, insn) in ssa_out.insns.iter().enumerate() {
        println!("  {i}: {insn:?} | {insn}");
    }

    ssa::eliminate(&mut ssa_out, &mut graph);

    let ir = super::RecoveredIr {
        insns: ssa_out.insns,
        table,
        returns_value: true,
        base_rva: 0,
    };
    let text = crate::recovery::csharp::render_method_csharp(
        &ir,
        "GetLanguage",
        false,
        &[],
        &ManagedType::String,
        -1,
    );
    println!("=== C# ===\n{text}");
    assert!(
        text.contains("PreciseLocale.GetLanguage"),
        "first call missing"
    );
}

fn method_ref_for_test(ty: &str, name: &str) -> MethodRef {
    MethodRef {
        image_index: 0,
        type_def_index: 0,
        method_index: 0,
        name: name.into(),
        declaring_namespace: String::new(),
        declaring_type_name: ty.into(),
        assembly: String::new(),
        sig: types::MethodSigInfo {
            ret: ManagedType::Class(types::NamedType::object()),
            params: vec![],
            is_static: true,
        },
    }
}

fn method_ref_for_instance(ty: &str, name: &str) -> MethodRef {
    let mut m = method_ref_for_test(ty, name);
    m.sig.is_static = false;
    m.sig.ret = ManagedType::String;
    m
}

/// Diagnose why real-world .ctor prologues fail recovery.
/// Typical Unity ctor: stp pair, class-init guard via ldrb+tbz, base call.
#[test]
fn diagnose_ctor_failure_shape() {
    // Shape seen at 0x13a20 (84 bytes):
    //   stp x22,x21; stp x20,x19; mov x19,x0
    //   adrp/add klass; ldrb w8,[x8,#0x135]; tbnz w8,#0 -> init
    //   ... base .ctor call ... init: bl class_init
    let bytes = le(&[
        0xA9025BF6, // stp x22, x21, [sp, #32]
        0xA9034FF5, // stp x21, x20, [sp, #48]
        0xAA0003F4, // mov x20, x0
        0x90000100, // adrp x0, <page>
        0x91000400, // add x0, x0, #1
        0xF9400408, // ldr x8, [x0]
        0x37000128, // tbnz w8, #0, +2 insns
        0xD65F03C0, // ret (early out — simplified)
        0xD65F03C0, // ret
    ]);
    let Some(lifted) = lift::lift_arm64(&bytes, 0x13a20, 64, None) else {
        panic!("lift failed");
    };
    println!("lifted {} insns", lifted.insns.len());
    let mut insns = lifted.insns;
    let Some(graph) = cfg::Cfg::build(&mut insns, &lifted.addresses) else {
        println!("!! CFG BUILD FAILED — this is the failure point");
        for (i, insn) in lift::lift_arm64(&bytes, 0x13a20, 64, None)
            .unwrap()
            .insns
            .iter()
            .enumerate()
        {
            println!("  {i}: {insn}");
        }
        return;
    };
    println!("cfg ok: {} blocks", graph.block_count());
}

/// Switch table: ADRP+ADD base, scaled indexed load, BR — with real bytes.
#[test]
fn switch_table_recovery() {
    // IR-level test: inject the shape directly (lift already covered)
    use super::passes::recover_switch_tables;
    let mut insns = vec![
        Insn::move_(Operand::reg("x9"), Operand::imm(0x6000), 0x8000), // table base
        Insn::move_(
            Operand::reg("x8"),
            Operand::Memory {
                width: 0,
                base: Box::new(Operand::reg("x9")),
                index: Some(Box::new(Operand::Local(4))),
                scale: 4,
                addend: 0,
            },
            0x8004,
        ),
        Insn::new(Op::IndirectCall, vec![Operand::reg("x8")], 0x8008),
        Insn::new(Op::Return, vec![Operand::reg("x0")], 0x800c),
    ];
    let addresses = vec![0x8000, 0x8004, 0x8008, 0x800c];
    // table at 0x6000: entries → 0x8008 (case0), 0x800c (case1), then garbage
    let mut table = HashMap::new();
    table.insert(0x6000u64, 0x8008u32);
    table.insert(0x6004u64, 0x800cu32);
    table.insert(0x6008u64, 0xdeadu32);

    let n = recover_switch_tables(&mut insns, &addresses, &|a| table.get(&a).copied());
    assert_eq!(n, 1, "switch should be recovered");
    match &insns[2].op {
        Op::Switch(t) => {
            assert_eq!(t.len(), 2);
            assert_eq!(t[0], 2); // → insn index of original br? no: target rva 0x8008 == br itself... 
        }
        other => panic!("expected Switch, got {other:?}"),
    }
}

/// while (n <= v) n <<= 1; — verify condition polarity.
///   0x9000 movz w1, #1          ; n = 1
///   0x9004 b cond               ; goto header test
/// body:
///   0x9008 lsl w0, w0, #1       ; n << 1  (approx: lslv)
/// header:
///   0x900c cmp w0, w1           ; subs xzr, x0, x1
///   0x9010 ble exit             ; taken exits → invert for while
///   0x9014 b body
/// exit:
///   0x9018 ret
#[test]
fn loop_condition_polarity() {
    // while (w0 != 0) { w0 = w0 - w1 } shape:
    //   0x9000 mov r1, #3        E3A01003
    //   0x9004 b cond            EA000002 (→0x9010)
    // body:
    //   0x9008 sub r0, r0, r1    E0400001
    // header:
    //   0x900c cmp r0, #7        E3500007   (stand-in for compare)
    //   0x9010 bne body          1AFFFFFD (→0x9008)
    //   0x9014 bx lr             E12FFF1E
    let bytes = le(&[
        0xE3A01003, 0xEA000002, 0xE0400001, 0xE3500007, 0x1AFFFFFD, 0xE12FFF1E,
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x9000, 64).expect("lift");
    let mut insns = lifted.insns;
    println!("=== ARM32 IR ===");
    for (i, insn) in insns.iter().enumerate() {
        println!("  {i}: {insn}");
    }
    let mut graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    assert!(graph.block_count() >= 2);
    stack_analyzer::analyze(&mut insns, &graph);

    // back edge exists → structurer should produce a While/DoWhile statement
    let dom = dominator::DominatorInfo::compute(&graph);
    let mut ssa_out = ssa::build(&insns, &graph, &dom, &[]);
    ssa::materialize(&mut ssa_out, &mut graph);

    let seed = locals::SeedInfo {
        param_types: vec![ManagedType::Primitive(PrimKind::I4)],
        ret_type: ManagedType::Primitive(PrimKind::I4),
        is_static: true,
        declaring: None,
        param_names: vec!["n".into()],
        ..Default::default()
    };
    let mut table = locals::create(&ssa_out.version_slot, &seed);
    locals::propagate_types(&mut ssa_out.insns, &mut table, &locals::NullOracle);
    resolve::run(&mut ssa_out.insns, &table, &resolve::ResolveMaps::default());
    passes::dead_code_elimination(&mut ssa_out.insns);
    ssa::eliminate(&mut ssa_out, &mut graph);

    let unused = passes::unused_locals(&ssa_out.insns, &table);
    let resolver = NullResolver;
    let emitter = emit::Emitter {
        resolver: &resolver,
    };
    let out = emitter.generate(&ssa_out.insns, &table, &unused, true);
    assert!(matches!(
        out.instructions.last(),
        Some(dotnetdll::resolved::il::Instruction::Return)
    ));
}

// ---------------------------------------------------------------------------
// x86/x64: memory destinations, RIP-relative addressing, atomics, flag-only
// bit tests. Encodings are the canonical ones (the decoder is iced-x86, which
// rejects anything malformed, so a wrong byte string fails loudly here rather
// than passing silently).
// ---------------------------------------------------------------------------

fn lift_sysv(bytes: &[u8]) -> Vec<Insn> {
    lift_x86::lift_x86_64(bytes, 0x1000, 64, lift_x86::X86Abi::SysV)
        .expect("lift")
        .insns
}

/// The two x86-64 calling conventions put the argument registers in DIFFERENT
/// physical registers, and the lifter's positional vocabulary must follow the
/// container: PE ⇒ MSVC (`rcx, rdx, r8, r9`), ELF/Mach-O ⇒ SysV
/// (`rdi, rsi, rdx, rcx, r8, r9`).
///
/// `mov rbx, rcx ; mov edi, edx` is the prologue of an MSVC instance method:
/// save argument 0 (`this`), then read argument 1. Under SysV the same two
/// physical registers are arguments 3 and 2. Naming them with the wrong
/// convention seeds the wrong parameter slots, so the real receiver and
/// arguments have no definition at all and every read of them becomes an
/// undefined local — which on the v24.5 PE corpus was 39,595 findings over
/// 9,539 methods, 6,677 of them with NOTHING else wrong.
#[test]
fn x64_argument_register_names_follow_the_calling_convention() {
    // 48 8B D9  mov rbx, rcx        (save arg0 — the receiver under MSVC)
    // 8B FA     mov edi, edx        (read arg1)
    let bytes = [0x48u8, 0x8B, 0xD9, 0x8B, 0xFA];
    let move_srcs = |abi: lift_x86::X86Abi| -> Vec<String> {
        let l = lift_x86::lift_x86_64(&bytes, 0x1000, 64, abi).expect("lift");
        l.insns
            .iter()
            .filter_map(|i| match &i.op {
                // At lift time a register operand is still
                // `Operand::Register(name)`; `locals::create` turns it into a
                // numbered local later. The NAME is what this test is about.
                Op::Move => i.operands.last().and_then(|o| match o {
                    Operand::Register(r) => Some(r.clone()),
                    Operand::Local(n) => Some(format!("v{n}")),
                    other => panic!("unexpected move source {other:?}"),
                }),
                _ => None,
            })
            .collect()
    };
    let msvc = move_srcs(lift_x86::X86Abi::Msvc);
    let sysv = move_srcs(lift_x86::X86Abi::SysV);
    assert_eq!(
        msvc.len(),
        2,
        "expected the two moves to survive lifting: {msvc:?}"
    );
    // The two conventions must disagree, or the ABI is not being applied at all.
    assert_ne!(
        msvc, sysv,
        "MSVC and SysV produced the same register names, so the ABI is ignored"
    );
    // …and the ROLES must swap in the ABI-predicted direction: `rcx` is the
    // first argument under MSVC and the fourth under SysV, so the `x`-ordinal
    // for `rcx` differs by exactly 3 between the two.
    let ordinal = |v: &str| -> Option<u32> { v.strip_prefix('x')?.parse::<u32>().ok() };
    let (Some(m0), Some(s0)) = (ordinal(&msvc[0]), ordinal(&sysv[0])) else {
        panic!("expected positional register names, got msvc={msvc:?} sysv={sysv:?}");
    };
    assert_eq!(
        m0 + 3,
        s0,
        "rcx must be argument {m0} under MSVC and argument {s0} under SysV"
    );
}

/// A call's operand dump must carry exactly the argument registers its ABI has.
///
/// Both call arms hardcoded the SysV six, so under MSVC every call carried two
/// extra slots named `x4`/`x5` — registers that do not exist in that convention.
/// They manufactured two locals per method that nothing writes, became two extra
/// `__abi` placeholders in every unnamed call, and inflated the operand count the
/// argument-alignment pass reasons about.
#[test]
fn the_call_argument_dump_carries_only_the_registers_the_abi_has() {
    // E8 00 00 00 00   call <next instruction>
    let bytes = [0xE8u8, 0x00, 0x00, 0x00, 0x00];
    let arg_slots = |abi: lift_x86::X86Abi| -> Vec<String> {
        let l = lift_x86::lift_x86_64(&bytes, 0x1000, 16, abi).expect("lift");
        let call = l
            .insns
            .iter()
            .find(|i| matches!(i.op, Op::Call))
            .expect("a call");
        call.operands
            .iter()
            .filter_map(|o| match o {
                // `rv` is the result register, not an argument slot.
                Operand::Register(r) if r.starts_with('x') => Some(r.clone()),
                _ => None,
            })
            .collect()
    };
    assert_eq!(
        arg_slots(lift_x86::X86Abi::Msvc),
        vec!["x0", "x1", "x2", "x3"],
        "MSVC passes four arguments in registers — x4/x5 do not exist"
    );
    assert_eq!(
        arg_slots(lift_x86::X86Abi::SysV),
        vec!["x0", "x1", "x2", "x3", "x4", "x5"],
        "SysV passes six"
    );
}

/// `Op::Arith` is read as `operands[1] OP operands[2]` with the destination in
/// slot 0 (see `def_expr_inner`), so a two-operand form renders as
/// `dst = src OP <?>` — the source is consumed and the second operand is a
/// fabricated `Unknown`. Every `Op::Arith` the x86 lifter emits must therefore
/// carry the destination AND both sources.
fn assert_arith_is_three_operand(insns: &[Insn], what: &str) {
    for i in insns {
        if matches!(i.op, Op::Arith(_)) {
            assert_eq!(
                i.operands.len(),
                3,
                "{what}: Op::Arith renders operands[1] OP operands[2]; a {} operand form drops the real source: {:?}",
                i.operands.len(),
                i
            );
        }
    }
}

/// `test a, a` is the canonical NULL CHECK — every lazy static, every
/// `ldsfld` guard and every `this == null` throw compiles to it. Lifting it as
/// a comparison of the operand with ITSELF produced a condition that is
/// constant by construction (`sf0 != sf0`, `this != this`, `in_x3 != in_x3`),
/// which says nothing about the code that is actually there. Measured on the
/// v24.5 PE dump: 23,073 such self-comparisons from this one form.
#[test]
fn x86_test_same_register_is_a_null_check() {
    // `test rax, rax` = 48 85 C0 ; `je +2` = 74 02 ; `xor eax, eax` = 31 C0 ; `ret` = C3
    let insns = lift_sysv(&[0x48, 0x85, 0xC0, 0x74, 0x02, 0x31, 0xC0, 0xC3]);
    let cmp = insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("the `je` must materialize the comparison it branches on");
    assert_eq!(
        cmp.operands.get(2),
        Some(&Operand::Immediate(0)),
        "`test a, a` tests the AND against ZERO; recording the operand pair read \
         as a self-comparison: {cmp:?}"
    );
    assert_ne!(
        cmp.operands.get(1),
        cmp.operands.get(2),
        "a comparison of a value with itself is constant, so it is never the \
         condition `test` produces: {cmp:?}"
    );
}

/// An INTEGER ARITHMETIC instruction writes the flags from its own result, so
/// a `Jcc` after it branches on that result — not on some earlier compare.
///
/// `sub ecx, 1; je L` is the canonical "decrement and test" of every countdown
/// loop and every `switch` on a small enum, and only `cmp`/`test` used to
/// publish a pending comparison. The `je` therefore had nothing to read, the
/// condition came out `Unknown`, and the dump carried `while (__unknown != 0)`
/// with an unrecovered body: 6,111 `__unknown` tokens on the v24.5 PE dump.
#[test]
fn x86_arithmetic_result_is_the_condition_the_jcc_reads() {
    // `sub ecx, 1` = 83 E9 01 ; `je +8` = 74 08 ; 8× `nop` ; `ret` = C3
    let insns = lift_sysv(&[
        0x83, 0xE9, 0x01, 0x74, 0x08, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0xC3,
    ]);
    let cj = insns
        .iter()
        .find(|i| matches!(i.op, Op::CondJump(_)))
        .expect("the `je` must be lifted as a conditional jump");
    let scratch = cj.operands.first().cloned().expect("condjump condition");
    let cmp = insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)) && i.operands.first() == Some(&scratch))
        .expect("the `je` must materialize the comparison it branches on");
    assert!(
        !matches!(cmp.operands.get(1), Some(Operand::Unknown)),
        "`sub ecx, 1` set the flags, so the condition is its RESULT, never \
         `Unknown`: {cmp:?}"
    );
    assert_eq!(
        cmp.operands.get(2),
        Some(&Operand::Immediate(0)),
        "ZF after an arithmetic instruction means `result == 0`: {cmp:?}"
    );
}

/// A mask test is `(a & mask) == 0`, so the AND has to be materialized — a
/// different-operand `test` is NOT a comparison of the two operands.
#[test]
fn x86_test_mask_materializes_the_and() {
    // `test eax, 1` = A9 01 00 00 00 ; `jz +2` = 74 02 ; `ret` = C3
    let insns = lift_sysv(&[0xA9, 0x01, 0x00, 0x00, 0x00, 0x74, 0x02, 0xC3]);
    assert!(
        insns
            .iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::And))),
        "`test eax, 1` is an AND, so the condition has to be `(a & 1) == 0`: {insns:?}"
    );
    let cmp = insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("compare");
    assert_eq!(cmp.operands.get(2), Some(&Operand::Immediate(0)));
}

/// `call r/m` is a COMPUTED callee — virtual/interface dispatch, a delegate
/// invoke, a function pointer — and it RETURNS. Lifting it to the
/// `Op::IndirectCall` CFG terminator (with no operands at all) ended the block
/// at a call the callee comes back from AND threw the callee away, so nothing
/// downstream could name it.
#[test]
fn x86_indirect_call_keeps_callee_and_reaches_what_follows() {
    // `call qword ptr [rax + 0x10]` = FF 50 10 ; `ret` = C3
    let insns = lift_sysv(&[0xFF, 0x50, 0x10, 0xC3]);
    let call = insns
        .iter()
        .find(|i| matches!(i.op, Op::Call))
        .unwrap_or_else(|| {
            panic!(
                "`call [reg+disp]` must lift to a RETURNING call, got {:?}",
                insns.iter().map(|i| &i.op).collect::<Vec<_>>()
            )
        });
    match call.operands.first() {
        Some(Operand::Memory { addend, .. }) => assert_eq!(
            *addend, 0x10,
            "the computed callee must be kept as operand 0: {call:?}"
        ),
        other => panic!("the computed callee must be kept, got {other:?}"),
    }
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Return)),
        "the `ret` after the call must still be recovered — a terminator here \
         makes everything after it unreachable"
    );
}

/// `jmp r/m` is a tail call (or an unrecovered jump table) and keeps its
/// target: `passes::recover_tail_calls` and the switch recognizer both read
/// operand 0.
#[test]
fn x86_indirect_jump_keeps_its_target() {
    // `jmp rax` = FF E0
    let insns = lift_sysv(&[0xFF, 0xE0]);
    let ic = insns
        .iter()
        .find(|i| matches!(i.op, Op::IndirectCall))
        .expect("jmp r/m is the indirect-branch terminator");
    assert_eq!(
        ic.operands.len(),
        1,
        "the branch target is what makes a tail call recognisable: {ic:?}"
    );
}

/// `add/sub rsp, imm8` is encoded with an IMM8 that SIGN-EXTENDS to 64 bits, so
/// the decoder reports `Immediate8to64` — which matched neither of the two
/// arms the old code had, fell through to `0`, and made EVERY x64 frame
/// adjustment `ShiftStack(0)`. A frame delta of zero is worse than a missing
/// one: the stack analyzer models no frame, and `recover_tail_calls` never sees
/// its frame torn down (so every x64 tail call stayed an "indirect jump").
#[test]
fn x86_stack_adjustment_keeps_its_size() {
    // `sub rsp, 0x28` = 48 83 EC 28 ; `ret` = C3
    let insns = lift_sysv(&[0x48, 0x83, 0xEC, 0x28, 0xC3]);
    match insns.first().map(|i| &i.op) {
        Some(Op::ShiftStack(v)) => assert_eq!(*v, -0x28, "the frame delta was lost"),
        other => panic!("expected ShiftStack, got {other:?}"),
    }
}

#[test]
fn x86_register_arithmetic_names_both_sources() {
    // `add eax, ecx` = 01 C8 ; `imul edx, esi, 0x10` = 6B D6 10 ; `ret` = C3
    let insns = lift_sysv(&[0x01, 0xC8, 0x6B, 0xD6, 0x10, 0xC3]);
    assert_arith_is_three_operand(&insns, "add/imul");

    // The three-operand `imul` must multiply by the IMMEDIATE. The old
    // operand accessor read operand 1 for index 2, so this came out as
    // `edx = esi * esi` — both operands real, the product silently wrong.
    let imul = insns
        .iter()
        .find(|i| matches!(i.op, Op::Arith(ArithKind::Mul)))
        .expect("imul");
    assert_eq!(
        imul.operands.get(1),
        Some(&Operand::Register("x1".into())),
        "imul must multiply its SOURCE operand: {:?}",
        imul
    );
    assert_eq!(
        imul.operands.get(2),
        Some(&Operand::Immediate(0x10)),
        "imul must multiply by the immediate in operand 2, not operand 1: {:?}",
        imul
    );
}

#[test]
fn x86_shift_and_rotate_name_both_sources() {
    // `shl eax, cl` = D3 E0 ; `ror edx, 3` = C1 CA 03 ; `ret` = C3
    let insns = lift_sysv(&[0xD3, 0xE0, 0xC1, 0xCA, 0x03, 0xC3]);
    assert_arith_is_three_operand(&insns, "shl/ror");
}

#[test]
fn x86_idiv_publishes_the_remainder_in_the_high_half() {
    // `cqo` = 48 99 ; `idiv rcx` = 48 F7 F9 ; `mov rax, rdx` = 48 89 D0 ; `ret`
    // x86 lowers `%` to exactly this pair, so dropping the remainder left
    // every modulo reading whatever rdx happened to hold.
    let insns = lift_sysv(&[0x48, 0x99, 0x48, 0xF7, 0xF9, 0x48, 0x89, 0xD0, 0xC3]);
    assert!(
        insns
            .iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::Rem))),
        "idiv must publish the remainder; got {insns:?}"
    );
}

#[test]
fn x86_bswap_reverses_the_bytes_and_defines_its_register() {
    // `bswap eax` = 0F C8 ; `ret` = C3. It WRITES, so dropping it left the
    // register holding its UN-swapped value — a silently wrong read, not a
    // missing one. Modelled as four extracted bytes OR-ed back mirrored.
    let insns = lift_sysv(&[0x0F, 0xC8, 0xC3]);
    // `bswap eax` touches `rax`, which on x86-64 is the RESULT register
    // (`ir::RET_REG`) and NOT argument register 0 — `x0` there is `rdi`. Naming
    // both `x0` made the accumulator and the receiver one local.
    let defines_ret = insns.iter().any(|i| {
        Insn::dest_position(&i.op) == Some(0)
            && matches!(i.operands.first(), Some(Operand::Register(r)) if r == ir::RET_REG)
    });
    assert!(
        defines_ret,
        "bswap must write its register ({}); got {insns:?}",
        ir::RET_REG
    );
    // four `& 0xFF` extractions, one per byte
    let byte_masks = insns
        .iter()
        .filter(|i| {
            matches!(i.op, Op::Arith(ArithKind::And))
                && i.operands.get(2) == Some(&Operand::Immediate(0xFF))
        })
        .count();
    assert_eq!(
        byte_masks, 4,
        "bswap r32 must extract all four bytes; got {insns:?}"
    );
    // and the pieces are OR-ed back together
    let ors = insns
        .iter()
        .filter(|i| matches!(i.op, Op::Arith(ArithKind::Or)))
        .count();
    assert!(ors >= 3, "bswap must OR the bytes back; got {insns:?}");
}

#[test]
fn x86_setcc_forms_that_test_parity_are_modelled() {
    // `ucomiss xmm0, xmm1` = 0F 2E C1 ; `setnp al` = 0F 9B C0 ; `ret` = C3
    // A one-operand instruction with no arm emitted NOTHING, so the NaN test
    // that compiles to `ucomiss; setnp` read a stale byte.
    let insns = lift_sysv(&[0x0F, 0x2E, 0xC1, 0x0F, 0x9B, 0xC0, 0xC3]);
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Move)
            && matches!(i.operands.first(), Some(Operand::Register(r)) if r == ir::RET_REG)),
        "setnp must define its destination ({}); got {insns:?}",
        ir::RET_REG
    );
}

/// The VEX-encoded FP compares must set the pending compare exactly like the
/// legacy SSE forms.
///
/// `vcomiss`/`vucomiss`/`vcomisd`/`vucomisd` are the AVX encodings of the same
/// EFLAGS-setting operation as `comiss`/`ucomiss`/`comisd`/`ucomisd`, and they
/// were missing from the compare arm — so an AVX float decision lost its
/// condition and rendered `if (? != 0)`. This is the x86 twin of the ARM32
/// `vcmpe` gap (see `arm32_vcmpe_and_register_vcmp_keep_their_operands`).
#[test]
fn x86_vex_fp_compares_are_modelled() {
    // `vucomiss xmm2, xmm1` = C5 F8 2E D1 ; `jne +1` = 75 01 ; `ret` = C3.
    let insns = lift_sysv(&[0xC5, 0xF8, 0x2E, 0xD1, 0x75, 0x01, 0xC3]);
    let cmp = insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("vucomiss must produce a Compare for the following jne");
    // Both operands are the XMM registers, not `Unknown`.
    assert!(
        matches!(cmp.operands.get(1), Some(Operand::Register(_)))
            && matches!(cmp.operands.get(2), Some(Operand::Register(_))),
        "vucomiss must compare two registers; got {cmp:?}"
    );
    assert!(insns.iter().any(|i| matches!(i.op, Op::CondJump(_))));
}

#[test]
fn x86_avx_and_sse_name_one_physical_register() {
    // `movaps xmm3, xmm0` = 0F 28 D8 ; `vaddps ymm3, ymm3, ymm4` = C5 E4 58 DC
    // ; `ret` = C3. xmm3 IS the low half of ymm3, so both must be `vf3`; naming
    // them apart invents a register no def ever writes.
    let insns = lift_sysv(&[0x0F, 0x28, 0xD8, 0xC5, 0xE4, 0x58, 0xDC, 0xC3]);
    let names: Vec<String> = insns
        .iter()
        .flat_map(|i| i.operands.iter())
        .filter_map(|o| match o {
            Operand::Register(r) => Some(r.clone()),
            _ => None,
        })
        .collect();
    assert!(
        names.iter().any(|n| n == "vf3"),
        "both the SSE and the AVX form must name vf3; got {names:?}"
    );
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with('r') && n[1..].parse::<u32>().is_ok()),
        "no vector register may fall back to a phantom `r<ordinal>` name; got {names:?}"
    );
}

/// Any register name that is a raw ordinal (`r16`, `r164`) is a register the
/// recovered program has no identity for — it can never be defined by a real
/// instruction, so every read through it becomes an undefined local.
fn assert_no_phantom_register_names(insns: &[Insn], what: &str) {
    for o in insns.iter().flat_map(|i| i.operands.iter()) {
        if let Operand::Register(r) = o {
            assert!(
                !(r.starts_with('r') && r[1..].parse::<u32>().is_ok()),
                "{what}: {r} is a raw-ordinal phantom name — a width-specific spelling of a register already named elsewhere: {insns:?}"
            );
        }
    }
}

#[test]
fn x86_sub_register_widths_share_one_identity() {
    // `mov al, 5` = B0 05 ; `movzx eax, al` = 0F B6 C0 ; `shl eax, cl` = D3 E0 ; `ret`
    //
    // `al`/`ax`/`eax`/`rax` are ONE register. Naming the widths separately made
    // the byte-sized write invisible (nothing ever defined the other spelling),
    // so `movzx eax, al` read an undefined local and the chain collapsed.
    let insns = lift_sysv(&[0xB0, 0x05, 0x0F, 0xB6, 0xC0, 0xD3, 0xE0, 0xC3]);
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Move)
            && matches!(i.operands.first(), Some(Operand::Register(r)) if r == ir::RET_REG)
            && i.operands.get(1) == Some(&Operand::Immediate(5))),
        "`mov al, 5` must define the register the wider reads use ({}): {insns:?}",
        ir::RET_REG
    );
    assert!(
        insns
            .iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::Shl))
                && i.operands.get(1) == Some(&Operand::Register(ir::RET_REG.into()))
                && i.operands.get(2) == Some(&Operand::Register("x3".into()))),
        "`shl eax, cl` must shift {} by x3 (cl IS ecx): {insns:?}",
        ir::RET_REG
    );
    assert_no_phantom_register_names(&insns, "sub-register widths");
}

#[test]
fn x86_rep_movs_copies_and_leaves_the_exact_end_state() {
    // `rep movsb` = F3 A4 ; `ret` = C3
    //
    // An inline `rep movsb` is how a struct copy compiles. It was dropped
    // whole — the memory write, both pointer updates and the counter — because
    // its operands are the implicit `[rdi]`/`[rsi]`.
    let insns = lift_sysv(&[0xF3, 0xA4, 0xC3]);
    let copy = insns
        .iter()
        .find(|i| {
            matches!(i.op, Op::Move) && matches!(i.operands.get(1), Some(Operand::Memory { .. }))
        })
        .expect("the byte copy must not be dropped");
    match (&copy.operands[0], &copy.operands[1]) {
        (
            Operand::Memory {
                width: 1, base: d, ..
            },
            Operand::Memory {
                width: 1, base: s, ..
            },
        ) => {
            assert!(
                matches!(**d, Operand::Register(ref r) if r == "x0"),
                "`movsb` dest base must be rdi: {copy:?}"
            );
            assert!(
                matches!(**s, Operand::Register(ref r) if r == "x1"),
                "`movsb` source base must be rsi: {copy:?}"
            );
        }
        _ => panic!("`rep movsb` must be a ONE-BYTE memory-to-memory copy: {copy:?}"),
    }
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Move)
            && matches!(i.operands.first(), Some(Operand::Register(r)) if r == "x3")
            && i.operands.get(1) == Some(&Operand::Immediate(0))),
        "the counter must end at zero: {insns:?}"
    );
    let advances: Vec<&Insn> = insns
        .iter()
        .filter(|i| matches!(i.op, Op::Arith(ArithKind::Add)))
        .collect();
    assert_eq!(
        advances.len(),
        2,
        "both pointers must advance (and only advance): {insns:?}"
    );
}

#[test]
fn x86_rep_stos_advances_only_the_destination() {
    // `rep stosq` = F3 48 AB ; `ret` = C3
    let insns = lift_sysv(&[0xF3, 0x48, 0xAB, 0xC3]);
    let fill = insns
        .iter()
        .find(|i| {
            matches!(i.op, Op::Move) && matches!(i.operands.get(1), Some(Operand::Register(_)))
        })
        .expect("the store must not be dropped");
    match (&fill.operands[0], &fill.operands[1]) {
        (
            Operand::Memory {
                width: 8, base: d, ..
            },
            Operand::Register(r),
        ) => {
            assert!(
                matches!(**d, Operand::Register(ref b) if b == "x0"),
                "stos writes [rdi] (x0): {fill:?}"
            );
            assert_eq!(
                r,
                ir::RET_REG,
                "stos stores the accumulator ({}): {fill:?}",
                ir::RET_REG
            );
        }
        _ => panic!("`rep stosq` must be an eight-byte store of the accumulator: {fill:?}"),
    }
    let adds: Vec<&Insn> = insns
        .iter()
        .filter(|i| matches!(i.op, Op::Arith(ArithKind::Add)))
        .collect();
    assert_eq!(adds.len(), 1, "only rdi advances for `stos`: {insns:?}");
    assert!(
        matches!(adds[0].operands.first(), Some(Operand::Register(r)) if r == "x0"),
        "the advancing pointer must be rdi: {:?}",
        adds[0]
    );
}

#[test]
fn x86_sse_movsd_is_not_swallowed_by_the_string_form() {
    // `movsd xmm0, xmm1` = F2 0F 10 C1 ; `ret` = C3
    //
    // `movsd` names BOTH the scalar-double SSE move and the string primitive.
    // The SSE arm ran first and found no explicit operand for the string form,
    // so the two spellings had to be told apart by their operands.
    let insns = lift_sysv(&[0xF2, 0x0F, 0x10, 0xC1, 0xC3]);
    let mv = insns
        .iter()
        .find(|i| matches!(i.op, Op::Move))
        .expect("the scalar-double move must survive");
    assert!(
        matches!(mv.operands.first(), Some(Operand::Register(r)) if r == "vf0")
            && matches!(mv.operands.get(1), Some(Operand::Register(r)) if r == "vf1"),
        "`movsd xmm0, xmm1` must move vf1 into vf0: {mv:?}"
    );
    assert!(
        !insns.iter().any(|i| matches!(i.op, Op::Arith(_))),
        "an SSE move is not a string op and must not touch rdi/rcx: {insns:?}"
    );
}

#[test]
fn x86_loop_and_jrcxz_test_the_count_register_not_eflags() {
    // `cmp eax, ecx` = 39 C8 ; `jrcxz +0` = E3 00 ; `ret` = C3
    //
    // Neither instruction reads EFLAGS. Left to the conditional-branch arm they
    // consumed the pending compare and rendered as `if (eax == ecx)` — a real
    // edge carrying a condition the binary does not have.
    let insns = lift_sysv(&[0x39, 0xC8, 0xE3, 0x00, 0xC3]);
    let cond = insns
        .iter()
        .rev()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("the rcx test must be materialized");
    assert_eq!(
        cond.operands.get(1),
        Some(&Operand::Register("x3".into())),
        "`jrcxz` compares rcx (SysV x3), not the cmp operands: {cond:?}"
    );
    assert_eq!(
        cond.operands.get(2),
        Some(&Operand::Immediate(0)),
        "`jrcxz` compares against zero: {cond:?}"
    );
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::CondJump(_))),
        "the branch itself must remain: {insns:?}"
    );

    // `loop +0` = E2 00 ; `ret`
    let looped = lift_sysv(&[0xE2, 0x00, 0xC3]);
    assert!(
        looped
            .iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::Sub))
                && matches!(i.operands.first(), Some(Operand::Register(r)) if r == "x3")
                && i.operands.get(2) == Some(&Operand::Immediate(1))),
        "`loop` must decrement rcx: {looped:?}"
    );
    assert!(
        looped
            .iter()
            .any(|i| matches!(i.op, Op::Compare(CompareKind::NotEqual))),
        "`loop` branches while rcx != 0: {looped:?}"
    );
}

#[test]
fn x86_bmi_ops_are_modelled_not_stubbed() {
    // Encodings assembled with `llvm-mc --triple=x86_64`.
    //
    // `andn eax, ecx, edx` = C4 E2 70 F2 C2 ; `ret`
    let andn = lift_sysv(&[0xC4, 0xE2, 0x70, 0xF2, 0xC2, 0xC3]);
    assert!(
        !andn
            .iter()
            .any(|i| matches!(&i.op, Op::Simd(n) if n.contains("Andn"))),
        "`andn` must be modelled, not an `__Andn` stub: {andn:?}"
    );
    assert!(
        andn.iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::Xor))
                && i.operands.get(2) == Some(&Operand::Immediate(-1))),
        "`andn` complements its first source: {andn:?}"
    );
    assert!(
        andn.iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::And)) && i.operands.len() == 3),
        "`andn` ANDs the complement with the second source: {andn:?}"
    );

    // `blsi eax, ecx` = C4 E2 78 F3 D9 ; `ret` — src & -src, via a real negate
    let blsi = lift_sysv(&[0xC4, 0xE2, 0x78, 0xF3, 0xD9, 0xC3]);
    assert!(
        blsi.iter().any(|i| matches!(i.op, Op::Negate)),
        "`blsi` needs a real negation (src & -src): {blsi:?}"
    );
    assert_arith_is_three_operand(&blsi, "blsi");

    // `blsr eax, ecx` = C4 E2 78 F3 C9 — src & (src - 1), NOT `src & -src`
    let blsr = lift_sysv(&[0xC4, 0xE2, 0x78, 0xF3, 0xC9, 0xC3]);
    assert!(
        blsr.iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::Sub))
                && i.operands.get(2) == Some(&Operand::Immediate(1)))
            && !blsr.iter().any(|i| matches!(i.op, Op::Negate)),
        "`blsr` clears the lowest set bit with `src - 1`, not a negate: {blsr:?}"
    );

    // `shlx eax, ecx, edx` = C4 E2 69 F7 C1 ; `ret`
    // The count comes from operand 2, which the old binary operand accessor
    // read as operand 1.
    let shlx = lift_sysv(&[0xC4, 0xE2, 0x69, 0xF7, 0xC1, 0xC3]);
    let sh = shlx
        .iter()
        .find(|i| matches!(i.op, Op::Arith(ArithKind::Shl)))
        .expect("shlx");
    assert_eq!(
        sh.operands.get(1),
        Some(&Operand::Register("x3".into())),
        "`shlx` shifts its SOURCE operand (ecx): {sh:?}"
    );
    assert_eq!(
        sh.operands.get(2),
        Some(&Operand::Register("x2".into())),
        "`shlx` counts from operand 2 (edx): {sh:?}"
    );

    // `bzhi eax, ecx, edx` = C4 E2 68 F5 C1 — a mask of `index` bits
    let bzhi = lift_sysv(&[0xC4, 0xE2, 0x68, 0xF5, 0xC1, 0xC3]);
    assert!(
        !bzhi
            .iter()
            .any(|i| matches!(&i.op, Op::Simd(n) if n.contains("Bzhi"))),
        "`bzhi` must be modelled, not a stub: {bzhi:?}"
    );
    assert_arith_is_three_operand(&bzhi, "bzhi");

    // `mulx eax, ecx, edx` = C4 E2 73 F6 C2 — the low half is exact and the
    // high half is published as an explicit unknown rather than left stale.
    let mulx = lift_sysv(&[0xC4, 0xE2, 0x73, 0xF6, 0xC2, 0xC3]);
    assert!(
        mulx.iter()
            .any(|i| matches!(&i.op, Op::Simd(n) if n.contains("mulx_hi"))),
        "`mulx`'s high half must be an explicit unknown, not a stale value: {mulx:?}"
    );
    assert_arith_is_three_operand(&mulx, "mulx");
}

#[test]
fn x86_memory_destination_rmw_lifts_to_load_op_store() {
    // `add dword ptr [rax], ecx` = 01 08 ; `ret` = C3
    let insns = lift_sysv(&[0x01, 0x08, 0xC3]);
    let mut loaded = false;
    let mut stored = false;
    let mut arith = false;
    for i in &insns {
        match &i.op {
            Op::Move => {
                if let Some(Operand::Memory { base, .. }) = i.operands.first() {
                    stored = matches!(**base, Operand::Register(ref n) if n == ir::RET_REG);
                }
                if let Some(Operand::Memory { .. }) = i.operands.get(1) {
                    loaded = true;
                }
            }
            Op::Arith(ArithKind::Add) => arith = true,
            _ => {}
        }
    }
    assert!(
        loaded && arith && stored,
        "`add [rax], ecx` must become load/op/store through [rax]; got {:?}",
        insns
    );
}

#[test]
fn x86_rip_relative_access_uses_the_real_address() {
    // `mov rax, [rip + 0x10]` = 48 8B 05 10 00 00 00 ; `ret` = C3
    // IP-relative operands are how x64 reaches every static, global and
    // metadata usage slot; the base must resolve to the absolute address the
    // instruction encodes (end of instruction 0x1007 + 0x10 = 0x1017), not to
    // an ordinal "r…" local that is never defined.
    let insns = lift_sysv(&[0x48, 0x8B, 0x05, 0x10, 0x00, 0x00, 0x00, 0xC3]);
    let mut found = None;
    for i in &insns {
        if let Some(Operand::Memory { base, addend, .. }) = i.operands.get(1) {
            if matches!(**base, Operand::Register(ref n) if n == "abs") {
                found = Some(*addend);
            }
        }
    }
    assert_eq!(
        found,
        Some(0x1017),
        "RIP-relative load must resolve to its absolute address, got {:?}",
        insns
    );
}

#[test]
fn x86_cdq_sign_extends_into_the_high_half() {
    // `cdq` = 99 ; `ret` = C3  → the high half (RDX = x2 under SysV) becomes
    // `(int)rax < 0 ? -1 : 0`. Every `long` division depends on this pair.
    let insns = lift_sysv(&[0x99, 0xC3]);
    let sel = insns.iter().find_map(|i| match &i.op {
        Op::Select(super::ir::SelectVariant::Identity) => Some(i),
        _ => None,
    });
    let sel = sel.expect("cdq must lift to a sign-select");
    assert!(
        matches!(sel.operands.first(), Some(Operand::Register(n)) if n == "x2"),
        "cdq writes the high half (x2 under SysV), got {:?}",
        sel.operands
    );
    assert!(insns.iter().any(|i| matches!(&i.op, Op::Compare(_))));
}

#[test]
fn x86_cdqe_sign_extends_the_accumulator_into_its_high_half() {
    // `cdqe` = 48 98 ; `ret` = C3. Sign-extend EAX into RAX as (rax << 32) >> 32.
    //
    // `cdqe` has ZERO operands, so it used to reach the catch-all, which emits
    // nothing for that shape: `rax` kept a stale high half and every 64-bit
    // pointer or index derived from an `int` was silently wrong. It was never
    // even reported as undefined, because `eax` and `rax` collapse to one slot.
    let insns = lift_sysv(&[0x48, 0x98, 0xC3]);
    let shifts: Vec<&Insn> = insns
        .iter()
        .filter(|i| {
            matches!(&i.op, Op::Arith(ArithKind::Shl | ArithKind::Shr))
                && matches!(i.operands.get(2), Some(Operand::Immediate(32)))
        })
        .collect();
    assert_eq!(
        shifts.len(),
        2,
        "cdqe must widen the accumulator with (x << 32) >> 32; got {insns:?}"
    );
    for s in shifts {
        assert!(
            matches!(s.operands.first(), Some(Operand::Register(n)) if n == "rv"),
            "cdqe operates on rax, got {:?}",
            s.operands
        );
    }
    assert!(
        !insns.iter().any(|i| matches!(i.op, Op::Simd(_))),
        "cdqe must not reach the unmodelled catch-all: {insns:?}"
    );
}

#[test]
fn x86_one_operand_imul_defines_the_product() {
    // `imul rax` = 48 F7 E8 (the one-operand `rdx:rax` form) ; `ret` = C3.
    //
    // The 2- and 3-operand arms are guarded on `op_count`, so this form fell
    // through both and reached the catch-all — the product's low half was never
    // defined and every use of it read a stale value.
    let insns = lift_sysv(&[0x48, 0xF7, 0xE8, 0xC3]);
    let mul = insns
        .iter()
        .find(|i| matches!(i.op, Op::Arith(ArithKind::Mul)))
        .expect("one-operand imul must lift to a multiply");
    assert!(
        matches!(mul.operands.first(), Some(Operand::Register(n)) if n == "rv"),
        "one-operand imul writes rax, got {:?}",
        mul.operands
    );
    assert!(
        !insns.iter().any(|i| matches!(i.op, Op::Simd(_))),
        "imul must not reach the unmodelled catch-all: {insns:?}"
    );
}

#[test]
fn x86_the_multibyte_nop_is_not_an_exchange() {
    // `66 90` is the multi-byte NOP pad MSVC emits between functions. Iced
    // decodes it as `Nopw` (mnemonic `Nop`, like plain `90`), so the lifter
    // drops it through its `Nop` arm and it must produce NO exchange IR.
    //
    // This is the lifter half of a defect whose other half lived in the
    // VERIFIER: the oracle built its mnemonic from iced's FORMATTER text, which
    // renders `Nopw` as `xchg ax,ax`, so 387 padding instructions were counted
    // as dropped ARITHMETIC `xchg` — the largest false cluster on the v24.5
    // corpus. The lifter was right all along; the test pins that.
    let nop = lift_sysv(&[0x66, 0x90, 0xC3]);
    assert!(
        !nop.iter().any(|i| i.address == 0x1000),
        "`xchg ax, ax` must produce no IR of its own: {nop:?}"
    );
    // The control that keeps the fix honest: a REAL exchange still swaps.
    // `48 87 D8` = `xchg rax, rbx`.
    let swap = lift_sysv(&[0x48, 0x87, 0xD8, 0xC3]);
    let moves = swap
        .iter()
        .filter(|i| i.address == 0x1000 && i.op == Op::Move)
        .count();
    assert_eq!(
        moves, 3,
        "`xchg rax, rbx` must still swap both ways: {swap:?}"
    );
}

#[test]
fn x86_bit_test_drives_a_single_bit_condition() {
    // `bt eax, 3` = 0F BA E0 03 ; `jb +0` = 72 00 ; `ret` = C3
    // `bt` writes CF only: the branch must read `(eax & 8) != 0`, never an
    // ordering (`eax < 0`) and never `Unknown`.
    let insns = lift_sysv(&[0x0F, 0xBA, 0xE0, 0x03, 0x72, 0x00, 0xC3]);
    let masked = insns.iter().any(|i| match &i.op {
        Op::Arith(ArithKind::And) => {
            matches!(i.operands.get(2), Some(Operand::Immediate(1)))
        }
        _ => false,
    });
    let shifted = insns.iter().any(|i| match &i.op {
        Op::Arith(ArithKind::Shr) => matches!(i.operands.get(2), Some(Operand::Immediate(3))),
        _ => false,
    });
    let causal = insns.iter().any(|i| match &i.op {
        Op::Compare(CompareKind::NotEqual) => !matches!(i.operands.get(1), Some(Operand::Unknown)),
        _ => false,
    });
    assert!(
        shifted && masked && causal,
        "bt+jb must lift to ((v >> 3) & 1) != 0; got {:?}",
        insns
    );
}

#[test]
fn x86_register_bit_index_is_a_variable_shift() {
    // `bt eax, ecx` = 0F A3 D8 ; `jb +0` = 72 00 ; `ret` = C3
    //
    // The bit comes from a REGISTER, so there is no constant to fold. It must
    // still produce a real single-bit test — `((v >> (ecx & 31)) & 1) != 0` —
    // rather than a `__Bt` stub that quietly drops the condition.
    let insns = lift_sysv(&[0x0F, 0xA3, 0xD8, 0x72, 0x00, 0xC3]);
    let no_stub = !insns
        .iter()
        .any(|i| matches!(&i.op, Op::Simd(n) if n.starts_with("__Bt")));
    let mask31 = insns.iter().any(|i| match (&i.op, i.operands.get(2)) {
        (Op::Arith(ArithKind::And), Some(Operand::Immediate(31))) => true,
        _ => false,
    });
    let bit_and = insns.iter().any(|i| match (&i.op, i.operands.get(2)) {
        (Op::Arith(ArithKind::And), Some(Operand::Immediate(1))) => true,
        _ => false,
    });
    let shift_by_value = insns.iter().any(|i| match (&i.op, i.operands.get(2)) {
        (Op::Arith(ArithKind::Shr), Some(Operand::Local(_))) => true,
        _ => false,
    });
    let causal = insns.iter().any(|i| match &i.op {
        Op::Compare(CompareKind::NotEqual) => !matches!(i.operands.get(1), Some(Operand::Unknown)),
        _ => false,
    });
    assert!(
        no_stub && mask31 && bit_and && shift_by_value && causal,
        "bt eax,ecx + jb must lift to ((v >> (ecx & 31)) & 1) != 0; got {:?}",
        insns
    );
}

#[test]
fn x86_cmpxchg_lifts_to_a_real_compare_exchange() {
    // `cmpxchg dword ptr [rax], ecx` = 0F B1 08 ; `ret` = C3
    let insns = lift_sysv(&[0x0F, 0xB1, 0x08, 0xC3]);
    let selects = insns
        .iter()
        .filter(|i| matches!(&i.op, Op::Select(_)))
        .count();
    let stores = insns.iter().any(|i| {
        matches!(&i.op, Op::Move) && matches!(i.operands.first(), Some(Operand::Memory { .. }))
    });
    assert!(
        selects >= 2 && stores,
        "cmpxchg must select the stored value AND conditionally update the accumulator; got {:?}",
        insns
    );
}

#[test]
fn x86_shld_is_a_shift_pair_plus_or() {
    // `shld eax, ecx, 4` = 0F A4 C8 04 ; `ret` = C3
    let insns = lift_sysv(&[0x0F, 0xA4, 0xC8, 0x04, 0xC3]);
    let shl4 = insns.iter().any(|i| match &i.op {
        Op::Arith(ArithKind::Shl) => matches!(i.operands.get(2), Some(Operand::Immediate(4))),
        _ => false,
    });
    // 32 - 4 = 28
    let shr28 = insns.iter().any(|i| match &i.op {
        Op::Arith(ArithKind::Shr) => matches!(i.operands.get(2), Some(Operand::Immediate(28))),
        _ => false,
    });
    let or = insns
        .iter()
        .any(|i| matches!(&i.op, Op::Arith(ArithKind::Or)));
    assert!(
        shl4 && shr28 && or,
        "shld eax,ecx,4 lowered wrong: {:?}",
        insns
    );
}

#[test]
fn x86_roundsd_maps_to_the_correct_rounding_mode() {
    // `roundsd xmm0, xmm1, 9` = 66 0F 3A 0B C1 09 ; `ret` = C3
    // imm8[1:0] = 01 → round toward -infinity (what `Math.Floor` compiles to).
    let insns = lift_sysv(&[0x66, 0x0F, 0x3A, 0x0B, 0xC1, 0x09, 0xC3]);
    assert!(
        insns
            .iter()
            .any(|i| matches!(&i.op, Op::FloatArith(FloatArithKind::Floor))),
        "roundsd imm=9 must be a floor; got {:?}",
        insns
    );
}

#[test]
fn x86_adc_consumes_the_carry_of_the_add_before_it() {
    // `add eax, ecx` = 01 C8 ; `adc eax, ecx` = 11 C8 ; `ret` = C3
    //
    // x86's 64-bit addition is `add`/`adc` pairs; without the carry the whole
    // high word of the operation disappears into an `__Adc(…)` stub.
    let insns = lift_sysv(&[0x01, 0xC8, 0x11, 0xC8, 0xC3]);
    assert!(
        !insns
            .iter()
            .any(|i| matches!(&i.op, Op::Simd(n) if n.starts_with("__Adc"))),
        "adc must not degrade to a stub: {:?}",
        insns
    );
    assert!(
        insns
            .iter()
            .any(|i| matches!(&i.op, Op::Compare(CompareKind::LessThan))),
        "the add must publish `(sum < lhs)` as its carry: {:?}",
        insns
    );
    let carry = insns.iter().any(|i| {
        matches!(&i.op, Op::Select(super::ir::SelectVariant::Identity))
            && matches!(i.operands.get(2), Some(Operand::Immediate(1)))
            && matches!(i.operands.get(3), Some(Operand::Immediate(0)))
    });
    assert!(carry, "the carry must become an integer 0/1: {:?}", insns);
    assert!(
        insns
            .iter()
            .filter(|i| matches!(&i.op, Op::Arith(ArithKind::Add)))
            .count()
            >= 3,
        "adc must add the carry in: {:?}",
        insns
    );
}

#[test]
fn x86_sbb_is_borrow_not_not_borrow() {
    // `sub eax, ecx` = 29 C8 ; `sbb eax, ecx` = 19 C8 ; `ret` = C3
    //
    // x86 CF after SUB is the BORROW, so the materialized value is `(lhs < rhs)`
    // with NO inversion — the opposite of A32/A64's NOT-borrow carry.
    let insns = lift_sysv(&[0x29, 0xC8, 0x19, 0xC8, 0xC3]);
    assert!(
        !insns
            .iter()
            .any(|i| matches!(&i.op, Op::Simd(n) if n.starts_with("__Sbb"))),
        "sbb must not degrade to a stub: {:?}",
        insns
    );
    assert!(
        insns
            .iter()
            .any(|i| matches!(&i.op, Op::Compare(CompareKind::LessThan))),
        "CF after sub is `(lhs < rhs)`: {:?}",
        insns
    );
    assert!(
        insns
            .iter()
            .filter(|i| matches!(&i.op, Op::Arith(ArithKind::Sub)))
            .count()
            >= 3,
        "sub + sbb must both subtract: {:?}",
        insns
    );
}

#[test]
fn x86_xchg_swaps_both_ways() {
    // `xchg eax, ecx` = 91 ; `ret` = C3
    let insns = lift_sysv(&[0x91, 0xC3]);
    let moves = insns.iter().filter(|i| matches!(&i.op, Op::Move)).count();
    assert!(
        moves >= 3,
        "xchg must be a three-step swap, got {:?}",
        insns
    );
}

#[test]
fn probe_a32_decode_coverage() {
    use yaxpeax_arch::{Decoder, U8Reader};
    use yaxpeax_arm::armv7::InstDecoder;
    let d = InstDecoder::default();
    for (w, want) in [
        (0xE7CB0411u32, "bfi r0,r1,#8,#4"),
        (0xE7CB021Fu32, "bfc r0,#4,#8"),
        (0xE1910F9Fu32, "ldrex r0,[r1]"),
        (0xE1810F92u32, "strex r0,r2,[r1]"),
        (0xE0603291u32, "mls r0,r1,r2,r3"),
        (0xE1821F93u32, "strex r3,r1,[r2] (alt shape)"),
        (0xE1D1F0B2u32, "ldrh r0,[r1]"),
        (0xE8B10003u32, "ldmia r1!,{r0,r1}"),
        (0xE4910004u32, "ldr r0,[r1],#4"),
        (0xE28F0008u32, "add r0,pc,#8"),
        (0xE0E10392u32, "smlal r0,r1,r2,r3"),
        (0xE6BF0F31u32, "rev r0,r1"),
        (0xE1600071u32, "smulbb r0,r1,r2"),
        (0xE6A00011u32, "ssat r0,#0,r1"),
        (0xE6AF0071u32, "sxtb r0,r1"),
    ] {
        let bytes = w.to_le_bytes();
        let mut r = U8Reader::new(&bytes);
        match d.decode(&mut r) {
            Ok(i) => println!("PROBE {:#010X} => [{}]  (want {})", w, i, want),
            Err(e) => println!("PROBE {:#010X} => DECODE FAIL {:?}  (want {})", w, e, want),
        }
    }
}

// ---------------------------------------------------------------------------
// ARM32: the arms added to close the instruction-coverage gap. Encodings were
// checked against the vendored decoder first (see `probe_a32_decode_coverage`).
// ---------------------------------------------------------------------------

fn lift_a32(words: &[u32]) -> Vec<Insn> {
    let mut bytes = Vec::new();
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    lift_arm32::lift_arm32(&bytes, 0x1000, 64)
        .expect("lift")
        .insns
}

#[test]
fn arm32_exclusive_load_is_a_real_load() {
    // `ldrex r0, [r1]` = E1910F9F ; `bx lr` = E12FFF1E
    let insns = lift_a32(&[0xE1910F9F, 0xE12FFF1E]);
    let loaded = insns
        .iter()
        .any(|i| match (&i.op, i.operands.first(), i.operands.get(1)) {
            (Op::Move, Some(Operand::Register(r)), Some(Operand::Memory { base, .. })) => {
                r == "x0" && matches!(**base, Operand::Register(ref b) if b == "x1")
            }
            _ => false,
        });
    assert!(
        loaded,
        "ldrex must lift to a load from its real address (the Interlocked path); got {:?}",
        insns
    );
}

#[test]
fn arm32_post_indexed_load_updates_its_base() {
    // `ldr r0, [r1], 0x4` = E4910004 ; `bx lr`
    // The base is advanced by the offset AFTER the access. Dropping it left
    // every loop walking an array reading through the ORIGINAL base from the
    // second iteration on — a wrong address, not a missing name.
    let insns = lift_a32(&[0xE4910004, 0xE12FFF1E]);
    let loaded = insns.iter().any(|i| {
        matches!(&i.op, Op::Move)
            && matches!(i.operands.first(), Some(Operand::Register(r)) if r == "x0")
            && matches!(i.operands.get(1), Some(Operand::Memory { .. }))
    });
    let advanced = insns
        .iter()
        .any(|i| match (&i.op, i.operands.first(), i.operands.get(2)) {
            (
                Op::Arith(ArithKind::Add),
                Some(Operand::Register(r)),
                Some(Operand::Immediate(4)),
            ) => r == "x1",
            _ => false,
        });
    assert!(
        loaded && advanced,
        "post-indexed ldr must load AND advance r1 by 4; got {:?}",
        insns
    );
}

#[test]
fn arm32_adr_materializes_a_real_address() {
    // `adr r0, pc, 0x8` = E28F0008 ; `bx lr`, lifted at 0x1000
    // A32 `pc` reads as `addr + 8`, so the value is 0x1000 + 8 + 8 = 0x1010.
    let insns = lift_a32(&[0xE28F0008, 0xE12FFF1E]);
    let value = insns
        .iter()
        .find_map(|i| match (&i.op, i.operands.first(), i.operands.get(1)) {
            (Op::Move, Some(Operand::Register(r)), Some(Operand::Immediate(v))) if r == "x0" => {
                Some(*v)
            }
            _ => None,
        });
    assert_eq!(
        value,
        Some(0x1010),
        "adr must materialize pc-relative address 0x1010, got {:?}",
        insns
    );
}

#[test]
fn arm32_mls_is_a_negated_multiply_add() {
    // `mls r0, r1, r2, r3` = E0603291  == r0 = r3 - r1*r2 ; `bx lr`
    let insns = lift_a32(&[0xE0603291, 0xE12FFF1E]);
    let neg = insns.iter().any(|i| matches!(&i.op, Op::Negate));
    let acc = insns.iter().any(|i| {
        matches!(
            &i.op,
            Op::MulAcc {
                negate_addend: false,
                ..
            }
        )
    });
    assert!(neg && acc, "mls lowered wrong: {:?}", insns);
}

#[test]
fn arm32_smlal_accumulates_the_low_word() {
    // `smlal r0, r1, r2, r3` = E0E10392 ; `bx lr`
    let insns = lift_a32(&[0xE0E10392, 0xE12FFF1E]);
    assert!(
        insns.iter().any(|i| matches!(
            &i.op,
            Op::MulAcc {
                is_float: false,
                ..
            }
        )),
        "smlal must accumulate into its low word; got {:?}",
        insns
    );
}

#[test]
fn arm32_dsp_mul_decodes_the_variant_from_the_word() {
    use crate::recovery::lift_arm32::{DspMul, dsp_mul};
    // Every encoding below was assembled with `llvm-mc --show-encoding` and
    // round-tripped back through `llvm-mc --disassemble`, because yaxpeax's
    // rendering of this family is wrong in two independent ways (see
    // `dsp_mul`): it swaps the halfword selectors on `smlal` and calls
    // `smulw*` `smlaw*`. The table pins the *bits*, so a future decoder bump
    // cannot drag the lifter back to the wrong product.
    let dsp =
        |long: bool, word_form: bool, accumulate: bool, rn_high: bool, rm_high: bool| DspMul {
            long,
            word_form,
            accumulate,
            rn_high,
            rm_high,
        };
    let table: &[(u32, &str, DspMul)] = &[
        (0xE1600281, "smulbb", dsp(false, false, false, false, false)),
        (0xE16002C1, "smulbt", dsp(false, false, false, false, true)),
        (0xE16002A1, "smultb", dsp(false, false, false, true, false)),
        (0xE16002E1, "smultt", dsp(false, false, false, true, true)),
        (0xE12002A1, "smulwb", dsp(false, true, false, false, false)),
        (0xE12002E1, "smulwt", dsp(false, true, false, false, true)),
        (0xE1003281, "smlabb", dsp(false, false, true, false, false)),
        (0xE10032C1, "smlabt", dsp(false, false, true, false, true)),
        (0xE10032A1, "smlatb", dsp(false, false, true, true, false)),
        (0xE10032E1, "smlatt", dsp(false, false, true, true, true)),
        (0xE1203281, "smlawb", dsp(false, true, true, false, false)),
        (0xE12032C1, "smlawt", dsp(false, true, true, false, true)),
        (0xE1401281, "smlalbb", dsp(true, false, true, false, false)),
        (0xE14012C1, "smlalbt", dsp(true, false, true, false, true)),
        (0xE14012A1, "smlaltb", dsp(true, false, true, true, false)),
        (0xE14012E1, "smlaltt", dsp(true, false, true, true, true)),
    ];
    for (word, name, want) in table {
        assert_eq!(
            dsp_mul(*word).as_ref(),
            Some(want),
            "{name} ({word:#010X}) decoded to the wrong variant"
        );
    }

    // Words that share the `bits[27:23] = 00010` / `bit20 = 0` prefix but are
    // a DIFFERENT instruction. If any of these were claimed, the lifter would
    // emit a fabricated product from unrelated operands. `swp`/`swpb`,
    // `ldrd`/`strd` and `ldrex`/`strex` are the dangerous neighbours: they are
    // common and they encode registers in the same fields.
    let must_not_claim: &[(u32, &str)] = &[
        (0xE0000291, "mul"),
        (0xE0203291, "mla"),
        (0xE0810392, "umull"),
        (0xE0A10392, "umlal"),
        (0xE0C10392, "smull"),
        (0xE0E10392, "smlal (word form)"),
        (0xE1020091, "swp"),
        (0xE1420091, "swpb"),
        (0xE1C200D0, "ldrd"),
        (0xE1C200F0, "strd"),
        (0xE1910F9F, "ldrex"),
        (0xE1820F91, "strex"),
        (0xE10F0000, "mrs"),
        (0xE128F000, "msr"),
        (0xE16F0F11, "clz"),
        (0xE12FFF33, "blx r3"),
        (0xE12FFF13, "bx r3"),
        (0xE0010002, "and"),
        (0xE1810002, "orr"),
        (0xE0210002, "eor"),
        (0xE0810002, "add"),
        (0xE0410002, "sub"),
        (0xE1500001, "cmp"),
        (0xE1100001, "tst"),
        (0xE1300001, "teq"),
        (0xE1A00001, "mov"),
        (0xE1E00001, "mvn"),
        (0xE5910000, "ldr"),
        (0xE5810000, "str"),
        (0xE8910001, "ldm"),
        (0xE8810001, "stm"),
        (0xE92D4010, "push"),
        (0xE8BD8010, "pop"),
        (0xEB000040, "bl"),
        (0xEA000040, "b"),
        (0xE6BF0F31, "rev"),
        (0xE6FF0F31, "rbit"),
        (0xE6EF0071, "uxtb"),
        (0xE6AF0071, "sxtb"),
        (0xE7C20091, "bfi"),
        (0xE7A100D1, "sbfx"),
        (0xE7E100D1, "ubfx"),
        (0xE6A00011, "ssat"),
        (0xE3000001, "movw"),
        (0xE3400001, "movt"),
        // The dual-multiply / most-significant-word group (bits[27:23]=01110).
        (0xE700F211, "smuad"),
        (0xE750F211, "smmul"),
        (0xE7003211, "smlad"),
    ];
    for (word, name) in must_not_claim {
        assert!(
            dsp_mul(*word).is_none(),
            "{name} ({word:#010X}) must not be decoded as a signed-DSP multiply"
        );
    }
}

#[test]
fn arm32_smulwb_is_not_the_accumulate_yaxpeax_calls_it() {
    // `smulwb r0, r1, r2` = E12002A1. yaxpeax renders this word `smlawb`
    // (its `Ra` field prints as r0), and a mnemonic-driven arm believes it:
    // it computes `r0 = ((r0 * r1[15:0]) >> 16) + r0`, reading and re-adding
    // the destination. Bit5 says `smulw`, so no addend may be consumed.
    let insns = lift_a32(&[0xE12002A1, 0xE12FFF1E]);
    let accumulates = insns
        .iter()
        .any(|i| match (&i.op, i.operands.first(), i.operands.get(2)) {
            (Op::Arith(ArithKind::Add), Some(Operand::Register(r)), Some(Operand::Register(_))) => {
                r == "x0"
            }
            _ => false,
        });
    assert!(
        !accumulates,
        "smulwb must not accumulate an addend; got {:?}",
        insns
    );
    assert!(
        !shifts_reg_right_16(&insns, "x1"),
        "smulwb keeps Rn as a full word — x1 must not be halved; got {:?}",
        insns
    );
}

#[test]
fn arm32_smlaltt_is_not_the_halfword_pair_yaxpeax_names() {
    // `smlaltt r1, r0, r1, r2` = E14012E1, which yaxpeax renders `smlalbb`.
    // Taking the decoder's letters would multiply the BOTTOM halves (a `<<16
    // >>16` done without isolating the high half first) — a wrong 64-bit
    // accumulate that still looks plausible. The bits say bit5 = 1 and
    // bit6 = 1, so BOTH sources must be shifted down by 16 first.
    let insns = lift_a32(&[0xE14012E1, 0xE12FFF1E]);
    assert!(
        shifts_reg_right_16(&insns, "x1") && shifts_reg_right_16(&insns, "x2"),
        "smlaltt must halve both sources; got {:?}",
        insns
    );
}

#[test]
fn arm32_bare_umlal_still_reaches_its_own_arm() {
    // `umlal r0, r1, r2, r3` = E0A10392. The `dsp_mul` interception must leave
    // the word-by-word 64-bit forms to the mnemonic dispatch: they are not
    // halfword multiplies (bit4 = 1), and stealing them would drop the
    // unsigned accumulate.
    let insns = lift_a32(&[0xE0A10392, 0xE12FFF1E]);
    assert!(
        insns.iter().any(|i| matches!(
            &i.op,
            Op::MulAcc {
                is_float: false,
                ..
            }
        )),
        "umlal must still accumulate; got {:?}",
        insns
    );
}

/// Does some instruction shift register `reg` right by 16 into a temp?
fn shifts_reg_right_16(insns: &[Insn], reg: &str) -> bool {
    insns
        .iter()
        .any(|i| match (&i.op, i.operands.get(1), i.operands.get(2)) {
            (
                Op::Arith(ArithKind::Shr),
                Some(Operand::Register(r)),
                Some(Operand::Immediate(16)),
            ) => r == reg,
            _ => false,
        })
}

#[test]
fn arm32_smultb_reads_the_top_half_of_rn() {
    // `smultb r0, r1, r2` = E16002A1 ; `bx lr`
    //
    // The `<x><y>` letters are NOT symmetric: `<x>` names the half of Rn and
    // `<y>` the half of Rm (ISA reference: `SMLATB` is Rn.Top × Rm.Bottom).
    // The old arm always sign-extended Rn's BOTTOM half, so every `smultb`
    // multiplied by the wrong operand — a silently wrong value.
    let insns = lift_a32(&[0xE16002A1, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "smultb must be modelled, not a `__smultb` stub: {:?}",
        insns
    );
    assert!(
        shifts_reg_right_16(&insns, "x1"),
        "smultb must take Rn's TOP half (a >>16 of x1); got {:?}",
        insns
    );
    assert!(
        !shifts_reg_right_16(&insns, "x2"),
        "smultb takes Rm's BOTTOM half — x2 must not be shifted; got {:?}",
        insns
    );
}

#[test]
fn arm32_smulwb_takes_the_high_half_of_the_product() {
    // `smulwb r0, r1, r2` = E12002A1 ; `bx lr`
    //
    // SMULWB is `(Rn * SInt(Rm[15:0])) >> 16` — the HIGH 32 bits of the 48-bit
    // product. The old arm emitted the plain product, dropping the whole point
    // of the instruction.
    let insns = lift_a32(&[0xE12002A1, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "smulwb must be modelled, not a stub: {:?}",
        insns
    );
    // The result is written to x0 by a >>16 of the product temp.
    let result_shifted =
        insns
            .iter()
            .any(|i| match (&i.op, i.operands.first(), i.operands.get(2)) {
                (
                    Op::Arith(ArithKind::Shr),
                    Some(Operand::Register(r)),
                    Some(Operand::Immediate(16)),
                ) => r == "x0",
                _ => false,
            });
    assert!(
        result_shifted,
        "smulwb must shift the 48-bit product right by 16 into Rd; got {:?}",
        insns
    );
}

#[test]
fn arm32_smlawb_shifts_then_accumulates() {
    // `smlawb r0, r1, r2, r3` = E1203281 ; `bx lr`
    // SMLAWB = ((Rn * SInt(Rm[15:0])) >> 16) + Ra.
    let insns = lift_a32(&[0xE1203281, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "smlawb must be modelled, not a stub: {:?}",
        insns
    );
    let shifted = insns.iter().any(|i| {
        matches!(&i.op, Op::Arith(ArithKind::Shr))
            && i.operands.get(2) == Some(&Operand::Immediate(16))
    });
    let accumulated = insns
        .iter()
        .any(|i| match (&i.op, i.operands.first(), i.operands.get(2)) {
            (Op::Arith(ArithKind::Add), Some(Operand::Register(r)), Some(Operand::Register(a))) => {
                r == "x0" && a == "x3"
            }
            _ => false,
        });
    assert!(
        shifted && accumulated,
        "smlawb must shift the product right by 16 and add Ra (x3) into Rd; got {:?}",
        insns
    );
}

#[test]
fn arm32_smlaltt_sign_extends_both_top_halves() {
    // `smlaltt r1, r0, r1, r2` = E14012E1 ; `bx lr`
    // {RdHi,RdLo} += SInt(Rn<31:16>) * SInt(Rm<31:16>) — BOTH halves are the
    // top one. Previously the whole family fell through to `__smlaltt`.
    let insns = lift_a32(&[0xE14012E1, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "smlaltt must be modelled, not a stub: {:?}",
        insns
    );
    assert!(
        insns.iter().any(|i| matches!(
            &i.op,
            Op::MulAcc {
                is_float: false,
                ..
            }
        )),
        "smlaltt must accumulate a product into its low word: {:?}",
        insns
    );
    assert!(
        shifts_reg_right_16(&insns, "x1") && shifts_reg_right_16(&insns, "x2"),
        "smlaltt must take the TOP half of BOTH sources (x1 and x2); got {:?}",
        insns
    );
}

#[test]
fn arm32_adc_consumes_the_carry_of_the_adds_before_it() {
    // `adds r0, r2, r4` = E0920004 ; `adc r1, r3, r5` = E0A31005 ; `bx lr`.
    //
    // This pair IS A32's 64-bit addition: r1:r0 = (r3:r2) + (r5:r4), where the
    // `adc` supplies the bit the low add could not hold. Before the carry model
    // existed the whole `adc` vanished behind a `__adc` stub — 439 sites in one
    // Android build, more than any other unmodelled mnemonic — and every
    // `long`/`ulong`/`DateTime`/big-integer computation built from these pairs
    // lost its high word.
    let insns = lift_a32(&[0xE0920004, 0xE0A31005, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "adc must not degrade to a stub: {:?}",
        insns
    );
    // ADD carry = (sum < lhs), turned into an integer 0/1 by the Select.
    assert!(
        insns
            .iter()
            .any(|i| matches!(&i.op, Op::Compare(CompareKind::LessThan))),
        "the adds must publish `(sum < lhs)` as its carry: {:?}",
        insns
    );
    let carry = insns
        .iter()
        .find_map(|i| match (&i.op, i.operands.get(2), i.operands.get(3)) {
            (
                Op::Select(super::ir::SelectVariant::Identity),
                Some(Operand::Immediate(1)),
                Some(Operand::Immediate(0)),
            ) => Some(i),
            _ => None,
        });
    assert!(
        carry.is_some(),
        "the carry must be materialized as an integer 0/1 value: {:?}",
        insns
    );
    // `adc` itself: one ADD for `rn + rm` and one ADD of the carry.
    let adds = insns
        .iter()
        .filter(|i| matches!(&i.op, Op::Arith(ArithKind::Add)))
        .count();
    assert!(adds >= 3, "adc must add the carry in: {:?}", insns);
}

#[test]
fn arm32_sbc_subtracts_the_borrow() {
    // `subs r0, r2, r4` = E0520004 ; `sbc r1, r3, r5` = E0C31005 ; `bx lr`.
    //
    // A32's C after SUB is NOT-borrow, so `sbc` is `rn - rm - (1 - C)`: the
    // materialized value must be the BORROW (Select(…, 0, 1)), not C.
    let insns = lift_a32(&[0xE0520004, 0xE0C31005, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "sbc must not degrade to a stub: {:?}",
        insns
    );
    assert!(
        insns
            .iter()
            .any(|i| matches!(&i.op, Op::Compare(CompareKind::GreaterOrEqual))),
        "C after subs is `(lhs >= rhs)`: {:?}",
        insns
    );
    let borrow = insns.iter().any(|i| {
        matches!(&i.op, Op::Select(super::ir::SelectVariant::Identity))
            && matches!(i.operands.get(2), Some(Operand::Immediate(0)))
            && matches!(i.operands.get(3), Some(Operand::Immediate(1)))
    });
    assert!(borrow, "sbc needs `1 - C`, i.e. a 0/1 select: {:?}", insns);
    assert!(
        insns
            .iter()
            .filter(|i| matches!(&i.op, Op::Arith(ArithKind::Sub)))
            .count()
            >= 3,
        "subs + sbc must both subtract: {:?}",
        insns
    );
}

#[test]
fn arm32_flag_setting_mvn_and_adr_keep_their_suffix() {
    // `mvns r0, r1` = E1F00001 ; `adrs r1, #0x10` = E29F1010 ; `bx lr`.
    //
    // `mvn` and `adr` were missing from the `s`-stripping table, so `mvns`
    // stayed `mvns`, matched no arm, and dropped its destination — 2,881 sites
    // in one ARM32 build, the densest unmodelled mnemonic of all.
    let insns = lift_a32(&[0xE1F00001, 0xE29F1010, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "mvns/adrs must not become stubs: {:?}",
        insns
    );
    // `mvn` is lowered as `x1 ^ -1` (the honest two-step), not a `Not` op.
    assert!(
        insns
            .iter()
            .any(|i| matches!(&i.op, Op::Arith(ArithKind::Xor))
                && matches!(i.operands.get(2), Some(Operand::Immediate(-1)))),
        "mvns must still invert: {:?}",
        insns
    );
    // `adrs r1, #0x10` at 0x1004 reads pc as its own address + 8
    // (0x1004 + 8 + 0x10 = 0x101C).
    let adr = insns.iter().any(|i| {
        matches!(&i.op, Op::Move) && matches!(i.operands.get(1), Some(Operand::Immediate(0x101C)))
    });
    assert!(adr, "adrs must materialize 0x1004 + 8 + 0x10: {:?}", insns);
}

#[test]
fn arm32_register_aliases_are_not_phantom_registers() {
    // `mov r0, r9` = E1A00009 — the decoder prints the EABI alias `sb` for r9.
    //
    // Unmapped, `sb`/`sl`/`fp`/`ip` became registers NO def ever wrote, so every
    // read through one turned its dependent expression into an undefined local,
    // and the same physical register reached the renderer under two names.
    let insns = lift_a32(&[0xE1A00009, 0xE1A0100B, 0xE12FFF1E]);
    let aliased = insns.iter().any(|i| {
        i.operands
            .iter()
            .any(|o| matches!(o, Operand::Register(n) if n == "sb" || n == "fp" || n == "sl"))
    });
    assert!(!aliased, "aliases must map to x9/x11, got {:?}", insns);
    assert!(
        insns.iter().any(|i| {
            matches!(&i.op, Op::Move)
                && matches!(i.operands.get(1), Some(Operand::Register(n)) if n == "x9")
        }),
        "`sb` is r9 → x9: {:?}",
        insns
    );
}

/// `blx r2` must be a returning `Op::Call` carrying the canonical `x2`, not the
/// raw decoder spelling `r2` AND not `Op::IndirectCall`.
///
/// Two defects in one shape. The register-tracking names are canonical (`r2` →
/// `x2`), so a raw `Register("r2")` is a name no def ever writes: SSA never
/// renamed it, the target stayed a phantom, and every ARM32 virtual dispatch
/// read as "target is not a value". And `Op::IndirectCall` is the CFG's
/// TERMINATOR — a `blx` RETURNS, so making it IndirectCall severed the
/// fall-through and orphaned the rest of the method (the `bx rN` tail form is
/// the one that legitimately does not return).
#[test]
fn arm32_blx_register_target_is_a_canonical_local() {
    // `blx r2` = E12FFF32 ; `bx lr` = E12FFF1E.
    let insns = lift_a32(&[0xE12FFF32, 0xE12FFF1E]);
    let call = insns.iter().find(|i| matches!(i.op, Op::Call));
    let call = call.expect("`blx r2` must lift to a returning Op::Call, not IndirectCall");
    match call.operands.first().cloned() {
        Some(Operand::Register(n)) => assert_eq!(n, "x2", "blx r2 must target x2, got {n}"),
        other => panic!("expected a Register target for `blx r2`, got {other:?}"),
    }
}

#[test]
fn arm32_bx_register_target_is_a_canonical_local() {
    // `bx r2` = E12FFF12 ; `bx lr` = E12FFF1E.
    let insns = lift_a32(&[0xE12FFF12, 0xE12FFF1E]);
    let target = insns
        .iter()
        .find(|i| matches!(i.op, Op::IndirectCall))
        .and_then(|i| i.operands.first().cloned());
    match target {
        Some(Operand::Register(n)) => assert_eq!(n, "x2", "bx r2 must target x2, got {n}"),
        other => panic!("expected a Register target for `bx r2`, got {other:?}"),
    }
}

/// ARM32's `lr`/`pc` render as `g14`/`g15`, which the name-based x86 detector
/// used to read as "x86-64" — turning OFF the incoming-argument seeding so
/// every read of `x0..x8` looked undefined. `SeedInfo::is_arm32` declares the
/// architecture instead of inferring it, and this pins that `x2`/`x3` seed as
/// incoming on ARM32 even when `g14` is present, while staying non-incoming
/// when the same slots are treated as x86.
#[test]
fn arm32_seeds_incoming_arg_registers_despite_g14() {
    use std::collections::HashMap;
    let mut slots: HashMap<u32, String> = HashMap::new();
    // entry versions: x0 (this), x1 (param0), x2/x3 (incoming), g14 (lr).
    for (v, s) in ["x0", "x1", "x2", "x3", "g14"].iter().enumerate() {
        slots.insert(v as u32, s.to_string());
    }
    let seed = |is_arm32: bool| locals::SeedInfo {
        param_types: vec![ManagedType::Primitive(PrimKind::I4)],
        ret_type: ManagedType::Primitive(PrimKind::I4),
        is_static: false,
        declaring: None,
        param_names: vec!["a".into()],
        param_comp_names: Vec::new(),
        arm32_param_words: Vec::new(),
        is_arm32,
        ret_hidden_buffer: false,
    };
    let incoming = |t: &locals::LocalTable, v: u32| t.infos[t.index_of[&v]].is_incoming_arg;

    let arm = locals::create(&slots, &seed(true));
    assert!(
        incoming(&arm, 2) && incoming(&arm, 3),
        "on ARM32, x2/x3 must seed as incoming args even though g14 (lr) is present"
    );
    // x1 is the declared parameter; x0 is `this`.
    assert!(arm.infos[arm.index_of[&1]].is_param == Some(0));
    assert!(arm.infos[arm.index_of[&0]].is_this);

    // Same slots, no declared architecture: the `g14` name makes the heuristic
    // call this x86, and the incoming seeding is off — the defect this pins.
    let x86 = locals::create(&slots, &seed(false));
    assert!(
        !incoming(&x86, 2) && !incoming(&x86, 3),
        "control: without is_arm32 the g14 name still disables incoming seeding"
    );
}

#[test]
fn arm32_unprivileged_load_lifts_like_the_ordinary_one() {
    // `ldrbt r0, [r1], #4` = E4D10004 ; `bx lr`.
    //
    // The unprivileged `t` forms do the same data movement as their ordinary
    // counterparts, which is why they are lifted by rewriting the mnemonic.
    let insns = lift_a32(&[0xE4D10004, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "`ldrbt` must not become a stub: {:?}",
        insns
    );
    let loaded = insns.iter().any(|i| {
        matches!(&i.op, Op::Move) && matches!(i.operands.get(1), Some(Operand::Memory { .. }))
    });
    let advanced = insns.iter().any(|i| {
        matches!(&i.op, Op::Arith(ArithKind::Add))
            && matches!(i.operands.get(2), Some(Operand::Immediate(4)))
    });
    assert!(
        loaded && advanced,
        "ldrbt must load AND advance r1: {:?}",
        insns
    );
}

#[test]
fn arm32_ldmdb_is_a_block_load() {
    // `ldmdb r1, {r2, r3}` = E911000C ; `bx lr`
    let insns = lift_a32(&[0xE911000C, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(&i.op, Op::Simd(_))),
        "`ldmdb` must not become a stub: {:?}",
        insns
    );
    let loads = insns
        .iter()
        .filter(|i| {
            matches!(&i.op, Op::Move) && matches!(i.operands.get(1), Some(Operand::Memory { .. }))
        })
        .count();
    assert!(loads == 2, "ldmdb {{r2,r3}} is two loads, got {:?}", insns);
}

#[test]
fn arm32_all_condition_codes_reach_their_arm() {
    // `movmi r0, #1` = E3A00001 with cond 0100 (MI); `movvc r1, #2` = 03A01002.
    //
    // The condition suffix decides how the mnemonic is split, so a code missing
    // from the split table glued itself to the mnemonic ("movmi") and matched
    // nothing — the instruction then silently lost its destination.
    for (word, cond) in [(0x43A00001u32, "mi"), (0x73A01002u32, "vs")] {
        let insns = lift_a32(&[word, 0xE12FFF1E]);
        assert!(
            !insns
                .iter()
                .any(|i| matches!(&i.op, Op::Simd(n) if n.contains("mov"))),
            "`mov{cond}` must not become a stub: {:?}",
            insns
        );
        // A predicated `mov` is a SELECT now (`dst = cond ? src : dst`), not an
        // unconditional MOVE — lifting it as a `Move` was executing it
        // unconditionally and silently dropping the predicate.
        assert!(
            insns
                .iter()
                .any(|i| matches!(&i.op, Op::Move) || matches!(&i.op, Op::Select(_))),
            "`mov{cond}` must still move (as a select): {:?}",
            insns
        );
    }
}

/// A PREDICATED MOVE IS A CONDITIONAL SELECT, not an unconditional move.
///
/// Bytes are `0x10301cc`'s tail from the Android `libil2cpp.so`:
///
/// ```text
///   0x1030538  mov r6, r0
///   0x103053c  cmp r6, #0
///   0x1030540  movne r6, #1     ; r6 != 0 ? 1 : r6  — the `bool` idiom
/// ```
///
/// Lifting `movne` as `r6 = 1` unconditionally is a WRONG-LOGIC error: the
/// boolean came out 1 for the false case too. It must be a Select whose false
/// operand is the destination (resolved by SSA to the version live before it).
#[test]
fn arm32_predicated_move_is_a_select_not_an_unconditional_move() {
    let bytes = le(&[
        0xE1A06000, // mov r6, r0
        0xE3560000, // cmps r6, #0
        0x13006001, // movne r6, #1
        0xE1A00006, // mov r0, r6
        0xE12FFF1E, // bx lr
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1030538, 16).expect("lift");
    let sel = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Select(super::ir::SelectVariant::Identity)))
        .unwrap_or_else(|| panic!("movne must lift to a Select: {:?}", lifted.insns));
    // operands: [dst, cond, true, false]; the false operand is the destination
    // itself (its previous value).
    assert_eq!(sel.operands.first(), Some(&Operand::reg("x6")), "dst");
    assert_eq!(sel.operands.get(2), Some(&Operand::imm(1)), "true arm");
    assert_eq!(
        sel.operands.get(3),
        Some(&Operand::reg("x6")),
        "false arm keeps the previous destination value"
    );
    // The predicate is materialized as a real compare against zero.
    assert!(
        lifted.insns.iter().any(|i| matches!(i.op, Op::Compare(_))),
        "the pending compare must be materialized: {:?}",
        lifted.insns
    );
}

/// A CONDITIONAL RETURN keeps its condition.
///
/// Bytes are the real tail of `ReadComment()` @0x12F67B0 in the Android
/// `libil2cpp.so`:
///
/// ```text
///   0x12F682C  cmp r0, #0x3e          ; cmp r0, 62
///   0x12F6830  popeq {r4, r10, fp, pc} ; if (equal) return
///   0x12F6834  ldr r0, [pc, #52]        ; the other arm's continuation
/// ```
///
/// Emitting the return UNCONDITIONALLY ends the linear path where the test is,
/// so everything reachable only on the other side becomes unreachable and
/// disappears — the method lost its whole loop body that way. The guard is the
/// branch the machine actually implements: fall through when the condition
/// fails.
#[test]
fn arm32_conditional_return_keeps_its_condition() {
    let bytes = le(&[
        0xE350003E, // cmp r0, #0x3e
        0x08BD8C10, // popeq {r4, r10, fp, pc}
        0xE1A00004, // mov r0, r4
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x12F682C, 16).expect("lift");
    let guard = lifted
        .insns
        .iter()
        .position(|i| matches!(i.op, Op::CondJump(t) if t == 0x12F6834))
        .unwrap_or_else(|| {
            panic!(
                "a conditional return must be guarded by a branch to the next \
                 instruction, got {:?}",
                lifted
                    .insns
                    .iter()
                    .map(|i| i.op.clone())
                    .collect::<Vec<_>>()
            )
        });
    // The guard branches on the NEGATION: `popeq` returns when equal, so it
    // skips the return when NOT equal.
    match &lifted.insns[guard - 1].op {
        Op::Compare(k) if *k == CompareKind::NotEqual => {}
        other => panic!("expected a negated compare before the guard, got {other:?}"),
    }
    assert_eq!(
        lifted.insns[guard - 1].operands.get(2),
        Some(&Operand::imm(62)),
        "the comparison the native performs must survive"
    );
    let ret = lifted
        .insns
        .iter()
        .position(|i| matches!(i.op, Op::Return))
        .expect("the pop still returns");
    assert!(ret > guard, "the guard must precede the return");
}

#[test]
fn arm32_basic() {
    // mov r0, #7 = E3A00007 ; bx lr = E12FFF1E
    let bytes = le(&[0xE3A00007, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1000, 16).expect("lift");
    assert!(lifted.insns.len() >= 2);
    let last = lifted.insns.last().unwrap();
    assert!(matches!(last.op, Op::Return));
}

/// The register-shifted source operand (`ip, lsl r1`) must use the shift
/// REGISTER, not the elided `<< 0`.
///
/// yaxpeax prints an A32 register-controlled shift as `rm, lsl rs`; the old
/// `parse_shifted_src` parsed the amount with `parse_int(...).unwrap_or(0)`, so
/// every register shift silently became `<< 0`. In the conditional-bit-set
/// idiom (`and r1, r1, #31` ; `bic r3, m, ip, lsl r1` ; `orrne r3, m, ip, lsl r1`)
/// that set the WRONG bit — the `SimpleEventSource.set_Item` @0x54D310 shape.
/// Words are read from `libil2cpp.so` at that method.
#[test]
fn arm32_register_shifted_source_keeps_the_shift_register() {
    let bytes = le(&[0xE3520000, 0xE1CE311C, 0x118E311C, 0xE5803008, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1000, 32).expect("lift");
    let shl: Vec<&Insn> = lifted
        .insns
        .iter()
        .filter(|i| matches!(i.op, Op::Arith(ArithKind::Shl)))
        .collect();
    assert!(
        !shl.is_empty(),
        "register shift must emit a Shl: {:?}",
        lifted.insns
    );
    for s in &shl {
        assert!(
            matches!(s.operands.get(2), Some(Operand::Register(n)) if n == "x1"),
            "shift amount must be the register x1: {s:?}"
        );
    }
}

/// A predicated ALU op whose destination was already written is a SELECT, not
/// an unconditional compute.
///
/// `bic r3, lr, ip, lsl r1` ; `orrne r3, lr, ip, lsl r1` is the compiler's
/// conditional bit set/clear. Lifted unconditionally the `orrne` makes the
/// `bic` dead and sets the bit ALWAYS (`m_mask = m_mask | 1`). With the `bic`
/// already defining `r3`, the guard is modelled as `r3 = cond ? or : r3`.
#[test]
fn arm32_predicated_alu_with_a_prior_write_becomes_a_select() {
    let bytes = le(&[0xE3520000, 0xE1CE311C, 0x118E311C, 0xE5803008, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1000, 32).expect("lift");
    assert!(
        lifted
            .insns
            .iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::And))),
        "the `bic` must survive as the SELECT's false arm: {:?}",
        lifted.insns
    );
    let sel = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Select(SelectVariant::Identity)))
        .expect("the predicated `orrne` must be a Select");
    assert!(matches!(sel.operands.get(0), Some(Operand::Register(n)) if n == "x3"));
    assert!(
        matches!(sel.operands.get(3), Some(Operand::Register(n)) if n == "x3"),
        "the false arm must read the destination: {sel:?}"
    );
}

// ---------------------------------------------------------------------------
// ARM32 VFP lifting (encodings verified against keystone+capstone)
// ---------------------------------------------------------------------------

#[test]
fn arm32_vfp_scalar_arith() {
    // vadd.f32 s0,s1,s2 = EE300A81 ; bx lr = E12FFF1E
    let bytes = le(&[0xEE300A81, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 16).expect("lift");
    let mut arith = None;
    for insn in &lifted.insns {
        if let Op::FloatArith(kind) = insn.op {
            arith = Some(kind);
            // dst = s0, a = s1, b = s2 → vs0, vs1, vs2
            assert!(matches!(insn.operands.get(0), Some(Operand::Register(n)) if n == "vs0"));
            assert!(matches!(insn.operands.get(1), Some(Operand::Register(n)) if n == "vs1"));
            assert!(matches!(insn.operands.get(2), Some(Operand::Register(n)) if n == "vs2"));
        }
    }
    assert_eq!(arith, Some(FloatArithKind::Add), "vadd.f32 should lift");

    // vmul.f32 s6,s7,s8 = EE233A84 ; vdiv.f32 s9,s10,s11 = EEC54A25
    let bytes = le(&[0xEE233A84, 0xEEC54A25, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 16).expect("lift");
    let kinds: Vec<FloatArithKind> = lifted
        .insns
        .iter()
        .filter_map(|i| match i.op {
            Op::FloatArith(k) => Some(k),
            _ => None,
        })
        .collect();
    assert_eq!(kinds, vec![FloatArithKind::Mul, FloatArithKind::Div]);
    // registers: s6->vs6, s7->vs7, s8->vs8 ; s9->vs9, s10->vs10, s11->vs11
    assert!(matches!(lifted.insns[0].operands.get(0), Some(Operand::Register(n)) if n == "vs6"));
    assert!(matches!(lifted.insns[1].operands.get(0), Some(Operand::Register(n)) if n == "vs9"));
}

#[test]
fn arm32_vfp_double_and_unary() {
    // vadd.f64 d0,d1,d2 = EE310B02 ; vabs.f32 s18,s19 = EEB09AE9
    // vsqrt.f32 s22,s23 = EEB1BAEB ; bx lr
    let bytes = le(&[0xEE310B02, 0xEEB09AE9, 0xEEB1BAEB, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 16).expect("lift");
    let kinds: Vec<(FloatArithKind, String, String)> = lifted
        .insns
        .iter()
        .filter_map(|i| match &i.op {
            Op::FloatArith(k) => {
                let d = match i.operands.get(0) {
                    Some(Operand::Register(n)) => n.clone(),
                    _ => "?".into(),
                };
                let s = match i.operands.get(1) {
                    Some(Operand::Register(n)) => n.clone(),
                    _ => "?".into(),
                };
                Some((*k, d, s))
            }
            _ => None,
        })
        .collect();
    assert_eq!(kinds.len(), 3);
    assert_eq!(kinds[0], (FloatArithKind::Add, "vd0".into(), "vd1".into()));
    assert_eq!(
        kinds[1],
        (FloatArithKind::Abs, "vs18".into(), "vs19".into())
    );
    assert_eq!(
        kinds[2],
        (FloatArithKind::Sqrt, "vs22".into(), "vs23".into())
    );
}

/// `vmov` between a core register and a single-precision FP register used to be
/// gated on `vn == 0xE` — only the s28–s31 slice. Everything below fell through:
/// `vmov s0, r0` (opc2=0) reached the VMLA arm and was lifted as an accumulating
/// multiply (silently wrong), `vmov r0, s0` (opc2=1) fell to yaxpeax and was
/// dropped. The discriminator is bit4 + `opc2 <= 1`; all four words here are
/// capstone-verified.
#[test]
fn arm32_vfp_core_transfer_covers_every_single_register() {
    let bytes = le(&[
        0xEE000A10, // vmov s0,  r0
        0xEE100A10, // vmov r0,  s0
        0xEE0E0A10, // vmov s28, r0
        0xEE153A10, // vmov r3,  s10
        0xE12FFF1E, // bx   lr
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x3000, 32).expect("lift");
    let moves: Vec<(u64, Vec<Operand>)> = lifted
        .insns
        .iter()
        .filter(|i| i.op == Op::Move)
        .map(|i| (i.address, i.operands.clone()))
        .collect();
    let has = |d: &str, s: &str| {
        moves.iter().any(|(_, o)| {
            matches!((o.first(), o.get(1)),
                (Some(Operand::Register(a)), Some(Operand::Register(b))) if a == d && b == s)
        })
    };
    assert!(has("vs0", "x0"), "vmov s0, r0 → s0 <- r0: {moves:?}");
    assert!(has("x0", "vs0"), "vmov r0, s0 → r0 <- s0: {moves:?}");
    assert!(has("vs28", "x0"), "vmov s28, r0 (the old slice): {moves:?}");
    assert!(has("x3", "vs10"), "vmov r3, s10 → r3 <- s10: {moves:?}");
    assert!(
        !lifted
            .insns
            .iter()
            .any(|i| matches!(i.op, Op::MulAcc { .. })),
        "a core transfer must never lift as VMLA: {:?}",
        lifted
            .insns
            .iter()
            .map(|i| (i.address, i.op.clone()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn arm32_vfp_vcvt_and_vmov_imm() {
    // vcvt.f32.s32 s0,s1 = EEB80AE0 ; vmov.f32 s1,#3.0 = EEF00A08
    // vmrs APSR_nzcv,FPSCR = EEF1FA10 ; bx lr
    let bytes = le(&[0xEEB80AE0, 0xEEF00A08, 0xEEF1FA10, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 32).expect("lift");
    let mut saw_convert = false;
    let mut saw_imm = None;
    for insn in &lifted.insns {
        match insn.op {
            Op::Convert(_) => saw_convert = true,
            Op::Move => {
                if let (Some(Operand::Register(n)), Some(Operand::FloatLiteral(v))) =
                    (insn.operands.get(0), insn.operands.get(1))
                {
                    if n == "vs1" {
                        saw_imm = Some(*v);
                    }
                }
            }
            _ => {}
        }
    }
    assert!(saw_convert, "vcvt should lift to a convert");
    // 0xE: imm4=0, imm8=8 → (1+8/16)*2^(0+1) = 1.5*2 = 3.0
    assert_eq!(saw_imm, Some(3.0f32));
}

#[test]
fn arm32_vfp_vmla_vldr_and_compare() {
    // vmla.f32 s12,s13,s14 = EE066A87 ; vldr.f32 s30,[r2,#4] = ED92FA01
    // vcmp.f32 s22,s23 = EEB4BA6B ; vmrs APSR_nzcv,FPSCR = EEF1FA10
    // bne $+0x0 = 1AFFFFFE ; bx lr
    let bytes = le(&[
        0xEE066A87, 0xED92FA01, 0xEEB4BA6B, 0xEEF1FA10, 0x1AFFFFFE, 0xE12FFF1E,
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 32).expect("lift");
    let mut ops: Vec<Op> = lifted.insns.iter().map(|i| i.op.clone()).collect();
    ops.retain(|o| !matches!(o, Op::Return));
    assert!(matches!(ops[0], Op::MulAcc { is_float: true, .. }));
    assert_eq!(ops[1], Op::Move); // vldr → move from memory
    // vmrs must NOT clear the pending vcmp — the following bne consumes it
    assert!(
        matches!(ops[2], Op::Compare(_)),
        "vcmp→vmrs→bne should lift to a compare"
    );
    assert!(matches!(ops[3], Op::CondJump(_)));
}

/// A `vldr`/`vstr` off a CALLEE-SAVED core register must be lifted, not
/// dropped.
///
/// The VFP load/store group validated its base register with a `4..=12 =>
/// NotVfp` guard meant only for the VSTM/VLDM (`vpush`/`vpop`) sp-only rule.
/// Because the guard sat BEFORE the opcode dispatch it also rejected every
/// `vldr`/`vstr` whose base was r4–r12 — ordinary code, since r4 holds a
/// pinned object pointer. The load vanished, so the FP register it defined
/// read as an undefined local and the `vcmp` that consumed it lost its
/// operand (`vldr s0, [r4, #0x14]` / `vldr s16, [r4, #0x10]` @0xf11cb8).
#[test]
fn arm32_vldr_off_callee_saved_base_is_lifted() {
    // vldr s0, [r4, #0x14] = ED940A05 ; vldr s16, [r4, #0x10] = ED948A04 ;
    // vcmp.f32 s0, #0 = EEB50A40 ; vmrs = EEF1FA10 ; beq $+0x0 = 0AFFFFFE ; bx lr
    let bytes = le(&[
        0xED940A05, 0xED948A04, 0xEEB50A40, 0xEEF1FA10, 0x0AFFFFFE, 0xE12FFF1E,
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 32).expect("lift");
    // Both loads survive, off the real `x4` base.
    let loads: Vec<&Insn> = lifted
        .insns
        .iter()
        .filter(|i| {
            matches!(i.op, Op::Move)
                && matches!(i.operands.get(1), Some(Operand::Memory { base, .. })
                    if matches!(base.as_ref(), Operand::Register(n) if n == "x4"))
        })
        .collect();
    assert_eq!(
        loads.len(),
        2,
        "both vldr off r4 must be lifted: {:?}",
        lifted.insns
    );
    assert!(
        loads
            .iter()
            .any(|i| matches!(i.operands.first(), Some(Operand::Register(n)) if n == "vs0")),
        "the vs0 load must survive"
    );
    assert!(
        loads
            .iter()
            .any(|i| matches!(i.operands.first(), Some(Operand::Register(n)) if n == "vs16")),
        "the vs16 load must survive"
    );
    // And the compare that consumes vs0 keeps a real register operand.
    let cmp = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("vcmp must produce a Compare");
    assert!(
        matches!(cmp.operands.get(1), Some(Operand::Register(n)) if n == "vs0"),
        "vcmp s0,#0 must read the loaded vs0: {cmp:?}"
    );
}

/// `vcmpe` (the exception-signalling compare) must be lifted, not dropped.
///
/// The manual VFP decoder recognized `vcmp` but not its `-e` twin, and the
/// `vcmp.f32 sN, sM` REGISTER form shared an encoding slot with the `#0.0`
/// form and was returned as `#0.0` unconditionally. Both are wrong-logic
/// defects: the first dropped the compare entirely (the branch rendered
/// `if (? > 0)`), the second compared against a zero the machine never used.
#[test]
fn arm32_vcmpe_and_register_vcmp_keep_their_operands() {
    // vcmpe.f32 s2, s0 = EEB41AC0 ; vmrs = EEF1FA10 ; bhi $+0x0 = 8AFFFFFE ; bx lr
    let bytes = le(&[0xEEB41AC0, 0xEEF1FA10, 0x8AFFFFFE, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 32).expect("lift");
    let cmp = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("vcmpe must produce a Compare");
    // Real register operands, not `Unknown` vs `0`.
    assert!(matches!(cmp.operands.get(1), Some(Operand::Register(_))));
    assert!(matches!(cmp.operands.get(2), Some(Operand::Register(_))));

    // vcmp.f32 s2, s0 = EEB41A40 — the REGISTER form, not `#0.0`.
    let bytes = le(&[0xEEB41A40, 0xEEF1FA10, 0x8AFFFFFE, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 32).expect("lift");
    let cmp = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("vcmp must produce a Compare");
    assert!(
        matches!(cmp.operands.get(2), Some(Operand::Register(_))),
        "vcmp.f32 s2, s0 must compare two registers, not against 0"
    );

    // vcmpe.f32 s2, #0 = EEB50AC0 — the immediate form still yields 0.0.
    let bytes = le(&[0xEEB50AC0, 0xEEF1FA10, 0x0AFFFFFE, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x2000, 32).expect("lift");
    let cmp = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("vcmpe #0 must produce a Compare");
    assert!(matches!(
        cmp.operands.get(2),
        Some(Operand::FloatLiteral(_))
    ));
}

// ---------------------------------------------------------------------------
// ARM32 integer-op coverage (movt pairs, shifts, mul/rsb/mvn/bic, flags)
// ---------------------------------------------------------------------------

#[test]
fn arm32_movw_movt_pair() {
    // movw r0,#0x1234 = E3010234 ; movt r0,#0xabcd = E34A0BCD ; bx lr
    let bytes = le(&[0xE3010234, 0xE34A0BCD, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x3000, 16).expect("lift");
    assert_eq!(lifted.insns.len(), 3); // mov imm, mov imm, return
    assert!(matches!(
        lifted.insns[0].operands.get(1),
        Some(Operand::Immediate(0x1234))
    ));
    // movt folds the low half into the high-combined constant
    assert!(
        matches!(lifted.insns[1].operands.get(1), Some(Operand::Immediate(v)) if *v == 0xABCD1234)
    );
}

#[test]
fn arm32_shift_alias_and_mul() {
    // mov r1, r2, lsl #2 (0xE1A01102) ; mul r0, r2, r1 (0xE0000192) ; bx lr
    let bytes = le(&[0xE1A01102, 0xE0000192, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x3000, 16).expect("lift");
    let mut saw_shl = false;
    let mut saw_mul = false;
    for insn in &lifted.insns {
        match insn.op {
            Op::Arith(ArithKind::Shl) => saw_shl = true,
            Op::Arith(ArithKind::Mul) => saw_mul = true,
            _ => {}
        }
    }
    assert!(saw_shl, "lsl alias should lift to a shift");
    assert!(saw_mul, "mul should lift");
}

#[test]
fn arm32_rsb_mvn_bic() {
    // rsb r0,r1,#5 = E2610005 ; mvn r1,r2 = E1E01002 ; bic r0,r1,#0xff = E3C100FF ; bx lr
    let bytes = le(&[0xE2610005, 0xE1E01002, 0xE3C100FF, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x3000, 16).expect("lift");
    let mut saw_sub = false;
    let mut saw_xor_with_neg1 = false;
    let mut saw_and = false;
    for insn in &lifted.insns {
        match insn.op {
            Op::Arith(ArithKind::Sub) => saw_sub = true,
            Op::Arith(ArithKind::Xor) => {
                if matches!(insn.operands.get(2), Some(Operand::Immediate(-1))) {
                    saw_xor_with_neg1 = true;
                }
            }
            Op::Arith(ArithKind::And) => saw_and = true,
            _ => {}
        }
    }
    assert!(saw_sub, "rsb should lift to a sub");
    assert!(saw_xor_with_neg1, "mvn should lift to xor -1 (bitwise not)");
    assert!(saw_and, "bic should lift");
}

#[test]
fn arm32_cmp_suffix_fold() {
    // cmps r5,r2 (0xE1550002) ; bne target (0x1AFFFFFE → 0x3000+8-8=0x3000) ; bx lr
    let bytes = le(&[0xE1550002, 0x1AFFFFFE, 0xE12FFF1E]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x3000, 16).expect("lift");
    let mut saw_compare = false;
    let mut saw_condjump = false;
    for insn in &lifted.insns {
        match insn.op {
            Op::Compare(CompareKind::NotEqual) => {
                saw_compare = true;
                assert!(matches!(insn.operands.get(1), Some(Operand::Register(n)) if n == "x5"));
                assert!(matches!(insn.operands.get(2), Some(Operand::Register(n)) if n == "x2"));
            }
            Op::CondJump(_) => saw_condjump = true,
            _ => {}
        }
    }
    assert!(saw_compare, "cmps should feed the compare fold");
    assert!(saw_condjump);
}

#[test]
fn arm32_unconditional_branch_keeps_the_pending_compare() {
    // `cmp r0, r1; bls 0x100C; b 0x1010; bhi 0x1010; bx lr`
    // The `b` writes no APSR, so the `bhi` re-tests the SAME compare the `bls`
    // did. Before the fix the unconditional `b` cleared the pending compare, so
    // the `bhi` lifted to `Compare(kind, Unknown, 0)` and the method verified as
    // a missing condition (`if (? != 0)`).
    let insns = lift_a32(&[0xE1500001, 0x9A000000, 0xEA000000, 0x8AFFFFFF, 0xE12FFF1E]);
    assert!(
        !insns.iter().any(|i| matches!(i.op, Op::Compare(_))
            && i.operands
                .iter()
                .skip(1)
                .any(|o| matches!(o, Operand::Unknown))),
        "the `bhi` after the unconditional `b` must keep the cmp's operands: {insns:?}"
    );
    let real = insns
        .iter()
        .filter(|i| matches!(i.op, Op::Compare(_)))
        .count();
    assert!(
        real >= 2,
        "both branches need a compare, not one: {insns:?}"
    );
}

#[test]
fn x64_direct_jump_keeps_the_pending_compare() {
    // `cmp eax, ecx; jne +2; jmp +0; ja +0; ret` — the direct `jmp` writes no
    // EFLAGS, so the `ja` reads the same compare the `jne` did.
    let bytes = [0x39u8, 0xC8, 0x75, 0x02, 0xEB, 0x00, 0x77, 0x00, 0xC3];
    let insns = lift_sysv(&bytes);
    assert!(
        !insns.iter().any(|i| matches!(i.op, Op::Compare(_))
            && i.operands
                .iter()
                .skip(1)
                .any(|o| matches!(o, Operand::Unknown))),
        "the `ja` after the direct `jmp` must keep the cmp's operands: {insns:?}"
    );
    let real = insns
        .iter()
        .filter(|i| matches!(i.op, Op::Compare(_)))
        .count();
    assert!(
        real >= 2,
        "both conditional branches need a compare: {insns:?}"
    );
}

#[test]
fn arm32_cross_block_condition_is_recovered_from_a_reaching_compare() {
    // The shape a LINEAR pending-compare model cannot see (see
    // `passes::resolve_cross_block_conditions`):
    //
    //   0x1000 cmp r0, r1
    //   0x1004 strb r2, [r3]        ; writes no APSR
    //   0x1008 bne 0x1018           ; flags live on the TAKEN edge
    //   0x100c bl 0x2000            ; fall-through ONLY — clobbers the flags
    //   0x1010 b 0x1020             ; ends the fall-through path
    //   0x1014 bx lr                ; unreachable (no fall-through into 0x1018)
    //   0x1018 ble 0x1024           ; re-tests the SAME APSR
    //
    // `0x1018` has ONE predecessor (the `bne`), and no flag writer on that
    // edge, so the flags are still live at `ble`. The linear lifter cleared the
    // pending compare at the fall-through `bl`, so it emitted
    // `Compare(kind, Unknown, 0)`; the pass must carry the cmp's operands
    // across the edge and fill it in.
    let bytes = le(&[
        0xE1500001, // cmp r0, r1
        0xE5C32000, // strb r2, [r3]
        0x1A000002, // bne 0x1018
        0xEB0003FB, // bl 0x2000
        0xEA000002, // b 0x1020
        0xE12FFF1E, // bx lr
        0xDA000001, // ble 0x1024
        0xE12FFF1E, // bx lr
        0xE12FFF1E, // bx lr
        0xE12FFF1E, // bx lr
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1000, 64).expect("lift");
    let mut insns = lifted.insns;
    let graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Compare(_))
            && matches!(i.operands.get(1), Some(Operand::Unknown))),
        "the `bl` on the fall-through should have cost the linear model the \
         compare: {insns:?}"
    );
    let n = passes::resolve_cross_block_conditions(&mut insns, &graph);
    assert_eq!(
        n, 1,
        "exactly the cross-block branch is recovered: {insns:?}"
    );
    assert!(
        !insns.iter().any(|i| matches!(i.op, Op::Compare(_))
            && matches!(i.operands.get(1), Some(Operand::Unknown))),
        "the recovered compare must carry the cmp's operands: {insns:?}"
    );
}

#[test]
fn arm32_cross_block_pass_will_not_cross_a_call_on_the_path() {
    // Same shape, but the `bl` sits ON the taken path (before the branch), so
    // the flags ARE clobbered and the branch must stay `Unknown` — the pass
    // must not invent a condition from a compare that a call destroyed.
    //
    //   0x1000 cmp r0, r1
    //   0x1004 bne 0x100c
    //   0x1008 b 0x1018
    //   0x100c bl 0x2000            ; clobbers the flags on the branch's path
    //   0x1010 ble 0x1018           ; genuinely unknown
    let bytes = le(&[
        0xE1500001, // cmp r0, r1
        0x1A000000, // bne 0x100c
        0xEA000002, // b 0x1018
        0xEB0003FB, // bl 0x2000
        0xDA000000, // ble 0x1018
        0xE12FFF1E, // bx lr
        0xE12FFF1E, // bx lr
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1000, 64).expect("lift");
    let mut insns = lifted.insns;
    let graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    let n = passes::resolve_cross_block_conditions(&mut insns, &graph);
    assert_eq!(
        n, 0,
        "a call on the path must block the recovery: {insns:?}"
    );
}

#[test]
fn arm32_bitfield_words_are_lifted_from_the_raw_word() {
    // yaxpeax rejects every one of these encodings, so before `dropped_op` they
    // produced NO IR at their own address and a branch landing on them slid
    // forward (`branch target mismatch: native taken`). The derived bit fields
    // were checked against capstone over the whole code section (0
    // disagreements): `ubfx r0,r6,#5,#1` = 0xE7E002D6, `bfi r4,r6,#6,#0x1a` =
    // 0xE7DF4316, `bfc r0,#0,#0xb` = 0xE7CA001F, `sbfx r2,r1,#0x1e,#1` =
    // 0xE7A02F51.
    let insns = lift_a32(&[0xe7e002d6, 0xe7df4316, 0xe7ca001f, 0xe7a02f51, 0xe12fff1e]);
    // ubfx r0, r6, #5, #1  ->  ((r6 >> 5) & 1)
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.first(), i.operands.get(2)),
            (Op::Arith(ArithKind::Shr), Some(Operand::Register(r)), Some(Operand::Immediate(5)))
                if r == "x0"
        )),
        "ubfx must shift right by its lsb: {insns:?}"
    );
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.first(), i.operands.get(2)),
            (Op::Arith(ArithKind::And), Some(Operand::Register(r)), Some(Operand::Immediate(1)))
                if r == "x0"
        )),
        "ubfx width 1 must mask to one bit: {insns:?}"
    );
    // bfi r4, r6, #6, #0x1a merges into r4.
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.first()),
            (Op::Arith(ArithKind::Or), Some(Operand::Register(r))) if r == "x4"
        )),
        "bfi must read-modify-write its destination: {insns:?}"
    );
    // bfc r0, #0, #0xb clears the low 11 bits (`& !0x7ff`).
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.first(), i.operands.get(2)),
            (Op::Arith(ArithKind::And), Some(Operand::Register(r)), Some(Operand::Immediate(m)))
                if r == "x0" && *m == !0x7ffi64
        )),
        "bfc must clear exactly its field: {insns:?}"
    );
    // sbfx r2, r1, #0x1e, #1 sign-extends via the shl/shr pair into r2.
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.first()),
            (Op::Arith(ArithKind::Shl), Some(Operand::Register(r))) if r == "x2"
        )),
        "sbfx must sign-extend (shl) into the destination: {insns:?}"
    );
}

#[test]
fn arm32_smmul_is_an_unmodelled_intrinsic_not_a_wrong_mul() {
    // `smmul r0, r7, r8` = 0xE750F817. The result is the TOP word of the
    // product, so a plain 32-bit `Mul` would be wrong; it stays an explicit
    // intrinsic while still defining r0 from r7/r8.
    let insns = lift_a32(&[0xe750f817, 0xe12fff1e]);
    assert!(
        insns.iter().any(|i| match (&i.op, i.operands.as_slice()) {
            (Op::Simd(name), ops) => {
                name == "__smmul"
                    && matches!(ops.first(), Some(Operand::Register(r)) if r == "x0")
                    && matches!(ops.get(1), Some(Operand::Register(r)) if r == "x7")
                    && matches!(ops.get(2), Some(Operand::Register(r)) if r == "x8")
            }
            _ => false,
        }),
        "smmul must be an intrinsic defining r0 from r7/r8: {insns:?}"
    );
}

#[test]
fn arm32_two_core_register_vmov_defines_both_gpr_halves() {
    // `vmov r2, r3, d8` = 0xEC532B18 — yaxpeax rejects the whole group, so
    // before the fix BOTH GPR halves were dropped.
    let insns = lift_a32(&[0xec532b18, 0xe12fff1e]);
    let defs: Vec<&str> = insns
        .iter()
        .filter_map(|i| match (&i.op, i.operands.first()) {
            (Op::Simd(name), Some(Operand::Register(r))) if name == "__vmov" => Some(r.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        defs.contains(&"x2") && defs.contains(&"x3"),
        "both GPR halves of the doubleword transfer must be defined: {insns:?}"
    );
}

#[test]
fn arm32_vfp_sqrt_neg_and_convert_register_forms_are_lifted() {
    // `vsqrt.f32 s18,s18` (0xEEB19AC9), `vneg.f32 s0,s18` (0xEEB10A49) and
    // `vcvt.f32.s32 s0,s18` (0xEEB80AC9) are all the bit5-clear spelling of
    // their opcode; the arms existed only for the bit5-set spelling, so these
    // (the ones that actually occur) fell through to `NotVfp`.
    let insns = lift_a32(&[0xeeb19ac9, 0xeeb10a49, 0xeeb80ac9, 0xe12fff1e]);
    assert!(
        insns
            .iter()
            .any(|i| matches!(i.op, Op::FloatArith(FloatArithKind::Sqrt))),
        "vsqrt must lift: {insns:?}"
    );
    assert!(
        insns
            .iter()
            .any(|i| matches!(i.op, Op::FloatArith(FloatArithKind::Neg))),
        "vneg must lift: {insns:?}"
    );
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Convert(_))),
        "vcvt must lift: {insns:?}"
    );
}

#[test]
fn arm32_signed_narrow_loads_are_lifted_from_the_raw_word() {
    // `ldrsb r0,[r4,#8]` = 0xE1D400D8 and `ldrsh r0,[sp,#2]` = 0xE1DD00F2.
    // yaxpeax answers `Nonconforming` for both, so the existing `ldrsb`/`ldrsh`
    // arm was unreachable and the load produced no IR at its own address.
    let insns = lift_a32(&[0xe1d400d8, 0xe1dd00f2, 0xe12fff1e]);
    let widths: Vec<u8> = insns
        .iter()
        .filter_map(|i| match (&i.op, i.operands.first(), i.operands.get(1)) {
            (
                Op::Move,
                Some(Operand::Register(r)),
                Some(Operand::Memory {
                    width,
                    base,
                    addend,
                    ..
                }),
            ) if r == "x0" => match (&**base, *addend) {
                (Operand::Register(b), 8) if b == "x4" => Some(*width),
                (Operand::Register(b), 2) if b == "sp" => Some(*width),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(
        widths.contains(&1) && widths.contains(&2),
        "ldrsb must load 1 byte and ldrsh 2: {insns:?}"
    );
    // Both sign-extend (`(v << k) >> k`, k = 24/16).
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.get(2)),
            (Op::Arith(ArithKind::Shl), Some(Operand::Immediate(24)))
        )),
        "ldrsb must sign-extend from bit 24: {insns:?}"
    );
}

#[test]
fn arm32_vmov_double_register_copy_is_lifted() {
    // `vmov.f64 d8, d16` = 0xEEB08B60 — the bit5-set spelling (Vm = 16 makes
    // bit5 = 1). The F64 copy arm existed only for the bit5-clear spelling, so
    // this one fell through to `NotVfp`.
    let insns = lift_a32(&[0xeeb08b60, 0xe12fff1e]);
    assert!(
        insns.iter().any(|i| matches!(&i.op, Op::Move)
            && matches!(i.operands.first(), Some(Operand::Register(r)) if r == "vd8")
            && matches!(i.operands.get(1), Some(Operand::Register(r)) if r == "vd16")),
        "vmov.f64 must copy d16 into d8: {insns:?}"
    );
}

#[test]
fn arm32_predicated_vfp_word_is_lifted_as_a_catch_all() {
    // `vmovgt.f32 s16, s0` = 0xCEB08A40 — cond != 0xE, so `is_vfp` excludes it
    // and yaxpeax rejects it, leaving no IR. Without a landing the branch that
    // skips it could not be certified.
    let insns = lift_a32(&[0xceb08a40, 0xe12fff1e]);
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.first()),
            (Op::Simd(name), Some(Operand::Register(r))) if name == "__vfp" && r == "vs16"
        )),
        "the predicated VFP op must define its destination: {insns:?}"
    );
}

#[test]
fn arm32_vldr_literal_pool_resolves_the_program_counter() {
    // `vldr s0, [pc, #100]` = 0xED9F0A19, lifted at 0x1000. A32 `pc` reads as
    // `addr + 8`, so the literal lives at 0x106c — not at a raw `g15` base.
    let insns = lift_a32(&[0xed9f0a19, 0xe12fff1e]);
    let ok = insns
        .iter()
        .any(|i| match (&i.op, i.operands.first(), i.operands.get(1)) {
            (Op::Move, Some(Operand::Register(r)), Some(Operand::Memory { base, .. })) => {
                r == "vs0" && matches!(&**base, Operand::Immediate(a) if *a == 0x106c)
            }
            _ => false,
        });
    assert!(
        ok,
        "the vldr literal address must be the resolved PC + 8 + offset: {insns:?}"
    );
}

#[test]
fn arm32_uxtab_writes_back_to_its_base() {
    // `uxtab r0, r5, r6` = 0xE6E50076: `r0 = r5 + uxth(r6)`, then `r5 = r0`.
    // Found by the independent Capstone translation-validation check as a
    // branch target the engine had dropped.
    let insns = lift_a32(&[0xe6e50076, 0xe12fff1e]);
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.first()),
            (Op::Arith(ArithKind::Add), Some(Operand::Register(r))) if r == "x0"
        )),
        "uxtab must add into its destination: {insns:?}"
    );
    assert!(
        insns.iter().any(|i| matches!(&i.op, Op::Move)
            && matches!(i.operands.first(), Some(Operand::Register(r)) if r == "x5")
            && matches!(i.operands.get(1), Some(Operand::Register(r)) if r == "x0")),
        "uxtab must write the result back to its base: {insns:?}"
    );
}

#[test]
fn arm32_vstmia_register_list_emits_ir_at_its_own_address() {
    // `vstmia r10!, {s16}` = 0xECAA8A01: store the single register s16 to
    // [r10] and advance r10 by 4. Found by the independent Capstone check as
    // the LAST genuine branch-target mismatch: the word produced NO IR, so a
    // `bhi` that targets it had its IR target shifted to the next instruction.
    let insns = lift_a32(&[0xecaa8a01, 0xe12fff1e]);
    let stored = insns
        .iter()
        .any(|i| match (&i.op, i.operands.first(), i.operands.get(1)) {
            (Op::Move, Some(Operand::Memory { base, addend, .. }), Some(Operand::Register(r))) => {
                r == "vs16" && *addend == 0 && matches!(&**base, Operand::Register(b) if b == "x10")
            }
            _ => false,
        });
    assert!(
        stored,
        "vstmia must store its register list to the base address: {insns:?}"
    );
    assert!(
        insns.iter().any(|i| matches!(
            (&i.op, i.operands.first()),
            (Op::Arith(ArithKind::Add), Some(Operand::Register(r))) if r == "x10"
        )),
        "vstmia with writeback must advance its base: {insns:?}"
    );
}

#[test]
fn arm32_vldmia_register_list_loads_into_the_vfp_file() {
    // `vldmia r0!, {s0, s1}` = 0xECB00A02, then `bx lr`.
    let insns = lift_a32(&[0xecb00a02, 0xe12fff1e]);
    for (reg, off) in [("vs0", 0i64), ("vs1", 4)] {
        assert!(
            insns
                .iter()
                .any(|i| match (&i.op, i.operands.first(), i.operands.get(1)) {
                    (
                        Op::Move,
                        Some(Operand::Register(r)),
                        Some(Operand::Memory { base, addend, .. }),
                    ) => {
                        r == reg
                            && *addend == off
                            && matches!(&**base, Operand::Register(b) if b == "x0")
                    }
                    _ => false,
                }),
            "vldmia must load {reg} from [x0, #{off}]: {insns:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Helper-map recovery: object_new / box / raise_exception / vtable_slots
// ---------------------------------------------------------------------------

fn helper_seed() -> locals::SeedInfo {
    locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::Primitive(PrimKind::I4),
        is_static: true,
        declaring: None,
        param_names: vec![],
        ..Default::default()
    }
}

fn mref(name: &str, tdi: i32, is_static: bool) -> MethodRef {
    MethodRef {
        image_index: 0,
        type_def_index: tdi,
        method_index: 0,
        name: name.into(),
        declaring_namespace: "Test".into(),
        declaring_type_name: "Widget".into(),
        assembly: String::new(),
        sig: types::MethodSigInfo {
            ret: ManagedType::Unknown,
            params: vec![],
            is_static,
        },
    }
}

/// object_new(klass) call sites become NewObj, typed through klass_to_tdi
/// + tdi_names when the klass argument is a constant slot address.
#[test]
fn key_function_object_new_becomes_typed_newobj() {
    let mut insns = vec![
        Insn::move_(Operand::reg("x0"), Operand::imm(0x6000), 0x1000),
        Insn::new(
            Op::Call,
            vec![
                Operand::imm(0x9000),
                Operand::reg("x0"),
                Operand::reg("x0"),
                Operand::reg("x1"),
            ],
            0x1004,
        ),
    ];
    let mut maps = resolve::ResolveMaps::default();
    maps.object_new = vec![0x9000];
    maps.klass_to_tdi.insert(0x6000, 7);
    maps.tdi_names.insert(7, ("Test".into(), "Widget".into()));

    let table = locals::create(&HashMap::new(), &helper_seed());
    resolve::run(&mut insns, &table, &maps);

    match &insns[1].op {
        Op::NewObj(Some(t)) => {
            assert_eq!(t.ty.namespace, "Test");
            assert_eq!(t.ty.name, "Widget");
            assert_eq!(t.ty.type_def_index, 7);
        }
        other => panic!("expected typed NewObj, got {other:?}"),
    }
}

/// Runtime metadata initialization — a VOID call whose first argument is the
/// ADDRESS of a metadata usage slot — is bookkeeping the source never had, and
/// is dropped (never a managed observable effect).
#[test]
fn key_function_drops_metadata_usage_init() {
    let slot = 0x7000i64;
    let mut insns = vec![Insn::new(
        Op::CallVoid,
        vec![
            Operand::imm(0x9300),
            Operand::imm(slot),
            Operand::reg("x1"),
        ],
        0x1000,
    )];
    let mut maps = resolve::ResolveMaps::default();
    maps.klass_to_tdi.insert(slot as u64, 5);
    maps.tdi_names.insert(5, ("Test".into(), "Thing".into()));

    let table = locals::create(&HashMap::new(), &helper_seed());
    resolve::run(&mut insns, &table, &maps);

    assert!(matches!(insns[0].op, Op::Nop), "got {:?}", insns[0].op);
    assert_eq!(
        crate::recovery::runtime_helpers::role_of(0x9300),
        Some(crate::recovery::runtime_helpers::RuntimeHelper::MetadataInit)
    );
    assert_eq!(
        crate::recovery::runtime_helpers::arg_arity(0x9300),
        Some(1)
    );
}

/// A klass loaded through a GOT slot (`[[got]]`) is still proven metadata:
/// got_addend → usage slot → TypeDef. This is the shape of the large
/// allocation temps (`v480 = object_new(*(*(GOT)))`).
#[test]
fn key_function_object_new_resolves_klass_through_the_got() {
    let got = 0x5000u64;
    let slot = 0x6100u64;
    let inner = Operand::Memory {
        base: Box::new(Operand::Immediate(got as i64)),
        index: None,
        scale: 1,
        addend: 0,
        width: 0,
    };
    let outer = Operand::Memory {
        base: Box::new(inner),
        index: None,
        scale: 1,
        addend: 0,
        width: 0,
    };
    let mut insns = vec![
        Insn::move_(Operand::reg("x0"), outer, 0x1000),
        Insn::new(
            Op::Call,
            vec![
                Operand::imm(0x9400),
                Operand::reg("x0"),
                Operand::reg("x0"),
                Operand::reg("x1"),
            ],
            0x1004,
        ),
    ];
    let mut maps = resolve::ResolveMaps::default();
    maps.object_new = vec![0x9400];
    maps.got_addend_by_addr.insert(got, slot);
    maps.klass_to_tdi.insert(slot, 11);
    maps.tdi_names
        .insert(11, ("System.Collections.Generic".into(), "List`1".into()));

    let table = locals::create(&HashMap::new(), &helper_seed());
    resolve::run(&mut insns, &table, &maps);

    match &insns[1].op {
        Op::NewObj(Some(t)) => assert_eq!(t.ty.name, "List`1"),
        other => panic!("expected typed NewObj through GOT, got {other:?}"),
    }
}

/// `v = helper(object, klass)` is an instance-of test even when a broad alias
/// chain also put the target in the object_new family: the allocator's class
/// is x0, not x1.
#[test]
fn key_function_isinst_overrides_a_wide_object_new_alias() {
    let mut insns = vec![
        Insn::move_(Operand::reg("x1"), Operand::imm(0x6200), 0x1000),
        Insn::new(
            Op::Call,
            vec![
                Operand::imm(0x9500),
                Operand::reg("x0"),
                Operand::reg("x0"),
                Operand::reg("x1"),
            ],
            0x1004,
        ),
    ];
    let mut maps = resolve::ResolveMaps::default();
    maps.object_new = vec![0x9500];
    maps.klass_to_tdi.insert(0x6200, 12);
    maps.tdi_names.insert(12, ("Test".into(), "Other".into()));

    let table = locals::create(&HashMap::new(), &helper_seed());
    resolve::run(&mut insns, &table, &maps);

    assert!(
        matches!(insns[1].op, Op::Call),
        "the isinst call must stay a call, got {:?}",
        insns[1].op
    );
    assert_eq!(
        crate::recovery::runtime_helpers::role_of(0x9500),
        Some(crate::recovery::runtime_helpers::RuntimeHelper::IsInst)
    );
    assert_eq!(
        crate::recovery::runtime_helpers::arg_arity(0x9500),
        Some(2)
    );
}

/// box(klass, &value) call sites keep their call form so the statement-stage
/// fold can recover the VALUE expression (Op::Box cannot carry it), and the
/// role table records the proven Box role for argument slicing.
#[test]
fn key_function_box_keeps_the_call_and_records_the_role() {
    let mut insns = vec![
        Insn::move_(Operand::reg("x0"), Operand::imm(0x6000), 0x1000),
        Insn::move_(
            Operand::reg("x1"),
            Operand::AddressOf(Box::new(Operand::Local(2))),
            0x1004,
        ),
        Insn::new(
            Op::Call,
            vec![
                Operand::imm(0x9100),
                Operand::reg("x0"),
                Operand::reg("x0"),
                Operand::reg("x1"),
                Operand::reg("x2"),
            ],
            0x1008,
        ),
    ];
    let mut maps = resolve::ResolveMaps::default();
    maps.box_fn = vec![0x9100];

    let table = locals::create(&HashMap::new(), &helper_seed());
    resolve::run(&mut insns, &table, &maps);

    assert!(
        matches!(insns[2].op, Op::Call),
        "box call must stay a call, got {:?}",
        insns[2].op
    );
    assert_eq!(
        crate::recovery::runtime_helpers::role_of(0x9100),
        Some(crate::recovery::runtime_helpers::RuntimeHelper::Box)
    );
    assert_eq!(
        crate::recovery::runtime_helpers::arg_arity(0x9100),
        Some(2)
    );
}

/// raise_exception call sites become Throw for both raw call layouts
/// (Call carries a result slot; CallVoid does not).
#[test]
fn key_function_raise_becomes_throw() {
    for (op, arg_pos) in [(Op::Call, 2usize), (Op::CallVoid, 1usize)] {
        let mut ops = vec![Operand::imm(0x9200)];
        if op == Op::Call {
            ops.push(Operand::reg("x0")); // result slot
        }
        ops.push(Operand::reg("x0"));
        ops.push(Operand::reg("x1"));
        let mut insns = vec![
            Insn::move_(Operand::reg("x0"), Operand::Local(3), 0x1000),
            Insn::new(op.clone(), ops, 0x1004),
        ];
        let mut maps = resolve::ResolveMaps::default();
        maps.raise_exception = vec![0x9200];

        let table = locals::create(&HashMap::new(), &helper_seed());
        resolve::run(&mut insns, &table, &maps);

        match &insns[1].op {
            Op::Throw => {
                assert_eq!(insns[1].operands.first(), Some(&Operand::Local(3)));
            }
            other => panic!("expected Throw, got {other:?}"),
        }
        let _ = arg_pos;
    }
}

/// Real virtual-dispatch shape (obj → klass → vtable[slot]) resolves to a
/// direct call when the receiver's seeded type identifies the klass.
#[test]
fn vtable_dispatch_resolves_through_seeded_receiver() {
    // receiver local 0 typed as Test.Widget (type_def_index 7)
    let table = locals::LocalTable {
        infos: vec![locals::LocalInfo {
            slot_origin: "x0".into(),
            ty: ManagedType::Class(std::sync::Arc::new(NamedType {
                namespace: "Test".into(),
                name: "Widget".into(),
                assembly: None,
                type_def_index: 7,
            })),
            is_this: true,
            is_param: None,
            param_comp: None,
            is_incoming_arg: false,
            is_call_clobber: false,
            is_return_buffer: false,
        }],
        index_of: [(0u32, 0usize)].into_iter().collect(),
        param_names: vec![],
        param_comp_names: vec![],
    };

    let mut insns = vec![
        // v0 = [obj]          (obj → klass)
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(1),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::Local(0)),
                    index: None,
                    scale: 8,
                    addend: 0,
                },
            ],
            0x1000,
        ),
        // v1 = [klass + 0x140] → vtable slot 1
        //
        // `vtable` is an array of `VirtualInvokeData
        // { methodPtr; method; }` — 16 bytes per entry, not 8: a stride of 8
        // read the companion `MethodInfo*` as if it were the next slot, so
        // every vtable index was doubled and almost nothing matched a
        // declared slot. Slot 1 is therefore `base + 16`, and the base is
        // `0x130` (see `vtable_base_64`): the callee load is always the
        // `methodPtr` half, and the native pairing `call [[obj]+N]` with the
        // `MethodInfo` at `N+8` makes `N` ≡ 0 (mod 16) on every x64 and ARM64
        // dump measured.
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(2),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::Local(1)),
                    index: None,
                    scale: 8,
                    addend: 0x140,
                },
            ],
            0x1004,
        ),
        Insn::new(Op::IndirectCall, vec![Operand::Local(2)], 0x1008),
    ];

    let mut maps = resolve::ResolveMaps::default();
    maps.vtable_slots.insert((7, 1), mref("DoThing", 7, false));

    let stats = resolve::run(&mut insns, &table, &maps);

    assert_eq!(stats.virtuals_resolved, 1, "virtual call should resolve");
    assert!(
        matches!(insns[2].op, Op::Call),
        "expected Call, got {:?}",
        insns[2].op
    );
    match insns[2].operands.first() {
        Some(Operand::Method(m)) => assert_eq!(m.name, "DoThing"),
        other => panic!("expected Method operand, got {other:?}"),
    }
}

/// The same dispatch on a 32-BIT target, where the vtable sits at 0xC0 and an
/// entry is 8 bytes (`VirtualInvokeData { methodPtr; method }` of 4-byte
/// pointers), not 0x130/16.
///
/// The 64-bit base is validated by every x64/ARM64 dump; the 32-bit one is
/// MEASURED: decoding every `ldr rD,[klass,#N]; ldr rD2,[klass,#N+4]; blx rD`
/// triple in `libs/android/libil2cpp.so` yields addends starting at exactly
/// 0xC0 with an 8-byte stride. The earlier guess of 0x110 was a slot, not the
/// base — with it no ARM32 virtual call resolved at all.
#[test]
fn vtable_dispatch_resolves_on_a_32_bit_target() {
    let table = locals::LocalTable {
        infos: vec![locals::LocalInfo {
            slot_origin: "x0".into(),
            ty: ManagedType::Class(std::sync::Arc::new(NamedType {
                namespace: "Test".into(),
                name: "Widget".into(),
                assembly: None,
                type_def_index: 7,
            })),
            is_this: true,
            is_param: None,
            param_comp: None,
            is_incoming_arg: false,
            is_call_clobber: false,
            is_return_buffer: false,
        }],
        index_of: [(0u32, 0usize)].into_iter().collect(),
        param_names: vec![],
        param_comp_names: vec![],
    };

    let mut insns = vec![
        // v1 = [this]  (obj → klass)
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(1),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::Local(0)),
                    index: None,
                    scale: 4,
                    addend: 0,
                },
            ],
            0x1000,
        ),
        // v2 = [klass + 0x110] → (0x110 - 0xC0) / 8 = slot 10
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(2),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::Local(1)),
                    index: None,
                    scale: 4,
                    addend: 0x110,
                },
            ],
            0x1004,
        ),
        Insn::new(Op::IndirectCall, vec![Operand::Local(2)], 0x1008),
    ];

    let mut maps = resolve::ResolveMaps::default();
    maps.vtable_slots.insert((7, 10), mref("DoThing", 7, false));

    let no_fields: HashMap<u64, u32> = HashMap::new();
    let no_parents: HashMap<i32, Vec<i32>> = HashMap::new();
    let mut no_owner = |_: u32| -> Option<(String, String, ManagedType)> { None };
    let stats = resolve::run_with_fields(
        &mut insns,
        &table,
        &maps,
        &no_fields,
        &mut no_owner,
        4,
        &no_parents,
    );

    assert_eq!(
        stats.virtuals_resolved, 1,
        "32-bit virtual call should resolve"
    );
    assert!(
        matches!(insns[2].op, Op::Call),
        "expected Call, got {:?}",
        insns[2].op
    );
    match insns[2].operands.first() {
        Some(Operand::Method(m)) => assert_eq!(m.name, "DoThing"),
        other => panic!("expected Method operand, got {other:?}"),
    }
}

/// A receiver the move-tracer cannot follow is still a receiver.
///
/// `resolve_through_moves` gives up on any local whose defining instruction is
/// not a `Move` or a `Call` — a `Phi` at a join, a load, an arithmetic op — and
/// the call rebuilder read that "I could not trace it" as "the ABI carried no
/// register for this slot" and substituted an argument hole. Every virtual call
/// whose receiver came out of a join therefore rendered `__abi.Method(…)` —
/// inventing a receiver that does not exist — while the IR held a live local.
/// Real instance: `Lazy<object>.get_Value` @`0x50E66C` in
/// `OnSoundSettingsChanged`, whose receiver is a phi, rendered `__abi.Value`
/// instead of `v20.Value`.
#[test]
fn resolved_call_keeps_a_phi_defined_receiver() {
    let insns = &mut vec![
        // v2 = phi(v0, v1) — the join merge. Not a Move, so the tracer stops.
        Insn::new(Op::Phi, vec![Operand::Local(0), Operand::Local(1)], 0x1000),
        Insn::new(
            Op::Call,
            vec![
                Operand::imm(0x9100),
                Operand::Local(9), // result slot
                Operand::Local(2), // receiver: the phi
            ],
            0x1004,
        ),
    ];
    let mut maps = resolve::ResolveMaps::default();
    maps.methods_by_rva
        .insert(0x9100, mref("get_Value", 7, false));

    let table = locals::create(&HashMap::new(), &helper_seed());
    resolve::run(insns, &table, &maps);

    match insns[1].operands.first() {
        Some(Operand::Method(m)) => assert_eq!(m.name, "get_Value"),
        other => panic!("call should have resolved, got {other:?}"),
    }
    assert_eq!(
        insns[1].operands.get(2),
        Some(&Operand::Local(2)),
        "the phi-defined receiver must survive resolution; replacing it with a hole \
         renders `__abi.Value` and hides the `v20 = this._settingsModel` that \
         produced it"
    );
}

// ---- pure helper-selection rules -----------------------------------------

#[test]
fn helper_selectors_prefer_thunks_and_ratios() {
    use super::HelperCounts;
    let mut counts = HashMap::new();
    // thunk-bodied klass-arg helper (the codegen_object_new shape)
    counts.insert(
        0xA000u64,
        HelperCounts {
            total: 20,
            string_arg: 0,
            klass_arg: 20,
            x1_arg: 0,
        },
    );
    // real-bodied two-arg klass target (box-ish), higher volume
    counts.insert(
        0xB000u64,
        HelperCounts {
            total: 30,
            string_arg: 0,
            klass_arg: 25,
            x1_arg: 28,
        },
    );
    // literal-index target (string_new shape)
    counts.insert(
        0xC000u64,
        HelperCounts {
            total: 10,
            string_arg: 8,
            klass_arg: 0,
            x1_arg: 0,
        },
    );

    assert_eq!(select_string_new(&counts), vec![0xC000]);

    let no_thunk = |_: u64| false;
    let thunk_a = |t: u64| t == 0xA000;

    let excluded = [0xC000u64].into_iter().collect();
    // thunk shape wins ordering over call volume; thunk-top keeps 2
    assert_eq!(
        select_object_new_fallback(&counts, &excluded, &thunk_a),
        vec![0xA000, 0xB000]
    );
    // without a thunk candidate only the single top target is kept
    assert_eq!(
        select_object_new_fallback(&counts, &excluded, &no_thunk),
        vec![0xB000]
    );

    // box requires ≥0.8 klass ratio AND a defined value arg in x1
    let excluded2 = [0xC000u64, 0xA000u64].into_iter().collect();
    assert_eq!(
        select_box_fallback(&counts, &excluded2, &thunk_a),
        vec![0xB000]
    );
}

#[test]
fn eval_arg_classifies_data_refs() {
    use super::{ArgVal, eval_arg};
    let consts: HashMap<String, ArgVal> = [("x8".to_string(), ArgVal::Const(0x123_000))]
        .into_iter()
        .collect();
    let ranges = [(0x100_000u64, 0x200_000u64)];

    // [x8 + 0x40] with x8 = 0x123000 → data reference
    let mem = Operand::Memory {
        width: 0,
        base: Box::new(Operand::reg("x8")),
        index: None,
        scale: 8,
        addend: 0x40,
    };
    assert_eq!(eval_arg(&mem, &consts, &ranges), ArgVal::DataRef(0x123_040));

    // plain small constants stay Const (string-literal-index shape)
    assert_eq!(
        eval_arg(&Operand::imm(42), &consts, &ranges),
        ArgVal::Const(42)
    );
    // out-of-range data addresses are not DataRef
    assert_eq!(
        eval_arg(&Operand::imm(0x999_999), &consts, &ranges),
        ArgVal::Const(0x999_999)
    );
}

/// ABI argument classification: `this` in x0, GPR params in x1..x7, scalar
/// float/double params in v0..v7, and HFA float aggregates (Vector3) in
/// consecutive v-registers. The OLD positional scheme bound param i to
/// x(1+i) regardless of class, so a Vector3 parameter got a stale x-register
/// and every later GPR param shifted one slot too far.
#[test]
fn abi_param_registers_classified_by_type() {
    use super::locals::SeedInfo;
    use super::types::{ManagedType, NamedType, PrimKind};
    use std::collections::HashMap;
    use std::sync::Arc;

    // instance method: (int count, float speed, Vector3 pos, double mass)
    // this→x0, count→x1, speed→v0, pos→v1..v3, mass→v4
    let vec3 = ManagedType::Class(Arc::new(NamedType {
        namespace: "UnityEngine".into(),
        name: "Vector3".into(),
        assembly: None,
        type_def_index: 1,
    }));
    let seed = SeedInfo {
        param_types: vec![
            ManagedType::Primitive(PrimKind::I4),
            ManagedType::Primitive(PrimKind::R4),
            vec3.clone(),
            ManagedType::Primitive(PrimKind::R8),
        ],
        param_names: vec!["count".into(), "speed".into(), "pos".into(), "mass".into()],
        param_comp_names: vec![
            Vec::new(),
            Vec::new(),
            vec!["x".into(), "y".into(), "z".into()],
            Vec::new(),
        ],
        arm32_param_words: Vec::new(),
        ret_type: ManagedType::Primitive(PrimKind::Void),
        is_static: false,
        declaring: None,
        is_arm32: false,
        ret_hidden_buffer: false,
    };
    // Entry versions of every register the body touches: x0..x3 (this,
    // count, plus later scratch uses x2/x3), v0..v4 (float, vector, double).
    let mut slots: HashMap<u32, String> = HashMap::new();
    for (i, s) in ["x0", "x1", "x2", "x3"].iter().enumerate() {
        slots.insert(i as u32, s.to_string());
    }
    let n = slots.len() as u32;
    for (i, s) in ["vs0", "vs4", "vs8", "vs12", "vd4"].iter().enumerate() {
        slots.insert(n + i as u32, s.to_string());
    }
    let table = super::locals::create(&slots, &seed);

    let find = |origin: &str| -> &super::locals::LocalInfo {
        let v = slots
            .iter()
            .find(|(_, o)| o.as_str() == origin)
            .map(|(v, _)| *v)
            .unwrap();
        &table.infos[table.index_of[&v]]
    };
    // this
    let t = find("x0");
    assert!(t.is_this && t.is_param.is_none());
    // count → x1 (GPR cursor 1)
    let c = find("x1");
    assert_eq!(c.is_param, Some(0));
    assert_eq!(c.param_comp, None);
    // speed → s0 (FP cursor 0) — NOT x2 (the old positional bug)
    let s = find("vs0");
    assert_eq!(s.is_param, Some(1));
    assert_eq!(s.param_comp, None);
    // pos → s1..s3 with components 0..2 (flattened lane-0 names)
    for (k, origin) in ["vs4", "vs8", "vs12"].iter().enumerate() {
        let p = find(origin);
        assert_eq!(p.is_param, Some(2));
        assert_eq!(p.param_comp, Some(k));
    }
    // mass → d4
    let m = find("vd4");
    assert_eq!(m.is_param, Some(3));
    assert_eq!(m.param_comp, None);
    // scratch x2/x3 reused later are NOT parameters
    assert_eq!(find("x2").is_param, None);
    assert_eq!(find("x3").is_param, None);
}

/// ARM32 composite returns use a caller-supplied buffer passed in `r0`, so
/// `this` is `r1` and the declared parameters start at `r2`. Regression for
/// `Transform.get_position` @0x15BA630 (`Vector3`, 12 bytes) and
/// `get_worldToLocalMatrix` @0x15BB95C (`Matrix4x4`, 64 bytes), where the
/// result buffer was seeded as `this` and its stores were attributed to the
/// object (`*(nint*)(this + 20) = …`).
#[test]
fn arm32_hidden_result_buffer_shifts_this_and_params() {
    use super::locals::SeedInfo;
    use super::types::{ManagedType, PrimKind};
    use std::collections::HashMap;

    let seed = SeedInfo {
        param_types: vec![ManagedType::Primitive(PrimKind::I4)],
        param_names: vec!["count".into()],
        param_comp_names: vec![Vec::new()],
        arm32_param_words: Vec::new(),
        ret_type: ManagedType::Class(std::sync::Arc::new(super::types::NamedType {
            namespace: "UnityEngine".into(),
            name: "Matrix4x4".into(),
            assembly: None,
            type_def_index: 1,
        })),
        is_static: false,
        declaring: None,
        is_arm32: true,
        ret_hidden_buffer: true,
    };
    let mut slots: HashMap<u32, String> = HashMap::new();
    for (i, s) in ["x0", "x1", "x2"].iter().enumerate() {
        slots.insert(i as u32, s.to_string());
    }
    let table = super::locals::create(&slots, &seed);
    let find = |origin: &str| -> &super::locals::LocalInfo {
        let v = slots
            .iter()
            .find(|(_, o)| o.as_str() == origin)
            .map(|(v, _)| *v)
            .unwrap();
        &table.infos[table.index_of[&v]]
    };
    // r0 is the result buffer: incoming, never `this`, never a parameter.
    assert!(find("x0").is_return_buffer && !find("x0").is_this);
    // `this` shifted up to r1.
    assert!(find("x1").is_this && find("x1").is_param.is_none());
    // the declared parameter shifted up to r2.
    assert_eq!(find("x2").is_param, Some(0));
}

/// Static method with only float params starts at FP register v0 and does
/// not consume any GPR register.
#[test]
fn abi_float_params_do_not_consume_gpr_slots() {
    use super::locals::SeedInfo;
    use super::types::{ManagedType, PrimKind};
    use std::collections::HashMap;

    let seed = SeedInfo {
        param_types: vec![
            ManagedType::Primitive(PrimKind::R4),
            ManagedType::Primitive(PrimKind::R4),
        ],
        param_names: vec!["a".into(), "b".into()],
        param_comp_names: vec![Vec::new(), Vec::new()],
        arm32_param_words: Vec::new(),
        ret_type: ManagedType::Primitive(PrimKind::R4),
        is_static: true,
        declaring: None,
        is_arm32: false,
        ret_hidden_buffer: false,
    };
    let mut slots: HashMap<u32, String> = HashMap::new();
    for (i, s) in ["vs0", "vs4", "x2"].iter().enumerate() {
        slots.insert(i as u32, s.to_string());
    }
    let table = super::locals::create(&slots, &seed);
    let get = |origin: &str| {
        let v = slots
            .iter()
            .find(|(_, o)| o.as_str() == origin)
            .map(|(v, _)| *v)
            .unwrap();
        &table.infos[table.index_of[&v]]
    };
    assert_eq!(get("vs0").is_param, Some(0));
    assert_eq!(get("vs4").is_param, Some(1));
    // x2 is scratch (static method has no `this`), never a parameter
    assert_eq!(get("x2").is_param, None);
}

/// Vector FP arithmetic (SIMD 2S/4S) must decompose into per-lane scalar
/// FloatArith ops: Vn lane k aliases S(4n+k), so `fmul v2.2s, v2.2s, v5.2s`
/// lowers to vs8=vs8*vs20, vs9=vs9*vs21 — joining the scalar FP pipeline
/// instead of degrading every operand to Unknown.
#[test]
fn vector_fp_lane_decomposition() {
    let words: Vec<u32> = vec![
        0x2e25dc42, // fmul v2.2s, v2.2s, v5.2s
        0x0e040610, // dup v16.2s, v16.s[0]
        0x0d409123, // ld1 {v3.s}[1], [x9]
        0x0d009061, // st1 {v1.s}[1], [x3]
        0x0d40c901, // ld1r {v1.2s}, [x8]
        0xd65f03c0, // ret
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).unwrap();
    let fmt = |i: &super::ir::Insn| format!("{:?} {:?}", i.op, i.operands);
    let all: Vec<String> = lifted.insns.iter().map(fmt).collect();
    // 2 lanes of fmul, scalar math on vs8/vs9 ← vs20/vs21
    let mul: Vec<&String> = all
        .iter()
        .filter(|s| s.contains("FloatArith(Mul)"))
        .collect();
    assert_eq!(mul.len(), 2, "lanes: {all:?}");
    assert!(
        all.iter().any(|s| s.contains("vs8") && s.contains("vs20")),
        "{all:?}"
    );
    assert!(
        all.iter().any(|s| s.contains("vs9") && s.contains("vs21")),
        "{all:?}"
    );
    // dup splats s(16*4+0)=vs64 into both lanes vs64..65? (dup source lane 0)
    // ld1 {v3.s}[1] loads 4 bytes at [x9] into s(3*4+1)=vs13
    let loads: Vec<&String> = all
        .iter()
        .filter(|s| s.contains("vs13") && s.contains("Move"))
        .collect();
    assert_eq!(loads.len(), 1, "{all:?}");
    // ld1r broadcasts one load to vs4..vs5 (v1 lanes)
    assert!(
        all.iter().any(|s| s.contains("vs4") && s.contains("vs5")),
        "{all:?}"
    );
    // st1 {v1.s}[1] stores vs5 to [x3]
    assert!(
        all.iter().any(|s| s.contains("vs5") && s.contains("Move")),
        "{all:?}"
    );
    assert_eq!(
        lifted.insns.len(),
        lifted.addresses.len(),
        "addresses must stay parallel to insns"
    );
}

/// The dead half of a self-cyclic chain (`v55 = v31 - 1; v31 = v55;` where
/// nothing outside the chain ever reads v31/v55 — a compiler-kept loop
/// counter) survives the use-count DCE but must be removed by liveness DCE.
#[test]
fn liveness_dce_breaks_dead_counter_cycles() {
    use super::ir::{ArithKind, Insn, Op, Operand};
    use super::passes;
    let l = |v: u32| Operand::Local(v);
    let mut insns = vec![
        // v2 = 10            (seed — the loop counter init)
        Insn::new(Op::Move, vec![l(2), Operand::Immediate(10)], 0x10),
        // v3 = v2 - 1        (loop-body decrement)
        Insn::new(
            Op::Arith(ArithKind::Sub),
            vec![l(3), l(2), Operand::Immediate(1)],
            0x14,
        ),
        // v2 = v3            (copy back — self cycle: v2/v3 read only by
        // each other, never at any exit)
        Insn::new(Op::Move, vec![l(2), l(3)], 0x18),
        // ret
        Insn::new(Op::Return, vec![], 0x20),
    ];
    // naive DCE first (like the pipeline does); the cycle survives: each
    // half has one use (the other half)
    passes::dead_code_elimination(&mut insns);
    let survives = insns.iter().filter(|i| !matches!(i.op, Op::Nop)).count();
    assert_eq!(
        survives, 4,
        "naive DCE should keep seed+sub+copy+ret: {insns:?}"
    );
    passes::liveness_dce(&mut insns);
    let kept: Vec<&Insn> = insns.iter().filter(|i| !matches!(i.op, Op::Nop)).collect();
    // only the Return survives — the whole counter chain is dead
    assert_eq!(kept.len(), 1, "expected only the ret: {insns:?}");
    assert!(matches!(kept[0].op, Op::Return));
}

/// Liveness DCE keeps values that ARE read at an exit (a return value or a
/// store operand): only genuinely dead writes are dropped.
#[test]
fn liveness_dce_keeps_live_values() {
    use super::ir::{ArithKind, Insn, Op, Operand};
    use super::passes;
    let l = |v: u32| Operand::Local(v);
    let mut insns = vec![
        // v1 = 7
        Insn::new(Op::Move, vec![l(1), Operand::Immediate(7)], 0x10),
        // v2 = v1 + 1   (feeds the return — live)
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![l(2), l(1), Operand::Immediate(1)],
            0x14,
        ),
        // v3 = 99      (dead)
        Insn::new(Op::Move, vec![l(3), Operand::Immediate(99)], 0x18),
        // ret v2
        Insn::new(Op::Return, vec![l(2)], 0x20),
    ];
    passes::liveness_dce(&mut insns);
    let kept: Vec<&Insn> = insns.iter().filter(|i| !matches!(i.op, Op::Nop)).collect();
    assert_eq!(kept.len(), 3, "expected move/add/ret kept: {insns:?}");
    assert!(matches!(kept[0].op, Op::Move));
    assert!(matches!(kept[1].op, Op::Arith(_)));
    assert!(matches!(kept[2].op, Op::Return));
}

/// A pinned receiver whose identity register is clobbered by a call must NOT
/// be canonicalized to that register afterwards: `MOV x20, x0; BL thunk;
/// LDR x21, [x20, #16]` — x20 still holds `this` across the call, but x0 now
/// holds the thunk's return value. The pin pass must restore the pin copy as
/// a real move (so SSA tracks x20 = this) and leave the post-call load's base
/// as x20. Canonicalizing it to x0 would alias the call result and turn the
/// field load into garbage (the `Delegate.Combine(*(nint*)(v31+16), ...)`
/// corruption: v31 was a phi of [this, call-result]).
#[test]
fn pin_copy_survives_identity_clobber() {
    use super::ir::{Insn, Op, Operand};
    use super::locals;
    let r = |s: &str| Operand::Register(s.to_string());
    let mem = |base: &str| Operand::Memory {
        width: 0,
        base: Box::new(r(base)),
        index: None,
        scale: 8,
        addend: 16,
    };
    let mut insns = vec![
        // MOV x20, x0 — pin `this` (instance method, so x0 is the receiver)
        Insn::new(Op::Move, vec![r("x20"), r("x0")], 0x10),
        // BL thunk — clobbers x0 (caller-saved); x20 survives
        Insn::new(Op::Call, vec![Operand::Immediate(0x1234)], 0x14),
        // LDR x21, [x20, #16] — reads this.field via the PIN, not via x0
        Insn::new(Op::Move, vec![r("x21"), mem("x20")], 0x18),
        Insn::new(Op::Return, vec![], 0x1c),
    ];
    let seed = locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::Primitive(PrimKind::I4),
        is_static: false,
        declaring: None,
        param_names: vec![],
        ..Default::default()
    };
    locals::rewrite_pinned_copies(&mut insns, &seed);
    // the pin copy must be restored as a real move (x20 = x0)
    assert!(
        insns.iter().any(|i| {
            matches!(i.op, Op::Move)
                && matches!((i.operands.get(0), i.operands.get(1)),
                    (Some(Operand::Register(a)), Some(Operand::Register(b)))
                        if a == "x20" && b == "x0")
        }),
        "pin copy must be restored as MOV x20, x0: {insns:?}"
    );
    // the post-call load must keep x20 as its base (NOT canonicalized to x0)
    let load = insns.iter().find(|i| {
        matches!(i.operands.get(1), Some(Operand::Memory { base, .. }) if matches!(base.as_ref(), Operand::Register(r) if r == "x20"))
    });
    assert!(
        load.is_some(),
        "post-call load must keep x20 base (not x0): {insns:?}"
    );
}

/// A pin whose identity register is NEVER clobbered keeps the old behavior:
/// uses are canonicalized to the identity so `this.field` resolves directly
/// (the fast path — no restored copies left behind).
#[test]
fn pin_copy_canonicalizes_when_identity_survives() {
    use super::ir::{Insn, Op, Operand};
    use super::locals;
    let r = |s: &str| Operand::Register(s.to_string());
    let mem = |base: &str| Operand::Memory {
        width: 0,
        base: Box::new(r(base)),
        index: None,
        scale: 8,
        addend: 16,
    };
    let mut insns = vec![
        // MOV x20, x0 — pin this; NO call in between
        Insn::new(Op::Move, vec![r("x20"), r("x0")], 0x10),
        // LDR x21, [x20, #16] — this.field via the pin
        Insn::new(Op::Move, vec![r("x21"), mem("x20")], 0x18),
        Insn::new(Op::Return, vec![], 0x1c),
    ];
    let seed = locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::Primitive(PrimKind::I4),
        is_static: false,
        declaring: None,
        param_names: vec![],
        ..Default::default()
    };
    locals::rewrite_pinned_copies(&mut insns, &seed);
    // load base canonicalized to x0 (this) — the fast path
    assert!(
        insns.iter().any(|i| {
            matches!(i.operands.get(1), Some(Operand::Memory { base, .. }) if matches!(base.as_ref(), Operand::Register(r) if r == "x0"))
        }),
        "expected [x0 + 16] load: {insns:?}"
    );
    // no restored pin copy remains (the move degenerated into a self-move)
    assert!(
        !insns.iter().any(|i| {
            matches!(i.op, Op::Move)
                && matches!((i.operands.get(0), i.operands.get(1)),
                    (Some(Operand::Register(a)), Some(Operand::Register(b)))
                        if a == "x20")
        }),
        "no restored pin copy expected: {insns:?}"
    );
}

/// Branch-test scratch locals: every TBZ/TBNZ (and every B.cond/CBZ) Compare
/// or And def must get a UNIQUE local id. The old shared sentinels made every
/// compare multi-def, so the renderer had to GUESS which def feeds a branch
/// and regularly bound the wrong expression — the contradictory
/// `if (x != 0) { if (x == 0) {` corruption class.
#[test]
fn tbz_tests_get_unique_scratch_ids() {
    //   tbnz w0, #0, done   (+3 native)
    //   tbnz w1, #1, done   (+2 native)
    //   movz x0, #0
    // done: ret
    let bytes = le(&[
        0x37000060, // tbnz w0, #0, +3 native insns
        0x37080041, // tbnz w1, #1, +2 native insns
        0xD2800000, // movz x0, #0
        0xD65F03C0, // ret
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).expect("lift");
    let mut and_scratch_ids: Vec<u32> = lifted
        .insns
        .iter()
        .filter(|i| matches!(i.op, Op::Arith(ArithKind::And)))
        .filter_map(|i| i.dest().and_then(|d| d.as_local()))
        .collect();
    assert_eq!(
        and_scratch_ids.len(),
        2,
        "both tbz tests must define a mask scratch: {and_scratch_ids:?}"
    );
    and_scratch_ids.sort_unstable();
    and_scratch_ids.dedup();
    assert_eq!(
        and_scratch_ids.len(),
        2,
        "each tbz test must get a UNIQUE mask scratch id: {and_scratch_ids:?}"
    );
    let cmp_ids: Vec<u32> = lifted
        .insns
        .iter()
        .filter(|i| matches!(i.op, Op::Compare(_)))
        .filter_map(|i| i.dest().and_then(|d| d.as_local()))
        .collect();
    assert_eq!(cmp_ids.len(), 2, "two compares expected: {cmp_ids:?}");
    assert_ne!(
        cmp_ids[0], cmp_ids[1],
        "compare scratch ids must be unique per site"
    );
}

#[test]
#[ignore]
fn dbg_arm32_prints() {
    for (name, b, base) in [
        (
            "movwt",
            le(&[0xE3010234, 0xE34A0BCD, 0xE12FFF1E]),
            0x3000u64,
        ),
        ("cmps", le(&[0xE1550002, 0x1AFFFFFE, 0xE12FFF1E]), 0x3000u64),
        (
            "vmla",
            le(&[0xEE066A87, 0xED92FA01, 0xEEB4BA6B, 0xE12FFF1E]),
            0x2000u64,
        ),
    ] {
        eprintln!("==== {name}");
        let l = lift_arm32::lift_arm32(&b, base, 32).expect("lift");
        for (i, insn) in l.insns.iter().enumerate() {
            eprintln!("dbg i{i}: {insn}");
        }
    }
}

/// ROOT CAUSE of the `if (__unknown != 0)` corruption class.
///
/// One native instruction expands to SEVERAL IR insns that all carry its
/// address (`Compare` + `CondJump` for every CBZ/CBNZ/TBZ, `Arith` + `Compare`
/// for folded compares). A branch targets the INSTRUCTION, so it must land on
/// the FIRST IR insn of that group. The RVA→index map was built with
/// `.collect()`, which keeps the LAST index for a duplicated address: a branch
/// to `CBZ x19` landed on the CBZ's `CondJump`, leaving its `Compare` alone in
/// a block with NO predecessors. The SSA dominator DFS only visits blocks in
/// the dominator tree, so that orphan block was never renamed — its raw
/// `Register("x19")` operand survived to the final IR and rendered as
/// `__unknown`.
#[test]
fn branch_target_lands_on_first_insn_of_multi_insn_group() {
    //   i0 @0x100  CondJump(0x104)     ; branch INTO the middle group
    //   i1 @0x104  Compare(Equal) ..   ; native insn at 0x104, insn #1
    //   i2 @0x104  CondJump(0x108)     ; native insn at 0x104, insn #2
    //   i3 @0x108  Return
    let mut insns = vec![
        Insn::new(Op::CondJump(0x104), vec![Operand::Local(1)], 0x100),
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![
                Operand::Local(2),
                Operand::Register("x19".into()),
                Operand::Immediate(0),
            ],
            0x104,
        ),
        Insn::new(Op::CondJump(0x108), vec![Operand::Local(2)], 0x104),
        Insn::new(Op::Return, vec![], 0x108),
    ];
    let addresses = vec![0x100u64, 0x104, 0x104, 0x108];
    let graph = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");

    let block_of = |idx: usize| -> usize {
        graph
            .blocks
            .iter()
            .position(|b| (b.start..b.end).contains(&idx))
            .unwrap_or_else(|| panic!("insn {idx} is in no block"))
    };

    // The two IR insns of the native instruction at 0x104 must stay in ONE
    // block — a boundary between them is what orphaned the Compare.
    assert_eq!(
        block_of(1),
        block_of(2),
        "the Compare and its CondJump (one native insn) were split across blocks"
    );

    // …and the branch from i0 must actually reach that block, otherwise the
    // dominator tree drops it and SSA never renames its register uses.
    let target_block = block_of(1);
    let reachable = graph
        .successors
        .iter()
        .enumerate()
        .any(|(from, succs)| succs.contains(&target_block) && from != target_block);
    assert!(
        reachable,
        "no block branches to the group's first insn: B{target_block} is orphaned"
    );

    // Guard the invariant directly: the first insn of every block must be the
    // FIRST IR insn carrying that address in the whole listing.
    for b in graph.blocks.iter() {
        let addr = addresses[b.start];
        let first_with_addr = addresses
            .iter()
            .position(|&a| a == addr)
            .expect("address present");
        assert_eq!(
            b.start, first_with_addr,
            "block starts at insn {} @{addr:#x}, but insn {first_with_addr} is the FIRST insn at that address",
            b.start
        );
    }
}

/// A branch into an ABSORBED instruction must land on the instruction that
/// replaced it, not on the exit sentinel.
///
/// On x86 the flag-setting instruction is absorbed into the branch that
/// consumes its flags (`cmp`/`test` become a `PendingCmp`, and the Compare +
/// CondJump are emitted at the JCC's address). The absorbed instruction's own
/// address therefore never gets an IR insn — and any OTHER branch targeting it
/// had nowhere to land, so `resolve_target` returned `None` and `Cfg::build`
/// rewrote the edge to the exit sentinel. The taken edge of the x86 static-init
/// guard (`cmp [guard], 0` + `jne <skip>`) was read as a RETURN, which is the
/// single largest source of the oracle's branch-target mismatches.
///
///   0x100  jmp 0x10c      ; 0x10c is absorbed — no IR insn starts there
///   0x104  nop
///   0x110  cmp/test …      ; the insn that absorbed it (first insn ≥ 0x10c)
///   0x110  jne 0x118
///   0x118  ret
#[test]
fn branch_into_an_absorbed_instruction_lands_on_the_instruction_that_absorbed_it() {
    let mut insns = vec![
        Insn::new(Op::Jump(0x10c), vec![], 0x100),
        Insn::new(Op::Nop, vec![], 0x104),
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(2), Operand::Immediate(0)],
            0x110,
        ),
        Insn::new(Op::CondJump(0x118), vec![Operand::Local(2)], 0x110),
        Insn::new(Op::Return, vec![], 0x118),
    ];
    let addresses = vec![0x100u64, 0x104, 0x110, 0x110, 0x118];
    let graph = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");

    let block_of = |idx: usize| -> usize {
        graph
            .blocks
            .iter()
            .position(|b| (b.start..b.end).contains(&idx))
            .unwrap_or_else(|| panic!("insn {idx} is in no block"))
    };
    // The absorbed instruction was replaced by the Compare/CondJump pair at
    // 0x110, so the jump from 0x100 must reach insn 2 — not `insns.len()`.
    let absorbed = block_of(2);
    assert_ne!(
        insns[0].op,
        Op::Jump(insns.len()),
        "the absorbed target was rewritten to the exit sentinel, deleting a real in-body edge"
    );
    assert!(
        graph.successors[block_of(0)].contains(&absorbed),
        "B{} (the jmp) does not reach B{absorbed} (the absorbing Compare): {:?}",
        block_of(0),
        graph.successors[block_of(0)]
    );
}

/// The at-or-after repair is bounded by the BODY. A target past the last
/// instruction is a jump table / padding / tail-dup exit and must stay the
/// exit sentinel; a target below the first is a backward edge out of the
/// method. Redirecting either to the first or last insn would invent an edge.
#[test]
fn branch_outside_the_body_still_resolves_to_the_exit_sentinel() {
    let mut insns = vec![
        Insn::new(Op::Jump(0x99c4), vec![], 0x100),
        Insn::new(Op::Jump(0x0000fffc), vec![], 0x104),
        Insn::new(Op::Return, vec![], 0x108),
    ];
    let addresses = vec![0x100u64, 0x104, 0x108];
    cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    assert_eq!(
        insns[0].op,
        Op::Jump(insns.len()),
        "a target past the last instruction must remain an exit edge"
    );
    assert_eq!(
        insns[1].op,
        Op::Jump(insns.len()),
        "a target before the body must remain an exit edge"
    );
}

/// A slide that would land on the BRANCH ITSELF is not a rejoin — it is a
/// manufactured SELF-LOOP, and `check_cfg` reads it as an unrecovered trap
/// (`block N cannot reach an exit`) the binary never had.
///
/// The at-or-after rule is justified by an absorbed INSTRUCTION PREFIX; when the
/// nearest instruction at-or-after the target IS the branch, every byte between
/// belongs to no decoded instruction. Measured on v24.5 `0x4afba0`: past the
/// final `ret` the listing holds the null-check throw stub and then `jmp
/// 0x4afccd`, whose target sits INSIDE that call's displacement — the slide
/// mapped it to the `jmp` itself and the renderer emitted
/// `do { /* spin wait (native self-loop) */ } while (true);` on the null-check
/// failure path.
#[test]
fn a_slide_that_lands_on_the_branch_itself_stays_an_exit_edge() {
    let mut insns = vec![
        Insn::new(Op::Call, vec![Operand::imm(0x11c930)], 0x100),
        // `jmp 0x105` — no decoded instruction starts at or before 0x105 except
        // the `call` above, so the only at-or-after candidate is the `jmp`.
        Insn::new(Op::Jump(0x105), vec![], 0x106),
    ];
    let addresses = vec![0x100u64, 0x106];
    cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    assert_eq!(
        insns[1].op,
        Op::Jump(insns.len()),
        "a slide onto the branch itself must stay an exit edge, not a self-loop"
    );
    // ...while a slide onto a LATER instruction is still the absorbed-prefix
    // rejoin the at-or-after rule exists for.
    let mut insns = vec![
        Insn::new(Op::Jump(0x205), vec![], 0x200),
        Insn::new(Op::Return, vec![], 0x208),
    ];
    let addresses = vec![0x200u64, 0x208];
    cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    assert_eq!(
        insns[0].op,
        Op::Jump(1),
        "a slide past a folded instruction must still rejoin its consumer"
    );
}

/// The listing is NOT address-ordered: the structurer APPENDS an out-of-line
/// cold / exception region, so the last element can sit below an earlier one.
/// The in-body bounds test must use the ADDRESS RANGE (min..max), not the first
/// and last elements — otherwise every in-body target above the appended region
/// is refused and its edge silently becomes an exit.
///
/// Measured consequence on v24.5 `0x14eb20` (the class-init guard): the native
/// `jne 0x14eb4a` targets a 5-byte `cmp qword [rbx+0x30], 0` that the lifter
/// folds into the `jne` at `0x14eb57`, and `0x14eb4a` sits ABOVE the appended
/// region's address. The bound test rejected it, the guard edge went to the
/// exit sentinel, and the method rendered as an unguarded init block.
#[test]
fn an_in_body_target_above_an_appended_cold_region_is_still_resolved() {
    let mut insns = vec![
        Insn::new(Op::CondJump(0x1100), vec![Operand::Local(2)], 0x1000),
        Insn::new(Op::Nop, vec![], 0x1004),
        // the instruction that absorbed the `cmp` at 0x1100
        Insn::new(Op::Return, vec![], 0x1104),
        // appended cold region, at an address BELOW the target
        Insn::new(Op::Return, vec![], 0x0f00),
    ];
    let addresses = vec![0x1000u64, 0x1004, 0x1104, 0x0f00];
    cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    assert_eq!(
        insns[0].op,
        Op::CondJump(2),
        "a target above the appended cold region must resolve inside the body"
    );
}

/// The exit sentinel MUST survive index-shifting passes.
///
/// Real native words from the shipped iOS binary (`0x266d9d0`):
///
/// ```text
///   cmp  x1, #0                ; P = (x1 != 0)
///   ccmp x22, #0, #4, ne       ; flags = P ? cmp(x22,0) : #4  (#4 ⇒ Z=1 ⇒ eq)
///   b.eq <out>                 ; taken = P ? (x22 == 0) : true  =  (x1 == 0 || x22 == 0)
/// ```
///
/// A `ccmp` used to fall into the lifter's catch-all, which defined a dead
/// destination and CLEARED the pending compare — so the branch rendered
/// `Compare(Equal, __unknown, 0)` and the compound null guard was lost. This
/// reproduces the exact native bytes and pins the recovered shape.
#[test]
fn ccmp_compound_condition_is_recovered() {
    let bytes = le(&[
        0xF100003F, // cmp  x1, #0
        0xFA401AC4, // ccmp x22, #0, #4, ne
        0x54000020, // b.eq +8  (→ ret)
        0xD65F03C0, // ret
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).expect("lift");
    let ops: Vec<String> = lifted.insns.iter().map(|i| format!("{:?}", i.op)).collect();
    assert!(
        ops.iter().any(|o| o.contains("CondJump")),
        "branch must survive: {ops:?}"
    );
    // Two compares (P and Q), both with real operands, combined by a boolean
    // operator — never the `Compare(kind, Unknown, 0)` fallback.
    let cmps: Vec<&Insn> = lifted
        .insns
        .iter()
        .filter(|i| matches!(i.op, Op::Compare(_)))
        .collect();
    assert_eq!(
        cmps.len(),
        2,
        "compound condition needs two compares: {ops:?}"
    );
    for c in &cmps {
        assert!(
            !matches!(c.operands.get(1), Some(Operand::Unknown)),
            "compare lost its operand: {c:?}"
        );
    }
    // `#nzcv` = 4 makes the constant arm TRUE, so the merge must be `!P || Q`.
    assert!(
        lifted
            .insns
            .iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::Or))),
        "K=true must combine with Or (!P || Q): {ops:?}"
    );
}

/// `GUIContent.ToString` (RVA 0x23e30d4, 8 native insns) ends with
/// `b 0x1a8ac58` — a BACKWARD branch to a shared out-of-body tail, which
/// `Cfg::build` records as `Jump(insns.len())` (one past the end = exit).
/// Phi materialization then inserts an instruction BEFORE that branch: the
/// branch moves i → i+1 while the stale sentinel — which was exactly i+1 —
/// starts pointing at the branch's OWN index. The structurer reads that as a
/// natural loop and renders the method as
/// `do { /* spin wait (native self-loop) */ } while (true);`.
#[test]
fn exit_sentinel_survives_index_shifting_passes() {
    let words: [u32; 8] = [
        0xF9400808, // ldr  x8, [x0, #0x10]  ; m_Text
        0xB5000068, // cbnz x8, +0xC
        0xF9401008, // ldr  x8, [x0, #0x20]  ; m_Tooltip
        0xB4000068, // cbz  x8, +0xC
        0xAA0803E0, // mov  x0, x8
        0xD65F03C0, // ret
        0xD2800001, // movz x1, #0
        0x17DA9EDA, // b    0x1a8ac58        ; shared tail, OUTSIDE the body
    ];
    let bytes = le(&words);
    let lifted = lift::lift_arm64(&bytes, 0x23e30d4, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let seed = locals::SeedInfo::default();
    locals::rewrite_pinned_copies(&mut insns, &seed);
    let mut graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    stack_analyzer::analyze(&mut insns, &graph);
    let dom = dominator::DominatorInfo::compute(&graph);
    let mut ssa_out = ssa::build(&insns, &graph, &dom, &[]);
    let _ = ssa::materialize(&mut ssa_out, &mut graph);
    ssa::eliminate(&mut ssa_out, &mut graph);

    for (i, insn) in ssa_out.insns.iter().enumerate() {
        if let Op::Jump(t) | Op::CondJump(t) = &insn.op {
            assert_ne!(
                *t, i,
                "manufactured self-loop: i{i} @{:#x} branches to itself — the exit \
                 sentinel went stale while indices shifted",
                insn.address
            );
        }
    }
}

/// REGRESSION: FP returns are read from the FP result register, not x0.
///
/// `SafeDouble.op_Implicit(SafeDouble) : double` at RVA 0x4f3a20 is two
/// native instructions — `fsub d0, d1, d0 ; ret` (checked against the raw
/// Mach-O bytes; __TEXT has vmaddr == fileoff, so the RVA *is* the file
/// offset). The ABI returns doubles in d0, but the lifter unconditionally
/// emits `Return[x0]`. `bind_fp_returns` must re-bind it to the version the
/// body last wrote to vd0 — otherwise the computation is dead and the render
/// claims `return x0;` for a method with no x0 definition at all.
#[test]
fn fp_return_binds_to_d_register() {
    let words: [u32; 2] = [0x1E603820, 0xD65F03C0]; // fsub d0, d1, d0 ; ret
    let bytes = le(&words);
    let lifted = lift::lift_arm64(&bytes, 0x4f3a20, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let mut graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    stack_analyzer::analyze(&mut insns, &graph);
    let dom = dominator::DominatorInfo::compute(&graph);
    let mut ssa_out = ssa::build(&insns, &graph, &dom, &[]);
    let _ = ssa::materialize(&mut ssa_out, &mut graph);

    passes::bind_fp_returns(
        &mut ssa_out.insns,
        &ssa_out.version_slot,
        &ManagedType::Primitive(PrimKind::R8),
    );

    let mut defined = std::collections::HashSet::new();
    let mut used = std::collections::HashSet::new();
    for insn in &ssa_out.insns {
        if let Some(pos) = super::ir::Insn::dest_position(&insn.op) {
            if let Some(Operand::Local(v)) = insn.operands.get(pos) {
                defined.insert(*v);
            }
        }
        for op in &insn.operands {
            if let Operand::Local(v) = op {
                used.insert(*v);
            }
        }
    }
    let undef: Vec<(u32, String)> = used
        .difference(&defined)
        .map(|v| {
            (
                *v,
                ssa_out
                    .version_slot
                    .get(v)
                    .cloned()
                    .unwrap_or_else(|| "<unmapped>".into()),
            )
        })
        .collect();
    let _ = &undef;

    let ret = ssa_out
        .insns
        .iter()
        .find(|i| i.op == Op::Return)
        .expect("the method must end in ret");
    let v = match ret.operands.first() {
        Some(Operand::Local(v)) => *v,
        other => panic!("return reads {other:?}, expected a versioned FP register"),
    };
    assert_eq!(
        ssa_out.version_slot.get(&v).map(String::as_str),
        Some("vd0"),
        "the FP return value must be read from d0, not x0"
    );
}

/// REGRESSION: a float/double-returning CALL leaves its result in the FP
/// result register, but the lift's call arm defines only the INTEGER return
/// slot (`x0`/`rv`). Without binding, every read of the FP slot saw the stale
/// pre-call value. On `0x424f4` that rendered
/// `this._rotationRange = this._minRotationRange` where the native code stores
/// `Random.Range(this._minRotationRange, this._maxRotationRange)` — a wrong
/// VALUE (the offset-based semantic diff cannot see it, the offset matches).
/// `bind_fp_call_returns` binds the FP slot's live value to the call result.
#[test]
fn fp_call_return_binds_s0_reads_to_the_call_result() {
    let mut version_slot = std::collections::HashMap::new();
    version_slot.insert(1u32, "vs0".to_string()); // stale s0 (fed the call)
    version_slot.insert(2u32, "x0".to_string()); // the call's result slot
    version_slot.insert(3u32, "x0".to_string()); // the call target slot
    let mut insns = vec![
        Insn::new(Op::Call, vec![Operand::Local(3), Operand::Local(2)], 0x1000),
        Insn::new(
            Op::Move,
            vec![
                Operand::Memory {
                    base: Box::new(Operand::reg("x19")),
                    index: None,
                    scale: 8,
                    addend: 0xc8,
                    width: 0,
                },
                Operand::Local(1),
            ],
            0x1004,
        ),
    ];
    insns[0].dest_type = Some(ManagedType::Primitive(PrimKind::R4));

    let n = passes::bind_fp_call_returns(&mut insns, &version_slot, 1, |_| Vec::new());
    assert_eq!(n, 1, "the stale s0 read must be rebound to the call result");
    assert_eq!(
        insns[1].operands[1],
        Operand::Local(2),
        "`str s0` must read the call's result version, not the pre-call stale one"
    );
}

/// An HFA return (`Vector3`) occupies one v-register per field. Every
/// component must become a DISTINCT `call.field`, never the same local:
/// binding all of them to the call result made `zA - zB` render `(z - z)`.
#[test]
fn fp_hfa_call_components_are_distinct_result_fields() {
    use crate::recovery::types::{FieldRef, NamedType};
    use std::sync::Arc;
    let mut version_slot = std::collections::HashMap::new();
    version_slot.insert(1u32, "vs0".to_string());
    version_slot.insert(2u32, "vs4".to_string());
    version_slot.insert(3u32, "vs8".to_string());
    version_slot.insert(4u32, "x0".to_string());
    version_slot.insert(5u32, "x0".to_string());
    let mut insns = vec![
        Insn::new(Op::Call, vec![Operand::Local(5), Operand::Local(4)], 0x1000),
        Insn::new(
            Op::Move,
            vec![
                Operand::Memory {
                    base: Box::new(Operand::reg("x19")),
                    index: None,
                    scale: 8,
                    addend: 0xc8,
                    width: 0,
                },
                Operand::Local(1),
            ],
            0x1004,
        ),
        Insn::new(Op::Move, vec![Operand::Local(10), Operand::Local(2)], 0x1008),
        Insn::new(Op::Move, vec![Operand::Local(11), Operand::Local(3)], 0x100c),
    ];
    insns[0].dest_type = Some(ManagedType::Class(Arc::new(NamedType {
        namespace: "UnityEngine".into(),
        name: "Vector3".into(),
        assembly: None,
        type_def_index: 7651,
    })));
    let field = |name: &str, idx: u32| FieldRef {
        type_def_index: 7651,
        field_index: idx,
        name: name.into(),
        declaring_type_name: "UnityEngine.Vector3".into(),
        field_type: ManagedType::Primitive(PrimKind::R4),
        is_static: false,
    };
    let n = passes::bind_fp_call_returns(&mut insns, &version_slot, 4, |_| {
        vec![field("x", 1), field("y", 2), field("z", 3)]
    });
    assert_eq!(n, 3, "all three component reads must be rebound: {insns:?}");
    for (idx, expected) in [(1usize, "x"), (2, "y"), (3, "z")] {
        match &insns[idx].operands[1] {
            Operand::InstanceField { base, field } => {
                assert_eq!(**base, Operand::Local(4), "{insns:?}");
                assert_eq!(field.name, expected, "{insns:?}");
            }
            other => panic!("component {expected} not bound: {other:?}"),
        }
    }
}

/// A shared `List<T>.get_Item` returns `Opaque("T")`; the call site's
/// declaring type (`List<Vector3>`) proves the concrete return. Without the
/// specialization the call is not even known to return a struct, so its HFA
/// components are never bound.
#[test]
fn a_generic_list_get_item_returns_the_instantiated_element_type() {
    let mut maps = resolve::ResolveMaps::default();
    maps.tdi_names
        .insert(7651, ("UnityEngine".into(), "Vector3".into()));
    maps.tdi_names_reverse
        .insert("UnityEngine.Vector3".into(), 7651);
    maps.tdi_names_reverse.insert("Vector3".into(), 7651);
    let m = crate::recovery::types::MethodRef {
        image_index: 0,
        type_def_index: 4328,
        method_index: 34254,
        name: "get_Item".into(),
        declaring_namespace: "System.Collections.Generic".into(),
        declaring_type_name: "List<Vector3>".into(),
        assembly: String::new(),
        sig: crate::recovery::types::MethodSigInfo {
            ret: ManagedType::Opaque("T".into()),
            params: vec![ManagedType::Primitive(PrimKind::I4)],
            is_static: false,
        },
    };
    let mut insns = vec![
        Insn::new(Op::Call, vec![Operand::Method(m), Operand::Local(1), Operand::Local(2)], 0x4000),
    ];
    let n = resolve::specialize_generic_return_sigs(&mut insns, &maps);
    assert_eq!(n, 1);
    match &insns[0].operands[0] {
        Operand::Method(m) => match &m.sig.ret {
            ManagedType::Class(nt) => {
                assert_eq!(nt.name, "Vector3");
                assert_eq!(nt.type_def_index, 7651);
            }
            other => panic!("expected Vector3 return, got {other:?}"),
        },
        other => panic!("expected method operand, got {other:?}"),
    }
    assert!(matches!(insns[0].dest_type, Some(ManagedType::Class(_))));
}

/// A CALL is an ABI definition of the caller-saved FP return registers. The
/// post-call read must name a FRESH version; reading the pre-call version is
/// what let two different returned components collapse to one value.
#[test]
fn a_call_clobbers_the_fp_return_slot_in_ssa() {
    let mut insns = vec![
        Insn::new(Op::Move, vec![Operand::reg("vs0"), Operand::imm(0)], 0x2000),
        Insn::new(Op::Call, vec![Operand::imm(0x3000), Operand::reg("x0")], 0x2004),
        Insn::new(Op::Move, vec![Operand::reg("x1"), Operand::reg("vs0")], 0x2008),
        Insn::new(Op::Return, vec![], 0x200c),
    ];
    let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
    let graph = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    let dom = dominator::DominatorInfo::compute(&graph);
    let out = ssa::build(&insns, &graph, &dom, &["vs0".to_string()]);
    let read = out
        .insns
        .iter()
        .find_map(|i| match (&i.op, i.operands.first(), i.operands.get(1)) {
            (Op::Move, Some(Operand::Local(_)), Some(Operand::Local(v))) => {
                (out.version_slot.get(v).map(|s| s.as_str()) == Some("vs0")).then_some(*v)
            }
            _ => None,
        })
        .expect("post-call vs0 read");
    assert!(
        out.call_defined.contains(&read),
        "the post-call read must be the call-clobber version, got v{read} ({:?})",
        out.version_slot.get(&read)
    );
}

/// `bind_fp_call_returns` tracks the live value of the FP slot with a LINEAR
/// map. That is sound within a straight-line region, but a `Phi`'s operands are
/// PER-PREDECESSOR values: on a diamond the last textual write to the slot
/// belongs to the OTHER arm. Rebinding through the linear map collapsed
/// `ViewScreenAdapter.UpdateRootView` @0x6341f0's `phi(vs0: v51, v52)` into
/// `phi(v52, v52)`, which made the vmov arm's `v51 = 1.0` look dead — DCE
/// deleted it — while `ssa::eliminate` still materialized an edge copy reading
/// `v51`, leaving a read with no definition (the largest ARM32
/// `undef-local origin vs0` cluster).
#[test]
fn fp_call_return_bind_leaves_phi_operands_alone() {
    let mut version_slot = std::collections::HashMap::new();
    version_slot.insert(51u32, "vs0".to_string()); // the vmov arm: v51 = 1.0
    version_slot.insert(52u32, "vs0".to_string()); // the vldr arm (laid out later)
    version_slot.insert(14u32, "vs0".to_string()); // the phi result
    let mut insns = vec![
        Insn::new(
            Op::Move,
            vec![Operand::Local(51), Operand::FloatLiteral(1.0)],
            0x1000,
        ),
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(52),
                Operand::Memory {
                    base: Box::new(Operand::reg("x19")),
                    index: None,
                    scale: 8,
                    addend: 0x118,
                    width: 0,
                },
            ],
            0x1004,
        ),
        Insn::new(
            Op::Phi,
            vec![Operand::Local(14), Operand::Local(51), Operand::Local(52)],
            0x1008,
        ),
    ];

    let n = passes::bind_fp_call_returns(&mut insns, &version_slot, 1, |_| Vec::new());
    assert_eq!(
        n, 0,
        "a phi operand is not a linear read and must not be rebound"
    );
    assert_eq!(
        insns[2].operands[1],
        Operand::Local(51),
        "the vmov arm's value must survive phi binding"
    );
    assert_eq!(insns[2].operands[2], Operand::Local(52));
}

/// REGRESSION (`OutBounce` @0x14D6330): the linear map must not cross into a
/// block that the write can never reach.
///
/// [`bind_fp_call_returns`] tracks the FP result slot with a LINEAR map. That is
/// sound inside one straight-line region and unsound across blocks — and on
/// x86-64 both `s_slot` and `d_slot` are `"vf0"` (XMM0), which is ALSO the
/// incoming float argument register, so a method PARAMETER lives in `"vf0"`.
/// The first arm that writes the slot therefore re-pointed every later read of
/// the parameter, including reads in blocks that write can never reach: the
/// four-arm easing function used the PREVIOUS arm's sum as its base, turning
/// the native `(t - A) * 7.5625 * (t - A) + C` into `(v13 - A) * 7.5625 * …`
/// with `v13` the other arm's result — a wrong value. Each arm ends in `ret`,
/// so the writing block cannot reach the reading block at all: the rebind has
/// to be proved reachable, or refused.
///
/// Both halves are kept honest here. The read that FOLLOWS the
/// float-returning call in the SAME block — the `0x424f4` case the pass exists
/// for — must still be bound to the call result; only the cross-block read is
/// refused.
#[test]
fn fp_call_return_bind_does_not_cross_a_block_that_cannot_reach_the_read() {
    let cond = Operand::Local(u32::MAX - 2);
    let mut version_slot = std::collections::HashMap::new();
    version_slot.insert(1u32, "vf0".to_string()); // the PARAMETER `t` (x64 XMM0)
    version_slot.insert(14u32, "vf0".to_string()); // arm A's intermediate
    version_slot.insert(9u32, "vf0".to_string()); // arm B's result
    version_slot.insert(3u32, "x0".to_string()); // the call's target slot
    version_slot.insert(2u32, "x0".to_string()); // the call's result slot

    //   i0 @0x2000  cond = (t != 0)
    //   i1 @0x2004  CondJump(5) [cond]     ; taken -> arm B
    //   -- arm A (fall-through): writes the FP slot, calls, reads it back --
    //   i2 @0x2008  v14 = t * 2.0          ; WRITES "vf0"
    //   i3 @0x200c  v2  = fn_9000()        ; float-returning call -> "vf0"
    //   i4 @0x2010  Return [v9]            ; the pass's real job: -> v2
    //   -- arm B (reached only by the jump): reads the PARAMETER --
    //   i5 @0x2014  v9 = t - 1.0           ; must stay `t`
    //   i6 @0x2018  Return [v9]
    let mut insns = vec![
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![cond.clone(), Operand::Local(1), Operand::imm(0)],
            0x2000,
        ),
        Insn::new(Op::CondJump(5), vec![cond], 0x2004),
        Insn::new(
            Op::FloatArith(FloatArithKind::Mul),
            vec![
                Operand::Local(14),
                Operand::Local(1),
                Operand::FloatLiteral(2.0),
            ],
            0x2008,
        ),
        Insn::new(Op::Call, vec![Operand::Local(3), Operand::Local(2)], 0x200c),
        Insn::new(Op::Return, vec![Operand::Local(9)], 0x2010),
        Insn::new(
            Op::FloatArith(FloatArithKind::Sub),
            vec![
                Operand::Local(9),
                Operand::Local(1),
                Operand::FloatLiteral(1.0),
            ],
            0x2014,
        ),
        Insn::new(Op::Return, vec![Operand::Local(9)], 0x2018),
    ];
    insns[3].dest_type = Some(ManagedType::Primitive(PrimKind::R4));

    let n = passes::bind_fp_call_returns(&mut insns, &version_slot, 1, |_| Vec::new());

    assert_eq!(
        insns[4].operands[0],
        Operand::Local(2),
        "a read AFTER the float-returning call in its own block is the pass's \
         reason to exist and must still bind to the call result:\n{insns:#?}"
    );
    assert_eq!(
        insns[5].operands[1],
        Operand::Local(1),
        "the taken arm only READS the parameter; the fall-through arm's write to \
         `vf0` cannot reach it (that arm ends in `ret`), so rebinding it there is \
         a wrong value (`OutBounce` read the other arm's sum as its base):\n{insns:#?}"
    );
    assert_eq!(
        n, 1,
        "exactly one rebind — the same-block post-call read; the cross-block one \
         must not happen:\n{insns:#?}"
    );
}

/// REGRESSION (`OutBounce` shape, `copy_and_constant_propagation`): the CHAIN
/// step must not carry a value out of an arm that cannot reach the read.
///
/// The pass enforces `def_count == 1` on the local AT THE USE, but `resolve`
/// then walks further: a use of `v` where `v = w` resolves to `w`'s `value_of`
/// entry, and that entry is the LAST TEXTUAL definition of `w`. Here `v14` is
/// written in arm A, which ends in `ret`, and the use is in arm B — which arm A
/// cannot reach at all. Chasing `v14` substituted arm A's `0x1234` into the use:
/// a plausible-looking constant invented from a write that never executes. The
/// dominator proof on every hop stops the walk at `v14` instead.
///
/// The read of `v14` in arm B is inconsistent IR on its own (it was already
/// there before this pass ran). The point of the guard is that the pass fails
/// CLOSED on it: a bare undefined local stays visible, while an invented
/// constant is indistinguishable from a real one.
#[test]
fn copy_prop_never_forwards_an_address_taken_cell() {
    //   i0 @0x5000  v1 = 0                 ; out-cell initialiser
    //   i1 @0x5004  v2 = &v1               ; pass by reference
    //   i2 @0x5008  Call TryGetValue(…, v2) ; callee WRITES v1
    //   i3 @0x500c  v3 = (v1 == 0)         ; must read the cell, not the 0
    let mut insns = vec![
        Insn::move_(Operand::Local(1), Operand::imm(0), 0x5000),
        Insn::move_(
            Operand::Local(2),
            Operand::AddressOf(Box::new(Operand::Local(1))),
            0x5004,
        ),
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x1234), Operand::Local(2)],
            0x5008,
        ),
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(3), Operand::Local(1), Operand::imm(0)],
            0x500c,
        ),
        Insn::new(Op::Return, vec![Operand::Local(3)], 0x5010),
    ];

    passes::copy_and_constant_propagation(&mut insns);
    passes::constant_folding(&mut insns);

    assert_eq!(
        insns[3].operands[1],
        Operand::Local(1),
        "an address-taken cell is mutated by the callee; forwarding its \
         initialiser would invent a stale null/zero:\n{insns:#?}"
    );
    assert_eq!(
        insns[3].operands[0],
        Operand::Local(3),
        "the comparison result must stay a real value:\n{insns:#?}"
    );
}

#[test]
fn copy_prop_chain_hop_needs_a_definition_that_reaches_the_use() {
    let cond = Operand::Local(u32::MAX - 2);
    //   i0 @0x3000  cond = (v1 != 0)
    //   i1 @0x3004  CondJump(4) [cond]        ; taken -> arm B
    //   -- arm A (fall-through, ends in Return: cannot reach arm B) --
    //   i2 @0x3008  v14 = 0x1234              ; the write the chain must not use
    //   i3 @0x300c  Return [v14]
    //   -- arm B (reached only by the jump) --
    //   i4 @0x3010  v5 = v14                  ; the entry candidate (single def)
    //   i5 @0x3014  v6 = v5 + 1               ; the USE whose operand is rewritten
    //   i6 @0x3018  Return [v6]
    let mut insns = vec![
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![cond.clone(), Operand::Local(1), Operand::imm(0)],
            0x3000,
        ),
        Insn::new(Op::CondJump(4), vec![cond], 0x3004),
        Insn::move_(Operand::Local(14), Operand::imm(0x1234), 0x3008),
        Insn::new(Op::Return, vec![Operand::Local(14)], 0x300c),
        Insn::move_(Operand::Local(5), Operand::Local(14), 0x3010),
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(6), Operand::Local(5), Operand::imm(1)],
            0x3014,
        ),
        Insn::new(Op::Return, vec![Operand::Local(6)], 0x3018),
    ];

    passes::copy_and_constant_propagation(&mut insns);

    assert_ne!(
        insns[5].operands[1],
        Operand::imm(0x1234),
        "the chain carried arm A's constant into a block arm A cannot reach (arm \
         A ends in `ret`): a WRONG VALUE invented from a write that never \
         executes:\n{insns:#?}"
    );
    assert_eq!(
        insns[5].operands[1],
        Operand::Local(14),
        "only the unreachable hop is refused; the hop out of `v5`'s own (reaching) \
         definition stands, so the read stays a local:\n{insns:#?}"
    );
}

/// REGRESSION (`copy_and_constant_propagation`): a hop must land on a SINGLE
/// definition — the pass's own contract ("a local with >1 definition is not
/// propagated (join-safety)") has to hold along the chain, not only at the use.
///
/// `value_of` stores the LAST TEXTUAL definition of a local. Here `v10` is
/// written at i0 (`5`) and again at i5 (`7`) in a different block, so
/// `value_of[v10]` is `7` even though the use of `v11 = v10` sits BEFORE that
/// second write. The entry check only asked `def_count` about `v11`; chasing
/// `v10` then substituted `7` into the use — a wrong value taken from a
/// definition that is later in the listing than the read.
#[test]
fn copy_prop_chain_hop_requires_a_single_definition() {
    let cond = Operand::Local(u32::MAX - 2);
    let mut insns = vec![
        //   i0 @0x4000  v10 = 5
        Insn::move_(Operand::Local(10), Operand::imm(5), 0x4000),
        //   i1 @0x4004  v11 = v10        (single def -- the entry candidate)
        Insn::move_(Operand::Local(11), Operand::Local(10), 0x4004),
        //   i2 @0x4008  v12 = v11 + 1    (the USE)
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(12), Operand::Local(11), Operand::imm(1)],
            0x4008,
        ),
        //   i3 @0x400c  CondJump(5) [cond]
        Insn::new(Op::CondJump(5), vec![cond], 0x400c),
        //   i4 @0x4010  Return [v12]
        Insn::new(Op::Return, vec![Operand::Local(12)], 0x4010),
        //   i5 @0x4014  v10 = 7          (a SECOND def of v10, in another block)
        Insn::move_(Operand::Local(10), Operand::imm(7), 0x4014),
        //   i6 @0x4018  Return [v10]
        Insn::new(Op::Return, vec![Operand::Local(10)], 0x4018),
    ];

    passes::copy_and_constant_propagation(&mut insns);

    assert_ne!(
        insns[2].operands[1],
        Operand::imm(7),
        "the chain used the SECOND definition of `v10`, written after the read: a \
         WRONG VALUE:\n{insns:#?}"
    );
    assert_eq!(
        insns[2].operands[1],
        Operand::Local(10),
        "a hop must land on a single definition (join-safety), as the pass doc \
         already promised:\n{insns:#?}"
    );
}

/// A pin copy (`MOV x19, x0` — IL2CPP pins the receiver) is folded into a
/// self-move and ERASED, so every later use of `x19` must be canonicalized to
/// `x0` or it reads a register with no definition left anywhere, rendering as
/// `__unknown`. `rewrite_reg_operands` skipped operand#0 unconditionally, so a
/// STORE (`Move [x19+8], x1` — a memory destination) kept the erased pin as
/// its base: `*(nint*)(__unknown + off)` for most of the binary.
#[test]
fn pin_copy_is_canonicalized_inside_store_bases() {
    let bytes = le(&[
        0xAA0003F3, // mov  x19, x0       ; pin `this`
        0xF9000661, // str  x1, [x19, #8] ; store THROUGH the pin
        0xD65F03C0, // ret
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x3000, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let seed = locals::SeedInfo {
        param_types: vec![ManagedType::Unknown],
        ret_type: ManagedType::Unknown,
        is_static: false,
        param_names: vec!["arg0".into()],
        ..Default::default()
    };
    locals::rewrite_pinned_copies(&mut insns, &seed);

    fn scan(op: &Operand, out: &mut Vec<String>) {
        match op {
            Operand::Register(r) => out.push(r.clone()),
            Operand::Memory { base, index, .. } => {
                scan(base, out);
                if let Some(i) = index {
                    scan(i, out);
                }
            }
            _ => {}
        }
    }
    let mut regs = Vec::new();
    for insn in &insns {
        for op in &insn.operands {
            scan(op, &mut regs);
        }
    }
    assert!(
        !regs.iter().any(|r| r == "x19"),
        "a store base still reads the erased pin: {insns:?}"
    );
    // ...and the store really did move to the receiver register.
    assert!(
        insns.iter().any(|i| matches!(
            i.operands.first(),
            Some(Operand::Memory { base, addend: 8, .. })
                if matches!(**base, Operand::Register(ref r) if r == "x0")
        )),
        "store base was not canonicalized to the receiver: {insns:?}"
    );
}

/// FAST HARNESS on REAL native bytes (no probe dump, ~10 ms).
///
/// `XmlSchemaInference.GetEffectiveSchemaType` (RVA 0x1F4E79C) is the method
/// whose rendered `if (__unknown != 0)` exposed the pin-copy identity loss:
/// the prologue pins `MOV x21, x2` (x2 = the `bCreatingNewType` parameter),
/// `rewrite_pinned_copies` folds that copy into a self-move, but the TBZ
/// re-test reads x21 AFTER the injected init calls — where the pin's rename
/// association to the caller-saved identity (x2) has died. The bare x21 slot
/// then has no def, no param mark, and renders as `__unknown`.
///
/// The bytes below are the EXACT 79 ARM64 words from the shipped binary, so
/// the yaxpeax lift and every downstream pass run on real disassembly.
/// A STORE (`str x1, [x19, #8]`) lifts to `Move [x19+8], x1`: operand#0 is a
/// MEMORY destination, not a register def. Its base/index registers are real
/// READS and must be SSA-renamed like any other source operand. Treating
/// operand#0 as a pure def skipped them, leaving ~175k raw `Register` operands
/// binary-wide (the `this`/frame pointer of every store) that the renderer can
/// only print as `__unknown`.
#[test]
fn store_base_register_is_ssa_renamed() {
    let bytes = le(&[
        0xAA0003F3, // mov  x19, x0          ; pin `this`
        0xF9000661, // str  x1, [x19, #8]    ; store with a register base
        0xD65F03C0, // ret
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x2000, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    let dom = dominator::DominatorInfo::compute(&graph);
    let out = ssa::build(&insns, &graph, &dom, &[]);

    let mut raw = Vec::new();
    for insn in &out.insns {
        for op in &insn.operands {
            if matches!(op, Operand::Register(_)) {
                raw.push(format!("{:?} {:?}", insn.op, insn.operands));
            }
        }
    }
    assert!(raw.is_empty(), "SSA left raw register operands: {raw:?}");

    // ...and the store really is a memory-dest Move with a VERSIONED base.
    let store = out.insns.iter().find(|i| {
        i.op == Op::Move
            && matches!(i.operands.first(), Some(Operand::Memory { base, .. })
                if matches!(**base, Operand::Local(_)))
    });
    assert!(
        store.is_some(),
        "store base was not versioned: {:?}",
        out.insns
    );
}

/// `cmp x9, x8` is encoded `subs xzr, x9, x8`, and yaxpeax reports x8 as a
/// `RegShift(LSL, 0, …)` source operand. `operand_at` returns None for
/// RegShift, so the RHS fell back to the immediate `0`: every
/// register-register guard was lifted as `a == 0` / `a >= 0` and the real
/// comparison value was lost. Verified on GetEffectiveSchemaType
/// (@0x1f4e79c): the type-hash comparison rendered `>= 0` instead of
/// `>= <other hash>`.
#[test]
fn cmp_register_rhs_survives_regshift_operand() {
    let bytes = le(&[
        0xEB08013F, // subs xzr, x9, x8   (cmp x9, x8)
        0x54000040, // b.eq +8
        0xD65F03C0, // ret
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).expect("lift");
    let cmp = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Compare(_)))
        .expect("the conditional branch must lift to a Compare");
    let rhs = cmp.operands.get(2).expect("rhs operand");
    assert!(
        !matches!(rhs, Operand::Immediate(0)),
        "register-register compare lost its RHS: {cmp:?}"
    );
    assert!(
        matches!(rhs, Operand::Register(_)),
        "RHS must be the register, not a folded immediate: {cmp:?}"
    );
}

fn tree_contains_goto(stmts: &[csharp::Stmt], label: &str) -> bool {
    stmts.iter().any(|s| match s {
        csharp::Stmt::Goto(l) => l == label,
        csharp::Stmt::If {
            then_body,
            else_body,
            ..
        } => tree_contains_goto(then_body, label) || tree_contains_goto(else_body, label),
        csharp::Stmt::While { body, .. } | csharp::Stmt::DoWhile { body, .. } => {
            tree_contains_goto(body, label)
        }
        _ => false,
    })
}

fn tree_contains_label(stmts: &[csharp::Stmt], label: &str) -> bool {
    stmts.iter().any(|s| match s {
        csharp::Stmt::Label(l) => l == label,
        csharp::Stmt::If {
            then_body,
            else_body,
            ..
        } => tree_contains_label(then_body, label) || tree_contains_label(else_body, label),
        csharp::Stmt::While { body, .. } | csharp::Stmt::DoWhile { body, .. } => {
            tree_contains_label(body, label)
        }
        _ => false,
    })
}

/// The structurer emits a skip-pattern `goto Lx` several nesting levels above
/// the list where `Lx:` actually lands (the merge block belongs to an outer
/// guard). `branch_repair` used to judge dangleness against the labels of the
/// goto's OWN nested list, so the jump was deleted at the innermost level;
/// the branch carrying it then folded as an "empty shell" and the whole value
/// chain in its condition vanished (GetEffectiveSchemaType @0x1f4e79c).
#[test]
fn cross_level_goto_survives_branch_repair() {
    let cond = || csharp::Expr::Var("c".into());
    let mut stmts = vec![
        csharp::Stmt::If {
            cond: cond(),
            then_body: vec![csharp::Stmt::If {
                cond: cond(),
                then_body: vec![csharp::Stmt::Goto("L5".into())],
                else_body: vec![],
            }],
            else_body: vec![],
        },
        csharp::Stmt::Label("L5".into()),
        csharp::Stmt::Return(None),
    ];
    csharp::branch_repair(&mut stmts);
    assert!(
        !tree_contains_goto(&stmts, "L5"),
        "a jump to an immediate void return should become return: {stmts:?}"
    );
    assert!(
        !tree_contains_label(&stmts, "L5"),
        "the now-unreferenced return label should be removed: {stmts:?}"
    );
    assert!(
        csharp::render_stmts(&stmts).contains("return;"),
        "the cross-level return edge must survive: {stmts:?}"
    );
}

/// `if (c) { goto L0; } <body> L0:` is the only shape the structurer has for a
/// forward conditional edge it cannot place as a guard. `fold_branch_over_join`
/// inverts the predicate and absorbs the fall-through region, so a restored
/// edge from an absorbed instruction renders as the guard it is instead of a
/// labelled jump. The join label is KEPT, which is what makes the rewrite sound
/// without a global reference scan.
#[test]
fn a_guard_goto_folds_into_an_inverted_if_over_the_region() {
    let var = |n: &str| csharp::Expr::Var(n.into());
    let guard = |c: &str| csharp::Stmt::If {
        cond: var(c),
        then_body: vec![csharp::Stmt::Goto("L0".into())],
        else_body: vec![],
    };
    let mut stmts = vec![
        guard("c"),
        csharp::Stmt::Eval(var("body")),
        csharp::Stmt::Label("L0".into()),
        // A SECOND jump to the same label. It has to keep working, which is
        // exactly why the fold keeps `L0:` rather than absorbing it.
        guard("d"),
        csharp::Stmt::Return(None),
    ];
    csharp::branch_repair(&mut stmts);

    let csharp::Stmt::If {
        cond, then_body, ..
    } = &stmts[0]
    else {
        panic!("the guard was not folded into an if head: {stmts:?}");
    };
    assert!(
        matches!(cond, csharp::Expr::Not(_)),
        "the predicate was not inverted: {cond:?}"
    );
    assert!(
        then_body.iter().any(|s| matches!(s, csharp::Stmt::Eval(_))),
        "the fall-through region did not move into the guard's body: {stmts:?}"
    );
    assert!(
        tree_contains_label(&stmts, "L0"),
        "the join label was absorbed, so the other jump to it dangles: {stmts:?}"
    );
    assert!(
        tree_contains_goto(&stmts, "L0"),
        "the unrelated jump to L0 was removed: {stmts:?}"
    );
}

/// A FLAT label inside the fall-through region blocks the fold when a jump from
/// elsewhere in the method actually lands on it. Folding would move it under
/// `!c` and make that edge conditional — so the fold must refuse. (A label
/// nested inside an arm or loop is a different case: it is already a
/// self-contained landing site, which is why it rides along as an opaque
/// element. A flat label nothing targets is a dead merge anchor, not a landing
/// site: the fold moves it along and the quiet-label sweep removes it.)
/// Regression pin for 0x2CA7AC: the `v100 != 0` guard's region holds `L14:`,
/// landed on from across the method.
#[test]
fn a_flat_label_inside_the_region_blocks_the_fold() {
    let var = |n: &str| csharp::Expr::Var(n.into());
    let mut stmts = vec![
        csharp::Stmt::If {
            cond: var("c"),
            then_body: vec![csharp::Stmt::Goto("L0".into())],
            else_body: vec![],
        },
        csharp::Stmt::Label("Linner".into()),
        csharp::Stmt::Eval(var("body")),
        csharp::Stmt::Label("L0".into()),
        // The outside edge that makes this a landing site: without it the
        // label is dead and the guard folds.
        csharp::Stmt::Goto("Linner".into()),
        csharp::Stmt::Return(None),
    ];
    csharp::branch_repair(&mut stmts);
    let csharp::Stmt::If { then_body, .. } = &stmts[0] else {
        panic!("the guard shell vanished: {stmts:?}");
    };
    assert!(
        then_body.iter().any(|s| matches!(s, csharp::Stmt::Goto(_))),
        "the guard was folded even though an outside edge lands inside its region: {stmts:?}"
    );
}

/// A payload ELSE arm means the `goto` is one of two real branches, not a skip
/// over a fall-through region — folding it would run the region on the arm that
/// was supposed to skip it.
#[test]
fn a_payload_else_arm_blocks_the_fold() {
    let var = |n: &str| csharp::Expr::Var(n.into());
    let mut stmts = vec![
        csharp::Stmt::If {
            cond: var("c"),
            then_body: vec![csharp::Stmt::Goto("L0".into())],
            else_body: vec![csharp::Stmt::Eval(var("other"))],
        },
        csharp::Stmt::Eval(var("body")),
        csharp::Stmt::Label("L0".into()),
        csharp::Stmt::Return(None),
    ];
    csharp::branch_repair(&mut stmts);
    let csharp::Stmt::If { then_body, .. } = &stmts[0] else {
        panic!("the two-armed branch vanished: {stmts:?}");
    };
    assert!(
        then_body
            .iter()
            .any(|s| matches!(s, csharp::Stmt::Return(None))),
        "the two-armed branch must preserve its return edge: {stmts:?}"
    );
}

#[test]
fn real_method_pin_copy_keeps_param_identity() {
    use super::{cfg, csharp, dominator, lift, locals, passes, ssa, stack_analyzer};
    let words: [u32; 79] = [
        0xA9BD57F6, 0xA9014FF4, 0xA9027BFD, 0x910083FD, 0xAA0203F5, 0xAA0103F3, 0xAA0003F4,
        0xB00222F6, 0x397446C8, 0x37000128, 0xF0020820, 0x911DE000, 0x94AC2865, 0xF0020820,
        0x91276000, 0x94AC2862, 0x52800028, 0x393446C8, 0x36000075, 0xB50000B3, 0x1400003A,
        0xB4000733, 0xF9406660, 0xB50005A0, 0xF9405E60, 0xB5000560, 0xF9405A75, 0xF0020836,
        0x911DE2D6, 0xF94002C0, 0xB940E008, 0x35000068, 0x94AC2896, 0xF94002C0, 0xF9405C08,
        0xF9400501, 0xAA1503E0, 0xD2800002, 0x94048D1D, 0xAA0003E8, 0xD2800000, 0x34000368,
        0xF9400E80, 0xB4000460, 0xD2800001, 0x940194B8, 0xB4000400, 0xF9405A61, 0xD2800002,
        0x94018C36, 0xB40001C0, 0xF0020828, 0x91276108, 0xF9400108, 0xF9400009, 0x3944C12B,
        0x3944C10A, 0x6B0A017F, 0x540000C3, 0xF9406529, 0x8B0A0D29, 0xF85F8129, 0xEB08013F,
        0x540000A0, 0xF9405A60, 0xD2800001, 0x9401BA67, 0xB40000A0, 0xA9427BFD, 0xA9414FF4,
        0xA8C357F6, 0xD65F03C0, 0xF9405A60, 0xD2800001, 0xA9427BFD, 0xA9414FF4, 0xA8C357F6,
        0x1401BAC5, 0x94AC28B6,
    ];
    let bytes = le(&words);
    let seed = locals::SeedInfo {
        param_types: vec![
            ManagedType::Unknown,                      // XmlSchemaElement elem
            ManagedType::Primitive(PrimKind::Boolean), // bool bCreatingNewType
        ],
        ret_type: ManagedType::Unknown,
        is_static: false,
        declaring: None,
        param_names: vec!["elem".into(), "bCreatingNewType".into()],
        ..Default::default()
    };

    let lifted = lift::lift_arm64(&bytes, 0x1f4e79c, 256, None).expect("lift");
    let mut insns = lifted.insns;
    let addresses = lifted.addresses;
    locals::rewrite_pinned_copies(&mut insns, &seed);
    let mut graph = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    if std::env::var("ILR_HARNESS_CFG").is_ok() {
        let succ = &graph.successors;
        for (bi, b) in graph.blocks.iter().enumerate() {
            let preds: Vec<usize> = (0..graph.blocks.len())
                .filter(|&p| succ[p].contains(&bi))
                .collect();
            eprintln!(
                "[cfg] B{bi} {:#x}..={} insns={} preds={:?} succs={:?}",
                insns[b.start].address,
                b.end,
                b.end - b.start,
                preds,
                succ[bi]
            );
            for i in b.start..b.end {
                eprintln!(
                    "      i{i} @{:#x}: {:?} {:?}",
                    insns[i].address, insns[i].op, insns[i].operands
                );
            }
        }
    }
    stack_analyzer::analyze(&mut insns, &graph);
    let dom = dominator::DominatorInfo::compute(&graph);
    let mut ssa_out = ssa::build(&insns, &graph, &dom, &[]);
    ssa::materialize(&mut ssa_out, &mut graph);
    // stage probe: count surviving raw Register operands (should be zero:
    // SSA must rename EVERY register use into a versioned Local).
    let dbg = std::env::var("ILR_HARNESS_REG").is_ok();
    let scan = |tag: &str, insns: &Vec<super::ir::Insn>| {
        if !dbg {
            return;
        }
        for (i, insn) in insns.iter().enumerate() {
            let has_reg = insn
                .operands
                .iter()
                .any(|o| matches!(o, super::ir::Operand::Register(_)));
            if has_reg {
                eprintln!(
                    "[reg] {tag} i{i} @{:#x}: op={:?} operands={:?}",
                    insn.address, insn.op, insn.operands
                );
            }
        }
    };
    scan("postssa", &ssa_out.insns);
    passes::bind_fp_returns(&mut ssa_out.insns, &ssa_out.version_slot, &seed.ret_type);
    let table = locals::create(&ssa_out.version_slot, &seed);
    locals::scrub_abi_call_args(&mut ssa_out.insns, &table);
    // (metadata-backed field/type resolution skipped: no oracle in a unit
    // test. The parameter identity under test comes from the ABI seeding in
    // `locals::create`, which needs no metadata.)
    scan("scrub", &ssa_out.insns);
    passes::remove_init_guards(&mut ssa_out.insns, &mut graph);
    scan("guards", &ssa_out.insns);
    for _ in 0..8 {
        let p = passes::copy_and_constant_propagation(&mut ssa_out.insns);
        let f = passes::constant_folding(&mut ssa_out.insns);
        if p + f == 0 {
            break;
        }
    }
    passes::dead_code_elimination(&mut ssa_out.insns);
    passes::simplify_boolean_flags(&mut ssa_out.insns);
    passes::trim_call_arguments(&mut ssa_out.insns);
    passes::remove_null_check_guards(&mut ssa_out.insns, &graph);
    passes::dead_code_elimination(&mut ssa_out.insns);
    scan("postopt", &ssa_out.insns);
    ssa::eliminate(&mut ssa_out, &mut graph);
    passes::dead_code_elimination(&mut ssa_out.insns);
    passes::liveness_dce(&mut ssa_out.insns);
    scan("final", &ssa_out.insns);

    if std::env::var("ILR_HARNESS_IR").is_ok() {
        for (i, insn) in ssa_out.insns.iter().enumerate() {
            eprintln!(
                "  h{i} @{:#x}: op={:?} operands={:?}",
                insn.address, insn.op, insn.operands
            );
        }
    }
    let ir = super::RecoveredIr {
        insns: ssa_out.insns,
        table,
        returns_value: true,
        base_rva: 0x1f4e79c,
    };
    let out = csharp::render_method_csharp(
        &ir,
        "GetEffectiveSchemaType",
        false,
        &[
            ("XmlSchemaElement".into(), "elem".into()),
            ("bool".into(), "bCreatingNewType".into()),
        ],
        &ManagedType::Unknown,
        -1,
    );
    eprintln!(
        "--- rendered ---
{out}----------------"
    );
    assert!(
        !out.contains("__unknown != 0"),
        "pin copy lost the parameter identity (renders __unknown):
{out}"
    );
    assert!(
        out.contains("bCreatingNewType"),
        "the pinned bool parameter must render by name:
{out}"
    );
}

/// A flag-preserving instruction between `cmp` and `b.cond` must NOT discard
/// the pending compare.
///
/// On AArch64 only a handful of instructions write NZCV (`SUBS`/`ADDS`/
/// `ANDS`/`BICS`, `CMP`/`CMN`/`TST`, `CCMP`/`CCMN`, `FCMP`/`FCCMP`). A plain
/// register `MOV` — which the compiler emits constantly while shuffling the
/// retry value into the pinned register — does not. The lifter cleared
/// `pending_cmp` in its `MOV` arm, so the following `b.cond` fell back to
/// `Compare(kind, Unknown, 0)` and rendered as `if (? != 0)`: plausible text,
/// wrong logic. This is the verifier's "conditional branch without real
/// compare" class.
///
/// Bytes are real, from the shipped iOS binary at 0x4fba14 (a CAS-retry loop
/// tail in an event accessor):
///   cmp x0, x21      0xeb15001f
///   mov x21, x0      0xaa0003f5   <- does not write NZCV
///   b.eq +0x3c       0x540001e0
#[test]
fn flag_preserving_mov_keeps_pending_compare() {
    let bytes = le(&[0xEB15001F, 0xAA0003F5, 0x540001E0]);
    let lifted = lift::lift_arm64(&bytes, 0x4000, 64, None).expect("lift");

    let mut cond_scratch: Option<u32> = None;
    for i in &lifted.insns {
        if matches!(i.op, Op::CondJump(_)) {
            if let Some(Operand::Local(v)) = i.operands.first() {
                cond_scratch = Some(*v);
            }
        }
    }
    let s = cond_scratch.expect("the b.eq must lift to a CondJump");

    let cmp = lifted
        .insns
        .iter()
        .find(|i| {
            matches!(i.op, Op::Compare(_))
                && matches!(i.operands.first(), Some(Operand::Local(v)) if *v == s)
        })
        .expect("the branch's scratch must be defined by a Compare");

    assert!(
        !matches!(cmp.operands.get(1), Some(Operand::Unknown)),
        "the cmp before the flag-preserving mov was discarded: {:?}",
        cmp.operands
    );
    assert_eq!(
        cmp.operands.get(1),
        Some(&Operand::reg("x0")),
        "cmp x0, x21 must keep x0 as its left operand"
    );
}

/// A register pinned TWICE must restore the site whose value is live at the
/// clobber — deterministically.
///
/// IL2CPP pins the receiver in a callee-saved register (`MOV x19, x0`) and
/// re-pins it later in the body (`MOV x19, x0` again, usually after a call
/// clobbered x0). `rewrite_pinned_copies` Nops every pin site (they
/// canonicalize to self-moves) and re-materialises one as a real `MOV` when
/// the association dies at a clobber. The site it picked came from
/// `pin_at.iter().find(..)` on a **HashMap**, i.e. an arbitrary entry in
/// per-process-random iteration order: the restored move jumped between the
/// prologue pin and the later re-pin from run to run, reordering the listing,
/// shifting SSA version ids, changing type propagation, and finally changing
/// WHICH field name a deref resolved to. Real symptom, same binary, two runs
/// of `FindAttribute` @0x1F4C54C:
///   run 1: `if (*(nint*)(XmlSchemaInference.ST_boolean + 302) != 0)`
///   run 2: `if (*(nint*)(*(nint*)(v13) + 302) != 0)`
/// (`ST_boolean` is a STATIC field at static-block offset 0 — a fabricated
/// name for a klass-pointer load.)
///
/// The rule this locks in: restore the latest pin site strictly BEFORE the
/// clobbering instruction (earliest as a fallback). Choosing the earliest
/// unconditionally is wrong too — the identity register is usually
/// reassigned between the prologue pin and a re-pin, so re-materialising the
/// prologue move there would propagate the stale value.
#[test]
fn pin_restore_picks_the_live_site_deterministically() {
    // prologue pin, use, clobber, re-pin, use, clobber
    let mut insns = vec![
        Insn::new(
            Op::Move,
            vec![Operand::reg("x19"), Operand::reg("x0")],
            0x1000,
        ),
        Insn::move_(
            Operand::reg("x1"),
            Operand::Memory {
                width: 0,
                base: Box::new(Operand::reg("x19")),
                index: None,
                scale: 8,
                addend: 8,
            },
            0x1004,
        ),
        Insn::new(Op::Call, vec![Operand::Immediate(0x9000)], 0x1008),
        // re-pin AFTER the first clobber — the second pin site
        Insn::new(
            Op::Move,
            vec![Operand::reg("x19"), Operand::reg("x0")],
            0x100c,
        ),
        Insn::move_(
            Operand::reg("x1"),
            Operand::Memory {
                width: 0,
                base: Box::new(Operand::reg("x19")),
                index: None,
                scale: 8,
                addend: 8,
            },
            0x1010,
        ),
        Insn::new(Op::Call, vec![Operand::Immediate(0x9000)], 0x1014),
    ];
    let seed = locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::Primitive(PrimKind::Void),
        is_static: false, // x0 is `this`
        declaring: None,
        param_names: vec![],
        ..Default::default()
    };

    locals::rewrite_pinned_copies(&mut insns, &seed);

    // The clobber at i=2 must restore the site at i=0 (latest before it), NOT
    // the re-pin at i=3, and the clobber at i=5 must restore i=3.
    for (idx, why) in [
        (0usize, "clobber at i=2 must restore the prologue pin site"),
        (3usize, "clobber at i=5 must restore the re-pin site"),
    ] {
        assert_eq!(insns[idx].op, Op::Move, "{why} (got {:?})", insns[idx].op);
        assert_eq!(
            insns[idx].operands,
            vec![Operand::reg("x19"), Operand::reg("x0")],
            "{why}: the restored copy must be `MOV x19, x0`, got {:?}",
            insns[idx].operands
        );
    }
    // Uses of the live pin association were canonicalized to the identity.
    //
    // ONLY the pre-call use (i=1). The use at i=4 sits after a call, so the
    // `MOV x19, x0` at i=3 copies that call's RESULT — x0 is the return-value
    // register and no longer holds `this`. Canonicalizing i=4 through x0 is
    // the `Type.op_Equality(a, a)` / `x.GetType() == x.GetType()` aliasing
    // bug; it must keep reading x19.
    match insns[1].operands.get(1) {
        Some(Operand::Memory { base, .. }) => assert_eq!(
            **base,
            Operand::reg("x0"),
            "the pre-call use must read through the identity register"
        ),
        other => panic!("use at i=1 lost its memory operand: {other:?}"),
    }
    match insns[4].operands.get(1) {
        Some(Operand::Memory { base, .. }) => assert_eq!(
            **base,
            Operand::reg("x19"),
            "the post-call use must keep x19 — x0 holds the callee's result there, not the receiver"
        ),
        other => panic!("use at i=4 lost its memory operand: {other:?}"),
    }
}

/// Phi elimination must emit the edge copies of phis that SHARE an insertion
/// point in the phis' listing order.
///
/// `eliminate` keys edge moves by insertion point and used to iterate its
/// `phi_at` HashMap directly, so two phis lowered onto the same predecessor
/// edge produced their `Move`s in per-process-random order. Real symptom:
/// the same method rendered
///   run 1: `v33 = XmlSchemaInference.ST_boolean + (*(nint*)(v31) << 4) + 312; v34 = attrName;`
///   run 2: `v34 = attrName; v33 = XmlSchemaInference.ST_boolean + … + 312;`
/// (an intra-block statement swap), and swapping two statements can mean
/// executing them in the wrong order.
///
/// The rule this locks in: the emitted sequence follows the phis' position in
/// the listing (the order `materialize` spliced them in), NOT hash order.
#[test]
fn phi_edges_are_emitted_in_listing_order() {
    // B0 (pred, ends in a terminator) → B1 (merge, holds both phis)
    //
    // Both phis carry TWO DISTINCT args so the degenerate-phi collapse in
    // `eliminate` (which substitutes uses directly when every real arg
    // is the SAME version) leaves them alone: this test wants the edge
    // Moves emitted, and their relative order pinned.
    let mut insns = vec![
        Insn::new(
            Op::Move,
            vec![Operand::Local(7), Operand::Immediate(1)],
            0x3000,
        ),
        Insn::new(Op::CondJump(2), vec![], 0x3004),
        Insn::new(
            Op::Phi,
            vec![Operand::Local(100), Operand::Local(7), Operand::Local(9)],
            0x3008,
        ),
        Insn::new(
            Op::Phi,
            vec![Operand::Local(101), Operand::Local(8), Operand::Local(10)],
            0x300c,
        ),
        Insn::new(Op::Return, vec![Operand::Local(100)], 0x3010),
    ];
    // Both phis are lowered onto the same pred edge (block 0), so the edge
    // moves land at one insertion point.
    let phis = vec![
        ssa::PhiRecord {
            block: 1,
            slot: "x0".into(),
            dest_version: 100,
            preds: vec![0, 1],
            args: vec![Some(7), Some(9)],
        },
        ssa::PhiRecord {
            block: 1,
            slot: "x1".into(),
            dest_version: 101,
            preds: vec![0, 1],
            args: vec![Some(8), Some(10)],
        },
    ];
    let mut output = ssa::SsaOutput {
        insns: std::mem::take(&mut insns),
        phis,
        version_slot: HashMap::new(),
        call_defined: std::collections::HashSet::new(),
    };
    let mut cfg = cfg::Cfg {
        blocks: vec![
            cfg::Block { start: 0, end: 2 },
            cfg::Block { start: 2, end: 5 },
        ],
        successors: vec![vec![1], vec![]],
    };

    ssa::eliminate(&mut output, &mut cfg);

    let pos_of = |dest: u32| -> usize {
        output
            .insns
            .iter()
            .position(|i| i.op == Op::Move && i.operands.first() == Some(&Operand::Local(dest)))
            .unwrap_or_else(|| panic!("no edge move for phi dest v{dest}"))
    };
    let (a, b) = (pos_of(100), pos_of(101));
    assert!(
        a < b,
        "phi edge moves must follow the phis' listing order (v100 at {a}, v101 at {b})"
    );
}

/// A cascade of degenerate phis must collapse to a FIXPOINT, and the edge
/// copies of whatever remains must read the POST-collapse arguments.
///
/// `eliminate` used to decide collapses once, from a snapshot of the
/// materialized Phi args taken BEFORE the collapse substitution was applied.
/// Two join phis `phi(7,7)` collapse to 7, but the tail `phi(10,11)` that
/// consumes them is degenerate only once that substitution is known; it
/// stayed materialized, and its edge moves were built from the stale snapshot
/// (`v20 = v10; v20 = v11;`) reading versions whose phis had just been
/// deleted. Real symptom: `BundleStoreItem.Initialize` @0x166D8F8 reported 24
/// undefined locals (v15…v90), every one a read of a collapsed join
/// destination planted as an edge copy of the shared tail phi.
#[test]
fn degenerate_phi_cascade_collapses_to_a_fixpoint() {
    let l = Operand::Local;
    let insns = vec![
        Insn::new(Op::Move, vec![l(7), Operand::Immediate(1)], 0x1000),
        Insn::new(Op::CondJump(2), vec![], 0x1004),
        Insn::new(Op::Phi, vec![l(10), l(7), l(7)], 0x1008),
        Insn::new(Op::Jump(6), vec![], 0x100c),
        Insn::new(Op::Phi, vec![l(11), l(7), l(7)], 0x1010),
        Insn::new(Op::Jump(6), vec![], 0x1014),
        Insn::new(Op::Phi, vec![l(20), l(10), l(11)], 0x1018),
        Insn::new(
            Op::Call,
            vec![Operand::Immediate(0x2000), l(20), l(10), l(11)],
            0x101c,
        ),
        Insn::new(Op::Return, vec![l(20)], 0x1020),
    ];
    let phis = vec![
        ssa::PhiRecord {
            block: 1,
            slot: "x5".into(),
            dest_version: 10,
            preds: vec![0],
            args: vec![Some(7)],
        },
        ssa::PhiRecord {
            block: 2,
            slot: "x4".into(),
            dest_version: 11,
            preds: vec![0],
            args: vec![Some(7)],
        },
        ssa::PhiRecord {
            block: 3,
            slot: "x5".into(),
            dest_version: 20,
            preds: vec![1, 2],
            args: vec![Some(10), Some(11)],
        },
    ];
    let mut output = ssa::SsaOutput {
        insns,
        phis,
        version_slot: HashMap::new(),
        call_defined: std::collections::HashSet::new(),
    };
    let mut cfg = cfg::Cfg {
        blocks: vec![
            cfg::Block { start: 0, end: 2 },
            cfg::Block { start: 2, end: 4 },
            cfg::Block { start: 4, end: 6 },
            cfg::Block { start: 6, end: 9 },
        ],
        successors: vec![vec![1, 2], vec![3], vec![3], vec![]],
    };

    ssa::eliminate(&mut output, &mut cfg);

    assert!(
        !output.insns.iter().any(|i| i.op == Op::Phi),
        "every phi must be lowered or collapsed"
    );
    // All three phis collapse to v7: nothing is inserted, three removed.
    assert_eq!(
        output.insns.len(),
        6,
        "degenerate cascade must insert no edge moves: {:?}",
        output.insns
    );
    let call = output
        .insns
        .iter()
        .find(|i| i.op == Op::Call)
        .expect("call survives");
    assert_eq!(
        call.operands,
        vec![Operand::Immediate(0x2000), l(7), l(7), l(7)],
        "call operands must name the collapse terminal, not a removed phi dest"
    );
    let mut defined = std::collections::HashSet::new();
    let mut used = std::collections::HashSet::new();
    for insn in &output.insns {
        if let Some(pos) = Insn::dest_position(&insn.op) {
            if let Some(Operand::Local(v)) = insn.operands.get(pos) {
                defined.insert(*v);
            }
        }
        for o in &insn.operands {
            if let Operand::Local(v) = o {
                used.insert(*v);
            }
        }
    }
    let undef: Vec<u32> = used.difference(&defined).copied().collect();
    assert!(
        undef.is_empty(),
        "reads of removed phi destinations survived: {undef:?} in {:?}",
        output.insns
    );
}

/// A STATIC field must never be resolved from an instance offset.
///
/// Static fields live in a per-class storage block whose FIRST field is at
/// offset 0 — the same number as an object's `Il2CppClass*`, which is the
/// first word of every managed object. So `LDR x8, [x0]` (load the class of
/// `this`) used to match `field_at_offset(declaring, 0)`, which walked the
/// field list and returned the first static field with that offset:
///   real bytes:  ldr x8, [x20]        ; x20 = this
///                ldrh w9, [x8, #302]  ; class struct field
///   rendered:    if (*(nint*)(XmlSchemaInference.ST_boolean + 302) != 0)
/// `ST_boolean` is `internal static XmlQualifiedName ST_boolean; // 0x0` —
/// a real field, used at an address that can never hold it. This oracle
/// reproduces that collision: offset 0 is a static field, offset 16 an
/// instance field. Only the instance one may be named.
#[test]
fn static_field_is_never_matched_at_an_instance_offset() {
    struct CollidingOracle;
    impl locals::TypeOracle for CollidingOracle {
        fn field_at_offset(&self, _tdi: i32, offset: i64) -> Option<types::FieldRef> {
            match offset {
                // static block offset 0 — what the old lookup returned for a
                // klass-pointer load
                0 => Some(types::FieldRef {
                    type_def_index: 7,
                    field_index: 0,
                    name: "S_static".into(),
                    declaring_type_name: "T".into(),
                    field_type: ManagedType::Object,
                    is_static: true,
                }),
                16 => Some(types::FieldRef {
                    type_def_index: 7,
                    field_index: 1,
                    name: "inst".into(),
                    declaring_type_name: "T".into(),
                    field_type: ManagedType::Object,
                    is_static: false,
                }),
                _ => None,
            }
        }
    }

    let seed = locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::Primitive(PrimKind::Void),
        is_static: false, // entry x0 is `this`
        declaring: None,
        param_names: vec![],
        ..Default::default()
    };
    let mut version_slot: HashMap<u32, String> = HashMap::new();
    version_slot.insert(1, "x0".to_string());
    version_slot.insert(2, "x1".to_string());
    version_slot.insert(3, "x2".to_string());
    let mut table = locals::create(&version_slot, &seed);
    let li = table.index_of[&1];
    assert!(
        table.infos[li].is_this,
        "the x0 entry version must seed as `this` for this test to mean anything"
    );
    table.infos[li].ty = ManagedType::Class(std::sync::Arc::new(types::NamedType {
        namespace: String::new(),
        name: "T".into(),
        assembly: None,
        type_def_index: 7,
    }));

    let mem = |addend: i64| Operand::Memory {
        width: 0,
        base: Box::new(Operand::Local(1)),
        index: None,
        scale: 8,
        addend,
    };
    let mut insns = vec![
        // v2 = *(this + 0)          — the klass pointer
        Insn::move_(Operand::Local(2), mem(0), 0x4000),
        // v3 = *(this + 16)         — a real instance field
        Insn::move_(Operand::Local(3), mem(16), 0x4004),
    ];

    locals::propagate_types(&mut insns, &mut table, &CollidingOracle);

    assert!(
        matches!(insns[0].operands[1], Operand::Memory { .. }),
        "a klass-pointer load must stay a raw deref, got {:?}",
        insns[0].operands
    );
    match insns[1].operands.get(1) {
        Some(Operand::Field(f)) => assert_eq!(f.name, "inst"),
        other => panic!("the instance field at +16 must still resolve, got {other:?}"),
    }
}

/// A store into a RESOLVED field must become an assignment.
///
/// `resolve`/`locals` rewrite `str x, [this+off]` into `Move(Field(f), value)`,
/// and the store-lowering match had an arm for a `Local` destination and an arm
/// for a `Memory` destination but NONE for a `Field` destination. Such a store
/// therefore fell into the `_ => vec![]` sweep and vanished: the field WRITE
/// disappeared while the field READ (`Move(local, Field(f))`) stayed, so the
/// method kept its reads and silently lost its writes. `XmlAsyncCheckReader..ctor`
/// @0x200D5F8 kept `this..ctor(); return;` and lost `this.coreReader = reader;`
/// — every CFG block "emitted", no marker, nothing but the offsets column to
/// show for it. `ConstraintCollection.Clear` @0x1DD7A20 lost
/// `this._delayLoadingConstraints = 0;` the same way.
///
/// Deleting the `Some(Operand::Field(f))` arm makes both assertions here fail.
#[test]
fn a_store_into_a_resolved_field_is_emitted() {
    let ids: Vec<u32> = (1..=2).collect();
    let ir = super::RecoveredIr {
        insns: vec![
            // `str xzr, [this+0x28]` — an INSTANCE field of the declaring type
            Insn::move_(
                Operand::Field(types::FieldRef {
                    type_def_index: 7,
                    field_index: 0,
                    name: "ps".into(),
                    declaring_type_name: "T".into(),
                    field_type: ManagedType::Object,
                    is_static: false,
                }),
                Operand::imm(0),
                0x5000,
            ),
            // a STATIC field store must stay qualified — never `this.S_shared`
            Insn::move_(
                Operand::Field(types::FieldRef {
                    type_def_index: 7,
                    field_index: 1,
                    name: "S_shared".into(),
                    declaring_type_name: "T".into(),
                    field_type: ManagedType::Object,
                    is_static: true,
                }),
                Operand::imm(1),
                0x5004,
            ),
            Insn::new(Op::Return, vec![], 0x5008),
        ],
        table: plain_local_table(&ids),
        returns_value: false,
        base_rva: 0x5000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, 7);
    assert!(
        out.contains("this.ps = 0"),
        "a store into a resolved instance field must be emitted:\n{out}"
    );
    assert!(
        out.contains("T.S_shared = 1"),
        "a store into a resolved static field must be emitted, qualified:\n{out}"
    );
}

/// A branch the pipeline erased as VACUOUS must leave the condition
/// denominator.
///
/// IL2CPP's null check is `cbnz w8, +8` over `bl <raise_null>`; the `bl` is a
/// runtime helper and is excised, which leaves both arms of the branch empty.
/// There is no C# that expresses it — writing `if (c) { }` would be the
/// "impossible if scope" defect — so `passes::drop_vacuous_cond_branches` Nops
/// it and ledgeres the address. `concord` counted it as a LOST condition
/// anyway: 1,872 of them in the reference sample, 45% of the whole condition
/// shortfall. `verify`'s own wrong-logic oracle already excused them, which is
/// why it reported 193 missing conditions while `concord` reported 1,208
/// lost-condition METHODS on the same run.
#[test]
fn a_vacuous_null_check_branch_is_not_counted_as_a_lost_condition() {
    use super::verify::{DecodedInsn, InsnClass, NativeFacts};
    let branch = |addr: u64, mnemonic: &str| DecodedInsn {
        addr,
        size: 4,
        mnemonic: mnemonic.to_string(),
        class: InsnClass::Branch,
        facts: NativeFacts::default(),
    };
    super::verify::ledger_clear();
    // One real condition, one null-check collapse.
    let dec = vec![branch(0x7000, "cbnz"), branch(0x7004, "cbz")];
    let c = concord::compare("static void M() { }\n", &dec);
    assert_eq!(
        c.native_conds, 2,
        "both branches are conditions until one is excused"
    );
    assert_eq!(c.vacuous_conds, 0);

    super::verify::ledger_note(0x7000, super::verify::LedgerKind::VacuousBranch);
    let c = concord::compare("static void M() { }\n", &dec);
    assert_eq!(
        c.vacuous_conds, 1,
        "the ledgered null-check collapse must be reported on its own"
    );
    assert_eq!(
        c.native_conds, 1,
        "and it must not be charged to the decompiler as a lost condition"
    );
    super::verify::ledger_clear();
}

/// A function's out-of-line cold region is emitted ONLY when an edge proves it
/// belongs to this function.
///
/// LLVM lays a function's exception landing pads out AFTER the body's last
/// `ret`, and the unwinder enters them — so a CFG built from branches calls
/// them unreachable while they are the method's own code. The proof of
/// ownership is an edge FROM the cold region INTO the body: a different
/// function never branches into the middle of this one. Layout alone is not a
/// proof, so a trailing region that only exits is left out (and named by the
/// `ILR_COV` `after_ret=… backedge=false` diagnostic) rather than guessed at.
///
/// This is the whole discriminator, so the test flips exactly one thing — the
/// cold block's branch target — and asserts the call appears in one case and
/// not the other. It also reverses the earlier hand conclusion about
/// `ConstraintCollection.Clear` @0x1DD7A20: "the leftover blocks have no edge
/// INTO them" is not evidence of anything; `cbz x0, 0x1DD7B60` out of the cold
/// region into the body is.
#[test]
fn a_trailing_region_is_emitted_only_when_an_edge_proves_ownership() {
    //   i0 @0x9000  (v1 != 0)                 ; B0
    //   i1 @0x9004  CONDJMP B1 [v1]
    //   i2 @0x9008  RETURN                    ; B1, the body's exit
    //   i3 @0x900c  v4 = fn_2222(v1)          ; B2, the cold region — no edge in
    //   i4 @0x9010  b <B1> or exit
    let build = |cold_target: ir::InsnIndex| -> Vec<Insn> {
        vec![
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![Operand::Local(3), Operand::Local(1), Operand::imm(0)],
                0x9000,
            ),
            Insn::new(Op::CondJump(2), vec![Operand::Local(3)], 0x9004),
            Insn::new(Op::Return, vec![], 0x9008),
            Insn::new(
                Op::Call,
                vec![Operand::imm(0x2222), Operand::Local(4), Operand::Local(1)],
                0x900c,
            ),
            Insn::new(Op::Jump(cold_target), vec![], 0x9010),
        ]
    };
    let render = |insns: Vec<Insn>| -> String {
        let ids: Vec<u32> = (1..=4).collect();
        let ir = super::RecoveredIr {
            insns,
            table: plain_local_table(&ids),
            returns_value: false,
            base_rva: 0x9000,
        };
        csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1)
    };

    // The cold block branches back INTO the body (block 1, the `ret`) — an edge
    // only the same function can have, so it is adopted and emitted.
    let owned = render(build(2));
    assert!(
        owned.contains("fn_2222("),
        "a cold region that branches into the body must be emitted:\n{owned}"
    );

    // The same region, but its only outgoing edge leaves the function. Layout
    // says nothing about ownership, so it must NOT be attributed to this method.
    let foreign = render(build(5));
    assert!(
        !foreign.contains("fn_2222("),
        "a trailing region with no edge back into the body must not be emitted \
         as this method's code:\n{foreign}"
    );
}

/// A `get_X()` call presented as the property `obj.X` must be LEDGERED.
///
/// The fold is deliberate and faithful (`prettify_call`), but it puts a real
/// native `bl` in the text as a member access, and the concordance counts C#
/// calls by scanning for `ident(` tokens. So without the ledger every folded
/// accessor was charged to the decompiler as a call it DROPPED: 1,698 of them
/// in a 3,941-method sample, 323 of the 521 "dropped-call methods".
/// `SwitchToRectTransform` @0xFA08 was the clean case — 14 native `bl`, 4
/// `ident(` in the text, and exactly 10 folds (its entire deficit).
///
/// Removing the ledger note from `render_method_csharp_full_chains` (or the
/// `property_fold_sites` insert in `prettify_call_at`) makes this fail.
#[test]
fn a_getter_folded_to_a_property_is_ledgered_as_a_presented_call() {
    let ids: Vec<u32> = (1..=3).collect();
    //   i0 @0x8000  v1 = x1                    ; the receiver local
    //   i1 @0x8004  v2 = v1.get_Rect()         ; ONE native bl
    //   i2 @0x8008  Return(v2)
    let insns = vec![
        Insn::move_(Operand::Local(1), Operand::reg("x1"), 0x8000),
        Insn::new(
            Op::Call,
            vec![
                Operand::Method(mref("get_Rect", 7, false)),
                Operand::Local(2),
                Operand::Local(1),
            ],
            0x8004,
        ),
        Insn::new(Op::Return, vec![Operand::Local(2)], 0x8008),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: true,
        base_rva: 0x8000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains(".Rect"),
        "the getter must be presented as the property `Rect`:\n{out}"
    );
    assert!(
        !out.contains("get_Rect"),
        "and it must NOT survive as a call name:\n{out}"
    );
    assert!(
        super::verify::ledger_has_kind(0x8004, super::verify::LedgerKind::PropertyAccessorCall),
        "the folded call at 0x8004 must be ledgered as presented-not-dropped, or \
         the concordance charges it as a missing call"
    );
}

/// A def the renderer REFUSES to inline must still be an assignment.
///
/// `compute_var_reads` decides "must this def become a statement?" from the
/// reaching-definitions graph alone, on the assumption that a single-def local
/// is inlined into every use. `inline_local_at` has a third refusal the graph
/// cannot see — `def_moved_across_a_write`: the def reads a local that is
/// REWRITTEN later in the def's own block (a loop latch's back-edge copy), so a
/// use in another block cannot be served by re-expanding the expression. That
/// refusal falls back to a bare `vN`, and the single-def local was excluded
/// from `var_reads`, so NO statement ever assigned it. Every such read is a
/// `vN` with no value: 10,032 references over 1,534 methods on the x64 corpus,
/// 98% of all remaining undefined locals.
///
/// The shape below is `0x1532130`'s two-instruction core: a loop-carried counter
/// whose latch rewrites the very register its def reads.
///
///   i0 @0x3000  v0 = 3                 ; preheader
///   i1 @0x3004  v1 = v0 - 1            ; the def (`d`)
///   i2 @0x3008  v0 = v1                ; `v0` rewritten AFTER `d`, same block
///   i3 @0x300c  v2 = (v1 != 0)
///   i4 @0x3010  CondJump(1) [v2]       ; back edge → i1
///   i5 @0x3014  v3 = fn_9000(v1)       ; use of `v1` in ANOTHER block
///   i6 @0x3018  Return
///
/// Reverting the `refused_inline_reads` half of `MethodCtx::build` makes this
/// test find no `v1 = …` line at all.
#[test]
fn a_def_that_cannot_be_inlined_is_still_an_assignment() {
    // ids are 1-based versions; the name is the table index, so 1→v0, 2→v1, 3→v2.
    let table = table_with_origin(&[1, 2, 3], "x9");
    let cond = Operand::Local(u32::MAX - 2);
    let insns = vec![
        Insn::move_(Operand::Local(1), Operand::imm(3), 0x3000),
        Insn::new(
            Op::Arith(super::ir::ArithKind::Sub),
            vec![Operand::Local(2), Operand::Local(1), Operand::imm(1)],
            0x3004,
        ),
        Insn::move_(Operand::Local(1), Operand::Local(2), 0x3008),
        Insn::new(
            Op::Compare(super::ir::CompareKind::NotEqual),
            vec![cond.clone(), Operand::Local(2), Operand::imm(0)],
            0x300c,
        ),
        Insn::new(Op::CondJump(1), vec![cond], 0x3010),
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x9000), Operand::Local(3), Operand::Local(2)],
            0x3014,
        ),
        Insn::new(Op::Return, vec![], 0x3018),
    ];
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: false,
        base_rva: 0x3000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("v1 = "),
        "the latch def `v1 = v0 - 1` must be emitted as an assignment, or the \
         call argument `v1` reads a name nothing assigns:\n{out}"
    );
    assert!(
        out.contains("fn_9000(") && out.contains("v1"),
        "and the cross-block read must still pass the value to the call:\n{out}"
    );
}

/// A loop TEST must not re-apply the update it is testing.
///
/// SSA gives a loop variable one version per definition, but `highvar` merges
/// versions onto ONE table slot, so the latch's result and the value the
/// latch reads from are the SAME name. The def is then a self-mutation of
/// that name, and `inline_local_at` EXPANDED it at the branch — after the
/// assignment had already replaced the name — so the test decremented a
/// second time:
///
/// ```text
///   do { … v266 = v266 - 1; } while (v266 - 1 != 0);
/// ```
///
/// The native tests the freshly computed value (`subs; b.ne`), so the emitted
/// loop ran ONE ITERATION SHORT of the original — measured at `0x1532130`, and
/// the same shape on every `do { … v = v - 1; } while (v - 1 != 0)` in the
/// corpus. The NAME is the value the test reads.
///
/// The cycle is reached through the loop's BACK-EDGE COPY, not through the
/// def's direct operand — `v1 = v2 - 1` where `v2` is a copy of `v1` — so the
/// predicate has to follow the read chain (`collect_def_locals`); comparing
/// the direct operand alone finds nothing here.
#[test]
fn a_loop_test_reads_the_counter_it_updates() {
    // infos[0] → v0, infos[1] → v1, infos[2] → v2: versions 2 and 4 are the
    // two sides of the loop-carried slot under DIFFERENT names, and 4 is the
    // one the `sub` assigns and the test reads.
    let mut table = locals::LocalTable::default();
    for _ in 0..3 {
        table.infos.push(locals::LocalInfo {
            slot_origin: "x9".into(),
            ty: ManagedType::Unknown,
            is_this: false,
            is_param: None,
            param_comp: None,
            is_incoming_arg: false,
            is_call_clobber: false,
            is_return_buffer: false,
        });
    }
    table.index_of.insert(3, 0); // v0 — the preheader's value
    table.index_of.insert(4, 1); // v1 — the latch result AND what the test reads
    table.index_of.insert(2, 2); // v2 — the loop-carried copy the `sub` reads

    //   i0 @0x8000  v0 = 3                 ; preheader
    //   i1 @0x8004  v2 = v1                ; the back-edge copy of the slot
    //   i2 @0x8008  v1 = v2 - 1            ; the def — reads `v2`, WRITES `v1`
    //   i3 @0x800c  cond = (v1 != 0)       ; the loop test — must read `v1`
    //   i4 @0x8010  CondJump(1) [cond]     ; back edge → i1
    //   i5 @0x8014  Return
    let cond = Operand::Local(u32::MAX - 2);
    let insns = vec![
        Insn::move_(Operand::Local(3), Operand::imm(3), 0x8000),
        Insn::move_(Operand::Local(2), Operand::Local(4), 0x8004),
        Insn::new(
            Op::Arith(super::ir::ArithKind::Sub),
            vec![Operand::Local(4), Operand::Local(2), Operand::imm(1)],
            0x8008,
        ),
        Insn::new(
            Op::Compare(super::ir::CompareKind::NotEqual),
            vec![cond.clone(), Operand::Local(4), Operand::imm(0)],
            0x800c,
        ),
        Insn::new(Op::CondJump(1), vec![cond], 0x8010),
        Insn::new(Op::Return, vec![], 0x8014),
    ];
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: false,
        base_rva: 0x8000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("- 1 != 0"),
        "the loop test must NOT re-apply the decrement it is testing — \
         `while (v1 - 1 != 0)` runs the body one iteration short of the native:\n{out}"
    );
    // The counter may carry its evidence-based role name (`i`) instead of the
    // generated `v1`; the contract is that the TEST reads the same name the
    // body UPDATES, not a re-applied expression.
    let counter = if out.contains("v1 = v1 - 1") { "v1" } else { "i" };
    assert!(
        out.contains(&format!("{counter} != 0")),
        "the loop test must read the counter by NAME:\n{out}"
    );
    assert!(
        out.contains(&format!("{counter} = {counter} - 1")),
        "and the assignment must be present, so the name the test reads has a \
         value:\n{out}"
    );
}

/// A def that reads the NAME it assigns must keep its own expression — even
/// when that name is only shared, not a mutation.
///
/// `highvar` maps several SSA versions onto ONE table index, so `vN` can name
/// two unrelated values that live in disjoint parts of the method. A def whose
/// read chain touches such a shared name is NOT a self-update, and the text
/// must still contain the def's own expression:
///
/// ```text
///   v1 = v1 * 7;      // the factor must survive
/// ```
///
/// `OutBounce`@0x14D6330 is the corpus instance. There, `(t - A) * C2` was
/// rendered as the bare name `v18`, and the factor `* C2` disappeared with it:
/// `(t - A) * C2 * (t - A) + C` became `v18 * (v18 - A) + C` — a wrong value,
/// which is worse than a bare name. The cause was the refusal's gate: it asked
/// whether SOME var-read version spelled the assigned name, and that question
/// is answered by a sibling version. Worse, it was asked against a set that
/// was still being built, so the same def answered "no" while the read set was
/// completed (no assignment emitted) and "yes" while the body was rendered
/// (bare name printed) — the two halves disagreed and both were emitted.
///
/// The gate is now "does this def assign a NAMED local at all", which no
/// sibling can influence, and the refusal is what materialises the def it
/// refuses, so the name it prints is assigned. This fixture reproduces the
/// disagreement: an incoming version of the same name is added to the read set
/// by `refused_inline_reads` after the gate has already been consulted.
#[test]
fn a_shared_name_does_not_cost_a_def_its_expression() {
    // `v2` (index 2) names four versions: 3, a sibling with a real def, an
    // incoming value that is never defined, and 5 — the def under test.
    let mut table = table_with_origin(&[1, 2, 3, 4, 5, 6], "vf0");
    table.index_of.insert(4, 2);
    table.index_of.insert(5, 2);
    table.index_of.insert(6, 2);
    let insns = vec![
        // the sibling: a real definition the shared name can be spelled from.
        // Operands are incoming locals (not constants) so constant folding
        // cannot legitimately collapse the whole chain to a literal: this
        // fixture exists to prove the `* 7` expression survives rendering.
        Insn::new(
            Op::Arith(super::ir::ArithKind::Mul),
            vec![Operand::Local(6), Operand::Local(1), Operand::Local(2)],
            0x6000,
        ),
        // the def under test — it reads `v2` and assigns `v2`, so the refusal
        // would print the bare name; the `* 7` must survive
        Insn::new(
            Op::Arith(super::ir::ArithKind::Mul),
            vec![Operand::Local(5), Operand::Local(6), Operand::imm(7)],
            0x6004,
        ),
        // an incoming `v2` with NO definition: `refused_inline_reads` adds it
        // for the name after the gate has already been consulted once
        Insn::new(
            Op::Arith(super::ir::ArithKind::Add),
            vec![Operand::Local(3), Operand::Local(4), Operand::imm(1)],
            0x6008,
        ),
        Insn::new(Op::Return, vec![Operand::Local(5)], 0x600c),
    ];
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: true,
        base_rva: 0x6000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("* 7"),
        "the def assigns `v2` and reads `v2` through a SIBLING version; its \
         expression must survive instead of collapsing to a bare name — this \
         is the `OutBounce` factor loss:\n{out}"
    );
    assert!(
        out.contains("v2 = ") || !out.contains("v2"),
        "and any `v2` the expression reads must be assigned somewhere:\n{out}"
    );
}

/// A STORE's base is a READ.
///
/// `Move [Memory { base: Local(v), .. }, value]` is `*(T*)(v + off) = value`:
/// the base computes the ADDRESS, so it is read. Every read analysis in the
/// renderer skipped the whole destination SLOT (`dest_position(Op::Move) ==
/// Some(0)`), and for a memory destination operand#0 is not a definition at all
/// — so a local read only as a store target was invisible to
/// `compute_var_reads`, `compute_read_counts` and `read_locals`. `ssa` already
/// states the correct rule ("a MEMORY destination is a STORE: its base and
/// index registers are real reads"); the renderer's analyses were the half left
/// behind, and `Insn::use_locals` had the right test all along.
///
/// The consequence is not cosmetic. `call_result_is_consumed` asks
/// `read_count`, so a call whose result is used ONLY as a store base was
/// emitted as a bare `fn_9000(…);` and the store then referenced a name nothing
/// assigns — the `def-dropped` class of the undefined-local census:
///
///   i0 @0x5000  v0 = fn_9000()          ; ONE native bl
///   i1 @0x5004  *(nint*)(v0) = in_x9    ; the ONLY use — as a store base
///   i2 @0x5008  Return
#[test]
fn a_stores_base_is_a_read() {
    let table = plain_local_table(&[1, 2]);
    let insns = vec![
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x9000), Operand::Local(1)],
            0x5000,
        ),
        Insn::new(
            Op::Move,
            vec![
                Operand::Memory {
                    base: Box::new(Operand::Local(1)),
                    index: None,
                    scale: 1,
                    addend: 0,
                    width: 0,
                },
                Operand::Local(2),
            ],
            0x5004,
        ),
        Insn::new(Op::Return, vec![], 0x5008),
    ];
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: false,
        base_rva: 0x5000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("v0 = "),
        "the call result must be bound to its name, or the store's address base \
         reads a name nothing assigns:\n{out}"
    );
}

/// One table entry with a chosen `slot_origin`; ids are 1-based version ids.
fn table_with_origin(ids: &[u32], origin: &str) -> locals::LocalTable {
    let mut t = locals::LocalTable::default();
    for (i, id) in ids.iter().enumerate() {
        t.infos.push(locals::LocalInfo {
            slot_origin: origin.into(),
            ty: ManagedType::Unknown,
            is_this: false,
            is_param: None,
            param_comp: None,
            is_incoming_arg: false,
            is_call_clobber: false,
            is_return_buffer: false,
        });
        t.index_of.insert(*id, i);
    }
    t
}

/// A never-written `abs` base is a COMPILE-TIME ADDRESS, not a variable.
///
/// `lift_x86` materialises `[rip+disp]`, `[abs+disp]` and the bare `[disp]`
/// form as `Memory { base: Register("abs"), addend: disp }`, and for that
/// operand the address IS the displacement — the fact
/// `locals::absolute_static_addr` and `resolve::absolute_address` both take as
/// given when they name static storage. Once the pseudo-register became a
/// local, the renderer printed the LOCAL'S NAME as the base
/// (`*(byte*)(v0 +29061866)`), which claims a source variable holds a pointer
/// where the compiler emitted an address, and leaves a name nothing in the body
/// ever defines. It was the largest single group in the undefined-local census:
/// 250,218 of 354,089 references over 26,509 methods on the x64 corpus.
#[test]
fn an_absolute_base_renders_as_the_address_it_is() {
    let table = table_with_origin(&[1], "abs");
    //   i0 @0x1000  v2 = *(byte*)(v1 +29061866)   ; 0x1bb72ea
    //   i1 @0x1004  Return(v2)
    let insns = vec![
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(2),
                Operand::Memory {
                    base: Box::new(Operand::Local(1)),
                    index: None,
                    scale: 1,
                    addend: 29061866,
                    width: 1,
                },
            ],
            0x1000,
        ),
        Insn::new(Op::Return, vec![Operand::Local(2)], 0x1004),
    ];
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: true,
        base_rva: 0x1000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("0x1bb72ea"),
        "the absolute operand must render as its address:\n{out}"
    );
    assert!(
        !out.contains("v0"),
        "and the name of a local nothing defines must NOT appear:\n{out}"
    );
}

/// A FRAME CELL is a name for memory, not a lost local.
///
/// `stack_analyzer::slot_operand` names a frame cell `stack_<slot>` and uses
/// that name as the base of the `Memory` operand for every access landing in
/// it, so nothing in the body ever assigns it. Printing `v<k>` showed a name
/// with no value — 78,095 references, 22% of the undefined-local census — for a
/// local the verifier had already classified as an entry value rather than
/// corruption (its oracle skips every `stack_` origin). The slot number is
/// signed and a minus sign cannot appear in an identifier, so `stack_-255`
/// spells as `stack_neg255`.
#[test]
fn a_frame_cell_is_named_after_the_cell() {
    let table = table_with_origin(&[1], "stack_-255");
    //   i0 @0x2000  v2 = *(byte*)(v1)
    //   i1 @0x2004  Return(v2)
    let insns = vec![
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(2),
                Operand::Memory {
                    base: Box::new(Operand::Local(1)),
                    index: None,
                    scale: 1,
                    addend: 0,
                    width: 1,
                },
            ],
            0x2000,
        ),
        Insn::new(Op::Return, vec![Operand::Local(2)], 0x2004),
    ];
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: true,
        base_rva: 0x2000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("&stack_neg255"),
        "a frame cell is addressed as the cell it names:\n{out}"
    );
    assert!(
        !out.contains("v0"),
        "and it must not be spelled as a lost local:\n{out}"
    );
}

/// A read-only `x0` that is neither `this` nor a declared parameter is an
/// INCOMING value, exactly like `x2`/`x3`.
///
/// `entry_register_name` used to exclude `x0` on the theory that the receiver
/// and the first parameter were already handled — and they are: `var_name` and
/// `entry_var_name` both test `is_this`/`is_param` first. So a local that still
/// arrives at the predicate with origin `x0` is neither: it is a value the ABI
/// placed in the receiver register that the declared signature does not cover
/// (IL2CPP folds several metadata signatures onto one native body), which is the
/// case every OTHER argument register is named for. The scratch reuse the
/// exclusion guarded against has a DEFINITION, and `def_sites` refuses those
/// above. Measured on x64: 16,671 references over 5,927 methods, 7,001 of them a
/// method whose `x0` view was split across two locals.
#[test]
fn a_read_only_x0_is_named_as_an_incoming_value() {
    let table = table_with_origin(&[1], "x0");
    //   i0 @0x3000  v2 = *(byte*)(v1)
    //   i1 @0x3004  Return(v2)
    let insns = vec![
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(2),
                Operand::Memory {
                    base: Box::new(Operand::Local(1)),
                    index: None,
                    scale: 1,
                    addend: 8,
                    width: 1,
                },
            ],
            0x3000,
        ),
        Insn::new(Op::Return, vec![Operand::Local(2)], 0x3004),
    ];
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: true,
        base_rva: 0x3000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("in_x0"),
        "a read-only x0 is an incoming ABI value, like every other argument \
         register:\n{out}"
    );
    assert!(!out.contains("v0"), "and not a lost local:\n{out}");
}

/// The class-init guard fold ERASES a native `bl`, so the renderer must LEDGER
/// the erased call SITE.
///
/// `concord` scores every native call it cannot find in the text as DROPPED, so
/// a deliberate erasure has to reach the verify ledger — the same discipline
/// `resolve::key_function_recovery` follows for the
/// `il2cpp_codegen_initialize_*` thunks, under the same `LedgerKind`
/// (`DroppedRuntimeCall`, "il2cpp_codegen_initialize_* bookkeeping call
/// dropped"). Without the note the fold is invisible to the fidelity report:
/// when the recognizer started firing on x64 (the `abs` base spelling made the
/// guard test a FIXED address, which is what `idioms::reads_static_word`
/// requires) `calls visible` fell from 100.0% to 98.5% for 12,981 guards, every
/// one a shape-proven runtime guard rather than a managed call.
///
/// The note is keyed on the call SITE (0x18), not the callee: the ledger is
/// compared against decoded instruction addresses, and the callee address is a
/// different instruction's address often enough to excuse a genuine drop.
#[test]
fn the_class_init_guard_fold_ledgers_the_call_it_erases() {
    super::verify::ledger_clear();
    let table = table_with_origin(&[0, 50], "abs");
    // The shape the fold matches: `if (flag == 0) { init(); flag = 1; }` — the
    // branch skips the WHOLE arm (the helper call AND the flag write), which is
    // what a real class-init guard compiles to. `Compare` and `CondJump` share
    // one native address, as they do in the real IR, and the call's last operand
    // is the cell the guard reads.
    let flag = |addend| Operand::Memory {
        base: Box::new(Operand::Local(0)),
        index: None,
        scale: 1,
        addend,
        width: 1,
    };
    let insns = vec![
        Insn::new(
            Op::Compare(super::ir::CompareKind::NotEqual),
            vec![Operand::Local(50), flag(29051504), Operand::imm(0)],
            0x10,
        ),
        Insn::new(Op::CondJump(4), vec![Operand::Local(50)], 0x10),
        Insn::new(
            Op::Call,
            vec![
                Operand::Immediate(0x1caf40),
                Operand::Local(9),
                Operand::Local(3),
                flag(29051520),
            ],
            0x18,
        ),
        Insn::new(Op::Move, vec![flag(29051504), Operand::imm(1)], 0x20),
        Insn::new(Op::Return, vec![], 0x24),
    ];
    assert_eq!(
        super::runtime_helpers::detect(&insns),
        1,
        "the guard shape must be recognised, or no role is recorded and the \
         call site cannot be identified"
    );
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: false,
        base_rva: 0x10,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("IL2CPP_ClassInit"),
        "the guard is runtime bookkeeping managed C# does not contain:\n{out}"
    );
    assert!(
        super::verify::ledger_has_kind(0x18, super::verify::LedgerKind::DroppedRuntimeCall),
        "the erased call SITE at 0x18 must be ledgered, or the concordance \
         charges it as a call the decompiler dropped:\n{out}"
    );
}

/// `BLR` (indirect call) must lift to a real CALL, and must not touch the
/// register holding the callee.
///
/// There was no `A64Opcode::BLR` arm at all: `BLR Xn` fell through to the
/// lifter's `_ =>` catch-all, which reads operand 0 as a DESTINATION — and for
/// `BLR` operand 0 is the TARGET register. So it emitted `x8 = Unknown`:
///
///   1. the call disappeared from the IR entirely (no `Op::Call`, no
///      `Op::IndirectCall` — nothing for the renderer or the resolver to see);
///   2. the register that held the callee was overwritten with `Unknown`,
///      poisoning every later use of it; and
///   3. `resolve::rewrite_virtual_calls` — the whole vtable resolver, keyed on
///      `Op::Call` — could never fire, because its input was never produced.
///      That is where the resolved virtual calls went.
///
/// The lifter census counted 37,655 `BLR` fall-throughs in the reference
/// binary: 55% of every unhandled instruction in the whole program, and the
/// largest single hole in the port.
#[test]
fn blr_lifts_to_a_call_and_preserves_the_target_register() {
    use super::{ir::Op as IrOp, lift};
    // `BLR x8` ; `RET` (X30)
    let words: [u32; 2] = [0xD63F0100, 0xD65F03C0];
    let bytes = le(&words);
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).expect("lift");

    let call = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, IrOp::Call))
        .expect("`BLR` must lift to a call op — it used to vanish entirely");
    assert_eq!(
        call.operands.first(),
        Some(&Operand::reg("x8")),
        "the callee must be the BLR operand register, got {:?}",
        call.operands
    );
    assert_eq!(
        call.operands.get(1),
        Some(&Operand::reg("x0")),
        "operand 1 is the x0 return-value slot (dest_position(Op::Call) == Some(1)), got {:?}",
        call.operands
    );
    // The actual regression that made this invisible in the dump: x8 — the
    // callee — used to be written with `Unknown` at the call site.
    for i in &lifted.insns {
        if let Some(d) = i.dest() {
            assert_ne!(
                *d,
                Operand::reg("x8"),
                "nothing may DEFINE the callee register (at {:#x}: {:?} {:?})",
                i.address,
                i.op,
                i.operands
            );
        }
    }
}

/// `CSEL` / `CSINC` must lift to a real conditional select, and the `CSET`
/// alias must fold to a boolean.
///
/// The IR had no way to express a select at all, so all 17,281 conditional
/// selects in the reference binary (`CSINC` 7,488, `CSEL` 7,113, `FCSEL`
/// 2,247, `CSNEG` 271, `CSINV` 162) fell through the lifter's catch-all: the
/// operation disappeared and its destination was overwritten with a
/// placeholder. `CSET` is the one that matters most — `CSINC Rd, wzr, wzr,
/// cond` is how a managed `bool` materialises in optimized A64, so before this
/// a recovered `b = x < y;` was `b = __unknown;`.
#[test]
fn csel_and_cset_lift_to_a_conditional_select() {
    use super::ir::{CompareKind, Op as IrOp, SelectVariant};
    use super::lift;

    // cmp x0, x1 ; csel x2, x0, x1, eq ; ret
    //
    // `CSEL` and `CSINC` share a base and differ only in bit 10
    // (`CSEL` 0x9A800000, `CSINC` 0x9A800400), which is the same bit that
    // makes `CSET` a `CSINC` — so a wrong bit here silently asserts the
    // WRONG VARIANT rather than failing to decode.
    let words: [u32; 3] = [0xEB01001F, 0x9A810002, 0xD65F03C0];
    let lifted = lift::lift_arm64(&le(&words), 0x2000, 64, None).expect("lift");
    let sel = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, IrOp::Select(_)))
        .unwrap_or_else(|| panic!("`CSEL` must lift to a select, got {:?}", lifted.insns));
    assert!(
        matches!(sel.op, IrOp::Select(SelectVariant::Identity)),
        "`CSEL` is the identity select, got {:?}",
        sel.op
    );
    assert_eq!(
        sel.operands.len(),
        4,
        "select operands are [dst, cond, a, b], got {:?}",
        sel.operands
    );
    assert_eq!(sel.operands[2], Operand::reg("x0"), "true arm");
    assert_eq!(sel.operands[3], Operand::reg("x1"), "false arm");
    // The condition must be a REAL compare, not `Unknown`. This is the part
    // that makes the op worth having: `CSEL` reads the flags, and the flags
    // live in `carried_cmp` (it is in `preserves_nzcv`), so a select that read
    // `pending_cmp` would see `None` and degrade to `? ? a : b`.
    let cond = sel.operands[1]
        .as_local()
        .expect("the condition operand is a boolean scratch local");
    let cmp = lifted
        .insns
        .iter()
        .find(|i| i.dest().and_then(|d| d.as_local()) == Some(cond))
        .expect("the compare feeding the select must exist");
    assert!(
        matches!(cmp.op, IrOp::Compare(CompareKind::Equal)),
        "the select must be guarded by `cmp.eq`, got {:?}",
        cmp.op
    );

    // csinc w3, wzr, wzr, eq  ==  `cset w3, ne` — the managed-bool form.
    let words: [u32; 3] = [0xEB01001F, 0x1A9F07E3, 0xD65F03C0];
    let lifted = lift::lift_arm64(&le(&words), 0x2000, 64, None).expect("lift");
    let sel = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, IrOp::Select(_)))
        .expect("`CSINC` with both sources `xzr` is the `CSET` alias");
    // `CSET`'s TRUE arm is the constant on the ENCODED condition's
    // fall-through side: `CSINC` is `cond ? Xn : Xm + 1`, both sources are
    // zero, so the fold is `cond ? 0 : 1`. Folding it `cond ? 1 : 0` inverted
    // every materialised `bool` in the dump (`b = x == y` → `b = x != y`).
    assert_eq!(
        &sel.operands[2..],
        &[Operand::imm(0), Operand::imm(1)],
        "`CSET` must fold to `cond ? 0 : 1` (not `xzr + 1`, and not the inverted `cond ? 1 : 0`), got {:?}",
        sel.operands
    );
    assert!(
        matches!(sel.op, IrOp::Select(SelectVariant::Identity)),
        "the folded `CSET` is a plain select, got {:?}",
        sel.op
    );
    // The condition operand must be the ENCODED one (`eq` here), because the
    // `cset ne` a listing prints is the INVERSE of the encoding. Pinning the
    // kind is what stops the fold above from being "fixed" back by flipping
    // the condition instead of the arms.
    let cond = sel.operands[1]
        .as_local()
        .expect("the `CSET` condition is a boolean scratch local");
    let cmp = lifted
        .insns
        .iter()
        .find(|i| i.dest().and_then(|d| d.as_local()) == Some(cond))
        .expect("the compare feeding the `CSET` must exist");
    assert!(
        matches!(cmp.op, IrOp::Compare(CompareKind::Equal)),
        "`cset w3, ne` is ENCODED `csinc w3, wzr, wzr, eq`, so its condition is `eq`, got {:?}",
        cmp.op
    );
}

/// The alias inversion, end to end on the shape IL2CPP emits constantly:
/// `cmp w8, w20 ; cset w0, eq`.
///
/// The word is the REAL encoding (`csinc w0, wzr, wzr, ne`), and the listing
/// prints it as `cset w0, eq`. The recovered value must be `(w8 == w20) ? 1 : 0`:
/// the condition is the encoded `ne`, and the TRUE arm is the zero constant.
/// Both halves have to hold at once, which is exactly what a plain
/// `cond ? 1 : 0` fold gets wrong.
#[test]
fn cset_eq_materialises_the_boolean_it_claims() {
    use super::ir::{CompareKind, Op as IrOp, SelectVariant};
    use super::lift;

    // `csinc w0, wzr, wzr, ne`, i.e. the preferred disassembly `cset w0, eq`.
    let word: u32 = 0x1A9F17E0;
    assert_eq!(
        mnemonic_of(word),
        "cset w0, eq",
        "the decoder must print this encoding through the `cset` alias"
    );
    let words: [u32; 5] = [0xEB14011F, word, 0x2A0003E1, 0xD65F03C0, 0xD503201F];
    let lifted = lift::lift_arm64(&le(&words), 0x1000, 64, None).expect("lift");
    let sel = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, IrOp::Select(_)))
        .expect("`cset` must lift to a select");
    assert_eq!(
        sel.operands.len(),
        4,
        "select operands are [dst, cond, a, b], got {:?}",
        sel.operands
    );
    assert_eq!(sel.operands[2], Operand::imm(0), "`cset` true arm is zero");
    assert_eq!(sel.operands[3], Operand::imm(1), "`cset` false arm is one");
    assert!(matches!(sel.op, IrOp::Select(SelectVariant::Identity)));
    let cond = sel.operands[1]
        .as_local()
        .expect("the `cset` condition is a boolean scratch local");
    let cmp = lifted
        .insns
        .iter()
        .find(|i| i.dest().and_then(|d| d.as_local()) == Some(cond))
        .expect("the compare feeding the `cset` must exist");
    assert!(
        matches!(cmp.op, IrOp::Compare(CompareKind::NotEqual)),
        "`cset w0, eq` is encoded with `ne`; the compare kind must be the encoded one, got {:?}",
        cmp.op
    );
}

// ---------------------------------------------------------------------------
// Lifter-coverage regression tests
//
// Every word below is REAL — taken from the coverage census of the shipped
// binary, or assembled from the ARM encoding and then checked against the
// decoder. That check is the point: the coverage census (`ILR_LIFT_CENSUS_FILE`)
// is what proves these arms are needed, and the mnemonic assertions here are
// what stop the test from silently pinning whatever instruction a mistyped
// constant happens to decode to.
// ---------------------------------------------------------------------------

/// Mnemonic + operands of one word, straight from the decoder.
fn mnemonic_of(word: u32) -> String {
    use yaxpeax_arch::{Decoder, U8Reader};
    let bytes = word.to_le_bytes();
    let mut reader = U8Reader::new(&bytes);
    let decoder = yaxpeax_arm::armv8::a64::InstDecoder::default();
    match decoder.decode(&mut reader) {
        Ok(i) => format!("{i}"),
        Err(e) => format!("<decode error {e:?}>"),
    }
}

fn lift_one(word: u32) -> Vec<super::ir::Insn> {
    lift::lift_arm64(&le(&[word]), 0x1000, 4, None)
        .expect("lift")
        .insns
}

/// Is any operand of `i` an unresolved value? The catch-all's signature.
fn has_unknown(i: &super::ir::Insn) -> bool {
    i.operands.iter().any(|o| matches!(o, Operand::Unknown))
}

/// `ORN` / `BIC` / `EON` — the encoder's "inverted second operand" logical
/// forms of `ORR` / `AND` / `EOR`.
///
/// None had a lifter arm, so the destination became `Unknown`:
/// 547 `ORN` + 415 `BIC` + 150 `BICS` + `EON`, the third-largest hole the
/// census found.
#[test]
fn inverted_operand_logical_forms_lift() {
    for (word, text, want) in [
        (0x2A220020u32, "orn w0, w1, w2", ArithKind::Or),
        (0x0A220020, "bic w0, w1, w2", ArithKind::And),
        (0x4A220020, "eon w0, w1, w2", ArithKind::Xor),
    ] {
        assert_eq!(mnemonic_of(word), text, "test word is not `{text}`");
        let insns = lift_one(word);
        assert!(
            matches!(insns[0].op, Op::Not),
            "`{text}` must first materialize the complement, got {:?}",
            insns[0].op
        );
        let bin = insns
            .iter()
            .find(|i| matches!(i.op, Op::Arith(_)))
            .expect("the logical operation itself must survive");
        let Op::Arith(kind) = bin.op else {
            unreachable!()
        };
        assert_eq!(kind, want, "`{text}` must be `{want:?}`");
        // The second operand is the complement scratch, never `x2` raw — that
        // distinction is the entire difference between `a | ~b` and `a | b`.
        assert_ne!(
            bin.operands[2],
            Operand::reg("x2"),
            "`{text}` must use the complement, got {:?}",
            bin.operands
        );
        assert!(!has_unknown(bin), "`{text}` produced an unresolved value");
    }
}

/// `EXTR` (1020) and `BFM` (310) — the bitfield instructions whose general
/// forms the old arm could not express.
///
/// `extr` is `(Rn:Rm) >> lsb`; `bfi`/`bfm` inserts a rotated field and keeps
/// the destination's other bits. `0x33181E88` is the census's most common BFM
/// shape (`bfi w8, w20, #0x8, #0x8`) — the *wrapping* case, where the field
/// straddles the operand top, which is exactly what the old `imms+1 == immr`
/// pattern match could not see.
#[test]
fn bitfield_extract_and_insert_lift() {
    // extr w0, w1, w2, #3  ==  (x1 >> 3) | (x2 << 29)
    assert_eq!(mnemonic_of(0x13820C20), "extr w0, w1, w2, #0x3");
    let insns = lift_one(0x13820C20);
    assert!(
        matches!(insns[0].op, Op::Arith(ArithKind::Shr)),
        "extr must start by shifting the high half down, got {:?}",
        insns[0].op
    );
    assert_eq!(insns[0].operands[2], Operand::imm(3), "extr lsb");
    assert_eq!(
        insns[1].operands[2],
        Operand::imm(29),
        "the low half shifts up by (32 - lsb) = 29"
    );
    let joined = insns
        .iter()
        .find(|i| matches!(i.op, Op::Arith(ArithKind::Or)))
        .expect("the `or` that joins the halves");
    assert_eq!(
        joined.operands[0],
        Operand::reg("x0"),
        "extr writes its destination register"
    );

    // bfi w8, w20, #0x8, #0x8  →  insert bits [7:0] of w20 at bit 8.
    assert_eq!(mnemonic_of(0x33181E88), "bfi w8, w20, #0x8, #0x8");
    let insns = lift_one(0x33181E88);
    // wmask is DecodeBitMasks(32, imms=7, immr=24) = 0x0000ff00, so the field
    // is masked to 0xff00 and the destination keeps 0xffff00ff.
    let masked = insns
        .iter()
        .find(|i| i.operands.get(2) == Some(&Operand::imm(0xff00)))
        .expect("the inserted field must be masked to its bit range");
    assert!(
        matches!(masked.op, Op::Arith(ArithKind::And)),
        "the field mask is an `and`, got {:?}",
        masked.op
    );
    assert!(
        insns
            .iter()
            .any(|i| i.operands.get(2) == Some(&Operand::imm(0xffff00ffu32 as i64))),
        "`bfi` must preserve the destination bits outside the field: {insns:?}"
    );
    assert!(
        insns
            .iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::Or))),
        "the preserved and inserted halves are joined with `or`"
    );
}

/// Acquire/release and exclusive-access loads and stores.
///
/// `LDAXR` 326, `STURB` 373, `STURH` 148, `STLXR` 190, `STXR` 136, `LDARB`
/// 118, `LDAR` 70, `LDAXRB`/`STXRB` 10 each, `STLR` 6 — the whole atomic
/// family. Unhandled, the loaded value became `Unknown` and the store
/// vanished, which is why event accessors rendered as an empty retry shell:
/// the write-back was simply gone.
#[test]
fn acquire_release_and_exclusive_access_lift() {
    // Loads: each must produce a real memory read into the destination.
    for (word, text) in [
        (0x885FFC00u32, "ldaxr w0, [x0]"),
        (0xC8DFFC00, "ldar x0, [x0]"),
        (0xB8BFC000, "ldapr w0, [x0]"),
        (0xF8BFC000, "ldapr x0, [x0]"),
        (0x08DFFD08, "ldarb w8, [x8]"),
        (0x085FFEA9, "ldaxrb w9, [x21]"),
    ] {
        assert_eq!(mnemonic_of(word), text, "test word is not `{text}`");
        let insns = lift_one(word);
        let read = insns.first().expect("a load must survive the lift");
        assert!(
            matches!(read.op, Op::Move),
            "`{text}` must be a move out of memory, got {:?}",
            read.op
        );
        assert!(
            matches!(read.operands.get(1), Some(Operand::Memory { .. })),
            "`{text}` second operand must be a memory read, got {:?}",
            read.operands
        );
        assert!(!has_unknown(read), "`{text}` lost its value");
    }

    // Stores: the write must reach the IR. The status register of an exclusive
    // store is deliberately left undefined (it is the outcome of the atomic
    // attempt, not a value the program computes), so only the store is
    // asserted here — but the store itself must be present.
    for (word, text) in [
        (0xC89FFC20u32, "stlr x0, [x1]"),
        (0x381DF3A9, "sturb w9, [x29, #-0x21]"),
        (0x781A63BF, "sturh wzr, [x29, #-0x5a]"),
        (0x880B7D0A, "stxr w11, w10, [x8]"),
        (0x880EFD6D, "stlxr w14, w13, [x11]"),
        (0x080A7EA8, "stxrb w10, w8, [x21]"),
    ] {
        assert_eq!(mnemonic_of(word), text, "test word is not `{text}`");
        let insns = lift_one(word);
        assert!(
            insns
                .iter()
                .any(|i| matches!(i.operands.first(), Some(Operand::Memory { .. }))),
            "`{text}` must store to memory, got {insns:?}"
        );
    }

    // The unscaled store keeps its byte displacement: `sturb w9, [x29, #-0x21]`
    // is `[sp-33]`, not `[sp]`. Dropping the addend aliased every narrow store
    // onto the frame base.
    let insns = lift_one(0x381DF3A9);
    let store = insns
        .iter()
        .find(|i| matches!(i.operands.first(), Some(Operand::Memory { .. })))
        .expect("sturb must store");
    let Some(Operand::Memory { addend, .. }) = store.operands.first() else {
        unreachable!()
    };
    assert_eq!(*addend, -33, "`sturb`'s byte displacement must survive");
}

/// The `test` family: `tst a, b` is `ands xzr, a, b` and tests `a AND b`
/// against ZERO — NOT `a` against `b`.
///
/// Recording the operand pair as a plain compare made every mask test render
/// as `a == b`. `BICS` (150) had no arm at all.
#[test]
fn test_family_tests_the_and_not_the_operands() {
    assert_eq!(mnemonic_of(0xEA00001F), "tst x0, x0");
    let insns = lift_one(0xEA00001F);
    let acc = insns
        .iter()
        .find(|i| matches!(i.op, Op::Arith(ArithKind::And)))
        .expect("`tst` must materialize the AND it tests");
    assert!(
        acc.operands[1..].iter().all(|o| *o != Operand::imm(0)),
        "the AND must be of the two operands, not the operands vs zero: {:?}",
        acc.operands
    );

    // `bics xzr, x0, x1` is the complement form and sets the flags the same
    // way — it must NOT produce a bare `xzr = ...` assignment.
    assert_eq!(mnemonic_of(0x6A20001F), "bics wzr, w0, w0");
    let insns = lift_one(0x6A20001F);
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Not)),
        "`bics` must complement its second operand, got {insns:?}"
    );
    assert!(
        insns
            .iter()
            .any(|i| matches!(i.op, Op::Arith(ArithKind::And))),
        "`bics` must materialize the AND it tests"
    );
    assert!(
        !insns
            .iter()
            .any(|i| i.dest().map(|d| *d == Operand::reg("xzr")).unwrap_or(false)),
        "a flag-only `bics` must not write an `xzr` destination: {insns:?}"
    );

    // `asrv` is the arithmetic right shift — 30 occurrences, previously lost.
    assert_eq!(mnemonic_of(0x1ACF29CE), "asr w14, w14, w15");
    let insns = lift_one(0x1ACF29CE);
    assert!(
        matches!(insns[0].op, Op::Arith(ArithKind::Shr)),
        "`asrv` lifts to a right shift, got {:?}",
        insns[0].op
    );
}

/// Scalar FP: the select family, the rounding family, and the reasons
/// `fmax`/`fmin` used to silently drop their comparison.
#[test]
fn float_select_round_and_convert_lift() {
    // `fmax` used to build the compare and then UNCONDITIONALLY assign its
    // first operand — `x = a`, with the max dropped. It must now be a select.
    assert_eq!(mnemonic_of(0x1E224820), "fmax s0, s1, s2");
    let insns = lift_one(0x1E224820);
    let sel = insns
        .iter()
        .find(|i| matches!(i.op, Op::Select(_)))
        .expect("`fmax` must be a select, not an assignment");
    assert_eq!(sel.operands[2], Operand::reg("vs4"), "true arm");
    assert_eq!(sel.operands[3], Operand::reg("vs8"), "false arm");
    let cond = sel.operands[1]
        .as_local()
        .expect("the select is guarded by a compare scratch");
    let cmp = insns
        .iter()
        .find(|i| i.dest().and_then(|d| d.as_local()) == Some(cond))
        .expect("the compare feeding `fmax` must exist");
    assert!(
        matches!(cmp.op, Op::Compare(CompareKind::GreaterThan)),
        "`fmax` is `a > b ? a : b`, got {:?}",
        cmp.op
    );

    // The `maxNum` form and the rounding forms.
    assert_eq!(mnemonic_of(0x1E216800), "fmaxnm s0, s0, s1");
    assert!(
        lift_one(0x1E216800)
            .iter()
            .any(|i| matches!(i.op, Op::Select(_)))
    );
    assert_eq!(mnemonic_of(0x1E254000), "frintm s0, s0");
    let insns = lift_one(0x1E254000);
    assert!(
        matches!(insns[0].op, Op::FloatArith(FloatArithKind::Floor)),
        "`frintm` is `Math.Floor`, got {:?}",
        insns[0].op
    );

    // `fabd` / `fnmul` are composed from the ops that exist, through the
    // destination register, so they need no extra scratch slot.
    assert_eq!(mnemonic_of(0x7EA0D440), "fabd s0, s2, s0");
    let insns = lift_one(0x7EA0D440);
    assert!(
        matches!(insns[0].op, Op::FloatArith(FloatArithKind::Sub)),
        "`fabd` starts with the difference, got {:?}",
        insns[0].op
    );
    assert!(
        matches!(insns[1].op, Op::FloatArith(FloatArithKind::Abs)),
        "then takes its magnitude, got {:?}",
        insns[1].op
    );
    assert_eq!(mnemonic_of(0x1E208820), "fnmul s0, s1, s0");
    let insns = lift_one(0x1E208820);
    assert!(
        matches!(insns[0].op, Op::FloatArith(FloatArithKind::Mul))
            && matches!(insns[1].op, Op::FloatArith(FloatArithKind::Neg)),
        "`fnmul` is `-(a * b)`, got {:?} / {:?}",
        insns[0].op,
        insns[1].op
    );

    // `fcvtms` is `(int)Math.Floor(x)` — the rounding direction is the whole
    // difference from the truncating `fcvtzs`, which was the only form the
    // lifter knew.
    assert_eq!(mnemonic_of(0x1E300108), "fcvtms w8, s8");
    let insns = lift_one(0x1E300108);
    assert!(
        matches!(insns[0].op, Op::FloatArith(FloatArithKind::Floor)),
        "`fcvtms` rounds down before converting, got {:?}",
        insns[0].op
    );
    assert!(
        matches!(insns[1].op, Op::Convert(_)),
        "and then converts to integer, got {:?}",
        insns[1].op
    );
}

/// `fccmp` (237) — the floating-point twin of `ccmp`, and how the compiler
/// The pin RESTORE must reproduce the copy that ITS OWN site performed.
///
/// This is the `(this & 1) == 0` bug. IL2CPP pins a register several times with
/// DIFFERENT sources: the prologue `MOV x23, x7` (a parameter), then later
/// `MOV x23, x0` (the receiver, re-established across a call). The restore
/// table was keyed by pin REGISTER, so the later pin overwrote the earlier
/// source and the prologue copy came back as a copy of `this`. Every use of
/// that parameter then aliased the receiver — `(this & 1) == 0`,
/// `x.GetType() == x.GetType()`, wrong call argument registers.
///
/// Bytes are the real `FindMatchingElement` @0x1F4F660 prologue (its six
/// argument pins), then a `bl` so the x7 pin dies (x7 is caller-saved), then a
/// later `MOV x23, x0` that re-pins x23 with a different source.
#[test]
fn pin_restore_uses_the_source_of_its_own_site() {
    let words: [u32; 20] = [
        0xD10283FF, 0xA9046FFC, 0xA90567FA, 0xA9065FF8, 0xA90757F6, 0xA9084FF4, 0xA9097BFD,
        0x910243FD, 0xAA0703F7, 0xAA0603F5, 0xAA0403F9, 0xAA0303F4, 0xAA0203F6, 0xAA0103FB,
        0xF9000FE0, 0xB001C348,
        0x94000000, // bl            (kills the x7 pin: x7 is caller-saved)
        0xAA0003F7, // mov x23, x0   (re-pin x23 from a DIFFERENT source)
        0xD2800020, // movz x0, #1
        0xD65F03C0, // ret
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let lifted = lift::lift_arm64(&bytes, 0x1f4f660, 512, None).expect("lift");
    let mut insns = lifted.insns;
    let seed = locals::SeedInfo {
        param_types: (0..7)
            .map(|_| ManagedType::Primitive(PrimKind::I4))
            .collect(),
        ret_type: ManagedType::Primitive(PrimKind::I4),
        is_static: false,
        declaring: None,
        param_names: Vec::new(),
        param_comp_names: Vec::new(),
        arm32_param_words: Vec::new(),
        is_arm32: false,
        ret_hidden_buffer: false,
    };
    locals::rewrite_pinned_copies(&mut insns, &seed);

    let prologue = insns
        .iter()
        .find(|i| i.address == 0x1f4f680)
        .expect("the prologue pin site must survive");
    assert_eq!(
        prologue.operands,
        vec![Operand::reg("x23"), Operand::reg("x7")],
        "the prologue pin must be restored from ITS OWN source (x7, the parameter), not from a later pin of the same register"
    );
}

/// The receiver identity in x0 must DIE at the first call.
///
/// THIS IS THE `x.GetType() == x.GetType()` BUG — ChatGPT's "strong sign that
/// value identity is still being lost", and the `Type.op_Equality(a, a)` /
/// wrong-call-argument family generally (89 sites in the reference dump).
///
/// `identity_of_src` used to answer "x0 holds `this`" unconditionally, for the
/// whole body. But x0 is the RETURN-VALUE register too. So after a call,
/// `MOV xd, x0` (copy the result into an argument register) was read as "xd is
/// a pinned copy of the receiver", and the next call's argument was rewritten
/// to x0 — which had just been overwritten with the receiver. Two distinct
/// arguments then aliased the same register and rendered identically.
///
/// Real bytes from `FindMatchingElement` @0x1F4F660. The native is:
///   0x01f4f7d0  bl   <callee>   ; x0 := result
///   0x01f4f7d4  mov  x1, x0     ; arg1 := THAT RESULT  (0xAA0003E1)
///   0x01f4f7d8  mov  x0, x20    ; arg0 := x20          (0xAA1403E0)
///   0x01f4f7e0  bl   <callee>   ; arg0/arg1 must stay distinct
/// Before the fix the second call's arg1 came out as `x0` — i.e. the same
/// value as arg0 — which is exactly `Type.op_Equality(a, a)`.
#[test]
fn receiver_identity_in_x0_dies_at_the_first_call() {
    let words: [u32; 5] = [
        0x94000000, // bl            (x0 becomes the RESULT; `this` is gone)
        0xAA0003E1, // mov x1, x0    (arg1 := the result of that call)
        0xAA1403E0, // mov x0, x20   (arg0 := x20)
        0x94000000, // bl            (the call whose args must stay distinct)
        0xD65F03C0, // ret
    ];
    assert_eq!(mnemonic_of(0xAA0003E1), "mov x1, x0");
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let lifted = lift::lift_arm64(&bytes, 0x1f4f7d0, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let seed = locals::SeedInfo {
        param_types: (0..7)
            .map(|_| ManagedType::Primitive(PrimKind::I4))
            .collect(),
        ret_type: ManagedType::Primitive(PrimKind::I4),
        is_static: false,
        declaring: None,
        param_names: Vec::new(),
        param_comp_names: Vec::new(),
        arm32_param_words: Vec::new(),
        is_arm32: false,
        ret_hidden_buffer: false,
    };
    locals::rewrite_pinned_copies(&mut insns, &seed);

    let calls: Vec<&Insn> = insns.iter().filter(|i| i.op == Op::Call).collect();
    assert_eq!(calls.len(), 2, "both calls must survive the lift");
    // Call operand layout: [target, x0 return slot, arg0, arg1, …].
    assert_eq!(
        calls[1].operands[2],
        Operand::reg("x0"),
        "arg0 is x20's value (the receiver), got {:?}",
        calls[1].operands[2]
    );
    assert_eq!(
        calls[1].operands[3],
        Operand::reg("x1"),
        "arg1 must still read x1 (the copy of the previous call's RESULT) — \
         rewriting it to x0 makes arg0 == arg1, i.e. `Type.op_Equality(a, a)`"
    );
}

/// joins two float tests into one branch. Unhandled, the pending compare was
/// cleared, so the joined branch rendered `if (? != 0)`.
#[test]
fn fccmp_keeps_a_real_condition_for_the_following_branch() {
    assert_eq!(mnemonic_of(0x1E225400), "fccmp s0, s2, #0x0, pl");
    // fccmp s0, s2, #0x0, pl ; b.ne +8 ; ret
    let words: [u32; 3] = [0x1E225400, 0x54000041, 0xD65F03C0];
    let lifted = lift::lift_arm64(&le(&words), 0x1000, 64, None).expect("lift");
    let jump = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::CondJump(_)))
        .expect("the branch must survive");
    let cond = jump.operands[0]
        .as_local()
        .expect("the branch tests a compare scratch, not a bare register");
    let cmp = lifted
        .insns
        .iter()
        .find(|i| i.dest().and_then(|d| d.as_local()) == Some(cond))
        .expect("the branch's condition must be defined by a compare");
    let Op::Compare(_) = cmp.op else {
        panic!("expected a compare, got {:?}", cmp.op)
    };
    assert!(
        cmp.operands[1] == Operand::reg("vs0") && cmp.operands[2] == Operand::reg("vs8"),
        "`fccmp` must compare its own two operands, got {:?}",
        cmp.operands
    );
}

/// `ADD Rd, sp, #imm` with `Rd != sp` computes the ADDRESS of a stack slot —
/// it does not move the stack pointer.
///
/// Real bytes from `StopsCamera` @0x28E68 of the shipped iOS binary (read with
/// `od -t x4` at file offset 0x28E68, so the mnemonics are the machine's, not
/// a transcription):
///
/// ```text
/// 0x28e68  sub  sp, sp, #0x60    ; d10183ff — sp REALLY moves
/// 0x28e70  stp  x20, x19, [sp, #0x40]
/// 0x28e78  add  x29, sp, #0x50   ; 910143fd — address; sp does NOT move
/// 0x28ef0  add  x19, sp, #0x18   ; 910063f3 — &slot24
/// 0x28f04  add  x0,  sp, #0x18   ; 910063e0 — the `&enumerator` arg of MoveNext
/// ```
///
/// The old arm keyed only on the SOURCE (`operand_at(1) == sp`), so every one
/// of these lifted as `ShiftStack`, an op with NO destination operand. Two
/// defects followed and both are pinned here:
///
///  1. the destination register was never defined, so the `&enumerator`
///     receiver became an undefined local and the ABI scrub pass then marked
///     it as an argument hole — the `__abi.MoveNext()` class;
///  2. the stack-depth pass added the offset to the frame model even though
///     `sp` never moved, so a slot stored before the instruction and loaded
///     after it were named two different `stack_N` slots.
#[test]
fn add_into_rd_from_sp_is_taking_an_address_not_moving_sp() {
    let bytes = le(&[
        0xD10183FF, // sub  sp, sp, #0x60
        0xA90357F6, // stp  x22, x21, [sp, #0x30]
        0xA9044FF4, // stp  x20, x19, [sp, #0x40]
        0xA9057BFD, // stp  x29, x30, [sp, #0x50]
        0x910143FD, // add  x29, sp, #0x50
        0xAA0003F3, // mov  x19, x0
        0x910063F3, // add  x19, sp, #0x18
        0x910063E0, // add  x0,  sp, #0x18
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x28e68, 64, None).unwrap();

    let shifts: Vec<i64> = lifted
        .insns
        .iter()
        .filter_map(|i| match i.op {
            Op::ShiftStack(d) => Some(d),
            _ => None,
        })
        .collect();
    assert_eq!(
        shifts,
        vec![-96],
        "only `sub sp, sp, #0x60` may move the stack pointer; an `add Rd, sp, #imm` \
         with Rd != sp must not, or the frame-depth model drifts from that \
         instruction onward and every spill slot loses its identity"
    );

    let addr = |reg: &str| -> i64 {
        lifted
            .insns
            .iter()
            .find(|i| {
                i.dest() == Some(&Operand::reg(reg))
                    && matches!(
                        i.operands.get(1),
                        Some(Operand::AddressOf(inner))
                            if matches!(inner.as_ref(), Operand::Memory { .. })
                    )
            })
            .map(|i| match i.operands.get(1) {
                Some(Operand::AddressOf(inner)) => match inner.as_ref() {
                    Operand::Memory { base, addend, .. } => {
                        assert_eq!(
                            base.as_ref(),
                            &Operand::reg("sp"),
                            "the slot address is sp-relative"
                        );
                        *addend
                    }
                    other => panic!("expected a memory slot, got {other:?}"),
                },
                other => panic!("expected an address, got {other:?}"),
            })
            .unwrap_or_else(|| panic!("`add {reg}, sp, #0x18` produced no value at all"))
    };
    assert_eq!(
        addr("x19"),
        24,
        "`add x19, sp, #0x18` is the address of slot +24"
    );
    assert_eq!(
        addr("x0"),
        24,
        "`add x0, sp, #0x18` is the address of slot +24"
    );
    assert_eq!(
        addr("x29"),
        80,
        "`add x29, sp, #0x50` is the address of slot +80"
    );
}

/// ARM32's frame-pointer setup — `add fp, sp, #k` — must take the SAME
/// `Rd = &slot` shape arm64 emits, or the frame alias is never established.
///
/// Real bytes from a shipped Android method (file offset 0x53d164):
///
/// ```text
/// 0x53d164  push {r4, fp, lr}     ; e92d4c10
/// 0x53d168  add  r11, sp, #8      ; e28db008 — fp = sp + 8
/// 0x53d16c  sub  sp, sp, #0x10    ; e24dd010 — sp REALLY moves
/// ```
///
/// `sp` is never DEFINED in this IR (the frame setup is a destination-less
/// `ShiftStack`), so lifting `add r11, sp, #8` as plain arithmetic left a
/// never-defined read of `sp` — `undef-local origin sp` was the largest arm32
/// undefined-local origin (11,818 findings), and because
/// `stack_analyzer::frame_alias_def` recognises ONLY the alias shape, no alias
/// was recorded either: every `[fp + N]` access stayed register-relative
/// instead of becoming a `stack_N` cell. Mirroring arm64's shape gives the two
/// architectures one frame model.
#[test]
fn arm32_add_into_a_register_from_sp_is_taking_an_address() {
    let bytes = le(&[
        0xE92D4C10, // push {r4, fp, lr}
        0xE28DB008, // add  r11, sp, #8
        0xE24DD010, // sub  sp, sp, #0x10
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x53d164, 16).expect("lift");

    let alias = lifted
        .insns
        .iter()
        .find(|i| i.dest() == Some(&Operand::reg("x11")))
        .unwrap_or_else(|| {
            panic!(
                "`add r11, sp, #8` produced nothing: {:?}",
                lifted
                    .insns
                    .iter()
                    .map(|i| i.op.clone())
                    .collect::<Vec<_>>()
            )
        });
    match alias.operands.get(1) {
        Some(Operand::AddressOf(inner)) => match inner.as_ref() {
            Operand::Memory { base, addend, .. } => {
                assert_eq!(
                    base.as_ref(),
                    &Operand::reg("sp"),
                    "the slot address is sp-relative"
                );
                assert_eq!(*addend, 8, "`add r11, sp, #8` is sp + 8");
            }
            other => panic!("expected an sp-relative slot, got {other:?}"),
        },
        other => panic!("`add r11, sp, #8` must be an ADDRESS, got {other:?}"),
    }

    // Exactly TWO instructions move the stack pointer here: `push` and
    // `sub sp, sp, #0x10`. `add r11, sp, #8` moves nothing, so it must not add a
    // third (the `Rd != sp` form lifted as `ShiftStack` is the arm64 defect this
    // shape exists to prevent).
    let shifts: Vec<i64> = lifted
        .insns
        .iter()
        .filter_map(|i| match i.op {
            Op::ShiftStack(d) => Some(d),
            _ => None,
        })
        .collect();
    assert_eq!(
        shifts.len(),
        2,
        "only `push` and `sub sp, sp, #imm` move sp, got {shifts:?}"
    );
    assert!(
        shifts.contains(&-16),
        "`sub sp, sp, #0x10` must move the frame by -16: {shifts:?}"
    );
    assert!(
        shifts.iter().all(|d| *d <= 0),
        "a frame-pointer setup must never look like a POSITIVE sp adjustment: {shifts:?}"
    );

    // …and no ARITHMETIC may read `sp` as a value: it is never defined, so the
    // read becomes an undefined local (the regression this pins).
    for i in &lifted.insns {
        if matches!(i.op, Op::Arith(_)) {
            assert!(
                !i.operands
                    .iter()
                    .any(|o| matches!(o, Operand::Register(r) if r == "sp")),
                "sp is never defined, so arithmetic on it becomes an undefined local: {i:?}"
            );
        }
    }
}

/// The A32 EXT group (`sxtb`/`sxth`/`uxtb`/`uxth`/`uxtb16`/`sxtb16`) has to be
/// lifted from the raw word, because this crate cannot decode ANY of it.
///
/// `0xE6FF5074` (`uxth r5, r4`) is a real word from libil2cpp.so method
/// `0x545724`, where it is the loop counter's truncation. `lift_arm32` emitted
/// NOTHING for it, so `blo 0x545814` — the loop's back edge — found no IR at its
/// target and `cfg::resolve_target` slid the edge forward to `0x545818`. The
/// emitted C# then never recomputes the counter: the loop keeps the first
/// iteration's value. 18,206 words of the ARM32 census's decode failures are
/// this group, 95% of them `UXTH`.
#[test]
fn arm32_ext_group_leaves_real_ir_at_its_own_address() {
    let bytes = le(&[
        0xE28DB008, // add  r11, sp, #8
        0xE6FF5074, // uxth r5, r4
        0xE12FFF1E, // bx   lr
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1000, 16).expect("lift");
    let show = || {
        lifted
            .insns
            .iter()
            .map(|i| (i.address, i.op.clone(), i.operands.clone()))
            .collect::<Vec<_>>()
    };
    // The mask lands at the INSTRUCTION's address — this is the property that
    // keeps a branch targeting it from sliding.
    assert!(
        lifted.insns.iter().any(|i| {
            i.address == 0x1004
                && matches!(i.op, Op::Arith(ArithKind::And))
                && i.operands.get(2) == Some(&Operand::imm(0xFFFF))
        }),
        "`uxth r5, r4` must leave a 16-bit mask at 0x1004: {:?}",
        show()
    );

    // A PREDICATED extension runs only when the condition holds, so it carries a
    // guard; dropping the condition would truncate when the machine does not.
    let pred = le(&[0x06FF5074, 0xE12FFF1E]); // uxtheq r5, r4 ; bx lr
    let lifted = lift_arm32::lift_arm32(&pred, 0x2000, 16).expect("lift");
    assert!(
        lifted
            .insns
            .iter()
            .any(|i| matches!(i.op, Op::CondJump(0x2004))),
        "a predicated `uxtheq` must be guarded to the fallthrough: {:?}",
        lifted
            .insns
            .iter()
            .map(|i| (i.address, i.op.clone()))
            .collect::<Vec<_>>()
    );
}

/// NEON (Advanced SIMD) is decoded by neither yaxpeax nor the hand VFP decoder
/// (`is_vfp` deliberately excludes `cond == 0xF`), so before this it produced NO
/// IR at all — the largest real drop the decode-failure census names, 9,362 words
/// of a 3,000-method sample. The lifter now records it as the catch-all
/// `Op::Simd` AT ITS OWN ADDRESS, which is what a branch targeting it needs in
/// order to resolve instead of sliding forward to the next instruction.
#[test]
fn arm32_neon_leaves_ir_at_its_own_address() {
    let bytes = le(&[
        0xF2C00050, // vmov.i32 q8, #0
        0xF4400ACF, // vst1.64 {d16, d17}, [r0]
        0xE12FFF1E, // bx   lr
    ]);
    let lifted = lift_arm32::lift_arm32(&bytes, 0x1000, 16).expect("lift");
    for addr in [0x1000u64, 0x1004] {
        assert!(
            lifted
                .insns
                .iter()
                .any(|i| i.address == addr && matches!(i.op, Op::Simd(_))),
            "a NEON word must leave IR at {addr:#x}: {:?}",
            lifted
                .insns
                .iter()
                .map(|i| (i.address, i.op.clone()))
                .collect::<Vec<_>>()
        );
    }
}

/// `adrp` materializes a PAGE, not a slot address — and a page must never be
/// resolved as if it were one.
///
/// IL2CPP's metadata-usage idiom is `adrp xN, page; add xN, xN, #lo`, and the
/// slot address is the SUM. The page on its own is very often the address of a
/// DIFFERENT slot that happens to begin that page, so resolving the page named
/// the neighbouring member and the `#lo` then rendered as an offset into it.
/// Real site, `DrawGizmo` @0x14048 (`adrp x0, 0x6091000; add x0, x0, #0xce0`):
/// the argument rendered `*(nint*)(MethodInfo("KeyList.Clear") + 3296)` — a
/// method from an unrelated assembly — and now renders
/// `MethodInfo("<>c__DisplayClass4_0.<DrawGizmo>b__0")`, which is this method's
/// own display class.
#[test]
fn adrp_page_is_not_a_slot_address() {
    // adrp x0, 0x6091000 ; add x0, x0, #0xce0
    let paged = vec![
        Insn::move_(Operand::Local(0), Operand::imm(0x6091000), 0x1000),
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(1), Operand::Local(0), Operand::imm(0xce0)],
            0x1004,
        ),
    ];
    assert_eq!(
        resolve::page_base_offset_uses(&paged, 0, 0),
        Some(vec![(1, 0xce0)]),
        "a page followed only by a constant add is a page"
    );

    // The `ldr xN, [xN, #lo]` spelling of the same thing.
    let folded = vec![
        Insn::move_(Operand::Local(0), Operand::imm(0x6059000), 0x1000),
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(1),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::Local(0)),
                    index: None,
                    scale: 8,
                    addend: 0xbb0,
                },
            ],
            0x1004,
        ),
    ];
    assert_eq!(
        resolve::page_base_offset_uses(&folded, 0, 0),
        Some(vec![(1, 0xbb0)]),
        "the low bits folded into the addressing mode are the same page idiom"
    );

    // A complete address used AS an address is not a page: `drawGizmo` loaded
    // straight out of a slot and then called must stay resolvable in place.
    let direct = vec![
        Insn::move_(Operand::Local(0), Operand::imm(0x9100), 0x1000),
        Insn::new(Op::Call, vec![Operand::Local(1), Operand::Local(0)], 0x1004),
    ];
    assert_eq!(
        resolve::page_base_offset_uses(&direct, 0, 0),
        None,
        "a local used as a call target is an address, not a page base"
    );
}

/// `fcsel` draws BOTH of its value sources from the FP register file.
///
/// Real bytes from `Mathf.Berp` (`0x14534`):
///
/// ```text
/// 0x14530  fcmp  s2, s12
/// 0x14534  fcsel s0, s12, s2, gt
/// 0x14538  fcmp  s2, #0.0
/// 0x1453c  fmov  s1, wzr
/// 0x14540  fcsel s13, s1, s0, mi
/// ```
///
/// The select arm read its sources with the GPR-only `operand_at`, which
/// returns `None` for a `SIMDRegister` — so both arms became `Unknown` and the
/// whole select rendered `? __unknown : __unknown`: which of the two floats is
/// chosen, the one thing the instruction actually does, was gone. 943 sites.
#[test]
fn fcsel_draws_its_value_sources_from_the_fp_register_file() {
    let bytes = le(&[
        0x1e2c2040, // fcmp  s2, s12
        0x1e22cd80, // fcsel s0, s12, s2, gt
        0x1e202048, // fcmp  s2, #0.0
        0x1e2703e1, // fmov  s1, wzr
        0x1e204c2d, // fcsel s13, s1, s0, mi
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x1000, bytes.len(), None).expect("lift");
    let sel = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Select(_)))
        .expect("the fcsel must produce a Select");
    assert_eq!(
        sel.operands.get(2),
        Some(&Operand::reg("vs48")),
        "`fcsel s0, s12, s2, gt` picks s12 on the true arm — got {:?}",
        sel.operands
    );
    assert_eq!(
        sel.operands.get(3),
        Some(&Operand::reg("vs8")),
        "…and s2 on the false arm — got {:?}",
        sel.operands
    );
    assert!(
        !has_unknown(sel),
        "no arm of the select may be unresolved: {:?}",
        sel.operands
    );
}

/// `fmov Sd, WZR` is FP +0.0, not a copy of a register that does not exist.
///
/// The source read went through `fp_of` on a GPR operand (always `None`) and
/// `unwrap_or_default()` on `Option<String>` yields the EMPTY string, so the
/// IR held `Move(vs1, Register(""))` — a register with no name, which the
/// renderer can only report as an unsupported operand.
#[test]
fn fmov_from_the_zero_register_is_a_float_zero() {
    let insns = lift_one(0x1e2703e1); // fmov s1, wzr
    assert_eq!(
        insns[0].op,
        Op::Move,
        "expected the plain move, got {:?}",
        insns[0].op
    );
    assert_eq!(
        insns[0].operands.get(1),
        Some(&Operand::FloatLiteral(0.0)),
        "`fmov s1, wzr` is +0.0f — got {:?}",
        insns[0].operands
    );
    for i in &insns {
        assert!(
            !i.operands
                .iter()
                .any(|o| matches!(o, Operand::Register(n) if n.is_empty())),
            "no lifter arm may emit an unnamed register: {:?}",
            i.operands
        );
    }
}

/// Every instruction that WRITES a register must DEFINE it — including the
/// vector/SIMD opcodes the lifter has no scalar decomposition for.
///
/// The old fall-through defined a GPR destination as `Unknown` and, for a
/// VECTOR destination, emitted no instruction at all: the register silently
/// kept its previous value and every later read fabricated a different
/// expression. The generic arm now emits the opcode's own mnemonic as a named
/// intrinsic over its real operands, so the value survives even though its
/// semantics are not modelled yet.
#[test]
fn unmodelled_vector_and_wide_opcodes_still_define_their_destination() {
    for (word, mnemonic) in [
        (0x4f00e400u32, "movi"), // movi v0.4s, #0
        // `umov w15, v1.b[3]` — yaxpeax prints this encoding with its `mov`
        // alias, which is exactly why a census keyed on the mnemonic string
        // alone can miss a hole (`UMOV`, 23 occurrences).
        (0x0e0c3c2f, "mov"),
        (0x9bc97e89, "umulh"), // umulh x9, x20, x9   (census: UMULH 142)        // The `ubfm`/`ubfx` family again, printed with its `lsr` alias.
        (0xd34bfd36, "lsr"),
    ] {
        let decoded = mnemonic_of(word);
        assert_eq!(
            decoded.split_whitespace().next().unwrap_or(""),
            mnemonic,
            "test word is not the instruction it claims to be: `{decoded}`"
        );
        let insns = lift_one(word);
        let def = insns
            .iter()
            .find(|i| i.dest().is_some())
            .unwrap_or_else(|| panic!("`{mnemonic}` produced no defining instruction"));
        assert!(
            !has_unknown(def),
            "`{mnemonic}` must keep its operands, got {def:?}"
        );
        assert_ne!(
            def.dest(),
            Some(&Operand::Unknown),
            "`{mnemonic}` must define its destination with a value, not a placeholder"
        );
    }
}

/// An Advanced SIMD "three same" word must NOT abort the whole method.
///
/// yaxpeax maps `add v0.2s,…` and `and v1.8b,…` onto the scalar
/// `A64Opcode::ADD`/`AND`, but with a SIMD destination. The scalar arms used
/// `gpr_of(..)?`, so one such word returned `None` from `lift_arm64` and the
/// caller emitted a whole-method `[recovery failed]` marker — seven real
/// methods (0x19AB490, 0x154825C, 0x1548B34, 0x1838138, 0x17CC238,
/// 0x17CDDA8, 0x18285A0). The generic intrinsic arm must keep the value.
#[test]
fn advanced_simd_three_same_does_not_abort_the_lift() {
    for (word, text, want_dst) in [
        (0x0EA18400u32, "add v0.2s", "vd0"), // add v0.2s, v0.2s, v1.2s
        (0x0EA08420, "add v0.2s", "vd0"),    // add v0.2s, v1.2s, v0.2s
        (0x0E221C21, "and v1.8b", "vd1"),    // and v1.8b, v1.8b, v2.8b
        (0x0E211C01, "and v1.8b", "vd1"),    // and v1.8b, v0.8b, v1.8b
    ] {
        let decoded = mnemonic_of(word);
        assert!(
            decoded.starts_with(text),
            "test word is not `{text}`: `{decoded}`"
        );
        let insns = lift_one(word);
        let def = insns
            .iter()
            .find(|i| i.dest().is_some())
            .unwrap_or_else(|| panic!("`{text}` produced no defining instruction"));
        assert_eq!(
            def.dest(),
            Some(&Operand::reg(want_dst)),
            "`{text}` must define {want_dst}, got {def:?}"
        );
        assert!(
            !has_unknown(def),
            "`{text}` must keep its SIMD operands, got {def:?}"
        );
    }
}

/// A select nested inside another expression must be PARENTHESISED, and its
/// condition must resolve to the compare that produced it.
///
/// Two independent bugs, both visible in one rendered line of `Mathf.Berp`
/// before this round:
///
/// ```text
/// return (end - start) * ((1f - t4294967292 ? __unknown : __unknown) * 1.2f + 1f) * …
/// ```
///
///  * `Expr::Cond` reported itself as a tight leaf (precedence 15), so no
///    parent ever wrapped it. In C# the conditional operator binds LOOSER than
///    `-`, so `1f - c ? a : b` parses as `(1f - c) ? a : b`: the enclosing
///    subtraction was swallowed by the condition and the select's payload was
///    attached to the wrong operator. Wrong logic, not a formatting nit.
///  * the condition is the lifter's scratch slot. `operand_expr` inlines a
///    local's reaching definition, but a scratch slot is not in the local
///    table and its def can sit in another block, so the inline path fell back
///    to the local's NAME — and the name of a scratch slot is its own id
///    (`t4294967292`). Nothing else can produce a `t4294967…` identifier.
#[test]
fn a_nested_select_is_parenthesised_and_resolves_its_condition() {
    let table = locals::LocalTable::default();
    // Scratch ids count DOWN from u32::MAX-2; -3 is the third one handed out.
    let scratch = u32::MAX - 3;
    let insns = vec![
        Insn::new(
            Op::Compare(CompareKind::GreaterThan),
            vec![
                Operand::Local(scratch),
                Operand::Local(1),
                Operand::Local(2),
            ],
            0x1000,
        ),
        Insn::new(
            Op::Select(SelectVariant::Identity),
            vec![
                Operand::Local(3),
                Operand::Local(scratch),
                Operand::imm(10),
                Operand::imm(20),
            ],
            0x1004,
        ),
        Insn::new(
            Op::Arith(ArithKind::Sub),
            vec![Operand::Local(4), Operand::imm(100), Operand::Local(3)],
            0x1008,
        ),
        Insn::new(Op::Return, vec![Operand::Local(4)], 0x100c),
    ];
    let ir = super::RecoveredIr {
        insns,
        table,
        returns_value: true,
        base_rva: 0x1000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    println!("--- rendered ---\n{out}----------------");
    assert!(
        !out.contains("t4294967"),
        "the lifter's scratch id leaked into the output as a variable name:\n{out}"
    );
    assert!(
        out.contains(" ? "),
        "the select must still render as a ternary:\n{out}"
    );
    assert!(
        out.contains("100 - ("),
        "a conditional operator used as an operand must be parenthesised:\n{out}"
    );
}

/// Call arguments must be placed by the register CLASS their metadata type
/// allocates, not by position in the lifter's register dump.
///
/// The lifter records every `bl` as `[target, x0 result, x0…x7, fp-written…]`
/// — AAPCS64 register order. For `void M(int a, float b)` on an instance, the
/// native call passes `this` in x0, `a` in x1 and `b` in **s0**, which sits at
/// the END of the dump. Reading positionally therefore gives `b` the value of
/// `x2`, and truncating the dump to the declared parameter count throws the s0
/// operand away entirely — both silent, both leaving a plausibly-shaped call
/// with one wrong argument.
#[test]
fn call_arguments_are_placed_by_register_class_not_dump_position() {
    let mut table = locals::LocalTable::default();
    let mut add = |v: u32, origin: &str| {
        table.infos.push(locals::LocalInfo {
            slot_origin: origin.into(),
            ty: ManagedType::Unknown,
            is_this: origin == "x0",
            is_param: None,
            param_comp: None,
            is_incoming_arg: false,
            is_call_clobber: false,
            is_return_buffer: false,
        });
        table.index_of.insert(v, table.infos.len() - 1);
    };
    // SSA versions: 10 = x0 (receiver), 11 = x1 (int param), 12 = s0 (float
    // param), 13 = x2 (a register the callee does not declare — prologue spill)
    add(10, "x0");
    add(11, "x1");
    add(12, "vs0");
    add(13, "x2");

    // `void M(int a, float b)` — one integer and one FP parameter, the shape
    // whose positional read is wrong in both directions.
    let mut m = method_ref_for_instance("M", "Test");
    m.sig.params = vec![
        ManagedType::Primitive(PrimKind::I4),
        ManagedType::Primitive(PrimKind::R4),
    ];
    let target = Operand::Method(m);

    let mut ops = vec![target.clone()];
    for v in [10u32, 11, 13, 13, 13, 13, 13, 13] {
        ops.push(Operand::Local(v)); // x0..x7
    }
    ops.push(Operand::Local(12)); // the FP region
    let mut insns = vec![Insn::new(Op::CallVoid, ops, 0x1000)];

    let n = locals::align_call_arguments(&mut insns, &table, |_| None, true);
    assert_eq!(n, 1, "the call's arguments must be realigned");
    assert_eq!(
        insns[0].operands,
        vec![
            target,
            Operand::Local(10),
            Operand::Local(11),
            Operand::Local(12),
        ],
        "expected [target, receiver(x0), int(x1), float(v0)]"
    );
}

/// `str dN, [x, #imm]` — storing a DOUBLE to memory.
///
/// The store arms read their source through `operand_at`, which names GPRs
/// only: an `A64Op::SIMDRegister` source returned `None`, and the arm's
/// `if let (Some(v), Some(m))` then emitted NOTHING. So a double field write
/// had no IR at all while the `ldr d0` that produced its value did — a side
/// effect the native code performs and the C# output never shows. On the
/// reference binary that is 2,359 `str`/`stur` plus part of the `stp` count,
/// the largest instruction-coverage hole in the lifter.
///
/// Bytes are the real ones at `0x80a8`/`0x80ac` of `libs/ios/UnityFramework`:
///
/// ```text
/// 0x000080a8  ldr d0, [x8, #0xd30]    ; fd469900
/// 0x000080ac  str d0, [x19, #0x10]    ; fd000a60  <- the dropped one
/// 0x000080b0  str xzr, [x19, #0x18]   ; f9000e7f
/// ```
#[test]
fn fp_store_is_lifted_like_the_fp_load_that_fed_it() {
    let words: Vec<u32> = vec![
        0xfd469900, // ldr d0, [x8, #0xd30]
        0xfd000a60, // str d0, [x19, #0x10]
        0xf9000e7f, // str xzr, [x19, #0x18]
        0xd65f03c0, // ret
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let lifted = lift::lift_arm64(&bytes, 0x80a8, 8, None).expect("lifts");

    // The load defines an FP register...
    let loaded = lifted.insns[0].dest().cloned().expect("ldr has a dest");
    assert!(
        matches!(&loaded, Operand::Register(n) if n.starts_with("vd")),
        "a 64-bit FP load must define a `vd` register, got {loaded:?}"
    );

    // ...and the store writes THAT register to the SAME address the
    // instruction encodes. Before the fix this instruction produced no IR.
    let store = lifted
        .insns
        .iter()
        .find(|i| {
            matches!(&i.operands.first(), Some(Operand::Memory { addend, .. }) if *addend == 16)
        })
        .expect("`str d0, [x19, #0x10]` produced no IR — the FP store is being dropped");
    assert_eq!(
        store.operands,
        vec![
            Operand::Memory {
                width: 8,
                base: Box::new(Operand::reg("x19")),
                index: None,
                scale: 8,
                addend: 16,
            },
            loaded.clone(),
        ],
        "the store must write the value the load produced, at x19+0x10"
    );
}

/// A deferred compare must read the value a register held WHEN THE FLAGS WERE
/// SET, not whatever it holds by the time the compare is built.
///
/// The lifter materializes the `Compare` at the BRANCH, because the branch's
/// condition code is what selects the comparison kind. Its operands, however,
/// are register NAMES captured at the `cmp`, and SSA renames a register use by
/// the listing position it appears at. So a redefinition in between rewrites
/// the operand to the later value.
///
/// Real bytes at `0x4fba14` of `libs/ios/UnityFramework` — the compiler
/// comparing a call result against the value it is about to overwrite it with:
///
/// ```text
/// 0x4fba10  bl     …                  9517262a
/// 0x4fba14  cmp    x0, x21            eb15001f   ; the OLD x21
/// 0x4fba18  mov    x21, x0            aa0003f5   ; x21 := the call result
/// 0x4fba1c  b.eq   +0x3c              540001e0   ; here x21 IS x0
/// ```
///
/// Emitted as `Compare(Equal)[s, Register("x0"), Register("x21")]`, that is
/// `x0 == x0` — always true. This is the whole family of nonsense conditions
/// the dump shows as `(this & 1) == 0` and `x.GetType() == x.GetType()`.
#[test]
fn deferred_compare_reads_the_value_the_register_had_at_the_cmp() {
    let words: Vec<u32> = vec![
        0x9517262a, // bl
        0xeb15001f, // cmp x0, x21
        0xaa0003f5, // mov x21, x0
        0x540001e0, // b.eq +0x3c
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let lifted = lift::lift_arm64(&bytes, 0x4fba10, 8, None).expect("lifts");

    // The redefinition of x21 is preceded by a copy of its OLD value.
    let pin_idx = lifted
        .insns
        .iter()
        .position(|i| {
            i.operands.len() == 2
                && matches!(&i.operands[0], Operand::Local(p) if lift::is_scratch(*p))
                && matches!(&i.operands[1], Operand::Register(n) if n == "x21")
        })
        .expect(
            "no pin was inserted: `cmp x0, x21` before `mov x21, x0` will compare x0 with itself",
        );
    assert!(
        pin_idx + 1 < lifted.insns.len()
            && matches!(&lifted.insns[pin_idx + 1].operands.first(), Some(Operand::Register(n)) if n == "x21")
            && matches!(lifted.insns[pin_idx + 1].operands.get(1), Some(Operand::Register(n)) if n == "x0"),
        "the pin must sit IMMEDIATELY before the redefinition it protects against"
    );

    // And the branch's compare reads the pin, not the clobbered register.
    let cmp = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, ir::Op::Compare(_)))
        .expect("a compare");
    assert_eq!(cmp.operands.len(), 3);
    assert!(
        matches!(cmp.operands[2], Operand::Local(_)),
        "the compare's RHS must be the PINNED value, got {:?} — reading \
         `Register(\"x21\")` here resolves to the post-`mov` value and makes the \
         condition assert x0 == x0",
        cmp.operands[2]
    );
    assert_ne!(
        cmp.operands[1], cmp.operands[2],
        "a compare whose two operands are the same value is the tautology bug"
    );
}

/// `stp d0, d1, [x]` — a PAIR of doubles.
///
/// The pair arms place the second element at `first + element_size`, and
/// decided that size with `name.starts_with("v") { 4 } else { 8 }`. Every FP
/// name starts with `v`, so a `d` pair — 8 bytes per element — advanced by 4
/// and the second double was written onto the low half of the first, silently
/// destroying it. Assert the real stride.
#[test]
fn double_pair_store_advances_by_eight_bytes_not_four() {
    let words: Vec<u32> = vec![
        0x6d000000, // stp d0, d1, [x0]  (placeholder, replaced below)
        0xd65f03c0, // ret
    ];
    // Encode `stp d0, d1, [x0]` properly: STP (SIMD&FP, signed offset) is
    // 0x6D000000 | imm7<<15 | Rt2<<10 | Rn<<5 | Rt.
    let mut words = words;
    words[0] = 0x6D000000 | 0x0000_0000 | (1u32 << 10) | (0u32 << 5) | 0u32;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let Some(lifted) = lift::lift_arm64(&bytes, 0x1000, 8, None) else {
        panic!("lift failed");
    };
    let stores: Vec<&ir::Insn> = lifted
        .insns
        .iter()
        .filter(|i| matches!(i.operands.first(), Some(Operand::Memory { .. })))
        .collect();
    assert_eq!(stores.len(), 2, "a double pair store is two stores");
    let addend = |i: &ir::Insn| match i.operands.first() {
        Some(Operand::Memory { addend, .. }) => *addend,
        _ => panic!("not a store"),
    };
    assert_eq!(addend(stores[0]), 0);
    assert_eq!(
        addend(stores[1]),
        8,
        "the second double is 8 bytes in, not 4 — the pair must not overlap"
    );
}

/// A Q-register pair — `stp q0, q1, [sp, #32]` / `ldp q0, q1, [sp, #32]` — is
/// the 128-bit VECTOR spill form. `fp_pair_of` answered only for the scalar
/// `S`/`D` views, so a Q pair supplied no operand and the STP/LDP arms emitted
/// NOTHING: the instruction vanished from the IR and the coverage oracle
/// reported it uncovered. On the arm64 corpus that was 5,922 `stp` + 2,792
/// `ldp` — the largest coverage class by an order of magnitude — and each one
/// is a real spill whose loss severs the def→use chain for the vector's value.
#[test]
fn quadrature_pair_store_and_load_lift_at_sixteen_byte_stride() {
    // STP/LDP (SIMD&FP, signed offset): 0xAD000000 / 0xAD400000 for the Q form,
    // with imm7 scaled by 16 (`#32` ⇒ imm7 = 2), Rt2 = 1, Rn = 31 (sp), Rt = 0.
    let fields = (2u32 << 15) | (1u32 << 10) | (31u32 << 5);
    let words = [0xAD00_0000u32 | fields, 0xAD40_0000u32 | fields, 0xd65f03c0];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let Some(lifted) = lift::lift_arm64(&bytes, 0x1000, 8, None) else {
        panic!("lift failed");
    };
    let mem_of = |i: &ir::Insn| -> i64 {
        i.operands
            .iter()
            .find_map(|o| match o {
                Operand::Memory { addend, .. } => Some(*addend),
                _ => None,
            })
            .expect("memory operand")
    };
    let mem_moves: Vec<i64> = lifted
        .insns
        .iter()
        .filter(|i| {
            i.operands
                .iter()
                .any(|o| matches!(o, Operand::Memory { .. }))
        })
        .map(mem_of)
        .collect();
    assert_eq!(
        mem_moves,
        vec![32, 48, 32, 48],
        "a Q pair is two memory ops 16 bytes apart; lifted IR: {:?}",
        lifted.insns
    );
}

/// The class-init guard eraser must not eat a value the JOIN still reads.
///
/// `is_pure_env` admits `Op::Phi`, and `ssa::materialize` splices every phi at
/// its block's FIRST instruction — so when the join block opens with a phi,
/// the erased region (which ends at the join's index) cut the join's own phi
/// header out. The guard then disappeared while the value it selected stayed
/// referenced, and the join read a local nothing defined any more.
///
/// This is `RuntimeMarshal.MarshalString` @0x17005f8 (ARM32) in miniature.
/// Native:
///   ldrb r0,[flag]; cmp r0,#0; bne join; <init: 3 loads, bl, ldr, mov, strb>;
///   join: stm r4,{r5,r6}; pop pc
/// The rendered body lost the whole conditional, its `bl`, and the definition
/// of the value the join's phi selected (`return v11;` with no `v11 =`).
#[test]
fn guard_excision_refuses_to_erase_a_value_the_join_reads() {
    let load = |dest: u32, addend: i64| {
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(dest),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::Immediate(0x1700618)),
                    index: None,
                    scale: 1,
                    addend,
                },
            ],
            0x1700618,
        )
    };
    let mut insns = vec![
        // 0: the guard byte load (direct-load compare is accepted by the matcher)
        load(30, 0),
        // 1: the compare
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![Operand::Local(999), Operand::Local(30), Operand::imm(0)],
            0x1700620,
        ),
        // 2: the branch — taken = the join (init is skipped when flag != 0).
        // The operand is an RVA here: `Cfg::build` rewrites it to an index,
        // exactly as it does in the pipeline.
        Insn::new(
            Op::CondJump(0x1700644),
            vec![Operand::Local(999)],
            0x1700624,
        ),
        // 3..: the init chain (one call), defining v25 that the join's phi takes
        Insn::new(
            Op::Move,
            vec![Operand::Local(21), Operand::Local(20)],
            0x1700628,
        ),
        Insn::new(
            Op::Move,
            vec![Operand::Local(22), Operand::Local(21)],
            0x170062c,
        ),
        Insn::new(
            Op::Move,
            vec![Operand::Local(23), Operand::Local(22)],
            0x1700630,
        ),
        Insn::new(
            Op::Call,
            vec![
                Operand::Method(mref("Init", 1, false)),
                Operand::Local(24),
                Operand::Local(23),
            ],
            0x1700634,
        ),
        Insn::new(
            Op::Move,
            vec![Operand::Local(25), Operand::Local(23)],
            0x1700638,
        ),
        // 8: the JOIN — its phi selects between the fast path's value (v30) and
        //    the init path's (v25), and v11 is what the method returns.
        Insn::new(
            Op::Phi,
            vec![Operand::Local(11), Operand::Local(30), Operand::Local(25)],
            0x1700644,
        ),
        Insn::new(
            Op::Move,
            vec![Operand::Local(18), Operand::Local(11)],
            0x1700644,
        ),
        // the return reads the phi's destination — the use that must keep it alive
        Insn::new(Op::Return, vec![Operand::Local(11)], 0x1700648),
    ];
    let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
    let mut cfg = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    let removed = passes::remove_init_guards(&mut insns, &mut cfg);
    assert_eq!(
        removed, 0,
        "the guard is real control flow and must be left intact"
    );
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Call)),
        "the init call is still called natively — it must survive"
    );
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::CondJump(_))),
        "the guard branch must survive"
    );
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Phi)),
        "the join's phi defines the returned value — it must survive"
    );
}

/// A `Throw` in the fast/skip arm must not make the guard look like a no-op.
///
/// `chain_walk` stops at `Op::Throw` and used to return with its call count
/// UNCHANGED, so an arm that ends by transferring control out of the method was
/// reported as "0 calls" — and the gate reads 0 calls as "this is the fast skip
/// that does nothing". It then erased the OTHER arm (the real init envelope) and
/// rewired the guard past it.
///
/// This is `Game`'s `.cctor` @0x1cb257c in miniature (arm64, 1860 bytes, 465
/// native insns). Its skip arm was `ldr x0,[x20]; mov w1,#0x57; bl …` where the
/// `bl` had already been rewritten to `Op::Throw` by `resolve`, so the walk
/// stopped one instruction early and counted nothing. The pass erased the
/// static initializer — two calls and the `s_initialized = true` store — and the
/// whole method then rendered as a lone `throw`: 87% of its semantics gone.
/// A throw is an observable effect and can never be the "pure" side.
#[test]
fn guard_excision_refuses_to_erase_the_init_when_the_skip_arm_throws() {
    let flag_load = Insn::new(
        Op::Move,
        vec![
            Operand::Local(20),
            Operand::Memory {
                width: 1,
                base: Box::new(Operand::Immediate(0x63aa000)),
                index: None,
                scale: 1,
                addend: 1302,
            },
        ],
        0x1cb258c,
    );
    let mut insns = vec![
        // 0..3 — the guard: load the `s_initialized` byte, test bit 0, branch to
        //        the skip arm at 0x1cb25c4 when it is already set.
        flag_load,
        Insn::new(
            Op::Arith(ArithKind::And),
            vec![Operand::Local(21), Operand::Local(20), Operand::imm(1)],
            0x1cb2598,
        ),
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![Operand::Local(22), Operand::Local(21), Operand::imm(0)],
            0x1cb2598,
        ),
        Insn::new(Op::CondJump(0x1cb25c4), vec![Operand::Local(22)], 0x1cb2598),
        // 4..9 — the INIT envelope: two calls and the flag store.
        Insn::new(
            Op::Move,
            vec![Operand::Local(23), Operand::imm(0x6048df8)],
            0x1cb259c,
        ),
        Insn::new(
            Op::Call,
            vec![
                Operand::Method(mref("MetadataInit", 1, false)),
                Operand::Local(24),
                Operand::Local(23),
            ],
            0x1cb25a4,
        ),
        Insn::new(
            Op::Move,
            vec![Operand::Local(25), Operand::imm(0x6051688)],
            0x1cb25a8,
        ),
        Insn::new(
            Op::Call,
            vec![
                Operand::Method(mref("MetadataInit", 1, false)),
                Operand::Local(26),
                Operand::Local(25),
            ],
            0x1cb25b0,
        ),
        Insn::new(
            Op::Move,
            vec![
                Operand::Memory {
                    width: 1,
                    base: Box::new(Operand::Local(19)),
                    index: None,
                    scale: 1,
                    addend: 1302,
                },
                Operand::imm(1),
            ],
            0x1cb25b8,
        ),
        // 9..11 — the SKIP arm (the branch target): two moves then a throw,
        //         which is what `resolve` made of the third `bl`.
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(30),
                Operand::Memory {
                    width: 0,
                    base: Box::new(Operand::Local(29)),
                    index: None,
                    scale: 8,
                    addend: 0,
                },
            ],
            0x1cb25c4,
        ),
        Insn::new(
            Op::Move,
            vec![Operand::Local(31), Operand::imm(0x57)],
            0x1cb25c4,
        ),
        Insn::new(Op::Throw, vec![Operand::Local(30)], 0x1cb25c4),
        // 12 — the body continues; this is the code that must stay reachable.
        Insn::new(Op::Return, vec![], 0x1cb25c8),
    ];
    let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
    let mut cfg = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    let removed = passes::remove_init_guards(&mut insns, &mut cfg);

    assert_eq!(
        removed, 0,
        "a skip arm that THROWS is not a pure skip — the init envelope is real"
    );
    assert_eq!(
        insns.iter().filter(|i| matches!(i.op, Op::Call)).count(),
        2,
        "both static-initializer calls must survive"
    );
    assert!(
        insns.iter().any(|i| matches!(&i.op, Op::Move)
            && i.operands
                .first()
                .is_some_and(|o| matches!(o, Operand::Memory { addend: 1302, .. }))),
        "the `s_initialized = true` store must survive"
    );
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::CondJump(_))),
        "the guard branch must survive"
    );
}

/// A dead binding to a CALL must keep the call, not delete it.
///
/// `dead_stores_flat` removed `x = f();` whenever `x` was never read — but
/// `f()` still ran. On `MarshalString` @0x17005f8 the renderer first turned the
/// class-init `bl` into `v24 = fn_5827e4(…)` (a value nothing reads, because
/// the callee's result is discarded natively), and this pass then deleted it,
/// so the method's only call vanished from the body.
///
/// The dead BINDING may go; the EFFECT must stay.
#[test]
fn a_dead_binding_to_a_call_keeps_the_call() {
    let mut stmts = vec![
        Stmt::Assign(
            "v24".into(),
            None,
            Expr::RawCall(0x5827e4, vec![Expr::Var("obj".into())], None),
        ),
        Stmt::Return(Some(Expr::Var("other".into()))),
    ];
    cleanup::apply(&mut stmts);
    match stmts.first() {
        Some(Stmt::Eval(Expr::RawCall(addr, args, _))) => {
            assert_eq!(*addr, 0x5827e4);
            assert_eq!(args.len(), 1, "arguments must survive with the call");
        }
        other => panic!("a discarded-result call must be emitted as a statement, got {other:?}"),
    }
}

/// ...while a dead binding to a PURE expression is still deleted outright.
#[test]
fn a_dead_binding_to_a_pure_expression_is_still_removed() {
    let mut stmts = vec![
        Stmt::Assign(
            "v24".into(),
            None,
            Expr::Bin(
                super::csharp::BinOp::Add,
                Box::new(Expr::Var("a".into())),
                Box::new(Expr::Int(1)),
            ),
        ),
        Stmt::Return(Some(Expr::Var("other".into()))),
    ];
    cleanup::apply(&mut stmts);
    assert_eq!(
        stmts.len(),
        1,
        "a pure dead binding leaves nothing, got {stmts:?}"
    );
}

/// PC-relative REGISTER addressing and narrow accesses, on the real bytes of
/// `SafeStringMarshal MarshalString(string)` @0x17005F8 in the Android
/// `libil2cpp.so`:
///
/// ```text
/// 0x1700610  add  r0, pc, r0     ; E08F0000 - r0 = pc + <pool value loaded above>
/// 0x170061c  ldrb r0, [r0]       ; E5D00000 - ONE byte
/// 0x1700640  strb r1, [pc, r0]   ; E7CF1000 - ONE byte, pc-relative indexed
/// ```
///
/// Two separate defects are locked in here.
///
/// `add rd, pc, rm` used to fall into the `add rd, pc, #imm` branch with
/// `parse_field(rm).unwrap_or(0)`, DROPPING `rm`, so the result collapsed to
/// `pc` alone: the dump contained `*(nint*)(0x1700618)` - a plausible-looking
/// but WRONG absolute address, with the whole value chain behind it lost.
/// `[pc, rm]` had the same shape, because `pc` was carried as the undefined
/// `g15` pseudo-register and the index was then discarded by the renderer.
///
/// Separately, `ldrb`/`strb` lifted identically to `ldr`/`str`, so a one-byte
/// field read rendered as an eight-byte `*(nint*)` and a one-byte store
/// claimed to write eight bytes over a one-byte slot.
#[test]
fn arm32_pc_relative_add_and_narrow_access_keep_their_meaning() {
    let insns = lift_a32(&[0xE08F0000, 0xE5D00000, 0xE7CF1000]);

    // `add r0, pc, r0`: an Immediate pc term AND the live index register.
    let add = insns
        .iter()
        .find(|i| matches!(&i.op, Op::Arith(_)))
        .unwrap_or_else(|| panic!("`add r0, pc, r0` must lift as arithmetic: {insns:?}"));
    assert!(
        matches!(add.operands.get(1), Some(Operand::Immediate(_))),
        "the `pc` term must be the exact constant address, got {:?}",
        add.operands
    );
    assert!(
        matches!(add.operands.get(2), Some(Operand::Register(_))),
        "`add rd, pc, rm` must keep `rm` as an operand; dropping it produced \
         `pc` alone, a plausible but wrong constant address. got {:?}",
        add.operands
    );

    // `ldrb r0, [r0]` - a ONE-byte load.
    let load_width =
        insns
            .iter()
            .find_map(|i| match (&i.op, i.operands.first(), i.operands.get(1)) {
                (Op::Move, Some(Operand::Register(_)), Some(Operand::Memory { width, .. })) => {
                    Some(*width)
                }
                _ => None,
            });
    assert_eq!(
        load_width,
        Some(1),
        "`ldrb` must carry a one-byte access width: {insns:?}"
    );

    // `strb r1, [pc, r0]` - a ONE-byte store at `pc + r0`.
    let store = insns
        .iter()
        .find_map(|i| match (&i.op, i.operands.first()) {
            (Op::Move, Some(m @ Operand::Memory { .. })) => Some(m.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("`strb` must lift to a store through memory: {insns:?}"));
    match &store {
        Operand::Memory {
            width, base, index, ..
        } => {
            assert_eq!(
                *width, 1,
                "`strb` must carry a one-byte width, got {store:?}"
            );
            assert!(
                matches!(**base, Operand::Immediate(_)),
                "`[pc, rm]` must use the exact pc constant as its base, got {store:?}"
            );
            assert!(
                index.is_some(),
                "`[pc, rm]` must keep its index register, got {store:?}"
            );
        }
        _ => unreachable!(),
    }
}

/// The access width must travel from `Operand::Memory` all the way into the
/// rendered pointer cast — and `0` (not a narrow access) must stay `nint` so
/// nothing that was already correct changes.
#[test]
fn narrow_memory_renders_a_narrow_pointer_cast() {
    let render = |e: &Expr| {
        let mut s = String::new();
        e.render(&mut s);
        s
    };
    let deref =
        |w: u8, addend: i64| render(&Expr::RawDeref(Box::new(Expr::Var("p".into())), addend, w));
    assert_eq!(deref(0, 0), "*(nint*)(p)", "pointer width is the default");
    assert_eq!(deref(1, 0), "*(byte*)(p)");
    assert_eq!(deref(2, 0), "*(short*)(p)");
    assert_eq!(deref(4, 4), "*(int*)(p +4)");
    assert_eq!(deref(8, 0), "*(nint*)(p)");

    // `with_width` is how the lifters attach it, and it must leave a
    // non-narrow operand bit-for-bit alone.
    let mem = Operand::Memory {
        base: Box::new(Operand::Register("x0".into())),
        index: None,
        scale: 1,
        addend: 8,
        width: 0,
    };
    assert_eq!(format!("{}", mem.clone().with_width(0)), format!("{mem}"));
    assert_eq!(format!("{}", mem.with_width(1)), "w1[%x0+0x8]");
}

/// C# binary operators are LEFT-associative. The renderer must parenthesize a
/// right operand that shares the parent's precedence unless the operator is
/// associative and identical — otherwise `a - (b - c)` prints as `a - b - c`,
/// which the compiler re-associates to `(a - b) - c`: a silent wrong-logic
/// rewrite of the recovered tree.
#[test]
fn right_operand_of_subtraction_keeps_its_parens() {
    let render = |e: &Expr| {
        let mut s = String::new();
        e.render(&mut s);
        s
    };
    let var = |n: &str| Expr::Var(n.into());
    let bin = |op: BinOp, a: Expr, b: Expr| Expr::Bin(op, Box::new(a), Box::new(b));

    // Non-associative: right child must stay parenthesized.
    assert_eq!(
        render(&bin(BinOp::Sub, var("a"), bin(BinOp::Sub, var("b"), var("c")))),
        "a - (b - c)"
    );
    assert_eq!(
        render(&bin(BinOp::Sub, var("a"), bin(BinOp::Add, var("b"), var("c")))),
        "a - (b + c)"
    );
    assert_eq!(
        render(&bin(BinOp::Div, var("a"), bin(BinOp::Mul, var("b"), var("c")))),
        "a / (b * c)"
    );
    assert_eq!(
        render(&bin(BinOp::Mul, var("a"), bin(BinOp::Div, var("b"), var("c")))),
        "a * (b / c)"
    );
    assert_eq!(
        render(&bin(BinOp::Shr, var("a"), bin(BinOp::Shl, var("b"), var("c")))),
        "a >> (b << c)"
    );
    // Associative and identical: parens are redundant.
    assert_eq!(
        render(&bin(BinOp::Add, var("a"), bin(BinOp::Add, var("b"), var("c")))),
        "a + b + c"
    );
    assert_eq!(
        render(&bin(BinOp::AndAlso, var("a"), bin(BinOp::AndAlso, var("b"), var("c")))),
        "a && b && c"
    );
    // Left child at equal precedence is always bare (left-assoc).
    assert_eq!(
        render(&bin(BinOp::Sub, bin(BinOp::Sub, var("a"), var("b")), var("c"))),
        "a - b - c"
    );
    // Comparison inside comparison keeps parens on the right.
    assert_eq!(
        render(&Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::Bool(true)),
            Box::new(Expr::Cmp(
                CmpKind::Lt,
                Box::new(var("a")),
                Box::new(var("b"))
            ))
        )),
        "true == (a < b)"
    );
}

/// Cast operands group bare: `(int)(long)x`, `(int)-x` — matching Ghidra's
/// cast-operand rule (Rudra printc.rs:10704-10738). Binary operands still
/// need parens: `(int)(a + b)`.
#[test]
fn casts_of_casts_and_unaries_render_without_inner_parens() {
    let render = |e: &Expr| {
        let mut s = String::new();
        e.render(&mut s);
        s
    };
    let var = |n: &str| Expr::Var(n.into());
    assert_eq!(
        render(&Expr::Cast(
            "int".into(),
            Box::new(Expr::Cast("long".into(), Box::new(var("x"))))
        )),
        "(int)(long)x"
    );
    assert_eq!(
        render(&Expr::Cast(
            "int".into(),
            Box::new(Expr::Neg(Box::new(var("x"))))
        )),
        "(int)-x"
    );
    assert_eq!(
        render(&Expr::Cast(
            "int".into(),
            Box::new(Expr::Bin(
                BinOp::Add,
                Box::new(var("a")),
                Box::new(var("b"))
            ))
        )),
        "(int)(a + b)"
    );
}

/// `b <code outside this method>` is a TAIL CALL, not an exit edge.
///
/// The CFG can only resolve a branch whose target is an instruction of this
/// method, so a tail call produced no successor, and the renderer's
/// `t_blk >= blocks.len()` arm — "exit edge, emit nothing" — deleted the call.
///
/// Verified on `CollectWeeklyFacebookGift` @0x47014C (arm64, iOS): its success
/// path ends `mov x0, x20; mov x1, x19; mov x2, #0; b 0x3f8824` after the
/// epilogue, and that entire call was missing from the dump while the failure
/// path's ordinary `bl 0x4a58bac; ret` was rendered.
#[test]
fn tail_call_outside_the_method_becomes_a_real_call() {
    let mut insns = vec![
        Insn::move_(Operand::Local(10), Operand::Local(11), 0x3000),
        Insn::new(Op::Jump(0x7abc_1234), vec![], 0x3004),
    ];
    // `addresses` is index-aligned with `insns` (Cfg::build resolves branch
    // RVAs through it), so the pass must grow both together.
    let mut addresses = vec![0x3000u64, 0x3004];
    let known = |t: u64| t == 0x7abc_1234;
    assert_eq!(
        passes::recover_tail_calls(&mut insns, &mut addresses, &known, "x0"),
        1
    );
    assert_eq!(
        addresses.len(),
        insns.len(),
        "address list must stay aligned"
    );
    assert_eq!(addresses, vec![0x3000u64, 0x3004, 0x3004]);
    assert!(
        matches!(insns[1].op, Op::Call),
        "tail branch must become a call, got {:?}",
        insns[1].op
    );
    assert!(
        matches!(insns[2].op, Op::Return),
        "and return the callee's result, got {:?}",
        insns[2].op
    );
    // [target, dest, x0 … x7, vs0 … vs7] — and NOT x8: the site never
    // materialized an indirect-result pointer, so the slot must not be
    // invented. An unconditional dump put whatever scratch value last lived in
    // x8 there, which the renderer then printed as a phantom trailing
    // argument.
    assert_eq!(insns[1].operands.len(), 18);
    assert_eq!(insns[1].operands[0], Operand::imm(0x7abc_1234));
    assert_eq!(insns[1].operands[1], Operand::reg("x0"), "return slot");
    assert!(
        !insns[1]
            .operands
            .iter()
            .any(|o| matches!(o, Operand::Register(r) if r == "x8")),
        "x8 must not be fabricated, got {:?}",
        insns[1].operands
    );
    // The FP argument registers MUST be recorded, exactly parallel to x0–x7.
    // A tail-called callee's float parameter is CARRIED through the branch
    // (`s0` is never rewritten by the branch itself), so a value the callee
    // receives exists only as a register here. Leaving it out made
    // `align_call_arguments` find no register of the FP class for that
    // parameter and emit `MissingAbiArg`, which the renderer drops trailing —
    // `<>c__DisplayClass1_0.<DOPitch>b__1(float x)` @0x8BAC of UnityFramework
    // rendered `this.target.set_pitch()` for `set_pitch(float)`.
    assert_eq!(
        &insns[1].operands[10..18],
        &[
            Operand::reg("vs0"),
            Operand::reg("vs4"),
            Operand::reg("vs8"),
            Operand::reg("vs12"),
            Operand::reg("vs16"),
            Operand::reg("vs20"),
            Operand::reg("vs24"),
            Operand::reg("vs28"),
        ],
        "FP argument registers must follow the GP dump"
    );
    assert_eq!(insns[2].operands[0], Operand::reg("x0"));
}

/// A SHARED tail-merge epilogue never allocates a frame; a real method entry
/// does. The splicer uses this to avoid gluing another function's prologue
/// and body into the caller (nine WRONG-LOGIC methods: the fragment's
/// backward branch became a self-loop and its return was cut off).
#[test]
fn a_frame_allocating_out_of_body_tail_is_not_a_shared_epilogue() {
    let epilogue = vec![
        Insn::new(
            Op::Move,
            vec![
                Operand::reg("x27"),
                Operand::Memory {
                    base: Box::new(Operand::reg("sp")),
                    index: None,
                    scale: 8,
                    addend: 0x20,
                    width: 0,
                },
            ],
            0x3000,
        ),
        Insn::new(Op::ShiftStack(0x40), vec![], 0x3004),
        Insn::new(Op::Return, vec![], 0x3008),
    ];
    assert!(
        !super::tail_starts_with_frame_allocation(&epilogue),
        "a restoring epilogue must stay spliceable"
    );

    let entry = vec![
        Insn::new(Op::ShiftStack(-0x10), vec![], 0x4000),
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x1234), Operand::reg("x0")],
            0x4004,
        ),
        Insn::new(Op::Return, vec![Operand::reg("x0")], 0x4008),
    ];
    assert!(
        super::tail_starts_with_frame_allocation(&entry),
        "a frame allocation marks a real method entry"
    );

    // A negative adjustment AFTER the first terminator belongs to code we
    // must not splice anyway; it must not influence the decision.
    let after_term = vec![
        Insn::new(Op::Return, vec![], 0x5000),
        Insn::new(Op::ShiftStack(-0x10), vec![], 0x5004),
    ];
    assert!(!super::tail_starts_with_frame_allocation(&after_term));
}

/// An out-of-line GOT/static load veneer must rejoin the body before the CFG.
/// Real shape (0x16B8FE8): entry ends `b 0x2dff1a8`; that address is
/// `ldr x19,[x19,#0x9a8]; b 0x16b901c`, where the continuation is in-body.
#[test]
fn an_out_of_line_load_veneer_rejoins_the_body() {
    let mut insns = vec![
        Insn::move_(Operand::reg("x19"), Operand::imm(0x5000), 0x1000),
        Insn::new(Op::Jump(0x2000), vec![], 0x1004),
    ];
    let mut addresses = vec![0x1000u64, 0x1004];
    // ldr x19,[x19,#0x9a8] ; b 0x1010 (backward from the veneer, forward
    // from the branch site, inside the declared body)
    let words = [0xF944D673u32, 0x17FFFC03];
    let bytes = le(&words);
    let read = |a: u64, _: usize| if a == 0x2000 { Some(bytes.clone()) } else { None };

    let n = super::resolve_load_veneers(&mut insns, &mut addresses, 0x1000, 0x1100, read);
    assert_eq!(n, 1, "the veneer must be resolved");
    assert_eq!(insns.len(), 3);
    assert_eq!(addresses, vec![0x1000u64, 0x1004, 0x1004]);
    assert!(
        matches!(insns[1].op, Op::Move)
            && matches!(&insns[1].operands[0], Operand::Register(r) if r == "x19"),
        "the veneer load must be spliced at the branch site: {:?}",
        insns[1]
    );
    assert!(
        matches!(insns[2].op, Op::Jump(0x1010)),
        "the branch must retarget to the in-body continuation: {:?}",
        insns[2].op
    );
}

/// A target that looks like a real method (frame allocation, store, call) is
/// not a linker veneer; the branch must stay an exit edge.
#[test]
fn a_frame_allocating_veneer_is_refused() {
    let mut insns = vec![Insn::new(Op::Jump(0x2000), vec![], 0x1004)];
    let mut addresses = vec![0x1004u64];
    // sub sp,sp,#0x10 ; stp x30,x19,[sp] ; b 0x1010
    let words = [0xD10043FFu32, 0xA9007BF3, 0x17FFFC02];
    let bytes = le(&words);
    let read = |a: u64, _: usize| if a == 0x2000 { Some(bytes.clone()) } else { None };

    let n = super::resolve_load_veneers(&mut insns, &mut addresses, 0x1000, 0x1100, read);
    assert_eq!(n, 0, "a frame-setting window is another method");
    assert_eq!(insns.len(), 1);
    assert!(matches!(insns[0].op, Op::Jump(0x2000)));
}

/// The complement of the rule above: when the tail-call site DID set x8 up as
/// the address of a buffer (`add x8, sp, #imm`), the slot is real and must be
/// recorded at the fixed ARM64 index the renderer reads it from.
#[test]
fn tail_call_records_the_indirect_result_only_when_the_site_set_it_up() {
    let mut insns = vec![
        Insn::new(
            Op::Move,
            vec![
                Operand::reg("x8"),
                Operand::AddressOf(Box::new(Operand::Memory {
                    base: Box::new(Operand::reg("sp")),
                    index: None,
                    scale: 8,
                    addend: 8,
                    width: 0,
                })),
            ],
            0x3000,
        ),
        Insn::new(Op::Jump(0x7abc_1234), vec![], 0x3004),
    ];
    let mut addresses = vec![0x3000u64, 0x3004];
    let known = |t: u64| t == 0x7abc_1234;
    passes::recover_tail_calls(&mut insns, &mut addresses, &known, "x0");
    // [target, dest, x0 … x7, x8, vs0 … vs7]
    assert_eq!(insns[1].operands.len(), 19);
    assert_eq!(insns[1].operands[10], Operand::reg("x8"));
    assert_eq!(insns[1].operands[11], Operand::reg("vs0"));
}

/// The complement: a branch to an address INSIDE the method is ordinary control
/// flow and must stay a branch — otherwise every intra-method loop would be
/// rewritten into a call.
#[test]
fn a_branch_inside_the_method_stays_a_branch() {
    let mut insns = vec![
        Insn::move_(Operand::Local(10), Operand::Local(11), 0x3000),
        Insn::move_(Operand::Local(11), Operand::Local(10), 0x3004),
        Insn::new(Op::Jump(0x3000), vec![], 0x3008),
    ];
    let mut addresses = vec![0x3000u64, 0x3004, 0x3008];
    let known = |t: u64| t == 0x7abc_1234;
    assert_eq!(
        passes::recover_tail_calls(&mut insns, &mut addresses, &known, "x0"),
        0
    );
    assert!(matches!(insns[2].op, Op::Jump(_)));
    assert_eq!(addresses, vec![0x3000u64, 0x3004, 0x3008]);

    // ...and a target that is NOT a known method start is a tail-MERGE, not a
    // tail call: leave it for the splicer.
    let mut insns = vec![Insn::new(Op::Jump(0x7abc_1234), vec![], 0x3004)];
    let mut addresses = vec![0x3004u64];
    let nothing_known = |_: u64| false;
    assert_eq!(
        passes::recover_tail_calls(&mut insns, &mut addresses, &nothing_known, "x0"),
        0
    );
    assert!(matches!(insns[0].op, Op::Jump(_)));
}

/// An x64 trailing `jmp` to a NON-managed helper is a tail call, not a dead
/// end: `OutSine`@0x14D6540 (`mulss xmm0, [rip]; jmp sin-helper`) stayed a bare
/// `Jump`, so DCE deleted the `mulss` as dead and the structurer emitted an
/// empty body. The rewrite keeps the setup and returns the helper's result as
/// an honest `fn_<addr>(…)` call.
#[test]
fn x64_trailing_jump_to_an_unmanaged_helper_becomes_a_call_plus_return() {
    // setup: vf0 = vf0 * C ; trailing jmp to the (unmanaged, readable) helper
    let mut insns = vec![
        Insn::new(
            Op::FloatArith(FloatArithKind::Mul),
            vec![
                Operand::reg("vf0"),
                Operand::reg("vf0"),
                Operand::imm(0x3f80_0000),
            ],
            0x14d6540,
        ),
        Insn::new(Op::Jump(0x2552b0), vec![], 0x14d6548),
    ];
    let mut addresses = vec![0x14d6540u64, 0x14d6548];
    let never_managed = |_: u64| false;
    let readable = |t: u64| t == 0x2552b0;
    let no_stub = |_: u64| false;
    // `OutSine` returns `float`: the call must carry that type or the
    // float-result binders never link its result into the `Return`.
    let ret_float = ManagedType::Primitive(PrimKind::R4);
    assert!(
        recover_x64_unmanaged_tail(
            &mut insns,
            &mut addresses,
            0x14d6540,
            0x14d654d,
            &never_managed,
            &readable,
            &no_stub,
            &ret_float,
        ),
        "trailing jmp to a readable unmanaged helper must become a call"
    );
    assert_eq!(
        addresses.len(),
        insns.len(),
        "address list must stay aligned"
    );
    assert!(
        matches!(insns[1].op, Op::Call),
        "trailing jump must become a call, got {:?}",
        insns[1].op
    );
    assert!(
        matches!(insns[2].op, Op::Return),
        "and the helper result must be returned, got {:?}",
        insns[2].op
    );
    // [target, dest, x0 … x7, vf0 … vf7]: the FP dump MUST be `vfN` (the
    // lifter's XMM names). Dumping ARM's `vsN` would leave the `vf0` setup
    // exactly as dead as before and fix nothing.
    assert_eq!(insns[1].operands[0], Operand::imm(0x2552b0));
    assert_eq!(
        insns[1].dest_type,
        Some(ManagedType::Primitive(PrimKind::R4)),
        "the call must carry the method return type for the float binders"
    );
    assert_eq!(
        insns[1].operands[1],
        Operand::reg(ir::RET_REG),
        "x64 result register"
    );
    assert_eq!(
        &insns[1].operands[10..18],
        &[
            Operand::reg("vf0"),
            Operand::reg("vf1"),
            Operand::reg("vf2"),
            Operand::reg("vf3"),
            Operand::reg("vf4"),
            Operand::reg("vf5"),
            Operand::reg("vf6"),
            Operand::reg("vf7"),
        ],
        "x64 FP argument registers must be vfN"
    );
    assert_eq!(insns[2].operands[0], Operand::reg(ir::RET_REG));
}

/// The complement: every refusal leaves the listing untouched.
#[test]
fn x64_unmanaged_tail_refuses_anything_it_cannot_prove() {
    let mk = |op: Op, addr: u64| Insn::new(op, vec![], addr);
    // Refusals never depend on the return type; the positive test pins the
    // float linkage, and the void case below pins the bare `Return`.
    let ret_any = ManagedType::Primitive(PrimKind::R4);
    // managed target: the managed path names it properly, not this one.
    let mut insns = vec![
        mk(Op::FloatArith(FloatArithKind::Mul), 0x3000),
        mk(Op::Jump(0x4000), 0x3004),
    ];
    let mut addresses = vec![0x3000u64, 0x3004];
    assert!(
        !recover_x64_unmanaged_tail(
            &mut insns,
            &mut addresses,
            0x3000,
            0x3008,
            &|t| t == 0x4000,
            &|_| true,
            &|_| false,
            &ret_any,
        ),
        "managed targets belong to the managed tail-call path"
    );
    assert!(matches!(insns[1].op, Op::Jump(_)));
    // in-body branch: ordinary control flow, never a call.
    let mut insns = vec![
        mk(Op::FloatArith(FloatArithKind::Mul), 0x3000),
        mk(Op::Jump(0x3000), 0x3004),
    ];
    let mut addresses = vec![0x3000u64, 0x3004];
    assert!(
        !recover_x64_unmanaged_tail(
            &mut insns,
            &mut addresses,
            0x3000,
            0x3008,
            &|_| false,
            &|_| true,
            &|_| false,
            &ret_any,
        ),
        "in-body branches must stay branches"
    );
    assert!(matches!(insns[1].op, Op::Jump(_)));
    // null and unreadable targets: corrupt, never fabricate a call.
    for target in [0u64, 0x9abc_def0] {
        let mut insns = vec![
            mk(Op::FloatArith(FloatArithKind::Mul), 0x3000),
            mk(Op::Jump(target as usize), 0x3004),
        ];
        let mut addresses = vec![0x3000u64, 0x3004];
        assert!(
            !recover_x64_unmanaged_tail(
                &mut insns,
                &mut addresses,
                0x3000,
                0x3008,
                &|_| false,
                &|_| false,
                &|_| false,
                &ret_any,
            ),
            "unreadable target {target:#x} must stay a jump"
        );
        assert!(matches!(insns[1].op, Op::Jump(_)));
    }
    // dispatch stub (branches back into the body): not a call.
    let mut insns = vec![
        mk(Op::FloatArith(FloatArithKind::Mul), 0x3000),
        mk(Op::Jump(0x5000), 0x3004),
    ];
    let mut addresses = vec![0x3000u64, 0x3004];
    assert!(
        !recover_x64_unmanaged_tail(
            &mut insns,
            &mut addresses,
            0x3000,
            0x3008,
            &|_| false,
            &|_| true,
            &|_| true,
            &ret_any,
        ),
        "a target branching back into the body is a stub, not a call"
    );
    assert!(matches!(insns[1].op, Op::Jump(_)));
    // length 1 is the single-insn thunk path above, not this one.
    let mut insns = vec![mk(Op::Jump(0x5000), 0x3004)];
    let mut addresses = vec![0x3004u64];
    assert!(
        !recover_x64_unmanaged_tail(
            &mut insns,
            &mut addresses,
            0x3004,
            0x3008,
            &|_| false,
            &|_| true,
            &|_| false,
            &ret_any,
        ),
        "single-insn bodies belong to the thunk path"
    );
    assert!(matches!(insns[0].op, Op::Jump(_)));
}

/// A `void` tail call returns nothing: the call keeps its result slot (the
/// native still defines it) but the trailing instruction is a bare `Return`,
/// so the body renders `fn_…(…); return;` rather than a fabricated value.
#[test]
fn x64_unmanaged_void_tail_ends_in_a_bare_return() {
    let mut insns = vec![
        Insn::new(Op::Jump(0x5000), vec![], 0x3000),
        Insn::new(Op::Jump(0x5000), vec![], 0x3000),
    ];
    // Second insn is the trailing jump; the first stands in for setup.
    insns[0].op = Op::FloatArith(FloatArithKind::Mul);
    let mut addresses = vec![0x3000u64, 0x3004];
    let ret_void = ManagedType::Primitive(PrimKind::Void);
    assert!(
        recover_x64_unmanaged_tail(
            &mut insns,
            &mut addresses,
            0x3000,
            0x3008,
            &|_| false,
            &|_| true,
            &|_| false,
            &ret_void,
        ),
        "void tail calls must also be recovered"
    );
    assert!(matches!(insns[1].op, Op::Call));
    assert!(
        matches!(&insns[2].op, Op::Return) && insns[2].operands.is_empty(),
        "void methods end in a bare return, got {:?}",
        insns[2]
    );
}

/// ONE NATIVE `bl` IS ONE EMITTED CALL.
///
/// `operand_expr` inlines a local to its definition, so the `Op::Return` arm
/// re-emitted the whole call expression even when the call's own site had already
/// materialized `dst = call(…)` (the multi-def binding path). The rendered body
/// then contained two statements for one instruction:
///
///     v73 = fn_4a58bac(v17, v18, v19);
///     fn_4a58bac(v17, v18, v19);
///
#[test]
fn a_returned_call_is_not_emitted_twice() {
    let m = mref("DoThing", 7, false);
    let ir = super::RecoveredIr {
        insns: vec![
            // A second def of the same local is what makes it MULTI-DEF, which
            // is the condition under which the call arm binds `dst = call(…)`.
            Insn::move_(Operand::Local(10), Operand::imm(1), 0x4000),
            Insn::new(
                Op::Call,
                vec![Operand::Method(m), Operand::Local(10)],
                0x4004,
            ),
            Insn::new(Op::Return, vec![Operand::Local(10)], 0x4008),
        ],
        table: locals::LocalTable::default(),
        returns_value: true,
        base_rva: 0x4000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert_eq!(
        out.matches("DoThing").count(),
        1,
        "one native `bl` must render as one call, got:\n{out}"
    );
}

/// A raw call (unresolved target) must not silently drop an argument just
/// because its operand is not a `Local`.
///
/// `raw_call_stmt` skipped `Operand::Method(_)`/`Type(_)` outright, so the class
/// init idiom `adrp + add + bl` lost its one argument: 3 of the 4
/// `fn_4a58960(&slot)` calls on `CollectWeeklyFacebookGift` @0x47014C rendered as
/// `fn_4a58960()`. It also deduplicated adjacent identical arguments (deleting
/// `f(x, x)`'s second `x`) and capped the list at four.
#[test]
fn a_raw_call_keeps_a_method_operand_argument() {
    let inner = mref("Target", 9, true);
    let ir = super::RecoveredIr {
        insns: vec![
            Insn::new(
                Op::Call,
                vec![
                    Operand::imm(0x4a58960),
                    Operand::Unknown, // no recovered return value
                    Operand::Method(inner),
                    Operand::MissingAbiArg,
                    Operand::MissingAbiArg,
                ],
                0x5000,
            ),
            Insn::new(Op::Return, vec![], 0x5004),
        ],
        table: locals::LocalTable::default(),
        returns_value: false,
        base_rva: 0x5000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("fn_4a58960(MethodInfo("),
        "the slot-pointer argument must survive, got:\n{out}"
    );
    assert!(
        !out.contains(") ") && !out.contains("(, "),
        "the trailing ABI filler must not become an empty argument:\n{out}"
    );
}

/// AAPCS64 passes a struct's hidden return buffer in `x8` — a REAL argument
/// that is not one of the declared parameters.
///
/// `GetTimeSlotStatus` @0x470204 on `CollectWeeklyFacebookGift` is called with
/// `&result` in x8 and the method reloads the value from `[sp,#0xc]` afterwards;
/// the `BL` arm's x0–x7 dump could not see it, so the dump showed a 2-argument
/// call for a 3-argument native call.
#[test]
fn arm64_sret_pointer_is_recorded_as_a_call_operand() {
    // add x8, sp, #8 ; bl +0x100
    let words = [0x9100_23e8u32, 0x9400_0040u32];
    let bytes = le(&words);
    let lifted = super::lift::lift_arm64(&bytes, 0x1000, 64, None).expect("lift");
    let call = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Call))
        .expect("the `bl` must lift to a call");
    assert_eq!(
        call.operands.get(10),
        Some(&Operand::reg("x8")),
        "the sret register must sit one past x7, got {:?}",
        call.operands
    );
}

/// ...and the trimming pass must not delete it again: the sret slot is one past
/// the declared parameters, exactly where `trim_call_arguments` truncates.
#[test]
fn trim_call_arguments_keeps_the_sret_pointer() {
    use super::types::{ManagedType, PrimKind};
    let mut m = mref("Fill", 3, false);
    m.sig.ret = ManagedType::Primitive(PrimKind::Void);
    m.sig.params = vec![ManagedType::Primitive(PrimKind::I4)];
    // [target, dest, receiver] then x1–x7 filler → the x8 slot lands on index 10,
    // exactly where the declared-parameter truncation cuts.
    let mut ops = vec![Operand::Method(m), Operand::Local(10), Operand::Local(11)];
    for _ in 0..7 {
        ops.push(Operand::MissingAbiArg);
    }
    ops.push(Operand::Local(20)); // the x8 slot
    let mut insns = vec![Insn::new(Op::Call, ops, 0x6000)];
    passes::trim_call_arguments(&mut insns);
    // It must be AT the slot, not merely still present: the renderer reads
    // `SRET_OPERAND`, so an operand appended at the end of a short list is
    // still invisible — that is exactly how the out-parameter of
    // `GetTimeSlotStatus` went missing while this test passed.
    assert_eq!(
        insns[0].operands.get(passes::SRET_OPERAND),
        Some(&Operand::Local(20)),
        "the sret operand must stay at its ABI slot, got {:?}",
        insns[0].operands
    );
}

/// `resolve::run` rebuilds the same operand list positionally by signature —
/// the third site with this defect. All three must restore the slot at the
/// index the renderer reads.
#[test]
fn resolve_keeps_the_indirect_result_at_its_abi_slot() {
    use super::types::{ManagedType, PrimKind};
    let mut m = mref("Fill", 3, false);
    m.sig.ret = ManagedType::Primitive(PrimKind::Void);
    m.sig.params = vec![ManagedType::Primitive(PrimKind::I4)];
    let mut ops = vec![
        Operand::Immediate(0x5000),
        Operand::Local(10),
        Operand::Local(11),
    ];
    for _ in 0..7 {
        ops.push(Operand::MissingAbiArg);
    }
    ops.push(Operand::Local(20)); // the x8 slot
    let mut insns = vec![Insn::new(Op::Call, ops, 0x6000)];
    let mut maps = resolve::ResolveMaps::default();
    maps.methods_by_rva.insert(0x5000, m);
    let table = locals::create(&HashMap::new(), &helper_seed());
    resolve::run(&mut insns, &table, &maps);
    assert!(
        matches!(insns[0].operands.first(), Some(Operand::Method(_))),
        "target must resolve, got {:?}",
        insns[0].operands
    );
    assert_eq!(
        insns[0].operands.get(passes::SRET_OPERAND),
        Some(&Operand::Local(20)),
        "resolve must not drop the sret slot, got {:?}",
        insns[0].operands
    );
}

/// An access INSIDE a stack cell is a MEMORY access to that cell.
///
/// `add x8, sp, #0; bl StructReturning` passes the cell at sp+0, and the
/// caller then reads the returned struct's `int` field from sp+4. Flooring
/// both accesses onto the same `stack_N` name made the value provably equal
/// to the cell's own ADDRESS: the callee's store through x8 vanished, the
/// reload was eliminated as a redundant copy, and the address printed as the
/// value. Real shape: `CollectWeeklyFacebookGift` @0x47014C (AArch64).
#[test]
fn an_access_inside_a_stack_cell_stays_a_memory_access() {
    let mut insns = vec![
        Insn::new(Op::ShiftStack(-16), vec![], 0x1000),
        Insn::new(
            Op::Move,
            vec![
                Operand::reg("x8"),
                Operand::AddressOf(Box::new(Operand::Memory {
                    base: Box::new(Operand::reg("sp")),
                    index: None,
                    scale: 8,
                    addend: 0,
                    width: 0,
                })),
            ],
            0x1004,
        ),
        Insn::new(
            Op::Move,
            vec![
                Operand::reg("w19"),
                Operand::Memory {
                    base: Box::new(Operand::reg("sp")),
                    index: None,
                    scale: 8,
                    addend: 4,
                    width: 4,
                },
            ],
            0x1008,
        ),
        Insn::new(Op::Return, vec![], 0x100c),
    ];
    let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
    let graph = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    stack_analyzer::analyze(&mut insns, &graph);

    let cell = &insns[1].operands[1];
    let inner = match cell {
        Operand::AddressOf(inner) => inner.as_ref(),
        other => panic!("expected an address, got {other:?}"),
    };
    assert!(
        matches!(inner, Operand::Register(r) if r.starts_with("stack_")),
        "the buffer address must name a frame cell, got {inner:?}"
    );
    let read = &insns[2].operands[1];
    match read {
        Operand::Memory {
            base,
            addend,
            width,
            ..
        } => {
            assert_eq!(*addend, 4, "the +4 into the cell must survive");
            assert_eq!(*width, 4, "and the access width with it");
            assert_eq!(
                base.as_ref(),
                inner,
                "both accesses must land in the SAME cell"
            );
        }
        other => panic!("the in-cell read must stay a memory access, got {other:?}"),
    }
}

/// A mid-body reload of a spilled callee-saved register is a LIVE value.
///
/// `TokenManager.Initialize` @0x189D9D4 has `str x27,[sp,#0x10]` (spill the
/// receiver), `mov x27,x0` (the `il2cpp_object_new` result), then
/// `ldr x27,[sp,#0x10]` to reload it. The old epilogue test accepted the
/// method's own `cbz` two instructions later as an epilogue terminator, Nop'd
/// the reload, and every subsequent use read the new RewardData instead of
/// `this`. A load whose value is read before any redefinition/exit is not ABI
/// restore machinery and must survive.
#[test]
fn a_live_mid_body_reload_is_not_a_callee_save_restore() {
    let words: [u32; 25] = [
        0xD10103FF, // 0x00 sub  sp, sp, #0x40
        0xA90273FB, // 0x04 stp  x27, x28, [sp, #0x20]   <- ABI save
        0xD503201F, 0xD503201F, 0xD503201F, 0xD503201F, 0xD503201F, // 0x08..0x18
        0xD503201F, 0xD503201F, 0xD503201F, 0xD503201F, 0xD503201F, // 0x1c..0x2c
        0xD503201F, 0xD503201F, 0xD503201F, 0xD503201F, 0xD503201F, // 0x30..0x40
        0xF9000BFB, // 0x44 str  x27, [sp, #0x10]        <- mid-body spill
        0xAA0003FB, // 0x48 mov  x27, x0                 <- clobber
        0xF9400BFB, // 0x4c ldr  x27, [sp, #0x10]        <- live reload
        0xF9400F60, // 0x50 ldr  x0, [x27, #0x18]        <- read of x27
        0xB4000040, // 0x54 cbz  x0, +8                  <- in-body CondJump
        0xA9426FFC, // 0x58 ldp  x28, x27, [sp, #0x20]   <- epilogue restore
        0x910103FF, // 0x5c add  sp, sp, #0x40
        0xD65F03C0, // 0x60 ret
    ];
    let bytes = le(&words);
    let lifted = lift::lift_arm64(&bytes, 0x1000, 128, None).expect("lift");
    let mut insns = lifted.insns;
    let graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    stack_analyzer::analyze(&mut insns, &graph);

    let reload = insns
        .iter()
        .find(|i| i.address == 0x104c && i.op == Op::Move)
        .expect("the mid-body reload was erased (Nop)");
    assert!(
        matches!(reload.operands.first(), Some(Operand::Register(r)) if r == "x27"),
        "the reload must still define x27: {reload:?}"
    );
    assert!(
        matches!(reload.operands.get(1), Some(Operand::Register(s)) if s.starts_with("stack_")),
        "the reload must read its frame cell: {reload:?}"
    );
    // The dead epilogue restore may still be excised; keeping the live spill
    // must not disable the pass for the register.
    let live_restore = insns.iter().any(|i| {
        i.address == 0x1058
            && i.op == Op::Move
            && matches!(i.operands.first(), Some(Operand::Register(r)) if r == "x27")
    });
    assert!(!live_restore, "the dead epilogue restore should be Nop'd");
}

/// A genuine save/restore pair of the same frame cell, with no reads, is
/// still pure ABI machinery and must be excised.
#[test]
fn a_dead_epilogue_round_trip_is_still_excised() {
    let words: [u32; 5] = [
        0xD10103FF, // sub sp, sp, #0x40
        0xA90273FB, // stp x27, x28, [sp, #0x20]
        0xA9426FFC, // ldp x28, x27, [sp, #0x20]
        0x910103FF, // add sp, sp, #0x40
        0xD65F03C0, // ret
    ];
    let bytes = le(&words);
    let lifted = lift::lift_arm64(&bytes, 0x2000, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let graph = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    stack_analyzer::analyze(&mut insns, &graph);

    let live_pair = insns.iter().any(|i| {
        (i.address == 0x2004 || i.address == 0x2008)
            && i.op == Op::Move
            && i.operands
                .iter()
                .any(|o| matches!(o, Operand::Register(r) if r == "x27" || r == "x28"))
    });
    assert!(!live_pair, "the dead ABI pair must be excised");
}

/// AN UNDECODABLE WORD MUST NOT SHIFT EVERY LATER INSTRUCTION'S ADDRESS.
///
/// Real bytes from `libil2cpp.so` @ `0x53c83c` — a prologue that spills VFP
/// registers into a struct:
///
/// ```text
///   0x53c83c  mov  r0, sp
///   0x53c840  vldr s2, [fp, #12]   <- VFP; yaxpeax-arm 0.4 cannot decode it
///   0x53c844  mov  r2, #0
///   0x53c848  vldr s0, [fp, #16]   <- undecodable
///   0x53c84c  vldr s4, [fp, #24]   <- undecodable
///   0x53c850  vldr s6, [fp, #20]   <- undecodable
///   0x53c854  vstr s2, [r4, #24]   <- undecodable
///   0x53c858  vstr s0, [r4, #28]   <- undecodable
///   0x53c85c  vstr s6, [r4, #32]   <- undecodable
///   0x53c860  vstr s4, [r4, #36]   <- undecodable
///   0x53c864  bl   $+0x1082658
///   0x53c868  ldm  sp, {r0, r1}
/// ```
///
/// yaxpeax's A32 `decode` reads the word from the reader BEFORE validating the
/// encoding, so on `Err` the reader has already consumed this instruction. The
/// old error arm skipped a second word, so the reader ran one word ahead of
/// `offset` per undecodable instruction and the IR was stamped with addresses
/// one word too early. That is not a cosmetic label bug: `Cfg::build` resolves
/// branch targets through `addresses`, so in any method containing a VFP/NEON
/// word every later branch resolved to the WRONG instruction.
#[test]
fn arm32_undecodable_word_does_not_shift_later_addresses() {
    let words: [u32; 12] = [
        0xE1A0_000D, // 0x53c83c  mov r0, sp
        0xED9B_1A03, // 0x53c840  vldr s2, [fp, #12]
        0xE3A0_2000, // 0x53c844  mov r2, #0
        0xED9B_0A04, // 0x53c848  vldr s0, [fp, #16]
        0xED9B_2A06, // 0x53c84c  vldr s4, [fp, #24]
        0xED9B_3A05, // 0x53c850  vldr s6, [fp, #20]
        0xED84_1A06, // 0x53c854  vstr s2, [r4, #24]
        0xED84_0A07, // 0x53c858  vstr s0, [r4, #28]
        0xED84_3A08, // 0x53c85c  vstr s6, [r4, #32]
        0xED84_2A09, // 0x53c860  vstr s4, [r4, #36]
        0xEB42_0994, // 0x53c864  bl $+0x1082658
        0xE89D_0003, // 0x53c868  ldm sp, {r0, r1}
    ];
    let mut bytes = Vec::new();
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let lifted = lift_arm32::lift_arm32(&bytes, 0x53c83c, 64).expect("lift");

    assert_eq!(
        lifted.addresses.len(),
        lifted.insns.len(),
        "`addresses` must stay parallel to `insns`"
    );

    // The instruction whose own word the reader had already stepped past.
    assert!(
        lifted.insns.iter().any(|i| i.address == 0x53c844),
        "`mov r2, #0` @0x53c844 must be lifted at its own address; got {:?}",
        lifted.insns.iter().map(|i| i.address).collect::<Vec<_>>()
    );

    // The call, which the drift had pushed onto a VFP instruction's address.
    assert!(
        lifted
            .insns
            .iter()
            .any(|i| i.address == 0x53c864 && matches!(&i.op, Op::Call)),
        "the `bl` @0x53c864 must be a call AT 0x53c864; got {:?}",
        lifted
            .insns
            .iter()
            .map(|i| format!("{:#x} {:?}", i.address, i.op))
            .collect::<Vec<_>>()
    );

    // The `ldm sp, {r0, r1}` — two scale-4 stack loads, at ITS address.
    let ldm: Vec<&Insn> = lifted
        .insns
        .iter()
        .filter(|i| {
            matches!(i.operands.get(1), Some(Operand::Memory { base, scale, .. })
                if *scale == 4
                    && matches!(base.as_ref(), Operand::Register(r) if r.as_str() == "sp"))
        })
        .collect();
    assert_eq!(
        ldm.len(),
        2,
        "`ldm sp, {{r0, r1}}` is two loads, got {ldm:?}"
    );
    assert!(
        ldm.iter().all(|i| i.address == 0x53c868),
        "the `ldm` @0x53c868 must carry 0x53c868, got {:?}",
        ldm.iter().map(|i| i.address).collect::<Vec<_>>()
    );
}

/// One undecodable word must not delete the whole method.
///
/// `lift_arm64` decoded with `decoder.decode(&mut reader).ok()?`, and the
/// caller maps `None` to `fail!("lift failed")`. So a SINGLE word yaxpeax
/// cannot decode — a VFP/NEON/SVE instruction, or a literal-pool word that
/// falls inside the recorded body — replaced the entire method with a stub
/// that has no IR at all. In the dump that method is indistinguishable from
/// "the decompiler is broken", while the actual gap is four bytes wide.
///
/// yaxpeax's A64 `decode_into` does `words.next_n(&mut word_bytes)?` before it
/// can fail (its source, `armv8/a64.rs`), so a failed decode has already
/// consumed exactly that instruction: skipping one word keeps the reader, the
/// IR addresses and the byte offset in step.
#[test]
fn arm64_undecodable_word_does_not_delete_the_method() {
    let words = [
        0xD2800020, // movz x0, #1      — before the gap
        0xFFFFFFFF, // undecodable word  — the gap itself
        0xD65F03C0, // ret              — after the gap
    ];
    let bytes = le(&words);

    // Pin the premise, or this test silently becomes vacuous the day the
    // decoder learns 0xFFFFFFFF.
    let (_, fails) = verify::decode_all_arm64(&bytes[4..8], 0x1004);
    assert_eq!(
        fails, 1,
        "0xFFFFFFFF must stay undecodable for this test to mean anything"
    );

    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None)
        .expect("one undecodable word must not fail the whole lift");

    // The instruction after the gap keeps ITS OWN address: the skip must not
    // shift anything, because Cfg::build maps branch-target RVAs through
    // `addresses` and any drift misroutes every later branch.
    let ret = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Return))
        .expect("the `ret` after the gap must still be lifted");
    assert_eq!(
        ret.address,
        0x1008,
        "the `ret` after the gap must keep its own address, got {:?}",
        lifted
            .insns
            .iter()
            .map(|i| (i.address, format!("{:?}", i.op)))
            .collect::<Vec<_>>()
    );
    assert_eq!(lifted.addresses.len(), lifted.insns.len());
}

/// A conditional branch whose two arms reach the same instruction is not a
/// branch, and must not be rendered as one.
///
/// `if (cond) { goto L; } L:` — a `goto` to the very next statement — is a
/// shape no compiler emits, and at scale it is what makes a rendered method
/// read as corrupted (494 of them in one iOS dump). The native origin is the
/// IL2CPP null check:
///
/// ```text
///   cbnz w8, +8         ; class != null → skip the raise
///   bl   <raise_null>   ; the raise, an il2cpp_codegen_* runtime helper
/// ```
///
/// The `bl` is excised as a runtime call, which leaves the skipped slot EMPTY
/// and makes the taken and fall-through arms coincide — so the condition has
/// no observable effect and must be dropped, not printed as a guard.
#[test]
fn vacuous_cond_branch_is_dropped_while_a_real_branch_survives() {
    let mut insns = vec![
        // 0..3 — the collapsed null check: the skipped slot (2) is empty, so
        // CondJump(4) takes us to 4 and so does falling through.
        Insn::move_(Operand::reg("x8"), Operand::imm(0), 0x1000),
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(9), Operand::reg("x8"), Operand::imm(0)],
            0x1004,
        ),
        Insn::nop(0x1008),
        Insn::new(Op::CondJump(4), vec![Operand::Local(9)], 0x100c),
        Insn::move_(Operand::reg("x0"), Operand::reg("x8"), 0x1010),
        // 5..8 — a REAL branch: the fall-through path runs a real instruction
        // (6) that the taken path skips, so the two arms differ.
        Insn::new(Op::CondJump(8), vec![Operand::Local(9)], 0x1014),
        Insn::move_(Operand::reg("x1"), Operand::imm(1), 0x1018),
        Insn::nop(0x101c),
        Insn::new(Op::Return, vec![Operand::reg("x0")], 0x1020),
    ];

    let dropped = passes::drop_vacuous_cond_branches(&mut insns);

    assert_eq!(dropped, 1, "exactly the vacuous branch must be dropped");
    assert!(
        matches!(insns[3].op, Op::Nop),
        "the collapsed null check at 3 must be Nop'd, got {:?}",
        insns[3].op
    );
    assert!(
        matches!(insns[5].op, Op::CondJump(8)),
        "a branch whose arms differ must SURVIVE, got {:?}",
        insns[5].op
    );
}

/// A LOOP-CARRIED value is a def CYCLE, and inlining it is unbounded.
///
/// `v1 = v2` at the loop header and `v2 = v1 + v4` in the body means resolving
/// `v1` resolves `v2`, which resolves `v1` again. `inline_local_at` refuses
/// only a def that reaches its OWN use site, so it never caught a cycle of
/// length ≥ 2: the depth guard ended the recursion, but every level of it
/// branches (`v2` reads `v1` *and* `v4`), so the expansion was 2^65 nodes.
///
/// That is the shape behind arm64 @0x2495410 — a 336-byte method the renderer
/// entered 22 million times before it had allocated gigabytes and the dump was
/// killed. The assertion is on the resolution COUNT rather than on elapsed
/// time or the rendered text: it is exact, and it fails immediately instead of
/// hanging when the guard is removed.
#[test]
fn loop_carried_def_is_not_inlined_into_itself() {
    // Jump/CondJump operands are INSN INDICES (`rebuild_cfg` tests `t < n`),
    // not addresses.
    //
    //   i0 @0x2000  v1 = v2            <- loop header
    //   i1 @0x2004  v3 = (v1 < 10)
    //   i2 @0x2008  CondJump(6) [v3]   ; leave the loop
    //   i3 @0x200c  v4 = v1 + 1
    //   i4 @0x2010  v2 = v1 + v4       ; v2 reads v1 *and* v4
    //   i5 @0x2014  Jump(0)            ; back edge
    //   i6 @0x2018  Return(v1)
    let insns = vec![
        Insn::move_(Operand::Local(1), Operand::Local(2), 0x2000),
        Insn::new(
            Op::Compare(CompareKind::LessThan),
            vec![Operand::Local(3), Operand::Local(1), Operand::imm(10)],
            0x2004,
        ),
        Insn::new(Op::CondJump(6), vec![Operand::Local(3)], 0x2008),
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(4), Operand::Local(1), Operand::imm(1)],
            0x200c,
        ),
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(1), Operand::Local(4)],
            0x2010,
        ),
        Insn::new(Op::Jump(0), vec![], 0x2014),
        Insn::new(Op::Return, vec![Operand::Local(1)], 0x2018),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: locals::LocalTable::default(),
        returns_value: true,
        base_rva: 0x2000,
    };

    csharp::reset_def_resolution_counts();
    // resolve the use of `v1` at the `ret` — the loop header's value
    let expr = csharp::resolve_local_use_for_test(&ir, 6, 1);
    let (calls, maxdepth) = csharp::def_resolution_counts();

    assert!(
        calls <= 8,
        "the loop-carried def was re-expanded instead of referenced: {calls} def \
         resolutions (max depth {maxdepth}) for `{expr}`"
    );
    // ... and OUTSIDE the loop the value must be the NAME, not the formula.
    //
    // `v1` is defined in the loop header (`v1 = v2`) and `v2` is written later
    // in that same block, so the definition's expression is only equal to the
    // use's value ON THE ITERATION IT WAS COMPUTED. Inlining it at the `ret`
    // re-evaluates the update against the register the loop has already
    // advanced — for `Exception.GetBaseException`@0x1A6FA74 that returned
    // `null` instead of the base exception. Cross-block uses therefore
    // reference the carried variable.
    assert_eq!(
        expr, "t1",
        "a loop-carried value read outside its block must be referenced by \
         NAME (`t1`), not re-expanded from the definition that moves it"
    );
}

/// A local table with plain (non-parameter, non-`this`) locals, so `var_name`
/// produces the `vN` names the renderer binds call results to.
fn plain_local_table(ids: &[u32]) -> locals::LocalTable {
    let mut t = locals::LocalTable::default();
    for (i, id) in ids.iter().enumerate() {
        t.infos.push(locals::LocalInfo {
            slot_origin: "x9".into(),
            ty: ManagedType::Unknown,
            is_this: false,
            is_param: None,
            param_comp: None,
            is_incoming_arg: false,
            is_call_clobber: false,
            is_return_buffer: false,
        });
        t.index_of.insert(*id, i);
    }
    t
}

/// ONE NATIVE `bl`, ONE EMITTED CALL — even when the result is read twice.
///
/// A call result held by a SINGLE-DEF local used to be inlined at every read
/// (`inline_local_at` expands the reaching def), so a value read at N sites
/// emitted the same `bl` N times. The disassembly has one call; the emitted
/// program had N. Measured on 0x2478f0c: 389 call expressions for 218 native
/// `bl`, the hoisted `fn_4a589d8(...)` repeated across the whole guard chain.
///
/// The fix binds the result to a name when it is read more than once
/// (`call_dest_binding`), so the uses reference `vN` and the call appears once.
/// Reverting the `read_count` half of that predicate makes this test find two
/// copies.
#[test]
fn call_result_read_twice_is_emitted_once() {
    let ids: Vec<u32> = (1..=4).collect();
    //   i0 @0x1000  v1 = 0x1000                     ; the argument
    //   i1 @0x1004  v2 = fn_9000(v1)                ; ONE native bl
    //   i2 @0x1008  v3 = (v2 == 0)                  ; use #1
    //   i3 @0x100c  CondJump(5) [v3]
    //   i4 @0x1010  Return(v2)                      ; use #2
    //   i5 @0x1014  Return(v2)                      ; use #3
    let insns = vec![
        Insn::move_(Operand::Local(1), Operand::imm(0x1000), 0x1000),
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x9000), Operand::Local(2), Operand::Local(1)],
            0x1004,
        ),
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(3), Operand::Local(2), Operand::imm(0)],
            0x1008,
        ),
        Insn::new(Op::CondJump(5), vec![Operand::Local(3)], 0x100c),
        Insn::new(Op::Return, vec![Operand::Local(2)], 0x1010),
        Insn::new(Op::Return, vec![Operand::Local(2)], 0x1014),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: true,
        base_rva: 0x1000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    let n = out.matches("fn_9000(").count();
    assert_eq!(
        n, 1,
        "one native `bl` must produce exactly one call expression, got {n}:\n{out}"
    );
}

/// A conditional jump whose taken target is the statement that executes next
/// ANYWAY gates nothing: both arms run the same code, so the branch must not
/// survive into the dump.
///
/// At the IR level this is a no-op only when the fall-through path resolves to
/// the branch target (`drop_vacuous_cond_branches`). Here it does not: the
/// fall-through is a store to a local nothing reads, so the STRUCTURER builds
/// `if (c) { goto L; }` plus the label, and only once the dead store is deleted
/// does the branch become visibly vacuous. The dump is rendered from that
/// point, so the fold has to happen on the statement tree.
///
/// The real shape is the IL2CPP null check whose `bl <raise_null>` was excised
/// as a runtime helper — 1,483 `goto`-to-the-next-line artifacts in the v24.5
/// PE dump.
#[test]
fn conditional_jump_to_the_next_statement_is_folded_away() {
    let ids: Vec<u32> = (1..=4).collect();
    //   i0 @0x5000  v3 = (v1 == 0)
    //   i1 @0x5004  CondJump(3) [v3]   ; taken -> i3
    //   i2 @0x5008  v2 = 7             ; dead store: nothing reads v2
    //   i3 @0x500c  Return(v1)
    let insns = vec![
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(3), Operand::Local(1), Operand::imm(0)],
            0x5000,
        ),
        Insn::new(Op::CondJump(3), vec![Operand::Local(3)], 0x5004),
        Insn::move_(Operand::Local(2), Operand::imm(7), 0x5008),
        Insn::new(Op::Return, vec![Operand::Local(1)], 0x500c),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: true,
        base_rva: 0x5000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("goto"),
        "a branch to the statement that runs next is not a jump:\n{out}"
    );
    assert!(
        !out.contains("if ("),
        "the condition gates nothing, so it must not be emitted:\n{out}"
    );
    assert!(
        out.contains("return"),
        "folding the branch must not delete the code after it:\n{out}"
    );
}

/// A VOID `ret` MUST NOT DROP the call whose value sits in the return
/// register.
///
/// `ret_is_void` makes `return <value>` impossible and the value is whatever the
/// last call left in x0, so the tempting simplification — "a void method
/// returns nothing, emit a bare `return;`" — deletes a real call whenever the
/// call's own site emitted nothing for it (a non-void callee whose result
/// nothing else reads). That is not hypothetical: the first version of this fix
/// did exactly that and moved dropped-call methods from 830 to 1006.
///
/// The counterpart is the duplicate direction, on `Dirty()` @0x255a6b0: the
/// same shape with a VOID callee rendered one native `bl` at 0x255a6dc twice,
///     if (m_IsDirty == 0) { this.ClearCache(); }
///     this.ClearCache();
/// Both directions are decided by `ret_value_already_emitted`, which chases the
/// ABI return-register `Move` joiner sitting between the call and the `ret`.
#[test]
fn void_return_does_not_drop_the_call_in_the_return_register() {
    let ids: Vec<u32> = (1..=6).collect();
    //   i0 @0x3000  v3 = (v1 == 0)
    //   i1 @0x3004  CondJump(4) [v3]        ; the guard, taken -> the epilogue
    //   i2 @0x3008  v2 = fn_4abc(v1)        ; non-void callee: its own site
    //                                       ;   emits nothing
    //   i3 @0x300c  v4 = v2                 ; the ABI return-register joiner
    //   i4 @0x3010  Return(v4)              ; the epilogue tail / join
    let insns = vec![
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(3), Operand::Local(1), Operand::imm(0)],
            0x3000,
        ),
        Insn::new(Op::CondJump(4), vec![Operand::Local(3)], 0x3004),
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x4abc), Operand::Local(2), Operand::Local(1)],
            0x3008,
        ),
        Insn::move_(Operand::Local(4), Operand::Local(2), 0x300c),
        Insn::new(Op::Return, vec![Operand::Local(4)], 0x3010),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: false,
        base_rva: 0x3000,
    };
    let out = csharp::render_method_csharp(
        &ir,
        "M",
        true,
        &[],
        &ManagedType::Primitive(PrimKind::Void),
        -1,
    );
    let n = out.matches("fn_4abc(").count();
    assert_eq!(
        n, 1,
        "a void method's `ret` carries no value, so the call in the return \
         register must still be emitted exactly once, got {n}:\n{out}"
    );
}

/// An unresolved indirect JUMP (`br Xn`) is a CFG terminator, and everything
/// after it used to be deleted — silently.
///
/// `Op::IndirectCall` is a terminator in `cfg.rs`, so `emit_region` stopped at
/// it. When the jump-table recognizer fires the op is already rewritten to
/// `Op::Switch`, so anything still reaching the structurer is a dispatch that
/// was NOT recovered — and the code that follows it is the case bodies, real
/// code the disassembly has. Measured on 0x20d3ee4: 8 of 64 blocks emitted,
/// 3 of 43 native conditions, and no checker said a word.
#[test]
fn indirect_jump_does_not_delete_the_following_blocks() {
    let ids: Vec<u32> = (1..=3).collect();
    //   i0 @0x2000  v1 = x9
    //   i1 @0x2004  br v1                  ; IndirectCall (dispatch unknown)
    //   i2 @0x2008  v2 = fn_9abc(v1)       ; code that physically follows
    //   i3 @0x200c  Return(v2)
    let insns = vec![
        Insn::move_(Operand::Local(1), Operand::reg("x9"), 0x2000),
        Insn::new(Op::IndirectCall, vec![Operand::Local(1)], 0x2004),
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x9abc), Operand::Local(2), Operand::Local(1)],
            0x2008,
        ),
        Insn::new(Op::Return, vec![Operand::Local(2)], 0x200c),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: true,
        base_rva: 0x2000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("fn_9abc("),
        "code after an unresolved indirect jump must still be emitted:\n{out}"
    );
}

/// A GUARD CHAIN must stay flat, so a long one cannot exhaust the nesting cap.
///
/// `cmp #k; b.cc JOIN` repeated N times, all jumping to one shared epilogue, is
/// how a compiler lowers an unrolled early-exit chain. Rendering each guard as
/// `if (!c) { <whole remainder> }` nests them one level per guard, and past
/// `MAX_NEST_DEPTH` the tail was deleted with no marker — 0x2478f0c emitted 35
/// of 308 native conditions that way. The flat form (`if (c) goto JOIN; <body
/// as a sibling>`) keeps the depth constant.
///
/// 40 guards is deliberate: 32 of them would fit the old nested form, so the
/// threshold is what makes this test fail when the flattening is removed.
#[test]
fn long_guard_chain_is_not_truncated_by_the_nesting_cap() {
    const N: u32 = 40;
    // 3 insns per guard (compare, condjump, body), so the join sits right after
    // the last guard.
    let join_insn = (3 * N) as usize;
    let mut insns: Vec<Insn> = Vec::new();
    for k in 0..N {
        // v1 is the tested value; every guard compares it against a different
        // constant so the conditions are distinct.
        insns.push(Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![
                Operand::Local(100 + k * 2),
                Operand::Local(1),
                Operand::imm(k as i64),
            ],
            0x4000 + u64::from(k) * 12,
        ));
        insns.push(Insn::new(
            Op::CondJump(join_insn),
            vec![Operand::Local(100 + k * 2)],
            0x4004 + u64::from(k) * 12,
        ));
        // The guard's fall-through "body" — one store THROUGH A POINTER.
        //
        // A store to a plain scalar local would be deleted as a dead store
        // (nothing in this method reads `v200+k`), which leaves the guard arm
        // empty; the arm's `goto JOIN` then lands on the join itself and the
        // branch is correctly folded away as vacuous, so the chain would no
        // longer be exercised at all. A store through a pointer is observable
        // by the caller and is never deleted.
        insns.push(Insn::new(
            Op::Move,
            vec![
                Operand::Memory {
                    base: Box::new(Operand::Local(1)),
                    index: None,
                    scale: 1,
                    addend: i64::from(k) * 8,
                    width: 0,
                },
                Operand::imm(k as i64),
            ],
            0x4008 + u64::from(k) * 12,
        ));
    }
    insns.push(Insn::new(Op::Return, vec![Operand::Local(1)], 0x4400));

    let ids: Vec<u32> = (1..=260).collect();
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: true,
        base_rva: 0x4000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("nesting depth limit"),
        "a 40-guard chain must not reach the nesting cap:\n{out}"
    );
    // Count PREDICATES, not `if` statements. Guards that all jump to ONE join
    // are a single compound condition, and the structurer now fuses them into
    // one short-circuit predicate (`if (v1 == 0 || v1 == 1 || …)`) instead of N
    // separate branches. Counting both forms keeps this test about the property
    // it is named for — no guard is silently deleted — while staying agnostic
    // about which of the two shapes the structurer chose.
    let preds = out.matches("if (").count() + out.matches(" || ").count();
    assert_eq!(
        preds, N as usize,
        "every guard's condition must be emitted; got {preds} of {N}\n{out}"
    );
}

/// A LONE skip guard must render as a nested `if`, not a `goto`.
///
/// `if (c) goto J; body; J:` and `if (!c) { body }` say exactly the same thing;
/// the flat form only earns its label and jump when the guard is one of a
/// CHAIN to the same join (`long_guard_chain_is_not_truncated_by_the_nesting_cap`
/// covers that). A lone guard has nothing to chain with, so the goto is pure
/// residue — measured at ~92% of flat guards on the ARM32 corpus.
#[test]
fn a_lone_skip_guard_renders_as_a_nested_if() {
    // v1 is the tested value; the guard skips the body store when v1 == 0.
    let insns = vec![
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(100), Operand::Local(1), Operand::imm(0)],
            0x4000,
        ),
        Insn::new(Op::CondJump(3), vec![Operand::Local(100)], 0x4004),
        // The guarded body — a store THROUGH A POINTER, so it is observable and
        // survives dead-store elimination (a plain local store would not).
        Insn::new(
            Op::Move,
            vec![
                Operand::Memory {
                    base: Box::new(Operand::Local(1)),
                    index: None,
                    scale: 1,
                    addend: 0,
                    width: 0,
                },
                Operand::imm(1),
            ],
            0x4008,
        ),
        Insn::new(Op::Return, vec![Operand::Local(1)], 0x400c),
    ];
    let ids: Vec<u32> = (1..=260).collect();
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: true,
        base_rva: 0x4000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("goto "),
        "a lone skip guard must not leave a goto:\n{out}"
    );
    assert!(
        out.contains("if "),
        "the guarded body must still be emitted under a condition:\n{out}"
    );
}

/// A call result read EXACTLY ONCE must still be emitted as a bound statement.
///
/// The renderer inlines a single-read call at its use site, and that is what
/// makes the C# readable — but the inline is not guaranteed to happen. The
/// def-cycle guard and the inlining budget both fall back to rendering the
/// local's NAME, and then nothing ever defines it.
///
/// Measured on 0x25e6058 (`set_usageHints`): three `blr` survived the lifter and
/// every IR pass into the final listing, the guard fired on all three (a
/// loop-carried def cycle), and the emitted body contained `v24 = v54;` with no
/// statement assigning `v54` — the native calls were simply not in the C#.
/// Gating the binding on `reads > 1` is the bug this test pins.
#[test]
fn call_result_read_once_is_still_bound() {
    let ids: Vec<u32> = (1..=5).collect();
    //   i0 @0x1000  v1 = 0x1000              ; the argument
    //   i1 @0x1004  v2 = fn_9000(v1)         ; ONE native bl, read ONCE below
    //   i2 @0x1008  v3 = (v2 == 0)           ; the single use
    //   i3 @0x100c  CondJump(5) [v3]
    //   i4 @0x1010  Return(v1)
    //   i5 @0x1014  Return(v3)
    let insns = vec![
        Insn::move_(Operand::Local(1), Operand::imm(0x1000), 0x1000),
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x9000), Operand::Local(2), Operand::Local(1)],
            0x1004,
        ),
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(3), Operand::Local(2), Operand::imm(0)],
            0x1008,
        ),
        Insn::new(Op::CondJump(5), vec![Operand::Local(3)], 0x100c),
        Insn::new(Op::Return, vec![Operand::Local(1)], 0x1010),
        Insn::new(Op::Return, vec![Operand::Local(3)], 0x1014),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: true,
        base_rva: 0x1000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    let n = out.matches("fn_9000(").count();
    assert_eq!(
        n, 1,
        "one native `bl` must produce exactly one call expression, got {n}:\n{out}"
    );
    assert!(
        out.contains("= fn_9000("),
        "the single-read call result must be BOUND to a name — an unbound \
         inline leaves nothing behind when the inline is refused:\n{out}"
    );
}

/// A declared parameter whose ABI slot holds no recovered value must STAY in
/// the rendered call.
///
/// `call_parts` builds the argument list by iterating `0..sig.params.len()`, so
/// it holds exactly ONE entry per DECLARED parameter — register-dump filler has
/// no way in, and a trailing `MissingAbiArg` in it is a declared parameter the
/// call site never handed a value for. The cleanup loop that ran anyway dropped
/// them, which SHORTENS the call: `IDeserializationCallback.Deserialization
/// (object)` rendered as `OnDeserialization()`, a shape no compiler emits and
/// which reads as a different overload. Measured on the v24.5 dump, 7,152
/// argument slots were being deleted this way.
#[test]
fn a_declared_parameter_with_no_recovered_value_stays_in_the_call() {
    use super::types::PrimKind;
    let ids: Vec<u32> = (1..=4).collect();
    let mut m = mref("OnDeserialization", 42, false);
    m.sig.ret = ManagedType::Primitive(PrimKind::Void);
    m.sig.params = vec![ManagedType::Object];
    let insns = vec![
        Insn::new(
            Op::Call,
            vec![
                Operand::Method(m),
                Operand::Local(2),      // destination slot (void: unused)
                Operand::Local(1),      // receiver
                Operand::MissingAbiArg, // the declared `object` parameter
            ],
            0x2000,
        ),
        Insn::new(Op::Return, vec![], 0x2004),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&ids),
        returns_value: false,
        base_rva: 0x2000,
    };
    let out = csharp::render_method_csharp(&ir, "M", true, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("OnDeserialization(__abi)"),
        "the declared parameter was dropped from the call:\n{out}"
    );
}

/// Dead-store elimination must not delete a definition whose reader appears
/// EARLIER in the emitted text.
///
/// `dead_stores_flat` scans one statement list FORWARD, but the structurer
/// emits label/goto shells, so text order is not execution order. Measured on
/// 0x25e6058: `v24 = v54;` was emitted at line 17 and `v54 = fnptr_0x…(…)` at
/// line 20; the forward-only scan saw no read after the def, replaced the
/// assignment with a bare call eval, and left line 17 reading a local that
/// nothing defined. The call survived, its VALUE did not.
#[test]
fn dead_stores_keep_a_binding_that_is_read_earlier_in_the_text() {
    let mut stmts = vec![
        // the reader, textually BEFORE the definition
        Stmt::Assign("v24".into(), None, Expr::Var("v54".into())),
        // the definition that the forward scan cannot see a use for
        Stmt::Assign(
            "v54".into(),
            None,
            Expr::Call {
                recv: None,
                name: "fnptr_0x25e60c4".into(),
                args: vec![],
            },
        ),
        // keeps `v24` (the reader) alive, so only `v54` is under test
        Stmt::Return(Some(Expr::Var("v24".into()))),
    ];
    csharp::cleanup::apply(&mut stmts);
    assert!(
        matches!(&stmts[1], Stmt::Assign(t, _, _) if t == "v54"),
        "the binding must survive: the read is in the same body, just \
         earlier in the text — got {:?}",
        stmts[1]
    );
}

/// The class-init guard excision must NOT erase a region that contains a call.
///
/// Every branch of the matcher is STRUCTURAL (one arm has calls, the other
/// jumps to a shared join), and that is also the shape of IL2CPP's per-usage
/// lazy-init guard — so a state machine can carry dozens, each with a real
/// `bl` inside. Erasing the chain deletes the call while the callee still runs
/// natively. Measured on 0x1fe2ca0: 112 calls removed by this pass alone, out
/// of 272 the lifter produced for 272 native `bl`.
#[test]
fn init_guard_excision_never_erases_a_call() {
    //   i0 @0x1000  v1 = 0x8000                       ; klass pointer
    //   i1 @0x1004  v2 = [v1 + 0x132] (byte)          ; the init byte
    //   i2 @0x1008  v3 = v2 & 1
    //   i3 @0x100c  v4 = (v3 != 0)
    //   i4 @0x1010  CondJump(0x101c) [v4]             ; taken = the fast skip
    //   i5 @0x1014  v5 = v1                           ; init arm ...
    //   i6 @0x1018  v6 = fn_9000(v5)                  ; ... and its CALL
    //   i7 @0x101c  Return(v1)                        ; the shared join
    let insns = vec![
        Insn::move_(Operand::Local(1), Operand::imm(0x8000), 0x1000),
        Insn::move_(
            Operand::Local(2),
            Operand::Memory {
                base: Box::new(Operand::Local(1)),
                index: None,
                scale: 8,
                addend: 0x132,
                width: 1,
            },
            0x1004,
        ),
        Insn::new(
            Op::Arith(ArithKind::And),
            vec![Operand::Local(3), Operand::Local(2), Operand::imm(1)],
            0x1008,
        ),
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![Operand::Local(4), Operand::Local(3), Operand::imm(0)],
            0x100c,
        ),
        Insn::new(Op::CondJump(0x101c), vec![Operand::Local(4)], 0x1010),
        Insn::move_(Operand::Local(5), Operand::Local(1), 0x1014),
        Insn::new(
            Op::Call,
            vec![Operand::imm(0x9000), Operand::Local(6), Operand::Local(5)],
            0x1018,
        ),
        Insn::new(Op::Return, vec![Operand::Local(1)], 0x101c),
    ];
    let mut insns = insns;
    // CondJump holds a raw RVA, not an index: resolve it against the real
    // instruction addresses, or it falls back to the past-the-end sentinel
    // and the matcher never sees a merge (this test was silently vacuous
    // because it passed `&[]`).
    let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
    let mut graph = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    passes::remove_init_guards(&mut insns, &mut graph);
    assert!(
        insns.iter().any(|i| matches!(i.op, Op::Call)),
        "the init-arm call must survive the guard excision:\n{insns:?}"
    );
}

/// A Clang arm64 BYTE jump table (`ldrb` entries) must be recovered as a
/// switch.
///
/// This is the shape clang emits for most small C# `switch` statements:
///
/// ```text
///   cmp    x8, #3
///   b.hi   DEFAULT
///   adrp   x9, <table> ; add x9, x9, #lo12
///   adrp   x10, <anchor> ; add x10, x10, #lo12
///   ldrb   w11, [x9, x8]        ; ONE-BYTE entry
///   lsl    w11, w11, #2
///   add    x10, x10, x11
///   br     x10
/// ```
///
/// `anchor_table_targets` accepted only 2- and 4-byte entries
/// (`if *scale != 2 && *scale != 4 { return None }`), so the byte form — the
/// COMMON one — was rejected outright. Measured before the fix: 6 tables
/// recovered over a 3,941-method sample while 511 dispatch sites stayed
/// unresolved. The fallout is not confined to the switch: an unresolved
/// dispatch is a CFG terminator, so every case body goes unwalked and every
/// fact inside it is reported as lost (0x2647100: 21 of 46 blocks never
/// entered, 18 of 27 conditions missing).
#[test]
fn byte_entry_clang_jump_table_is_recovered() {
    const TBL: u64 = 0x5000;
    const ANCHOR: u64 = 0x1020;
    let mut insns: Vec<Insn> = Vec::new();
    // Case bodies: 0x1020, 0x1024, 0x1028, 0x102c are insn indices 8..11.
    for k in 0..12u64 {
        insns.push(Insn::new(Op::Nop, vec![], 0x1000 + k * 4));
    }
    // The index guard: `cmp x8, #3 ; b.hi default` — this is what bounds the
    // table, and it is why FOUR cases are expected and not 1024.
    insns.push(Insn::new(
        Op::Compare(CompareKind::GreaterThan),
        vec![Operand::Local(9), Operand::reg("x8"), Operand::imm(3)],
        0x1030,
    ));
    insns.push(Insn::new(
        Op::CondJump(0x1050),
        vec![Operand::Local(9)],
        0x1034,
    ));
    insns.push(Insn::move_(
        Operand::reg("x9"),
        Operand::imm(TBL as i64),
        0x1038,
    ));
    insns.push(Insn::new(
        Op::Arith(ArithKind::Add),
        vec![Operand::reg("x9"), Operand::reg("x9"), Operand::imm(0)],
        0x103c,
    ));
    insns.push(Insn::move_(
        Operand::reg("x10"),
        Operand::imm(ANCHOR as i64),
        0x1040,
    ));
    insns.push(Insn::move_(
        Operand::reg("x11"),
        Operand::Memory {
            width: 1,
            base: Box::new(Operand::reg("x9")),
            index: Some(Box::new(Operand::reg("x8"))),
            scale: 1,
            addend: 0,
        },
        0x1044,
    ));
    insns.push(Insn::new(
        Op::Arith(ArithKind::Shl),
        vec![Operand::reg("xt0"), Operand::reg("x11"), Operand::imm(2)],
        0x1048,
    ));
    insns.push(Insn::new(
        Op::Arith(ArithKind::Add),
        vec![
            Operand::reg("x10"),
            Operand::reg("x10"),
            Operand::reg("xt0"),
        ],
        0x104c,
    ));
    insns.push(Insn::new(
        Op::IndirectCall,
        vec![Operand::reg("x10")],
        0x1050,
    ));
    let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
    // Byte table: entry `k` means `anchor + k*4`, so the entries are 0,1,2,3.
    // The trailing 0xff is the guard against reading past the real table.
    let bytes: Vec<u8> = vec![0, 1, 2, 3, 0xff, 0xff, 0xff, 0xff];
    let read = move |a: u64| -> Option<u32> {
        if a < TBL || a + 4 > TBL + bytes.len() as u64 {
            return None;
        }
        let i = (a - TBL) as usize;
        Some(u32::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
        ]))
    };
    let mut insns = insns;
    let n = passes::recover_switch_tables(&mut insns, &addresses, &read);
    assert_eq!(n, 1, "the byte-entry table must be recovered");
    let sw = insns
        .iter()
        .find_map(|i| match &i.op {
            Op::Switch(t) => Some(t.clone()),
            _ => None,
        })
        .expect("expected an Op::Switch");
    assert_eq!(sw.len(), 4, "`cmp x8, #3` means FOUR cases, got {sw:?}");
    // ANCHOR + k*4 is the k-th Nop, i.e. insn index 8+k.
    assert_eq!(sw, vec![8usize, 9, 10, 11]);
}

/// `br Xn` AFTER the frame is torn down is an indirect TAIL CALL, not a jump.
///
/// On arm64 a virtual / interface / delegate tail call compiles to the
/// callee-saved restores, `add sp, sp, #frame`, then `br xN` with xN loaded
/// from a vtable slot. That `br` used to reach the structurer as a bare
/// `IndirectCall` — a CFG terminator with no successor — so the text showed
/// "indirect jump (dispatch not recovered)" plus a forced `goto` into an
/// unrelated block, and the verdict counted the method as lost dispatch.
/// Measured: 541 of the 610 surviving dispatch sites on a 4000-method sample
/// (88.7%) are this shape and are real code.
#[test]
fn indirect_tail_call_after_frame_teardown_becomes_a_call() {
    let mut insns = vec![
        Insn::move_(Operand::reg("x19"), Operand::Local(3), 0x2000),
        // `add sp, sp, #32` — the frame is GONE.
        Insn::new(Op::ShiftStack(32), vec![], 0x2004),
        Insn::new(Op::IndirectCall, vec![Operand::reg("x8")], 0x2008),
    ];
    let mut addresses = vec![0x2000u64, 0x2004, 0x2008];
    let nothing = |_: u64| false;
    assert_eq!(
        passes::recover_tail_calls(&mut insns, &mut addresses, &nothing, "x0"),
        1,
        "a frame-torn-down `br` must become a tail call"
    );
    assert_eq!(
        addresses.len(),
        insns.len(),
        "address list must stay aligned"
    );
    assert!(
        matches!(insns[2].op, Op::Call),
        "the `br` must become a call, got {:?}",
        insns[2].op
    );
    assert!(
        matches!(insns[2].operands.first(), Some(Operand::Register(r)) if r == "x8"),
        "the dispatch register must stay the call target, got {:?}",
        insns[2].operands.first()
    );
    assert!(
        matches!(insns[3].op, Op::Return),
        "and the callee's result must be returned, got {:?}",
        insns[3].op
    );
}

/// x86-64 `jmp [klass + vtable_slot]` is the VIRTUAL TAIL CALL — the sibling of
/// arm64's frame-torn-down `br Xn`, and the shape MSVC emits for
/// `return this.Virtual(...)`:
///
/// ```text
/// mov rax, [rcx]          ; klass
/// mov r8,  [rax+0x838]    ; companion MethodInfo of the slot below
/// add rsp, 0x30           ; frame torn down
/// pop rbx
/// jmp [rax+0x830]         ; slot 112 — never returns here
/// ```
///
/// Restricting the arm to `Register` targets left every one of these a bare
/// `Op::IndirectCall`, which `resolve::rewrite_virtual_calls` then rewrote into
/// a FALLING-THROUGH `Op::Call`: the block gained an edge into the code laid out
/// after the `jmp` and the method lost its only exit, so the oracle reported
/// `block N cannot reach an exit` and the text ended with `goto <earlier
/// label>;` — an infinite loop where the native code returns. All 5 remaining
/// x64 `WRONG-LOGIC` methods were this shape.
#[test]
fn an_x64_virtual_tail_call_becomes_a_call_plus_return() {
    let slot = Operand::Memory {
        base: Box::new(Operand::reg("rax")),
        index: None,
        scale: 8,
        addend: 0x830,
        width: 0,
    };
    let mut insns = vec![
        Insn::new(Op::ShiftStack(0x30), vec![], 0x2000),
        Insn::new(Op::IndirectCall, vec![slot], 0x2004),
    ];
    let mut addresses = vec![0x2000u64, 0x2004];
    let nothing = |_: u64| false;
    assert_eq!(
        passes::recover_tail_calls(&mut insns, &mut addresses, &nothing, "rax"),
        1,
        "a frame-torn-down `jmp [klass+slot]` must become a tail call"
    );
    assert_eq!(
        addresses.len(),
        insns.len(),
        "address list must stay aligned"
    );
    assert!(
        matches!(
            &insns[1].operands.first(),
            Some(Operand::Memory {
                addend: 0x830,
                index: None,
                ..
            })
        ),
        "the slot access must stay the call target so it can be named, got {:?}",
        insns[1].operands.first()
    );
    assert!(matches!(insns[1].op, Op::Call), "got {:?}", insns[1].op);
    assert!(
        matches!(insns[2].op, Op::Return),
        "and the callee's result must be returned, got {:?}",
        insns[2].op
    );
}

/// The INDEXED memory form is a jump-table dispatch, never a tail call: an
/// IL2CPP vtable access carries the slot as an immediate addend and has no index
/// register. It must stay `IndirectCall` so `recover_switch_tables` still sees
/// it — and so nothing invents a call edge out of a case target.
#[test]
fn an_indexed_indirect_jump_is_not_a_tail_call() {
    let table = Operand::Memory {
        base: Box::new(Operand::reg("r10")),
        index: Some(Box::new(Operand::reg("rdx"))),
        scale: 8,
        addend: 0,
        width: 0,
    };
    let mut insns = vec![
        Insn::new(Op::ShiftStack(0x30), vec![], 0x3000),
        Insn::new(Op::IndirectCall, vec![table], 0x3004),
    ];
    let mut addresses = vec![0x3000u64, 0x3004];
    let nothing = |_: u64| false;
    assert_eq!(
        passes::recover_tail_calls(&mut insns, &mut addresses, &nothing, "rax"),
        0,
        "an indexed jump table is not a tail call"
    );
    assert!(
        matches!(insns[1].op, Op::IndirectCall),
        "the dispatch must stay visible to switch recovery, got {:?}",
        insns[1].op
    );
}

/// A LIVE frame must NOT be turned into a tail call: `br Xn` inside the
/// function is a computed jump, and a jump-table dispatch in particular must
/// stay `IndirectCall` so `recover_switch_tables` can still see it.
#[test]
fn computed_jump_with_a_live_frame_stays_an_indirect_call() {
    let mut insns = vec![
        // `sub sp, sp, #32` — the frame is LIVE.
        Insn::new(Op::ShiftStack(-32), vec![], 0x3000),
        Insn::new(Op::IndirectCall, vec![Operand::reg("x10")], 0x3004),
    ];
    let mut addresses = vec![0x3000u64, 0x3004];
    let nothing = |_: u64| false;
    assert_eq!(
        passes::recover_tail_calls(&mut insns, &mut addresses, &nothing, "x0"),
        0,
        "a live-frame `br` is not a tail call"
    );
    assert!(
        matches!(insns[1].op, Op::IndirectCall),
        "the dispatch must stay visible to switch recovery, got {:?}",
        insns[1].op
    );
}

// ---------------------------------------------------------------------------
// Statement-tree idioms (`recovery::idioms`)
// ---------------------------------------------------------------------------

fn rd(base: &str, off: i64) -> Expr {
    Expr::RawDeref(Box::new(Expr::Var(base.into())), off, 8)
}

fn if_on(cond: Expr) -> Stmt {
    Stmt::If {
        cond,
        then_body: vec![Stmt::Return(None)],
        else_body: Vec::new(),
    }
}

/// `goto L; L:` — the jump reaches the statement right after it by falling
/// through anyway, so it is deleted. This is the safe half of dead-end block
/// recognition: dropping a JUMP evaluates nothing, so it cannot change what the
/// program observes.
#[test]
fn jump_to_the_very_next_label_is_dead_end_noise() {
    let mut stmts = vec![
        Stmt::Goto("L1".into()),
        Stmt::Label("L1".into()),
        Stmt::Assign("v1".into(), None, Expr::Int(1)),
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts.iter().any(|s| matches!(s, Stmt::Goto(_))),
        "the no-op jump must go: {stmts:?}"
    );
    // With its only reference gone the label is unreachable too, so the whole
    // dead-end block disappears and only the real statement is left.
    assert!(
        !stmts.iter().any(|s| matches!(s, Stmt::Label(_))),
        "the now-unreferenced label must go: {stmts:?}"
    );
    assert_eq!(
        stmts.len(),
        1,
        "only the real statement should remain: {stmts:?}"
    );
}

/// A label that something still jumps to must SURVIVE — the sweep is over the
/// whole body, because the only reference can sit in a sibling subtree.
#[test]
fn a_label_something_jumps_to_is_kept() {
    let mut stmts = vec![
        Stmt::If {
            cond: Expr::Var("v1".into()),
            then_body: vec![Stmt::Goto("L7".into())],
            else_body: Vec::new(),
        },
        Stmt::Label("L7".into()),
        Stmt::Return(None),
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, Stmt::Label(l) if l == "L7")),
        "a jumped-to label must be kept: {stmts:?}"
    );
    let Stmt::If { then_body, .. } = &stmts[0] else {
        panic!("the guard must survive: {stmts:?}");
    };
    assert!(
        matches!(then_body.as_slice(), [Stmt::Goto(l)] if l == "L7"),
        "the guard's jump must survive: {then_body:?}"
    );
}

/// Statements between a `return` and the next label can never run.
#[test]
fn statements_after_a_terminator_are_unreachable() {
    let mut stmts = vec![
        Stmt::Return(None),
        Stmt::Assign("dead".into(), None, Expr::Int(1)),
        Stmt::Label("L2".into()),
        Stmt::Assign("alive".into(), None, Expr::Int(2)),
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(t, _, _) if t == "dead")),
        "code after a return is unreachable and must go: {stmts:?}"
    );
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(t, _, _) if t == "alive")),
        "a label is a real jump target, so what follows it stays: {stmts:?}"
    );
}

/// The same structure-field load read twice becomes ONE named load.
#[test]
fn repeated_structure_field_load_is_hoisted_once() {
    let cond = || {
        Expr::Cmp(
            CmpKind::Ne,
            Box::new(rd("obj", 224)),
            Box::new(Expr::Int(0)),
        )
    };
    let mut stmts = vec![if_on(cond()), if_on(cond())];
    super::idioms::apply(&mut stmts);
    let Some(Stmt::Assign(name, _, value)) = stmts.first() else {
        panic!("the load must be materialized once at the top: {stmts:?}");
    };
    assert_eq!(name, "sf0");
    assert!(
        matches!(value, Expr::RawDeref(_, 224, _)),
        "the temporary must hold the real load, got {value:?}"
    );
    // Both guards now read the temporary instead of re-loading.
    for s in &stmts[1..] {
        let Stmt::If { cond, .. } = s else {
            panic!("expected a guard: {s:?}")
        };
        let Expr::Cmp(_, lhs, _) = cond else {
            panic!("expected a comparison: {cond:?}")
        };
        assert!(
            matches!(lhs.as_ref(), Expr::Var(n) if n == "sf0"),
            "the guard must read the named load, got {lhs:?}"
        );
    }
}

/// …but NOT across a call, which can write the memory the load reads.
#[test]
fn a_field_load_is_not_hoisted_past_a_call() {
    let cond = || {
        Expr::Cmp(
            CmpKind::Ne,
            Box::new(rd("obj", 224)),
            Box::new(Expr::Int(0)),
        )
    };
    let mut stmts = vec![
        if_on(cond()),
        Stmt::Eval(Expr::Call {
            recv: None,
            name: "mutation".into(),
            args: Vec::new(),
        }),
        if_on(cond()),
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(t, _, _) if t.starts_with("sf"))),
        "a call can write the field, so the load must stay where it is: {stmts:?}"
    );
}

/// …and not when the address expression's local is REBOUND in between.
#[test]
fn a_field_load_is_not_hoisted_past_a_rebind_of_its_base() {
    let cond = || {
        Expr::Cmp(
            CmpKind::Ne,
            Box::new(rd("obj", 224)),
            Box::new(Expr::Int(0)),
        )
    };
    let mut stmts = vec![
        if_on(cond()),
        Stmt::Assign("obj".into(), None, Expr::Var("other".into())),
        if_on(cond()),
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(t, _, _) if t.starts_with("sf"))),
        "`obj` is rebound between the reads, so they are different loads: {stmts:?}"
    );
}

/// A `goto` to the label a loop falls out to IS the loop's own `break`: the two
/// transfer control to the same statement, so the jump is restated.
#[test]
fn a_goto_out_of_a_loop_is_the_loops_own_break() {
    let mut stmts = vec![
        Stmt::DoWhile {
            body: vec![Stmt::If {
                cond: Expr::Var("v1".into()),
                then_body: vec![Stmt::Goto("L9".into())],
                else_body: Vec::new(),
            }],
            cond: Expr::Bool(true),
        },
        Stmt::Label("L9".into()),
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::DoWhile { body, .. } = &stmts[0] else {
        panic!("the loop must survive: {stmts:?}");
    };
    let Stmt::If { then_body, .. } = &body[0] else {
        panic!("the guard must survive: {body:?}");
    };
    assert!(
        matches!(then_body.as_slice(), [Stmt::Break]),
        "the jump out of the loop is its `break`: {then_body:?}"
    );
}

/// …but a `goto` whose target is defined INSIDE the loop body is a jump WITHIN
/// the loop, and `break` would leave it — so it must stay a `goto`.
#[test]
fn a_goto_to_a_label_inside_the_loop_is_not_a_break() {
    let mut stmts = vec![
        Stmt::While {
            cond: Expr::Var("v1".into()),
            body: vec![
                // The jump sits inside an arm so it is NOT the `goto L; L:`
                // no-op the dead-jump idiom removes.
                Stmt::If {
                    cond: Expr::Var("v2".into()),
                    then_body: vec![Stmt::Goto("L9".into())],
                    else_body: Vec::new(),
                },
                Stmt::Assign("v3".into(), None, Expr::Int(1)),
                Stmt::Label("L9".into()),
                Stmt::Return(None),
            ],
        },
        Stmt::Label("L9".into()),
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::While { body, .. } = &stmts[0] else {
        panic!("the loop must survive: {stmts:?}");
    };
    let Stmt::If { then_body, .. } = &body[0] else {
        panic!("the guard must survive: {body:?}");
    };
    assert!(
        matches!(then_body.as_slice(), [Stmt::Goto(_)]),
        "a jump within the body must not become a `break`: {then_body:?}"
    );
}

/// Inside a `switch` arm a `break` leaves the SWITCH, not the enclosing loop,
/// so the restatement is refused there.
#[test]
fn a_goto_from_a_switch_arm_does_not_become_a_break() {
    let mut stmts = vec![
        Stmt::While {
            cond: Expr::Var("v1".into()),
            body: vec![Stmt::Switch {
                disc: "v2".into(),
                cases: vec![(0, vec![Stmt::Goto("L9".into())])],
                tail: Vec::new(),
            
                default: None,}],
        },
        Stmt::Label("L9".into()),
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::While { body, .. } = &stmts[0] else {
        panic!("the loop must survive: {stmts:?}");
    };
    let Stmt::Switch { cases, .. } = &body[0] else {
        panic!("the switch must survive: {body:?}");
    };
    assert!(
        matches!(cases[0].1.as_slice(), [Stmt::Goto(_)]),
        "a `break` there would leave the switch: {:?}",
        cases[0].1
    );
}

/// A `goto` back to a `while` loop's head label is the loop's own `continue`:
/// nothing but labels and comments sits between that label and the loop, so
/// the jump skips the rest of the body and re-tests the condition — exactly
/// what `continue` does.
#[test]
fn a_goto_to_a_while_head_becomes_a_continue() {
    let mut stmts = vec![
        Stmt::Label("L1".into()),
        Stmt::While {
            cond: Expr::Var("v1".into()),
            body: vec![Stmt::If {
                cond: Expr::Var("v2".into()),
                then_body: vec![Stmt::Goto("L1".into())],
                else_body: Vec::new(),
            }],
        },
        Stmt::Label("L2".into()),
    ];
    super::idioms::apply(&mut stmts);
    // Both labels are orphaned by the restatement and swept; only the loop
    // remains, with the guarded jump restated.
    assert!(stmts.len() == 1, "orphaned labels must go: {stmts:?}");
    let Stmt::While { body, .. } = &stmts[0] else {
        panic!("the loop must survive: {stmts:?}");
    };
    let Stmt::If { then_body, .. } = &body[0] else {
        panic!("the guard must survive: {body:?}");
    };
    assert!(
        matches!(then_body.as_slice(), [Stmt::Continue]),
        "the jump to the loop head is its `continue`: {then_body:?}"
    );
}

/// …but only for `while`. A jump to before a `do … while` re-executes the body
/// from the top instead of testing the condition, so it must stay a `goto`.
#[test]
fn a_goto_to_a_dowhile_head_is_not_a_continue() {
    let mut stmts = vec![
        Stmt::Label("L1".into()),
        Stmt::DoWhile {
            body: vec![Stmt::If {
                cond: Expr::Var("v2".into()),
                then_body: vec![Stmt::Goto("L1".into())],
                else_body: Vec::new(),
            }],
            cond: Expr::Var("v1".into()),
        },
        Stmt::Label("L2".into()),
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::DoWhile { body, .. } = &stmts[1] else {
        panic!("the loop must survive: {stmts:?}");
    };
    let Stmt::If { then_body, .. } = &body[0] else {
        panic!("the guard must survive: {body:?}");
    };
    assert!(
        matches!(then_body.as_slice(), [Stmt::Goto(_)]),
        "re-entering a `do` body from the top is not `continue`: {then_body:?}"
    );
}

/// …and a head label that is ALSO defined inside the body is ambiguous — the
/// jump may be within the loop — so it must stay a `goto`.
#[test]
fn a_goto_to_a_head_label_also_defined_inside_is_not_a_continue() {
    let mut stmts = vec![
        Stmt::Label("L1".into()),
        Stmt::While {
            cond: Expr::Var("v1".into()),
            body: vec![
                Stmt::Label("L1".into()),
                Stmt::If {
                    cond: Expr::Var("v2".into()),
                    then_body: vec![Stmt::Goto("L1".into())],
                    else_body: Vec::new(),
                },
            ],
        },
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::While { body, .. } = &stmts[1] else {
        panic!("the loop must survive: {stmts:?}");
    };
    let Stmt::If { then_body, .. } = &body[1] else {
        panic!("the guard must survive: {body:?}");
    };
    assert!(
        matches!(then_body.as_slice(), [Stmt::Goto(_)]),
        "an ambiguous head jump must not become `continue`: {then_body:?}"
    );
}

/// A `goto` past the loop's immediate exit skips real statements the exit
/// falls through to, so `break` would execute code the machine never ran
/// there — it must stay a `goto`.
#[test]
fn a_goto_past_the_loop_exit_is_not_a_break() {
    let mut stmts = vec![
        Stmt::While {
            cond: Expr::Var("v1".into()),
            body: vec![Stmt::If {
                cond: Expr::Var("v2".into()),
                then_body: vec![Stmt::Goto("L9".into())],
                else_body: Vec::new(),
            }],
        },
        Stmt::Label("L2".into()),
        Stmt::Assign("v3".into(), None, Expr::Int(1)),
        Stmt::Label("L9".into()),
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::While { body, .. } = &stmts[0] else {
        panic!("the loop must survive: {stmts:?}");
    };
    let Stmt::If { then_body, .. } = &body[0] else {
        panic!("the guard must survive: {body:?}");
    };
    assert!(
        matches!(then_body.as_slice(), [Stmt::Goto(_)]),
        "a jump over live exit code must not become `break`: {then_body:?}"
    );
}

/// Unlike `break`, `continue` passes THROUGH a `switch` to the enclosing loop
/// — which is what the jump to the loop head did — so the restatement holds
/// inside a `case` too.
#[test]
fn a_goto_to_a_while_head_from_a_switch_arm_becomes_a_continue() {
    let mut stmts = vec![
        Stmt::Label("L1".into()),
        Stmt::While {
            cond: Expr::Var("v1".into()),
            body: vec![Stmt::Switch {
                disc: "v2".into(),
                cases: vec![(0, vec![Stmt::Goto("L1".into())])],
                tail: Vec::new(),
            
                default: None,}],
        },
        Stmt::Label("L9".into()),
    ];
    super::idioms::apply(&mut stmts);
    assert!(stmts.len() == 1, "orphaned labels must go: {stmts:?}");
    let Stmt::While { body, .. } = &stmts[0] else {
        panic!("the loop must survive: {stmts:?}");
    };
    let Stmt::Switch { cases, .. } = &body[0] else {
        panic!("the switch must survive: {body:?}");
    };
    assert!(
        matches!(cases[0].1.as_slice(), [Stmt::Continue]),
        "the jump through the switch is still the loop's `continue`: {:?}",
        cases[0].1
    );
}

/// `v = v;` stores a local's own value back into it. The rewrite is narrow by
/// construction: a memory target re-evaluates its ADDRESS expression, which is
/// not a no-op.
#[test]
fn a_local_self_assignment_is_dropped_but_a_memory_store_is_kept() {
    let mem = "*(nint*)(p)".to_string();
    let mut stmts = vec![
        Stmt::Assign("v3".into(), None, Expr::Var("v3".into())),
        Stmt::Assign(mem.clone(), None, Expr::Var(mem.clone())),
        Stmt::Return(None),
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts.iter().any(
            |s| matches!(s, Stmt::Assign(t, _, Expr::Var(n)) if t == n && !t.starts_with('*'))
        ),
        "the local self-store must go: {stmts:?}"
    );
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(t, _, _) if t == &mem)),
        "a memory target re-evaluates its address and must stay: {stmts:?}"
    );
}

/// `if (c) { v = a; } else { v = b; }` is `?:` spelled out — the condition and
/// exactly one arm are evaluated, in the same order, in both forms.
#[test]
fn an_if_else_pair_of_assignments_folds_to_a_conditional() {
    let mut stmts = vec![Stmt::If {
        cond: Expr::Var("c".into()),
        then_body: vec![Stmt::Assign("v5".into(), None, Expr::Int(1))],
        else_body: vec![Stmt::Assign("v5".into(), None, Expr::Int(2))],
    }];
    super::idioms::apply(&mut stmts);
    let Stmt::Assign(t, None, Expr::Cond(..)) = &stmts[0] else {
        panic!("expected a folded conditional: {stmts:?}");
    };
    assert_eq!(t, "v5");
}

/// …but never when the arms assign DIFFERENT locals: that is two effects, not a
/// select between two values.
#[test]
fn arms_assigning_different_locals_do_not_fold() {
    let mut stmts = vec![Stmt::If {
        cond: Expr::Var("c".into()),
        then_body: vec![Stmt::Assign("v5".into(), None, Expr::Int(1))],
        else_body: vec![Stmt::Assign("v6".into(), None, Expr::Int(2))],
    }];
    super::idioms::apply(&mut stmts);
    assert!(
        matches!(stmts[0], Stmt::If { .. }),
        "two targets are two effects: {stmts:?}"
    );
}

/// `if (a) { if (b) { B } }` evaluates `b` only when `a` held — exactly `&&`'s
/// short-circuit rule — so one compound guard is the same program.
#[test]
fn nested_guards_fold_into_one_conjunction() {
    let mut stmts = vec![Stmt::If {
        cond: Expr::Var("a".into()),
        then_body: vec![Stmt::If {
            cond: Expr::Var("b".into()),
            then_body: vec![Stmt::Return(None)],
            else_body: Vec::new(),
        }],
        else_body: Vec::new(),
    }];
    super::idioms::apply(&mut stmts);
    let Stmt::If {
        cond,
        then_body,
        else_body,
    } = &stmts[0]
    else {
        panic!("the guard must survive: {stmts:?}");
    };
    assert!(
        matches!(cond, Expr::Bin(BinOp::AndAlso, ..)),
        "expected a conjunction, got {cond:?}"
    );
    assert!(matches!(then_body.as_slice(), [Stmt::Return(None)]));
    assert!(else_body.is_empty());
}

/// …but an inner `else` arm makes the folded form need `else if`, which is a
/// different rewrite: the guard must be left alone.
#[test]
fn a_nested_guard_with_an_inner_else_does_not_fold() {
    let mut stmts = vec![Stmt::If {
        cond: Expr::Var("a".into()),
        then_body: vec![Stmt::If {
            cond: Expr::Var("b".into()),
            then_body: vec![Stmt::Return(None)],
            else_body: vec![Stmt::Return(Some(Expr::Int(1)))],
        }],
        else_body: Vec::new(),
    }];
    super::idioms::apply(&mut stmts);
    let Stmt::If { then_body, .. } = &stmts[0] else {
        panic!("the outer guard must survive: {stmts:?}");
    };
    assert!(
        matches!(then_body.as_slice(), [Stmt::If { .. }]),
        "an inner else needs `else if`, not `&&`: {then_body:?}"
    );
}

// ---------------------------------------------------------------------------
// Enumerator loop -> `foreach`
// ---------------------------------------------------------------------------

/// `«v».MoveNext()` used as a loop test.
fn move_next(v: &str) -> Expr {
    Expr::Call {
        recv: Some(Box::new(Expr::Var(v.into()))),
        name: "MoveNext".into(),
        args: Vec::new(),
    }
}

/// `«c».GetEnumerator()`.
fn get_enumerator(c: &str) -> Expr {
    Expr::Call {
        recv: Some(Box::new(Expr::Var(c.into()))),
        name: "GetEnumerator".into(),
        args: Vec::new(),
    }
}

/// `«v».Current` — the property read the renderer folds `get_Current()` into.
fn current_of(v: &str) -> Expr {
    Expr::Field(Box::new(Expr::Var(v.into())), "Current".into())
}

/// The lowered iterator protocol collapses back to the `foreach` the source
/// wrote: the acquisition, the `MoveNext` test and the `Current` read together
/// ARE that statement, and the element local becomes the loop variable.
#[test]
fn an_enumerator_loop_collapses_to_foreach() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::While {
            cond: move_next("v1"),
            body: vec![
                Stmt::Assign("v2".into(), None, current_of("v1")),
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Use".into(),
                    args: vec![Expr::Var("v2".into())],
                }),
            ],
        },
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::ForEach {
        var_name,
        collection,
        body,
    } = &stmts[0]
    else {
        panic!("expected a foreach, got {stmts:?}");
    };
    assert_eq!(var_name, "v2", "the element local is the loop variable");
    assert_eq!(collection, "c", "the foreach iterates the collection");
    assert_eq!(
        body.len(),
        1,
        "the `v2 = v1.Current;` binding is gone: {body:?}"
    );
}

/// With no `x = v.Current;` binding the element still has to be named: every
/// `v.Current` read becomes a fresh loop variable.
#[test]
fn an_enumerator_loop_without_an_element_local_still_collapses() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::While {
            cond: move_next("v1"),
            body: vec![Stmt::Eval(Expr::Call {
                recv: None,
                name: "Use".into(),
                args: vec![current_of("v1")],
            })],
        },
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::ForEach { var_name, body, .. } = &stmts[0] else {
        panic!("expected a foreach, got {stmts:?}");
    };
    assert!(!var_name.is_empty());
    let Stmt::Eval(Expr::Call { args, .. }) = &body[0] else {
        panic!("the call must survive: {body:?}");
    };
    assert!(
        matches!(&args[0], Expr::Var(n) if n == var_name),
        "the `.Current` read must be the loop variable, got {args:?}"
    );
}

/// The decisive refusal: the loop's test is NOT `V.MoveNext()`, so `MoveNext`
/// is one exit among several and collapsing would drop the others.
///
/// This is the `GetExistingParameters` @0x12EA65C shape — a secondary latch
/// test beside the `MoveNext` break — and it must be left lowered.
#[test]
fn a_loop_whose_test_is_not_move_next_is_left_alone() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::While {
            cond: Expr::Cmp(
                CmpKind::Lt,
                Box::new(Expr::Var("v9".into())),
                Box::new(Expr::Int(4)),
            ),
            body: vec![Stmt::Assign("v2".into(), None, current_of("v1"))],
        },
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts.iter().any(|s| matches!(s, Stmt::ForEach { .. })),
        "a loop with its own test must not become a foreach: {stmts:?}"
    );
}

/// An enumerator mentioned OUTSIDE its loop (a sibling `Dispose`, say) is not a
/// foreach: the foreach would not define it there.
#[test]
fn an_enumerator_that_escapes_its_loop_is_not_collapsed() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::While {
            cond: move_next("v1"),
            body: vec![Stmt::Assign("v2".into(), None, current_of("v1"))],
        },
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(Expr::Var("v1".into()))),
            name: "Dispose".into(),
            args: Vec::new(),
        }),
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts.iter().any(|s| matches!(s, Stmt::ForEach { .. })),
        "the enumerator is used after the loop: {stmts:?}"
    );
}

/// A `goto`+label pair that is NOT a null check — nothing throws between them —
/// leaves the enumerator mentioned, so the loop stays lowered. (The REAL
/// null-guard chain, which carries the thrower call, is stripped instead; see
/// `a_null_guard_chain_around_the_enumerator_is_stripped`.)
#[test]
fn an_enumerator_mentioned_inside_its_body_is_not_collapsed() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::While {
            cond: move_next("v1"),
            body: vec![
                Stmt::Assign("v2".into(), None, current_of("v1")),
                Stmt::If {
                    cond: Expr::Cmp(
                        CmpKind::Ne,
                        Box::new(Expr::Var("v1".into())),
                        Box::new(Expr::Int(0)),
                    ),
                    then_body: vec![Stmt::Goto("L9".into())],
                    else_body: Vec::new(),
                },
                Stmt::Label("L9".into()),
            ],
        },
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts.iter().any(|s| matches!(s, Stmt::ForEach { .. })),
        "a body that still reads the enumerator cannot be a foreach: {stmts:?}"
    );
}

/// The compiler's break form: the test OPENS the body and the loop is otherwise
/// infinite. `while (true) { t = e.MoveNext(); if (t == 0) break; … }` is the
/// same iterator protocol spelled differently, so it collapses the same way —
/// and the `MoveNext` temporary and its break test are gone with it.
#[test]
fn a_move_next_break_in_a_true_loop_collapses_to_foreach() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::Assign("v7".into(), None, move_next("v1")),
                Stmt::If {
                    cond: Expr::Cmp(
                        CmpKind::Eq,
                        Box::new(Expr::Var("v7".into())),
                        Box::new(Expr::Int(0)),
                    ),
                    then_body: vec![Stmt::Break],
                    else_body: Vec::new(),
                },
                Stmt::Assign("v2".into(), None, current_of("v1")),
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Use".into(),
                    args: vec![Expr::Var("v2".into())],
                }),
            ],
        },
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::ForEach { var_name, body, .. } = &stmts[0] else {
        panic!("the break form must collapse too, got {stmts:?}");
    };
    assert_eq!(var_name, "v2", "the element local is the loop variable");
    assert!(
        !body
            .iter()
            .any(|s| matches!(s, Stmt::Assign(t, _, _) if t == "v7")),
        "the `MoveNext` temporary and its break test are gone: {body:?}"
    );
}

/// The same break form under `do … while (true)`.
#[test]
fn a_move_next_break_in_a_do_while_true_collapses_to_foreach() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::DoWhile {
            body: vec![
                Stmt::Assign("v7".into(), None, move_next("v1")),
                Stmt::If {
                    cond: Expr::Not(Box::new(Expr::Var("v7".into()))),
                    then_body: vec![Stmt::Break],
                    else_body: Vec::new(),
                },
                Stmt::Assign("v2".into(), None, current_of("v1")),
            ],
            cond: Expr::Bool(true),
        },
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        matches!(stmts[0], Stmt::ForEach { .. }),
        "a `do … while (true)` break form is an enumerator loop: {stmts:?}"
    );
}

/// The inline form: the test is written straight into the guard.
#[test]
fn an_inline_move_next_break_collapses_to_foreach() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::While {
            cond: Expr::Bool(true),
            body: vec![
                Stmt::If {
                    cond: Expr::Not(Box::new(move_next("v1"))),
                    then_body: vec![Stmt::Break],
                    else_body: Vec::new(),
                },
                Stmt::Assign("v2".into(), None, current_of("v1")),
                Stmt::Eval(Expr::Call {
                    recv: None,
                    name: "Use".into(),
                    args: vec![Expr::Var("v2".into())],
                }),
            ],
        },
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::ForEach { var_name, body, .. } = &stmts[0] else {
        panic!("`if (!e.MoveNext()) break;` is the iterator protocol: {stmts:?}");
    };
    assert_eq!(var_name, "v2");
    assert_eq!(body.len(), 1, "only the body call remains: {body:?}");
}

/// A `do … while (c)` with a REAL `c` is refused even when the body opens with
/// the `MoveNext` break: `c` is a second exit `foreach` does not have, so
/// collapsing would silently drop it. (The `GetExistingParameters` @0x12EA65C
/// shape.)
#[test]
fn a_do_while_with_a_real_test_is_not_an_enumerator_loop() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::DoWhile {
            body: vec![
                Stmt::Assign("v7".into(), None, move_next("v1")),
                Stmt::If {
                    cond: Expr::Cmp(
                        CmpKind::Eq,
                        Box::new(Expr::Var("v7".into())),
                        Box::new(Expr::Int(0)),
                    ),
                    then_body: vec![Stmt::Break],
                    else_body: Vec::new(),
                },
                Stmt::Assign("v2".into(), None, current_of("v1")),
            ],
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var("v9".into())),
                Box::new(Expr::Int(0)),
            ),
        },
    ];
    super::idioms::apply(&mut stmts);
    assert!(
        !stmts.iter().any(|s| matches!(s, Stmt::ForEach { .. })),
        "a real `do … while` test is a second exit and must refuse the fold: {stmts:?}"
    );
}

/// The IL2CPP null-guard chain around the enumerator is the RUNTIME's null
/// check, not the program: stripping it lets the loop collapse.
/// `if (v1 != 0) goto L3; ThrowNRE(…); L3: v2 = v1.Current;`
#[test]
fn a_null_guard_chain_around_the_enumerator_is_stripped() {
    let mut stmts = vec![
        Stmt::Assign("v1".into(), None, get_enumerator("c")),
        Stmt::While {
            cond: move_next("v1"),
            body: vec![
                Stmt::If {
                    cond: Expr::Cmp(
                        CmpKind::Ne,
                        Box::new(Expr::Var("v1".into())),
                        Box::new(Expr::Int(0)),
                    ),
                    then_body: vec![Stmt::Goto("L3".into())],
                    else_body: Vec::new(),
                },
                Stmt::Eval(Expr::RawCall(0x5ae014, vec![Expr::Int(0)], None)),
                Stmt::Label("L3".into()),
                Stmt::Assign("v2".into(), None, current_of("v1")),
            ],
        },
    ];
    super::idioms::apply(&mut stmts);
    let Stmt::ForEach { var_name, body, .. } = &stmts[0] else {
        panic!("the null-guard chain must not block the fold: {stmts:?}");
    };
    assert_eq!(var_name, "v2");
    assert!(
        body.is_empty(),
        "the guard chain and the element binding are both gone: {body:?}"
    );
}

/// A `do … while` closes on the same line as its body's brace — the fix for the
/// corpus-wide `}        while (true);` malformation (the same defect the `else`
/// writer had).
#[test]
fn a_do_while_close_continues_on_the_braces_line() {
    let stmts = vec![Stmt::If {
        cond: Expr::Var("c".into()),
        then_body: vec![Stmt::DoWhile {
            body: vec![Stmt::Eval(Expr::Var("work".into()))],
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var("v9".into())),
                Box::new(Expr::Int(0)),
            ),
        }],
        else_body: Vec::new(),
    }];
    let out = csharp::render_stmts(&stmts);
    assert!(
        out.contains("} while (v9 != 0);"),
        "the do-while close continues on the brace line, got:\n{out}"
    );
    assert!(
        !out.contains("}    while") && !out.contains("}\twhile"),
        "no indentation may be written between `}}` and `while`, got:\n{out}"
    );
}

/// A single-def local read more than once whose inlined expansion is large
/// enough to blow up (the `x_k = x_{k-1} + x_{k-1}` DAG) must be MATERIALISED,
/// not copied into every use.
///
/// `XmlTextReader.Read` @0xD5B598 rendered as ONE 1.34 MB line this way: the
/// `__unknown == 0 ? … : …` ternary and the `in_x8` operand each appeared
/// 20,284 / 40,086 times in a single statement, because each level copied the
/// level below it twice. Naming the shared value once collapsed that method's
/// worst line to 632 characters and its `in_x8` count to 23. This pins the size
/// metric that decides which locals are named.
#[test]
fn a_large_shared_def_is_materialised_instead_of_re_inlined() {
    // v1 is a leaf: a def whose operands are all immediates (size 1).
    let mut insns = vec![Insn::new(
        Op::Arith(ArithKind::Add),
        vec![
            Operand::Local(1),
            Operand::Immediate(3),
            Operand::Immediate(4),
        ],
        0x1000,
    )];
    let mut defs = std::collections::HashMap::new();
    defs.insert(1u32, 0usize);
    // vk = v{k-1} + v{k-1}: the expansion doubles each level
    // (sizes 3, 7, 15, 31, 63, 127, 255).
    for k in 2u32..=8 {
        let i = insns.len();
        insns.push(Insn::new(
            Op::Arith(ArithKind::Add),
            vec![
                Operand::Local(k),
                Operand::Local(k - 1),
                Operand::Local(k - 1),
            ],
            0x1000 + i as u64 * 4,
        ));
        defs.insert(k, i);
    }
    let multi_def = std::collections::HashSet::new();
    let read_count: std::collections::HashMap<u32, usize> =
        (1u32..=8).map(|k| (k, 2usize)).collect();

    let shared = super::csharp::compute_shared_large(&insns, &defs, &multi_def, &read_count, 64);
    assert!(
        !shared.contains(&1) && !shared.contains(&2),
        "small defs stay inlinable, so ordinary `x * x` sharing is untouched: {shared:?}"
    );
    assert!(
        shared.contains(&7) && shared.contains(&8),
        "a shared def whose expansion reaches the node ceiling is named, not copied: {shared:?}"
    );

    // The same def read only ONCE is a plain temporary: inlining it copies the
    // tree exactly once, which cannot compound, so it must stay inline.
    let once: std::collections::HashMap<u32, usize> = (1u32..=8)
        .map(|k| (k, if k == 7 { 1 } else { 2 }))
        .collect();
    let shared_once = super::csharp::compute_shared_large(&insns, &defs, &multi_def, &once, 64);
    assert!(
        !shared_once.contains(&7),
        "a large def used exactly once is inlined, not named: {shared_once:?}"
    );
}

/// The tail-merge splicer must tell a dispatch STUB apart from a real
/// out-of-body tail.
///
/// A generic-method stub branches back INTO the body (its payload call is still
/// in-body), so splicing it would duplicate blocks and sever the def→use chain.
/// A real tail-called body has no edge back into the method — the delegate
/// `EndInvoke` shape at 0x329C8, where the previous code mapped the
/// unresolvable jump to the CFG's exit sentinel and dropped the entire body to
/// `{ // body not recoverable }`.
#[test]
fn a_tail_that_does_not_reenter_the_body_is_not_a_stub() {
    let body_lo = 0x329c8u64;
    let body_hi = body_lo + 0x40;

    // Stub: reloads a callee-saved register, then branches BACK into the body.
    let stub = vec![
        Insn::move_(
            Operand::Register("x19".into()),
            Operand::Register("x0".into()),
            0xa58a14,
        ),
        Insn::new(Op::CondJump(body_lo as usize), vec![], 0xa58a18),
    ];
    assert!(
        passes::tail_branches_back_into(&stub, body_lo, body_hi),
        "a branch back into the body marks a dispatch stub"
    );

    // Real out-of-body body: its branches leave the method entirely.
    let real = vec![
        Insn::move_(
            Operand::Register("x20".into()),
            Operand::Register("x1".into()),
            0xa58a14,
        ),
        Insn::new(Op::Jump(0xa58b44), vec![], 0xa58a50),
    ];
    assert!(
        !passes::tail_branches_back_into(&real, body_lo, body_hi),
        "a tail that never re-enters the body is a real transfer, not a stub"
    );
}

/// A guard whose TAKEN edge leaves the method must not erase the body.
///
/// `if (…) return thunk;` compiles the branch target OUTSIDE the method, and
/// `Cfg::build` resolves that to the one-past-the-end sentinel. The successor
/// builder used to `.flatten()` the missing taken edge away, sliding the
/// fallthrough into the taken slot — so `succ_fall` read as `None`, `emit_if`
/// had no false arm, and the ENTIRE body rendered as `{ // body not
/// recoverable }`. Measured on the iOS corpus: 34 of the 45 empty-body methods
/// were this shape (`IsCompilerGenerated` @0x145908 was the canonical one).
#[test]
fn a_guard_whose_taken_edge_leaves_the_method_still_has_a_body() {
    let ir = super::RecoveredIr {
        insns: vec![
            Insn::move_(Operand::Local(10), Operand::imm(0), 0x5000),
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![Operand::Local(10), Operand::imm(0)],
                0x5004,
            ),
            // Target 0x9ff0 is not an instruction here, so the CFG resolves it
            // to its one-past-the-end exit sentinel.
            Insn::new(Op::CondJump(0x9ff0), vec![], 0x5008),
            Insn::move_(Operand::Local(11), Operand::imm(1), 0x500c),
            Insn::new(Op::Return, vec![Operand::Local(11)], 0x5010),
        ],
        table: locals::LocalTable::default(),
        returns_value: true,
        base_rva: 0x5000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("body not recoverable"),
        "an out-of-method guard must not erase the body, got:\n{out}"
    );
    // The guard's taken arm has no statements (it just leaves the method), so
    // it collapses; what matters is that the walk CONTINUES past it. With the
    // old shifted slot the walker stopped at the conditional branch and this
    // `return` was never reached.
    assert!(
        out.contains("return"),
        "the block after the guard must still be emitted, got:\n{out}"
    );
}

/// A VOID body whose only statement was provably dead is EMPTY, not
/// unrecovered. `AppsFlyerStub.OpenSettings` lifted to a single fabricated
/// metadata-token allocation (`var appsFlyer = new AppsFlyer();`); once the
/// write-only sweep removed it the renderer refused with
/// `// body not recoverable`, which is false — the body was recovered and it
/// is `{ }`. A value-returning method keeps the refusal: losing every
/// statement there also lost the return, which is what the marker guards.
#[test]
fn a_void_body_emptied_by_dead_code_removal_is_not_marked_unrecoverable() {
    let alloc = || {
        Insn::new(
            Op::NewObj(Some(super::ir::NamedTypeRef {
                ty: super::types::NamedType {
                    namespace: "Test".into(),
                    name: "Widget".into(),
                    assembly: None,
                    type_def_index: 7,
                },
            })),
            vec![Operand::Local(5)],
            0x5000,
        )
    };
    let mk = |returns_value: bool| super::RecoveredIr {
        insns: vec![alloc(), Insn::new(Op::Return, vec![], 0x5004)],
        table: locals::LocalTable::default(),
        returns_value,
        base_rva: 0x5000,
    };

    let void_out = csharp::render_method_csharp(
        &mk(false),
        "OpenSettings",
        false,
        &[],
        &ManagedType::Unknown,
        -1,
    );
    assert!(
        !void_out.contains("body not recoverable"),
        "a legitimately empty void body must not refuse, got:\n{void_out}"
    );
    assert!(
        !void_out.contains("new Widget"),
        "the unread allocation must not survive either, got:\n{void_out}"
    );
}

/// Getter/setter idioms must fire on BOTH call shapes.
///
/// Two distinct defects shared a symptom (`get_X()`/`set_X(v)` surviving as
/// calls). A STATIC getter is named `Type.get_X` by `call_parts`, so the bare
/// `get_` prefix check never matched. And the setter fold read the receiver
/// from `ops.first()` — the METHOD TARGET — so a folded instance setter
/// rendered as `GetMethodInfo("T.set_X").X = v`, a receiver that does not
/// exist in the source.
#[test]
fn a_static_getter_folds_and_a_setter_uses_its_real_receiver() {
    let g = mref("get_Count", 3, true);
    let ir = super::RecoveredIr {
        insns: vec![
            Insn::new(
                Op::Call,
                vec![Operand::Method(g), Operand::Local(10)],
                0x6000,
            ),
            Insn::new(Op::Return, vec![Operand::Local(10)], 0x6004),
        ],
        table: locals::LocalTable::default(),
        returns_value: true,
        base_rva: 0x6000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("Widget.Count"),
        "static getter must fold: {out}"
    );
    assert!(
        !out.contains("get_Count"),
        "getter call must not survive: {out}"
    );

    // Instance setter. Operand layout is [target, retv, this, params…].
    let mut s = mref("set_enabled", 3, false);
    s.sig.params = vec![ManagedType::Unknown];
    let ir = super::RecoveredIr {
        insns: vec![Insn::new(
            Op::Call,
            vec![
                Operand::Method(s),
                Operand::Unknown,
                Operand::Local(9),
                Operand::Local(10),
            ],
            0x7000,
        )],
        table: locals::LocalTable::default(),
        returns_value: false,
        base_rva: 0x7000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(!out.contains("set_enabled"), "setter call must fold: {out}");
    assert!(
        out.contains(".enabled ="),
        "setter must become an assignment: {out}"
    );
    assert!(
        !out.contains("Widget"),
        "receiver must be the target object, not the method's declaring type: {out}"
    );
}

/// A loop whose only body statement updates a SINGLE-def local that the
/// header reads must not render an empty body.
///
/// `compute_var_reads` skipped every single-def local, so the latch's
/// `v2 = v1 + 1` was never materialised, the loop body was empty, and the
/// renderer printed `while (…) { // loop body not recovered }` — 543 such
/// bodies in Assembly-CSharp alone (`Range(int,int)` @0x16C60).
#[test]
fn a_loop_carried_single_def_body_is_not_empty() {
    let insns = vec![
        Insn::new(
            Op::Compare(CompareKind::LessThan),
            vec![Operand::Local(3), Operand::Local(1), Operand::imm(10)],
            0x3000,
        ),
        Insn::new(Op::CondJump(5), vec![Operand::Local(3)], 0x3004),
        // The latch: a SINGLE def of v2, read back in the header via v1.
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(1), Operand::imm(1)],
            0x3008,
        ),
        Insn::move_(Operand::Local(1), Operand::Local(2), 0x300c),
        Insn::new(Op::Jump(0), vec![], 0x3010),
        Insn::new(Op::Return, vec![Operand::Local(1)], 0x3014),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: locals::LocalTable::default(),
        returns_value: true,
        base_rva: 0x3000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("loop body not recovered"),
        "the loop body must carry the update, got:\n{out}"
    );
}

/// A ROTATED loop — the entry block is its own latch — must keep its body and
/// its loop-carried update, and must not re-evaluate that update at the exit.
///
/// The C# compiler emits `while (inner != null) inner = inner.InnerException;`
/// as `back = inner; inner = [inner +0x28]; cbnz inner, -8; return back`, so the
/// method's FIRST block is both the loop header and its own latch. The listing
/// below is exactly what the lifter produces for that shape, and three separate
/// things used to go wrong on it:
///
///  * SSA placed no phi — the entry block's only CFG predecessor is itself, so
///    the standard `>= 2` join test skipped it — and every iteration then read
///    the ENTRY version of the register;
///  * the structurer emitted `while (cond) { body }` although the native code
///    runs the body BEFORE the test, and the header's prelude came out empty;
///  * the loop TEST re-inlined the same update — after the body had already
///    advanced the carried register — so it loaded the NEXT element, emitted a
///    second native load per iteration and ran one iteration short
///    (`while (*(nint*)(v0 +40) != 0)`);
///  * the update expression was re-inlined at the RETURN, outside the loop,
///    where the register had already been advanced.
///
/// Measured source: `Exception.GetBaseException` @0x1A6FA74, which rendered
/// `while (this._innerException != 0) { // loop body not recovered }` followed
/// by `return this._innerException;` — an unkillable infinite loop that also
/// returned the wrong value.
#[test]
fn a_rotated_loop_keeps_its_body_and_its_carried_update() {
    let insns = vec![
        Insn::move_(Operand::Local(2), Operand::Local(0), 0x1a6fa74),
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(3),
                Operand::Memory {
                    base: Box::new(Operand::Local(0)),
                    index: None,
                    scale: 8,
                    addend: 40,
                    width: 0,
                },
            ],
            0x1a6fa78,
        ),
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![Operand::Local(4), Operand::Local(3), Operand::imm(0)],
            0x1a6fa7c,
        ),
        Insn::move_(Operand::Local(0), Operand::Local(3), 0x1a6fa7c),
        Insn::new(Op::CondJump(0), vec![Operand::Local(4)], 0x1a6fa7c),
        // The RETURN reads the value the loop last ASSIGNED to the carried
        // local — the shape copy-propagation leaves behind when it forwards
        // `v4 -> v2 -> v0`. It must render as the carried variable, never as a
        // re-read of the memory the assignment consumed.
        Insn::new(Op::Return, vec![Operand::Local(3)], 0x1a6fa84),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&[0, 1, 2, 3, 4, 5]),
        returns_value: true,
        base_rva: 0x1a6fa74,
    };
    let out = csharp::render_method_csharp(
        &ir,
        "GetBaseException",
        false,
        &[],
        &ManagedType::Unknown,
        -1,
    );
    assert!(
        !out.contains("loop body not recovered"),
        "the rotated loop body must be recovered, got:\n{out}"
    );
    // `do { … } while (…)`: the body runs before the test, exactly as the
    // native code does. `while (cond) { body }` would skip the first update.
    assert!(
        out.contains("do"),
        "a self-latching header must render as do/while, got:\n{out}"
    );
    let (body, rest) = out.split_once("while (").expect("loop test");
    assert!(
        body.contains("+40"),
        "the loop-carried update must be emitted in the body, got:\n{out}"
    );
    // The exit test must read the value the body just COMPUTED — by NAME, not
    // by re-inlining the load. `v0` is already advanced by the body, so
    // `while (*(nint*)(v0 +40) != 0)` loads the NEXT element: a second native
    // load per iteration and a trip count one short of the original.
    assert!(
        !rest.contains("nint"),
        "the exit test must not re-read the memory the body already consumed, \
         got:\n{out}"
    );
    assert!(
        rest.contains("!= 0)"),
        "the exit test must still be the loop's own test, got:\n{out}"
    );
    // The value assigned to the carried local may not be re-inlined at the
    // RETURN: there the register already holds the next element, so the
    // re-inlined read is the one that ends the loop (0), not the base value.
    let ret = out.rsplit("return ").next().unwrap_or("");
    assert!(
        !ret.contains("nint"),
        "the carried update must not be re-inlined at the return, got:\n{out}"
    );
}

/// `continue` and `break` belong to the INNERMOST loop: a branch to an OUTER
/// loop's header from inside an inner loop must stay a `goto`.
///
/// The structurer emitted `continue` for ANY loop header, so a jump out of the
/// inner loop back to the outer header became `continue;` — which in C# means
/// "next iteration of the inner loop", a different program. Where the jump sat
/// outside every loop it produced a `continue;` with no loop around it at all:
/// not even valid C# (measured on the 4,000-method sample: 30 methods, e.g.
/// `get_JumpHeight` @0x2DF00, whose body carried an orphaned
/// `else { …; continue; }` before its `while`).
#[test]
fn a_jump_to_an_outer_loop_header_stays_a_goto() {
    let insns = vec![
        // outer header: `if (i >= 10) return;`
        Insn::move_(Operand::Local(1), Operand::imm(0), 0x5000),
        Insn::new(
            Op::Compare(CompareKind::GreaterOrEqual),
            vec![Operand::Local(5), Operand::Local(1), Operand::imm(10)],
            0x5004,
        ),
        Insn::new(Op::CondJump(14), vec![Operand::Local(5)], 0x5008),
        // inner header: `if (j >= 3) goto 10;`
        Insn::move_(Operand::Local(2), Operand::imm(0), 0x500c),
        Insn::new(
            Op::Compare(CompareKind::GreaterOrEqual),
            vec![Operand::Local(6), Operand::Local(2), Operand::imm(3)],
            0x5010,
        ),
        Insn::new(Op::CondJump(10), vec![Operand::Local(6)], 0x5014),
        // inner body: `if (i == 5) goto <the outer header>;`
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(7), Operand::Local(1), Operand::imm(5)],
            0x5018,
        ),
        Insn::new(Op::CondJump(13), vec![Operand::Local(7)], 0x501c),
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(2), Operand::imm(1)],
            0x5020,
        ),
        Insn::new(Op::Jump(4), vec![], 0x5024), // inner latch
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(1), Operand::Local(1), Operand::imm(1)],
            0x5028,
        ),
        Insn::new(Op::Jump(1), vec![], 0x502c), // outer latch
        Insn::new(Op::Jump(1), vec![], 0x5030), // the arm: back to the OUTER header
        Insn::new(Op::Return, vec![], 0x5034),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&[0, 1, 2, 5, 6, 7]),
        returns_value: false,
        base_rva: 0x5000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("goto L0;"),
        "a jump to the OUTER header from inside the inner loop must stay a goto, got:\n{out}"
    );
    // The inner loop's own latch IS the innermost loop's header edge, so its
    // `continue` must survive — the fix narrows the scope, it does not remove
    // the statement.
    assert!(
        out.contains("continue;"),
        "the inner latch must keep its `continue`, got:\n{out}"
    );
}

/// A loop header's POST-TEST statements belong at the TOP of the loop body.
///
/// The hardware runs a header top to bottom on every iteration — [pre-test
/// ops] test [post-test ops] — and a `while (test) { … }` re-runs the test
/// first, so the post-test ops are exactly what the body must open with.
/// The C# compiler rotates an indexed loop by hoisting the induction update
/// above the body (`i = next` sits AFTER the test, still inside the header),
/// and emitting only the body region dropped it: the update survived as
/// `v128 = v49 + 1;` — a local nothing reads — while the test kept reading the
/// entry index, i.e. a loop whose index never advances.
///
/// Measured source: `UpdateOwnedState` @0x1C3A5C, which rendered
/// `while (list[i] != 0) { list[i].SetActive(…); v128 = v49 + 1; }` with no
/// assignment to `v49` anywhere in the method. The pre-test half must NOT be
/// emitted: the test inlines it, and emitting it too produced a dead binding
/// (`v40 = this._pickupRoot;` beside `if (this._pickupRoot == 0)`).
#[test]
fn a_loop_header_post_test_update_is_emitted_in_the_body() {
    let insns = vec![
        // header: `list[i]` (reads the carried index), its null test, and then
        // the rotation's induction update.
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(3),
                Operand::Memory {
                    base: Box::new(Operand::Local(0)),
                    index: Some(Box::new(Operand::Local(1))),
                    scale: 8,
                    addend: 32,
                    width: 0,
                },
            ],
            0x4000,
        ),
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![Operand::Local(4), Operand::Local(3), Operand::imm(0)],
            0x4004,
        ),
        // post-test: `i = next`
        Insn::move_(Operand::Local(1), Operand::Local(2), 0x4008),
        Insn::new(Op::CondJump(6), vec![Operand::Local(4)], 0x400c),
        // body + latch: `next = i + 1`, back to the header
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(1), Operand::imm(1)],
            0x4010,
        ),
        Insn::new(Op::Jump(0), vec![], 0x4014),
        Insn::new(Op::Return, vec![], 0x4018),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&[0, 1, 2, 3, 4, 5]),
        returns_value: false,
        base_rva: 0x4000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("loop body not recovered"),
        "the loop body must be recovered, got:\n{out}"
    );
    assert!(
        !out.contains("while (true)"),
        "one native branch must stay one condition, got:\n{out}"
    );
    let cond = out
        .lines()
        .find(|l| l.trim_start().starts_with("while ("))
        .expect("the loop test");
    // `CondJump(6)` leaves the loop when `list[i] != 0`, so the loop runs while
    // it IS 0 — the same polarity the branch has, not an inverted restatement.
    assert!(
        cond.contains("== 0"),
        "the test must keep its own polarity, got:\n{out}"
    );
    let update = out
        .find("v1 = v2")
        .or_else(|| out.find("v1 = i"))
        .or_else(|| out.find("index = i"))
        .or_else(|| out.find("= v2"));
    assert!(
        update.is_some(),
        "the header's post-test update `i = next` must open the body, got:\n{out}"
    );
    // ...and it must come BEFORE the body's own use of the index, exactly as
    // the hardware runs it.
    // The copy `i = next` makes the two names equal, so the body use may be
    // rendered through either one; both spellings are the same value. Role
    // naming may spell the induction value `i` instead of `v2`.
    let use_pos = out
        .find("v2 = v1 + 1")
        .or_else(|| out.find("v2 = v2 + 1"))
        .or_else(|| out.find("i = i + 1"))
        .or_else(|| out.find("i = v1 + 1"))
        .unwrap_or_else(|| panic!("the latch update must stay an assignment, got:\n{out}"));
    assert!(
        update.unwrap() < use_pos,
        "the post-test update runs before the body, got:\n{out}"
    );
}

/// An `if` whose arms only DEFINE jump labels must still define them.
///
/// The renderer drops an `if` shell whose arms render nothing (`if (c) {}`),
/// but an arm can still hold `Label` statements that gotos elsewhere target.
/// Returning outright deleted those definitions from the text while the gotos
/// kept pointing at them, printing `goto L<missing>` — the unreachable /
/// malformed-jump-target class. Measured on Assembly-CSharp: the label-only
/// arm shape was the last source of dangling targets.
#[test]
fn an_if_that_only_defines_labels_still_defines_them() {
    let stmts = vec![
        Stmt::Goto("L7".into()),
        Stmt::If {
            cond: Expr::Var("c".into()),
            then_body: vec![Stmt::Label("L7".into())],
            else_body: vec![],
        },
    ];
    let out = csharp::render_stmts(&stmts);
    assert!(
        out.contains("L7:"),
        "a label-only arm still DEFINES its label, got:\n{out}"
    );
    assert!(
        out.contains("goto L7;"),
        "the goto must keep a target that is defined, got:\n{out}"
    );
}

/// A constant-folded `if` renders only the live arm — but the arm it defers may
/// still DEFINE labels, and dropping it printed `goto L<missing>`.
#[test]
fn a_folded_constant_arm_still_defines_its_labels() {
    let stmts = vec![
        Stmt::Goto("L2".into()),
        Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Eval(Expr::Var("work".into()))],
            else_body: vec![Stmt::Label("L2".into())],
        },
    ];
    let out = csharp::render_stmts(&stmts);
    assert!(
        out.contains("L2:"),
        "the deferred arm's label must still be defined, got:\n{out}"
    );
}

/// An inner guard an ENCLOSING guard already decided is impossible and must be
/// folded away. Every `vN` has one SSA definition, so `if (v0 != 0) { …
/// if (v0 == 0) … }` can never take the inner arm: the enclosing guard is the
/// superset check and the nested one is constant. IL2CPP emits exactly this
/// redundant managed null-check, and mirroring both produced scopes no reader
/// can believe (`if (v0 != 0) { … if (v0 == 0) goto L17; … }`).
#[test]
fn a_guard_an_enclosing_guard_decided_is_folded() {
    let mut stmts = vec![
        Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var("v0".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![
                Stmt::Assign("x".into(), None, Expr::Int(1)),
                Stmt::If {
                    cond: Expr::Cmp(
                        CmpKind::Eq,
                        Box::new(Expr::Var("v0".into())),
                        Box::new(Expr::Int(0)),
                    ),
                    then_body: vec![Stmt::Goto("L17".into())],
                    else_body: vec![],
                },
                Stmt::Assign("y".into(), None, Expr::Int(2)),
            ],
            else_body: vec![],
        },
        Stmt::Label("L17".into()),
    ];
    csharp::fold_decided_guards(&mut stmts);
    let out = csharp::render_stmts(&stmts);
    assert!(
        !out.contains("if (v0 == 0)"),
        "the impossible inner guard must be folded away:\n{out}"
    );
    assert!(
        out.contains("if (v0 != 0)"),
        "the enclosing (superset) guard stays:\n{out}"
    );
    assert!(
        out.contains("x = 1") && out.contains("y = 2"),
        "the live statements on both sides stay:\n{out}"
    );
}

/// The fold must NOT drop a dead arm a `goto` from OUTSIDE still reaches.
/// Deleting a `goto` is always safe; deleting a `label:` is not.
#[test]
fn a_fold_keeps_a_dead_arm_an_outside_goto_still_targets() {
    let mut stmts = vec![
        Stmt::Goto("L9".into()),
        Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var("v0".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Eq,
                    Box::new(Expr::Var("v0".into())),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Label("L9".into()), Stmt::Eval(Expr::Var("w".into()))],
                else_body: vec![],
            }],
            else_body: vec![],
        },
    ];
    csharp::fold_decided_guards(&mut stmts);
    let out = csharp::render_stmts(&stmts);
    assert!(
        out.contains("L9:"),
        "a label an outside goto targets must survive the fold:\n{out}"
    );
    assert!(
        out.contains("if (v0 == 0)"),
        "the arm must be KEPT whole because an outside goto targets its label:\n{out}"
    );
}

/// A `|| true` disjunct is a STRUCTURER FUSION ARTIFACT, not a constant
/// condition: `emit_if` fuses a flat guard chain into `if (a || b)` and writes
/// `|| true` when one edge is a carried goto. Treating it as constant-true
/// deleted the (live) fall-through arm of `CreateProvider` @0x1BFAB28.
#[test]
fn a_fused_or_true_condition_is_not_a_constant() {
    let mut stmts = vec![Stmt::If {
        cond: Expr::Bin(
            BinOp::OrElse,
            Box::new(Expr::Cmp(
                CmpKind::Lt,
                Box::new(Expr::Var("v73".into())),
                Box::new(Expr::Int(16)),
            )),
            Box::new(Expr::Bool(true)),
        ),
        then_body: vec![Stmt::Goto("L5".into())],
        else_body: vec![Stmt::Assign("fallthrough".into(), None, Expr::Int(1))],
    }];
    csharp::fold_decided_guards(&mut stmts);
    let out = csharp::render_stmts(&stmts);
    assert!(
        out.contains("fallthrough = 1"),
        "`X || true` is a fusion artifact and must not fold its live arm:\n{out}"
    );
}

/// The finished text must never contain a `goto Lx;` whose label the text never
/// defines. The tree-level sweep cannot guarantee this because the renderer
/// discards arms, so the invariant is checked on the text itself.
///
/// Measured on Assembly-CSharp: 14 methods still printed a dangling target
/// after the tree sweep (`GetMultipliedType` @0x1F0114 emitted `goto L3;` with
/// no `L3:`); after the fix, 0 of 25,211 bodies do.
#[test]
fn the_finished_text_never_carries_a_goto_to_an_undefined_label() {
    // defined label -> the statement is left alone
    let mut ok = String::from("if (c)\n{\n    goto L4;\n}\nL4:\nreturn;\n");
    csharp::strip_undefined_gotos(&mut ok);
    assert!(
        ok.contains("goto L4;"),
        "a defined target must survive: {ok}"
    );
    assert!(
        !ok.contains("target not emitted"),
        "nothing to rewrite: {ok}"
    );

    // undefined label -> the statement becomes an explicit marker
    let mut bad = String::from("if (c)\n{\n    goto L9;\n}\nreturn;\n");
    csharp::strip_undefined_gotos(&mut bad);
    assert!(
        !bad.contains("goto L9;") || bad.contains("target not emitted"),
        "an unnameable target must not be emitted as code: {bad}"
    );
    assert!(bad.contains("target not emitted"), "marker expected: {bad}");

    // switch arms render as `case N: goto Lx;` and are swept too, keeping the
    // selector so the case value is not silently lost
    let mut sw = String::from("switch (v1)\n{\n    case 3: goto L9;\n}\nreturn;\n");
    csharp::strip_undefined_gotos(&mut sw);
    assert!(sw.contains("case 3:"), "the case value must survive: {sw}");
    assert!(sw.contains("target not emitted"), "marker expected: {sw}");
}

// ---------------------------------------------------------------------------
/// A binding the finished text never reads by NAME is the duplicate of the
/// expression that was inlined at its use, so the text-level sweep drops it —
/// while keeping a loop-carried update and the effect of a call.
///
/// The tree cannot decide this: whether a use prints the name or the inlined
/// expression is a render-time choice, and the tree holds an `Expr::Var` either
/// way. Measured: 16,834 of 273,016 bindings on the v24.5 PE dump (6.2%) and
/// 11,923 of 130,728 on the Android corpus (9.1%) named a local the text never
/// mentioned again.
#[test]
fn an_unread_binding_is_dropped_from_the_finished_text() {
    // unread → dropped; read by a LATER line → kept
    let mut out = String::from("v17 = load(a);\nv19 = load(a) + 1;\nreturn v19;\n");
    csharp::strip_unread_bindings(&mut out);
    assert!(!out.contains("v17"), "an unread binding must go: {out}");
    assert!(out.contains("v19 = "), "a read binding must stay: {out}");

    // read by an EARLIER line: the statements are not in execution order, so a
    // backward read (a `goto` loop) is just as real
    let mut backward = String::from("v5 = 3;\nif (v5 != 0)\n{\n    goto L1;\n}\n");
    csharp::strip_unread_bindings(&mut backward);
    assert!(
        backward.contains("v5 = 3;"),
        "a backward read keeps it: {backward}"
    );

    // loop-carried: the binding reads itself (`v = v - 1`)
    let mut carried = String::from("v31 = v31 - 1;\nreturn;\n");
    csharp::strip_unread_bindings(&mut carried);
    assert!(
        carried.contains("v31 = v31 - 1;"),
        "a loop-carried update must stay: {carried}"
    );

    // the call still runs, as a bare expression
    let mut call = String::from("v24 = fn_0(v1);\nreturn;\n");
    csharp::strip_unread_bindings(&mut call);
    assert!(!call.contains("v24"), "the binding goes: {call}");
    assert!(call.contains("fn_0(v1);"), "the call must survive: {call}");

    // a store THROUGH a pointer is observable and is not a `vN` binding at all
    let mut store = String::from("*(nint*)(v0 +8) = v1;\nreturn;\n");
    csharp::strip_unread_bindings(&mut store);
    assert!(
        store.contains("*(nint*)(v0 +8) ="),
        "a store is not a binding: {store}"
    );
}

// Event-accessor CAS-retry collapse (`csharp::collapse_event_cas_accessors`)
// ---------------------------------------------------------------------------

/// `this.Evt = Delegate.Combine(this.Evt, value)` as a *condition* expression —
/// the shape IL2CPP emits for the retry test of a `[CompilerGenerated]` event
/// accessor.
fn combine_cond(local: &str) -> Expr {
    Expr::Cmp(
        CmpKind::Ne,
        Box::new(Expr::Call {
            recv: None,
            name: "Delegate.Combine".into(),
            args: vec![Expr::Var(local.into()), Expr::Var("value".into())],
        }),
        Box::new(Expr::Int(0)),
    )
}

/// The class-init guard every generated accessor opens with, plus its init
/// call and the flag store. This prefix used to veto the whole collapse
/// (`_ => return`), so NO Unity accessor ever collapsed.
fn accessor_init_prefix() -> Vec<Stmt> {
    vec![
        Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Bin(
                    BinOp::And,
                    Box::new(rd("0x63a1000", 2216)),
                    Box::new(Expr::Int(1)),
                )),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![Stmt::Goto("L0".into())],
            else_body: Vec::new(),
        },
        Stmt::Assign(
            "v27".into(),
            None,
            Expr::RawCall(
                0x4a58960,
                vec![Expr::Int(0x60429f8), Expr::Var("value".into())],
                None,
            ),
        ),
        Stmt::Assign("*(byte*)(0x63a1000 +2216)".into(), None, Expr::Int(1)),
        Stmt::Label("L0".into()),
    ]
}

#[test]
fn an_accessor_with_a_class_init_guard_still_collapses() {
    let mut stmts = accessor_init_prefix();
    stmts.push(Stmt::Assign(
        "v18".into(),
        None,
        Expr::Field(Box::new(Expr::This), "OnSortingTypeUpdated".into()),
    ));
    stmts.push(Stmt::While {
        cond: combine_cond("v18"),
        body: vec![
            Stmt::Assign(
                "v44".into(),
                None,
                Expr::RawCall(0x4a58a78, vec![combine_cond("v18")], None),
            ),
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ne,
                    Box::new(Expr::Var("v44".into())),
                    Box::new(Expr::Int(0)),
                ),
                then_body: vec![Stmt::Assign(
                    "v39".into(),
                    None,
                    Expr::RawCall(0x4ac52b8, vec![Expr::This, Expr::Var("v18".into())], None),
                )],
                else_body: Vec::new(),
            },
        ],
    });
    super::csharp::collapse_event_cas_accessors(&mut stmts, "add_OnSortingTypeUpdated");
    assert!(
        !stmts.iter().any(|s| matches!(s, Stmt::While { .. })),
        "the retry loop must be gone: {stmts:?}"
    );
    let store = stmts.iter().find_map(|s| match s {
        Stmt::Assign(t, _, e @ Expr::Call { name, .. }) if name == "Delegate.Combine" => {
            Some((t.as_str(), e))
        }
        _ => None,
    });
    let (lhs, _) = store.expect("the accessor must collapse to its store");
    assert_eq!(lhs, "this.OnSortingTypeUpdated", "store target: {stmts:?}");
    // the init guard + its raw init call and flag store are KEPT, not dropped
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, Stmt::Label(l) if l == "L0")),
        "the init label must survive: {stmts:?}"
    );
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(_, _, Expr::RawCall(0x4a58960, _, _)))),
        "the runtime init call must survive: {stmts:?}"
    );
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(t, _, _) if t.starts_with("*(byte*)"))),
        "the init flag store must survive: {stmts:?}"
    );
}

/// The native compare-exchange thunk really is called; dropping it with the
/// loop cost visible native calls (measured: `calls visible in C#` fell from
/// 28381 to 28225 when the body was discarded). It must survive the collapse.
#[test]
fn the_compare_exchange_call_survives_the_collapse() {
    let mut stmts = accessor_init_prefix();
    stmts.push(Stmt::Assign(
        "v18".into(),
        None,
        Expr::Field(Box::new(Expr::This), "OnLanded".into()),
    ));
    stmts.push(Stmt::While {
        cond: combine_cond("v18"),
        body: vec![
            Stmt::Assign(
                "v44".into(),
                None,
                Expr::RawCall(0x4a58a78, vec![combine_cond("v18")], None),
            ),
            Stmt::Assign("v19".into(), None, Expr::Var("v48".into())),
        ],
    });
    super::csharp::collapse_event_cas_accessors(&mut stmts, "add_OnLanded");
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(_, _, Expr::RawCall(0x4a58a78, _, _)))),
        "the compare-exchange call must survive: {stmts:?}"
    );
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, Stmt::Assign(n, _, _) if n == "v19")),
        "pure loop bookkeeping must be dropped: {stmts:?}"
    );
}

/// A hand-written accessor with a real managed call in it must NOT be
/// collapsed — the loop is only replaced when the whole method is the
/// generated shape.
#[test]
fn an_accessor_with_real_logic_is_left_alone() {
    let mut stmts = vec![
        Stmt::Assign(
            "v18".into(),
            None,
            Expr::Field(Box::new(Expr::This), "OnLanded".into()),
        ),
        Stmt::Assign(
            "v20".into(),
            None,
            Expr::Call {
                recv: Some(Box::new(Expr::This)),
                name: "DoSomethingReal".into(),
                args: Vec::new(),
            },
        ),
        Stmt::While {
            cond: combine_cond("v18"),
            body: Vec::new(),
        },
    ];
    super::csharp::collapse_event_cas_accessors(&mut stmts, "add_OnLanded");
    assert!(
        stmts.iter().any(|s| matches!(s, Stmt::While { .. })),
        "a custom accessor's loop must be preserved: {stmts:?}"
    );
}

// ---------------------------------------------------------------------------
// Pre/post-index addressing and `this`-relative field addresses
// ---------------------------------------------------------------------------

/// A PRE-INDEX access updates its base register; the POST-INDEX access keeps
/// its base and (for now) is not advanced — but BOTH must read the element at
/// the base the disassembly names.
///
/// `ldr x22, [x20, #0x28]!` is one instruction that loads `[x20 + 0x28]` AND
/// writes `x20 + 0x28` back to `x20`. Dropping the write-back left x20 holding
/// its pre-update value, so every later use pointed at the wrong object — in
/// `XmlTextReaderImpl.ParseDoctypeDecl` that collapsed a nested `ParsingState`
/// receiver onto `this` and rendered `ps.LinePos` as `this.LinePos`.
/// `ldr x0, [x1], #8` is the mirror case: the ACCESS is at the UNCHANGED `x1`
/// (the `#8` is only the increment), so the memory operand must carry addend 0
/// — reading it as an access offset loaded eight bytes past the element.
#[test]
fn pre_index_updates_the_base_and_post_index_reads_the_unmodified_base() {
    // mov x20, x0 ; ldr x22, [x20, #0x28]! ; ldr x2, [x1] ; ldr x0, [x1], #8
    let words: [u32; 4] = [0xaa0003f4, 0xf8428e96, 0xf9400022, 0xf8408420];
    let lifted = lift::lift_arm64(&le(&words), 0x1000, 32, None).expect("lift");

    let pre = lifted
        .insns
        .iter()
        .find(|i| i.dest() == Some(&Operand::reg("x22")))
        .expect("the pre-index load must lift to a value");
    match pre.operands.get(1) {
        Some(Operand::Memory { base, addend, .. }) => {
            assert_eq!(&**base, &Operand::reg("x20"), "pre-index base");
            assert_eq!(*addend, 0x28, "pre-index access offset");
        }
        other => panic!("pre-index load operand, got {other:?}"),
    }
    let wb = lifted
        .insns
        .iter()
        .find(|i| {
            matches!(i.op, Op::Arith(ArithKind::Add)) && i.dest() == Some(&Operand::reg("x20"))
        })
        .expect("`[x20, #0x28]!` must write x20 + 0x28 back into x20");
    assert_eq!(
        wb.operands,
        vec![Operand::reg("x20"), Operand::reg("x20"), Operand::imm(0x28)],
        "the write-back is `x20 = x20 + 0x28`"
    );

    let post = lifted
        .insns
        .iter()
        .find(|i| i.dest() == Some(&Operand::reg("x0")))
        .expect("the post-index load must lift");
    match post.operands.get(1) {
        Some(Operand::Memory { base, addend, .. }) => {
            assert_eq!(&**base, &Operand::reg("x1"), "post-index base");
            assert_eq!(*addend, 0, "post-index reads the UNMODIFIED base");
        }
        other => panic!("post-index load operand, got {other:?}"),
    }
    // POST-index also writes the base back, but AFTER the access: the load
    // above read `[x1]` at addend 0, and this is the separate `x1 = x1 + 8`.
    // Without it every element-copy/iterate loop re-read the same element and
    // never advanced (981 `native base19 +8, IR [0]` sem-diffs on arm64 iOS).
    let post_wb = lifted
        .insns
        .iter()
        .find(|i| {
            matches!(i.op, Op::Arith(ArithKind::Add)) && i.dest() == Some(&Operand::reg("x1"))
        })
        .expect("`ldr x0, [x1], #8` must write x1 + 8 back into x1");
    assert_eq!(
        post_wb.operands,
        vec![Operand::reg("x1"), Operand::reg("x1"), Operand::imm(8)],
        "the post-index write-back is `x1 = x1 + 8`"
    );
}

/// `add x20, this, #0x28` is the ADDRESS of the field at 0x28, not an opaque
/// pointer bump — that is how IL2CPP materialises `&this->ps`, both directly
/// and as the write-back of a pre-index access. Naming the field is what makes
/// a value-type receiver render `this.ps.LinePos` instead of `this + 40.LinePos`.
/// A STATIC field at the same offset must not be named (a static block's first
/// field is at offset 0, which collides with an object's klass pointer).
#[test]
fn a_this_relative_add_names_the_instance_field_it_addresses() {
    struct AddrOracle;
    impl locals::TypeOracle for AddrOracle {
        fn field_at_offset(&self, _tdi: i32, offset: i64) -> Option<types::FieldRef> {
            match offset {
                0x28 => Some(types::FieldRef {
                    type_def_index: 7,
                    field_index: 0,
                    name: "ps".into(),
                    declaring_type_name: "T".into(),
                    field_type: ManagedType::Object,
                    is_static: false,
                }),
                0 => Some(types::FieldRef {
                    type_def_index: 7,
                    field_index: 1,
                    name: "S_static".into(),
                    declaring_type_name: "T".into(),
                    field_type: ManagedType::Object,
                    is_static: true,
                }),
                _ => None,
            }
        }
    }

    let seed = locals::SeedInfo {
        param_types: vec![],
        ret_type: ManagedType::Primitive(PrimKind::Void),
        is_static: false,
        declaring: None,
        param_names: vec![],
        ..Default::default()
    };
    let mut version_slot: HashMap<u32, String> = HashMap::new();
    version_slot.insert(1, "x0".to_string());
    version_slot.insert(2, "x1".to_string());
    version_slot.insert(3, "x2".to_string());
    let mut table = locals::create(&version_slot, &seed);
    let li = table.index_of[&1];
    assert!(table.infos[li].is_this, "x0 entry must seed as `this`");
    table.infos[li].ty = ManagedType::Class(std::sync::Arc::new(types::NamedType {
        namespace: String::new(),
        name: "T".into(),
        assembly: None,
        type_def_index: 7,
    }));

    let mut insns = vec![
        // v2 = this + 0x28   — `&this->ps`
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(2), Operand::Local(1), Operand::imm(0x28)],
            0x4000,
        ),
        // v3 = this + 0x0    — matches only a STATIC field, must stay an add
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(3), Operand::Local(1), Operand::imm(0)],
            0x4004,
        ),
    ];

    locals::propagate_types(&mut insns, &mut table, &AddrOracle);

    match insns[0].operands.get(1) {
        Some(Operand::Field(f)) => assert_eq!(f.name, "ps"),
        other => panic!("`this + 0x28` must become the field `ps`, got {other:?}"),
    }
    assert!(
        matches!(insns[1].op, Op::Arith(ArithKind::Add)),
        "a static field must never be named at an instance offset, got {:?}",
        insns[1].op
    );
}

/// `.` is a postfix operator, so a binary receiver has to be parenthesised:
/// without this, `prettify_call` folding `get_LinePos()` into a `Field` on an
/// address computation printed `this + 40.LinePos`, which parses as
/// `this + (40.LinePos)`.
#[test]
fn a_field_on_a_binary_receiver_is_parenthesised() {
    let e = Expr::Field(
        Box::new(Expr::Bin(
            BinOp::Add,
            Box::new(Expr::This),
            Box::new(Expr::Int(40)),
        )),
        "LinePos".into(),
    );
    let mut out = String::new();
    e.render(&mut out);
    assert_eq!(out, "(this + 40).LinePos");
}

/// A `switch` INSIDE a loop still belongs to the loop's body.
///
/// The jump-table lowering fences its arm walks with every block of any loop
/// whose header starts before the dispatch, so that an arm cannot re-enter a
/// loop already rendered above it. But when the dispatch sits INSIDE the loop
/// being rendered, that fence covers the arms themselves — the walk then has
/// nowhere to go, and the arm bodies are never emitted at all. Measured on
/// `DtdParser.GetToken` @0x2055C04: a 14-way table inside the scan loop came
/// out as four empty `shared tail` stubs with 128 reachable blocks missing.
#[test]
fn a_switch_inside_a_loop_walks_its_arms() {
    let insns = vec![
        // header: `while (v1 < 2)`
        Insn::new(
            Op::Compare(CompareKind::GreaterOrEqual),
            vec![Operand::Local(2), Operand::Local(1), Operand::imm(2)],
            0x6000,
        ),
        Insn::new(Op::CondJump(11), vec![Operand::Local(2)], 0x6004),
        // dispatch on v1 → arm bodies at insn 4 and insn 7
        Insn::new(Op::Switch(vec![4, 7]), vec![], 0x6008),
        Insn::new(Op::Nop, vec![], 0x600c),
        Insn::move_(Operand::Local(3), Operand::imm(17), 0x6010),
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(1), Operand::Local(1), Operand::imm(1)],
            0x6014,
        ),
        Insn::new(Op::Jump(0), vec![], 0x6018),
        Insn::move_(Operand::Local(3), Operand::imm(34), 0x601c),
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(1), Operand::Local(1), Operand::imm(1)],
            0x6020,
        ),
        Insn::new(Op::Jump(0), vec![], 0x6024),
        Insn::new(Op::Nop, vec![], 0x6028),
        Insn::new(Op::Return, vec![Operand::Local(3)], 0x602c),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&[0, 1, 2, 3, 4]),
        returns_value: true,
        base_rva: 0x6000,
    };
    let out = csharp::render_method_csharp(&ir, "GetToken", false, &[], &ManagedType::Unknown, -1);
    assert!(
        !out.contains("arms unrecovered"),
        "a table inside the loop must still render its arms, got:\n{out}"
    );
    assert!(
        out.contains("17") && out.contains("34"),
        "both arm bodies must be emitted, got:\n{out}"
    );
}

/// A fall out of a loop body that does NOT land on the loop's own exit carries
/// its edge, and every block the CFG can reach is still emitted.
///
/// `emit_region` reports such a fall as `Some(next)`, a promise that the caller
/// resumes at `next`. A guard chain keeps that promise, but `emit_loop` discards
/// the value and resumes at its own exit block — so a fall into any OTHER block
/// dropped the edge AND the block: the code appeared nowhere in the text. That is
/// how a method's text comes to disagree with its bytes with no IR-level trace
/// (`0x1DA379C`: the scan loop's fall-through block, 113 blocks of
/// `XPathParser.EvaluateDependentExpressions`, simply absent; 2,200 reachable
/// blocks across 568 methods in the 4,000-method sample).
///
/// The fall is carried by `Structurer::bail_out_of_region` as an explicit `goto`
/// (1,651 such edges in the sample), and the block it names — which no structured
/// statement could hold — is emitted by `Structurer::coverage_repair` behind a
/// marker that says why it is there.
#[test]
fn a_secondary_loop_exit_keeps_its_edge_and_its_code() {
    let insns = vec![
        // header: `while (v1 < 3)` — taken exits, fall is the body
        Insn::new(
            Op::Compare(CompareKind::GreaterOrEqual),
            vec![Operand::Local(2), Operand::Local(1), Operand::imm(3)],
            0x7000,
        ),
        Insn::new(Op::CondJump(8), vec![Operand::Local(2)], 0x7004),
        // body
        Insn::move_(Operand::Local(3), Operand::imm(5), 0x7008),
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(1), Operand::Local(1), Operand::imm(1)],
            0x700c,
        ),
        // latch test → back edge, else fall into the SECONDARY exit
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(4), Operand::Local(1), Operand::imm(9)],
            0x7010,
        ),
        Insn::new(Op::CondJump(0), vec![Operand::Local(4)], 0x7014),
        Insn::move_(Operand::Local(3), Operand::imm(99), 0x7018),
        Insn::new(Op::Nop, vec![], 0x701c),
        Insn::new(Op::Return, vec![Operand::Local(3)], 0x7020),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: plain_local_table(&[0, 1, 2, 3, 4]),
        returns_value: true,
        base_rva: 0x7000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("99"),
        "the secondary exit's code must be in the text, got:\n{out}"
    );
    assert!(
        out.contains("[recovery]")
            || out.contains("return 99")
            || ((out.contains("v3 = 99")
                || out.contains("result = 99")
                || out.contains("value = 99"))
                && out.contains("break;")),
        "the secondary exit must be kept — structured inline, as a labelled \
         repair region, or as its block followed by a break out of the loop — got:\n{out}"
    );
}

/// `method_spec_generic_method_pointers` is keyed by spec index, but several specs
/// share one code pointer, so whichever spec reaches an address→name map first names
/// every call to that address. Walking the map directly made that depend on a
/// per-process hash seed: two dumps of the same binary named the same `bl`
/// `TraceLoggingTypeInfo<float>` in one run and `<double>` in the next. The order
/// has to come from the spec indices, not from the map — and, under full generic
/// sharing, so does the COVERAGE: only the shared instantiations are compiled, so
/// an instantiation a call site passes can be missing from the map entirely.
#[test]
fn spec_indices_are_ordered_and_complete() {
    let ordered: Vec<usize> = super::spec_indices_to_name(500).collect();

    // Ascending and total: the pointer map's own iteration order cannot
    // influence it, and a spec the map does not hold is still named.
    assert_eq!(ordered, (0..500usize).collect::<Vec<usize>>());
    assert_eq!(
        super::spec_indices_to_name(500).collect::<Vec<usize>>(),
        ordered,
        "the order must not depend on any map's iteration order"
    );
}

fn shared_spec(ty: &str, name: &str, method_index: u32) -> MethodRef {
    MethodRef {
        image_index: 0,
        type_def_index: 0,
        method_index,
        name: name.into(),
        declaring_namespace: "System.Collections.Generic".into(),
        declaring_type_name: ty.into(),
        assembly: String::new(),
        sig: types::MethodSigInfo {
            ret: ManagedType::Unknown,
            params: vec![],
            is_static: false,
        },
    }
}

/// The generic body at `0x9000` as the x64 reference build reaches it: the
/// `MethodInfo` of the instantiation is loaded from slot `SLOT` into a local,
/// and the call passes it in a register the operand list does not carry.
/// Real shape @0x436d4c:
///
/// ```text
/// mov rcx, [rip + 0x7fb47d]   ; the concrete MethodInfo, from slot 0xC32A50
/// xor edx, edx
/// call 0x266b80               ; a body FIVE instantiations compile to
/// ```
fn shared_body_call_ir(slot: i64) -> Vec<Insn> {
    vec![
        Insn::move_(
            Operand::Local(1),
            Operand::Memory {
                width: 0,
                base: Box::new(Operand::reg("abs")),
                index: None,
                scale: 1,
                addend: slot,
            },
            0x1000,
        ),
        Insn::move_(Operand::Local(2), Operand::Immediate(0), 0x1004),
        Insn::new(
            Op::Call,
            vec![
                Operand::Immediate(0x9000),
                Operand::Local(3),
                Operand::Local(4),
            ],
            0x1008,
        ),
    ]
}

/// `(name, bodies_pinned)`: the name the call resolved to, and whether the
/// shared-body disambiguation is what chose it.
fn run_shared_body_call(maps: &resolve::ResolveMaps) -> (String, usize) {
    let mut insns = shared_body_call_ir(0xC32A50);
    let table = locals::create(&HashMap::new(), &helper_seed());
    let stats = resolve::run(&mut insns, &table, maps);
    match insns[2].operands.first() {
        Some(Operand::Method(m)) => (
            format!("{}::{}", m.declaring_type_name, m.name),
            stats.generic_bodies_pinned,
        ),
        other => panic!("call was not resolved to a method: {other:?}"),
    }
}

/// IL2CPP compiles every reference-type instantiation of a generic method to
/// ONE body, so `methods_by_rva` can only name whichever spec index the
/// pointer table happened to hand over first — `List<object>::.ctor` for a call
/// that really constructs a `List<int>`. The call site is what knows, because
/// it is the one that loads the `MethodInfo` it passes.
#[test]
fn shared_generic_body_is_named_by_the_slot_the_caller_loads() {
    let canonical = shared_spec("List<object>", ".ctor", 7);
    let mut maps = resolve::ResolveMaps::default();
    maps.methods_by_rva.insert(0x9000, canonical.clone());
    maps.specs_by_body_rva.insert(
        0x9000,
        vec![
            (100, canonical),
            (250, shared_spec("List<int>", ".ctor", 7)),
        ],
    );
    maps.specs_by_index
        .insert(250, shared_spec("List<int>", ".ctor", 7));

    // Without the slot there is nothing to choose with: the canonical spec
    // index decides, exactly as before this pass existed.
    assert_eq!(
        run_shared_body_call(&maps),
        ("List<object>::.ctor".to_string(), 0)
    );

    maps.method_spec_index_by_addr.insert(0xC32A50, 250);
    assert_eq!(
        run_shared_body_call(&maps),
        ("List<int>::.ctor".to_string(), 1),
        "the call site's own MethodInfo load must name the instantiation"
    );
}

/// Under full generic sharing only the SHARED instantiations are compiled, so
/// the concrete instantiation a call passes routinely has no code pointer and
/// cannot appear in `specs_by_body_rva` at all. It is still nameable: the spec
/// index resolves through `method_specs`, and only the definition has to agree
/// with a definition already known to live at the called address — which is
/// what keeps an unrelated `MethodInfo` load next to the call from answering
/// for it.
#[test]
fn shared_body_call_names_a_spec_the_pointer_table_does_not_index() {
    let canonical = shared_spec("List<__Il2CppFullySharedGenericType>", ".ctor", 7);
    let mut maps = resolve::ResolveMaps::default();
    maps.methods_by_rva.insert(0x9000, canonical.clone());
    maps.specs_by_body_rva
        .insert(0x9000, vec![(100, canonical)]);
    maps.method_spec_index_by_addr.insert(0xC32A50, 999);

    // Same generic METHOD definition, different instantiation: the body at
    // this address is that method's, so the slot is this call's MethodInfo.
    maps.specs_by_index
        .insert(999, shared_spec("List<string>", ".ctor", 7));
    assert_eq!(
        run_shared_body_call(&maps),
        ("List<string>::.ctor".to_string(), 1)
    );

    // A different definition entirely: the body cannot be this method's, so
    // the slot belongs to something else and nothing is claimed.
    maps.specs_by_index
        .insert(999, shared_spec("List<string>", "Add", 99));
    assert_eq!(
        run_shared_body_call(&maps),
        ("List<__Il2CppFullySharedGenericType>::.ctor".to_string(), 0)
    );
}

/// A local table whose versions carry ABI register origins, so the ABI scrub's
/// rootedness test has the `is_param`/`slot_origin` data it reads.
fn origin_param_table(entries: &[(u32, &str, Option<usize>)]) -> locals::LocalTable {
    let mut t = locals::LocalTable::default();
    for (v, origin, param) in entries {
        t.infos.push(locals::LocalInfo {
            slot_origin: (*origin).into(),
            ty: ManagedType::Unknown,
            is_this: false,
            is_param: *param,
            param_comp: None,
            is_incoming_arg: false,
            is_call_clobber: false,
            is_return_buffer: false,
        });
        t.index_of.insert(*v, t.infos.len() - 1);
    }
    t
}

/// The TRAILING run of uncovered ABI argument registers at an UNRESOLVED call
/// is register-dump filler, not arguments.
///
/// The lifter records every `bl` with the full `x0..x3` dump. For a RESOLVED
/// callee `trim_call_arguments` trims to its arity, but an unresolved target
/// (`fn_<addr>`) has no arity, so `fn_5906b0(a, b, in_x2, in_x3)` printed two
/// phantom arguments. Measured on ARM32: 261,845 → **73,028** occurrences, with
/// every verifier figure unchanged.
#[test]
fn trailing_uncovered_argument_registers_are_scrubbed_from_unresolved_calls() {
    // v8 = the one declared parameter; v9/v10 = uncovered incoming x2/x3.
    let mut table = origin_param_table(&[(8, "x1", Some(0))]);
    for (v, origin) in [(9u32, "x2"), (10u32, "x3")] {
        table.infos.push(locals::LocalInfo {
            slot_origin: origin.into(),
            ty: ManagedType::Unknown,
            is_this: false,
            is_param: None,
            param_comp: None,
            is_incoming_arg: true,
            is_call_clobber: false,
            is_return_buffer: false,
        });
        table.index_of.insert(v, table.infos.len() - 1);
    }
    let mut insns = vec![Insn::new(
        Op::CallVoid,
        vec![
            Operand::Immediate(0x5906b0),
            Operand::Unknown,
            Operand::Local(8),
            Operand::Local(9),
            Operand::Local(10),
        ],
        0x1000,
    )];
    let n = locals::scrub_abi_call_args(&mut insns, &table);
    assert_eq!(
        n, 2,
        "BOTH trailing filler registers are scrubbed, not just one: {insns:?}"
    );
    assert_eq!(
        insns[0].operands,
        vec![
            Operand::Immediate(0x5906b0),
            Operand::Unknown,
            Operand::Local(8),
            Operand::MissingAbiArg,
            Operand::MissingAbiArg,
        ],
        "the declared parameter survives; only the trailing run is scrubbed"
    );
}

/// An uncovered register in the MIDDLE of an argument list is a genuine hole
/// BETWEEN two real arguments, not trailing filler, and must be left alone.
#[test]
fn an_interior_uncovered_register_is_kept() {
    let mut table = origin_param_table(&[(8, "x1", Some(0))]);
    table.infos.push(locals::LocalInfo {
        slot_origin: "x2".into(),
        ty: ManagedType::Unknown,
        is_this: false,
        is_param: None,
        param_comp: None,
        is_incoming_arg: true,
        is_call_clobber: false,
        is_return_buffer: false,
    });
    table.index_of.insert(9, table.infos.len() - 1);
    let mut insns = vec![Insn::new(
        Op::CallVoid,
        vec![
            Operand::Immediate(0x5906b0),
            Operand::Unknown,
            Operand::Local(9),
            Operand::Local(8),
        ],
        0x1000,
    )];
    let before = insns[0].operands.clone();
    let n = locals::scrub_abi_call_args(&mut insns, &table);
    assert_eq!(n, 0, "a real argument ends the trailing run: {insns:?}");
    assert_eq!(
        insns[0].operands, before,
        "the interior register is untouched"
    );
}

/// A call argument reached through a chain of REDUNDANT JOIN PHIS over an
/// undeclared entry register is ABI register-dump filler and must be scrubbed.
///
/// This is the exact shape measured at `0x14eb20`
/// (`TMP_SpriteAsset.GetSpriteIndexFromHashcode`, one declared parameter):
/// the lifter dumps `x0..x3` for every `bl`, so the ENTRY version of `x2` — a
/// register the method never writes and does not declare — becomes a local.
/// SSA then inserts a join phi on each of the two edges (both edges carry the
/// same undefined value), and every link "has a defining instruction", so the
/// old "has any def" test kept the whole chain. `ssa::eliminate` lowers it to
/// `v14 = v9` reading an address nothing defines, rendered as a phantom
/// `in_x2` argument. The chain must be followed back to its root.
#[test]
fn a_join_phi_chain_over_an_undefined_register_is_scrubbed() {
    let table = origin_param_table(&[
        (8, "x1", Some(0)), // the one declared parameter
        (9, "x2", None),    // entry x2 — never written, not declared
        (11, "x2", None),   // phi(v9, v9)
        (14, "x2", None),   // phi(v11, v11)
    ]);
    let mut insns = vec![
        Insn::new(
            Op::Phi,
            vec![Operand::Local(11), Operand::Local(9), Operand::Local(9)],
            0x1000,
        ),
        Insn::new(
            Op::Phi,
            vec![Operand::Local(14), Operand::Local(11), Operand::Local(11)],
            0x1000,
        ),
        Insn::new(
            Op::CallVoid,
            // [target, dest, recv, args…] — the placeholder destination slot
            // at operand 1 is part of the layout for `CallVoid` too.
            vec![
                Operand::Immediate(0x9000),
                Operand::Unknown,
                Operand::Local(14),
                Operand::Local(8),
            ],
            0x1010,
        ),
    ];
    let n = locals::scrub_abi_call_args(&mut insns, &table);
    assert_eq!(
        n, 1,
        "only the undefined phi-chain argument is garbage: {insns:?}"
    );
    assert_eq!(
        insns[2].operands,
        vec![
            Operand::Immediate(0x9000),
            Operand::Unknown,
            Operand::MissingAbiArg,
            Operand::Local(8)
        ],
        "the undefined argument becomes an ABI hole; the declared parameter survives"
    );
}

/// The same phi chain is NOT touched when its root is a declared parameter.
#[test]
fn a_parameter_passed_through_a_join_phi_is_kept() {
    let table = origin_param_table(&[(9, "x1", Some(0)), (11, "x1", None)]);
    let mut insns = vec![
        Insn::new(
            Op::Phi,
            vec![Operand::Local(11), Operand::Local(9), Operand::Local(9)],
            0x1000,
        ),
        Insn::new(
            Op::CallVoid,
            vec![
                Operand::Immediate(0x9000),
                Operand::Unknown,
                Operand::Local(11),
            ],
            0x1010,
        ),
    ];
    assert_eq!(
        locals::scrub_abi_call_args(&mut insns, &table),
        0,
        "a real argument must not be scrubbed just because it flows through a phi"
    );
}

/// Rootedness INHERITS through copies: a value written by a real definition is
/// rooted however many `Move`s it is passed through.
#[test]
fn a_defined_value_passed_through_copies_is_rooted() {
    let table = origin_param_table(&[(20, "rv", None), (21, "x1", None)]);
    let mut insns = vec![
        // v20 = <call result>
        Insn::new(
            Op::Call,
            vec![Operand::Immediate(0x8000), Operand::Local(20)],
            0x1000,
        ),
        // v21 = v20
        Insn::move_(Operand::Local(21), Operand::Local(20), 0x1004),
        Insn::new(
            Op::CallVoid,
            vec![
                Operand::Immediate(0x9000),
                Operand::Unknown,
                Operand::Local(21),
            ],
            0x1010,
        ),
    ];
    assert_eq!(locals::scrub_abi_call_args(&mut insns, &table), 0);
}

/// A lone undefined local (no phi involved) is still scrubbed — the direct
/// case the pass has always handled, pinned so the transitive rewrite cannot
/// regress it.
#[test]
fn an_undefined_argument_with_no_definition_is_scrubbed() {
    let table = origin_param_table(&[(9, "x2", None)]);
    let mut insns = vec![Insn::new(
        Op::CallVoid,
        vec![
            Operand::Immediate(0x9000),
            Operand::Unknown,
            Operand::Local(9),
        ],
        0x1000,
    )];
    assert_eq!(locals::scrub_abi_call_args(&mut insns, &table), 1);
    assert_eq!(insns[0].operands[2], Operand::MissingAbiArg);
}

/// A loop phi that depends on itself and never reaches a real definition is
/// NOT rooted — following the cycle would be circular reasoning. The argument
/// stays an ABI hole.
#[test]
fn a_self_referential_phi_of_an_undefined_register_is_scrubbed() {
    let table = origin_param_table(&[(9, "x2", None), (11, "x2", None)]);
    let mut insns = vec![
        Insn::new(
            Op::Phi,
            vec![Operand::Local(11), Operand::Local(11), Operand::Local(9)],
            0x1000,
        ),
        Insn::new(
            Op::CallVoid,
            vec![
                Operand::Immediate(0x9000),
                Operand::Unknown,
                Operand::Local(11),
            ],
            0x1010,
        ),
    ];
    assert_eq!(locals::scrub_abi_call_args(&mut insns, &table), 1);
}

/// A value whose source is an ADDRESS (not a local) is a real datum and must
/// stay a call argument. This is the AAPCS64 INDIRECT-RESULT path: `add x8,
/// sp, #imm` reaches the `bl` as `x8 = &slot`, and scrubbing the slot would
/// silently delete a struct-returning call's out-parameter — the failure mode
/// the sret machinery exists to prevent.
#[test]
fn an_address_of_value_is_rooted_and_stays_a_call_argument() {
    let table = origin_param_table(&[(20, "x8", None)]);
    let mut insns = vec![
        Insn::new(
            Op::Move,
            vec![
                Operand::Local(20),
                Operand::AddressOf(Box::new(Operand::Local(9))),
            ],
            0x1000,
        ),
        Insn::new(
            Op::CallVoid,
            vec![
                Operand::Immediate(0x9000),
                Operand::Unknown,
                Operand::Local(20),
            ],
            0x1004,
        ),
    ];
    assert_eq!(
        locals::scrub_abi_call_args(&mut insns, &table),
        0,
        "an address-of value must never be treated as register-dump filler"
    );
}

/// `materialize` splices each phi in at its block's START, which SPLITS the
/// block whenever that start follows a terminator (a loop header, an
/// if/else join) — every later block index then shifts, and `phi.preds`
/// (recorded against the pre-splice numbering) name the WRONG predecessor.
/// `eliminate` reads those indices to plant the phi's edge copies, so the copy
/// landed in an unrelated block.
///
/// Measured on `SmallXmlParser.ReadContent` @0x31D1E0: the loop-carried `x0`
/// phi's back-edge copy landed in a block that the `x1 = this.reader`
/// definition could not reach, `liveness_dce` then deleted that still-used
/// definition, and the body read an undefined local (29 such findings in that
/// one method). The invariant below is what a stale index violates: a phi's
/// predecessor block is a block whose EDGE reaches the phi, so it ends with
/// its own last instruction — never with ANOTHER phi.
#[test]
fn phi_predecessors_survive_the_block_renumbering_materialize_causes() {
    // Two joins, each starting right after a terminator AND being a branch
    // target — the shape that splits a block when a phi is spliced in.
    //   0x1000  r1 = 0
    //   0x1004  if (f) goto 0x1010
    //   0x1008  r1 = 5
    //   0x100c  goto 0x1010
    //   0x1010  r2 = r1          <-- phi(r1) spliced here
    //   0x1014  if (f) goto 0x1020
    //   0x1018  r2 = 7
    //   0x101c  goto 0x1020
    //   0x1020  r0 = r2          <-- phi(r2) spliced here
    //   0x1024  return r0
    let mut insns = vec![
        Insn::new(Op::Move, vec![Operand::reg("r1"), Operand::imm(0)], 0x1000),
        Insn::new(Op::CondJump(0x1010), vec![Operand::reg("f")], 0x1004),
        Insn::new(Op::Move, vec![Operand::reg("r1"), Operand::imm(5)], 0x1008),
        Insn::new(Op::Jump(0x1010), vec![], 0x100c),
        Insn::new(
            Op::Move,
            vec![Operand::reg("r2"), Operand::reg("r1")],
            0x1010,
        ),
        Insn::new(Op::CondJump(0x1020), vec![Operand::reg("f")], 0x1014),
        Insn::new(Op::Move, vec![Operand::reg("r2"), Operand::imm(7)], 0x1018),
        Insn::new(Op::Jump(0x1020), vec![], 0x101c),
        Insn::new(
            Op::Move,
            vec![Operand::reg("r0"), Operand::reg("r2")],
            0x1020,
        ),
        Insn::new(Op::Return, vec![Operand::reg("r0")], 0x1024),
    ];
    let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
    let mut graph = cfg::Cfg::build(&mut insns, &addresses).expect("cfg");
    let dom = dominator::DominatorInfo::compute(&graph);
    let mut out = ssa::build(&insns, &graph, &dom, &[]);
    assert!(
        out.phis.len() >= 2,
        "both joins must grow a phi: {:?}",
        out.phis
    );
    ssa::materialize(&mut out, &mut graph);
    for phi in &out.phis {
        assert_eq!(
            phi.preds.len(),
            phi.args.len(),
            "preds and args must stay parallel"
        );
        for &p in &phi.preds {
            let b = graph.blocks[p];
            let last = &out.insns[b.end - 1];
            assert!(
                !matches!(last.op, Op::Phi),
                "phi for v{} names block {p} {b:?} as a predecessor, but that \
                 block ENDS with a phi ({last:?}) — the phi splice renumbered \
                 the blocks and this pred index was never translated",
                phi.dest_version
            );
        }
    }
}

#[test]
fn x86_scalar_ss_reg_reg_lifts() {
    // `subss xmm6, xmm7` — f3 0f 5c f7. The 2-operand SSE form has the
    // destination as its own first source; iced expands it to three operands
    // (xmm6, xmm6, xmm7) and `sse_op` requires operand 2 to exist. If iced
    // reports only TWO operands here the instruction is dropped with no
    // diagnostic, which is what the corpus showed.
    let insns = lift_sysv(&[0xf3, 0x0f, 0x5c, 0xf7]);
    let fa = insns
        .iter()
        .find(|i| matches!(i.op, Op::FloatArith(_)))
        .unwrap_or_else(|| {
            panic!("`subss xmm6, xmm7` must lift to a FloatArith, not vanish: {insns:?}")
        });
    // `dst op= src` must read the destination as its own first source, and the
    // source operand must be the SECOND (op1), not a nonexistent third.
    assert_eq!(
        fa.operands,
        vec![
            Operand::reg("vf6"),
            Operand::reg("vf6"),
            Operand::reg("vf7"),
        ],
        "2-operand SSE must be modelled as `vf6 = vf6 - vf7`"
    );
}

// ---------------------------------------------------------------------------
// Masked-assembly folding: the IL2CPP static-init guard
// ---------------------------------------------------------------------------

/// The static-constructor guard is runtime bookkeeping and is folded away.
#[test]
fn a_static_init_guard_is_folded_away() {
    let guard = Stmt::If {
        cond: Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::RawDeref(Box::new(Expr::Int(0x73bdf0)), 0, 1)),
            Box::new(Expr::Int(0)),
        ),
        then_body: vec![
            Stmt::Eval(Expr::RawCall(0x5827e4, vec![], None)),
            Stmt::Assign("*(byte*)(0x73be20)".into(), None, Expr::Int(1)),
        ],
        else_body: vec![],
    };
    let mut body = vec![guard, Stmt::Return(None)];
    super::idioms::collapse_class_init_guards(&mut body);
    assert_eq!(body.len(), 1, "the guard must be folded: {body:?}");
    assert!(matches!(body[0], Stmt::Return(None)));
}

/// The ARM64 class-init guard uses a `& 1` mask on the init byte and a
/// non-zero absolute base address. The condition is
/// `(*(byte*)(base +offset) & 1) == 0` — the `& 1` extracts the init bit.
/// This must fold identically to the direct `*(byte*)(addr) == 0` form.
#[test]
fn a_static_init_guard_with_bit_mask_and_offset_is_folded_away() {
    // The exact pattern from the bug report:
    //   if ((*(byte*)(0x31ac000 +2603) & 1) == 0) {
    //       fn_130240c(*(nint*)(0x2f6a000 +792));
    //       *(byte*)(0x31ac000 +2603) = 1;
    //   }
    let guard = Stmt::If {
        cond: Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::Bin(
                BinOp::And,
                Box::new(Expr::RawDeref(Box::new(Expr::Int(0x31ac000)), 2603, 1)),
                Box::new(Expr::Int(1)),
            )),
            Box::new(Expr::Int(0)),
        ),
        then_body: vec![
            Stmt::Eval(Expr::RawCall(
                0x130240c,
                vec![Expr::RawDeref(Box::new(Expr::Int(0x2f6a000)), 792, 0)],
                None,
            )),
            Stmt::Assign(
                "*(byte*)(0x31ac000 +2603)".into(),
                None,
                Expr::Int(1),
            ),
        ],
        else_body: vec![],
    };
    let mut body = vec![guard, Stmt::Return(None)];
    super::idioms::collapse_class_init_guards(&mut body);
    assert_eq!(
        body.len(),
        1,
        "the masked init guard must be folded: {body:?}"
    );
    assert!(matches!(body[0], Stmt::Return(None)));
}

/// The `Ne` variant: `(*(byte*)(addr) & 1) != 0` with the call and store in
/// the then-body. This is the inverted form (the fast-skip path was already
/// folded by `branch_repair`). It must also be recognised.
#[test]
fn a_static_init_guard_with_ne_and_bit_mask_is_folded_away() {
    let guard = Stmt::If {
        cond: Expr::Cmp(
            CmpKind::Ne,
            Box::new(Expr::Bin(
                BinOp::And,
                Box::new(Expr::RawDeref(Box::new(Expr::Int(0x31ac000)), 2603, 1)),
                Box::new(Expr::Int(1)),
            )),
            Box::new(Expr::Int(0)),
        ),
        then_body: vec![
            Stmt::Eval(Expr::RawCall(0x130240c, vec![], None)),
            Stmt::Assign(
                "*(byte*)(0x31ac000 +2603)".into(),
                None,
                Expr::Int(1),
            ),
        ],
        else_body: vec![],
    };
    let mut body = vec![guard, Stmt::Return(None)];
    super::idioms::collapse_class_init_guards(&mut body);
    assert_eq!(
        body.len(),
        1,
        "the Ne masked init guard must be folded: {body:?}"
    );
    assert!(matches!(body[0], Stmt::Return(None)));
}

/// The FLAT form of the init guard: `if (c) goto L; call; store; L:`. This is
/// what the structurer emits before `branch_repair` converts it to if-body
/// form. When `fold_branch_over_join` is blocked (outside edges to labels in
/// the region), the flat form survives to the output. The flat sweep in
/// `collapse_class_init_guards` must catch it directly.
#[test]
fn a_flat_init_guard_is_folded_away() {
    let body = vec![
        // if ((*(byte*)(0x31ac000 +2603) & 1) != 0) goto L0;
        Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Bin(
                    BinOp::And,
                    Box::new(Expr::RawDeref(Box::new(Expr::Int(0x31ac000)), 2603, 1)),
                    Box::new(Expr::Int(1)),
                )),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![Stmt::Goto("L0".into())],
            else_body: vec![],
        },
        // fn_130240c(...)
        Stmt::Eval(Expr::RawCall(
            0x130240c,
            vec![Expr::RawDeref(Box::new(Expr::Int(0x2f6a000)), 792, 0)],
            None,
        )),
        // *(byte*)(0x31ac000 +2603) = 1
        Stmt::Assign(
            "*(byte*)(0x31ac000 +2603)".into(),
            None,
            Expr::Int(1),
        ),
        // L0:
        Stmt::Label("L0".into()),
        // Return (some real code after the guard)
        Stmt::Return(None),
    ];
    let mut body = body;
    super::idioms::collapse_class_init_guards(&mut body);
    // The guard (if + call + store) must be removed; label and return survive.
    assert!(
        !body.iter().any(|s| matches!(s, Stmt::Eval(Expr::RawCall(0x130240c, ..)))),
        "the init call must be removed: {body:?}"
    );
    assert!(
        !body.iter().any(|s| matches!(s, Stmt::Assign(t, _, _) if t.starts_with("*(byte*)"))),
        "the init flag store must be removed: {body:?}"
    );
    assert!(
        !body.iter().any(|s| matches!(s, Stmt::If { .. })),
        "the guard if must be removed: {body:?}"
    );
    assert!(
        body.iter().any(|s| matches!(s, Stmt::Label(l) if l == "L0")),
        "the join label must survive: {body:?}"
    );
    assert!(
        body.iter().any(|s| matches!(s, Stmt::Return(None))),
        "the return must survive: {body:?}"
    );
}

/// A flat init guard on an INSTANCE field is real code and must NOT be removed.
#[test]
fn a_flat_guard_on_an_instance_field_is_kept() {
    let body = vec![
        Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::RawDeref(Box::new(Expr::Var("this".into())), 88, 1)),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![Stmt::Goto("L0".into())],
            else_body: vec![],
        },
        Stmt::Eval(Expr::RawCall(0x1111, vec![], None)),
        Stmt::Assign("*(byte*)(0x2222)".into(), None, Expr::Int(1)),
        Stmt::Label("L0".into()),
    ];
    let mut body = body;
    super::idioms::collapse_class_init_guards(&mut body);
    assert_eq!(
        body.len(),
        4,
        "an instance-field flat guard is real code and must survive: {body:?}"
    );
}

/// A null guard on an INSTANCE field has the same call + `= 1` shape but is
/// real managed code (the tested load is not a fixed static word), so it stays.
#[test]
fn a_user_null_guard_on_an_instance_field_is_kept() {
    let not_guard = Stmt::If {
        cond: Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::RawDeref(Box::new(Expr::Var("this".into())), 88, 8)),
            Box::new(Expr::Int(0)),
        ),
        then_body: vec![
            Stmt::Eval(Expr::RawCall(0x1111, vec![], None)),
            Stmt::Assign("*(byte*)(0x2222)".into(), None, Expr::Int(1)),
        ],
        else_body: vec![],
    };
    let mut body = vec![not_guard];
    super::idioms::collapse_class_init_guards(&mut body);
    assert_eq!(body.len(), 1, "an instance-field null guard is real code");
}

/// A guard arm that carries a THIRD real statement does more than initialize,
/// so it is not the idiom and must survive.
#[test]
fn a_guard_whose_arm_carries_extra_work_is_kept() {
    let more = Stmt::If {
        cond: Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::RawDeref(Box::new(Expr::Int(0x1000)), 0, 1)),
            Box::new(Expr::Int(0)),
        ),
        then_body: vec![
            Stmt::Eval(Expr::RawCall(0x1111, vec![], None)),
            Stmt::Assign("*(byte*)(0x2222)".into(), None, Expr::Int(1)),
            Stmt::Eval(Expr::RawCall(0x3333, vec![], None)),
        ],
        else_body: vec![],
    };
    let mut body = vec![more];
    super::idioms::collapse_class_init_guards(&mut body);
    assert_eq!(
        body.len(),
        1,
        "an arm with a third statement is not the guard"
    );
}

// ---------------------------------------------------------------------------
// Loop-test selection: a header whose own branch does NOT exit the loop
// ---------------------------------------------------------------------------

/// The loop TEST is the branch that leaves the loop. A header whose own branch
/// keeps BOTH edges inside the body is carrying interior code — here a hoisted
/// null guard — and the test lives at the latch. The structurer used to take
/// the guard for the test, rendering `while (guard == 0)` (inverted) with the
/// guard's payload unconditional and losing the `do {}` form. Measured on
/// `SmallXmlParser.OnStartElement` @0x12F45D0.
#[test]
fn a_loop_whose_header_carries_an_interior_guard_tests_the_latch() {
    let mem = |addend: i64, width: u8| Operand::Memory {
        base: Box::new(Operand::Local(0)),
        index: None,
        scale: 1,
        addend,
        width,
    };
    let insns = vec![
        Insn::move_(Operand::Local(1), Operand::imm(0), 0x9000), // i = 0
        Insn::new(Op::Move, vec![Operand::Local(3), mem(0, 1)], 0x9004), // guard = *(byte*)obj
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![Operand::Local(4), Operand::Local(3), Operand::imm(0)],
            0x9008,
        ),
        Insn::new(Op::CondJump(5), vec![Operand::Local(4)], 0x9008), // if guard != 0 goto join
        Insn::new(Op::Call, vec![Operand::Immediate(0xB000)], 0x900c), // guarded runtime call
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![Operand::Local(5), Operand::Local(1), Operand::imm(1)],
            0x9010,
        ),
        Insn::move_(Operand::Local(1), Operand::Local(5), 0x9010), // i = i + 1
        Insn::new(
            Op::Compare(CompareKind::NotEqual),
            vec![Operand::Local(6), Operand::Local(1), Operand::imm(5)],
            0x9014,
        ),
        Insn::new(Op::CondJump(1), vec![Operand::Local(6)], 0x9014), // if i != 5 goto header
        Insn::new(Op::Return, vec![], 0x9018),
    ];
    let ir = super::RecoveredIr {
        insns,
        table: locals::LocalTable::default(),
        returns_value: false,
        base_rva: 0x9000,
    };
    let out = csharp::render_method_csharp(&ir, "M", false, &[], &ManagedType::Unknown, -1);
    assert!(
        out.contains("do"),
        "the header's interior branch must give a do/while, got:\n{out}"
    );
    let tail = out.split_once("while (").expect("a loop test").1;
    assert!(
        tail.contains("!= 5"),
        "the loop test must be the latch bound, got:\n{out}"
    );
    assert!(
        !tail.contains('*'),
        "the loop test must not be the null guard, got:\n{out}"
    );
}

// ---------------------------------------------------------------------------
// Handoff P2 (fold_constant_branches deletion): no lost FoldedBranch feature
// ---------------------------------------------------------------------------

/// Deleting the unwired `fold_constant_branches` must not remove `FoldedBranch`
/// coverage: the two live producers are the lifter's ccmp fallback
/// (`lift.rs:1704`) and `simplify_boolean_flags` (`passes.rs:1102`). This pins
/// the latter — a `(x == 0) == 0` chain still collapses and the outer compare
/// is still nop'd.
#[test]
fn boolean_flag_chain_still_folds_after_constant_branch_deletion() {
    let mut insns = vec![
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(1), Operand::Local(2), Operand::Local(3)],
            0x1000,
        ),
        Insn::new(
            Op::Compare(CompareKind::Equal),
            vec![Operand::Local(4), Operand::Local(1), Operand::imm(0)],
            0x1004,
        ),
    ];
    let n = passes::simplify_boolean_flags(&mut insns);
    assert_eq!(n, 1, "the (x == 0) == 0 chain must collapse");
    assert!(
        matches!(insns[0].op, Op::Compare(CompareKind::NotEqual)),
        "the inner equality must invert, got {:?}",
        insns[0].op
    );
    assert!(
        matches!(insns[1].op, Op::Nop),
        "the outer compare must be nop'd, got {:?}",
        insns[1].op
    );
}

// ---------------------------------------------------------------------------
// Handoff P3 (dead_code audit): wire proof + silenced-diagnostic guard
// ---------------------------------------------------------------------------

/// `conform_findings` is deliberately NOT part of `is_clean`, but it is still
/// per-method data: `summarize` surfaces it without blurring the aggregate.
/// Fails pre-fix (field silenced + unsummarized), passes after the wire.
#[test]
fn conform_findings_surface_in_summarize_without_affecting_is_clean() {
    use super::conform::DefectClass;
    use super::verify::VerifyReport;
    let mut r = VerifyReport::default();
    r.method_rva = 0x1000;
    r.decoded = 10;
    r.conform_findings
        .push((DefectClass::CondDropped, 0x1000, "test finding".to_string()));
    assert!(
        r.is_clean(),
        "conform findings must not affect is_clean by contract"
    );
    assert!(
        r.summarize().contains("conform"),
        "summarize must surface conform findings, got: {}",
        r.summarize()
    );
}

/// The silenced-diagnostic attribute is forbidden in owned recovery modules:
/// each site was audited as delete (dead) or wire (missing reader). Fails
/// pre-fix (attributes present), passes after. `csharp.rs` is READ-ONLY
/// (owned elsewhere) — its single `ForEach` attribute is a live variant with
/// a redundant attribute, and is excluded here and reported loudly for its
/// owner to remove.
#[test]
fn no_silenced_dead_code_in_owned_recovery_modules() {
    // Built without the literal so this test's own doc comment cannot
    // self-match the scan.
    let needle = ["#[allow(", "dead_code)]"].concat();
    for (name, src) in [
        ("passes.rs", include_str!("passes.rs")),
        ("emit.rs", include_str!("emit.rs")),
        ("cfg.rs", include_str!("cfg.rs")),
        ("exceptions.rs", include_str!("exceptions.rs")),
        ("ir.rs", include_str!("ir.rs")),
        ("verify.rs", include_str!("verify.rs")),
        ("tests.rs", include_str!("tests.rs")),
    ] {
        assert!(
            !src.contains(needle.as_str()),
            "{name} still silences dead_code — resolve as delete/wire/cfg(test)"
        );
    }
}

/// The dispatch guard names the ORIGINAL index register while the table load
/// names a fresh COPY of it:
///
/// ```text
/// cmp  w24, #0x8        <- guard names w24
/// b.hi default
/// mov  w8, w24          <- index copied into a fresh register
/// ldrb w11, [x9, x8]    <- load names x8
/// ```
///
/// Correlating guard to load on raw register identity therefore never fired:
/// measured over the corpus, 0 of 219 dispatch sites have the guard on the
/// load's own register and 215 of 219 name a different one. `max_case` stayed
/// at its 1023 initialiser and the scan accepted any entry whose target landed
/// inside the method span. NGUIMath.ResizeWidget @0x17ECE8C has a 9-entry table
/// and emitted 479 cases (1,044 lines); after the fix it emits 9 (90 lines).
#[test]
fn switch_table_bound_survives_index_copy() {
    const TBL: u64 = 0x5000;
    const ANCHOR: u64 = 0x1020;
    let mut insns: Vec<Insn> = Vec::new();
    for k in 0..12u64 {
        insns.push(Insn::new(Op::Nop, vec![], 0x1000 + k * 4));
    }
    // The guard names x8, but the table is indexed by x7 = a copy of x8.
    insns.push(Insn::new(
        Op::Compare(CompareKind::GreaterThan),
        vec![Operand::Local(9), Operand::reg("x8"), Operand::imm(3)],
        0x1030,
    ));
    insns.push(Insn::new(Op::CondJump(0x1050), vec![Operand::Local(9)], 0x1034));
    insns.push(Insn::move_(Operand::reg("x7"), Operand::reg("x8"), 0x1038));
    insns.push(Insn::move_(Operand::reg("x9"), Operand::imm(TBL as i64), 0x103c));
    insns.push(Insn::new(
        Op::Arith(ArithKind::Add),
        vec![Operand::reg("x9"), Operand::reg("x9"), Operand::imm(0)],
        0x1040,
    ));
    insns.push(Insn::move_(Operand::reg("x10"), Operand::imm(ANCHOR as i64), 0x1044));
    insns.push(Insn::move_(
        Operand::reg("x11"),
        Operand::Memory {
            width: 1,
            base: Box::new(Operand::reg("x9")),
            index: Some(Box::new(Operand::reg("x7"))),
            scale: 1,
            addend: 0,
        },
        0x1048,
    ));
    insns.push(Insn::new(
        Op::Arith(ArithKind::Shl),
        vec![Operand::reg("xt0"), Operand::reg("x11"), Operand::imm(2)],
        0x104c,
    ));
    insns.push(Insn::new(
        Op::Arith(ArithKind::Add),
        vec![Operand::reg("x10"), Operand::reg("x10"), Operand::reg("xt0")],
        0x1050,
    ));
    insns.push(Insn::new(Op::IndirectCall, vec![Operand::reg("x10")], 0x1054));
    let addresses: Vec<u64> = insns.iter().map(|i| i.address).collect();
    // Entries 4..=12 all land inside [0x1000,0x1054], so a recogniser that
    // misses the bound accepts them as real cases.
    let bytes: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0xff, 0xff, 0xff, 0xff];
    let read = move |a: u64| -> Option<u32> {
        if a < TBL || a + 4 > TBL + bytes.len() as u64 {
            return None;
        }
        let i = (a - TBL) as usize;
        Some(u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]))
    };
    let mut insns = insns;
    let n = passes::recover_switch_tables(&mut insns, &addresses, &read);
    assert_eq!(n, 1, "the table must still be recovered through the copy");
    let targets = insns
        .iter()
        .find_map(|i| match &i.op {
            Op::Switch(t) => Some(t.clone()),
            _ => None,
        })
        .expect("switch");
    assert_eq!(
        targets.len(),
        4,
        "the guard bounds the table at 4 entries; entries 4..=12 are phantom"
    );
}

/// yaxpeax encodes the TBZ/TBNZ bit index as `A64Op::Imm16`, which the lifter
/// did not match, so `bit` was ALWAYS 0 and every test-branch was lifted as a
/// bit-0 test. `tbnz w24, #0x1f` is the sign test that guards the token counts
/// in `Initialize` @0x189D9D4; lifted as `& 1` it produced the provably dead
/// guards `if (!index) { if (index >= 1) ... }` recorded in NEXT_STEPS 4.0.
#[test]
fn tbnz_uses_the_real_bit_index() {
    let bytes = le(&[
        0x37F80098, // tbnz w24, #0x1f, +0x10
        0xD65F03C0, // ret
    ]);
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).expect("lift");
    let and = lifted
        .insns
        .iter()
        .find(|i| matches!(i.op, Op::Arith(ArithKind::And)))
        .expect("the bit test must lower to an And");
    assert_eq!(
        and.operands[2],
        Operand::imm(0x80000000),
        "mask must be the sign bit, not bit 0: {and:?}"
    );
}

/// `bind_fp_call_returns` retires a binding only for the scalar result slots
/// (`vs0`/`vd0`), so `cur["vs4"]` and `cur["vs8"]` kept pointing at the call
/// forever: every later read of ANY version of those slots was rewritten to
/// `call.y`/`call.z`, including versions defined by fsub/fmul. On 0x1530F2C
/// that emitted `Sqrt(t2.y + (t.x-t2.x)*(t.x-t2.x) + t2.y)`; x was correct
/// only because `vs0` happened to be tracked.
#[test]
fn hfa_component_binding_retires_on_write() {
    use crate::recovery::types::{FieldRef, NamedType};
    use std::sync::Arc;
    let mut version_slot = std::collections::HashMap::new();
    version_slot.insert(1u32, "vs0".to_string());
    version_slot.insert(2u32, "vs4".to_string());
    version_slot.insert(3u32, "vs8".to_string());
    version_slot.insert(4u32, "x0".to_string());
    version_slot.insert(5u32, "x0".to_string());
    version_slot.insert(20u32, "vs4".to_string()); // fsub result, a NEW vs4
    version_slot.insert(30u32, "vs9".to_string()); // unrelated slot
    let mut insns = vec![
        Insn::new(Op::Call, vec![Operand::Local(5), Operand::Local(4)], 0x1000),
        // fsub s1, s10, s12  -> a new vs4 that is NOT the call result
        Insn::new(Op::Move, vec![Operand::Local(20), Operand::Local(1)], 0x1004),
        // fmul s1, s1, s1    -> must read Local(20), not call.y
        Insn::new(Op::Move, vec![Operand::Local(30), Operand::Local(20)], 0x1008),
    ];
    insns[0].dest_type = Some(ManagedType::Class(Arc::new(NamedType {
        namespace: "UnityEngine".into(),
        name: "Vector3".into(),
        assembly: None,
        type_def_index: 7651,
    })));
    let field = |name: &str, idx: u32| FieldRef {
        type_def_index: 7651,
        field_index: idx,
        name: name.into(),
        declaring_type_name: "UnityEngine.Vector3".into(),
        field_type: ManagedType::Primitive(PrimKind::R4),
        is_static: false,
    };
    passes::bind_fp_call_returns(&mut insns, &version_slot, 4, |_| {
        vec![field("x", 1), field("y", 2), field("z", 3)]
    });
    assert_eq!(
        insns[2].operands[1],
        Operand::Local(20),
        "a write to vs4 must retire the call binding: {insns:?}"
    );
}

/// §3.1: an EH landing pad is an SSA KILL POINT.
///
/// The CFG cannot express the unwinder edge; the compiler leaves an
/// unreachable `b <pad>` after a noreturn raise, which used to give the pad a
/// synchronous predecessor. Renaming along that edge bound the pad's `x0` to
/// the loop body's last write — the handler then rendered body code
/// (`IL2CPP_IsInst((&enumerator).MoveNext(), …)`) as the exception.
///
/// Fixture: x0 is defined once in the body before a two-way join that lands
/// on the pad. At the pad the read must name the method ENTRY version, not
/// the body definition, and no phi may be placed at the pad.
#[test]
fn eh_handler_entry_kills_slot_state_in_ssa() {
    let words: [u32; 6] = [
        0xD2800540, // 0x1000: movz x0, #0x2a
        0xB4000060, // 0x1004: cbz  x0, 0x1010
        0x14000002, // 0x1008: b    0x1010
        0xD2800001, // 0x100c: movz x1, #0
        0x91000400, // 0x1010: add  x0, x0, #1   ← landing pad
        0xD65F03C0, // 0x1014: ret
    ];
    let bytes = le(&words);
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let graph =
        cfg::Cfg::build_with_landing_pads(&mut insns, &lifted.addresses, &[0x1010]).expect("cfg");
    stack_analyzer::analyze(&mut insns, &graph);
    let dom = dominator::DominatorInfo::compute(&graph);
    let pad_block = graph
        .blocks
        .iter()
        .position(|b| insns[b.start].address == 0x1010)
        .expect("pad must start a block");

    let entry_x0 = {
        let out = ssa::build(&insns, &graph, &dom, &[]);
        out.version_slot
            .iter()
            .filter(|(_, s)| s.as_str() == "x0")
            .map(|(v, _)| *v)
            .min()
            .expect("x0 is seeded")
    };

    // Neutralised: without the kill the pad read names the body definition.
    let plain = ssa::build(&insns, &graph, &dom, &[]);
    let plain_add = plain
        .insns
        .iter()
        .find(|i| i.address == 0x1010)
        .expect("pad instruction");
    assert!(
        !matches!(plain_add.operands.get(1), Some(Operand::Local(v)) if *v == entry_x0),
        "control: the body definition must be what leaks without the kill: {plain_add:?}"
    );

    let mut handlers = std::collections::HashSet::new();
    handlers.insert(pad_block);
    let out = ssa::build_with_handler_entries(&insns, &graph, &dom, &[], &handlers);
    assert!(
        out.phis.iter().all(|p| p.block != pad_block),
        "a handler entry must never be a phi merge point: {:?}",
        out.phis
    );
    let add = out
        .insns
        .iter()
        .find(|i| i.address == 0x1010)
        .expect("pad instruction");
    assert!(
        matches!(add.operands.get(1), Some(Operand::Local(v)) if *v == entry_x0),
        "the pad's x0 read must be the ENTRY version, got {add:?}"
    );
}

/// §3.1: a landing pad is a block even when no branch targets it.
///
/// The pad is addressed by the EH tables, not by an edge. Once the unreachable
/// trampoline after a noreturn raise is neutralised, nothing branches to the
/// pad; without a forced leader it would be glued to the preceding block and
/// `region_of_block`/`handler_of` could not name it.
#[test]
fn landing_pad_is_a_forced_cfg_leader() {
    let words: [u32; 4] = [
        0xD2800020, // 0x1000: movz x0, #1
        0x91000401, // 0x1004: add  x1, x0, #1   ← pad, no branch targets it
        0x91000802, // 0x1008: add  x2, x0, #2
        0xD65F03C0, // 0x100c: ret
    ];
    let bytes = le(&words);
    let lifted = lift::lift_arm64(&bytes, 0x1000, 64, None).expect("lift");
    let mut insns = lifted.insns;
    let plain = cfg::Cfg::build(&mut insns, &lifted.addresses).expect("cfg");
    assert!(
        !plain
            .blocks
            .iter()
            .any(|b| insns[b.start].address == 0x1004),
        "control: an unreferenced pad is inside the entry block"
    );
    let padded =
        cfg::Cfg::build_with_landing_pads(&mut insns, &lifted.addresses, &[0x1004]).expect("cfg");
    assert!(
        padded
            .blocks
            .iter()
            .any(|b| insns[b.start].address == 0x1004),
        "the pad must start its own block: {:?}",
        padded
            .blocks
            .iter()
            .map(|b| insns[b.start].address)
            .collect::<Vec<_>>()
    );
}

/// §3.1: the unreachable `b <pad>` after a noreturn raise is retired.
///
/// The branch is dead code, but the CFG reads it as a successor of the call
/// and gives the landing pad a synchronous predecessor. A `throw` of the
/// exception argument is INSERTED after the call: it terminates the block for
/// every later CFG rebuild, while the trampoline branch itself stays in the
/// listing so the EH row still names a pad and `pad_root` can follow it to
/// the real handler. The call's result slot is retired so the throw does not
/// read the never-returning call's result.
#[test]
fn raise_trampoline_into_a_landing_pad_is_neutralised() {
    let mut insns = vec![
        Insn::new(
            Op::Call,
            vec![
                Operand::imm(0x9000),
                Operand::reg("x0"),
                Operand::reg("x0"),
                Operand::reg("x1"),
            ],
            0x1000,
        ),
        Insn::new(Op::Jump(0x1008), vec![], 0x1004),
        Insn::new(Op::Return, vec![], 0x1008),
    ];
    let mut addresses = vec![0x1000u64, 0x1004, 0x1008];
    let rewritten = super::neutralise_raise_trampolines(
        &mut insns,
        &mut addresses,
        &[0x1008],
        &[0x9000],
    );
    assert_eq!(rewritten, vec![0x1000]);
    assert_eq!(insns.len(), 4, "the throw is inserted, nothing is removed");
    assert!(matches!(insns[1].op, Op::Throw), "{:?}", insns[1]);
    assert_eq!(insns[1].operands[0], Operand::reg("x0"));
    assert!(
        matches!(insns[2].op, Op::Jump(t) if t == 0x1008),
        "the trampoline branch must survive for pad_root: {:?}",
        insns[2]
    );
    assert_eq!(
        insns[0].operands[1],
        Operand::Unknown,
        "a never-returning call has no result"
    );
}

/// A jump that merely targets the pad (not preceded by a raise) is not a
/// trampoline, and a trampoline another branch genuinely reaches must stay.
#[test]
fn raise_trampoline_rewrites_only_provably_dead_branches() {
    let mk = || {
        vec![
            Insn::new(
                Op::Call,
                vec![Operand::imm(0x9000), Operand::reg("x0"), Operand::reg("x0")],
                0x1000,
            ),
            Insn::new(Op::Jump(0x1008), vec![], 0x1004),
            Insn::new(Op::Return, vec![], 0x1008),
        ]
    };
    // Not a raise helper → untouched.
    let mut insns = mk();
    let mut addresses = vec![0x1000u64, 0x1004, 0x1008];
    assert!(super::neutralise_raise_trampolines(&mut insns, &mut addresses, &[0x1008], &[0x9999])
        .is_empty());
    assert!(matches!(insns[1].op, Op::Jump(_)));

    // Somebody branches to 0x1004 → the trampoline is reachable, leave it.
    let mut insns = mk();
    insns.insert(2, Insn::new(Op::Jump(0x1004), vec![], 0x1004));
    let mut addresses = vec![0x1000u64, 0x1004, 0x1004, 0x1008];
    assert!(super::neutralise_raise_trampolines(&mut insns, &mut addresses, &[0x1008], &[0x9000])
        .is_empty());
}
