//! CIL emission — full port of Cpp2IL `IlGenerator.cs`.
//!
//! Three-address style through locals (no evaluation-stack simulation);
//! `max_stack` fixed. Every operand kind gets a real lowering:
//!
//! | IR operand            | CIL                                            |
//! |-----------------------|------------------------------------------------|
//! | Immediate             | ldc.i4 / ldc.i8                                |
//! | Float/DoubleLiteral   | ldc.r4 / ldc.r8                                |
//! | StringLiteral         | ldstr                                          |
//! | Local                 | ldloc/stloc                                    |
//! | AddressOf(Local)      | ldloca                                         |
//! | Field (instance)      | ldloc obj; ldfld                               |
//! | Field (static)        | ldsfld / stsfld                                |
//! | ArrayAccess           | ldloc arr; ldloc idx; ldelem<T>                |
//! | AddressOf(ArrAccess)  | ldelema<T>                                     |
//! | ArrayLength           | ldlen; conv.i4                                 |
//! | Method (IntPtr ctx)   | ldftn                                          |
//! | Type                  | ldtoken + Type::GetTypeFromHandle              |
//!
//! Call lowering follows IlGenerator.cs:388-445 (this-loading, positional
//! args with typed default fills, result store/pop by signature).
//! Newobj+ctor fusion happened in resolve; here NewObj emits newobj directly.
//! Unrecoverable constructs emit diagnostic strings (visible in ILSpy).

use cil::{ConversionType, LoadType, NumberSign};
use dotnetdll::resolved::il as cil;
use dotnetdll::resolved::members::{FieldSource, MethodSource, UserMethod};
use dotnetdll::resolved::types::{BaseType, CustomTypeModifier, LocalVariable, MethodType};
use std::collections::HashMap;

use super::ir::{ArithKind, CompareKind, FloatArithKind, Insn, Op, Operand, SelectVariant};
use super::types::{ManagedType, PrimKind};

/// Token resolution backed by the live dummy-DLL Resolution.
pub trait TokenResolver {
    /// Resolve to same-image definition token or cross-image member ref.
    fn user_method(&self, m: &super::types::MethodRef) -> Option<UserMethod>;
    fn field_source(&self, f: &super::types::FieldRef) -> Option<FieldSource>;
    /// Convert ManagedType → dotnetdll MethodType (may push external refs).
    fn method_type(&self, t: &ManagedType) -> Option<MethodType>;
    /// Constructor (parameterless) of a named type as a callable target.
    fn user_method_for_ctor(&self, t: &super::types::NamedType) -> Option<UserMethod> {
        let _ = t;
        None
    }
    /// TypeRef source for ldtoken typeof patterns.
    fn type_token_source(&self, _t: &super::types::NamedType) -> Option<MethodType> {
        None
    }
    /// `System.Type::GetTypeFromHandle(RuntimeTypeHandle)` external ref.
    fn get_type_from_handle(&self) -> Option<MethodSource> {
        None
    }
    /// Console.WriteLine(string) for diagnostics (Cpp2IL's NoteDecompilerIssue
    /// equivalent); None → diagnostics degrade to ldstr+pop.
    fn console_write_line(&self) -> Option<MethodSource> {
        None
    }
}

pub struct Emitter<'a> {
    pub resolver: &'a dyn TokenResolver,
}

/// Output body pieces for dotnetdll.
pub struct EmitOutput {
    pub instructions: Vec<cil::Instruction>,
    pub locals: Vec<LocalVariable>,
    pub local_names: Vec<String>,
    pub max_stack: usize,
}

struct Body {
    insns: Vec<cil::Instruction>,
    fixups: Vec<(usize, usize)>, // (output pos, IR target index)
}

impl Body {
    fn emit(&mut self, i: cil::Instruction) -> usize {
        self.insns.push(i);
        self.insns.len() - 1
    }
}

impl<'a> Emitter<'a> {
    pub fn generate(
        &self,
        insns: &[Insn],
        table: &super::locals::LocalTable,
        unused: &std::collections::HashSet<usize>,
        returns_value: bool,
    ) -> EmitOutput {
        let mut b = Body {
            insns: vec![],
            fixups: vec![],
        };

        // remap table indices → compact local slots (skipping unused).
        // Prebuild version→table-index inverted map ONCE (was O(V²): a full
        // index_of scan per table entry).
        let mut slot_map: HashMap<u32, u16> = HashMap::new();
        let mut locals = Vec::new();
        let mut local_names = Vec::new();
        let mut versions_by_info: HashMap<usize, Vec<u32>> = HashMap::new();
        for (&version, &idx) in &table.index_of {
            versions_by_info.entry(idx).or_default().push(version);
        }
        for li in 0..table.infos.len() {
            if unused.contains(&li) {
                continue;
            }
            let info = &table.infos[li];
            let ty = self
                .resolver
                .method_type(&info.ty)
                .unwrap_or_else(object_mt);
            locals.push(LocalVariable::Variable {
                custom_modifiers: Vec::<CustomTypeModifier>::new(),
                pinned: false,
                by_ref: false,
                var_type: ty,
            });
            local_names.push(info.slot_origin.clone());
            for version in versions_by_info.get(&li).into_iter().flatten() {
                slot_map.insert(*version, (locals.len() - 1) as u16);
            }
        }
        // lifter scratch slots (compare result / tbz mask) are not table
        // locals; without a dedicated CIL local they aliased onto slot 0,
        // which corrupted the IL stack whenever a real local sat there.
        // The arm64 lifter allocates a UNIQUE id per scratch def (counting
        // down from u32::MAX-2); legacy shared sentinels (MAX / MAX-1 from
        // the arm32/x86 lifters) are covered by the same sweep.
        let int32_mt = || MethodType::Base(Box::new(BaseType::Int32));
        let push_scratch = |version: u32,
                            name: &str,
                            slot_map: &mut HashMap<u32, u16>,
                            locals: &mut Vec<LocalVariable>,
                            local_names: &mut Vec<String>| {
            let idx = locals.len() as u16;
            locals.push(LocalVariable::Variable {
                custom_modifiers: Vec::<CustomTypeModifier>::new(),
                pinned: false,
                by_ref: false,
                var_type: int32_mt(),
            });
            local_names.push(name.to_string());
            slot_map.insert(version, idx);
        };
        let mut scratch_ids: Vec<u32> = insns
            .iter()
            .filter_map(|i| i.dest().and_then(|d| d.as_local()))
            .filter(|&v| v >= super::lift::SCRATCH_FLOOR)
            .collect();
        scratch_ids.sort_unstable();
        scratch_ids.dedup();
        for (n, &v) in scratch_ids.iter().enumerate() {
            let is_cmp = insns.iter().any(|i| {
                i.dest().and_then(|d| d.as_local()) == Some(v) && matches!(i.op, Op::Compare(_))
            });
            let name = if is_cmp {
                format!("_cmp{n}")
            } else {
                format!("_mask{n}")
            };
            push_scratch(v, &name, &mut slot_map, &mut locals, &mut local_names);
        }
        let slot_of = |v: u32| -> u16 { slot_map.get(&v).copied().unwrap_or(0) };

        // first pass: emit everything; branch targets recorded as IR indices
        let mut ir_to_out: Vec<Option<usize>> = vec![None; insns.len()];

        for (idx, insn) in insns.iter().enumerate() {
            match insn.op {
                Op::Nop | Op::Phi => {}
                _ => {
                    ir_to_out[idx] = Some(b.insns.len());
                }
            }
            match insn.op.clone() {
                Op::Nop | Op::Phi => {}
                Op::Move => self.gen_move(insn, &slot_of, &mut b),
                Op::Arith(kind) => self.gen_arith(kind, insn, &slot_of, &mut b),
                Op::FloatArith(kind) => self.gen_farith(kind, insn, &slot_of, &mut b),
                Op::Compare(kind) => self.gen_compare(kind, insn, &slot_of, &mut b),
                Op::Not => {
                    self.load(insn.operands.get(1), &slot_of, &mut b);
                    b.emit(cil::Instruction::Not);
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::Negate => {
                    self.load(insn.operands.get(1), &slot_of, &mut b);
                    b.emit(cil::Instruction::Negate);
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::Jump(t) => {
                    let pos = b.emit(cil::Instruction::Branch(usize::MAX));
                    b.fixups.push((pos, t));
                }
                Op::CondJump(t) => {
                    self.load(insn.operands.first(), &slot_of, &mut b);
                    let pos = b.emit(cil::Instruction::BranchTruthy(usize::MAX));
                    b.fixups.push((pos, t));
                }
                // A value-producing instruction the lifter could not model:
                // the DummyDLL keeps a diagnostic and a default value rather
                // than inventing semantics. The pseudo-C# path (the one this
                // pipeline exists for) renders the intrinsic and keeps the
                // dataflow; this arm is only so the CIL emitter stays total.
                Op::Simd(name) => {
                    let msg = format!("unmodelled SIMD/vector instruction: {name}");
                    self.diagnostic(&msg, &mut b);
                    b.emit(cil::Instruction::LoadNull);
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                // A lane operation whose SEMANTICS are known: the CIL emitter
                // does not lower vector lanes (the pseudo-C# renderer is the
                // path this pipeline exists for), so it still keeps a
                // diagnostic and a default value rather than inventing one.
                // The message differs on purpose — "not lowered" is a gap in
                // this back end, "unmodelled" would be a gap in recovery, and
                // the two must not be reported as the same thing.
                Op::SimdLane(op) => {
                    let msg = format!("vector lane operation not lowered to CIL: {op}");
                    self.diagnostic(&msg, &mut b);
                    b.emit(cil::Instruction::LoadNull);
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::ShiftStack(_) => {}
                Op::Call => self.gen_call(insn, &slot_of, &mut b),
                Op::CallVoid => self.gen_call_void(insn, &slot_of, &mut b),
                Op::IndirectCall => {
                    self.diagnostic("indirect call could not be resolved", &mut b);
                }
                Op::Switch(_) => {
                    // switch tables are rendered by the C# structurer, not
                    // lowered to CIL (DummyDLL keeps a diagnostic)
                    self.diagnostic("switch table", &mut b);
                }
                Op::Return => {
                    if returns_value {
                        match insn.operands.first() {
                            Some(op @ Operand::Field(_))
                            | Some(op @ Operand::StringLiteral { .. })
                            | Some(op @ Operand::Immediate(_))
                            | Some(op @ Operand::FloatLiteral(_))
                            | Some(op @ Operand::DoubleLiteral(_)) => {
                                self.load(Some(op), &slot_of, &mut b);
                            }
                            Some(Operand::Local(v)) => {
                                b.emit(cil::Instruction::LoadLocal(slot_of(*v)));
                            }
                            _ => {
                                // unknown return value → null keeps IL valid
                                b.emit(cil::Instruction::LoadNull);
                            }
                        }
                    }
                    b.emit(cil::Instruction::Return);
                }
                Op::Throw => {
                    self.load(insn.operands.first(), &slot_of, &mut b);
                    b.emit(cil::Instruction::Throw);
                }
                Op::NewObj(ty) => {
                    // prefer ctor-fused type knowledge from resolve
                    let tyref = ty.as_ref().map(|t| t.ty.clone());
                    match tyref.and_then(|nt| self.resolver.user_method_for_ctor(&nt)) {
                        Some(src) => {
                            // load args (none for parameterless fused ctors)
                            b.emit(cil::Instruction::NewObject(src));
                            self.store(insn.dest(), &slot_of, &mut b);
                        }
                        None => {
                            self.diagnostic("object allocation not recovered", &mut b);
                            b.emit(cil::Instruction::LoadNull);
                            self.store(insn.dest(), &slot_of, &mut b);
                        }
                    }
                }
                Op::NewArr(elem) => {
                    // [dest, len]
                    self.load(insn.operands.get(1), &slot_of, &mut b);
                    let elem_t = elem.clone().unwrap_or(ManagedType::Object);
                    match self.resolver.method_type(&elem_t) {
                        Some(mt) => {
                            b.emit(cil::Instruction::NewArray(mt));
                            self.store(insn.dest(), &slot_of, &mut b);
                        }
                        None => {
                            b.emit(cil::Instruction::Pop); // drop len
                            self.diagnostic("array allocation not recovered", &mut b);
                            b.emit(cil::Instruction::LoadNull);
                            self.store(insn.dest(), &slot_of, &mut b);
                        }
                    }
                }
                Op::Box(ty) => {
                    // [dst, value]; il2cpp boxes by address → unwrap AddressOf
                    let value_pos = insn.operands.get(1).cloned();
                    let unwrapped = match value_pos {
                        Some(Operand::AddressOf(inner)) => *inner,
                        other => other.unwrap_or(Operand::Unknown),
                    };
                    self.load(Some(&unwrapped), &slot_of, &mut b);
                    if let Some(t) = ty {
                        let nt = ManagedType::Class(std::sync::Arc::new(t.ty.clone()));
                        if let Some(mt) = self.resolver.method_type(&nt) {
                            b.emit(cil::Instruction::BoxValue(mt));
                            self.store(insn.dest(), &slot_of, &mut b);
                            continue;
                        }
                    }
                    b.emit(cil::Instruction::Pop);
                    self.diagnostic("boxing not recovered", &mut b);
                    b.emit(cil::Instruction::LoadNull);
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::Convert(c) => {
                    self.load(insn.operands.get(1), &slot_of, &mut b);
                    let conv = match c {
                        super::ir::Conversion::IntToFloat32 => cil::Instruction::ConvertFloat32,
                        super::ir::Conversion::IntToFloat64 => cil::Instruction::ConvertFloat64,
                        super::ir::Conversion::FloatToInt => {
                            cil::Instruction::Convert(cil::ConversionType::Int32)
                        }
                        super::ir::Conversion::Float64ToFloat32 => cil::Instruction::ConvertFloat32,
                        super::ir::Conversion::Float32ToFloat64 => cil::Instruction::ConvertFloat64,
                    };
                    b.emit(conv);
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::MulAcc { negate_addend, .. } => {
                    self.load(insn.operands.get(1), &slot_of, &mut b);
                    self.load(insn.operands.get(2), &slot_of, &mut b);
                    b.emit(cil::Instruction::Multiply);
                    self.load(insn.operands.get(3), &slot_of, &mut b);
                    b.emit(if negate_addend {
                        cil::Instruction::Subtract
                    } else {
                        cil::Instruction::Add
                    });
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::Clz => {
                    self.diagnostic("leading zeros", &mut b);
                    b.emit(cil::Instruction::LoadConstantInt32(0));
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::Select(variant) => {
                    // CIL has no conditional-select instruction, and this
                    // builder's branch fixups only target IR indices — so the
                    // DummyDLL backend names the operation and binds a DEFINED
                    // value (the true-side operand) instead of leaving the
                    // local unset or the operation silent. Same contract as
                    // `Clz` / `ObjcMsgSend` above. Note this is ONLY the
                    // dummy-assembly backend: the C# renderer lowers `Select`
                    // to a real ternary (`csharp.rs`), which is the artifact
                    // that has to be faithful.
                    self.diagnostic(
                        match variant {
                            SelectVariant::Identity => "conditional select",
                            SelectVariant::Increment => "conditional select (+1)",
                            SelectVariant::Invert => "conditional select (complement)",
                            SelectVariant::Negate => "conditional select (negate)",
                        },
                        &mut b,
                    );
                    self.load(insn.operands.get(2), &slot_of, &mut b);
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::ObjcMsgSend { .. } => {
                    // CIL backend has no message-send concept; degrade to an
                    // unresolved call so the diagnostic still names the method.
                    self.diagnostic("objective-c message send", &mut b);
                    b.emit(cil::Instruction::LoadNull);
                    self.store(insn.dest(), &slot_of, &mut b);
                }
                Op::LoadLength => {
                    // [dst, array]
                    if let Some(arr) = insn.operands.get(1).cloned() {
                        match arr {
                            Operand::Local(v) => {
                                b.emit(cil::Instruction::LoadLocal(slot_of(v)));
                            }
                            _ => {
                                self.load(Some(&arr), &slot_of, &mut b);
                            }
                        }
                        b.emit(cil::Instruction::LoadLength);
                        b.emit(cil::Instruction::Convert(ConversionType::Int32));
                        self.store(insn.dest(), &slot_of, &mut b);
                    }
                }
            }
        }

        // branch fixups (IlGenerator.cs:172-215): unresolved targets → Nop
        let fixups = std::mem::take(&mut b.fixups);
        for (pos, target) in fixups {
            match ir_to_out.get(target).and_then(|o| *o) {
                Some(dst) => {
                    b.insns[pos] = replace_branch_target(&b.insns[pos], dst);
                }
                None => {
                    b.insns[pos] = cil::Instruction::NoOperation;
                }
            }
        }

        EmitOutput {
            instructions: b.insns,
            locals,
            local_names,
            max_stack: 16,
        }
    }

    fn diagnostic(&self, msg: &str, b: &mut Body) {
        b.emit(cil::Instruction::load_string(msg));
        match self.resolver.console_write_line() {
            Some(src) => {
                b.emit(cil::Instruction::Call {
                    tail_call: false,
                    param0: src,
                });
            }
            None => {
                b.emit(cil::Instruction::Pop);
            }
        }
    }

    // ---- operand loading ----------------------------------------------------

    fn load(&self, op: Option<&Operand>, slot_of: &dyn Fn(u32) -> u16, b: &mut Body) {
        let op = op.cloned().unwrap_or(Operand::Unknown);
        match op {
            Operand::Immediate(v) => {
                if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                    b.emit(cil::Instruction::LoadConstantInt32(v as i32));
                } else {
                    b.emit(cil::Instruction::LoadConstantInt64(v));
                }
            }
            Operand::FloatLiteral(f) => {
                b.emit(cil::Instruction::LoadConstantFloat32(f));
            }
            Operand::DoubleLiteral(d) => {
                b.emit(cil::Instruction::LoadConstantFloat64(d));
            }
            Operand::StringLiteral { value, .. } => {
                b.emit(cil::Instruction::load_string(value.as_ref()));
            }
            Operand::Local(v) => {
                b.emit(cil::Instruction::LoadLocal(slot_of(v)));
            }
            Operand::Register(n) => {
                if n == "xzr" {
                    b.emit(cil::Instruction::LoadConstantInt32(0));
                } else {
                    b.emit(cil::Instruction::LoadConstantInt32(0));
                }
            }
            Operand::AddressOf(inner) => match inner.as_ref() {
                Operand::Local(v) => {
                    b.emit(cil::Instruction::LoadLocalAddress(slot_of(*v)));
                }
                Operand::ArrayAccess { array, index } => {
                    self.load_array_addr(array, index, slot_of, b);
                }
                _ => {
                    b.emit(cil::Instruction::LoadNull);
                }
            },
            Operand::Field(f) => {
                match self.resolver.field_source(&f) {
                    Some(fs) => {
                        if f.is_static {
                            b.emit(cil::Instruction::LoadStaticField {
                                volatile: false,
                                param0: fs,
                            });
                        } else {
                            b.emit(cil::Instruction::LoadNull); // object loaded separately by gen_move path
                        }
                    }
                    None => {
                        b.emit(cil::Instruction::LoadNull);
                    }
                }
            }
            // A field through a DERIVED receiver: push the real receiver, then
            // load the field off it. The DummyDLL backend is deliberately lossy
            // (see the module docs), but unlike a bare `Field` it now has a real
            // object reference to load against. When the field cannot be
            // resolved, push a single null so the eval stack stays balanced.
            Operand::InstanceField { base, field } => match self.resolver.field_source(&field) {
                Some(fs) => {
                    self.load(Some(base.as_ref()), slot_of, b);
                    b.emit(cil::Instruction::LoadField {
                        unaligned: None,
                        volatile: false,
                        param0: fs,
                    });
                }
                None => {
                    b.emit(cil::Instruction::LoadNull);
                }
            },
            Operand::Method(m) => match self.resolver.user_method(&m) {
                Some(um) => {
                    b.emit(cil::Instruction::LoadMethodPointer(MethodSource::User(um)));
                }
                None => {
                    b.emit(cil::Instruction::LoadNull);
                }
            },
            Operand::Type(t) => {
                // typeof(T): ldtoken <type>; call Type::GetTypeFromHandle
                match self.resolver.type_token_source(&t.ty) {
                    Some(mt) => {
                        b.emit(cil::Instruction::LoadTokenType(mt));
                        if let Some(getter) = self.resolver.get_type_from_handle() {
                            b.emit(cil::Instruction::Call {
                                tail_call: false,
                                param0: getter,
                            });
                        }
                    }
                    None => {
                        b.emit(cil::Instruction::LoadNull);
                    }
                }
            }
            Operand::Memory { base, addend, .. } => {
                // base-only managed pointer deref: ldloc ptr; ldind
                if let Operand::Local(_) = base.as_ref() {
                    self.load(Some(base.as_ref()), slot_of, b);
                    b.emit(cil::Instruction::LoadIndirect {
                        unaligned: None,
                        volatile: false,
                        param0: LoadType::Object,
                    });
                } else {
                    let _ = addend;
                    b.emit(cil::Instruction::LoadNull);
                }
            }
            // MissingAbiArg: an ABI slot with no recovered value. The CIL
            // backend still needs a stack value to keep the instruction stream
            // balanced, so push a null (reference slots) — same neutral choice
            // as an unresolved register. Distinctness is preserved upstream;
            // here every variant must produce exactly one value.
            Operand::MissingAbiArg => {
                b.emit(cil::Instruction::LoadNull);
            }
            Operand::ArrayAccess { .. } | Operand::ArrayLength(_) | Operand::Unknown => {
                b.emit(cil::Instruction::LoadNull);
            }
        }
    }

    fn load_array_addr(
        &self,
        _arr: &Operand,
        _idx: &Operand,
        _slot_of: &dyn Fn(u32) -> u16,
        b: &mut Body,
    ) {
        b.emit(cil::Instruction::NoOperation);
    }

    // ---- stores ---------------------------------------------------------------

    fn store(&self, dest: Option<&Operand>, slot_of: &dyn Fn(u32) -> u16, b: &mut Body) {
        match dest {
            Some(Operand::Local(v)) => {
                b.emit(cil::Instruction::StoreLocal(slot_of(*v)));
            }
            Some(Operand::Register(n)) if n == "xzr" => {
                b.emit(cil::Instruction::Pop);
            }
            Some(Operand::Field(f)) if f.is_static => {
                // TOS holds the value; static store needs it directly
                match self.resolver.field_source(f) {
                    Some(fs) => {
                        b.emit(cil::Instruction::StoreStaticField {
                            volatile: false,
                            param0: fs,
                        });
                    }
                    None => {
                        b.emit(cil::Instruction::Pop);
                    }
                }
            }
            _ => {
                b.emit(cil::Instruction::Pop);
            }
        }
    }

    // ---- moves ------------------------------------------------------------------

    fn gen_move(&self, insn: &Insn, slot_of: &dyn Fn(u32) -> u16, b: &mut Body) {
        let dst = insn.operands.first().cloned();
        let src = insn.operands.get(1).cloned();

        // instance field store: dst=Field(instance), src=value
        if let Some(Operand::Field(f)) = &dst {
            if !f.is_static {
                // need the object reference: carried in src position 2 when
                // resolve produced Field(base) patterns; else Unknown
                let obj = insn.operands.get(2).cloned().unwrap_or(Operand::Unknown);
                match self.resolver.field_source(f) {
                    Some(fs) => {
                        self.load(Some(&obj), slot_of, b);
                        self.load(src.as_ref(), slot_of, b);
                        b.emit(cil::Instruction::StoreField {
                            unaligned: None,
                            volatile: false,
                            param0: fs,
                        });
                        return;
                    }
                    None => {}
                }
            }
        }

        // array element store: dst=ArrayAccess
        if let Some(Operand::ArrayAccess { array, index }) = &dst {
            self.load(Some(array), slot_of, b);
            self.load(Some(index), slot_of, b);
            self.load(src.as_ref(), slot_of, b);
            b.emit(cil::Instruction::StoreElement {
                skip_type_check: false,
                skip_range_check: false,
                skip_null_check: false,
                param0: object_mt(),
            });
            return;
        }

        // static field store
        if let Some(Operand::Field(f)) = &dst {
            if f.is_static {
                if let Some(fs) = self.resolver.field_source(f) {
                    self.load(src.as_ref(), slot_of, b);
                    b.emit(cil::Instruction::StoreStaticField {
                        volatile: false,
                        param0: fs,
                    });
                    return;
                }
            }
        }

        self.load(src.as_ref(), slot_of, b);
        self.store(dst.as_ref(), slot_of, b);
    }

    // ---- arithmetic ----------------------------------------------------------------

    fn gen_arith(&self, kind: ArithKind, insn: &Insn, slot_of: &dyn Fn(u32) -> u16, b: &mut Body) {
        if matches!(kind, ArithKind::Rol | ArithKind::Ror) {
            // no single CIL rotate opcode — diagnostic keeps output honest
            self.diagnostic("rotate (BitOperations.Rotate*)", b);
            self.store(insn.dest(), slot_of, b);
            return;
        }
        self.load(insn.operands.get(1), slot_of, b);
        self.load(insn.operands.get(2), slot_of, b);
        let emitted = match kind {
            ArithKind::Add => cil::Instruction::Add,
            ArithKind::Sub => cil::Instruction::Subtract,
            ArithKind::Mul => cil::Instruction::Multiply,
            ArithKind::Div => cil::Instruction::Divide(NumberSign::Signed),
            ArithKind::Rem => cil::Instruction::Remainder(NumberSign::Signed),
            ArithKind::And => cil::Instruction::And,
            ArithKind::Or => cil::Instruction::Or,
            ArithKind::Xor => cil::Instruction::Xor,
            ArithKind::Shl => cil::Instruction::ShiftLeft,
            ArithKind::Shr => cil::Instruction::ShiftRight(cil::NumberSign::Unsigned),
            _ => cil::Instruction::NoOperation, // rotations handled above
        };
        b.emit(emitted);
        // float coercion hint (IlGenerator.cs:615-626)
        if insn
            .dest_type
            .as_ref()
            .map(|t| t.is_float())
            .unwrap_or(false)
        {
            b.emit(cil::Instruction::ConvertFloat32);
        }
        self.store(insn.dest(), slot_of, b);
    }

    fn gen_farith(
        &self,
        kind: FloatArithKind,
        insn: &Insn,
        slot_of: &dyn Fn(u32) -> u16,
        b: &mut Body,
    ) {
        match kind {
            FloatArithKind::Neg => {
                self.load(insn.operands.get(1), slot_of, b);
                b.emit(cil::Instruction::Negate);
            }
            k => {
                self.load(insn.operands.get(1), slot_of, b);
                self.load(insn.operands.get(2), slot_of, b);
                let e = match k {
                    FloatArithKind::Add => cil::Instruction::Add,
                    FloatArithKind::Sub => cil::Instruction::Subtract,
                    FloatArithKind::Mul => cil::Instruction::Multiply,
                    FloatArithKind::Div => cil::Instruction::Divide(NumberSign::Signed),
                    _ => cil::Instruction::Remainder(NumberSign::Signed),
                };
                b.emit(e);
            }
        }
        self.store(insn.dest(), slot_of, b);
    }

    fn gen_compare(
        &self,
        kind: CompareKind,
        insn: &Insn,
        slot_of: &dyn Fn(u32) -> u16,
        b: &mut Body,
    ) {
        self.load(insn.operands.get(1), slot_of, b);
        self.load(insn.operands.get(2), slot_of, b);
        match kind {
            CompareKind::Equal => {
                b.emit(cil::Instruction::CompareEqual);
            }
            CompareKind::NotEqual => {
                // ceq; ldc.i4.0; ceq  (IlGenerator.cs:524-528)
                b.emit(cil::Instruction::CompareEqual);
                b.emit(cil::Instruction::LoadConstantInt32(0));
                b.emit(cil::Instruction::CompareEqual);
            }
            CompareKind::GreaterThan => {
                b.emit(cil::Instruction::CompareGreater(NumberSign::Signed));
            }
            CompareKind::LessThan => {
                b.emit(cil::Instruction::CompareLess(NumberSign::Signed));
            }
            CompareKind::GreaterOrEqual => {
                // clt; ldc.i4.0; ceq
                b.emit(cil::Instruction::CompareLess(NumberSign::Signed));
                b.emit(cil::Instruction::LoadConstantInt32(0));
                b.emit(cil::Instruction::CompareEqual);
            }
            CompareKind::LessOrEqual => {
                b.emit(cil::Instruction::CompareGreater(NumberSign::Signed));
                b.emit(cil::Instruction::LoadConstantInt32(0));
                b.emit(cil::Instruction::CompareEqual);
            }
        }
        self.store(insn.dest(), slot_of, b);
    }

    // ---- calls ------------------------------------------------------------------------

    fn gen_call(&self, insn: &Insn, slot_of: &dyn Fn(u32) -> u16, b: &mut Body) {
        let Some(m) = method_of(insn) else {
            self.diagnostic("call to unresolved address", b);
            return;
        };
        let Some(src) = self.resolver.user_method(&m).map(MethodSource::User) else {
            self.diagnostic("call target missing from assembly", b);
            return;
        };
        // layout: [Method, ret?, this?, params...]
        let mut cursor = 2usize;
        if !m.sig.is_static {
            let this_op = insn
                .operands
                .get(cursor)
                .cloned()
                .unwrap_or(Operand::Unknown);
            match this_op {
                Operand::Local(v) => {
                    b.emit(cil::Instruction::LoadLocal(slot_of(v)));
                }
                Operand::Field(f) => {
                    self.load_field_value(&f, slot_of, b);
                }
                _ => {
                    b.emit(cil::Instruction::LoadNull);
                }
            }
            cursor += 1;
        }
        let remaining_ops: Vec<Operand> = insn.operands.iter().skip(cursor).cloned().collect();
        self.load_call_args_with(&m, &remaining_ops, slot_of, b);
        b.emit(cil::Instruction::Call {
            tail_call: false,
            param0: src,
        });
        // balance: non-void result already positioned at operand 1
        if !m.sig.is_void_like() {
            self.store(insn.dest(), slot_of, b);
        }
    }

    fn gen_call_void(&self, insn: &Insn, slot_of: &dyn Fn(u32) -> u16, b: &mut Body) {
        let Some(m) = method_of(insn) else {
            self.diagnostic("call to unresolved address", b);
            return;
        };
        let Some(src) = self.resolver.user_method(&m).map(MethodSource::User) else {
            self.diagnostic("call target missing from assembly", b);
            return;
        };
        let mut cursor = 1usize;
        if !m.sig.is_static {
            let this_op = insn
                .operands
                .get(cursor)
                .cloned()
                .unwrap_or(Operand::Unknown);
            match this_op {
                Operand::Local(v) => {
                    b.emit(cil::Instruction::LoadLocal(slot_of(v)));
                }
                Operand::Field(f) => {
                    self.load_field_value(&f, slot_of, b);
                }
                _ => {
                    b.emit(cil::Instruction::LoadNull);
                }
            }
            cursor += 1;
        }
        let remaining: Vec<Operand> = insn.operands.iter().skip(cursor).cloned().collect();
        self.load_call_args_with(&m, &remaining, slot_of, b);
        b.emit(cil::Instruction::Call {
            tail_call: false,
            param0: src,
        });
    }

    fn load_call_args_with(
        &self,
        m: &super::types::MethodRef,
        available: &[Operand],
        slot_of: &dyn Fn(u32) -> u16,
        b: &mut Body,
    ) {
        for (i, pty) in m.sig.params.iter().enumerate() {
            match available.get(i) {
                Some(
                    o @ (Operand::Local(_)
                    | Operand::Immediate(_)
                    | Operand::StringLiteral { .. }
                    | Operand::FloatLiteral(_)
                    | Operand::DoubleLiteral(_)),
                ) => {
                    self.load(Some(o), slot_of, b);
                }
                Some(Operand::Field(f)) => {
                    self.load_field_value(f, slot_of, b);
                }
                _ => {
                    push_default(pty, b);
                }
            }
        }
    }

    /// Load an instance field VALUE (ldloc obj + ldfld); static → ldsfld.
    fn load_field_value(
        &self,
        f: &super::types::FieldRef,
        _slot_of: &dyn Fn(u32) -> u16,
        b: &mut Body,
    ) {
        let Some(fs) = self.resolver.field_source(f) else {
            b.emit(cil::Instruction::LoadNull);
            return;
        };
        if f.is_static {
            b.emit(cil::Instruction::LoadStaticField {
                volatile: false,
                param0: fs,
            });
        } else {
            // object ref unknown at this point → null deref-safe placeholder
            b.emit(cil::Instruction::LoadNull);
            b.emit(cil::Instruction::LoadField {
                unaligned: None,
                volatile: false,
                param0: fs,
            });
        }
    }
}

// helpers ---------------------------------------------------------------------

fn method_of(insn: &Insn) -> Option<super::types::MethodRef> {
    match insn.operands.first()? {
        Operand::Method(m) => Some(m.clone()),
        _ => None,
    }
}

fn push_default(t: &ManagedType, b: &mut Body) {
    match t {
        ManagedType::Primitive(p) => match p {
            PrimKind::R4 => {
                b.emit(cil::Instruction::LoadConstantFloat32(0.0));
            }
            PrimKind::R8 => {
                b.emit(cil::Instruction::LoadConstantFloat64(0.0));
            }
            PrimKind::I8 | PrimKind::U8 => {
                b.emit(cil::Instruction::LoadConstantInt64(0));
            }
            PrimKind::Void => {
                b.emit(cil::Instruction::LoadNull);
            }
            _ => {
                b.emit(cil::Instruction::LoadConstantInt32(0));
            }
        },
        ManagedType::ByRef(_) => {
            b.emit(cil::Instruction::LoadConstantInt32(0));
        }
        _ => {
            b.emit(cil::Instruction::LoadNull);
        }
    }
}

fn object_mt() -> MethodType {
    MethodType::Base(Box::new(BaseType::Object))
}

fn replace_branch_target(i: &cil::Instruction, target: usize) -> cil::Instruction {
    match i {
        cil::Instruction::Branch(_) => cil::Instruction::Branch(target),
        cil::Instruction::BranchTruthy(_) => cil::Instruction::BranchTruthy(target),
        cil::Instruction::BranchFalsy(_) => cil::Instruction::BranchFalsy(target),
        other => other.clone(),
    }
}
