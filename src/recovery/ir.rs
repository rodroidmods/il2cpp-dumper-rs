//! Instruction-set-independent IR for native→CIL recovery.
//!
//! Faithful port of Cpp2IL's ISIL (`Cpp2IL.Core/ISIL/*.cs`) semantics:
//! three-address form, identity-by-index instructions (branch targets are
//! indices into the flat listing), destination positional by opcode
//! (Call: [target, returnValue, this?, params...], CallVoid: [target, this?, ...]).
//!
//! Transient ops (Phi) exist only between SSA construction and elimination.

pub use super::types::{FieldRef, ManagedType, MethodRef};
use super::resolve::ResolveMaps;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// The IR name of the register a call's result lands in when that register is
/// NOT argument register 0.
///
/// On the ARM ABIs (AAPCS64 / AAPCS32) the result register and argument
/// register 0 are the SAME physical register (`x0` / `r0`), so one name is
/// right for both roles and the whole pipeline can say "x0". On x86-64 they
/// are two different physical registers — `rax` for the result and `rdi`
/// (SysV) / `rcx` (MSVC) for argument 0 — so naming both `x0` makes one name
/// carry two independent values. A call result then retires the receiver
/// identity, and an `x0` argument slot can resolve to a version that really
/// holds a discarded return value.
///
/// The split is what makes `x0` mean "argument 0 / `this`" on EVERY
/// architecture, which is what `locals::create` (receiver seeding),
/// `locals::rewrite_pinned_copies` (receiver pinning) and
/// `csharp::entry_register_name` already assume.
pub const RET_REG: &str = "rv";

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// Hardware register or virtual stack slot ("x0", "rv", "stack_4").
    ///
    /// See [`RET_REG`] for why the x86 result register is named apart from
    /// argument 0.
    Register(String),
    /// SSA-versioned / final local variable slot.
    Local(u32),
    Immediate(i64),
    FloatLiteral(f32),
    DoubleLiteral(f64),
    StringLiteral {
        index: u32,
        value: ArcString,
    },
    Method(MethodRef),
    Field(FieldRef),
    /// An INSTANCE FIELD access on a receiver that is NOT `this`.
    ///
    /// `Field` carries no receiver, so the typing pass could only name a field
    /// when the base was provably `this`; every other field access stayed a raw
    /// `*(nint*)(base + off)` even when both the base's class and the offset were
    /// known — the census's `vbase`/`dchain` families, ~64% of the arm64 dump.
    /// Carrying the base here lets the renderer emit `base.Field` and keeps the
    /// receiver in the output. The two must ship together: naming the field while
    /// dropping the base is exactly the `__unknown.field` corruption the old
    /// `base_is_this` gate existed to prevent.
    InstanceField {
        base: Box<Operand>,
        field: FieldRef,
    },
    Type(NamedTypeRef),
    /// `[base + index*scale + addend]`
    Memory {
        base: Box<Operand>,
        index: Option<Box<Operand>>,
        scale: u8,
        addend: i64,
        /// ACCESS WIDTH in bytes: 1/2/4/8, or 0 for "pointer-width / not
        /// stated" (the default every non-narrow access uses).
        ///
        /// This is not cosmetic. `ldrb`/`strb`/`ldrh`/`strh` were lifted
        /// identically to `ldr`/`str` — same `Move`, same `Memory` — so a
        /// one-byte `bool`/`byte`/`char` field read rendered as
        /// `*(nint*)(x+off)`: an EIGHT-byte load of an object whose declared
        /// size is one. Same slot, same address, wrong access — and a
        /// byte STORE written as an eight-byte store clobbers the seven
        /// following fields. Carrying the width lets the renderer emit the
        /// access the native code actually performs.
        width: u8,
    },
    /// Managed address-of (`&local`, `&arr[i]`).
    AddressOf(Box<Operand>),
    ArrayAccess {
        array: Box<Operand>,
        index: Box<Operand>,
    },
    ArrayLength(Box<Operand>),
    /// An ABI argument slot whose value could not be recovered. Semantically
    /// DISTINCT from `Unknown` (a genuinely unresolved value) and from a real
    /// `default(T)` — this slot exists only because ARM64 passes fewer live
    /// registers than the signature declares. The renderer drops these
    /// positionally; folding/analysis passes must NOT substitute values for
    /// them.
    MissingAbiArg,
    Unknown,
}

/// Cheap shared string for literal values (methods get cloned often).
pub type ArcString = std::sync::Arc<str>;

/// Reference to a managed type for newobj/newarr/box/typeof/castclass.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedTypeRef {
    pub ty: super::types::NamedType,
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register(n) => write!(f, "%{n}"),
            Self::Local(i) => write!(f, "v{i}"),
            Self::Immediate(v) => write!(f, "#{v:#x}"),
            Self::FloatLiteral(v) => write!(f, "{v}f"),
            Self::DoubleLiteral(v) => write!(f, "{v}d"),
            Self::StringLiteral { value, .. } => write!(f, "{value:?}"),
            Self::Method(m) => write!(f, "{}", m.full_name()),
            Self::Field(fd) => write!(f, "{}::{}", fd.declaring_type_name, fd.name),
            Self::InstanceField { base, field } => write!(f, "{base}.{}", field.name),
            Self::Type(t) => write!(f, "typeof({})", t.ty),
            Self::Memory {
                base,
                index,
                scale,
                addend,
                width,
            } => {
                // Only narrow accesses carry a width; `0` is the pointer-width
                // default and stays invisible so existing IR dumps are stable.
                if *width != 0 {
                    write!(f, "w{width}")?;
                }
                write!(f, "[{base}")?;
                if let Some(idx) = index {
                    write!(f, "+{idx}*{scale}")?;
                }
                write!(f, "{addend:+#x}]")
            }
            Self::AddressOf(t) => write!(f, "&{t}"),
            Self::ArrayAccess { array, index } => write!(f, "{array}[{index}]"),
            Self::ArrayLength(a) => write!(f, "len({a})"),
            Self::MissingAbiArg => write!(f, "<abi?>"),
            Self::Unknown => write!(f, "?"),
        }
    }
}

impl Operand {
    pub fn reg(name: impl Into<String>) -> Self {
        Self::Register(name.into())
    }

    pub fn imm(v: i64) -> Self {
        Self::Immediate(v)
    }

    /// Attach an ACCESS WIDTH (in bytes) to a `Memory` operand.
    ///
    /// `0` means "not a narrow access" and returns the operand untouched, so
    /// every non-narrow site keeps the pointer-width default and nothing
    /// changes for it. A narrow access (`ldrb`/`strb` = 1, `ldrh`/`strh` = 2)
    /// has to keep its width all the way to the renderer: a one-byte field
    /// READ written as `*(nint*)` reads eight bytes, and the STORE form writes
    /// eight bytes over a one-byte slot.
    pub fn with_width(self, width: u8) -> Self {
        if width == 0 {
            return self;
        }
        match self {
            Self::Memory {
                base,
                index,
                scale,
                addend,
                ..
            } => Self::Memory {
                base,
                index,
                scale,
                addend,
                width,
            },
            other => other,
        }
    }

    pub fn as_local(&self) -> Option<u32> {
        match self {
            Self::Local(i) => Some(*i),
            _ => None,
        }
    }

    /// Registers/slots this operand READS from (for def/use analysis).
    pub fn uses(&self, out: &mut Vec<u32>) {
        match self {
            Self::Local(i) => out.push(*i),
            Self::Memory { base, index, .. } => {
                base.uses(out);
                if let Some(i) = index {
                    i.uses(out);
                }
            }
            Self::InstanceField { base, .. } => base.uses(out),
            Self::AddressOf(inner) | Self::ArrayLength(inner) => inner.uses(out),
            Self::ArrayAccess { array, index } => {
                array.uses(out);
                index.uses(out);
            }
            // Leaf operands carry no sub-operand, and a register NAME is a
            // pre-SSA spelling that this function (which reports local ids)
            // does not own. They are listed EXPLICITLY, with no catch-all, so
            // that adding an Operand variant is a COMPILE ERROR here instead
            // of silently contributing no uses.
            //
            // That catch-all was the class-level hole behind the in_xN family:
            // every private copy of this walker inherited it, one copy
            // (passes::collect_locals) then failed to see the receiver of a
            // field access, DCE deleted the definition that computed it, and
            // 7,453 reads fell back to an incoming-register name. Exhaustive
            // matching is the durable fix; the missing arm was only the
            // symptom.
            Self::Register(_)
            | Self::Immediate(_)
            | Self::FloatLiteral(_)
            | Self::DoubleLiteral(_)
            | Self::StringLiteral { .. }
            | Self::Method(_)
            | Self::Field(_)
            | Self::Type(_)
            | Self::MissingAbiArg
            | Self::Unknown => {}
        }
    }

    /// True if operand is a pure value usable in constant folding.
    pub fn is_pure_value(&self) -> bool {
        matches!(
            self,
            Self::Immediate(_) | Self::FloatLiteral(_) | Self::DoubleLiteral(_)
        )
    }
}

/// Numeric conversion directions for [`Op::Convert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversion {
    IntToFloat32,
    IntToFloat64,
    FloatToInt,
    Float32ToFloat64,
    Float64ToFloat32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareKind {
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
}

impl CompareKind {
    pub fn invert(self) -> Self {
        match self {
            Self::Equal => Self::NotEqual,
            Self::NotEqual => Self::Equal,
            Self::LessThan => Self::GreaterOrEqual,
            Self::GreaterOrEqual => Self::LessThan,
            Self::LessOrEqual => Self::GreaterThan,
            Self::GreaterThan => Self::LessOrEqual,
        }
    }

    pub fn is_signed_order(self) -> bool {
        !matches!(self, Self::Equal | Self::NotEqual)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Rol,
    Ror,
}

/// Float-specific arithmetic (single/double decided by operand types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatArithKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    /// Unary scalar FP square root (FSQRT / VSQRT / sqrtss / sqrtsd).
    Sqrt,
    /// Unary scalar FP absolute value (FABS / VABS / andps-mask).
    Abs,
    /// Unary FP rounding toward -infinity (FRINTM — `Math.Floor`).
    Floor,
    /// Unary FP rounding toward +infinity (FRINTP — `Math.Ceiling`).
    Ceiling,
    /// Unary FP rounding toward zero (FRINTZ — `Math.Truncate`).
    Truncate,
    /// Unary FP rounding to nearest integer (FRINTN/FRINTA — `Math.Round`).
    /// The tie rule (even vs away-from-zero) is not carried: the two encodings
    /// differ only in it and neither is recoverable from the operand.
    Round,
}

/// What a [`Op::Select`] does to its FALSE-side operand.
///
/// This is the whole of A64's conditional-select family: `CSEL` is the
/// identity, `CSINC` adds one, `CSINV` complements, `CSNEG` negates. The
/// `CSET` / `CSETM` aliases are `CSINC` / `CSINV` with both sources `xzr`,
/// which the lifter folds to a constant pair (`cond ? 1 : 0` and
/// `cond ? -1 : 0`) — that is how a managed `bool` materialises in optimized
/// A64, so getting these right is the difference between `b = x < y;` and
/// `b = __unknown;`. Same variants serve x86 `CMOVcc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectVariant {
    /// `CSEL` / `FCSEL`: `cond ? a : b`.
    Identity,
    /// `CSINC`: `cond ? a : b + 1`.
    Increment,
    /// `CSINV`: `cond ? a : ~b`.
    Invert,
    /// `CSNEG`: `cond ? a : -b`.
    Negate,
}

/// Which input lane feeds one OUTPUT lane of a vector rearrange.
///
/// This is the x86 twin of the p-code subpiece model: Ghidra addresses a
/// register as `(space, offset, size)` subpieces, so a lane permutation is a
/// list of COPYs of subpieces, and the semantics live in the CONSUMER rather
/// than in a mnemonic string. A register-granular IR cannot name a subpiece,
/// but it can name a *lane index* — which is the whole content of the operation
/// for the shuffle/unpack/shift family. Carrying the indices explicitly is what
/// turns `__shufps(a, b, 0x1b)` from a name we do not have into an operation we
/// do: a lane mapping can be folded, range-checked, and inverted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanePick {
    /// Lane `n` of source operand 1 (`operands[1]`), counting from the LOW end.
    A(u8),
    /// Lane `n` of source operand 2 (`operands[2]`).
    B(u8),
    /// A zero lane — the fill of a byte shift or a widen.
    Zero,
    /// An all-ones lane — `pcmpeq* reg, reg`.
    Ones,
}

impl LanePick {
    /// The lane index this pick reads, if it reads one.
    pub fn index(self) -> Option<u8> {
        match self {
            Self::A(n) | Self::B(n) => Some(n),
            Self::Zero | Self::Ones => None,
        }
    }

    /// True when the pick reads one of the source operands.
    pub fn is_source(self) -> bool {
        self.index().is_some()
    }

    /// True when the pick reads the SECOND source operand (`operands[2]`).
    pub fn is_second_source(self) -> bool {
        matches!(self, Self::B(_))
    }
}

/// A whole-register vector operation whose semantics are EXACT.
///
/// Deliberately not a mnemonic string: every fact a consumer needs to reason
/// about the value is present as data, so the operation can be folded, proved,
/// and rendered rather than merely named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimdOp {
    /// `dst = rearrange(a, b)` with one [`LanePick`] per OUTPUT lane, LOW lane
    /// first.
    ///
    /// Operands are `[dst, a]` or `[dst, a, b]`: `LanePick::A(n)` reads lane
    /// `n` of `operands[1]` and `LanePick::B(n)` reads lane `n` of
    /// `operands[2]`, so a one-source form (a shift, a `pshufd`, an extract)
    /// simply never emits `B`. `elem_bytes` is the LANE WIDTH the indices count
    /// in — 1 (byte shift), 2 (`pshuflw`, `pextrw`), 4 (`shufps`, `pshufd`) or
    /// 8 (`shufpd`, `unpcklpd`) — because the same index means a different
    /// field at each width, and a `psrldq 4` is a 4-BYTE shift while a
    /// `pshufd` field 4 is a DWORD.
    Arrange {
        elem_bytes: u8,
        lanes: Vec<LanePick>,
    },
    /// `dst = a OP b` computed INDEPENDENTLY in every lane (`paddd`, `psubq`,
    /// `paddb`, …).
    ///
    /// Operands: `[dst, a, b]`. The arithmetic is the lane's own
    /// `elem_bytes`-wide modular operation: no carry ever crosses a lane
    /// boundary, which is exactly why a single register-wide add is the wrong
    /// model for it (and why these stayed stubs).
    PackedArith { elem_bytes: u8, kind: ArithKind },
}

impl SimdOp {
    /// How many lanes the operation produces.
    pub fn out_lanes(&self) -> usize {
        match self {
            Self::Arrange { lanes, .. } => lanes.len(),
            // A 128-bit register split into `elem_bytes`-wide lanes.
            Self::PackedArith { elem_bytes, .. } => 16 / (*elem_bytes).max(1) as usize,
        }
    }

    /// The width in bytes of one lane.
    pub fn elem_bytes(&self) -> u8 {
        match self {
            Self::Arrange { elem_bytes, .. } | Self::PackedArith { elem_bytes, .. } => *elem_bytes,
        }
    }

    /// The lane index of the LAST output lane, i.e. the highest index any
    /// consumer must be able to address (`out_lanes() - 1`).
    pub fn last_lane(&self) -> u8 {
        self.out_lanes().saturating_sub(1).min(u8::MAX as usize) as u8
    }

    /// True when the operation reads `operands[2]`.
    pub fn reads_second_source(&self) -> bool {
        match self {
            Self::Arrange { lanes, .. } => lanes.iter().any(|l| l.is_second_source()),
            Self::PackedArith { .. } => true,
        }
    }

    /// Every lane index this operation reads, in output order (duplicates
    /// kept). Used to prove a lowering stays inside its source register.
    pub fn source_lanes(&self) -> Vec<LanePick> {
        match self {
            Self::Arrange { lanes, .. } => lanes.clone(),
            Self::PackedArith { .. } => (0..self.out_lanes() as u8).map(LanePick::A).collect(),
        }
    }

    /// Is every lane index this operation reads a legal lane of a 128-bit
    /// register at this lane width? A lowering that selects lane 5 of a
    /// 4-lane register is a bug in the lifter, and this is the check that
    /// turns it into a reported fact instead of a silent wrong value.
    ///
    /// The bound is the REGISTER's lane count (`16 / elem_bytes`), not the
    /// number of output lanes: an extract such as `pextrw` produces ONE lane
    /// while still selecting from all eight word lanes of its source, so
    /// bounding by the output count would reject every legal extract.
    pub fn source_lanes_in_range(&self) -> bool {
        match self {
            Self::Arrange { elem_bytes, lanes } => {
                let lanes_per_reg = (16 / (*elem_bytes).max(1)) as usize;
                lanes
                    .iter()
                    .all(|l| l.index().map_or(true, |n| (n as usize) < lanes_per_reg))
            }
            // Lane i of `a` and lane i of `b` are both legal by construction.
            Self::PackedArith { .. } => true,
        }
    }
}

impl fmt::Display for SimdOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arrange { elem_bytes, lanes } => {
                write!(f, "arrange{}", (*elem_bytes as u32) * 8)?;
                for (i, l) in lanes.iter().enumerate() {
                    write!(f, "{}", if i == 0 { "[" } else { "," })?;
                    match l {
                        LanePick::A(n) => write!(f, "a{n}")?,
                        LanePick::B(n) => write!(f, "b{n}")?,
                        LanePick::Zero => write!(f, "z")?,
                        LanePick::Ones => write!(f, "o")?,
                    }
                }
                write!(f, "]")
            }
            Self::PackedArith { elem_bytes, kind } => {
                write!(f, "p{kind:?}.{}", (*elem_bytes as u32) * 8)
            }
        }
    }
}

pub type InsnIndex = usize;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Move,
    Arith(ArithKind),
    FloatArith(FloatArithKind),
    Compare(CompareKind),
    Not,
    Negate,

    Jump(InsnIndex),
    CondJump(InsnIndex),

    /// Native stack adjustment (removed by stack analyzer).
    ShiftStack(i64),

    Call,
    CallVoid,
    /// Indirect call — rewritten to direct calls when resolvable; else diagnostic.
    IndirectCall,

    Return,
    Throw,
    NewObj(Option<NamedTypeRef>),
    NewArr(Option<ManagedType>),
    Box(Option<NamedTypeRef>),
    LoadLength,

    /// Jump-table switch: targets are instruction indices per case value.
    Switch(Vec<usize>),

    /// Numeric conversion (int↔float, widen/narrow).
    Convert(Conversion),

    /// Fused multiply-add: dest = op1 * op2 [+|−] op3 (negate flag for MSUB).
    MulAcc {
        negate_addend: bool,
        is_float: bool,
    },

    /// Count leading zeros.
    Clz,

    /// A value-producing instruction whose semantics are not modelled yet,
    /// carried as its own mnemonic over its real operands: `[dst, srcs…]`.
    ///
    /// This exists to uphold one invariant: **an instruction that writes a
    /// register must define that register.** The lifter's fall-through used to
    /// satisfy that by writing `dst = Unknown` — or, when the destination was a
    /// VECTOR register, by writing nothing at all, so the register silently
    /// kept its previous value and every later read fabricated a different
    /// expression. Both destroy dataflow. An intrinsic keeps the destination
    /// and every source, and names the operation with the opcode's own
    /// mnemonic (`__movi(...)`, `__umulh(...)`, `__tbl(...)`), which is
    /// exactly the trade `fnptr_0x…` makes for an unresolved call: a name we do
    /// not have yet must never cost the VALUE.
    Simd(String),

    /// A vector operation whose semantics ARE recovered: an explicit lane
    /// rearrange or a per-lane integer arithmetic.
    ///
    /// This is the successor to `Simd` for the one family that dominated the
    /// remaining x64 stub population — lane arrangement (`shufps` 2,385,
    /// `psrldq` 1,255, `unpcklps` 1,026, `unpcklpd`, `unpckhpd`, `pshufd`,
    /// `pextrw`, `punpcklqdq`) and packed integer arithmetic (`paddd`,
    /// `paddq`, `psubq`). Those mnemonics have no whole-register equivalent (a
    /// register-granular model computes `addps` correctly and `paddd`
    /// incorrectly — the packed integer forms carry between lanes and the FP
    /// ones do not), so they cannot go through `Op::Arith`, but their meaning
    /// is still fully known: it is a lane mapping. Keeping that mapping as
    /// DATA lets a consumer fold it, range-check it and render it, instead of
    /// printing a name and hoping.
    ///
    /// Operands: `[dst, a, b?]` — see [`SimdOp`].
    SimdLane(SimdOp),

    /// Conditional select — `dst = cond ? a : f(b)`.
    ///
    /// Operands: `[dst, cond, a, b]`. `cond` is a BOOLEAN VALUE: exactly the
    /// kind `Op::Compare` produces and `Op::CondJump` consumes. Reusing that
    /// representation instead of inventing a second way to say "a condition"
    /// is the point — SSA renaming, type propagation, DCE and the renderer's
    /// condition inlining all already understand a boolean operand, so this op
    /// attaches to the existing machinery rather than needing its own.
    ///
    /// A64 census: 17,281 occurrences in the reference binary
    /// (`CSINC` 7,488, `CSEL` 7,113, `FCSEL` 2,247, `CSNEG` 271, `CSINV` 162),
    /// every one of which used to fall through the lifter's catch-all and
    /// become `dst = Unknown`, silently deleting the branchless min/max/select
    /// idiom the compiler emits for `Math.Min`, `?:` on constants, and every
    /// clamps/clamp-like helper.
    Select(SelectVariant),

    /// Objective-C message send with a stub-fixed selector:
    /// dest(x0 slot) = [receiver sel:arg…]. Recovered from Apple-platform
    /// selector stubs; carries the REAL operation name (setBool:forKey: …).
    ObjcMsgSend {
        selector: String,
    },

    /// SSA phi — inserted at dominance frontiers, eliminated before emit.
    Phi,

    /// Marker for analysis-internal no-ops (never emitted).
    Nop,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Move => write!(f, "mov"),
            Self::Arith(k) => write!(f, "{k:?}"),
            Self::FloatArith(k) => write!(f, "f{k:?}"),
            Self::Compare(k) => write!(f, "cmp.{k:?}"),
            Self::Not => write!(f, "not"),
            Self::Negate => write!(f, "neg"),
            Self::Jump(t) => write!(f, "br -> {t}"),
            Self::CondJump(t) => write!(f, "br.true -> {t}"),
            Self::ShiftStack(d) => write!(f, "add sp, {d}"),
            Self::Call | Self::CallVoid | Self::IndirectCall => write!(f, "call"),
            Self::Return => write!(f, "ret"),
            Self::Throw => write!(f, "throw"),
            Self::NewObj(_) => write!(f, "newobj"),
            Self::NewArr(_) => write!(f, "newarr"),
            Self::Box(_) => write!(f, "box"),
            Self::LoadLength => write!(f, "ldlen"),
            Self::Switch(t) => write!(f, "switch[{}]", t.len()),
            Self::Convert(c) => write!(f, "conv.{c:?}"),
            Self::MulAcc { negate_addend, .. } => {
                write!(f, "madd{}", if *negate_addend { "-" } else { "+" })
            }
            Self::Clz => write!(f, "clz"),
            Self::Simd(name) => write!(f, "{}", name.to_lowercase()),
            Self::SimdLane(s) => write!(f, "{s}"),
            Self::Select(v) => write!(f, "sel.{v:?}"),
            Self::ObjcMsgSend { selector } => write!(f, "objc_msg[{selector}]"),
            Self::Phi => write!(f, "phi"),
            Self::Nop => write!(f, "nop"),
        }
    }
}

/// One IR instruction. Identity is its position in the flat listing; branch
/// targets are indices into the same Vec (Cpp2IL `Instruction.cs` convention:
/// no content-based Hash/Eq so targets stay stable across rewrites).
#[derive(Debug, Clone)]
pub struct Insn {
    pub op: Op,
    pub operands: Vec<Operand>,
    /// Optional type hint for the result (drives float coercion & elem types).
    pub dest_type: Option<ManagedType>,
    /// Original native address (diagnostics + debugging).
    pub address: u64,
}

impl Insn {
    pub fn new(op: Op, operands: Vec<Operand>, address: u64) -> Self {
        Self {
            op,
            operands,
            dest_type: None,
            address,
        }
    }

    pub fn move_(dest: Operand, src: Operand, address: u64) -> Self {
        Self::new(Op::Move, vec![dest, src], address)
    }

    pub fn nop(address: u64) -> Self {
        Self::new(Op::Nop, vec![], address)
    }

    /// Operand position holding this instruction's destination, per opcode
    /// convention (None for void ops). ONE source of truth: every pass
    /// (SSA, reaching defs, renderer) must agree on which slot is the write
    /// or def/use tracking silently drops whole op families. `Convert`,
    /// `MulAcc` and `Clz` write their result to operand 0 exactly like
    /// `Arith`; `Phi` is `[dest, args…]`.
    pub fn dest_position(op: &Op) -> Option<usize> {
        match op {
            Op::Call => Some(1),
            Op::Move
            | Op::Arith(_)
            | Op::FloatArith(_)
            | Op::Compare(_)
            | Op::Not
            | Op::Negate
            | Op::Convert(_)
            | Op::MulAcc { .. }
            | Op::Clz
            | Op::NewObj(_)
            | Op::NewArr(_)
            | Op::Box(_)
            | Op::LoadLength
            | Op::Phi
            | Op::Select(_)
            | Op::Simd(_)
            | Op::SimdLane(_) => Some(0),
            _ => None,
        }
    }

    /// Destination operand per opcode convention (None for void ops).
    pub fn dest(&self) -> Option<&Operand> {
        Self::dest_position(&self.op).and_then(|p| self.operands.get(p))
    }

    pub fn dest_mut(&mut self) -> Option<&mut Operand> {
        Self::dest_position(&self.op).and_then(|p| self.operands.get_mut(p))
    }

    /// Whether this instruction defines (writes) its destination local.
    pub fn is_definition(&self) -> bool {
        Self::dest_position(&self.op).is_some()
            && self.dest().map(|d| d.as_local().is_some()).unwrap_or(false)
    }

    /// Locals read by this instruction (excluding its destination).
    pub fn use_locals(&self) -> Vec<u32> {
        let mut uses = Vec::new();
        for (i, op) in self.operands.iter().enumerate() {
            // skip destination positions
            let is_dest_pos = Self::dest_position(&self.op) == Some(i);
            if is_dest_pos && matches!(op, Operand::Local(_)) {
                continue;
            }
            // AddressOf(local) is a definition-like use for escape analysis but
            // still counts as a use for liveness.
            op.uses(&mut uses);
        }
        uses
    }

    /// Can dead-code elimination remove this when no one uses the result?
    pub fn is_removable_if_dead(&self) -> bool {
        matches!(
            self.op,
            Op::Move | Op::Arith(_) | Op::FloatArith(_) | Op::Compare(_) | Op::Not
                | Op::Negate | Op::Convert(_) | Op::MulAcc { .. } | Op::Clz | Op::Phi
                | Op::LoadLength | Op::Select(_)
                // A lane op is a PURE function of its operands, so unlike the
                // semantic-free `Op::Simd` intrinsic (deliberately NOT listed
                // here — an unmodelled operation must not be deleted) a dead
                // one can be removed without losing an observation.
                | Op::SimdLane(_)
        )
    }

    pub fn is_terminator(&self) -> bool {
        matches!(
            self.op,
            Op::Jump(_)
                | Op::CondJump(_)
                | Op::Return
                | Op::Throw
                | Op::IndirectCall
                | Op::Switch(_)
        )
    }

    pub fn has_side_effects(&self) -> bool {
        matches!(
            self.op,
            Op::Call | Op::CallVoid | Op::IndirectCall | Op::Return | Op::Throw
        ) || self.is_terminator()
    }
}

impl fmt::Display for Insn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.op)?;
        let start = match self.op {
            Op::Call => 2.min(self.operands.len()),
            Op::CallVoid => 1.min(self.operands.len()),
            _ => 0,
        };
        for (i, o) in self.operands.iter().enumerate() {
            if i < start {
                continue;
            }
            if i == start || (start == 0 && i == 0) {
                write!(f, " {o}")?;
            } else {
                write!(f, ", {o}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IR type propagation (post-SSA)
// ---------------------------------------------------------------------------

/// What an SSA version can be shown to hold, for the purpose of naming a
/// subsequent `[base + offset]` access against the metadata field layouts.
///
/// The lattice is deliberately FLAT — three usable points and one internal
/// conflict marker:
///
/// * `Unknown` — no information. This is the bottom of the lattice and also
///   what a version with no usable definition carries.
/// * `Tdi(tdi)` — a value that denotes an instance of TypeDef `tdi` (an object
///   reference, or, by the convention `ResolveMaps::field_layouts` already
///   uses, the address of a value-type instance).
/// * `TdiOff(tdi, off)` — the ADDRESS of the field at byte offset `off` in an
///   instance of `tdi`. This is the type of `obj + 24`: it must not collapse
///   to `Tdi` (that is the historical bug where `base + off` was typed as the
///   base's class, so the load looked like field 0 of the base) and must not
///   collapse to `int` (that is what lost every chained `*(nint*)(v + 24)`).
/// * `Conflict` — two definitions proved DIFFERENT types. Joins only ever
///   keep equal known types; a conflict is *never* unioned into a base class,
///   because naming a derived field on a base instance (or vice versa) would
///   be a lie. `Conflict` exists only during the fixpoint: copies of a
///   conflicting value must not be re-typed optimistically, so it behaves as
///   a poison marker. It is stripped from the returned map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrType {
    Unknown,
    Tdi(i32),
    TdiOff(i32, i64),
    /// Internal join result: the definitions disagreed.
    Conflict,
}

impl IrType {
    /// The TypeDef index, when this is a plain instance type.
    pub fn tdi(self) -> Option<i32> {
        match self {
            Self::Tdi(t) if t >= 0 => Some(t),
            _ => None,
        }
    }

    fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// `value + delta` in address space.
    ///
    /// Only ADD may turn an instance into a field address (`Tdi + k`), and
    /// only ADD/SUB adjust an existing field address. An instance minus a
    /// constant is a pointer *before* the object, which is not a managed
    /// field, so it stays unknown.
    fn offset_by(self, delta: i64, allow_new_address: bool) -> Self {
        match self {
            Self::Tdi(t) if allow_new_address => Self::TdiOff(t, delta),
            Self::TdiOff(t, off) => match off.checked_add(delta) {
                Some(total) => Self::TdiOff(t, total),
                None => Self::Unknown,
            },
            Self::Conflict => Self::Conflict,
            Self::Tdi(_) | Self::Unknown => Self::Unknown,
        }
    }
}

/// Equal-only join. Unknown is "no information", not a distinct type, so it
/// is the identity; every two DIFFERENT known types join to `Conflict`.
fn join_types(a: IrType, b: IrType) -> IrType {
    match (a, b) {
        (IrType::Unknown, x) | (x, IrType::Unknown) => x,
        (x, y) if x == y => x,
        _ => IrType::Conflict,
    }
}

/// Where a `[base + addend]` address points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddrLoc {
    /// An instance field of `tdi` (offset already includes every folded
    /// constant).
    Field(i32, i64),
    /// An untyped storage cell: the SSA version `base` plus an offset. Two
    /// accesses with the same key denote the same bytes (SSA guarantees the
    /// base version's value is single-assignment), which is what makes a
    /// store→load type transfer sound without alias analysis.
    Cell(u32, i64),
    /// A fixed absolute address.
    Abs(u64),
}

/// Identity of a store/load cell for the straight-line store→load scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StoreKey {
    Cell(u32, i64),
    Abs(u64),
}

/// Result of [`infer_ir_types`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IrTypeInfo {
    /// SSA version → inferred type. `Unknown`/`Conflict` entries are omitted.
    pub version_types: HashMap<u32, IrType>,
}

impl IrTypeInfo {
    pub fn get(&self, version: u32) -> IrType {
        self.version_types
            .get(&version)
            .copied()
            .unwrap_or(IrType::Unknown)
    }
}

/// Run the IR edge type-propagation fixpoint over a post-SSA listing.
///
/// The edges are the ones the plan names:
///
/// * `COPY dst <- src` (a `Move` from a local) propagates the source type.
/// * merge/`Phi` joins only EQUAL known types (never unions); a version with
///   several definitions (SSA phi elimination plants one `Move` per edge) gets
///   a type only when every informative definition agrees.
/// * `dst = load(p)` takes the field's type at `p`'s `(tdi, offset)` from the
///   metadata layouts, or the type of the value most recently STORED to the
///   same cell in the same straight-line run.
/// * `store(p, v)` records `v`'s type for later loads of that exact cell.
/// * `base + off` keeps `(tdi, off)` as [`IrType::TdiOff`] instead of
///   collapsing to `int`/`Tdi`.
///
/// Roots (values that introduce a type without a same-method definition) are
/// the typed producers the IR already carries: `NewObj`, a resolved `Call`'s
/// return signature, `Field`/`InstanceField` operands, and the optional
/// `seeds` map (the local table's metadata-derived types, so `this` and typed
/// parameters can start a chain).
pub fn infer_ir_types(insns: &[Insn], maps: &ResolveMaps) -> IrTypeInfo {
    infer_ir_types_seeded(insns, maps, &HashMap::new())
}

/// [`infer_ir_types`] with externally-supplied entry types (see
/// `field_annotation::infer_ir_type_map`, which seeds from the `LocalTable`).
pub fn infer_ir_types_seeded(
    insns: &[Insn],
    maps: &ResolveMaps,
    seeds: &HashMap<u32, IrType>,
) -> IrTypeInfo {
    if insns.is_empty() {
        return IrTypeInfo::default();
    }

    // Definition sites per version. A version with multiple sites is a merge
    // (phi elimination / loop rotation); SSA itself gives one.
    let mut def_sites: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, insn) in insns.iter().enumerate() {
        if let Some(Operand::Local(v)) = insn.dest() {
            def_sites.entry(*v).or_default().push(i);
        }
    }

    let roots = alias_roots(insns);
    let boundary = run_boundaries(insns);

    let mut env: HashMap<u32, IrType> = HashMap::new();
    for (&v, ty) in seeds {
        if !def_sites.contains_key(&v) && !ty.is_unknown() {
            env.insert(v, *ty);
        }
    }

    // A flat lattice with equal-only joins is not monotone under a naive
    // worklist (a version can move known → unknown when a second definition
    // disagrees), so each round RECOMPUTES every definition from the previous
    // round's environment. Chains settle in 2–4 rounds; the cap turns a
    // pathological cycle into a deterministic answer instead of a hang.
    const MAX_ROUNDS: usize = 24;
    for _round in 0..MAX_ROUNDS {
        // Straight-line store→load snapshot, computed from the previous env.
        let stored = scan_stores(insns, &env, maps, &roots, &boundary);

        let mut next_env: HashMap<u32, IrType> = HashMap::new();
        for (&v, sites) in &def_sites {
            let mut acc = IrType::Unknown;
            let mut informative = false;
            for &i in sites {
                let t = def_type(&insns[i], &env, maps, &roots, stored[i]);
                if t.is_unknown() {
                    continue;
                }
                informative = true;
                acc = join_types(acc, t);
            }
            // A merge with no informative arm may still fall back to the
            // table's own type for this exact version (e.g. a `Box`, whose
            // boxed-data offsets are not modelled here). A CONFLICT is real
            // evidence of disagreement and never falls back.
            if !informative {
                if let Some(seed) = seeds.get(&v).filter(|t| !t.is_unknown()) {
                    acc = *seed;
                    informative = true;
                }
            }
            if informative {
                next_env.insert(v, acc);
            }
        }
        // Versions with no definition are entry values; re-apply seeds every
        // round so they survive the rebuild above.
        for (&v, ty) in seeds {
            if !def_sites.contains_key(&v) && !ty.is_unknown() {
                next_env.insert(v, *ty);
            }
        }

        if next_env == env {
            env = next_env;
            break;
        }
        env = next_env;
    }

    let version_types = env
        .into_iter()
        .filter(|(_, t)| matches!(t, IrType::Tdi(_) | IrType::TdiOff(_, _)))
        .collect();
    IrTypeInfo { version_types }
}

/// The type a single definition produces.
fn def_type(
    insn: &Insn,
    env: &HashMap<u32, IrType>,
    maps: &ResolveMaps,
    roots: &HashMap<u32, u32>,
    stored: Option<IrType>,
) -> IrType {
    match &insn.op {
        Op::Move => {
            let from_src = operand_value_type(insn.operands.get(1), env, maps, roots, stored);
            if !from_src.is_unknown() {
                return from_src;
            }
            // `resolve`/`interprocedural` stamp resolved field loads with the
            // field's declared type; use it when this pass modelled the load
            // conservatively (narrow-width refusal, unmodelled base).
            tdi_of_managed(insn.dest_type.as_ref(), maps)
        }
        Op::Phi => {
            let mut acc = IrType::Unknown;
            for arg in insn.operands.iter().skip(1) {
                let t = operand_value_type(Some(arg), env, maps, roots, None);
                acc = join_types(acc, t);
            }
            acc
        }
        // A branchless select between two references has the common type only
        // when both arms are the same type; Identity is the only variant that
        // returns an input unchanged.
        Op::Select(SelectVariant::Identity) => {
            let a = operand_value_type(insn.operands.get(2), env, maps, roots, None);
            let b = operand_value_type(insn.operands.get(3), env, maps, roots, None);
            join_types(a, b)
        }
        Op::Arith(ArithKind::Add) => {
            let a = insn.operands.get(1);
            let b = insn.operands.get(2);
            match (a, b) {
                (Some(Operand::Immediate(k)), other) => {
                    operand_value_type(other, env, maps, roots, None).offset_by(*k, true)
                }
                (other, Some(Operand::Immediate(k))) => {
                    operand_value_type(other, env, maps, roots, None).offset_by(*k, true)
                }
                _ => IrType::Unknown,
            }
        }
        Op::Arith(ArithKind::Sub) => match insn.operands.get(2) {
            Some(Operand::Immediate(k)) => {
                operand_value_type(insn.operands.get(1), env, maps, roots, None)
                    .offset_by(-*k, false)
            }
            _ => IrType::Unknown,
        },
        Op::NewObj(Some(t)) => tdi_of_named_type(&t.ty),
        Op::Call => {
            let from_dest = tdi_of_managed(insn.dest_type.as_ref(), maps);
            if !from_dest.is_unknown() {
                return from_dest;
            }
            match insn.operands.first() {
                Some(Operand::Method(m)) => tdi_of_managed(Some(&m.sig.ret), maps),
                _ => IrType::Unknown,
            }
        }
        _ => IrType::Unknown,
    }
}

/// The address a `[base + addend]` operand denotes, when it can be proven.
fn address_loc(
    base: &Operand,
    addend: i64,
    env: &HashMap<u32, IrType>,
    _roots: &HashMap<u32, u32>,
) -> Option<AddrLoc> {
    match base {
        Operand::Local(v) => match env.get(v).copied().unwrap_or(IrType::Unknown) {
            IrType::Tdi(t) => Some(AddrLoc::Field(t, addend)),
            IrType::TdiOff(t, off) => off
                .checked_add(addend)
                .map(|total| AddrLoc::Field(t, total)),
            // `Conflict` still identifies a cell: the VALUE type is unknown but
            // the address `version + addend` is well defined.
            IrType::Conflict | IrType::Unknown => Some(AddrLoc::Cell(*v, addend)),
        },
        Operand::Immediate(a) => Some(AddrLoc::Abs(a.wrapping_add(addend) as u64)),
        _ => None,
    }
}

/// What a load through `Memory` can be shown to produce.
fn load_type(
    base: &Operand,
    addend: i64,
    width: u8,
    index_present: bool,
    env: &HashMap<u32, IrType>,
    maps: &ResolveMaps,
    roots: &HashMap<u32, u32>,
    stored: Option<IrType>,
) -> IrType {
    // Only pointer-width accesses can produce an object reference. 1/2/4 are
    // narrow integer accesses; treating one as a reference would name fields
    // on an `int`.
    if index_present || !(width == 0 || width == 8) {
        return IrType::Unknown;
    }
    // A store earlier in the same straight-line run is the most precise fact:
    // it is the exact value the load will observe. Unknown/Conflict stores do
    // not veto the declared layout below — a value stored into a field is
    // assignable to the field's declared type.
    if let Some(t @ (IrType::Tdi(_) | IrType::TdiOff(_, _))) = stored {
        return t;
    }
    let loc = match address_loc(base, addend, env, roots) {
        Some(loc) => loc,
        None => return IrType::Unknown,
    };
    match loc {
        AddrLoc::Field(tdi, off) => {
            match maps.field_types.get(&(tdi, off)).copied() {
                Some(ftdi) if ftdi >= 0 => IrType::Tdi(ftdi),
                _ => IrType::Unknown,
            }
        }
        // An untyped cell without a preceding stored value has no recoverable
        // type; an absolute address is only typed by the store scan (static
        // field layouts name the DECLARING type, not the field's type).
        AddrLoc::Cell(_, _) | AddrLoc::Abs(_) => IrType::Unknown,
    }
}

/// Type of a value operand (not an address).
fn operand_value_type(
    op: Option<&Operand>,
    env: &HashMap<u32, IrType>,
    maps: &ResolveMaps,
    roots: &HashMap<u32, u32>,
    stored: Option<IrType>,
) -> IrType {
    match op {
        Some(Operand::Local(v)) => env.get(v).copied().unwrap_or(IrType::Unknown),
        Some(Operand::Field(f)) => tdi_of_managed(Some(&f.field_type), maps),
        Some(Operand::InstanceField { field, .. }) => {
            tdi_of_managed(Some(&field.field_type), maps)
        }
        Some(Operand::Memory {
            base,
            addend,
            index,
            width,
            ..
        }) => load_type(
            base,
            *addend,
            *width,
            index.is_some(),
            env,
            maps,
            roots,
            stored,
        ),
        // `Operand::Type`/klass immediates deliberately do NOT become `Tdi`:
        // a klass pointer is not an instance, and `klass + 184` (static_fields)
        // must never be mistaken for an instance field at 184. The statement
        // pass has its own klass-aware path (`infer_klass_var_types`).
        _ => IrType::Unknown,
    }
}

/// `NamedType` → `Tdi`, for producers whose result is an instance reference.
fn tdi_of_named_type(t: &super::types::NamedType) -> IrType {
    if t.type_def_index >= 0 {
        IrType::Tdi(t.type_def_index)
    } else {
        IrType::Unknown
    }
}

/// `ManagedType` → `Tdi` when it carries a TypeDef index.
fn tdi_of_managed(ty: Option<&ManagedType>, _maps: &ResolveMaps) -> IrType {
    match ty {
        Some(ManagedType::Class(nt)) if nt.type_def_index >= 0 => {
            IrType::Tdi(nt.type_def_index)
        }
        // A by-ref/pointer value is an ADDRESS of the type, and this pass's
        // `Tdi` convention already treats `Tdi(t)` as "a base whose
        // `[v + off]` is a field of t" — exactly what a by-ref struct base is.
        Some(ManagedType::ByRef(inner)) | Some(ManagedType::Pointer(inner)) => {
            tdi_of_managed(Some(inner), _maps)
        }
        _ => IrType::Unknown,
    }
}

/// Resolve single-definition copy chains so a store through `v` and a load
/// through `w = v` share one cell key. Only SSA `Move dst, Local(src)` edges
/// with a single definition of `dst` are followed: a multiply-assigned version
/// has no single value.
fn alias_roots(insns: &[Insn]) -> HashMap<u32, u32> {
    let mut copy_src: HashMap<u32, u32> = HashMap::new();
    let mut multi: HashSet<u32> = HashSet::new();
    for insn in insns {
        if insn.op != Op::Move {
            continue;
        }
        let Some(Operand::Local(dst)) = insn.dest() else {
            continue;
        };
        let Some(Operand::Local(src)) = insn.operands.get(1) else {
            continue;
        };
        if copy_src.insert(*dst, *src).is_some() {
            multi.insert(*dst);
        }
    }
    let mut roots: HashMap<u32, u32> = HashMap::new();
    for &v in copy_src.keys() {
        if multi.contains(&v) {
            continue;
        }
        let mut cur = v;
        let mut seen: HashSet<u32> = HashSet::new();
        let mut ok = true;
        // Bound the walk: copy chains from phi elimination are short, and a
        // cycle is a malformed graph rather than a reason to spin.
        for _ in 0..16 {
            if !seen.insert(cur) {
                ok = false;
                break;
            }
            match copy_src.get(&cur) {
                Some(&next) if !multi.contains(&next) => cur = next,
                _ => break,
            }
        }
        if ok && cur != v {
            roots.insert(v, cur);
        }
    }
    roots
}

/// Straight-line-region boundaries for the store→load scan: entry, every
/// branch target, and every instruction after a terminator. Stores do not flow
/// across a join without SSA memory renaming, so the scan restarts there.
fn run_boundaries(insns: &[Insn]) -> Vec<bool> {
    let n = insns.len();
    let mut boundary = vec![false; n];
    if n == 0 {
        return boundary;
    }
    boundary[0] = true;
    for insn in insns {
        match &insn.op {
            Op::Jump(t) | Op::CondJump(t) => {
                if *t < n {
                    boundary[*t] = true;
                }
            }
            Op::Switch(targets) => {
                for &t in targets {
                    if t < n {
                        boundary[t] = true;
                    }
                }
            }
            _ => {}
        }
    }
    for i in 1..n {
        if insns[i - 1].is_terminator() {
            boundary[i] = true;
        }
    }
    boundary
}

fn is_call_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Call | Op::CallVoid | Op::IndirectCall | Op::ObjcMsgSend { .. }
    )
}

/// Store→load snapshot for one fixpoint round.
///
/// For each instruction, the type of the value the load will observe when a
/// STORE to the same cell precedes it in the same straight-line run (with no
/// call in between — a call owns the outgoing argument area, so a spill below
/// `sp` is no longer provably the stored value afterwards).
fn scan_stores(
    insns: &[Insn],
    env: &HashMap<u32, IrType>,
    maps: &ResolveMaps,
    roots: &HashMap<u32, u32>,
    boundary: &[bool],
) -> Vec<Option<IrType>> {
    let mut out = vec![None; insns.len()];
    let mut run: HashMap<StoreKey, IrType> = HashMap::new();
    for (i, insn) in insns.iter().enumerate() {
        if boundary.get(i).copied().unwrap_or(true) {
            run.clear();
        }
        if is_call_op(&insn.op) {
            run.clear();
            continue;
        }
        if insn.op != Op::Move {
            continue;
        }
        // Store: Move(Memory{..}, value)
        if let Some(Operand::Memory {
            base,
            addend,
            index: None,
            width,
            ..
        }) = insn.operands.first()
        {
            if !(*width == 0 || *width == 8) {
                continue;
            }
            let key = match base.as_ref() {
                Operand::Local(v) => {
                    Some(StoreKey::Cell(*roots.get(v).unwrap_or(v), *addend))
                }
                Operand::Immediate(a) => {
                    Some(StoreKey::Abs(a.wrapping_add(*addend) as u64))
                }
                _ => None,
            };
            if let Some(key) = key {
                let ty = operand_value_type(insn.operands.get(1), env, maps, roots, None);
                run.insert(key, ty);
            }
            continue;
        }
        // Load: Move(dst, Memory{..})
        if let Some(Operand::Memory {
            base,
            addend,
            index,
            width,
            ..
        }) = insn.operands.get(1)
        {
            if index.is_some() || !(*width == 0 || *width == 8) {
                continue;
            }
            let key = match base.as_ref() {
                Operand::Local(v) => {
                    Some(StoreKey::Cell(*roots.get(v).unwrap_or(v), *addend))
                }
                Operand::Immediate(a) => {
                    Some(StoreKey::Abs(a.wrapping_add(*addend) as u64))
                }
                _ => None,
            };
            if let Some(key) = key {
                out[i] = run.get(&key).copied();
            }
        }
    }
    out
}

#[cfg(test)]
mod ir_type_tests {
    use super::*;
    use crate::recovery::types::{NamedType, PrimKind};
    use std::sync::Arc;

    fn empty_maps() -> ResolveMaps {
        ResolveMaps {
            methods_by_rva: HashMap::new(),
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
            vtable_slots: HashMap::new(),
            klass_to_tdi: HashMap::new(),
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

    fn maps_with_fields(fields: &[(i32, i64, i32)]) -> ResolveMaps {
        let mut maps = empty_maps();
        for &(tdi, off, ftdi) in fields {
            maps.field_types.insert((tdi, off), ftdi);
        }
        maps
    }

    fn local(v: u32) -> Operand {
        Operand::Local(v)
    }

    fn move_local(dst: u32, src: u32) -> Insn {
        Insn::move_(local(dst), local(src), 0x1000)
    }

    fn add_imm(dst: u32, src: u32, k: i64) -> Insn {
        Insn::new(
            Op::Arith(ArithKind::Add),
            vec![local(dst), local(src), Operand::Immediate(k)],
            0x1000,
        )
    }

    fn load(dst: u32, base: u32, addend: i64, width: u8) -> Insn {
        Insn::new(
            Op::Move,
            vec![
                local(dst),
                Operand::Memory {
                    base: Box::new(local(base)),
                    index: None,
                    scale: 1,
                    addend,
                    width,
                },
            ],
            0x1000,
        )
    }

    fn store(base: u32, addend: i64, src: u32) -> Insn {
        Insn::new(
            Op::Move,
            vec![
                Operand::Memory {
                    base: Box::new(local(base)),
                    index: None,
                    scale: 1,
                    addend,
                    width: 0,
                },
                local(src),
            ],
            0x1000,
        )
    }

    fn newobj(dst: u32, tdi: i32) -> Insn {
        Insn::new(
            Op::NewObj(Some(NamedTypeRef {
                ty: NamedType {
                    namespace: String::new(),
                    name: format!("T{tdi}"),
                    assembly: None,
                    type_def_index: tdi,
                },
            })),
            vec![local(dst)],
            0x1000,
        )
    }

    fn seeded(seeds: &[(u32, IrType)], insns: &[Insn], maps: &ResolveMaps) -> IrTypeInfo {
        let seeds: HashMap<u32, IrType> = seeds.iter().copied().collect();
        infer_ir_types_seeded(insns, maps, &seeds)
    }

    /// A COPY chain propagates the entry type through every hop.
    #[test]
    fn copy_chain_propagates_the_seeded_type() {
        let info = seeded(
            &[(0, IrType::Tdi(5))],
            &[move_local(1, 0), move_local(2, 1)],
            &empty_maps(),
        );
        assert_eq!(info.get(1), IrType::Tdi(5));
        assert_eq!(info.get(2), IrType::Tdi(5));
    }

    /// `base + off` keeps the (tdi, off) pair — it must NOT become `Tdi`
    /// (that would make the next load look like field 0 of the base) and must
    /// not become `int` (that is the historical loss).
    #[test]
    fn base_plus_offset_keeps_a_field_address() {
        let info = seeded(
            &[(0, IrType::Tdi(5))],
            &[add_imm(1, 0, 24), add_imm(2, 1, 8)],
            &empty_maps(),
        );
        assert_eq!(info.get(1), IrType::TdiOff(5, 24));
        assert_eq!(info.get(2), IrType::TdiOff(5, 32));
        assert_eq!(info.get(1).tdi(), None, "an address is not an instance");
    }

    /// `dst = load(base + addend)` takes the FIELD's declared type from the
    /// metadata layout, for both an instance base and a field-address base.
    #[test]
    fn loads_take_the_metadata_field_type() {
        let maps = maps_with_fields(&[(5, 16, 10), (10, 8, 20)]);
        let info = seeded(
            &[(0, IrType::Tdi(5))],
            &[
                // w = *(T5*)(obj + 16)  -> T10
                load(1, 0, 16, 0),
                // addr = obj + 16 ; x = *(T10*)(addr) -> T10
                add_imm(2, 0, 16),
                load(3, 2, 0, 0),
                // y = *(T20*)(x + 8) -> T20 (chained)
                load(4, 3, 8, 0),
            ],
            &maps,
        );
        assert_eq!(info.get(1), IrType::Tdi(10));
        assert_eq!(info.get(2), IrType::TdiOff(5, 16));
        assert_eq!(info.get(3), IrType::Tdi(10));
        assert_eq!(info.get(4), IrType::Tdi(20));
    }

    /// A STORE records the stored value's type for a later load of the same
    /// cell — here the metadata layout knows nothing about the field.
    #[test]
    fn a_store_types_the_later_load_of_the_same_cell() {
        let maps = empty_maps();
        let info = infer_ir_types(
            &[
                newobj(3, 10),
                store(0, 24, 3),
                load(1, 0, 24, 0),
            ],
            &maps,
        );
        assert_eq!(info.get(1), IrType::Tdi(10));
    }

    /// A store/load through a COPY of the base hits the same cell too.
    #[test]
    fn a_store_then_copy_then_load_shares_the_cell() {
        let info = infer_ir_types(
            &[
                newobj(3, 10),
                store(0, 24, 3),
                move_local(4, 0),
                load(1, 4, 24, 0),
            ],
            &empty_maps(),
        );
        assert_eq!(info.get(1), IrType::Tdi(10));
    }

    /// A call between the store and the load invalidates the spill.
    #[test]
    fn a_call_breaks_the_store_to_load_transfer() {
        let call = Insn::new(Op::Call, vec![Operand::Unknown, local(9)], 0x1000);
        let info = infer_ir_types(
            &[newobj(3, 10), store(0, 24, 3), call, load(1, 0, 24, 0)],
            &empty_maps(),
        );
        assert_eq!(info.get(1), IrType::Unknown);
    }

    /// Merge/phi joins keep only EQUAL known types; disagreement is a
    /// conflict (dropped), never a union to a base class.
    #[test]
    fn a_merge_of_equal_types_keeps_the_type_and_a_conflict_drops_it() {
        let phi = |dst: u32, a: u32, b: u32| {
            Insn::new(
                Op::Phi,
                vec![local(dst), local(a), local(b)],
                0x1000,
            )
        };
        let maps = empty_maps();
        let equal = seeded(
            &[(1, IrType::Tdi(5)), (2, IrType::Tdi(5))],
            &[phi(3, 1, 2)],
            &maps,
        );
        assert_eq!(equal.get(3), IrType::Tdi(5));

        let disagree = seeded(
            &[(1, IrType::Tdi(5)), (2, IrType::Tdi(6))],
            &[phi(3, 1, 2)],
            &maps,
        );
        assert_eq!(disagree.get(3), IrType::Unknown);

        // A copy of a conflicting value must not be re-typed by another arm.
        let poisoned = seeded(
            &[(1, IrType::Tdi(5)), (2, IrType::Tdi(6))],
            &[phi(3, 1, 2), move_local(4, 3)],
            &maps,
        );
        assert_eq!(poisoned.get(4), IrType::Unknown);
    }

    /// The same rule for phi ELIMINATION: several Moves define one version.
    #[test]
    fn merge_defined_by_edge_moves_requires_agreement() {
        let maps = empty_maps();
        let info = seeded(
            &[(1, IrType::Tdi(5)), (2, IrType::Tdi(5))],
            &[move_local(3, 1), move_local(3, 2)],
            &maps,
        );
        assert_eq!(info.get(3), IrType::Tdi(5));

        let conflict = seeded(
            &[(1, IrType::Tdi(5)), (2, IrType::TdiOff(5, 8))],
            &[move_local(3, 1), move_local(3, 2)],
            &maps,
        );
        assert_eq!(conflict.get(3), IrType::Unknown);
    }

    /// A narrow load is an integer access, never an object reference.
    #[test]
    fn narrow_loads_are_not_typed_as_references() {
        let maps = maps_with_fields(&[(5, 16, 10)]);
        let info = seeded(
            &[(0, IrType::Tdi(5))],
            &[load(1, 0, 16, 1), load(2, 0, 16, 2), load(3, 0, 16, 4)],
            &maps,
        );
        assert_eq!(info.get(1), IrType::Unknown);
        assert_eq!(info.get(2), IrType::Unknown);
        assert_eq!(info.get(3), IrType::Unknown);
        // ...while the pointer-width form of the same offset is typed.
        let wide = seeded(&[(0, IrType::Tdi(5))], &[load(4, 0, 16, 0)], &maps);
        assert_eq!(wide.get(4), IrType::Tdi(10));
    }

    /// `NewObj` is a root; a call's resolved return type is a root.
    #[test]
    fn producers_seed_types() {
        let call = Insn::new(
            Op::Call,
            vec![Operand::Unknown, local(2)],
            0x1000,
        );
        let mut call = call;
        call.dest_type = Some(ManagedType::Class(Arc::new(NamedType {
            namespace: String::new(),
            name: "T7".into(),
            assembly: None,
            type_def_index: 7,
        })));
        let info = infer_ir_types(&[newobj(1, 6), call], &empty_maps());
        assert_eq!(info.get(1), IrType::Tdi(6));
        assert_eq!(info.get(2), IrType::Tdi(7));
    }

    /// An `int`-typed value never becomes an address: adding a constant to an
    /// Unknown value stays Unknown.
    #[test]
    fn adding_to_an_untyped_value_produces_nothing() {
        let info = infer_ir_types(&[add_imm(1, 0, 16)], &empty_maps());
        assert_eq!(info.get(1), IrType::Unknown);
    }

    /// A `Sub` of an instance is not an instance field address.
    #[test]
    fn subtracting_from_an_instance_is_not_a_field_address() {
        let sub = Insn::new(
            Op::Arith(ArithKind::Sub),
            vec![local(1), local(0), Operand::Immediate(8)],
            0x1000,
        );
        let info = seeded(&[(0, IrType::Tdi(5))], &[sub], &empty_maps());
        assert_eq!(info.get(1), IrType::Unknown);
        // ...but subtracting from an existing field address walks backwards.
        let sub2 = Insn::new(
            Op::Arith(ArithKind::Sub),
            vec![local(2), local(1), Operand::Immediate(8)],
            0x1000,
        );
        let chain = seeded(
            &[(0, IrType::Tdi(5))],
            &[add_imm(1, 0, 24), sub2],
            &empty_maps(),
        );
        assert_eq!(chain.get(2), IrType::TdiOff(5, 16));
    }

    /// `PrimKind` is linked from the test only to keep the type imports live.
    #[test]
    fn field_type_is_metadata_derived() {
        let maps = maps_with_fields(&[(5, 0, 10)]);
        let info = seeded(&[(0, IrType::Tdi(5))], &[load(1, 0, 0, 0)], &maps);
        assert_eq!(info.get(1), IrType::Tdi(10));
        let _ = PrimKind::I4;
    }
}

#[cfg(test)]
mod simd_lane_tests {
    use super::*;

    fn arrange(elem_bytes: u8, lanes: Vec<LanePick>) -> SimdOp {
        SimdOp::Arrange { elem_bytes, lanes }
    }

    /// The four DWORD lanes of `shufps d, s, 0x1b` — 0x1b = 0b00_01_10_11, so
    /// lane 0 takes d[3], lane 1 d[2], lane 2 s[1], lane 3 s[0].
    #[test]
    fn a_lane_mapping_is_fully_determined_by_its_picks() {
        let op = arrange(
            4,
            vec![
                LanePick::A(3),
                LanePick::A(2),
                LanePick::B(1),
                LanePick::B(0),
            ],
        );
        assert_eq!(op.out_lanes(), 4);
        assert_eq!(op.elem_bytes(), 4);
        assert!(op.reads_second_source());
        assert!(op.source_lanes_in_range());
        assert_eq!(op.to_string(), "arrange32[a3,a2,b1,b0]");
    }

    /// A one-source form never reads `operands[2]`.
    #[test]
    fn a_one_source_arrange_does_not_claim_a_second_source() {
        let op = arrange(4, vec![LanePick::A(0); 4]);
        assert!(!op.reads_second_source());
        assert!(op.source_lanes_in_range());
    }

    /// An out-of-range pick is INVALID — the check exists so a lifter bug is a
    /// reported fact rather than a silent wrong value.
    #[test]
    fn an_out_of_range_lane_pick_is_rejected() {
        // four DWORD lanes, so index 4 does not exist
        let op = arrange(
            4,
            vec![
                LanePick::A(0),
                LanePick::A(1),
                LanePick::A(2),
                LanePick::A(4),
            ],
        );
        assert!(!op.source_lanes_in_range());
    }

    /// A byte shift is 16 BYTE lanes, so its indices are legal up to 15.
    #[test]
    fn a_byte_shift_addresses_sixteen_lanes() {
        let mut lanes: Vec<LanePick> = (4..16).map(LanePick::A).collect();
        lanes.extend(std::iter::repeat_n(LanePick::Zero, 4));
        let op = arrange(1, lanes);
        assert_eq!(op.out_lanes(), 16);
        assert_eq!(op.elem_bytes(), 1);
        assert!(op.source_lanes_in_range());
        assert_eq!(op.last_lane(), 15);
    }

    /// Packed integer arithmetic covers every lane of the register.
    #[test]
    fn packed_arithmetic_covers_the_whole_register() {
        let op = SimdOp::PackedArith {
            elem_bytes: 4,
            kind: ArithKind::Add,
        };
        assert_eq!(op.out_lanes(), 4);
        assert!(op.reads_second_source());
        assert!(op.source_lanes_in_range());
        assert_eq!(op.to_string(), "pAdd.32");
        let q = SimdOp::PackedArith {
            elem_bytes: 8,
            kind: ArithKind::Sub,
        };
        assert_eq!(q.out_lanes(), 2);
        assert_eq!(q.to_string(), "pSub.64");
    }

    /// An EXTRACT selects from every lane of the register it reads, even
    /// though it produces only one — the check must not confuse the two.
    #[test]
    fn an_extract_may_select_any_lane_of_its_register() {
        // 8 word lanes in a register; an extract of lane 7 is perfectly legal
        // even though the result is a single lane.
        let op = arrange(2, vec![LanePick::A(7)]);
        assert_eq!(op.out_lanes(), 1);
        assert!(op.source_lanes_in_range());
        // ...and lane 8 is not.
        assert!(!arrange(2, vec![LanePick::A(8)]).source_lanes_in_range());
    }

    /// A lane op DEFINES its destination, and it is pure.
    #[test]
    fn a_lane_op_defines_its_destination_and_is_pure() {
        // A real definition has a LOCAL destination, so the destination here is
        // a local slot and the source is the register.
        let i = Insn::new(
            Op::SimdLane(arrange(4, vec![LanePick::A(0); 4])),
            vec![Operand::Local(7), Operand::Register("vf0".into())],
            0x1000,
        );
        assert_eq!(Insn::dest_position(&i.op), Some(0));
        assert!(i.is_definition());
        assert!(i.is_removable_if_dead(), "a lane op is a pure function");
        assert!(!i.has_side_effects());
        assert!(!i.is_terminator());
        assert_eq!(i.use_locals(), Vec::<u32>::new());
        // ...and the REGISTER form of the same op is not a definition, which is
        // the invariant every def/use pass relies on.
        let r = Insn::new(
            Op::SimdLane(arrange(4, vec![LanePick::A(0); 4])),
            vec![
                Operand::Register("vf0".into()),
                Operand::Register("vf1".into()),
            ],
            0x1000,
        );
        assert!(!r.is_definition());
    }

    /// A constant lane (`Ones`) reads no source at all.
    #[test]
    fn a_constant_lane_reads_nothing() {
        let op = arrange(4, vec![LanePick::Ones; 4]);
        assert!(!op.reads_second_source());
        assert!(!op.source_lanes().iter().any(|l| l.is_source()));
        assert!(op.source_lanes_in_range());
    }
}
