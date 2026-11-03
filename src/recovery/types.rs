//! Managed type model used throughout recovery — independent of dotnetdll's
//! Resolution-bound types so analysis passes never need live token tables.
//! Conversion to `dotnetdll` signatures happens only in the emitter via the
//! TokenResolver (mirrors Cpp2IL's TypeSignature split between analysis and
//! AsmResolver output stages).

use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimKind {
    Boolean,
    Char,
    I1,
    U1,
    I2,
    U2,
    I4,
    U4,
    I8,
    U8,
    R4,
    R8,
    I,
    U,
    Void,
}

impl PrimKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Boolean => "bool",
            Self::Char => "char",
            Self::I1 => "sbyte",
            Self::U1 => "byte",
            Self::I2 => "short",
            Self::U2 => "ushort",
            Self::I4 => "int",
            Self::U4 => "uint",
            Self::I8 => "long",
            Self::U8 => "ulong",
            Self::R4 => "float",
            Self::R8 => "double",
            Self::I => "nint",
            Self::U => "nuint",
            Self::Void => "void",
        }
    }

    /// Value-type-ness drives ldnull-vs-ldc.i4.0 default fills.
    pub fn is_value_type(self) -> bool {
        !matches!(self, Self::Void)
    }

    pub fn il_size(self) -> usize {
        match self {
            Self::I1 | Self::U1 | Self::Boolean => 1,
            Self::I2 | Self::U2 | Self::Char => 2,
            Self::I4 | Self::U4 | Self::R4 => 4,
            _ => 8,
        }
    }
}

/// Reference to a named managed type (class/value-type/interface/array...).
#[derive(Debug, Clone, PartialEq)]
pub struct NamedType {
    pub namespace: String,
    pub name: String,
    /// Assembly simple name for external references.
    pub assembly: Option<String>,
    /// TypeDef index in global metadata (-1 when unknown).
    pub type_def_index: i32,
}

impl fmt::Display for NamedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.namespace.is_empty() {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}.{}", self.namespace, self.name)
        }
    }
}

impl NamedType {
    pub fn object() -> Arc<Self> {
        Arc::new(Self {
            namespace: "System".into(),
            name: "Object".into(),
            assembly: Some("mscorlib".into()),
            type_def_index: -1,
        })
    }

    pub fn string() -> Arc<Self> {
        Arc::new(Self {
            namespace: "System".into(),
            name: "String".into(),
            assembly: Some("mscorlib".into()),
            type_def_index: -1,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ManagedType {
    Primitive(PrimKind),
    /// Reference to a class/struct/interface by name.
    Class(Arc<NamedType>),
    Object,
    String,
    SzArray(Box<ManagedType>),
    ByRef(Box<ManagedType>),
    Pointer(Box<ManagedType>),
    IntPtr,
    /// Typed-but-unmodelled (generic instances etc.) — emitted as object.
    Opaque(String),
    Unknown,
}

impl Default for ManagedType {
    fn default() -> Self {
        ManagedType::Unknown
    }
}

impl ManagedType {
    pub fn unknown_is_reference(&self) -> bool {
        matches!(
            self,
            Self::Object
                | Self::String
                | Self::Class(_)
                | Self::SzArray(_)
                | Self::Opaque(_)
                | Self::Unknown
                | Self::ByRef(_)
                | Self::Pointer(_)
        )
    }

    /// Default-value push kind: reference → ldnull, value primitive → ldc.0.
    pub fn is_reference_like(&self) -> bool {
        self.unknown_is_reference()
    }

    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Self::Primitive(PrimKind::R4) | Self::Primitive(PrimKind::R8)
        )
    }

    /// Metadata-free value-type-ness: `Some(true)` for primitives (value types
    /// by CLR definition — including `IntPtr`, the `System.IntPtr` struct) and
    /// `Some(false)` for the kinds that are always references; `None` whenever
    /// only metadata can decide.
    ///
    /// `None` is a refusal, never a guess: `Class` needs its TypeDef bitfield
    /// (`TypeOracle::is_struct_value_type`), and `ByRef`/`Pointer`/`Opaque`/
    /// `Unknown` carry no Layout to judge. The field-address seeding in
    /// `locals::propagate_types` uses this to refuse primitives: an `int` field
    /// IS a value type, but it has no subfields to resolve, so seeding its
    /// address as a struct base would only add `ByRef` noise to the dump —
    /// only `Class` + a metadata-confirmed struct seeds.
    pub fn metadata_free_value_type_ness(&self) -> Option<bool> {
        match self {
            Self::Primitive(p) => Some(p.is_value_type()),
            Self::Object | Self::String | Self::SzArray(_) => Some(false),
            Self::IntPtr => Some(true),
            Self::Class(_)
            | Self::ByRef(_)
            | Self::Pointer(_)
            | Self::Opaque(_)
            | Self::Unknown => None,
        }
    }

    pub fn is_void(&self) -> bool {
        matches!(self, Self::Primitive(PrimKind::Void) | Self::Unknown)
            // Unknown treated as void-ish only for return typing decisions
            && !matches!(self, Self::Opaque(_))
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::Primitive(p) => p.name().to_string(),
            Self::Class(n) => n.to_string(),
            Self::Object => "object".into(),
            Self::String => "string".into(),
            Self::SzArray(e) => format!("{}[]", e.display_name()),
            Self::ByRef(e) => format!("ref {}", e.display_name()),
            Self::Pointer(e) => format!("{}*", e.display_name()),
            Self::IntPtr => "IntPtr".into(),
            Self::Opaque(s) => s.clone(),
            Self::Unknown => "<unknown>".into(),
        }
    }
}

/// Full signature info carried by resolved method refs — the emitter needs
/// param/return ManagedTypes to load arguments and balance the stack.
#[derive(Debug, Clone, PartialEq)]
pub struct MethodSigInfo {
    pub ret: ManagedType,
    pub params: Vec<ManagedType>,
    pub is_static: bool,
}

/// Semantic reference to a managed method (resolved from a call target RVA).
#[derive(Debug, Clone, PartialEq)]
pub struct MethodRef {
    pub image_index: usize,
    pub type_def_index: i32,
    pub method_index: u32,
    pub name: String,
    pub declaring_namespace: String,
    pub declaring_type_name: String,
    /// Assembly simple name of the declaring image (for cross-image externals).
    pub assembly: String,
    pub sig: MethodSigInfo,
}

impl MethodRef {
    pub fn full_name(&self) -> String {
        if self.declaring_namespace.is_empty() {
            format!("{}::{}", self.declaring_type_name, self.name)
        } else {
            format!(
                "{}.{}::{}",
                self.declaring_namespace, self.declaring_type_name, self.name
            )
        }
    }
}

/// Semantic reference to a field, resolved from a `[base + offset]` access.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldRef {
    pub type_def_index: i32,
    pub field_index: u32,
    pub name: String,
    pub declaring_type_name: String,
    pub field_type: ManagedType,
    pub is_static: bool,
}

impl MethodSigInfo {
    /// Void-ish returns need no result store/pop balancing decision.
    pub fn is_void_like(&self) -> bool {
        matches!(
            self.ret,
            ManagedType::Primitive(PrimKind::Void) | ManagedType::Unknown
        )
    }
}
