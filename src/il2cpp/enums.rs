#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Il2CppTypeEnum {
    End = 0x00,
    Void = 0x01,
    Boolean = 0x02,
    Char = 0x03,
    I1 = 0x04,
    U1 = 0x05,
    I2 = 0x06,
    U2 = 0x07,
    I4 = 0x08,
    U4 = 0x09,
    I8 = 0x0A,
    U8 = 0x0B,
    R4 = 0x0C,
    R8 = 0x0D,
    String = 0x0E,
    Ptr = 0x0F,
    ByRef = 0x10,
    ValueType = 0x11,
    Class = 0x12,
    Var = 0x13,
    Array = 0x14,
    GenericInst = 0x15,
    TypedByRef = 0x16,
    I = 0x18,
    U = 0x19,
    FnPtr = 0x1B,
    Object = 0x1C,
    SzArray = 0x1D,
    MVar = 0x1E,
    CModReqd = 0x1F,
    CModOpt = 0x20,
    Internal = 0x21,
    Modifier = 0x40,
    Sentinel = 0x41,
    Pinned = 0x45,
    Enum = 0x55,
    Il2CppTypeIndex = 0xFF,
}

impl Il2CppTypeEnum {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::End),
            0x01 => Some(Self::Void),
            0x02 => Some(Self::Boolean),
            0x03 => Some(Self::Char),
            0x04 => Some(Self::I1),
            0x05 => Some(Self::U1),
            0x06 => Some(Self::I2),
            0x07 => Some(Self::U2),
            0x08 => Some(Self::I4),
            0x09 => Some(Self::U4),
            0x0A => Some(Self::I8),
            0x0B => Some(Self::U8),
            0x0C => Some(Self::R4),
            0x0D => Some(Self::R8),
            0x0E => Some(Self::String),
            0x0F => Some(Self::Ptr),
            0x10 => Some(Self::ByRef),
            0x11 => Some(Self::ValueType),
            0x12 => Some(Self::Class),
            0x13 => Some(Self::Var),
            0x14 => Some(Self::Array),
            0x15 => Some(Self::GenericInst),
            0x16 => Some(Self::TypedByRef),
            0x18 => Some(Self::I),
            0x19 => Some(Self::U),
            0x1B => Some(Self::FnPtr),
            0x1C => Some(Self::Object),
            0x1D => Some(Self::SzArray),
            0x1E => Some(Self::MVar),
            0x1F => Some(Self::CModReqd),
            0x20 => Some(Self::CModOpt),
            0x21 => Some(Self::Internal),
            0x40 => Some(Self::Modifier),
            0x41 => Some(Self::Sentinel),
            0x45 => Some(Self::Pinned),
            0x55 => Some(Self::Enum),
            0xFF => Some(Self::Il2CppTypeIndex),
            _ => None,
        }
    }

    pub fn type_name(&self) -> Option<&'static str> {
        match self {
            Self::Void => Some("void"),
            Self::Boolean => Some("bool"),
            Self::Char => Some("char"),
            Self::I1 => Some("sbyte"),
            Self::U1 => Some("byte"),
            Self::I2 => Some("short"),
            Self::U2 => Some("ushort"),
            Self::I4 => Some("int"),
            Self::U4 => Some("uint"),
            Self::I8 => Some("long"),
            Self::U8 => Some("ulong"),
            Self::R4 => Some("float"),
            Self::R8 => Some("double"),
            Self::String => Some("string"),
            Self::Object => Some("object"),
            Self::TypedByRef => Some("TypedReference"),
            Self::I => Some("IntPtr"),
            Self::U => Some("UIntPtr"),
            _ => None,
        }
    }

    pub fn is_il2cpp_primitive(&self) -> bool {
        matches!(
            self,
            Self::Boolean
                | Self::Char
                | Self::I1
                | Self::U1
                | Self::I2
                | Self::U2
                | Self::I4
                | Self::U4
                | Self::I8
                | Self::U8
                | Self::R4
                | Self::R8
                | Self::I
                | Self::U
        )
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::I1
                | Self::U1
                | Self::I2
                | Self::U2
                | Self::I4
                | Self::U4
                | Self::I8
                | Self::U8
                | Self::R4
                | Self::R8
                | Self::I
                | Self::U
        )
    }

    pub fn is_reference_type(&self) -> bool {
        matches!(
            self,
            Self::String
                | Self::Object
                | Self::Class
                | Self::Array
                | Self::SzArray
                | Self::GenericInst
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Il2CppRGCTXDataType {
    Invalid = 0,
    Type = 1,
    Class = 2,
    Method = 3,
    Array = 4,
    Constrained = 5,
}

impl Il2CppRGCTXDataType {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Invalid),
            1 => Some(Self::Type),
            2 => Some(Self::Class),
            3 => Some(Self::Method),
            4 => Some(Self::Array),
            5 => Some(Self::Constrained),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Il2CppMetadataUsage {
    TypeInfo = 1,
    Il2CppType = 2,
    MethodDef = 3,
    FieldInfo = 4,
    StringLiteral = 5,
    MethodRef = 6,
    FieldRva = 7,
}

impl Il2CppMetadataUsage {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::TypeInfo),
            2 => Some(Self::Il2CppType),
            3 => Some(Self::MethodDef),
            4 => Some(Self::FieldInfo),
            5 => Some(Self::StringLiteral),
            6 => Some(Self::MethodRef),
            7 => Some(Self::FieldRva),
            _ => None,
        }
    }

    pub fn encoded_index_shift(version: f64) -> u32 {
        if version >= 19.0 { 5 } else { 3 }
    }

    pub fn encoded_index_mask(version: f64) -> u32 {
        if version >= 19.0 { 0x1F } else { 0x7 }
    }
}

pub mod type_attributes {
    pub const VISIBILITY_MASK: u32 = 0x00000007;
    pub const NOT_PUBLIC: u32 = 0x00000000;
    pub const PUBLIC: u32 = 0x00000001;
    pub const NESTED_PUBLIC: u32 = 0x00000002;
    pub const NESTED_PRIVATE: u32 = 0x00000003;
    pub const NESTED_FAMILY: u32 = 0x00000004;
    pub const NESTED_ASSEMBLY: u32 = 0x00000005;
    pub const NESTED_FAM_AND_ASSEM: u32 = 0x00000006;
    pub const NESTED_FAM_OR_ASSEM: u32 = 0x00000007;
    pub const LAYOUT_MASK: u32 = 0x00000018;
    pub const AUTO_LAYOUT: u32 = 0x00000000;
    pub const SEQUENTIAL_LAYOUT: u32 = 0x00000008;
    pub const EXPLICIT_LAYOUT: u32 = 0x00000010;
    pub const CLASS_SEMANTICS_MASK: u32 = 0x00000020;
    pub const CLASS: u32 = 0x00000000;
    pub const INTERFACE: u32 = 0x00000020;
    pub const ABSTRACT: u32 = 0x00000080;
    pub const SEALED: u32 = 0x00000100;
    pub const SPECIAL_NAME: u32 = 0x00000400;
    pub const IMPORT: u32 = 0x00001000;
    pub const SERIALIZABLE: u32 = 0x00002000;
    pub const STRING_FORMAT_MASK: u32 = 0x00030000;
    pub const ANSI_CLASS: u32 = 0x00000000;
    pub const UNICODE_CLASS: u32 = 0x00010000;
    pub const AUTO_CLASS: u32 = 0x00020000;
    pub const BEFORE_FIELD_INIT: u32 = 0x00100000;
    pub const FORWARDER: u32 = 0x00200000;
    pub const RT_SPECIAL_NAME: u32 = 0x00000800;
    pub const HAS_SECURITY: u32 = 0x00040000;
}

pub mod field_attributes {
    pub const FIELD_ACCESS_MASK: u32 = 0x0007;
    pub const COMPILER_CONTROLLED: u32 = 0x0000;
    pub const PRIVATE: u32 = 0x0001;
    pub const FAM_AND_ASSEM: u32 = 0x0002;
    pub const ASSEMBLY: u32 = 0x0003;
    pub const FAMILY: u32 = 0x0004;
    pub const FAM_OR_ASSEM: u32 = 0x0005;
    pub const PUBLIC: u32 = 0x0006;
    pub const STATIC: u32 = 0x0010;
    pub const INIT_ONLY: u32 = 0x0020;
    pub const LITERAL: u32 = 0x0040;
    pub const NOT_SERIALIZED: u32 = 0x0080;
    pub const SPECIAL_NAME: u32 = 0x0200;
    pub const PINVOKE_IMPL: u32 = 0x2000;
    pub const RT_SPECIAL_NAME: u32 = 0x0400;
    pub const HAS_FIELD_MARSHAL: u32 = 0x1000;
    pub const HAS_DEFAULT: u32 = 0x8000;
    pub const HAS_FIELD_RVA: u32 = 0x0100;
}

pub mod method_attributes {
    pub const MEMBER_ACCESS_MASK: u32 = 0x0007;
    pub const COMPILER_CONTROLLED: u32 = 0x0000;
    pub const PRIVATE: u32 = 0x0001;
    pub const FAM_AND_ASSEM: u32 = 0x0002;
    pub const ASSEM: u32 = 0x0003;
    pub const FAMILY: u32 = 0x0004;
    pub const FAM_OR_ASSEM: u32 = 0x0005;
    pub const PUBLIC: u32 = 0x0006;
    pub const STATIC: u32 = 0x0010;
    pub const FINAL: u32 = 0x0020;
    pub const VIRTUAL: u32 = 0x0040;
    pub const HIDE_BY_SIG: u32 = 0x0080;
    pub const VTABLE_LAYOUT_MASK: u32 = 0x0100;
    pub const REUSE_SLOT: u32 = 0x0000;
    pub const NEW_SLOT: u32 = 0x0100;
    pub const CHECK_ACCESS_ON_OVERRIDE: u32 = 0x0200;
    pub const ABSTRACT: u32 = 0x0400;
    pub const SPECIAL_NAME: u32 = 0x0800;
    pub const PINVOKE_IMPL: u32 = 0x2000;
    pub const UNMANAGED_EXPORT: u32 = 0x0008;
    pub const RT_SPECIAL_NAME: u32 = 0x1000;
    pub const HAS_SECURITY: u32 = 0x4000;
    pub const REQUIRE_SEC_OBJECT: u32 = 0x8000;
}

pub mod param_attributes {
    pub const IN: u32 = 0x0001;
    pub const OUT: u32 = 0x0002;
    pub const OPTIONAL: u32 = 0x0010;
    pub const HAS_DEFAULT: u32 = 0x1000;
    pub const HAS_FIELD_MARSHAL: u32 = 0x2000;
}
