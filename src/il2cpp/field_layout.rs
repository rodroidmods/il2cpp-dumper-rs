//! Field offset decoding and layout analysis for IL2CPP static / thread-static /
//! FieldRVA (PrivateImplementationDetails) fields.

use crate::il2cpp::base::Il2Cpp;
use crate::il2cpp::enums::{Il2CppTypeEnum, field_attributes};
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::structures::{Il2CppFieldDefinition, Il2CppType, Il2CppTypeDefinition};

pub const THREAD_STATIC_OFFSET_FLAG: u32 = 0x8000_0000;

/// IL2CPP stores a field offset of `-1` meaning "this field has no static /
/// instance offset recorded" (a null `fieldOffsets` entry, or a read that fell
/// outside the binary). It is *not* a thread-static encoding.
pub const NO_FIELD_OFFSET: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticFieldKind {
    Instance,
    NormalStatic,
    ThreadStatic,
    FieldRva,
    Literal,
}

#[derive(Debug, Clone)]
pub struct DecodedFieldOffset {
    pub raw: i32,
    pub effective: i32,
    pub is_thread_static: bool,
}

#[derive(Debug, Clone)]
pub struct FieldLayoutInfo {
    pub kind: StaticFieldKind,
    pub raw_offset: i32,
    pub effective_offset: i32,
    pub is_thread_static: bool,
    pub has_field_rva: bool,
    pub thread_static_fields_size: Option<u32>,
    pub static_fields_size: Option<u32>,
    pub default_value_metadata_offset: Option<u64>,
    pub rva_data_size: usize,
    pub metadata_bytes: Option<Vec<u8>>,
    pub binary_va: Option<u64>,
    pub binary_rva: Option<u64>,
    pub binary_bytes: Option<Vec<u8>>,
}

impl FieldLayoutInfo {
    pub fn is_static_storage(&self) -> bool {
        matches!(
            self.kind,
            StaticFieldKind::NormalStatic
                | StaticFieldKind::ThreadStatic
                | StaticFieldKind::FieldRva
        )
    }
}

/// Decode a raw field offset from the `fieldOffsets` table.
///
/// Thread-static fields are encoded by ORing the offset with `0x8000_0000`, so
/// `0x80000000` is the *first* field of the thread-static block (offset 0) and
/// `0x80000008` the second at 8. Note that `-1` (`0xFFFFFFFF`) also has the top
/// bit set but is the "no offset" sentinel — masking it would turn it into the
/// bogus offset `0x7FFFFFFF`, so it is excluded explicitly.
pub fn decode_field_offset(raw: i32) -> DecodedFieldOffset {
    let flagged = (raw as u32) & THREAD_STATIC_OFFSET_FLAG != 0;
    if raw != NO_FIELD_OFFSET && flagged {
        let effective = (raw as u32) & !THREAD_STATIC_OFFSET_FLAG;
        DecodedFieldOffset {
            raw,
            effective: effective as i32,
            is_thread_static: true,
        }
    } else {
        DecodedFieldOffset {
            raw,
            effective: raw,
            is_thread_static: false,
        }
    }
}

/// True for the type kinds whose FieldRVA blob size we can determine: a plain
/// value/class type, not an array, pointer or generic instance. Cpp2IL applies
/// the same filter — arrays and pointers have no static blob.
pub fn has_static_blob_type(field_type: &Il2CppType) -> bool {
    matches!(
        Il2CppTypeEnum::from_u8(field_type.type_enum),
        Some(Il2CppTypeEnum::Class) | Some(Il2CppTypeEnum::ValueType)
    )
}

/// The declaration size of a `PrivateImplementationDetails` blob is baked into
/// the generated struct's own name — `<PrivateImplementationDetails>.__StaticArrayInitTypeSize=108`.
/// That number is authoritative: `Il2CppTypeDefinitionSizes.native_size` is `-1`
/// (or wrong) on several IL2CPP versions, which is exactly why Cpp2IL prefers the
/// name over the sizes table.
pub fn static_array_init_size(
    field_type: &Il2CppType,
    _il2cpp: &Il2Cpp,
    metadata: &Metadata,
) -> Option<usize> {
    let type_def = metadata.type_defs.get(field_type.datapoint as usize)?;
    let name = metadata.get_string_from_index(type_def.name_index).ok()?;
    parse_static_array_init_size(&name)
}

/// Parse the `N` out of `__StaticArrayInitTypeSize=N`.
pub fn parse_static_array_init_size(name: &str) -> Option<usize> {
    let n = name.strip_prefix(STATIC_ARRAY_INIT_PREFIX)?;
    n.parse::<usize>().ok().filter(|n| *n > 0)
}

pub const STATIC_ARRAY_INIT_PREFIX: &str = "__StaticArrayInitTypeSize=";

/// Size, in bytes, of the initializer stored in the metadata's
/// `fieldAndParameterDefaultValueData` blob for a `hasFieldRVA` field.
///
/// Returns 0 when the size cannot be established, in which case no bytes are
/// reported (an invented default would be worse than a visible gap).
pub fn estimate_field_rva_size(
    field_type: &Il2CppType,
    il2cpp: &Il2Cpp,
    metadata: &Metadata,
) -> usize {
    if !has_static_blob_type(field_type) {
        return 0;
    }

    // `__StaticArrayInitTypeSize=N` wins over the sizes table.
    if let Some(n) = static_array_init_size(field_type, il2cpp, metadata) {
        return n;
    }

    // Otherwise the blob is exactly the field's own type, whose size lives in the
    // field type's *own* type-definition-sizes entry (not the declaring type's).
    if let Some(sizes) = il2cpp
        .type_definition_sizes
        .get(field_type.datapoint as usize)
    {
        if sizes.instance_size > 0 {
            return sizes.instance_size as usize;
        }
        if sizes.native_size > 0 {
            return sizes.native_size as usize;
        }
    }

    primitive_size(field_type, il2cpp).unwrap_or(0)
}

fn primitive_size(field_type: &Il2CppType, il2cpp: &Il2Cpp) -> Option<usize> {
    match Il2CppTypeEnum::from_u8(field_type.type_enum)? {
        Il2CppTypeEnum::Boolean | Il2CppTypeEnum::I1 | Il2CppTypeEnum::U1 => Some(1),
        Il2CppTypeEnum::Char | Il2CppTypeEnum::I2 | Il2CppTypeEnum::U2 => Some(2),
        Il2CppTypeEnum::I4 | Il2CppTypeEnum::U4 | Il2CppTypeEnum::R4 => Some(4),
        Il2CppTypeEnum::I8 | Il2CppTypeEnum::U8 | Il2CppTypeEnum::R8 => Some(8),
        Il2CppTypeEnum::I | Il2CppTypeEnum::U => Some(if il2cpp.is_32bit { 4 } else { 8 }),
        _ => None,
    }
}

/// Compute static field offsets for a type from type sizes/alignments.
///
/// Used as a fallback when the binary's field offset data is zeroed out
/// (common in Unity 2022+ / IL2CPP v29+ where field offsets are runtime-initialized).
/// Returns a vec indexed by field_index_in_type for STATIC fields only,
/// or None if the type has no static fields or sizes are unavailable.
pub fn compute_static_field_offsets(
    td: &Il2CppTypeDefinition,
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
) -> Option<Vec<u32>> {
    let field_start = td.field_start as usize;
    let field_count = td.field_count as usize;
    let ptr_size: u32 = if il2cpp.is_32bit { 4 } else { 8 };

    let mut offsets = Vec::with_capacity(field_count);
    let mut current_offset: u32 = 0;
    let mut has_any_static = false;

    for fi in 0..field_count {
        let global_fi = field_start + fi;
        let Some(field_def) = metadata.field_defs.get(global_fi) else {
            offsets.push(0);
            continue;
        };
        let Some(field_type) = il2cpp.types.get(field_def.type_index as usize) else {
            offsets.push(0);
            continue;
        };

        let is_static = (field_type.attrs & field_attributes::STATIC) != 0;
        let is_literal = (field_type.attrs & field_attributes::LITERAL) != 0;
        let has_field_rva = (field_type.attrs & field_attributes::HAS_FIELD_RVA) != 0;

        if !is_static || is_literal || has_field_rva {
            offsets.push(0);
            continue;
        }

        has_any_static = true;
        let (size, align) = static_field_size_and_align(field_type, il2cpp, metadata, ptr_size);
        if align > 0 {
            let align_mask = align - 1;
            current_offset = (current_offset + align_mask) & !align_mask;
        }
        offsets.push(current_offset);
        current_offset += size;
    }

    if !has_any_static {
        return None;
    }
    Some(offsets)
}

fn static_field_size_and_align(
    field_type: &Il2CppType,
    il2cpp: &Il2Cpp,
    metadata: &Metadata,
    ptr_size: u32,
) -> (u32, u32) {
    if let Some(prim) = primitive_size(field_type, il2cpp) {
        return (prim as u32, prim as u32);
    }
    match Il2CppTypeEnum::from_u8(field_type.type_enum) {
        Some(Il2CppTypeEnum::String)
        | Some(Il2CppTypeEnum::Object)
        | Some(Il2CppTypeEnum::Class)
        | Some(Il2CppTypeEnum::Array)
        | Some(Il2CppTypeEnum::SzArray)
        | Some(Il2CppTypeEnum::Ptr)
        | Some(Il2CppTypeEnum::ByRef) => (ptr_size, ptr_size),
        Some(Il2CppTypeEnum::ValueType) | Some(Il2CppTypeEnum::Enum) => {
            let klass_idx = field_type.datapoint as usize;
            if let Some(sizes) = il2cpp.type_definition_sizes.get(klass_idx) {
                let size = sizes.instance_size;
                if size > 0 {
                    let align = if Il2CppTypeEnum::from_u8(field_type.type_enum)
                        == Some(Il2CppTypeEnum::Enum)
                    {
                        size.min(8)
                    } else {
                        value_type_alignment(size, metadata, il2cpp, klass_idx)
                    };
                    return (size, align);
                }
            }
            (ptr_size, ptr_size)
        }
        Some(Il2CppTypeEnum::GenericInst) => {
            if field_type.valuetype != 0 {
                let klass_idx = field_type.datapoint as usize;
                if let Some(sizes) = il2cpp.type_definition_sizes.get(klass_idx) {
                    let size = sizes.instance_size;
                    if size > 0 {
                        return (size, value_type_alignment(size, metadata, il2cpp, klass_idx));
                    }
                }
            }
            (ptr_size, ptr_size)
        }
        _ => (ptr_size, ptr_size),
    }
}

fn value_type_alignment(
    size: u32,
    _metadata: &Metadata,
    _il2cpp: &Il2Cpp,
    _klass_idx: usize,
) -> u32 {
    if size >= 8 { 8 } else if size >= 4 { 4 } else if size >= 2 { 2 } else { 1 }
}

/// Read `size` bytes out of the metadata's `fieldAndParameterDefaultValueData`
/// section. Returns `None` unless the *whole* requested range is inside both the
/// declared section and the file — silently returning a short slice would report
/// `data_size` for bytes that are not there.
pub fn read_metadata_bytes(metadata: &Metadata, offset: u64, size: usize) -> Option<Vec<u8>> {
    if size == 0 {
        return Some(Vec::new());
    }
    let data_base = metadata
        .header
        .field_and_parameter_default_value_data_offset as u64;
    let data_end = data_base.saturating_add(
        metadata
            .header
            .field_and_parameter_default_value_data_size
            .max(0) as u64,
    );
    let end = offset.checked_add(size as u64)?;
    if offset < data_base || end > data_end {
        return None;
    }
    let data = metadata.stream.data();
    let (start, end) = (offset as usize, end as usize);
    if end > data.len() {
        return None;
    }
    Some(data[start..end].to_vec())
}

pub fn analyze_field_layout(
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
    _type_def: &Il2CppTypeDefinition,
    type_def_index: usize,
    field_index_in_type: usize,
    field_index: usize,
    _field_def: &Il2CppFieldDefinition,
    field_type: &Il2CppType,
    max_rva_bytes: usize,
) -> FieldLayoutInfo {
    let is_literal = (field_type.attrs & field_attributes::LITERAL) != 0;
    let is_static = (field_type.attrs & field_attributes::STATIC) != 0;
    let has_field_rva = (field_type.attrs & field_attributes::HAS_FIELD_RVA) != 0;

    let raw_offset = il2cpp.get_raw_field_offset(type_def_index, field_index_in_type, field_index);
    let decoded = decode_field_offset(raw_offset);

    let type_sizes = il2cpp.type_definition_sizes.get(type_def_index);
    let thread_static_fields_size = type_sizes.map(|s| s.thread_static_fields_size);
    let static_fields_size = type_sizes.map(|s| s.static_fields_size);

    let mut kind = if is_literal {
        StaticFieldKind::Literal
    } else if decoded.is_thread_static {
        StaticFieldKind::ThreadStatic
    } else if has_field_rva {
        StaticFieldKind::FieldRva
    } else if is_static {
        StaticFieldKind::NormalStatic
    } else {
        StaticFieldKind::Instance
    };

    // A `hasFieldRVA` field's initializer lives *inline* in the metadata's
    // `fieldAndParameterDefaultValueData` blob, packed back-to-back with the other
    // default values (the declared `__StaticArrayInitTypeSize=N` matches the offset
    // delta to the next entry). It is not a pointer to a copy in the binary.
    let rva_data_size = if has_field_rva {
        estimate_field_rva_size(field_type, il2cpp, metadata).min(max_rva_bytes)
    } else if is_literal {
        primitive_size(field_type, il2cpp)
            .unwrap_or(0)
            .min(max_rva_bytes)
    } else {
        0
    };

    let mut default_value_metadata_offset = None;
    let mut metadata_bytes = None;

    if let Some(fdv) = metadata.get_field_default_value(field_index as i32) {
        if fdv.data_index >= 0 {
            let meta_off = metadata.get_default_value_offset(fdv.data_index);
            default_value_metadata_offset = Some(meta_off);
            if has_field_rva || is_literal {
                metadata_bytes = read_metadata_bytes(metadata, meta_off, rva_data_size);
            }
        }
    }

    if !decoded.is_thread_static
        && is_static
        && !is_literal
        && !has_field_rva
        && thread_static_fields_size.unwrap_or(0) > 0
    {
        if let (Some(ts_size), Some(ss_size)) = (thread_static_fields_size, static_fields_size) {
            let eff = decoded.effective as u32;
            if eff >= ss_size && eff < ss_size.saturating_add(ts_size) {
                kind = StaticFieldKind::ThreadStatic;
            }
        }
    }

    // FieldRVA blobs are metadata bytes; there is no binary address to report.
    let binary_va = None;
    let binary_rva = None;
    let binary_bytes = None;

    FieldLayoutInfo {
        kind,
        raw_offset,
        effective_offset: decoded.effective,
        is_thread_static: matches!(kind, StaticFieldKind::ThreadStatic),
        has_field_rva,
        thread_static_fields_size,
        static_fields_size,
        default_value_metadata_offset,
        rva_data_size,
        metadata_bytes,
        binary_va,
        binary_rva,
        binary_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_static_flag_marks_the_field_and_strips_the_bit() {
        // `0x80000000` is the *first* field of the TLS block, offset 0 — not "no offset".
        let first = decode_field_offset(0x8000_0000u32 as i32);
        assert!(first.is_thread_static);
        assert_eq!(first.effective, 0);

        let second = decode_field_offset(0x8000_0008u32 as i32);
        assert!(second.is_thread_static);
        assert_eq!(second.effective, 8);
    }

    #[test]
    fn the_minus_one_sentinel_is_not_a_thread_static_offset() {
        // `-1` has the top bit set but means "no offset recorded". Masking the bit
        // off would invent the offset 0x7FFFFFFF.
        let none = decode_field_offset(NO_FIELD_OFFSET);
        assert!(!none.is_thread_static);
        assert_eq!(none.effective, NO_FIELD_OFFSET);
    }

    #[test]
    fn ordinary_offsets_are_left_alone() {
        let decoded = decode_field_offset(0x40);
        assert!(!decoded.is_thread_static);
        assert_eq!(decoded.effective, 0x40);
    }

    #[test]
    fn static_array_init_size_comes_from_the_type_name() {
        assert_eq!(
            parse_static_array_init_size("__StaticArrayInitTypeSize=108"),
            Some(108)
        );
        assert_eq!(
            parse_static_array_init_size("__StaticArrayInitTypeSize=1"),
            Some(1)
        );
        // A type that is not a blob, a blob without a length, and a zero-length
        // blob (which would make the byte count meaningless) all yield nothing.
        assert_eq!(parse_static_array_init_size("Vector3"), None);
        assert_eq!(
            parse_static_array_init_size("__StaticArrayInitTypeSize="),
            None
        );
        assert_eq!(
            parse_static_array_init_size("__StaticArrayInitTypeSize=0"),
            None
        );
        assert_eq!(
            parse_static_array_init_size("__StaticArrayInitTypeSize=abc"),
            None
        );
    }
}
