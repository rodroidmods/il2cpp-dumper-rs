//! Field offset decoding and layout analysis for IL2CPP static / thread-static /
//! FieldRVA (PrivateImplementationDetails) fields.

use crate::il2cpp::base::Il2Cpp;
use crate::il2cpp::enums::{field_attributes, Il2CppTypeEnum};
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::structures::{Il2CppFieldDefinition, Il2CppType, Il2CppTypeDefinition};

pub const THREAD_STATIC_OFFSET_FLAG: u32 = 0x8000_0000;

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

pub fn decode_field_offset(raw: i32) -> DecodedFieldOffset {
    if raw < 0 || (raw as u32) & THREAD_STATIC_OFFSET_FLAG != 0 {
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

pub fn estimate_field_rva_size(
    field_type: &Il2CppType,
    type_def_index: usize,
    il2cpp: &Il2Cpp,
) -> usize {
    if let Some(sizes) = il2cpp.type_definition_sizes.get(type_def_index) {
        if sizes.instance_size > 0 {
            return sizes.instance_size as usize;
        }
        if sizes.native_size > 0 {
            return sizes.native_size as usize;
        }
        if sizes.static_fields_size > 0 {
            return sizes.static_fields_size as usize;
        }
    }

    match Il2CppTypeEnum::from_u8(field_type.type_enum) {
        Some(Il2CppTypeEnum::Boolean) => 1,
        Some(Il2CppTypeEnum::Char) | Some(Il2CppTypeEnum::I2) | Some(Il2CppTypeEnum::U2) => 2,
        Some(Il2CppTypeEnum::I4) | Some(Il2CppTypeEnum::U4) | Some(Il2CppTypeEnum::R4) => 4,
        Some(Il2CppTypeEnum::I8) | Some(Il2CppTypeEnum::U8) | Some(Il2CppTypeEnum::R8) => 8,
        Some(Il2CppTypeEnum::I) | Some(Il2CppTypeEnum::U) => {
            if il2cpp.is_32bit { 4 } else { 8 }
        }
        _ => 64,
    }
}

pub fn read_metadata_bytes(metadata: &Metadata, offset: u64, size: usize) -> Option<Vec<u8>> {
    if size == 0 {
        return Some(Vec::new());
    }
    let data_base = metadata.header.field_and_parameter_default_value_data_offset as u64;
    let data_end = data_base + metadata.header.field_and_parameter_default_value_data_size as u64;
    let end = offset.saturating_add(size as u64);
    if offset < data_base || end > data_end {
        return None;
    }
    let data = metadata.stream.data();
    let start = offset as usize;
    let end = (start + size).min(data.len());
    if start >= data.len() {
        return None;
    }
    Some(data[start..end].to_vec())
}

pub fn read_binary_bytes(il2cpp: &Il2Cpp, va: u64, size: usize) -> Option<Vec<u8>> {
    if size == 0 || va == 0 {
        return None;
    }
    il2cpp.read_bytes_at_rva(il2cpp.get_rva(va), size)
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

    let raw_offset = il2cpp.get_raw_field_offset(
        type_def_index,
        field_index_in_type,
        field_index,
    );
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

    let rva_data_size = estimate_field_rva_size(field_type, type_def_index, il2cpp)
        .min(max_rva_bytes);

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

    let mut binary_va = None;
    let mut binary_rva = None;
    let mut binary_bytes = None;

    if has_field_rva {
        if let Some(meta_off) = default_value_metadata_offset {
            if let Some(blob) = read_metadata_bytes(metadata, meta_off, il2cpp.stream.pointer_size()) {
                let va = if il2cpp.is_32bit {
                    u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as u64
                } else if blob.len() >= 8 {
                    u64::from_le_bytes([
                        blob[0], blob[1], blob[2], blob[3], blob[4], blob[5], blob[6], blob[7],
                    ])
                } else {
                    0
                };
                if va > 0x1000 {
                    binary_va = Some(va);
                    binary_rva = Some(il2cpp.get_rva(va));
                    binary_bytes = read_binary_bytes(il2cpp, va, rva_data_size);
                }
            }
        }
    }

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
