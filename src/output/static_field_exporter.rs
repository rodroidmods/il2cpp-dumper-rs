//! Thread-static fields, FieldRVA blobs, and PrivateImplementationDetails export.

use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::config::Config;
use crate::error::Result;
use crate::executor::Il2CppExecutor;
use crate::il2cpp::base::Il2Cpp;
use crate::il2cpp::enums::field_attributes;
use crate::il2cpp::field_layout::{
    analyze_field_layout, decode_field_offset, FieldLayoutInfo, StaticFieldKind,
};
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::structures::{Il2CppFieldDefinition, Il2CppType, Il2CppTypeDefinition};
use crate::output::script_json::{ScriptFieldInfo, ScriptJson};

const HEX_CHARS: &[u8; 16] = b"0123456789ABCDEF";

pub fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX_CHARS[(b >> 4) as usize] as char);
        out.push(HEX_CHARS[(b & 0x0F) as usize] as char);
    }
    out
}

pub fn bytes_to_hex_preview(bytes: &[u8], max_bytes: usize) -> String {
    if bytes.len() <= max_bytes {
        return bytes_to_hex(bytes);
    }
    format!("{}...", bytes_to_hex(&bytes[..max_bytes]))
}

pub fn bytes_to_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if bytes.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 63) as usize] as char);
        out.push(TABLE[((triple >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { TABLE[((triple >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[(triple & 63) as usize] as char } else { '=' });
    }
    out
}

pub fn format_hex_dump(bytes: &[u8], bytes_per_line: usize) -> String {
    let bpl = bytes_per_line.max(1);
    let mut out = String::new();
    for (line_idx, chunk) in bytes.chunks(bpl).enumerate() {
        let base = line_idx * bpl;
        let _ = write!(out, "{:04X}: ", base);
        for (i, b) in chunk.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            let _ = write!(out, "{:02X}", b);
        }
        out.push('\n');
    }
    out
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StaticFieldEntry {
    pub declaring_type: String,
    pub field_name: String,
    pub field_path: String,
    pub kind: String,
    pub token: u32,
    pub raw_offset: i32,
    pub effective_offset: i32,
    pub is_thread_static: bool,
    pub has_field_rva: bool,
    pub thread_static_block_size: Option<u32>,
    pub static_block_size: Option<u32>,
    pub metadata_offset: Option<u64>,
    pub metadata_rva: Option<u64>,
    pub binary_va: Option<u64>,
    pub binary_rva: Option<u64>,
    pub data_size: usize,
    pub hex: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ascii_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hex_dump: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StaticFieldSummary {
    pub total_fields_scanned: usize,
    pub thread_static_count: usize,
    pub field_rva_count: usize,
    pub normal_static_count: usize,
    pub types_with_thread_static_block: usize,
    pub private_implementation_details_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StaticMetadataDocument {
    pub il2cpp_version: f64,
    pub pointer_size: u8,
    pub summary: StaticFieldSummary,
    pub fields: Vec<StaticFieldEntry>,
}

fn ascii_preview(bytes: &[u8], max: usize) -> String {
    bytes.iter()
        .take(max)
        .map(|&b| if (0x20..=0x7E).contains(&b) { b as char } else { '.' })
        .collect()
}

fn kind_str(kind: StaticFieldKind) -> &'static str {
    match kind {
        StaticFieldKind::Instance => "Instance",
        StaticFieldKind::NormalStatic => "NormalStatic",
        StaticFieldKind::ThreadStatic => "ThreadStatic",
        StaticFieldKind::FieldRva => "FieldRva",
        StaticFieldKind::Literal => "Literal",
    }
}

pub struct StaticFieldCatalog {
    pub entries: Vec<StaticFieldEntry>,
    pub summary: StaticFieldSummary,
}

impl StaticFieldCatalog {
    pub fn collect(
        executor: &mut Il2CppExecutor,
        metadata: &mut Metadata,
        il2cpp: &mut Il2Cpp,
        config: &Config,
    ) -> Result<Self> {
        let max_rva = config.max_field_rva_dump_bytes;
        let mut entries = Vec::new();
        let mut recorded_tokens = std::collections::HashSet::new();
        let mut thread_static_count = 0usize;
        let mut field_rva_count = 0usize;
        let mut normal_static_count = 0usize;
        let mut types_with_ts_block = 0usize;
        let mut pid_types: Vec<String> = Vec::new();

        let image_defs = metadata.image_defs.clone();
        for image_def in &image_defs {
            let type_end = image_def.type_start as usize + image_def.type_count as usize;
            for type_def_index in image_def.type_start as usize..type_end {
                let type_def = metadata.type_defs[type_def_index].clone();
                let type_name = executor.get_type_def_name(
                    &type_def, type_def_index, metadata, il2cpp, true, true,
                );

                if type_name.contains("PrivateImplementationDetails")
                    || type_name.contains("<PrivateImplementationDetails>")
                {
                    pid_types.push(type_name.clone());
                }

                if let Some(sizes) = il2cpp.type_definition_sizes.get(type_def_index) {
                    if sizes.thread_static_fields_size > 0 {
                        types_with_ts_block += 1;
                    }
                }

                let field_end = type_def.field_start as usize + type_def.field_count as usize;
                for field_index in type_def.field_start as usize..field_end {
                    let field_def = metadata.field_defs[field_index].clone();
                    let field_type = il2cpp.types[field_def.type_index as usize].clone();
                    let field_attrs = field_type.attrs;
                    if (field_attrs & (field_attributes::STATIC | field_attributes::LITERAL | field_attributes::HAS_FIELD_RVA)) == 0 {
                        continue;
                    }

                    let field_name = metadata.get_string_from_index(field_def.name_index)?;
                    let field_index_in_type = field_index - type_def.field_start as usize;

                    let layout = analyze_field_layout(
                        metadata,
                        il2cpp,
                        &type_def,
                        type_def_index,
                        field_index_in_type,
                        field_index,
                        &field_def,
                        &field_type,
                        max_rva,
                    );

                    if !layout.is_static_storage() {
                        continue;
                    }

                    match layout.kind {
                        StaticFieldKind::ThreadStatic => thread_static_count += 1,
                        StaticFieldKind::FieldRva => field_rva_count += 1,
                        StaticFieldKind::NormalStatic => normal_static_count += 1,
                        _ => continue,
                    }

                    if !matches!(
                        layout.kind,
                        StaticFieldKind::ThreadStatic | StaticFieldKind::FieldRva
                    ) {
                        continue;
                    }

                    if !recorded_tokens.insert(field_def.token) {
                        continue;
                    }

                    let (hex, base64, ascii, hex_dump) = if config.dump_field_rva_data
                        && layout.kind == StaticFieldKind::FieldRva
                    {
                        if let Some(bytes) = layout.binary_bytes.as_ref().or(layout.metadata_bytes.as_ref()) {
                            (
                                bytes_to_hex_preview(bytes, 128),
                                Some(bytes_to_base64(bytes)),
                                Some(ascii_preview(bytes, 64)),
                                None,
                            )
                        } else {
                            (String::new(), None, None, None)
                        }
                    } else {
                        (String::new(), None, None, None)
                    };

                    let metadata_rva = layout.default_value_metadata_offset.map(|off| {
                        off.saturating_sub(metadata.header.field_and_parameter_default_value_data_offset as u64)
                    });

                    entries.push(StaticFieldEntry {
                        declaring_type: type_name.clone(),
                        field_name: field_name.clone(),
                        field_path: format!("{}.{}", type_name, field_name),
                        kind: kind_str(layout.kind).to_string(),
                        token: field_def.token,
                        raw_offset: layout.raw_offset,
                        effective_offset: layout.effective_offset,
                        is_thread_static: layout.is_thread_static,
                        has_field_rva: layout.has_field_rva,
                        thread_static_block_size: layout.thread_static_fields_size,
                        static_block_size: layout.static_fields_size,
                        metadata_offset: layout.default_value_metadata_offset,
                        metadata_rva,
                        binary_va: layout.binary_va,
                        binary_rva: layout.binary_rva,
                        data_size: layout.rva_data_size,
                        hex,
                        base64,
                        ascii_preview: ascii,
                        hex_dump,
                    });
                }
            }
        }

        pid_types.sort();
        pid_types.dedup();

        let summary = StaticFieldSummary {
            total_fields_scanned: metadata.field_defs.len(),
            thread_static_count,
            field_rva_count,
            normal_static_count,
            types_with_thread_static_block: types_with_ts_block,
            private_implementation_details_types: pid_types,
        };

        Ok(Self { entries, summary })
    }

    pub fn enrich_script_json(&self, script: &mut ScriptJson, il2cpp: &Il2Cpp) {
        for entry in &self.entries {
            if entry.kind != "ThreadStatic" && entry.kind != "FieldRva" {
                continue;
            }
            let addr = entry.binary_rva
                .or(entry.metadata_rva)
                .unwrap_or(0);
            if addr == 0 && entry.effective_offset < 0 {
                continue;
            }
            script.field_rvas.push(ScriptFieldInfo {
                address: addr,
                name: entry.field_path.clone(),
                value: if entry.hex.is_empty() {
                    format!(
                        "{} offset=0x{:X} kind={}",
                        entry.field_path, entry.effective_offset, entry.kind
                    )
                } else {
                    format!(
                        "{} kind={} hex={}",
                        entry.field_path, entry.kind, entry.hex
                    )
                },
            });
        }

        for entry in &self.entries {
            if !entry.is_thread_static {
                continue;
            }
            script.field_infos.push(ScriptFieldInfo {
                address: entry.effective_offset as u64,
                name: format!("{}_ThreadStatic", entry.field_path.replace('.', "_")),
                value: format!(
                    "ThreadStatic offset=0x{:X} raw=0x{:X}",
                    entry.effective_offset,
                    decode_field_offset(entry.raw_offset).effective
                ),
            });
            let _ = il2cpp;
        }
    }
}

pub struct StaticFieldExporter;

impl StaticFieldExporter {
    pub fn write_document(
        catalog: &StaticFieldCatalog,
        il2cpp: &Il2Cpp,
        output_dir: &str,
    ) -> Result<()> {
        let doc = StaticMetadataDocument {
            il2cpp_version: il2cpp.version,
            pointer_size: if il2cpp.is_32bit { 4 } else { 8 },
            summary: catalog.summary.clone(),
            fields: catalog.entries.clone(),
        };

        let path = Path::new(output_dir).join("static_metadata.json");
        let json = serde_json::to_string_pretty(&doc)
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        fs::write(&path, json)?;
        Ok(())
    }
}

pub fn write_dump_cs_field_annotations(
    buf: &mut String,
    executor: &mut Il2CppExecutor,
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
    config: &Config,
    type_def: &Il2CppTypeDefinition,
    type_def_index: usize,
    field_index: usize,
    field_index_in_type: usize,
    field_def: &Il2CppFieldDefinition,
    field_type: &Il2CppType,
    indent: &str,
) -> Result<()> {
    if !config.dump_static_field_metadata {
        return Ok(());
    }

    let field_attrs = field_type.attrs;
    if (field_attrs & (field_attributes::STATIC | field_attributes::LITERAL | field_attributes::HAS_FIELD_RVA)) == 0 {
        return Ok(());
    }

    let layout = analyze_field_layout(
        metadata,
        il2cpp,
        type_def,
        type_def_index,
        field_index_in_type,
        field_index,
        field_def,
        field_type,
        config.max_field_rva_dump_bytes,
    );

    let type_name = executor.get_type_def_name(
        type_def, type_def_index, metadata, il2cpp, true, true,
    );
    let field_name = metadata.get_string_from_index(field_def.name_index)?;
    let field_path = format!("{type_name}.{field_name}");

    if layout.has_field_rva || layout.kind == StaticFieldKind::FieldRva {
        writeln!(buf, "{indent}\t// FieldRVA: {field_path}").ok();
        if let Some(off) = layout.default_value_metadata_offset {
            writeln!(buf, "{indent}\t//   Metadata offset: 0x{off:X}").ok();
        }
        if let Some(va) = layout.binary_va {
            writeln!(buf, "{indent}\t//   Binary VA: 0x{va:X} (RVA 0x{:X})", layout.binary_rva.unwrap_or(0)).ok();
        }
        if config.dump_field_rva_data {
            if let Some(bytes) = layout.binary_bytes.as_ref().or(layout.metadata_bytes.as_ref()) {
                writeln!(buf, "{indent}\t//   Size: {} bytes", bytes.len()).ok();
                writeln!(buf, "{indent}\t//   Hex: {}", bytes_to_hex_preview(bytes, 128)).ok();
                if bytes.len() <= 64 {
                    writeln!(buf, "{indent}\t//   ASCII: {}", ascii_preview(bytes, 64)).ok();
                }
                for line in format_hex_dump(bytes, 16).lines().take(8) {
                    writeln!(buf, "{indent}\t//   {line}").ok();
                }
            }
        }
    }

    if layout.is_thread_static {
        writeln!(buf, "{indent}\t// ThreadStatic field: {field_path}").ok();
        writeln!(
            buf,
            "{indent}\t//   TLS offset in type block: 0x{:X} (raw IL2CPP offset 0x{:X})",
            layout.effective_offset,
            layout.raw_offset as u32
        ).ok();
        writeln!(
            buf,
            "{indent}\t//   Not a binary RVA — per-thread storage; xref the field name in dump.cs / disasm"
        ).ok();
        if let Some(ts) = layout.thread_static_fields_size {
            if ts > 0 {
                writeln!(buf, "{indent}\t//   ThreadStatic block size: 0x{ts:X}").ok();
            }
        }
        if let Some(ss) = layout.static_fields_size {
            writeln!(buf, "{indent}\t//   Normal static block size: 0x{ss:X}").ok();
        }
    }

    Ok(())
}

pub fn dummy_dll_layout_attrs(
    layout: &FieldLayoutInfo,
    config: &Config,
) -> Vec<(&'static str, String)> {
    let mut attrs = Vec::new();
    if layout.is_thread_static {
        attrs.push(("ThreadStatic", "true".to_string()));
        attrs.push(("Offset", format!("0x{:X}", layout.effective_offset)));
    }
    if layout.has_field_rva {
        if let Some(off) = layout.default_value_metadata_offset {
            attrs.push(("MetadataOffset", format!("0x{off:X}")));
        }
        if config.dump_field_rva_data {
            if let Some(bytes) = layout.binary_bytes.as_ref().or(layout.metadata_bytes.as_ref()) {
                attrs.push(("FieldRvaHex", bytes_to_hex_preview(bytes, 64)));
            }
        }
        if let Some(va) = layout.binary_va {
            attrs.push(("FieldRvaAddress", format!("0x{va:X}")));
        }
    }
    attrs
}
