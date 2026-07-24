use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Mutex;
use crate::io::{checked_array_len, signed_size_to_u64, BinaryStream};
use crate::error::{Error, Result};
use super::structures::*;

pub const METADATA_MAGIC: u32 = 0xFAB11BAF;

fn map_by_key_dedup<T, K>(
    items: impl IntoIterator<Item = T>,
    key_fn: impl Fn(&T) -> K,
    label: &str,
) -> HashMap<K, T>
where
    K: Eq + Hash + Copy + Display,
{
    let mut map = HashMap::new();
    let mut dup_count = 0u32;
    let mut first_dup: Option<K> = None;
    for item in items {
        let key = key_fn(&item);
        if map.insert(key, item).is_some() {
            dup_count += 1;
            if first_dup.is_none() {
                first_dup = Some(key);
            }
        }
    }
    if dup_count > 0 {
        eprintln!(
            "Warning: {label}: {dup_count} duplicate key(s) (e.g. {}); keeping last entry",
            first_dup.map(|k| k.to_string()).unwrap_or_default()
        );
    }
    map
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataVariant {
    Standard,
    Codm,
}

fn detect_codm_variant(stream: &mut BinaryStream, file_size: u64) -> bool {
    const PAIR_COUNT: usize = 33;
    const HEADER_BYTES: u64 = 8 + (PAIR_COUNT as u64) * 8;

    let saved = stream.position();
    if file_size < HEADER_BYTES {
        return false;
    }

    stream.set_position(8);
    let mut pairs: Vec<(i32, i32)> = Vec::with_capacity(PAIR_COUNT);
    for _ in 0..PAIR_COUNT {
        let off = match stream.read_i32() {
            Ok(v) => v,
            Err(_) => {
                stream.set_position(saved);
                return false;
            }
        };
        let size = match stream.read_i32() {
            Ok(v) => v,
            Err(_) => {
                stream.set_position(saved);
                return false;
            }
        };
        pairs.push((off, size));
    }
    stream.set_position(saved);

    let mut any_nonzero = false;
    for &(off, size) in &pairs {
        if off < 0 || size < 0 {
            return false;
        }
        let end = (off as u64).saturating_add(size as u64);
        if end > file_size {
            return false;
        }
        if size > 0 {
            any_nonzero = true;
        }
    }
    if !any_nonzero {
        return false;
    }

    pairs.iter().any(|&(_, size)| {
        size > 0 && size % 80 == 0 && size >= 80 * 16 && (size as u64) < file_size
    })
}

pub struct Metadata {
    pub stream: BinaryStream,
    pub version: f64,
    pub variant: MetadataVariant,
    pub header: Il2CppGlobalMetadataHeader,
    pub unity_version: Option<UnityVersion>,

    pub image_defs: Vec<Il2CppImageDefinition>,
    pub assembly_defs: Vec<Il2CppAssemblyDefinition>,
    pub type_defs: Vec<Il2CppTypeDefinition>,
    pub method_defs: Vec<Il2CppMethodDefinition>,
    pub parameter_defs: Vec<Il2CppParameterDefinition>,
    pub field_defs: Vec<Il2CppFieldDefinition>,
    pub property_defs: Vec<Il2CppPropertyDefinition>,
    pub event_defs: Vec<Il2CppEventDefinition>,
    pub generic_containers: Vec<Il2CppGenericContainer>,
    pub generic_parameters: Vec<Il2CppGenericParameter>,
    pub string_literals: Vec<Il2CppStringLiteral>,
    pub field_refs: Vec<Il2CppFieldRef>,

    pub interface_indices: Vec<i32>,
    pub interface_offsets: Vec<Il2CppInterfaceOffset>,
    pub nested_type_indices: Vec<i32>,
    pub constraint_indices: Vec<i32>,
    pub vtable_methods: Vec<u32>,
    pub type_inline_arrays: Vec<Il2CppInlineArrayLength>,
    pub referenced_assemblies: Vec<i32>,
    pub type_definition_sizes: Vec<Il2CppTypeDefinitionSizes>,

    pub attribute_type_ranges: Vec<Il2CppCustomAttributeTypeRange>,
    pub attribute_types: Vec<i32>,
    pub attribute_data_ranges: Vec<Il2CppCustomAttributeDataRange>,

    pub metadata_usage_dic: HashMap<u32, HashMap<u32, u32>>,
    pub metadata_usages_count: usize,

    field_default_values: HashMap<i32, Il2CppFieldDefaultValue>,
    param_default_values: HashMap<i32, Il2CppParameterDefaultValue>,
    attribute_type_ranges_dic: HashMap<usize, HashMap<u32, usize>>,
    string_cache: Mutex<HashMap<i32, String>>,
    pub rgctx_entries: Vec<Il2CppRGCTXDefinition>,

    pub type_def_offset_to_index: HashMap<u64, usize>,
    pub generic_param_offset_to_index: HashMap<u64, usize>,
}

impl Metadata {
    pub fn new(data: Vec<u8>) -> Result<Self> {
        Self::new_with_options(data, None, false)
    }

    pub fn new_with_unity_version(data: Vec<u8>, unity_version_str: Option<&str>) -> Result<Self> {
        Self::new_with_options(data, unity_version_str, false)
    }

    pub fn new_with_options(
        data: Vec<u8>,
        unity_version_str: Option<&str>,
        force_codm: bool,
    ) -> Result<Self> {
        let file_len = data.len() as u64;
        if file_len < 8 {
            return Err(Error::InvalidMetadata(format!(
                "File too small ({file_len} bytes). Expected global-metadata.dat (usually multi‑MB). \
                 Check the path or re-extract from the APK/IPA."
            )));
        }

        let mut stream = BinaryStream::new(data);
        stream.set_position(0);

        let sanity = stream
            .read_u32()
            .map_err(|e| e.with_metadata_context(file_len))?;
        if sanity != METADATA_MAGIC {
            let b0 = (sanity & 0xFF) as u8;
            let b1 = ((sanity >> 8) & 0xFF) as u8;
            let b2 = ((sanity >> 16) & 0xFF) as u8;
            let b3 = ((sanity >> 24) & 0xFF) as u8;
            return Err(Error::InvalidMetadata(format!(
                "Wrong magic number 0x{sanity:08X} (bytes {b0:02X} {b1:02X} {b2:02X} {b3:02X}). \
                 Expected AF 1B B1 FA. This usually means the file is encrypted, obfuscated, \
                 truncated, or not global-metadata.dat. Protected games often require dumping \
                 decrypted metadata from memory at runtime."
            )));
        }

        let version_raw = stream
            .read_i32()
            .map_err(|e| e.with_metadata_context(file_len))?;
        if version_raw < 0 || version_raw > 200 {
            return Err(Error::InvalidMetadata(format!(
                "Invalid metadata version field ({version_raw}). \
                 Header looks corrupted — often encryption or a truncated file. \
                 Valid files start with AF 1B B1 FA followed by a small version (16–106)."
            )));
        }
        if version_raw < 16 {
            return Err(Error::UnsupportedVersion(version_raw));
        }
        let known_versions = [16, 17, 19, 20, 21, 22, 24, 27, 29, 31, 33, 35, 38, 39, 104, 105, 106];
        if !known_versions.contains(&version_raw) {
            eprintln!("Warning: Unknown metadata version {version_raw}. Attempting anyway, results may be incorrect.");
        }
        if version_raw >= 38 {
            eprintln!("Info: Unity 6 metadata (v{version_raw}) detected. Using variable-width index mode.");
        }

        let variant = if force_codm {
            if version_raw != 23 {
                eprintln!(
                    "Warning: --codm forced but metadata version is {version_raw} (CODM uses v23). Continuing with CODM layout anyway."
                );
            } else {
                eprintln!("Info: CODM metadata variant forced via config/CLI.");
            }
            MetadataVariant::Codm
        } else if version_raw == 23 {
            let file_size = stream.len();
            if detect_codm_variant(&mut stream, file_size) {
                eprintln!("Info: CODM metadata variant auto-detected (custom v23 schema).");
                MetadataVariant::Codm
            } else {
                MetadataVariant::Standard
            }
        } else {
            MetadataVariant::Standard
        };

        let unity_version = unity_version_str.and_then(UnityVersion::parse);

        let version = if let Some(ref uv) = unity_version {
            let resolved = uv.resolve_sub_version(version_raw);
            if resolved != version_raw as f64 {
                eprintln!("Info: Unity {uv} detected. Resolved version {version_raw} -> {resolved}");
            }
            resolved
        } else {
            version_raw as f64
        };

        stream.set_position(0);
        let header = match variant {
            MetadataVariant::Codm => Il2CppGlobalMetadataHeader::read_codm(&mut stream)
                .map_err(|e| e.with_metadata_context(file_len))?,
            MetadataVariant::Standard => Il2CppGlobalMetadataHeader::read(&mut stream, version)
                .map_err(|e| e.with_metadata_context(file_len))?,
        };

        Self::validate_header_ranges(&header, file_len)?;

        let mut meta = Self {
            stream,
            version,
            variant,
            header,
            unity_version,
            image_defs: Vec::new(),
            assembly_defs: Vec::new(),
            type_defs: Vec::new(),
            method_defs: Vec::new(),
            parameter_defs: Vec::new(),
            field_defs: Vec::new(),
            property_defs: Vec::new(),
            event_defs: Vec::new(),
            generic_containers: Vec::new(),
            generic_parameters: Vec::new(),
            string_literals: Vec::new(),
            field_refs: Vec::new(),
            interface_indices: Vec::new(),
            interface_offsets: Vec::new(),
            nested_type_indices: Vec::new(),
            constraint_indices: Vec::new(),
            vtable_methods: Vec::new(),
            type_inline_arrays: Vec::new(),
            referenced_assemblies: Vec::new(),
            type_definition_sizes: Vec::new(),
            attribute_type_ranges: Vec::new(),
            attribute_types: Vec::new(),
            attribute_data_ranges: Vec::new(),
            metadata_usage_dic: HashMap::new(),
            metadata_usages_count: 0,
            field_default_values: HashMap::new(),
            param_default_values: HashMap::new(),
            attribute_type_ranges_dic: HashMap::new(),
            string_cache: Mutex::new(HashMap::new()),
            rgctx_entries: Vec::new(),
            type_def_offset_to_index: HashMap::new(),
            generic_param_offset_to_index: HashMap::new(),
        };

        if meta.unity_version.is_none() && meta.variant != MetadataVariant::Codm {
            meta.detect_subversion()
                .map_err(|e| e.with_metadata_context(file_len))?;
        }
        let widths = IndexWidths::from_header(&meta.header, meta.version);
        set_index_widths(widths);
        meta.load_metadata()
            .map_err(|e| e.with_metadata_context(file_len))?;
        Ok(meta)
    }

    /// Catch encrypted/garbage headers early: section offset+size must fit in the file.
    fn validate_header_ranges(header: &Il2CppGlobalMetadataHeader, file_len: u64) -> Result<()> {
        let sections: [(&str, i32, i32); 10] = [
            ("string_literal", header.string_literal_offset, header.string_literal_size),
            ("string", header.string_offset, header.string_size),
            ("type_definitions", header.type_definitions_offset, header.type_definitions_size),
            ("methods", header.methods_offset, header.methods_size),
            ("fields", header.fields_offset, header.fields_size),
            ("parameters", header.parameters_offset, header.parameters_size),
            ("images", header.images_offset, header.images_size),
            ("assemblies", header.assemblies_offset, header.assemblies_size),
            ("properties", header.properties_offset, header.properties_size),
            ("events", header.events_offset, header.events_size),
        ];

        let mut bad = Vec::new();
        for (name, off, size) in sections {
            if off < 0 || size < 0 {
                bad.push(format!("{name}: negative offset/size (off={off}, size={size})"));
                continue;
            }
            if size == 0 {
                continue;
            }
            let off_u = off as u64;
            let size_u = size as u64;
            if off_u >= file_len {
                bad.push(format!(
                    "{name}: offset 0x{off_u:X} is past end of file ({file_len} bytes)"
                ));
            } else if off_u.saturating_add(size_u) > file_len {
                bad.push(format!(
                    "{name}: range 0x{off_u:X}+{size_u} exceeds file size {file_len}"
                ));
            } else if size_u > file_len {
                bad.push(format!(
                    "{name}: size {size_u} larger than entire file ({file_len})"
                ));
            }
        }

        if !bad.is_empty() {
            let detail = bad.join("; ");
            return Err(Error::InvalidMetadata(format!(
                "Metadata header section ranges are invalid ({detail}). \
                 This usually means encrypted/obfuscated global-metadata.dat or a truncated extract. \
                 Valid files start with AF 1B B1 FA. Protected games often need a runtime memory dump."
            )));
        }
        Ok(())
    }

    fn detect_subversion(&mut self) -> Result<()> {
        if self.version == 24.0 {
            if self.header.string_literal_offset == 264 {
                self.version = 24.2;
                self.stream.set_position(0);
                self.header = Il2CppGlobalMetadataHeader::read(&mut self.stream, self.version)?;
            } else {
                let images = self.read_metadata_array::<Il2CppImageDefinition>(
                    signed_size_to_u64(self.header.images_offset, "images_offset")?,
                    signed_size_to_u64(self.header.images_size, "images_size")?,
                    None,
                )?;
                if images.iter().any(|img| img.token != 1) {
                    self.version = 24.1;
                }
            }

            if self.version != 24.0 {
                self.stream.set_position(0);
                self.header = Il2CppGlobalMetadataHeader::read(&mut self.stream, self.version)?;
            }
        }
        Ok(())
    }

    fn load_metadata(&mut self) -> Result<()> {
        if self.variant == MetadataVariant::Codm {
            return self.load_metadata_codm();
        }
        let h = self.header.clone();
        let v38 = self.version >= 38.0;
        let _widths = IndexWidths::from_header(&h, self.version);

        macro_rules! load {
            ($t:ty, $off:expr, $size:expr, $count:expr) => {{
                let off = signed_size_to_u64($off, stringify!($off))?;
                let size = signed_size_to_u64($size, stringify!($size))?;
                let cnt = if v38 && $count > 0 { Some($count as usize) } else { None };
                self.read_metadata_array::<$t>(off, size, cnt)?
            }};
        }

        self.image_defs = load!(Il2CppImageDefinition, h.images_offset, h.images_size, 0);

        if self.version == 24.2 {
            let assembly_element_size = 68u64;
            if h.assemblies_size as u64 / assembly_element_size < self.image_defs.len() as u64 {
                self.version = 24.4;
            }
        }

        self.assembly_defs = load!(Il2CppAssemblyDefinition, h.assemblies_offset, h.assemblies_size, 0);

        if v38 {
            let (type_defs, td_offsets) = self.read_metadata_array_with_offsets::<Il2CppTypeDefinition>(
                signed_size_to_u64(h.type_definitions_offset, "type_definitions_offset")?,
                signed_size_to_u64(h.type_definitions_size, "type_definitions_size")?,
                if h.type_definitions_count > 0 { Some(h.type_definitions_count as usize) } else { None },
            )?;
            self.type_defs = type_defs;
            self.type_def_offset_to_index = td_offsets;
        } else {
            self.type_defs = load!(Il2CppTypeDefinition, h.type_definitions_offset, h.type_definitions_size, h.type_definitions_count);
        }

        self.method_defs = load!(Il2CppMethodDefinition, h.methods_offset, h.methods_size, h.methods_count);
        self.parameter_defs = load!(Il2CppParameterDefinition, h.parameters_offset, h.parameters_size, h.parameters_count);
        self.field_defs = load!(Il2CppFieldDefinition, h.fields_offset, h.fields_size, h.fields_count);

        let mut field_defaults_vec = load!(Il2CppFieldDefaultValue, h.field_default_values_offset, h.field_default_values_size, 0);
        field_defaults_vec.retain(|v| v.field_index >= 0);
        if self.version >= 104.0 && !field_defaults_vec.is_empty() {
            if let Some(last) = field_defaults_vec.last() {
                if last.field_index == -1 {
                    field_defaults_vec.pop();
                }
            }
        }
        self.field_default_values = map_by_key_dedup(
            field_defaults_vec,
            |v| v.field_index,
            "field_default_values",
        );

        let mut param_defaults = load!(Il2CppParameterDefaultValue, h.parameter_default_values_offset, h.parameter_default_values_size, 0);
        param_defaults.retain(|v| v.parameter_index >= 0);
        self.param_default_values = map_by_key_dedup(
            param_defaults,
            |v| v.parameter_index,
            "parameter_default_values",
        );

        self.property_defs = load!(Il2CppPropertyDefinition, h.properties_offset, h.properties_size, h.properties_count);

        self.interface_indices = {
            let type_idx_sz = IndexWidths::get_type_index_size(&h).max(1);
            let size = signed_size_to_u64(h.interfaces_size, "interfaces_size")?;
            let off = signed_size_to_u64(h.interfaces_offset, "interfaces_offset")?;
            let count = checked_array_len(
                size / type_idx_sz as u64,
                type_idx_sz,
                Some(self.stream.remaining_from(off)),
                "interfaces",
            )?;
            let mut out = Vec::with_capacity(count);
            self.stream.set_position(off);
            for _ in 0..count {
                out.push(self.stream.read_variable_index(type_idx_sz as u8)?);
            }
            out
        };

        self.interface_offsets = {
            let type_idx_sz = IndexWidths::get_type_index_size(&h).max(1);
            let each_sz = type_idx_sz + 4;
            let size = signed_size_to_u64(h.interface_offsets_size, "interface_offsets_size")?;
            let off = signed_size_to_u64(h.interface_offsets_offset, "interface_offsets_offset")?;
            let count = checked_array_len(
                size / each_sz as u64,
                each_sz,
                Some(self.stream.remaining_from(off)),
                "interface_offsets",
            )?;
            let mut out = Vec::with_capacity(count);
            self.stream.set_position(off);
            for _ in 0..count {
                out.push(Il2CppInterfaceOffset::read(&mut self.stream)?);
            }
            out
        };

        {
            let size = signed_size_to_u64(h.nested_types_size, "nested_types_size")?;
            let off = signed_size_to_u64(h.nested_types_offset, "nested_types_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "nested_types")?;
            self.nested_type_indices = self.stream.read_i32_array(off, count)?;
        }

        self.event_defs = load!(Il2CppEventDefinition, h.events_offset, h.events_size, h.events_count);
        self.generic_containers = load!(Il2CppGenericContainer, h.generic_containers_offset, h.generic_containers_size, h.generic_containers_count);

        if v38 {
            let (generic_params, gp_offsets) = self.read_metadata_array_with_offsets::<Il2CppGenericParameter>(
                signed_size_to_u64(h.generic_parameters_offset, "generic_parameters_offset")?,
                signed_size_to_u64(h.generic_parameters_size, "generic_parameters_size")?,
                if h.generic_parameters_count > 0 { Some(h.generic_parameters_count as usize) } else { None },
            )?;
            self.generic_parameters = generic_params;
            self.generic_param_offset_to_index = gp_offsets;
        } else {
            self.generic_parameters = load!(Il2CppGenericParameter, h.generic_parameters_offset, h.generic_parameters_size, h.generic_parameters_count);
        }

        {
            let size = signed_size_to_u64(h.generic_parameter_constraints_size, "generic_parameter_constraints_size")?;
            let off = signed_size_to_u64(h.generic_parameter_constraints_offset, "generic_parameter_constraints_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "generic_parameter_constraints")?;
            self.constraint_indices = self.stream.read_i32_array(off, count)?;
        }

        {
            let size = signed_size_to_u64(h.vtable_methods_size, "vtable_methods_size")?;
            let off = signed_size_to_u64(h.vtable_methods_offset, "vtable_methods_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "vtable_methods")?;
            self.vtable_methods = self.stream.read_u32_array(off, count)?;
        }

        self.string_literals = load!(Il2CppStringLiteral, h.string_literal_offset, h.string_literal_size, 0);

        if self.version > 16.0 {
            self.field_refs = load!(Il2CppFieldRef, h.field_refs_offset, h.field_refs_size, 0);

            if self.version < 27.0 {
                let usage_lists = load!(Il2CppMetadataUsageList, h.metadata_usage_lists_offset, h.metadata_usage_lists_count, 0);
                let usage_pairs = load!(Il2CppMetadataUsagePair, h.metadata_usage_pairs_offset, h.metadata_usage_pairs_count, 0);
                self.process_metadata_usage(&usage_lists, &usage_pairs);
            }
        }

        if self.version > 20.0 && self.version < 29.0 {
            self.attribute_type_ranges = load!(Il2CppCustomAttributeTypeRange, h.attributes_info_offset, h.attributes_info_count, 0);
            let size = signed_size_to_u64(h.attribute_types_count, "attribute_types_count")?;
            let off = signed_size_to_u64(h.attribute_types_offset, "attribute_types_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "attribute_types")?;
            self.attribute_types = self.stream.read_i32_array(off, count)?;
        }

        if self.version >= 29.0 {
            self.attribute_data_ranges = load!(Il2CppCustomAttributeDataRange, h.attribute_data_range_offset, h.attribute_data_range_size, 0);
        }

        if self.version > 24.0 {
            self.build_attribute_lookup();
        }

        self.metadata_usages_count = self.calculate_metadata_usages_count();

        if v38 {
            self.type_inline_arrays = if self.version >= 104.0 {
                load!(Il2CppInlineArrayLength, h.type_inline_arrays_offset, h.type_inline_arrays_size, h.type_inline_arrays_count)
            } else {
                Vec::new()
            };
        }

        if h.referenced_assemblies_offset > 0 && h.referenced_assemblies_size > 0 {
            let size = signed_size_to_u64(h.referenced_assemblies_size, "referenced_assemblies_size")?;
            let off = signed_size_to_u64(h.referenced_assemblies_offset, "referenced_assemblies_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "referenced_assemblies")?;
            self.referenced_assemblies = self.stream.read_i32_array(off, count)?;
        }

        Ok(())
    }

    fn load_metadata_codm(&mut self) -> Result<()> {
        let h = self.header.clone();
        let _widths = IndexWidths::from_header(&h, self.version);

        macro_rules! load_c {
            ($t:ty, $off:expr, $size:expr) => {{
                let off = signed_size_to_u64($off, stringify!($off))?;
                let size = signed_size_to_u64($size, stringify!($size))?;
                self.read_metadata_array_codm::<$t>(off, size)?
            }};
        }
        macro_rules! load_std {
            ($t:ty, $off:expr, $size:expr) => {{
                let off = signed_size_to_u64($off, stringify!($off))?;
                let size = signed_size_to_u64($size, stringify!($size))?;
                self.read_metadata_array::<$t>(off, size, None)?
            }};
        }

        self.image_defs = load_c!(Il2CppImageDefinition, h.images_offset, h.images_size);
        self.assembly_defs = load_c!(Il2CppAssemblyDefinition, h.assemblies_offset, h.assemblies_size);
        self.type_defs = load_c!(Il2CppTypeDefinition, h.type_definitions_offset, h.type_definitions_size);
        self.method_defs = load_c!(Il2CppMethodDefinition, h.methods_offset, h.methods_size);
        self.parameter_defs = load_c!(Il2CppParameterDefinition, h.parameters_offset, h.parameters_size);
        self.field_defs = load_c!(Il2CppFieldDefinition, h.fields_offset, h.fields_size);

        let mut field_defaults_vec = load_std!(Il2CppFieldDefaultValue, h.field_default_values_offset, h.field_default_values_size);
        field_defaults_vec.retain(|v| v.field_index >= 0);
        self.field_default_values = map_by_key_dedup(
            field_defaults_vec,
            |v| v.field_index,
            "field_default_values(codm)",
        );

        let mut param_defaults = load_std!(Il2CppParameterDefaultValue, h.parameter_default_values_offset, h.parameter_default_values_size);
        param_defaults.retain(|v| v.parameter_index >= 0);
        self.param_default_values = map_by_key_dedup(
            param_defaults,
            |v| v.parameter_index,
            "parameter_default_values(codm)",
        );

        self.property_defs = load_c!(Il2CppPropertyDefinition, h.properties_offset, h.properties_size);

        self.interface_indices = {
            let type_idx_sz = IndexWidths::get_type_index_size(&h).max(1);
            let size = signed_size_to_u64(h.interfaces_size, "interfaces_size")?;
            let off = signed_size_to_u64(h.interfaces_offset, "interfaces_offset")?;
            let count = checked_array_len(
                size / type_idx_sz as u64,
                type_idx_sz,
                Some(self.stream.remaining_from(off)),
                "interfaces(codm)",
            )?;
            let mut out = Vec::with_capacity(count);
            self.stream.set_position(off);
            for _ in 0..count {
                out.push(self.stream.read_variable_index(type_idx_sz as u8)?);
            }
            out
        };

        self.interface_offsets = {
            let type_idx_sz = IndexWidths::get_type_index_size(&h).max(1);
            let each_sz = type_idx_sz + 4;
            let size = signed_size_to_u64(h.interface_offsets_size, "interface_offsets_size")?;
            let off = signed_size_to_u64(h.interface_offsets_offset, "interface_offsets_offset")?;
            let count = checked_array_len(
                size / each_sz as u64,
                each_sz,
                Some(self.stream.remaining_from(off)),
                "interface_offsets(codm)",
            )?;
            let mut out = Vec::with_capacity(count);
            self.stream.set_position(off);
            for _ in 0..count {
                out.push(Il2CppInterfaceOffset::read(&mut self.stream)?);
            }
            out
        };

        {
            let size = signed_size_to_u64(h.nested_types_size, "nested_types_size")?;
            let off = signed_size_to_u64(h.nested_types_offset, "nested_types_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "nested_types(codm)")?;
            self.nested_type_indices = self.stream.read_i32_array(off, count)?;
        }

        self.event_defs = load_c!(Il2CppEventDefinition, h.events_offset, h.events_size);
        self.generic_containers = load_c!(Il2CppGenericContainer, h.generic_containers_offset, h.generic_containers_size);
        self.generic_parameters = load_c!(Il2CppGenericParameter, h.generic_parameters_offset, h.generic_parameters_size);

        {
            let size = signed_size_to_u64(h.generic_parameter_constraints_size, "generic_parameter_constraints_size")?;
            let off = signed_size_to_u64(h.generic_parameter_constraints_offset, "generic_parameter_constraints_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "generic_parameter_constraints(codm)")?;
            self.constraint_indices = self.stream.read_i32_array(off, count)?;
        }

        {
            let size = signed_size_to_u64(h.vtable_methods_size, "vtable_methods_size")?;
            let off = signed_size_to_u64(h.vtable_methods_offset, "vtable_methods_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "vtable_methods(codm)")?;
            self.vtable_methods = self.stream.read_u32_array(off, count)?;
        }

        self.string_literals = load_std!(Il2CppStringLiteral, h.string_literal_offset, h.string_literal_size);

        self.field_refs = load_c!(Il2CppFieldRef, h.field_refs_offset, h.field_refs_size);

        if self.version < 27.0 {
            let usage_lists = load_c!(Il2CppMetadataUsageList, h.metadata_usage_lists_offset, h.metadata_usage_lists_count);
            let usage_pairs = load_std!(Il2CppMetadataUsagePair, h.metadata_usage_pairs_offset, h.metadata_usage_pairs_count);
            self.process_metadata_usage(&usage_lists, &usage_pairs);
        }

        if self.version < 29.0 {
            self.attribute_type_ranges = load_c!(Il2CppCustomAttributeTypeRange, h.attributes_info_offset, h.attributes_info_count);
            let size = signed_size_to_u64(h.attribute_types_count, "attribute_types_count")?;
            let off = signed_size_to_u64(h.attribute_types_offset, "attribute_types_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "attribute_types(codm)")?;
            self.attribute_types = self.stream.read_i32_array(off, count)?;
        }

        if self.version > 24.0 {
            self.build_attribute_lookup();
        }

        self.metadata_usages_count = self.calculate_metadata_usages_count();

        if h.referenced_assemblies_offset > 0 && h.referenced_assemblies_size > 0 {
            let size = signed_size_to_u64(h.referenced_assemblies_size, "referenced_assemblies_size")?;
            let off = signed_size_to_u64(h.referenced_assemblies_offset, "referenced_assemblies_offset")?;
            let count = checked_array_len(size / 4, 4, Some(self.stream.remaining_from(off)), "referenced_assemblies(codm)")?;
            self.referenced_assemblies = self.stream.read_i32_array(off, count)?;
        }

        Ok(())
    }

    fn read_metadata_array_codm<T: CodmReadable>(&mut self, offset: u64, size: u64) -> Result<Vec<T>> {
        if offset == 0 || size == 0 {
            return Ok(Vec::new());
        }
        let elem = T::codm_size_dispatch();
        if elem == 0 {
            return Err(Error::InvalidMetadata("CODM struct size missing".into()));
        }
        let count = checked_array_len(
            size / elem as u64,
            elem,
            Some(self.stream.remaining_from(offset)),
            "read_metadata_array_codm",
        )?;
        self.stream.set_position(offset);
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(T::read_codm_dispatch(&mut self.stream)?);
        }
        Ok(items)
    }

    fn read_metadata_array<T: MetadataReadable>(&mut self, offset: u64, size: u64, count_override: Option<usize>) -> Result<Vec<T>> {
        if offset == 0 || size == 0 {
            return Ok(Vec::new());
        }
        if size > self.stream.len() {
            return Err(Error::InvalidMetadata(format!(
                "metadata section size {size} exceeds file length {}",
                self.stream.len()
            )));
        }
        let count = if let Some(c) = count_override {
            if c == 0 { return Ok(Vec::new()); }
            checked_array_len(
                c as u64,
                1,
                Some(self.stream.remaining_from(offset)),
                "read_metadata_array(count_override)",
            )?
        } else {
            let element_size = T::byte_size(self.version) as u64;
            if element_size == 0 {
                self.stream.set_position(offset);
                let first = T::read(&mut self.stream, self.version)?;
                let consumed = self.stream.position() - offset;
                if consumed == 0 {
                    return Ok(Vec::new());
                }
                let num_elements = checked_array_len(
                    size / consumed,
                    consumed as usize,
                    Some(self.stream.remaining_from(offset)),
                    "read_metadata_array(variable)",
                )?;
                if num_elements == 0 {
                    return Ok(Vec::new());
                }
                let mut items = Vec::with_capacity(num_elements);
                items.push(first);
                for _ in 1..num_elements {
                    items.push(T::read(&mut self.stream, self.version)?);
                }
                return Ok(items);
            }
            checked_array_len(
                size / element_size,
                element_size as usize,
                Some(self.stream.remaining_from(offset)),
                "read_metadata_array",
            )?
        };
        self.stream.set_position(offset);
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(T::read(&mut self.stream, self.version)?);
        }
        Ok(items)
    }

    fn read_metadata_array_with_offsets<T: MetadataReadable>(
        &mut self, offset: u64, size: u64, count_override: Option<usize>,
    ) -> Result<(Vec<T>, HashMap<u64, usize>)> {
        if offset == 0 || size == 0 {
            return Ok((Vec::new(), HashMap::new()));
        }
        if size > self.stream.len() {
            return Err(Error::InvalidMetadata(format!(
                "metadata section size {size} exceeds file length {}",
                self.stream.len()
            )));
        }

        let count = if let Some(c) = count_override {
            if c == 0 { return Ok((Vec::new(), HashMap::new())); }
            checked_array_len(
                c as u64,
                1,
                Some(self.stream.remaining_from(offset)),
                "read_metadata_array_with_offsets(count_override)",
            )?
        } else {
            let element_size = T::byte_size(self.version) as u64;
            if element_size == 0 {
                self.stream.set_position(offset);
                let before = self.stream.position();
                let first = T::read(&mut self.stream, self.version)?;
                let consumed = self.stream.position() - before;
                if consumed == 0 {
                    return Ok((Vec::new(), HashMap::new()));
                }
                let num_elements = checked_array_len(
                    size / consumed,
                    consumed as usize,
                    Some(self.stream.remaining_from(offset)),
                    "read_metadata_array_with_offsets(variable)",
                )?;
                let mut items = Vec::with_capacity(num_elements);
                let mut offset_map = HashMap::with_capacity(num_elements);
                offset_map.insert(before - offset, 0);
                items.push(first);
                for i in 1..num_elements {
                    let pos_before = self.stream.position();
                    items.push(T::read(&mut self.stream, self.version)?);
                    offset_map.insert(pos_before - offset, i);
                }
                return Ok((items, offset_map));
            }
            checked_array_len(
                size / element_size,
                element_size as usize,
                Some(self.stream.remaining_from(offset)),
                "read_metadata_array_with_offsets",
            )?
        };

        self.stream.set_position(offset);
        let mut items = Vec::with_capacity(count);
        let mut offset_map = HashMap::with_capacity(count);
        for i in 0..count {
            let pos_before = self.stream.position();
            items.push(T::read(&mut self.stream, self.version)?);
            offset_map.insert(pos_before - offset, i);
        }
        Ok((items, offset_map))
    }

    fn process_metadata_usage(
        &mut self,
        lists: &[Il2CppMetadataUsageList],
        pairs: &[Il2CppMetadataUsagePair],
    ) {
        for i in 1..=7u32 {
            self.metadata_usage_dic.insert(i, HashMap::new());
        }

        for list in lists {
            for i in 0..list.count as usize {
                let offset = list.start as usize + i;
                if offset >= pairs.len() {
                    continue;
                }
                let pair = &pairs[offset];
                let usage = (pair.encoded_source_index & 0xE0000000) >> 29;
                let decoded = if self.version >= 27.0 {
                    (pair.encoded_source_index & 0x1FFFFFFE) >> 1
                } else {
                    pair.encoded_source_index & 0x1FFFFFFF
                };

                if usage >= 1 && usage <= 7 {
                    if let Some(dic) = self.metadata_usage_dic.get_mut(&usage) {
                        dic.insert(pair.destination_index, decoded);
                    }
                }
            }
        }
    }

    fn calculate_metadata_usages_count(&self) -> usize {
        let mut max_index = 0u32;
        for dic in self.metadata_usage_dic.values() {
            if let Some(&m) = dic.keys().max() {
                if m > max_index {
                    max_index = m;
                }
            }
        }
        if max_index == 0 && self.metadata_usage_dic.values().all(|d| d.is_empty()) {
            0
        } else {
            max_index as usize + 1
        }
    }

    fn build_attribute_lookup(&mut self) {
        self.attribute_type_ranges_dic.clear();

        for (img_idx, image_def) in self.image_defs.iter().enumerate() {
            let mut dic = HashMap::new();
            let mut dup_tokens = 0u32;
            let mut first_dup_token: Option<u32> = None;
            let end = image_def.custom_attribute_start as usize + image_def.custom_attribute_count as usize;

            for i in image_def.custom_attribute_start as usize..end {
                let token = if self.version >= 29.0 {
                    self.attribute_data_ranges.get(i).map(|r| r.token)
                } else {
                    self.attribute_type_ranges.get(i).map(|r| r.token)
                };
                if let Some(token) = token {
                    if dic.insert(token, i).is_some() {
                        dup_tokens += 1;
                        if first_dup_token.is_none() {
                            first_dup_token = Some(token);
                        }
                    }
                }
            }

            if dup_tokens > 0 {
                eprintln!(
                    "Warning: image {img_idx}: {dup_tokens} duplicate custom-attribute token(s) (e.g. 0x{:X}); keeping last",
                    first_dup_token.unwrap_or(0)
                );
            }

            self.attribute_type_ranges_dic.insert(img_idx, dic);
        }
    }

    pub fn get_string_from_index(&self, index: i32) -> Result<String> {
        if let Ok(cache) = self.string_cache.lock() {
            if let Some(cached) = cache.get(&index) {
                return Ok(cached.clone());
            }
        }
        let offset = self.header.string_offset as u64 + index as u64;
        let result = self.stream.peek_string_to_null_at(offset)?;
        if let Ok(mut cache) = self.string_cache.lock() {
            cache.entry(index).or_insert_with(|| result.clone());
        }
        Ok(result)
    }

    pub fn get_string_literal_from_index(&self, index: usize) -> Result<String> {
        if index >= self.string_literals.len() {
            return Err(crate::error::Error::Other(format!("String literal index {} out of bounds (len {})", index, self.string_literals.len())));
        }
        let sl = &self.string_literals[index];
        let data_base = self.header.string_literal_data_offset as u64;
        if self.version < 35.0 {
            let bytes = self.stream.peek_bytes_at(data_base + sl.data_index as u64, sl.length as usize)?;
            Ok(String::from_utf8_lossy(bytes).to_string())
        } else {
            let data_end = self.header.string_literal_data_offset as u64 + self.header.string_literal_data_size as u64;
            let next_offset = self.string_literals.get(index + 1)
                .map(|n| n.data_index as u64)
                .unwrap_or(data_end - data_base);
            let len = (next_offset - sl.data_index as u64) as usize;
            let bytes = self.stream.peek_bytes_at(data_base + sl.data_index as u64, len)?;
            Ok(String::from_utf8_lossy(bytes).to_string())
        }
    }

    pub fn peek_bytes_at(&self, offset: u64, len: usize) -> Result<&[u8]> {
        self.stream.peek_bytes_at(offset, len)
    }

    pub fn get_field_default_value(&self, field_index: i32) -> Option<&Il2CppFieldDefaultValue> {
        self.field_default_values.get(&field_index)
    }

    pub fn get_parameter_default_value(&self, param_index: i32) -> Option<&Il2CppParameterDefaultValue> {
        self.param_default_values.get(&param_index)
    }

    pub fn get_default_value_offset(&self, index: i32) -> u64 {
        self.header.field_and_parameter_default_value_data_offset as u64 + index as u64
    }

    pub fn get_custom_attribute_index(
        &self,
        image_index: usize,
        _custom_attribute_index: i32,
        token: u32,
    ) -> Option<usize> {
        if self.version > 24.0 {
            self.attribute_type_ranges_dic
                .get(&image_index)
                .and_then(|dic| dic.get(&token))
                .copied()
        } else if _custom_attribute_index >= 0 {
            Some(_custom_attribute_index as usize)
        } else {
            None
        }
    }
}

pub trait MetadataReadable: Sized {
    fn read(stream: &mut BinaryStream, version: f64) -> Result<Self>;
    fn byte_size(version: f64) -> usize;
    fn read_codm(_stream: &mut BinaryStream) -> Result<Self> {
        Err(Error::InvalidMetadata("CODM variant not supported for this struct".into()))
    }
    fn codm_byte_size() -> usize {
        0
    }
}

macro_rules! impl_metadata_readable {
    ($t:ty) => {
        impl MetadataReadable for $t {
            fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
                <$t>::read(stream, version)
            }
            fn byte_size(version: f64) -> usize {
                <$t>::byte_size(version)
            }
        }
    };
}

macro_rules! impl_metadata_readable_simple {
    ($t:ty, $size:expr) => {
        impl MetadataReadable for $t {
            fn read(stream: &mut BinaryStream, _version: f64) -> Result<Self> {
                <$t>::read(stream)
            }
            fn byte_size(_version: f64) -> usize {
                $size
            }
        }
    };
}

impl_metadata_readable!(Il2CppImageDefinition);
impl_metadata_readable!(Il2CppAssemblyDefinition);
impl_metadata_readable!(Il2CppTypeDefinition);
impl_metadata_readable!(Il2CppMethodDefinition);
impl_metadata_readable!(Il2CppParameterDefinition);
impl_metadata_readable!(Il2CppFieldDefinition);
impl_metadata_readable!(Il2CppFieldDefaultValue);
impl_metadata_readable!(Il2CppParameterDefaultValue);
impl_metadata_readable!(Il2CppPropertyDefinition);
impl_metadata_readable!(Il2CppEventDefinition);
impl_metadata_readable!(Il2CppGenericContainer);
impl_metadata_readable!(Il2CppGenericParameter);
impl_metadata_readable!(Il2CppStringLiteral);
impl_metadata_readable!(Il2CppFieldRef);
impl_metadata_readable!(Il2CppInlineArrayLength);
impl_metadata_readable_simple!(Il2CppMetadataUsageList, 8);
impl_metadata_readable_simple!(Il2CppMetadataUsagePair, 8);
impl_metadata_readable!(Il2CppCustomAttributeTypeRange);
impl_metadata_readable_simple!(Il2CppCustomAttributeDataRange, 8);

macro_rules! impl_codm_overrides {
    ($t:ty) => {
        impl CodmReadable for $t {
            fn read_codm_dispatch(stream: &mut BinaryStream) -> Result<Self> {
                <$t>::read_codm(stream)
            }
            fn codm_size_dispatch() -> usize {
                <$t>::CODM_BYTE_SIZE
            }
        }
    };
}

pub trait CodmReadable: Sized {
    fn read_codm_dispatch(stream: &mut BinaryStream) -> Result<Self>;
    fn codm_size_dispatch() -> usize;
}

impl_codm_overrides!(Il2CppImageDefinition);
impl_codm_overrides!(Il2CppAssemblyDefinition);
impl_codm_overrides!(Il2CppTypeDefinition);
impl_codm_overrides!(Il2CppMethodDefinition);
impl_codm_overrides!(Il2CppParameterDefinition);
impl_codm_overrides!(Il2CppFieldDefinition);
impl_codm_overrides!(Il2CppPropertyDefinition);
impl_codm_overrides!(Il2CppEventDefinition);
impl_codm_overrides!(Il2CppGenericContainer);
impl_codm_overrides!(Il2CppGenericParameter);
impl_codm_overrides!(Il2CppFieldRef);
impl_codm_overrides!(Il2CppMetadataUsageList);
impl_codm_overrides!(Il2CppCustomAttributeTypeRange);
