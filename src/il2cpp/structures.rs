use crate::error::Result;
use crate::io::BinaryStream;
use crate::read_versioned;
use std::cell::RefCell;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnityVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub build_type: UnityBuildType,
    pub build_number: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnityBuildType {
    Alpha = 0,
    Beta = 1,
    Final = 2,
    Patch = 3,
}

impl UnityVersion {
    pub fn new(
        major: u32,
        minor: u32,
        patch: u32,
        build_type: UnityBuildType,
        build_number: u32,
    ) -> Self {
        Self {
            major,
            minor,
            patch,
            build_type,
            build_number,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let major;
        let minor;
        let mut patch = 0u32;
        let mut build_type = UnityBuildType::Final;
        let mut build_number = 0u32;

        let parts: Vec<&str> = s.splitn(3, '.').collect();
        if parts.len() < 2 {
            return None;
        }

        major = parts[0].parse().ok()?;
        minor = parts[1].parse().ok()?;

        if parts.len() == 3 {
            let rest = parts[2];
            let type_pos = rest.find(|c: char| c == 'f' || c == 'b' || c == 'a' || c == 'p');
            if let Some(pos) = type_pos {
                patch = rest[..pos].parse().ok()?;
                let type_char = rest.as_bytes()[pos];
                build_type = match type_char {
                    b'a' => UnityBuildType::Alpha,
                    b'b' => UnityBuildType::Beta,
                    b'f' => UnityBuildType::Final,
                    b'p' => UnityBuildType::Patch,
                    _ => UnityBuildType::Final,
                };
                build_number = rest[pos + 1..].parse().unwrap_or(1);
            } else {
                patch = rest.parse().unwrap_or(0);
            }
        }

        Some(Self {
            major,
            minor,
            patch,
            build_type,
            build_number,
        })
    }

    pub fn gte(&self, major: u32, minor: u32, patch: u32, bt: UnityBuildType, bn: u32) -> bool {
        let lhs = (
            self.major,
            self.minor,
            self.patch,
            self.build_type as u32,
            self.build_number,
        );
        let rhs = (major, minor, patch, bt as u32, bn);
        lhs >= rhs
    }

    pub fn gte_simple(&self, major: u32, minor: u32, patch: u32) -> bool {
        (self.major, self.minor, self.patch) >= (major, minor, patch)
    }

    pub fn gte_major_minor(&self, major: u32, minor: u32) -> bool {
        (self.major, self.minor) >= (major, minor)
    }

    pub fn gte_major(&self, major: u32) -> bool {
        self.major >= major
    }

    pub fn resolve_sub_version(&self, raw_version: i32) -> f64 {
        match raw_version {
            24 => {
                if self.gte_simple(2020, 1, 11) {
                    24.4
                } else if self.gte_major(2020) {
                    24.3
                } else if self.gte_simple(2019, 4, 21) {
                    24.5
                } else if self.gte_simple(2019, 4, 15) {
                    24.4
                } else if self.gte_simple(2019, 3, 7) {
                    24.3
                } else if self.gte_major(2019) {
                    24.2
                } else if self.gte_simple(2018, 4, 34) {
                    24.15
                } else if self.gte_major_minor(2018, 3) {
                    24.1
                } else {
                    24.0
                }
            }
            27 => {
                if self.gte_major_minor(2021, 1) {
                    27.2
                } else if self.gte_simple(2020, 2, 4) {
                    27.1
                } else {
                    27.0
                }
            }
            29 => {
                if self.gte(2022, 1, 0, UnityBuildType::Beta, 7) {
                    29.1
                } else {
                    29.0
                }
            }
            31 => {
                if self.gte(2022, 3, 33, UnityBuildType::Final, 1) {
                    31.1
                } else {
                    31.0
                }
            }
            // Unity 6 moved a lot of tables around, and 106/106.1/107 are three
            // subtly different layouts behind two version numbers:
            //
            //   6.5.0a6 => 106    generic parameter / field / default-value indices
            //                     become variable width, and Il2CppGenericContainer
            //                     packs type_argc into 16 bits and is_method into 8
            //   6.6.0a6 => 106.1  adds the alwaysInitMetadataUsages count+pointer pair
            //                     to Il2CppMetadataRegistration and drops the second
            //                     value of MetadataUsageType
            //   6.6.0a7 => 107    no functional change from 106.1 at all
            //   6.5.1f1 => 107    6.5.1 backported the *number* without the changes,
            //                     so 6.5.1's "107" is really a plain 106
            //   6.6.0b1 => 108    moves the invoker pointer table, rgctx data, static
            //                     constructor type indices, generic method tables
            //                     and method specs out of the binary and into the
            //                     metadata
            //
            // So "107" never names a layout of its own and the Unity version — not
            // the declared number — decides which of the two real layouts applies.
            106 | 107 => {
                if self.unity_6_at_least(6, UnityBuildType::Alpha, 6) {
                    106.1
                } else {
                    106.0
                }
            }
            _ => raw_version as f64,
        }
    }

    /// Unity 6 is written `6000.x.y` by the engine and `6.x.y` by some tooling and
    /// release notes. Both name the same release, so compare them as one.
    fn unity_6_at_least(&self, minor: u32, bt: UnityBuildType, bn: u32) -> bool {
        let major = if self.major == 6 { 6000 } else { self.major };
        (
            major,
            self.minor,
            self.patch,
            self.build_type as u32,
            self.build_number,
        ) >= (6000, minor, 0, bt as u32, bn)
    }
}

impl fmt::Display for UnityVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = match self.build_type {
            UnityBuildType::Alpha => 'a',
            UnityBuildType::Beta => 'b',
            UnityBuildType::Final => 'f',
            UnityBuildType::Patch => 'p',
        };
        write!(
            f,
            "{}.{}.{}{}{}",
            self.major, self.minor, self.patch, t, self.build_number
        )
    }
}

fn decode_packing_size(encoded: u32) -> u32 {
    match encoded {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 4,
        4 => 8,
        5 => 16,
        6 => 32,
        7 => 64,
        8 => 128,
        _ => 0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Il2CppPackingSizeEnum {
    Zero = 0,
    One = 1,
    Two = 2,
    Four = 3,
    Eight = 4,
    Sixteen = 5,
    ThirtyTwo = 6,
    SixtyFour = 7,
    OneHundredTwentyEight = 8,
}

impl Il2CppPackingSizeEnum {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            3 => Some(Self::Four),
            4 => Some(Self::Eight),
            5 => Some(Self::Sixteen),
            6 => Some(Self::ThirtyTwo),
            7 => Some(Self::SixtyFour),
            8 => Some(Self::OneHundredTwentyEight),
            _ => None,
        }
    }

    pub fn numerical_value(self) -> u32 {
        decode_packing_size(self as u32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IndexWidths {
    pub type_def: u8,
    pub generic_container: u8,
    pub type_index: u8,
    pub parameter_def: u8,
    pub interface_offset: u8,
    pub event: u8,
    pub property: u8,
    pub nested_type: u8,
    pub method: u8,
    pub generic_param: u8,
    pub field: u8,
    pub default_value_data: u8,
    // v108+ only: indices into the tables that moved into the metadata. The
    // referenced tables live in the binary, so their widths cannot be derived from
    // a metadata section count; each is recovered from the element size of the
    // metadata section that stores it (see `IndexWidths::from_header`).
    pub generic_inst: u8,
    pub method_spec: u8,
    pub generic_method_pointer: u8,
    pub invoker: u8,
    pub adjustor_thunk: u8,
}

impl Default for IndexWidths {
    fn default() -> Self {
        Self {
            type_def: 4,
            generic_container: 4,
            type_index: 4,
            parameter_def: 4,
            interface_offset: 4,
            event: 4,
            property: 4,
            nested_type: 4,
            method: 4,
            generic_param: 4,
            field: 4,
            default_value_data: 4,
            generic_inst: 4,
            method_spec: 4,
            generic_method_pointer: 4,
            invoker: 4,
            adjustor_thunk: 4,
        }
    }
}

impl IndexWidths {
    pub fn from_header(h: &Il2CppGlobalMetadataHeader, version: f64) -> Self {
        if version < 38.0 {
            return Self::default();
        }
        fn w(count: i32) -> u8 {
            if count <= 0 {
                4
            } else if count <= 255 {
                1
            } else if count <= 65535 {
                2
            } else {
                4
            }
        }
        let type_index = if h.interface_offsets_count > 0 && h.interface_offsets_size > 0 {
            let bpe = h.interface_offsets_size / h.interface_offsets_count;
            if bpe > 4 { (bpe - 4) as u8 } else { 4 }
        } else {
            4
        };
        let method_width = if version >= 105.0 {
            w(h.methods_count)
        } else {
            4
        };
        Self {
            type_def: w(h.type_definitions_count),
            generic_container: w(h.generic_containers_count),
            type_index,
            parameter_def: if version >= 39.0 {
                w(h.parameters_count)
            } else {
                4
            },
            interface_offset: if version >= 104.0 {
                w(h.interface_offsets_count)
            } else {
                4
            },
            event: if version >= 104.0 {
                w(h.events_count)
            } else {
                4
            },
            property: if version >= 104.0 {
                w(h.properties_count)
            } else {
                4
            },
            nested_type: if version >= 104.0 {
                w(h.nested_types_count)
            } else {
                4
            },
            method: method_width,
            generic_param: if version >= 106.0 {
                w(h.generic_parameters_count)
            } else {
                4
            },
            field: if version >= 106.0 {
                w(h.fields_count)
            } else {
                4
            },
            default_value_data: if version >= 106.0 {
                w(h.field_and_parameter_default_value_data_count)
            } else {
                4
            },
            ..IndexWidths::v108(h, version, method_width)
        }
    }

    /// The five widths that only exist from v108 on.
    ///
    /// v108 moves the method-spec tables, the generic method table, the invoker
    /// index table, the RGCTX data and the static-constructor type list into the
    /// metadata, and makes the indices *into the binary's* generic-inst, generic
    /// method pointer, invoker and adjustor-thunk tables variable width. Those
    /// tables are not in the metadata file, so instead of counting entries the
    /// width is recovered from the element size of the metadata section that
    /// stores the index: a spec-table element is a method-definition index plus
    /// one or two generic-inst indices, and so on down the chain.
    fn v108(h: &Il2CppGlobalMetadataHeader, version: f64, method_width: u8) -> Self {
        let mut widths = Self::default();
        if version < 108.0 {
            return widths;
        }
        let m = method_width as i32;
        let mut generic_inst = 4i32;
        if h.method_specs_on_generic_type_count > 0 {
            generic_inst =
                h.method_specs_on_generic_type_size / h.method_specs_on_generic_type_count - m;
        } else if h.generic_method_specs_on_type_count > 0 {
            generic_inst =
                h.generic_method_specs_on_type_size / h.generic_method_specs_on_type_count - m;
        } else if h.method_specs_count > 0 {
            // The full spec table stores classIndex *and* methodIndex.
            generic_inst = (h.method_specs_size / h.method_specs_count - m) / 2;
        }

        // The three spec tables share one index space, so the width depends on the
        // combined entry count rather than any single table's.
        let spec_count = (h.method_specs_on_generic_type_count
            + h.generic_method_specs_on_type_count
            + h.method_specs_count) as i64;
        let method_spec = if spec_count <= 255 {
            1
        } else if spec_count <= 65535 {
            2
        } else {
            4
        };

        let invoker = if h.invoker_indices_count > 0 && h.invoker_indices_size > 0 {
            (h.invoker_indices_size / h.invoker_indices_count) as u8
        } else {
            4
        };

        // genericMethodTable element = spec index + generic method pointer index +
        // invoker index (the adjustor variant adds the adjustor thunk index).
        let mut generic_method_pointer = 4i32;
        if h.generic_method_functions_definitions_count > 0 {
            generic_method_pointer = h.generic_method_functions_definitions_size
                / h.generic_method_functions_definitions_count
                - method_spec as i32
                - invoker as i32;
        }
        let mut adjustor_thunk = 4i32;
        if h.generic_method_functions_definitions_with_adjustor_count > 0 {
            adjustor_thunk = h.generic_method_functions_definitions_with_adjustor_size
                / h.generic_method_functions_definitions_with_adjustor_count
                - method_spec as i32
                - invoker as i32
                - generic_method_pointer;
        }

        // A derived width of 0 or something absurd means the sections do not line
        // up; fall back to pointer-width indices rather than reading 0-byte fields.
        fn sane(w: i32) -> u8 {
            if (1..=8).contains(&w) { w as u8 } else { 4 }
        }
        widths.generic_inst = sane(generic_inst);
        widths.method_spec = method_spec as u8;
        widths.generic_method_pointer = sane(generic_method_pointer);
        widths.invoker = sane(invoker as i32);
        widths.adjustor_thunk = sane(adjustor_thunk);
        widths
    }

    pub fn get_type_index_size(h: &Il2CppGlobalMetadataHeader) -> usize {
        if h.interface_offsets_count > 0 && h.interface_offsets_size > 0 {
            let bpe = (h.interface_offsets_size / h.interface_offsets_count) as usize;
            if bpe > 4 { bpe - 4 } else { 4 }
        } else {
            4
        }
    }
}

thread_local! {
    static INDEX_WIDTHS: RefCell<IndexWidths> = RefCell::new(IndexWidths::default());
}

pub fn set_index_widths(widths: IndexWidths) {
    INDEX_WIDTHS.with(|w| *w.borrow_mut() = widths);
}

#[inline(always)]
fn read_type_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().type_index))
}
#[inline(always)]
fn read_type_def_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().type_def))
}
#[inline(always)]
fn read_gc_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().generic_container))
}
#[inline(always)]
fn read_param_def_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().parameter_def))
}
#[inline(always)]
fn read_ioffset_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().interface_offset))
}
#[inline(always)]
fn read_event_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().event))
}
#[inline(always)]
fn read_property_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().property))
}
#[inline(always)]
fn read_nested_type_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().nested_type))
}
#[inline(always)]
fn read_method_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().method))
}
#[inline(always)]
fn read_gp_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().generic_param))
}
#[inline(always)]
fn read_field_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().field))
}
#[inline(always)]
fn read_dvdata_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().default_value_data))
}
// v108+ only: indices into the tables that moved into the metadata.
#[inline(always)]
fn read_generic_inst_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().generic_inst))
}
#[inline(always)]
fn read_method_spec_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().method_spec))
}
#[inline(always)]
fn read_gmptr_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().generic_method_pointer))
}
#[inline(always)]
fn read_invoker_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().invoker))
}
#[inline(always)]
fn read_adjustor_thunk_idx(s: &mut BinaryStream) -> Result<i32> {
    INDEX_WIDTHS.with(|iw| s.read_variable_index(iw.borrow().adjustor_thunk))
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppGlobalMetadataHeader {
    pub sanity: u32,
    pub version: i32,
    pub string_literal_offset: i32,
    pub string_literal_size: i32,
    pub string_literal_data_offset: i32,
    pub string_literal_data_size: i32,
    pub string_offset: i32,
    pub string_size: i32,
    pub events_offset: i32,
    pub events_size: i32,
    pub properties_offset: i32,
    pub properties_size: i32,
    pub methods_offset: i32,
    pub methods_size: i32,
    pub parameter_default_values_offset: i32,
    pub parameter_default_values_size: i32,
    pub field_default_values_offset: i32,
    pub field_default_values_size: i32,
    pub field_and_parameter_default_value_data_offset: i32,
    pub field_and_parameter_default_value_data_size: i32,
    pub field_marshaled_sizes_offset: i32,
    pub field_marshaled_sizes_size: i32,
    pub parameters_offset: i32,
    pub parameters_size: i32,
    pub fields_offset: i32,
    pub fields_size: i32,
    pub generic_parameters_offset: i32,
    pub generic_parameters_size: i32,
    pub generic_parameter_constraints_offset: i32,
    pub generic_parameter_constraints_size: i32,
    pub generic_containers_offset: i32,
    pub generic_containers_size: i32,
    pub nested_types_offset: i32,
    pub nested_types_size: i32,
    pub interfaces_offset: i32,
    pub interfaces_size: i32,
    pub vtable_methods_offset: i32,
    pub vtable_methods_size: i32,
    pub interface_offsets_offset: i32,
    pub interface_offsets_size: i32,
    pub type_definitions_offset: i32,
    pub type_definitions_size: i32,
    pub rgctx_entries_offset: i32,
    pub rgctx_entries_count: i32,
    pub images_offset: i32,
    pub images_size: i32,
    pub assemblies_offset: i32,
    pub assemblies_size: i32,
    pub metadata_usage_lists_offset: i32,
    pub metadata_usage_lists_count: i32,
    pub metadata_usage_pairs_offset: i32,
    pub metadata_usage_pairs_count: i32,
    pub field_refs_offset: i32,
    pub field_refs_size: i32,
    pub referenced_assemblies_offset: i32,
    pub referenced_assemblies_size: i32,
    pub attributes_info_offset: i32,
    pub attributes_info_count: i32,
    pub attribute_types_offset: i32,
    pub attribute_types_count: i32,
    pub attribute_data_offset: i32,
    pub attribute_data_size: i32,
    pub attribute_data_range_offset: i32,
    pub attribute_data_range_size: i32,
    pub unresolved_virtual_call_parameter_types_offset: i32,
    pub unresolved_virtual_call_parameter_types_size: i32,
    pub unresolved_virtual_call_parameter_ranges_offset: i32,
    pub unresolved_virtual_call_parameter_ranges_size: i32,
    pub windows_runtime_type_names_offset: i32,
    pub windows_runtime_type_names_size: i32,
    pub windows_runtime_strings_offset: i32,
    pub windows_runtime_strings_size: i32,
    pub exported_type_definitions_offset: i32,
    pub exported_type_definitions_size: i32,
    // v38+ section count fields (0 for pre-v38, used for variable-width index computation)
    pub type_definitions_count: i32,
    pub generic_containers_count: i32,
    pub interface_offsets_count: i32,
    pub parameters_count: i32,
    pub events_count: i32,
    pub properties_count: i32,
    pub nested_types_count: i32,
    pub methods_count: i32,
    pub generic_parameters_count: i32,
    pub fields_count: i32,
    pub field_and_parameter_default_value_data_count: i32,
    // v104+
    pub type_inline_arrays_offset: i32,
    pub type_inline_arrays_size: i32,
    pub type_inline_arrays_count: i32,
    // v108+ (Unity 6.6.0b1). These nine sections are the tables that moved out of
    // the binary — out of `Il2CppMetadataRegistration` and out of each
    // `Il2CppCodeGenModule` — and into the metadata file.
    pub method_specs_on_generic_type_offset: i32,
    pub method_specs_on_generic_type_size: i32,
    pub method_specs_on_generic_type_count: i32,
    pub generic_method_specs_on_type_offset: i32,
    pub generic_method_specs_on_type_size: i32,
    pub generic_method_specs_on_type_count: i32,
    pub method_specs_offset: i32,
    pub method_specs_size: i32,
    pub method_specs_count: i32,
    pub generic_method_functions_definitions_offset: i32,
    pub generic_method_functions_definitions_size: i32,
    pub generic_method_functions_definitions_count: i32,
    pub generic_method_functions_definitions_with_adjustor_offset: i32,
    pub generic_method_functions_definitions_with_adjustor_size: i32,
    pub generic_method_functions_definitions_with_adjustor_count: i32,
    pub invoker_indices_offset: i32,
    pub invoker_indices_size: i32,
    pub invoker_indices_count: i32,
    pub rgctx_ranges_offset: i32,
    pub rgctx_ranges_size: i32,
    pub rgctx_ranges_count: i32,
    pub rgctx_values_offset: i32,
    pub rgctx_values_size: i32,
    pub rgctx_values_count: i32,
    pub static_constructor_type_indices_offset: i32,
    pub static_constructor_type_indices_size: i32,
    pub static_constructor_type_indices_count: i32,
}

impl Il2CppGlobalMetadataHeader {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let v38 = version >= 38.0;
        macro_rules! sec {
            () => {{
                let o = stream.read_i32()?;
                let s = stream.read_i32()?;
                let c = if v38 { stream.read_i32()? } else { 0 };
                (o, s, c)
            }};
        }
        let sanity = stream.read_u32()?;
        let version_field = stream.read_i32()?;
        let (string_literal_offset, string_literal_size, _) = sec!();
        let (string_literal_data_offset, string_literal_data_size, _) = sec!();
        let (string_offset, string_size, _) = sec!();
        let (events_offset, events_size, events_count) = sec!();
        let (properties_offset, properties_size, properties_count) = sec!();
        let (methods_offset, methods_size, methods_count) = sec!();
        let (parameter_default_values_offset, parameter_default_values_size, _) = sec!();
        let (field_default_values_offset, field_default_values_size, _) = sec!();
        let (
            field_and_parameter_default_value_data_offset,
            field_and_parameter_default_value_data_size,
            field_and_parameter_default_value_data_count,
        ) = sec!();
        let (field_marshaled_sizes_offset, field_marshaled_sizes_size, _) = sec!();
        let (parameters_offset, parameters_size, parameters_count) = sec!();
        let (fields_offset, fields_size, fields_count) = sec!();
        let (generic_parameters_offset, generic_parameters_size, generic_parameters_count) = sec!();
        let (generic_parameter_constraints_offset, generic_parameter_constraints_size, _) = sec!();
        let (generic_containers_offset, generic_containers_size, generic_containers_count) = sec!();
        let (nested_types_offset, nested_types_size, nested_types_count) = sec!();
        let (interfaces_offset, interfaces_size, _) = sec!();
        let (vtable_methods_offset, vtable_methods_size, _) = sec!();
        let (interface_offsets_offset, interface_offsets_size, interface_offsets_count) = sec!();
        let (type_definitions_offset, type_definitions_size, type_definitions_count) = sec!();
        let (type_inline_arrays_offset, type_inline_arrays_size, type_inline_arrays_count) =
            if version >= 104.0 { sec!() } else { (0, 0, 0) };
        let (rgctx_entries_offset, rgctx_entries_count) = if version <= 24.15 {
            let (o, s, _) = sec!();
            (o, s)
        } else {
            (0, 0)
        };
        let (images_offset, images_size, _) = sec!();
        let (assemblies_offset, assemblies_size, _) = sec!();
        let (metadata_usage_lists_offset, metadata_usage_lists_count) = if version < 27.0 {
            let (o, s, _) = sec!();
            (o, s)
        } else {
            (0, 0)
        };
        let (metadata_usage_pairs_offset, metadata_usage_pairs_count) = if version < 27.0 {
            let (o, s, _) = sec!();
            (o, s)
        } else {
            (0, 0)
        };
        let (field_refs_offset, field_refs_size, _) = sec!();
        let (referenced_assemblies_offset, referenced_assemblies_size, _) = sec!();
        let (
            attributes_info_offset,
            attributes_info_count,
            attribute_types_offset,
            attribute_types_count,
            attribute_data_offset,
            attribute_data_size,
            attribute_data_range_offset,
            attribute_data_range_size,
        ) = if version < 29.0 {
            let (ao, ac, _) = sec!();
            let (to, tc, _) = sec!();
            (ao, ac, to, tc, 0, 0, 0, 0)
        } else {
            let (ddo, dds, _) = sec!();
            let (dro, drs, _) = sec!();
            (0, 0, 0, 0, ddo, dds, dro, drs)
        };
        let (
            unresolved_virtual_call_parameter_types_offset,
            unresolved_virtual_call_parameter_types_size,
            _,
        ) = sec!();
        let (
            unresolved_virtual_call_parameter_ranges_offset,
            unresolved_virtual_call_parameter_ranges_size,
            _,
        ) = sec!();
        let (windows_runtime_type_names_offset, windows_runtime_type_names_size, _) = sec!();
        let (windows_runtime_strings_offset, windows_runtime_strings_size) = if version >= 27.0 {
            let (o, s, _) = sec!();
            (o, s)
        } else {
            (0, 0)
        };
        let (exported_type_definitions_offset, exported_type_definitions_size) = if version >= 24.0
        {
            let (o, s, _) = sec!();
            (o, s)
        } else {
            (0, 0)
        };
        let mut v108 = [0i32; 27];
        if version >= 108.0 {
            for slot in v108.chunks_mut(3) {
                let (o, s, c) = sec!();
                slot[0] = o;
                slot[1] = s;
                slot[2] = c;
            }
        }
        let [
            method_specs_on_generic_type_offset,
            method_specs_on_generic_type_size,
            method_specs_on_generic_type_count,
            generic_method_specs_on_type_offset,
            generic_method_specs_on_type_size,
            generic_method_specs_on_type_count,
            method_specs_offset,
            method_specs_size,
            method_specs_count,
            generic_method_functions_definitions_offset,
            generic_method_functions_definitions_size,
            generic_method_functions_definitions_count,
            generic_method_functions_definitions_with_adjustor_offset,
            generic_method_functions_definitions_with_adjustor_size,
            generic_method_functions_definitions_with_adjustor_count,
            invoker_indices_offset,
            invoker_indices_size,
            invoker_indices_count,
            rgctx_ranges_offset,
            rgctx_ranges_size,
            rgctx_ranges_count,
            rgctx_values_offset,
            rgctx_values_size,
            rgctx_values_count,
            static_constructor_type_indices_offset,
            static_constructor_type_indices_size,
            static_constructor_type_indices_count,
        ] = v108;
        Ok(Self {
            sanity,
            version: version_field,
            string_literal_offset,
            string_literal_size,
            string_literal_data_offset,
            string_literal_data_size,
            string_offset,
            string_size,
            events_offset,
            events_size,
            properties_offset,
            properties_size,
            methods_offset,
            methods_size,
            parameter_default_values_offset,
            parameter_default_values_size,
            field_default_values_offset,
            field_default_values_size,
            field_and_parameter_default_value_data_offset,
            field_and_parameter_default_value_data_size,
            field_marshaled_sizes_offset,
            field_marshaled_sizes_size,
            parameters_offset,
            parameters_size,
            fields_offset,
            fields_size,
            generic_parameters_offset,
            generic_parameters_size,
            generic_parameter_constraints_offset,
            generic_parameter_constraints_size,
            generic_containers_offset,
            generic_containers_size,
            nested_types_offset,
            nested_types_size,
            interfaces_offset,
            interfaces_size,
            vtable_methods_offset,
            vtable_methods_size,
            interface_offsets_offset,
            interface_offsets_size,
            type_definitions_offset,
            type_definitions_size,
            rgctx_entries_offset,
            rgctx_entries_count,
            images_offset,
            images_size,
            assemblies_offset,
            assemblies_size,
            metadata_usage_lists_offset,
            metadata_usage_lists_count,
            metadata_usage_pairs_offset,
            metadata_usage_pairs_count,
            field_refs_offset,
            field_refs_size,
            referenced_assemblies_offset,
            referenced_assemblies_size,
            attributes_info_offset,
            attributes_info_count,
            attribute_types_offset,
            attribute_types_count,
            attribute_data_offset,
            attribute_data_size,
            attribute_data_range_offset,
            attribute_data_range_size,
            unresolved_virtual_call_parameter_types_offset,
            unresolved_virtual_call_parameter_types_size,
            unresolved_virtual_call_parameter_ranges_offset,
            unresolved_virtual_call_parameter_ranges_size,
            windows_runtime_type_names_offset,
            windows_runtime_type_names_size,
            windows_runtime_strings_offset,
            windows_runtime_strings_size,
            exported_type_definitions_offset,
            exported_type_definitions_size,
            type_definitions_count,
            generic_containers_count,
            interface_offsets_count,
            parameters_count,
            events_count,
            properties_count,
            nested_types_count,
            methods_count,
            generic_parameters_count,
            fields_count,
            field_and_parameter_default_value_data_count,
            type_inline_arrays_offset,
            type_inline_arrays_size,
            type_inline_arrays_count,
            method_specs_on_generic_type_offset,
            method_specs_on_generic_type_size,
            method_specs_on_generic_type_count,
            generic_method_specs_on_type_offset,
            generic_method_specs_on_type_size,
            generic_method_specs_on_type_count,
            method_specs_offset,
            method_specs_size,
            method_specs_count,
            generic_method_functions_definitions_offset,
            generic_method_functions_definitions_size,
            generic_method_functions_definitions_count,
            generic_method_functions_definitions_with_adjustor_offset,
            generic_method_functions_definitions_with_adjustor_size,
            generic_method_functions_definitions_with_adjustor_count,
            invoker_indices_offset,
            invoker_indices_size,
            invoker_indices_count,
            rgctx_ranges_offset,
            rgctx_ranges_size,
            rgctx_ranges_count,
            rgctx_values_offset,
            rgctx_values_size,
            rgctx_values_count,
            static_constructor_type_indices_offset,
            static_constructor_type_indices_size,
            static_constructor_type_indices_count,
        })
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let sanity = stream.read_u32()?;
        let version_field = stream.read_i32()?;
        let string_literal_offset = stream.read_i32()?;
        let string_literal_size = stream.read_i32()?;
        let string_literal_data_offset = stream.read_i32()?;
        let string_literal_data_size = stream.read_i32()?;
        let string_offset = stream.read_i32()?;
        let string_size = stream.read_i32()?;
        let events_offset = stream.read_i32()?;
        let events_size = stream.read_i32()?;
        let properties_offset = stream.read_i32()?;
        let properties_size = stream.read_i32()?;
        let methods_offset = stream.read_i32()?;
        let methods_size = stream.read_i32()?;
        let parameter_default_values_offset = stream.read_i32()?;
        let parameter_default_values_size = stream.read_i32()?;
        let field_default_values_offset = stream.read_i32()?;
        let field_default_values_size = stream.read_i32()?;
        let field_and_parameter_default_value_data_offset = stream.read_i32()?;
        let field_and_parameter_default_value_data_size = stream.read_i32()?;
        let field_marshaled_sizes_offset = stream.read_i32()?;
        let field_marshaled_sizes_size = stream.read_i32()?;
        let parameters_offset = stream.read_i32()?;
        let parameters_size = stream.read_i32()?;
        let fields_offset = stream.read_i32()?;
        let fields_size = stream.read_i32()?;
        let generic_parameters_offset = stream.read_i32()?;
        let generic_parameters_size = stream.read_i32()?;
        let generic_parameter_constraints_offset = stream.read_i32()?;
        let generic_parameter_constraints_size = stream.read_i32()?;
        let generic_containers_offset = stream.read_i32()?;
        let generic_containers_size = stream.read_i32()?;
        let nested_types_offset = stream.read_i32()?;
        let nested_types_size = stream.read_i32()?;
        let interfaces_offset = stream.read_i32()?;
        let interfaces_size = stream.read_i32()?;
        let vtable_methods_offset = stream.read_i32()?;
        let vtable_methods_size = stream.read_i32()?;
        let interface_offsets_offset = stream.read_i32()?;
        let interface_offsets_size = stream.read_i32()?;
        let type_definitions_offset = stream.read_i32()?;
        let type_definitions_size = stream.read_i32()?;
        let rgctx_entries_offset = stream.read_i32()?;
        let rgctx_entries_count = stream.read_i32()?;
        let images_offset = stream.read_i32()?;
        let images_size = stream.read_i32()?;
        let assemblies_offset = stream.read_i32()?;
        let assemblies_size = stream.read_i32()?;
        let metadata_usage_lists_offset = stream.read_i32()?;
        let metadata_usage_lists_count = stream.read_i32()?;
        let metadata_usage_pairs_offset = stream.read_i32()?;
        let metadata_usage_pairs_count = stream.read_i32()?;
        let field_refs_offset = stream.read_i32()?;
        let field_refs_size = stream.read_i32()?;
        let referenced_assemblies_offset = stream.read_i32()?;
        let referenced_assemblies_size = stream.read_i32()?;
        let attributes_info_offset = stream.read_i32()?;
        let attributes_info_count = stream.read_i32()?;
        let attribute_types_offset = stream.read_i32()?;
        let attribute_types_count = stream.read_i32()?;
        let unresolved_virtual_call_parameter_types_offset = stream.read_i32()?;
        let unresolved_virtual_call_parameter_types_size = stream.read_i32()?;
        let unresolved_virtual_call_parameter_ranges_offset = stream.read_i32()?;
        let unresolved_virtual_call_parameter_ranges_size = stream.read_i32()?;
        let windows_runtime_type_names_offset = stream.read_i32()?;
        let windows_runtime_type_names_size = stream.read_i32()?;
        Ok(Self {
            sanity,
            version: version_field,
            string_literal_offset,
            string_literal_size,
            string_literal_data_offset,
            string_literal_data_size,
            string_offset,
            string_size,
            events_offset,
            events_size,
            properties_offset,
            properties_size,
            methods_offset,
            methods_size,
            parameter_default_values_offset,
            parameter_default_values_size,
            field_default_values_offset,
            field_default_values_size,
            field_and_parameter_default_value_data_offset,
            field_and_parameter_default_value_data_size,
            field_marshaled_sizes_offset,
            field_marshaled_sizes_size,
            parameters_offset,
            parameters_size,
            fields_offset,
            fields_size,
            generic_parameters_offset,
            generic_parameters_size,
            generic_parameter_constraints_offset,
            generic_parameter_constraints_size,
            generic_containers_offset,
            generic_containers_size,
            nested_types_offset,
            nested_types_size,
            interfaces_offset,
            interfaces_size,
            vtable_methods_offset,
            vtable_methods_size,
            interface_offsets_offset,
            interface_offsets_size,
            type_definitions_offset,
            type_definitions_size,
            rgctx_entries_offset,
            rgctx_entries_count,
            images_offset,
            images_size,
            assemblies_offset,
            assemblies_size,
            metadata_usage_lists_offset,
            metadata_usage_lists_count,
            metadata_usage_pairs_offset,
            metadata_usage_pairs_count,
            field_refs_offset,
            field_refs_size,
            referenced_assemblies_offset,
            referenced_assemblies_size,
            attributes_info_offset,
            attributes_info_count,
            attribute_types_offset,
            attribute_types_count,
            attribute_data_offset: 0,
            attribute_data_size: 0,
            attribute_data_range_offset: 0,
            attribute_data_range_size: 0,
            unresolved_virtual_call_parameter_types_offset,
            unresolved_virtual_call_parameter_types_size,
            unresolved_virtual_call_parameter_ranges_offset,
            unresolved_virtual_call_parameter_ranges_size,
            windows_runtime_type_names_offset,
            windows_runtime_type_names_size,
            windows_runtime_strings_offset: 0,
            windows_runtime_strings_size: 0,
            exported_type_definitions_offset: 0,
            exported_type_definitions_size: 0,
            type_definitions_count: 0,
            generic_containers_count: 0,
            interface_offsets_count: 0,
            parameters_count: 0,
            events_count: 0,
            properties_count: 0,
            nested_types_count: 0,
            methods_count: 0,
            generic_parameters_count: 0,
            fields_count: 0,
            field_and_parameter_default_value_data_count: 0,
            type_inline_arrays_offset: 0,
            type_inline_arrays_size: 0,
            type_inline_arrays_count: 0,
            // Everything the CODM variant does not have (the v38+ counts and the
            // whole v108 tail) is absent rather than zero-valued in the file.
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppImageDefinition {
    pub name_index: i32,
    pub assembly_index: i32,
    pub type_start: i32,
    pub type_count: i32,
    pub exported_type_start: i32,
    pub exported_type_count: i32,
    pub entry_point_index: i32,
    pub token: u32,
    pub custom_attribute_start: i32,
    pub custom_attribute_count: i32,
    /// v108+. Per-image slices of the tables that moved into the metadata: the
    /// invoker-index table and the static-constructor type list are flat arrays
    /// that each image windows into, while the RGCTX data is a list of token
    /// ranges (`rgctx_ranges_count` of them starting at `rgctx_ranges_start`)
    /// indexing into the single global `rgctx_values` array.
    pub invoker_indices_start: i32,
    pub rgctx_ranges_start: i32,
    pub rgctx_ranges_count: i32,
    pub static_constructor_start: i32,
    pub static_constructor_count: i32,
}

impl Il2CppImageDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let name_index = stream.read_i32()?;
        let assembly_index = stream.read_i32()?;
        let type_start = read_type_def_idx(stream)?;
        let type_count = stream.read_u32()?;
        let exported_type_start = if version >= 24.0 {
            read_type_def_idx(stream)?
        } else {
            0
        };
        let exported_type_count = if version >= 24.0 {
            stream.read_u32()?
        } else {
            0
        };
        let entry_point_index = read_method_idx(stream)?;
        let token = read_versioned!(stream, version, 19.0, 999.0, read_u32, 0);
        let custom_attribute_start = read_versioned!(stream, version, 24.1, 999.0, read_i32, 0);
        let custom_attribute_count = read_versioned!(stream, version, 24.1, 999.0, read_i32, 0);
        let invoker_indices_start = if version >= 108.0 {
            stream.read_i32()?
        } else {
            0
        };
        let rgctx_ranges_start = if version >= 108.0 {
            stream.read_i32()?
        } else {
            0
        };
        let rgctx_ranges_count = if version >= 108.0 {
            stream.read_u32()? as i32
        } else {
            0
        };
        let static_constructor_start = if version >= 108.0 {
            read_type_def_idx(stream)?
        } else {
            0
        };
        let static_constructor_count = if version >= 108.0 {
            stream.read_u32()? as i32
        } else {
            0
        };
        Ok(Self {
            name_index,
            assembly_index,
            type_start,
            type_count: type_count as i32,
            exported_type_start,
            exported_type_count: exported_type_count as i32,
            entry_point_index,
            token,
            custom_attribute_start,
            custom_attribute_count,
            invoker_indices_start,
            rgctx_ranges_start,
            rgctx_ranges_count,
            static_constructor_start,
            static_constructor_count,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // type_start/exported_type_start/entry_point use variable indices
        let mut size = 4 * 4; // name_index, assembly_index, type_start, type_count
        if version >= 24.0 {
            size += 8;
        } // exported_type_start, exported_type_count
        if version >= 19.0 {
            size += 4;
        } // entry_point_index
        if version >= 19.0 {
            size += 4;
        } // token
        if version >= 24.1 {
            size += 8;
        } // custom_attribute_start, custom_attribute_count
        if version >= 108.0 {
            size += 20;
        } // invokerIndicesStart..staticConstructorCount
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let name_index = stream.read_u32()? as i32;
        let assembly_index = stream.read_i32()?;
        let type_start = stream.read_i32()?;
        let type_count = stream.read_u32()? as i32;
        let entry_point_index = stream.read_i32()?;
        Ok(Self {
            name_index,
            assembly_index,
            type_start,
            type_count,
            exported_type_start: 0,
            exported_type_count: 0,
            entry_point_index,
            token: 1,
            custom_attribute_start: 0,
            custom_attribute_count: 0,
            ..Default::default()
        })
    }

    pub const CODM_BYTE_SIZE: usize = 20;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppAssemblyNameDefinition {
    pub name_index: i32,
    pub culture_index: i32,
    pub hash_value_index: i32,
    pub public_key_index: i32,
    pub hash_alg: i32,
    pub hash_len: i32,
    pub flags: u32,
    pub major: i32,
    pub minor: i32,
    pub build: i32,
    pub revision: i32,
    pub public_key_token: [u8; 8],
}

impl Il2CppAssemblyNameDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let name_index = stream.read_i32()?;
        let culture_index = stream.read_i32()?;
        let hash_value_index = if version <= 24.3 && (version - 24.15).abs() > 0.001 {
            stream.read_i32()?
        } else {
            0
        };
        let public_key_index = stream.read_i32()?;
        let hash_alg = stream.read_i32()?;
        let hash_len = stream.read_i32()?;
        let flags = stream.read_u32()?;
        let major = stream.read_i32()?;
        let minor = stream.read_i32()?;
        let build = stream.read_i32()?;
        let revision = stream.read_i32()?;
        let token_bytes = stream.read_bytes(8)?;
        let mut public_key_token = [0u8; 8];
        public_key_token.copy_from_slice(&token_bytes);
        Ok(Self {
            name_index,
            culture_index,
            hash_value_index,
            public_key_index,
            hash_alg,
            hash_len,
            flags,
            major,
            minor,
            build,
            revision,
            public_key_token,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppAssemblyDefinition {
    pub image_index: i32,
    pub token: u32,
    pub module_token: u32,
    pub custom_attribute_index: i32,
    pub referenced_assembly_start: i32,
    pub referenced_assembly_count: i32,
    pub aname: Il2CppAssemblyNameDefinition,
}

impl Il2CppAssemblyDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let image_index = stream.read_i32()?;
        let token = read_versioned!(stream, version, 24.1, 999.0, read_u32, 0);
        let module_token = read_versioned!(stream, version, 38.0, 999.0, read_u32, 0);
        let custom_attribute_index = read_versioned!(stream, version, 0.0, 24.0, read_i32, 0);
        let referenced_assembly_start = read_versioned!(stream, version, 20.0, 999.0, read_i32, 0);
        let referenced_assembly_count = read_versioned!(stream, version, 20.0, 999.0, read_i32, 0);
        let aname = Il2CppAssemblyNameDefinition::read(stream, version)?;
        Ok(Self {
            image_index,
            token,
            custom_attribute_index,
            referenced_assembly_start,
            referenced_assembly_count,
            aname,
            module_token,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // module_token + aname may differ
        let mut size = 4; // image_index
        if version >= 24.1 {
            size += 4;
        } // token
        if version <= 24.0 {
            size += 4;
        } // custom_attribute_index
        if version >= 20.0 {
            size += 8;
        } // referenced_assembly_start + count
        size += 4 * 10 + 8; // aname fields
        if version <= 24.3 {
            size += 4;
        } // hash_value_index
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let image_index = stream.read_i32()?;
        let custom_attribute_index = stream.read_i16()? as i32;
        let referenced_assembly_start = stream.read_i32()?;
        let referenced_assembly_count = stream.read_i32()?;
        let aname = Il2CppAssemblyNameDefinition::read(stream, 23.0)?;
        Ok(Self {
            image_index,
            token: 0,
            module_token: 0,
            custom_attribute_index,
            referenced_assembly_start,
            referenced_assembly_count,
            aname,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 4 + 2 + 4 + 4 + 4 * 11 + 8;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppTypeDefinition {
    pub name_index: i32,
    pub namespace_index: i32,
    pub custom_attribute_index: i32,
    pub byval_type_index: i32,
    pub byref_type_index: i32,
    pub declaring_type_index: i32,
    pub parent_index: i32,
    pub element_type_index: i32,
    pub rgctx_start_index: i32,
    pub rgctx_count: i32,
    pub generic_container_index: i32,
    pub delegate_wrapper_from_managed_to_native_index: i32,
    pub marshaling_functions_index: i32,
    pub ccw_function_index: i32,
    pub guid_index: i32,
    pub flags: u32,
    pub field_start: i32,
    pub method_start: i32,
    pub event_start: i32,
    pub property_start: i32,
    pub nested_types_start: i32,
    pub interfaces_start: i32,
    pub vtable_start: i32,
    pub interface_offsets_start: i32,
    pub method_count: u16,
    pub property_count: u16,
    pub field_count: u16,
    pub event_count: u16,
    pub nested_type_count: u16,
    pub vtable_count: u16,
    pub interfaces_count: u16,
    pub interface_offsets_count: u16,
    pub bitfield: u32,
    pub token: u32,
}

impl Il2CppTypeDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let name_index = stream.read_i32()?;
        let namespace_index = stream.read_i32()?;
        let custom_attribute_index = read_versioned!(stream, version, 0.0, 24.0, read_i32, 0);
        let byval_type_index = read_type_idx(stream)?;
        let byref_type_index = read_versioned!(stream, version, 0.0, 24.5, read_i32, 0);
        let declaring_type_index = read_type_idx(stream)?;
        let parent_index = read_type_idx(stream)?;
        let element_type_index = if version < 35.0 {
            stream.read_i32()?
        } else {
            -1
        };
        let rgctx_start_index = read_versioned!(stream, version, 0.0, 24.15, read_i32, 0);
        let rgctx_count = read_versioned!(stream, version, 0.0, 24.15, read_i32, 0);
        let generic_container_index = read_gc_idx(stream)?;
        let delegate_wrapper_from_managed_to_native_index =
            read_versioned!(stream, version, 0.0, 22.0, read_i32, 0);
        let marshaling_functions_index = read_versioned!(stream, version, 0.0, 22.0, read_i32, 0);
        let ccw_function_index = read_versioned!(stream, version, 21.0, 22.0, read_i32, 0);
        let guid_index = read_versioned!(stream, version, 21.0, 22.0, read_i32, 0);
        let flags = stream.read_u32()?;
        let field_start = read_field_idx(stream)?;
        let method_start = read_method_idx(stream)?;
        let event_start = read_event_idx(stream)?;
        let property_start = read_property_idx(stream)?;
        let nested_types_start = read_nested_type_idx(stream)?;
        let interfaces_start = read_ioffset_idx(stream)?;
        let vtable_start = stream.read_i32()?;
        let interface_offsets_start = read_ioffset_idx(stream)?;
        let method_count = stream.read_u16()?;
        let property_count = stream.read_u16()?;
        let field_count = stream.read_u16()?;
        let event_count = stream.read_u16()?;
        let nested_type_count = stream.read_u16()?;
        let vtable_count = stream.read_u16()?;
        let interfaces_count = stream.read_u16()?;
        let interface_offsets_count = stream.read_u16()?;
        let bitfield = stream.read_u32()?;
        let token = read_versioned!(stream, version, 19.0, 999.0, read_u32, 0);
        Ok(Self {
            name_index,
            namespace_index,
            custom_attribute_index,
            byval_type_index,
            byref_type_index,
            declaring_type_index,
            parent_index,
            element_type_index,
            rgctx_start_index,
            rgctx_count,
            generic_container_index,
            delegate_wrapper_from_managed_to_native_index,
            marshaling_functions_index,
            ccw_function_index,
            guid_index,
            flags,
            field_start,
            method_start,
            event_start,
            property_start,
            nested_types_start,
            interfaces_start,
            vtable_start,
            interface_offsets_start,
            method_count,
            property_count,
            field_count,
            event_count,
            nested_type_count,
            vtable_count,
            interfaces_count,
            interface_offsets_count,
            bitfield,
            token,
        })
    }

    pub fn is_value_type(&self) -> bool {
        (self.bitfield & 0x1) == 1
    }
    pub fn is_enum(&self) -> bool {
        ((self.bitfield >> 1) & 0x1) == 1
    }
    pub fn has_finalizer(&self) -> bool {
        ((self.bitfield >> 2) & 0x1) == 1
    }
    pub fn has_cctor(&self) -> bool {
        ((self.bitfield >> 3) & 0x1) == 1
    }
    pub fn is_blittable(&self) -> bool {
        ((self.bitfield >> 4) & 0x1) == 1
    }
    pub fn is_import_or_windows_runtime(&self) -> bool {
        ((self.bitfield >> 5) & 0x1) == 1
    }
    pub fn packing_size(&self) -> u32 {
        decode_packing_size((self.bitfield >> 6) & 0xF)
    }
    pub fn packing_size_is_default(&self) -> bool {
        ((self.bitfield >> 10) & 0x1) == 1
    }
    pub fn class_size_is_default(&self) -> bool {
        ((self.bitfield >> 11) & 0x1) == 1
    }
    pub fn specified_packing_size(&self) -> u32 {
        decode_packing_size((self.bitfield >> 12) & 0xF)
    }
    pub fn is_by_ref_like(&self) -> bool {
        ((self.bitfield >> 16) & 0x1) == 1
    }
    pub fn has_inline_array(&self) -> bool {
        ((self.bitfield >> 17) & 0x1) == 1
    }
    pub fn is_abstract(&self) -> bool {
        (self.flags & 0x80) != 0
    }
    pub fn is_interface(&self) -> bool {
        (self.flags & 0x20) != 0
    }
    pub fn is_sealed(&self) -> bool {
        (self.flags & 0x100) != 0
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // variable-size in v38+, use count-based loading
        let mut size = 0usize;
        size += 4 + 4; // name_index, namespace_index
        if version <= 24.0 {
            size += 4;
        } // custom_attribute_index
        size += 4; // byval_type_index
        if version <= 24.5 {
            size += 4;
        } // byref_type_index
        size += 4; // declaring_type_index
        size += 4; // parent_index
        if version < 35.0 {
            size += 4;
        } // element_type_index
        if version <= 24.15 {
            size += 8;
        } // rgctx_start_index, rgctx_count
        size += 4; // generic_container_index
        if version <= 22.0 {
            size += 8;
        } // delegate_wrapper, marshaling
        if version >= 21.0 && version <= 22.0 {
            size += 8;
        } // ccw, guid
        size += 4; // flags
        size += 4 * 8; // field_start..interface_offsets_start (all i32 pre-v38)
        size += 2 * 8; // method_count..interface_offsets_count
        size += 4; // bitfield
        if version >= 19.0 {
            size += 4;
        } // token
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let name_index = stream.read_u32()? as i32;
        let namespace_index = stream.read_u32()? as i32;
        let byval_type_index = stream.read_i32()?;
        let byref_type_index = stream.read_i32()?;
        let declaring_type_index = stream.read_i32()?;
        let parent_index = stream.read_i32()?;
        let element_type_index = stream.read_i32()?;
        let flags = stream.read_u32()?;
        let field_start = stream.read_i32()?;
        let method_start = stream.read_i32()?;
        let vtable_start = stream.read_i32()?;
        let custom_attribute_index = stream.read_i16()? as i32;
        let rgctx_start_index = stream.read_i16()? as i32;
        let rgctx_count = stream.read_i16()? as i32;
        let generic_container_index = stream.read_i16()? as i32;
        let event_start = stream.read_u16()? as i32;
        let property_start = stream.read_u16()? as i32;
        let nested_types_start = stream.read_u16()? as i32;
        let interfaces_start = stream.read_u16()? as i32;
        let interface_offsets_start = stream.read_u16()? as i32;
        let method_count = stream.read_u16()?;
        let property_count = stream.read_u16()?;
        let field_count = stream.read_u16()?;
        let event_count = stream.read_u16()?;
        let nested_type_count = stream.read_u16()?;
        let vtable_count = stream.read_u16()?;
        let interfaces_count = stream.read_u16()?;
        let interface_offsets_count = stream.read_u16()?;
        let bitfield = stream.read_u16()? as u32;
        Ok(Self {
            name_index,
            namespace_index,
            custom_attribute_index,
            byval_type_index,
            byref_type_index,
            declaring_type_index,
            parent_index,
            element_type_index,
            rgctx_start_index,
            rgctx_count,
            generic_container_index,
            delegate_wrapper_from_managed_to_native_index: 0,
            marshaling_functions_index: 0,
            ccw_function_index: 0,
            guid_index: 0,
            flags,
            field_start,
            method_start,
            event_start,
            property_start,
            nested_types_start,
            interfaces_start,
            vtable_start,
            interface_offsets_start,
            method_count,
            property_count,
            field_count,
            event_count,
            nested_type_count,
            vtable_count,
            interfaces_count,
            interface_offsets_count,
            bitfield,
            token: 0,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 80;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppMethodDefinition {
    pub name_index: u32,
    pub declaring_type: i32,
    pub return_type: i32,
    pub return_parameter_token: i32,
    pub parameter_start: i32,
    pub custom_attribute_index: i32,
    pub generic_container_index: i32,
    pub method_index: i32,
    pub invoker_index: i32,
    pub delegate_wrapper_index: i32,
    pub rgctx_start_index: i32,
    pub rgctx_count: i32,
    pub token: u32,
    pub flags: u16,
    pub iflags: u16,
    pub slot: u16,
    pub parameter_count: u16,
}

impl Il2CppMethodDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        Self::read_with_stride(stream, version, Self::byte_size(version))
    }

    /// Read one record at the record size the *bytes* confirmed.
    ///
    /// Unity emits two different v31 records and the version number does not
    /// distinguish them: measured on real files, Unity WebGL 2022.3 has a
    /// 36-byte record (no `methodIndex`) while Free Fire MAX has a 40-byte one
    /// (with it). The record size is the only evidence, so the stride verified
    /// by `layout::detect_method_stride` selects the field layout here.
    ///
    /// Reading a 36-byte record with the 40-byte layout shifts every field
    /// after `returnParameterToken`: `parameterStart` comes out as the constant
    /// that sits in that slot — `0x08000000`, i.e. `count=134217728` (and twice
    /// it, `268435456`, is the number in the reported array-size failure).
    /// Every parameter list and `methodIndex` in such a dump is then wrong.
    pub fn read_with_stride(
        stream: &mut BinaryStream,
        version: f64,
        stride: usize,
    ) -> Result<Self> {
        let name_index = stream.read_u32()?;
        let declaring_type = read_type_def_idx(stream)?;
        let return_type = read_type_idx(stream)?;
        let return_parameter_token = read_versioned!(stream, version, 31.0, 999.0, read_i32, 0);
        let parameter_start = read_param_def_idx(stream)?;
        let custom_attribute_index = read_versioned!(stream, version, 0.0, 24.0, read_i32, 0);
        let generic_container_index = read_gc_idx(stream)?;
        // `method_index` is present in the v24 line (≤ 24.15, this codebase's
        // convention for that range) AND again from v31.0.
        //
        // v31 used to be modelled as "v29 + returnParameterToken" (36 bytes).
        // It is 40: Unity 2020.2 metadata carries `methodIndex` between
        // `genericContainerIndex` and `token`. Measured on a real v31 file
        // (Free Fire MAX), the field is a dense 0..161468 index — every slot
        // occupied, -1 for the 178,717 method defs with no native body — and
        // the 13,608,080-byte methods section divides exactly by 40 only
        // (by 36 it leaves a remainder, which is what desynced every record
        // after the first and produced the out-of-range indices).
        let declared = Self::byte_size(version);
        let v31_line = version >= 31.0 && version < 38.0;
        let has_method_index = if v31_line && declared > 0 && stride > 0 {
            // 40-byte records carry it, 36-byte records do not.
            stride >= declared
        } else {
            version <= 24.15
        };
        let method_index = if has_method_index {
            stream.read_i32()?
        } else {
            0
        };
        let invoker_index = read_versioned!(stream, version, 0.0, 24.15, read_i32, 0);
        let delegate_wrapper_index = read_versioned!(stream, version, 0.0, 24.15, read_i32, 0);
        let rgctx_start_index = read_versioned!(stream, version, 0.0, 24.15, read_i32, 0);
        let rgctx_count = read_versioned!(stream, version, 0.0, 24.15, read_i32, 0);
        let token = stream.read_u32()?;
        let flags = stream.read_u16()?;
        let iflags = stream.read_u16()?;
        let slot = stream.read_u16()?;
        let parameter_count = stream.read_u16()?;
        Ok(Self {
            name_index,
            declaring_type,
            return_type,
            return_parameter_token,
            parameter_start,
            custom_attribute_index,
            generic_container_index,
            method_index,
            invoker_index,
            delegate_wrapper_index,
            rgctx_start_index,
            rgctx_count,
            token,
            flags,
            iflags,
            slot,
            parameter_count,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // variable-size in v38+ (variable-width indices)
        let mut size = 4 + 4 + 4; // name_index, declaring_type, return_type
        if version >= 31.0 {
            size += 4;
        } // return_parameter_token
        size += 4; // parameter_start
        if version <= 24.0 {
            size += 4;
        } // custom_attribute_index
        size += 4; // generic_container_index
        if version <= 24.15 {
            size += 4 * 5; // method_index, invoker_index, delegate_wrapper_index, rgctx_start_index, rgctx_count
        } else if version >= 31.0 && version < 38.0 {
            size += 4; // method_index — v31+ carries it again (see `read`)
        }
        size += 4; // token
        size += 2 * 4; // flags, iflags, slot, parameter_count
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let name_index = stream.read_u32()?;
        let method_index = stream.read_i32()?;
        let return_type = stream.read_i32()?;
        let parameter_start = stream.read_i32()?;
        let token = stream.read_u32()?;
        let declaring_type = stream.read_u16()? as i32;
        let custom_attribute_index = stream.read_i16()? as i32;
        let generic_container_index = stream.read_i16()? as i32;
        let invoker_index = stream.read_u16()? as i32;
        let delegate_wrapper_index = stream.read_i16()? as i32;
        let rgctx_start_index = stream.read_i16()? as i32;
        let rgctx_count = stream.read_u16()? as i32;
        let flags = stream.read_u16()?;
        let iflags = stream.read_u16()?;
        let slot = stream.read_u16()?;
        let parameter_count = stream.read_u16()?;
        Ok(Self {
            name_index,
            declaring_type,
            return_type,
            return_parameter_token: 0,
            parameter_start,
            custom_attribute_index,
            generic_container_index,
            method_index,
            invoker_index,
            delegate_wrapper_index,
            rgctx_start_index,
            rgctx_count,
            token,
            flags,
            iflags,
            slot,
            parameter_count,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 42;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppParameterDefinition {
    pub name_index: i32,
    pub token: u32,
    pub custom_attribute_index: i32,
    pub type_index: i32,
}

impl Il2CppParameterDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let name_index = stream.read_i32()?;
        let token = stream.read_u32()?;
        let custom_attribute_index = read_versioned!(stream, version, 0.0, 24.0, read_i32, 0);
        let type_index = read_type_idx(stream)?;
        Ok(Self {
            name_index,
            token,
            custom_attribute_index,
            type_index,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // type_index is variable
        let mut size = 4 + 4 + 4; // name_index, token, type_index
        if version <= 24.0 {
            size += 4;
        } // custom_attribute_index
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let name_index = stream.read_u32()? as i32;
        let custom_attribute_index = stream.read_i16()? as i32;
        let type_index = stream.read_i32()?;
        Ok(Self {
            name_index,
            token: 0,
            custom_attribute_index,
            type_index,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 10;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppFieldDefinition {
    pub name_index: i32,
    pub type_index: i32,
    pub custom_attribute_index: i32,
    pub token: u32,
}

impl Il2CppFieldDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let name_index = stream.read_i32()?;
        let type_index = read_type_idx(stream)?;
        let custom_attribute_index = read_versioned!(stream, version, 0.0, 24.0, read_i32, 0);
        let token = read_versioned!(stream, version, 19.0, 999.0, read_u32, 0);
        Ok(Self {
            name_index,
            type_index,
            custom_attribute_index,
            token,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // type_index is variable
        let mut size = 4 + 4; // name_index, type_index
        if version <= 24.0 {
            size += 4;
        }
        if version >= 19.0 {
            size += 4;
        }
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let name_index = stream.read_u32()? as i32;
        let type_index = stream.read_i32()?;
        let custom_attribute_index = stream.read_i16()? as i32;
        Ok(Self {
            name_index,
            type_index,
            custom_attribute_index,
            token: 0,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 10;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppFieldDefaultValue {
    pub field_index: i32,
    pub type_index: i32,
    pub data_index: i32,
}

impl Il2CppFieldDefaultValue {
    pub fn read(stream: &mut BinaryStream, _version: f64) -> Result<Self> {
        let field_index = read_field_idx(stream)?;
        let type_index = read_type_idx(stream)?;
        let data_index = read_dvdata_idx(stream)?;
        Ok(Self {
            field_index,
            type_index,
            data_index,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // all three fields are variable
        12
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppParameterDefaultValue {
    pub parameter_index: i32,
    pub type_index: i32,
    pub data_index: i32,
}

impl Il2CppParameterDefaultValue {
    pub fn read(stream: &mut BinaryStream, _version: f64) -> Result<Self> {
        let parameter_index = read_param_def_idx(stream)?;
        let type_index = read_type_idx(stream)?;
        let data_index = read_dvdata_idx(stream)?;
        Ok(Self {
            parameter_index,
            type_index,
            data_index,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // paramIndex (v39+) and typeIndex (v38+) are variable
        12
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppPropertyDefinition {
    pub name_index: i32,
    pub get: i32,
    pub set: i32,
    pub attrs: u32,
    pub custom_attribute_index: i32,
    pub token: u32,
}

impl Il2CppPropertyDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let name_index = stream.read_i32()?;
        let get = read_method_idx(stream)?;
        let set = read_method_idx(stream)?;
        let attrs = stream.read_u32()?;
        let custom_attribute_index = read_versioned!(stream, version, 0.0, 24.0, read_i32, 0);
        let token = read_versioned!(stream, version, 19.0, 999.0, read_u32, 0);
        Ok(Self {
            name_index,
            get,
            set,
            attrs,
            custom_attribute_index,
            token,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 105.0 {
            return 0;
        } // get/set use variable method indices
        let mut size = 4 * 4; // name_index, get, set, attrs
        if version <= 24.0 {
            size += 4;
        }
        if version >= 19.0 {
            size += 4;
        }
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let name_index = stream.read_u32()? as i32;
        let get = stream.read_i16()? as i32;
        let set = stream.read_i16()? as i32;
        let attrs = stream.read_u16()? as u32;
        let custom_attribute_index = stream.read_u16()? as i32;
        Ok(Self {
            name_index,
            get,
            set,
            attrs,
            custom_attribute_index,
            token: 0,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 12;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppEventDefinition {
    pub name_index: i32,
    pub type_index: i32,
    pub add: i32,
    pub remove: i32,
    pub raise: i32,
    pub custom_attribute_index: i32,
    pub token: u32,
}

impl Il2CppEventDefinition {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let name_index = stream.read_i32()?;
        let type_index = read_type_idx(stream)?;
        let add = read_method_idx(stream)?;
        let remove = read_method_idx(stream)?;
        let raise = read_method_idx(stream)?;
        let custom_attribute_index = read_versioned!(stream, version, 0.0, 24.0, read_i32, 0);
        let token = read_versioned!(stream, version, 19.0, 999.0, read_u32, 0);
        Ok(Self {
            name_index,
            type_index,
            add,
            remove,
            raise,
            custom_attribute_index,
            token,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // type_index is variable; method indices variable in v105+
        let mut size = 4 * 5;
        if version <= 24.0 {
            size += 4;
        }
        if version >= 19.0 {
            size += 4;
        }
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let name_index = stream.read_u32()? as i32;
        let type_index = stream.read_i32()?;
        let add = stream.read_i16()? as i32;
        let remove = stream.read_i16()? as i32;
        let raise = stream.read_i16()? as i32;
        let custom_attribute_index = stream.read_i16()? as i32;
        Ok(Self {
            name_index,
            type_index,
            add,
            remove,
            raise,
            custom_attribute_index,
            token: 0,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 16;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppGenericContainer {
    pub owner_index: i32,
    pub type_argc: i32,
    pub is_method: i32,
    pub generic_parameter_start: i32,
}

impl Il2CppGenericContainer {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let owner_index = stream.read_i32()?;
        let (type_argc, is_method) = if version >= 106.0 {
            (stream.read_u16()? as i32, stream.read_u8()? as i32)
        } else {
            (stream.read_i32()?, stream.read_i32()?)
        };
        let generic_parameter_start = read_gp_idx(stream)?;
        Ok(Self {
            owner_index,
            type_argc,
            is_method,
            generic_parameter_start,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 106.0 {
            return 0;
        } // variable due to generic_param_idx + packed fields
        16 // pre-v106: 4+4+4+4
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let owner_index = stream.read_i32()?;
        let generic_parameter_start = stream.read_i32()?;
        let type_argc = stream.read_i16()? as i32;
        let is_method = stream.read_i16()? as i32;
        Ok(Self {
            owner_index,
            type_argc,
            is_method,
            generic_parameter_start,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 12;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppGenericParameter {
    pub owner_index: i32,
    pub name_index: u32,
    pub constraints_start: i16,
    pub constraints_count: i16,
    pub num: u16,
    pub flags: u16,
}

impl Il2CppGenericParameter {
    pub fn read(stream: &mut BinaryStream, _version: f64) -> Result<Self> {
        Ok(Self {
            owner_index: read_gc_idx(stream)?,
            name_index: stream.read_u32()?,
            constraints_start: stream.read_i16()?,
            constraints_count: stream.read_i16()?,
            num: stream.read_u16()?,
            flags: stream.read_u16()?,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // ownerIndex (gc_idx) is variable
        16 // 4+4+2+2+2+2
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let name_index = stream.read_u32()?;
        let owner_index = stream.read_u16()? as i32;
        let constraints_start = stream.read_i16()?;
        let constraints_count = stream.read_i16()?;
        let num = stream.read_u16()?;
        let flags = stream.read_u16()?;
        Ok(Self {
            owner_index,
            name_index,
            constraints_start,
            constraints_count,
            num,
            flags,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 14;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppCustomAttributeTypeRange {
    pub token: u32,
    pub start: i32,
    pub count: i32,
}

impl Il2CppCustomAttributeTypeRange {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        Ok(Self {
            token: read_versioned!(stream, version, 24.1, 999.0, read_u32, 0),
            start: stream.read_i32()?,
            count: stream.read_i32()?,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        let mut size = 8; // start, count
        if version >= 24.1 {
            size += 4;
        } // token
        size
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let start = stream.read_u16()? as i32;
        let count = stream.read_u16()? as i32;
        Ok(Self {
            token: 0,
            start,
            count,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 4;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppCustomAttributeDataRange {
    pub token: u32,
    pub start_offset: u32,
}

impl Il2CppCustomAttributeDataRange {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            token: stream.read_u32()?,
            start_offset: stream.read_u32()?,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppMetadataUsageList {
    pub start: u32,
    pub count: u32,
}

impl Il2CppMetadataUsageList {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            start: stream.read_u32()?,
            count: stream.read_u32()?,
        })
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let start = stream.read_u32()?;
        let count = stream.read_u16()? as u32;
        Ok(Self { start, count })
    }

    pub const CODM_BYTE_SIZE: usize = 6;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppMetadataUsagePair {
    pub destination_index: u32,
    pub encoded_source_index: u32,
}

impl Il2CppMetadataUsagePair {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            destination_index: stream.read_u32()?,
            encoded_source_index: stream.read_u32()?,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppStringLiteral {
    pub length: u32, // 0 for v35+ (length determined from next entry offset)
    pub data_index: u32,
}

impl Il2CppStringLiteral {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let length = if version < 35.0 {
            stream.read_u32()?
        } else {
            0
        };
        let data_index = stream.read_u32()?;
        Ok(Self { length, data_index })
    }

    pub fn byte_size(version: f64) -> usize {
        if version < 35.0 { 8 } else { 4 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppFieldRef {
    pub type_index: i32,
    pub field_index: i32,
}

impl Il2CppFieldRef {
    pub fn read(stream: &mut BinaryStream, _version: f64) -> Result<Self> {
        let type_index = read_type_idx(stream)?;
        let field_index = read_field_idx(stream)?;
        Ok(Self {
            type_index,
            field_index,
        })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // type_index and field_index are variable
        8
    }

    pub fn read_codm(stream: &mut BinaryStream) -> Result<Self> {
        let type_index = stream.read_i32()?;
        let field_index = stream.read_i16()? as i32;
        Ok(Self {
            type_index,
            field_index,
        })
    }

    pub const CODM_BYTE_SIZE: usize = 6;
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppInterfaceOffset {
    pub type_index: i32,
    pub offset: i32,
}

impl Il2CppInterfaceOffset {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        let type_index = read_type_idx(stream)?;
        let offset = stream.read_i32()?;
        Ok(Self { type_index, offset })
    }

    pub fn byte_size(type_index_size: usize) -> usize {
        type_index_size + 4
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppInlineArrayLength {
    pub type_index: i32,
    pub length: i32,
}

impl Il2CppInlineArrayLength {
    pub fn read(stream: &mut BinaryStream, _version: f64) -> Result<Self> {
        let type_index = read_type_idx(stream)?;
        let length = stream.read_i32()?;
        Ok(Self { type_index, length })
    }

    pub fn byte_size(version: f64) -> usize {
        if version >= 38.0 {
            return 0;
        } // type_index is variable
        8
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppRGCTXDefinitionData {
    pub rgctx_data_dummy: i32,
}

impl Il2CppRGCTXDefinitionData {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            rgctx_data_dummy: stream.read_i32()?,
        })
    }

    pub fn method_index(&self) -> i32 {
        self.rgctx_data_dummy
    }

    pub fn type_index(&self) -> i32 {
        self.rgctx_data_dummy
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppRGCTXConstrainedData {
    pub type_index: i32,
    pub encoded_method_index: i32,
}

impl Il2CppRGCTXConstrainedData {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            type_index: stream.read_i32()?,
            encoded_method_index: stream.read_i32()?,
        })
    }

    pub fn method_index(&self) -> i32 {
        self.encoded_method_index
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppRGCTXDefinition {
    pub rgctx_type: i64,
    pub def_data: Option<Il2CppRGCTXDefinitionData>,
    pub constrained_data: Option<Il2CppRGCTXConstrainedData>,
    pub data_va: u64,
}

impl Il2CppRGCTXDefinition {
    /// v108 metadata form. Now that it lives in the metadata file the record is
    /// packed down to five bytes — a single type byte followed by one 32-bit data
    /// word — instead of the binary's type word plus a relocated pointer.
    pub fn read_v108_metadata(stream: &mut BinaryStream) -> Result<Self> {
        let rgctx_type = stream.read_u8()? as i64;
        Ok(Self {
            rgctx_type,
            def_data: Some(Il2CppRGCTXDefinitionData::read(stream)?),
            constrained_data: None,
            data_va: 0,
        })
    }

    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let rgctx_type = if version < 29.0 {
            stream.read_i32()? as i64
        } else {
            stream.read_i64()?
        };

        let mut def_data = None;
        let constrained_data = None;
        let mut data_va = 0u64;

        if version < 27.2 {
            def_data = Some(Il2CppRGCTXDefinitionData::read(stream)?);
        } else {
            data_va = stream.read_ptr()?;
        }

        Ok(Self {
            rgctx_type,
            def_data,
            constrained_data,
            data_va,
        })
    }

    pub fn resolve_data(
        &mut self,
        stream: &mut BinaryStream,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Result<()> {
        if self.data_va == 0 || self.def_data.is_some() || self.constrained_data.is_some() {
            return Ok(());
        }

        let raw_offset = map_vatr(self.data_va)?;
        let saved_pos = stream.position();
        stream.set_position(raw_offset);

        if self.rgctx_type == 5 {
            self.constrained_data = Some(Il2CppRGCTXConstrainedData::read(stream)?);
        } else {
            self.def_data = Some(Il2CppRGCTXDefinitionData::read(stream)?);
        }

        stream.set_position(saved_pos);
        Ok(())
    }

    pub fn method_index(&self) -> i32 {
        if let Some(ref d) = self.def_data {
            d.method_index()
        } else if let Some(ref c) = self.constrained_data {
            c.method_index()
        } else {
            -1
        }
    }

    pub fn type_index(&self) -> i32 {
        if let Some(ref d) = self.def_data {
            d.type_index()
        } else if let Some(ref c) = self.constrained_data {
            c.type_index
        } else {
            -1
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppCodeRegistration {
    pub method_pointers_count: u64,
    pub method_pointers: u64,
    pub delegate_wrappers_from_native_to_managed_count: u64,
    pub delegate_wrappers_from_native_to_managed: u64,
    pub reverse_pinvoke_wrapper_count: u64,
    pub reverse_pinvoke_wrappers: u64,
    pub delegate_wrappers_from_managed_to_native_count: u64,
    pub delegate_wrappers_from_managed_to_native: u64,
    pub marshaling_functions_count: u64,
    pub marshaling_functions: u64,
    pub ccw_marshaling_functions_count: u64,
    pub ccw_marshaling_functions: u64,
    pub generic_method_pointers_count: u64,
    pub generic_method_pointers: u64,
    pub generic_adjustor_thunks: u64,
    pub invoker_pointers_count: u64,
    pub invoker_pointers: u64,
    pub custom_attribute_count: u64,
    pub custom_attribute_generators: u64,
    pub guid_count: u64,
    pub guids: u64,
    pub unresolved_virtual_call_count: u64,
    pub unresolved_virtual_call_pointers: u64,
    pub unresolved_instance_call_pointers: u64,
    pub unresolved_static_call_pointers: u64,
    pub interop_data_count: u64,
    pub interop_data: u64,
    pub windows_runtime_factory_count: u64,
    pub windows_runtime_factory_table: u64,
    pub code_gen_modules_count: u64,
    pub code_gen_modules: u64,
}

/// `unresolvedInstanceCallPointers` / `unresolvedStaticCallPointers`: added in
/// v29.1, then v31.0 backported only the `returnTypeToken` half of that release
/// and not the code-registration half, so v31.0 does *not* have the pair and
/// v31.1 does.
pub fn has_unresolved_indirect_call_pointers(version: f64) -> bool {
    (version >= 29.1 && version < 31.0) || version >= 31.1
}

impl Il2CppCodeRegistration {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        Ok(Self {
            method_pointers_count: if version <= 24.1 {
                stream.read_ptr()?
            } else {
                0
            },
            method_pointers: if version <= 24.1 {
                stream.read_ptr()?
            } else {
                0
            },
            delegate_wrappers_from_native_to_managed_count: if version <= 21.0 {
                stream.read_ptr()?
            } else {
                0
            },
            delegate_wrappers_from_native_to_managed: if version <= 21.0 {
                stream.read_ptr()?
            } else {
                0
            },
            reverse_pinvoke_wrapper_count: if version >= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            reverse_pinvoke_wrappers: if version >= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            delegate_wrappers_from_managed_to_native_count: if version <= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            delegate_wrappers_from_managed_to_native: if version <= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            marshaling_functions_count: if version <= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            marshaling_functions: if version <= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            ccw_marshaling_functions_count: if version >= 21.0 && version <= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            ccw_marshaling_functions: if version >= 21.0 && version <= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            generic_method_pointers_count: stream.read_ptr()?,
            generic_method_pointers: stream.read_ptr()?,
            generic_adjustor_thunks: if version >= 24.5 {
                stream.read_ptr()?
            } else {
                0
            },
            invoker_pointers_count: stream.read_ptr()?,
            invoker_pointers: stream.read_ptr()?,
            custom_attribute_count: if version <= 24.5 {
                stream.read_ptr()?
            } else {
                0
            },
            custom_attribute_generators: if version <= 24.5 {
                stream.read_ptr()?
            } else {
                0
            },
            guid_count: if version >= 21.0 && version <= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            guids: if version >= 21.0 && version <= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            unresolved_virtual_call_count: if version >= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            unresolved_virtual_call_pointers: if version >= 22.0 {
                stream.read_ptr()?
            } else {
                0
            },
            // v31.0's `returnTypeToken` addition was backported to 2021.3.40 without
            // the v29.1 code-registration changes, which is why a 31.1 exists; the
            // pair is absent at exactly v31.0 and present either side of it.
            unresolved_instance_call_pointers: if has_unresolved_indirect_call_pointers(version) {
                stream.read_ptr()?
            } else {
                0
            },
            unresolved_static_call_pointers: if has_unresolved_indirect_call_pointers(version) {
                stream.read_ptr()?
            } else {
                0
            },
            interop_data_count: if version >= 23.0 {
                stream.read_ptr()?
            } else {
                0
            },
            interop_data: if version >= 23.0 {
                stream.read_ptr()?
            } else {
                0
            },
            windows_runtime_factory_count: if version >= 24.3 {
                stream.read_ptr()?
            } else {
                0
            },
            windows_runtime_factory_table: if version >= 24.3 {
                stream.read_ptr()?
            } else {
                0
            },
            code_gen_modules_count: if version >= 24.2 {
                stream.read_ptr()?
            } else {
                0
            },
            code_gen_modules: if version >= 24.2 {
                stream.read_ptr()?
            } else {
                0
            },
        })
    }

    pub fn field_count(version: f64) -> usize {
        let mut count = 0usize;
        if version <= 24.1 {
            count += 2;
        }
        if version <= 21.0 {
            count += 2;
        }
        if version >= 22.0 {
            count += 2;
        }
        if version <= 22.0 {
            count += 4;
        }
        if version >= 21.0 && version <= 22.0 {
            count += 2;
        }
        count += 4;
        if version >= 24.5 {
            count += 1;
        }
        if version <= 24.5 {
            count += 2;
        }
        if version >= 21.0 && version <= 22.0 {
            count += 2;
        }
        if version >= 22.0 {
            count += 2;
        }
        if has_unresolved_indirect_call_pointers(version) {
            count += 2;
        }
        if version >= 23.0 {
            count += 2;
        }
        if version >= 24.3 {
            count += 2;
        }
        if version >= 24.2 {
            count += 2;
        }
        count
    }

    pub fn struct_size(is_32bit: bool, version: f64) -> usize {
        let ptr = if is_32bit { 4 } else { 8 };
        Self::field_count(version) * ptr
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppMetadataRegistration {
    pub generic_classes_count: u64,
    pub generic_classes: u64,
    pub generic_insts_count: u64,
    pub generic_insts: u64,
    pub generic_method_table_count: u64,
    pub generic_method_table: u64,
    pub types_count: u64,
    pub types: u64,
    pub method_specs_count: u64,
    pub method_specs: u64,
    pub method_references_count: u64,
    pub method_references: u64,
    pub field_offsets_count: u64,
    pub field_offsets: u64,
    pub type_definitions_sizes_count: u64,
    pub type_definitions_sizes: u64,
    pub metadata_usages_count: u64,
    pub metadata_usages: u64,
    /// v106.1+. The runtime seeds these metadata-usage slots during initialisation
    /// instead of on first use; the count is pointer-width like `metadataUsagesCount`.
    /// Nothing in the dump pipeline consumes them — they are read (and skipped) so
    /// the struct's layout, and therefore `struct_size`, stays correct.
    pub always_init_metadata_usages_count: u64,
    pub always_init_metadata_usages: u64,
}

impl Il2CppMetadataRegistration {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        Ok(Self {
            generic_classes_count: stream.read_ptr()?,
            generic_classes: stream.read_ptr()?,
            generic_insts_count: stream.read_ptr()?,
            generic_insts: stream.read_ptr()?,
            // v108 (Unity 6.6.0b1) moved the generic method table and the method
            // specs out of this struct and into the metadata file, so on v108+ the
            // two count/pointer pairs are simply absent — not zero-filled. Reading
            // them anyway shifts every following field by two slots, land
            // `fieldOffsets` on this struct's own tail, and make the whole
            // registration look invalid.
            generic_method_table_count: if version < 108.0 {
                stream.read_ptr()?
            } else {
                0
            },
            generic_method_table: if version < 108.0 {
                stream.read_ptr()?
            } else {
                0
            },
            types_count: stream.read_ptr()?,
            types: stream.read_ptr()?,
            method_specs_count: if version < 108.0 {
                stream.read_ptr()?
            } else {
                0
            },
            method_specs: if version < 108.0 {
                stream.read_ptr()?
            } else {
                0
            },
            method_references_count: if version <= 16.0 {
                stream.read_ptr()?
            } else {
                0
            },
            method_references: if version <= 16.0 {
                stream.read_ptr()?
            } else {
                0
            },
            field_offsets_count: stream.read_ptr()?,
            field_offsets: stream.read_ptr()?,
            type_definitions_sizes_count: stream.read_ptr()?,
            type_definitions_sizes: stream.read_ptr()?,
            metadata_usages_count: if version >= 19.0 {
                stream.read_ptr()?
            } else {
                0
            },
            metadata_usages: if version >= 19.0 {
                stream.read_ptr()?
            } else {
                0
            },
            always_init_metadata_usages_count: if version >= 106.1 {
                stream.read_ptr()?
            } else {
                0
            },
            always_init_metadata_usages: if version >= 106.1 {
                stream.read_ptr()?
            } else {
                0
            },
        })
    }

    /// Field count of the struct as laid out in the binary, in pointer-sized
    /// slots: seven count/pointer pairs plus the optional ones. 16 slots on
    /// v19–106, 18 on v106.1/v107, 14 on v108+.
    pub fn field_count(version: f64) -> usize {
        let mut count = 14usize; // 7 pairs of count+pointer
        if version <= 16.0 {
            count += 2;
        } // methodReferences count+pointer
        if version >= 19.0 {
            count += 2;
        } // metadataUsages count+pointer
        if version >= 106.1 {
            count += 2;
        } // alwaysInitMetadataUsages count+pointer
        if version >= 108.0 {
            count -= 4;
        } // genericMethodTable and methodSpecs pairs
        count
    }
}

pub const CODM_TYPE_ENUM_KEY: u8 = 0x35;

#[derive(Debug, Clone, Default)]
pub struct Il2CppType {
    pub datapoint: u64,
    pub bits: u32,
    pub attrs: u32,
    pub type_enum: u8,
    pub num_mods: u8,
    pub byref: u8,
    pub pinned: u8,
    pub valuetype: u8,
}

impl Il2CppType {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        let datapoint = stream.read_ptr()?;
        let bits = stream.read_u32()?;
        let mut t = Self {
            datapoint,
            bits,
            ..Default::default()
        };
        t.attrs = bits & 0xFFFF;
        t.type_enum = ((bits >> 16) & 0xFF) as u8;
        Ok(t)
    }

    pub fn init(&mut self, version: f64) {
        self.attrs = self.bits & 0xFFFF;
        self.type_enum = ((self.bits >> 16) & 0xFF) as u8;
        if version >= 27.2 {
            self.num_mods = ((self.bits >> 24) & 0x1F) as u8;
            self.byref = ((self.bits >> 29) & 1) as u8;
            self.pinned = ((self.bits >> 30) & 1) as u8;
            self.valuetype = (self.bits >> 31) as u8;
        } else {
            self.num_mods = ((self.bits >> 24) & 0x3F) as u8;
            self.byref = ((self.bits >> 30) & 1) as u8;
            self.pinned = (self.bits >> 31) as u8;
        }
    }

    pub fn init_codm(&mut self, version: f64) {
        let b2 = ((self.bits >> 16) & 0xFF) as u8;
        let b3 = ((self.bits >> 24) & 0xFF) as u8;
        let dp_hi = (self.datapoint >> 32) as u32;
        let xor_te = b2 ^ CODM_TYPE_ENUM_KEY;
        let xor_te_valid = matches!(xor_te,
            0x01..=0x16 | 0x18 | 0x19 | 0x1B..=0x21 | 0x40 | 0x41 | 0x45 | 0x55 | 0xFF);
        let cur_te_valid = matches!(b2,
            0x01..=0x16 | 0x18 | 0x19 | 0x1B..=0x21 | 0x40 | 0x41 | 0x45 | 0x55 | 0xFF);
        let marker = (b3 & 0x1F) == (CODM_TYPE_ENUM_KEY & 0x1F)
            || dp_hi == 0x35353535
            || (!cur_te_valid && xor_te_valid);
        if marker {
            self.bits ^= 0x35353535;
            self.datapoint ^= 0x3535353535353535;
        }
        self.init(version);
    }

    pub fn klass_index(&self) -> u64 {
        self.datapoint
    }

    pub fn type_handle(&self) -> u64 {
        self.datapoint
    }

    pub fn type_ptr(&self) -> u64 {
        self.datapoint
    }

    pub fn array(&self) -> u64 {
        self.datapoint
    }

    pub fn generic_parameter_index(&self) -> u64 {
        self.datapoint
    }

    pub fn generic_parameter_handle(&self) -> u64 {
        self.datapoint
    }

    pub fn generic_class(&self) -> u64 {
        self.datapoint
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppGenericClass {
    pub type_definition_index: u64,
    pub type_ptr: u64,
    pub context: Il2CppGenericContext,
    pub cached_class: u64,
}

impl Il2CppGenericClass {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let type_definition_index;
        let type_ptr;
        if version >= 27.0 {
            type_definition_index = 0;
            type_ptr = stream.read_ptr()?;
        } else {
            type_definition_index = stream.read_ptr()?;
            type_ptr = 0;
        }
        let context = Il2CppGenericContext::read(stream)?;
        let cached_class = stream.read_ptr()?;
        Ok(Self {
            type_definition_index,
            type_ptr,
            context,
            cached_class,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppGenericContext {
    pub class_inst: u64,
    pub method_inst: u64,
}

impl Il2CppGenericContext {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            class_inst: stream.read_ptr()?,
            method_inst: stream.read_ptr()?,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppGenericInst {
    pub type_argc: u64,
    pub type_argv: u64,
}

impl Il2CppGenericInst {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            type_argc: stream.read_ptr()?,
            type_argv: stream.read_ptr()?,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppArrayType {
    pub etype: u64,
    pub rank: i32,
    pub numsizes: u8,
    pub numlobounds: u8,
    pub sizes: u64,
    pub lobounds: u64,
}

impl Il2CppArrayType {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        let etype = stream.read_ptr()?;
        let rank = stream.read_u8()?;
        let numsizes = stream.read_u8()?;
        let numlobounds = stream.read_u8()?;
        let _padding = stream.read_u8()?;
        let sizes = stream.read_ptr()?;
        let lobounds = stream.read_ptr()?;
        Ok(Self {
            etype,
            rank: rank as i32,
            numsizes,
            numlobounds,
            sizes,
            lobounds,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppGenericMethodFunctionsDefinitions {
    pub generic_method_index: i32,
    pub indices: Il2CppGenericMethodIndices,
}

impl Il2CppGenericMethodFunctionsDefinitions {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        Ok(Self {
            generic_method_index: stream.read_i32()?,
            indices: Il2CppGenericMethodIndices::read(stream, version)?,
        })
    }

    /// v108 metadata form: variable-width indices into the spec table and the
    /// binary's generic method pointer / invoker tables. `adjustorThunk` is absent
    /// here and lives in the separate "with adjustor" table, so it stays -1
    /// (IL2CPP's own "no thunk" marker).
    pub fn read_v108(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            generic_method_index: read_method_spec_idx(stream)?,
            indices: Il2CppGenericMethodIndices {
                method_index: read_gmptr_idx(stream)?,
                invoker_index: read_invoker_idx(stream)?,
                adjustor_thunk: NULL_INDEX,
            },
        })
    }

    /// `genericMethodFunctionsDefinitionsWithAdjustor`, v108+.
    pub fn read_v108_with_adjustor(stream: &mut BinaryStream) -> Result<Self> {
        let mut entry = Self::read_v108(stream)?;
        entry.indices.adjustor_thunk = read_adjustor_thunk_idx(stream)?;
        Ok(entry)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppGenericMethodIndices {
    pub method_index: i32,
    pub invoker_index: i32,
    pub adjustor_thunk: i32,
}

impl Il2CppGenericMethodIndices {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        Ok(Self {
            method_index: stream.read_i32()?,
            invoker_index: stream.read_i32()?,
            adjustor_thunk: if (version >= 24.5 && (version - 27.0).abs() > 0.001)
                || version >= 27.1
            {
                stream.read_i32()?
            } else {
                0
            },
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppMethodSpec {
    pub method_definition_index: i32,
    pub class_index_index: i32,
    pub method_index_index: i32,
}

/// An index that is not present. Cpp2IL calls the same value `Il2CppVariableWidthIndex.Null`
/// and treats any negative index as absent.
pub const NULL_INDEX: i32 = -1;

impl Il2CppMethodSpec {
    /// The binary form (pre-v108): three fixed 32-bit indices.
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            method_definition_index: stream.read_i32()?,
            class_index_index: stream.read_i32()?,
            method_index_index: stream.read_i32()?,
        })
    }

    // v108 (Unity 6.6.0b1) moved the specs into the metadata as three separate
    // tables that share one index space, in this order: methods that only need
    // class-level generic arguments, then only method-level, then both. Read
    // together they are the previous `methodSpecs` array, which is why the absent
    // half of the first two tables is a null index rather than a zero.

    /// `methodSpecsOnGenericType`: a method definition plus its class generic instantiation.
    pub fn read_v108_on_generic_type(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            method_definition_index: read_method_idx(stream)?,
            class_index_index: read_generic_inst_idx(stream)?,
            method_index_index: NULL_INDEX,
        })
    }

    /// `genericMethodSpecsOnType`: a method definition plus its method generic instantiation.
    pub fn read_v108_spec_on_type(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            method_definition_index: read_method_idx(stream)?,
            class_index_index: NULL_INDEX,
            method_index_index: read_generic_inst_idx(stream)?,
        })
    }

    /// `methodSpecs`: both instantiations.
    pub fn read_v108_full(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            method_definition_index: read_method_idx(stream)?,
            class_index_index: read_generic_inst_idx(stream)?,
            method_index_index: read_generic_inst_idx(stream)?,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppCodeGenModule {
    pub module_name: u64,
    pub method_pointer_count: i64,
    pub method_pointers: u64,
    pub adjustor_thunk_count: u64,
    pub adjustor_thunks: u64,
    pub invoker_indices: u64,
    pub reverse_pinvoke_wrapper_count: u64,
    pub reverse_pinvoke_wrapper_indices: u64,
    pub rgctx_ranges_count: i64,
    pub rgctx_ranges: u64,
    pub rgctxs_count: i64,
    pub rgctxs: u64,
    pub debugger_metadata: u64,
    pub custom_attribute_cache_generator: u64,
    pub module_initializer: u64,
    pub static_constructor_type_indices: u64,
    pub metadata_registration: u64,
    pub code_registration: u64,
}

impl Il2CppCodeGenModule {
    pub fn read(stream: &mut BinaryStream, version: f64) -> Result<Self> {
        let module_name = stream.read_ptr()?;
        let method_pointer_count = stream.read_ptr_signed()?;
        let method_pointers = stream.read_ptr()?;

        let has_adjustor_thunks = (version >= 24.5 && version < 27.0) || version >= 27.1;
        let adjustor_thunk_count = if has_adjustor_thunks {
            stream.read_ptr()?
        } else {
            0
        };
        let adjustor_thunks = if has_adjustor_thunks {
            stream.read_ptr()?
        } else {
            0
        };

        // v108 dropped `invokerIndices`, `rgctxRanges{,Count}`, `rgctxs{,Count}`
        // and `staticConstructorTypeIndices` from the module struct; those
        // tables moved into the global metadata header. Reading them anyway
        // desynchronises every field after them, so gate each on the version
        // exactly as `Il2CppCodeGenModule.Read` does.
        let has_rgctx_and_invoker = version < 108.0;

        let invoker_indices = if has_rgctx_and_invoker {
            stream.read_ptr()?
        } else {
            0
        };
        let reverse_pinvoke_wrapper_count = stream.read_ptr()?;
        let reverse_pinvoke_wrapper_indices = stream.read_ptr()?;
        let rgctx_ranges_count = if has_rgctx_and_invoker {
            stream.read_ptr_signed()?
        } else {
            0
        };
        let rgctx_ranges = if has_rgctx_and_invoker {
            stream.read_ptr()?
        } else {
            0
        };
        let rgctxs_count = if has_rgctx_and_invoker {
            stream.read_ptr_signed()?
        } else {
            0
        };
        let rgctxs = if has_rgctx_and_invoker {
            stream.read_ptr()?
        } else {
            0
        };
        let debugger_metadata = stream.read_ptr()?;

        let mut custom_attribute_cache_generator = 0u64;
        let mut module_initializer = 0u64;
        let mut static_constructor_type_indices = 0u64;
        let mut metadata_registration = 0u64;
        let mut code_registration = 0u64;

        if version >= 27.0 {
            if version < 29.0 {
                custom_attribute_cache_generator = stream.read_ptr()?;
            }
            module_initializer = stream.read_ptr()?;
            if has_rgctx_and_invoker {
                static_constructor_type_indices = stream.read_ptr()?;
            }
            metadata_registration = stream.read_ptr()?;
            code_registration = stream.read_ptr()?;
        }

        Ok(Self {
            module_name,
            method_pointer_count,
            method_pointers,
            adjustor_thunk_count,
            adjustor_thunks,
            invoker_indices,
            reverse_pinvoke_wrapper_count,
            reverse_pinvoke_wrapper_indices,
            rgctx_ranges_count,
            rgctx_ranges,
            rgctxs_count,
            rgctxs,
            debugger_metadata,
            custom_attribute_cache_generator,
            module_initializer,
            static_constructor_type_indices,
            metadata_registration,
            code_registration,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppRange {
    pub start: i32,
    pub length: i32,
}

impl Il2CppRange {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            start: stream.read_i32()?,
            length: stream.read_i32()?,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppTokenRangePair {
    pub token: u32,
    pub range: Il2CppRange,
}

impl Il2CppTokenRangePair {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            token: stream.read_u32()?,
            range: Il2CppRange::read(stream)?,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Il2CppTypeDefinitionSizes {
    pub instance_size: u32,
    pub native_size: i32,
    pub static_fields_size: u32,
    pub thread_static_fields_size: u32,
}

impl Il2CppTypeDefinitionSizes {
    pub fn read(stream: &mut BinaryStream) -> Result<Self> {
        Ok(Self {
            instance_size: stream.read_u32()?,
            native_size: stream.read_i32()?,
            static_fields_size: stream.read_u32()?,
            thread_static_fields_size: stream.read_u32()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every assertion here is the only record size that divides the methods
    /// section of a real file exactly (a remainder means the stride is wrong):
    ///
    ///   v24.1  3,577,964 / 52 =  68,807  (Android libil2cpp, Unity 2018)
    ///   v29    3,240,480 / 32 = 101,265  (iOS UnityFramework)
    ///   v31   13,608,080 / 40 = 340,202  (Free Fire MAX)
    ///
    /// The v31 case is the one that was wrong: modelled as 36 bytes, which
    /// leaves a remainder, so every record after the first decoded shifted by
    /// four bytes and the section turned into out-of-range indices.
    #[test]
    fn method_definition_size_matches_the_real_metadata_sections() {
        assert_eq!(Il2CppMethodDefinition::byte_size(24.1), 52);
        assert_eq!(Il2CppMethodDefinition::byte_size(27.0), 32);
        assert_eq!(Il2CppMethodDefinition::byte_size(29.0), 32);
        assert_eq!(Il2CppMethodDefinition::byte_size(31.0), 40);
        assert_eq!(Il2CppMethodDefinition::byte_size(37.0), 40);
        // v38+ is variable-width; handled by the read-first-element path.
        assert_eq!(Il2CppMethodDefinition::byte_size(38.0), 0);

        // The three real metadata sections, from the files themselves.
        assert_eq!(3_577_964 % Il2CppMethodDefinition::byte_size(24.1), 0);
        assert_eq!(3_240_480 % Il2CppMethodDefinition::byte_size(29.0), 0);
        assert_eq!(13_608_080 % Il2CppMethodDefinition::byte_size(31.0), 0);
        // …and the size this file used to be decoded with does NOT divide it.
        assert_ne!(13_608_080 % 36, 0);
    }

    /// `read` must place every field where `byte_size` says it is: the v31
    /// record is name/declaring/return/returnParameterToken/parameterStart/
    /// genericContainer/methodIndex/token/flags/iflags/slot/parameterCount.
    #[test]
    fn v31_method_definition_read_matches_layout() {
        let mut data = Vec::new();
        let mut push32 = |v: u32| data.extend_from_slice(&v.to_le_bytes());
        push32(0x47); // name_index
        push32(1); // declaring_type
        push32(0x420); // return_type
        push32(0x0800_0001); // return_parameter_token (v31+)
        push32(0xFFFF_FFFF); // parameter_start (-1: no parameters)
        push32(0xFFFF_FFFF); // generic_container_index (-1)
        push32(5); // method_index (v31+ — the field that was missing)
        push32(0x0600_0001); // token
        data.extend_from_slice(&0x0886u16.to_le_bytes()); // flags
        data.extend_from_slice(&0x0000u16.to_le_bytes()); // iflags
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // slot
        data.extend_from_slice(&2u16.to_le_bytes()); // parameter_count
        assert_eq!(data.len(), 40);

        // Two records in a row: the second must still land on its own fields,
        // which is exactly what a wrong stride breaks.
        let mut two = data.clone();
        two.extend_from_slice(&data);
        let mut stream = BinaryStream::new(two);

        let a = Il2CppMethodDefinition::read(&mut stream, 31.0).unwrap();
        assert_eq!(a.name_index, 0x47);
        assert_eq!(a.return_parameter_token, 0x0800_0001);
        assert_eq!(a.parameter_start, -1);
        assert_eq!(a.method_index, 5);
        assert_eq!(a.token, 0x0600_0001);
        assert_eq!(a.flags, 0x0886);
        assert_eq!(a.parameter_count, 2);

        let b = Il2CppMethodDefinition::read(&mut stream, 31.0).unwrap();
        assert_eq!(b.name_index, a.name_index);
        assert_eq!(b.token, a.token);
        assert_eq!(b.method_index, a.method_index);
    }

    /// The old 36-byte stride put the token where `method_index` lives, so the
    /// token pattern is a cheap way to see the desync: reading bytes 24..28 as
    /// the token yields the method index, not a `0x06xxxxxx` method token.
    #[test]
    fn v31_wrong_stride_puts_method_index_where_the_token_belongs() {
        let mut data = Vec::new();
        let mut push32 = |v: u32| data.extend_from_slice(&v.to_le_bytes());
        push32(0x47);
        push32(1);
        push32(0x420);
        push32(0x0800_0001);
        push32(0xFFFF_FFFF);
        push32(0xFFFF_FFFF);
        push32(5); // method_index
        push32(0x0600_0001); // token
        data.extend_from_slice(&[0u8; 8]);

        let wrong = u32::from_le_bytes(data[24..28].try_into().unwrap());
        let right = u32::from_le_bytes(data[28..32].try_into().unwrap());
        assert_eq!(wrong, 5, "bytes 24..28 hold method_index");
        assert_eq!(
            right & 0xFF00_0000,
            0x0600_0000,
            "bytes 28..32 hold the token"
        );
    }

    fn unity(v: &str) -> UnityVersion {
        UnityVersion::parse(v).expect("version parses")
    }

    /// Unity 6 shipped three layouts behind two version numbers. The declared
    /// number alone cannot pick: 6.5.1's "107" is a plain 106 layout, while
    /// 6.6.0a6's "106" is already 106.1 — so the Unity version decides.
    #[test]
    fn v106_107_and_108_resolve_by_unity_version_not_by_number() {
        // The engine's own spelling, and the short form some tooling emits.
        for (uv, want106, want107) in [
            ("6.5.0a6", 106.0, 106.0),
            ("6.5.1f1", 106.0, 106.0),
            ("6.5.0a7", 106.0, 106.0),
            ("6000.6.0a6", 106.1, 106.1),
            ("6000.6.0b1", 106.1, 106.1),
            ("6.6.0a6", 106.1, 106.1),
            ("6.6.0a7", 106.1, 106.1),
            ("6.6.0f1", 106.1, 106.1),
        ] {
            assert_eq!(unity(uv).resolve_sub_version(106), want106, "v106 on {uv}");
            assert_eq!(unity(uv).resolve_sub_version(107), want107, "v107 on {uv}");
        }
        // v108 is a layout of its own; nothing is folded into it.
        assert_eq!(unity("6.6.0b1").resolve_sub_version(108), 108.0);
        assert_eq!(unity("6000.6.0b1").resolve_sub_version(108), 108.0);
        // Earlier versions keep resolving exactly as before.
        assert_eq!(unity("2021.3.0f1").resolve_sub_version(29), 29.0);
        assert_eq!(unity("2022.3.24f1").resolve_sub_version(29), 29.1);
        assert_eq!(unity("2022.3.33f1").resolve_sub_version(31), 31.1);
        assert_eq!(unity("2019.4.21f1").resolve_sub_version(24), 24.5);
    }

    /// Layout of `Il2CppMetadataRegistration`, whose two structural changes are
    /// only visible as a *shift*: v106.1 appends a count/pointer pair, and v108
    /// removes the `genericMethodTable` and `methodSpecs` pairs from the middle.
    /// Reading a v108 struct with the v106 layout therefore lands
    /// `fieldOffsets` two slots late — on this struct's own tail.
    #[test]
    fn registration_layout_shifts_with_v106_1_and_v108() {
        // One distinct marker per pointer-sized slot (all eight/ten slots are
        // 8 bytes here), so every field can be identified by its value.
        let mut data = Vec::new();
        for i in 1..=20u64 {
            data.extend_from_slice(&i.to_le_bytes());
        }
        let mut stream = BinaryStream::new(data);
        stream.is_32bit = false;

        // v29: 16 slots, ending with metadataUsages.
        let mr = Il2CppMetadataRegistration::read(
            &mut {
                let mut s = stream.clone();
                s
            },
            29.0,
        )
        .unwrap();
        assert_eq!(mr.field_offsets_count, 11);
        assert_eq!(mr.field_offsets, 12);
        assert_eq!(mr.type_definitions_sizes_count, 13);
        assert_eq!(mr.metadata_usages_count, 15);
        assert_eq!(mr.always_init_metadata_usages_count, 0);
        assert_eq!(Il2CppMetadataRegistration::field_count(29.0), 16);

        // v106.1: same, plus the alwaysInit pair, so the struct is two slots wider.
        let mut v106 = stream.clone();
        let mr = Il2CppMetadataRegistration::read(&mut v106, 106.1).unwrap();
        assert_eq!(mr.field_offsets_count, 11);
        assert_eq!(mr.metadata_usages_count, 15);
        assert_eq!(mr.always_init_metadata_usages_count, 17);
        assert_eq!(mr.always_init_metadata_usages, 18);
        assert_eq!(Il2CppMetadataRegistration::field_count(106.1), 18);

        // v108: the genericMethodTable pair (slots 5/6) and the methodSpecs pair
        // (9/10) are gone, so everything after them moves up four slots.
        let mut v108 = stream.clone();
        let mr = Il2CppMetadataRegistration::read(&mut v108, 108.0).unwrap();
        assert_eq!(mr.types_count, 5);
        assert_eq!(mr.types, 6);
        assert_eq!(mr.field_offsets_count, 7);
        assert_eq!(mr.field_offsets, 8);
        assert_eq!(mr.type_definitions_sizes_count, 9);
        assert_eq!(mr.type_definitions_sizes, 10);
        assert_eq!(mr.metadata_usages_count, 11);
        assert_eq!(mr.always_init_metadata_usages_count, 13);
        assert_eq!(Il2CppMetadataRegistration::field_count(108.0), 14);
        // The v106 reading puts fieldOffsets where v108 has the sizes pointer:
        // the shift is exactly what makes a wrong-layout read look plausible.
        assert_ne!(mr.field_offsets, 12);
    }

    /// v108 recovers five index widths from the element sizes of the metadata
    /// sections that store the indices, because the tables they point into live
    /// in the binary and have no count in the metadata file.
    #[test]
    fn v108_index_widths_come_from_the_section_element_sizes() {
        let mut h = Il2CppGlobalMetadataHeader {
            methods_count: 100_000, // 4-byte method indices
            // 4,000 spec-table entries of 5 bytes = 4-byte method + 1-byte generic inst
            method_specs_on_generic_type_count: 4_000,
            method_specs_on_generic_type_size: 4_000 * 5,
            // 32,000 spec table entries share one index space -> 2-byte spec index
            generic_method_specs_on_type_count: 28_000,
            generic_method_specs_on_type_size: 28_000 * 6,
            invoker_indices_count: 1_000,
            invoker_indices_size: 1_000 * 2,
            // spec(2) + generic method pointer(?) + invoker(2) = 8
            generic_method_functions_definitions_count: 500,
            generic_method_functions_definitions_size: 500 * 8,
            ..Default::default()
        };
        let widths = IndexWidths::from_header(&h, 108.0);
        assert_eq!(widths.method, 4);
        assert_eq!(widths.generic_inst, 1);
        assert_eq!(widths.method_spec, 2);
        assert_eq!(widths.invoker, 2);
        assert_eq!(widths.generic_method_pointer, 4);
        // With no "with adjustor" section the width falls back to pointer width.
        assert_eq!(widths.adjustor_thunk, 4);

        // The "with adjustor" table adds one more index: 2 + 4 + 2 + 4 = 12 bytes.
        h.generic_method_functions_definitions_with_adjustor_count = 100;
        h.generic_method_functions_definitions_with_adjustor_size = 100 * 12;
        let widths = IndexWidths::from_header(&h, 108.0);
        assert_eq!(widths.adjustor_thunk, 4);

        // Pre-v108 never derives any of them.
        let widths = IndexWidths::from_header(&h, 29.1);
        assert_eq!(widths.generic_inst, 4);
        assert_eq!(widths.method_spec, 4);
    }
}
