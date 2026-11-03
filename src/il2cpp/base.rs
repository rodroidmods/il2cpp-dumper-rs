use super::structures::*;
use crate::disassembler::Architecture;
use crate::error::{Error, Result};
use crate::io::{BinaryStream, checked_array_len};
use crate::search::SearchSection;
use std::collections::{BTreeSet, HashMap};

pub fn auto_plus_count_limit(is_wasm: bool) -> u64 {
    if is_wasm { 0x35000 } else { 0x50000 }
}

/// Move an address by a whole number of pointer slots, in either direction.
fn shift_address(address: u64, slots: i64, ptr_size: u64) -> u64 {
    if slots >= 0 {
        address.saturating_add(slots as u64 * ptr_size)
    } else {
        address.saturating_sub(slots.unsigned_abs() * ptr_size)
    }
}

pub fn apply_auto_plus_heuristics(
    stream: &mut BinaryStream,
    mut code_registration: u64,
    cr: &Il2CppCodeRegistration,
    count_limit: u64,
) -> u64 {
    if code_registration == 0 || stream.version < 24.2 {
        return code_registration;
    }
    let ptr_size = stream.pointer_size() as u64;

    if stream.version == 31.0 {
        if cr.generic_method_pointers_count > count_limit {
            code_registration = code_registration.saturating_sub(ptr_size * 2);
        } else {
            stream.version = 29.0;
            crate::output::note("Change il2cpp version to: 29");
        }
    }
    if stream.version == 29.0 {
        if cr.generic_method_pointers_count > count_limit {
            stream.version = 29.1;
            code_registration = code_registration.saturating_sub(ptr_size * 2);
            crate::output::note("Change il2cpp version to: 29.1");
        }
    }
    if stream.version == 27.0 {
        if cr.reverse_pinvoke_wrapper_count > count_limit {
            stream.version = 27.1;
            code_registration = code_registration.saturating_sub(ptr_size);
            crate::output::note("Change il2cpp version to: 27.1");
        }
    }
    if stream.version == 24.4 {
        code_registration = code_registration.saturating_sub(ptr_size * 2);
        if cr.reverse_pinvoke_wrapper_count > count_limit {
            stream.version = 24.5;
            code_registration = code_registration.saturating_sub(ptr_size);
            crate::output::note("Change il2cpp version to: 24.5");
        }
    }
    if stream.version == 24.2 && cr.interop_data_count == 0 {
        stream.version = 24.3;
        code_registration = code_registration.saturating_sub(ptr_size * 2);
        crate::output::note("Change il2cpp version to: 24.3");
    }
    code_registration
}

pub fn refine_code_registration(
    stream: &mut BinaryStream,
    code_registration: u64,
    map_vatr: &dyn Fn(u64) -> Result<u64>,
    count_limit: u64,
) -> Result<u64> {
    if code_registration == 0 || stream.version < 24.2 {
        return Ok(code_registration);
    }
    let cr_offset = map_vatr(code_registration)?;
    stream.set_position(cr_offset);
    let version = stream.version;
    let cr = Il2CppCodeRegistration::read(stream, version)?;
    Ok(apply_auto_plus_heuristics(
        stream,
        code_registration,
        &cr,
        count_limit,
    ))
}

/// Does an array of `count` pointers have any chance of existing in the image?
///
/// A registration is located by searching the *bytes*, so the struct that is
/// there can be a shifted view of something else — in which case a *pointer*
/// lands in a *count* slot. A real array always fits in the image (a dumped
/// image contains its BSS as zeros, so this holds for dumped files too), so a
/// count that cannot fit proves the count is not a count.
fn count_fits(count: u64, ptr_size: u64, file_len: u64) -> bool {
    count.saturating_mul(ptr_size) <= file_len
}

/// Every count in a registration must name a pointer that exists.
fn check_count_pointer_pairs(
    pairs: &[(&'static str, u64, u64)],
    ptr_size: u64,
    file_len: u64,
) -> std::result::Result<(), String> {
    for (name, count, pointer) in pairs {
        if *count == 0 {
            continue;
        }
        if *pointer == 0 {
            return Err(format!("{name}Count = {count} but {name} is null"));
        }
        if !count_fits(*count, ptr_size, file_len) {
            return Err(format!(
                "{name}Count = {count} (0x{count:x}) needs {} bytes, more than the whole image ({file_len})",
                count.saturating_mul(ptr_size)
            ));
        }
    }
    Ok(())
}

/// Is `cr` a possible `Il2CppCodeRegistration` for this version?
///
/// `Err` carries the exact field that cannot be right, so a caller can reject
/// the candidate *and* say why.
pub fn code_registration_plausibility(
    cr: &Il2CppCodeRegistration,
    version: f64,
    ptr_size: u64,
    file_len: u64,
) -> std::result::Result<(), String> {
    let mut pairs: Vec<(&'static str, u64, u64)> = vec![
        (
            "genericMethodPointers",
            cr.generic_method_pointers_count,
            cr.generic_method_pointers,
        ),
        (
            "invokerPointers",
            cr.invoker_pointers_count,
            cr.invoker_pointers,
        ),
    ];
    if version <= 24.1 {
        pairs.push((
            "methodPointers",
            cr.method_pointers_count,
            cr.method_pointers,
        ));
    }
    if version >= 22.0 {
        // The `name` is the field *family*: the message reads `{name}Count`.
        pairs.push((
            "reversePInvokeWrapper",
            cr.reverse_pinvoke_wrapper_count,
            cr.reverse_pinvoke_wrappers,
        ));
        pairs.push((
            "unresolvedVirtualCall",
            cr.unresolved_virtual_call_count,
            cr.unresolved_virtual_call_pointers,
        ));
    }
    if version >= 23.0 {
        pairs.push(("interopData", cr.interop_data_count, cr.interop_data));
    }
    if version >= 24.2 {
        pairs.push((
            "codeGenModules",
            cr.code_gen_modules_count,
            cr.code_gen_modules,
        ));
    }
    if version >= 24.3 {
        pairs.push((
            "windowsRuntimeFactory",
            cr.windows_runtime_factory_count,
            cr.windows_runtime_factory_table,
        ));
    }
    check_count_pointer_pairs(&pairs, ptr_size, file_len)
}

/// Is `mr` a possible `Il2CppMetadataRegistration` for this version?
pub fn metadata_registration_plausibility(
    mr: &Il2CppMetadataRegistration,
    version: f64,
    ptr_size: u64,
    file_len: u64,
) -> std::result::Result<(), String> {
    let mut pairs: Vec<(&'static str, u64, u64)> = vec![
        (
            "genericClasses",
            mr.generic_classes_count,
            mr.generic_classes,
        ),
        ("genericInsts", mr.generic_insts_count, mr.generic_insts),
        (
            "genericMethodTable",
            mr.generic_method_table_count,
            mr.generic_method_table,
        ),
        ("types", mr.types_count, mr.types),
        ("methodSpecs", mr.method_specs_count, mr.method_specs),
        ("fieldOffsets", mr.field_offsets_count, mr.field_offsets),
        (
            "typeDefinitionsSizes",
            mr.type_definitions_sizes_count,
            mr.type_definitions_sizes,
        ),
    ];
    if version >= 19.0 {
        pairs.push((
            "metadataUsages",
            mr.metadata_usages_count,
            mr.metadata_usages,
        ));
    }
    check_count_pointer_pairs(&pairs, ptr_size, file_len)
}

/// A pointer array whose count came from a located registration and could not
/// possibly exist.
///
/// One impossible count must not abort a whole run: it is recorded, the array
/// is skipped, and everything else is still recovered. When the registration is
/// right this never fires; when it is wrong this is the report that names the
/// field, instead of an anonymous `Invalid array size: count=…`.
#[derive(Debug, Clone)]
pub struct RejectedArray {
    pub field: &'static str,
    pub address: u64,
    pub count: u64,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct VaSegment {
    pub vaddr: u64,
    pub memsz: u64,
    pub offset: u64,
}

pub struct Il2Cpp {
    pub stream: BinaryStream,
    pub version: f64,
    pub is_32bit: bool,

    pub types: Vec<Il2CppType>,
    pub method_pointers: Vec<u64>,
    pub generic_method_pointers: Vec<u64>,
    pub invoker_pointers: Vec<u64>,
    pub field_offsets: Vec<Vec<i32>>,
    pub metadata_usages: Vec<u64>,
    pub custom_attribute_generators: Vec<u64>,

    pub generic_insts: Vec<Il2CppGenericInst>,
    pub generic_inst_pointers: Vec<u64>,
    pub method_specs: Vec<Il2CppMethodSpec>,
    pub generic_method_table: Vec<Il2CppGenericMethodFunctionsDefinitions>,

    pub code_gen_modules: HashMap<String, Il2CppCodeGenModule>,
    pub code_gen_module_method_pointers: HashMap<String, Vec<u64>>,

    pub method_definition_method_specs: HashMap<usize, Vec<usize>>,
    pub method_spec_generic_method_pointers: HashMap<usize, u64>,

    pub code_registration: u64,
    pub metadata_registration: u64,
    pub is_dumped: bool,
    pub image_base: u64,
    /// Assembly names from the metadata. Used to find code-gen modules inside a
    /// memory dump when the section-based search cannot (see `dump_scan`).
    pub image_names: Vec<String>,

    field_offset_pointers: Vec<u64>,
    field_offsets_are_pointers: bool,
    type_dic: HashMap<u64, usize>,
    pub va_segments: Vec<VaSegment>,
    /// Exception-handling tables (try/catch regions), populated per format
    /// at init (Mach-O __unwind_info, ELF .eh_frame, PE .pdata stub).
    pub eh_tables: crate::recovery::exceptions::EhTables,
    pub data_sections: Vec<SearchSection>,
    pub rgctxs_dictionary: HashMap<String, HashMap<u32, Vec<Il2CppRGCTXDefinition>>>,
    pub is_pe: bool,
    pub reverse_pinvoke_wrappers: Vec<u64>,
    pub unresolved_virtual_call_pointers: Vec<u64>,
    pub arch: Option<Architecture>,
    pub e_machine: u16,
    pub exported_symbols: Vec<String>,
    pub api_export_rvas: HashMap<String, u64>,
    /// GOT/lazy-stub slot RVA → imported symbol name (dyld bind tables /
    /// ELF relocations). Authoritative naming for codegen helpers that call
    /// through `__got` stubs — turns `fn_4a58bac` chains into
    /// `___cxa_throw`-style identifications with zero guessing.
    pub bind_symbols: HashMap<u64, String>,
    /// RVA → name of DEFINED `__cxa`-family functions inside the image.
    /// Android .so files statically link libc++, so `__cxa_throw` is an export
    /// of this binary rather than an import with a PLT stub: raise helpers
    /// `BL` it directly and no bind table names the target. This map lets the
    /// raise classifier recognise those terminals by symbol-table name.
    pub cxa_func_rvas: HashMap<u64, String>,
    /// GOT VA → addend for RELATIVE relocations. The addend is the static
    /// field storage address in .bss/.data. Used by recovery to resolve
    /// `*(nint*)(GOT_VA + offset)` patterns to named static fields.
    pub got_relative_relocations: HashMap<u64, u64>,
    pub codm: bool,
    pub type_definition_sizes: Vec<Il2CppTypeDefinitionSizes>,
    /// Arrays that were skipped because their count cannot exist in the image.
    /// Empty for a healthy file; on a mislocated registration it says exactly
    /// which field is impossible instead of failing the whole dump.
    pub rejected_arrays: Vec<RejectedArray>,
}

impl Il2Cpp {
    pub fn new(mut stream: BinaryStream, version: f64, is_32bit: bool) -> Self {
        stream.version = version;
        stream.is_32bit = is_32bit;
        Self {
            stream,
            version,
            is_32bit,
            types: Vec::new(),
            method_pointers: Vec::new(),
            generic_method_pointers: Vec::new(),
            invoker_pointers: Vec::new(),
            field_offsets: Vec::new(),
            metadata_usages: Vec::new(),
            custom_attribute_generators: Vec::new(),
            generic_insts: Vec::new(),
            generic_inst_pointers: Vec::new(),
            method_specs: Vec::new(),
            generic_method_table: Vec::new(),
            code_gen_modules: HashMap::new(),
            code_gen_module_method_pointers: HashMap::new(),
            method_definition_method_specs: HashMap::new(),
            method_spec_generic_method_pointers: HashMap::new(),
            code_registration: 0,
            metadata_registration: 0,
            is_dumped: false,
            image_base: 0,
            image_names: Vec::new(),
            field_offset_pointers: Vec::new(),
            field_offsets_are_pointers: false,
            type_dic: HashMap::new(),
            va_segments: Vec::new(),
            eh_tables: crate::recovery::exceptions::EhTables::None,
            data_sections: Vec::new(),
            rgctxs_dictionary: HashMap::new(),
            is_pe: false,
            reverse_pinvoke_wrappers: Vec::new(),
            unresolved_virtual_call_pointers: Vec::new(),
            arch: None,
            e_machine: 0,
            exported_symbols: Vec::new(),
            api_export_rvas: HashMap::new(),
            bind_symbols: HashMap::new(),
            cxa_func_rvas: HashMap::new(),
            got_relative_relocations: HashMap::new(),
            codm: false,
            type_definition_sizes: Vec::new(),
            rejected_arrays: Vec::new(),
        }
    }

    pub fn from_elf(elf: &crate::formats::elf::Elf) -> Self {
        let method_definition_method_specs: HashMap<usize, Vec<usize>> = elf
            .method_definition_method_specs
            .iter()
            .map(|(k, v)| (*k as usize, v.clone()))
            .collect();
        Self {
            stream: elf.stream.clone(),
            version: elf.stream.version,
            is_32bit: elf.is_32bit,
            types: elf.types.clone(),
            method_pointers: elf.method_pointers.clone(),
            generic_method_pointers: elf.generic_method_pointers.clone(),
            invoker_pointers: elf.invoker_pointers.clone(),
            field_offsets: Vec::new(),
            metadata_usages: elf.metadata_usages.clone(),
            custom_attribute_generators: elf.custom_attribute_generators.clone(),
            generic_insts: elf.generic_insts.clone(),
            generic_inst_pointers: elf.generic_inst_pointers.clone(),
            method_specs: elf.method_specs.clone(),
            generic_method_table: elf.generic_method_table.clone(),
            code_gen_modules: elf.code_gen_modules.clone(),
            code_gen_module_method_pointers: elf.code_gen_module_method_pointers.clone(),
            method_definition_method_specs,
            method_spec_generic_method_pointers: elf.method_spec_generic_method_pointers.clone(),
            code_registration: 0,
            metadata_registration: 0,
            is_dumped: elf.is_dumped,
            image_base: elf.stream.image_base,
            image_names: Vec::new(),
            field_offset_pointers: elf.field_offsets.clone(),
            field_offsets_are_pointers: elf.field_offsets_are_pointers,
            type_dic: elf.type_dic.clone(),
            eh_tables: crate::recovery::exceptions::EhTables::None, // filled by init_elf
            va_segments: elf
                .segments
                .iter()
                .map(|s| VaSegment {
                    vaddr: s.p_vaddr,
                    memsz: s.p_memsz,
                    offset: s.p_offset,
                })
                .collect(),
            data_sections: elf
                .segments
                .iter()
                .filter(|s| s.p_memsz != 0 && matches!(s.p_flags, 2 | 4 | 6))
                .map(|s| {
                    SearchSection::new(
                        s.p_offset,
                        s.p_offset + s.p_filesz,
                        s.p_vaddr,
                        s.p_vaddr + s.p_memsz,
                    )
                })
                .collect(),
            rgctxs_dictionary: elf.rgctxs_dictionary.clone(),
            is_pe: false,
            reverse_pinvoke_wrappers: Vec::new(),
            unresolved_virtual_call_pointers: Vec::new(),
            arch: Architecture::from_elf_machine(elf.header.e_machine),
            e_machine: elf.header.e_machine,
            exported_symbols: Vec::new(),
            api_export_rvas: HashMap::new(),
            bind_symbols: HashMap::new(),
            cxa_func_rvas: HashMap::new(),
            got_relative_relocations: elf.got_relative_relocations.clone(),
            codm: elf.codm_diag,
            type_definition_sizes: elf.type_definition_sizes.clone(),
            rejected_arrays: Vec::new(),
        }
    }

    pub fn init_with_auto_plus(
        &mut self,
        code_registration: u64,
        metadata_registration: u64,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
        is_wasm: bool,
    ) -> Result<()> {
        let orig_version = self.stream.version;
        let limit = auto_plus_count_limit(is_wasm);
        let cr =
            match refine_code_registration(&mut self.stream, code_registration, map_vatr, limit) {
                Ok(c) => c,
                Err(_) => code_registration,
            };
        self.version = self.stream.version;

        // The address the version heuristics hand back is only trusted while a
        // real `Il2CppCodeRegistration` can be sitting there. A heuristic can
        // shift the address onto the wrong field — which reads a *pointer* as a
        // *count*, and a single impossible count used to fail the whole run.
        // FF MAX ships the 31.1 code-registration layout while the metadata
        // says 31; without this the module arrays read as (0,0) and every
        // method pointer becomes -1 (see ffmax-static-dump-31.1-notes.md).
        // The witness runs BEFORE resolution: resolving under the wrong 31.0
        // layout is what emitted the harmless-but-alarming "does not look like
        // a valid struct" warning, and the witness already proves the base.
        let cr = match self.upgrade_v31_1_by_module_witness(cr, map_vatr) {
            Some(adopted) => adopted,
            None => self.resolve_code_registration(cr, map_vatr),
        };

        let result = match self.init(cr, metadata_registration, map_vatr) {
            Ok(()) => Ok(()),
            Err(e) if orig_version == 31.0 && self.stream.version != 29.0 => {
                self.stream.version = 29.0;
                self.version = 29.0;
                crate::output::note(&format!("Retry init with il2cpp version 29 ({e})"));
                self.init(cr, metadata_registration, map_vatr)
            }
            Err(e) => Err(e),
        };
        if result.is_ok() {
            self.report_rejected_arrays();
        }
        result
    }

    pub fn init(
        &mut self,
        code_registration: u64,
        metadata_registration: u64,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Result<()> {
        self.code_registration = code_registration;
        self.metadata_registration = metadata_registration;
        self.stream.is_32bit = self.is_32bit;
        self.version = self.stream.version;

        let mr_offset = self.resolve_metadata_registration(metadata_registration, map_vatr)?;
        let mr = {
            self.stream.set_position(mr_offset);
            Il2CppMetadataRegistration::read(&mut self.stream, self.version)?
        };

        let types_offset = map_vatr(mr.types)?;
        self.stream.set_position(types_offset);
        let type_ptrs = self.stream.read_ptr_array_inline(mr.types_count as usize)?;

        self.types.clear();
        self.type_dic.clear();
        let tolerant = self.codm && self.is_dumped;
        let mut decoded = 0usize;
        let mut skipped = 0usize;
        for (idx, ptr) in type_ptrs.iter().enumerate() {
            if tolerant {
                let t_offset = match map_vatr(*ptr) {
                    Ok(o) => o,
                    Err(_) => {
                        self.types.push(Il2CppType::default());
                        skipped += 1;
                        continue;
                    }
                };
                self.stream.set_position(t_offset);
                let mut t = match Il2CppType::read(&mut self.stream) {
                    Ok(t) => t,
                    Err(_) => {
                        self.types.push(Il2CppType::default());
                        skipped += 1;
                        continue;
                    }
                };
                let pre = t.type_enum;
                t.init_codm(self.version);
                if pre != t.type_enum {
                    decoded += 1;
                }
                self.types.push(t);
                self.type_dic.insert(*ptr, idx);
            } else {
                let t_offset = map_vatr(*ptr).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("type[{}] ptr=0x{:x}: {}", idx, ptr, e),
                    )
                })?;
                self.stream.set_position(t_offset);
                let mut t = Il2CppType::read(&mut self.stream)?;
                if self.codm {
                    t.init_codm(self.version);
                } else {
                    t.init(self.version);
                }
                self.types.push(t);
                self.type_dic.insert(*ptr, idx);
            }
        }
        if tolerant {
            eprintln!(
                "[CODM] init_codm decoded {} of {} Il2CppType entries (skipped {})",
                decoded,
                type_ptrs.len(),
                skipped
            );
        }

        if mr.generic_insts_count > 0 {
            self.generic_inst_pointers = self.read_ptr_array_at(
                "genericInsts",
                mr.generic_insts,
                mr.generic_insts_count,
                map_vatr,
            );

            self.generic_insts.clear();
            for ptr in &self.generic_inst_pointers.clone() {
                let Ok(offset) = map_vatr(*ptr) else { continue };
                self.stream.set_position(offset);
                match Il2CppGenericInst::read(&mut self.stream) {
                    Ok(inst) => self.generic_insts.push(inst),
                    Err(_) => break,
                }
            }
        }

        if mr.method_specs_count > 0 && mr.method_specs > 0 {
            let ms_offset = map_vatr(mr.method_specs)?;
            self.stream.set_position(ms_offset);
            self.method_specs.clear();
            for _ in 0..mr.method_specs_count {
                self.method_specs
                    .push(Il2CppMethodSpec::read(&mut self.stream)?);
            }
        }

        if mr.generic_method_table > 0 && mr.generic_method_table_count > 0 {
            let gmt_offset = map_vatr(mr.generic_method_table)?;
            self.stream.set_position(gmt_offset);
            self.generic_method_table.clear();
            for _ in 0..mr.generic_method_table_count {
                self.generic_method_table
                    .push(Il2CppGenericMethodFunctionsDefinitions::read(
                        &mut self.stream,
                        self.version,
                    )?);
            }
        }

        let cr_offset = map_vatr(code_registration)?;
        self.stream.set_position(cr_offset);
        let cr = Il2CppCodeRegistration::read(&mut self.stream, self.version)?;

        if self.codm {
            if cr.method_pointers > 0 && cr.method_pointers_count > 0 {
                self.method_pointers = self.read_ptr_array_at(
                    "methodPointers",
                    cr.method_pointers,
                    cr.method_pointers_count,
                    map_vatr,
                );
            }

            if cr.generic_method_pointers > 0 && cr.generic_method_pointers_count > 0 {
                self.generic_method_pointers = self.read_ptr_array_at(
                    "genericMethodPointers",
                    cr.generic_method_pointers,
                    cr.generic_method_pointers_count,
                    map_vatr,
                );
            }

            if cr.invoker_pointers > 0 && cr.invoker_pointers_count > 0 {
                self.invoker_pointers = self.read_ptr_array_at(
                    "invokerPointers",
                    cr.invoker_pointers,
                    cr.invoker_pointers_count,
                    map_vatr,
                );
            }

            if self.version < 27.0
                && cr.custom_attribute_generators > 0
                && cr.custom_attribute_count > 0
            {
                self.custom_attribute_generators = self.read_ptr_array_at(
                    "customAttributeGenerators",
                    cr.custom_attribute_generators,
                    cr.custom_attribute_count,
                    map_vatr,
                );
            }

            if mr.metadata_usages > 0 && mr.metadata_usages_count > 0 && self.version < 27.0 {
                self.metadata_usages = self.read_ptr_array_at(
                    "metadataUsages",
                    mr.metadata_usages,
                    mr.metadata_usages_count,
                    map_vatr,
                );
            }

            self.field_offsets_are_pointers = self.version > 21.0;
            if mr.field_offsets > 0 && mr.field_offsets_count > 0 {
                match map_vatr(mr.field_offsets) {
                    Ok(fo_offset) => {
                        if self.field_offsets_are_pointers {
                            self.field_offset_pointers = self.read_ptr_array_field(
                                "fieldOffsets",
                                fo_offset,
                                mr.field_offsets_count,
                            );
                        } else if let Ok(n) = checked_array_len(
                            mr.field_offsets_count,
                            4,
                            Some(self.stream.remaining_from(fo_offset)),
                            "field_offsets(raw)",
                        ) {
                            self.stream.set_position(fo_offset);
                            let mut raw = Vec::with_capacity(n);
                            for _ in 0..n {
                                match self.stream.read_i32() {
                                    Ok(v) => raw.push(v),
                                    Err(_) => break,
                                }
                            }
                            self.field_offsets = vec![raw];
                        } else {
                            eprintln!(
                                "[WARN] field_offsets_count invalid: {}",
                                mr.field_offsets_count
                            );
                        }
                    }
                    Err(e) => eprintln!("[WARN] Failed to map field_offsets: {e}"),
                }
            }

            if mr.type_definitions_sizes > 0 && mr.type_definitions_sizes_count > 0 {
                if let Ok(sizes_offset) = map_vatr(mr.type_definitions_sizes) {
                    if let Ok(n) = checked_array_len(
                        mr.type_definitions_sizes_count,
                        std::mem::size_of::<Il2CppTypeDefinitionSizes>().max(1),
                        Some(self.stream.remaining_from(sizes_offset)),
                        "type_definition_sizes",
                    ) {
                        self.stream.set_position(sizes_offset);
                        self.type_definition_sizes.clear();
                        self.type_definition_sizes.reserve(n);
                        for _ in 0..n {
                            match Il2CppTypeDefinitionSizes::read(&mut self.stream) {
                                Ok(s) => self.type_definition_sizes.push(s),
                                Err(_) => break,
                            }
                        }
                    }
                }
            }
        } else {
            if cr.method_pointers > 0 && cr.method_pointers_count > 0 {
                self.method_pointers = self.read_ptr_array_at(
                    "methodPointers",
                    cr.method_pointers,
                    cr.method_pointers_count,
                    map_vatr,
                );
            }

            if cr.generic_method_pointers > 0 && cr.generic_method_pointers_count > 0 {
                self.generic_method_pointers = self.read_ptr_array_at(
                    "genericMethodPointers",
                    cr.generic_method_pointers,
                    cr.generic_method_pointers_count,
                    map_vatr,
                );
            }

            if cr.invoker_pointers > 0 && cr.invoker_pointers_count > 0 {
                self.invoker_pointers = self.read_ptr_array_at(
                    "invokerPointers",
                    cr.invoker_pointers,
                    cr.invoker_pointers_count,
                    map_vatr,
                );
            }

            if self.version < 27.0
                && cr.custom_attribute_generators > 0
                && cr.custom_attribute_count > 0
            {
                self.custom_attribute_generators = self.read_ptr_array_at(
                    "customAttributeGenerators",
                    cr.custom_attribute_generators,
                    cr.custom_attribute_count,
                    map_vatr,
                );
            }

            if mr.metadata_usages > 0 && mr.metadata_usages_count > 0 && self.version < 27.0 {
                self.metadata_usages = self.read_ptr_array_at(
                    "metadataUsages",
                    mr.metadata_usages,
                    mr.metadata_usages_count,
                    map_vatr,
                );
            }

            self.field_offsets_are_pointers = self.version > 21.0;
            if mr.field_offsets > 0 && mr.field_offsets_count > 0 {
                let fo_offset = map_vatr(mr.field_offsets)?;
                if self.field_offsets_are_pointers {
                    self.field_offset_pointers = self.read_ptr_array_field(
                        "fieldOffsets",
                        fo_offset,
                        mr.field_offsets_count,
                    );
                } else {
                    match checked_array_len(
                        mr.field_offsets_count,
                        4,
                        Some(self.stream.remaining_from(fo_offset)),
                        "fieldOffsets",
                    ) {
                        Ok(n) => {
                            self.stream.set_position(fo_offset);
                            let mut raw = Vec::with_capacity(n);
                            for _ in 0..n {
                                match self.stream.read_i32() {
                                    Ok(v) => raw.push(v),
                                    Err(_) => break,
                                }
                            }
                            self.field_offsets = vec![raw];
                        }
                        Err(e) => self.rejected_arrays.push(RejectedArray {
                            field: "fieldOffsets",
                            address: fo_offset,
                            count: mr.field_offsets_count,
                            detail: e.to_string(),
                        }),
                    }
                }
            }

            if mr.type_definitions_sizes > 0 && mr.type_definitions_sizes_count > 0 {
                let sizes_offset = map_vatr(mr.type_definitions_sizes)?;
                let pointers = self.read_ptr_array_field(
                    "typeDefinitionsSizes",
                    sizes_offset,
                    mr.type_definitions_sizes_count,
                );
                let mapped: Vec<Option<u64>> = pointers
                    .iter()
                    .map(|&ptr| if ptr == 0 { None } else { map_vatr(ptr).ok() })
                    .collect();
                self.type_definition_sizes = read_type_definition_sizes(&mut self.stream, &mapped);
            }
        }

        self.build_method_spec_lookup();

        if self.version >= 24.2 {
            self.load_code_gen_modules(&cr, map_vatr)?;

            // A raw memory dump can hold its registrations in memory the ELF
            // segments do not describe — the on-disk `.so` is a stub and the
            // real code plus its tables are unpacked at runtime. When the
            // section-based search produced no modules, find them from the
            // data itself so method addresses are not silently lost: an empty
            // module map is what makes every method print `RVA: -1`.
            if self.code_gen_module_method_pointers.is_empty() && self.image_base != 0 {
                let names = self.image_names.clone();
                let n = self.locate_code_gen_modules_in_dump(&names);
                if n > 0 {
                    eprintln!("Located {n} code-gen modules by name in the dump's runtime region");
                }
            }
        }

        Ok(())
    }

    /// Read a pointer array whose count and address came from a *located*
    /// registration.
    ///
    /// A single impossible count must not abort a whole dump: it is recorded in
    /// `rejected_arrays`, the array is skipped, and the rest of the run — every
    /// type, method and field — is still recovered. When the registration is
    /// right this never fires.
    fn read_ptr_array_field(
        &mut self,
        field: &'static str,
        mapped_offset: u64,
        count: u64,
    ) -> Vec<u64> {
        if count == 0 {
            return Vec::new();
        }
        let ptr_size = self.stream.pointer_size();
        match checked_array_len(
            count,
            ptr_size,
            Some(self.stream.remaining_from(mapped_offset)),
            field,
        ) {
            Ok(n) => {
                self.stream.set_position(mapped_offset);
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    match self.stream.read_ptr() {
                        Ok(p) => out.push(p),
                        Err(_) => break,
                    }
                }
                out
            }
            Err(e) => {
                self.rejected_arrays.push(RejectedArray {
                    field,
                    address: mapped_offset,
                    count,
                    detail: e.to_string(),
                });
                Vec::new()
            }
        }
    }

    /// `read_ptr_array_field` for a runtime *address*: the address is mapped
    /// first, and an address that is not mapped is reported the same way as an
    /// impossible count rather than failing the run.
    fn read_ptr_array_at(
        &mut self,
        field: &'static str,
        address: u64,
        count: u64,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Vec<u64> {
        if count == 0 {
            return Vec::new();
        }
        match map_vatr(address) {
            Ok(offset) => self.read_ptr_array_field(field, offset, count),
            Err(e) => {
                self.rejected_arrays.push(RejectedArray {
                    field,
                    address,
                    count,
                    detail: format!("array address 0x{address:x} is not mapped: {e}"),
                });
                Vec::new()
            }
        }
    }

    /// Do the first entries of a pointer array actually look like addresses?
    ///
    /// This is the evidence that separates the real struct from a shifted view
    /// of some other data: a genuine table of code/data pointers has non-zero
    /// entries that all map into the image. Null entries are allowed (a method
    /// without a native body legitimately has one), so only a table with no
    /// live entry at all is treated as no evidence.
    fn sample_looks_like_pointers(
        stream: &mut BinaryStream,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
        start: u64,
        count: u64,
        ptr_size: u64,
        file_len: u64,
    ) -> bool {
        let n = count.min(8);
        if n == 0 {
            return true;
        }
        if start.saturating_add(n.saturating_mul(ptr_size)) > file_len {
            return false;
        }
        stream.set_position(start);
        let mut live = 0u32;
        for _ in 0..n {
            let Ok(p) = stream.read_ptr() else {
                return false;
            };
            if p == 0 {
                continue;
            }
            match map_vatr(p) {
                Ok(off) if off < file_len => live += 1,
                _ => return false,
            }
        }
        live > 0
    }

    /// How much evidence is there that `mr` is the real registration?
    ///
    /// `None` means it cannot be one at all; otherwise the number of its arrays
    /// whose contents look like pointer tables.
    fn score_metadata_registration(
        &mut self,
        mr: &Il2CppMetadataRegistration,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Option<u32> {
        let ptr_size = self.stream.pointer_size() as u64;
        let file_len = self.stream.len();
        metadata_registration_plausibility(mr, self.version, ptr_size, file_len).ok()?;
        let mut score = 0u32;
        for (addr, count) in [
            (mr.types, mr.types_count),
            (mr.type_definitions_sizes, mr.type_definitions_sizes_count),
            (mr.generic_insts, mr.generic_insts_count),
            (mr.generic_classes, mr.generic_classes_count),
        ] {
            if addr == 0 || count == 0 {
                continue;
            }
            let Ok(off) = map_vatr(addr) else { continue };
            if Self::sample_looks_like_pointers(
                &mut self.stream,
                map_vatr,
                off,
                count,
                ptr_size,
                file_len,
            ) {
                score += 1;
            }
        }
        Some(score)
    }

    /// The CodeRegistration twin of `score_metadata_registration`.
    fn score_code_registration(
        &mut self,
        cr: &Il2CppCodeRegistration,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Option<u32> {
        let ptr_size = self.stream.pointer_size() as u64;
        let file_len = self.stream.len();
        code_registration_plausibility(cr, self.version, ptr_size, file_len).ok()?;
        let mut score = 0u32;
        for (addr, count) in [
            (cr.method_pointers, cr.method_pointers_count),
            (cr.generic_method_pointers, cr.generic_method_pointers_count),
            (cr.invoker_pointers, cr.invoker_pointers_count),
            (
                cr.reverse_pinvoke_wrappers,
                cr.reverse_pinvoke_wrapper_count,
            ),
        ] {
            if addr == 0 || count == 0 {
                continue;
            }
            let Ok(off) = map_vatr(addr) else { continue };
            if Self::sample_looks_like_pointers(
                &mut self.stream,
                map_vatr,
                off,
                count,
                ptr_size,
                file_len,
            ) {
                score += 1;
            }
        }
        Some(score)
    }

    /// The offsets, in pointer slots, worth checking around a located
    /// registration. The version heuristics shift by one or two slots, so a
    /// small neighbourhood covers every layout guess they make.
    fn candidate_shifts() -> Vec<i64> {
        let mut shifts = vec![0i64];
        for d in 1..=4i64 {
            shifts.push(d);
            shifts.push(-d);
        }
        shifts
    }

    /// Map a located `MetadataRegistration` and check that the struct which is
    /// actually there can be one.
    ///
    /// The bytes-based search hands back the address of a *count field* in some
    /// layouts, and one field of slack is enough to read a pointer as a count —
    /// the `count=268435456` failure. Rather than trusting the address, the
    /// candidate is verified and, when it cannot be a registration, its
    /// neighbours are tried. The located address is kept when it is at least as
    /// good as everything around it, so a healthy file never moves.
    fn resolve_metadata_registration(
        &mut self,
        address: u64,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Result<u64> {
        let ptr_size = self.stream.pointer_size() as u64;
        let file_len = self.stream.len();
        let fallback = map_vatr(address)?;
        let mut best: Option<(u32, i64, u64)> = None;
        let mut first_reason = String::new();

        for shift in Self::candidate_shifts() {
            let cand = shift_address(address, shift, ptr_size);
            let Ok(offset) = map_vatr(cand) else { continue };
            if offset >= file_len {
                continue;
            }
            self.stream.set_position(offset);
            let Ok(mr) = Il2CppMetadataRegistration::read(&mut self.stream, self.version) else {
                continue;
            };
            match self.score_metadata_registration(&mr, map_vatr) {
                Some(score) => {
                    let better = match best {
                        None => true,
                        Some((best_score, best_shift, _)) => {
                            score > best_score
                                || (score == best_score && shift.abs() < best_shift.abs())
                        }
                    };
                    if better {
                        best = Some((score, shift, offset));
                    }
                }
                None => {
                    if first_reason.is_empty() {
                        first_reason = metadata_registration_plausibility(
                            &mr,
                            self.version,
                            ptr_size,
                            file_len,
                        )
                        .err()
                        .unwrap_or_default();
                    }
                }
            }
        }

        match best {
            Some((score, shift, offset)) => {
                if shift != 0 {
                    eprintln!(
                        "[WARN] MetadataRegistration 0x{address:x} is not a valid struct ({first_reason}); \
                         using 0x{:x} ({shift:+} pointer slots), where {score} of its arrays read as pointer tables",
                        shift_address(address, shift, ptr_size)
                    );
                }
                Ok(offset)
            }
            None => {
                eprintln!(
                    "[WARN] MetadataRegistration 0x{address:x} does not look like a valid struct ({first_reason}); \
                     continuing — impossible arrays are reported and skipped"
                );
                Ok(fallback)
            }
        }
    }

    /// The CodeRegistration twin of `resolve_metadata_registration`, applied to
    /// the address the version heuristics produced.
    /// Exact-ish witness for the FF MAX-style v31.1 layout (see
    /// `ffmax-static-dump-31.1-notes.md` §8.1). Metadata only says "31", and
    /// the 31.0 layout reads the codeGenModules pair two slots early — at the
    /// true 31.1 base that yields `(0, 0)` and every method becomes -1.
    ///
    /// Under 31.1 the module count is the metadata image count; when the image
    /// count is not known yet (it is populated after init in most formats) the
    /// silent-failure signature itself is the witness: the 31.0 view has
    /// `codeGenModulesCount == 0` and the 31.1 view has a small non-zero count
    /// whose array is an in-image pointer array. The address is kept; only the
    /// version is upgraded before `init` reads the module arrays.
    fn upgrade_v31_1_by_module_witness(
        &mut self,
        code_registration: u64,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Option<u64> {
        if self.stream.version != 31.0 {
            return None;
        }
        let ptr = self.stream.pointer_size() as u64;
        let f310 = Il2CppCodeRegistration::field_count(31.0) as u64;
        let f311 = Il2CppCodeRegistration::field_count(31.1) as u64;
        let dbg = std::env::var("ILR_31DBG").is_ok();
        // Try the resolved base first, then the layout shifts that separate the
        // 31.0 and 31.1 views of the same words (the pair is two slots wide).
        for slots in [0i64, -2, 2, -1, 1] {
            let cand = if slots >= 0 {
                code_registration.saturating_add(slots as u64 * ptr)
            } else {
                code_registration.saturating_sub(slots.unsigned_abs() * ptr)
            };
            let read_slot = |stream: &mut BinaryStream, addr: u64| -> Result<u64> {
                stream.set_position(map_vatr(addr)?);
                stream.read_ptr()
            };
            let Ok(count310) =
                read_slot(&mut self.stream, cand + (f310 - 2) * ptr)
            else {
                continue;
            };
            let Ok(count311) =
                read_slot(&mut self.stream, cand + (f311 - 2) * ptr)
            else {
                continue;
            };
            if !v31_1_module_witness(self.image_names.len() as u64, count310, count311) {
                if dbg {
                    eprintln!(
                        "[31dbg] cand={cand:#x} slots={slots} img={} c310={count310} c311={count311} rejected",
                        self.image_names.len()
                    );
                }
                continue;
            }
            let Ok(modules) =
                read_slot(&mut self.stream, cand + (f311 - 1) * ptr)
            else {
                continue;
            };
            if modules == 0 {
                continue;
            }
            let mut ok = true;
            for k in 0..3u64.min(count311) {
                match read_slot(&mut self.stream, modules + k * ptr) {
                    Ok(p) if p != 0 && map_vatr(p).is_ok() => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            self.stream.version = 31.1;
            crate::output::note("Detected il2cpp version 31.1 (codeGenModules witness)");
            if dbg {
                eprintln!(
                    "[31dbg] ACCEPT cand={cand:#x} slots={slots} c310={count310} c311={count311} modules={modules:#x}"
                );
            }
            return Some(cand);
        }
        None
    }

    fn resolve_code_registration(
        &mut self,
        address: u64,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> u64 {
        let ptr_size = self.stream.pointer_size() as u64;
        let file_len = self.stream.len();
        let mut best: Option<(u32, i64, u64)> = None;
        let mut first_reason = String::new();

        for shift in Self::candidate_shifts() {
            let cand = shift_address(address, shift, ptr_size);
            let Ok(offset) = map_vatr(cand) else { continue };
            if offset >= file_len {
                continue;
            }
            self.stream.set_position(offset);
            let Ok(cr) = Il2CppCodeRegistration::read(&mut self.stream, self.version) else {
                continue;
            };
            match self.score_code_registration(&cr, map_vatr) {
                Some(score) => {
                    let better = match best {
                        None => true,
                        Some((best_score, best_shift, _)) => {
                            score > best_score
                                || (score == best_score && shift.abs() < best_shift.abs())
                        }
                    };
                    if better {
                        best = Some((score, shift, cand));
                    }
                }
                None => {
                    if first_reason.is_empty() {
                        first_reason =
                            code_registration_plausibility(&cr, self.version, ptr_size, file_len)
                                .err()
                                .unwrap_or_default();
                    }
                }
            }
        }

        match best {
            Some((score, shift, cand)) if shift != 0 => {
                eprintln!(
                    "[WARN] CodeRegistration 0x{address:x} is not a valid struct ({first_reason}); \
                     using 0x{cand:x} ({shift:+} pointer slots), where {score} of its arrays read as pointer tables"
                );
                cand
            }
            Some((_, _, _)) => address,
            None => {
                eprintln!(
                    "[WARN] CodeRegistration 0x{address:x} does not look like a valid struct ({first_reason}); \
                     continuing — impossible arrays are reported and skipped"
                );
                address
            }
        }
    }

    /// Print what had to be skipped, so a partial dump is never mistaken for a
    /// complete one.
    fn report_rejected_arrays(&self) {
        if self.rejected_arrays.is_empty() {
            return;
        }
        eprintln!(
            "[WARN] {} array(s) could not be loaded from the located registrations:",
            self.rejected_arrays.len()
        );
        for r in self.rejected_arrays.iter().take(8) {
            eprintln!("[WARN]   {} @0x{:x}: {}", r.field, r.address, r.detail);
        }
        if self.rejected_arrays.len() > 8 {
            eprintln!("[WARN]   … and {} more", self.rejected_arrays.len() - 8);
        }
        eprintln!(
            "[WARN] A count that cannot exist in the image usually means the located \
             registration is wrong (packed or obfuscated build). The dump continues; \
             method addresses and generic names that depend on the missing arrays will be absent."
        );
    }

    /// Index `method_specs` and `generic_method_table` by method definition.
    ///
    /// Both tables are read during `init_*`, but on v108 they do not live in the
    /// binary at all — `Metadata::apply_v108_tables` overlays the metadata-side
    /// ones afterwards and calls this again, so the caller must be able to reach
    /// it.
    pub fn build_method_spec_lookup(&mut self) {
        self.method_definition_method_specs.clear();
        self.method_spec_generic_method_pointers.clear();

        for (table_idx, table) in self.generic_method_table.iter().enumerate() {
            if (table.generic_method_index as usize) < self.method_specs.len() {
                let ms = &self.method_specs[table.generic_method_index as usize];
                let method_def_idx = ms.method_definition_index as usize;

                self.method_definition_method_specs
                    .entry(method_def_idx)
                    .or_default()
                    .push(table.generic_method_index as usize);

                let method_idx = table.indices.method_index as usize;
                if method_idx < self.generic_method_pointers.len() {
                    self.method_spec_generic_method_pointers.insert(
                        table.generic_method_index as usize,
                        self.generic_method_pointers[method_idx],
                    );
                }
            }
            let _ = table_idx;
        }
    }

    /// Find code-gen modules directly in a memory dump, by assembly name.
    ///
    /// Every module starts with a pointer to its assembly name, and the
    /// metadata lists those names, so each module is reachable without knowing
    /// where the registration tables are. Each hit is confirmed by the name at
    /// the far end of the pointer, and the method-pointer array is checked to
    /// look like addresses before it is believed. Returns how many modules
    /// were accepted.
    pub fn locate_code_gen_modules_in_dump(&mut self, names: &[String]) -> usize {
        if names.is_empty() {
            return 0;
        }
        let base = self.image_base;
        let located =
            crate::il2cpp::dump_scan::locate_modules_by_name(self.stream.data(), base, names);
        let mut accepted = 0usize;
        for module in located {
            let Some(offset) = module.method_pointers.checked_sub(base) else {
                continue;
            };
            if offset + 8 > self.stream.len() {
                continue;
            }
            self.stream.set_position(offset);
            let Ok(ptrs) = self
                .stream
                .read_ptr_array_inline(module.method_pointer_count as usize)
            else {
                continue;
            };
            self.code_gen_module_method_pointers
                .insert(module.name.clone(), ptrs);
            accepted += 1;
        }
        accepted
    }

    fn load_code_gen_modules(
        &mut self,
        cr: &Il2CppCodeRegistration,
        map_vatr: &dyn Fn(u64) -> Result<u64>,
    ) -> Result<()> {
        if cr.code_gen_modules == 0 || cr.code_gen_modules_count == 0 {
            return Ok(());
        }

        let modules_offset = map_vatr(cr.code_gen_modules)?;
        let module_ptrs = self
            .stream
            .read_ptr_array(modules_offset, cr.code_gen_modules_count as usize)?;

        for ptr in module_ptrs {
            let mod_offset = map_vatr(ptr)?;
            self.stream.set_position(mod_offset);
            let module = Il2CppCodeGenModule::read(&mut self.stream, self.version)?;

            let name_offset = map_vatr(module.module_name)?;
            let module_name = self.stream.read_string_to_null_at(name_offset)?;

            let method_ptrs = if module.method_pointers > 0 && module.method_pointer_count > 0 {
                let mp_offset = map_vatr(module.method_pointers)?;
                match self
                    .stream
                    .read_ptr_array(mp_offset, module.method_pointer_count as usize)
                {
                    Ok(v) => v,
                    Err(_) => {
                        let n = module.method_pointer_count as u64;
                        let max = self.stream.len() / self.stream.pointer_size() as u64;
                        if n > 0 && n <= max && n <= crate::io::MAX_BINARY_ARRAY_ELEMS as u64 {
                            vec![0u64; n as usize]
                        } else {
                            Vec::new()
                        }
                    }
                }
            } else {
                Vec::new()
            };

            self.code_gen_module_method_pointers
                .insert(module_name.clone(), method_ptrs);
            self.code_gen_modules.insert(module_name, module);
        }

        Ok(())
    }

    pub fn get_method_pointer(&self, image_name: &str, method_def: &Il2CppMethodDefinition) -> u64 {
        if self.version >= 24.2 {
            if let Some(ptrs) = self.code_gen_module_method_pointers.get(image_name) {
                let method_pointer_index = (method_def.token & 0x00FFFFFF) as usize;
                if method_pointer_index > 0 && method_pointer_index <= ptrs.len() {
                    return ptrs[method_pointer_index - 1];
                }
            }
            0
        } else {
            let idx = method_def.method_index as usize;
            if idx < self.method_pointers.len() {
                self.method_pointers[idx]
            } else {
                0
            }
        }
    }

    pub fn get_raw_field_offset(
        &self,
        type_def_index: usize,
        field_index_in_type: usize,
        field_index: usize,
    ) -> i32 {
        if self.field_offsets_are_pointers {
            if type_def_index >= self.field_offset_pointers.len() {
                return -1;
            }
            let ptr = self.field_offset_pointers[type_def_index];
            if ptr == 0 {
                return -1;
            }
            let target_va = ptr + (field_index_in_type as u64 * 4);
            let read_pos = match self.map_vatr(target_va) {
                Ok(p) => p,
                Err(_) => return -1,
            };
            self.stream.peek_i32_at(read_pos).unwrap_or(-1)
        } else if !self.field_offsets.is_empty() {
            let flat = &self.field_offsets[0];
            if field_index < flat.len() {
                flat[field_index]
            } else {
                -1
            }
        } else {
            -1
        }
    }

    pub fn get_field_offset_from_index(
        &self,
        type_def_index: usize,
        field_index_in_type: usize,
        field_index: usize,
        is_value_type: bool,
        is_static: bool,
    ) -> i32 {
        let raw = self.get_raw_field_offset(type_def_index, field_index_in_type, field_index);
        let decoded = super::field_layout::decode_field_offset(raw);
        if raw < 0 && !decoded.is_thread_static {
            return raw;
        }
        let offset = decoded.effective;

        if offset > 0 && is_value_type && !is_static {
            let header = if self.is_32bit { 8 } else { 16 };
            return offset - header;
        }

        offset
    }

    pub fn peek_generic_class(&self, addr: u64) -> Result<Il2CppGenericClass> {
        let offset = self.map_vatr(addr)?;
        let mut r = self.stream.slice_reader_at(offset);
        let type_definition_index;
        let type_ptr;
        if self.version >= 27.0 {
            type_definition_index = 0;
            type_ptr = r.read_ptr()?;
        } else {
            type_definition_index = r.read_ptr()?;
            type_ptr = 0;
        }
        let class_inst = r.read_ptr()?;
        let method_inst = r.read_ptr()?;
        let cached_class = r.read_ptr()?;
        Ok(Il2CppGenericClass {
            type_definition_index,
            type_ptr,
            context: Il2CppGenericContext {
                class_inst,
                method_inst,
            },
            cached_class,
        })
    }

    pub fn peek_generic_inst(&self, addr: u64) -> Result<Il2CppGenericInst> {
        let offset = self.map_vatr(addr)?;
        let mut r = self.stream.slice_reader_at(offset);
        Ok(Il2CppGenericInst {
            type_argc: r.read_ptr()?,
            type_argv: r.read_ptr()?,
        })
    }

    pub fn peek_ptr_array(&self, addr: u64, count: u64) -> Result<Vec<u64>> {
        let offset = self.map_vatr(addr)?;
        let mut r = self.stream.slice_reader_at(offset);
        let n = crate::io::checked_array_len(
            count,
            self.stream.pointer_size(),
            Some(self.stream.remaining_from(offset)),
            "peek_ptr_array",
        )?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(r.read_ptr()?);
        }
        Ok(out)
    }

    pub fn get_rva(&self, pointer: u64) -> u64 {
        if self.image_base > 0 {
            pointer.wrapping_sub(self.image_base)
        } else {
            pointer
        }
    }

    pub fn get_il2cpp_type(&self, pointer: u64) -> Option<&Il2CppType> {
        self.type_dic
            .get(&pointer)
            .and_then(|idx| self.types.get(*idx))
    }

    pub fn map_vatr(&self, addr: u64) -> Result<u64> {
        for seg in &self.va_segments {
            if addr >= seg.vaddr && addr <= seg.vaddr + seg.memsz {
                return Ok(addr - seg.vaddr + seg.offset);
            }
        }
        // Runtime pointers of a memory dump (`image_base + file offset`) are not
        // described by any segment, and they are how method addresses are
        // stored. Runtime addresses sit far above every link-time segment, so
        // this can never shadow the mapping above.
        if self.image_base != 0 && addr >= self.image_base {
            let offset = addr - self.image_base;
            if offset < self.stream.len() {
                return Ok(offset);
            }
        }
        Err(Error::AddressNotMapped(addr))
    }

    pub fn map_rtva(&self, offset: u64) -> u64 {
        for seg in &self.va_segments {
            if offset >= seg.offset && offset <= seg.offset + seg.memsz {
                return offset - seg.offset + seg.vaddr;
            }
        }
        0
    }

    pub fn read_generic_class(&self, addr: u64) -> Result<Il2CppGenericClass> {
        self.peek_generic_class(addr)
    }

    pub fn read_generic_inst(&self, addr: u64) -> Result<Il2CppGenericInst> {
        self.peek_generic_inst(addr)
    }

    pub fn read_ptr_array(&self, addr: u64, count: u64) -> Result<Vec<u64>> {
        self.peek_ptr_array(addr, count)
    }

    pub fn detect_architecture(&self) -> Architecture {
        if let Some(arch) = self.arch {
            return arch;
        }
        Architecture::from_bitness(self.is_32bit, self.is_pe)
    }

    pub fn build_sorted_method_addresses(&self) -> Vec<u64> {
        let mut addrs = BTreeSet::new();

        for &ptr in &self.method_pointers {
            if ptr > 0 {
                addrs.insert(self.get_rva(ptr));
            }
        }

        for &ptr in &self.generic_method_pointers {
            if ptr > 0 {
                addrs.insert(self.get_rva(ptr));
            }
        }

        for ptrs in self.code_gen_module_method_pointers.values() {
            for &ptr in ptrs {
                if ptr > 0 {
                    addrs.insert(self.get_rva(ptr));
                }
            }
        }

        for (&_spec_idx, &ptr) in &self.method_spec_generic_method_pointers {
            if ptr > 0 {
                addrs.insert(self.get_rva(ptr));
            }
        }

        addrs.into_iter().collect()
    }

    /// Byte extent of the method at `rva`.
    ///
    /// When the next method start is known, the distance to it is the method's
    /// REAL extent and must not be clipped: a body truncated mid-way still has
    /// its branches encoded, so the recovery produced control flow that jumped
    /// past the end of the listing — `0x4661F0` (a 32 KB `Encoding` static
    /// constructor) alone reported 333 "branch target mismatch" findings, all of
    /// them branches into the half of the body the old `0x4000` cap had cut off.
    /// `MAX_MEASURED_BODY` is only a sanity bound; the callers that decode the
    /// body apply their own real caps (`max_method_bytes`, `MAX_IR_INSNS`).
    ///
    /// `MAX_BODY` remains the conservative fallback for the two cases where the
    /// extent is genuinely unknown: the method is the last start (no successor)
    /// or its RVA is not in the start set at all.
    pub fn get_method_body_size(&self, rva: u64, sorted_addrs: &[u64]) -> usize {
        const MAX_BODY: usize = 0x4000;
        const MAX_MEASURED_BODY: usize = 0x10000;
        const MIN_BODY: usize = 4;

        match sorted_addrs.binary_search(&rva) {
            Ok(idx) => {
                if idx + 1 < sorted_addrs.len() {
                    let next = sorted_addrs[idx + 1];
                    let diff = (next - rva) as usize;
                    diff.min(MAX_MEASURED_BODY).max(MIN_BODY)
                } else {
                    MAX_BODY
                }
            }
            Err(_) => MAX_BODY,
        }
    }

    pub fn read_bytes_at_rva(&self, rva: u64, size: usize) -> Option<Vec<u8>> {
        let va = if self.image_base > 0 {
            rva.wrapping_add(self.image_base)
        } else {
            rva
        };

        let file_offset = match self.map_vatr(va) {
            Ok(o) => o,
            Err(_) => {
                if rva < self.stream.len() {
                    rva
                } else {
                    return None;
                }
            }
        };

        let data = self.stream.data();
        let start = file_offset as usize;
        let end = (start + size).min(data.len());

        if start >= data.len() {
            return None;
        }

        Some(data[start..end].to_vec())
    }
}

/// Read the `typeDefinitionsSizes` table.
///
/// In every IL2CPP version this field is declared `const Il2CppTypeDefinitionSizes**`
/// (see `Il2CppMetadataRegistration` in the Unity headers), i.e. an array of
/// *pointers* to 16-byte sizes structs that live in the binary's read-only data —
/// Cpp2IL reads it the same way. Treating the field as an inline array of structs
/// slices each pointer in half: `instance_size` comes back looking like an address
/// and `thread_static_fields_size` is always 0, which silently breaks every
/// thread-static and FieldRVA layout decision built on top of it.
///
/// `mapped` is the caller-resolved file offset for each slot (`None` for a null
/// pointer or an address the container cannot map), so the address-space mapping
/// stays with the platform-specific loader.
pub fn read_type_definition_sizes(
    stream: &mut BinaryStream,
    mapped: &[Option<u64>],
) -> Vec<Il2CppTypeDefinitionSizes> {
    let mut out = Vec::with_capacity(mapped.len());
    for slot in mapped {
        match slot {
            Some(offset) => {
                stream.set_position(*offset);
                out.push(Il2CppTypeDefinitionSizes::read(stream).unwrap_or_default());
            }
            None => out.push(Il2CppTypeDefinitionSizes::default()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u64(buf: &mut [u8], at: usize, v: u64) {
        buf[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }

    /// A synthetic 64-bit image holding a valid `Il2CppCodeRegistration` at
    /// 0x100, with three pointer tables it names. 0x108 — one slot into the
    /// struct — stands in for the kind of address a version heuristic returns
    /// when it guesses the layout wrong.
    fn synthetic_code_registration_image() -> Vec<u8> {
        const CR: usize = 0x100;
        let mut buf = vec![0u8; 0x1000];
        put_u64(&mut buf, CR + 2 * 8, 3); // reversePInvokeWrapperCount
        put_u64(&mut buf, CR + 3 * 8, 0x400); // reversePInvokeWrappers
        put_u64(&mut buf, CR + 8 * 8, 4); // genericMethodPointersCount
        put_u64(&mut buf, CR + 9 * 8, 0x500); // genericMethodPointers
        put_u64(&mut buf, CR + 11 * 8, 2); // invokerPointersCount
        put_u64(&mut buf, CR + 12 * 8, 0x600); // invokerPointers
        for (i, addr) in [0x800u64, 0x810, 0x820].into_iter().enumerate() {
            put_u64(&mut buf, 0x400 + i * 8, addr);
        }
        for (i, addr) in [0x900u64, 0x910, 0x920, 0x930].into_iter().enumerate() {
            put_u64(&mut buf, 0x500 + i * 8, addr);
        }
        for (i, addr) in [0xA00u64, 0xA10].into_iter().enumerate() {
            put_u64(&mut buf, 0x600 + i * 8, addr);
        }
        buf
    }

    /// The reported failure: `count=268435456` in a count slot is a *pointer*
    /// that landed one field early, and it is rejected from the bytes alone.
    #[test]
    fn a_pointer_read_as_a_count_is_not_plausible() {
        let mut cr = Il2CppCodeRegistration::default();
        cr.generic_method_pointers_count = 268_435_456;
        cr.generic_method_pointers = 268_435_456;
        let reason = code_registration_plausibility(&cr, 29.1, 8, 32 * 1024 * 1024)
            .expect_err("268M pointers cannot exist in a 32 MB image");
        assert!(reason.contains("genericMethodPointersCount"), "{reason}");

        // The same struct is accepted once the count is a real count.
        let mut ok = Il2CppCodeRegistration::default();
        ok.generic_method_pointers_count = 40_000;
        ok.generic_method_pointers = 0x2000;
        assert!(code_registration_plausibility(&ok, 29.1, 8, 32 * 1024 * 1024).is_ok());
    }

    /// A count with no pointer is equally impossible — this is the rule that
    /// catches a struct read one pointer slot early, where the count slot holds
    /// the *next* field's pointer and the pointer slot holds nothing.
    #[test]
    fn a_count_without_its_pointer_is_not_plausible() {
        let mut shifts_into_a_pointer = Il2CppCodeRegistration::default();
        shifts_into_a_pointer.reverse_pinvoke_wrapper_count = 0x400;
        shifts_into_a_pointer.reverse_pinvoke_wrappers = 0;
        let reason = code_registration_plausibility(&shifts_into_a_pointer, 29.1, 8, 1 << 30)
            .expect_err("a count with a null pointer cannot be real");
        assert!(reason.contains("reversePInvokeWrapperCount"), "{reason}");
    }

    #[test]
    fn a_metadata_count_without_its_pointer_is_not_plausible() {
        let mut mr = Il2CppMetadataRegistration::default();
        mr.types_count = 43_956;
        mr.types = 0;
        let reason = metadata_registration_plausibility(&mr, 31.0, 8, 1 << 30)
            .expect_err("typesCount without `types` cannot be real");
        assert!(reason.contains("typesCount"), "{reason}");
    }

    /// When the heuristic address is off by a slot, the struct that is actually
    /// there is found by stepping back onto the real start.
    #[test]
    fn the_code_registration_is_re_anchored_when_the_heuristic_is_off() {
        let image = synthetic_code_registration_image();
        let mut il2cpp = Il2Cpp::new(BinaryStream::new(image), 29.1, false);
        let identity = |a: u64| Ok(a);
        assert_eq!(
            il2cpp.resolve_code_registration(0x108, &identity),
            0x100,
            "one slot into the struct must not be accepted"
        );
    }

    /// A healthy file never moves: the located address wins whenever it is at
    /// least as good as everything around it.
    #[test]
    fn a_healthy_address_is_never_moved() {
        let image = synthetic_code_registration_image();
        let mut il2cpp = Il2Cpp::new(BinaryStream::new(image), 29.1, false);
        let identity = |a: u64| Ok(a);
        assert_eq!(il2cpp.resolve_code_registration(0x100, &identity), 0x100);
    }

    /// An impossible array is recorded and skipped; it never aborts the run.
    #[test]
    fn an_impossible_array_is_reported_not_fatal() {
        let mut il2cpp = Il2Cpp::new(BinaryStream::new(vec![0u8; 0x100]), 29.1, false);
        let got = il2cpp.read_ptr_array_field("methodPointers", 0, 268_435_456);
        assert!(got.is_empty());
        assert_eq!(il2cpp.rejected_arrays.len(), 1);
        assert_eq!(il2cpp.rejected_arrays[0].field, "methodPointers");
        assert_eq!(il2cpp.rejected_arrays[0].count, 268_435_456);
    }

    /// `typeDefinitionsSizes` is an array of pointers, and the sizes structs it
    /// points at are separate. Reading the field as an inline struct array returns
    /// the low half of each pointer as `instance_size` (an address-shaped number)
    /// and 0 for every `thread_static_fields_size`. This image reproduces exactly
    /// that shape — pointers at 0x200, structs at 0x300 — so the two readings can
    /// be told apart.
    #[test]
    fn type_definition_sizes_are_reached_through_their_pointers() {
        let mut buf = vec![0u8; 0x400];
        // Two size structs: {instance 16, native -1, static 8, thread 0} and
        // {instance 40, native -1, static 0, thread 24}.
        buf[0x300..0x310]
            .copy_from_slice(&[16, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 8, 0, 0, 0, 0, 0, 0, 0]);
        buf[0x310..0x320]
            .copy_from_slice(&[40, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 24, 0, 0, 0]);
        // The pointer array the metadata registration names, plus one null slot.
        put_u64(&mut buf, 0x200, 0x300);
        put_u64(&mut buf, 0x208, 0x310);
        put_u64(&mut buf, 0x210, 0);

        let mut stream = BinaryStream::new(buf);
        let sizes = read_type_definition_sizes(&mut stream, &[Some(0x300), Some(0x310), None]);

        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes[0].instance_size, 16);
        assert_eq!(sizes[0].static_fields_size, 8);
        assert_eq!(sizes[1].instance_size, 40);
        assert_eq!(sizes[1].thread_static_fields_size, 24);
        // A null slot has no sizes; it must not borrow the next struct's bytes.
        assert_eq!(sizes[2].instance_size, 0);
        assert_eq!(sizes[2].thread_static_fields_size, 0);
    }

    /// The pointer-array reading must not be reachable by reading the pointer
    /// table itself as structs: that is what the bug did, and it is visible as an
    /// address-sized `instance_size` plus a permanently zero thread-static size.
    #[test]
    fn reading_the_pointer_table_as_structs_is_visibly_wrong() {
        let mut buf = vec![0u8; 0x400];
        buf[0x300..0x310]
            .copy_from_slice(&[16, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 8, 0, 0, 0, 0, 0, 0, 0]);
        put_u64(&mut buf, 0x200, 0x300);

        let mut stream = BinaryStream::new(buf.clone());
        let correct = read_type_definition_sizes(&mut stream, &[Some(0x300)]);
        let mut stream = BinaryStream::new(buf);
        let sequential = Il2CppTypeDefinitionSizes::read({
            stream.set_position(0x200);
            &mut stream
        })
        .unwrap();

        assert_eq!(correct[0].instance_size, 16);
        assert_ne!(sequential.instance_size, correct[0].instance_size);
        assert_eq!(sequential.instance_size, 0x300);
    }

    /// A pointer table whose entries do not map is not evidence for a struct.
    #[test]
    fn sample_entries_must_map_for_a_table_to_count() {
        let mut buf = vec![0u8; 0x100];
        put_u64(&mut buf, 0, 0x80);
        put_u64(&mut buf, 8, 0x90);
        let identity = |a: u64| Ok(a);
        let mut stream = BinaryStream::new(buf);
        assert!(Il2Cpp::sample_looks_like_pointers(
            &mut stream,
            &identity,
            0,
            2,
            8,
            0x100
        ));
        // All-zero table: no evidence either way, so it is not a negative.
        let mut zero = BinaryStream::new(vec![0u8; 0x100]);
        assert!(!Il2Cpp::sample_looks_like_pointers(
            &mut zero, &identity, 0, 2, 8, 0x100
        ));
        // A pointer past the end of the image can never be real.
        let mut wild = vec![0u8; 0x100];
        put_u64(&mut wild, 0, 0x1000_0000);
        let mut stream = BinaryStream::new(wild);
        assert!(!Il2Cpp::sample_looks_like_pointers(
            &mut stream,
            &identity,
            0,
            1,
            8,
            0x100
        ));
    }
}

/// The witness predicate behind [`Il2Cpp::upgrade_v31_1_by_module_witness`].
///
/// With a known image count the 31.1 `codeGenModulesCount` must equal it
/// exactly (FF MAX: 57) and the 31.0 view must differ. Without it, only the
/// silent-failure signature is accepted: the 31.0 view sees 0 modules where a
/// real image (v24.2+) always has at least one, and the 31.1 view sees a small
/// non-zero count whose array is validated by the caller.
fn v31_1_module_witness(image_count: u64, count310: u64, count311: u64) -> bool {
    if image_count > 0 {
        count310 != image_count && count311 == image_count
    } else {
        count310 == 0 && count311 > 0 && count311 <= 1024
    }
}

#[cfg(test)]
mod witness_tests {
    use super::v31_1_module_witness;

    #[test]
    fn ffmax_witness_upgrades() {
        // image_count = 57, 31.0 view reads the WNF count (0), 31.1 the
        // module count (57). Raw words: notes §1/§2.
        assert!(v31_1_module_witness(57, 0, 57));
    }

    #[test]
    fn a_true_31_0_registration_is_not_upgraded() {
        assert!(!v31_1_module_witness(57, 57, 0));
        assert!(!v31_1_module_witness(57, 57, 58));
    }

    #[test]
    fn unknown_image_count_requires_the_silent_failure_signature() {
        assert!(v31_1_module_witness(0, 0, 57));
        assert!(!v31_1_module_witness(0, 5, 57));
        assert!(!v31_1_module_witness(0, 0, 0));
        assert!(!v31_1_module_witness(0, 0, 4096));
    }
}
