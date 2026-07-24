mod arm;
mod x86;

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use std::sync::Arc;

const RIP_PSEUDO_REG: u16 = u16::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Arm32,
    Arm64,
    X86,
    X64,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Architecture::Arm32 => write!(f, "ARM32"),
            Architecture::Arm64 => write!(f, "ARM64"),
            Architecture::X86 => write!(f, "x86"),
            Architecture::X64 => write!(f, "x86_64"),
        }
    }
}

impl Architecture {
    pub fn from_elf_machine(e_machine: u16) -> Option<Self> {
        match e_machine {
            40 => Some(Architecture::Arm32),
            183 => Some(Architecture::Arm64),
            3 => Some(Architecture::X86),
            62 => Some(Architecture::X64),
            _ => None,
        }
    }

    pub fn from_pe_machine(machine: u16) -> Option<Self> {
        match machine {
            0x14C => Some(Architecture::X86),
            0x8664 => Some(Architecture::X64),
            0x1C0 | 0x1C4 => Some(Architecture::Arm32),
            0xAA64 => Some(Architecture::Arm64),
            _ => None,
        }
    }

    pub fn from_macho_cputype(cputype: u32) -> Option<Self> {
        match cputype {
            12 => Some(Architecture::Arm32),
            0x0100_000C => Some(Architecture::Arm64),
            7 => Some(Architecture::X86),
            0x0100_0007 => Some(Architecture::X64),
            _ => None,
        }
    }

    pub fn from_bitness(is_32bit: bool, is_pe: bool) -> Self {
        if is_pe {
            if is_32bit { Architecture::X86 } else { Architecture::X64 }
        } else {
            if is_32bit { Architecture::Arm32 } else { Architecture::Arm64 }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RegRegAccess {
    pub base_reg: u16,
    pub index_reg: u16,
    pub shift: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct LoadInfo {
    pub dest_reg: u16,
    pub base_reg: u16,
    pub offset: i64,
}

#[derive(Debug, Clone, Copy)]
pub enum ConstantOp {
    MovImm { dest_reg: u16, value: u64 },
    MovKeep { dest_reg: u16, imm16: u16, shift: u8 },
    AddSubImm { dest_reg: u16, src_reg: u16, imm: i64 },
    MovReg { dest_reg: u16, src_reg: u16 },
    Adrp { dest_reg: u16, page: u64 },
    Kill { dest_reg: u16 },
    KillPair { dest_reg1: u16, dest_reg2: u16 },
}

#[derive(Debug, Clone)]
pub struct DisassembledInstruction {
    pub address: u64,
    pub size: usize,
    pub raw_bytes: Vec<u8>,
    pub mnemonic: String,
    pub operands: String,
    pub is_call: bool,
    pub is_return: bool,
    pub is_branch: bool,
    pub is_unconditional_branch: bool,
    pub call_target: Option<u64>,
    pub branch_target: Option<u64>,
    pub condition_code: Option<String>,
    pub memory_offset: Option<i64>,
    pub reg_reg_access: Option<RegRegAccess>,
    pub constant_op: Option<ConstantOp>,
    pub load_info: Option<LoadInfo>,
    pub indirect_call_reg: Option<u16>,
}

impl fmt::Display for DisassembledInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.operands.is_empty() {
            write!(f, "{}", self.mnemonic)
        } else {
            write!(f, "{} {}", self.mnemonic, self.operands)
        }
    }
}

pub struct DisassemblyContext {
    /// Shared across methods of the same type (Arc so per-method ctx is cheap).
    pub field_offsets: Arc<HashMap<i32, String>>,
    pub string_literals: Arc<HashMap<u64, String>>,
    pub type_names: Arc<HashMap<u64, String>>,
    pub method_refs: Arc<HashMap<u64, String>>,
    pub field_refs: Arc<HashMap<u64, String>>,
    pub vtable_methods: Arc<HashMap<i32, String>>,
    /// Per-method register aliases (this, params) — not shared.
    pub register_names: HashMap<String, String>,
}

impl DisassemblyContext {
    pub fn new() -> Self {
        Self {
            field_offsets: Arc::new(HashMap::new()),
            string_literals: Arc::new(HashMap::new()),
            type_names: Arc::new(HashMap::new()),
            method_refs: Arc::new(HashMap::new()),
            field_refs: Arc::new(HashMap::new()),
            vtable_methods: Arc::new(HashMap::new()),
            register_names: HashMap::new(),
        }
    }

    /// Clone shared maps via Arc; start with empty per-method register names.
    pub fn share_maps_from(base: &Self) -> Self {
        Self {
            field_offsets: Arc::clone(&base.field_offsets),
            string_literals: Arc::clone(&base.string_literals),
            type_names: Arc::clone(&base.type_names),
            method_refs: Arc::clone(&base.method_refs),
            field_refs: Arc::clone(&base.field_refs),
            vtable_methods: Arc::clone(&base.vtable_methods),
            register_names: HashMap::new(),
        }
    }

    pub fn resolve_register(&self, reg: &str) -> String {
        let lower = reg.to_lowercase();
        if let Some(name) = self.register_names.get(&lower) {
            return name.clone();
        }
        reg.to_string()
    }

    pub fn resolve_operands(&self, operands: &str) -> String {
        let mut result = operands.to_string();
        for (reg, name) in &self.register_names {
            let upper = reg.to_uppercase();
            let patterns = [
                format!("{}, ", upper),
                format!("{},", upper),
                format!("[{},", upper),
                format!("[{}]", upper),
                format!("{} ", upper),
            ];
            for pat in &patterns {
                if result.contains(pat.as_str()) {
                    result = result.replace(pat.as_str(), &pat.replace(&upper, name));
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataAnnotationKind {
    StringLiteral,
    TypeInfo,
    MethodRef,
    FieldRef,
}

#[derive(Debug, Clone)]
pub struct MetadataAnnotation {
    pub kind: MetadataAnnotationKind,
    pub label: String,
}

pub struct Disassembler {
    arch: Architecture,
    rva_to_name: HashMap<u64, String>,
    global_annotations: HashMap<u64, MetadataAnnotation>,
    string_new_wrapper_rvas: HashSet<u64>,
    box_helper_rvas: HashSet<u64>,
    object_new_helper_rvas: HashSet<u64>,
    unbox_helper_rvas: HashSet<u64>,
    string_literals_by_index: HashMap<u32, String>,
    static_field_offsets_by_typeinfo: HashMap<u64, HashMap<i64, String>>,
    type_short_names: HashMap<u64, String>,
}

impl Disassembler {
    pub fn new(arch: Architecture) -> Self {
        Self {
            arch,
            rva_to_name: HashMap::new(),
            global_annotations: HashMap::new(),
            string_new_wrapper_rvas: HashSet::new(),
            box_helper_rvas: HashSet::new(),
            object_new_helper_rvas: HashSet::new(),
            unbox_helper_rvas: HashSet::new(),
            string_literals_by_index: HashMap::new(),
            static_field_offsets_by_typeinfo: HashMap::new(),
            type_short_names: HashMap::new(),
        }
    }

    pub fn add_string_new_wrapper_rva(&mut self, rva: u64) {
        self.string_new_wrapper_rvas.insert(rva);
    }

    pub fn add_box_helper_rva(&mut self, rva: u64) {
        self.box_helper_rvas.insert(rva);
    }

    pub fn add_object_new_helper_rva(&mut self, rva: u64) {
        self.object_new_helper_rvas.insert(rva);
    }

    pub fn add_unbox_helper_rva(&mut self, rva: u64) {
        self.unbox_helper_rvas.insert(rva);
    }

    pub fn add_static_field(&mut self, type_info_rva: u64, type_short_name: &str, offset: i64, field_name: &str) {
        self.static_field_offsets_by_typeinfo
            .entry(type_info_rva)
            .or_insert_with(HashMap::new)
            .insert(offset, field_name.to_string());
        self.type_short_names
            .entry(type_info_rva)
            .or_insert_with(|| type_short_name.to_string());
    }

    pub fn lookup_static_field(&self, type_info_rva: u64, offset: i64) -> Option<(&str, &str)> {
        let fields = self.static_field_offsets_by_typeinfo.get(&type_info_rva)?;
        let name = fields.get(&offset)?;
        let type_name = self.type_short_names.get(&type_info_rva)?;
        Some((type_name.as_str(), name.as_str()))
    }

    pub fn type_name_at_va(&self, va: u64) -> Option<&str> {
        self.global_annotations.get(&va).and_then(|ann| {
            if matches!(ann.kind, MetadataAnnotationKind::TypeInfo) {
                Some(ann.label.as_str())
            } else {
                None
            }
        })
    }

    pub fn set_string_literal_table(&mut self, table: HashMap<u32, String>) {
        self.string_literals_by_index = table;
    }

    pub fn arch(&self) -> Architecture {
        self.arch
    }
    pub fn set_method_names(&mut self, map: HashMap<u64, String>) {
        self.rva_to_name = map;
    }

    pub fn has_method_name(&self, rva: u64) -> bool {
        self.rva_to_name.contains_key(&rva)
    }

    pub fn annotation_count(&self) -> usize {
        self.global_annotations.len()
    }

    pub fn add_string_literal(&mut self, rva: u64, value: String) {
        let escaped = value
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
            .replace('\0', "\\0");
        let truncated = if escaped.len() > 60 {
            let safe_end = escaped.char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= 57)
                .last()
                .unwrap_or(0);
            format!("{}...", &escaped[..safe_end])
        } else {
            escaped
        };
        self.global_annotations.insert(rva, MetadataAnnotation {
            kind: MetadataAnnotationKind::StringLiteral,
            label: format!("\"{}\"", truncated),
        });
    }

    pub fn add_type_info(&mut self, rva: u64, type_name: String) {
        self.global_annotations.insert(rva, MetadataAnnotation {
            kind: MetadataAnnotationKind::TypeInfo,
            label: format!("typeof({})", type_name),
        });
    }

    pub fn add_method_ref(&mut self, rva: u64, full_name: String) {
        self.global_annotations.insert(rva, MetadataAnnotation {
            kind: MetadataAnnotationKind::MethodRef,
            label: full_name,
        });
    }

    pub fn add_field_ref(&mut self, rva: u64, full_name: String) {
        self.global_annotations.insert(rva, MetadataAnnotation {
            kind: MetadataAnnotationKind::FieldRef,
            label: full_name,
        });
    }

    pub fn disassemble(&self, bytes: &[u8], base_address: u64, max_instructions: usize) -> Vec<DisassembledInstruction> {
        match self.arch {
            Architecture::Arm64 => arm::disassemble_arm64(bytes, base_address, max_instructions),
            Architecture::Arm32 => arm::disassemble_arm32(bytes, base_address, max_instructions),
            Architecture::X86 => x86::disassemble_x86(bytes, base_address, max_instructions, 32),
            Architecture::X64 => x86::disassemble_x86(bytes, base_address, max_instructions, 64),
        }
    }

    pub fn format_method_body(
        &self,
        bytes: &[u8],
        base_address: u64,
        max_instructions: usize,
        indent: &str,
        ctx: Option<&DisassemblyContext>,
        show_hex_bytes: bool,
        show_field_names: bool,
        show_annotations: bool,
        show_cfg: bool,
    ) -> String {
        let instructions = self.disassemble(bytes, base_address, max_instructions);
        if instructions.is_empty() {
            return String::new();
        }

        let propagation = if show_field_names || show_annotations {
            compute_propagation(&instructions, self.arch)
        } else {
            PropagationResults {
                reg_reg_offsets: HashMap::new(),
                call_arg_w0: HashMap::new(),
                vtable_call_offsets: HashMap::new(),
                call_arg_x0_va: HashMap::new(),
                call_arg_x1_va: HashMap::new(),
            }
        };
        let reg_reg_offsets = &propagation.reg_reg_offsets;
        let call_arg_w0 = &propagation.call_arg_w0;
        let vtable_call_offsets = &propagation.vtable_call_offsets;
        let call_arg_x0_va = &propagation.call_arg_x0_va;
        let call_arg_x1_va = &propagation.call_arg_x1_va;

        let init_check = if show_annotations {
            detect_init_check_ranges(&instructions, self.arch, &self.rva_to_name)
        } else {
            InitCheckRanges::default()
        };

        let switch_dispatches = if show_annotations {
            detect_switch_dispatches(&instructions, self.arch)
        } else {
            HashMap::new()
        };

        let static_field_accesses = if show_annotations {
            detect_static_field_accesses(&instructions, self.arch, &self.global_annotations)
        } else {
            HashMap::new()
        };

        let total_bytes: usize = instructions.iter().map(|i| i.size).sum();

        let reg_names = ctx.map(|c| &c.register_names);

        let cfg = if show_cfg {
            Some(CfgAnalysis::build(&instructions, reg_names))
        } else {
            None
        };

        let mut buf = String::with_capacity(instructions.len() * 120);
        buf.push_str(&format!(
            "{indent}\t\t/* Disassembly ({}, {} instructions, 0x{:X} bytes):\n",
            self.arch,
            instructions.len(),
            total_bytes,
        ));

        if let Some(ref cfg) = cfg {
            if !cfg.blocks.is_empty() {
                buf.push_str(&format!(
                    "{indent}\t\t   // CFG: {} blocks, {} branches",
                    cfg.blocks.len(),
                    cfg.edge_count,
                ));
                if cfg.loop_count > 0 {
                    buf.push_str(&format!(", {} loop(s)", cfg.loop_count));
                }
                buf.push('\n');
            }
        }

        let mut adrp_page: Option<u64> = None;
        let mut _last_cmp_operands: Option<String> = None;

        for (idx, insn) in instructions.iter().enumerate() {
            if let Some(ref cfg) = cfg {
                if let Some(block_header) = cfg.block_headers.get(&insn.address) {
                    buf.push_str(&format!(
                        "{indent}\t\t   // {}\n",
                        block_header,
                    ));
                }
            }

            if show_hex_bytes {
                let hex = format_hex_bytes(&insn.raw_bytes);
                buf.push_str(&format!(
                    "{indent}\t\t   0x{:08X}:  {:<12} {}",
                    insn.address, hex, insn
                ));
            } else {
                buf.push_str(&format!(
                    "{indent}\t\t   0x{:08X}:  {}",
                    insn.address, insn
                ));
            }

            let mut annotated = false;

            if init_check.range_starts.contains(&insn.address) {
                buf.push_str("  // [init check]");
                annotated = true;
            } else if init_check.suppressed.contains(&insn.address) {
                annotated = true;
            }

            if !annotated && show_annotations {
                if let Some(sw) = switch_dispatches.get(&insn.address) {
                    match *sw {
                        SwitchAnnotation::TableBase { table_va } => {
                            buf.push_str(&format!("  // switch: table base = 0x{:X}", table_va));
                            annotated = true;
                        }
                        SwitchAnnotation::OffsetLoad { table_va, index_reg, shift } => {
                            buf.push_str(&format!(
                                "  // switch: offset = table[X{}] (table=0x{:X}, entry={}B)",
                                index_reg, table_va, 1u32 << shift,
                            ));
                            annotated = true;
                        }
                        SwitchAnnotation::TargetCompute { table_va } => {
                            buf.push_str(&format!("  // switch: target = table+offset (table=0x{:X})", table_va));
                            annotated = true;
                        }
                        SwitchAnnotation::Dispatch { table_va, index_reg } => {
                            buf.push_str(&format!(
                                "  // switch (X{}) → table 0x{:X}",
                                index_reg, table_va,
                            ));
                            annotated = true;
                        }
                    }
                }
            }

            if !annotated && show_annotations && insn.is_call && insn.call_target.is_none() {
                if let Some(&voff) = vtable_call_offsets.get(&insn.address) {
                    if let Some(ctx) = ctx {
                        if let Some(name) = ctx.vtable_methods.get(&(voff as i32)) {
                            buf.push_str(&format!("  // virtual call: {name}"));
                            annotated = true;
                        }
                    }
                    if !annotated {
                        buf.push_str(&format!("  // virtual call: vtable+0x{:X}", voff));
                        annotated = true;
                    }
                }
            }

            if !annotated && show_annotations {
                if let Some(target) = insn.call_target {
                    if self.string_new_wrapper_rvas.contains(&target) {
                        if let Some(&w0) = call_arg_w0.get(&insn.address) {
                            let idx = w0 as u32;
                            if let Some(s) = self.string_literals_by_index.get(&idx) {
                                buf.push_str(&format!("  // string_new(\"{}\")", crate::utils::escape_string_preview(s, 60)));
                                annotated = true;
                            } else {
                                buf.push_str(&format!("  // string_new(#{idx})"));
                                annotated = true;
                            }
                        } else {
                            buf.push_str("  // CALL → string_new_wrapper");
                            annotated = true;
                        }
                    } else if self.box_helper_rvas.contains(&target) {
                        if let Some(&va) = call_arg_x0_va.get(&insn.address) {
                            if let Some(type_name) = self.type_name_at_va(va) {
                                buf.push_str(&format!("  // box({type_name})"));
                                annotated = true;
                            }
                        }
                        if !annotated {
                            buf.push_str("  // CALL → il2cpp_codegen_box");
                            annotated = true;
                        }
                    } else if self.object_new_helper_rvas.contains(&target) {
                        if let Some(&va) = call_arg_x0_va.get(&insn.address) {
                            if let Some(type_name) = self.type_name_at_va(va) {
                                buf.push_str(&format!("  // new {type_name}()"));
                                annotated = true;
                            }
                        }
                        if !annotated {
                            buf.push_str("  // CALL → il2cpp_codegen_object_new");
                            annotated = true;
                        }
                    } else if self.unbox_helper_rvas.contains(&target) {
                        let type_via_x1 = call_arg_x1_va.get(&insn.address)
                            .and_then(|&va| self.type_name_at_va(va));
                        let type_via_x0 = call_arg_x0_va.get(&insn.address)
                            .and_then(|&va| self.type_name_at_va(va));
                        if let Some(type_name) = type_via_x1.or(type_via_x0) {
                            buf.push_str(&format!("  // unbox<{type_name}>"));
                            annotated = true;
                        } else {
                            buf.push_str("  // CALL → il2cpp_unbox");
                            annotated = true;
                        }
                    } else if let Some(name) = self.rva_to_name.get(&target) {
                        buf.push_str(&format!("  // CALL → {name}"));
                        annotated = true;
                    } else if let Some(ann) = self.global_annotations.get(&target) {
                        buf.push_str(&format!("  // CALL → {}", ann.label));
                        annotated = true;
                    } else {
                        buf.push_str(&format!("  // CALL → sub_{:X}", target));
                        annotated = true;
                    }
                }
            }

            if !annotated && insn.is_branch && !insn.is_call && !insn.is_return {
                if let Some(target) = insn.branch_target {
                    if insn.is_unconditional_branch {
                        if let Some(name) = self.rva_to_name.get(&target) {
                            buf.push_str(&format!("  // TAIL CALL → {name}"));
                            annotated = true;
                        }
                    }
                }
                if !annotated {
                    if let Some(ref cfg) = cfg {
                        if let Some(edge_label) = cfg.edge_labels.get(&insn.address) {
                            buf.push_str(&format!("  // {}", edge_label));
                            annotated = true;
                        }
                    }
                }
            }

            if !annotated && show_annotations && insn.mnemonic == "ADRP" {
                let page = extract_adrp_page(insn);
                adrp_page = page;
            }

            if !annotated && show_annotations && (insn.mnemonic == "LDR" || insn.mnemonic == "ADD") {
                if let Some(page_base) = adrp_page {
                    if let Some(offset) = insn.memory_offset {
                        let full_addr = page_base.wrapping_add(offset as u64);
                        if let Some(ann) = self.global_annotations.get(&full_addr) {
                            let prefix = match ann.kind {
                                MetadataAnnotationKind::StringLiteral => "str",
                                MetadataAnnotationKind::TypeInfo => "type",
                                MetadataAnnotationKind::MethodRef => "method",
                                MetadataAnnotationKind::FieldRef => "field",
                            };
                            buf.push_str(&format!("  // {}: {}", prefix, ann.label));
                            annotated = true;
                        }
                    }
                }

                if !annotated {
                    if let Some(offset) = insn.memory_offset {
                        if let Some(ann) = self.global_annotations.get(&(offset as u64)) {
                            let prefix = match ann.kind {
                                MetadataAnnotationKind::StringLiteral => "str",
                                MetadataAnnotationKind::TypeInfo => "type",
                                MetadataAnnotationKind::MethodRef => "method",
                                MetadataAnnotationKind::FieldRef => "field",
                            };
                            buf.push_str(&format!("  // {}: {}", prefix, ann.label));
                            annotated = true;
                        }
                    }
                }
            }

            if !annotated && show_annotations {
                if let Some(&(typeinfo_va, field_offset)) = static_field_accesses.get(&insn.address) {
                    if let Some((type_name, field_name)) = self.lookup_static_field(typeinfo_va, field_offset) {
                        buf.push_str(&format!("  // {}.{}", type_name, field_name));
                        annotated = true;
                    } else {
                        if let Some(ann) = self.global_annotations.get(&typeinfo_va) {
                            buf.push_str(&format!("  // {}.<static+0x{:X}>", ann.label, field_offset));
                            annotated = true;
                        }
                    }
                }
            }

            if !annotated && show_field_names {
                if let Some(offset) = insn.memory_offset {
                    if let Some(ctx) = ctx {
                        let operand_text = &insn.operands;
                        let is_sp_access = operand_text.contains("sp,")
                            || operand_text.contains("sp]")
                            || operand_text.contains("SP,")
                            || operand_text.contains("SP]");

                        if !is_sp_access {
                            if let Some(field_name) = ctx.field_offsets.get(&(offset as i32)) {
                                if !insn.is_call {
                                    buf.push_str(&format!("  // this.{field_name}"));
                                    annotated = true;
                                }
                            }
                        }

                        if !annotated && !is_sp_access {
                            if let Some(vtable_method) = ctx.vtable_methods.get(&(offset as i32)) {
                                buf.push_str(&format!("  // vtable: {vtable_method}"));
                            }
                        }
                    }
                }

                if !annotated {
                    if let Some(&effective_offset) = reg_reg_offsets.get(&insn.address) {
                        if let Some(ctx) = ctx {
                            let is_sp_access = insn.reg_reg_access
                                .map(|rra| rra.base_reg == 31)
                                .unwrap_or(false)
                                || insn.operands.contains("sp,")
                                || insn.operands.contains("SP,");

                            if !is_sp_access {
                                if let Some(field_name) = ctx.field_offsets.get(&(effective_offset as i32)) {
                                    if !insn.is_call {
                                        buf.push_str(&format!("  // this.{field_name}"));
                                        annotated = true;
                                    }
                                }

                                if !annotated {
                                    if let Some(vtable_method) = ctx.vtable_methods.get(&(effective_offset as i32)) {
                                        buf.push_str(&format!("  // vtable: {vtable_method}"));
                                        annotated = true;
                                    }
                                }
                            }
                        }

                        if !annotated && show_annotations {
                            if let Some(ann) = self.global_annotations.get(&(effective_offset as u64)) {
                                let prefix = match ann.kind {
                                    MetadataAnnotationKind::StringLiteral => "str",
                                    MetadataAnnotationKind::TypeInfo => "type",
                                    MetadataAnnotationKind::MethodRef => "method",
                                    MetadataAnnotationKind::FieldRef => "field",
                                };
                                buf.push_str(&format!("  // {}: {}", prefix, ann.label));
                                let _ = annotated;
                            }
                        }
                    }
                }
            }

            if insn.mnemonic == "CMP" || insn.mnemonic == "TST"
                || insn.mnemonic == "TEST" || insn.mnemonic == "SUBS"
                || insn.mnemonic == "CCMP"
            {
                _last_cmp_operands = Some(insn.operands.clone());
            }

            buf.push('\n');

            if let Some(ref cfg) = cfg {
                if insn.is_branch && !insn.is_call {
                    if let Some(separator) = cfg.block_separators.get(&insn.address) {
                        buf.push_str(&format!(
                            "{indent}\t\t   // {}\n",
                            separator,
                        ));
                    }
                }
            }

            if insn.is_unconditional_branch && !insn.is_call {
                if idx + 1 < instructions.len() {
                    if cfg.is_none() {
                        buf.push('\n');
                    }
                }
            }
        }

        buf.push_str(&format!("{indent}\t\t*/\n"));
        buf
    }
}

struct CfgAnalysis {
    blocks: Vec<BlockInfo>,
    block_headers: HashMap<u64, String>,
    block_separators: HashMap<u64, String>,
    edge_labels: HashMap<u64, String>,
    edge_count: usize,
    loop_count: usize,
}

struct BlockInfo {
    _start_addr: u64,
    _end_addr: u64,
    _id: usize,
}

impl CfgAnalysis {
    fn build(instructions: &[DisassembledInstruction], reg_names: Option<&HashMap<String, String>>) -> Self {
        if instructions.is_empty() {
            return Self {
                blocks: Vec::new(),
                block_headers: HashMap::new(),
                block_separators: HashMap::new(),
                edge_labels: HashMap::new(),
                edge_count: 0,
                loop_count: 0,
            };
        }

        let addr_set: HashSet<u64> = instructions.iter().map(|i| i.address).collect();
        let first_addr = instructions[0].address;
        let last_addr = instructions.last().unwrap().address;

        let mut branch_targets: BTreeMap<u64, Vec<IncomingEdge>> = BTreeMap::new();

        let mut last_cmp: Option<String> = None;

        for (_idx, insn) in instructions.iter().enumerate() {
            if insn.mnemonic == "CMP" || insn.mnemonic == "TST"
                || insn.mnemonic == "TEST" || insn.mnemonic == "SUBS"
                || insn.mnemonic == "CCMP"
            {
                last_cmp = Some(resolve_operands_with_names(&insn.operands, reg_names));
            }

            if !insn.is_branch || insn.is_call {
                continue;
            }

            if let Some(target) = insn.branch_target {
                if !addr_set.contains(&target) {
                    continue;
                }

                let is_back_edge = target <= insn.address;

                let condition_text = build_condition_text(insn, &last_cmp, is_back_edge, reg_names);

                branch_targets
                    .entry(target)
                    .or_insert_with(Vec::new)
                    .push(IncomingEdge {
                        from_addr: insn.address,
                        condition: condition_text,
                        is_back_edge,
                    });
            }

            if insn.condition_code.is_some() && !insn.is_unconditional_branch {
                let fall_through = insn.address + insn.size as u64;
                if addr_set.contains(&fall_through) {
                    let negated = negate_condition_text(insn, &last_cmp, reg_names);
                    branch_targets
                        .entry(fall_through)
                        .or_insert_with(Vec::new)
                        .push(IncomingEdge {
                            from_addr: insn.address,
                            condition: negated,
                            is_back_edge: false,
                        });
                }
            }
        }

        let mut block_starts: HashSet<u64> = HashSet::new();
        block_starts.insert(first_addr);
        for target in branch_targets.keys() {
            block_starts.insert(*target);
        }
        for insn in instructions {
            if insn.is_branch && !insn.is_call {
                let after = insn.address + insn.size as u64;
                if addr_set.contains(&after) {
                    block_starts.insert(after);
                }
            }
        }

        let mut sorted_starts: Vec<u64> = block_starts.into_iter().collect();
        sorted_starts.sort_unstable();

        let mut blocks: Vec<BlockInfo> = Vec::new();
        for (id, &start) in sorted_starts.iter().enumerate() {
            let end = sorted_starts.get(id + 1).copied()
                .unwrap_or(last_addr + instructions.last().unwrap().size as u64);
            blocks.push(BlockInfo {
                _start_addr: start,
                _end_addr: end,
                _id: id,
            });
        }

        let mut edge_count = 0;
        let mut loop_count = 0;
        let mut block_headers: HashMap<u64, String> = HashMap::new();
        let mut block_separators: HashMap<u64, String> = HashMap::new();
        let mut edge_labels: HashMap<u64, String> = HashMap::new();

        if blocks.len() > 1 {
            block_headers.insert(first_addr, format!(
                "═══ Block 0 (entry) {}",
                "═".repeat(30)
            ));
        }

        for (target_addr, edges) in &branch_targets {
            if *target_addr == first_addr { continue; }

            let block_idx = sorted_starts.iter().position(|a| *a == *target_addr);
            let block_label = block_idx.map(|i| format!("Block {}", i)).unwrap_or_default();

            let mut header_parts: Vec<String> = Vec::new();
            let mut has_back_edge = false;

            for edge in edges {
                edge_count += 1;
                if edge.is_back_edge {
                    has_back_edge = true;
                    loop_count += 1;
                }
                if !edge.condition.is_empty() {
                    header_parts.push(edge.condition.clone());
                }
            }

            if has_back_edge {
                let header = format!(
                    "──── ↑ loop target ({}) {} (from 0x{:X})",
                    header_parts.first().unwrap_or(&String::new()),
                    "─".repeat(20),
                    edges.iter().find(|e| e.is_back_edge).map(|e| e.from_addr).unwrap_or(0),
                );
                block_headers.insert(*target_addr, header);
            } else if header_parts.len() == 1 {
                let header = format!(
                    "──── {} {} {}",
                    block_label,
                    header_parts[0],
                    "─".repeat(20),
                );
                block_headers.insert(*target_addr, header);
            } else if header_parts.len() > 1 {
                let header = format!(
                    "──── {} (from {} paths) {}",
                    block_label,
                    header_parts.len(),
                    "─".repeat(18),
                );
                block_headers.insert(*target_addr, header);
            } else {
                let header = format!(
                    "──── {} {}",
                    block_label,
                    "─".repeat(30),
                );
                block_headers.insert(*target_addr, header);
            }
        }

        for insn in instructions {
            if !insn.is_branch || insn.is_call || insn.is_return { continue; }

            if let Some(target) = insn.branch_target {
                if !addr_set.contains(&target) { continue; }

                let is_back_edge = target <= insn.address;

                if is_back_edge {
                    edge_labels.insert(insn.address, format!("↑ loop back to 0x{:08X}", target));
                } else if insn.is_unconditional_branch {
                    edge_labels.insert(insn.address, format!("goto 0x{:08X}", target));
                } else if let Some(ref cond) = insn.condition_code {
                    let branch_label = build_branch_label(insn, cond, reg_names);
                    edge_labels.insert(insn.address, branch_label);
                }
            }

            if insn.condition_code.is_some() && !insn.is_unconditional_branch {
                let sep_addr = insn.address;
                if insn.branch_target.map(|t| t > insn.address).unwrap_or(false) {
                    block_separators.insert(sep_addr, String::new());
                }
            }
        }

        let last_block_entry = sorted_starts.last().copied().unwrap_or(0);
        let has_ret_in_last = instructions.iter().any(|i| i.is_return);
        if blocks.len() > 1 && has_ret_in_last {
            for insn in instructions.iter().rev() {
                if insn.is_return {
                    if insn.address >= last_block_entry && !block_headers.contains_key(&insn.address) {
                    }
                    break;
                }
            }
        }

        Self {
            blocks,
            block_headers,
            block_separators,
            edge_labels,
            edge_count,
            loop_count,
        }
    }
}

struct IncomingEdge {
    from_addr: u64,
    condition: String,
    is_back_edge: bool,
}

fn resolve_reg(reg: &str, reg_names: Option<&HashMap<String, String>>) -> String {
    if let Some(names) = reg_names {
        let lower = reg.to_lowercase().replace(' ', "");
        if let Some(name) = names.get(&lower) {
            return name.clone();
        }
    }
    reg.to_string()
}

fn resolve_operands_with_names(operands: &str, reg_names: Option<&HashMap<String, String>>) -> String {
    let names = match reg_names {
        Some(n) if !n.is_empty() => n,
        _ => return operands.to_string(),
    };

    let mut result = operands.to_string();
    for (reg_lower, name) in names {
        let reg_upper = reg_lower.to_uppercase();
        let lower = reg_lower.clone();

        let pats = [
            (format!("{}, ", reg_upper), format!("{}, ", name)),
            (format!("{},", reg_upper), format!("{},", name)),
            (format!("{}, ", lower), format!("{}, ", name)),
            (format!("{},", lower), format!("{},", name)),
        ];

        for (from, to) in &pats {
            result = result.replace(from.as_str(), to.as_str());
        }
    }
    result
}

fn build_condition_text(
    insn: &DisassembledInstruction,
    last_cmp: &Option<String>,
    _is_back_edge: bool,
    reg_names: Option<&HashMap<String, String>>,
) -> String {
    match insn.mnemonic.as_str() {
        "CBZ" => {
            let reg = insn.operands.split(',').next().unwrap_or("?").trim();
            let name = resolve_reg(reg, reg_names);
            format!("if ({} == null)", name)
        }
        "CBNZ" => {
            let reg = insn.operands.split(',').next().unwrap_or("?").trim();
            let name = resolve_reg(reg, reg_names);
            format!("if ({} != null)", name)
        }
        "TBZ" => {
            let parts: Vec<&str> = insn.operands.splitn(3, ',').collect();
            if parts.len() >= 2 {
                let reg = parts[0].trim();
                let name = resolve_reg(reg, reg_names);
                let bit = parts[1].trim().trim_start_matches('#');
                if bit == "0" {
                    format!("if (!{}.initialized)", name)
                } else {
                    format!("if (bit{} of {} == 0)", bit, name)
                }
            } else {
                "if (bit == 0)".to_string()
            }
        }
        "TBNZ" => {
            let parts: Vec<&str> = insn.operands.splitn(3, ',').collect();
            if parts.len() >= 2 {
                let reg = parts[0].trim();
                let name = resolve_reg(reg, reg_names);
                let bit = parts[1].trim().trim_start_matches('#');
                if bit == "0" {
                    format!("if ({}.initialized)", name)
                } else {
                    format!("if (bit{} of {} != 0)", bit, name)
                }
            } else {
                "if (bit != 0)".to_string()
            }
        }
        _ => {
            if let Some(ref cond) = insn.condition_code {
                if let Some(cmp_ops) = last_cmp {
                    let parts: Vec<&str> = cmp_ops.splitn(2, ',').collect();
                    if parts.len() == 2 {
                        let lhs = parts[0].trim();
                        let rhs = parts[1].trim();
                        format!("if ({} {} {})", lhs, cond, rhs)
                    } else {
                        format!("if ({})", cond)
                    }
                } else {
                    format!("if ({})", cond)
                }
            } else {
                String::new()
            }
        }
    }
}

fn negate_condition_text(
    insn: &DisassembledInstruction,
    last_cmp: &Option<String>,
    reg_names: Option<&HashMap<String, String>>,
) -> String {
    match insn.mnemonic.as_str() {
        "CBZ" => {
            let reg = insn.operands.split(',').next().unwrap_or("?").trim();
            let name = resolve_reg(reg, reg_names);
            format!("else ({} != null)", name)
        }
        "CBNZ" => {
            let reg = insn.operands.split(',').next().unwrap_or("?").trim();
            let name = resolve_reg(reg, reg_names);
            format!("else ({} == null)", name)
        }
        "TBZ" => {
            let parts: Vec<&str> = insn.operands.splitn(3, ',').collect();
            if parts.len() >= 2 {
                let reg = parts[0].trim();
                let name = resolve_reg(reg, reg_names);
                let bit = parts[1].trim().trim_start_matches('#');
                if bit == "0" {
                    format!("else ({}.initialized)", name)
                } else {
                    format!("else (bit{} of {} != 0)", bit, name)
                }
            } else {
                "else (bit != 0)".to_string()
            }
        }
        "TBNZ" => {
            let parts: Vec<&str> = insn.operands.splitn(3, ',').collect();
            if parts.len() >= 2 {
                let reg = parts[0].trim();
                let name = resolve_reg(reg, reg_names);
                let bit = parts[1].trim().trim_start_matches('#');
                if bit == "0" {
                    format!("else (!{}.initialized)", name)
                } else {
                    format!("else (bit{} of {} == 0)", bit, name)
                }
            } else {
                "else (bit == 0)".to_string()
            }
        }
        _ => {
            if let Some(ref cond) = insn.condition_code {
                let negated = negate_operator(cond);
                if let Some(cmp_ops) = last_cmp {
                    let parts: Vec<&str> = cmp_ops.splitn(2, ',').collect();
                    if parts.len() == 2 {
                        let lhs = parts[0].trim();
                        let rhs = parts[1].trim();
                        format!("else ({} {} {})", lhs, negated, rhs)
                    } else {
                        format!("else ({})", negated)
                    }
                } else {
                    format!("else ({})", negated)
                }
            } else {
                "else".to_string()
            }
        }
    }
}

fn negate_operator(op: &str) -> &str {
    match op {
        "==" => "!=",
        "!=" => "==",
        ">" => "<=",
        "<" => ">=",
        ">=" => "<",
        "<=" => ">",
        "> (unsigned)" => "<= (unsigned)",
        "< (unsigned)" => ">= (unsigned)",
        ">= (unsigned)" => "< (unsigned)",
        "<= (unsigned)" => "> (unsigned)",
        _ => op,
    }
}

fn build_branch_label(insn: &DisassembledInstruction, cond: &str, reg_names: Option<&HashMap<String, String>>) -> String {
    match insn.mnemonic.as_str() {
        "CBZ" => {
            let reg = insn.operands.split(',').next().unwrap_or("?").trim();
            let name = resolve_reg(reg, reg_names);
            format!("if ({} == null) goto 0x{:08X}", name, insn.branch_target.unwrap_or(0))
        }
        "CBNZ" => {
            let reg = insn.operands.split(',').next().unwrap_or("?").trim();
            let name = resolve_reg(reg, reg_names);
            format!("if ({} != null) goto 0x{:08X}", name, insn.branch_target.unwrap_or(0))
        }
        "TBZ" => {
            let parts: Vec<&str> = insn.operands.splitn(3, ',').collect();
            let reg = parts.first().map(|s| s.trim()).unwrap_or("?");
            let name = resolve_reg(reg, reg_names);
            let bit = parts.get(1).map(|s| s.trim().trim_start_matches('#')).unwrap_or("0");
            if bit == "0" {
                format!("if (!{}.initialized) goto 0x{:08X}", name, insn.branch_target.unwrap_or(0))
            } else {
                format!("if (bit{} of {} == 0) goto 0x{:08X}", bit, name, insn.branch_target.unwrap_or(0))
            }
        }
        "TBNZ" => {
            let parts: Vec<&str> = insn.operands.splitn(3, ',').collect();
            let reg = parts.first().map(|s| s.trim()).unwrap_or("?");
            let name = resolve_reg(reg, reg_names);
            let bit = parts.get(1).map(|s| s.trim().trim_start_matches('#')).unwrap_or("0");
            if bit == "0" {
                format!("if ({}.initialized) goto 0x{:08X}", name, insn.branch_target.unwrap_or(0))
            } else {
                format!("if (bit{} of {} != 0) goto 0x{:08X}", bit, name, insn.branch_target.unwrap_or(0))
            }
        }
        _ => {
            format!("if ({}) goto 0x{:08X}", cond, insn.branch_target.unwrap_or(0))
        }
    }
}

fn format_hex_bytes(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        hex.push_str(&format!("{:02X}", b));
    }
    hex
}

fn extract_adrp_page(insn: &DisassembledInstruction) -> Option<u64> {
    let operands = &insn.operands;
    if let Some(dollar_pos) = operands.find("$+") {
        let after = &operands[dollar_pos + 2..];
        let hex_str = after.trim().trim_start_matches("0x").trim_start_matches("0X");
        if let Ok(offset) = u64::from_str_radix(hex_str, 16) {
            return Some(insn.address.wrapping_add(offset));
        }
    }
    if let Some(dollar_pos) = operands.find("$-") {
        let after = &operands[dollar_pos + 2..];
        let hex_str = after.trim().trim_start_matches("0x").trim_start_matches("0X");
        if let Ok(offset) = u64::from_str_radix(hex_str, 16) {
            return Some(insn.address.wrapping_sub(offset));
        }
    }
    if let Some(hash_pos) = operands.find("#0x") {
        let after = &operands[hash_pos + 3..];
        let end = after.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(after.len());
        if let Ok(val) = u64::from_str_radix(&after[..end], 16) {
            return Some(val);
        }
    }
    None
}

pub struct PropagationResults {
    pub reg_reg_offsets: HashMap<u64, i64>,
    pub call_arg_w0: HashMap<u64, u64>,
    pub vtable_call_offsets: HashMap<u64, i64>,
    pub call_arg_x0_va: HashMap<u64, u64>,
    pub call_arg_x1_va: HashMap<u64, u64>,
}

pub fn analyze_propagation(
    instructions: &[DisassembledInstruction],
    arch: Architecture,
) -> PropagationResults {
    compute_propagation(instructions, arch)
}

fn compute_propagation(
    instructions: &[DisassembledInstruction],
    arch: Architecture,
) -> PropagationResults {
    let mut reg_state: HashMap<u16, u64> = HashMap::new();
    let mut result: HashMap<u64, i64> = HashMap::new();
    let mut call_arg_w0: HashMap<u64, u64> = HashMap::new();
    let mut vtable_call_offsets: HashMap<u64, i64> = HashMap::new();
    let mut reg_load_offset: HashMap<u16, i64> = HashMap::new();
    let mut reg_load_va: HashMap<u16, u64> = HashMap::new();
    let mut call_arg_x0_va: HashMap<u64, u64> = HashMap::new();
    let mut call_arg_x1_va: HashMap<u64, u64> = HashMap::new();

    let branch_targets: HashSet<u64> = instructions.iter()
        .filter_map(|i| i.branch_target)
        .chain(instructions.iter().filter_map(|i| i.call_target))
        .collect();

    let internal_targets: HashSet<u64> = {
        let addr_set: HashSet<u64> = instructions.iter().map(|i| i.address).collect();
        branch_targets.iter().filter(|a| addr_set.contains(a)).copied().collect()
    };

    for insn in instructions {
        if internal_targets.contains(&insn.address) {
            reg_state.clear();
            reg_load_offset.clear();
            reg_load_va.clear();
        }

        if let Some(ref rra) = insn.reg_reg_access {
            if let Some(&index_val) = reg_state.get(&rra.index_reg) {
                let effective_offset = (index_val << rra.shift as u64) as i64;
                result.insert(insn.address, effective_offset);
            }
        }

        if insn.is_call {
            if let Some(target_reg) = insn.indirect_call_reg {
                if let Some(&offset) = reg_load_offset.get(&target_reg) {
                    vtable_call_offsets.insert(insn.address, offset);
                }
            }
        }

        if let Some(ref li) = insn.load_info {
            reg_load_offset.insert(li.dest_reg, li.offset);
            if li.base_reg == RIP_PSEUDO_REG {
                reg_load_va.insert(li.dest_reg, li.offset as u64);
            } else if let Some(&base_va) = reg_state.get(&li.base_reg) {
                let load_va = (base_va as i64).wrapping_add(li.offset) as u64;
                reg_load_va.insert(li.dest_reg, load_va);
            } else {
                reg_load_va.remove(&li.dest_reg);
            }
        }

        if let Some(ref op) = insn.constant_op {
            let writes_load: bool = insn.load_info.is_some();
            let mut kill_load = |r: u16| {
                if !writes_load || insn.load_info.map(|li| li.dest_reg != r).unwrap_or(true) {
                    reg_load_offset.remove(&r);
                    reg_load_va.remove(&r);
                }
            };
            match *op {
                ConstantOp::MovImm { dest_reg, .. }
                | ConstantOp::MovKeep { dest_reg, .. }
                | ConstantOp::AddSubImm { dest_reg, .. }
                | ConstantOp::MovReg { dest_reg, .. }
                | ConstantOp::Adrp { dest_reg, .. }
                | ConstantOp::Kill { dest_reg } => kill_load(dest_reg),
                ConstantOp::KillPair { dest_reg1, dest_reg2 } => {
                    kill_load(dest_reg1);
                    kill_load(dest_reg2);
                }
            }

            match *op {
                ConstantOp::MovImm { dest_reg, value } => {
                    reg_state.insert(dest_reg, value);
                }
                ConstantOp::MovKeep { dest_reg, imm16, shift } => {
                    if let Some(&current) = reg_state.get(&dest_reg) {
                        let mask = !(0xFFFFu64 << shift);
                        let new_val = (current & mask) | ((imm16 as u64) << shift);
                        reg_state.insert(dest_reg, new_val);
                    }
                }
                ConstantOp::AddSubImm { dest_reg, src_reg, imm } => {
                    if let Some(&src_val) = reg_state.get(&src_reg) {
                        reg_state.insert(dest_reg, (src_val as i64 + imm) as u64);
                    } else {
                        reg_state.remove(&dest_reg);
                    }
                }
                ConstantOp::MovReg { dest_reg, src_reg } => {
                    if let Some(&src_val) = reg_state.get(&src_reg) {
                        reg_state.insert(dest_reg, src_val);
                    } else {
                        reg_state.remove(&dest_reg);
                    }
                }
                ConstantOp::Adrp { dest_reg, page } => {
                    reg_state.insert(dest_reg, page);
                }
                ConstantOp::Kill { dest_reg } => {
                    reg_state.remove(&dest_reg);
                }
                ConstantOp::KillPair { dest_reg1, dest_reg2 } => {
                    reg_state.remove(&dest_reg1);
                    reg_state.remove(&dest_reg2);
                }
            }
        }

        if insn.is_call {
            let arg_reg: u16 = match arch {
                Architecture::Arm64 | Architecture::Arm32 => 0,
                Architecture::X86 | Architecture::X64 => 1,
            };
            if let Some(&v) = reg_state.get(&arg_reg) {
                call_arg_w0.insert(insn.address, v);
            }
            if let Some(&va) = reg_load_va.get(&arg_reg) {
                call_arg_x0_va.insert(insn.address, va);
            }
            let arg1_reg: u16 = match arch {
                Architecture::Arm64 | Architecture::Arm32 => 1,
                Architecture::X86 | Architecture::X64 => 2,
            };
            if let Some(&va) = reg_load_va.get(&arg1_reg) {
                call_arg_x1_va.insert(insn.address, va);
            }

            match arch {
                Architecture::Arm64 => {
                    for r in 0..=18u16 {
                        reg_state.remove(&r);
                        reg_load_offset.remove(&r);
                        reg_load_va.remove(&r);
                    }
                    reg_state.remove(&30);
                    reg_load_offset.remove(&30);
                    reg_load_va.remove(&30);
                }
                Architecture::Arm32 => {
                    for r in 0..=3u16 {
                        reg_state.remove(&r);
                        reg_load_offset.remove(&r);
                        reg_load_va.remove(&r);
                    }
                    reg_state.remove(&12);
                    reg_state.remove(&14);
                    reg_load_offset.remove(&12);
                    reg_load_offset.remove(&14);
                    reg_load_va.remove(&12);
                    reg_load_va.remove(&14);
                }
                Architecture::X86 | Architecture::X64 => {
                    reg_state.clear();
                    reg_load_offset.clear();
                    reg_load_va.clear();
                }
            }
        }
    }

    PropagationResults {
        reg_reg_offsets: result,
        call_arg_w0,
        vtable_call_offsets,
        call_arg_x0_va,
        call_arg_x1_va,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SwitchAnnotation {
    TableBase { table_va: u64 },
    OffsetLoad { table_va: u64, index_reg: u16, shift: u8 },
    TargetCompute { table_va: u64 },
    Dispatch { table_va: u64, index_reg: u16 },
}

#[derive(Debug, Clone, Copy)]
enum SwOrigin {
    Page(u64),
    TableBase(u64),
    LoadedOffset { table_va: u64, index_reg: u16 },
    Target { table_va: u64, index_reg: u16 },
}

fn detect_switch_dispatches(
    instructions: &[DisassembledInstruction],
    arch: Architecture,
) -> HashMap<u64, SwitchAnnotation> {
    match arch {
        Architecture::Arm64 => detect_switch_arm64(instructions),
        Architecture::X86 | Architecture::X64 => detect_switch_x86(instructions),
        _ => HashMap::new(),
    }
}

fn detect_switch_arm64(
    instructions: &[DisassembledInstruction],
) -> HashMap<u64, SwitchAnnotation> {
    let mut out: HashMap<u64, SwitchAnnotation> = HashMap::new();
    if instructions.is_empty() { return out; }

    let branch_targets: HashSet<u64> = instructions.iter()
        .filter_map(|i| i.branch_target)
        .chain(instructions.iter().filter_map(|i| i.call_target))
        .collect();
    let internal_targets: HashSet<u64> = {
        let addrs: HashSet<u64> = instructions.iter().map(|i| i.address).collect();
        branch_targets.iter().filter(|a| addrs.contains(a)).copied().collect()
    };

    let mut state: HashMap<u16, SwOrigin> = HashMap::new();

    let load_mnemonics = ["LDR", "LDRSW", "LDRH", "LDRSH", "LDRB", "LDRSB"];

    for insn in instructions {
        if internal_targets.contains(&insn.address) {
            state.clear();
        }

        let mut handled = false;

        if let Some(ref op) = insn.constant_op {
            match *op {
                ConstantOp::Adrp { dest_reg, page } => {
                    state.insert(dest_reg, SwOrigin::Page(page));
                    handled = true;
                }
                ConstantOp::AddSubImm { dest_reg, src_reg, imm } if imm >= 0 => {
                    if let Some(&SwOrigin::Page(page)) = state.get(&src_reg) {
                        let table_va = page.wrapping_add(imm as u64);
                        state.insert(dest_reg, SwOrigin::TableBase(table_va));
                        out.insert(insn.address, SwitchAnnotation::TableBase { table_va });
                        handled = true;
                    } else {
                        state.remove(&dest_reg);
                        handled = true;
                    }
                }
                ConstantOp::MovReg { dest_reg, src_reg } => {
                    if let Some(&origin) = state.get(&src_reg) {
                        state.insert(dest_reg, origin);
                    } else {
                        state.remove(&dest_reg);
                    }
                    handled = true;
                }
                ConstantOp::MovImm { dest_reg, .. }
                | ConstantOp::MovKeep { dest_reg, .. }
                | ConstantOp::AddSubImm { dest_reg, .. }
                | ConstantOp::Kill { dest_reg } => {
                    state.remove(&dest_reg);
                    handled = true;
                }
                ConstantOp::KillPair { dest_reg1, dest_reg2 } => {
                    state.remove(&dest_reg1);
                    state.remove(&dest_reg2);
                    handled = true;
                }
            }
        }

        if !handled {
            if let Some(ref rra) = insn.reg_reg_access {
                if load_mnemonics.contains(&insn.mnemonic.as_str()) {
                    if let Some(&SwOrigin::TableBase(table_va)) = state.get(&rra.base_reg) {
                        if let Some(ref li) = insn.load_info {
                            state.insert(li.dest_reg, SwOrigin::LoadedOffset {
                                table_va,
                                index_reg: rra.index_reg,
                            });
                            out.insert(insn.address, SwitchAnnotation::OffsetLoad {
                                table_va, index_reg: rra.index_reg, shift: rra.shift,
                            });
                            handled = true;
                        } else if let Some(dest) = parse_first_register(&insn.operands) {
                            state.insert(dest, SwOrigin::LoadedOffset {
                                table_va,
                                index_reg: rra.index_reg,
                            });
                            out.insert(insn.address, SwitchAnnotation::OffsetLoad {
                                table_va, index_reg: rra.index_reg, shift: rra.shift,
                            });
                            handled = true;
                        }
                    }
                }
            }
        }

        if !handled && insn.mnemonic == "ADD" {
            if let Some((dest, srcs)) = parse_add_three_regs(&insn.operands) {
                let mut table_va: Option<u64> = None;
                let mut index_reg: Option<u16> = None;
                for s in &srcs {
                    if let Some(&SwOrigin::TableBase(va)) = state.get(s) {
                        table_va = Some(va);
                    }
                    if let Some(&SwOrigin::LoadedOffset { table_va: lva, index_reg: ir, .. }) = state.get(s) {
                        if table_va == Some(lva) || table_va.is_none() {
                            table_va = Some(lva);
                            index_reg = Some(ir);
                        }
                    }
                }
                if let (Some(va), Some(ir)) = (table_va, index_reg) {
                    state.insert(dest, SwOrigin::Target { table_va: va, index_reg: ir });
                    out.insert(insn.address, SwitchAnnotation::TargetCompute { table_va: va });
                    handled = true;
                } else {
                    state.remove(&dest);
                    handled = true;
                }
            }
        }

        if insn.mnemonic == "BR" {
            if let Some(target_reg) = insn.indirect_call_reg {
                if let Some(&SwOrigin::Target { table_va, index_reg }) = state.get(&target_reg) {
                    out.insert(insn.address, SwitchAnnotation::Dispatch { table_va, index_reg });
                }
            }
            state.clear();
        }

        if insn.is_call {
            for r in 0..=18u16 { state.remove(&r); }
            state.remove(&30);
        }
        let _ = handled;
    }

    out
}

fn detect_switch_x86(
    instructions: &[DisassembledInstruction],
) -> HashMap<u64, SwitchAnnotation> {
    let mut out: HashMap<u64, SwitchAnnotation> = HashMap::new();
    if instructions.is_empty() { return out; }

    let branch_targets: HashSet<u64> = instructions.iter()
        .filter_map(|i| i.branch_target)
        .chain(instructions.iter().filter_map(|i| i.call_target))
        .collect();
    let internal_targets: HashSet<u64> = {
        let addrs: HashSet<u64> = instructions.iter().map(|i| i.address).collect();
        branch_targets.iter().filter(|a| addrs.contains(a)).copied().collect()
    };

    let mut state: HashMap<u16, SwOrigin> = HashMap::new();

    for insn in instructions {
        if internal_targets.contains(&insn.address) {
            state.clear();
        }

        let has_reg_reg = insn.reg_reg_access.is_some();
        let is_load_mnem = matches!(insn.mnemonic.as_str(),
            "MOV" | "MOVZX" | "MOVSX" | "MOVSXD");

        if let Some(ref op) = insn.constant_op {
            match *op {
                ConstantOp::Adrp { dest_reg, page } => {
                    state.insert(dest_reg, SwOrigin::TableBase(page));
                    out.insert(insn.address, SwitchAnnotation::TableBase { table_va: page });
                }
                ConstantOp::MovImm { dest_reg, value } if value > 0x1000 => {
                    state.insert(dest_reg, SwOrigin::TableBase(value));
                }
                ConstantOp::MovReg { dest_reg, src_reg } => {
                    if let Some(&origin) = state.get(&src_reg) {
                        state.insert(dest_reg, origin);
                    } else {
                        state.remove(&dest_reg);
                    }
                }
                ConstantOp::AddSubImm { dest_reg, src_reg, imm: 0 } => {
                    let dest_origin = state.get(&dest_reg).copied();
                    let src_origin = state.get(&src_reg).copied();
                    let combined = x86_switch_combine(dest_origin, src_origin)
                        .or_else(|| x86_switch_combine(src_origin, dest_origin));
                    if let Some((table_va, index_reg)) = combined {
                        state.insert(dest_reg, SwOrigin::Target { table_va, index_reg });
                        out.insert(insn.address, SwitchAnnotation::TargetCompute { table_va });
                    } else {
                        state.remove(&dest_reg);
                    }
                }
                ConstantOp::Kill { dest_reg } if has_reg_reg && is_load_mnem => {
                    if let Some(ref rra) = insn.reg_reg_access {
                        if let Some(&SwOrigin::TableBase(table_va)) = state.get(&rra.base_reg) {
                            state.insert(dest_reg, SwOrigin::LoadedOffset {
                                table_va,
                                index_reg: rra.index_reg,
                            });
                            out.insert(insn.address, SwitchAnnotation::OffsetLoad {
                                table_va,
                                index_reg: rra.index_reg,
                                shift: rra.shift,
                            });
                        } else {
                            state.remove(&dest_reg);
                        }
                    }
                }
                ConstantOp::Kill { dest_reg }
                | ConstantOp::MovImm { dest_reg, .. }
                | ConstantOp::MovKeep { dest_reg, .. }
                | ConstantOp::AddSubImm { dest_reg, .. } => {
                    state.remove(&dest_reg);
                }
                ConstantOp::KillPair { dest_reg1, dest_reg2 } => {
                    state.remove(&dest_reg1);
                    state.remove(&dest_reg2);
                }
            }
        }

        if insn.is_unconditional_branch && !insn.is_call {
            if let Some(target_reg) = insn.indirect_call_reg {
                match state.get(&target_reg).copied() {
                    Some(SwOrigin::Target { table_va, index_reg })
                    | Some(SwOrigin::LoadedOffset { table_va, index_reg }) => {
                        out.insert(insn.address, SwitchAnnotation::Dispatch { table_va, index_reg });
                    }
                    _ => {}
                }
            }
            if let Some(ref rra) = insn.reg_reg_access {
                let table_va = state.get(&rra.base_reg)
                    .and_then(|o| if let SwOrigin::TableBase(va) = *o { Some(va) } else { None })
                    .or_else(|| insn.memory_offset.map(|d| d as u64));
                if let Some(tva) = table_va {
                    out.insert(insn.address, SwitchAnnotation::Dispatch {
                        table_va: tva,
                        index_reg: rra.index_reg,
                    });
                }
            }
            state.clear();
        }

        if insn.is_call {
            state.clear();
        }
    }

    out
}

fn x86_switch_combine(a: Option<SwOrigin>, b: Option<SwOrigin>) -> Option<(u64, u16)> {
    match (a, b) {
        (Some(SwOrigin::TableBase(va)), Some(SwOrigin::LoadedOffset { table_va, index_reg }))
            if va == table_va => Some((va, index_reg)),
        _ => None,
    }
}

fn detect_static_field_accesses(
    instructions: &[DisassembledInstruction],
    _arch: Architecture,
    global_annotations: &HashMap<u64, MetadataAnnotation>,
) -> HashMap<u64, (u64, i64)> {
    let mut out: HashMap<u64, (u64, i64)> = HashMap::new();

    let branch_targets: HashSet<u64> = instructions.iter()
        .filter_map(|i| i.branch_target)
        .chain(instructions.iter().filter_map(|i| i.call_target))
        .collect();
    let internal_targets: HashSet<u64> = {
        let addrs: HashSet<u64> = instructions.iter().map(|i| i.address).collect();
        branch_targets.iter().filter(|a| addrs.contains(a)).copied().collect()
    };

    let mut reg_state: HashMap<u16, u64> = HashMap::new();
    let mut reg_load_va: HashMap<u16, u64> = HashMap::new();
    let mut reg_static_fields_for: HashMap<u16, u64> = HashMap::new();

    let is_typeinfo = |va: u64| -> bool {
        global_annotations.get(&va)
            .map(|a| matches!(a.kind, MetadataAnnotationKind::TypeInfo))
            .unwrap_or(false)
    };

    for insn in instructions {
        if internal_targets.contains(&insn.address) {
            reg_state.clear();
            reg_load_va.clear();
            reg_static_fields_for.clear();
        }

        if let Some(ref li) = insn.load_info {
            if li.base_reg == RIP_PSEUDO_REG {
                reg_load_va.insert(li.dest_reg, li.offset as u64);
            } else if let Some(&base_va) = reg_state.get(&li.base_reg) {
                let load_va = (base_va as i64).wrapping_add(li.offset) as u64;
                reg_load_va.insert(li.dest_reg, load_va);
            } else {
                reg_load_va.remove(&li.dest_reg);
            }

            if let Some(&typeinfo_va) = reg_load_va.get(&li.base_reg) {
                if is_typeinfo(typeinfo_va) && li.offset > 0 && li.offset < 0x400 {
                    reg_static_fields_for.insert(li.dest_reg, typeinfo_va);
                }
            } else if let Some(&typeinfo_va) = reg_static_fields_for.get(&li.base_reg) {
                if li.dest_reg != li.base_reg {
                    out.insert(insn.address, (typeinfo_va, li.offset));
                    reg_static_fields_for.remove(&li.dest_reg);
                }
            } else {
                reg_static_fields_for.remove(&li.dest_reg);
            }
        }

        if let Some(ref op) = insn.constant_op {
            let writes_load = insn.load_info.is_some();
            let mut kill = |r: u16| {
                if !writes_load || insn.load_info.map(|li| li.dest_reg != r).unwrap_or(true) {
                    reg_load_va.remove(&r);
                    reg_static_fields_for.remove(&r);
                }
            };
            match *op {
                ConstantOp::MovImm { dest_reg, value } => {
                    reg_state.insert(dest_reg, value);
                    kill(dest_reg);
                }
                ConstantOp::MovKeep { dest_reg, imm16, shift } => {
                    if let Some(&current) = reg_state.get(&dest_reg) {
                        let mask = !(0xFFFFu64 << shift);
                        reg_state.insert(dest_reg, (current & mask) | ((imm16 as u64) << shift));
                    }
                    kill(dest_reg);
                }
                ConstantOp::AddSubImm { dest_reg, src_reg, imm } => {
                    if let Some(&src_val) = reg_state.get(&src_reg) {
                        reg_state.insert(dest_reg, (src_val as i64 + imm) as u64);
                    } else {
                        reg_state.remove(&dest_reg);
                    }
                    kill(dest_reg);
                }
                ConstantOp::MovReg { dest_reg, src_reg } => {
                    if let Some(&v) = reg_state.get(&src_reg) {
                        reg_state.insert(dest_reg, v);
                    } else {
                        reg_state.remove(&dest_reg);
                    }
                    if let Some(&v) = reg_load_va.get(&src_reg) {
                        reg_load_va.insert(dest_reg, v);
                    } else {
                        reg_load_va.remove(&dest_reg);
                    }
                    if let Some(&v) = reg_static_fields_for.get(&src_reg) {
                        reg_static_fields_for.insert(dest_reg, v);
                    } else {
                        reg_static_fields_for.remove(&dest_reg);
                    }
                }
                ConstantOp::Adrp { dest_reg, page } => {
                    reg_state.insert(dest_reg, page);
                    kill(dest_reg);
                }
                ConstantOp::Kill { dest_reg } => {
                    reg_state.remove(&dest_reg);
                    kill(dest_reg);
                }
                ConstantOp::KillPair { dest_reg1, dest_reg2 } => {
                    reg_state.remove(&dest_reg1);
                    reg_state.remove(&dest_reg2);
                    kill(dest_reg1);
                    kill(dest_reg2);
                }
            }
        }

        if insn.is_call {
            for r in 0..=18u16 {
                reg_state.remove(&r);
                reg_load_va.remove(&r);
                reg_static_fields_for.remove(&r);
            }
            reg_state.remove(&30);
            reg_load_va.remove(&30);
            reg_static_fields_for.remove(&30);
        }
    }

    out
}

fn parse_first_register(operands: &str) -> Option<u16> {
    let first = operands.split(',').next()?.trim().trim_start_matches('[');
    parse_arm64_reg(first)
}

fn parse_add_three_regs(operands: &str) -> Option<(u16, Vec<u16>)> {
    let parts: Vec<&str> = operands.split(',').map(|s| s.trim()).collect();
    if parts.len() < 3 { return None; }
    let dest = parse_arm64_reg(parts[0])?;
    let src1 = parse_arm64_reg(parts[1])?;
    let src2 = parse_arm64_reg(parts[2])?;
    Some((dest, vec![src1, src2]))
}

fn parse_arm64_reg(s: &str) -> Option<u16> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']').trim_end_matches(',').trim();
    let s = s.trim_start_matches(|c: char| c == 'X' || c == 'x' || c == 'W' || c == 'w');
    if s == "sp" || s == "SP" || s == "zr" || s == "ZR" { return Some(31); }
    s.parse::<u16>().ok()
}

#[derive(Default)]
struct InitCheckRanges {
    range_starts: HashSet<u64>,
    suppressed: HashSet<u64>,
}

fn detect_init_check_ranges(
    instructions: &[DisassembledInstruction],
    arch: Architecture,
    rva_to_name: &HashMap<u64, String>,
) -> InitCheckRanges {
    match arch {
        Architecture::Arm64 => detect_init_check_arm64(instructions, rva_to_name),
        Architecture::X86 | Architecture::X64 => detect_init_check_x86(instructions, rva_to_name),
        _ => InitCheckRanges::default(),
    }
}

fn detect_init_check_arm64(
    instructions: &[DisassembledInstruction],
    rva_to_name: &HashMap<u64, String>,
) -> InitCheckRanges {
    let mut out = InitCheckRanges::default();
    if instructions.len() < 4 {
        return out;
    }

    let scan_limit = instructions.len().min(20);

    let mut i = 0;
    while i < scan_limit {
        let insn = &instructions[i];

        let is_load_bit = insn.mnemonic == "LDR"
            || insn.mnemonic == "LDRB"
            || insn.mnemonic == "LDRH"
            || insn.mnemonic == "LDARB"
            || insn.mnemonic == "LDAR";

        if !is_load_bit {
            i += 1;
            continue;
        }

        let mut tbz_idx: Option<usize> = None;
        for j in (i + 1)..(i + 5).min(instructions.len()) {
            let m = instructions[j].mnemonic.as_str();
            if m == "TBZ" || m == "TBNZ" || m == "CBZ" || m == "CBNZ" {
                if let Some(target) = instructions[j].branch_target {
                    if target > instructions[j].address
                        && target <= instructions[j].address.wrapping_add(0x80)
                    {
                        tbz_idx = Some(j);
                        break;
                    }
                }
            }
            if instructions[j].is_call || instructions[j].is_return {
                break;
            }
        }

        let tbz_idx = match tbz_idx {
            Some(t) => t,
            None => { i += 1; continue; }
        };

        let test_insn = &instructions[tbz_idx];
        let target = test_insn.branch_target.unwrap();
        let bit_zero = test_insn.mnemonic == "CBZ"
            || test_insn.mnemonic == "CBNZ"
            || test_insn.operands.contains("#0,")
            || test_insn.operands.contains("#0]")
            || test_insn.operands.trim_end().ends_with("#0");
        if !bit_zero {
            i += 1;
            continue;
        }

        let mut end_idx: Option<usize> = None;
        let mut has_runtime_call = false;
        for j in (tbz_idx + 1)..instructions.len().min(tbz_idx + 16) {
            let cand = &instructions[j];
            if cand.address >= target {
                end_idx = Some(j);
                break;
            }
            if cand.is_call {
                if let Some(t) = cand.call_target {
                    if !rva_to_name.contains_key(&t) {
                        has_runtime_call = true;
                    }
                }
            }
            if cand.is_return {
                break;
            }
        }

        let end_idx = match end_idx {
            Some(e) if has_runtime_call => e,
            _ => { i += 1; continue; }
        };

        out.range_starts.insert(insn.address);
        for k in (i + 1)..end_idx {
            out.suppressed.insert(instructions[k].address);
        }

        i = end_idx;
    }

    out
}

fn detect_init_check_x86(
    instructions: &[DisassembledInstruction],
    rva_to_name: &HashMap<u64, String>,
) -> InitCheckRanges {
    let mut out = InitCheckRanges::default();
    if instructions.len() < 4 { return out; }

    let scan_limit = instructions.len().min(24);
    let mut i = 0;

    while i < scan_limit {
        let insn = &instructions[i];

        let is_mem_load = insn.load_info.is_some()
            && matches!(insn.mnemonic.as_str(), "MOV" | "MOVZX" | "MOVSX" | "MOVSXD");
        let is_cmp_test_mem = matches!(insn.mnemonic.as_str(), "CMP" | "TEST")
            && insn.memory_offset.is_some();

        if !is_mem_load && !is_cmp_test_mem {
            i += 1;
            continue;
        }

        let start_idx = i;
        let test_start = if is_cmp_test_mem {
            i
        } else {
            let mut found = None;
            for j in (i + 1)..(i + 4).min(instructions.len()) {
                if matches!(instructions[j].mnemonic.as_str(), "CMP" | "TEST") {
                    found = Some(j);
                    break;
                }
                if instructions[j].is_branch || instructions[j].is_return { break; }
            }
            match found { Some(f) => f, None => { i += 1; continue; } }
        };

        let mut jcc_idx: Option<usize> = None;
        for j in test_start..(test_start + 3).min(instructions.len()) {
            let m = instructions[j].mnemonic.as_str();
            if matches!(m, "JNE" | "JE" | "JNZ" | "JZ" | "JNS" | "JS") {
                if let Some(target) = instructions[j].branch_target {
                    if target > instructions[j].address
                        && target.saturating_sub(instructions[j].address) <= 0x80
                    {
                        jcc_idx = Some(j);
                        break;
                    }
                }
            }
            if instructions[j].is_unconditional_branch || instructions[j].is_return { break; }
        }

        let jcc_idx = match jcc_idx { Some(j) => j, None => { i += 1; continue; } };
        let target = instructions[jcc_idx].branch_target.unwrap();

        let mut end_idx: Option<usize> = None;
        let mut has_runtime_call = false;
        for j in (jcc_idx + 1)..instructions.len().min(jcc_idx + 20) {
            let cand = &instructions[j];
            if cand.address >= target { end_idx = Some(j); break; }
            if cand.is_call {
                if let Some(t) = cand.call_target {
                    if !rva_to_name.contains_key(&t) { has_runtime_call = true; }
                } else {
                    has_runtime_call = true;
                }
            }
            if cand.is_return { break; }
        }

        let end_idx = match end_idx {
            Some(e) if has_runtime_call => e,
            _ => { i += 1; continue; }
        };

        out.range_starts.insert(instructions[start_idx].address);
        for k in (start_idx + 1)..end_idx {
            out.suppressed.insert(instructions[k].address);
        }
        i = end_idx;
    }

    out
}
