use super::cpp_type_dependency_graph::{
    CppCompilerLayout, CppTypeDependencyGraph, GraphField, GraphType, GraphTypeKind,
};
use super::cpp_type_model::{CppTypeGroup, CppTypeGroupRegistry};
use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;

static C_PRIMITIVE_TYPES: &[&str] = &[
    "void",
    "bool",
    "char",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
    "float",
    "double",
    "intptr_t",
    "uintptr_t",
    "Il2CppChar",
];

fn needs_struct_prefix(type_name: &str) -> bool {
    let base = type_name.trim_end_matches('*');
    !C_PRIMITIVE_TYPES.contains(&base)
}

fn strip_pointer(type_name: &str) -> (&str, u32) {
    let bytes = type_name.as_bytes();
    let mut depth: u32 = 0;
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b'*' {
        depth += 1;
        end -= 1;
    }
    let trimmed = type_name[..end].trim_end();
    (trimmed, depth)
}

#[derive(Debug, Clone)]
pub struct CppField {
    pub type_name: String,
    pub field_name: String,
    pub is_value_type: bool,
    pub is_custom_type: bool,
}

#[derive(Debug, Clone)]
pub struct CppVTableEntry {
    pub method_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CppRGCTXEntry {
    pub rgctx_type: i64,
    pub type_name: Option<String>,
    pub class_name: Option<String>,
    pub method_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CppTypeDecl {
    pub name: String,
    pub is_value_type: bool,
    pub parent: Option<String>,
    pub instance_fields: Vec<CppField>,
    pub static_fields: Vec<CppField>,
    pub vtable: Vec<CppVTableEntry>,
    pub rgctxs: Vec<CppRGCTXEntry>,
}

pub struct CppHeaderEmitter {
    pub compiler_layout: CppCompilerLayout,
    pub is_32bit: bool,
    pub is_pe: bool,
    pub forward_declared: HashSet<String>,
    pub emitted: HashSet<String>,
}

impl CppHeaderEmitter {
    pub fn new(compiler_layout: CppCompilerLayout, is_32bit: bool, is_pe: bool) -> Self {
        Self {
            compiler_layout,
            is_32bit,
            is_pe,
            forward_declared: HashSet::new(),
            emitted: HashSet::new(),
        }
    }

    fn classify_field(field: &CppField, known_value_types: &HashSet<String>) -> GraphField {
        let (base_name, pointer_depth) = strip_pointer(&field.type_name);
        let is_pointer = pointer_depth > 0;
        let element_name = base_name.to_string();
        let is_custom = field.is_custom_type;
        let terminal_is_value = is_custom && known_value_types.contains(&element_name);

        let (element_kind, pointer_terminal_kind) = if is_pointer {
            if !is_custom {
                (GraphTypeKind::Primitive, Some(GraphTypeKind::Primitive))
            } else if terminal_is_value {
                (GraphTypeKind::ValueStruct, Some(GraphTypeKind::ValueStruct))
            } else {
                (
                    GraphTypeKind::ReferenceClass,
                    Some(GraphTypeKind::ReferenceClass),
                )
            }
        } else if !is_custom {
            (GraphTypeKind::Primitive, None)
        } else if field.is_value_type || terminal_is_value {
            (GraphTypeKind::ValueStruct, None)
        } else {
            (GraphTypeKind::ReferenceClass, None)
        };

        GraphField {
            element_name: element_name.clone(),
            element_kind,
            is_instance: true,
            is_pointer,
            pointer_depth,
            is_array: false,
            array_element_name: None,
            array_element_kind: None,
            enum_underlying_name: None,
            pointer_terminal_kind,
            pointer_terminal_name: if is_pointer { Some(element_name) } else { None },
        }
    }

    fn to_graph_type(decl: &CppTypeDecl, known_value_types: &HashSet<String>) -> GraphType {
        let kind = if decl.is_value_type {
            GraphTypeKind::ValueStruct
        } else {
            GraphTypeKind::ReferenceClass
        };

        let mut fields: Vec<GraphField> = decl
            .instance_fields
            .iter()
            .map(|f| Self::classify_field(f, known_value_types))
            .collect();
        for f in fields.iter_mut() {
            f.is_instance = true;
        }

        let mut static_fields: Vec<GraphField> = decl
            .static_fields
            .iter()
            .map(|f| Self::classify_field(f, known_value_types))
            .collect();
        for f in static_fields.iter_mut() {
            f.is_instance = false;
        }

        GraphType {
            name: decl.name.clone(),
            kind,
            parent: decl.parent.clone(),
            enum_underlying_name: None,
            fields,
            static_fields,
        }
    }

    pub fn emit_all_with_groups(
        &mut self,
        types: &[CppTypeDecl],
        groups: Option<&CppTypeGroupRegistry>,
    ) -> Result<String> {
        let known_type_names: HashSet<String> = types.iter().map(|t| t.name.clone()).collect();
        let known_value_types: HashSet<String> = types
            .iter()
            .filter(|t| t.is_value_type)
            .map(|t| t.name.clone())
            .collect();
        let type_map: HashMap<String, &CppTypeDecl> =
            types.iter().map(|t| (t.name.clone(), t)).collect();

        let mut graph = CppTypeDependencyGraph::new();
        let graph_types: Vec<GraphType> = types
            .iter()
            .map(|d| Self::to_graph_type(d, &known_value_types))
            .collect();
        graph.add_types_bulk(&graph_types);

        let ordered_names = graph
            .derive_dependency_order()
            .map_err(|cycle| Error::CyclicDependency(cycle.to_string()))?;
        let fwd_candidates = graph.get_forward_declaration_candidates();

        let total_estimate = types.len() * 512;
        let mut buf = String::with_capacity(total_estimate);

        let mut sorted_fwd: Vec<&String> = fwd_candidates.iter().collect();
        sorted_fwd.sort();
        if !sorted_fwd.is_empty() {
            Self::write_section_banner(
                &mut buf,
                CppTypeGroup::RequiredForwardDefinitions.section_header(),
            );
            for fwd_name in sorted_fwd {
                if !known_type_names.contains(fwd_name)
                    && self.forward_declared.insert(fwd_name.clone())
                {
                    writeln!(buf, "struct {};", fwd_name).ok();
                }
            }
            writeln!(buf).ok();
        }

        let group_order = [
            CppTypeGroup::TypesFromMethods,
            CppTypeGroup::TypesFromGenericMethods,
            CppTypeGroup::TypesFromUsages,
            CppTypeGroup::UnusedConcreteTypes,
        ];

        if let Some(reg) = groups {
            let assigned: HashMap<String, CppTypeGroup> = ordered_names
                .iter()
                .filter_map(|n| reg.group_of(n).map(|g| (n.clone(), g)))
                .collect();

            for &group in &group_order {
                let mut wrote_banner = false;
                for name in &ordered_names {
                    if assigned.get(name).copied() != Some(group) {
                        continue;
                    }
                    if let Some(decl) = type_map.get(name) {
                        if !self.emitted.insert(decl.name.clone()) {
                            continue;
                        }
                        if !wrote_banner {
                            Self::write_section_banner(&mut buf, group.section_header());
                            wrote_banner = true;
                        }
                        self.emit_type_decl(&mut buf, decl, &type_map);
                        writeln!(buf).ok();
                    }
                }
            }

            let mut wrote_unassigned = false;
            for name in &ordered_names {
                if assigned.contains_key(name) {
                    continue;
                }
                if let Some(decl) = type_map.get(name) {
                    if !self.emitted.insert(decl.name.clone()) {
                        continue;
                    }
                    if !wrote_unassigned {
                        Self::write_section_banner(&mut buf, "Unclassified application types");
                        wrote_unassigned = true;
                    }
                    self.emit_type_decl(&mut buf, decl, &type_map);
                    writeln!(buf).ok();
                }
            }
        } else {
            for name in &ordered_names {
                if let Some(decl) = type_map.get(name) {
                    if self.emitted.insert(decl.name.clone()) {
                        self.emit_type_decl(&mut buf, decl, &type_map);
                        writeln!(buf).ok();
                    }
                }
            }
        }

        Ok(buf)
    }

    fn write_section_banner(buf: &mut String, name: &str) {
        writeln!(
            buf,
            "// ******************************************************************************"
        )
        .ok();
        writeln!(buf, "// * {name}").ok();
        writeln!(
            buf,
            "// ******************************************************************************"
        )
        .ok();
        writeln!(buf).ok();
    }

    pub fn emit_all(&mut self, types: &[CppTypeDecl]) -> Result<String> {
        let known_type_names: HashSet<String> = types.iter().map(|t| t.name.clone()).collect();
        let known_value_types: HashSet<String> = types
            .iter()
            .filter(|t| t.is_value_type)
            .map(|t| t.name.clone())
            .collect();
        let type_map: HashMap<String, &CppTypeDecl> =
            types.iter().map(|t| (t.name.clone(), t)).collect();

        let mut graph = CppTypeDependencyGraph::new();
        let graph_types: Vec<GraphType> = types
            .iter()
            .map(|d| Self::to_graph_type(d, &known_value_types))
            .collect();
        graph.add_types_bulk(&graph_types);

        let ordered_names = graph
            .derive_dependency_order()
            .map_err(|cycle| Error::CyclicDependency(cycle.to_string()))?;
        let fwd_candidates = graph.get_forward_declaration_candidates();

        let total_estimate = types.len() * 512;
        let mut buf = String::with_capacity(total_estimate);

        let mut sorted_fwd: Vec<&String> = fwd_candidates.iter().collect();
        sorted_fwd.sort();
        for fwd_name in sorted_fwd {
            if !known_type_names.contains(fwd_name)
                && self.forward_declared.insert(fwd_name.clone())
            {
                writeln!(buf, "struct {};", fwd_name).ok();
            }
        }
        if !fwd_candidates.is_empty() {
            writeln!(buf).ok();
        }

        for name in &ordered_names {
            if let Some(decl) = type_map.get(name) {
                if self.emitted.insert(decl.name.clone()) {
                    self.emit_type_decl(&mut buf, decl, &type_map);
                    writeln!(buf).ok();
                }
            }
        }

        Ok(buf)
    }

    fn emit_type_decl(
        &mut self,
        buf: &mut String,
        decl: &CppTypeDecl,
        type_map: &HashMap<String, &CppTypeDecl>,
    ) {
        self.emit_fields_struct(buf, decl, type_map);
        self.emit_rgctx_struct(buf, decl);
        self.emit_vtable_struct(buf, decl);
        self.emit_class_struct(buf, decl);
        self.emit_object_struct(buf, decl);
        self.emit_static_fields_struct(buf, decl);
    }

    fn emit_fields_struct(
        &mut self,
        buf: &mut String,
        decl: &CppTypeDecl,
        type_map: &HashMap<String, &CppTypeDecl>,
    ) {
        match self.compiler_layout {
            CppCompilerLayout::GCC => {
                self.emit_fields_struct_gcc(buf, decl, type_map);
            }
            CppCompilerLayout::MSVC => {
                self.emit_fields_struct_msvc(buf, decl);
            }
        }
    }

    fn emit_fields_struct_gcc(
        &mut self,
        buf: &mut String,
        decl: &CppTypeDecl,
        type_map: &HashMap<String, &CppTypeDecl>,
    ) {
        if !decl.is_value_type && decl.parent.is_none() {
            let align = if self.is_32bit { 4 } else { 8 };
            writeln!(
                buf,
                "struct __attribute__((aligned({}))) {}_Fields {{",
                align, decl.name
            )
            .ok();
        } else {
            writeln!(buf, "struct {}_Fields {{", decl.name).ok();
        }

        self.write_flattened_parent_fields(buf, decl, type_map);

        for field in &decl.instance_fields {
            Self::write_field_line(buf, field);
        }

        if decl.instance_fields.is_empty() && decl.parent.is_none() {
            writeln!(buf, "\tuint8_t __padding_empty;").ok();
        }

        writeln!(buf, "}};").ok();
    }

    fn write_flattened_parent_fields(
        &self,
        buf: &mut String,
        decl: &CppTypeDecl,
        type_map: &HashMap<String, &CppTypeDecl>,
    ) {
        if let Some(parent_name) = &decl.parent {
            if let Some(parent_decl) = type_map.get(parent_name) {
                self.write_flattened_parent_fields(buf, parent_decl, type_map);
                for field in &parent_decl.instance_fields {
                    Self::write_field_line(buf, field);
                }
            }
        }
    }

    fn emit_fields_struct_msvc(&mut self, buf: &mut String, decl: &CppTypeDecl) {
        if let Some(parent_name) = &decl.parent {
            writeln!(
                buf,
                "struct {}_Fields : {}_Fields {{",
                decl.name, parent_name
            )
            .ok();
        } else if !decl.is_value_type && self.is_pe {
            let align = if self.is_32bit { 4 } else { 8 };
            writeln!(
                buf,
                "struct __declspec(align({})) {}_Fields {{",
                align, decl.name
            )
            .ok();
        } else {
            writeln!(buf, "struct {}_Fields {{", decl.name).ok();
        }

        for field in &decl.instance_fields {
            Self::write_field_line(buf, field);
        }

        if decl.instance_fields.is_empty() && decl.parent.is_none() {
            writeln!(buf, "\tuint8_t __padding_empty;").ok();
        }

        writeln!(buf, "}};").ok();
    }

    fn write_field_line(buf: &mut String, field: &CppField) {
        if field.is_custom_type && needs_struct_prefix(&field.type_name) {
            writeln!(buf, "\tstruct {} {};", field.type_name, field.field_name).ok();
        } else {
            writeln!(buf, "\t{} {};", field.type_name, field.field_name).ok();
        }
    }

    fn emit_rgctx_struct(&self, buf: &mut String, decl: &CppTypeDecl) {
        if decl.rgctxs.is_empty() {
            return;
        }
        writeln!(buf, "struct {}_RGCTXs {{", decl.name).ok();
        for (i, rgctx) in decl.rgctxs.iter().enumerate() {
            match rgctx.rgctx_type as i32 {
                1 => {
                    let tn = rgctx.type_name.as_deref().unwrap_or("unknown");
                    writeln!(buf, "\tIl2CppType* _{i}_{tn};").ok();
                }
                2 => {
                    let cn = rgctx.class_name.as_deref().unwrap_or("unknown");
                    writeln!(buf, "\tIl2CppClass* _{i}_{cn};").ok();
                }
                3 => {
                    let mn = rgctx.method_name.as_deref().unwrap_or("unknown");
                    writeln!(buf, "\tMethodInfo* _{i}_{mn};").ok();
                }
                _ => {}
            }
        }
        writeln!(buf, "}};").ok();
    }

    fn emit_vtable_struct(&self, buf: &mut String, decl: &CppTypeDecl) {
        if decl.vtable.is_empty() {
            return;
        }
        writeln!(buf, "struct {}_VTable {{", decl.name).ok();
        for (i, entry) in decl.vtable.iter().enumerate() {
            let method_name = entry.method_name.as_deref().unwrap_or("unknown");
            writeln!(buf, "\tVirtualInvokeData _{i}_{method_name};").ok();
        }
        writeln!(buf, "}};").ok();
    }

    fn emit_class_struct(&self, buf: &mut String, decl: &CppTypeDecl) {
        writeln!(buf, "struct {}_c {{", decl.name).ok();
        writeln!(buf, "\tIl2CppClass_1 _1;").ok();

        if !decl.static_fields.is_empty() {
            writeln!(buf, "\tstruct {}_StaticFields* static_fields;", decl.name).ok();
        } else {
            writeln!(buf, "\tvoid* static_fields;").ok();
        }

        if !decl.rgctxs.is_empty() {
            writeln!(buf, "\t{}_RGCTXs* rgctx_data;", decl.name).ok();
        } else {
            writeln!(buf, "\tIl2CppRGCTXData* rgctx_data;").ok();
        }

        writeln!(buf, "\tIl2CppClass_2 _2;").ok();

        if !decl.vtable.is_empty() {
            writeln!(buf, "\t{}_VTable vtable;", decl.name).ok();
        } else {
            writeln!(buf, "\tVirtualInvokeData vtable[32];").ok();
        }

        writeln!(buf, "}};").ok();
    }

    fn emit_object_struct(&self, buf: &mut String, decl: &CppTypeDecl) {
        writeln!(buf, "struct {}_o {{", decl.name).ok();
        if !decl.is_value_type {
            writeln!(buf, "\t{}_c *klass;", decl.name).ok();
            writeln!(buf, "\tvoid *monitor;").ok();
        }
        writeln!(buf, "\t{}_Fields fields;", decl.name).ok();
        writeln!(buf, "}};").ok();
    }

    fn emit_static_fields_struct(&self, buf: &mut String, decl: &CppTypeDecl) {
        if decl.static_fields.is_empty() {
            return;
        }
        writeln!(buf, "struct {}_StaticFields {{", decl.name).ok();
        for field in &decl.static_fields {
            Self::write_field_line(buf, field);
        }
        writeln!(buf, "}};").ok();
    }
}
