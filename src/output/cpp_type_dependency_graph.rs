use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppCompilerLayout {
    MSVC,
    GCC,
}

impl CppCompilerLayout {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "MSVC" => Self::MSVC,
            _ => Self::GCC,
        }
    }

    pub fn guess_from_binary(is_pe: bool) -> Self {
        if is_pe { Self::MSVC } else { Self::GCC }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Value,
    Reference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphTypeKind {
    Primitive,
    ValueStruct,
    Enum,
    ReferenceClass,
    Array,
    Pointer,
}

#[derive(Debug, Clone)]
pub struct GraphField {
    pub element_name: String,
    pub element_kind: GraphTypeKind,
    pub is_instance: bool,
    pub is_pointer: bool,
    pub pointer_depth: u32,
    pub is_array: bool,
    pub array_element_name: Option<String>,
    pub array_element_kind: Option<GraphTypeKind>,
    pub enum_underlying_name: Option<String>,
    pub pointer_terminal_kind: Option<GraphTypeKind>,
    pub pointer_terminal_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GraphType {
    pub name: String,
    pub kind: GraphTypeKind,
    pub parent: Option<String>,
    pub enum_underlying_name: Option<String>,
    pub fields: Vec<GraphField>,
    pub static_fields: Vec<GraphField>,
}

#[derive(Debug, Clone)]
struct CppTypeNode {
    #[allow(dead_code)]
    id: NodeId,
    name: String,
    kind: GraphTypeKind,
    incoming_value: BTreeSet<NodeId>,
    outgoing_value: BTreeSet<NodeId>,
    incoming_ref: BTreeSet<NodeId>,
    outgoing_ref: BTreeSet<NodeId>,
}

impl CppTypeNode {
    fn new(id: NodeId, name: &str, kind: GraphTypeKind) -> Self {
        Self {
            id,
            name: name.to_string(),
            kind,
            incoming_value: BTreeSet::new(),
            outgoing_value: BTreeSet::new(),
            incoming_ref: BTreeSet::new(),
            outgoing_ref: BTreeSet::new(),
        }
    }
}

#[derive(Debug)]
pub struct CyclicDependency {
    pub chains: Vec<(String, Vec<String>)>,
}

impl std::fmt::Display for CyclicDependency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Circular value-type dependency detected across {} types:",
            self.chains.len()
        )?;
        for (name, deps) in &self.chains {
            writeln!(f, "  {name} -> [{}]", deps.join(", "))?;
        }
        Ok(())
    }
}

impl std::error::Error for CyclicDependency {}

pub struct CppTypeDependencyGraph {
    nodes: Vec<CppTypeNode>,
    name_to_id: HashMap<String, NodeId>,
    already_processed: HashSet<NodeId>,
    processing_queue: VecDeque<NodeId>,
    queued_nodes: HashSet<NodeId>,
}

impl CppTypeDependencyGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            name_to_id: HashMap::new(),
            already_processed: HashSet::new(),
            processing_queue: VecDeque::new(),
            queued_nodes: HashSet::new(),
        }
    }

    fn get_or_create_node(&mut self, name: &str, kind: GraphTypeKind) -> NodeId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = NodeId(self.nodes.len());
        self.nodes.push(CppTypeNode::new(id, name, kind));
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    fn add_edge(&mut self, from: NodeId, to: NodeId, kind: EdgeKind) {
        if from == to {
            return;
        }
        if self.already_processed.contains(&to) {
            return;
        }
        match kind {
            EdgeKind::Value => {
                self.nodes[from.0].outgoing_value.insert(to);
                self.nodes[to.0].incoming_value.insert(from);
            }
            EdgeKind::Reference => {
                self.nodes[from.0].outgoing_ref.insert(to);
                self.nodes[to.0].incoming_ref.insert(from);
            }
        }
    }

    fn kind_is_value(kind: GraphTypeKind) -> bool {
        matches!(
            kind,
            GraphTypeKind::ValueStruct | GraphTypeKind::Enum | GraphTypeKind::Primitive
        )
    }

    fn kind_is_forward_declarable(kind: GraphTypeKind) -> bool {
        matches!(
            kind,
            GraphTypeKind::ReferenceClass | GraphTypeKind::ValueStruct
        )
    }

    pub fn add_type(&mut self, ty: &GraphType) {
        let node_id = self.get_or_create_node(&ty.name, ty.kind);

        if let Some(parent_name) = &ty.parent {
            let parent_id = self.get_or_create_node(parent_name, GraphTypeKind::ValueStruct);
            self.add_edge(node_id, parent_id, EdgeKind::Value);
        }

        if matches!(ty.kind, GraphTypeKind::Enum) {
            if let Some(underlying) = &ty.enum_underlying_name {
                let uid = self.get_or_create_node(underlying, GraphTypeKind::Primitive);
                self.add_edge(node_id, uid, EdgeKind::Value);
            }
        }

        for field in ty.fields.iter().filter(|f| f.is_instance) {
            self.wire_field_edges(node_id, field);
        }

        for field in &ty.static_fields {
            self.wire_field_edges(node_id, field);
        }

        if !self.processing_queue.contains(&node_id) && !self.already_processed.contains(&node_id) {
            self.processing_queue.push_back(node_id);
            self.queued_nodes.insert(node_id);
        }
    }

    fn wire_field_edges(&mut self, from: NodeId, field: &GraphField) {
        if field.is_pointer {
            if let (Some(term_name), Some(term_kind)) =
                (&field.pointer_terminal_name, field.pointer_terminal_kind)
            {
                if matches!(term_kind, GraphTypeKind::Enum) {
                    let tid = self.get_or_create_node(term_name, term_kind);
                    self.add_edge(from, tid, EdgeKind::Value);
                    return;
                }
            }
            let tid = self.get_or_create_node(&field.element_name, field.element_kind);
            self.add_edge(from, tid, EdgeKind::Reference);
            return;
        }

        if field.is_array {
            if let (Some(aname), Some(akind)) =
                (&field.array_element_name, field.array_element_kind)
            {
                let tid = self.get_or_create_node(aname, akind);
                let edge = if Self::kind_is_value(akind) {
                    EdgeKind::Value
                } else {
                    EdgeKind::Reference
                };
                self.add_edge(from, tid, edge);
                return;
            }
        }

        let tid = self.get_or_create_node(&field.element_name, field.element_kind);
        let edge = if Self::kind_is_value(field.element_kind) {
            EdgeKind::Value
        } else {
            EdgeKind::Reference
        };
        self.add_edge(from, tid, edge);
    }

    pub fn add_types_bulk(&mut self, types: &[GraphType]) {
        for ty in types {
            self.add_type(ty);
        }
    }

    pub fn derive_dependency_order(
        &mut self,
    ) -> std::result::Result<Vec<String>, CyclicDependency> {
        let mut ordered = Vec::with_capacity(self.processing_queue.len());

        loop {
            if self.processing_queue.is_empty() {
                break;
            }

            let remaining_before = self.processing_queue.len();

            for _ in 0..remaining_before {
                let node_id = match self.processing_queue.pop_front() {
                    Some(id) => id,
                    None => break,
                };

                let has_blocking_value_dep = self.nodes[node_id.0]
                    .outgoing_value
                    .iter()
                    .any(|dep| self.queued_nodes.contains(dep));
                let can_emit = !has_blocking_value_dep;
                if !can_emit {
                    self.processing_queue.push_back(node_id);
                    continue;
                }

                ordered.push(self.nodes[node_id.0].name.clone());

                let incoming_value: Vec<NodeId> = self.nodes[node_id.0]
                    .incoming_value
                    .iter()
                    .copied()
                    .collect();
                for ref_id in &incoming_value {
                    if self.queued_nodes.contains(ref_id) {
                        self.nodes[ref_id.0].outgoing_value.remove(&node_id);
                    }
                }

                let incoming_ref: Vec<NodeId> =
                    self.nodes[node_id.0].incoming_ref.iter().copied().collect();
                for ref_id in &incoming_ref {
                    self.nodes[ref_id.0].outgoing_ref.remove(&node_id);
                }

                self.nodes[node_id.0].incoming_value.clear();
                self.nodes[node_id.0].incoming_ref.clear();
                self.already_processed.insert(node_id);
                self.queued_nodes.remove(&node_id);
            }

            if self.processing_queue.len() == remaining_before {
                let mut chains: Vec<(String, Vec<String>)> =
                    Vec::with_capacity(self.processing_queue.len());
                for id in self.processing_queue.iter() {
                    let node = &self.nodes[id.0];
                    let deps: Vec<String> = node
                        .outgoing_value
                        .iter()
                        .filter(|d| self.queued_nodes.contains(d))
                        .map(|d| self.nodes[d.0].name.clone())
                        .collect();
                    chains.push((node.name.clone(), deps));
                }
                return Err(CyclicDependency { chains });
            }
        }

        Ok(ordered)
    }

    pub fn get_forward_declaration_candidates(&self) -> HashSet<String> {
        let mut fwd = HashSet::new();
        for node in &self.nodes {
            for dep_id in &node.outgoing_ref {
                let target = &self.nodes[dep_id.0];
                if Self::kind_is_forward_declarable(target.kind) {
                    fwd.insert(target.name.clone());
                }
            }
        }
        fwd
    }

    pub fn kind_of(&self, name: &str) -> Option<GraphTypeKind> {
        self.name_to_id.get(name).map(|id| self.nodes[id.0].kind)
    }

    pub fn all_known_names(&self) -> HashSet<String> {
        self.name_to_id.keys().cloned().collect()
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.name_to_id.clear();
        self.already_processed.clear();
        self.processing_queue.clear();
        self.queued_nodes.clear();
    }
}

impl Default for CppTypeDependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}
