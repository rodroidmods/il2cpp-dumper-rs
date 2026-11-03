use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CppTypeGroup {
    RequiredForwardDefinitions,
    TypesFromMethods,
    TypesFromGenericMethods,
    TypesFromUsages,
    UnusedConcreteTypes,
}

impl CppTypeGroup {
    pub fn tag(self) -> &'static str {
        match self {
            CppTypeGroup::RequiredForwardDefinitions => "required_forward_definitions",
            CppTypeGroup::TypesFromMethods => "types_from_methods",
            CppTypeGroup::TypesFromGenericMethods => "types_from_generic_methods",
            CppTypeGroup::TypesFromUsages => "types_from_usages",
            CppTypeGroup::UnusedConcreteTypes => "unused_concrete_types",
        }
    }

    pub fn section_header(self) -> &'static str {
        match self {
            CppTypeGroup::RequiredForwardDefinitions => "Required forward definitions",
            CppTypeGroup::TypesFromMethods => "Application types from method calls",
            CppTypeGroup::TypesFromGenericMethods => "Application types from generic methods",
            CppTypeGroup::TypesFromUsages => "Application types from usages",
            CppTypeGroup::UnusedConcreteTypes => "Application unused value types",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplexValueKind {
    Struct,
    Union,
    Enum,
    Class,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CppType {
    Primitive {
        name: String,
        size_bytes: u32,
        alignment_bytes: u32,
    },
    Pointer {
        element: Box<CppType>,
        word_size_bytes: u32,
    },
    Array {
        element: Box<CppType>,
        length: u32,
    },
    Alias {
        name: String,
        element: Box<CppType>,
    },
    FnPtr {
        name: String,
        return_type: Box<CppType>,
        arguments: Vec<(String, CppType)>,
        word_size_bytes: u32,
    },
    Complex {
        name: String,
        kind: ComplexValueKind,
        size_bytes: u32,
        alignment_bytes: u32,
    },
    Enum {
        name: String,
        underlying: Box<CppType>,
    },
    ForwardDeclaration {
        name: String,
    },
}

impl CppType {
    pub fn name(&self) -> String {
        match self {
            CppType::Primitive { name, .. } => name.clone(),
            CppType::Pointer { element, .. } => format!("{} *", element.name()),
            CppType::Array { element, .. } => element.name(),
            CppType::Alias { name, .. } => name.clone(),
            CppType::FnPtr { name, .. } => name.clone(),
            CppType::Complex { name, .. } => name.clone(),
            CppType::Enum { name, .. } => name.clone(),
            CppType::ForwardDeclaration { name } => name.clone(),
        }
    }

    pub fn size_bytes(&self) -> u32 {
        match self {
            CppType::Primitive { size_bytes, .. } => *size_bytes,
            CppType::Pointer {
                word_size_bytes, ..
            } => *word_size_bytes,
            CppType::Array { element, length } => element.size_bytes().saturating_mul(*length),
            CppType::Alias { element, .. } => element.size_bytes(),
            CppType::FnPtr {
                word_size_bytes, ..
            } => *word_size_bytes,
            CppType::Complex { size_bytes, .. } => *size_bytes,
            CppType::Enum { underlying, .. } => underlying.size_bytes(),
            CppType::ForwardDeclaration { .. } => 0,
        }
    }

    pub fn alignment_bytes(&self) -> u32 {
        match self {
            CppType::Primitive {
                alignment_bytes, ..
            } => *alignment_bytes,
            CppType::Pointer {
                word_size_bytes, ..
            } => *word_size_bytes,
            CppType::Array { element, .. } => element.alignment_bytes(),
            CppType::Alias { element, .. } => element.alignment_bytes(),
            CppType::FnPtr {
                word_size_bytes, ..
            } => *word_size_bytes,
            CppType::Complex {
                alignment_bytes, ..
            } => *alignment_bytes,
            CppType::Enum { underlying, .. } => underlying.alignment_bytes(),
            CppType::ForwardDeclaration { .. } => 0,
        }
    }

    pub fn is_enum(&self) -> bool {
        matches!(self, CppType::Enum { .. })
    }

    pub fn is_complex_struct(&self) -> bool {
        matches!(
            self,
            CppType::Complex {
                kind: ComplexValueKind::Struct | ComplexValueKind::Class,
                ..
            }
        )
    }

    pub fn is_forward_declarable(&self) -> bool {
        matches!(
            self,
            CppType::Complex {
                kind: ComplexValueKind::Struct | ComplexValueKind::Class | ComplexValueKind::Union,
                ..
            }
        )
    }

    pub fn as_pointer(&self, word_size_bytes: u32) -> CppType {
        CppType::Pointer {
            element: Box::new(self.clone()),
            word_size_bytes,
        }
    }

    pub fn as_array(&self, length: u32) -> CppType {
        CppType::Array {
            element: Box::new(self.clone()),
            length,
        }
    }

    pub fn as_alias(&self, name: &str) -> CppType {
        CppType::Alias {
            name: name.to_string(),
            element: Box::new(self.clone()),
        }
    }

    pub fn to_field_string(&self, field_name: &str) -> String {
        match self {
            CppType::Pointer { element, .. } => {
                let inner_field = format!("*{field_name}");
                element.to_field_string(&inner_field)
            }
            CppType::Array { element, length } => {
                format!("{} [{length}]", element.to_field_string(field_name))
            }
            CppType::FnPtr {
                return_type,
                arguments,
                ..
            } => {
                let args = arguments
                    .iter()
                    .map(|(n, t)| {
                        if n.is_empty() {
                            t.name()
                        } else {
                            format!("{} {}", t.name(), n)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} (*{field_name})({args})", return_type.name())
            }
            CppType::Complex { name, kind, .. } => {
                let prefix = match kind {
                    ComplexValueKind::Struct | ComplexValueKind::Class => "struct ",
                    ComplexValueKind::Union => "union ",
                    ComplexValueKind::Enum => "enum ",
                };
                format!("{prefix}{name} {field_name}")
            }
            CppType::Enum { name, .. } => format!("{name} {field_name}"),
            CppType::ForwardDeclaration { name } => format!("struct {name} {field_name}"),
            _ => format!("{} {field_name}", self.name()),
        }
    }
}

#[derive(Debug, Default)]
pub struct CppTypeGroupRegistry {
    type_groups: HashMap<String, CppTypeGroup>,
    group_members: HashMap<CppTypeGroup, Vec<String>>,
}

impl CppTypeGroupRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assign(&mut self, type_name: &str, group: CppTypeGroup) {
        if let Some(existing) = self.type_groups.get(type_name).copied() {
            if existing == group {
                return;
            }
        } else {
            self.group_members
                .entry(group)
                .or_default()
                .push(type_name.to_string());
        }
        self.type_groups.insert(type_name.to_string(), group);
    }

    pub fn group_of(&self, type_name: &str) -> Option<CppTypeGroup> {
        self.type_groups.get(type_name).copied()
    }

    pub fn members(&self, group: CppTypeGroup) -> &[String] {
        self.group_members
            .get(&group)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn promote(&mut self, type_name: &str, new_group: CppTypeGroup) {
        if let Some(old) = self.type_groups.insert(type_name.to_string(), new_group) {
            if let Some(list) = self.group_members.get_mut(&old) {
                list.retain(|n| n != type_name);
            }
            if old == new_group {
                return;
            }
        }
        let entry = self.group_members.entry(new_group).or_default();
        if !entry.iter().any(|n| n == type_name) {
            entry.push(type_name.to_string());
        }
    }

    pub fn all_names(&self) -> impl Iterator<Item = &String> {
        self.type_groups.keys()
    }
}
