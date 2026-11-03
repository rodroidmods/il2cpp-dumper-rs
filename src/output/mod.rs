/// Print an informational note to stderr, matching the CLI's own output.
///
/// The library emits these while the CLI is drawing its banner (version
/// resolution, metadata-variant notes). An unstyled `Info: …` line in the
/// middle of that output reads like a stray debug print; the `ℹ` marker keeps
/// it visually part of the CLI's progress output. No terminal styling crate is
/// used here: this module is shared verbatim with the Tauri/Android ports,
/// which do not depend on `console`.
pub fn note(msg: &str) {
    eprintln!("  ℹ {msg}");
}

pub mod cpp_scaffolding;
pub mod decompiler;
pub mod dummy_assembly_generator;
pub mod embedded_scripts;
pub mod generics;
pub mod header_constants;
pub mod name_mangler;
pub mod pseudo_c;
pub mod script_json;
pub mod static_field_exporter;
pub mod struct_generator;

pub use cpp_scaffolding::CppScaffolding;
pub use decompiler::Il2CppDecompiler;
pub use name_mangler::MangledNameBuilder;
pub use script_json::*;
pub use struct_generator::StructGenerator;
pub mod cpp_ast;
pub mod cpp_type_dependency_graph;
pub mod cpp_type_model;

pub mod header_manager;
pub mod recovery_integration;
pub mod unity_version;
