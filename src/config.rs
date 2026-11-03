use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize a `dump*Target` value leniently. The field is documented as
/// 0 = Both, 1 = Flat, 2 = Split; anything outside that range (e.g. a stale
/// value from an older config) is clamped to 0 instead of failing the whole
/// config load — a single bad field used to silently disable every other
/// setting (the `dumpPseudoCTarget: 5000` bug).
fn de_target<'de, D>(deserializer: D) -> Result<u8, D::Error>
where
    D: Deserializer<'de>,
{
    let v = u32::deserialize(deserializer)?;
    Ok(v.min(2) as u8)
}

/// Primary method-body output mode for `dump.cs`.
///
/// Prefer this over the legacy `dumpDisassembly` / `dumpPseudoC` pair. Those
/// booleans remain for backward-compatible config.json files and are kept in
/// sync by [`Config::reconcile_method_body_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MethodBodyMode {
    #[default]
    Off,
    Assembly,
    Csharp,
    Both,
}

impl MethodBodyMode {
    pub fn from_flags(dump_disassembly: bool, dump_pseudo_c: bool) -> Self {
        match (dump_disassembly, dump_pseudo_c) {
            (true, true) => Self::Both,
            (true, false) => Self::Assembly,
            (false, true) => Self::Csharp,
            (false, false) => Self::Off,
        }
    }

    pub fn flags(self) -> (bool, bool) {
        match self {
            Self::Off => (false, false),
            Self::Assembly => (true, false),
            Self::Csharp => (false, true),
            Self::Both => (true, true),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Assembly => "assembly",
            Self::Csharp => "csharp",
            Self::Both => "both",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "false" | "0" => Some(Self::Off),
            "assembly" | "asm" | "disassembly" | "disasm" => Some(Self::Assembly),
            "csharp" | "c#" | "cs" | "pseudo" | "pseudo-c" | "pseudoc" => Some(Self::Csharp),
            "both" | "all" => Some(Self::Both),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Config {
    pub dump_method: bool,
    pub dump_field: bool,
    pub dump_property: bool,
    pub dump_attribute: bool,
    pub dump_method_offset: bool,
    pub dump_field_offset: bool,
    pub dump_type_def_index: bool,
    pub dump_assembly_name: bool,
    pub generate_struct: bool,
    pub generate_dummy_dll: bool,
    pub require_any_key: bool,
    pub dummy_dll_add_token: bool,
    pub force_il2cpp_version: bool,
    pub force_version: f64,
    pub force_dump: bool,
    pub no_redirected_pointer: bool,
    pub split_dump_per_type: bool,
    pub generate_generics_dump: bool,
    pub dump_generics_rgctx: bool,
    pub dump_generics_method_specs: bool,
    pub dump_generics_custom_attributes: bool,
    pub dump_generics_string_literals: bool,
    pub dump_generics_metadata_usages: bool,
    pub dump_generics_vtables: bool,
    pub dump_generics_interfaces: bool,
    /// Preferred single switch for method-body output:
    /// `"off"` | `"assembly"` | `"csharp"` | `"both"`.
    /// This is the only method-body mode field written to config.json.
    pub method_body_mode: MethodBodyMode,
    /// Internal / legacy: accepted on deserialize for old config.json files,
    /// kept in sync with [`Self::method_body_mode`], never written back out.
    #[serde(default, skip_serializing)]
    pub dump_disassembly: bool,
    /// Where to embed assembly listings: 0 = Both, 1 = Flat dump.cs, 2 = Split.
    #[serde(
        rename = "assemblyTarget",
        alias = "dumpDisassemblyTarget",
        deserialize_with = "de_target"
    )]
    pub dump_disassembly_target: u8,
    /// Internal / legacy: accepted on deserialize for old config.json files,
    /// kept in sync with [`Self::method_body_mode`], never written back out.
    /// Engine code still reads this after [`Self::reconcile_method_body_mode`].
    #[serde(default, skip_serializing)]
    pub dump_pseudo_c: bool,
    /// Where to embed C# pseudo: 0 = Both, 1 = Flat dump.cs, 2 = Split.
    #[serde(
        rename = "csharpTarget",
        alias = "dumpPseudoCTarget",
        deserialize_with = "de_target"
    )]
    pub dump_pseudo_c_target: u8,
    /// Offer the interactive assembly/namespace/class picker before decompiling
    /// C#. Off (the default): C# mode decompiles every method with no prompt.
    #[serde(rename = "csharpInteractive", alias = "dumpPseudoCInteractive")]
    pub dump_pseudo_c_interactive: bool,
    /// Skip C# decompilation of methods whose body is larger than this many
    /// bytes (0 = no limit).
    #[serde(rename = "csharpMaxBodySize", alias = "dumpPseudoCMaxBodySize")]
    pub dump_pseudo_c_max_body_size: usize,
    /// Restrict method bodies (C# pseudo, IL recovery, and assembly) to these
    /// assemblies (empty = all).
    #[serde(
        rename = "csharpIncludeAssemblies",
        alias = "dumpPseudoCIncludeAssemblies"
    )]
    pub dump_pseudo_c_include_assemblies: Vec<String>,
    /// Restrict method bodies to these types / namespace prefixes (empty = all).
    #[serde(rename = "csharpIncludeTypes", alias = "dumpPseudoCIncludeTypes")]
    pub dump_pseudo_c_include_types: Vec<String>,
    /// Keep the FULL metadata dump when a class filter is active: every type
    /// and method header is still emitted, but only the selected
    /// assemblies/types get method bodies — the rest render as `{ }`.
    #[serde(rename = "csharpFullDump", alias = "dumpPseudoCFullDump")]
    pub dump_pseudo_c_full_dump: bool,
    #[serde(rename = "assemblyHexBytes", alias = "dumpDisassemblyHexBytes")]
    pub dump_disassembly_hex_bytes: bool,
    #[serde(rename = "assemblyFieldNames", alias = "dumpDisassemblyFieldNames")]
    pub dump_disassembly_field_names: bool,
    #[serde(rename = "assemblyAnnotations", alias = "dumpDisassemblyAnnotations")]
    pub dump_disassembly_annotations: bool,
    #[serde(rename = "assemblyCfg", alias = "dumpDisassemblyCfg")]
    pub dump_disassembly_cfg: bool,
    #[serde(
        rename = "maxAssemblyInstructions",
        alias = "maxDisassemblyInstructions"
    )]
    pub max_disassembly_instructions: usize,
    pub generate_cpp_scaffold: bool,
    pub mangle_names: bool,
    pub enhanced_ida_metadata: bool,
    pub generate_unity_headers: bool,
    pub compiler_layout: String,
    pub use_topological_sort: bool,
    pub codm: bool,
    /// Thread-static fields, FieldRVA blobs, and `static_metadata.json` export.
    pub dump_static_field_metadata: bool,
    /// Include hex/metadata for FieldRVA (`PrivateImplementationDetails`) entries.
    pub dump_field_rva_data: bool,
    pub max_field_rva_dump_bytes: usize,
    /// Recover real CIL method bodies into the generated DummyDLLs
    /// (Cpp2IL-style IL recovery), so dnSpy/ILSpy show decompilable C# instead
    /// of empty stubs. Requires `generate_dummy_dll`. Supported on ARM64, x64
    /// and ARM32. Toggle per run with `--il-recovery` / `--no-il-recovery`.
    pub il_recovery: bool,
    /// Pseudo-code engine: `"native"` (the built-in recovery pipeline). The
    /// `"ghidra"`/`"retdec"` backends were removed; any other value disables
    /// pseudo-C rather than silently falling back to a different engine.
    pub pseudo_c_engine: String,
    /// Grade every recovered method with the differential verifier and stamp the
    /// non-clean ones into the output, so a reader can tell a body the oracle
    /// proved from one it could not. Costs the verification pass (~5% of a dump)
    /// on top of decompilation. Off = the previous output, byte-for-byte.
    pub annotate_recovery: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dump_method: true,
            dump_field: true,
            dump_property: true,
            dump_attribute: true,
            dump_method_offset: true,
            dump_field_offset: true,
            dump_type_def_index: true,
            dump_assembly_name: true,
            generate_struct: true,
            generate_dummy_dll: true,
            require_any_key: true,
            dummy_dll_add_token: true,
            force_il2cpp_version: false,
            force_version: 24.3,
            force_dump: false,
            no_redirected_pointer: false,
            split_dump_per_type: true,
            generate_generics_dump: true,
            dump_generics_rgctx: true,
            dump_generics_method_specs: true,
            dump_generics_custom_attributes: true,
            dump_generics_string_literals: true,
            dump_generics_metadata_usages: true,
            dump_generics_vtables: true,
            dump_generics_interfaces: true,
            method_body_mode: MethodBodyMode::Off,
            dump_disassembly: false,
            dump_disassembly_target: 0,
            dump_pseudo_c: false,
            dump_pseudo_c_target: 0,
            dump_pseudo_c_interactive: false,
            dump_pseudo_c_max_body_size: 0, // 0 = decompile every method body
            dump_pseudo_c_include_assemblies: Vec::new(),
            dump_pseudo_c_include_types: Vec::new(),
            dump_pseudo_c_full_dump: false,
            dump_disassembly_hex_bytes: true,
            dump_disassembly_field_names: true,
            dump_disassembly_annotations: true,
            dump_disassembly_cfg: true,
            max_disassembly_instructions: 512,
            generate_cpp_scaffold: true,
            mangle_names: true,
            enhanced_ida_metadata: true,
            generate_unity_headers: true,
            compiler_layout: "GCC".to_string(),
            use_topological_sort: true,
            codm: false,
            dump_static_field_metadata: false,
            dump_field_rva_data: false,
            max_field_rva_dump_bytes: 4096,
            il_recovery: false,
            pseudo_c_engine: "native".to_string(),
            annotate_recovery: true,
        }
    }
}

impl Config {
    /// Keep [`Self::method_body_mode`] and the legacy boolean pair aligned.
    ///
    /// - Old configs (no `methodBodyMode`, only `dumpPseudoC` / `dumpDisassembly`):
    ///   derive the mode from the booleans.
    /// - New configs (`methodBodyMode` set): mode wins and rewrites the booleans.
    pub fn reconcile_method_body_mode(&mut self) {
        let from_flags = MethodBodyMode::from_flags(self.dump_disassembly, self.dump_pseudo_c);
        if self.method_body_mode == MethodBodyMode::Off && from_flags != MethodBodyMode::Off {
            self.method_body_mode = from_flags;
        } else {
            self.apply_method_body_mode_to_flags();
        }
    }

    /// Force the legacy booleans from [`Self::method_body_mode`].
    pub fn apply_method_body_mode_to_flags(&mut self) {
        let (disasm, pseudo) = self.method_body_mode.flags();
        self.dump_disassembly = disasm;
        self.dump_pseudo_c = pseudo;
    }

    /// After CLI/UI toggles flip the legacy booleans, refresh the mode field.
    pub fn sync_method_body_mode_from_flags(&mut self) {
        self.method_body_mode =
            MethodBodyMode::from_flags(self.dump_disassembly, self.dump_pseudo_c);
    }

    pub fn set_method_body_mode(&mut self, mode: MethodBodyMode) {
        self.method_body_mode = mode;
        self.apply_method_body_mode_to_flags();
    }

    pub fn load_from_file(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Config = serde_json::from_str(&content)?;
        config.reconcile_method_body_mode();
        Ok(config)
    }

    pub fn save_to_file(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut synced = self.clone();
        synced.apply_method_body_mode_to_flags();
        let json = serde_json::to_string_pretty(&synced)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_config_derives_mode_from_bools() {
        let json = r#"{ "dumpPseudoC": true, "dumpDisassembly": false }"#;
        let mut cfg: Config = serde_json::from_str(json).unwrap();
        cfg.reconcile_method_body_mode();
        assert_eq!(cfg.method_body_mode, MethodBodyMode::Csharp);
        assert!(cfg.dump_pseudo_c);
        assert!(!cfg.dump_disassembly);
    }

    #[test]
    fn mode_wins_over_stale_bools() {
        let json =
            r#"{ "methodBodyMode": "assembly", "dumpPseudoC": true, "dumpDisassembly": false }"#;
        let mut cfg: Config = serde_json::from_str(json).unwrap();
        cfg.reconcile_method_body_mode();
        assert_eq!(cfg.method_body_mode, MethodBodyMode::Assembly);
        assert!(cfg.dump_disassembly);
        assert!(!cfg.dump_pseudo_c);
    }

    #[test]
    fn new_option_names_round_trip() {
        let json = r#"{
            "methodBodyMode": "both",
            "assemblyTarget": 1,
            "csharpTarget": 2,
            "csharpInteractive": true,
            "csharpMaxBodySize": 100,
            "csharpIncludeAssemblies": ["Assembly-CSharp"],
            "csharpIncludeTypes": ["Game.Player"],
            "assemblyHexBytes": false,
            "assemblyFieldNames": false,
            "assemblyAnnotations": false,
            "assemblyCfg": false,
            "maxAssemblyInstructions": 64
        }"#;
        let mut cfg: Config = serde_json::from_str(json).unwrap();
        cfg.reconcile_method_body_mode();
        assert_eq!(cfg.method_body_mode, MethodBodyMode::Both);
        assert_eq!(cfg.dump_disassembly_target, 1);
        assert_eq!(cfg.dump_pseudo_c_target, 2);
        assert!(cfg.dump_pseudo_c_interactive);
        assert_eq!(cfg.dump_pseudo_c_max_body_size, 100);
        assert_eq!(
            cfg.dump_pseudo_c_include_assemblies,
            vec!["Assembly-CSharp".to_string()]
        );
        assert_eq!(
            cfg.dump_pseudo_c_include_types,
            vec!["Game.Player".to_string()]
        );
        assert!(!cfg.dump_disassembly_hex_bytes);
        assert_eq!(cfg.max_disassembly_instructions, 64);
        let out = serde_json::to_string_pretty(&cfg).unwrap();
        assert!(out.contains("assemblyTarget"));
        assert!(out.contains("csharpIncludeAssemblies"));
        assert!(!out.contains("dumpDisassemblyTarget"));
        assert!(!out.contains("dumpPseudoCIncludeAssemblies"));
        assert!(!out.contains("\"dumpDisassembly\""));
        assert!(!out.contains("\"dumpPseudoC\""));
    }

    #[test]
    fn full_dump_with_class_filter_round_trips() {
        let cfg: Config =
            serde_json::from_str(r#"{ "csharpIncludeAssemblies": ["Assembly-CSharp.dll"], "csharpFullDump": true }"#)
                .unwrap();
        assert!(cfg.dump_pseudo_c_full_dump);
        let out = serde_json::to_string(&cfg).unwrap();
        assert!(out.contains("\"csharpFullDump\":true"));
        // The default is OFF, so an older config keeps the old behaviour.
        let old: Config = serde_json::from_str("{}").unwrap();
        assert!(!old.dump_pseudo_c_full_dump);
    }

    #[test]
    fn legacy_option_aliases_still_load() {
        let json = r#"{
            "dumpPseudoC": true,
            "dumpDisassemblyTarget": 2,
            "dumpPseudoCTarget": 1,
            "dumpPseudoCInteractive": true,
            "dumpPseudoCMaxBodySize": 50,
            "dumpPseudoCIncludeAssemblies": ["A"],
            "maxDisassemblyInstructions": 128
        }"#;
        let mut cfg: Config = serde_json::from_str(json).unwrap();
        cfg.reconcile_method_body_mode();
        assert_eq!(cfg.method_body_mode, MethodBodyMode::Csharp);
        assert_eq!(cfg.dump_disassembly_target, 2);
        assert_eq!(cfg.dump_pseudo_c_target, 1);
        assert!(cfg.dump_pseudo_c_interactive);
        assert_eq!(cfg.dump_pseudo_c_max_body_size, 50);
        assert_eq!(cfg.max_disassembly_instructions, 128);
    }

    #[test]
    fn both_mode_round_trips() {
        let mut cfg = Config::default();
        cfg.set_method_body_mode(MethodBodyMode::Both);
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(
            json.contains(r#""methodBodyMode":"both""#)
                || json.contains(r#""methodBodyMode": "both""#)
        );
        // Legacy bools must not pollute saved config.json.
        assert!(!json.contains("dumpDisassembly"));
        assert!(!json.contains("dumpPseudoC"));
        let mut again: Config = serde_json::from_str(&json).unwrap();
        again.reconcile_method_body_mode();
        assert_eq!(again.method_body_mode, MethodBodyMode::Both);
        assert!(again.dump_disassembly && again.dump_pseudo_c);
    }
}
