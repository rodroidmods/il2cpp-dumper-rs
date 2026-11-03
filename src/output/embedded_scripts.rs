use crate::error::Result;
use std::fs;
use std::path::Path;

const SCRIPTS: &[(&str, &str)] = &[
    ("ida.py", include_str!("../../scripts/ida.py")),
    ("ida_py3.py", include_str!("../../scripts/ida_py3.py")),
    (
        "ida_with_struct.py",
        include_str!("../../scripts/ida_with_struct.py"),
    ),
    (
        "ida_with_struct_py3.py",
        include_str!("../../scripts/ida_with_struct_py3.py"),
    ),
    ("ghidra.py", include_str!("../../scripts/ghidra.py")),
    (
        "ghidra_with_struct.py",
        include_str!("../../scripts/ghidra_with_struct.py"),
    ),
    (
        "ghidra_wasm.py",
        include_str!("../../scripts/ghidra_wasm.py"),
    ),
    (
        "il2cpp_header_to_ghidra.py",
        include_str!("../../scripts/il2cpp_header_to_ghidra.py"),
    ),
    (
        "il2cpp_header_to_binja.py",
        include_str!("../../scripts/il2cpp_header_to_binja.py"),
    ),
    ("hopper-py3.py", include_str!("../../scripts/hopper-py3.py")),
];

const BINJA_SCRIPTS: &[(&str, &str)] = &[
    (
        "__init__.py",
        include_str!("../../scripts/Il2CppBinaryNinja/__init__.py"),
    ),
    (
        "plugin.json",
        include_str!("../../scripts/Il2CppBinaryNinja/plugin.json"),
    ),
];

pub fn write_scripts(output_dir: &Path) -> Result<()> {
    for (name, content) in SCRIPTS {
        fs::write(output_dir.join(name), content)?;
    }

    let binja_dir = output_dir.join("Il2CppBinaryNinja");
    fs::create_dir_all(&binja_dir)?;
    for (name, content) in BINJA_SCRIPTS {
        fs::write(binja_dir.join(name), content)?;
    }

    Ok(())
}
