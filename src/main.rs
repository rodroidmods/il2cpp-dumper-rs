use std::fs;
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use clap::Parser;
use console::{Emoji, style};
use dialoguer::{Input, Select};
use indicatif::{ProgressBar, ProgressStyle};

use il2cpp_dumper::config::{Config, MethodBodyMode};
use il2cpp_dumper::error::Result;
use il2cpp_dumper::executor::Il2CppExecutor;
use il2cpp_dumper::formats::elf::Elf;
use il2cpp_dumper::formats::macho::MachO;
use il2cpp_dumper::formats::nso::Nso;
use il2cpp_dumper::formats::pe::Pe;
use il2cpp_dumper::formats::wasm::Wasm;
use il2cpp_dumper::il2cpp::base::{Il2Cpp, VaSegment};
use il2cpp_dumper::il2cpp::metadata::{Metadata, MetadataVariant};
use il2cpp_dumper::output::decompiler::Il2CppDecompiler;
use il2cpp_dumper::output::struct_generator::StructGenerator;

/// Dump base supplied on the command line (`--dump-base`). A global rather
/// than an environment variable on purpose: the platform initialisers take the
/// value at the point the file is recognised as a dump, which is several calls
/// below argument parsing, and threading it through every signature for one
/// optional number would touch more code than it is worth. Set once in `run()`
/// before any thread is spawned, read once per run in the initialiser.
static CLI_DUMP_BASE: AtomicU64 = AtomicU64::new(0);

const MAGIC_METADATA: u32 = 0xFAB11BAF;
const MAGIC_ELF: u32 = 0x464C457F;
const MAGIC_PE: u16 = 0x5A4D;
const MAGIC_MACHO32: u32 = 0xFEEDFACE;
const MAGIC_MACHO64: u32 = 0xFEEDFACF;
const MAGIC_MACHOFAT: u32 = 0xBEBAFECA;
const MAGIC_NSO: u32 = 0x304F534E;
const MAGIC_WASM: u32 = 0x6D736100;

static SPARKLE: Emoji<'_, '_> = Emoji("✨ ", "* ");
static PACKAGE: Emoji<'_, '_> = Emoji("📦 ", "");
static GEAR: Emoji<'_, '_> = Emoji("⚙️  ", "");
static MAG: Emoji<'_, '_> = Emoji("🔍 ", "");
static LOCK: Emoji<'_, '_> = Emoji("🔓 ", "");
static SHIELD: Emoji<'_, '_> = Emoji("🛡️  ", "");
static WARN: Emoji<'_, '_> = Emoji("⚠️  ", "! ");
static FOLDER: Emoji<'_, '_> = Emoji("📂 ", "");
static ROCKET: Emoji<'_, '_> = Emoji("🚀 ", "");

const BANNER: &str = r#"
  ╦╦  ╔═╗╔═╗╔═╗  ╔╦╗╦ ╦╔╦╗╔═╗╔═╗╦═╗
  ║║  ╠═╝║  ╠═╝   ║║║ ║║║║╠═╝║╣ ╠╦╝
  ╩╩═╝╚  ╚═╝╩    ═╩╝╚═╝╩ ╩╩  ╚═╝╩╚═
"#;

#[derive(Parser)]
#[command(
    name = "il2cpp_dumper",
    version,
    about = "Recover C# from an IL2CPP binary: dump metadata, build DummyDlls, and lift native method bodies back to C#.",
    long_about = "Recover C# from an IL2CPP binary.\n\n\
Reads an IL2CPP binary (ELF, PE, Mach-O, NSO or WASM) plus its global-metadata.dat and writes a\n\
Dump0/ directory containing dump.cs (every type and method), an optional DummyDll/ .NET assembly,\n\
and a recovery report. With --cs the native method bodies are recovered as real CIL, so dnSpy /\n\
ILSpy decompile them to C# exactly like an ordinary managed assembly.\n\n\
The three positionals are the same as --binary / --metadata / --output.",
    after_help = "EXAMPLES:\n  \
# Full C# recovery of an Android .so into ./Dump0\n  \
il2cpp_dumper libil2cpp.so global-metadata.dat --cs\n\n  \
# x64 PE into an explicit output directory\n  \
il2cpp_dumper GameAssembly.dll global-metadata.dat -o out --cs\n\n  \
# Decompile only one assembly's method bodies\n  \
il2cpp_dumper libil2cpp.so global-metadata.dat --cs --pseudo-include-assembly Assembly-CSharp\n\n  \
# A memory-dumped image whose offsets are already runtime addresses\n  \
il2cpp_dumper dumped.bin global-metadata.dat --dump-base 0x7074053000 --cs"
)]
struct Cli {
    /// Path to the IL2CPP binary: an ELF .so, a PE .dll/.exe, a Mach-O image,
    /// an NSO or a WASM module. Same as the first positional.
    #[arg(
        long = "binary",
        short = 'b',
        value_name = "FILE",
        help_heading = "Input"
    )]
    il2cpp_binary_flag: Option<String>,
    /// Path to global-metadata.dat. Same as the second positional.
    #[arg(
        long = "metadata",
        short = 'm',
        value_name = "FILE",
        help_heading = "Input"
    )]
    metadata_flag: Option<String>,
    /// Directory to write the Dump0/ folder into [default: .]. Same as the
    /// third positional.
    #[arg(
        long = "output",
        short = 'o',
        value_name = "DIR",
        help_heading = "Input"
    )]
    output_dir_flag: Option<String>,
    /// Load a config.json instead of the built-in defaults. Command-line flags
    /// override the keys it sets.
    #[arg(
        long = "config",
        short = 'c',
        value_name = "FILE",
        help_heading = "Input"
    )]
    config: Option<String>,
    /// Force the CODM metadata/container variant. It is normally detected from
    /// the metadata itself; set this when the image uses the variant but
    /// detection misses it.
    #[arg(long = "codm", help_heading = "Container")]
    codm: bool,
    /// Export thread-static / FieldRVA metadata (`static_metadata.json`, dump.cs annotations).
    #[arg(long = "dump-static-metadata", help_heading = "Output")]
    dump_static_metadata: bool,
    /// Skip thread-static / FieldRVA static metadata export.
    #[arg(long = "no-dump-static-metadata", help_heading = "Output")]
    no_dump_static_metadata: bool,
    /// Treat the binary as a memory dump even when auto-detection does not
    /// recognise it (its pointers are already runtime addresses).
    #[arg(long = "force-dump", help_heading = "Container")]
    force_dump: bool,
    /// Base address of a memory dump, as hex (e.g. 0x7074053000). A dumped
    /// binary's tables use runtime addresses; without the base the address
    /// mapping is wrong and method offsets come out as -1. Equivalent to
    /// setting ILR_DUMP_BASE, and works without a terminal.
    #[arg(long = "dump-base", value_name = "HEX", help_heading = "Container")]
    dump_base: Option<String>,
    /// Method-body output mode: `off`, `assembly`, `csharp`, or `both`.
    /// Preferred over the legacy dumpDisassembly / dumpPseudoC pair in config.json.
    #[arg(
        long = "method-body-mode",
        value_name = "MODE",
        help_heading = "Decompilation"
    )]
    method_body_mode: Option<String>,
    /// Restrict method bodies (assembly / C# / IL recovery) to this assembly
    /// (repeatable, e.g. --pseudo-include-assembly Assembly-CSharp).
    #[arg(long = "pseudo-include-assembly", action = clap::ArgAction::Append, value_name = "ASSEMBLY", help_heading = "Decompilation")]
    pseudo_include_assembly: Vec<String>,
    /// Restrict method bodies to this type or namespace (repeatable, e.g.
    /// --pseudo-include-type UnityEngine.GameObject). A bare namespace matches
    /// every type inside it.
    #[arg(long = "pseudo-include-type", action = clap::ArgAction::Append, value_name = "TYPE", help_heading = "Decompilation")]
    pseudo_include_type: Vec<String>,
    /// Show the interactive assembly/namespace/class picker before dumping
    /// method bodies. Off by default: bodies dump for every method without
    /// prompting. Same effect as `"csharpInteractive": true` in the config.
    #[arg(long = "pseudo-select", help_heading = "Decompilation")]
    pseudo_select: bool,
    /// Force the interactive selection off even when `csharpInteractive`
    /// is set in the config (scripts / CI).
    #[arg(long = "no-pseudo-prompt", help_heading = "Decompilation")]
    no_pseudo_prompt: bool,
    /// Skip pseudo-C decompilation of methods whose body is larger than this
    /// many bytes (the giant coroutine/parser methods are the slowest).
    #[arg(
        long = "pseudo-max-body-size",
        value_name = "BYTES",
        help_heading = "Decompilation"
    )]
    pseudo_max_body_size: Option<usize>,
    /// Generate the DummyDll/ directory (a real .NET assembly skeleton with
    /// metadata, field offsets and stubs). Controlled by config.json `generateDummyDll`.
    #[arg(long = "generate-dummy-dll", help_heading = "C# recovery", hide = true)]
    generate_dummy_dll: bool,
    /// Skip DummyDll generation (overrides `generateDummyDll` in the config).
    #[arg(long = "no-generate-dummy-dll", help_heading = "C# recovery", hide = true)]
    no_generate_dummy_dll: bool,
    /// Recover real CIL method bodies into the DummyDlls. Controlled by config.json `methodBodyMode`.
    #[arg(long = "il-recovery", help_heading = "C# recovery", hide = true)]
    il_recovery: bool,
    /// Do not recover CIL method bodies; leave every DummyDll method a stub
    /// (overrides `ilRecovery` in the config).
    #[arg(long = "no-il-recovery", help_heading = "C# recovery", hide = true)]
    no_il_recovery: bool,
    /// One-flag shortcut for the C# decompilation workflow. Controlled by config.json `methodBodyMode: "csharp"`.
    #[arg(long = "cs", help_heading = "C# recovery", hide = true)]
    cs: bool,
    /// Do not grade the recovered methods. By default every method is run
    /// through the differential verifier and the ones it cannot prove clean are
    /// stamped in the output (and counted in the dump banner), so a reader can
    /// tell recovered logic from a body with known findings. This flag skips
    /// that pass for a faster dump at the cost of unlabelled output.
    #[arg(long = "no-annotate-recovery", help_heading = "C# recovery")]
    no_annotate_recovery: bool,
    /// Path to the IL2CPP binary (same as --binary).
    #[arg(value_name = "BINARY")]
    binary_pos: Option<String>,
    /// Path to global-metadata.dat (same as --metadata).
    #[arg(value_name = "METADATA")]
    metadata_pos: Option<String>,
    /// Directory to write the Dump0/ folder into (same as --output) [default: .].
    #[arg(value_name = "OUTPUT")]
    output_pos: Option<String>,
}

impl Cli {
    fn resolve(&self) -> std::result::Result<(String, String, String), String> {
        let binary = self
            .il2cpp_binary_flag
            .clone()
            .or_else(|| self.binary_pos.clone())
            .ok_or_else(|| {
                "no IL2CPP binary given — pass it as the first argument or with --binary/-b"
                    .to_string()
            })?;
        let metadata = self.metadata_flag.clone()
            .or_else(|| self.metadata_pos.clone())
            .ok_or_else(|| "no metadata given — pass global-metadata.dat as the second argument or with --metadata/-m".to_string())?;
        let output = self
            .output_dir_flag
            .clone()
            .or_else(|| self.output_pos.clone())
            .unwrap_or_else(|| ".".into());
        Ok((binary, metadata, output))
    }
}

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg} {elapsed:.dim}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

fn print_banner() {
    let version = env!("CARGO_PKG_VERSION");
    println!();
    println!(
        "  {} {}",
        style("Rodroid").magenta().bold(),
        style("Il2CppDumper").cyan().bold()
    );
    for line in BANNER.lines() {
        println!("  {}", style(line).cyan().bold());
    }
    println!(
        "  {}{}",
        style("  Version").dim(),
        style(format!(" v{version}")).magenta().bold()
    );
    println!("  {}", style("─────────────────────────────────────").dim());
    println!();
}

fn print_info(label: &str, value: &str) {
    println!(
        "  {} {}",
        style(format!("{label}:")).green().bold(),
        style(value).cyan()
    );
}

fn print_address(label: &str, addr: u64) {
    println!(
        "  {} {}{}",
        style(format!("{label}:")).green().bold(),
        style("0x").dim(),
        style(format!("{addr:x}")).yellow().bold()
    );
}

fn print_warn(msg: &str) {
    println!("  {} {}", WARN, style(msg).yellow());
}

fn print_success(msg: &str) {
    println!("  {} {}", style("✓").green().bold(), style(msg).green());
}

fn print_detection(label: &str) {
    println!(
        "  {} {}{}",
        MAG,
        style("Detected ").dim(),
        style(label).white().bold()
    );
}

fn read_magic_u32(data: &[u8]) -> u32 {
    if data.len() < 4 {
        return 0;
    }
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

fn read_magic_u16(data: &[u8]) -> u16 {
    if data.len() < 2 {
        return 0;
    }
    u16::from_le_bytes([data[0], data[1]])
}

fn is_valid_metadata_version(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    let ver = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    ver > 0 && ver < 200
}

fn try_decrypt_metadata(data: &mut Vec<u8>) -> Option<String> {
    if data.len() < 16 {
        return None;
    }

    let magic = MAGIC_METADATA.to_le_bytes();

    let k1 = magic[0] ^ data[0];
    if k1 != 0 && (1..4).all(|i| (magic[i] ^ data[i]) == k1) {
        let mut test = data.clone();
        for b in test.iter_mut() {
            *b ^= k1;
        }
        if is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!("Single-byte XOR (key: 0x{k1:02X})"));
        }
    }

    let key4: [u8; 4] = std::array::from_fn(|i| magic[i] ^ data[i]);
    if key4 != [0u8; 4] && !key4.windows(2).all(|w| w[0] == w[1]) {
        let mut test = data.clone();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key4[i % 4];
        }
        if is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!(
                "4-byte XOR (key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]
            ));
        }
    }

    let key8: [u8; 8] = std::array::from_fn(|i| if i < 4 { magic[i] ^ data[i] } else { data[i] });
    if key8[0..4] != [0u8; 4] {
        let expected_ver_bytes: Vec<u8> = (4..8).map(|i| data[i] ^ key8[i]).collect();
        let test_ver = i32::from_le_bytes([
            expected_ver_bytes[0],
            expected_ver_bytes[1],
            expected_ver_bytes[2],
            expected_ver_bytes[3],
        ]);
        if test_ver > 0 && test_ver < 200 {
            let mut test = data.clone();
            for (i, b) in test.iter_mut().enumerate() {
                *b ^= key8[i % 8];
            }
            if is_valid_metadata_version(&test) {
                *data = test;
                return Some(format!(
                    "8-byte XOR (key: {:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X})",
                    key8[0], key8[1], key8[2], key8[3], key8[4], key8[5], key8[6], key8[7]
                ));
            }
        }
    }

    for key_len in [16usize, 32, 64, 128, 256] {
        if data.len() < key_len * 2 {
            continue;
        }
        let key: Vec<u8> = (0..key_len)
            .map(|i| {
                if i < 4 {
                    magic[i] ^ data[i]
                } else {
                    data[i] ^ 0
                }
            })
            .collect();
        if key[0..4] == [0u8; 4] {
            continue;
        }
        let mut test = data[..8].to_vec();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key[i % key_len];
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            for (i, b) in data.iter_mut().enumerate() {
                *b ^= key[i % key_len];
            }
            return Some(format!("{key_len}-byte rolling XOR key"));
        }
    }

    {
        let mut test = data.clone();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key4[i % 4] ^ (i as u8);
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!(
                "Position-dependent XOR (base key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]
            ));
        }
    }

    {
        let mut test = data.clone();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key4[i % 4] ^ ((i & 0xFF) as u8);
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!(
                "Masked position XOR (base key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]
            ));
        }
    }

    {
        let header_size = 256usize.min(data.len());
        let mut test = data.clone();
        for i in 0..header_size {
            test[i] ^= key4[i % 4];
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!(
                "Header-only XOR ({header_size} bytes, key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]
            ));
        }
    }

    None
}

fn detect_unity_version(data: &[u8]) -> Option<String> {
    let mut best: Option<String> = None;
    let mut i = 0;
    while i + 12 < data.len() {
        // "2022.3.24f1" (pre-Unity 6) or "6000.6.0a6" (Unity 6, which the engine
        // spells with its full major version number).
        let looks_like_version = (data[i] == b'2'
            && data[i + 1] == b'0'
            && data[i + 2].is_ascii_digit()
            && data[i + 3].is_ascii_digit()
            && data[i + 4] == b'.'
            && data[i + 5].is_ascii_digit())
            || (data[i] == b'6'
                && data[i + 1] == b'0'
                && data[i + 2] == b'0'
                && data[i + 3] == b'0'
                && data[i + 4] == b'.'
                && data[i + 5].is_ascii_digit());
        if looks_like_version {
            let max_len = std::cmp::min(24, data.len() - i);
            let end = data[i..i + max_len]
                .iter()
                .position(|&b| !b.is_ascii_alphanumeric() && b != b'.')
                .unwrap_or(max_len);
            let candidate = &data[i..i + end];
            if candidate.len() >= 8 && candidate.len() <= 20 {
                if let Ok(s) = std::str::from_utf8(candidate) {
                    if s.chars().filter(|c| *c == '.').count() == 2
                        && (s.contains('f')
                            || s.contains('b')
                            || s.contains('a')
                            || s.contains('p'))
                        && s.ends_with(|c: char| c.is_ascii_digit())
                    {
                        if best.as_ref().map_or(true, |prev| s > prev.as_str()) {
                            best = Some(s.to_string());
                        }
                    }
                }
            }
            i += end.max(1);
        } else {
            i += 1;
        }
    }
    best
}

fn validate_hex(input: &String) -> std::result::Result<(), String> {
    let trimmed = input
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X");
    if trimmed.is_empty() {
        return Err("Address cannot be empty".into());
    }
    u64::from_str_radix(trimmed, 16)
        .map(|_| ())
        .map_err(|_| format!("'{}' is not a valid hex address", input))
}

fn parse_hex_input(input: &str) -> u64 {
    let trimmed = input
        .trim()
        .trim_start_matches("0x")
        .trim_start_matches("0X");
    u64::from_str_radix(trimmed, 16).unwrap_or(0)
}

/// The runtime base requested for a memory dump, whichever channel supplied it:
/// `--dump-base HEX` for a terminal, `ILR_DUMP_BASE` for callers that already
/// know it (the library front ends set it before calling in).
fn requested_runtime_base() -> Option<u64> {
    std::env::var("ILR_DUMP_BASE")
        .ok()
        .map(|s| parse_hex_input(&s))
        .filter(|v| *v != 0)
        .or_else(|| {
            let v = CLI_DUMP_BASE.load(Ordering::Relaxed);
            (v != 0).then_some(v)
        })
}

fn prompt_dump_address() -> Option<u64> {
    println!();
    println!(
        "  {} {}",
        WARN,
        style("This appears to be a memory dump file.")
            .yellow()
            .bold()
    );
    println!(
        "  {}",
        style("  Enter the il2cpp dump base address, or 0 to skip.").dim()
    );
    println!();

    let input: String = Input::new()
        .with_prompt(format!("  {} Dump base address (hex)", SHIELD))
        .default("0".into())
        .validate_with(validate_hex)
        .interact_text()
        .unwrap_or_else(|_| "0".into());

    let addr = parse_hex_input(&input);
    if addr != 0 { Some(addr) } else { None }
}

fn prompt_manual_addresses() -> Result<(u64, u64)> {
    println!();
    println!(
        "  {} {}",
        style("✗").red().bold(),
        style("Auto-detection failed. Manual input required.").red()
    );
    println!(
        "  {}",
        style("  Provide the registration addresses in hexadecimal.").dim()
    );
    println!();

    let cr_input: String = Input::new()
        .with_prompt(format!("  {} CodeRegistration (hex)", style("→").cyan()))
        .validate_with(|input: &String| {
            let trimmed = input
                .trim()
                .trim_start_matches("0x")
                .trim_start_matches("0X");
            if trimmed.is_empty() || trimmed == "0" {
                return Err("CodeRegistration address is required".to_string());
            }
            validate_hex(input)
        })
        .interact_text()
        .map_err(|_| {
            il2cpp_dumper::error::Error::Other("Failed to read CodeRegistration input".into())
        })?;

    let mr_input: String = Input::new()
        .with_prompt(format!(
            "  {} MetadataRegistration (hex)",
            style("→").cyan()
        ))
        .validate_with(|input: &String| {
            let trimmed = input
                .trim()
                .trim_start_matches("0x")
                .trim_start_matches("0X");
            if trimmed.is_empty() || trimmed == "0" {
                return Err("MetadataRegistration address is required".to_string());
            }
            validate_hex(input)
        })
        .interact_text()
        .map_err(|_| {
            il2cpp_dumper::error::Error::Other("Failed to read MetadataRegistration input".into())
        })?;

    let cr = parse_hex_input(&cr_input);
    let mr = parse_hex_input(&mr_input);
    println!();
    print_address("CodeRegistration", cr);
    print_address("MetadataRegistration", mr);
    println!();

    Ok((cr, mr))
}

fn resolve_codm(config: &Config, metadata: &Metadata) -> bool {
    config.codm || metadata.variant == MetadataVariant::Codm
}

fn init_elf(data: Vec<u8>, metadata: &Metadata, config: &Config) -> Result<Il2Cpp> {
    let is_64 = data.len() > 4 && data[4] == 2;
    print_detection(&format!("ELF{} format", if is_64 { "64" } else { "32" }));

    let mut elf = if metadata.variant == MetadataVariant::Codm {
        Elf::new_with_codm_diag(data, !is_64, true)?
    } else {
        Elf::new(data, !is_64)?
    };

    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };

    elf.set_properties(version, metadata.metadata_usages_count as u64);
    print_info("IL2CPP Version", &elf.stream.version.to_string());

    // A memory dump may be handed over with its runtime base (`--dump-base` /
    // ILR_DUMP_BASE). Two different facts can be true of such a file, and
    // conflating them is what leaves every method address at -1:
    //
    //   1. Its sections are described in the dump's own address space
    //      (`p_offset == p_vaddr`), so they must be rewritten before anything
    //      can be read. That is `is_dumped` plus the rewrite inside `Elf::load`.
    //   2. It contains *runtime* pointers (`image_base + file offset`) while
    //      still carrying the on-disk layout — a hybrid dump. Those need the
    //      base only as a fallback address translation, and as the anchor for
    //      locating the code-gen modules by assembly name (`il2cpp::dump_scan`).
    //
    // Rewriting the sections of a hybrid file destroys the on-disk mapping the
    // registration search depends on: the metadata registration is then never
    // found, `types` comes up empty, and output generation panics — while the
    // modules, whose pointers are runtime addresses, are not found either. So a
    // base given on the command line selects case 2, the rewrite is asked for
    // explicitly with `--force-dump`, and an interactive terminal keeps exactly
    // its previous behaviour.
    let runtime_base = requested_runtime_base();
    if config.force_dump || (runtime_base.is_none() && elf.check_dump()) {
        if let Some(addr) = runtime_base.or_else(prompt_dump_address) {
            elf.stream.image_base = addr;
            elf.is_dumped = true;
            print_info("Dump base", &format!("0x{addr:X}"));
            if !config.no_redirected_pointer {
                elf.load()?;
            }
        }
    }
    // Deliberately no `else` branch: `Elf::new` already loaded the file, so an
    // ordinary on-disk binary (and case 2) needs nothing further. Re-loading
    // here would run `process_relocations` a SECOND time on a file that is not
    // a dump, and relocation application is not always idempotent. The dump
    // path above re-loads because the base has to be known first —
    // `fix_program_segments`/`fix_dynamic_section` rewrite the mapping.

    let sp = spinner("Searching for registrations...");
    let method_count = metadata
        .method_defs
        .iter()
        .filter(|m| m.method_index >= 0)
        .count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();

    let mut helper = elf.get_section_helper(method_count, type_count, image_count);
    let code_reg = helper.find_code_registration();
    let metadata_reg = if metadata.variant == MetadataVariant::Codm {
        helper
            .find_metadata_registration_codm()
            .or_else(|| helper.find_metadata_registration())
    } else {
        helper.find_metadata_registration()
    };
    sp.finish_and_clear();

    // Print CR/MR only once — after the strategy that actually succeeds
    // (avoid printing a section-scan candidate, then printing again from ARM32/symbol).
    let mut found = false;
    let mut final_cr: Option<u64> = None;
    let mut final_mr: Option<u64> = None;
    let mut how = "";

    if elf.auto_plus_init(code_reg, metadata_reg).unwrap_or(false) {
        found = true;
        final_cr = code_reg;
        final_mr = metadata_reg;
        how = "Section scan";
    }

    if !found {
        if let Ok(Some((cr, mr))) = elf.symbol_search() {
            if elf.init_with_auto_plus(cr, mr).is_ok() {
                found = true;
                final_cr = Some(cr);
                final_mr = Some(mr);
                how = "Symbol table";
            }
        }
    }

    if !found {
        if let Some((cr, mr)) = elf.search_arm32(version) {
            if elf.init_with_auto_plus(cr, mr).is_ok() {
                found = true;
                final_cr = Some(cr);
                final_mr = Some(mr);
                how = "ARM32 search pattern";
            }
        }
    }

    if !found && (runtime_base.is_some() || elf.is_dumped) {
        // A dumped protected game keeps its registrations in memory the ELF
        // segments do not describe, so the section, symbol and ARM32 searches
        // all have nothing to find — the `.so` here is a stub. The method
        // tables are still reachable by name, and that lookup runs once the
        // metadata's assembly names are known (see `il2cpp::dump_scan`), so
        // continue instead of demanding addresses from a terminal. Under
        // `nohup` the prompt reads EOF and the whole run dies with every
        // method address left at -1, which is the symptom this avoids.
        print_warn(
            "Registrations not found by section/symbol search; \
             locating code-gen modules by name in the dump",
        );
    } else if !found {
        // prompt_manual_addresses already prints CR/MR once
        let (cr, mr) = prompt_manual_addresses()?;
        elf.init_with_auto_plus(cr, mr)?;
    } else {
        if !how.is_empty() {
            print_detection(how);
        }
        if let Some(cr) = final_cr {
            print_address("CodeRegistration", cr);
        }
        if let Some(mr) = final_mr {
            print_address("MetadataRegistration", mr);
        }
    }

    let elf_exports = elf.list_exported_symbols().unwrap_or_default();
    let mut il2cpp = Il2Cpp::from_elf(&elf);
    il2cpp.image_names = metadata
        .image_defs
        .iter()
        .filter_map(|image| metadata.get_string_from_index(image.name_index).ok())
        .collect();

    // Now that the assembly names are known, remember the dump's runtime base
    // and locate the code-gen modules with it. With the section search having
    // found nothing, this is the only path that turns a method registration into
    // a real address. Every hit is validated by the name at the far end of the
    // pointer and by the shape of the pointer array (`dump_scan`), so a wrong
    // base finds nothing rather than inventing addresses.
    if let Some(base) = runtime_base {
        if il2cpp.image_base == 0 {
            il2cpp.image_base = base;
        }
        // The file is a memory image even when its sections are still laid out
        // the on-disk way; say so, because `image_base` alone would otherwise
        // be ambiguous.
        il2cpp.is_dumped = true;
    }
    if il2cpp.image_base != 0 && il2cpp.code_gen_module_method_pointers.is_empty() {
        let names = il2cpp.image_names.clone();
        let located = il2cpp.locate_code_gen_modules_in_dump(&names);
        let pointers: usize = il2cpp
            .code_gen_module_method_pointers
            .values()
            .map(|v| v.len())
            .sum();
        if located > 0 {
            print_success(&format!(
                "Located {} code-gen modules in the dump ({pointers} method addresses)",
                located
            ));
        } else {
            print_warn(
                "Could not locate code-gen modules in the dump; method addresses will be -1",
            );
        }
    }
    il2cpp.codm = il2cpp.codm || resolve_codm(config, metadata);
    // Exception tables: .eh_frame → (func, lsda) map + .gcc_except_table bytes.
    {
        let fdes = elf
            .section_bytes_by_name(".eh_frame")
            .map(|(addr, bytes)| {
                il2cpp_dumper::recovery::exceptions::scan_eh_frame_fdes(
                    &bytes,
                    addr,
                    elf.image_base(),
                    if elf.is_32bit { 4 } else { 8 },
                )
            })
            .unwrap_or_default();
        if !fdes.is_empty() {
            if let Some((tab_rva, tab_bytes)) = elf.section_bytes_by_name(".gcc_except_table") {
                il2cpp.eh_tables = il2cpp_dumper::recovery::exceptions::EhTables::Elf {
                    fdes,
                    table: tab_bytes,
                    table_rva: tab_rva,
                    ptr_size: if elf.is_32bit { 4 } else { 8 },
                };
            }
        }
    }
    // ARM EHABI: a 32-bit ARM ELF carries `.ARM.exidx`/`.ARM.extab` in place of
    // DWARF `.eh_frame`, so the `fdes` scan above finds nothing on an ARM32
    // binary. Decode those sections instead — the compact-model descriptor list
    // is what names a managed `try`/`finally` region.
    if il2cpp.eh_tables.is_empty() {
        if let Some((exidx_rva, exidx)) = elf.section_bytes_by_name(".ARM.exidx") {
            if !exidx.is_empty() {
                let (extab_rva, extab) = elf
                    .section_bytes_by_name(".ARM.extab")
                    .unwrap_or((0, Vec::new()));
                if std::env::var("ILR_EHDBG").is_ok() {
                    eprintln!(
                        "[eh] ARM EHABI: {} exidx entries @{exidx_rva:#x}, {} extab bytes @{extab_rva:#x}",
                        exidx.len() / 8,
                        extab.len()
                    );
                }
                il2cpp.eh_tables = il2cpp_dumper::recovery::exceptions::EhTables::ArmEhAbi {
                    exidx,
                    exidx_rva,
                    extab,
                    extab_rva,
                };
            }
        }
    }
    il2cpp.exported_symbols = elf_exports.iter().map(|(n, _)| n.clone()).collect();
    for (name, addr) in elf_exports {
        let rva = il2cpp.get_rva(addr);
        if name.starts_with("il2cpp_") || name.starts_with("mono_") {
            il2cpp.api_export_rvas.insert(name, rva);
        } else if name.starts_with("__cxa") || name.starts_with("___cxa") {
            il2cpp.cxa_func_rvas.insert(rva, name);
        }
    }
    // PLT/GOT relocations (JUMP_SLOT/GLOB_DAT) name every imported slot —
    // the ELF analogue of dyld bind opcodes — so helpers calling through
    // PLT stubs get real names (libc, __cxa_*, …).
    for (slot_va, name) in &elf.bind_symbols {
        let rva = il2cpp.get_rva(*slot_va);
        if rva != 0 {
            il2cpp.bind_symbols.insert(rva, name.clone());
        }
    }
    Ok(il2cpp)
}

fn init_pe(data: Vec<u8>, metadata: &Metadata, config: &Config) -> Result<Il2Cpp> {
    let mut pe = Pe::new(data)?;
    print_detection(&format!(
        "PE{} format",
        if pe.is_32bit { "32" } else { "64" }
    ));

    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };

    pe.stream.version = version;
    pe.stream.is_32bit = pe.is_32bit;
    print_info("IL2CPP Version", &version.to_string());

    if config.force_dump || pe.check_dump() {
        if let Some(addr) = prompt_dump_address() {
            pe.stream.image_base = addr;
        }
    }

    let sp = spinner("Searching for registrations...");
    let method_count = metadata
        .method_defs
        .iter()
        .filter(|m| m.method_index >= 0)
        .count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();
    let mu_count = metadata.metadata_usages_count;

    let mut cr_addr = 0u64;
    let mut mr_addr = 0u64;

    if let Ok(Some((cr, mr))) = pe.symbol_search() {
        sp.finish_and_clear();
        print_detection("Symbol table");
        print_address("CodeRegistration", cr);
        print_address("MetadataRegistration", mr);
        cr_addr = cr;
        mr_addr = mr;
    }

    if cr_addr == 0 || mr_addr == 0 {
        let mut helper =
            pe.get_section_helper(method_count, type_count, mu_count, image_count, version);
        let code_reg = helper.find_code_registration();
        let metadata_reg = helper.find_metadata_registration();
        sp.finish_and_clear();
        if let (Some(cr), Some(mr)) = (code_reg, metadata_reg) {
            print_address("CodeRegistration", cr);
            print_address("MetadataRegistration", mr);
            cr_addr = cr;
            mr_addr = mr;
        }
    } else {
        sp.finish_and_clear();
    }

    if cr_addr == 0 || mr_addr == 0 {
        let (cr, mr) = prompt_manual_addresses()?;
        cr_addr = cr;
        mr_addr = mr;
    }

    let pe_image_base = pe.image_base();
    let va_segments: Vec<VaSegment> = pe
        .sections
        .iter()
        .map(|s| VaSegment {
            vaddr: s.virtual_address as u64 + pe_image_base,
            memsz: s.virtual_size as u64,
            offset: s.pointer_to_raw_data as u64,
        })
        .collect();

    let mut il2cpp = Il2Cpp::new(pe.stream.clone(), version, pe.is_32bit);
    il2cpp.va_segments = va_segments;
    il2cpp.image_base = pe_image_base;
    il2cpp.is_pe = true;
    il2cpp.codm = resolve_codm(config, metadata);
    il2cpp.arch = Some(if pe.is_32bit {
        il2cpp_dumper::disassembler::Architecture::X86
    } else {
        il2cpp_dumper::disassembler::Architecture::X64
    });
    il2cpp.init_with_auto_plus(cr_addr, mr_addr, &|addr| pe.map_vatr(addr), false)?;
    il2cpp.data_sections = pe.data_search_sections();
    // Exception tables: PE .pdata RUNTIME_FUNCTION list (func → unwind info).
    // SEH scope tables (real try/catch) are a follow-up; x64 C++ EH from
    // mingw builds carries GNUC-style LSDA reachable through .xdata.
    if let Some((pdata_rva, pdata)) = pe.section_bytes_by_name(".pdata") {
        let _ = pdata_rva;
        let mut funcs: Vec<(u64, u64)> = Vec::new();
        let entry_size = if pe.is_32bit { 8 } else { 12 };
        let mut off = 0usize;
        while off + entry_size <= pdata.len() {
            let rd = |o: usize| -> u32 {
                u32::from_le_bytes([pdata[o], pdata[o + 1], pdata[o + 2], pdata[o + 3]])
            };
            let start = rd(off);
            let end = rd(off + 4);
            if end == 0 {
                break; // sentinel
            }
            let unwind = if entry_size == 12 { rd(off + 8) } else { 0 };
            if start != 0 {
                funcs.push((start as u64, unwind as u64));
            }
            off += entry_size;
        }
        funcs.sort_unstable();
        // The `.xdata` section carries each function's UNWIND_INFO; the PE
        // SEH scope tables (`C_SCOPE_TABLE`) inside it decode to real
        // try/catch/finally regions (see `recovery::exceptions`).
        let (xdata_rva, xdata) = pe
            .section_bytes_by_name(".xdata")
            .unwrap_or((0, Vec::new()));
        il2cpp.eh_tables = il2cpp_dumper::recovery::exceptions::EhTables::Pe(
            il2cpp_dumper::recovery::exceptions::PeExceptionTables {
                funcs,
                xdata,
                xdata_rva,
                ptr_size: if pe.is_32bit { 4 } else { 8 },
            },
        );
    }
    if let Ok(exports) = pe.list_exported_symbols() {
        il2cpp.exported_symbols = exports.iter().map(|(n, _)| n.clone()).collect();
        for (name, rva) in exports {
            if name.starts_with("il2cpp_") || name.starts_with("mono_") {
                il2cpp.api_export_rvas.insert(name, rva);
            }
        }
    }
    // Import Directory / delay-load imports name every IAT slot — the PE
    // analogue of dyld bind opcodes — so helpers calling through imports
    // get real `dll!symbol` names.
    let _ = pe.collect_bind_symbols();
    for (slot_va, name) in &pe.bind_symbols {
        let rva = il2cpp.get_rva(*slot_va);
        if rva != 0 {
            il2cpp.bind_symbols.insert(rva, name.clone());
        }
    }
    Ok(il2cpp)
}

fn init_macho_fat(data: Vec<u8>, metadata: &Metadata, config: &Config) -> Result<Il2Cpp> {
    use il2cpp_dumper::formats::macho::{MH_MAGIC_64, extract_fat_slice, parse_fat};

    let arches = parse_fat(&data)?;
    print_detection(&format!("Fat Mach-O with {} architectures", arches.len()));
    println!();

    let items: Vec<String> = arches
        .iter()
        .enumerate()
        .map(|(i, arch)| {
            if arch.magic == MH_MAGIC_64 {
                format!("{}. 64-bit", i + 1)
            } else {
                format!("{}. 32-bit", i + 1)
            }
        })
        .collect();

    let selection = Select::new()
        .with_prompt(format!("  {} Select target architecture", GEAR))
        .items(&items)
        .default(0)
        .interact()
        .unwrap_or(0);

    let slice = extract_fat_slice(&data, &arches[selection])?;
    init_macho(slice, metadata, config)
}

fn init_macho(data: Vec<u8>, metadata: &Metadata, config: &Config) -> Result<Il2Cpp> {
    let magic = read_magic_u32(&data);
    let is_64 = magic == MAGIC_MACHO64;
    print_detection(&format!(
        "Mach-O {} format",
        if is_64 { "64-bit" } else { "32-bit" }
    ));

    let mut macho = if metadata.variant == MetadataVariant::Codm {
        MachO::new_with_codm_fixups(data, !is_64, true)?
    } else {
        MachO::new(data, !is_64)?
    };

    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };

    macho.stream.version = version;
    macho.stream.is_32bit = macho.is_32bit;
    print_info("IL2CPP Version", &version.to_string());

    let sp = spinner("Searching for registrations...");
    let method_count = metadata
        .method_defs
        .iter()
        .filter(|m| m.method_index >= 0)
        .count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();
    let mu_count = metadata.metadata_usages_count;

    let mut cr_addr = 0u64;
    let mut mr_addr = 0u64;

    if let Some((cr, mr)) = macho.symbol_search() {
        sp.finish_and_clear();
        print_detection("Symbol table");
        print_address("CodeRegistration", cr);
        print_address("MetadataRegistration", mr);
        cr_addr = cr;
        mr_addr = mr;
    }

    if cr_addr == 0 || mr_addr == 0 {
        if let Some((cr, mr)) = macho.search_mod_init_func(version) {
            sp.finish_and_clear();
            print_detection("__mod_init_func section");
            print_address("CodeRegistration", cr);
            print_address("MetadataRegistration", mr);
            cr_addr = cr;
            mr_addr = mr;
        }
    }

    if cr_addr == 0 || mr_addr == 0 {
        let mut helper =
            macho.get_section_helper(method_count, type_count, mu_count, image_count, version);
        let code_reg = helper.find_code_registration();
        let metadata_reg = if metadata.variant == MetadataVariant::Codm {
            helper
                .find_metadata_registration_codm()
                .or_else(|| helper.find_metadata_registration())
        } else {
            helper.find_metadata_registration()
        };
        sp.finish_and_clear();
        if let (Some(cr), Some(mr)) = (code_reg, metadata_reg) {
            print_address("CodeRegistration", cr);
            print_address("MetadataRegistration", mr);
            cr_addr = cr;
            mr_addr = mr;
        }
    } else {
        sp.finish_and_clear();
    }

    if cr_addr == 0 || mr_addr == 0 {
        let (cr, mr) = prompt_manual_addresses()?;
        cr_addr = cr;
        mr_addr = mr;
    }

    let va_segments: Vec<VaSegment> = macho
        .segments
        .iter()
        .map(|s| VaSegment {
            vaddr: s.vmaddr,
            memsz: s.vmsize,
            offset: s.fileoff,
        })
        .collect();

    // Helper: try a (cr, mr) pair with il2cpp.init and tell the caller
    // whether it actually parsed cleanly. The auto-detection on Mach-O
    // sometimes lands on byte-pattern false positives that look right
    // (matching type_count somewhere in __DATA_CONST) but blow up the
    // moment we dereference the struct's pointer fields -- typically
    // surfacing as "failed to fill whole buffer" when read_exact runs
    // past EOF. Catching that here lets us prompt the user for known-
    // good addresses (e.g. extracted from IDA) instead of dying.
    let try_init = |cr: u64, mr: u64| -> Result<Il2Cpp> {
        let mut il2cpp = Il2Cpp::new(macho.stream.clone(), version, macho.is_32bit);
        il2cpp.va_segments = va_segments.clone();
        il2cpp.codm = resolve_codm(config, metadata);
        il2cpp.init_with_auto_plus(cr, mr, &|addr| macho.map_vatr(addr), false)?;
        Ok(il2cpp)
    };

    let mut il2cpp = match try_init(cr_addr, mr_addr) {
        Ok(i) => i,
        Err(e) => {
            print_warn(&format!(
                "Auto-detected CodeRegistration / MetadataRegistration failed to parse ({}). \
                 Likely a byte-pattern false positive. Please enter the correct addresses \
                 (e.g. from IDA / Ghidra cross-references on s_Il2CppCodeRegistration / \
                 s_Il2CppMetadataRegistration).",
                e
            ));
            let (cr2, mr2) = prompt_manual_addresses()?;
            cr_addr = cr2;
            mr_addr = mr2;
            try_init(cr_addr, mr_addr)?
        }
    };

    il2cpp.data_sections = macho.data_search_sections();
    // Exception tables: __unwind_info (function → LSDA index) + __gcc_except_tab.
    {
        use il2cpp_dumper::recovery::exceptions::{EhTables, MachOExceptionTables};
        if let (Some((ui_rva, unwind_info)), Some((tab_rva, tab_bytes))) = (
            macho.section_bytes_by_name("__unwind_info"),
            macho.section_bytes_by_name("__gcc_except_tab"),
        ) {
            let _ = ui_rva; // offsets inside __unwind_info are section-relative
            il2cpp.eh_tables = EhTables::MachO(MachOExceptionTables {
                unwind_info,
                gcc_except_tab: tab_bytes,
                gcc_except_tab_file_rva: tab_rva,
                image_base: macho.image_base(),
                ptr_size: if macho.is_32bit { 4 } else { 8 },
            });
        }
    }

    if macho.is_32bit {
        for ptr in il2cpp.method_pointers.iter_mut() {
            if *ptr > 0 {
                *ptr -= 1;
            }
        }
        for ptr in il2cpp.custom_attribute_generators.iter_mut() {
            if *ptr > 0 {
                *ptr -= 1;
            }
        }
    }

    let macho_exports = macho.list_exported_symbols();
    il2cpp.exported_symbols = macho_exports.iter().map(|(n, _)| n.clone()).collect();
    for (name, addr) in macho_exports {
        if name.starts_with("il2cpp_") || name.starts_with("mono_") {
            let rva = il2cpp.get_rva(addr);
            il2cpp.api_export_rvas.insert(name, rva);
        }
    }
    // dyld bind tables name every GOT/lazy-stub slot authoritatively —
    // codegen helpers calling through those stubs can then be identified by
    // the import they reach (___cxa_throw, il2cpp_vm::*, …).
    for (slot_va, name) in &macho.bind_symbols {
        let rva = il2cpp.get_rva(*slot_va);
        if rva != 0 {
            il2cpp.bind_symbols.insert(rva, name.clone());
        }
    }

    Ok(il2cpp)
}

fn init_nso(data: Vec<u8>, metadata: &Metadata, config: &Config) -> Result<Il2Cpp> {
    print_detection("NSO (Nintendo Switch) format");

    let mut nso = Nso::new(data)?;

    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };

    nso.stream.version = version;
    nso.stream.is_32bit = nso.is_32bit;
    print_info("IL2CPP Version", &version.to_string());

    let sp = spinner("Searching for registrations...");
    let method_count = metadata
        .method_defs
        .iter()
        .filter(|m| m.method_index >= 0)
        .count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();
    let mu_count = metadata.metadata_usages_count;

    let mut helper =
        nso.get_section_helper(method_count, type_count, mu_count, image_count, version);
    let code_reg = helper.find_code_registration();
    let metadata_reg = helper.find_metadata_registration();
    sp.finish_and_clear();

    let (cr_addr, mr_addr) = if let (Some(cr), Some(mr)) = (code_reg, metadata_reg) {
        print_address("CodeRegistration", cr);
        print_address("MetadataRegistration", mr);
        (cr, mr)
    } else {
        prompt_manual_addresses()?
    };

    let stream_len = nso.stream.data().len() as u64;
    let mut il2cpp = Il2Cpp::new(nso.stream.clone(), version, nso.is_32bit);
    il2cpp.va_segments = vec![VaSegment {
        vaddr: 0,
        memsz: stream_len,
        offset: 0,
    }];
    il2cpp.codm = resolve_codm(config, metadata);
    il2cpp.init_with_auto_plus(cr_addr, mr_addr, &|addr| nso.map_vatr(addr), false)?;
    il2cpp.data_sections = nso.data_search_sections();

    if let Ok(nso_exports) = nso.list_exported_symbols() {
        il2cpp.exported_symbols = nso_exports.iter().map(|(n, _)| n.clone()).collect();
        for (name, addr) in nso_exports {
            if name.starts_with("il2cpp_") || name.starts_with("mono_") {
                let rva = il2cpp.get_rva(addr);
                il2cpp.api_export_rvas.insert(name, rva);
            }
        }
    }

    Ok(il2cpp)
}

fn init_wasm(data: Vec<u8>, metadata: &Metadata, config: &Config) -> Result<Il2Cpp> {
    print_detection("WebAssembly (WASM) format");

    let mut wasm = Wasm::new(data)?;

    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };

    wasm.stream.version = version;
    wasm.stream.is_32bit = wasm.is_32bit;
    print_info("IL2CPP Version", &version.to_string());

    let sp = spinner("Searching for registrations...");
    let method_count = metadata
        .method_defs
        .iter()
        .filter(|m| m.method_index >= 0)
        .count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();
    let mu_count = metadata.metadata_usages_count;

    let mut helper =
        wasm.get_section_helper(method_count, type_count, mu_count, image_count, version);
    let code_reg = helper.find_code_registration();
    let metadata_reg = helper.find_metadata_registration();
    sp.finish_and_clear();

    let (cr_addr, mr_addr) = if let (Some(cr), Some(mr)) = (code_reg, metadata_reg) {
        print_address("CodeRegistration", cr);
        print_address("MetadataRegistration", mr);
        (cr, mr)
    } else {
        prompt_manual_addresses()?
    };

    let stream_len = wasm.stream.data().len() as u64;
    let mut il2cpp = Il2Cpp::new(wasm.stream.clone(), version, wasm.is_32bit);
    il2cpp.va_segments = vec![VaSegment {
        vaddr: 0,
        memsz: stream_len,
        offset: 0,
    }];
    // WebAssembly bytecode, not machine code — see `Architecture::Wasm`.
    // Without this the 32-bit/non-PE default (`Arm32`) applies and the ARM
    // disassembler happily decodes wasm bytes as instructions.
    il2cpp.arch = Some(il2cpp_dumper::disassembler::Architecture::Wasm);
    il2cpp.codm = resolve_codm(config, metadata);
    il2cpp.init_with_auto_plus(cr_addr, mr_addr, &|addr| wasm.map_vatr(addr), true)?;
    il2cpp.data_sections = wasm.data_search_sections();
    Ok(il2cpp)
}

fn detect_format(data: &[u8]) -> &'static str {
    let magic32 = read_magic_u32(data);
    let magic16 = read_magic_u16(data);
    match magic32 {
        MAGIC_ELF => "elf",
        MAGIC_MACHO32 | MAGIC_MACHO64 => "macho",
        MAGIC_MACHOFAT => "macho_fat",
        MAGIC_NSO => "nso",
        MAGIC_WASM => "wasm",
        _ if magic16 == MAGIC_PE => "pe",
        _ => "unknown",
    }
}

fn format_file_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Discard any input already queued in the terminal. Without this, a stale
/// newline left in the input buffer (e.g. the Enter that launched the command,
/// or a key pressed while the binary was still compiling) is read by
/// dialoguer's raw-mode key reader as an immediate `Enter`, which instantly
/// confirms the checkbox menu with every item selected — the user never gets
/// to see or touch the checkboxes.
#[cfg(unix)]
fn drain_pending_input() {
    use std::os::fd::AsRawFd;
    // TCFLUSH: discard data received but not yet read. Ignore errors (e.g.
    // not a TTY) — in that case there is nothing stale to worry about.
    unsafe {
        let _ = libc::tcflush(std::io::stdin().as_raw_fd(), libc::TCIFLUSH);
    }
}

#[cfg(not(unix))]
fn drain_pending_input() {
    // Windows console input is read through the console crate; queued keys
    // there are intentional user input, so nothing to drain.
}

/// Interactive pseudo-C selection: checkbox lists of assemblies, then (if a
/// subset is picked) namespaces inside them, then (if a subset of namespaces
/// is picked) classes inside those. Only the chosen subset is decompiled.
///
/// Runs when method bodies are enabled, the user asked for the picker
/// (`--pseudo-select` / `csharpInteractive`), and no assembly/type filter
/// is already set in config.json or via --pseudo-include-* flags. A
/// non-interactive stdin (piped) skips the prompt and dumps all bodies.
fn prompt_pseudo_c_selection(
    config: &mut Config,
    metadata: &Metadata,
    il2cpp: &Il2Cpp,
    executor: &mut Il2CppExecutor,
) {
    use dialoguer::{Confirm, MultiSelect};
    use std::io::IsTerminal;

    if !std::io::stdin().is_terminal() {
        print_warn(
            "Method bodies: stdin is not a terminal, skipping assembly/type selection — dumping all \
             (use --pseudo-include-assembly / --pseudo-include-type to filter non-interactively)",
        );
        return;
    }
    // Clear any stale keystrokes so the first prompt actually waits for input.
    drain_pending_input();

    use il2cpp_dumper::output::pseudo_c::{PseudoAssemblyEntry, list_filter_tree};

    let tree = list_filter_tree(il2cpp, metadata, executor);
    if tree.is_empty() {
        print_warn("Method bodies: no decompilable methods found");
        return;
    }

    println!();
    println!(
        "  {}",
        style("─── Pseudo-C Selection ─────────────────").dim()
    );
    println!();

    // ---- Level 0: quick preset ---------------------------------------------
    // 135 assemblies is far too many to scroll one-by-one, so offer presets
    // first. The checkbox list still appears afterwards, but only for the
    // (much smaller) narrowed set, pre-checked, for fine-tuning.
    drain_pending_input();
    let presets = vec![
        format!("Decompile everything (all {} assemblies)", tree.len()),
        "Game code only — Assembly-CSharp*".to_string(),
        "Filter by keyword (type to search)".to_string(),
        "Pick manually from the checkbox list".to_string(),
    ];
    let preset = Select::new()
        .with_prompt("  What should be decompiled to C?  (↑/↓ to move, enter = confirm)")
        .items(&presets)
        .default(0)
        .interact()
        .unwrap_or(0);

    // Candidate set the checkbox list will operate on.
    let candidates: Vec<&PseudoAssemblyEntry> = match preset {
        0 => {
            // Everything: no filter at all.
            print_success(&format!("Decompiling all {} assemblies", tree.len()));
            return;
        }
        1 => {
            // The usual want: the game's own code, not the framework.
            tree.iter()
                .filter(|a| {
                    a.assembly
                        .trim_end_matches(".dll")
                        .starts_with("Assembly-CSharp")
                })
                .collect()
        }
        2 => {
            // Type a few keywords; every assembly whose name contains any of
            // them (case-insensitive) is offered, pre-checked.
            drain_pending_input();
            let kw: String = Input::new()
                .with_prompt("  Keywords (space/comma separated, e.g. SYBO UnityEngine)")
                .allow_empty(true)
                .interact_text()
                .unwrap_or_default();
            let keys: Vec<String> = kw
                .split(|c: char| c == ' ' || c == ',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            if keys.is_empty() {
                print_success(&format!("Decompiling all {} assemblies", tree.len()));
                return;
            }
            tree.iter()
                .filter(|a| {
                    let bare = a.assembly.trim_end_matches(".dll").to_lowercase();
                    keys.iter().any(|k| bare.contains(k.as_str()))
                })
                .collect()
        }
        _ => {
            // Full manual list.
            tree.iter().collect()
        }
    };
    if candidates.is_empty() {
        print_warn("No assemblies matched — decompiling all assemblies");
        return;
    }

    // ---- Level 1: checkbox list over the (narrowed) candidate set ----------
    let cand_items: Vec<String> = candidates
        .iter()
        .map(|a| {
            format!(
                "{}  ({} methods)",
                a.assembly.trim_end_matches(".dll"),
                a.method_count
            )
        })
        .collect();
    let defaults = vec![true; cand_items.len()];
    drain_pending_input();
    let chosen_asm = match MultiSelect::new()
        .with_prompt("  Select assemblies to decompile to C  (space = toggle, enter = confirm)")
        .items(&cand_items)
        .defaults(&defaults)
        .interact()
    {
        Ok(v) => v,
        Err(_) => {
            print_warn("Pseudo-C selection cancelled — decompiling all assemblies");
            return;
        }
    };
    if chosen_asm.is_empty() {
        print_warn("No assemblies selected — decompiling all assemblies");
        return;
    }
    let sel_assemblies: Vec<&PseudoAssemblyEntry> =
        chosen_asm.iter().map(|&i| candidates[i]).collect();
    config.dump_pseudo_c_include_assemblies =
        sel_assemblies.iter().map(|a| a.assembly.clone()).collect();
    let total: usize = sel_assemblies.iter().map(|a| a.method_count).sum();
    print_success(&format!(
        "Selected {} of {} assemblies ({} methods)",
        sel_assemblies.len(),
        candidates.len(),
        total
    ));

    // ---- Level 2: namespaces inside the selected assemblies ----------------
    // Default to `no` so a plain Enter skips the drill-down and goes straight
    // to decompiling the selected assemblies (the common want). Stale input is
    // drained above, so Enter can only come from the user pressing it now.
    drain_pending_input();
    let confirm_ns = Confirm::new()
        .with_prompt("  Filter by namespace inside the selected assemblies? (y/n)")
        .default(false)
        .interact()
        .unwrap_or(false);
    if !confirm_ns {
        return;
    }
    let mut ns_counts: Vec<(String, usize)> = Vec::new();
    for a in &sel_assemblies {
        for ns in &a.namespaces {
            if let Some(e) = ns_counts.iter_mut().find(|(n, _)| n == &ns.namespace) {
                e.1 += ns.method_count;
            } else {
                ns_counts.push((ns.namespace.clone(), ns.method_count));
            }
        }
    }
    if ns_counts.is_empty() {
        return;
    }
    let ns_items: Vec<String> = ns_counts
        .iter()
        .map(|(n, c)| format!("{n}  ({c} methods)"))
        .collect();
    let defaults = vec![true; ns_items.len()];
    let chosen_ns = match MultiSelect::new()
        .with_prompt("  Select namespaces to decompile  (space = toggle, enter = confirm)")
        .items(&ns_items)
        .defaults(&defaults)
        .interact()
    {
        Ok(v) => v,
        Err(_) => return,
    };
    if chosen_ns.is_empty() || chosen_ns.len() == ns_counts.len() {
        return; // nothing narrowed -> keep the assembly-level filter only
    }
    config.dump_pseudo_c_include_types =
        chosen_ns.iter().map(|&i| ns_counts[i].0.clone()).collect();
    let ns_total: usize = chosen_ns.iter().map(|&i| ns_counts[i].1).sum();
    print_success(&format!(
        "Selected {} of {} namespaces ({} methods)",
        chosen_ns.len(),
        ns_counts.len(),
        ns_total
    ));

    // ---- Level 3: classes inside the selected namespaces -------------------
    drain_pending_input();
    let confirm_cls = Confirm::new()
        .with_prompt("  Filter by class inside the selected namespaces? (y/n)")
        .default(false)
        .interact()
        .unwrap_or(false);
    if !confirm_cls {
        return;
    }
    let selected_ns: std::collections::HashSet<&str> =
        chosen_ns.iter().map(|&i| ns_counts[i].0.as_str()).collect();
    let mut cls_counts: Vec<(String, usize)> = Vec::new();
    for a in &sel_assemblies {
        for ns in &a.namespaces {
            if !selected_ns.contains(ns.namespace.as_str()) {
                continue;
            }
            for t in &ns.types {
                if let Some(e) = cls_counts.iter_mut().find(|(n, _)| n == &t.full_name) {
                    e.1 += t.method_count;
                } else {
                    cls_counts.push((t.full_name.clone(), t.method_count));
                }
            }
        }
    }
    if cls_counts.is_empty() {
        return;
    }
    let cls_items: Vec<String> = cls_counts
        .iter()
        .map(|(n, c)| format!("{n}  ({c} methods)"))
        .collect();
    let defaults = vec![true; cls_items.len()];
    let chosen_cls = match MultiSelect::new()
        .with_prompt("  Select classes to decompile  (space = toggle, enter = confirm)")
        .items(&cls_items)
        .defaults(&defaults)
        .interact()
    {
        Ok(v) => v,
        Err(_) => return,
    };
    if chosen_cls.is_empty() || chosen_cls.len() == cls_counts.len() {
        return; // keep the namespace-level filter
    }
    config.dump_pseudo_c_include_types = chosen_cls
        .iter()
        .map(|&i| cls_counts[i].0.clone())
        .collect();
    let cls_total: usize = chosen_cls.iter().map(|&i| cls_counts[i].1).sum();
    print_success(&format!(
        "Selected {} of {} classes ({} methods)",
        chosen_cls.len(),
        cls_counts.len(),
        cls_total
    ));
}

fn run() -> Result<()> {
    let start_time = Instant::now();
    let cli = Cli::parse();
    // Hand the dump base to the platform initialisers through the environment:
    // they take the value at the point where the file is recognised as a dump,
    // and the same channel serves the non-CLI front ends.
    if let Some(base) = &cli.dump_base {
        CLI_DUMP_BASE.store(parse_hex_input(base), Ordering::Relaxed);
    }

    let (il2cpp_binary_path, metadata_path, output_dir_arg) =
        cli.resolve().map_err(il2cpp_dumper::error::Error::Other)?;

    print_banner();

    let mut config = if let Some(config_path) = &cli.config {
        Config::load_from_file(config_path).unwrap_or_else(|e| {
            print_warn(&format!(
                "Failed to load config from {}: {}",
                config_path, e
            ));
            Config::default()
        })
    } else if std::path::Path::new("config.json").exists() {
        Config::load_from_file("config.json").unwrap_or_else(|e| {
            print_warn(&format!("Failed to load config.json: {}", e));
            Config::default()
        })
    } else {
        Config::default()
    };
    // load_from_file already reconciles; defaults / failed loads still need it.
    config.reconcile_method_body_mode();

    if cli.codm {
        config.codm = true;
    }
    if cli.dump_static_metadata {
        config.dump_static_field_metadata = true;
    }
    if cli.no_dump_static_metadata {
        config.dump_static_field_metadata = false;
    }
    if cli.force_dump {
        config.force_dump = true;
    }
    if let Some(raw) = &cli.method_body_mode {
        match MethodBodyMode::parse(raw) {
            Some(mode) => config.set_method_body_mode(mode),
            None => print_warn(&format!(
                "Unknown --method-body-mode `{raw}` (use off|assembly|csharp|both); leaving config unchanged"
            )),
        }
    }
    if !cli.pseudo_include_assembly.is_empty() {
        config.dump_pseudo_c_include_assemblies = cli.pseudo_include_assembly.clone();
    }
    if !cli.pseudo_include_type.is_empty() {
        config.dump_pseudo_c_include_types = cli.pseudo_include_type.clone();
    }
    if let Some(size) = cli.pseudo_max_body_size {
        config.dump_pseudo_c_max_body_size = size;
    }
    if cli.generate_dummy_dll {
        config.generate_dummy_dll = true;
    }
    if cli.no_generate_dummy_dll {
        config.generate_dummy_dll = false;
    }
    if cli.il_recovery || cli.cs {
        config.il_recovery = true;
    }
    if cli.no_il_recovery {
        config.il_recovery = false;
    }
    if cli.no_annotate_recovery {
        config.annotate_recovery = false;
    }
    if cli.pseudo_select {
        config.dump_pseudo_c_interactive = true;
    }
    if config.il_recovery && !config.generate_dummy_dll {
        print_warn("--il-recovery needs DummyDll output; enabling generateDummyDll");
        config.generate_dummy_dll = true;
    }
    // Keep mode ↔ legacy bools aligned after any flag overrides.
    config.sync_method_body_mode_from_flags();

    let base_dir = std::path::Path::new(&output_dir_arg);
    let mut dump_num = 0u32;
    while base_dir.join(format!("Dump{dump_num}")).exists() {
        dump_num += 1;
    }
    let output_dir = base_dir
        .join(format!("Dump{dump_num}"))
        .to_string_lossy()
        .to_string();
    fs::create_dir_all(&output_dir).ok();
    println!(
        "  {}{} {}",
        FOLDER,
        style("Output").dim(),
        style(&output_dir).white().bold().underlined()
    );
    println!();

    println!(
        "  {}",
        style("─── Binary Analysis ───────────────────").dim()
    );
    println!();

    let sp = spinner("Loading IL2CPP binary...");
    let il2cpp_bytes = fs::read(&il2cpp_binary_path)?;
    let binary_size = il2cpp_bytes.len() as u64;
    sp.finish_and_clear();
    print_success(&format!(
        "Binary loaded: {} ({})",
        style(&il2cpp_binary_path).white().bold(),
        style(format_file_size(binary_size)).cyan()
    ));

    let unity_version_str = detect_unity_version(&il2cpp_bytes);
    if let Some(ref uv) = unity_version_str {
        print_info("Unity Version", uv);
    }

    let sp = spinner("Loading metadata...");
    let mut metadata_bytes = fs::read(&metadata_path)?;
    let metadata_size = metadata_bytes.len() as u64;
    sp.finish_and_clear();
    print_success(&format!(
        "Metadata loaded: {} ({})",
        style(&metadata_path).white().bold(),
        style(format_file_size(metadata_size)).cyan()
    ));

    let metadata_magic = read_magic_u32(&metadata_bytes);
    if metadata_magic != MAGIC_METADATA {
        match try_decrypt_metadata(&mut metadata_bytes) {
            Some(scheme) => {
                println!(
                    "  {} {}{}",
                    LOCK,
                    style("Decrypted metadata: ").dim(),
                    style(&scheme).yellow().bold()
                );
            }
            None => {
                let b0 = (metadata_magic & 0xFF) as u8;
                let b1 = ((metadata_magic >> 8) & 0xFF) as u8;
                let b2 = ((metadata_magic >> 16) & 0xFF) as u8;
                let b3 = ((metadata_magic >> 24) & 0xFF) as u8;
                return Err(il2cpp_dumper::error::Error::InvalidMetadata(format!(
                    "Wrong magic 0x{metadata_magic:08X} (bytes {b0:02X} {b1:02X} {b2:02X} {b3:02X}). \
                     Expected AF 1B B1 FA. File is likely encrypted, obfuscated, truncated, or not \
                     global-metadata.dat. Encryption scheme not recognized by built-in decryptors. \
                     Protected games often need a runtime memory dump of decrypted metadata."
                )));
            }
        }
    }

    let sp = spinner("Parsing metadata structures...");
    let mut metadata =
        Metadata::new_with_options(metadata_bytes, unity_version_str.as_deref(), config.codm)?;
    sp.finish_and_clear();
    print_info("Metadata Version", &metadata.version.to_string());
    print_info("Type Definitions", &format!("{}", metadata.type_defs.len()));
    print_info(
        "Method Definitions",
        &format!("{}", metadata.method_defs.len()),
    );

    println!();
    println!(
        "  {}",
        style("─── Format Detection ──────────────────").dim()
    );
    println!();

    let format = detect_format(&il2cpp_bytes);

    let mut il2cpp = match format {
        "elf" => init_elf(il2cpp_bytes, &metadata, &config)?,
        "pe" => init_pe(il2cpp_bytes, &metadata, &config)?,
        "macho" => init_macho(il2cpp_bytes, &metadata, &config)?,
        "macho_fat" => init_macho_fat(il2cpp_bytes, &metadata, &config)?,
        "nso" => init_nso(il2cpp_bytes, &metadata, &config)?,
        "wasm" => init_wasm(il2cpp_bytes, &metadata, &config)?,
        _ => {
            let magic = read_magic_u32(&il2cpp_bytes);
            return Err(il2cpp_dumper::error::Error::Other(format!(
                "Unsupported binary format (magic: 0x{magic:08X})"
            )));
        }
    };

    // The load base of a dump can be *derived* from the type handles the
    // metadata stores — but only when the caller did not supply one. An
    // explicit base is ground truth, and this derivation reads a type handle
    // whose region need not be the code's: a hybrid dump carries both the
    // on-disk layout and a runtime copy, so the derived value is a different
    // address space entirely. Substituting it silently shifts every printed RVA
    // (and the base is also what runtime pointers are resolved against), which
    // is worse than not deriving at all, because the addresses still look
    // plausible. Deriving is for the case where nothing is known.
    if il2cpp.version >= 27.0
        && il2cpp.is_dumped
        && il2cpp.image_base == 0
        && requested_runtime_base().is_none()
    {
        if let Some(type_def) = metadata.type_defs.first() {
            let byval_idx = type_def.byval_type_index as usize;
            if byval_idx < il2cpp.types.len() {
                let il2cpp_type = &il2cpp.types[byval_idx];
                let type_handle = il2cpp_type.type_handle();
                il2cpp.image_base =
                    type_handle.wrapping_sub(metadata.header.type_definitions_offset as u64);
            }
        }
    }

    println!();
    println!(
        "  {}",
        style("─── Output Generation ─────────────────").dim()
    );
    println!();

    // v108 keeps the method specs, generic method tables and RGCTX data in the
    // metadata file; everything downstream reads them off `Il2Cpp` as it always
    // has, so they are handed over once here. No-op before v108.
    metadata.apply_v108_tables(&mut il2cpp);

    let mut executor = Il2CppExecutor::new(&metadata, &mut il2cpp)?;

    // When method bodies are enabled (C# pseudo, assembly, and/or IL recovery)
    // and no assembly/type filter is already set (config.json or
    // --pseudo-include-* flags), offer an interactive checkbox selection:
    // assemblies -> namespaces -> classes. Only the selected subset gets
    // method bodies.
    //
    // NOTE: the dump spinner must NOT be running during these prompts. Its
    // steady-tick redraws (every 80 ms, `\r`-based) fight dialoguer's
    // raw-mode prompt rendering on real terminals: the prompt text gets
    // overwritten by the next spinner tick, so the user never sees
    // "Filter by namespace? (y/n)", presses Enter, nothing happens (no
    // default), and the whole thing looks frozen for minutes. Create the
    // spinner only after selection completes.
    // The picker is opt-in. Enabling bodies alone dumps every method (the
    // common want, and what a config.json user expects); the checkbox menu
    // only appears when the user explicitly asks for it — `--pseudo-select` or
    // `"csharpInteractive": true` — and never when a filter is already set
    // or `--no-pseudo-prompt` was given.
    let bodies_enabled = config.dump_pseudo_c || config.dump_disassembly || config.il_recovery;
    let want_interactive = bodies_enabled
        && (cli.pseudo_select || config.dump_pseudo_c_interactive)
        && !cli.no_pseudo_prompt
        && config.dump_pseudo_c_include_assemblies.is_empty()
        && config.dump_pseudo_c_include_types.is_empty();
    if want_interactive {
        prompt_pseudo_c_selection(&mut config, &metadata, &il2cpp, &mut executor);
    }
    let sp_dump = spinner("Generating dump.cs...");
    // Filters apply to C# pseudo, IL recovery, and native disassembly bodies.
    let bodies_filtered = config.dump_pseudo_c || config.dump_disassembly || config.il_recovery;
    if bodies_filtered && !config.dump_pseudo_c_include_assemblies.is_empty() {
        let list = config.dump_pseudo_c_include_assemblies.join(", ");
        let total = config.dump_pseudo_c_include_assemblies.len();
        if list.len() > 120 {
            let shown: Vec<&str> = config
                .dump_pseudo_c_include_assemblies
                .iter()
                .take(6)
                .map(|s| s.trim_end_matches(".dll").as_ref())
                .collect();
            print_info(
                "Method-body assemblies",
                &format!("{} total — {}", total, shown.join(", ")),
            );
        } else {
            print_info("Method-body assemblies", &list);
        }
    }
    if bodies_filtered && !config.dump_pseudo_c_include_types.is_empty() {
        let list = config.dump_pseudo_c_include_types.join(", ");
        let total = config.dump_pseudo_c_include_types.len();
        if list.len() > 120 {
            let shown: Vec<&str> = config
                .dump_pseudo_c_include_types
                .iter()
                .take(6)
                .map(String::as_str)
                .collect();
            print_info(
                "Method-body types",
                &format!("{} total — {}", total, shown.join(", ")),
            );
        } else {
            print_info("Method-body types", &list);
        }
    }

    Il2CppDecompiler::decompile(
        &mut executor,
        &mut metadata,
        &mut il2cpp,
        &config,
        &il2cpp_binary_path,
        &output_dir,
        |msg| {
            sp_dump.set_message(msg.to_string());
            // Some terminals (especially Windows / WSL wrappers) don't render
            // indicatif's in-place `\r` message redraws, so the spinner looks
            // frozen on "Generating dump.cs..." even though it is working. Print
            // every milestone as a permanent, scrolling line too — it cannot be
            // hidden, and it tells the user exactly which phase is running.
            sp_dump.println(format!("  {msg}"));
        },
    )?;
    sp_dump.finish_and_clear();
    print_success("dump.cs generated");

    let mut generated_files: Vec<String> = vec!["dump.cs".into()];

    let static_catalog = if config.dump_static_field_metadata {
        let sp = spinner("Analyzing thread-static / FieldRVA metadata...");
        let catalog = il2cpp_dumper::output::static_field_exporter::StaticFieldCatalog::collect(
            &mut executor,
            &mut metadata,
            &mut il2cpp,
            &config,
        )?;
        il2cpp_dumper::output::static_field_exporter::StaticFieldExporter::write_document(
            &catalog,
            &il2cpp,
            &output_dir,
        )?;
        sp.finish_and_clear();
        print_success(&format!(
            "static_metadata.json generated ({} thread-static, {} FieldRVA)",
            catalog.summary.thread_static_count, catalog.summary.field_rva_count,
        ));
        generated_files.push("static_metadata.json".into());
        Some(catalog)
    } else {
        None
    };

    if config.generate_struct {
        let sp = spinner("Generating structs...");
        StructGenerator::write_all(
            &mut executor,
            &mut metadata,
            &mut il2cpp,
            &config,
            &output_dir,
            static_catalog.as_ref(),
        )?;
        il2cpp_dumper::output::embedded_scripts::write_scripts(std::path::Path::new(&output_dir))?;
        sp.finish_and_clear();
        print_success("script.json, il2cpp.h, il2cpp-functions.h, stringliteral.json generated");
        generated_files.extend([
            "script.json".into(),
            "il2cpp.h".into(),
            "il2cpp-functions.h".into(),
            "stringliteral.json".into(),
        ]);
    }

    if config.generate_dummy_dll {
        let sp = spinner("Generating dummy DLLs...");
        il2cpp_dumper::output::dummy_assembly_generator::generate_dummy_dlls(
            &mut executor,
            &mut metadata,
            &mut il2cpp,
            &config,
            &output_dir,
        )?;
        sp.finish_and_clear();
        print_success("Dummy DLL files generated");
        generated_files.push("DummyDll/*.dll".into());
    }

    if config.generate_generics_dump {
        let sp = spinner("Generating generics dump...");
        let generics_path = std::path::Path::new(&output_dir).join("generics_dump.txt");
        if let Err(e) = il2cpp_dumper::output::generics::dump_generics(
            &generics_path.to_string_lossy(),
            &mut metadata,
            &mut il2cpp,
            &mut executor,
            &config,
        ) {
            sp.finish_and_clear();
            print_warn(&format!("Failed to generate generics_dump.txt: {}", e));
        } else {
            sp.finish_and_clear();
            print_success("generics_dump.txt generated");
            generated_files.push("generics_dump.txt".into());
        }
    }

    let elapsed = start_time.elapsed();

    println!();
    println!(
        "  {}",
        style("═══════════════════════════════════════").green()
    );
    println!(
        "  {} {}",
        SPARKLE,
        style("All tasks completed successfully!").green().bold()
    );
    println!(
        "  {}",
        style("═══════════════════════════════════════").green()
    );
    println!();
    println!(
        "  {}{} {}",
        FOLDER,
        style("Output Directory:").dim(),
        style(&output_dir).white().bold().underlined()
    );
    println!(
        "  {}{} {}",
        PACKAGE,
        style("Generated Files:").dim(),
        style(generated_files.join(", ")).cyan()
    );
    // Point the user straight at the decompilation input; the recovered
    // bodies live in DummyDll, and knowing that path is the whole workflow.
    if config.il_recovery {
        println!(
            "  {} {} {}",
            style("→").green().bold(),
            style("Decompile as C#:").dim(),
            style(format!("open {output_dir}/DummyDll in dnSpy / ILSpy"))
                .green()
                .bold()
        );
    }
    println!(
        "  {}{} {}",
        ROCKET,
        style("Elapsed:").dim(),
        style(format!("{:.2}s", elapsed.as_secs_f64()))
            .magenta()
            .bold()
    );
    println!(
        "  {}",
        style("───────────────────────────────────────").dim()
    );
    println!();

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!();
        eprintln!("  {} {}", style("✗ ERROR:").red().bold(), style(&e).red());
        eprintln!();
        process::exit(1);
    }
}
