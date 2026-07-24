use std::fs;
use std::process;
use std::time::Instant;

use clap::Parser;
use console::{style, Emoji};
use dialoguer::{Input, Select};
use indicatif::{ProgressBar, ProgressStyle};

use il2cpp_dumper::config::Config;
use il2cpp_dumper::error::Result;
use il2cpp_dumper::il2cpp::metadata::{Metadata, MetadataVariant};
use il2cpp_dumper::il2cpp::base::{Il2Cpp, VaSegment};
use il2cpp_dumper::executor::Il2CppExecutor;
use il2cpp_dumper::output::decompiler::Il2CppDecompiler;
use il2cpp_dumper::output::struct_generator::StructGenerator;
use il2cpp_dumper::formats::elf::Elf;
use il2cpp_dumper::formats::pe::Pe;
use il2cpp_dumper::formats::macho::MachO;
use il2cpp_dumper::formats::nso::Nso;
use il2cpp_dumper::formats::wasm::Wasm;

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
#[command(name = "il2cpp_dumper", version, about = "IL2CPP Dumper - Rust Port")]
struct Cli {
    #[arg(long = "binary", short = 'b')]
    il2cpp_binary_flag: Option<String>,
    #[arg(long = "metadata", short = 'm')]
    metadata_flag: Option<String>,
    #[arg(long = "output", short = 'o')]
    output_dir_flag: Option<String>,
    #[arg(long = "config", short = 'c')]
    config: Option<String>,
    #[arg(long = "codm")]
    codm: bool,
    /// Export thread-static / FieldRVA metadata (`static_metadata.json`, dump.cs annotations).
    #[arg(long = "dump-static-metadata")]
    dump_static_metadata: bool,
    /// Skip thread-static / FieldRVA static metadata export.
    #[arg(long = "no-dump-static-metadata")]
    no_dump_static_metadata: bool,
    #[arg(long = "force-dump")]
    force_dump: bool,
    #[arg(num_args = 0..=3)]
    positional: Vec<String>,
}

impl Cli {
    fn resolve(&self) -> std::result::Result<(String, String, String), String> {
        let binary = self.il2cpp_binary_flag.clone()
            .or_else(|| self.positional.get(0).cloned())
            .ok_or_else(|| "missing il2cpp binary path (positional or --binary)".to_string())?;
        let metadata = self.metadata_flag.clone()
            .or_else(|| self.positional.get(1).cloned())
            .ok_or_else(|| "missing metadata path (positional or --metadata)".to_string())?;
        let output = self.output_dir_flag.clone()
            .or_else(|| self.positional.get(2).cloned())
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
    println!(
        "  {}",
        style("─────────────────────────────────────").dim()
    );
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
    if data.len() < 4 { return 0; }
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

fn read_magic_u16(data: &[u8]) -> u16 {
    if data.len() < 2 { return 0; }
    u16::from_le_bytes([data[0], data[1]])
}

fn is_valid_metadata_version(data: &[u8]) -> bool {
    if data.len() < 8 { return false; }
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
        for b in test.iter_mut() { *b ^= k1; }
        if is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!("Single-byte XOR (key: 0x{k1:02X})"));
        }
    }

    let key4: [u8; 4] = std::array::from_fn(|i| magic[i] ^ data[i]);
    if key4 != [0u8; 4] && !key4.windows(2).all(|w| w[0] == w[1]) {
        let mut test = data.clone();
        for (i, b) in test.iter_mut().enumerate() { *b ^= key4[i % 4]; }
        if is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!("4-byte XOR (key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]));
        }
    }

    let key8: [u8; 8] = std::array::from_fn(|i| {
        if i < 4 { magic[i] ^ data[i] } else { data[i] }
    });
    if key8[0..4] != [0u8; 4] {
        let expected_ver_bytes: Vec<u8> = (4..8).map(|i| data[i] ^ key8[i]).collect();
        let test_ver = i32::from_le_bytes([expected_ver_bytes[0], expected_ver_bytes[1], expected_ver_bytes[2], expected_ver_bytes[3]]);
        if test_ver > 0 && test_ver < 200 {
            let mut test = data.clone();
            for (i, b) in test.iter_mut().enumerate() { *b ^= key8[i % 8]; }
            if is_valid_metadata_version(&test) {
                *data = test;
                return Some(format!("8-byte XOR (key: {:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X})",
                    key8[0], key8[1], key8[2], key8[3], key8[4], key8[5], key8[6], key8[7]));
            }
        }
    }

    for key_len in [16usize, 32, 64, 128, 256] {
        if data.len() < key_len * 2 { continue; }
        let key: Vec<u8> = (0..key_len).map(|i| if i < 4 { magic[i] ^ data[i] } else { data[i] ^ 0 }).collect();
        if key[0..4] == [0u8; 4] { continue; }
        let mut test = data[..8].to_vec();
        for (i, b) in test.iter_mut().enumerate() { *b ^= key[i % key_len]; }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            for (i, b) in data.iter_mut().enumerate() { *b ^= key[i % key_len]; }
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
            return Some(format!("Position-dependent XOR (base key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]));
        }
    }

    {
        let mut test = data.clone();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key4[i % 4] ^ ((i & 0xFF) as u8);
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!("Masked position XOR (base key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]));
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
            return Some(format!("Header-only XOR ({header_size} bytes, key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]));
        }
    }

    None
}

fn detect_unity_version(data: &[u8]) -> Option<String> {
    let mut best: Option<String> = None;
    let mut i = 0;
    while i + 12 < data.len() {
        if data[i] == b'2' && data[i + 1] == b'0'
            && data[i + 2].is_ascii_digit() && data[i + 3].is_ascii_digit()
            && data[i + 4] == b'.'
            && data[i + 5].is_ascii_digit()
        {
            let max_len = std::cmp::min(24, data.len() - i);
            let end = data[i..i + max_len].iter().position(|&b| {
                !b.is_ascii_alphanumeric() && b != b'.'
            }).unwrap_or(max_len);
            let candidate = &data[i..i + end];
            if candidate.len() >= 8 && candidate.len() <= 20 {
                if let Ok(s) = std::str::from_utf8(candidate) {
                    if s.chars().filter(|c| *c == '.').count() == 2
                        && (s.contains('f') || s.contains('b') || s.contains('a') || s.contains('p'))
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
    let trimmed = input.trim().trim_start_matches("0x").trim_start_matches("0X");
    if trimmed.is_empty() {
        return Err("Address cannot be empty".into());
    }
    u64::from_str_radix(trimmed, 16)
        .map(|_| ())
        .map_err(|_| format!("'{}' is not a valid hex address", input))
}

fn parse_hex_input(input: &str) -> u64 {
    let trimmed = input.trim().trim_start_matches("0x").trim_start_matches("0X");
    u64::from_str_radix(trimmed, 16).unwrap_or(0)
}

fn prompt_dump_address() -> Option<u64> {
    println!();
    println!(
        "  {} {}",
        WARN,
        style("This appears to be a memory dump file.").yellow().bold()
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
    if addr != 0 {
        Some(addr)
    } else {
        None
    }
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
            let trimmed = input.trim().trim_start_matches("0x").trim_start_matches("0X");
            if trimmed.is_empty() || trimmed == "0" {
                return Err("CodeRegistration address is required".to_string());
            }
            validate_hex(input)
        })
        .interact_text()
        .map_err(|_| il2cpp_dumper::error::Error::Other("Failed to read CodeRegistration input".into()))?;

    let mr_input: String = Input::new()
        .with_prompt(format!("  {} MetadataRegistration (hex)", style("→").cyan()))
        .validate_with(|input: &String| {
            let trimmed = input.trim().trim_start_matches("0x").trim_start_matches("0X");
            if trimmed.is_empty() || trimmed == "0" {
                return Err("MetadataRegistration address is required".to_string());
            }
            validate_hex(input)
        })
        .interact_text()
        .map_err(|_| il2cpp_dumper::error::Error::Other("Failed to read MetadataRegistration input".into()))?;

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

    if config.force_dump || elf.check_dump() {
        if let Some(addr) = prompt_dump_address() {
            elf.stream.image_base = addr;
            elf.is_dumped = true;
            if !config.no_redirected_pointer {
                elf.load()?;
            }
        }
    }

    let sp = spinner("Searching for registrations...");
    let method_count = metadata.method_defs.iter().filter(|m| m.method_index >= 0).count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();

    let mut helper = elf.get_section_helper(method_count, type_count, image_count);
    let code_reg = helper.find_code_registration();
    let metadata_reg = if metadata.variant == MetadataVariant::Codm {
        helper.find_metadata_registration_codm().or_else(|| helper.find_metadata_registration())
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

    if !found {
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
    il2cpp.codm = il2cpp.codm || resolve_codm(config, metadata);
    il2cpp.exported_symbols = elf_exports.iter().map(|(n, _)| n.clone()).collect();
    for (name, addr) in elf_exports {
        if name.starts_with("il2cpp_") || name.starts_with("mono_") {
            let rva = il2cpp.get_rva(addr);
            il2cpp.api_export_rvas.insert(name, rva);
        }
    }
    Ok(il2cpp)
}

fn init_pe(data: Vec<u8>, metadata: &Metadata, config: &Config) -> Result<Il2Cpp> {
    let mut pe = Pe::new(data)?;
    print_detection(&format!("PE{} format", if pe.is_32bit { "32" } else { "64" }));

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
    let method_count = metadata.method_defs.iter().filter(|m| m.method_index >= 0).count();
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
        let mut helper = pe.get_section_helper(method_count, type_count, mu_count, image_count, version);
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
    let va_segments: Vec<VaSegment> = pe.sections.iter().map(|s| {
        VaSegment {
            vaddr: s.virtual_address as u64 + pe_image_base,
            memsz: s.virtual_size as u64,
            offset: s.pointer_to_raw_data as u64,
        }
    }).collect();

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
    if let Ok(exports) = pe.list_exported_symbols() {
        il2cpp.exported_symbols = exports.iter().map(|(n, _)| n.clone()).collect();
        for (name, rva) in exports {
            if name.starts_with("il2cpp_") || name.starts_with("mono_") {
                il2cpp.api_export_rvas.insert(name, rva);
            }
        }
    }
    Ok(il2cpp)
}

fn init_macho_fat(data: Vec<u8>, metadata: &Metadata, config: &Config) -> Result<Il2Cpp> {
    use il2cpp_dumper::formats::macho::{parse_fat, extract_fat_slice, MH_MAGIC_64};

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
    print_detection(&format!("Mach-O {} format", if is_64 { "64-bit" } else { "32-bit" }));

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
    let method_count = metadata.method_defs.iter().filter(|m| m.method_index >= 0).count();
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
        let mut helper = macho.get_section_helper(method_count, type_count, mu_count, image_count, version);
        let code_reg = helper.find_code_registration();
        let metadata_reg = if metadata.variant == MetadataVariant::Codm {
            helper.find_metadata_registration_codm().or_else(|| helper.find_metadata_registration())
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

    let va_segments: Vec<VaSegment> = macho.segments.iter().map(|s| {
        VaSegment {
            vaddr: s.vmaddr,
            memsz: s.vmsize,
            offset: s.fileoff,
        }
    }).collect();

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

    if macho.is_32bit {
        for ptr in il2cpp.method_pointers.iter_mut() {
            if *ptr > 0 { *ptr -= 1; }
        }
        for ptr in il2cpp.custom_attribute_generators.iter_mut() {
            if *ptr > 0 { *ptr -= 1; }
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
    let method_count = metadata.method_defs.iter().filter(|m| m.method_index >= 0).count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();
    let mu_count = metadata.metadata_usages_count;

    let mut helper = nso.get_section_helper(method_count, type_count, mu_count, image_count, version);
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
    il2cpp.va_segments = vec![VaSegment { vaddr: 0, memsz: stream_len, offset: 0 }];
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
    let method_count = metadata.method_defs.iter().filter(|m| m.method_index >= 0).count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();
    let mu_count = metadata.metadata_usages_count;

    let mut helper = wasm.get_section_helper(method_count, type_count, mu_count, image_count, version);
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
    il2cpp.va_segments = vec![VaSegment { vaddr: 0, memsz: stream_len, offset: 0 }];
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

fn run() -> Result<()> {
    let start_time = Instant::now();
    let cli = Cli::parse();
    let (il2cpp_binary_path, metadata_path, output_dir_arg) = cli.resolve()
        .map_err(il2cpp_dumper::error::Error::Other)?;

    print_banner();

    let mut config = if let Some(config_path) = &cli.config {
        Config::load_from_file(config_path).unwrap_or_else(|e| {
            print_warn(&format!("Failed to load config from {}: {}", config_path, e));
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

    if cli.codm { config.codm = true; }
    if cli.dump_static_metadata { config.dump_static_field_metadata = true; }
    if cli.no_dump_static_metadata { config.dump_static_field_metadata = false; }
    if cli.force_dump { config.force_dump = true; }

    let base_dir = std::path::Path::new(&output_dir_arg);
    let mut dump_num = 0u32;
    while base_dir.join(format!("Dump{dump_num}")).exists() {
        dump_num += 1;
    }
    let output_dir = base_dir.join(format!("Dump{dump_num}")).to_string_lossy().to_string();
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
    let mut metadata = Metadata::new_with_options(
        metadata_bytes,
        unity_version_str.as_deref(),
        config.codm,
    )?;
    sp.finish_and_clear();
    print_info("Metadata Version", &metadata.version.to_string());
    print_info("Type Definitions", &format!("{}", metadata.type_defs.len()));
    print_info("Method Definitions", &format!("{}", metadata.method_defs.len()));

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
            return Err(il2cpp_dumper::error::Error::Other(
                format!("Unsupported binary format (magic: 0x{magic:08X})")
            ));
        }
    };

    if il2cpp.version >= 27.0 && il2cpp.is_dumped {
        if let Some(type_def) = metadata.type_defs.first() {
            let byval_idx = type_def.byval_type_index as usize;
            if byval_idx < il2cpp.types.len() {
                let il2cpp_type = &il2cpp.types[byval_idx];
                let type_handle = il2cpp_type.type_handle();
                il2cpp.image_base = type_handle.wrapping_sub(metadata.header.type_definitions_offset as u64);
            }
        }
    }

    println!();
    println!(
        "  {}",
        style("─── Output Generation ─────────────────").dim()
    );
    println!();

    let sp_dump = spinner("Generating dump.cs...");
    let mut executor = Il2CppExecutor::new(&metadata, &mut il2cpp)?;

    Il2CppDecompiler::decompile(&mut executor, &mut metadata, &mut il2cpp, &config, &output_dir, |msg| {
        sp_dump.set_message(msg.to_string());
    })?;
    sp_dump.finish_and_clear();
    print_success("dump.cs generated");

    let mut generated_files: Vec<String> = vec!["dump.cs".into()];

    let static_catalog = if config.dump_static_field_metadata {
        let sp = spinner("Analyzing thread-static / FieldRVA metadata...");
        let catalog = il2cpp_dumper::output::static_field_exporter::StaticFieldCatalog::collect(
            &mut executor, &mut metadata, &mut il2cpp, &config,
        )?;
        il2cpp_dumper::output::static_field_exporter::StaticFieldExporter::write_document(
            &catalog, &il2cpp, &output_dir,
        )?;
        sp.finish_and_clear();
        print_success(&format!(
            "static_metadata.json generated ({} thread-static, {} FieldRVA)",
            catalog.summary.thread_static_count,
            catalog.summary.field_rva_count,
        ));
        generated_files.push("static_metadata.json".into());
        Some(catalog)
    } else {
        None
    };

    if config.generate_struct {
        let sp = spinner("Generating structs...");
        StructGenerator::write_all(
            &mut executor, &mut metadata, &mut il2cpp, &config, &output_dir,
            static_catalog.as_ref(),
        )?;
        il2cpp_dumper::output::embedded_scripts::write_scripts(std::path::Path::new(&output_dir))?;
        sp.finish_and_clear();
        print_success("script.json, il2cpp.h, il2cpp-functions.h, stringliteral.json generated");
        generated_files.extend(["script.json".into(), "il2cpp.h".into(), "il2cpp-functions.h".into(), "stringliteral.json".into()]);
    }

    if config.generate_dummy_dll {
        let sp = spinner("Generating dummy DLLs...");
        il2cpp_dumper::output::dummy_assembly_generator::generate_dummy_dlls(
            &mut executor, &mut metadata, &mut il2cpp, &config, &output_dir,
        )?;
        sp.finish_and_clear();
        print_success("Dummy DLL files generated");
        generated_files.push("DummyDll/*.dll".into());
    }

    if config.generate_generics_dump {
        let sp = spinner("Generating generics dump...");
        let generics_path = std::path::Path::new(&output_dir).join("generics_dump.txt");
        if let Err(e) = il2cpp_dumper::output::generics::dump_generics(
            &generics_path.to_string_lossy(), &mut metadata, &mut il2cpp, &mut executor, &config
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
    println!(
        "  {}{} {}",
        ROCKET,
        style("Elapsed:").dim(),
        style(format!("{:.2}s", elapsed.as_secs_f64())).magenta().bold()
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
        eprintln!(
            "  {} {}",
            style("✗ ERROR:").red().bold(),
            style(&e).red()
        );
        eprintln!();
        process::exit(1);
    }
}