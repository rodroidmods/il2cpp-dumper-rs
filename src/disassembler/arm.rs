use super::{ConstantOp, DisassembledInstruction, LoadInfo, RegRegAccess};
use yaxpeax_arch::{Decoder, LengthedInstruction, U8Reader};
use yaxpeax_arm::armv7::{InstDecoder as A32Decoder, Instruction as A32Instruction};
use yaxpeax_arm::armv8::a64::{
    InstDecoder as A64Decoder, Instruction as A64Instruction, Opcode as A64Opcode,
    Operand as A64Operand, SizeCode as A64SizeCode,
};

pub fn disassemble_arm64(
    bytes: &[u8],
    base_address: u64,
    max_instructions: usize,
) -> Vec<DisassembledInstruction> {
    let mut result = Vec::with_capacity(max_instructions.min(512));
    let decoder = A64Decoder::default();
    let mut reader = U8Reader::new(bytes);
    let mut offset: u64 = 0;
    let mut seen_ret = false;

    for _ in 0..max_instructions {
        if offset as usize >= bytes.len() {
            break;
        }

        let insn: A64Instruction = match decoder.decode(&mut reader) {
            Ok(i) => i,
            Err(_) => break,
        };

        let address = base_address + offset;
        let size = insn.len().to_const() as usize;

        let raw_start = offset as usize;
        let raw_end = (raw_start + size).min(bytes.len());
        let raw_bytes = bytes[raw_start..raw_end].to_vec();

        let full_text = format!("{}", insn);
        let (mnemonic, operands) = split_mnemonic_operands(&full_text);

        let is_return = matches!(insn.opcode, A64Opcode::RET);

        let is_call = matches!(insn.opcode, A64Opcode::BL | A64Opcode::BLR);

        let is_unconditional_branch =
            matches!(insn.opcode, A64Opcode::B | A64Opcode::BR | A64Opcode::RET);

        let is_conditional_branch = matches!(
            insn.opcode,
            A64Opcode::Bcc(_) | A64Opcode::CBZ | A64Opcode::CBNZ | A64Opcode::TBZ | A64Opcode::TBNZ
        );

        let is_branch = is_unconditional_branch || is_call || is_conditional_branch;

        let call_target = if is_call {
            extract_arm64_branch_target(&insn, address)
        } else {
            None
        };

        let branch_target = if is_branch && !is_call && !is_return {
            extract_arm64_branch_target(&insn, address)
        } else {
            None
        };

        let condition_code = extract_arm64_condition(&insn, &operands);

        let memory_offset = extract_arm64_memory_offset(&insn);
        let reg_reg_access = extract_arm64_reg_reg_access(&insn);
        let constant_op = extract_arm64_constant_op(&insn, address);
        let load_info = extract_arm64_load_info(&insn, &mnemonic);
        let indirect_call_reg =
            if (is_call && call_target.is_none()) || matches!(insn.opcode, A64Opcode::BR) {
                extract_arm64_indirect_call_reg(&insn)
            } else {
                None
            };

        result.push(DisassembledInstruction {
            address,
            size,
            raw_bytes,
            mnemonic,
            operands,
            is_call,
            is_return,
            is_branch,
            is_unconditional_branch,
            call_target,
            branch_target,
            condition_code,
            memory_offset,
            reg_reg_access,
            constant_op,
            load_info,
            indirect_call_reg,
        });

        offset += size as u64;

        if is_return {
            if seen_ret {
                break;
            }
            seen_ret = true;
        }
    }

    result
}

pub fn disassemble_arm32(
    bytes: &[u8],
    base_address: u64,
    max_instructions: usize,
) -> Vec<DisassembledInstruction> {
    let mut result = Vec::with_capacity(max_instructions.min(512));
    let decoder = A32Decoder::default();
    let mut reader = U8Reader::new(bytes);
    let mut offset: u64 = 0;
    let mut seen_ret = false;

    for _ in 0..max_instructions {
        if offset as usize >= bytes.len() {
            break;
        }

        let insn: A32Instruction = match decoder.decode(&mut reader) {
            Ok(i) => i,
            Err(_) => break,
        };

        let address = base_address + offset;
        let size = insn.len().to_const() as usize;

        let raw_start = offset as usize;
        let raw_end = (raw_start + size).min(bytes.len());
        let raw_bytes = bytes[raw_start..raw_end].to_vec();

        let full_text = format!("{}", insn);
        let (mnemonic, operands) = split_mnemonic_operands(&full_text);

        let is_return = full_text.contains("pop") && full_text.contains("pc")
            || full_text.starts_with("bx lr")
            || full_text == "bx r14"
            || (full_text.starts_with("ldm") && full_text.contains("pc"));

        let is_call = full_text.starts_with("bl ")
            || full_text.starts_with("blx ")
            || full_text.starts_with("blal ")
            || full_text.starts_with("blxal ");

        let mnemonic_upper = mnemonic.to_uppercase();

        let is_unconditional_branch = mnemonic_upper == "B" || mnemonic_upper == "BX" || is_return;

        let is_conditional_branch = !is_call
            && !is_unconditional_branch
            && ((mnemonic_upper.starts_with('B')
                && !matches!(
                    mnemonic_upper.as_str(),
                    "BIC" | "BFC" | "BFI" | "BL" | "BLX" | "BKPT" | "BX"
                ))
                || matches!(mnemonic_upper.as_str(), "CBZ" | "CBNZ"));

        let is_branch = is_unconditional_branch || is_call || is_conditional_branch;

        let call_target = if is_call {
            extract_arm32_branch_target(&full_text, address)
        } else {
            None
        };

        let branch_target = if is_branch && !is_call && !is_return {
            extract_arm32_branch_target(&full_text, address)
        } else {
            None
        };

        let condition_code = extract_arm32_condition(&mnemonic_upper);

        let memory_offset = extract_arm32_memory_offset(&full_text);
        let reg_reg_access = extract_arm32_reg_reg_access(&full_text, &mnemonic_upper);
        let constant_op = extract_arm32_constant_op(&full_text, &mnemonic_upper, address);
        let load_info = extract_arm32_load_info(&full_text, &mnemonic_upper);
        let indirect_call_reg = extract_arm32_indirect_call_reg(&full_text, &mnemonic_upper);

        result.push(DisassembledInstruction {
            address,
            size,
            raw_bytes,
            mnemonic,
            operands,
            is_call,
            is_return,
            is_branch,
            is_unconditional_branch,
            call_target,
            branch_target,
            condition_code,
            memory_offset,
            reg_reg_access,
            constant_op,
            load_info,
            indirect_call_reg,
        });

        offset += size as u64;

        if is_return {
            if seen_ret {
                break;
            }
            seen_ret = true;
        }
    }

    result
}

fn split_mnemonic_operands(text: &str) -> (String, String) {
    let trimmed = text.trim();
    if let Some(pos) = trimmed.find(' ') {
        let mnemonic = trimmed[..pos].to_uppercase();
        let operands = trimmed[pos..].trim().to_string();
        (mnemonic, operands)
    } else {
        (trimmed.to_uppercase(), String::new())
    }
}

fn extract_arm32_reg_num(s: &str) -> Option<u16> {
    let s = s.trim().trim_start_matches(|c: char| !c.is_alphanumeric());
    let s = s.trim_start_matches(|c: char| c == 'r' || c == 'R');
    if s == "sp" || s == "SP" || s == "r13" || s == "R13" {
        return Some(13);
    }
    if s == "lr" || s == "LR" || s == "r14" || s == "R14" {
        return Some(14);
    }
    if s == "pc" || s == "PC" || s == "r15" || s == "R15" {
        return Some(15);
    }
    s.split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse::<u16>()
        .ok()
}

fn parse_arm32_operand_reg(token: &str) -> Option<u16> {
    let t = token.trim().trim_end_matches('!').trim_end_matches(',');
    let t = t.trim_start_matches('[').trim_end_matches(']');
    extract_arm32_reg_num(t)
}

fn parse_arm32_immediate(token: &str) -> Option<i64> {
    let t = token.trim().trim_start_matches('#');
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()
    } else if let Some(neg) = t.strip_prefix("-0x").or_else(|| t.strip_prefix("-0X")) {
        i64::from_str_radix(neg, 16).ok().map(|v| -v)
    } else {
        t.parse::<i64>().ok()
    }
}

fn extract_arm32_constant_op(
    text: &str,
    mnemonic_upper: &str,
    _address: u64,
) -> Option<ConstantOp> {
    let bare = strip_arm32_condition(mnemonic_upper);

    match bare {
        "MOV" | "MOVS" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            let src = parts[1].trim();
            if src.starts_with('#') || src.starts_with("0x") {
                let val = parse_arm32_immediate(src)? as u64;
                return Some(ConstantOp::MovImm {
                    dest_reg: dest,
                    value: val,
                });
            }
            if let Some(src_reg) = parse_arm32_operand_reg(src.split(',').next().unwrap_or(src)) {
                if !src.contains("lsl")
                    && !src.contains("LSL")
                    && !src.contains("lsr")
                    && !src.contains("LSR")
                    && !src.contains("asr")
                    && !src.contains("ASR")
                    && !src.contains("ror")
                    && !src.contains("ROR")
                {
                    return Some(ConstantOp::MovReg {
                        dest_reg: dest,
                        src_reg,
                    });
                }
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }

        "MOVW" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            let val = parse_arm32_immediate(parts[1].trim())? as u64;
            Some(ConstantOp::MovImm {
                dest_reg: dest,
                value: val & 0xFFFF,
            })
        }

        "MOVT" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            let hi = (parse_arm32_immediate(parts[1].trim())? as u64) & 0xFFFF;
            Some(ConstantOp::MovKeep {
                dest_reg: dest,
                imm16: hi as u16,
                shift: 16,
            })
        }

        "MVN" | "MVNS" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            let src = parts[1].trim();
            if src.starts_with('#') {
                let val = parse_arm32_immediate(src)? as u64;
                return Some(ConstantOp::MovImm {
                    dest_reg: dest,
                    value: !val & 0xFFFF_FFFF,
                });
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }

        "ADD" | "ADDS" | "ADC" | "ADCS" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(3, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            if parts.len() == 2 {
                let src2 = parts[1].trim();
                if src2.starts_with('#') || src2.starts_with("0x") {
                    let imm = parse_arm32_immediate(src2)?;
                    return Some(ConstantOp::AddSubImm {
                        dest_reg: dest,
                        src_reg: dest,
                        imm,
                    });
                }
                return Some(ConstantOp::Kill { dest_reg: dest });
            }
            let src1 = parse_arm32_operand_reg(parts[1])?;
            let src2 = parts[2].trim();
            if src2.starts_with('#') || src2.starts_with("0x") {
                let imm = parse_arm32_immediate(src2)?;
                return Some(ConstantOp::AddSubImm {
                    dest_reg: dest,
                    src_reg: src1,
                    imm,
                });
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }

        "SUB" | "SUBS" | "SBC" | "SBCS" | "RSB" | "RSBS" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(3, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            if parts.len() == 2 {
                let src2 = parts[1].trim();
                if src2.starts_with('#') || src2.starts_with("0x") {
                    let imm = -(parse_arm32_immediate(src2)?);
                    return Some(ConstantOp::AddSubImm {
                        dest_reg: dest,
                        src_reg: dest,
                        imm,
                    });
                }
                return Some(ConstantOp::Kill { dest_reg: dest });
            }
            let src1 = parse_arm32_operand_reg(parts[1])?;
            let src2 = parts[2].trim();
            if src2.starts_with('#') || src2.starts_with("0x") {
                let imm = -(parse_arm32_immediate(src2)?);
                return Some(ConstantOp::AddSubImm {
                    dest_reg: dest,
                    src_reg: src1,
                    imm,
                });
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }

        "ORR" | "ORRS" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(3, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            if parts.len() >= 3 {
                let src1 = parse_arm32_operand_reg(parts[1]);
                let src2 = parts[2].trim();
                if let Some(s1) = src1 {
                    if let Some(s2) = parse_arm32_operand_reg(src2) {
                        if s1 == dest && s2 == dest {
                            return Some(ConstantOp::Kill { dest_reg: dest });
                        }
                    }
                }
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }

        "EOR" | "EORS" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(3, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            if parts.len() >= 3 {
                let src1 = parse_arm32_operand_reg(parts[1]);
                let src2 = parts[2].trim();
                if let (Some(s1), Some(s2)) = (src1, parse_arm32_operand_reg(src2)) {
                    if s1 == dest && s2 == dest {
                        return Some(ConstantOp::MovImm {
                            dest_reg: dest,
                            value: 0,
                        });
                    }
                }
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }

        "AND" | "ANDS" | "BIC" | "BICS" | "LSL" | "LSR" | "ASR" | "ROR" | "MUL" | "MLA" | "MLS"
        | "SMULL" | "UMULL" | "SMLAL" | "UMLAL" | "SDIV" | "UDIV" | "CLZ" | "RBIT" | "REV"
        | "REV16" | "REVSH" | "UXTB" | "UXTH" | "SXTB" | "SXTH" | "UBFX" | "SBFX" | "BFI"
        | "BFC" | "USAT" | "SSAT" | "PKHBT" | "PKHTB" | "LDR" | "LDRB" | "LDRH" | "LDRSB"
        | "LDRSH" | "LDRD" | "LDREX" | "LDREXB" | "LDREXH" | "LDREXD" | "LDA" | "LDAB" | "LDAH"
        | "LDAEX" | "LDAEXB" | "LDAEXH" | "LDAEXD" | "POP" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.is_empty() {
                return None;
            }
            let first = parts[0]
                .trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim_end_matches('!');
            if bare == "POP" {
                let reg_list = parse_arm32_reg_list(first);
                if reg_list.len() == 1 {
                    return Some(ConstantOp::Kill {
                        dest_reg: reg_list[0],
                    });
                } else if reg_list.len() == 2 {
                    return Some(ConstantOp::KillPair {
                        dest_reg1: reg_list[0],
                        dest_reg2: reg_list[1],
                    });
                }
                return None;
            }
            let dest = parse_arm32_operand_reg(first)?;
            Some(ConstantOp::Kill { dest_reg: dest })
        }

        "LDM" | "LDMIA" | "LDMIB" | "LDMDA" | "LDMDB" | "LDMFD" | "LDMEA" | "LDMED" | "LDMFA"
        | "LDMCA" => {
            if let (Some(bs), Some(be)) = (text.find('{'), text.find('}')) {
                let regs = parse_arm32_reg_list(&text[bs..=be]);
                let loadable: Vec<u16> = regs.into_iter().filter(|&r| r != 13 && r != 15).collect();
                return match loadable.len() {
                    0 => None,
                    1 => Some(ConstantOp::Kill {
                        dest_reg: loadable[0],
                    }),
                    _ => Some(ConstantOp::KillPair {
                        dest_reg1: loadable[0],
                        dest_reg2: loadable[1],
                    }),
                };
            }
            None
        }

        "ADR" | "ADRL" => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let parts: Vec<&str> = rest.splitn(2, ',').collect();
            if parts.len() < 2 {
                return None;
            }
            let dest = parse_arm32_operand_reg(parts[0])?;
            let target_str = parts[1].trim();
            let page: u64 = if let Some(rel) = target_str.strip_prefix("$+") {
                let hex = rel.trim().trim_start_matches("0x").trim_start_matches("0X");
                let offset = i64::from_str_radix(hex, 16)
                    .unwrap_or_else(|_| rel.trim().parse::<i64>().unwrap_or(0));
                (_address as i64).wrapping_add(offset) as u64
            } else if let Some(rel) = target_str.strip_prefix("$-") {
                let hex = rel.trim().trim_start_matches("0x").trim_start_matches("0X");
                let offset = i64::from_str_radix(hex, 16)
                    .unwrap_or_else(|_| rel.trim().parse::<i64>().unwrap_or(0));
                (_address as i64).wrapping_sub(offset) as u64
            } else {
                parse_arm32_immediate(target_str).unwrap_or(0) as u64
            };
            Some(ConstantOp::Adrp {
                dest_reg: dest,
                page,
            })
        }

        "STR" | "STRB" | "STRH" | "STRD" | "STREX" | "STREXB" | "STREXH" | "STREXD" | "STL"
        | "STLB" | "STLH" | "STLEX" | "STLEXB" | "STLEXH" | "STLEXD" | "PUSH" | "CMP" | "CMN"
        | "TST" | "TEQ" | "B" | "BL" | "BX" | "BLX" | "CBZ" | "CBNZ" | "SVC" | "BKPT" | "HLT"
        | "UDF" | "NOP" | "WFE" | "WFI" | "SEV" | "DMB" | "DSB" | "ISB" | "PLD" | "PLI"
        | "VMRS" | "VMSR" => None,

        _ => {
            let ops: Vec<&str> = text.splitn(3, ' ').collect();
            if ops.len() < 2 {
                return None;
            }
            let rest = ops[1..].join(" ");
            let first = rest.splitn(2, ',').next()?.trim().trim_end_matches('!');
            if let Some(dest) = parse_arm32_operand_reg(first) {
                Some(ConstantOp::Kill { dest_reg: dest })
            } else {
                None
            }
        }
    }
}

fn strip_arm32_condition(mnemonic: &str) -> &str {
    let conditions = [
        "EQ", "NE", "CS", "HS", "CC", "LO", "MI", "PL", "VS", "VC", "HI", "LS", "GE", "LT", "GT",
        "LE", "AL",
    ];
    for cond in &conditions {
        if mnemonic.ends_with(cond) && mnemonic.len() > cond.len() {
            let base = &mnemonic[..mnemonic.len() - cond.len()];
            if !base.is_empty() {
                return base;
            }
        }
    }
    mnemonic
}

fn parse_arm32_reg_list(text: &str) -> Vec<u16> {
    let inner = text.trim().trim_start_matches('{').trim_end_matches('}');
    let mut regs = Vec::new();
    for token in inner.split(',') {
        let t = token.trim();
        if t.contains('-') {
            let range: Vec<&str> = t.splitn(2, '-').collect();
            if range.len() == 2 {
                if let (Some(start), Some(end)) = (
                    extract_arm32_reg_num(range[0]),
                    extract_arm32_reg_num(range[1]),
                ) {
                    for r in start..=end {
                        regs.push(r);
                    }
                }
            }
        } else if let Some(r) = extract_arm32_reg_num(t) {
            regs.push(r);
        }
    }
    regs
}

fn extract_arm32_load_info(text: &str, mnemonic_upper: &str) -> Option<LoadInfo> {
    let bare = strip_arm32_condition(mnemonic_upper);
    let is_load = matches!(
        bare,
        "LDR"
            | "LDRB"
            | "LDRH"
            | "LDRSB"
            | "LDRSH"
            | "LDREX"
            | "LDREXB"
            | "LDREXH"
            | "LDA"
            | "LDAB"
            | "LDAH"
    );
    if !is_load {
        return None;
    }

    let bracket_start = text.find('[')?;
    let bracket_end = text.find(']')?;
    let inner = &text[bracket_start + 1..bracket_end];

    let parts: Vec<&str> = inner.splitn(2, ',').collect();
    let base_reg = extract_arm32_reg_num(parts[0].trim())?;

    let offset = if parts.len() >= 2 {
        let src = parts[1].trim();
        if src.starts_with('#') || src.starts_with("0x") || src.starts_with('-') {
            parse_arm32_immediate(src).unwrap_or(0)
        } else {
            0
        }
    } else {
        0
    };

    let ops_text = &text[text.find(' ').unwrap_or(0)..bracket_start];
    let dest_token = ops_text.split(',').next().unwrap_or("").trim();
    let dest_reg = extract_arm32_reg_num(dest_token)?;

    Some(LoadInfo {
        dest_reg,
        base_reg,
        offset,
    })
}

fn extract_arm32_indirect_call_reg(text: &str, mnemonic_upper: &str) -> Option<u16> {
    let bare = strip_arm32_condition(mnemonic_upper);
    if bare != "BX" && bare != "BLX" {
        return None;
    }
    let after = text.splitn(2, ' ').nth(1)?.trim();
    if after.starts_with('r') || after.starts_with('R') {
        extract_arm32_reg_num(after)
    } else {
        None
    }
}

fn extract_arm32_reg_reg_access(text: &str, mnemonic_upper: &str) -> Option<RegRegAccess> {
    let bare = strip_arm32_condition(mnemonic_upper);
    let is_mem = matches!(
        bare,
        "LDR" | "LDRB" | "LDRH" | "LDRSB" | "LDRSH" | "STR" | "STRB" | "STRH"
    );
    if !is_mem {
        return None;
    }

    let bracket_start = text.find('[')?;
    let bracket_end = text.find(']')?;
    let inner = &text[bracket_start + 1..bracket_end];

    let parts: Vec<&str> = inner.splitn(3, ',').collect();
    if parts.len() < 2 {
        return None;
    }

    let base_reg = extract_arm32_reg_num(parts[0].trim())?;
    let index_part = parts[1].trim();

    if index_part.starts_with('#') {
        return None;
    }

    let index_reg = extract_arm32_reg_num(index_part.split_whitespace().next()?)?;

    let shift: u8 = if parts.len() >= 3 {
        let shift_part = parts[2].trim().to_uppercase();
        if shift_part.starts_with("LSL #") || shift_part.starts_with("LSL#") {
            let n = shift_part
                .trim_start_matches("LSL")
                .trim()
                .trim_start_matches('#');
            n.parse::<u8>().unwrap_or(0)
        } else {
            0
        }
    } else {
        0
    };

    Some(RegRegAccess {
        base_reg,
        index_reg,
        shift,
    })
}

fn extract_arm32_branch_target(text: &str, current_address: u64) -> Option<u64> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() >= 2 {
        let target_str = parts.last()?;
        let target_str = target_str.trim_start_matches("$+");
        let target_str = target_str.trim_start_matches("0x").trim_start_matches("0X");

        if let Ok(offset) = i64::from_str_radix(target_str, 16) {
            return Some((current_address as i64 + offset) as u64);
        }

        if let Ok(offset) = target_str.parse::<i64>() {
            return Some((current_address as i64 + offset) as u64);
        }
    }
    None
}

fn extract_arm32_condition(mnemonic: &str) -> Option<String> {
    if mnemonic == "CBZ" {
        return Some("== 0".to_string());
    }
    if mnemonic == "CBNZ" {
        return Some("!= 0".to_string());
    }
    if mnemonic.len() < 2 {
        return None;
    }
    if !mnemonic.starts_with('B') {
        return None;
    }
    if matches!(
        mnemonic,
        "B" | "BL" | "BX" | "BLX" | "BIC" | "BFC" | "BFI" | "BKPT"
    ) {
        return None;
    }

    let suffix = &mnemonic[1..];
    let cond = match suffix {
        "EQ" => "==",
        "NE" => "!=",
        "CS" | "HS" => ">= (unsigned)",
        "CC" | "LO" => "< (unsigned)",
        "MI" => "< 0",
        "PL" => ">= 0",
        "VS" => "overflow",
        "VC" => "!overflow",
        "HI" => "> (unsigned)",
        "LS" => "<= (unsigned)",
        "GE" => ">=",
        "LT" => "<",
        "GT" => ">",
        "LE" => "<=",
        _ => return None,
    };
    Some(cond.to_string())
}

fn extract_arm32_memory_offset(text: &str) -> Option<i64> {
    if let Some(bracket_start) = text.find('[') {
        if let Some(hash_pos) = text[bracket_start..].find('#') {
            let after_hash = &text[bracket_start + hash_pos + 1..];
            let end = after_hash
                .find(']')
                .or_else(|| after_hash.find(','))
                .unwrap_or(after_hash.len());
            let num_str = after_hash[..end].trim();

            if let Some(stripped) = num_str.strip_prefix("0x") {
                if let Ok(val) = i64::from_str_radix(stripped, 16) {
                    return Some(val);
                }
            } else if let Some(stripped) = num_str.strip_prefix("-0x") {
                if let Ok(val) = i64::from_str_radix(stripped, 16) {
                    return Some(-val);
                }
            } else if let Ok(val) = num_str.parse::<i64>() {
                return Some(val);
            }
        }
    }
    None
}

fn extract_arm64_branch_target(insn: &A64Instruction, current_address: u64) -> Option<u64> {
    for i in 0..4 {
        match insn.operands[i] {
            A64Operand::PCOffset(offset) => {
                return Some((current_address as i64 + offset as i64) as u64);
            }
            A64Operand::Imm64(val) => {
                return Some(val);
            }
            A64Operand::Imm16(val) => {
                return Some(val as u64);
            }
            A64Operand::Nothing => break,
            _ => {}
        }
    }
    None
}

fn extract_arm64_condition(insn: &A64Instruction, operands: &str) -> Option<String> {
    match insn.opcode {
        A64Opcode::Bcc(cond) => {
            let cond_str = match cond {
                0 => "==",            // EQ
                1 => "!=",            // NE
                2 => ">= (unsigned)", // CS/HS
                3 => "< (unsigned)",  // CC/LO
                4 => "< 0",           // MI (negative)
                5 => ">= 0",          // PL (positive/zero)
                6 => "overflow",      // VS
                7 => "!overflow",     // VC
                8 => "> (unsigned)",  // HI
                9 => "<= (unsigned)", // LS
                10 => ">=",           // GE
                11 => "<",            // LT
                12 => ">",            // GT
                13 => "<=",           // LE
                14 => "always",       // AL
                _ => "?",
            };
            Some(cond_str.to_string())
        }
        A64Opcode::CBZ => {
            let reg = operands.split(',').next().unwrap_or("?").trim();
            Some(format!("{} == null", reg))
        }
        A64Opcode::CBNZ => {
            let reg = operands.split(',').next().unwrap_or("?").trim();
            Some(format!("{} != null", reg))
        }
        A64Opcode::TBZ => {
            let parts: Vec<&str> = operands.splitn(3, ',').collect();
            if parts.len() >= 2 {
                let reg = parts[0].trim();
                let bit = parts[1].trim().trim_start_matches('#');
                Some(format!("bit{} of {} == 0", bit, reg))
            } else {
                Some("bit == 0".to_string())
            }
        }
        A64Opcode::TBNZ => {
            let parts: Vec<&str> = operands.splitn(3, ',').collect();
            if parts.len() >= 2 {
                let reg = parts[0].trim();
                let bit = parts[1].trim().trim_start_matches('#');
                Some(format!("bit{} of {} != 0", bit, reg))
            } else {
                Some("bit != 0".to_string())
            }
        }
        _ => None,
    }
}

fn extract_arm64_memory_offset(insn: &A64Instruction) -> Option<i64> {
    for i in 0..4 {
        match insn.operands[i] {
            A64Operand::RegRegOffset(..) => {}
            A64Operand::RegPreIndex(_, offset, _) => {
                return Some(offset as i64);
            }
            A64Operand::RegPostIndex(_, offset) => {
                return Some(offset as i64);
            }
            A64Operand::RegPostIndexReg(_, _) => {}
            A64Operand::Nothing => break,
            _ => {}
        }
    }

    let text = format!("{}", insn);
    if let Some(bracket_start) = text.find('[') {
        if let Some(hash_pos) = text[bracket_start..].find('#') {
            let after_hash = &text[bracket_start + hash_pos + 1..];
            let end = after_hash
                .find(']')
                .or_else(|| after_hash.find(','))
                .unwrap_or(after_hash.len());
            let num_str = after_hash[..end].trim().trim_end_matches('!');

            if let Some(stripped) = num_str
                .strip_prefix("0x")
                .or_else(|| num_str.strip_prefix("0X"))
            {
                if let Ok(val) = i64::from_str_radix(stripped, 16) {
                    return Some(val);
                }
            } else if let Some(stripped) = num_str
                .strip_prefix("-0x")
                .or_else(|| num_str.strip_prefix("-0X"))
            {
                if let Ok(val) = i64::from_str_radix(stripped, 16) {
                    return Some(-val);
                }
            } else if let Ok(val) = num_str.parse::<i64>() {
                return Some(val);
            }
        }
    }

    None
}

fn extract_arm64_load_info(insn: &A64Instruction, mnemonic: &str) -> Option<LoadInfo> {
    let is_load = matches!(
        mnemonic,
        "LDR"
            | "LDRB"
            | "LDRH"
            | "LDRSB"
            | "LDRSH"
            | "LDRSW"
            | "LDUR"
            | "LDURB"
            | "LDURH"
            | "LDURSB"
            | "LDURSH"
            | "LDURSW"
            | "LDAR"
            | "LDARB"
            | "LDARH"
            | "LDAXR"
            | "LDXR"
    );
    if !is_load {
        return None;
    }

    let dest_reg = match insn.operands[0] {
        A64Operand::Register(_, r) => r,
        _ => return None,
    };

    for i in 1..4 {
        match insn.operands[i] {
            A64Operand::RegPreIndex(base, offset, _) => {
                return Some(LoadInfo {
                    dest_reg,
                    base_reg: base,
                    offset: offset as i64,
                });
            }
            A64Operand::RegPostIndex(base, offset) => {
                return Some(LoadInfo {
                    dest_reg,
                    base_reg: base,
                    offset: offset as i64,
                });
            }
            A64Operand::Nothing => break,
            _ => {}
        }
    }
    None
}

fn extract_arm64_indirect_call_reg(insn: &A64Instruction) -> Option<u16> {
    if !matches!(insn.opcode, A64Opcode::BLR | A64Opcode::BR) {
        return None;
    }
    match insn.operands[0] {
        A64Operand::Register(_, r) | A64Operand::RegisterOrSP(_, r) => Some(r),
        _ => None,
    }
}

fn extract_arm64_reg_reg_access(insn: &A64Instruction) -> Option<RegRegAccess> {
    for i in 0..4 {
        if let A64Operand::RegRegOffset(base, index, _size, _style, shift) = insn.operands[i] {
            return Some(RegRegAccess {
                base_reg: base,
                index_reg: index,
                shift,
            });
        }
    }
    None
}

fn extract_arm64_constant_op(insn: &A64Instruction, address: u64) -> Option<ConstantOp> {
    match insn.opcode {
        A64Opcode::MOVZ => {
            let (reg, size) = match insn.operands[0] {
                A64Operand::Register(s, r) if r != 31 => (r, s),
                _ => return None,
            };
            if let A64Operand::ImmShift(imm, shift) = insn.operands[1] {
                let mut value = (imm as u64) << shift;
                if size == A64SizeCode::W {
                    value = value as u32 as u64;
                }
                return Some(ConstantOp::MovImm {
                    dest_reg: reg,
                    value,
                });
            }
            None
        }
        A64Opcode::MOVK => {
            let reg = match insn.operands[0] {
                A64Operand::Register(_, r) if r != 31 => r,
                _ => return None,
            };
            if let A64Operand::ImmShift(imm, shift) = insn.operands[1] {
                return Some(ConstantOp::MovKeep {
                    dest_reg: reg,
                    imm16: imm,
                    shift,
                });
            }
            None
        }
        A64Opcode::MOVN => {
            let (reg, size) = match insn.operands[0] {
                A64Operand::Register(s, r) if r != 31 => (r, s),
                _ => return None,
            };
            if let A64Operand::ImmShift(imm, shift) = insn.operands[1] {
                let mut value = !((imm as u64) << shift);
                if size == A64SizeCode::W {
                    value = value as u32 as u64;
                }
                return Some(ConstantOp::MovImm {
                    dest_reg: reg,
                    value,
                });
            }
            None
        }
        A64Opcode::ADD | A64Opcode::SUB => {
            let dest = match insn.operands[0] {
                A64Operand::Register(_, r) | A64Operand::RegisterOrSP(_, r) if r != 31 => r,
                _ => return None,
            };
            let src = match insn.operands[1] {
                A64Operand::Register(_, r) | A64Operand::RegisterOrSP(_, r) => r,
                _ => return None,
            };
            let imm_val = match insn.operands[2] {
                A64Operand::Immediate(imm) => imm as i64,
                A64Operand::ImmShift(imm, shift) => (imm as i64) << shift,
                _ => return Some(ConstantOp::Kill { dest_reg: dest }),
            };
            let imm = if matches!(insn.opcode, A64Opcode::SUB) {
                -imm_val
            } else {
                imm_val
            };
            Some(ConstantOp::AddSubImm {
                dest_reg: dest,
                src_reg: src,
                imm,
            })
        }
        A64Opcode::ADDS | A64Opcode::SUBS => {
            let dest = match insn.operands[0] {
                A64Operand::Register(_, r) => r,
                _ => return None,
            };
            if dest == 31 {
                return None;
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }
        A64Opcode::EOR => {
            let dest = match insn.operands[0] {
                A64Operand::Register(_, r) if r != 31 => r,
                _ => return None,
            };
            let is_self_xor = matches!(insn.operands[1], A64Operand::Register(_, r) if r == dest)
                && matches!(insn.operands[2], A64Operand::Register(_, r) if r == dest);
            if is_self_xor {
                return Some(ConstantOp::MovImm {
                    dest_reg: dest,
                    value: 0,
                });
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }
        A64Opcode::ORR => {
            let dest = match insn.operands[0] {
                A64Operand::Register(_, r) if r != 31 => r,
                _ => return None,
            };
            let is_zr_src = matches!(insn.operands[1], A64Operand::Register(_, 31));
            if is_zr_src {
                match insn.operands[2] {
                    A64Operand::Register(_, src) => {
                        return Some(ConstantOp::MovReg {
                            dest_reg: dest,
                            src_reg: src,
                        });
                    }
                    A64Operand::Imm64(imm) => {
                        return Some(ConstantOp::MovImm {
                            dest_reg: dest,
                            value: imm,
                        });
                    }
                    _ => {}
                }
            }
            Some(ConstantOp::Kill { dest_reg: dest })
        }
        A64Opcode::ADRP => {
            let reg = match insn.operands[0] {
                A64Operand::Register(_, r) if r != 31 => r,
                _ => return None,
            };
            if let A64Operand::PCOffset(offset) = insn.operands[1] {
                let page = (address as i64 + offset) as u64;
                return Some(ConstantOp::Adrp {
                    dest_reg: reg,
                    page,
                });
            }
            None
        }
        A64Opcode::LDP | A64Opcode::LDNP | A64Opcode::LDXP => {
            let r1 = match insn.operands[0] {
                A64Operand::Register(_, r) if r != 31 => r,
                _ => return None,
            };
            let r2 = match insn.operands[1] {
                A64Operand::Register(_, r) if r != 31 => Some(r),
                _ => None,
            };
            if let Some(r2) = r2 {
                Some(ConstantOp::KillPair {
                    dest_reg1: r1,
                    dest_reg2: r2,
                })
            } else {
                Some(ConstantOp::Kill { dest_reg: r1 })
            }
        }
        A64Opcode::STR
        | A64Opcode::STRB
        | A64Opcode::STRH
        | A64Opcode::STUR
        | A64Opcode::STURB
        | A64Opcode::STURH
        | A64Opcode::STLR
        | A64Opcode::STLRB
        | A64Opcode::STLRH
        | A64Opcode::STP
        | A64Opcode::STNP
        | A64Opcode::CCMP
        | A64Opcode::CCMN
        | A64Opcode::B
        | A64Opcode::BR
        | A64Opcode::BL
        | A64Opcode::BLR
        | A64Opcode::RET
        | A64Opcode::CBZ
        | A64Opcode::CBNZ
        | A64Opcode::TBZ
        | A64Opcode::TBNZ
        | A64Opcode::ISB
        | A64Opcode::SB
        | A64Opcode::HINT
        | A64Opcode::CLREX
        | A64Opcode::SVC
        | A64Opcode::HVC
        | A64Opcode::SMC
        | A64Opcode::BRK
        | A64Opcode::HLT
        | A64Opcode::MSR
        | A64Opcode::UDF
        | A64Opcode::Invalid => None,
        A64Opcode::DMB(_) | A64Opcode::DSB(_) | A64Opcode::SYS(_) => None,
        _ => match insn.operands[0] {
            A64Operand::Register(_, r) if r != 31 => Some(ConstantOp::Kill { dest_reg: r }),
            _ => None,
        },
    }
}
