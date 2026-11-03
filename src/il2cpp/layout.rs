//! Evidence-based validation of metadata record sizes.
//!
//! A metadata section is a flat array of fixed-size records, so `size /
//! record_size` is the element count — and if `record_size` is wrong the whole
//! section decodes shifted and turns into out-of-range indices. That failure is
//! indistinguishable from "this game protects/obfuscates its metadata" unless
//! the layout is checked against the bytes.
//!
//! This module checks it. Candidates are scored on facts that a wrong stride
//! cannot satisfy by accident:
//!
//!   * `nameIndex` must land on a printable character of the string section;
//!   * `declaringType` must be a valid type definition index;
//!   * the record's token must be a method token (`0x06xxxxxx`);
//!   * the `parameterCount` fields must sum to exactly the number of parameter
//!     definitions the header declares.
//!
//! The last one is the strongest: it ties two independent sections together, so
//! a stride that is off by even one word fails it. It is what identified the
//! v31 method layout (40 bytes, not 36) on a real file.

/// The result of validating the methods section.
#[derive(Debug, Clone)]
pub struct MethodLayout {
    /// Stride the record actually uses.
    pub stride: usize,
    /// Stride the version tables say it uses.
    pub declared: usize,
    /// True when `stride` passed every check.
    pub accepted: bool,
    /// Human-readable justification, for the console.
    pub reason: String,
}

/// Upper bound on the number of records examined per candidate. The parameter
/// sum is computed over the whole section (it is a two-section invariant), but
/// the sampling checks stop early on the first clearly-wrong candidate.
const MAX_RECORDS: usize = 400_000;

/// Strides we are willing to consider. Every real Il2CppMethodDefinition is
/// 32..=80 bytes and 4-byte aligned.
const MIN_STRIDE: usize = 16;
const MAX_STRIDE: usize = 96;

fn u32_at(data: &[u8], at: usize) -> Option<u32> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn i32_at(data: &[u8], at: usize) -> Option<i32> {
    u32_at(data, at).map(|v| v as i32)
}

fn u16_at(data: &[u8], at: usize) -> Option<u16> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
}

#[derive(Default)]
struct Score {
    name_ok: usize,
    decl_ok: usize,
    token_ok: usize,
    parameter_sum: u64,
    count: usize,
}

impl Score {
    fn name_fraction(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.name_ok as f64 / self.count as f64
        }
    }
    fn decl_fraction(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.decl_ok as f64 / self.count as f64
        }
    }
    fn token_fraction(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.token_ok as f64 / self.count as f64
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn score_stride(
    data: &[u8],
    methods_offset: u64,
    methods_size: u64,
    string_offset: u64,
    string_size: u64,
    type_definition_count: usize,
    parameter_definition_count: usize,
    stride: usize,
) -> Option<Score> {
    if stride < MIN_STRIDE || stride > MAX_STRIDE || stride % 4 != 0 {
        return None;
    }
    if methods_size % stride as u64 != 0 {
        return None;
    }
    let count = (methods_size / stride as u64) as usize;
    if count == 0 || count > MAX_RECORDS {
        return None;
    }
    let base = methods_offset as usize;
    let str_base = string_offset as usize;
    let str_end = str_base.saturating_add(string_size as usize);

    let mut s = Score::default();
    for i in 0..count {
        let at = base + i * stride;
        let Some(name) = u32_at(data, at) else { break };
        let Some(decl) = i32_at(data, at + 4) else {
            break;
        };
        s.count += 1;

        let name_at = str_base.saturating_add(name as usize);
        if name_at < str_end {
            if let Some(&c) = data.get(name_at) {
                if (0x20..0x7f).contains(&c) {
                    s.name_ok += 1;
                }
            }
        }
        if decl >= 0 && (decl as usize) < type_definition_count {
            s.decl_ok += 1;
        }
        if stride >= 16 {
            // The method token sits 12 bytes before the end of the record in
            // every layout Unity has shipped (token, then flags/iflags, then
            // slot/parameterCount).
            if let Some(tok) = u32_at(data, at + stride - 12) {
                if tok & 0xFF00_0000 == 0x0600_0000 {
                    s.token_ok += 1;
                }
            }
        }
        if let Some(pc) = u16_at(data, at + stride - 2) {
            s.parameter_sum += pc as u64;
        }
    }
    if s.count != count {
        return None;
    }
    if parameter_definition_count > 0 && s.parameter_sum != parameter_definition_count as u64 {
        // Keep the score for reporting, but flag it as unusable via the sum.
        return Some(s);
    }
    Some(s)
}

fn passes(s: &Score, parameter_definition_count: usize) -> bool {
    s.count > 0
        && s.name_fraction() >= 0.995
        && s.decl_fraction() >= 0.995
        && s.token_fraction() >= 0.995
        && (parameter_definition_count == 0 || s.parameter_sum == parameter_definition_count as u64)
}

/// Decide the method-definition stride from the bytes themselves.
#[allow(clippy::too_many_arguments)]
pub fn detect_method_stride(
    data: &[u8],
    methods_offset: u64,
    methods_size: u64,
    string_offset: u64,
    string_size: u64,
    type_definition_count: usize,
    parameter_definition_count: usize,
    declared_stride: usize,
) -> MethodLayout {
    let mut candidates: Vec<usize> = Vec::new();
    for stride in (MIN_STRIDE..=MAX_STRIDE).step_by(4) {
        candidates.push(stride);
    }
    if !candidates.contains(&declared_stride) {
        candidates.push(declared_stride);
    }

    let mut best: Option<(usize, Score, bool)> = None;
    // The declared stride wins ties: it is the version table's answer, and the
    // check exists to catch it being wrong, not to second-guess it.
    for stride in candidates {
        let Some(s) = score_stride(
            data,
            methods_offset,
            methods_size,
            string_offset,
            string_size,
            type_definition_count,
            parameter_definition_count,
            stride,
        ) else {
            continue;
        };
        let ok = passes(&s, parameter_definition_count);
        let better = match &best {
            None => true,
            Some((best_stride, best_score, best_ok)) => {
                let best_rank = (
                    *best_ok,
                    best_score.name_fraction() + best_score.decl_fraction(),
                );
                let this_rank = (ok, s.name_fraction() + s.decl_fraction());
                this_rank > best_rank
                    || (this_rank == best_rank
                        && stride == declared_stride
                        && *best_stride != declared_stride)
            }
        };
        if better {
            best = Some((stride, s, ok));
        }
    }

    match best {
        None => MethodLayout {
            stride: declared_stride,
            declared: declared_stride,
            accepted: false,
            reason: format!("no candidate stride divides the {methods_size}-byte methods section"),
        },
        Some((stride, score, ok)) if ok && stride != declared_stride => MethodLayout {
            stride,
            declared: declared_stride,
            accepted: true,
            reason: format!(
                "record size is {stride} bytes, not {declared_stride} ({} records, \
                 parameterCount sums to {})",
                (methods_size / stride as u64),
                score.parameter_sum
            ),
        },
        Some((stride, score, ok)) => MethodLayout {
            stride: if ok { stride } else { declared_stride },
            declared: declared_stride,
            accepted: ok,
            reason: if ok {
                format!(
                    "{stride}-byte records confirmed ({} records, parameterCount sums to {})",
                    score.count, score.parameter_sum
                )
            } else {
                format!(
                    "could not confirm a layout: best candidate was {stride} bytes \
                     (names {:.1}%, declaringType {:.1}%, token {:.1}%, \
                     parameterCount sum {} vs {parameter_definition_count} parameters)",
                    score.name_fraction() * 100.0,
                    score.decl_fraction() * 100.0,
                    score.token_fraction() * 100.0,
                    score.parameter_sum,
                )
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRING_OFF: usize = 64;
    const STRINGS: &[u8] = b"alpha\0beta\0gamma\0delta\0";
    const METHODS_OFF: usize = 256;

    /// Build a metadata image whose methods section uses `stride`-byte records.
    fn image(stride: usize, records: usize) -> Vec<u8> {
        let mut data = vec![0u8; METHODS_OFF + stride * records + 64];
        data[STRING_OFF..STRING_OFF + STRINGS.len()].copy_from_slice(STRINGS);
        for i in 0..records {
            let at = METHODS_OFF + i * stride;
            let name_off = (((i % 4) * 6) as u32).to_le_bytes();
            data[at..at + 4].copy_from_slice(&name_off);
            data[at + 4..at + 8].copy_from_slice(&1i32.to_le_bytes());
            data[at + stride - 12..at + stride - 8]
                .copy_from_slice(&(0x0600_0000u32 | (i as u32 + 1)).to_le_bytes());
            data[at + stride - 2..at + stride].copy_from_slice(&1u16.to_le_bytes());
        }
        data
    }

    fn detect(data: &[u8], stride: usize, records: usize, declared: usize) -> MethodLayout {
        detect_method_stride(
            data,
            METHODS_OFF as u64,
            (stride * records) as u64,
            STRING_OFF as u64,
            STRINGS.len() as u64,
            4,
            records,
            declared,
        )
    }

    /// The v31 case: records are 40 bytes, the tables said 36.
    #[test]
    fn detects_a_record_larger_than_the_version_table() {
        let data = image(40, 500);
        let got = detect(&data, 40, 500, 36);
        assert!(got.accepted, "reason: {}", got.reason);
        assert_eq!(got.stride, 40);
        assert_eq!(got.declared, 36);
    }

    /// The mirror case: the declared stride is too *large*, so it does not even
    /// divide the section and the bytes win.
    #[test]
    fn corrects_a_declared_stride_that_does_not_divide_the_section() {
        let data = image(36, 500);
        let got = detect(&data, 36, 500, 40);
        assert!(got.accepted, "reason: {}", got.reason);
        assert_eq!(got.stride, 36);
    }

    /// When no candidate stride divides the section at all, nothing is claimed:
    /// the declared size is kept and the caller is told it could not validate.
    #[test]
    fn reports_a_section_no_candidate_can_explain() {
        let mut data = image(40, 10);
        // A section length that is not a multiple of 4 cannot be a record array
        // at any of the candidate sizes.
        let odd_size = 34u64;
        data.truncate(METHODS_OFF + odd_size as usize);
        let got = detect_method_stride(
            &data,
            METHODS_OFF as u64,
            odd_size,
            STRING_OFF as u64,
            STRINGS.len() as u64,
            4,
            10,
            40,
        );
        assert!(!got.accepted, "reason: {}", got.reason);
        assert_eq!(got.stride, 40, "the declared stride is kept");
    }

    /// A section whose parameter counts disagree with the parameter section is
    /// not trusted, even when its size divides and its names look fine.
    #[test]
    fn rejects_a_stride_whose_parameter_counts_do_not_add_up() {
        let data = image(40, 500);
        let got = detect_method_stride(
            &data,
            METHODS_OFF as u64,
            40 * 500,
            STRING_OFF as u64,
            STRINGS.len() as u64,
            4,
            499, // one parameter short
            40,
        );
        assert!(!got.accepted, "reason: {}", got.reason);
    }

    /// The correct stride must be confirmed, not merely tolerated.
    #[test]
    fn confirms_the_declared_stride_when_it_is_right() {
        let data = image(32, 300);
        let got = detect(&data, 32, 300, 32);
        assert!(got.accepted, "reason: {}", got.reason);
        assert_eq!(got.stride, 32);
    }
}
