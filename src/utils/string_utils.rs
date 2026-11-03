pub static CPP_RESERVED_KEYWORDS: &[&str] = &[
    "klass",
    "monitor",
    "register",
    "_cs",
    "auto",
    "friend",
    "template",
    "flat",
    "default",
    "_ds",
    "interrupt",
    "unsigned",
    "signed",
    "asm",
    "if",
    "case",
    "break",
    "continue",
    "do",
    "new",
    "_",
    "short",
    "union",
    "class",
    "namespace",
    "volatile",
    "const",
    "extern",
    "static",
    "struct",
    "typedef",
    "enum",
    "return",
    "switch",
    "goto",
    "void",
    "for",
    "while",
    "else",
    "sizeof",
    "this",
    "public",
    "private",
    "protected",
    "operator",
    "true",
    "false",
    "nullptr",
    "method",
];

pub static CPP_RESERVED_SPECIAL: &[&str] = &["inline", "near", "far"];

pub static LIBC_RESERVED_NAMES: &[&str] = &[
    "__sF",
    "__sE",
    "__sS",
    "__stdin",
    "__stdout",
    "__stderr",
    "__cleanup",
    "__progname",
    "__environ",
    "errno",
    "stdin",
    "stdout",
    "stderr",
];

#[derive(Debug, Clone, Copy, Default)]
pub struct NameSanitizerOptions {
    pub allow_dollar: bool,
    pub avoid_double_underscore_prefix: bool,
}

pub fn sanitize_path_component(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => out.push('_'),
            c if (c as u32) < 0x20 => out.push('_'),
            c => out.push(c),
        }
    }
    let trimmed = out
        .trim_end_matches(|c: char| c == '.' || c == ' ')
        .to_string();
    if trimmed.is_empty() {
        "_".into()
    } else {
        trimmed
    }
}

pub fn sanitize_cpp_identifier(name: &str, opts: NameSanitizerOptions) -> String {
    if CPP_RESERVED_KEYWORDS.contains(&name) {
        return format!("_{name}");
    }
    if CPP_RESERVED_SPECIAL.contains(&name) {
        return format!("_{name}_");
    }
    let first = name.chars().next();
    if first.map(|c| c.is_ascii_digit()).unwrap_or(false) {
        return format!("_{name}");
    }

    let mut result = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || (opts.allow_dollar && c == '$') {
            result.push(c);
        } else {
            result.push('_');
        }
    }

    if opts.avoid_double_underscore_prefix && result.starts_with("__") {
        result = format!("f{result}");
    }

    if LIBC_RESERVED_NAMES.contains(&result.as_str()) {
        result = format!("_{result}_");
    }

    result
}

pub fn sanitize_mangled_identifier_chars(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for c in id.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    out
}

pub fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\0' => result.push_str("\\0"),
            c if c.is_control() => {
                result.push_str(&format!("\\x{:02X}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

pub fn escape_string_preview(s: &str, max_len: usize) -> String {
    let escaped = escape_string(s);
    if escaped.len() <= max_len {
        return escaped;
    }
    let cutoff = max_len.saturating_sub(3);
    let safe_end = escaped
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= cutoff)
        .last()
        .unwrap_or(0);
    format!("{}...", &escaped[..safe_end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("he\"llo"), "he\\\"llo");
        assert_eq!(escape_string("line\nnew"), "line\\nnew");
        assert_eq!(escape_string("tab\there"), "tab\\there");
        assert_eq!(escape_string("back\\slash"), "back\\\\slash");
    }
}
