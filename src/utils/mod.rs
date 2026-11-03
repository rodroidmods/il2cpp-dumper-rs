pub mod pattern_search;
pub mod string_utils;

pub use pattern_search::find_bytes;
pub use string_utils::{
    CPP_RESERVED_KEYWORDS, CPP_RESERVED_SPECIAL, LIBC_RESERVED_NAMES, NameSanitizerOptions,
    escape_string, escape_string_preview, sanitize_cpp_identifier,
    sanitize_mangled_identifier_chars, sanitize_path_component,
};
