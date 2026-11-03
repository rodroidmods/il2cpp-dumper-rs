//! Phase 8: ptr_no_offset resolution — infer type context for GOT entries.
//!
//! ## What this does
//!
//! When a GOT entry points to a .data address that holds an Il2CppClass* pointer,
//! the address is runtime-initialized and can't be read from the binary. The
//! `field_annotation` pass generates `ptr_0x{addr}` patterns for these.
//!
//! This pass uses method context to infer the type: when a method in type T
//! accesses `ptr_0xY`, it's likely accessing T's Il2CppClass (for static field
//! access, type checks, etc.).
//!
//! ## How it works
//!
//! 1. During rendering, when `field_annotation` generates `ptr_0x{addend}`, it
//!    records the declaring type in a global map.
//! 2. After all methods are rendered, we resolve unambiguous ones (where all
//!    usages are from the same type).
//! 3. We do text replacement on the output to replace `void*.ptr_0x{addr}` with
//!    `T.klass(T)` or just `T` depending on context.
//!
//! ## Safety
//!
//! Only resolves when ALL usages of a ptr come from methods of the same type.
//! Ambiguous ptrs (used by multiple types) are left as `ptr_0x{addr}`.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Global state (accumulated during rendering)
// ---------------------------------------------------------------------------

/// Global map: ptr addend → set of declaring type names that use it.
/// Accumulated during rendering, resolved after.
static PTR_USAGE: Mutex<Option<HashMap<u64, HashSet<String>>>> = Mutex::new(None);

/// Initialize the global ptr usage map (call before rendering starts).
pub fn init() {
    let mut guard = PTR_USAGE.lock().unwrap();
    *guard = Some(HashMap::new());
}

/// Record that a method with the given declaring type uses a ptr at the given addend.
pub fn record_usage(addend: u64, declaring_type: &str) {
    let mut guard = PTR_USAGE.lock().unwrap();
    if let Some(map) = guard.as_mut() {
        map.entry(addend)
            .or_insert_with(HashSet::new)
            .insert(declaring_type.to_string());
    }
}

/// Resolve unambiguous ptrs and return a map: addend → type name.
/// Call after all methods are rendered.
pub fn resolve() -> HashMap<u64, String> {
    let mut guard = PTR_USAGE.lock().unwrap();
    let map = guard.take().unwrap_or_default();
    let mut resolved = HashMap::new();
    for (addend, types) in map {
        if types.len() == 1 {
            if let Some(type_name) = types.into_iter().next() {
                resolved.insert(addend, type_name);
            }
        }
    }
    resolved
}

/// Build replacement tables once (call before per-method loop).
/// Returns HashMaps for O(1) lookup instead of O(n) iteration.
pub fn build_replacement_tables(resolved: &HashMap<u64, String>) -> (HashMap<String, String>, HashMap<String, String>) {
    let mut ptr_replacements: HashMap<String, String> = HashMap::with_capacity(resolved.len());
    let mut deref_replacements: HashMap<String, String> = HashMap::with_capacity(resolved.len());
    for (&addend, type_name) in resolved {
        // Replace ptr_0x{addr} with typeof(T) — the managed equivalent of
        // loading the Il2CppClass* for a type.
        ptr_replacements.insert(
            format!("void*.ptr_0x{:x}", addend),
            format!("typeof({})", type_name),
        );
        // The deref pattern *(nint*)(typeof(T)) was how IL2CPP loaded the
        // klass pointer from the GOT. In managed C# this is just typeof(T).
        deref_replacements.insert(
            format!("*(nint*)(typeof({}))", type_name),
            format!("typeof({})", type_name),
        );
    }
    (ptr_replacements, deref_replacements)
}

/// Apply text-level replacements to a single method's text.
/// Uses HashMap lookup for O(1) replacement instead of O(n) iteration.
pub fn apply_replacements_with_tables(
    text: &mut String,
    ptr_replacements: &HashMap<String, String>,
    deref_replacements: &HashMap<String, String>,
) {
    if ptr_replacements.is_empty() && deref_replacements.is_empty() {
        return;
    }
    // Early exit: skip if no ptr_0x patterns in this method
    if !text.contains("ptr_0x") {
        return;
    }
    // Process line-by-line: skip lines that don't contain any patterns
    let mut result = String::with_capacity(text.len());
    for line in text.lines() {
        if line.contains("ptr_0x") {
            let mut processed = line.to_string();
            // Find all ptr_0x patterns and replace them
            // Look for "void*.ptr_0x" pattern
            while let Some(start) = processed.find("void*.ptr_0x") {
                let after_prefix = start + "void*.ptr_0x".len();
                // Find the end of the hex address (next non-hex char or end of string)
                let end = processed[after_prefix..]
                    .find(|c: char| !c.is_ascii_hexdigit())
                    .map(|i| after_prefix + i)
                    .unwrap_or(processed.len());
                let pattern = &processed[start..end];
                if let Some(replacement) = ptr_replacements.get(pattern) {
                    processed = format!("{}{}{}", &processed[..start], replacement, &processed[end..]);
                } else {
                    break; // No replacement found, skip this pattern
                }
            }
            // Look for "*(nint*)(typeof(T))" pattern
            while let Some(start) = processed.find("*(nint*)(typeof(") {
                let after_prefix = start + "*(nint*)(".len();
                // Find the closing paren
                if let Some(end_offset) = processed[after_prefix..].find("))") {
                    let end = after_prefix + end_offset + 2;
                    let pattern = &processed[start..end];
                    if let Some(replacement) = deref_replacements.get(pattern) {
                        processed = format!("{}{}{}", &processed[..start], replacement, &processed[end..]);
                    } else {
                        break; // No replacement found, skip this pattern
                    }
                } else {
                    break;
                }
            }
            result.push_str(&processed);
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }
    *text = result;
}

/// Report the number of resolved ptrs.
pub fn report(resolved_count: usize) -> String {
    if resolved_count == 0 {
        return String::new();
    }
    format!("[ptr_resolution] resolved {} ptr_no_offset patterns\n", resolved_count)
}
