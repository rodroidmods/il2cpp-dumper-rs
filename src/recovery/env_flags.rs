//! Cached boolean environment flags for per-method hot paths.
//!
//! `std::env::var` takes the process environment lock and allocates. The
//! render pipeline asked for the same handful of debug flags millions of
//! times (46 passes x 3 lookups x 24k methods, plus per-node checks inside
//! `il2cpp_cleanup`), and every lookup serializes the rayon workers. These
//! flags are configured before the dump starts, so the answer is cached per
//! name.
//!
//! Test builds bypass the cache: unit tests toggle flags between cases.

#[cfg(not(test))]
use std::collections::HashMap;
#[cfg(not(test))]
use std::sync::{Mutex, OnceLock};

#[cfg(not(test))]
pub(crate) fn flag(name: &'static str) -> bool {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(v) = guard.get(name) {
        return *v;
    }
    let v = std::env::var(name).is_ok();
    guard.insert(name, v);
    v
}

#[cfg(test)]
pub(crate) fn flag(name: &'static str) -> bool {
    std::env::var(name).is_ok()
}
