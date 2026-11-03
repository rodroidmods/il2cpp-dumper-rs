pub mod config;
pub mod disassembler;
pub mod error;
pub mod executor;
pub mod formats;
pub mod il2cpp;
pub mod io;
pub mod output;
pub mod recovery;
pub mod search;
pub mod utils;

/// `eprintln!` for the decompiler's internal progress logs (`[debug] …`).
///
/// Those lines (ELF relocation counts, GOT/static-field layout population)
/// are diagnostics, not part of the run's output, and are silent unless a
/// developer sets `ILR_DEBUG=1`.
#[macro_export]
macro_rules! dbg_eprintln {
    ($($arg:tt)*) => {
        if crate::recovery::env_flags::flag("ILR_DEBUG") {
            eprintln!($($arg)*);
        }
    };
}
