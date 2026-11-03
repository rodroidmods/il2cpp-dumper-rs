<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" />
  <img src="https://img.shields.io/badge/IL2CPP-v16--v39-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🛡️ Rodroid Il2CppDumper V8.5</h1>

<p align="center">
  <b>A blazing-fast, cross-platform IL2CPP binary dumper written in Rust.</b><br/>
  Full rewrite from the original C# <a href="https://github.com/Perfare/Il2CppDumper">Il2CppDumper</a> with significant performance improvements, modern CLI UI, and advanced features.
</p>

<p align="center">
  <a href="https://t.me/+WmudnO0-xoNhMDQ8">📢 Telegram Channel</a> &nbsp;·&nbsp;
  <a href="https://t.me/+QylrYL1GNsJiYjc0">💬 Telegram Group</a> &nbsp;·&nbsp;
  <b>Dev:</b> <a href="https://t.me/rodroidmods"><code>@rodroidmods</code></a>
</p>
<p align="center">
  <a href="https://github.com/rodroidmods/rodroid-il2cppdumper/releases">📥 <b>Download prebuilt releases</b></a> (APK, EXE, DMG, AppImage, DEB, RPM, IPA)<br/>
  Available at <a href="https://github.com/rodroidmods/rodroid-il2cppdumper/releases">github.com/rodroidmods/rodroid-il2cppdumper/releases</a>
</p>

---

## 🏆 Feature Comparison — Rust vs C# vs Python

| Feature | [Python](https://github.com/springmusk026/Il2CppDumper-Python) | [C#](https://github.com/Perfare/Il2CppDumper) | **Rust (This)** |
|---------|:---:|:---:|:---:|
| **dump.cs generation** | ⚠️ Basic (no properties) | ✅ Full | ✅ Full |
| **DummyDLL generation** | ❌ No | ✅ Yes | ✅ Yes (parallel) |
| **il2cpp.h struct gen** | ⚠️ Basic | ✅ Full | ✅ Full |
| **script.json** | ✅ Yes | ✅ Yes | ✅ Yes |
| **stringliteral.json** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Split Dump Per Type (DiffableCS)** | ❌ No | ❌ No | ✅ **Yes (parallel)** |
| **C# Decompiler (native bodies → C#)** | ❌ No | ❌ No | ✅ **Yes (pre-alpha)** |
| **Variable-width indices (v39/Unity 6)** | ❌ No | ❌ No | ✅ **Yes** |
| **Auto-XOR Metadata Decryption** | ❌ No | ❌ No | ✅ **Yes** |
| **Latest Unity Formats (v104, v106)** | ❌ No | ❌ No | ✅ **Yes** |
| **Assembly name in dump.cs** | ❌ No | ❌ No | ✅ **Yes** |
| **Unity version detection** | ❌ No | ❌ No | ✅ **Auto-detect** |
| **Inline Disassembly (ARM64/ARM32/x86/x64)** | ❌ No | ❌ No | ✅ **Yes** |
| **Control Flow Graph (CFG) Analysis** | ❌ No | ❌ No | ✅ **Yes** |
| **Metadata Annotations (strings, types, vtable)** | ❌ No | ❌ No | ✅ **Yes** |
| **Semantic Variable Tracking** | ❌ No | ❌ No | ✅ **Yes** |
| **Forward Constant Propagation** | ❌ No | ❌ No | ✅ **Yes** |
| **Backward Slicing (Vtable Resolution)** | ❌ No | ⚠️ Partial (Cpp2IL) | ✅ **Yes** |
| **Init-Check Folding** | ❌ No | ❌ No | ✅ **Yes** |
| **String Literal Indirect Resolution (`il2cpp_string_new_wrapper`)** | ❌ No | ❌ No | ✅ **Yes** |
| **Generic Instantiation Tracking** | ❌ No | ❌ No | ✅ **Yes** |
| **Switch Table Reconstruction** | ❌ No | ❌ No | ✅ **Yes** |
| **Boxing/Unboxing Pattern Detection** | ❌ No | ❌ No | ✅ **Yes** |
| **Static Field Access Annotation** | ❌ No | ❌ No | ✅ **Yes** |
| **CODM (Call of Duty Mobile) Support** | ❌ No | ❌ No | ✅ **Yes (Android + iOS, 32/64-bit)** |
| **Fat Mach-O (Universal)** | ❌ No | ✅ Yes | ✅ Yes |
| **WASM (WebGL)** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Dump file support** | ❌ No | ✅ Yes | ✅ Yes (+ ELF reload) |
| **v27+ ImageBase fix** | ❌ No | ✅ Yes | ✅ Yes |
| **Parallel dump / DummyDLL / script.json / structs (`rayon`)** | ❌ No | ❌ No | ✅ **Yes** |
| **Safe capacity checks (no alloc panic)** | ❌ No | ❌ No | ✅ **Yes** |
| **Clear metadata protection errors** | ⚠️ Generic | ⚠️ Generic | ✅ **Yes** |
| **AutoPlus version refine (e.g. metadata 31 → CR layout 29)** | ❌ No | ⚠️ Manual | ✅ **Yes (all formats)** |
| **Duplicate metadata key tolerance** | ❌ Crashes | ❌ Crashes | ✅ **Yes** |
| **Auto-numbered output dirs** | ❌ No | ❌ No | ✅ **Dump0/, Dump1/...** |
| **Modern CLI UI (spinners, colors, prompts)** | ❌ No | ❌ No | ✅ **Yes** |
| **Cross-platform binary** | ⚠️ Needs Python | ⚠️ Needs .NET | ✅ **Standalone** |
| **C++ Scaffold (il2cpp-functions.h)** | ❌ No | ❌ No | ✅ **Yes** |
| **C++ Name Mangling (Itanium ABI)** | ❌ No | ❌ No | ✅ **Yes** |
| **Unity Header Auto-Detection** | ❌ No | ✅ Yes | ✅ **Yes (version-matched)** |
| **cpp_project/ Scaffolding** | ❌ No | ✅ Yes | ✅ **Yes** |
| **Topological Sort (type ordering)** | ❌ No | ❌ No | ✅ **Yes** |
| **Type Group Classification** | ❌ No | ❌ No | ✅ **Yes** |
| **Enhanced IDA Metadata** | ❌ No | ❌ No | ✅ **Yes** |
| **ELF Section Header Symbol Fallback** | ❌ No | ✅ Yes | ✅ **Yes** |
| **GUI** | ❌ No | ✅ WinForms | ✅ **Jetpack Compose + Tauri** |
| **Embeddable as library** | ❌ No | ❌ No | ✅ **Rust crate / JNI** |

### ⚡ Performance

| Phase | Python | C# | **Rust** |
|-------|:---:|:---:|:---:|
| Metadata loading | ~3.7s | ~2s | **~0.5s** |
| Binary loading | ~5.2s | ~3s | **~0.8s** |
| Search & Init | ~5.4s | ~2s | **~0.3s** |
| dump.cs | ~14s | ~5s | **~1–3s** (parallel; +disasm scales with max insn) |
| Struct generation | ~6.4s | ~5s | **~2–4s** (parallel headers / script) |
| DummyDLL | ❌ N/A | ~3s | **~1–2s** (parallel) |
| **Total** | **~35s** | **~20s** | **~4–12s** (disasm-heavy dumps still multi-core) |

> **Often 3–5× faster than C# / Python** on multi-core — same dump logic, map-reduce with `rayon`.

---

## ✨ Features

### Core Dumping
- **dump.cs** — Full C# class/method/field/property decompilation with RVA/VA/Offset and assembly names
- **Inline Disassembly** — Optional per-method native assembly embedded directly in dump.cs
- **DiffableCs** — Splits classes into individual `.cs` files by namespace, parallelized with `rayon`
- **script.json** — Method addresses/signatures for IDA/Ghidra scripting
- **il2cpp.h** — C struct definitions for native analysis with topological type ordering
- **stringliteral.json** — All string literal values and indices
- **DummyDLL** — Reconstructed .NET assemblies for dnSpy/ILSpy (parallelized)

### C# Decompiler — PRE-ALPHA (v8)
- **Recovered method bodies** — native ARM64 / ARM32 / x86-64 IL2CPP bodies
  rendered as C#-style pseudocode inline in `dump.cs` (and/or the split dump),
  in parallel via `rayon`.
- **High-level constructs** — `if`/`while`/`for`/`foreach`/`switch`,
  `try`/`catch`/`finally`, `&&`/`||` guard chains, ternaries, object
  initializers, `ref`/`out` parameters, closures, delegates.
- **Metadata-aware naming** — fields, static fields, string literals, types,
  vtable/devirtualised calls, generic instantiations, class-init checks.
- **Engine internal calls** — Unity icall stubs (`…_Injected`) and delegate
  runtime methods render as `[MethodImpl(InternalCall)] extern` declarations,
  not as resolver bodies.
- **Scoped by filter** — `csharpIncludeAssemblies` / `csharpIncludeTypes`
  limit which bodies are recovered; `csharpFullDump` keeps the rest of the dump
  intact.
- **Pre-alpha:** output is readable C#, not guaranteed-compilable C#. Progress
  and remaining gaps are tracked in `HANDOFF.md` §1 and `NEXT_STEPS.md` §0.

### C++ Headers & Scaffolding
- **il2cpp-functions.h** — C++ scaffold with function pointer typedefs for hooking
- **Itanium ABI Name Mangling** — Correct C++ mangled names for all IL2CPP types
- **Unity Header Auto-Detection** — Version-matched `il2cpp-types.h` and `il2cpp-api.h` from embedded header database
- **cpp_project/** — Ready-to-compile C++ project scaffold with includes and CMake structure
- **Topological Sort** — Types emitted in dependency order; circular dependencies detected with fallback
- **Type Group Classification** — Types categorized into forward declarations, method types, generic types, usage types
- **Compiler Layout** — GCC (`__attribute__`) or MSVC (`__declspec`) layout attributes
- **Enhanced IDA Metadata** — Extra type info annotations for IDA Pro scripts

### Disassembly Engine
- **Multi-Architecture** — ARM64 (`yaxpeax-arm`), ARM32, x86/x64 (`iced-x86`)
- **Control Flow Graph (CFG)** — Basic block detection, branch targets, loop back-edges, `if (condition)` reconstruction
- **Metadata Annotations** — String literals, type info, method/field references, vtable resolution via `ADRP+LDR` patterns
- **Semantic Variable Tracking** — Maps registers to parameter names (`X0` → `this`, `X1` → `arg0`)
- **Forward Constant Propagation** — Tracks register values (MOVZ/MOVK/ADD/SUB/ORR/ADRP) across instructions to resolve register+register memory accesses like `LDR X0, [X22, X21, LSL #3]` into field names (e.g. `// this.<>2__current`). First IL2CPP tool to annotate indexed field accesses.
- **Backward Slicing for Vtable Resolution** — On `BLR Xn`, walks backward through `LDR X8, [X0]` (klass) → `LDR X9, [X8, #N]` (vtable slot) chains to resolve indirect calls into `// virtual call: TypeName.MethodName` instead of opaque `sub_XXXXXX`.
- **Initialization-Check Folding** — Detects and collapses `il2cpp_codegen_initialize_method`, `Il2CppCodeGenWriteBarrier`, and `TBZ/TBNZ`-on-bit-0 prologue patterns into a single `// [init check]` annotation, drastically reducing method-body noise.
- **Indirect String Literal Resolution** — Tracks the literal index in `W0/W1` at `il2cpp_string_new_wrapper` call sites and resolves through `metadata_string_literals` to annotate the actual string content (`// "Hello, world"`).
- **Generic Instantiation Tracking** — Resolves calls into `MethodInfo*` slots via `method_definition_method_specs` to annotate the concrete specialization (e.g. `// → List<int>.Add(this, item)`).
- **Switch Table Reconstruction** — Detects ARM64 jump-table prologues (`ADRP+ADD+LDR Xn, [Xn, Xidx, lsl #2]+BR Xn`) using reg+reg propagation and emits `switch (var)` blocks in the CFG.
- **Boxing / Unboxing Detection** — Annotates `il2cpp_codegen_box` / `il2cpp_unbox` call sites with the resolved boxed type from the first-arg type pointer.
- **Static Field Access Annotation** — Resolves the `ADRP+ADD → LDR X8, [Xklass, #static_fields_offset] → LDR Wd, [X8, #field_offset]` pattern into `// SomeClass.staticField` using the existing klass identification map.
- **Configurable** — Toggle hex bytes, field names, annotations, CFG independently

### Supported Platforms

| Platform | Format | Status |
|----------|--------|:---:|
| Android | ELF32 / ELF64 | ✅ |
| iOS / macOS | Mach-O / Fat Mach-O | ✅ |
| Windows | PE32 / PE64 | ✅ |
| Nintendo Switch | NSO | ✅ |
| WebGL | WASM | ✅ |

### IL2CPP Versions
- **v16 – v39** (Unity 5.3 → Unity 6)
- Variable-width indices for v39/Unity 6
- Newer metadata formats: `v104`, `v106`, `v106.1`, `v107`, `v108`
- Auto XOR metadata decryption (1-byte, 4-byte, 8-byte, rolling, position-dependent, header-only)
- Manual version override via config

### CODM (Call of Duty Mobile)
- Custom v23 metadata layout with two-slot `type_definitions_count` fingerprint anchor
- Android packed relocations (`DT_ANDROID_RELA` / `DT_ANDROID_REL`, APS2 + SLEB128) for 32-bit and 64-bit ELF
- iOS chained fixups (`LC_DYLD_CHAINED_FIXUPS`) and legacy rebase opcodes (`LC_DYLD_INFO_ONLY`) for 32-bit and 64-bit Mach-O
- Pointer formats: `DYLD_CHAINED_PTR_64`, `_64_OFFSET`, ARM64E variants
- **Self-healing Mach-O VA resolver** — if a slot's chain wasn't walked, `map_vatr` decodes the raw chained pointer on the fly (CODM-gated, no impact on standard Mach-O)
- **Il2CppType decryption** — XOR-with-`0x35` stream cipher on `bits` (4 bytes) and `datapoint` (8 bytes), with three combined detection markers:
  1. `bits` byte 3 low 5 bits == `0x15` (encrypted `num_mods` signature)
  2. `datapoint` high 32 bits == `0x35353535` (encrypted klass_index high half)
  3. Current `type_enum` invalid + XOR'd `type_enum` valid (catches obfuscator-only patterns like `0x27 → Class`, `0x24 → ValueType`)
- ~95 % of encrypted Il2CppType entries recovered (class names, value types, generic instances, field type references). The remaining ~5 % are **intentional decoy fields** with the `LITERAL` (0x40) attribute flag set on real instance fields — same limitation as the C# CODM dumper; not an encryption gap.
- Toggle via `--codm` flag or `Codm: true` in config — additive code path, leaves standard Unity games untouched

### What's new in v8.5 — the pre-alpha decompiler sync

**Version 8.5.0. The C# decompiler is declared PRE-ALPHA.** The metadata path
(`dump.cs`, DummyDLL, `il2cpp.h`, `script.json`, `stringliteral.json`) is
stable; recovered bodies are still improving and are not yet guaranteed to
compile. Alpha is gated on the worklist in `NEXT_STEPS.md` §0.

#### Fixed in v8.5

- **Free Fire, Free Fire MAX and Free Fire India:** an IL2CPP **version
  detection** failure made these games fail to dump at all. The version is now
  detected correctly and they dump successfully.
- **CODM:** a Call of Duty Mobile binary could not produce a **full dump**;
  v8.5 writes the complete dump.
- **Zero-warning build**, and **all engine debug output is off by default** —
  every trace is behind an `ILR_*` environment gate (`ILR_DEBUG=1` to re-enable
  the progress logs).
- The identical v8.5 engine now ships in the CLI, the Tauri desktop/mobile app
  and the Android app (app version 8.5, `versionCode` 85).

#### Recovered C# — what the pre-alpha engine added over the v8 beta

The engine grew from ~111k lines (v8.0) to **~225k lines**; the decompiler
(`src/recovery/`) alone went from 33 files / 78k lines to **118 files /
187k lines**.

- **Icall stubs render as `extern`.** Unity engine methods are native internal
  calls — their IL2CPP body is the lazy resolver
  (`*(nint*)(slot) = fn_resolve(…)` then `blr`), not managed code. Those bodies
  now print Unity's own shape,
  `[MethodImpl(MethodImplOptions.InternalCall)] public extern …;`. On the second
  study corpus (*Tomb of Maze*): resolver noise **386 → 0**, extern declarations
  **685 → 1,067**.
- **Honest unreachable warnings.** `// WARNING: N of M blocks unreachable` now
  fires only when a reachable block genuinely ends in an unresolved indirect
  dispatch — it used to stamp neighbouring one-line functions whose decode
  window merely overlapped (179 → 3 on the same corpus).
- **Delegate `Invoke` sugar** and delegate runtime methods
  (`Invoke`/`BeginInvoke`/`EndInvoke`) render as declarations instead of their
  `br` dispatch.
- **Guard-chain flattening** (`if (!x) return;` early-exit cascades unrolled),
  **tail-return ternaries** (`if (C) return E; return F;` →
  `return C ? E : F;`), **guard merging** to `&&`-chains, **object
  initializers**, **switch-arm grouping**.
- **Closure/display-class fields**: `*(nint*)(v351 +32)` becomes
  `v351.cellData`, including fields of those fields (−702 raw sites).
- **Byref bodies**: `ref`/`out` parameters deref correctly, and dead-store
  elimination now refuses to delete a store to a `ref`/`out` parameter.
- **Class-init handling**: raw byte stores `*(byte*)(0x31ac000 +2603) = 1` take
  the `ClassInit.init_…` name their own guard already used (1,714 stores
  named).
- **Empty bodies recovered** (`// body not recoverable` 1,248 → 418; the
  residue is genuine argument-shuffling thunks).
- **`E; return;` is `return E;`.** In a value-returning method a bare
  expression immediately before an operand-less `return` can only be the
  return value — the shape left behind when the walker loses the link between
  a call's result slot and the return register (`Camera.get_current` used to
  print `Camera.currentInternal; return;`). This single fold took
  `// body not recoverable` from **418 → 5** on the reference corpus and
  **33 → 6** on the second one.
- **No fabricated allocations.** The runtime's class-init thunk is sampled into
  the allocator family, so its unused result printed as
  `var camera = new Camera();` (`Camera.FireOnPreCull`) or as a stub method's
  whole body (`AppsFlyerStub.OpenSettings`). An unread, argument-less
  allocation is now removed as what it is — dead code: **1,945 → 39** such
  statements on the reference corpus, ~780 real allocations left intact.

Measured on the reference Temple Run 2 corpus (`OUT_R208AC`, 24,195 methods,
`Assembly-CSharp` only): **512,595 lines / 18.1 MB**, `target not emitted`
**291** (from 6,449), raw `*(…)` accesses **36,410** (from 55,121),
`*(nint*)(…)` **26,434**, gotos **10,263**, `// body not recoverable` **5**
(from 1,248), verifier annotations **1,280 suspect / 248 undefined locals**,
duplicate labels **0**, quiet full dump **178–200 s** on a 2-core host. Second
corpus (*Tomb of Maze*, with `UnityEngine.CoreModule` selected): **199,598
lines**, **1,067** icall externs, `body not recoverable` **6**.

#### New option

- **`csharpFullDump`** (default `false`): with a class filter active, keep the
  complete type/method dump and limit recovered **bodies** to the selection —
  every other method renders `{ }`. Without it, a filter reduces the dump
  itself to the selected types.

### What's new in v8.0 — the C# Decompiler release

**Version 8 brings a new C# decompiler to Rodroid Il2CppDumper**, built over
1.5 months of focused development. It recovers native IL2CPP method bodies into
readable C#-style output, bringing method logic alongside the types, fields,
signatures, and metadata already present in `dump.cs`.

#### A new C# decompiler

- Recover native method bodies into C#-style pseudocode for supported native
  architectures.
- Run method recovery in parallel with Rayon, while keeping the same recovery
  pipeline and output features available in single-worker mode.
- Limit body recovery to selected assemblies or types when you want a focused
  dump.

#### Better compatibility with real-world game dumps

- **Free Fire and Free Fire MAX memory dumps:** fixed crashes encountered when
  dumping these games from memory.
- **Unity 6000.1:** fixed a metadata parsing issue affecting this version.
- **New metadata formats:** added support for metadata versions **106.1, 107,
  and 108**.
- **Additional reliability fixes:** improved handling across metadata parsing,
  native analysis, and dump generation.

Together, these changes make v8 a major step forward for recovering both IL2CPP
metadata and the native method logic behind it.

### What's new in v7.0

#### Stability & correctness
- 🐛 **Fixed `capacity overflow` panic** — untrusted counts from metadata/registration no longer call `Vec::with_capacity` with huge values; invalid sizes return a clear error instead of crashing
- 🐛 **Fixed duplicate-key crash on metadata init** — same class of bug as C# `System.ArgumentException: An item with the same key has already been added` on `fieldDefaultValues` / attribute tokens; we dedupe (keep last) and continue (Steam / Windows games)
- 🐛 **AutoPlus version refine (metadata vs CodeRegistration layout)** — some games report metadata version **31** while native CR/MR still use a **29-era** layout. Manual “fix version” in other dumpers is not required: we refine version from registration counts / heuristics and apply it on **all formats** (ELF / PE / Mach-O / NSO / WASM), CLI + Android + Tauri
- 🛡️ **Safer array reads** — signed size validation, file-bounds checks, ELF hash bucket / registration count caps
- 🛡️ **Single CR/MR log** — no more double-print of CodeRegistration when section scan fails and ARM32/symbol path succeeds

#### Better errors (encrypted / protected games)
- 📣 **Actionable metadata failures** — wrong magic shows hex bytes + expected `AF 1B B1 FA`; truncated/encrypted files no longer look like raw `IO error: failed to fill whole buffer`
- 📣 **Header range validation** — section offset+size checked against file length before table parse
- 📣 **`Error::user_message()` / `with_metadata_context()`** — clear hints for Android / Tauri UI and CLI (encryption, truncation, wrong file)

#### Performance (real multi-core, same dump logic)
- ⚡ **Parallel `dump.cs` type dumping** — rayon map over types; merge preserves order
- ⚡ **Parallel method disassembly** — function-boundary map-reduce of method bodies inside large types (≥4 bodies); shared flat+DiffableCs ASM cache
- ⚡ **Parallel DummyDLL** — each image assembly built concurrently
- ⚡ **Parallel `script.json` / MethodInfo / scaffolding entries** — map-reduce with worker executors
- ⚡ **Parallel `il2cpp.h` StructInfo** — non-generic type defs + MethodInfo fragments; array classes map-reduced
- ⚡ **Parallel generics dump sections** — RGCTX, method specs, attributes, strings, vtables, interfaces
- ⚡ **Parallel annotation build for disasm** — legacy usage table + v27 data-section scan
- ⚡ **Cheap disasm context** — field/vtable maps shared via `Arc` (no heavy clone per method)

#### Concurrency foundation
- `SliceReader` / peek APIs for thread-safe binary reads without shared cursor fights
- Thread-safe string cache; field offsets / generic peeks on shared `&Il2Cpp`
- Worker `Il2CppExecutor::new_for_worker` for parallel name caches

### What's new in v6.1

#### CODM resilient initialization
- 🐛 **Fixed CODM disk-based dumps crashing with "Auto mode failed"** — CODM binaries have correct `CodeRegistration`/`MetadataRegistration` addresses, but some internal struct fields (e.g. `generic_method_pointers`, `invoker_pointers`) point to obfuscated/invalid addresses. Previously, any single unmappable pointer in `load_pointers`, `load_types`, or `load_generics` would abort the entire `init()`, causing all fallback strategies to fail even though the addresses were found correctly.
- 🛡️ **CODM-conditional error resilience** — when `codm_diag` / `codm` is active, individual pointer loads now use `match`/skip instead of strict `?`. Bad pointers print a `[WARN]` and are skipped; the dump proceeds with whatever data IS available. Non-CODM games retain strict `?` error propagation so false-positive addresses are caught properly.
- 🔄 **Affected files**: `formats/elf.rs` (`load_pointers`, `load_types`, `load_generics`), `il2cpp/base.rs` (`Il2Cpp::init` pointer loading), `main.rs` (CLI init flow fallback)
- ✅ **All targets synced**: CLI, Tauri desktop/mobile, Android JNI — same engine fix across all platforms

### What's new in v6

#### Static metadata & thread-static
- 🧵 **Thread-static field support** — TLS offset decode (`0x80000000` flag), comment annotations in dump.cs, DummyDll `ThreadStaticAttribute`
- 📦 **FieldRVA / PrivateImplementationDetails** — optional `static_metadata.json`, optional blob export, `FieldRvaAttribute` in DummyDll
- 🍎 **iOS CODM auto-detection** — `resolve_codm()` when metadata variant is CODM (same as Android ELF path)
- 🔍 **Unity 27+ FieldRva metadata usage (case 7)** — `script.json` `FieldRvas` on PE / Mach-O / NSO / WASM via `data_sections` scan
- 📐 **`Il2CppTypeDefinitionSizes` loading** — thread-static block heuristics from `MetadataRegistration` on all formats
- 🛠️ **Memory-dumped CODM fixes** — improved dump-file / image-base handling so memory-captured CODM binaries (Android ELF + iOS Mach-O) init and decrypt types reliably alongside standard dump support
- ⚙️ **Configurable static metadata** — `dumpStaticFieldMetadata` master toggle (off by default for speed); nested `dumpFieldRvaData` (off by default — blobs make JSON huge) + `maxFieldRvaDumpBytes`
- 🖥️ **GUI parity** — Jetpack Compose + Tauri desktop apps ship the same v6 engine and config keys

#### Disassembler cross-platform parity
Full ARM32 + x86/x64 annotation parity with ARM64 (see [`DISASSEMBLER_UPGRADE.md`](DISASSEMBLER_UPGRADE.md)):
- **ARM32 rewrite** — propagation engine now runs on 32-bit Android: `ConstantOp`, `LoadInfo`, `RegRegAccess`, `IndirectCallReg`, CBZ/CBNZ, ADR/ADRL, LDM block loads
- **x86/x64 LEA RIP-relative** — `RIP_PSEUDO_REG` sentinel + `ConstantOp::Adrp` tracking; MOVZX/MOVSX loads; 25+ GP-writing mnemonics emit proper `Kill`
- **Switch table reconstruction** — `detect_switch_dispatches` on x86/x64 PIC (`lea rcx,[rip+table]` → `movsxd` → `add` → `jmp`) and ARM32
- **Static field annotation** — all architectures: `// TypeName.fieldName` from typeinfo + static-fields offset chains (x86 RIP-relative included)
- **Init-check suppression** — x86/x64 `movzx`/`test`/`jne`/`call runtime_class_init` pattern folded to `// [init check]`
- **x86 gap fixes** — LEA base+disp propagation, XADD/MOVBE/BMI2/MUL/DIV RDX:RAX kill coverage

> Deep dives: [`STATIC_FIELD_UPGRADE.md`](STATIC_FIELD_UPGRADE.md) · [`DISASSEMBLER_UPGRADE.md`](DISASSEMBLER_UPGRADE.md)

### What's new in v5.5
- 🔐 Il2CppType byte-stream XOR decryption for CODM (recovers Quaternion/Vector3/Class fields previously shown as `UnknownType(0x27)` or `object`)
- 🍎 Fixed iOS Mach-O chained-fixup edge cases — first-pointer rebasing now works on CODM iOS builds
- 📊 Three-marker detection scheme with <0.1 % false-positive rate on clean entries
- 🧹 Removed stale debug eprintlns, cleaner stderr output

### Search Strategies
- **SectionHelper** — Format-aware section scanning
- **Symbol Search** — ELF/Mach-O symbol table lookup
- **ARM32 Search** — Dedicated binary pattern matching
- **\_\_mod\_init\_func** — Mach-O initializer analysis
- **Manual mode** — Enter addresses as fallback

---

## 📦 Installation

### From crates.io
```bash
cargo install il2cpp_dumper
```

This installs the latest release globally. Run `il2cpp_dumper` from anywhere.

### From Source
```bash
git clone https://github.com/rodroidmods/il2cpp-dumper-rs.git
cd il2cpp-dumper-rs/il2cpp_dumper
cargo build --release
```

The binary will be at `target/release/il2cpp_dumper` (`.exe` on Windows).

### Cross-Compilation
```bash
# Linux
cargo build --release --target x86_64-unknown-linux-gnu

# macOS
cargo build --release --target aarch64-apple-darwin

# Android (via cross)
cross build --release --target aarch64-linux-android
```

---

## 🔧 Usage

```bash
il2cpp_dumper <il2cpp-binary> <global-metadata.dat> [output-dir]
```

### Examples
```bash
il2cpp_dumper libil2cpp.so global-metadata.dat
il2cpp_dumper GameAssembly.dll global-metadata.dat ./output
il2cpp_dumper UnityFramework global-metadata.dat
```

### Example Output
```
  Rodroid Il2CppDumper

    ╦╦  ╔═╗╔═╗╔═╗  ╔╦╗╦ ╦╔╦╗╔═╗╔═╗╦═╗
    ║║  ╠═╝║  ╠═╝   ║║║ ║║║║╠═╝║╣ ╠╦╝
    ╩╩═╝╚  ╚═╝╩    ═╩╝╚═╝╩ ╩╩  ╚═╝╩╚═
    Version v8.5.0
  ─────────────────────────────────────

  📂 Output .\Dump0

  ─── Binary Analysis ───────────────────

  ✓ Binary loaded: libil2cpp.so (52.43 MB)
  Unity Version: 2022.3.62f2
  ✓ Metadata loaded: global-metadata.dat (10.88 MB)
  Metadata Version: 31
  Type Definitions: 13815
  Method Definitions: 93772

  ─── Format Detection ──────────────────

  🔍 Detected ELF64 format
  IL2CPP Version: 31
  CodeRegistration: 0x44e5ff8
  MetadataRegistration: 0x465a328

  ─── Output Generation ─────────────────

  ✓ dump.cs generated
  ✓ script.json, il2cpp.h, stringliteral.json generated
  ✓ Dummy DLL files generated

  ═══════════════════════════════════════
  ✨ All tasks completed successfully!
  ═══════════════════════════════════════

  📂 Output Directory: .\Dump0
  📦 Generated Files: dump.cs, script.json, il2cpp.h, stringliteral.json, DummyDll/*.dll
  🚀 Elapsed: 8.35s
  ───────────────────────────────────────
```

---

## ⚙️ Configuration

Place `config.json` next to the binary (or pass `--config path/to/config.json`). All keys use **camelCase** (see bundled [`config.json`](config.json)).

### CLI flags (override config)
| Flag | Effect |
|------|--------|
| `--codm` | Force CODM metadata layout + reloc/fixup paths |
| `--force-dump` | Treat input as a memory dump; prompt for image base |
| `--dump-static-metadata` | Enable thread-static / FieldRVA export pipeline |
| `--no-dump-static-metadata` | Disable static metadata export |

### Key groups

| Group | Keys |
|-------|------|
| **Output** | `dumpMethod`, `dumpField`, `dumpProperty`, `dumpAttribute`, `dumpMethodOffset`, `dumpFieldOffset`, `dumpTypeDefIndex`, `dumpAssemblyName`, `splitDumpPerType` |
| **Generation** | `generateStruct`, `generateDummyDll`, `dummyDllAddToken`, `requireAnyKey` |
| **Generics** | `generateGenericsDump`, `dumpGenericsRgctx`, `dumpGenericsMethodSpecs`, `dumpGenericsCustomAttributes`, `dumpGenericsStringLiterals`, `dumpGenericsMetadataUsages`, `dumpGenericsVtables`, `dumpGenericsInterfaces` |
| **Disassembly** | `dumpDisassembly`, `dumpDisassemblyTarget` (0=Both, 1=dump.cs, 2=DiffableCs), `dumpDisassemblyHexBytes`, `dumpDisassemblyFieldNames`, `dumpDisassemblyAnnotations`, `dumpDisassemblyCfg`, `maxDisassemblyInstructions` |
| **C++ headers** | `generateCppScaffold`, `mangleNames`, `enhancedIdaMetadata`, `generateUnityHeaders`, `compilerLayout` (`GCC`/`MSVC`), `useTopologicalSort` |
| **Static metadata (v6)** | `dumpStaticFieldMetadata` (default `false`), `dumpFieldRvaData`, `maxFieldRvaDumpBytes` |
| **C# decompiler (v8, pre-alpha)** | `csharpTarget` (0=Both, 1=dump.cs, 2=Split), `csharpInteractive`, `csharpMaxBodySize` (0 = no limit), `csharpIncludeAssemblies`, `csharpIncludeTypes`, **`csharpFullDump`** (v8.5: keep every type when a filter is active, bodies only for the selection) |
| **Advanced** | `forceIl2cppVersion`, `forceVersion`, `forceDump`, `noRedirectedPointer`, `codm` |

### v6 static metadata example
```json
{
  "dumpStaticFieldMetadata": true,
  "dumpFieldRvaData": false,
  "maxFieldRvaDumpBytes": 4096
}
```
When enabled, outputs `static_metadata.json` and enriches dump.cs / DummyDll / script.json. Set `dumpFieldRvaData` to `true` only when you need base64 hex blobs in JSON (can be tens of MB on large games). Leave the master toggle **off** for faster dumps when you only need `dump.cs`.

---

## 🏗️ Architecture

```
il2cpp_dumper/src/
├── main.rs                           # CLI, format detection, orchestration
├── config.rs                         # Configuration handling
├── formats/                          # Binary format parsers
│   ├── elf.rs                        # ELF32/64 (Android, Linux)
│   ├── pe.rs                         # PE32/64 (Windows)
│   ├── macho.rs                      # Mach-O + Fat Mach-O (iOS, macOS)
│   ├── nso.rs                        # NSO (Nintendo Switch)
│   └── wasm.rs                       # WebAssembly (WebGL)
├── il2cpp/                           # IL2CPP structures and metadata
│   ├── base.rs                       # Il2Cpp main struct, type_definition_sizes
│   ├── field_layout.rs               # Thread-static / FieldRVA offset decode (v6)
│   ├── metadata.rs                   # Metadata parser (+ CODM variant)
│   └── structures.rs                 # IL2CPP type definitions
├── search/                           # Registration search algorithms
│   └── section_helper.rs
├── executor/                         # IL2CPP type resolution engine
├── disassembler/                     # Multi-arch disassembly engine
│   ├── mod.rs                        # CFG analysis, annotations, formatting
│   ├── arm.rs                        # ARM64/ARM32 decoder (yaxpeax)
│   └── x86.rs                        # x86/x64 decoder (iced-x86)
└── output/                           # Output generators
    ├── decompiler.rs                 # dump.cs + inline disassembly
    ├── static_field_exporter.rs      # static_metadata.json + field annotations (v6)
    ├── struct_generator.rs           # script.json, il2cpp.h, v27 FieldRva scan
    ├── dummy_assembly_generator.rs   # DummyDLL (parallel)
    ├── cpp_scaffolding.rs            # il2cpp-functions.h generation
    ├── cpp_ast.rs                    # C++ AST emission with group annotations
    ├── cpp_type_model.rs             # C++ type model from IL2CPP types
    ├── cpp_type_dependency_graph.rs  # Topological sort + cycle detection
    ├── name_mangler.rs               # Itanium ABI C++ name mangling
    ├── header_manager.rs             # Unity header version matching
    └── unity_version.rs              # Unity version parsing & ranges
```

---

## 📜 License

MIT

---

## 🙏 Credits

- [Perfare/Il2CppDumper](https://github.com/Perfare/Il2CppDumper) — Original C# implementation
- [SamboyCoding/Cpp2IL](https://github.com/SamboyCoding/Cpp2IL) — Advanced C# IL2CPP analysis tool
- [springmusk026/Il2CppDumper-Python](https://github.com/springmusk026/Il2CppDumper-Python) — Python port
- [LukeFZ/Il2CppInspectorRedux](https://github.com/LukeFZ/Il2CppInspectorRedux) — Thanks for the code i used in v4 of my il2cppdumper, but its more faster since the logic its on rust and not C#
- [dotnetdll](https://crates.io/crates/dotnetdll) — .NET DLL generation crate
- [rayon](https://crates.io/crates/rayon) — Parallel processing
- [console-rs](https://github.com/console-rs) — Terminal styling ecosystem (`console`, `indicatif`, `dialoguer`)

---

## 📬 Community

| | Link |
|---|---|
| 📢 **Telegram Channel** | [Join Channel](https://t.me/+WmudnO0-xoNhMDQ8) |
| 💬 **Telegram Group** | [Join Group](https://t.me/+QylrYL1GNsJiYjc0) |
| 👤 **Developer** | [`@rodroidmods`](https://t.me/rodroidmods) |

---

> **⚠️ Disclaimer**: This tool is for educational and research purposes only. Respect game developers' rights and terms of service.
