use crate::utils::pattern_search::find_bytes;

#[derive(Debug, Clone)]
pub struct SearchSection {
    pub offset: u64,
    pub offset_end: u64,
    pub address: u64,
    pub address_end: u64,
}

impl SearchSection {
    pub fn new(offset: u64, offset_end: u64, address: u64, address_end: u64) -> Self {
        Self {
            offset,
            offset_end,
            address,
            address_end,
        }
    }
}

pub struct SectionHelper<'a> {
    data: &'a [u8],
    is_32bit: bool,
    version: f64,
    is_elf: bool,
    _sections: Vec<SearchSection>,
    data_sections: Vec<SearchSection>,
    code_sections: Vec<SearchSection>,
    bss_sections: Vec<SearchSection>,
    method_count: usize,
    type_definitions_count: usize,
    _metadata_usages_count: usize,
    image_count: usize,
    pointer_in_exec: bool,
}

/// The file offset of the first pointer-aligned *address* inside `section`.
///
/// Alignment is a property of the address space, not of the file. For ELF,
/// Mach-O and PE the two coincide (sections are page-aligned), which is why
/// keying the search on the file offset went unnoticed for so long. In a Wasm
/// build Unity emits **one data segment per static object**, placed at an
/// arbitrary file offset, so the VA of a pointer and its file offset disagree:
/// in a real Unity WebGL build 17,267 of 23,164 data segments start at a
/// non-4-aligned file offset, and a scan that requires the *file* offset to be
/// aligned therefore never sees three quarters of the pointers — including
/// every `Il2CppCodeGenModule` and both registration structs.
fn first_aligned_offset(section: &SearchSection, ptr_size: usize) -> usize {
    let skew = (section.address % ptr_size as u64) as usize;
    (section.offset as usize).saturating_add((ptr_size - skew) % ptr_size)
}

impl<'a> SectionHelper<'a> {
    pub fn new(
        data: &'a [u8],
        is_32bit: bool,
        version: f64,
        is_elf: bool,
        sections: Vec<SearchSection>,
        data_sections: Vec<SearchSection>,
        code_sections: Vec<SearchSection>,
        bss_sections: Vec<SearchSection>,
        method_count: usize,
        type_definitions_count: usize,
        metadata_usages_count: usize,
        image_count: usize,
    ) -> Self {
        Self {
            data,
            is_32bit,
            version,
            is_elf,
            _sections: sections,
            data_sections,
            code_sections,
            bss_sections,
            method_count,
            type_definitions_count,
            _metadata_usages_count: metadata_usages_count,
            image_count,
            pointer_in_exec: false,
        }
    }

    fn ptr_size(&self) -> usize {
        if self.is_32bit { 4 } else { 8 }
    }

    /// Visit every pointer-sized word stored at a pointer-aligned *address*.
    ///
    /// `f` receives `(file_offset, va)` and returns `true` to stop the walk.
    /// This is the only correct way to enumerate pointers: see
    /// [`first_aligned_offset`] for why the file offset must not be used.
    fn visit_aligned_pointers<F>(&self, sections: &[SearchSection], mut f: F) -> bool
    where
        F: FnMut(usize, u64) -> bool,
    {
        let ptr_size = self.ptr_size();
        for section in sections {
            let end = (section.offset_end as usize).min(self.data.len());
            let mut offset = first_aligned_offset(section, ptr_size);
            while offset + ptr_size <= end {
                let va = section.address + (offset - section.offset as usize) as u64;
                if f(offset, va) {
                    return true;
                }
                offset += ptr_size;
            }
        }
        false
    }

    pub fn find_code_registration(&mut self) -> Option<u64> {
        if self.version >= 24.2 {
            if self.is_elf {
                let result = self.find_code_registration_2019(&true);
                if result.is_some() {
                    self.pointer_in_exec = true;
                    return result;
                }
                return self.find_code_registration_2019(&false);
            } else {
                let result = self.find_code_registration_2019(&false);
                if result.is_some() {
                    return result;
                }
                let result = self.find_code_registration_2019(&true);
                if result.is_some() {
                    self.pointer_in_exec = true;
                    return result;
                }
                return None;
            }
        }
        self.find_code_registration_old()
    }

    pub fn find_metadata_registration(&self) -> Option<u64> {
        if self.version < 19.0 {
            return None;
        }
        if self.version >= 27.0 {
            return self.find_metadata_registration_v21();
        }
        self.find_metadata_registration_old()
    }

    fn offset_to_va(&self, offset: usize) -> Option<u64> {
        for section in &self.data_sections {
            if offset as u64 >= section.offset && (offset as u64) < section.offset_end {
                return Some(offset as u64 - section.offset + section.address);
            }
        }
        None
    }

    fn find_refs_fast(&self, addr: u64) -> Vec<(usize, u64)> {
        let target = if self.is_32bit {
            (addr as u32) as u64
        } else {
            addr
        };
        let mut refs = Vec::new();
        self.visit_aligned_pointers(&self.data_sections, |offset, va| {
            if self.read_ptr_at(offset) == Some(target) {
                refs.push((offset, va));
            }
            false
        });
        refs
    }

    fn search_bytes_iter(data: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut results = Vec::new();
        let mut start = 0;
        while let Some(idx) = find_bytes(&data[start..], pattern) {
            results.push(start + idx);
            start = start + idx + 1;
        }
        results
    }

    /// The index of `addrCodeGenModulePtrs` in `Il2CppCodeRegistration`, which is
    /// how far back the 2019+ locator has to walk from the pointer it found.
    ///
    /// The field is last, so this is the struct's size in slots minus one — and the
    /// struct's size changes with the version (the v24.5 generic-adjustor pointer,
    /// the v29.1 indirect-call pair, ...). Hard-coding it left the located address
    /// short of the struct on some versions, which then only worked because the
    /// caller guessed a shift.
    fn code_registration_module_field_index(&self) -> u64 {
        crate::il2cpp::structures::Il2CppCodeRegistration::field_count(self.version) as u64 - 1
    }

    fn find_code_registration_2019(&self, use_exec: &bool) -> Option<u64> {
        let result = self.find_code_registration_2019_with_feature(use_exec, b"mscorlib.dll\x00");
        if result.is_some() {
            return result;
        }
        self.find_code_registration_2019_with_feature(use_exec, b"System.Private.CoreLib.dll\x00")
    }

    fn find_code_registration_2019_with_feature(
        &self,
        use_exec: &bool,
        feature_bytes: &[u8],
    ) -> Option<u64> {
        let ptr_size = self.ptr_size();

        let sections = if *use_exec {
            &self.code_sections
        } else {
            &self.data_sections
        };

        for section in sections {
            let start = section.offset as usize;
            let end = section.offset_end as usize;
            if start >= self.data.len() || end > self.data.len() {
                continue;
            }
            let section_data = &self.data[start..end];

            for index in Self::search_bytes_iter(section_data, feature_bytes) {
                let dll_va = index as u64 + section.address;

                let refs1 = self.find_refs_fast(dll_va);

                for (_, ref_va) in &refs1 {
                    let refs2 = self.find_refs_fast(*ref_va);

                    for (_ref_offset2, ref_va2) in &refs2 {
                        if self.version >= 27.0 {
                            let min_target =
                                ref_va2 - (self.image_count as u64 - 1) * ptr_size as u64;
                            let max_target = *ref_va2;

                            // `codeGenModulesCount` followed by a pointer into
                            // the module array. Both words sit at aligned
                            // addresses, which is *not* the same as aligned file
                            // offsets (see `first_aligned_offset`).
                            let count_value = self.image_count as u64;
                            let mut found: Option<u64> = None;
                            self.visit_aligned_pointers(&self.data_sections, |offset, _va| {
                                if self.read_ptr_at(offset) != Some(count_value) {
                                    return false;
                                }
                                let next_offset = offset + ptr_size;
                                let Some(pv) = self.read_ptr_at(next_offset) else {
                                    return false;
                                };
                                if pv < min_target || pv > max_target {
                                    return false;
                                }
                                let i = (ref_va2 - pv) / ptr_size as u64;
                                if i >= self.image_count as u64
                                    || pv != ref_va2 - i * ptr_size as u64
                                {
                                    return false;
                                }
                                if let Some(ref_va3) = self.offset_to_va(next_offset) {
                                    // `ref_va3` is the struct's `addrCodeGenModulePtrs`
                                    // field, which is the last one, so the struct
                                    // starts exactly its own size minus one slot back.
                                    found = Some(
                                        ref_va3
                                            - ptr_size as u64
                                                * self.code_registration_module_field_index(),
                                    );
                                    return true;
                                }
                                false
                            });
                            if let Some(cr) = found {
                                return Some(cr);
                            }
                        } else {
                            for i in 0..self.image_count {
                                let target = ref_va2 - (i as u64) * ptr_size as u64;
                                let refs3 = self.find_refs_fast(target);
                                for (_, ref_va3) in &refs3 {
                                    // Same walk-back as the v27+ branch above; the
                                    // hard-coded 13 here was four slots short on
                                    // v24.5, where `addrCodeGenModulePtrs` is slot 16.
                                    return Some(
                                        ref_va3
                                            - ptr_size as u64
                                                * self.code_registration_module_field_index(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn find_code_registration_old(&self) -> Option<u64> {
        let count_bytes = if self.is_32bit {
            (self.method_count as u32).to_le_bytes().to_vec()
        } else {
            (self.method_count as u64).to_le_bytes().to_vec()
        };

        for section in &self.data_sections {
            let start = section.offset as usize;
            let end = section.offset_end as usize;
            if start >= self.data.len() || end > self.data.len() {
                continue;
            }
            let slice = &self.data[start..end];
            let mut search_start = 0;

            while let Some(pos) = find_bytes(&slice[search_start..], &count_bytes) {
                let abs_pos = start + search_start + pos;
                let ptr_size = self.ptr_size();
                let addr = section.address + (search_start + pos) as u64;

                if self.is_valid_code_registration(addr, abs_pos, ptr_size) {
                    return Some(addr);
                }
                search_start += pos + 1;
            }
        }
        None
    }

    fn find_metadata_registration_old(&self) -> Option<u64> {
        let count_bytes = if self.is_32bit {
            (self.type_definitions_count as u32).to_le_bytes().to_vec()
        } else {
            (self.type_definitions_count as u64).to_le_bytes().to_vec()
        };

        for section in &self.data_sections {
            let start = section.offset as usize;
            let end = section.offset_end as usize;
            if start >= self.data.len() || end > self.data.len() {
                continue;
            }
            let slice = &self.data[start..end];
            let mut search_start = 0;

            while let Some(pos) = find_bytes(&slice[search_start..], &count_bytes) {
                let abs_pos = start + search_start + pos;
                let ptr_size = self.ptr_size();
                let addr = section.address + (search_start + pos) as u64;

                if self.is_valid_metadata_registration(addr, abs_pos, ptr_size) {
                    return Some(self.metadata_registration_base(addr, abs_pos, ptr_size));
                }
                search_start += pos + 1;
            }
        }
        None
    }

    /// Walk back from a count word to the first field of the registration.
    ///
    /// The scan keys on the type-definition count, and *two* counts in the struct
    /// equal it: `fieldOffsetsCount` (slot 10) and `typeDefinitionsSizesCount`
    /// (slot 12). Which one the scan landed on can be read off the neighbouring
    /// words, because they are adjacent. This matters in practice: a
    /// `fieldOffsets` entry is null when the field has no offset, so the pointer
    /// sample in `is_valid_metadata_registration` usually rejects slot 10 and the
    /// scan returns slot 12 — handing back an address twelve slots past the start
    /// of the struct, where `genericClassesCount` reads as a pointer and the whole
    /// registration looks like garbage.
    fn metadata_registration_base(&self, addr: u64, abs_pos: usize, ptr_size: usize) -> u64 {
        let type_count = self.type_definitions_count as u64;
        let index = if self.read_ptr_at(abs_pos + ptr_size * 2) == Some(type_count) {
            // Followed by the type-definitions-sizes count: this is fieldOffsetsCount.
            10
        } else if abs_pos >= ptr_size * 2
            && self.read_ptr_at(abs_pos - ptr_size * 2) == Some(type_count)
        {
            // Preceded by the field-offsets count: this is typeDefinitionsSizesCount.
            12
        } else {
            // Neither neighbour is the pair, so the word is not one of the two
            // count/pointer pairs the layout guarantees. Leave the address alone
            // rather than shifting it on a guess.
            0
        };
        addr - ptr_size as u64 * index as u64
    }

    fn find_metadata_registration_v21(&self) -> Option<u64> {
        let ptr_size = self.ptr_size();
        let type_count = self.type_definitions_count;

        // The count pair is looked up at pointer-aligned *addresses*: a Wasm
        // data segment can start at a non-aligned file offset, so scanning for
        // aligned file offsets would never see most of the image.
        let mut found: Option<u64> = None;
        self.visit_aligned_pointers(&self.data_sections, |offset, va| {
            if self.read_ptr_at(offset) != Some(type_count as u64) {
                return false;
            }
            let second_offset = offset + 2 * ptr_size;
            if self.read_ptr_at(second_offset) != Some(type_count as u64) {
                return false;
            }
            let Some(pointer_va) = self.read_ptr_at(offset + 3 * ptr_size) else {
                return false;
            };
            let Some(pointer_offset) = self.va_to_offset_data(pointer_va) else {
                return false;
            };
            let sample_size = std::cmp::min(10, type_count);
            for i in 0..sample_size {
                let sample_offset = pointer_offset + i * ptr_size;
                let Some(ptr_val) = self.read_ptr_at(sample_offset) else {
                    return false;
                };
                let in_range = if self.pointer_in_exec {
                    self.is_in_code_sections(ptr_val)
                } else {
                    self.is_in_data_sections(ptr_val)
                };
                if !in_range {
                    return false;
                }
            }
            // The two equal counts are `fieldOffsetsCount` and
            // `typeDefinitionsSizesCount` — both are the type-definition count —
            // so `va` is the address of `fieldOffsetsCount` and the struct starts
            // that many slots before it. It sits ten slots in up to v107 (three
            // count/pointer pairs precede it), but v108 removed the
            // `genericMethodTable` pair and the `methodSpecs` pair — four slots —
            // so on v108+ the pair is at index 6. Getting this wrong shifts the
            // whole registration four slots and reads a pointer as the first
            // count. (Cpp2IL does the same arithmetic from the other end:
            // `typeDefinitionsSizesCount` is 4 fields from the end, or 6 on
            // v106.1+, of a struct whose size is version-dependent.)
            let field_offsets_index = if self.version >= 108.0 { 6 } else { 10 };
            found = Some(va - ptr_size as u64 * field_offsets_index);
            true
        });
        found
    }

    fn va_to_offset_data(&self, va: u64) -> Option<usize> {
        for sec in &self.data_sections {
            if va >= sec.address && va < sec.address_end {
                return Some((va - sec.address + sec.offset) as usize);
            }
        }
        None
    }

    fn is_valid_code_registration(&self, _addr: u64, offset: usize, ptr_size: usize) -> bool {
        if offset + ptr_size > self.data.len() {
            return false;
        }
        let pointer_va = match self.read_ptr_at(offset + ptr_size) {
            Some(p) if p > 0 => p,
            _ => return false,
        };
        let pointer_offset = match self.va_to_offset_data(pointer_va) {
            Some(o) => o,
            None => return self.is_in_code_sections(pointer_va),
        };
        let check_count = std::cmp::min(self.method_count, 100);
        for i in 0..check_count {
            let entry_offset = pointer_offset + i * ptr_size;
            if entry_offset + ptr_size > self.data.len() {
                return false;
            }
            match self.read_ptr_at(entry_offset) {
                Some(ptr) if self.is_in_code_sections(ptr) => {}
                _ => return false,
            }
        }
        true
    }

    fn is_valid_metadata_registration(&self, _addr: u64, offset: usize, ptr_size: usize) -> bool {
        if offset + ptr_size > self.data.len() {
            return false;
        }
        let pointer_va = match self.read_ptr_at(offset + ptr_size) {
            Some(p) if p > 0 => p,
            _ => return false,
        };
        let pointer_offset = match self.va_to_offset_data(pointer_va) {
            Some(o) => o,
            None => return self.is_in_bss_sections(pointer_va),
        };
        let check_count = std::cmp::min(self.type_definitions_count, 100);
        for i in 0..check_count {
            let entry_offset = pointer_offset + i * ptr_size;
            if entry_offset + ptr_size > self.data.len() {
                return false;
            }
            match self.read_ptr_at(entry_offset) {
                Some(ptr) if self.is_in_data_sections(ptr) || self.is_in_bss_sections(ptr) => {}
                _ => return false,
            }
        }
        true
    }

    fn is_in_data_sections(&self, addr: u64) -> bool {
        self.data_sections
            .iter()
            .any(|s| addr >= s.address && addr <= s.address_end)
    }

    fn is_in_code_sections(&self, addr: u64) -> bool {
        self.code_sections
            .iter()
            .any(|s| addr >= s.address && addr <= s.address_end)
    }

    fn is_in_bss_sections(&self, addr: u64) -> bool {
        self.bss_sections
            .iter()
            .any(|s| addr >= s.address && addr <= s.address_end)
    }

    fn _addr_to_offset(&self, addr: u64) -> Option<u64> {
        for section in &self._sections {
            if addr >= section.address && addr < section.address_end {
                return Some(section.offset + (addr - section.address));
            }
        }
        None
    }

    fn read_ptr_at(&self, offset: usize) -> Option<u64> {
        if self.is_32bit {
            if offset + 4 > self.data.len() {
                return None;
            }
            Some(u32::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
            ]) as u64)
        } else {
            if offset + 8 > self.data.len() {
                return None;
            }
            Some(u64::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
                self.data[offset + 4],
                self.data[offset + 5],
                self.data[offset + 6],
                self.data[offset + 7],
            ]))
        }
    }
}

impl<'a> SectionHelper<'a> {
    pub fn find_metadata_registration_codm(&self) -> Option<u64> {
        let ptr_size = self.ptr_size();

        let count_bytes = if self.is_32bit {
            (self.type_definitions_count as u32).to_le_bytes().to_vec()
        } else {
            (self.type_definitions_count as u64).to_le_bytes().to_vec()
        };

        let mut fallback: Option<u64> = None;

        for section in &self.data_sections {
            let start = section.offset as usize;
            let end = section.offset_end as usize;
            if start >= self.data.len() || end > self.data.len() {
                continue;
            }
            let slice = &self.data[start..end];
            let mut search_start = 0usize;

            while let Some(pos) = find_bytes(&slice[search_start..], &count_bytes) {
                let abs_pos = start + search_start + pos;
                search_start = (search_start + pos) + 1;

                if (section.address + (abs_pos as u64 - section.offset)) % ptr_size as u64 != 0 {
                    continue;
                }
                // The struct starts `tds_count_index` slots before this word, and
                // v108's registration is four slots shorter than v107's.
                let tds_count_index = if self.version >= 108.0 {
                    8usize
                } else {
                    12usize
                };
                if abs_pos < ptr_size * tds_count_index {
                    continue;
                }
                if abs_pos + ptr_size * 4 > self.data.len() {
                    continue;
                }

                let prev_count_offset = abs_pos.wrapping_sub(ptr_size * 2);
                let prev_count = match self.read_ptr_at(prev_count_offset) {
                    Some(v) => v,
                    None => continue,
                };
                if prev_count != self.type_definitions_count as u64 {
                    continue;
                }

                let prev_ptr_offset = abs_pos - ptr_size;
                let prev_ptr = match self.read_ptr_at(prev_ptr_offset) {
                    Some(v) if v > 0 => v,
                    _ => continue,
                };
                if !self.is_in_data_sections(prev_ptr) && !self.is_in_bss_sections(prev_ptr) {
                    continue;
                }

                let cur_ptr_offset = abs_pos + ptr_size;
                let cur_ptr = match self.read_ptr_at(cur_ptr_offset) {
                    Some(v) if v > 0 => v,
                    _ => continue,
                };
                if !self.is_in_data_sections(cur_ptr) && !self.is_in_bss_sections(cur_ptr) {
                    continue;
                }

                let mu_count_offset = abs_pos + ptr_size * 2;
                let mu_count_val = match self.read_ptr_at(mu_count_offset) {
                    Some(v) => v as usize,
                    None => continue,
                };
                if mu_count_val == 0 || mu_count_val > 10_000_000 {
                    continue;
                }

                let mu_ptr_offset = abs_pos + ptr_size * 3;
                let mu_va = match self.read_ptr_at(mu_ptr_offset) {
                    Some(v) if v > 0 => v,
                    _ => continue,
                };
                if !self.is_in_data_sections(mu_va) && !self.is_in_bss_sections(mu_va) {
                    continue;
                }

                // `abs_pos` is `typeDefinitionsSizesCount`, which is slot 12 of the
                // registration up to v107 and slot 8 from v108 on, because v108
                // dropped the `genericMethodTable` and `methodSpecs` pairs. Using
                // the v107 index on a v108 image puts the base four slots early.
                let addr_va = section.address + (abs_pos as u64 - section.offset);
                let mr_base = addr_va - (ptr_size as u64) * tds_count_index as u64;

                if self.is_codm_metadata_usages_pointer(mu_va, mu_count_val, ptr_size) {
                    return Some(mr_base);
                }

                if fallback.is_none() {
                    fallback = Some(mr_base);
                }
            }
        }
        fallback
    }

    fn is_codm_metadata_usages_pointer(
        &self,
        mu_va: u64,
        mu_count: usize,
        ptr_size: usize,
    ) -> bool {
        let mu_offset = match self.va_to_offset_data(mu_va) {
            Some(o) => o,
            None => return self.is_in_bss_sections(mu_va),
        };
        let check_count = std::cmp::min(mu_count, 100);
        let mut valid = 0usize;
        for i in 0..check_count {
            let entry_offset = mu_offset + i * ptr_size;
            if entry_offset + ptr_size > self.data.len() {
                return false;
            }
            match self.read_ptr_at(entry_offset) {
                Some(ptr) if ptr == 0 => {}
                Some(ptr)
                    if self.is_in_bss_sections(ptr)
                        || self.is_in_data_sections(ptr)
                        || self.is_in_code_sections(ptr) =>
                {
                    valid += 1;
                }
                _ => return false,
            }
        }
        valid * 2 >= check_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PTR: usize = 8;
    const SECTION_ADDR: u64 = 0x1000;

    /// `Il2CppMetadataRegistration` as the file lays it out, in pointer-sized
    /// slots. v108 removed the `genericMethodTable` pair and the `methodSpecs`
    /// pair; `type_count` is both `fieldOffsetsCount` and
    /// `typeDefinitionsSizesCount`, which is the pair the locator keys on.
    fn metadata_registration_words(version: f64, type_count: u64, a_table: u64) -> Vec<u64> {
        let mut w = vec![
            1, a_table, // genericClasses
            1, a_table, // genericInsts
        ];
        if version < 108.0 {
            w.extend_from_slice(&[1, a_table]); // genericMethodTable
        }
        w.extend_from_slice(&[100, a_table]); // numTypes, types
        if version < 108.0 {
            w.extend_from_slice(&[1, a_table]); // methodSpecs
        }
        w.extend_from_slice(&[
            type_count, a_table, // fieldOffsets
            type_count, a_table, // typeDefinitionsSizes
            0, 0, // metadataUsages (always null on v27+)
            1, a_table, // alwaysInitMetadataUsages
        ]);
        w
    }

    const MR_OFFSET: usize = 0x400;
    const MR_ADDR: u64 = SECTION_ADDR + MR_OFFSET as u64;

    /// Build an image whose only registration sits at a known address, and return
    /// the address the locator reports. `typeDefinitionsSizes` is an array of
    /// pointers to separate `Il2CppTypeDefinitionSizes` structs, which is what the
    /// locator's pointer sample walks.
    fn located_registration(version: f64, type_count: u64) -> Option<u64> {
        let mut data = vec![0u8; 0x800];
        let tds_addr = SECTION_ADDR + 0x200;
        for i in 0..type_count as usize {
            let ptr = (SECTION_ADDR + 0x300 + (i * 16) as u64).to_le_bytes();
            data[0x200 + i * PTR..0x200 + i * PTR + PTR].copy_from_slice(&ptr);
        }
        for (i, w) in metadata_registration_words(version, type_count, tds_addr)
            .iter()
            .enumerate()
        {
            let at = MR_OFFSET + i * PTR;
            data[at..at + PTR].copy_from_slice(&w.to_le_bytes());
        }

        let section = SearchSection::new(
            0,
            data.len() as u64,
            SECTION_ADDR,
            SECTION_ADDR + data.len() as u64,
        );
        let helper = SectionHelper::new(
            &data,
            false,
            version,
            false,
            vec![section.clone()],
            vec![section.clone()],
            vec![section.clone()],
            vec![section],
            0,
            type_count as usize,
            0,
            0,
        );
        helper.find_metadata_registration()
    }

    /// v108's registration is four slots shorter than v107's (`genericMethodTable`
    /// and `methodSpecs` moved into the metadata file), so the word the locator
    /// keys on — `typeDefinitionsSizesCount`, which equals `fieldOffsetsCount` —
    /// moved from slot 12 to slot 8. The v107 back-off put the returned address
    /// four slots before the struct, where reading a pointer as `genericClassesCount`
    /// fails plausibility and the caller has to guess.
    #[test]
    fn the_v108_metadata_registration_is_located_at_its_first_field() {
        assert_eq!(located_registration(108.0, 5), Some(MR_ADDR));
        // The v107 shape still lands on its own first field.
        assert_eq!(located_registration(106.1, 5), Some(MR_ADDR));
        // And with only two types the pointer sample still validates.
        assert_eq!(located_registration(108.0, 2), Some(MR_ADDR));
    }

    /// `find_code_registration_post_2019` walks back from the word that points at
    /// the code-gen module array — the struct's last field — by the struct's own
    /// size. That size changes with the version (v24.5 adds the generic-adjustor
    /// pointer, v29.1 the indirect-call pair), so the walk-back has to be derived
    /// rather than hard-coded: the hard-coded thirteen was four slots short here.
    #[test]
    fn the_code_registration_is_walked_back_by_its_own_size() {
        use crate::il2cpp::structures::Il2CppCodeRegistration;
        assert_eq!(Il2CppCodeRegistration::field_count(24.5), 17);
        assert_eq!(Il2CppCodeRegistration::field_count(24.2), 14);
        // v31.0 has neither the v29.1 changes nor the 31.1 fix, so its pair is absent.
        assert_eq!(Il2CppCodeRegistration::field_count(31.0), 15);
        assert_eq!(Il2CppCodeRegistration::field_count(31.1), 17);

        const CR_OFF: usize = 0x200;
        const CR_ADDR: u64 = SECTION_ADDR + CR_OFF as u64;
        let mut data = vec![0u8; 0x400];
        data[0x100..0x10d].copy_from_slice(b"mscorlib.dll\0");
        let module_addr = SECTION_ADDR + 0x140;
        let module_entry_addr = SECTION_ADDR + 0x180;
        // The module's `moduleName`, which is what the string search finds a
        // reference to first.
        data[0x140..0x148].copy_from_slice(&(SECTION_ADDR + 0x100).to_le_bytes());
        // The codegen-module array's entry for it.
        data[0x180..0x188].copy_from_slice(&module_addr.to_le_bytes());
        // The registration's `addrCodeGenModulePtrs`, the struct's last field.
        let last_field = CR_OFF + (Il2CppCodeRegistration::field_count(24.5) - 1) * PTR;
        data[last_field..last_field + PTR].copy_from_slice(&module_entry_addr.to_le_bytes());

        let section = SearchSection::new(
            0,
            data.len() as u64,
            SECTION_ADDR,
            SECTION_ADDR + data.len() as u64,
        );
        let mut helper = SectionHelper::new(
            &data,
            false,
            24.5,
            false,
            vec![section.clone()],
            vec![section.clone()],
            vec![section.clone()],
            vec![section],
            0,
            0,
            0,
            1,
        );
        assert_eq!(helper.find_code_registration(), Some(CR_ADDR));
    }

    /// The pre-v27 path scans for the type-definition count and validates whatever
    /// pointer follows it. Both `fieldOffsetsCount` and `typeDefinitionsSizesCount`
    /// equal that count, and a real `fieldOffsets` array contains nulls for fields
    /// without an offset, so the scan normally lands on `typeDefinitionsSizesCount`
    /// — twelve slots into the struct. The value it returns has to be walked back to
    /// the struct's first field, or `genericClassesCount` reads as a pointer.
    #[test]
    fn a_registration_found_through_its_type_definition_sizes_is_walked_back() {
        const TYPE_COUNT: u64 = 200;
        const MR_OFF: usize = 0x1000;
        let mut data = vec![0u8; 0x2000];
        // `typeDefinitionsSizes`: all pointers, so the sample accepts it.
        let tds_addr = SECTION_ADDR + 0x800;
        for i in 0..100usize {
            let ptr = (SECTION_ADDR + 0x1800).to_le_bytes();
            data[0x800 + i * PTR..0x800 + i * PTR + PTR].copy_from_slice(&ptr);
        }
        // `fieldOffsets`: leading null, which is what makes the scan skip slot 10.
        let field_offsets_addr = SECTION_ADDR + 0x100;
        for i in 1..100usize {
            let ptr = (SECTION_ADDR + 0x1800).to_le_bytes();
            data[0x100 + i * PTR..0x100 + i * PTR + PTR].copy_from_slice(&ptr);
        }
        let mut words = vec![
            1u64,
            field_offsets_addr,
            1,
            field_offsets_addr,
            1,
            field_offsets_addr,
            100,
            field_offsets_addr,
            1,
            field_offsets_addr,
        ];
        words.extend_from_slice(&[
            TYPE_COUNT,
            field_offsets_addr,
            TYPE_COUNT,
            tds_addr,
            1,
            field_offsets_addr,
        ]);
        assert_eq!(words.len(), 16); // the pre-v27 registration is 16 slots
        for (i, w) in words.iter().enumerate() {
            let at = MR_OFF + i * PTR;
            data[at..at + PTR].copy_from_slice(&w.to_le_bytes());
        }

        let section = SearchSection::new(
            0,
            data.len() as u64,
            SECTION_ADDR,
            SECTION_ADDR + data.len() as u64,
        );
        let helper = SectionHelper::new(
            &data,
            false,
            24.5,
            false,
            vec![section.clone()],
            vec![section.clone()],
            vec![section.clone()],
            vec![section],
            0,
            TYPE_COUNT as usize,
            0,
            0,
        );
        assert_eq!(
            helper.find_metadata_registration(),
            Some(SECTION_ADDR + MR_OFF as u64)
        );
    }
}
