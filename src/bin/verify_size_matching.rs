use il2cpp_dumper::formats::elf::Elf;
use il2cpp_dumper::il2cpp::base::Il2Cpp;
use il2cpp_dumper::il2cpp::metadata::Metadata;
use std::collections::{HashSet, BTreeMap};
use std::fs;

fn main() {
    let libil2cpp_path = "/workspaces/il2cppdumper-continuation/libs/android/temple-run-2/libil2cpp.so";
    let metadata_path = "/workspaces/il2cppdumper-continuation/libs/android/temple-run-2/global-metadata.dat";
    
    eprintln!("[*] Reading files...");
    let elf_data = fs::read(libil2cpp_path).expect("Failed to read libil2cpp.so");
    let metadata_data = fs::read(metadata_path).expect("Failed to read metadata");
    
    eprintln!("[*] Parsing ELF...");
    let is_64 = elf_data.len() > 4 && elf_data[4] == 2;
    let elf = Elf::new(elf_data, !is_64).expect("Failed to parse ELF");
    let il2cpp = Il2Cpp::from_elf(&elf);
    
    eprintln!("[*] Parsing metadata...");
    let metadata = Metadata::new(metadata_data).expect("Failed to parse metadata");
    
    eprintln!("[*] Type definition sizes: {}", il2cpp.type_definition_sizes.len());
    eprintln!("[*] Type defs: {}", metadata.type_defs.len());
    
    // Collect class information
    struct ClassInfo {
        tdi: i32,
        name: String,
        namespace: String,
        static_fields_size: u32,
        static_field_count: u32,
    }
    
    let mut classes = Vec::new();
    
    for (tdi, td) in metadata.type_defs.iter().enumerate() {
        let type_name = metadata.get_string_from_index(td.name_index).unwrap_or_default();
        let namespace = metadata.get_string_from_index(td.namespace_index).unwrap_or_default();
        
        // Count static fields
        let field_start = td.field_start as usize;
        let field_count = td.field_count as usize;
        
        let mut static_field_count = 0u32;
        
        for fi in 0..field_count {
            let global_fi = field_start + fi;
            if let Some(field_def) = metadata.field_defs.get(global_fi) {
                if let Some(field_il2cpp_type) = il2cpp.types.get(field_def.type_index as usize) {
                    let is_static = (field_il2cpp_type.attrs & 0x0010) != 0;
                    if is_static {
                        static_field_count += 1;
                    }
                }
            }
        }
        
        // Get static_fields_size
        let static_fields_size = il2cpp.type_definition_sizes.get(tdi)
            .map(|s| s.static_fields_size)
            .unwrap_or(0);
        
        if static_fields_size > 0 && static_field_count > 0 {
            classes.push(ClassInfo {
                tdi: tdi as i32,
                name: type_name,
                namespace,
                static_fields_size,
                static_field_count,
            });
        }
    }
    
    eprintln!("[*] Found {} classes with static fields", classes.len());
    
    // Analyze size collisions
    let mut size_to_classes: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (idx, class) in classes.iter().enumerate() {
        size_to_classes.entry(class.static_fields_size)
            .or_insert_with(Vec::new)
            .push(idx);
    }
    
    println!("SIZE_COLLISION_ANALYSIS");
    println!("total_classes:{}", classes.len());
    println!("unique_sizes:{}", size_to_classes.len());
    println!("collision_ratio:{:.2}", classes.len() as f64 / size_to_classes.len() as f64);
    
    // Size distribution
    let sizes: Vec<u32> = classes.iter().map(|c| c.static_fields_size).collect();
    let min_size = sizes.iter().min().unwrap_or(&0);
    let max_size = sizes.iter().max().unwrap_or(&0);
    let avg_size = sizes.iter().sum::<u32>() as f64 / sizes.len() as f64;
    
    println!("min_size:{}", min_size);
    println!("max_size:{}", max_size);
    println!("avg_size:{:.1}", avg_size);
    
    // Collision details
    let collisions: Vec<_> = size_to_classes.iter()
        .filter(|(_, indices)| indices.len() > 1)
        .collect();
    
    println!("sizes_with_collisions:{}", collisions.len());
    
    if !collisions.is_empty() {
        let max_collision_count = collisions.iter().map(|(_, indices)| indices.len()).max().unwrap_or(0);
        println!("max_collision_count:{}", max_collision_count);
        
        println!("TOP_COLLISIONS");
        let mut sorted_collisions = collisions.clone();
        sorted_collisions.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        for (size, indices) in sorted_collisions.iter().take(20) {
            println!("{}:{}", size, indices.len());
        }
    }
    
    // Edge cases
    println!("EDGE_CASES");
    let small = classes.iter().filter(|c| c.static_fields_size < 16).count();
    let large = classes.iter().filter(|c| c.static_fields_size > 1000).count();
    let single_field = classes.iter().filter(|c| c.static_field_count == 1).count();
    let many_fields = classes.iter().filter(|c| c.static_field_count > 20).count();
    
    println!("small_classes_lt_16:{}", small);
    println!("large_classes_gt_1000:{}", large);
    println!("single_field_classes:{}", single_field);
    println!("many_fields_classes_gt_20:{}", many_fields);
    
    // Sample classes for verification
    println!("SAMPLE_CLASSES");
    let sample_size = std::cmp::min(200, classes.len());
    let mut rng = simple_rng(42);
    let mut sampled_indices = Vec::new();
    while sampled_indices.len() < sample_size {
        let idx = (rng() % classes.len() as u64) as usize;
        if !sampled_indices.contains(&idx) {
            sampled_indices.push(idx);
        }
    }
    
    for &idx in &sampled_indices {
        let class = &classes[idx];
        println!("{}|{}|{}|{}|{}", 
            class.tdi, 
            class.name, 
            class.namespace,
            class.static_fields_size,
            class.static_field_count
        );
    }
    
    // Analyze GOT relocations
    println!("GOT_ANALYSIS");
    println!("total_got_relocations:{}", il2cpp.got_relative_relocations.len());
    
    let mut unique_addends: Vec<u64> = il2cpp.got_relative_relocations.values().copied().collect();
    unique_addends.sort_unstable();
    unique_addends.dedup();
    println!("unique_addends:{}", unique_addends.len());
    
    // Partition into blocks
    let gap_threshold = 0x1000u64;
    let mut blocks: Vec<(u64, u64)> = Vec::new();
    if !unique_addends.is_empty() {
        let mut block_start = unique_addends[0];
        let mut block_end = unique_addends[0];
        
        for &addr in &unique_addends[1..] {
            if addr - block_end > gap_threshold {
                blocks.push((block_start, block_end - block_start + 8));
                block_start = addr;
            }
            block_end = addr;
        }
        blocks.push((block_start, block_end - block_start + 8));
    }
    
    println!("blocks_partitioned:{}", blocks.len());
    
    if !blocks.is_empty() {
        let block_sizes: Vec<u64> = blocks.iter().map(|(_, size)| *size).collect();
        let min_block = block_sizes.iter().min().unwrap_or(&0);
        let max_block = block_sizes.iter().max().unwrap_or(&0);
        let avg_block = block_sizes.iter().sum::<u64>() as f64 / block_sizes.len() as f64;
        
        println!("min_block_size:{}", min_block);
        println!("max_block_size:{}", max_block);
        println!("avg_block_size:{:.1}", avg_block);
    }
    
    // Simulate matching
    println!("MATCHING_SIMULATION");
    
    // Count entries per block
    let mut block_entry_counts: Vec<u32> = Vec::new();
    for &(block_start, block_size) in &blocks {
        let count = unique_addends.iter()
            .filter(|&&addr| addr >= block_start && addr < block_start + block_size)
            .count() as u32;
        block_entry_counts.push(count);
    }
    
    // Create block fingerprints
    struct BlockFP {
        size: u64,
        entries: u32,
    }
    
    let mut block_fps: Vec<BlockFP> = blocks.iter().enumerate()
        .map(|(idx, (_, size))| BlockFP {
            size: *size,
            entries: block_entry_counts[idx],
        })
        .collect();
    
    block_fps.sort_by_key(|b| b.size);
    
    // Greedy matching
    let mut used_classes = HashSet::new();
    let mut matches = 0;
    let mut rejected = 0;
    let mut size_diffs = Vec::new();
    
    for block_fp in &block_fps {
        let mut best_match: Option<(usize, f64)> = None;
        let mut best_score = 0.0f64;
        
        for (class_idx, class) in classes.iter().enumerate() {
            if used_classes.contains(&class_idx) {
                continue;
            }
            
            if block_fp.size > class.static_fields_size as u64 {
                continue;
            }
            
            let size_diff = class.static_fields_size as u64 - block_fp.size;
            let size_score = if size_diff == 0 {
                1.0
            } else {
                1.0 / (1.0 + size_diff as f64 / 64.0)
            };
            
            let count_diff = (class.static_field_count as i32 - block_fp.entries as i32).abs();
            let count_score = if count_diff == 0 {
                1.0
            } else {
                1.0 / (1.0 + count_diff as f64 / 4.0)
            };
            
            let sig_score = 0.5;
            let total_score = 0.6 * size_score + 0.3 * count_score + 0.1 * sig_score;
            
            if total_score > best_score {
                best_score = total_score;
                best_match = Some((class_idx, total_score));
            }
        }
        
        if let Some((class_idx, score)) = best_match {
            if score >= 0.75 {
                used_classes.insert(class_idx);
                matches += 1;
                let size_diff = classes[class_idx].static_fields_size as u64 - block_fp.size;
                size_diffs.push(size_diff);
            } else {
                rejected += 1;
            }
        }
    }
    
    println!("matches:{}", matches);
    println!("rejected:{}", rejected);
    println!("match_rate:{:.1}", matches as f64 / blocks.len() as f64 * 100.0);
    
    if !size_diffs.is_empty() {
        let avg_size_diff = size_diffs.iter().sum::<u64>() as f64 / size_diffs.len() as f64;
        let exact_matches = size_diffs.iter().filter(|&&d| d == 0).count();
        println!("avg_size_diff:{:.1}", avg_size_diff);
        println!("exact_size_matches:{}", exact_matches);
    }
}

fn simple_rng(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state >> 33
    }
}
