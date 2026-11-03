//! Dominator tree + dominance frontiers over the recovery CFG.
//!
//! Standard iterative dataflow algorithm — adequate for the method-sized
//! graphs we build (Cpp2IL uses the same shape in `Graphs/DominatorInfo.cs`).

use super::cfg::Cfg;

pub struct DominatorInfo {
    /// Immediate dominator per block (None for entry).
    pub idom: Vec<Option<usize>>,
    /// Reverse-postorder used during computation & SSA renaming.
    pub rpo: Vec<usize>,
    /// Dominance frontiers per block (sorted, deduplicated).
    pub frontiers: Vec<Vec<usize>>,
    /// Children of each block in the dominator tree.
    pub dom_children: Vec<Vec<usize>>,
}

impl DominatorInfo {
    pub fn compute(cfg: &Cfg) -> Self {
        Self::compute_from(cfg, 0)
    }

    /// Same computation but with an explicit entry node. The reversed graph's
    /// root is the VIRTUAL EXIT (`n`), not node 0 — post-dominators are
    /// dominators on the reversed graph, so the DFS/RPO must start there.
    /// Starting at 0 silently produced `ipdom = None` for every block and
    /// branch merging collapsed to the bounded reachability fallback.
    pub fn compute_from(cfg: &Cfg, entry: usize) -> Self {
        let n = cfg.blocks.len();
        let succs: Vec<Vec<usize>> = (0..n).map(|i| cfg.successors[i].clone()).collect();
        let preds: Vec<Vec<usize>> = {
            let mut p = vec![Vec::new(); n];
            for (i, s) in succs.iter().enumerate() {
                for &t in s {
                    p[t].push(i);
                }
            }
            p
        };

        // reverse postorder from the designated entry
        let rpo = Self::reverse_postorder_from(&succs, entry);
        // map block → rpo position
        let mut order_of = vec![usize::MAX; n];
        for (pos, &b) in rpo.iter().enumerate() {
            order_of[b] = pos;
        }

        let entry = entry.min(n.saturating_sub(1));
        let mut idom: Vec<Option<usize>> = vec![None; n];
        let mut processed = vec![false; n];
        processed[entry] = true;

        let intersect = |mut f1: usize, mut f2: usize, idom: &[Option<usize>]| -> usize {
            while f1 != f2 {
                while order_of[f1] > order_of[f2] {
                    f1 = idom[f1].expect("dominator walk before entry");
                }
                while order_of[f2] > order_of[f1] {
                    f2 = idom[f2].expect("dominator walk before entry");
                }
            }
            f1
        };

        let mut changed = true;
        while changed {
            changed = false;
            for &b in &rpo {
                if b == entry {
                    continue;
                }
                // pick first processed predecessor as seed
                let mut new_idom: Option<usize> = None;
                for &p in &preds[b] {
                    if !processed[p] || order_of[p] == usize::MAX {
                        continue;
                    }
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(existing) => intersect(existing, p, &idom),
                    });
                }
                if let Some(nid) = new_idom {
                    if idom[b] != Some(nid) {
                        idom[b] = Some(nid);
                        processed[b] = true;
                        changed = true;
                    }
                }
            }
        }

        // dominance frontiers (Cytron) — sorted vectors instead of HashSets
        // (much lower allocation/hash overhead for method-sized graphs)
        let mut frontier_sets: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in 0..n {
            // A block whose ONLY predecessor is itself is the entry block of a
            // ROTATED loop — the C# compiler emits `do { body } while (test)`
            // as `body; test; bne body` and the method's first block IS the
            // loop header. The edge from the caller into that block is not a
            // CFG edge, so `preds[b]` holds just the back edge and the
            // standard `>= 2` join test skipped it: its own dominance frontier
            // (itself) was never recorded, no phi was ever placed for the
            // loop-carried registers, and every iteration read the ENTRY
            // version (`while (this._innerException != 0)` with an empty body
            // — an unkillable infinite loop). Treat it as the join it is: it
            // arrives from the caller AND from itself.
            let self_only = preds[b].len() == 1 && preds[b][0] == b;
            if preds[b].len() >= 2 || self_only {
                for &p in &preds[b] {
                    let mut runner = p;
                    while Some(runner) != idom[b] {
                        if !frontier_sets[runner].contains(&b) {
                            frontier_sets[runner].push(b);
                        }
                        match idom[runner] {
                            Some(up) => runner = up,
                            None => break,
                        }
                    }
                }
            }
        }
        let mut frontiers: Vec<Vec<usize>> = frontier_sets;
        for f in frontiers.iter_mut() {
            f.sort_unstable();
        }

        let mut dom_children = vec![Vec::new(); n];
        for (b, d) in idom.iter().enumerate() {
            if let Some(d) = d {
                dom_children[*d].push(b);
            }
        }

        Self {
            idom,
            rpo,
            frontiers,
            dom_children,
        }
    }

    fn reverse_postorder_from(succs: &[Vec<usize>], entry: usize) -> Vec<usize> {
        let n = succs.len();
        if n == 0 {
            return vec![];
        }
        let entry = entry.min(n - 1);
        let mut visited = vec![false; n];
        let mut post = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
        visited[entry] = true;
        while let Some((node, idx)) = stack.pop() {
            let children = &succs[node];
            if idx < children.len() {
                stack.push((node, idx + 1));
                let child = children[idx];
                if !visited[child] {
                    visited[child] = true;
                    stack.push((child, 0));
                }
            } else {
                post.push(node);
            }
        }
        post.reverse();
        post
    }

    /// Does block `a` dominate block `b`?
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        let mut cur = Some(b);
        while let Some(c) = cur {
            if c == a {
                return true;
            }
            cur = self.idom[c];
        }
        false
    }
}
