// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cache-optimal flat tree representation for prediction.
//!
//! `FlatTree` stores nodes in DFS pre-order for cache-friendly traversal.
//! Predictions and probabilities are stored in compact cold arrays indexed
//! by leaf position.

use super::{LEAF_SENTINEL, TreeNode};

/// Hot-path node for prediction — 16 bytes, **4 per 64-byte cache line**.
///
/// Laid out in DFS pre-order: the left child of node `i` is always
/// `i + 1` (the next node in memory), so no left-index field is needed.
/// Only the right-child index is stored explicitly.
///
/// Leaf nodes are marked by `right == LEAF_SENTINEL`.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FlatNode {
    /// Right child index in DFS order, or `LEAF_SENTINEL` if this is a leaf.
    /// Left child is always `self_idx + 1` (implicit in DFS layout).
    pub right: u32,
    /// Index of feature to split on (ignored for leaves).
    pub feature_idx: u32,
    /// Split threshold — go left if `feature <= threshold` (ignored for leaves).
    pub threshold: f64,
}



/// A cache-optimal decision tree stored as a contiguous DFS pre-ordered array.
///
/// - Left child is always the **next** node in memory (free prefetch)
/// - 16-byte nodes → 4 fit per 64-byte cache line
/// - A depth-8 tree (~255 nodes × 16B = 4KB) fits entirely in L1 cache
/// - Prediction / probability data stored **only for leaf nodes** in compact cold arrays
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(clippy::unsafe_derive_deserialize)]
pub struct FlatTree {
    /// DFS pre-ordered hot nodes (16 bytes each).
    ///
    /// For leaf nodes (`right == LEAF_SENTINEL`), `feature_idx` stores the
    /// leaf data index into `predictions` and `leaf_probas` — no separate
    /// `leaf_indices` Vec needed.
    pub nodes: Vec<FlatNode>,
    /// Prediction value for each **leaf** node (indexed by leaf position).
    pub predictions: Vec<f64>,
    /// Flat array of leaf probabilities as f32: `leaf_probas[leaf_idx * n_classes + class_idx]`.
    /// Empty for regression trees (n_classes_stored == 0).
    pub leaf_probas: Vec<f32>,
    /// Number of classes (stride for `leaf_probas`).
    pub n_classes_stored: u32,
}

impl FlatTree {
    /// Flatten a recursive `TreeNode` into a DFS pre-ordered `FlatTree`.
    ///
    /// Predictions and probabilities are stored **only for leaf nodes**.
    /// The leaf data index is embedded directly in `FlatNode::feature_idx`
    /// for leaf nodes, eliminating a separate `leaf_indices` Vec.
    pub fn from_tree_node(root: &TreeNode, n_classes: usize) -> Self {
        let mut nodes = Vec::new();
        let mut predictions = Vec::new();
        let mut leaf_probas: Vec<f32> = Vec::new();
        let mut leaf_count: u32 = 0;
        Self::flatten_dfs(
            root, &mut nodes, &mut predictions, &mut leaf_probas,
            &mut leaf_count, n_classes,
        );
        FlatTree {
            nodes,
            predictions,
            leaf_probas,
            n_classes_stored: n_classes as u32,
        }
    }

    /// Recursive DFS pre-order flattening.
    ///
    /// For leaf nodes, `feature_idx` is repurposed to store the leaf data
    /// index (into `predictions` / `leaf_probas`).  Internal nodes use it
    /// normally as the split feature index.
    fn flatten_dfs(
        node: &TreeNode,
        nodes: &mut Vec<FlatNode>,
        predictions: &mut Vec<f64>,
        leaf_probas: &mut Vec<f32>,
        leaf_count: &mut u32,
        n_classes: usize,
    ) {
        match node {
            TreeNode::Leaf {
                prediction, n_samples, class_counts, ..
            } => {
                let li = *leaf_count;
                *leaf_count += 1;
                nodes.push(FlatNode {
                    right: LEAF_SENTINEL,
                    feature_idx: li, // repurposed: leaf data index
                    threshold: 0.0,
                });
                predictions.push(*prediction);
                Self::append_proba(leaf_probas, class_counts, *n_samples, n_classes);
            }
            TreeNode::Split {
                feature_idx, threshold, left, right,
                class_counts: _, n_samples: _, prediction: _, ..
            } => {
                let my_idx = nodes.len();
                nodes.push(FlatNode {
                    right: 0, // placeholder — patched after left subtree
                    feature_idx: *feature_idx as u32,
                    threshold: *threshold,
                });

                // Recurse left — left child is always my_idx + 1.
                Self::flatten_dfs(left, nodes, predictions, leaf_probas, leaf_count, n_classes);

                // Right child index = current length (after entire left subtree).
                nodes[my_idx].right = nodes.len() as u32;
                Self::flatten_dfs(right, nodes, predictions, leaf_probas, leaf_count, n_classes);
            }
        }
    }

    /// Append probabilities for one leaf to the flat f32 array.
    fn append_proba(probas: &mut Vec<f32>, class_counts: &[usize], n_samples: usize, n_classes: usize) {
        if n_classes > 0 && n_samples > 0 {
            let total = n_samples as f64;
            for i in 0..n_classes {
                let count = if i < class_counts.len() { class_counts[i] } else { 0 };
                probas.push((count as f64 / total) as f32);
            }
        }
    }

    /// Predict for a single sample — zero-overhead DFS traversal.
    ///
    /// # Safety invariants (verified by `debug_assert!` in debug builds)
    /// - All `right` indices and implicit left indices (`idx + 1`) are in-bounds
    /// - All `feature_idx` values are valid indices into `sample`
    /// - Guaranteed by `from_tree_node()` construction
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    pub fn predict_sample(&self, sample: &[f64]) -> f64 {
        let nodes = self.nodes.as_slice();
        let preds = self.predictions.as_slice();
        debug_assert!(!nodes.is_empty());
        let mut idx = 0usize;
        loop {
            debug_assert!(idx < nodes.len());
            // SAFETY: idx is 0 (initial) or from DFS indices validated at construction.
            let node = unsafe { nodes.get_unchecked(idx) };
            if node.right == LEAF_SENTINEL {
                // feature_idx stores leaf data index for leaf nodes.
                let li = node.feature_idx as usize;
                return unsafe { *preds.get_unchecked(li) };
            }
            // SAFETY: feature_idx < n_features, validated during fit().
            let feat_val = unsafe { *sample.get_unchecked(node.feature_idx as usize) };
            idx = if feat_val <= node.threshold {
                idx + 1                    // left child: next in DFS order
            } else {
                node.right as usize        // right child: stored index
            };
        }
    }

    /// Predict class probabilities for a single sample.
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    pub fn predict_proba_sample(&self, sample: &[f64], n_classes: usize) -> Vec<f64> {
        let nodes = self.nodes.as_slice();
        let nc = self.n_classes_stored as usize;
        debug_assert!(!nodes.is_empty());
        let mut idx = 0usize;
        loop {
            debug_assert!(idx < nodes.len());
            let node = unsafe { nodes.get_unchecked(idx) };
            if node.right == LEAF_SENTINEL {
                let li = node.feature_idx as usize;
                let start = li * nc;
                let mut result = vec![0.0; n_classes];
                let copy_len = n_classes.min(nc);
                for (i, p) in self.leaf_probas[start..start + copy_len].iter().enumerate() {
                    result[i] = *p as f64;
                }
                return result;
            }
            let feat_val = unsafe { *sample.get_unchecked(node.feature_idx as usize) };
            idx = if feat_val <= node.threshold {
                idx + 1
            } else {
                node.right as usize
            };
        }
    }

    // ── GBT compatibility methods ──

    /// Number of nodes in the tree (DFS array length).
    ///
    /// Used by GBT for sizing per-node accumulation arrays.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Read a leaf's prediction value given its DFS node index.
    ///
    /// Panics in debug if `node_idx` is not a leaf.
    #[inline]
    pub fn leaf_prediction(&self, node_idx: usize) -> f64 {
        let node = &self.nodes[node_idx];
        debug_assert!(node.right == LEAF_SENTINEL, "node_idx {node_idx} is not a leaf");
        self.predictions[node.feature_idx as usize]
    }

    /// Set a leaf's prediction value given its DFS node index.
    ///
    /// Used by GBT Newton-Raphson correction. Panics in debug if not a leaf.
    #[inline]
    pub fn set_leaf_prediction(&mut self, node_idx: usize, value: f64) {
        let node = &self.nodes[node_idx];
        debug_assert!(node.right == LEAF_SENTINEL, "node_idx {node_idx} is not a leaf");
        self.predictions[node.feature_idx as usize] = value;
    }

    /// Check if a DFS node index is a leaf.
    #[inline]
    pub fn is_leaf(&self, node_idx: usize) -> bool {
        self.nodes[node_idx].right == LEAF_SENTINEL
    }

    /// Predict for all samples.
    pub fn predict(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|row| self.predict_sample(row)).collect()
    }

    /// Route a single sample to its leaf and return the leaf node index.
    ///
    /// This is used by GBT for Newton-Raphson leaf value correction.
    #[inline(always)]
    #[allow(unsafe_code, clippy::inline_always)]
    pub(crate) fn apply_sample(&self, sample: &[f64]) -> usize {
        let nodes = self.nodes.as_slice();
        debug_assert!(!nodes.is_empty());
        let mut idx = 0usize;
        loop {
            debug_assert!(idx < nodes.len());
            let node = unsafe { nodes.get_unchecked(idx) };
            if node.right == LEAF_SENTINEL {
                return idx;
            }
            let feat_val = unsafe { *sample.get_unchecked(node.feature_idx as usize) };
            idx = if feat_val <= node.threshold {
                idx + 1
            } else {
                node.right as usize
            };
        }
    }

    /// Route all samples to their leaf nodes, returning leaf indices.
    pub(crate) fn apply(&self, features: &[Vec<f64>]) -> Vec<usize> {
        features.iter().map(|row| self.apply_sample(row)).collect()
    }

    /// Depth of the tree.
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() { return 0; }
        Self::depth_at(&self.nodes, 0, 1)
    }

    fn depth_at(nodes: &[FlatNode], idx: usize, d: usize) -> usize {
        if idx >= nodes.len() { return 0; }
        let node = &nodes[idx];
        if node.right == LEAF_SENTINEL {
            d
        } else {
            let l = Self::depth_at(nodes, idx + 1, d + 1);
            let r = Self::depth_at(nodes, node.right as usize, d + 1);
            l.max(r)
        }
    }

    /// Number of leaf nodes.
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.right == LEAF_SENTINEL).count()
    }
}
