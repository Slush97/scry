//! CART (Classification And Regression Trees) implementation.
//!
//! Implements the full CART algorithm with Gini impurity, entropy,
//! and MSE split criteria. Supports feature bagging for Random Forest.
//!
//! Trees are built recursively using `TreeNode`, then flattened into a
//! contiguous `FlatTree` (Vec<FlatNode>) for cache-optimal prediction.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::weights::{ClassWeight, compute_sample_weights};

// ---------------------------------------------------------------------------
// Split criterion
// ---------------------------------------------------------------------------

/// Split quality criterion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SplitCriterion {
    /// Gini impurity: `1 - Σ(pᵢ²)`.
    Gini,
    /// Information entropy: `-Σ(pᵢ log₂ pᵢ)`.
    Entropy,
    /// Mean squared error (for regression).
    Mse,
}

// ---------------------------------------------------------------------------
// Flat tree — DFS hot/cold cache-optimal representation
// ---------------------------------------------------------------------------

/// Leaf sentinel — stored in `FlatNode::right` to indicate a leaf node.
const LEAF_SENTINEL: u32 = u32::MAX;

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

// ---------------------------------------------------------------------------
// Recursive tree node (used for building, then flattened)
// ---------------------------------------------------------------------------

/// A node in the decision tree (recursive representation).
///
/// Used during tree construction, then flattened into a `FlatTree` for
/// cache-optimal prediction. Exposed publicly for visualization.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TreeNode {
    /// A leaf node — produces a prediction.
    Leaf {
        /// Predicted class (classification) or value (regression).
        prediction: f64,
        /// Number of training samples that reached this node.
        n_samples: usize,
        /// Class distribution at this node (classification only).
        class_counts: Vec<usize>,
        /// Impurity at this node.
        impurity: f64,
    },
    /// An internal split node.
    Split {
        /// Index of the feature used for the split.
        feature_idx: usize,
        /// Threshold value: left if ≤ threshold, right if > threshold.
        threshold: f64,
        /// Left child (≤ threshold).
        left: Box<TreeNode>,
        /// Right child (> threshold).
        right: Box<TreeNode>,
        /// Number of training samples that reached this node.
        n_samples: usize,
        /// Impurity at this node (before split).
        impurity: f64,
        /// Class distribution at this node.
        class_counts: Vec<usize>,
        /// Majority class prediction at this node.
        prediction: f64,
    },
}

impl TreeNode {
    /// Predict for a single sample by walking the tree.
    pub fn predict(&self, sample: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { prediction, .. } => *prediction,
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                if sample[*feature_idx] <= *threshold {
                    left.predict(sample)
                } else {
                    right.predict(sample)
                }
            }
        }
    }

    /// Get class probabilities for a single sample.
    pub fn predict_proba(&self, sample: &[f64], n_classes: usize) -> Vec<f64> {
        match self {
            TreeNode::Leaf {
                class_counts,
                n_samples,
                ..
            } => {
                let mut proba = vec![0.0; n_classes];
                let total = *n_samples as f64;
                for (i, &count) in class_counts.iter().enumerate() {
                    if i < n_classes {
                        proba[i] = count as f64 / total;
                    }
                }
                proba
            }
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                if sample[*feature_idx] <= *threshold {
                    left.predict_proba(sample, n_classes)
                } else {
                    right.predict_proba(sample, n_classes)
                }
            }
        }
    }

    /// Depth of this subtree.
    pub fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    /// Number of leaf nodes in this subtree.
    pub fn n_leaves(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => {
                left.n_leaves() + right.n_leaves()
            }
        }
    }

    /// Number of samples at this node.
    pub fn n_samples(&self) -> usize {
        match self {
            TreeNode::Leaf { n_samples, .. } | TreeNode::Split { n_samples, .. } => *n_samples,
        }
    }

    /// Sum of weighted leaf impurities: Σ(impurity_leaf × n_samples_leaf).
    ///
    /// This is R(T_t) in the cost-complexity pruning literature.
    pub fn total_leaf_impurity(&self) -> f64 {
        match self {
            TreeNode::Leaf { impurity, n_samples, .. } => *impurity * (*n_samples as f64),
            TreeNode::Split { left, right, .. } => {
                left.total_leaf_impurity() + right.total_leaf_impurity()
            }
        }
    }

    /// Minimal cost-complexity pruning (MCCP).
    ///
    /// Recursively prunes subtrees whose effective alpha is ≤ `ccp_alpha`.
    /// Effective alpha = (R(t) - R(T_t)) / (|T_t| - 1), where R(t) is the
    /// re-substitution error if this node were a leaf and R(T_t) is the
    /// total leaf impurity of the subtree.
    ///
    /// This matches sklearn's `ccp_alpha` parameter behavior.
    pub fn prune_ccp(self, ccp_alpha: f64) -> TreeNode {
        match self {
            TreeNode::Leaf { .. } => self,
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
                n_samples,
                impurity,
                class_counts,
                prediction,
            } => {
                // Recursively prune children first (bottom-up).
                let pruned_left = left.prune_ccp(ccp_alpha);
                let pruned_right = right.prune_ccp(ccp_alpha);

                // Build the pruned split node to compute its subtree stats.
                let subtree = TreeNode::Split {
                    feature_idx,
                    threshold,
                    left: Box::new(pruned_left),
                    right: Box::new(pruned_right),
                    n_samples,
                    impurity,
                    class_counts: class_counts.clone(),
                    prediction,
                };

                let n_leaves = subtree.n_leaves();
                if n_leaves <= 1 {
                    return subtree;
                }

                // R(t) = impurity if this node were a leaf.
                let r_node = impurity * (n_samples as f64);
                // R(T_t) = total leaf impurity of subtree.
                let r_subtree = subtree.total_leaf_impurity();

                let effective_alpha = (r_node - r_subtree) / (n_leaves as f64 - 1.0);

                if effective_alpha <= ccp_alpha {
                    // Collapse to leaf.
                    TreeNode::Leaf {
                        prediction,
                        n_samples,
                        class_counts,
                        impurity,
                    }
                } else {
                    subtree
                }
            }
        }
    }

    /// Compute the cost-complexity pruning path.
    ///
    /// Returns `(ccp_alphas, total_impurities)` — a sequence of effective
    /// alpha values and the corresponding total tree impurity at each
    /// pruning step. Useful for elbow-method selection of `ccp_alpha`.
    pub fn cost_complexity_pruning_path(&self) -> (Vec<f64>, Vec<f64>) {
        let mut alphas = vec![0.0];
        let mut impurities = vec![self.total_leaf_impurity()];

        let mut current = self.clone();
        loop {
            // Find the minimum effective alpha across all internal nodes.
            let min_alpha = Self::min_effective_alpha(&current);
            match min_alpha {
                None => break, // no more internal nodes
                Some(alpha) => {
                    current = current.prune_ccp(alpha);
                    alphas.push(alpha);
                    impurities.push(current.total_leaf_impurity());
                }
            }
        }
        (alphas, impurities)
    }

    /// Find the minimum effective alpha among all internal nodes.
    fn min_effective_alpha(node: &TreeNode) -> Option<f64> {
        match node {
            TreeNode::Leaf { .. } => None,
            TreeNode::Split {
                left, right, n_samples, impurity, ..
            } => {
                let n_leaves = node.n_leaves();
                let r_node = impurity * (*n_samples as f64);
                let r_subtree = node.total_leaf_impurity();
                let my_alpha = if n_leaves > 1 {
                    Some((r_node - r_subtree) / (n_leaves as f64 - 1.0))
                } else {
                    None
                };

                let left_alpha = Self::min_effective_alpha(left);
                let right_alpha = Self::min_effective_alpha(right);

                [my_alpha, left_alpha, right_alpha]
                    .iter()
                    .filter_map(|a| *a)
                    .reduce(f64::min)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Classifier
// ---------------------------------------------------------------------------

/// CART decision tree for classification.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecisionTreeClassifier {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    criterion: SplitCriterion,
    ccp_alpha: f64,
    /// Class weighting strategy for imbalanced datasets.
    pub(crate) class_weight: ClassWeight,
    /// Per-sample weights computed from `class_weight` during fit.
    pub(crate) sample_weights: Option<Vec<f64>>,
    /// Flattened tree for cache-optimal prediction.
    pub(crate) flat_tree: Option<FlatTree>,
    n_classes: usize,
    n_features: usize,
    pub(crate) feature_importances_: Vec<f64>,
}

impl DecisionTreeClassifier {
    /// Create a new classifier with default parameters.
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: SplitCriterion::Gini,
            ccp_alpha: 0.0,
            class_weight: ClassWeight::Uniform,
            sample_weights: None,
            flat_tree: None,
            n_classes: 0,
            n_features: 0,
            feature_importances_: Vec::new(),
        }
    }

    /// Set maximum tree depth.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }

    /// Set minimum samples required to split an internal node.
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set minimum samples required in a leaf node.
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set maximum features to consider per split (for random forest).
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    /// Set the split criterion.
    pub fn criterion(mut self, c: SplitCriterion) -> Self {
        self.criterion = c;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    ///
    /// When set to [`ClassWeight::Balanced`], minority classes receive
    /// higher weight in impurity calculations, improving their recall.
    ///
    /// # Example
    /// ```
    /// use scry_learn::tree::DecisionTreeClassifier;
    /// use scry_learn::weights::ClassWeight;
    ///
    /// let dt = DecisionTreeClassifier::new()
    ///     .class_weight(ClassWeight::Balanced);
    /// ```
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Set cost-complexity pruning parameter.
    ///
    /// Subtrees with effective alpha ≤ `ccp_alpha` are pruned after
    /// tree construction. A value of 0.0 (default) disables pruning.
    /// Larger values produce smaller, more regularized trees.
    ///
    /// # Example
    /// ```
    /// use scry_learn::tree::DecisionTreeClassifier;
    ///
    /// let dt = DecisionTreeClassifier::new()
    ///     .ccp_alpha(0.01);
    /// ```
    pub fn ccp_alpha(mut self, alpha: f64) -> Self {
        self.ccp_alpha = alpha;
        self
    }

    /// Train the decision tree on a dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        let indices: Vec<usize> = (0..data.n_samples()).collect();
        self.fit_on_indices(data, &indices)
    }

    /// Train the decision tree on a dataset using a subset of sample indices.
    ///
    /// This is the production path used by Random Forest — avoids copying
    /// the dataset by training directly on indices into the original data.
    ///
    /// Internally uses pre-sorted indices: sorts once at the root (O(n·log n)),
    /// then partitions at each node (O(n) per feature per node) — matching
    /// scikit-learn's optimized CART implementation.
    pub(crate) fn fit_on_indices(&mut self, data: &Dataset, sample_indices: &[usize]) -> Result<()> {
        let n_features = data.n_features();

        // Pre-sort sample indices by each feature value (once at root).
        let mut sorted_by_feature: Vec<Vec<usize>> = Vec::with_capacity(n_features);
        for feat_idx in 0..n_features {
            let col = &data.features[feat_idx];
            let mut sorted = sample_indices.to_vec();
            sorted.sort_unstable_by(|&a, &b| {
                col[a].partial_cmp(&col[b]).unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted_by_feature.push(sorted);
        }

        self.fit_on_indices_presorted(data, sample_indices, &sorted_by_feature)
    }

    /// Train using pre-sorted indices shared across trees (RF memory optimization).
    ///
    /// `global_sorted` contains ALL dataset indices sorted by each feature.
    /// The membership bitset filters to only the bootstrap sample for this tree.
    /// This avoids allocating `sorted_by_feature` per tree during RF training.
    pub(crate) fn fit_on_indices_presorted(
        &mut self,
        data: &Dataset,
        sample_indices: &[usize],
        global_sorted: &[Vec<usize>],
    ) -> Result<()> {
        let n = sample_indices.len();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_features = data.n_features();
        self.n_classes = data.n_classes();
        self.feature_importances_ = vec![0.0; self.n_features];

        // Compute per-sample weights if class_weight is non-uniform.
        let weights = match &self.class_weight {
            ClassWeight::Uniform => None,
            cw => Some(compute_sample_weights(&data.target, cw)),
        };
        self.sample_weights = weights;

        // Membership bitset for partitioning at each node.
        // Must cover ALL dataset indices (not just the bootstrap sample) because
        // build_tree_presorted iterates over global_sorted which contains 0..n_samples.
        let membership_len = global_sorted.first().map_or(0, Vec::len);
        let mut membership = vec![false; membership_len];
        for &i in sample_indices {
            membership[i] = true;
        }

        let tree = if self.sample_weights.is_some() {
            self.build_tree_presorted_weighted(
                data, global_sorted, &mut membership, n, 0,
            )
        } else {
            self.build_tree_presorted(
                data, global_sorted, &mut membership, n, 0,
            )
        };

        // Apply cost-complexity pruning if requested.
        let tree = if self.ccp_alpha > 0.0 {
            tree.prune_ccp(self.ccp_alpha)
        } else {
            tree
        };

        // Flatten recursive tree into contiguous array for prediction.
        let flat = FlatTree::from_tree_node(&tree, self.n_classes);
        self.flat_tree = Some(flat);

        // Normalize feature importances to sum to 1.
        let total: f64 = self.feature_importances_.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importances_ {
                *imp /= total;
            }
        }

        // Free training-only data.
        self.sample_weights = None;

        Ok(())
    }

    /// Predict class labels for a feature matrix.
    ///
    /// `features` is row-major: `features[sample_idx][feature_idx]`.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let ft = self.flat_tree.as_ref().ok_or(ScryLearnError::NotFitted)?;
        Ok(ft.predict(features))
    }

    /// Predict class probabilities for a feature matrix.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let ft = self.flat_tree.as_ref().ok_or(ScryLearnError::NotFitted)?;
        let n_classes = self.n_classes;
        Ok(features
            .iter()
            .map(|row| ft.predict_proba_sample(row, n_classes))
            .collect())
    }

    /// Get feature importances (sum of weighted impurity decreases).
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if self.flat_tree.is_none() {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(self.feature_importances_.clone())
    }

    /// Get the flat tree (for direct access).
    pub fn flat_tree(&self) -> Option<&FlatTree> {
        self.flat_tree.as_ref()
    }

    /// Tree depth.
    pub fn depth(&self) -> usize {
        self.flat_tree.as_ref().map_or(0, FlatTree::depth)
    }

    /// Number of leaf nodes.
    pub fn n_leaves(&self) -> usize {
        self.flat_tree.as_ref().map_or(0, FlatTree::n_leaves)
    }

    /// Number of features the model was trained on.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Compute the cost-complexity pruning path for this classifier.
    ///
    /// Trains an unpruned tree, then returns `(ccp_alphas, total_impurities)`,
    /// a sequence of effective alpha values and the corresponding total tree
    /// impurity at each pruning step. Useful for selecting `ccp_alpha` via
    /// the elbow method.
    ///
    /// The classifier must be fitted before calling this method.
    pub fn cost_complexity_pruning_path(&self, data: &Dataset) -> Result<(Vec<f64>, Vec<f64>)> {
        // Build an unpruned tree to compute the path.
        let mut unpruned = self.clone();
        unpruned.ccp_alpha = 0.0;
        unpruned.fit(data)?;

        // Rebuild the recursive tree from the dataset to get the path.
        let indices: Vec<usize> = (0..data.n_samples()).collect();
        let n_features = data.n_features();
        let mut sorted_by_feature: Vec<Vec<usize>> = Vec::with_capacity(n_features);
        for feat_idx in 0..n_features {
            let col = &data.features[feat_idx];
            let mut sorted = indices.clone();
            sorted.sort_unstable_by(|&a, &b| {
                col[a].partial_cmp(&col[b]).unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted_by_feature.push(sorted);
        }
        let max_idx = indices.iter().copied().max().unwrap_or(0);
        let mut membership = vec![false; max_idx + 1];
        for &i in &indices {
            membership[i] = true;
        }
        let n = indices.len();
        let tree = if unpruned.sample_weights.is_some() {
            unpruned.build_tree_presorted_weighted(
                data, &sorted_by_feature, &mut membership, n, 0,
            )
        } else {
            unpruned.build_tree_presorted(
                data, &sorted_by_feature, &mut membership, n, 0,
            )
        };
        Ok(tree.cost_complexity_pruning_path())
    }

    // -----------------------------------------------------------------------
    // Pre-sorted recursive tree building
    // -----------------------------------------------------------------------

    /// Build tree using pre-sorted indices.
    ///
    /// `sorted_by_feature[feat_idx]` contains ALL sample indices sorted by
    /// that feature's value. `membership[idx]` is true iff `idx` belongs
    /// to the current node. The sorted arrays are filtered on-the-fly
    /// using the membership bitset.
    fn build_tree_presorted(
        &mut self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        membership: &mut [bool],
        n_samples: usize,
        depth: usize,
    ) -> TreeNode {
        // Collect active indices and class counts.
        let mut class_counts = vec![0usize; self.n_classes];
        let mut active_indices = Vec::with_capacity(n_samples);
        // Use any feature's sorted order to gather active indices.
        for &idx in &sorted_by_feature[0] {
            if membership[idx] {
                active_indices.push(idx);
                let c = data.target[idx] as usize;
                if c < self.n_classes {
                    class_counts[c] += 1;
                }
            }
        }
        let n_actual = active_indices.len();
        let impurity = compute_impurity(&class_counts, n_actual, self.criterion);

        // Check stopping conditions.
        let max_depth_reached = self.max_depth.is_some_and(|d| depth >= d);
        let too_few_samples = n_actual < self.min_samples_split;
        let is_pure = impurity < 1e-12;

        if max_depth_reached || too_few_samples || is_pure {
            return TreeNode::Leaf {
                prediction: majority_class(&class_counts),
                n_samples: n_actual,
                class_counts,
                impurity,
            };
        }

        // Find best split — scan pre-sorted arrays (O(n) per feature).
        let best = self.find_best_split_presorted(
            data, sorted_by_feature, membership, &class_counts, n_actual,
        );

        // Pre-compute prediction before class_counts might be moved.
        let node_prediction = majority_class(&class_counts);

        match best {
            None => TreeNode::Leaf {
                prediction: node_prediction,
                n_samples: n_actual,
                class_counts,
                impurity,
            },
            Some(split) => {
                // Partition: mark left/right using membership bitset.
                let col = &data.features[split.feature_idx];
                let mut left_count = 0usize;
                let mut right_count = 0usize;
                let mut right_indices = Vec::new();

                for &idx in &active_indices {
                    if col[idx] <= split.threshold {
                        left_count += 1;
                        // Already in membership, stays.
                    } else {
                        right_count += 1;
                        right_indices.push(idx);
                    }
                }

                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    return TreeNode::Leaf {
                        prediction: node_prediction,
                        n_samples: n_actual,
                        class_counts,
                        impurity,
                    };
                }

                // Record feature importance.
                let n_total = sorted_by_feature[0].len() as f64;
                let weighted_impurity_decrease = (n_actual as f64 / n_total)
                    * (impurity - split.impurity_decrease);
                self.feature_importances_[split.feature_idx] +=
                    weighted_impurity_decrease.max(0.0);

                // Remove right-side indices from membership for left child.
                for &idx in &right_indices {
                    membership[idx] = false;
                }

                let left = self.build_tree_presorted(
                    data, sorted_by_feature, membership, left_count, depth + 1,
                );

                // Swap: remove left from membership, add right.
                for &idx in &active_indices {
                    if col[idx] <= split.threshold {
                        membership[idx] = false;
                    }
                }
                for &idx in &right_indices {
                    membership[idx] = true;
                }

                let right = self.build_tree_presorted(
                    data, sorted_by_feature, membership, right_count, depth + 1,
                );

                // Restore membership for parent.
                for &idx in &active_indices {
                    membership[idx] = true;
                }

                TreeNode::Split {
                    feature_idx: split.feature_idx,
                    threshold: split.threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                    n_samples: n_actual,
                    impurity,
                    class_counts,
                    prediction: node_prediction,
                }
            }
        }
    }

    /// Scan pre-sorted feature arrays to find the best split — O(n) per feature.
    fn find_best_split_presorted(
        &self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        membership: &[bool],
        parent_counts: &[usize],
        n_parent: usize,
    ) -> Option<BestSplit> {
        let n_features = data.n_features();
        let mut best: Option<BestSplit> = None;

        // Feature subset selection (for random forest).
        #[allow(clippy::option_if_let_else)]
        let feature_indices: Vec<usize> = if let Some(max_f) = self.max_features {
            let mut rng = fastrand::Rng::new();
            let mut all: Vec<usize> = (0..n_features).collect();
            let m = max_f.min(n_features);
            for i in 0..m {
                let j = rng.usize(i..n_features);
                all.swap(i, j);
            }
            all.truncate(m);
            all
        } else {
            (0..n_features).collect()
        };

        for &feat_idx in &feature_indices {
            let col = &data.features[feat_idx];
            let sorted = &sorted_by_feature[feat_idx];

            // Scan the pre-sorted array, skipping non-member indices.
            let mut left_counts = vec![0usize; self.n_classes];
            let mut left_n = 0;
            let mut prev_val = f64::NEG_INFINITY;
            let mut prev_was_member = false;

            for &idx in sorted {
                if !membership[idx] {
                    continue;
                }

                let val = col[idx];

                // Check threshold between previous and current value.
                if prev_was_member && left_n > 0 && (val - prev_val).abs() > 1e-12 {
                    let right_n = n_parent - left_n;
                    if left_n >= self.min_samples_leaf && right_n >= self.min_samples_leaf {
                        let right_counts: Vec<usize> = parent_counts
                            .iter()
                            .zip(left_counts.iter())
                            .map(|(&p, &l)| p - l)
                            .collect();

                        let left_imp = compute_impurity(&left_counts, left_n, self.criterion);
                        let right_imp = compute_impurity(&right_counts, right_n, self.criterion);
                        let weighted_imp = (left_n as f64 * left_imp + right_n as f64 * right_imp)
                            / n_parent as f64;

                        let threshold = (prev_val + val) / 2.0;

                        let is_better = best
                            .as_ref()
                            .is_none_or(|b| weighted_imp < b.impurity_decrease);

                        if is_better {
                            best = Some(BestSplit {
                                feature_idx: feat_idx,
                                threshold,
                                impurity_decrease: weighted_imp,
                            });
                        }
                    }
                }

                // Add current sample to left side.
                let class = data.target[idx] as usize;
                if class < self.n_classes {
                    left_counts[class] += 1;
                }
                left_n += 1;
                prev_val = val;
                prev_was_member = true;
            }
        }

        best
    }

    // -------------------------------------------------------------------
    // Weighted tree building (class_weight support)
    // -------------------------------------------------------------------

    /// Build tree using pre-sorted indices with per-sample weights.
    fn build_tree_presorted_weighted(
        &mut self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        membership: &mut [bool],
        _expected_n: usize,
        depth: usize,
    ) -> TreeNode {
        let weights = self.sample_weights.as_ref().expect("weights must be set");

        // Collect active indices and weighted class counts.
        let mut active_indices = Vec::new();
        let mut w_counts = vec![0.0_f64; self.n_classes];
        let mut w_total = 0.0_f64;

        for &idx in &sorted_by_feature[0] {
            if membership[idx] {
                active_indices.push(idx);
                let c = data.target[idx] as usize;
                let w = weights[idx];
                if c < self.n_classes {
                    w_counts[c] += w;
                }
                w_total += w;
            }
        }
        let n_actual = active_indices.len();
        // For class_counts in TreeNode, use unweighted counts.
        let class_counts: Vec<usize> = {
            let mut cc = vec![0usize; self.n_classes];
            for &idx in &active_indices {
                let c = data.target[idx] as usize;
                if c < self.n_classes {
                    cc[c] += 1;
                }
            }
            cc
        };
        let impurity = compute_impurity_weighted(&w_counts, w_total, self.criterion);

        // Check stopping conditions.
        let max_depth_reached = self.max_depth.is_some_and(|d| depth >= d);
        let too_few_samples = n_actual < self.min_samples_split;
        let is_pure = impurity < 1e-12;

        if max_depth_reached || too_few_samples || is_pure {
            return TreeNode::Leaf {
                prediction: weighted_majority_class(&w_counts),
                n_samples: n_actual,
                class_counts,
                impurity,
            };
        }

        let best = self.find_best_split_presorted_weighted(
            data, sorted_by_feature, membership, &w_counts, w_total, n_actual,
        );

        let node_prediction = weighted_majority_class(&w_counts);

        match best {
            None => TreeNode::Leaf {
                prediction: node_prediction,
                n_samples: n_actual,
                class_counts,
                impurity,
            },
            Some(split) => {
                let col = &data.features[split.feature_idx];
                let mut left_count = 0usize;
                let mut right_count = 0usize;
                let mut right_indices = Vec::new();

                for &idx in &active_indices {
                    if col[idx] <= split.threshold {
                        left_count += 1;
                    } else {
                        right_count += 1;
                        right_indices.push(idx);
                    }
                }

                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    return TreeNode::Leaf {
                        prediction: node_prediction,
                        n_samples: n_actual,
                        class_counts,
                        impurity,
                    };
                }

                // Record feature importance.
                let n_total = sorted_by_feature[0].len() as f64;
                let weighted_impurity_decrease = (n_actual as f64 / n_total)
                    * (impurity - split.impurity_decrease);
                self.feature_importances_[split.feature_idx] +=
                    weighted_impurity_decrease.max(0.0);

                // Remove right-side indices from membership for left child.
                for &idx in &right_indices {
                    membership[idx] = false;
                }

                let left = self.build_tree_presorted_weighted(
                    data, sorted_by_feature, membership, left_count, depth + 1,
                );

                // Swap: remove left from membership, add right.
                for &idx in &active_indices {
                    if col[idx] <= split.threshold {
                        membership[idx] = false;
                    }
                }
                for &idx in &right_indices {
                    membership[idx] = true;
                }

                let right = self.build_tree_presorted_weighted(
                    data, sorted_by_feature, membership, right_count, depth + 1,
                );

                // Restore membership for parent.
                for &idx in &active_indices {
                    membership[idx] = true;
                }

                TreeNode::Split {
                    feature_idx: split.feature_idx,
                    threshold: split.threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                    n_samples: n_actual,
                    impurity,
                    class_counts,
                    prediction: node_prediction,
                }
            }
        }
    }

    /// Weighted variant of find_best_split_presorted — uses f64 weighted counts.
    fn find_best_split_presorted_weighted(
        &self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        membership: &[bool],
        parent_w_counts: &[f64],
        w_parent_total: f64,
        n_parent: usize,
    ) -> Option<BestSplit> {
        let weights = self.sample_weights.as_ref().expect("weights must be set");
        let n_features = data.n_features();
        let mut best: Option<BestSplit> = None;

        // Feature subset selection (for random forest).
        #[allow(clippy::option_if_let_else)]
        let feature_indices: Vec<usize> = if let Some(max_f) = self.max_features {
            let mut rng = fastrand::Rng::new();
            let mut all: Vec<usize> = (0..n_features).collect();
            let m = max_f.min(n_features);
            for i in 0..m {
                let j = rng.usize(i..n_features);
                all.swap(i, j);
            }
            all.truncate(m);
            all
        } else {
            (0..n_features).collect()
        };

        for &feat_idx in &feature_indices {
            let col = &data.features[feat_idx];
            let sorted = &sorted_by_feature[feat_idx];

            let mut left_w_counts = vec![0.0_f64; self.n_classes];
            let mut left_w_total = 0.0_f64;
            let mut left_n = 0usize;
            let mut prev_val = f64::NEG_INFINITY;
            let mut prev_was_member = false;

            for &idx in sorted {
                if !membership[idx] {
                    continue;
                }

                let val = col[idx];
                let w = weights[idx];

                if prev_was_member && left_n > 0 && (val - prev_val).abs() > 1e-12 {
                    let right_n = n_parent - left_n;
                    if left_n >= self.min_samples_leaf && right_n >= self.min_samples_leaf {
                        let right_w_total = w_parent_total - left_w_total;
                        let right_w_counts: Vec<f64> = parent_w_counts
                            .iter()
                            .zip(left_w_counts.iter())
                            .map(|(&p, &l)| (p - l).max(0.0))
                            .collect();

                        let left_imp = compute_impurity_weighted(
                            &left_w_counts, left_w_total, self.criterion,
                        );
                        let right_imp = compute_impurity_weighted(
                            &right_w_counts, right_w_total, self.criterion,
                        );
                        let weighted_imp = (left_w_total * left_imp + right_w_total * right_imp)
                            / w_parent_total;

                        let threshold = (prev_val + val) / 2.0;

                        let is_better = best
                            .as_ref()
                            .is_none_or(|b| weighted_imp < b.impurity_decrease);

                        if is_better {
                            best = Some(BestSplit {
                                feature_idx: feat_idx,
                                threshold,
                                impurity_decrease: weighted_imp,
                            });
                        }
                    }
                }

                // Add current sample to left side.
                let class = data.target[idx] as usize;
                if class < self.n_classes {
                    left_w_counts[class] += w;
                }
                left_w_total += w;
                left_n += 1;
                prev_val = val;
                prev_was_member = true;
            }
        }

        best
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Decision Tree Regressor
// ---------------------------------------------------------------------------

/// CART decision tree for regression.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecisionTreeRegressor {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    ccp_alpha: f64,
    /// Flattened tree for cache-optimal prediction.
    pub(crate) flat_tree: Option<FlatTree>,
    n_features: usize,
    pub(crate) feature_importances_: Vec<f64>,
}

impl DecisionTreeRegressor {
    /// Create a new regressor with default parameters.
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            ccp_alpha: 0.0,
            flat_tree: None,
            n_features: 0,
            feature_importances_: Vec::new(),
        }
    }

    /// Set maximum tree depth.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }

    /// Set minimum samples required to split.
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set minimum samples required in a leaf.
    pub fn min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set maximum features per split (for random forest).
    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    /// Set cost-complexity pruning parameter.
    ///
    /// Subtrees with effective alpha ≤ `ccp_alpha` are pruned after
    /// tree construction. A value of 0.0 (default) disables pruning.
    /// Larger values produce smaller, more regularized trees.
    pub fn ccp_alpha(mut self, alpha: f64) -> Self {
        self.ccp_alpha = alpha;
        self
    }

    /// Train on a dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        let indices: Vec<usize> = (0..data.n_samples()).collect();
        self.fit_on_indices(data, &indices)
    }

    /// Train on a dataset using a subset of sample indices.
    ///
    /// Production path for Random Forest — trains directly on indices
    /// into the original data, avoiding dataset copies.
    pub(crate) fn fit_on_indices(&mut self, data: &Dataset, sample_indices: &[usize]) -> Result<()> {
        let n = sample_indices.len();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        self.n_features = data.n_features();
        self.feature_importances_ = vec![0.0; self.n_features];

        // Pre-sort sample indices by each feature (once at root).
        let n_features = data.n_features();
        let mut sorted_by_feature: Vec<Vec<usize>> = Vec::with_capacity(n_features);
        for feat_idx in 0..n_features {
            let col = &data.features[feat_idx];
            let mut sorted = sample_indices.to_vec();
            sorted.sort_unstable_by(|&a, &b| {
                col[a].partial_cmp(&col[b]).unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted_by_feature.push(sorted);
        }

        let max_idx = sample_indices.iter().copied().max().unwrap_or(0);
        let mut membership = vec![false; max_idx + 1];
        for &i in sample_indices {
            membership[i] = true;
        }

        let tree = self.build_tree_presorted_reg(
            data, &sorted_by_feature, &mut membership, n, 0,
        );

        // Apply cost-complexity pruning if requested.
        let tree = if self.ccp_alpha > 0.0 {
            tree.prune_ccp(self.ccp_alpha)
        } else {
            tree
        };

        // Flatten recursive tree into contiguous array for prediction.
        // Regression trees don't need class probabilities (n_classes=0).
        let flat = FlatTree::from_tree_node(&tree, 0);
        self.flat_tree = Some(flat);

        let total: f64 = self.feature_importances_.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importances_ {
                *imp /= total;
            }
        }
        Ok(())
    }

    /// Predict values.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let ft = self.flat_tree.as_ref().ok_or(ScryLearnError::NotFitted)?;
        Ok(ft.predict(features))
    }

    /// Feature importances.
    pub fn feature_importances(&self) -> Result<Vec<f64>> {
        if self.flat_tree.is_none() {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(self.feature_importances_.clone())
    }

    /// Get the flat tree.
    pub fn flat_tree(&self) -> Option<&FlatTree> {
        self.flat_tree.as_ref()
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    fn build_tree_presorted_reg(
        &mut self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        membership: &mut [bool],
        n_samples: usize,
        depth: usize,
    ) -> TreeNode {
        // Collect active indices and compute mean/MSE.
        let mut active_indices = Vec::with_capacity(n_samples);
        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        for &idx in &sorted_by_feature[0] {
            if membership[idx] {
                active_indices.push(idx);
                let v = data.target[idx];
                sum += v;
                sq_sum += v * v;
            }
        }
        let n_actual = active_indices.len();
        if n_actual == 0 {
            return TreeNode::Leaf {
                prediction: 0.0,
                n_samples: 0,
                class_counts: Vec::new(),
                impurity: 0.0,
            };
        }
        let mean = sum / n_actual as f64;
        let mse = sq_sum / n_actual as f64 - mean * mean;

        let max_depth_reached = self.max_depth.is_some_and(|d| depth >= d);
        let too_few = n_actual < self.min_samples_split;

        if max_depth_reached || too_few || mse < 1e-12 {
            return TreeNode::Leaf {
                prediction: mean,
                n_samples: n_actual,
                class_counts: Vec::new(),
                impurity: mse,
            };
        }

        let best = self.find_best_split_presorted_reg(
            data, sorted_by_feature, membership, sum, sq_sum, n_actual,
        );

        match best {
            None => TreeNode::Leaf {
                prediction: mean,
                n_samples: n_actual,
                class_counts: Vec::new(),
                impurity: mse,
            },
            Some(split) => {
                let col = &data.features[split.feature_idx];
                let mut left_count = 0usize;
                let mut right_count = 0usize;
                let mut right_indices = Vec::new();

                for &idx in &active_indices {
                    if col[idx] <= split.threshold {
                        left_count += 1;
                    } else {
                        right_count += 1;
                        right_indices.push(idx);
                    }
                }

                if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    return TreeNode::Leaf {
                        prediction: mean,
                        n_samples: n_actual,
                        class_counts: Vec::new(),
                        impurity: mse,
                    };
                }

                let n_total = sorted_by_feature[0].len() as f64;
                let decrease = (n_actual as f64 / n_total) * (mse - split.impurity_decrease);
                self.feature_importances_[split.feature_idx] += decrease.max(0.0);

                for &idx in &right_indices {
                    membership[idx] = false;
                }

                let left = self.build_tree_presorted_reg(
                    data, sorted_by_feature, membership, left_count, depth + 1,
                );

                for &idx in &active_indices {
                    if col[idx] <= split.threshold {
                        membership[idx] = false;
                    }
                }
                for &idx in &right_indices {
                    membership[idx] = true;
                }

                let right = self.build_tree_presorted_reg(
                    data, sorted_by_feature, membership, right_count, depth + 1,
                );

                for &idx in &active_indices {
                    membership[idx] = true;
                }

                TreeNode::Split {
                    feature_idx: split.feature_idx,
                    threshold: split.threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                    n_samples: n_actual,
                    impurity: mse,
                    class_counts: Vec::new(),
                    prediction: mean,
                }
            }
        }
    }

    fn find_best_split_presorted_reg(
        &self,
        data: &Dataset,
        sorted_by_feature: &[Vec<usize>],
        membership: &[bool],
        total_sum: f64,
        total_sq: f64,
        n_parent: usize,
    ) -> Option<BestSplit> {
        let n_features = data.n_features();
        let mut best: Option<BestSplit> = None;

        #[allow(clippy::option_if_let_else)]
        let feature_indices: Vec<usize> = if let Some(max_f) = self.max_features {
            let mut rng = fastrand::Rng::new();
            let mut all: Vec<usize> = (0..n_features).collect();
            let m = max_f.min(n_features);
            for i in 0..m {
                let j = rng.usize(i..n_features);
                all.swap(i, j);
            }
            all.truncate(m);
            all
        } else {
            (0..n_features).collect()
        };

        for &feat_idx in &feature_indices {
            let col = &data.features[feat_idx];
            let sorted = &sorted_by_feature[feat_idx];

            let mut left_sum = 0.0;
            let mut left_sq_sum = 0.0;
            let mut left_n = 0usize;
            let mut prev_val = f64::NEG_INFINITY;
            let mut prev_was_member = false;

            for &idx in sorted {
                if !membership[idx] {
                    continue;
                }

                let feat_val = col[idx];

                // Check threshold between previous and current.
                if prev_was_member && left_n > 0 && (feat_val - prev_val).abs() > 1e-12 {
                    let right_n = n_parent - left_n;
                    if left_n >= self.min_samples_leaf && right_n >= self.min_samples_leaf {
                        let left_mse = left_sq_sum / left_n as f64
                            - (left_sum / left_n as f64).powi(2);
                        let right_sum = total_sum - left_sum;
                        let right_sq = total_sq - left_sq_sum;
                        let right_mse = right_sq / right_n as f64
                            - (right_sum / right_n as f64).powi(2);

                        let weighted = (left_n as f64 * left_mse + right_n as f64 * right_mse)
                            / n_parent as f64;

                        let threshold = (prev_val + feat_val) / 2.0;

                        let is_better = best.as_ref().is_none_or(|b| weighted < b.impurity_decrease);
                        if is_better {
                            best = Some(BestSplit {
                                feature_idx: feat_idx,
                                threshold,
                                impurity_decrease: weighted,
                            });
                        }
                    }
                }

                let target_val = data.target[idx];
                left_sum += target_val;
                left_sq_sum += target_val * target_val;
                left_n += 1;
                prev_val = feat_val;
                prev_was_member = true;
            }
        }
        best
    }
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct BestSplit {
    feature_idx: usize,
    threshold: f64,
    impurity_decrease: f64,
}



fn compute_impurity(counts: &[usize], n: usize, criterion: SplitCriterion) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    match criterion {
        SplitCriterion::Gini => {
            let sum_sq: f64 = counts
                .iter()
                .map(|&c| {
                    let p = c as f64 / n_f;
                    p * p
                })
                .sum();
            1.0 - sum_sq
        }
        SplitCriterion::Entropy => {
            let mut entropy = 0.0;
            for &c in counts {
                if c > 0 {
                    let p = c as f64 / n_f;
                    entropy -= p * p.log2();
                }
            }
            entropy
        }
        SplitCriterion::Mse => {
            // MSE is not applicable for class counts — used only in regressor.
            0.0
        }
    }
}

fn majority_class(counts: &[usize]) -> f64 {
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &count)| count)
        .map_or(0.0, |(idx, _)| idx as f64)
}

// ---------------------------------------------------------------------------
// Weighted impurity helpers (for class_weight support)
// ---------------------------------------------------------------------------

fn compute_impurity_weighted(counts: &[f64], total: f64, criterion: SplitCriterion) -> f64 {
    if total < 1e-12 {
        return 0.0;
    }
    match criterion {
        SplitCriterion::Gini => {
            let sum_sq: f64 = counts.iter().map(|&c| { let p = c / total; p * p }).sum();
            1.0 - sum_sq
        }
        SplitCriterion::Entropy => {
            let mut entropy = 0.0;
            for &c in counts {
                if c > 1e-12 {
                    let p = c / total;
                    entropy -= p * p.log2();
                }
            }
            entropy
        }
        SplitCriterion::Mse => 0.0,
    }
}

fn weighted_majority_class(counts: &[f64]) -> f64 {
    counts
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0.0, |(idx, _)| idx as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linearly_separable() -> Dataset {
        // Class 0: x < 5, Class 1: x >= 5
        let features = vec![(0..20).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        Dataset::new(features, target, vec!["x".into()], "class")
    }

    #[test]
    fn test_decision_tree_perfect_split() {
        let data = make_linearly_separable();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = dt.predict(&matrix).unwrap();
        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.95,
            "expected ≥95% accuracy on linearly separable data, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_feature_importance_sums_to_one() {
        let data = make_linearly_separable();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let importances = dt.feature_importances().unwrap();
        let total: f64 = importances.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "feature importances should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_max_depth() {
        let data = make_linearly_separable();
        let mut dt = DecisionTreeClassifier::new().max_depth(2);
        dt.fit(&data).unwrap();
        assert!(dt.depth() <= 2 + 1); // depth includes leaf
    }

    #[test]
    fn test_predict_proba() {
        let data = make_linearly_separable();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let sample_class0 = vec![2.0]; // clearly class 0
        let proba = dt.predict_proba(&[sample_class0]).unwrap();
        assert!(proba[0][0] > 0.5, "should predict class 0 with >50%");
    }

    #[test]
    fn test_regressor_basic() {
        // y = x
        let features = vec![(0..50).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut dt = DecisionTreeRegressor::new().max_depth(10);
        dt.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = dt.predict(&matrix).unwrap();

        // Should get low MSE on training data.
        let mse: f64 = preds
            .iter()
            .zip(data.target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / data.n_samples() as f64;

        assert!(mse < 5.0, "MSE on training data should be low, got {mse}");
    }

    #[test]
    fn test_not_fitted_error() {
        let dt = DecisionTreeClassifier::new();
        assert!(dt.predict(&[vec![1.0]]).is_err());
    }

    // -------------------------------------------------------------------
    // Cost-complexity pruning tests
    // -------------------------------------------------------------------

    fn make_iris_like() -> Dataset {
        // A small 3-class dataset with enough samples to build a deep tree.
        let mut rng = fastrand::Rng::with_seed(42);
        let n = 150;
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for _ in 0..50 {
            f1.push(rng.f64() * 2.0);
            f2.push(rng.f64() * 2.0);
            target.push(0.0);
        }
        for _ in 0..50 {
            f1.push(rng.f64() * 2.0 + 3.0);
            f2.push(rng.f64() * 2.0 + 3.0);
            target.push(1.0);
        }
        for _ in 0..50 {
            f1.push(rng.f64() * 2.0 + 6.0);
            f2.push(rng.f64() * 2.0);
            target.push(2.0);
        }
        Dataset::new(
            vec![f1, f2],
            target,
            vec!["f1".into(), "f2".into()],
            "class",
        )
    }

    #[test]
    fn test_ccp_alpha_reduces_depth() {
        let data = make_iris_like();

        let mut dt_full = DecisionTreeClassifier::new();
        dt_full.fit(&data).unwrap();
        let depth_full = dt_full.depth();
        let leaves_full = dt_full.n_leaves();

        let mut dt_pruned = DecisionTreeClassifier::new().ccp_alpha(0.02);
        dt_pruned.fit(&data).unwrap();
        let depth_pruned = dt_pruned.depth();
        let leaves_pruned = dt_pruned.n_leaves();

        eprintln!("Full tree: depth={depth_full}, leaves={leaves_full}");
        eprintln!("Pruned tree: depth={depth_pruned}, leaves={leaves_pruned}");

        assert!(
            leaves_pruned <= leaves_full,
            "Pruned tree should have ≤ leaves than full: {leaves_pruned} vs {leaves_full}"
        );
    }

    #[test]
    fn test_ccp_alpha_zero_no_change() {
        let data = make_iris_like();

        let mut dt_zero = DecisionTreeClassifier::new().ccp_alpha(0.0);
        dt_zero.fit(&data).unwrap();
        let mut dt_default = DecisionTreeClassifier::new();
        dt_default.fit(&data).unwrap();

        assert_eq!(
            dt_zero.n_leaves(),
            dt_default.n_leaves(),
            "ccp_alpha=0.0 should not change the tree"
        );
    }

    #[test]
    fn test_ccp_alpha_large_collapses_to_root() {
        let data = make_iris_like();
        let mut dt = DecisionTreeClassifier::new().ccp_alpha(1000.0);
        dt.fit(&data).unwrap();
        assert_eq!(dt.n_leaves(), 1, "Very large ccp_alpha should collapse to a single leaf");
    }

    #[test]
    fn test_regressor_ccp_alpha() {
        let features = vec![(0..100).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut dt_full = DecisionTreeRegressor::new();
        dt_full.fit(&data).unwrap();

        let mut dt_pruned = DecisionTreeRegressor::new().ccp_alpha(0.01);
        dt_pruned.fit(&data).unwrap();

        let full_leaves = dt_full.flat_tree().unwrap().n_leaves();
        let pruned_leaves = dt_pruned.flat_tree().unwrap().n_leaves();

        eprintln!("Regressor: full={full_leaves} leaves, pruned={pruned_leaves} leaves");
        assert!(
            pruned_leaves <= full_leaves,
            "Pruned regressor should have ≤ leaves: {pruned_leaves} vs {full_leaves}"
        );
    }

    #[test]
    fn test_pruning_path_monotonic() {
        let data = make_iris_like();
        let mut dt = DecisionTreeClassifier::new();
        dt.fit(&data).unwrap();

        let (alphas, impurities) = dt.cost_complexity_pruning_path(&data).unwrap();

        assert!(alphas.len() >= 2, "Should have at least 2 pruning steps");
        // Alphas should be monotonically non-decreasing.
        for w in alphas.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "Alphas should be monotonically non-decreasing: {} -> {}",
                w[0], w[1]
            );
        }
        // Impurities should be monotonically non-decreasing.
        for w in impurities.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "Impurities should be non-decreasing: {} -> {}",
                w[0], w[1]
            );
        }
        eprintln!("Pruning path: {} steps", alphas.len());
        eprintln!("Alphas: {:?}", &alphas[..alphas.len().min(5)]);
    }
}
