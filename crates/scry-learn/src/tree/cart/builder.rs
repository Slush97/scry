//! Decision tree classifier and regressor implementations.
//!
//! Contains the CART tree-building algorithm with pre-sorted indices,
//! feature bagging, class weighting, and cost-complexity pruning.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::weights::{ClassWeight, compute_sample_weights};

use super::{
    BestSplit, FlatTree, SplitCriterion, TreeNode,
    compute_impurity, compute_impurity_weighted,
    majority_class, weighted_majority_class,
};

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
            let mut rng = crate::rng::FastRng::new(0);
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
            let mut rng = crate::rng::FastRng::new(0);
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
            let mut rng = crate::rng::FastRng::new(0);
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
