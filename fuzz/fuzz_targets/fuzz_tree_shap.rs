//! Fuzz target: TreeSHAP explainability.
//!
//! Builds valid FlatTrees from fuzz bytes (reusing `build_subtree` pattern
//! from `fuzz_cart_predict`) and exercises `tree_shap()`. Max depth 3 and
//! max 4 features to keep SHAP (O(2^depth)) fast.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::explain::tree_shap;
use scry_learn::tree::{FlatNode, FlatTree};

const LEAF_SENTINEL: u32 = u32::MAX;

/// Recursively build a valid DFS pre-ordered tree into `nodes`.
fn build_subtree(
    nodes: &mut Vec<FlatNode>,
    leaf_count: &mut u32,
    data: &[u8],
    cursor: &mut usize,
    n_features: u32,
    depth: usize,
    max_depth: usize,
) {
    let make_leaf = depth >= max_depth || *cursor >= data.len() || data[*cursor] % 3 == 0;
    if *cursor < data.len() {
        *cursor += 1;
    }

    if make_leaf {
        let li = *leaf_count;
        *leaf_count += 1;
        nodes.push(FlatNode::new(LEAF_SENTINEL, li, 0.0));
        return;
    }

    let my_idx = nodes.len();

    let feat_idx = if *cursor < data.len() {
        let v = data[*cursor] as u32 % n_features;
        *cursor += 1;
        v
    } else {
        0
    };

    let threshold = if *cursor + 4 <= data.len() {
        let v = f32::from_le_bytes([
            data[*cursor],
            data[*cursor + 1],
            data[*cursor + 2],
            data[*cursor + 3],
        ]);
        *cursor += 4;
        if v.is_finite() { v as f64 } else { 0.0 }
    } else {
        0.0
    };

    nodes.push(FlatNode::new(0, feat_idx, threshold));

    build_subtree(nodes, leaf_count, data, cursor, n_features, depth + 1, max_depth);

    nodes[my_idx].right = nodes.len() as u32;

    build_subtree(nodes, leaf_count, data, cursor, n_features, depth + 1, max_depth);
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    let n_features = ((data[cursor] % 4) as usize).max(1);
    cursor += 1;
    let max_depth = ((data[cursor] % 3) as usize).max(1); // 1-3 depth for SHAP speed
    cursor += 1;

    // Build a valid tree.
    let mut nodes = Vec::new();
    let mut leaf_count = 0u32;
    build_subtree(
        &mut nodes,
        &mut leaf_count,
        data,
        &mut cursor,
        n_features as u32,
        0,
        max_depth,
    );

    if nodes.is_empty() || leaf_count == 0 {
        return;
    }

    let n_leaves = leaf_count as usize;

    // Build predictions for leaves.
    let mut predictions = Vec::with_capacity(n_leaves);
    for _ in 0..n_leaves {
        if cursor + 4 <= data.len() {
            let v = f32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;
            predictions.push(if v.is_finite() { v as f64 } else { 0.0 });
        } else {
            predictions.push(0.0);
        }
    }

    let tree = FlatTree::new(nodes.clone(), predictions, vec![], 0);

    // Build node_counts (one per node, fuzz-derived).
    let mut node_counts = Vec::with_capacity(nodes.len());
    for _ in 0..nodes.len() {
        if cursor < data.len() {
            node_counts.push((data[cursor] as usize).max(1));
            cursor += 1;
        } else {
            node_counts.push(1);
        }
    }

    // Build a fuzz-derived sample.
    let mut sample = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        if cursor + 4 <= data.len() {
            let v = f32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;
            sample.push(if v.is_finite() { v as f64 } else { 0.0 });
        } else {
            sample.push(0.0);
        }
    }

    // Exercise tree_shap — must not panic.
    let _ = tree_shap(&tree, &sample, &node_counts);
});
