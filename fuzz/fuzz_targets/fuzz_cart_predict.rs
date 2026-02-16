//! Fuzz target: CART unsafe predict paths.
//!
//! Builds structurally valid `FlatTree` structs from fuzz bytes and exercises
//! `predict_sample` and `predict_proba_sample` with fuzz-derived thresholds
//! and sample values. The tree topology is valid (all paths reach leaves,
//! all indices in-bounds) but thresholds, predictions, and sample values
//! are fuzz-controlled.
//!
//! We only care about no-panic, no-OOB, no-UB — correctness is not checked.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::tree::{FlatNode, FlatTree};

const LEAF_SENTINEL: u32 = u32::MAX;

/// Recursively build a valid DFS pre-ordered tree into `nodes`.
/// Returns the number of leaves created in this subtree.
fn build_subtree(
    nodes: &mut Vec<FlatNode>,
    leaf_count: &mut u32,
    data: &[u8],
    cursor: &mut usize,
    n_features: u32,
    depth: usize,
    max_depth: usize,
) {
    // Decide leaf vs split from fuzz byte.
    let make_leaf = depth >= max_depth || *cursor >= data.len() || data[*cursor] % 3 == 0;
    if *cursor < data.len() {
        *cursor += 1;
    }

    if make_leaf {
        let li = *leaf_count;
        *leaf_count += 1;
        nodes.push(FlatNode {
            right: LEAF_SENTINEL,
            feature_idx: li,
            threshold: 0.0,
        });
        return;
    }

    // Internal node — placeholder, will patch `right` after left subtree.
    let my_idx = nodes.len();

    // feature_idx from fuzz, clamped to valid range.
    let feat_idx = if *cursor < data.len() {
        let v = data[*cursor] as u32 % n_features;
        *cursor += 1;
        v
    } else {
        0
    };

    // threshold from fuzz.
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

    nodes.push(FlatNode {
        right: 0, // placeholder
        feature_idx: feat_idx,
        threshold,
    });

    // Left child: next in DFS order (my_idx + 1).
    build_subtree(nodes, leaf_count, data, cursor, n_features, depth + 1, max_depth);

    // Patch right child index = current length.
    nodes[my_idx].right = nodes.len() as u32;

    // Right child.
    build_subtree(nodes, leaf_count, data, cursor, n_features, depth + 1, max_depth);
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = 0;

    // Parse parameters.
    let n_features = ((data[cursor] % 8) as usize).max(1);
    cursor += 1;
    let n_classes = ((data[cursor] % 4) as usize).max(2);
    cursor += 1;
    let max_depth = ((data[cursor] % 5) as usize).max(1); // 1-5 depth
    cursor += 1;

    // Build a valid DFS pre-ordered tree.
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

    // Build predictions array (one per leaf).
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

    // Build leaf_probas array (n_leaves * n_classes).
    let probas_len = n_leaves * n_classes;
    let mut leaf_probas = Vec::with_capacity(probas_len);
    for _ in 0..probas_len {
        if cursor + 4 <= data.len() {
            let v = f32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;
            leaf_probas.push(if v.is_finite() { v } else { 0.0 });
        } else {
            leaf_probas.push(0.0);
        }
    }

    let tree = FlatTree {
        nodes,
        predictions,
        leaf_probas,
        n_classes_stored: n_classes as u32,
    };

    // Build a fuzz-derived sample vector.
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

    // Exercise predict methods — these must not panic or trigger UB.
    let _ = tree.predict_sample(&sample);
    let _ = tree.predict_proba_sample(&sample, n_classes);
    let _ = tree.predict(&[sample]);
});
