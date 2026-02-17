//! Integration test: end-to-end sparse Dataset workflow.

use scry_learn::dataset::Dataset;
use scry_learn::sparse::CscMatrix;
use scry_learn::split::train_test_split;

#[test]
fn test_sparse_end_to_end() {
    // 10 samples × 3 features, mostly zeros.
    let col0 = vec![1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 0.0, 0.0, 10.0];
    let col1 = vec![0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 0.0];
    let col2 = vec![0.0, 0.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 9.0, 0.0];
    let csc = CscMatrix::from_dense(&[col0, col1, col2]);
    let target = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let ds = Dataset::from_sparse(
        csc,
        target,
        vec!["f0".into(), "f1".into(), "f2".into()],
        "label",
    );
    assert!(ds.is_sparse());
    assert_eq!(ds.n_samples(), 10);
    assert_eq!(ds.n_features(), 3);

    // Train/test split preserves sparsity.
    let (train, test) = train_test_split(&ds, 0.3, 42);
    assert!(train.is_sparse());
    assert!(test.is_sparse());
    assert_eq!(train.n_samples() + test.n_samples(), 10);
    assert_eq!(train.n_features(), 3);
    assert_eq!(test.n_features(), 3);

    // CSC accessor works on split subsets.
    let train_csc = train.sparse_csc().expect("train should be sparse");
    assert_eq!(train_csc.n_rows(), train.n_samples());
    assert_eq!(train_csc.n_cols(), 3);

    let test_csc = test.sparse_csc().expect("test should be sparse");
    assert_eq!(test_csc.n_rows(), test.n_samples());
    assert_eq!(test_csc.n_cols(), 3);
}

#[test]
fn test_sparse_ensure_dense_after_split() {
    let col0 = vec![1.0, 0.0, 2.0, 0.0];
    let col1 = vec![0.0, 3.0, 0.0, 4.0];
    let csc = CscMatrix::from_dense(&[col0, col1]);
    let target = vec![0.0, 1.0, 0.0, 1.0];

    let ds = Dataset::from_sparse(csc, target, vec!["a".into(), "b".into()], "t");
    let (mut train, _test) = train_test_split(&ds, 0.5, 42);

    // Densify and verify features are accessible.
    train.ensure_dense();
    assert_eq!(train.features.len(), 2);
    assert_eq!(train.n_samples(), train.features[0].len());
}
