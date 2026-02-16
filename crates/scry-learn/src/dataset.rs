//! Tabular dataset container for ML workflows.
//!
//! [`Dataset`] provides a lightweight, column-major representation of
//! features + target, with CSV loading and basic column access.

#[cfg(feature = "csv")]
use crate::error::{Result, ScryLearnError};
use crate::matrix::DenseMatrix;

/// A tabular dataset with features and a target column.
///
/// Features are stored column-major (`features[feature_idx][sample_idx]`)
/// for cache-friendly access during tree split evaluation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dataset {
    /// Feature columns: `features[feature_idx][sample_idx]`.
    pub features: Vec<Vec<f64>>,
    /// Target values: `target[sample_idx]`.
    pub target: Vec<f64>,
    /// Feature column names.
    pub feature_names: Vec<String>,
    /// Target column name.
    pub target_name: String,
    /// Class label mapping (index → label string) for classification tasks.
    pub class_labels: Option<Vec<String>>,
    /// Contiguous column-major feature matrix.
    #[cfg_attr(feature = "serde", serde(skip))]
    matrix: Option<DenseMatrix>,
    /// Lazily-computed contiguous row-major feature buffer.
    ///
    /// Layout: `[sample_0_feat_0, sample_0_feat_1, ..., sample_n_feat_m]`.
    /// Populated on first call to [`flat_feature_matrix`].
    #[cfg_attr(feature = "serde", serde(skip))]
    row_major_cache: Option<Vec<f64>>,
}

impl Dataset {
    /// Create a dataset from pre-computed features and target.
    pub fn new(
        features: Vec<Vec<f64>>,
        target: Vec<f64>,
        feature_names: Vec<String>,
        target_name: impl Into<String>,
    ) -> Self {
        let matrix = DenseMatrix::from_col_major(features.clone()).ok();
        Self {
            features,
            target,
            feature_names,
            target_name: target_name.into(),
            class_labels: None,
            matrix,
            row_major_cache: None,
        }
    }

    /// Create a dataset from a [`DenseMatrix`], target, and column names.
    ///
    /// The `features` field is populated from the matrix for backward compat.
    pub fn from_matrix(
        matrix: DenseMatrix,
        target: Vec<f64>,
        feature_names: Vec<String>,
        target_name: impl Into<String>,
    ) -> Self {
        let features = matrix.to_col_vecs();
        Self {
            features,
            target,
            feature_names,
            target_name: target_name.into(),
            class_labels: None,
            matrix: Some(matrix),
            row_major_cache: None,
        }
    }

    /// The contiguous column-major feature matrix.
    ///
    /// Always available after construction via [`new`](Self::new),
    /// [`from_matrix`](Self::from_matrix), or [`from_csv`](Self::from_csv).
    #[inline]
    pub fn matrix(&self) -> &DenseMatrix {
        self.matrix.as_ref().expect("DenseMatrix not initialized")
    }

    /// Load a dataset from a CSV file.
    ///
    /// The `target_column` is extracted as the target; all other numeric
    /// columns become features. String columns are label-encoded automatically.
    ///
    /// Requires the `csv` feature.
    #[cfg(feature = "csv")]
    pub fn from_csv(path: &str, target_column: &str) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(ScryLearnError::Io)?;
        Self::from_csv_reader(file, target_column)
    }

    /// Load a dataset from any reader producing CSV data.
    ///
    /// Requires the `csv` feature.
    #[cfg(feature = "csv")]
    pub fn from_csv_reader(rdr: impl std::io::Read, target_column: &str) -> Result<Self> {
        let mut csv_rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(rdr);

        let headers: Vec<String> = csv_rdr
            .headers()
            .map_err(|e| ScryLearnError::Csv(e.to_string()))?
            .iter()
            .map(std::string::ToString::to_string)
            .collect();

        let target_idx = headers
            .iter()
            .position(|h| h.eq_ignore_ascii_case(target_column))
            .ok_or_else(|| ScryLearnError::InvalidColumn(target_column.to_string()))?;

        // Collect all rows as string records.
        let mut rows: Vec<Vec<String>> = Vec::new();
        for result in csv_rdr.records() {
            let record = result.map_err(|e| ScryLearnError::Csv(e.to_string()))?;
            rows.push(record.iter().map(std::string::ToString::to_string).collect());
        }

        if rows.is_empty() {
            return Err(ScryLearnError::EmptyDataset);
        }

        // Determine which columns are features (all except target).
        let feature_indices: Vec<usize> = (0..headers.len())
            .filter(|&i| i != target_idx)
            .collect();

        let n_samples = rows.len();
        let n_features = feature_indices.len();

        // Parse target — try numeric first, fall back to label encoding.
        let (target, class_labels) = parse_target_column(&rows, target_idx);

        // Parse feature columns — try numeric, label-encode strings.
        let mut features = vec![vec![0.0; n_samples]; n_features];
        let mut feature_names = Vec::with_capacity(n_features);

        for (feat_col, &col_idx) in feature_indices.iter().enumerate() {
            feature_names.push(headers[col_idx].clone());
            for (row_idx, row) in rows.iter().enumerate() {
                let val = row.get(col_idx).map_or("", std::string::String::as_str);
                features[feat_col][row_idx] = val.parse::<f64>().unwrap_or(f64::NAN);
            }
        }

        let matrix = DenseMatrix::from_col_major(features.clone()).ok();
        Ok(Self {
            features,
            target,
            feature_names,
            target_name: headers[target_idx].clone(),
            class_labels,
            matrix,
            row_major_cache: None,
        })
    }

    /// Number of samples (rows).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.target.len()
    }

    /// Number of features (columns).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Number of unique classes in the target (for classification).
    pub fn n_classes(&self) -> usize {
        self.class_labels.as_ref().map_or_else(
            || {
                let mut vals: Vec<i64> = self.target.iter().map(|&v| v as i64).collect();
                vals.sort_unstable();
                vals.dedup();
                vals.len()
            },
            Vec::len,
        )
    }

    /// Get a single feature column by index.
    pub fn feature(&self, idx: usize) -> &[f64] {
        &self.features[idx]
    }

    /// Get a single sample (row) as a vector of feature values.
    pub fn sample(&self, idx: usize) -> Vec<f64> {
        self.features.iter().map(|col| col[idx]).collect()
    }

    /// Get the feature matrix as row-major `[n_samples][n_features]`.
    pub fn feature_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_samples();
        let m = self.n_features();
        let mut matrix = vec![vec![0.0; m]; n];
        for (j, feat_col) in self.features.iter().enumerate() {
            for (i, &val) in feat_col.iter().enumerate() {
                matrix[i][j] = val;
            }
        }
        matrix
    }

    /// Get a contiguous row-major feature buffer, computing on first call.
    ///
    /// Layout: `[sample_0_feat_0, sample_0_feat_1, ..., sample_n_feat_m]`.
    /// Subsequent calls return the cached slice without recomputation.
    pub fn flat_feature_matrix(&mut self) -> &[f64] {
        if self.row_major_cache.is_none() {
            let n = self.n_samples();
            let m = self.n_features();
            let mut buf = vec![0.0; n * m];
            if let Some(mat) = &self.matrix {
                let src = mat.as_slice();
                for j in 0..m {
                    let col_off = j * n;
                    for i in 0..n {
                        buf[i * m + j] = src[col_off + i];
                    }
                }
            } else {
                for j in 0..m {
                    for i in 0..n {
                        buf[i * m + j] = self.features[j][i];
                    }
                }
            }
            self.row_major_cache = Some(buf);
        }
        self.row_major_cache.as_ref().unwrap()
    }

    /// Get a zero-copy row slice from a pre-computed flat feature buffer.
    ///
    /// `cache` should be the result of [`flat_feature_matrix`].
    #[inline]
    pub fn sample_row<'a>(&self, cache: &'a [f64], idx: usize) -> &'a [f64] {
        let m = self.n_features();
        &cache[idx * m..(idx + 1) * m]
    }

    /// Create a subset of this dataset with the given sample indices.
    pub fn subset(&self, indices: &[usize]) -> Self {
        let features: Vec<Vec<f64>> = self
            .features
            .iter()
            .map(|col| indices.iter().map(|&i| col[i]).collect())
            .collect();
        let target = indices.iter().map(|&i| self.target[i]).collect();
        let matrix = DenseMatrix::from_col_major(features.clone()).ok();
        Self {
            features,
            target,
            feature_names: self.feature_names.clone(),
            target_name: self.target_name.clone(),
            class_labels: self.class_labels.clone(),
            matrix,
            row_major_cache: None,
        }
    }

    /// Rebuild the internal [`DenseMatrix`] from the current `features`.
    ///
    /// Call this after mutating `features` in place (e.g. after a
    /// transformer's `transform()` step) so that [`matrix()`](Self::matrix)
    /// returns up-to-date data.
    pub fn sync_matrix(&mut self) {
        self.matrix = DenseMatrix::from_col_major(self.features.clone()).ok();
        self.row_major_cache = None;
    }

    /// Attach class labels for classification.
    pub fn with_class_labels(mut self, labels: Vec<String>) -> Self {
        self.class_labels = Some(labels);
        self
    }
}

#[cfg(feature = "csv")]
/// Parse a target column: try numeric, fall back to label encoding.
///
/// Returns `(encoded_values, Option<class_labels>)`.
fn parse_target_column(rows: &[Vec<String>], col_idx: usize) -> (Vec<f64>, Option<Vec<String>>) {
    // Try parsing all as numeric first.
    let numeric: Vec<Option<f64>> = rows
        .iter()
        .map(|row| {
            row.get(col_idx)
                .and_then(|s| s.parse::<f64>().ok())
        })
        .collect();

    let all_numeric = numeric.iter().all(std::option::Option::is_some);
    if all_numeric {
        return (numeric.into_iter().map(|v| v.unwrap()).collect(), None);
    }

    // Label-encode string values.
    let mut labels: Vec<String> = Vec::new();
    let mut encoded = Vec::with_capacity(rows.len());

    for row in rows {
        let val = row.get(col_idx).map_or("", std::string::String::as_str);
        let idx = labels.iter().position(|l| l == val).unwrap_or_else(|| {
            labels.push(val.to_string());
            labels.len() - 1
        });
        encoded.push(idx as f64);
    }

    (encoded, Some(labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_new() {
        let features = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let target = vec![0.0, 1.0, 0.0];
        let ds = Dataset::new(
            features,
            target,
            vec!["f1".into(), "f2".into()],
            "label",
        );
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.feature(0), &[1.0, 2.0, 3.0]);
        assert_eq!(ds.sample(1), vec![2.0, 5.0]);
    }

    #[cfg(feature = "csv")]
    #[test]
    fn test_dataset_from_csv_reader() {
        let csv = "f1,f2,target\n1.0,4.0,a\n2.0,5.0,b\n3.0,6.0,a\n";
        let ds = Dataset::from_csv_reader(csv.as_bytes(), "target").unwrap();
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.target, vec![0.0, 1.0, 0.0]);
        assert_eq!(
            ds.class_labels,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn test_dataset_subset() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let target = vec![0.0, 1.0, 0.0, 1.0];
        let ds = Dataset::new(features, target, vec!["a".into(), "b".into()], "t");
        let sub = ds.subset(&[0, 2]);
        assert_eq!(sub.n_samples(), 2);
        assert_eq!(sub.feature(0), &[1.0, 3.0]);
        assert_eq!(sub.target, vec![0.0, 0.0]);
    }

    #[cfg(feature = "csv")]
    #[test]
    fn test_empty_csv() {
        let csv = "f1,target\n";
        let err = Dataset::from_csv_reader(csv.as_bytes(), "target");
        assert!(err.is_err());
    }

    #[test]
    fn test_n_classes() {
        let ds = Dataset::new(
            vec![vec![1.0, 2.0, 3.0]],
            vec![0.0, 1.0, 2.0],
            vec!["f".into()],
            "t",
        );
        assert_eq!(ds.n_classes(), 3);
    }

    #[test]
    fn test_matrix_accessor() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let ds = Dataset::new(features, vec![0.0, 1.0], vec!["a".into(), "b".into()], "t");
        let mat = ds.matrix();
        assert_eq!(mat.n_rows(), 2);
        assert_eq!(mat.n_cols(), 2);
        assert_eq!(mat.col(0), &[1.0, 2.0]);
        assert_eq!(mat.col(1), &[3.0, 4.0]);
    }

    #[test]
    fn test_from_matrix() {
        let mat = DenseMatrix::from_col_major(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let ds = Dataset::from_matrix(mat, vec![0.0, 1.0], vec!["a".into(), "b".into()], "t");
        assert_eq!(ds.n_samples(), 2);
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.feature(0), &[1.0, 2.0]);
        assert_eq!(ds.matrix().col(1), &[3.0, 4.0]);
    }
}
