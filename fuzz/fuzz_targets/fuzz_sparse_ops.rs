//! Fuzz target: Sparse matrix operations.
//!
//! Builds CsrMatrix from fuzz-derived triplets and exercises core operations.
//! Tests degenerate cases: duplicate indices, empty rows, out-of-order entries.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::sparse::CsrMatrix;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let mut cursor = 0;

    let n_rows = (data[cursor] % 10).max(1) as usize;
    cursor += 1;
    let n_cols = (data[cursor] % 10).max(1) as usize;
    cursor += 1;
    let n_triplets = (data[cursor] % 20) as usize;
    cursor += 1;

    // Build triplets from fuzz bytes.
    let mut rows = Vec::with_capacity(n_triplets);
    let mut cols = Vec::with_capacity(n_triplets);
    let mut vals = Vec::with_capacity(n_triplets);

    for _ in 0..n_triplets {
        if cursor + 3 > data.len() {
            break;
        }
        let r = data[cursor] as usize % n_rows;
        cursor += 1;
        let c = data[cursor] as usize % n_cols;
        cursor += 1;
        let v = (data[cursor] as f64 / 128.0) - 1.0;
        cursor += 1;
        rows.push(r);
        cols.push(c);
        vals.push(v);
    }

    // Exercise from_triplets — may fail on degenerate input, that's fine.
    if let Ok(csr) = CsrMatrix::from_triplets(&rows, &cols, &vals, n_rows, n_cols) {
        // Exercise accessors.
        let _ = csr.n_rows();
        let _ = csr.n_cols();
        let _ = csr.nnz();
        let _ = csr.to_dense();

        // Exercise get on valid indices.
        for r in 0..n_rows.min(3) {
            for c in 0..n_cols.min(3) {
                let _ = csr.get(r, c);
            }
        }

        // Exercise dot_vec.
        let vec = vec![1.0; n_cols];
        let _ = csr.dot_vec(&vec);

        // Exercise row iteration.
        for r in 0..n_rows.min(3) {
            let row = csr.row(r);
            let _ = row.dot(&vec);
        }

        // Convert to CSC and exercise.
        let csc = csr.to_csc();
        let _ = csc.n_rows();
        let _ = csc.n_cols();
        let _ = csc.to_dense();
    }
});
