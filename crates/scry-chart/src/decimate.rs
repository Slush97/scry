// SPDX-License-Identifier: MIT OR Apache-2.0
//! Level-of-detail decimation for large data-series rendering.
//!
//! When a line or scatter chart has far more data points than pixels,
//! rendering every point wastes CPU and can produce visual noise.
//! These functions thin a series to a visually representative subset.

/// Largest-Triangle-Three-Buckets (LTTB) decimation.
///
/// Reduces `data` to `target_n` points while preserving the visual shape.
/// Returns the original data if `target_n >= data.len()` or `target_n < 3`.
///
/// # References
///
/// Sveinn Steinarsson, "Downsampling Time Series for Visual Representation",
/// University of Iceland, 2013.
///
/// # Examples
///
/// ```
/// use scry_chart::decimate::lttb;
///
/// let data: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, (i as f64).sin())).collect();
/// let reduced = lttb(&data, 100);
/// assert_eq!(reduced.len(), 100);
/// ```
#[must_use]
pub fn lttb(data: &[(f64, f64)], target_n: usize) -> Vec<(f64, f64)> {
    let n = data.len();
    if target_n >= n || target_n < 3 || n < 3 {
        return data.to_vec();
    }

    let mut out = Vec::with_capacity(target_n);

    // Always keep the first point.
    out.push(data[0]);

    let bucket_size = (n - 2) as f64 / (target_n - 2) as f64;

    let mut prev_selected = 0_usize;

    for bucket_i in 0..(target_n - 2) {
        // Current bucket bounds (indices into data, excluding first & last).
        let bucket_start = ((bucket_i as f64 * bucket_size) as usize) + 1;
        let bucket_end = ((((bucket_i + 1) as f64) * bucket_size) as usize + 1).min(n - 1);

        // Next bucket: compute average point for the triangle area test.
        let next_start = bucket_end;
        let next_end = ((((bucket_i + 2) as f64) * bucket_size) as usize + 1).min(n - 1);

        let (avg_x, avg_y) = if next_start < next_end {
            let count = (next_end - next_start) as f64;
            let sx: f64 = data[next_start..next_end].iter().map(|p| p.0).sum();
            let sy: f64 = data[next_start..next_end].iter().map(|p| p.1).sum();
            (sx / count, sy / count)
        } else if next_start < n {
            data[next_start]
        } else {
            data[n - 1]
        };

        // Select the point in the current bucket that maximizes the triangle
        // area formed by (prev_selected, candidate, avg_next_bucket).
        let (px, py) = data[prev_selected];
        let mut best_area = -1.0_f64;
        let mut best_idx = bucket_start;

        for idx in bucket_start..bucket_end {
            let (cx, cy) = data[idx];
            // Signed area × 2 (we only need the max, sign doesn't matter).
            let area = ((px - avg_x) * (cy - py) - (px - cx) * (avg_y - py)).abs();
            if area > best_area {
                best_area = area;
                best_idx = idx;
            }
        }

        out.push(data[best_idx]);
        prev_selected = best_idx;
    }

    // Always keep the last point.
    out.push(data[n - 1]);

    out
}

/// Min-max decimation: for each bucket, keep the point with the minimum and
/// maximum Y value. Output length is at most `2 * buckets`.
///
/// Useful for preserving peaks and valleys. Returns the original data if
/// `buckets * 2 >= data.len()`.
///
/// # Examples
///
/// ```
/// use scry_chart::decimate::min_max_decimate;
///
/// let data: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, (i as f64).sin())).collect();
/// let reduced = min_max_decimate(&data, 100);
/// assert!(reduced.len() <= 200);
/// ```
#[must_use]
pub fn min_max_decimate(data: &[(f64, f64)], buckets: usize) -> Vec<(f64, f64)> {
    let n = data.len();
    if buckets == 0 || buckets * 2 >= n || n < 3 {
        return data.to_vec();
    }

    let bucket_size = n as f64 / buckets as f64;
    let mut out = Vec::with_capacity(buckets * 2);

    for b in 0..buckets {
        let start = (b as f64 * bucket_size) as usize;
        let end = (((b + 1) as f64 * bucket_size) as usize).min(n);

        if start >= end {
            continue;
        }

        let mut min_idx = start;
        let mut max_idx = start;

        for idx in start..end {
            if data[idx].1 < data[min_idx].1 {
                min_idx = idx;
            }
            if data[idx].1 > data[max_idx].1 {
                max_idx = idx;
            }
        }

        // Emit in index order to preserve left-to-right drawing.
        if min_idx <= max_idx {
            out.push(data[min_idx]);
            if min_idx != max_idx {
                out.push(data[max_idx]);
            }
        } else {
            out.push(data[max_idx]);
            if min_idx != max_idx {
                out.push(data[min_idx]);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lttb_identity_when_small() {
        let data: Vec<(f64, f64)> = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)];
        let result = lttb(&data, 10);
        assert_eq!(result, data);
    }

    #[test]
    fn lttb_reduces_correctly() {
        let data: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, (i as f64).sin())).collect();
        let result = lttb(&data, 20);
        assert_eq!(result.len(), 20);
        // First and last points are preserved.
        assert_eq!(result[0], data[0]);
        assert_eq!(result[19], data[99]);
    }

    #[test]
    fn lttb_empty_data() {
        let data: Vec<(f64, f64)> = vec![];
        assert!(lttb(&data, 10).is_empty());
    }

    #[test]
    fn min_max_identity_when_small() {
        let data: Vec<(f64, f64)> = vec![(0.0, 1.0), (1.0, 2.0)];
        let result = min_max_decimate(&data, 10);
        assert_eq!(result, data);
    }

    #[test]
    fn min_max_reduces_correctly() {
        let data: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, (i as f64).sin())).collect();
        let result = min_max_decimate(&data, 10);
        assert!(result.len() <= 20);
        assert!(result.len() >= 10);
    }

    #[test]
    fn min_max_preserves_extremes() {
        // Data with a clear peak and valley.
        let data = vec![
            (0.0, 0.0),
            (1.0, 10.0), // peak
            (2.0, 0.0),
            (3.0, -10.0), // valley
            (4.0, 0.0),
            (5.0, 5.0),
        ];
        let result = min_max_decimate(&data, 2);
        // Should contain the peak and valley values.
        let ys: Vec<f64> = result.iter().map(|p| p.1).collect();
        assert!(ys.contains(&10.0), "peak should be preserved");
        assert!(ys.contains(&-10.0), "valley should be preserved");
    }
}
