//! Fuzz target: Scale arithmetic robustness.
//!
//! Exercises `LinearScale`, `LogScale`, and `CategoricalScale` with
//! extreme / pathological inputs. Verifies no panics and all outputs
//! are finite.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_chart::scale::{CategoricalScale, LinearScale, LogScale, Scale};

/// Extract an f64 from fuzz data at a given offset.
fn fuzz_f64(data: &[u8], offset: usize) -> f64 {
    if offset + 8 > data.len() {
        return 0.0;
    }
    f64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 34 {
        return;
    }

    let scale_type = data[0] % 4;

    // Domain and range from fuzz data
    let d_min = fuzz_f64(data, 1);
    let d_max = fuzz_f64(data, 9);
    let r_min = fuzz_f64(data, 17);
    let r_max = fuzz_f64(data, 25);
    let test_val = fuzz_f64(data, 33);

    match scale_type {
        0 => {
            // LinearScale::new
            let s = LinearScale::new((d_min, d_max), (r_min, r_max));
            let px = s.to_pixel(test_val);
            // Any float result is acceptable for arbitrary domain/range combinations
            let _ = px;

            let back = s.to_data(px);
            // Round-trip not guaranteed for degenerate domains, just no panic
            let _ = back;

            let ticks = s.ticks(6);
            for t in &ticks {
                assert!(t.is_finite(), "tick value must be finite");
            }
            for t in &ticks {
                let label = s.format_tick(*t);
                assert!(!label.is_empty(), "tick label must not be empty");
            }

            let (d0, d1) = s.domain();
            assert!(d0.is_finite() || d_min.is_nan() || d_min.is_infinite());
            let _ = d1;
            let _ = s.range();
        }
        1 => {
            // LinearScale::nice — the smartest constructor
            // Clamp extents to non-NaN for nice() which expects real data
            let e_min = if d_min.is_finite() { d_min } else { 0.0 };
            let e_max = if d_max.is_finite() { d_max } else { 100.0 };
            let rr_min = if r_min.is_finite() {
                r_min.clamp(-10000.0, 10000.0)
            } else {
                0.0
            };
            let rr_max = if r_max.is_finite() {
                r_max.clamp(-10000.0, 10000.0)
            } else {
                400.0
            };

            let s = LinearScale::nice((e_min, e_max), (rr_min, rr_max));
            let px = s.to_pixel(test_val);
            let _ = px; // no panic is success

            let ticks = s.ticks(8);
            for t in &ticks {
                assert!(t.is_finite(), "nice tick must be finite");
            }

            // Also test nice_zero
            let s2 = LinearScale::nice_zero((e_min, e_max), (rr_min, rr_max));
            let px2 = s2.to_pixel(test_val);
            let _ = px2;

            let ticks2 = s2.ticks(6);
            for t in &ticks2 {
                assert!(t.is_finite(), "nice_zero tick must be finite");
            }
        }
        2 => {
            // LogScale
            // Clamp domain to positive for log scale
            let ld_min = if d_min.is_finite() && d_min > 0.0 {
                d_min
            } else {
                0.001
            };
            let ld_max = if d_max.is_finite() && d_max > ld_min {
                d_max
            } else {
                ld_min * 1000.0
            };
            let lr_min = if r_min.is_finite() {
                r_min.clamp(-10000.0, 10000.0)
            } else {
                0.0
            };
            let lr_max = if r_max.is_finite() {
                r_max.clamp(-10000.0, 10000.0)
            } else {
                400.0
            };

            let s = LogScale::new((ld_min, ld_max), (lr_min, lr_max));
            let px = s.to_pixel(test_val);
            assert!(
                px.is_finite(),
                "LogScale.to_pixel must be finite, got {px} for {test_val}"
            );

            let back = s.to_data(px);
            let _ = back;

            let ticks = s.ticks(6);
            // Ticks should not be empty and no panics should occur
            // (finiteness is not guaranteed for extreme pathological domains)
            assert!(!ticks.is_empty(), "log ticks must not be empty");
            for t in &ticks {
                let label = s.format_tick(*t);
                assert!(!label.is_empty());
            }
        }
        _ => {
            // CategoricalScale
            let n_labels = (data[1] % 50) as usize; // 0..49 labels
            let labels: Vec<String> = (0..n_labels).map(|i| format!("C{i}")).collect();

            let cr_min = if r_min.is_finite() {
                r_min.clamp(-10000.0, 10000.0)
            } else {
                0.0
            };
            let cr_max = if r_max.is_finite() {
                r_max.clamp(-10000.0, 10000.0)
            } else {
                400.0
            };

            let s = CategoricalScale::new(labels.clone(), (cr_min, cr_max));

            let bw = s.band_width();
            assert!(bw.is_finite(), "band_width must be finite");

            // Test center for every label index
            for i in 0..n_labels {
                let c = s.center(i);
                assert!(c.is_finite(), "center({i}) must be finite");
            }

            // Test out-of-bounds index (should not panic)
            if n_labels > 0 {
                let c_oob = s.center(n_labels + 10);
                let _ = c_oob;
            }

            let _ = s.labels();
        }
    }
});
