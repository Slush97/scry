// SPDX-License-Identifier: MIT OR Apache-2.0
//! Extract data extents from [`Chart`] enum variants.
//!
//! Provides a utility function to compute the (x_min, x_max, y_min, y_max)
//! bounding box of a chart's data, enabling unified domain computation
//! for shared-axis subplot grids.

use crate::chart::Chart;

/// Compute the data extent of a chart as `(x_min, x_max, y_min, y_max)`.
///
/// Returns `None` for chart types that don't have conventional XY axes
/// (e.g., Pie, Radar, Gauge, Funnel, Sparkline).
///
/// # Example
///
/// ```
/// use scry_chart::chart::Chart;
/// use scry_chart::chart::extent::data_extent;
///
/// let chart = Chart::line(&[1.0, 5.0, 3.0]).build();
/// let ext = data_extent(&chart).unwrap();
/// assert_eq!(ext.0, 0.0); // x_min
/// assert_eq!(ext.2, 1.0); // y_min
/// assert_eq!(ext.3, 5.0); // y_max
/// ```
#[must_use]
pub fn data_extent(chart: &Chart) -> Option<(f64, f64, f64, f64)> {
    match chart {
        Chart::Line(c) => {
            let ys: Vec<f64> = c
                .series
                .iter()
                .flat_map(|s| s.values().iter().copied())
                .collect();
            if ys.is_empty() {
                return None;
            }
            let x_max = c.x_values.as_ref().map_or_else(
                || (ys.len().saturating_sub(1)) as f64,
                |xv| xv.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
            let x_min = c.x_values.as_ref().map_or_else(
                || 0.0,
                |xv| xv.iter().copied().fold(f64::INFINITY, f64::min),
            );
            let y_min = ys.iter().copied().fold(f64::INFINITY, f64::min);
            let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Some((x_min, x_max, y_min, y_max))
        }
        Chart::Scatter(c) => {
            let xs = c.x.values();
            let ys = c.y.values();
            if xs.is_empty() {
                return None;
            }
            let mut x_min = f64::INFINITY;
            let mut x_max = f64::NEG_INFINITY;
            let mut y_min = f64::INFINITY;
            let mut y_max = f64::NEG_INFINITY;
            for &v in xs {
                if v < x_min {
                    x_min = v;
                }
                if v > x_max {
                    x_max = v;
                }
            }
            for &v in ys {
                if v < y_min {
                    y_min = v;
                }
                if v > y_max {
                    y_max = v;
                }
            }
            // Include extra series
            for (ex, ey) in &c.extra_series {
                for &v in ex.values() {
                    if v < x_min {
                        x_min = v;
                    }
                    if v > x_max {
                        x_max = v;
                    }
                }
                for &v in ey.values() {
                    if v < y_min {
                        y_min = v;
                    }
                    if v > y_max {
                        y_max = v;
                    }
                }
            }
            Some((x_min, x_max, y_min, y_max))
        }
        Chart::Bar(c) => {
            let ys: Vec<f64> = c
                .series
                .iter()
                .flat_map(|s| s.values().iter().copied())
                .collect();
            if ys.is_empty() {
                return None;
            }
            let n = c.labels.len();
            let y_min = ys.iter().copied().fold(0.0_f64, f64::min);
            let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Some((0.0, (n.saturating_sub(1)) as f64, y_min, y_max))
        }
        Chart::Histogram(c) => {
            let vals = c.data.values();
            if vals.is_empty() {
                return None;
            }
            let x_min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let x_max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            // Y extent isn't meaningful until binning, so estimate
            Some((x_min, x_max, 0.0, vals.len() as f64))
        }
        Chart::Bubble(c) => {
            let xs = c.x.values();
            let ys = c.y.values();
            if xs.is_empty() {
                return None;
            }
            let mut x_min = f64::INFINITY;
            let mut x_max = f64::NEG_INFINITY;
            let mut y_min = f64::INFINITY;
            let mut y_max = f64::NEG_INFINITY;
            for &v in xs {
                if v < x_min {
                    x_min = v;
                }
                if v > x_max {
                    x_max = v;
                }
            }
            for &v in ys {
                if v < y_min {
                    y_min = v;
                }
                if v > y_max {
                    y_max = v;
                }
            }
            Some((x_min, x_max, y_min, y_max))
        }
        Chart::Candlestick(c) => {
            if c.data.is_empty() {
                return None;
            }
            let x_min = c.data.iter().map(|e| e.x).fold(f64::INFINITY, f64::min);
            let x_max = c.data.iter().map(|e| e.x).fold(f64::NEG_INFINITY, f64::max);
            let y_min = c.data.iter().map(|e| e.low).fold(f64::INFINITY, f64::min);
            let y_max = c
                .data
                .iter()
                .map(|e| e.high)
                .fold(f64::NEG_INFINITY, f64::max);
            Some((x_min, x_max, y_min, y_max))
        }
        Chart::BoxPlot(_)
        | Chart::Heatmap(_)
        | Chart::Waterfall(_)
        | Chart::Lollipop(_)
        | Chart::Violin(_) => {
            // These types have non-trivial axis semantics; skip for now.
            None
        }
        // Types without conventional XY axes
        Chart::Pie(_)
        | Chart::Radar(_)
        | Chart::Sparkline(_)
        | Chart::Funnel(_)
        | Chart::Gauge(_) => None,
    }
}
