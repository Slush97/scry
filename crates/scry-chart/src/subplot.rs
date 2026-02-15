//! Subplot / multi-panel grid — compose multiple charts into a single image.
//!
//! The [`SubplotGrid`] builder lets you place individual [`Chart`]s into a
//! row × column grid, render them independently, and composite the results
//! into one output image. Each cell is rendered at its own resolution, so
//! different chart types can share a canvas without layout interference.
//!
//! # Example
//!
//! ```ignore
//! use scry_chart::{Chart, SubplotGrid};
//!
//! let grid = SubplotGrid::new(2, 2)
//!     .gap(12)
//!     .set(0, 0, Chart::line(&[1.0, 4.0, 2.0]).title("A").build())
//!     .set(0, 1, Chart::scatter(&[(1.0, 2.0)]).title("B").build())
//!     .set(1, 0, Chart::bar(vec!["x"], &[3.0]).title("C").build())
//!     .set(1, 1, Chart::histogram(&[1.0, 2.0, 3.0]).title("D").build());
//!
//! scry_chart::export::save_subplot_png(&grid, 1600, 1000, "subplot.png")?;
//! ```

use crate::chart::Chart;
use crate::theme::Theme;
use scry_engine::style::Color;

/// Controls axis coordination across [`SubplotGrid`] cells.
///
/// When axes are shared, the grid renders with unified domains and
/// suppresses redundant tick labels on interior panels for a cleaner
/// multi-panel layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SharedAxisMode {
    /// Each chart has independent axes (default).
    #[default]
    None,
    /// Charts in the same column share the X axis.
    /// Only the bottom row shows X tick labels.
    ShareX,
    /// Charts in the same row share the Y axis.
    /// Only the leftmost column shows Y tick labels.
    ShareY,
    /// Both X and Y axes are shared.
    ShareBoth,
}

impl SharedAxisMode {
    /// Whether the X axis is shared.
    #[must_use]
    pub fn shares_x(self) -> bool {
        matches!(self, Self::ShareX | Self::ShareBoth)
    }

    /// Whether the Y axis is shared.
    #[must_use]
    pub fn shares_y(self) -> bool {
        matches!(self, Self::ShareY | Self::ShareBoth)
    }
}

/// A grid of charts composited into a single image.
///
/// Charts are placed into cells via `set(row, col, chart)`. Empty cells
/// are filled with the grid's background color.
#[must_use]
pub struct SubplotGrid {
    /// Number of grid rows.
    pub rows: usize,
    /// Number of grid columns.
    pub cols: usize,
    /// Pixel gap between cells.
    pub gap: u32,
    /// Overall grid title (centered above all cells).
    pub title: Option<String>,
    /// Background color for the grid and empty cells.
    pub background: Color,
    /// Axis coordination mode.
    pub shared_axes: SharedAxisMode,
    /// Cell contents, stored row-major: `cells[row * cols + col]`.
    pub(crate) cells: Vec<Option<Chart>>,
}

impl SubplotGrid {
    /// Create a new `rows × cols` subplot grid.
    ///
    /// All cells start empty. Use [`set`](Self::set) to populate them.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "SubplotGrid requires at least 1x1");
        Self {
            rows,
            cols,
            gap: 8,
            title: None,
            background: Theme::dark().background,
            shared_axes: SharedAxisMode::None,
            cells: vec![None; rows * cols],
        }
    }

    /// Set the pixel gap between cells (default: 8).
    pub fn gap(mut self, gap: u32) -> Self {
        self.gap = gap;
        self
    }

    /// Set an overall title displayed above the grid.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the background color for the grid and empty cells.
    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Share the X axis across columns (only bottom row shows labels).
    pub fn share_x_axis(mut self) -> Self {
        self.shared_axes = match self.shared_axes {
            SharedAxisMode::ShareY => SharedAxisMode::ShareBoth,
            _ => SharedAxisMode::ShareX,
        };
        self
    }

    /// Share the Y axis across rows (only leftmost column shows labels).
    pub fn share_y_axis(mut self) -> Self {
        self.shared_axes = match self.shared_axes {
            SharedAxisMode::ShareX => SharedAxisMode::ShareBoth,
            _ => SharedAxisMode::ShareY,
        };
        self
    }

    /// Share both X and Y axes.
    pub fn share_both_axes(mut self) -> Self {
        self.shared_axes = SharedAxisMode::ShareBoth;
        self
    }

    /// Place a chart at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.rows` or `col >= self.cols`.
    pub fn set(mut self, row: usize, col: usize, chart: Chart) -> Self {
        assert!(
            row < self.rows && col < self.cols,
            "SubplotGrid::set({row}, {col}) out of bounds for {r}×{c} grid",
            r = self.rows,
            c = self.cols,
        );
        self.cells[row * self.cols + col] = Some(chart);
        self
    }

    /// Get the chart at `(row, col)`, if any.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&Chart> {
        self.cells.get(row * self.cols + col).and_then(|c| c.as_ref())
    }

    /// Iterator over `(row, col, Option<&Chart>)` for all cells.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, Option<&Chart>)> {
        let cols = self.cols;
        self.cells
            .iter()
            .enumerate()
            .map(move |(i, c)| (i / cols, i % cols, c.as_ref()))
    }

    /// Mutable access to the underlying cell storage.
    ///
    /// Used by the export layer to inject unified domain ranges when
    /// axes are shared.
    pub fn cells_mut(&mut self) -> &mut [Option<Chart>] {
        &mut self.cells
    }
}
