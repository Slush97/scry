//! Streaming chart with ring buffer backing and auto-scrolling time axis.
//!
//! Provides [`StreamingChart`] — a chart that accepts data via [`push()`] and
//! renders a sliding window of the most recent data points, backed by a
//! fixed-capacity ring buffer for O(1) append with bounded memory.
//!
//! # Example
//!
//! ```ignore
//! use scry_chart::streaming::StreamingChart;
//!
//! let mut chart = StreamingChart::new()
//!     .window_size(100)
//!     .title("CPU Usage")
//!     .y_range(0.0, 100.0);
//!
//! loop {
//!     let cpu = read_cpu_usage();
//!     chart.push_now(cpu);
//!     chart.render_inline(800, 400).unwrap();
//!     std::thread::sleep(std::time::Duration::from_millis(100));
//! }
//! ```

use crate::chart::{Chart, LineChart};
use crate::data::Series;
use crate::error::ChartError;
use crate::export::render_to_png;
use crate::theme::Theme;

// ---------------------------------------------------------------------------
// RingBuffer
// ---------------------------------------------------------------------------

/// Fixed-capacity ring buffer with O(1) push and ordered iteration.
#[derive(Clone, Debug)]
struct RingBuffer<T> {
    data: Vec<Option<T>>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(None);
        }
        Self {
            data,
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// Push an item. Overwrites the oldest element when full.
    fn push(&mut self, item: T) {
        self.data[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Iterate from oldest to newest.
    fn iter(&self) -> RingBufferIter<'_, T> {
        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        RingBufferIter {
            buf: self,
            pos: start,
            remaining: self.len,
        }
    }

    /// Current number of elements.
    fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer has reached capacity.
    #[cfg(test)]
    fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Remove all elements.
    #[cfg(test)]
    fn clear(&mut self) {
        for slot in &mut self.data {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }
}

/// Iterator over ring buffer elements, oldest to newest.
struct RingBufferIter<'a, T> {
    buf: &'a RingBuffer<T>,
    pos: usize,
    remaining: usize,
}

impl<'a, T> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = self.buf.data[self.pos].as_ref()?;
        self.pos = (self.pos + 1) % self.buf.capacity;
        self.remaining -= 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for RingBufferIter<'_, T> {}

// ---------------------------------------------------------------------------
// StreamingChart
// ---------------------------------------------------------------------------

/// A chart that accepts streaming data via `push()` and renders a sliding
/// window of the most recent data points.
///
/// Backed by a ring buffer for O(1) append with bounded memory.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct StreamingChart {
    /// Per-series ring buffers of `(x, y)` pairs.
    buffers: Vec<RingBuffer<(f64, f64)>>,
    /// Maximum number of points to display per series.
    window_size: usize,
    /// Chart title.
    title: Option<String>,
    /// Fixed Y-axis range (auto-scales if `None`).
    y_range: Option<(f64, f64)>,
    /// Number of series.
    n_series: usize,
    /// Visual theme.
    theme: Theme,
    /// Series labels (for legend).
    labels: Vec<String>,
}

impl StreamingChart {
    /// Create a new streaming chart with default settings.
    ///
    /// Defaults: 1 series, window size 100, auto-scaling Y axis, dark theme.
    #[must_use]
    pub fn new() -> Self {
        let window_size = 100;
        Self {
            buffers: vec![RingBuffer::new(window_size)],
            window_size,
            title: None,
            y_range: None,
            n_series: 1,
            theme: Theme::default(),
            labels: vec![String::new()],
        }
    }

    /// Set the window size (max points displayed per series). Default: 100.
    #[must_use]
    pub fn window_size(mut self, n: usize) -> Self {
        let n = n.max(1);
        self.window_size = n;
        self.buffers = (0..self.n_series).map(|_| RingBuffer::new(n)).collect();
        self
    }

    /// Set the chart title.
    #[must_use]
    pub fn title(mut self, t: impl Into<String>) -> Self {
        self.title = Some(t.into());
        self
    }

    /// Set fixed Y-axis range. If not set, auto-scales to visible data.
    #[must_use]
    pub fn y_range(mut self, min: f64, max: f64) -> Self {
        self.y_range = Some((min, max));
        self
    }

    /// Set the number of series. Default: 1.
    ///
    /// Resets all buffers when changed.
    #[must_use]
    pub fn n_series(mut self, n: usize) -> Self {
        let n = n.max(1);
        self.n_series = n;
        self.buffers = (0..n).map(|_| RingBuffer::new(self.window_size)).collect();
        self.labels.resize(n, String::new());
        self
    }

    /// Set the visual theme.
    #[must_use]
    pub fn theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    /// Set series labels (for legend).
    #[must_use]
    pub fn labels(mut self, labels: Vec<impl Into<String>>) -> Self {
        self.labels = labels.into_iter().map(Into::into).collect();
        self.labels.resize(self.n_series, String::new());
        self
    }

    /// Push a new data point to series 0.
    ///
    /// If the buffer is full, the oldest point is dropped.
    pub fn push(&mut self, x: f64, y: f64) {
        self.buffers[0].push((x, y));
    }

    /// Push a data point with a timestamp of "now" (convenience for time-series).
    pub fn push_now(&mut self, y: f64) {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.push(t, y);
    }

    /// Push a value to a specific series.
    ///
    /// # Panics
    ///
    /// Panics if `series >= self.n_series`.
    pub fn push_series(&mut self, series: usize, x: f64, y: f64) {
        self.buffers[series].push((x, y));
    }

    /// Total number of data points across all series.
    #[must_use]
    pub fn total_points(&self) -> usize {
        self.buffers.iter().map(|b| b.len()).sum()
    }

    /// Number of data points in series 0 (or specified series).
    #[must_use]
    pub fn points_in_series(&self, series: usize) -> usize {
        self.buffers.get(series).map_or(0, |b| b.len())
    }

    /// Build a snapshot `Chart` from the current buffer contents.
    ///
    /// Returns `None` if all buffers are empty (no data to render).
    #[must_use]
    pub fn snapshot(&self) -> Option<Chart> {
        // Collect all series data
        let mut all_x: Vec<Vec<f64>> = Vec::new();
        let mut all_y: Vec<Vec<f64>> = Vec::new();
        let mut any_data = false;

        for buf in &self.buffers {
            let (xs, ys): (Vec<f64>, Vec<f64>) = buf.iter().copied().unzip();
            if !xs.is_empty() {
                any_data = true;
            }
            all_x.push(xs);
            all_y.push(ys);
        }

        if !any_data {
            return None;
        }

        // Determine X range: union across all series
        let x_min = all_x
            .iter()
            .filter_map(|xs| xs.first().copied())
            .reduce(f64::min)?;
        let x_max = all_x
            .iter()
            .filter_map(|xs| xs.last().copied())
            .reduce(f64::max)?;

        // Build series
        let series_list: Vec<Series> = all_y
            .iter()
            .enumerate()
            .filter(|(_, ys)| !ys.is_empty())
            .map(|(i, ys)| {
                let label = self
                    .labels
                    .get(i)
                    .filter(|l| !l.is_empty())
                    .cloned()
                    .unwrap_or_else(|| {
                        if self.n_series > 1 {
                            format!("Series {}", i)
                        } else {
                            String::new()
                        }
                    });
                Series::new(label, ys.clone())
            })
            .collect();

        // Use the first non-empty series' x values for the shared axis
        let x_values = all_x.into_iter().find(|xs| !xs.is_empty())?;

        let mut line = LineChart::new(series_list)
            .x_values(x_values)
            .theme(self.theme.clone());

        if let Some(ref t) = self.title {
            line = line.title(t.clone());
        }

        // Set X range to the visible window
        line.config.axes.x_range = Some((x_min, x_max));

        // Set Y range
        if let Some((y_min, y_max)) = self.y_range {
            line.config.axes.y_range = Some((y_min, y_max));
        }

        Some(line.build())
    }

    /// Render the current window to PNG bytes.
    ///
    /// Returns `Ok(bytes)` on success, or `Err` if rendering fails.
    /// Returns an empty-data error if no points have been pushed.
    pub fn render(&self, width: u32, height: u32) -> Result<Vec<u8>, ChartError> {
        let chart = self.snapshot().ok_or(ChartError::EmptyData)?;
        render_to_png(&chart, width, height)
    }

    /// Render the current window to raw RGBA bytes.
    pub fn render_rgba(&self, width: u32, height: u32) -> Result<Vec<u8>, ChartError> {
        let chart = self.snapshot().ok_or(ChartError::EmptyData)?;
        crate::export::render_to_rgba(&chart, width, height)
    }

    /// Render and display inline in the terminal (one-shot).
    ///
    /// Renders the chart to PNG and displays it using the auto-detected
    /// terminal graphics protocol (Kitty/iTerm2).
    #[cfg(feature = "inline")]
    pub fn render_inline(&self, width: u32, height: u32) -> Result<(), ChartError> {
        let png = self.render(width, height)?;
        crate::inline::display_inline_auto(&png).map_err(|e| ChartError::Io(e.to_string()))
    }

    /// Render one frame of a live-updating chart.
    ///
    /// On `frame_number == 0`, displays normally. On subsequent frames,
    /// moves the cursor up to overwrite the previous image for smooth updates.
    #[cfg(feature = "inline")]
    pub fn render_frame(
        &self,
        width: u32,
        height: u32,
        frame_number: u64,
    ) -> Result<(), ChartError> {
        let png = self.render(width, height)?;
        crate::inline::display_frame(&png, height, frame_number)
            .map_err(|e| ChartError::Io(e.to_string()))
    }
}

impl Default for StreamingChart {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: RingBuffer basic — push 5 items into capacity-3 buffer,
    //         verify only last 3 remain
    #[test]
    fn ring_buffer_basic() {
        let mut buf = RingBuffer::new(3);
        for i in 0..5 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 3);
        let items: Vec<_> = buf.iter().copied().collect();
        assert_eq!(items, vec![2, 3, 4]);
    }

    // Test 2: RingBuffer iteration order — oldest to newest
    #[test]
    fn ring_buffer_iteration_order() {
        let mut buf = RingBuffer::new(5);
        for i in 0..5 {
            buf.push(i * 10);
        }
        let items: Vec<_> = buf.iter().copied().collect();
        assert_eq!(items, vec![0, 10, 20, 30, 40]);

        // Push two more to wrap around
        buf.push(50);
        buf.push(60);
        let items: Vec<_> = buf.iter().copied().collect();
        assert_eq!(items, vec![20, 30, 40, 50, 60]);
    }

    // Test 3: RingBuffer empty — iter returns nothing
    #[test]
    fn ring_buffer_empty() {
        let buf = RingBuffer::<i32>::new(10);
        assert_eq!(buf.len(), 0);
        assert!(!buf.is_full());
        assert_eq!(buf.iter().count(), 0);
    }

    // Test 4: StreamingChart push — push 10 points, verify snapshot produces a chart
    #[test]
    fn streaming_chart_push() {
        let mut chart = StreamingChart::new().window_size(50);
        for i in 0..10 {
            chart.push(i as f64, (i * i) as f64);
        }
        let snap = chart.snapshot();
        assert!(
            snap.is_some(),
            "snapshot should produce a chart after pushing data"
        );
    }

    // Test 5: StreamingChart window — push 200 points into window_size=100,
    //         verify only 100 visible
    #[test]
    fn streaming_chart_window() {
        let mut chart = StreamingChart::new().window_size(100);
        for i in 0..200 {
            chart.push(i as f64, i as f64);
        }
        assert_eq!(chart.points_in_series(0), 100);

        // The snapshot should have the last 100 points (100..200)
        let snap = chart.snapshot().unwrap();
        let cfg = snap.config().expect("snapshot should have config");
        // X range is set to the visible window
        let (x_min, x_max) = cfg.axes.x_range.expect("x_range should be set");
        assert!(
            (x_min - 100.0).abs() < 1e-9,
            "x_min should be 100, got {}",
            x_min
        );
        assert!(
            (x_max - 199.0).abs() < 1e-9,
            "x_max should be 199, got {}",
            x_max
        );
    }

    // Test 6: StreamingChart auto-scale Y — push varying values,
    //         verify Y range covers min/max of visible window
    #[test]
    fn streaming_chart_auto_scale_y() {
        let mut chart = StreamingChart::new().window_size(50);
        for i in 0..50 {
            let y = if i % 2 == 0 { -10.0 } else { 42.0 };
            chart.push(i as f64, y);
        }
        let snap = chart.snapshot().unwrap();
        let cfg = snap.config().expect("snapshot should have config");
        // No y_range set = auto-scale
        assert!(cfg.axes.y_range.is_none());
        // Verify data extent covers the expected range
        let extent = snap.data_extent().expect("should have data extent");
        assert!((extent.2 - (-10.0)).abs() < 1e-9, "y_min should be -10");
        assert!((extent.3 - 42.0).abs() < 1e-9, "y_max should be 42");
    }

    // Test 7: StreamingChart fixed Y range — set y_range, verify it's respected
    #[test]
    fn streaming_chart_fixed_y_range() {
        let mut chart = StreamingChart::new().window_size(10).y_range(0.0, 100.0);
        chart.push(1.0, 50.0);

        let snap = chart.snapshot().unwrap();
        let cfg = snap.config().expect("snapshot should have config");
        assert_eq!(cfg.axes.y_range, Some((0.0, 100.0)));
    }

    // Test 8: Multi-series — push to 3 series, verify all render
    #[test]
    fn multi_series() {
        let mut chart = StreamingChart::new()
            .n_series(3)
            .window_size(20)
            .labels(vec!["CPU", "Mem", "Disk"]);

        for i in 0..10 {
            let x = i as f64;
            chart.push_series(0, x, i as f64 * 1.0);
            chart.push_series(1, x, i as f64 * 2.0);
            chart.push_series(2, x, i as f64 * 3.0);
        }

        assert_eq!(chart.points_in_series(0), 10);
        assert_eq!(chart.points_in_series(1), 10);
        assert_eq!(chart.points_in_series(2), 10);
        assert_eq!(chart.total_points(), 30);

        // Verify snapshot is produced and renders without panic
        let snap = chart.snapshot().unwrap();
        // data_extent reflects all 3 series (y_max = 9*3 = 27 for series 2)
        let (_, _, _y_min, y_max) = snap.data_extent().expect("extent should be Some");
        assert!(
            y_max >= 27.0 - 1e-6,
            "y_max {y_max} should reflect 3rd series (max = 27)"
        );
        // Render must not panic and must produce a correctly-sized canvas
        let rendered = snap.render(400, 300);
        assert_eq!(rendered.canvas.width(), 400, "canvas width should be 400");
        assert_eq!(rendered.canvas.height(), 300, "canvas height should be 300");
    }

    // Test 9: push_now — verify timestamp is reasonable (within last second)
    #[test]
    fn push_now_timestamp() {
        let mut chart = StreamingChart::new().window_size(5);
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        chart.push_now(42.0);

        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Extract the x value from the buffer
        let (x, y) = *chart.buffers[0].iter().next().unwrap();
        assert!((y - 42.0).abs() < 1e-9);
        assert!(
            x >= before && x <= after,
            "timestamp {x} should be between {before} and {after}"
        );
    }

    // Test 10: Empty chart render — render with 0 points should return EmptyData error
    #[test]
    fn empty_chart_render() {
        let chart = StreamingChart::new();
        let result = chart.render(800, 400);
        assert!(result.is_err(), "rendering empty chart should fail");
    }

    // Bonus: RingBuffer clear
    #[test]
    fn ring_buffer_clear() {
        let mut buf = RingBuffer::new(5);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.iter().count(), 0);
    }

    // Bonus: RingBuffer is_full
    #[test]
    fn ring_buffer_is_full() {
        let mut buf = RingBuffer::new(3);
        assert!(!buf.is_full());
        buf.push(1);
        buf.push(2);
        assert!(!buf.is_full());
        buf.push(3);
        assert!(buf.is_full());
        buf.push(4); // overwrites oldest
        assert!(buf.is_full());
        assert_eq!(buf.len(), 3);
    }

    // Bonus: Default trait
    #[test]
    fn default_streaming_chart() {
        let chart = StreamingChart::default();
        assert_eq!(chart.window_size, 100);
        assert_eq!(chart.n_series, 1);
        assert!(chart.title.is_none());
        assert!(chart.y_range.is_none());
    }

    // Bonus: snapshot returns None for empty chart
    #[test]
    fn snapshot_empty_returns_none() {
        let chart = StreamingChart::new();
        assert!(chart.snapshot().is_none());
    }
}
