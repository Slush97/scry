//! 3D scene graph for visualization.
//!
//! Provides [`Scene3D`] — a collection of 3D drawing primitives (point clouds,
//! line segments, labels) that can be projected and rasterized. The scene graph
//! is renderer-agnostic; it contains geometry data only.

use super::camera::Vec3;
use scry_engine::style::Color;

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// A collection of 3D points forming a scatter or point cloud.
#[derive(Clone, Debug)]
pub struct PointCloud3D {
    /// 3D positions of each point.
    pub points: Vec<Vec3>,
    /// Per-point colors. Length must match `points`.
    pub colors: Vec<Color>,
    /// Per-point radii in screen pixels. Length must match `points`.
    pub sizes: Vec<f32>,
    /// Optional series label (used in legends).
    pub label: Option<String>,
}

/// A line segment in 3D space.
#[derive(Clone, Debug)]
pub struct LineSegment3D {
    /// Start point.
    pub start: Vec3,
    /// End point.
    pub end: Vec3,
    /// Line color.
    pub color: Color,
    /// Line width in screen pixels.
    pub width: f32,
}

/// A text label positioned in 3D space.
#[derive(Clone, Debug)]
pub struct Label3D {
    /// 3D position of the label anchor.
    pub position: Vec3,
    /// Text content.
    pub text: String,
    /// Text color.
    pub color: Color,
    /// Font size in screen pixels.
    pub font_size: f32,
    /// Whether the label always faces the camera (billboard mode).
    pub billboard: bool,
}

// ---------------------------------------------------------------------------
// AxisConfig3D
// ---------------------------------------------------------------------------

/// Configuration for 3D axis rendering.
#[derive(Clone, Debug)]
pub struct AxisConfig3D {
    /// X-axis label.
    pub x_label: String,
    /// Y-axis label.
    pub y_label: String,
    /// Z-axis label.
    pub z_label: String,
    /// Whether to draw grid lines on the XZ plane.
    pub show_grid: bool,
    /// Color of grid lines.
    pub grid_color: Color,
    /// Color of axis lines (spine).
    pub axis_color: Color,
    /// Color of axis labels and tick labels.
    pub label_color: Color,
    /// Number of tick marks per axis.
    pub tick_count: usize,
    /// Minimum corner of the normalized bounding box (scene space).
    pub min: Vec3,
    /// Maximum corner of the normalized bounding box (scene space).
    pub max: Vec3,
    /// Original data minimum (for real-value tick labels).
    /// When set, tick labels show values mapped from this range.
    pub data_min: Option<Vec3>,
    /// Original data maximum (for real-value tick labels).
    pub data_max: Option<Vec3>,
}

impl Default for AxisConfig3D {
    fn default() -> Self {
        Self {
            x_label: "X".into(),
            y_label: "Y".into(),
            z_label: "Z".into(),
            show_grid: true,
            grid_color: Color::from_rgba8(80, 80, 100, 120),
            axis_color: Color::from_rgba8(180, 180, 200, 255),
            label_color: Color::from_rgba8(220, 220, 240, 255),
            tick_count: 5,
            min: Vec3::ZERO,
            max: Vec3::new(1.0, 1.0, 1.0),
            data_min: None,
            data_max: None,
        }
    }
}

/// Format a tick value with appropriate decimal places based on range.
fn format_tick(val: f64, range: f64) -> String {
    if range >= 100.0 {
        format!("{:.0}", val)
    } else if range >= 1.0 {
        format!("{:.1}", val)
    } else {
        format!("{:.2}", val)
    }
}

// ---------------------------------------------------------------------------
// Scene3D
// ---------------------------------------------------------------------------

/// A 3D scene containing geometry to be projected and rendered.
///
/// The scene graph is independent of any rendering backend — it stores
/// only geometry, colors, and labels. Rendering is done by a
/// [`Rasterizer3D`](super::Rasterizer3D) implementation.
#[derive(Clone, Debug)]
pub struct Scene3D {
    /// Point cloud collections.
    pub point_clouds: Vec<PointCloud3D>,
    /// Line segments (axis lines, grid lines, etc.).
    pub line_segments: Vec<LineSegment3D>,
    /// Billboard text labels.
    pub labels: Vec<Label3D>,
    /// Background color.
    pub background: Color,
    /// Optional chart title.
    pub title: Option<String>,
}

impl Scene3D {
    /// Create an empty scene with the given background color.
    #[must_use]
    pub fn new(background: Color) -> Self {
        Self {
            point_clouds: Vec::new(),
            line_segments: Vec::new(),
            labels: Vec::new(),
            background,
            title: None,
        }
    }

    /// Add a point cloud to the scene.
    pub fn add_point_cloud(&mut self, cloud: PointCloud3D) {
        self.point_clouds.push(cloud);
    }

    /// Add a line segment to the scene.
    pub fn add_line_segment(&mut self, segment: LineSegment3D) {
        self.line_segments.push(segment);
    }

    /// Add a label to the scene.
    pub fn add_label(&mut self, label: Label3D) {
        self.labels.push(label);
    }

    /// Generate axis lines, grid planes, tick marks, and labels from config.
    ///
    /// This populates the scene's `line_segments` and `labels` with the
    /// 3D axis system including:
    /// - 3 axis spine lines (X, Y, Z)
    /// - Grid lines on the XZ plane (at y = min.y)
    /// - Tick marks on each axis
    /// - Axis endpoint labels
    pub fn build_axes(&mut self, config: &AxisConfig3D) {
        let min = config.min;
        let max = config.max;

        // --- Axis spine lines ---
        // X axis
        self.line_segments.push(LineSegment3D {
            start: Vec3::new(min.x, min.y, min.z),
            end: Vec3::new(max.x, min.y, min.z),
            color: config.axis_color,
            width: 1.5,
        });
        // Y axis
        self.line_segments.push(LineSegment3D {
            start: Vec3::new(min.x, min.y, min.z),
            end: Vec3::new(min.x, max.y, min.z),
            color: config.axis_color,
            width: 1.5,
        });
        // Z axis
        self.line_segments.push(LineSegment3D {
            start: Vec3::new(min.x, min.y, min.z),
            end: Vec3::new(min.x, min.y, max.z),
            color: config.axis_color,
            width: 1.5,
        });

        let tick_count = config.tick_count.max(2);

        // --- Grid lines on XZ plane (y = min.y) ---
        if config.show_grid {
            for i in 0..=tick_count {
                let t = i as f32 / tick_count as f32;

                // Lines parallel to Z at various X positions
                let x = min.x + (max.x - min.x) * t;
                self.line_segments.push(LineSegment3D {
                    start: Vec3::new(x, min.y, min.z),
                    end: Vec3::new(x, min.y, max.z),
                    color: config.grid_color,
                    width: 0.5,
                });

                // Lines parallel to X at various Z positions
                let z = min.z + (max.z - min.z) * t;
                self.line_segments.push(LineSegment3D {
                    start: Vec3::new(min.x, min.y, z),
                    end: Vec3::new(max.x, min.y, z),
                    color: config.grid_color,
                    width: 0.5,
                });
            }
        }

        // --- Tick labels ---
        let label_offset = (max.x - min.x).max(max.z - min.z) * 0.08;

        // Data ranges for real-value tick labels
        let d_min = config.data_min.unwrap_or(min);
        let d_max = config.data_max.unwrap_or(max);
        let x_range = (d_max.x - d_min.x) as f64;
        let y_range = (d_max.y - d_min.y) as f64;
        let z_range = (d_max.z - d_min.z) as f64;

        for i in 0..=tick_count {
            let t = i as f32 / tick_count as f32;

            // X-axis ticks
            let x_pos = min.x + (max.x - min.x) * t;
            let x_data = f64::from(d_min.x) + x_range * f64::from(t);
            self.labels.push(Label3D {
                position: Vec3::new(x_pos, min.y - label_offset, min.z - label_offset),
                text: format_tick(x_data, x_range),
                color: config.label_color,
                font_size: 10.0,
                billboard: true,
            });

            // Y-axis ticks
            let y_pos = min.y + (max.y - min.y) * t;
            let y_data = f64::from(d_min.y) + y_range * f64::from(t);
            self.labels.push(Label3D {
                position: Vec3::new(min.x - label_offset, y_pos, min.z - label_offset),
                text: format_tick(y_data, y_range),
                color: config.label_color,
                font_size: 10.0,
                billboard: true,
            });

            // Z-axis ticks
            let z_pos = min.z + (max.z - min.z) * t;
            let z_data = f64::from(d_min.z) + z_range * f64::from(t);
            self.labels.push(Label3D {
                position: Vec3::new(min.x - label_offset, min.y - label_offset, z_pos),
                text: format_tick(z_data, z_range),
                color: config.label_color,
                font_size: 10.0,
                billboard: true,
            });
        }

        // --- Axis endpoint labels ---
        let axis_label_offset = label_offset * 2.5;

        self.labels.push(Label3D {
            position: Vec3::new(max.x + axis_label_offset, min.y, min.z),
            text: config.x_label.clone(),
            color: config.label_color,
            font_size: 13.0,
            billboard: true,
        });
        self.labels.push(Label3D {
            position: Vec3::new(min.x, max.y + axis_label_offset, min.z),
            text: config.y_label.clone(),
            color: config.label_color,
            font_size: 13.0,
            billboard: true,
        });
        self.labels.push(Label3D {
            position: Vec3::new(min.x, min.y, max.z + axis_label_offset),
            text: config.z_label.clone(),
            color: config.label_color,
            font_size: 13.0,
            billboard: true,
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_starts_empty() {
        let scene = Scene3D::new(Color::BLACK);
        assert!(scene.point_clouds.is_empty());
        assert!(scene.line_segments.is_empty());
        assert!(scene.labels.is_empty());
    }

    #[test]
    fn add_point_cloud() {
        let mut scene = Scene3D::new(Color::BLACK);
        scene.add_point_cloud(PointCloud3D {
            points: vec![Vec3::ZERO, Vec3::X],
            colors: vec![Color::RED, Color::BLUE],
            sizes: vec![3.0, 3.0],
            label: Some("test".into()),
        });
        assert_eq!(scene.point_clouds.len(), 1);
        assert_eq!(scene.point_clouds[0].points.len(), 2);
    }

    #[test]
    fn build_axes_generates_geometry() {
        let mut scene = Scene3D::new(Color::BLACK);
        let config = AxisConfig3D {
            tick_count: 5,
            show_grid: true,
            ..Default::default()
        };
        scene.build_axes(&config);

        // 3 axis spines + grid lines (6 per tick_count+1 = 12 lines)
        assert!(
            scene.line_segments.len() >= 3 + 12,
            "expected at least 15 line segments, got {}",
            scene.line_segments.len()
        );

        // Tick labels: 3 axes × (tick_count + 1) + 3 axis labels
        let expected_labels = 3 * (5 + 1) + 3;
        assert_eq!(
            scene.labels.len(),
            expected_labels,
            "expected {} labels, got {}",
            expected_labels,
            scene.labels.len()
        );
    }

    #[test]
    fn build_axes_no_grid() {
        let mut scene = Scene3D::new(Color::BLACK);
        let config = AxisConfig3D {
            show_grid: false,
            tick_count: 3,
            ..Default::default()
        };
        scene.build_axes(&config);

        // Only 3 axis spines, no grid lines
        assert_eq!(
            scene.line_segments.len(),
            3,
            "without grid should have exactly 3 axis lines"
        );
    }

    #[test]
    fn axis_labels_are_billboard() {
        let mut scene = Scene3D::new(Color::BLACK);
        scene.build_axes(&AxisConfig3D::default());

        for label in &scene.labels {
            assert!(label.billboard, "all axis labels should be billboard");
        }
    }
}
