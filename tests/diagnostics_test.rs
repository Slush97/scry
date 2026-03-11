//! Integration tests for the scry-engine diagnostics suite.
//!
//! Tests scene validation, error chain propagation, engine report generation,
//! and macro compilation.

use scry_engine::diagnostics::EngineReport;
use scry_engine::scene::style::{Color, ShapeStyle};
use scry_engine::scene::validate::{validate_scene, WarningSeverity};
use scry_engine::scene::{DrawCommand, PixelCanvas};
use scry_engine::PixelCanvasError;

// ---------------------------------------------------------------------------
// Scene validation
// ---------------------------------------------------------------------------

#[test]
fn validate_degenerate_circle() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, 0.0)
        .fill(Color::RED)
        .done();

    let warnings = validate_scene(&canvas);
    assert!(
        warnings
            .iter()
            .any(|w| w.severity == WarningSeverity::Warning
                && w.message.contains("zero or negative radius")),
        "Expected warning about zero-radius circle, got: {warnings:?}",
    );
}

#[test]
fn validate_zero_area_rect() {
    let canvas = PixelCanvas::new(100, 100)
        .rect(10.0, 10.0, 0.0, 50.0)
        .fill(Color::BLUE)
        .done();

    let warnings = validate_scene(&canvas);
    assert!(
        warnings
            .iter()
            .any(|w| w.message.contains("zero or negative dimensions")),
        "Expected rect dimension warning, got: {warnings:?}",
    );
}

#[test]
fn validate_empty_polyline() {
    let canvas = PixelCanvas::from_commands(
        100,
        100,
        vec![DrawCommand::Polyline {
            points: vec![],
            closed: false,
            style: ShapeStyle::default(),
        }],
        Color::TRANSPARENT,
    );

    let warnings = validate_scene(&canvas);
    assert!(
        warnings
            .iter()
            .any(|w| w.severity == WarningSeverity::Error
                && w.message.contains("fewer than 2 points")),
        "Expected error for empty polyline, got: {warnings:?}",
    );
}

#[test]
fn validate_valid_scene_is_clean() {
    let canvas = PixelCanvas::new(200, 200)
        .background(Color::BLACK)
        .circle(100.0, 100.0, 50.0)
        .fill(Color::RED)
        .done()
        .rect(10.0, 10.0, 80.0, 40.0)
        .fill(Color::GREEN)
        .done()
        .line(0.0, 0.0, 200.0, 200.0)
        .color(Color::WHITE)
        .width(2.0)
        .done();

    let warnings = validate_scene(&canvas);
    assert!(
        warnings.is_empty(),
        "Expected no warnings for a valid scene, got: {warnings:?}",
    );
}

// ---------------------------------------------------------------------------
// Engine report
// ---------------------------------------------------------------------------

#[test]
fn engine_report_snapshot() {
    let report = EngineReport::snapshot();

    // Should always produce a non-empty display string
    let s = format!("{report}");
    assert!(s.contains("GPU available"));
    assert!(s.contains("Features"));
    assert!(s.contains("Scene warnings"));
}

#[test]
fn engine_report_for_canvas_includes_warnings() {
    let canvas = PixelCanvas::new(100, 100)
        .circle(50.0, 50.0, -5.0) // negative radius
        .fill(Color::RED)
        .done();

    let report = EngineReport::for_canvas(&canvas);
    assert!(
        !report.scene_warnings.is_empty(),
        "Expected scene warnings in the report",
    );
    // The report display should mention the warning count
    let s = format!("{report}");
    assert!(s.contains("1"), "Expected '1' warning in display: {s}");
}

// ---------------------------------------------------------------------------
// Error chain propagation
// ---------------------------------------------------------------------------

#[test]
fn transport_error_into_pixel_canvas_error() {
    let transport_err =
        scry_engine::transport::error::TransportError::UnsupportedProtocol("test".into());
    let canvas_err: PixelCanvasError = transport_err.into();

    // Should be the TransportLayer variant
    let msg = format!("{canvas_err}");
    assert!(
        msg.contains("test"),
        "TransportError context should be preserved: {msg}",
    );
}

#[test]
fn io_error_into_pixel_canvas_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "test pipe");
    let canvas_err: PixelCanvasError = io_err.into();

    let msg = format!("{canvas_err}");
    assert!(
        msg.contains("test pipe"),
        "io::Error message should be preserved: {msg}",
    );
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_error_into_pixel_canvas_error() {
    let gpu_err = scry_engine::gpu::GpuError::NoAdapter;
    let canvas_err: PixelCanvasError = gpu_err.into();

    let msg = format!("{canvas_err}");
    assert!(
        msg.contains("adapter") || msg.contains("Adapter"),
        "GpuError should propagate: {msg}",
    );
}

// ---------------------------------------------------------------------------
// Macro compilation smoke test
// ---------------------------------------------------------------------------

#[test]
fn logging_macros_compile() {
    // These should compile and not panic regardless of features
    scry_engine::scry_warn!("diagnostics test: warn level");
    scry_engine::scry_info!("diagnostics test: info level");
    scry_engine::scry_error!("diagnostics test: error level");
    scry_engine::scry_debug!("diagnostics test: debug level");
    scry_engine::scry_trace!("diagnostics test: trace level");
}
