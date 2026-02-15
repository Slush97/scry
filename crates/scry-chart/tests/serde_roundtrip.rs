//! Round-trip serialization tests for chart configuration types.
//!
//! These tests verify that all config types survive a JSON round-trip
//! when the `serde` feature is enabled.

#![cfg(feature = "serde")]

use scry_chart::annotation::Annotation;
use scry_chart::axis::LabelRotation;
use scry_chart::chart::{ChartConfig, ReferenceLine};
use scry_chart::formatter::LocaleConfig;
use scry_chart::legend::{LegendConfig, LegendOrientation, LegendPosition, SwatchShape};
use scry_chart::margin::Margin;
use scry_chart::theme::Theme;
use scry_engine::style::Color;

/// Helper: serialize to JSON, then deserialize back.
fn roundtrip_json<T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug>(
    val: &T,
) -> T {
    let json = serde_json::to_string_pretty(val).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn theme_roundtrip() {
    let theme = Theme::dark();
    let back = roundtrip_json(&theme);
    assert_eq!(back.palette.len(), theme.palette.len());
}

#[test]
fn chart_config_roundtrip() {
    let mut cfg = ChartConfig::default();
    cfg.title = Some("Test Chart".to_string());
    cfg.subtitle = Some("Subtitle".to_string());
    cfg.x_label = Some("X".to_string());
    cfg.y_label = Some("Y".to_string());
    cfg.dpi = 288;
    cfg.show_legend = false;
    cfg.x_range = Some((0.0, 100.0));
    cfg.y_range = Some((-5.0, 50.0));
    cfg.show_trend = true;
    cfg.x_tick_rotation = LabelRotation::Diagonal;

    let back = roundtrip_json(&cfg);
    assert_eq!(back.title.as_deref(), Some("Test Chart"));
    assert_eq!(back.subtitle.as_deref(), Some("Subtitle"));
    assert_eq!(back.dpi, 288);
    assert_eq!(back.show_legend, false);
    assert_eq!(back.x_range, Some((0.0, 100.0)));
    assert_eq!(back.show_trend, true);

    // Formatter fields should be None after roundtrip (they are skipped)
    assert!(back.x_tick_formatter.is_none());
    assert!(back.y_tick_formatter.is_none());
    assert!(back.secondary_y_formatter.is_none());
}

#[test]
fn reference_line_roundtrip() {
    let line = ReferenceLine::new(42.0)
        .color(Color::from_rgba8(255, 0, 0, 200))
        .label("threshold");
    let back = roundtrip_json(&line);
    assert_eq!(back.value, 42.0);
    assert_eq!(back.label.as_deref(), Some("threshold"));
    assert_eq!(back.color, Color::from_rgba8(255, 0, 0, 200));
}

#[test]
fn legend_config_roundtrip() {
    let mut legend = LegendConfig::default();
    legend.position = LegendPosition::BottomLeft;
    legend.swatch_shape = SwatchShape::Circle;
    legend.orientation = LegendOrientation::Horizontal;
    legend.columns = 3;
    legend.title = Some("Legend".to_string());

    let back = roundtrip_json(&legend);
    assert_eq!(back.position, LegendPosition::BottomLeft);
    assert_eq!(back.swatch_shape, SwatchShape::Circle);
    assert_eq!(back.orientation, LegendOrientation::Horizontal);
    assert_eq!(back.columns, 3);
    assert_eq!(back.title.as_deref(), Some("Legend"));
}

#[test]
fn annotation_roundtrip() {
    let ann = Annotation::new(1.0, 2.0, "Peak")
        .with_arrow()
        .with_offset(5.0, -10.0);
    let back = roundtrip_json(&ann);
    assert_eq!(back.x, 1.0);
    assert_eq!(back.y, 2.0);
    assert_eq!(back.text, "Peak");
    assert!(back.arrow);
}

#[test]
fn margin_roundtrip() {
    let m = Margin::new(10.0, 20.0, 30.0, 40.0);
    let back = roundtrip_json(&m);
    assert_eq!(back, m);
}

#[test]
fn locale_config_roundtrip() {
    let locale = LocaleConfig::european();
    let back = roundtrip_json(&locale);
    assert_eq!(back.decimal_separator, ',');
    assert_eq!(back.thousands_separator, '.');
}

#[test]
fn color_roundtrip() {
    let c = Color::from_rgba8(128, 64, 32, 200);
    let back = roundtrip_json(&c);
    assert_eq!(back, c);
}

#[test]
fn label_rotation_roundtrip() {
    for rot in [
        LabelRotation::Horizontal,
        LabelRotation::Diagonal,
        LabelRotation::Vertical,
        LabelRotation::Angle(37.5),
    ] {
        let back = roundtrip_json(&rot);
        assert_eq!(back, rot);
    }
}

#[test]
fn config_with_reference_lines_and_annotations() {
    let mut cfg = ChartConfig::default();
    cfg.h_lines.push(ReferenceLine::new(50.0).label("median"));
    cfg.v_lines.push(ReferenceLine::new(10.0));
    cfg.annotations
        .push(Annotation::new(5.0, 50.0, "Important"));
    cfg.locale = Some(LocaleConfig::swiss());
    cfg.margin = Some(Margin::new(5.0, 10.0, 5.0, 10.0));

    let back = roundtrip_json(&cfg);
    assert_eq!(back.h_lines.len(), 1);
    assert_eq!(back.v_lines.len(), 1);
    assert_eq!(back.annotations.len(), 1);
    assert!(back.locale.is_some());
    assert!(back.margin.is_some());
}
