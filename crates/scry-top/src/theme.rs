use scry_chart::prelude::Theme;
use scry_engine::style::Color;

/// "Monitor" dark theme — near-black background, muted neons.
pub fn monitor_theme() -> Theme {
    Theme::dark()
        .with_palette(vec![
            Color::from_rgba8(80, 160, 255, 255),  // blue
            Color::from_rgba8(100, 220, 160, 255), // green
            Color::from_rgba8(255, 180, 80, 255),  // amber
            Color::from_rgba8(220, 100, 255, 255), // purple
            Color::from_rgba8(255, 100, 120, 255), // coral
            Color::from_rgba8(100, 220, 220, 255), // cyan
            Color::from_rgba8(220, 220, 100, 255), // yellow
            Color::from_rgba8(180, 130, 255, 255), // lavender
        ])
        .with_grid(|g| {
            g.color = Color::from_rgba8(40, 40, 60, 80);
            g.width = 0.5;
        })
        .with_series(|s| {
            s.fill_opacity = 0.15;
            s.line_width = 1.5;
        })
}

/// CPU core colors — HSL hue rotation from blue through green to amber.
pub fn cpu_core_colors(n: usize) -> Vec<Color> {
    (0..n)
        .map(|i| {
            let hue = 200.0 + (i as f32 / n.max(1) as f32) * 160.0;
            Color::from_hsl(hue % 360.0, 0.7, 0.55)
        })
        .collect()
}

/// Memory panel colors.
pub fn memory_colors() -> (Color, Color) {
    (
        Color::from_rgba8(80, 140, 255, 255),  // RAM — blue
        Color::from_rgba8(180, 100, 255, 255), // Swap — purple
    )
}

/// Network panel colors.
pub fn network_colors() -> (Color, Color) {
    (
        Color::from_rgba8(100, 220, 160, 255), // rx — green
        Color::from_rgba8(255, 120, 100, 255), // tx — coral
    )
}
