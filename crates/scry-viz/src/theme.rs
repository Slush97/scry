//! Color palettes for the visualizers.

use scry_engine::style::Color;

pub struct Theme {
    pub name: &'static str,
    pub bg: Color,
    pub accent: Color,
    stops: &'static [(u8, u8, u8)],
}

impl Theme {
    /// Sample the palette gradient at `t` in 0..1 (oklab interpolation).
    pub fn sample(&self, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0) * (self.stops.len() - 1) as f32;
        let i = (t as usize).min(self.stops.len() - 2);
        let (r0, g0, b0) = self.stops[i];
        let (r1, g1, b1) = self.stops[i + 1];
        Color::from_rgba8(r0, g0, b0, 255).mix(Color::from_rgba8(r1, g1, b1, 255), t - i as f32)
    }
}

pub const THEMES: &[Theme] = &[
    Theme {
        name: "neon",
        bg: Color::from_rgba8(8, 8, 16, 255),
        accent: Color::from_rgba8(255, 60, 220, 255),
        stops: &[
            (0, 240, 255),
            (80, 120, 255),
            (200, 60, 255),
            (255, 60, 180),
        ],
    },
    Theme {
        name: "aurora",
        bg: Color::from_rgba8(6, 10, 14, 255),
        accent: Color::from_rgba8(120, 255, 190, 255),
        stops: &[
            (40, 220, 130),
            (60, 230, 200),
            (90, 160, 255),
            (170, 110, 255),
        ],
    },
    Theme {
        name: "sunset",
        bg: Color::from_rgba8(14, 8, 12, 255),
        accent: Color::from_rgba8(255, 140, 70, 255),
        stops: &[
            (255, 220, 90),
            (255, 140, 60),
            (255, 70, 90),
            (160, 60, 200),
        ],
    },
    Theme {
        name: "matrix",
        bg: Color::from_rgba8(4, 8, 4, 255),
        accent: Color::from_rgba8(140, 255, 140, 255),
        stops: &[(20, 90, 30), (30, 180, 60), (90, 255, 110), (210, 255, 190)],
    },
    Theme {
        name: "ice",
        bg: Color::from_rgba8(8, 10, 16, 255),
        accent: Color::from_rgba8(160, 220, 255, 255),
        stops: &[
            (60, 100, 200),
            (90, 170, 240),
            (150, 220, 255),
            (240, 250, 255),
        ],
    },
    Theme {
        name: "ember",
        bg: Color::from_rgba8(12, 6, 4, 255),
        accent: Color::from_rgba8(255, 120, 40, 255),
        stops: &[(80, 20, 10), (200, 60, 20), (255, 140, 40), (255, 230, 120)],
    },
];
