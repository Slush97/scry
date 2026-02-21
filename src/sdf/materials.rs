// SPDX-License-Identifier: MIT OR Apache-2.0
//! Materials for SDF objects: solid, water, and volumetric fire.

use crate::scene::style::Color;

/// Surface or volumetric material attached to an [`SdfObject`](super::SdfObject).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Material {
    /// Opaque surface with Phong shading.
    Solid {
        /// Base color.
        color: Color,
        /// Mirror reflectivity (0 = matte, 1 = perfect mirror).
        reflectivity: f32,
        /// Specular highlight power.
        specular: f32,
    },
    /// Animated water surface with Fresnel reflections.
    Water {
        /// Tint applied to the refracted color.
        tint: Color,
        /// Index of refraction (water ≈ 1.33).
        ior: f32,
        /// Wave amplitude in world units.
        amplitude: f32,
        /// Wave spatial frequency.
        frequency: f32,
    },
    /// Volumetric fire rendered by front-to-back ray marching through FBM noise.
    Fire {
        /// Overall brightness multiplier.
        intensity: f32,
        /// Scale of the noise field.
        noise_scale: f32,
        /// Animation speed multiplier.
        speed: f32,
    },
    /// Checkerboard pattern (two alternating colors on an infinite plane).
    Checkerboard {
        /// First color (e.g. white squares).
        color_a: Color,
        /// Second color (e.g. black squares).
        color_b: Color,
        /// World-space scale of each square (default 1.0).
        scale: f32,
        /// Mirror reflectivity (0 = matte, 1 = perfect mirror).
        reflectivity: f32,
        /// Specular highlight power.
        specular: f32,
    },
    /// Semi-transparent glass with Fresnel reflections and refraction.
    ///
    /// Works on any shape (spheres, boxes, etc.). Uses Schlick Fresnel
    /// to blend between reflection and refraction, with an optional tint
    /// applied to refracted light.
    Glass {
        /// Tint applied to refracted light (white = clear glass).
        tint: Color,
        /// Index of refraction (glass ≈ 1.5, diamond ≈ 2.42).
        ior: f32,
        /// Overall opacity: 0 = fully transparent, 1 = opaque tint only.
        opacity: f32,
        /// Chromatic dispersion: IOR offset per channel (0 = none).
        /// Red uses `ior - dispersion`, blue uses `ior + dispersion`.
        dispersion: f32,
    },
    /// Rainbow/spectral color mapped from angular position in local object space.
    ///
    /// Maps `atan2(local.z, local.x)` to a full HSL hue rotation.
    /// Ideal for flat discs or rings to create prismatic color wheels.
    Rainbow {
        /// Saturation (0–1, default 0.9).
        saturation: f32,
        /// Lightness (0–1, default 0.55).
        lightness: f32,
        /// Hue rotation offset in radians (animated by passing `time * speed`).
        hue_offset: f32,
        /// Specular highlight power.
        specular: f32,
    },
    /// Translucent material with subsurface scattering (jade, wax, marble, skin).
    ///
    /// Light bleeds through thin parts of the geometry, creating a warm
    /// back-illumination effect. Uses SDF thickness estimation — cheap and
    /// effective for any shape.
    Subsurface {
        /// Surface color.
        color: Color,
        /// Subsurface scatter color (light bleeding through thin areas).
        scatter_color: Color,
        /// Thickness scale factor (higher = more translucency, default 0.5).
        thickness: f32,
        /// Specular highlight power.
        specular: f32,
    },
}

impl Material {
    /// Convenience: matte solid with no reflection.
    pub fn matte(color: Color) -> Self {
        Self::Solid {
            color,
            reflectivity: 0.0,
            specular: 32.0,
        }
    }

    /// Convenience: mirror-like surface.
    pub fn mirror(color: Color, reflectivity: f32) -> Self {
        Self::Solid {
            color,
            reflectivity,
            specular: 64.0,
        }
    }

    /// Convenience: default water material.
    pub fn water() -> Self {
        Self::Water {
            tint: Color::from_rgba8(20, 60, 120, 255),
            ior: 1.33,
            amplitude: 0.08,
            frequency: 3.0,
        }
    }

    /// Convenience: default fire material.
    pub fn fire() -> Self {
        Self::Fire {
            intensity: 1.5,
            noise_scale: 2.5,
            speed: 1.0,
        }
    }

    /// Convenience: checkerboard pattern with two alternating colors.
    pub fn checkerboard(color_a: Color, color_b: Color) -> Self {
        Self::Checkerboard {
            color_a,
            color_b,
            scale: 1.0,
            reflectivity: 0.0,
            specular: 32.0,
        }
    }

    /// Convenience: clear glass with given IOR and optional tint.
    pub fn glass(tint: Color, ior: f32) -> Self {
        Self::Glass {
            tint,
            ior,
            opacity: 0.0,
            dispersion: 0.0,
        }
    }

    /// Convenience: glass with chromatic dispersion (prismatic edges).
    pub fn glass_dispersive(tint: Color, ior: f32, dispersion: f32) -> Self {
        Self::Glass {
            tint,
            ior,
            opacity: 0.0,
            dispersion,
        }
    }

    /// Convenience: rainbow/spectral material with default settings.
    pub fn rainbow() -> Self {
        Self::Rainbow {
            saturation: 0.9,
            lightness: 0.55,
            hue_offset: 0.0,
            specular: 32.0,
        }
    }

    /// Convenience: rainbow with animated hue offset.
    pub fn rainbow_animated(hue_offset: f32) -> Self {
        Self::Rainbow {
            saturation: 0.9,
            lightness: 0.55,
            hue_offset,
            specular: 32.0,
        }
    }

    /// Convenience: translucent subsurface scattering material.
    ///
    /// Great for jade, wax, marble, or skin-like surfaces.
    pub fn subsurface(color: Color, scatter_color: Color) -> Self {
        Self::Subsurface {
            color,
            scatter_color,
            thickness: 0.5,
            specular: 32.0,
        }
    }
}

/// Map a fire temperature `t` in `[0, 1]` to a color ramp:
/// black → dark red → red → orange → yellow → white.
pub fn fire_color_ramp(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let r = (t * 3.0).clamp(0.0, 1.0);
    let g = ((t - 0.33) * 3.0).clamp(0.0, 1.0);
    let b = ((t - 0.66) * 3.0).clamp(0.0, 1.0);
    Color { r, g, b, a: 1.0 }
}

/// Convert HSL to linear RGB color.
///
/// `h` is hue in `[0, 1]` (wraps), `s` is saturation `[0, 1]`, `l` is lightness `[0, 1]`.
pub fn hsl_to_color(h: f32, s: f32, l: f32) -> Color {
    let h = h.fract();
    let h = if h < 0.0 { h + 1.0 } else { h };
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if h6 < 1.0 {
        (c, x, 0.0)
    } else if h6 < 2.0 {
        (x, c, 0.0)
    } else if h6 < 3.0 {
        (0.0, c, x)
    } else if h6 < 4.0 {
        (0.0, x, c)
    } else if h6 < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c * 0.5;
    Color {
        r: (r1 + m).clamp(0.0, 1.0),
        g: (g1 + m).clamp(0.0, 1.0),
        b: (b1 + m).clamp(0.0, 1.0),
        a: 1.0,
    }
}

/// Schlick's approximation for Fresnel reflectance.
///
/// `cos_theta` is the cosine of the angle between the view direction and surface
/// normal. `ior` is the index of refraction of the material.
pub fn fresnel(cos_theta: f32, ior: f32) -> f32 {
    let r0 = ((1.0 - ior) / (1.0 + ior)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cos_theta.clamp(0.0, 1.0)).powi(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fire_ramp_endpoints() {
        let black = fire_color_ramp(0.0);
        assert!(black.r < 0.01 && black.g < 0.01 && black.b < 0.01);

        let white = fire_color_ramp(1.0);
        assert!(white.r > 0.99 && white.g > 0.99 && white.b > 0.99);
    }

    #[test]
    fn fire_ramp_monotonic() {
        let mut prev_r = 0.0_f32;
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let c = fire_color_ramp(t);
            assert!(c.r >= prev_r - 1e-6);
            prev_r = c.r;
        }
    }

    #[test]
    fn fresnel_at_normal_incidence() {
        // At normal incidence for glass (ior=1.5), R0 ≈ 0.04
        let f = fresnel(1.0, 1.5);
        assert!((f - 0.04).abs() < 0.01);
    }

    #[test]
    fn fresnel_at_grazing() {
        // At grazing angle, Fresnel → 1.0
        let f = fresnel(0.0, 1.5);
        assert!((f - 1.0).abs() < 0.01);
    }
}
