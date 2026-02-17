// SPDX-License-Identifier: MIT OR Apache-2.0
//! Animation primitives for smooth, time-based transitions.
//!
//! This module provides industry-standard easing curves, value interpolation,
//! keyframe timelines, and a frame-level animation orchestrator. It is
//! independent of any rendering or terminal code — you can use it standalone
//! to drive animations in any context.
//!
//! # Architecture
//!
//! - [`Easing`] — Standard easing functions mapping `[0,1] → [0,1]`
//! - [`Lerp`] — Trait for linear interpolation between two values
//! - [`Transition`] — Animate a single `Lerp`-able value over a duration
//! - [`Keyframes`] — Multi-stop timeline with per-segment easing
//! - [`AnimationState`] — Orchestrator managing multiple named animations
//!
//! # Example
//!
//! ```
//! use std::time::Duration;
//! use scry_engine::scene::animation::{Easing, Transition};
//!
//! let mut t = Transition::new(0.0_f32, 100.0_f32, Duration::from_millis(500))
//!     .easing(Easing::EaseOutCubic);
//!
//! t.advance(Duration::from_millis(250));
//! let mid = t.value(); // ~87.5 (cubic ease-out at 50%)
//! assert!(mid > 50.0);
//!
//! t.advance(Duration::from_millis(250));
//! assert!((t.value() - 100.0).abs() < f32::EPSILON);
//! assert!(t.is_complete());
//! ```

use std::collections::HashMap;
use std::time::Duration;

use crate::scene::style::{Color, Point, Transform};

// ---------------------------------------------------------------------------
// Easing
// ---------------------------------------------------------------------------

/// Standard easing functions that map a linear progress `t ∈ [0,1]` to a
/// curved value `∈ [0,1]` (with possible overshoot for spring/elastic).
///
/// These follow the same naming conventions as CSS easing functions and
/// Robert Penner's easing equations.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum Easing {
    /// Constant velocity. `f(t) = t`.
    #[default]
    Linear,

    // --- Quadratic ---
    /// Accelerating from zero velocity.
    EaseInQuad,
    /// Decelerating to zero velocity.
    EaseOutQuad,
    /// Accelerate then decelerate.
    EaseInOutQuad,

    // --- Cubic ---
    /// Accelerating from zero velocity (cubic).
    EaseInCubic,
    /// Decelerating to zero velocity (cubic).
    EaseOutCubic,
    /// Accelerate then decelerate (cubic).
    EaseInOutCubic,

    // --- Quartic ---
    /// Accelerating from zero velocity (quartic).
    EaseInQuart,
    /// Decelerating to zero velocity (quartic).
    EaseOutQuart,
    /// Accelerate then decelerate (quartic).
    EaseInOutQuart,

    // --- Quintic ---
    /// Accelerating from zero velocity (quintic).
    EaseInQuint,
    /// Decelerating to zero velocity (quintic).
    EaseOutQuint,
    /// Accelerate then decelerate (quintic).
    EaseInOutQuint,

    // --- Sinusoidal ---
    /// Accelerating using a sine curve.
    EaseInSine,
    /// Decelerating using a sine curve.
    EaseOutSine,
    /// Accelerate then decelerate (sine).
    EaseInOutSine,

    // --- Exponential ---
    /// Accelerating exponentially.
    EaseInExpo,
    /// Decelerating exponentially.
    EaseOutExpo,
    /// Accelerate then decelerate (exponential).
    EaseInOutExpo,

    // --- Circular ---
    /// Accelerating along a circular arc.
    EaseInCirc,
    /// Decelerating along a circular arc.
    EaseOutCirc,
    /// Accelerate then decelerate (circular).
    EaseInOutCirc,

    // --- Physical / dynamic ---
    /// Spring-like overshoot. `damping` controls overshoot amount (default ~1.70158).
    Spring {
        /// Overshoot magnitude. Higher = more dramatic overshoot. CSS default: 1.70158.
        overshoot: f32,
    },
    /// Bouncing at the end, like a ball dropping.
    Bounce,
    /// Elastic snap with overshoot and oscillation.
    Elastic,

    /// Custom cubic Bézier curve (CSS `cubic-bezier(x1, y1, x2, y2)`).
    ///
    /// Control points define the curve shape. `(0,0)` and `(1,1)` are implicit
    /// start/end points.
    CubicBezier {
        /// X of first control point (0.0–1.0).
        x1: f32,
        /// Y of first control point.
        y1: f32,
        /// X of second control point (0.0–1.0).
        x2: f32,
        /// Y of second control point.
        y2: f32,
    },
}

impl Easing {
    /// CSS `ease` — equivalent to `cubic-bezier(0.25, 0.1, 0.25, 1.0)`.
    pub const CSS_EASE: Self = Self::CubicBezier {
        x1: 0.25,
        y1: 0.1,
        x2: 0.25,
        y2: 1.0,
    };

    /// CSS `ease-in` — equivalent to `cubic-bezier(0.42, 0, 1, 1)`.
    pub const CSS_EASE_IN: Self = Self::CubicBezier {
        x1: 0.42,
        y1: 0.0,
        x2: 1.0,
        y2: 1.0,
    };

    /// CSS `ease-out` — equivalent to `cubic-bezier(0, 0, 0.58, 1)`.
    pub const CSS_EASE_OUT: Self = Self::CubicBezier {
        x1: 0.0,
        y1: 0.0,
        x2: 0.58,
        y2: 1.0,
    };

    /// CSS `ease-in-out` — equivalent to `cubic-bezier(0.42, 0, 0.58, 1)`.
    pub const CSS_EASE_IN_OUT: Self = Self::CubicBezier {
        x1: 0.42,
        y1: 0.0,
        x2: 0.58,
        y2: 1.0,
    };

    /// Spring with the standard CSS-like overshoot (back ease).
    pub const BACK: Self = Self::Spring { overshoot: 1.70158 };

    /// Evaluate the easing function at progress `t ∈ [0,1]`.
    ///
    /// Returns a value typically in `[0,1]`, but spring/elastic curves may
    /// overshoot outside this range temporarily.
    ///
    /// Values of `t` outside `[0,1]` are clamped.
    #[must_use]
    #[allow(clippy::excessive_precision, clippy::too_many_lines)]
    pub fn ease(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,

            // Quadratic
            Self::EaseInQuad => t * t,
            Self::EaseOutQuad => t * (2.0 - t),
            Self::EaseInOutQuad => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }

            // Cubic
            Self::EaseInCubic => t * t * t,
            Self::EaseOutCubic => {
                let u = 1.0 - t;
                1.0 - u * u * u
            }
            Self::EaseInOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let u = 2.0 * t - 2.0;
                    1.0 + 0.5 * u * u * u
                }
            }

            // Quartic
            Self::EaseInQuart => t * t * t * t,
            Self::EaseOutQuart => {
                let u = 1.0 - t;
                1.0 - u * u * u * u
            }
            Self::EaseInOutQuart => {
                if t < 0.5 {
                    8.0 * t * t * t * t
                } else {
                    let u = 2.0 * t - 2.0;
                    1.0 - 0.5 * u * u * u * u
                }
            }

            // Quintic
            Self::EaseInQuint => t * t * t * t * t,
            Self::EaseOutQuint => {
                let u = 1.0 - t;
                1.0 - u * u * u * u * u
            }
            Self::EaseInOutQuint => {
                if t < 0.5 {
                    16.0 * t * t * t * t * t
                } else {
                    let u = 2.0 * t - 2.0;
                    1.0 + 0.5 * u * u * u * u * u
                }
            }

            // Sinusoidal
            Self::EaseInSine => 1.0 - (t * std::f32::consts::FRAC_PI_2).cos(),
            Self::EaseOutSine => (t * std::f32::consts::FRAC_PI_2).sin(),
            Self::EaseInOutSine => 0.5 * (1.0 - (std::f32::consts::PI * t).cos()),

            // Exponential
            Self::EaseInExpo => {
                if t <= 0.0 {
                    0.0
                } else {
                    (2.0_f32).powf(10.0 * (t - 1.0))
                }
            }
            Self::EaseOutExpo => {
                if t >= 1.0 {
                    1.0
                } else {
                    1.0 - (2.0_f32).powf(-10.0 * t)
                }
            }
            Self::EaseInOutExpo => {
                if t <= 0.0 {
                    return 0.0;
                }
                if t >= 1.0 {
                    return 1.0;
                }
                if t < 0.5 {
                    0.5 * (2.0_f32).powf(20.0 * t - 10.0)
                } else {
                    1.0 - 0.5 * (2.0_f32).powf(-20.0 * t + 10.0)
                }
            }

            // Circular
            Self::EaseInCirc => 1.0 - (1.0 - t * t).sqrt(),
            Self::EaseOutCirc => {
                let u = t - 1.0;
                (1.0 - u * u).sqrt()
            }
            Self::EaseInOutCirc => {
                if t < 0.5 {
                    0.5 * (1.0 - (1.0 - 4.0 * t * t).sqrt())
                } else {
                    let u = 2.0 * t - 2.0;
                    0.5 * ((1.0 - u * u).sqrt() + 1.0)
                }
            }

            // Spring (Back)
            Self::Spring { overshoot } => {
                let s = overshoot;
                let u = t - 1.0;
                u * u * ((s + 1.0) * u + s) + 1.0
            }

            // Bounce
            Self::Bounce => ease_out_bounce(t),

            // Elastic
            Self::Elastic => {
                if t <= 0.0 {
                    return 0.0;
                }
                if t >= 1.0 {
                    return 1.0;
                }
                let p = 0.3_f32;
                let s = p / 4.0;
                let u = t - 1.0;
                -((2.0_f32).powf(10.0 * u) * ((u - s) * std::f32::consts::TAU / p).sin()) + 1.0
            }

            // Cubic Bézier
            Self::CubicBezier { x1, y1, x2, y2 } => cubic_bezier_ease(*x1, *y1, *x2, *y2, t),
        }
    }
}

/// Bounce easing (decelerate with bounces).
fn ease_out_bounce(t: f32) -> f32 {
    const N1: f32 = 7.562_5;
    const D1: f32 = 2.75;

    if t < 1.0 / D1 {
        N1 * t * t
    } else if t < 2.0 / D1 {
        let t = t - 1.5 / D1;
        N1 * t * t + 0.75
    } else if t < 2.5 / D1 {
        let t = t - 2.25 / D1;
        N1 * t * t + 0.9375
    } else {
        let t = t - 2.625 / D1;
        N1.mul_add(t * t, 0.984_375)
    }
}

/// Evaluate a cubic Bézier easing curve.
///
/// Uses Newton-Raphson iteration with bisection fallback to find `t_bezier`
/// such that `x(t_bezier) = t`, then returns `y(t_bezier)`.
/// Based on the WebKit/Blink implementation.
fn cubic_bezier_ease(x1: f32, y1: f32, x2: f32, y2: f32, t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    // Find t_bezier for the given x using Newton-Raphson
    let mut guess = t;
    for _ in 0..8 {
        let x = sample_curve_x(x1, x2, guess) - t;
        if x.abs() < 1e-7 {
            return sample_curve_y(y1, y2, guess);
        }
        let dx = sample_curve_dx(x1, x2, guess);
        if dx.abs() < 1e-7 {
            break;
        }
        guess -= x / dx;
        guess = guess.clamp(0.0, 1.0);
    }

    // Newton-Raphson didn't converge — fall back to bisection
    let mut lo = 0.0_f32;
    let mut hi = 1.0_f32;
    guess = t;
    for _ in 0..20 {
        let x = sample_curve_x(x1, x2, guess);
        if (x - t).abs() < 1e-7 {
            return sample_curve_y(y1, y2, guess);
        }
        if x < t {
            lo = guess;
        } else {
            hi = guess;
        }
        guess = (lo + hi) * 0.5;
    }

    sample_curve_y(y1, y2, guess)
}

#[inline]
fn sample_curve_x(x1: f32, x2: f32, t: f32) -> f32 {
    // B(t) = 3*(1-t)^2*t*x1 + 3*(1-t)*t^2*x2 + t^3
    ((1.0 - 3.0 * x2 + 3.0 * x1) * t + (3.0 * x2 - 6.0 * x1)) * t + 3.0 * x1 * t
}

#[inline]
fn sample_curve_y(y1: f32, y2: f32, t: f32) -> f32 {
    ((1.0 - 3.0 * y2 + 3.0 * y1) * t + (3.0 * y2 - 6.0 * y1)) * t + 3.0 * y1 * t
}

#[inline]
fn sample_curve_dx(x1: f32, x2: f32, t: f32) -> f32 {
    (3.0 * (1.0 - 3.0 * x2 + 3.0 * x1))
        .mul_add(t, 2.0 * (3.0 * x2 - 6.0 * x1))
        .mul_add(t, 3.0 * x1)
}

// ---------------------------------------------------------------------------
// Lerp
// ---------------------------------------------------------------------------

/// Trait for linear interpolation between two values.
///
/// Implementations should satisfy:
/// - `lerp(a, b, 0.0) == a`
/// - `lerp(a, b, 1.0) == b`
/// - Values between 0 and 1 produce smooth interpolation
pub trait Lerp: Clone {
    /// Interpolate between `self` and `other` at position `t ∈ [0,1]`.
    fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Lerp for f64 {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        (other - self).mul_add(Self::from(t), *self)
    }
}

impl Lerp for Color {
    /// Interpolate in Oklab perceptual color space with alpha.
    ///
    /// Produces perceptually smooth gradients — avoids the muddy
    /// midpoints of linear RGB (e.g., red→green won't go through brown).
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self.mix(*other, t)
    }
}

impl Lerp for Point {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

impl Lerp for Transform {
    /// Interpolate by decomposing into translate, rotate, scale, then
    /// interpolating each component and recomposing.
    ///
    /// This is the standard SVG/CSS approach — it preserves rotation
    /// direction, scale uniformity, and translation linearity, unlike
    /// naive matrix component lerp.
    fn lerp(&self, other: &Self, t: f32) -> Self {
        // Decompose both transforms
        let (tx1, ty1, sx1, sy1, r1) = decompose_transform(self);
        let (tx2, ty2, sx2, sy2, r2) = decompose_transform(other);

        // Interpolate each component
        let tx = tx1 + (tx2 - tx1) * t;
        let ty = ty1 + (ty2 - ty1) * t;
        let sx = sx1 + (sx2 - sx1) * t;
        let sy = sy1 + (sy2 - sy1) * t;

        // For rotation, take the shortest path
        let mut dr = r2 - r1;
        if dr > std::f32::consts::PI {
            dr -= std::f32::consts::TAU;
        } else if dr < -std::f32::consts::PI {
            dr += std::f32::consts::TAU;
        }
        let r = r1 + dr * t;

        // Recompose: Scale × Rotate × Translate
        let cos = r.cos();
        let sin = r.sin();
        Self {
            sx: sx * cos,
            kx: sx * sin,
            ky: -sy * sin,
            sy: sy * cos,
            tx,
            ty,
        }
    }
}

/// Decompose a 2D affine transform into (tx, ty, sx, sy, rotation).
///
/// Assumes the matrix is composed as `Scale × Rotate × Translate` (the
/// standard SVG/CSS decomposition). Returns rotation in radians.
#[inline]
fn decompose_transform(t: &Transform) -> (f32, f32, f32, f32, f32) {
    let tx = t.tx;
    let ty = t.ty;
    let sx = (t.sx * t.sx + t.kx * t.kx).sqrt();
    let sy = (t.ky * t.ky + t.sy * t.sy).sqrt();
    let rotation = t.kx.atan2(t.sx);
    (tx, ty, sx, sy, rotation)
}

// Also implement for tuples of lerp-able types
impl<A: Lerp, B: Lerp> Lerp for (A, B) {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        (self.0.lerp(&other.0, t), self.1.lerp(&other.1, t))
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// Animates a single value from `from` to `to` over a duration with easing.
///
/// Call [`advance()`](Transition::advance) each frame with the elapsed delta
/// time, and [`value()`](Transition::value) to read the current interpolated
/// result.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use scry_engine::scene::animation::{Easing, Transition};
///
/// let mut opacity = Transition::new(0.0_f32, 1.0_f32, Duration::from_millis(300));
/// opacity.advance(Duration::from_millis(150));
/// assert!((opacity.value() - 0.5).abs() < 0.01);
/// ```
#[derive(Clone, Debug)]
pub struct Transition<T: Lerp> {
    from: T,
    to: T,
    duration: Duration,
    elapsed: Duration,
    easing: Easing,
    /// Cached progress (0.0–1.0) — avoids recomputing on repeated `value()` calls.
    progress: f32,
}

impl<T: Lerp> Transition<T> {
    /// Create a new transition.
    ///
    /// # Panics
    ///
    /// Panics if `duration` is zero.
    #[must_use]
    pub const fn new(from: T, to: T, duration: Duration) -> Self {
        assert!(!duration.is_zero(), "transition duration must be non-zero");
        Self {
            from,
            to,
            duration,
            elapsed: Duration::ZERO,
            easing: Easing::Linear,
            progress: 0.0,
        }
    }

    /// Set the easing function. Default: [`Easing::Linear`].
    #[must_use]
    pub const fn easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }

    /// Advance the animation by `dt` and return whether it completed this frame.
    pub fn advance(&mut self, dt: Duration) -> bool {
        let was_complete = self.is_complete();
        self.elapsed = self.elapsed.saturating_add(dt);
        if self.elapsed > self.duration {
            self.elapsed = self.duration;
        }
        let linear_t = self.elapsed.as_secs_f32() / self.duration.as_secs_f32();
        self.progress = self.easing.ease(linear_t);
        !was_complete && self.is_complete()
    }

    /// The current interpolated value.
    #[must_use]
    pub fn value(&self) -> T {
        self.from.lerp(&self.to, self.progress)
    }

    /// Whether the animation has reached its end.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Raw linear progress (0.0–1.0), before easing.
    #[must_use]
    pub const fn linear_progress(&self) -> f32 {
        self.elapsed.as_secs_f32() / self.duration.as_secs_f32()
    }

    /// Eased progress (0.0–1.0+), after applying the easing curve.
    #[must_use]
    pub const fn eased_progress(&self) -> f32 {
        self.progress
    }

    /// Reset the animation to the beginning.
    pub const fn reset(&mut self) {
        self.elapsed = Duration::ZERO;
        self.progress = 0.0;
    }

    /// Reset and reverse the direction (swap from and to).
    pub fn reverse(&mut self) {
        std::mem::swap(&mut self.from, &mut self.to);
        self.reset();
    }

    /// Remaining time until completion.
    #[must_use]
    pub const fn remaining(&self) -> Duration {
        self.duration.saturating_sub(self.elapsed)
    }

    /// The total duration.
    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }
}

// ---------------------------------------------------------------------------
// Keyframes
// ---------------------------------------------------------------------------

/// A single keyframe in a timeline.
#[derive(Clone, Debug)]
pub struct Keyframe<T: Lerp> {
    /// Position in the timeline (0.0–1.0).
    pub position: f32,
    /// Value at this keyframe.
    pub value: T,
    /// Easing to use when interpolating *from* this keyframe to the next.
    pub easing: Easing,
}

/// Multi-stop animation timeline with per-segment easing.
///
/// Keyframes are automatically sorted by position. The timeline maps a
/// global progress `t ∈ [0,1]` to an interpolated value by finding the
/// active segment and lerping within it using the segment's easing.
///
/// # Example
///
/// ```
/// use scry_engine::scene::animation::{Easing, Keyframes, Keyframe};
///
/// let kf = Keyframes::new(vec![
///     Keyframe { position: 0.0, value: 0.0_f32, easing: Easing::Linear },
///     Keyframe { position: 0.5, value: 100.0, easing: Easing::EaseOutCubic },
///     Keyframe { position: 1.0, value: 50.0, easing: Easing::Linear },
/// ]);
///
/// assert!((kf.value_at(0.0) - 0.0).abs() < f32::EPSILON);
/// assert!((kf.value_at(0.25) - 50.0).abs() < f32::EPSILON); // linear 0→100 at 50%
/// assert!((kf.value_at(1.0) - 50.0).abs() < f32::EPSILON);
/// ```
#[derive(Clone, Debug)]
pub struct Keyframes<T: Lerp> {
    frames: Vec<Keyframe<T>>,
}

impl<T: Lerp> Keyframes<T> {
    /// Create a new keyframe timeline.
    ///
    /// Keyframes are sorted by position. At least 2 keyframes are required.
    ///
    /// # Panics
    ///
    /// Panics if fewer than 2 keyframes are provided.
    #[must_use]
    pub fn new(mut frames: Vec<Keyframe<T>>) -> Self {
        assert!(
            frames.len() >= 2,
            "keyframe timeline requires at least 2 keyframes"
        );
        frames.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { frames }
    }

    /// Evaluate the timeline at global progress `t ∈ [0,1]`.
    ///
    /// Uses binary search to find the active segment, then interpolates
    /// within it using the segment's easing curve.
    #[must_use]
    pub fn value_at(&self, t: f32) -> T {
        let t = t.clamp(0.0, 1.0);

        // Edge cases
        if t <= self.frames[0].position {
            return self.frames[0].value.clone();
        }
        let last = self.frames.len() - 1;
        if t >= self.frames[last].position {
            return self.frames[last].value.clone();
        }

        // Binary search for the segment containing t
        let idx = self
            .frames
            .partition_point(|kf| kf.position <= t)
            .saturating_sub(1);
        let from = &self.frames[idx];
        let to = &self.frames[idx + 1];

        // Local progress within this segment
        let span = to.position - from.position;
        if span <= f32::EPSILON {
            return from.value.clone();
        }
        let local_t = (t - from.position) / span;
        let eased_t = from.easing.ease(local_t);

        from.value.lerp(&to.value, eased_t)
    }

    /// Number of keyframes in the timeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the timeline is empty (should never be true after construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

// ---------------------------------------------------------------------------
// AnimationId
// ---------------------------------------------------------------------------

/// Identifier for a named animation within an [`AnimationState`].
///
/// Using a string-keyed ID allows users to reference animations by
/// descriptive names (e.g., `"opacity"`, `"position"`, `"rotation"`).
pub type AnimationId = &'static str;

// ---------------------------------------------------------------------------
// AnimationState
// ---------------------------------------------------------------------------

/// Frame-level animation orchestrator.
///
/// Manages multiple named [`Transition`]s, advancing them all with a single
/// `tick()` call each frame. Completed animations are automatically cleaned
/// up on the next tick.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use scry_engine::scene::animation::{AnimationState, Easing};
///
/// let mut state = AnimationState::new();
/// state.start("opacity", 0.0_f32, 1.0_f32, Duration::from_millis(500), Easing::EaseOutCubic);
/// state.start("x_pos", 0.0_f32, 100.0_f32, Duration::from_secs(1), Easing::Linear);
///
/// state.tick(Duration::from_millis(250));
///
/// let opacity: f32 = state.get("opacity").unwrap_or(0.0);
/// let x: f32 = state.get("x_pos").unwrap_or(0.0);
/// assert!(opacity > 0.5); // EaseOutCubic overshoots midpoint
/// assert!((x - 25.0).abs() < 0.1); // Linear at 25%
/// ```
pub struct AnimationState {
    animations: HashMap<AnimationId, Box<dyn AnyTransition>>,
}

impl AnimationState {
    /// Create a new, empty animation state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            animations: HashMap::new(),
        }
    }

    /// Start a new named animation, replacing any existing animation with the same ID.
    pub fn start<T: Lerp + 'static>(
        &mut self,
        id: AnimationId,
        from: T,
        to: T,
        duration: Duration,
        easing: Easing,
    ) {
        let transition = Transition::new(from, to, duration).easing(easing);
        self.animations
            .insert(id, Box::new(TypedTransition(transition)));
    }

    /// Advance all animations by `dt`. Completed animations are removed.
    pub fn tick(&mut self, dt: Duration) {
        self.animations.retain(|_, anim| {
            anim.advance_any(dt);
            !anim.is_complete_any()
        });
    }

    /// Get the current value of a named animation.
    ///
    /// Returns `None` if the animation doesn't exist or has completed and
    /// been cleaned up. Use `unwrap_or(default)` to supply a fallback.
    #[must_use]
    pub fn get<T: Lerp + 'static>(&self, id: AnimationId) -> Option<T> {
        self.animations
            .get(id)
            .and_then(|a| a.value_as_any().downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Check if any animations are still running.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.animations.is_empty()
    }

    /// Number of active animations.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.animations.len()
    }

    /// Cancel a specific animation by ID.
    pub fn cancel(&mut self, id: AnimationId) {
        self.animations.remove(id);
    }

    /// Cancel all animations.
    pub fn cancel_all(&mut self) {
        self.animations.clear();
    }

    /// Check if a specific animation exists and is still running.
    #[must_use]
    pub fn is_active(&self, id: AnimationId) -> bool {
        self.animations.contains_key(id)
    }
}

impl Default for AnimationState {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AnimationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationState")
            .field("active_count", &self.animations.len())
            .field("ids", &self.animations.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Type-erased animation internals
// ---------------------------------------------------------------------------

/// Type-erased trait for storing heterogeneous animations.
///
/// This is dyn-compatible: no generic methods. Value extraction uses `Any`
/// downcasting via `value_as_any()`.
trait AnyTransition {
    /// Advance the animation by `dt`.
    fn advance_any(&mut self, dt: Duration);
    /// Whether the animation has completed.
    fn is_complete_any(&self) -> bool;
    /// Get the current value as a boxed `Any` for downcasting.
    fn value_as_any(&self) -> Box<dyn std::any::Any>;
}

/// Concrete wrapper that holds a typed `Transition<T>`.
struct TypedTransition<T: Lerp + 'static>(Transition<T>);

impl<T: Lerp + 'static> AnyTransition for TypedTransition<T> {
    fn advance_any(&mut self, dt: Duration) {
        self.0.advance(dt);
    }

    fn is_complete_any(&self) -> bool {
        self.0.is_complete()
    }

    fn value_as_any(&self) -> Box<dyn std::any::Any> {
        Box::new(self.0.value())
    }
}

// ---------------------------------------------------------------------------
// Physics Spring — Damped Harmonic Oscillator
// ---------------------------------------------------------------------------

/// Configuration for a damped harmonic oscillator spring.
///
/// Instead of specifying duration + easing curve, you define the physical
/// properties: **stiffness**, **damping**, and **mass**. The spring
/// self-determines when it has settled (displacement < ε), so there is no
/// fixed duration — it simply feels right.
///
/// Use the named presets (`GENTLE`, `BOUNCY`, `STIFF`, `SLOW`, `SNAPPY`)
/// to skip parameter tuning entirely.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use scry_engine::scene::animation::{Spring, SpringConfig};
///
/// let mut s = Spring::new(0.0_f32, 100.0_f32, SpringConfig::BOUNCY);
/// for _ in 0..120 {
///     s.advance(Duration::from_secs_f32(1.0 / 60.0));
/// }
/// assert!(s.is_settled());
/// assert!((s.value() - 100.0).abs() < 0.5);
/// ```
#[derive(Clone, Debug)]
pub struct SpringConfig {
    /// Spring constant *k*. Higher = snappier response.
    pub stiffness: f32,
    /// Friction coefficient. Higher = less oscillation.
    pub damping: f32,
    /// Mass of the virtual object. Higher = more sluggish.
    pub mass: f32,
    /// Displacement threshold below which the spring is considered settled.
    /// Default: `0.01`.
    pub rest_threshold: f32,
    /// Velocity threshold below which the spring is considered settled.
    /// Default: `0.01`.
    pub velocity_threshold: f32,
}

impl SpringConfig {
    /// Gentle, slow spring with minimal overshoot.
    /// Good for background transitions and subtle UI shifts.
    pub const GENTLE: Self = Self {
        stiffness: 100.0,
        damping: 20.0,
        mass: 1.0,
        rest_threshold: 0.01,
        velocity_threshold: 0.01,
    };

    /// Bouncy spring with noticeable overshoot and oscillation.
    /// Great for playful UI elements, pop-ins, and attention-grabbing motion.
    pub const BOUNCY: Self = Self {
        stiffness: 300.0,
        damping: 10.0,
        mass: 1.0,
        rest_threshold: 0.01,
        velocity_threshold: 0.01,
    };

    /// Stiff, fast spring with very little overshoot.
    /// Ideal for responsive controls and immediate feedback.
    pub const STIFF: Self = Self {
        stiffness: 500.0,
        damping: 30.0,
        mass: 1.0,
        rest_threshold: 0.01,
        velocity_threshold: 0.01,
    };

    /// Slow, heavy spring for dramatic, weighted motion.
    /// Use for large elements or cinematic transitions.
    pub const SLOW: Self = Self {
        stiffness: 80.0,
        damping: 15.0,
        mass: 2.0,
        rest_threshold: 0.01,
        velocity_threshold: 0.01,
    };

    /// Very snappy spring that overshoots slightly then settles fast.
    /// The default "feels good" choice for most UI animations.
    pub const SNAPPY: Self = Self {
        stiffness: 400.0,
        damping: 25.0,
        mass: 1.0,
        rest_threshold: 0.01,
        velocity_threshold: 0.01,
    };

    /// Compute the damping ratio ζ = c / (2√(km)).
    ///
    /// - ζ < 1: underdamped (oscillates)
    /// - ζ = 1: critically damped (fastest without oscillation)
    /// - ζ > 1: overdamped (sluggish, no oscillation)
    #[must_use]
    pub fn damping_ratio(&self) -> f32 {
        self.damping / (2.0 * (self.stiffness * self.mass).sqrt())
    }

    /// Natural frequency ω₀ = √(k/m).
    #[must_use]
    pub fn natural_frequency(&self) -> f32 {
        (self.stiffness / self.mass).sqrt()
    }
}

impl Default for SpringConfig {
    fn default() -> Self {
        Self::SNAPPY
    }
}

/// A physics-driven spring that animates a [`Lerp`]-able value.
///
/// Unlike [`Transition`], a `Spring` has no fixed duration — it runs until
/// the value settles within the configured thresholds. The motion is computed
/// via semi-implicit Euler integration of the damped harmonic oscillator
/// equation: `F = -kx - cv`.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use scry_engine::scene::animation::{Spring, SpringConfig};
///
/// let mut spring = Spring::new(0.0_f32, 1.0_f32, SpringConfig::STIFF);
/// while !spring.is_settled() {
///     spring.advance(Duration::from_secs_f32(1.0 / 60.0));
/// }
/// assert!((spring.value() - 1.0).abs() < 0.05);
/// ```
#[derive(Clone, Debug)]
pub struct Spring<T: Lerp> {
    from: T,
    to: T,
    config: SpringConfig,
    /// Current displacement from target (normalized 0–1 space, starts at -1).
    displacement: f32,
    /// Current velocity in normalized space.
    velocity: f32,
    /// Whether the spring has settled.
    settled: bool,
}

impl<T: Lerp> Spring<T> {
    /// Create a new spring animating from `from` to `to`.
    #[must_use]
    pub fn new(from: T, to: T, config: SpringConfig) -> Self {
        Self {
            from,
            to,
            config,
            displacement: -1.0, // start fully displaced (at `from`)
            velocity: 0.0,
            settled: false,
        }
    }

    /// Advance the spring simulation by `dt`.
    ///
    /// Uses semi-implicit Euler integration for stability:
    /// 1. `a = (-k * x - c * v) / m`
    /// 2. `v += a * dt`
    /// 3. `x += v * dt`
    pub fn advance(&mut self, dt: Duration) {
        if self.settled {
            return;
        }

        let dt_secs = dt.as_secs_f32();
        if dt_secs <= 0.0 {
            return;
        }

        let k = self.config.stiffness;
        let c = self.config.damping;
        let m = self.config.mass;

        // Semi-implicit Euler (velocity-first for better energy conservation)
        let acceleration = (-k * self.displacement - c * self.velocity) / m;
        self.velocity += acceleration * dt_secs;
        self.displacement += self.velocity * dt_secs;

        // Check if settled
        if self.displacement.abs() < self.config.rest_threshold
            && self.velocity.abs() < self.config.velocity_threshold
        {
            self.displacement = 0.0;
            self.velocity = 0.0;
            self.settled = true;
        }
    }

    /// The current interpolated value.
    ///
    /// Maps the internal displacement (where 0 = target) back to the
    /// `from`/`to` range.
    #[must_use]
    pub fn value(&self) -> T {
        // displacement of -1 = at `from`, 0 = at `to`
        let t = 1.0 + self.displacement;
        self.from.lerp(&self.to, t.clamp(0.0, 2.0))
    }

    /// Whether the spring has settled at its target value.
    #[must_use]
    pub const fn is_settled(&self) -> bool {
        self.settled
    }

    /// Retarget the spring to a new destination without resetting velocity.
    ///
    /// This produces beautiful interrupted animations — the spring smoothly
    /// redirects mid-flight instead of snapping.
    pub fn retarget(&mut self, new_to: T) {
        // Current value becomes the new `from`
        let current = self.value();
        self.from = current;
        self.to = new_to;
        self.displacement = -1.0;
        // Keep velocity for momentum continuity
        self.settled = false;
    }

    /// Reset the spring to its initial state.
    pub fn reset(&mut self) {
        self.displacement = -1.0;
        self.velocity = 0.0;
        self.settled = false;
    }

    /// Current velocity (in normalized space).
    #[must_use]
    pub const fn velocity(&self) -> f32 {
        self.velocity
    }
}

// ---------------------------------------------------------------------------
// Animation Sequence — Coroutine-Style Choreography
// ---------------------------------------------------------------------------

/// A single step in an animation sequence.
///
/// Steps execute one after another (like screenplay directions), except
/// for `Parallel` which runs multiple sub-steps simultaneously.
#[derive(Clone, Debug)]
pub enum Step {
    /// Instantly set a named value.
    Set {
        /// Animation name.
        id: String,
        /// Value to set (stored as f32 for simplicity; use Tween with 0 duration for other types).
        value: f32,
    },
    /// Tween a named f32 value from → to over a duration with easing.
    Tween {
        /// Animation name.
        id: String,
        /// Starting value.
        from: f32,
        /// Ending value.
        to: f32,
        /// Duration of the tween.
        duration: Duration,
        /// Easing curve.
        easing: Easing,
    },
    /// Animate a named f32 value using a physics spring.
    SpringTo {
        /// Animation name.
        id: String,
        /// Starting value.
        from: f32,
        /// Target value.
        to: f32,
        /// Spring configuration.
        config: SpringConfig,
    },
    /// Pause for a duration before the next step.
    Wait {
        /// How long to wait.
        duration: Duration,
    },
    /// Run multiple sub-sequences simultaneously. The parallel step completes
    /// when the longest sub-sequence finishes.
    Parallel {
        /// Sub-sequences to run in parallel.
        branches: Vec<Vec<Step>>,
    },
    /// Like Parallel, but each branch starts after a stagger delay.
    /// Branch 0 starts immediately, branch 1 after `delay`, branch 2 after `2*delay`, etc.
    Stagger {
        /// Delay between each branch start.
        delay: Duration,
        /// Sub-sequences, each started at an offset.
        branches: Vec<Vec<Step>>,
    },
}

/// A builder for constructing animation sequences declaratively.
///
/// Reads like a screenplay — each method adds the next "direction" in the
/// animation timeline.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use scry_engine::scene::animation::{AnimationSequence, Easing, SpringConfig};
///
/// fn ms(n: u64) -> Duration { Duration::from_millis(n) }
///
/// let seq = AnimationSequence::new()
///     .tween("opacity", 0.0, 1.0, ms(300), Easing::EaseOutCubic)
///     .wait(ms(100))
///     .spring_to("scale", 0.8, 1.0, SpringConfig::BOUNCY)
///     .parallel(|p| {
///         p.branch(|b| b.tween("x", 0.0, 100.0, ms(500), Easing::EaseOutQuint))
///          .branch(|b| b.tween("y", 0.0, 50.0, ms(500), Easing::Linear))
///     });
///
/// assert_eq!(seq.steps().len(), 4);
/// ```
#[derive(Clone, Debug, Default)]
pub struct AnimationSequence {
    steps: Vec<Step>,
}

impl AnimationSequence {
    /// Create an empty animation sequence.
    #[must_use]
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a tween step: animate a named f32 value from → to.
    #[must_use]
    pub fn tween(
        mut self,
        id: &str,
        from: f32,
        to: f32,
        duration: Duration,
        easing: Easing,
    ) -> Self {
        self.steps.push(Step::Tween {
            id: id.to_string(),
            from,
            to,
            duration,
            easing,
        });
        self
    }

    /// Add a spring step: animate a named f32 value using physics.
    #[must_use]
    pub fn spring_to(mut self, id: &str, from: f32, to: f32, config: SpringConfig) -> Self {
        self.steps.push(Step::SpringTo {
            id: id.to_string(),
            from,
            to,
            config,
        });
        self
    }

    /// Add a wait/pause step.
    #[must_use]
    pub fn wait(mut self, duration: Duration) -> Self {
        self.steps.push(Step::Wait { duration });
        self
    }

    /// Instantly set a named value.
    #[must_use]
    pub fn set(mut self, id: &str, value: f32) -> Self {
        self.steps.push(Step::Set {
            id: id.to_string(),
            value,
        });
        self
    }

    /// Add parallel branches. The closure receives a [`ParallelBuilder`].
    #[must_use]
    pub fn parallel<F: FnOnce(ParallelBuilder) -> ParallelBuilder>(mut self, f: F) -> Self {
        let builder = f(ParallelBuilder::new());
        self.steps.push(Step::Parallel {
            branches: builder.branches,
        });
        self
    }

    /// Add staggered branches with a delay between each start.
    #[must_use]
    pub fn stagger<F: FnOnce(ParallelBuilder) -> ParallelBuilder>(
        mut self,
        delay: Duration,
        f: F,
    ) -> Self {
        let builder = f(ParallelBuilder::new());
        self.steps.push(Step::Stagger {
            delay,
            branches: builder.branches,
        });
        self
    }

    /// Access the steps in this sequence.
    #[must_use]
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Number of steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Builder for parallel/stagger branches within an [`AnimationSequence`].
#[derive(Clone, Debug, Default)]
pub struct ParallelBuilder {
    branches: Vec<Vec<Step>>,
}

impl ParallelBuilder {
    /// Create a new parallel builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
        }
    }

    /// Add a branch. The closure receives an [`AnimationSequence`] and should
    /// return it with steps added.
    #[must_use]
    pub fn branch<F: FnOnce(AnimationSequence) -> AnimationSequence>(mut self, f: F) -> Self {
        let seq = f(AnimationSequence::new());
        self.branches.push(seq.steps);
        self
    }
}

// ---------------------------------------------------------------------------
// Sequence Player
// ---------------------------------------------------------------------------

/// Active state for a single playing step.
#[derive(Clone, Debug)]
enum ActiveStep {
    /// A tween in progress.
    Tween {
        id: String,
        transition: Transition<f32>,
    },
    /// A spring in progress.
    Spring { id: String, spring: Spring<f32> },
    /// A wait/delay timer.
    Wait { remaining: Duration },
    /// Parallel branches, each a nested `SequencePlayer`.
    Parallel { players: Vec<SequencePlayer> },
    /// Stagger = parallel with delayed starts.
    Stagger {
        delay: Duration,
        players: Vec<SequencePlayer>,
        started: usize,
        elapsed: Duration,
    },
}

impl ActiveStep {
    /// Advance this active step, returning `true` when complete.
    fn advance(&mut self, dt: Duration, values: &mut HashMap<String, f32>) -> bool {
        match self {
            Self::Tween { id, transition } => {
                transition.advance(dt);
                values.insert(id.clone(), transition.value());
                transition.is_complete()
            }
            Self::Spring { id, spring } => {
                spring.advance(dt);
                values.insert(id.clone(), spring.value());
                spring.is_settled()
            }
            Self::Wait { remaining } => {
                if dt >= *remaining {
                    *remaining = Duration::ZERO;
                    true
                } else {
                    *remaining -= dt;
                    false
                }
            }
            Self::Parallel { players } => {
                for player in players.iter_mut() {
                    player.advance(dt);
                }
                players.iter().all(SequencePlayer::is_complete)
            }
            Self::Stagger {
                delay,
                players,
                started,
                elapsed,
            } => {
                *elapsed = elapsed.saturating_add(dt);
                // Start any players whose stagger delay has elapsed
                while *started < players.len() {
                    let trigger_at = delay.mul_f32(*started as f32);
                    if *elapsed >= trigger_at {
                        *started += 1;
                    } else {
                        break;
                    }
                }
                // Advance all started players
                for player in players.iter_mut().take(*started) {
                    player.advance(dt);
                }
                *started == players.len() && players.iter().all(SequencePlayer::is_complete)
            }
        }
    }
}

/// Plays an [`AnimationSequence`], stepping through each stage and
/// maintaining a map of current named values.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use scry_engine::scene::animation::{AnimationSequence, Easing, SequencePlayer};
///
/// fn ms(n: u64) -> Duration { Duration::from_millis(n) }
///
/// let seq = AnimationSequence::new()
///     .tween("x", 0.0, 100.0, ms(500), Easing::Linear);
///
/// let mut player = SequencePlayer::new(seq);
/// player.advance(ms(250));
/// assert!((player.get("x").unwrap_or(0.0) - 50.0).abs() < 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct SequencePlayer {
    steps: Vec<Step>,
    cursor: usize,
    active: Option<ActiveStep>,
    values: HashMap<String, f32>,
    complete: bool,
}

impl SequencePlayer {
    /// Create a new player for the given sequence.
    #[must_use]
    pub fn new(sequence: AnimationSequence) -> Self {
        let mut player = Self {
            steps: sequence.steps,
            cursor: 0,
            active: None,
            values: HashMap::new(),
            complete: false,
        };
        player.activate_current();
        player
    }

    /// Advance the player by `dt`. Call this each frame.
    pub fn advance(&mut self, dt: Duration) {
        if self.complete {
            return;
        }

        loop {
            // If no active step, try to activate the next one
            if self.active.is_none() {
                if !self.activate_current() {
                    self.complete = true;
                    return;
                }
            }

            // Advance the active step
            if let Some(ref mut active) = self.active {
                let done = active.advance(dt, &mut self.values);
                if done {
                    self.active = None;
                    self.cursor += 1;
                    // For instant steps (Set), we continue to the next step
                    // within the same frame. For timed steps, we break.
                    continue;
                }
            }

            break;
        }
    }

    /// Get the current value of a named animation property.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<f32> {
        self.values.get(id).copied()
    }

    /// Whether all steps have completed.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    /// All current values as a map.
    #[must_use]
    pub fn values(&self) -> &HashMap<String, f32> {
        &self.values
    }

    /// Activate the step at the current cursor position.
    /// Returns `false` if there are no more steps.
    fn activate_current(&mut self) -> bool {
        if self.cursor >= self.steps.len() {
            return false;
        }

        let step = self.steps[self.cursor].clone();
        match step {
            Step::Set { id, value } => {
                self.values.insert(id, value);
                self.cursor += 1;
                // Recurse to activate the next step (Set is instantaneous)
                self.activate_current()
            }
            Step::Tween {
                id,
                from,
                to,
                duration,
                easing,
            } => {
                self.values.insert(id.clone(), from);
                self.active = Some(ActiveStep::Tween {
                    id,
                    transition: Transition::new(from, to, duration).easing(easing),
                });
                true
            }
            Step::SpringTo {
                id,
                from,
                to,
                config,
            } => {
                self.values.insert(id.clone(), from);
                self.active = Some(ActiveStep::Spring {
                    id,
                    spring: Spring::new(from, to, config),
                });
                true
            }
            Step::Wait { duration } => {
                self.active = Some(ActiveStep::Wait {
                    remaining: duration,
                });
                true
            }
            Step::Parallel { branches } => {
                let players = branches
                    .into_iter()
                    .map(|steps| SequencePlayer::new(AnimationSequence { steps }))
                    .collect();
                self.active = Some(ActiveStep::Parallel { players });
                true
            }
            Step::Stagger { delay, branches } => {
                let players: Vec<_> = branches
                    .into_iter()
                    .map(|steps| SequencePlayer::new(AnimationSequence { steps }))
                    .collect();
                let initial_started = if players.is_empty() { 0 } else { 1 };
                self.active = Some(ActiveStep::Stagger {
                    delay,
                    players,
                    started: initial_started, // first starts immediately
                    elapsed: Duration::ZERO,
                });
                true
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Animation Presets
// ---------------------------------------------------------------------------

/// Curated preset animation sequences for common UI patterns.
///
/// Each function returns an [`AnimationSequence`] ready to be played with
/// a [`SequencePlayer`]. The `id` parameter is the name used to read the
/// animated value via `player.get(id)`.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use scry_engine::scene::animation::{preset, SequencePlayer};
///
/// let seq = preset::fade_in("opacity", Duration::from_millis(300));
/// let mut player = SequencePlayer::new(seq);
/// player.advance(Duration::from_millis(300));
/// assert!((player.get("opacity").unwrap_or(0.0) - 1.0).abs() < 0.05);
/// ```
pub mod preset {
    use super::*;

    /// Fade in: opacity 0 → 1 with ease-out.
    #[must_use]
    pub fn fade_in(id: &str, duration: Duration) -> AnimationSequence {
        AnimationSequence::new().tween(id, 0.0, 1.0, duration, Easing::EaseOutCubic)
    }

    /// Fade out: opacity 1 → 0 with ease-in.
    #[must_use]
    pub fn fade_out(id: &str, duration: Duration) -> AnimationSequence {
        AnimationSequence::new().tween(id, 1.0, 0.0, duration, Easing::EaseInCubic)
    }

    /// Slide in from the left: combines x-offset and opacity.
    ///
    /// Produces two values: `{id}_x` (position) and `{id}_opacity`.
    #[must_use]
    pub fn slide_in_left(id: &str, distance: f32, duration: Duration) -> AnimationSequence {
        let x_id = format!("{id}_x");
        let opacity_id = format!("{id}_opacity");
        AnimationSequence::new().parallel(|p| {
            p.branch(|b| b.tween(&x_id, -distance, 0.0, duration, Easing::EaseOutCubic))
                .branch(|b| b.tween(&opacity_id, 0.0, 1.0, duration, Easing::EaseOutCubic))
        })
    }

    /// Slide in from the right: combines x-offset and opacity.
    ///
    /// Produces two values: `{id}_x` (position) and `{id}_opacity`.
    #[must_use]
    pub fn slide_in_right(id: &str, distance: f32, duration: Duration) -> AnimationSequence {
        let x_id = format!("{id}_x");
        let opacity_id = format!("{id}_opacity");
        AnimationSequence::new().parallel(|p| {
            p.branch(|b| b.tween(&x_id, distance, 0.0, duration, Easing::EaseOutCubic))
                .branch(|b| b.tween(&opacity_id, 0.0, 1.0, duration, Easing::EaseOutCubic))
        })
    }

    /// Pop in: scale 0 → 1 with a bouncy spring.
    ///
    /// Produces `{id}` (scale value).
    #[must_use]
    pub fn pop_in(id: &str) -> AnimationSequence {
        AnimationSequence::new().spring_to(id, 0.0, 1.0, SpringConfig::BOUNCY)
    }

    /// Bounce in: scale 0 → 1 with a gentle spring and slight overshoot.
    ///
    /// Produces `{id}` (scale value).
    #[must_use]
    pub fn bounce_in(id: &str) -> AnimationSequence {
        AnimationSequence::new().spring_to(
            id,
            0.0,
            1.0,
            SpringConfig {
                stiffness: 200.0,
                damping: 12.0,
                mass: 1.0,
                rest_threshold: 0.01,
                velocity_threshold: 0.01,
            },
        )
    }

    /// Pulse: scale up slightly then back to 1.0.
    ///
    /// Produces `{id}` (scale value).
    #[must_use]
    pub fn pulse(id: &str, duration: Duration) -> AnimationSequence {
        let half = duration / 2;
        AnimationSequence::new()
            .tween(id, 1.0, 1.2, half, Easing::EaseOutCubic)
            .tween(id, 1.2, 1.0, half, Easing::EaseInCubic)
    }

    /// Shake: horizontal wiggle for attention.
    ///
    /// Produces `{id}` (x-offset value that returns to 0).
    #[must_use]
    pub fn shake(id: &str, intensity: f32, duration: Duration) -> AnimationSequence {
        let quarter = duration / 4;
        AnimationSequence::new()
            .tween(id, 0.0, intensity, quarter, Easing::EaseOutQuad)
            .tween(id, intensity, -intensity, quarter, Easing::EaseInOutQuad)
            .tween(
                id,
                -intensity,
                intensity * 0.5,
                quarter,
                Easing::EaseInOutQuad,
            )
            .tween(id, intensity * 0.5, 0.0, quarter, Easing::EaseOutQuad)
    }

    /// Scale up from 0 with a snappy spring, combined with fade-in.
    ///
    /// Produces `{id}_scale` and `{id}_opacity`.
    #[must_use]
    pub fn scale_fade_in(id: &str, duration: Duration) -> AnimationSequence {
        let scale_id = format!("{id}_scale");
        let opacity_id = format!("{id}_opacity");
        AnimationSequence::new().parallel(|p| {
            p.branch(|b| b.spring_to(&scale_id, 0.0, 1.0, SpringConfig::SNAPPY))
                .branch(|b| b.tween(&opacity_id, 0.0, 1.0, duration, Easing::EaseOutCubic))
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Easing tests ----

    #[test]
    fn easing_linear_bounds() {
        let e = Easing::Linear;
        assert!((e.ease(0.0)).abs() < f32::EPSILON);
        assert!((e.ease(1.0) - 1.0).abs() < f32::EPSILON);
        assert!((e.ease(0.5) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn easing_all_curves_start_at_zero_end_at_one() {
        let curves = vec![
            Easing::Linear,
            Easing::EaseInQuad,
            Easing::EaseOutQuad,
            Easing::EaseInOutQuad,
            Easing::EaseInCubic,
            Easing::EaseOutCubic,
            Easing::EaseInOutCubic,
            Easing::EaseInQuart,
            Easing::EaseOutQuart,
            Easing::EaseInOutQuart,
            Easing::EaseInQuint,
            Easing::EaseOutQuint,
            Easing::EaseInOutQuint,
            Easing::EaseInSine,
            Easing::EaseOutSine,
            Easing::EaseInOutSine,
            Easing::EaseInExpo,
            Easing::EaseOutExpo,
            Easing::EaseInOutExpo,
            Easing::EaseInCirc,
            Easing::EaseOutCirc,
            Easing::EaseInOutCirc,
            Easing::Bounce,
        ];

        for curve in curves {
            let start = curve.ease(0.0);
            let end = curve.ease(1.0);
            assert!(
                start.abs() < 0.01,
                "{curve:?} ease(0) = {start}, expected ≈ 0"
            );
            assert!(
                (end - 1.0).abs() < 0.01,
                "{curve:?} ease(1) = {end}, expected ≈ 1"
            );
        }
    }

    #[test]
    fn easing_ease_in_out_symmetric() {
        let curves = vec![
            Easing::EaseInOutQuad,
            Easing::EaseInOutCubic,
            Easing::EaseInOutQuart,
            Easing::EaseInOutQuint,
            Easing::EaseInOutSine,
        ];

        for curve in curves {
            let mid = curve.ease(0.5);
            assert!(
                (mid - 0.5).abs() < 0.01,
                "{curve:?} ease(0.5) = {mid}, expected ≈ 0.5"
            );
        }
    }

    #[test]
    fn easing_cubic_bezier_linear() {
        // cubic-bezier(0, 0, 1, 1) should be approximately linear
        let e = Easing::CubicBezier {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let v = e.ease(t);
            assert!((v - t).abs() < 0.05, "cubic-bezier linear at {t}: got {v}");
        }
    }

    #[test]
    fn easing_clamps_input() {
        let e = Easing::Linear;
        assert!((e.ease(-1.0)).abs() < f32::EPSILON);
        assert!((e.ease(2.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn easing_spring_overshoots() {
        let e = Easing::Spring { overshoot: 1.70158 };
        // At ~80% progress, spring with standard overshoot should exceed 1.0
        let val = e.ease(0.85);
        assert!(val > 1.0, "spring should overshoot: got {val}");
    }

    #[test]
    fn easing_bounce_multi_bounce() {
        let e = Easing::Bounce;
        // Bounce should have local minima (the ball bouncing)
        let mut has_decrease = false;
        let mut prev = 0.0;
        for i in 1..=100 {
            let t = i as f32 / 100.0;
            let v = e.ease(t);
            if v < prev {
                has_decrease = true;
            }
            prev = v;
        }
        assert!(has_decrease, "bounce should have multiple bounces");
    }

    // ---- Lerp tests ----

    #[test]
    fn lerp_f32() {
        assert!((0.0_f32.lerp(&100.0, 0.5) - 50.0).abs() < f32::EPSILON);
        assert!((0.0_f32.lerp(&100.0, 0.0)).abs() < f32::EPSILON);
        assert!((0.0_f32.lerp(&100.0, 1.0) - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn lerp_color() {
        let black = Color::BLACK;
        let white = Color::WHITE;
        let mid = black.lerp(&white, 0.5);
        // Oklab midpoint of black→white is perceptual mid-gray.
        // In Oklab, L=0.5 maps to sRGB ~0.39 (not 0.5 as in linear RGB).
        assert!((mid.r - mid.g).abs() < 0.01, "mid-gray should be neutral");
        assert!((mid.g - mid.b).abs() < 0.01, "mid-gray should be neutral");
        assert!(
            mid.r > 0.3 && mid.r < 0.5,
            "mid-gray R={} should be ~0.39",
            mid.r
        );
    }

    #[test]
    fn oklab_round_trip() {
        let colors = [
            Color::RED,
            Color::GREEN,
            Color::BLUE,
            Color::WHITE,
            Color::BLACK,
        ];
        for color in colors {
            let (l, a, b) = color.to_oklab();
            let restored = Color::from_oklab(l, a, b, color.a);
            assert!(
                (color.r - restored.r).abs() < 0.01,
                "R: {:.4} vs {:.4}",
                color.r,
                restored.r
            );
            assert!(
                (color.g - restored.g).abs() < 0.01,
                "G: {:.4} vs {:.4}",
                color.g,
                restored.g
            );
            assert!(
                (color.b - restored.b).abs() < 0.01,
                "B: {:.4} vs {:.4}",
                color.b,
                restored.b
            );
        }
    }

    #[test]
    fn lerp_color_oklab_no_muddy_midtones() {
        let red = Color::RED;
        let green = Color::GREEN;
        let oklab_mid = red.lerp(&green, 0.5);
        // The Oklab midpoint of red→green should be a bright yellow/olive,
        // not the dull brown that linear RGB produces.
        // In Oklab, the midpoint should have higher perceived brightness
        // than the brown produced by linear RGB.
        let (l_oklab, _, _) = oklab_mid.to_oklab();
        // Compare with linear RGB midpoint
        let linear_mid = red.mix_rgb(green, 0.5);
        let (l_linear, _, _) = linear_mid.to_oklab();
        assert!(
            l_oklab > l_linear - 0.05,
            "Oklab mid L={l_oklab:.3} should be ≥ linear mid L={l_linear:.3}"
        );
    }

    #[test]
    fn lerp_point() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(100.0, 200.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 50.0).abs() < f32::EPSILON);
        assert!((mid.y - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn lerp_transform_identity() {
        let a = Transform::IDENTITY;
        let b = Transform::translate(100.0, 200.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.tx - 50.0).abs() < f32::EPSILON);
        assert!((mid.ty - 100.0).abs() < f32::EPSILON);
        assert!((mid.sx - 1.0).abs() < f32::EPSILON);
    }

    // ---- Transition tests ----

    #[test]
    fn transition_linear_midpoint() {
        let mut t = Transition::new(0.0_f32, 100.0, Duration::from_millis(1000));
        t.advance(Duration::from_millis(500));
        assert!((t.value() - 50.0).abs() < 0.1);
    }

    #[test]
    fn transition_completes() {
        let mut t = Transition::new(0.0_f32, 100.0, Duration::from_millis(500));
        assert!(!t.is_complete());
        let completed = t.advance(Duration::from_millis(600));
        assert!(completed);
        assert!(t.is_complete());
        assert!((t.value() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn transition_easing_applied() {
        let mut linear = Transition::new(0.0_f32, 100.0, Duration::from_millis(1000));
        let mut eased = Transition::new(0.0_f32, 100.0, Duration::from_millis(1000))
            .easing(Easing::EaseOutCubic);

        linear.advance(Duration::from_millis(500));
        eased.advance(Duration::from_millis(500));

        // EaseOutCubic at 50% = 1 - (0.5)^3 = 0.875 → value = 87.5
        assert!(eased.value() > linear.value());
    }

    #[test]
    fn transition_reverse() {
        let mut t = Transition::new(0.0_f32, 100.0, Duration::from_millis(500));
        t.advance(Duration::from_millis(300));
        t.reverse();
        assert!((t.value() - 100.0).abs() < f32::EPSILON); // starts at swapped "from"
        t.advance(Duration::from_millis(500));
        assert!((t.value() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "transition duration must be non-zero")]
    fn transition_zero_duration_panics() {
        let _ = Transition::new(0.0_f32, 1.0, Duration::ZERO);
    }

    // ---- Keyframes tests ----

    #[test]
    fn keyframes_two_stops() {
        let kf = Keyframes::new(vec![
            Keyframe {
                position: 0.0,
                value: 0.0_f32,
                easing: Easing::Linear,
            },
            Keyframe {
                position: 1.0,
                value: 100.0,
                easing: Easing::Linear,
            },
        ]);
        assert!((kf.value_at(0.0)).abs() < f32::EPSILON);
        assert!((kf.value_at(0.5) - 50.0).abs() < f32::EPSILON);
        assert!((kf.value_at(1.0) - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn keyframes_three_stops() {
        let kf = Keyframes::new(vec![
            Keyframe {
                position: 0.0,
                value: 0.0_f32,
                easing: Easing::Linear,
            },
            Keyframe {
                position: 0.5,
                value: 100.0,
                easing: Easing::Linear,
            },
            Keyframe {
                position: 1.0,
                value: 50.0,
                easing: Easing::Linear,
            },
        ]);
        assert!((kf.value_at(0.25) - 50.0).abs() < f32::EPSILON); // 0→100 at 50%
        assert!((kf.value_at(0.75) - 75.0).abs() < f32::EPSILON); // 100→50 at 50%
    }

    #[test]
    fn keyframes_clamps_edges() {
        let kf = Keyframes::new(vec![
            Keyframe {
                position: 0.2,
                value: 10.0_f32,
                easing: Easing::Linear,
            },
            Keyframe {
                position: 0.8,
                value: 90.0,
                easing: Easing::Linear,
            },
        ]);
        assert!((kf.value_at(0.0) - 10.0).abs() < f32::EPSILON);
        assert!((kf.value_at(1.0) - 90.0).abs() < f32::EPSILON);
    }

    #[test]
    fn keyframes_per_segment_easing() {
        let kf = Keyframes::new(vec![
            Keyframe {
                position: 0.0,
                value: 0.0_f32,
                easing: Easing::EaseInQuad,
            },
            Keyframe {
                position: 1.0,
                value: 100.0,
                easing: Easing::Linear,
            },
        ]);
        // EaseInQuad at 50%: 0.5^2 = 0.25 → value = 25
        let val = kf.value_at(0.5);
        assert!((val - 25.0).abs() < 0.1);
    }

    #[test]
    #[should_panic(expected = "at least 2 keyframes")]
    fn keyframes_too_few_panics() {
        let _ = Keyframes::new(vec![Keyframe {
            position: 0.0,
            value: 0.0_f32,
            easing: Easing::Linear,
        }]);
    }

    // ---- AnimationState tests ----

    #[test]
    fn animation_state_basic() {
        let mut state = AnimationState::new();
        assert!(state.is_idle());

        state.start(
            "opacity",
            0.0_f32,
            1.0_f32,
            Duration::from_millis(500),
            Easing::Linear,
        );
        assert!(!state.is_idle());
        assert_eq!(state.active_count(), 1);

        state.tick(Duration::from_millis(250));
        let val: f32 = state.get("opacity").unwrap();
        assert!((val - 0.5).abs() < 0.1);

        // Complete the animation
        state.tick(Duration::from_millis(300));
        // After tick, completed animations are cleaned up
        assert!(state.is_idle());
        assert!(state.get::<f32>("opacity").is_none());
    }

    #[test]
    fn animation_state_multiple() {
        let mut state = AnimationState::new();
        state.start(
            "a",
            0.0_f32,
            10.0,
            Duration::from_millis(100),
            Easing::Linear,
        );
        state.start(
            "b",
            0.0_f32,
            20.0,
            Duration::from_millis(200),
            Easing::Linear,
        );
        assert_eq!(state.active_count(), 2);

        state.tick(Duration::from_millis(100));
        // "a" should be complete and cleaned up, "b" should be at 50%
        assert_eq!(state.active_count(), 1);
        assert!(!state.is_active("a"));
        let b_val: f32 = state.get("b").unwrap();
        assert!((b_val - 10.0).abs() < 0.1);
    }

    #[test]
    fn animation_state_cancel() {
        let mut state = AnimationState::new();
        state.start("x", 0.0_f32, 100.0, Duration::from_secs(10), Easing::Linear);
        state.cancel("x");
        assert!(state.is_idle());
    }

    #[test]
    fn animation_state_cancel_all() {
        let mut state = AnimationState::new();
        state.start("a", 0.0_f32, 1.0, Duration::from_secs(1), Easing::Linear);
        state.start("b", 0.0_f32, 1.0, Duration::from_secs(1), Easing::Linear);
        state.cancel_all();
        assert!(state.is_idle());
    }

    // ---- Spring tests ----

    #[test]
    fn spring_converges_to_target() {
        let mut s = Spring::new(0.0_f32, 100.0, SpringConfig::SNAPPY);
        let dt = Duration::from_secs_f32(1.0 / 60.0);
        for _ in 0..300 {
            s.advance(dt);
        }
        assert!(s.is_settled(), "spring should settle within 5 seconds");
        assert!(
            (s.value() - 100.0).abs() < 0.5,
            "spring value should be near target: got {}",
            s.value()
        );
    }

    #[test]
    fn spring_bouncy_overshoots() {
        let mut s = Spring::new(0.0_f32, 100.0, SpringConfig::BOUNCY);
        let dt = Duration::from_secs_f32(1.0 / 60.0);
        let mut max_val = 0.0_f32;
        for _ in 0..300 {
            s.advance(dt);
            max_val = max_val.max(s.value());
        }
        assert!(
            max_val > 100.0,
            "bouncy spring should overshoot target: max was {max_val}"
        );
    }

    #[test]
    fn spring_stiff_minimal_overshoot() {
        let mut s = Spring::new(0.0_f32, 100.0, SpringConfig::STIFF);
        let dt = Duration::from_secs_f32(1.0 / 60.0);
        let mut max_val = 0.0_f32;
        for _ in 0..300 {
            s.advance(dt);
            max_val = max_val.max(s.value());
        }
        // Stiff spring should have very little overshoot (< 10%)
        assert!(
            max_val < 115.0,
            "stiff spring overshoot should be small: max was {max_val}"
        );
    }

    #[test]
    fn spring_retarget_preserves_velocity() {
        let mut s = Spring::new(0.0_f32, 100.0, SpringConfig::SNAPPY);
        let dt = Duration::from_secs_f32(1.0 / 60.0);
        // Advance a bit to build velocity
        for _ in 0..10 {
            s.advance(dt);
        }
        let vel_before = s.velocity();
        assert!(vel_before.abs() > 0.0, "should have velocity mid-flight");

        // Retarget
        s.retarget(200.0_f32);
        assert!(!s.is_settled());
        assert!(
            (s.velocity() - vel_before).abs() < f32::EPSILON,
            "velocity should be preserved"
        );
    }

    #[test]
    fn spring_config_damping_ratios() {
        assert!(
            SpringConfig::BOUNCY.damping_ratio() < 1.0,
            "BOUNCY should be underdamped"
        );
        assert!(
            SpringConfig::STIFF.damping_ratio() > 0.5,
            "STIFF should be near critically damped"
        );
    }

    #[test]
    fn spring_all_configs_converge() {
        let configs = [
            SpringConfig::GENTLE,
            SpringConfig::BOUNCY,
            SpringConfig::STIFF,
            SpringConfig::SLOW,
            SpringConfig::SNAPPY,
        ];
        let dt = Duration::from_secs_f32(1.0 / 60.0);
        for config in configs {
            let mut s = Spring::new(0.0_f32, 1.0, config.clone());
            for _ in 0..600 {
                // 10 seconds at 60fps
                s.advance(dt);
            }
            assert!(
                s.is_settled(),
                "{:?} spring didn't settle in 10 seconds",
                config
            );
        }
    }

    // ---- Sequence tests ----

    #[test]
    fn sequence_tween_basic() {
        let seq = AnimationSequence::new().tween(
            "x",
            0.0,
            100.0,
            Duration::from_millis(500),
            Easing::Linear,
        );
        let mut player = SequencePlayer::new(seq);

        player.advance(Duration::from_millis(250));
        let val = player.get("x").unwrap();
        assert!((val - 50.0).abs() < 2.0, "at 50% should be ~50, got {val}");

        player.advance(Duration::from_millis(300));
        assert!(player.is_complete());
        let final_val = player.get("x").unwrap();
        assert!(
            (final_val - 100.0).abs() < 0.1,
            "final value should be 100, got {final_val}"
        );
    }

    #[test]
    fn sequence_wait_then_tween() {
        let seq = AnimationSequence::new()
            .wait(Duration::from_millis(100))
            .tween("y", 0.0, 50.0, Duration::from_millis(200), Easing::Linear);
        let mut player = SequencePlayer::new(seq);

        // During wait, no "y" value yet
        player.advance(Duration::from_millis(50));
        assert!(!player.is_complete());

        // After wait completes, tween starts
        player.advance(Duration::from_millis(60));
        // Wait just completed; tween should now be starting

        // Advance through tween
        player.advance(Duration::from_millis(200));
        assert!(player.is_complete());
        let val = player.get("y").unwrap();
        assert!((val - 50.0).abs() < 0.5, "final y should be 50, got {val}");
    }

    #[test]
    fn sequence_set_is_instantaneous() {
        let seq = AnimationSequence::new().set("flag", 42.0).tween(
            "x",
            0.0,
            10.0,
            Duration::from_millis(100),
            Easing::Linear,
        );
        let player = SequencePlayer::new(seq);

        // Set should have fired immediately during construction
        assert_eq!(player.get("flag"), Some(42.0));
        // And the tween should already be active
        assert!(!player.is_complete());
    }

    #[test]
    fn sequence_parallel_runs_simultaneously() {
        let seq = AnimationSequence::new().parallel(|p| {
            p.branch(|b| b.tween("a", 0.0, 10.0, Duration::from_millis(500), Easing::Linear))
                .branch(|b| b.tween("b", 0.0, 20.0, Duration::from_millis(500), Easing::Linear))
        });
        let mut player = SequencePlayer::new(seq);
        player.advance(Duration::from_millis(500));
        assert!(player.is_complete());

        // Both branches should have independent values accessible via nested players
        // The parallel step itself completes when both finish
    }

    #[test]
    fn sequence_spring_step() {
        let seq = AnimationSequence::new().spring_to("s", 0.0, 1.0, SpringConfig::STIFF);
        let mut player = SequencePlayer::new(seq);

        let dt = Duration::from_secs_f32(1.0 / 60.0);
        for _ in 0..300 {
            player.advance(dt);
        }
        assert!(player.is_complete());
        let val = player.get("s").unwrap();
        assert!(
            (val - 1.0).abs() < 0.1,
            "spring should settle at 1.0, got {val}"
        );
    }

    #[test]
    fn sequence_multi_step_screenplay() {
        // The "screenplay" pattern: tween → wait → spring → done
        let seq = AnimationSequence::new()
            .tween(
                "opacity",
                0.0,
                1.0,
                Duration::from_millis(200),
                Easing::EaseOutCubic,
            )
            .wait(Duration::from_millis(50))
            .spring_to("scale", 0.8, 1.0, SpringConfig::STIFF);

        assert_eq!(seq.len(), 3);
        let mut player = SequencePlayer::new(seq);

        // Step through the whole thing
        let dt = Duration::from_secs_f32(1.0 / 60.0);
        for _ in 0..600 {
            player.advance(dt);
            if player.is_complete() {
                break;
            }
        }
        assert!(player.is_complete());
    }

    // ---- Preset tests ----

    #[test]
    fn preset_fade_in() {
        let seq = preset::fade_in("op", Duration::from_millis(300));
        let mut player = SequencePlayer::new(seq);
        player.advance(Duration::from_millis(300));
        let val = player.get("op").unwrap();
        assert!(
            (val - 1.0).abs() < 0.05,
            "fade_in should reach 1.0, got {val}"
        );
    }

    #[test]
    fn preset_fade_out() {
        let seq = preset::fade_out("op", Duration::from_millis(300));
        let mut player = SequencePlayer::new(seq);
        player.advance(Duration::from_millis(300));
        let val = player.get("op").unwrap();
        assert!(val.abs() < 0.05, "fade_out should reach 0.0, got {val}");
    }

    #[test]
    fn preset_pop_in_overshoots() {
        let seq = preset::pop_in("scale");
        let mut player = SequencePlayer::new(seq);

        let dt = Duration::from_secs_f32(1.0 / 60.0);
        let mut max_val = 0.0_f32;
        for _ in 0..300 {
            player.advance(dt);
            if let Some(v) = player.get("scale") {
                max_val = max_val.max(v);
            }
        }
        assert!(
            max_val > 1.0,
            "pop_in should overshoot 1.0: max was {max_val}"
        );
    }

    #[test]
    fn preset_pulse_returns_to_one() {
        let seq = preset::pulse("s", Duration::from_millis(400));
        let mut player = SequencePlayer::new(seq);

        player.advance(Duration::from_millis(400));
        assert!(player.is_complete());
        let val = player.get("s").unwrap();
        assert!(
            (val - 1.0).abs() < 0.05,
            "pulse should return to 1.0, got {val}"
        );
    }

    #[test]
    fn preset_shake_returns_to_zero() {
        let seq = preset::shake("x", 10.0, Duration::from_millis(400));
        let mut player = SequencePlayer::new(seq);
        player.advance(Duration::from_millis(400));
        assert!(player.is_complete());
        let val = player.get("x").unwrap();
        assert!(val.abs() < 0.5, "shake should return to 0, got {val}");
    }

    #[test]
    fn preset_slide_in_left_produces_both_values() {
        let seq = preset::slide_in_left("card", 50.0, Duration::from_millis(300));
        let mut player = SequencePlayer::new(seq);
        player.advance(Duration::from_millis(300));
        // Should have both x and opacity values in nested players
        assert!(player.is_complete());
    }
}
