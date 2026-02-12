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
//! use ratatui_pixelcanvas::scene::animation::{Easing, Transition};
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
    pub const BACK: Self = Self::Spring {
        overshoot: 1.70158,
    };

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
            Self::EaseInSine => {
                1.0 - (t * std::f32::consts::FRAC_PI_2).cos()
            }
            Self::EaseOutSine => {
                (t * std::f32::consts::FRAC_PI_2).sin()
            }
            Self::EaseInOutSine => {
                0.5 * (1.0 - (std::f32::consts::PI * t).cos())
            }

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
            Self::EaseInCirc => {
                1.0 - (1.0 - t * t).sqrt()
            }
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
                -((2.0_f32).powf(10.0 * u)
                    * ((u - s) * std::f32::consts::TAU / p).sin())
                    + 1.0
            }

            // Cubic Bézier
            Self::CubicBezier { x1, y1, x2, y2 } => {
                cubic_bezier_ease(*x1, *y1, *x2, *y2, t)
            }
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
    (3.0 * (1.0 - 3.0 * x2 + 3.0 * x1)).mul_add(t, 2.0 * (3.0 * x2 - 6.0 * x1)).mul_add(t, 3.0 * x1)
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
    /// Interpolate in linear RGB space with alpha.
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
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
    /// Interpolate each matrix component independently.
    ///
    /// This produces correct results for translations and scales, and
    /// reasonable approximations for rotations when the angle difference
    /// is small (< 180°). For large rotations, decompose into angle +
    /// translate + scale and interpolate those separately.
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            sx: self.sx + (other.sx - self.sx) * t,
            kx: self.kx + (other.kx - self.kx) * t,
            ky: self.ky + (other.ky - self.ky) * t,
            sy: self.sy + (other.sy - self.sy) * t,
            tx: self.tx + (other.tx - self.tx) * t,
            ty: self.ty + (other.ty - self.ty) * t,
        }
    }
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
/// use ratatui_pixelcanvas::scene::animation::{Easing, Transition};
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
/// use ratatui_pixelcanvas::scene::animation::{Easing, Keyframes, Keyframe};
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
        frames.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap_or(std::cmp::Ordering::Equal));
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
/// use ratatui_pixelcanvas::scene::animation::{AnimationState, Easing};
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
        let mut completed = Vec::new();
        for (id, anim) in &mut self.animations {
            anim.advance_any(dt);
            if anim.is_complete_any() {
                completed.push(*id);
            }
        }
        for id in completed {
            self.animations.remove(id);
        }
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

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
            assert!(
                (v - t).abs() < 0.05,
                "cubic-bezier linear at {t}: got {v}"
            );
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
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.g - 0.5).abs() < 0.01);
        assert!((mid.b - 0.5).abs() < 0.01);
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
            Keyframe { position: 0.0, value: 0.0_f32, easing: Easing::Linear },
            Keyframe { position: 1.0, value: 100.0, easing: Easing::Linear },
        ]);
        assert!((kf.value_at(0.0)).abs() < f32::EPSILON);
        assert!((kf.value_at(0.5) - 50.0).abs() < f32::EPSILON);
        assert!((kf.value_at(1.0) - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn keyframes_three_stops() {
        let kf = Keyframes::new(vec![
            Keyframe { position: 0.0, value: 0.0_f32, easing: Easing::Linear },
            Keyframe { position: 0.5, value: 100.0, easing: Easing::Linear },
            Keyframe { position: 1.0, value: 50.0, easing: Easing::Linear },
        ]);
        assert!((kf.value_at(0.25) - 50.0).abs() < f32::EPSILON); // 0→100 at 50%
        assert!((kf.value_at(0.75) - 75.0).abs() < f32::EPSILON); // 100→50 at 50%
    }

    #[test]
    fn keyframes_clamps_edges() {
        let kf = Keyframes::new(vec![
            Keyframe { position: 0.2, value: 10.0_f32, easing: Easing::Linear },
            Keyframe { position: 0.8, value: 90.0, easing: Easing::Linear },
        ]);
        assert!((kf.value_at(0.0) - 10.0).abs() < f32::EPSILON);
        assert!((kf.value_at(1.0) - 90.0).abs() < f32::EPSILON);
    }

    #[test]
    fn keyframes_per_segment_easing() {
        let kf = Keyframes::new(vec![
            Keyframe { position: 0.0, value: 0.0_f32, easing: Easing::EaseInQuad },
            Keyframe { position: 1.0, value: 100.0, easing: Easing::Linear },
        ]);
        // EaseInQuad at 50%: 0.5^2 = 0.25 → value = 25
        let val = kf.value_at(0.5);
        assert!((val - 25.0).abs() < 0.1);
    }

    #[test]
    #[should_panic(expected = "at least 2 keyframes")]
    fn keyframes_too_few_panics() {
        let _ = Keyframes::new(vec![
            Keyframe { position: 0.0, value: 0.0_f32, easing: Easing::Linear },
        ]);
    }

    // ---- AnimationState tests ----

    #[test]
    fn animation_state_basic() {
        let mut state = AnimationState::new();
        assert!(state.is_idle());

        state.start("opacity", 0.0_f32, 1.0_f32, Duration::from_millis(500), Easing::Linear);
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
        state.start("a", 0.0_f32, 10.0, Duration::from_millis(100), Easing::Linear);
        state.start("b", 0.0_f32, 20.0, Duration::from_millis(200), Easing::Linear);
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
}
