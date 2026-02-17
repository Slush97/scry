// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bicubic (Catmull-Rom) upscaler for RGBA pixel buffers.
//!
//! Separable two-pass implementation: horizontal then vertical. Each pass
//! evaluates a 4-tap Catmull-Rom kernel, giving smooth interpolation without
//! the blurriness of bilinear or the ringing of sharper cubic variants.
//!
//! Zero dependencies — operates on raw `&[u8]` RGBA data.

/// Compute the 4 Catmull-Rom basis weights for fractional position `t` in `[0, 1)`.
///
/// The taps correspond to samples at positions −1, 0, +1, +2 relative to the
/// integer part of the source coordinate.
#[inline]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Upscale an RGBA buffer from `(src_w, src_h)` to `(dst_w, dst_h)` using a
/// separable Catmull-Rom bicubic filter.
///
/// `src` must contain exactly `src_w * src_h * 4` bytes (RGBA). Returns a
/// `Vec<u8>` of length `dst_w * dst_h * 4`.
///
/// Edge pixels are handled by clamping source indices (no wrapping).
pub fn upscale_bicubic(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    debug_assert_eq!(src.len(), (src_w * src_h * 4) as usize);

    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }

    // Pass 1: horizontal — produces (dst_w × src_h) intermediate buffer
    let mut tmp = vec![0u8; (dst_w * src_h * 4) as usize];
    let x_ratio = src_w as f32 / dst_w as f32;

    for y in 0..src_h {
        let src_row = (y * src_w * 4) as usize;
        let dst_row = (y * dst_w * 4) as usize;
        for x in 0..dst_w {
            let sx = x as f32 * x_ratio + (x_ratio - 1.0) * 0.5;
            let si = sx.floor() as i32;
            let t = sx - si as f32;
            let w = catmull_rom_weights(t);

            let mut rgba = [0.0f32; 4];
            for (k, &wk) in w.iter().enumerate() {
                let cx = (si + k as i32 - 1).clamp(0, src_w as i32 - 1) as usize;
                let off = src_row + cx * 4;
                for c in 0..4 {
                    rgba[c] += src[off + c] as f32 * wk;
                }
            }

            let dst_off = dst_row + x as usize * 4;
            for c in 0..4 {
                tmp[dst_off + c] = rgba[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    // Pass 2: vertical — produces (dst_w × dst_h) final buffer
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];
    let y_ratio = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        let sy = y as f32 * y_ratio + (y_ratio - 1.0) * 0.5;
        let si = sy.floor() as i32;
        let t = sy - si as f32;
        let w = catmull_rom_weights(t);

        let dst_row = (y * dst_w * 4) as usize;
        for x in 0..dst_w {
            let mut rgba = [0.0f32; 4];
            for (k, &wk) in w.iter().enumerate() {
                let cy = (si + k as i32 - 1).clamp(0, src_h as i32 - 1) as usize;
                let off = cy * (dst_w as usize) * 4 + x as usize * 4;
                for c in 0..4 {
                    rgba[c] += tmp[off + c] as f32 * wk;
                }
            }

            let dst_off = dst_row + x as usize * 4;
            for c in 0..4 {
                dst[dst_off + c] = rgba[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_same_size() {
        let src = vec![128u8; 4 * 4 * 4]; // 4×4 RGBA
        let out = upscale_bicubic(&src, 4, 4, 4, 4);
        assert_eq!(out, src);
    }

    #[test]
    fn output_has_correct_length() {
        let src = vec![100u8; 8 * 6 * 4]; // 8×6
        let out = upscale_bicubic(&src, 8, 6, 32, 24);
        assert_eq!(out.len(), 32 * 24 * 4);
    }

    #[test]
    fn uniform_image_stays_uniform() {
        let src = vec![42u8; 4 * 4 * 4];
        let out = upscale_bicubic(&src, 4, 4, 16, 16);
        // Every pixel should be very close to 42 (exact for uniform input)
        for &v in &out {
            assert!(
                (v as i32 - 42).unsigned_abs() <= 1,
                "expected ~42, got {v}"
            );
        }
    }

    #[test]
    fn upscale_2x_produces_smooth_gradient() {
        // 2×1 image: black → white
        let src = [0, 0, 0, 255, 255, 255, 255, 255];
        let out = upscale_bicubic(&src, 2, 1, 4, 1);
        assert_eq!(out.len(), 16);
        // Middle pixels should be interpolated (not just black or white)
        let mid_r = out[4]; // pixel 1 red channel
        assert!(mid_r > 0 && mid_r < 255, "expected interpolated value, got {mid_r}");
    }

    #[test]
    fn weights_sum_to_one() {
        for i in 0..10 {
            let t = i as f32 / 10.0;
            let w = catmull_rom_weights(t);
            let sum: f32 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "weights at t={t} sum to {sum}, expected 1.0"
            );
        }
    }
}
