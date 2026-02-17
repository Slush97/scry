// SPDX-License-Identifier: MIT OR Apache-2.0
//! Deterministic value noise and fractal Brownian motion for SDF materials.
//!
//! Ported from the permutation-table noise in `examples/fluid_symphony.rs`,
//! extended to 3D.

use super::math::Vec3;

/// Permutation table (doubled for index wrapping).
const PERM: [u8; 512] = {
    let base: [u8; 256] = [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30,
        69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94,
        252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171,
        168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60,
        211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1,
        216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
        164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
        126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
        213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39,
        253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
        242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49,
        192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
        138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    ];
    let mut out = [0u8; 512];
    let mut i = 0;
    while i < 512 {
        out[i] = base[i & 255];
        i += 1;
    }
    out
};

/// Smoothstep interpolation factor.
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Hash a 3D integer coordinate into 0..255 via the permutation table.
fn hash3(x: i32, y: i32, z: i32) -> u8 {
    let xi = (x & 255) as usize;
    let yi = (y & 255) as usize;
    let zi = (z & 255) as usize;
    PERM[(PERM[(PERM[xi] as usize + yi) & 511] as usize + zi) & 511]
}

/// 3D value noise, output in `[0, 1]`.
pub fn noise3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = smooth(xf);
    let v = smooth(yf);
    let w = smooth(zf);

    let c000 = f32::from(hash3(xi, yi, zi)) / 255.0;
    let c100 = f32::from(hash3(xi + 1, yi, zi)) / 255.0;
    let c010 = f32::from(hash3(xi, yi + 1, zi)) / 255.0;
    let c110 = f32::from(hash3(xi + 1, yi + 1, zi)) / 255.0;
    let c001 = f32::from(hash3(xi, yi, zi + 1)) / 255.0;
    let c101 = f32::from(hash3(xi + 1, yi, zi + 1)) / 255.0;
    let c011 = f32::from(hash3(xi, yi + 1, zi + 1)) / 255.0;
    let c111 = f32::from(hash3(xi + 1, yi + 1, zi + 1)) / 255.0;

    // Trilinear interpolation
    let x00 = c000 + u * (c100 - c000);
    let x10 = c010 + u * (c110 - c010);
    let x01 = c001 + u * (c101 - c001);
    let x11 = c011 + u * (c111 - c011);

    let y0 = x00 + v * (x10 - x00);
    let y1 = x01 + v * (x11 - x01);

    y0 + w * (y1 - y0)
}

/// 2D value noise, output in `[0, 1]`.
pub fn noise2d(x: f32, y: f32) -> f32 {
    noise3d(x, y, 0.0)
}

/// 3D fractal Brownian motion (multi-octave noise), output roughly in `[0, 1]`.
pub fn fbm3d(p: Vec3, octaves: u32) -> f32 {
    let mut value = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for _ in 0..octaves {
        value += amp * noise3d(p.x * freq, p.y * freq, p.z * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    value
}

/// 2D fractal Brownian motion, output roughly in `[0, 1]`.
pub fn fbm2d(x: f32, y: f32, octaves: u32) -> f32 {
    let mut value = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for _ in 0..octaves {
        value += amp * noise2d(x * freq, y * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise3d_in_range() {
        for i in 0..100 {
            let x = i as f32 * 0.37;
            let y = i as f32 * 0.73;
            let z = i as f32 * 0.51;
            let v = noise3d(x, y, z);
            assert!(
                (0.0..=1.0).contains(&v),
                "noise3d({x}, {y}, {z}) = {v} out of [0,1]"
            );
        }
    }

    #[test]
    fn noise3d_deterministic() {
        let a = noise3d(1.5, 2.3, 0.7);
        let b = noise3d(1.5, 2.3, 0.7);
        assert!((a - b).abs() < 1e-10);
    }

    #[test]
    fn fbm3d_in_reasonable_range() {
        let v = fbm3d(Vec3::new(3.14, 2.71, 1.41), 4);
        // FBM with 4 octaves: sum of amplitudes = 0.5 + 0.25 + 0.125 + 0.0625 = 0.9375
        assert!(v >= 0.0 && v <= 1.0, "fbm3d out of range: {v}");
    }

    #[test]
    fn fbm2d_in_reasonable_range() {
        let v = fbm2d(5.0, 7.0, 4);
        assert!(v >= 0.0 && v <= 1.0, "fbm2d out of range: {v}");
    }
}
