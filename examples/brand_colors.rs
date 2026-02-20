//! Brand color scheme explorer — renders three brand options as PNG swatches.
//!
//! Run with: `cargo run --example brand_colors --release`

use scry_engine::rasterize::Rasterizer;
use scry_engine::scene::style::Point;
use scry_engine::scene::PixelCanvas;
use scry_engine::style::Color;

const W: u32 = 1200;
const H: u32 = 700;

/// Draws a rounded rectangle swatch of a single color with a label strip below.
fn swatch(canvas: PixelCanvas, x: f32, y: f32, w: f32, h: f32, color: Color) -> PixelCanvas {
    canvas
        .rect(x, y, w, h)
        .fill(color)
        .corner_radius(12.0)
        .done()
}

/// Draws a horizontal gradient bar.
fn gradient_bar(
    canvas: PixelCanvas,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    colors: &[(f32, Color)],
) -> PixelCanvas {
    let mut g = canvas
        .gradient(x, y, w, h)
        .linear(Point::new(x, y), Point::new(x + w, y));
    for &(pos, color) in colors {
        g = g.stop(pos, color);
    }
    g.done()
}

// ─────────────────────────────── Option A ───────────────────────────────

fn option_a_crystal_blue() -> PixelCanvas {
    let bg = Color::from_rgba8(250, 251, 254, 255);
    let primary = Color::from_rgba8(74, 158, 255, 255);     // #4A9EFF
    let secondary = Color::from_rgba8(34, 197, 94, 255);    // #22C55E
    let accent = Color::from_rgba8(244, 114, 182, 255);     // #F472B6
    let dark = Color::from_rgba8(30, 41, 59, 255);           // #1E293B
    let muted = Color::from_rgba8(148, 163, 184, 255);       // #94A3B8

    let c = PixelCanvas::new(W, H).background(bg);

    // Title bar
    let c = c.rect(0.0, 0.0, W as f32, 80.0)
        .fill(Color::WHITE)
        .done();

    // "scry" wordmark area in primary
    let c = c.rect(40.0, 20.0, 120.0, 40.0)
        .fill(primary)
        .corner_radius(8.0)
        .done();

    // Main color swatches — large row
    let sw = 160.0;
    let sh = 160.0;
    let gap = 24.0;
    let start_x = 40.0;
    let start_y = 110.0;

    let c = swatch(c, start_x, start_y, sw, sh, primary);
    let c = swatch(c, start_x + sw + gap, start_y, sw, sh, secondary);
    let c = swatch(c, start_x + (sw + gap) * 2.0, start_y, sw, sh, accent);
    let c = swatch(c, start_x + (sw + gap) * 3.0, start_y, sw, sh, dark);
    let c = swatch(c, start_x + (sw + gap) * 4.0, start_y, sw, sh, muted);

    // Labels beneath swatches
    let label_y = start_y + sh + 8.0;
    let label_h = 6.0;
    let c = c.rect(start_x, label_y, sw, label_h).fill(primary).done();
    let c = c.rect(start_x + sw + gap, label_y, sw, label_h).fill(secondary).done();
    let c = c.rect(start_x + (sw + gap) * 2.0, label_y, sw, label_h).fill(accent).done();
    let c = c.rect(start_x + (sw + gap) * 3.0, label_y, sw, label_h).fill(dark).done();
    let c = c.rect(start_x + (sw + gap) * 4.0, label_y, sw, label_h).fill(muted).done();

    // Gradient preview
    let c = gradient_bar(
        c,
        40.0,
        330.0,
        W as f32 - 80.0,
        40.0,
        &[
            (0.0, primary),
            (0.33, secondary),
            (0.66, accent),
            (1.0, primary),
        ],
    );

    // UI mock: card on light bg
    let c = c.rect(40.0, 400.0, 350.0, 250.0)
        .fill(Color::WHITE)
        .corner_radius(16.0)
        .stroke(Color::from_rgba8(226, 232, 240, 255), 1.0)
        .done();

    // Card header
    let c = c.rect(56.0, 416.0, 200.0, 24.0)
        .fill(dark)
        .corner_radius(4.0)
        .done();

    // Card body lines
    let c = c.rect(56.0, 456.0, 300.0, 12.0).fill(muted).corner_radius(2.0).done();
    let c = c.rect(56.0, 478.0, 260.0, 12.0).fill(muted).corner_radius(2.0).done();
    let c = c.rect(56.0, 500.0, 280.0, 12.0).fill(muted).corner_radius(2.0).done();

    // Card button
    let c = c.rect(56.0, 536.0, 120.0, 36.0)
        .fill(primary)
        .corner_radius(8.0)
        .done();

    // Accent circle
    let c = c.circle(330.0, 560.0, 24.0)
        .fill(secondary)
        .done();

    // Dark panel UI mock
    let c = c.rect(420.0, 400.0, 350.0, 250.0)
        .fill(dark)
        .corner_radius(16.0)
        .done();

    // Panel elements
    let c = c.rect(440.0, 420.0, 200.0, 16.0)
        .fill(Color::from_rgba8(255, 255, 255, 200))
        .corner_radius(3.0)
        .done();
    let c = c.rect(440.0, 452.0, 310.0, 8.0).fill(Color::from_rgba8(255, 255, 255, 60)).corner_radius(2.0).done();
    let c = c.rect(440.0, 468.0, 280.0, 8.0).fill(Color::from_rgba8(255, 255, 255, 60)).corner_radius(2.0).done();

    // Chart bars in panel
    let bar_base = 580.0;
    let c = c.rect(440.0, bar_base, 30.0, -80.0).fill(primary).corner_radius(4.0).done();
    let c = c.rect(480.0, bar_base, 30.0, -120.0).fill(secondary).corner_radius(4.0).done();
    let c = c.rect(520.0, bar_base, 30.0, -60.0).fill(accent).corner_radius(4.0).done();
    let c = c.rect(560.0, bar_base, 30.0, -100.0).fill(primary).corner_radius(4.0).done();
    let c = c.rect(600.0, bar_base, 30.0, -140.0).fill(secondary).corner_radius(4.0).done();
    let c = c.rect(640.0, bar_base, 30.0, -90.0).fill(accent).corner_radius(4.0).done();

    // Icon circles top-right
    let c = c.circle(900.0, 150.0, 60.0).fill(primary).done();
    let c = c.circle(900.0, 150.0, 40.0)
        .fill(Color::from_rgba8(255, 255, 255, 80))
        .done();
    let c = c.circle(1050.0, 180.0, 45.0).fill(secondary).done();
    let c = c.circle(1050.0, 180.0, 28.0)
        .fill(Color::from_rgba8(255, 255, 255, 80))
        .done();

    // Floating orbs
    let c = c.circle(850.0, 400.0, 80.0)
        .fill(Color::from_rgba8(74, 158, 255, 40))
        .done();
    let c = c.circle(1000.0, 500.0, 100.0)
        .fill(Color::from_rgba8(34, 197, 94, 30))
        .done();

    c
}

// ─────────────────────────────── Option B ───────────────────────────────

fn option_b_obsidian_electric() -> PixelCanvas {
    let bg = Color::from_rgba8(15, 23, 42, 255);             // #0F172A
    let primary = Color::from_rgba8(56, 189, 248, 255);      // #38BDF8
    let secondary = Color::from_rgba8(167, 139, 250, 255);   // #A78BFA
    let accent = Color::from_rgba8(251, 191, 36, 255);       // #FBBF24
    let surface = Color::from_rgba8(30, 41, 59, 255);        // #1E293B
    let muted = Color::from_rgba8(71, 85, 105, 255);         // #475569

    let c = PixelCanvas::new(W, H).background(bg);

    // Top bar surface
    let c = c.rect(0.0, 0.0, W as f32, 80.0)
        .fill(surface)
        .done();

    // "scry" wordmark in electric blue
    let c = c.rect(40.0, 20.0, 120.0, 40.0)
        .fill(primary)
        .corner_radius(8.0)
        .done();

    // Color swatches
    let sw = 160.0;
    let sh = 160.0;
    let gap = 24.0;
    let start_x = 40.0;
    let start_y = 110.0;

    let c = swatch(c, start_x, start_y, sw, sh, primary);
    let c = swatch(c, start_x + sw + gap, start_y, sw, sh, secondary);
    let c = swatch(c, start_x + (sw + gap) * 2.0, start_y, sw, sh, accent);
    let c = swatch(c, start_x + (sw + gap) * 3.0, start_y, sw, sh, surface);
    let c = swatch(c, start_x + (sw + gap) * 4.0, start_y, sw, sh, muted);

    // Glow lines beneath swatches
    let label_y = start_y + sh + 8.0;
    let c = c.rect(start_x, label_y, sw, 6.0).fill(primary).done();
    let c = c.rect(start_x + sw + gap, label_y, sw, 6.0).fill(secondary).done();
    let c = c.rect(start_x + (sw + gap) * 2.0, label_y, sw, 6.0).fill(accent).done();
    let c = c.rect(start_x + (sw + gap) * 3.0, label_y, sw, 6.0).fill(Color::from_rgba8(71, 85, 105, 200)).done();
    let c = c.rect(start_x + (sw + gap) * 4.0, label_y, sw, 6.0).fill(Color::from_rgba8(51, 65, 85, 200)).done();

    // Gradient: blue → violet → gold
    let c = gradient_bar(
        c,
        40.0,
        330.0,
        W as f32 - 80.0,
        40.0,
        &[
            (0.0, primary),
            (0.5, secondary),
            (1.0, accent),
        ],
    );

    // Dark card
    let c = c.rect(40.0, 400.0, 350.0, 250.0)
        .fill(surface)
        .corner_radius(16.0)
        .stroke(Color::from_rgba8(56, 189, 248, 40), 1.0)
        .done();

    // Card header
    let c = c.rect(56.0, 416.0, 200.0, 24.0)
        .fill(Color::from_rgba8(255, 255, 255, 220))
        .corner_radius(4.0)
        .done();

    // Card body lines
    let c = c.rect(56.0, 456.0, 300.0, 10.0).fill(muted).corner_radius(2.0).done();
    let c = c.rect(56.0, 476.0, 260.0, 10.0).fill(muted).corner_radius(2.0).done();
    let c = c.rect(56.0, 496.0, 280.0, 10.0).fill(muted).corner_radius(2.0).done();

    // CTA button — electric blue
    let c = c.rect(56.0, 530.0, 120.0, 36.0)
        .fill(primary)
        .corner_radius(8.0)
        .done();

    // Accent indicator
    let c = c.circle(330.0, 560.0, 20.0)
        .fill(accent)
        .done();

    // Second card — deeper
    let c = c.rect(420.0, 400.0, 350.0, 250.0)
        .fill(Color::from_rgba8(22, 33, 50, 255))
        .corner_radius(16.0)
        .stroke(Color::from_rgba8(167, 139, 250, 40), 1.0)
        .done();

    // Panel header
    let c = c.rect(440.0, 420.0, 200.0, 14.0)
        .fill(Color::from_rgba8(255, 255, 255, 180))
        .corner_radius(3.0)
        .done();

    // Chart bars — electric palette
    let bar_base = 580.0;
    let c = c.rect(440.0, bar_base, 30.0, -80.0).fill(primary).corner_radius(4.0).done();
    let c = c.rect(480.0, bar_base, 30.0, -130.0).fill(secondary).corner_radius(4.0).done();
    let c = c.rect(520.0, bar_base, 30.0, -60.0).fill(accent).corner_radius(4.0).done();
    let c = c.rect(560.0, bar_base, 30.0, -110.0).fill(primary).corner_radius(4.0).done();
    let c = c.rect(600.0, bar_base, 30.0, -150.0).fill(secondary).corner_radius(4.0).done();
    let c = c.rect(640.0, bar_base, 30.0, -90.0).fill(accent).corner_radius(4.0).done();

    // Crystal orb — the scrying metaphor
    let c = c.circle(950.0, 250.0, 90.0)
        .fill(Color::from_rgba8(56, 189, 248, 15))
        .done();
    let c = c.circle(950.0, 250.0, 70.0)
        .fill(Color::from_rgba8(167, 139, 250, 20))
        .done();
    let c = c.circle(950.0, 250.0, 50.0)
        .fill(Color::from_rgba8(56, 189, 248, 30))
        .done();
    let c = c.circle(950.0, 250.0, 30.0)
        .fill(Color::from_rgba8(255, 255, 255, 20))
        .done();
    let c = c.circle(950.0, 250.0, 90.0)
        .stroke(Color::from_rgba8(56, 189, 248, 60), 2.0)
        .done();

    // Floating glow orbs
    let c = c.circle(850.0, 500.0, 60.0)
        .fill(Color::from_rgba8(56, 189, 248, 20))
        .done();
    let c = c.circle(1050.0, 450.0, 80.0)
        .fill(Color::from_rgba8(167, 139, 250, 15))
        .done();
    let c = c.circle(1100.0, 600.0, 50.0)
        .fill(Color::from_rgba8(251, 191, 36, 15))
        .done();

    // Subtle grid lines
    let c = c.line(800.0, 400.0, 1160.0, 400.0)
        .color(Color::from_rgba8(255, 255, 255, 15))
        .width(1.0)
        .done();
    let c = c.line(800.0, 500.0, 1160.0, 500.0)
        .color(Color::from_rgba8(255, 255, 255, 15))
        .width(1.0)
        .done();
    let c = c.line(800.0, 600.0, 1160.0, 600.0)
        .color(Color::from_rgba8(255, 255, 255, 15))
        .width(1.0)
        .done();

    c
}

// ─────────────────────────────── Option C ───────────────────────────────

fn option_c_prismatic() -> PixelCanvas {
    let bg = Color::from_rgba8(15, 20, 35, 255);
    let deep_navy = Color::from_rgba8(30, 41, 59, 255);      // #1E293B
    let prism_red = Color::from_rgba8(248, 113, 113, 255);    // #F87171
    let prism_orange = Color::from_rgba8(251, 146, 60, 255);  // #FB923C
    let prism_yellow = Color::from_rgba8(250, 204, 21, 255);  // #FACC15
    let prism_green = Color::from_rgba8(74, 222, 128, 255);   // #4ADE80
    let prism_blue = Color::from_rgba8(96, 165, 250, 255);    // #60A5FA
    let prism_violet = Color::from_rgba8(167, 139, 250, 255); // #A78BFA

    let c = PixelCanvas::new(W, H).background(bg);

    // Top bar
    let c = c.rect(0.0, 0.0, W as f32, 80.0)
        .fill(deep_navy)
        .done();

    // "scry" wordmark — prismatic gradient
    let c = gradient_bar(c, 40.0, 20.0, 120.0, 40.0, &[
        (0.0, prism_blue),
        (0.5, prism_violet),
        (1.0, prism_red),
    ]);

    // Color swatches — the full spectrum
    let sw = 120.0;
    let sh = 160.0;
    let gap = 16.0;
    let start_x = 40.0;
    let start_y = 110.0;

    let colors = [prism_red, prism_orange, prism_yellow, prism_green, prism_blue, prism_violet, deep_navy];
    let mut c = c;
    for (i, &color) in colors.iter().enumerate() {
        let x = start_x + (sw + gap) * i as f32;
        c = swatch(c, x, start_y, sw, sh, color);
        c = c.rect(x, start_y + sh + 8.0, sw, 6.0).fill(color).done();
    }

    // Full prismatic gradient — the signature element
    let c = gradient_bar(
        c,
        40.0,
        330.0,
        W as f32 - 80.0,
        50.0,
        &[
            (0.0, prism_red),
            (0.17, prism_orange),
            (0.33, prism_yellow),
            (0.5, prism_green),
            (0.67, prism_blue),
            (0.83, prism_violet),
            (1.0, prism_red),
        ],
    );

    // Dark card with prismatic accent stroke
    let c = c.rect(40.0, 410.0, 350.0, 240.0)
        .fill(deep_navy)
        .corner_radius(16.0)
        .done();

    // Prismatic top border on card
    let c = gradient_bar(c, 40.0, 410.0, 350.0, 4.0, &[
        (0.0, prism_blue),
        (0.5, prism_violet),
        (1.0, prism_red),
    ]);

    // Card content
    let c = c.rect(56.0, 430.0, 200.0, 18.0)
        .fill(Color::from_rgba8(255, 255, 255, 200))
        .corner_radius(3.0)
        .done();
    let muted = Color::from_rgba8(100, 116, 139, 255);
    let c = c.rect(56.0, 462.0, 300.0, 10.0).fill(muted).corner_radius(2.0).done();
    let c = c.rect(56.0, 480.0, 260.0, 10.0).fill(muted).corner_radius(2.0).done();

    // Prismatic button
    let c = gradient_bar(c, 56.0, 520.0, 140.0, 36.0, &[
        (0.0, prism_blue),
        (1.0, prism_violet),
    ]);

    // Chart card
    let c = c.rect(420.0, 410.0, 350.0, 240.0)
        .fill(Color::from_rgba8(22, 28, 45, 255))
        .corner_radius(16.0)
        .done();

    // Prismatic top border
    let c = gradient_bar(c, 420.0, 410.0, 350.0, 4.0, &[
        (0.0, prism_green),
        (0.5, prism_yellow),
        (1.0, prism_orange),
    ]);

    // Chart bars — each a different spectrum color
    let bar_base = 590.0;
    let c = c.rect(440.0, bar_base, 30.0, -80.0).fill(prism_red).corner_radius(4.0).done();
    let c = c.rect(480.0, bar_base, 30.0, -130.0).fill(prism_orange).corner_radius(4.0).done();
    let c = c.rect(520.0, bar_base, 30.0, -60.0).fill(prism_yellow).corner_radius(4.0).done();
    let c = c.rect(560.0, bar_base, 30.0, -110.0).fill(prism_green).corner_radius(4.0).done();
    let c = c.rect(600.0, bar_base, 30.0, -150.0).fill(prism_blue).corner_radius(4.0).done();
    let c = c.rect(640.0, bar_base, 30.0, -90.0).fill(prism_violet).corner_radius(4.0).done();

    // Prism refraction — triangle with rainbow
    let prism_cx = 950.0;
    let prism_cy = 200.0;
    let prism_r = 70.0;

    // Triangular prism (approximated as 3 lines)
    let c = c.line(prism_cx, prism_cy - prism_r, prism_cx - prism_r * 0.866, prism_cy + prism_r * 0.5)
        .color(Color::from_rgba8(255, 255, 255, 120))
        .width(2.0)
        .done();
    let c = c.line(prism_cx - prism_r * 0.866, prism_cy + prism_r * 0.5, prism_cx + prism_r * 0.866, prism_cy + prism_r * 0.5)
        .color(Color::from_rgba8(255, 255, 255, 120))
        .width(2.0)
        .done();
    let c = c.line(prism_cx + prism_r * 0.866, prism_cy + prism_r * 0.5, prism_cx, prism_cy - prism_r)
        .color(Color::from_rgba8(255, 255, 255, 120))
        .width(2.0)
        .done();

    // Incoming light beam
    let c = c.line(800.0, prism_cy, prism_cx - 40.0, prism_cy)
        .color(Color::from_rgba8(255, 255, 255, 100))
        .width(3.0)
        .done();

    // Refracted rainbow beams
    let rainbow = [prism_red, prism_orange, prism_yellow, prism_green, prism_blue, prism_violet];
    let mut c = c;
    for (i, &color) in rainbow.iter().enumerate() {
        let angle = -0.4 + i as f32 * 0.16;
        let end_x = prism_cx + 40.0 + 180.0 * angle.cos();
        let end_y = prism_cy + 180.0 * angle.sin();
        c = c.line(prism_cx + 40.0, prism_cy, end_x, end_y)
            .color(color)
            .width(2.0)
            .done();
    }

    // Floating prismatic orbs
    let c = c.circle(880.0, 500.0, 50.0)
        .fill(Color::from_rgba8(96, 165, 250, 20))
        .done();
    let c = c.circle(1050.0, 550.0, 70.0)
        .fill(Color::from_rgba8(167, 139, 250, 15))
        .done();
    let c = c.circle(1100.0, 420.0, 40.0)
        .fill(Color::from_rgba8(248, 113, 113, 15))
        .done();

    c
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::Path::new("/tmp/scry_brand");
    std::fs::create_dir_all(out_dir)?;

    // ── Option A: Crystal Blue ──
    let canvas_a = option_a_crystal_blue();
    let pixmap_a = Rasterizer::rasterize(&canvas_a)?;
    let path_a = out_dir.join("option_a_crystal_blue.png");
    pixmap_a.save_png(&path_a)?;
    eprintln!("✓ Saved {}", path_a.display());

    // ── Option B: Obsidian + Electric ──
    let canvas_b = option_b_obsidian_electric();
    let pixmap_b = Rasterizer::rasterize(&canvas_b)?;
    let path_b = out_dir.join("option_b_obsidian_electric.png");
    pixmap_b.save_png(&path_b)?;
    eprintln!("✓ Saved {}", path_b.display());

    // ── Option C: Prismatic ──
    let canvas_c = option_c_prismatic();
    let pixmap_c = Rasterizer::rasterize(&canvas_c)?;
    let path_c = out_dir.join("option_c_prismatic.png");
    pixmap_c.save_png(&path_c)?;
    eprintln!("✓ Saved {}", path_c.display());

    eprintln!("\n🎨 All three brand options rendered to {}", out_dir.display());
    eprintln!("   Open them to compare!");

    Ok(())
}
