// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLI binary to generate the Sigil Mono TTF file.

fn main() {
    let params = scry_font::FontParams::default();
    let ttf_bytes = scry_font::generate_font(&params);

    let out_path = "SigilMono-Regular.ttf";
    std::fs::write(out_path, &ttf_bytes).expect("failed to write TTF");
    println!(
        "✓ Generated {} ({} bytes, {} glyphs)",
        out_path,
        ttf_bytes.len(),
        "44+" // approximate
    );
}
