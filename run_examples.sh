#!/usr/bin/env bash
# Run through all ratatui-pixelcanvas examples, grouped by theme.
# Press 'q' inside each example to advance to the next one.
# Ctrl+C to stop at any time.

set -euo pipefail
cd "$(dirname "$0")"

TOTAL=21
IDX=0

run() {
    local label="$1" name="$2" pkg="${3:-}"
    IDX=$((IDX + 1))
    echo "[$IDX/$TOTAL] $label: $name"
    if [[ -n "$pkg" ]]; then
        cargo run --example "$name" --release -p "$pkg" || true
    else
        cargo run --example "$name" --release || true
    fi
    echo ""
}

header() {
    echo ""
    echo "──── $1 ────"
    echo ""
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ratatui-pixelcanvas — Example Runner ($TOTAL examples)          ║"
echo "║   Press 'q' inside each to advance · Ctrl+C to stop        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Getting Started ────────────────────────────────────────────────
header "🟢 Getting Started"
run "Basics"          simple_shapes
run "3D wireframe"    cube_3d

# ── Feature Demos ─────────────────────────────────────────────────
header "🔷 Feature Demos"
run "Shape showcase"  showcase
run "Full features"   feature_showcase
run "New features"    new_features
run "Animations"      animation_demo

# ── Optical Illusions ─────────────────────────────────────────────
header "👁️  Optical Illusions"
run "Illusions"       illusions
run "Mind benders"    mind_benders

# ── Generative Art ────────────────────────────────────────────────
header "🎨 Generative Art"
run "Fractal dreams"     fractal_dreams
run "Sacred geometry"    sacred_geometry
run "Fluid symphony"     fluid_symphony
run "Hypnotic tunnels"   hypnotic_tunnels
run "Aurora borealis"    aurora_borealis

# ── Stress Tests ──────────────────────────────────────────────────
header "⚡ Stress Tests"
run "Powertest"       powertest

# ── Charts (pixelchart) ──────────────────────────────────────────
header "📊 Charts (pixelchart)"
run "Scatter"          scatter_demo     pixelchart
run "Dashboard"        dashboard        pixelchart
run "Chart demo"       demo             pixelchart
run "Chart showcase"   showcase         pixelchart
run "All charts"       chart_showcase   pixelchart
run "Interactive"      interactive      pixelchart
run "Robustness"       robustness_test  pixelchart

echo ""
echo "✅ All $TOTAL examples complete."
