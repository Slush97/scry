# Examples

Curated examples organized by crate. Each demonstrates a specific API concept.

## scry-engine (`examples/`)

| Example | Features | Description |
|---------|----------|-------------|
| `simple_shapes` | — | Basic drawing primitives: rectangles, circles, lines, polygons |
| `line_drawing` | `svg`, `widget` | SVG line drawing animation with performance diagnostics |
| `animation_demo` | — | Animation system: easing, keyframes, timelines |
| `spring_sequence_demo` | — | Spring physics, coroutine sequencing, preset library |
| `sdf_showcase` | `sdf` | Interactive ray-marched 3D scenes in the terminal |
| `text3d_showcase` | `sdf-text` | Extruded 3D text with five material presets |
| `cube_3d` | — | 3D wireframe cube rendered from scratch |
| `scatter3d` | — | Interactive 3D scatter plot in the terminal |
| `masonic_mirror` | `sdf`, `widget` | SDF 3D mirror room with checkerboard, pillars, sphere |
| `showcase` | — | All drawing primitives and style options |
| `chart_integration` | — | Chart features accessed from within the engine |
| `ml_3d_viz` | — | 3D ML visualization demo |
| `mission_control` | — | Space telemetry dashboard — full integration demo |
| `pixel_dashboard` | — | Composability showcase: multiple widgets |
| `dual_y_demo` | — | Dual Y-axis line chart, PNG export |
| `window_demo` | `window` | Native OS window via the `window` backend |
| `fastfetch_anim` | — | Replace the ASCII logo with a live pixel animation |
| `circus_ball` | `text`, `widget` | Flash-style bouncing ball animation |
| `particle_life` | — | Emergent behavior from simple interaction rules |

### Gallery (`examples/gallery/`)

Artistic demos showcasing what's possible with the rendering engine.

| Example | Description |
|---------|-------------|
| `aurora_borealis` | Northern lights simulation |
| `fractal_dreams` | Animated Mandelbrot/Julia set explorer |
| `fluid_symphony` | Curl-noise vector field with flowing particles |
| `hypnotic_tunnels` | Infinite recursive polygon tunnel vortex |
| `illusions` | Optical illusions exploiting human visual perception |
| `mind_benders` | Advanced optical illusions |
| `obsidian_mirror` | Esoteric scrying animation (requires `sdf`) |
| `postmodern_manifesto` | Layered collage: gradients, transforms, blending |
| `sacred_geometry` | Living geometric construction |
| `wave_interference` | Interactive physics wave visualization |

---

## scry-chart (`crates/scry-chart/examples/`)

| Example | Description |
|---------|-------------|
| `demo` | Every chart type, feature, and theme |
| `showcase` | Comprehensive feature showcase |
| `render_png` | Export charts to PNG files |
| `render_all` | Batch render every chart type to PNG |
| `scatter_demo` | Interactive scatter plot |
| `subplot_demo` | 2×2 grid of different chart types |
| `dashboard` | Multi-chart split layout |
| `interactive` | Crosshair, tooltip, zoom/pan |
| `chart_gallery` | One PNG per chart type for visual reference |
| `advanced_charts` | Tier 2 chart types: funnel, gauge, heatmap, treemap |
| `font_scaling_demo` | Charts at multiple resolutions |

---

## scry-learn (`crates/scry-learn/examples/`)

| Example | Features | Description |
|---------|----------|-------------|
| `industry_report` | — | 5-fold CV accuracy benchmarks on UCI datasets |
| `ml_viz_showcase` | `viz` | ML + chart integration showcase |
| `live_training` | `live-plot` | Live training visualization with inline charts |
