---
name: pixelchart
description: Generate pixel-perfect charts in the terminal from data
---

# Pixelchart Skill

This project includes `pixelchart`, a CLI tool that renders pixel-perfect,
anti-aliased charts directly in the terminal via the Kitty graphics protocol.

## When to Use

Use this skill whenever the user:
- Asks to visualize data, metrics, or statistics
- Wants to see a chart, plot, or graph
- Needs to generate chart images for documentation or reports
- Says things like "plot this", "chart this", "graph this", "show me a chart"

## How to Use

Pipe a JSON chart spec to `pixelchart render`:

```bash
echo '<JSON>' | pixelchart render [--output file.png]
```

If `--output` is omitted, the chart displays inline in the terminal.
If `--output` is specified, it saves to a PNG file.

## JSON Spec Format

Every spec has this shape:
```json
{
  "type": "<chart_type>",
  "data": { ... },
  "title": "Optional title",
  "x_label": "Optional X label",
  "y_label": "Optional Y label",
  "theme": "dark"
}
```

### Chart Types

| Type | `data` shape | Best for |
|------|-------------|----------|
| `line` | `{"y": [...]}` or `{"series": [...]}` | Time series, trends |
| `scatter` | `{"x": [...], "y": [...]}` | Correlations |
| `bar` | `{"labels": [...], "values": [...]}` | Categorical comparison |
| `histogram` | `{"values": [...]}` | Frequency distribution |
| `boxplot` | `{"groups": [{"label": "...", "values": [...]}]}` | Statistical summary |
| `heatmap` | `{"grid": [[...]]}` | 2D intensity |
| `pie` | `{"labels": [...], "values": [...]}` | Proportions |

### Multi-series Line Charts

```json
{
  "type": "line",
  "data": {
    "series": [
      {"label": "Series A", "values": [1, 2, 3]},
      {"label": "Series B", "values": [3, 2, 1]}
    ]
  }
}
```

### Line Chart Options

Add to `data`:
- `"smooth": true` — Catmull-Rom spline interpolation
- `"filled": true` — fill area under lines
- `"points": true` — show data point markers
- `"step": true` — stairstep rendering

### Explicit X Values

```json
{"type": "line", "data": {"x": [0, 0.5, 1.0, 1.5], "y": [1, 4, 2, 8]}}
```

## CLI Reference

```
pixelchart render [OPTIONS]
  -t, --type <TYPE>      Chart type (overrides JSON)
  -d, --data <JSON>      Inline JSON (alternative to stdin)
      --title <TEXT>      Chart title
      --x-label <TEXT>    X-axis label
      --y-label <TEXT>    Y-axis label
  -W, --width <PX>       Image width (default: 800)
  -H, --height <PX>      Image height (default: 500)
      --theme <THEME>     dark|light (default: dark)
  -o, --output <PATH>    Save to PNG file

pixelchart show <path.png>
  Display an existing PNG inline in the terminal

pixelchart info
  Print terminal capabilities and chart type examples
```

## Important Notes

- The binary is at `target/debug/pixelchart` (dev) or `target/release/pixelchart` (release)
- Build with: `cargo build -p pixelchart-cli`
- Inline display requires a Kitty-compatible terminal (Kitty, WezTerm, Ghostty)
- For non-Kitty terminals, always use `--output file.png` to save to a file
