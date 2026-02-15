---
name: pixelchart
description: Generate pixel-perfect charts in the terminal from data using the pixelchart CLI
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
- Has CSV/TSV data they want plotted
- Wants to compare series, show distributions, or display proportions

## Prerequisites

Build the CLI if not already built:
```bash
cargo build -p pixelchart-cli
```

The binary is at `target/debug/pixelchart` (dev) or `target/release/pixelchart` (release).

## Commands

### 1. `render` — JSON → Chart

Pipe a JSON chart spec to render:
```bash
echo '<JSON>' | pixelchart render [OPTIONS]
```

Or use `--data` flag:
```bash
pixelchart render --data '<JSON>' [OPTIONS]
```

Options:
```
-t, --type <TYPE>       Chart type (overrides JSON "type" field)
-d, --data <JSON>       Inline JSON (alternative to stdin)
    --title <TEXT>       Chart title
    --x-label <TEXT>     X-axis label
    --y-label <TEXT>     Y-axis label
-W, --width <PX>        Image width (default: 800)
-H, --height <PX>       Image height (default: 500)
    --theme <THEME>      dark|light (default: dark)
-o, --output <PATH>     Save to PNG file (omit for inline terminal display)
```

### 2. `plot` — CSV/TSV → Chart

Plot directly from CSV data — the most convenient way to chart tabular data:
```bash
cat data.csv | pixelchart plot -y revenue,expenses --title "Financial"
pixelchart plot --csv data.csv -x date -y price -t line --sort
```

Options:
```
-t, --type <TYPE>       Chart type (auto-detected if omitted)
    --csv <FILE>        CSV file (reads stdin if omitted)
-x, --x <COL>          Column name for X axis
-y, --y <COL,COL,...>   Column name(s) for Y axis (comma-separated for multi-series)
    --title <TEXT>       Chart title
    --x-label <TEXT>     X-axis label (defaults to column name)
    --y-label <TEXT>     Y-axis label (defaults to column name)
-W, --width <PX>        Image width (default: 800)
-H, --height <PX>       Image height (default: 500)
    --theme <THEME>      dark|light|pastel|ocean|forest (default: dark)
-o, --output <PATH>     Save to PNG file
    --delimiter <CHAR>   Field delimiter (default: comma, use 'tab' for TSV)
    --no-header          CSV has no header row
    --y-min <NUM>        Override Y-axis minimum
    --y-max <NUM>        Override Y-axis maximum
    --x-min <NUM>        Override X-axis minimum
    --x-max <NUM>        Override X-axis maximum
    --bins <N>           Number of histogram bins
    --sort               Sort data by X column before plotting
    --no-bg              Transparent background
    --skip-rows <N>      Skip N data rows after header
    --max-rows <N>       Maximum rows to read (default: 100000)
```

### 3. `example` — Built-in Demos

Render built-in example charts (no data needed):
```bash
pixelchart example                    # Gallery of all types
pixelchart example line               # Specific type
pixelchart example --output demo.png  # Save to file
```

### 4. `show` — Display PNG

Display an existing PNG inline in the terminal:
```bash
pixelchart show chart.png
```

### 5. `info` — Terminal Capabilities

Print terminal capabilities and chart type examples:
```bash
pixelchart info
```

## JSON Spec Format

Every spec has this shape:
```json
{
  "type": "<chart_type>",
  "data": { ... },
  "title": "Optional title",
  "x_label": "Optional X label",
  "y_label": "Optional Y label",
  "theme": "dark",
  "width": 800,
  "height": 500,
  "x_range": [0, 100],
  "y_range": [0, 50],
  "transparent_bg": false
}
```

### Chart Types & Data Shapes

#### Line Chart — trends, time series
```json
{"type":"line","data":{"y":[1,4,2,8,5]},"title":"Revenue"}
```

Multi-series:
```json
{"type":"line","data":{"series":[
  {"label":"Revenue","values":[1,2,3,4]},
  {"label":"Cost","values":[0.5,1,2,3]}
]},"title":"Comparison"}
```

With explicit X values:
```json
{"type":"line","data":{"x":[0,0.5,1.0,1.5],"y":[1,4,2,8]}}
```

Line options (add to `data`):
- `"smooth": true` — Catmull-Rom spline interpolation
- `"filled": true` — gradient fill under curve
- `"points": true` — show data point markers
- `"step": true` — stairstep rendering

#### Scatter Plot — correlations, distributions
```json
{"type":"scatter","data":{"x":[1,2,3,4,5],"y":[2,4,3,8,5]},"title":"Correlation"}
```

#### Bar Chart — categorical comparisons
```json
{"type":"bar","data":{"labels":["Q1","Q2","Q3","Q4"],"values":[100,200,150,300]},"title":"Sales"}
```

#### Histogram — frequency distributions
```json
{"type":"histogram","data":{"values":[1.2,3.4,2.1,5.6,4.3,2.8,3.1,4.7]},"title":"Distribution","bins":10}
```

#### Box Plot — statistical summaries
```json
{"type":"boxplot","data":{"groups":[
  {"label":"Control","values":[1,2,3,4,5,6,7]},
  {"label":"Treatment","values":[2,4,6,8,10,12]}
]}}
```
Type aliases: `"box"`, `"box_plot"`, `"boxplot"`

#### Heatmap — 2D intensity grids
```json
{"type":"heatmap","data":{"grid":[[1,2,3],[4,5,6],[7,8,9]]},"title":"Heatmap"}
```

#### Pie Chart — proportions
```json
{"type":"pie","data":{"labels":["Engineering","Marketing","Sales","Ops"],"values":[40,25,20,15]},"title":"Budget"}
```

## Themes

Available themes: `dark` (default), `light`, `pastel`, `ocean`, `forest`

## Important Notes

- Inline display requires a Kitty-compatible terminal (Kitty, WezTerm, Ghostty)
- Sixel-capable terminals (foot, mlterm, xterm) also work
- For non-Kitty terminals, always use `--output file.png` to save to a file
- The `plot` command auto-detects chart type from column shapes when `-t` is omitted
- All chart images are anti-aliased, pixel-perfect vector renders — not ASCII art
- PNG export works everywhere regardless of terminal support

## Common Workflows

### Quick data visualization
```bash
echo '{"type":"line","data":{"y":[10,25,18,30,22,35,28,42]},"title":"Weekly Growth"}' | pixelchart render
```

### CSV exploration
```bash
cat sales.csv | pixelchart plot -x month -y revenue,profit --title "Sales Trends" --sort
```

### Save for documentation
```bash
echo '{"type":"bar","data":{"labels":["v1","v2","v3"],"values":[200,800,1500]}}' \
  | pixelchart render --output perf_comparison.png --width 1200 --height 600
```

### Gallery of all chart types
```bash
pixelchart example --output gallery.png
```
