---
description: "Generate a pixel-perfect chart in the terminal using pixelchart"
allowed-tools: ["Bash"]
argument-hint: "<chart description, JSON spec, or CSV file path>"
---

Generate a pixel-perfect chart using the `pixelchart` CLI tool. The tool renders
anti-aliased charts via the Kitty graphics protocol (or saves to PNG).

## Usage

The user will describe the data they want to visualize. You should:

1. Determine the most appropriate chart type
2. Choose the best approach: JSON spec via `render`, or CSV via `plot`
3. Execute the command

Ensure the binary is built first: `cargo build -p pixelchart-cli`

## Chart Types & JSON Formats

**Line chart** — time series, trends:
```bash
echo '{"type":"line","data":{"y":[1,4,2,8,5]},"title":"My Line Chart"}' | pixelchart render
```

Multi-series:
```bash
echo '{"type":"line","data":{"series":[{"label":"Revenue","values":[1,2,3]},{"label":"Cost","values":[0.5,1,2]}]},"title":"Comparison"}' | pixelchart render
```

Options: `"filled":true`, `"smooth":true`, `"step":true`, `"points":true`

With explicit X values:
```bash
echo '{"type":"line","data":{"x":[0,0.5,1.0,1.5],"y":[1,4,2,8]}}' | pixelchart render
```

**Scatter plot** — correlations, distributions:
```bash
echo '{"type":"scatter","data":{"x":[1,2,3],"y":[4,5,6]},"title":"Scatter"}' | pixelchart render
```

**Bar chart** — categorical comparisons:
```bash
echo '{"type":"bar","data":{"labels":["Q1","Q2","Q3"],"values":[100,200,150]},"title":"Sales"}' | pixelchart render
```

**Histogram** — frequency distribution:
```bash
echo '{"type":"histogram","data":{"values":[1.2,3.4,2.1,5.6,4.3]},"title":"Distribution"}' | pixelchart render
```

**Box plot** — statistical summary:
```bash
echo '{"type":"boxplot","data":{"groups":[{"label":"A","values":[1,2,3,4,5]},{"label":"B","values":[2,4,6,8]}]}}' | pixelchart render
```

**Heatmap** — 2D intensity grid:
```bash
echo '{"type":"heatmap","data":{"grid":[[1,2],[3,4]]},"title":"Heatmap"}' | pixelchart render
```

**Pie chart** — proportions:
```bash
echo '{"type":"pie","data":{"labels":["A","B","C"],"values":[40,35,25]},"title":"Share"}' | pixelchart render
```

## CSV Plotting

For tabular data, use `plot` instead of `render`:
```bash
cat data.csv | pixelchart plot -y revenue,cost --title "Trends" --sort
pixelchart plot --csv data.csv -x date -y price -t line --sort
```

CSV-specific options:
- `-x COL` — X-axis column name
- `-y COL1,COL2` — Y-axis column(s), comma-separated for multi-series
- `--delimiter 'tab'` — TSV support
- `--no-header` — CSV without header row
- `--sort` — sort by X column before plotting
- `--y-min/--y-max/--x-min/--x-max` — axis range overrides
- `--bins N` — histogram bin count
- `--no-bg` — transparent background

## Common Options (all commands)

- `--output chart.png` — save to file instead of inline display
- `--title "Title"` — chart title
- `--theme dark|light|pastel|ocean|forest` — color theme
- `--width 1200 --height 800` — custom dimensions
- `--x-label "Time" --y-label "Value"` — axis labels

## Advanced JSON Options

These can be added to the JSON spec:
- `"x_range": [0, 100]` — explicit X domain
- `"y_range": [0, 50]` — explicit Y domain
- `"transparent_bg": true` — no background fill
- `"bins": 20` — histogram bin count

## Built-in Examples

Gallery of all chart types (no data required):
```bash
pixelchart example
pixelchart example line --output line_demo.png
```

$ARGUMENTS
