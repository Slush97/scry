---
description: "Generate a pixel-perfect chart in the terminal using pixelchart"
allowed-tools: ["Bash"]
argument-hint: "<chart description or JSON spec>"
---

Generate a pixel-perfect chart using the `pixelchart` CLI tool. The tool renders
anti-aliased charts via the Kitty graphics protocol (or saves to PNG).

## Usage

The user will describe the data they want to visualize. You should:

1. Determine the most appropriate chart type
2. Format the data as a JSON spec
3. Pipe it to `pixelchart render`

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

## Common Options

- `--output chart.png` — save to file instead of inline display
- `--title "Title"` — override title
- `--theme light` — use light theme (default: dark)
- `--width 1200 --height 800` — custom dimensions
- `--x-label "Time" --y-label "Value"` — axis labels

$ARGUMENTS
