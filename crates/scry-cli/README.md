# scry-cli

**CLI tool for generating pixel-perfect charts in the terminal** — powered by [`scry-chart`](../scry-chart) and [`scry-engine`](https://github.com/Slush97/scry).

## Installation

```bash
cargo install --path crates/scry-cli
```

## Commands

### `render` — Chart from JSON

```bash
# From stdin
echo '{"type":"line","data":{"y":[1,4,2,8,5]},"title":"Revenue"}' | scry-chart render

# From a JSON string
scry-chart render --data '{"type":"scatter","data":{"x":[1,2,3],"y":[4,5,6]}}'

# Save to file
scry-chart render --data '...' -o chart.png
```

### `plot` — Chart from CSV

```bash
# Auto-detect chart type from CSV
scry-chart plot --csv data.csv --x date --y revenue,costs

# Histogram from a single column  
cat measurements.csv | scry-chart plot -t histogram --y value --bins 20

# Custom axis bounds and theme
scry-chart plot --csv sales.csv --x month --y total \
  --y-min 0 --y-max 1000 --theme ocean --title "Monthly Sales"

# TSV with custom delimiter
scry-chart plot --csv data.tsv --delimiter tab --x col1 --y col2
```

### `example` — Built-in demos

```bash
# Gallery of all chart types
scry-chart example

# Specific chart type
scry-chart example scatter
scry-chart example candlestick

# Save to file
scry-chart example bar -o bar_chart.png
```

### `show` — Display a PNG

```bash
scry-chart show chart.png
```

### `info` — Terminal capabilities

```bash
scry-chart info
```

## Supported Chart Types

`line`, `scatter`, `bar`, `histogram`, `boxplot`, `heatmap`, `pie`, `radar`, `candlestick`

## Options

| Flag | Description |
|------|-------------|
| `-t, --type` | Chart type (auto-detected if omitted) |
| `-W, --width` | Image width in pixels (default: 800) |
| `-H, --height` | Image height in pixels (default: 500) |
| `--theme` | Theme name: `dark`, `light`, `ocean`, `forest`, `pastel` |
| `-o, --output` | Output file path (PNG/SVG; inline display if omitted) |
| `--no-bg` | Transparent background |
| `--title` | Chart title |
| `--x-label` | X-axis label |
| `--y-label` | Y-axis label |

## Requirements

- A terminal with [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) support for inline display
- Or use `-o file.png` / `-o file.svg` for file output

## License

MIT OR Apache-2.0
