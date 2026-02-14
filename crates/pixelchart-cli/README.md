# pixelchart-cli

**CLI tool for generating pixel-perfect charts in the terminal** — powered by [`pixelchart`](../pixelchart) and [`ratatui-pixelcanvas`](https://github.com/Slush97/ratatui-pixelcanvas).

## Installation

```bash
cargo install --path crates/pixelchart-cli
```

## Commands

### `render` — Chart from JSON

```bash
# From stdin
echo '{"type":"line","data":{"y":[1,4,2,8,5]},"title":"Revenue"}' | pixelchart render

# From a JSON string
pixelchart render --data '{"type":"scatter","data":{"x":[1,2,3],"y":[4,5,6]}}'

# Save to file
pixelchart render --data '...' -o chart.png
```

### `plot` — Chart from CSV

```bash
# Auto-detect chart type from CSV
pixelchart plot --csv data.csv --x date --y revenue,costs

# Histogram from a single column  
cat measurements.csv | pixelchart plot -t histogram --y value --bins 20

# Custom axis bounds and theme
pixelchart plot --csv sales.csv --x month --y total \
  --y-min 0 --y-max 1000 --theme ocean --title "Monthly Sales"

# TSV with custom delimiter
pixelchart plot --csv data.tsv --delimiter tab --x col1 --y col2
```

### `example` — Built-in demos

```bash
# Gallery of all chart types
pixelchart example

# Specific chart type
pixelchart example scatter
pixelchart example candlestick

# Save to file
pixelchart example bar -o bar_chart.png
```

### `show` — Display a PNG

```bash
pixelchart show chart.png
```

### `info` — Terminal capabilities

```bash
pixelchart info
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
