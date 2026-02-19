# scry-chart

**Anti-aliased, pixel-perfect charts for the terminal** — powered by [`scry-engine`](https://github.com/Slush97/scry).

Unlike text-based charting widgets, `scry-chart` renders true vector graphics using the Kitty graphics protocol, producing smooth lines, gradient fills, and anti-aliased curves directly in your terminal emulator.

## Quick Start

```rust
use scry-chart::prelude::*;

// Three lines of code to a production-quality chart:
let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0, 9.0])
    .title("Revenue")
    .build();

let widget = ChartWidget::new(&chart);
// Render `widget` in your ratatui layout
```

## Chart Types

| Type | Constructor | Description |
|------|------------|-------------|
| **Line** | `Charts::line(y)` | Continuous lines with optional fill, smooth curves, step lines |
| **Line XY** | `Charts::line_xy(x, y)` | Lines with explicit (non-uniform) x values |
| **Scatter** | `Charts::scatter(x, y)` | Individual data points with configurable markers |
| **Bar** | `Charts::bar(labels, values)` | Vertical bars with gradient fills |
| **Histogram** | `Charts::histogram(values, bins)` | Distribution visualization |
| **Box Plot** | `Charts::boxplot(groups)` | Statistical distribution summary |
| **Heatmap** | `Charts::heatmap(matrix)` | 2D intensity grid with color mapping |
| **Pie** | `Charts::pie(labels, values)` | Proportional slices with labels |
| **Radar** | `Charts::radar(axes)` | Spider/radar chart for multi-axis comparison |
| **Candlestick** | `Charts::candlestick(ohlc)` | OHLC financial chart with bullish/bearish coloring |

## Line Chart Options

```rust
Charts::line(&data)
    .smooth()           // Catmull-Rom spline interpolation
    .step()             // Stairstep line rendering
    .filled()           // Gradient fill under curve
    .with_points()      // Show data point markers
    .line_width(3.0)    // Custom line width
    .title("My Chart")
    .x_label("Time")
    .y_label("Value")
    .build()
```

## Scatter Plot Options

```rust
Charts::scatter(x_data, y_data)
    .size(6.0)                    // Marker radius
    .marker(Marker::Diamond)     // Circle, Square, Diamond, Cross, Triangle
    .connected()                  // Connect points with lines
    .build()
```

## Interactivity

`scry-chart` supports mouse-driven interactivity out of the box:

```rust
let mut state = ChartState::default();

// In your event handler:
state.handle_mouse(event, widget_area);

// Features:
// - Crosshair cursor following mouse position
// - Tooltip showing data coordinates
// - Scroll to zoom in/out
// - Click-drag to pan
```

## Styling & Themes

```rust
use scry-chart::theme::Theme;

let chart = Charts::line(&data)
    .theme(Theme::dark())          // Built-in themes
    .h_line(50.0)                  // Horizontal reference line
    .v_line(3.0)                   // Vertical reference line
    .annotate(2.0, 8.0, "Peak")   // Data annotations
    .build();
```

## Multi-Series Charts

```rust
let chart = Charts::line(&revenue)
    .add_series(Series::new("Expenses", expenses))
    .add_series(Series::new("Profit", profit))
    .title("Financial Overview")
    .build();
```

## Tick Formatting & Locales

```rust
use scry-chart::formatter::*;

Charts::line(&data)
    .y_formatter(CurrencyFormatter::usd())    // $1,234.56
    .x_formatter(SiFormatter::default())      // 1.5K, 2.3M
    .european_locale()                         // 1.234,56
    .build()
```

## Export

```rust
use scry-chart::export::save_png;
use scry-chart::svg_export::save_svg;

save_png(&chart, 800, 500, "chart.png").unwrap();
save_svg(&chart, 800, 500, "chart.svg").unwrap();
```

## How It Works

`scry-chart` renders charts as true pixel graphics via `scry-engine`:

1. **Layout engine** computes axes, margins, and plot area proportionally
2. **Vector renderer** emits `tiny-skia` drawing commands (anti-aliased lines, gradient fills, curves)
3. **Transport layer** sends pixels via Kitty graphics protocol (also supports Sixel, halfblock fallback)
4. **Widget integration** composites pixel output with ratatui text labels

This means charts look as good as a desktop plotting library — but run entirely in your terminal.

## Requirements

- A terminal that supports the [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) (Kitty, Ghostty, WezTerm) for best quality
- Sixel-capable terminals (foot, mlterm, xterm) also work
- Fallback to Unicode half-block characters for basic terminal support

## License

MIT OR Apache-2.0
