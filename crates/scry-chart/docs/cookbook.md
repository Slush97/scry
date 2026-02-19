# scry-chart Cookbook

Practical recipes for common charting tasks.  
All examples assume `use scry_chart::prelude::*;`.

---

## Quick Line Chart

```rust
let chart = Charts::line(&[1.0, 4.0, 2.0, 8.0, 5.0])
    .title("Quick Line")
    .build();
```

## Area Chart (filled + smooth)

```rust
let chart = Charts::area(&[3.0, 7.0, 4.0, 9.0, 6.0])
    .title("Revenue Trend")
    .build();
```

## Multi-Series Line

```rust
let chart = LineChart::new(vec![
    Series::new("Rev", vec![10.0, 30.0, 20.0, 50.0]),
    Series::new("Cost", vec![15.0, 25.0, 35.0, 30.0]),
])
.title("P&L")
.x_label("Quarter")
.build();
```

## Scatter with Asymmetric Error Bars

```rust
let y = Series::from_values(vec![10.0, 20.0, 15.0])
    .with_error_asymmetric(
        vec![2.0, 3.0, 1.0],  // lower offsets
        vec![5.0, 4.0, 6.0],  // upper offsets
    );
let chart = ScatterChart::new(
    Series::from_values(vec![1.0, 2.0, 3.0]),
    y,
).title("Measurements").build();
```

## Contour Plot

```rust
// 2D scalar field (row-major)
let grid = vec![
    vec![0.0, 1.0, 2.0, 3.0],
    vec![1.0, 2.0, 3.0, 4.0],
    vec![2.0, 3.0, 4.0, 5.0],
    vec![3.0, 4.0, 5.0, 6.0],
];
let chart = Charts::contour(grid)
    .levels(8)
    .filled()
    .title("Temperature Field")
    .build();
```

## LOD Decimation for Large Datasets

```rust
use scry_chart::decimate::{lttb, min_max_decimate};

// 100K data points → 500 representative points
let data: Vec<(f64, f64)> = expensive_query();
let thinned = lttb(&data, 500);

// Or keep peaks/valleys with min-max:
let thinned = min_max_decimate(&data, 250); // up to 500 points
```

## Text Utilities

```rust
use scry_chart::text_utils::{wrap_text, ellipsize};

// Wrap long titles across multiple lines:
let lines = wrap_text("A Very Long Chart Title", 120.0, 8.0);

// Truncate axis labels with ellipsis:
let short = ellipsize("United States of America", 80.0, 7.0);
// → "United St…"
```

## Validated Chart Construction

```rust
let result = Charts::contour(vec![vec![1.0, 2.0], vec![3.0]])
    .try_build();
assert!(result.is_err()); // JaggedGrid

let result = Charts::gauge(f64::NAN).try_build();
assert!(result.is_err()); // AllNonFinite

let result = Charts::funnel(
    vec!["A".into(), "B".into()],
    &[100.0],               // length mismatch
).try_build();
assert!(result.is_err()); // InvalidConfig
```

## Custom Formatters

```rust
use scry_chart::prelude::*;

// Format ticks as currency:
let chart = Charts::bar(labels, &values)
    .y_formatter(FnFormatter::new(|v| format!("${:.0}", v)))
    .build();

// Locale-aware number grouping:
let chart = Charts::line(&data)
    .locale(LocaleConfig { grouping: Some(','), ..Default::default() })
    .build();
```

## SVG Export (with accessibility)

```rust
use scry_chart::svg_export::render_to_svg;

let chart = Charts::line(&[1.0, 2.0, 3.0])
    .title("Accessible Chart")
    .build();
let svg_string = render_to_svg(&chart, 800, 500);
// Output includes role="img", <title>, and <desc> elements.
```

## Themes

```rust
// Dark (default), Light, Colorblind-safe:
let chart = Charts::line(&data)
    .theme(Theme::dark())
    .build();

let chart = Charts::line(&data)
    .theme(Theme::colorblind())
    .build();
```
