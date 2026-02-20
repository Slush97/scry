// Legend auto-positioning demo — generates sample PNGs.
//
// Run with:
//   cargo run --example legend_demo -p scry-chart

use scry_chart::prelude::*;

fn main() {
    let out = std::path::Path::new("crates/scry-chart/legend_demo");
    std::fs::create_dir_all(out).unwrap();

    // 1. Rising quadratic — data fills top-right, legend dodges.
    {
        let ys_a: Vec<f64> = (0..20).map(|i| (i as f64).powi(2)).collect();
        let ys_b: Vec<f64> = (0..20).map(|i| (i as f64).powi(2) * 0.8 + 10.0).collect();
        let chart = LineChart::new(vec![
            Series::new("Revenue", ys_a),
            Series::new("Costs", ys_b),
        ])
        .title("Quadratic Rise — Legend Dodges Top-Right Data")
        .theme(Theme::dark())
        .build();
        save_png(&chart, 700, 450, out.join("quadratic_rise.png")).unwrap();
        println!("✓ quadratic_rise.png");
    }

    // 2. Multi-series bar chart with tall bars.
    {
        let chart = BarChart::new(
            vec!["Q1".into(), "Q2".into(), "Q3".into(), "Q4".into()],
            vec![
                Series::new("Product A", vec![80.0, 120.0, 95.0, 140.0]),
                Series::new("Product B", vec![60.0, 90.0, 110.0, 130.0]),
                Series::new("Product C", vec![40.0, 70.0, 100.0, 150.0]),
            ],
        )
        .title("Quarterly Sales — Legend Avoids Bars")
        .theme(Theme::dark())
        .build();
        save_png(&chart, 700, 450, out.join("bar_chart_dodge.png")).unwrap();
        println!("✓ bar_chart_dodge.png");
    }

    // 3. Scatter with data in all corners — legend should move outside.
    {
        let xs_a = Series::new("Sensor A", vec![
            0.0, 1.0, 2.0, 98.0, 99.0, 100.0,
            0.0, 1.0, 2.0, 98.0, 99.0, 100.0,
        ]);
        let ys_a = Series::new("Sensor A", vec![
            0.0, 2.0, 1.0, 98.0, 100.0, 99.0,
            50.0, 48.0, 52.0, 50.0, 48.0, 52.0,
        ]);
        let xs_b = Series::new("Sensor B", vec![
            0.0, 1.0, 2.0, 98.0, 99.0, 100.0,
            0.0, 1.0, 2.0, 98.0, 99.0, 100.0,
        ]);
        let ys_b = Series::new("Sensor B", vec![
            100.0, 98.0, 99.0, 0.0, 2.0, 1.0,
            50.0, 52.0, 48.0, 50.0, 52.0, 48.0,
        ]);
        let chart = ScatterChart::new(xs_a, ys_a)
            .add_series(xs_b, ys_b)
            .title("All-Corners Data — Legend Promoted Outside")
            .theme(Theme::dark())
            .build();
        save_png(&chart, 700, 450, out.join("scatter_all_corners.png")).unwrap();
        println!("✓ scatter_all_corners.png");
    }

    // 4. Clean data — all points below midline, legend stays in top-right.
    {
        let chart = LineChart::new(vec![
            Series::new("Sensor A", vec![3.0, 5.0, 2.0, 4.0, 6.0, 3.0]),
            Series::new("Sensor B", vec![1.0, 3.0, 4.0, 2.0, 5.0, 4.0]),
        ])
        .title("Low Values — Legend Stays Top-Right")
        .theme(Theme::dark())
        .build();
        save_png(&chart, 700, 450, out.join("flat_data_default.png")).unwrap();
        println!("✓ flat_data_default.png");
    }

    // 5. Histogram multi-series — legend avoids bins.
    {
        let primary = Series::new("Group A", vec![
            1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
            6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
        ]);
        let extra = Series::new("Group B", vec![
            5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
            10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
        ]);
        let chart = Histogram::new(primary)
            .add_series(extra)
            .title("Histogram — Legend Avoids Bins")
            .theme(Theme::dark())
            .build();
        save_png(&chart, 700, 450, out.join("histogram_dodge.png")).unwrap();
        println!("✓ histogram_dodge.png");
    }

    println!("\nAll demo charts saved to {}", out.display());
}
