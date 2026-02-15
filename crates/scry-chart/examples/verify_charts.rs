//! Quick verify for histogram axis alignment fix.
use scry_chart::prelude::*;

fn main() {
    let data: Vec<f64> = (0..100)
        .map(|i| (i as f64 * 0.1).sin() * 50.0 + 50.0)
        .collect();
    let chart = Chart::histogram(&data)
        .title("Distribution")
        .x_label("Value")
        .y_label("Frequency")
        .bins(15)
        .build();
    save_png(&chart, 800, 500, "/tmp/scry_chart_verify/histogram.png").unwrap();
    println!("✓ histogram.png");
}
