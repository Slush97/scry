// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live training visualization demo.
//!
//! Trains an MLP classifier on a synthetic dataset with a `LivePlotCallback`
//! that renders a live loss curve in the terminal each epoch.
//!
//! Run with:
//! ```bash
//! cargo run -p scry-learn --features live-plot --example live_training --release
//! ```

use scry_learn::dataset::Dataset;
use scry_learn::neural::live_plot::LivePlotCallback;
use scry_learn::neural::MLPClassifier;
use scry_learn::prelude::*;

fn main() {
    // Generate a synthetic 3-class classification dataset.
    let mut rng = fastrand::Rng::with_seed(42);
    let n_per_class = 50;
    let mut f1 = Vec::new();
    let mut f2 = Vec::new();
    let mut f3 = Vec::new();
    let mut f4 = Vec::new();
    let mut target = Vec::new();

    for class in 0..3 {
        let cx = class as f64 * 3.0;
        let cy = class as f64 * 2.0;
        for _ in 0..n_per_class {
            f1.push(cx + rng.f64() * 2.0 - 1.0);
            f2.push(cy + rng.f64() * 2.0 - 1.0);
            f3.push(cx * 0.5 + rng.f64());
            f4.push(cy * 0.3 + rng.f64());
            target.push(class as f64);
        }
    }

    let data = Dataset::new(
        vec![f1, f2, f3, f4],
        target,
        vec!["f1".into(), "f2".into(), "f3".into(), "f4".into()],
        "class",
    );
    let (train, test) = train_test_split(&data, 0.2, 42);

    println!(
        "Synthetic dataset: {} train, {} test samples, 3 classes",
        train.n_samples(),
        test.n_samples()
    );
    println!("Training MLP with live loss plot...\n");

    let mut clf = MLPClassifier::new()
        .hidden_layers(&[64, 32])
        .learning_rate(0.005)
        .max_iter(100)
        .batch_size(32)
        .early_stopping(true)
        .n_iter_no_change(15)
        .seed(42)
        .callback(Box::new(LivePlotCallback::new()));

    clf.fit(&train).expect("training failed");

    let preds = clf
        .predict(&test.feature_matrix())
        .expect("prediction failed");
    let acc = accuracy(&test.target, &preds);
    println!("\nFinal test accuracy: {acc:.1}%");
    println!("Epochs trained: {}", clf.loss_curve().len());
}
