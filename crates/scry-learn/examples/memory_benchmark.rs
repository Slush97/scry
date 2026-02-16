//! Memory footprint benchmark: measures RSS delta and serialized size per model.
//!
//! Run with:
//!   cargo run -p scry-learn --example memory_benchmark --features serde --release

use scry_learn::prelude::*;

fn read_rss_kb() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(kb_str) = parts.get(1) {
                return kb_str.parse().unwrap_or(0);
            }
        }
    }
    0
}

fn synthetic_dataset(n_samples: usize, n_features: usize, n_classes: usize) -> Dataset {
    let mut features = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let col: Vec<f64> = (0..n_samples)
            .map(|i| ((i * (j + 3) + 7) % 997) as f64 / 997.0 * 10.0)
            .collect();
        features.push(col);
    }
    let target: Vec<f64> = (0..n_samples).map(|i| (i % n_classes) as f64).collect();
    let names: Vec<String> = (0..n_features).map(|j| format!("f{j}")).collect();
    Dataset::new(features, target, names, "target")
}

fn regression_dataset(n_samples: usize, n_features: usize) -> Dataset {
    let mut features = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let col: Vec<f64> = (0..n_samples)
            .map(|i| ((i * (j + 5) + 13) % 1009) as f64 / 100.0)
            .collect();
        features.push(col);
    }
    let target: Vec<f64> = (0..n_samples)
        .map(|i| (i as f64 * 0.3) + ((i * 7 + 11) % 53) as f64 * 0.1)
        .collect();
    let names: Vec<String> = (0..n_features).map(|j| format!("f{j}")).collect();
    Dataset::new(features, target, names, "y")
}

struct BenchResult {
    model_name: String,
    n_samples: usize,
    rss_delta_kb: isize,
    serialized_bytes: usize,
    fit_ms: u128,
}

fn bench_model<F>(name: &str, n_samples: usize, fit_fn: F) -> BenchResult
where
    F: FnOnce() -> Vec<u8>,
{
    let rss_before = read_rss_kb();
    let start = std::time::Instant::now();
    let serialized = fit_fn();
    let fit_ms = start.elapsed().as_millis();
    let rss_after = read_rss_kb();

    BenchResult {
        model_name: name.to_string(),
        n_samples,
        rss_delta_kb: rss_after as isize - rss_before as isize,
        serialized_bytes: serialized.len(),
        fit_ms,
    }
}

fn main() {
    println!("Memory Footprint Benchmark");
    println!("==========================\n");

    let sizes = [1_000, 10_000, 100_000];
    let n_features = 10;
    let n_classes = 3;

    let mut results: Vec<BenchResult> = Vec::new();

    for &n in &sizes {
        println!("--- Dataset: {n} samples x {n_features} features ---\n");

        // Decision Tree Classifier
        {
            let data = synthetic_dataset(n, n_features, n_classes);
            let r = bench_model("DecisionTreeClassifier", n, || {
                let mut m = DecisionTreeClassifier::new().max_depth(10);
                m.fit(&data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }

        // Random Forest Classifier
        {
            let data = synthetic_dataset(n, n_features, n_classes);
            let r = bench_model("RandomForestClassifier", n, || {
                let mut m = RandomForestClassifier::new()
                    .n_estimators(10)
                    .max_depth(10)
                    .seed(42);
                m.fit(&data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }

        // Gradient Boosting Classifier
        {
            let data = synthetic_dataset(n, n_features, n_classes);
            let r = bench_model("GradientBoostingClassifier", n, || {
                let mut m = GradientBoostingClassifier::new()
                    .n_estimators(10)
                    .max_depth(5)
                    .seed(42);
                m.fit(&data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }

        // Linear Regression
        {
            let data = regression_dataset(n, n_features);
            let r = bench_model("LinearRegression", n, || {
                let mut m = LinearRegression::new();
                m.fit(&data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }

        // Logistic Regression
        {
            let data = synthetic_dataset(n, n_features, n_classes);
            let r = bench_model("LogisticRegression", n, || {
                let mut m = LogisticRegression::new();
                m.fit(&data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }

        // KNN Classifier
        {
            let data = synthetic_dataset(n, n_features, n_classes);
            let r = bench_model("KnnClassifier", n, || {
                let mut m = KnnClassifier::new().k(5);
                m.fit(&data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }

        // KMeans
        {
            let data = synthetic_dataset(n, n_features, n_classes);
            let r = bench_model("KMeans", n, || {
                let mut m = KMeans::new(n_classes).seed(42);
                m.fit(&data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }

        // Gaussian NB
        {
            let data = synthetic_dataset(n, n_features, n_classes);
            let r = bench_model("GaussianNb", n, || {
                let mut m = GaussianNb::new();
                m.fit(&data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }

        // PCA
        {
            let mut data = synthetic_dataset(n, n_features, n_classes);
            let r = bench_model("PCA", n, || {
                let mut m = Pca::new();
                m.fit_transform(&mut data).unwrap();
                serde_json::to_vec(&m).unwrap()
            });
            results.push(r);
        }
    }

    // Print table
    println!("\n{:<30} {:>10} {:>12} {:>14} {:>10}",
        "Model", "Samples", "RSS delta", "Serialized", "Fit (ms)");
    println!("{:-<78}", "");

    for r in &results {
        let serial_str = if r.serialized_bytes < 1024 {
            format!("{} B", r.serialized_bytes)
        } else if r.serialized_bytes < 1024 * 1024 {
            format!("{:.1} KB", r.serialized_bytes as f64 / 1024.0)
        } else {
            format!("{:.1} MB", r.serialized_bytes as f64 / (1024.0 * 1024.0))
        };

        let rss_str = if r.rss_delta_kb.abs() < 1024 {
            format!("{} KB", r.rss_delta_kb)
        } else {
            format!("{:.1} MB", r.rss_delta_kb as f64 / 1024.0)
        };

        println!("{:<30} {:>10} {:>12} {:>14} {:>10}",
            r.model_name, r.n_samples, rss_str, serial_str, r.fit_ms);
    }
}
