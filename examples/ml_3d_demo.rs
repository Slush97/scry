//! 3D ML visualization demo.
//!
//! Shows three scenes using scry-chart's `Chart3D` with scry-learn models:
//!
//! 1. **Iris PCA** — 4D Iris data projected to 3 principal components, colored by species
//! 2. **KMeans Clustering** — synthetic 3D blobs with cluster assignments
//! 3. **Random Forest Feature Importance** — 3D scatter of importances
//!
//! Usage:
//!   cargo run --example ml_3d_demo
//!   cargo run --example ml_3d_demo -- --save   # save PNGs instead of terminal display

use scry_chart::chart3d::Chart3D;
use scry_learn::cluster::KMeans;
use scry_learn::dataset::Dataset;
use scry_learn::metrics::accuracy;
use scry_learn::preprocess::{Pca, StandardScaler, Transformer};
use scry_learn::split::train_test_split;
use scry_learn::tree::RandomForestClassifier;

fn main() {
    let save = std::env::args().any(|a| a == "--save");

    println!("=== scry 3D ML Demo ===\n");

    scene_iris_pca(save);
    scene_kmeans_clusters(save);
    scene_feature_importance_3d(save);

    println!("\nDone!");
}

// ─── Scene 1: Iris PCA 3D ───────────────────────────────────────────────

fn scene_iris_pca(save: bool) {
    println!("[1/3] Iris dataset → PCA 3D projection");

    let ds = iris_dataset();

    // Standardize then reduce to 3 components
    let mut pca_ds = ds.clone();
    let mut scaler = StandardScaler::new();
    scaler.fit_transform(&mut pca_ds).expect("scaler fit");
    let mut pca = Pca::with_n_components(3);
    pca.fit_transform(&mut pca_ds).expect("pca fit");

    let var = pca.explained_variance_ratio();
    println!(
        "  Explained variance: PC1={:.1}% PC2={:.1}% PC3={:.1}% (total {:.1}%)",
        var[0] * 100.0,
        var[1] * 100.0,
        var[2] * 100.0,
        var.iter().sum::<f64>() * 100.0,
    );

    // Extract the 3 principal components (column-major)
    let pc1 = &pca_ds.features[0];
    let pc2 = &pca_ds.features[1];
    let pc3 = &pca_ds.features[2];
    let classes: Vec<usize> = pca_ds.target.iter().map(|&v| v as usize).collect();

    let chart = Chart3D::scatter(pc1, pc2, pc3)
        .title("Iris PCA — 3 Components")
        .x_label("PC1")
        .y_label("PC2")
        .z_label("PC3")
        .color_by_class(&classes)
        .point_size(5.0);

    show_or_save(chart, "iris_pca_3d.png", save);
}

// ─── Scene 2: KMeans on 3D blobs ────────────────────────────────────────

fn scene_kmeans_clusters(save: bool) {
    println!("[2/3] Synthetic 3D blobs → KMeans clustering");

    let mut rng = fastrand::Rng::with_seed(42);
    let n_per = 50;

    let centers: [(f64, f64, f64); 3] = [(2.0, 2.0, 2.0), (-2.0, -1.0, 3.0), (1.0, -3.0, -1.0)];
    let mut xs = Vec::with_capacity(n_per * 3);
    let mut ys = Vec::with_capacity(n_per * 3);
    let mut zs = Vec::with_capacity(n_per * 3);
    let mut true_labels = Vec::with_capacity(n_per * 3);

    for (ci, &(cx, cy, cz)) in centers.iter().enumerate() {
        for _ in 0..n_per {
            xs.push(cx + (rng.f64() - 0.5) * 2.0);
            ys.push(cy + (rng.f64() - 0.5) * 2.0);
            zs.push(cz + (rng.f64() - 0.5) * 2.0);
            true_labels.push(ci);
        }
    }

    // Build dataset (column-major)
    let features = vec![xs.clone(), ys.clone(), zs.clone()];
    let target: Vec<f64> = true_labels.iter().map(|&v| v as f64).collect();
    let ds = Dataset::new(
        features,
        target,
        vec!["x".into(), "y".into(), "z".into()],
        "cluster",
    );

    let mut km = KMeans::new(3).max_iter(100).seed(42);
    km.fit(&ds).expect("kmeans fit");

    // predict expects column-major &[Vec<f64>]
    let cols = vec![xs.clone(), ys.clone(), zs.clone()];
    let preds = km.predict(&cols).expect("kmeans predict");

    let chart = Chart3D::scatter(&xs, &ys, &zs)
        .title("KMeans — 3 Clusters")
        .x_label("X")
        .y_label("Y")
        .z_label("Z")
        .color_by_class(&preds)
        .point_size(5.0);

    show_or_save(chart, "kmeans_3d.png", save);
}

// ─── Scene 3: Feature importance as 3D scatter ──────────────────────────

fn scene_feature_importance_3d(save: bool) {
    println!("[3/3] Random Forest on Iris → feature importance in 3D");

    let ds = iris_dataset();
    let (train, test) = train_test_split(&ds, 0.3, 42);

    let mut rf = RandomForestClassifier::new().n_estimators(100).max_depth(5).seed(42);
    rf.fit(&train).expect("rf fit");

    // predict expects row-major &[Vec<f64>] — each inner vec is one sample
    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| test.features.iter().map(|col| col[i]).collect())
        .collect();
    let preds = rf.predict(&test_rows).expect("rf predict");
    let acc = accuracy(&test.target, &preds);
    println!("  RF accuracy: {:.1}%", acc * 100.0);

    let importances = rf.feature_importances().expect("importances");
    let names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"];

    println!("  Importances:");
    for (name, imp) in names.iter().zip(importances.iter()) {
        println!("    {name}: {imp:.3}");
    }

    // Visualize: place each feature as a point where x=index, y=importance,
    // z=importance×2 (height emphasis), sized by importance
    let n = importances.len();
    let fx: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let fy: Vec<f64> = importances.clone();
    let fz: Vec<f64> = importances.iter().map(|v| v * 2.0).collect();
    let sizes: Vec<f32> = importances.iter().map(|v| 4.0 + (*v as f32) * 20.0).collect();

    let chart = Chart3D::scatter(&fx, &fy, &fz)
        .title("Feature Importance — Random Forest (Iris)")
        .x_label("Feature")
        .y_label("Importance")
        .z_label("Emphasis")
        .sizes(sizes)
        .color_by_class(&(0..n).collect::<Vec<_>>());

    show_or_save(chart, "rf_importance_3d.png", save);
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn show_or_save(chart: Chart3D, filename: &str, save: bool) {
    if save {
        chart.save_png(800, 600, filename).expect("save png");
        println!("  → Saved {filename}");
    } else {
        chart.show().expect("show chart");
    }
}

fn iris_dataset() -> Dataset {
    let sepal_length = vec![
        5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4,
        5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5,
        4.9, 5.0, 5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0, 7.0,
        6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8,
        6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4,
        6.0, 6.7, 6.3, 5.6, 5.5, 5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8,
        7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7,
        6.0, 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1, 7.7,
        6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9,
    ];
    let sepal_width = vec![
        3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9,
        3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2,
        3.1, 3.2, 3.5, 3.6, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3, 3.2,
        3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7,
        2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 3.2, 3.0, 2.8, 3.0, 2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3.0,
        3.4, 3.1, 2.3, 3.0, 2.5, 2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8, 3.3, 2.7,
        3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6,
        2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0,
        3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.5, 3.0, 3.4, 3.0,
    ];
    let petal_length = vec![
        1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3,
        1.5, 1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4,
        1.5, 1.2, 1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7,
        4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1,
        4.5, 3.9, 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5,
        4.5, 4.7, 4.4, 4.1, 4.0, 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1,
        5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9,
        5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1,
        5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.9, 5.7, 5.2, 5.0, 5.2, 5.4, 5.1, 5.1,
    ];
    let petal_width = vec![
        0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4,
        0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2,
        0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4,
        1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0,
        1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5,
        1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9,
        2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 1.8, 1.8, 2.1, 2.4,
        2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3,
        2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8,
    ];
    let target: Vec<f64> = (0..150)
        .map(|i| if i < 50 { 0.0 } else if i < 100 { 1.0 } else { 2.0 })
        .collect();

    Dataset::new(
        vec![sepal_length, sepal_width, petal_length, petal_width],
        target,
        vec![
            "sepal_length".into(),
            "sepal_width".into(),
            "petal_length".into(),
            "petal_width".into(),
        ],
        "species",
    )
}
