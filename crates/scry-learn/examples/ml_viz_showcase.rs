//! ML Visualization Showcase
//!
//! Generates PNG outputs for all 15 viz functions in scry-learn.
//!
//! Run: `cargo run -p scry-learn --example ml_viz_showcase`

use scry_chart::export::save_png;
use scry_learn::metrics::{classification_report, confusion_matrix, PrCurve, RocCurve};
use scry_learn::viz;

fn main() {
    let out = "ml_viz_output";
    std::fs::create_dir_all(out).unwrap();

    // --- Phase 1: Fixed existing charts ---

    // 1. Confusion Matrix (raw)
    let y_true = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let y_pred = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
    let cm = confusion_matrix(&y_true, &y_pred);
    let chart = viz::confusion_matrix_chart(&cm, false);
    save_png(&chart, 600, 500, &format!("{out}/confusion_matrix.png")).unwrap();
    println!("✓ confusion_matrix.png");

    // 2. Confusion Matrix (normalized)
    let chart = viz::confusion_matrix_chart(&cm, true);
    save_png(
        &chart,
        600,
        500,
        &format!("{out}/confusion_matrix_norm.png"),
    )
    .unwrap();
    println!("✓ confusion_matrix_norm.png");

    // 3. ROC Curve
    let roc_a = RocCurve::new(
        vec![0.0, 0.1, 0.2, 0.5, 1.0],
        vec![0.0, 0.4, 0.7, 0.9, 1.0],
        vec![0.9, 0.7, 0.5, 0.3],
        0.87,
    );
    let roc_b = RocCurve::new(
        vec![0.0, 0.2, 0.4, 0.6, 1.0],
        vec![0.0, 0.3, 0.5, 0.7, 1.0],
        vec![0.9, 0.7, 0.5, 0.3],
        0.72,
    );
    let chart = viz::roc_chart(&[("Model A", &roc_a), ("Model B", &roc_b)]);
    save_png(&chart, 700, 500, &format!("{out}/roc_curve.png")).unwrap();
    println!("✓ roc_curve.png");

    // 4. Feature Importances (top 5)
    let feat_names: Vec<String> = (0..8).map(|i| format!("feature_{i}")).collect();
    let importances = vec![0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05];
    let chart = viz::feature_importance_chart(&feat_names, &importances, Some(5));
    save_png(&chart, 700, 400, &format!("{out}/feature_importance.png")).unwrap();
    println!("✓ feature_importance.png");

    // 5. Residual Plot
    let y_true_reg = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_pred_reg = vec![1.2, 1.8, 3.3, 3.7, 5.1, 5.9, 7.2, 7.8];
    let chart = viz::residual_plot(&y_true_reg, &y_pred_reg);
    save_png(&chart, 700, 500, &format!("{out}/residual_plot.png")).unwrap();
    println!("✓ residual_plot.png");

    // 6. Regularization Path
    let lambdas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
    let coefficients = vec![
        vec![2.5, -1.8, 0.9],
        vec![2.0, -1.2, 0.7],
        vec![1.0, -0.5, 0.3],
        vec![0.3, -0.1, 0.1],
        vec![0.05, -0.01, 0.01],
    ];
    let coef_names = vec![
        "Weight".to_string(),
        "Height".to_string(),
        "Age".to_string(),
    ];
    let chart = viz::regularization_path_chart(&lambdas, &coefficients, &coef_names);
    save_png(&chart, 700, 500, &format!("{out}/regularization_path.png")).unwrap();
    println!("✓ regularization_path.png");

    // --- Phase 2: Classification & Regression ---

    // 7. PR Curve
    let pr_a = PrCurve::new(
        vec![1.0, 0.95, 0.9, 0.8, 0.6, 0.4],
        vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vec![0.9, 0.8, 0.6, 0.4, 0.2],
        0.78,
    );
    let chart = viz::pr_chart(&[("Classifier", &pr_a)]);
    save_png(&chart, 700, 500, &format!("{out}/pr_curve.png")).unwrap();
    println!("✓ pr_curve.png");

    // 8. Learning Curve
    let chart = viz::learning_curve(
        &[50.0, 100.0, 200.0, 400.0, 800.0],
        &[0.65, 0.72, 0.80, 0.85, 0.88],
        &[0.60, 0.68, 0.73, 0.76, 0.78],
    );
    save_png(&chart, 700, 500, &format!("{out}/learning_curve.png")).unwrap();
    println!("✓ learning_curve.png");

    // 9. Prediction Error
    let chart = viz::prediction_error_chart(&y_true_reg, &y_pred_reg);
    save_png(&chart, 600, 600, &format!("{out}/prediction_error.png")).unwrap();
    println!("✓ prediction_error.png");

    // 10. Calibration Curve
    let chart = viz::calibration_chart(&[
        (
            "Logistic Regression",
            &[0.1, 0.3, 0.5, 0.7, 0.9][..],
            &[0.12, 0.28, 0.52, 0.68, 0.88][..],
        ),
        (
            "Random Forest",
            &[0.15, 0.35, 0.55, 0.75, 0.95][..],
            &[0.1, 0.3, 0.5, 0.7, 0.9][..],
        ),
    ]);
    save_png(&chart, 700, 500, &format!("{out}/calibration.png")).unwrap();
    println!("✓ calibration.png");

    // 11. Classification Report Heatmap
    let report_y_true = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let report_y_pred = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
    let report = classification_report(&report_y_true, &report_y_pred);
    let chart = viz::class_report_chart(&report);
    save_png(&chart, 600, 400, &format!("{out}/class_report.png")).unwrap();
    println!("✓ class_report.png");

    // 12. Metric Comparison
    let model_names: Vec<String> = vec!["DT".into(), "RF".into(), "SVM".into(), "KNN".into()];
    let chart = viz::metric_comparison_chart(
        &model_names,
        &[
            ("Accuracy", &[0.82, 0.91, 0.87, 0.84][..]),
            ("Precision", &[0.80, 0.89, 0.85, 0.83][..]),
            ("Recall", &[0.78, 0.88, 0.84, 0.81][..]),
            ("F1", &[0.79, 0.88, 0.84, 0.82][..]),
        ],
    );
    save_png(&chart, 800, 500, &format!("{out}/metric_comparison.png")).unwrap();
    println!("✓ metric_comparison.png");

    // --- Phase 3: Unsupervised & Tree ---

    // 13. Elbow Chart
    let chart = viz::elbow_chart(
        &[2, 3, 4, 5, 6, 7, 8],
        &[500.0, 280.0, 150.0, 100.0, 85.0, 78.0, 75.0],
        Some(4),
    );
    save_png(&chart, 700, 500, &format!("{out}/elbow.png")).unwrap();
    println!("✓ elbow.png");

    // 14. Cluster Scatter
    let chart = viz::cluster_scatter(
        &[1.0, 1.5, 2.0, 5.0, 5.5, 6.0, 9.0, 9.5, 10.0],
        &[2.0, 2.5, 1.5, 5.0, 5.5, 4.5, 8.0, 8.5, 7.5],
        &[0, 0, 0, 1, 1, 1, 2, 2, 2],
    );
    save_png(&chart, 600, 600, &format!("{out}/cluster_scatter.png")).unwrap();
    println!("✓ cluster_scatter.png");

    // 15. Silhouette Plot
    let chart = viz::silhouette_chart(
        &[0, 0, 0, 1, 1, 1, 2, 2, 2],
        &[0.8, 0.75, 0.6, 0.9, 0.85, 0.7, 0.5, 0.45, 0.3],
    );
    save_png(&chart, 700, 500, &format!("{out}/silhouette.png")).unwrap();
    println!("✓ silhouette.png");

    println!("\n🎉 All 15 charts saved to {out}/");
}
