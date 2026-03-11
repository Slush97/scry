use scry_chart::export;
use scry_chart::prelude::*;

fn main() {
    let dir = "output/gantt";
    std::fs::create_dir_all(dir).unwrap();

    // ── Project Timeline (Dark theme) — day-index X axis ──
    let chart = Charts::gantt(vec![
        GanttTask::new("Research", 0.0, 3.0)
            .group("Phase 1")
            .progress(1.0),
        GanttTask::new("Design", 2.0, 6.0)
            .group("Phase 1")
            .progress(0.8),
        GanttTask::new("Prototype", 4.0, 8.0)
            .group("Phase 2")
            .progress(0.5),
        GanttTask::new("Implement", 7.0, 14.0)
            .group("Phase 2")
            .progress(0.3),
        GanttTask::new("Testing", 12.0, 17.0).group("Phase 3"),
        GanttTask::new("Documentation", 14.0, 18.0).group("Phase 3"),
        GanttTask::new("Deploy", 17.0, 19.0).group("Launch"),
    ])
    .title("Product Development Timeline")
    .x_label("Day")
    .theme(Theme::dark())
    .build();

    let path = format!("{dir}/gantt_dark.png");
    export::save_png(&chart, 900, 450, &path).expect("export failed");
    println!("Saved {path}");

    // ── Sprint plan (Light theme) — day-index X axis ──
    let chart = Charts::gantt(vec![
        GanttTask::new("Auth Module", 0.0, 5.0)
            .group("Backend")
            .progress(1.0),
        GanttTask::new("API Gateway", 3.0, 8.0)
            .group("Backend")
            .progress(0.6),
        GanttTask::new("Login UI", 2.0, 6.0)
            .group("Frontend")
            .progress(0.9),
        GanttTask::new("Dashboard", 5.0, 12.0)
            .group("Frontend")
            .progress(0.2),
        GanttTask::new("Load Tests", 8.0, 11.0).group("QA"),
        GanttTask::new("E2E Tests", 10.0, 14.0).group("QA"),
    ])
    .title("Sprint 12 — Feature Delivery")
    .x_label("Sprint Day")
    .theme(Theme::light())
    .build();

    let path = format!("{dir}/gantt_light.png");
    export::save_png(&chart, 900, 450, &path).expect("export failed");
    println!("Saved {path}");

    // ── Time-based axis (epoch seconds) — 2 weeks of work ──
    // Start: Jan 6, 2025 00:00 UTC  (epoch 1736121600)
    let day = 86400.0_f64;
    let base = 1_736_121_600.0; // 2025-01-06 00:00 UTC
    let chart = Charts::gantt(vec![
        GanttTask::new("Sprint Planning", base, base + 1.0 * day)
            .group("Meetings")
            .progress(1.0),
        GanttTask::new("Backend API", base + 1.0 * day, base + 6.0 * day)
            .group("Engineering")
            .progress(0.7),
        GanttTask::new("Frontend", base + 3.0 * day, base + 8.0 * day)
            .group("Engineering")
            .progress(0.4),
        GanttTask::new("QA Testing", base + 7.0 * day, base + 10.0 * day).group("QA"),
        GanttTask::new("Code Review", base + 5.0 * day, base + 9.0 * day).group("Engineering"),
        GanttTask::new("Staging Deploy", base + 10.0 * day, base + 11.0 * day).group("DevOps"),
        GanttTask::new("Sprint Retro", base + 11.0 * day, base + 11.5 * day).group("Meetings"),
    ])
    .title("Sprint 23 — Jan 6–17, 2025")
    .x_label("Date")
    .time_axis()
    .theme(Theme::dark())
    .build();

    let path = format!("{dir}/gantt_time.png");
    export::save_png(&chart, 900, 450, &path).expect("export failed");
    println!("Saved {path}");

    println!("\nDone! Check {dir}/");
}
