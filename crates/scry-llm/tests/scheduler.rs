use scry_llm::optim::scheduler::CosineScheduler;

#[test]
fn warmup_ramps_linearly() {
    let sched = CosineScheduler::new(100, 1000, 6e-4, 1e-5);

    // Step 0: lr = 0
    let lr0 = sched.get_lr(0);
    assert!((lr0).abs() < 1e-9, "step 0 should be ~0, got {lr0}");

    // Step 50: halfway through warmup -> peak/2
    let lr50 = sched.get_lr(50);
    assert!(
        (lr50 - 3e-4).abs() < 1e-7,
        "step 50 should be ~3e-4, got {lr50}"
    );

    // Step 100: end of warmup -> peak_lr
    let lr100 = sched.get_lr(100);
    assert!(
        (lr100 - 6e-4).abs() < 1e-7,
        "step 100 should be ~6e-4, got {lr100}"
    );
}

#[test]
fn warmup_is_monotonically_increasing() {
    let sched = CosineScheduler::new(200, 2000, 1e-3, 0.0);
    let mut prev = 0.0f32;
    for step in 0..=200 {
        let lr = sched.get_lr(step);
        assert!(lr >= prev, "warmup not monotone at step {step}: {prev} > {lr}");
        prev = lr;
    }
}

#[test]
fn decay_is_monotonically_decreasing() {
    let sched = CosineScheduler::new(100, 1000, 6e-4, 1e-5);
    let mut prev = sched.get_lr(100);
    for step in 101..=1000 {
        let lr = sched.get_lr(step);
        assert!(
            lr <= prev + 1e-10,
            "decay not monotone at step {step}: {prev} < {lr}"
        );
        prev = lr;
    }
}

#[test]
fn at_total_steps_returns_min_lr() {
    let sched = CosineScheduler::new(100, 1000, 6e-4, 1e-5);
    let lr = sched.get_lr(1000);
    assert!(
        (lr - 1e-5).abs() < 1e-9,
        "at total_steps should be min_lr, got {lr}"
    );
}

#[test]
fn after_total_steps_clamps_to_min_lr() {
    let sched = CosineScheduler::new(100, 1000, 6e-4, 1e-5);
    let lr = sched.get_lr(5000);
    assert!(
        (lr - 1e-5).abs() < 1e-9,
        "after total_steps should be min_lr, got {lr}"
    );
}

#[test]
fn zero_warmup_steps() {
    let sched = CosineScheduler::new(0, 1000, 6e-4, 0.0);
    // Step 0 should be peak_lr (no warmup)
    let lr0 = sched.get_lr(0);
    assert!(
        (lr0 - 6e-4).abs() < 1e-9,
        "step 0 with no warmup should be peak_lr, got {lr0}"
    );
}

#[test]
fn halfway_decay_returns_midpoint() {
    let sched = CosineScheduler::new(0, 1000, 1.0, 0.0);
    // At halfway through cosine decay, cos(pi/2) = 0, so lr = 0.5
    let lr = sched.get_lr(500);
    assert!(
        (lr - 0.5).abs() < 1e-5,
        "halfway decay should be ~0.5, got {lr}"
    );
}
