//! Pure-Rust L-BFGS optimizer for smooth unconstrained optimization.
//!
//! Implements the limited-memory BFGS algorithm with two-loop recursion
//! (Nocedal & Wright, 2006) and Armijo backtracking line search.
//!
//! This is the gold-standard optimizer for smooth convex problems such as
//! logistic regression — scikit-learn uses it as their default solver.

use std::collections::VecDeque;

/// Configuration for the L-BFGS minimizer.
pub(crate) struct LbfgsConfig {
    /// Maximum number of L-BFGS iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the gradient infinity-norm.
    pub tolerance: f64,
    /// Number of correction pairs to store (default 10).
    pub history_size: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tolerance: 1e-6,
            history_size: 10,
        }
    }
}

/// Minimize `f(x)` using L-BFGS.
///
/// # Arguments
///
/// * `x0` — initial parameter vector (mutated in-place to the solution).
/// * `eval` — closure `|x: &[f64], grad: &mut [f64]| -> f64` that computes the
///   function value and writes the gradient into `grad`.
/// * `config` — optimizer configuration.
///
/// # Returns
///
/// The final objective value.
pub(crate) fn minimize(
    x0: &mut [f64],
    mut eval: impl FnMut(&[f64], &mut [f64]) -> f64,
    config: &LbfgsConfig,
) -> f64 {
    let n = x0.len();
    let mut grad = vec![0.0; n];
    let mut f = eval(x0, &mut grad);

    // History of correction pairs (s_k, y_k) and ρ_k = 1 / (y_k · s_k).
    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(config.history_size);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(config.history_size);
    let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(config.history_size);

    let mut x_prev = vec![0.0; n];
    let mut grad_prev = vec![0.0; n];
    let mut direction = vec![0.0; n];

    for _iter in 0..config.max_iter {
        // Check convergence: ||grad||_inf < tolerance.
        let grad_norm = grad.iter().fold(0.0_f64, |mx, &g| mx.max(g.abs()));
        if grad_norm < config.tolerance {
            break;
        }

        // Save previous state.
        x_prev.copy_from_slice(x0);
        grad_prev.copy_from_slice(&grad);

        // ─── Two-loop recursion ───────────────────────────────────
        // Compute search direction d = -H_k * grad  (approximate).
        direction.copy_from_slice(&grad);

        let m = s_hist.len();
        let mut alphas = vec![0.0; m];

        // First loop (reverse).
        for i in (0..m).rev() {
            let s = &s_hist[i];
            let rho = rho_hist[i];
            let mut dot = 0.0;
            for j in 0..n {
                dot += s[j] * direction[j];
            }
            alphas[i] = rho * dot;
            let y = &y_hist[i];
            for j in 0..n {
                direction[j] -= alphas[i] * y[j];
            }
        }

        // Initial Hessian scaling: H_0 = γ I where γ = (s_{k-1} · y_{k-1}) / (y_{k-1} · y_{k-1}).
        if m > 0 {
            let s_last = &s_hist[m - 1];
            let y_last = &y_hist[m - 1];
            let mut sy = 0.0;
            let mut yy = 0.0;
            for j in 0..n {
                sy += s_last[j] * y_last[j];
                yy += y_last[j] * y_last[j];
            }
            let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
            for d in &mut direction {
                *d *= gamma;
            }
        }

        // Second loop (forward).
        for i in 0..m {
            let y = &y_hist[i];
            let rho = rho_hist[i];
            let mut dot = 0.0;
            for j in 0..n {
                dot += y[j] * direction[j];
            }
            let beta = rho * dot;
            let s = &s_hist[i];
            for j in 0..n {
                direction[j] += (alphas[i] - beta) * s[j];
            }
        }

        // Negate to get descent direction.
        for d in &mut direction {
            *d = -*d;
        }

        // ─── Armijo backtracking line search ──────────────────────
        // Find step α such that f(x + α·d) ≤ f(x) + c·α·(grad · d).
        let c_armijo = 1e-4;
        let rho_bt = 0.5;
        let mut step = 1.0;

        let dir_deriv: f64 = grad.iter().zip(direction.iter()).map(|(&g, &d)| g * d).sum();

        // Guard: if direction is not a descent direction, fall back to steepest descent.
        if dir_deriv >= 0.0 {
            for (d, g) in direction.iter_mut().zip(grad.iter()) {
                *d = -*g;
            }
            // No line search needed for steepest descent; use a small step.
            step = 0.01 / grad_norm.max(1.0);
        }

        let dir_deriv_ls: f64 = grad.iter().zip(direction.iter()).map(|(&g, &d)| g * d).sum();

        let mut x_trial: Vec<f64> = x0.to_vec();
        for _ls in 0..20 {
            for j in 0..n {
                x_trial[j] = x0[j] + step * direction[j];
            }
            let f_trial = eval(&x_trial, &mut grad);
            if f_trial <= f + c_armijo * step * dir_deriv_ls {
                f = f_trial;
                x0.copy_from_slice(&x_trial);
                break;
            }
            step *= rho_bt;
        }

        // ─── Update history ──────────────────────────────────────
        let mut s_k = vec![0.0; n];
        let mut y_k = vec![0.0; n];
        let mut sy = 0.0;
        for j in 0..n {
            s_k[j] = x0[j] - x_prev[j];
            y_k[j] = grad[j] - grad_prev[j];
            sy += s_k[j] * y_k[j];
        }

        // Only add to history if curvature condition holds.
        if sy > 1e-16 {
            if s_hist.len() == config.history_size {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(s_k);
            y_hist.push_back(y_k);
        }
    }

    f
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    /// Minimum at (1, 1) with f = 0.
    #[test]
    fn test_rosenbrock() {
        let mut x = vec![-1.0, 1.0];
        let config = LbfgsConfig {
            max_iter: 500,
            tolerance: 1e-10,
            history_size: 10,
        };

        let f = minimize(
            &mut x,
            |x, grad| {
                let a = 1.0 - x[0];
                let b = x[1] - x[0] * x[0];
                grad[0] = -2.0 * a - 400.0 * x[0] * b;
                grad[1] = 200.0 * b;
                a * a + 100.0 * b * b
            },
            &config,
        );

        assert!(f < 1e-10, "expected f ≈ 0, got {f}");
        assert!((x[0] - 1.0).abs() < 1e-4, "expected x[0] ≈ 1, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "expected x[1] ≈ 1, got {}", x[1]);
    }

    /// Simple quadratic: f(x) = 0.5 * x^T x, gradient = x, minimum at origin.
    #[test]
    fn test_quadratic() {
        let mut x = vec![3.0, -4.0, 5.0];
        let config = LbfgsConfig::default();

        let f = minimize(
            &mut x,
            |x, grad| {
                let mut val = 0.0;
                for (g, &xi) in grad.iter_mut().zip(x.iter()) {
                    *g = xi;
                    val += 0.5 * xi * xi;
                }
                val
            },
            &config,
        );

        assert!(f < 1e-12, "expected f ≈ 0, got {f}");
        for (i, &xi) in x.iter().enumerate() {
            assert!(xi.abs() < 1e-6, "expected x[{i}] ≈ 0, got {xi}");
        }
    }
}
