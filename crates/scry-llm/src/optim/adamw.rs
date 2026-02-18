use std::collections::HashMap;

use crate::backend::MathBackend;
use crate::tensor::shape::Shape;
use crate::tensor::TensorId;

/// `AdamW` optimizer configuration.
#[derive(Clone, Debug)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 6e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
        }
    }
}

/// `AdamW` optimizer with per-parameter momentum states.
pub struct AdamW<B: MathBackend> {
    pub config: AdamWConfig,
    step_count: u32,
    states: HashMap<TensorId, (B::Storage, B::Storage)>,
}

impl<B: MathBackend> AdamW<B> {
    pub fn new(config: AdamWConfig) -> Self {
        Self {
            config,
            step_count: 0,
            states: HashMap::new(),
        }
    }

    /// Set learning rate (for LR scheduling).
    pub fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    /// Perform one optimization step.
    ///
    /// `params`: mutable references to parameter storage, keyed by `TensorId`.
    /// `grads`: gradient storage from backward pass.
    pub fn step(
        &mut self,
        params: &mut [(TensorId, &mut B::Storage, &Shape)],
        grads: &HashMap<TensorId, B::Storage>,
    ) {
        self.step_count += 1;

        for (id, param, shape) in params.iter_mut() {
            let Some(grad) = grads.get(id) else {
                continue;
            };

            let (m, v) = self
                .states
                .entry(*id)
                .or_insert_with(|| (B::zeros(shape), B::zeros(shape)));

            B::adamw_step(
                param,
                grad,
                m,
                v,
                self.config.lr,
                self.config.beta1,
                self.config.beta2,
                self.config.eps,
                self.config.weight_decay,
                self.step_count,
            );
        }
    }
}
