use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::backend::MathBackend;
use crate::error::ScryLlmError;
use crate::nn::gpt2::{Gpt2Config, Gpt2Model};
use crate::nn::Module;
use crate::optim::adamw::{AdamW, AdamWConfig};
use crate::tensor::TensorId;

/// Save a training checkpoint: model parameters, optimizer state, step, and seed.
///
/// Layout in the safetensors file:
/// - `"param.0"`, `"param.1"`, ... — model parameters in `Module::parameters()` order
/// - `"optim.m.0"`, `"optim.v.0"`, ... — optimizer first/second moments (same order)
/// - JSON metadata header: step, seed, lr, beta1, beta2, eps, weight_decay
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn save_checkpoint<B: MathBackend>(
    path: &Path,
    model: &Gpt2Model<B>,
    optimizer: &AdamW<B>,
    step: usize,
    rng_seed: u64,
) -> crate::error::Result<()> {
    let params = model.parameters();
    let optim_states = optimizer.states();

    // We need to collect all byte buffers first so they live long enough for TensorView refs
    let mut all_bufs: Vec<Vec<u8>> = Vec::new();
    let mut tensor_specs: Vec<(String, Vec<usize>, usize)> = Vec::new(); // (name, shape, buf_idx)

    // Parameters
    for (i, param) in params.iter().enumerate() {
        let data = B::to_vec(&param.data);
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shape: Vec<usize> = param.shape.dims().to_vec();
        let idx = all_bufs.len();
        all_bufs.push(bytes);
        tensor_specs.push((format!("param.{i}"), shape, idx));
    }

    // Optimizer states
    let param_ids: Vec<TensorId> = params.iter().map(|p| p.id).collect();
    for (i, id) in param_ids.iter().enumerate() {
        if let Some((m, v)) = optim_states.get(id) {
            let shape: Vec<usize> = params[i].shape.dims().to_vec();

            let m_data = B::to_vec(m);
            let m_bytes: Vec<u8> = m_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let m_idx = all_bufs.len();
            all_bufs.push(m_bytes);
            tensor_specs.push((format!("optim.m.{i}"), shape.clone(), m_idx));

            let v_data = B::to_vec(v);
            let v_bytes: Vec<u8> = v_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let v_idx = all_bufs.len();
            all_bufs.push(v_bytes);
            tensor_specs.push((format!("optim.v.{i}"), shape, v_idx));
        }
    }

    // Build TensorViews referencing the buffers
    let mut tensors: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for (name, shape, idx) in &tensor_specs {
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.clone(),
            &all_bufs[*idx],
        )
        .map_err(|e| ScryLlmError::CheckpointError(format!("tensor view error: {e}")))?;
        tensors.push((name.clone(), view));
    }

    // Metadata
    let metadata = HashMap::from([
        ("step".to_string(), step.to_string()),
        ("seed".to_string(), rng_seed.to_string()),
        (
            "optim_step_count".to_string(),
            optimizer.step_count().to_string(),
        ),
        ("lr".to_string(), optimizer.config.lr.to_string()),
        ("beta1".to_string(), optimizer.config.beta1.to_string()),
        ("beta2".to_string(), optimizer.config.beta2.to_string()),
        ("eps".to_string(), optimizer.config.eps.to_string()),
        (
            "weight_decay".to_string(),
            optimizer.config.weight_decay.to_string(),
        ),
    ]);

    safetensors::tensor::serialize_to_file(tensors, &Some(metadata), path)
        .map_err(|e| ScryLlmError::CheckpointError(format!("serialize error: {e}")))?;

    Ok(())
}

/// Load a training checkpoint.
///
/// Returns `(model, optimizer, step, rng_seed)`.
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be read or parsed.
pub fn load_checkpoint<B: MathBackend>(
    path: &Path,
    config: &Gpt2Config,
) -> crate::error::Result<(Gpt2Model<B>, AdamW<B>, usize, u64)> {
    let data = std::fs::read(path)
        .map_err(|e| ScryLlmError::CheckpointError(format!("read error: {e}")))?;

    let (_, meta_obj) = safetensors::SafeTensors::read_metadata(&data)
        .map_err(|e| ScryLlmError::CheckpointError(format!("metadata error: {e}")))?;

    let loaded = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| ScryLlmError::CheckpointError(format!("deserialize error: {e}")))?;

    // Read metadata
    let meta = meta_obj
        .metadata()
        .as_ref()
        .ok_or_else(|| ScryLlmError::CheckpointError("no metadata in checkpoint".into()))?;

    let step: usize = meta
        .get("step")
        .ok_or_else(|| ScryLlmError::CheckpointError("missing 'step' in metadata".into()))?
        .parse()
        .map_err(|e| ScryLlmError::CheckpointError(format!("parse step: {e}")))?;

    let seed: u64 = meta
        .get("seed")
        .ok_or_else(|| ScryLlmError::CheckpointError("missing 'seed' in metadata".into()))?
        .parse()
        .map_err(|e| ScryLlmError::CheckpointError(format!("parse seed: {e}")))?;

    let optim_step_count: u32 = meta
        .get("optim_step_count")
        .and_then(|s: &String| s.parse().ok())
        .unwrap_or(step as u32);

    let adamw_config = AdamWConfig {
        lr: meta.get("lr").and_then(|s: &String| s.parse().ok()).unwrap_or(6e-4),
        beta1: meta.get("beta1").and_then(|s: &String| s.parse().ok()).unwrap_or(0.9),
        beta2: meta.get("beta2").and_then(|s: &String| s.parse().ok()).unwrap_or(0.95),
        eps: meta.get("eps").and_then(|s: &String| s.parse().ok()).unwrap_or(1e-8),
        weight_decay: meta.get("weight_decay").and_then(|s: &String| s.parse().ok()).unwrap_or(0.1),
    };

    // Reconstruct model with dummy init, then overwrite parameters
    let mut rng = fastrand::Rng::with_seed(0);
    let mut model = Gpt2Model::<B>::new(config.clone(), &mut rng);

    let n_params = model.parameters().len();
    for i in 0..n_params {
        let tensor_name = format!("param.{i}");
        let t = loaded
            .tensor(&tensor_name)
            .map_err(|e| ScryLlmError::CheckpointError(format!("missing {tensor_name}: {e}")))?;
        let floats = bytes_to_f32(t.data());
        let shape = {
            let params = model.parameters();
            params[i].shape.clone()
        };
        let mut params = model.parameters_mut();
        params[i].data = Arc::new(B::from_vec(floats, &shape));
    }

    // Reconstruct optimizer states
    let param_ids: Vec<TensorId> = model.parameters().iter().map(|p| p.id).collect();
    let mut optim_states: HashMap<TensorId, (B::Storage, B::Storage)> = HashMap::new();

    for (i, id) in param_ids.iter().enumerate() {
        let m_name = format!("optim.m.{i}");
        let v_name = format!("optim.v.{i}");
        if let (Ok(m_tensor), Ok(v_tensor)) = (loaded.tensor(&m_name), loaded.tensor(&v_name)) {
            let shape = model.parameters()[i].shape.clone();
            let m = B::from_vec(bytes_to_f32(m_tensor.data()), &shape);
            let v = B::from_vec(bytes_to_f32(v_tensor.data()), &shape);
            optim_states.insert(*id, (m, v));
        }
    }

    let optimizer = AdamW::from_state(adamw_config, optim_step_count, optim_states);

    Ok((model, optimizer, step, seed))
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
