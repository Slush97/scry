pub mod shape;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::backend::DeviceBackend;
use shape::Shape;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Unique identifier for a tensor, used as key in autograd tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

impl TensorId {
    fn next() -> Self {
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// N-dimensional tensor backed by a device-specific storage.
///
/// `data` is wrapped in `Arc` so that recording to the autograd tape
/// can capture weight tensors via Arc clone (O(1) refcount bump)
/// instead of a full device memcpy.
#[derive(Clone, Debug)]
pub struct Tensor<B: DeviceBackend> {
    pub id: TensorId,
    pub data: Arc<B::Storage>,
    pub shape: Shape,
}

impl<B: DeviceBackend> Tensor<B> {
    pub fn new(data: B::Storage, shape: Shape) -> Self {
        Self {
            id: TensorId::next(),
            data: Arc::new(data),
            shape,
        }
    }

    pub fn zeros(shape: Shape) -> Self {
        let data = B::zeros(&shape);
        Self::new(data, shape)
    }

    pub fn ones(shape: Shape) -> Self {
        let data = B::ones(&shape);
        Self::new(data, shape)
    }

    /// Creates a tensor from a `Vec<f32>` and a [`Shape`].
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `shape.numel()`.
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.numel(),
            "Tensor::from_vec: data length {} != shape numel {}",
            data.len(),
            shape.numel()
        );
        let storage = B::from_vec(data, &shape);
        Self::new(storage, shape)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        B::to_vec(&self.data)
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get exclusive mutable access to the storage.
    ///
    /// # Panics
    ///
    /// Panics if other references to the `Arc` still exist (e.g. tape not dropped).
    pub fn data_mut(&mut self) -> &mut B::Storage {
        Arc::get_mut(&mut self.data)
            .expect("Tensor::data_mut: Arc still shared — drop the tape first")
    }
}
