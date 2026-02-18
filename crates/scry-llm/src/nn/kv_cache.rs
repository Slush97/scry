use crate::backend::MathBackend;

/// Per-layer KV cache for autoregressive inference.
///
/// Stores accumulated K and V projections for each attention head.
pub struct LayerKvCache<B: MathBackend> {
    /// `k_per_head[h]`: `[cached_seq, d_head]`
    pub k_per_head: Vec<B::Storage>,
    /// `v_per_head[h]`: `[cached_seq, d_head]`
    pub v_per_head: Vec<B::Storage>,
    /// Number of cached tokens so far.
    pub seq_len: usize,
    pub n_heads: usize,
    pub d_head: usize,
}

impl<B: MathBackend> LayerKvCache<B> {
    pub fn new(n_heads: usize, d_head: usize) -> Self {
        Self {
            k_per_head: Vec::new(),
            v_per_head: Vec::new(),
            seq_len: 0,
            n_heads,
            d_head,
        }
    }

    /// Append new K and V for a single token to each head.
    /// `new_k[h]`: `[1, d_head]`, `new_v[h]`: `[1, d_head]`.
    pub fn append(&mut self, new_k: Vec<B::Storage>, new_v: Vec<B::Storage>) {
        if self.seq_len == 0 {
            self.k_per_head = new_k;
            self.v_per_head = new_v;
        } else {
            for h in 0..self.n_heads {
                self.k_per_head[h] =
                    B::concat_rows(&self.k_per_head[h], &new_k[h], self.seq_len, 1, self.d_head);
                self.v_per_head[h] =
                    B::concat_rows(&self.v_per_head[h], &new_v[h], self.seq_len, 1, self.d_head);
            }
        }
        self.seq_len += 1;
    }
}

/// Full model KV cache: one [`LayerKvCache`] per transformer block.
pub struct KvCache<B: MathBackend> {
    pub layers: Vec<LayerKvCache<B>>,
}

impl<B: MathBackend> KvCache<B> {
    pub fn new(n_layers: usize, n_heads: usize, d_head: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| LayerKvCache::new(n_heads, d_head))
            .collect();
        Self { layers }
    }
}
