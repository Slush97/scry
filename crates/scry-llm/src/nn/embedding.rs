use crate::autograd::ops;
use crate::autograd::GradTape;
use crate::backend::MathBackend;
use crate::nn::init;
use crate::nn::Module;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Token + position embedding layer.
pub struct EmbeddingLayer<B: MathBackend> {
    pub token_embedding: Tensor<B>,
    pub position_embedding: Tensor<B>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
}

impl<B: MathBackend> EmbeddingLayer<B> {
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        rng: &mut fastrand::Rng,
    ) -> Self {
        let tok_data = init::normal_vec(rng, vocab_size * d_model, 0.0, 0.02);
        let pos_data = init::normal_vec(rng, max_seq_len * d_model, 0.0, 0.02);
        Self {
            token_embedding: Tensor::from_vec(tok_data, Shape::new(&[vocab_size, d_model])),
            position_embedding: Tensor::from_vec(pos_data, Shape::new(&[max_seq_len, d_model])),
            vocab_size,
            max_seq_len,
            d_model,
        }
    }

    /// Forward: look up token + position embeddings and add them.
    /// `token_ids`: sequence of token indices.
    pub fn forward(&self, token_ids: &[usize], tape: &mut GradTape<B>) -> Tensor<B> {
        let seq_len = token_ids.len();
        let positions: Vec<usize> = (0..seq_len).collect();

        let tok_emb = ops::embedding(
            &self.token_embedding,
            token_ids,
            self.vocab_size,
            self.d_model,
            Some(tape),
        );
        let pos_emb = ops::embedding(
            &self.position_embedding,
            &positions,
            self.max_seq_len,
            self.d_model,
            Some(tape),
        );
        ops::add(&tok_emb, &pos_emb, Some(tape))
    }

    /// Batched forward: `token_ids` is `[batch_size * seq_len]` flat, with position
    /// indices resetting per sequence.
    pub fn forward_batch(
        &self,
        token_ids: &[usize],
        batch_size: usize,
        seq_len: usize,
        tape: &mut GradTape<B>,
    ) -> Tensor<B> {
        // Position indices reset per sequence: [0,1,...,seq-1, 0,1,...,seq-1, ...]
        let positions: Vec<usize> = (0..batch_size).flat_map(|_| 0..seq_len).collect();

        let tok_emb = ops::embedding(
            &self.token_embedding,
            token_ids,
            self.vocab_size,
            self.d_model,
            Some(tape),
        );
        let pos_emb = ops::embedding(
            &self.position_embedding,
            &positions,
            self.max_seq_len,
            self.d_model,
            Some(tape),
        );
        ops::add(&tok_emb, &pos_emb, Some(tape))
    }

    pub fn forward_inference(&self, token_ids: &[usize]) -> Tensor<B> {
        let seq_len = token_ids.len();
        let positions: Vec<usize> = (0..seq_len).collect();

        let tok_emb = ops::embedding(
            &self.token_embedding,
            token_ids,
            self.vocab_size,
            self.d_model,
            None,
        );
        let pos_emb = ops::embedding(
            &self.position_embedding,
            &positions,
            self.max_seq_len,
            self.d_model,
            None,
        );
        ops::add(&tok_emb, &pos_emb, None)
    }
}

impl<B: MathBackend> Module<B> for EmbeddingLayer<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.token_embedding, &self.position_embedding]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>> {
        vec![&mut self.token_embedding, &mut self.position_embedding]
    }
}
