use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// 1D convolution layer for Whisper's audio feature extraction stem.
///
/// Whisper uses two `Conv1D` layers to downsample the mel spectrogram:
///   - Conv1D(80, `d_model`, kernel=3, stride=1, padding=1)
///   - `Conv1D(d_model`, `d_model`, kernel=3, stride=2, padding=1)
///
/// Input shape: `[channels_in, length]`
/// Output shape: `[channels_out, output_length]`
pub struct Conv1d<B: MathBackend> {
    /// Weight tensor: `[out_channels, in_channels, kernel_size]`.
    pub weight: Tensor<B>,
    /// Bias tensor: `[out_channels]`.
    pub bias: Tensor<B>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Convolution kernel size.
    pub kernel_size: usize,
    /// Convolution stride.
    pub stride: usize,
    /// Padding added to both sides of the input.
    pub padding: usize,
}

impl<B: MathBackend> Conv1d<B> {
    /// Create a new `Conv1D` layer with random initialization.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        rng: &mut fastrand::Rng,
    ) -> Self {
        let n = in_channels * out_channels * kernel_size;
        let std_dev = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let w_data: Vec<f32> = (0..n)
            .map(|_| (rng.f64() * 2.0 - 1.0) * std_dev)
            .map(|v| v as f32)
            .collect();
        let b_data = vec![0.0f32; out_channels];

        Self {
            weight: Tensor::from_vec(
                w_data,
                Shape::new(&[out_channels, in_channels, kernel_size]),
            ),
            bias: Tensor::from_vec(b_data, Shape::new(&[out_channels])),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass: `input` is `[in_channels, length]` → `[out_channels, out_length]`.
    ///
    /// Performs convolution by unrolling to matrix multiplication for efficiency.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let length = input.shape.dims()[1];
        let padded_len = length + 2 * self.padding;
        let out_length = (padded_len - self.kernel_size) / self.stride + 1;

        let input_vec = input.to_vec();
        let weight_vec = self.weight.to_vec();
        let bias_vec = self.bias.to_vec();

        // Build im2col matrix: [kernel_size * in_channels, out_length]
        let col_rows = self.kernel_size * self.in_channels;
        let mut col = vec![0.0f32; col_rows * out_length];

        for out_pos in 0..out_length {
            let in_start = out_pos * self.stride;
            for c in 0..self.in_channels {
                for k in 0..self.kernel_size {
                    let in_pos = in_start + k;
                    let val = if in_pos >= self.padding && in_pos < self.padding + length {
                        input_vec[c * length + (in_pos - self.padding)]
                    } else {
                        0.0 // zero padding
                    };
                    col[(c * self.kernel_size + k) * out_length + out_pos] = val;
                }
            }
        }

        // weight reshaped: [out_channels, in_channels * kernel_size]
        // matmul: [out_channels, col_rows] @ [col_rows, out_length] → [out_channels, out_length]
        let col_tensor =
            Tensor::<B>::from_vec(col, Shape::new(&[col_rows, out_length]));
        let w_reshaped = Tensor::<B>::from_vec(
            weight_vec,
            Shape::new(&[self.out_channels, col_rows]),
        );

        let out = scry_llm::ops::matmul(
            &w_reshaped,
            &col_tensor,
            self.out_channels,
            col_rows,
            out_length,
            false,
            false,
        );

        // Add bias: broadcast [out_channels, 1] over [out_channels, out_length]
        let bias_tensor = Tensor::<B>::from_vec(bias_vec, Shape::new(&[self.out_channels, 1]));
        scry_llm::ops::add(&out, &bias_tensor)
    }
}

impl<B: MathBackend> Module<B> for Conv1d<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weight, &self.bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn conv1d_output_shape_stride1() {
        let mut rng = fastrand::Rng::with_seed(42);
        let conv = Conv1d::<CpuBackend>::new(80, 512, 3, 1, 1, &mut rng);
        let input = Tensor::<CpuBackend>::from_vec(
            vec![0.0f32; 80 * 3000],
            Shape::new(&[80, 3000]),
        );
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[512, 3000]);
    }

    #[test]
    fn conv1d_output_shape_stride2() {
        let mut rng = fastrand::Rng::with_seed(42);
        let conv = Conv1d::<CpuBackend>::new(512, 512, 3, 2, 1, &mut rng);
        let input = Tensor::<CpuBackend>::from_vec(
            vec![0.0f32; 512 * 3000],
            Shape::new(&[512, 3000]),
        );
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[512, 1500]);
    }
}
