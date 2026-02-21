use std::f64::consts::PI;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Whisper uses 80 mel bands, 400-sample window (25ms at 16kHz), 160-sample hop (10ms).
pub const WHISPER_N_MELS: usize = 80;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// Maximum audio chunk duration: 30 seconds.
pub const WHISPER_CHUNK_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize * 30; // 480_000

/// Compute the log-mel spectrogram of an audio signal.
///
/// Input: `samples` — mono PCM audio at 16kHz (f32).
/// Output: `[n_mels, n_frames]` row-major f32 array, where
///         `n_frames = samples.len() / hop_length + 1`.
///
/// This matches the output of `whisper.log_mel_spectrogram()` in the Python reference.
pub fn log_mel_spectrogram(samples: &[f32]) -> MelSpectrogram {
    let n_fft = WHISPER_N_FFT;
    let hop = WHISPER_HOP_LENGTH;
    let n_mels = WHISPER_N_MELS;
    let n_freq = n_fft / 2 + 1; // 201

    // Pad to at least one full frame
    let padded_len = if samples.len() < n_fft {
        n_fft
    } else {
        samples.len()
    };
    let n_frames = padded_len / hop + 1;

    // Create Hann window
    let window = hann_window(n_fft);

    // Compute mel filterbank
    let filters = mel_filterbank(n_mels, n_fft, WHISPER_SAMPLE_RATE);

    // Set up FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // STFT → power spectrum → mel → log
    let mut mel_spec = vec![0.0f32; n_mels * n_frames];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;

        // Extract frame with zero-padding at boundaries
        let mut frame: Vec<Complex<f64>> = (0..n_fft)
            .map(|i| {
                let sample = if start + i < samples.len() {
                    f64::from(samples[start + i])
                } else {
                    0.0
                };
                Complex::new(sample * window[i], 0.0)
            })
            .collect();

        // In-place FFT
        fft.process(&mut frame);

        // Power spectrum (magnitude squared) for positive frequencies
        let power: Vec<f64> = frame[..n_freq]
            .iter()
            .map(rustfft::num_complex::Complex::norm_sqr)
            .collect();

        // Apply mel filterbank
        for (mel_idx, filter) in filters.iter().enumerate() {
            let mut energy = 0.0f64;
            for (freq_idx, &coeff) in filter.iter().enumerate() {
                energy += coeff * power[freq_idx];
            }
            // Clamp to minimum value to avoid log(0)
            let clamped = energy.max(1e-10);
            mel_spec[mel_idx * n_frames + frame_idx] = clamped.log10().max(-10.0) as f32;
        }
    }

    // Normalize: scale to match Whisper's reference implementation
    // Whisper applies: log_spec = torch.clamp(log_spec, min=log_spec.max() - 8.0)
    //                  log_spec = (log_spec + 4.0) / 4.0
    let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let clamp_min = max_val - 8.0;
    for v in &mut mel_spec {
        *v = (*v).max(clamp_min);
        *v = (*v + 4.0) / 4.0;
    }

    MelSpectrogram {
        data: mel_spec,
        n_mels,
        n_frames,
    }
}

/// A computed mel spectrogram, stored row-major as `[n_mels, n_frames]`.
#[derive(Clone, Debug)]
pub struct MelSpectrogram {
    /// Row-major data: `data[mel * n_frames + frame]`.
    pub data: Vec<f32>,
    /// Number of mel frequency bands (typically 80).
    pub n_mels: usize,
    /// Number of time frames.
    pub n_frames: usize,
}

impl MelSpectrogram {
    /// Pad or truncate to exactly `target_frames` time frames.
    /// Whisper expects 3000 frames for a 30-second chunk.
    pub fn pad_or_truncate(&self, target_frames: usize) -> Self {
        let mut data = vec![0.0f32; self.n_mels * target_frames];
        let copy_frames = self.n_frames.min(target_frames);
        for mel in 0..self.n_mels {
            let src_start = mel * self.n_frames;
            let dst_start = mel * target_frames;
            data[dst_start..dst_start + copy_frames]
                .copy_from_slice(&self.data[src_start..src_start + copy_frames]);
        }
        Self {
            data,
            n_mels: self.n_mels,
            n_frames: target_frames,
        }
    }
}

/// Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = (PI * i as f64) / n as f64;
            x.sin().powi(2)
        })
        .collect()
}

/// Compute mel filterbank matrix `[n_mels, n_freq]`.
///
/// Returns a vector of `n_mels` filters, each a vector of `n_freq` coefficients.
/// Uses the Slaney formula (same as `librosa.filters.mel`).
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<f64>> {
    let n_freq = n_fft / 2 + 1;
    let sr = f64::from(sample_rate);

    // Mel scale: Hz → mel and mel → Hz
    let hz_to_mel = |f: f64| -> f64 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f64| -> f64 { 700.0 * (10.0_f64.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sr / 2.0);

    // n_mels + 2 evenly spaced points in mel scale
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz points to FFT bin indices (fractional)
    let fft_bins: Vec<f64> = hz_points
        .iter()
        .map(|&f| f * n_fft as f64 / sr)
        .collect();

    // Build triangular filters
    let mut filters = vec![vec![0.0f64; n_freq]; n_mels];
    for m in 0..n_mels {
        let left = fft_bins[m];
        let center = fft_bins[m + 1];
        let right = fft_bins[m + 2];

        for k in 0..n_freq {
            let kf = k as f64;
            if kf >= left && kf <= center {
                let denom = center - left;
                if denom > 0.0 {
                    filters[m][k] = (kf - left) / denom;
                }
            } else if kf > center && kf <= right {
                let denom = right - center;
                if denom > 0.0 {
                    filters[m][k] = (right - kf) / denom;
                }
            }
        }

        // Slaney normalization: scale by 2 / (hz_high - hz_low)
        let hz_width = hz_points[m + 2] - hz_points[m];
        if hz_width > 0.0 {
            let norm = 2.0 / hz_width;
            for v in &mut filters[m] {
                *v *= norm;
            }
        }
    }

    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_spectrogram_shape() {
        // 1 second of silence at 16kHz
        let samples = vec![0.0f32; 16_000];
        let mel = log_mel_spectrogram(&samples);
        assert_eq!(mel.n_mels, 80);
        // n_frames = 16000 / 160 + 1 = 101
        assert_eq!(mel.n_frames, 101);
        assert_eq!(mel.data.len(), 80 * 101);
    }

    #[test]
    fn mel_spectrogram_pad_truncate() {
        let samples = vec![0.0f32; 16_000];
        let mel = log_mel_spectrogram(&samples);
        let padded = mel.pad_or_truncate(3000);
        assert_eq!(padded.n_frames, 3000);
        assert_eq!(padded.data.len(), 80 * 3000);
    }

    #[test]
    fn hann_window_endpoints() {
        let w = hann_window(400);
        assert_eq!(w.len(), 400);
        // Hann window is ~0 at endpoints
        assert!(w[0].abs() < 1e-10);
        // Peak at center
        assert!((w[200] - 1.0).abs() < 0.01);
    }

    #[test]
    fn mel_filterbank_shape() {
        let filters = mel_filterbank(80, 400, 16_000);
        assert_eq!(filters.len(), 80);
        assert_eq!(filters[0].len(), 201); // n_fft/2 + 1
    }

    #[test]
    fn mel_no_nan() {
        // Sine wave at 440Hz
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let mel = log_mel_spectrogram(&samples);
        assert!(!mel.data.iter().any(|v| v.is_nan()));
    }
}
