//! FFT analysis, semantic band groups, smoothing, auto-gain, and beat detection.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::analysis::{grouped_band_energy, smooth_control, AnalysisFrame};
use crate::audio::{SharedAudio, SAMPLE_RATE};

const FFT_SIZE: usize = 2048;
pub const WAVEFORM_LEN: usize = 1024;
const FREQ_MIN: f32 = 30.0;
const FREQ_MAX: f32 = 16000.0;
const DB_RANGE: f32 = 50.0;
const BEAT_HISTORY: usize = 43;

pub struct Analyzer {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    samples: Vec<f32>,
    fft_buf: Vec<Complex<f32>>,
    /// (start_bin, end_bin) per band, log-spaced.
    band_bins: Vec<(usize, usize)>,
    /// (start_hz, end_hz) per band, aligned with `band_bins`.
    band_ranges_hz: Vec<(f32, f32)>,
    prev_bands: Vec<f32>,
    ref_db: f32,
    bass_history: [f32; BEAT_HISTORY],
    bass_head: usize,
    since_beat: f32,
    beat_armed: bool,
    pub frame: AnalysisFrame,
}

impl Analyzer {
    pub fn new(num_bands: usize) -> Self {
        let window = (0..FFT_SIZE)
            .map(|i| {
                let t = i as f32 / (FFT_SIZE - 1) as f32;
                0.5 - 0.5 * (std::f32::consts::TAU * t).cos()
            })
            .collect();

        let bin_hz = SAMPLE_RATE as f32 / FFT_SIZE as f32;
        let log_min = FREQ_MIN.ln();
        let log_max = FREQ_MAX.ln();
        let band_bins: Vec<_> = (0..num_bands)
            .map(|i| {
                let f0 = (log_min + (log_max - log_min) * i as f32 / num_bands as f32).exp();
                let f1 = (log_min + (log_max - log_min) * (i + 1) as f32 / num_bands as f32).exp();
                let b0 = (f0 / bin_hz) as usize;
                let b1 = ((f1 / bin_hz) as usize).max(b0 + 1).min(FFT_SIZE / 2);
                (b0.min(FFT_SIZE / 2 - 1), b1)
            })
            .collect();
        let band_ranges_hz = band_bins
            .iter()
            .map(|&(b0, b1)| (b0 as f32 * bin_hz, b1 as f32 * bin_hz))
            .collect();

        Self {
            fft: FftPlanner::new().plan_fft_forward(FFT_SIZE),
            window,
            samples: vec![0.0; FFT_SIZE],
            fft_buf: vec![Complex::default(); FFT_SIZE],
            band_bins,
            band_ranges_hz,
            prev_bands: vec![0.0; num_bands],
            ref_db: -30.0,
            bass_history: [0.0; BEAT_HISTORY],
            bass_head: 0,
            since_beat: 1.0,
            beat_armed: true,
            frame: AnalysisFrame::silent(num_bands, WAVEFORM_LEN, SAMPLE_RATE),
        }
    }

    pub fn update(&mut self, audio: &SharedAudio, dt: f32) {
        self.frame.dt_s = dt;
        self.frame.time_s += dt;

        audio.latest(&mut self.samples);

        let rms = (self.samples.iter().map(|s| s * s).sum::<f32>() / FFT_SIZE as f32).sqrt();

        for (i, (&s, &w)) in self.samples.iter().zip(&self.window).enumerate() {
            self.fft_buf[i] = Complex::new(s * w, 0.0);
        }
        self.fft.process(&mut self.fft_buf);

        let bin_hz = SAMPLE_RATE as f32 / FFT_SIZE as f32;
        let mut max_db = f32::MIN;
        let mut raw = vec![0.0f32; self.band_bins.len()];
        for (i, &(b0, b1)) in self.band_bins.iter().enumerate() {
            let mut sum = 0.0;
            for bin in b0..b1 {
                sum += self.fft_buf[bin].norm();
            }
            let mag = sum / (b1 - b0) as f32 / FFT_SIZE as f32;
            // Mild high-frequency tilt: treble carries less energy than bass.
            let center_hz = (b0 + b1) as f32 * 0.5 * bin_hz;
            let db = 20.0 * (mag * (center_hz / 100.0).powf(0.35) + 1e-10).log10();
            raw[i] = db;
            max_db = max_db.max(db);
        }

        // Auto-gain: reference level tracks the loudest band, decaying slowly
        // so quiet passages still fill the display without pinning silence.
        self.ref_db = (self.ref_db - 6.0 * dt).max(max_db).max(-45.0);

        for (i, &db) in raw.iter().enumerate() {
            let v = ((db - (self.ref_db - DB_RANGE)) / DB_RANGE).clamp(0.0, 1.0);
            let cur = self.frame.bands[i];
            self.frame.bands[i] = if v > cur {
                cur + (v - cur) * (1.0 - (-30.0 * dt).exp())
            } else {
                cur + (v - cur) * (1.0 - (-8.0 * dt).exp())
            };

            let peak = self.frame.peaks[i] - 0.35 * dt;
            self.frame.peaks[i] = peak.max(self.frame.bands[i]).max(0.0);
        }

        // Silence gate: with no signal the AGC floor would otherwise
        // amplify noise into a full display.
        if rms < 1e-4 {
            for v in &mut self.frame.bands {
                *v *= (-8.0 * dt).exp();
            }
        }

        let positive_flux = self
            .frame
            .bands
            .iter()
            .zip(&self.prev_bands)
            .map(|(&now, &prev)| (now - prev).max(0.0))
            .sum::<f32>()
            / self.frame.bands.len().max(1) as f32;
        self.prev_bands.clone_from(&self.frame.bands);
        smooth_control(
            &mut self.frame.transient,
            (positive_flux * 4.0).min(1.0),
            35.0,
            10.0,
            dt,
        );

        let groups = grouped_band_energy(&self.frame.bands, &self.band_ranges_hz);
        smooth_control(&mut self.frame.bass, groups.bass, 12.0, 8.0, dt);
        smooth_control(&mut self.frame.low_mid, groups.low_mid, 10.0, 7.0, dt);
        smooth_control(&mut self.frame.high_mid, groups.high_mid, 10.0, 7.0, dt);
        smooth_control(&mut self.frame.treble, groups.treble, 10.0, 7.0, dt);

        // Beat: bass energy spikes above its recent average.
        let bass_now = groups.bass;
        let mean = self.bass_history.iter().sum::<f32>() / BEAT_HISTORY as f32;
        self.bass_history[self.bass_head] = bass_now;
        self.bass_head = (self.bass_head + 1) % BEAT_HISTORY;
        self.since_beat += dt;
        if self.beat_armed && bass_now > mean * 1.35 && bass_now > 0.15 && self.since_beat > 0.2 {
            self.frame.beat.onset = 1.0;
            self.frame.beat.envelope = 1.0;
            self.frame.beat.confidence = ((bass_now - mean) / (mean + 0.05)).clamp(0.0, 1.0);
            self.since_beat = 0.0;
            self.beat_armed = false;
        } else {
            self.frame.beat.onset = 0.0;
            self.frame.beat.envelope *= (-5.0 * dt).exp();
            self.frame.beat.confidence *= (-3.0 * dt).exp();
            if bass_now < mean * 1.1 || bass_now < 0.10 {
                self.beat_armed = true;
            }
        }

        self.frame.rms = (rms * 8.0).min(1.0);
        self.frame.tonal.spectral_centroid =
            spectral_centroid(&self.frame.bands, &self.band_ranges_hz, FREQ_MAX);
        self.fill_waveform(audio);
    }

    /// Copy the latest samples, aligned to a rising zero-crossing so the
    /// oscilloscope doesn't jitter horizontally.
    fn fill_waveform(&mut self, audio: &SharedAudio) {
        let mut buf = vec![0.0f32; WAVEFORM_LEN * 2];
        audio.latest(&mut buf);
        let mut start = 0;
        for i in 1..WAVEFORM_LEN {
            if buf[i - 1] <= 0.0 && buf[i] > 0.0 {
                start = i;
                break;
            }
        }
        self.frame
            .waveform
            .copy_from_slice(&buf[start..start + WAVEFORM_LEN]);
    }
}

fn spectral_centroid(bands: &[f32], ranges_hz: &[(f32, f32)], max_hz: f32) -> Option<f32> {
    let mut weighted = 0.0;
    let mut energy = 0.0;

    for (&band, &(lo, hi)) in bands.iter().zip(ranges_hz) {
        let center = (lo + hi) * 0.5;
        weighted += center * band;
        energy += band;
    }

    (energy > 1e-4).then(|| (weighted / energy / max_hz).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_peaks_in_matching_band() {
        let samples: Vec<f32> = (0..FFT_SIZE * 2)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
            .collect();
        let audio = crate::audio::from_samples(&samples);

        let mut analyzer = Analyzer::new(64);
        for _ in 0..30 {
            analyzer.update(&audio, 1.0 / 60.0);
        }

        let argmax = analyzer
            .frame
            .bands
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        // 440 Hz falls in band ~27 of 64 log-spaced bands over 30..16000 Hz.
        assert!((26..=29).contains(&argmax), "argmax band {argmax}");
        assert!(analyzer.frame.bands[argmax] > 0.5);
        assert!(analyzer.frame.rms > 0.5);
        assert!(analyzer.frame.low_mid > 0.05);
    }

    #[test]
    fn silence_decays_to_zero() {
        let audio = crate::audio::from_samples(&vec![0.0; FFT_SIZE]);
        let mut analyzer = Analyzer::new(32);
        for _ in 0..120 {
            analyzer.update(&audio, 1.0 / 60.0);
        }
        assert!(analyzer.frame.bands.iter().all(|&v| v < 0.05));
        assert!(analyzer.frame.beat.envelope < 0.05);
        assert!(analyzer.frame.transient < 0.05);
        assert_eq!(analyzer.frame.beat.onset, 0.0);
    }

    #[test]
    fn bass_impulse_triggers_one_gated_beat() {
        let samples: Vec<f32> = (0..FFT_SIZE * 2)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let env = if i < FFT_SIZE / 2 { 1.0 } else { 0.25 };
                (std::f32::consts::TAU * 65.0 * t).sin() * env
            })
            .collect();
        let audio = crate::audio::from_samples(&samples);

        let mut analyzer = Analyzer::new(64);
        let mut onsets = 0;
        for _ in 0..90 {
            analyzer.update(&audio, 1.0 / 60.0);
            if analyzer.frame.beat.onset > 0.5 {
                onsets += 1;
            }
        }

        assert_eq!(onsets, 1);
    }
}
