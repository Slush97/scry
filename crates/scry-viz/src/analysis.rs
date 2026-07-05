//! Stable visual-facing audio analysis contract.

/// Frequency groups used by visual modes for broad musical roles.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BandGroups {
    pub bass: f32,
    pub low_mid: f32,
    pub high_mid: f32,
    pub treble: f32,
}

/// Beat and tempo state. Live analysis may leave tempo fields unset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeatState {
    /// One-frame beat onset impulse, normalized 0..1.
    pub onset: f32,
    /// Decaying beat envelope, normalized 0..1.
    pub envelope: f32,
    /// Confidence that the current onset/envelope is musically meaningful.
    pub confidence: f32,
    pub bpm: Option<f32>,
    pub phase: Option<f32>,
    pub bar_phase: Option<f32>,
}

/// Tonal/timbre state. Live mode fills this opportunistically.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TonalState {
    pub spectral_centroid: Option<f32>,
    pub spectral_rolloff: Option<f32>,
    pub chroma: Option<[f32; 12]>,
    pub key: Option<String>,
}

/// One frame of analysis results consumed by visualizers.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisFrame {
    pub time_s: f32,
    pub dt_s: f32,
    pub sample_rate: u32,
    /// Smoothed per-band levels, 0..1, low to high frequency.
    pub bands: Vec<f32>,
    /// Falling peak markers per band, 0..1.
    pub peaks: Vec<f32>,
    /// Recent mono waveform, zero-crossing aligned, roughly -1..1.
    pub waveform: Vec<f32>,
    /// Overall loudness, 0..1.
    pub rms: f32,
    /// Smoothed low-frequency level, 0..1.
    pub bass: f32,
    /// Smoothed low-mid energy, 0..1.
    pub low_mid: f32,
    /// Smoothed high-mid energy, 0..1.
    pub high_mid: f32,
    /// Smoothed treble energy, 0..1.
    pub treble: f32,
    /// Short-term positive spectrum change, 0..1.
    pub transient: f32,
    pub beat: BeatState,
    pub tonal: TonalState,
}

impl AnalysisFrame {
    pub fn silent(num_bands: usize, waveform_len: usize, sample_rate: u32) -> Self {
        Self {
            time_s: 0.0,
            dt_s: 0.0,
            sample_rate,
            bands: vec![0.0; num_bands],
            peaks: vec![0.0; num_bands],
            waveform: vec![0.0; waveform_len],
            rms: 0.0,
            bass: 0.0,
            low_mid: 0.0,
            high_mid: 0.0,
            treble: 0.0,
            transient: 0.0,
            beat: BeatState::default(),
            tonal: TonalState::default(),
        }
    }

    #[cfg(test)]
    pub fn fixture(num_bands: usize, waveform_len: usize, sample_rate: u32) -> Self {
        let mut frame = Self::silent(num_bands, waveform_len, sample_rate);
        frame.dt_s = 1.0 / 60.0;
        frame.rms = 0.45;
        frame.bass = 0.60;
        frame.low_mid = 0.42;
        frame.high_mid = 0.36;
        frame.treble = 0.50;
        frame.transient = 0.35;
        frame.beat = BeatState {
            onset: 1.0,
            envelope: 0.80,
            confidence: 0.75,
            bpm: None,
            phase: None,
            bar_phase: None,
        };

        for (i, band) in frame.bands.iter_mut().enumerate() {
            let t = i as f32 / (num_bands.saturating_sub(1).max(1)) as f32;
            *band = (0.18 + 0.72 * (std::f32::consts::TAU * t).sin().abs()).clamp(0.0, 1.0);
        }
        frame.peaks.clone_from(&frame.bands);
        for peak in &mut frame.peaks {
            *peak = (*peak + 0.12).min(1.0);
        }
        for (i, sample) in frame.waveform.iter_mut().enumerate() {
            *sample = (std::f32::consts::TAU * i as f32 / 96.0).sin() * 0.55;
        }
        frame
    }
}

/// Average normalized band values into the broad musical roles used by live
/// visuals. `band_ranges_hz` must line up one-to-one with `bands`.
pub fn grouped_band_energy(bands: &[f32], band_ranges_hz: &[(f32, f32)]) -> BandGroups {
    BandGroups {
        bass: band_energy_in_range(bands, band_ranges_hz, 20.0, 150.0),
        low_mid: band_energy_in_range(bands, band_ranges_hz, 150.0, 700.0),
        high_mid: band_energy_in_range(bands, band_ranges_hz, 700.0, 3500.0),
        treble: band_energy_in_range(bands, band_ranges_hz, 3500.0, 16_000.0),
    }
}

pub fn band_energy_in_range(
    bands: &[f32],
    band_ranges_hz: &[(f32, f32)],
    min_hz: f32,
    max_hz: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut count = 0usize;

    for (&band, &(start_hz, end_hz)) in bands.iter().zip(band_ranges_hz) {
        if end_hz > min_hz && start_hz < max_hz {
            sum += band;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        (sum / count as f32).clamp(0.0, 1.0)
    }
}

pub fn smooth_control(current: &mut f32, target: f32, attack: f32, release: f32, dt: f32) {
    let rate = if target > *current { attack } else { release };
    *current += (target - *current) * (1.0 - (-rate * dt).exp());
    *current = (*current).clamp(0.0, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_energy_uses_matching_frequency_ranges() {
        let bands = [0.8, 0.4, 0.2, 0.6];
        let ranges = [
            (30.0, 120.0),
            (180.0, 500.0),
            (900.0, 2000.0),
            (5000.0, 9000.0),
        ];

        let groups = grouped_band_energy(&bands, &ranges);

        assert_eq!(groups.bass, 0.8);
        assert_eq!(groups.low_mid, 0.4);
        assert_eq!(groups.high_mid, 0.2);
        assert_eq!(groups.treble, 0.6);
    }
}
