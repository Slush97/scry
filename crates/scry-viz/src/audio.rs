//! Audio capture from PulseAudio/PipeWire into a shared sample ring.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use libpulse_binding::def::BufferAttr;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

pub const SAMPLE_RATE: u32 = 44100;
const CHUNK_FRAMES: usize = 512;
const RING_CAPACITY: usize = 1 << 14;

/// Ring of the most recent mono samples, written by the capture thread.
struct Ring {
    buf: Vec<f32>,
    head: usize,
}

impl Ring {
    fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.buf[self.head] = s;
            self.head = (self.head + 1) % RING_CAPACITY;
        }
    }

    /// Copy the newest `out.len()` samples, oldest-first.
    fn latest(&self, out: &mut [f32]) {
        let n = out.len().min(RING_CAPACITY);
        let start = (self.head + RING_CAPACITY - n) % RING_CAPACITY;
        for (i, slot) in out.iter_mut().enumerate().take(n) {
            *slot = self.buf[(start + i) % RING_CAPACITY];
        }
    }
}

#[derive(Clone)]
pub struct SharedAudio {
    ring: Arc<Mutex<Ring>>,
}

impl SharedAudio {
    pub fn latest(&self, out: &mut [f32]) {
        self.ring.lock().unwrap().latest(out);
    }
}

#[cfg(test)]
pub fn from_samples(samples: &[f32]) -> SharedAudio {
    let mut ring = Ring {
        buf: vec![0.0; RING_CAPACITY],
        head: 0,
    };
    ring.push(samples);
    SharedAudio {
        ring: Arc::new(Mutex::new(ring)),
    }
}

fn connect(device: Option<&str>) -> Result<Simple, libpulse_binding::error::PAErr> {
    let spec = Spec {
        format: if cfg!(target_endian = "little") {
            Format::F32le
        } else {
            Format::F32be
        },
        channels: 1,
        rate: SAMPLE_RATE,
    };
    let attr = BufferAttr {
        maxlength: u32::MAX,
        tlength: u32::MAX,
        prebuf: u32::MAX,
        minreq: u32::MAX,
        fragsize: (CHUNK_FRAMES * 4) as u32,
    };
    Simple::new(
        None,
        "scry-viz",
        Direction::Record,
        device,
        "visualizer",
        &spec,
        None,
        Some(&attr),
    )
}

/// Spawn the capture thread. Blocks until the pulse connection either
/// succeeds or fails, so setup errors surface before entering the UI.
pub fn spawn_capture(device: Option<String>) -> Result<SharedAudio, String> {
    let ring = Arc::new(Mutex::new(Ring {
        buf: vec![0.0; RING_CAPACITY],
        head: 0,
    }));
    let shared = SharedAudio { ring: ring.clone() };
    let (status_tx, status_rx) = mpsc::channel();

    thread::spawn(move || {
        // Default: monitor of the default sink (i.e. whatever is playing).
        // Fall back to the default source if the monitor special isn't supported.
        let conn = match device.as_deref() {
            Some(dev) => connect(Some(dev)).map_err(|e| format!("device '{dev}': {e}")),
            None => connect(Some("@DEFAULT_MONITOR@"))
                .or_else(|_| connect(None))
                .map_err(|e| format!("default source: {e}")),
        };

        let pulse = match conn {
            Ok(p) => {
                let _ = status_tx.send(Ok(()));
                p
            }
            Err(e) => {
                let _ = status_tx.send(Err(e));
                return;
            }
        };

        let mut bytes = [0u8; CHUNK_FRAMES * 4];
        let mut samples = [0.0f32; CHUNK_FRAMES];
        loop {
            if pulse.read(&mut bytes).is_err() {
                return;
            }
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                samples[i] = f32::from_ne_bytes(chunk.try_into().unwrap());
            }
            ring.lock().unwrap().push(&samples);
        }
    });

    status_rx
        .recv()
        .map_err(|_| "capture thread died".to_string())?
        .map_err(|e| format!("failed to open audio capture ({e})"))?;
    Ok(shared)
}
