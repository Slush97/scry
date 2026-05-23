/// Fixed-capacity ring buffer for time-series data.
pub struct RingBuffer {
    buf: Vec<f64>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            capacity,
            head: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, value: f64) {
        self.buf[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Returns data oldest-first as a Vec.
    pub fn as_vec(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.len);
        if self.len < self.capacity {
            out.extend_from_slice(&self.buf[..self.len]);
        } else {
            out.extend_from_slice(&self.buf[self.head..]);
            out.extend_from_slice(&self.buf[..self.head]);
        }
        out
    }
}
