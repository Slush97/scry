// SPDX-License-Identifier: MIT OR Apache-2.0
//! Grow-only GPU buffer pool with reallocation tracking.
//!
//! [`BufferPool`] centralizes buffer management for GPU rendering contexts
//! (SDF compute, 2D rasterizer).  All buffers use grow-only allocation:
//! a buffer is only reallocated when the requested size exceeds the current
//! capacity, never when it shrinks.  This eliminates per-frame churn
//! during terminal resize events.
//!
//! # Reallocation Tracking
//!
//! Each call to [`get_or_grow`](BufferPool::get_or_grow) returns whether
//! the buffer was reallocated.  Callers use this to decide whether
//! bind groups need to be rebuilt.

use std::collections::HashMap;

// ── Buffer keys ────────────────────────────────────────────────────

/// Identifies a buffer slot in the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferKey {
    /// SDF compute output (`STORAGE` | `COPY_SRC`).
    SdfOutput,
    /// SDF readback staging (`MAP_READ` | `COPY_DST`).
    SdfReadback,
    /// SDF uniforms (`UNIFORM` | `COPY_DST`).
    SdfUniforms,
    /// SDF scene objects (`STORAGE` | `COPY_DST`).
    SdfObjects,
    /// SDF scene lights (`STORAGE` | `COPY_DST`).
    SdfLights,
    /// SDF glyph metadata (`STORAGE` | `COPY_DST`).
    SdfGlyphMeta,
    /// SDF glyph grids (`STORAGE` | `COPY_DST`).
    SdfGlyphGrids,
    /// 2D shape instance buffer (`STORAGE` | `COPY_DST`).
    RasterShapes,
    /// 2D line vertex buffer (`STORAGE` | `COPY_DST`).
    RasterLines,
    /// 2D viewport uniforms (`UNIFORM` | `COPY_DST`).
    RasterUniforms,
    /// 2D readback staging (`MAP_READ` | `COPY_DST`).
    RasterReadback,
}

// ── Pooled buffer entry ────────────────────────────────────────────

struct PooledBuffer {
    buffer: wgpu::Buffer,
    /// Current allocated capacity in bytes (only grows).
    capacity: u64,
}

// ── Buffer pool ────────────────────────────────────────────────────

/// Grow-only buffer pool.
///
/// Buffers are created on first use and only reallocated when a larger
/// size is requested.  Call [`invalidate_all`] after a device-lost event
/// to drop all buffers and start fresh.
pub struct BufferPool {
    buffers: HashMap<BufferKey, PooledBuffer>,
}

impl BufferPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Get or create a buffer for `key`.
    ///
    /// - If the buffer exists and `capacity >= needed_bytes`: returns the
    ///   existing buffer, `reallocated = false`.
    /// - Otherwise: creates a new buffer with `needed_bytes` capacity,
    ///   `reallocated = true`.
    ///
    /// # Panics
    ///
    /// Panics if `needed_bytes` is 0.
    pub fn get_or_grow(
        &mut self,
        key: BufferKey,
        needed_bytes: u64,
        usage: wgpu::BufferUsages,
        device: &wgpu::Device,
        label: &str,
    ) -> (&wgpu::Buffer, bool) {
        debug_assert!(needed_bytes > 0, "buffer size must be > 0");

        let entry = self.buffers.get(&key);
        let needs_realloc = entry.is_none() || entry.is_some_and(|e| e.capacity < needed_bytes);

        if needs_realloc {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: needed_bytes,
                usage,
                mapped_at_creation: false,
            });
            self.buffers.insert(
                key,
                PooledBuffer {
                    buffer,
                    capacity: needed_bytes,
                },
            );
            let entry = self.buffers.get(&key).unwrap();
            (&entry.buffer, true)
        } else {
            let entry = self.buffers.get(&key).unwrap();
            (&entry.buffer, false)
        }
    }

    /// Get a buffer reference without potentially growing.
    ///
    /// Returns `None` if the buffer hasn't been created yet.
    pub fn get(&self, key: BufferKey) -> Option<&wgpu::Buffer> {
        self.buffers.get(&key).map(|e| &e.buffer)
    }

    /// Drop all buffers (e.g. after device-lost recovery).
    pub fn invalidate_all(&mut self) {
        self.buffers.clear();
    }

    /// Number of buffers currently in the pool.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.buffers.len()
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("num_buffers", &self.buffers.len())
            .finish()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a test device (requires GPU feature)
    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        ))
        .ok()?;
        Some((device, queue))
    }

    #[test]
    fn pool_creates_on_first_access() {
        let Some((device, _queue)) = test_device() else {
            eprintln!("skipping test: no GPU available");
            return;
        };
        let mut pool = BufferPool::new();
        let (_, reallocated) = pool.get_or_grow(
            BufferKey::SdfOutput,
            1024,
            wgpu::BufferUsages::STORAGE,
            &device,
            "test",
        );
        assert!(reallocated);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn pool_reuses_when_large_enough() {
        let Some((device, _queue)) = test_device() else {
            eprintln!("skipping test: no GPU available");
            return;
        };
        let mut pool = BufferPool::new();
        // First allocation: 1024 bytes
        pool.get_or_grow(
            BufferKey::SdfOutput,
            1024,
            wgpu::BufferUsages::STORAGE,
            &device,
            "test",
        );
        // Second request: 512 bytes (fits in existing 1024)
        let (_, reallocated) = pool.get_or_grow(
            BufferKey::SdfOutput,
            512,
            wgpu::BufferUsages::STORAGE,
            &device,
            "test",
        );
        assert!(!reallocated);
    }

    #[test]
    fn pool_grows_when_too_small() {
        let Some((device, _queue)) = test_device() else {
            eprintln!("skipping test: no GPU available");
            return;
        };
        let mut pool = BufferPool::new();
        pool.get_or_grow(
            BufferKey::SdfOutput,
            512,
            wgpu::BufferUsages::STORAGE,
            &device,
            "test",
        );
        let (_, reallocated) = pool.get_or_grow(
            BufferKey::SdfOutput,
            1024,
            wgpu::BufferUsages::STORAGE,
            &device,
            "test",
        );
        assert!(reallocated);
    }

    #[test]
    fn pool_never_shrinks() {
        let Some((device, _queue)) = test_device() else {
            eprintln!("skipping test: no GPU available");
            return;
        };
        let mut pool = BufferPool::new();
        pool.get_or_grow(
            BufferKey::SdfOutput,
            1024,
            wgpu::BufferUsages::STORAGE,
            &device,
            "test",
        );
        // Request smaller — should reuse the 1024 buffer
        let (_, reallocated) = pool.get_or_grow(
            BufferKey::SdfOutput,
            256,
            wgpu::BufferUsages::STORAGE,
            &device,
            "test",
        );
        assert!(!reallocated, "should not shrink");
    }

    #[test]
    fn invalidate_clears_all() {
        let Some((device, _queue)) = test_device() else {
            eprintln!("skipping test: no GPU available");
            return;
        };
        let mut pool = BufferPool::new();
        pool.get_or_grow(
            BufferKey::SdfOutput,
            512,
            wgpu::BufferUsages::STORAGE,
            &device,
            "out",
        );
        pool.get_or_grow(
            BufferKey::SdfObjects,
            256,
            wgpu::BufferUsages::STORAGE,
            &device,
            "obj",
        );
        pool.get_or_grow(
            BufferKey::SdfLights,
            128,
            wgpu::BufferUsages::STORAGE,
            &device,
            "lit",
        );
        assert_eq!(pool.len(), 3);
        pool.invalidate_all();
        assert_eq!(pool.len(), 0);
        // Next get should create fresh
        let (_, reallocated) = pool.get_or_grow(
            BufferKey::SdfOutput,
            512,
            wgpu::BufferUsages::STORAGE,
            &device,
            "out",
        );
        assert!(reallocated);
    }
}
