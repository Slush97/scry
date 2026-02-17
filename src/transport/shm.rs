// SPDX-License-Identifier: MIT OR Apache-2.0
//! POSIX shared memory buffer for zero-copy Kitty protocol transmission.
//!
//! When the terminal and the application share a filesystem (the common
//! local case), pixel data can be placed in a POSIX shared memory object
//! (`shm_open`) and the terminal reads it directly — bypassing base64
//! encoding and stdout I/O entirely.
//!
//! # Safety
//!
//! This module uses `unsafe` for the POSIX `shm_open`, `ftruncate`, `mmap`,
//! `munmap`, and `shm_unlink` FFI calls. The public API is safe: [`ShmBuffer`]
//! owns the mapping and cleans up on drop.
//!
//! # Feature gate
//!
//! This module is only available with the `shm` feature (`--features shm`).

#![allow(unsafe_code)]

use std::ffi::CString;
use std::io;

/// A POSIX shared memory buffer suitable for Kitty `t=s` transmission.
///
/// The buffer is created via `shm_open`, sized with `ftruncate`, and
/// memory-mapped for fast writes. The Kitty protocol reads directly
/// from this object, so no base64 or pipe I/O is needed.
///
/// On drop, the mapping is unmapped and the shm object is unlinked.
#[allow(clippy::redundant_pub_crate)]
pub(crate) struct ShmBuffer {
    /// Name of the shared memory object (without leading `/`).
    name: String,
    /// C-compatible name for POSIX API calls.
    c_name: CString,
    /// Pointer to the memory-mapped region.
    ptr: *mut u8,
    /// Total size of the mapping in bytes.
    capacity: usize,
}

impl std::fmt::Debug for ShmBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShmBuffer")
            .field("name", &self.name)
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

// SAFETY: The pointer is exclusively owned by this struct and never
// aliased. The mmap region is process-local.
#[allow(clippy::non_send_fields_in_send_ty)]
// ShmBuffer owns its mapping exclusively and is safe to move between threads.
unsafe impl Send for ShmBuffer {}

impl ShmBuffer {
    /// Create a new shared memory buffer with the given capacity.
    ///
    /// The `name` should be a unique identifier (e.g. `scry-12345`).
    /// It will be prefixed with `/` for the POSIX `shm_open` call.
    pub(crate) fn new(name: &str, capacity: usize) -> io::Result<Self> {
        let c_name = CString::new(format!("/{name}"))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        // SAFETY: shm_open creates or opens a POSIX shared memory object.
        // O_CREAT | O_RDWR, mode 0o600 (owner read/write only).
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        // Size the object
        // SAFETY: ftruncate sets the size of the shared memory object.
        #[allow(clippy::cast_possible_wrap)]
        let ret = unsafe { libc::ftruncate(fd, capacity as libc::off_t) };
        if ret < 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // Map it
        // SAFETY: mmap maps the shared memory into the process address space.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        // Close fd — the mapping keeps the object alive
        unsafe { libc::close(fd) };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            name: name.to_string(),
            c_name,
            ptr: ptr.cast::<u8>(),
            capacity,
        })
    }

    /// Write data into the shared memory buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() > self.capacity`.
    pub(crate) fn write(&self, data: &[u8]) {
        assert!(
            !self.ptr.is_null(),
            "ShmBuffer::write: buffer has been invalidated (null pointer)",
        );
        assert!(
            data.len() <= self.capacity,
            "ShmBuffer::write: data ({}) exceeds capacity ({})",
            data.len(),
            self.capacity,
        );
        // SAFETY: ptr is non-null (checked above), valid, and exclusively
        // owned. data.len() <= capacity (checked above), so the write is
        // within bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
    }

    /// The shared memory object name (without leading `/`), suitable
    /// for the Kitty protocol escape sequence.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Total capacity in bytes.
    pub(crate) const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Resize the shared memory buffer if `new_capacity` differs.
    ///
    /// This unmaps the old region, resizes with `ftruncate`, and remaps.
    pub(crate) fn resize(&mut self, new_capacity: usize) -> io::Result<()> {
        if new_capacity == self.capacity {
            return Ok(());
        }

        // Unmap current
        // SAFETY: ptr and capacity are valid from our mmap.
        unsafe { libc::munmap(self.ptr.cast(), self.capacity) };

        // Immediately invalidate so that if re-mmap fails and the caller
        // drops us, we won't munmap a stale/dangling pointer.
        self.ptr = std::ptr::null_mut();
        self.capacity = 0;

        // Reopen, resize, remap
        let fd = unsafe { libc::shm_open(self.c_name.as_ptr(), libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        #[allow(clippy::cast_possible_wrap)]
        let ret = unsafe { libc::ftruncate(fd, new_capacity as libc::off_t) };
        if ret < 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                new_capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        unsafe { libc::close(fd) };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        self.ptr = ptr.cast::<u8>();
        self.capacity = new_capacity;
        Ok(())
    }
}

impl Drop for ShmBuffer {
    fn drop(&mut self) {
        // SAFETY: Only munmap if we still hold a valid mapping.
        // ptr can be null if a resize() failed between munmap and re-mmap.
        unsafe {
            if !self.ptr.is_null() {
                libc::munmap(self.ptr.cast(), self.capacity);
            }
            libc::shm_unlink(self.c_name.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_write_read_cleanup() {
        let name = format!("scry-test-{}", std::process::id());
        let buf = ShmBuffer::new(&name, 1024).unwrap();
        assert_eq!(buf.capacity(), 1024);

        let data = vec![42u8; 512];
        buf.write(&data);

        // Verify the data was written
        let slice = unsafe { std::slice::from_raw_parts(buf.ptr, 512) };
        assert!(slice.iter().all(|&b| b == 42));

        drop(buf);

        // After drop, shm_open should fail (object was unlinked)
        let c_name = CString::new(format!("/{name}")).unwrap();
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDONLY, 0) };
        assert!(fd < 0, "shm object should have been unlinked on drop");
    }

    #[test]
    fn resize_grows_buffer() {
        let name = format!("scry-resize-{}", std::process::id());
        let mut buf = ShmBuffer::new(&name, 256).unwrap();
        assert_eq!(buf.capacity(), 256);

        buf.resize(1024).unwrap();
        assert_eq!(buf.capacity(), 1024);

        // Can write to full new capacity
        let data = vec![0xAB_u8; 1024];
        buf.write(&data);
    }
}
