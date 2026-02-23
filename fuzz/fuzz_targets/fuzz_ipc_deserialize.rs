//! Fuzz target: IPC command/response/event deserialization.
//!
//! Exercises `IpcCommand::deserialize`, `IpcResponse::deserialize`, and
//! `IpcEvent::deserialize` with arbitrary payloads. No panics should occur.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_engine::transport::ipc::{IpcCommand, IpcEvent, IpcResponse};

fuzz_target!(|data: &[u8]| {
    // Exercise all three deserializers with raw fuzz data.
    // All should return Ok or Err — never panic.
    let _ = IpcCommand::deserialize(data);
    let _ = IpcResponse::deserialize(data);
    let _ = IpcEvent::deserialize(data);
});
