//! Fuzz target: VTE terminal parser → SecurityGate → TerminalGrid pipeline.
//!
//! Feeds arbitrary bytes through the complete terminal parse chain to
//! exercise escape sequence handling, grid mutation, and security filtering.
//! This is the highest-risk attack surface (raw PTY data → state mutation).

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_terminal::grid::TerminalGrid;
use scry_terminal::security::{ResponsePolicy, SecurityGate};
use scry_terminal::vt::VtHandler;

fuzz_target!(|data: &[u8]| {
    let mut grid = TerminalGrid::new(80, 24, 100);
    let mut security = SecurityGate::new(ResponsePolicy::default());
    let mut parser = vte::Parser::new();

    let mut handler = VtHandler::new(&mut grid, &mut security);
    for &byte in data {
        parser.advance(&mut handler, byte);
    }

    // Success = no panic, no UB, no OOM
});
