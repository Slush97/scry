// SPDX-License-Identifier: MIT OR Apache-2.0
//! `scry fetch` — trigger the native fetch splash in scry-terminal.
//!
//! When running inside scry-terminal, sends a `ShowFetch` IPC command to
//! activate the GPU-composited fetch overlay. Falls back to plain text
//! output when running in other terminals.

use std::os::unix::net::UnixStream;

use scry_engine::transport::ipc::{self, IpcCommand};

use super::sysinfo_fetch::SysInfo;

/// Show animated system information (add `scry fetch` to .bashrc / .zshrc).
#[derive(Debug, clap::Args)]
pub struct FetchArgs {
    /// Skip native overlay, just print system info as plain text.
    #[arg(long)]
    pub plain: bool,
}

/// Entry point for `scry fetch`.
pub fn run(args: &FetchArgs) -> Result<(), String> {
    // If running inside scry-terminal and not forced to plain, trigger native fetch
    if !args.plain {
        if let Ok(sock_path) = std::env::var("SCRY_TERMINAL_SOCK") {
            return trigger_native_fetch(&sock_path);
        }
    }

    // Fallback: plain text output for non-scry terminals
    let info = SysInfo::collect();
    run_plain(&info);
    Ok(())
}

/// Send a `ShowFetch` IPC command to the terminal.
fn trigger_native_fetch(sock_path: &str) -> Result<(), String> {
    let mut stream = UnixStream::connect(sock_path)
        .map_err(|e| format!("failed to connect to scry-terminal: {e}"))?;

    ipc::send_command_with_fd(&mut stream, &IpcCommand::ShowFetch, None)
        .map_err(|e| format!("failed to send ShowFetch: {e}"))?;

    let _response = ipc::recv_response(&mut stream)
        .map_err(|e| format!("failed to read response: {e}"))?;

    Ok(())
}

/// Plain text fallback (for non-scry terminals).
fn run_plain(info: &SysInfo) {
    println!();
    println!("  {}", info.user_at_host);
    let sep = "─".repeat(info.user_at_host.chars().count() + 2);
    println!("  {sep}");
    for (icon, label, value) in info.rows() {
        println!("  ▎ {icon} {label:<10}  {value}");
    }
    println!();
}
