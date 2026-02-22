//! **Native system-info collector** — used by `fastfetch_anim` as a drop-in
//! replacement for the external `fastfetch` binary.
//!
//! All data is gathered from `/proc`, `/etc`, and environment variables.
//! The only subprocess we spawn is a package-manager query (pacman / dpkg /
//! rpm), which falls back to `"?"` gracefully.

#![allow(dead_code)]

use std::fs;
use std::process::Command;

// ─────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────

/// Snapshot of system information.
#[derive(Debug)]
pub struct SysInfo {
    /// `user@hostname`
    pub user_at_host: String,
    /// Pretty OS name, e.g. `"Arch Linux"`
    pub os: String,
    /// Kernel release, e.g. `"6.8.9-arch1-1"`
    pub kernel: String,
    /// Human-readable uptime, e.g. `"2h 34m"`
    pub uptime: String,
    /// Shell basename, e.g. `"zsh"`
    pub shell: String,
    /// Terminal program, e.g. `"scry-terminal"`
    pub terminal: String,
    /// Desktop environment / WM, e.g. `"Hyprland"`
    pub de_wm: String,
    /// Package count string, e.g. `"1342 (pacman)"`
    pub packages: String,
    /// Memory string, e.g. `"3.2 GiB / 31.2 GiB"`
    pub memory: String,
    /// CPU model + core count, e.g. `"AMD Ryzen 9 7950X (32)"`
    pub cpu: String,
}

impl SysInfo {
    /// Collect all fields.  Never panics; missing data becomes `"?"`.
    pub fn collect() -> Self {
        Self {
            user_at_host: user_at_host(),
            os: os_name(),
            kernel: kernel_version(),
            uptime: uptime_human(),
            shell: shell_name(),
            terminal: terminal_name(),
            de_wm: de_wm(),
            packages: packages(),
            memory: memory(),
            cpu: cpu(),
        }
    }

    /// Ordered list of `(icon, label, value)` rows for display.
    pub fn rows(&self) -> Vec<(&str, &str, &str)> {
        vec![
            ("󰀧", "OS",       &self.os),
            ("", "Kernel",   &self.kernel),
            ("󰔟", "Uptime",   &self.uptime),
            ("", "Shell",    &self.shell),
            ("", "Terminal", &self.terminal),
            ("󰖲", "DE / WM",  &self.de_wm),
            ("󰏗", "Packages", &self.packages),
            ("", "Memory",   &self.memory),
            ("󰘚", "CPU",      &self.cpu),
        ]
    }

    /// Convert to a plain vector of display strings (for ratatui).
    /// Format: `"user@host"` on line 0, then `"Key  value"` per row.
    pub fn to_lines(&self) -> Vec<String> {
        let mut out = vec![self.user_at_host.clone()];
        for (_, label, value) in self.rows() {
            out.push(format!("{label}  {value}"));
        }
        out
    }
}

// ─────────────────────────────────────────────
//  Collectors
// ─────────────────────────────────────────────

fn user_at_host() -> String {
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "user".into());
    let host = fs::read_to_string("/proc/sys/kernel/hostname")
        .unwrap_or_else(|_| "localhost\n".into())
        .trim()
        .to_string();
    format!("{user}@{host}")
}

fn os_name() -> String {
    // Try /etc/os-release first
    if let Ok(content) = fs::read_to_string("/etc/os-release") {
        for line in content.lines() {
            if let Some(val) = line.strip_prefix("PRETTY_NAME=") {
                return val.trim_matches('"').to_string();
            }
        }
    }
    // Fallback: /etc/issue
    if let Ok(content) = fs::read_to_string("/etc/issue") {
        let first = content.lines().next().unwrap_or("").trim().to_string();
        if !first.is_empty() {
            return first.replace("\\n", "").replace("\\l", "").trim().to_string();
        }
    }
    "Linux".into()
}

fn kernel_version() -> String {
    // /proc/version is the most portable
    if let Ok(content) = fs::read_to_string("/proc/version") {
        // Format: "Linux version 6.8.9-arch1-1 (build@host) ..."
        if let Some(ver) = content.split_whitespace().nth(2) {
            return ver.to_string();
        }
    }
    "?".into()
}

fn uptime_human() -> String {
    if let Ok(content) = fs::read_to_string("/proc/uptime") {
        if let Some(secs_str) = content.split_whitespace().next() {
            if let Ok(secs_f) = secs_str.parse::<f64>() {
                let total = secs_f as u64;
                let days  = total / 86400;
                let hours = (total % 86400) / 3600;
                let mins  = (total % 3600) / 60;
                return match (days, hours, mins) {
                    (0, 0, m) => format!("{m}m"),
                    (0, h, m) => format!("{h}h {m}m"),
                    (d, h, m) => format!("{d}d {h}h {m}m"),
                };
            }
        }
    }
    "?".into()
}

fn shell_name() -> String {
    std::env::var("SHELL")
        .ok()
        .and_then(|p| p.split('/').last().map(ToString::to_string))
        .unwrap_or_else(|| "?".into())
}

fn terminal_name() -> String {
    // TERM_PROGRAM is set by many modern terminals (kitty, scry-terminal, etc.)
    std::env::var("TERM_PROGRAM")
        .or_else(|_| std::env::var("TERM"))
        .unwrap_or_else(|_| "?".into())
}

fn de_wm() -> String {
    // Wayland compositors
    if let Ok(de) = std::env::var("XDG_CURRENT_DESKTOP") {
        return de;
    }
    if std::env::var("WAYLAND_DISPLAY").is_ok() {
        // Could be bare Wayland without XDG_CURRENT_DESKTOP (e.g. sway)
        if let Ok(name) = std::env::var("SWAYSOCK")
            .map(|_| "sway".to_string())
            .or_else(|_| std::env::var("HYPRLAND_INSTANCE_SIGNATURE").map(|_| "Hyprland".to_string()))
        {
            return name;
        }
        return "Wayland".into();
    }
    if std::env::var("DISPLAY").is_ok() {
        return "X11".into();
    }
    "?".into()
}

fn packages() -> String {
    // Try pacman first (Arch)
    if let Ok(out) = Command::new("pacman").args(["-Q", "--quiet"]).output() {
        if out.status.success() {
            let count = String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.is_empty())
                .count();
            return format!("{count} (pacman)");
        }
    }
    // dpkg (Debian / Ubuntu)
    if let Ok(out) = Command::new("dpkg-query")
        .args(["-f", ".\n", "-W"])
        .output()
    {
        if out.status.success() {
            let count = String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.is_empty())
                .count();
            return format!("{count} (dpkg)");
        }
    }
    // rpm (Fedora / RHEL)
    if let Ok(out) = Command::new("rpm").args(["-qa"]).output() {
        if out.status.success() {
            let count = String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.is_empty())
                .count();
            return format!("{count} (rpm)");
        }
    }
    "?".into()
}

fn memory() -> String {
    let content = match fs::read_to_string("/proc/meminfo") {
        Ok(c) => c,
        Err(_) => return "?".into(),
    };

    let mut total_kb: u64 = 0;
    let mut avail_kb: u64 = 0;

    for line in content.lines() {
        if let Some(val) = parse_meminfo_line(line, "MemTotal:") {
            total_kb = val;
        } else if let Some(val) = parse_meminfo_line(line, "MemAvailable:") {
            avail_kb = val;
        }
    }

    if total_kb == 0 {
        return "?".into();
    }

    let used_kb = total_kb.saturating_sub(avail_kb);
    format!("{} / {}", gib(used_kb), gib(total_kb))
}

fn parse_meminfo_line(line: &str, key: &str) -> Option<u64> {
    let rest = line.strip_prefix(key)?.trim();
    rest.split_whitespace().next()?.parse().ok()
}

fn gib(kb: u64) -> String {
    let gib = kb as f64 / 1_048_576.0; // 1024^2 kB → GiB
    format!("{gib:.1} GiB")
}

fn cpu() -> String {
    let content = match fs::read_to_string("/proc/cpuinfo") {
        Ok(c) => c,
        Err(_) => return "?".into(),
    };

    let mut model = String::new();
    let mut count: usize = 0;

    for line in content.lines() {
        if line.starts_with("model name") && model.is_empty() {
            if let Some(val) = line.splitn(2, ':').nth(1) {
                model = val.trim().to_string();
            }
        }
        if line.starts_with("processor") {
            count += 1;
        }
    }

    // Clean up verbose Intel/AMD suffixes
    let model = model
        .replace("(R)", "")
        .replace("(TM)", "")
        .replace("  ", " ");
    let model = model.trim();

    if model.is_empty() {
        return "?".into();
    }

    format!("{model} ({count})")
}
