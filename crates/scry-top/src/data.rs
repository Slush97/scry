use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use sysinfo::{Disks, Networks, System};

pub struct DiskInfo {
    pub name: String,
    pub used_pct: f64,
}

pub struct NetInfo {
    pub rx_bytes_sec: f64,
    pub tx_bytes_sec: f64,
}

pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub cpu_pct: f32,
    pub mem_pct: f32,
    pub status: String,
}

pub struct SystemSnapshot {
    pub cpu_per_core: Vec<f32>,
    pub cpu_global: f32,
    pub mem_used: u64,
    pub mem_total: u64,
    pub swap_used: u64,
    pub swap_total: u64,
    pub disks: Vec<DiskInfo>,
    pub networks: Vec<NetInfo>,
    pub processes: Vec<ProcessInfo>,
    pub load_avg: [f64; 3],
    pub uptime_secs: u64,
    pub hostname: String,
}

/// Spawns a background thread that sends `SystemSnapshot` at ~1 Hz.
/// Returns the receiver. The thread exits when the receiver is dropped.
pub fn spawn_poller() -> mpsc::Receiver<SystemSnapshot> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let mut sys = System::new_all();
        let mut networks = Networks::new_with_refreshed_list();
        let disks = Disks::new_with_refreshed_list();
        let hostname = System::host_name().unwrap_or_default();

        // Initial refresh to get baseline CPU readings
        sys.refresh_all();
        thread::sleep(Duration::from_millis(500));

        // Track previous network bytes for rate computation
        let mut prev_net: Vec<(String, u64, u64)> = Vec::new();

        loop {
            sys.refresh_all();
            networks.refresh(true);

            // CPU
            let cpu_per_core: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
            let cpu_global = sys.global_cpu_usage();

            // Memory
            let mem_used = sys.used_memory();
            let mem_total = sys.total_memory();
            let swap_used = sys.used_swap();
            let swap_total = sys.total_swap();

            // Disks
            let disk_info: Vec<DiskInfo> = disks
                .iter()
                .filter(|d| d.total_space() > 0)
                .map(|d| {
                    let used = d.total_space() - d.available_space();
                    let pct = used as f64 / d.total_space() as f64 * 100.0;
                    DiskInfo {
                        name: d.mount_point().to_string_lossy().into_owned(),
                        used_pct: pct,
                    }
                })
                .collect();

            // Networks — compute rates
            let mut net_info = Vec::new();
            let mut new_prev = Vec::new();
            for (name, data) in &networks {
                let rx = data.total_received();
                let tx_total = data.total_transmitted();
                let (rx_rate, tx_rate) = prev_net
                    .iter()
                    .find(|(n, _, _)| n == name)
                    .map(|(_, prev_rx, prev_tx)| {
                        (
                            (rx.saturating_sub(*prev_rx)) as f64,
                            (tx_total.saturating_sub(*prev_tx)) as f64,
                        )
                    })
                    .unwrap_or((0.0, 0.0));
                new_prev.push((name.clone(), rx, tx_total));
                if rx_rate > 0.0 || tx_rate > 0.0 || !net_info.is_empty() {
                    net_info.push(NetInfo {
                        rx_bytes_sec: rx_rate,
                        tx_bytes_sec: tx_rate,
                    });
                }
            }
            prev_net = new_prev;

            // Processes — top 50 by CPU
            let mut processes: Vec<ProcessInfo> = sys
                .processes()
                .values()
                .map(|p| ProcessInfo {
                    pid: p.pid().as_u32(),
                    name: p.name().to_string_lossy().into_owned(),
                    cpu_pct: p.cpu_usage(),
                    mem_pct: if mem_total > 0 {
                        p.memory() as f32 / mem_total as f32 * 100.0
                    } else {
                        0.0
                    },
                    status: format!("{:?}", p.status()),
                })
                .collect();
            processes.sort_by(|a, b| {
                b.cpu_pct
                    .partial_cmp(&a.cpu_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            processes.truncate(100);

            // Load average
            let la = System::load_average();
            let load_avg = [la.one, la.five, la.fifteen];

            let snap = SystemSnapshot {
                cpu_per_core,
                cpu_global,
                mem_used,
                mem_total,
                swap_used,
                swap_total,
                disks: disk_info,
                networks: net_info,
                processes,
                load_avg,
                uptime_secs: System::uptime(),
                hostname: hostname.clone(),
            };

            if tx.send(snap).is_err() {
                break; // Receiver dropped — exit thread
            }

            thread::sleep(Duration::from_secs(1));
        }
    });

    rx
}
