use ratatui::layout::{Constraint, Layout};
use ratatui::Frame;

use crate::data::SystemSnapshot;
use crate::panel::cpu::CpuPanel;
use crate::panel::disk::DiskPanel;
use crate::panel::header::HeaderPanel;
use crate::panel::memory::MemoryPanel;
use crate::panel::network::NetworkPanel;
use crate::panel::process::ProcessPanel;
use crate::panel::Panel;

pub struct App {
    header: HeaderPanel,
    cpu: CpuPanel,
    memory: MemoryPanel,
    network: NetworkPanel,
    disk: DiskPanel,
    process: ProcessPanel,
    pub paused: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            header: HeaderPanel::new(),
            cpu: CpuPanel::new(),
            memory: MemoryPanel::new(),
            network: NetworkPanel::new(),
            disk: DiskPanel::new(),
            process: ProcessPanel::new(),
            paused: false,
        }
    }

    pub fn update(&mut self, snap: &SystemSnapshot) {
        if self.paused {
            return;
        }
        self.header.update(snap);
        self.cpu.update(snap);
        self.memory.update(snap);
        self.network.update(snap);
        self.disk.update(snap);
        self.process.update(snap);
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let rows = Layout::vertical([
            Constraint::Length(1),      // header
            Constraint::Percentage(30), // cpu + memory
            Constraint::Percentage(25), // network + disk
            Constraint::Min(6),         // process table
        ])
        .split(frame.area());

        // Header
        self.header.render(frame, rows[0]);

        // CPU | Memory
        let mid_top = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(rows[1]);
        self.cpu.render(frame, mid_top[0]);
        self.memory.render(frame, mid_top[1]);

        // Network | Disk
        let mid_bot = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(rows[2]);
        self.network.render(frame, mid_bot[0]);
        self.disk.render(frame, mid_bot[1]);

        // Process table
        self.process.render(frame, rows[3]);
    }

    pub fn flush_all(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.cpu.flush()?;
        self.memory.flush()?;
        self.network.flush()?;
        self.disk.flush()?;
        Ok(())
    }

    pub fn cleanup(&mut self) {
        self.cpu.cleanup();
        self.memory.cleanup();
        self.network.cleanup();
        self.disk.cleanup();
    }

    pub fn cycle_sort(&mut self) {
        self.process.cycle_sort();
    }
}
