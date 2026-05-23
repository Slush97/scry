//! Tool definitions and dispatch.
//!
//! Each handler renders to a temporary PNG via the `scry` CLI's `--output`
//! flag, then returns the bytes as MCP `image` content. Claude Code
//! attaches that inline in the conversation, the same way it shows
//! user-uploaded images. No `/dev/tty` painting, no fighting with the TUI.

use std::path::PathBuf;
use std::process::Command;

use base64::Engine;
use serde_json::{json, Value};

pub fn descriptors() -> Value {
    json!([
        {
            "name": "render_chart",
            "description":
                "Render a chart from a JSON spec to a PNG and return it as inline \
                 image content. The image is shown in the conversation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "string",
                        "description":
                            "JSON chart spec. Minimum: {\"type\":\"line\",\"data\":{\"y\":[1,2,3]}}. \
                             Supported types: line, scatter, bar, histogram, boxplot, heatmap, pie, \
                             radar, candlestick, bubble, violin, sparkline, waterfall, funnel, \
                             gauge, lollipop, gantt."
                    },
                    "width":  { "type": "integer", "minimum": 64, "maximum": 4096 },
                    "height": { "type": "integer", "minimum": 64, "maximum": 4096 },
                    "theme":  { "type": "string", "enum": ["dark", "light", "pastel", "ocean", "forest", "colorblind"] }
                },
                "required": ["spec"]
            }
        },
        {
            "name": "render_example_chart",
            "description":
                "Render a built-in example chart (no data required) to a PNG and \
                 return it as inline image content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["line","scatter","bar","histogram","boxplot","heatmap","pie",
                                 "radar","candlestick","bubble","violin","sparkline","waterfall",
                                 "funnel","gauge","lollipop","gantt"]
                    },
                    "width":  { "type": "integer", "minimum": 64, "maximum": 4096 },
                    "height": { "type": "integer", "minimum": 64, "maximum": 4096 },
                    "theme":  { "type": "string" }
                }
            }
        },
        {
            "name": "render_image",
            "description":
                "Read an image file (PNG/JPEG) from disk and return it as inline \
                 image content in the conversation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path to the image file." }
                },
                "required": ["path"]
            }
        }
    ])
}

pub fn dispatch(name: &str, args: &Value) -> Result<Value, String> {
    match name {
        "render_chart" => render_chart(args),
        "render_example_chart" => render_example_chart(args),
        "render_image" => render_image(args),
        other => Err(format!("unknown tool: {other}")),
    }
}

fn scry_bin() -> String {
    std::env::var("SCRY_BIN").unwrap_or_else(|_| "scry".to_string())
}

fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join("scry-mcp");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn cached_png_path(stem: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    cache_dir().join(format!("{stem}-{nanos}.png"))
}

fn run(mut cmd: Command, stdin_bytes: Option<&[u8]>, label: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::Stdio;

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    if stdin_bytes.is_some() {
        cmd.stdin(Stdio::piped());
    }
    let mut child = cmd.spawn().map_err(|e| format!("spawn scry: {e}"))?;
    if let (Some(bytes), Some(mut stdin)) = (stdin_bytes, child.stdin.take()) {
        stdin
            .write_all(bytes)
            .map_err(|e| format!("write stdin: {e}"))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("wait scry: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{label} failed: {stderr}"));
    }
    Ok(())
}

fn open_in_viewer(path: &std::path::Path) -> Option<String> {
    let viewer = std::env::var("SCRY_VIEWER").unwrap_or_else(|_| "xdg-open".to_string());
    Command::new(&viewer)
        .arg(path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .stdin(std::process::Stdio::null())
        .spawn()
        .ok()
        .map(|_| viewer)
}

fn png_response(path: &std::path::Path) -> Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let viewer_msg = match open_in_viewer(path) {
        Some(v) => format!("Saved: {}\nOpened with: {v}", path.display()),
        None => format!(
            "Saved: {} (open it manually — set SCRY_VIEWER to override)",
            path.display()
        ),
    };
    Ok(json!([
        { "type": "text",  "text": viewer_msg },
        { "type": "image", "data": b64, "mimeType": "image/png" },
    ]))
}

fn render_chart(args: &Value) -> Result<Value, String> {
    let spec = args
        .get("spec")
        .and_then(Value::as_str)
        .ok_or("missing 'spec'")?;
    let out = cached_png_path("chart");

    let mut cmd = Command::new(scry_bin());
    cmd.args(["chart", "render", "-o"]).arg(&out);
    if let Some(w) = args.get("width").and_then(Value::as_u64) {
        cmd.args(["-W", &w.to_string()]);
    }
    if let Some(h) = args.get("height").and_then(Value::as_u64) {
        cmd.args(["-H", &h.to_string()]);
    }
    if let Some(theme) = args.get("theme").and_then(Value::as_str) {
        cmd.args(["--theme", theme]);
    }
    run(cmd, Some(spec.as_bytes()), "render_chart")?;
    png_response(&out)
}

fn render_example_chart(args: &Value) -> Result<Value, String> {
    let stem = args
        .get("chart_type")
        .and_then(Value::as_str)
        .unwrap_or("example");
    let out = cached_png_path(stem);
    let mut cmd = Command::new(scry_bin());
    cmd.args(["chart", "example"]);
    if let Some(t) = args.get("chart_type").and_then(Value::as_str) {
        cmd.arg(t);
    }
    cmd.arg("-o").arg(&out);
    if let Some(w) = args.get("width").and_then(Value::as_u64) {
        cmd.args(["-W", &w.to_string()]);
    }
    if let Some(h) = args.get("height").and_then(Value::as_u64) {
        cmd.args(["-H", &h.to_string()]);
    }
    if let Some(theme) = args.get("theme").and_then(Value::as_str) {
        cmd.args(["--theme", theme]);
    }
    run(cmd, None, "render_example_chart")?;
    png_response(&out)
}

fn render_image(args: &Value) -> Result<Value, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or("missing 'path'")?;
    png_response(std::path::Path::new(path))
}
