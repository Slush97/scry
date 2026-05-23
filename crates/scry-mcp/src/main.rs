// SPDX-License-Identifier: MIT OR Apache-2.0
//! MCP server that renders scry charts and images directly into the host terminal.
//!
//! Tool results return a short text summary to the model; the actual pixels
//! are written to `/dev/tty` so they appear inline in the user's terminal
//! (Kitty, Ghostty, WezTerm, iTerm2, …) rather than being passed through
//! Claude Code's tool-result rendering path.

mod jsonrpc;
mod tools;

use std::io::{self, BufRead, Write};

use serde_json::{json, Value};

use jsonrpc::{Request, Response};

const SERVER_NAME: &str = "scry-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROTOCOL_VERSION: &str = "2024-11-05";

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::error(Value::Null, -32700, format!("parse error: {e}"));
                writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).ok();
                out.flush().ok();
                continue;
            }
        };

        // Notifications (no id) get no response.
        let is_notification = req.id.is_none();
        let id = req.id.clone().unwrap_or(Value::Null);

        let resp = match req.method.as_str() {
            "initialize" => Some(handle_initialize(id)),
            "tools/list" => Some(handle_tools_list(id)),
            "tools/call" => Some(handle_tools_call(id, req.params.unwrap_or(Value::Null))),
            "ping" => Some(Response::result(id, json!({}))),
            _ if is_notification => None,
            _ => Some(Response::error(
                id,
                -32601,
                format!("unknown method: {}", req.method),
            )),
        };

        if let Some(resp) = resp {
            writeln!(out, "{}", serde_json::to_string(&resp).unwrap()).ok();
            out.flush().ok();
        }
    }
}

fn handle_initialize(id: Value) -> Response {
    Response::result(
        id,
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
        }),
    )
}

fn handle_tools_list(id: Value) -> Response {
    Response::result(id, json!({ "tools": tools::descriptors() }))
}

fn handle_tools_call(id: Value, params: Value) -> Response {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(Value::Null);

    match tools::dispatch(name, &args) {
        Ok(content) => Response::result(id, json!({ "content": content, "isError": false })),
        Err(msg) => Response::result(
            id,
            json!({
                "content": [{ "type": "text", "text": msg }],
                "isError": true,
            }),
        ),
    }
}
