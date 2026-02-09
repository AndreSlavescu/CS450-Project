#!/usr/bin/env python3
"""
Local Perfetto trace viewer.

Usage:
    python3 src/csrc/profiler/view_trace.py
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GPU Profiler — Perfetto Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: #1a1a2e;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .toolbar {
    background: #16213e;
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid #0f3460;
    flex-shrink: 0;
  }
  .toolbar h1 {
    font-size: 14px;
    font-weight: 600;
    color: #e94560;
  }
  .toolbar .status {
    font-size: 12px;
    color: #888;
    margin-left: auto;
  }
  .toolbar .status.ok { color: #4ecca3; }
  .toolbar .status.loading { color: #f0c040; }
  .toolbar .status.error { color: #e94560; }
  .toolbar button {
    background: #0f3460;
    color: #e0e0e0;
    border: 1px solid #1a1a4e;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
    font-family: inherit;
  }
  .toolbar button:hover { background: #1a1a4e; }
  .toolbar input[type="file"] { display: none; }
  .toolbar label.btn {
    background: #0f3460;
    color: #e0e0e0;
    border: 1px solid #1a1a4e;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
  }
  .toolbar label.btn:hover { background: #1a1a4e; }
  iframe {
    flex: 1;
    border: none;
    width: 100%;
  }
</style>
</head>
<body>

<div class="toolbar">
  <h1>GPU Profiler</h1>
  <button id="reload-btn">Reload Trace</button>
  <label class="btn">
    Load File
    <input type="file" id="file-input" accept=".json,.pb,.pftrace">
  </label>
  <span class="status loading" id="status">Connecting to Perfetto...</span>
</div>

<iframe id="perfetto" src="https://ui.perfetto.dev/" allow="usb"></iframe>

<script>
const iframe = document.getElementById('perfetto');
const status = document.getElementById('status');
let perfettoReady = false;
let pendingTrace = null;

// Listen for PONG from Perfetto UI — signals it's ready for trace data.
window.addEventListener('message', (e) => {
  if (e.origin !== 'https://ui.perfetto.dev') return;
  if (e.data === 'PONG') {
    perfettoReady = true;
    if (pendingTrace) {
      sendTrace(pendingTrace.buffer, pendingTrace.title);
      pendingTrace = null;
    }
  }
});

// Keep pinging until Perfetto responds.
const pingInterval = setInterval(() => {
  if (perfettoReady) { clearInterval(pingInterval); return; }
  iframe.contentWindow.postMessage('PING', 'https://ui.perfetto.dev');
}, 200);

function sendTrace(buffer, title) {
  iframe.contentWindow.postMessage({
    perfetto: {
      buffer: buffer,
      title: title || 'GPU Profiler Trace',
      keepApiOpen: true,
    }
  }, 'https://ui.perfetto.dev');
  status.textContent = `Loaded: ${title} (${(buffer.byteLength / 1024).toFixed(1)} KB)`;
  status.className = 'status ok';
}

function loadFromServer() {
  status.textContent = 'Loading trace...';
  status.className = 'status loading';
  fetch('/trace.json')
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.arrayBuffer();
    })
    .then(buf => {
      const title = 'TRACE_FILE_NAME';
      if (perfettoReady) {
        sendTrace(buf, title);
      } else {
        pendingTrace = { buffer: buf, title: title };
        status.textContent = 'Trace loaded, waiting for Perfetto...';
      }
    })
    .catch(err => {
      status.textContent = `Error: ${err.message}`;
      status.className = 'status error';
    });
}

// Reload button
document.getElementById('reload-btn').addEventListener('click', loadFromServer);

// Local file picker
document.getElementById('file-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  status.textContent = `Loading ${file.name}...`;
  status.className = 'status loading';
  file.arrayBuffer().then(buf => {
    if (perfettoReady) {
      sendTrace(buf, file.name);
    } else {
      pendingTrace = { buffer: buf, title: file.name };
    }
  });
});

// Auto-load trace from server on page load.
loadFromServer();
</script>
</body>
</html>"""

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACE = SCRIPT_DIR / "traces" / "trace.json"


def make_handler(trace_path: Path):
    trace_name = trace_path.name

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path == "/":
                html = VIEWER_HTML.replace("TRACE_FILE_NAME", trace_name)
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/trace.json":
                if not trace_path.exists():
                    self.send_response(404)
                    self._cors()
                    self.end_headers()
                    self.wfile.write(b"trace file not found")
                    return
                data = trace_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self._cors()
                self.end_headers()
                self.wfile.write(data)

            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser(description="View a Perfetto trace locally")
    parser.add_argument("trace", nargs="?", default=str(DEFAULT_TRACE),
                        help=f"Path to trace JSON file (default: {DEFAULT_TRACE})")
    parser.add_argument("--port", type=int, default=8384,
                        help="Local server port (default: 8384)")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open browser")
    args = parser.parse_args()

    trace_path = Path(args.trace).resolve()
    if not trace_path.exists():
        print(f"Warning: {trace_path} not found (you can still load files via the UI)")

    handler = make_handler(trace_path)
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"Serving trace viewer at {url}")
        print(f"Trace file: {trace_path}")
        print("Press Ctrl+C to stop\n")

        if not args.no_open:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
