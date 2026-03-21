#!/usr/bin/env python3
import http.client
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("LOG_DIR", ROOT / "logs"))
PID_DIR = Path(os.environ.get("PID_DIR", ROOT / ".pids"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8680")
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://127.0.0.1:8682")
BACKEND_PARTS = urlsplit(BACKEND_BASE_URL)
BACKEND_HOST = BACKEND_PARTS.hostname or "127.0.0.1"
BACKEND_PORT = BACKEND_PARTS.port or 8682
BACKEND_SCHEME = BACKEND_PARTS.scheme or "http"
DGX_HOST = os.environ.get("DGX_HOST", "")
DGX_USER = os.environ.get("DGX_USER", "")
DGX_RPC_PORT = os.environ.get("DGX_RPC_PORT", "50052")
CONTROL_PATH = str(PID_DIR / "ssh-dgx.ctl")
HISTORY = deque(maxlen=512)
HISTORY_WINDOW_SECONDS = 20 * 60


def run(cmd, timeout=4):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except Exception:
        return 1, "", ""


def fetch_text(url, timeout=2):
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def format_compact_number(value):
    try:
        value = float(value)
    except Exception:
        return "--"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def parse_metric(metrics, name, as_float=False):
    pattern = re.compile(rf"^{re.escape(name)}\s+(.+)$", re.MULTILINE)
    match = pattern.search(metrics)
    if not match:
        return None
    raw = match.group(1).strip()
    if as_float:
        try:
            return float(raw)
        except Exception:
            return None
    return raw


def detect_mode():
    if not DGX_HOST or not DGX_USER:
        return "VISION"
    code, _, _ = run([
        "ssh", "-O", "check", "-o", f"ControlPath={CONTROL_PATH}", f"{DGX_USER}@{DGX_HOST}"
    ], timeout=2)
    return "DISTRIBUTED" if code == 0 else "VISION"


def mac_ram_used():
    code, vm_out, _ = run(["vm_stat"], timeout=2)
    if code != 0:
        return "--"
    page_match = re.search(r"page size of (\d+) bytes", vm_out)
    if not page_match:
        return "--"
    page_size = int(page_match.group(1))
    total_code, total_out, _ = run(["sysctl", "-n", "hw.memsize"], timeout=2)
    if total_code != 0 or not total_out.isdigit():
        return "--"
    total_bytes = int(total_out)

    def extract(name):
        match = re.search(rf"{re.escape(name)}:\s+(\d+)\.", vm_out)
        return int(match.group(1)) if match else 0

    free_pages = extract("Pages free")
    speculative_pages = extract("Pages speculative")
    inactive_pages = extract("Pages inactive")
    free_bytes = (free_pages + speculative_pages + inactive_pages) * page_size
    used_gb = (total_bytes - free_bytes) / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    used_pct = int(round((used_gb / total_gb) * 100)) if total_gb else 0
    return f"{used_pct}%"


def mac_gpu_util():
    cmd = [
        "sh", "-lc",
        "ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '\"Device Utilization %\"=[0-9]*' | awk -F'=' '{print $NF}' | head -1",
    ]
    code, out, _ = run(cmd, timeout=3)
    return f"{out}%" if code == 0 and out else "--"


def dgx_snapshot():
    if detect_mode() != "DISTRIBUTED":
        return {"ram": "--", "gpu": "--", "rpc": "--"}
    remote = (
        "echo \"MEM:$(free -m 2>/dev/null | awk '/^Mem:/{printf \"%d %d\", $3, $2}')\"; "
        "echo \"GPU_UTIL:$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')\"; "
        "echo \"GPU_MEM:$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)\"; "
        "echo \"RPC:$(pgrep -x rpc-server >/dev/null 2>&1 && echo UP || echo DOWN)\""
    )
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=5",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={CONTROL_PATH}",
        "-o", "ControlPersist=600",
        f"{DGX_USER}@{DGX_HOST}",
        remote,
    ]
    code, out, _ = run(cmd, timeout=5)
    if code != 0 or not out:
        return {"ram": "--", "gpu": "--", "rpc": "DOWN"}

    mem = re.search(r"^MEM:(\d+) (\d+)$", out, re.MULTILINE)
    gpu_util = re.search(r"^GPU_UTIL:(.*)$", out, re.MULTILINE)
    gpu_mem = re.search(r"^GPU_MEM:(.*)$", out, re.MULTILINE)
    rpc = re.search(r"^RPC:(.*)$", out, re.MULTILINE)
    ram = "--"
    gpu = "--"

    if mem:
        used = int(mem.group(1))
        total = int(mem.group(2))
        if total > 0:
            ram = f"{(used * 100) // total}%"

    if gpu_mem and gpu_mem.group(1).strip():
        parts = [p.strip() for p in gpu_mem.group(1).split(",")]
        if len(parts) >= 1 and parts[0].isdigit():
            used_mb = int(parts[0])
            gpu = f"{(gpu_util.group(1).strip() if gpu_util else '?')}%/{used_mb // 1024}G"
    elif gpu_util and gpu_util.group(1).strip():
        gpu = f"{gpu_util.group(1).strip()}%/UMA"

    return {"ram": ram, "gpu": gpu, "rpc": (rpc.group(1).strip() if rpc else "--")}


def llama_snapshot():
    health = bool(fetch_text(f"{BACKEND_BASE_URL}/health", timeout=2))
    metrics = fetch_text(f"{BACKEND_BASE_URL}/metrics", timeout=2) if health else ""
    pp = parse_metric(metrics, "llamacpp:prompt_tokens_seconds", as_float=True)
    tg = parse_metric(metrics, "llamacpp:predicted_tokens_seconds", as_float=True)
    reqs = parse_metric(metrics, "llamacpp:requests_processing")
    prompt_total = parse_metric(metrics, "llamacpp:prompt_tokens_total", as_float=True)
    gen_total = parse_metric(metrics, "llamacpp:tokens_predicted_total", as_float=True)
    return {
        "status": "UP" if health else "DOWN",
        "pp": f"{pp:.1f}" if pp is not None else "--",
        "tg": f"{tg:.1f}" if tg is not None else "--",
        "reqs": reqs if reqs is not None else "--",
        "prompt_tokens": format_compact_number(prompt_total) if prompt_total is not None else "--",
        "gen_tokens": format_compact_number(gen_total) if gen_total is not None else "--",
    }


def log_tail():
    log_file = LOG_DIR / "llama-server.log"
    if not log_file.exists():
        return []
    try:
        lines = log_file.read_text(errors="replace").splitlines()
    except Exception:
        return []
    return lines[-5:]


def take_snapshot():
    now = time.time()
    mode = detect_mode()
    dgx = dgx_snapshot()
    llama = llama_snapshot()
    snapshot = {
        "captured_at": now,
        "timestamp": subprocess.run(["date", "+%H:%M:%S"], capture_output=True, text=True).stdout.strip(),
        "mode": mode,
        "mac_ram": mac_ram_used(),
        "mac_gpu": mac_gpu_util(),
        "dgx_ram": dgx["ram"],
        "dgx_gpu": dgx["gpu"],
        "rpc_server": dgx["rpc"],
        "llama_server": llama["status"],
        "pp": llama["pp"],
        "tg": llama["tg"],
        "reqs": llama["reqs"],
        "prompt_tokens": llama["prompt_tokens"],
        "gen_tokens": llama["gen_tokens"],
        "log_lines": log_tail(),
        "endpoints": {
            "monitor": f"{PUBLIC_BASE_URL}/monitor",
            "api": f"{PUBLIC_BASE_URL}/v1",
            "health": f"{PUBLIC_BASE_URL}/health",
            "metrics": f"{PUBLIC_BASE_URL}/metrics",
            "rpc": f"{DGX_HOST}:{DGX_RPC_PORT}" if DGX_HOST else "--",
            "backend": BACKEND_BASE_URL,
        },
    }
    HISTORY.append(snapshot)
    cutoff = now - HISTORY_WINDOW_SECONDS
    while HISTORY and HISTORY[0].get("captured_at", 0) < cutoff:
        HISTORY.popleft()
    return snapshot


INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>rz-rpc-llm Monitor</title>
  <style>
    :root {
      --bg: #0a0f16;
      --panel: rgba(13, 21, 33, 0.86);
      --line: rgba(122, 162, 255, 0.18);
      --text: #e8eef8;
      --muted: #93a3b8;
      --good: #2fd27f;
      --warn: #ffbd45;
      --bad: #ff6b6b;
      --cyan: #5ed3f3;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(94, 211, 243, 0.13), transparent 34%),
        radial-gradient(circle at top right, rgba(255, 123, 192, 0.12), transparent 28%),
        linear-gradient(180deg, #09111b, #05080d 70%);
      min-height: 100vh;
    }
    .wrap { max-width: 1500px; margin: 0 auto; padding: 24px; }
    .hero {
      display: grid; gap: 18px; margin-bottom: 18px; padding: 20px 22px;
      grid-template-columns: 1fr;
      align-items: start;
      background: linear-gradient(135deg, rgba(16,28,45,0.9), rgba(8,14,23,0.84));
      border: 1px solid var(--line); border-radius: 22px; box-shadow: var(--shadow);
    }
    .hero-main { display: grid; gap: 16px; }
    .hero h1 { margin: 0; font-size: clamp(24px, 4vw, 40px); }
    .hero p { margin: 0; color: var(--muted); max-width: 72ch; }
    .meta, .cards { display: grid; gap: 14px; }
    .meta { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .cards { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin: 18px 0; }
    .card, .panel {
      background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
      padding: 16px; box-shadow: var(--shadow); backdrop-filter: blur(14px);
    }
    .label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .value { margin-top: 8px; font-size: 28px; font-weight: 700; }
    .value.small { font-size: 20px; }
    .pill {
      display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px;
      border-radius: 999px; background: rgba(255,255,255,0.04); border: 1px solid var(--line);
      font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
    }
    .dot { width: 9px; height: 9px; border-radius: 999px; display: inline-block; }
    .good { color: var(--good); } .warn { color: var(--warn); } .bad { color: var(--bad); }
    .dot.good { background: var(--good); box-shadow: 0 0 18px rgba(47,210,127,0.7); }
    .dot.warn { background: var(--warn); box-shadow: 0 0 18px rgba(255,189,69,0.6); }
    .dot.bad { background: var(--bad); box-shadow: 0 0 18px rgba(255,107,107,0.65); }
    .panel h2 { margin: 0 0 12px; font-size: 15px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .panel.edge { padding-left: 0; padding-right: 0; overflow: hidden; }
    .panel.edge h2, .panel.edge .history-note { padding-left: 16px; padding-right: 16px; }
    .history-note { margin: -4px 0 10px; color: var(--muted); font-size: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
    th, td { padding: 9px 6px; border-bottom: 1px solid rgba(255,255,255,0.06); text-align: right; }
    th:first-child, td:first-child {
      width: 154px; min-width: 154px; text-align: left; position: sticky; left: 0;
      background: rgba(12,19,30,0.98); z-index: 1;
    }
    thead th { color: var(--muted); font-weight: 600; }
    .history {
      width: 100%; overflow: hidden; border-top: 1px solid rgba(255,255,255,0.06);
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .log {
      margin: 0; padding: 14px; min-height: 250px; border-radius: 14px; overflow: auto;
      background: linear-gradient(180deg, rgba(5,8,14,0.95), rgba(8,11,18,0.95));
      border: 1px solid rgba(255,255,255,0.06); color: #d7f5e7; line-height: 1.55;
    }
    .endpoint-list { display: grid; gap: 10px; }
    .endpoint-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .endpoint-item {
      min-width: 0; padding: 12px 14px; border-radius: 14px;
      background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    }
    .endpoint-list a { color: var(--cyan); text-decoration: none; word-break: break-all; }
    .footer { margin-top: 14px; color: var(--muted); font-size: 12px; }
    @media (max-width: 1100px) { .hero p { max-width: none; } }
    @media (max-width: 960px) {
      .wrap { padding: 16px; }
      th:first-child, td:first-child { width: 118px; min-width: 118px; }
      .endpoint-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      .endpoint-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <div class=\"hero-main\">
        <div>
          <h1>rz-rpc-llm monitor</h1>
          <p>Single-port gateway on <code>/monitor</code> for the same developer loop as <code>./deploy.sh monitor</code>: health, throughput, rolling history, and the last 5 log lines.</p>
        </div>
        <div class=\"meta\" id=\"meta\"></div>
      </div>
    </section>

    <section class=\"cards\" id=\"cards\"></section>

    <section class=\"panel edge\">
      <h2>Rolling History</h2>
      <div class=\"history-note\">20 minute view with denser samples near the present. New samples land on the right edge and the oldest buckets fall away on the left.</div>
      <div class=\"history\"><table id=\"history\"></table></div>
    </section>

    <section class=\"panel\" style=\"margin-top: 16px;\">
      <h2>Live Log</h2>
      <pre class=\"log\" id=\"log\"></pre>
    </section>

    <section class=\"panel\" style=\"margin-top: 16px;\">
      <h2>Endpoints</h2>
      <div class=\"endpoint-list endpoint-grid\" id=\"endpoints\"></div>
    </section>
    <div class=\"footer\" id=\"footer\"></div>
  </div>
  <script>
    const HISTORY_WINDOW_SECONDS = 20 * 60;
    let lastData = null;
    const rows = [
      ["Mac RAM used", "mac_ram"],
      ["Mac GPU util", "mac_gpu"],
      ["DGX RAM used", "dgx_ram", "DISTRIBUTED"],
      ["DGX GPU util", "dgx_gpu", "DISTRIBUTED"],
      ["rpc-server", "rpc_server", "DISTRIBUTED"],
      ["llama-server", "llama_server"],
      ["pp (t/s)", "pp"],
      ["tg (t/s)", "tg"],
      ["reqs", "reqs"],
      ["prompt tokens", "prompt_tokens"],
      ["gen tokens", "gen_tokens"],
    ];

    function stateClass(value) {
      if (value === "UP") return "good";
      if (value === "DOWN") return "bad";
      if (typeof value === "string" && value.includes("%")) {
        const match = value.match(/(\\d+)%/);
        if (match) {
          const pct = Number(match[1]);
          if (pct < 50) return "good";
          if (pct < 80) return "warn";
          return "bad";
        }
      }
      return "good";
    }

    function card(label, value) {
      return `<div class=\"card\"><div class=\"label\">${label}</div><div class=\"value ${String(value).length > 9 ? "small" : ""} ${stateClass(value)}\">${value}</div></div>`;
    }

    function historyColumnCount() {
      if (window.innerWidth < 640) return 18;
      if (window.innerWidth < 960) return 24;
      return 32;
    }

    function historyBuckets(history, count) {
      const now = history.length ? history[history.length - 1].captured_at : (Date.now() / 1000);
      const formatAge = (seconds) => {
        if (seconds >= 60) return `-${Math.round(seconds / 60)}m`;
        return `-${Math.round(seconds)}s`;
      };
      const boundaries = Array.from({ length: count + 1 }, (_, index) => {
        const distance = (count - index) / count;
        return HISTORY_WINDOW_SECONDS * (distance ** 2);
      });
      const newestFirst = [...history].reverse();

      return Array.from({ length: count }, (_, index) => {
        const olderAge = boundaries[index];
        const newerAge = boundaries[index + 1];
        const labelAge = Math.max(0, Math.round((olderAge + newerAge) / 2));
        const sample = newestFirst.find((entry) => {
          const age = now - entry.captured_at;
          const withinOlderBound = age <= olderAge || index === 0;
          const withinNewerBound = age > newerAge || index === count - 1;
          return withinOlderBound && withinNewerBound;
        });

        return sample || {
          label: formatAge(labelAge),
          empty: true,
        };
      }).map((bucket) => ({
        ...bucket,
        label: bucket.label || formatAge(Math.max(0, Math.round(now - bucket.captured_at))),
      }));
    }

    function render(data) {
      const latest = data.latest;
      document.getElementById("meta").innerHTML = [
        ["mode", latest.mode],
        ["updated", latest.timestamp],
        ["llama", latest.llama_server],
        ["rpc", latest.rpc_server],
      ].map(([k, v]) => `<div class=\"pill\"><span class=\"dot ${stateClass(v)}\"></span>${k}: ${v}</div>`).join("");

      document.getElementById("cards").innerHTML = [
        card("pp (t/s)", latest.pp),
        card("tg (t/s)", latest.tg),
        card("reqs", latest.reqs),
        card("prompt tokens", latest.prompt_tokens),
        card("gen tokens", latest.gen_tokens),
        card("mac ram", latest.mac_ram),
        card("mac gpu", latest.mac_gpu),
        ...(latest.mode === "DISTRIBUTED" ? [card("dgx ram", latest.dgx_ram), card("dgx gpu", latest.dgx_gpu)] : []),
      ].join("");

      const visibleRows = rows.filter((row) => !row[2] || row[2] === latest.mode);
      const rawHistory = data.history;
      const history = historyBuckets(rawHistory, historyColumnCount());
      document.getElementById("history").innerHTML =
        `<thead><tr><th>metric</th>${history.map((h) => `<th>${h.label}</th>`).join("")}</tr></thead>` +
        `<tbody>${visibleRows.map(([label, key]) => `<tr><td>${label}</td>${history.map((h) => `<td class=\"${h.empty ? "" : stateClass(h[key])}\">${h.empty ? "--" : h[key]}</td>`).join("")}</tr>`).join("")}</tbody>`;

      document.getElementById("endpoints").innerHTML = Object.entries(latest.endpoints)
        .map(([k, v]) => `<div class=\"endpoint-item\"><div class=\"label\">${k}</div><a href=\"${String(v).startsWith("http") ? v : "#"}\" target=\"_blank\">${v}</a></div>`)
        .join("");

      document.getElementById("log").textContent = latest.log_lines.join("\\n") || "No llama-server log yet.";
      document.getElementById("footer").textContent = `Single-port gateway • polling /monitor/api every 3s • ${rawHistory.length} raw samples across the last 20 minutes • ${history.length} visible history buckets`;
    }

    async function refresh() {
      try {
        const res = await fetch('/monitor/api', { cache: 'no-store' });
        const data = await res.json();
        lastData = data;
        render(data);
      } catch (err) {
        document.getElementById("footer").textContent = `Monitor fetch failed: ${err}`;
      }
    }

    refresh();
    setInterval(refresh, 3000);
    window.addEventListener("resize", () => {
      if (lastData) render(lastData);
    });
  </script>
</body>
</html>
"""


def proxy_request(handler):
    body = b""
    content_length = handler.headers.get("Content-Length")
    if content_length:
        body = handler.rfile.read(int(content_length))

    connection = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=600)
    headers = {k: v for k, v in handler.headers.items() if k.lower() not in {"host", "connection", "accept-encoding"}}
    headers["Host"] = f"{BACKEND_HOST}:{BACKEND_PORT}"

    try:
        connection.request(handler.command, handler.path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
    except Exception as exc:
        error = json.dumps({"error": f"backend unavailable: {exc}"}).encode("utf-8")
        handler.send_response(502)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(error)))
        handler.end_headers()
        handler.wfile.write(error)
        return
    finally:
        connection.close()

    handler.send_response(response.status, response.reason)
    excluded = {"connection", "transfer-encoding", "content-length", "server", "date"}
    for key, value in response.getheaders():
        if key.lower() not in excluded:
            handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(payload)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path in ("/monitor", "/monitor/"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/monitor/api", "/monitor/api/"):
            latest = take_snapshot()
            body = json.dumps({"latest": latest, "history": list(HISTORY)}, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        proxy_request(self)

    def do_POST(self):
        proxy_request(self)

    def do_OPTIONS(self):
        proxy_request(self)

    def do_HEAD(self):
        proxy_request(self)

    def log_message(self, format, *args):
        sys.stdout.write("monitor-web: " + (format % args) + "\n")
        sys.stdout.flush()


def main():
    host = os.environ.get("MONITOR_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("MONITOR_WEB_PORT", "8680"))
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "--host" and index + 1 < len(args):
            host = args[index + 1]
        if arg == "--port" and index + 1 < len(args):
            port = int(args[index + 1])
    take_snapshot()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"monitor-web gateway listening on http://{host}:{port}/monitor -> {BACKEND_BASE_URL}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
