#!/usr/bin/env python3
import configparser
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
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
PORTAL_SOLO_ONLY = os.environ.get("PORTAL_SOLO_ONLY", "0") == "1"
CONTROL_PATH = str(PID_DIR / "ssh-dgx.ctl")
DEPLOY_SCRIPT = str(ROOT / "deploy.sh")
HISTORY = deque(maxlen=512)
HISTORY_WINDOW_SECONDS = 20 * 60
_task_prompt_tokens = {}
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(s):
    return _ANSI_RE.sub('', s)


# ---------------------------------------------------------------------------
# Deployment state machine
# ---------------------------------------------------------------------------

class DeploymentState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"
        self.deploy_id = ""
        self.step = ""
        self.model_name = ""
        self.started_at = 0.0
        self.log_lines = []
        self.log_cursor = 0
        self.process = None
        self.serving_model = ""
        self.serving_params = {}

    def start(self, model, deploy_id, params=None):
        with self.lock:
            if self.status == "deploying":
                return False
            self.status = "deploying"
            self.deploy_id = deploy_id
            self.step = "initializing"
            self.model_name = model
            self._pending_params = params or {}
            self.started_at = time.time()
            self.log_lines = []
            self.log_cursor = 0
            self.process = None
            return True

    def append_log(self, line):
        with self.lock:
            self.log_lines.append(line)
            self.log_cursor += 1
            if len(self.log_lines) > 5000:
                self.log_lines = self.log_lines[-4000:]

    def get_logs_since(self, cursor):
        with self.lock:
            total = self.log_cursor
            available = len(self.log_lines)
            skip = max(0, available - (total - cursor))
            lines = self.log_lines[skip:] if cursor < total else []
            return lines, total

    def set_step(self, name):
        with self.lock:
            self.step = name

    def finish(self, success):
        with self.lock:
            self.status = "success" if success else "failed"
            if success:
                self.serving_model = self.model_name
                self.serving_params = dict(self._pending_params)
            self.process = None

    def cancel(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self.status = "failed"
            self.step = "cancelled"

    def to_dict(self):
        with self.lock:
            return {
                "status": self.status,
                "deploy_id": self.deploy_id,
                "step": self.step,
                "model_name": self.model_name,
                "serving_model": self.serving_model,
                "serving_params": dict(self.serving_params),
                "started_at": self.started_at,
                "elapsed": (time.time() - self.started_at) if self.status == "deploying" else 0,
            }


DEPLOY = DeploymentState()


# ---------------------------------------------------------------------------
# Model registry & tag helpers
# ---------------------------------------------------------------------------

def parse_models_conf():
    conf = configparser.ConfigParser()
    conf.read(str(ROOT / "models.conf"))
    models = []
    for section in conf.sections():
        m = dict(conf[section])
        if PORTAL_SOLO_ONLY and m.get("solo") != "yes":
            continue
        m["id"] = section
        if PORTAL_SOLO_ONLY:
            m["default_mode"] = "solo"
        models.append(m)
    return models


def get_model_conf(model_id):
    conf = configparser.ConfigParser()
    conf.read(str(ROOT / "models.conf"))
    if conf.has_section(model_id):
        return dict(conf[model_id])
    return None


def _format_age(seconds):
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    else:
        return f"{int(seconds / 86400)}d ago"


def get_llama_tags():
    llama_dir = ROOT / "llama.cpp"
    if not (llama_dir / ".git").is_dir():
        return []
    run_cmd(["git", "-C", str(llama_dir), "fetch", "--tags", "--prune"], timeout=30)

    # Find current tag
    current_tag = None
    rc, desc, _ = run_cmd(["git", "-C", str(llama_dir), "describe", "--tags", "--exact-match", "HEAD"], timeout=5)
    if rc == 0 and desc.strip():
        current_tag = desc.strip()

    # Get b* tags with their creation time
    code, out, _ = run_cmd([
        "git", "-C", str(llama_dir),
        "for-each-ref", "--sort=-version:refname",
        "--format=%(refname:short) %(creatordate:unix)",
        "refs/tags/b[0-9]*",
    ], timeout=10)
    if code != 0:
        return []

    now = time.time()
    three_days = 3 * 86400
    results = []
    current_entry = None

    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        tag, ts = parts[0], parts[1]
        try:
            age = now - float(ts)
        except ValueError:
            continue
        age_label = _format_age(age)
        is_current = tag == current_tag
        entry = {
            "tag": tag,
            "label": f"[current] {tag} ({age_label})" if is_current else f"{tag} ({age_label})",
            "current": is_current,
        }
        if is_current:
            current_entry = entry
        # Cap: within 3 days or top 5, whichever is more
        if age <= three_days or len(results) < 5:
            results.append(entry)
        else:
            break

    # Ensure current tag is always included
    if current_entry and current_entry not in results:
        results.append(current_entry)

    return results


# ---------------------------------------------------------------------------
# Deployment pipeline — runs in a daemon thread
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r'═{4,}')


def _run_step(step_name, cmd, env):
    """Run a single deploy.sh step, streaming output to DEPLOY."""
    DEPLOY.set_step(step_name)
    DEPLOY.append_log(f"\n>>> {step_name}\n")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, cwd=str(ROOT),
    )
    with DEPLOY.lock:
        DEPLOY.process = proc
    for line in proc.stdout:
        DEPLOY.append_log(strip_ansi(line.rstrip('\n')))
    proc.wait()
    with DEPLOY.lock:
        DEPLOY.process = None
    if proc.returncode != 0:
        DEPLOY.append_log(f"Step '{step_name}' failed (exit {proc.returncode})")
        return False
    return True


def run_deployment(params):
    """Full deployment pipeline. Called in a daemon thread."""
    env = {**os.environ, "TERM": "dumb", "SKIP_MONITOR_RESTART": "1"}
    model = params.get("model", "")
    mode = params.get("mode", "")
    if PORTAL_SOLO_ONLY:
        mode = "solo"
    vision = params.get("vision", False)
    tag = params.get("tag", "")
    ctx = params.get("ctx", "")
    parallel = params.get("parallel", "")
    split_mode = params.get("split_mode", "")
    tensor_split = params.get("tensor_split", "")
    skip_clone = params.get("skip_clone", False)
    skip_build = params.get("skip_build", False)
    skip_download = params.get("skip_download", False)

    try:
        # 1. Clone
        if not skip_clone and tag:
            if not _run_step("clone", ["bash", DEPLOY_SCRIPT, "debug", "clone", "--tag", tag], env):
                DEPLOY.finish(False)
                return
        elif not skip_clone:
            llama_dir = ROOT / "llama.cpp"
            if not (llama_dir / ".git").is_dir():
                if not _run_step("clone", ["bash", DEPLOY_SCRIPT, "debug", "clone", "--tag", "latest"], env):
                    DEPLOY.finish(False)
                    return

        # 2. Build DGX (if distributed)
        if not skip_build and mode == "distributed":
            if not _run_step("build-dgx", ["bash", DEPLOY_SCRIPT, "debug", "build-dgx"], env):
                DEPLOY.finish(False)
                return

        # 3. Build Mac
        if not skip_build:
            if not _run_step("build-mac", ["bash", DEPLOY_SCRIPT, "debug", "build-mac"], env):
                DEPLOY.finish(False)
                return

        # 4. Download model
        if not skip_download:
            dl_cmd = ["bash", DEPLOY_SCRIPT, "debug", "download", "--model", model]
            if vision:
                dl_cmd.append("--vision")
            if not _run_step("download", dl_cmd, env):
                DEPLOY.finish(False)
                return

        # 5. Stop existing servers (always stop both to handle mode switches)
        _run_step("stop-llama", ["bash", DEPLOY_SCRIPT, "debug", "stop-llama"], env)
        if not PORTAL_SOLO_ONLY:
            _run_step("stop-rpc", ["bash", DEPLOY_SCRIPT, "debug", "stop-rpc"], env)

        # 6. Start RPC if distributed
        if mode == "distributed":
            if not _run_step("start-rpc", ["bash", DEPLOY_SCRIPT, "debug", "start-rpc"], env):
                DEPLOY.finish(False)
                return

        # 8. Start llama-server
        start_cmd = ["bash", DEPLOY_SCRIPT, "debug", "start-llama", "--model", model]
        if vision:
            start_cmd.append("--vision")
        if mode == "solo":
            start_cmd.append("--solo")
        elif mode == "distributed":
            start_cmd.append("--distributed")
        if ctx:
            start_cmd.extend(["--ctx", str(ctx)])
        if parallel:
            start_cmd.extend(["--parallel", str(parallel)])
        if mode == "distributed" and split_mode:
            start_cmd.extend(["--split-mode", split_mode])
        if mode == "distributed" and tensor_split:
            start_cmd.extend(["--tensor-split", tensor_split])
        if not _run_step("start-llama", start_cmd, env):
            DEPLOY.finish(False)
            return

        DEPLOY.append_log("\nDeployment complete.")
        DEPLOY.finish(True)

    except Exception as exc:
        DEPLOY.append_log(f"\nDeployment error: {exc}")
        DEPLOY.finish(False)


def handle_deploy_stream(handler):
    """SSE endpoint streaming deployment log."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()
    cursor = 0
    while True:
        lines, cursor = DEPLOY.get_logs_since(cursor)
        for line in lines:
            data = json.dumps({"line": line, "step": DEPLOY.step, "status": DEPLOY.status})
            handler.wfile.write(f"data: {data}\n\n".encode())
        try:
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        if DEPLOY.status in ("success", "failed", "idle"):
            data = json.dumps({"done": True, "status": DEPLOY.status})
            handler.wfile.write(f"data: {data}\n\n".encode())
            try:
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Monitoring helpers (unchanged from original)
# ---------------------------------------------------------------------------

def run_cmd(cmd, timeout=4):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except Exception:
        return 1, "", ""


# Keep 'run' as alias for existing code
run = run_cmd


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
        v = value / 1_000_000
        return f"{v:.0f}M" if v >= 10 else f"{v:.1f}M"
    if value >= 1_000:
        v = value / 1_000
        return f"{v:.0f}K" if v >= 10 else f"{v:.1f}K"
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
    serving_mode = (DEPLOY.to_dict().get("serving_params", {}) or {}).get("mode", "")
    if PORTAL_SOLO_ONLY:
        return "VISION"
    if serving_mode == "solo":
        return "VISION"
    if serving_mode == "distributed":
        return "DISTRIBUTED"
    if DGX_HOST and DGX_USER:
        return "DISTRIBUTED"
    return "VISION"


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
    rpc = re.search(r"^RPC:(.*)$", out, re.MULTILINE)
    ram = "--"
    gpu = "--"

    if mem:
        used = int(mem.group(1))
        total = int(mem.group(2))
        if total > 0:
            ram = f"{(used * 100) // total}%"

    if gpu_util and gpu_util.group(1).strip():
        gpu = f"{gpu_util.group(1).strip()}%"

    return {"ram": ram, "gpu": gpu, "rpc": (rpc.group(1).strip() if rpc else "--")}


def retry_dgx_ssh(timeout=20):
    if PORTAL_SOLO_ONLY:
        return False, "DGX disabled in managed solo mode"
    if not DGX_HOST or not DGX_USER:
        return False, "DGX SSH config missing"
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=1",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={CONTROL_PATH}",
        "-o", "ControlPersist=600",
        f"{DGX_USER}@{DGX_HOST}",
        "echo",
        "SSH OK",
    ]
    code, out, err = run(cmd, timeout=timeout)
    if code == 0:
        return True, f"DGX SSH connected: {DGX_USER}@{DGX_HOST}"
    detail = (err or out or "RPC server offline").strip()
    if len(detail) > 240:
        detail = detail[:237] + "..."
    return False, f"RPC server offline ({detail})"


def current_backend_model():
    raw = fetch_text(f"{BACKEND_BASE_URL}/v1/models", timeout=3)
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except Exception:
        return ""
    data = payload.get("data") or []
    for item in data:
        model_id = (item or {}).get("id", "")
        if model_id and model_id != "rz-llm-default":
            return model_id
    return ""


def format_ctx_compact(n):
    try:
        n = int(n)
    except Exception:
        return "--"
    if n <= 0:
        return "0"
    if n < 10000:
        return str(n)
    if n < 1_000_000:
        return f"{n // 1000}K"
    return f"{n // 1_000_000}M"


_NEW_PROMPT_RE = re.compile(
    r"id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|.*new prompt.*task\.n_tokens\s*=\s*(\d+)"
)


def _refresh_task_prompt_cache():
    log_file = LOG_DIR / "llama-server.log"
    if not log_file.exists():
        return
    try:
        size = log_file.stat().st_size
        with open(log_file, "r", errors="replace") as f:
            if size > 65536:
                f.seek(size - 65536)
                f.readline()
            for line in f:
                m = _NEW_PROMPT_RE.search(line)
                if m:
                    task_id = int(m.group(2))
                    n_tokens = int(m.group(3))
                    _task_prompt_tokens[task_id] = n_tokens
    except Exception:
        pass


def slots_snapshot(health):
    if not health:
        return []
    raw = fetch_text(f"{BACKEND_BASE_URL}/slots", timeout=4)
    if not raw:
        return []
    try:
        slots = json.loads(raw)
    except Exception:
        return []
    _refresh_task_prompt_cache()
    result = []
    for s in slots:
        sid = s.get("id", 0)
        active = s.get("is_processing", False)
        nt = (s.get("next_token") or [{}])[0]
        n_decoded = nt.get("n_decoded", 0)
        task_id = s.get("id_task", -1)

        if active:
            prompt_n = _task_prompt_tokens.get(task_id, 0)
            total_ctx = prompt_n + n_decoded
        else:
            total_ctx = 0

        result.append({
            "id": sid,
            "active": active,
            "ctx": total_ctx,
        })
    return result


def llama_snapshot():
    health = bool(fetch_text(f"{BACKEND_BASE_URL}/health", timeout=2))
    metrics = fetch_text(f"{BACKEND_BASE_URL}/metrics", timeout=2) if health else ""
    pp = parse_metric(metrics, "llamacpp:prompt_tokens_seconds", as_float=True)
    tg = parse_metric(metrics, "llamacpp:predicted_tokens_seconds", as_float=True)
    reqs = parse_metric(metrics, "llamacpp:requests_processing")
    prompt_total = parse_metric(metrics, "llamacpp:prompt_tokens_total", as_float=True)
    gen_total = parse_metric(metrics, "llamacpp:tokens_predicted_total", as_float=True)
    slots = slots_snapshot(health)
    return {
        "status": "UP" if health else "DOWN",
        "pp": (f"{pp:.0f}" if pp >= 100 else f"{pp:.1f}") if pp is not None else "--",
        "tg": (f"{tg:.0f}" if tg >= 100 else f"{tg:.1f}") if tg is not None else "--",
        "reqs": reqs if reqs is not None else "--",
        "prompt_tokens": format_compact_number(prompt_total) if prompt_total is not None else "--",
        "gen_tokens": format_compact_number(gen_total) if gen_total is not None else "--",
        "slots": slots,
    }


def log_tail():
    log_file = LOG_DIR / "llama-server.log"
    if not log_file.exists():
        return []
    try:
        lines = log_file.read_text(errors="replace").splitlines()
    except Exception:
        return []
    filtered = [l for l in lines if "GET /slots" not in l]
    return filtered[-5:]


def take_snapshot():
    now = time.time()
    mode = detect_mode()
    dgx = dgx_snapshot()
    llama = llama_snapshot()
    slot_fields = {}
    slots = llama.get("slots", [])
    for s in slots:
        sid = s["id"]
        if s["active"]:
            slot_fields[f"slot_ctx_{sid}"] = format_ctx_compact(s["ctx"])
        else:
            slot_fields[f"slot_ctx_{sid}"] = "--"
    deploy_state = DEPLOY.to_dict()
    backend_model = current_backend_model() if llama.get("status") == "UP" else ""
    snapshot = {
        "captured_at": now,
        "timestamp": subprocess.run(["date", "+%H:%M:%S"], capture_output=True, text=True).stdout.strip(),
        "mode": mode,
        "serving_model": deploy_state.get("serving_model", ""),
        "backend_model": backend_model,
        "serving_params": deploy_state.get("serving_params", {}),
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
        "n_slots": len(slots),
        **slot_fields,
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


# ---------------------------------------------------------------------------
# Portal HTML — deploy panel + integrated monitor
# ---------------------------------------------------------------------------

PORTAL_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>rz-rpc-llm Portal</title>
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
    .hero-main { display: flex; justify-content: space-between; align-items: start; gap: 16px; flex-wrap: wrap; }
    .hero h1 { margin: 0; font-size: clamp(24px, 4vw, 40px); }
    .hero p { margin: 0; color: var(--muted); max-width: 72ch; }
    .hero-status { text-align: right; font-size: 12px; line-height: 1.7; color: var(--muted); white-space: nowrap; }
    .hero-status .model-name { font-size: 15px; font-weight: 700; color: var(--fg); }
    .hero-status .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px; }
    .status-banner {
      display: none; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
      padding: 12px 14px; border-radius: 16px;
      background: linear-gradient(135deg, rgba(255,107,107,0.16), rgba(255,189,69,0.12));
      border: 1px solid rgba(255,107,107,0.28);
    }
    .status-banner.visible { display: flex; }
    .status-banner-copy { display: grid; gap: 3px; }
    .status-banner-title { font-size: 13px; font-weight: 700; color: var(--text); letter-spacing: 0.04em; text-transform: uppercase; }
    .status-banner-text { font-size: 12px; color: var(--muted); }
    .cards { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
    .card, .panel {
      background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
      padding: 10px 12px; box-shadow: var(--shadow); backdrop-filter: blur(14px);
    }
    .card { flex: 1 1 0; min-width: 90px; }
    .label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .value { margin-top: 4px; font-size: 18px; font-weight: 700; }
    .value.small { font-size: 15px; }
    .panel { border-radius: 18px; padding: 16px; }
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
      margin: 0; padding: 14px; min-height: 150px; border-radius: 14px; overflow: auto;
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

    /* Deploy panel */
    .deploy-section { margin-bottom: 18px; }
    .model-cards { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin: 12px 0; }
    .model-card {
      padding: 14px 16px; border-radius: 14px; cursor: pointer;
      background: rgba(255,255,255,0.03); border: 2px solid rgba(255,255,255,0.08);
      transition: border-color 0.2s;
    }
    .model-card:hover { border-color: rgba(94, 211, 243, 0.4); }
    .model-card.selected { border-color: var(--cyan); background: rgba(94, 211, 243, 0.08); }
    .model-card .model-name { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
    .model-card .model-meta { font-size: 12px; color: var(--muted); display: flex; gap: 10px; flex-wrap: wrap; }
    .model-card .badge {
      display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px;
      background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    }
    .model-card .badge.yes { color: var(--good); border-color: rgba(47,210,127,0.3); }
    .model-card .badge.no { color: var(--muted); }
    .deploy-controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin: 14px 0; }
    .deploy-controls label { font-size: 13px; color: var(--muted); }
    .deploy-controls select, .deploy-controls input[type=number] {
      background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15);
      color: var(--text); padding: 6px 10px; border-radius: 8px; font-family: inherit; font-size: 13px;
    }
    .deploy-controls input[type=checkbox] { accent-color: var(--cyan); }
    details { margin: 10px 0; }
    details summary {
      cursor: pointer; color: var(--muted); font-size: 13px; padding: 6px 0;
      user-select: none;
    }
    details[open] summary { color: var(--cyan); }
    .advanced-grid { display: flex; flex-wrap: wrap; gap: 14px; padding: 10px 0; }
    .btn {
      padding: 10px 24px; border-radius: 10px; font-family: inherit; font-size: 14px;
      font-weight: 700; cursor: pointer; border: none; transition: all 0.2s;
    }
    .btn-deploy {
      background: linear-gradient(135deg, #2fd27f, #1ba865); color: #0a0f16;
    }
    .btn-deploy:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(47,210,127,0.4); }
    .btn-deploy:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }
    .btn-cancel {
      background: rgba(255,107,107,0.2); color: var(--bad); border: 1px solid rgba(255,107,107,0.3);
    }
    .btn-cancel:hover { background: rgba(255,107,107,0.3); }
    .btn-refresh {
      background: rgba(94,211,243,0.1); color: var(--cyan); border: 1px solid rgba(94,211,243,0.2);
      padding: 6px 14px; font-size: 12px;
    }
    .btn-refresh:hover { background: rgba(94,211,243,0.2); }
    .deploy-progress {
      margin-top: 14px; padding: 14px; border-radius: 14px;
      background: rgba(5,8,14,0.8); border: 1px solid rgba(255,255,255,0.08);
    }
    .deploy-progress .step-indicator {
      display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
      font-size: 14px; font-weight: 600;
    }
    .spinner {
      width: 16px; height: 16px; border: 2px solid rgba(94,211,243,0.3);
      border-top-color: var(--cyan); border-radius: 50%;
      animation: spin 0.8s linear infinite; display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .deploy-log {
      max-height: 400px; overflow-y: auto; padding: 10px; border-radius: 10px;
      background: rgba(0,0,0,0.4); font-size: 12px; line-height: 1.5;
      color: #b8d4c8; white-space: pre-wrap; word-break: break-all;
    }
    .result-banner {
      padding: 10px 16px; border-radius: 10px; margin-top: 10px;
      font-weight: 600; font-size: 14px;
    }
    .result-banner.success { background: rgba(47,210,127,0.15); color: var(--good); border: 1px solid rgba(47,210,127,0.3); }
    .result-banner.failed { background: rgba(255,107,107,0.15); color: var(--bad); border: 1px solid rgba(255,107,107,0.3); }
    @media (max-width: 960px) {
      .wrap { padding: 16px; }
      th:first-child, td:first-child { width: 118px; min-width: 118px; }
      .endpoint-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) { .endpoint-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-main">
        <div>
          <h1>rz-rpc-llm</h1>
          <p>Management portal</p>
        </div>
        <div class="hero-status" id="hero-status">No model deployed</div>
      </div>
      <div class="status-banner" id="dgx-offline-banner">
        <div class="status-banner-copy">
          <div class="status-banner-title">DGX Offline</div>
          <div class="status-banner-text" id="dgx-offline-text">RPC server offline.</div>
        </div>
        <button class="btn btn-refresh" id="retry-dgx-btn" onclick="retryDgxSsh()">Retry DGX SSH</button>
      </div>
      <div class="meta" id="meta"></div>
    </section>

    <!-- Deploy Panel -->
    <section class="panel deploy-section" id="deploy-panel">
      <h2>Deploy</h2>
      <div id="model-list" class="model-cards"></div>
      <div class="deploy-controls" id="deploy-controls">
            <label id="mode-label">Mode:
          <select id="mode-select">
            <option value="">Auto (default)</option>
            <option value="solo">Solo (Mac only)</option>
            <option value="distributed">Distributed (Mac + DGX)</option>
          </select>
        </label>
        <label id="vision-label" style="display:none">
          <input type="checkbox" id="vision-check"> Vision
        </label>
        <label>Tag:
          <select id="tag-select"><option value="">Loading...</option></select>
        </label>
        <button class="btn btn-refresh" id="refresh-tags-btn" onclick="loadTags()">Refresh Tags</button>
        <details id="advanced">
          <summary>Advanced Options</summary>
          <div class="advanced-grid">
            <label>Context: <input type="number" id="ctx-input" placeholder="default" style="width:90px"></label>
            <label>Parallel: <input type="number" id="parallel-input" placeholder="default" style="width:70px"></label>
            <div id="distributed-opts" style="display:none">
              <label>Split mode:
                <select id="split-mode-select">
                  <option value="layer">layer</option>
                  <option value="row">row</option>
                  <option value="none">none</option>
                </select>
              </label>
              <label>Tensor split (DGX:Mac):
                <select id="tensor-split-select">
                  <option value="0,20">0:100</option>
                  <option value="1,19">5:95</option>
                  <option value="2,18">10:90</option>
                  <option value="3,17">15:85</option>
                  <option value="4,16">20:80</option>
                  <option value="5,15">25:75</option>
                  <option value="6,14">30:70</option>
                  <option value="7,13">35:65</option>
                  <option value="8,12">40:60</option>
                  <option value="9,11">45:55</option>
                  <option value="10,10">50:50</option>
                  <option value="11,9">55:45</option>
                  <option value="12,8">60:40</option>
                  <option value="13,7">65:35</option>
                  <option value="14,6">70:30</option>
                  <option value="15,5">75:25</option>
                  <option value="16,4">80:20</option>
                  <option value="17,3">85:15</option>
                  <option value="18,2">90:10</option>
                  <option value="19,1">95:5</option>
                  <option value="20,0">100:0</option>
                </select>
              </label>
            </div>
            <label><input type="checkbox" id="skip-clone"> Skip clone</label>
            <label><input type="checkbox" id="skip-build"> Skip build</label>
            <label><input type="checkbox" id="skip-download"> Skip download</label>
          </div>
        </details>
      </div>
      <div style="display:flex;gap:12px;align-items:center">
        <button class="btn btn-deploy" id="deploy-btn" onclick="startDeploy()">Deploy</button>
        <button class="btn btn-cancel" id="undeploy-btn" onclick="undeploy()">Undeploy</button>
        <span id="deploy-hint" style="font-size:12px;color:var(--muted)">Select a model to deploy</span>
      </div>
      <div id="deploy-progress" style="display:none" class="deploy-progress">
        <div class="step-indicator">
          <span class="spinner" id="deploy-spinner"></span>
          <span id="deploy-step">Initializing...</span>
          <button class="btn btn-cancel" onclick="cancelDeploy()">Cancel</button>
        </div>
        <div class="deploy-log" id="deploy-log"></div>
        <div id="deploy-result"></div>
      </div>
    </section>

    <!-- Monitor -->
    <section class="cards" id="cards"></section>
    <section class="panel edge">
      <h2>Rolling History</h2>
      <div class="history-note">20 minute view with denser samples near the present.</div>
      <div class="history"><table id="history"></table></div>
    </section>
    <section class="panel" style="margin-top: 16px;">
      <h2>Live Log</h2>
      <pre class="log" id="log"></pre>
    </section>
    <section class="panel" style="margin-top: 16px;">
      <h2>Endpoints</h2>
      <div class="endpoint-list endpoint-grid" id="endpoints"></div>
    </section>
    <div class="footer" id="footer"></div>
  </div>
    <script>
    const PORTAL_SOLO_ONLY = __PORTAL_SOLO_ONLY__;
    const HISTORY_WINDOW_SECONDS = 20 * 60;
    let lastData = null;
    let selectedModel = null;
    let modelsData = [];
    let eventSource = null;

    // --- Model loading ---
    async function loadModels() {
      try {
        const res = await fetch('/api/models');
        modelsData = await res.json();
        const container = document.getElementById('model-list');
        container.innerHTML = modelsData.map(m => `
          <div class="model-card" data-id="${m.id}" onclick="selectModel('${m.id}')">
            <div class="model-name">${m.display_name || m.id}</div>
            <div class="model-meta">
              <span class="badge">${m.default_mode}</span>
              <span class="badge ${m.solo === 'yes' ? 'yes' : 'no'}">solo: ${m.solo}</span>
              <span class="badge ${m.vision === 'yes' ? 'yes' : 'no'}">vision: ${m.vision}</span>
              <span class="badge">ctx: ${Number(m.ctx_size || 0).toLocaleString()}</span>
              <span class="badge">parallel: ${m.parallel || '?'}</span>
            </div>
          </div>
        `).join('');
      } catch (e) { console.error('loadModels:', e); }
    }

    function applyPortalMode() {
      if (!PORTAL_SOLO_ONLY) return;
      document.getElementById('mode-label').style.display = 'none';
      document.getElementById('distributed-opts').style.display = 'none';
      document.getElementById('dgx-offline-banner').classList.remove('visible');
      const distributedOpt = document.querySelector('#mode-select option[value="distributed"]');
      if (distributedOpt) distributedOpt.remove();
      document.getElementById('mode-select').value = 'solo';
    }

    function selectModel(id) {
      selectedModel = id;
      document.querySelectorAll('.model-card').forEach(c => c.classList.toggle('selected', c.dataset.id === id));
      const m = modelsData.find(x => x.id === id);
      if (!m) return;
      // Update mode options
      const modeSelect = document.getElementById('mode-select');
      const soloOpt = modeSelect.querySelector('[value=solo]');
      if (soloOpt) soloOpt.disabled = m.solo !== 'yes';
      if (PORTAL_SOLO_ONLY) {
        modeSelect.value = 'solo';
      }
      // Set mode to model's default
      else if (m.default_mode === 'solo') modeSelect.value = 'solo';
      else if (m.default_mode === 'distributed') modeSelect.value = 'distributed';
      else modeSelect.value = '';
      // Vision toggle
      const visionLabel = document.getElementById('vision-label');
      visionLabel.style.display = m.vision === 'yes' ? '' : 'none';
      document.getElementById('vision-check').checked = m.vision === 'yes';
      // Populate defaults in advanced options
      document.getElementById('ctx-input').value = m.ctx_size || '';
      document.getElementById('ctx-input').placeholder = m.ctx_size || 'default';
      document.getElementById('parallel-input').value = m.parallel || '';
      document.getElementById('parallel-input').placeholder = m.parallel || 'default';
      // Distributed options
      updateDistributedOpts(modeSelect.value, m);
      // Reset tag to current and skip options
      const curOpt = Array.from(document.getElementById('tag-select').options).find(o => o.textContent.startsWith('[current]'));
      document.getElementById('tag-select').value = curOpt ? curOpt.value : '';
      document.getElementById('skip-clone').checked = false;
      document.getElementById('skip-build').checked = false;
      document.getElementById('skip-download').checked = false;
      document.getElementById('deploy-hint').textContent = `Ready to deploy ${m.display_name || m.id}`;
      document.getElementById('deploy-hint').style.color = 'var(--muted)';
      document.getElementById('deploy-btn').disabled = false;
    }

    function updateDistributedOpts(mode, m) {
      const show = !PORTAL_SOLO_ONLY && mode === 'distributed';
      document.getElementById('distributed-opts').style.display = show ? '' : 'none';
      if (show) {
        document.getElementById('split-mode-select').value = m.split_mode || 'layer';
        const ts = document.getElementById('tensor-split-select');
        const modelTs = m.tensor_split || '2,3';
        // Find closest ratio option
        const parts = modelTs.split(',').map(Number);
        const pct = Math.round((parts[0] / (parts[0] + parts[1])) * 20);
        const bestVal = `${pct},${20 - pct}`;
        const opt = Array.from(ts.options).find(o => o.value === bestVal);
        ts.value = opt ? bestVal : '8,12';
      }
    }

    document.getElementById('mode-select').addEventListener('change', function() {
      const m = modelsData.find(x => x.id === selectedModel);
      if (m) updateDistributedOpts(this.value || m.default_mode, m);
    });

    // --- Tag loading ---
    async function loadTags() {
      const btn = document.getElementById('refresh-tags-btn');
      try {
        btn.textContent = 'Loading...';
        btn.disabled = true;
        const res = await fetch('/api/tags');
        const tags = await res.json();
        const sel = document.getElementById('tag-select');
        const current = tags.find(t => t.current);
        sel.innerHTML = tags.map(t => `<option value="${t.tag}"${t.current ? ' selected' : ''}>${t.label}</option>`).join('');
        if (!current && tags.length) sel.value = tags[0].tag;
        btn.textContent = 'Refresh Tags';
        btn.disabled = false;
      } catch (e) {
        console.error('loadTags:', e);
        btn.textContent = 'Refresh Tags';
        btn.disabled = false;
      }
    }

    // --- Deployment ---
    async function startDeploy() {
      if (!selectedModel) return;
      const m = modelsData.find(x => x.id === selectedModel);
      const mode = PORTAL_SOLO_ONLY ? 'solo' : document.getElementById('mode-select').value;
      const vision = document.getElementById('vision-check').checked;
      const tag = document.getElementById('tag-select').value;
      const modeLabel = mode || m.default_mode || 'auto';
      if (!confirm(`Deploy ${m.display_name || m.id} in ${modeLabel} mode?`)) return;

      const body = {
        model: selectedModel,
        mode: mode || m.default_mode || '',
        vision: vision,
        tag: tag,
        ctx: document.getElementById('ctx-input').value || '',
        parallel: document.getElementById('parallel-input').value || '',
        split_mode: (!PORTAL_SOLO_ONLY && (mode || m.default_mode) === 'distributed') ? (document.getElementById('split-mode-select').value || '') : '',
        tensor_split: (!PORTAL_SOLO_ONLY && (mode || m.default_mode) === 'distributed') ? (document.getElementById('tensor-split-select').value || '') : '',
        skip_clone: document.getElementById('skip-clone').checked,
        skip_build: document.getElementById('skip-build').checked,
        skip_download: document.getElementById('skip-download').checked,
      };

      try {
        const res = await fetch('/api/deploy', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
        if (res.status === 409) { alert('A deployment is already in progress.'); return; }
        if (!res.ok) { alert('Deploy failed to start'); return; }
        showDeployProgress();
        streamDeployLogs();
      } catch (e) { alert('Deploy error: ' + e); }
    }

    function showDeployProgress() {
      document.getElementById('deploy-progress').style.display = '';
      document.getElementById('deploy-log').textContent = '';
      document.getElementById('deploy-result').innerHTML = '';
      document.getElementById('deploy-btn').disabled = true;
      document.getElementById('deploy-spinner').style.display = '';
    }

    function streamDeployLogs() {
      if (eventSource) eventSource.close();
      eventSource = new EventSource('/api/deploy/stream');
      const logEl = document.getElementById('deploy-log');
      eventSource.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.done) {
          eventSource.close();
          eventSource = null;
          document.getElementById('deploy-spinner').style.display = 'none';
          document.getElementById('deploy-btn').disabled = false;
          const resultEl = document.getElementById('deploy-result');
          if (data.status === 'success') {
            resultEl.innerHTML = '<div class="result-banner success">Deployment successful</div>';
            // Auto-collapse after success
            setTimeout(() => {
              document.getElementById('deploy-progress').style.display = 'none';
              document.getElementById('deploy-hint').textContent = 'Last deployment succeeded';
              document.getElementById('deploy-hint').style.color = 'var(--green, #22c55e)';
            }, 3000);
          } else {
            resultEl.innerHTML = '<div class="result-banner failed">Deployment failed</div>';
          }
          return;
        }
        if (data.step) document.getElementById('deploy-step').textContent = data.step;
        if (data.line !== undefined) {
          logEl.textContent += data.line + '\n';
          logEl.scrollTop = logEl.scrollHeight;
        }
      };
      eventSource.onerror = () => {
        // Auto-reconnect is built into EventSource; if terminal, close
        if (eventSource && eventSource.readyState === 2) {
          eventSource.close();
          eventSource = null;
          document.getElementById('deploy-spinner').style.display = 'none';
          document.getElementById('deploy-btn').disabled = false;
        }
      };
    }

    async function cancelDeploy() {
      if (!confirm('Cancel the running deployment?')) return;
      await fetch('/api/deploy/cancel', { method: 'POST' });
    }

    async function undeploy() {
      const prompt = PORTAL_SOLO_ONLY ? 'Stop llama-server?' : 'Stop llama-server and rpc-server?';
      if (!confirm(prompt)) return;
      document.getElementById('undeploy-btn').disabled = true;
      document.getElementById('undeploy-btn').textContent = 'Stopping...';
      try {
        const res = await fetch('/api/undeploy', { method: 'POST' });
        const data = await res.json();
        document.getElementById('deploy-hint').textContent = data.message || 'Servers stopped';
        document.getElementById('deploy-hint').style.color = 'var(--muted)';
      } catch (e) {
        alert('Undeploy error: ' + e);
      }
      document.getElementById('undeploy-btn').disabled = false;
      document.getElementById('undeploy-btn').textContent = 'Undeploy';
    }

    async function retryDgxSsh() {
      const btn = document.getElementById('retry-dgx-btn');
      const hint = document.getElementById('deploy-hint');
      const bannerText = document.getElementById('dgx-offline-text');
      btn.disabled = true;
      btn.textContent = 'Retrying...';
      hint.textContent = 'Retrying DGX SSH...';
      hint.style.color = 'var(--muted)';
      bannerText.textContent = 'Retrying DGX SSH...';
      try {
        const res = await fetch('/api/dgx/retry-ssh', { method: 'POST' });
        const data = await res.json();
        hint.textContent = data.message || 'DGX SSH retry finished';
        hint.style.color = data.connected ? 'var(--green, #22c55e)' : 'var(--bad, #ff6b6b)';
        bannerText.textContent = data.message || 'DGX SSH retry finished';
      } catch (e) {
        hint.textContent = 'DGX SSH retry failed';
        hint.style.color = 'var(--bad, #ff6b6b)';
        bannerText.textContent = 'RPC server offline. Retry DGX SSH failed.';
      }
      btn.disabled = false;
      btn.textContent = 'Retry DGX SSH';
    }

    // --- Check deploy status on load (reconnect if mid-deploy) ---
    async function checkDeployStatus() {
      try {
        const res = await fetch('/api/status');
        const st = await res.json();
        if (st.status === 'deploying') {
          showDeployProgress();
          document.getElementById('deploy-step').textContent = st.step || 'deploying...';
          streamDeployLogs();
        }
      } catch (e) {}
    }

    // --- Monitor (existing logic) ---
    const baseRows = [
      ["llama-server", "llama_server"],
      ["rpc-server", "rpc_server", "DISTRIBUTED"],
      ["Mac RAM used", "mac_ram"],
      ["DGX RAM used", "dgx_ram", "DISTRIBUTED"],
      ["Mac GPU util", "mac_gpu"],
      ["DGX GPU util", "dgx_gpu", "DISTRIBUTED"],
      ["prompt tokens", "prompt_tokens"],
      ["gen tokens", "gen_tokens"],
      ["pp (t/s)", "pp"],
      ["tg (t/s)", "tg"],
      ["reqs", "reqs"],
    ];

    let maxSlotsSeen = 0;
    function buildRows(latest) {
      const rows = [...baseRows];
      const nSlots = latest.n_slots || 0;
      if (nSlots > maxSlotsSeen) maxSlotsSeen = nSlots;
      for (let i = 0; i < maxSlotsSeen; i++) {
        rows.push(["ctx s" + i, "slot_ctx_" + i]);
      }
      return rows;
    }

    function stateClass(value, key) {
      if (value === "UP") return "good";
      if (value === "DOWN") return "bad";
      if (key && key.startsWith("slot_ctx_")) {
        const s = String(value).trim();
        if (s === "--") return "";
        let tokens = 0;
        const mK = s.match(/(\d+)K/);
        const mM = s.match(/(\d+)M/);
        if (mK) tokens = Number(mK[1]) * 1000;
        else if (mM) tokens = Number(mM[1]) * 1000000;
        else tokens = Number(s) || 0;
        if (tokens < 30000) return "good";
        if (tokens < 80000) return "warn";
        return "bad";
      }
      if (typeof value === "string" && value.includes("%")) {
        const match = value.match(/(\d+)%/);
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
      return `<div class="card"><div class="label">${label}</div><div class="value ${String(value).length > 9 ? "small" : ""} ${stateClass(value)}">${value}</div></div>`;
    }

    function historyColumnCount() {
      if (window.innerWidth < 480) return 8;
      if (window.innerWidth < 640) return 12;
      if (window.innerWidth < 960) return 24;
      return 32;
    }

    function historyBuckets(history, count) {
      const windowSec = count >= 24 ? 20 * 60 : 5 * 60;
      const formatAge = (seconds) => {
        if (seconds >= 60) return `-${Math.round(seconds / 60)}m`;
        return `-${Math.round(seconds)}s`;
      };
      if (!history.length) {
        return Array.from({ length: count }, (_, index) => ({
          label: formatAge((((count - index) / count) ** 2) * windowSec),
          empty: true,
        }));
      }
      const now = history[history.length - 1].captured_at;
      const cutoff = now - windowSec;
      const filtered = history.filter(e => e.captured_at >= cutoff);
      if (filtered.length <= count) {
        return filtered.map((entry) => ({
          ...entry,
          label: formatAge(Math.max(0, Math.round(now - entry.captured_at))),
        }));
      }
      return Array.from({ length: count }, (_, index) => {
        const progress = count === 1 ? 1 : index / (count - 1);
        const weighted = 1 - ((1 - progress) ** 2);
        const sourceIndex = Math.max(0, Math.min(filtered.length - 1, Math.round(weighted * (filtered.length - 1))));
        const entry = filtered[sourceIndex];
        return {
          ...entry,
          label: formatAge(Math.max(0, Math.round(now - entry.captured_at))),
        };
      });
    }

    function renderMonitor(data) {
      const latest = data.latest;
      const dgxBannerEl = document.getElementById('dgx-offline-banner');
      const dgxBannerTextEl = document.getElementById('dgx-offline-text');
      const dgxOffline = !PORTAL_SOLO_ONLY && latest.mode === 'DISTRIBUTED' && latest.rpc_server !== 'UP';
      const currentModelId = latest.serving_model || latest.backend_model || '';
      const currentModel = currentModelId
        ? modelsData.find(x => x.id === currentModelId || x.alias === currentModelId)
        : null;
      // Hero status — current deployed model
      const heroEl = document.getElementById("hero-status");
      if (currentModelId && latest.llama_server === "UP") {
        const p = latest.serving_params || {};
        const name = (currentModel && currentModel.display_name) || currentModelId;
        const parts = [`<span class="dot good"></span><span class="model-name">${name}</span>`];
        parts.push(`${latest.mode.toLowerCase()} · ${latest.n_slots || '?'} slots`);
        if (p.vision) parts.push('vision');
        if (p.ctx) parts.push(`ctx ${Number(p.ctx).toLocaleString()}`);
        if (latest.mode === 'DISTRIBUTED' && p.tensor_split) parts.push(`split ${p.tensor_split}`);
        heroEl.innerHTML = parts.join('<br>');
      } else if (latest.llama_server === "UP") {
        heroEl.innerHTML = `<span class="dot good"></span><span class="model-name">Model running</span><br>${latest.mode.toLowerCase()} · ${latest.n_slots || '?'} slots`;
      } else if (latest.llama_server === "DOWN") {
        heroEl.innerHTML = '<span class="dot bad"></span>No model running';
      } else {
        heroEl.innerHTML = '<span class="dot"></span>Checking...';
      }

      document.getElementById("meta").innerHTML = [
        ["mode", latest.mode],
        ["updated", latest.timestamp],
        ["llama", latest.llama_server],
        ["rpc", latest.rpc_server],
      ].filter(([k]) => !PORTAL_SOLO_ONLY || k !== 'rpc')
       .map(([k, v]) => `<div class="pill"><span class="dot ${stateClass(v)}"></span>${k}: ${v}</div>`).join("");

      if (dgxOffline) {
        dgxBannerEl.classList.add('visible');
        dgxBannerTextEl.textContent = latest.rpc_server === 'DOWN'
          ? 'RPC server offline. Retry DGX SSH after the Spark comes back.'
          : `RPC server status: ${latest.rpc_server}. Retry DGX SSH to restore the DGX connection.`;
      } else {
        dgxBannerEl.classList.remove('visible');
      }

      document.getElementById("cards").innerHTML = [
        card("pp (t/s)", latest.pp),
        card("tg (t/s)", latest.tg),
        card("reqs", latest.reqs),
        card("prompt tokens", latest.prompt_tokens),
        card("gen tokens", latest.gen_tokens),
        card("mac ram", latest.mac_ram),
        card("mac gpu", latest.mac_gpu),
        ...(latest.mode === "DISTRIBUTED" ? [card("dgx ram", latest.dgx_ram), card("dgx gpu util", latest.dgx_gpu)] : []),
      ].join("");

      const rows = buildRows(latest);
      const visibleRows = rows.filter((row) => !row[2] || row[2] === latest.mode);
      const rawHistory = data.history;
      const history = historyBuckets(rawHistory, historyColumnCount());
      document.getElementById("history").innerHTML =
        `<thead><tr><th>metric</th>${history.map((h) => `<th>${h.label}</th>`).join("")}</tr></thead>` +
        `<tbody>${visibleRows.map(([label, key]) => `<tr><td>${label}</td>${history.map((h) => { const v = h.empty ? "--" : (h[key] == null ? "--" : h[key]); return `<td class="${h.empty || v === "--" ? "" : stateClass(v, key)}">${v}</td>`; }).join("")}</tr>`).join("")}</tbody>`;

      document.getElementById("endpoints").innerHTML = Object.entries(latest.endpoints)
        .filter(([k, v]) => (!PORTAL_SOLO_ONLY || k !== 'rpc') && v !== '--')
        .map(([k, v]) => `<div class="endpoint-item"><div class="label">${k}</div><a href="${String(v).startsWith("http") ? v : "#"}" target="_blank">${v}</a></div>`)
        .join("");

      document.getElementById("log").textContent = latest.log_lines.join("\n") || "No llama-server log yet.";
      document.getElementById("footer").textContent = `Single-port gateway \u2022 polling /monitor/api every 3s \u2022 ${rawHistory.length} raw samples \u2022 ${history.length} visible buckets`;
    }

    async function refresh() {
      try {
        const res = await fetch('/monitor/api', { cache: 'no-store' });
        const data = await res.json();
        lastData = data;
        renderMonitor(data);
      } catch (err) {
        document.getElementById("footer").textContent = `Monitor fetch failed: ${err}`;
      }
    }

    // --- Init ---
    document.getElementById('deploy-btn').disabled = true;
    applyPortalMode();
    loadModels();
    loadTags();
    checkDeployStatus();
    refresh();
    setInterval(refresh, 3000);
    window.addEventListener("resize", () => { if (lastData) renderMonitor(lastData); });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Proxy + /v1/models interception
# ---------------------------------------------------------------------------

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

    # Intercept /v1/models to inject rz-llm-default alias
    if handler.command == "GET" and handler.path.rstrip("/") == "/v1/models":
        try:
            models_json = json.loads(payload)
            if isinstance(models_json.get("data"), list) and models_json["data"]:
                alias = dict(models_json["data"][0])
                alias["id"] = "rz-llm-default"
                alias["owned_by"] = "roundzero-ai"
                models_json["data"].append(alias)
                payload = json.dumps(models_json).encode("utf-8")
        except Exception:
            pass

    handler.send_response(response.status, response.reason)
    excluded = {"connection", "transfer-encoding", "content-length", "server", "date"}
    for key, value in response.getheaders():
        if key.lower() not in excluded:
            handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(payload)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

def _json_response(handler, data, status=200):
    body = json.dumps(data, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler, html):
    body = html.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"

        # Portal
        if path in ("/", "/monitor"):
            _html_response(self, PORTAL_HTML.replace("__PORTAL_SOLO_ONLY__", "true" if PORTAL_SOLO_ONLY else "false"))
            return

        # Monitor API (unchanged)
        if path == "/monitor/api":
            latest = take_snapshot()
            _json_response(self, {"latest": latest, "history": list(HISTORY)})
            return

        # Portal API
        if path == "/api/models":
            _json_response(self, parse_models_conf())
            return

        if path == "/api/tags":
            _json_response(self, get_llama_tags())
            return

        if path == "/api/status":
            _json_response(self, DEPLOY.to_dict())
            return

        if path == "/api/deploy/stream":
            handle_deploy_stream(self)
            return

        # Proxy everything else
        proxy_request(self)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/api/deploy":
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                params = json.loads(raw)
            except Exception:
                _json_response(self, {"error": "invalid JSON"}, 400)
                return
            if not params.get("model"):
                _json_response(self, {"error": "model required"}, 400)
                return
            model_conf = get_model_conf(params["model"])
            if not model_conf:
                _json_response(self, {"error": "unknown model"}, 400)
                return
            if PORTAL_SOLO_ONLY:
                if model_conf.get("solo") != "yes":
                    _json_response(self, {"error": "model is not available in managed solo mode"}, 400)
                    return
                params["mode"] = "solo"
            deploy_id = str(uuid.uuid4())[:8]
            if not DEPLOY.start(params["model"], deploy_id, params):
                _json_response(self, {"error": "deployment already in progress"}, 409)
                return
            thread = threading.Thread(target=run_deployment, args=(params,), daemon=True)
            thread.start()
            _json_response(self, {"deploy_id": deploy_id, "status": "deploying"}, 202)
            return

        if path == "/api/deploy/cancel":
            DEPLOY.cancel()
            _json_response(self, {"status": "cancelled"})
            return

        if path == "/api/undeploy":
            stopped = []
            env = dict(os.environ, TERM="dumb", SKIP_MONITOR_RESTART="1")
            steps = [("llama-server", "stop-llama")]
            if not PORTAL_SOLO_ONLY:
                steps.append(("rpc-server", "stop-rpc"))
            for name, step in steps:
                try:
                    r = subprocess.run(
                        ["bash", DEPLOY_SCRIPT, "debug", step],
                        capture_output=True, text=True, timeout=60, env=env,
                    )
                    if r.returncode == 0:
                        stopped.append(name)
                except Exception:
                    pass
            if stopped:
                DEPLOY.serving_model = ""
                DEPLOY.serving_params = {}
            msg = f"Stopped: {', '.join(stopped)}" if stopped else "No servers were running"
            _json_response(self, {"status": "ok", "stopped": stopped, "message": msg})
            return

        if path == "/api/dgx/retry-ssh":
            connected, message = retry_dgx_ssh()
            _json_response(self, {
                "status": "ok" if connected else "offline",
                "connected": connected,
                "message": message,
            }, 200 if connected else 503)
            return

        # Proxy everything else
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
    print(f"monitor-web gateway listening on http://{host}:{port}/ -> {BACKEND_BASE_URL}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
