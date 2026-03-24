> Load this when planning or implementing SGLang Docker backend support.

# Expansion Plan: SGLang Docker Backend

## Context

SGLang is ~2x faster than llama.cpp for Qwen3.5 inference (proven in rz-mcp-web). This plan extends the existing deployment platform to support SGLang as a second backend, running in Docker on DGX Spark (same machine used for RPC), while the Mac Studio remains the control plane (deploy.sh, monitor_web.py, portal).

**Target model**: `AxionML/Qwen3.5-122B-A10B-NVFP4` (~75.6GB, fits DGX Spark 128GB)
**Key flag**: `--quantization modelopt_fp4`
**Risk**: NVFP4 has reported ARM64/Grace compatibility issues. Fallback: change to FP8 quantization in models.conf (no code change needed).

---

## Architecture After Change

```
Client --- HTTP :8680 ---> Mac Studio (control plane)
                             |
                             +- monitor_web.py gateway (:8680)
                             |   +- proxies /v1/*, /health, /metrics to ACTIVE backend
                             |   +- serves portal UI + monitor dashboard
                             |   +- injects rz-llm-default model alias
                             |   +- orchestrates deploy/undeploy via deploy.sh
                             |
                             +--- Backend A: llama-server (local :8682)
                             |     for backend=llama models (existing)
                             |
                             +--- Backend B: SGLang Docker (remote DGX_HOST:30000)
                                   for backend=sglang models (new)
                                   managed via SSH -> docker compose up/down
```

One model at a time. Stop-all-before-deploy ensures clean backend switching.

---

## Files to Change

### 1. `compose.sglang.yml` (NEW — project root)

Docker Compose file for SGLang, pushed to DGX at deploy time via SCP.

```yaml
services:
  sglang:
    image: ${SGLANG_IMAGE}
    container_name: rz-sglang
    command: >-
      python3 -m sglang.launch_server
        --model-path ${SGLANG_MODEL}
        --quantization ${SGLANG_QUANTIZATION:-}
        --host 0.0.0.0 --port ${SGLANG_PORT:-30000}
        --attention-backend ${SGLANG_ATTENTION_BACKEND:-triton}
        --mem-fraction-static ${SGLANG_MEM_FRACTION:-0.90}
        --enable-metrics
        --trust-remote-code
    environment:
      HF_TOKEN: ${HF_TOKEN:-}
      NVIDIA_VISIBLE_DEVICES: all
    volumes:
      - ${HF_HOME:-/home/${DGX_USER}/.cache/huggingface}:/root/.cache/huggingface
    ports:
      - "${SGLANG_PORT:-30000}:${SGLANG_PORT:-30000}"
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:${SGLANG_PORT:-30000}/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 300s
    restart: unless-stopped
```

Parameterized via `.env` file created on DGX during deployment.

### 2. `models.conf` — Add `backend` field + new model

- Add `backend = sglang` to new model section `[qwen3.5-122b-nvfp4]`
- Existing models implicitly default to `backend = llama` (no change needed to them)
- SGLang-specific fields use `sglang_` prefix: `sglang_image`, `sglang_quantization`, `sglang_port`, `sglang_attention_backend`, `sglang_mem_fraction`, `sglang_reasoning_parser`

Example entry:
```ini
[qwen3.5-122b-nvfp4]
display_name = Qwen3.5-122B-A10B (NVFP4)
backend = sglang
repo = AxionML/Qwen3.5-122B-A10B-NVFP4
alias = qwen3.5-122b-nvfp4
default_mode = sglang
solo = no
vision = no
sglang_image = lmsysorg/sglang:latest
sglang_quantization = modelopt_fp4
sglang_port = 30000
sglang_attention_backend = triton
sglang_mem_fraction = 0.90
sglang_reasoning_parser = qwen3
ctx_size = 131072
parallel = 1
```

### 3. `defaults.env` — Add SGLang infrastructure defaults

```bash
# --- SGLang Docker (DGX Spark, GPU inference) ---
SGLANG_PORT=30000
SGLANG_IMAGE="lmsysorg/sglang:latest"
SGLANG_COMPOSE_DIR="/home/${DGX_USER}/rz-rpc-llm/sglang"
SGLANG_HEALTH_TIMEOUT=600
```

### 4. `deploy.sh` — New SGLang lifecycle commands

**New functions:**
- `cmd_start_sglang --model NAME`: Push compose file + .env to DGX via SCP, pull image, `docker compose up -d`, poll health at `http://DGX_HOST:PORT/health` (up to 10 min), write backend marker files
- `cmd_stop_sglang`: SSH to DGX, `docker compose down`, clean up marker files

**Backend marker files** (`.pids/active-backend.type` and `.pids/active-backend.url`):
- Written by both `cmd_start_sglang` (type=sglang, url=http://DGX:PORT) and the existing `_launch_llama_server` (type=llama, url=http://127.0.0.1:8682)
- Read by `cmd_start_monitor_web` to set `BACKEND_BASE_URL` dynamically

**Modifications to existing functions:**
- `cmd_debug()`: Add `start-sglang` and `stop-sglang` cases
- `cmd_stop()`: Also stop SGLang container (safe no-op if not running)
- `cmd_managed()` trap: Also stop SGLang on Ctrl+C
- `cmd_status()`: Show SGLang container status on DGX
- `cmd_start_monitor_web()`: Read `active-backend.url` file to set `BACKEND_BASE_URL`
- `load_model_registry()`: Default `MODEL_BACKEND` to `llama` if not set

### 5. `monitor_web.py` — Dynamic backend routing + SGLang monitoring

**Backend routing** (currently static globals at lines 23-27):
- Replace static `BACKEND_BASE_URL` / `BACKEND_HOST` / `BACKEND_PORT` with thread-safe mutable state via `get_backend_url()` / `set_backend(url, type)` functions
- `proxy_request()`: Use `get_backend_parts()` instead of static `BACKEND_HOST`/`BACKEND_PORT`
- All 5 references to `BACKEND_BASE_URL` updated to use getter

**Deployment pipeline** (`run_deployment()`):
- Read `backend` from model config via `parse_models_conf()`
- If `backend == "sglang"`: skip clone/build/download, stop all backends (llama + rpc + sglang), then call `deploy.sh debug start-sglang --model NAME`, update backend URL via `set_backend()`
- If `backend == "llama"`: existing pipeline (also stops sglang before starting)

**Monitoring** (`take_snapshot()`):
- New `sglang_snapshot()`: health check + parse SGLang Prometheus metrics (different metric names than llama.cpp). No `/slots` endpoint — slots section hidden for SGLang
- `detect_mode()`: Return `"SGLANG"` when backend type is sglang
- `dgx_snapshot()`: Check `docker ps -f name=rz-sglang` instead of `pgrep rpc-server` when in SGLang mode
- `log_tail()`: Show hint to use `docker logs rz-sglang` on DGX (no local log file for SGLang)

**Undeploy** (`/api/undeploy`):
- Add `("sglang", "stop-sglang")` to the stop loop
- Reset backend to default llama URL after undeploy

**Portal UI** (PORTAL_HTML):
- Model cards: Show backend badge (`llama` vs `sglang`)
- Deploy form: Hide irrelevant controls for SGLang (mode selector, vision toggle, tag selector, skip checkboxes, tensor split/split mode). Auto-skip clone/build/download
- Monitor dashboard: Show `"SGLANG"` mode in hero status, hide slot rows, show DGX RAM/GPU stats
- Meta pills: Show `sglang: UP/DOWN` instead of `llama/rpc` when in SGLang mode

---

## Implementation Sequence

1. **Infrastructure**: `compose.sglang.yml`, `defaults.env`, `models.conf`
2. **deploy.sh**: `cmd_start_sglang`, `cmd_stop_sglang`, marker files, modify debug/stop/managed/status/monitor-web-start
3. **monitor_web.py backend routing**: Mutable backend state, proxy_request, set_backend in deployment
4. **monitor_web.py monitoring**: sglang_snapshot, detect_mode, dgx_snapshot, log_tail, take_snapshot
5. **monitor_web.py portal UI**: Model cards, deploy form, monitor dashboard adaptation

---

## Verification

1. **CLI smoke test**: `./deploy.sh debug start-sglang --model qwen3.5-122b-nvfp4` — verify container starts on DGX, health passes
2. **Proxy test**: `curl http://127.0.0.1:8680/v1/models` — should show SGLang model + `rz-llm-default` alias
3. **Chat test**: `curl -X POST http://127.0.0.1:8680/v1/chat/completions -d '{"model":"rz-llm-default","messages":[...]}'`
4. **Portal deploy**: Select NVFP4 model card, deploy from browser, verify streaming log
5. **Backend switch**: Deploy llama.cpp model -> deploy SGLang model -> switch back. Verify proxy routes correctly each time
6. **Undeploy**: Click Undeploy while SGLang running — verify Docker container stopped
7. **Ctrl+C**: In managed mode with SGLang running — verify cleanup stops container on DGX
8. **NVFP4 fallback**: If NVFP4 fails on ARM64, update `sglang_quantization = fp8` and `repo = Qwen/Qwen3.5-122B-A10B` in models.conf — no code change needed

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NVFP4 on ARM64 Grace CPU | CUDA illegal instruction crash | Change quantization to fp8 in models.conf (zero code change) |
| SGLang metric name differences | Monitor shows `--` for metrics | Log raw `/metrics` on first health check; resilient parsing with fallbacks |
| First-run model download (~75GB) | 30+ min before health passes | `SGLANG_HEALTH_TIMEOUT=600` + compose `start_period: 300s`; pre-download with `huggingface-cli` |
| Network latency for health/metrics polling | Slower monitor updates | 5s timeout for SGLang checks vs 2s for local llama.cpp |
| Concurrent backend state during switch | Race condition in proxy | Thread-safe `get_backend_url()` with `threading.Lock` |

---

## Reference: rz-mcp-web SGLang Experience

The rz-mcp-web project at `~/Workspace/rz-mcp-web` has a working SGLang Docker deployment for DGX Spark:
- Docker image: `lmsysorg/sglang:v0.5.9-cu130`
- Compose files: `docker/compose.sglang.yml`, env: `docker/env/dgx-spark.env`
- Deployment script: `docker/up-dgx-spark.sh`
- Key learnings in `doc/learnings.md` and `doc/benchmarking.md`
