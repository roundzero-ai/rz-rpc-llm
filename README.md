# rz-rpc-llm

A web-managed deployment system for running large language models via `llama.cpp` on a Mac Studio M2 Ultra (192 GB), with optional layer offloading to a DGX Spark GB10 via RPC.

A single command — `./deploy.sh managed` — starts a web portal where you can select models, deploy with one click, monitor hardware in real time, and switch between models without touching the terminal again. Models and their runtime parameters are defined in `models.conf`; adding a new model requires no code changes.

## Architecture

```text
Terminal                          Browser
  |                                |
  | ./deploy.sh managed            |
  |   1. prompt SSH password       |
  |   2. establish ControlMaster   |
  |   3. start monitor_web.py      |
  |   4. block (trap Ctrl+C)       |
  |                                |
  |                          http://HOST:8680/
  |                           Portal + Monitor
  |                                |
  |                          Deploy / Redeploy /
  |                          Undeploy from browser
  |                                |
  | Ctrl+C ────────────────────>   |
  |   stop llama-server            |
  |   stop rpc-server              |
  |   stop monitor-web             |
  |   exit                         |
```

```text
Client ─── HTTP :8680 ───> Mac Studio (Metal)
                             ├─ llama-server (backend on :8682)
                             ├─ monitor_web.py gateway (public on :8680)
                             │   ├─ proxies /v1, /health, /metrics to backend
                             │   ├─ serves portal UI at /
                             │   ├─ serves monitor dashboard at /monitor
                             │   ├─ injects rz-llm-default model alias
                             │   └─ orchestrates deploy/undeploy via deploy.sh
                             │
                             └── RPC :50052 (distributed mode only)
                                   └─> DGX Spark GB10 (CUDA)
                                         └─ rpc-server (remote GPU offload)
```

## Deployment modes

| Mode | Hardware | When |
|---|---|---|
| Solo | Mac Studio only | Model fits in 192 GB (e.g., Qwen3.5-122B-A10B, Qwen3.5-27B) |
| Distributed | Mac Studio + DGX Spark GB10 | Model needs both devices (e.g., MiniMax-M2.5) |

Each model in `models.conf` declares a default mode and whether solo is allowed. In distributed mode, the portal exposes tensor split ratio (DGX:Mac, 5% steps) and split mode (layer/row/none) controls.

## Available models

| Model | Size | Quant | Mode | Vision | Parallel |
|---|---|---|---|---|---|
| MiniMax-M2.5 | ~180 GB | UD-Q6_K_XL | distributed | no | 2 |
| Qwen3.5-122B-A10B | ~105 GB | UD-Q6_K_XL | solo | yes | 4 |
| Qwen3.5-35B-A3B | ~35 GB | Q8_0 | solo | yes | 4 |
| Qwen3.5-27B | ~17 GB | UD-Q4_K_XL | solo | yes | 8 |

## Repository layout

```text
rz-rpc-llm/
├── deploy.sh              # deployment CLI + managed mode
├── models.conf            # model registry (checked in)
├── monitor_web.py         # web portal, monitor gateway, deploy orchestrator
├── defaults.env           # infrastructure defaults (checked in)
├── config.env.example     # local override template
├── config.env             # optional local overrides (gitignored)
├── llama.cpp/             # cloned by deploy.sh (gitignored)
├── models/                # downloaded GGUF files (gitignored)
├── logs/                  # runtime logs (gitignored)
└── .pids/                 # pid files and SSH control socket (gitignored)
```

## Requirements

### Mac Studio

- macOS with Xcode Command Line Tools
- `cmake`, `python3`
- `huggingface-cli` (`pip install huggingface_hub`)

### DGX Spark GB10 (distributed mode only)

- SSH access from the Mac
- CUDA toolchain and `cmake`
- Port `50052` reachable from the Mac

### Hugging Face tokens

For gated model downloads, export a token in your shell:

```bash
export HF_TOKEN="hf_..."
```

Do not store tokens in repo config files.

## Quick start

```bash
git clone https://github.com/roundzero-ai/rz-rpc-llm.git
cd rz-rpc-llm
chmod +x deploy.sh
./deploy.sh managed
```

This starts the management portal:
1. Prompts for DGX SSH password in the terminal
2. Establishes a persistent SSH ControlMaster connection (auto-reconnects if dropped)
3. Starts the web portal at `http://HOST:8680/`
4. Blocks — Ctrl+C gracefully stops all services and exits

### Web portal

Open `http://127.0.0.1:8680/` in a browser to:

- **Select a model** — cards show each model's mode, vision support, context size, and parallel slots
- **Configure** — mode, vision, llama.cpp tag (with relative age), context, parallel slots; distributed mode adds tensor split ratio and split mode
- **Deploy** — one click with confirmation; streaming progress log shows each pipeline step (clone, build, download, start)
- **Undeploy** — gracefully stops llama-server and rpc-server with full memory reclaim
- **Switch models** — deploying a new model automatically stops the previous deployment first
- **Monitor** — live hardware stats, server health, rolling history, per-slot KV tracking, log tail

The portal is responsive: on mobile, rolling history adapts to fewer columns with a shorter (5-minute) window.

### CLI alternative

For scripted or headless use, `run` drives the same pipeline from the terminal:

```bash
# Vision model (Qwen3.5-122B, Mac solo)
./deploy.sh run --model qwen3.5 --vision

# Distributed text (MiniMax, Mac + DGX)
./deploy.sh run --model minimax

# Skip steps on subsequent runs
./deploy.sh run --model qwen3.5 --vision --skip-clone --skip-build
```

## Commands

```bash
./deploy.sh <command> [options]
```

| Command | Purpose |
|---|---|
| `managed` | **Recommended.** Web portal for deploy, monitor, and redeploy |
| `run --model NAME [flags]` | CLI pipeline: clone, build, download, start, terminal monitor |
| `stop` | Stop all services (llama-server, rpc-server, monitor-web) |
| `status` | Show running process status |
| `logs [llama\|rpc\|monitor]` | Tail logs |
| `monitor [INTERVAL] [WIDTH]` | Terminal heartbeat monitor |
| `models` | List available models from `models.conf` |
| `debug <step> [args...]` | Run individual pipeline steps |

### `run` flags

| Flag | Description |
|---|---|
| `--model NAME` | Required. Model key from `models.conf` (case-insensitive, prefix match) |
| `--vision` | Enable vision mode (model must have `vision = yes`) |
| `--solo` | Force solo mode |
| `--distributed` | Force distributed mode |
| `--tag TAG` | Pin a specific `llama.cpp` tag |
| `--ctx N` | Override context size |
| `--parallel N` | Override parallel slots |
| `--skip-clone` | Skip clone/fetch step |
| `--skip-build` | Skip build step |
| `--skip-download` | Skip model download |

### `debug` steps

For manual control during development:

```bash
./deploy.sh debug clone --tag latest
./deploy.sh debug build-mac
./deploy.sh debug build-dgx
./deploy.sh debug download --model qwen3.5-27b --vision
./deploy.sh debug start-rpc
./deploy.sh debug start-llama --model minimax --distributed
```

## Configuration

### Three layers

1. **`defaults.env`** (checked in) — infrastructure: DGX connection, listen ports, global performance flags
2. **`config.env`** (gitignored) — local overrides for your specific setup
3. **`models.conf`** (checked in) — per-model runtime parameters

Model-specific settings (threads, context size, sampling, tensor split) live in `models.conf`, not in env files.

### Adding a new model

Add a section to `models.conf`:

```ini
[my-new-model]
display_name = My New Model
repo = huggingface/repo-name
pattern = *Q6_K*
file = Q6_K/model-file-00001-of-00003.gguf
alias = my-model
default_mode = solo
solo = yes
vision = no
threads = 8
threads_batch = 8
ctx_size = 131072
parallel = 4
batch_size = 2048
ubatch_size = 512
n_gpu_layers = all
split_mode = none
cache_type_k = q8_0
cache_type_v = q8_0
temp = 1.0
top_p = 0.95
top_k = 40
min_p = 0.01
repeat_penalty = 1.0
presence_penalty = 0.0
reasoning_format = auto
```

For vision models, add `mm_proj` and `mm_proj_pattern`:

```ini
vision = yes
mm_proj = mmproj-BF16.gguf
mm_proj_pattern = mmproj*.gguf
```

For distributed models, add `split_mode` and `tensor_split`:

```ini
default_mode = distributed
solo = no
split_mode = layer
tensor_split = 2,3
```

The model appears in the portal immediately — no restart needed (the portal reads `models.conf` on each request).

### Local overrides

```bash
cp config.env.example config.env
```

Typical overrides:

```bash
DGX_HOST="192.168.88.66"
DGX_USER="your_user"
```

Overrides are reported at startup so drift is visible.

## API

All endpoints are served through a single port (`:8680`). The monitor gateway proxies OpenAI-compatible requests to the llama-server backend (`:8682`) and serves the portal and management API directly.

### OpenAI-compatible

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | List loaded models (includes `rz-llm-default` alias, `owned_by: roundzero-ai`) |
| `POST /v1/chat/completions` | Chat completions |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |

### Portal and monitor

| Endpoint | Purpose |
|---|---|
| `GET /` | Web management portal |
| `GET /monitor` | Monitor dashboard (also embedded in portal) |
| `GET /monitor/api` | JSON snapshot + rolling history |

### Deployment management

| Endpoint | Purpose |
|---|---|
| `GET /api/models` | Model registry from `models.conf` |
| `GET /api/tags` | Available `llama.cpp` build tags with relative age |
| `GET /api/status` | Current deployment state |
| `POST /api/deploy` | Start a deployment pipeline |
| `GET /api/deploy/stream` | SSE stream of deployment log |
| `POST /api/deploy/cancel` | Cancel a running deployment |
| `POST /api/undeploy` | Stop llama-server and rpc-server |

### Model-agnostic requests

The `/v1/models` response injects an `rz-llm-default` alias pointing to whatever model is currently loaded. Clients can use this to avoid hardcoding model names:

```bash
curl http://127.0.0.1:8680/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rz-llm-default",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

### Disabling Qwen thinking mode

```json
{
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }
}
```

## Monitor

The monitor tracks system state across both devices and the inference server.

### Status cards

Compact cards showing real-time values: server health, pp/tg speed, request counts, prompt/gen tokens, Mac/DGX RAM and GPU utilization.

### Rolling history

A time-series table with color-coded thresholds. Rows: llama-server, rpc-server, Mac/DGX RAM, Mac/DGX GPU, prompt/gen tokens, pp, tg, reqs, per-slot KV context. On desktop (24+ columns), shows a 20-minute window with denser samples near the present. On mobile (<24 columns), shows a 5-minute window.

### Live log

Tails the last lines of the llama-server log, updating every 3 seconds.

### Terminal monitor

The `run` command starts both the browser monitor and a terminal heartbeat monitor. The terminal monitor can also be started standalone:

```bash
./deploy.sh monitor 15
```

`Ctrl+C` from the terminal monitor stops `llama-server`.

## Build details

### Mac build

`debug build-mac` builds `llama-server` locally with:

- `-DGGML_METAL=ON`, `-DGGML_RPC=ON`

### DGX build

`debug build-dgx` syncs llama.cpp to the DGX via rsync and builds `rpc-server` with:

- `-DGGML_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES="121"`
- `-DGGML_RPC=ON`, `-DLLAMA_FLASH_ATTENTION=ON`

### SSH behavior

Distributed mode uses SSH ControlMaster for connection reuse:

- Control socket: `.pids/ssh-dgx.ctl`
- `managed` mode: persistent connection with 30-second keepalive check and auto-reconnect
- `run` mode: `ControlPersist=600` (10 minutes)
- `rsync` and `ssh` reuse the same connection

## Process lifecycle

### Managed mode (`./deploy.sh managed`)

The terminal process is the supervisor. It establishes SSH, starts the portal, and blocks. All child processes (llama-server, rpc-server, monitor-web) are tied to this parent:

- **Deploy from portal**: orchestrates pipeline steps as subprocesses (`deploy.sh debug clone`, `build-mac`, `build-dgx`, `download`, `start-rpc`, `start-llama`), streaming output via SSE
- **Redeploy**: always stops both llama-server and rpc-server before starting the new deployment, regardless of mode change
- **Undeploy**: sends SIGTERM, waits up to 30 seconds for graceful exit, then SIGKILL if needed
- **Ctrl+C**: trap stops llama-server, rpc-server, monitor-web, closes SSH, then exits
- **SSH keepalive**: checks every 30 seconds, auto-reconnects if the connection drops

The portal process (`monitor_web.py`) is protected during web-triggered deploys via `SKIP_MONITOR_RESTART=1` — the pipeline uses individual `debug` steps instead of `run` to avoid restarting the portal mid-deploy.

If you update the portal code while `managed` mode is already running, restart only the web monitor to pick up UI changes:

```bash
./deploy.sh debug stop-monitor-web && ./deploy.sh debug start-monitor-web
```

### CLI mode (`./deploy.sh run`)

Runs the full pipeline sequentially in the terminal, then enters the terminal heartbeat monitor. `Ctrl+C` stops llama-server.

## Performance: token generation on Apple Silicon

Running large models on Apple Silicon unified memory makes token generation fundamentally **memory-bandwidth-bound**. The M2 Ultra has ~800 GB/s bandwidth shared between CPU and GPU. Every decode step reads the full model weights plus KV cache through that single pipe.

### Lessons learned

**1. `--ctx-size` with `--kv-unified` is the total shared pool, not per-slot.**

We originally multiplied `ctx_size × parallel` before passing it to `--ctx-size`. With `--kv-unified`, llama.cpp treats it as the entire shared KV pool. The multiplication caused a 4× overallocation (13 GB KV cache instead of 3.25 GB), dropping tg from ~20 to ~3.

**2. `--mmap` is optimal on Apple Silicon — `--mlock` and `--no-mmap` both hurt.**

| Flag | Effect on Apple Silicon | Recommendation |
|---|---|---|
| `--mmap` (default) | Metal creates GPU buffers directly from mmap'd regions via `buffer_from_host_ptr` — **zero-copy**. On unified memory the GPU reads weights straight from the memory-mapped pages with no transfer or duplication. | **Use (default)** |
| `--mlock` | Calls `mlock()` to pin all model pages in physical RAM. On Apple Silicon the GPU already accesses the same physical memory — there is no discrete VRAM to "lock into". Pinning prevents macOS from optimizing page placement and adds wired-memory pressure to the unified memory controller. We measured a tg drop when enabling this. | **Don't use** |
| `--no-mmap` | Forces explicit `read()` into allocated memory, losing the zero-copy Metal buffer path. Slower model loading, no inference benefit. On CUDA it enables async pinned-memory uploads, but on Metal it's strictly worse. | **Don't use** |

The same recommendation applies to both solo and distributed modes — the Mac side is always Metal on unified memory. RPC-offloaded layers run on the DGX's own CUDA memory and are unaffected by Mac-side mmap/mlock settings.

**3. CPU threads compete with the GPU for bandwidth.**

On unified memory, CPU and GPU share the same memory bus. During token generation the GPU is the bottleneck, but active CPU threads still consume bandwidth for their own memory accesses. Reducing `--threads` from 14 to 8 in solo mode gives the GPU more of the ~800 GB/s pipe. Prompt processing may be slightly slower, but token generation — the user-visible latency — improves.

**4. KV cache grows linearly and is the main tg scaling factor.**

For hybrid attention+SSM models (e.g. Qwen3.5-122B with 12 full-attention layers out of 48), the KV cache at q8_0 grows linearly with context:
- 25K tokens: ~300 MB read per token
- 50K tokens: ~600 MB read per token
- 100K tokens: ~1.2 GB read per token

All of this competes with model weight reads on every decode step. The monitor's `ctx s0` row tracks this in real time so you can correlate tg drops with context growth.

**5. `--batch-size` and `--ubatch-size` are fine at defaults (2048 / 512).**

`batch_size` is the logical maximum tokens per `llama_decode` call (application level). `ubatch_size` is the physical chunk size sent to the GPU per iteration (device level). Increasing `ubatch_size` reduces kernel launch overhead but the gains are marginal. The user-visible bottleneck is token generation (tg), not prompt processing (pp) — and batch/ubatch settings have no effect on tg speed.

**6. `--defrag-thold` is deprecated in recent llama.cpp.**

Newer versions handle KV cache defragmentation automatically. The flag is accepted but ignored.

**7. DGX Spark GB10 GPU utilization always reports 0%.**

The GB10's unified memory architecture means `nvidia-smi` returns 0% for `utilization.gpu` and `[N/A]` for `memory.used`/`memory.total`. This is a driver limitation (580.x), not a misconfiguration. The monitor displays the reported value as-is. RAM utilization is read from `/proc/meminfo` instead.

**8. Graceful shutdown matters for memory reclaim.**

Sending SIGTERM to llama-server is not enough — the process needs time to unmap the model and release memory back to the OS. The undeploy path waits up to 30 seconds for the process to exit, then escalates to SIGKILL. Without this wait, macOS may not reclaim memory immediately, making it appear as though the model is still loaded.

### Current tuning

| Setting | Solo (Qwen3.5-122B) | Solo (Qwen3.5-27B) | Distributed (MiniMax) | Rationale |
|---|---|---|---|---|
| `--mmap` | yes | yes | yes | Zero-copy Metal buffer path |
| `--mlock` | no | no | no | Hurts on unified memory |
| `--ctx-size` | 131072 | 131072 | 131072 | 128K shared KV pool |
| `--parallel` | 4 | 8 | 2 | More slots for smaller models |
| `--threads` | 8 | 8 | 14 | Solo: fewer threads = less GPU bandwidth contention |
| `--split-mode` | none | none | layer | Layer-wise split across devices |
| `--tensor-split` | — | — | 2,3 (40:60 DGX:Mac) | Tunable in portal (5% steps) |
| `--kv-unified` | on | on | on | Shared KV pool across slots |
| `--flash-attn` | on | on | on | Required for efficient long-context attention |
| `--batch-size` | 2048 | 2048 | 2048 | Upstream default, marginal gains from increasing |
| `--ubatch-size` | 512 | 512 | 512 | Upstream default |
| `--cache-type-k/v` | q8_0 | q8_0 | q8_0 | Good quality; q4_0 would halve KV bandwidth at quality cost |

## Troubleshooting

### `llama-server` exits during startup

```bash
./deploy.sh logs llama
```

Common causes: wrong model file path, missing mmproj, insufficient RAM, mismatched llama.cpp versions between Mac and DGX.

### Download fails

- Export `HF_TOKEN` in your shell
- Accept the model license on Hugging Face
- Verify `huggingface-cli` is installed

### DGX SSH fails

```bash
ssh user@your-dgx-host echo OK
```

Update `DGX_HOST`, `DGX_USER` in `config.env`.

### Memory not reclaimed after undeploy

The undeploy path waits up to 30 seconds for graceful shutdown. If memory still appears used, check for orphaned processes:

```bash
./deploy.sh status
pgrep -a llama-server
```
