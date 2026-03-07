# rz-rpc-llm

Splits a large GGUF model across two devices using [llama.cpp](https://github.com/ggml-org/llama.cpp) RPC for distributed inference.

| Device | Role | Backend | Memory |
|---|---|---|---|
| Mac Studio M2 Ultra | `llama-server` (coordinator + Metal) | Metal | 192GB unified |
| NVIDIA DGX Spark GB10 | `rpc-server` (GPU offload) | CUDA SM121 | ~128GB unified (Grace Blackwell UMA) |

The Mac hosts the code, downloads the model, and runs `llama-server`. The DGX handles GPU layers via `rpc-server` over the local network. The RPC server **must start before** `llama-server`.

---

## Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Mac Studio M2 Ultra        │  RPC    │  DGX Spark GB10              │
│                             │ (TCP)   │                              │
│  llama-server               │◄───────►│  rpc-server                  │
│  - Metal GPU backend        │  :50052 │  - CUDA SM121 backend        │
│  - OpenAI-compatible API    │         │  - GPU layer offload         │
│  - Serves on :8680          │         │                              │
│  - Coordinates inference    │         │                              │
└─────────────────────────────┘         └──────────────────────────────┘
        ▲
        │ HTTP :8680
        │
   Client (curl, app, etc.)
```

**Tensor split**: `LLAMA_TENSOR_SPLIT="2,3"` means 2 parts to Mac (Metal), 3 parts to DGX (CUDA). Layers are distributed proportionally. Tune based on available memory and model size.

---

## Project Structure

```
rz-rpc-llm/
├── deploy.sh           # Main deployment script (all commands)
├── config.env.example  # Template — copy to config.env
├── config.env          # Local config (gitignored, contains secrets)
├── .gitignore
├── README.md
├── llama.cpp/          # Cloned llama.cpp source (gitignored)
├── models/             # Downloaded GGUF files (gitignored)
├── .pids/              # PID files and SSH control sockets (gitignored)
└── logs/               # Server logs (gitignored)
```

---

## Requirements

- **Mac Studio**: macOS, Xcode CLT, cmake, python3, `huggingface-cli` (`pip install huggingface_hub`)
- **DGX Spark GB10**: Ubuntu/Linux, CUDA toolkit (nvcc, cmake), SSH server
- **Network**: Both devices on the same LAN; DGX port 50052 open to Mac
- **SSH**: Password or key-based auth to DGX (ControlMaster is used — password prompted once per session)
- **Hugging Face**: Account + token for gated models

---

## Quick Start

### 1. Clone this repo (on Mac Studio)

```bash
git clone https://github.com/roundzero-ai/rz-rpc-llm.git
cd rz-rpc-llm
```

### 2. Configure

```bash
cp config.env.example config.env
```

Edit `config.env` — at minimum set:

```bash
DGX_HOST="192.168.86.242"   # DGX IP address
DGX_USER="your_user"         # SSH username on DGX
HF_TOKEN="hf_..."            # Hugging Face token
```

All other values (ports, model params, llama-server flags) have working defaults.

### 3. Full deploy

```bash
chmod +x deploy.sh
./deploy.sh deploy --tag b8223
```

This runs the full pipeline:
1. Clone `llama.cpp` at the specified tag
2. Rsync source to DGX and build with CUDA (SM121)
3. Build locally on Mac with Metal + RPC
4. Download model from HuggingFace
5. Start `rpc-server` on DGX
6. Start `llama-server` on Mac (polls `/health` up to 10 min for large model loads)
7. Enter heartbeat monitor (Ctrl+C stops all servers)

---

## Commands

```
./deploy.sh <command> [options]
```

| Command | Description |
|---|---|
| `clone [--tag TAG]` | Clone llama.cpp (default: HEAD) |
| `build-dgx` | Rsync source to DGX and build with CUDA |
| `build-mac` | Build locally on Mac with Metal |
| `download [--repo R] [--pattern P] [--dir D]` | Download GGUF model from HuggingFace |
| `start-rpc` | Start rpc-server on DGX via SSH |
| `stop-rpc` | Stop rpc-server on DGX |
| `start-llama [--model-file F] [--alias A] [--ctx N] [--parallel N]` | Start llama-server locally |
| `stop-llama` | Stop local llama-server |
| `deploy [--tag T] [--model-file F] [--alias A] [--skip-clone] [--skip-build] [--skip-download]` | Full pipeline |
| `monitor [INTERVAL]` | Rolling table heartbeat monitor (default 30s) |
| `status` | Show running process status for both devices |
| `logs [llama\|rpc]` | Tail server logs (default: llama) |

### Examples

```bash
# First-time full deploy at a specific llama.cpp tag
./deploy.sh deploy --tag b8223

# Rebuild and restart, keep existing model download
./deploy.sh deploy --tag b8223 --skip-download

# Just restart servers (already built and downloaded)
./deploy.sh deploy --skip-clone --skip-build --skip-download

# Start servers individually with a different model
./deploy.sh start-rpc
./deploy.sh start-llama --model-file "UD-Q4_K_M/model-00001.gguf" --alias "minimax-q4"

# Download a different quantization
./deploy.sh download --pattern "*UD-Q4_K_M*"

# Monitor with 15-second heartbeat interval
./deploy.sh monitor 15

# Check status
./deploy.sh status
```

---

## Monitor (Rolling Table)

`./deploy.sh monitor [INTERVAL]` displays a live rolling table that overwrites in place. Each column is a heartbeat snapshot; oldest columns scroll off as new ones arrive.

```
                   │ 12:30:00  │ 12:30:30  │ 12:31:00  │ 12:31:30  │
───────────────────┼───────────┼───────────┼───────────┼───────────┤
Mac memory         │  85G/44%  │  85G/44%  │  84G/44%  │  85G/44%  │
Mac GPU            │       23% │       18% │       45% │       12% │
DGX memory         │  12G/15%  │  12G/15%  │  12G/16%  │  12G/15%  │
DGX GPU            │   67%/UMA │   72%/UMA │   55%/UMA │   60%/UMA │
rpc-server         │        UP │        UP │        UP │        UP │
llama-server       │        UP │        UP │        UP │        UP │
pp (t/s)           │     331.9 │     328.5 │     335.1 │     330.0 │
tg (t/s)           │      19.9 │      20.0 │      19.9 │      20.0 │
reqs processing    │         2 │         1 │         3 │         2 │
prompt tokens      │      1.5K │      2.1K │      3.2K │      4.5K │
gen tokens         │      3.5K │      4.2K │      5.8K │      7.1K │
───────────────────┴───────────┴───────────┴───────────┴───────────┘
```

**Metrics collected per heartbeat:**

| Row | Source | Description |
|---|---|---|
| Mac memory | `vm_stat` + `sysctl hw.memsize` | Used/total RAM as percentage |
| Mac GPU | `ioreg IOAccelerator` | GPU Device Utilization % (no sudo) |
| DGX memory | `free -m` via SSH | Used/total RAM as percentage |
| DGX GPU | `nvidia-smi` via SSH | GPU utilization % + VRAM (or "UMA" for unified memory) |
| rpc-server | `pgrep` via SSH | Process running status |
| llama-server | `curl /health` | HTTP health check |
| pp (t/s) | `/metrics` `llamacpp:prompt_tokens_seconds` | Prompt processing throughput |
| tg (t/s) | `/metrics` `llamacpp:predicted_tokens_seconds` | Token generation throughput |
| reqs processing | `/metrics` `llamacpp:requests_processing` | Currently in-flight requests |
| prompt tokens | `/metrics` `llamacpp:prompt_tokens_total` | Cumulative prompt tokens |
| gen tokens | `/metrics` `llamacpp:tokens_predicted_total` | Cumulative generated tokens |

**Design notes:**
- Column width is fixed at 10 chars. Values are formatted to fit (K/M suffixes for large token counts, 1 decimal for speeds).
- Number of visible columns adapts to terminal width.
- All DGX stats are collected in a **single SSH call** per heartbeat to minimize latency.
- Mac memory uses `vm_stat` (instant) rather than `memory_pressure` (slow, ~1-2s).
- Ctrl+C gracefully stops both servers before exiting.
- Cursor is hidden during monitor and restored on exit.

---

## Configuration Reference (`config.env`)

### Connection

| Variable | Default | Description |
|---|---|---|
| `DGX_HOST` | `192.168.86.242` | DGX Spark IP address |
| `DGX_USER` | `user` | SSH username on DGX |
| `DGX_REMOTE_DIR` | `/home/user/rz-rpc-llm/llama.cpp` | Remote path for rsync + build |
| `DGX_RPC_PORT` | `50052` | rpc-server listen port |
| `DGX_RPC_BIND` | `0.0.0.0` | rpc-server bind address |

### Local Paths

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CPP_DIR` | `./llama.cpp` | Local llama.cpp clone (relative to project root or absolute) |
| `MODELS_DIR` | `./models` | Model download directory |
| `LLAMA_HOST` | `0.0.0.0` | llama-server bind address |
| `LLAMA_PORT` | `8680` | llama-server listen port |

### Model

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required for gated models)* | Hugging Face access token |
| `DEFAULT_MODEL_REPO` | `unsloth/MiniMax-M2.5-GGUF` | HF repo for `download` command |
| `DEFAULT_MODEL_PATTERN` | `*UD-Q6_K_XL*` | File glob for `download` command |
| `DEFAULT_MODEL_FILE` | `UD-Q6_K_XL/MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf` | GGUF file passed to `--model` (relative to `MODELS_DIR`) |
| `DEFAULT_MODEL_ALIAS` | `minimax-m2.5` | `--alias` for llama-server |

### Inference Parameters

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CTX_SIZE` | `131072` | Context window size |
| `LLAMA_PARALLEL` | `4` | Parallel request slots |
| `LLAMA_TENSOR_SPLIT` | `2,3` | Layer split ratio (Mac:DGX) |
| `LLAMA_SPLIT_MODE` | `layer` | Split mode (`layer` or `row`) |
| `LLAMA_N_GPU_LAYERS` | `all` | Number of layers to offload to GPU |
| `LLAMA_THREADS` | `14` | CPU threads for inference |
| `LLAMA_THREADS_BATCH` | `20` | CPU threads for batch processing |
| `LLAMA_BATCH_SIZE` | `8192` | Batch size |
| `LLAMA_UBATCH_SIZE` | `2048` | Micro-batch size |
| `LLAMA_PRIO` | `3` | Process priority |
| `LLAMA_TEMP` | `1.0` | Sampling temperature |
| `LLAMA_TOP_P` | `0.95` | Top-p sampling |
| `LLAMA_TOP_K` | `40` | Top-k sampling |
| `LLAMA_MIN_P` | `0.01` | Min-p sampling |
| `LLAMA_REPEAT_PENALTY` | `1.0` | Repetition penalty |
| `LLAMA_CACHE_TYPE_K` | `q8_0` | KV cache quantization for keys |
| `LLAMA_CACHE_TYPE_V` | `q8_0` | KV cache quantization for values |
| `LLAMA_CACHE_REUSE` | `256` | Cache reuse window |

### Hardcoded llama-server Flags

These are always passed and not configurable via `config.env`:

| Flag | Description |
|---|---|
| `--jinja` | Enable Jinja2 chat template processing |
| `--reasoning-format auto` | Auto-detect reasoning format |
| `--no-mmap` | Disable memory-mapped I/O (required for `--mlock`) |
| `--mlock` | Lock model in RAM (prevents swapping) |
| `--kv-offload` | Offload KV cache to GPU |
| `--kv-unified` | Use unified KV cache across devices |
| `--flash-attn on` | Enable flash attention |
| `--cont-batching` | Enable continuous batching |
| `--no-context-shift` | Disable context shift (use with large context) |
| `--cache-prompt` | Cache prompt for reuse |
| `--metrics` | Enable Prometheus `/metrics` endpoint |

---

## Endpoints

Once running, the llama-server exposes:

| Endpoint | Description |
|---|---|
| `http://localhost:8680/v1` | OpenAI-compatible API (chat completions, completions) |
| `http://localhost:8680/v1/models` | List loaded models |
| `http://localhost:8680/health` | Health check (`{"status":"ok"}`) |
| `http://localhost:8680/metrics` | Prometheus metrics (throughput, tokens, latency) |

### Key Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `llamacpp:prompt_tokens_seconds` | gauge | Prompt processing speed (tokens/s) |
| `llamacpp:predicted_tokens_seconds` | gauge | Token generation speed (tokens/s) |
| `llamacpp:prompt_tokens_total` | counter | Total prompt tokens processed |
| `llamacpp:tokens_predicted_total` | counter | Total tokens generated |
| `llamacpp:tokens_predicted_seconds_total` | counter | Total generation time (seconds) |
| `llamacpp:requests_processing` | gauge | Currently processing requests |
| `llamacpp:n_tokens_max` | counter | Largest observed token count in a single request |

---

## SSH Details

The script uses SSH ControlMaster to avoid repeated password prompts:

- Control socket: `.pids/ssh-dgx.ctl`
- `ControlPersist=600` (10 minutes)
- All SSH and rsync calls reuse the same master connection
- The `build-dgx` command uses `bash -l` (login shell) on DGX because non-interactive SSH does not source `.bashrc`/`.profile`, so `cmake`, `nvcc`, and `find` would not be in PATH

---

## Build Details

### DGX Build (CUDA)

```
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="121" \
    -DGGML_CPU_AARCH64=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_FLASH_ATTENTION=ON
```

- SM121 = Blackwell architecture (GB10)
- `GGML_CPU_AARCH64=ON` for ARM64 Grace CPU
- Static linking (`BUILD_SHARED_LIBS=OFF`)
- The `rpc-server` binary location varies by llama.cpp version — the script uses `find` to locate it dynamically

### Mac Build (Metal)

```
cmake -B build \
    -DGGML_METAL=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_FLASH_ATTENTION=ON
```

---

## Troubleshooting

**SSH fails to DGX**
```bash
# Set up key-based auth
ssh-copy-id user@192.168.86.242

# Or test manually
ssh user@192.168.86.242 echo OK
```

**rpc-server not reachable from Mac**
- Check DGX firewall: port 50052 must be open
- Check DGX log: `./deploy.sh logs rpc`
- Test connectivity: `nc -z 192.168.86.242 50052`

**llama-server exits immediately**
```bash
./deploy.sh logs llama
```
Common causes:
- Model file path wrong (check `DEFAULT_MODEL_FILE` in `config.env`)
- Not enough RAM for `--mlock` — model + KV cache must fit in memory
- RPC server not running — always start rpc first
- llama.cpp version mismatch between Mac and DGX builds

**Health check never succeeds (timeout after 10 min)**
- Large models (>100GB) can take several minutes to load
- Check `logs/llama-server.log` for progress
- Ensure both devices have enough free memory

**HF download fails**
- Set `HF_TOKEN` in `config.env` or export in shell
- Accept model license on huggingface.co
- Ensure `huggingface-cli` is installed: `pip install huggingface_hub`

**Monitor shows "DGX GPU: N/A"**
- The DGX Spark GB10 uses unified memory (UMA) — `nvidia-smi memory.used` returns "Not Supported"
- GPU utilization % should still work; memory is reported via `free` instead

**Stale PID / lock files**
- `.pids/llama-server.pid` — delete if llama-server crashed without cleanup
- `.git/index.lock` — delete if a git operation was interrupted
