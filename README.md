# rz-rpc-llm

Deploys GGUF language models with [llama.cpp](https://github.com/ggml-org/llama.cpp) across Mac Studio and NVIDIA DGX Spark GB10.

Two deployment modes:

| Mode | Hardware | Use Case |
|---|---|---|
| **Distributed** | Mac Studio + DGX Spark | Text-only, leverages both devices via RPC |
| **Vision** | Mac Studio solo | Text + image multimodal (Qwen3.5-122B) |

| Device | Role | Backend | Memory |
|---|---|---|---|
| Mac Studio M2 Ultra | `llama-server` (coordinator + Metal) | Metal | 192GB unified |
| NVIDIA DGX Spark GB10 | `rpc-server` (GPU offload) | CUDA SM121 | ~128GB unified |

---

## Architecture

### Distributed (text-only — Mac + DGX)

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

**Tensor split**: `LLAMA_TENSOR_SPLIT="2,3"` means 2 parts to Mac (Metal), 3 parts to DGX (CUDA).

### Vision (Mac solo — no DGX required)

```
┌─────────────────────────────────────┐
│  Mac Studio M2 Ultra                │
│                                     │
│  llama-server                       │
│  - Full model on Metal GPU          │
│  - Vision mmproj on Metal           │
│  - 262K context window              │
│  - OpenAI-compatible API            │
│  - Serves on :8680                 │
└─────────────────────────────────────┘
        ▲
        │ HTTP :8680
        │
   Client (curl, app, etc.)
```

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
- **DGX Spark GB10** (distributed mode only): Ubuntu/Linux, CUDA toolkit (nvcc, cmake), SSH server
- **Network**: Both devices on the same LAN; DGX port 50052 open to Mac
- **SSH**: Password or key-based auth to DGX (ControlMaster is used — password prompted once per session)
- **Hugging Face**: Account + token for gated models

---

## Quick Start

### 1. Clone this repo

```bash
git clone https://github.com/roundzero-ai/rz-rpc-llm.git
cd rz-rpc-llm
chmod +x deploy.sh
```

### 2. Configure

```bash
cp config.env.example config.env
```

Edit `config.env` — at minimum set:

```bash
DGX_HOST="192.168.86.242"   # DGX IP address (for distributed mode)
DGX_USER="your_user"         # SSH username on DGX
HF_TOKEN="hf_..."            # Hugging Face token
```

### 3. Choose your deployment mode

#### Vision mode (Mac solo — recommended for agents + coding)

```bash
# Download Qwen3.5-122B model + vision projector (~122GB total)
./deploy.sh download --vision

# Start vision server
./deploy.sh start-llama --vision
```

#### Distributed mode (Mac + DGX — text-only)

```bash
# Full pipeline: clone → build → download → start
./deploy.sh deploy --tag b8223
```

---

## Deployment Modes

### Vision Mode (Qwen3.5-122B — Multimodal)

Runs **Qwen3.5-122B-A10B** with vision encoder on Mac Studio alone. No DGX required.

**Capabilities:**
- Text + image understanding
- Native 262K context window
- Thinking/reasoning mode (default) for complex tasks
- Agentic tool calling
- 201 language support

**System requirements (Mac Studio M2 Ultra 192GB):**
- ~105GB for UD-Q6_K_XL model weights
- ~0.9GB for mmproj-BF16 vision projector
- ~25GB for 262K KV cache (q8_0)
- **Total ~130GB — fits comfortably in 192GB**

**Download:**
```bash
./deploy.sh download --vision
# Downloads: Qwen3.5-122B-A10B UD-Q6_K_XL (4-part) + mmproj*.gguf files
```

**Start:**
```bash
./deploy.sh start-llama --vision
```

**Parameters (optimized for 192GB Mac Studio):**
- Context: 262,144 tokens (native max)
- Batch: 2048 / Ubatch: 512
- KV cache: q8_0 quantization
- Flash Attention: on
- Split mode: none (single device)

#### Distributed Mode (MiniMax-M2.5 — Text-only)

Splits MiniMax-M2.5 across Mac Studio + DGX Spark via RPC. Best for pure text inference when both devices are available.

**Start:**
```bash
./deploy.sh deploy
```

**Parameters:**
- Context: 131,072 tokens
- Tensor split: 2:3 (Mac:DGX)
- KV cache: q8_0 quantization

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
| `download [--repo R] [--pattern P] [--dir D] [--vision]` | Download GGUF model |
| `download --vision` | Download Qwen3.5-122B + mmproj for multimodal |
| `start-rpc` | Start rpc-server on DGX via SSH |
| `stop-rpc` | Stop rpc-server on DGX |
| `start-llama [--model-file F] [--alias A] [--ctx N] [--parallel N] [--vision]` | Start llama-server |
| `start-llama --vision` | Start with Qwen3.5 vision defaults (Mac solo, --mmproj, 262K ctx) |
| `stop-llama` | Stop local llama-server |
| `deploy [--tag T] [--skip-clone] [--skip-build] [--skip-download]` | Full pipeline (distributed mode) |
| `monitor [INTERVAL]` | Rolling table heartbeat monitor (default 30s) |
| `status` | Show running process status |
| `logs [llama\|rpc]` | Tail server logs (default: llama) |

### Examples

```bash
# --- Vision mode (Qwen3.5-122B, Mac solo) ---
./deploy.sh download --vision          # Download model + mmproj
./deploy.sh start-llama --vision       # Start vision server
./deploy.sh stop-llama                 # Stop vision server

# --- Distributed mode (Mac + DGX) ---
./deploy.sh deploy --tag b8223         # Full pipeline
./deploy.sh deploy --skip-download     # Rebuild, keep model

# --- Individual steps ---
./deploy.sh start-rpc                  # Start DGX RPC server
./deploy.sh start-llama                # Start llama-server (distributed)

# --- Monitoring ---
./deploy.sh monitor 15                 # 15-second heartbeat
./deploy.sh status
./deploy.sh logs llama
```

---

## Monitor (Rolling Table)

`./deploy.sh monitor [INTERVAL]` displays a live rolling table that overwrites in place.

```
                    │ 12:30:00  │ 12:30:30  │ 12:31:00  │ 12:31:30  │
────────────────────┼───────────┼───────────┼───────────┼───────────┤
Mac memory          │  85G/44%  │  85G/44%  │  84G/44%  │  85G/44%  │
Mac GPU             │       23%  │       18%  │       45%  │       12%  │
DGX memory          │  12G/15%  │  12G/15%  │  12G/16%  │  12G/15%  │
DGX GPU             │   67%/UMA │   72%/UMA │   55%/UMA │   60%/UMA │
rpc-server          │        UP │        UP │        UP │        UP │
llama-server        │        UP │        UP │        UP │        UP │
pp (t/s)            │     331.9 │     328.5 │     335.1 │     330.0 │
tg (t/s)            │      19.9 │      20.0 │      19.9 │      20.0 │
reqs processing     │         2  │         1  │         3  │         2  │
prompt tokens       │      1.5K  │      2.1K  │      3.2K  │      4.5K  │
gen tokens          │      3.5K  │      4.2K  │      5.8K  │      7.1K  │
────────────────────┴───────────┴───────────┴───────────┴───────────┘
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
- All DGX stats are collected in a **single SSH call** per heartbeat to minimize latency.
- Mac memory uses `vm_stat` (instant) rather than `memory_pressure` (slow, ~1-2s).
- Ctrl+C gracefully stops both servers before exiting.

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
| `LLAMA_CPP_DIR` | `./llama.cpp` | Local llama.cpp clone |
| `MODELS_DIR` | `./models` | Model download directory |
| `LLAMA_HOST` | `0.0.0.0` | llama-server bind address |
| `LLAMA_PORT` | `8680` | llama-server listen port |

### Vision Model (Qwen3.5-122B)

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_VISION_REPO` | `unsloth/Qwen3.5-122B-A10B-GGUF` | HuggingFace repo |
| `DEFAULT_VISION_MODEL_FILE` | `UD-Q6_K_XL/...-00001-of-00004.gguf` | GGUF file (first shard) |
| `DEFAULT_VISION_MM_PROJ` | `mmproj-BF16.gguf` | Vision projector file |
| `DEFAULT_VISION_ALIAS` | `qwen3.5-122b-vision` | Model alias for API |
| `DEFAULT_MM_PROJ_PATTERN` | `mmproj*.gguf` | Glob for mmproj download |

### Inference Parameters

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CTX_SIZE` | `131072` | Context window size |
| `LLAMA_PARALLEL` | `1` | Parallel request slots |
| `LLAMA_THREADS` | `14` | CPU threads for inference |
| `LLAMA_THREADS_BATCH` | `14` | CPU threads for batch processing |
| `LLAMA_BATCH_SIZE` | `2048` | Logical batch size |
| `LLAMA_UBATCH_SIZE` | `512` | Physical micro-batch size |
| `LLAMA_N_GPU_LAYERS` | `all` | Layers offloaded to GPU |
| `LLAMA_TENSOR_SPLIT` | `2,3` | Layer split ratio (Mac:DGX) |
| `LLAMA_SPLIT_MODE` | `layer` | Split mode |
| `LLAMA_PRIO` | `3` | Process priority (0-3) |
| `LLAMA_FLASH_ATTN` | `on` | Flash attention |
| `LLAMA_CACHE_TYPE_K` | `q8_0` | KV cache type for keys |
| `LLAMA_CACHE_TYPE_V` | `q8_0` | KV cache type for values |
| `LLAMA_CACHE_PROMPT` | `1` | Enable prompt caching across requests |
| `LLAMA_CACHE_RAM` | `0` | Prompt cache RAM cap in MiB |
| `LLAMA_CACHE_REUSE` | `64` | Cache reuse window |
| `LLAMA_CONT_BATCHING` | `1` | Enable continuous batching |
| `LLAMA_NO_CONTEXT_SHIFT` | `0` | Disable context shifting |

### Vision-Specific Parameters

| Variable | Default | Description |
|---|---|---|
| `LLAMA_VISION_CTX_SIZE` | `262144` | Vision model context (262K native) |
| `LLAMA_VISION_PARALLEL` | `1` | Parallel slots for vision |
| `LLAMA_VISION_BATCH_SIZE` | `2048` | Vision batch size |
| `LLAMA_VISION_UBATCH_SIZE` | `512` | Vision micro-batch size |
| `LLAMA_VISION_N_GPU_LAYERS` | `all` | GPU layers for vision |
| `LLAMA_VISION_SPLIT_MODE` | `none` | No split (single device) |

### Sampling Parameters

| Variable | Default | Description |
|---|---|---|
| `LLAMA_TEMP` | `1.0` | Temperature (general reasoning) |
| `LLAMA_TOP_P` | `0.95` | Top-p sampling |
| `LLAMA_TOP_K` | `20` | Top-k sampling |
| `LLAMA_MIN_P` | `0.0` | Min-p sampling |
| `LLAMA_REPEAT_PENALTY` | `1.0` | Repetition penalty |
| `LLAMA_PRESENCE_PENALTY` | `1.5` | Presence penalty |
| `LLAMA_FREQUENCY_PENALTY` | `0.0` | Frequency penalty |
| `LLAMA_TEMP_CODING` | `0.6` | Temperature for coding |
| `LLAMA_TOP_P_CODING` | `0.95` | Top-p for coding |
| `LLAMA_TOP_K_CODING` | `20` | Top-k for coding |
| `LLAMA_PRESENCE_PENALTY_CODING` | `0.0` | Coding presence penalty |
| `LLAMA_TEMP_INSTRUCT` | `0.7` | Temperature for instruct/non-thinking |
| `LLAMA_TOP_P_INSTRUCT` | `0.8` | Top-p for instruct mode |
| `LLAMA_TOP_K_INSTRUCT` | `20` | Top-k for instruct mode |
| `LLAMA_PRESENCE_PENALTY_INSTRUCT` | `1.5` | Instruct presence penalty |

---

## Qwen3.5-122B Sampling Guide

Qwen3.5 supports **thinking mode** (default) — the model generates `<thinking>` content before the final response. Use different sampling parameters based on use case:

| Mode | temp | top_p | top_k | presence_penalty | Use Case |
|---|---|---|---|---|---|
| **General reasoning** | 1.0 | 0.95 | 20 | 1.5 | Math, analysis, complex tasks |
| **Coding** | 0.6 | 0.95 | 20 | 0.0 | Precise code generation |
| **Instruct** (no thinking) | 0.7 | 0.8 | 20 | 1.5 | Direct Q&A, summaries |

**Recommended output length:** 32,768 tokens for most queries; 81,920 for math/programming benchmarks.

**Disable thinking** (get direct answers):
```json
{
  "extra_body": {
    "chat_template_kwargs": {"enable_thinking": false}
  }
}
```

---

## Endpoints

| Endpoint | Description |
|---|---|
| `http://localhost:8680/v1` | OpenAI-compatible API |
| `http://localhost:8680/v1/models` | List loaded models |
| `http://localhost:8680/health` | Health check |
| `http://localhost:8680/metrics` | Prometheus metrics |

### Key Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `llamacpp:prompt_tokens_seconds` | gauge | Prompt processing speed (tokens/s) |
| `llamacpp:predicted_tokens_seconds` | gauge | Token generation speed (tokens/s) |
| `llamacpp:prompt_tokens_total` | counter | Total prompt tokens processed |
| `llamacpp:tokens_predicted_total` | counter | Total tokens generated |
| `llamacpp:requests_processing` | gauge | Currently processing requests |

---

## API Usage Examples

### Vision (Qwen3.5-122B)

```bash
# Image understanding
curl http://127.0.0.1:8680/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b-vision",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/screenshot.png"}},
      {"type": "text", "text": "Describe this image in detail."}
    ]}],
    "max_tokens": 8192,
    "temperature": 1.0,
    "top_p": 0.95,
    "extra_body": {"top_k": 20}
  }'

# Coding with thinking
curl http://127.0.0.1:8680/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b-vision",
    "messages": [{"role": "user", "content": "Write a Python async context manager for database connections"}],
    "max_tokens": 8192,
    "temperature": 0.6,
    "top_p": 0.95,
    "extra_body": {"top_k": 20, "presence_penalty": 0.0}
  }'

# Disable thinking (instruct mode)
curl http://127.0.0.1:8680/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b-vision",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.8,
    "extra_body": {
      "top_k": 20,
      "chat_template_kwargs": {"enable_thinking": false}
    }
  }'
```

---

## SSH Details (Distributed Mode)

The script uses SSH ControlMaster to avoid repeated password prompts:

- Control socket: `.pids/ssh-dgx.ctl`
- `ControlPersist=600` (10 minutes)
- All SSH and rsync calls reuse the same master connection
- The `build-dgx` command uses `bash -l` (login shell) on DGX because non-interactive SSH does not source `.bashrc`/`.profile`

---

## Build Details

### DGX Build (CUDA)

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="121" \
    -DGGML_CPU_AARCH64=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release
```

- SM121 = Blackwell GB10 architecture
- Static linking
- `rpc-server` binary location varies by llama.cpp version — the script locates it dynamically with `find`

### Mac Build (Metal)

```bash
cmake -B build \
    -DGGML_METAL=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release
```

---

## Troubleshooting

**llama-server exits immediately:**
```bash
./deploy.sh logs llama
```
Common causes:
- Model file path wrong (check `DEFAULT_MODEL_FILE` in `config.env`)
- Not enough RAM — model + KV cache must fit in memory
- rpc-server not running (for distributed mode)
- llama.cpp version mismatch between Mac and DGX builds

**Health check never succeeds (timeout after 10 min):**
- Large models (>100GB) can take several minutes to load
- Check `logs/llama-server.log` for progress
- Ensure enough free memory

**HF download fails:**
- Set `HF_TOKEN` in `config.env` or export in shell
- Accept model license on huggingface.co
- Ensure `huggingface-cli` is installed: `pip install huggingface_hub`

**Stale PID / lock files:**
- `.pids/llama-server.pid` — delete if llama-server crashed
- `.git/index.lock` — delete if git operation was interrupted

**Distributed mode — SSH fails to DGX:**
```bash
ssh-copy-id user@192.168.86.242
ssh user@192.168.86.242 echo OK
```

**Distributed mode — rpc-server not reachable:**
- Check DGX firewall: port 50052 must be open
- Test: `nc -z 192.168.86.242 50052`
