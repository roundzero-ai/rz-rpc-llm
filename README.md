# rz-rpc-llm

A deployment wrapper around `llama.cpp` for running large language models on a Mac Studio M2 Ultra (192 GB), optionally offloading layers to a DGX Spark GB10 via RPC.

Models and their runtime parameters are defined in `models.conf` — adding a new model requires no code changes.

## Deployment modes

| Mode | Hardware | When |
|---|---|---|
| Solo | Mac Studio only | Model fits in 192 GB (e.g., Qwen3.5-122B-A10B) |
| Distributed | Mac Studio + DGX Spark GB10 | Model needs both devices (e.g., MiniMax-M2.5) |

Each model in `models.conf` declares a default mode and whether solo is allowed. The user can override with `--solo` or `--distributed`.

## Architecture

```text
Client
  |
  | HTTP :8680
  v
Mac Studio (Metal)
  - llama-server (backend on :8682)
  - monitor-web gateway (public on :8680)
  - proxies /v1, /health, /metrics to backend
  - serves /monitor UI directly
  |
  | RPC :50052 (distributed mode only)
  v
DGX Spark GB10 (CUDA)
  - rpc-server
  - remote GPU offload target
```

## Repository layout

```text
rz-rpc-llm/
|- deploy.sh              # deployment CLI
|- models.conf            # model registry (checked in)
|- monitor_web.py         # browser monitor gateway
|- defaults.env           # infrastructure defaults (checked in)
|- config.env.example     # local override template
|- config.env             # optional local overrides (gitignored)
|- README.md
|- llama.cpp/             # cloned by deploy.sh (gitignored)
|- models/                # downloaded GGUF files (gitignored)
|- logs/                  # runtime logs (gitignored)
`- .pids/                 # pid files and SSH control socket (gitignored)
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
```

### Vision (Qwen3.5, Mac solo)

```bash
./deploy.sh run --model qwen3.5 --vision
```

This single command will:
1. Clone `llama.cpp` (latest tag)
2. Build `llama-server` on Mac
3. Download the Qwen3.5-122B model + mmproj from Hugging Face
4. Start `llama-server` and the monitor-web gateway
5. Enter the terminal heartbeat monitor

### Distributed text (MiniMax, Mac + DGX)

```bash
./deploy.sh run --model minimax
```

Same pipeline, but also builds and starts `rpc-server` on the DGX.

### Skip steps on subsequent runs

```bash
./deploy.sh run --model qwen3.5 --vision --skip-clone --skip-build
```

## Commands

```bash
./deploy.sh <command> [options]
```

| Command | Purpose |
|---|---|
| `run --model NAME [flags]` | Full pipeline: clone, build, download, start, monitor |
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
./deploy.sh debug download --vision
./deploy.sh debug start-rpc
./deploy.sh debug start-llama --vision
```

## Configuration

### Three layers

1. **`defaults.env`** (checked in) — infrastructure settings: DGX connection, listen ports, global performance flags
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
parallel = 1
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

Then run:

```bash
./deploy.sh run --model my-new-model
```

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

## API endpoints

| Endpoint | Purpose |
|---|---|
| `http://127.0.0.1:8680/v1` | OpenAI-compatible API |
| `http://127.0.0.1:8680/v1/models` | List loaded models |
| `http://127.0.0.1:8680/health` | Health check |
| `http://127.0.0.1:8680/metrics` | Prometheus metrics |
| `http://127.0.0.1:8680/monitor` | Browser monitor UI |
| `http://127.0.0.1:8680/monitor/api` | JSON snapshot + rolling history |

## API examples

### Vision request

```bash
curl http://127.0.0.1:8680/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b-vision",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://example.com/screenshot.png"}},
      {"type": "text", "text": "Describe this image in detail."}
    ]}],
    "max_tokens": 8192
  }'
```

### Text request

```bash
curl http://127.0.0.1:8680/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.5",
    "messages": [
      {"role": "user", "content": "Summarize the main tradeoffs of RPC-based model split inference."}
    ],
    "max_tokens": 2048
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

The `run` command automatically starts both the terminal monitor and the browser monitor.

The browser monitor is at `http://127.0.0.1:8680/monitor` (shares the same port as the API). It shows:

- Mac/DGX memory and GPU utilization
- Server health, pp/tg speed, request counts
- Per-slot KV context tracking
- Rolling history with color-coded thresholds
- Live llama-server log tail

The terminal monitor can also be started standalone:

```bash
./deploy.sh monitor 15
```

`Ctrl+C` from the monitor stops `llama-server`.

## Performance: token generation on Apple Silicon

Running large models on Apple Silicon unified memory makes token generation fundamentally **memory-bandwidth-bound**. The M2 Ultra has ~800 GB/s bandwidth shared between CPU and GPU. Every decode step reads ~105 GB of model weights plus KV cache through that single pipe.

### Lessons learned (Qwen3.5-122B-A10B on M2 Ultra 192 GB)

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

On unified memory, CPU and GPU share the same memory bus. During token generation the GPU is the bottleneck, but active CPU threads still consume bandwidth for their own memory accesses. Reducing `--threads` from 14 to 8 in vision mode gives the GPU more of the ~800 GB/s pipe. Prompt processing may be slightly slower, but token generation — the user-visible latency — improves.

**4. KV cache grows linearly and is the main tg scaling factor.**

For this hybrid attention+SSM model (12 full-attention layers out of 48), the KV cache at q8_0 grows linearly with context:
- 25K tokens: ~300 MB read per token
- 50K tokens: ~600 MB read per token
- 100K tokens: ~1.2 GB read per token

All of this competes with the ~105 GB model weight reads on every decode step. The monitor's `ctx s0` row tracks this in real time so you can correlate tg drops with context growth.

**5. `--batch-size` and `--ubatch-size` are fine at defaults (2048 / 512).**

`batch_size` is the logical maximum tokens per `llama_decode` call (application level). `ubatch_size` is the physical chunk size sent to the GPU per iteration (device level). During prompt processing, the engine splits the prompt into `ubatch_size`-token chunks — so a 40K prompt with `ubatch_size=512` runs ~78 GPU iterations.

Increasing `ubatch_size` reduces the number of iterations (and kernel launch overhead), but the gains are marginal because kernel launches are cheap relative to the actual compute. More importantly, the user-visible bottleneck is token generation (tg), not prompt processing (pp) — and batch/ubatch settings have no effect on tg speed. For distributed mode, larger ubatches reduce RPC round-trips but the data transferred per trip scales proportionally, so total transfer time is roughly constant.

We tested and kept the upstream defaults: `batch_size=2048`, `ubatch_size=512` for both models.

**6. `--defrag-thold` is deprecated in recent llama.cpp.**

Newer versions handle KV cache defragmentation automatically. The flag is accepted but ignored. We keep it for backward compatibility.

### Current tuning

| Setting | Solo (Qwen3.5) | Distributed (MiniMax) | Rationale |
|---|---|---|---|
| `--mmap` | yes | yes | Zero-copy Metal buffer path |
| `--mlock` | no | no | Hurts on unified memory |
| `--ctx-size` | 131072 | 131072 | 128K shared KV pool |
| `--parallel` | 4 | 1 | Solo can serve concurrent requests |
| `--threads` | 8 | 14 | Solo: fewer threads = less GPU bandwidth contention |
| `--kv-unified` | on | on | Shared KV pool across slots |
| `--flash-attn` | on | on | Required for efficient long-context attention |
| `--batch-size` | 2048 | 2048 | Logical token cap per decode call; headroom for continuous batching |
| `--ubatch-size` | 512 | 512 | GPU compute chunk; upstream default, marginal gains from increasing |
| `--cache-type-k/v` | q8_0 | q8_0 | Good quality; q4_0 would halve KV bandwidth at quality cost |

## Build details

### DGX build

`debug build-dgx` syncs llama.cpp to the DGX and builds `rpc-server` with:

- `-DGGML_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES="121"`
- `-DGGML_RPC=ON`, `-DLLAMA_FLASH_ATTENTION=ON`

### Mac build

`debug build-mac` builds `llama-server` locally with:

- `-DGGML_METAL=ON`, `-DGGML_RPC=ON`

## SSH behavior

Distributed mode uses SSH ControlMaster:

- Control socket: `.pids/ssh-dgx.ctl`
- Control persist: 10 minutes
- `rsync` and `ssh` reuse the same connection

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

Update `DGX_HOST`, `DGX_USER`, `DGX_REMOTE_DIR` in `config.env`.
