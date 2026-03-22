# rz-rpc-llm

`rz-rpc-llm` is a small deployment wrapper around `llama.cpp` for two specific setups:

- `MiniMax-M2.5` in distributed text-only mode across a Mac Studio and a DGX Spark GB10 via RPC
- `Qwen3.5-122B-A10B` in vision mode on a Mac Studio alone with an `mmproj` projector

The script handles cloning `llama.cpp`, building the right binaries on each machine, downloading the expected GGUF files, starting servers, and monitoring runtime health.

## What this repo does

There are two deployment modes.

| Mode | Model | Hardware | Purpose |
|---|---|---|---|
| `distributed` | `MiniMax-M2.5` | Mac Studio + DGX Spark GB10 | Text-only inference with RPC offload |
| `vision` | `Qwen3.5-122B-A10B` | Mac Studio only | Multimodal text + image inference |

The local API surface is always `llama-server` on the Mac, exposed as an OpenAI-compatible endpoint on port `8680` by default.

## Architecture

### Distributed mode

In distributed mode, the Mac runs `llama-server` and the DGX runs `rpc-server`.

```text
Client
  |
  | HTTP :8680
  v
Mac Studio (Metal)
  - llama-server
  - OpenAI-compatible API
  - coordinates inference
  - shares model execution over RPC
  |
  | RPC :50052
  v
DGX Spark GB10 (CUDA)
  - rpc-server
  - remote GPU offload target
```

Key defaults from `defaults.env`:

- model: `unsloth/MiniMax-M2.5-GGUF`
- tensor split: `2,3` (Mac:DGX)
- context: `131072`
- sampling: `temp=1.0`, `top_p=0.95`, `top_k=40`, `min_p=0.01`

### Vision mode

In vision mode, everything runs locally on the Mac, including the projector.

```text
Client
  |
  | HTTP :8680
  v
Mac Studio (Metal)
  - llama-server
  - Qwen3.5-122B-A10B GGUF
  - mmproj projector
  - no DGX required
```

Key defaults from `defaults.env`:

- model: `unsloth/Qwen3.5-122B-A10B-GGUF`
- context: `131072` (128K shared KV pool)
- slots: `4`
- threads: `8` (reduced to avoid GPU bandwidth contention)
- sampling: `temp=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`
- split mode: `none`

## Repository layout

```text
rz-rpc-llm/
|- deploy.sh
|- monitor_web.py
|- defaults.env
|- config.env.example
|- config.env              # optional local overrides, gitignored
|- README.md
|- llama.cpp/              # cloned by deploy.sh, gitignored
|- models/                 # downloaded GGUF files, gitignored
|- logs/                   # runtime logs, gitignored
`- .pids/                  # pid files and SSH control socket, gitignored
```

## Requirements

### Mac Studio

- macOS
- Xcode Command Line Tools
- `cmake`
- `python3`
- `huggingface-cli` from `huggingface_hub`

### DGX Spark GB10

Distributed mode also needs:

- SSH access from the Mac
- CUDA toolchain and `cmake`
- port `50052` reachable from the Mac

### Hugging Face

For gated model downloads, export a token in your shell:

```bash
export HF_TOKEN="hf_..."
```

Do not store tokens in repo config files.

## Configuration model

This repo now uses two layers of configuration:

- `defaults.env`: checked-in, non-secret defaults that make the repo runnable after clone
- `config.env`: optional local overrides, gitignored

If you need machine-specific settings, start from the template:

```bash
cp config.env.example config.env
```

Typical overrides:

```bash
DGX_HOST="192.168.86.242"
DGX_USER="your_user"
DGX_REMOTE_DIR="/home/your_user/rz-rpc-llm/llama.cpp"
```

## Quick start

### 1. Clone the repo

```bash
git clone https://github.com/roundzero-ai/rz-rpc-llm.git
cd rz-rpc-llm
chmod +x deploy.sh
```

### 2. Pick a mode

#### Vision mode

This is the simplest path and works from checked-in defaults.

```bash
./deploy.sh clone
./deploy.sh build-mac
./deploy.sh download --vision
./deploy.sh start-llama --vision
```

`start-llama --vision` automatically drops into the monitor after the server becomes healthy.
The browser monitor is available on the same public port at `http://127.0.0.1:8680/monitor`.

Or, if you already have `llama.cpp` built:

```bash
./deploy.sh download --vision
./deploy.sh start-llama --vision
```

#### Distributed mode

Set your DGX overrides first if the defaults do not match your box, then run:

```bash
./deploy.sh deploy --tag b8223
```

That pipeline does:

- clone `llama.cpp`
- build on DGX
- build on Mac
- download the default MiniMax model
- start `rpc-server`
- start local `llama-server`

After startup, the OpenAI API stays on `http://127.0.0.1:8680/v1` and the browser monitor lives on `http://127.0.0.1:8680/monitor`.

## Commands

`deploy.sh` is the whole interface.

```bash
./deploy.sh <command> [options]
```

### Lifecycle commands

| Command | Purpose |
|---|---|
| `clone [--tag TAG]` | Clone `llama.cpp`, optionally checkout a tag or `latest` |
| `build-dgx` | Sync `llama.cpp` to the DGX and build `rpc-server` |
| `build-mac` | Build local `llama-server` with Metal |
| `download [--repo R] [--pattern P] [--dir D]` | Download the default text model or a custom GGUF pattern |
| `download --vision` | Download the Qwen vision model and `mmproj` |
| `start-rpc` | Start `rpc-server` on the DGX |
| `stop-rpc` | Stop `rpc-server` on the DGX |
| `start-llama [--model-file F] [--alias A] [--ctx N] [--parallel N]` | Start local `llama-server` in distributed mode |
| `start-llama --vision` | Start local `llama-server` in vision mode |
| `stop-llama` | Stop local `llama-server` |
| `start-monitor-web [--port P]` | Start the single-port gateway that serves both API and monitor UI |
| `stop-monitor-web` | Stop the browser dashboard |
| `deploy [--tag T] [--skip-clone] [--skip-build] [--skip-download]` | Full distributed pipeline |
| `full ...` | Alias for `deploy` |

### Observability commands

| Command | Purpose |
|---|---|
| `status` | Show Mac and DGX process status |
| `logs [llama|rpc|monitor]` | Tail local or remote logs |
| `monitor [INTERVAL] [TABLE_WIDTH]` | Show rolling health + performance monitor |

## Recommended workflows

### Vision workflow

```bash
./deploy.sh clone
./deploy.sh build-mac
./deploy.sh download --vision
./deploy.sh start-llama --vision
./deploy.sh status
```

### Distributed workflow

```bash
./deploy.sh clone --tag b8223
./deploy.sh build-dgx
./deploy.sh build-mac
./deploy.sh download
./deploy.sh start-rpc
./deploy.sh start-llama
./deploy.sh monitor 30
```

### Rebuild without re-downloading

```bash
./deploy.sh deploy --skip-download
```

## Models and runtime defaults

### Distributed text deployment

- model repo: `unsloth/MiniMax-M2.5-GGUF`
- default file: `UD-Q6_K_XL/MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf`
- alias: `minimax-m2.5`
- context: `131072`
- slots: `1`
- tensor split: `2,3`
- KV cache: `q8_0`
- flash attention: `on`

MiniMax defaults baked into `deploy.sh`:

| Setting | Value |
|---|---|
| `temperature` | `1.0` |
| `top_p` | `0.95` |
| `top_k` | `40` |
| `min_p` | `0.01` |
| `repeat_penalty` | `1.0` |
| `presence_penalty` | `0.0` |

### Vision deployment

- model repo: `unsloth/Qwen3.5-122B-A10B-GGUF`
- default file: `UD-Q6_K_XL/Qwen3.5-122B-A10B-UD-Q6_K_XL-00001-of-00004.gguf`
- projector: `mmproj-BF16.gguf`
- alias: `qwen3.5-122b-vision`
- context: `131072` (shared KV pool via `--kv-unified`)
- slots: `4`
- threads: `8` (reduced for bandwidth)
- KV cache: `q8_0`
- flash attention: `on`

Qwen defaults baked into `deploy.sh`:

| Use case | temp | top_p | top_k | min_p | presence_penalty |
|---|---:|---:|---:|---:|---:|
| general reasoning | 1.0 | 0.95 | 20 | 0.0 | 1.5 |
| coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| direct / non-thinking | 0.7 | 0.8 | 20 | 0.0 | 1.5 |

To disable Qwen thinking mode in API requests:

```json
{
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }
}
```

## API endpoints

Once `llama-server` is up, the important endpoints are:

| Endpoint | Purpose |
|---|---|
| `http://127.0.0.1:8680/v1` | OpenAI-compatible API |
| `http://127.0.0.1:8680/v1/models` | list loaded models |
| `http://127.0.0.1:8680/health` | health check |
| `http://127.0.0.1:8680/metrics` | Prometheus metrics |
| `http://127.0.0.1:8680/monitor` | colorful browser monitor UI on the same port as the API |
| `http://127.0.0.1:8680/monitor/api` | JSON snapshot + rolling history for the browser UI |

## API examples

### Vision request with an image

```bash
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
```

### Text request in distributed mode

```bash
curl http://127.0.0.1:8680/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.5",
    "messages": [
      {"role": "user", "content": "Summarize the main tradeoffs of RPC-based model split inference."}
    ],
    "max_tokens": 2048,
    "temperature": 1.0,
    "top_p": 0.95,
    "extra_body": {"top_k": 40, "min_p": 0.01}
  }'
```

## Monitor

`./deploy.sh monitor` renders a rolling terminal dashboard.

If you want the same information in a browser, start the web monitor:

```bash
./deploy.sh start-monitor-web
open http://127.0.0.1:8680/monitor
```

In the single-port setup, `8680` is the public gateway:

- `/v1`, `/health`, and `/metrics` are proxied to the internal `llama-server` backend
- `/monitor` and `/monitor/api` are served directly by the gateway
- `start-llama` will also bring the gateway up so the API and monitor stay on one port

By default the internal backend listens on `127.0.0.1:8682` and is not meant to be called directly.

It shows:

- Mac memory and GPU utilization
- DGX memory and GPU utilization in distributed mode
- `rpc-server` and `llama-server` health
- prompt processing speed (`pp`)
- token generation speed (`tg`)
- active requests, prompt tokens, and generated tokens
- a live tail of the last 5 `llama-server` log lines
- a colorful browser dashboard with rolling history and endpoint links

Example:

```bash
./deploy.sh monitor 15
```

Developer-focused UX example:

```text
                    | 12:30:00 | 12:30:15 | 12:30:30 | 12:30:45 |
--------------------+----------+----------+----------+----------+
Mac RAM used        |  85G/44% |  86G/45% |  86G/45% |  87G/45% |
Mac GPU util        |      23% |      41% |      38% |      19% |
DGX RAM used        |  12G/15% |  12G/15% |  13G/16% |  13G/16% |
DGX GPU util        |   67%/8G |   71%/8G |   63%/8G |   58%/8G |
rpc-server          |       UP |       UP |       UP |       UP |
llama-server        |       UP |       UP |       UP |       UP |
pp (t/s)            |    331.9 |    328.5 |    335.1 |    330.0 |
tg (t/s)            |     19.9 |     20.0 |     19.9 |     20.0 |
reqs                |        2 |        1 |        3 |        2 |
prompt tokens       |     1.5K |     2.1K |     3.2K |     4.5K |
gen tokens          |     3.5K |     4.2K |     5.8K |     7.1K |

⚡ live @ 12:30:45 --- llama-server log (last 5 lines) ---
srv  update_slots: all slots are idle
srv  request: POST /v1/chat/completions 127.0.0.1 200
slot 0: prompt eval time = 3.01 s / 998 tokens
slot 0: generation eval time = 21.44 s / 428 tokens
metrics: prompt=331.9 t/s, generation=20.0 t/s
```

This is useful when you are tuning split ratios, context size, or model choice and want one terminal view for health, throughput, and the latest server behavior.

Notes:

- `Ctrl+C` from the monitor stops `llama-server`
- `start-llama --vision` automatically enters the monitor after startup
- DGX stats are collected with a single SSH call per heartbeat
- the live log section now shows the last 5 lines only

## Build details

### DGX build

`build-dgx` syncs the local `llama.cpp` tree to `DGX_REMOTE_DIR` and builds `rpc-server` remotely with CUDA and RPC enabled.

Important flags in the remote build:

- `-DGGML_CUDA=ON`
- `-DGGML_CUDA_FA_ALL_QUANTS=ON`
- `-DCMAKE_CUDA_ARCHITECTURES="121"`
- `-DGGML_RPC=ON`
- `-DLLAMA_FLASH_ATTENTION=ON`

### Mac build

`build-mac` builds `llama-server` locally with:

- `-DGGML_METAL=ON`
- `-DGGML_RPC=ON`
- `-DCMAKE_BUILD_TYPE=Release`

## SSH behavior

Distributed mode uses SSH ControlMaster so you do not get repeatedly prompted for credentials.

- control socket: `.pids/ssh-dgx.ctl`
- control persist: 10 minutes
- `rsync` and `ssh` reuse the same control connection
- remote builds run in `bash -l` so DGX toolchain paths are available

## Troubleshooting

### `llama-server` exits during startup

Check:

```bash
./deploy.sh logs llama
```

Common causes:

- wrong `DEFAULT_MODEL_FILE` or `DEFAULT_VISION_MODEL_FILE`
- missing `mmproj` in vision mode
- not enough free RAM for weights plus KV cache
- `rpc-server` not running for distributed mode
- Mac and DGX built from mismatched `llama.cpp` versions

### Download fails

- export `HF_TOKEN` in your shell
- accept the model license on Hugging Face
- verify `huggingface-cli` is installed

### DGX SSH fails

Try:

```bash
ssh user@your-dgx-host echo OK
```

If needed, update `DGX_HOST`, `DGX_USER`, and `DGX_REMOTE_DIR` in `config.env`.

### RPC server is not reachable

- make sure port `50052` is open
- verify the DGX host is reachable from the Mac
- test with `nc -z <DGX_HOST> 50052`

### Health check takes a long time

Large models can take several minutes to mmap and initialize. Check:

- `logs/llama-server.log`
- `http://127.0.0.1:8680/health`
- `http://127.0.0.1:8680/metrics`

## Performance: token generation speed on Apple Silicon

Running large models on Apple Silicon unified memory makes token generation (tg) fundamentally **memory-bandwidth-bound**. The M2 Ultra has ~800 GB/s bandwidth shared between CPU and GPU. Every token generation step must read model weights, KV cache, and SSM state through that single pipe. As context grows, tg drops because the KV cache read grows linearly.

### What we learned (Qwen3.5-122B-A10B on M2 Ultra 192 GB)

**1. `--ctx-size` with `--kv-unified` is the total shared pool, not per-slot.**

We originally multiplied `ctx_size × parallel` before passing it to `--ctx-size`. With `--kv-unified`, llama.cpp treats `--ctx-size` as the entire shared KV pool. The multiplication caused a 4× overallocation (13 GB KV cache instead of 3.25 GB), starving bandwidth and dropping tg from ~20 to ~3.

Fix: pass the per-slot context value directly. llama.cpp handles slot sharing internally.

**2. `--mlock` hurts more than it helps on unified memory.**

`mlock` pins all model pages in physical memory, which sounds beneficial. But on Apple Silicon the GPU already accesses the same physical memory — there is no discrete VRAM to "lock into". The pinning prevents macOS from optimizing page placement and adds pressure to the unified memory controller. Removing `--mlock` lets the OS manage pages more efficiently and reduces bandwidth contention.

**3. CPU threads compete with the GPU for bandwidth.**

On unified memory, CPU and GPU share the same memory bus. During token generation the GPU is the bottleneck, but active CPU threads still consume bandwidth for their own memory accesses (thread stacks, scheduler, OS overhead). Reducing `--threads` from 14 to 8 in vision mode gives the GPU more of the bandwidth pipe. Prompt processing (pp) may be slightly slower, but token generation (tg) — the user-visible latency — improves.

**4. KV cache grows linearly and is the main tg scaling factor.**

For this hybrid attention+SSM model (12 full-attention layers out of 48), the KV cache at q8_0 uses roughly:
- 25K context: ~300 MB read per token
- 50K context: ~600 MB read per token
- 100K context: ~1.2 GB read per token

All of this competes with the ~105 GB model weight reads on every decode step. The monitor's `ctx s0` row tracks this in real time so you can correlate tg drops with context growth.

**5. `--defrag-thold` is deprecated in recent llama.cpp.**

Newer versions handle KV cache defragmentation automatically. The flag is accepted but ignored. We keep it in the config for backward compatibility but it has no effect.

### Current tuning (vision mode)

| Setting | Value | Rationale |
|---|---|---|
| `--ctx-size` | 131072 | 128K shared KV pool — keeps tg acceptable through the window |
| `--parallel` | 4 | 4 slots sharing the unified KV pool |
| `--threads` | 8 | fewer CPU threads = less bandwidth contention with GPU |
| `--mlock` | removed | let macOS manage page placement on unified memory |
| `--kv-unified` | on | shared KV pool across slots |
| `--flash-attn` | on | required for efficient attention at long context |
| `--cache-type-k/v` | q8_0 | good quality; q4_0 would halve KV bandwidth at some quality cost |

## File reference

The most important files are:

- `deploy.sh`: the deployment entrypoint and runtime orchestration
- `defaults.env`: checked-in model, hardware, and runtime defaults
- `config.env.example`: optional local override template
