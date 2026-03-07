# rz-rpc-llm

Runs a large GGUF model split across two devices using [llama.cpp](https://github.com/ggml-org/llama.cpp) RPC:

| Device | Role | Backend |
|---|---|---|
| Mac Studio M2 Ultra 192GB | llama-server (coordinator + Metal) | Metal |
| DGX Spark GB10 | rpc-server (GPU offload) | CUDA SM121 |

The Mac hosts the code, downloads the model, and runs `llama-server`. The DGX handles GPU layers via `rpc-server`. The RPC server **must start before** `llama-server`.

---

## Requirements

- Mac Studio M2 Ultra with macOS, Xcode CLT, cmake, python3
- DGX Spark GB10 accessible via SSH with CUDA toolkit installed
- SSH key-based auth to DGX (no password prompts)
- Hugging Face account + token for gated models

---

## Quick Start

### 1. Clone this repo (on Mac Studio only)

```bash
git clone <this-repo> rz-rpc-llm
cd rz-rpc-llm
```

### 2. Configure

```bash
cp config.env.example config.env
```

Edit `config.env`:

```bash
DGX_HOST="192.168.86.242"   # DGX IP address
DGX_USER="your_user"         # SSH username on DGX
HF_TOKEN="hf_..."            # Hugging Face token
```

All other values (ports, model params, llama-server flags) have working defaults but can be tuned.

### 3. Full deploy at a specific tag

```bash
chmod +x deploy.sh
./deploy.sh deploy --tag b8223
```

This runs the full pipeline:
1. Clone `llama.cpp` at tag `b8223`
2. Rsync source to DGX → build with CUDA
3. Build locally on Mac with Metal
4. Download model from HuggingFace
5. Start `rpc-server` on DGX
6. Start `llama-server` on Mac (waits for health check)

---

## Commands

```
./deploy.sh clone    [--tag TAG]
./deploy.sh build-dgx
./deploy.sh build-mac
./deploy.sh download [--repo REPO] [--pattern GLOB] [--dir DIR]
./deploy.sh start-rpc
./deploy.sh stop-rpc
./deploy.sh start-llama [--model-file PATH] [--alias NAME] [--ctx N] [--parallel N]
./deploy.sh stop-llama
./deploy.sh deploy   [--tag TAG] [--model-file PATH] [--alias NAME]
                     [--skip-clone] [--skip-build] [--skip-download]
./deploy.sh status
./deploy.sh logs     [llama|rpc]
```

### `clone`

```bash
./deploy.sh clone --tag b8223    # specific commit/tag
./deploy.sh clone                # HEAD
```

Clones `https://github.com/ggml-org/llama.cpp.git` into `./llama.cpp`. If already cloned, fetches and checks out the requested tag.

### `build-dgx`

```bash
./deploy.sh build-dgx
```

Rsyncs `./llama.cpp` to `DGX_USER@DGX_HOST:DGX_REMOTE_DIR` (skipping `.git` and build artifacts), then SSHes in and runs cmake with:

- `-DGGML_CUDA=ON`
- `-DCMAKE_CUDA_ARCHITECTURES=121`
- `-DGGML_CPU_AARCH64=ON`
- `-DGGML_RPC=ON`
- `-DLLAMA_FLASH_ATTENTION=ON`

### `build-mac`

```bash
./deploy.sh build-mac
```

Builds locally with Metal + RPC:

- `-DGGML_METAL=ON`
- `-DGGML_RPC=ON`
- `-DLLAMA_FLASH_ATTENTION=ON`

### `download`

```bash
./deploy.sh download
./deploy.sh download --repo "unsloth/MiniMax-M2.5-GGUF" --pattern "*UD-Q4_K_M*"
./deploy.sh download --dir /Volumes/external/models
```

Creates a Python venv (`.venv/`), installs `huggingface_hub`, and downloads matching GGUF files. Uses `HF_TOKEN` from `config.env` or your shell environment.

### `start-rpc` / `stop-rpc`

```bash
./deploy.sh start-rpc     # SSH to DGX, start rpc-server in background
./deploy.sh stop-rpc      # SSH to DGX, kill rpc-server
```

Verifies the RPC port is reachable from Mac after starting.

### `start-llama` / `stop-llama`

```bash
./deploy.sh start-llama
./deploy.sh start-llama --model-file "UD-Q4_K_M/model-00001.gguf" --alias "minimax-q4"
./deploy.sh start-llama --ctx 65536 --parallel 2
```

`--model-file` is relative to `MODELS_DIR` (from `config.env`) or absolute. Waits for the `/health` endpoint to respond before returning.

PID is stored in `.pids/llama-server.pid`. Logs go to `logs/llama-server.log`.

### `deploy`

```bash
# First-time full deploy
./deploy.sh deploy --tag b8223

# Rebuild and restart, keeping existing download
./deploy.sh deploy --tag b8223 --skip-download

# Just restart servers (already built and downloaded)
./deploy.sh deploy --skip-clone --skip-build --skip-download

# Different model
./deploy.sh deploy --skip-clone --skip-build \
    --model-file "UD-Q4_K_M/MiniMax-M2.5-UD-Q4_K_M-00001-of-00003.gguf" \
    --alias "minimax-q4"
```

### `status`

```bash
./deploy.sh status
```

Shows whether `llama-server` (local) and `rpc-server` (DGX) are running, with health check for llama-server.

### `logs`

```bash
./deploy.sh logs           # tail llama-server log
./deploy.sh logs llama     # same
./deploy.sh logs rpc       # tail rpc-server log on DGX via SSH
```

---

## Configuration Reference (`config.env`)

| Variable | Default | Description |
|---|---|---|
| `DGX_HOST` | `192.168.86.242` | DGX Spark IP |
| `DGX_USER` | `user` | SSH user on DGX |
| `DGX_REMOTE_DIR` | `/home/user/rz-rpc-llm/llama.cpp` | Where to rsync source on DGX |
| `DGX_RPC_PORT` | `50052` | rpc-server port |
| `LLAMA_CPP_DIR` | `./llama.cpp` | Local llama.cpp clone |
| `MODELS_DIR` | `./models` | Model download directory |
| `LLAMA_HOST` | `0.0.0.0` | llama-server bind address |
| `LLAMA_PORT` | `8680` | llama-server port |
| `HF_TOKEN` | *(required for gated models)* | Hugging Face access token |
| `DEFAULT_MODEL_REPO` | `unsloth/MiniMax-M2.5-GGUF` | HF repo for download |
| `DEFAULT_MODEL_PATTERN` | `*UD-Q6_K_XL*` | File glob for download |
| `DEFAULT_MODEL_FILE` | `UD-Q6_K_XL/MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf` | First GGUF passed to `--model` |
| `DEFAULT_MODEL_ALIAS` | `minimax-m2.5` | `--alias` for llama-server |
| `LLAMA_CTX_SIZE` | `131072` | Context size |
| `LLAMA_PARALLEL` | `4` | Parallel request slots |
| `LLAMA_TENSOR_SPLIT` | `2,3` | Layer split ratio (Mac:DGX) |
| `LLAMA_THREADS` | `14` | CPU threads |
| `LLAMA_THREADS_BATCH` | `20` | Batch CPU threads |

---

## Endpoints

Once running:

| Endpoint | Description |
|---|---|
| `http://localhost:8680/v1` | OpenAI-compatible API |
| `http://localhost:8680/health` | Health check |
| `http://localhost:8680/metrics` | Prometheus metrics |
| `http://localhost:8680/v1/models` | List loaded models |

---

## Tensor Split

`LLAMA_TENSOR_SPLIT="2,3"` means 2 parts to Mac (Metal), 3 parts to DGX (CUDA). Tune based on VRAM availability and model size. The DGX GB10 has ~96GB unified GPU memory; Mac M2 Ultra has 192GB shared.

---

## Troubleshooting

**SSH fails to DGX**
```bash
ssh-copy-id user@192.168.86.242
```

**rpc-server not reachable**
- Check DGX firewall: port 50052 must be open to Mac
- Check DGX log: `./deploy.sh logs rpc`

**llama-server exits immediately**
```bash
./deploy.sh logs llama
```
Common causes: model file path wrong, not enough RAM for `--mlock`, RPC server not running.

**HF download fails**
- Set `HF_TOKEN` in `config.env` or export it in your shell
- Accept model license on huggingface.co
