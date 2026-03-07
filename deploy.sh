#!/usr/bin/env bash
# ==============================================================================
# rz-rpc-llm — LLaMA.cpp RPC Deployment
#
# Splits inference across two devices:
#   - DGX Spark GB10 (CUDA)  → rpc-server
#   - Mac Studio M2 Ultra    → llama-server (local + RPC backend)
#
# Usage: ./deploy.sh <command> [options]
#   clone   [--tag TAG]              Clone llama.cpp (default: latest)
#   build-dgx                        Sync source to DGX and build
#   build-mac                        Build on local Mac
#   download [--repo R] [--pattern P] [--model-file F] [--alias A]
#   start-rpc                        Start rpc-server on DGX
#   stop-rpc                         Stop rpc-server on DGX
#   start-llama [--model-file F] [--alias A] [--ctx N] [--parallel N]
#   stop-llama                       Stop local llama-server
#   deploy  [--tag TAG] [--model-file F] [--alias A]
#                                    Full pipeline: clone→build→start
#   status                           Show running processes
#   logs    [rpc|llama]              Tail logs
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.env"
PID_DIR="${SCRIPT_DIR}/.pids"
LOG_DIR="${SCRIPT_DIR}/logs"

# ------------------------------------------------------------------------------
# Colours & logging
# ------------------------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()         { echo -e "${CYAN}[$(date '+%H:%M:%S')]${RESET} $*"; }
log_ok()      { echo -e "${GREEN}[$(date '+%H:%M:%S')] OK${RESET} $*"; }
log_warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN${RESET} $*"; }
log_error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR${RESET} $*" >&2; }
log_section() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════${RESET}"; \
                echo -e "${BOLD}${CYAN}  $*${RESET}"; \
                echo -e "${BOLD}${CYAN}══════════════════════════════════════${RESET}"; }
die()         { log_error "$*"; exit 1; }

# ------------------------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------------------------
load_config() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        die "config.env not found. Copy config.env.example → config.env and edit it."
    fi
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
    log "Loaded config: ${CONFIG_FILE}"

    # Expand relative paths relative to SCRIPT_DIR
    if [[ "${LLAMA_CPP_DIR}" != /* ]]; then
        LLAMA_CPP_DIR="${SCRIPT_DIR}/${LLAMA_CPP_DIR#./}"
    fi
    if [[ "${MODELS_DIR}" != /* ]]; then
        MODELS_DIR="${SCRIPT_DIR}/${MODELS_DIR#./}"
    fi

    # Allow HF_TOKEN override from environment
    HF_TOKEN="${HF_TOKEN:-}"
    if [[ -n "${HF_TOKEN_ENV:-}" ]]; then
        HF_TOKEN="${HF_TOKEN_ENV}"
    fi
}

# Resolve model file (relative to MODELS_DIR if not absolute)
resolve_model_file() {
    local mf="$1"
    if [[ "${mf}" == /* ]]; then
        echo "${mf}"
    else
        echo "${MODELS_DIR}/${mf}"
    fi
}

# ------------------------------------------------------------------------------
# SSH helpers
# ------------------------------------------------------------------------------
DGX_SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

ssh_dgx() {
    ssh "${DGX_SSH_OPTS[@]}" "${DGX_USER}@${DGX_HOST}" "$@"
}

# Test SSH connectivity to DGX
check_dgx_ssh() {
    log "Testing SSH connection to DGX (${DGX_USER}@${DGX_HOST})..."
    if ! ssh_dgx "echo OK" &>/dev/null; then
        die "Cannot SSH to DGX at ${DGX_USER}@${DGX_HOST}. Check host, user, and SSH keys."
    fi
    log_ok "DGX SSH connection OK"
}

# ------------------------------------------------------------------------------
# Command: clone
# ------------------------------------------------------------------------------
cmd_clone() {
    local tag=""
    local repo="https://github.com/ggml-org/llama.cpp.git"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tag|-t) tag="$2"; shift 2 ;;
            --repo)   repo="$2"; shift 2 ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    log_section "Clone llama.cpp"
    log "Repo : ${repo}"
    log "Tag  : ${tag:-latest (HEAD)}"
    log "Dest : ${LLAMA_CPP_DIR}"

    if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
        log_warn "llama.cpp already cloned at ${LLAMA_CPP_DIR}"
        log "Fetching latest..."
        git -C "${LLAMA_CPP_DIR}" fetch --tags --prune
    else
        log "Cloning..."
        git clone --depth 1 "${repo}" "${LLAMA_CPP_DIR}"
        # Re-fetch with full history only if a specific tag is requested
        if [[ -n "${tag}" ]]; then
            git -C "${LLAMA_CPP_DIR}" fetch --unshallow 2>/dev/null || true
            git -C "${LLAMA_CPP_DIR}" fetch --tags
        fi
    fi

    if [[ -n "${tag}" ]]; then
        log "Checking out tag/commit: ${tag}"
        git -C "${LLAMA_CPP_DIR}" checkout "${tag}"
        log_ok "Checked out: $(git -C "${LLAMA_CPP_DIR}" describe --always --tags)"
    else
        log_ok "At HEAD: $(git -C "${LLAMA_CPP_DIR}" log -1 --oneline)"
    fi
}

# ------------------------------------------------------------------------------
# Command: build-dgx
# ------------------------------------------------------------------------------
cmd_build_dgx() {
    log_section "Build llama.cpp on DGX Spark GB10"
    load_config
    check_dgx_ssh

    [[ -d "${LLAMA_CPP_DIR}/.git" ]] || die "llama.cpp not cloned. Run: ./deploy.sh clone"

    log "Syncing source to DGX:${DGX_REMOTE_DIR} ..."
    ssh_dgx "mkdir -p '${DGX_REMOTE_DIR}'"
    rsync -az --info=progress2 \
        --exclude='.git' \
        --exclude='build/' \
        --exclude='*.o' \
        "${LLAMA_CPP_DIR}/" \
        "${DGX_USER}@${DGX_HOST}:${DGX_REMOTE_DIR}/"
    log_ok "Sync complete"

    log "Building on DGX (CUDA, SM121)..."
    ssh_dgx bash -s <<'REMOTE_BUILD'
set -euo pipefail
LLAMA_DIR="$(echo "${DGX_REMOTE_DIR:-}")"
# DGX_REMOTE_DIR is inherited via env; fall back to positional if needed
cd "${DGX_REMOTE_DIR}"

echo "[DGX] cmake configure..."
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="121" \
    -DGGML_CPU_AARCH64=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_FLASH_ATTENTION=ON

echo "[DGX] cmake build ($(nproc) jobs)..."
cmake --build build --config Release -j$(nproc)

echo "[DGX] Build complete: $(ls build/bin/)"
REMOTE_BUILD
    log_ok "DGX build complete"
}

# Override DGX_REMOTE_DIR in SSH environment
cmd_build_dgx() {
    log_section "Build llama.cpp on DGX Spark GB10"
    load_config
    check_dgx_ssh

    [[ -d "${LLAMA_CPP_DIR}/.git" ]] || die "llama.cpp not cloned. Run: ./deploy.sh clone"

    log "Syncing source to ${DGX_USER}@${DGX_HOST}:${DGX_REMOTE_DIR} ..."
    ssh_dgx "mkdir -p '${DGX_REMOTE_DIR}'"
    rsync -az --info=progress2 \
        --exclude='.git' \
        --exclude='build/' \
        --exclude='*.o' \
        "${LLAMA_CPP_DIR}/" \
        "${DGX_USER}@${DGX_HOST}:${DGX_REMOTE_DIR}/"
    log_ok "Sync complete"

    log "Building on DGX (CUDA, SM121, $(ssh_dgx nproc) cores)..."
    ssh_dgx "
        set -euo pipefail
        cd '${DGX_REMOTE_DIR}'
        echo '[DGX] cmake configure...'
        cmake -B build \
            -DGGML_CUDA=ON \
            -DGGML_CUDA_FA_ALL_QUANTS=ON \
            -DCMAKE_CUDA_ARCHITECTURES='121' \
            -DGGML_CPU_AARCH64=ON \
            -DBUILD_SHARED_LIBS=OFF \
            -DGGML_RPC=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLAMA_FLASH_ATTENTION=ON
        echo '[DGX] cmake build...'
        cmake --build build --config Release -j\$(nproc)
        echo '[DGX] Binaries:' \$(ls build/bin/)
    "
    log_ok "DGX build complete"
}

# ------------------------------------------------------------------------------
# Command: build-mac
# ------------------------------------------------------------------------------
cmd_build_mac() {
    log_section "Build llama.cpp on Mac Studio (Metal)"
    load_config

    [[ -d "${LLAMA_CPP_DIR}/.git" ]] || die "llama.cpp not cloned. Run: ./deploy.sh clone"

    cd "${LLAMA_CPP_DIR}"
    log "cmake configure (Metal, RPC)..."
    cmake -B build \
        -DGGML_METAL=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_RPC=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_FLASH_ATTENTION=ON

    local jobs
    jobs=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)
    log "cmake build (${jobs} jobs)..."
    cmake --build build --config Release -j"${jobs}"

    log_ok "Mac build complete. Binaries: $(ls "${LLAMA_CPP_DIR}/build/bin/")"
}

# ------------------------------------------------------------------------------
# Command: download
# ------------------------------------------------------------------------------
cmd_download() {
    local repo=""
    local pattern=""
    local local_dir=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo)        repo="$2";       shift 2 ;;
            --pattern|-p)  pattern="$2";    shift 2 ;;
            --dir)         local_dir="$2";  shift 2 ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    load_config
    repo="${repo:-${DEFAULT_MODEL_REPO}}"
    pattern="${pattern:-${DEFAULT_MODEL_PATTERN}}"
    local_dir="${local_dir:-${MODELS_DIR}}"

    log_section "Download Model"
    log "HF repo  : ${repo}"
    log "Pattern  : ${pattern}"
    log "Local dir: ${local_dir}"

    local token="${HF_TOKEN:-}"
    [[ -z "${token}" ]] && log_warn "HF_TOKEN not set — download may fail for gated models"

    mkdir -p "${local_dir}"

    # Set up venv if needed
    local venv_dir="${SCRIPT_DIR}/.venv"
    if [[ ! -x "${venv_dir}/bin/python3" ]]; then
        log "Creating Python venv at ${venv_dir}..."
        python3 -m venv "${venv_dir}"
    fi

    log "Installing/upgrading huggingface_hub..."
    "${venv_dir}/bin/pip" install -q -U "huggingface_hub[cli]"

    log "Downloading (this may take a while)..."
    if [[ -n "${token}" ]]; then
        HF_TOKEN="${token}" "${venv_dir}/bin/huggingface-cli" download "${repo}" \
            --local-dir "${local_dir}" \
            --include "${pattern}"
    else
        "${venv_dir}/bin/huggingface-cli" download "${repo}" \
            --local-dir "${local_dir}" \
            --include "${pattern}"
    fi

    log_ok "Download complete → ${local_dir}"
    log "Files:"
    find "${local_dir}" -name "*.gguf" | sort | sed 's/^/  /'
}

# ------------------------------------------------------------------------------
# Command: start-rpc
# ------------------------------------------------------------------------------
cmd_start_rpc() {
    log_section "Start RPC Server on DGX Spark GB10"
    load_config
    check_dgx_ssh

    local rpc_bin="${DGX_REMOTE_DIR}/build/bin/rpc-server"
    local log_file="/tmp/rpc-server.log"

    log "Checking for existing rpc-server on DGX..."
    if ssh_dgx "pgrep -x rpc-server" &>/dev/null; then
        log_warn "rpc-server already running on DGX ($(ssh_dgx "pgrep -x rpc-server"))"
        log_warn "Run './deploy.sh stop-rpc' first if you want to restart."
        return 0
    fi

    log "Starting rpc-server on DGX (${DGX_HOST}:${DGX_RPC_PORT})..."
    ssh_dgx "
        set -euo pipefail
        if [[ ! -x '${rpc_bin}' ]]; then
            echo 'ERROR: rpc-server binary not found at ${rpc_bin}' >&2
            echo 'Run: ./deploy.sh build-dgx' >&2
            exit 1
        fi
        nohup '${rpc_bin}' -H '${DGX_RPC_BIND}' -p '${DGX_RPC_PORT}' \
            > '${log_file}' 2>&1 &
        RPC_PID=\$!
        sleep 1
        if kill -0 \$RPC_PID 2>/dev/null; then
            echo \$RPC_PID
        else
            echo 'ERROR: rpc-server failed to start. Check ${log_file}' >&2
            cat '${log_file}' >&2
            exit 1
        fi
    "
    local dgx_pid
    dgx_pid="$(ssh_dgx "pgrep -x rpc-server | head -1")"
    log_ok "rpc-server started on DGX (PID ${dgx_pid}, port ${DGX_RPC_PORT})"
    log "Log: ${DGX_HOST}:${log_file}"

    # Verify port is reachable from Mac
    log "Verifying RPC port reachability from Mac..."
    local retries=5
    while (( retries-- > 0 )); do
        if nc -z -w3 "${DGX_HOST}" "${DGX_RPC_PORT}" 2>/dev/null; then
            log_ok "RPC endpoint ${DGX_HOST}:${DGX_RPC_PORT} is reachable"
            return 0
        fi
        log "Waiting for port... (${retries} retries left)"
        sleep 2
    done
    log_warn "Could not verify RPC port ${DGX_HOST}:${DGX_RPC_PORT} — check firewall/network"
}

# ------------------------------------------------------------------------------
# Command: stop-rpc
# ------------------------------------------------------------------------------
cmd_stop_rpc() {
    log_section "Stop RPC Server on DGX"
    load_config
    check_dgx_ssh

    if ssh_dgx "pgrep -x rpc-server" &>/dev/null; then
        ssh_dgx "pkill -x rpc-server && echo 'rpc-server stopped'"
        log_ok "rpc-server stopped on DGX"
    else
        log_warn "No rpc-server process found on DGX"
    fi
}

# ------------------------------------------------------------------------------
# Command: start-llama
# ------------------------------------------------------------------------------
cmd_start_llama() {
    local model_file=""
    local model_alias=""
    local ctx_size=""
    local parallel=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-file|-m) model_file="$2";  shift 2 ;;
            --alias|-a)      model_alias="$2"; shift 2 ;;
            --ctx|-c)        ctx_size="$2";    shift 2 ;;
            --parallel|-p)   parallel="$2";    shift 2 ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    load_config
    model_file="${model_file:-${DEFAULT_MODEL_FILE}}"
    model_alias="${model_alias:-${DEFAULT_MODEL_ALIAS}}"
    ctx_size="${ctx_size:-${LLAMA_CTX_SIZE}}"
    parallel="${parallel:-${LLAMA_PARALLEL}}"

    local resolved_model
    resolved_model="$(resolve_model_file "${model_file}")"

    log_section "Start llama-server on Mac Studio"
    log "Model file : ${resolved_model}"
    log "Model alias: ${model_alias}"
    log "Context    : ${ctx_size}"
    log "Parallel   : ${parallel}"
    log "RPC backend: ${DGX_HOST}:${DGX_RPC_PORT}"
    log "Listening  : ${LLAMA_HOST}:${LLAMA_PORT}"

    local llama_bin="${LLAMA_CPP_DIR}/build/bin/llama-server"
    [[ -x "${llama_bin}" ]] || die "llama-server not found at ${llama_bin}. Run: ./deploy.sh build-mac"
    [[ -f "${resolved_model}" ]] || die "Model file not found: ${resolved_model}. Run: ./deploy.sh download"

    mkdir -p "${PID_DIR}" "${LOG_DIR}"
    local pid_file="${PID_DIR}/llama-server.pid"
    local log_file="${LOG_DIR}/llama-server.log"

    if [[ -f "${pid_file}" ]]; then
        local old_pid
        old_pid="$(cat "${pid_file}")"
        if kill -0 "${old_pid}" 2>/dev/null; then
            log_warn "llama-server already running (PID ${old_pid})"
            log_warn "Run './deploy.sh stop-llama' first to restart."
            return 0
        else
            log_warn "Stale PID file, cleaning up..."
            rm -f "${pid_file}"
        fi
    fi

    log "Launching llama-server..."
    nohup "${llama_bin}" \
        --model            "${resolved_model}" \
        --alias            "${model_alias}" \
        --jinja \
        --reasoning-format auto \
        --temp             "${LLAMA_TEMP}" \
        --top-p            "${LLAMA_TOP_P}" \
        --top-k            "${LLAMA_TOP_K}" \
        --min-p            "${LLAMA_MIN_P}" \
        --repeat-penalty   "${LLAMA_REPEAT_PENALTY}" \
        --ctx-size         "${ctx_size}" \
        --host             "${LLAMA_HOST}" \
        --port             "${LLAMA_PORT}" \
        --prio             "${LLAMA_PRIO}" \
        --parallel         "${parallel}" \
        --rpc              "${DGX_HOST}:${DGX_RPC_PORT}" \
        --split-mode       "${LLAMA_SPLIT_MODE}" \
        --tensor-split     "${LLAMA_TENSOR_SPLIT}" \
        --threads          "${LLAMA_THREADS}" \
        --threads-batch    "${LLAMA_THREADS_BATCH}" \
        --batch-size       "${LLAMA_BATCH_SIZE}" \
        --ubatch-size      "${LLAMA_UBATCH_SIZE}" \
        --n-gpu-layers     "${LLAMA_N_GPU_LAYERS}" \
        --no-mmap \
        --mlock \
        --kv-offload \
        --kv-unified \
        --flash-attn       on \
        --cont-batching \
        --no-context-shift \
        --cache-type-k     "${LLAMA_CACHE_TYPE_K}" \
        --cache-type-v     "${LLAMA_CACHE_TYPE_V}" \
        --cache-prompt \
        --cache-reuse      "${LLAMA_CACHE_REUSE}" \
        --metrics \
        > "${log_file}" 2>&1 &
    local llama_pid=$!
    echo "${llama_pid}" > "${pid_file}"

    log "PID ${llama_pid} — waiting for server to become ready..."
    local retries=30
    while (( retries-- > 0 )); do
        if curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" &>/dev/null; then
            log_ok "llama-server ready at http://${LLAMA_HOST}:${LLAMA_PORT}"
            log_ok "OpenAI-compatible endpoint: http://127.0.0.1:${LLAMA_PORT}/v1"
            log_ok "Metrics: http://127.0.0.1:${LLAMA_PORT}/metrics"
            log "Log: ${log_file}"
            return 0
        fi
        if ! kill -0 "${llama_pid}" 2>/dev/null; then
            log_error "llama-server exited unexpectedly. Last log lines:"
            tail -30 "${log_file}" >&2
            die "llama-server failed to start"
        fi
        printf '.'
        sleep 2
    done
    echo
    log_warn "Server did not respond within timeout — check logs: ${log_file}"
    log_warn "It may still be loading the model. Try: ./deploy.sh status"
}

# ------------------------------------------------------------------------------
# Command: stop-llama
# ------------------------------------------------------------------------------
cmd_stop_llama() {
    log_section "Stop llama-server"
    load_config
    local pid_file="${PID_DIR}/llama-server.pid"

    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
            rm -f "${pid_file}"
            log_ok "llama-server (PID ${pid}) stopped"
        else
            log_warn "PID ${pid} not running, cleaning up stale PID file"
            rm -f "${pid_file}"
        fi
    else
        # Fallback: kill by name
        if pgrep -x llama-server &>/dev/null; then
            pkill -x llama-server && log_ok "llama-server stopped (by name)"
        else
            log_warn "No llama-server process found"
        fi
    fi
}

# ------------------------------------------------------------------------------
# Command: deploy (full pipeline)
# ------------------------------------------------------------------------------
cmd_deploy() {
    local tag=""
    local model_file=""
    local model_alias=""
    local skip_clone=0
    local skip_build=0
    local skip_download=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tag|-t)        tag="$2";          shift 2 ;;
            --model-file|-m) model_file="$2";   shift 2 ;;
            --alias|-a)      model_alias="$2";  shift 2 ;;
            --skip-clone)    skip_clone=1;      shift ;;
            --skip-build)    skip_build=1;      shift ;;
            --skip-download) skip_download=1;   shift ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    load_config

    log_section "Full Deploy Pipeline"
    log "Tag          : ${tag:-latest}"
    log "Model file   : ${model_file:-${DEFAULT_MODEL_FILE}}"
    log "Skip clone   : ${skip_clone}"
    log "Skip build   : ${skip_build}"
    log "Skip download: ${skip_download}"

    if (( ! skip_clone )); then
        local clone_args=()
        [[ -n "${tag}" ]] && clone_args+=(--tag "${tag}")
        cmd_clone "${clone_args[@]}"
    fi

    if (( ! skip_build )); then
        cmd_build_dgx
        cmd_build_mac
    fi

    if (( ! skip_download )); then
        cmd_download
    fi

    # Start RPC first, then llama-server
    cmd_start_rpc

    local llama_args=()
    [[ -n "${model_file}" ]]  && llama_args+=(--model-file "${model_file}")
    [[ -n "${model_alias}" ]] && llama_args+=(--alias "${model_alias}")
    cmd_start_llama "${llama_args[@]}"

    log_section "Deploy Complete"
    log_ok "RPC server : ${DGX_HOST}:${DGX_RPC_PORT}"
    log_ok "LLM server : http://${LLAMA_HOST}:${LLAMA_PORT}"
    log_ok "OpenAI API : http://127.0.0.1:${LLAMA_PORT}/v1"
}

# ------------------------------------------------------------------------------
# Command: status
# ------------------------------------------------------------------------------
cmd_status() {
    load_config
    log_section "Status"

    # llama-server
    local pid_file="${PID_DIR}/llama-server.pid"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            log_ok "llama-server  RUNNING  (PID ${pid}, port ${LLAMA_PORT})"
            if curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" &>/dev/null; then
                log_ok "              health check: OK"
            else
                log_warn "              health check: not responding yet"
            fi
        else
            log_warn "llama-server  STOPPED  (stale PID ${pid})"
        fi
    else
        if pgrep -x llama-server &>/dev/null; then
            log_ok "llama-server  RUNNING  (PID $(pgrep -x llama-server | head -1), unmanaged)"
        else
            log_warn "llama-server  STOPPED"
        fi
    fi

    # rpc-server
    if check_dgx_ssh &>/dev/null; then
        if ssh_dgx "pgrep -x rpc-server" &>/dev/null; then
            local rpid
            rpid="$(ssh_dgx "pgrep -x rpc-server | head -1")"
            log_ok "rpc-server    RUNNING  (DGX PID ${rpid}, ${DGX_HOST}:${DGX_RPC_PORT})"
        else
            log_warn "rpc-server    STOPPED  (DGX ${DGX_HOST})"
        fi
    else
        log_warn "rpc-server    UNKNOWN  (DGX not reachable)"
    fi
}

# ------------------------------------------------------------------------------
# Command: logs
# ------------------------------------------------------------------------------
cmd_logs() {
    load_config
    local target="${1:-llama}"

    case "${target}" in
        llama)
            local log_file="${LOG_DIR}/llama-server.log"
            [[ -f "${log_file}" ]] || die "No llama-server log at ${log_file}"
            log "Tailing ${log_file} (Ctrl+C to stop)"
            tail -f "${log_file}"
            ;;
        rpc)
            log "Tailing rpc-server log on DGX (Ctrl+C to stop)"
            ssh_dgx "tail -f /tmp/rpc-server.log"
            ;;
        *) die "Unknown log target '${target}'. Use: rpc | llama" ;;
    esac
}

# ------------------------------------------------------------------------------
# Usage
# ------------------------------------------------------------------------------
usage() {
    cat <<EOF
${BOLD}rz-rpc-llm${RESET} — LLaMA.cpp RPC deployment across DGX Spark + Mac Studio

${BOLD}USAGE${RESET}
  ./deploy.sh <command> [options]

${BOLD}COMMANDS${RESET}
  ${CYAN}clone${RESET}    [--tag TAG]
      Clone llama.cpp (optionally at a specific tag/commit, e.g. b8223)

  ${CYAN}build-dgx${RESET}
      Rsync source to DGX and build with CUDA (SM121)

  ${CYAN}build-mac${RESET}
      Build locally on Mac with Metal

  ${CYAN}download${RESET} [--repo REPO] [--pattern GLOB] [--dir DIR]
      Download model from HuggingFace
      Defaults from config.env: ${BOLD}DEFAULT_MODEL_REPO, DEFAULT_MODEL_PATTERN${RESET}

  ${CYAN}start-rpc${RESET}
      Start rpc-server on DGX (must come before start-llama)

  ${CYAN}stop-rpc${RESET}
      Stop rpc-server on DGX

  ${CYAN}start-llama${RESET} [--model-file PATH] [--alias NAME] [--ctx N] [--parallel N]
      Start llama-server locally (uses RPC backend on DGX)
      PATH is relative to MODELS_DIR or absolute

  ${CYAN}stop-llama${RESET}
      Stop local llama-server

  ${CYAN}deploy${RESET}   [--tag TAG] [--model-file PATH] [--alias NAME]
                 [--skip-clone] [--skip-build] [--skip-download]
      Full pipeline: clone → build-dgx → build-mac → download → start-rpc → start-llama

  ${CYAN}status${RESET}
      Show running process status for both devices

  ${CYAN}logs${RESET}     [llama|rpc]
      Tail logs (default: llama)

${BOLD}EXAMPLES${RESET}
  # First-time full deploy at tag b8223
  ./deploy.sh deploy --tag b8223

  # Already built; just restart servers with a different model
  ./deploy.sh start-rpc
  ./deploy.sh start-llama --model-file "UD-Q4_K_M/model-00001.gguf" --alias "minimax-q4"

  # Download a different quantization
  ./deploy.sh download --pattern "*UD-Q4_K_M*"

  # Check health
  ./deploy.sh status
  curl http://localhost:${LLAMA_PORT:-8680}/health

EOF
}

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
main() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 0
    fi

    local cmd="$1"; shift

    case "${cmd}" in
        clone)       load_config; cmd_clone "$@" ;;
        build-dgx)   cmd_build_dgx "$@" ;;
        build-mac)   cmd_build_mac "$@" ;;
        download)    cmd_download "$@" ;;
        start-rpc)   cmd_start_rpc "$@" ;;
        stop-rpc)    cmd_stop_rpc "$@" ;;
        start-llama) cmd_start_llama "$@" ;;
        stop-llama)  cmd_stop_llama "$@" ;;
        deploy)      cmd_deploy "$@" ;;
        status)      cmd_status "$@" ;;
        logs)        cmd_logs "$@" ;;
        help|--help|-h) usage ;;
        *) log_error "Unknown command: ${cmd}"; usage; exit 1 ;;
    esac
}

main "$@"
